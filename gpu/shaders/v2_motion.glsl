// Motion version 2, the synthesis of V2Mode::Performance (the "grid morph"): the push-constant block, the cleaned block field and the
// sampling helpers of v2_clean.comp and v2_morph.comp. Included after common.glsl (the bindings every pass shares; the CrosshairV2
// tap rejection). V2's estimate (v2_est.glsl) still pushes V1's PC block; these passes push this one over the same 128-byte range of
// the same pipeline layout (framegen.cpp V2PC).

layout(push_constant) uniform V2PC {
    int W, H;          // full resolution
    int fw, fh;        // flow resolution
    int bw, bh;        // finest block grid (8x8 flow pixels per block: 16x16 px at flowScale 2)
    int fscale;        // flow scale (1, 2, 4)
    int fIn;           // clean: the block field it reads (offset in field[]: after sub-pixel refinement, else the raw search)
    int n;             // morph: samples of this output (1 = recordSynth, 2..16 = recordSynthBlur)
    int gauss;         // morph: sample weights 0 = even, 1 = Gaussian (blur_resolve.comp's)
    int fix;           // morph: 0 = main pass (counts the untrusted tiles), 1 = fix-up pass (redoes the output if a sample flicked)
    int cnt;           // morph: u[cnt + k] = untrusted-pixel count of sample k (recordSynth: 0 = U_UNTRUSTED; recordSynthBlur: 32)
    float cutSad;      // Settings::sceneCutSad
    float globalFrac;  // Settings::globalFallback
    float sigma2Inv;   // clean: 1 / sigma^2 of the bilateral smoothing's range weight (sigma in flow pixels of half-vector)
    float trustMid;    // morph: an 8x8 tile is untrusted if its mean mismatch exceeds this (V1's trust ramp midpoint)
    float t[16];       // morph: time of each sample in [0, 1]; <= 0 / >= 1 = the real frame prev / curr
} pc;

// The cleaned block field (v2_clean.comp writes it once per estimate, v2_morph.comp reads it): bw x bh half-vectors in flow pixels
// on the midpoint grid, like V1's block field. Binding 13: V2's own buffer (MotionGroup::bufs), bound only to V2's sets.
layout(set = 0, binding = 13, std430) buffer V2Field { vec2 vf[]; };

const int V2_BLOCK = 8;
const int U_SADSUM_ = 1;   // == motion_v1.glsl U_SADSUM (the estimate's scene-cut sum)

vec2 VF(ivec2 b) {
    b = clamp(b, ivec2(0), ivec2(pc.bw - 1, pc.bh - 1));
    return vf[b.y * pc.bw + b.x];
}

// The half-vector in full-resolution pixels at the full-resolution position x (pixel-centre coordinates): the block field upsampled
// bilinearly straight to full resolution (pixfield.comp's candidate 0, the "bilinear block blend", without the flow-res step) x
// fscale. Smooth everywhere: neighbouring blocks' motions blend over one block, so content stretches between them instead of tearing.
vec2 hvAt(vec2 x) {
    float S = float(pc.fscale);
    vec2 s = (x + 0.5) * (1.0 / (float(V2_BLOCK) * S)) - 0.5;
    vec2 fl = floor(s);
    vec2 f = s - fl;
    ivec2 i = ivec2(fl);
    return S * mix(mix(VF(i), VF(i + ivec2(1, 0)), f.x), mix(VF(i + ivec2(0, 1)), VF(i + ivec2(1, 1)), f.x), f.y);
}

// A bilinear tap of an Rgba8 frame at p (pixel-centre coordinates), the position rounded to 1/32 px like every warp tap of the library
// (ui_reject.glsl's footprint assumes it).
vec4 tapRaw(sampler2D t, vec2 p) {
    p = roundEven(p * 32.0) * (1.0 / 32.0);
    return textureLod(t, (p + 0.5) / vec2(pc.W, pc.H), 0.0);
}
