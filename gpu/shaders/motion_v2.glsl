// Motion version 2 (Settings::motionVersion = V2): the push constants and helpers of V2's own synth / estimate passes (v2_occl.comp,
// v2_resolve.comp). Included after common.glsl. Rgba8 only (settingsError() refuses V2 for every other Format), so there is no YUV or
// other-format path here. V2 shares V1's estimate passes and its error / lookup / fallback passes (they push V1's block,
// motion_v1.glsl); its own passes push this block over the same 128-byte range of the same pipeline layout.
//
// V2's own buffer (binding 13, framegen.cpp MotionGroup::bufs[0]):
//   words [0, 4 fw fh): the occlusion map, one uvec4 per flow-resolution pixel of the MIDPOINT grid (v2_occl.comp):
//     x = packHalf2x16(background half-vector), y = packHalf2x16(foreground half-vector) (flow-resolution units, like field[]),
//     z = side: 0 = no clean occlusion, 1 = covered (take prev), 2 = uncovered (take curr), w = 0 (reserved).
//   words [clsOff, clsOff + eStride H): the class map of the latest synth (debug, recordDebugMaps), one byte per pixel, 4 per
//     word, low byte first: bits 0-2 class (V2_CLS_*), bit 3 = side curr, bits 4-7 = weight of the hold / one-sided output x 15.

layout(set = 0, binding = 13, std430) buffer V2Buf { uint v2b[]; };

layout(push_constant) uniform PC2 {
    int W, H;          // full resolution
    int fw, fh;        // flow resolution (= the finest pyramid level)
    int fscale;        // flow scale (1, 2, 4)
    int pf;            // offset of the per-pixel field the pass reads (the midpoint field, or the field at t)
    int offA, offC;    // v2_occl: luma offsets of the finest level
    int reach;         // v2_occl: side-vector distance (flow px)
    int clsOff;        // word offset of the class map in v2b[]
    float s;           // 2t - 1
    float wA, wC;      // 1 - t, t
    float trust0, trustScale;   // the held class's ramp (V2Knobs trustLow, 1 / (trustHigh - trustLow))
    float occ0, occScale;       // the occlusion ramp (V2Knobs occLow, 1 / (occHigh - occLow))
    float pickBias;             // V2Knobs pickBias
    float edgeMin;              // v2_occl: min normal motion difference, flow-resolution half-vector units (occlWidth / fscale)
    float confMax;              // v2_occl: V2Knobs confidence
} pc;

const int E_OFF = 64;          // packed mismatch in u[] (== motion_v1.glsl)
#define E_STRIDE ((pc.W + 3) >> 2)
const int U_UNTRUSTED = 0;
#define NEAR_C (pc.wC > 0.5)
#define MID (pc.wC == 0.5)
const uint V2_CLS_TRUSTED = 0u, V2_CLS_HELD = 1u, V2_CLS_OCC_FIELD = 2u, V2_CLS_OCC_BG = 3u, V2_CLS_OCC_FG = 4u;

// luma texel of the finest level, clamp to edge; bilinear tap (as motion_v1.glsl's L / sampleL)
float L(int off, int x, int y, int w, int h) {
    return luma[off + clamp(y, 0, h - 1) * w + clamp(x, 0, w - 1)];
}
float sampleL(int off, vec2 p, int w, int h) {
    ivec2 X = ivec2(roundEven(p * 32.0));
    ivec2 i = X >> 5;
    vec2 f = vec2(X & 31) * (1.0 / 32.0);
    float p00 = L(off, i.x, i.y, w, h), p01 = L(off, i.x + 1, i.y, w, h);
    float p10 = L(off, i.x, i.y + 1, w, h), p11 = L(off, i.x + 1, i.y + 1, w, h);
    return p00 * ((1.0 - f.y) * (1.0 - f.x)) + p01 * ((1.0 - f.y) * f.x)
         + p10 * (f.y * (1.0 - f.x)) + p11 * (f.y * f.x);
}

// == motion_v1.glsl halfVec (the field at pc.pf)
vec2 halfVec(ivec2 p) {
    int S = pc.fscale;
    p = min(p, ivec2(S * pc.fw - 1, S * pc.fh - 1));
    vec2 s = (vec2(p) + 0.5) * (1.0 / float(S)) - 0.5;
    vec2 fl = floor(s);
    vec2 f = s - fl;
    ivec2 i0 = clamp(ivec2(fl), ivec2(0), ivec2(pc.fw - 1, pc.fh - 1));
    ivec2 i1 = clamp(ivec2(fl) + 1, ivec2(0), ivec2(pc.fw - 1, pc.fh - 1));
    vec2 v00 = field[pc.pf + i0.y * pc.fw + i0.x], v01 = field[pc.pf + i0.y * pc.fw + i1.x];
    vec2 v10 = field[pc.pf + i1.y * pc.fw + i0.x], v11 = field[pc.pf + i1.y * pc.fw + i1.x];
    return float(S) * mix(mix(v00, v01, f.x), mix(v10, v11, f.x), f.y);
}

vec3 sampleRGBRaw(sampler2D t, vec2 p) {
    p = roundEven(p * 32.0) * (1.0 / 32.0);
    return textureLod(t, (p + 0.5) / vec2(pc.W, pc.H), 0.0).rgb;
}

// motion_v1.glsl mismatch() with a given full-resolution half-vector hv (the warp at time t: a(p - 2t hv), c(p + 2(1 - t) hv))
uint mismatchV(ivec2 p, vec2 hv, out vec3 wa, out vec3 wc) {
    precise float ka = 2.0 * pc.wC, kc = 2.0 * pc.wA;
    precise vec2 pa = vec2(p) - ka * hv;
    precise vec2 pc_ = vec2(p) + kc * hv;
    vec3 ra = sampleRGBRaw(texPrev, pa), rc = sampleRGBRaw(texCurr, pc_);
    uiReject(pa, pc_, ra, rc);
    wa = FG_FETCH(ra);
    wc = FG_FETCH(rc);
    vec3 d = FG_DIFF(wa, wc);
    return FG_PACKE(max(d.r, max(d.g, d.b)));
}

// The occlusion map entry of the midpoint position of full-resolution pixel p at time t: the content at p sits at the midpoint
// m = p - (2t - 1) hv (hv = the field at t at p; lookup.comp's relation), nearest flow pixel of m.
uvec4 occAt(ivec2 p, vec2 hv, float s) {
    vec2 m = vec2(p) - s * hv;
    ivec2 j = clamp(ivec2(floor((m + 0.5) / float(pc.fscale))), ivec2(0), ivec2(pc.fw - 1, pc.fh - 1));
    int i = 4 * (j.y * pc.fw + j.x);
    return uvec4(v2b[i], v2b[i + 1], v2b[i + 2], v2b[i + 3]);
}

uint eAtU(ivec2 p) {
    p = clamp(p, ivec2(0), ivec2(pc.W - 1, pc.H - 1));
    return (u[E_OFF + p.y * E_STRIDE + (p.x >> 2)] >> (8 * (p.x & 3))) & 255u;
}
