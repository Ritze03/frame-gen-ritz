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
#ifndef V2_RO
#define V2_RO
#endif
layout(set = 0, binding = 13, std430) V2_RO buffer V2Field { vec2 vf[]; };

// Binding 14: V2's second buffer (MotionGroup::bufs[V2B_QUAL]), uint words, N = fw * fh, N8 = s8W() * s8H() (the 8x8 cells of the
// flow grid, the block grid): [0, 4N) Quality's claims and coverage (v2_quality.glsl); both modes' static-structure mask S
// [4N, 5N) (v2_static.comp), its cell maxima [5N, 5N + N8), their enclosure E [5N + N8, 5N + 2 N8) (float bits) and the cells'
// "overlay near" bits C [5N + 2 N8, 5N + 3 N8) (v2_enclose.comp).
layout(set = 0, binding = 14, std430) V2_RO buffer V2Q { uint qb[]; };

const int V2_BLOCK = 8;

int sOff() { return 4 * pc.fw * pc.fh; }
int s8W() { return (pc.fw + 7) >> 3; }
int s8H() { return (pc.fh + 7) >> 3; }
int s8Off() { return 5 * pc.fw * pc.fh; }
int eOff() { return s8Off() + s8W() * s8H(); }
int cOff() { return eOff() + s8W() * s8H(); }
float sF(ivec2 m) {
    m = clamp(m, ivec2(0), ivec2(pc.fw - 1, pc.fh - 1));
    return uintBitsToFloat(qb[sOff() + m.y * pc.fw + m.x]);
}
float eF(ivec2 c) {
    c = clamp(c, ivec2(0), ivec2(s8W() - 1, s8H() - 1));
    return uintBitsToFloat(qb[eOff() + c.y * s8W() + c.x]);
}
// The cell of full-resolution position x, and whether its C says an overlay is near (most of a frame: no; one load, nothing else
// is read)
ivec2 sCell(vec2 x) { return clamp(ivec2(floor((x + 0.5) * (1.0 / float(V2_BLOCK * pc.fscale)))), ivec2(0), ivec2(s8W() - 1, s8H() - 1)); }
uint sCellBits(ivec2 c) { return qb[cOff() + c.y * s8W() + c.x]; }
bool sNear(ivec2 c) { return (sCellBits(c) & 1u) != 0u; }
ivec2 sTexel(vec2 x) { return ivec2(floor((x + 0.5) * (1.0 / float(pc.fscale)))); }
// The static-structure mask S (nearest flow pixel), its enclosure E (bilinear on the cell grid) and S dilated by ~1 flow px (the
// largest of the 4 S texels around x, as sOverAt) at full-resolution position x (pixel-centre coordinates)
vec3 sAt(vec2 x, uint bits) {   // bits: sCellBits of x's cell
    if ((bits & 1u) == 0u) return vec3(0.0);
    vec2 s = (x + 0.5) * (1.0 / float(V2_BLOCK * pc.fscale)) - 0.5;
    vec2 fl = floor(s);
    vec2 f = s - fl;
    ivec2 i = ivec2(fl);
    float E = mix(mix(eF(i), eF(i + ivec2(1, 0)), f.x), mix(eF(i + ivec2(0, 1)), eF(i + ivec2(1, 1)), f.x), f.y);
    ivec2 m = ivec2(floor((x + 0.5) * (1.0 / float(pc.fscale)) - 0.5)), n = sTexel(x);   // (n is one of the 4 around x)
    float s00 = sF(m), s10 = sF(m + ivec2(1, 0)), s01 = sF(m + ivec2(0, 1)), s11 = sF(m + ivec2(1, 1));
    float sn = n.y == m.y ? (n.x == m.x ? s00 : s10) : (n.x == m.x ? s01 : s11);
    return vec3(sn, E, max(max(s00, s10), max(s01, s11)));
}
// The overlay at full-resolution position x for the tap rejection: the largest of the 4 S texels around x (S dilated by about one
// flow pixel: the anti-aliased rim of a stroke, a gap in its mask), or E of its cell
vec2 sOverAt(vec2 x, ivec2 c) {   // c = sCell(x)
    if (!sNear(c)) return vec2(0.0);
    ivec2 i = ivec2(floor((x + 0.5) * (1.0 / float(pc.fscale)) - 0.5));   // the 4 S texels around x (S dilated by ~1 flow px)
    return vec2(max(max(sF(i), sF(i + ivec2(1, 0))), max(sF(i + ivec2(0, 1)), sF(i + ivec2(1, 1)))), eF(c));
}
const int U_V2SAD_ = 3;   // == v2_est.glsl U_V2SAD: Performance's scene-cut sum (integer finest SADs; V1 / Quality: u[1])

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

// Formats (shaders/format.glsl; the synthesis, v2_morph.glsl / v2_quality.glsl, is built once per format, the other V2 passes read
// only the luma, which luma.comp already gives in levels of a perceptually encoded 8-bit picture for every format). The synthesis
// blends "working values" in the source's own space: V2_LEVELS of a raw sample (Rgba8 / Rgb10: 8-bit levels, RgbaF16: the value
// itself, encoded or linear light, YUV: (Y, Cb, Cr) levels). Every threshold of it is in 8-bit levels, so every comparison is made
// on V2_LV of working values: the levels of a perceptually encoded 8-bit picture (RgbaF16 linear: the PQ code of each channel x 255,
// encoded: x 255; YUV: the R'G'B' the (Y, Cb, Cr) stand for, full range; V1's mismatch, FG_DIFF, measures the same). V2_D: the
// per-channel distance of two working values, V2_G: their difference (a gradient). V2_TAP: a bilinear tap (vec4; YUV: Y at p, CbCr
// at p + 0.5, see tapYuv), V2_PX: a texel (YUV: Y at p, CbCr of p's 2x2 block), V2_UIP: the position whose CrosshairV2 footprint
// (ui_reject.glsl) covers both. V2_QUANT: a sub-frame exactly as recordSynth stores it, in working values (the blur averages those).
// The Rgba8 spelling of every macro is the historical code (its SPIR-V is unchanged).
#if defined(FG_YUV)
// YUV: the Y tap at p and the CbCr tap at p + 0.5 (the centre of p's 2x2 block: for an even-even p, the block whose chroma texel this
// pixel stores, at the zero vector that texel itself), both bilinear; alpha 1. The chroma plane is ceil(W / 2) x ceil(H / 2).
vec4 tapYuv(sampler2D ty, sampler2D tc, vec2 p) {
    p = roundEven(p * 32.0) * (1.0 / 32.0);
    vec2 cs = 2.0 * vec2((pc.W + 1) >> 1, (pc.H + 1) >> 1);
    return vec4(textureLod(ty, (p + 0.5) / vec2(pc.W, pc.H), 0.0).r, textureLod(tc, (p + 1.0) / cs, 0.0).rg, 1.0);
}
vec3 v2QL(vec3 l) { return clamp(floor(l / YU_QL + 0.5), 0.0, YU_CODES) * YU_QL; }   // levels -> the levels of the codes they store as
#define V2_TAP(t, tc, p) tapYuv(t, tc, p)
#define V2_PX(t, tc, p) vec4(texelFetch(t, p, 0).r, texelFetch(tc, (p) >> 1, 0).rg, 1.0)
#define V2_LEVELS(x) ((x) * YU_SCALE)
#define V2_LV(x) fgYuvDiff(x)
#define V2_D(a, b) abs(fgYuvDiff((a) - (b)))
#define V2_G(a, b) fgYuvDiff((a) - (b))
#define V2_UIP(p) ((p) + 0.5)   // the chroma tap's 2x2 texels: 4x4 pixels that contain the Y tap's 2x2 too
#define V2_QUANT(x) v2QL(x)
#else
#define V2_TAP(t, tc, p) tapRaw(t, p)
#define V2_PX(t, tc, p) texelFetch(t, p, 0)
#define V2_LEVELS(x) FG_FETCH(x)
#define V2_UIP(p) (p)
#if defined(FG_FORMAT_F16)
#define V2_LV(x) fgMetric(x)
#define V2_D(a, b) abs(fgMetric(a) - fgMetric(b))
#define V2_G(a, b) (fgMetric(a) - fgMetric(b))
// the nearest half float, round to nearest even, as a float (the store's conversion and packHalf2x16 may truncate instead: AMD's do;
// v2_morph.glsl rounds its output with this before the store, so the store is exact)
vec3 v2Half(vec3 x) {
    x = clamp(x, -65504.0, 65504.0);
    uvec3 b = floatBitsToUint(x);
    vec3 n = uintBitsToFloat((b + 0xFFFu + ((b >> 13) & 1u)) & ~0x1FFFu);   // 10 mantissa bits
    return mix(n, roundEven(x * 16777216.0) * (1.0 / 16777216.0), lessThan(abs(x), vec3(6.103515625e-05)));   // subnormals: 2^-24 steps
}
#define V2_QUANT(x) v2Half(x)   // the R16G16B16A16_SFLOAT store (no clamp of the range)
#elif defined(FG_FORMAT_RGB10)
#define V2_LV(x) (x)
#define V2_D(a, b) abs((a) - (b))
#define V2_G(a, b) ((a) - (b))
#define V2_QUANT(x) (floor(clamp(x, 0.0, 255.0) * (1023.0 / 255.0) + 0.5) * (255.0 / 1023.0))   // the 10-bit code FG_STORE gives
#else
#define V2_LV(x) (x)
#define V2_D(a, b) abs(a - b)
#define V2_G(a, b) a - b
#define V2_QUANT(x) clamp(floor(x + 0.5), 0.0, 255.0)
#endif
#endif
