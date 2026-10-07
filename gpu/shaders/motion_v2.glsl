// Motion version 2 (Settings::motionVersion = V2): the push constants and helpers of V2's own synth / estimate passes (v2_smooth,
// v2_occl, v2_refine, v2_lookup, v2_error, v2_resolve). Included after common.glsl. Rgba8 only (settingsError() refuses V2 for every
// other Format), so there is no YUV or other-format path here. V2 shares V1's estimate passes up to the pixel field and its fallback
// pass (they push V1's block, motion_v1.glsl); its own passes push this block over the same 128-byte range of the same pipeline
// layout.
//
// V2's own buffer (binding 13, framegen.cpp MotionGroup::bufs[0]):
//   words [0, 4 fw fh): the occlusion map, one uvec4 per flow-resolution pixel of the MIDPOINT grid (v2_occl.comp):
//     x = packHalf2x16(background half-vector), y = packHalf2x16(foreground half-vector) (flow-resolution units, like field[]),
//     z = side: 0 = no clean occlusion, 1 = covered (take prev), 2 = uncovered (take curr), w = 1: the content rescue put this
//     pixel's vector back (v2_merge.comp; the resolve does not hold it), else 0.
//   words [clsOff, clsOff + eStride H): the class map of the latest synth (debug, recordDebugMaps), one byte per pixel, 4 per
//     word, low byte first: bits 0-2 class (V2_CLS_*: 0 trusted, 1 held, clean occlusion with 2 the field's vector, 3 vB, 4 vF, 5 zero), bit 3 = side curr, bits 4-7 = weight of the hold / one-sided output x 15.
//   then the temporal store (D19, I10: the previous pair's field, v2_temporal.glsl; written by v2_seed / v2_occl, read by v2_seed,
//     v2_search, v2_smooth, v2_occl).
//
// V2's fields (D17, D13). The estimate adds two passes after the pixel field:
//   v2_smooth.comp (flow res): the edge-aware field hE, a guided weighted vector median of the pixel field (written to field[] at the
//     slot V1 uses for its field at t, which V2 never needs: framegen.cpp fieldT).
//   v2_refine.comp (the refinement grid rw x rh, Settings::v2RefineRes % of the frame): the edge-preserving upsample of hE, written
//     to rf[] (binding 14, framegen.cpp MotionGroup::bufs[1]) at offset 0 in FULL-RESOLUTION half-vector units, packed (one uint per
//     grid point, v2_pack.glsl: 1/32 px fixed point; I10, half the memory of vec2, and at t = 0.5 exactly the taps of the unrounded
//     vector).
// Every full-resolution pass of V2 (v2_error, v2_resolve, the blur passes) reads its half-vector from rf[] with halfVec() below:
// bilinear on the refinement grid, i.e. the refined field itself at 100 %. The field at t != 0.5 (v2_lookup.comp) is looked up on the
// same grid, into rf[] at offset rw * rh (the blur passes compute the same values on the fly: blur_motion_v2.glsl halfVecT).

layout(set = 0, binding = 13, std430) buffer V2Buf { uint v2b[]; };
layout(set = 0, binding = 14, std430) buffer V2Field { uint rf[]; };   // refined fields (full-resolution half-vector units, packed:
                                                                        // v2_pack.glsl)

layout(push_constant) uniform PC2 {
    int W, H;          // full resolution
    int fw, fh;        // flow resolution (= the finest pyramid level)
    int fscale;        // flow scale (1, 2, 4)
    int pf;            // offset in field[] of the flow-resolution field the pass reads (v2_smooth: the pixel field; v2_occl,
                       // v2_refine: the edge-aware field hE)
    int offA, offC;    // v2_occl: luma offsets of the finest level
    int reach;         // v2_occl: side-vector distance (flow px)
    int clsOff;        // word offset of the class map in v2b[]
    float s;           // 2t - 1
    float wA, wC;      // 1 - t, t
    float trust0, trustScale;   // the held class's ramp (V2Knobs trustLow, 1 / (trustHigh - trustLow))
    float occ0, occScale;       // the occlusion ramp (V2Knobs occLow, 1 / (occHigh - occLow))
    float pickBias;             // V2Knobs pickBias
    float edgeMin;              // v2_occl: min normal motion difference, flow-resolution half-vector units (occlWidth / fscale);
                                // v2_resolve (synth): V2Knobs holdFeather (px)
    float confMax;              // v2_occl: V2Knobs confidence; v2_resolve (synth): V2Knobs occSoft
    float zeroBonus;            // Settings::hudBonus (the zero candidate at a clean occlusion)
    int rw, rh;                 // the refinement grid (V2Knobs refineRes % of W x H)
    int rpf;                    // offset in rf[] of the refined field the pass reads (0 = the midpoint field, rw * rh = at t)
    int rfOut;                  // v2_refine / v2_lookup: offset in rf[] written
    int fE;                     // offset in field[] of the edge-aware flow-resolution field (v2_smooth writes it)
    float edgeK;                // 1 / V2Knobs edgeSigma (levels): the range weight of the guide
    int tFlags;                 // the temporal store (v2_temporal.glsl): bit 0 = parity of this estimate (slot written; the
                                // previous pair's is the other), bit 1 = the host's "a previous V2 estimate exists" (v2_seed.comp)
    float tBonus;               // v2_smooth: V2Knobs temporalBonus (levels, 3x3 mean): how much worse the compensated previous
                                // vector may explain a pixel and still replace its own
    int lw, lh;                 // v2_seed: the coarsest pyramid level's size (its luma at offA / offC)
    float rescue;               // v2_rescue: V2Knobs rescue (levels): the largest 3x3 mean |a - c| of a rescued match
} pc;

#define V2_RF(i) rf[(i)]
#include "v2_pack.glsl"

const int E_OFF = 64;          // packed mismatch in u[] (== motion_v1.glsl)
#define E_STRIDE ((pc.W + 3) >> 2)
const int U_UNTRUSTED = 0;
#define NEAR_C (pc.wC > 0.5)
#define MID (pc.wC == 0.5)
const uint V2_E_UNKNOWN = 255u;   // v2_error.comp: a warp tap was on the UI (known e is capped at 254)
const uint V2_CLS_TRUSTED = 0u, V2_CLS_HELD = 1u, V2_CLS_OCC_FIELD = 2u, V2_CLS_OCC_BG = 3u, V2_CLS_OCC_FG = 4u, V2_CLS_OCC_ZERO = 5u;
const uint V2_CLS_RESCUED = 6u;   // (I11) the content rescue's vector, not held

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

// The full-resolution half-vector at pixel p: the refined field at rpf, bilinear on the refinement grid (pixel centres aligned like
// cv2.resize). At 100 % refinement the grid is the frame: the refined vector of p itself (fraction 0, exact).
vec2 halfVec(ivec2 p) {
    if (pc.rw == pc.W && pc.rh == pc.H)   // 100 %: one read (the bilinear weights below would be 1, 0, 0, 0)
        return unpackV(rf[pc.rpf + clamp(p.y, 0, pc.H - 1) * pc.rw + clamp(p.x, 0, pc.W - 1)]);
    vec2 s =(vec2(p) + 0.5) * (vec2(pc.rw, pc.rh) / vec2(pc.W, pc.H)) - 0.5;
    vec2 fl = floor(s);
    vec2 f = s - fl;
    ivec2 i0 = clamp(ivec2(fl), ivec2(0), ivec2(pc.rw - 1, pc.rh - 1));
    ivec2 i1 = clamp(ivec2(fl) + 1, ivec2(0), ivec2(pc.rw - 1, pc.rh - 1));
    vec2 v00 = unpackV(rf[pc.rpf + i0.y * pc.rw + i0.x]), v01 = unpackV(rf[pc.rpf + i0.y * pc.rw + i1.x]);
    vec2 v10 = unpackV(rf[pc.rpf + i1.y * pc.rw + i0.x]), v11 = unpackV(rf[pc.rpf + i1.y * pc.rw + i1.x]);
    return mix(mix(v00, v01, f.x), mix(v10, v11, f.x), f.y);
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

// v2_error.comp's e at p with the half-vector hv: as mismatchV, but V2_E_UNKNOWN when a tap was on its frame's UI (CrosshairV2 tap
// rejection: the pixel shows the other frame's tap for both, so its mismatch says nothing; v2_resolve_core.glsl's trust box skips it)
uint mismatchE(ivec2 p, vec2 hv) {
    precise float ka = 2.0 * pc.wC, kc = 2.0 * pc.wA;
    precise vec2 pa = vec2(p) - ka * hv;
    precise vec2 pc_ = vec2(p) + kc * hv;
    bool ha, hc;
    if (uiRejectHits(pa, pc_, ha, hc)) return V2_E_UNKNOWN;
    vec3 d = FG_DIFF(FG_FETCH(sampleRGBRaw(texPrev, pa)), FG_FETCH(sampleRGBRaw(texCurr, pc_)));
    return min(FG_PACKE(max(d.r, max(d.g, d.b))), V2_E_UNKNOWN - 1u);
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
