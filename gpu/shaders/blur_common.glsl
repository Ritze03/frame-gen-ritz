// Shared declarations of the motion-blur passes (recordSynthBlur). The descriptor set is the one of every other pass
// (framegen.cpp: binding 0 = luma is not used here); bindings 6 and 7 are the blur scratch. Own push-constant block:
// the passes of ONE blurred output run as one dispatch each, with the sample index in gl_WorkGroupID.z (k0 + z).
// The warp / sampling code is a verbatim copy of common.glsl's (halfVec, sampleRGB, mismatch), reading the
// per-sample field from the blur scratch instead of the single scratch field, so that a sub-frame of the blur is
// bit-identical to the same t through recordSynth().

#include "format.glsl"

layout(set = 0, binding = 1, std430) buffer FieldBuf { vec2 field[]; };   // estimate: per-pixel field (read only here)
layout(set = 0, binding = 2, std430) buffer UBuf { uint u[]; };           // counters (and the estimate's SAD sum)
layout(set = 0, binding = 3) uniform sampler2D texPrev;
layout(set = 0, binding = 4) uniform sampler2D texCurr;
layout(set = 0, binding = 5, FG_FMT) uniform writeonly image2D outImg;
#ifdef FG_YUV
layout(set = 0, binding = 8) uniform sampler2D texPrevC;
layout(set = 0, binding = 9) uniform sampler2D texCurrC;
layout(set = 0, binding = 10, FG_FMTC) uniform writeonly image2D outImgC;
#endif
layout(set = 0, binding = 6, std430) buffer BField { vec2 bf[]; };        // field at t_k, flow res, n slots of fw*fh
layout(set = 0, binding = 7, std430) buffer BErr { uint be[]; };          // packed mismatch of sample k, n slots

layout(push_constant) uniform BPC {
    int W, H;          // full resolution
    int fw, fh;        // flow resolution
    int fscale;        // flow scale (1, 2, 4)
    int n;             // samples of the output
    int bw, bh;        // finest block grid (scene-cut mean)
    int k0;            // first sample the dispatch covers (sample = k0 + gl_WorkGroupID.z)
    int pf;            // offset of the estimate's per-pixel field in field[]
    int gauss;         // 0 = even weights, 1 = Gaussian
    int fix;           // resolve: 0 = main pass (counts the untrusted pixels), 1 = fix-up pass (redoes the output if needed)
    float cutSad, trust0, trustScale, globalFrac;   // runtime settings
    float t[16];       // time of each sample, in [0, 1]; <= 0 / >= 1 = the real frame prev / curr
} pc;

int eStride() { return (pc.W + 3) >> 2; }
int eWords() { return eStride() * pc.H; }
const int CNT0 = 32;      // u[CNT0 + k]: untrusted-pixel count of sample k (the estimate uses u[0..~16])

vec2 BF(int slot, int off) { return bf[slot * pc.fw * pc.fh + off]; }

// == common.glsl halfVec, field of sample `slot` instead of field[pf..]
vec2 halfVec(int slot, ivec2 p) {
    int S = pc.fscale;
    p = min(p, ivec2(S * pc.fw - 1, S * pc.fh - 1));
    vec2 s = (vec2(p) + 0.5) * (1.0 / float(S)) - 0.5;
    vec2 fl = floor(s);
    vec2 f = s - fl;
    ivec2 i0 = clamp(ivec2(fl), ivec2(0), ivec2(pc.fw - 1, pc.fh - 1));
    ivec2 i1 = clamp(ivec2(fl) + 1, ivec2(0), ivec2(pc.fw - 1, pc.fh - 1));
    vec2 v00 = BF(slot, i0.y * pc.fw + i0.x), v01 = BF(slot, i0.y * pc.fw + i1.x);
    vec2 v10 = BF(slot, i1.y * pc.fw + i0.x), v11 = BF(slot, i1.y * pc.fw + i1.x);
    return float(S) * mix(mix(v00, v01, f.x), mix(v10, v11, f.x), f.y);
}

#ifndef FG_YUV
vec3 sampleRGB(sampler2D tx, vec2 p) {
    p = roundEven(p * 32.0) * (1.0 / 32.0);
    return FG_FETCH(textureLod(tx, (p + 0.5) / vec2(pc.W, pc.H), 0.0).rgb);
}
#endif
#ifdef FG_YUV
vec3 sampleYUV(sampler2D ty, sampler2D tc, vec2 p) {
    p = roundEven(p * 32.0) * (1.0 / 32.0);
    vec2 uv = (p + 0.5) / vec2(pc.W, pc.H);
    return vec3(textureLod(ty, uv, 0.0).r, textureLod(tc, uv, 0.0).rg) * YU_SCALE;
}
vec2 sampleC(sampler2D tc, vec2 p) {
    p = roundEven(p * 32.0) * (1.0 / 32.0);
    return textureLod(tc, (p + 0.5) / vec2(pc.W, pc.H), 0.0).rg * YU_SCALE;
}
#define FG_SAMPLE(t, tc, p) sampleYUV(t, tc, p)
#else
#define FG_SAMPLE(t, tc, p) sampleRGB(t, p)
#endif

// == common.glsl mismatch() at t = pc.t[k]: wA = 1 - t, wC = t
uint mismatch(int slot, float t, ivec2 p, out vec3 wa, out vec3 wc) {
    float wA = 1.0 - t, wC = t;
    vec2 hv = halfVec(slot, p);
    precise float ka = 2.0 * wC, kc = 2.0 * wA;
    precise vec2 pa = vec2(p) - ka * hv;
    precise vec2 pc_ = vec2(p) + kc * hv;
    wa = FG_SAMPLE(texPrev, texPrevC, pa);
    wc = FG_SAMPLE(texCurr, texCurrC, pc_);
    vec3 d = FG_DIFF(wa, wc);
    return FG_PACKE(max(d.r, max(d.g, d.b)));
}

#ifdef FG_YUV
// == common.glsl warpChroma() at t: the chroma of the 2x2 block of the even pixel p, warped with sample `slot`'s field
void warpChroma(int slot, float t, ivec2 p, out vec2 ca, out vec2 cc) {
    float wA = 1.0 - t, wC = t;
    vec2 hv = halfVec(slot, p);
    precise float ka = 2.0 * wC, kc = 2.0 * wA;
    vec2 q = vec2(p) + 0.5;
    precise vec2 pa = q - ka * hv;
    precise vec2 pc_ = q + kc * hv;
    ca = sampleC(texPrevC, pa);
    cc = sampleC(texCurrC, pc_);
}
#endif

// --- the 7x7 mean of the packed mismatch of sample `slot` (resolve.comp's trust stage)
const int TS = 16, AP = 3, TT = TS + 2 * AP;   // a workgroup is one 16 x 16 tile
shared float te[TT * TT];
shared float hs[TT * TS];

uint eAt(int slot, ivec2 p) {
    p = clamp(p, ivec2(0), ivec2(pc.W - 1, pc.H - 1));
    return (be[slot * eWords() + p.y * eStride() + (p.x >> 2)] >> (8 * (p.x & 3))) & 255u;
}

// Mean mismatch of the 7x7 window at this thread's pixel of tile `tile` (16 x 16 pixels). Every thread of the
// workgroup must call it (barriers).
float boxMean(int slot, ivec2 tile) {
    int lid = int(gl_LocalInvocationIndex);
    ivec2 t0 = tile * TS - AP;
    barrier();   // the previous use of te/hs is finished
    for (int i = lid; i < TT * TT; i += TS * TS)
        te[i] = float(eAt(slot, t0 + ivec2(i % TT, i / TT)));
    barrier();
    for (int i = lid; i < TT * TS; i += TS * TS) {   // horizontal 7-sums: TT rows x TS cols
        int r = i / TS, c = i % TS;
        float s = 0.0;
        for (int k = 0; k < 7; k++) s += te[r * TT + c + k];
        hs[i] = s;
    }
    barrier();
    ivec2 l = ivec2(gl_LocalInvocationID.xy);
    float s = 0.0;
    for (int k = 0; k < 7; k++) s += hs[(l.y + k) * TS + l.x];
    return s * (1.0 / 49.0);
}
