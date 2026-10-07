// Motion version 2 of the motion-blur passes (recordSynthBlur): the push constants and helpers of v2_blur_lookup.comp,
// v2_blur_error.comp and v2_blur_resolve.comp. Included after blur_common.glsl. All three push this block over the same 128-byte
// range as V1's blur passes push theirs. Rgba8 only. The helpers are those of motion_v2.glsl reading the per-sample field from the
// blur scratch (bf[], one slot of rw x rh per sample: V2's refined field at t_k), so a sub-frame is the computation of V2's
// recordSynth at its t (v2_lookup, v2_error, v2_resolve_core.glsl).

layout(set = 0, binding = 13, std430) buffer V2Buf { uint v2b[]; };   // motion_v2.glsl: the occlusion map (class map: not written here)
layout(set = 0, binding = 14, std430) buffer V2Field { vec2 rf[]; };  // motion_v2.glsl: the refined midpoint field at offset 0

layout(push_constant) uniform BPC2 {
    int wh;            // full resolution W | H << 16
    int fwh;           // flow resolution fw | fh << 16
    int rwh;           // the refinement grid rw | rh << 16
    int misc;          // fscale (bits 0-3) | n << 4 (samples of the output, 5 bits) | k0 << 9 (first sample a lookup / error dispatch
                       // covers, sample = k0 + gl_WorkGroupID.z, 5 bits) | flags << 16 (bit 16: Gaussian weights, else even; bit 17:
                       // the fix-up pass, which redoes the output if needed, else the main pass, which counts the untrusted pixels)
    int blocks;        // finest block count bw * bh (scene-cut mean)
    float cutSad, trust0, trustScale, globalFrac;   // runtime settings (trust: V2Knobs trustLow / trustHigh)
    float occ0, occScale, pickBias, zeroBonus;      // V2Knobs: the occlusion ramp, pickBias; Settings::hudBonus
    float t[16];       // time of each sample, in [0, 1]; <= 0 / >= 1 = the real frame prev / curr
} pc;                  // (3 words free)

#define P_W (pc.wh & 0xFFFF)
#define P_H (pc.wh >> 16)
#define P_FW (pc.fwh & 0xFFFF)
#define P_FH (pc.fwh >> 16)
#define P_RW (pc.rwh & 0xFFFF)
#define P_RH (pc.rwh >> 16)
#define P_FSCALE (pc.misc & 15)
#define P_N ((pc.misc >> 4) & 31)
#define P_K0 ((pc.misc >> 9) & 31)
#define P_GAUSS ((pc.misc & 0x10000) != 0)
#define P_FIX ((pc.misc & 0x20000) != 0)

const uint V2_E_UNKNOWN = 255u;   // == motion_v2.glsl
const uint V2_CLS_TRUSTED = 0u, V2_CLS_HELD = 1u, V2_CLS_OCC_FIELD = 2u, V2_CLS_OCC_BG = 3u, V2_CLS_OCC_FG = 4u, V2_CLS_OCC_ZERO = 5u;
int eStride() { return (P_W + 3) >> 2; }
int eWords() { return eStride() * P_H; }
const int CNT0 = 32;      // u[CNT0 + k]: untrusted-pixel count of sample k (== blur_motion_v1.glsl)

vec2 BF(int slot, int off) { return bf[slot * P_RW * P_RH + off]; }

// == motion_v2.glsl halfVec, the refined field of sample `slot`
vec2 halfVec(int slot, ivec2 p) {
    int rw = P_RW, rh = P_RH;
    if (rw == P_W && rh == P_H)   // 100 %: one read (== motion_v2.glsl)
        return BF(slot, clamp(p.y, 0, rh - 1) * rw + clamp(p.x, 0, rw - 1));
    vec2 s =(vec2(p) + 0.5) * (vec2(rw, rh) / vec2(P_W, P_H)) - 0.5;
    vec2 fl = floor(s);
    vec2 f = s - fl;
    ivec2 i0 = clamp(ivec2(fl), ivec2(0), ivec2(rw - 1, rh - 1));
    ivec2 i1 = clamp(ivec2(fl) + 1, ivec2(0), ivec2(rw - 1, rh - 1));
    vec2 v00 = BF(slot, i0.y * rw + i0.x), v01 = BF(slot, i0.y * rw + i1.x);
    vec2 v10 = BF(slot, i1.y * rw + i0.x), v11 = BF(slot, i1.y * rw + i1.x);
    return mix(mix(v00, v01, f.x), mix(v10, v11, f.x), f.y);
}

vec3 sampleRGBRaw(sampler2D tx, vec2 p) {
    p = roundEven(p * 32.0) * (1.0 / 32.0);
    return textureLod(tx, (p + 0.5) / vec2(P_W, P_H), 0.0).rgb;
}

// == motion_v2.glsl mismatchV() at time t (wA = 1 - t, wC = t)
uint mismatchV(float t, ivec2 p, vec2 hv, out vec3 wa, out vec3 wc) {
    float wA = 1.0 - t, wC = t;
    precise float ka = 2.0 * wC, kc = 2.0 * wA;
    precise vec2 pa = vec2(p) - ka * hv;
    precise vec2 pc_ = vec2(p) + kc * hv;
    vec3 ra = sampleRGBRaw(texPrev, pa), rc = sampleRGBRaw(texCurr, pc_);
    uiReject(pa, pc_, ra, rc);
    wa = FG_FETCH(ra);
    wc = FG_FETCH(rc);
    vec3 d = FG_DIFF(wa, wc);
    return FG_PACKE(max(d.r, max(d.g, d.b)));
}

// == motion_v2.glsl mismatchE at time t
uint mismatchE(float t, ivec2 p, vec2 hv) {
    float wA = 1.0 - t, wC = t;
    precise float ka = 2.0 * wC, kc = 2.0 * wA;
    precise vec2 pa = vec2(p) - ka * hv;
    precise vec2 pc_ = vec2(p) + kc * hv;
    bool ha, hc;
    if (uiRejectHits(pa, pc_, ha, hc)) return V2_E_UNKNOWN;
    vec3 d = FG_DIFF(FG_FETCH(sampleRGBRaw(texPrev, pa)), FG_FETCH(sampleRGBRaw(texCurr, pc_)));
    return min(FG_PACKE(max(d.r, max(d.g, d.b))), V2_E_UNKNOWN - 1u);
}

// == motion_v2.glsl occAt
uvec4 occAt(ivec2 p, vec2 hv, float s) {
    vec2 m = vec2(p) - s * hv;
    ivec2 j = clamp(ivec2(floor((m + 0.5) / float(P_FSCALE))), ivec2(0), ivec2(P_FW - 1, P_FH - 1));
    int i = 4 * (j.y * P_FW + j.x);
    return uvec4(v2b[i], v2b[i + 1], v2b[i + 2], v2b[i + 3]);
}

uint eAt(int slot, ivec2 p) {
    p = clamp(p, ivec2(0), ivec2(P_W - 1, P_H - 1));
    return (be[slot * eWords() + p.y * eStride() + (p.x >> 2)] >> (8 * (p.x & 3))) & 255u;
}
