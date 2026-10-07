// Motion version 2 of the motion-blur passes (recordSynthBlur): the push constants and helpers of v2_blur_resolve.comp. Included after
// blur_common.glsl. V2 shares V1's blur_lookup.comp and blur_error.comp (they push V1's block, blur_motion_v1.glsl); its resolve pushes
// this block over the same 128-byte range (32 words: full). Rgba8 only. The helpers are those of motion_v2.glsl reading the per-sample field from the
// blur scratch, so a sub-frame is the computation of v2_resolve.comp at its t (v2_resolve_core.glsl).

layout(set = 0, binding = 13, std430) buffer V2Buf { uint v2b[]; };   // motion_v2.glsl: the occlusion map (class map: not written here)

layout(push_constant) uniform BPC2 {
    int W, H;          // full resolution
    int fw, fh;        // flow resolution
    int fscale;        // flow scale (1, 2, 4)
    int n;             // samples of the output
    int blocks;        // finest block count bw * bh (scene-cut mean)
    int flags;         // bit 0: Gaussian weights (else even); bit 1: the fix-up pass (redoes the output if needed; else the main pass,
                       // which counts the untrusted pixels)
    float cutSad, trust0, trustScale, globalFrac;   // runtime settings (trust: V2Knobs trustLow / trustHigh)
    float occ0, occScale, pickBias, zeroBonus;      // V2Knobs: the occlusion ramp, pickBias; Settings::hudBonus
    float t[16];       // time of each sample, in [0, 1]; <= 0 / >= 1 = the real frame prev / curr
} pc;

const uint V2_CLS_TRUSTED = 0u, V2_CLS_HELD = 1u, V2_CLS_OCC_FIELD = 2u, V2_CLS_OCC_BG = 3u, V2_CLS_OCC_FG = 4u, V2_CLS_OCC_ZERO = 5u;
int eStride() { return (pc.W + 3) >> 2; }
int eWords() { return eStride() * pc.H; }
const int CNT0 = 32;      // u[CNT0 + k]: untrusted-pixel count of sample k (== blur_motion_v1.glsl)

vec2 BF(int slot, int off) { return bf[slot * pc.fw * pc.fh + off]; }

// == motion_v2.glsl halfVec, field of sample `slot`
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

vec3 sampleRGBRaw(sampler2D tx, vec2 p) {
    p = roundEven(p * 32.0) * (1.0 / 32.0);
    return textureLod(tx, (p + 0.5) / vec2(pc.W, pc.H), 0.0).rgb;
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

// == motion_v2.glsl occAt
uvec4 occAt(ivec2 p, vec2 hv, float s) {
    vec2 m = vec2(p) - s * hv;
    ivec2 j = clamp(ivec2(floor((m + 0.5) / float(pc.fscale))), ivec2(0), ivec2(pc.fw - 1, pc.fh - 1));
    int i = 4 * (j.y * pc.fw + j.x);
    return uvec4(v2b[i], v2b[i + 1], v2b[i + 2], v2b[i + 3]);
}

uint eAt(int slot, ivec2 p) {
    p = clamp(p, ivec2(0), ivec2(pc.W - 1, pc.H - 1));
    return (be[slot * eWords() + p.y * eStride() + (p.x >> 2)] >> (8 * (p.x & 3))) & 255u;
}
