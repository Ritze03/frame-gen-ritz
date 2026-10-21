// V2's synthesis (included by v2_morph.comp = Performance and v2_qmorph.comp = Quality, which defines V2_QUALITY), the synthesis in one full-resolution pass (recordSynth: n = 1; recordSynthBlur: n sub-frames averaged in registers,
// each exactly what recordSynth(t_k) gives), plus a fix-up pass that is a no-op unless the whole-frame flick fallback fired:
//   g   = the cleaned block field upsampled bilinearly to this pixel (v2_motion.glsl hvAt), taken to time t by the fixed point
//         g = h(q - (2t - 1) g) (2 steps from h(q): lookup.comp's, here per pixel on the smooth field);
//   out = (1 - t) prev(q - 2t g) + t curr(q + 2(1 - t) g), both taps bilinear, CrosshairV2's tap rejection on the pair;
//   blended toward the zero vector's (1 - t) prev(q) + t curr(q) where the two frames show q (and a line of its neighbours) the same
//   but the morph's taps disagree: HUD text and other static overlays, static background that the smooth field would drag along.
// No per-pixel trust and no hold: where the two taps disagree (occlusion, a flash) the blend morphs between them; a real frame is never
// shown locally (the zero vector only where both frames AGREE). The only real-frame outputs are whole-frame decisions, uniform over the
// dispatch: a scene cut (mean finest SAD >
// cutSad, known from the estimate) or a flick (more than globalFrac of the frame untrusted) present the nearer real frame (prev for
// t <= 0.5, curr after), as fallback.comp does for V1; and a blur sub-frame at t <= 0 / t >= 1 is the real frame prev / curr
// (blur_resolve.comp's rule).
// The flick count needs the whole frame, so (as blur_resolve.comp does) the main pass (fix = 0) assumes no flick and counts: an 8x8
// tile (= this workgroup) is untrusted when the mean of its per-pixel mismatch max-channel |warped prev - warped curr| exceeds
// trustMid (V1: a 7x7 sliding mean past the trust ramp's midpoint); its valid pixels go to u[cnt + k]. The fix-up pass (fix = 1, one
// workgroup per 64x64 region) exits at once unless some sample flicked, and then redoes the output with the final counts.
#include "common.glsl"
#include "v2_motion.glsl"
#ifdef V2_QUALITY
#include "v2_quality.glsl"
#endif
layout(local_size_x = 8, local_size_y = 8) in;
shared uint esum[16];
const float V2_Z0 = 8.0, V2_Z1 = 12.0, V2_ZMAX = 6.0;
// max-channel |prev - curr| at pixel p (clamped to the frame), 8-bit levels
float zAt(ivec2 p) {
    p = clamp(p, ivec2(0), ivec2(pc.W - 1, pc.H - 1));
    vec3 d = abs(FG_FETCH(texelFetch(texPrev, p, 0).rgb) - FG_FETCH(texelFetch(texCurr, p, 0).rgb));
    return max(d.r, max(d.g, d.b));
}

// One generated sub-frame at time t (0 < t < 1): 8-bit levels, unrounded; al = the taps' alpha, blended alike; e = the mismatch.
vec3 morph(ivec2 p, float t, out float al, out uint e) {
    vec2 q = vec2(p);
#ifdef V2_QUALITY
#define V2_HV hvQ   // the per-pixel field
#else
#define V2_HV hvAt  // the block field
#endif
    vec2 g = V2_HV(q);
    if (t != 0.5) {
        float s = 2.0 * t - 1.0;
        for (int k = 0; k < 2; k++) g = V2_HV(q - s * g);
    }
    vec2 pa = q - (2.0 * t) * g;
    vec2 pc_ = q + (2.0 - 2.0 * t) * g;
    vec4 a = tapRaw(texPrev, pa), c = tapRaw(texCurr, pc_);
    vec3 ra = a.rgb, rc = c.rgb;
    uiReject(pa, pc_, ra, rc);
    vec3 wa = FG_FETCH(ra), wc = FG_FETCH(rc);
#ifdef V2_QUALITY
    if (pc.gauss >= 2) qRejectStatic(pa, pc_, wa, wc);   // TEMP knob: a tap on a static overlay (subtitle text) takes the other tap
#endif
    vec3 d = abs(wa - wc);
    float em = max(d.r, max(d.g, d.b));
#ifdef V2_QUALITY
    float wt = qWeightC(q, g, t);   // the visible side: curr's share of the blend
#else
    float wt = t;
#endif
    al = (1.0 - wt) * a.a + wt * c.a;
    vec3 m = (1.0 - wt) * wa + wt * wc;
#ifdef V2_QUALITY
    // TEMP experiment: blur along the motion where the taps disagree
    if (pc.fIn > 0 && em > Q_B0) {
        float beta = clamp((em - Q_B0) * (1.0 / Q_B1), 0.0, 1.0);
        vec2 st = (float(pc.fIn) * 0.01) * (2.0 * g);   // half-length of the streak
        vec3 acc = vec3(0.0);
        for (int j = -2; j <= 2; j++) {
            if (j == 0) continue;
            vec2 o = st * (float(j) * 0.5);
            vec2 qa = pa + o, qc = pc_ + o;
            vec3 xa = tapRaw(texPrev, qa).rgb, xc = tapRaw(texCurr, qc).rgb;
            uiReject(qa, qc, xa, xc);
            vec3 ya = FG_FETCH(xa), yc = FG_FETCH(xc);
            if (pc.gauss >= 2) qRejectStatic(qa, qc, ya, yc);
            acc += (1.0 - wt) * ya + wt * yc;
        }
        m = mix(m, (m + acc) * 0.2, beta);
    }
#endif
    // the zero vector where it fits clearly better: a pixel the two frames show unchanged (a HUD element, text, static background
    // next to a moving object, which the smooth field would drag along) keeps its place
#ifdef V2_QUALITY
    // Quality: its field matches the moving background well, so over a static overlay (subtitle text over a moving scene) both taps
    // can agree on the background and em is small: a pixel with strong local structure counts as disagreeing that much
    float emz = em < V2_Z0 + V2_Z1 + V2_ZMAX ? max(em, qStructure(p)) : em;
    float zq = qZeroMask(q);   // v2_qfield's static-overlay mask
#else
    float emz = em, zq = 0.0;
#endif
    if (emz > V2_Z0 || zq > 0.0) {   // (only then can it win)
        vec4 s0 = texelFetch(texPrev, p, 0), s1 = texelFetch(texCurr, p, 0);
        vec3 sa = FG_FETCH(s0.rgb), sc = FG_FETCH(s1.rgb);
        vec3 dz = abs(sa - sc);
        float z = max(dz.r, max(dz.g, dz.b));
        // ... and so do both neighbours along x or along y (a stroke of text or of a HUD element): a lone pixel whose colours happen to
        // match in a moving texture does not switch (it would be a speck)
        float zh = max(zAt(p + ivec2(-1, 0)), zAt(p + ivec2(1, 0))), zv = max(zAt(p + ivec2(0, -1)), zAt(p + ivec2(0, 1)));
        float zs = max(z, min(zh, zv));
        float ws = max(clamp((emz - zs - V2_Z0) * (1.0 / V2_Z1), 0.0, 1.0) * clamp(1.0 - zs * (1.0 / V2_ZMAX), 0.0, 1.0), zq);
        // not next to CrosshairV2's UI (its own patch and tap rejection handle it; a static crosshair would vouch for the pixels around it)
        if (ws > 0.0 && (uiHitA(p - 1, p + 1) || uiHitC(p - 1, p + 1))) ws = 0.0;
        m = mix(m, (1.0 - t) * sa + t * sc, ws);
        al = mix(al, (1.0 - t) * s0.a + t * s1.a, ws);
        em = mix(em, z, ws);
    }
    e = uint(em);
    return m;
}

// The output pixel p. counting: add each interior sample's mismatch to esum[k] (all invocations of the workgroup call it, p inside or
// not: subgroup adds), flicks unknown (assumed none); else the flicks are read from the final counts.
void pixelOut(ivec2 p, bool counting, bool cut) {
    bool inside = p.x < pc.W && p.y < pc.H;
    float limit = pc.globalFrac * float(pc.W) * float(pc.H);
    vec3 acc = vec3(0.0);
    float accA = 0.0, ws = 0.0;
    for (int k = 0; k < pc.n; k++) {
        float t = pc.t[k];
        // midpoint rule: slice centres map to x = (2k + 1 - N) / N, the window ends to x = -+1 (blur_resolve.comp)
        float x = (2.0 * float(k) + 1.0 - float(pc.n)) / float(pc.n);
        float w = (pc.gauss & 1) != 0 ? exp(-2.0 * x * x) : 1.0;
        bool real = t <= 0.0 || t >= 1.0;
        bool fb = cut || (!counting && float(u[pc.cnt + k]) > limit);   // uniform
        vec3 v = vec3(0.0);
        float a = 0.0;
        uint e = 0u;
        if (inside) {
            if (real || fb) {   // a real frame: the nearer one (t <= 0: prev, t >= 1: curr)
                vec4 r = t > 0.5 ? texelFetch(texCurr, p, 0) : texelFetch(texPrev, p, 0);
                v = FG_FETCH(r.rgb);
                a = r.a;
            } else {
                v = clamp(floor(morph(p, t, a, e) + 0.5), 0.0, 255.0);   // the 8-bit sub-frame recordSynth(t) stores
            }
        }
        if (counting && !real && !cut) {   // uniform
            uint s = subgroupAdd(e);
            if (subgroupElect()) atomicAdd(esum[k], s);
        }
        acc += w * v;
        accA += w * a;
        ws += w;
    }
    if (inside) imageStore(outImg, p, FG_STORE(acc / ws, accA / ws));
}

void main() {
    int lid = int(gl_LocalInvocationIndex);
    float meanSad = float(u[U_V2SAD_]) * (1.0 / 256.0) / float(pc.bw * pc.bh);
    bool cut = meanSad > pc.cutSad;
    if (pc.fix == 0) {
        if (lid < 16) esum[lid] = 0u;
        barrier();
        pixelOut(ivec2(gl_GlobalInvocationID.xy), true, cut);
        barrier();
        if (lid < pc.n && !cut) {
            ivec2 o = ivec2(gl_WorkGroupID.xy) * 8;
            ivec2 nv = min(ivec2(8), ivec2(pc.W, pc.H) - o);
            uint valid = uint(nv.x * nv.y);
            if (float(esum[lid]) > pc.trustMid * float(valid)) atomicAdd(u[pc.cnt + lid], valid);
        }
        return;
    }
    // fix-up: only if some interpolated sample flicked (and no cut, which the main pass already applied)
    if (cut) return;
    bool flicked = false;
    float limit = pc.globalFrac * float(pc.W) * float(pc.H);
    for (int k = 0; k < pc.n; k++)
        if (pc.t[k] > 0.0 && pc.t[k] < 1.0 && float(u[pc.cnt + k]) > limit) flicked = true;
    if (!flicked) return;
    for (int j = 0; j < 8; j++)
        for (int i = 0; i < 8; i++)
            pixelOut(ivec2(gl_WorkGroupID.xy) * 64 + ivec2(i, j) * 8 + ivec2(gl_LocalInvocationID.xy), false, false);
}
