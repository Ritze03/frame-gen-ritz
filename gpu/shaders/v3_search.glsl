// The body of v3_search.comp / v3_search_top.comp (gpu/V3.md section 3.5): V3S_TOP = 1 builds the top level's exhaustive search (the only level
// that needs the patch in shared memory: kept out of the other pipeline, whose workgroup occupancy depends on its shared memory), 0 the
// candidate search of the other levels. Both include the Lucas-Kanade refinement of level 0 (a one-level pyramid has its level 0 at the top).
// V3J = 1 (the v3_search_j / v3_search_top_j builds, Settings::v3ColourWindow, gpu/V3.md hook (j)): every window pixel carries the weight
// wgt = exp(-|colour(x) - colour(centre)|_1 / 24) of the role-0 frame's raw (Y, Cb, Cr) and the cost is sum(wgt term) / sum(wgt). V3J = 0 is the unchanged plain build.
#ifndef V3J
#define V3J 0
#endif
#include "common.glsl"
#include "v3_common.glsl"
#include "v3_est.glsl"
#if V3S_TOP
const int WG = 256;   // the top level: 4x the threads for the 289 candidates (threads 64 and up only work in the exhaustive part)
#else
const int WG = 64;
#endif
layout(local_size_x = WG) in;

const float BAD = 1000.0;
const float V3_DOMCAP = 0.5;   // the prior of a dominant-motion candidate is capped (levels): it can still win against a far parent where the match is much better

shared vec4 sf[64];        // the window's features of the role-0 frame
shared int sval[64];       // 1: the window pixel lies inside the level
shared ivec2 cv[16];       // candidate vectors
shared int cflag[16];
shared float cres[16], cpur[16];   // total (with the prior) and pure cost of each candidate
shared ivec2 refV;
shared float pSum[64], pCnt[64];
#if V3J
shared float sw[64];       // hook (j): the window pixel's colour weight (0 outside the level)
shared float pWgt[64];
#endif
shared float red[5 * 64];
shared uint blkDetU;       // task 105: the window's summed detail x 256 (hudB > 0 only; integer atomics, no extra barrier)
#if V3S_TOP
shared float topc[289];
shared float topp[64];
shared int topi[64];
shared uvec2 spatch[24 * 24];   // the search region of the role-1 frame (packed texels): the window +-8
#endif

int tid;
int dir, f0, f1;
ivec2 blk, wo;
V3Lv L;
int fOff, nWin;
vec2 dMu;   // (K, gm) of v3_est.glsl

bool inside(ivec2 t) { return all(greaterThanEqual(t, ivec2(0))) && all(lessThan(t, ivec2(L.w, L.h))); }
vec2 recXY(int i) { return vec2(fl[i], fl[i + 1]); }

// one round: the 16 candidates cv / cflag (the callers have written and synchronised them), cres / cpur written, synchronised on return
void evalRound() {
    const int c = tid >> 2, q = tid & 3;
    float s = 0.0, n = 0.0;
#if V3J
    float wsum = 0.0;
#endif
    if ((cflag[c] & 1) != 0) {
        const ivec2 v = cv[c];
        // (branch-free: the taps are always read, clamped into the level, and masked, so that the loads of the 16 pixels are issued together)
        for (int k = q * 16; k < q * 16 + 16; k++) {
            const ivec2 t = wo + ivec2(k & 7, k >> 3) + v;
            const float ok = (sval[k] != 0 && inside(t)) ? 1.0 : 0.0;
#if V3J
            const float tm = v3Term(sf[k], v3FeatAt(L, fOff, f1, clamp(t, ivec2(0), ivec2(L.w - 1, L.h - 1))), dMu);
            s += ok * sw[k] * tm;
            wsum += ok * sw[k];
#else
            s += ok * v3Term(sf[k], v3FeatAt(L, fOff, f1, clamp(t, ivec2(0), ivec2(L.w - 1, L.h - 1))), dMu);
#endif
            n += ok;
        }
    }
    pSum[tid] = s;
    pCnt[tid] = n;
#if V3J
    pWgt[tid] = wsum;
#endif
    barrier();
    if (tid < 16) {
        const float S = pSum[4 * tid] + pSum[4 * tid + 1] + pSum[4 * tid + 2] + pSum[4 * tid + 3];
        const float N = pCnt[4 * tid] + pCnt[4 * tid + 1] + pCnt[4 * tid + 2] + pCnt[4 * tid + 3];
        float cost = BAD;
#if V3J
        const float Wt = pWgt[4 * tid] + pWgt[4 * tid + 1] + pWgt[4 * tid + 2] + pWgt[4 * tid + 3];
        if ((cflag[tid] & 1) != 0 && N * 2.0 >= float(nWin) && N > 0.0 && Wt > 0.0) cost = S / Wt;
#else
        if ((cflag[tid] & 1) != 0 && N * 2.0 >= float(nWin) && N > 0.0) cost = S / N;
#endif
        const ivec2 d = abs(cv[tid] - refV);
        cpur[tid] = cost;
        float bon = 0.0;
        if (pc.hudB > 0.0 && cost < BAD && cv[tid] == ivec2(0)) {   // task 105: the zero vector's bonus; the gate is the reference (parent / own previous) vector's length
            const float ev = v3HudEv(float(blkDetU) * (1.0 / 256.0) / float(max(nWin, 1)));
            bon = pc.hudB * ev * v3HudAbs(cost) * v3HudGate(length(vec2(refV)) * float((1 << pc.lvl) * pc.fscale));
        }
        cres[tid] = (cflag[tid] & 8) != 0 ? 1e9 : cost + ((cflag[tid] & 2) != 0 ? 0.0 : min(V3_LAMBDA * float(d.x + d.y), (cflag[tid] & 4) != 0 ? V3_DOMCAP : 1e9)) - bon;
    }
    barrier();
}

// the best of cres[0..n-1] (lowest index on a tie); callers barrier before overwriting
void best(int n, out ivec2 v, out float tot, out float pure) {
    int bi = 0;
    float bt = cres[0];
    for (int i = 1; i < n; i++)
        if (cres[i] < bt) { bt = cres[i]; bi = i; }
    v = cv[bi]; tot = bt; pure = cpur[bi];
}

// the ring: the 8 integer neighbours of v, one round; v / tot / pure become the better of v and them. With probes (level 1, iteration 1) the same round also
// evaluates the 8 ambiguity probes around the incoming v (+-6 and +-12 level px along x, then along y; only windows wholly inside the level) and returns
// the pure cost of the best probe along x and along y (BAD: none)
void ring(inout ivec2 v, inout float tot, inout float pure, bool probes, out float pmx, out float pmy) {
    if (tid < 16) {
        if (tid < 8) {
            const int k = tid < 4 ? tid : tid + 1;   // the 3x3 without its centre (4)
            cv[tid] = v + ivec2(k % 3 - 1, k / 3 - 1);
            cflag[tid] = 1;
        } else {
            const int k = tid - 8, mag = (k & 1) != 0 ? 12 : 6, sg = (k & 2) != 0 ? -1 : 1;
            cv[tid] = v + ((k >> 2) == 0 ? ivec2(sg * mag, 0) : ivec2(0, sg * mag));
            const ivec2 tw = wo + cv[tid];   // a probe window that is clipped by the level's border matches on fewer pixels and is not comparable
            cflag[tid] = (probes && all(greaterThanEqual(tw, ivec2(0))) && all(lessThan(tw + 7, ivec2(L.w, L.h)))) ? 1 : 0;
        }
    }
    barrier();
    evalRound();
    ivec2 rv; float rt, rp;
    best(8, rv, rt, rp);
    pmx = BAD; pmy = BAD;
    for (int i = 0; i < 4; i++) { pmx = min(pmx, cpur[8 + i]); pmy = min(pmy, cpur[12 + i]); }
    if (rt < tot) { v = rv; tot = rt; pure = rp; }
    barrier();
}
void ring(inout ivec2 v, inout float tot, inout float pure) { float a, b; ring(v, tot, pure, false, a, b); }

// the sum of 5 values per thread over the 64 threads (the result in every thread): 5 x 8 partial sums of 8, then 8 per value
shared float redp[40];
void reduce5(inout float a[5]) {
    if (tid < 64)
        for (int k = 0; k < 5; k++) red[k * 64 + tid] = a[k];
    barrier();
    if (tid < 40) {
        float s = 0.0;
        for (int i = 0; i < 8; i++) s += red[(tid >> 3) * 64 + (tid & 7) * 8 + i];
        redp[tid] = s;
    }
    barrier();
    for (int k = 0; k < 5; k++) {
        float s = 0.0;
        for (int i = 0; i < 8; i++) s += redp[k * 8 + i];
        a[k] = s;
    }
}

bool insideF(vec2 tq) { return all(greaterThanEqual(tq, vec2(0.0))) && all(lessThanEqual(tq, vec2(float(L.w - 1), float(L.h - 1)))); }

// the pure cost at a sub-pixel vector: bilinear taps, one window pixel per thread
float costAt(vec2 v) {
    float a[5] = float[5](0.0, 0.0, 0.0, 0.0, 0.0);
    if (tid < 64 && sval[tid] != 0) {
        const vec2 tq = vec2(wo + ivec2(tid & 7, tid >> 3)) + v;
        if (insideF(tq)) {
#if V3J
            a[0] = sw[tid] * v3Term(sf[tid], v3FeatBil(L, fOff, f1, tq), dMu);
            a[2] = sw[tid];
#else
            a[0] = v3Term(sf[tid], v3FeatBil(L, fOff, f1, tq), dMu);
#endif
            a[1] = 1.0;
        }
    }
    reduce5(a);
#if V3J
    return (a[1] * 2.0 >= float(nWin) && a[1] > 0.0 && a[2] > 0.0) ? a[0] / a[2] : BAD;
#else
    return (a[1] * 2.0 >= float(nWin) && a[1] > 0.0) ? a[0] / a[1] : BAD;
#endif
}

// Lucas-Kanade sub-pixel refinement (section 3.5, level 0): 2 iterations, each accepted only if the cost falls
vec2 lk(ivec2 v0, inout float cost) {
    vec2 v = vec2(v0);
    for (int it = 0; it < 2; it++) {
        float a[5] = float[5](0.0, 0.0, 0.0, 0.0, 0.0);   // Axx, Axy, Ayy, bx, by
        if (tid < 64 && sval[tid] != 0) {
            const vec2 tq = vec2(wo + ivec2(tid & 7, tid >> 3)) + v;
            if (insideF(tq)) {
                const vec4 g = v3FeatBil(L, fOff, f1, tq);
                const vec4 gx = v3FeatBil(L, fOff, f1, tq + vec2(0.5, 0.0)) - v3FeatBil(L, fOff, f1, tq - vec2(0.5, 0.0));
                const vec4 gy = v3FeatBil(L, fOff, f1, tq + vec2(0.0, 0.5)) - v3FeatBil(L, fOff, f1, tq - vec2(0.0, 0.5));
                vec4 r = sf[tid] - dMu.y * g;
                r.w += dMu.x;
#if V3J
                const vec4 wx = (sw[tid] * V3_CW) * dMu.y * gx, wy = (sw[tid] * V3_CW) * dMu.y * gy;
#else
                const vec4 wx = V3_CW * dMu.y * gx, wy = V3_CW * dMu.y * gy;
#endif
                a[0] = dot(wx, dMu.y * gx); a[1] = dot(wx, dMu.y * gy); a[2] = dot(wy, dMu.y * gy);
                a[3] = dot(wx, r); a[4] = dot(wy, r);
            }
        }
        reduce5(a);
        const float axx = a[0] + 8.0, axy = a[1], ayy = a[2] + 8.0;
        const float det = axx * ayy - axy * axy;
        const vec2 d = clamp(vec2(ayy * a[3] - axy * a[4], axx * a[4] - axy * a[3]) / det, vec2(-1.0), vec2(1.0));
        const vec2 vn = v + d;
        const float cn = costAt(vn);
        if (!(cn < cost)) break;
        v = vn;
        cost = cn;
    }
    return v;
}

// the vector of a parent-level block (level l + 1, slot pfin) as an integer vector at this level (x2, rounded)
ivec2 parentVec(const V3Lv Lp, int pfin, ivec2 pb) { return ivec2(roundEven(2.0 * recXY(v3BlkIdx(Lp, dir, pfin, pb)))); }

// the stats of v3_stats.comp (G.dom, G.aff; zero and invalid until it has run, so levels at and above LH see no candidates): dominant motion k and the
// camera model, as an integer vector of this level (x 2^-l, F10 negated), valid bit in the second result
bool domValid(int k) { return (ax[gOff() + G_VALID] & (1u << k)) != 0u; }
ivec2 domVec(int k, int l) {
    const int g = gOff() + G_DOM + 2 * k;
    const vec2 d = vec2(uintBitsToFloat(ax[g]), uintBitsToFloat(ax[g + 1]));
    return ivec2(roundEven((dir == 0 ? d : -d) * (1.0 / float(1 << l))));
}
bool camValid() { return (ax[gOff() + G_VALID] & 256u) != 0u; }
vec2 camAff(vec2 x) {   // the affine model at level-0 flow px x
    const int g = gOff() + G_AFF;
    const vec2 XY = (x - vec2(pc.cx, pc.cy)) / vec2(pc.cx, pc.cy);
    const vec3 a = vec3(uintBitsToFloat(ax[g]), uintBitsToFloat(ax[g + 1]), uintBitsToFloat(ax[g + 2]));
    const vec3 b = vec3(uintBitsToFloat(ax[g + 3]), uintBitsToFloat(ax[g + 4]), uintBitsToFloat(ax[g + 5]));
    return vec2(dot(a, vec3(1.0, XY)), dot(b, vec3(1.0, XY)));
}
// the camera model at the centre c (level-0 flow px) of this block: F01 d(c); F10 -d(c - d(c)) (the content at curr position c came from c + F10 = c - d)
ivec2 camVec(vec2 c, int l) {
    vec2 d = camAff(c);
    if (dir == 1) d = -camAff(c - d);
    return ivec2(roundEven(d * (1.0 / float(1 << l))));
}
vec2 blkCentre0(int l) { return l == 0 ? vec2(blk * 8 + 4) : vec2(blk * 4 + 2) * float(1 << l); }

void main() {
    tid = int(gl_LocalInvocationIndex);
    dir = int(gl_WorkGroupID.z);
    f0 = dir; f1 = 1 - dir;
    blk = ivec2(gl_WorkGroupID.xy);
    const int l = pc.lvl;
    L = v3Lv(l);
    int P, Bw, nLev;
    v3Totals(P, Bw, nLev);
    fOff = 2 * P;
    dMu = v3DMu(dir);
    // the window of the role-0 frame: level >= 1 the 4x4 block with a 2 px apron, level 0 the 8x8 block
    wo = l == 0 ? blk * 8 : blk * 4 - 2;
    {
        if (tid < 64) {
            const ivec2 x = wo + ivec2(tid & 7, tid >> 3);
            const bool in_ = inside(x);
            sval[tid] = in_ ? 1 : 0;
            sf[tid] = in_ ? v3FeatAt(L, fOff, f0, x) : vec4(0.0);
        }
        nWin = (min(wo.x + 8, L.w) - max(wo.x, 0)) * (min(wo.y + 8, L.h) - max(wo.y, 0));
        if (tid == 0) blkDetU = 0u;
#if V3J
        // hook (j): the weights, once per block, from the role-0 frame's raw planes; the centre colour = the mean of the 4 pixels at the middle of the window
        // (the block's centre: window pixels 3..4 in x and y), clamped into the level
        if (tid < 64) {
            const ivec2 x = wo + ivec2(tid & 7, tid >> 3);
            const ivec2 cc = wo + 3;
            const vec3 cen = 0.25 * (v3RawC(L, f0, cc) + v3RawC(L, f0, cc + ivec2(1, 0)) + v3RawC(L, f0, cc + ivec2(0, 1)) + v3RawC(L, f0, cc + 1));
            sw[tid] = inside(x) ? v3ColourWgt(v3RawAt(L, f0, x), cen) : 0.0;
        }
#endif
    }
    barrier();
    if (pc.hudB > 0.0 && tid < 64 && sval[tid] != 0) atomicAdd(blkDetU, uint(v3Detail(sf[tid]) * 256.0 + 0.5));   // task 105: read after evalRound's first barrier

    ivec2 bv = ivec2(0);
    float bTot = 0.0, bPure = 0.0, amb = 0.0, zc0 = 0.0;
    int slot = 0;
#if V3S_TOP
    {
        // exhaustive +-8: one thread per candidate, 2 per thread; a tiny bias to the shorter vector settles ties (flat blocks) at zero
        for (int i = tid; i < 24 * 24; i += WG) {   // the region wo - 8 .. wo + 15 of the other frame, once, into shared memory
            const ivec2 t = wo - 8 + ivec2(i % 24, i / 24);
            spatch[i] = inside(t) ? pyr[fOff + 2 * L.off + f1 * L.w * L.h + t.y * L.w + t.x] : uvec2(0u);
        }
        barrier();
        for (int k = 0; k < 2; k++) {
            const int idx = tid + WG * k;
            if (idx >= 289) break;
            const ivec2 v = ivec2(idx % 17 - 8, idx / 17 - 8);
            float s = 0.0, n = 0.0;
#if V3J
            float wsum = 0.0;
#endif
            for (int i = 0; i < 64; i++) {
                const ivec2 d = ivec2(i & 7, i >> 3);
                const float ok = (sval[i] != 0 && inside(wo + d + v)) ? 1.0 : 0.0;
                const uvec2 pk = spatch[(d.y + v.y + 8) * 24 + d.x + v.x + 8];
#if V3J
                s += ok * sw[i] * v3Term(sf[i], vec4(unpackHalf2x16(pk.x), unpackHalf2x16(pk.y)), dMu);
                wsum += ok * sw[i];
#else
                s += ok * v3Term(sf[i], vec4(unpackHalf2x16(pk.x), unpackHalf2x16(pk.y)), dMu);
#endif
                n += ok;
            }
#if V3J
            topc[idx] = (n * 2.0 >= float(nWin) && n > 0.0 && wsum > 0.0) ? s / wsum + 0.002 * float(abs(v.x) + abs(v.y)) : BAD;
#else
            topc[idx] = (n * 2.0 >= float(nWin) && n > 0.0) ? s / n + 0.002 * float(abs(v.x) + abs(v.y)) : BAD;
#endif
        }
        barrier();
        if (tid < 64) {   // each of 64 threads takes the best of its 5 candidates (idx = tid + 64 k)
            float bm = 1e9;
            int bi = 0;
            for (int k = 0; k < 5; k++) {
                const int idx = tid + 64 * k;
                if (idx < 289 && topc[idx] < bm) { bm = topc[idx]; bi = idx; }
            }
            topp[tid] = bm;
            topi[tid] = bi;
        }
        barrier();
        float gm = topp[0];
        int gi = topi[0];
        for (int i = 1; i < 64; i++)
            if (topp[i] < gm || (topp[i] == gm && topi[i] < gi)) { gm = topp[i]; gi = topi[i]; }
        const ivec2 v0 = ivec2(gi % 17 - 8, gi / 17 - 8);
        // the second best among the candidates at least 3 px from the best (the top level's ambiguity)
        float sm = 1e9;
        if (tid < 64)
            for (int k = 0; k < 5; k++) {
                const int idx = tid + 64 * k;
                if (idx >= 289) break;
                const ivec2 d = abs(ivec2(idx % 17 - 8, idx / 17 - 8) - v0);
                if (max(d.x, d.y) >= 3) sm = min(sm, topc[idx]);
            }
        barrier();   // (everyone has read topp)
        if (tid < 64) topp[tid] = sm;
        barrier();
        float gs = topp[0];
        for (int i = 1; i < 64; i++) gs = min(gs, topp[i]);
        bv = v0;
        bPure = gm >= BAD ? BAD : gm - 0.002 * float(abs(v0.x) + abs(v0.y));
        amb = 1.0 - smoothstep(0.5, 3.0, gs - gm);
        barrier();
    }
#else
    if (l >= 1 && pc.iter == 0) {
        const V3Lv Lp = v3Lv(l + 1);
#ifdef V35
        const int pfin = (l + 1 == nLev - 1) ? 0 : (pc.mode & 2) != 0 ? 0 : 1;   // V3.5: bit 1 of the mode = the parent level's regularisation ended in slot 0
#else
        const int pfin = (l + 1 == nLev - 1) ? 0 : 1;
#endif
        const ivec2 pb = clamp(blk >> 1, ivec2(0), ivec2(Lp.gw - 1, Lp.gh - 1));
        const ivec2 sg = ivec2((blk.x & 1) != 0 ? 1 : -1, (blk.y & 1) != 0 ? 1 : -1);
        if (tid == 0) refV = parentVec(Lp, pfin, pb);
        if (tid < 16) {
            ivec2 v = ivec2(0);
            int fg = 0;
            if (tid == 0) { v = parentVec(Lp, pfin, pb); fg = 3; }
            else if (tid == 1) { v = parentVec(Lp, pfin, pb + ivec2(sg.x, 0)); fg = 1; }
            else if (tid == 2) { v = parentVec(Lp, pfin, pb + ivec2(0, sg.y)); fg = 1; }
            else if (tid == 3) { v = parentVec(Lp, pfin, pb + sg); fg = 1; }
            else if (tid == 4) { fg = 1; }
            else if (tid == 5) { if (camValid()) { v = camVec(blkCentre0(l), l); fg = 1; } }
            else if (tid >= 6 && tid <= 9) { if (domValid(tid - 6)) { v = domVec(tid - 6, l); fg = 5; } }
            else if (tid == 10) {   // the reverse vector of the other direction at the parent's target
                const vec2 target = vec2(2 * blk + 1) + 0.5 * vec2(parentVec(Lp, pfin, pb));   // block centre in level l + 1 px, plus the parent's vector
                const ivec2 tb = ivec2(floor(target * 0.25));
                v = ivec2(roundEven(-2.0 * recXY(v3BlkIdx(Lp, 1 - dir, pfin, tb)))); fg = 1;
            }
            cv[tid] = v;
            cflag[tid] = fg;
        }
        barrier();
        evalRound();
        best(16, bv, bTot, bPure);
        barrier();
        ring(bv, bTot, bPure);
        slot = 0;
    } else if (l >= 1) {   // iteration 1
        const vec2 v0 = recXY(v3BlkIdx(L, dir, 0, blk));
        if (tid == 0) refV = ivec2(roundEven(v0));
        if (tid < 16) {
            ivec2 v = ivec2(0);
            int fg = 0;
            if (tid == 0) { v = ivec2(roundEven(v0)); fg = 3; }
            else if (tid <= 4) {
                const ivec2 o = tid == 1 ? ivec2(-1, 0) : tid == 2 ? ivec2(1, 0) : tid == 3 ? ivec2(0, -1) : ivec2(0, 1);
                v = ivec2(roundEven(recXY(v3BlkIdx(L, dir, 0, blk + o)))); fg = 1;
            } else if (tid == 5) {   // the other direction's reverse vector at the target of the iteration-0 result (bilinear over its block records)
                const vec2 u = (vec2(4 * blk + 2) + v0 - 2.0) * 0.25;
                const ivec2 i0 = ivec2(floor(u));
                const vec2 a = u - vec2(i0);
                const vec2 o = mix(mix(recXY(v3BlkIdx(L, 1 - dir, 0, i0)), recXY(v3BlkIdx(L, 1 - dir, 0, i0 + ivec2(1, 0))), a.x),
                                   mix(recXY(v3BlkIdx(L, 1 - dir, 0, i0 + ivec2(0, 1))), recXY(v3BlkIdx(L, 1 - dir, 0, i0 + ivec2(1, 1))), a.x), a.y);
                v = ivec2(roundEven(-o)); fg = 1;
            } else if (tid == 6) { fg = 1; }
            else if (tid >= 7 && tid <= 9) { if (domValid(tid - 7)) { v = domVec(tid - 7, l); fg = 5; } }
            else if (tid == 10) { if (camValid()) { v = camVec(blkCentre0(l), l); fg = 1; } }
            else if (tid == 11) { if (domValid(3)) { v = domVec(3, l); fg = 5; } }
            cv[tid] = v;
            cflag[tid] = fg;
        }
        barrier();
        evalRound();
        best(16, bv, bTot, bPure);
        barrier();
        float pmx = BAD, pmy = BAD;
        ring(bv, bTot, bPure, l == 1, pmx, pmy);
        slot = 1;
        if (l == 1) {   // the ambiguity probes (section 3.5, As built E2): amb from how much worse the best probe along each axis is than the final vector
            // the structure of the window along x and y: the mean weighted absolute difference of neighbouring high-pass features. A probe along an axis along which the
            // window has no structure (a line pattern shifted along its lines: the aperture problem, or a flat wall) matches as well as the best by construction: it
            // says nothing, so that axis counts with the weight smoothstep(0.6, 1.4, structure)
            float ea[5] = float[5](0.0, 0.0, 0.0, 0.0, 0.0);
            if (tid < 64 && sval[tid] != 0) {
                const int ix = tid & 7, iy = tid >> 3;
                if (ix < 7 && sval[tid + 1] != 0) { ea[0] = dot(abs(sf[tid].xyz - sf[tid + 1].xyz), V3_CW.xyz); ea[2] = 1.0; }
                if (iy < 7 && sval[tid + 8] != 0) { ea[1] = dot(abs(sf[tid].xyz - sf[tid + 8].xyz), V3_CW.xyz); ea[3] = 1.0; }
            }
            reduce5(ea);
            const vec2 gs = vec2(ea[0] / max(ea[2], 1.0), ea[1] / max(ea[3], 1.0));
            const float ax0 = pmx < BAD ? (1.0 - smoothstep(0.12, 0.40, (pmx - bPure) / max(pmx, 0.5))) * smoothstep(0.6, 1.4, gs.x) : 0.0;
            const float ay0 = pmy < BAD ? (1.0 - smoothstep(0.12, 0.40, (pmy - bPure) / max(pmy, 0.5))) * smoothstep(0.6, 1.4, gs.y) : 0.0;
            amb = max(ax0, ay0);
        }
    } else {   // level 0 below a coarser level
        const V3Lv Lp = v3Lv(1);
#ifdef V35
        const int pfin = nLev == 2 ? 0 : (pc.mode & 2) != 0 ? 0 : 1;
#else
        const int pfin = nLev == 2 ? 0 : 1;
#endif
        const ivec2 pb = clamp(blk, ivec2(0), ivec2(Lp.gw - 1, Lp.gh - 1));
        if (tid == 0) refV = parentVec(Lp, pfin, pb);
        if (tid < 16) {
            ivec2 v = ivec2(0);
            int fg = 0;
            if (tid <= 4) {
                const ivec2 o = tid == 0 ? ivec2(0) : tid == 1 ? ivec2(-1, 0) : tid == 2 ? ivec2(1, 0) : tid == 3 ? ivec2(0, -1) : ivec2(0, 1);
                v = parentVec(Lp, pfin, pb + o);
                fg = tid == 0 ? 3 : 1;
            } else if (tid == 5 && pc.hudB > 0.0) { fg = 9; }   // task 105: the zero vector, measured only (cflag bit 3: never chosen; its cost is the block's evidence for the dense pass). As a candidate it let a static inpainted patch pull the blocks around it to zero (uitest, CrosshairV2)
            cv[tid] = v;
            cflag[tid] = fg;
        }
        barrier();
        evalRound();
        best(16, bv, bTot, bPure);
        if (tid == 0 && pc.hudB > 0.0)   // task 105: the block's evidence of a still part (detail x the zero vector's pure cost, candidate slot 5, on ramps 3x as wide as the bonus's), for the dense pass's pre-test
            zc0 = v3HudEv(float(blkDetU) * (1.0 / 256.0) / float(max(nWin, 1))) * (1.0 - smoothstep(3.0 * pc.hp[0], 3.0 * pc.hp[1], cpur[5]));
        barrier();
        ring(bv, bTot, bPure);
        amb = fl[v3BlkIdx(Lp, dir, pfin, pb) + 3];
    }
#endif
    vec2 vOut = vec2(bv);
    if (l == 0) vOut = lk(bv, bPure);   // (the whole workgroup: uniform)
    if (tid == 0) {
        const int o = v3BlkIdx(L, dir, slot, blk);
        fl[o] = vOut.x; fl[o + 1] = vOut.y; fl[o + 2] = min(bPure, 255.0); fl[o + 3] = amb;
        if (l == 0 && pc.hudB > 0.0 && V3S_TOP == 0) fl[v3BlkIdx(L, dir, 1, blk) + 2] = zc0;   // (level 0 has no slot 1 otherwise)
    }
}
