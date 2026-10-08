// The body of search.comp (V1, and V2 without its temporal seed) and v2_search.comp (V2_SEED: the previous pair's field as a
// 7th seed candidate; it must beat the others by SEED_MARGIN, I12). Included after the includes and the layout of the pass.

const int MAXR = 6, MAXT = 8 + 2 * MAXR, MAXC = (2 * MAXR + 1) * (2 * MAXR + 1);
shared float ta[MAXT * MAXT], tc[MAXT * MAXT];
#ifdef V2_SEED
const int NSEED = 7;   // + the temporal seed (v2_search.comp)
#else
const int NSEED = 6;
#endif
shared float red[NSEED][64];
shared float cost[MAXC], sadc[MAXC];
shared ivec2 seedS;

void main() {
    int lid = int(gl_LocalInvocationIndex);
    ivec2 blk = ivec2(gl_WorkGroupID.xy);
    ivec2 b0 = blk * BLOCK;
    ivec2 q = b0 + ivec2(lid & 7, lid >> 3);
#ifdef V2_SEED
    bool tv = v2SeedValid();
    ivec2 ts = v2SeedAt(blk);
#endif
    if (pc.medIdx < 0) {
#ifdef V2_SEED
        // coarsest level: zero (V1's seed), or the temporal seed if its SAD + LAMBDA is lower (both always measured: the same work
        // whether the seed is valid or not)
        red[0][lid] = abs(L(pc.offA, q.x, q.y, pc.w, pc.h) - L(pc.offC, q.x, q.y, pc.w, pc.h));
        red[1][lid] = abs(L(pc.offA, q.x - ts.x, q.y - ts.y, pc.w, pc.h) - L(pc.offC, q.x + ts.x, q.y + ts.y, pc.w, pc.h));
        barrier();
        for (int s = 32; s > 0; s >>= 1) {
            if (lid < s) { red[0][lid] += red[0][lid + s]; red[1][lid] += red[1][lid + s]; }
            barrier();
        }
        if (lid == 0) seedS = tv && red[1][0] * (1.0 / 64.0) + LAMBDA + SEED_MARGIN < red[0][0] * (1.0 / 64.0) ? ts : ivec2(0);
#else
        if (lid == 0) seedS = ivec2(0);
#endif
    } else {
        int px = min(blk.x / 2, pc.pw - 1), py = min(blk.y / 2, pc.ph - 1);
        int sy = (blk.y & 1) == 0 ? -1 : 1, sx = (blk.x & 1) == 0 ? -1 : 1;
        ivec2 cand[NSEED];
        ivec2 pp[4] = ivec2[4](ivec2(0, 0), ivec2(0, sy), ivec2(sx, 0), ivec2(sx, sy));
        for (int k = 0; k < 4; k++) {
            ivec2 c = clamp(ivec2(px, py) + pp[k], ivec2(0), ivec2(pc.pw - 1, pc.ph - 1));
            cand[k] = 2 * ivec2(roundEven(field[pc.fPar + c.y * pc.pw + c.x]));
        }
        cand[4] = ivec2(0);
        cand[5] = ivec2(roundEven(vec2(uintBitsToFloat(u[pc.medIdx]), uintBitsToFloat(u[pc.medIdx + 1]))));
#ifdef V2_SEED
        cand[6] = ts;
#endif
        for (int k = 0; k < NSEED; k++)
            red[k][lid] = abs(L(pc.offA, q.x - cand[k].x, q.y - cand[k].y, pc.w, pc.h)
                            - L(pc.offC, q.x + cand[k].x, q.y + cand[k].y, pc.w, pc.h));
        barrier();
        for (int s = 32; s > 0; s >>= 1) {
            if (lid < s)
                for (int k = 0; k < NSEED; k++) red[k][lid] += red[k][lid + s];
            barrier();
        }
        if (lid == 0) {
            int bi = 0;
            float best = red[0][0] * (1.0 / 64.0);
            for (int k = 1; k < 6; k++) {
                float c = red[k][0] * (1.0 / 64.0) + LAMBDA;
                if (c < best) { best = c; bi = k; }
            }
#ifdef V2_SEED
            if (tv && red[6][0] * (1.0 / 64.0) + LAMBDA + SEED_MARGIN < best) bi = 6;   // the temporal seed (after the others: ties keep
                                                                                   // V1's pick; I12: it must win by SEED_MARGIN)
#endif
            seedS = cand[bi];
        }
    }
    barrier();
    ivec2 s = seedS;
    int R = pc.radius, T = 8 + 2 * R, D = 2 * R + 1;
    for (int i = lid; i < T * T; i += 64) {
        int x = i % T, y = i / T;
        ta[i] = L(pc.offA, b0.x - s.x - R + x, b0.y - s.y - R + y, pc.w, pc.h);
        tc[i] = L(pc.offC, b0.x + s.x - R + x, b0.y + s.y - R + y, pc.w, pc.h);
    }
    barrier();
    for (int ci = lid; ci < D * D; ci += 64) {
        int dx = ci % D - R, dy = ci / D - R;
        float sum = 0.0;
        for (int y = 0; y < 8; y++)
            for (int x = 0; x < 8; x++)
                sum += abs(ta[(y - dy + R) * T + (x - dx + R)] - tc[(y + dy + R) * T + (x + dx + R)]);
        sadc[ci] = sum * (1.0 / 64.0);
        cost[ci] = sadc[ci] + LAMBDA * float(abs(dx) + abs(dy));
    }
    barrier();
    if (lid == 0) {
        int bi = 0;
        float best = cost[0];
        for (int ci = 1; ci < D * D; ci++)
            if (cost[ci] < best) { best = cost[ci]; bi = ci; }
        field[pc.fOut + blk.y * pc.bw + blk.x] = vec2(s + ivec2(bi % D - R, bi / D - R));
        if (pc.sadAcc != 0) atomicAdd(u[U_SADSUM], uint(roundEven(sadc[bi] * 256.0)));   // as subpix.comp
    }
}
