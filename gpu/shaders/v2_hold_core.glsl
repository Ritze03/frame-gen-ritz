// V2: the soft hold edge (I11, D18), one map per synthesised t, shared by v2_hold.comp (recordSynth) and v2_blur_hold.comp (each
// sub-frame of recordSynthBlur). V1's held regions end in a hard ~4 px edge (the 7x7 box of e and the trust ramp): the "too sharp
// islands", a patch of the unwarped nearer frame |v| px off from the warped content around it. Here the frame is cut into 4 x 4 px
// cells (cw x ch = ceil(W / 4) x ceil(H / 4)); a cell whose mean known e is past the middle of the trust ramp is a hold source of
// weight hc = 2 ramp - 1, and around it the hold weight falls off linearly over F px (V2Knobs holdFeather) along x and along y (a
// product of the two: a rounded square): uf(c) = max over the sources k of hc_k (1 - |dx| 4 / F)+ (1 - |dy| 4 / F)+. The held region
// itself keeps its full hold, only its outside gets the soft edge. v2_resolve_core.glsl takes max(Ut, uf) (uf bilinear between the
// cell centres). One workgroup per 16 x 16 cells (+ the reach R = ceil(F / 4) <= 8 cells around them in shared memory).
// The includer defines:
//   HOLD_EW(p)        the packed word of e holding full-resolution pixel p (p.x a multiple of 4; low byte first)
//   HOLD_OUT(c, v)    stores uf v of cell c
//   HOLD_W, HOLD_H    the full resolution; HOLD_F the feather (px, > 0); HOLD_T0, HOLD_TS the trust ramp (trust0, 1 / width)
const int HC = 4, HT = 16, HRMAX = 8, HN = HT + 2 * HRMAX;
shared float hsrc[HN * HN];
shared float hrow[HN * HT];

void holdMain() {
    int lid = int(gl_LocalInvocationIndex);
    float F = HOLD_F;
    int R = min(int(ceil(F / float(HC))), HRMAX);
    int n = HT + 2 * R;
    ivec2 cw = ivec2((HOLD_W + HC - 1) / HC, (HOLD_H + HC - 1) / HC);
    ivec2 c0 = ivec2(gl_WorkGroupID.xy) * HT - R;
    for (int i = lid; i < n * n; i += HT * HT) {
        ivec2 c = c0 + ivec2(i % n, i / n);
        float se = 0.0, ke = 0.0;
        if (all(greaterThanEqual(c, ivec2(0))) && all(lessThan(c, cw)))
            for (int j = 0; j < HC; j++) {
                int y = c.y * HC + j;
                if (y >= HOLD_H) break;
                uint w = HOLD_EW(ivec2(c.x * HC, y));
                for (int b = 0; b < 4; b++) {
                    uint e = (w >> (8 * b)) & 255u;
                    if (e != 255u) { se += float(e); ke += 1.0; }   // (255: e unknown, a tap on the UI)
                }
            }
        float rr = ke > 0.0 ? clamp((se / ke - HOLD_T0) * HOLD_TS, 0.0, 1.0) : 0.0;
        hsrc[i] = clamp(2.0 * rr - 1.0, 0.0, 1.0);
    }
    barrier();
    float k = float(HC) / F;
    for (int i = lid; i < n * HT; i += HT * HT) {   // along x
        int row = i / HT, col = i % HT;
        float mx = 0.0;
        for (int d = -R; d <= R; d++) mx = max(mx, hsrc[row * n + col + R + d] * max(1.0 - float(abs(d)) * k, 0.0));
        hrow[row * HT + col] = mx;
    }
    barrier();
    ivec2 l = ivec2(gl_LocalInvocationID.xy);
    ivec2 c = ivec2(gl_WorkGroupID.xy) * HT + l;
    if (any(greaterThanEqual(c, cw))) return;
    float mx = 0.0;   // along y
    for (int d = -R; d <= R; d++) mx = max(mx, hrow[(l.y + R + d) * HT + l.x] * max(1.0 - float(abs(d)) * k, 0.0));
    HOLD_OUT(c, mx);
}
