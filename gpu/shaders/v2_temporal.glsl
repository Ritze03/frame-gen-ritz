// V2, the temporal store (D19): the previous pair's field, kept for the next estimate. Included after v2_pack.glsl by the passes that
// use it (v2_seed, v2_search, v2_smooth, v2_occl), with either push-constant block (motion_v1.glsl's PC and motion_v2.glsl's PC2 both
// have W, H, fw, fh) and with v2b[] (binding 13) and u[] declared.
//
// Layout in v2b[] (words), after the occlusion map (4 fw fh), the class map (E_STRIDE H) and the content rescue's scratch (3 fw fh,
// v2_rescue.comp: rOff = 4 fw fh + ((W + 3) / 4) H; [rOff, + fw fh) the coverage bits (0: a shown, 1: c shown, 2:
// rescued by v2_merge, 3: filled by v2_close), [rOff + fw fh, + fw fh) the rescue keys,
// [rOff + 2 fw fh, + fw fh) the rescued vectors per source pixel; I12: the keys' words, dead after v2_merge.comp, then hold the
// occlusion gate's per-pixel statistics (v2_occl -> v2_gate.comp) and, before them, the rescue check's (v2_rcheck_e -> v2_rcheck.comp):
// v2GOff()) and the soft hold edge's map of the latest synth (v2_hold.comp,
// hOff = rOff + 3 fw fh, one float per 4 x 4 px cell: ceil(W / 4) x ceil(H / 4)): tOff = hOff + ceil(W / 4) ceil(H / 4),
//   [tOff + s fw fh, + fw fh)    s = 0, 1: the edge-aware field of an estimate (packed, v2_pack.glsl, flow units), ping-pong: the
//                                estimate of parity p (PC2 tFlags bit 0) reads slot 1 - p (the previous pair's) and writes slot p
//                                (v2_occl.comp)
//   [tOff + 2 fw fh, + fw fh)    vT: the previous field motion-compensated to this pair's midpoint (v2_seed.comp, packed); dead
//                                after v2_rescue.comp: I12's v2_merge / v2_close.comp keep the edge-aware vector of each pixel they
//                                rescue there, for v2_rcheck.comp to put back
//   [tOff + 3 fw fh + s n, + n)  s = 0, 1: the coarsest pyramid level of curr (float bits, n = its pixel count), same ping-pong:
//                                the next estimate compares its prev with it (is this pair the continuation of the last one?)
// u[V2_U_SEED] (cleared with the estimate's counters): v2_seed.comp's verdict, the mean |prev - last curr| of the coarsest level
// x 2^20, + 2^31 if the host had no valid previous estimate (framegen.cpp v2SeedOk; or Settings::v2Temporal off). The seed is valid
// below V2_SEED_MAX (0.25 levels: the same frame gives exactly 0).
const int V2_U_SEED = 30;
const int V2_U_VOTES = 29;     // (I11) votes of the content rescue (v2_rescue.comp; cleared with the estimate's counters)
const int V2_U_RESCUED = 31;   // (I11) pixels the content rescue put back (v2_merge.comp; cleared with the estimate's counters)
const uint V2_SEED_MAX = 262144u;
bool v2SeedValid() { return u[V2_U_SEED] < V2_SEED_MAX; }
int v2ROff() { return 4 * pc.fw * pc.fh + ((pc.W + 3) >> 2) * pc.H; }   // the content rescue's scratch (v2_rescue.comp)
// (I12) the occlusion gate's per-pixel statistics (v2_occl.comp -> v2_gate.comp): the content rescue's keys, dead after v2_merge.comp
int v2GOff() { return v2ROff() + pc.fw * pc.fh; }
int v2HOff() { return v2ROff() + 3 * pc.fw * pc.fh; }   // the soft hold edge's map (v2_hold.comp), ceil(W / 4) x ceil(H / 4)
int v2TOff() { return v2HOff() + ((pc.W + 3) >> 2) * ((pc.H + 3) >> 2); }
int v2POff(int slot) { return v2TOff() + slot * pc.fw * pc.fh; }
int v2VTOff() { return v2TOff() + 2 * pc.fw * pc.fh; }
int v2LOff(int slot, int n) { return v2TOff() + 3 * pc.fw * pc.fh + slot * n; }
// the compensated previous field at flow pixel q (clamped), flow-resolution half-vector units
vec2 v2VT(ivec2 q) {
    q = clamp(q, ivec2(0), ivec2(pc.fw - 1, pc.fh - 1));
    return unpackV(v2b[v2VTOff() + q.y * pc.fw + q.x]);
}
// the previous field (slot) at the nearest flow pixel of x
vec2 v2PAt(int slot, vec2 x) {
    ivec2 q = clamp(ivec2(roundEven(x)), ivec2(0), ivec2(pc.fw - 1, pc.fh - 1));
    return unpackV(v2b[v2POff(slot) + q.y * pc.fw + q.x]);
}
