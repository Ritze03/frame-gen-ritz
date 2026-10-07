// V2, the temporal store (D19): the previous pair's field, kept for the next estimate. Included after v2_pack.glsl by the passes that
// use it (v2_seed, v2_search, v2_smooth, v2_occl), with either push-constant block (motion_v1.glsl's PC and motion_v2.glsl's PC2 both
// have W, H, fw, fh) and with v2b[] (binding 13) and u[] declared.
//
// Layout in v2b[] (words), after the occlusion map (4 fw fh) and the class map (E_STRIDE H): tOff = 4 fw fh + ((W + 3) / 4) H,
//   [tOff + s fw fh, + fw fh)    s = 0, 1: the edge-aware field of an estimate (packed, v2_pack.glsl, flow units), ping-pong: the
//                                estimate of parity p (PC2 tFlags bit 0) reads slot 1 - p (the previous pair's) and writes slot p
//                                (v2_occl.comp)
//   [tOff + 2 fw fh, + fw fh)    vT: the previous field motion-compensated to this pair's midpoint (v2_seed.comp, packed)
//   [tOff + 3 fw fh + s n, + n)  s = 0, 1: the coarsest pyramid level of curr (float bits, n = its pixel count), same ping-pong:
//                                the next estimate compares its prev with it (is this pair the continuation of the last one?)
// u[V2_U_SEED] (cleared with the estimate's counters): v2_seed.comp's verdict, the mean |prev - last curr| of the coarsest level
// x 2^20, + 2^31 if the host had no valid previous estimate (framegen.cpp v2SeedOk; or Settings::v2Temporal off). The seed is valid
// below V2_SEED_MAX (0.25 levels: the same frame gives exactly 0).
const int V2_U_SEED = 30;
const uint V2_SEED_MAX = 262144u;
bool v2SeedValid() { return u[V2_U_SEED] < V2_SEED_MAX; }
int v2TOff() { return 4 * pc.fw * pc.fh + ((pc.W + 3) >> 2) * pc.H; }
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
