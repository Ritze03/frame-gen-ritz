// V2 Quality: coverage of the cleaned block field (which real frame shows the content of each midpoint location), shared by
// v2_qown.comp, v2_qcov.comp (once per estimate) and v2_qmorph.comp (the gather). Included after common.glsl and v2_motion.glsl.
//
// The field lives on the midpoint grid: the content at midpoint location m is prev(m - h) = curr(m + h). Where something is covered
// or revealed between the two real frames, one of those two taps shows another object: the one in front, whose own midpoint location
// maps to the same pixel with a vector that matches. So every midpoint location (flow resolution) is splatted to both real frames
// (v2_qown: the footprint pixel of each frame keeps the best-matching claim, atomicMin of key | vector), and a location still sees
// its content in a frame if it won its footprint there, or nearly (its own match is within Q_MARGIN of the winner's), or the winner
// moves the same way (within Q_TOL): v2_qcov. A footprint outside the frame does not see it (content entering / leaving the view).
// The two flags, box-filtered over (2 Q_R + 1)^2 flow pixels for soft ramps, are the coverage (cP, cC) of the location; the gather
// leans each output pixel to the frame that still shows it (qWeightC).
//
// Binding 14: V2's Quality buffer (MotionGroup::bufs[V2B_QUAL]), uint words, N = fw * fh:
//   [0, N) prev's claims, [N, 2N) curr's claims (packed: key << 20 | (vx + 512) << 10 | (vy + 512), v in quarter flow px; reset to
//   0xFFFFFFFF before the splat), [2N, 3N) each location's own packed claim, [3N, 4N) the coverage packUnorm2x16(cP, cC).
layout(set = 0, binding = 14, std430) buffer V2Q { uint qb[]; };

const float Q_KEY = 8.0;      // key units per luma level (12 bits: up to 511 levels)
const uint Q_MARGIN = 16u;    // 2 luma levels
const float Q_TOL = 0.75;     // flow px of half-vector (max-abs): "moves the same way"
const int Q_R = 2;            // box radius of the coverage (flow px)
const float Q_EPS = 0.03;
const float Q_SGAIN = 0.5;    // qStructure: weight of the local contrast against the zero vector's own mismatch     // floor of a side's weight: both sides hidden -> the plain (1 - t, t) blend

const float Q_FSAME = 0.5;   // v2_qfield: block vectors within this (flow px, max-abs) count as one motion

int qN() { return pc.fw * pc.fh; }
int qPF() { return 2 * pc.bw * pc.bh; }   // the per-pixel field in vf[] (v2_qfield.comp), after the cleaned and the fitted block fields
vec2 qF(ivec2 m) {
    m = clamp(m, ivec2(0), ivec2(pc.fw - 1, pc.fh - 1));
    return vf[qPF() + m.y * pc.fw + m.x];
}
// The per-pixel field (full-res px of half-vector) at full-resolution position x: bilinear (halfVec of motion_v1.glsl)
vec2 hvQ(vec2 x) {
    float S = float(pc.fscale);
    vec2 s = (x + 0.5) * (1.0 / S) - 0.5;
    vec2 fl = floor(s);
    vec2 f = s - fl;
    ivec2 i = ivec2(fl);
    return S * mix(mix(qF(i), qF(i + ivec2(1, 0)), f.x), mix(qF(i + ivec2(0, 1)), qF(i + ivec2(1, 1)), f.x), f.y);
}

// The block field (flow px of half-vector) bilinear at flow position m (pixel-centre coordinates): hvAt at flow resolution.
vec2 hvFlow(vec2 m) {
    vec2 s = (m + 0.5) * (1.0 / float(V2_BLOCK)) - 0.5;
    vec2 fl = floor(s);
    vec2 f = s - fl;
    ivec2 i = ivec2(fl);
    return mix(mix(VF(i), VF(i + ivec2(1, 0)), f.x), mix(VF(i + ivec2(0, 1)), VF(i + ivec2(1, 1)), f.x), f.y);
}

// Flow-resolution luma of prev (frame a) / curr (frame c), bilinear, clamp to edge (level 0 of buf[0]: a at 0, c at fw * fh).
float qL(int off, ivec2 i) {
    i = clamp(i, ivec2(0), ivec2(pc.fw - 1, pc.fh - 1));
    return luma[off + i.y * pc.fw + i.x];
}
float qLuma(int off, vec2 p) {
    vec2 fl = floor(p);
    vec2 f = p - fl;
    ivec2 i = ivec2(fl);
    return mix(mix(qL(off, i), qL(off, i + ivec2(1, 0)), f.x), mix(qL(off, i + ivec2(0, 1)), qL(off, i + ivec2(1, 1)), f.x), f.y);
}

ivec2 qQuant(vec2 h) { return clamp(ivec2(roundEven(h * 4.0)), ivec2(-512), ivec2(511)); }
uint qPack(uint key, ivec2 v) { return (min(key, 4095u) << 20) | (uint(v.x + 512) << 10) | uint(v.y + 512); }
// the coverage word: cP, cC (11 bits each), the static-overlay mask z (10 bits)
uint qPackCov(vec2 c, float z) {
    uvec2 q = uvec2(roundEven(clamp(c, 0.0, 1.0) * 2047.0));
    return q.x | (q.y << 11) | (uint(roundEven(clamp(z, 0.0, 1.0) * 1023.0)) << 22);
}
vec3 qUnpackCov(uint w) { return vec3(float(w & 2047u) * (1.0 / 2047.0), float((w >> 11) & 2047u) * (1.0 / 2047.0), float(w >> 22) * (1.0 / 1023.0)); }
vec2 qVec(uint w) { return vec2(ivec2(int((w >> 10) & 1023u), int(w & 1023u)) - 512) * 0.25; }
// the footprint pixel of p (flow coordinates), -1 if outside the frame
int qPix(vec2 p) {
    ivec2 i = ivec2(floor(p + 0.5));
    return all(greaterThanEqual(i, ivec2(0))) && all(lessThan(i, ivec2(pc.fw, pc.fh))) ? i.y * pc.fw + i.x : -1;
}

#ifdef V2_QUALITY
// Local structure at full-res pixel p: the largest max-channel difference of prev's pixel to its right and lower neighbours (8-bit
// levels), scaled by pc.cnt... (TEMP: knob in pc.globalFrac? no: constant)
float qStructure(ivec2 p) {
    ivec2 mx = ivec2(pc.W - 1, pc.H - 1);
    vec3 c = FG_FETCH(texelFetch(texPrev, p, 0).rgb);
    vec3 r = FG_FETCH(texelFetch(texPrev, min(p + ivec2(1, 0), mx), 0).rgb);
    vec3 d = FG_FETCH(texelFetch(texPrev, min(p + ivec2(0, 1), mx), 0).rgb);
    vec3 a = max(abs(c - r), abs(c - d));
    return Q_SGAIN * max(a.r, max(a.g, a.b));
}

// The static-overlay mask (v2_qfield.comp) at full-resolution position q, bilinear
float qZeroMask(vec2 q) {
    vec2 m = clamp((q + 0.5) * (1.0 / float(pc.fscale)) - 0.5, vec2(0.0), vec2(pc.fw - 1, pc.fh - 1));
    vec2 fl = floor(m);
    vec2 f = m - fl;
    ivec2 i = ivec2(fl);
    ivec2 j = min(i + 1, ivec2(pc.fw - 1, pc.fh - 1));
    int o = 3 * qN();
    float z00 = float(qb[o + i.y * pc.fw + i.x] >> 22), z01 = float(qb[o + i.y * pc.fw + j.x] >> 22);
    float z10 = float(qb[o + j.y * pc.fw + i.x] >> 22), z11 = float(qb[o + j.y * pc.fw + j.x] >> 22);
    return mix(mix(z00, z01, f.x), mix(z10, z11, f.x), f.y) * (1.0 / 1023.0);
}

// The gather's share of curr at output pixel q (full resolution) at time t, g the half-vector there (full-res px): the coverage of
// the midpoint location the content comes from (q - (2t - 1) g), bilinear.
float qWeightC(vec2 q, vec2 g, float t) {
    float S = float(pc.fscale);
    vec2 m = (q - (2.0 * t - 1.0) * g + 0.5) * (1.0 / S) - 0.5;
    m = clamp(m, vec2(0.0), vec2(pc.fw - 1, pc.fh - 1));
    vec2 fl = floor(m);
    vec2 f = m - fl;
    ivec2 i = ivec2(fl);
    ivec2 j = min(i + 1, ivec2(pc.fw - 1, pc.fh - 1));
    int o = 3 * qN();
    vec3 c00 = qUnpackCov(qb[o + i.y * pc.fw + i.x]), c01 = qUnpackCov(qb[o + i.y * pc.fw + j.x]);
    vec3 c10 = qUnpackCov(qb[o + j.y * pc.fw + i.x]), c11 = qUnpackCov(qb[o + j.y * pc.fw + j.x]);
    vec3 cv = mix(mix(c00, c01, f.x), mix(c10, c11, f.x), f.y);
    float a = (1.0 - t) * (cv.x + pc.sigma2Inv), c = t * (cv.y + pc.sigma2Inv);
    return c / (a + c);
}
#endif
