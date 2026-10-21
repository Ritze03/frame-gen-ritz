// Motion version 2, V2Mode::Quality (the "coverage morph"): constants, buffer layout and helpers of its passes. Included after
// common.glsl and v2_motion.glsl. Per estimate, after Performance's re-pick and clean-up of the block field (v2_repick, v2_clean):
//   v2_qfield  the per-pixel field (flow resolution): the cleaned block field upsampled with edge awareness (each flow pixel blends the
//              block vectors around it by how well each matches there), and the static-overlay mask z (a pixel both real frames show
//              the same while the field moves: subtitle text, HUD), where the field is set to zero;
//   v2_qown    the coverage splat: every midpoint location claims its footprint pixel in prev (m - h) and in curr (m + h), the pixel
//              keeps the best-matching claim; a static-overlay pixel claims its own place in both frames;
//   v2_qcov    the coverage: does each frame still show the content of each midpoint location (cP, cC), packed with z.
// Per output (v2_qmorph = v2_morph.glsl with V2_QUALITY): one gather like Performance's, on the per-pixel field, whose blend leans to
// the frame that still shows the content (qWeightC) instead of the fixed (1 - t, t); a tap that lands on a static overlay while the
// field moves takes the other tap (qRejectStatic); where the two taps still disagree, the result is blurred along the motion; and the
// static-overlay mask shows the real frames' own pixels at the zero vector (the frames agree there, so this is no hold).
//
// Why coverage on the midpoint grid: the field is estimated at t = 0.5 (prev(m - h) = curr(m + h)). Where something is covered or
// revealed between the two real frames, one of those two taps shows another object, the one in front, whose own midpoint location
// maps to the same pixel with a vector that matches. A location still sees its content in a frame if it won its footprint there, or
// nearly (its own match is within Q_MARGIN of the winner's), or the winner moves the same way (within Q_TOL), or the winner matches
// badly itself (worse than Q_WIN: no evidence of an occluder, e.g. a HUD digit that changed); a footprint outside the frame does not
// see it (content entering / leaving the view). Computed once per estimate, so 3x / 4x and the motion blur's sub-frames read the same
// maps: the content at output q, time t comes from the midpoint location q - (2t - 1) g.
//
// Binding 14: V2's Quality buffer (MotionGroup::bufs[V2B_QUAL]), uint words, N = fw * fh:
//   [0, N) prev's claims, [N, 2N) curr's claims (key << 20 | (vx + 512) << 10 | (vy + 512), v in quarter flow px, key in 1/Q_KEY
//   luma levels; reset to 0xFFFFFFFF before the splat), [2N, 3N) each location's own claim, [3N, 4N) the coverage word: cP, cC
//   (11 bits each), z (10 bits) (v2_qfield writes z there as a float first, v2_qcov packs it).
// The per-pixel field: vf[qPF() + m] (binding 13, after the cleaned and the fitted block fields), flow px of half-vector.
layout(set = 0, binding = 14, std430) buffer V2Q { uint qb[]; };

// v2_qfield: the edge-aware upsample
const float Q_FSAME = 0.5;    // block vectors within this (flow px, max-abs) count as one motion: the plain bilinear blend
const float Q_FS = 2.0;       // softness of the candidate choice: weight exp(-(e - e_min) / (Q_FS + Q_FK * e_min)), e in luma levels
const float Q_FK = 1.0;       // ... wider when even the best candidate matches badly (a HUD digit that changed is no motion)
const float Q_FB = 0.1;       // floor of a candidate's spatial (bilinear) weight
// v2_qfield: the static-overlay mask
const float Q_ZMOVE = 1.0;    // only where a block vector around moves by more than this (flow px of half-vector)
const float Q_ZCON = 16.0;    // ... and the pixel has structure: luma range of its 3x3 (flow res) at least this
const float Q_ZR0 = 0.04, Q_ZR1 = 0.08;   // mean-removed zero-vector mismatch / that range: <= R0 -> 1, >= R1 -> 0
// v2_qown / v2_qcov: the coverage
const float Q_KEY = 8.0;      // key units per luma level (12 bits: up to 511 levels)
const float Q_MARGIN = 8.0;   // luma levels a location's own match may be worse than its footprint's winner and still see it
const float Q_WIN = 6.0;      // a winner that matches worse than this (luma levels) hides nothing
const float Q_TOL = 0.75;     // flow px of half-vector (max-abs): the winner "moves the same way"
const int Q_R = 2;            // box radius of the coverage (flow px): soft ramps
// v2_qmorph
const float Q_EPS = 0.03;     // floor of a side's coverage in the blend: both sides hidden -> the plain (1 - t, t) blend
const float Q_B0 = 16.0, Q_B1 = 24.0;   // blur along the motion where the taps differ by more than Q_B0 levels (full at Q_B0 + Q_B1)
const float Q_BLEN = 0.25;    // ... over +-Q_BLEN of the motion between the real frames
const float Q_SGAIN = 0.5;    // local contrast that counts as a disagreement of the taps for the zero-vector test (qStructure)

int qN() { return pc.fw * pc.fh; }
int qPF() { return 2 * pc.bw * pc.bh; }
vec2 qF(ivec2 m) {
    m = clamp(m, ivec2(0), ivec2(pc.fw - 1, pc.fh - 1));
    return vf[qPF() + m.y * pc.fw + m.x];
}
// The per-pixel field (full-res px of half-vector) at full-resolution position x: bilinear (motion_v1.glsl's halfVec)
vec2 hvQ(vec2 x) {
    float S = float(pc.fscale);
    vec2 s = (x + 0.5) * (1.0 / S) - 0.5;
    vec2 fl = floor(s);
    vec2 f = s - fl;
    ivec2 i = ivec2(fl);
    return S * mix(mix(qF(i), qF(i + ivec2(1, 0)), f.x), mix(qF(i + ivec2(0, 1)), qF(i + ivec2(1, 1)), f.x), f.y);
}

// Flow-resolution luma of prev (frame a) / curr (frame c), clamp to edge (level 0 of buf[0]: a at 0, c at fw * fh); bilinear qLuma.
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
vec2 qVec(uint w) { return vec2(ivec2(int((w >> 10) & 1023u), int(w & 1023u)) - 512) * 0.25; }
uint qPackCov(vec2 c, float z) {
    uvec2 q = uvec2(roundEven(clamp(c, 0.0, 1.0) * 2047.0));
    return q.x | (q.y << 11) | (uint(roundEven(clamp(z, 0.0, 1.0) * 1023.0)) << 22);
}
vec3 qUnpackCov(uint w) {
    return vec3(float(w & 2047u) * (1.0 / 2047.0), float((w >> 11) & 2047u) * (1.0 / 2047.0), float(w >> 22) * (1.0 / 1023.0));
}
// the footprint pixel of p (flow coordinates), -1 if outside the frame
int qPix(vec2 p) {
    ivec2 i = ivec2(floor(p + 0.5));
    return all(greaterThanEqual(i, ivec2(0))) && all(lessThan(i, ivec2(pc.fw, pc.fh))) ? i.y * pc.fw + i.x : -1;
}

#ifdef V2_QUALITY
// The coverage word bilinear at full-resolution position q: (cP, cC, z)
vec3 qCovAt(vec2 q) {
    vec2 m = clamp((q + 0.5) * (1.0 / float(pc.fscale)) - 0.5, vec2(0.0), vec2(pc.fw - 1, pc.fh - 1));
    vec2 fl = floor(m);
    vec2 f = m - fl;
    ivec2 i = ivec2(fl);
    ivec2 j = min(i + 1, ivec2(pc.fw - 1, pc.fh - 1));
    int o = 3 * qN();
    vec3 c00 = qUnpackCov(qb[o + i.y * pc.fw + i.x]), c01 = qUnpackCov(qb[o + i.y * pc.fw + j.x]);
    vec3 c10 = qUnpackCov(qb[o + j.y * pc.fw + i.x]), c11 = qUnpackCov(qb[o + j.y * pc.fw + j.x]);
    return mix(mix(c00, c01, f.x), mix(c10, c11, f.x), f.y);
}

// The gather's share of curr at time t for content coming from the midpoint location whose coverage is cv (qCovAt there)
float qWeightC(vec3 cv, float t) {
    float a = (1.0 - t) * (cv.x + Q_EPS), c = t * (cv.y + Q_EPS);
    return c / (a + c);
}

// The static-overlay mask at the flow pixel nearest to full-resolution position q
float qZAt(vec2 q) {
    ivec2 m = clamp(ivec2(floor((q + 0.5) * (1.0 / float(pc.fscale)))), ivec2(0), ivec2(pc.fw - 1, pc.fh - 1));
    return float(qb[3 * qN() + m.y * pc.fw + m.x] >> 22) * (1.0 / 1023.0);
}

// A tap on a static overlay that moves with the field picks the overlay up at the wrong place (a dragged copy of subtitle text): the
// tap takes the other one (as ui_reject.glsl does with CrosshairV2's mask), softly by how much more of the mask it hits.
void qRejectStatic(vec2 pa, vec2 pc_, inout vec3 wa, inout vec3 wc) {
    if (all(lessThan(abs(pc_ - pa), vec2(1.0)))) return;   // (nearly) the zero vector: both taps are the overlay's own place
    float za = qZAt(pa), zc = qZAt(pc_);
    vec3 a = wa;
    wa = mix(wa, wc, clamp(2.0 * (za - zc), 0.0, 1.0));
    wc = mix(wc, a, clamp(2.0 * (zc - za), 0.0, 1.0));
}

// Local structure at full-res pixel p, 8-bit levels x Q_SGAIN: the largest max-channel difference of prev's pixel to its right and
// lower neighbours. The field matches a moving background well, so over a static overlay both taps can agree on the background:
// structure there counts as that much disagreement for the zero-vector test (v2_morph.glsl).
float qStructure(ivec2 p) {
    ivec2 mx = ivec2(pc.W - 1, pc.H - 1);
    vec3 c = FG_FETCH(texelFetch(texPrev, p, 0).rgb);
    vec3 r = FG_FETCH(texelFetch(texPrev, min(p + ivec2(1, 0), mx), 0).rgb);
    vec3 d = FG_FETCH(texelFetch(texPrev, min(p + ivec2(0, 1), mx), 0).rgb);
    vec3 a = max(abs(c - r), abs(c - d));
    return Q_SGAIN * max(a.r, max(a.g, a.b));
}
#endif
