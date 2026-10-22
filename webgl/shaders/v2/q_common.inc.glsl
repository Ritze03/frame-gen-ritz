// V2 Quality ("coverage morph") estimate passes: constants and helpers shared by v2/qfield, v2/qclaim, v2/qown.vert, v2/qvis and
// v2/qcov (the host assembles common.glsl + this file + the program file). Port of gpu/shaders/v2_quality.glsl. Per estimate, after
// Performance's re-pick and clean-up of the block field:
//   v2/qfield  the per-pixel field (flow resolution): the cleaned block field upsampled with edge awareness, and the static-overlay
//              mask z (the field is zero where z >= 0.5)                                                         -> v2.qf (xy = h, z)
//   v2/qclaim  each midpoint location's own claim: key = |luma_a(m - h) - luma_c(m + h)| in 1/Q_KEY levels, the quantised vector,
//              and its footprint pixels in prev (m - h) and curr (m + h)                                        -> v2.qclaim
//   v2/qown    the coverage splat (the library's atomicMin of key << 20 | vector): per side the static-overlay claims (v2/qown_z),
//              then one MIN-blended point per location carrying the word as float bits (v2.qownA / v2.qownC, see v2/qown.vert.glsl)
//   v2/qvis    does each frame still show the content of each location (0 / 1 per side)                         -> v2.qvis
//   v2/qcov    the coverage: qvis box-filtered over (2 Q_R + 1)^2, quantised as the library packs it, with z     -> v2.qcov

const float Q_FSAME = 0.5;
const float Q_FS = 2.0;
const float Q_FK = 1.0;
const float Q_FB = 0.1;
const float Q_ZMOVE = 1.0;
const float Q_ZCON = 16.0;
const float Q_ZR0 = 0.04, Q_ZR1 = 0.08;
const float Q_KEY = 8.0;
const float Q_MARGIN = 8.0;
const float Q_WIN = 6.0;
const float Q_TOL = 0.75;
const int Q_R = 2;
const int Q_VBLOCK = 8;

// flow-resolution luma (level 0), clamp to edge, and its plain bilinear (no 1/32 quantisation: v2_quality.glsl qLuma)
float qL(sampler2D t, ivec2 i) { return texelFetch(t, clamp(i, ivec2(0), textureSize(t, 0) - 1), 0).r; }
float qLuma(sampler2D t, vec2 p) {
    vec2 fl = floor(p);
    vec2 f = p - fl;
    ivec2 i = ivec2(fl);
    return mix(mix(qL(t, i), qL(t, i + ivec2(1, 0)), f.x), mix(qL(t, i + ivec2(0, 1)), qL(t, i + ivec2(1, 1)), f.x), f.y);
}
// the per-pixel field (v2.qf xy), clamp to edge
vec2 qFt(sampler2D t, ivec2 m) { return texelFetch(t, clamp(m, ivec2(0), textureSize(t, 0) - 1), 0).xy; }

ivec2 qQuant(vec2 h) { return clamp(ivec2(roundEven(h * 4.0)), ivec2(-512), ivec2(511)); }
// the low 20 bits of the library's claim word (the quantised vector)
int qVbits(ivec2 v) { return ((v.x + 512) << 10) | (v.y + 512); }
vec2 qVec(int vb) { return vec2(ivec2((vb >> 10) & 1023, vb & 1023) - 512) * 0.25; }
// the footprint pixel of p (flow coordinates) packed x | y << 16, -1 if outside the frame
int qPix(vec2 p, ivec2 fs) {
    ivec2 i = ivec2(floor(p + 0.5));
    return all(greaterThanEqual(i, ivec2(0))) && all(lessThan(i, fs)) ? (i.x | (i.y << 16)) : -1;
}
ivec2 qUnpix(int w) { return ivec2(w & 0xFFFF, w >> 16); }
