// The static guard (settings.learnedStaticGuard, learned estimate only; ml-fg scripts/static_guard.py, authoritative, and
// superdoc/features/static-guard.md). Where a and c are locally identical and the motion there is small, the output pixel becomes
// the plain time mix (1 - t) a + t c instead of the warped blend. Assembled (between common.glsl and the program file) into
// learned/sg_h.frag.glsl, learned/sg_v.frag.glsl and the 'final_sg' variant of synth/final.frag.glsl (FG_SGUARD).
// Units: colour differences in 8-bit levels of the backend's mismatch metric FG_D (common.glsl: Rgba8 integer levels; RgbaF16 the
// levels of the perceptual encoding, i.e. value x 255, or the PQ code x 255 for linear input), distances in full-res px.

// ==== STATIC GUARD CONSTANTS: static_guard.py DEFAULTS (ml-fg may send final values; this block is the only place to change) ====
const float SG_LO = 7.65;          // colorStatic = 1 - smoothstep(SG_LO, SG_HI, d), d = 3x3 mean of s
const float SG_HI = 22.95;
const float SG_HALO_LO = 5.1;      // uncertain = smoothstep(SG_HALO_LO, SG_HALO_HI, d); wHalo = 1 - smoothstep(.., dMin)
const float SG_HALO_HI = 7.65;
const int SG_HALO = 4;             // dMin = min of s over (2 SG_HALO + 1)^2 (9x9)
const float SG_MOTION_LO = 0.5;    // gate = 1 - smoothstep(SG_MOTION_LO, SG_MOTION_HI, motion), full-res px of the larger warp side
const float SG_MOTION_HI = 1.0;
const float SG_MIN_W = 0.001;      // wStatic <= this: the output pixel is left bit-exact
// ==================================================================================================================================

// static_guard.py smoothstep: clip((x - lo) / max(hi - lo, 1e-6), 0, 1), then x^2 (3 - 2x)
float sgStep(float lo, float hi, float x) {
    float v = clamp((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0);
    return v * v * (3.0 - 2.0 * v);
}

// s(p) = max over R, G, B of |a - c| (levels, FG_D), replicate border
float sgDiff(sampler2D a, sampler2D c, ivec2 p) {
    vec3 d = FG_D(fgPx(a, p), fgPx(c, p));
    return max(d.r, max(d.g, d.b));
}

// wStatic at full-res pixel p (static_guard.py weight()): dd = (d, dMin) from learned/sg_v.frag.glsl; field = the half-vector field
// on the grid of t (pixfield at t = 0.5, else fieldT: interp.field_at), flow res; ka = 2 max(t, 1 - t); S = flowScale.
// motion = cv2.resize(ka * S * |field|, (W, H), INTER_LINEAR): the magnitude at flow res, then bilinear with half-pixel centres at
// the scale fw / W (clamp-to-edge, as cv2 for an upscale).
float sgWeight(vec2 dd, sampler2D field, ivec2 p, ivec2 frame, float ka, int S) {
    ivec2 fs = textureSize(field, 0);
    vec2 sc = clamp((vec2(p) + 0.5) * (vec2(fs) / vec2(frame)) - 0.5, vec2(0.0), vec2(fs - 1));
    vec2 fl = floor(sc);
    ivec2 i0 = ivec2(fl), i1 = min(i0 + 1, fs - 1);
    vec2 f = sc - fl;
    float m00 = length(texelFetch(field, i0, 0).xy), m01 = length(texelFetch(field, ivec2(i1.x, i0.y), 0).xy);
    float m10 = length(texelFetch(field, ivec2(i0.x, i1.y), 0).xy), m11 = length(texelFetch(field, i1, 0).xy);
    float mag = (m00 * (1.0 - f.x) + m01 * f.x) * (1.0 - f.y) + (m10 * (1.0 - f.x) + m11 * f.x) * f.y;
    float motion = ka * float(S) * mag;
    float gate = 1.0 - sgStep(SG_MOTION_LO, SG_MOTION_HI, motion);
    float colorStatic = 1.0 - sgStep(SG_LO, SG_HI, dd.x);
    float uncertain = sgStep(SG_HALO_LO, SG_HALO_HI, dd.x);
    float wLocal = colorStatic * (1.0 - uncertain * (1.0 - gate));
    float wHalo = 1.0 - sgStep(SG_HALO_LO, SG_HALO_HI, dd.y);
    return max(wLocal, wHalo) * gate;
}
