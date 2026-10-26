// Shared by the learned motion estimator's passes (settings.motion 'learned'; ml-fg export/SPEC.md, export/reference.py).
// Assembled between common.glsl and the program file.

// The duplicate-pair skip: the net has no zero-motion prior (on a == c it still outputs motion), so a pair whose mean absolute
// difference is at most ML_DUP_DIFF levels (over every pixel and the R, G, B channels; FGR's duplicate test, proto/harness.py:29,50
// DUP_DIFF and gpu/compare.py:56,63) is a duplicate: every net pass writes zeros (and skips its work), the field is 0 and sadSum 0
// (what proto/interp.py's estimate gives for a == c). uDup is the 1x1 sum of |a - c| over all pixels and channels
// (learned/lv0.frag.glsl's second output + reduce_sad).
const float ML_DUP_DIFF = 0.6;
uniform ivec2 uFrameSize;
uniform sampler2D uDup;
bool mlDup() {
    return texelFetch(uDup, ivec2(0), 0).r <= ML_DUP_DIFF * 3.0 * float(uFrameSize.x) * float(uFrameSize.y);
}

// reference.py bilinear(): true float bilinear (no 1/32 quantisation), clamp-to-edge on an image of `size` texels (the level, or
// the unpadded flow grid): x = clamp(x, 0, w-1), x0 = floor(x), x1 = min(x0+1, w-1), fx = x - x0; every component of the texels.
vec4 mlBil(sampler2D t, ivec2 size, vec2 p) {
    vec2 c = clamp(p, vec2(0.0), vec2(size - 1));
    vec2 fl = floor(c);
    ivec2 i0 = ivec2(fl);
    ivec2 i1 = min(i0 + 1, size - 1);
    vec2 f = c - fl;
    vec4 p00 = texelFetch(t, i0, 0), p01 = texelFetch(t, ivec2(i1.x, i0.y), 0);
    vec4 p10 = texelFetch(t, ivec2(i0.x, i1.y), 0), p11 = texelFetch(t, i1, 0);
    return p00 * ((1.0 - f.y) * (1.0 - f.x)) + p01 * ((1.0 - f.y) * f.x) + p10 * (f.y * (1.0 - f.x)) + p11 * (f.y * f.x);
}

// SPEC stage 4a (reference.py upsample2, then * 2): the field of level l at texel p from the coarser level's field hc (RG32F, its
// exact size): bilinear with half-pixel centres and edge clamp, along y first, then x, then doubled (level pixels). The match pass
// and the level's last conv both compute it (h stays float32: it is never stored in the half-float activations).
vec2 mlFieldUp(sampler2D hc, ivec2 p) {
    ivec2 n = textureSize(hc, 0);
    ivec2 k = p >> 1;
    ivec2 kn = ivec2((p.x & 1) == 0 ? max(k.x - 1, 0) : min(k.x + 1, n.x - 1), (p.y & 1) == 0 ? max(k.y - 1, 0) : min(k.y + 1, n.y - 1));
    vec2 g0 = 0.75 * texelFetch(hc, k, 0).xy + 0.25 * texelFetch(hc, ivec2(k.x, kn.y), 0).xy;     // along y (column k.x)
    vec2 g1 = 0.75 * texelFetch(hc, ivec2(kn.x, k.y), 0).xy + 0.25 * texelFetch(hc, kn, 0).xy;   // along y (column kn.x)
    return (0.75 * g0 + 0.25 * g1) * 2.0;                                                          // along x, then x2
}
