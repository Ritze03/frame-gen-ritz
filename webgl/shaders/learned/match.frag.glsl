// SPEC stage 4a + 4b at one level (reference.py upsample2 + match): the coarser field upsampled and doubled (mlFieldUp), then the
// 14-channel conv input from la, lc (this level's luma, 0..1) and h, plus the variant's extra channels (the "xc" vector below):
// colour nets (ML_COLOUR, SPEC 10.3, 25 channels) and learned-features nets (ML_FEAT, SPEC 11.4, 27 channels). Rendered over
// the level (uSize) into the level-0-sized MRT; texels beyond the level are never read (the convs check uSize).
// Channels (the net's order): 0 wa, 1 wc, 2 wa - wc, 3..11 cost(dy,dx) = |la(q - h - d) - lc(q + h + d)|, d = (dx, dy), dy outer,
// 12 hx / 4, 13 hy / 4; then xc[0..ML_ND-1] = the extra planes' difference at d = 0 (channels 14..), xc[ML_ND + k] = the extra planes'
// L1 cost at offset k (channels 14 + ML_ND + k), the planes sampled at the luma's coordinates:
//   ML_COLOUR (ML_ND 2): uEx = this level's chroma (r = Cb_a, g = Cr_a, b = Cb_c, a = Cr_c): dCb, dCr; cost |dCb| + |dCr|
//   ML_FEAT   (ML_ND 4): uFa, uFc = the level's 4 feature maps of a, c (the extractor's output): dF_0..3; cost sum |dF_j|
// Out, settings.motionPrecision 'fp32' (RGBA32F x3 + RG32F, the 4th RGBA32F with extra channels): oX0 = channels 0..3, oX1 4..7, oX2
// 8..11, oX3 = 12..15; channels 16..27 come from a second pass (ML_CB 1: oX0 = 16..19, oX1 = 20..23, oX2 = 24..27; 7 input textures
// would need 7 draw buffers, WebGL2 guarantees 4); 'fp16' (ML_PACKED: RGBA32UI, packHalf2x16 as conv_h.frag.glsl, 8 channels per
// texture): oP0 = channels 0..7, oP1 = 8..15 (14, 15 zero for a luma net), with extra channels oP2 = 16..23, oP3 = 24..31 (zero beyond
// the net's). ML_CB is always defined (0 or 1). The level's last conv recomputes h itself in float32 (mlFieldUp).
uniform sampler2D uPyr;   // this level: r = la, g = lc (exact level size)
uniform sampler2D uHc;    // the field after level l+1 (RG32F, its exact size); unused at the top level
#ifdef ML_COLOUR
#define ML_ND 2
uniform sampler2D uEx;
#endif
#ifdef ML_FEAT
#define ML_ND 4
uniform sampler2D uFa;
uniform sampler2D uFc;
#endif
#ifdef ML_ND
#define ML_EXTRA 1
#endif
uniform ivec2 uSize;
uniform int uTop;
#ifdef ML_PACKED
layout(location = 0) out uvec4 oP0;
layout(location = 1) out uvec4 oP1;
#ifdef ML_EXTRA
layout(location = 2) out uvec4 oP2;
layout(location = 3) out uvec4 oP3;
#endif
#elif ML_CB
layout(location = 0) out vec4 oX0;
layout(location = 1) out vec4 oX1;
layout(location = 2) out vec4 oX2;
#else
layout(location = 0) out vec4 oX0;
layout(location = 1) out vec4 oX1;
layout(location = 2) out vec4 oX2;
layout(location = 3) out vec4 oX3;
#endif

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 x0 = vec4(0.0), x1 = vec4(0.0), x2 = vec4(0.0), x3 = vec4(0.0);
    float xc[16];   // the extra channels (channels 14..)
    for (int k = 0; k < 16; k++) xc[k] = 0.0;
    if (!mlDup()) {
        vec2 h = uTop != 0 ? vec2(0.0) : mlFieldUp(uHc, p);
        vec2 q = vec2(p);
        float c[9];
        float wa = 0.0, wc = 0.0;
        for (int k = 0; k < 9; k++) {
            vec2 d = vec2(float(k % 3 - 1), float(k / 3 - 1));   // dy outer, dx inner
            float sa = mlBil(uPyr, uSize, (q - h) - d).r;
            float sc = mlBil(uPyr, uSize, (q + h) + d).g;
            c[k] = abs(sa - sc);
            if (k == 4) { wa = sa; wc = sc; }
#ifdef ML_EXTRA
#ifndef ML_PACKED
#if !ML_CB
            if (k == 4)   // (fp32: the first pass only needs d = 0, the costs are the second pass)
#endif
#endif
            {
#ifdef ML_COLOUR
                vec4 dd = vec4(mlBil(uEx, uSize, (q - h) - d).xy - mlBil(uEx, uSize, (q + h) + d).zw, 0.0, 0.0);
#else
                vec4 dd = mlBil(uFa, uSize, (q - h) - d) - mlBil(uFc, uSize, (q + h) + d);
#endif
                xc[ML_ND + k] = abs(dd.x) + abs(dd.y) + abs(dd.z) + abs(dd.w);
                if (k == 4) { for (int j = 0; j < ML_ND; j++) xc[j] = dd[j]; }
            }
#endif
        }
#if ML_CB
        x0 = vec4(xc[2], xc[3], xc[4], xc[5]);     // channels 16..27 = xc[2..13]
        x1 = vec4(xc[6], xc[7], xc[8], xc[9]);
        x2 = vec4(xc[10], xc[11], xc[12], xc[13]);
#else
        x0 = vec4(wa, wc, wa - wc, c[0]);
        x1 = vec4(c[1], c[2], c[3], c[4]);
        x2 = vec4(c[5], c[6], c[7], c[8]);
        x3 = vec4(h * 0.25, xc[0], xc[1]);
#endif
    }
#ifdef ML_PACKED
    oP0 = uvec4(packHalf2x16(x0.xy), packHalf2x16(x0.zw), packHalf2x16(x1.xy), packHalf2x16(x1.zw));
    oP1 = uvec4(packHalf2x16(x2.xy), packHalf2x16(x2.zw), packHalf2x16(x3.xy), packHalf2x16(x3.zw));
#ifdef ML_EXTRA
    oP2 = uvec4(packHalf2x16(vec2(xc[2], xc[3])), packHalf2x16(vec2(xc[4], xc[5])), packHalf2x16(vec2(xc[6], xc[7])), packHalf2x16(vec2(xc[8], xc[9])));
    oP3 = uvec4(packHalf2x16(vec2(xc[10], xc[11])), packHalf2x16(vec2(xc[12], xc[13])), 0u, 0u);
#endif
#elif ML_CB
    oX0 = x0; oX1 = x1; oX2 = x2;
#else
    oX0 = x0; oX1 = x1; oX2 = x2; oX3 = x3;
#endif
}
