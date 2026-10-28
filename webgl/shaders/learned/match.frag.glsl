// SPEC stage 4a + 4b at one level (reference.py upsample2 + match): the coarser field upsampled and doubled (mlFieldUp), then the
// 14-channel conv input from la, lc (this level's luma, 0..1) and h (25 channels for a colour net, SPEC section 10.3). Rendered over
// the level (uSize) into the level-0-sized MRT; texels beyond the level are never read (the convs check uSize).
// Channels (the net's order): 0 wa, 1 wc, 2 wa - wc, 3..11 cost(dy,dx) = |la(q - h - d) - lc(q + h + d)|, d = (dx, dy), dy outer,
// 12 hx / 4, 13 hy / 4; colour nets (ML_COLOUR; uEx = this level's chroma, r = Cb_a, g = Cr_a, b = Cb_c, a = Cr_c, sampled at the
// luma's coordinates): 14 Cb_a - Cb_c, 15 Cr_a - Cr_c (at d = 0), 16..24 |dCb_d| + |dCr_d| (cost order).
// Out, settings.motionPrecision 'fp32' (RGBA32F x3 + RG32F, the 4th RGBA32F with ML_COLOUR): oX0 = channels 0..3, oX1 4..7, oX2 8..11,
// oX3 = 12..15; a colour net's channels 16..24 come from a second pass (ML_CB 1: oX0 = 16..19, oX1 = 20..23, oX2 = 24; a 7-texture
// input would need 7 draw buffers, WebGL2 guarantees 4); 'fp16' (ML_PACKED: RGBA32UI, packHalf2x16 as conv_h.frag.glsl, 8 channels
// per texture): oP0 = channels 0..7, oP1 = 8..15 (14, 15 zero for a luma net), with ML_COLOUR oP2 = 16..23, oP3 = 24 (+ 7 zeros).
// ML_CB is always defined (0 or 1). The level's last conv recomputes h itself in float32 (mlFieldUp).
uniform sampler2D uPyr;   // this level: r = la, g = lc (exact level size)
uniform sampler2D uHc;    // the field after level l+1 (RG32F, its exact size); unused at the top level
#ifdef ML_COLOUR
uniform sampler2D uEx;
#endif
uniform ivec2 uSize;
uniform int uTop;
#ifdef ML_PACKED
layout(location = 0) out uvec4 oP0;
layout(location = 1) out uvec4 oP1;
#ifdef ML_COLOUR
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
    float cc[9];   // the colour costs (channels 16..24)
    for (int k = 0; k < 9; k++) cc[k] = 0.0;
    if (!mlDup()) {
        vec2 h = uTop != 0 ? vec2(0.0) : mlFieldUp(uHc, p);
        vec2 q = vec2(p);
        float c[9];
        float wa = 0.0, wc = 0.0;
        vec2 d0 = vec2(0.0);
        for (int k = 0; k < 9; k++) {
            vec2 d = vec2(float(k % 3 - 1), float(k / 3 - 1));   // dy outer, dx inner
            float sa = mlBil(uPyr, uSize, (q - h) - d).r;
            float sc = mlBil(uPyr, uSize, (q + h) + d).g;
            c[k] = abs(sa - sc);
            if (k == 4) { wa = sa; wc = sc; }
#ifdef ML_COLOUR
#ifndef ML_PACKED
#if !ML_CB
            if (k == 4)   // (fp32: the luma pass only needs d = 0, the costs are the second pass)
#endif
#endif
            {
                vec2 dc = mlBil(uEx, uSize, (q - h) - d).xy - mlBil(uEx, uSize, (q + h) + d).zw;
                cc[k] = abs(dc.x) + abs(dc.y);
                if (k == 4) d0 = dc;
            }
#endif
        }
#if ML_CB
        x0 = vec4(cc[0], cc[1], cc[2], cc[3]);
        x1 = vec4(cc[4], cc[5], cc[6], cc[7]);
        x2 = vec4(cc[8], 0.0, 0.0, 0.0);
#else
        x0 = vec4(wa, wc, wa - wc, c[0]);
        x1 = vec4(c[1], c[2], c[3], c[4]);
        x2 = vec4(c[5], c[6], c[7], c[8]);
        x3 = vec4(h * 0.25, d0);
#endif
    }
#ifdef ML_PACKED
    oP0 = uvec4(packHalf2x16(x0.xy), packHalf2x16(x0.zw), packHalf2x16(x1.xy), packHalf2x16(x1.zw));
    oP1 = uvec4(packHalf2x16(x2.xy), packHalf2x16(x2.zw), packHalf2x16(x3.xy), packHalf2x16(x3.zw));
#ifdef ML_COLOUR
    oP2 = uvec4(packHalf2x16(vec2(cc[0], cc[1])), packHalf2x16(vec2(cc[2], cc[3])), packHalf2x16(vec2(cc[4], cc[5])), packHalf2x16(vec2(cc[6], cc[7])));
    oP3 = uvec4(packHalf2x16(vec2(cc[8], 0.0)), 0u, 0u, 0u);
#endif
#elif ML_CB
    oX0 = x0; oX1 = x1; oX2 = x2;
#else
    oX0 = x0; oX1 = x1; oX2 = x2; oX3 = x3;
#endif
}
