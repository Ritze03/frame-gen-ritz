// SPEC stage 4a + 4b at one level (reference.py upsample2 + match): the coarser field upsampled and doubled (mlFieldUp), then the
// 14-channel conv input from la, lc (this level's luma, 0..1) and h. Rendered over the level (uSize) into the level-0-sized MRT;
// texels beyond the level are never read (the convs check uSize).
// Channels (the net's order): 0 wa, 1 wc, 2 wa - wc, 3..11 cost(dy,dx) = |la(q - h - d) - lc(q + h + d)|, d = (dx, dy), dy outer,
// 12 hx / 4, 13 hy / 4. Out, settings.motionPrecision 'fp32' (RGBA32F x3 + RG32F): oX0 = channels 0..3, oX1 4..7, oX2 8..11,
// oX3 = 12, 13; 'fp16' (ML_PACKED: RGBA32UI x2, packHalf2x16 as conv_h.frag.glsl): oP0 = channels 0..7, oP1 = 8..13 (+ 2 zeros).
// The level's last conv recomputes h itself in float32 (mlFieldUp).
uniform sampler2D uPyr;   // this level: r = la, g = lc (exact level size)
uniform sampler2D uHc;    // the field after level l+1 (RG32F, its exact size); unused at the top level
uniform ivec2 uSize;
uniform int uTop;
#ifdef ML_PACKED
layout(location = 0) out uvec4 oP0;
layout(location = 1) out uvec4 oP1;
#else
layout(location = 0) out vec4 oX0;
layout(location = 1) out vec4 oX1;
layout(location = 2) out vec4 oX2;
layout(location = 3) out vec4 oX3;
#endif

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 x0 = vec4(0.0), x1 = vec4(0.0), x2 = vec4(0.0), x3 = vec4(0.0);
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
        }
        x0 = vec4(wa, wc, wa - wc, c[0]);
        x1 = vec4(c[1], c[2], c[3], c[4]);
        x2 = vec4(c[5], c[6], c[7], c[8]);
        x3 = vec4(h * 0.25, 0.0, 0.0);
    }
#ifdef ML_PACKED
    oP0 = uvec4(packHalf2x16(x0.xy), packHalf2x16(x0.zw), packHalf2x16(x1.xy), packHalf2x16(x1.zw));
    oP1 = uvec4(packHalf2x16(x2.xy), packHalf2x16(x2.zw), packHalf2x16(x3.xy), 0u);
#else
    oX0 = x0; oX1 = x1; oX2 = x2; oX3 = x3;
#endif
}
