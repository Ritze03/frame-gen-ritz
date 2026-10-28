// Level 0 of the net's own pyramid (SPEC stages 1-3, reference.py luma / flow_luma / pyramid): per texel of the PADDED flow grid
// (uSize = the unpadded fw x fh; the texture is padded to a multiple of 16), the flow-res luma of a and c with edge replication.
// Luma: cv2 5.0.0 BGR2GRAY, (3735 B + 19235 G + 9798 R + 16384) >> 15 on integer levels (RgbaF16: the same weights on the levels of
// the perceptual encoding, unrounded); flow res: the mean of the valid pixels of the 2x2 cell (cv2 INTER_AREA, as luma.frag.glsl).
// oLv: r = ya / 255, g = yc / 255 (the net's level 0, 0..1), b = ya, a = yc (0..255, for the SAD).
// ML_COLOUR (a colour net, SPEC section 10.1): a third output oEx = (Cb_a, Cr_a, Cb_c, Cr_c) / 255 at the same texels: Cb = 0.564 (B - Y),
// Cr = 0.713 (R - Y) with the UNROUNDED Y = 0.299 R + 0.587 G + 0.114 B on the levels, the same 2x2 cell mean (and replicate padding).
// oDup: the first step of the duplicate test (ml_common.inc.glsl mlDup): the sum of |a - c| over R, G, B (levels, FG_D) of the
// full-res pixels of this flow texel's cell, the last row / column of cells also taking the frame's remaining pixels (so the
// texels of the unpadded grid cover every pixel exactly once); 0 in the padding. reduce_sad sums it to 1x1.
uniform sampler2D uPrev;
uniform sampler2D uCurr;
uniform ivec2 uSize;
layout(location = 0) out vec4 oLv;
layout(location = 1) out vec4 oDup;
#ifdef ML_COLOUR
layout(location = 2) out vec4 oEx;
#endif

float mlLuma(sampler2D t, ivec2 p) {
#ifdef FG_F16
    return dot(fgMetric(texelFetch(t, p, 0).rgb), vec3(9798.0, 19235.0, 3735.0)) * (1.0 / 32768.0);
#else
    uvec3 c = uvec3(texelFetch(t, p, 0).rgb * 255.0 + 0.5);
    return float((c.r * 9798u + c.g * 19235u + c.b * 3735u + 16384u) >> 15);
#endif
}

#ifdef ML_COLOUR
vec2 mlChroma(sampler2D t, ivec2 p) {
#ifdef FG_F16
    vec3 c = fgMetric(texelFetch(t, p, 0).rgb);
#else
    vec3 c = floor(texelFetch(t, p, 0).rgb * 255.0 + 0.5);
#endif
    float y = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;
    return vec2(0.564 * (c.b - y), 0.713 * (c.r - y));
}
#endif

void main() {
    ivec2 q = ivec2(gl_FragCoord.xy);
    ivec2 o = min(q, uSize - 1);   // replicate padding (right and bottom)
    float sa = 0.0, sc = 0.0;
    vec2 ea = vec2(0.0), ec = vec2(0.0);
    int n = 0;
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
            ivec2 p = 2 * o + ivec2(i, j);
            if (p.x < uFrameSize.x && p.y < uFrameSize.y) {
                sa += mlLuma(uPrev, p);
                sc += mlLuma(uCurr, p);
#ifdef ML_COLOUR
                ea += mlChroma(uPrev, p);
                ec += mlChroma(uCurr, p);
#endif
                n++;
            }
        }
    float s = n == 4 ? 0.25 : (n == 2 ? 0.5 : 1.0);   // (exact: the sums are integers)
    float ya = sa * s, yc = sc * s;
    oLv = vec4(ya / 255.0, yc / 255.0, ya, yc);
#ifdef ML_COLOUR
    oEx = vec4(ea * s, ec * s) / 255.0;
#endif
    float dup = 0.0;
    if (q.x < uSize.x && q.y < uSize.y) {
        ivec2 e = ivec2(q.x == uSize.x - 1 ? uFrameSize.x : min(2 * q.x + 2, uFrameSize.x),    // end of the cell (exclusive)
                        q.y == uSize.y - 1 ? uFrameSize.y : min(2 * q.y + 2, uFrameSize.y));
        for (int y = 2 * q.y; y < e.y; y++)
            for (int x = 2 * q.x; x < e.x; x++) {
                vec3 d = FG_D(fgPx(uPrev, ivec2(x, y)), fgPx(uCurr, ivec2(x, y)));
                dup += (d.r + d.g) + d.b;
            }
    }
    oDup = vec4(dup);
}
