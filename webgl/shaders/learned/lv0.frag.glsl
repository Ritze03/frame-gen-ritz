// Level 0 of the net's own pyramid (SPEC stages 1-3, reference.py luma / flow_luma / pyramid): per texel of the PADDED flow grid
// (uSize = the unpadded fw x fh; the texture is padded to a multiple of 16), the flow-res luma of a and c with edge replication.
// Luma: cv2 5.0.0 BGR2GRAY, (3735 B + 19235 G + 9798 R + 16384) >> 15 on integer levels (RgbaF16: the same weights on the levels of
// the perceptual encoding, unrounded); flow res: the mean of the valid pixels of the 2x2 cell (cv2 INTER_AREA, as luma.frag.glsl).
// Out: r = ya / 255, g = yc / 255 (the net's level 0, 0..1), b = ya, a = yc (0..255, for the SAD).
uniform sampler2D uPrev;
uniform sampler2D uCurr;
uniform ivec2 uSize;
layout(location = 0) out vec4 oLv;

float mlLuma(sampler2D t, ivec2 p) {
#ifdef FG_F16
    return dot(fgMetric(texelFetch(t, p, 0).rgb), vec3(9798.0, 19235.0, 3735.0)) * (1.0 / 32768.0);
#else
    uvec3 c = uvec3(texelFetch(t, p, 0).rgb * 255.0 + 0.5);
    return float((c.r * 9798u + c.g * 19235u + c.b * 3735u + 16384u) >> 15);
#endif
}

void main() {
    ivec2 o = min(ivec2(gl_FragCoord.xy), uSize - 1);   // replicate padding (right and bottom)
    float sa = 0.0, sc = 0.0;
    int n = 0;
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
            ivec2 p = 2 * o + ivec2(i, j);
            if (p.x < uFrameSize.x && p.y < uFrameSize.y) {
                sa += mlLuma(uPrev, p);
                sc += mlLuma(uCurr, p);
                n++;
            }
        }
    float s = n == 4 ? 0.25 : (n == 2 ? 0.5 : 1.0);   // (exact: the sums are integers)
    float ya = sa * s, yc = sc * s;
    oLv = vec4(ya / 255.0, yc / 255.0, ya, yc);
}
