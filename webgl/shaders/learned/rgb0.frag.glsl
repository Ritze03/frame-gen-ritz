// Learned-features nets (SPEC 11.1-11.2, level 0): the flow-res RGB planes of ONE frame on the PADDED flow grid (uSize = the unpadded
// fw x fh): per texel the mean of the valid pixels of the 2x2 cell (stage 2's filter on R, G, B separately, not rounded), edge
// replicated beyond fw x fh, divided by 255. out: r, g, b = R, G, B in 0..1, a = 0. (RgbaF16: the levels of the perceptual encoding.)
uniform sampler2D uSrc;
uniform ivec2 uSize;
layout(location = 0) out vec4 oRgb;

vec3 mlLevels(ivec2 p) {
#ifdef FG_F16
    return fgMetric(texelFetch(uSrc, p, 0).rgb);
#else
    return floor(texelFetch(uSrc, p, 0).rgb * 255.0 + 0.5);
#endif
}

void main() {
    ivec2 o = min(ivec2(gl_FragCoord.xy), uSize - 1);   // replicate padding (right and bottom)
    vec3 s = vec3(0.0);
    int n = 0;
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
            ivec2 p = 2 * o + ivec2(i, j);
            if (p.x < uFrameSize.x && p.y < uFrameSize.y) { s += mlLevels(p); n++; }
        }
    oRgb = vec4(s * (n == 4 ? 0.25 : (n == 2 ? 0.5 : 1.0)) / 255.0, 0.0);
}
