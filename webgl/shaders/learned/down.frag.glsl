// Net pyramid level l+1 = 2x2 mean of level l (SPEC stage 3; reference.py pyramid(): ((p00 + p01) + (p10 + p11)) * 0.25, every
// level an exact half of its parent). r = la, g = lc. ML_V4 (a colour net's chroma pyramid, SPEC section 10.2): all four channels
// (Cb_a, Cr_a, Cb_c, Cr_c) the same way.
uniform sampler2D uSrc;
layout(location = 0) out vec4 oLv;

void main() {
    ivec2 p = 2 * ivec2(gl_FragCoord.xy);
#ifdef ML_V4
    vec4 p00 = texelFetch(uSrc, p, 0), p01 = texelFetch(uSrc, p + ivec2(1, 0), 0);
    vec4 p10 = texelFetch(uSrc, p + ivec2(0, 1), 0), p11 = texelFetch(uSrc, p + ivec2(1, 1), 0);
    oLv = ((p00 + p01) + (p10 + p11)) * 0.25;
#else
    vec2 p00 = texelFetch(uSrc, p, 0).rg, p01 = texelFetch(uSrc, p + ivec2(1, 0), 0).rg;
    vec2 p10 = texelFetch(uSrc, p + ivec2(0, 1), 0).rg, p11 = texelFetch(uSrc, p + ivec2(1, 1), 0).rg;
    oLv = vec4(((p00 + p01) + (p10 + p11)) * 0.25, 0.0, 0.0);
#endif
}
