// SPEC stage 4a + 4b at one level (reference.py upsample2 + match): the coarser field upsampled (bilinear, half-pixel centres, edge
// clamp; y first, then x) and doubled, then the 14-channel conv input from la, lc (this level's luma, 0..1) and h. Rendered over the
// level (uSize) into the level-0-sized MRT; texels beyond the level are never read (the convs check uSize).
// Out (channel order of the net): oX0 = (wa, wc, wa - wc, cost(-1,-1)), oX1 = cost(-1,0), (-1,1), (0,-1), (0,0),
// oX2 = cost(0,1), (1,-1), (1,0), (1,1), oX3 = (hx / 4, hy / 4, 0, 0): cost(dy,dx) = |la(q - h - d) - lc(q + h + d)|, d = (dx, dy).
// h itself is 4 * oX3.xy (exact: a power of two), which the level's last conv reads back.
uniform sampler2D uPyr;   // this level: r = la, g = lc (exact level size)
uniform sampler2D uHc;    // the field after level l+1 (RG32F, its exact size); unused at the top level
uniform ivec2 uSize;
uniform int uTop;
layout(location = 0) out vec4 oX0;
layout(location = 1) out vec4 oX1;
layout(location = 2) out vec4 oX2;
layout(location = 3) out vec4 oX3;

vec2 fieldUp(ivec2 p) {
    ivec2 n = textureSize(uHc, 0);
    ivec2 k = p >> 1;
    ivec2 kn = ivec2((p.x & 1) == 0 ? max(k.x - 1, 0) : min(k.x + 1, n.x - 1), (p.y & 1) == 0 ? max(k.y - 1, 0) : min(k.y + 1, n.y - 1));
    vec2 g0 = 0.75 * texelFetch(uHc, k, 0).xy + 0.25 * texelFetch(uHc, ivec2(k.x, kn.y), 0).xy;      // along y (column k.x)
    vec2 g1 = 0.75 * texelFetch(uHc, ivec2(kn.x, k.y), 0).xy + 0.25 * texelFetch(uHc, kn, 0).xy;    // along y (column kn.x)
    return (0.75 * g0 + 0.25 * g1) * 2.0;                                                            // along x, then x2
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    if (mlDup()) {
        oX0 = vec4(0.0); oX1 = vec4(0.0); oX2 = vec4(0.0); oX3 = vec4(0.0);
        return;
    }
    vec2 h = uTop != 0 ? vec2(0.0) : fieldUp(p);
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
    oX0 = vec4(wa, wc, wa - wc, c[0]);
    oX1 = vec4(c[1], c[2], c[3], c[4]);
    oX2 = vec4(c[5], c[6], c[7], c[8]);
    oX3 = vec4(h * 0.25, 0.0, 0.0);
}
