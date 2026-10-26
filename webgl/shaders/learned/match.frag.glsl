// SPEC stage 4a + 4b at one level (reference.py upsample2 + match): the coarser field upsampled and doubled (mlFieldUp), then the
// 14-channel conv input from la, lc (this level's luma, 0..1) and h. Rendered over the level (uSize) into the level-0-sized MRT;
// texels beyond the level are never read (the convs check uSize).
// Out (channel order of the net): oX0 = (wa, wc, wa - wc, cost(-1,-1)), oX1 = cost(-1,0), (-1,1), (0,-1), (0,0),
// oX2 = cost(0,1), (1,-1), (1,0), (1,1), oX3 = (hx / 4, hy / 4): cost(dy,dx) = |la(q - h - d) - lc(q + h + d)|, d = (dx, dy).
// (The textures are RGBA16F / RG16F with settings.motionPrecision 'fp16', RGBA32F / RG32F with 'fp32'; the level's last conv
// recomputes h itself in float32, mlFieldUp.)
uniform sampler2D uPyr;   // this level: r = la, g = lc (exact level size)
uniform sampler2D uHc;    // the field after level l+1 (RG32F, its exact size); unused at the top level
uniform ivec2 uSize;
uniform int uTop;
layout(location = 0) out vec4 oX0;
layout(location = 1) out vec4 oX1;
layout(location = 2) out vec4 oX2;
layout(location = 3) out vec4 oX3;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    if (mlDup()) {
        oX0 = vec4(0.0); oX1 = vec4(0.0); oX2 = vec4(0.0); oX3 = vec4(0.0);
        return;
    }
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
    oX0 = vec4(wa, wc, wa - wc, c[0]);
    oX1 = vec4(c[1], c[2], c[3], c[4]);
    oX2 = vec4(c[5], c[6], c[7], c[8]);
    oX3 = vec4(h * 0.25, 0.0, 0.0);
}
