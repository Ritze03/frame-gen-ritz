// Pixel field: 5x5 box (replicate border) of each candidate's error image, bonuses for the bilinear (k=0) and
// zero (k=6) candidates, strict <, first wins. pixfield.comp:47-61; PY:181-183.
uniform sampler2D uPfErr0;
uniform sampler2D uPfErr1;
uniform sampler2D uBlockField;   // sub2
uniform float uSmoothBonus;
uniform float uZeroBonus;
layout(location = 0) out vec4 oField;

void main() {
    ivec2 q = ivec2(gl_FragCoord.xy);
    ivec2 fs = textureSize(uPfErr0, 0);
    // per-candidate sums, each accumulated y outer / x inner (component-wise, so order matches pixfield.comp)
    vec4 s0 = vec4(0.0);
    vec3 s1 = vec3(0.0);
    for (int y = -2; y <= 2; y++)
        for (int x = -2; x <= 2; x++) {
            ivec2 p = clamp(q + ivec2(x, y), ivec2(0), fs - 1);
            s0 += texelFetch(uPfErr0, p, 0);
            s1 += texelFetch(uPfErr1, p, 0).xyz;
        }
    int bk = 0;
    float best = 0.0;
    for (int k = 0; k < 7; k++) {
        float sum = k < 4 ? s0[k] : s1[k - 4];
        float c = sum * (1.0 / 25.0);
        if (k == 0) c -= uSmoothBonus;
        if (k == 6) c -= uZeroBonus;
        if (k == 0 || c < best) { best = c; bk = k; }
    }
    oField = vec4(fgPixCand(bk, q, uBlockField), 0.0, 0.0);
}
