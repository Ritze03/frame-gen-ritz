// V2 finest-level search pick (v2_search.comp, the part after the SAD scan): the raw argmin vector of the block (NO 3x3 median: V2
// medians later, in v2/clean) plus the equiangular (V-shaped, Shimizu & Okutomi) sub-pixel fit of the SAD around it in x and in y
// (none at the search window's edge). cost = SAD/64 + LAMBDA*|d|_1, strict <, row-major, first wins. Writes V2's "fitted finest field".
uniform sampler2D uSearchSad;
uniform sampler2D uSeed;
uniform float uLambda;
uniform int uRadius;
layout(location = 0) out vec4 oField;

float vFit(float cm, float c0, float cp) {
    float den = 2.0 * (max(cm, cp) - c0);
    return den > 0.0 ? clamp((cm - cp) / den, -0.5, 0.5) : 0.0;
}

float sadAt(ivec2 b, int D, int x, int y) {
    return texelFetch(uSearchSad, ivec2(D * b.x + x, D * b.y + y), 0).r * (1.0 / 64.0);
}

void main() {
    ivec2 b = ivec2(gl_FragCoord.xy);
    int R = uRadius, D = 2 * R + 1;
    ivec2 seed = ivec2(roundEven(texelFetch(uSeed, b, 0).xy));
    int bi = 0;
    float best = 0.0;
    for (int ci = 0; ci < D * D; ci++) {
        int dx = ci % D - R, dy = ci / D - R;
        float cost = sadAt(b, D, ci % D, ci / D) + uLambda * float(abs(dx) + abs(dy));
        if (ci == 0 || cost < best) { best = cost; bi = ci; }
    }
    int x = bi % D, y = bi / D;
    vec2 v = vec2(seed + ivec2(x - R, y - R));
    float c0 = sadAt(b, D, x, y);
    if (x > 0 && x < D - 1) v.x += vFit(sadAt(b, D, x - 1, y), c0, sadAt(b, D, x + 1, y));
    if (y > 0 && y < D - 1) v.y += vFit(sadAt(b, D, x, y - 1), c0, sadAt(b, D, x, y + 1));
    oField = vec4(v, 0.0, 0.0);
}
