// Search argmin fused with the 3x3 per-component median (replicate border): the argmin of each of the 9
// neighbouring blocks is recomputed here. cost = SAD/64 + LAMBDA*|d|_1, strict <, row-major (dy outer),
// first wins. search.comp:63-79; median3.comp:15-27; PY:121-128.
uniform sampler2D uSearchSad;
uniform sampler2D uSeed;
uniform float uLambda;
uniform int uRadius;
layout(location = 0) out vec4 oField;

// winning vector (seed + best offset) of block nb (already clamped to the grid)
vec2 bestVec(ivec2 nb) {
    int R = uRadius, D = 2 * R + 1;
    ivec2 seed = ivec2(roundEven(texelFetch(uSeed, nb, 0).xy));
    int bi = 0;
    float best = 0.0;
    for (int ci = 0; ci < D * D; ci++) {
        int dx = ci % D - R, dy = ci / D - R;
        float sum = texelFetch(uSearchSad, ivec2(D * nb.x + ci % D, D * nb.y + ci / D), 0).r;
        float cost = sum * (1.0 / 64.0) + uLambda * float(abs(dx) + abs(dy));
        if (ci == 0 || cost < best) { best = cost; bi = ci; }
    }
    return vec2(seed + ivec2(bi % D - R, bi / D - R));
}

void main() {
    ivec2 b = ivec2(gl_FragCoord.xy);
    ivec2 nbs = textureSize(uSeed, 0);
    float xs[9], ys[9];
    int k = 0;
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++) {
            ivec2 c = clamp(b + ivec2(i, j), ivec2(0), nbs - 1);
            vec2 v = bestVec(c);
            xs[k] = v.x; ys[k] = v.y; k++;
        }
    oField = vec4(fgMed9(xs), fgMed9(ys), 0.0, 0.0);
}
