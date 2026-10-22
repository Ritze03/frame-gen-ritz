// Search argmin per block (search.comp:63-79; PY:121-127): the winning vector seed + d of the D x D candidates,
// cost = SAD/64 + LAMBDA*|d|_1, strict <, row-major (dy outer), first wins. Writes the raw (un-medianed) block field fieldRaw[i];
// estimate/median3 takes the 3x3 median (median3.comp) into field[i].
uniform sampler2D uSearchSad;
uniform sampler2D uSeed;
uniform float uLambda;
uniform int uRadius;
layout(location = 0) out vec4 oField;

void main() {
    ivec2 b = ivec2(gl_FragCoord.xy);
    int R = uRadius, D = 2 * R + 1;
    ivec2 seed = ivec2(roundEven(texelFetch(uSeed, b, 0).xy));
    int bi = 0;
    float best = 0.0;
    for (int ci = 0; ci < D * D; ci++) {
        int dx = ci % D - R, dy = ci / D - R;
        float sum = texelFetch(uSearchSad, ivec2(D * b.x + ci % D, D * b.y + ci / D), 0).r;
        float cost = sum * (1.0 / 64.0) + uLambda * float(abs(dx) + abs(dy));
        if (ci == 0 || cost < best) { best = cost; bi = ci; }
    }
    oField = vec4(vec2(seed + ivec2(bi % D - R, bi / D - R)), 0.0, 0.0);
}
