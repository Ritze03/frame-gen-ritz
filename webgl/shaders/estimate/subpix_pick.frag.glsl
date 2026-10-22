// Sub-pixel argmin: cost = SAD/64 + LAMBDA*step*|d|_1, strict <, row-major. xy = base + d_best*step,
// z = winning mean SAD. subpix.comp:50-65.
uniform sampler2D uSubSad;
uniform sampler2D uBase;
uniform float uLambda;
uniform float uStep;
layout(location = 0) out vec4 oSub;

void main() {
    ivec2 blk = ivec2(gl_FragCoord.xy);
    int bi = 0;
    float best = 0.0, bestSad = 0.0;
    for (int ci = 0; ci < 9; ci++) {
        vec2 d = vec2(ci % 3 - 1, ci / 3 - 1);
        float sad = texelFetch(uSubSad, ivec2(3 * blk.x + ci % 3, 3 * blk.y + ci / 3), 0).r * (1.0 / 64.0);
        float cost = sad + uLambda * uStep * (abs(d.x) + abs(d.y));
        if (ci == 0 || cost < best) { best = cost; bestSad = sad; bi = ci; }
    }
    vec2 base = texelFetch(uBase, blk, 0).xy;
    oSub = vec4(base + vec2(bi % 3 - 1, bi / 3 - 1) * uStep, bestSad, 0.0);
}
