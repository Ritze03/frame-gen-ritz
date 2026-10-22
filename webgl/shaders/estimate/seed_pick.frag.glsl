// Seed choice: candidate 0 (own parent) free, 1..5 pay LAMBDA; strict <, ascending k, first wins.
// search.comp:44-52; PY:143-144.
uniform sampler2D uSeedSad;
uniform sampler2D uParent;   // field[i+1]
uniform sampler2D uMedian;   // median[i]
uniform float uLambda;
layout(location = 0) out vec4 oSeed;

void main() {
    ivec2 blk = ivec2(gl_FragCoord.xy);
    int bi = 0;
    float best = texelFetch(uSeedSad, ivec2(6 * blk.x, blk.y), 0).r * (1.0 / 64.0);
    for (int k = 1; k < 6; k++) {
        float c = texelFetch(uSeedSad, ivec2(6 * blk.x + k, blk.y), 0).r * (1.0 / 64.0) + uLambda;
        if (c < best) { best = c; bi = k; }
    }
    oSeed = vec4(vec2(fgSeedCand(bi, blk, uParent, uMedian)), 0.0, 0.0);
}
