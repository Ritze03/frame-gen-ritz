// V2 Performance block re-pick, pick (v2_repick.comp step 1): cost = SAD/16 + LAMBDA for a neighbour's vector (own: none), strict <, own
// first (own wins ties). Writes xy = winning vector, z = its raw 16-px SAD (the refinement's running best), w = roundEven(256 * mean
// SAD) / 256 (this block's term of the scene-cut sum, the library's 1/256 fixed point).
uniform sampler2D uRepSad;
uniform sampler2D uVfit;
uniform float uLambda;
layout(location = 0) out vec4 oRep;

void main() {
    ivec2 blk = ivec2(gl_FragCoord.xy);
    float s[9];
    for (int k = 0; k < 9; k++) s[k] = texelFetch(uRepSad, ivec2(3 * blk.x + k % 3, 3 * blk.y + k / 3), 0).r;
    int bi = 0;
    float best = 0.0;
    for (int k = 0; k < 9; k++) {
        float c = s[k] * (1.0 / 16.0) + (k == 0 ? 0.0 : uLambda);
        if (k == 0 || c < best) { best = c; bi = k; }
    }
    oRep = vec4(fgRepCand(bi, blk, uVfit), s[bi], roundEven(s[bi] * 16.0) * (1.0 / 256.0));
}
