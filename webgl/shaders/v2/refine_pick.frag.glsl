// V2 Performance refinement, pick (v2_repick.comp step 2): the 8 neighbours in row-major order; a candidate replaces the running best when
// its raw SAD is strictly smaller than the running best (so the first of equals wins). Writes xy = vector, z = raw 16-px SAD,
// w = roundEven(256 * mean SAD) / 256 (scene-cut term).
uniform sampler2D uRepSad;
uniform sampler2D uBase;
uniform float uStep;
layout(location = 0) out vec4 oRep;

void main() {
    ivec2 blk = ivec2(gl_FragCoord.xy);
    vec4 b = texelFetch(uBase, blk, 0);
    vec2 v0 = b.xy, v = v0;
    float c0 = b.z;
    for (int j = 0; j < 9; j++) {
        if (j == 4) continue;
        float s = texelFetch(uRepSad, ivec2(3 * blk.x + j % 3, 3 * blk.y + j / 3), 0).r;
        if (s < c0) { c0 = s; v = v0 + uStep * vec2(j % 3 - 1, j / 3 - 1); }
    }
    oRep = vec4(v, c0, roundEven(c0 * 16.0) * (1.0 / 256.0));
}
