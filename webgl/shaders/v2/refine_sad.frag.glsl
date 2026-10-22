// V2 Performance refinement, SADs (v2_repick.comp step 2): the 8 neighbours at +-uStep of the block's current vector (texel
// (3bx+cx, 3by+cy) = offset (cx-1, cy-1)*uStep; the centre texel is computed but not used), on the 16-pixel quincunx. Raw sum of 16.
uniform sampler2D uLumaA;
uniform sampler2D uLumaC;
uniform sampler2D uBase;   // rep0 / rep1: xy = vector
uniform float uStep;
layout(location = 0) out vec4 oSad;

void main() {
    ivec2 f = ivec2(gl_FragCoord.xy);
    ivec2 blk = f / 3;
    vec2 v = texelFetch(uBase, blk, 0).xy + uStep * vec2(f % 3 - 1);
    oSad = vec4(fgQuinSad(uLumaA, uLumaC, blk, v), 0.0, 0.0, 0.0);
}
