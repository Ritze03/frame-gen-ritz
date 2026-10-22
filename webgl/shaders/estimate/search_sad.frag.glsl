// Integer search: raw 64-px SAD of candidate seed + d for every d in [-R,R]^2, tiled D x D per block.
// Texel (D*bx+cx, D*by+cy) = offset (cx-R, cy-R). search.comp:57-71; PY:116-120.
uniform sampler2D uLumaA;
uniform sampler2D uLumaC;
uniform sampler2D uSeed;
uniform int uRadius;
layout(location = 0) out vec4 oSad;

void main() {
    int D = 2 * uRadius + 1;
    ivec2 f = ivec2(gl_FragCoord.xy);
    ivec2 blk = f / D;
    ivec2 dd = f % D - uRadius;
    ivec2 seed = ivec2(roundEven(texelFetch(uSeed, blk, 0).xy));
    oSad = vec4(fgBlockSad(uLumaA, uLumaC, blk, seed + dd), 0.0, 0.0, 0.0);
}
