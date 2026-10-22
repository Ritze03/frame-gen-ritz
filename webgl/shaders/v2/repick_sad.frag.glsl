// V2 Performance block re-pick, SADs (v2_repick.comp step 1): texel (3bx+cx, 3by+cy) = candidate k = 3cy+cx of block (bx,by): k = 0 the
// block's own fitted vector, k = 1..8 its 8 neighbours' (library order, see fgRepCand), each snapped to the quarter-pel grid and
// measured on the block's 16-pixel quincunx. Raw sum of 16.
uniform sampler2D uLumaA;
uniform sampler2D uLumaC;
uniform sampler2D uVfit;
layout(location = 0) out vec4 oSad;

void main() {
    ivec2 f = ivec2(gl_FragCoord.xy);
    ivec2 blk = f / 3;
    int k = (f.y % 3) * 3 + f.x % 3;
    oSad = vec4(fgQuinSad(uLumaA, uLumaC, blk, fgRepCand(k, blk, uVfit)), 0.0, 0.0, 0.0);
}
