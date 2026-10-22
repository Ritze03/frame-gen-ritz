// Seed candidate costs: raw 64-px SAD of each of the 6 seed candidates of a block. Texel x = 6*bx + k.
// search.comp:35-37; PY:143 (block_sad).
uniform sampler2D uLumaA;
uniform sampler2D uLumaC;
uniform sampler2D uParent;   // field[i+1]
uniform sampler2D uMedian;   // median[i]
layout(location = 0) out vec4 oSad;

void main() {
    ivec2 f = ivec2(gl_FragCoord.xy);
    ivec2 blk = ivec2(f.x / 6, f.y);
    int k = f.x % 6;
    ivec2 cand = fgSeedCand(k, blk, uParent, uMedian);
    oSad = vec4(fgBlockSad(uLumaA, uLumaC, blk, cand), 0.0, 0.0, 0.0);
}
