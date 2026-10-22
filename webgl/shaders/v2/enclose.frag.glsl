// V2 static mask, enclosure and "overlay near" bits (v2_enclose.comp), per 8x8 cell of the flow grid (= the block grid), on the
// cleaned block field and the cells' S maxima:
//   E: the cell lies between static structure on all four sides, within E_R cells to its left, right, above and below (the inside of an
//      overlay: the tinted, nearly flat background of a HUD panel or chat box between its text lines and its border).
//   C bit 0: E, or static structure in the cell or one of its 8 neighbours; bit 1: a warp tap of a pixel of this cell can land near
//      static structure: the taps lie on the segment c +- 2 v (v the cleaned half-vector of the cell's block, in cells), walked in steps
//      of at most two cells with the box of cells around each (3x3, 5x5 where the cleaned vectors of the 3x3 blocks around stray from v
//      by a cell or more, E_RMAX).
// Writes (E, C as a float 0..3). The library's 8 helper threads per cell are one serial loop here.
uniform sampler2D uS8;   // per-cell maxima of S
uniform sampler2D uVf;   // cleaned block field
layout(location = 0) out vec4 oEC;

const int E_R = 6;
const int E_RMAX = 1;
const int V2_BLOCK = 8;

float s8(ivec2 c) {
    if (any(lessThan(c, ivec2(0))) || any(greaterThanEqual(c, textureSize(uS8, 0)))) return 0.0;
    return texelFetch(uS8, c, 0).r;
}
vec2 VF(ivec2 b) { return texelFetch(uVf, clamp(b, ivec2(0), textureSize(uVf, 0) - 1), 0).xy; }

void main() {
    ivec2 c = ivec2(gl_FragCoord.xy);
    const ivec2 dirs[4] = ivec2[4](ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1));
    float e = 3.0e38;
    for (int h = 0; h < 4; h++) {
        float d = 0.0;
        for (int k = 1; k <= E_R; k++) d = max(d, s8(c + k * dirs[h]));
        e = min(e, d);
    }
    vec2 v = VF(c) * (2.0 / float(V2_BLOCK));   // 2 v in cells
    float dv = 0.0;
    for (int k = 0; k < 9; k++) dv = max(dv, length(VF(c + ivec2(k % 3 - 1, k / 3 - 1)) * (2.0 / float(V2_BLOCK)) - v));
    int R = 1 + min(int(dv), E_RMAX);
    int nbx = (2 * R + 1) * (2 * R + 1);
    int ns = int(ceil(0.5 * max(abs(v.x), abs(v.y)))) + 1;
    float nb = 0.0, tap = 0.0;
    for (int k = 0; k < 9; k++) nb = max(nb, s8(c + ivec2(k % 3 - 1, k / 3 - 1)));
    for (int st = 0; st <= 2 * ns; st++) {
        ivec2 p0 = ivec2(floor(vec2(c) + 0.5 + v * (float(st - ns) / float(ns))));
        for (int of = 0; of < nbx; of++) tap = max(tap, s8(p0 + ivec2(of % (2 * R + 1) - R, of / (2 * R + 1) - R)));
    }
    bool near = max(e, nb) > 0.0;
    oEC = vec4(e, float((near ? 1 : 0) | (tap > 0.0 ? 2 : 0)), 0.0, 0.0);
}
