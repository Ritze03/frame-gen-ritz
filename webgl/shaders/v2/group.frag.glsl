// V2 Performance synthesis, step 1 (one fragment per aligned 4x4 pixel group): the choice of the block-vector candidate. The library does
// it per 4x4 lane group with subgroup shuffles (v2_morph.glsl sum16): the group sums the taps' mismatch em of its 16 pixels (s0) and, for
// each of the 4 block vectors around the content's place, the candidate's mismatch (sk); the smallest sum wins (strict <, k ascending,
// the pixel's own taps keep ties). Output: x = s0, y = the winning sum sb, z = winner index + 1 (0 = the pixel's own taps win), w = 1
// for a group that the frame's right / bottom edge cuts (v2/morph decides those per pixel, as the library does).
// The candidates are tried only where the library's subgroup vote says so: some pixel of the 8x8 tile has em > V2_C0 (tileLit).
layout(location = 0) out vec4 oGroup;

void main() {
    ivec2 g0 = 4 * ivec2(gl_FragCoord.xy);
    if (g0.x + 3 >= uFrameSize.x || g0.y + 3 >= uFrameSize.y) { oGroup = vec4(0.0, 0.0, 0.0, 1.0); return; }
    float s0 = 0.0;
    for (int i = 0; i < 16; i++) s0 += texelFetch(uEm, g0 + ivec2(i & 3, i >> 2), 0).r;
    float sb = s0;
    int win = 0;
    if (tileLit(g0)) {
        bool nearT = (sCellBits(sCell(vec2(g0))) & 2u) != 0u;
        vec2 gv[16];
        for (int i = 0; i < 16; i++) gv[i] = gAt(vec2(g0 + ivec2(i & 3, i >> 2)), uT);
        for (int k = 0; k < 4; k++) {
            float sk = 0.0;
            for (int i = 0; i < 16; i++) sk += candTap(vec2(g0 + ivec2(i & 3, i >> 2)), uT, gv[i], nearT, k).ek;
            if (sk < sb) { sb = sk; win = k + 1; }
        }
    }
    oGroup = vec4(s0, sb, float(win), 0.0);
}
