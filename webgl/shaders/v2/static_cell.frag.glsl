// V2 static-structure mask, step 3 (v2_static.comp, per 8x8 cell of the flow grid = the block grid): the cell's largest S, if at least
// S_CELL of its pixels are in the mask (S >= 0.5: a stroke, an edge of a HUD element; not a stray match or two of a moving texture), else 0.
// The input of v2/enclose. (S >= 0, so the maximum of the floats is the maximum of their bit patterns the library compares.)
uniform sampler2D uSmask;
layout(location = 0) out vec4 oCell;

const float S_CELL = 4.0;

void main() {
    ivec2 c = ivec2(gl_FragCoord.xy);
    ivec2 sz = textureSize(uSmask, 0);
    float mx = 0.0, cnt = 0.0;
    for (int j = 0; j < 8; j++)
        for (int i = 0; i < 8; i++) {
            ivec2 m = 8 * c + ivec2(i, j);
            if (m.x < sz.x && m.y < sz.y) {
                float s = texelFetch(uSmask, m, 0).r;
                if (s > 0.0) mx = max(mx, s);
                if (s >= 0.5) cnt += 1.0;
            }
        }
    oCell = vec4(cnt >= S_CELL ? mx : 0.0, 0.0, 0.0, 0.0);
}
