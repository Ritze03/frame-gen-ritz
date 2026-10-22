// Pyramid level = INTER_AREA 0.5 of the previous level: mean of the valid pixels of each 2x2 cell.
// down.comp:11-24; PY:76.
uniform sampler2D uSrc;
uniform ivec2 uSrcSize;
layout(location = 0) out vec4 oLuma;

void main() {
    ivec2 o = ivec2(gl_FragCoord.xy);
    float sum = 0.0;
    int n = 0;
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
            ivec2 p = 2 * o + ivec2(i, j);
            if (p.x < uSrcSize.x && p.y < uSrcSize.y) {
                sum += texelFetch(uSrc, p, 0).r;
                n++;
            }
        }
    float s = n == 4 ? 0.25 : 1.0 / float(n);
    oLuma = vec4(sum * s);
}
