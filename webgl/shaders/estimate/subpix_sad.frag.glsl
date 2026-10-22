// Sub-pixel refinement SADs: 3x3 offsets (d * uStep) around the base vector, bilinear taps (1/32 quantised),
// per row sums then sum of rows. Texel (3bx+cx, 3by+cy) = offset (cx-1, cy-1)*step. subpix.comp:38-56; PY:162-163.
uniform sampler2D uLumaA;
uniform sampler2D uLumaC;
uniform sampler2D uBase;   // field[0] or sub1
uniform float uStep;
layout(location = 0) out vec4 oSad;

void main() {
    ivec2 f = ivec2(gl_FragCoord.xy);
    ivec2 blk = f / 3;
    vec2 d = vec2(f % 3 - 1);
    vec2 h = texelFetch(uBase, blk, 0).xy + d * uStep;
    float sum = 0.0;
    for (int y = 0; y < 8; y++) {
        float row = 0.0;
        for (int x = 0; x < 8; x++) {
            vec2 q = vec2(blk * BLOCK + ivec2(x, y));
            row += abs(fgSampleL(uLumaA, q - h) - fgSampleL(uLumaC, q + h));
        }
        sum += row;
    }
    oSad = vec4(sum, 0.0, 0.0, 0.0);
}
