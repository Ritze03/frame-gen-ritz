// Sum reduction with fixed 16x16 fan-in over component uChan (replaces the u32 fixed-point atomic of
// subpix.comp:67). Output size = ceil(src/16) per axis.
uniform sampler2D uSrc;
uniform ivec2 uSrcSize;
uniform int uChan;
layout(location = 0) out vec4 oSum;

void main() {
    ivec2 o = ivec2(gl_FragCoord.xy);
    float sum = 0.0;
    for (int j = 0; j < 16; j++)
        for (int i = 0; i < 16; i++) {
            ivec2 p = 16 * o + ivec2(i, j);
            if (p.x < uSrcSize.x && p.y < uSrcSize.y) sum += texelFetch(uSrc, p, 0)[uChan];
        }
    oSum = vec4(sum);
}
