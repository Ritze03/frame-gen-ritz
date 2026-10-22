// Sum reduction with uFan x uFan fan-in over component uChan (replaces the u32 fixed-point atomic of
// subpix.comp:67). Output size = ceil(src/uFan) per axis.
uniform sampler2D uSrc;
uniform ivec2 uSrcSize;
uniform int uFan;   // fan-in per axis (16; 8 for the first step over the full-res trust)
uniform int uChan;
layout(location = 0) out vec4 oSum;

void main() {
    ivec2 o = ivec2(gl_FragCoord.xy);
    float sum = 0.0;
    for (int j = 0; j < uFan; j++)
        for (int i = 0; i < uFan; i++) {
            ivec2 p = uFan * o + ivec2(i, j);
            if (p.x < uSrcSize.x && p.y < uSrcSize.y) sum += texelFetch(uSrc, p, 0)[uChan];
        }
    oSum = vec4(sum);
}
