// Flow-res luma: cv2 BGR2GRAY (rounded to uint8; RgbaF16: the same weights on its levels, not rounded) then INTER_AREA 1/S = mean of the valid pixels of each S x S
// cell. luma.comp:22-25,61-70; PY:150.
uniform sampler2D uFrame;
uniform ivec2 uFrameSize;
uniform int uFlowScale;
layout(location = 0) out vec4 oLuma;

void main() {
    ivec2 o = ivec2(gl_FragCoord.xy);
    float sum = 0.0;
    int n = 0;
    for (int j = 0; j < uFlowScale; j++)
        for (int i = 0; i < uFlowScale; i++) {
            ivec2 p = uFlowScale * o + ivec2(i, j);
            if (p.x < uFrameSize.x && p.y < uFrameSize.y) {
                sum += fgLumaPx(uFrame, p);   // Rgba8: the integer luma (fgLuma8); RgbaF16: of the levels, unrounded
                n++;
            }
        }
    oLuma = vec4(sum / float(n));
}
