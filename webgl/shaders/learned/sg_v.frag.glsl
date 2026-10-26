// Static guard, vertical half (learned/sg_h.frag.glsl): r = d = the 3x3 mean of s (cv2.blur, replicate border), g = dMin = the min
// of s over the (2 SG_HALO + 1)^2 window (cv2.erode; a replicated tap repeats a pixel of the window, so it is the min over the
// pixels inside the frame). RG16F, read by the 'final_sg' variant of synth/final.frag.glsl.
uniform ivec2 uFrameSize;
uniform sampler2D uH;
layout(location = 0) out vec4 oD;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float sum = 0.0, mn = 1e30;
    for (int j = -SG_HALO; j <= SG_HALO; j++) {
        vec2 h = texelFetch(uH, ivec2(p.x, clamp(p.y + j, 0, uFrameSize.y - 1)), 0).rg;
        if (j >= -1 && j <= 1) sum += h.r;
        mn = min(mn, h.g);
    }
    oD = vec4(sum * (1.0 / 9.0), mn, 0.0, 0.0);
}
