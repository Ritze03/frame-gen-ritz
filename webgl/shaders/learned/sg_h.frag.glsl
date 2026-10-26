// Static guard, horizontal half of its two filters of s = max-channel |a - c| (learned/sguard.inc.glsl sgDiff), full res,
// replicate border: r = the sum of s over the 3 pixels (x - 1 .. x + 1) of the row, g = the min of s over the 2 SG_HALO + 1 pixels
// (x - SG_HALO .. x + SG_HALO). learned/sg_v.frag.glsl finishes the 3x3 mean and the 9x9 min. Depends on the frames only (once per
// pair). RG16F: for Rgba8 both are integers <= 765, exact.
uniform ivec2 uFrameSize;
uniform sampler2D uPrev;
uniform sampler2D uCurr;
layout(location = 0) out vec4 oH;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float sum = 0.0, mn = 1e30;
    for (int i = -SG_HALO; i <= SG_HALO; i++) {
        float s = sgDiff(uPrev, uCurr, ivec2(clamp(p.x + i, 0, uFrameSize.x - 1), p.y));
        if (i >= -1 && i <= 1) sum += s;
        mn = min(mn, s);
    }
    oH = vec4(sum, mn, 0.0, 0.0);
}
