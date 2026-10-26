// First step of the duplicate-pair sum (ml_common.inc.glsl mlDup): per uFan x uFan tile of full-res pixels, the sum of |a - c| over
// R, G, B in levels (Rgba8: integer levels; RgbaF16: levels of the perceptual encoding, FG_D). reduce_sad sums the rest to 1x1.
// proto/harness.py:32 err() / gpu/compare.py mad(): np.abs(a - c) averaged over channels and pixels.
uniform sampler2D uPrev;
uniform sampler2D uCurr;
uniform int uFan;
layout(location = 0) out vec4 oSum;

void main() {
    ivec2 o = ivec2(gl_FragCoord.xy);
    float sum = 0.0;
    for (int j = 0; j < uFan; j++)
        for (int i = 0; i < uFan; i++) {
            ivec2 p = uFan * o + ivec2(i, j);
            if (p.x < uFrameSize.x && p.y < uFrameSize.y) {
                vec3 d = FG_D(fgPx(uPrev, p), fgPx(uCurr, p));
                sum += (d.r + d.g) + d.b;
            }
        }
    oSum = vec4(sum);
}
