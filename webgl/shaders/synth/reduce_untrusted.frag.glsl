// synth/reduce_untrusted: uFan x uFan fan-in sum reduction. Step 0 (uMode = 1) counts pixels with trust > 0.5;
// later steps (uMode = 0) sum the previous partial counts. Final 1x1 = untrusted pixel count
// (resolve.comp:43; interp.py:238 (u>0.5).mean()). Counts stay exact in fp32 (< 2^24).
uniform sampler2D uSrc;
uniform ivec2 uSrcSize;
uniform int uFan;   // fan-in per axis (16; 8 for the first step over the full-res trust)
uniform int uMode;

layout(location = 0) out vec4 oSum;

void main() {
    ivec2 o = ivec2(gl_FragCoord.xy);
    float sum = 0.0;
    for (int j = 0; j < uFan; j++) {
        for (int i = 0; i < uFan; i++) {
            ivec2 p = uFan * o + ivec2(i, j);
            if (p.x < uSrcSize.x && p.y < uSrcSize.y) {
                float v = texelFetch(uSrc, p, 0).r;
                sum += (uMode != 0) ? (v > 0.5 ? 1.0 : 0.0) : v;
            }
        }
    }
    oSum = vec4(sum);
}
