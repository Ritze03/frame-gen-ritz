// One 3x3 conv layer of the net (SPEC stage 4c/4d; reference.py conv3x3 + level_net), or the part of it that writes ML_COUT RGBA
// textures (4 channels each). Compiled per shape by the host's #defines:
//   ML_CIN   input textures (RGBA, 4 channels each; the 14-channel input is 4 with 2 zero channels), 1..6
//   ML_COUT  output textures of this pass, 1..3 (a layer of width 24 is two passes)
//   ML_LAST  the layer's linear 2-channel output: writes h + 2 * out (h = 4 * uX3.xy, the match pass's hx / 4, hy / 4) instead of
//            clamp(out, 0, 1)
// out[o](q) = b[o] + sum over taps (kx, ky), inputs i of W[o][i][ky][kx] * in[i](q + (kx - 1, ky - 1)), zero padding outside the
// level (uSize: the textures are level-0 sized, the region beyond the level holds other levels' data and is never read).
// Weights (uniform block MlW, bound per pass by the host from the net's buffer): uW[(o * 9 + tap) * ML_CIN + j] is the 4x4 block
// from input texture j to output texture o at tap = ky * 3 + kx, column c = input channel 4j + c, row r = output channel 4o + r
// (zero beyond the layer's channels); uB[o] the biases.
uniform ivec2 uSize;
uniform sampler2D uIn0;
#if ML_CIN > 1
uniform sampler2D uIn1;
#endif
#if ML_CIN > 2
uniform sampler2D uIn2;
#endif
#if ML_CIN > 3
uniform sampler2D uIn3;
#endif
#if ML_CIN > 4
uniform sampler2D uIn4;
#endif
#if ML_CIN > 5
uniform sampler2D uIn5;
#endif
#if ML_LAST
uniform sampler2D uX3;
#endif
layout(std140) uniform MlW {
    mat4 uW[ML_COUT * 9 * ML_CIN];
    vec4 uB[ML_COUT];
};
#if ML_LAST
layout(location = 0) out vec4 oH;
#else
layout(location = 0) out vec4 o0;
#if ML_COUT > 1
layout(location = 1) out vec4 o1;
#endif
#if ML_COUT > 2
layout(location = 2) out vec4 o2;
#endif
#endif

#define ML_TAP(J, S) { vec4 v = texelFetch(S, q, 0); for (int o = 0; o < ML_COUT; o++) acc[o] += uW[(o * 9 + t) * ML_CIN + J] * v; }

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    if (mlDup()) {
#if ML_LAST
        oH = vec4(0.0);
#else
        o0 = vec4(0.0);
#if ML_COUT > 1
        o1 = vec4(0.0);
#endif
#if ML_COUT > 2
        o2 = vec4(0.0);
#endif
#endif
        return;
    }
    vec4 acc[ML_COUT];
    for (int o = 0; o < ML_COUT; o++) acc[o] = uB[o];
    for (int t = 0; t < 9; t++) {
        ivec2 q = p + ivec2(t % 3 - 1, t / 3 - 1);
        if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, uSize))) continue;   // zero padding
        ML_TAP(0, uIn0)
#if ML_CIN > 1
        ML_TAP(1, uIn1)
#endif
#if ML_CIN > 2
        ML_TAP(2, uIn2)
#endif
#if ML_CIN > 3
        ML_TAP(3, uIn3)
#endif
#if ML_CIN > 4
        ML_TAP(4, uIn4)
#endif
#if ML_CIN > 5
        ML_TAP(5, uIn5)
#endif
    }
#if ML_LAST
    vec2 h = 4.0 * texelFetch(uX3, p, 0).xy;
    oH = vec4(h + 2.0 * acc[0].xy, 0.0, 0.0);
#else
    o0 = clamp(acc[0], 0.0, 1.0);
#if ML_COUT > 1
    o1 = clamp(acc[1], 0.0, 1.0);
#endif
#if ML_COUT > 2
    o2 = clamp(acc[2], 0.0, 1.0);
#endif
#endif
}
