// MotionVersion::V3, Settings::v3Blend (experimental; gpu/V3.md 3.16 "As built (learned blend)"): the blend weights from a small per-pixel net instead of
// the rule. Included by v3_synth.comp only in the -DV3B=1 build, after v3_weights.glsl. Per output pixel, all colours / 255, m(v) = 4 mean|v|:
//   x[0..11] c0..c3 (CAp CAc CBp CBc), x[12..17] oA holeA uA oB holeB uB, x[18] t, x[19..20] min(len, 256) / 64, x[21..24] m(c0-c1) m(c2-c3) m(c0-c2) m(c1-c3)
//   six-choice net only: x[25..27] e0 = prev(x), x[28..30] e1 = curr(x), x[31] m(e0-e1), x[32] m(c0-e0), x[33] m(c3-e1).
// h = lrelu(W1 x + b1) (H 8, h = max(z, 0.1 z)), logits z2 = W2 h + b2 (K 4 or 6), w = softmax(p + z2) with the prior p_k = log(max(r_k, 1e-3)) (r = the rule's
// weights, v3Weights unchanged), p_4 = p_5 = -5. The weights are in v3_blend_weights.glsl (generated: gpu/ml/blend/export_glsl.py).
#include "v3_blend_weights.glsl"
float v3bM(vec3 v) { return (4.0 / 3.0) * (abs(v.r) + abs(v.g) + abs(v.b)); }
// var = 1 + 3 * weight set + net: net 0, 1 (var 1, 2, 4, 5, 7, ..): four choices (w[0..3] of o replaced, w4 = w5 = 0); net 2 (var 3, 6, 9, ..): six choices (o.w is renormalised over the four taps, w4, w5 the shares of e0, e1; dw = the four
// weights before the renormalisation, for the debug dump). e0 / e1: the working values of prev / curr at the pixel (levels).
void v3BlendNet(in V3WIn i, vec3 e0, vec3 e1, int var, inout V3WOut o, out float w4, out float w5, out float dw[4]) {
    const int v = var - 1;            // the weight index (< V3B_NETS): 3 * weight set + net, net 0, 1, 2 = A, B, C
    const int nv = (var - 1) % 3 + 1;  // the net structure: 1, 2 = four choices, 3 = six choices
    const float k255 = 1.0 / 255.0;
    const vec3 c0 = i.CAp * k255, c1 = i.CAc * k255, c2 = i.CBp * k255, c3 = i.CBc * k255;
    float x[25];
    x[0] = c0.r; x[1] = c0.g; x[2] = c0.b; x[3] = c1.r; x[4] = c1.g; x[5] = c1.b;
    x[6] = c2.r; x[7] = c2.g; x[8] = c2.b; x[9] = c3.r; x[10] = c3.g; x[11] = c3.b;
    x[12] = i.oA; x[13] = i.holeA; x[14] = i.uA; x[15] = i.oB; x[16] = i.holeB; x[17] = i.uB;
    x[18] = i.t;
    x[19] = min(i.lenA, 256.0) * (1.0 / 64.0); x[20] = min(i.lenB, 256.0) * (1.0 / 64.0);
    x[21] = v3bM(c0 - c1); x[22] = v3bM(c2 - c3); x[23] = v3bM(c0 - c2); x[24] = v3bM(c1 - c3);
    float xe[9];
    const vec3 f0 = e0 * k255, f1 = e1 * k255;
    xe[0] = f0.r; xe[1] = f0.g; xe[2] = f0.b; xe[3] = f1.r; xe[4] = f1.g; xe[5] = f1.b;
    xe[6] = v3bM(f0 - f1); xe[7] = v3bM(c0 - f0); xe[8] = v3bM(c3 - f1);
    float h[8];
    for (int j = 0; j < 8; j++) {
        float z = V3B_B1[v * 8 + j];
        for (int n = 0; n < 25; n++) z += V3B_W1[(v * 8 + j) * 34 + n] * x[n];
        if (nv == 3) for (int n = 0; n < 9; n++) z += V3B_W1[(v * 8 + j) * 34 + 25 + n] * xe[n];
        h[j] = max(z, 0.1 * z);
    }
    float l[6];
    const float r[4] = float[4](o.w[0], o.w[1], o.w[2], o.w[3]);
    for (int k = 0; k < 6; k++) {
        float z = V3B_B2[v * 6 + k];
        for (int j = 0; j < 8; j++) z += V3B_W2[(v * 6 + k) * 8 + j] * h[j];
        l[k] = z + (k < 4 ? log(max(r[k], 1e-3)) : -5.0);
    }
    float mx = max(max(l[0], l[1]), max(l[2], l[3]));
    if (nv == 3) mx = max(mx, max(l[4], l[5]));
    float e[6];
    float sum = 0.0;
    for (int k = 0; k < 6; k++) {
        e[k] = (k < 4 || nv == 3) ? exp(l[k] - mx) : 0.0;
        sum += e[k];
    }
    const float inv = 1.0 / sum;
    for (int k = 0; k < 4; k++) dw[k] = e[k] * inv;
    w4 = e[4] * inv;
    w5 = e[5] * inv;
    if (nv == 3) {
        const float s4 = 1.0 / max(dw[0] + dw[1] + dw[2] + dw[3], 1e-12);
        for (int k = 0; k < 4; k++) o.w[k] = dw[k] * s4;
    } else {
        for (int k = 0; k < 4; k++) o.w[k] = dw[k];
    }
}
