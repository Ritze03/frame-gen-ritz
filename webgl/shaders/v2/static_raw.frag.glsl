// V2 static-structure mask, step 1 (v2_static.comp rawAt(), flow resolution): where both real frames show the same STRUCTURE at a pixel,
// whatever changes behind it. Over the 4 neighbours n (right, left, down, up; replicate clamp) the luma gradients ga = a(m) - a(n) and
// gc = c(m) - c(n): the structure agrees when sum |ga - gc| is small against sum max(|ga|, |gc|); it needs gradients (D0) and a bend
// (the smaller frame's sum of second differences, L0), i.e. an edge or a stroke, not a flat place or a ramp.
uniform sampler2D uLumaA;
uniform sampler2D uLumaC;
layout(location = 0) out vec4 oRaw;

const float S_EPS = 8.0;
const float S_R0 = 0.2, S_R1 = 0.4;
const float S_D0 = 16.0, S_D1 = 24.0;
const float S_L0 = 8.0, S_L1 = 16.0;

void main() {
    ivec2 m = ivec2(gl_FragCoord.xy);
    float a0 = fgFetchL(uLumaA, m), c0 = fgFetchL(uLumaC, m);
    const ivec2 o[4] = ivec2[4](ivec2(1, 0), ivec2(-1, 0), ivec2(0, 1), ivec2(0, -1));
    float num = 0.0, den = 0.0;
    float ga[4], gc[4];
    for (int k = 0; k < 4; k++) {
        ga[k] = a0 - fgFetchL(uLumaA, m + o[k]);
        gc[k] = c0 - fgFetchL(uLumaC, m + o[k]);
        num += abs(ga[k] - gc[k]);
        den += max(abs(ga[k]), abs(gc[k]));
    }
    float r = num * (1.0 / (den + S_EPS));
    float lap = min(abs(ga[0] + ga[1]) + abs(ga[2] + ga[3]), abs(gc[0] + gc[1]) + abs(gc[2] + gc[3]));
    float raw = clamp((S_R1 - r) * (1.0 / (S_R1 - S_R0)), 0.0, 1.0) * clamp((den - S_D0) * (1.0 / S_D1), 0.0, 1.0) *
                clamp((lap - S_L0) * (1.0 / S_L1), 0.0, 1.0);
    oRaw = vec4(raw, 0.0, 0.0, 0.0);
}
