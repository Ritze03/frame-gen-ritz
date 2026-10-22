// Per-pixel candidate errors e_k = |A(q - v_k) - C(q + v_k)| at flow res, k = 0..6; each window pixel uses its
// own candidate vector. pixfield.comp:41-45; PY:177-180.
uniform sampler2D uLumaA;
uniform sampler2D uLumaC;
uniform sampler2D uBlockField;   // sub2
layout(location = 0) out vec4 oErr0;
layout(location = 1) out vec4 oErr1;

float err(int k, ivec2 q) {
    vec2 v = fgPixCand(k, q, uBlockField);
    return abs(fgSampleL(uLumaA, vec2(q) - v) - fgSampleL(uLumaC, vec2(q) + v));
}

void main() {
    ivec2 q = ivec2(gl_FragCoord.xy);
    oErr0 = vec4(err(0, q), err(1, q), err(2, q), err(3, q));
    oErr1 = vec4(err(4, q), err(5, q), err(6, q), 0.0);
}
