// V2 Performance field clean-up (v2_clean.comp): a 3x3 vector median of the re-picked block field (the neighbour whose summed L1 distance
// to the other 8 is smallest, the centre wins ties), then a light 3x3 bilateral smoothing ([1 2 1]^2 spatial weights times
// 1 / (1 + |v - v_centre|^2 / sigma^2), sigma = 1 flow px of half-vector). The medians of the 8 neighbours are taken from the raw field
// with replicate clamping at THEIR positions (also outside the grid), as the library's tile does.
uniform sampler2D uField;   // rep2: xy
layout(location = 0) out vec4 oVf;

const float SIGMA2_INV = 1.0;   // 1 / kV2CleanSigma^2

vec2 rawAt(ivec2 b) { return texelFetch(uField, clamp(b, ivec2(0), textureSize(uField, 0) - 1), 0).xy; }

vec2 medAt(ivec2 c) {
    vec2 v[9];
    v[0] = rawAt(c);
    int k = 1;
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++)
            if (i != 0 || j != 0) v[k++] = rawAt(c + ivec2(i, j));
    int best = 0;
    float bestC = 0.0;
    for (int ci = 0; ci < 9; ci++) {
        float s = 0.0;
        for (int o = 0; o < 9; o++) {
            vec2 d = abs(v[o] - v[ci]);
            s += d.x + d.y;
        }
        if (ci == 0 || s < bestC) { bestC = s; best = ci; }
    }
    return v[best];
}

void main() {
    ivec2 b = ivec2(gl_FragCoord.xy);
    vec2 m = medAt(b);
    vec2 acc = vec2(0.0);
    float ws = 0.0;
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++) {
            vec2 v = medAt(b + ivec2(i, j));
            vec2 d = v - m;
            float w = float((2 - abs(i)) * (2 - abs(j))) / (1.0 + dot(d, d) * SIGMA2_INV);
            acc += w * v;
            ws += w;
        }
    oVf = vec4(acc / ws, 0.0, 0.0);
}
