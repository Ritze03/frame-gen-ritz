// V2 Quality, the per-pixel field (flow resolution, once per estimate; gpu/shaders/v2_qfield.comp):
//   1. the cleaned block field upsampled with edge awareness: each flow pixel blends the 4 block vectors around it (the bilinear
//      corners) with weights (bilinear + Q_FB) * exp(-(e_k - e_min) / (Q_FS + Q_FK * e_min)), e_k the mean
//      |luma_a(m + o - v_k) - luma_c(m + o + v_k)| over its 3x3; where the 4 vectors agree (within Q_FSAME) the bilinear blend as is;
//   2. the static-overlay mask z: where the field moves and the pixel has structure, the zero vector's mismatch with its 3x3 mean
//      removed, relative to the structure. z >= 0.5: the field is zero there.
// Output v2.qf: xy = the per-pixel half-vector (flow px), z = the mask (unquantised; v2/qcov packs it).
uniform sampler2D uLumaA;   // level-0 luma of prev / curr
uniform sampler2D uLumaC;
uniform sampler2D uVf;      // cleaned block field
layout(location = 0) out vec4 oQf;

vec2 VF(ivec2 b) { return texelFetch(uVf, clamp(b, ivec2(0), textureSize(uVf, 0) - 1), 0).xy; }

float err(vec2 m, vec2 v) {
    float e = 0.0;
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++) {
            vec2 p = m + vec2(i, j);
            e += abs(qLuma(uLumaA, p - v) - qLuma(uLumaC, p + v));
        }
    return e * (1.0 / 9.0);
}

float err0(ivec2 m) {
    float d[9];
    float s = 0.0;
    for (int k = 0; k < 9; k++) {
        ivec2 p = m + ivec2(k % 3 - 1, k / 3 - 1);
        d[k] = qL(uLumaA, p) - qL(uLumaC, p);
        s += d[k];
    }
    s *= 1.0 / 9.0;
    float e = 0.0;
    for (int k = 0; k < 9; k++) e += abs(d[k] - s);
    return e * (1.0 / 9.0);
}

void main() {
    ivec2 m = ivec2(gl_FragCoord.xy);
    vec2 s = (vec2(m) + 0.5) * (1.0 / float(Q_VBLOCK)) - 0.5;
    vec2 fl = floor(s);
    vec2 f = s - fl;
    ivec2 i = ivec2(fl);
    vec2 v[4] = vec2[4](VF(i), VF(i + ivec2(1, 0)), VF(i + ivec2(0, 1)), VF(i + ivec2(1, 1)));
    float b[4] = float[4]((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
    vec2 lo = min(min(v[0], v[1]), min(v[2], v[3])), hi = max(max(v[0], v[1]), max(v[2], v[3]));
    vec2 h = b[0] * v[0] + b[1] * v[1] + b[2] * v[2] + b[3] * v[3];
    if (max(hi.x - lo.x, hi.y - lo.y) > Q_FSAME) {
        float e[4];
        float emin = 1e9;
        for (int k = 0; k < 4; k++) { e[k] = err(vec2(m), v[k]); emin = min(emin, e[k]); }
        float inv = 1.0 / (Q_FS + Q_FK * emin);
        vec2 acc = vec2(0.0);
        float ws = 0.0;
        for (int k = 0; k < 4; k++) {
            float w = (b[k] + Q_FB) * exp(-(e[k] - emin) * inv);
            acc += w * v[k];
            ws += w;
        }
        h = acc / ws;
    }
    float z = 0.0;
    if (max(max(abs(lo.x), abs(hi.x)), max(abs(lo.y), abs(hi.y))) > Q_ZMOVE) {
        float mn = 1e9, mx = -1e9;
        for (int k = 0; k < 9; k++) {
            float l = qL(uLumaA, m + ivec2(k % 3 - 1, k / 3 - 1));
            mn = min(mn, l);
            mx = max(mx, l);
        }
        float con = mx - mn;
        if (con >= Q_ZCON) z = clamp((Q_ZR1 - err0(m) / con) * (1.0 / (Q_ZR1 - Q_ZR0)), 0.0, 1.0);
    }
    if (z >= 0.5) h = vec2(0.0);
    oQf = vec4(h, z, 0.0);
}
