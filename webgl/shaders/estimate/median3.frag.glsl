// 3x3 per-component median of the raw block field (replicate border): median3.comp:15-27; PY:128. fieldRaw[i] -> field[i].
uniform sampler2D uRaw;
layout(location = 0) out vec4 oField;

void main() {
    ivec2 b = ivec2(gl_FragCoord.xy);
    ivec2 mx = textureSize(uRaw, 0) - 1;
    float xs[9], ys[9];
    for (int k = 0; k < 9; k++) {
        vec2 v = texelFetch(uRaw, clamp(b + ivec2(k % 3 - 1, k / 3 - 1), ivec2(0), mx), 0).xy;
        xs[k] = v.x; ys[k] = v.y;
    }
    oField = vec4(fgMed9(xs), fgMed9(ys), 0.0, 0.0);
}
