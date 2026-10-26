// First step of the net's SAD (SPEC stage 6, reference.py flow_sad): per uFan x uFan tile of flow pixels q, the sum of
// |ya(q - h) - yc(q + h)| with h = the final field (pixfield), true bilinear with clamp-to-edge on the UNPADDED flow grid (the
// field's size), ya, yc on the 0..255 scale (lv0's b, a). reduce_sad sums the rest; learned/sadout.frag.glsl scales the total.
uniform sampler2D uLv0;     // the net's level 0 (padded): b = ya, a = yc
uniform sampler2D uField;   // pixfield, fw x fh
uniform int uFan;
layout(location = 0) out vec4 oSum;

void main() {
    ivec2 o = ivec2(gl_FragCoord.xy);
    ivec2 fs = textureSize(uField, 0);
    float sum = 0.0;
    for (int j = 0; j < uFan; j++)
        for (int i = 0; i < uFan; i++) {
            ivec2 p = uFan * o + ivec2(i, j);
            if (p.x < fs.x && p.y < fs.y) {
                vec2 h = texelFetch(uField, p, 0).xy;
                sum += abs(mlBil(uLv0, fs, vec2(p) - h).b - mlBil(uLv0, fs, vec2(p) + h).a);
            }
        }
    oSum = vec4(sum);
}
