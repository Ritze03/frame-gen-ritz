// V2 Quality, the coverage splat's static-overlay claims (gpu/shaders/v2_qown.comp: a location with z >= 0.5 claims its own place in
// both frames with key 0 and the zero vector), drawn full screen WITHOUT blending before the MIN-blended points of v2/qown, so it also
// initialises the claim texture: the word of that claim where z >= 0.5, else "unclaimed" (the largest word, key 2024 > Q_WIN * Q_KEY).
// Words as in v2/qown.vert.glsl: (key << 20 | vector) + 2^23, stored as the float with those bits.
uniform sampler2D uQf;    // v2.qf: z
layout(location = 0) out vec4 oClaim;

void main() {
    bool z = texelFetch(uQf, ivec2(gl_FragCoord.xy), 0).z >= 0.5;
    oClaim = vec4(uintBitsToFloat(z ? uint((512 << 10) | 512) + 0x800000u : 0x7F000000u), 0.0, 0.0, 0.0);
}
