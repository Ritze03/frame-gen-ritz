// conv.frag.glsl for settings.motionPrecision 'fp16': the activations are half floats packed 8 to an RGBA32UI texel
// (packHalf2x16: texel channel k holds channels 2k, 2k + 1 of the texture's 8), so a layer of width 24 is one pass with 3 output
// textures and each tap is one 128-bit fetch per 8 channels. Arithmetic, weights and the field stay float32. #defines:
//   ML_CIN   packed input textures, 1..3 (the 14-channel input: 2)
//   ML_CG    4-channel input groups the weights cover, 3..6 (= ceil(channels / 4); odd: the last texture's upper half is unused)
//   ML_COUT  4-channel output groups, 1..6: groups 0..2 take their weights from block MlW, 3..5 from MlW2 (each block within the
//            16 KB uniform block minimum); output texture k packs groups 2k, 2k + 1
//   ML_LAST  the 2-channel output layer (ML_COUT 1): writes h + 2 * out to an RG32F target, as conv.frag.glsl
//   ML_COUTA min(ML_COUT, 3): the groups of block MlW
// Weight blocks: uW[(o * 9 + tap) * ML_CG + g] = the 4x4 block from input group g to output group o (MlW2: o - 3), as conv.frag.glsl.

uniform ivec2 uSize;
uniform highp usampler2D uIn0;
#if ML_CIN > 1
uniform highp usampler2D uIn1;
#endif
#if ML_CIN > 2
uniform highp usampler2D uIn2;
#endif
#if ML_LAST
uniform sampler2D uHc;
uniform int uTop;
#endif
layout(std140) uniform MlW {
    mat4 uW[ML_COUTA * 9 * ML_CG];
    vec4 uB[ML_COUTA];
};
#if ML_COUT > 3
layout(std140) uniform MlW2 {
    mat4 uW2[(ML_COUT - 3) * 9 * ML_CG];
    vec4 uB2[ML_COUT - 3];
};
#endif
#if ML_LAST
layout(location = 0) out vec4 oH;
#else
layout(location = 0) out uvec4 o0;
#if ML_COUT > 2
layout(location = 1) out uvec4 o1;
#endif
#if ML_COUT > 4
layout(location = 2) out uvec4 o2;
#endif
#endif

#if ML_COUT > 3
#define ML_GRP2(G, v) for (int o = 3; o < ML_COUT; o++) acc[o] += uW2[((o - 3) * 9 + t) * ML_CG + (G)] * v;
#else
#define ML_GRP2(G, v)
#endif
#define ML_GRP(G, v) { vec4 w_ = v; for (int o = 0; o < ML_COUTA; o++) acc[o] += uW[(o * 9 + t) * ML_CG + (G)] * w_; ML_GRP2(G, w_) }
#define ML_LO(u) vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y))
#define ML_HI(u) vec4(unpackHalf2x16(u.z), unpackHalf2x16(u.w))

uvec4 mlPack(vec4 a, vec4 b) {
    return uvec4(packHalf2x16(a.xy), packHalf2x16(a.zw), packHalf2x16(b.xy), packHalf2x16(b.zw));
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    if (mlDup()) {
#if ML_LAST
        oH = vec4(0.0);
#else
        o0 = uvec4(0u);
#if ML_COUT > 2
        o1 = uvec4(0u);
#endif
#if ML_COUT > 4
        o2 = uvec4(0u);
#endif
#endif
        return;
    }
    vec4 acc[ML_COUT];
    for (int o = 0; o < ML_COUTA; o++) acc[o] = uB[o];
#if ML_COUT > 3
    for (int o = 3; o < ML_COUT; o++) acc[o] = uB2[o - 3];
#endif
    for (int t = 0; t < 9; t++) {
        ivec2 q = p + ivec2(t % 3 - 1, t / 3 - 1);
        if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, uSize))) continue;   // zero padding
        uvec4 u = texelFetch(uIn0, q, 0);
        ML_GRP(0, ML_LO(u))
#if ML_CG > 1
        ML_GRP(1, ML_HI(u))
#endif
#if ML_CIN > 1
        u = texelFetch(uIn1, q, 0);
        ML_GRP(2, ML_LO(u))
#if ML_CG > 3
        ML_GRP(3, ML_HI(u))
#endif
#endif
#if ML_CIN > 2
        u = texelFetch(uIn2, q, 0);
        ML_GRP(4, ML_LO(u))
#if ML_CG > 5
        ML_GRP(5, ML_HI(u))
#endif
#endif
    }
#if ML_LAST
    vec2 h = uTop != 0 ? vec2(0.0) : mlFieldUp(uHc, p);
    oH = vec4(h + 2.0 * acc[0].xy, 0.0, 0.0);
#else
    for (int o = 0; o < ML_COUT; o++) acc[o] = clamp(acc[o], 0.0, 1.0);
#if ML_COUT > 1
    o0 = mlPack(acc[0], acc[1]);
#else
    o0 = mlPack(acc[0], vec4(0.0));
#endif
#if ML_COUT > 3
    o1 = mlPack(acc[2], acc[3]);
#elif ML_COUT > 2
    o1 = mlPack(acc[2], vec4(0.0));
#endif
#if ML_COUT > 5
    o2 = mlPack(acc[4], acc[5]);
#elif ML_COUT > 4
    o2 = mlPack(acc[4], vec4(0.0));
#endif
#endif
}
