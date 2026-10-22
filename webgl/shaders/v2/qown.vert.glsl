#version 300 es
// V2 Quality, the coverage splat (gpu/shaders/v2_qown.comp) as one MIN-blended point draw per side into a fw x fh R32F claim texture
// (uSide 0 = prev's footprints, 1 = curr's; initialised by v2/qown_z with the static-overlay claims and "unclaimed"). The library keeps
// per pixel the smallest word key << 20 | vector (atomicMin). Here each point writes the word W = (min(key, 255) << 20 | vector) +
// 2^23 as the float with those bits: positive normal floats order like their bit patterns and MIN blending selects without arithmetic,
// so the blend keeps exactly the smallest W (2^23 keeps W above the denormals, which a GPU may flush; W < 2^28). The key clamp changes
// no result: v2/qvis only asks whether the winning key is above Q_WIN * Q_KEY = 48 (then the location sees its content whatever the
// vector) and, below that, compares it with the location's own (unclamped) key; keys above 255 never win against one of 48 or less.
// One vertex per location m = gl_VertexID; a footprint outside the frame is clipped.
precision highp float;
precision highp int;
precision highp isampler2D;
uniform highp isampler2D uClaim;   // v2.qclaim: key, vector bits, prev footprint, curr footprint
uniform int uSide;
flat out uint vW;

void main() {
    ivec2 fs = textureSize(uClaim, 0);
    ivec2 m = ivec2(gl_VertexID % fs.x, gl_VertexID / fs.x);
    ivec4 c = texelFetch(uClaim, m, 0);
    int pix = uSide == 0 ? c.z : c.w;
    ivec2 p = ivec2(pix & 0xFFFF, pix >> 16);
    vW = ((uint(min(c.x, 255)) << 20) | uint(c.y)) + 0x800000u;
    gl_PointSize = 1.0;
    gl_Position = pix >= 0 ? vec4((vec2(p) + 0.5) / vec2(fs) * 2.0 - 1.0, 0.0, 1.0) : vec4(2.0, 2.0, 2.0, 1.0);
}
