#version 300 es
// V2 Quality, the coverage splat (gpu/shaders/v2_qown.comp) as MIN-blended point draws into fw x fh R32F claim textures, one per side
// (uSide 0 = prev's footprints, 1 = curr's; cleared to 1e30 = unclaimed). The library keeps per pixel the smallest key << 20 | vector
// (atomicMin). The same minimum, lexicographic on (key, vector), is: pass 0 (uPass 0) the smallest key per pixel (v2.qkeyA / C), pass 1
// the smallest vector bits among the claims whose key equals that minimum (v2.qvecA / C). Both are integers below 2^24, exact in fp32.
// 2 vertices per location m = gl_VertexID >> 1: kind 0 its claim at its footprint in this side's frame, kind 1 a static-overlay
// location (z >= 0.5) claiming its own place with key 0 and the zero vector. A vertex with nothing to write is clipped.
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp isampler2D;
uniform highp isampler2D uClaim;   // v2.qclaim: key, vector bits, prev footprint, curr footprint
uniform sampler2D uQf;             // v2.qf: z = the static-overlay mask
uniform sampler2D uKey;            // pass 1: this side's v2.qkey
uniform int uPass;
uniform int uSide;
flat out float vVal;

void main() {
    ivec2 fs = textureSize(uClaim, 0);
    int idx = gl_VertexID >> 1, kind = gl_VertexID & 1;
    ivec2 m = ivec2(idx % fs.x, idx / fs.x);
    ivec4 c = texelFetch(uClaim, m, 0);
    int pix = -1, key = c.x, vb = c.y;
    if (kind == 0) pix = uSide == 0 ? c.z : c.w;
    else if (texelFetch(uQf, m, 0).z >= 0.5) { pix = m.x | (m.y << 16); key = 0; vb = (512 << 10) | 512; }
    ivec2 p = ivec2(pix & 0xFFFF, pix >> 16);
    bool on = pix >= 0;
    if (on && uPass == 1) on = texelFetch(uKey, p, 0).r == float(key);
    vVal = uPass == 0 ? float(key) : float(vb);
    gl_PointSize = 1.0;
    gl_Position = on ? vec4((vec2(p) + 0.5) / vec2(fs) * 2.0 - 1.0, 0.0, 1.0) : vec4(2.0, 2.0, 2.0, 1.0);
}
