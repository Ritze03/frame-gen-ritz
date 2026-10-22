// V2 Quality, each midpoint location's own claim (flow resolution; the per-location part of gpu/shaders/v2_qown.comp): with h the
// per-pixel field at m, the key |luma_a(m - h) - luma_c(m + h)| (bilinear) in 1/Q_KEY levels (rounded, at most 4095), the vector
// quantised to quarter flow pixels, and the footprint pixels m - h (prev) and m + h (curr), -1 outside the frame.
// Output v2.qclaim (RGBA32I): x = key, y = vector bits ((vx + 512) << 10 | (vy + 512)), z = prev footprint, w = curr footprint
// (packed x | y << 16).
uniform sampler2D uLumaA;
uniform sampler2D uLumaC;
uniform sampler2D uQf;   // v2.qf
layout(location = 0) out ivec4 oClaim;

void main() {
    ivec2 m = ivec2(gl_FragCoord.xy);
    ivec2 fs = textureSize(uQf, 0);
    vec2 h = texelFetch(uQf, m, 0).xy;
    vec2 pa = vec2(m) - h, pc_ = vec2(m) + h;
    float k = abs(qLuma(uLumaA, pa) - qLuma(uLumaC, pc_));
    int key = int(min(uint(k * Q_KEY + 0.5), 4095u));
    oClaim = ivec4(key, qVbits(qQuant(h)), qPix(pa, fs), qPix(pc_, fs));
}
