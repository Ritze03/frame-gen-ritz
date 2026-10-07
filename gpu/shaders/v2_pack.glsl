// V2: the packed half-vector of its stored fields (the refined fields rf[], binding 14; the temporal store in v2b[], binding 13):
// two signed 16-bit fixed-point numbers in 1/32 pixel in one uint (x in the low half), range +-1023.97 px of the field's own units.
// Why 1/32: every warp tap rounds its position to 1/32 px (sampleRGBRaw / sampleL), so at t = 0.5 (taps p -+ h at an integer p) a
// vector rounded to 1/32 takes exactly the taps of the unrounded one; the fields need half the memory of vec2 (D19, I10). The
// largest half-vector the search can reach is ~250 px (flowScale 4 / 2160p): far inside the range.
uint packV(vec2 v) {
    ivec2 i = ivec2(roundEven(clamp(v, vec2(-1023.96875), vec2(1023.96875)) * 32.0));
    return uint(i.x & 0xFFFF) | (uint(i.y) << 16);
}
vec2 unpackV(uint w) {
    return vec2(float(bitfieldExtract(int(w), 0, 16)), float(bitfieldExtract(int(w), 16, 16))) * (1.0 / 32.0);
}
// the stored value of v (what packV / unpackV give back)
vec2 quantV(vec2 v) { return unpackV(packV(v)); }

#ifdef V2_RF
// The field at time t of the refined midpoint field (v2_lookup.comp; the blur passes compute it on the fly with the very same code,
// so a blurred sub-frame takes exactly the vectors of recordSynth at its t): content at midpoint m sits at q = m + s h(m) at time t
// (s = 2t - 1), so g(q) = h(q - s g(q)): 2 bilinear fixed-point steps started from h(q), in grid coordinates (the vector scaled by the
// grid's resolution), rounded to the stored precision. The includer defines V2_RF(i) = the packed midpoint field at grid index i.
vec2 v2RF(int x, int y, int rw, int rh) { return unpackV(V2_RF(clamp(y, 0, rh - 1) * rw + clamp(x, 0, rw - 1))); }
vec2 v2SampleRF(vec2 p, int rw, int rh) {   // == motion_v1.glsl sampleF on the refinement grid
    ivec2 X = ivec2(roundEven(p * 32.0));
    ivec2 i = X >> 5;
    vec2 f = vec2(X & 31) * (1.0 / 32.0);
    vec2 p00 = v2RF(i.x, i.y, rw, rh), p01 = v2RF(i.x + 1, i.y, rw, rh);
    vec2 p10 = v2RF(i.x, i.y + 1, rw, rh), p11 = v2RF(i.x + 1, i.y + 1, rw, rh);
    return p00 * ((1.0 - f.y) * (1.0 - f.x)) + p01 * ((1.0 - f.y) * f.x)
         + p10 * (f.y * (1.0 - f.x)) + p11 * (f.y * f.x);
}
vec2 v2LookupT(ivec2 q, float s, int rw, int rh, int W, int H) {
    vec2 sc = vec2(rw, rh) / vec2(W, H);   // grid pixels per full-resolution pixel
    vec2 g = v2RF(q.x, q.y, rw, rh);
    for (int k = 0; k < 2; k++)   // LOOKUP_ITERS
        g = v2SampleRF(vec2(q) - s * (g * sc), rw, rh);
    return quantV(g);
}
#endif
