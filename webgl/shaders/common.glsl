#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp usampler2D;
precision highp isampler2D;
// Shared helpers for every fragment program (LAYOUT.md section 1.2). The host prepends this file to each
// *.frag.glsl. No uniforms, no outs, no main. Every texture read is a texelFetch; all bilinear sampling is
// manual (cv2 conventions: integer position = pixel centre, no +0.5). Mirrors gpu/shaders/common.glsl,
// pixfield/search/median3/luma .comp of frame-gen-ritz b77d058.

const int BLOCK = 8;

// FG_FETCH (format.glsl:49): round half up of a non-negative value
float fgRound(float x) { return floor(x + 0.5); }
vec3 fgRound(vec3 x) { return floor(x + 0.5); }

// L() RC:55 -- luma texel, clamp-to-edge (cv2 BORDER_REPLICATE == pad_to_block)
float fgFetchL(sampler2D t, ivec2 p) {
    return texelFetch(t, clamp(p, ivec2(0), textureSize(t, 0) - 1), 0).r;
}

// F() RC:72 -- two-channel field texel, clamp-to-edge
vec2 fgFetchF(sampler2D t, ivec2 p) {
    return texelFetch(t, clamp(p, ivec2(0), textureSize(t, 0) - 1), 0).rg;
}

// sampleL() RC:61-69 -- cv2.remap INTER_LINEAR / BORDER_REPLICATE on a float image, map rounded to 1/32 px
float fgSampleL(sampler2D t, vec2 p) {
    ivec2 X = ivec2(roundEven(p * 32.0));
    ivec2 i = X >> 5;
    vec2 f = vec2(X & 31) * (1.0 / 32.0);
    float p00 = fgFetchL(t, i), p01 = fgFetchL(t, i + ivec2(1, 0));
    float p10 = fgFetchL(t, i + ivec2(0, 1)), p11 = fgFetchL(t, i + ivec2(1, 1));
    return p00 * ((1.0 - f.y) * (1.0 - f.x)) + p01 * ((1.0 - f.y) * f.x)
         + p10 * (f.y * (1.0 - f.x)) + p11 * (f.y * f.x);
}

// sampleF() RC:75-83 -- same on the 2-channel field
vec2 fgSampleF(sampler2D t, vec2 p) {
    ivec2 X = ivec2(roundEven(p * 32.0));
    ivec2 i = X >> 5;
    vec2 f = vec2(X & 31) * (1.0 / 32.0);
    vec2 p00 = fgFetchF(t, i), p01 = fgFetchF(t, i + ivec2(1, 0));
    vec2 p10 = fgFetchF(t, i + ivec2(0, 1)), p11 = fgFetchF(t, i + ivec2(1, 1));
    return p00 * ((1.0 - f.y) * (1.0 - f.x)) + p01 * ((1.0 - f.y) * f.x)
         + p10 * (f.y * (1.0 - f.x)) + p11 * (f.y * f.x);
}

// RGBA8 texel as integer levels 0..255 (replicate clamp), alpha ignored
vec3 fgFetchRGB8(sampler2D t, ivec2 p) {
    return fgRound(texelFetch(t, clamp(p, ivec2(0), textureSize(t, 0) - 1), 0).rgb * 255.0);
}

// cv2.remap on uint8 RGB (RC:102 sampleRGBRaw + FG_FETCH, mismatch RC:136): 1/32 quantised bilinear, replicate
// clamp, result rounded to an integer-valued vec3 in 0..255.
vec3 fgSampleRGB8(sampler2D t, vec2 p) {
    ivec2 X = ivec2(roundEven(p * 32.0));
    ivec2 i = X >> 5;
    vec2 f = vec2(X & 31) * (1.0 / 32.0);
    vec3 p00 = fgFetchRGB8(t, i), p01 = fgFetchRGB8(t, i + ivec2(1, 0));
    vec3 p10 = fgFetchRGB8(t, i + ivec2(0, 1)), p11 = fgFetchRGB8(t, i + ivec2(1, 1));
    vec3 s = p00 * ((1.0 - f.y) * (1.0 - f.x)) + p01 * ((1.0 - f.y) * f.x)
           + p10 * (f.y * (1.0 - f.x)) + p11 * (f.y * f.x);
    return floor(s + 0.5);
}

// ---- frame formats (gpu/shaders/format.glsl's conventions). The host defines FG_F16 for settings.inputFormat 'rgba16f', with
// FG_XF_LINEAR (settings.transfer 'linear') and FG_XF_WHITE (settings.linearWhiteNits). Frames are read as WORKING VALUES:
// Rgba8: integer levels 0..255 (FG_FETCH); RgbaF16: the source's own values (warped and blended in its own space, never clamped).
// Distances and gradients are in levels of a perceptually encoded 8-bit picture: FG_D(a, b) = |level(a) - level(b)|, FG_G its signed
// difference, FG_LV(x) = level(x) (Rgba8: x; RgbaF16 encoded: x * 255; linear: the PQ code of the value x 255).
//   fgPx(t, p)    a texel (replicate clamp)         fgTap(t, p)   a 1/32-px bilinear tap (Rgba8: rounded to levels)
//   fgLumaPx(t,p) the luma (levels) of a texel       fgStore(o)    a working value -> the output texture's value
//   fgPackW / fgUnpackW: a working value in V1's warpA / warpC (Rgba8: RGBA8 levels / 255; RgbaF16: RGBA16F as is)
#ifdef FG_F16
#ifndef FG_XF_LINEAR
#define FG_XF_LINEAR 0
#endif
#ifndef FG_XF_WHITE
#define FG_XF_WHITE 80.0
#endif
float fgPq(float v) {   // linear value -> PQ code in [0, 1]
    float y = clamp(v * (FG_XF_WHITE / 10000.0), 0.0, 1.0);
    float p = pow(y, 0.1593017578125);
    return pow((0.8359375 + 18.8515625 * p) / (1.0 + 18.6875 * p), 78.84375);
}
vec3 fgMetric(vec3 v) {
#if FG_XF_LINEAR
    return vec3(fgPq(v.r), fgPq(v.g), fgPq(v.b)) * 255.0;
#else
    return v * 255.0;
#endif
}
vec3 fgPx(sampler2D t, ivec2 p) { return texelFetch(t, clamp(p, ivec2(0), textureSize(t, 0) - 1), 0).rgb; }
vec3 fgTap(sampler2D t, vec2 p) {
    ivec2 X = ivec2(roundEven(p * 32.0));
    ivec2 i = X >> 5;
    vec2 f = vec2(X & 31) * (1.0 / 32.0);
    vec3 p00 = fgPx(t, i), p01 = fgPx(t, i + ivec2(1, 0)), p10 = fgPx(t, i + ivec2(0, 1)), p11 = fgPx(t, i + ivec2(1, 1));
    return p00 * ((1.0 - f.y) * (1.0 - f.x)) + p01 * ((1.0 - f.y) * f.x) + p10 * (f.y * (1.0 - f.x)) + p11 * (f.y * f.x);
}
#define FG_LV(x) fgMetric(x)
#define FG_D(a, b) abs(fgMetric(a) - fgMetric(b))
#define FG_G(a, b) (fgMetric(a) - fgMetric(b))
float fgLumaPx(sampler2D t, ivec2 p) { return dot(fgMetric(texelFetch(t, p, 0).rgb), vec3(4899.0, 9617.0, 1868.0)) * (1.0 / 16384.0); }
vec4 fgStore(vec3 o) { return vec4(o, 1.0); }
vec4 fgPackW(vec3 w) { return vec4(w, 1.0); }
vec3 fgUnpackW(vec4 t) { return t.rgb; }
#else
vec3 fgPx(sampler2D t, ivec2 p) { return fgFetchRGB8(t, p); }
vec3 fgTap(sampler2D t, vec2 p) { return fgSampleRGB8(t, p); }
#define FG_LV(x) (x)
#define FG_D(a, b) abs((a) - (b))
#define FG_G(a, b) ((a) - (b))
float fgLumaPx(sampler2D t, ivec2 p) {   // = fgLuma8 of the texel's levels
    uvec3 c = uvec3(texelFetch(t, p, 0).rgb * 255.0 + 0.5);
    return float((c.r * 4899u + c.g * 9617u + c.b * 1868u + 8192u) >> 14);
}
vec4 fgStore(vec3 o) { return vec4(clamp(floor(o + 0.5), 0.0, 255.0) * (1.0 / 255.0), 1.0); }
vec4 fgPackW(vec3 w) { return vec4(w * (1.0 / 255.0), 1.0); }
vec3 fgUnpackW(vec4 t) { return fgRound(t.rgb * 255.0); }
#endif

// halfVec() RC:86-97 -- flow-res field -> full-res pixel p, in full-res pixels (cv2.resize INTER_LINEAR,
// edge pad, * S). No 1/32 quantisation here.
vec2 fgHalfVec(sampler2D field, ivec2 p, int S) {
    ivec2 fs = textureSize(field, 0);
    p = min(p, S * fs - 1);
    vec2 s = (vec2(p) + 0.5) / float(S) - 0.5;
    vec2 fl = floor(s);
    vec2 f = s - fl;
    ivec2 i0 = clamp(ivec2(fl), ivec2(0), fs - 1);
    ivec2 i1 = clamp(ivec2(fl) + 1, ivec2(0), fs - 1);
    vec2 v00 = texelFetch(field, ivec2(i0.x, i0.y), 0).xy, v01 = texelFetch(field, ivec2(i1.x, i0.y), 0).xy;
    vec2 v10 = texelFetch(field, ivec2(i0.x, i1.y), 0).xy, v11 = texelFetch(field, ivec2(i1.x, i1.y), 0).xy;
    return float(S) * mix(mix(v00, v01, f.x), mix(v10, v11, f.x), f.y);
}

// blockV() pixfield.comp:13 -- block-field vector, block index clamped to the grid
vec2 fgBlockV(sampler2D bf, ivec2 b) {
    return texelFetch(bf, clamp(b, ivec2(0), textureSize(bf, 0) - 1), 0).xy;
}

// cand() pixfield.comp:18-34 -- per-pixel candidate k (0..6) at flow pixel q
vec2 fgPixCand(int k, ivec2 q, sampler2D bf) {
    ivec2 b = q / BLOCK;
    if (k == 0) {   // cv2.resize(v, (bw*8, bh*8), INTER_LINEAR)
        vec2 s = (vec2(q) + 0.5) / float(BLOCK) - 0.5;
        vec2 fl = floor(s);
        vec2 f = s - fl;
        ivec2 i = ivec2(fl);
        return mix(mix(fgBlockV(bf, i), fgBlockV(bf, i + ivec2(1, 0)), f.x),
                   mix(fgBlockV(bf, i + ivec2(0, 1)), fgBlockV(bf, i + ivec2(1, 1)), f.x), f.y);
    }
    if (k == 1) return fgBlockV(bf, b);
    if (k == 2) return fgBlockV(bf, b + ivec2(0, -1));
    if (k == 3) return fgBlockV(bf, b + ivec2(0, 1));
    if (k == 4) return fgBlockV(bf, b + ivec2(-1, 0));
    if (k == 5) return fgBlockV(bf, b + ivec2(1, 0));
    return vec2(0.0);
}

// search.comp:25-34 -- seed candidate k (0..5) of fine block blk: 4 parent candidates (own + 3 nearest, doubled),
// zero, global median (medsel already stored 2*median)
ivec2 fgSeedCand(int k, ivec2 blk, sampler2D parent, sampler2D median) {
    if (k == 5) return ivec2(roundEven(texelFetch(median, ivec2(0), 0).xy));
    if (k == 4) return ivec2(0);
    ivec2 pb = textureSize(parent, 0);
    int px = min(blk.x / 2, pb.x - 1), py = min(blk.y / 2, pb.y - 1);
    int sy = (blk.y & 1) == 0 ? -1 : 1, sx = (blk.x & 1) == 0 ? -1 : 1;
    ivec2 pp = ivec2(0);
    if (k == 1) pp = ivec2(0, sy);
    else if (k == 2) pp = ivec2(sx, 0);
    else if (k == 3) pp = ivec2(sx, sy);
    ivec2 c = clamp(ivec2(px, py) + pp, ivec2(0), pb - 1);
    return 2 * ivec2(roundEven(texelFetch(parent, c, 0).xy));
}

// luma.comp:24 -- cv2 BGR2GRAY on integer levels, rounded to uint8
uint fgLuma8(uvec3 rgb) {
    return (rgb.r * 4899u + rgb.g * 9617u + rgb.b * 1868u + 8192u) >> 14;
}

// median3.comp:7-13 -- median of 9 (partial selection sort; v is a by-value copy)
float fgMed9(float v[9]) {
    for (int i = 0; i <= 4; i++)
        for (int j = i + 1; j < 9; j++)
            if (v[j] < v[i]) { float t = v[i]; v[i] = v[j]; v[j] = t; }
    return v[4];
}

// Extra (estimate-internal): raw SAD over the 64 pixels of block blk (y outer, x inner) for the integer
// half-vector cand: sum |A(q - cand) - C(q + cand)|. search.comp:35-37,66-68. Shared by seed_sad / search_sad.
float fgBlockSad(sampler2D lumaA, sampler2D lumaC, ivec2 blk, ivec2 cand) {
    float sum = 0.0;
    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++) {
            ivec2 q = blk * BLOCK + ivec2(x, y);
            sum += abs(fgFetchL(lumaA, q - cand) - fgFetchL(lumaC, q + cand));
        }
    return sum;
}

// ---- Motion version 2 (Performance) helpers: frame-gen-ritz framegen-v2 @ 273bbad, gpu/shaders/v2_repick.comp ----

// Candidate k of block blk for the re-pick (0 = the block's own vector, then its 8 neighbours in the library's order), taken from the
// search's sub-pixel fitted finest field and snapped to the quarter-pel grid.
vec2 fgRepCand(int k, ivec2 blk, sampler2D vfit) {
    int j = k == 0 ? 4 : (k <= 4 ? k - 1 : k);
    ivec2 b = clamp(blk + ivec2(j % 3 - 1, j / 3 - 1), ivec2(0), textureSize(vfit, 0) - 1);
    return roundEven(texelFetch(vfit, b, 0).xy * 4.0) * 0.25;
}

// The block's 16-pixel quincunx (every other pixel of every other row, shifted by one on alternate rows): raw sum of
// |a(q - v) - c(q + v)| with bilinear luma taps (1/32 quantised).
float fgQuinSad(sampler2D lumaA, sampler2D lumaC, ivec2 blk, vec2 v) {
    float sum = 0.0;
    for (int i = 0; i < 16; i++) {
        int ry = i >> 2;
        vec2 q = vec2(blk * BLOCK + ivec2(2 * (i & 3) + (ry & 1), 2 * ry));
        sum += abs(fgSampleL(lumaA, q - v) - fgSampleL(lumaC, q + v));
    }
    return sum;
}
