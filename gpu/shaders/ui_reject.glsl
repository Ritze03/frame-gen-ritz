// CrosshairV2: warp taps that land on the UI are rejected. Included by common.glsl and blur_common.glsl (after the
// declarations of format.glsl), used by mismatch() and warpChroma().
//
// Why: V2 hands the host's real frames to the warp, UI included. Each generated pixel blends a tap of prev at p - 2t*h with a tap of
// curr at p + 2(1-t)*h; where the background pans, a tap that lands on the static crosshair copies it to "crosshair +- motion*t": a
// second, full-strength crosshair next to the real one (the "doubled crosshair"). The V2 patch only pastes the real crosshair at its
// true place, so the copy stayed. A tap that lands on the UI pixels of ITS OWN frame therefore takes the other frame's tap for both
// (the pixel is then a plain still sample of the background the UI covers there); both taps on UI, or neither, stay as they are.
//
// The UI pixels are the V2 mask of each frame (framegen.cpp binds the pair's entries to the synth sets, binding 11 = prev's, 12 =
// curr's; a pair without V2 masks gets a zeroed buffer, whose rectangle is empty: nothing is ever rejected). Header words of a V2 mask
// buffer: [1] = maskX | maskY << 16, [2] = maskW | maskH << 16 (0 = no mask), [3] = words per row; mask bytes (4 pixels per word,
// 0 = not UI) from word 4. ui_v2_mask.comp writes the header.
layout(set = 0, binding = 11, std430) readonly buffer UiRejA { uint urA[]; };
layout(set = 0, binding = 12, std430) readonly buffer UiRejC { uint urC[]; };

// Is any pixel of lo..hi (inclusive, frame coordinates) in the mask?
#define FG_UI_HIT(BUF, NAME) \
bool NAME(ivec2 lo, ivec2 hi) { \
    uint wh = BUF[2]; \
    if (wh == 0u) return false; \
    ivec2 o = ivec2(int(BUF[1] & 0xFFFFu), int(BUF[1] >> 16)); \
    ivec2 sz = ivec2(int(wh & 0xFFFFu), int(wh >> 16)); \
    int stride = int(BUF[3]); \
    lo = max(lo - o, ivec2(0)); \
    hi = min(hi - o, sz - 1); \
    for (int y = lo.y; y <= hi.y; y++) \
        for (int x = lo.x; x <= hi.x; x++) \
            if (((BUF[4 + y * stride + (x >> 2)] >> (8 * (x & 3))) & 255u) != 0u) return true; \
    return false; \
}
FG_UI_HIT(urA, uiHitA)
FG_UI_HIT(urC, uiHitC)

// The pixels a sample at p (pixel-centre coordinates, taken like sampleRGB / sampleYUV: rounded to 1/32 px, bilinear) reads with a
// non-zero weight: the 2x2 pixels around it; in YUV, the 2x2 texels of the half-resolution chroma plane around it, i.e. up to 4x4
// pixels (which contain the luma taps).
void uiFootprint(vec2 p, out ivec2 lo, out ivec2 hi) {
    ivec2 X = ivec2(roundEven(p * 32.0));
#ifdef FG_YUV
    ivec2 c = X - 16;                       // chroma texel coordinate * 64: (x + 0.5) / 2 - 0.5
    ivec2 i = c >> 6;
    ivec2 n = ivec2(notEqual(c & 63, ivec2(0)));   // the second texel has a weight only if the fraction is not 0
    lo = 2 * i;
    hi = 2 * (i + n) + 1;
#else
    lo = X >> 5;
    hi = lo + ivec2(notEqual(X & 31, ivec2(0)));
#endif
}

// pa is prev's tap, pc_ curr's. A tap on its own frame's UI takes the other tap (both on UI: unchanged).
bool uiRejectHits(vec2 pa, vec2 pc_, out bool ha, out bool hc) {
    ivec2 lo, hi;
    uiFootprint(pa, lo, hi);
    ha = uiHitA(lo, hi);
    uiFootprint(pc_, lo, hi);
    hc = uiHitC(lo, hi);
    return ha != hc;
}
void uiReject(vec2 pa, vec2 pc_, inout vec3 wa, inout vec3 wc) {
    bool ha, hc;
    if (!uiRejectHits(pa, pc_, ha, hc)) return;
    if (ha) wa = wc; else wc = wa;
}
#ifdef FG_YUV
void uiRejectC(vec2 pa, vec2 pc_, inout vec2 ca, inout vec2 cc) {
    bool ha, hc;
    if (!uiRejectHits(pa, pc_, ha, hc)) return;
    if (ha) ca = cc; else cc = ca;
}
#endif
