// Per-format knobs of the colour passes. Included by common.glsl, blur_common.glsl and ui_common.glsl.
//
// The library builds each colour shader once per image format, from the same source, selected by one -D define:
//   (none)           Rgba8   B8G8R8A8 / R8G8B8A8 UNORM          the historical shaders
//   FG_FORMAT_RGB10  Rgb10   A2B10G10R10 / A2R10G10B10 UNORM
//   FG_FORMAT_F16    RgbaF16 R16G16B16A16_SFLOAT
//   FG_FORMAT_NV12   Nv12    8-bit 4:2:0, planes R8 (Y) + R8G8 (CbCr)
//   FG_FORMAT_P010   P010    10-bit 4:2:0 in the high bits of 16-bit words, planes R16 (Y) + R16G16 (CbCr)
//
// THE RGBA8 PATH MUST STAY EXACTLY WHAT IT WAS: with no define, every macro below expands to the token sequence the
// shaders used to spell out by hand, and nothing else is declared, so the generated SPIR-V is identical to the one of
// the library before formats existed (checked by cmp, see README.md "Formats"). Do not add anything to that path.
// (Since the CrosshairV2 tap rejection the warp passes -- error, resolve, blur_error, blur_resolve -- also read the V2 masks
// (ui_reject.glsl), so their SPIR-V is no longer cmp-identical; their OUTPUT still is: the other modes are checked by hash.)
//
// "Levels": every setting that is in 8-bit levels (trustLow/High, sceneCutSad, hudBonus, searchPenalty, uiStillThreshold,
// ...) keeps that meaning in every format. The motion estimate runs on luma in levels, and the mismatch metric and the
// stillness test run on the signal converted to levels of a perceptually encoded 8-bit picture: 10-bit codes / 4 (x * 255),
// PQ-coded linear light (floats with Transfer::Linear), or YCbCr differences converted to R'G'B' differences.
// Blending and warping always stay in the source's own space.

#ifndef FG_FORMAT_GLSL
#define FG_FORMAT_GLSL

#if defined(FG_FORMAT_NV12) || defined(FG_FORMAT_P010)
#define FG_YUV 1
#endif

#if defined(FG_FORMAT_RGB10)
#define FG_FMT rgb10_a2
#define FG_FETCH(x) ((x) * 255.0)
#define FG_STORE(o, al) vec4(clamp(o, 0.0, 255.0) * (1.0 / 255.0), al)
#define FG_DIFF(wa, wc) abs((wa) - (wc))
#define FG_PACKE(x) uint(min(x, 255.0))
#define FG_STILL(x) floor((x) * 255.0 + 0.5)
#elif defined(FG_FORMAT_F16)
#define FG_FMT rgba16f
#define FG_FETCH(x) (x)
#define FG_STORE(o, al) vec4(o, al)
#define FG_DIFF(wa, wc) abs(fgMetric(wa) - fgMetric(wc))
#define FG_PACKE(x) uint(min(x, 255.0))
#define FG_STILL(x) clamp(floor(fgMetric(x) + 0.5), 0.0, 255.0)
#elif defined(FG_YUV)
#define FG_DIFF(wa, wc) abs(fgYuvDiff((wa) - (wc)))
#define FG_PACKE(x) uint(min(x, 255.0))
#else
// Rgba8: the historical spelling of every site.
#define FG_FMT rgba8
#define FG_FETCH(x) floor(x * 255.0 + 0.5)
#define FG_STORE(o, al) vec4(clamp(floor(o + 0.5), 0.0, 255.0) * (1.0 / 255.0), al)
#define FG_DIFF(wa, wc) abs(wa - wc)
#define FG_PACKE(x) uint(x)
#define FG_STILL(x) floor(x * 255.0 + 0.5)
#endif

#if defined(FG_FORMAT_F16)
// Transfer of the float input (InitInfo::transfer): 0 = values are display-encoded (sRGB / gamma / PQ / HLG): the
// level is value * 255; 1 = linear light (scRGB or plain linear), 1.0 = XF_WHITE nits: the level is the PQ code of the
// luminance, so a level step weighs about the same in the shadows and in the highlights.
layout(constant_id = 0) const int XF_LINEAR = 0;
layout(constant_id = 1) const float XF_WHITE = 80.0;

float fgPq(float v) {   // linear value -> PQ code in [0, 1]
    float y = clamp(v * (XF_WHITE / 10000.0), 0.0, 1.0);
    float p = pow(y, 0.1593017578125);
    return pow((0.8359375 + 18.8515625 * p) / (1.0 + 18.6875 * p), 78.84375);
}
vec3 fgMetric(vec3 v) {   // source value -> level of a perceptually encoded 8-bit picture
    if (XF_LINEAR != 0) return vec3(fgPq(v.r), fgPq(v.g), fgPq(v.b)) * 255.0;
    return v * 255.0;
}
#endif

#if defined(FG_YUV)
#if defined(FG_FORMAT_P010)
#define FG_FMT r16
#define FG_FMTC rg16
#else
#define FG_FMT r8
#define FG_FMTC rg8
#endif
// Plane samples are normalised UNORM. A "level" is an 8-bit-equivalent code: 8-bit codes as they are, 10-bit codes / 4.
layout(constant_id = 0) const float YU_SCALE = 255.0;     // plane sample (0..1) -> level
layout(constant_id = 1) const float YU_QL = 1.0;          // one code step in levels (1, or 0.25 for 10-bit)
layout(constant_id = 2) const float YU_CODES = 255.0;     // largest code (255, 1023)
layout(constant_id = 3) const float YU_BLACK = 0.0;       // luma black in levels (16 limited range, 0 full)
layout(constant_id = 4) const float YU_YS = 1.0;          // luma gain to full-range levels (255/219 limited, 1 full)
layout(constant_id = 5) const float YU_RCR = 1.5748;      // R' per Cr level (range gain included)
layout(constant_id = 6) const float YU_BCB = 1.8556;      // B' per Cb level
layout(constant_id = 7) const float YU_GCB = -0.1873;     // G' per Cb level
layout(constant_id = 8) const float YU_GCR = -0.4681;     // G' per Cr level

// difference of two (Y, Cb, Cr) level triples -> the difference of the R'G'B' they stand for, in full-range levels
vec3 fgYuvDiff(vec3 d) {
    return vec3(YU_YS * d.x + YU_RCR * d.z, YU_YS * d.x + YU_GCB * d.y + YU_GCR * d.z, YU_YS * d.x + YU_BCB * d.y);
}
// a level -> the plane sample that stores it (rounded to the code grid, clamped to the code range)
float fgOut(float l) {
    return clamp(floor(l / YU_QL + 0.5), 0.0, YU_CODES) * YU_QL / YU_SCALE;
}
vec2 fgOut2(vec2 l) { return vec2(fgOut(l.x), fgOut(l.y)); }
// luma level of a Y sample as the motion estimate sees it (full-range levels)
float fgLuma(float y) { return (y * YU_SCALE - YU_BLACK) * YU_YS; }
// stillness: (Y, Cb, Cr) samples -> integer levels 0..255 (the UI state packs bytes)
vec3 fgStillYuv(float y, vec2 c) { return clamp(floor(vec3(y, c) * YU_SCALE + 0.5), 0.0, 255.0); }
#endif

#endif
