// Shared declarations for all framegen passes, of every motion version (Settings::motionVersion): the format, the descriptor
// bindings and the CrosshairV2 tap rejection, so a format or UI fix lands once for all versions. Each pass includes this, then its
// version's motion part (motion_v1.glsl: push constants, warp, mismatch). Mirrors proto/interp.py (the frozen reference).
#include "format.glsl"

layout(set = 0, binding = 0, std430) buffer LumaBuf { float luma[]; };   // luma pyramids of a and c
layout(set = 0, binding = 1, std430) buffer FieldBuf { vec2 field[]; };  // half-vector fields
layout(set = 0, binding = 2, std430) buffer UBuf { uint u[]; };          // counters, medians, packed mismatch
layout(set = 0, binding = 3) uniform sampler2D texPrev;                  // frame a (linear, clamp to edge)
layout(set = 0, binding = 4) uniform sampler2D texCurr;                  // frame c
layout(set = 0, binding = 5, FG_FMT) uniform writeonly image2D outImg;
#ifdef FG_YUV
layout(set = 0, binding = 8) uniform sampler2D texPrevC;                 // CbCr planes of a and c (half resolution)
layout(set = 0, binding = 9) uniform sampler2D texCurrC;
layout(set = 0, binding = 10, FG_FMTC) uniform writeonly image2D outImgC;  // CbCr plane of the output
#endif
#include "ui_reject.glsl"   // CrosshairV2: bindings 11 / 12 and the tap rejection of mismatch() / warpChroma()
