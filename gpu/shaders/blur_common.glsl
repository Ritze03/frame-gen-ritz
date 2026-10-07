// Shared declarations of the motion-blur passes (recordSynthBlur), of every motion version: the format, the descriptor bindings
// and the CrosshairV2 tap rejection. The descriptor set is the one of every other pass (framegen.cpp: binding 0 = luma is not
// used here); bindings 6 and 7 are the blur scratch. Each blur pass includes this, then its version's motion part
// (blur_motion_v1.glsl).

#include "format.glsl"

layout(set = 0, binding = 1, std430) buffer FieldBuf { vec2 field[]; };   // estimate: per-pixel field (read only here)
layout(set = 0, binding = 2, std430) buffer UBuf { uint u[]; };           // counters (and the estimate's SAD sum)
layout(set = 0, binding = 3) uniform sampler2D texPrev;
layout(set = 0, binding = 4) uniform sampler2D texCurr;
layout(set = 0, binding = 5, FG_FMT) uniform writeonly image2D outImg;
#ifdef FG_YUV
layout(set = 0, binding = 8) uniform sampler2D texPrevC;
layout(set = 0, binding = 9) uniform sampler2D texCurrC;
layout(set = 0, binding = 10, FG_FMTC) uniform writeonly image2D outImgC;
#endif
layout(set = 0, binding = 6, std430) buffer BField { vec2 bf[]; };        // field at t_k, flow res, n slots of fw*fh
layout(set = 0, binding = 7, std430) buffer BErr { uint be[]; };          // packed mismatch of sample k, n slots
#include "ui_reject.glsl"   // CrosshairV2: bindings 11 / 12 and the tap rejection of mismatch() / warpChroma()
