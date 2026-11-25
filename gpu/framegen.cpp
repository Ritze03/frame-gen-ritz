// framegen: Vulkan compute port of proto/interp.py. See framegen.h for the API contract.
#include "framegen.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <cassert>

#include "luma_spv.h"
#include "down_spv.h"
#include "medhist_spv.h"
#include "medhist_wide_spv.h"
#include "search_spv.h"
#include "search_wide_spv.h"
#include "median3_spv.h"
#include "subpix_spv.h"
#include "pixfield_spv.h"
#include "error_spv.h"
#include "resolve_spv.h"
#include "fallback_spv.h"
#include "lookup_spv.h"
#include "ui_detect_spv.h"
#include "ui_box_update_spv.h"
#include "ui_box_mask_spv.h"
#include "ui_mask_spv.h"
#include "ui_inpaint_spv.h"
#include "ui_patch_spv.h"
#include "ui_v2_update_spv.h"
#include "ui_v2_mask_spv.h"
#include "ui_v2_lumafill_spv.h"
#include "blur_lookup_spv.h"
#include "blur_error_spv.h"
#include "blur_resolve_spv.h"
// MotionVersion::V2 (every format: only its synthesis, v2_morph / v2_morph_s / v2_qmorph / v2_qmorph_s, has format variants, below)
#include "v2_medhist_spv.h"
#include "v2_medhist_wide_spv.h"
#include "v2_search_spv.h"
#include "v2_search_wide_spv.h"
#include "v2_qfield_spv.h"
#include "v2_qown_spv.h"
#include "v2_qown_wide_spv.h"
#include "v2_qcov_spv.h"
#include "v2_qcov_wide_spv.h"
#include "v2_qmorph_spv.h"
#include "v2_clean_spv.h"
#include "v2_morph_spv.h"
#include "v2_repick_spv.h"
#include "v2_static_spv.h"
#include "v2_enclose_spv.h"
#include "v2_morph_s_spv.h"
#include "v2_qmorph_s_spv.h"
// MotionVersion::V3 (gpu/V3.md): its raw planes (v3_planes) and its synthesis (v3_synth) read colour and have a variant per format (probed per format below)
#include "v3_synth_spv.h"
#include "v3_synth_b_spv.h"   // Settings::v3Blend: the synthesis with the learned-blend nets, -DV3B=1 (Rgba8 / Bgra8 and Rgb10)
#include "v3_synth_l_spv.h"   // hook (l) v3DetailSplit: the same synthesis built with -DV3L=1 (per format below)
// (the estimate's other passes: plain builds, they work on V3's own planes and the luma)
#include "v3_planes_spv.h"
#include "v3_pyr_spv.h"
#include "v3_hist_spv.h"
#include "v3_search_spv.h"
#include "v3_search_top_spv.h"
#include "v3_dense_spv.h"
#include "v3_wide_spv.h"
#include "v3_search_j_spv.h"
#include "v3_search_top_j_spv.h"
#include "v3_dense_j_spv.h"
#include "v3_stats_spv.h"
// (Y1: occlusion, fill, projection; the synthesis above is the real one)
#include "v3_occl_spv.h"
#include "v3_fill_spv.h"
#include "v3_project_spv.h"
// (Y2: the static-structure enclosure and the pair's decisions; the static mask itself is V2's v2_static)
#include "v3_enclose_spv.h"
#include "v3_decide_spv.h"
// (task 104, Settings::v3Guard: V2 Quality's vectors correct V3's dense field; V2's own estimate passes above are reused)
#include "v3_anchor_spv.h"
#include "v3_guard_spv.h"
#include "v3_dense_35_spv.h"
#include "v3_search_35_spv.h"   // (V3.5: the block search reading its parent level's regularised records from slot 0 or 1, v3_search.comp with -DV35=1)
#include "v35_reg_spv.h"
#include "v35_stats_spv.h"
#include "v35_rgn_spv.h"      // Settings::v3Motion = 1 (V3.5): the block field regularisation
// MotionVersion::Learned: its passes and the nets' weights (gpu/ml/embed_weights.py), probed as a set: a host build that does not
// generate them gets Learned refused with a reason. Only its duplicate test (ml_dup) and its static guard (ml_guard) read colour and
// have format variants (probed per format below).
#if __has_include("ml_conv_spv.h") && __has_include("ml_zero_spv.h") && __has_include("ml_guard_spv.h") && __has_include("ml_weights.h")
#define FG_HAVE_ML 1
#include "ml_dup_spv.h"
#include "ml_reduce_spv.h"
#include "ml_pyr_spv.h"
#include "ml_match_spv.h"
#include "ml_conv_spv.h"
#include "ml_out_spv.h"
#include "ml_zero_spv.h"
#include "ml_guard_spv.h"
#include "ml_weights.h"
#endif

// The colour shaders of the formats other than Rgba8 (shaders/format.glsl), when built.
// rgb10: all its variant headers are generated together (gpu/meson.build `formats` option), so one probe stands for the set.
#if __has_include("luma_rgb10_spv.h")
#define FG_HAVE_RGB10 1
#include "luma_rgb10_spv.h"
#include "error_rgb10_spv.h"
#include "resolve_rgb10_spv.h"
#include "fallback_rgb10_spv.h"
#include "ui_detect_rgb10_spv.h"
#include "ui_box_update_rgb10_spv.h"
#include "ui_inpaint_rgb10_spv.h"
#include "ui_patch_rgb10_spv.h"
#include "ui_v2_update_rgb10_spv.h"
#include "ui_v2_lumafill_rgb10_spv.h"
#include "blur_error_rgb10_spv.h"
#include "blur_resolve_rgb10_spv.h"
#endif
// ... and MotionVersion::V2's synthesis of rgb10 (probed on its own: a host build that lists only the colour shaders above gets
// V2 refused for this format, with a reason)
#if __has_include("v2_morph_rgb10_spv.h") && defined(FG_HAVE_RGB10)
#define FG_HAVE_V2_RGB10 1
#include "v2_morph_rgb10_spv.h"
#include "v2_morph_s_rgb10_spv.h"
#include "v2_qmorph_rgb10_spv.h"
#include "v2_qmorph_s_rgb10_spv.h"
#endif
// ... and MotionVersion::Learned's colour passes of rgb10, its duplicate test and static guard (probed on their own, like V2's synthesis)
#if __has_include("ml_dup_rgb10_spv.h") && __has_include("ml_guard_rgb10_spv.h") && defined(FG_HAVE_RGB10) && defined(FG_HAVE_ML)
#define FG_HAVE_ML_RGB10 1
#include "ml_dup_rgb10_spv.h"
#include "ml_guard_rgb10_spv.h"
#endif
// ... and MotionVersion::V3's colour passes of rgb10, its raw planes and its synthesis (probed on their own, like V2's synthesis)
#if __has_include("v3_synth_rgb10_spv.h") && __has_include("v3_synth_l_rgb10_spv.h") && __has_include("v3_planes_rgb10_spv.h") && defined(FG_HAVE_RGB10)
#define FG_HAVE_V3_RGB10 1
#include "v3_synth_rgb10_spv.h"
#include "v3_synth_l_rgb10_spv.h"
#if __has_include("v3_synth_b_rgb10_spv.h")
#define FG_HAVE_V3B_RGB10 1
#include "v3_synth_b_rgb10_spv.h"   // Settings::v3Blend in rgb10 (-DFG_FORMAT_RGB10=1 -DV3B=1); a host build without it runs the rule
#endif
#include "v3_planes_rgb10_spv.h"
#endif
// f16: all its variant headers are generated together (gpu/meson.build `formats` option), so one probe stands for the set.
#if __has_include("luma_f16_spv.h")
#define FG_HAVE_F16 1
#include "luma_f16_spv.h"
#include "error_f16_spv.h"
#include "resolve_f16_spv.h"
#include "fallback_f16_spv.h"
#include "ui_detect_f16_spv.h"
#include "ui_box_update_f16_spv.h"
#include "ui_inpaint_f16_spv.h"
#include "ui_patch_f16_spv.h"
#include "ui_v2_update_f16_spv.h"
#include "ui_v2_lumafill_f16_spv.h"
#include "blur_error_f16_spv.h"
#include "blur_resolve_f16_spv.h"
#endif
// ... and MotionVersion::V2's synthesis of f16 (probed on its own: a host build that lists only the colour shaders above gets
// V2 refused for this format, with a reason)
#if __has_include("v2_morph_f16_spv.h") && defined(FG_HAVE_F16)
#define FG_HAVE_V2_F16 1
#include "v2_morph_f16_spv.h"
#include "v2_morph_s_f16_spv.h"
#include "v2_qmorph_f16_spv.h"
#include "v2_qmorph_s_f16_spv.h"
#endif
#if __has_include("ml_dup_f16_spv.h") && __has_include("ml_guard_f16_spv.h") && defined(FG_HAVE_F16) && defined(FG_HAVE_ML)
#define FG_HAVE_ML_F16 1
#include "ml_dup_f16_spv.h"
#include "ml_guard_f16_spv.h"
#endif
// ... and MotionVersion::V3's colour passes of f16, its raw planes and its synthesis (probed on their own, like V2's synthesis)
#if __has_include("v3_synth_f16_spv.h") && __has_include("v3_synth_l_f16_spv.h") && __has_include("v3_planes_f16_spv.h") && defined(FG_HAVE_F16)
#define FG_HAVE_V3_F16 1
#include "v3_synth_f16_spv.h"
#include "v3_synth_l_f16_spv.h"
#include "v3_planes_f16_spv.h"
#endif
// nv12: all its variant headers are generated together (gpu/meson.build `formats` option), so one probe stands for the set.
#if __has_include("luma_nv12_spv.h")
#define FG_HAVE_NV12 1
#include "luma_nv12_spv.h"
#include "error_nv12_spv.h"
#include "resolve_nv12_spv.h"
#include "fallback_nv12_spv.h"
#include "ui_detect_nv12_spv.h"
#include "ui_box_update_nv12_spv.h"
#include "ui_inpaint_nv12_spv.h"
#include "ui_patch_nv12_spv.h"
#include "ui_v2_update_nv12_spv.h"
#include "ui_v2_lumafill_nv12_spv.h"
#include "blur_error_nv12_spv.h"
#include "blur_resolve_nv12_spv.h"
#include "ui_inpaint_chroma_nv12_spv.h"
#include "ui_patch_chroma_nv12_spv.h"
#endif
// ... and MotionVersion::V2's synthesis of nv12 (probed on its own: a host build that lists only the colour shaders above gets
// V2 refused for this format, with a reason)
#if __has_include("v2_morph_nv12_spv.h") && defined(FG_HAVE_NV12)
#define FG_HAVE_V2_NV12 1
#include "v2_morph_nv12_spv.h"
#include "v2_morph_s_nv12_spv.h"
#include "v2_qmorph_nv12_spv.h"
#include "v2_qmorph_s_nv12_spv.h"
#endif
#if __has_include("ml_dup_nv12_spv.h") && __has_include("ml_guard_nv12_spv.h") && defined(FG_HAVE_NV12) && defined(FG_HAVE_ML)
#define FG_HAVE_ML_NV12 1
#include "ml_dup_nv12_spv.h"
#include "ml_guard_nv12_spv.h"
#endif
// ... and MotionVersion::V3's colour passes of nv12, its raw planes and its synthesis (probed on their own, like V2's synthesis)
#if __has_include("v3_synth_nv12_spv.h") && __has_include("v3_synth_l_nv12_spv.h") && __has_include("v3_planes_nv12_spv.h") && defined(FG_HAVE_NV12)
#define FG_HAVE_V3_NV12 1
#include "v3_synth_nv12_spv.h"
#include "v3_synth_l_nv12_spv.h"
#include "v3_planes_nv12_spv.h"
#endif
// p010: all its variant headers are generated together (gpu/meson.build `formats` option), so one probe stands for the set.
#if __has_include("luma_p010_spv.h")
#define FG_HAVE_P010 1
#include "luma_p010_spv.h"
#include "error_p010_spv.h"
#include "resolve_p010_spv.h"
#include "fallback_p010_spv.h"
#include "ui_detect_p010_spv.h"
#include "ui_box_update_p010_spv.h"
#include "ui_inpaint_p010_spv.h"
#include "ui_patch_p010_spv.h"
#include "ui_v2_update_p010_spv.h"
#include "ui_v2_lumafill_p010_spv.h"
#include "blur_error_p010_spv.h"
#include "blur_resolve_p010_spv.h"
#include "ui_inpaint_chroma_p010_spv.h"
#include "ui_patch_chroma_p010_spv.h"
#endif
// ... and MotionVersion::V2's synthesis of p010 (probed on its own: a host build that lists only the colour shaders above gets
// V2 refused for this format, with a reason)
#if __has_include("v2_morph_p010_spv.h") && defined(FG_HAVE_P010)
#define FG_HAVE_V2_P010 1
#include "v2_morph_p010_spv.h"
#include "v2_morph_s_p010_spv.h"
#include "v2_qmorph_p010_spv.h"
#include "v2_qmorph_s_p010_spv.h"
#endif
#if __has_include("ml_dup_p010_spv.h") && __has_include("ml_guard_p010_spv.h") && defined(FG_HAVE_P010) && defined(FG_HAVE_ML)
#define FG_HAVE_ML_P010 1
#include "ml_dup_p010_spv.h"
#include "ml_guard_p010_spv.h"
#endif
// ... and MotionVersion::V3's colour passes of p010, its raw planes and its synthesis (probed on their own, like V2's synthesis)
#if __has_include("v3_synth_p010_spv.h") && __has_include("v3_synth_l_p010_spv.h") && __has_include("v3_planes_p010_spv.h") && defined(FG_HAVE_P010)
#define FG_HAVE_V3_P010 1
#include "v3_synth_p010_spv.h"
#include "v3_synth_l_p010_spv.h"
#include "v3_planes_p010_spv.h"
#endif
// Settings::debugView (task 136), each build probed as a set (a host build that does not list them gets the view refused by setSettings, not a build error): Rgba8 / Bgra8 plain,
// rgb10 and f16 per format (Nv12 / P010 draw nothing and have none).
#if __has_include("debug_motion_spv.h") && __has_include("debug_motion_v2_spv.h") && __has_include("debug_motion_v3_spv.h") && __has_include("debug_ui_spv.h")
#define FG_HAVE_DBG 1
#include "debug_motion_spv.h"
#include "debug_motion_v2_spv.h"
#include "debug_motion_v3_spv.h"
#include "debug_ui_spv.h"
#endif
#if __has_include("debug_motion_rgb10_spv.h") && __has_include("debug_motion_v2_rgb10_spv.h") && __has_include("debug_motion_v3_rgb10_spv.h") && __has_include("debug_ui_rgb10_spv.h") && defined(FG_HAVE_RGB10)
#define FG_HAVE_DBG_RGB10 1
#include "debug_motion_rgb10_spv.h"
#include "debug_motion_v2_rgb10_spv.h"
#include "debug_motion_v3_rgb10_spv.h"
#include "debug_ui_rgb10_spv.h"
#endif
#if __has_include("debug_motion_f16_spv.h") && __has_include("debug_motion_v2_f16_spv.h") && __has_include("debug_motion_v3_f16_spv.h") && __has_include("debug_ui_f16_spv.h") && defined(FG_HAVE_F16)
#define FG_HAVE_DBG_F16 1
#include "debug_motion_f16_spv.h"
#include "debug_motion_v2_f16_spv.h"
#include "debug_motion_v3_f16_spv.h"
#include "debug_ui_f16_spv.h"
#endif

namespace framegen {

bool formatOf(VkFormat f, Format* out) {
    Format r;
    switch (f) {
    case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_R8G8B8A8_UNORM: r = Format::Rgba8; break;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2R10G10B10_UNORM_PACK32: r = Format::Rgb10; break;
    case VK_FORMAT_R16G16B16A16_SFLOAT: r = Format::RgbaF16; break;
    case VK_FORMAT_R8_UNORM: r = Format::Nv12; break;
    case VK_FORMAT_R16_UNORM: r = Format::P010; break;
    default: return false;
    }
    if (out) *out = r;
    return true;
}

namespace {
struct Spv { const uint32_t* code = nullptr; size_t size = 0; };
#define SPV(n) Spv{n##_spv, sizeof(n##_spv)}
// The format-specific shaders of one Format (the others - motion estimate - are the same for every format).
struct ColourSpv { Spv luma, error, resolve, fallback, uiDetect, uiBoxUpdate, uiInpaint, uiPatch, blurError, blurResolve, uiInpaintC, uiPatchC, uiV2Update, uiV2Fill,
    v2Morph, v2MorphS, v2QMorph, v2QMorphS, mlDup, mlGuard, v3Synth, v3Planes, v3SynthL, v3SynthB,
    dbgMotion, dbgMotionV2, dbgMotionV3, dbgUi; };   // (Settings::debugView: the motion tint of V1 / Learned, V2, V3 and the UI views; empty = this build has none)
#define COLOUR_SET(f) ColourSpv{SPV(luma##f), SPV(error##f), SPV(resolve##f), SPV(fallback##f), SPV(ui_detect##f), \
    SPV(ui_box_update##f), SPV(ui_inpaint##f), SPV(ui_patch##f), SPV(blur_error##f), SPV(blur_resolve##f), Spv{}, Spv{}, \
    SPV(ui_v2_update##f), SPV(ui_v2_lumafill##f), Spv{}, Spv{}, Spv{}, Spv{}}
#define V2_SET(c, f) (c.v2Morph = SPV(v2_morph##f), c.v2MorphS = SPV(v2_morph_s##f), c.v2QMorph = SPV(v2_qmorph##f), c.v2QMorphS = SPV(v2_qmorph_s##f))
#define V3_SET(c, f) (c.v3Synth = SPV(v3_synth##f), c.v3Planes = SPV(v3_planes##f), c.v3SynthL = SPV(v3_synth_l##f))   // (V3's two colour passes: the raw planes and the synthesis)
#define ML_SET(c, f) (c.mlDup = SPV(ml_dup##f), c.mlGuard = SPV(ml_guard##f))
#define DBG_SET(c, f) (c.dbgMotion = SPV(debug_motion##f), c.dbgMotionV2 = SPV(debug_motion_v2##f), c.dbgMotionV3 = SPV(debug_motion_v3##f), c.dbgUi = SPV(debug_ui##f))
// false = this format's variants were not built into the library (meson option `formats`)
bool colourSpv(Format f, ColourSpv& c) {
    switch (f) {
    case Format::Rgba8:
        c = COLOUR_SET(); V2_SET(c, ); V3_SET(c, ); c.v3SynthB = SPV(v3_synth_b);   // (v3Blend: Rgba8 / Bgra8 and Rgb10; the other formats keep it empty = the rule)
#ifdef FG_HAVE_DBG
        DBG_SET(c, );
#endif
#ifdef FG_HAVE_ML
        ML_SET(c, );
#endif
        return true;
#ifdef FG_HAVE_RGB10
    case Format::Rgb10:
        c = COLOUR_SET(_rgb10);
#ifdef FG_HAVE_V2_RGB10
        V2_SET(c, _rgb10);
#endif
#ifdef FG_HAVE_ML_RGB10
        ML_SET(c, _rgb10);
#endif
#ifdef FG_HAVE_V3_RGB10
        V3_SET(c, _rgb10);
#ifdef FG_HAVE_V3B_RGB10
        c.v3SynthB = SPV(v3_synth_b_rgb10);
#endif
#endif
#ifdef FG_HAVE_DBG_RGB10
        DBG_SET(c, _rgb10);
#endif
        return true;
#endif
#ifdef FG_HAVE_F16
    case Format::RgbaF16:
        c = COLOUR_SET(_f16);
#ifdef FG_HAVE_V2_F16
        V2_SET(c, _f16);
#endif
#ifdef FG_HAVE_ML_F16
        ML_SET(c, _f16);
#endif
#ifdef FG_HAVE_V3_F16
        V3_SET(c, _f16);
#endif
#ifdef FG_HAVE_DBG_F16
        DBG_SET(c, _f16);
#endif
        return true;
#endif
#ifdef FG_HAVE_NV12
    case Format::Nv12:
        c = COLOUR_SET(_nv12); c.uiInpaintC = SPV(ui_inpaint_chroma_nv12); c.uiPatchC = SPV(ui_patch_chroma_nv12);
#ifdef FG_HAVE_V2_NV12
        V2_SET(c, _nv12);
#endif
#ifdef FG_HAVE_ML_NV12
        ML_SET(c, _nv12);
#endif
#ifdef FG_HAVE_V3_NV12
        V3_SET(c, _nv12);
#endif
        return true;
#endif
#ifdef FG_HAVE_P010
    case Format::P010:
        c = COLOUR_SET(_p010); c.uiInpaintC = SPV(ui_inpaint_chroma_p010); c.uiPatchC = SPV(ui_patch_chroma_p010);
#ifdef FG_HAVE_V2_P010
        V2_SET(c, _p010);
#endif
#ifdef FG_HAVE_ML_P010
        ML_SET(c, _p010);
#endif
#ifdef FG_HAVE_V3_P010
        V3_SET(c, _p010);
#endif
        return true;
#endif
    default: return false;
    }
}
inline bool isYuv(Format f) { return f == Format::Nv12 || f == Format::P010; }
const char* const kDbgNotBuilt = "the Settings::debugView shader variants of this format (debug_motion, debug_motion_v2, debug_motion_v3, debug_ui) are not built into the library";
const char* const kV2NotBuilt = "the MotionVersion::V2 shader variants of this format (v2_morph, v2_morph_s, v2_qmorph, v2_qmorph_s) are not built into the library";
const char* const kV3NotBuilt = "the MotionVersion::V3 shader variants of this format (v3_planes, v3_synth, v3_synth_l) are not built into the library";
const char* const kMlNotBuilt = "the MotionVersion::Learned shaders of this format (ml_dup, ml_reduce, ml_pyr, ml_match, ml_conv, ml_out, ml_zero, ml_guard) or its weights (ml_weights.h, gpu/ml/embed_weights.py) are not built into the library";
const char* const kMlFlowScale = "MotionVersion::Learned needs flowScale 2 (its nets were trained at 2)";
constexpr int groups4(int c) { return (c + 3) / 4; }
// The one net of MotionVersion::Learned (LearnedNet::E2v5, ml-fg's export run "e2v5"; the other nine were removed from the library in task 154, they are in the
// git history): its width (channels per hidden layer; the layout and the pipelines depend on it) and its first-conv input channels (14: the luma net).
struct MlNetInfo { const char* name; int width, inCh; };
constexpr MlNetInfo kMlNetInfo[] = {{"e2v5", 24, 14}};
constexpr int kMlNets = (int)(sizeof(kMlNetInfo) / sizeof(kMlNetInfo[0]));
static_assert(kMlNets == 1, "one embedded net");
#ifdef FG_HAVE_ML
// The embedded net (gpu/ml/embed_weights.py): weights in shaders/ml_conv.comp's layout, the table of each
// (level, layer)'s weight / bias offsets (vec4 units, relative to the net's start in the weights buffer)
struct MlNet { const uint32_t* w; size_t words; uint32_t width; uint32_t inCh; const uint32_t (*table)[5][2]; };
#define ML_NET(n) MlNet{kMlWeights_##n, sizeof(kMlWeights_##n) / 4, kMlWidth_##n, kMlInCh_##n, kMlTable_##n}
constexpr MlNet kMlNet[] = {ML_NET(e2v5)};
static_assert(kMlNet[0].width == (uint32_t)kMlNetInfo[0].width && kMlNet[0].inCh == (uint32_t)kMlNetInfo[0].inCh && kMlVariant_e2v5 == 0,
              "kMlNetInfo's width and input channels are those of ml_weights.h (a net loaded into the wrong pipeline fails the build)");
// (every net's weights are a whole number of vec4: embed_weights.py pads each layer to 4 channels, so each net starts on a vec4)
#endif
// the view / clean-copy formats of a Format: the library's own clean copies are in the source's own format, in logical channel order
VkFormat cleanFormat(Format f) {
    switch (f) {
    case Format::Rgb10: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case Format::RgbaF16: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case Format::Nv12: return VK_FORMAT_R8_UNORM;
    case Format::P010: return VK_FORMAT_R16_UNORM;
    default: return VK_FORMAT_R8G8B8A8_UNORM;
    }
}
VkFormat cleanFormatC(Format f) { return f == Format::P010 ? VK_FORMAT_R16G16_UNORM : VK_FORMAT_R8G8_UNORM; }
}  // namespace

#define FG_FUNCS(X) \
    X(CreateBuffer) X(DestroyBuffer) X(GetBufferMemoryRequirements) X(AllocateMemory) X(FreeMemory) \
    X(BindBufferMemory) X(CreateSampler) X(DestroySampler) X(CreateDescriptorSetLayout) \
    X(DestroyDescriptorSetLayout) X(CreatePipelineLayout) X(DestroyPipelineLayout) X(CreateShaderModule) \
    X(DestroyShaderModule) X(CreateComputePipelines) X(DestroyPipeline) X(CreateDescriptorPool) \
    X(DestroyDescriptorPool) X(AllocateDescriptorSets) X(UpdateDescriptorSets) X(CreateQueryPool) \
    X(DestroyQueryPool) X(GetQueryPoolResults) X(CmdBindPipeline) X(CmdBindDescriptorSets) \
    X(CmdPushConstants) X(CmdDispatch) X(CmdPipelineBarrier) X(CmdFillBuffer) X(CmdResetQueryPool) \
    X(CmdWriteTimestamp) X(CreateImage) X(DestroyImage) X(GetImageMemoryRequirements) X(BindImageMemory) \
    X(CreateImageView) X(DestroyImageView) X(CmdCopyBuffer) X(CmdUpdateBuffer)

// Must match the push-constant block in shaders/motion_v1.glsl.
struct PC {
    int32_t w, h, bw, bh, pw, ph, offA, offC, srcA, srcC, fIn, fOut, fPar, radius, medIdx,
        W, H, fw, fh, pf, fscale, sadAcc;
    float s, wA, wC;
    float lambda, smoothBonus, zeroBonus, cutSad, trust0, trustScale, globalFrac;
};
static_assert(sizeof(PC) <= 128, "push constants beyond the guaranteed minimum");
// The default search shaders hold windows up to +-6 (FG_MAXR in search.comp / v2_search.comp); a radius above it takes the
// *_wide build (FG_MAXR 12, the largest that fits the guaranteed 16 KiB of shared memory). Chosen from the settings only.
static const int kSearchMaxR = 6;
static const int kEOff = 64;   // == E_OFF in motion_v1.glsl

// The longest integer half-vector (flow pixels) the search can reach at level j (0 = finest) of an n-level pyramid: the top level's
// radius Rt, then 2 x the parent's + R per level. Used to pick the wide median histogram and the wide Quality claims.
static int64_t reachBound(int n, int j, int R, int Rt) {
    const int64_t a = (int64_t)1 << (n - 1 - j);
    return Rt * a + (int64_t)R * (a - 1);
}
// Are the reach settings off their defaults? The wide variants of the two clipping passes are used only then (default settings keep
// the default shaders at every size, so their output never changes), and only where the bound below says they are needed.
static bool reachNonDefault(const Settings& s) {
    const Settings d;
    return s.searchRadius != d.searchRadius || s.searchRadiusTop != d.searchRadiusTop || s.pyramidMinSide != d.pyramidMinSide;
}
// The median histogram of the finest-to-coarsest step whose parent is level `parent` (shaders/medhist.comp, v2_medhist.comp: 256 bins
// of [-128, 127] parent pixels): -1 = the default shader, else the *_wide shader's bin shift SH (first-pass bins of 2^SH parent pixels,
// the smallest that holds reachBound of the parent, then an exact second pass; pushed in pc.radius, which the pass does not read). FG_FORCE_WIDE (a test hook, set
// with -Dcpp_args=-DFG_FORCE_WIDE) uses the wide shader everywhere, FG_NO_WIDE never (shifttest.py's counter-check).
static int medhistShift(const Settings& s, int n, int parent) {
    const int64_t b = reachBound(n, parent, s.searchRadius, s.searchRadiusTop);
    int sh = 0;
    while ((b >> sh) > 127) sh++;
#if defined(FG_NO_WIDE)
    return -1;
#elif defined(FG_FORCE_WIDE)
    return sh;
#else
    return reachNonDefault(s) && b > 127 ? sh : -1;
#endif
}
// V2 Quality's claims (shaders/v2_qown.comp, v2_qcov.comp): the *_wide pair when the finest half-vector can leave qQuant's
// [-128, 127.75] flow px (integer bound + 1.25 of V2's fit and fractional steps > 127.75, i.e. bound > 126).
static bool qualityWide(const Settings& s, int n) {
#if defined(FG_NO_WIDE)
    return false;
#elif defined(FG_FORCE_WIDE)
    return true;
#else
    return reachNonDefault(s) && reachBound(n, 0, s.searchRadius, s.searchRadiusTop) > 126;
#endif
}

// Must match the push-constant block in shaders/v2_motion.glsl (V2 Performance's synthesis passes; V2's estimate pushes PC). Pushed
// over the same range of the same pipeline layout as PC.
struct V2PC {
    int32_t W, H, fw, fh, bw, bh, fscale, fIn, n, gauss, fix, cnt;
    float cutSad, globalFrac, sigma2Inv, trustMid;
    float t[16];
};
static_assert(sizeof(V2PC) == sizeof(PC), "V2PC is pushed over PC's range");
// Must match the push-constant block in shaders/ml_common.glsl (MotionVersion::Learned's passes). Pushed over the same range of the
// same pipeline layout as PC.
struct MLPC {
    int32_t W, H, fw, fh, pw, ph, lvl, actIn, actOut, wOff, bOff, nPart, offYa, offYc, bw, bh, pf, mode;
    float dupDiff;
    int32_t ext, nin, feat, xwOff, xbOff;
    int32_t pad_[8];
};
static_assert(sizeof(MLPC) == sizeof(PC), "MLPC is pushed over PC's range");
// Must match the push-constant block in shaders/v3_common.glsl (MotionVersion::V3's passes, gpu/V3.md section 5). Pushed over the same range of
// the same pipeline layout as PC. The first 8 ints are V2PC's, so v2_static.comp can be pushed with it. (The page's listing adds up to 108 bytes:
// the padding is 7 floats, not 2.)
struct V3PC {
    int32_t W, H, fw, fh, bw, bh, fscale, fIn;
    int32_t lw, lh, gw, gh;
    int32_t lvl, iter;
    int32_t mode, n;
    float t0, t1;
    float cutThresh, globalFrac;
    float smear, occlSoft, unsure;
    float cx, cy;
    int32_t fdOff;   // Y1: word offset of FD in V3B_FLOW (16 B); FF = +4 N, TF = +8 N
    float hudB;      // task 105: Settings::hudBonus in V3 cost units (0 = off: every V3 pass is then exactly the pass without the bonus)
    float hp[5];     // task 105: the bonus's ramps (v3_est.glsl): c0 lo, hi; detail lo; gate lo, hi (frame px)
};
static_assert(sizeof(V3PC) == sizeof(PC), "V3PC is pushed over PC's range");
// task 105: hudBonus in V3 (gpu/V3.md "As built (HUD bonus, task 105)"): K converts V1's levels (5x5 mean abs luma error) to V3's cost units; P = the ramps
const float kV3HudK = 0.7f;
const float kV3HudP[5] = {1.0f, 3.0f, 1.5f, 4.0f, 10.0f};
enum V3Mode : int { V3M_GAUSS = 1, V3M_DETAIL = 2, V3M_DUMP = 8, V3M_BLEND_SHIFT = 4 };   // (bits 4..7: Settings::v3Blend, <= kV3BlendMax <= 15)   // synth mode bits (shaders/v3_common.glsl)
// MotionVersion::Learned's duplicate test: FGR's own, proto/harness.py DUP_DIFF (and gpu/compare.py): a pair whose mean over the
// pixels of the channel-mean |a - c| is at most this many 8-bit levels is a duplicate ("re-encoded duplicate frames differ by up to
// ~0.5"). The net is skipped there and the field is zero (shaders/ml_dup.comp).
static const float kMlDupDiff = 0.6f;
static const int kMlLevels = 5, kMlLayers = 5;   // == ML_LEVELS in ml_common.glsl; layers per level (4 hidden + the linear one)
// A level of fewer 16x16 tiles than this runs its convs split, one output group per workgroup (shaders/ml_conv.comp GOT): on the RX
// 7900 XTX a whole-pixel workgroup of e2v3 takes ~15 us however small the level, so the coarse levels cost latency, not work.
static const uint32_t kMlSplitTiles = 256;

// ==== Settings::learnedStaticGuard: THE constants of the static guard (shaders/ml_guard.comp), the one place they are defined ====
// ml-fg scripts/static_guard.py DEFAULTS (authoritative; superdoc/features/static-guard.md): MONZikWasTaken/Framegen's 0..1
// thresholds x 255. Levels are 8-bit levels in every Format (the mismatch metric's convention), distances full-resolution pixels.
// ml-fg may send other final defaults after its scoring (I31): change them here only.
struct StaticGuardParams {
    float lo, hi;              // colorStatic = 1 - smoothstep(lo, hi, d), d = 3x3 mean of the max-channel |A - C|
    float haloLo, haloHi;      // uncertain = smoothstep(haloLo, haloHi, d); wHalo = 1 - smoothstep(haloLo, haloHi, dMin)
    int halo;                  // dMin = the min of the per-pixel |A - C| over (2 halo + 1)^2 (9x9), 1..kGuardHaloMax
    float motionLo, motionHi;  // gate = 1 - smoothstep(motionLo, motionHi, motion), motion in full-res px of the larger warp side
    float wMin;                // wStatic <= wMin leaves the pixel bit-exact
};
static constexpr StaticGuardParams kStaticGuard = {7.65f, 22.95f, 5.1f, 7.65f, 4, 0.5f, 1.0f, 0.001f};
static constexpr int kGuardHaloMax = 8;   // == GUARD_HALO_MAX in shaders/ml_guard.comp (its shared-memory apron)
static_assert(kStaticGuard.halo >= 1 && kStaticGuard.halo <= kGuardHaloMax, "ml_guard.comp's apron");
// Must match the push-constant block in shaders/ml_guard.comp. Pushed over the same range of the same pipeline layout as PC.
struct GuardPC {
    int32_t W, H, fw, fh, pf, bw, bh, cnt0, nCnt, halo;
    float wA, wC, motionK, cutSad, globalFrac;
    float lo, hi, haloLo, haloHi, motionLo, motionHi, wMin;
    int32_t pad_[10];
};
static_assert(sizeof(GuardPC) == sizeof(PC), "GuardPC is pushed over PC's range");

// V2 Performance's field clean-up (v2_clean.comp): sigma of the bilateral smoothing's range weight, in flow pixels of half-vector
// (1 flow px of half-vector = 2 * flowScale px of motion between the real frames).
static const float kV2CleanSigma = 1.0f;

// Must match the push-constant block in shaders/ui_common.glsl.
struct UiPC {
    int32_t W, H, boxX, boxY, boxW, boxH, cell, cw, ch, wholeScreen, still;
    float thr;
    int32_t reach, reset, tilesX, tilesY, stillLimit, maskX, maskY, maskW, maskH, maskStride, maskOff, gate, track;
    float cthr;                                          // CrosshairV2 (appended: the other passes never read these)
    int32_t leave, reentry, binary, lumaOff, open;       // open: ui_inpaint*.comp, see ui_common.glsl
    int32_t dbg;                                         // Settings::debugView's pass (debug_ui.comp only; the other UI passes' blocks end before it): 1 = UiMask, 2 = UiClean, | 16 = the box outline
};
static_assert(sizeof(UiPC) <= 128, "push constants beyond the guaranteed minimum");

// Must match the push-constant block in shaders/debug_motion.comp (Settings::debugView = Motion; pushed over PC's range of the synth pipeline layout).
struct DbgPC {
    int32_t W, H, fw, fh, fscale, pf, bw, bh, quality, fdOff, gOff, nInt;
    float m0, m1, maxA, cscale;
};
static_assert(sizeof(DbgPC) <= sizeof(PC), "DbgPC is pushed over PC's range");
enum DbgPipe { D_MOTION, D_MOTION_V2, D_MOTION_V3, D_UI, D_COUNT };
enum UiPipe { U_DETECT, U_BOX_UPDATE, U_BOX_MASK, U_MASK, U_INPAINT, U_PATCH, U_V2_UPDATE, U_V2_MASK, U_V2_FILL, U_COUNT, U_INPAINT_C = U_COUNT, U_PATCH_C, U_COUNT_ALL };   // _C: the CbCr plane (YUV only)
static const int kUiCell = 2;    // whole-screen detection resolution: 1/kUiCell per axis (1, 2 or 4; 8 x cell divides the tile)
static const int kUiTile = 32;   // == TILE in ui_common.glsl
static const int kUiInpTile = 16;   // == T in ui_inpaint.comp

// Must match the push-constant block in shaders/blur_motion_v1.glsl.
struct BlurPC {
    int32_t W, H, fw, fh, fscale, n, bw, bh, k0, pf, gauss, fix;
    float cutSad, trust0, trustScale, globalFrac;
    float t[16];
};
static_assert(sizeof(BlurPC) <= 128, "push constants beyond the guaranteed minimum");
enum BlurPipe { B_LOOKUP, B_ERROR, B_RESOLVE, B_COUNT_ };
static const int kBlurCnt0 = 32;   // == CNT0 in blur_motion_v1.glsl: u[] words 32.. hold the per-sample untrusted counts

struct UiBuf { VkBuffer b = VK_NULL_HANDLE; VkDeviceMemory m = VK_NULL_HANDLE; VkDeviceSize size = 0; };
struct RejPair { VkBuffer a = VK_NULL_HANDLE, c = VK_NULL_HANDLE; };   // the V2 masks of prev / curr, bound to synth sets 11 / 12
struct UiBox { int x, y, w, h; bool operator==(const UiBox& o) const { return x == o.x && y == o.y && w == o.w && h == o.h; } };
// One observed real frame: the clean copy, the mask of the box (full res) and, whole screen, of the whole frame.
struct UiEntry {
    VkImage img = VK_NULL_HANDLE; VkDeviceMemory imgMem = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE;
    VkImage imgC = VK_NULL_HANDLE; VkDeviceMemory imgMemC = VK_NULL_HANDLE; VkImageView viewC = VK_NULL_HANDLE;   // YUV: the CbCr plane
    Frame frame() const { return Frame(view, viewC); }
    UiBuf box;   // header (4 words) + the box mask; grown lazily when the box grows
    UiBuf ws;    // header + tile flags + the full-frame mask (allocated with WholeScreen)
    VkImageView key = VK_NULL_HANDLE;   // the host view this entry describes (null = dead)
    uint64_t seq = 0;                   // observe order (LRU, and "is this still the frame the estimate saw")
    UiBox bx{};                         // the box at observe time
    UiBox wr{};                         // its work rectangle at observe time (the box plus its margin): the rectangle of the mask buffer, the inpaint and the patch
    bool hasBox = false, hasWs = false;
    bool v2 = false;                    // observed under CrosshairV2: the box buffer holds a V2 mask (255 paste / 64 confirmed), no clean copy
    bool active = false;                // in the LRU set (Settings::uiHistory of them); an inactive one keeps its memory
};

// P_V2_*: MotionVersion::V2's own passes (empty in V1's group; V2's group leaves V1's P_MEDHIST / P_SEARCH / P_SUBPIX empty instead).
// P_ML_*: MotionVersion::Learned's (its group has V1's luma and synthesis, none of V1's search); the convs are one pipeline per layer
// shape (first / hidden / last), shaders/ml_conv.comp specialised.
enum Pipe { P_LUMA, P_DOWN, P_MEDHIST, P_MEDHIST_WIDE, P_SEARCH, P_SEARCH_WIDE, P_MEDIAN3, P_SUBPIX, P_PIXFIELD, P_ERROR, P_RESOLVE, P_FALLBACK, P_LOOKUP,
            P_V2_MEDHIST, P_V2_MEDHIST_WIDE, P_V2_SEARCH, P_V2_SEARCH_WIDE, P_V2_CLEAN, P_V2_MORPH, P_V2_REPICK, P_V2_QFIELD, P_V2_QOWN, P_V2_QOWN_WIDE, P_V2_QCOV, P_V2_QCOV_WIDE, P_V2_QMORPH, P_V2_STATIC, P_V2_ENCLOSE, P_V2_MORPH_S, P_V2_QMORPH_S,
            P_ML_DUP, P_ML_REDUCE, P_ML_PYR, P_ML_OUT, P_ML_MATCH, P_ML_CONV = P_ML_MATCH + 2, P_ML_ZERO = P_ML_CONV + 10, P_ML_GUARD,
            // MotionVersion::V3 (gpu/V3.md section 7; the skeleton fills P_LUMA and P_V3_SYNTH only, the others are reserved for the steps E1..Y2)
            P_V3_PLANES, P_V3_PYR, P_V3_HIST, P_V3_SEARCH, P_V3_WIDE, P_V3_STATS, P_V3_DENSE, P_V3_OCCL, P_V3_FILL, P_V3_ENCLOSE, P_V3_DECIDE, P_V3_PROJECT, P_V3_SYNTH,
            P_V3_SYNTH_L,   // (hook (l) v3DetailSplit: the synthesis built with -DV3L=1)
            P_V3_SYNTH_B,   // (Settings::v3Blend: the synthesis built with -DV3B=1, Rgba8 / Bgra8 and Rgb10)
            P_V3_SEARCH_J, P_V3_SEARCH_TOP_J, P_V3_DENSE_J,   // (hook (j) v3ColourWindow: the search and dense passes built with the colour-window weights; the plain ones stay as they are)
            P_V3_SEARCH_TOP,   // (E1: the top level's exhaustive search, its own pipeline: its shared patch must not cost the other levels' occupancy)
            P_V3_ANCHOR, P_V3_GUARD,   // (task 104, Settings::v3Guard: created with the V2 estimate passes P_DOWN / P_V2_MEDHIST* / P_V2_SEARCH* / P_V2_REPICK / P_V2_CLEAN / P_V2_QFIELD in V3's group)
            P_V35_REG, P_V35_DENSE, P_V35_STATS, P_V35_RGN, P_V35_SEARCH,   // (Settings::v3Motion = 1, V3.5: created only while it is on, v35Ensure)
            P_COUNT };
// The match and conv pipelines exist per activation storage (half: Settings::learnedHalf, specialization constant ML_HALF); convs per
// shape (0 = first, 14 -> w; 1 = hidden, w -> w; 2 = last, w -> 2) and, but the last, whole or split (one output group per
// workgroup: the small levels)
inline Pipe mlMatchPipe(bool half) { return (Pipe)(P_ML_MATCH + half); }
inline Pipe mlConvPipe(bool half, int shape, bool split) {
    return (Pipe)(P_ML_CONV + 5 * half + (shape == 2 ? 4 : 2 * shape + split));
}

struct Level { int w, h, bw, bh, offA, offC, fS, fM; };

// The resources of one motion version (Settings::motionVersion): created by motionEnsure() the first time the version is selected
// (init, setSettings, reconfigure), kept while another one is selected, freed by releaseUnusedMotion() / destroy(). V1: the pipelines
// of the estimate / synth passes and of the motion-blur passes over the shared layouts, set rings and scratch buffers. V2: the
// pipelines of BOTH V2Modes (the mode is runtime: a switch creates nothing) and none of V1's but luma and pyramid. Its estimate is V1's
// block field computed by its own, faster passes (P_V2_MEDHIST / SEARCH, shaders/v2_est.glsl); both modes then re-pick and clean it
// (P_V2_REPICK / CLEAN); Performance morphs with it in one pass (P_V2_MORPH), Quality derives its per-pixel field and coverage
// (P_V2_QFIELD / QOWN / QCOV, shaders/v2_quality.glsl) and morphs with those (P_V2_QMORPH). Its buffers (`bufs`, created by
// motionSized()) are its own. No motion-blur pipelines: both morphs average the blur's sub-frames themselves.
// Learned: V1's luma, synthesis and motion-blur pipelines plus its own (P_ML_*); its buffers (MLB_*) are the net's state and
// activations (sized for the selected net, grown when a wider one is selected); the weights of both nets (Impl::mlWeights) are
// size-independent and created with the pipelines.
static const int kMotionVersions = 4;   // MotionVersion::V1, V2, Learned, V3
// MotionGroup::bufs of V2: binding 13 (vec2): the cleaned block field, the fitted finest field (bw x bh each), Quality's per-pixel
// field (fw x fh); binding 14 (uint): Quality's claims, own claims and coverage (4 x fw x fh, shaders/v2_quality.glsl)
enum { V2B_FIELD, V2B_QUAL, V2B_COUNT };
// MotionGroup::bufs of Learned: binding 13 (float): the pyramid, the field of every level, the partial sums; binding 14 (vec4): the
// activations, two regions (shaders/ml_common.glsl)
enum { MLB_STATE, MLB_ACT, MLB_COUNT };
// MotionGroup::bufs of V3 (gpu/V3.md section 4), bound to bindings 13, 14, 15 of V3's sets: the raw planes and features of both frames at
// all levels (uvec2 texels), the static layout + maps + the pair's state G (uint words), the block records / dense and filled fields /
// T-field (floats)
enum { V3B_PYR, V3B_AUX, V3B_FLOW, V3B_COUNT };
// Settings::v3Guard: V2's two estimate buffers (the V2B_FIELD and V2B_QUAL layouts of V2's group), held in V3's own group beside the three above
// while the setting has been on (appended to V3's MotionGroup::bufs: 5 buffers then, 3 before)
enum { V3B_GFIELD = V3B_COUNT, V3B_GQUAL, V3B_GCOUNT };
struct MotionGroup {
    VkPipeline pipes[P_COUNT]{};
    VkPipeline blurPipes[B_COUNT_]{};
    std::vector<UiBuf> bufs;   // size-dependent buffers of this version only (V1: none; V2: V2B_*), freed with the size like buf[]
    bool live = false;         // created (the pipelines; the buffers whenever a size is held)
};

// The device limits a frame size must fit (InitInfo::getProps2; without it the guarantees of every Vulkan device, except the
// storage-buffer range, whose guarantee of 128 MiB the library has always gone past on the host's word: the largest range a
// device can report, 4 GiB - 1)
struct DevLimits {
    uint32_t img2D = 0;                          // maxImageDimension2D (0 = unknown: the host's images exist, so they fit)
    uint64_t sbRange = 0xFFFFFFFFull;            // maxStorageBufferRange
    uint32_t wg[3] = {65535u, 65535u, 65535u};   // maxComputeWorkGroupCount
};
// The motion estimate's layout of a size: the flow resolution, its pyramid (interp.flow(): luma at 1/flowScale, cv2 rounding, down to
// minSide = Settings::pyramidMinSide, 48 by default), the luma words and the field entries (vec2) of the shared buffers. createSizeDependent and sizeLimitError use it.
struct SizeLayout {
    int fw = 0, fh = 0;
    std::vector<Level> lv;
    int lumaWords = 0, fieldSub = 0, fieldPF = 0, fieldT = 0, fieldEntries = 0;
};
static int rintHalfEven(double v) { return (int)std::nearbyint(v); }   // == cv2 saturate_cast<int>
static SizeLayout sizeLayout(uint32_t W, uint32_t H, uint32_t flowScale, uint32_t minSide) {
    SizeLayout L;
    L.fw = rintHalfEven(W * (1.0 / flowScale));
    L.fh = rintHalfEven(H * (1.0 / flowScale));
    int w = L.fw, h = L.fh, lumaOff = 0, fieldOff = 0;
    for (;;) {
        Level l{};
        l.w = w; l.h = h;
        l.bw = (w + 7) / 8; l.bh = (h + 7) / 8;   // pad_to_block
        l.offA = lumaOff; l.offC = lumaOff + w * h;
        lumaOff += 2 * w * h;
        l.fS = fieldOff; l.fM = fieldOff + l.bw * l.bh;
        fieldOff += 2 * l.bw * l.bh;
        L.lv.push_back(l);
        if ((uint32_t)(std::min(w, h) / 2) < minSide) break;
        w = rintHalfEven(w * 0.5);
        h = rintHalfEven(h * 0.5);
    }
    L.fieldSub = fieldOff;
    fieldOff += L.lv[0].bw * L.lv[0].bh;
    L.fieldPF = fieldOff;
    fieldOff += L.fw * L.fh;
    L.fieldT = fieldOff;   // per-pixel field looked up at time t (synth scratch, t != 0.5)
    fieldOff += L.fw * L.fh;
    L.lumaWords = lumaOff;
    L.fieldEntries = fieldOff;
    return L;
}
// MotionVersion::Learned's layout of a size (shaders/ml_common.glsl): the net's padded level 0 (fw, fh rounded up to a multiple of
// 16), the sum S of its 5 level sizes, the partial sums of its two reductions, and per net the activation regions: R0 holds the
// input (14 channels, 4 groups; colour 25, 7) and two hidden layers (width / 4 groups each), R1 the other two.
inline int mlWidth() { return kMlNetInfo[0].width; }
struct MlLayout {
    int pw = 0, ph = 0, n0 = 0, S = 0, nDup = 0, nOut = 0;
    int gi = 4;   // the channel groups of the first-layer input
    VkDeviceSize stateBytes = 0;
    int g = 0, r0 = 0, groups = 0;   // the net's groups per hidden layer, R0's groups, R0 + R1
    VkDeviceSize elem = 0, actBytes = 0;   // bytes per group of 4 channels (16 float32, 8 half), the activations
};
static MlLayout mlLayout(uint32_t W, uint32_t H, int fw, int fh, const Settings& s) {
    MlLayout L;
    L.pw = (fw + 15) & ~15;
    L.ph = (fh + 15) & ~15;
    L.n0 = L.pw * L.ph;
    for (int l = 0; l < kMlLevels; l++) L.S += (L.pw >> l) * (L.ph >> l);
    L.nDup = (int)(((W + 31) / 32) * ((H + 31) / 32));   // ml_dup.comp: 32x32 pixels per workgroup
    L.nOut = ((fw + 15) / 16) * ((fh + 15) / 16);       // ml_out.comp: 16x16 flow pixels per workgroup
    L.gi = groups4(kMlNetInfo[0].inCh);
    // ml_common.glsl's state: pyramid + h (4S), partial sums
    L.stateBytes = ((VkDeviceSize)4 * L.S + std::max(L.nDup, L.nOut)) * 4;
    L.g = mlWidth() / 4;
    L.r0 = std::max(L.gi, L.g);
    L.groups = L.r0 + L.g;
    L.elem = s.learnedHalf ? 8 : 16;
    L.actBytes = (VkDeviceSize)L.groups * L.n0 * L.elem;
    return L;
}

// MotionVersion::V3's layout of a size (gpu/V3.md sections 2 and 4): its own pyramid (halve while the short side exceeds 32; level 0 = the
// flow grid; 8x8 blocks at level 0, 4x4 above) and the sizes of its three buffers (ns = the T-field slots held: 1 in the skeleton).
struct V3Layout {
    int fw = 0, fh = 0, n = 0;
    int w[16]{}, h[16]{}, gw[16]{}, gh[16]{};
    uint64_t N = 0, N8 = 0, P = 0, B = 0;
    uint64_t auxWords = 0;
    uint64_t pyrBytes() const { return 32 * P; }
    uint64_t auxBytes() const { return 4 * auxWords; }
    uint64_t flowBytes(uint64_t ns) const { return 64 * B + 32 * N + 16 * N * ns; }
    uint64_t gOff() const { return 5 * N + 3 * N8; }   // words of V3B_AUX before the state G
};
static const int kV3GWords = 4096;
static V3Layout v3Layout(uint32_t W, uint32_t H, uint32_t flowScale) {
    V3Layout L;
    L.fw = rintHalfEven(W * (1.0 / flowScale));
    L.fh = rintHalfEven(H * (1.0 / flowScale));
    int w = L.fw, h = L.fh;
    for (;;) {
        const int bs = L.n == 0 ? 8 : 4;
        L.w[L.n] = w; L.h[L.n] = h; L.gw[L.n] = (w + bs - 1) / bs; L.gh[L.n] = (h + bs - 1) / bs;
        L.P += (uint64_t)w * h;
        L.B += (uint64_t)L.gw[L.n] * L.gh[L.n];
        L.n++;
        if (std::min(w, h) <= 32 || L.n == 16) break;
        w = rintHalfEven(w * 0.5);
        h = rintHalfEven(h * 0.5);
    }
    L.N = (uint64_t)L.fw * L.fh;
    L.N8 = (uint64_t)((L.fw + 7) / 8) * ((L.fh + 7) / 8);
    L.auxWords = 5 * L.N + 3 * L.N8 + kV3GWords;
    return L;
}

// Null if a W x H frame fits the device under these settings (the selected motion version's buffers and dispatches), else why not.
// Below kMinSize the flow resolution degenerates (W or H = 1 -> size-0 buffers). The largest buffers grow with W * H / flowScale^2
// (V2 Quality's claims and coverage, V1's motion-blur field scratch), the widest dispatch with W / 8 (and V2's static-mask cells).
static const char* sizeLimitError(uint32_t W, uint32_t H, const Settings& s, const DevLimits& lim) {
    if (W < Interpolator::kMinSize || H < Interpolator::kMinSize) return "the frame size is below Interpolator::kMinSize (16 pixels) in width or height";
    if (lim.img2D && (W > lim.img2D || H > lim.img2D)) return "the frame size exceeds the device's maxImageDimension2D";
    if ((uint64_t)W * H > (1ull << 30)) return "the frame size exceeds 2^30 pixels (32-bit indexing of the library's buffers)";
    const SizeLayout L = sizeLayout(W, H, s.flowScale, s.pyramidMinSide);
    const uint64_t fwh = (uint64_t)L.fw * L.fh, cells = (uint64_t)((L.fw + 7) / 8) * ((L.fh + 7) / 8);
    uint64_t big = std::max({(uint64_t)L.lumaWords * 4, (uint64_t)L.fieldEntries * 8, (uint64_t)(64 + (W + 3) / 4 * (uint64_t)H) * 4});
    uint64_t gx = (W + 7) / 8, gy = (H + 7) / 8;
    if (s.motionVersion == MotionVersion::V3) {
        // V3 owns three buffers (the pyramid, the aux words, the flow records; no V1 blur scratch: its blur is its synthesis) and its widest
        // dispatches are v3_synth (W / 8), v3_dense ((fw + 4) / 8) and v3_search (the widest block grid); v3_wide / v3_project use z <= 16
        const V3Layout V = v3Layout(W, H, s.flowScale);
        big = std::max({big, V.pyrBytes(), V.auxBytes(), V.flowBytes(1)});
        gx = std::max<uint64_t>(gx, (uint64_t)(V.fw + 4 + 7) / 8);
        for (int l = 0; l < V.n; l++) gx = std::max<uint64_t>(gx, V.gw[l]);
        if (lim.wg[2] < 16) return "a dispatch of MotionVersion::V3 exceeds the device's maxComputeWorkGroupCount in z (16 needed)";
    } else if (s.motionVersion == MotionVersion::V2) {
        big = std::max({big, ((uint64_t)L.lv[0].bw * L.lv[0].bh * 2 + fwh) * 8, (fwh * 5 + 3 * cells) * 4});
        gx = std::max(gx, (cells + 7) / 8);   // v2_enclose.comp: 8 cells per workgroup, one row of workgroups
    } else {
        if (s.motionVersion == MotionVersion::Learned) {   // the net's state and activations (its dispatches are narrower than W / 8)
            const MlLayout M = mlLayout(W, H, L.fw, L.fh, s);
            big = std::max({big, (uint64_t)M.stateBytes, (uint64_t)M.actBytes});
        }
        if (s.blurSamples >= 2)   // recordSynthBlur's scratch (grown by the first blurred record)
            big = std::max({big, (uint64_t)s.blurSamples * fwh * 8, (uint64_t)s.blurSamples * ((W + 3) / 4) * (uint64_t)H * 4});
    }
    if (big > lim.sbRange) return "a buffer of this frame size exceeds the device's maxStorageBufferRange (a larger flowScale needs less)";
    if (gx > lim.wg[0] || gy > lim.wg[1]) return "a dispatch of this frame size exceeds the device's maxComputeWorkGroupCount (a larger flowScale needs fewer)";
    return nullptr;
}

struct Interpolator::Impl {
#define X(n) PFN_vk##n n = nullptr;
    FG_FUNCS(X)
#undef X
    InitInfo info{};
    Format fmt = Format::Rgba8;
    bool yuv = false;
    ColourSpv cs{};
    VkFormat cleanFmt = VK_FORMAT_R8G8B8A8_UNORM, cleanFmtC = VK_FORMAT_UNDEFINED;
    Settings set{};   // active settings (info.settings is only the init value)
    const char* v2Why = nullptr;   // why the device cannot run MotionVersion::V2 (InitInfo::getProps2), null = it can / unchecked
    DevLimits lim{};                // the device limits a size is checked against (sizeLimitError)
    const char* sizeWhy = nullptr;  // why the last rebuild() refused its size (null: it did not, or an allocation failed)
    VkPhysicalDeviceMemoryProperties memProps{};

    VkSampler sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    bool makePipe(const Spv& sp, VkPipelineLayout lay, VkPipeline& out, const VkSpecializationInfo* spec);
    // Specialization constants of the colour shaders (init(), shaders/format.glsl), kept for the pipelines of a motion version
    // that is created later
    union W32 { uint32_t u; float f; };
    W32 specData[9]{};
    VkSpecializationMapEntry specEnts[9]{};
    VkSpecializationInfo specInfo{};
    const VkSpecializationInfo* spec = nullptr;
    // the motion versions' pipelines and buffers (MotionGroup); mg() = the selected one, which every record uses
    MotionGroup mo[kMotionVersions];
    MotionGroup& mg() { return mo[(int)set.motionVersion]; }
    bool motionEnsure();
    bool v35Ensure();    // Settings::v3Motion = 1: V3.5's pipelines in V3's group (created the first time it is on)
    bool guardEnsure();   // Settings::v3Guard: V2's estimate pipelines + buffers in V3's group
    bool motionSized(MotionGroup& g);
    void motionFree(MotionGroup& g);
    // motion blur (recordSynthBlur): own push-constant layout over the same set layout, 3 pipelines (per motion version, in
    // MotionGroup), and a scratch that is created by the first blur record and only ever grows (the old buffer waits in the graveyard)
    VkPipelineLayout blurLayout = VK_NULL_HANDLE;
    UiBuf blurField, blurErr;
    std::vector<UiBuf> blurGrave;
    bool growBlur(UiBuf& b, VkDeviceSize bytes);
    VkDescriptorPool pool = VK_NULL_HANDLE;
    // The V2 masks of the pair, for the synth sets (bindings 11 / 12, shaders/ui_reject.glsl). A pair without V2 masks gets `rejNone`,
    // a zeroed buffer whose empty rectangle makes the shaders reject nothing. Allocated once at init, zeroed by the first synth record.
    static constexpr VkDeviceSize kRejNoneBytes = 16;
    UiBuf rejNone;
    bool rejNoneZero = false;
    RejPair rejectMasks(VkCommandBuffer cb);
    // Two rings: synth sets (all 6 bindings) and estimate sets (binding 5, out, never written: no estimate
    // pass uses it, and no stale view of an earlier synth can linger in it).
    enum { R_SYNTH, R_EST };
    VkDescriptorSet sets[2][kSetRing]{};
    uint32_t setIdx[2]{};
    uint32_t frameSets[2]{};     // sets taken from each ring since the last beginFrame()
    // latest recordEstimate (recordSynth uses its results)
    bool haveEst = false;
    VkImageView estPrev = VK_NULL_HANDLE, estCurr = VK_NULL_HANDLE;   // the Y / only views of the estimate's pair
    // V2: the estimate ends after the search (its fitted finest field); what each V2Mode derives from it (v2Tail) is recorded once per
    // estimate, by the estimate for the mode selected then, else by the first synth of the other mode (a switch of the mode keeps the
    // estimate): both modes the re-picked and cleaned field (v2Clean), Quality also its coverage (v2Cov).
    bool v2Clean = false, v2Cov = false;
    bool mv2() const { return set.motionVersion == MotionVersion::V2; }
    bool v2NeedCov() const { return mv2() && set.v2Mode == V2Mode::Quality && !v2Cov; }   // the tail still has to splat (reset the claims first)
    void v2ResetClaims(VkCommandBuffer cb);
    void v2Tail(VkCommandBuffer cb);
    // MotionVersion::Learned: both nets' weights (binding 15; created with the version's pipelines, uploaded by its first estimate:
    // mlWeightsUp), the activation buffer's size (it grows for float32 storage after half, the old one waits in mlGrave until the
    // size is freed), and the test hooks (setDebugLearned)
    UiBuf mlWeights;
    bool mlWeightsUp = false;
    VkDeviceSize mlActBytes = 0, mlStateBytes = 0;
    std::vector<UiBuf> mlGrave;
    LearnedDebug mlDbg{};
    bool mlearned() const { return set.motionVersion == MotionVersion::Learned; }
    bool mlSized();
    void mlUpload(VkCommandBuffer cb);
    // Settings::learnedStaticGuard: the pass over a finished generated frame (shaders/ml_guard.comp), recorded with a synth set bound
    bool mlGuardOn() const { return mlearned() && set.learnedStaticGuard; }
    void mlGuard(VkCommandBuffer cb, float t, float motionK, int pf, int cnt0, int nCnt);
    void mlEstimate(VkCommandBuffer cb);
    bool v2Synth(VkCommandBuffer cb, Frame srcP, Frame srcC, Frame curr, Frame out, const float* ts, int n, bool gauss);
    // MotionVersion::V3 (gpu/V3.md). v3Fresh: its buffers were just created (zeroed by the next estimate, so nothing reads stale memory),
    // v3Dirty: the latest estimate holds injected flows (the next normal estimate zeroes them: the skeleton writes none of its own),
    // v3inj: set around recordEstimate by recordDebugV3Inject.
    struct V3Inject { VkBuffer buf; VkDeviceSize f01, f10; };
    bool v3Fresh = false, v3Dirty = false;
    uint32_t v3Ns = 1;   // the T-field slots V3B_FLOW holds (Y1: grown, with a copy of the estimate, by the first blurred record that needs more)
    const V3Inject* v3inj = nullptr;
    bool mv3() const { return set.motionVersion == MotionVersion::V3; }
    void v3Reset(VkCommandBuffer cb);
    // dumpCount > 0 (recordDebugV3Weights): the synth also writes the weight stage's records of every dumpStride-th pixel, dumpCount of them, into the
    // T-field slots after the interior ones (grown to hold them)
    bool v3Synth(VkCommandBuffer cb, Frame srcP, Frame srcC, Frame curr, Frame out, float t0, float t1, int n, bool gauss, uint32_t dumpStride = 0, uint32_t dumpCount = 0);

    // size dependent
    uint32_t W = 0, H = 0;
    int fw = 0, fh = 0;
    std::vector<Level> lv;
    int fieldSub = 0, fieldPF = 0, fieldT = 0;
    VkBuffer buf[3]{};
    VkDeviceMemory mem[3]{};
    VkDeviceSize bufSize[3]{};

    // ---- static-UI protection (framegen.h, recordObserve) ----
    VkDescriptorSetLayout uiDsl = VK_NULL_HANDLE;
    VkPipelineLayout uiLayout = VK_NULL_HANDLE;
    VkPipeline uiPipes[U_COUNT_ALL]{};
    // ---- Settings::debugView (task 136): pipelines created by dbgEnsure() the first time a view needs them (setSettings / reconfigure / init), freed by destroy() ----
    VkPipeline dbgPipes[D_COUNT]{};
    bool dbgEnsure();
    bool dbgOn() const { return set.debugView != DebugView::Off && !yuv; }   // a view is selected and this format draws one
    // descriptor sets of the UI ring one synthesis takes for the view (UiMask: the whole-screen mask's and the box mask's; UiClean: one), besides the patch's
    uint32_t dbgUiSets() const { return dbgOn() && (set.debugView == DebugView::UiMask || set.debugView == DebugView::UiClean) ? 2u : 0u; }
    // the last pass over a finished generated frame (after the UI patch): `synth` = the record's synth set (bound again with `layout` for the motion tint), pf = V1 / Learned's field offset,
    // nInt = the interior samples of this record (0: every sample is a real frame: nothing is drawn)
    void debugDraw(VkCommandBuffer cb, VkDescriptorSet synth, Frame curr, Frame out, int pf, int nInt);
    VkDescriptorPool uiPool = VK_NULL_HANDLE;
    VkDescriptorSet uiSets[kUiSets]{};
    uint32_t uiSetIdx = 0, uiFrameSets = 0;
    UiEntry uiE[kUiFramesMax];
    UiBuf uiBoxState, uiCellState;   // per-pixel box state / per-cell state (words, see ui_common.glsl)
    std::vector<UiBuf> uiGrave;      // grown-out buffers: freed with the rest (no GPU wait ever needed to grow)
    std::vector<VkImage> uiPendingInit;   // new images still in UNDEFINED layout
    bool uiImgs = false, uiWsRes = false;
    bool uiExplicit = false;         // recordObserve() has been called: estimates do not observe implicitly
    bool uiBoxHist = false, uiCellHist = false;   // the state buffers describe the previous real frame
    UiBox uiPrevBox{};
    uint64_t uiSeq = 0;
    int uiLast = -1;                 // entry of the newest observe
    int cw = 0, ch = 0, tilesX = 0, tilesY = 0;
    // the pair of the latest recordEstimate, when it ran on clean copies
    bool estUi = false;
    bool estClean = false;           // ... and the pair was read from the clean copies (every mode but CrosshairV2)
    int estEnt[2]{-1, -1};
    uint64_t estSeq[2]{};
    bool profPending = false;        // an observe opened the current profile and no estimate has continued it yet

    bool v2() const { return set.uiProtection == UiProtection::CrosshairV2; }
    bool v2Fill() const { return v2() && set.uiEstimateFill; }   // CrosshairV2 with the luma fill of the estimate on
    bool uiActive() const { return W && set.uiProtection != UiProtection::Off && uiImgs && (set.uiProtection != UiProtection::WholeScreen || uiWsRes); }
    uint32_t uiObserveSets() const { return set.uiProtection == UiProtection::WholeScreen ? 2 : 1; }
    UiBox computeBox() const;
    UiBox computeWork(const UiBox& box) const;
    uint32_t localType(uint32_t bits) const;
    bool allocBuf(UiBuf& b, VkDeviceSize bytes, VkBufferUsageFlags extra = 0);
    void freeBuf(UiBuf& b);
    bool growBuf(UiBuf& b, VkDeviceSize bytes);
    bool allocImage(VkFormat f, uint32_t w, uint32_t h, VkImage& img, VkDeviceMemory& mem, VkImageView& view);
    bool allocEntryImage(UiEntry& e);
    bool uiEnsure();
    void uiFreeEntry(UiEntry& e);
    void uiDestroy();
    void uiAdopted(const Settings& old);
    bool uiChanged(const Settings& old, const UiBox& oldBox, const UiBox& oldWork) const {   // the UI mode or its box changed (set = the new settings)
        if (set.uiProtection != old.uiProtection) return true;
        if (set.uiProtection == UiProtection::Off) return false;
        const UiBox b = computeBox();
        return !(b == oldBox) || !(computeWork(b) == oldWork);
    }
    void uiForget(VkImageView v) {   // a refused observe / a track of v: the entry keyed by v no longer holds v's content (task 112)
        const int e = uiFind(v);
        if (e >= 0) { uiE[e].key = VK_NULL_HANDLE; uiE[e].seq = 0; if (uiLast == e) uiLast = -1; }
    }
    int uiFind(VkImageView v) const;
    int uiPick(VkImageView v) const;
    VkDescriptorSet takeUiSet(Frame src, Frame img, const UiBuf& s, const UiBuf& m);
    VkDescriptorSet takeUiSetV2(Frame src, const UiBuf& s, const UiBuf& m);
    bool uiObserveRecord(VkCommandBuffer cb, Frame view);
    bool uiObserveRecordV2(VkCommandBuffer cb, Frame view);
    bool uiTrackRecord(VkCommandBuffer cb, Frame view);
    bool uiTrackRecordV2(VkCommandBuffer cb, Frame view);
    void uiRecordLumaFill(VkCommandBuffer cb, Frame frame, const UiEntry& e, int lumaOff);
    void uiFlushPendingInit(VkCommandBuffer cb);
    void uiRecordPatch(VkCommandBuffer cb, Frame curr, Frame out, const UiEntry& e);

    // profiling
    bool profiling = false;
    VkQueryPool qpool = VK_NULL_HANDLE;
    static constexpr uint32_t kMaxQ = 64;
    std::vector<const char*> qnames;
    uint32_t qcount = 0;

    VkDescriptorSet takeSet(int ring, Frame prev, Frame curr, Frame out, bool blur = false, RejPair rej = {}, bool guardView = false);   // guardView (V3 + Settings::v3Guard): bindings 13 / 14 are V2's estimate buffers (V2's passes run in it)
    bool framesOk(const Frame& a) const { return a.view && (!yuv || a.chroma); }
    bool createSizeDependent();
    bool rebuild(uint32_t w, uint32_t h, VkFormat fmt, bool force);
    void destroySizeDependent();
    bool makeBuffer(int i, VkDeviceSize size);
};


bool Interpolator::Impl::makeBuffer(int i, VkDeviceSize size) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (i >= 1) bci.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;   // fields / counters + mismatch: recordDebugCounters() / recordDebugMaps() copy them out
    if (CreateBuffer(info.device, &bci, nullptr, &buf[i]) != VK_SUCCESS) { buf[i] = VK_NULL_HANDLE; return false; }
    VkMemoryRequirements req;
    GetBufferMemoryRequirements(info.device, buf[i], &req);
    uint32_t type = UINT32_MAX;
    for (uint32_t t = 0; t < memProps.memoryTypeCount && type == UINT32_MAX; t++)
        if ((req.memoryTypeBits & (1u << t)) && (memProps.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            type = t;
    if (type == UINT32_MAX) return false;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    if (AllocateMemory(info.device, &mai, nullptr, &mem[i]) != VK_SUCCESS) { mem[i] = VK_NULL_HANDLE; return false; }
    if (BindBufferMemory(info.device, buf[i], mem[i], 0) != VK_SUCCESS) return false;
    bufSize[i] = size;
    return true;
}

bool Interpolator::Impl::createSizeDependent() {
    // interp.flow(): luma at 1/FLOW_SCALE (cv2 rounds the size), pyramid down to Settings::pyramidMinSide (sizeLayout).
    SizeLayout L = sizeLayout(W, H, set.flowScale, set.pyramidMinSide);
    fw = L.fw; fh = L.fh;
    lv = L.lv;
    fieldSub = L.fieldSub;
    fieldPF = L.fieldPF;
    fieldT = L.fieldT;
    int uCount = kEOff + (int)(W + 3) / 4 * (int)H;   // counters/medians, then the packed mismatch
    if (!makeBuffer(0, (VkDeviceSize)L.lumaWords * 4)) return false;
    if (!makeBuffer(1, (VkDeviceSize)L.fieldEntries * 8)) return false;
    if (!makeBuffer(2, (VkDeviceSize)uCount * 4)) return false;
    // static-UI protection: the detection grids, and the resources of the active mode
    cw = ((int)W + kUiCell - 1) / kUiCell; ch = ((int)H + kUiCell - 1) / kUiCell;
    tilesX = ((int)W + kUiTile - 1) / kUiTile; tilesY = ((int)H + kUiTile - 1) / kUiTile;
    if (!uiEnsure()) return false;
    // the motion versions' own buffers (of every version created so far, so a switch back needs no allocation)
    for (MotionGroup& g : mo)
        if (g.live && !motionSized(g)) return false;
    return true;
}

void Interpolator::Impl::destroySizeDependent() {
    for (int i = 0; i < 3; i++) {
        if (buf[i]) DestroyBuffer(info.device, buf[i], nullptr);
        if (mem[i]) FreeMemory(info.device, mem[i], nullptr);
        buf[i] = VK_NULL_HANDLE; mem[i] = VK_NULL_HANDLE;
    }
    uiDestroy();
    for (MotionGroup& g : mo) {
        for (UiBuf& b : g.bufs) freeBuf(b);
        g.bufs.clear();
    }
    for (UiBuf& b : mlGrave) freeBuf(b);
    mlGrave.clear();
    mlActBytes = 0;
    freeBuf(blurField);
    freeBuf(blurErr);
    for (UiBuf& b : blurGrave) freeBuf(b);
    blurGrave.clear();
}

// ---------------------------------------------------------------------------------------------------------
// Static-UI protection: resources
// ---------------------------------------------------------------------------------------------------------

// The crosshair box: explicit pixel sizes win over the height fraction; clamped into the frame.
void Interpolator::boxSize(const Settings& set, uint32_t w, uint32_t h, uint32_t* boxW, uint32_t* boxH) {
    const int fw_ = (int)w, fh_ = (int)h;
    int side;
    if (set.uiBoxHeightFrac > 0.0f)
        side = std::min(std::max((int)std::lround((double)h * set.uiBoxHeightFrac), 8), std::min(fw_, fh_));
    else   // legacy: area fraction, old 32 px minimum
        side = std::min(std::max((int)std::lround(std::sqrt((double)set.uiCrosshairArea * fw_ * fh_)), 32), std::min(fw_, fh_));
    *boxW = set.uiBoxWidth ? (uint32_t)std::min(std::max((int)set.uiBoxWidth, 4), fw_) : (uint32_t)side;
    *boxH = set.uiBoxHeight ? (uint32_t)std::min(std::max((int)set.uiBoxHeight, 4), fh_) : (uint32_t)side;
}

// The box rectangle in the frame: boxSize() centred on the frame centre plus the offset, clamped into the frame.
void Interpolator::boxRect(const Settings& set, uint32_t w, uint32_t h, int32_t r[4]) {
    uint32_t bw, bh;
    boxSize(set, w, h, &bw, &bh);
    const int fw_ = (int)w, fh_ = (int)h;
    r[2] = (int32_t)bw; r[3] = (int32_t)bh;
    r[0] = std::min(std::max((fw_ - r[2]) / 2 + set.uiBoxOffsetX, 0), fw_ - r[2]);
    r[1] = std::min(std::max((fh_ - r[3]) / 2 + set.uiBoxOffsetY, 0), fh_ - r[3]);
}

// How far past the box a mode protects (the "work rectangle", workRect()): the V1 mask dilates 2 px, V2's grow is 1 px. Both modes look
// for UI only inside the box; the margin lets the rim of what they found, and V1's inpaint source, extend past its edge. WholeScreen
// keeps its box exactly as it was (its detection and inpaint of the rest of the frame are not box-bound).
uint32_t Interpolator::boxMargin(const Settings& set) {
    return set.uiProtection == UiProtection::Crosshair ? 2u : set.uiProtection == UiProtection::CrosshairV2 ? 1u : 0u;
}

void Interpolator::workRect(const Settings& set, uint32_t w, uint32_t h, int32_t r[4]) {
    boxRect(set, w, h, r);
    const int m = (int)boxMargin(set);
    const int x0 = std::max(r[0] - m, 0), y0 = std::max(r[1] - m, 0);
    const int x1 = std::min(r[0] + r[2] + m, (int)w), y1 = std::min(r[1] + r[3] + m, (int)h);
    r[0] = x0; r[1] = y0; r[2] = x1 - x0; r[3] = y1 - y0;
}

UiBox Interpolator::Impl::computeBox() const {
    int32_t r[4];
    Interpolator::boxRect(set, W, H, r);
    return UiBox{r[0], r[1], r[2], r[3]};
}

UiBox Interpolator::Impl::computeWork(const UiBox& b) const {
    const int m = (int)Interpolator::boxMargin(set);
    const int x0 = std::max(b.x - m, 0), y0 = std::max(b.y - m, 0);
    const int x1 = std::min(b.x + b.w + m, (int)W), y1 = std::min(b.y + b.h + m, (int)H);
    return UiBox{x0, y0, x1 - x0, y1 - y0};
}

uint32_t Interpolator::Impl::localType(uint32_t bits) const {
    for (uint32_t t = 0; t < memProps.memoryTypeCount; t++)
        if ((bits & (1u << t)) && (memProps.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) return t;
    return UINT32_MAX;
}

bool Interpolator::Impl::allocBuf(UiBuf& b, VkDeviceSize bytes, VkBufferUsageFlags extra) {
    b = UiBuf{};
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | extra;
    if (CreateBuffer(info.device, &bci, nullptr, &b.b) != VK_SUCCESS) { b.b = VK_NULL_HANDLE; return false; }
    VkMemoryRequirements req;
    GetBufferMemoryRequirements(info.device, b.b, &req);
    uint32_t type = localType(req.memoryTypeBits);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    if (type == UINT32_MAX || AllocateMemory(info.device, &mai, nullptr, &b.m) != VK_SUCCESS ||
        BindBufferMemory(info.device, b.b, b.m, 0) != VK_SUCCESS) {
        freeBuf(b);
        return false;
    }
    b.size = bytes;
    return true;
}

void Interpolator::Impl::freeBuf(UiBuf& b) {
    if (b.b) DestroyBuffer(info.device, b.b, nullptr);
    if (b.m) FreeMemory(info.device, b.m, nullptr);
    b = UiBuf{};
}

// Makes b hold at least `bytes`. A buffer that is too small is NOT freed (earlier records may still use it): it goes
// to the graveyard, which is emptied by resize/reconfigure/destroy (their retire rule), so growing needs no GPU wait.
// 25 % headroom, so a size slider does not allocate per step.
bool Interpolator::Impl::growBuf(UiBuf& b, VkDeviceSize bytes) {
    if (b.b && b.size >= bytes) return true;
    UiBuf nb;
    if (!allocBuf(nb, bytes + bytes / 4)) return false;
    if (b.b) uiGrave.push_back(b);
    b = nb;
    return true;
}

// growBuf for the motion-blur scratch: same rules, own graveyard (emptied by resize/reconfigure/destroy).
bool Interpolator::Impl::growBlur(UiBuf& b, VkDeviceSize bytes) {
    if (b.b && b.size >= bytes) return true;
    UiBuf nb;
    if (!allocBuf(nb, bytes)) return false;
    if (b.b) blurGrave.push_back(b);
    b = nb;
    return true;
}

bool Interpolator::Impl::allocImage(VkFormat f, uint32_t w, uint32_t h, VkImage& img, VkDeviceMemory& mem, VkImageView& view) {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = f;   // in logical channel order, in the source's own format: only the library reads and writes it
    ici.extent = {w, h, 1};
    ici.mipLevels = ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (CreateImage(info.device, &ici, nullptr, &img) != VK_SUCCESS) { img = VK_NULL_HANDLE; return false; }
    VkMemoryRequirements req;
    GetImageMemoryRequirements(info.device, img, &req);
    uint32_t type = localType(req.memoryTypeBits);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    if (type == UINT32_MAX || AllocateMemory(info.device, &mai, nullptr, &mem) != VK_SUCCESS) { mem = VK_NULL_HANDLE; return false; }
    if (BindImageMemory(info.device, img, mem, 0) != VK_SUCCESS) return false;
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = img;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = f;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (CreateImageView(info.device, &vci, nullptr, &view) != VK_SUCCESS) { view = VK_NULL_HANDLE; return false; }
    uiPendingInit.push_back(img);
    return true;
}

// One clean copy: the frame in its own format (Rgba8: always R8G8B8A8, as before), for YUV a second image for CbCr.
bool Interpolator::Impl::allocEntryImage(UiEntry& e) {
    if (!allocImage(cleanFmt, W, H, e.img, e.imgMem, e.view)) return false;
    return !yuv || allocImage(cleanFmtC, (W + 1) / 2, (H + 1) / 2, e.imgC, e.imgMemC, e.viewC);
}

// Creates what the active mode needs and does not have yet; never frees. Called by setSettings/reconfigure (the new
// settings already in `set`) and by (re)creation of the size-dependent resources. The per-box buffers are created
// lazily by the observe that needs them. False = could not allocate (the caller refuses the settings).
// Settings::uiHistory = the number of active entries: a raise first re-activates entries that kept their memory, then
// allocates new ones; a lowering deactivates the least recently observed (their frames are forgotten, the memory is
// kept until releaseUnusedUi()).
bool Interpolator::Impl::uiEnsure() {
    if (!W || set.uiProtection == UiProtection::Off) return true;
    const int want = (int)std::min(std::max(set.uiHistory, kUiFrames), kUiFramesMax);
    const bool needImg = !v2();   // CrosshairV2 reads the host's frames directly: no clean copies
    const bool first = !uiImgs;
    int nAct = 0;
    for (const UiEntry& e : uiE) nAct += e.active;
    std::vector<int> added;   // activated by this call (rolled back if a later allocation fails)
    auto rollback = [&]() { for (int i : added) { uiE[i].active = false; uiE[i].key = VK_NULL_HANDLE; uiE[i].seq = 0; } };
    for (int i = 0; nAct < want && i < (int)kUiFramesMax; i++) {
        UiEntry& e = uiE[i];
        if (e.active) continue;
        if (needImg && !e.img && !allocEntryImage(e)) {
            if (first) uiDestroy(); else { uiFreeEntry(e); rollback(); }
            return false;
        }
        e.active = true; e.key = VK_NULL_HANDLE; e.seq = 0;
        added.push_back(i);
        nAct++;
    }
    // entries that were activated under V2 have no clean copy: a switch to another mode needs them now
    if (needImg)
        for (UiEntry& e : uiE)
            if (e.active && !e.img && !allocEntryImage(e)) {
                if (first) uiDestroy(); else { uiFreeEntry(e); rollback(); }
                return false;
            }
    if (set.uiProtection == UiProtection::WholeScreen) {
        const VkDeviceSize wsWords = 4 + (VkDeviceSize)tilesX * tilesY + (VkDeviceSize)((W + 3) / 4) * H;
        const bool firstWs = !uiWsRes;
        bool ok = uiCellState.b || allocBuf(uiCellState, (VkDeviceSize)cw * ch * 4);
        for (UiEntry& e : uiE) ok = ok && (!e.active || e.ws.b || allocBuf(e.ws, wsWords * 4));
        if (!ok) {
            if (firstWs) {
                freeBuf(uiCellState);
                for (UiEntry& e : uiE) freeBuf(e.ws);
            } else {
                for (int i : added) freeBuf(uiE[i].ws);
            }
            if (first) uiDestroy(); else rollback();
            return false;
        }
        uiWsRes = true;
    }
    // lower: forget the least recently observed frames beyond `want`
    while (nAct > want) {
        int lo = -1;
        for (int i = 0; i < (int)kUiFramesMax; i++)
            if (uiE[i].active && (lo < 0 || uiE[i].seq < uiE[lo].seq)) lo = i;
        uiE[lo].active = false; uiE[lo].key = VK_NULL_HANDLE; uiE[lo].seq = 0;
        nAct--;
    }
    uiImgs = true;
    return true;
}

// Frees one entry's resources (retire rule of resize()/destroy()).
void Interpolator::Impl::uiFreeEntry(UiEntry& e) {
    VkDevice d = info.device;
    for (int i = 0; i < 2; i++) {   // luma / only image, then the CbCr plane
        VkImageView& v = i ? e.viewC : e.view;
        VkImage& im = i ? e.imgC : e.img;
        VkDeviceMemory& mm = i ? e.imgMemC : e.imgMem;
        if (v) DestroyImageView(d, v, nullptr);
        if (im) {
            uiPendingInit.erase(std::remove(uiPendingInit.begin(), uiPendingInit.end(), im), uiPendingInit.end());
            DestroyImage(d, im, nullptr);
        }
        if (mm) FreeMemory(d, mm, nullptr);
    }
    freeBuf(e.box);
    freeBuf(e.ws);
    e = UiEntry{};
}

// Frees every UI resource. Retire rule of resize()/destroy(): all earlier records must have retired.
void Interpolator::Impl::uiDestroy() {
    for (UiEntry& e : uiE) uiFreeEntry(e);
    freeBuf(uiBoxState);
    freeBuf(uiCellState);
    for (UiBuf& b : uiGrave) freeBuf(b);
    uiGrave.clear();
    uiPendingInit.clear();
    uiImgs = uiWsRes = false;
    uiBoxHist = uiCellHist = false;
    uiLast = -1;
    estUi = false;
}

// New settings were adopted (old = the previous ones): switching off forgets the observed frames (their views may
// be reused by the host without notice), and a gap in observation restarts the stillness counters.
void Interpolator::Impl::uiAdopted(const Settings& old) {
    // (seq too: a synth of the pair estimated before the change then fails its sequence check, task 112)
    if (set.uiProtection == UiProtection::Off) {
        for (UiEntry& e : uiE) { e.key = VK_NULL_HANDLE; e.seq = 0; }
        uiLast = -1;
    }
    if (set.uiProtection != UiProtection::WholeScreen) uiCellHist = false;
    if (set.uiProtection == UiProtection::Off || old.uiProtection == UiProtection::Off) uiBoxHist = false;
    // V2 keeps another state layout and another mask format than the other modes: nothing observed under one is valid under the other
    if (set.uiProtection != old.uiProtection && (set.uiProtection == UiProtection::CrosshairV2 || old.uiProtection == UiProtection::CrosshairV2)) {
        for (UiEntry& e : uiE) { e.key = VK_NULL_HANDLE; e.seq = 0; }
        uiLast = -1;
        uiBoxHist = false;
    }
}

int Interpolator::Impl::uiFind(VkImageView v) const {
    for (int i = 0; i < (int)kUiFramesMax; i++)
        if (v && uiE[i].active && uiE[i].key == v) return i;
    return -1;
}

// The entry to observe `v` into: the one already keyed by it, else the least recently observed.
int Interpolator::Impl::uiPick(VkImageView v) const {
    int f = uiFind(v);
    if (f >= 0) return f;
    int best = -1;
    for (int i = 0; i < (int)kUiFramesMax; i++)
        if (uiE[i].active && (best < 0 || uiE[i].seq < uiE[best].seq)) best = i;
    return best;
}

VkDescriptorSet Interpolator::Impl::takeUiSet(Frame src, Frame img, const UiBuf& s, const UiBuf& m) {
    VkDescriptorSet set_ = uiSets[uiSetIdx];
    uiSetIdx = (uiSetIdx + 1) % kUiSets;
    uiFrameSets++;
    VkDescriptorImageInfo ii[4] = {{VK_NULL_HANDLE, src.view, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, img.view, VK_IMAGE_LAYOUT_GENERAL},
                                   {VK_NULL_HANDLE, src.chroma, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, img.chroma, VK_IMAGE_LAYOUT_GENERAL}};
    VkDescriptorBufferInfo bi[2] = {{s.b, 0, VK_WHOLE_SIZE}, {m.b, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wr[6]{};
    const uint32_t n = yuv ? 6 : 4;
    for (uint32_t i = 0; i < n; i++) {
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = set_;
        wr[i].dstBinding = i;
        wr[i].descriptorCount = 1;
    }
    wr[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr[0].pImageInfo = &ii[0];
    wr[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    wr[1].pImageInfo = &ii[1];
    wr[2].descriptorType = wr[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr[2].pBufferInfo = &bi[0];
    wr[3].pBufferInfo = &bi[1];
    if (yuv) {   // the CbCr planes: sampled source (4), storage destination (5)
        wr[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        wr[4].pImageInfo = &ii[2];
        wr[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        wr[5].pImageInfo = &ii[3];
    }
    UpdateDescriptorSets(info.device, n, wr, 0, nullptr);
    return set_;
}

// The set of a CrosshairV2 pass: the sampled frame (+ its CbCr plane for YUV), the state buffer and the mask buffer. No storage
// image: the V2 passes never write one and the bindings they do not use stay as they are.
VkDescriptorSet Interpolator::Impl::takeUiSetV2(Frame src, const UiBuf& s, const UiBuf& m) {
    VkDescriptorSet set_ = uiSets[uiSetIdx];
    uiSetIdx = (uiSetIdx + 1) % kUiSets;
    uiFrameSets++;
    VkDescriptorImageInfo ii[2] = {{VK_NULL_HANDLE, src.view, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, src.chroma, VK_IMAGE_LAYOUT_GENERAL}};
    VkDescriptorBufferInfo bi[2] = {{s.b, 0, VK_WHOLE_SIZE}, {m.b, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wr[4]{};
    uint32_t n = 0;
    auto add = [&](uint32_t binding, VkDescriptorType t) -> VkWriteDescriptorSet& {
        VkWriteDescriptorSet& w = wr[n++];
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = set_;
        w.dstBinding = binding;
        w.descriptorCount = 1;
        w.descriptorType = t;
        return w;
    };
    add(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo = &ii[0];
    add(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).pBufferInfo = &bi[0];
    add(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).pBufferInfo = &bi[1];
    if (yuv) add(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo = &ii[1];
    UpdateDescriptorSets(info.device, n, wr, 0, nullptr);
    return set_;
}

bool Interpolator::Impl::makePipe(const Spv& sp, VkPipelineLayout lay, VkPipeline& out, const VkSpecializationInfo* spec) {
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = sp.size;
    smci.pCode = sp.code;
    VkShaderModule sm;
    if (CreateShaderModule(info.device, &smci, nullptr, &sm) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", spec};
    cpci.layout = lay;
    VkResult r = CreateComputePipelines(info.device, info.pipelineCache, 1, &cpci, nullptr, &out);
    DestroyShaderModule(info.device, sm, nullptr);
    if (r != VK_SUCCESS) { out = VK_NULL_HANDLE; return false; }
    return true;
}

namespace {
// Device support of one format for what the library does with it: sampled (linearly filtered) and storage, optimal tiling.
bool formatUsable(PFN_vkGetPhysicalDeviceFormatProperties fp, VkPhysicalDevice phys, VkFormat f) {
    VkFormatProperties p{};
    fp(phys, f, &p);
    const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
                                      VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
    return (p.optimalTilingFeatures & need) == need;
}

// Device support of MotionVersion::V2's subgroup use (null = supported): v2_search.comp (arithmetic), v2_repick.comp (clusters of
// 16), v2_enclose.comp (clusters of 8, shuffles), v2_morph.glsl (vote, arithmetic, shuffles; its 4x4 groups need 32 lanes, else a
// per-pixel choice), all in compute shaders.
const char* v2SubgroupError(PFN_vkGetPhysicalDeviceProperties2 fp, VkPhysicalDevice phys) {
    VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &sg;
    fp(phys, &p2);
    const VkSubgroupFeatureFlags need = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT |
                                        VK_SUBGROUP_FEATURE_ARITHMETIC_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
                                        VK_SUBGROUP_FEATURE_CLUSTERED_BIT;
    if (!(sg.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) || (sg.supportedOperations & need) != need)
        return "MotionVersion::V2 needs compute-stage subgroups with the basic, vote, arithmetic, shuffle and clustered operations; use MotionVersion::V1";
    if (sg.subgroupSize < 16) return "MotionVersion::V2 needs subgroups of at least 16 lanes; use MotionVersion::V1";
    return nullptr;
}
}  // namespace

bool Interpolator::init(const InitInfo& ii) {
    destroy();
    err_ = nullptr;
    if (!ii.getMemProps || !ii.gdpa || !ii.device || !validate(ii.settings)) { err_ = "missing getMemProps / gdpa / device, or invalid settings"; return false; }
    Format fmt;
    if (!formatOf(ii.viewFormat, &fmt)) { err_ = "unsupported viewFormat (see framegen::Format)"; return false; }
    ColourSpv cs{};
    if (!colourSpv(fmt, cs)) { err_ = "the shader variants of this format are not built into the library (meson option 'formats')"; return false; }
    if (const char* e = settingsError(ii.settings, fmt)) { err_ = e; return false; }
    const char* v2Why = ii.getProps2 ? v2SubgroupError(ii.getProps2, ii.phys) : nullptr;
    DevLimits lim{};
    if (ii.getProps2) {
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        ii.getProps2(ii.phys, &p2);
        const VkPhysicalDeviceLimits& l = p2.properties.limits;
        lim.img2D = l.maxImageDimension2D;
        lim.sbRange = l.maxStorageBufferRange;
        for (int i = 0; i < 3; i++) lim.wg[i] = l.maxComputeWorkGroupCount[i];
    }
    if (const char* e = sizeLimitError(ii.width, ii.height, ii.settings, lim)) { err_ = e; return false; }
    if (ii.settings.motionVersion == MotionVersion::V2 && v2Why) { err_ = v2Why; return false; }
    if (ii.settings.motionVersion == MotionVersion::V3 && ii.settings.v3Guard != V3Guard::Off && v2Why) { err_ = v2Why; return false; }
    if (ii.settings.motionVersion == MotionVersion::V2 && !cs.v2Morph.code) { err_ = kV2NotBuilt; return false; }
    if (ii.settings.motionVersion == MotionVersion::Learned && !cs.mlDup.code) { err_ = kMlNotBuilt; return false; }
    if (ii.settings.motionVersion == MotionVersion::V3 && (!cs.v3Synth.code || !cs.v3Planes.code)) { err_ = kV3NotBuilt; return false; }
    if (fmt == Format::RgbaF16 && ii.transfer == Transfer::Linear && !(ii.linearWhiteNits > 0.0f && ii.linearWhiteNits <= 10000.0f)) {
        err_ = "linearWhiteNits must be in (0, 10000]";
        return false;
    }
    if (fmt != Format::Rgba8 && ii.getFormatProps) {   // never for Rgba8: existing hosts behave exactly as before
        if (!formatUsable(ii.getFormatProps, ii.phys, ii.viewFormat)) { err_ = "the device cannot sample (linear) and store the view format"; return false; }
        if (!formatUsable(ii.getFormatProps, ii.phys, cleanFormat(fmt))) { err_ = "the device cannot sample and store the format of the clean copies"; return false; }
        if (isYuv(fmt) && !formatUsable(ii.getFormatProps, ii.phys, cleanFormatC(fmt))) { err_ = "the device cannot sample and store the chroma plane format"; return false; }
    }
    m = new Impl();
    m->info = ii;
    m->set = ii.settings;
    m->fmt = fmt;
    m->v2Why = v2Why;
    m->lim = lim;
    m->yuv = isYuv(fmt);
    m->cs = cs;
    m->cleanFmt = cleanFormat(fmt);
    m->cleanFmtC = m->yuv ? cleanFormatC(fmt) : VK_FORMAT_UNDEFINED;
    VkDevice d = ii.device;
#define X(n) m->n = (PFN_vk##n)ii.gdpa(d, "vk" #n); if (!m->n) { destroy(); err_ = "a device function could not be loaded through gdpa"; return false; }
    FG_FUNCS(X)
#undef X
    ii.getMemProps(ii.phys, &m->memProps);
    const char* fail = "a Vulkan object could not be created";   // err_ of every `destroy(); return false;` below

    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod = 0.0f;
    if (m->CreateSampler(d, &sci, nullptr, &m->sampler) != VK_SUCCESS) { destroy(); err_ = fail; return false; }

    // bindings 0-2 buffers, 3-4 prev/curr, 5 out; 6-7: motion-blur scratch (only the blur passes use them);
    // YUV: 8-9 the CbCr planes of prev/curr, 10 the CbCr plane of out; 11-12 (synth sets): the V2 masks of prev / curr (ui_reject.glsl);
    // 13: MotionVersion::V2's cleaned block field, 14: V2 Quality's claims and coverage (written only to V2's sets,
    // shaders/v2_motion.glsl, v2_quality.glsl); MotionVersion::Learned's sets: 13 the net's state, 14 its activations, 15 its weights
    // (shaders/ml_common.glsl); 16: V3's sets with Settings::v3Guard on only, V2's estimate buffer (V2B_FIELD layout) that v3_guard reads
    uint32_t nb = m->yuv ? 11 : 8;
    VkDescriptorSetLayoutBinding b[17]{};
    VkDescriptorType types[11] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
    for (uint32_t i = 0; i < nb; i++) {
        b[i].binding = i;
        b[i].descriptorType = types[i];
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        if (i == 3 || i == 4 || i == 8 || i == 9) b[i].pImmutableSamplers = &m->sampler;
    }
    for (uint32_t k = 11; k <= 16; k++, nb++) {
        b[nb].binding = k;
        b[nb].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[nb].descriptorCount = 1;
        b[nb].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dci.bindingCount = nb;
    dci.pBindings = b;
    if (m->CreateDescriptorSetLayout(d, &dci, nullptr, &m->dsl) != VK_SUCCESS) { destroy(); err_ = fail; return false; }

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &m->dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    if (m->CreatePipelineLayout(d, &plci, nullptr, &m->layout) != VK_SUCCESS) { destroy(); err_ = fail; return false; }

    // Specialization constants of the colour shaders (shaders/format.glsl): the transfer of float input, or the
    // sample scale / range / matrix of YUV. Plain Rgba8 and Rgb10 have none.
    Impl::W32* sd = m->specData;
    VkSpecializationMapEntry* ents = m->specEnts;
    VkSpecializationInfo& specInfo = m->specInfo;
    const VkSpecializationInfo*& spec = m->spec;
    if (fmt == Format::RgbaF16) {
        sd[0].u = ii.transfer == Transfer::Linear ? 1u : 0u;
        sd[1].f = ii.linearWhiteNits;
        specInfo.mapEntryCount = 2;
    } else if (m->yuv) {
        double kr = 0.2126, kb = 0.0722;   // BT.709
        if (ii.yuvMatrix == YuvMatrix::Bt601) { kr = 0.299; kb = 0.114; }
        if (ii.yuvMatrix == YuvMatrix::Bt2020) { kr = 0.2627; kb = 0.0593; }
        const double kg = 1.0 - kr - kb;
        const double sY = ii.yuvFullRange ? 1.0 : 255.0 / 219.0, sC = ii.yuvFullRange ? 1.0 : 255.0 / 224.0;
        const bool p010 = fmt == Format::P010;
        sd[0].f = p010 ? 65535.0f / 256.0f : 255.0f;   // YU_SCALE: R16 sample -> level (code10 / 4), or R8 sample -> level
        sd[1].f = p010 ? 0.25f : 1.0f;                  // YU_QL
        sd[2].f = p010 ? 1023.0f : 255.0f;              // YU_CODES
        sd[3].f = ii.yuvFullRange ? 0.0f : 16.0f;       // YU_BLACK
        sd[4].f = (float)sY;                            // YU_YS
        sd[5].f = (float)(sC * 2.0 * (1.0 - kr));                // YU_RCR
        sd[6].f = (float)(sC * 2.0 * (1.0 - kb));                // YU_BCB
        sd[7].f = (float)(-sC * 2.0 * (1.0 - kb) * kb / kg);     // YU_GCB
        sd[8].f = (float)(-sC * 2.0 * (1.0 - kr) * kr / kg);     // YU_GCR
        specInfo.mapEntryCount = 9;
    }
    if (specInfo.mapEntryCount) {
        for (uint32_t i = 0; i < specInfo.mapEntryCount; i++) ents[i] = {i, i * 4, 4};
        specInfo.pMapEntries = ents;
        specInfo.dataSize = specInfo.mapEntryCount * 4;
        specInfo.pData = sd;
        spec = &specInfo;
    }

    // motion-blur passes: the same set layout, their own (larger) push-constant block
    {
        VkPushConstantRange bpcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BlurPC)};
        VkPipelineLayoutCreateInfo bplci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        bplci.setLayoutCount = 1;
        bplci.pSetLayouts = &m->dsl;
        bplci.pushConstantRangeCount = 1;
        bplci.pPushConstantRanges = &bpcr;
        if (m->CreatePipelineLayout(d, &bplci, nullptr, &m->blurLayout) != VK_SUCCESS) { destroy(); err_ = fail; return false; }
    }
    // the pipelines of the selected motion version (another version's are created when it is first selected)
    if (!m->motionEnsure()) { destroy(); err_ = fail; return false; }

    // one pool for the sets of both rings, sized by what the layout above holds (every set is allocated with all of it)
    uint32_t nBuf = 0, nSamp = 0, nImg = 0;
    for (uint32_t i = 0; i < nb; i++) {
        nBuf += b[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        nSamp += b[i].descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        nImg += b[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    }
    VkDescriptorPoolSize ps[3] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nBuf * 2 * kSetRing},
                                  {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, nSamp * 2 * kSetRing},
                                  {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, nImg * 2 * kSetRing}};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 2 * kSetRing;
    dpci.poolSizeCount = 3;
    dpci.pPoolSizes = ps;
    if (m->CreateDescriptorPool(d, &dpci, nullptr, &m->pool) != VK_SUCCESS) { destroy(); err_ = fail; return false; }
    VkDescriptorSetLayout layouts[2 * kSetRing];
    for (auto& l : layouts) l = m->dsl;
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = m->pool;
    dsai.descriptorSetCount = 2 * kSetRing;
    dsai.pSetLayouts = layouts;
    if (m->AllocateDescriptorSets(d, &dsai, &m->sets[0][0]) != VK_SUCCESS) { destroy(); err_ = fail; return false; }
    if (!m->allocBuf(m->rejNone, Impl::kRejNoneBytes)) { destroy(); err_ = fail; return false; }

    // static-UI protection: own set layout (sampled frame, storage image, 2 storage buffers; YUV: + the CbCr planes),
    // push constants, pipelines and a ring of kUiSets sets. Created always (small); resources only when the mode needs them.
    {
        const uint32_t nu = m->yuv ? 6 : 4;
        VkDescriptorSetLayoutBinding ub[6]{};
        VkDescriptorType ut[6] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
        for (uint32_t i = 0; i < nu; i++) {
            ub[i].binding = i;
            ub[i].descriptorType = ut[i];
            ub[i].descriptorCount = 1;
            ub[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        ub[0].pImmutableSamplers = &m->sampler;
        ub[4].pImmutableSamplers = &m->sampler;
        VkDescriptorSetLayoutCreateInfo udci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        udci.bindingCount = nu;
        udci.pBindings = ub;
        if (m->CreateDescriptorSetLayout(d, &udci, nullptr, &m->uiDsl) != VK_SUCCESS) { destroy(); err_ = fail; return false; }
        VkPushConstantRange upcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(UiPC)};
        VkPipelineLayoutCreateInfo uplci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        uplci.setLayoutCount = 1;
        uplci.pSetLayouts = &m->uiDsl;
        uplci.pushConstantRangeCount = 1;
        uplci.pPushConstantRanges = &upcr;
        if (m->CreatePipelineLayout(d, &uplci, nullptr, &m->uiLayout) != VK_SUCCESS) { destroy(); err_ = fail; return false; }
        struct { Spv sp; const VkSpecializationInfo* sc; } uspv[U_COUNT_ALL] = {
            {cs.uiDetect, spec}, {cs.uiBoxUpdate, spec}, {SPV(ui_box_mask), nullptr}, {SPV(ui_mask), nullptr},
            {cs.uiInpaint, spec}, {cs.uiPatch, spec}, {cs.uiV2Update, spec}, {SPV(ui_v2_mask), nullptr}, {cs.uiV2Fill, spec},
            {cs.uiInpaintC, spec}, {cs.uiPatchC, spec}};
        for (int i = 0; i < (m->yuv ? (int)U_COUNT_ALL : (int)U_COUNT); i++)
            if (!m->makePipe(uspv[i].sp, m->uiLayout, m->uiPipes[i], uspv[i].sc)) { destroy(); err_ = fail; return false; }
        const uint32_t nuS = m->yuv ? 2 : 1;
        VkDescriptorPoolSize ups[3] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, nuS * kUiSets},
                                       {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, nuS * kUiSets},
                                       {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * kUiSets}};
        VkDescriptorPoolCreateInfo udpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        udpci.maxSets = kUiSets;
        udpci.poolSizeCount = 3;
        udpci.pPoolSizes = ups;
        if (m->CreateDescriptorPool(d, &udpci, nullptr, &m->uiPool) != VK_SUCCESS) { destroy(); err_ = fail; return false; }
        VkDescriptorSetLayout ul[kUiSets];
        for (auto& l : ul) l = m->uiDsl;
        VkDescriptorSetAllocateInfo udsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        udsai.descriptorPool = m->uiPool;
        udsai.descriptorSetCount = kUiSets;
        udsai.pSetLayouts = ul;
        if (m->AllocateDescriptorSets(d, &udsai, m->uiSets) != VK_SUCCESS) { destroy(); err_ = fail; return false; }
    }
    if (!m->dbgEnsure()) { destroy(); err_ = kDbgNotBuilt; return false; }   // Settings::debugView set at init(): its pipelines

    if (!resize(ii.width, ii.height, ii.viewFormat)) {
        const char* why = m->sizeWhy ? m->sizeWhy : "a buffer of this frame size could not be allocated";
        destroy();
        err_ = why;
        return false;
    }
    return true;
}

bool Interpolator::resize(uint32_t w, uint32_t h, VkFormat fmt) {
    return m && m->rebuild(w, h, fmt, false);
}

bool Interpolator::Impl::rebuild(uint32_t w, uint32_t h, VkFormat fmt, bool force) {
    // Invariant: W != 0 <=> all size-dependent buffers exist. Any failure leaves W = H = 0.
    // The pipelines belong to the Format chosen at init(): a view format of another Format is refused (BGRA <-> RGBA is fine).
    // The size must fit the device under the settings (sizeLimitError: kMinSize, buffer ranges, dispatch sizes); sizeWhy says why not.
    Format nf;
    sizeWhy = nullptr;
    if (!formatOf(fmt, &nf) || nf != this->fmt) sizeWhy = "resize to a view format of another Format (the pipelines belong to the Format of init())";
    else sizeWhy = sizeLimitError(w, h, set, lim);
    bool ok = !sizeWhy;
    if (ok && !force && w == W && h == H) { info.viewFormat = fmt; return true; }   // buffers do not depend on the format
    destroySizeDependent();
    W = H = 0;
    haveEst = false;
    if (!ok) return false;
    info.viewFormat = fmt;
    W = w; H = h;
    if (!createSizeDependent()) {
        destroySizeDependent();
        W = H = 0;
        return false;
    }
    return true;
}

const char* learnedNetName(LearnedNet n) { return n == LearnedNet::E2v5 ? kMlNetInfo[0].name : nullptr; }
bool learnedNetFromName(const char* name, LearnedNet* out) {
    if (!name || strcmp(name, kMlNetInfo[0].name)) return false;
    if (out) *out = LearnedNet::E2v5;
    return true;
}

bool Interpolator::validate(const Settings& s) {
    auto in = [](float v, float lo, float hi) { return std::isfinite(v) && v >= lo && v <= hi; };
    return in(s.trustLow, 0.0f, 255.0f) && in(s.trustHigh, 0.0f, 255.0f) && s.trustHigh - s.trustLow >= 1.0f &&
           in(s.globalFallback, 0.0f, 1.0f) && s.globalFallback > 0.0f &&
           in(s.sceneCutSad, 0.0f, 255.0f) && s.sceneCutSad > 0.0f &&
           in(s.hudBonus, 0.0f, 255.0f) && in(s.smoothBonus, 0.0f, 255.0f) && in(s.searchPenalty, 0.0f, 255.0f) &&
           s.searchRadius >= 1 && s.searchRadius <= 12 && s.searchRadiusTop >= 1 && s.searchRadiusTop <= 12 &&
           (s.flowScale == 1 || s.flowScale == 2 || s.flowScale == 4) && s.pyramidMinSide >= 16 && s.pyramidMinSide <= 256 &&
           (uint32_t)s.uiProtection <= (uint32_t)UiProtection::CrosshairV2 && (uint32_t)s.motionVersion <= (uint32_t)MotionVersion::V3 &&
           (uint32_t)s.v2Mode <= (uint32_t)V2Mode::Performance && (uint32_t)s.v3Guard <= (uint32_t)V3Guard::V2Motion && s.v3Blend <= kV3BlendMax && s.v3Motion <= kV3MotionMax && (uint32_t)s.debugView <= (uint32_t)DebugView::UiClean && s.learnedNet == LearnedNet::E2v5 &&
           in(s.v3Smear, 0.0f, 1.0f) && in(s.v3OcclSoft, 0.25f, 4.0f) && in(s.v3Unsure, 0.25f, 4.0f) && in(s.v3SceneCut, 0.0f, 4.0f) &&
           s.blurSamples <= kBlurMaxSamples && (uint32_t)s.blurWeights <= (uint32_t)BlurWeights::Gaussian &&
           in(s.uiCrosshairArea, 0.0f, 1.0f) && s.uiCrosshairArea > 0.0f &&
           in(s.uiBoxHeightFrac, 0.0f, 1.0f) &&
           s.uiBoxWidth <= 16384 && s.uiBoxHeight <= 16384 &&
           s.uiBoxOffsetX >= -16384 && s.uiBoxOffsetX <= 16384 && s.uiBoxOffsetY >= -16384 && s.uiBoxOffsetY <= 16384 &&
           s.uiStillFrames >= 1 && s.uiStillFrames <= 255 && in(s.uiStillThreshold, 0.0f, 255.0f) &&
           s.uiInpaintReach >= 1 && s.uiInpaintReach <= 16 &&
           in(s.uiMaxStillFraction, 0.0f, 1.0f) && s.uiMaxStillFraction > 0.0f &&
           in(s.uiContrastThreshold, 0.0f, 255.0f) && s.uiLeaveFrames >= 1 && s.uiLeaveFrames <= 255 && s.uiReentryFrames <= 255 &&
           s.uiHistory >= kUiFrames && s.uiHistory <= kUiFramesMax;
}

const char* Interpolator::settingsError(const Settings& s, Format f) {
    if (!validate(s)) return "invalid settings: a field is out of range or not finite (see Interpolator::validate)";
    if (s.motionVersion == MotionVersion::Learned && s.flowScale != 2) return kMlFlowScale;
    (void)f;   // (no Format rule today: every motion version runs every Format)
    return nullptr;
}

const char* Interpolator::settingsError(const Settings& s) const {
    if (const char* e = settingsError(s, m ? m->fmt : Format::Rgba8)) return e;
    if (m && s.motionVersion == MotionVersion::V2 && m->v2Why) return m->v2Why;
    if (m && s.motionVersion == MotionVersion::V3 && s.v3Guard != V3Guard::Off && m->v2Why) return m->v2Why;   // (the guard runs V2's estimate: the same device requirement)
    if (m && s.debugView != DebugView::Off && !m->yuv && !(m->cs.dbgMotion.code && m->cs.dbgMotionV2.code && m->cs.dbgMotionV3.code && m->cs.dbgUi.code)) return kDbgNotBuilt;
    if (m && s.motionVersion == MotionVersion::V2 && !m->cs.v2Morph.code) return kV2NotBuilt;
    if (m && s.motionVersion == MotionVersion::Learned && !m->cs.mlDup.code) return kMlNotBuilt;
    if (m && s.motionVersion == MotionVersion::V3 && (!m->cs.v3Synth.code || !m->cs.v3Planes.code)) return kV3NotBuilt;
    return m && m->W ? sizeLimitError(m->W, m->H, s, m->lim) : nullptr;   // the held size under these settings
}

const char* Interpolator::sizeError(uint32_t w, uint32_t h) const {
    if (!m) return "not initialised";
    return sizeLimitError(w, h, m->set, m->lim);
}

bool Interpolator::setSettings(const Settings& s) {
    if (!m || settingsError(s) || s.flowScale != m->set.flowScale || s.pyramidMinSide != m->set.pyramidMinSide) return false;
    // UI protection, and a motion version selected for the first time: creating what the new settings need is allowed here (it
    // frees nothing, so no GPU wait); false = the creation failed and nothing changed (a version created before a later failure
    // is kept, unused, for releaseUnusedMotion()).
    Settings old = m->set;
    const UiBox oldBox = m->computeBox(), oldWork = m->computeWork(oldBox);
    m->set = s;
    if (!m->motionEnsure() || !m->uiEnsure() || !m->dbgEnsure()) { m->set = old; return false; }
    m->uiAdopted(old);
    // A switch of the motion version drops the estimate, the way the library already has for "no estimate" (after init/resize):
    // recordSynth / recordSynthBlur return false until the next recordEstimate, which runs the new version. So does a change of
    // the UI protection mode or of its box (task 112): the remaining synths of the pair would paste the new mode's patch over the
    // old mode's mask.
    if (s.motionVersion != old.motionVersion || m->uiChanged(old, oldBox, oldWork)) m->haveEst = false;
    return true;
}

bool Interpolator::reconfigure(const Settings& s) {
    if (!m || settingsError(s)) return false;
    bool structural = s.flowScale != m->set.flowScale || s.pyramidMinSide != m->set.pyramidMinSide;
    Settings old = m->set;
    const UiBox oldBox = m->computeBox(), oldWork = m->computeWork(oldBox);
    m->set = s;
    if (!m->motionEnsure()) { m->set = old; return false; }
    // re-create only if a size is held (W == 0: a later resize() creates the buffers with the new settings)
    if (structural && m->W) return m->rebuild(m->W, m->H, m->info.viewFormat, true);   // (drops the estimate anyway)
    if (!m->uiEnsure() || !m->dbgEnsure()) { m->set = old; return false; }
    m->uiAdopted(old);
    if (s.motionVersion != old.motionVersion || m->uiChanged(old, oldBox, oldWork)) m->haveEst = false;   // as in setSettings
    return true;
}

void Interpolator::releaseUnusedUi() {
    if (!m) return;
    for (UiEntry& e : m->uiE)
        if (!e.active && (e.img || e.ws.b || e.box.b)) m->uiFreeEntry(e);
}

void Interpolator::releaseUnusedMotion() {
    if (!m) return;
    for (int v = 0; v < kMotionVersions; v++)
        if (v != (int)m->set.motionVersion && m->mo[v].live) m->motionFree(m->mo[v]);
    // Settings::v3Guard back to Off: V2's two estimate buffers held for the guard go (the host promises no work in flight, as for the rest of this call)
    MotionGroup& g3 = m->mo[(int)MotionVersion::V3];
    if (m->set.v3Guard == V3Guard::Off && g3.bufs.size() == V3B_GCOUNT) {
        m->freeBuf(g3.bufs[V3B_GFIELD]);
        m->freeBuf(g3.bufs[V3B_GQUAL]);
        g3.bufs.resize(V3B_COUNT);
    }
}

// Creates the resources of the selected motion version if it has none yet (see MotionGroup); never frees another version's.
// False = a creation failed: what this call created is freed again and the caller refuses the settings.
bool Interpolator::Impl::motionEnsure() {
    MotionGroup& g = mg();
    if (g.live) {
        if (mv3() && set.v3Motion != 0 && !v35Ensure()) return false;
        if (mv3() && set.v3Guard != V3Guard::Off && !guardEnsure()) return false;   // (Settings::v3Guard turned on: V2's estimate passes and buffers)
        return !W || !mlearned() || mlSized();   // (Learned: float32 activations after half ones grow the activation buffer)
    }
    // V1: the motion estimate is the same code for every format; the four colour passes of the estimate/synth set are the format's,
    // as are the two colour passes of the blur. V2: both modes' pipelines (see MotionGroup), the format's luma and synthesis; empty
    // slots are skipped.
    struct { Spv sp; const VkSpecializationInfo* sc; } spv[P_COUNT] = {
        {cs.luma, spec}, {SPV(down), nullptr}, {SPV(medhist), nullptr}, {SPV(medhist_wide), nullptr}, {SPV(search), nullptr}, {SPV(search_wide), nullptr}, {SPV(median3), nullptr},
        {SPV(subpix), nullptr}, {SPV(pixfield), nullptr}, {cs.error, spec}, {cs.resolve, spec}, {cs.fallback, spec},
        {SPV(lookup), nullptr}};
    if (set.motionVersion == MotionVersion::V2) {
        // V1's luma and pyramid; the rest is V2's own: its faster passes of V1's block field, then each mode's field and synthesis
        for (int i = P_MEDHIST; i <= P_LOOKUP; i++) spv[i].sp = Spv{};
        spv[P_V2_MEDHIST].sp = SPV(v2_medhist);
        spv[P_V2_MEDHIST_WIDE].sp = SPV(v2_medhist_wide);
        spv[P_V2_SEARCH].sp = SPV(v2_search);
        spv[P_V2_SEARCH_WIDE].sp = SPV(v2_search_wide);
        spv[P_V2_CLEAN].sp = SPV(v2_clean);
        spv[P_V2_MORPH] = {cs.v2Morph, spec};   // (the synthesis: the format's own variant)
        spv[P_V2_REPICK].sp = SPV(v2_repick);
        spv[P_V2_QFIELD].sp = SPV(v2_qfield);
        spv[P_V2_QOWN].sp = SPV(v2_qown);
        spv[P_V2_QOWN_WIDE].sp = SPV(v2_qown_wide);
        spv[P_V2_QCOV].sp = SPV(v2_qcov);
        spv[P_V2_QCOV_WIDE].sp = SPV(v2_qcov_wide);
        spv[P_V2_QMORPH] = {cs.v2QMorph, spec};
        spv[P_V2_STATIC].sp = SPV(v2_static);
        spv[P_V2_ENCLOSE].sp = SPV(v2_enclose);
        spv[P_V2_MORPH_S] = {cs.v2MorphS, spec};
        spv[P_V2_QMORPH_S] = {cs.v2QMorphS, spec};
    }
    if (set.motionVersion == MotionVersion::V3) {
        // V1's luma only (the format's); the rest is V3's own. Built so far: the synthesis (skeleton) and the estimate's planes, pyramid,
        // histograms, search and dense pass (E1); the other P_V3_* slots are reserved for the later steps
        for (int i = P_DOWN; i <= P_LOOKUP; i++) spv[i].sp = Spv{};
        spv[P_V3_PLANES] = {cs.v3Planes, spec};   // (reads colour: the format's own variant)
        spv[P_V3_PYR].sp = SPV(v3_pyr);
        spv[P_V3_HIST].sp = SPV(v3_hist);
        spv[P_V3_SEARCH].sp = SPV(v3_search);
        spv[P_V3_SEARCH_TOP].sp = SPV(v3_search_top);
        spv[P_V3_DENSE].sp = SPV(v3_dense);
        spv[P_V3_SEARCH_J].sp = SPV(v3_search_j);
        spv[P_V3_SEARCH_TOP_J].sp = SPV(v3_search_top_j);
        spv[P_V3_DENSE_J].sp = SPV(v3_dense_j);
        spv[P_V3_WIDE].sp = SPV(v3_wide);
        spv[P_V3_STATS].sp = SPV(v3_stats);
        spv[P_V3_OCCL].sp = SPV(v3_occl);
        spv[P_V3_FILL].sp = SPV(v3_fill);
        spv[P_V3_PROJECT].sp = SPV(v3_project);
        spv[P_V3_ENCLOSE].sp = SPV(v3_enclose);
        spv[P_V3_DECIDE].sp = SPV(v3_decide);
        spv[P_V2_STATIC].sp = SPV(v2_static);   // V2's shader as is (V3 pushes V3PC, whose first 8 ints are V2PC's; b14 has V2B_QUAL's S / s8 offsets)
        spv[P_V3_SYNTH] = {cs.v3Synth, spec};
        spv[P_V3_SYNTH_L] = {cs.v3SynthL, spec};
        spv[P_V3_SYNTH_B] = {cs.v3SynthB, spec};
    }
    // Learned: V1's luma, synthesis and blur, its own passes instead of V1's search; the convs specialised per layer shape and net
    // (shaders/ml_conv.comp: GI, GO, LINEAR), and the weights buffer
    struct MlSpec { int32_t gi, go, linear, got, half, ext; } mlSpec[2 * 5 + 2]{};
    VkSpecializationMapEntry mlEnt[6] = {{20, 0, 4}, {21, 4, 4}, {22, 8, 4}, {23, 12, 4}, {24, 16, 4}, {25, 20, 4}};   // (25 = ML_EXT: ml_match only)
    VkSpecializationInfo mlSi[2 * 5 + 2]{};
    if (mlearned()) {
#ifdef FG_HAVE_ML
        for (int i = P_DOWN; i <= P_PIXFIELD; i++) spv[i].sp = Spv{};
        spv[P_ML_DUP] = {cs.mlDup, spec};   // (the duplicate test reads colour: the format's own variant)
        spv[P_ML_REDUCE].sp = SPV(ml_reduce);
        spv[P_ML_PYR].sp = SPV(ml_pyr);
        spv[P_ML_OUT].sp = SPV(ml_out);
        spv[P_ML_ZERO].sp = SPV(ml_zero);         // Settings::learnedZeroTest (ml_out's pass with the zero-motion test)
        spv[P_ML_GUARD] = {cs.mlGuard, spec};     // Settings::learnedStaticGuard (reads and writes colour: the format's variant)
        for (int half = 0; half < 2; half++) {
            const int jm = 2 * 5 + half;
            mlSpec[jm].half = half;
            mlSi[jm] = {6, mlEnt, sizeof(MlSpec), &mlSpec[jm]};
            spv[mlMatchPipe(half)] = {SPV(ml_match), &mlSi[jm]};
            const int gw = mlWidth() / 4, gi = groups4(kMlNetInfo[0].inCh);
            // the input channels (14 padded to 16) -> w, w -> w (each whole and split), w -> 2 (padded to 4)
            const MlSpec shapes[5] = {{gi, gw, 0, gw, half, 0}, {gi, gw, 0, 1, half, 0}, {gw, gw, 0, gw, half, 0}, {gw, gw, 0, 1, half, 0}, {gw, 1, 1, 1, half, 0}};
            for (int k = 0; k < 5; k++) {
                const int j = 5 * half + k;
                mlSpec[j] = shapes[k];
                mlSi[j] = {6, mlEnt, sizeof(MlSpec), &mlSpec[j]};
                spv[P_ML_CONV + j] = {SPV(ml_conv), &mlSi[j]};
            }
        }
        size_t words = 0;
        for (const MlNet& n : kMlNet) words += n.words;
        if (!allocBuf(mlWeights, (VkDeviceSize)words * 4)) return false;
        mlWeightsUp = false;
#else
        return false;   // (refused before: settingsError, kMlNotBuilt)
#endif
    }
    struct { Spv sp; const VkSpecializationInfo* sc; } bspv[B_COUNT_] = {
        {SPV(blur_lookup), nullptr}, {cs.blurError, spec}, {cs.blurResolve, spec}};
    bool ok = true;
    for (int i = 0; ok && i < P_COUNT; i++) ok = !spv[i].sp.code || makePipe(spv[i].sp, layout, g.pipes[i], spv[i].sc);
    // V2's and V3's motion blur is their synthesis pass with n samples (v2Synth, v3Synth): no blur pipelines
    for (int i = 0; ok && set.motionVersion != MotionVersion::V2 && set.motionVersion != MotionVersion::V3 && i < B_COUNT_; i++) ok = makePipe(bspv[i].sp, blurLayout, g.blurPipes[i], bspv[i].sc);
    ok = ok && (!W || motionSized(g));
    if (!ok) { motionFree(g); return false; }
    g.live = true;
    if (mv3() && set.v3Motion != 0 && !v35Ensure()) { motionFree(g); return false; }
    if (mv3() && set.v3Guard != V3Guard::Off && !guardEnsure()) { motionFree(g); return false; }
    return true;
}

// Settings::v3Motion = 1 (V3 selected, its group live): V3.5's pipelines. Created the first time the setting is on (setSettings / init), never otherwise; stay until the
// group is freed. False: a creation failed (what exists stays; nothing is half-used: the estimate checks every pipe it needs).
bool Interpolator::Impl::v35Ensure() {
    MotionGroup& g = mo[(int)MotionVersion::V3];
    const struct { Pipe p; Spv sp; } list[] = {{P_V35_REG, SPV(v35_reg)}, {P_V35_DENSE, SPV(v3_dense_35)}, {P_V35_STATS, SPV(v35_stats)}, {P_V35_RGN, SPV(v35_rgn)}, {P_V35_SEARCH, SPV(v3_search_35)}};
    for (const auto& e : list)
        if (!g.pipes[e.p] && !makePipe(e.sp, layout, g.pipes[e.p], nullptr)) return false;
    return true;
}

// Settings::v3Guard on (V3 selected, its group live): the pipelines of V2's estimate (V1's pyramid pass, V2's median histograms and searches, the re-pick,
// the clean-up and Quality's per-pixel field; V2's synthesis and coverage passes are not needed) and v3_guard in V3's group, and, while a size is held, V2's two
// estimate buffers beside V3's three. Creates what is missing, never frees (releaseUnusedMotion does). False: a creation failed, nothing is half-used
// (a pipeline that exists stays; the buffers are all or none).
bool Interpolator::Impl::guardEnsure() {
    MotionGroup& g = mo[(int)MotionVersion::V3];
    const struct { Pipe p; Spv sp; } list[] = {{P_DOWN, SPV(down)}, {P_V2_MEDHIST, SPV(v2_medhist)}, {P_V2_MEDHIST_WIDE, SPV(v2_medhist_wide)}, {P_V2_SEARCH, SPV(v2_search)},
                                               {P_V2_SEARCH_WIDE, SPV(v2_search_wide)}, {P_V2_REPICK, SPV(v2_repick)}, {P_V2_CLEAN, SPV(v2_clean)}, {P_V2_QFIELD, SPV(v2_qfield)},
                                               {P_V3_ANCHOR, SPV(v3_anchor)}, {P_V3_GUARD, SPV(v3_guard)}};
    for (const auto& e : list)
        if (!g.pipes[e.p] && !makePipe(e.sp, layout, g.pipes[e.p], nullptr)) return false;
    return !W || motionSized(g);
}

// The size-dependent buffers of one motion version, beside the shared buf[] (called while a size is held: by motionEnsure and
// createSizeDependent; destroySizeDependent frees them). V1 has none of its own; V2: V2B_*.
bool Interpolator::Impl::motionSized(MotionGroup& g) {
    if (&g == &mo[(int)MotionVersion::Learned]) return mlSized();
    if (&g == &mo[(int)MotionVersion::V3]) {
        if (g.bufs.empty()) {
            // gpu/V3.md section 4; all three are copied out by recordDebugV3. The T-field part of V3B_FLOW is held for one interior sample (ns = 1)
            const V3Layout V = v3Layout(W, H, set.flowScale);
            g.bufs.resize(V3B_COUNT);
            if (allocBuf(g.bufs[V3B_PYR], V.pyrBytes(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT) && allocBuf(g.bufs[V3B_AUX], V.auxBytes(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT) &&
                allocBuf(g.bufs[V3B_FLOW], V.flowBytes(1), VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) {
                v3Fresh = true;
                v3Ns = 1;
            } else {
                for (UiBuf& b : g.bufs) freeBuf(b);
                g.bufs.clear();
                return false;
            }
        }
        if (set.v3Guard != V3Guard::Off && g.bufs.size() == V3B_COUNT) {   // Settings::v3Guard: V2's estimate buffers (the sizes of V2's own, below)
            g.bufs.resize(V3B_GCOUNT);
            if (!allocBuf(g.bufs[V3B_GFIELD], ((VkDeviceSize)lv[0].bw * lv[0].bh * 2 + (VkDeviceSize)fw * fh) * 8) ||
                !allocBuf(g.bufs[V3B_GQUAL], ((VkDeviceSize)fw * fh * 5 + (VkDeviceSize)3 * ((fw + 7) / 8) * ((fh + 7) / 8)) * 4)) {
                freeBuf(g.bufs[V3B_GFIELD]);
                freeBuf(g.bufs[V3B_GQUAL]);
                g.bufs.resize(V3B_COUNT);
                return false;
            }
        }
        return true;
    }
    if (&g != &mo[(int)MotionVersion::V2] || !g.bufs.empty()) return true;
    g.bufs.resize(V2B_COUNT);
    // the cleaned field, then the search's fitted finest field (v2_est.glsl); recordDebugMaps copies the cleaned one
    if (allocBuf(g.bufs[V2B_FIELD], ((VkDeviceSize)lv[0].bw * lv[0].bh * 2 + (VkDeviceSize)fw * fh) * 8, VK_BUFFER_USAGE_TRANSFER_SRC_BIT) &&
        allocBuf(g.bufs[V2B_QUAL], ((VkDeviceSize)fw * fh * 5 + (VkDeviceSize)3 * ((fw + 7) / 8) * ((fh + 7) / 8)) * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT))
        return true;
    for (UiBuf& b : g.bufs) freeBuf(b);
    g.bufs.clear();
    return false;
}

// Learned's size-dependent buffers (MLB_*): the state once per size, the
// activations grown whenever the storage needs more bytes than they hold (the outgrown buffer may still be in a pending record: mlGrave, freed with the size). False = an
// allocation failed (what this call allocated is freed; an existing smaller activation buffer stays).
bool Interpolator::Impl::mlSized() {
    MotionGroup& g = mo[(int)MotionVersion::Learned];
    const MlLayout L = mlLayout(W, H, fw, fh, set);
    if (g.bufs.empty()) {
        g.bufs.resize(MLB_COUNT);
        mlActBytes = 0;
        if (!allocBuf(g.bufs[MLB_STATE], L.stateBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) { g.bufs.clear(); return false; }
        mlStateBytes = L.stateBytes;
    }
    if (mlActBytes >= L.actBytes) return true;
    UiBuf nb;
    if (!allocBuf(nb, L.actBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) {
        if (!mlActBytes) { for (UiBuf& b : g.bufs) freeBuf(b); g.bufs.clear(); }
        return false;
    }
    if (g.bufs[MLB_ACT].b) mlGrave.push_back(g.bufs[MLB_ACT]);
    g.bufs[MLB_ACT] = nb;
    mlActBytes = L.actBytes;
    return true;
}

// Learned: the weights of the net into mlWeights, once after it was created (CmdUpdateBuffer, at most 64 KiB per command).
// Recorded between the barriers of a counter clear (Rec::clearCounters: a transfer write ordered after every earlier record and before
// this one's passes). No-op unless pending.
void Interpolator::Impl::mlUpload(VkCommandBuffer cb) {
#ifdef FG_HAVE_ML
    if (!mlWeights.b || mlWeightsUp) return;
    VkDeviceSize off = 0;
    for (const MlNet& n : kMlNet)
        for (size_t w = 0; w < n.words; ) {
            const size_t k = std::min<size_t>(n.words - w, 65536 / 4);
            CmdUpdateBuffer(cb, mlWeights.b, off, k * 4, n.w + w);
            w += k;
            off += k * 4;
        }
    mlWeightsUp = true;
#else
    (void)cb;
#endif
}

// Frees one motion version's resources (retire rule of resize()/destroy()).
void Interpolator::Impl::motionFree(MotionGroup& g) {
    for (VkPipeline p : g.pipes) if (p) DestroyPipeline(info.device, p, nullptr);
    for (VkPipeline p : g.blurPipes) if (p) DestroyPipeline(info.device, p, nullptr);
    for (UiBuf& b : g.bufs) freeBuf(b);
    if (&g == &mo[(int)MotionVersion::Learned]) {
        freeBuf(mlWeights);
        mlWeightsUp = false;
        for (UiBuf& b : mlGrave) freeBuf(b);
        mlGrave.clear();
        mlActBytes = 0;
    }
    g = MotionGroup{};
}

Format Interpolator::format() const {
    return m ? m->fmt : Format::Rgba8;
}

bool Interpolator::v3BlendActive() const {
    ColourSpv c;
    return m && m->mv3() && m->set.v3Blend != 0 && !m->set.v3DetailSplit && colourSpv(m->fmt, c) && c.v3SynthB.code;
}

bool Interpolator::v3MotionActive() const {
    return m && m->mv3() && m->set.v3Motion != 0 && m->mo[(int)MotionVersion::V3].pipes[P_V35_REG] != VK_NULL_HANDLE;
}

Settings Interpolator::settings() const {
    return m ? m->set : Settings{};
}

void Interpolator::destroy() {
    if (!m) return;
    VkDevice d = m->info.device;
    if (m->DestroyBuffer) { m->destroySizeDependent(); m->freeBuf(m->rejNone); }
    if (m->qpool) m->DestroyQueryPool(d, m->qpool, nullptr);
    if (m->pool) m->DestroyDescriptorPool(d, m->pool, nullptr);
    if (m->DestroyPipeline) for (MotionGroup& g : m->mo) m->motionFree(g);
    for (auto p : m->uiPipes) if (p) m->DestroyPipeline(d, p, nullptr);
    for (auto p : m->dbgPipes) if (p) m->DestroyPipeline(d, p, nullptr);
    if (m->blurLayout) m->DestroyPipelineLayout(d, m->blurLayout, nullptr);
    if (m->uiPool) m->DestroyDescriptorPool(d, m->uiPool, nullptr);
    if (m->uiLayout) m->DestroyPipelineLayout(d, m->uiLayout, nullptr);
    if (m->uiDsl) m->DestroyDescriptorSetLayout(d, m->uiDsl, nullptr);
    if (m->layout) m->DestroyPipelineLayout(d, m->layout, nullptr);
    if (m->dsl) m->DestroyDescriptorSetLayout(d, m->dsl, nullptr);
    if (m->sampler) m->DestroySampler(d, m->sampler, nullptr);
    delete m;
    m = nullptr;
}

bool Interpolator::setProfiling(bool on) {
    if (!m) return false;
    m->profiling = false;
    if (on && !m->qpool) {
        VkQueryPoolCreateInfo q{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        q.queryType = VK_QUERY_TYPE_TIMESTAMP;
        q.queryCount = Impl::kMaxQ;
        if (m->CreateQueryPool(m->info.device, &q, nullptr, &m->qpool) != VK_SUCCESS) {
            m->qpool = VK_NULL_HANDLE;
            return false;
        }
    }
    m->profiling = on;
    return true;
}

void Interpolator::beginFrame() {
    if (m) m->frameSets[0] = m->frameSets[1] = m->uiFrameSets = 0;
}

bool Interpolator::readTimings(std::vector<const char*>& names, std::vector<uint64_t>& ticks) {
    if (!m || !m->qpool || m->qcount < 2) return false;
    std::vector<uint64_t> ts(m->qcount);
    if (m->GetQueryPoolResults(m->info.device, m->qpool, 0, m->qcount, ts.size() * 8, ts.data(), 8,
                               VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS)
        return false;
    names.assign(m->qnames.begin() + 1, m->qnames.end());
    ticks.clear();
    for (uint32_t i = 1; i < m->qcount; i++) ticks.push_back(ts[i] - ts[i - 1]);
    return true;
}

bool Interpolator::recordDebugCounters(VkCommandBuffer cb, VkBuffer dst, VkDeviceSize dstOffset) {
    if (!m || !m->W || !m->buf[2]) return false;
    Impl& I = *m;
    // the synth's atomics (and the estimate's SAD sum) -> this copy; nothing of ours is written, so the next record's own
    // clearCounters barrier (CS RW -> transfer write) already orders its fill after this read
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    I.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    // u[U_UNTRUSTED], u[U_SADSUM]; V2 (both modes): its own scene-cut sum u[3] (v2_est.glsl U_V2SAD, the re-pick's)
    VkBufferCopy c[2] = {{0, dstOffset, 4}, {I.mv2() ? 12u : 4u, dstOffset + 4, 4}};
    I.CmdCopyBuffer(cb, I.buf[2], dst, 2, c);
    return true;
}

uint32_t Interpolator::debugSadBlocks() const {
    return m && !m->lv.empty() ? (uint32_t)(m->lv[0].bw * m->lv[0].bh) : 0;
}

bool Interpolator::debugMapLayout(DebugMapLayout& l) const {
    if (!m || !m->W || m->lv.empty()) return false;
    const Impl& I = *m;
    l.W = I.W; l.H = I.H; l.fw = (uint32_t)I.fw; l.fh = (uint32_t)I.fh;
    l.bw = (uint32_t)I.lv[0].bw; l.bh = (uint32_t)I.lv[0].bh; l.fscale = I.set.flowScale;
    l.eStride = (I.W + 3) / 4;
    l.blockOff = 0;
    l.pixOff = (VkDeviceSize)l.bw * l.bh * 8;
    l.eOff = l.pixOff + (VkDeviceSize)l.fw * l.fh * 8;
    l.size = l.eOff + (VkDeviceSize)l.eStride * l.H * 4;
    return true;
}

bool Interpolator::recordDebugMaps(VkCommandBuffer cb, VkBuffer dst, VkDeviceSize dstOffset) {
    DebugMapLayout L;
    if (!debugMapLayout(L) || !m->buf[1] || !m->buf[2]) return false;
    Impl& I = *m;
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    I.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    const MotionGroup& v2g = I.mo[(int)MotionVersion::V2];
    if (I.mv3() && !I.mo[(int)MotionVersion::V3].bufs.empty()) {
        // V3 in V1's slots (gpu/V3.md section 2, display only): the block slot holds F01 at the level-0 block centres (bw x bh), the pixel slot F01 at
        // the flow pixels (fw x fh), the mismatch slot zeros. NOT halved and not resampled to the midpoint grid (that needs a pass, the skeleton has
        // none): the vectors are full prev->curr vectors, which gpu/fgmaps.py halves for a V3 dump (header word 14 = 3).
        const V3Layout V = v3Layout(I.W, I.H, I.set.flowScale);
        const VkBuffer fb = I.mo[(int)MotionVersion::V3].bufs[V3B_FLOW].b;
        const VkDeviceSize ff0 = 64 * (VkDeviceSize)V.B + 2 * (VkDeviceSize)V.N * 8;   // FF[0] in V3B_FLOW
        VkBufferCopy pc{ff0, dstOffset + L.pixOff, (VkDeviceSize)V.N * 8};
        I.CmdCopyBuffer(cb, fb, dst, 1, &pc);
        std::vector<VkBufferCopy> bc;
        bc.reserve((size_t)L.bw * L.bh);
        for (uint32_t j = 0; j < L.bh; j++)
            for (uint32_t i = 0; i < L.bw; i++) {
                const uint32_t x = std::min(8 * i + 4, L.fw - 1), y = std::min(8 * j + 4, L.fh - 1);
                bc.push_back({ff0 + ((VkDeviceSize)y * L.fw + x) * 8, dstOffset + L.blockOff + ((VkDeviceSize)j * L.bw + i) * 8, 8});
            }
        I.CmdCopyBuffer(cb, fb, dst, (uint32_t)bc.size(), bc.data());
        I.CmdFillBuffer(cb, dst, dstOffset + L.eOff, L.size - L.eOff, 0);
        return true;
    }
    if (I.mv2() && I.v2Clean && !v2g.bufs.empty()) {
        // V2: the cleaned block field (both modes). Performance: no pixel field, no e (zeros). Quality (with its coverage): its
        // per-pixel field (v2_qfield.comp) in the pixel-field slot, and the e slot starts with the coverage, fw * fh words
        // packUnorm2x16(cP, cC) (shaders/v2_quality.glsl) if they fit (flowScale >= 2), the rest is zero
        VkBufferCopy f{0, dstOffset + L.blockOff, L.pixOff - L.blockOff};
        I.CmdCopyBuffer(cb, v2g.bufs[V2B_FIELD].b, dst, 1, &f);
        VkDeviceSize z = L.pixOff;
        if (I.set.v2Mode == V2Mode::Quality && I.v2Cov) {
            const VkDeviceSize n = (VkDeviceSize)I.fw * I.fh * 4;
            VkBufferCopy p{(VkDeviceSize)L.bw * L.bh * 16, dstOffset + L.pixOff, L.eOff - L.pixOff};
            I.CmdCopyBuffer(cb, v2g.bufs[V2B_FIELD].b, dst, 1, &p);
            z = L.eOff;
            if (n <= L.size - L.eOff) {
                VkBufferCopy c{3 * n, dstOffset + L.eOff, n};
                I.CmdCopyBuffer(cb, v2g.bufs[V2B_QUAL].b, dst, 1, &c);
                z += n;
            }
        }
        if (z < L.size) I.CmdFillBuffer(cb, dst, dstOffset + z, L.size - z, 0);
        return true;
    }
    int blockIn = I.set.subpixel ? I.fieldSub : I.lv[0].fM;   // == pc.fIn of the pixel-field pass
    VkBufferCopy f[2] = {{(VkDeviceSize)blockIn * 8, dstOffset + L.blockOff, L.pixOff - L.blockOff},
                         {(VkDeviceSize)I.fieldPF * 8, dstOffset + L.pixOff, L.eOff - L.pixOff}};
    if (I.mlearned()) I.CmdFillBuffer(cb, dst, dstOffset + L.blockOff, L.pixOff - L.blockOff, 0);   // Learned: no block field
    I.CmdCopyBuffer(cb, I.buf[1], dst, I.mlearned() ? 1 : 2, I.mlearned() ? &f[1] : f);
    VkBufferCopy e{(VkDeviceSize)kEOff * 4, dstOffset + L.eOff, L.size - L.eOff};
    I.CmdCopyBuffer(cb, I.buf[2], dst, 1, &e);
    return true;
}

bool Interpolator::debugV3Layout(DebugV3Layout& l) const {
    if (!m || !m->W || !m->mv3() || m->lv.empty()) return false;
    const Impl& I = *m;
    const V3Layout V = v3Layout(I.W, I.H, I.set.flowScale);
    l = DebugV3Layout{};
    l.W = I.W; l.H = I.H; l.fw = (uint32_t)V.fw; l.fh = (uint32_t)V.fh; l.bw = (uint32_t)I.lv[0].bw; l.bh = (uint32_t)I.lv[0].bh;
    l.fscale = I.set.flowScale; l.nLevels = (uint32_t)V.n;
    for (int i = 0; i < V.n; i++) { l.lw[i] = (uint32_t)V.w[i]; l.lh[i] = (uint32_t)V.h[i]; l.lgw[i] = (uint32_t)V.gw[i]; l.lgh[i] = (uint32_t)V.gh[i]; }
    const VkDeviceSize n2 = (VkDeviceSize)V.N * 8, n1 = (VkDeviceSize)V.N * 4;
    l.ff0 = 64; l.ff1 = l.ff0 + n2; l.fd0 = l.ff1 + n2; l.fd1 = l.fd0 + n2;
    l.map0 = l.fd1 + n2; l.map1 = l.map0 + n1; l.s = l.map1 + n1; l.blk = l.s + n1;
    VkDeviceSize o = l.blk;
    for (int i = 0; i < V.n; i++) { l.blkLevel[i] = o; o += (VkDeviceSize)V.gw[i] * V.gh[i] * 4 * 16; }
    l.g = o; o += 64 * 4;   // G words 0..63
    l.cells = o; o += (VkDeviceSize)V.N8 * 12;   // (Y2) the static mask's cell maxima s8, enclosure E and "overlay near" bits C (N8 words each)
    l.tf = o; o += (VkDeviceSize)V.N * 16;   // (task 109) T-field slot 0: the latest synth's projected record (4 words per flow px), its t in header word 14
    l.v35 = o; o += 200 * 4;   // (task 161) G words 3896..4095: V3.5's regional camera models (and the diagnosis counters)
    l.size = o;
    return true;
}

bool Interpolator::recordDebugV3(VkCommandBuffer cb, VkBuffer dst, VkDeviceSize dstOffset) {
    DebugV3Layout L;
    if (!debugV3Layout(L) || m->mo[(int)MotionVersion::V3].bufs.empty()) return false;
    Impl& I = *m;
    const MotionGroup& g = I.mo[(int)MotionVersion::V3];
    const V3Layout V = v3Layout(I.W, I.H, I.set.flowScale);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    I.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    // the 16-word header: the constants by CmdUpdateBuffer, then G.mode, G.unsureCount, G.failMix copied from the aux buffer
    const uint32_t hdr[10] = {0x42443356u, 2u, L.W, L.H, L.fw, L.fh, L.bw, L.bh, L.fscale, L.nLevels};
    I.CmdUpdateBuffer(cb, dst, dstOffset, sizeof(hdr), hdr);
    const VkDeviceSize g0 = (VkDeviceSize)V.gOff() * 4;
    VkBufferCopy h[4] = {{g0 + 4 * 4, dstOffset + 40, 4}, {g0 + 2 * 4, dstOffset + 44, 4}, {g0 + 24 * 4, dstOffset + 48, 4}, {g0 + 29 * 4, dstOffset + 52, 4}};   // G_MODE, G_UNSURE_COUNT, G_FAIL_MIX, G_HIST_D
    I.CmdCopyBuffer(cb, g.bufs[V3B_AUX].b, dst, 4, h);
    VkBufferCopy tk{g0 + 1200 * 4, dstOffset + 56, 4};   // word 14: G_TK[0], the t of T-field slot 0 (task 109)
    I.CmdCopyBuffer(cb, g.bufs[V3B_AUX].b, dst, 1, &tk);
    I.CmdFillBuffer(cb, dst, dstOffset + 60, 4, 0);   // word 15 spare
    const VkDeviceSize n2 = (VkDeviceSize)V.N * 8, n1 = (VkDeviceSize)V.N * 4, blkB = 64 * (VkDeviceSize)V.B;   // V3B_FLOW: BLK, FD[2], FF[2], TF
    VkBufferCopy fc[4] = {{blkB + 2 * n2, dstOffset + L.ff0, 2 * n2}, {blkB, dstOffset + L.fd0, 2 * n2},   // FF[0..1], FD[0..1] (adjacent in dst)
                          {0, dstOffset + L.blk, blkB}, {blkB + 4 * n2, dstOffset + L.tf, 2 * n2}};   // TF slot 0 (task 109)
    I.CmdCopyBuffer(cb, g.bufs[V3B_FLOW].b, dst, 4, fc);
    VkBufferCopy ac[2] = {{0, dstOffset + L.map0, 2 * n1}, {(VkDeviceSize)4 * V.N * 4, dstOffset + L.s, n1}};   // MAP[0..1]; S at word 4N
    I.CmdCopyBuffer(cb, g.bufs[V3B_AUX].b, dst, 2, ac);
    VkBufferCopy gc[2] = {{g0, dstOffset + L.g, 64 * 4}, {(VkDeviceSize)5 * V.N * 4, dstOffset + L.cells, (VkDeviceSize)V.N8 * 12}};   // the state G, words 0..63 (step E2: dominant motions, camera, wide counters); s8, E, C
    I.CmdCopyBuffer(cb, g.bufs[V3B_AUX].b, dst, 2, gc);
    VkBufferCopy rc{g0 + 3896 * 4, dstOffset + L.v35, 200 * 4};
    I.CmdCopyBuffer(cb, g.bufs[V3B_AUX].b, dst, 1, &rc);
    return true;
}

bool Interpolator::recordDebugV3Inject(VkCommandBuffer cb, Frame prev, Frame curr, VkBuffer flows, VkDeviceSize f01Off, VkDeviceSize f10Off) {
    if (!m || !m->W || !m->mv3() || m->mo[(int)MotionVersion::V3].bufs.empty() || !flows) return false;
    Impl::V3Inject inj{flows, f01Off, f10Off};
    m->v3inj = &inj;
    const bool ok = recordEstimate(cb, prev, curr);
    m->v3inj = nullptr;
    return ok;
}

// The weights dump is a recordSynth(t) whose synth also writes the weight stage's 28-float records (gpu/V3.md 3.16) into T-field slots after the
// interior one, every `stride`-th pixel of the raster, `count` of them (all pixels where the frame is small): V3B_FLOW grows by the slots they need
// (the first such call), and recordDebugV3Weights copies the records out. No descriptor binding is spent on the destination (all 16 are taken).
bool Interpolator::debugV3WeightsLayout(uint32_t maxPixels, DebugV3Weights& l) const {
    if (!m || !m->W || !m->mv3()) return false;
    const uint64_t P = (uint64_t)m->W * m->H;
    const uint64_t want = std::min<uint64_t>(P, maxPixels ? maxPixels : 65536);
    l.stride = (uint32_t)((P + want - 1) / want);
    l.count = (uint32_t)((P + l.stride - 1) / l.stride);
    l.size = (VkDeviceSize)l.count * 28 * 4;
    return true;
}

bool Interpolator::recordDebugV3Weights(VkCommandBuffer cb, Frame prev, Frame curr, Frame out, float t, VkBuffer dst, VkDeviceSize dstOffset, uint32_t maxPixels) {
    DebugV3Weights L;
    if (!debugV3WeightsLayout(maxPixels, L) || !dst || !(t > 0.0f && t < 1.0f)) return false;
    Impl& I = *m;
    if (!I.haveEst || prev.view != I.estPrev || curr.view != I.estCurr || !I.framesOk(prev) || !I.framesOk(curr) || !I.framesOk(out)) return false;
    if (I.frameSets[Impl::R_SYNTH] >= kSetRing) return false;
    Frame srcP = prev, srcC = curr;
    if (I.estUi) {
        const UiEntry &ep = I.uiE[I.estEnt[0]], &ec = I.uiE[I.estEnt[1]];
        if (ep.seq != I.estSeq[0] || ec.seq != I.estSeq[1] || (I.estClean && (!ep.view || !ec.view))) return false;
        if (I.uiFrameSets + 2 + I.dbgUiSets() > kUiSets) return false;
        if (I.estClean) { srcP = ep.frame(); srcC = ec.frame(); }
    }
    if (!I.v3Synth(cb, srcP, srcC, curr, out, t, t, 1, false, L.stride, L.count)) return false;
    const V3Layout V = v3Layout(I.W, I.H, I.set.flowScale);
    const VkDeviceSize slot1 = ((VkDeviceSize)16 * V.B + 8 * V.N + 4 * V.N) * 4;   // T-field slot 1 (after the interior sample's slot 0), in bytes
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    I.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    VkBufferCopy c{slot1, dstOffset, L.size};
    I.CmdCopyBuffer(cb, I.mo[(int)MotionVersion::V3].bufs[V3B_FLOW].b, dst, 1, &c);
    return true;
}

void Interpolator::setDebugLearned(const LearnedDebug& d) {
    if (m) m->mlDbg = d;
}

bool Interpolator::debugLearnedLayout(DebugLearnedLayout& l) const {
    if (!m || !m->W || !m->mlearned() || m->mo[(int)MotionVersion::Learned].bufs.empty()) return false;
    const Impl& I = *m;
    const MlLayout L = mlLayout(I.W, I.H, I.fw, I.fh, I.set);
    l.half = I.set.learnedHalf;
    l.fw = (uint32_t)I.fw; l.fh = (uint32_t)I.fh; l.pw = (uint32_t)L.pw; l.ph = (uint32_t)L.ph; l.width = (uint32_t)mlWidth();
    const VkDeviceSize st = 0;   // the state copy (shaders/ml_common.glsl), then the activations
    VkDeviceSize o = 0;
    for (int k = 0; k < kMlLevels; k++) { l.pyrOff[k] = st + o * 4; o += 2 * (VkDeviceSize)(L.pw >> k) * (L.ph >> k); }
    for (int k = 0; k < kMlLevels; k++) { l.hOff[k] = st + o * 4; o += 2 * (VkDeviceSize)(L.pw >> k) * (L.ph >> k); }
    l.actOff[0] = (st + L.stateBytes + 15) & ~(VkDeviceSize)15;
    l.actOff[1] = l.actOff[0] + (VkDeviceSize)L.r0 * L.n0 * L.elem;
    l.size = l.actOff[0] + L.actBytes;
    return true;
}

bool Interpolator::recordDebugLearned(VkCommandBuffer cb, VkBuffer dst, VkDeviceSize dstOffset) {
    DebugLearnedLayout L;
    if (!debugLearnedLayout(L)) return false;
    Impl& I = *m;
    const MotionGroup& g = I.mo[(int)MotionVersion::Learned];
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    I.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    const MlLayout M = mlLayout(I.W, I.H, I.fw, I.fh, I.set);
    VkBufferCopy sc{0, dstOffset + L.pyrOff[0], M.stateBytes};
    I.CmdCopyBuffer(cb, g.bufs[MLB_STATE].b, dst, 1, &sc);
    VkBufferCopy ac{0, dstOffset + L.actOff[0], M.actBytes};
    I.CmdCopyBuffer(cb, g.bufs[MLB_ACT].b, dst, 1, &ac);
    return true;
}

VkDescriptorSet Interpolator::Impl::takeSet(int ring, Frame prev, Frame curr, Frame out, bool blur, RejPair rej, bool guardView) {
    VkDescriptorSet set = sets[ring][setIdx[ring]];
    setIdx[ring] = (setIdx[ring] + 1) % kSetRing;
    frameSets[ring]++;
    VkDescriptorBufferInfo bi[3], bb[2] = {{blurField.b, 0, VK_WHOLE_SIZE}, {blurErr.b, 0, VK_WHOLE_SIZE}};
    VkDescriptorImageInfo ii[6] = {{VK_NULL_HANDLE, prev.view, VK_IMAGE_LAYOUT_GENERAL},
                                   {VK_NULL_HANDLE, curr.view, VK_IMAGE_LAYOUT_GENERAL},
                                   {VK_NULL_HANDLE, out.view, VK_IMAGE_LAYOUT_GENERAL},
                                   {VK_NULL_HANDLE, prev.chroma, VK_IMAGE_LAYOUT_GENERAL},
                                   {VK_NULL_HANDLE, curr.chroma, VK_IMAGE_LAYOUT_GENERAL},
                                   {VK_NULL_HANDLE, out.chroma, VK_IMAGE_LAYOUT_GENERAL}};
    VkDescriptorBufferInfo rj[2] = {{rej.a ? rej.a : rejNone.b, 0, VK_WHOLE_SIZE}, {rej.c ? rej.c : rejNone.b, 0, VK_WHOLE_SIZE}};
    const MotionGroup& v2g = mo[(int)MotionVersion::V2];
    const MotionGroup& v3gb = mo[(int)MotionVersion::V3];
    const bool gbufs = v3gb.bufs.size() == V3B_GCOUNT;   // V3's group holds V2's estimate buffers (Settings::v3Guard has been on)
    VkDescriptorBufferInfo vb = {v2g.bufs.empty() ? VK_NULL_HANDLE : v2g.bufs[V2B_FIELD].b, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo qb = {v2g.bufs.empty() ? VK_NULL_HANDLE : v2g.bufs[V2B_QUAL].b, 0, VK_WHOLE_SIZE};
    if (guardView && gbufs) {
        vb.buffer = v3gb.bufs[V3B_GFIELD].b;
        qb.buffer = v3gb.bufs[V3B_GQUAL].b;
    }
    const MotionGroup& mlg = mo[(int)MotionVersion::Learned];
    VkDescriptorBufferInfo mb[3] = {{mlg.bufs.empty() ? VK_NULL_HANDLE : mlg.bufs[MLB_STATE].b, 0, VK_WHOLE_SIZE},
                                    {mlg.bufs.empty() ? VK_NULL_HANDLE : mlg.bufs[MLB_ACT].b, 0, VK_WHOLE_SIZE},
                                    {mlWeights.b, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wr[17]{};
    uint32_t n = 0;
    auto add = [&](uint32_t binding, VkDescriptorType t) -> VkWriteDescriptorSet& {
        VkWriteDescriptorSet& w = wr[n++];
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = set;
        w.dstBinding = binding;
        w.descriptorCount = 1;
        w.descriptorType = t;
        return w;
    };
    const VkDescriptorType BUF = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, SMP = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                           IMG = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    for (uint32_t i = 0; i < 3; i++) {
        bi[i] = {buf[i], 0, VK_WHOLE_SIZE};
        add(i, BUF).pBufferInfo = &bi[i];
    }
    add(3, SMP).pImageInfo = &ii[0];
    add(4, SMP).pImageInfo = &ii[1];
    const bool synth = ring == R_SYNTH;   // estimate sets never get the output (no estimate pass uses it)
    if (synth) add(5, IMG).pImageInfo = &ii[2];
    if (blur) {   // the motion-blur scratch (the caller has made sure it exists)
        add(6, BUF).pBufferInfo = &bb[0];
        add(7, BUF).pBufferInfo = &bb[1];
    }
    if (yuv) {   // the CbCr planes
        add(8, SMP).pImageInfo = &ii[3];
        add(9, SMP).pImageInfo = &ii[4];
        if (synth) add(10, IMG).pImageInfo = &ii[5];
    }
    if (synth) {   // the V2 masks of the pair (warp tap rejection)
        add(11, BUF).pBufferInfo = &rj[0];
        add(12, BUF).pBufferInfo = &rj[1];
    }
    if ((this->set.motionVersion == MotionVersion::V2 || (guardView && gbufs)) && vb.buffer) {   // V2's sets (and V3's guard view of them) only (V1's stay as they were)
        add(13, BUF).pBufferInfo = &vb;
        add(14, BUF).pBufferInfo = &qb;
    }
    const MotionGroup& v3g = mo[(int)MotionVersion::V3];
    VkDescriptorBufferInfo v3b[3];
    VkDescriptorBufferInfo gb = {gbufs ? v3g.bufs[V3B_GFIELD].b : VK_NULL_HANDLE, 0, VK_WHOLE_SIZE};
    if (mv3() && !v3g.bufs.empty() && !guardView) {   // V3's sets only
        for (uint32_t i = 0; i < 3; i++) {
            v3b[i] = {v3g.bufs[V3B_PYR + i].b, 0, VK_WHOLE_SIZE};
            add(13 + i, BUF).pBufferInfo = &v3b[i];
        }
        if (gbufs && this->set.v3Guard != V3Guard::Off) add(16, BUF).pBufferInfo = &gb;   // v3_guard: V2's per-pixel field
    }
    if (mlearned() && mb[0].buffer && mb[1].buffer && mb[2].buffer)   // Learned's sets only
        for (uint32_t i = 0; i < 3; i++) add(13 + i, BUF).pBufferInfo = &mb[i];
    UpdateDescriptorSets(info.device, n, wr, 0, nullptr);
    return set;
}

namespace {
const VkPipelineStageFlags CS = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
const VkAccessFlags RW = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
uint32_t divUp(int a, int b) { return (uint32_t)((a + b - 1) / b); }

// Command helpers shared by recordEstimate and recordSynth.
struct Rec {
    Interpolator::Impl& I;
    VkCommandBuffer cb;
    PC pc{};
    Rec(Interpolator::Impl& i, VkCommandBuffer c) : I(i), cb(c) {
        pc.W = (int)I.W; pc.H = (int)I.H; pc.fw = I.fw; pc.fh = I.fh;
        pc.pf = I.fieldPF;
        pc.bw = I.lv[0].bw; pc.bh = I.lv[0].bh;   // synth: finest block grid (scene-cut mean)
        pc.fscale = (int)I.set.flowScale;
        pc.wA = pc.wC = 0.5f;   // t = 0.5
        // runtime settings, copied into this record's push constants
        const Settings& s = I.set;
        pc.lambda = s.searchPenalty; pc.smoothBonus = s.smoothBonus; pc.zeroBonus = s.hudBonus;
        pc.cutSad = s.sceneCutSad; pc.trust0 = s.trustLow; pc.trustScale = 1.0f / (s.trustHigh - s.trustLow); pc.globalFrac = s.globalFallback;
    }
    // Appends a timestamp to the current profile (one is open only after a profiled recordEstimate).
    void stamp(const char* name) {
        if (!I.profiling || !I.qpool || I.qcount == 0 || I.qcount >= Interpolator::Impl::kMaxQ) return;
        I.CmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, I.qpool, I.qcount++);
        I.qnames.push_back(name);
    }
    void barrier(VkPipelineStageFlags src, VkAccessFlags srcA, VkPipelineStageFlags dst, VkAccessFlags dstA) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, srcA, dstA};
        I.CmdPipelineBarrier(cb, src, dst, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    void cbar() { barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, RW); }
    // WAR/WAW against every earlier record's use of our buffers, then zero the first `words` of u[] (claims: also reset V2 Quality's).
    void clearCounters(uint32_t words, VkDescriptorSet set, bool claims = false) {
        barrier(CS, RW, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        I.CmdFillBuffer(cb, I.buf[2], 0, words * 4, 0);
        if (claims) I.v2ResetClaims(cb);
        if (I.mv3() && words > 1) I.v3Reset(cb);   // (V3's estimate: its state G, and the buffers of a first use)
        I.mlUpload(cb);   // (Learned: its weights, once)
        barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, CS, RW);
        I.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, I.layout, 0, 1, &set, 0, nullptr);
    }
    void run(Pipe p, uint32_t gx, uint32_t gy) {
        I.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, I.mg().pipes[p]);
        I.CmdPushConstants(cb, I.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC), &pc);
        I.CmdDispatch(cb, gx, gy, 1);
    }
    // V2 Performance's synthesis passes: their own block (V2PC) over the same range
    void runV2(Pipe p, const V2PC& v, uint32_t gx, uint32_t gy) {
        I.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, I.mg().pipes[p]);
        I.CmdPushConstants(cb, I.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(V2PC), &v);
        I.CmdDispatch(cb, gx, gy, 1);
    }
    // V3's passes: their own block (V3PC) over the same range
    void runV3(Pipe p, const V3PC& v, uint32_t gx, uint32_t gy, uint32_t gz = 1) {
        I.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, I.mg().pipes[p]);
        I.CmdPushConstants(cb, I.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(V3PC), &v);
        I.CmdDispatch(cb, gx, gy, gz);
    }
    V3PC v3pc() const {
        V3PC v{};
        v.W = (int)I.W; v.H = (int)I.H; v.fw = I.fw; v.fh = I.fh;
        v.bw = I.lv[0].bw; v.bh = I.lv[0].bh; v.fscale = (int)I.set.flowScale;
        v.cutThresh = I.set.v3SceneCut; v.globalFrac = I.set.globalFallback;
        v.smear = I.set.v3Smear; v.occlSoft = I.set.v3OcclSoft; v.unsure = I.set.v3Unsure;
        v.cx = 0.5f * (float)I.fw; v.cy = 0.5f * (float)I.fh;
        v.fdOff = (int)(16 * v3Layout(I.W, I.H, I.set.flowScale).B);   // FD in V3B_FLOW (words)
        v.hudB = I.set.hudBonus * kV3HudK;
        for (int i = 0; i < 5; i++) v.hp[i] = kV3HudP[i];
        {   // the c0 ramp widens with the strength above Normal: a stronger setting accepts a looser match of the zero vector as stillness
            const float f = std::max(1.0f, std::pow(I.set.hudBonus, 0.75f));
            v.hp[0] *= f; v.hp[1] *= f;
        }
        return v;
    }
    V2PC v2pc() const {
        V2PC v{};
        v.W = (int)I.W; v.H = (int)I.H; v.fw = I.fw; v.fh = I.fh;
        v.bw = I.lv[0].bw; v.bh = I.lv[0].bh; v.fscale = (int)I.set.flowScale;
        v.cutSad = I.set.sceneCutSad; v.globalFrac = I.set.globalFallback;
        v.sigma2Inv = 1.0f / (kV2CleanSigma * kV2CleanSigma);
        v.trustMid = 0.5f * (I.set.trustLow + I.set.trustHigh);
        return v;
    }
};
}  // namespace

namespace {
// V2's block search of the estimate (v2_est.glsl; MotionVersion::V2 and, with Settings::v3Guard, V3's estimate): V1's block field, faster: no median3
// passes, each consumer takes the 3x3 median of the raw search output itself; medhist's histogram is weighted per parent and scanned in parallel.
// The luma pyramid (P_DOWN) is recorded before it.
void recordV2Search(Rec& r) {
    Interpolator::Impl& I = r.I;
    PC& pc = r.pc;
    static const char* lvNames[] = {"level 0 (finest)", "level 1", "level 2", "level 3", "level 4", "level 5", "level 6", "level 7"};
    const int n = (int)I.lv.size();
    // V1's block field, faster (shaders/v2_est.glsl): no median3 passes, each consumer takes the 3x3 median of the raw search
    // output itself; medhist's histogram is weighted per parent and scanned in parallel
    for (int i = n - 1; i >= 0; i--) {
        const Level& l = I.lv[i];
        pc.w = l.w; pc.h = l.h; pc.bw = l.bw; pc.bh = l.bh;
        pc.offA = l.offA; pc.offC = l.offC;
        if (i == n - 1) {
            pc.medIdx = -1;
            pc.radius = I.set.searchRadiusTop;
        } else {
            const Level& p = I.lv[i + 1];
            pc.pw = p.bw; pc.ph = p.bh; pc.fPar = p.fS;   // the parent's RAW search output
            pc.medIdx = 4 + 2 * i;
            pc.radius = I.set.searchRadius;
            const int sh = medhistShift(I.set, n, i + 1);
            if (sh >= 0) {
                pc.radius = sh;   // (the wide histogram's bin shift; the search below needs the radius again)
                r.run(P_V2_MEDHIST_WIDE, 1, 1);
                pc.radius = I.set.searchRadius;
            } else {
                r.run(P_V2_MEDHIST, 1, 1);
            }
            r.cbar();
        }
        pc.fOut = l.fS;
        // finest level (bits): 1 = V1's scene-cut sum (sub-pixel refinement off: the search's SAD is final), 2 = Performance's
        // block field (V2B_FIELD at bw * bh) and its own scene-cut sum, 4 = with the equiangular sub-pixel fit
        pc.sadAcc = i == 0 ? (!I.set.subpixel ? 1 : 0) | 2 | (I.set.subpixel ? 4 : 0) : 0;
        r.run(pc.radius > kSearchMaxR ? P_V2_SEARCH_WIDE : P_V2_SEARCH, l.bw, l.bh);
        r.cbar();
        r.stamp(lvNames[std::min(i, 7)]);
    }
}
// Settings::v3Guard: what V2 Quality's estimate gives the guard, recorded in a set whose bindings 13 / 14 are V2's buffers: the luma pyramid, the block
// search, the re-pick and clean-up of the finest block field and Quality's per-pixel field pf (V2B_FIELD at vec2 2 bw bh). Not recorded: V2's static mask and
// enclosure (the field does not read them), the coverage splat (the guard does not read it) and everything of the synthesis. The same calls V2's estimate and
// v2Tail make for them, in the same order.
void recordV2Estimate(Rec& r, const std::vector<Level>& lv) {
    Interpolator::Impl& I = r.I;
    PC& pc = r.pc;
    for (size_t i = 1; i < lv.size(); i++) {
        const Level &s = lv[i - 1], &t = lv[i];
        pc.w = t.w; pc.h = t.h; pc.pw = s.w; pc.ph = s.h;
        pc.offA = t.offA; pc.offC = t.offC; pc.srcA = s.offA; pc.srcC = s.offC;
        r.run(P_DOWN, divUp(t.w, 8), divUp(t.h, 8));
        r.cbar();
    }
    r.stamp("v2 pyramid");
    recordV2Search(r);
    Rec t(I, r.cb);
    const Level& l0 = lv[0];
    PC& tp = t.pc;
    tp.w = l0.w; tp.h = l0.h; tp.bw = l0.bw; tp.bh = l0.bh; tp.offA = l0.offA; tp.offC = l0.offC;
    V2PC v = t.v2pc();
    tp.fIn = l0.bw * l0.bh; tp.fOut = l0.fM;
    t.run(P_V2_REPICK, divUp(l0.bw, 2), divUp(l0.bh, 2));   // 2x2 blocks per workgroup
    t.cbar();
    v.fIn = l0.fM;
    t.runV2(P_V2_CLEAN, v, divUp(l0.bw, 8), divUp(l0.bh, 8));
    t.cbar();
    t.stamp("v2 field clean");
    t.runV2(P_V2_QFIELD, v, divUp(I.fw, 8), divUp(I.fh, 8));
    t.cbar();
    t.stamp("v2 pixel field");
}
}  // namespace

// The masks the synth passes of the latest estimate's pair reject warp taps with (CrosshairV2: the entries' own V2 masks). The first
// call also records the one-time zeroing of `rejNone` (its barrier orders the fill before this record's passes).
RejPair Interpolator::Impl::rejectMasks(VkCommandBuffer cb) {
    if (!rejNoneZero) {
        Rec r(*this, cb);
        r.barrier(CS, RW, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        CmdFillBuffer(cb, rejNone.b, 0, kRejNoneBytes, 0);
        r.barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, CS, RW);
        rejNoneZero = true;
    }
    RejPair r{rejNone.b, rejNone.b};
    if (estUi) {
        const UiEntry &ep = uiE[estEnt[0]], &ec = uiE[estEnt[1]];
        if (ep.v2 && ec.v2 && ep.hasBox && ec.hasBox) { r.a = ep.box.b; r.c = ec.box.b; }
    }
    return r;
}

// ---------------------------------------------------------------------------------------------------------
// Static-UI protection: recording
// ---------------------------------------------------------------------------------------------------------
namespace {
// 8x8 groups covering the chroma texels (2x2 pixels) that overlap the pixel span [o, o + n)
uint32_t chromaGroups(int o, int n) { return divUp(((o + n - 1) >> 1) - (o >> 1) + 1, 8); }
void uiDispatch(Interpolator::Impl& I, VkCommandBuffer cb, UiPipe p, VkDescriptorSet set, const UiPC& pc, uint32_t gx, uint32_t gy) {
    I.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, I.uiPipes[p]);
    I.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, I.uiLayout, 0, 1, &set, 0, nullptr);
    I.CmdPushConstants(cb, I.uiLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(UiPC), &pc);
    I.CmdDispatch(cb, gx, gy, 1);
}
}  // namespace

// Moves freshly created clean images UNDEFINED -> GENERAL (one barrier for all of them).
void Interpolator::Impl::uiFlushPendingInit(VkCommandBuffer cb) {
    if (!uiPendingInit.empty()) {
        std::vector<VkImageMemoryBarrier> bars;
        for (VkImage im : uiPendingInit) {
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.srcAccessMask = 0;
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = im;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            bars.push_back(b);
        }
        CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, CS, 0, 0, nullptr, 0, nullptr, (uint32_t)bars.size(), bars.data());
        uiPendingInit.clear();
    }
}

// CrosshairV2 push constants shared by its observe / track / fill passes.
static UiPC v2Pc(const Interpolator::Impl& I, const UiBox& bx, const UiBox& wr) {
    UiPC pc{};
    pc.W = (int)I.W; pc.H = (int)I.H;
    pc.boxX = bx.x; pc.boxY = bx.y; pc.boxW = bx.w; pc.boxH = bx.h;
    pc.maskX = wr.x; pc.maskY = wr.y; pc.maskW = wr.w; pc.maskH = wr.h; pc.maskStride = (wr.w + 3) / 4; pc.maskOff = 4;   // the work rectangle: state layout and mask
    pc.still = (int)I.set.uiStillFrames;
    pc.thr = I.set.uiStillThreshold;
    pc.cthr = I.set.uiContrastThreshold;
    pc.leave = (int)I.set.uiLeaveFrames;
    pc.reentry = (int)I.set.uiReentryFrames;
    return pc;
}

// CrosshairV2 observe of one real frame: the state machine of the box (ui_v2_update) and then its mask (ui_v2_mask). No
// clean copy, no inpaint: the entry only holds the mask. The caller has checked uiActive() and the set ring (1 set).
bool Interpolator::Impl::uiObserveRecordV2(VkCommandBuffer cb, Frame view) {
    const UiBox bx = computeBox(), wr = computeWork(bx);
    const int e = uiPick(view.view);
    UiEntry& E = uiE[e];
    const VkDeviceSize maskWords = 4 + (VkDeviceSize)((wr.w + 3) / 4) * wr.h;
    const VkBuffer oldState = uiBoxState.b;
    if (!growBuf(E.box, maskWords * 4) || !growBuf(uiBoxState, (VkDeviceSize)wr.w * wr.h * 16)) return false;   // 4 state words per pixel of the work rectangle
    if (uiBoxState.b != oldState || !(bx == uiPrevBox)) uiBoxHist = false;
    uiPrevBox = bx;
    E.key = view.view;
    E.seq = ++uiSeq;
    E.bx = bx;
    E.wr = wr;
    E.hasBox = true;
    E.hasWs = false;
    E.v2 = true;
    uiLast = e;

    Rec r(*this, cb);
    // WAR/WAW against every earlier record's use of this entry's mask and of the state buffer (the fill and the patch of an
    // earlier pair read the mask; the previous observe or track wrote the state)
    r.barrier(CS, RW, CS, RW);
    UiPC pc = v2Pc(*this, bx, wr);
    pc.reset = !uiBoxHist;
    VkDescriptorSet boxSet = takeUiSetV2(view, uiBoxState, E.box);
    uiDispatch(*this, cb, U_V2_UPDATE, boxSet, pc, divUp(wr.w, 8), divUp(wr.h, 8));
    r.cbar();
    uiDispatch(*this, cb, U_V2_MASK, boxSet, pc, divUp(pc.maskStride, 8), divUp(wr.h, 8));
    r.cbar();
    r.stamp("ui_detect");
    uiBoxHist = true;
    uiCellHist = false;
    return true;
}

// CrosshairV2 track: the same ui_v2_update pass an observe runs, on the same state, and nothing else (no mask, no entry).
bool Interpolator::Impl::uiTrackRecordV2(VkCommandBuffer cb, Frame view) {
    const UiBox bx = computeBox(), wr = computeWork(bx);
    const VkBuffer oldState = uiBoxState.b;
    if (!growBuf(uiBoxState, (VkDeviceSize)wr.w * wr.h * 16)) return false;
    if (uiBoxState.b != oldState || !(bx == uiPrevBox)) uiBoxHist = false;
    uiPrevBox = bx;
    Rec r(*this, cb);
    r.cbar();
    UiPC pc = v2Pc(*this, bx, wr);
    pc.reset = !uiBoxHist;
    pc.track = 1;
    VkDescriptorSet boxSet = takeUiSetV2(view, uiBoxState, uiBoxState);   // the mask binding is a placeholder: ui_v2_update never touches it
    uiDispatch(*this, cb, U_V2_UPDATE, boxSet, pc, divUp(wr.w, 8), divUp(wr.h, 8));
    r.cbar();
    uiBoxHist = true;
    uiCellHist = false;
    return true;
}

// CrosshairV2, estimate: the flow-resolution luma of one frame of the pair (already written by luma.comp at lumaOff in buf[0])
// gets the cells that hold a UI pixel re-averaged over their not-UI pixels (ui_v2_lumafill.comp). The caller restores its
// own descriptor set afterwards.
void Interpolator::Impl::uiRecordLumaFill(VkCommandBuffer cb, Frame frame, const UiEntry& E, int lumaOff) {
    if (!E.hasBox) return;
    UiPC pc = v2Pc(*this, E.bx, E.wr);   // the mask rectangle = the work rectangle the entry was observed with
    pc.cell = (int)set.flowScale; pc.cw = fw; pc.ch = fh;
    pc.reach = 8;
    pc.lumaOff = lumaOff;
    UiBuf luma; luma.b = buf[0]; luma.size = bufSize[0];
    VkDescriptorSet fillSet = takeUiSetV2(frame, luma, E.box);
    const int S = (int)set.flowScale;
    const int cells0 = E.wr.x / S, cells1 = (E.wr.x + E.wr.w - 1) / S, rows0 = E.wr.y / S, rows1 = (E.wr.y + E.wr.h - 1) / S;
    uiDispatch(*this, cb, U_V2_FILL, fillSet, pc, divUp(cells1 - cells0 + 1, 8), divUp(rows1 - rows0 + 1, 8));
}

// Records the observe of one real frame into the entry it picks: clean copy, stillness, masks, inpaint.
// The caller has checked uiActive() and the set ring (uiObserveSets()).
bool Interpolator::Impl::uiObserveRecord(VkCommandBuffer cb, Frame view) {
    if (v2()) return uiObserveRecordV2(cb, view);
    const bool ws = set.uiProtection == UiProtection::WholeScreen;
    const UiBox bx = computeBox(), wr = computeWork(bx);   // wr = the box plus its margin (none for WholeScreen): the mask, inpaint and patch rectangle
    const int e = uiPick(view.view);
    UiEntry& E = uiE[e];
    // buffers first (they may fail and must not leave a half-observed entry): the box mask of this entry (work
    // rectangle), and the box state (box only: detection never looks outside it); growing the state buffer or moving
    // the box restarts the box counters
    const VkDeviceSize maskWords = 4 + (VkDeviceSize)((wr.w + 3) / 4) * wr.h;
    const VkBuffer oldState = uiBoxState.b;
    if (!growBuf(E.box, maskWords * 4) || !growBuf(uiBoxState, (VkDeviceSize)bx.w * bx.h * 4)) return false;
    if (uiBoxState.b != oldState || !(bx == uiPrevBox)) uiBoxHist = false;
    uiPrevBox = bx;

    uiFlushPendingInit(cb);

    E.key = view.view;
    E.seq = ++uiSeq;
    E.bx = bx;
    E.wr = wr;
    E.hasBox = true;
    E.hasWs = ws;
    E.v2 = false;
    uiLast = e;

    Rec r(*this, cb);
    // WAR/WAW against every earlier record's use of this entry's buffers and clean image, then zero the header
    // (+ tile flags) of the entry's mask buffer, which the passes accumulate into
    r.barrier(CS, RW, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    if (ws) CmdFillBuffer(cb, E.ws.b, 0, (VkDeviceSize)(4 + tilesX * tilesY) * 4, 0);
    else CmdFillBuffer(cb, E.box.b, 0, 16, 0);
    r.barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, CS, RW);

    UiPC pc{};
    pc.W = (int)W; pc.H = (int)H;
    pc.boxX = bx.x; pc.boxY = bx.y; pc.boxW = bx.w; pc.boxH = bx.h;
    pc.cell = kUiCell; pc.cw = cw; pc.ch = ch;
    pc.still = (int)set.uiStillFrames;
    pc.thr = set.uiStillThreshold;
    pc.reach = (int)set.uiInpaintReach;
    pc.tilesX = tilesX; pc.tilesY = tilesY;
    pc.stillLimit = (int)std::floor((double)set.uiMaxStillFraction * cw * ch);
    const int fullStride = ((int)W + 3) / 4, boxStride = (wr.w + 3) / 4;

    VkDescriptorSet boxSet = takeUiSet(view, E.frame(), uiBoxState, E.box);
    VkDescriptorSet wsSet = ws ? takeUiSet(view, E.frame(), uiCellState, E.ws) : VK_NULL_HANDLE;

    // 1. clean copy (+ whole screen: the still cells, tile flags)
    pc.wholeScreen = ws;
    pc.reset = !uiCellHist;
    uiDispatch(*this, cb, U_DETECT, ws ? wsSet : boxSet, pc, divUp(cw, 8), divUp(ch, 8));
    // 2. box: counters, then (next barrier) its mask
    pc.reset = !uiBoxHist;
    uiDispatch(*this, cb, U_BOX_UPDATE, boxSet, pc, divUp(bx.w, 8), divUp(bx.h, 8));
    r.cbar();
    pc.maskX = wr.x; pc.maskY = wr.y; pc.maskW = wr.w; pc.maskH = wr.h; pc.maskStride = boxStride; pc.maskOff = 4; pc.gate = 0;
    uiDispatch(*this, cb, U_BOX_MASK, boxSet, pc, divUp(boxStride, 8), divUp(wr.h, 8));
    // 3. whole screen: the full-resolution mask of the flagged tiles
    if (ws) {
        pc.maskX = pc.maskY = 0; pc.maskW = (int)W; pc.maskH = (int)H; pc.maskStride = fullStride;
        pc.maskOff = 4 + tilesX * tilesY; pc.gate = 1;
        uiDispatch(*this, cb, U_MASK, wsSet, pc, tilesX, tilesY * 4);
    }
    r.cbar();
    r.stamp("ui_detect");
    // 4. inpaint the clean copy (whole screen first: the box, last, owns its pixels)
    if (ws) {
        uiDispatch(*this, cb, U_INPAINT, wsSet, pc, divUp((int)W, kUiInpTile), divUp((int)H, kUiInpTile));
        if (yuv) uiDispatch(*this, cb, U_INPAINT_C, wsSet, pc, divUp(((int)W + 1) / 2, 8), divUp(((int)H + 1) / 2, 8));   // the other plane: no hazard
    }
    r.cbar();
    pc.maskX = wr.x; pc.maskY = wr.y; pc.maskW = wr.w; pc.maskH = wr.h; pc.maskStride = boxStride; pc.maskOff = 4; pc.gate = 0;
    pc.open = boxMargin(set) > 0;   // the rectangle is the box plus a margin: beyond it is real, unmasked background (a valid walk end)
    uiDispatch(*this, cb, U_INPAINT, boxSet, pc, divUp(wr.w, kUiInpTile), divUp(wr.h, kUiInpTile));
    if (yuv) uiDispatch(*this, cb, U_INPAINT_C, boxSet, pc, chromaGroups(wr.x, wr.w), chromaGroups(wr.y, wr.h));
    pc.open = 0;
    r.cbar();
    r.stamp("ui_inpaint");

    uiBoxHist = true;
    uiCellHist = ws;
    return true;
}

// Records the track of one real frame (recordTrack): ONLY the stillness state is advanced. The same two passes as an
// observe runs for it, on the same state buffers with the same `reset` handling: ui_box_update (box counters) and, for
// WholeScreen, ui_detect with pc.track = 1 (cell counters; no clean copy, no tile flags, no mask). Nothing else of an
// observe runs (mask, inpaint, the copy), no entry is picked and no LRU slot is touched. The shaders still declare the
// observe's bindings, so the set gets harmless placeholders: the first active entry's clean image (never written with
// track = 1) and the box state as the "mask" (never touched with track = 1).
bool Interpolator::Impl::uiTrackRecord(VkCommandBuffer cb, Frame view) {
    if (v2()) return uiTrackRecordV2(cb, view);
    const bool ws = set.uiProtection == UiProtection::WholeScreen;
    const UiBox bx = computeBox();
    const UiEntry* ph = nullptr;
    for (const UiEntry& e : uiE) if (e.active && e.img) { ph = &e; break; }
    if (!ph) return false;
    const VkBuffer oldState = uiBoxState.b;
    if (!growBuf(uiBoxState, (VkDeviceSize)bx.w * bx.h * 4)) return false;
    if (uiBoxState.b != oldState || !(bx == uiPrevBox)) uiBoxHist = false;
    uiPrevBox = bx;
    uiFlushPendingInit(cb);

    Rec r(*this, cb);
    r.cbar();   // earlier records read/wrote the state buffers (mask build, observe, ...)
    UiPC pc{};
    pc.W = (int)W; pc.H = (int)H;
    pc.boxX = bx.x; pc.boxY = bx.y; pc.boxW = bx.w; pc.boxH = bx.h;
    pc.cell = kUiCell; pc.cw = cw; pc.ch = ch;
    pc.still = (int)set.uiStillFrames;
    pc.thr = set.uiStillThreshold;
    pc.tilesX = tilesX; pc.tilesY = tilesY;
    pc.track = 1;
    if (ws) {
        VkDescriptorSet wsSet = takeUiSet(view, ph->frame(), uiCellState, uiBoxState);
        pc.wholeScreen = 1;
        pc.reset = !uiCellHist;
        uiDispatch(*this, cb, U_DETECT, wsSet, pc, divUp(cw, 8), divUp(ch, 8));
    }
    VkDescriptorSet boxSet = takeUiSet(view, ph->frame(), uiBoxState, uiBoxState);
    pc.wholeScreen = 0;
    pc.reset = !uiBoxHist;
    uiDispatch(*this, cb, U_BOX_UPDATE, boxSet, pc, divUp(bx.w, 8), divUp(bx.h, 8));
    r.cbar();
    uiBoxHist = true;
    uiCellHist = ws;
    return true;
}

// After a generated frame is complete: paste the real pixels of the frame the synth leads up to (curr) over it.
void Interpolator::Impl::uiRecordPatch(VkCommandBuffer cb, Frame curr, Frame out, const UiEntry& E) {
    Rec r(*this, cb);
    UiPC pc{};
    pc.W = (int)W; pc.H = (int)H;
    pc.tilesX = tilesX; pc.tilesY = tilesY;
    pc.stillLimit = (int)std::floor((double)set.uiMaxStillFraction * cw * ch);
    if (E.hasWs) {
        pc.maskW = (int)W; pc.maskH = (int)H; pc.maskStride = ((int)W + 3) / 4; pc.maskOff = 4 + tilesX * tilesY; pc.gate = 1;
        VkDescriptorSet wsSet = takeUiSet(curr, out, uiCellState, E.ws);
        uiDispatch(*this, cb, U_PATCH, wsSet, pc, tilesX, tilesY * 4);
        if (yuv) uiDispatch(*this, cb, U_PATCH_C, wsSet, pc, divUp(((int)W + 1) / 2, 8), divUp(((int)H + 1) / 2, 8));
        r.barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, RW);
    }
    if (E.hasBox) {
        pc.maskX = E.wr.x; pc.maskY = E.wr.y; pc.maskW = E.wr.w; pc.maskH = E.wr.h;   // the work rectangle: the box plus its margin
        pc.maskStride = (E.wr.w + 3) / 4; pc.maskOff = 4; pc.gate = 0;
        pc.binary = E.v2;   // CrosshairV2: only mask bytes of 255 (unchanged in this pair) are pasted
        VkDescriptorSet boxSet = takeUiSet(curr, out, uiBoxState, E.box);
        uiDispatch(*this, cb, U_PATCH, boxSet, pc, divUp(pc.maskStride, 8), divUp(E.wr.h, 8));
        if (yuv) uiDispatch(*this, cb, U_PATCH_C, boxSet, pc, chromaGroups(E.wr.x, E.wr.w), chromaGroups(E.wr.y, E.wr.h));
    }
}

// ---------------------------------------------------------------------------------------------------------
// Settings::debugView (task 136): the last pass over a finished generated frame
// ---------------------------------------------------------------------------------------------------------
// Creates the pipelines the selected view needs (the motion tint of the selected motion version, or the UI pass), the first time they are needed; they stay until destroy().
// Nothing for Off and for the YUV formats (no view is drawn). False: this build lacks the variant, or the pipeline creation failed.
bool Interpolator::Impl::dbgEnsure() {
    if (!dbgOn()) return true;
    DbgPipe p = D_UI;
    Spv sp = cs.dbgUi;
    VkPipelineLayout lay = uiLayout;
    if (set.debugView == DebugView::Motion) {
        lay = layout;
        if (mv2()) { p = D_MOTION_V2; sp = cs.dbgMotionV2; }
        else if (mv3()) { p = D_MOTION_V3; sp = cs.dbgMotionV3; }
        else { p = D_MOTION; sp = cs.dbgMotion; }   // V1 and Learned: the field in buf[1]
    }
    if (dbgPipes[p]) return true;
    return sp.code && makePipe(sp, lay, dbgPipes[p], nullptr);
}

namespace {
void uiDispatchP(Interpolator::Impl& I, VkCommandBuffer cb, VkPipeline pipe, VkDescriptorSet set, const UiPC& pc, uint32_t gx, uint32_t gy) {
    I.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    I.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, I.uiLayout, 0, 1, &set, 0, nullptr);
    I.CmdPushConstants(cb, I.uiLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(UiPC), &pc);
    I.CmdDispatch(cb, gx, gy, 1);
}
}  // namespace

void Interpolator::Impl::debugDraw(VkCommandBuffer cb, VkDescriptorSet synth, Frame curr, Frame out, int pf, int nInt) {
    if (!dbgOn() || nInt <= 0) return;
    Rec r(*this, cb);
    if (set.debugView == DebugView::Motion) {
        const VkPipeline pipe = dbgPipes[mv2() ? D_MOTION_V2 : mv3() ? D_MOTION_V3 : D_MOTION];
        if (!pipe) return;
        DbgPC d{};
        d.W = (int)W; d.H = (int)H; d.fw = fw; d.fh = fh; d.fscale = (int)set.flowScale;
        d.pf = pf; d.bw = lv[0].bw; d.bh = lv[0].bh; d.quality = set.v2Mode == V2Mode::Quality;
        const V3Layout V = v3Layout(W, H, set.flowScale);
        d.fdOff = (int)(16 * V.B); d.gOff = (int)V.gOff(); d.nInt = nInt;
        d.m0 = (float)set.flowScale; d.m1 = 16.0f; d.maxA = 0.6f;
        d.cscale = fmt == Format::Rgba8 ? 0.9f : 0.6f;   // HDR formats: PQ code values stay moderate
        r.barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, RW);   // the synthesis' / the patch's writes of `out`
        CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &synth, 0, nullptr);   // (the patch left a UI set bound)
        CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        CmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DbgPC), &d);
        CmdDispatch(cb, divUp((int)W, 8), divUp((int)H, 8), 1);
        return;
    }
    // UiMask / UiClean: only where the pair was observed under a protection mode (estUi), as the patch
    if (!estUi || !dbgPipes[D_UI]) return;
    const UiEntry& E = uiE[estEnt[1]];
    const bool clean = set.debugView == DebugView::UiClean && !E.v2;
    const bool fill = set.debugView == DebugView::UiClean && E.v2;   // CrosshairV2 keeps no clean copy: its pixels are filled from their surroundings in `out`
    const Frame src = clean ? E.frame() : curr;
    UiPC pc{};
    pc.W = (int)W; pc.H = (int)H;
    pc.tilesX = tilesX; pc.tilesY = tilesY;
    pc.stillLimit = (int)std::floor((double)set.uiMaxStillFraction * cw * ch);
    r.barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, RW);
    if (E.hasWs) {   // the same masks, gate and suppression as the patch
        pc.maskW = (int)W; pc.maskH = (int)H; pc.maskStride = ((int)W + 3) / 4; pc.maskOff = 4 + tilesX * tilesY; pc.gate = 1;
        pc.dbg = clean ? 2 : 1;
        uiDispatchP(*this, cb, dbgPipes[D_UI], takeUiSet(src, out, uiCellState, E.ws), pc, tilesX, tilesY * 4);
        if (clean) return;   // the whole frame is the clean copy; the box lies inside it
        r.barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, RW);
    }
    if (E.hasBox) {
        pc.boxX = E.bx.x; pc.boxY = E.bx.y; pc.boxW = E.bx.w; pc.boxH = E.bx.h;
        pc.maskX = E.wr.x; pc.maskY = E.wr.y; pc.maskW = E.wr.w; pc.maskH = E.wr.h;
        pc.maskStride = (E.wr.w + 3) / 4; pc.maskOff = 4; pc.gate = 0;
        pc.binary = E.v2;
        pc.dbg = clean ? 2 : fill ? 3 : (1 | 16);
        pc.reach = (int)set.uiInpaintReach;
        uiDispatchP(*this, cb, dbgPipes[D_UI], takeUiSet(src, out, uiBoxState, E.box), pc, divUp(pc.maskStride, 8), divUp(E.wr.h, 8));
    }
}

bool Interpolator::recordObserve(VkCommandBuffer cb, Frame frame) {
    if (!m || !m->W) return false;
    Impl& I = *m;
    if (!I.framesOk(frame)) return false;
    if (I.set.uiProtection == UiProtection::Off) return true;
    // Every refusal from here on forgets the entry keyed by this view (task 112): it holds what the view showed before, and
    // an estimate that found it by view would run on that old content (the next estimate of a pair with this view now
    // runs unprotected, a synth of a pair estimated with the old entry fails its sequence check).
    if (!I.uiActive()) { I.uiForget(frame.view); return false; }
    I.uiExplicit = true;
    if (I.uiFrameSets + I.uiObserveSets() > kUiSets) { I.uiForget(frame.view); return false; }
    if (I.profiling && I.qpool && !I.profPending) {   // an observe opens a profile that the next recordEstimate continues
        I.qcount = 0;
        I.qnames.clear();
        I.CmdResetQueryPool(cb, I.qpool, 0, Impl::kMaxQ);
        I.CmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, I.qpool, I.qcount++);
        I.qnames.push_back("start");
    }
    I.profPending = I.profiling && I.qpool;
    if (!I.uiObserveRecord(cb, frame)) { I.uiForget(frame.view); return false; }
    return true;
}

bool Interpolator::recordTrack(VkCommandBuffer cb, Frame frame) {
    if (!m || !m->W) return false;
    Impl& I = *m;
    if (!I.framesOk(frame)) return false;
    if (I.set.uiProtection == UiProtection::Off) return true;
    // A tracked view holds a new frame that has no clean copy: an entry still keyed by it (an earlier observe of the same
    // view) is forgotten whatever the outcome (task 112; a pair with this view then runs unprotected, as documented)
    I.uiForget(frame.view);
    if (!I.uiActive()) return false;
    if (I.uiFrameSets + I.uiObserveSets() > kUiSets) return false;
    return I.uiTrackRecord(cb, frame);
}

bool Interpolator::recordEstimate(VkCommandBuffer cb, Frame prev, Frame curr) {
    if (!m || !m->W) return false;
    Impl& I = *m;
    if (!I.framesOk(prev) || !I.framesOk(curr)) return false;
    // The next set in the ring may still be referenced by an earlier record of this submission.
    // Settings::v3Guard: V2's estimate passes run in a second set (V2's buffers at 13 / 14), the estimate takes two
    const bool guard = I.mv3() && I.set.v3Guard != V3Guard::Off && !I.v3inj && I.mo[(int)MotionVersion::V3].bufs.size() == V3B_GCOUNT && I.mo[(int)MotionVersion::V3].pipes[P_V3_GUARD] && I.mo[(int)MotionVersion::V3].pipes[P_V3_ANCHOR];
    assert(I.frameSets[Impl::R_EST] + (guard ? 1u : 0u) < kSetRing && "framegen: more than kSetRing recordEstimate() since beginFrame() (two sets each with Settings::v3Guard)");
    if (I.frameSets[Impl::R_EST] + (guard ? 1u : 0u) >= kSetRing) return false;
    // Static-UI protection: the estimate runs on the clean copies of the pair. A host that never calls recordObserve
    // gets the implicit observes here (prev if unseen, curr unless it is the newest observed view).
    // With explicit observing, a pair whose views the library has not both observed (protection was just switched on,
    // or an observe was skipped / the entry evicted) runs unprotected, like Off: no frame is lost.
    int needObs = 0;
    bool ui = I.uiActive();
    if (ui) {
        if (!I.uiExplicit) {
            needObs += I.uiFind(prev.view) < 0;
            needObs += I.uiLast < 0 || I.uiE[I.uiLast].key != curr.view;
        } else if (I.uiFind(prev.view) < 0 || I.uiFind(curr.view) < 0) {
            ui = false;
        }
        if (ui && I.uiFrameSets + needObs * I.uiObserveSets() + (I.v2Fill() ? 2 : 0) > kUiSets) return false;   // V2 with the fill: + its two luma fills
    }

    if (!I.profPending) {
        I.qcount = 0;
        I.qnames.clear();
        if (I.profiling && I.qpool) {   // a new profile
            I.CmdResetQueryPool(cb, I.qpool, 0, Impl::kMaxQ);
            I.CmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, I.qpool, I.qcount++);
            I.qnames.push_back("start");
        }
    }
    I.profPending = false;   // an open profile that an observe started is continued by this estimate

    Frame estP = prev, estC = curr;
    int ep = -1, ec = -1;
    if (ui) {
        if (!I.uiExplicit) {
            if (I.uiFind(prev.view) < 0 && !I.uiObserveRecord(cb, prev)) return false;
            if ((I.uiLast < 0 || I.uiE[I.uiLast].key != curr.view) && !I.uiObserveRecord(cb, curr)) return false;
        }
        ep = I.uiFind(prev.view);
        ec = I.uiFind(curr.view);
        if (ep < 0 || ec < 0 || ep == ec) return false;   // (implicit: the second observe evicted the first: cannot happen with at least kUiFrames >= 3 active entries)
        if (!I.v2()) {   // V2 estimates on the host's own frames (its masked pixels are filled in the luma only)
            estP = I.uiE[ep].frame();
            estC = I.uiE[ec].frame();
        }
    }
    VkDescriptorSet set = I.takeSet(Impl::R_EST, estP, estC, Frame());
    I.haveEst = true;
    I.estPrev = prev.view;
    I.estCurr = curr.view;
    I.estUi = ui;
    I.estClean = ui && !I.v2();
    if (ui) {
        I.estEnt[0] = ep; I.estEnt[1] = ec;
        I.estSeq[0] = I.uiE[ep].seq; I.estSeq[1] = I.uiE[ec].seq;
    }
    Rec r(I, cb);
    if (I.mv2()) I.v2Clean = I.v2Cov = false;   // a new estimate: each mode's tail is recorded again
    r.clearCounters(64, set, I.v2NeedCov());   // untrusted count, SAD sum, per-level medians (+ V2 Quality's claims)
    PC& pc = r.pc;

    // 1. luma + 2x downsample
    const Level& l0 = I.lv[0];
    pc.w = l0.w; pc.h = l0.h; pc.offA = l0.offA; pc.offC = l0.offC;
    int lt = I.set.flowScale == 4 ? 4 : 8;   // flow pixels per workgroup side (luma.comp main4 for fscale 4)
    r.run(P_LUMA, divUp(l0.w, lt), divUp(l0.h, lt));
    r.cbar();
    r.stamp("luma");
    if (I.mv3()) {
        // V3 (gpu/V3.md E2): V2's static-structure mask (v2_static.comp, unchanged) into V3B_AUX's S / s8 words, from the real frames' luma, before
        // CrosshairV2's estimate fill takes the UI out of it (so a static crosshair is seen as static). Same ordering as V2's block below.
        V3PC sv = r.v3pc();
        sv.fIn = I.set.v2StaticMask ? 0 : 1;   // v2_static.comp: 1 = write an empty mask (Settings::v2StaticMask off)
        r.runV3(P_V2_STATIC, sv, divUp(I.fw, 16), divUp(I.fh, 16));
        r.stamp("static");
        if (ui && I.v2Fill()) r.cbar();
    }
    if (I.mv2()) {
        // V2: its static-structure mask (v2_static.comp) from the real frames' luma, before CrosshairV2's estimate fill takes the UI
        // out of it (the mask sees the crosshair as the static overlay it is); reads only the luma, writes its own slot of V2B_QUAL:
        // no barrier against the pyramid, one against the fill (which rewrites the luma). (Intended: syncval with its shader-accesses
        // heuristic reports a WRITE_AFTER_READ on binding 0 here, the pyramid writing levels 1.. of the buffer this pass reads level 0
        // of: disjoint ranges.)
        V2PC sv = r.v2pc();
        sv.fIn = I.set.v2StaticMask ? 0 : 1;   // v2_static.comp: 1 = write an empty mask (Settings::v2StaticMask off)
        r.runV2(P_V2_STATIC, sv, divUp(I.fw, 16), divUp(I.fh, 16));
        if (ui && I.v2Fill()) r.cbar();
    }
    if (ui && I.v2Fill()) {   // CrosshairV2, uiEstimateFill: the UI pixels of both frames are missing for the estimate (luma only, never the colour passes)
        I.uiRecordLumaFill(cb, prev, I.uiE[ep], l0.offA);
        I.uiRecordLumaFill(cb, curr, I.uiE[ec], l0.offC);
        r.cbar();
        I.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, I.layout, 0, 1, &set, 0, nullptr);   // the fills bound the UI set
        r.stamp("ui_fill");
    }
    if (I.mv3()) {
        // MotionVersion::V3 (gpu/V3.md section 1, E4..E8; build step E1): planes, histograms, pyramid, the block search (top level exhaustive, two
        // iterations on levels n-2 .. 1, one with Lucas-Kanade on level 0) and the dense field. The state G was zeroed by clearCounters
        // (v3Reset). Later steps add, in order: the static mask (E2 of the page), the wide search and the stats (E2), occlusion, fill, enclosure,
        // decision (Y1, Y2). With v3inj set (recordDebugV3Inject) the search and the pyramid above level 0 are replaced by the caller's flows:
        // FD / FF are copied in and v3_dense runs in inject mode (it writes the injected flows' matching costs).
        const MotionGroup& g = I.mo[(int)MotionVersion::V3];
        const V3Layout V = v3Layout(I.W, I.H, I.set.flowScale);
        const bool inj = I.v3inj != nullptr;
        const bool v35 = I.set.v3Motion != 0 && I.mg().pipes[P_V35_REG];   // Settings::v3Motion = 1 (V3.5)
        if (guard) {   // Settings::v3Guard: V2 Quality's per-pixel field of this pair (V2's own passes, in a set whose bindings 13 / 14 are V2's buffers)
            const VkDescriptorSet gset = I.takeSet(Impl::R_EST, estP, estC, Frame(), false, RejPair(), true);
            I.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, I.layout, 0, 1, &gset, 0, nullptr);
            recordV2Estimate(r, I.lv);
            I.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, I.layout, 0, 1, &set, 0, nullptr);
            // V2's field re-anchored to V3's grids (into the FF region of V3B_FLOW, free until v3_fill)
            r.runV3(P_V3_ANCHOR, r.v3pc(), divUp(I.fw, 8), divUp(I.fh, 8), 2);
            r.cbar();
            r.stamp("anchor");
        }
        const int hookJ = I.set.v3ColourWindow && !v35 ? 1 : 0;   // hook (j): the search and dense passes of the colour-window build (P_V3_*_J); mode bit 0 stays set for them (they do not read it)
        V3PC v = r.v3pc();
        r.runV3(P_V3_PLANES, v, divUp(V.fw, I.set.flowScale == 4 ? 4 : 8), divUp(V.fh, I.set.flowScale == 4 ? 4 : 8));
        r.cbar();
        r.stamp("planes");
        r.runV3(P_V3_HIST, v, divUp(V.fw, 16), divUp(V.fh, 16));
        r.cbar();
        r.stamp("hist");
        for (int l = 0; l < (inj ? 1 : V.n); l++) {
            V3PC pv = v;
            pv.lvl = l; pv.mode = l == 0 ? 1 : 0; pv.lw = V.w[l]; pv.lh = V.h[l];
            r.runV3(P_V3_PYR, pv, divUp(V.w[l], 16), divUp(V.h[l], 16), 2);
            r.cbar();
        }
        r.stamp("pyramid");
        if (!inj) {
            static const char* const lvNames[] = {"level 0 (finest)", "level 1", "level 2", "level 3", "level 4", "level 5", "level 6", "level 7", "level 8", "level 9",
                                                  "level 10", "level 11", "level 12", "level 13", "level 14", "level 15"};
            static const char* const topName = "level (top)";
            int v35Fin[16] = {};   // V3.5: the slot that holds each level's final block records (the search's own: 1 on the levels in between, else 0)
            for (int l = V.n - 1; l >= 0; l--) {
                V3PC sv = v;
                sv.lvl = l; sv.lw = V.w[l]; sv.lh = V.h[l]; sv.gw = V.gw[l]; sv.gh = V.gh[l];
                sv.mode = hookJ;   // search: bit 0 hook (j)
                const int its = (l >= 1 && l < V.n - 1) ? 2 : 1;
                for (int it = 0; it < its; it++) {
                    sv.iter = it;
                    V3PC sq = sv;
                    if (v35 && l + 1 < V.n) sq.mode = (l + 1 < V.n - 1 && v35Fin[l + 1] == 0) ? 2 : 0;   // the parent level's regularised slot (V3.5's search variant)
                    r.runV3(l == V.n - 1 ? (hookJ ? P_V3_SEARCH_TOP_J : P_V3_SEARCH_TOP) : v35 ? P_V35_SEARCH : (hookJ ? P_V3_SEARCH_J : P_V3_SEARCH), sq, V.gw[l], V.gh[l], 2);   // (the top level: exhaustive, its own pipeline)
                    r.cbar();
                }
                r.stamp(l == V.n - 1 && V.n > 1 ? topName : lvNames[l]);
                if (V.n >= 4 && l == V.n - 2) {   // the wide search on the blocks where level L_wide's result is unsure (section 3.6)
                    r.runV3(P_V3_WIDE, sv, V.gw[l], V.gh[l], 2);
                    r.cbar();
                    r.stamp("wide");
                }
                if (v35) {
                    // V3.5: from level LH down to 1 the camera / dominant-motion fit is redone on this level's (not yet regularised) blocks, texture-weighted (v35_stats): the
                    // coarse level's integer vectors (8 flow px apart at LH) give only a crude model, the finer levels' a precise one; the model is what the search below takes as candidates
                    const int fin = (l >= 1 && l < V.n - 1) ? 1 : 0;
                    if (V.n >= 3 && l >= 1 && l <= std::max(1, V.n - 3)) {
                        r.runV3(P_V35_STATS, sv, 1, 1, 1);
                        r.cbar();
                        V3PC gv = sv;
                        gv.iter = fin;
                        r.runV3(P_V35_RGN, gv, 12, 1, 1);   // the regional camera models of this level (round 2)
                        r.cbar();
                        r.stamp("stats");
                    }
                    // the level's block field regularised, one Jacobi step per dispatch: level 0 (slot 0, V3's result -> slot 1, the dense pass offers both) and every level but the one
                    // below the top make one step (the level's final slot -> the other; the search below is told which slot holds the result, v35Fin), the level below the top two
                    // (final slot -> the other -> back, the second step reads the blocks' static data the first one wrote); the top level (a handful of blocks) none
                    const int nReg = (l + 1 == V.n && V.n > 3) ? 0 : l == 0 ? 1 : (V.n <= 3 || l + 2 == V.n) ? 2 : 1;
                    int cur = fin;
                    for (int k = 0; k < nReg; k++) {
                        V3PC rv = sv;
                        rv.mode = l == 0 ? (1 << 20) : nReg >= 2 ? (k == 0 ? (1 << 21) : (1 << 22)) : 0;
                        rv.iter = cur;
                        cur = 1 - cur;
                        r.runV3(P_V35_REG, rv, divUp(V.gw[l], 8), divUp(V.gh[l], 8), 2);
                        r.cbar();
                    }
                    v35Fin[l] = cur;
                    r.stamp("reg");
                } else if (V.n >= 3 && l == std::max(1, V.n - 3)) {   // the dominant motions and the camera from level LH (section 3.7)
                    r.runV3(P_V3_STATS, sv, 1, 1, 1);
                    r.cbar();
                    r.stamp("stats");
                }
            }
        } else {
            const VkDeviceSize n8 = (VkDeviceSize)I.fw * I.fh * 8;
            const VkDeviceSize fd = 64 * V.B, ff = fd + 2 * n8;   // V3B_FLOW: BLK, FD[2], FF[2], TF
            r.barrier(CS, RW, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            // (FD and FF: until step Y1 there is no fill pass, FF = FD)
            VkBufferCopy c0[2] = {{I.v3inj->f01, fd, n8}, {I.v3inj->f01, ff, n8}}, c1[2] = {{I.v3inj->f10, fd + n8, n8}, {I.v3inj->f10, ff + n8, n8}};
            I.CmdCopyBuffer(cb, I.v3inj->buf, g.bufs[V3B_FLOW].b, 2, c0);
            I.CmdCopyBuffer(cb, I.v3inj->buf, g.bufs[V3B_FLOW].b, 2, c1);
            r.barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, CS, RW);
            I.v3Dirty = true;
            r.stamp("inject");
        }
        V3PC dv = v;
        dv.mode = hookJ | (inj ? 4 : 0);   // dense: bit 0 hook (j), bit 1 also write FF (not since v3_fill writes it, step Y1), bit 2 inject
        r.runV3(v35 ? P_V35_DENSE : hookJ ? P_V3_DENSE_J : P_V3_DENSE, dv, divUp(V.fw + 4, 8), divUp(V.fh + 4, 8), 2);
        r.cbar();
        r.stamp("dense");
        if (guard) {   // Settings::v3Guard (gpu/V3.md "As built (motion guard, task 104)"): V2's vector where it explains the frames better (Guard) or everywhere (V2Motion)
            V3PC gv = v;
            const int gj = hookJ ? 2 : 0;
#ifdef FG_GUARD_NEVER
            const int gn = 8;   // (the control build: the pass runs, the rule is false: nothing is replaced)
#else
            const int gn = 0;
#endif
            if (I.set.v3Guard == V3Guard::V2Motion) {
                gv.mode = 1 | gj | gn;
                r.runV3(P_V3_GUARD, gv, divUp(V.fw, 16), divUp(V.fh, 16), 2);
                r.cbar();
            } else {
                gv.mode = gj | gn;
                r.runV3(P_V3_GUARD, gv, divUp(V.fw, 16), divUp(V.fh, 16), 2);   // the mark (reads FD only)
                r.cbar();
                gv.mode = 16 | gj | gn;
                r.runV3(P_V3_GUARD, gv, divUp(V.fw, 16), divUp(V.fh, 16), 2);   // the partner marks
                r.cbar();
                gv.mode = 4 | gj | gn;
                r.runV3(P_V3_GUARD, gv, divUp(V.fw, 16), divUp(V.fh, 16), 2);   // the replacement
                r.cbar();
            }
            r.stamp("guard");
        }
        // Y1: forward-backward consistency, the garbage measure and the MAP word (e8, amb8, g8), then the vector fill into FF (section 3.9, 3.10)
        V3PC ov = v;
        ov.mode = inj ? 4 : 0;   // occl: bit 2 = inject (the block records are stale: amb = 0)
        r.runV3(P_V3_OCCL, ov, divUp(V.fw, 16), divUp(V.fh, 16), 2);
        r.cbar();
        r.stamp("occlusion");
        r.runV3(P_V3_FILL, ov, divUp(V.fw, 16), divUp(V.fh, 16), 2);   // fill: bit 2 = inject (injected flows are the exact ones: nothing to fill, tuning D5)
        r.cbar();
        r.stamp("fill");
        // Y2: the static mask's enclosure E / near bits C (section 3.13, from v2_static's cell maxima and the filled F01), and the pair's decisions
        // (section 3.12: the histogram scene cut, the soft fallback mix, the counters); independent of each other (they write disjoint words)
        r.runV3(P_V3_ENCLOSE, v, divUp(divUp(V.fw, 8) * divUp(V.fh, 8), 64), 1);
        r.stamp("enclose");
        r.runV3(P_V3_DECIDE, v, 1, 1);
        r.cbar();
        r.stamp("decide");
        return true;
    }
    if (I.mlearned()) {   // the learned estimator instead of V1's pyramid and search (V1's luma above, V1's synthesis after it)
        I.mlEstimate(cb);
        return true;
    }
    // 2. pyramid
    for (size_t i = 1; i < I.lv.size(); i++) {
        const Level &s = I.lv[i - 1], &t = I.lv[i];
        pc.w = t.w; pc.h = t.h; pc.pw = s.w; pc.ph = s.h;
        pc.offA = t.offA; pc.offC = t.offC; pc.srcA = s.offA; pc.srcC = s.offC;
        r.run(P_DOWN, divUp(t.w, 8), divUp(t.h, 8));
        r.cbar();
    }
    r.stamp("pyramid");
    // 3-5. per level, coarse -> fine
    static const char* lvNames[] = {"level 0 (finest)", "level 1", "level 2", "level 3", "level 4", "level 5", "level 6", "level 7"};
    int n = (int)I.lv.size();
    if (I.set.motionVersion == MotionVersion::V2) {
        recordV2Search(r);   // V1's block field, faster (shaders/v2_est.glsl): the same passes Settings::v3Guard runs in V3's estimate
        // the rest is each mode's: the re-pick and clean-up, and Quality's coverage (v2Tail)
        I.v2Tail(cb);
        return true;
    }
    for (int i = n - 1; i >= 0; i--) {
        const Level& l = I.lv[i];
        pc.w = l.w; pc.h = l.h; pc.bw = l.bw; pc.bh = l.bh;
        pc.offA = l.offA; pc.offC = l.offC;
        if (i == n - 1) {
            pc.medIdx = -1;
            pc.radius = I.set.searchRadiusTop;
        } else {
            const Level& p = I.lv[i + 1];
            pc.pw = p.bw; pc.ph = p.bh; pc.fPar = p.fM;
            pc.medIdx = 4 + 2 * i;
            pc.radius = I.set.searchRadius;
            const int sh = medhistShift(I.set, n, i + 1);
            if (sh >= 0) {
                pc.radius = sh;   // (the wide histogram's bin shift; the search below needs the radius again)
                r.run(P_MEDHIST_WIDE, 1, 1);
                pc.radius = I.set.searchRadius;
            } else {
                r.run(P_MEDHIST, 1, 1);
            }
            r.cbar();
        }
        pc.fOut = l.fS;
        pc.sadAcc = i == 0 && !I.set.subpixel;   // without sub-pixel refinement the search's SAD is final
        r.run(pc.radius > kSearchMaxR ? P_SEARCH_WIDE : P_SEARCH, l.bw, l.bh);
        r.cbar();
        pc.fIn = l.fS; pc.fOut = l.fM;
        r.run(P_MEDIAN3, divUp(l.bw, 8), divUp(l.bh, 8));
        r.cbar();
        r.stamp(lvNames[std::min(i, 7)]);
    }
    // 6. sub-pixel refinement (+ SAD sum for the scene-cut test)
    pc.w = l0.w; pc.h = l0.h; pc.bw = l0.bw; pc.bh = l0.bh; pc.offA = l0.offA; pc.offC = l0.offC;
    pc.fIn = l0.fM;
    if (I.set.subpixel) {
        pc.fOut = I.fieldSub;
        r.run(P_SUBPIX, l0.bw, l0.bh);
        r.cbar();
        r.stamp("subpixel");
        pc.fIn = I.fieldSub;
    }
    // 8. per-pixel vector choice at flow res
    r.run(P_PIXFIELD, divUp(I.fw, 16), divUp(I.fh, 16));
    r.cbar();
    r.stamp("pixel field");
    return true;
}

// MotionVersion::Learned's estimate after P_LUMA (and CrosshairV2's luma fill), with the estimate's set bound (shaders/ml_*.comp,
// ml-fg export/SPEC.md stages 3-6): FGR's duplicate test (a duplicate skips the net: every pass exits at once, the field is zero),
// the net's own pyramid, per level coarse -> fine the upsampled field + 14-channel input and the five convs (the last adds to the
// field), then the cropped field into V1's per-pixel field (I.fieldPF) and the SAD into V1's scene-cut sum (u[U_SADSUM]).
void Interpolator::Impl::mlEstimate(VkCommandBuffer cb) {
#ifdef FG_HAVE_ML
    Rec r(*this, cb);
    const MlLayout L = mlLayout(W, H, fw, fh, set);
    const bool half = set.learnedHalf;
    const int r1 = L.r0 * L.n0;                                             // activation region R1 (vec4 offset)
    const Level& l0 = lv[0];
    MLPC m{};
    m.W = (int)W; m.H = (int)H; m.fw = fw; m.fh = fh; m.pw = L.pw; m.ph = L.ph;
    m.offYa = l0.offA; m.offYc = l0.offC; m.bw = l0.bw; m.bh = l0.bh; m.pf = fieldPF; m.dupDiff = kMlDupDiff;
    auto run = [&](Pipe p, uint32_t gx, uint32_t gy, uint32_t gz = 1) {
        CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, mg().pipes[p]);
        CmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MLPC), &m);
        CmdDispatch(cb, gx, gy, gz);
    };
    auto stop = [&](int lvl, int step) { return mlDbg.stopLevel == lvl && mlDbg.stopStep == step; };
    if (mlDbg.lumaSrc) {   // test only (setDebugLearned): the reference's flow-res lumas over P_LUMA's
        r.barrier(CS, RW, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        const VkDeviceSize n = (VkDeviceSize)fw * fh * 4;
        VkBufferCopy c[2] = {{0, (VkDeviceSize)l0.offA * 4, n}, {n, (VkDeviceSize)l0.offC * 4, n}};
        CmdCopyBuffer(cb, mlDbg.lumaSrc, buf[0], 2, c);
        r.barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, CS, RW);
    }
    // the duplicate test (full res) and its decision, then the pyramid
    m.nPart = L.nDup;
    run(P_ML_DUP, divUp((int)W, 32), divUp((int)H, 32));
    r.cbar();
    m.mode = 0;
    run(P_ML_REDUCE, 1, 1);
    r.cbar();
    r.stamp("duplicate test");
    run(P_ML_PYR, (uint32_t)L.pw / 16, (uint32_t)L.ph / 16);
    r.cbar();
    r.stamp("net pyramid");
    static const char* lvNames[kMlLevels] = {"net level 0", "net level 1", "net level 2", "net level 3", "net level 4"};
    for (int lvl = kMlLevels - 1; lvl >= 0; lvl--) {
        const uint32_t gx = divUp(L.pw >> lvl, 16), gy = divUp(L.ph >> lvl, 16);
        const bool split = gx * gy < kMlSplitTiles;   // a small level: one output group per workgroup (ml_conv.comp)
        m.lvl = lvl;
        m.actOut = 0;   // the net's input (14 channels) -> R0
        run(mlMatchPipe(half), gx, gy);
        r.cbar();
        if (stop(lvl, 0)) return;
        for (int k = 0; k < kMlLayers; k++) {   // R0 -> R1 -> R0 -> R1 -> R0 -> the field
            m.wOff = (int)kMlNet[0].table[lvl][k][0];
            m.bOff = (int)kMlNet[0].table[lvl][k][1];
            m.actIn = k % 2 ? r1 : 0;
            m.actOut = k % 2 ? 0 : r1;
            const int shape = k == 0 ? 0 : k == kMlLayers - 1 ? 2 : 1;
            run(mlConvPipe(half, shape, split), gx, gy, shape == 2 || !split ? 1 : (uint32_t)L.g);
            r.cbar();
            if (stop(lvl, k + 1)) return;
        }
        r.stamp(lvNames[lvl]);
    }
    if (mlDbg.fieldSrc) {   // test only (setDebugLearned): a given field (fw x fh) over the net's, the input of the passes below
        r.barrier(CS, RW, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        std::vector<VkBufferCopy> c((size_t)fh);
        for (int y = 0; y < fh; y++)
            c[y] = {(VkDeviceSize)y * fw * 8, ((VkDeviceSize)2 * L.S + (VkDeviceSize)2 * y * L.pw) * 4, (VkDeviceSize)fw * 8};
        CmdCopyBuffer(cb, mlDbg.fieldSrc, mo[(int)MotionVersion::Learned].bufs[MLB_STATE].b, (uint32_t)fh, c.data());
        r.barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, CS, RW);
    }
    // the field V1's synthesis reads, and the scene-cut sum: the net's cropped field (ml_out), or with Settings::learnedZeroTest the
    // field after ml-fg's zero-motion test and the SAD of that final field (ml_zero: it reads the net's field in MLB_STATE, which it
    // never writes)
    m.nPart = L.nOut;
    m.mode = mlDbg.zeroCosts ? 2 : 0;
    run(set.learnedZeroTest ? P_ML_ZERO : P_ML_OUT, divUp(fw, 16), divUp(fh, 16));
    r.cbar();
    m.mode = 1;
    run(P_ML_REDUCE, 1, 1);
    r.cbar();
    r.stamp(set.learnedZeroTest ? "net output + zero test" : "net output");
#else
    (void)cb;
#endif
}

// Settings::learnedStaticGuard (shaders/ml_guard.comp, ml-fg scripts/static_guard.py) over the finished generated frame in `out`, after
// the whole-frame fallback pass and before the UI patch, with a synth set bound with `layout` and the synth's writes ordered before it.
// t = the time of the mix (1 - t) A + t C, motionK = 2 max(t, 1 - t) * flowScale (the larger warp side's full-res displacement per
// flow px of the field), pf = the field on the grid of t; it leaves the frame untouched where a whole-frame fallback was taken
// (the scene cut, or one of the untrusted counts u[cnt0 .. cnt0 + nCnt) above globalFallback). The constants: kStaticGuard.
void Interpolator::Impl::mlGuard(VkCommandBuffer cb, float t, float motionK, int pf, int cnt0, int nCnt) {
    GuardPC g{};
    g.W = (int)W; g.H = (int)H; g.fw = fw; g.fh = fh; g.pf = pf; g.bw = lv[0].bw; g.bh = lv[0].bh; g.cnt0 = cnt0; g.nCnt = nCnt;
    g.halo = kStaticGuard.halo;
    g.wA = (float)(1.0 - (double)t);   // float32(1 - t), float32(t): resolve.comp's (and numpy's) time mix
    g.wC = t;
    g.motionK = motionK;
    g.cutSad = set.sceneCutSad; g.globalFrac = set.globalFallback;
    g.lo = kStaticGuard.lo; g.hi = kStaticGuard.hi; g.haloLo = kStaticGuard.haloLo; g.haloHi = kStaticGuard.haloHi;
    g.motionLo = kStaticGuard.motionLo; g.motionHi = kStaticGuard.motionHi; g.wMin = kStaticGuard.wMin;
    CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, mg().pipes[P_ML_GUARD]);
    CmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GuardPC), &g);
    CmdDispatch(cb, divUp((int)W, 16), divUp((int)H, 16), 1);
}
// motionK of mlGuard for one instant t: static_guard.py's 2 max(t, 1 - t) * FLOW_SCALE
static float guardMotionK(double t, uint32_t flowScale) { return (float)(2.0 * std::max(t, 1.0 - t) * flowScale); }

// V2 Quality's claims (shaders/v2_quality.glsl: prev's and curr's, 2 x fw x fh words) -> 0xFFFFFFFF, "unclaimed", before its splat.
// Recorded between the barriers of a counter clear (a transfer write ordered after every earlier record and before this one's passes).
void Interpolator::Impl::v2ResetClaims(VkCommandBuffer cb) {
    CmdFillBuffer(cb, mo[(int)MotionVersion::V2].bufs[V2B_QUAL].b, 0, (VkDeviceSize)fw * fh * 8, 0xFFFFFFFFu);
}

// V2: what the selected mode derives from the estimate (the search's fitted finest field, V2B_FIELD at bw * bh), if this estimate has
// not recorded it yet. Recorded where the estimate or synth set is bound (with `layout`) and every earlier record's use of the buffers
// is ordered (the estimate's / synth's counter clear, which also reset Quality's claims if v2NeedCov()).
//   both modes: the block re-pick of the fitted field into l0.fM (free in V2: V1's median3 output, which V2 never computes), then the
//            clean-up into V2B_FIELD (v2Clean);
//   Quality: then its per-pixel field and static-overlay mask (v2_qfield) and its coverage (v2_qown: the splat to both real frames,
//            v2_qcov: which frame still shows what), v2Cov.
void Interpolator::Impl::v2Tail(VkCommandBuffer cb) {
    const bool cov = v2NeedCov();
    if (v2Clean && !cov) return;
    Rec r(*this, cb);
    const Level& l0 = lv[0];
    PC& pc = r.pc;
    pc.w = l0.w; pc.h = l0.h; pc.bw = l0.bw; pc.bh = l0.bh; pc.offA = l0.offA; pc.offC = l0.offC;
    V2PC v = r.v2pc();
    if (!v2Clean) {
        pc.fIn = l0.bw * l0.bh; pc.fOut = l0.fM;
        r.run(P_V2_REPICK, divUp(l0.bw, 2), divUp(l0.bh, 2));   // 2x2 blocks per workgroup
        r.cbar();
        v.fIn = l0.fM;
        r.runV2(P_V2_CLEAN, v, divUp(l0.bw, 8), divUp(l0.bh, 8));
        r.cbar();
        r.runV2(P_V2_ENCLOSE, v, divUp(((fw + 7) / 8) * ((fh + 7) / 8), 8), 1);   // the static mask's cells (8 per workgroup), on the cleaned field
        r.cbar();
        r.stamp("field clean");
        v2Clean = true;
    }
    if (!cov) return;
    r.runV2(P_V2_QFIELD, v, divUp(fw, 8), divUp(fh, 8));
    r.cbar();
    r.stamp("pixel field");
    const bool wide = qualityWide(set, (int)lv.size());
    if (wide) {   // the claims are key | location index (IB bits, at least 1: ceil(log2(N))), KS low key bits dropped to fit
        int ib = 1;
        while (((int64_t)1 << ib) < (int64_t)fw * fh) ib++;
        v.cnt = ib;
        v.fix = std::max(0, 11 - (32 - ib));
    }
    r.runV2(wide ? P_V2_QOWN_WIDE : P_V2_QOWN, v, divUp(fw, 8), divUp(fh, 8));
    r.cbar();
    r.stamp("splat");
    r.runV2(wide ? P_V2_QCOV_WIDE : P_V2_QCOV, v, divUp(fw, 8), divUp(fh, 8));
    r.cbar();
    r.stamp("coverage");
    v2Cov = true;
}

// V2: recordSynth (n = 1, ts = {t}) and recordSynthBlur (n samples at ts) in one pass, v2_morph.comp (Performance) / v2_qmorph.comp
// (Quality), after the mode's tail if this estimate has none yet; then the UI patch as V1's. The caller has checked the pair, the
// rings and the UI entries.
bool Interpolator::Impl::v2Synth(VkCommandBuffer cb, Frame srcP, Frame srcC, Frame curr, Frame out, const float* ts, int n, bool gauss) {
    VkDescriptorSet set_ = takeSet(R_SYNTH, srcP, srcC, out, false, rejectMasks(cb));
    Rec r(*this, cb);
    // orders this record after every earlier one (shared buffers), zeroes the untrusted count(s): recordSynth's in u[U_UNTRUSTED]
    // (recordDebugCounters reads it), the blur's per sample in u[kBlurCnt0 + k] as V1's blur
    V2PC v = r.v2pc();
    v.cnt = n == 1 ? 0 : kBlurCnt0;
    r.barrier(CS, RW, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    CmdFillBuffer(cb, buf[2], (VkDeviceSize)v.cnt * 4, n == 1 ? 4 : 16 * 4, 0);
    if (v2NeedCov()) v2ResetClaims(cb);   // Quality after a Performance estimate: its coverage
    r.barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, CS, RW);
    CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set_, 0, nullptr);
    v2Tail(cb);
    const Pipe morph = set.v2Mode == V2Mode::Quality ? P_V2_QMORPH : P_V2_MORPH;
    v.n = n;
    v.gauss = gauss;
    for (int k = 0; k < n; k++) v.t[k] = ts[k];
    int nIntV2 = 0;   // the interior samples (a real frame is not tinted)
    for (int k = 0; k < n; k++) nIntV2 += ts[k] > 0.0f && ts[k] < 1.0f;
    // the tiles near a static overlay go to the static variant of the pass (V2's static-structure mask), the others to the plain
    // one; both write disjoint tiles and only add to the counts: no barrier between them. (Intended: syncval with its shader-accesses
    // heuristic reports a WRITE_AFTER_WRITE on binding 5, the output image, between them, and in Nv12 / P010 also on binding 10, its
    // CbCr plane: each 8x8 tile, and the 4x4 chroma texels of its 2x2 blocks, are stored by one of the two.)
    const Pipe morphS = set.v2Mode == V2Mode::Quality ? P_V2_QMORPH_S : P_V2_MORPH_S;
    r.runV2(morph, v, divUp((int)W, 8), divUp((int)H, 8));
    r.runV2(morphS, v, divUp((int)W, 8), divUp((int)H, 8));
    r.cbar();
    const char* name = n == 1 ? "morph" : "blur morph";
    r.stamp(name);
    v.fix = 1;   // exits at once unless a sample turned out to have too many untrusted pixels
    r.runV2(morphS, v, divUp((int)W, 64), divUp((int)H, 64));
    name = "fallback";
    if (estUi) {
        r.barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, RW);
        r.stamp(name);
        uiRecordPatch(cb, curr, out, uiE[estEnt[1]]);   // the real UI pixels back over the generated frame
        debugDraw(cb, set_, curr, out, 0, nIntV2);   // Settings::debugView (Off: nothing)
        r.barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, VK_ACCESS_SHADER_READ_BIT);   // compute-only queue safe
        r.stamp("ui_patch");
        return true;
    }
    debugDraw(cb, set_, curr, out, 0, nIntV2);
    r.barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, VK_ACCESS_SHADER_READ_BIT);   // compute-only queue safe
    r.stamp(name);
    return true;
}

// V3: what an estimate clears, recorded inside Rec::clearCounters' transfer window (every earlier record's use of the buffers is ordered before
// it, the passes after it): the state G (4096 words of V3B_AUX; the estimate's counters and decisions accumulate there); the whole of V3's
// buffers once after they were created, so that no pass or debug read ever sees uninitialised memory; and, after an injection, the flows.
void Interpolator::Impl::v3Reset(VkCommandBuffer cb) {
    const MotionGroup& g = mo[(int)MotionVersion::V3];
    if (g.bufs.empty()) return;
    const V3Layout V = v3Layout(W, H, set.flowScale);
    if (v3Fresh) {
        for (const UiBuf& b : g.bufs) CmdFillBuffer(cb, b.b, 0, VK_WHOLE_SIZE, 0);
        v3Fresh = false;
        v3Dirty = false;
        return;
    }
    CmdFillBuffer(cb, g.bufs[V3B_AUX].b, V.gOff() * 4, (VkDeviceSize)kV3GWords * 4, 0);
    if (v3Dirty) {   // FD and FF hold the flows of an injection
        CmdFillBuffer(cb, g.bufs[V3B_FLOW].b, 64 * V.B, 32 * V.N, 0);
        v3Dirty = false;
    }
}

// V3: recordSynth (n = 1, t0 = t1 = t) and recordSynthBlur (n samples over [t0, t1]): v3_project (the T-field of every interior sample, flow
// resolution) and v3_synth (full resolution, all samples averaged in registers; gpu/V3.md sections 3.14, 3.15), then the UI patch as V2's. The caller
// has checked the pair, the rings and the UI entries. The interior samples (0 < t_k < 1, contiguous) are the ones that need a T-field slot; a
// sample at t <= 0 / >= 1 is a real frame. A record that needs more slots than V3B_FLOW holds grows it first (a copy of the whole buffer, the
// estimate included, into a larger one; the old buffer waits in blurGrave).
bool Interpolator::Impl::v3Synth(VkCommandBuffer cb, Frame srcP, Frame srcC, Frame curr, Frame out, float t0, float t1, int n, bool gauss, uint32_t dumpStride, uint32_t dumpCount) {
    int lo = n, hi = -1;
    float tk[16]{};   // the sample times, as recordSynthBlur computes them (double arithmetic): the shaders read these, so a sub-frame is recordSynth(t_k) exactly
    for (int k = 0; k < n && k < 16; k++) {
        tk[k] = std::min(std::max((float)((double)t0 + ((double)t1 - (double)t0) * (k + 0.5) / n), t0), t1);
        if (tk[k] > 0.0f && tk[k] < 1.0f) { lo = std::min(lo, k); hi = k; }
    }
    const int nInt = hi - lo + 1;   // 0: every sample is a real frame
    Rec r(*this, cb);
    // orders this record after every earlier one (shared buffers) and zeroes u[0] (the untrusted count, which recordDebugCounters reads)
    r.barrier(CS, RW, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT);
    // T-field slots this record needs: the interior samples', and (the weights dump) the slots after them that hold dumpCount records of 28 floats
    const int dumpSlots = dumpCount ? (int)(((uint64_t)dumpCount * 112 + 16 * v3Layout(W, H, set.flowScale).N - 1) / (16 * v3Layout(W, H, set.flowScale).N)) : 0;
    const int need = nInt + dumpSlots;
    if (need > (int)v3Ns) {
        MotionGroup& g = mo[(int)MotionVersion::V3];
        const V3Layout V = v3Layout(W, H, set.flowScale);
        const VkDeviceSize bytes = V.flowBytes((uint64_t)need);
        UiBuf nb;
        if (bytes > lim.sbRange || !allocBuf(nb, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) return false;
        VkBufferCopy cp{0, 0, g.bufs[V3B_FLOW].size};
        CmdCopyBuffer(cb, g.bufs[V3B_FLOW].b, nb.b, 1, &cp);
        blurGrave.push_back(g.bufs[V3B_FLOW]);
        g.bufs[V3B_FLOW] = nb;
        v3Ns = (uint32_t)need;
    }
    VkDescriptorSet set_ = takeSet(R_SYNTH, srcP, srcC, out, false, rejectMasks(cb));
    CmdFillBuffer(cb, buf[2], 0, 4, 0);
    CmdUpdateBuffer(cb, mo[(int)MotionVersion::V3].bufs[V3B_AUX].b, ((VkDeviceSize)v3Layout(W, H, set.flowScale).gOff() + 1200) * 4, sizeof(tk), tk);   // G.tk (G_TK)
    r.barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, CS, RW);
    CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set_, 0, nullptr);
    V3PC v = r.v3pc();
    v.n = n;
    v.t0 = t0;
    v.t1 = t1;
    v.lvl = nInt;
    v.iter = nInt > 0 ? lo : 0;
    v.mode = 0;
    if (nInt > 0) {
        r.runV3(P_V3_PROJECT, v, divUp(fw, 8), divUp(fh, 8), (uint32_t)nInt);
        r.cbar();
        r.stamp("project");
    }
    // synth mode: bit 0 Gaussian blur weights, bit 1 hook (l) v3DetailSplit (also selects the -DV3L=1 build P_V3_SYNTH_L, which does not read it), bit 3 the weights dump
    // (lw = the stride of the dumped pixels in the raster, lh = their number)
    // Settings::v3Blend >= 1 selects the -DV3B=1 build (P_V3_SYNTH_B; bits 4..7 = the variant 1..kV3BlendMax), unless the detail split is on or this format has no such build (then the rule)
    const bool blend = set.v3Blend != 0 && !set.v3DetailSplit && mo[(int)MotionVersion::V3].pipes[P_V3_SYNTH_B];
    v.mode = (gauss ? V3M_GAUSS : 0) | (set.v3DetailSplit ? V3M_DETAIL : 0) | (dumpCount ? V3M_DUMP : 0) | (blend ? (int)set.v3Blend << V3M_BLEND_SHIFT : 0);
    v.lw = (int)dumpStride;
    v.lh = (int)dumpCount;
    r.runV3(set.v3DetailSplit ? P_V3_SYNTH_L : blend ? P_V3_SYNTH_B : P_V3_SYNTH, v, divUp((int)W, 8), divUp((int)H, 8));
    const char* name = n == 1 ? "synth" : "blur synth";
    if (estUi) {
        r.barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, RW);
        r.stamp(name);
        uiRecordPatch(cb, curr, out, uiE[estEnt[1]]);   // the real UI pixels back over the generated frame
        debugDraw(cb, set_, curr, out, 0, dumpCount ? 0 : nInt);   // Settings::debugView (Off: nothing)
        r.barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, VK_ACCESS_SHADER_READ_BIT);   // compute-only queue safe
        r.stamp("ui_patch");
        return true;
    }
    debugDraw(cb, set_, curr, out, 0, dumpCount ? 0 : nInt);
    r.barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, VK_ACCESS_SHADER_READ_BIT);   // compute-only queue safe
    r.stamp(name);
    return true;
}

bool Interpolator::recordSynth(VkCommandBuffer cb, Frame prev, Frame curr, Frame out, float t) {
    if (!m || !m->W || !(t > 0.0f && t < 1.0f)) return false;
    Impl& I = *m;
    if (!I.haveEst || prev.view != I.estPrev || curr.view != I.estCurr || !I.framesOk(prev) || !I.framesOk(curr) || !I.framesOk(out)) return false;
    assert(I.frameSets[Impl::R_SYNTH] < kSetRing && "framegen: more than kSetRing recordSynth() since beginFrame()");
    if (I.frameSets[Impl::R_SYNTH] >= kSetRing) return false;
    Frame srcP = prev, srcC = curr;
    if (I.estUi) {
        // the entries the estimate saw must still be those frames (not evicted / re-observed since); the clean copies too
        const UiEntry &ep = I.uiE[I.estEnt[0]], &ec = I.uiE[I.estEnt[1]];
        if (ep.seq != I.estSeq[0] || ec.seq != I.estSeq[1] || (I.estClean && (!ep.view || !ec.view))) return false;
        if (I.uiFrameSets + 2 + I.dbgUiSets() > kUiSets) return false;
        if (I.estClean) { srcP = ep.frame(); srcC = ec.frame(); }
    }
    if (I.mv2()) return I.v2Synth(cb, srcP, srcC, curr, out, &t, 1, false);
    if (I.mv3()) return I.v3Synth(cb, srcP, srcC, curr, out, t, t, 1, false);
    VkDescriptorSet set = I.takeSet(Impl::R_SYNTH, srcP, srcC, out, false, I.rejectMasks(cb));

    Rec r(I, cb);
    // Serialises this synth after the estimate and after any earlier synth (shared field-at-t, mismatch and
    // counter scratch): the CS -> transfer -> CS chain orders all earlier reads/writes of our buffers.
    r.clearCounters(1, set);   // untrusted count only; the SAD sum belongs to the estimate
    PC& pc = r.pc;
    if (t != 0.5f) {
        // numpy float32 constants of interp.synth / field_at (computed in double, then rounded); the warp
        // factors float32(2t), float32(2(1 - t)) are exactly 2 wC, 2 wA (scaling by 2 commutes with rounding)
        double td = t;
        pc.s = (float)(2.0 * td - 1.0);
        pc.wA = (float)(1.0 - td);
        pc.wC = (float)td;
        // 9a. field lookup at time t (flow res)
        pc.fIn = I.fieldPF; pc.fOut = I.fieldT;
        r.run(P_LOOKUP, divUp(I.fw, 8), divUp(I.fh, 8));
        r.cbar();
        pc.pf = I.fieldT;
        r.stamp("lookup");
    }
    // 9. full-res mismatch
    r.run(P_ERROR, divUp(((int)I.W + 3) / 4, 8), divUp((int)I.H, 8));
    r.cbar();
    r.stamp("mismatch");
    // 10-11. box + trust + resolve
    r.run(P_RESOLVE, divUp((int)I.W, 16), divUp((int)I.H, 16));
    r.cbar();
    r.stamp("resolve");
    // global fallbacks (scene cut / too much untrusted) -> the nearer real frame
    r.run(P_FALLBACK, divUp((int)I.W, 64), divUp((int)I.H, 64));
    const char* last = "fallback";
    if (I.mlGuardOn()) {   // Learned, Settings::learnedStaticGuard: over the finished frame, before the UI patch
        r.barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, RW);
        r.stamp(last);
        I.mlGuard(cb, t, guardMotionK(t, I.set.flowScale), pc.pf, 0, 1);   // the field at t; u[0] = U_UNTRUSTED
        last = "static guard";
    }
    if (I.estUi) {
        r.barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, RW);
        r.stamp(last);
        I.uiRecordPatch(cb, curr, out, I.uiE[I.estEnt[1]]);   // the real UI pixels back over the generated frame
        I.debugDraw(cb, set, curr, out, pc.pf, 1);   // Settings::debugView (Off: nothing)
        r.barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, VK_ACCESS_SHADER_READ_BIT);   // compute-only queue safe
        r.stamp("ui_patch");
        return true;
    }
    I.debugDraw(cb, set, curr, out, pc.pf, 1);
    r.barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, VK_ACCESS_SHADER_READ_BIT);   // compute-only queue safe
    r.stamp(last);
    return true;
}

bool Interpolator::recordSynthBlur(VkCommandBuffer cb, Frame prev, Frame curr, Frame out, float t0, float t1) {
    if (!m || !m->W || !(t0 >= 0.0f && t1 <= 1.0f && t0 <= t1)) return false;   // (also false for NaN)
    Impl& I = *m;
    const uint32_t N = I.set.blurSamples;
    const bool blur = N >= 2 && t0 < t1;
    if (!blur && t1 > 0.0f && t1 < 1.0f) return recordSynth(cb, prev, curr, out, t1);   // one interior instant
    if (!I.haveEst || prev.view != I.estPrev || curr.view != I.estCurr || !I.framesOk(prev) || !I.framesOk(curr) || !I.framesOk(out)) return false;
    assert(I.frameSets[Impl::R_SYNTH] < kSetRing && "framegen: more than kSetRing recordSynth()/recordSynthBlur() since beginFrame()");
    if (I.frameSets[Impl::R_SYNTH] >= kSetRing) return false;
    Frame srcP = prev, srcC = curr;
    if (I.estUi) {
        const UiEntry &ep = I.uiE[I.estEnt[0]], &ec = I.uiE[I.estEnt[1]];
        if (ep.seq != I.estSeq[0] || ec.seq != I.estSeq[1] || (I.estClean && (!ep.view || !ec.view))) return false;
        if (I.uiFrameSets + 2 + I.dbgUiSets() > kUiSets) return false;
        if (I.estClean) { srcP = ep.frame(); srcC = ec.frame(); }
    }

    // the samples: the MIDPOINTS of N equal slices of [t0, t1], t_k = t0 + (t1 - t0)(k + 0.5)/N (a box-filter shutter).
    // Why: with both ends sampled, windows that tile (100 % blur) counted the real frame at the shared end in BOTH
    // neighbours. A non-blur output (blur off, or t0 == t1) is the single instant t1 as before.
    const int n = blur ? (int)N : 1;
    BlurPC bp{};
    for (int k = 0; k < n; k++) {
        double t = blur ? t0 + ((double)t1 - (double)t0) * (k + 0.5) / n : (double)t1;
        bp.t[k] = std::min(std::max((float)t, t0), t1);
    }
    int lo = n, hi = -1;   // the interior samples (0 < t < 1) are contiguous: only they need a synthesis
    for (int k = 0; k < n; k++)
        if (bp.t[k] > 0.0f && bp.t[k] < 1.0f) { lo = std::min(lo, k); hi = k; }
    const int nInt = hi - lo + 1;
    if (I.mv2()) return I.v2Synth(cb, srcP, srcC, curr, out, bp.t, n, I.set.blurWeights == BlurWeights::Gaussian);
    if (I.mv3()) return I.v3Synth(cb, srcP, srcC, curr, out, blur ? t0 : t1, t1, n, I.set.blurWeights == BlurWeights::Gaussian);

    // scratch: n slots of the field at t (flow res) and of the packed mismatch (full res)
    const VkDeviceSize errWords = (VkDeviceSize)((I.W + 3) / 4) * I.H;
    if (!I.growBlur(I.blurField, (VkDeviceSize)n * I.fw * I.fh * 8) || !I.growBlur(I.blurErr, (VkDeviceSize)n * errWords * 4)) return false;
    VkDescriptorSet set = I.takeSet(Impl::R_SYNTH, srcP, srcC, out, true, I.rejectMasks(cb));

    Rec r(I, cb);
    // serialises this record after everything before it (shared scratch), then zeroes the per-sample counters
    r.barrier(CS, RW, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    I.CmdFillBuffer(cb, I.buf[2], kBlurCnt0 * 4, 16 * 4, 0);
    r.barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, CS, RW);
    I.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, I.blurLayout, 0, 1, &set, 0, nullptr);

    const Settings& st = I.set;
    bp.W = (int)I.W; bp.H = (int)I.H; bp.fw = I.fw; bp.fh = I.fh; bp.fscale = (int)st.flowScale;
    bp.n = n; bp.bw = I.lv[0].bw; bp.bh = I.lv[0].bh; bp.k0 = lo; bp.pf = I.fieldPF;
    bp.gauss = st.blurWeights == BlurWeights::Gaussian;
    bp.cutSad = st.sceneCutSad; bp.trust0 = st.trustLow; bp.trustScale = 1.0f / (st.trustHigh - st.trustLow);
    bp.globalFrac = st.globalFallback;
    auto run = [&](BlurPipe p, uint32_t gx, uint32_t gy, uint32_t gz) {
        I.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, I.mg().blurPipes[p]);
        I.CmdPushConstants(cb, I.blurLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BlurPC), &bp);
        I.CmdDispatch(cb, gx, gy, gz);
    };
    if (nInt > 0) {
        run(B_LOOKUP, divUp(I.fw, 8), divUp(I.fh, 8), (uint32_t)nInt);
        r.cbar();
        r.stamp("blur lookup");
        run(B_ERROR, divUp(((int)I.W + 3) / 4, 8), divUp((int)I.H, 8), (uint32_t)nInt);
        r.cbar();
        r.stamp("blur mismatch");
    }
    bp.fix = 0;
    run(B_RESOLVE, divUp((int)I.W, 16), divUp((int)I.H, 16), 1);
    r.cbar();
    r.stamp("blur resolve");
    bp.fix = 1;   // exits at once unless a sample turned out to have too many untrusted pixels
    run(B_RESOLVE, divUp((int)I.W, 64), divUp((int)I.H, 64), 1);
    const char* last = "blur fixup";
    if (I.mlGuardOn() && nInt > 0) {
        // Learned, Settings::learnedStaticGuard, once over the AVERAGE (before the UI patch): the time mix at the samples' weighted mean
        // time (the time mix is linear in t, so this is the average of the samples' own time mixes, real-frame samples included), the
        // motion gate with the midpoint field and the largest motionK of the interior samples (the gate closes if any sample moves); no
        // guard when a sample took a whole-frame fallback (the scene cut, or an interior sample's untrusted count past globalFallback)
        double ws = 0, wt = 0, mk = 0;
        for (int k = 0; k < n; k++) {
            const double x = (2.0 * k + 1.0 - n) / n, w = bp.gauss ? std::exp(-2.0 * x * x) : 1.0;   // == blur_resolve.comp's weights
            ws += w;
            wt += w * bp.t[k];
            if (k >= lo && k <= hi) mk = std::max(mk, (double)guardMotionK(bp.t[k], st.flowScale));
        }
        r.barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, RW);
        r.stamp(last);
        I.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, I.layout, 0, 1, &set, 0, nullptr);   // the guard pushes PC's range
        I.mlGuard(cb, (float)(wt / ws), (float)mk, I.fieldPF, kBlurCnt0 + lo, nInt);
        last = "static guard";
    }
    if (I.estUi) {
        r.barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, RW);
        r.stamp(last);
        I.uiRecordPatch(cb, curr, out, I.uiE[I.estEnt[1]]);   // the real UI pixels back over the blurred frame
        I.debugDraw(cb, set, curr, out, I.fieldPF, nInt);   // Settings::debugView, once over the average (Off: nothing)
        r.barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, VK_ACCESS_SHADER_READ_BIT);
        r.stamp("ui_patch");
        return true;
    }
    I.debugDraw(cb, set, curr, out, I.fieldPF, nInt);
    r.barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, VK_ACCESS_SHADER_READ_BIT);
    r.stamp(last);
    return true;
}

bool Interpolator::record(VkCommandBuffer cb, Frame prev, Frame curr, Frame out, float t) {
    if (!m || !m->W || !(t > 0.0f && t < 1.0f)) return false;
    // both rings must have room, so that a refused record() records nothing
    assert(m->frameSets[0] < kSetRing && m->frameSets[1] < kSetRing && "framegen: more than kSetRing records since beginFrame()");
    if (m->frameSets[Impl::R_EST] >= kSetRing || m->frameSets[Impl::R_SYNTH] >= kSetRing) return false;
    if (m->uiActive() && m->uiFrameSets + 2 * m->uiObserveSets() + 2 + m->dbgUiSets() + (m->v2Fill() ? 2 : 0) > kUiSets) return false;   // 2 implicit observes + the patch (+ V2's two luma fills)
    return recordEstimate(cb, prev, curr) && recordSynth(cb, prev, curr, out, t);
}

}  // namespace framegen

