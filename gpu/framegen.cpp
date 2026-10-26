// framegen: Vulkan compute port of proto/interp.py. See framegen.h for the API contract.
#include "framegen.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <cassert>

#include "luma_spv.h"
#include "down_spv.h"
#include "medhist_spv.h"
#include "search_spv.h"
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
#include "v2_search_spv.h"
#include "v2_qfield_spv.h"
#include "v2_qown_spv.h"
#include "v2_qcov_spv.h"
#include "v2_qmorph_spv.h"
#include "v2_clean_spv.h"
#include "v2_morph_spv.h"
#include "v2_repick_spv.h"
#include "v2_static_spv.h"
#include "v2_enclose_spv.h"
#include "v2_morph_s_spv.h"
#include "v2_qmorph_s_spv.h"
// MotionVersion::Learned: its passes and the nets' weights (gpu/ml/embed_weights.py), probed as a set: a host build that does not
// generate them gets Learned refused with a reason. Only its duplicate test (ml_dup) has format variants (probed per format below).
#if __has_include("ml_conv_spv.h") && __has_include("ml_weights.h")
#define FG_HAVE_ML 1
#include "ml_dup_spv.h"
#include "ml_reduce_spv.h"
#include "ml_pyr_spv.h"
#include "ml_match_spv.h"
#include "ml_conv_spv.h"
#include "ml_out_spv.h"
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
// ... and MotionVersion::Learned's duplicate test of rgb10 (probed on its own, like V2's synthesis)
#if __has_include("ml_dup_rgb10_spv.h") && defined(FG_HAVE_RGB10) && defined(FG_HAVE_ML)
#define FG_HAVE_ML_RGB10 1
#include "ml_dup_rgb10_spv.h"
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
#if __has_include("ml_dup_f16_spv.h") && defined(FG_HAVE_F16) && defined(FG_HAVE_ML)
#define FG_HAVE_ML_F16 1
#include "ml_dup_f16_spv.h"
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
#if __has_include("ml_dup_nv12_spv.h") && defined(FG_HAVE_NV12) && defined(FG_HAVE_ML)
#define FG_HAVE_ML_NV12 1
#include "ml_dup_nv12_spv.h"
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
#if __has_include("ml_dup_p010_spv.h") && defined(FG_HAVE_P010) && defined(FG_HAVE_ML)
#define FG_HAVE_ML_P010 1
#include "ml_dup_p010_spv.h"
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
    v2Morph, v2MorphS, v2QMorph, v2QMorphS, mlDup; };
#define COLOUR_SET(f) ColourSpv{SPV(luma##f), SPV(error##f), SPV(resolve##f), SPV(fallback##f), SPV(ui_detect##f), \
    SPV(ui_box_update##f), SPV(ui_inpaint##f), SPV(ui_patch##f), SPV(blur_error##f), SPV(blur_resolve##f), Spv{}, Spv{}, \
    SPV(ui_v2_update##f), SPV(ui_v2_lumafill##f), Spv{}, Spv{}, Spv{}, Spv{}}
#define V2_SET(c, f) (c.v2Morph = SPV(v2_morph##f), c.v2MorphS = SPV(v2_morph_s##f), c.v2QMorph = SPV(v2_qmorph##f), c.v2QMorphS = SPV(v2_qmorph_s##f))
#define ML_SET(c, f) (c.mlDup = SPV(ml_dup##f))
// false = this format's variants were not built into the library (meson option `formats`)
bool colourSpv(Format f, ColourSpv& c) {
    switch (f) {
    case Format::Rgba8:
        c = COLOUR_SET(); V2_SET(c, );
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
        return true;
#endif
    default: return false;
    }
}
inline bool isYuv(Format f) { return f == Format::Nv12 || f == Format::P010; }
const char* const kV2NotBuilt = "the MotionVersion::V2 shader variants of this format (v2_morph, v2_morph_s, v2_qmorph, v2_qmorph_s) are not built into the library";
const char* const kMlNotBuilt = "the MotionVersion::Learned shaders of this format (ml_dup, ml_reduce, ml_pyr, ml_match, ml_conv, ml_out) or its weights (ml_weights.h, gpu/ml/embed_weights.py) are not built into the library";
const char* const kMlFlowScale = "MotionVersion::Learned needs flowScale 2 (its nets were trained at 2)";
#ifdef FG_HAVE_ML
// The embedded nets, in LearnedNet order (gpu/ml/embed_weights.py): weights in shaders/ml_conv.comp's layout, the table of each
// (level, layer)'s weight / bias offsets (vec4 units, relative to the net's start in the weights buffer)
struct MlNet { const uint32_t* w; size_t words; uint32_t width; const uint32_t (*table)[5][2]; };
const MlNet kMlNet[2] = {{kMlWeights_e2v3, sizeof(kMlWeights_e2v3) / 4, kMlWidth_e2v3, kMlTable_e2v3},
                         {kMlWeights_e1v3, sizeof(kMlWeights_e1v3) / 4, kMlWidth_e1v3, kMlTable_e1v3}};
static_assert(kMlWidth_e2v3 == 24 && kMlWidth_e1v3 == 12, "mlWidth()");
static_assert(sizeof(kMlWeights_e2v3) % 16 == 0, "the second net starts on a vec4");
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
static const int kEOff = 64;   // == E_OFF in motion_v1.glsl

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
    int32_t pad_[13];
};
static_assert(sizeof(MLPC) == sizeof(PC), "MLPC is pushed over PC's range");
// MotionVersion::Learned's duplicate test: FGR's own, proto/harness.py DUP_DIFF (and gpu/compare.py): a pair whose mean over the
// pixels of the channel-mean |a - c| is at most this many 8-bit levels is a duplicate ("re-encoded duplicate frames differ by up to
// ~0.5"). The net is skipped there and the field is zero (shaders/ml_dup.comp).
static const float kMlDupDiff = 0.6f;
static const int kMlLevels = 5, kMlLayers = 5;   // == ML_LEVELS in ml_common.glsl; layers per level (4 hidden + the linear one)

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
};
static_assert(sizeof(UiPC) <= 128, "push constants beyond the guaranteed minimum");
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
// shape (first / hidden / last) and net (LearnedNet), shaders/ml_conv.comp specialised.
enum Pipe { P_LUMA, P_DOWN, P_MEDHIST, P_SEARCH, P_MEDIAN3, P_SUBPIX, P_PIXFIELD, P_ERROR, P_RESOLVE, P_FALLBACK, P_LOOKUP,
            P_V2_MEDHIST, P_V2_SEARCH, P_V2_CLEAN, P_V2_MORPH, P_V2_REPICK, P_V2_QFIELD, P_V2_QOWN, P_V2_QCOV, P_V2_QMORPH, P_V2_STATIC, P_V2_ENCLOSE, P_V2_MORPH_S, P_V2_QMORPH_S,
            P_ML_DUP, P_ML_REDUCE, P_ML_PYR, P_ML_MATCH, P_ML_OUT, P_ML_CONV, P_COUNT = P_ML_CONV + 6 };
static const int kMlNets = 2;   // LearnedNet::E2v3, E1v3
inline Pipe mlConvPipe(int net, int shape) { return (Pipe)(P_ML_CONV + 3 * net + shape); }   // shape 0 = first, 1 = hidden, 2 = last

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
static const int kMotionVersions = 3;   // MotionVersion::V1, V2, Learned
// MotionGroup::bufs of V2: binding 13 (vec2): the cleaned block field, the fitted finest field (bw x bh each), Quality's per-pixel
// field (fw x fh); binding 14 (uint): Quality's claims, own claims and coverage (4 x fw x fh, shaders/v2_quality.glsl)
enum { V2B_FIELD, V2B_QUAL, V2B_COUNT };
// MotionGroup::bufs of Learned: binding 13 (float): the pyramid, the field of every level, the partial sums; binding 14 (vec4): the
// activations, two regions (shaders/ml_common.glsl)
enum { MLB_STATE, MLB_ACT, MLB_COUNT };
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
// MIN_SIDE), the luma words and the field entries (vec2) of the shared buffers. createSizeDependent and sizeLimitError use it.
struct SizeLayout {
    int fw = 0, fh = 0;
    std::vector<Level> lv;
    int lumaWords = 0, fieldSub = 0, fieldPF = 0, fieldT = 0, fieldEntries = 0;
};
static int rintHalfEven(double v) { return (int)std::nearbyint(v); }   // == cv2 saturate_cast<int>
static SizeLayout sizeLayout(uint32_t W, uint32_t H, uint32_t flowScale) {
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
        if (std::min(w, h) / 2 < 48) break;
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
// 14-channel input (4 groups) and two hidden layers (width / 4 groups each), R1 the other two.
inline int mlWidth(LearnedNet n) { return n == LearnedNet::E1v3 ? 12 : 24; }
struct MlLayout {
    int pw = 0, ph = 0, n0 = 0, S = 0, nDup = 0, nOut = 0;
    VkDeviceSize stateBytes = 0;
    int g = 0, r0 = 0, groups = 0;   // the net's groups per hidden layer, R0's groups, R0 + R1
    VkDeviceSize actBytes = 0;
};
static MlLayout mlLayout(uint32_t W, uint32_t H, int fw, int fh, LearnedNet net) {
    MlLayout L;
    L.pw = (fw + 15) & ~15;
    L.ph = (fh + 15) & ~15;
    L.n0 = L.pw * L.ph;
    for (int l = 0; l < kMlLevels; l++) L.S += (L.pw >> l) * (L.ph >> l);
    L.nDup = (int)(((W + 31) / 32) * ((H + 31) / 32));   // ml_dup.comp: 32x32 pixels per workgroup
    L.nOut = ((fw + 15) / 16) * ((fh + 15) / 16);       // ml_out.comp: 16x16 flow pixels per workgroup
    L.stateBytes = ((VkDeviceSize)4 * L.S + std::max(L.nDup, L.nOut)) * 4;
    L.g = mlWidth(net) / 4;
    L.r0 = std::max(4, L.g);
    L.groups = L.r0 + L.g;
    L.actBytes = (VkDeviceSize)L.groups * L.n0 * 16;
    return L;
}

// Null if a W x H frame fits the device under these settings (the selected motion version's buffers and dispatches), else why not.
// Below kMinSize the flow resolution degenerates (W or H = 1 -> size-0 buffers). The largest buffers grow with W * H / flowScale^2
// (V2 Quality's claims and coverage, V1's motion-blur field scratch), the widest dispatch with W / 8 (and V2's static-mask cells).
static const char* sizeLimitError(uint32_t W, uint32_t H, const Settings& s, const DevLimits& lim) {
    if (W < Interpolator::kMinSize || H < Interpolator::kMinSize) return "the frame size is below Interpolator::kMinSize (16 pixels) in width or height";
    if (lim.img2D && (W > lim.img2D || H > lim.img2D)) return "the frame size exceeds the device's maxImageDimension2D";
    if ((uint64_t)W * H > (1ull << 30)) return "the frame size exceeds 2^30 pixels (32-bit indexing of the library's buffers)";
    const SizeLayout L = sizeLayout(W, H, s.flowScale);
    const uint64_t fwh = (uint64_t)L.fw * L.fh, cells = (uint64_t)((L.fw + 7) / 8) * ((L.fh + 7) / 8);
    uint64_t big = std::max({(uint64_t)L.lumaWords * 4, (uint64_t)L.fieldEntries * 8, (uint64_t)(64 + (W + 3) / 4 * (uint64_t)H) * 4});
    uint64_t gx = (W + 7) / 8, gy = (H + 7) / 8;
    if (s.motionVersion == MotionVersion::V2) {
        big = std::max({big, ((uint64_t)L.lv[0].bw * L.lv[0].bh * 2 + fwh) * 8, (fwh * 5 + 3 * cells) * 4});
        gx = std::max(gx, (cells + 7) / 8);   // v2_enclose.comp: 8 cells per workgroup, one row of workgroups
    } else {
        if (s.motionVersion == MotionVersion::Learned) {   // the net's state and activations (its dispatches are narrower than W / 8)
            const MlLayout M = mlLayout(W, H, L.fw, L.fh, s.learnedNet);
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
    // mlWeightsUp), the activation buffer's capacity in vec4 groups per level-0 pixel (it grows for a wider net, the old one waits in
    // mlGrave until the size is freed), and the test hooks (setDebugLearned)
    UiBuf mlWeights;
    bool mlWeightsUp = false;
    int mlActGroups = 0;
    std::vector<UiBuf> mlGrave;
    LearnedDebug mlDbg{};
    bool mlearned() const { return set.motionVersion == MotionVersion::Learned; }
    bool mlSized();
    void mlUpload(VkCommandBuffer cb);
    void mlEstimate(VkCommandBuffer cb);
    bool v2Synth(VkCommandBuffer cb, Frame srcP, Frame srcC, Frame curr, Frame out, const float* ts, int n, bool gauss);

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

    VkDescriptorSet takeSet(int ring, Frame prev, Frame curr, Frame out, bool blur = false, RejPair rej = {});
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
    // interp.flow(): luma at 1/FLOW_SCALE (cv2 rounds the size), pyramid down to MIN_SIDE (sizeLayout).
    SizeLayout L = sizeLayout(W, H, set.flowScale);
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
    mlActGroups = 0;
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
    if (set.uiProtection == UiProtection::Off) {
        for (UiEntry& e : uiE) e.key = VK_NULL_HANDLE;
        uiLast = -1;
    }
    if (set.uiProtection != UiProtection::WholeScreen) uiCellHist = false;
    if (set.uiProtection == UiProtection::Off || old.uiProtection == UiProtection::Off) uiBoxHist = false;
    // V2 keeps another state layout and another mask format than the other modes: nothing observed under one is valid under the other
    if (set.uiProtection != old.uiProtection && (set.uiProtection == UiProtection::CrosshairV2 || old.uiProtection == UiProtection::CrosshairV2)) {
        for (UiEntry& e : uiE) e.key = VK_NULL_HANDLE;
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
    if (ii.settings.motionVersion == MotionVersion::V2 && !cs.v2Morph.code) { err_ = kV2NotBuilt; return false; }
    if (ii.settings.motionVersion == MotionVersion::Learned && !cs.mlDup.code) { err_ = kMlNotBuilt; return false; }
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
    // (shaders/ml_common.glsl)
    uint32_t nb = m->yuv ? 11 : 8;
    VkDescriptorSetLayoutBinding b[16]{};
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
    for (uint32_t k = 11; k <= 15; k++, nb++) {
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

bool Interpolator::validate(const Settings& s) {
    auto in = [](float v, float lo, float hi) { return std::isfinite(v) && v >= lo && v <= hi; };
    return in(s.trustLow, 0.0f, 255.0f) && in(s.trustHigh, 0.0f, 255.0f) && s.trustHigh - s.trustLow >= 1.0f &&
           in(s.globalFallback, 0.0f, 1.0f) && s.globalFallback > 0.0f &&
           in(s.sceneCutSad, 0.0f, 255.0f) && s.sceneCutSad > 0.0f &&
           in(s.hudBonus, 0.0f, 255.0f) && in(s.smoothBonus, 0.0f, 255.0f) && in(s.searchPenalty, 0.0f, 255.0f) &&
           (s.flowScale == 1 || s.flowScale == 2 || s.flowScale == 4) &&
           (uint32_t)s.uiProtection <= (uint32_t)UiProtection::CrosshairV2 && (uint32_t)s.motionVersion <= (uint32_t)MotionVersion::Learned &&
           (uint32_t)s.v2Mode <= (uint32_t)V2Mode::Performance && (uint32_t)s.learnedNet <= (uint32_t)LearnedNet::E1v3 &&
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
    if (m && s.motionVersion == MotionVersion::V2 && !m->cs.v2Morph.code) return kV2NotBuilt;
    if (m && s.motionVersion == MotionVersion::Learned && !m->cs.mlDup.code) return kMlNotBuilt;
    return m && m->W ? sizeLimitError(m->W, m->H, s, m->lim) : nullptr;   // the held size under these settings
}

const char* Interpolator::sizeError(uint32_t w, uint32_t h) const {
    if (!m) return "not initialised";
    return sizeLimitError(w, h, m->set, m->lim);
}

bool Interpolator::setSettings(const Settings& s) {
    if (!m || settingsError(s) || s.flowScale != m->set.flowScale) return false;
    // UI protection, and a motion version selected for the first time: creating what the new settings need is allowed here (it
    // frees nothing, so no GPU wait); false = the creation failed and nothing changed (a version created before a later failure
    // is kept, unused, for releaseUnusedMotion()).
    Settings old = m->set;
    m->set = s;
    if (!m->motionEnsure() || !m->uiEnsure()) { m->set = old; return false; }
    m->uiAdopted(old);
    // A switch of the motion version drops the estimate, the way the library already has for "no estimate" (after init/resize):
    // recordSynth / recordSynthBlur return false until the next recordEstimate, which runs the new version.
    if (s.motionVersion != old.motionVersion) m->haveEst = false;
    return true;
}

bool Interpolator::reconfigure(const Settings& s) {
    if (!m || settingsError(s)) return false;
    bool structural = s.flowScale != m->set.flowScale;
    Settings old = m->set;
    m->set = s;
    if (!m->motionEnsure()) { m->set = old; return false; }
    // re-create only if a size is held (W == 0: a later resize() creates the buffers with the new settings)
    if (structural && m->W) return m->rebuild(m->W, m->H, m->info.viewFormat, true);   // (drops the estimate anyway)
    if (!m->uiEnsure()) { m->set = old; return false; }
    m->uiAdopted(old);
    if (s.motionVersion != old.motionVersion) m->haveEst = false;   // as in setSettings
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
}

// Creates the resources of the selected motion version if it has none yet (see MotionGroup); never frees another version's.
// False = a creation failed: what this call created is freed again and the caller refuses the settings.
bool Interpolator::Impl::motionEnsure() {
    MotionGroup& g = mg();
    if (g.live) return !W || !mlearned() || mlSized();   // (Learned: a wider net than before grows the activations)
    // V1: the motion estimate is the same code for every format; the four colour passes of the estimate/synth set are the format's,
    // as are the two colour passes of the blur. V2: both modes' pipelines (see MotionGroup), the format's luma and synthesis; empty
    // slots are skipped.
    struct { Spv sp; const VkSpecializationInfo* sc; } spv[P_COUNT] = {
        {cs.luma, spec}, {SPV(down), nullptr}, {SPV(medhist), nullptr}, {SPV(search), nullptr}, {SPV(median3), nullptr},
        {SPV(subpix), nullptr}, {SPV(pixfield), nullptr}, {cs.error, spec}, {cs.resolve, spec}, {cs.fallback, spec},
        {SPV(lookup), nullptr}};
    if (set.motionVersion == MotionVersion::V2) {
        // V1's luma and pyramid; the rest is V2's own: its faster passes of V1's block field, then each mode's field and synthesis
        for (int i = P_MEDHIST; i <= P_LOOKUP; i++) spv[i].sp = Spv{};
        spv[P_V2_MEDHIST].sp = SPV(v2_medhist);
        spv[P_V2_SEARCH].sp = SPV(v2_search);
        spv[P_V2_CLEAN].sp = SPV(v2_clean);
        spv[P_V2_MORPH] = {cs.v2Morph, spec};   // (the synthesis: the format's own variant)
        spv[P_V2_REPICK].sp = SPV(v2_repick);
        spv[P_V2_QFIELD].sp = SPV(v2_qfield);
        spv[P_V2_QOWN].sp = SPV(v2_qown);
        spv[P_V2_QCOV].sp = SPV(v2_qcov);
        spv[P_V2_QMORPH] = {cs.v2QMorph, spec};
        spv[P_V2_STATIC].sp = SPV(v2_static);
        spv[P_V2_ENCLOSE].sp = SPV(v2_enclose);
        spv[P_V2_MORPH_S] = {cs.v2MorphS, spec};
        spv[P_V2_QMORPH_S] = {cs.v2QMorphS, spec};
    }
    // Learned: V1's luma, synthesis and blur, its own passes instead of V1's search; the convs specialised per layer shape and net
    // (shaders/ml_conv.comp: GI, GO, LINEAR), and the weights buffer
    struct MlSpec { int32_t gi, go, linear; } mlSpec[kMlNets * 3]{};
    VkSpecializationMapEntry mlEnt[3] = {{20, 0, 4}, {21, 4, 4}, {22, 8, 4}};
    VkSpecializationInfo mlSi[kMlNets * 3]{};
    if (mlearned()) {
#ifdef FG_HAVE_ML
        for (int i = P_DOWN; i <= P_PIXFIELD; i++) spv[i].sp = Spv{};
        spv[P_ML_DUP] = {cs.mlDup, spec};   // (the duplicate test reads colour: the format's own variant)
        spv[P_ML_REDUCE].sp = SPV(ml_reduce);
        spv[P_ML_PYR].sp = SPV(ml_pyr);
        spv[P_ML_MATCH].sp = SPV(ml_match);
        spv[P_ML_OUT].sp = SPV(ml_out);
        for (int net = 0; net < kMlNets; net++) {
            const int gw = mlWidth((LearnedNet)net) / 4;
            const MlSpec shapes[3] = {{4, gw, 0}, {gw, gw, 0}, {gw, 1, 1}};   // 14 (padded to 16) -> w, w -> w, w -> 2 (padded to 4)
            for (int k = 0; k < 3; k++) {
                const int j = 3 * net + k;
                mlSpec[j] = shapes[k];
                mlSi[j] = {3, mlEnt, sizeof(MlSpec), &mlSpec[j]};
                spv[mlConvPipe(net, k)] = {SPV(ml_conv), &mlSi[j]};
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
    // V2's motion blur is its synthesis pass with n samples (v2Synth): no blur pipelines
    for (int i = 0; ok && set.motionVersion != MotionVersion::V2 && i < B_COUNT_; i++) ok = makePipe(bspv[i].sp, blurLayout, g.blurPipes[i], bspv[i].sc);
    ok = ok && (!W || motionSized(g));
    if (!ok) { motionFree(g); return false; }
    g.live = true;
    return true;
}

// The size-dependent buffers of one motion version, beside the shared buf[] (called while a size is held: by motionEnsure and
// createSizeDependent; destroySizeDependent frees them). V1 has none of its own; V2: V2B_*.
bool Interpolator::Impl::motionSized(MotionGroup& g) {
    if (&g == &mo[(int)MotionVersion::Learned]) return mlSized();
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

// Learned's size-dependent buffers (MLB_*) for the selected net: the state once per size, the activations grown whenever the net needs
// more groups than they hold (the outgrown buffer may still be in a pending record: mlGrave, freed with the size). False = an
// allocation failed (what this call allocated is freed; an existing smaller activation buffer stays).
bool Interpolator::Impl::mlSized() {
    MotionGroup& g = mo[(int)MotionVersion::Learned];
    const MlLayout L = mlLayout(W, H, fw, fh, set.learnedNet);
    if (g.bufs.empty()) {
        g.bufs.resize(MLB_COUNT);
        mlActGroups = 0;
        if (!allocBuf(g.bufs[MLB_STATE], L.stateBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) { g.bufs.clear(); return false; }
    }
    if (mlActGroups >= L.groups) return true;
    UiBuf nb;
    if (!allocBuf(nb, L.actBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) {
        if (!mlActGroups) { for (UiBuf& b : g.bufs) freeBuf(b); g.bufs.clear(); }
        return false;
    }
    if (g.bufs[MLB_ACT].b) mlGrave.push_back(g.bufs[MLB_ACT]);
    g.bufs[MLB_ACT] = nb;
    mlActGroups = L.groups;
    return true;
}

// Learned: the weights of both nets into mlWeights, once after it was created (CmdUpdateBuffer, at most 64 KiB per command).
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
        mlActGroups = 0;
    }
    g = MotionGroup{};
}

Format Interpolator::format() const {
    return m ? m->fmt : Format::Rgba8;
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

void Interpolator::setDebugLearned(const LearnedDebug& d) {
    if (m) m->mlDbg = d;
}

bool Interpolator::debugLearnedLayout(DebugLearnedLayout& l) const {
    if (!m || !m->W || !m->mlearned() || m->mo[(int)MotionVersion::Learned].bufs.empty()) return false;
    const Impl& I = *m;
    const MlLayout L = mlLayout(I.W, I.H, I.fw, I.fh, I.set.learnedNet);
    l.fw = (uint32_t)I.fw; l.fh = (uint32_t)I.fh; l.pw = (uint32_t)L.pw; l.ph = (uint32_t)L.ph; l.width = (uint32_t)mlWidth(I.set.learnedNet);
    const VkDeviceSize st = 0;   // the state copy (shaders/ml_common.glsl), then the activations
    VkDeviceSize o = 0;
    for (int k = 0; k < kMlLevels; k++) { l.pyrOff[k] = st + o * 4; o += 2 * (VkDeviceSize)(L.pw >> k) * (L.ph >> k); }
    for (int k = 0; k < kMlLevels; k++) { l.hOff[k] = st + o * 4; o += 2 * (VkDeviceSize)(L.pw >> k) * (L.ph >> k); }
    l.actOff[0] = (st + L.stateBytes + 15) & ~(VkDeviceSize)15;
    l.actOff[1] = l.actOff[0] + (VkDeviceSize)L.r0 * L.n0 * 16;
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
    const MlLayout M = mlLayout(I.W, I.H, I.fw, I.fh, I.set.learnedNet);
    VkBufferCopy sc{0, dstOffset + L.pyrOff[0], M.stateBytes};
    I.CmdCopyBuffer(cb, g.bufs[MLB_STATE].b, dst, 1, &sc);
    VkBufferCopy ac{0, dstOffset + L.actOff[0], M.actBytes};
    I.CmdCopyBuffer(cb, g.bufs[MLB_ACT].b, dst, 1, &ac);
    return true;
}

VkDescriptorSet Interpolator::Impl::takeSet(int ring, Frame prev, Frame curr, Frame out, bool blur, RejPair rej) {
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
    VkDescriptorBufferInfo vb = {v2g.bufs.empty() ? VK_NULL_HANDLE : v2g.bufs[V2B_FIELD].b, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo qb = {v2g.bufs.empty() ? VK_NULL_HANDLE : v2g.bufs[V2B_QUAL].b, 0, VK_WHOLE_SIZE};
    const MotionGroup& mlg = mo[(int)MotionVersion::Learned];
    VkDescriptorBufferInfo mb[3] = {{mlg.bufs.empty() ? VK_NULL_HANDLE : mlg.bufs[MLB_STATE].b, 0, VK_WHOLE_SIZE},
                                    {mlg.bufs.empty() ? VK_NULL_HANDLE : mlg.bufs[MLB_ACT].b, 0, VK_WHOLE_SIZE},
                                    {mlWeights.b, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wr[16]{};
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
    if (this->set.motionVersion == MotionVersion::V2 && vb.buffer) {   // V2's sets only (V1's stay as they were)
        add(13, BUF).pBufferInfo = &vb;
        add(14, BUF).pBufferInfo = &qb;
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

bool Interpolator::recordObserve(VkCommandBuffer cb, Frame frame) {
    if (!m || !m->W) return false;
    Impl& I = *m;
    if (!I.framesOk(frame)) return false;
    if (I.set.uiProtection == UiProtection::Off) return true;
    if (!I.uiActive()) return false;
    I.uiExplicit = true;
    if (I.uiFrameSets + I.uiObserveSets() > kUiSets) return false;
    if (I.profiling && I.qpool && !I.profPending) {   // an observe opens a profile that the next recordEstimate continues
        I.qcount = 0;
        I.qnames.clear();
        I.CmdResetQueryPool(cb, I.qpool, 0, Impl::kMaxQ);
        I.CmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, I.qpool, I.qcount++);
        I.qnames.push_back("start");
    }
    I.profPending = I.profiling && I.qpool;
    return I.uiObserveRecord(cb, frame);
}

bool Interpolator::recordTrack(VkCommandBuffer cb, Frame frame) {
    if (!m || !m->W) return false;
    Impl& I = *m;
    if (!I.framesOk(frame)) return false;
    if (I.set.uiProtection == UiProtection::Off) return true;
    if (!I.uiActive()) return false;
    if (I.uiFrameSets + I.uiObserveSets() > kUiSets) return false;
    return I.uiTrackRecord(cb, frame);
}

bool Interpolator::recordEstimate(VkCommandBuffer cb, Frame prev, Frame curr) {
    if (!m || !m->W) return false;
    Impl& I = *m;
    if (!I.framesOk(prev) || !I.framesOk(curr)) return false;
    // The next set in the ring may still be referenced by an earlier record of this submission.
    assert(I.frameSets[Impl::R_EST] < kSetRing && "framegen: more than kSetRing recordEstimate() since beginFrame()");
    if (I.frameSets[Impl::R_EST] >= kSetRing) return false;
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
        // V1's block field, faster (shaders/v2_est.glsl): no median3 passes, each consumer takes the 3x3 median of the raw search
        // output itself; medhist's histogram is weighted per parent and scanned in parallel
        for (int i = n - 1; i >= 0; i--) {
            const Level& l = I.lv[i];
            pc.w = l.w; pc.h = l.h; pc.bw = l.bw; pc.bh = l.bh;
            pc.offA = l.offA; pc.offC = l.offC;
            if (i == n - 1) {
                pc.medIdx = -1;
                pc.radius = 6;   // TOP_RADIUS
            } else {
                const Level& p = I.lv[i + 1];
                pc.pw = p.bw; pc.ph = p.bh; pc.fPar = p.fS;   // the parent's RAW search output
                pc.medIdx = 4 + 2 * i;
                pc.radius = 2;   // RADIUS
                r.run(P_V2_MEDHIST, 1, 1);
                r.cbar();
            }
            pc.fOut = l.fS;
            // finest level (bits): 1 = V1's scene-cut sum (sub-pixel refinement off: the search's SAD is final), 2 = Performance's
            // block field (V2B_FIELD at bw * bh) and its own scene-cut sum, 4 = with the equiangular sub-pixel fit
            pc.sadAcc = i == 0 ? (!I.set.subpixel ? 1 : 0) | 2 | (I.set.subpixel ? 4 : 0) : 0;
            r.run(P_V2_SEARCH, l.bw, l.bh);
            r.cbar();
            r.stamp(lvNames[std::min(i, 7)]);
        }
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
            pc.radius = 6;   // TOP_RADIUS
        } else {
            const Level& p = I.lv[i + 1];
            pc.pw = p.bw; pc.ph = p.bh; pc.fPar = p.fM;
            pc.medIdx = 4 + 2 * i;
            pc.radius = 2;   // RADIUS
            r.run(P_MEDHIST, 1, 1);
            r.cbar();
        }
        pc.fOut = l.fS;
        pc.sadAcc = i == 0 && !I.set.subpixel;   // without sub-pixel refinement the search's SAD is final
        r.run(P_SEARCH, l.bw, l.bh);
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
    const MlLayout L = mlLayout(W, H, fw, fh, set.learnedNet);
    const int net = (int)set.learnedNet;
    const uint32_t base = net == 0 ? 0u : (uint32_t)(kMlNet[0].words / 4);   // the net's start in mlWeights (vec4 units)
    const int r1 = L.r0 * L.n0;                                             // activation region R1 (vec4 offset)
    const Level& l0 = lv[0];
    MLPC m{};
    m.W = (int)W; m.H = (int)H; m.fw = fw; m.fh = fh; m.pw = L.pw; m.ph = L.ph;
    m.offYa = l0.offA; m.offYc = l0.offC; m.bw = l0.bw; m.bh = l0.bh; m.pf = fieldPF; m.dupDiff = kMlDupDiff;
    auto run = [&](Pipe p, uint32_t gx, uint32_t gy) {
        CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, mg().pipes[p]);
        CmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MLPC), &m);
        CmdDispatch(cb, gx, gy, 1);
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
        m.lvl = lvl;
        m.actOut = 0;   // the 14-channel input -> R0
        run(P_ML_MATCH, gx, gy);
        r.cbar();
        if (stop(lvl, 0)) return;
        for (int k = 0; k < kMlLayers; k++) {   // R0 -> R1 -> R0 -> R1 -> R0 -> the field
            m.wOff = (int)(base + kMlNet[net].table[lvl][k][0]);
            m.bOff = (int)(base + kMlNet[net].table[lvl][k][1]);
            m.actIn = k % 2 ? r1 : 0;
            m.actOut = k % 2 ? 0 : r1;
            run(mlConvPipe(net, k == 0 ? 0 : k == kMlLayers - 1 ? 2 : 1), gx, gy);
            r.cbar();
            if (stop(lvl, k + 1)) return;
        }
        r.stamp(lvNames[lvl]);
    }
    // the field V1's synthesis reads, and the scene-cut sum
    m.nPart = L.nOut;
    run(P_ML_OUT, divUp(fw, 16), divUp(fh, 16));
    r.cbar();
    m.mode = 1;
    run(P_ML_REDUCE, 1, 1);
    r.cbar();
    r.stamp("net output");
#else
    (void)cb;
#endif
}

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
    r.runV2(P_V2_QOWN, v, divUp(fw, 8), divUp(fh, 8));
    r.cbar();
    r.stamp("splat");
    r.runV2(P_V2_QCOV, v, divUp(fw, 8), divUp(fh, 8));
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
        r.barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, VK_ACCESS_SHADER_READ_BIT);   // compute-only queue safe
        r.stamp("ui_patch");
        return true;
    }
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
        if (I.uiFrameSets + 2 > kUiSets) return false;
        if (I.estClean) { srcP = ep.frame(); srcC = ec.frame(); }
    }
    if (I.mv2()) return I.v2Synth(cb, srcP, srcC, curr, out, &t, 1, false);
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
    if (I.estUi) {
        r.barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, RW);
        r.stamp("fallback");
        I.uiRecordPatch(cb, curr, out, I.uiE[I.estEnt[1]]);   // the real UI pixels back over the generated frame
        r.barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, VK_ACCESS_SHADER_READ_BIT);   // compute-only queue safe
        r.stamp("ui_patch");
        return true;
    }
    r.barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, VK_ACCESS_SHADER_READ_BIT);   // compute-only queue safe
    r.stamp("fallback");
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
        if (I.uiFrameSets + 2 > kUiSets) return false;
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
    if (I.estUi) {
        r.barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, RW);
        r.stamp("blur fixup");
        I.uiRecordPatch(cb, curr, out, I.uiE[I.estEnt[1]]);   // the real UI pixels back over the blurred frame
        r.barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, VK_ACCESS_SHADER_READ_BIT);
        r.stamp("ui_patch");
        return true;
    }
    r.barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, VK_ACCESS_SHADER_READ_BIT);
    r.stamp("blur fixup");
    return true;
}

bool Interpolator::record(VkCommandBuffer cb, Frame prev, Frame curr, Frame out, float t) {
    if (!m || !m->W || !(t > 0.0f && t < 1.0f)) return false;
    // both rings must have room, so that a refused record() records nothing
    assert(m->frameSets[0] < kSetRing && m->frameSets[1] < kSetRing && "framegen: more than kSetRing records since beginFrame()");
    if (m->frameSets[Impl::R_EST] >= kSetRing || m->frameSets[Impl::R_SYNTH] >= kSetRing) return false;
    if (m->uiActive() && m->uiFrameSets + 2 * m->uiObserveSets() + 2 + (m->v2Fill() ? 2 : 0) > kUiSets) return false;   // 2 implicit observes + the patch (+ V2's two luma fills)
    return recordEstimate(cb, prev, curr) && recordSynth(cb, prev, curr, out, t);
}

}  // namespace framegen

