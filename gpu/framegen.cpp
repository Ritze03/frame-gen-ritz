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
struct ColourSpv { Spv luma, error, resolve, fallback, uiDetect, uiBoxUpdate, uiInpaint, uiPatch, blurError, blurResolve, uiInpaintC, uiPatchC, uiV2Update, uiV2Fill; };
#define COLOUR_SET(f) ColourSpv{SPV(luma##f), SPV(error##f), SPV(resolve##f), SPV(fallback##f), SPV(ui_detect##f), \
    SPV(ui_box_update##f), SPV(ui_inpaint##f), SPV(ui_patch##f), SPV(blur_error##f), SPV(blur_resolve##f), Spv{}, Spv{}, \
    SPV(ui_v2_update##f), SPV(ui_v2_lumafill##f)}
// false = this format's variants were not built into the library (meson option `formats`)
bool colourSpv(Format f, ColourSpv& c) {
    switch (f) {
    case Format::Rgba8: c = COLOUR_SET(); return true;
#ifdef FG_HAVE_RGB10
    case Format::Rgb10: c = COLOUR_SET(_rgb10); return true;
#endif
#ifdef FG_HAVE_F16
    case Format::RgbaF16: c = COLOUR_SET(_f16); return true;
#endif
#ifdef FG_HAVE_NV12
    case Format::Nv12: c = COLOUR_SET(_nv12); c.uiInpaintC = SPV(ui_inpaint_chroma_nv12); c.uiPatchC = SPV(ui_patch_chroma_nv12); return true;
#endif
#ifdef FG_HAVE_P010
    case Format::P010: c = COLOUR_SET(_p010); c.uiInpaintC = SPV(ui_inpaint_chroma_p010); c.uiPatchC = SPV(ui_patch_chroma_p010); return true;
#endif
    default: return false;
    }
}
inline bool isYuv(Format f) { return f == Format::Nv12 || f == Format::P010; }
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
    X(CreateImageView) X(DestroyImageView) X(CmdCopyBuffer)

// Must match the push-constant block in shaders/motion_v1.glsl.
struct PC {
    int32_t w, h, bw, bh, pw, ph, offA, offC, srcA, srcC, fIn, fOut, fPar, radius, medIdx,
        W, H, fw, fh, pf, fscale, sadAcc;
    float s, wA, wC;
    float lambda, smoothBonus, zeroBonus, cutSad, trust0, trustScale, globalFrac;
};
static_assert(sizeof(PC) <= 128, "push constants beyond the guaranteed minimum");
static const int kEOff = 64;   // == E_OFF in motion_v1.glsl

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

enum Pipe { P_LUMA, P_DOWN, P_MEDHIST, P_SEARCH, P_MEDIAN3, P_SUBPIX, P_PIXFIELD, P_ERROR, P_RESOLVE, P_FALLBACK, P_LOOKUP, P_COUNT };

struct Level { int w, h, bw, bh, offA, offC, fS, fM; };

// The resources of one motion version (Settings::motionVersion): created by motionEnsure() the first time the version is selected
// (init, setSettings, reconfigure), kept while another one is selected, freed by releaseUnusedMotion() / destroy(). V1: the pipelines
// of the estimate / synth passes and of the motion-blur passes over the shared layouts, set rings and scratch buffers. V2 aliases V1
// for now: its group holds its own pipelines of V1's shaders, so the switch, the release and the re-creation are the real ones; its
// own shaders (v2_*.comp) replace them, and its own size-dependent buffers go into `bufs` (created by motionSized()).
static const int kMotionVersions = 2;   // MotionVersion::V1, V2
struct MotionGroup {
    VkPipeline pipes[P_COUNT]{};
    VkPipeline blurPipes[B_COUNT_]{};
    std::vector<UiBuf> bufs;   // size-dependent buffers of this version only (none yet), freed with the size like buf[]
    bool live = false;         // created (the pipelines; the buffers whenever a size is held)
};

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
    bool allocBuf(UiBuf& b, VkDeviceSize bytes);
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

static int rintHalfEven(double v) { return (int)std::nearbyint(v); }   // == cv2 saturate_cast<int>

bool Interpolator::Impl::makeBuffer(int i, VkDeviceSize size) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (i == 2) bci.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;   // the counters: recordDebugCounters() copies them out
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
    // interp.flow(): luma at 1/FLOW_SCALE (cv2 rounds the size), pyramid down to MIN_SIDE.
    fw = rintHalfEven(W * (1.0 / set.flowScale));
    fh = rintHalfEven(H * (1.0 / set.flowScale));
    lv.clear();
    int w = fw, h = fh, lumaOff = 0, fieldOff = 0;
    for (;;) {
        Level l{};
        l.w = w; l.h = h;
        l.bw = (w + 7) / 8; l.bh = (h + 7) / 8;   // pad_to_block
        l.offA = lumaOff; l.offC = lumaOff + w * h;
        lumaOff += 2 * w * h;
        l.fS = fieldOff; l.fM = fieldOff + l.bw * l.bh;
        fieldOff += 2 * l.bw * l.bh;
        lv.push_back(l);
        if (std::min(w, h) / 2 < 48) break;
        w = rintHalfEven(w * 0.5);
        h = rintHalfEven(h * 0.5);
    }
    fieldSub = fieldOff;
    fieldOff += lv[0].bw * lv[0].bh;
    fieldPF = fieldOff;
    fieldOff += fw * fh;
    fieldT = fieldOff;   // per-pixel field looked up at time t (synth scratch, t != 0.5)
    fieldOff += fw * fh;
    int uCount = kEOff + (int)(W + 3) / 4 * (int)H;   // counters/medians, then the packed mismatch
    if (!makeBuffer(0, (VkDeviceSize)lumaOff * 4)) return false;
    if (!makeBuffer(1, (VkDeviceSize)fieldOff * 8)) return false;
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

bool Interpolator::Impl::allocBuf(UiBuf& b, VkDeviceSize bytes) {
    b = UiBuf{};
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
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
    // YUV: 8-9 the CbCr planes of prev/curr, 10 the CbCr plane of out; 11-12 (synth sets): the V2 masks of prev / curr (ui_reject.glsl)
    uint32_t nb = m->yuv ? 11 : 8;
    VkDescriptorSetLayoutBinding b[13]{};
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
    for (uint32_t k = 11; k <= 12; k++, nb++) {
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

    if (!resize(ii.width, ii.height, ii.viewFormat)) { destroy(); err_ = "resize failed: size below kMinSize or an allocation failed"; return false; }
    return true;
}

bool Interpolator::resize(uint32_t w, uint32_t h, VkFormat fmt) {
    return m && m->rebuild(w, h, fmt, false);
}

bool Interpolator::Impl::rebuild(uint32_t w, uint32_t h, VkFormat fmt, bool force) {
    // Invariant: W != 0 <=> all size-dependent buffers exist. Any failure leaves W = H = 0.
    // The pipelines belong to the Format chosen at init(): a view format of another Format is refused (BGRA <-> RGBA is fine).
    // Below kMinSize the flow resolution (W/2 x H/2, rounded) degenerates (W or H = 1 -> size-0 buffers).
    Format nf;
    bool ok = formatOf(fmt, &nf) && nf == this->fmt && w >= kMinSize && h >= kMinSize;
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
           (uint32_t)s.uiProtection <= (uint32_t)UiProtection::CrosshairV2 && (uint32_t)s.motionVersion <= (uint32_t)MotionVersion::V2 &&
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
    if (s.motionVersion == MotionVersion::V2 && f != Format::Rgba8)
        return "MotionVersion::V2 supports only Format::Rgba8 (8-bit RGBA / BGRA views); Rgb10, RgbaF16 (encoded or linear), Nv12 and P010 need MotionVersion::V1";
    return nullptr;
}

bool Interpolator::setSettings(const Settings& s) {
    if (!m || settingsError(s, m->fmt) || s.flowScale != m->set.flowScale) return false;
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
    if (!m || settingsError(s, m->fmt)) return false;
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
    if (g.live) return true;
    // V1 (and V2 while it aliases V1): the motion estimate is the same code for every format; the four colour passes of the
    // estimate/synth set are the format's, as are the two colour passes of the blur
    struct { Spv sp; const VkSpecializationInfo* sc; } spv[P_COUNT] = {
        {cs.luma, spec}, {SPV(down), nullptr}, {SPV(medhist), nullptr}, {SPV(search), nullptr}, {SPV(median3), nullptr},
        {SPV(subpix), nullptr}, {SPV(pixfield), nullptr}, {cs.error, spec}, {cs.resolve, spec}, {cs.fallback, spec},
        {SPV(lookup), nullptr}};
    struct { Spv sp; const VkSpecializationInfo* sc; } bspv[B_COUNT_] = {
        {SPV(blur_lookup), nullptr}, {cs.blurError, spec}, {cs.blurResolve, spec}};
    bool ok = true;
    for (int i = 0; ok && i < P_COUNT; i++) ok = makePipe(spv[i].sp, layout, g.pipes[i], spv[i].sc);
    for (int i = 0; ok && i < B_COUNT_; i++) ok = makePipe(bspv[i].sp, blurLayout, g.blurPipes[i], bspv[i].sc);
    ok = ok && (!W || motionSized(g));
    if (!ok) { motionFree(g); return false; }
    g.live = true;
    return true;
}

// The size-dependent buffers of one motion version, beside the shared buf[] (called while a size is held: by motionEnsure and
// createSizeDependent; destroySizeDependent frees them). V1 and today's V2 have none of their own.
bool Interpolator::Impl::motionSized(MotionGroup& g) {
    (void)g;
    return true;
}

// Frees one motion version's resources (retire rule of resize()/destroy()).
void Interpolator::Impl::motionFree(MotionGroup& g) {
    for (VkPipeline p : g.pipes) if (p) DestroyPipeline(info.device, p, nullptr);
    for (VkPipeline p : g.blurPipes) if (p) DestroyPipeline(info.device, p, nullptr);
    for (UiBuf& b : g.bufs) freeBuf(b);
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
    VkBufferCopy c{0, dstOffset, 8};   // u[U_UNTRUSTED], u[U_SADSUM]
    I.CmdCopyBuffer(cb, I.buf[2], dst, 1, &c);
    return true;
}

uint32_t Interpolator::debugSadBlocks() const {
    return m && !m->lv.empty() ? (uint32_t)(m->lv[0].bw * m->lv[0].bh) : 0;
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
    VkWriteDescriptorSet wr[13]{};
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
    // WAR/WAW against every earlier record's use of our buffers, then zero the first `words` of u[].
    void clearCounters(uint32_t words, VkDescriptorSet set) {
        barrier(CS, RW, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        I.CmdFillBuffer(cb, I.buf[2], 0, words * 4, 0);
        barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, CS, RW);
        I.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, I.layout, 0, 1, &set, 0, nullptr);
    }
    void run(Pipe p, uint32_t gx, uint32_t gy) {
        I.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, I.mg().pipes[p]);
        I.CmdPushConstants(cb, I.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC), &pc);
        I.CmdDispatch(cb, gx, gy, 1);
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
    r.clearCounters(64, set);   // untrusted count, SAD sum, per-level medians
    PC& pc = r.pc;

    // 1. luma + 2x downsample
    const Level& l0 = I.lv[0];
    pc.w = l0.w; pc.h = l0.h; pc.offA = l0.offA; pc.offC = l0.offC;
    int lt = I.set.flowScale == 4 ? 4 : 8;   // flow pixels per workgroup side (luma.comp main4 for fscale 4)
    r.run(P_LUMA, divUp(l0.w, lt), divUp(l0.h, lt));
    r.cbar();
    r.stamp("luma");
    if (ui && I.v2Fill()) {   // CrosshairV2, uiEstimateFill: the UI pixels of both frames are missing for the estimate (luma only, never the colour passes)
        I.uiRecordLumaFill(cb, prev, I.uiE[ep], l0.offA);
        I.uiRecordLumaFill(cb, curr, I.uiE[ec], l0.offC);
        r.cbar();
        I.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, I.layout, 0, 1, &set, 0, nullptr);   // the fills bound the UI set
        r.stamp("ui_fill");
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

