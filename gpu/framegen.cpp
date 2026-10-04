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

namespace framegen {

#define FG_FUNCS(X) \
    X(CreateBuffer) X(DestroyBuffer) X(GetBufferMemoryRequirements) X(AllocateMemory) X(FreeMemory) \
    X(BindBufferMemory) X(CreateSampler) X(DestroySampler) X(CreateDescriptorSetLayout) \
    X(DestroyDescriptorSetLayout) X(CreatePipelineLayout) X(DestroyPipelineLayout) X(CreateShaderModule) \
    X(DestroyShaderModule) X(CreateComputePipelines) X(DestroyPipeline) X(CreateDescriptorPool) \
    X(DestroyDescriptorPool) X(AllocateDescriptorSets) X(UpdateDescriptorSets) X(CreateQueryPool) \
    X(DestroyQueryPool) X(GetQueryPoolResults) X(CmdBindPipeline) X(CmdBindDescriptorSets) \
    X(CmdPushConstants) X(CmdDispatch) X(CmdPipelineBarrier) X(CmdFillBuffer) X(CmdResetQueryPool) \
    X(CmdWriteTimestamp) X(CreateImage) X(DestroyImage) X(GetImageMemoryRequirements) X(BindImageMemory) \
    X(CreateImageView) X(DestroyImageView)

// Must match the push-constant block in shaders/common.glsl.
struct PC {
    int32_t w, h, bw, bh, pw, ph, offA, offC, srcA, srcC, fIn, fOut, fPar, radius, medIdx,
        W, H, fw, fh, pf, fscale, sadAcc;
    float s, wA, wC;
    float lambda, smoothBonus, zeroBonus, cutSad, trust0, trustScale, globalFrac;
};
static_assert(sizeof(PC) <= 128, "push constants beyond the guaranteed minimum");
static const int kEOff = 64;   // == E_OFF in common.glsl

// Must match the push-constant block in shaders/ui_common.glsl.
struct UiPC {
    int32_t W, H, boxX, boxY, boxW, boxH, cell, cw, ch, wholeScreen, still;
    float thr;
    int32_t reach, reset, tilesX, tilesY, stillLimit, maskX, maskY, maskW, maskH, maskStride, maskOff, gate;
};
static_assert(sizeof(UiPC) <= 128, "push constants beyond the guaranteed minimum");
enum UiPipe { U_DETECT, U_BOX_UPDATE, U_BOX_MASK, U_MASK, U_INPAINT, U_PATCH, U_COUNT };
static const int kUiCell = 2;    // whole-screen detection resolution: 1/kUiCell per axis (1, 2 or 4; 8 x cell divides the tile)
static const int kUiTile = 32;   // == TILE in ui_common.glsl
static const int kUiInpTile = 16;   // == T in ui_inpaint.comp

struct UiBuf { VkBuffer b = VK_NULL_HANDLE; VkDeviceMemory m = VK_NULL_HANDLE; VkDeviceSize size = 0; };
struct UiBox { int x, y, w, h; bool operator==(const UiBox& o) const { return x == o.x && y == o.y && w == o.w && h == o.h; } };
// One observed real frame: the clean copy, the mask of the box (full res) and, whole screen, of the whole frame.
struct UiEntry {
    VkImage img = VK_NULL_HANDLE; VkDeviceMemory imgMem = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE;
    UiBuf box;   // header (4 words) + the box mask; grown lazily when the box grows
    UiBuf ws;    // header + tile flags + the full-frame mask (allocated with WholeScreen)
    VkImageView key = VK_NULL_HANDLE;   // the host view this entry describes (null = dead)
    uint64_t seq = 0;                   // observe order (LRU, and "is this still the frame the estimate saw")
    UiBox bx{};                         // the box at observe time
    bool hasBox = false, hasWs = false;
};

enum Pipe { P_LUMA, P_DOWN, P_MEDHIST, P_SEARCH, P_MEDIAN3, P_SUBPIX, P_PIXFIELD, P_ERROR, P_RESOLVE, P_FALLBACK, P_LOOKUP, P_COUNT };

struct Level { int w, h, bw, bh, offA, offC, fS, fM; };

struct Interpolator::Impl {
#define X(n) PFN_vk##n n = nullptr;
    FG_FUNCS(X)
#undef X
    InitInfo info{};
    Settings set{};   // active settings (info.settings is only the init value)
    VkPhysicalDeviceMemoryProperties memProps{};

    VkSampler sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipes[P_COUNT]{};
    VkDescriptorPool pool = VK_NULL_HANDLE;
    // Two rings: synth sets (all 6 bindings) and estimate sets (binding 5, out, never written: no estimate
    // pass uses it, and no stale view of an earlier synth can linger in it).
    enum { R_SYNTH, R_EST };
    VkDescriptorSet sets[2][kSetRing]{};
    uint32_t setIdx[2]{};
    uint32_t frameSets[2]{};     // sets taken from each ring since the last beginFrame()
    // latest recordEstimate (recordSynth uses its results)
    bool haveEst = false;
    VkImageView estPrev = VK_NULL_HANDLE, estCurr = VK_NULL_HANDLE;

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
    VkPipeline uiPipes[U_COUNT]{};
    VkDescriptorPool uiPool = VK_NULL_HANDLE;
    VkDescriptorSet uiSets[kUiSets]{};
    uint32_t uiSetIdx = 0, uiFrameSets = 0;
    UiEntry uiE[kUiFrames];
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
    int estEnt[2]{-1, -1};
    uint64_t estSeq[2]{};
    bool profPending = false;        // an observe opened the current profile and no estimate has continued it yet

    bool uiActive() const { return W && set.uiProtection != UiProtection::Off && uiImgs && (set.uiProtection != UiProtection::WholeScreen || uiWsRes); }
    uint32_t uiObserveSets() const { return set.uiProtection == UiProtection::WholeScreen ? 2 : 1; }
    UiBox computeBox() const;
    uint32_t localType(uint32_t bits) const;
    bool allocBuf(UiBuf& b, VkDeviceSize bytes);
    void freeBuf(UiBuf& b);
    bool growBuf(UiBuf& b, VkDeviceSize bytes);
    bool allocEntryImage(UiEntry& e);
    bool uiEnsure();
    void uiDestroy();
    void uiAdopted(const Settings& old);
    int uiFind(VkImageView v) const;
    int uiPick(VkImageView v) const;
    VkDescriptorSet takeUiSet(VkImageView src, VkImageView img, const UiBuf& s, const UiBuf& m);
    bool uiObserveRecord(VkCommandBuffer cb, VkImageView view);
    void uiRecordPatch(VkCommandBuffer cb, VkImageView curr, VkImageView out, const UiEntry& e);

    // profiling
    bool profiling = false;
    VkQueryPool qpool = VK_NULL_HANDLE;
    static constexpr uint32_t kMaxQ = 64;
    std::vector<const char*> qnames;
    uint32_t qcount = 0;

    VkDescriptorSet takeSet(int ring, VkImageView prev, VkImageView curr, VkImageView out);
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
    return true;
}

void Interpolator::Impl::destroySizeDependent() {
    for (int i = 0; i < 3; i++) {
        if (buf[i]) DestroyBuffer(info.device, buf[i], nullptr);
        if (mem[i]) FreeMemory(info.device, mem[i], nullptr);
        buf[i] = VK_NULL_HANDLE; mem[i] = VK_NULL_HANDLE;
    }
    uiDestroy();
}

// ---------------------------------------------------------------------------------------------------------
// Static-UI protection: resources
// ---------------------------------------------------------------------------------------------------------

// The crosshair box: explicit pixel sizes win over the area fraction; clamped into the frame.
UiBox Interpolator::Impl::computeBox() const {
    const int fw_ = (int)W, fh_ = (int)H;
    int side = (int)std::lround(std::sqrt((double)set.uiCrosshairArea * fw_ * fh_));
    side = std::min(std::max(side, 32), std::min(fw_, fh_));
    UiBox b{};
    b.w = set.uiBoxWidth ? std::min(std::max((int)set.uiBoxWidth, 4), fw_) : side;
    b.h = set.uiBoxHeight ? std::min(std::max((int)set.uiBoxHeight, 4), fh_) : side;
    b.x = std::min(std::max((fw_ - b.w) / 2 + set.uiBoxOffsetX, 0), fw_ - b.w);
    b.y = std::min(std::max((fh_ - b.h) / 2 + set.uiBoxOffsetY, 0), fh_ - b.h);
    return b;
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

bool Interpolator::Impl::allocEntryImage(UiEntry& e) {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8B8A8_UNORM;   // always RGBA8, in logical channel order: only the library reads and writes it
    ici.extent = {W, H, 1};
    ici.mipLevels = ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (CreateImage(info.device, &ici, nullptr, &e.img) != VK_SUCCESS) { e.img = VK_NULL_HANDLE; return false; }
    VkMemoryRequirements req;
    GetImageMemoryRequirements(info.device, e.img, &req);
    uint32_t type = localType(req.memoryTypeBits);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    if (type == UINT32_MAX || AllocateMemory(info.device, &mai, nullptr, &e.imgMem) != VK_SUCCESS) { e.imgMem = VK_NULL_HANDLE; return false; }
    if (BindImageMemory(info.device, e.img, e.imgMem, 0) != VK_SUCCESS) return false;
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = e.img;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R8G8B8A8_UNORM;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (CreateImageView(info.device, &vci, nullptr, &e.view) != VK_SUCCESS) { e.view = VK_NULL_HANDLE; return false; }
    uiPendingInit.push_back(e.img);
    return true;
}

// Creates what the active mode needs and does not have yet; never frees. Called by setSettings/reconfigure (the new
// settings already in `set`) and by (re)creation of the size-dependent resources. The per-box buffers are created
// lazily by the observe that needs them. False = could not allocate (the caller refuses the settings).
bool Interpolator::Impl::uiEnsure() {
    if (!W || set.uiProtection == UiProtection::Off) return true;
    if (!uiImgs) {
        for (UiEntry& e : uiE)
            if (!allocEntryImage(e)) { uiDestroy(); return false; }
        uiImgs = true;
    }
    if (set.uiProtection == UiProtection::WholeScreen && !uiWsRes) {
        const VkDeviceSize wsWords = 4 + (VkDeviceSize)tilesX * tilesY + (VkDeviceSize)((W + 3) / 4) * H;
        bool ok = allocBuf(uiCellState, (VkDeviceSize)cw * ch * 4);
        for (UiEntry& e : uiE) ok = ok && allocBuf(e.ws, wsWords * 4);
        if (!ok) {
            freeBuf(uiCellState);
            for (UiEntry& e : uiE) freeBuf(e.ws);
            return false;
        }
        uiWsRes = true;
    }
    return true;
}

// Frees every UI resource. Retire rule of resize()/destroy(): all earlier records must have retired.
void Interpolator::Impl::uiDestroy() {
    VkDevice d = info.device;
    for (UiEntry& e : uiE) {
        if (e.view) DestroyImageView(d, e.view, nullptr);
        if (e.img) DestroyImage(d, e.img, nullptr);
        if (e.imgMem) FreeMemory(d, e.imgMem, nullptr);
        freeBuf(e.box);
        freeBuf(e.ws);
        e = UiEntry{};
    }
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
}

int Interpolator::Impl::uiFind(VkImageView v) const {
    for (int i = 0; i < (int)kUiFrames; i++)
        if (v && uiE[i].key == v) return i;
    return -1;
}

// The entry to observe `v` into: the one already keyed by it, else the least recently observed.
int Interpolator::Impl::uiPick(VkImageView v) const {
    int f = uiFind(v);
    if (f >= 0) return f;
    int best = 0;
    for (int i = 1; i < (int)kUiFrames; i++)
        if (uiE[i].seq < uiE[best].seq) best = i;
    return best;
}

VkDescriptorSet Interpolator::Impl::takeUiSet(VkImageView src, VkImageView img, const UiBuf& s, const UiBuf& m) {
    VkDescriptorSet set_ = uiSets[uiSetIdx];
    uiSetIdx = (uiSetIdx + 1) % kUiSets;
    uiFrameSets++;
    VkDescriptorImageInfo ii[2] = {{VK_NULL_HANDLE, src, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, img, VK_IMAGE_LAYOUT_GENERAL}};
    VkDescriptorBufferInfo bi[2] = {{s.b, 0, VK_WHOLE_SIZE}, {m.b, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wr[4]{};
    for (int i = 0; i < 4; i++) {
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
    UpdateDescriptorSets(info.device, 4, wr, 0, nullptr);
    return set_;
}

bool Interpolator::init(const InitInfo& ii) {
    destroy();
    if (!ii.getMemProps || !ii.gdpa || !ii.device || !validate(ii.settings)) return false;
    m = new Impl();
    m->info = ii;
    m->set = ii.settings;
    VkDevice d = ii.device;
#define X(n) m->n = (PFN_vk##n)ii.gdpa(d, "vk" #n); if (!m->n) { destroy(); return false; }
    FG_FUNCS(X)
#undef X
    ii.getMemProps(ii.phys, &m->memProps);

    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod = 0.0f;
    if (m->CreateSampler(d, &sci, nullptr, &m->sampler) != VK_SUCCESS) { destroy(); return false; }

    VkDescriptorSetLayoutBinding b[6]{};
    VkDescriptorType types[6] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
    for (int i = 0; i < 6; i++) {
        b[i].binding = i;
        b[i].descriptorType = types[i];
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        if (i == 3 || i == 4) b[i].pImmutableSamplers = &m->sampler;
    }
    VkDescriptorSetLayoutCreateInfo dci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dci.bindingCount = 6;
    dci.pBindings = b;
    if (m->CreateDescriptorSetLayout(d, &dci, nullptr, &m->dsl) != VK_SUCCESS) { destroy(); return false; }

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &m->dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    if (m->CreatePipelineLayout(d, &plci, nullptr, &m->layout) != VK_SUCCESS) { destroy(); return false; }

    struct { const uint32_t* code; size_t size; } spv[P_COUNT] = {
        {luma_spv, sizeof(luma_spv)}, {down_spv, sizeof(down_spv)}, {medhist_spv, sizeof(medhist_spv)},
        {search_spv, sizeof(search_spv)}, {median3_spv, sizeof(median3_spv)}, {subpix_spv, sizeof(subpix_spv)},
        {pixfield_spv, sizeof(pixfield_spv)}, {error_spv, sizeof(error_spv)}, {resolve_spv, sizeof(resolve_spv)},
        {fallback_spv, sizeof(fallback_spv)}, {lookup_spv, sizeof(lookup_spv)}};
    for (int i = 0; i < P_COUNT; i++) {
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = spv[i].size;
        smci.pCode = spv[i].code;
        VkShaderModule sm;
        if (m->CreateShaderModule(d, &smci, nullptr, &sm) != VK_SUCCESS) { destroy(); return false; }
        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", nullptr};
        cpci.layout = m->layout;
        VkResult r = m->CreateComputePipelines(d, ii.pipelineCache, 1, &cpci, nullptr, &m->pipes[i]);
        m->DestroyShaderModule(d, sm, nullptr);
        if (r != VK_SUCCESS) { destroy(); return false; }
    }

    VkDescriptorPoolSize ps[3] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * 2 * kSetRing},
                                  {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 * 2 * kSetRing},
                                  {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * kSetRing}};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 2 * kSetRing;
    dpci.poolSizeCount = 3;
    dpci.pPoolSizes = ps;
    if (m->CreateDescriptorPool(d, &dpci, nullptr, &m->pool) != VK_SUCCESS) { destroy(); return false; }
    VkDescriptorSetLayout layouts[2 * kSetRing];
    for (auto& l : layouts) l = m->dsl;
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = m->pool;
    dsai.descriptorSetCount = 2 * kSetRing;
    dsai.pSetLayouts = layouts;
    if (m->AllocateDescriptorSets(d, &dsai, &m->sets[0][0]) != VK_SUCCESS) { destroy(); return false; }

    // static-UI protection: own set layout (sampled frame, storage image, 2 storage buffers), push constants,
    // pipelines and a ring of kUiSets sets. Created always (small); resources only when the mode needs them.
    {
        VkDescriptorSetLayoutBinding ub[4]{};
        VkDescriptorType ut[4] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
        for (int i = 0; i < 4; i++) {
            ub[i].binding = i;
            ub[i].descriptorType = ut[i];
            ub[i].descriptorCount = 1;
            ub[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        ub[0].pImmutableSamplers = &m->sampler;
        VkDescriptorSetLayoutCreateInfo udci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        udci.bindingCount = 4;
        udci.pBindings = ub;
        if (m->CreateDescriptorSetLayout(d, &udci, nullptr, &m->uiDsl) != VK_SUCCESS) { destroy(); return false; }
        VkPushConstantRange upcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(UiPC)};
        VkPipelineLayoutCreateInfo uplci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        uplci.setLayoutCount = 1;
        uplci.pSetLayouts = &m->uiDsl;
        uplci.pushConstantRangeCount = 1;
        uplci.pPushConstantRanges = &upcr;
        if (m->CreatePipelineLayout(d, &uplci, nullptr, &m->uiLayout) != VK_SUCCESS) { destroy(); return false; }
        struct { const uint32_t* code; size_t size; } uspv[U_COUNT] = {
            {ui_detect_spv, sizeof(ui_detect_spv)}, {ui_box_update_spv, sizeof(ui_box_update_spv)},
            {ui_box_mask_spv, sizeof(ui_box_mask_spv)}, {ui_mask_spv, sizeof(ui_mask_spv)},
            {ui_inpaint_spv, sizeof(ui_inpaint_spv)}, {ui_patch_spv, sizeof(ui_patch_spv)}};
        for (int i = 0; i < U_COUNT; i++) {
            VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            smci.codeSize = uspv[i].size;
            smci.pCode = uspv[i].code;
            VkShaderModule sm;
            if (m->CreateShaderModule(d, &smci, nullptr, &sm) != VK_SUCCESS) { destroy(); return false; }
            VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", nullptr};
            cpci.layout = m->uiLayout;
            VkResult r = m->CreateComputePipelines(d, ii.pipelineCache, 1, &cpci, nullptr, &m->uiPipes[i]);
            m->DestroyShaderModule(d, sm, nullptr);
            if (r != VK_SUCCESS) { destroy(); return false; }
        }
        VkDescriptorPoolSize ups[3] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kUiSets},
                                       {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kUiSets},
                                       {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * kUiSets}};
        VkDescriptorPoolCreateInfo udpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        udpci.maxSets = kUiSets;
        udpci.poolSizeCount = 3;
        udpci.pPoolSizes = ups;
        if (m->CreateDescriptorPool(d, &udpci, nullptr, &m->uiPool) != VK_SUCCESS) { destroy(); return false; }
        VkDescriptorSetLayout ul[kUiSets];
        for (auto& l : ul) l = m->uiDsl;
        VkDescriptorSetAllocateInfo udsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        udsai.descriptorPool = m->uiPool;
        udsai.descriptorSetCount = kUiSets;
        udsai.pSetLayouts = ul;
        if (m->AllocateDescriptorSets(d, &udsai, m->uiSets) != VK_SUCCESS) { destroy(); return false; }
    }

    if (!resize(ii.width, ii.height, ii.viewFormat)) { destroy(); return false; }
    return true;
}

bool Interpolator::resize(uint32_t w, uint32_t h, VkFormat fmt) {
    return m && m->rebuild(w, h, fmt, false);
}

bool Interpolator::Impl::rebuild(uint32_t w, uint32_t h, VkFormat fmt, bool force) {
    // Invariant: W != 0 <=> all size-dependent buffers exist. Any failure leaves W = H = 0.
    // The output is written through an `rgba8` storage declaration: only 8-bit UNORM views.
    // Below kMinSize the flow resolution (W/2 x H/2, rounded) degenerates (W or H = 1 -> size-0 buffers).
    bool ok = (fmt == VK_FORMAT_B8G8R8A8_UNORM || fmt == VK_FORMAT_R8G8B8A8_UNORM) && w >= kMinSize && h >= kMinSize;
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
           (uint32_t)s.uiProtection <= (uint32_t)UiProtection::WholeScreen &&
           in(s.uiCrosshairArea, 0.0f, 1.0f) && s.uiCrosshairArea > 0.0f &&
           s.uiBoxWidth <= 16384 && s.uiBoxHeight <= 16384 &&
           s.uiBoxOffsetX >= -16384 && s.uiBoxOffsetX <= 16384 && s.uiBoxOffsetY >= -16384 && s.uiBoxOffsetY <= 16384 &&
           s.uiStillFrames >= 1 && s.uiStillFrames <= 255 && in(s.uiStillThreshold, 0.0f, 255.0f) &&
           s.uiInpaintReach >= 1 && s.uiInpaintReach <= 16 &&
           in(s.uiMaxStillFraction, 0.0f, 1.0f) && s.uiMaxStillFraction > 0.0f;
}

bool Interpolator::setSettings(const Settings& s) {
    if (!m || !validate(s) || s.flowScale != m->set.flowScale) return false;
    // UI protection: creating what the new mode needs is allowed here (it frees nothing, so no GPU wait); false =
    // the allocation failed and nothing changed.
    Settings old = m->set;
    m->set = s;
    if (!m->uiEnsure()) { m->set = old; return false; }
    m->uiAdopted(old);
    return true;
}

bool Interpolator::reconfigure(const Settings& s) {
    if (!m || !validate(s)) return false;
    bool structural = s.flowScale != m->set.flowScale;
    Settings old = m->set;
    m->set = s;
    // re-create only if a size is held (W == 0: a later resize() creates the buffers with the new settings)
    if (structural && m->W) return m->rebuild(m->W, m->H, m->info.viewFormat, true);
    if (!m->uiEnsure()) { m->set = old; return false; }
    m->uiAdopted(old);
    return true;
}

Settings Interpolator::settings() const {
    return m ? m->set : Settings{};
}

void Interpolator::destroy() {
    if (!m) return;
    VkDevice d = m->info.device;
    if (m->DestroyBuffer) m->destroySizeDependent();
    if (m->qpool) m->DestroyQueryPool(d, m->qpool, nullptr);
    if (m->pool) m->DestroyDescriptorPool(d, m->pool, nullptr);
    for (auto p : m->pipes) if (p) m->DestroyPipeline(d, p, nullptr);
    for (auto p : m->uiPipes) if (p) m->DestroyPipeline(d, p, nullptr);
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

VkDescriptorSet Interpolator::Impl::takeSet(int ring, VkImageView prev, VkImageView curr, VkImageView out) {
    VkDescriptorSet set = sets[ring][setIdx[ring]];
    setIdx[ring] = (setIdx[ring] + 1) % kSetRing;
    frameSets[ring]++;
    VkDescriptorBufferInfo bi[3];
    VkDescriptorImageInfo ii[3] = {{VK_NULL_HANDLE, prev, VK_IMAGE_LAYOUT_GENERAL},
                                   {VK_NULL_HANDLE, curr, VK_IMAGE_LAYOUT_GENERAL},
                                   {VK_NULL_HANDLE, out, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet wr[6]{};
    uint32_t n = ring == R_SYNTH ? 6 : 5;   // estimate sets never get binding 5 (out)
    for (uint32_t i = 0; i < n; i++) {
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = set;
        wr[i].dstBinding = i;
        wr[i].descriptorCount = 1;
        if (i < 3) {
            bi[i] = {buf[i], 0, VK_WHOLE_SIZE};
            wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[i].pBufferInfo = &bi[i];
        } else {
            wr[i].descriptorType = i == 5 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            wr[i].pImageInfo = &ii[i - 3];
        }
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
        I.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, I.pipes[p]);
        I.CmdPushConstants(cb, I.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC), &pc);
        I.CmdDispatch(cb, gx, gy, 1);
    }
};
}  // namespace

// ---------------------------------------------------------------------------------------------------------
// Static-UI protection: recording
// ---------------------------------------------------------------------------------------------------------
namespace {
void uiDispatch(Interpolator::Impl& I, VkCommandBuffer cb, UiPipe p, VkDescriptorSet set, const UiPC& pc, uint32_t gx, uint32_t gy) {
    I.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, I.uiPipes[p]);
    I.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, I.uiLayout, 0, 1, &set, 0, nullptr);
    I.CmdPushConstants(cb, I.uiLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(UiPC), &pc);
    I.CmdDispatch(cb, gx, gy, 1);
}
}  // namespace

// Records the observe of one real frame into the entry it picks: clean copy, stillness, masks, inpaint.
// The caller has checked uiActive() and the set ring (uiObserveSets()).
bool Interpolator::Impl::uiObserveRecord(VkCommandBuffer cb, VkImageView view) {
    const bool ws = set.uiProtection == UiProtection::WholeScreen;
    const UiBox bx = computeBox();
    const int e = uiPick(view);
    UiEntry& E = uiE[e];
    // buffers first (they may fail and must not leave a half-observed entry): the box mask of this entry, and the
    // box state; growing the state buffer or moving the box restarts the box counters
    const VkDeviceSize maskWords = 4 + (VkDeviceSize)((bx.w + 3) / 4) * bx.h;
    const VkBuffer oldState = uiBoxState.b;
    if (!growBuf(E.box, maskWords * 4) || !growBuf(uiBoxState, (VkDeviceSize)bx.w * bx.h * 4)) return false;
    if (uiBoxState.b != oldState || !(bx == uiPrevBox)) uiBoxHist = false;
    uiPrevBox = bx;

    // flush the layout transition of freshly created images
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

    E.key = view;
    E.seq = ++uiSeq;
    E.bx = bx;
    E.hasBox = true;
    E.hasWs = ws;
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
    const int fullStride = ((int)W + 3) / 4, boxStride = (bx.w + 3) / 4;

    VkDescriptorSet boxSet = takeUiSet(view, E.view, uiBoxState, E.box);
    VkDescriptorSet wsSet = ws ? takeUiSet(view, E.view, uiCellState, E.ws) : VK_NULL_HANDLE;

    // 1. clean copy (+ whole screen: the still cells, tile flags)
    pc.wholeScreen = ws;
    pc.reset = !uiCellHist;
    uiDispatch(*this, cb, U_DETECT, ws ? wsSet : boxSet, pc, divUp(cw, 8), divUp(ch, 8));
    // 2. box: counters, then (next barrier) its mask
    pc.reset = !uiBoxHist;
    uiDispatch(*this, cb, U_BOX_UPDATE, boxSet, pc, divUp(bx.w, 8), divUp(bx.h, 8));
    r.cbar();
    pc.maskX = bx.x; pc.maskY = bx.y; pc.maskW = bx.w; pc.maskH = bx.h; pc.maskStride = boxStride; pc.maskOff = 4; pc.gate = 0;
    uiDispatch(*this, cb, U_BOX_MASK, boxSet, pc, divUp(boxStride, 8), divUp(bx.h, 8));
    // 3. whole screen: the full-resolution mask of the flagged tiles
    if (ws) {
        pc.maskX = pc.maskY = 0; pc.maskW = (int)W; pc.maskH = (int)H; pc.maskStride = fullStride;
        pc.maskOff = 4 + tilesX * tilesY; pc.gate = 1;
        uiDispatch(*this, cb, U_MASK, wsSet, pc, tilesX, tilesY * 4);
    }
    r.cbar();
    r.stamp("ui_detect");
    // 4. inpaint the clean copy (whole screen first: the box, last, owns its pixels)
    if (ws) uiDispatch(*this, cb, U_INPAINT, wsSet, pc, divUp((int)W, kUiInpTile), divUp((int)H, kUiInpTile));
    r.cbar();
    pc.maskX = bx.x; pc.maskY = bx.y; pc.maskW = bx.w; pc.maskH = bx.h; pc.maskStride = boxStride; pc.maskOff = 4; pc.gate = 0;
    uiDispatch(*this, cb, U_INPAINT, boxSet, pc, divUp(bx.w, kUiInpTile), divUp(bx.h, kUiInpTile));
    r.cbar();
    r.stamp("ui_inpaint");

    uiBoxHist = true;
    uiCellHist = ws;
    return true;
}

// After a generated frame is complete: paste the real pixels of the frame the synth leads up to (curr) over it.
void Interpolator::Impl::uiRecordPatch(VkCommandBuffer cb, VkImageView curr, VkImageView out, const UiEntry& E) {
    Rec r(*this, cb);
    UiPC pc{};
    pc.W = (int)W; pc.H = (int)H;
    pc.tilesX = tilesX; pc.tilesY = tilesY;
    pc.stillLimit = (int)std::floor((double)set.uiMaxStillFraction * cw * ch);
    if (E.hasWs) {
        pc.maskW = (int)W; pc.maskH = (int)H; pc.maskStride = ((int)W + 3) / 4; pc.maskOff = 4 + tilesX * tilesY; pc.gate = 1;
        uiDispatch(*this, cb, U_PATCH, takeUiSet(curr, out, uiCellState, E.ws), pc, tilesX, tilesY * 4);
        r.barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, RW);
    }
    if (E.hasBox) {
        pc.maskX = E.bx.x; pc.maskY = E.bx.y; pc.maskW = E.bx.w; pc.maskH = E.bx.h;
        pc.maskStride = (E.bx.w + 3) / 4; pc.maskOff = 4; pc.gate = 0;
        uiDispatch(*this, cb, U_PATCH, takeUiSet(curr, out, uiBoxState, E.box), pc, divUp(pc.maskStride, 8), divUp(E.bx.h, 8));
    }
}

bool Interpolator::recordObserve(VkCommandBuffer cb, VkImageView frame) {
    if (!m || !m->W || !frame) return false;
    Impl& I = *m;
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

bool Interpolator::recordEstimate(VkCommandBuffer cb, VkImageView prev, VkImageView curr) {
    if (!m || !m->W) return false;
    Impl& I = *m;
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
            needObs += I.uiFind(prev) < 0;
            needObs += I.uiLast < 0 || I.uiE[I.uiLast].key != curr;
        } else if (I.uiFind(prev) < 0 || I.uiFind(curr) < 0) {
            ui = false;
        }
        if (ui && I.uiFrameSets + needObs * I.uiObserveSets() > kUiSets) return false;
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

    VkImageView estP = prev, estC = curr;
    int ep = -1, ec = -1;
    if (ui) {
        if (!I.uiExplicit) {
            if (I.uiFind(prev) < 0 && !I.uiObserveRecord(cb, prev)) return false;
            if ((I.uiLast < 0 || I.uiE[I.uiLast].key != curr) && !I.uiObserveRecord(cb, curr)) return false;
        }
        ep = I.uiFind(prev);
        ec = I.uiFind(curr);
        if (ep < 0 || ec < 0 || ep == ec) return false;   // (implicit: the second observe evicted the first: cannot happen with kUiFrames >= 3)
        estP = I.uiE[ep].view;
        estC = I.uiE[ec].view;
    }
    VkDescriptorSet set = I.takeSet(Impl::R_EST, estP, estC, VK_NULL_HANDLE);
    I.haveEst = true;
    I.estPrev = prev;
    I.estCurr = curr;
    I.estUi = ui;
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

bool Interpolator::recordSynth(VkCommandBuffer cb, VkImageView prev, VkImageView curr, VkImageView out, float t) {
    if (!m || !m->W || !(t > 0.0f && t < 1.0f)) return false;
    Impl& I = *m;
    if (!I.haveEst || prev != I.estPrev || curr != I.estCurr) return false;
    assert(I.frameSets[Impl::R_SYNTH] < kSetRing && "framegen: more than kSetRing recordSynth() since beginFrame()");
    if (I.frameSets[Impl::R_SYNTH] >= kSetRing) return false;
    VkImageView srcP = prev, srcC = curr;
    if (I.estUi) {
        // the clean copies the estimate saw must still be those frames (not evicted / re-observed since)
        const UiEntry &ep = I.uiE[I.estEnt[0]], &ec = I.uiE[I.estEnt[1]];
        if (ep.seq != I.estSeq[0] || ec.seq != I.estSeq[1] || !ep.view || !ec.view) return false;
        if (I.uiFrameSets + 2 > kUiSets) return false;
        srcP = ep.view; srcC = ec.view;
    }
    VkDescriptorSet set = I.takeSet(Impl::R_SYNTH, srcP, srcC, out);

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

bool Interpolator::record(VkCommandBuffer cb, VkImageView prev, VkImageView curr, VkImageView out, float t) {
    if (!m || !m->W || !(t > 0.0f && t < 1.0f)) return false;
    // both rings must have room, so that a refused record() records nothing
    assert(m->frameSets[0] < kSetRing && m->frameSets[1] < kSetRing && "framegen: more than kSetRing records since beginFrame()");
    if (m->frameSets[Impl::R_EST] >= kSetRing || m->frameSets[Impl::R_SYNTH] >= kSetRing) return false;
    if (m->uiActive() && m->uiFrameSets + 2 * m->uiObserveSets() + 2 > kUiSets) return false;   // 2 implicit observes + the patch
    return recordEstimate(cb, prev, curr) && recordSynth(cb, prev, curr, out, t);
}

}  // namespace framegen

