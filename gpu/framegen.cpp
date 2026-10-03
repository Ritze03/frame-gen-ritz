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

namespace framegen {

#define FG_FUNCS(X) \
    X(CreateBuffer) X(DestroyBuffer) X(GetBufferMemoryRequirements) X(AllocateMemory) X(FreeMemory) \
    X(BindBufferMemory) X(CreateSampler) X(DestroySampler) X(CreateDescriptorSetLayout) \
    X(DestroyDescriptorSetLayout) X(CreatePipelineLayout) X(DestroyPipelineLayout) X(CreateShaderModule) \
    X(DestroyShaderModule) X(CreateComputePipelines) X(DestroyPipeline) X(CreateDescriptorPool) \
    X(DestroyDescriptorPool) X(AllocateDescriptorSets) X(UpdateDescriptorSets) X(CreateQueryPool) \
    X(DestroyQueryPool) X(GetQueryPoolResults) X(CmdBindPipeline) X(CmdBindDescriptorSets) \
    X(CmdPushConstants) X(CmdDispatch) X(CmdPipelineBarrier) X(CmdFillBuffer) X(CmdResetQueryPool) \
    X(CmdWriteTimestamp)

// Must match the push-constant block in shaders/common.glsl.
struct PC {
    int32_t w, h, bw, bh, pw, ph, offA, offC, srcA, srcC, fIn, fOut, fPar, radius, medIdx,
        W, H, fw, fh, pf, eOff, eStride, nBlocks;
};

enum Pipe { P_LUMA, P_DOWN, P_MEDHIST, P_SEARCH, P_MEDIAN3, P_SUBPIX, P_PIXFIELD, P_ERROR, P_RESOLVE, P_FALLBACK, P_COUNT };

struct Level { int w, h, bw, bh, offA, offC, fS, fM; };

struct Interpolator::Impl {
#define X(n) PFN_vk##n n = nullptr;
    FG_FUNCS(X)
#undef X
    InitInfo info{};
    VkPhysicalDeviceMemoryProperties memProps{};

    VkSampler sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipes[P_COUNT]{};
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet sets[kSetRing]{};
    uint32_t setIdx = 0;
    uint32_t frameRecords = 0;   // records since the last beginFrame()

    // size dependent
    uint32_t W = 0, H = 0;
    int fw = 0, fh = 0;
    std::vector<Level> lv;
    int fieldSub = 0, fieldPF = 0, eOff = 0, eStride = 0;
    VkBuffer buf[3]{};
    VkDeviceMemory mem[3]{};
    VkDeviceSize bufSize[3]{};

    // profiling
    bool profiling = false;
    VkQueryPool qpool = VK_NULL_HANDLE;
    static constexpr uint32_t kMaxQ = 64;
    std::vector<const char*> qnames;
    uint32_t qcount = 0;

    bool createSizeDependent();
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
    fw = rintHalfEven(W * 0.5);
    fh = rintHalfEven(H * 0.5);
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
    eStride = (int)(W + 3) / 4;
    eOff = 64;
    int uCount = eOff + eStride * (int)H;
    if (!makeBuffer(0, (VkDeviceSize)lumaOff * 4)) return false;
    if (!makeBuffer(1, (VkDeviceSize)fieldOff * 8)) return false;
    if (!makeBuffer(2, (VkDeviceSize)uCount * 4)) return false;
    return true;
}

void Interpolator::Impl::destroySizeDependent() {
    for (int i = 0; i < 3; i++) {
        if (buf[i]) DestroyBuffer(info.device, buf[i], nullptr);
        if (mem[i]) FreeMemory(info.device, mem[i], nullptr);
        buf[i] = VK_NULL_HANDLE; mem[i] = VK_NULL_HANDLE;
    }
}

bool Interpolator::init(const InitInfo& ii) {
    destroy();
    if (!ii.getMemProps || !ii.gdpa || !ii.device) return false;
    m = new Impl();
    m->info = ii;
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
        {fallback_spv, sizeof(fallback_spv)}};
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

    VkDescriptorPoolSize ps[3] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * kSetRing},
                                  {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 * kSetRing},
                                  {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kSetRing}};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = kSetRing;
    dpci.poolSizeCount = 3;
    dpci.pPoolSizes = ps;
    if (m->CreateDescriptorPool(d, &dpci, nullptr, &m->pool) != VK_SUCCESS) { destroy(); return false; }
    VkDescriptorSetLayout layouts[kSetRing];
    for (auto& l : layouts) l = m->dsl;
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = m->pool;
    dsai.descriptorSetCount = kSetRing;
    dsai.pSetLayouts = layouts;
    if (m->AllocateDescriptorSets(d, &dsai, m->sets) != VK_SUCCESS) { destroy(); return false; }

    if (!resize(ii.width, ii.height, ii.viewFormat)) { destroy(); return false; }
    return true;
}

bool Interpolator::resize(uint32_t w, uint32_t h, VkFormat fmt) {
    if (!m) return false;
    // Invariant: W != 0 <=> all size-dependent buffers exist. Any failure leaves W = H = 0.
    // The output is written through an `rgba8` storage declaration: only 8-bit UNORM views.
    // Below kMinSize the flow resolution (W/2 x H/2, rounded) degenerates (W or H = 1 -> size-0 buffers).
    bool ok = (fmt == VK_FORMAT_B8G8R8A8_UNORM || fmt == VK_FORMAT_R8G8B8A8_UNORM) && w >= kMinSize && h >= kMinSize;
    if (ok && w == m->W && h == m->H) { m->info.viewFormat = fmt; return true; }   // buffers do not depend on the format
    m->destroySizeDependent();
    m->W = m->H = 0;
    if (!ok) return false;
    m->info.viewFormat = fmt;
    m->W = w; m->H = h;
    if (!m->createSizeDependent()) {
        m->destroySizeDependent();
        m->W = m->H = 0;
        return false;
    }
    return true;
}

void Interpolator::destroy() {
    if (!m) return;
    VkDevice d = m->info.device;
    if (m->DestroyBuffer) m->destroySizeDependent();
    if (m->qpool) m->DestroyQueryPool(d, m->qpool, nullptr);
    if (m->pool) m->DestroyDescriptorPool(d, m->pool, nullptr);
    for (auto p : m->pipes) if (p) m->DestroyPipeline(d, p, nullptr);
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
    if (m) m->frameRecords = 0;
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

bool Interpolator::record(VkCommandBuffer cb, VkImageView prev, VkImageView curr, VkImageView out, float) {
    if (!m || !m->W) return false;
    Impl& I = *m;
    // The next set in the ring may still be referenced by an earlier record of this submission.
    assert(I.frameRecords < kSetRing && "framegen: more than kSetRing records since beginFrame()");
    if (I.frameRecords >= kSetRing) return false;
    I.frameRecords++;
    VkDevice d = I.info.device;
    VkDescriptorSet set = I.sets[I.setIdx];
    I.setIdx = (I.setIdx + 1) % kSetRing;

    VkDescriptorBufferInfo bi[3];
    VkDescriptorImageInfo ii[3] = {{VK_NULL_HANDLE, prev, VK_IMAGE_LAYOUT_GENERAL},
                                   {VK_NULL_HANDLE, curr, VK_IMAGE_LAYOUT_GENERAL},
                                   {VK_NULL_HANDLE, out, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet wr[6]{};
    for (int i = 0; i < 6; i++) {
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = set;
        wr[i].dstBinding = i;
        wr[i].descriptorCount = 1;
        if (i < 3) {
            bi[i] = {I.buf[i], 0, VK_WHOLE_SIZE};
            wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[i].pBufferInfo = &bi[i];
        } else {
            wr[i].descriptorType = i == 5 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            wr[i].pImageInfo = &ii[i - 3];
        }
    }
    I.UpdateDescriptorSets(d, 6, wr, 0, nullptr);

    I.qcount = 0;
    I.qnames.clear();
    auto stamp = [&](const char* name) {
        if (!I.profiling || !I.qpool || I.qcount >= Impl::kMaxQ) return;
        I.CmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, I.qpool, I.qcount++);
        I.qnames.push_back(name);
    };
    auto barrier = [&](VkPipelineStageFlags src, VkAccessFlags srcA, VkPipelineStageFlags dst, VkAccessFlags dstA) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, srcA, dstA};
        I.CmdPipelineBarrier(cb, src, dst, 0, 1, &mb, 0, nullptr, 0, nullptr);
    };
    const VkPipelineStageFlags CS = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    const VkAccessFlags RW = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    auto cbar = [&] { barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, RW); };

    if (I.profiling && I.qpool) I.CmdResetQueryPool(cb, I.qpool, 0, Impl::kMaxQ);
    stamp("start");
    // WAR against the previous record's reads of our buffers, then clear the counters.
    barrier(CS, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    I.CmdFillBuffer(cb, I.buf[2], 0, 64 * 4, 0);
    barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, CS, RW);
    I.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, I.layout, 0, 1, &set, 0, nullptr);

    PC pc{};
    pc.W = (int)I.W; pc.H = (int)I.H; pc.fw = I.fw; pc.fh = I.fh;
    pc.pf = I.fieldPF; pc.eOff = I.eOff; pc.eStride = I.eStride;
    pc.nBlocks = I.lv[0].bw * I.lv[0].bh;
    auto run = [&](Pipe p, uint32_t gx, uint32_t gy) {
        I.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, I.pipes[p]);
        I.CmdPushConstants(cb, I.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC), &pc);
        I.CmdDispatch(cb, gx, gy, 1);
    };
    auto div = [](int a, int b) { return (uint32_t)((a + b - 1) / b); };

    // 1. luma + 2x downsample
    const Level& l0 = I.lv[0];
    pc.w = l0.w; pc.h = l0.h; pc.offA = l0.offA; pc.offC = l0.offC;
    run(P_LUMA, div(l0.w, 8), div(l0.h, 8));
    cbar();
    stamp("luma");
    // 2. pyramid
    for (size_t i = 1; i < I.lv.size(); i++) {
        const Level &s = I.lv[i - 1], &t = I.lv[i];
        pc.w = t.w; pc.h = t.h; pc.pw = s.w; pc.ph = s.h;
        pc.offA = t.offA; pc.offC = t.offC; pc.srcA = s.offA; pc.srcC = s.offC;
        run(P_DOWN, div(t.w, 8), div(t.h, 8));
        cbar();
    }
    stamp("pyramid");
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
            run(P_MEDHIST, 1, 1);
            cbar();
        }
        pc.fOut = l.fS;
        run(P_SEARCH, l.bw, l.bh);
        cbar();
        pc.fIn = l.fS; pc.fOut = l.fM;
        run(P_MEDIAN3, div(l.bw, 8), div(l.bh, 8));
        cbar();
        stamp(lvNames[std::min(i, 7)]);
    }
    // 6. sub-pixel refinement (+ SAD sum for the scene-cut test)
    pc.w = l0.w; pc.h = l0.h; pc.bw = l0.bw; pc.bh = l0.bh; pc.offA = l0.offA; pc.offC = l0.offC;
    pc.fIn = l0.fM; pc.fOut = I.fieldSub;
    run(P_SUBPIX, l0.bw, l0.bh);
    cbar();
    stamp("subpixel");
    // 8. per-pixel vector choice at flow res
    pc.fIn = I.fieldSub;
    run(P_PIXFIELD, div(I.fw, 16), div(I.fh, 16));
    cbar();
    stamp("pixel field");
    // 9. full-res mismatch
    run(P_ERROR, div(I.eStride, 8), div((int)I.H, 8));
    cbar();
    stamp("mismatch");
    // 10-11. box + trust + resolve
    run(P_RESOLVE, div((int)I.W, 16), div((int)I.H, 16));
    cbar();
    stamp("resolve");
    // global fallbacks (scene cut / too much untrusted) -> frame a
    run(P_FALLBACK, div((int)I.W, 64), div((int)I.H, 64));
    barrier(CS, VK_ACCESS_SHADER_WRITE_BIT, CS, VK_ACCESS_SHADER_READ_BIT);   // compute-only queue safe
    stamp("fallback");
    return true;
}

}  // namespace framegen
