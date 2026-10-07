// blurtest: validation + benchmark of framegen's interpolation-based motion blur (recordSynthBlur).
//   blurtest bench [W H]        GPU cost, per real pair, back-to-back (minimum of 21 submissions): the estimate, N x
//                               recordSynth (+ a naive blend pass), recordSynthBlur for N = 2..8 and the frame-generation
//                               case (M blurred outputs per pair); flowScale 2 and 4; default sizes 1280x960, 1920x1080,
//                               2560x1440
//   blurtest check [--bgra] [W H]   correctness (default 1280x720): the FNV hashes of the blur-off output (compare them with a
//                               build of the library before the blur: compile this file with -DBLURTEST_NO_BLUR against it),
//                               recordSynthBlur == recordSynth for one instant, == the CPU average of the recordSynth sub-frames
//                               (every N 2..8, windows, weights, with scene cut / flick fallbacks), the smear of a moving
//                               object's edge against the ideal model, the crosshair pixel-exact under UI protection, the
//                               descriptor rings, validation of the settings.
//   option --algo v1|v2         the motion version (Settings::motionVersion) every Interpolator is created and set with (default v1)
#include "framegen.h"
#include "blendbench_spv.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

// protection modes of the tests: 0 Off, 1 Crosshair, 2 WholeScreen, 3 CrosshairV2 (compile with -DBLURTEST_NO_V2 against a library without it)
#ifdef BLURTEST_NO_V2
#define BLURTEST_NO_MOTION   // a library before CrosshairV2 has no motion version either
#define BT_NMODES 3
#else
#define BT_NMODES 4
#endif
static framegen::UiProtection btMode(int m) {
    using framegen::UiProtection;
#ifndef BLURTEST_NO_V2
    if (m == 3) return UiProtection::CrosshairV2;
#endif
    return m == 0 ? UiProtection::Off : m == 1 ? UiProtection::Crosshair : UiProtection::WholeScreen;
}
static const char* btModeName(int m) { return m == 0 ? "Off" : m == 1 ? "Crosshair" : m == 2 ? "WholeScreen" : "CrosshairV2"; }

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d\n", #x, r_); exit(1); } } while (0)

static VkDevice dev;
static VkPhysicalDevice phys;
static VkPhysicalDeviceMemoryProperties memProps;
static VkQueue queue;
static VkCommandBuffer cb;
static VkFence fence;
static bool bgra = false;
static double tsPeriodMs = 0;

static uint32_t memType(uint32_t bits, VkMemoryPropertyFlags f) {
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & f) == f) return i;
    fprintf(stderr, "no memory type\n");
    exit(1);
}
struct Img { VkImage img; VkDeviceMemory mem; VkImageView view; };
static VkFormat fmt() { return bgra ? VK_FORMAT_B8G8R8A8_UNORM : VK_FORMAT_R8G8B8A8_UNORM; }
static Img makeImage(uint32_t w, uint32_t h, VkImageUsageFlags usage) {
    Img r{};
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D; ici.format = fmt(); ici.extent = {w, h, 1};
    ici.mipLevels = ici.arrayLayers = 1; ici.samples = VK_SAMPLE_COUNT_1_BIT; ici.tiling = VK_IMAGE_TILING_OPTIMAL; ici.usage = usage;
    CK(vkCreateImage(dev, &ici, nullptr, &r.img));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev, r.img, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size, memType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    CK(vkAllocateMemory(dev, &mai, nullptr, &r.mem));
    CK(vkBindImageMemory(dev, r.img, r.mem, 0));
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = r.img; vci.viewType = VK_IMAGE_VIEW_TYPE_2D; vci.format = fmt();
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    CK(vkCreateImageView(dev, &vci, nullptr, &r.view));
    return r;
}
static void imgBarrier(VkImage img, VkImageLayout from, VkAccessFlags srcA, VkAccessFlags dstA) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = srcA; b.dstAccessMask = dstA; b.oldLayout = from; b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.image = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
}
static void begin() {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CK(vkBeginCommandBuffer(cb, &bi));
}
static void submit() {
    CK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    CK(vkQueueSubmit(queue, 1, &si, fence));
    CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
    CK(vkResetFences(dev, 1, &fence));
}

static VkBuffer stg;
static VkDeviceMemory stgMem;
static uint8_t* stgMap;
static size_t stgBytes;
static void initStaging(size_t bytes) {
    stgBytes = bytes;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes; bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    CK(vkCreateBuffer(dev, &bci, nullptr, &stg));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, stg, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size,
                             memType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    CK(vkAllocateMemory(dev, &mai, nullptr, &stgMem));
    CK(vkBindBufferMemory(dev, stg, stgMem, 0));
    CK(vkMapMemory(dev, stgMem, 0, bytes, 0, (void**)&stgMap));
}
static void swz(uint8_t* p, size_t bytes) { if (bgra) for (size_t i = 0; i < bytes; i += 4) std::swap(p[i], p[i + 2]); }
static void upload(const std::vector<uint8_t>& rgba, Img& img, uint32_t w, uint32_t h) {
    memcpy(stgMap, rgba.data(), rgba.size());
    swz(stgMap, rgba.size());
    VkBufferImageCopy rg{};
    rg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; rg.imageExtent = {w, h, 1};
    begin();
    imgBarrier(img.img, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    vkCmdCopyBufferToImage(cb, stg, img.img, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
    imgBarrier(img.img, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    submit();
}
static std::vector<uint8_t> download(Img& img, uint32_t w, uint32_t h) {
    VkBufferImageCopy rg{};
    rg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; rg.imageExtent = {w, h, 1};
    begin();
    imgBarrier(img.img, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    vkCmdCopyImageToBuffer(cb, img.img, VK_IMAGE_LAYOUT_GENERAL, stg, 1, &rg);
    submit();
    std::vector<uint8_t> r(stgMap, stgMap + (size_t)w * h * 4);
    swz(r.data(), r.size());
    return r;
}
// ---------------------------------------------------------------- synthetic scene
static const int kPanX = 14, kPanY = 6;   // background pan per frame (even: the mid frame is an integer shift)
static uint32_t hash2(int x, int y, uint32_t seed) {
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u + seed * 2246822519u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return h ^ (h >> 16);
}
static float vnoise(float x, float y, float scale, uint32_t seed) {
    x /= scale; y /= scale;
    int xi = (int)std::floor(x), yi = (int)std::floor(y);
    float fx = x - xi, fy = y - yi;
    auto v = [&](int a, int b) { return (hash2(xi + a, yi + b, seed) & 1023u) / 1023.0f; };
    float a = v(0, 0) * (1 - fx) + v(1, 0) * fx, b = v(0, 1) * (1 - fx) + v(1, 1) * fx;
    return a * (1 - fy) + b * fy;
}
struct Scene {
    int W, H, NF;
    int bgW, bgH;
    std::vector<uint8_t> bg;      // bgW x bgH RGBA, panned over
    struct Obj { int x0, y0, w, h, vx, vy; uint8_t r, g, b; };   // independently moving foreground rectangles
    std::vector<Obj> objs;
    std::vector<uint8_t> ui;      // 0 none, 1 crosshair, 2 hud (per pixel of the overlay)
    std::vector<uint8_t> uiCol;   // overlay RGB(A)
    int cx, cy;                   // crosshair centre
    int hx0, hy0, hx1, hy1;       // hud box
    Scene(int w, int h, int nf, int cxOff, int cyOff) : W(w), H(h), NF(nf) {
        bgW = W + kPanX * (NF + 1);
        bgH = H + kPanY * (NF + 1);
        bg.resize((size_t)bgW * bgH * 4);
        for (int y = 0; y < bgH; y++)
            for (int x = 0; x < bgW; x++)
                for (int c = 0; c < 3; c++) {
                    float v = 0.5f * vnoise(x, y, 40, c + 1) + 0.3f * vnoise(x, y, 12, c + 11) + 0.15f * vnoise(x, y, 4, c + 21);
                    v += ((hash2(x, y, c + 31) & 255) / 255.0f - 0.5f) * 0.08f;
                    bg[((size_t)y * bgW + x) * 4 + c] = (uint8_t)std::min(255.0f, std::max(0.0f, 30.0f + 190.0f * v));
                }
        for (size_t i = 3; i < bg.size(); i += 4) bg[i] = 255;
        // moving objects: some pass over the crosshair and the HUD box on the last pair
        for (int i = 0; i < 16; i++) {
            Obj o;
            o.w = 30 + (int)(hash2(i, 1, 77) % 60); o.h = 30 + (int)(hash2(i, 2, 77) % 60);
            o.vx = 2 * ((int)(hash2(i, 3, 77) % 21) - 10); o.vy = 2 * ((int)(hash2(i, 4, 77) % 11) - 5);
            // aim the object so that it is near the crosshair (i < 8) or the HUD (i >= 8) at frame 10.5
            int tx = i < 8 ? W / 2 + cxOff + (int)(hash2(i, 5, 77) % 41) - 20 : 40 + (int)(hash2(i, 6, 77) % 200);
            int ty = i < 8 ? H / 2 + cyOff + (int)(hash2(i, 7, 77) % 41) - 20 : 30 + (int)(hash2(i, 8, 77) % 60);
            o.x0 = tx - o.w / 2 - o.vx * 21 / 2; o.y0 = ty - o.h / 2 - o.vy * 21 / 2;
            o.r = 40 + hash2(i, 9, 77) % 200; o.g = 40 + hash2(i, 10, 77) % 200; o.b = 40 + hash2(i, 11, 77) % 200;
            objs.push_back(o);
        }
        ui.assign((size_t)W * H, 0);
        uiCol.assign((size_t)W * H * 4, 0);
        cx = W / 2 + cxOff; cy = H / 2 + cyOff;
        auto put = [&](int x, int y, uint8_t k, uint8_t r, uint8_t g, uint8_t b) {
            if (x < 0 || y < 0 || x >= W || y >= H) return;
            ui[(size_t)y * W + x] = k;
            uint8_t* p = &uiCol[((size_t)y * W + x) * 4];
            p[0] = r; p[1] = g; p[2] = b; p[3] = 255;
        };
        // crosshair: arms 3 px thick, from 5 to 16 px out, centre dot 3x3; 1 px black outline, then colour
        auto inCross = [&](int dx, int dy, int grow) {
            int ax = std::abs(dx), ay = std::abs(dy);
            bool hArm = ay <= 1 + grow && ax >= 5 - grow && ax <= 16 + grow;
            bool vArm = ax <= 1 + grow && ay >= 5 - grow && ay <= 16 + grow;
            bool dot = ax <= 1 + grow && ay <= 1 + grow;
            return hArm || vArm || dot;
        };
        for (int dy = -20; dy <= 20; dy++)
            for (int dx = -20; dx <= 20; dx++) {
                if (inCross(dx, dy, 1)) put(cx + dx, cy + dy, 1, 0, 0, 0);
                if (inCross(dx, dy, 0)) put(cx + dx, cy + dy, 1, 255, 64, 255);
            }
        // hud box: dark panel with a static bar pattern
        hx0 = 40; hy0 = 30; hx1 = 40 + std::max(W / 11, 64); hy1 = 30 + std::max(H / 12, 40);
        for (int y = hy0; y < hy1; y++)
            for (int x = hx0; x < hx1; x++) {
                bool bar = ((y - hy0) / 6) % 2 == 0 && (hash2((x - hx0) / 5, (y - hy0) / 12, 5) & 3) != 0 && x > hx0 + 6 && y > hy0 + 6 && x < hx1 - 6;
                if (bar) put(x, y, 2, 235, 235, 235); else put(x, y, 2, 20, 30, 90);
            }
    }
    // the frame at time t (in frames; t * 2 is an integer): panned background, moving objects, static UI on top
    std::vector<uint8_t> frame(double t) const {
        std::vector<uint8_t> f((size_t)W * H * 4);
        int ox = (int)std::lround(t * kPanX), oy = (int)std::lround(t * kPanY);
        for (int y = 0; y < H; y++) memcpy(&f[(size_t)y * W * 4], &bg[((size_t)(y + oy) * bgW + ox) * 4], (size_t)W * 4);
        for (const Obj& o : objs) {
            int px = o.x0 + (int)std::lround(o.vx * t), py = o.y0 + (int)std::lround(o.vy * t);
            for (int y = std::max(py, 0); y < std::min(py + o.h, H); y++)
                for (int x = std::max(px, 0); x < std::min(px + o.w, W); x++) {
                    bool chk = (((x - px) / 6) + ((y - py) / 6)) & 1;
                    uint8_t* d = &f[((size_t)y * W + x) * 4];
                    d[0] = chk ? o.r : o.r / 2; d[1] = chk ? o.g : o.g / 2; d[2] = chk ? o.b : o.b / 2; d[3] = 255;
                }
        }
        for (size_t i = 0; i < (size_t)W * H; i++)
            if (ui[i]) memcpy(&f[i * 4], &uiCol[i * 4], 4);
        return f;
    }
};

// ---------------------------------------------------------------- common
using namespace framegen;
static uint64_t fnv(const std::vector<uint8_t>& v) {
    uint64_t h = 1469598103934665603ull;
    for (uint8_t b : v) { h ^= b; h *= 1099511628211ull; }
    return h;
}
// --algo: the motion version every Interpolator of the run is created and set with
#ifndef BLURTEST_NO_MOTION
static MotionVersion gAlgo = MotionVersion::V1;
static MotionPreset gPreset = MotionPreset::A;   // --preset (V2's tuning: B exercises the soft hold edge's per-sample maps)
static Settings algo(Settings s) { s.motionVersion = gAlgo; s.v2Preset = gPreset; return s; }
#else
static Settings algo(Settings s) { return s; }
#endif
static bool makeInterp(Interpolator& fg, int W, int H, const Settings& s) {
    InitInfo info;
    info.phys = phys; info.device = dev; info.gdpa = vkGetDeviceProcAddr; info.width = W; info.height = H;
    info.viewFormat = fmt(); info.getMemProps = vkGetPhysicalDeviceMemoryProperties; info.settings = algo(s);
    return fg.init(info);
}
static void destroyImg(Img& i) { vkDestroyImageView(dev, i.view, nullptr); vkDestroyImage(dev, i.img, nullptr); vkFreeMemory(dev, i.mem, nullptr); }
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static double vmin(const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); }

// ---------------------------------------------------------------- bench
// Back-to-back timing: one command buffer holds P pairs, each one recordEstimate plus `body` (the synths of that pair),
// bracketed by two timestamps; the MINIMUM of 21 submissions after 10 warm-up ones, per pair (the GPU is shared with the desktop).
static VkQueryPool tsPool;
static double timePairs(Interpolator& fg, Img* ring, int P, const std::function<void(int)>& body) {
    std::vector<double> v;
    for (int rep = 0; rep < 31; rep++) {   // the first 10 are warm-up (GPU clocks)
        begin();
        fg.beginFrame();
        vkCmdResetQueryPool(cb, tsPool, 0, 2);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, tsPool, 0);
        for (int p = 0; p < P; p++) {
            Img &prv = ring[p % 3], &cur = ring[(p + 1) % 3];
            if (!fg.recordEstimate(cb, prv.view, cur.view)) { fprintf(stderr, "estimate failed\n"); exit(1); }
            body(p);
        }
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, tsPool, 1);
        submit();
        uint64_t t2[2];
        CK(vkGetQueryPoolResults(dev, tsPool, 0, 2, sizeof(t2), t2, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        if (rep >= 10) v.push_back((t2[1] - t2[0]) * tsPeriodMs / P);
    }
    return vmin(v);   // the minimum of 21: the desktop / a game on the same GPU only ever adds time
}

// ---- the naive blend (benchmark only): a compute pass that averages up to 8 separately synthesised images
static VkPipeline blendPipe;
static VkPipelineLayout blendLayout;
static VkDescriptorSet blendSet;
static void initBlend(Img* srcs, Img& dst) {
    static VkDescriptorSetLayout dsl;
    static VkDescriptorPool pool;
    static VkSampler smp;
    if (!smp) {
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
        CK(vkCreateSampler(dev, &sci, nullptr, &smp));
        VkDescriptorSetLayoutBinding b[9]{};
        for (int i = 0; i < 9; i++) {
            b[i].binding = i; b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            b[i].descriptorType = i < 8 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            if (i < 8) b[i].pImmutableSamplers = &smp;
        }
        VkDescriptorSetLayoutCreateInfo dci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dci.bindingCount = 9; dci.pBindings = b;
        CK(vkCreateDescriptorSetLayout(dev, &dci, nullptr, &dsl));
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 36};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1; plci.pSetLayouts = &dsl; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
        CK(vkCreatePipelineLayout(dev, &plci, nullptr, &blendLayout));
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = sizeof(blendbench_spv); smci.pCode = blendbench_spv;
        VkShaderModule sm;
        CK(vkCreateShaderModule(dev, &smci, nullptr, &sm));
        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", nullptr};
        cpci.layout = blendLayout;
        CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &blendPipe));
        vkDestroyShaderModule(dev, sm, nullptr);
        VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}};
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = 1; dpci.poolSizeCount = 2; dpci.pPoolSizes = ps;
        CK(vkCreateDescriptorPool(dev, &dpci, nullptr, &pool));
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = pool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
        CK(vkAllocateDescriptorSets(dev, &dsai, &blendSet));
    }
    VkDescriptorImageInfo ii[9];
    VkWriteDescriptorSet wr[9]{};
    for (int i = 0; i < 9; i++) {
        ii[i] = {VK_NULL_HANDLE, i < 8 ? srcs[i].view : dst.view, VK_IMAGE_LAYOUT_GENERAL};
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wr[i].dstSet = blendSet; wr[i].dstBinding = i; wr[i].descriptorCount = 1;
        wr[i].descriptorType = i < 8 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        wr[i].pImageInfo = &ii[i];
    }
    vkUpdateDescriptorSets(dev, 9, wr, 0, nullptr);
}
static void recordBlend(int n, int W, int H) {
    struct { int n; float w[8]; } pc{n, {}};
    for (int k = 0; k < n; k++) pc.w[k] = 1.0f / n;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, blendPipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, blendLayout, 0, 1, &blendSet, 0, nullptr);
    vkCmdPushConstants(cb, blendLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cb, (W + 15) / 16, (H + 15) / 16, 1);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

static void benchOne(int W, int H, uint32_t flowScale, const Settings& base) {
    Scene sc(W, H, 4, 0, 0);
    Img ring[3];
    for (auto& r : ring) r = makeImage(W, H, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    Img out = makeImage(W, H, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    Img subs[8];   // the naive path's separate sub-frames
    for (auto& i : subs) i = makeImage(W, H, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    begin();
    for (auto& r : ring) imgBarrier(r.img, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    imgBarrier(out.img, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    for (auto& i : subs) imgBarrier(i.img, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_SHADER_READ_BIT);
    submit();
    initBlend(subs, out);
    for (int i = 0; i < 3; i++) upload(sc.frame(i), ring[i], W, H);
    Settings s = base;
    s.flowScale = flowScale;
    Interpolator fg;
    if (!makeInterp(fg, W, H, s)) { fprintf(stderr, "init failed\n"); exit(1); }
    VkQueryPoolCreateInfo qp{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qp.queryType = VK_QUERY_TYPE_TIMESTAMP; qp.queryCount = 2;
    CK(vkCreateQueryPool(dev, &qp, nullptr, &tsPool));
    printf("%dx%d flowScale %u (%s):\n", W, H, flowScale, flowScale == 2 ? "Quality" : "Performance");
    // per-pass of one synth (profiling, not back-to-back)
    {
        fg.setProfiling(true);
        std::vector<std::vector<double>> per;
        std::vector<const char*> names;
        for (int it = 0; it < 12; it++) {
            begin(); fg.beginFrame();
            if (!fg.recordEstimate(cb, ring[0].view, ring[1].view) || !fg.recordSynth(cb, ring[0].view, ring[1].view, out.view, 0.3f)) exit(1);
            submit();
            std::vector<uint64_t> ticks;
            if (!fg.readTimings(names, ticks)) exit(1);
            if (it < 2) continue;
            per.resize(ticks.size());
            for (size_t i = 0; i < ticks.size(); i++) per[i].push_back(ticks[i] * tsPeriodMs);
        }
        fg.setProfiling(false);
        printf("   passes of 1 pair + 1 synth (t=0.3, isolated, each incl. ramp-up):");
        for (size_t i = 0; i < names.size(); i++) printf(" %s %.3f", names[i], median(per[i]));
        printf(" ms\n");
    }
    const int P = 4;
    double est = timePairs(fg, ring, P, [&](int) {});
    printf("   estimate only (motion search of one pair): %.4f ms\n", est);
    // naive: N x recordSynth per pair (the sub-frames each into their own image, no blend counted); slope by least squares
    std::vector<double> nn, tt;
    printf("   naive N x recordSynth per pair (no blend): total ms");
    for (int n = 1; n <= 8; n++) {
        double t = timePairs(fg, ring, P, [&](int p) {
            for (int k = 0; k < n; k++)
                if (!fg.recordSynth(cb, ring[p % 3].view, ring[(p + 1) % 3].view, out.view, (k + 1.0f) / (n + 1.0f))) { fprintf(stderr, "synth failed\n"); exit(1); }
        });
        printf("  N=%d %.4f", n, t);
        nn.push_back(n); tt.push_back(t);
    }
    double mn = 0, mt = 0, sxy = 0, sxx = 0;
    for (size_t i = 0; i < nn.size(); i++) { mn += nn[i]; mt += tt[i]; }
    mn /= nn.size(); mt /= nn.size();
    for (size_t i = 0; i < nn.size(); i++) { sxy += (nn[i] - mn) * (tt[i] - mt); sxx += (nn[i] - mn) * (nn[i] - mn); }
    const double synthMs = sxy / sxx;
    printf("\n   => one synth ~ %.4f ms (least-squares slope over N = 1..8), fixed part %.4f ms\n", synthMs, mt - synthMs * mn);
    printf("   naive N x recordSynth + one blend pass (separate images), total ms per pair incl. the estimate:");
    std::vector<double> nb;
    for (int n = 2; n <= 8; n++) {
        double t = timePairs(fg, ring, P, [&](int p) {
            for (int k = 0; k < n; k++)
                if (!fg.recordSynth(cb, ring[p % 3].view, ring[(p + 1) % 3].view, subs[k].view, (k + 1.0f) / (n + 1.0f))) { fprintf(stderr, "synth failed\n"); exit(1); }
            recordBlend(n, W, H);
        });
        printf("  N=%d %.4f", n, t);
        nb.push_back(t);
    }
    printf("\n");
#ifndef BLURTEST_NO_BLUR
    auto blurRun = [&](int N, float w0, float w1, int outputs, BlurWeights bw) {
        Settings sb = s;
        sb.blurSamples = N; sb.blurWeights = bw;
        fg.setSettings(algo(sb));
        int PP = std::max(1, std::min(P, 32 / std::max(outputs, 1)));
        return timePairs(fg, ring, PP, [&](int p) {
            for (int o = 0; o < outputs; o++) {
                float c0 = (o + 0.5f) / outputs;   // outputs spread over the pair; window = [c - w/2, c + w/2] slid into [0, 1]
                float w = w1 - w0, a = std::min(std::max(c0 - w / 2, 0.0f), 1.0f - w);
                if (w0 > 0.0f || w1 < 1.0f) { /* explicit window */ }
                if (!fg.recordSynthBlur(cb, ring[p % 3].view, ring[(p + 1) % 3].view, out.view, outputs == 1 ? w0 : a, outputs == 1 ? w1 : a + w)) { fprintf(stderr, "blur failed\n"); exit(1); }
            }
        });
    };
    printf("   recordSynthBlur, ONE output per pair (total with the estimate / blur part only), per N:\n");
    for (int win = 0; win < 2; win++) {
        printf("     window %s:", win == 0 ? "[0.25,0.75] (all interpolated)" : "[0.5,1] (trailing, ends on the real frame)");
        for (int N = 2; N <= 8; N++) {
            double t = blurRun(N, win == 0 ? 0.25f : 0.5f, win == 0 ? 0.75f : 1.0f, 1, BlurWeights::Gaussian);
            printf("  N=%d %.4f/%.4f", N, t, t - est);
        }
        printf(" ms\n");
    }
    // the frame-generation-plus-blur case: M outputs per real pair, each blurred over N samples with a window of 0.25 pair
    printf("   M outputs per pair x N samples (window 0.25 of the pair, interpolated): blur total / naive N x M synths / plain M synths (blur off), ms per pair incl. the estimate\n");
    for (int M : {4, 5})
        for (int N : {2, 4, 8}) {
            double tb = blurRun(N, 0.0f, 0.25f, M, BlurWeights::Gaussian);
            double plain = timePairs(fg, ring, std::max(1, std::min(P, 32 / M)), [&](int p) {
                for (int o = 0; o < M; o++)
                    if (!fg.recordSynth(cb, ring[p % 3].view, ring[(p + 1) % 3].view, out.view, (o + 0.5f) / M)) { fprintf(stderr, "synth failed\n"); exit(1); }
            });
            double naive = M * N > 32 ? -1.0 : timePairs(fg, ring, std::max(1, std::min(P, 32 / (M * N))), [&](int p) {
                for (int o = 0; o < M * N; o++)
                    if (!fg.recordSynth(cb, ring[p % 3].view, ring[(p + 1) % 3].view, out.view, (o + 0.5f) / (M * N))) { fprintf(stderr, "synth failed\n"); exit(1); }
            });
            printf("     M=%d N=%d: blur %.4f   naive %.4f   plain %.4f   (blur adds %.4f ms over plain FG, %.4f per output)\n", M, N, tb, naive, plain, tb - plain, (tb - plain) / M);
        }
    {   // per-pass of one blurred output (profiling, isolated)
        Settings sb = s; sb.blurSamples = 4;
        fg.setSettings(algo(sb));
        fg.setProfiling(true);
        std::vector<std::vector<double>> per;
        std::vector<const char*> names;
        for (int it = 0; it < 14; it++) {
            begin(); fg.beginFrame();
            if (!fg.recordEstimate(cb, ring[0].view, ring[1].view) || !fg.recordSynthBlur(cb, ring[0].view, ring[1].view, out.view, 0.25f, 0.75f)) exit(1);
            submit();
            std::vector<uint64_t> ticks;
            if (!fg.readTimings(names, ticks)) exit(1);
            if (it < 4) continue;
            per.resize(ticks.size());
            for (size_t i = 0; i < ticks.size(); i++) per[i].push_back(ticks[i] * tsPeriodMs);
        }
        fg.setProfiling(false);
        printf("   passes of 1 pair + 1 blurred output (N=4, [0.25,0.75], isolated):");
        for (size_t i = 0; i < names.size(); i++) if (!strncmp(names[i], "blur", 4)) printf(" %s %.3f", names[i], median(per[i]));
        printf(" ms\n");
        fg.setSettings(algo(s));
    }
#endif
    vkDeviceWaitIdle(dev);
    vkDestroyQueryPool(dev, tsPool, nullptr);
    fg.destroy();
    for (auto& r : ring) destroyImg(r);
    for (auto& i : subs) destroyImg(i);
    destroyImg(out);
}



// ---------------------------------------------------------------- scene for the blur checks
// A textured panning background (the G channel is flat 130, so a G profile across the object's edge is noise free), one
// solid dark rectangle moving right at kObjV px per real frame, and the static crosshair + HUD box of the UI tests.
static const int kObjV = 16;   // object speed, px per real frame (the library tracks it; 24 is already beyond what it follows)
struct BScene {
    Scene sc;
    int W, H, objW = 200, objH = 120, xr10, oy;
    BScene(int w, int h) : sc(w, h, 12, 0, 0), W(w), H(h) {
        sc.objs.clear();
        for (size_t i = 0; i < sc.bg.size(); i += 4) sc.bg[i + 1] = 130;
        xr10 = (int)(W * 0.40);
        oy = (int)(H * 0.25);
    }
    // 1 within 4 px of a UI pixel
    std::vector<uint8_t> nearUi() const {
        std::vector<uint8_t> n((size_t)W * H, 0);
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                if (sc.ui[(size_t)y * W + x])
                    for (int dy = -4; dy <= 4; dy++)
                        for (int dx = -4; dx <= 4; dx++) {
                            int X = x + dx, Y = y + dy;
                            if (X >= 0 && Y >= 0 && X < W && Y < H) n[(size_t)Y * W + X] = 1;
                        }
        return n;
    }
    // integer frame index f: the object's right edge at frame f is xr10 + kObjV (f - 10)
    std::vector<uint8_t> frame(int f) const {
        std::vector<uint8_t> o = sc.frame((double)f);   // background + UI; no objects
        int xr = xr10 + kObjV * (f - 10);
        for (int y = oy; y < oy + objH; y++)
            for (int x = std::max(xr - objW, 0); x < std::min(xr, W); x++) {
                uint8_t* d = &o[((size_t)y * W + x) * 4];
                // R and B carry a checker texture that moves with the object (so the motion search has something to
                // lock on to); G is flat 8: the G profile across the edge is exact
                int u = (x - (xr - objW)) / 5, v = (y - oy) / 5;
                bool chk = (u + v) & 1;
                d[0] = chk ? 200 : 60; d[1] = 8; d[2] = ((u / 2 + v) & 1) ? 190 : 70; d[3] = 255;
            }
        for (size_t i = 0; i < (size_t)W * H; i++)   // the UI stays on top
            if (sc.ui[i]) memcpy(&o[i * 4], &sc.uiCol[i * 4], 4);
        return o;
    }
};

struct Ctx {
    BScene bs;
    int W, H;
    Interpolator fg;
    Img ring[4], outs[10];
    Img *prv = nullptr, *cur = nullptr;
    std::vector<uint8_t> real10, real11;
    Ctx(int w, int h, const Settings& s) : bs(w, h), W(w), H(h) {
        if (!makeInterp(fg, W, H, s)) { fprintf(stderr, "init failed\n"); exit(1); }
        for (auto& r : ring) r = makeImage(W, H, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        for (auto& o : outs) o = makeImage(W, H, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        begin();
        for (auto& r : ring) imgBarrier(r.img, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
        for (auto& o : outs) imgBarrier(o.img, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
        submit();
        for (int f = 0; f < 12; f++) {
            Img& c = ring[f % 4];
            auto fr = bs.frame(f);
            if (f == 10) real10 = fr;
            if (f == 11) real11 = fr;
            upload(fr, c, W, H);
            begin();
            fg.beginFrame();
            if (!fg.recordObserve(cb, c.view)) { fprintf(stderr, "observe failed\n"); exit(1); }
            if (f == 11) {
                prv = &ring[10 % 4]; cur = &c;
                if (!fg.recordEstimate(cb, prv->view, cur->view)) { fprintf(stderr, "estimate failed\n"); exit(1); }
            }
            submit();
        }
    }
    ~Ctx() {
        vkDeviceWaitIdle(dev);
        fg.destroy();
        for (auto& r : ring) destroyImg(r);
        for (auto& o : outs) destroyImg(o);
    }
    std::vector<uint8_t> get(int i) { return download(outs[i], W, H); }
};

static int fails = 0;
static void expect(bool ok, const char* what) {
    printf("  %s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) fails++;
}

static void synthOut(Ctx& c, int i, float t) {
    begin();
    c.fg.beginFrame();
    if (!c.fg.recordSynth(cb, c.prv->view, c.cur->view, c.outs[i].view, t)) { fprintf(stderr, "recordSynth(%g) failed\n", t); exit(1); }
    submit();
}

#ifndef BLURTEST_NO_BLUR
// the sample times and weights exactly as documented (framegen.h), for the CPU references
static std::vector<float> sampleTimes(int N, float t0, float t1) {
    std::vector<float> t(N);
    // midpoints of N equal slices of [t0, t1] (box-filter shutter); never t0 or t1 themselves
    for (int k = 0; k < N; k++) t[k] = std::min(std::max((float)(t0 + ((double)t1 - t0) * (k + 0.5) / N), t0), t1);
    return t;
}
static std::vector<double> sampleWeights(int N, BlurWeights w) {
    std::vector<double> g(N, 1.0);
    double sum = 0;
    for (int k = 0; k < N; k++) {
        double x = (2.0 * k + 1 - N) / N;   // slice centre; the window ends are x = -+1
        if (w == BlurWeights::Gaussian) g[k] = std::exp(-2.0 * x * x);
        sum += g[k];
    }
    for (auto& v : g) v /= sum;
    return g;
}

static Settings withBlur(Settings s, uint32_t n, BlurWeights w) { s.blurSamples = n; s.blurWeights = w; return s; }

// One blurred output into outs[i]
static void blurOut(Ctx& c, int i, float t0, float t1) {
    begin();
    c.fg.beginFrame();
    if (!c.fg.recordSynthBlur(cb, c.prv->view, c.cur->view, c.outs[i].view, t0, t1)) { fprintf(stderr, "recordSynthBlur(%g,%g) failed\n", t0, t1); exit(1); }
    submit();
}

// Edge profile of the object's right edge: c(x) = object share of column x (G channel, rows well inside the object's
// height), for x from xr10 - 10 to xr10 + kObjV + 10 (the object covers everything left of it). From it: the mean
// position mu of the edge (every sample's edge e_k weighted by w_k: mu = sum_x c(x) + x_start), its standard deviation
// sigma, and the extent of the columns with 0.02 < c < 0.98 (first / last such column: lo, hi; an image without one, a sharp
// step, has the step's column as both). The profile of a blur is the weighted sum of its samples' profiles, so its moments
// follow from theirs (a mixture: mean sum w_k mu_k, variance sum w_k (sigma_k^2 + (mu_k - mu)^2); sub-pixel, antialiasing-proof).
struct Edge { double mu, sigma, ext, lo, hi; };
static Edge measureEdge(const BScene& bs, const std::vector<uint8_t>& img) {
    int x0 = bs.xr10 - 10, x1 = bs.xr10 + kObjV + 10;
    std::vector<double> c;
    for (int x = x0; x <= x1; x++) {
        double s = 0;
        int rows = 0;
        for (int y = bs.oy + 15; y < bs.oy + bs.objH - 15; y++, rows++) s += img[((size_t)y * bs.W + x) * 4 + 1];
        c.push_back(1.0 - (s / rows - 8.0) / (130.0 - 8.0));
    }
    double m1 = 0, m2 = 0;
    for (size_t i = 0; i < c.size(); i++) { m1 += c[i]; m2 += (2.0 * i + 1.0) * c[i]; }
    Edge e;
    e.mu = x0 + m1;
    e.sigma = std::sqrt(std::max(m2 - m1 * m1, 0.0));
    double a = -1, b = -1;
    for (size_t i = 0; i < c.size(); i++) if (c[i] > 0.02 && c[i] < 0.98) { if (a < 0) a = (double)i; b = (double)i; }
    e.ext = a < 0 ? 0 : b - a + 1;
    if (a < 0) a = b = std::floor(m1);   // a sharp step: the column where the object ends
    e.lo = x0 + a; e.hi = x0 + b;
    return e;
}
// The same on the ideal model: object edges at xr10 + kObjV t_k, coverage min(max(e - x, 0), 1) per pixel
static Edge modelEdge(const BScene& bs, const std::vector<float>& t, const std::vector<double>& w) {
    int x0 = bs.xr10 - 10, x1 = bs.xr10 + kObjV + 10;
    std::vector<uint8_t> img((size_t)bs.W * bs.H * 4, 0);
    for (int y = bs.oy + 15; y < bs.oy + bs.objH - 15; y++)
        for (int x = x0; x <= x1; x++) {
            double cov = 0;
            for (size_t k = 0; k < t.size(); k++) cov += w[k] * std::min(std::max(bs.xr10 + kObjV * (double)t[k] - x, 0.0), 1.0);
            img[((size_t)y * bs.W + x) * 4 + 1] = (uint8_t)std::lround(8.0 + (1.0 - cov) * 122.0);
        }
    return measureEdge(bs, img);
}
#endif

static int check(int W, int H) {
#ifndef BLURTEST_NO_MOTION
    printf("check %dx%d %s, motion version %s (--algo)\n", W, H, bgra ? "BGRA8" : "RGBA8", gAlgo == MotionVersion::V2 ? "V2" : "V1");
#else
    printf("check %dx%d %s\n", W, H, bgra ? "BGRA8" : "RGBA8");
#endif
    const char* wn[2] = {"even", "gaussian"};
#ifdef BLURTEST_NO_BLUR
    (void)wn;
#endif
    // ---- 1. blur off: byte identical (FNV of generated frames; compare against a build of the library before blur)
    {
        printf("[hashes: protection Off / Crosshair / WholeScreen, t = 0.2 0.5 0.75 of the last pair]\n");
        for (int mi = 0; mi < BT_NMODES; mi++) {
            const UiProtection mode = btMode(mi);
            Settings s;
            s.uiProtection = mode; s.uiBoxWidth = s.uiBoxHeight = 64;   // fits the 35 px test crosshair
            Ctx c(W, H, s);
            printf("HASH mode %d:", (int)mode);
            float ts[3] = {0.2f, 0.5f, 0.75f};
            for (int i = 0; i < 3; i++) { synthOut(c, i, ts[i]); printf(" %016llx", (unsigned long long)fnv(c.get(i))); }
            printf("\n");
        }
#ifndef BLURTEST_NO_BLUR
        // the blurred outputs too (4 Gaussian samples; windows 0-0.5, 0.25-0.75, 0.5-1 of the last pair): diff two builds of the library
        for (int mi = 0; mi < BT_NMODES; mi++) {
            const UiProtection mode = btMode(mi);
            Settings s;
            s.uiProtection = mode; s.uiBoxWidth = s.uiBoxHeight = 64;
            Ctx c(W, H, withBlur(s, 4, BlurWeights::Gaussian));
            printf("HASH blur mode %d:", (int)mode);
            float w0[3] = {0.0f, 0.25f, 0.5f};
            for (int i = 0; i < 3; i++) { blurOut(c, i, w0[i], w0[i] + 0.5f); printf(" %016llx", (unsigned long long)fnv(c.get(i))); }
            printf("\n");
        }
#endif
    }
#ifdef BLURTEST_NO_BLUR
    return 0;
#else
    // ---- 2. blur off / single instant == recordSynth
    {
        Settings s;
        s.uiProtection = UiProtection::Crosshair; s.uiBoxWidth = s.uiBoxHeight = 64;
        s.blurSamples = 0;
        Ctx c(W, H, s);
        synthOut(c, 0, 0.3f);
        blurOut(c, 1, 0.1f, 0.3f);   // blur off: the window collapses to its end
        expect(c.get(0) == c.get(1), "blur off: recordSynthBlur(t0, t1) == recordSynth(t1) (bit-exact, UI protection on)");
        Settings b = withBlur(s, 4, BlurWeights::Gaussian);
        c.fg.setSettings(algo(b));
        blurOut(c, 2, 0.3f, 0.3f);
        expect(c.get(0) == c.get(2), "blur on, t0 == t1: == recordSynth(t) (bit-exact)");
        blurOut(c, 3, 0.5f, 0.5f);
        synthOut(c, 4, 0.5f);
        expect(c.get(3) == c.get(4), "blur on, t0 == t1 == 0.5: == recordSynth(0.5) (bit-exact)");
        // a real-frame instant: the UI pixels are the real ones (<= 1 level), everything further than 4 px from them is
        // bit-identical to the real frame (the ring right around a protected UI is the library's inpainted clean copy,
        // as in every other fallback of the library)
        auto sameAsReal = [&](const std::vector<uint8_t>& g, const std::vector<uint8_t>& real, const std::vector<uint8_t>& nu) {
            size_t badFar = 0, badUi = 0;
            for (size_t i = 0; i < (size_t)W * H; i++) {
                int d = 0;
                for (int ch = 0; ch < 3; ch++) d = std::max(d, std::abs((int)g[i * 4 + ch] - (int)real[i * 4 + ch]));
                if (c.bs.sc.ui[i]) badUi += d > 1;
                else if (!nu[i]) badFar += d != 0;
            }
            return badFar == 0 && badUi == 0;
        };
        auto nu = c.bs.nearUi();
        blurOut(c, 5, 1.0f, 1.0f);
        expect(sameAsReal(c.get(5), c.real11, nu), "t0 == t1 == 1: the real frame curr (UI pixels <= 1 level, bit-exact beyond 4 px of them)");
        blurOut(c, 6, 0.0f, 0.0f);
        
        {   // t = 0 shows prev: UI pixels come from curr (static, identical), the rest is prev
            auto g = c.get(6);
            size_t badFar = 0;
            for (size_t i = 0; i < (size_t)W * H; i++)
                if (!c.bs.sc.ui[i] && !nu[i]) for (int ch = 0; ch < 3; ch++) badFar += g[i * 4 + ch] != c.real10[i * 4 + ch];
            expect(badFar == 0, "t0 == t1 == 0: the real frame prev (bit-exact beyond 4 px of the UI)");
        }
    }
    // ---- 3. the blur equals the average of the sub-frames recordSynth produces (protection Off: the paste-back is
    //         applied to a recordSynth output but after the average of a blur, so they only agree without it).
    //         Also with the whole-frame fallbacks forced: a scene cut (sceneCutSad tiny) and "too much untrusted"
    //         (globalFallback tiny / small: some samples of a window flicked, others not)
    {
        printf("[blur == CPU average of N recordSynth outputs / real frames, N = 2..8, 5 windows, both weights]\n");
        struct Var { const char* name; float cut, flick; };
        const Var vars[] = {{"default settings", 30.0f, 0.15f}, {"scene cut forced", 0.5f, 0.15f}, {"flick forced (0.1% untrusted)", 30.0f, 0.001f},
                            {"flick on some samples (1.5%)", 30.0f, 0.015f}, {"flick on some samples (4%)", 30.0f, 0.04f}};
        for (const Var& var : vars) {
            Settings s;
            s.sceneCutSad = var.cut; s.globalFallback = var.flick;
            Ctx c(W, H, s);
            const float wins[5][2] = {{0.5f, 1.0f}, {0.25f, 0.75f}, {0.0f, 1.0f}, {0.1f, 0.4f}, {0.6f, 0.9f}};
            int worst = 0;
            size_t tot = 0, off1 = 0;
            bool allOk = true;
            for (int N = 2; N <= 8; N++)
                for (auto& wd : wins) {
                    if (&var != &vars[0] && (N != 3 && N != 4 && N != 7)) continue;   // the fallback variants: a subset of N
                    auto t = sampleTimes(N, wd[0], wd[1]);
                    std::vector<std::vector<uint8_t>> sub(N);
                    c.fg.setSettings(algo(withBlur(s, N, BlurWeights::Even)));
                    for (int k = 0; k < N; k++) {
                        if (t[k] <= 0.0f) sub[k] = c.real10;
                        else if (t[k] >= 1.0f) sub[k] = c.real11;
                        else { synthOut(c, 0, t[k]); sub[k] = c.get(0); }
                    }
                    for (int wi = 0; wi < 2; wi++) {
                        BlurWeights bw = wi ? BlurWeights::Gaussian : BlurWeights::Even;
                        c.fg.setSettings(algo(withBlur(s, N, bw)));
                        blurOut(c, 1, wd[0], wd[1]);
                        auto got = c.get(1);
                        auto w = sampleWeights(N, bw);
                        int mx = 0;
                        size_t n1 = 0;
                        for (size_t i = 0; i < got.size(); i++) {
                            double a = 0;
                            for (int k = 0; k < N; k++) a += w[k] * sub[k][i];
                            int d = std::abs((int)got[i] - (int)std::floor(a + 0.5));
                            mx = std::max(mx, d);
                            n1 += d > 0;
                        }
                        worst = std::max(worst, mx);
                        tot += got.size(); off1 += n1;
                        if (mx > 1) { allOk = false; printf("   N=%d [%g,%g] %s: max diff %d, %zu bytes differ\n", N, wd[0], wd[1], wn[wi], mx, n1); }
                    }
                }
            printf("   %s: worst difference %d level(s); %zu of %zu bytes (%.5f %%) differ by 1\n", var.name, worst, off1, tot, 100.0 * off1 / tot);
            expect(allOk, (std::string("blur == average of the recordSynth sub-frames (<= 1 level): ") + var.name).c_str());
        }
    }
    {   // the fix-up pass really ran: a forced flick changes the output (nearer real frames instead of interpolation)
        Settings s0, s1;
        s1.globalFallback = 0.001f;
        Ctx c0(W, H, s0), c1(W, H, s1);
        c0.fg.setSettings(algo(withBlur(s0, 4, BlurWeights::Even)));
        c1.fg.setSettings(algo(withBlur(s1, 4, BlurWeights::Even)));
        blurOut(c0, 0, 0.25f, 0.75f);
        blurOut(c1, 0, 0.25f, 0.75f);
        auto a = c0.get(0), b = c1.get(0);
        size_t diff = 0;
        for (size_t i = 0; i < a.size(); i++) diff += a[i] != b[i];
        printf("   forced flick (globalFallback 0.001) changes %zu bytes of the N=4 [0.25,0.75] output\n", diff);
        expect(diff > 0, "the whole-frame flick fallback is applied by the fix-up pass");
    }
    // ---- 4. the edge of a moving object smears over speed x (t1 - t0); the crosshair stays pixel-exact
    for (int mode = 0; mode < BT_NMODES; mode++) {
        UiProtection pm = btMode(mode);
        printf("[edge smear, protection %s]\n", btModeName(mode));
        Settings s;
        s.uiProtection = pm; s.uiBoxWidth = s.uiBoxHeight = 64;
        Ctx c(W, H, s);
        // a plain interpolated frame for reference
        synthOut(c, 0, 0.75f);
        Edge e0 = measureEdge(c.bs, c.get(0));
        bool xhOk = true, muOk = true, sigOk = true, extOk = true;
        double worstMu = 0, worstSig = 0, worstExt = -1e9, worstIdealSig = 0;
        for (int N : {2, 3, 4, 5, 6, 7, 8})
            for (auto& wd : std::vector<std::array<float, 2>>{{0.5f, 1.0f}, {0.25f, 0.75f}, {0.0f, 1.0f}}) {
                auto t = sampleTimes(N, wd[0], wd[1]);
                // the edges of the single frames this window is made of, measured (their own position, spread and extent:
                // an interpolated frame's edge need not sit at the ideal xr10 + speed t, e.g. V2's soft hold edge, preset B,
                // spreads a semi-transparent strip on both sides of it, biased toward the nearer real frame)
                std::vector<Edge> se(N);
                for (int k = 0; k < N; k++) {
                    if (t[k] <= 0.0f) se[k] = measureEdge(c.bs, c.real10);
                    else if (t[k] >= 1.0f) se[k] = measureEdge(c.bs, c.real11);
                    else { synthOut(c, 2, t[k]); se[k] = measureEdge(c.bs, c.get(2)); }
                }
                for (int wi = 0; wi < 2; wi++) {
                    BlurWeights bw = wi ? BlurWeights::Gaussian : BlurWeights::Even;
                    c.fg.setSettings(algo(withBlur(s, N, bw)));
                    blurOut(c, 1, wd[0], wd[1]);
                    auto img = c.get(1);
                    auto w = sampleWeights(N, bw);
                    Edge m = measureEdge(c.bs, img), r = modelEdge(c.bs, t, w);
                    // the prediction from the sub-frames' measured moments: the blur's profile is their weighted mixture
                    double muP = 0, var = 0;
                    for (int k = 0; k < N; k++) muP += w[k] * se[k].mu;
                    for (int k = 0; k < N; k++) var += w[k] * (se[k].sigma * se[k].sigma + (se[k].mu - muP) * (se[k].mu - muP));
                    double sigPred = std::sqrt(var);
                    double idealVar = r.sigma * r.sigma;   // (info: the ideal model + each frame's softness, exact only for edges at xr10 + speed t)
                    for (int k = 0; k < N; k++) idealVar += w[k] * se[k].sigma * se[k].sigma;
                    double dMu = std::fabs(m.mu - r.mu), dSig = std::fabs(m.sigma - sigPred);
                    worstMu = std::max(worstMu, dMu); worstSig = std::max(worstSig, dSig);
                    worstIdealSig = std::max(worstIdealSig, std::fabs(m.sigma - std::sqrt(idealVar)));
                    muOk = muOk && dMu <= 2.0;
                    sigOk = sigOk && dSig <= 0.1;
                    if (dMu > 2.0 || dSig > 0.1)
                        printf("   N=%d [%g,%g] %s: mean edge %.2f (model %.2f), sigma %.2f (from the sub-frames: %.2f)\n", N, wd[0], wd[1], wn[wi], m.mu, r.mu, m.sigma, sigPred);
                    if (wi == 0) {
                        // even weights (each >= 1/8: every sub-frame's transition shows in the blur's): the smear extent
                        // spans at least the sub-frames' mean edges (max mu_k - min mu_k, the measured counterpart of
                        // speed x (t1 - t0) x (N - 1)/N) and lies within the union of their transitions [min lo_k, max hi_k]
                        double muLo = 1e9, muHi = -1e9, uLo = 1e9, uHi = -1e9;
                        for (int k = 0; k < N; k++) {
                            muLo = std::min(muLo, se[k].mu); muHi = std::max(muHi, se[k].mu);
                            uLo = std::min(uLo, se[k].lo); uHi = std::max(uHi, se[k].hi);
                        }
                        double D = muHi - muLo, U = uHi - uLo + 1;
                        bool ok = m.ext >= D - 1.0 && m.lo >= uLo - 1.0 && m.hi <= uHi + 1.0;
                        worstExt = std::max(worstExt, std::max(D - 1.0 - m.ext, std::max(uLo - 1.0 - m.lo, m.hi - uHi - 1.0)));
                        if (!ok) { extOk = false; printf("   N=%d [%g,%g] even: extent %.1f px [%g, %g], sub-frame mean edges span %.1f px, union of their transitions [%g, %g] (%.0f px)\n", N, wd[0], wd[1], m.ext, m.lo, m.hi, D, uLo, uHi, U); }
                    }
                    if (pm != UiProtection::Off) {
                        size_t bad = 0;
                        for (size_t i = 0; i < (size_t)W * H; i++)
                            if (c.bs.sc.ui[i] == 1) {
                                int d = 0;
                                for (int ch = 0; ch < 3; ch++) d = std::max(d, std::abs((int)img[i * 4 + ch] - (int)c.real11[i * 4 + ch]));
                                bad += d > 1;
                            }
                        xhOk = xhOk && bad == 0;
                        if (bad) printf("   N=%d [%g,%g] %s: %zu crosshair pixels differ\n", N, wd[0], wd[1], wn[wi], bad);
                    }
                }
            }
        printf("   plain recordSynth(0.75): edge sigma %.2f px; worst |mean - model| %.2f px, worst |sigma - from sub-frames| %.3f px (vs the ideal model + softness: %.2f px), worst extent bound violation %.2f px (<= 0: inside)\n", e0.sigma, worstMu, worstSig, worstIdealSig, worstExt);
        expect(muOk, "mean edge position within 2 px of the ideal model (N = 2..8, even + gaussian, 3 windows)");
        expect(sigOk, "edge spread (sigma) within 0.1 px of the mixture of the measured sub-frame edges: sqrt(sum w_k (sigma_k^2 + (mu_k - mu)^2))");
        expect(extOk, "even weights: smear extent >= sub-frame mean-edge span - 1 px, within the union of the sub-frame transitions +- 1 px");
        if (pm != UiProtection::Off) expect(xhOk, "crosshair pixels equal the real frame (<= 1 level) in every blurred output");
    }
    // ---- 4b. midpoint rule: windows that tile (100 % blur) share no sample, and the weights cover the interval uniformly
    {
        printf("[midpoint sampling: tiling at 100 %%]\n");
        bool noEnds = true, tiles = true, gaussSym = true;
        for (int N : {2, 3, 4, 5, 8, 16}) {
            auto a = sampleTimes(N, 0.0f, 0.5f), b = sampleTimes(N, 0.5f, 1.0f), z = sampleTimes(N, 0.0f, 1.0f);
            for (float t : z) noEnds = noEnds && t > 0.0f && t < 1.0f;
            // adjacent windows [0, .5] and [.5, 1]: no shared instant, and together they are the 2N midpoints of the pair
            for (int k = 0; k < N; k++) {
                tiles = tiles && a[k] < 0.5f && b[k] > 0.5f;
                tiles = tiles && std::fabs(a[k] - (k + 0.5) / (2.0 * N)) < 1e-6 && std::fabs(b[k] - (0.5 + (k + 0.5) / (2.0 * N))) < 1e-6;
            }
            auto gw = sampleWeights(N, BlurWeights::Gaussian);
            for (int k = 0; k < N; k++) gaussSym = gaussSym && std::fabs(gw[k] - gw[N - 1 - k]) < 1e-12;
        }
        expect(noEnds, "midpoint rule: no sample of a blurred window is exactly t = 0 or t = 1");
        expect(tiles, "windows [0, .5] and [.5, 1] share no sample; together = 2N uniform midpoints of the pair");
        expect(gaussSym, "Gaussian weights are symmetric about the window centre");
        // GPU: two tiled outputs of the same pair; each real frame is blended in only through interpolated samples, so
        // output([0,1]) (Even, N = 8) equals the average of output([0,.5]) and output([.5,1]) (Even, N = 4) up to rounding
        Settings s; s.uiProtection = UiProtection::Off;
        Ctx c(W, H, s);
        c.fg.setSettings(algo(withBlur(s, 4, BlurWeights::Even)));
        blurOut(c, 0, 0.0f, 0.5f); blurOut(c, 1, 0.5f, 1.0f);
        auto ha = c.get(0), hb = c.get(1);
        c.fg.setSettings(algo(withBlur(s, 8, BlurWeights::Even)));
        blurOut(c, 2, 0.0f, 1.0f);
        auto whole = c.get(2);
        int worst = 0;
        for (size_t i = 0; i < (size_t)W * H * 4; i++) { if (i % 4 == 3) continue; worst = std::max(worst, std::abs((int)whole[i] - (((int)ha[i] + (int)hb[i] + 1) >> 1))); }
        printf("   N=8 [0,1] vs mean of N=4 [0,.5] and [.5,1]: worst difference %d levels\n", worst);
        expect(worst <= 2, "tiled windows: one 8-sample [0,1] output == the mean of the two 4-sample half windows (<= 2 levels)");
    }
    // ---- 5. rings: 8 outputs x 8 samples in one pair; mixed N in one submission; 32 outputs; the 33rd is refused
    {
        printf("[descriptor rings]\n");
        for (int mode = 0; mode < BT_NMODES; mode++) {
            Settings s;
            s.uiProtection = btMode(mode);
            s.uiBoxWidth = s.uiBoxHeight = 64;
            s.blurSamples = 8;
            Ctx c(W, H, s);
            begin();
            c.fg.beginFrame();
            bool ok = true;
            for (int i = 0; i < 8; i++) ok = ok && c.fg.recordSynthBlur(cb, c.prv->view, c.cur->view, c.outs[i].view, i / 8.0f, std::min(1.0f, (i + 1) / 8.0f + 0.25f));
            submit();
            expect(ok, (std::string("8 outputs x 8 samples in one pair, one submission (") + btModeName(mode) + ")").c_str());
            if (mode == 0) {   // mixed N in one submission (the scratch grows between records): each equals its own single record
                Settings s2 = s;
                s2.blurSamples = 2;
                c.fg.setSettings(algo(s2));
                begin();
                c.fg.beginFrame();
                bool a = c.fg.recordSynthBlur(cb, c.prv->view, c.cur->view, c.outs[0].view, 0.25f, 0.75f);
                s2.blurSamples = 8;
                c.fg.setSettings(algo(s2));
                bool b = c.fg.recordSynthBlur(cb, c.prv->view, c.cur->view, c.outs[1].view, 0.25f, 0.75f);
                s2.blurSamples = 3;
                c.fg.setSettings(algo(s2));
                bool d = c.fg.recordSynthBlur(cb, c.prv->view, c.cur->view, c.outs[2].view, 0.25f, 0.75f);
                submit();
                auto o0 = c.get(0), o1 = c.get(1), o2 = c.get(2);
                s2.blurSamples = 2; c.fg.setSettings(algo(s2)); blurOut(c, 3, 0.25f, 0.75f);
                s2.blurSamples = 8; c.fg.setSettings(algo(s2)); blurOut(c, 4, 0.25f, 0.75f);
                s2.blurSamples = 3; c.fg.setSettings(algo(s2)); blurOut(c, 5, 0.25f, 0.75f);
                expect(a && b && d && o0 == c.get(3) && o1 == c.get(4) && o2 == c.get(5), "N = 2, 8, 3 recorded in ONE submission (scratch growing) == each recorded alone");
            }
            if (mode == 0) {
                begin();
                c.fg.beginFrame();
                int okn = 0;
                for (int i = 0; i < 32; i++) okn += c.fg.recordSynthBlur(cb, c.prv->view, c.cur->view, c.outs[i % 10].view, 0.25f, 0.75f);
                // the 33rd would assert in a debug build; this build has NDEBUG (release) and returns false
#ifdef NDEBUG
                bool r33 = c.fg.recordSynthBlur(cb, c.prv->view, c.cur->view, c.outs[0].view, 0.25f, 0.75f);
#else
                bool r33 = false;
#endif
                submit();
                expect(okn == 32 && !r33, "32 blurred outputs per submission fit the synth ring, the 33rd is refused");
            }
        }
    }
    // ---- 6. settings validation
    {
        Settings bad;
        bad.blurSamples = 17; expect(!Interpolator::validate(bad), "validate rejects blurSamples 17");
        bad = Settings{}; bad.blurSamples = 16; expect(Interpolator::validate(bad), "validate accepts blurSamples 16");
        bad = Settings{}; bad.blurSamples = 1; expect(Interpolator::validate(bad), "validate accepts blurSamples 1 (= off)");
        bad = Settings{}; bad.blurWeights = (BlurWeights)2; expect(!Interpolator::validate(bad), "validate rejects an unknown blurWeights");
        Settings d;
        expect(d.blurSamples == 0 && d.blurWeights == BlurWeights::Gaussian, "defaults: blurSamples 0 (off), Gaussian");
        Settings s; Ctx c(W, H, s);
        expect(!c.fg.recordSynthBlur(cb, c.prv->view, c.cur->view, c.outs[0].view, 0.7f, 0.2f), "t0 > t1 refused");
        expect(!c.fg.recordSynthBlur(cb, c.prv->view, c.cur->view, c.outs[0].view, -0.1f, 0.5f), "t0 < 0 refused");
        expect(!c.fg.recordSynthBlur(cb, c.prv->view, c.cur->view, c.outs[0].view, 0.5f, 1.1f), "t1 > 1 refused");
    }
    return fails ? 1 : 0;
#endif
}

int main(int argc, char** argv) {
    std::vector<std::string> pos;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bgra")) bgra = true;
#ifndef BLURTEST_NO_MOTION
        else if (!strcmp(argv[i], "--algo") && i + 1 < argc) {
            const char* a = argv[++i];
            if (!strcmp(a, "v1")) gAlgo = MotionVersion::V1;
            else if (!strcmp(a, "v2")) gAlgo = MotionVersion::V2;
            else { fprintf(stderr, "--algo v1|v2\n"); return 2; }
        }
        else if (!strcmp(argv[i], "--preset") && i + 1 < argc) {
            const char c = argv[++i][0];
            if (c < 'a' || c > 'd') { fprintf(stderr, "--preset a|b|c|d\n"); return 2; }
            gPreset = (MotionPreset)(c - 'a');
        }
#endif
        else pos.push_back(argv[i]);
    }
    if (pos.empty() || (pos[0] != "check" && pos[0] != "bench")) { fprintf(stderr, "usage: blurtest check|bench [--bgra] [--algo v1|v2] [--preset a|b|c|d] [W H]\n"); return 2; }
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    VkInstance inst;
    CK(vkCreateInstance(&ici, nullptr, &inst));
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(inst, &n, pds.data());
    phys = pds[0];
    for (auto p : pds) { VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(p, &pp); if (pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { phys = p; break; } }
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);
    tsPeriodMs = props.limits.timestampPeriod * 1e-6;
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, nullptr);
    std::vector<VkQueueFamilyProperties> qf(n);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, qf.data());
    uint32_t qfi = 0;
    for (uint32_t i = 0; i < n; i++) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = i; break; }
    fprintf(stderr, "device: %s\n", props.deviceName);
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, qfi, 1, &prio};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    CK(vkCreateDevice(phys, &dci, nullptr, &dev));
    vkGetDeviceQueue(dev, qfi, 0, &queue);
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qfi};
    VkCommandPool cpool;
    CK(vkCreateCommandPool(dev, &cpci, nullptr, &cpool));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, cpool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    CK(vkAllocateCommandBuffers(dev, &cbai, &cb));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    CK(vkCreateFence(dev, &fci, nullptr, &fence));

    int rc = 0;
    initStaging((size_t)2560 * 1440 * 4);
    if (pos[0] == "bench") {
        std::vector<std::pair<int, int>> sizes = {{1280, 960}, {1920, 1080}, {2560, 1440}};
        if (pos.size() >= 3) sizes = {{atoi(pos[1].c_str()), atoi(pos[2].c_str())}};
        for (auto [w, h] : sizes)
            for (uint32_t fs : {2u, 4u}) benchOne(w, h, fs, Settings{});
    }
    if (pos[0] == "check") {
        int W = pos.size() >= 3 ? atoi(pos[1].c_str()) : 1280, H = pos.size() >= 3 ? atoi(pos[2].c_str()) : 720;
        rc = check(W, H);
        printf(rc ? "RESULT: FAIL (%d)\n" : "RESULT: PASS\n", fails);
    }
    vkDeviceWaitIdle(dev);
    return rc;
}
