// mltest: MotionVersion::Learned against ml-fg's test vectors, and its duplicate-frame skip (minitest cases).
//   mltest [--list | substring]   run every case (or those whose name / tag contains the substring)
// Environment:
//   MLFG_EXPORT  ml-fg's export directory (default /home/mo/Schinken/Programming/Local/ml-fg/export): <run>/testvec/<pair>/ holds
//                a.png, c.png, vec.json and the expected float32 arrays (SPEC.md section 8). The vectors are read in place, never
//                copied into this repo. Cases whose vectors are missing fail with the path.
//   MLTEST_DUMP  a directory (created): per net and pair the library's pf, `<run>_<pair>_{ref,fgr}.f32` ([fh][fw][2] float32) and
//                `<run>_<pair>_{ref,fgr}.json` (sad, cut; also refh / fgrh = learnedHalf, and the zero test's kinds below), for gpu/ml/synthcheck.py (the end-to-end dB check through proto/interp.py).
//
// c2smoke (the stand-in of the colour variant, ml-fg export/c2smoke-colour, SPEC.md section 10): the nine pairs of e2v5 plus `colour`,
// 25 input channels (in_top), the flow-res chroma planes (ca.f32 / cc.f32) read back from the net's chroma pyramids, the max pf
// tolerance 0.15 px.
//
// Per net (e2v3: static motion far cut odd hd overlay; e1v3: motion cut odd overlay; e2v4: static motion far cut cut2 odd hd overlay;
// e1v4: motion cut cut2 odd overlay; e2v5: static motion far cut cut2 odd hd overlay; e1v5: motion cut cut2 odd overlay) and pair,
// SPEC.md section 8 in order (steps 1-6 with Settings::learnedZeroTest off: the net's own field):
//   ref  the library's net fed the reference's flow-res lumas (cv2 5.0.0's 15-bit luma, computed here; Interpolator::
//        setDebugLearned lumaSrc): the exact-port check. The lumas the net read == ya / yc exactly, its level-0 pyramid == ya / 255
//        and every coarser level == the 2x2 means (bit-exact, computed here); the coarsest level's in_top, act_top0..3, raw_top
//        (abs <= 5e-5, via stopLevel / stopStep); h_level4..1 and pf (mean abs <= 2e-4 px, max abs <= 5e-2 px); sad relative
//        <= 1e-3; the identical cut decision.
//   fgr  the production path: the library's own 14-bit luma (P_LUMA). Its lumas differ from ya / yc by multiples of 0.25 on < 1 %
//        of the pixels (SPEC.md section 7); pf / sad are printed (SPEC.md: a pointwise pf check does not apply, judge it end to end
//        with synthcheck.py: >= 45 dB vs the synthesis of the expected pf); the cut decision must still be identical.
// Then the zero-motion test (Settings::learnedZeroTest, SPEC.md stage 7 / section 8 step 7) against the *_zt vectors: zeroChecks().
// MLTEST_DUMP also gets its fields, kinds stagezt / refzt / fgrzt / refhzt / fgrhzt (synthcheck.py compares them with pf_zt.f32).
// Duplicates: two identical frames (and a near-duplicate below FGR's DUP_DIFF) give a zero field, a zero SAD sum and, through the
// unchanged synthesis, an output identical to the input; a pair above the threshold runs the net.
#include "framegen.h"
#include "minitest.h"

#include <png.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d\n", #x, r_); exit(1); } } while (0)

using namespace framegen;

// ---- tolerances: ml-fg reference.py PF_MAX_TOL, PF_MEAN_TOL, SAD_TOL, TOP_TOL (SPEC.md section 8)
// (colour nets: SPEC.md section 8 / 10, reference.py PF_MAX_TOL_COLOUR: the max pf tolerance is 0.15 px, they amplify fp32 noise ~3x more;
// the flow-res chroma planes within CHROMA_TOL levels)
static const double kPfMaxTol = 5e-2, kPfMaxTolColour = 1.5e-1, kPfMeanTol = 2e-4, kSadTol = 1e-3, kTopTol = 5e-5, kChromaTol = 2e-3;

// ---- Vulkan context, created by the first case that needs it
static VkInstance inst;
static VkPhysicalDevice phys;
static VkDevice dev;
static VkQueue queue;
static VkCommandPool cpool;
static VkCommandBuffer cb;
static VkFence fence;
static VkPhysicalDeviceMemoryProperties memProps;

static uint32_t memType(uint32_t bits, VkMemoryPropertyFlags f) {
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & f) == f) return i;
    fprintf(stderr, "no memory type\n");
    exit(1);
}

static void gpuInit() {
    if (dev) return;
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.pApplicationName = "mltest";
    ai.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    CK(vkCreateInstance(&ici, nullptr, &inst));
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(inst, &n, pds.data());
    phys = pds[0];
    for (auto p : pds) {
        VkPhysicalDeviceProperties pp;
        vkGetPhysicalDeviceProperties(p, &pp);
        if (pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { phys = p; break; }
    }
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, nullptr);
    std::vector<VkQueueFamilyProperties> qf(n);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, qf.data());
    uint32_t qfi = 0;
    for (uint32_t i = 0; i < n; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = i; break; }
    printf("device: %s\n", props.deviceName);
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, qfi, 1, &prio};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    CK(vkCreateDevice(phys, &dci, nullptr, &dev));
    vkGetDeviceQueue(dev, qfi, 0, &queue);
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qfi};
    CK(vkCreateCommandPool(dev, &cpci, nullptr, &cpool));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, cpool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    CK(vkAllocateCommandBuffers(dev, &cbai, &cb));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    CK(vkCreateFence(dev, &fci, nullptr, &fence));
}
static void begin() {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CK(vkBeginCommandBuffer(cb, &bi));
}
static void submit() {
    CK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    CK(vkQueueSubmit(queue, 1, &si, fence));
    CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
    CK(vkResetFences(dev, 1, &fence));
}
static void hostBarrier() {
    VkMemoryBarrier hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, nullptr, 0, nullptr);
}

// a host-visible buffer (staging, debug copies)
struct HBuf {
    VkBuffer b = VK_NULL_HANDLE;
    VkDeviceMemory m = VK_NULL_HANDLE;
    uint8_t* map = nullptr;
    size_t size = 0;
    void make(size_t bytes) {
        free();
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = bytes;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        CK(vkCreateBuffer(dev, &bci, nullptr, &b));
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(dev, b, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size,
                                 memType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
        CK(vkAllocateMemory(dev, &mai, nullptr, &m));
        CK(vkBindBufferMemory(dev, b, m, 0));
        CK(vkMapMemory(dev, m, 0, bytes, 0, (void**)&map));
        size = bytes;
    }
    void free() {
        if (b) { vkDestroyBuffer(dev, b, nullptr); vkFreeMemory(dev, m, nullptr); }
        b = VK_NULL_HANDLE; m = VK_NULL_HANDLE; map = nullptr; size = 0;
    }
};

struct Img { VkImage img = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE; };
static Img makeImage(uint32_t w, uint32_t h, VkImageUsageFlags usage) {
    Img r;
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D; ici.format = VK_FORMAT_R8G8B8A8_UNORM; ici.extent = {w, h, 1};
    ici.mipLevels = ici.arrayLayers = 1; ici.samples = VK_SAMPLE_COUNT_1_BIT; ici.tiling = VK_IMAGE_TILING_OPTIMAL; ici.usage = usage;
    CK(vkCreateImage(dev, &ici, nullptr, &r.img));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev, r.img, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size, memType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    CK(vkAllocateMemory(dev, &mai, nullptr, &r.mem));
    CK(vkBindImageMemory(dev, r.img, r.mem, 0));
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = r.img; vci.viewType = VK_IMAGE_VIEW_TYPE_2D; vci.format = VK_FORMAT_R8G8B8A8_UNORM;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    CK(vkCreateImageView(dev, &vci, nullptr, &r.view));
    return r;
}
static void destroyImg(Img& i) {
    vkDestroyImageView(dev, i.view, nullptr);
    vkDestroyImage(dev, i.img, nullptr);
    vkFreeMemory(dev, i.mem, nullptr);
    i = Img{};
}
static void imgBarrier(VkImage img, VkImageLayout from, VkAccessFlags srcA, VkAccessFlags dstA) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = srcA; b.dstAccessMask = dstA; b.oldLayout = from; b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.image = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
}
// RGBA8 bytes -> img (from UNDEFINED the first time)
static void upload(HBuf& stg, const std::vector<uint8_t>& rgba, Img& img, uint32_t w, uint32_t h, bool fresh) {
    if (stg.size < rgba.size()) stg.make(rgba.size());
    memcpy(stg.map, rgba.data(), rgba.size());
    VkBufferImageCopy rg{};
    rg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    rg.imageExtent = {w, h, 1};
    begin();
    imgBarrier(img.img, fresh ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    vkCmdCopyBufferToImage(cb, stg.b, img.img, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
    imgBarrier(img.img, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    submit();
}
static std::vector<uint8_t> download(HBuf& stg, Img& img, uint32_t w, uint32_t h) {
    const size_t bytes = (size_t)w * h * 4;
    if (stg.size < bytes) stg.make(bytes);
    VkBufferImageCopy rg{};
    rg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    rg.imageExtent = {w, h, 1};
    begin();
    imgBarrier(img.img, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    vkCmdCopyImageToBuffer(cb, img.img, VK_IMAGE_LAYOUT_GENERAL, stg.b, 1, &rg);
    hostBarrier();
    submit();
    return std::vector<uint8_t>(stg.map, stg.map + bytes);
}

// ---- test vectors
static std::string exportDir() {
    const char* e = getenv("MLFG_EXPORT");
    return e && *e ? e : "/home/mo/Schinken/Programming/Local/ml-fg/export";
}
// the export directory of a net: its name, except the test stand-in of the colour variant (the real g2v2 / m2v2 are g2v2 / m2v2)
static std::string runDir(const std::string& run) { return run == "c2smoke" ? "c2smoke-colour" : run; }
static bool readFile(const std::string& path, std::vector<uint8_t>& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.resize((size_t)n);
    bool ok = fread(out.data(), 1, out.size(), f) == out.size();
    fclose(f);
    return ok;
}
static bool readF32(const std::string& path, std::vector<float>& out) {
    std::vector<uint8_t> b;
    if (!readFile(path, b) || b.size() % 4) return false;
    out.resize(b.size() / 4);
    memcpy(out.data(), b.data(), b.size());
    return true;
}
// PNG -> RGBA8 (alpha 255), no colour management (the vectors are plain 8-bit RGB)
static bool readPng(const std::string& path, std::vector<uint8_t>& rgba, uint32_t& w, uint32_t& h) {
    png_image im{};
    im.version = PNG_IMAGE_VERSION;
    if (!png_image_begin_read_from_file(&im, path.c_str())) return false;
    im.format = PNG_FORMAT_RGBA;
    w = im.width; h = im.height;
    rgba.resize(PNG_IMAGE_SIZE(im));
    return png_image_finish_read(&im, nullptr, rgba.data(), 0, nullptr) != 0;
}
// the number after "key": in vec.json (flat keys only: sad, cut)
static double jsonNum(const std::string& js, const char* key) {
    std::string k = std::string("\"") + key + "\":";
    size_t p = js.find(k);
    if (p == std::string::npos) return NAN;
    const char* s = js.c_str() + p + k.size();
    while (*s == ' ') s++;
    if (!strncmp(s, "true", 4)) return 1;
    if (!strncmp(s, "false", 5)) return 0;
    return strtod(s, nullptr);
}

// SPEC.md stages 1-2 with cv2 5.0.0's luma: Y = (3735 B + 19235 G + 9798 R + 16384) >> 15, then INTER_AREA 1/2: the mean of the
// valid pixels of each 2x2 cell, fw = round-half-even(W / 2)
static std::vector<float> flowLuma(const std::vector<uint8_t>& rgba, uint32_t W, uint32_t H, int& fw, int& fh) {
    fw = (int)std::nearbyint(W / 2.0);
    fh = (int)std::nearbyint(H / 2.0);
    std::vector<float> y((size_t)fw * fh);
    for (int j = 0; j < fh; j++)
        for (int i = 0; i < fw; i++) {
            uint32_t s = 0, n = 0;
            for (int dy = 0; dy < 2; dy++)
                for (int dx = 0; dx < 2; dx++) {
                    uint32_t x = 2 * i + dx, yy = 2 * j + dy;
                    if (x >= W || yy >= H) continue;
                    const uint8_t* p = &rgba[((size_t)yy * W + x) * 4];
                    s += (p[2] * 3735u + p[1] * 19235u + p[0] * 9798u + 16384u) >> 15;
                    n++;
                }
            y[(size_t)j * fw + i] = (float)s / (float)n;   // exact: a multiple of 0.25 (or 0.5 / 1) below 256
        }
    return y;
}

struct Stat { double max = 0, mean = 0; size_t n = 0; };
static Stat diff(const float* a, const float* b, size_t n) {
    Stat s;
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        double d = std::fabs((double)a[i] - b[i]);
        if (!(d <= s.max)) s.max = d;   // (NaN -> NaN)
        sum += d;
    }
    s.mean = n ? sum / n : 0;
    s.n = n;
    return s;
}

// One Interpolator over one pair's two frames (and an output), Learned with the given net.
struct Rig {
    uint32_t W = 0, H = 0;
    Interpolator fg;
    Img a, c, out;
    HBuf stg, dbg, cnt, maps, luma;
    Interpolator::DebugMapLayout ml{};
    bool init(uint32_t w, uint32_t h, Settings s) {
        gpuInit();
        W = w; H = h;
        InitInfo info;
        info.phys = phys; info.device = dev; info.gdpa = vkGetDeviceProcAddr; info.width = W; info.height = H;
        info.viewFormat = VK_FORMAT_R8G8B8A8_UNORM;
        info.getMemProps = vkGetPhysicalDeviceMemoryProperties; info.getProps2 = vkGetPhysicalDeviceProperties2;
        info.settings = s;
        if (!fg.init(info)) { printf("  init failed: %s\n", fg.initError() ? fg.initError() : "?"); return false; }
        a = makeImage(W, H, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        c = makeImage(W, H, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        out = makeImage(W, H, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        begin();
        imgBarrier(out.img, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_SHADER_WRITE_BIT);
        submit();
        cnt.make(8);
        fg.debugMapLayout(ml);
        maps.make(ml.size);
        return true;
    }
    void frames(const std::vector<uint8_t>& pa, const std::vector<uint8_t>& pc, bool fresh) {
        upload(stg, pa, a, W, H, fresh);
        upload(stg, pc, c, W, H, fresh);
    }
    // one estimate (+ a synth at 0.5 into out if synth); reads back the field, the counters and (learned) the net's buffers
    std::vector<float> pf;
    uint32_t sadSum = 0;
    bool run(bool synth, bool learnedDbg) {
        begin();
        fg.beginFrame();
        if (!(synth ? fg.record(cb, a.view, c.view, out.view, 0.5f) : fg.recordEstimate(cb, a.view, c.view))) { vkEndCommandBuffer(cb); return false; }
        fg.recordDebugCounters(cb, cnt.b, 0);
        fg.recordDebugMaps(cb, maps.b, 0);
        Interpolator::DebugLearnedLayout L{};
        if (learnedDbg && fg.debugLearnedLayout(L)) {
            if (dbg.size < L.size) dbg.make(L.size);
            fg.recordDebugLearned(cb, dbg.b, 0);
        }
        hostBarrier();
        submit();
        sadSum = ((uint32_t*)cnt.map)[1];
        pf.assign((float*)(maps.map + ml.pixOff), (float*)(maps.map + ml.pixOff) + (size_t)ml.fw * ml.fh * 2);
        return true;
    }
    double sad() const { return sadSum / 256.0 / fg.debugSadBlocks(); }
    void setLumaSrc(const std::vector<float>& ya, const std::vector<float>& yc) {
        luma.make((ya.size() + yc.size()) * 4);
        memcpy(luma.map, ya.data(), ya.size() * 4);
        memcpy(luma.map + ya.size() * 4, yc.data(), yc.size() * 4);
        Interpolator::LearnedDebug d;
        d.lumaSrc = luma.b;
        fg.setDebugLearned(d);
    }
    void stopAt(int level, int step) {
        Interpolator::LearnedDebug d;
        d.lumaSrc = luma.b;
        d.stopLevel = level;
        d.stopStep = step;
        fg.setDebugLearned(d);
    }
    // the zero test's hooks: the reference's lumas (ref) or the library's own, the given field over the net's (or the net's own), and
    // the cost maps of the zero test (read back by zeroMaps())
    void zeroDbg(bool ref, const std::vector<float>* pfIn) {
        Interpolator::LearnedDebug d;
        if (ref) d.lumaSrc = luma.b;
        if (pfIn) {
            field.make(pfIn->size() * 4);
            memcpy(field.map, pfIn->data(), pfIn->size() * 4);
            d.fieldSrc = field.b;
        }
        d.zeroCosts = true;
        fg.setDebugLearned(d);
    }
    // after a run(.., true) with zeroCosts: C(h), C(0) and the mask per flow pixel
    void zeroMaps(std::vector<float>& ch, std::vector<float>& c0, std::vector<uint8_t>& zero) {
        Interpolator::DebugLearnedLayout L{};
        fg.debugLearnedLayout(L);
        const size_t n = (size_t)L.fw * L.fh;
        const float* p = (const float*)(dbg.map + L.actOff[0]);
        ch.resize(n); c0.resize(n); zero.resize(n);
        for (size_t i = 0; i < n; i++) { ch[i] = p[4 * i]; c0[i] = p[4 * i + 1]; zero[i] = p[4 * i + 2] != 0; }
    }
    HBuf field;
    ~Rig() {
        if (!dev) return;
        vkDeviceWaitIdle(dev);
        fg.destroy();
        for (Img* i : {&a, &c, &out}) if (i->img) destroyImg(*i);
        for (HBuf* b : {&stg, &dbg, &cnt, &maps, &luma, &field}) b->free();
    }
};

static void dumpPf(const std::string& net, const std::string& pair, const char* kind, const std::vector<float>& pf, double sad, bool cut) {
    const char* d = getenv("MLTEST_DUMP");
    if (!d || !*d) return;
    std::filesystem::create_directories(d);
    std::string base = std::string(d) + "/" + net + "_" + pair + "_" + kind;
    FILE* f = fopen((base + ".f32").c_str(), "wb");
    if (f) { fwrite(pf.data(), 4, pf.size(), f); fclose(f); }
    f = fopen((base + ".json").c_str(), "w");
    if (f) { fprintf(f, "{\"sad\": %.9g, \"cut\": %s}\n", sad, cut ? "true" : "false"); fclose(f); }
}

static void zeroChecks(const char* run, const char* pair, const std::string& dir, const std::string& js, const Settings& s, Rig& rig,
                       const std::vector<float>& expPf, int fw, int fh, bool colour);

// The LearnedNet of an export run (its directory name = framegen::learnedNetName). A new net: its TEST_CASEs below.
static LearnedNet netOf(const char* run) {
    LearnedNet n{};
    if (learnedNetFromName(run, &n)) return n;
    fprintf(stderr, "unknown net %s\n", run);
    exit(2);
}

// The whole SPEC.md section 8 procedure for one net and pair (see the top of the file), then the zero test's (zeroChecks).
static void checkPair(const char* run, const char* pair) {
    const LearnedNet net = netOf(run);
    const std::string dir = exportDir() + "/" + runDir(run) + "/testvec/" + pair + "/";
    INFO(run << " " << pair << " (" << dir << ")");
    std::vector<uint8_t> js8, pa, pc;
    uint32_t W = 0, H = 0, W2 = 0, H2 = 0;
    REQUIRE(readFile(dir + "vec.json", js8));
    const std::string js(js8.begin(), js8.end());
    REQUIRE(readPng(dir + "a.png", pa, W, H));
    REQUIRE(readPng(dir + "c.png", pc, W2, H2));
    REQUIRE((W == W2 && H == H2));
    const double expSad = jsonNum(js, "sad");
    const bool expCut = jsonNum(js, "cut") != 0;
    std::vector<float> expPf;
    REQUIRE(readF32(dir + "pf.f32", expPf));

    // 1. weights: the manifest checks (sha256, total_elements, shapes, offsets) run at build time (gpu/ml/embed_weights.py)
    // 2. the flow-res lumas (cv2 5.0.0) computed here == the vectors' ya / yc (not given for hd)
    int fw = 0, fh = 0;
    std::vector<float> ya = flowLuma(pa, W, H, fw, fh), yc = flowLuma(pc, W, H, fw, fh);
    std::vector<float> eya, eyc;
    const bool haveY = readF32(dir + "ya.f32", eya) && readF32(dir + "yc.f32", eyc);
    if (haveY) {
        CHECK(eya == ya);
        CHECK(eyc == yc);
    }
    REQUIRE(expPf.size() == (size_t)fw * fh * 2);

    Settings s;
    s.motionVersion = MotionVersion::Learned;
    s.learnedNet = net;
    s.learnedZeroTest = false;   // steps 1-6: the net's own field (the pf / sad vectors)
    Rig rig;
    REQUIRE(rig.init(W, H, s));
    rig.frames(pa, pc, true);

    // ---- fgr: the production path (the library's 14-bit luma)
    REQUIRE(rig.run(false, true));
    Interpolator::DebugLearnedLayout L{};
    REQUIRE(rig.fg.debugLearnedLayout(L));
    REQUIRE((L.fw == (uint32_t)fw && L.fh == (uint32_t)fh));
    const bool colour = L.ext != 0;   // the net variant (SPEC.md section 10)
    const double pfMaxTol = colour ? kPfMaxTolColour : kPfMaxTol;
    // colour nets: the flow-res chroma planes the library computed (level 0 of its chroma pyramids, x 255) against ca.f32 / cc.f32
    // (not given for hd): within CHROMA_TOL levels. Both runs below use them (the lumaSrc hook replaces the luma only).
    double chromaMax = 0;
    if (colour) {
        std::vector<float> eca, ecc;
        const bool have = readF32(dir + "ca.f32", eca) && readF32(dir + "cc.f32", ecc);
        if (have) {
            REQUIRE((eca.size() == (size_t)2 * fw * fh && ecc.size() == eca.size()));
            for (int p = 0; p < 2; p++)
                for (int which = 0; which < 2; which++) {
                    const std::vector<float>& e = which ? ecc : eca;
                    const float* g = (const float*)(rig.dbg.map + L.xOff[0]) + (size_t)(2 * p + which) * L.pw * L.ph;
                    for (int j = 0; j < fh; j++)
                        for (int i = 0; i < fw; i++)
                            chromaMax = std::max(chromaMax, std::fabs((double)g[(size_t)j * L.pw + i] * 255.0 - e[((size_t)p * fh + j) * fw + i]));
                }
            printf("  %s %-6s chroma planes: max |d| %.2e levels (tolerance %.0e)\n", run, pair, chromaMax, kChromaTol);
            CHECK(chromaMax <= kChromaTol);
        } else {
            printf("  %s %-6s chroma planes: no ca.f32 / cc.f32 (not checked)\n", run, pair);
        }
    }
    auto lumaOf = [&](int which) {   // the flow-res luma the net read, from its level-0 pyramid (luma / 255, correctly rounded)
        std::vector<float> y((size_t)fw * fh);
        const float* p = (const float*)(rig.dbg.map + L.pyrOff[0]) + (which ? (size_t)L.pw * L.ph : 0);
        for (int j = 0; j < fh; j++)
            for (int i = 0; i < fw; i++) y[(size_t)j * fw + i] = std::nearbyint(p[(size_t)j * L.pw + i] * 1020.0) / 4.0f;
        return y;
    };
    {
        size_t nd = 0, total = 0;
        bool quarter = true;
        for (int which = 0; which < 2; which++) {
            std::vector<float> g = lumaOf(which);
            const std::vector<float>& e = which ? yc : ya;
            for (size_t i = 0; i < g.size(); i++) {
                const float d = std::fabs(g[i] - e[i]);
                nd += d != 0;
                quarter = quarter && std::fmod(d, 0.25f) == 0;
            }
            total += g.size();
        }
        const double frac = (double)nd / total;
        printf("  %s %-6s fgr: luma differs from cv2's on %.3f %% of the flow pixels (multiples of 0.25: %s)\n", run, pair, 100 * frac, quarter ? "yes" : "NO");
        CHECK(quarter);
        CHECK(frac < 0.01);
    }
    const Stat fgrPf = diff(rig.pf.data(), expPf.data(), expPf.size());
    const double fgrSad = rig.sad();
    const bool fgrCut = fgrSad > s.sceneCutSad;
    printf("  %s %-6s fgr: pf max %.2e mean %.2e px, sad %.4f vs %.4f (rel %.1e), cut %d vs %d\n", run, pair, fgrPf.max, fgrPf.mean,
           fgrSad, expSad, std::fabs(fgrSad - expSad) / std::max(1.0, expSad), fgrCut, expCut);
    CHECK(fgrCut == expCut);
    dumpPf(run, pair, "fgr", rig.pf, fgrSad, fgrCut);

    // ---- ref: the reference's lumas (the exact-port check)
    rig.setLumaSrc(ya, yc);
    REQUIRE(rig.run(false, true));
    REQUIRE(rig.fg.debugLearnedLayout(L));
    CHECK(lumaOf(0) == ya);
    CHECK(lumaOf(1) == yc);
    {   // the pyramid: level 0 = the padded luma / 255 (correctly rounded), level l + 1 = the ordered 2x2 means, bit-exact
        bool exact = true;
        std::vector<float> lev[2];
        for (int which = 0; which < 2; which++) {
            const std::vector<float>& y = which ? yc : ya;
            std::vector<float>& p = lev[which];
            p.resize((size_t)L.pw * L.ph);
            for (uint32_t j = 0; j < L.ph; j++)
                for (uint32_t i = 0; i < L.pw; i++)
                    p[(size_t)j * L.pw + i] = y[(size_t)std::min<int>(j, fh - 1) * fw + std::min<int>(i, fw - 1)] / 255.0f;
        }
        for (int l = 0; l < 5; l++) {
            const uint32_t w = L.pw >> l, h = L.ph >> l;
            for (int which = 0; which < 2; which++) {
                const float* g = (const float*)(rig.dbg.map + L.pyrOff[l]) + (which ? (size_t)w * h : 0);
                size_t bad = 0, first = 0;
                for (size_t i = (size_t)w * h; i-- > 0;)
                    if (memcmp(&g[i], &lev[which][i], 4)) { bad++; first = i; }
                if (bad) printf("  %s %-6s ref: pyramid level %d %s: %zu values differ (first at %zu: %.9g vs %.9g)\n", run, pair, l,
                                which ? "c" : "a", bad, first, g[first], lev[which][first]);
                exact = exact && !bad;
                std::vector<float> nx((size_t)(w / 2) * (h / 2));
                for (uint32_t j = 0; j < h / 2; j++)
                    for (uint32_t i = 0; i < w / 2; i++) {
                        const float* p = &lev[which][(size_t)2 * j * w + 2 * i];
                        nx[(size_t)j * (w / 2) + i] = ((p[0] + p[1]) + (p[w] + p[w + 1])) * 0.25f;
                    }
                lev[which] = nx;
            }
        }
        CHECK(exact);
    }
    // h_level4..1 (padded level size, [h][w][2]) and pf
    double lvMax = 0, lvMean = 0;
    for (int l = 4; l >= 1; l--) {
        std::vector<float> e;
        REQUIRE(readF32(dir + "h_level" + std::to_string(l) + ".f32", e));
        const size_t n = (size_t)(L.pw >> l) * (L.ph >> l) * 2;
        REQUIRE(e.size() == n);
        const Stat st = diff((const float*)(rig.dbg.map + L.hOff[l]), e.data(), n);
        printf("  %s %-6s ref: h_level%d max %.2e mean %.2e px\n", run, pair, l, st.max, st.mean);
        lvMax = std::max(lvMax, st.max);
        lvMean = std::max(lvMean, st.mean);
        CHECK(st.max <= pfMaxTol);
        CHECK(st.mean <= kPfMeanTol);
    }
    const Stat refPf = diff(rig.pf.data(), expPf.data(), expPf.size());
    const double refSad = rig.sad(), sadRel = std::fabs(refSad - expSad) / std::max(1.0, expSad);
    const bool refCut = refSad > s.sceneCutSad;
    printf("  %s %-6s ref: pf max %.2e mean %.2e px, sad %.4f vs %.4f (rel %.1e), cut %d vs %d\n", run, pair, refPf.max, refPf.mean,
           refSad, expSad, sadRel, refCut, expCut);
    CHECK(refPf.max <= pfMaxTol);
    CHECK(refPf.mean <= kPfMeanTol);
    CHECK(sadRel <= kSadTol);
    CHECK(refCut == expCut);
    dumpPf(run, pair, "ref", rig.pf, refSad, refCut);

    // the coarsest level's intermediates (stopped estimates): in_top, act_top0..3 ([C][h][w]), raw_top = h_level4 / 2 (h starts at 0)
    const uint32_t tw = L.pw >> 4, th = L.ph >> 4, tn = tw * th;
    const int width = (int)L.width;
    double topMax = 0;
    for (int step = 0; step <= 5; step++) {
        rig.stopAt(4, step);
        REQUIRE(rig.run(false, true));
        std::string name = step == 0 ? "in_top" : step == 5 ? "raw_top" : "act_top" + std::to_string(step - 1);
        std::vector<float> e;
        REQUIRE(readF32(dir + name + ".f32", e));
        const int ch = step == 0 ? (colour ? 25 : 14) : step == 5 ? 2 : width;
        REQUIRE(e.size() == (size_t)ch * tn);
        std::vector<float> g((size_t)ch * tn);
        if (step == 5) {
            const float* h4 = (const float*)(rig.dbg.map + L.hOff[4]);
            for (uint32_t i = 0; i < tn; i++) { g[i] = h4[2 * i] / 2; g[tn + i] = h4[2 * i + 1] / 2; }
        } else {
            const int region = step == 0 || step % 2 == 0 ? 0 : 1;   // input, layers 1 and 3: R0; layers 0 and 2: R1
            const float* r = (const float*)(rig.dbg.map + L.actOff[region]);   // level 4: group k at 4 * k * tn floats
            for (int k = 0; k < ch; k++)
                for (uint32_t i = 0; i < tn; i++) g[(size_t)k * tn + i] = r[(size_t)(k / 4) * tn * 4 + i * 4 + k % 4];
        }
        const Stat st = diff(g.data(), e.data(), g.size());
        printf("  %s %-6s ref: %-8s max %.2e\n", run, pair, name.c_str(), st.max);
        topMax = std::max(topMax, st.max);
        CHECK(st.max <= kTopTol);
    }
    // ---- Settings::learnedHalf (fp16 activations): no pointwise tolerance (SPEC.md section 7), the same cut; synthcheck.py judges it
    for (int ref = 0; ref < 2; ref++) {
        Settings sh = s;
        sh.learnedHalf = true;
        REQUIRE(rig.fg.setSettings(sh));
        if (ref) rig.setLumaSrc(ya, yc);
        else rig.fg.setDebugLearned(Interpolator::LearnedDebug{});
        REQUIRE(rig.run(false, false));
        const Stat st = diff(rig.pf.data(), expPf.data(), expPf.size());
        const double sd = rig.sad();
        printf("  %s %-6s %s half: pf max %.2e mean %.2e px, sad %.4f vs %.4f (rel %.1e), cut %d vs %d\n", run, pair, ref ? "ref" : "fgr",
               st.max, st.mean, sd, expSad, std::fabs(sd - expSad) / std::max(1.0, expSad), sd > s.sceneCutSad, expCut);
        CHECK((sd > s.sceneCutSad) == expCut);
        dumpPf(run, pair, ref ? "refh" : "fgrh", rig.pf, sd, sd > s.sceneCutSad);
    }
    printf("RESULT %s %s ya/yc %s | top max %.1e | h_level max %.1e mean %.1e | ref pf max %.2e mean %.2e sad rel %.1e cut %s | "
           "fgr pf max %.2e mean %.2e sad %.4f (exp %.4f) cut %s\n", run, pair, haveY ? "exact" : "(none)", topMax, lvMax, lvMean,
           refPf.max, refPf.mean, sadRel, refCut == expCut ? "same" : "DIFFERENT", fgrPf.max, fgrPf.mean, fgrSad, expSad,
           fgrCut == expCut ? "same" : "DIFFERENT");
    zeroChecks(run, pair, dir, js, s, rig, expPf, fw, fh, colour);
}

// ---- Settings::learnedZeroTest (SPEC.md stage 7; section 8 step 7 and its tolerances) against the pair's *_zt vectors, five ways:
//   stagezt  the stage alone: the reference's lumas and the vectors' own pf.f32 as the field (setDebugLearned fieldSrc), so mask, costs,
//            pf_zt, sad_zt see exactly the torch run's input: mask agreement >= 99.9 %, cost_h_zt / cost0_zt abs <= 2e-3 levels (not
//            given for hd), pf_zt where the masks agree (max <= 5e-2, mean <= 2e-4 px), sad_zt relative <= 1e-3, the same cut_zt; and
//            the output field is exactly pf.f32 with the mask applied
//   refzt    the whole chain on the reference's lumas (the library's net, then its zero test): the same mask / pf / sad / cut gates
//   fgrzt    the production path (the library's 14-bit luma): the mask agreement, pf and sad are printed (the 14-bit luma moves the
//            costs; SPEC.md: judge it end to end, synthcheck.py), the cut must be identical
//   refhzt, fgrhzt   the same with Settings::learnedHalf (printed; the cut must be identical)
// Each run's field is dumped as <run>_<pair>_<kind> for synthcheck.py (which compares it with synth(pf_zt.f32, sad_zt)).
static const double kZeroAgree = 0.999, kCostTol = 2e-3;   // reference.py ZERO_AGREE, COST_TOL
static void zeroChecks(const char* run, const char* pair, const std::string& dir, const std::string& js, const Settings& s, Rig& rig,
                       const std::vector<float>& expPf, int fw, int fh, bool colour) {
    std::vector<float> expZt, eCh, eC0;
    std::vector<uint8_t> expMask;
    if (!readF32(dir + "pf_zt.f32", expZt) || !readFile(dir + "zero_zt.u8", expMask)) {
        printf("  %s %-6s: no *_zt vectors, zero test not checked\n", run, pair);
        return;
    }
    const size_t n = (size_t)fw * fh;
    REQUIRE(expZt.size() == 2 * n);
    REQUIRE(expMask.size() == n);
    const bool haveCost = readF32(dir + "cost_h_zt.f32", eCh) && readF32(dir + "cost0_zt.f32", eC0);
    const double expSad = jsonNum(js, "sad_zt");
    const bool expCut = jsonNum(js, "cut_zt") != 0;
    size_t expZeroed = 0;
    for (uint8_t v : expMask) expZeroed += v != 0;
    struct Z { const char* kind; bool ref, half, stage; } runs[] = {
        {"stagezt", true, false, true}, {"refzt", true, false, false}, {"fgrzt", false, false, false},
        {"refhzt", true, true, false}, {"fgrhzt", false, true, false}};
    for (const Z& z : runs) {
        Settings sz = s;
        sz.learnedZeroTest = true;
        sz.learnedHalf = z.half;
        REQUIRE(rig.fg.setSettings(sz));
        rig.zeroDbg(z.ref, z.stage ? &expPf : nullptr);
        REQUIRE(rig.run(false, true));
        std::vector<float> ch, c0;
        std::vector<uint8_t> mask;
        rig.zeroMaps(ch, c0, mask);
        size_t agree = 0, zeroed = 0, applied = 0;
        double pfMax = 0, pfSum = 0;
        for (size_t i = 0; i < n; i++) {
            const bool zm = mask[i] != 0;
            zeroed += zm;
            if (zm == (expMask[i] != 0)) {
                agree++;
                for (int k = 0; k < 2; k++) {
                    const double d = std::fabs((double)rig.pf[2 * i + k] - expZt[2 * i + k]);
                    if (!(d <= pfMax)) pfMax = d;
                    pfSum += d;
                }
            }
            if (zm) applied += rig.pf[2 * i] == 0 && rig.pf[2 * i + 1] == 0;
            else if (z.stage) applied += !memcmp(&rig.pf[2 * i], &expPf[2 * i], 8);   // the given field where the test kept it
            else applied++;
        }
        const double agreeFrac = (double)agree / n, pfMean = agree ? pfSum / (2.0 * agree) : 0;
        const double sd = rig.sad(), rel = std::fabs(sd - expSad) / std::max(1.0, expSad);
        const bool cut = sd > s.sceneCutSad;
        Stat chS, c0S;
        if (haveCost && z.stage) { chS = diff(ch.data(), eCh.data(), n); c0S = diff(c0.data(), eC0.data(), n); }
        printf("ZT %s %-7s %-7s: mask agree %.4f %% (%zu px flipped), zeroed %.2f %% (exp %.2f %%), pf on agreeing px max %.2e mean %.2e, "
               "sad %.4f vs %.4f (rel %.1e), cut %d vs %d", run, pair, z.kind, 100 * agreeFrac, n - agree, 100.0 * zeroed / n,
               100.0 * expZeroed / n, pfMax, pfMean, sd, expSad, rel, cut, expCut);
        if (haveCost && z.stage) printf(", cost_h max %.1e, cost0 max %.1e", chS.max, c0S.max);
        printf("\n");
        CHECK(applied == n);   // the output field is the mask applied to the field the test read
        CHECK(cut == expCut);
        if (z.ref && !z.half) {   // the exact float32 port's gates
            CHECK(agreeFrac >= kZeroAgree);
            CHECK(pfMax <= (colour ? kPfMaxTolColour : kPfMaxTol));
            CHECK(pfMean <= kPfMeanTol);
            CHECK(rel <= kSadTol);
            if (haveCost && z.stage) { CHECK(chS.max <= kCostTol); CHECK(c0S.max <= kCostTol); }
        }
        dumpPf(run, pair, z.kind, rig.pf, sd, cut);
    }
    rig.fg.setDebugLearned(Interpolator::LearnedDebug{});
}

TEST_CASE("e2v3 static", "[vectors][e2v3]") { checkPair("e2v3", "static"); }
TEST_CASE("e2v3 motion", "[vectors][e2v3]") { checkPair("e2v3", "motion"); }
TEST_CASE("e2v3 far", "[vectors][e2v3]") { checkPair("e2v3", "far"); }
TEST_CASE("e2v3 cut", "[vectors][e2v3]") { checkPair("e2v3", "cut"); }
TEST_CASE("e2v3 odd", "[vectors][e2v3]") { checkPair("e2v3", "odd"); }
TEST_CASE("e2v3 hd", "[vectors][e2v3]") { checkPair("e2v3", "hd"); }
TEST_CASE("e2v3 overlay", "[vectors][e2v3]") { checkPair("e2v3", "overlay"); }
TEST_CASE("e1v3 motion", "[vectors][e1v3]") { checkPair("e1v3", "motion"); }
TEST_CASE("e1v3 cut", "[vectors][e1v3]") { checkPair("e1v3", "cut"); }
TEST_CASE("e1v3 odd", "[vectors][e1v3]") { checkPair("e1v3", "odd"); }
TEST_CASE("e1v3 overlay", "[vectors][e1v3]") { checkPair("e1v3", "overlay"); }
// e2v4: its "cut" pair is NOT a cut (sad 28.3) and "cut2" is (sad 42.8): every check reads cut / cut_zt from vec.json
TEST_CASE("e2v4 static", "[vectors][e2v4]") { checkPair("e2v4", "static"); }
TEST_CASE("e2v4 motion", "[vectors][e2v4]") { checkPair("e2v4", "motion"); }
TEST_CASE("e2v4 far", "[vectors][e2v4]") { checkPair("e2v4", "far"); }
TEST_CASE("e2v4 cut", "[vectors][e2v4]") { checkPair("e2v4", "cut"); }
TEST_CASE("e2v4 cut2", "[vectors][e2v4]") { checkPair("e2v4", "cut2"); }
TEST_CASE("e2v4 odd", "[vectors][e2v4]") { checkPair("e2v4", "odd"); }
TEST_CASE("e2v4 hd", "[vectors][e2v4]") { checkPair("e2v4", "hd"); }
TEST_CASE("e2v4 overlay", "[vectors][e2v4]") { checkPair("e2v4", "overlay"); }
// e1v4: both "cut" (sad 43.2) and "cut2" (sad 56.1) are cuts (vec.json)
TEST_CASE("e1v4 motion", "[vectors][e1v4]") { checkPair("e1v4", "motion"); }
TEST_CASE("e1v4 cut", "[vectors][e1v4]") { checkPair("e1v4", "cut"); }
TEST_CASE("e1v4 cut2", "[vectors][e1v4]") { checkPair("e1v4", "cut2"); }
TEST_CASE("e1v4 odd", "[vectors][e1v4]") { checkPair("e1v4", "odd"); }
TEST_CASE("e1v4 overlay", "[vectors][e1v4]") { checkPair("e1v4", "overlay"); }
// e2v5: "cut" is a cut both ways but only just with the zero test (sad 32.2 net-only, 30.17 with it, threshold 30); "cut2" (sad 45.5 /
// 43.0) is the safe cut (vec.json)
TEST_CASE("e2v5 static", "[vectors][e2v5]") { checkPair("e2v5", "static"); }
TEST_CASE("e2v5 motion", "[vectors][e2v5]") { checkPair("e2v5", "motion"); }
TEST_CASE("e2v5 far", "[vectors][e2v5]") { checkPair("e2v5", "far"); }
TEST_CASE("e2v5 cut", "[vectors][e2v5]") { checkPair("e2v5", "cut"); }
TEST_CASE("e2v5 cut2", "[vectors][e2v5]") { checkPair("e2v5", "cut2"); }
TEST_CASE("e2v5 odd", "[vectors][e2v5]") { checkPair("e2v5", "odd"); }
TEST_CASE("e2v5 hd", "[vectors][e2v5]") { checkPair("e2v5", "hd"); }
TEST_CASE("e2v5 overlay", "[vectors][e2v5]") { checkPair("e2v5", "overlay"); }
// e1v5: "cut" is a cut net-only (sad 31.3) but NOT with the zero test (29.1); "cut2" (53.1 / 50.8) is a cut both ways (vec.json)
TEST_CASE("e1v5 motion", "[vectors][e1v5]") { checkPair("e1v5", "motion"); }
TEST_CASE("e1v5 cut", "[vectors][e1v5]") { checkPair("e1v5", "cut"); }
TEST_CASE("e1v5 cut2", "[vectors][e1v5]") { checkPair("e1v5", "cut2"); }
TEST_CASE("e1v5 odd", "[vectors][e1v5]") { checkPair("e1v5", "odd"); }
TEST_CASE("e1v5 overlay", "[vectors][e1v5]") { checkPair("e1v5", "overlay"); }

// c2smoke: the stand-in of the colour variant. Its cut decisions come from vec.json (cut = the net alone, cut_zt = with the zero test).
// "colour" is the chroma-only pair: a port that ignores chroma fails its pf check by several px (SPEC.md section 10.6); with the zero test
// on it is 100 % zeroed (the luma-only zero test, section 10.4)
TEST_CASE("c2smoke static", "[vectors][c2smoke]") { checkPair("c2smoke", "static"); }
TEST_CASE("c2smoke motion", "[vectors][c2smoke]") { checkPair("c2smoke", "motion"); }
TEST_CASE("c2smoke far", "[vectors][c2smoke]") { checkPair("c2smoke", "far"); }
TEST_CASE("c2smoke cut", "[vectors][c2smoke]") { checkPair("c2smoke", "cut"); }
TEST_CASE("c2smoke cut2", "[vectors][c2smoke]") { checkPair("c2smoke", "cut2"); }
TEST_CASE("c2smoke odd", "[vectors][c2smoke]") { checkPair("c2smoke", "odd"); }
TEST_CASE("c2smoke hd", "[vectors][c2smoke]") { checkPair("c2smoke", "hd"); }
TEST_CASE("c2smoke overlay", "[vectors][c2smoke]") { checkPair("c2smoke", "overlay"); }
TEST_CASE("c2smoke colour", "[vectors][c2smoke]") { checkPair("c2smoke", "colour"); }

// ---- duplicates (shaders/ml_dup.comp: FGR's DUP_DIFF test)
// A frame of the motion pair (or a smooth synthetic one if the vectors are missing)
static std::vector<uint8_t> someFrame(uint32_t& W, uint32_t& H) {
    std::vector<uint8_t> f;
    if (readPng(exportDir() + "/e2v3/testvec/motion/a.png", f, W, H)) return f;
    W = 640; H = 360;
    f.resize((size_t)W * H * 4);
    for (uint32_t y = 0; y < H; y++)
        for (uint32_t x = 0; x < W; x++) {
            uint8_t* p = &f[((size_t)y * W + x) * 4];
            p[0] = (uint8_t)(128 + 100 * std::sin(x * 0.05) * std::cos(y * 0.07));
            p[1] = (uint8_t)((x * 3 + y * 5) & 255);
            p[2] = (uint8_t)(128 + 90 * std::sin((x + y) * 0.031));
            p[3] = 255;
        }
    return f;
}

static void duplicateCase(LearnedNet net) {
    uint32_t W, H;
    const std::vector<uint8_t> a = someFrame(W, H);
    for (int algo = 0; algo < 3; algo++) {   // Learned (zero test on, the default; off), and V1 as the control (its exact answer for a still pair)
        Settings s;
        s.motionVersion = algo == 2 ? MotionVersion::V1 : MotionVersion::Learned;
        s.learnedNet = net;
        s.learnedZeroTest = algo == 0;
        Rig rig;
        REQUIRE(rig.init(W, H, s));
        rig.frames(a, a, true);
        REQUIRE(rig.run(true, false));
        const std::vector<uint8_t> out = download(rig.stg, rig.out, W, H);
        size_t changed = 0;
        for (size_t i = 0; i < out.size(); i += 4)
            changed += out[i] != a[i] || out[i + 1] != a[i + 1] || out[i + 2] != a[i + 2];
        float pfMax = 0;
        for (float v : rig.pf) pfMax = std::max(pfMax, std::fabs(v));
        printf("  %s identical frames %ux%u: output pixels != input %zu, max |pf| %g, SAD sum %u\n", algo == 2 ? "V1        " : algo ? "Learned   " : "Learned zt",
               W, H, changed, pfMax, rig.sadSum);
        CHECK(changed == 0);
        CHECK(pfMax == 0);
        CHECK(rig.sadSum == 0);
    }
    // near-duplicates around FGR's threshold (mean over pixels of the channel-mean |a - c|, DUP_DIFF = 0.6 levels): every channel of
    // every k-th pixel +-1 level (a mean of 1/k), k = 2 (0.5: a duplicate) and k = 1 (1.0: not)
    // (zero test on, the default, and off: with it the net's field on this flat +-1 change is zeroed almost everywhere, so only the
    // SAD sum shows that the net ran; without it the field is the net's own)
    for (int zt = 0; zt < 2; zt++)
        for (int k : {2, 1}) {
            std::vector<uint8_t> c = a;
            for (size_t i = 0, px = 0; i < c.size(); i += 4, px++)
                if (px % k == 0)
                    for (int ch = 0; ch < 3; ch++) c[i + ch] = c[i + ch] < 128 ? c[i + ch] + 1 : c[i + ch] - 1;
            Settings s;
            s.motionVersion = MotionVersion::Learned;
            s.learnedNet = net;
            s.learnedZeroTest = zt != 0;
            Rig rig;
            REQUIRE(rig.init(W, H, s));
            rig.frames(a, c, true);
            REQUIRE(rig.run(true, false));
            float pfMax = 0;
            for (float v : rig.pf) pfMax = std::max(pfMax, std::fabs(v));
            printf("  Learned%s mean |a - c| %.2f: max |pf| %g, SAD sum %u (%s)\n", zt ? " zt" : "   ", 1.0 / k, pfMax, rig.sadSum,
                   k == 2 ? "duplicate: zero" : "the net ran");
            if (k == 2) { CHECK(pfMax == 0); CHECK(rig.sadSum == 0); }
            else { CHECK((zt || pfMax > 0)); CHECK(rig.sadSum > 0); }
        }
}
TEST_CASE("e2v3 duplicate frames", "[dup][e2v3]") { duplicateCase(LearnedNet::E2v3); }
TEST_CASE("e1v3 duplicate frames", "[dup][e1v3]") { duplicateCase(LearnedNet::E1v3); }
TEST_CASE("e2v4 duplicate frames", "[dup][e2v4]") { duplicateCase(LearnedNet::E2v4); }
TEST_CASE("e1v4 duplicate frames", "[dup][e1v4]") { duplicateCase(LearnedNet::E1v4); }
TEST_CASE("e2v5 duplicate frames", "[dup][e2v5]") { duplicateCase(LearnedNet::E2v5); }
TEST_CASE("e1v5 duplicate frames", "[dup][e1v5]") { duplicateCase(LearnedNet::E1v5); }
TEST_CASE("c2smoke duplicate frames", "[dup][c2smoke]") { duplicateCase(LearnedNet::C2Smoke); }

// ---- the settings API
TEST_CASE("Learned settings", "[api]") {
    Settings s;
    s.motionVersion = MotionVersion::Learned;
    CHECK(Interpolator::validate(s));
    CHECK(Interpolator::settingsError(s, Format::Rgba8) == nullptr);
    Settings f4 = s;
    f4.flowScale = 4;
    CHECK(Interpolator::validate(f4));   // in range, but not for Learned:
    CHECK(Interpolator::settingsError(f4, Format::Rgba8) != nullptr);
    CHECK(s.learnedNet == LearnedNet::E2v5);   // the default net
    CHECK(s.learnedZeroTest);                  // the zero test: on by default
    CHECK_FALSE(s.learnedStaticGuard);         // the static guard: off by default
    Settings bad = s;
    bad.learnedNet = (LearnedNet)kLearnedNets;   // the first value past the last net
    CHECK_FALSE(Interpolator::validate(bad));
    CHECK(learnedNetName((LearnedNet)kLearnedNets) == nullptr);
    CHECK(!strcmp(learnedNetName(LearnedNet::E2v4), "e2v4"));
    CHECK(!strcmp(learnedNetName(LearnedNet::E2v5), "e2v5"));
    CHECK(!strcmp(learnedNetName(LearnedNet::E1v5), "e1v5"));
    CHECK(!strcmp(learnedNetName(LearnedNet::C2Smoke), "c2smoke"));
    CHECK((uint32_t)LearnedNet::E1v5 == 5);    // the values are never renumbered
    CHECK((uint32_t)LearnedNet::C2Smoke == 6);
    LearnedNet byName = LearnedNet::E2v3;
    CHECK(learnedNetFromName("e1v3", &byName));
    CHECK(byName == LearnedNet::E1v3);
    CHECK_FALSE(learnedNetFromName("e9v9", &byName));
    CHECK(byName == LearnedNet::E1v3);
    CHECK_FALSE(Interpolator::validate(bad));
    bad = s;
    bad.motionVersion = (MotionVersion)3;
    CHECK_FALSE(Interpolator::validate(bad));

    // runtime: V1 -> Learned (drops the estimate) -> a net switch (keeps it) -> V1 (drops it); a size beyond the device refused
    uint32_t W, H;
    const std::vector<uint8_t> a = someFrame(W, H);
    std::vector<uint8_t> c = a;
    for (size_t i = 0; i < c.size(); i++) if (i % 4 != 3) c[i] = (uint8_t)std::min(255, c[i] + (int)((i / 4) % 7));
    Rig rig;
    REQUIRE(rig.init(W, H, Settings{}));
    rig.frames(a, c, true);
    REQUIRE(rig.run(true, false));
    s.learnedNet = LearnedNet::E1v3;
    REQUIRE(rig.fg.setSettings(s));
    begin();
    rig.fg.beginFrame();
    CHECK_FALSE(rig.fg.recordSynth(cb, rig.a.view, rig.c.view, rig.out.view, 0.5f));   // the version switch dropped the estimate
    submit();
    REQUIRE(rig.run(true, false));
    const std::vector<float> pf1 = rig.pf;
    s.learnedNet = LearnedNet::E2v3;   // wider: the activations grow
    REQUIRE(rig.fg.setSettings(s));
    begin();
    rig.fg.beginFrame();
    CHECK(rig.fg.recordSynth(cb, rig.a.view, rig.c.view, rig.out.view, 0.5f));   // a net switch keeps it
    submit();
    REQUIRE(rig.run(true, false));
    CHECK(rig.pf != pf1);   // another net, another field
    s.learnedNet = LearnedNet::E1v3;
    REQUIRE(rig.fg.setSettings(s));
    REQUIRE(rig.run(true, false));
    CHECK(rig.pf == pf1);   // deterministic, and the same after the buffer grew
    s.learnedNet = LearnedNet::E2v3;
    REQUIRE(rig.fg.setSettings(s));
    REQUIRE(rig.run(true, false));
    const std::vector<float> pf3 = rig.pf;
    s.learnedNet = LearnedNet::E2v4;   // the same width: the third net's weights (after the other two in the weights buffer)
    REQUIRE(rig.fg.setSettings(s));
    REQUIRE(rig.run(true, false));
    CHECK(rig.pf != pf3);
    const std::vector<float> pf4 = rig.pf;
    s.learnedNet = LearnedNet::E2v5;   // the same width again: the fifth net's weights
    REQUIRE(rig.fg.setSettings(s));
    REQUIRE(rig.run(true, false));
    CHECK(rig.pf != pf4);
    CHECK(rig.pf != pf3);
    const std::vector<float> pf5 = rig.pf;
    s.learnedNet = LearnedNet::C2Smoke;   // the colour variant: the state grows (its extra planes), another input layout
    REQUIRE(rig.fg.setSettings(s));
    REQUIRE(rig.run(true, false));
    CHECK(rig.pf != pf5);
    s.learnedNet = LearnedNet::E2v5;      // and back: the same field as before (the luma variant ignores the grown state)
    REQUIRE(rig.fg.setSettings(s));
    REQUIRE(rig.run(true, false));
    CHECK(rig.pf == pf5);
    REQUIRE(rig.fg.setSettings(Settings{}));
    rig.fg.releaseUnusedMotion();
    REQUIRE(rig.run(true, false));
    Settings f4r = s;
    f4r.flowScale = 4;
    CHECK_FALSE(rig.fg.reconfigure(f4r));
}
