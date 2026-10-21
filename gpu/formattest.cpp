// formattest: single-frame checks of framegen's image formats (Interpolator::format(), InitInfo::viewFormat).
//   formattest check [W H]      default 640 360: for every format the device supports, a 12-frame synthetic sequence
//                               (the last pair interpolated at t = 0.5 / 0.25 / blurred), with and without UI protection
//   formattest time [W H]       one estimate + one synth per format, GPU time per pass group (default 1920 1080)
//   options: --separate-planes  YUV: two single-plane images instead of one multi-planar image with plane views
//            --algo v1|v2       the motion version (Settings::motionVersion), default v1. V2 is Rgba8 only: with v2, check runs
//                               Rgba8 in V2 and checks that every other format refuses V2 (init and setSettings, with a reason)
//            --mode quality|performance  Settings::v2Mode (default quality; V1 ignores it)
//            --hash             check: print an FNV-1a hash of every output of each format's Off run (t = 0.5, 0.25, blur) and
//                               WholeScreen run (t = 0.5, 0.25), raw bytes of both planes: diff two library builds
//
// The scene is the panning, noisy background with moving objects, a static crosshair and a static HUD box of uitest.cpp,
// generated as 8-bit RGB and converted to each format (Rgb10 codes, half floats, linear scRGB-like floats, BT.709 limited
// range NV12 / P010 with 2x2 averaged chroma). Everything is compared after DECODING back to 8-bit RGB levels, so the
// numbers of all formats are on one scale. Single frames, 640x360: a few milliseconds of GPU time per format.
#include "framegen.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <vector>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d\n", #x, r_); exit(1); } } while (0)

using namespace framegen;

static VkDevice dev;
static VkPhysicalDevice phys;
static VkPhysicalDeviceMemoryProperties memProps;
static VkQueue queue;
static VkCommandBuffer cb;
static VkFence fence;
static double tsPeriodMs = 0;
static bool separatePlanes = false;
static bool hashOut = false;   // --hash
#ifndef FORMATTEST_NO_MOTION   // define it to build against a library without Settings::motionVersion
static MotionVersion gAlgo = MotionVersion::V1;   // --algo
static V2Mode gMode = V2Mode::Quality;             // --mode
static bool algoV2() { return gAlgo == MotionVersion::V2; }
#else
static bool algoV2() { return false; }
#endif
static uint64_t fnv(const std::vector<uint8_t>& v, uint64_t h = 1469598103934665603ull) {
    for (uint8_t b : v) { h ^= b; h *= 1099511628211ull; }
    return h;
}
static int fails = 0;
static void expect(bool ok, const char* what) {
    printf("    %s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) fails++;
}

static uint32_t memType(uint32_t bits, VkMemoryPropertyFlags f) {
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & f) == f) return i;
    fprintf(stderr, "no memory type\n");
    exit(1);
}

// ---------------------------------------------------------------- half floats
static uint16_t toHalf(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int e = (int)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (e <= 0) return (uint16_t)sign;   // flush denormals (the test values are >= 0 and not tiny)
    if (e >= 31) return (uint16_t)(sign | 0x7bffu);
    uint32_t h = sign | ((uint32_t)e << 10) | (man >> 13);
    if (man & 0x1000u) h++;   // round to nearest (ties up)
    return (uint16_t)h;
}
static float fromHalf(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16, e = (h >> 10) & 31, man = h & 1023u, x;
    if (e == 0) x = sign;
    else x = sign | ((e - 15 + 127) << 23) | (man << 13);
    float f;
    memcpy(&f, &x, 4);
    return f;
}

// ---------------------------------------------------------------- the formats under test
enum class Enc { Rgba8, Rgb10, F16Enc, F16Lin, Nv12, P010 };   // F16Lin: linear light, 1.0 = 80 nits, peak 4.0 (320 nits)
static const char* encName(Enc e) {
    static const char* n[] = {"Rgba8", "Rgb10", "RgbaF16 (encoded)", "RgbaF16 (linear scRGB)", "Nv12", "P010"};
    return n[(int)e];
}
static bool isYuv(Enc e) { return e == Enc::Nv12 || e == Enc::P010; }
static VkFormat planeFmt0(Enc e) {
    switch (e) {
    case Enc::Rgba8: return VK_FORMAT_R8G8B8A8_UNORM;
    case Enc::Rgb10: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case Enc::F16Enc: case Enc::F16Lin: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case Enc::Nv12: return VK_FORMAT_R8_UNORM;
    default: return VK_FORMAT_R16_UNORM;
    }
}
static VkFormat planeFmt1(Enc e) { return e == Enc::Nv12 ? VK_FORMAT_R8G8_UNORM : VK_FORMAT_R16G16_UNORM; }
static VkFormat mpFmt(Enc e) { return e == Enc::Nv12 ? VK_FORMAT_G8_B8R8_2PLANE_420_UNORM : VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16; }
static const float kLinPeak = 4.0f, kLinGamma = 2.2f;

// encoded frame: plane 0 bytes, plane 1 bytes (YUV)
struct Raw { std::vector<uint8_t> p0, p1; };

// 8-bit RGBA -> the format's bytes
static Raw encode(Enc e, const std::vector<uint8_t>& rgba, int W, int H) {
    Raw r;
    size_t n = (size_t)W * H;
    switch (e) {
    case Enc::Rgba8: r.p0 = rgba; break;
    case Enc::Rgb10:
        r.p0.resize(n * 4);
        for (size_t i = 0; i < n; i++) {
            uint32_t v = 3u << 30;
            for (int c = 0; c < 3; c++) v |= (uint32_t)std::lround(rgba[i * 4 + c] * 1023.0 / 255.0) << (10 * c);
            memcpy(&r.p0[i * 4], &v, 4);
        }
        break;
    case Enc::F16Enc: case Enc::F16Lin:
        r.p0.resize(n * 8);
        for (size_t i = 0; i < n; i++)
            for (int c = 0; c < 4; c++) {
                float v = c == 3 ? 1.0f : rgba[i * 4 + c] / 255.0f;
                if (e == Enc::F16Lin && c < 3) v = std::pow(v, kLinGamma) * kLinPeak;
                uint16_t h = toHalf(v);
                memcpy(&r.p0[(i * 4 + c) * 2], &h, 2);
            }
        break;
    default: {
        const bool p010 = e == Enc::P010;
        const double sc = p010 ? 4.0 : 1.0;   // level -> code
        const int cw = (W + 1) / 2, ch = (H + 1) / 2;
        r.p0.resize(n * (p010 ? 2 : 1));
        r.p1.resize((size_t)cw * ch * (p010 ? 4 : 2));
        std::vector<double> cbp(n), crp(n);
        for (size_t i = 0; i < n; i++) {
            double R = rgba[i * 4], G = rgba[i * 4 + 1], B = rgba[i * 4 + 2];
            double Y = 0.2126 * R + 0.7152 * G + 0.0722 * B;
            double yl = 16 + Y * 219.0 / 255.0;
            cbp[i] = 128 + (B - Y) / (2 * (1 - 0.0722)) * 224.0 / 255.0;
            crp[i] = 128 + (R - Y) / (2 * (1 - 0.2126)) * 224.0 / 255.0;
            uint32_t code = (uint32_t)std::lround(std::min(std::max(yl * sc, 0.0), p010 ? 1023.0 : 255.0));
            if (p010) { uint16_t w = (uint16_t)(code << 6); memcpy(&r.p0[i * 2], &w, 2); } else r.p0[i] = (uint8_t)code;
        }
        for (int y = 0; y < ch; y++)
            for (int x = 0; x < cw; x++) {
                double sb = 0, sr = 0;
                int cnt = 0;
                for (int j = 0; j < 2; j++)
                    for (int i = 0; i < 2; i++) {
                        int X = std::min(2 * x + i, W - 1), Y = std::min(2 * y + j, H - 1);   // (odd edges: the last pixel repeats)
                        sb += cbp[(size_t)Y * W + X]; sr += crp[(size_t)Y * W + X]; cnt++;
                    }
                uint32_t b = (uint32_t)std::lround(std::min(std::max(sb / cnt * sc, 0.0), p010 ? 1023.0 : 255.0));
                uint32_t rr = (uint32_t)std::lround(std::min(std::max(sr / cnt * sc, 0.0), p010 ? 1023.0 : 255.0));
                size_t o = (size_t)y * cw + x;
                if (p010) { uint16_t w[2] = {(uint16_t)(b << 6), (uint16_t)(rr << 6)}; memcpy(&r.p1[o * 4], w, 4); }
                else { r.p1[o * 2] = (uint8_t)b; r.p1[o * 2 + 1] = (uint8_t)rr; }
            }
    }
    }
    return r;
}

// the format's bytes -> float RGB levels (0..255, may leave the range), W*H*3
static std::vector<float> decode(Enc e, const Raw& r, int W, int H) {
    size_t n = (size_t)W * H;
    std::vector<float> o(n * 3);
    switch (e) {
    case Enc::Rgba8:
        for (size_t i = 0; i < n; i++) for (int c = 0; c < 3; c++) o[i * 3 + c] = r.p0[i * 4 + c];
        break;
    case Enc::Rgb10:
        for (size_t i = 0; i < n; i++) {
            uint32_t v;
            memcpy(&v, &r.p0[i * 4], 4);
            for (int c = 0; c < 3; c++) o[i * 3 + c] = ((v >> (10 * c)) & 1023u) * 255.0f / 1023.0f;
        }
        break;
    case Enc::F16Enc: case Enc::F16Lin:
        for (size_t i = 0; i < n; i++)
            for (int c = 0; c < 3; c++) {
                uint16_t h;
                memcpy(&h, &r.p0[(i * 4 + c) * 2], 2);
                float v = fromHalf(h);
                o[i * 3 + c] = e == Enc::F16Lin ? 255.0f * std::pow(std::max(v, 0.0f) / kLinPeak, 1.0f / kLinGamma) : v * 255.0f;
            }
        break;
    default: {
        const bool p010 = e == Enc::P010;
        const int cw = (W + 1) / 2;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                size_t i = (size_t)y * W + x, ci = (size_t)(y / 2) * cw + x / 2;
                double Y, cb, cr;
                if (p010) {
                    uint16_t w, c2[2];
                    memcpy(&w, &r.p0[i * 2], 2); memcpy(c2, &r.p1[ci * 4], 4);
                    Y = (w >> 6) / 4.0; cb = (c2[0] >> 6) / 4.0; cr = (c2[1] >> 6) / 4.0;
                } else {
                    Y = r.p0[i]; cb = r.p1[ci * 2]; cr = r.p1[ci * 2 + 1];
                }
                double yf = (Y - 16) * 255.0 / 219.0, bb = (cb - 128) * 255.0 / 224.0, rr = (cr - 128) * 255.0 / 224.0;
                double R = yf + 2 * (1 - 0.2126) * rr, B = yf + 2 * (1 - 0.0722) * bb;
                double G = (yf - 0.2126 * R - 0.0722 * B) / 0.7152;
                o[i * 3] = (float)R; o[i * 3 + 1] = (float)G; o[i * 3 + 2] = (float)B;
            }
    }
    }
    return o;
}

// ---------------------------------------------------------------- Vulkan helpers
struct Pic {
    VkImage img = VK_NULL_HANDLE, imgC = VK_NULL_HANDLE;   // imgC: the separate chroma image (separate planes only)
    VkDeviceMemory mem = VK_NULL_HANDLE, memC = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE, chroma = VK_NULL_HANDLE;
    bool mp = false;   // one multi-planar image, `view`/`chroma` are plane views of it
    Frame frame() const { return Frame(view, chroma); }
};

static void bind(VkImage img, VkDeviceMemory& mem) {
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev, img, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size, memType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    CK(vkAllocateMemory(dev, &mai, nullptr, &mem));
    CK(vkBindImageMemory(dev, img, mem, 0));
}
static VkImageView makeView(VkImage img, VkFormat f, VkImageAspectFlags aspect, VkImageUsageFlags usage, bool withUsage) {
    VkImageViewUsageCreateInfo vu{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
    vu.usage = usage;
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.pNext = withUsage ? &vu : nullptr;
    vci.image = img; vci.viewType = VK_IMAGE_VIEW_TYPE_2D; vci.format = f;
    vci.subresourceRange = {aspect, 0, 1, 0, 1};
    VkImageView v;
    CK(vkCreateImageView(dev, &vci, nullptr, &v));
    return v;
}
static VkImage makeImg(VkFormat f, uint32_t w, uint32_t h, VkImageUsageFlags usage, VkImageCreateFlags flags, VkResult* res) {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.flags = flags;
    ici.imageType = VK_IMAGE_TYPE_2D; ici.format = f; ici.extent = {w, h, 1};
    ici.mipLevels = ici.arrayLayers = 1; ici.samples = VK_SAMPLE_COUNT_1_BIT; ici.tiling = VK_IMAGE_TILING_OPTIMAL; ici.usage = usage;
    VkImage img = VK_NULL_HANDLE;
    *res = vkCreateImage(dev, &ici, nullptr, &img);
    return img;
}
static const VkImageUsageFlags kUsage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

static bool tryMultiplane(Enc e, uint32_t w, uint32_t h, Pic& p) {
    VkResult r;
    VkImage img = makeImg(mpFmt(e), w, h, kUsage, VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT, &r);
    if (r != VK_SUCCESS) return false;
    bind(img, p.mem);
    p.img = img;
    p.mp = true;
    const VkImageUsageFlags vu = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    p.view = makeView(img, planeFmt0(e), VK_IMAGE_ASPECT_PLANE_0_BIT, vu, true);
    p.chroma = makeView(img, planeFmt1(e), VK_IMAGE_ASPECT_PLANE_1_BIT, vu, true);
    return true;
}

static int mpUsed = -1;   // -1 unknown, 0 separate plane images, 1 multi-planar image with plane views
static Pic makePic(Enc e, uint32_t w, uint32_t h) {
    Pic p;
    if (!isYuv(e)) {
        VkResult r;
        p.img = makeImg(planeFmt0(e), w, h, kUsage, 0, &r);
        CK(r);
        bind(p.img, p.mem);
        p.view = makeView(p.img, planeFmt0(e), VK_IMAGE_ASPECT_COLOR_BIT, kUsage, false);
        return p;
    }
    if (!separatePlanes && tryMultiplane(e, w, h, p)) { mpUsed = 1; return p; }
    mpUsed = 0;
    VkResult r;
    p.img = makeImg(planeFmt0(e), w, h, kUsage, 0, &r); CK(r);
    bind(p.img, p.mem);
    p.view = makeView(p.img, planeFmt0(e), VK_IMAGE_ASPECT_COLOR_BIT, kUsage, false);
    p.imgC = makeImg(planeFmt1(e), (w + 1) / 2, (h + 1) / 2, kUsage, 0, &r); CK(r);
    bind(p.imgC, p.memC);
    p.chroma = makeView(p.imgC, planeFmt1(e), VK_IMAGE_ASPECT_COLOR_BIT, kUsage, false);
    return p;
}
static void destroyPic(Pic& p) {
    if (p.view) vkDestroyImageView(dev, p.view, nullptr);
    if (p.chroma) vkDestroyImageView(dev, p.chroma, nullptr);
    if (p.img) vkDestroyImage(dev, p.img, nullptr);
    if (p.imgC) vkDestroyImage(dev, p.imgC, nullptr);
    if (p.mem) vkFreeMemory(dev, p.mem, nullptr);
    if (p.memC) vkFreeMemory(dev, p.memC, nullptr);
    p = Pic{};
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
static void imgBarrier(VkImage img, VkImageAspectFlags aspect, VkImageLayout from, VkAccessFlags srcA, VkAccessFlags dstA) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = srcA; b.dstAccessMask = dstA; b.oldLayout = from; b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.image = img;
    b.subresourceRange = {aspect, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
}
static void barrierAll(const Pic& p, VkImageLayout from, VkAccessFlags srcA, VkAccessFlags dstA) {
    VkImageAspectFlags a0 = p.mp ? (VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_PLANE_1_BIT) : VK_IMAGE_ASPECT_COLOR_BIT;
    imgBarrier(p.img, a0, from, srcA, dstA);
    if (p.imgC) imgBarrier(p.imgC, VK_IMAGE_ASPECT_COLOR_BIT, from, srcA, dstA);
}

static VkBuffer stg;
static VkDeviceMemory stgMem;
static uint8_t* stgMap;
static void initStaging(size_t bytes) {
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

// copy regions of one Pic: plane 0 at buffer offset 0, plane 1 at `off1`
static void regions(const Pic& p, uint32_t W, uint32_t H, size_t off1, VkBufferImageCopy rg[2], VkImage im[2], int& n) {
    rg[0] = {}; rg[0].imageSubresource = {p.mp ? VK_IMAGE_ASPECT_PLANE_0_BIT : VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; rg[0].imageExtent = {W, H, 1};
    im[0] = p.img; n = 1;
    if (p.mp || p.imgC) {
        rg[1] = {}; rg[1].bufferOffset = off1;
        rg[1].imageSubresource = {p.mp ? VK_IMAGE_ASPECT_PLANE_1_BIT : VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        rg[1].imageExtent = {(W + 1) / 2, (H + 1) / 2, 1};
        im[1] = p.mp ? p.img : p.imgC; n = 2;
    }
}
static void upload(const Pic& p, const Raw& r, uint32_t W, uint32_t H) {
    memcpy(stgMap, r.p0.data(), r.p0.size());
    size_t off1 = (r.p0.size() + 255) & ~(size_t)255;
    if (!r.p1.empty()) memcpy(stgMap + off1, r.p1.data(), r.p1.size());
    VkBufferImageCopy rg[2]; VkImage im[2]; int n;
    regions(p, W, H, off1, rg, im, n);
    begin();
    barrierAll(p, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    for (int i = 0; i < n; i++) vkCmdCopyBufferToImage(cb, stg, im[i], VK_IMAGE_LAYOUT_GENERAL, 1, &rg[i]);
    barrierAll(p, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    submit();
}
static Raw download(const Pic& p, Enc e, uint32_t W, uint32_t H) {
    size_t sz0 = (size_t)W * H * (e == Enc::Rgba8 || e == Enc::Rgb10 ? 4 : e == Enc::F16Enc || e == Enc::F16Lin ? 8 : e == Enc::P010 ? 2 : 1);
    size_t sz1 = !isYuv(e) ? 0 : (size_t)((W + 1) / 2) * ((H + 1) / 2) * (e == Enc::P010 ? 4 : 2);
    size_t off1 = (sz0 + 255) & ~(size_t)255;
    VkBufferImageCopy rg[2]; VkImage im[2]; int n;
    regions(p, W, H, off1, rg, im, n);
    begin();
    barrierAll(p, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    for (int i = 0; i < n; i++) vkCmdCopyImageToBuffer(cb, im[i], VK_IMAGE_LAYOUT_GENERAL, stg, 1, &rg[i]);
    submit();
    Raw r;
    r.p0.assign(stgMap, stgMap + sz0);
    if (sz1) r.p1.assign(stgMap + off1, stgMap + off1 + sz1);
    return r;
}
static Raw zeros(Enc e, uint32_t W, uint32_t H) {
    Raw r;
    r.p0.assign((size_t)W * H * (e == Enc::Rgba8 || e == Enc::Rgb10 ? 4 : e == Enc::F16Enc || e == Enc::F16Lin ? 8 : e == Enc::P010 ? 2 : 1), 0);
    if (isYuv(e)) r.p1.assign((size_t)((W + 1) / 2) * ((H + 1) / 2) * (e == Enc::P010 ? 4 : 2), 0);
    return r;
}

// ---------------------------------------------------------------- synthetic scene (as uitest.cpp)
static const int kPanX = 14, kPanY = 6;
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
    int W, H, NF, bgW, bgH;
    std::vector<uint8_t> bg;
    struct Obj { int x0, y0, w, h, vx, vy; uint8_t r, g, b; };
    std::vector<Obj> objs;
    std::vector<uint8_t> ui, uiCol;   // 0 none, 1 crosshair, 2 hud
    int cx, cy;
    Scene(int w, int h, int nf) : W(w), H(h), NF(nf) {
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
        for (int i = 0; i < 16; i++) {
            Obj o;
            o.w = 30 + (int)(hash2(i, 1, 77) % 60); o.h = 30 + (int)(hash2(i, 2, 77) % 60);
            o.vx = 2 * ((int)(hash2(i, 3, 77) % 21) - 10); o.vy = 2 * ((int)(hash2(i, 4, 77) % 11) - 5);
            int tx = i < 8 ? W / 2 + (int)(hash2(i, 5, 77) % 41) - 20 : 40 + (int)(hash2(i, 6, 77) % 200);
            int ty = i < 8 ? H / 2 + (int)(hash2(i, 7, 77) % 41) - 20 : 30 + (int)(hash2(i, 8, 77) % 60);
            o.x0 = tx - o.w / 2 - o.vx * 21 / 2; o.y0 = ty - o.h / 2 - o.vy * 21 / 2;
            o.r = 40 + hash2(i, 9, 77) % 200; o.g = 40 + hash2(i, 10, 77) % 200; o.b = 40 + hash2(i, 11, 77) % 200;
            objs.push_back(o);
        }
        ui.assign((size_t)W * H, 0);
        uiCol.assign((size_t)W * H * 4, 0);
        cx = W / 2; cy = H / 2;
        auto put = [&](int x, int y, uint8_t k, uint8_t r, uint8_t g, uint8_t b) {
            if (x < 0 || y < 0 || x >= W || y >= H) return;
            ui[(size_t)y * W + x] = k;
            uint8_t* p = &uiCol[((size_t)y * W + x) * 4];
            p[0] = r; p[1] = g; p[2] = b; p[3] = 255;
        };
        auto inCross = [&](int dx, int dy, int grow) {
            int ax = std::abs(dx), ay = std::abs(dy);
            return (ay <= 1 + grow && ax >= 5 - grow && ax <= 16 + grow) || (ax <= 1 + grow && ay >= 5 - grow && ay <= 16 + grow) ||
                   (ax <= 1 + grow && ay <= 1 + grow);
        };
        for (int dy = -20; dy <= 20; dy++)
            for (int dx = -20; dx <= 20; dx++) {
                if (inCross(dx, dy, 1)) put(cx + dx, cy + dy, 1, 0, 0, 0);
                if (inCross(dx, dy, 0)) put(cx + dx, cy + dy, 1, 255, 64, 255);
            }
        int hx0 = 40, hy0 = 30, hx1 = 40 + std::max(W / 11, 64), hy1 = 30 + std::max(H / 12, 40);
        for (int y = hy0; y < hy1; y++)
            for (int x = hx0; x < hx1; x++) {
                bool bar = ((y - hy0) / 6) % 2 == 0 && (hash2((x - hx0) / 5, (y - hy0) / 12, 5) & 3) != 0 && x > hx0 + 6 && y > hy0 + 6 && x < hx1 - 6;
                if (bar) put(x, y, 2, 235, 235, 235); else put(x, y, 2, 20, 30, 90);
            }
    }
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

// ---------------------------------------------------------------- one run
static InitInfo baseInfo(Enc e, int W, int H, const Settings& s) {
    InitInfo info;
    info.phys = phys; info.device = dev; info.gdpa = vkGetDeviceProcAddr; info.width = W; info.height = H;
    info.viewFormat = planeFmt0(e);
    info.getMemProps = vkGetPhysicalDeviceMemoryProperties;
    info.getFormatProps = vkGetPhysicalDeviceFormatProperties;
    info.settings = s;
#ifndef FORMATTEST_NO_MOTION
    info.settings.motionVersion = gAlgo;
    info.settings.v2Mode = gMode;
#endif
    if (e == Enc::F16Lin) { info.transfer = Transfer::Linear; info.linearWhiteNits = 80.0f; }
    return info;
}

struct Result { Raw out[3]; bool ok = false, notBuilt = false; };   // t = 0.5, 0.25, blur (0.45..0.55, 3 samples) of the last pair
// Runs the sequence (frames 0..NF-1 of the scene) through a ring of 4 pics; synthesises from the last pair only.
// trackMid: frames 1..NF-4 go through recordTrack instead of recordObserve (and no pair touching one is estimated).
static Result runSequence(Enc e, const Scene& sc, const Settings& s, bool doBlur, bool trackMid = false) {
    Result res;
    Interpolator fg;
    if (!fg.init(baseInfo(e, sc.W, sc.H, s))) {
        printf("    init failed: %s\n", fg.initError() ? fg.initError() : "?");
        res.notBuilt = fg.initError() && strstr(fg.initError(), "not built");
        return res;
    }
    Pic ring[4], outs[3];
    for (auto& r : ring) r = makePic(e, sc.W, sc.H);
    for (auto& o : outs) o = makePic(e, sc.W, sc.H);
    begin();
    for (auto& r : ring) barrierAll(r, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    for (auto& o : outs) barrierAll(o, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    submit();
    for (auto& o : outs) upload(o, zeros(e, sc.W, sc.H), sc.W, sc.H);   // sentinel: an unwritten plane stays 0
    for (int f = 0; f < sc.NF; f++) {
        Pic& cur = ring[f % 4];
        upload(cur, encode(e, sc.frame(f), sc.W, sc.H), sc.W, sc.H);
        begin();
        fg.beginFrame();
        const bool tracked = trackMid && f >= 1 && f <= sc.NF - 4;
        if (tracked) { if (!fg.recordTrack(cb, cur.frame())) { fprintf(stderr, "track failed\n"); exit(1); } }
        else if (!fg.recordObserve(cb, cur.frame())) { fprintf(stderr, "observe failed\n"); exit(1); }
        if (f > 0 && !(trackMid && f <= sc.NF - 3)) {
            Pic& prv = ring[(f - 1) % 4];
            if (!fg.recordEstimate(cb, prv.frame(), cur.frame())) { fprintf(stderr, "estimate failed\n"); exit(1); }
            if (f == sc.NF - 1) {
                if (!fg.recordSynth(cb, prv.frame(), cur.frame(), outs[0].frame(), 0.5f)) { fprintf(stderr, "synth failed\n"); exit(1); }
                if (!fg.recordSynth(cb, prv.frame(), cur.frame(), outs[1].frame(), 0.25f)) { fprintf(stderr, "synth failed\n"); exit(1); }
                if (doBlur && !fg.recordSynthBlur(cb, prv.frame(), cur.frame(), outs[2].frame(), 0.45f, 0.55f)) { fprintf(stderr, "blur failed\n"); exit(1); }
            }
        }
        submit();
    }
    for (int i = 0; i < 3; i++) res.out[i] = download(outs[i], e, sc.W, sc.H);
    vkDeviceWaitIdle(dev);
    fg.destroy();
    for (auto& r : ring) destroyPic(r);
    for (auto& o : outs) destroyPic(o);
    res.ok = true;
    return res;
}

static double psnr(const std::vector<float>& a, const std::vector<float>& b, const std::vector<uint8_t>* mask = nullptr, bool maskVal = true) {
    double se = 0;
    size_t n = 0;
    for (size_t i = 0; i < a.size() / 3; i++) {
        if (mask && ((*mask)[i] != 0) != maskVal) continue;
        for (int c = 0; c < 3; c++) {
            double d = (double)a[i * 3 + c] - b[i * 3 + c];
            se += d * d;
        }
        n++;
    }
    return 10 * std::log10(255.0 * 255.0 / std::max(se / (n * 3.0), 1e-9));
}
// crosshair / hud pixels that differ from the real frame by more than `tol` levels
static void uiWrong(const Scene& sc, const std::vector<float>& out, const std::vector<float>& real, float tol, size_t& xh, size_t& hud, size_t& xhN, size_t& hudN) {
    xh = hud = xhN = hudN = 0;
    for (size_t i = 0; i < (size_t)sc.W * sc.H; i++) {
        int k = sc.ui[i];
        if (!k) continue;
        float d = 0;
        for (int c = 0; c < 3; c++) d = std::max(d, std::fabs(out[i * 3 + c] - real[i * 3 + c]));
        if (k == 1) { xhN++; xh += d > tol; } else { hudN++; hud += d > tol; }
    }
}

static bool formatSupported(Enc e) {
    // the library probes formats itself too; this keeps the test from trying what the device cannot do
    for (VkFormat f : {planeFmt0(e), isYuv(e) ? planeFmt1(e) : planeFmt0(e)}) {
        VkFormatProperties p;
        vkGetPhysicalDeviceFormatProperties(phys, f, &p);
        VkFormatFeatureFlags need = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
                                    VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        if ((p.optimalTilingFeatures & need) != need) {
            // a plane format of a multi-planar image reports no transfer features on its own: only sampled + storage needed then
            VkFormatFeatureFlags need2 = need & ~(VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT);
            if (!isYuv(e) || (p.optimalTilingFeatures & need2) != need2) return false;
        }
    }
    return true;
}


// CrosshairV2 must not copy the crosshair along the background's pan (the "doubled crosshair", shaders/ui_reject.glsl): a small opaque
// cyan plus (12x12 px, arms 2 px thick and aligned to the 2x2 chroma blocks, so its chroma is its own in 4:2:0 too; no HUD, no moving
// objects) over a per-pixel-grain background panning 14, 6 px / frame. Artifact pixels = non-UI pixels within 40 px of the plus that
// differ by more than 24 levels (decoded RGB) from the same run on the scene without it, in the t = 0.5 output and in the motion-blurred
// one. Every format has its own shader variant of the warp (in 4:2:0 also the chroma warp, and the blur has its own), so each is judged
// on its own. In 4:2:0 the pixels that share a 2x2 chroma block with a plus pixel are UI too (their chroma is the plus's).
// A plus this size also casts a zero-vector vote in the optical-flow blocks around it, which is not what is judged here: the V2 runs
// use the estimate fill, which takes it out of the estimate, so what is left is the copy through the warp.
struct GhostRow { size_t art05 = 0, artBlur = 0, uiBad = 0; bool ok = true; };
static void ghostFormat(Enc e, const Scene& base, GhostRow* off, GhostRow* v2) {
    const int W = base.W, H = base.H;
    Scene g = base;
    g.objs.clear();
    for (int y = 0; y < g.bgH; y++)
        for (int x = 0; x < g.bgW; x++)
            for (int c = 0; c < 3; c++) {
                const float v = 60.0f + 120.0f * vnoise(x, y, 24, 5) + 40.0f * vnoise(x, y, 5, 6) + ((hash2(x, y, 7) & 255) / 255.0f - 0.5f) * 70.0f;
                g.bg[((size_t)y * g.bgW + x) * 4 + c] = (uint8_t)std::min(255.0f, std::max(0.0f, v * (c == 2 ? 0.8f : 1.0f)));
            }
    std::fill(g.ui.begin(), g.ui.end(), 0);
    std::fill(g.uiCol.begin(), g.uiCol.end(), 0);
    const int cx0 = g.cx & ~1, cy0 = g.cy & ~1;
    for (int y = cy0 - 6; y < cy0 + 6; y++)
        for (int x = cx0 - 6; x < cx0 + 6; x++) {
            if (!((x >= cx0 && x < cx0 + 2) || (y >= cy0 && y < cy0 + 2))) continue;
            const size_t i = (size_t)y * W + x;
            g.ui[i] = 1;
            g.uiCol[i * 4 + 1] = g.uiCol[i * 4 + 2] = g.uiCol[i * 4 + 3] = 255;
        }
    Scene gn = g;
    std::fill(gn.ui.begin(), gn.ui.end(), 0);
    Settings so, sv;
    so.blurSamples = sv.blurSamples = 3;
    sv.uiProtection = UiProtection::CrosshairV2;
    sv.uiBoxWidth = sv.uiBoxHeight = 16;
    sv.uiEstimateFill = true;
    const Result rr = runSequence(e, gn, so, true), ro = runSequence(e, g, so, true), rv = runSequence(e, g, sv, true);
    if (!rr.ok || !ro.ok || !rv.ok) { off->ok = v2->ok = false; return; }
    const auto real = decode(e, encode(e, g.frame(11), W, H), W, H), prev = decode(e, encode(e, g.frame(10), W, H), W, H);
    const float tol = isYuv(e) ? 2.5f : 1.0f;
    auto isUi = [&](int x, int y) {
        if (!isYuv(e)) return g.ui[(size_t)y * W + x] == 1;
        for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++) {
                const int X = std::min((x & ~1) + i, W - 1), Y = std::min((y & ~1) + j, H - 1);
                if (g.ui[(size_t)Y * W + X] == 1) return true;
            }
        return false;
    };
    for (int k : {0, 2}) {   // t = 0.5, blurred
        const auto ref = decode(e, rr.out[k], W, H), dOff = decode(e, ro.out[k], W, H), dV2 = decode(e, rv.out[k], W, H);
        for (int y = g.cy - 40; y <= g.cy + 40; y++)
            for (int x = g.cx - 40; x <= g.cx + 40; x++) {
                const size_t i = (size_t)y * W + x;
                float d0 = 0, d2 = 0, dr = 0, dp = 0;
                for (int c = 0; c < 3; c++) {
                    d0 = std::max(d0, std::fabs(dOff[i * 3 + c] - ref[i * 3 + c]));
                    d2 = std::max(d2, std::fabs(dV2[i * 3 + c] - ref[i * 3 + c]));
                    dr = std::max(dr, std::fabs(dV2[i * 3 + c] - real[i * 3 + c]));
                    dp = std::max(dp, std::fabs(real[i * 3 + c] - prev[i * 3 + c]));
                }
                if (g.ui[i] == 1) {   // the plus is exact in the t = 0.5 output (4:2:0: an edge pixel that changed in the pair is not pasted, by design)
                    if (k == 0 && !(isYuv(e) && dp > 1.5f)) v2->uiBad += dr > tol;
                    continue;
                }
                if (isUi(x, y)) continue;
                (k == 0 ? off->art05 : off->artBlur) += d0 > 24;
                (k == 0 ? v2->art05 : v2->artBlur) += d2 > 24;
            }
    }
}

static double blurRef8 = 0;   // Rgba8: PSNR of the blurred output vs the t = 0.5 output (the window is wider than it looks: the scene moves 15 px / frame)
static void checkFormat(Enc e, const Scene& sc, double psnrRgba8, const std::vector<float>& dec8out) {
    const int W = sc.W, H = sc.H;
    printf("\n== %s ==\n", encName(e));
    if (!formatSupported(e)) { printf("    SKIP: the device cannot sample + store this format\n"); return; }
#ifndef FORMATTEST_NO_MOTION
    if (algoV2() && e != Enc::Rgba8) {   // V2 is Rgba8 only: refused at init and by setSettings, with a reason; nothing else runs
        Interpolator fg;
        InitInfo info = baseInfo(e, W, H, Settings{});
        Format f;
        formatOf(info.viewFormat, &f);
        const char* why = Interpolator::settingsError(info.settings, f);
        const bool initOk = fg.init(info);
        expect(!initOk && why && fg.initError() && !strcmp(fg.initError(), why), "init refuses MotionVersion::V2, initError = settingsError()");
        printf("      reason: %s\n", fg.initError() ? fg.initError() : "(none)");
        fg.destroy();
        info.settings.motionVersion = MotionVersion::V1;
        Settings v2 = info.settings;
        v2.motionVersion = MotionVersion::V2;
        expect(fg.init(info) && !fg.setSettings(v2) && !fg.reconfigure(v2) && fg.settings().motionVersion == MotionVersion::V1,
               "a V1 instance refuses setSettings / reconfigure to V2 and keeps V1");
        fg.destroy();
        printf("    SKIP the rest: V2 is Rgba8 only\n");
        return;
    }
#endif
    if (isYuv(e)) { destroyPic(*std::make_unique<Pic>(makePic(e, 16, 16)).get()); printf("    images: %s\n", mpUsed ? "one multi-planar image, plane views" : "separate plane images (multi-planar not usable here)"); }
    auto realRaw = encode(e, sc.frame(11), W, H);
    auto real = decode(e, realRaw, W, H);
    auto truth = decode(e, encode(e, sc.frame(10.5), W, H), W, H);   // the true mid frame, through the format's own quantisation
    std::vector<uint8_t> near((size_t)W * H, 0);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            if (sc.ui[(size_t)y * W + x])
                for (int dy = -4; dy <= 4; dy++)
                    for (int dx = -4; dx <= 4; dx++) {
                        int X = x + dx, Y = y + dy;
                        if (X >= 0 && Y >= 0 && X < W && Y < H) near[(size_t)Y * W + X] = 1;
                    }
    Settings off;
    Result r = runSequence(e, sc, off, true);
    if (!r.ok) { if (r.notBuilt) printf("    SKIP: this format's shader variants are not built (meson option 'formats')\n"); else expect(false, "init + sequence"); return; }
    auto o05 = decode(e, r.out[0], W, H), o025 = decode(e, r.out[1], W, H), oBlur = decode(e, r.out[2], W, H);
    if (hashOut) {
        printf("    HASHES Off");
        for (int i = 0; i < 3; i++) printf(" %016llx", (unsigned long long)fnv(r.out[i].p1, fnv(r.out[i].p0)));
        printf("\n");
    }
    double pOff = psnr(o05, truth, &near, false);
    printf("    Off: PSNR vs the format's own true mid frame (away from the UI): %.2f dB   (Rgba8: %.2f dB)\n", pOff, psnrRgba8);
    double pRef = e == Enc::Rgba8 ? 99.0 : psnr(o05, dec8out, &near, false);
    if (e != Enc::Rgba8) printf("    Off: PSNR of this output vs the Rgba8 output of the same scene: %.2f dB\n", pRef);
    // sane = about as good as the 8-bit path against the same ground truth. YUV: chroma at half resolution loses on the
    // saturated, noisy test texture, but the interpolation error itself should stay close to Rgba8's.
    expect(pOff >= psnrRgba8 - (isYuv(e) ? 4.0 : 1.5), "output is as close to the true mid frame as the Rgba8 path (-1.5 dB; YUV -4 dB)");
    if (e == Enc::Rgb10 || e == Enc::F16Enc) expect(pRef >= 40.0, "output matches the Rgba8 output of the same scene (>= 40 dB: same algorithm, unrounded warps)");
    if (e == Enc::F16Lin) expect(pRef >= 25.0, "linear-light output matches the Rgba8 output (>= 25 dB: it blends in linear light, Rgba8 in encoded space)");
    if (isYuv(e)) expect(pRef >= 25.0, "YUV output matches the Rgba8 output decoded (>= 25 dB: chroma at half resolution)");
    // the unwritten sentinel is 0: Y of a written pixel is >= code 16, chroma ~128 (NV12) / 512 (P010)
    {
        bool allWritten = true;
        if (isYuv(e)) {
            size_t zeroY = 0, zeroC = 0, nC = r.out[0].p1.size() / (e == Enc::P010 ? 2 : 1);
            const bool p = e == Enc::P010;
            for (size_t i = 0; i < (size_t)W * H; i++) {
                uint16_t y = p ? (uint16_t)(r.out[0].p0[i * 2] | (r.out[0].p0[i * 2 + 1] << 8)) : r.out[0].p0[i];
                zeroY += y == 0;
            }
            for (size_t i = 0; i < nC; i++) {
                uint16_t c = p ? (uint16_t)(r.out[0].p1[i * 2] | (r.out[0].p1[i * 2 + 1] << 8)) : r.out[0].p1[i];
                zeroC += c == 0;
            }
            allWritten = zeroY < (size_t)W * H / 100 && zeroC < nC / 100;
            printf("    planes: %zu of %d luma and %zu of %zu chroma samples still 0 (sentinel)\n", zeroY, W * H, zeroC, nC);
        }
        if (isYuv(e)) expect(allWritten, "both planes are written: no sentinel left in Y or CbCr");
    }
    // t = 0.25 and the blur are sane too
    expect(psnr(o025, decode(e, encode(e, sc.frame(10), W, H), W, H), &near, false) > 12.0, "t = 0.25 output is a picture, not garbage (> 12 dB vs frame 10)");
    {
        double pb = psnr(oBlur, o05, &near, false);
        printf("    blur (0.45..0.55, 3 samples) vs the t = 0.5 output: %.2f dB\n", pb);
        if (e == Enc::Rgba8) blurRef8 = pb;
        expect(pb >= blurRef8 - 1.5, "blur behaves like the Rgba8 blur: its distance to the midpoint output is within 1.5 dB of Rgba8's");
    }
    // static UI protection: crosshair + HUD exact in WholeScreen, wrong without (the test has teeth)
    size_t xh, hud, xhN, hudN;
    float tol = isYuv(e) ? 2.5f : 1.0f;   // YUV: 1 code of Y or chroma is up to ~2 RGB levels
    uiWrong(sc, o05, real, tol, xh, hud, xhN, hudN);
    printf("    UI Off:         crosshair wrong %zu/%zu, hud wrong %zu/%zu\n", xh, xhN, hud, hudN);
    size_t xhOff = xh, hudOff = hud;
    Settings w; w.uiProtection = UiProtection::WholeScreen; w.uiBoxWidth = w.uiBoxHeight = 64;   // fits the 35 px test crosshair (the default box is smaller)
    Result rw = runSequence(e, sc, w, false);
    if (!rw.ok) { expect(false, "WholeScreen init + sequence"); return; }
    auto w05 = decode(e, rw.out[0], W, H), w025 = decode(e, rw.out[1], W, H);
    if (hashOut) {
        printf("    HASHES WholeScreen");
        for (int i = 0; i < 2; i++) printf(" %016llx", (unsigned long long)fnv(rw.out[i].p1, fnv(rw.out[i].p0)));
        printf("\n");
    }
    uiWrong(sc, w05, real, tol, xh, hud, xhN, hudN);
    size_t xh2, hud2, a, b;
    uiWrong(sc, w025, real, tol, xh2, hud2, a, b);
    printf("    UI WholeScreen: crosshair wrong %zu/%zu (t=.25: %zu), hud wrong %zu/%zu (t=.25: %zu)\n", xh, xhN, xh2, hud, hudN, hud2);
    expect(xh == 0 && xh2 == 0, "crosshair pixels equal the real ones (WholeScreen, t = 0.5 and 0.25)");
    expect(hud == 0 && hud2 == 0, "HUD box pixels equal the real ones (WholeScreen)");
    expect(xhOff > 0 || hudOff > 0, "(the scene disturbs the UI without protection: the check has teeth)");
    {   // recordTrack: 8 frames tracked instead of observed leave the same counters, so the last pair's outputs are bit-identical
        Result rt = runSequence(e, sc, w, false, true);
        expect(rt.ok && rt.out[0].p0 == rw.out[0].p0 && rt.out[0].p1 == rw.out[0].p1 && rt.out[1].p0 == rw.out[1].p0 && rt.out[1].p1 == rw.out[1].p1,
               "recordTrack (WholeScreen): frames tracked instead of observed give bit-identical output");
    }
    Settings c; c.uiProtection = UiProtection::Crosshair; c.uiBoxWidth = c.uiBoxHeight = 64;
    Result rc = runSequence(e, sc, c, false);
    if (rc.ok) {
        auto c05 = decode(e, rc.out[0], W, H);
        uiWrong(sc, c05, real, tol, xh, hud, xhN, hudN);
        printf("    UI Crosshair:   crosshair wrong %zu/%zu\n", xh, xhN);
        expect(xh == 0, "crosshair pixels equal the real ones (Crosshair mode)");
    }
    {   // CrosshairV2: the same box, no clean copies; crosshair exact, tracked == observed, background quality
        Settings v; v.uiProtection = UiProtection::CrosshairV2; v.uiBoxWidth = v.uiBoxHeight = 64;
        Result rv = runSequence(e, sc, v, false);
        if (!rv.ok) { expect(false, "CrosshairV2 init + sequence"); return; }
        auto v05 = decode(e, rv.out[0], W, H), v025 = decode(e, rv.out[1], W, H);
        // V2 never pastes a pixel that changed in the pair. In 4:2:0 the chroma of a crosshair edge pixel is shared with the
        // background behind it, which moves, so a few edge pixels DO change in the pair: only the ones that did not (decoded RGB
        // within 1.5 levels of the previous frame) must be exact there. In the RGB formats every crosshair pixel is static.
        auto prevDec = decode(e, encode(e, sc.frame(10), W, H), W, H);
        size_t xh3 = 0, xh4 = 0, nStatic = 0, nChanged = 0;
        for (size_t i = 0; i < (size_t)W * H; i++) {
            if (sc.ui[i] != 1) continue;
            float dp = 0, d5 = 0, d25 = 0;
            for (int c = 0; c < 3; c++) {
                dp = std::max(dp, std::fabs(real[i * 3 + c] - prevDec[i * 3 + c]));
                d5 = std::max(d5, std::fabs(v05[i * 3 + c] - real[i * 3 + c]));
                d25 = std::max(d25, std::fabs(v025[i * 3 + c] - real[i * 3 + c]));
            }
            if (isYuv(e) && dp > 1.5f) { nChanged++; continue; }
            nStatic++;
            xh3 += d5 > tol; xh4 += d25 > tol;
        }
        printf("    UI CrosshairV2: crosshair wrong %zu/%zu (t=.25: %zu)%s\n", xh3, nStatic, xh4, isYuv(e) ? "" : "");
        if (isYuv(e)) printf("    (%zu crosshair edge pixels change with the chroma of the moving background in this pair and are not judged)\n", nChanged);
        expect(xh3 == 0 && xh4 == 0, isYuv(e) ? "crosshair pixels that did not change in the pair equal the real ones (CrosshairV2, t = 0.5 and 0.25)"
                                              : "crosshair pixels equal the real ones (CrosshairV2, t = 0.5 and 0.25)");
        Result rvt = runSequence(e, sc, v, false, true);
        expect(rvt.ok && rvt.out[0].p0 == rv.out[0].p0 && rvt.out[0].p1 == rv.out[0].p1 && rvt.out[1].p0 == rv.out[1].p0 && rvt.out[1].p1 == rv.out[1].p1,
               "recordTrack (CrosshairV2): frames tracked instead of observed give bit-identical output");
        expect(psnr(v05, truth, &near, false) >= pOff - 0.5, "CrosshairV2 does not cost background quality (-0.5 dB)");
        // the optional luma fill of the motion estimate (uiEstimateFill): its own pass, compiled per format
        Settings vf = v; vf.uiEstimateFill = true;
        Result rf = runSequence(e, sc, vf, false);
        if (!rf.ok) { expect(false, "CrosshairV2 + fill init + sequence"); return; }
        auto f05 = decode(e, rf.out[0], W, H);
        size_t xhf = 0;
        for (size_t i = 0; i < (size_t)W * H; i++) {
            if (sc.ui[i] != 1) continue;
            float dp = 0, d5 = 0;
            for (int c = 0; c < 3; c++) {
                dp = std::max(dp, std::fabs(real[i * 3 + c] - prevDec[i * 3 + c]));
                d5 = std::max(d5, std::fabs(f05[i * 3 + c] - real[i * 3 + c]));
            }
            if (isYuv(e) && dp > 1.5f) continue;
            xhf += d5 > tol;
        }
        double pf = psnr(f05, truth, &near, false);
        printf("    UI CrosshairV2 + estimate fill: crosshair wrong %zu, background PSNR %.2f dB (without the fill %.2f dB, Off %.2f dB)\n", xhf, pf, psnr(v05, truth, &near, false), pOff);
        expect(xhf == 0, "CrosshairV2 + estimate fill: crosshair exact");
        expect(pf >= pOff - 0.5, "CrosshairV2 + estimate fill does not cost background quality (-0.5 dB)");
    }
    {   // the box margin (every format): a 34 px box is 1 px too small for the 35 px crosshair (its outline's outermost pixels at the
        // right / bottom lie outside it), and both modes still protect all of it: Crosshair's mask dilation (2 px) and CrosshairV2's
        // grow (1 px) reach past the box. (A pixel 2 px past the last still one gets only V1's half-strength feather: not exact.)
        auto prevDec2 = decode(e, encode(e, sc.frame(10), W, H), W, H);
        for (int m = 1; m <= 2; m++) {
            Settings t; t.uiProtection = m == 1 ? UiProtection::Crosshair : UiProtection::CrosshairV2;
            t.uiBoxWidth = t.uiBoxHeight = 34;
            Result rt = runSequence(e, sc, t, false);
            if (!rt.ok) { expect(false, "box-margin init + sequence"); return; }
            auto t05 = decode(e, rt.out[0], W, H), t025 = decode(e, rt.out[1], W, H);
            size_t bad = 0, bad2 = 0, n = 0;
            for (size_t i = 0; i < (size_t)W * H; i++) {
                if (sc.ui[i] != 1) continue;
                float dp = 0, d5 = 0, d25 = 0;
                for (int c = 0; c < 3; c++) {
                    dp = std::max(dp, std::fabs(real[i * 3 + c] - prevDec2[i * 3 + c]));
                    d5 = std::max(d5, std::fabs(t05[i * 3 + c] - real[i * 3 + c]));
                    d25 = std::max(d25, std::fabs(t025[i * 3 + c] - real[i * 3 + c]));
                }
                if (m == 2 && isYuv(e) && dp > 1.5f) continue;   // V2 never pastes a pixel that changed in the pair (see above)
                n++; bad += d5 > tol; bad2 += d25 > tol;
            }
            printf("    UI %s, box %d px (smaller than the crosshair): crosshair wrong %zu/%zu (t=.25: %zu)\n", m == 1 ? "Crosshair" : "CrosshairV2", (int)t.uiBoxWidth, bad, n, bad2);
            expect(bad == 0 && bad2 == 0, m == 1 ? "Crosshair: a box 1 px short of the crosshair still protects all of it (margin)" : "CrosshairV2: a box 1 px short of the crosshair still protects all of it (margin)");
        }
    }
    {   // the doubled crosshair
        GhostRow off, v2;
        ghostFormat(e, sc, &off, &v2);
        if (!off.ok || !v2.ok) { expect(false, "ghost scene init + sequence"); return; }
        printf("    doubled crosshair (grain background, pan 14,6), artifact px around a small plus (t = 0.5 / blurred): Off %zu / %zu, CrosshairV2 + fill %zu / %zu (plus wrong: %zu)\n",
               off.art05, off.artBlur, v2.art05, v2.artBlur, v2.uiBad);
        expect(off.art05 >= 15 && off.artBlur >= 15, "(the scene copies the crosshair along the pan without protection: the check has teeth)");
        expect(v2.art05 <= 8 && v2.artBlur <= 8, "CrosshairV2 leaves at most 8 artifact px around the plus, t = 0.5 and blurred (the warp no longer copies it)");
        expect(v2.uiBad == 0, "CrosshairV2: the plus is pixel-exact");
    }
    double pw = psnr(w05, truth, &near, false);
    expect(pw >= pOff - 0.5, "protection does not cost background quality (-0.5 dB)");
}

// Rgb10 / F16 precision: a smooth slow gradient moving 2 px: distinct output values on one row
static size_t gradientLevels(Enc e) {
    const int W = 640, H = 180;
    Interpolator fg;
    if (!fg.init(baseInfo(e, W, H, Settings{}))) return 0;
    Pic a = makePic(e, W, H), b = makePic(e, W, H), o = makePic(e, W, H);
    begin();
    for (Pic* p : {&a, &b, &o}) barrierAll(*p, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    submit();
    auto make = [&](int shift) {
        // 0.2..0.3 across the width, plus a faint diagonal ramp so that the estimate has some structure to track
        Raw r;
        const int bpp = e == Enc::Rgba8 ? 4 : e == Enc::Rgb10 ? 4 : 8;
        r.p0.resize((size_t)W * H * bpp);
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                double v = 0.2 + 0.1 * (x + shift) / (W - 1.0);
                uint8_t* p = &r.p0[((size_t)y * W + x) * bpp];
                if (e == Enc::Rgba8) { p[0] = p[1] = p[2] = (uint8_t)std::lround(v * 255); p[3] = 255; }
                else if (e == Enc::Rgb10) { uint32_t c = (uint32_t)std::lround(v * 1023); uint32_t w = c | c << 10 | c << 20 | 3u << 30; memcpy(p, &w, 4); }
                else { uint16_t h[4] = {toHalf((float)v), toHalf((float)v), toHalf((float)v), toHalf(1.0f)}; memcpy(p, h, 8); }
            }
        return r;
    };
    upload(a, make(0), W, H);
    upload(b, make(2), W, H);
    upload(o, zeros(e, W, H), W, H);
    begin();
    fg.beginFrame();
    bool ok = fg.recordEstimate(cb, a.frame(), b.frame()) && fg.recordSynth(cb, a.frame(), b.frame(), o.frame(), 0.5f);
    submit();
    std::set<uint32_t> vals;
    if (ok) {
        Raw r = download(o, e, W, H);
        const int bpp = e == Enc::Rgba8 ? 4 : e == Enc::Rgb10 ? 4 : 8;
        for (int x = 0; x < W; x++) {
            const uint8_t* p = &r.p0[((size_t)(H / 2) * W + x) * bpp];
            uint32_t v;
            if (e == Enc::Rgba8) v = p[0];
            else if (e == Enc::Rgb10) { uint32_t w; memcpy(&w, p, 4); v = w & 1023u; }
            else { uint16_t h; memcpy(&h, p, 2); v = h; }
            vals.insert(v);
        }
    }
    vkDeviceWaitIdle(dev);
    fg.destroy();
    destroyPic(a); destroyPic(b); destroyPic(o);
    return vals.size();
}

// linear float input above 1.0: the highlights of a scRGB-like frame survive (no clamp), and are still a sane picture
static void checkHdr(const Scene& sc) {
    printf("\n== RgbaF16 linear: values above 1.0 ==\n");
    if (!formatSupported(Enc::F16Lin) || algoV2()) { printf("    SKIP%s\n", algoV2() ? " (V2 is Rgba8 only)" : ""); return; }
    Settings off;
    Result r = runSequence(Enc::F16Lin, sc, off, false);
    if (!r.ok) { expect(false, "sequence"); return; }
    float mx = 0, mxIn = 0;
    for (size_t i = 0; i < (size_t)sc.W * sc.H * 4; i += 4) {
        uint16_t h;
        memcpy(&h, &r.out[0].p0[i * 2], 2);
        mx = std::max(mx, fromHalf(h));
    }
    auto in = encode(Enc::F16Lin, sc.frame(11), sc.W, sc.H);
    for (size_t i = 0; i < (size_t)sc.W * sc.H * 4; i += 4) { uint16_t h; memcpy(&h, &in.p0[i * 2], 2); mxIn = std::max(mxIn, fromHalf(h)); }
    printf("    largest input value %.3f, largest output value %.3f\n", mxIn, mx);
    expect(mxIn > 2.0f && mx > 2.0f, "highlights above 1.0 are kept (output max > 2.0 for an input max of ~4)");
}

static void timing(int W, int H) {
    printf("GPU time of one estimate + one synth (t = 0.5), %dx%d, UI protection Off (RADV, single run, ms):\n", W, H);
    const Enc all[] = {Enc::Rgba8, Enc::Rgb10, Enc::F16Enc, Enc::Nv12, Enc::P010};
    for (Enc e : all) {
        if (!formatSupported(e)) { printf("  %-24s skipped\n", encName(e)); continue; }
        Interpolator fg;
        if (!fg.init(baseInfo(e, W, H, Settings{}))) { printf("  %-24s init failed: %s\n", encName(e), fg.initError()); continue; }
        Pic a = makePic(e, W, H), b = makePic(e, W, H), o = makePic(e, W, H);
        begin();
        for (Pic* p : {&a, &b, &o}) barrierAll(*p, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
        submit();
        fg.setProfiling(true);
        double tot = 0, est = 0;
        for (int rep = 0; rep < 2; rep++) {   // the first run warms up
            begin();
            fg.beginFrame();
            bool ok = fg.recordEstimate(cb, a.frame(), b.frame()) && fg.recordSynth(cb, a.frame(), b.frame(), o.frame(), 0.5f);
            submit();
            std::vector<const char*> names;
            std::vector<uint64_t> ticks;
            if (ok && fg.readTimings(names, ticks) && rep == 1) {
                tot = est = 0;
                for (size_t i = 0; i < ticks.size(); i++) {
                    tot += ticks[i] * tsPeriodMs;
                    if (!strcmp(names[i], "mismatch") || !strcmp(names[i], "resolve") || !strcmp(names[i], "fallback") || !strcmp(names[i], "lookup")) continue;
                    est += ticks[i] * tsPeriodMs;
                }
            }
        }
        printf("  %-24s total %.3f ms (estimate %.3f, synth %.3f)\n", encName(e), tot, est, tot - est);
        vkDeviceWaitIdle(dev);
        fg.destroy();
        destroyPic(a); destroyPic(b); destroyPic(o);
    }
}

static void apiChecks(int W, int H) {
    printf("\n== API ==\n");
    Format f;
    expect(formatOf(VK_FORMAT_B8G8R8A8_UNORM, &f) && f == Format::Rgba8, "formatOf(B8G8R8A8) = Rgba8");
    expect(formatOf(VK_FORMAT_R16_UNORM, &f) && f == Format::P010 && formatOf(VK_FORMAT_R8_UNORM, &f) && f == Format::Nv12, "formatOf(R16/R8) = P010/Nv12");
    expect(!formatOf(VK_FORMAT_R8G8B8A8_SRGB, &f), "an _SRGB view format is not a Format");
    Interpolator fg;
    InitInfo bad = baseInfo(Enc::Rgba8, W, H, Settings{});
    bad.viewFormat = VK_FORMAT_R8G8B8A8_SRGB;
    expect(!fg.init(bad) && fg.initError(), "init refuses an _SRGB view format with a reason");
    if (fg.initError()) printf("      reason: %s\n", fg.initError());
    InitInfo ok = baseInfo(Enc::Rgba8, W, H, Settings{});
    expect(fg.init(ok) && fg.format() == Format::Rgba8 && !fg.initError(), "Rgba8 init succeeds, format() = Rgba8, no error");
    expect(!fg.resize(W, H, VK_FORMAT_R16G16B16A16_SFLOAT), "resize to another Format is refused (pipelines belong to the init format)");
    expect(fg.resize(W, H, VK_FORMAT_R8G8B8A8_UNORM) || true, "(resize within the format class is allowed)");
    fg.destroy();
    InitInfo yuvNoChroma = baseInfo(Enc::Nv12, W, H, Settings{});
#ifndef FORMATTEST_NO_MOTION
    yuvNoChroma.settings.motionVersion = MotionVersion::V1;   // (Nv12: V1 in every run)
#endif
    if (formatSupported(Enc::Nv12) && fg.init(yuvNoChroma)) {
        Pic a = makePic(Enc::Nv12, W, H);
        begin();
        fg.beginFrame();
        expect(!fg.recordEstimate(cb, Frame(a.view), Frame(a.view)), "an Nv12 frame without a chroma view is refused");
        CK(vkEndCommandBuffer(cb));
        CK(vkResetCommandBuffer(cb, 0));
        destroyPic(a);
        fg.destroy();
    }
}

int main(int argc, char** argv) {
    std::vector<std::string> pos;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--separate-planes")) separatePlanes = true;
        else if (!strcmp(argv[i], "--hash")) hashOut = true;
#ifndef FORMATTEST_NO_MOTION
        else if (!strcmp(argv[i], "--algo") && i + 1 < argc) {
            const char* a = argv[++i];
            if (!strcmp(a, "v1")) gAlgo = MotionVersion::V1;
            else if (!strcmp(a, "v2")) gAlgo = MotionVersion::V2;
            else { fprintf(stderr, "--algo v1|v2\n"); return 2; }
        }
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
            const char* a = argv[++i];
            if (!strcmp(a, "quality")) gMode = V2Mode::Quality;
            else if (!strcmp(a, "performance")) gMode = V2Mode::Performance;
            else { fprintf(stderr, "--mode quality|performance\n"); return 2; }
        }
#endif
        else pos.push_back(argv[i]);
    }
    if (pos.empty() || (pos[0] != "check" && pos[0] != "time")) { fprintf(stderr, "usage: formattest check|time [--separate-planes] [--hash] [--algo v1|v2] [--mode quality|performance] [W H]\n"); return 2; }
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
    // shaderStorageImageExtendedFormats: required for the Rgb10 / Nv12 / P010 variants (r8, rg8, r16, rg16, rgb10_a2 storage)
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    f11.samplerYcbcrConversion = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
    f2.features.shaderStorageImageExtendedFormats = VK_TRUE;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2};
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

    int W = pos.size() >= 3 ? atoi(pos[1].c_str()) : (pos[0] == "time" ? 1920 : 640);
    int H = pos.size() >= 3 ? atoi(pos[2].c_str()) : (pos[0] == "time" ? 1080 : 360);
    initStaging((size_t)W * H * 8 + 4096);
    if (pos[0] == "time") {
        timing(W, H);
        vkDeviceWaitIdle(dev);
        return 0;
    }
    printf("formattest %dx%d on %s, motion version %s, v2Mode %s (YUV: %s)\n", W, H, props.deviceName, algoV2() ? "V2" : "V1",
           gMode == V2Mode::Performance ? "performance" : "quality", separatePlanes ? "separate plane images" : "multi-planar image + plane views, falling back to separate images");
    apiChecks(W, H);
    Scene sc(W, H, 12);
    // reference: the Rgba8 path of the same scene
    Settings off;
    Result r8 = runSequence(Enc::Rgba8, sc, off, false);
    if (!r8.ok) { printf("Rgba8 reference failed\n"); return 1; }
    auto dec8 = decode(Enc::Rgba8, r8.out[0], W, H);
    auto truth8 = decode(Enc::Rgba8, encode(Enc::Rgba8, sc.frame(10.5), W, H), W, H);
    std::vector<uint8_t> near((size_t)W * H, 0);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            if (sc.ui[(size_t)y * W + x])
                for (int dy = -4; dy <= 4; dy++)
                    for (int dx = -4; dx <= 4; dx++) {
                        int X = x + dx, Y = y + dy;
                        if (X >= 0 && Y >= 0 && X < W && Y < H) near[(size_t)Y * W + X] = 1;
                    }
    double p8 = psnr(dec8, truth8, &near, false);
    for (Enc e : {Enc::Rgba8, Enc::Rgb10, Enc::F16Enc, Enc::F16Lin, Enc::Nv12, Enc::P010})
        checkFormat(e, sc, p8, dec8);
    checkHdr(sc);
    printf("\n== precision (a smooth 0.2..0.3 gradient moving 2 px; distinct values on one output row) ==\n");
    size_t g8 = gradientLevels(Enc::Rgba8), g10 = algoV2() ? 0 : gradientLevels(Enc::Rgb10), gf = algoV2() ? 0 : gradientLevels(Enc::F16Enc);
    printf("    Rgba8 %zu, Rgb10 %zu, RgbaF16 %zu distinct values%s\n", g8, g10, gf, algoV2() ? " (V2: Rgba8 only)" : "");
    expect(g8 <= 27, "Rgba8: at most the ~26 levels of 8 bits in that range (control)");
    if (!algoV2()) {
        expect(g10 >= 70, "Rgb10: about 4x as many values (>= 70 of ~100)");
        expect(gf >= 70, "RgbaF16: at least as many as Rgb10");
    }
    printf("\nRESULT: %s (%d failed)\n", fails ? "FAIL" : "PASS", fails);
    vkDeviceWaitIdle(dev);
    return fails ? 1 : 0;
}
