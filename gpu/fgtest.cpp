// fgtest: runs framegen::Interpolator on raw RGBA8 frame pairs and/or benchmarks it.
//   fgtest [--bgra] [--format F] [--split] [--t T1,T2,..] [--bench N [--mult M]] [--set k=v,..] [--then k=v,..] [--stats] [--window N]
//          [--dump DIR] W H prev.rgba curr.rgba out.rgba [prev curr out ...]
// Files are tightly packed RGBA8, W*H*4 bytes. --t: generated times (default 0.5). With one t and no --split
// each pair uses record(t); otherwise recordEstimate once + recordSynth per t, each into its own output
// image, all in one command buffer. With several t the k-th output goes to out.rgba.<k>.
// --bench N --mult M: per real pair one estimate + M-1 synths at t = k/M (M = 2: record(0.5)). --bgra uses B8G8R8A8_UNORM images/views for
// inputs and output (the file data stays RGBA; channels are swizzled on upload/readback).
// --format rgba8|rgb10|f16|f16lin|nv12|p010: the image format of inputs and outputs (files stay RGBA8, converted on upload and
// readback; rgb10 / f16 / p010 as PQ HDR, f16lin as scRGB, see Fmt below).
// --set k=v[,k=v..]: framegen::Settings (keys = field names: trustLow trustHigh globalFallback sceneCutSad hudBonus
// smoothBonus searchPenalty subpixel flowScale v2StaticMask; algo=v1|v2|learned = motionVersion; v2Mode=quality|performance;
// learnedNet=e2v3|e1v3|e2v4|e1v4|e2v5|e1v5|g2v2|g2v3 (MotionVersion::Learned's net), learnedHalf, learnedZeroTest, learnedStaticGuard (0|1); ui=off|crosshair|wholescreen|crosshairv2 =
// uiProtection, with uiHistory uiBoxHeightFrac uiBoxWidth uiBoxHeight uiCrosshairArea uiBoxOffsetX uiBoxOffsetY uiStillFrames
// uiStillThreshold uiInpaintReach uiMaxStillFraction uiContrastThreshold uiLeaveFrames uiReentryFrames uiEstimateFill), passed to init.
// UI protection (ui != off in --set or --then): every real frame is recordObserve'd once, in pair order, into a ring of 3 input
// images; consecutive pairs (pair k+1's prev path == pair k's curr path) form one frame stream, as a host sees it. A fresh
// Interpolator (start, --window) has to warm up: CrosshairV2 / Crosshair mask a pixel only after uiStillFrames (8) still frames,
// i.e. from pair uiStillFrames - 1 (0-based) of a consecutive run on. --then k=v[,..]: per pair, in the
// same command buffer as record(t[0]) -> out.rgba.0, setSettings(--set + these) then record(t[0]) -> out.rgba.1 (runtime-change
// test; --then algo=.. switches the motion version between the two records).
// --stats: after each pair print "STATS <pair> <untrusted px> <SAD sum x256> <SAD blocks> <W*H>" to stdout (the inputs of the
// whole-frame fallback / scene-cut decisions of the pair's last synth, via Interpolator::recordDebugCounters; read-only).
// --window N: destroy + init the Interpolator before every N-th pair (pair 0, N, 2N, ..): independent windows of consecutive
// pairs in one call, no state carried across (bench/score; UI protection warms up again per window). Neither changes any output.
// --dump DIR (created if missing): after each pair write DIR/pair_NNNN.fgmaps (Interpolator::recordDebugMaps, read-only, output unchanged): a 64-byte
// header (uint32 magic 'FGMP', version 5, W, H, fw, fh, bw, bh, fscale, eStride, float trustLow, trustHigh, smoothBonus, hudBonus,
// uint32 motionVersion, v2Mode), then the block field (bw*bh vec2 f32), the per-pixel half-vector field (fw*fh vec2 f32, midpoint
// grid) and the packed mismatch e (eStride*H uint32, 4 px per word, low byte first) of the pair's last record. gpu/fgmaps.py turns it
// into PNG maps. MotionVersion::V2: Performance writes its cleaned block field and zeros; Quality its cleaned block field, its
// per-pixel field and, at the start of the e part, its coverage (see fgmaps.py). (Version 1: the same maps, header words 14-15 = 0. Versions 2-4 were the old MotionVersion::V2's, tag v2-occlusion:
// the same maps followed by its own, which no longer exist.)
// All images are kept in VK_IMAGE_LAYOUT_GENERAL, like gamescope.
#include "framegen.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d\n", #x, r_); exit(1); } } while (0)

// "k=v,k=v" -> fields of s; false on an unknown key
static bool parseSettings(char* str, framegen::Settings& s) {
    for (char* tok = strtok(str, ","); tok; tok = strtok(nullptr, ",")) {
        char* eq = strchr(tok, '=');
        if (!eq) return false;
        *eq = 0;
        double v = atof(eq + 1);
        std::string k = tok;
        if (k == "algo") {   // the motion version: v1 | v2 | learned
            std::string a = eq + 1;
            if (a == "v1") s.motionVersion = framegen::MotionVersion::V1;
            else if (a == "v2") s.motionVersion = framegen::MotionVersion::V2;
            else if (a == "learned") s.motionVersion = framegen::MotionVersion::Learned;
            else { fprintf(stderr, "algo=v1|v2|learned, not %s\n", eq + 1); return false; }
        }
        else if (k == "learnedNet") {   // MotionVersion::Learned's net by its name (framegen::learnedNetName: e2v3 | e1v3 | e2v4 | e1v4 | e2v5 | e1v5 | c2smoke)
            if (!framegen::learnedNetFromName(eq + 1, &s.learnedNet)) {
                fprintf(stderr, "learnedNet=");
                for (uint32_t n = 0; n < framegen::kLearnedNets; n++) fprintf(stderr, "%s%s", n ? "|" : "", framegen::learnedNetName((framegen::LearnedNet)n));
                fprintf(stderr, ", not %s\n", eq + 1);
                return false;
            }
        }
        else if (k == "v2Mode") {   // MotionVersion::V2's mode: quality | performance
            std::string a = eq + 1;
            if (a == "quality") s.v2Mode = framegen::V2Mode::Quality;
            else if (a == "performance") s.v2Mode = framegen::V2Mode::Performance;
            else { fprintf(stderr, "v2Mode=quality|performance, not %s\n", eq + 1); return false; }
        }
        else if (k == "trustLow") s.trustLow = (float)v;
        else if (k == "trustHigh") s.trustHigh = (float)v;
        else if (k == "globalFallback") s.globalFallback = (float)v;
        else if (k == "sceneCutSad") s.sceneCutSad = (float)v;
        else if (k == "hudBonus") s.hudBonus = (float)v;
        else if (k == "smoothBonus") s.smoothBonus = (float)v;
        else if (k == "searchPenalty") s.searchPenalty = (float)v;
        else if (k == "subpixel") s.subpixel = v != 0;
        else if (k == "v2StaticMask") s.v2StaticMask = v != 0;
        else if (k == "learnedHalf") s.learnedHalf = v != 0;
        else if (k == "learnedZeroTest") s.learnedZeroTest = v != 0;
        else if (k == "learnedStaticGuard") s.learnedStaticGuard = v != 0;
        else if (k == "flowScale") s.flowScale = (uint32_t)v;
        else if (k == "ui") {   // Settings::uiProtection: off | crosshair | wholescreen | crosshairv2
            std::string a = eq + 1;
            if (a == "off") s.uiProtection = framegen::UiProtection::Off;
            else if (a == "crosshair") s.uiProtection = framegen::UiProtection::Crosshair;
            else if (a == "wholescreen") s.uiProtection = framegen::UiProtection::WholeScreen;
            else if (a == "crosshairv2") s.uiProtection = framegen::UiProtection::CrosshairV2;
            else { fprintf(stderr, "ui=off|crosshair|wholescreen|crosshairv2, not %s\n", eq + 1); return false; }
        }
        else if (k == "uiHistory") s.uiHistory = (uint32_t)v;
        else if (k == "uiBoxHeightFrac") s.uiBoxHeightFrac = (float)v;
        else if (k == "uiBoxWidth") s.uiBoxWidth = (uint32_t)v;
        else if (k == "uiBoxHeight") s.uiBoxHeight = (uint32_t)v;
        else if (k == "uiCrosshairArea") s.uiCrosshairArea = (float)v;
        else if (k == "uiBoxOffsetX") s.uiBoxOffsetX = (int32_t)v;
        else if (k == "uiBoxOffsetY") s.uiBoxOffsetY = (int32_t)v;
        else if (k == "uiStillFrames") s.uiStillFrames = (uint32_t)v;
        else if (k == "uiStillThreshold") s.uiStillThreshold = (float)v;
        else if (k == "uiInpaintReach") s.uiInpaintReach = (uint32_t)v;
        else if (k == "uiMaxStillFraction") s.uiMaxStillFraction = (float)v;
        else if (k == "uiContrastThreshold") s.uiContrastThreshold = (float)v;
        else if (k == "uiLeaveFrames") s.uiLeaveFrames = (uint32_t)v;
        else if (k == "uiReentryFrames") s.uiReentryFrames = (uint32_t)v;
        else if (k == "uiEstimateFill") s.uiEstimateFill = v != 0;
        else { fprintf(stderr, "unknown setting %s\n", tok); return false; }
    }
    return true;
}

static VkDevice dev;
static VkPhysicalDevice phys;
static VkPhysicalDeviceMemoryProperties memProps;

static uint32_t memType(uint32_t bits, VkMemoryPropertyFlags f) {
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & f) == f) return i;
    fprintf(stderr, "no memory type\n");
    exit(1);
}

// --format: the image format of the inputs and outputs. The files stay RGBA8; fgtest converts on upload and back on readback:
//   rgba8   R8G8B8A8_UNORM (default; --bgra: B8G8R8A8), the bytes as they are
//   rgb10   A2B10G10R10_UNORM_PACK32 holding PQ (HDR10): an 8-bit value v is taken as display light 1000 * (v / 255)^2.4 nits
//           (v >= 131 is brighter than SDR white, 203 nits), stored as its PQ code
//   f16     R16G16B16A16_SFLOAT, Transfer::Encoded: the same PQ code as a float
//   f16lin  R16G16B16A16_SFLOAT, Transfer::Linear (scRGB, 1.0 = 80 nits): the same light, nits / 80 (up to 12.5)
//   nv12    two planes R8 + R8G8 (separate images), BT.709 limited range, the 8-bit values as R'G'B' (SDR), chroma 2x2 averaged
//   p010    two planes R16 + R16G16, BT.2020 limited range PQ (HDR10 video), the light of rgb10
// Readback inverts the mapping and rounds to 8 bit, so outputs of every format compare with the rgba8 output of the same pair.
enum class Fmt { Rgba8, Rgb10, F16, F16Lin, Nv12, P010 };
static Fmt gFmt = Fmt::Rgba8;
static bool fmtYuv() { return gFmt == Fmt::Nv12 || gFmt == Fmt::P010; }
static VkFormat fmtPlane0(bool bgra) {
    switch (gFmt) {
    case Fmt::Rgb10: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case Fmt::F16: case Fmt::F16Lin: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case Fmt::Nv12: return VK_FORMAT_R8_UNORM;
    case Fmt::P010: return VK_FORMAT_R16_UNORM;
    default: return bgra ? VK_FORMAT_B8G8R8A8_UNORM : VK_FORMAT_R8G8B8A8_UNORM;
    }
}
static VkFormat fmtPlane1() { return gFmt == Fmt::P010 ? VK_FORMAT_R16G16_UNORM : VK_FORMAT_R8G8_UNORM; }
static size_t fmtBpp0() { return gFmt == Fmt::F16 || gFmt == Fmt::F16Lin ? 8 : gFmt == Fmt::Nv12 ? 1 : gFmt == Fmt::P010 ? 2 : 4; }
static size_t fmtBpp1() { return gFmt == Fmt::P010 ? 4 : 2; }

static uint16_t toHalf(float f) {   // round to nearest, ties up; denormals flushed (the values here are >= 0 and not tiny)
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int e = (int)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (e <= 0) return (uint16_t)sign;
    if (e >= 31) return (uint16_t)(sign | 0x7bffu);
    uint32_t h = sign | ((uint32_t)e << 10) | (man >> 13);
    if (man & 0x1000u) h++;
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
static const double kHdrPeak = 1000.0, kHdrGamma = 2.4;
static double pqEnc(double nits) {
    const double p = std::pow(std::min(std::max(nits, 0.0) / 10000.0, 1.0), 0.1593017578125);
    return std::pow((0.8359375 + 18.8515625 * p) / (1.0 + 18.6875 * p), 78.84375);
}
static double pqDec(double code) {
    const double e = std::pow(std::min(std::max(code, 0.0), 1.0), 1.0 / 78.84375);
    return 10000.0 * std::pow(std::max(e - 0.8359375, 0.0) / (18.8515625 - 18.6875 * e), 1.0 / 0.1593017578125);
}
static double nitsOf(int v) { return kHdrPeak * std::pow(v / 255.0, kHdrGamma); }
static uint8_t v8OfNits(double nits) { return (uint8_t)std::lround(std::min(std::max(255.0 * std::pow(std::max(nits, 0.0) / kHdrPeak, 1.0 / kHdrGamma), 0.0), 255.0)); }

// RGBA8 file bytes (W*H*4) -> plane 0 at dst, plane 1 at dst + off1 (YUV)
static void encodeFrame(const uint8_t* rgba, uint32_t W, uint32_t H, uint8_t* dst, size_t off1) {
    const size_t n = (size_t)W * H;
    static double pq[256];
    for (int v = 0; v < 256; v++) pq[v] = pqEnc(nitsOf(v));
    switch (gFmt) {
    case Fmt::Rgba8: memcpy(dst, rgba, n * 4); return;
    case Fmt::Rgb10:
        for (size_t i = 0; i < n; i++) {
            uint32_t w = 3u << 30;
            for (int c = 0; c < 3; c++) w |= (uint32_t)std::lround(pq[rgba[i * 4 + c]] * 1023.0) << (10 * c);
            memcpy(dst + i * 4, &w, 4);
        }
        return;
    case Fmt::F16: case Fmt::F16Lin: {
        uint16_t lut[256];
        for (int v = 0; v < 256; v++) lut[v] = toHalf((float)(gFmt == Fmt::F16 ? pq[v] : nitsOf(v) / 80.0));
        const uint16_t one = toHalf(1.0f);
        for (size_t i = 0; i < n; i++) {
            uint16_t h[4] = {lut[rgba[i * 4]], lut[rgba[i * 4 + 1]], lut[rgba[i * 4 + 2]], one};
            memcpy(dst + i * 8, h, 8);
        }
        return;
    }
    default: {
        const bool p010 = gFmt == Fmt::P010;
        const double kr = p010 ? 0.2627 : 0.2126, kb = p010 ? 0.0593 : 0.0722, kg = 1 - kr - kb;
        const double code = p010 ? 4.0 : 1.0;   // 8-bit level -> code
        const uint32_t cw = (W + 1) / 2, ch = (H + 1) / 2;
        std::vector<float> cb(n), cr(n);
        for (size_t i = 0; i < n; i++) {
            double c3[3];
            for (int c = 0; c < 3; c++) c3[c] = p010 ? pq[rgba[i * 4 + c]] : rgba[i * 4 + c] / 255.0;
            const double Y = kr * c3[0] + kg * c3[1] + kb * c3[2];
            cb[i] = (float)((c3[2] - Y) / (2 * (1 - kb)));
            cr[i] = (float)((c3[0] - Y) / (2 * (1 - kr)));
            const uint32_t y = (uint32_t)std::lround(std::min(std::max((16 + 219 * Y) * code, 0.0), 255.0 * code + (p010 ? 3 : 0)));
            if (p010) { uint16_t w = (uint16_t)(y << 6); memcpy(dst + i * 2, &w, 2); } else dst[i] = (uint8_t)y;
        }
        for (uint32_t y = 0; y < ch; y++)
            for (uint32_t x = 0; x < cw; x++) {
                double sb = 0, sr = 0;
                for (uint32_t j = 0; j < 2; j++)
                    for (uint32_t i = 0; i < 2; i++) {
                        const size_t k = (size_t)std::min(2 * y + j, H - 1) * W + std::min(2 * x + i, W - 1);
                        sb += cb[k]; sr += cr[k];
                    }
                const double mx = 255.0 * code + (p010 ? 3 : 0);
                const uint32_t b = (uint32_t)std::lround(std::min(std::max((128 + 224 * sb / 4) * code, 0.0), mx));
                const uint32_t r = (uint32_t)std::lround(std::min(std::max((128 + 224 * sr / 4) * code, 0.0), mx));
                const size_t o = (size_t)y * cw + x;
                if (p010) { uint16_t w[2] = {(uint16_t)(b << 6), (uint16_t)(r << 6)}; memcpy(dst + off1 + o * 4, w, 4); }
                else { dst[off1 + o * 2] = (uint8_t)b; dst[off1 + o * 2 + 1] = (uint8_t)r; }
            }
    }
    }
}
// ... and back: plane 0 at src, plane 1 at src + off1 -> RGBA8 (alpha 255)
static void decodeFrame(const uint8_t* src, size_t off1, uint32_t W, uint32_t H, uint8_t* rgba) {
    const size_t n = (size_t)W * H;
    switch (gFmt) {
    case Fmt::Rgba8: memcpy(rgba, src, n * 4); return;
    case Fmt::Rgb10: {
        uint8_t lut[1024];
        for (int c = 0; c < 1024; c++) lut[c] = v8OfNits(pqDec(c / 1023.0));
        for (size_t i = 0; i < n; i++) {
            uint32_t w;
            memcpy(&w, src + i * 4, 4);
            for (int c = 0; c < 3; c++) rgba[i * 4 + c] = lut[(w >> (10 * c)) & 1023u];
            rgba[i * 4 + 3] = 255;
        }
        return;
    }
    case Fmt::F16: case Fmt::F16Lin: {
        static std::vector<uint8_t> lut;
        lut.resize(65536);
        for (uint32_t h = 0; h < 65536; h++) {
            const double f = fromHalf((uint16_t)h);
            lut[h] = v8OfNits(gFmt == Fmt::F16 ? pqDec(f) : f * 80.0);
        }
        for (size_t i = 0; i < n; i++) {
            uint16_t h[4];
            memcpy(h, src + i * 8, 8);
            for (int c = 0; c < 3; c++) rgba[i * 4 + c] = lut[h[c]];
            rgba[i * 4 + 3] = 255;
        }
        return;
    }
    default: {
        const bool p010 = gFmt == Fmt::P010;
        const double kr = p010 ? 0.2627 : 0.2126, kb = p010 ? 0.0593 : 0.0722, kg = 1 - kr - kb;
        const uint32_t cw = (W + 1) / 2;
        std::vector<uint8_t> pqLut(4097);
        for (int k = 0; k <= 4096; k++) pqLut[k] = v8OfNits(pqDec(k / 4096.0));
        for (uint32_t y = 0; y < H; y++)
            for (uint32_t x = 0; x < W; x++) {
                const size_t i = (size_t)y * W + x, ci = (size_t)(y / 2) * cw + x / 2;
                double Y, b, r;
                if (p010) {
                    uint16_t w, c2[2];
                    memcpy(&w, src + i * 2, 2); memcpy(c2, src + off1 + ci * 4, 4);
                    Y = (w >> 6) / 4.0; b = (c2[0] >> 6) / 4.0; r = (c2[1] >> 6) / 4.0;
                } else {
                    Y = src[i]; b = src[off1 + ci * 2]; r = src[off1 + ci * 2 + 1];
                }
                const double yf = (Y - 16) / 219.0, cbf = (b - 128) / 224.0, crf = (r - 128) / 224.0;
                const double R = yf + 2 * (1 - kr) * crf, B = yf + 2 * (1 - kb) * cbf, G = (yf - kr * R - kb * B) / kg;
                const double c3[3] = {R, G, B};
                for (int c = 0; c < 3; c++)
                    rgba[i * 4 + c] = p010 ? pqLut[(int)std::lround(std::min(std::max(c3[c], 0.0), 1.0) * 4096.0)]
                                           : (uint8_t)std::lround(std::min(std::max(c3[c] * 255.0, 0.0), 255.0));
                rgba[i * 4 + 3] = 255;
            }
    }
    }
}

// One frame image; YUV (--format nv12 / p010): the Y plane, and the CbCr plane as a second image (imgC)
struct Img {
    VkImage img = VK_NULL_HANDLE, imgC = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE, memC = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE, viewC = VK_NULL_HANDLE;
    framegen::Frame fr() const { return framegen::Frame(view, viewC); }
};

static Img makeImage(uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage, bool chroma = true) {
    Img r{};
    if (chroma && fmtYuv()) {   // the CbCr plane: a second image at half resolution
        Img c = makeImage((w + 1) / 2, (h + 1) / 2, fmtPlane1(), usage, false);
        r.imgC = c.img; r.memC = c.mem; r.viewC = c.view;
    }
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = fmt;
    ici.extent = {w, h, 1};
    ici.mipLevels = ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    CK(vkCreateImage(dev, &ici, nullptr, &r.img));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev, r.img, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size,
                             memType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    CK(vkAllocateMemory(dev, &mai, nullptr, &r.mem));
    CK(vkBindImageMemory(dev, r.img, r.mem, 0));
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = r.img;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = fmt;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    CK(vkCreateImageView(dev, &vci, nullptr, &r.view));
    return r;
}

static void imgBarrier1(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkAccessFlags srcA, VkAccessFlags dstA) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = srcA;
    b.dstAccessMask = dstA;
    b.oldLayout = from;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
}
static void imgBarrier(VkCommandBuffer cb, const Img& i, VkImageLayout from, VkAccessFlags srcA, VkAccessFlags dstA) {
    imgBarrier1(cb, i.img, from, srcA, dstA);
    if (i.imgC) imgBarrier1(cb, i.imgC, from, srcA, dstA);
}
static void destroyImg(Img& i) {
    vkDestroyImageView(dev, i.view, nullptr);
    vkDestroyImage(dev, i.img, nullptr);
    vkFreeMemory(dev, i.mem, nullptr);
    if (i.imgC) { vkDestroyImageView(dev, i.viewC, nullptr); vkDestroyImage(dev, i.imgC, nullptr); vkFreeMemory(dev, i.memC, nullptr); }
}

int main(int argc, char** argv) {
    bool bgra = false;
    int bench = 0, mult = 2;
    bool split = false, stats = false;
    int window = 0;
    const char* dumpDir = nullptr;
    std::vector<float> ts{0.5f};
    framegen::Settings setA;
    char* thenArg = nullptr;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bgra")) bgra = true;
        else if (!strcmp(argv[i], "--format") && i + 1 < argc) {
            const std::string a = argv[++i];
            if (a == "rgba8") gFmt = Fmt::Rgba8;
            else if (a == "rgb10") gFmt = Fmt::Rgb10;
            else if (a == "f16") gFmt = Fmt::F16;
            else if (a == "f16lin") gFmt = Fmt::F16Lin;
            else if (a == "nv12") gFmt = Fmt::Nv12;
            else if (a == "p010") gFmt = Fmt::P010;
            else { fprintf(stderr, "--format rgba8|rgb10|f16|f16lin|nv12|p010\n"); return 2; }
        }
        else if (!strcmp(argv[i], "--bench") && i + 1 < argc) bench = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mult") && i + 1 < argc) mult = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--split")) split = true;
        else if (!strcmp(argv[i], "--stats")) stats = true;
        else if (!strcmp(argv[i], "--window") && i + 1 < argc) window = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dumpDir = argv[++i];
        else if (!strcmp(argv[i], "--set") && i + 1 < argc) { if (!parseSettings(argv[++i], setA)) return 2; }
        else if (!strcmp(argv[i], "--then") && i + 1 < argc) thenArg = argv[++i];
        else if (!strcmp(argv[i], "--t") && i + 1 < argc) {
            ts.clear();
            for (char* tok = strtok(argv[++i], ","); tok; tok = strtok(nullptr, ",")) ts.push_back((float)atof(tok));
        }
        else pos.push_back(argv[i]);
    }
    if (pos.size() < 5 || (pos.size() - 2) % 3) {
        fprintf(stderr, "usage: fgtest [--bgra] [--format F] [--split] [--t ..] [--set ..] [--then ..] [--stats] [--window N] [--dump DIR] [--bench N] W H prev.rgba curr.rgba out.rgba [...]\n");
        return 2;
    }
    framegen::Settings setB = setA;
    if (thenArg && !parseSettings(thenArg, setB)) return 2;
    if (thenArg) ts.resize(2, ts[0]);   // two outputs per pair
    uint32_t W = atoi(pos[0].c_str()), H = atoi(pos[1].c_str());
    size_t bytes = (size_t)W * H * 4;   // one RGBA8 file
    VkFormat fmt = fmtPlane0(bgra);
    const size_t off1 = ((size_t)W * H * fmtBpp0() + 255) & ~(size_t)255;   // staging: plane 0, then the CbCr plane (YUV)
    const size_t stgBytes = std::max(bytes, off1 + (fmtYuv() ? (size_t)((W + 1) / 2) * ((H + 1) / 2) * fmtBpp1() : 0));
    std::vector<uint8_t> fileBuf(gFmt == Fmt::Rgba8 ? 0 : bytes);

    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.pApplicationName = "fgtest";
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
    static const char* fmtNames[] = {"RGBA8", "Rgb10 (PQ)", "RgbaF16 (PQ)", "RgbaF16 (linear scRGB)", "Nv12 (BT.709)", "P010 (BT.2020 PQ)"};
    fprintf(stderr, "device: %s, queue family %u, format %s\n", props.deviceName, qfi, gFmt == Fmt::Rgba8 && bgra ? "BGRA8" : fmtNames[(int)gFmt]);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, qfi, 1, &prio};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    // no extra features for RGBA8, as in gamescope's minimum; the other formats store rgb10_a2 / r8 / rg8 / r16 / rg16 images
    VkPhysicalDeviceFeatures feats{};
    feats.shaderStorageImageExtendedFormats = VK_TRUE;
    if (gFmt != Fmt::Rgba8) dci.pEnabledFeatures = &feats;
    CK(vkCreateDevice(phys, &dci, nullptr, &dev));
    VkQueue queue;
    vkGetDeviceQueue(dev, qfi, 0, &queue);

    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
                                 VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qfi};
    VkCommandPool cpool;
    CK(vkCreateCommandPool(dev, &cpci, nullptr, &cpool));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, cpool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cb;
    CK(vkAllocateCommandBuffers(dev, &cbai, &cb));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    CK(vkCreateFence(dev, &fci, nullptr, &fence));
    framegen::Interpolator fg;
    auto begin = [&] {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CK(vkBeginCommandBuffer(cb, &bi));
        fg.beginFrame();   // one submission per command buffer
    };
    auto submit = [&] {
        CK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        CK(vkQueueSubmit(queue, 1, &si, fence));
        CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
        CK(vkResetFences(dev, 1, &fence));
    };

    // staging buffer
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = stgBytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer stg;
    CK(vkCreateBuffer(dev, &bci, nullptr, &stg));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, stg, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size,
                             memType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    VkDeviceMemory stgMem;
    CK(vkAllocateMemory(dev, &mai, nullptr, &stgMem));
    CK(vkBindBufferMemory(dev, stg, stgMem, 0));
    uint8_t* map;
    CK(vkMapMemory(dev, stgMem, 0, stgBytes, 0, (void**)&map));
    // --stats: 8 bytes of host-visible counters
    bci.size = 8;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer dbg;
    CK(vkCreateBuffer(dev, &bci, nullptr, &dbg));
    vkGetBufferMemoryRequirements(dev, dbg, &req);
    VkMemoryAllocateInfo dmai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size,
                              memType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    VkDeviceMemory dbgMem;
    CK(vkAllocateMemory(dev, &dmai, nullptr, &dbgMem));
    CK(vkBindBufferMemory(dev, dbg, dbgMem, 0));
    uint32_t* dbgMap;
    CK(vkMapMemory(dev, dbgMem, 0, 8, 0, (void**)&dbgMap));

    Img prev = makeImage(W, H, fmt, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    Img curr = makeImage(W, H, fmt, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    // UI protection: the input images become a ring of 3 real frames (the library keys its observed frames by view), see the loop
    const bool uiOn = setA.uiProtection != framegen::UiProtection::Off || setB.uiProtection != framegen::UiProtection::Off;
    std::vector<Img*> inputs{&prev, &curr};
    Img spare{};
    if (uiOn) { spare = makeImage(W, H, fmt, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT); inputs.push_back(&spare); }
    size_t nOut = std::max(ts.size(), (size_t)std::max(mult - 1, 1));
    std::vector<Img> outs;
    for (size_t k = 0; k < nOut; k++)
        outs.push_back(makeImage(W, H, fmt, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT));
    Img& out = outs[0];
    begin();
    for (Img* i : inputs) imgBarrier(cb, *i, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    for (Img& o : outs) imgBarrier(cb, o, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    submit();

    framegen::InitInfo info;
    info.phys = phys;
    info.device = dev;
    info.gdpa = vkGetDeviceProcAddr;
    info.width = W;
    info.height = H;
    info.viewFormat = fmt;
    info.getMemProps = vkGetPhysicalDeviceMemoryProperties; info.getProps2 = vkGetPhysicalDeviceProperties2;
    info.settings = setA;
    if (gFmt != Fmt::Rgba8) info.getFormatProps = vkGetPhysicalDeviceFormatProperties;
    if (gFmt == Fmt::F16Lin) { info.transfer = framegen::Transfer::Linear; info.linearWhiteNits = 80.0f; }
    if (gFmt == Fmt::P010) info.yuvMatrix = framegen::YuvMatrix::Bt2020;
    if (!fg.init(info)) { fprintf(stderr, "fg.init failed (invalid --set?): %s\n", fg.initError() ? fg.initError() : "?"); return 1; }
    // --dump: host-visible copy of the debug maps
    framegen::Interpolator::DebugMapLayout dl{};
    VkBuffer dmp = VK_NULL_HANDLE;
    VkDeviceMemory dmpMem = VK_NULL_HANDLE;
    uint8_t* dmpMap = nullptr;
    if (dumpDir) {
        std::error_code ec;
        std::filesystem::create_directories(dumpDir, ec);   // mkdir -p
        if (ec) { fprintf(stderr, "cannot create %s: %s\n", dumpDir, ec.message().c_str()); return 1; }
        if (!fg.debugMapLayout(dl)) { fprintf(stderr, "debugMapLayout failed\n"); return 1; }
        VkBufferCreateInfo dbci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        dbci.size = dl.size;
        dbci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        CK(vkCreateBuffer(dev, &dbci, nullptr, &dmp));
        VkMemoryRequirements dreq;
        vkGetBufferMemoryRequirements(dev, dmp, &dreq);
        VkMemoryAllocateInfo dma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, dreq.size,
                                 memType(dreq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
        CK(vkAllocateMemory(dev, &dma, nullptr, &dmpMem));
        CK(vkBindBufferMemory(dev, dmp, dmpMem, 0));
        CK(vkMapMemory(dev, dmpMem, 0, dl.size, 0, (void**)&dmpMap));
    }

    VkBufferImageCopy region{}, regionC{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {W, H, 1};
    regionC.bufferOffset = off1;
    regionC.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    regionC.imageExtent = {(W + 1) / 2, (H + 1) / 2, 1};
    auto swz = [&](uint8_t* p) { if (bgra && gFmt == Fmt::Rgba8) for (size_t i = 0; i < bytes; i += 4) std::swap(p[i], p[i + 2]); };
    // observe: with UI protection on, recordObserve the frame in the upload's command buffer, right after its pixels are final
    auto upload = [&](const std::string& path, Img& img, bool observe = false) {
        FILE* f = fopen(path.c_str(), "rb");
        uint8_t* dst = gFmt == Fmt::Rgba8 ? map : fileBuf.data();
        if (!f || fread(dst, 1, bytes, f) != bytes) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(1); }
        fclose(f);
        swz(map);
        if (gFmt != Fmt::Rgba8) encodeFrame(fileBuf.data(), W, H, map, off1);
        begin();
        imgBarrier(cb, img, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        vkCmdCopyBufferToImage(cb, stg, img.img, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        if (img.imgC) vkCmdCopyBufferToImage(cb, stg, img.imgC, VK_IMAGE_LAYOUT_GENERAL, 1, &regionC);
        imgBarrier(cb, img, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        if (observe && !fg.recordObserve(cb, img.fr())) { fprintf(stderr, "fg.recordObserve failed\n"); exit(1); }
        submit();
    };

    // UI protection on (--set/--then ui=..): fgtest drives it like a host. Every real frame is uploaded into a ring of 3 input
    // images and recordObserve'd ONCE when it arrives, in pair order: a pair whose prev is the previous pair's curr (the same
    // path string) reuses that image and its observation, so a run of consecutive pairs is one stream of real frames with
    // the stillness counters (and CrosshairV2's hysteresis / pair gate) advancing one frame per pair. A pair that does not
    // continue the previous one observes its prev and then its curr (the counters bridge the gap, as for a host that skipped
    // frames). --window re-init starts a fresh stream (as after a host resize): UI protection warms up again.
    Img* P = &prev;
    Img* C = &curr;
    int lastSlot = -1;          // ring slot of the previous pair's curr, -1 = none (start, after a --window re-init)
    std::string lastPath;

    for (size_t p = 2; p + 2 < pos.size(); p += 3) {
        size_t pairIdx = (p - 2) / 3;
        if (window > 0 && pairIdx > 0 && pairIdx % window == 0) {   // a fresh Interpolator: nothing carried over
            vkDeviceWaitIdle(dev);
            if (!fg.init(info)) { fprintf(stderr, "fg.init (--window) failed\n"); return 1; }
            lastSlot = -1;
        }
        if (!uiOn) {
            upload(pos[p], prev);
            upload(pos[p + 1], curr);
        } else {
            int ps = lastSlot;
            if (ps < 0 || pos[p] != lastPath) {   // not a continuation: prev is a new real frame
                ps = (lastSlot + 1) % 3;
                upload(pos[p], *inputs[ps], true);
            }
            int cs = 0;
            while (cs == ps || cs == lastSlot) cs++;
            upload(pos[p + 1], *inputs[cs], true);
            P = inputs[ps];
            C = inputs[cs];
            lastSlot = cs;
            lastPath = pos[p + 1];
        }
        Img& prev = *P;
        Img& curr = *C;
        begin();
        if (thenArg) {
            if (!fg.record(cb, prev.fr(), curr.fr(), outs[0].fr(), ts[0])) { fprintf(stderr, "fg.record failed\n"); return 1; }
            if (!fg.setSettings(setB)) { fprintf(stderr, "setSettings failed (invalid or structural --then?)\n"); return 1; }
            if (!fg.record(cb, prev.fr(), curr.fr(), outs[1].fr(), ts[0])) { fprintf(stderr, "fg.record failed\n"); return 1; }
            fg.setSettings(setA);   // the next pair starts from A again
        } else if (ts.size() == 1 && !split) {
            if (!fg.record(cb, prev.fr(), curr.fr(), out.fr(), ts[0])) { fprintf(stderr, "fg.record failed\n"); return 1; }
        } else {
            if (!fg.recordEstimate(cb, prev.fr(), curr.fr())) { fprintf(stderr, "fg.recordEstimate failed\n"); return 1; }
            for (size_t k = 0; k < ts.size(); k++)
                if (!fg.recordSynth(cb, prev.fr(), curr.fr(), outs[k].fr(), ts[k])) { fprintf(stderr, "fg.recordSynth failed\n"); return 1; }
        }
        if (stats) {
            if (!fg.recordDebugCounters(cb, dbg, 0)) { fprintf(stderr, "recordDebugCounters failed\n"); return 1; }
            VkMemoryBarrier hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, nullptr, 0, nullptr);
        }
        if (dumpDir) {
            if (!fg.recordDebugMaps(cb, dmp, 0)) { fprintf(stderr, "recordDebugMaps failed\n"); return 1; }
            VkMemoryBarrier hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, nullptr, 0, nullptr);
        }
        submit();
        if (dumpDir) {
            char path[4096];
            snprintf(path, sizeof path, "%s/pair_%04zu.fgmaps", dumpDir, pairIdx);
            FILE* f = fopen(path, "wb");
            if (!f) { fprintf(stderr, "cannot write %s\n", path); return 1; }
            uint32_t hdr[16]{0x504d4746u, 5, dl.W, dl.H, dl.fw, dl.fh, dl.bw, dl.bh, dl.fscale, dl.eStride};
            float hf[4]{setB.trustLow, setB.trustHigh, setB.smoothBonus, setB.hudBonus};   // the settings of the last record
            memcpy(hdr + 10, hf, sizeof hf);
            hdr[14] = (uint32_t)setB.motionVersion; hdr[15] = (uint32_t)setB.v2Mode;
            fwrite(hdr, 1, sizeof hdr, f);
            fwrite(dmpMap, 1, dl.size, f);
            fclose(f);
        }
        if (stats) printf("STATS %zu %u %u %u %zu\n", pairIdx, dbgMap[0], dbgMap[1], fg.debugSadBlocks(), (size_t)W * H);
        for (size_t k = 0; k < ts.size(); k++) {
            begin();
            imgBarrier(cb, outs[k], VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            vkCmdCopyImageToBuffer(cb, outs[k].img, VK_IMAGE_LAYOUT_GENERAL, stg, 1, &region);
            if (outs[k].imgC) vkCmdCopyImageToBuffer(cb, outs[k].imgC, VK_IMAGE_LAYOUT_GENERAL, stg, 1, &regionC);
            submit();
            swz(map);
            if (gFmt != Fmt::Rgba8) decodeFrame(map, off1, W, H, fileBuf.data());
            std::string path = pos[p + 2] + (ts.size() > 1 ? "." + std::to_string(k) : "");
            FILE* f = fopen(path.c_str(), "wb");
            fwrite(gFmt == Fmt::Rgba8 ? map : fileBuf.data(), 1, bytes, f);
            fclose(f);
        }
    }

    // one real pair: 2x = record(0.5); M x = recordEstimate + M-1 recordSynth at k/M into distinct outputs
    auto pair = [&]() -> bool {   // the last pair's frames (with UI protection: observed)
        if (mult <= 2) return fg.record(cb, P->fr(), C->fr(), out.fr());
        if (!fg.recordEstimate(cb, P->fr(), C->fr())) return false;
        for (int k = 1; k < mult; k++)
            if (!fg.recordSynth(cb, P->fr(), C->fr(), outs[k - 1].fr(), (float)k / mult)) return false;
        return true;
    };
    if (bench > 0) {
        double tp = props.limits.timestampPeriod * 1e-6;   // ticks -> ms
        if (!fg.setProfiling(true)) { fprintf(stderr, "setProfiling failed\n"); return 1; }
        std::vector<const char*> names;
        std::vector<std::vector<double>> per;
        std::vector<double> total;
        // Each run: 8 unprofiled records to bring the GPU clocks up (as a running game would), then the
        // profiled one, all in one command buffer. Isolated single submits mostly measure clock ramp-up.
        for (int it = 0; it < bench + 5; it++) {
            begin();
            fg.setProfiling(false);
            for (int k = 0; k < 8; k++) pair();
            fg.setProfiling(true);
            if (!pair()) { fprintf(stderr, "fg.record failed\n"); return 1; }
            submit();
            std::vector<uint64_t> ticks;
            if (!fg.readTimings(names, ticks)) { fprintf(stderr, "readTimings failed\n"); return 1; }
            if (it < 5) continue;   // warm-up
            per.resize(ticks.size());
            double sum = 0;
            for (size_t i = 0; i < ticks.size(); i++) { per[i].push_back(ticks[i] * tp); sum += ticks[i] * tp; }
            total.push_back(sum);
        }
        auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        auto mn = [](const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); };
        printf("timing %ux%u %dx, %d runs (profiled pair after 8 warm pairs), ms: median (min)\n", W, H, mult, bench);
        for (size_t i = 0; i < names.size(); i++) printf("  %-18s %8.4f (%.4f)\n", names[i], med(per[i]), mn(per[i]));
        printf("  %-18s %8.4f (%.4f)\n", "TOTAL per pair", med(total), mn(total));

        // Back-to-back: 'bench' records in one command buffer, GPU kept busy -> steady clocks.
        fg.setProfiling(false);
        VkQueryPoolCreateInfo qp{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qp.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qp.queryCount = 2;
        VkQueryPool pool;
        CK(vkCreateQueryPool(dev, &qp, nullptr, &pool));
        int batch = std::min(bench, (int)(framegen::Interpolator::kSetRing - 1) / std::max(mult - 1, 1));
        std::vector<double> per1;
        for (int rep = 0; rep < 5; rep++) {
            begin();
            vkCmdResetQueryPool(cb, pool, 0, 2);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool, 0);
            for (int i = 0; i < batch; i++)
                if (!pair()) { fprintf(stderr, "fg.record failed\n"); return 1; }
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, 1);
            submit();
            uint64_t ts[2];
            CK(vkGetQueryPoolResults(dev, pool, 0, 2, sizeof(ts), ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            per1.push_back((ts[1] - ts[0]) * tp / batch);
        }
        printf("  back-to-back (%d pairs per command buffer, 5 reps): per pair median %.4f (min %.4f) ms\n", batch, med(per1), mn(per1));
        vkDestroyQueryPool(dev, pool, nullptr);
    }

    vkDeviceWaitIdle(dev);
    fg.destroy();
    for (Img& o : outs) destroyImg(o);
    for (Img* i : inputs) destroyImg(*i);
    vkDestroyBuffer(dev, stg, nullptr);
    vkDestroyBuffer(dev, dbg, nullptr);
    if (dmp) { vkDestroyBuffer(dev, dmp, nullptr); vkFreeMemory(dev, dmpMem, nullptr); }
    vkFreeMemory(dev, dbgMem, nullptr);
    vkFreeMemory(dev, stgMem, nullptr);
    vkDestroyFence(dev, fence, nullptr);
    vkDestroyCommandPool(dev, cpool, nullptr);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    return 0;
}
