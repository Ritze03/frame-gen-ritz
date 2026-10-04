// uitest: validation + benchmark of framegen's static-UI protection (Settings::uiProtection).
//   uitest check [--bgra] [--hash] [W H]     default 1280 720
//   uitest deep [W H]                        Settings::uiHistory: a synth from a pair 10 frames back still finds its clean copies (default 640x360)
//   uitest bench [W H]                       per-pass GPU times, moving and still scene (default 1920x1080 and 2560x1440)
// check: a synthetic sequence of 12 frames: a textured background panning 14 px right / 6 px down per frame with 16 independently moving objects (some crossing the crosshair and the HUD), a static opaque crosshair
// (plus-shaped arms, centre dot, 1 px black outline) and a static opaque HUD box in a corner. The last pair
// (frames 10, 11) is interpolated at t = 0.5 and 0.25 with protection Off / Crosshair / WholeScreen (explicit
// observes; Crosshair and WholeScreen also with implicit observes), and with an explicit and an offset box.
// Asserts: the crosshair pixels of the generated frame equal the real ones (<= 1 level) in Crosshair and
// WholeScreen, the HUD box's only in WholeScreen, background PSNR does not drop; --hash prints an FNV-1a hash of
// every generated frame of the Off run (compare against a build of the library before protection existed: build
// this file with -DUITEST_NO_UI against it).
#include "framegen.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

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

// ---------------------------------------------------------------- one run
using namespace framegen;
struct Result { std::vector<uint8_t> out[2]; };   // t = 0.5, 0.25 of the last pair
static uint64_t fnv(const std::vector<uint8_t>& v) {
    uint64_t h = 1469598103934665603ull;
    for (uint8_t b : v) { h ^= b; h *= 1099511628211ull; }
    return h;
}

static bool makeInterp(Interpolator& fg, int W, int H, const Settings& s) {
    InitInfo info;
    info.phys = phys; info.device = dev; info.gdpa = vkGetDeviceProcAddr; info.width = W; info.height = H;
    info.viewFormat = fmt(); info.getMemProps = vkGetPhysicalDeviceMemoryProperties; info.settings = s;
    return fg.init(info);
}

// Runs the 12-frame sequence through a ring of 4 views; observes explicitly or implicitly. hashes (Off check).
static Result runSequence(const Scene& sc, const Settings& s, bool explicitObserve, std::vector<uint64_t>* hashes,
                          const std::function<void(Interpolator&, int)>& hook = nullptr) {
    Interpolator fg;
    if (!makeInterp(fg, sc.W, sc.H, s)) { fprintf(stderr, "init failed\n"); exit(1); }
    Img ring[4];
    for (auto& r : ring) r = makeImage(sc.W, sc.H, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    Img outs[2];
    for (auto& o : outs) o = makeImage(sc.W, sc.H, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    begin();
    for (auto& r : ring) imgBarrier(r.img, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    for (auto& o : outs) imgBarrier(o.img, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    submit();
    Result res;
    for (int f = 0; f < sc.NF; f++) {
        Img& cur = ring[f % 4];
        upload(sc.frame(f), cur, sc.W, sc.H);
        if (hook) hook(fg, f);   // runtime settings changes between submissions
        begin();
        fg.beginFrame();
#ifndef UITEST_NO_UI
        if (explicitObserve && !fg.recordObserve(cb, cur.view)) { fprintf(stderr, "observe failed\n"); exit(1); }
#endif
        if (f > 0) {
            Img& prv = ring[(f - 1) % 4];
            bool last = f == sc.NF - 1;
            if (!fg.recordEstimate(cb, prv.view, cur.view)) { fprintf(stderr, "estimate failed\n"); exit(1); }
            if (!fg.recordSynth(cb, prv.view, cur.view, outs[0].view, 0.5f)) { fprintf(stderr, "synth failed\n"); exit(1); }
            if (last && !fg.recordSynth(cb, prv.view, cur.view, outs[1].view, 0.25f)) { fprintf(stderr, "synth failed\n"); exit(1); }
        }
        submit();
        if (f > 0) {
            auto o = download(outs[0], sc.W, sc.H);
            if (hashes) hashes->push_back(fnv(o));
            if (f == sc.NF - 1) { res.out[0] = o; res.out[1] = download(outs[1], sc.W, sc.H); }
        }
    }
    vkDeviceWaitIdle(dev);
    fg.destroy();
    for (auto& r : ring) { vkDestroyImageView(dev, r.view, nullptr); vkDestroyImage(dev, r.img, nullptr); vkFreeMemory(dev, r.mem, nullptr); }
    for (auto& o : outs) { vkDestroyImageView(dev, o.view, nullptr); vkDestroyImage(dev, o.img, nullptr); vkFreeMemory(dev, o.mem, nullptr); }
    return res;
}

struct Metrics { size_t xhBad = 0, xhN = 0, hudBad = 0, hudN = 0; double psnrBg = 0, psnrAll = 0; };
// t = 0.5: truth = pan (10 + 0.5) steps; t = 0.25: truth would be at a non-integer pan: only UI checks there.
static Metrics measure(const Scene& sc, const std::vector<uint8_t>& out, const std::vector<uint8_t>& real, const std::vector<uint8_t>* truth) {
    Metrics m;
    int W = sc.W, H = sc.H;
    std::vector<uint8_t> near((size_t)W * H, 0);   // within 4 px of UI
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            if (sc.ui[(size_t)y * W + x])
                for (int dy = -4; dy <= 4; dy++)
                    for (int dx = -4; dx <= 4; dx++) {
                        int X = x + dx, Y = y + dy;
                        if (X >= 0 && Y >= 0 && X < W && Y < H) near[(size_t)Y * W + X] = 1;
                    }
    double seBg = 0, seAll = 0;
    size_t nBg = 0;
    for (size_t i = 0; i < (size_t)W * H; i++) {
        int k = sc.ui[i];
        if (k) {
            int d = 0;
            for (int c = 0; c < 3; c++) d = std::max(d, std::abs((int)out[i * 4 + c] - (int)real[i * 4 + c]));
            if (k == 1) { m.xhN++; m.xhBad += d > 1; } else { m.hudN++; m.hudBad += d > 1; }
        }
        if (truth) {
            for (int c = 0; c < 3; c++) {
                double d = (double)out[i * 4 + c] - (*truth)[i * 4 + c];
                seAll += d * d;
                if (!near[i]) { seBg += d * d; if (c == 0) nBg++; }
            }
        }
    }
    if (truth) {
        m.psnrAll = 10 * std::log10(255.0 * 255.0 / (seAll / ((double)W * H * 3)));
        m.psnrBg = 10 * std::log10(255.0 * 255.0 / (seBg / (nBg * 3.0)));
    }
    return m;
}

static int fails = 0;
static void expect(bool ok, const char* what) {
    printf("  %s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) fails++;
}

static int check(int W, int H, bool hash) {
    Scene sc(W, H, 12, 0, 0);
    auto real11 = sc.frame(11);
    auto truth = sc.frame(10.5);
    printf("check %dx%d %s: crosshair %zu px, hud box %zu px (centre %d,%d)\n", W, H, bgra ? "BGRA8" : "RGBA8",
           (size_t)std::count(sc.ui.begin(), sc.ui.end(), 1), (size_t)std::count(sc.ui.begin(), sc.ui.end(), 2), sc.cx, sc.cy);
    Settings off;
    std::vector<uint64_t> hOff;
    Result rOff = runSequence(sc, off, true, &hOff);
    Metrics mOff = measure(sc, rOff.out[0], real11, &truth);
    printf(" Off:          crosshair wrong %zu/%zu, hud wrong %zu/%zu, PSNR bg %.2f dB all %.2f dB\n", mOff.xhBad, mOff.xhN, mOff.hudBad, mOff.hudN, mOff.psnrBg, mOff.psnrAll);
    if (hash) { printf("HASHES"); for (uint64_t h : hOff) printf(" %016llx", (unsigned long long)h); printf("\n"); }
#ifdef UITEST_NO_UI
    return 0;
#else
    auto run = [&](const char* name, Settings s, bool expl, bool wantXh, bool wantHud, bool wantFull = true) {
        Scene* use = &sc;
        Result r = runSequence(*use, s, expl, nullptr);
        Metrics m = measure(sc, r.out[0], real11, &truth);
        Metrics m2 = measure(sc, r.out[1], real11, nullptr);
        printf(" %-13s crosshair wrong %zu/%zu, hud wrong %zu/%zu, PSNR bg %.2f dB all %.2f dB  (t=0.25: crosshair wrong %zu, hud wrong %zu)\n",
               name, m.xhBad, m.xhN, m.hudBad, m.hudN, m.psnrBg, m.psnrAll, m2.xhBad, m2.hudBad);
        if (wantXh) expect(m.xhBad == 0 && m2.xhBad == 0, "crosshair pixels equal the real ones (<= 1 level), t = 0.5 and 0.25");
        if (wantHud) expect(m.hudBad == 0 && m2.hudBad == 0, "HUD box pixels equal the real ones (<= 1 level), t = 0.5 and 0.25");
        if (wantFull) expect(m.psnrBg >= mOff.psnrBg - 0.3, "background PSNR within 0.3 dB of Off (or better)");
        return r;
    };
    Settings c1; c1.uiProtection = UiProtection::Crosshair;
    Settings w1; w1.uiProtection = UiProtection::WholeScreen;
    Result rc = run("Crosshair", c1, true, true, false);
    Result rw = run("WholeScreen", w1, true, true, true);
    Result rci = run("Crosshair(i)", c1, false, true, false);
    Result rwi = run("WholeScreen(i)", w1, false, true, true);
    expect(rc.out[0] == rci.out[0] && rw.out[0] == rwi.out[0], "implicit observe == explicit observe (bit-exact)");
    // explicit pixel box (box 80x60 around the crosshair), and an explicit box that is too small to cover the crosshair
    Settings c2 = c1; c2.uiBoxWidth = 80; c2.uiBoxHeight = 60;
    run("Crosshair box80x60", c2, true, true, false);
    Settings c3 = c1; c3.uiBoxWidth = 20; c3.uiBoxHeight = 20;
    printf(" (a 20x20 box is smaller than the 35x35 px crosshair: its outer ends are NOT protected, expected)\n");
    run("Crosshair box20x20", c3, false, false, false, false);
    // offset box: crosshair at an offset from the centre
    Scene so(W, H, 12, 37, -23);
    auto realO = so.frame(11);
    auto truthO = so.frame(10.5);
    {
        Settings c4 = c1; c4.uiBoxWidth = 120; c4.uiBoxHeight = 120; c4.uiBoxOffsetX = 37; c4.uiBoxOffsetY = -23;
        Result r4 = runSequence(so, c4, true, nullptr);
        Metrics m4 = measure(so, r4.out[0], realO, &truthO), m4b = measure(so, r4.out[1], realO, nullptr);
        printf(" Offset crosshair (+37,-23), box 120x120 offset (+37,-23): crosshair wrong %zu/%zu (t=0.25: %zu), PSNR bg %.2f dB\n", m4.xhBad, m4.xhN, m4b.xhBad, m4.psnrBg);
        expect(m4.xhBad == 0 && m4b.xhBad == 0, "offset box protects the off-centre crosshair exactly");
        Settings c5 = c1;   // default centred box on an off-centre crosshair (37,-23 is inside the 3% box): still covered
        Settings c6 = c1; c6.uiBoxWidth = 60; c6.uiBoxHeight = 60;   // centred 60x60: the crosshair at (+37,-23) lies outside it
        Result r6 = runSequence(so, c6, true, nullptr);
        Metrics m6 = measure(so, r6.out[0], realO, &truthO);
        printf(" Offset crosshair, centred 60x60 box (no offset, misses it): crosshair wrong %zu/%zu (informational)\n", m6.xhBad, m6.xhN);
        (void)c5;
        Settings c7 = w1; c7.uiBoxWidth = 120; c7.uiBoxHeight = 120; c7.uiBoxOffsetX = 37; c7.uiBoxOffsetY = -23;
        Result r7 = runSequence(so, c7, true, nullptr);
        Metrics m7 = measure(so, r7.out[0], realO, &truthO);
        printf(" Offset, WholeScreen + offset box: crosshair wrong %zu, hud wrong %zu, PSNR bg %.2f dB\n", m7.xhBad, m7.hudBad, m7.psnrBg);
        expect(m7.xhBad == 0 && m7.hudBad == 0, "WholeScreen + offset box: crosshair and HUD exact");
    }
    // runtime switching: Off -> Crosshair -> WholeScreen mid-stream, and WholeScreen -> Off
    {
        Result r = runSequence(sc, off, true, nullptr, [&](Interpolator& fg, int f) {
            Settings x;
            if (f == 1) { x.uiProtection = UiProtection::Crosshair; if (!fg.setSettings(x)) { printf("setSettings failed\n"); exit(1); } }
            if (f == 3) { x.uiProtection = UiProtection::WholeScreen; if (!fg.setSettings(x)) { printf("setSettings failed\n"); exit(1); } }
        });
        Metrics m = measure(sc, r.out[0], real11, &truth);
        printf(" Runtime Off -> Crosshair (frame 1) -> WholeScreen (frame 3): crosshair wrong %zu, hud wrong %zu\n", m.xhBad, m.hudBad);
        expect(m.xhBad == 0 && m.hudBad == 0, "runtime switch to WholeScreen mid-stream protects crosshair and HUD");
        Result r2 = runSequence(sc, w1, true, nullptr, [&](Interpolator& fg, int f) {
            if (f == 9) { if (!fg.setSettings(Settings{})) { printf("setSettings failed\n"); exit(1); } }
        });
        expect(r2.out[0] == rOff.out[0] && r2.out[1] == rOff.out[1], "runtime switch WholeScreen -> Off: later output bit-identical to an Off run");
    }
    // settings validation
    {
        Settings bad;
        bad.uiBoxWidth = 20000; expect(!Interpolator::validate(bad), "validate rejects uiBoxWidth 20000");
        bad = Settings{}; bad.uiBoxOffsetX = -20000; expect(!Interpolator::validate(bad), "validate rejects uiBoxOffsetX -20000");
        bad = Settings{}; bad.uiCrosshairArea = 0.0f; expect(!Interpolator::validate(bad), "validate rejects uiCrosshairArea 0");
        bad = Settings{}; bad.uiInpaintReach = 17; expect(!Interpolator::validate(bad), "validate rejects uiInpaintReach 17");
        bad = Settings{}; bad.uiStillFrames = 0; expect(!Interpolator::validate(bad), "validate rejects uiStillFrames 0");
        bad = Settings{}; bad.uiProtection = (UiProtection)3; expect(!Interpolator::validate(bad), "validate rejects an unknown mode");
        expect(Interpolator::validate(Settings{}), "defaults valid");
    }
    return fails ? 1 : 0;
#endif
}

// ---------------------------------------------------------------- deep history (Settings::uiHistory)
#ifndef UITEST_NO_UI
// 24 real frames observed into a ring of 24 views, then ONE pair 10 frames behind the newest (13, 14) is estimated and
// synthesised -- what a lag-spike buffer's pacer does. With uiHistory >= 14 the pair still has its clean copies and
// is protected; with the default 3 (or 10: frame 13 just evicted) it runs unprotected, bit-identical to Off.
// Afterwards: lower to 3 + releaseUnusedUi(), the newest pair must still be protected, and a raise works again.
static int deepHistory(int W, int H) {
    const int NF = 24, A = 13, B = 14;
    Scene sc(W, H, NF, 0, 0);
    auto realB = sc.frame(B);
    printf("deep history %dx%d: pair (%d, %d): its older frame is %d frames behind the newest (%d)\n", W, H, A, B, NF - 1 - A, NF - 1);
    auto run = [&](const char* name, Settings s, bool tail, bool* tailOk) {
        Interpolator fg;
        if (!makeInterp(fg, W, H, s)) { fprintf(stderr, "init failed\n"); exit(1); }
        std::vector<Img> ring(NF);
        for (auto& r : ring) r = makeImage(W, H, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        Img out = makeImage(W, H, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        begin();
        for (auto& r : ring) imgBarrier(r.img, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
        imgBarrier(out.img, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
        submit();
        for (int f = 0; f < NF; f++) {
            upload(sc.frame(f), ring[f], W, H);
            begin();
            fg.beginFrame();
            if (!fg.recordObserve(cb, ring[f].view)) { fprintf(stderr, "observe failed\n"); exit(1); }
            submit();
        }
        begin();
        fg.beginFrame();
        if (!fg.recordEstimate(cb, ring[A].view, ring[B].view) || !fg.recordSynth(cb, ring[A].view, ring[B].view, out.view, 0.5f)) { fprintf(stderr, "deep synth failed\n"); exit(1); }
        submit();
        std::vector<uint8_t> res = download(out, W, H);
        if (tail) {
            // lower to the default: the pair is evicted; release the memory (everything retired: waited above)
            Settings lo = s; lo.uiHistory = 3;
            if (!fg.setSettings(lo)) { printf("setSettings(lower) failed\n"); exit(1); }
            vkDeviceWaitIdle(dev);
            fg.releaseUnusedUi();
            // the newest pair is still protected
            begin();
            fg.beginFrame();
            if (!fg.recordEstimate(cb, ring[NF - 2].view, ring[NF - 1].view) || !fg.recordSynth(cb, ring[NF - 2].view, ring[NF - 1].view, out.view, 0.5f)) { fprintf(stderr, "tail synth failed\n"); exit(1); }
            submit();
            auto r2 = download(out, W, H);
            Metrics m = measure(sc, r2, sc.frame(NF - 1), nullptr);
            *tailOk = m.xhBad == 0;
            // and a raise allocates again (the settings call must succeed)
            if (!fg.setSettings(s)) { printf("setSettings(raise) failed\n"); exit(1); }
        }
        vkDeviceWaitIdle(dev);
        fg.destroy();
        for (auto& r : ring) { vkDestroyImageView(dev, r.view, nullptr); vkDestroyImage(dev, r.img, nullptr); vkFreeMemory(dev, r.mem, nullptr); }
        vkDestroyImageView(dev, out.view, nullptr); vkDestroyImage(dev, out.img, nullptr); vkFreeMemory(dev, out.mem, nullptr);
        Metrics m = measure(sc, res, realB, nullptr);
        printf(" %-22s crosshair wrong %zu/%zu, hud wrong %zu/%zu\n", name, m.xhBad, m.xhN, m.hudBad, m.hudN);
        return res;
    };
    for (UiProtection mode : {UiProtection::Crosshair, UiProtection::WholeScreen}) {
        const char* nm = mode == UiProtection::Crosshair ? "Crosshair" : "WholeScreen";
        Settings off, s3, s10, s14;
        s3.uiProtection = s10.uiProtection = s14.uiProtection = mode;
        s10.uiHistory = 10; s14.uiHistory = 14;
        printf("%s:\n", nm);
        auto rOff = run("Off", off, false, nullptr);
        auto r3 = run("uiHistory 3 (default)", s3, false, nullptr);
        auto r10 = run("uiHistory 10", s10, false, nullptr);
        bool tailOk = false;
        auto r14 = run("uiHistory 14", s14, true, &tailOk);
        Metrics m14 = measure(sc, r14, realB, nullptr);
        expect(r3 == rOff, "default history: a pair 10 frames back is unprotected (frame evicted), identical to Off");
        expect(r10 == rOff, "uiHistory 10: frame 13 is just evicted, unprotected, identical to Off");
        expect(r14 != rOff, "uiHistory 14: the pair 10 frames back still has its clean copies (output differs from Off)");
        expect(m14.xhBad == 0, "uiHistory 14: crosshair exact in a synth from the pair 10 frames back");
        if (mode == UiProtection::WholeScreen) expect(m14.hudBad == 0, "uiHistory 14: HUD box exact in a synth from the pair 10 frames back");
        expect(tailOk, "lowered to 3 and released: the newest pair is still protected; raising again works");
    }
    Settings bad; bad.uiHistory = 2; expect(!Interpolator::validate(bad), "validate rejects uiHistory 2");
    bad.uiHistory = Interpolator::kUiFramesMax + 1; expect(!Interpolator::validate(bad), "validate rejects uiHistory above kUiFramesMax");
    bad.uiHistory = Interpolator::kUiFramesMax; expect(Interpolator::validate(bad), "validate accepts uiHistory = kUiFramesMax");
    return fails ? 1 : 0;
}
#endif

// ---------------------------------------------------------------- bench
#ifndef UITEST_NO_UI
static void bench(int W, int H) {
    const int NIMG = 16;
    Scene sc(W, H, NIMG, 0, 0);
    printf("\n== %dx%d ==\n", W, H);
    struct Mode { const char* name; Settings s; };
    Settings sOff, sC, sW, sWn;
    sC.uiProtection = UiProtection::Crosshair;
    sW.uiProtection = UiProtection::WholeScreen;
    sWn = sW; sWn.uiMaxStillFraction = 1.0f;
    Mode modes[] = {{"Off", sOff}, {"Crosshair", sC}, {"WholeScreen", sW}, {"WholeScreen, no still-skip", sWn}};
    Img ring[NIMG];
    for (auto& r : ring) r = makeImage(W, H, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    Img out = makeImage(W, H, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    begin();
    for (auto& r : ring) imgBarrier(r.img, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    imgBarrier(out.img, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    submit();
    for (int scene = 0; scene < 3; scene++) {   // 0 moving, 1 still, 2 70 % still (left 70 % identical every frame, right 30 % panning)
        for (int i = 0; i < NIMG; i++) {
            auto f = sc.frame(scene == 1 ? 0 : i);
            if (scene == 2) {
                auto still = sc.frame(0);
                for (int y = 0; y < H; y++) memcpy(&f[(size_t)y * W * 4], &still[(size_t)y * W * 4], (size_t)(W * 7 / 10) * 4);
            }
            upload(f, ring[i], W, H);
        }
        printf("-- %s scene (%s; static crosshair + HUD box)\n", scene == 0 ? "moving" : scene == 1 ? "still" : "70% still", scene == 0 ? "background pans 14,6 px/frame + 16 moving objects" : scene == 1 ? "every frame identical" : "left 70% of every frame identical, right 30% as the moving scene");
        for (auto& md : modes) {
            if (scene == 1 && md.s.uiProtection == UiProtection::Off) continue;
            Interpolator fg;
            if (!makeInterp(fg, W, H, md.s)) { fprintf(stderr, "init failed\n"); exit(1); }
            int pair = 0;
            auto onePair = [&]() {
                Img& cur = ring[pair % NIMG];
                Img& prv = ring[(pair + NIMG - 1) % NIMG];
                if (pair++ == 0 && !fg.recordObserve(cb, prv.view)) { fprintf(stderr, "observe failed\n"); exit(1); }
                if (!fg.recordObserve(cb, cur.view)) { fprintf(stderr, "observe failed\n"); exit(1); }
                if (!fg.recordEstimate(cb, prv.view, cur.view) || !fg.recordSynth(cb, prv.view, cur.view, out.view, 0.5f)) { fprintf(stderr, "record failed\n"); exit(1); }
            };
            fg.setProfiling(true);
            std::vector<const char*> names;
            std::vector<std::vector<double>> per;
            std::vector<double> total;
            for (int it = 0; it < 25; it++) {
                begin();
                fg.beginFrame();
                fg.setProfiling(false);
                for (int k = 0; k < 8; k++) onePair();
                fg.setProfiling(true);
                onePair();
                submit();
                std::vector<uint64_t> ticks;
                if (!fg.readTimings(names, ticks)) { fprintf(stderr, "readTimings failed\n"); exit(1); }
                if (it < 5) continue;
                per.resize(ticks.size());
                double sum = 0;
                for (size_t i = 0; i < ticks.size(); i++) { per[i].push_back(ticks[i] * tsPeriodMs); sum += ticks[i] * tsPeriodMs; }
                total.push_back(sum);
            }
            auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
            double uiSum = 0;
            // back-to-back: 8 pairs per command buffer, GPU kept busy -> the marginal cost of a pair, 7 reps
            fg.setProfiling(false);
            VkQueryPoolCreateInfo qp{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qp.queryType = VK_QUERY_TYPE_TIMESTAMP; qp.queryCount = 2;
            VkQueryPool qpool;
            CK(vkCreateQueryPool(dev, &qp, nullptr, &qpool));
            std::vector<double> b2b;
            for (int rep = 0; rep < 9; rep++) {
                begin();
                fg.beginFrame();
                vkCmdResetQueryPool(cb, qpool, 0, 2);
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);
                for (int k = 0; k < 8; k++) onePair();
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 1);
                submit();
                uint64_t t2[2];
                CK(vkGetQueryPoolResults(dev, qpool, 0, 2, sizeof(t2), t2, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
                if (rep >= 2) b2b.push_back((t2[1] - t2[0]) * tsPeriodMs / 8);
            }
            vkDestroyQueryPool(dev, qpool, nullptr);
            printf("  %-28s", md.name);
            for (size_t i = 0; i < names.size(); i++)
                if (!strncmp(names[i], "ui_", 3)) { double v = med(per[i]); uiSum += v; printf(" %s %.4f", names[i], v); }
            printf("  | UI total %.4f ms, whole pair %.4f ms | back-to-back per pair %.4f ms\n", uiSum, med(total), med(b2b));
            vkDeviceWaitIdle(dev);
            fg.destroy();
        }
    }
    for (auto& r : ring) { vkDestroyImageView(dev, r.view, nullptr); vkDestroyImage(dev, r.img, nullptr); vkFreeMemory(dev, r.mem, nullptr); }
    vkDestroyImageView(dev, out.view, nullptr); vkDestroyImage(dev, out.img, nullptr); vkFreeMemory(dev, out.mem, nullptr);
}
#endif

int main(int argc, char** argv) {
    std::vector<std::string> pos;
    bool hash = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bgra")) bgra = true;
        else if (!strcmp(argv[i], "--hash")) hash = true;
        else pos.push_back(argv[i]);
    }
    if (pos.empty() || (pos[0] != "check" && pos[0] != "bench" && pos[0] != "deep")) { fprintf(stderr, "usage: uitest check|bench|deep [--bgra] [--hash] [W H]\n"); return 2; }
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
    if (pos[0] == "deep") {
#ifdef UITEST_NO_UI
        fprintf(stderr, "deep needs the UI API\n");
        return 2;
#else
        int W = pos.size() >= 3 ? atoi(pos[1].c_str()) : 640, H = pos.size() >= 3 ? atoi(pos[2].c_str()) : 360;
        initStaging((size_t)W * H * 4);
        rc = deepHistory(W, H);
        printf(rc ? "RESULT: FAIL (%d)\n" : "RESULT: PASS\n", fails);
#endif
    } else if (pos[0] == "check") {
        int W = pos.size() >= 3 ? atoi(pos[1].c_str()) : 1280, H = pos.size() >= 3 ? atoi(pos[2].c_str()) : 720;
        initStaging((size_t)W * H * 4);
        rc = check(W, H, hash);
        printf(rc ? "RESULT: FAIL (%d)\n" : "RESULT: PASS\n", fails);
    } else {
#ifdef UITEST_NO_UI
        fprintf(stderr, "bench needs the UI API\n");
        return 2;
#else
        initStaging((size_t)2560 * 1440 * 4);
        if (pos.size() >= 3) bench(atoi(pos[1].c_str()), atoi(pos[2].c_str()));
        else { bench(1920, 1080); bench(2560, 1440); }
#endif
    }
    vkDeviceWaitIdle(dev);
    return rc;
}
