// uitest: validation + benchmark of framegen's static-UI protection (Settings::uiProtection).
//   uitest check [--bgra] [--hash] [W H]     default 1280 720
//   option --algo v1|v2 (every mode): the motion version (Settings::motionVersion) every run starts with (default v1)
//   option --mode quality|performance (every mode): Settings::v2Mode of every run (default quality; V1 ignores it)
//   uitest deep [W H]                        Settings::uiHistory: a synth from a pair 10 frames back still finds its clean copies (default 640x360)
//   uitest track [W H]                       recordTrack: UI protection kept warm while the host does not generate (default 640x360)
//   uitest trackbench [W H]                  GPU cost of recordTrack vs recordObserve (default 1920x1080 and 2560x1440)
//   uitest bench [W H]                       per-pass GPU times, moving and still scene (default 1920x1080 and 2560x1440)
// check also covers CrosshairV2 (the V2 block at its end): pixel-exact crosshair, the shrinking / growing mask control against V1,
// hysteresis, fast re-entry, the pair gate, mode switching, validation, and (ghostCheck) the "doubled crosshair": 1280x960 CS2-like
// scenes with an absolute limit on the artifact pixels around a panned-over crosshair. deep and track run the V2 modes too.
// check: a synthetic sequence of 12 frames: a textured background panning 14 px right / 6 px down per frame with 16 independently moving objects (some crossing the crosshair and the HUD), a static opaque crosshair
// (plus-shaped arms, centre dot, 1 px black outline) and a static opaque HUD box in a corner. The last pair
// (frames 10, 11) is interpolated at t = 0.5 and 0.25 with protection Off / Crosshair / WholeScreen (explicit
// observes; Crosshair and WholeScreen also with implicit observes), and with an explicit and an offset box.
// Asserts: the crosshair pixels of the generated frame equal the real ones (<= 1 level) in Crosshair and
// WholeScreen, the HUD box's only in WholeScreen, background PSNR does not drop; --hash prints an FNV-1a hash of
// every generated frame of the Off run (compare against a build of the library before protection existed: build
// this file with -DUITEST_NO_UI against it).
// check also covers the motion version (motionCheck): V2 refused for every Format but Rgba8, a switch drops the estimate,
// releaseUnusedMotion() frees the unselected version's pipelines and a later switch back re-creates them.
#include "framegen.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#ifdef UITEST_NO_UI
#define UITEST_NO_V2   // a library without the feature has no CrosshairV2 either
#endif
#ifdef UITEST_NO_V2
#define UITEST_NO_MOTION   // nor a motion version (Settings::motionVersion)
#endif
// the modes the shared tests (deep, track, trackbench, bench) loop over besides Crosshair and WholeScreen
#ifdef UITEST_NO_V2
#define UT_V2_MODE
#define UT_V2_ENUM (UiProtection)0
#else
#define UT_V2_MODE , UiProtection::CrosshairV2
#define UT_V2_ENUM UiProtection::CrosshairV2
#endif
static const char* modeName(int m) { return m == 1 ? "Crosshair" : m == 2 ? "WholeScreen" : m == 3 ? "CrosshairV2" : "Off"; }

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
static int kPanX = 14, kPanY = 6;   // background pan per frame (even: the mid frame is an integer shift); ghostCheck() varies it
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
    // uiHook (tests): called for every UI pixel of an INTEGER frame f with the UI colour in px (RGBA, may be rewritten: erase
    // it by copying `under`, the pixel without UI) so a test can make the crosshair change, shrink or flash.
    std::function<void(int f, int x, int y, uint8_t* px, const uint8_t* under)> uiHook;
    Scene(int w, int h, int nf, int cxOff, int cyOff, int arm = 16, int nObj = 16) : W(w), H(h), NF(nf) {
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
        for (int i = 0; i < nObj; i++) {
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
            bool hArm = ay <= 1 + grow && ax >= 5 - grow && ax <= arm + grow;
            bool vArm = ax <= 1 + grow && ay >= 5 - grow && ay <= arm + grow;
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
    std::vector<uint8_t> frame(double t, bool withUi = true) const {
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
        if (withUi)
            for (size_t i = 0; i < (size_t)W * H; i++)
                if (ui[i]) {
                    uint8_t under[4];
                    memcpy(under, &f[i * 4], 4);
                    memcpy(&f[i * 4], &uiCol[i * 4], 4);
                    if (uiHook && t == std::floor(t)) uiHook((int)t, (int)(i % W), (int)(i / W), &f[i * 4], under);
                }
        return f;
    }
};

// ---------------------------------------------------------------- one run
using namespace framegen;
struct Result { std::vector<uint8_t> out[2]; };   // t = 0.5, 0.25 of the last pair
// --algo: the motion version every Interpolator of the run is created and set with
#ifndef UITEST_NO_MOTION
static MotionVersion gAlgo = MotionVersion::V1;
static V2Mode gMode = V2Mode::Quality;   // --mode
static Settings algo(Settings s) { s.motionVersion = gAlgo; s.v2Mode = gMode; return s; }
#else
static Settings algo(Settings s) { return s; }
#endif
static uint64_t fnv(const std::vector<uint8_t>& v) {
    uint64_t h = 1469598103934665603ull;
    for (uint8_t b : v) { h ^= b; h *= 1099511628211ull; }
    return h;
}

static bool makeInterp(Interpolator& fg, int W, int H, const Settings& s) {
    InitInfo info;
    info.phys = phys; info.device = dev; info.gdpa = vkGetDeviceProcAddr; info.width = W; info.height = H;
    info.viewFormat = fmt(); info.getMemProps = vkGetPhysicalDeviceMemoryProperties; info.settings = algo(s);
    return fg.init(info);
}

// Runs the 12-frame sequence through a ring of 4 views; observes explicitly or implicitly. hashes (Off check).
static Result runSequence(const Scene& sc, const Settings& s, bool explicitObserve, std::vector<uint64_t>* hashes,
                          const std::function<void(Interpolator&, int)>& hook = nullptr, bool blur2 = false) {
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
            // blur2: the second output is a motion-blurred one (window [0.25, 0.75], blurSamples of the settings) instead of t = 0.25
            if (last && !(blur2 ? fg.recordSynthBlur(cb, prv.view, cur.view, outs[1].view, 0.25f, 0.75f) : fg.recordSynth(cb, prv.view, cur.view, outs[1].view, 0.25f))) { fprintf(stderr, "synth failed\n"); exit(1); }
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

#ifndef UITEST_NO_MOTION
static void motionCheck(const Scene& sc);
#endif
#ifndef UITEST_NO_V2
static int v2Check(int W, int H);
static void marginCheck(int W, int H);
static const char* dumpDir = nullptr;   // --dump DIR: raw RGBA of the t = 0.5 output of the large-box cases (to diff two library builds)
#endif
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
    // the test crosshair is 35x35 px incl. outline: size the box via the height fraction to fit it (default 0.025 is
    // 18 px at 720, smaller than this crosshair)
    const float kFitFrac = 43.0f / H;   // 43 px: the crosshair plus 4 px of room per side
    Settings c1; c1.uiProtection = UiProtection::Crosshair; c1.uiBoxHeightFrac = kFitFrac;
    Settings w1; w1.uiProtection = UiProtection::WholeScreen; w1.uiBoxHeightFrac = kFitFrac;
    if (hash) {   // the same for the two older modes: compare against a build of the library before CrosshairV2 (-DUITEST_NO_V2)
        Settings v1x = c1; v1x.uiProtection = (UiProtection)3;   // CrosshairV2, same 43 px box
        const std::pair<const char*, Settings> modes[3] = {{"CROSSHAIR", c1}, {"WHOLESCREEN", w1}, {"CROSSHAIRV2", v1x}};
        for (const auto& md : modes) {
            std::vector<uint64_t> h;
            Result r = runSequence(sc, md.second, true, &h);
            printf("HASHES-%s", md.first);
            for (uint64_t x : h) printf(" %016llx", (unsigned long long)x);
            printf("\n");
            if (dumpDir) {
                for (int k = 0; k < 2; k++) {
                    std::string fn = std::string(dumpDir) + "/" + md.first + (k ? "_t025.rgba" : "_t05.rgba");
                    if (FILE* f = fopen(fn.c_str(), "wb")) { fwrite(r.out[k].data(), 1, r.out[k].size(), f); fclose(f); }
                }
            }
        }
    }
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
        Settings c5 = c1;   // default centred box on an off-centre crosshair (37,-23 is inside the fitted box): still covered
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
            if (f == 1) { x.uiProtection = UiProtection::Crosshair; if (!fg.setSettings(algo(x))) { printf("setSettings failed\n"); exit(1); } }
            if (f == 3) { x.uiProtection = UiProtection::WholeScreen; if (!fg.setSettings(algo(x))) { printf("setSettings failed\n"); exit(1); } }
        });
        Metrics m = measure(sc, r.out[0], real11, &truth);
        printf(" Runtime Off -> Crosshair (frame 1) -> WholeScreen (frame 3): crosshair wrong %zu, hud wrong %zu\n", m.xhBad, m.hudBad);
        expect(m.xhBad == 0 && m.hudBad == 0, "runtime switch to WholeScreen mid-stream protects crosshair and HUD");
        Result r2 = runSequence(sc, w1, true, nullptr, [&](Interpolator& fg, int f) {
            if (f == 9) { if (!fg.setSettings(algo(Settings{}))) { printf("setSettings failed\n"); exit(1); } }
        });
        expect(r2.out[0] == rOff.out[0] && r2.out[1] == rOff.out[1], "runtime switch WholeScreen -> Off: later output bit-identical to an Off run");
    }
    // box sizing: Interpolator::boxSize (pure arithmetic, same numbers the library protects)
    {
        auto side = [&](Settings s, uint32_t w, uint32_t h, uint32_t* bh = nullptr) {
            uint32_t bw, bh_; Interpolator::boxSize(s, w, h, &bw, &bh_); if (bh) *bh = bh_; return bw; };
        Settings d;   // defaults: uiBoxHeightFrac 0.025
        expect(side(d, 1280, 960) == 24 && side(d, 1920, 1080) == 27 && side(d, 2560, 1440) == 36 && side(d, 3840, 2160) == 54,
               "default box 24 / 27 / 36 / 54 px at 960 / 1080 / 1440 / 2160 high");
        Settings t = d; t.uiBoxHeightFrac = 0.001f;
        expect(side(t, 1280, 960) == 8, "minimum box side is 8 px (frac 0.001 at 960 high)");
        t.uiBoxHeightFrac = 1.0f;
        expect(side(t, 1280, 960) == 960, "frac 1 is clamped to min(w, h)");
        Settings e = d; e.uiBoxWidth = 100; e.uiBoxHeight = 50;
        uint32_t eh; uint32_t ew = side(e, 1280, 960, &eh);
        expect(ew == 100 && eh == 50, "an explicit px size wins over the fraction");
        Settings l = d; l.uiBoxHeightFrac = 0.0f; l.uiCrosshairArea = 0.03f;
        expect(side(l, 1280, 960) == 192, "legacy: frac 0, area 0.03 gives 192 px at 1280x960");
        l.uiCrosshairArea = 0.0001f;
        expect(side(l, 1280, 960) == 32, "legacy keeps its 32 px minimum");
    }
    // DEFAULT fraction + small crosshair that fits: pixel-exact (default 0.025 is 18 px at 720; a crosshair with
    // arms out to 7 px plus outline is 17 px)
    {
        Scene sm(W, H, 12, 0, 0, 7);
        auto realS = sm.frame(11);
        Settings cd; cd.uiProtection = UiProtection::Crosshair;
        uint32_t bw, bh; Interpolator::boxSize(cd, W, H, &bw, &bh);
        Result rs = runSequence(sm, cd, true, nullptr);
        Metrics ms = measure(sm, rs.out[0], realS, nullptr), ms2 = measure(sm, rs.out[1], realS, nullptr);
        printf(" Small crosshair (17 px) in the default %ux%u box: crosshair wrong %zu/%zu\n", bw, bh, ms.xhBad, ms.xhN);
        expect(bw >= 17 && ms.xhBad == 0 && ms2.xhBad == 0, "default-fraction box around a small crosshair is pixel-exact");
    }
    // settings validation
    {
        Settings bad;
        bad.uiBoxHeightFrac = -0.1f; expect(!Interpolator::validate(bad), "validate rejects uiBoxHeightFrac < 0");
        bad = Settings{}; bad.uiBoxHeightFrac = 1.5f; expect(!Interpolator::validate(bad), "validate rejects uiBoxHeightFrac > 1");
        bad = Settings{}; bad.uiBoxHeightFrac = std::nanf(""); expect(!Interpolator::validate(bad), "validate rejects uiBoxHeightFrac NaN");
        bad = Settings{}; bad.uiBoxHeightFrac = 0.0f; expect(Interpolator::validate(bad), "validate accepts uiBoxHeightFrac 0 (legacy)");
        bad = Settings{};
        bad.uiBoxWidth = 20000; expect(!Interpolator::validate(bad), "validate rejects uiBoxWidth 20000");
        bad = Settings{}; bad.uiBoxOffsetX = -20000; expect(!Interpolator::validate(bad), "validate rejects uiBoxOffsetX -20000");
        bad = Settings{}; bad.uiCrosshairArea = 0.0f; expect(!Interpolator::validate(bad), "validate rejects uiCrosshairArea 0");
        bad = Settings{}; bad.uiInpaintReach = 17; expect(!Interpolator::validate(bad), "validate rejects uiInpaintReach 17");
        bad = Settings{}; bad.uiStillFrames = 0; expect(!Interpolator::validate(bad), "validate rejects uiStillFrames 0");
        bad = Settings{}; bad.uiProtection = (UiProtection)4; expect(!Interpolator::validate(bad), "validate rejects an unknown mode");
        expect(Interpolator::validate(Settings{}), "defaults valid");
    }
#ifndef UITEST_NO_MOTION
    motionCheck(sc);
#endif
#ifndef UITEST_NO_V2
    marginCheck(W, H);
    return v2Check(W, H);
#else
    return fails ? 1 : 0;
#endif
#endif
}

#ifndef UITEST_NO_V2
// ---------------------------------------------------------------- the box margin (what a box that just fits the crosshair protects)
// A tiny crosshair (a 1 px plus: 8 px wide, 7 px tall, 14 px) with a 1 px rim around it, centred, over a panning background, in a
// box that is exactly as big as the crosshair plus rim (11x11), one px too small for it (9x9: the rim's right / bottom pixel is
// outside) or 9x9 and shifted so its left / top rim pixel is outside. Reported: UI pixels (core and rim) that differ from the real
// frame, in the t = 0.5 output, the t = 0.25 output and the motion-blurred output; and the artifact pixels around it: pixels that are
// not UI and differ by more than 24 levels from the same run on a scene without the crosshair (Off), within 48 px of it.
// The rim is either a static outline (opaque black: both modes can see it) or an anti-aliased one (50 % black over the moving
// background, so it changes every frame: only V1's dilation covers it; V2 pastes a pixel only if it did not change in the pair, by design).
static void tinyCrosshair(Scene& sc, bool movingRim) {
    std::fill(sc.ui.begin(), sc.ui.end(), 0);
    std::fill(sc.uiCol.begin(), sc.uiCol.end(), 0);
    auto core = [](int dx, int dy) { return (dy == 0 && dx >= -4 && dx <= 3) || (dx == 0 && dy >= -3 && dy <= 3); };
    for (int dy = -8; dy <= 8; dy++)
        for (int dx = -8; dx <= 8; dx++) {
            bool rim = false;
            for (int y = -1; y <= 1; y++)
                for (int x = -1; x <= 1; x++) rim = rim || core(dx + x, dy + y);
            if (!rim) continue;
            const size_t i = (size_t)(sc.cy + dy) * sc.W + sc.cx + dx;
            const bool c = core(dx, dy);
            sc.ui[i] = c ? 1 : 3;
            uint8_t* p = &sc.uiCol[i * 4];
            p[0] = c ? 255 : 0; p[1] = 0; p[2] = c ? 255 : 0; p[3] = 255;
        }
    sc.uiHook = nullptr;
    if (movingRim)
        sc.uiHook = [&sc](int, int x, int y, uint8_t* px, const uint8_t* under) {
            if (sc.ui[(size_t)y * sc.W + x] == 3) for (int c = 0; c < 3; c++) px[c] = (uint8_t)(under[c] / 2);
        };
}
struct MarginRow { size_t uiN = 0, bad05 = 0, bad025 = 0, badBlur = 0, art05 = 0, artBlur = 0; };
static MarginRow marginMeasure(const Scene& sc, const Result& plain, const Result& blur, const std::vector<uint8_t>& real, const Result& refPlain, const Result& refBlur) {
    MarginRow m;
    auto maxd = [&](const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, size_t i) {
        int d = 0;
        for (int c = 0; c < 3; c++) d = std::max(d, std::abs((int)a[i * 4 + c] - (int)b[i * 4 + c]));
        return d;
    };
    for (int y = std::max(sc.cy - 48, 0); y <= std::min(sc.cy + 48, sc.H - 1); y++)
        for (int x = std::max(sc.cx - 48, 0); x <= std::min(sc.cx + 48, sc.W - 1); x++) {
            const size_t i = (size_t)y * sc.W + x;
            if (sc.ui[i]) {
                m.uiN++;
                m.bad05 += maxd(plain.out[0], real, i) > 1;
                m.bad025 += maxd(plain.out[1], real, i) > 1;
                m.badBlur += maxd(blur.out[1], real, i) > 1;
            } else {
                m.art05 += maxd(plain.out[0], refPlain.out[0], i) > 24;
                m.artBlur += maxd(blur.out[1], refBlur.out[1], i) > 24;
            }
        }
    return m;
}
static void marginCheck(int W, int H) {
    printf("\nBox margin (tiny crosshair: 8x7 px core + 1 px rim, boxes that just fit it):\n");
    Settings off;
    Settings offBlur = off; offBlur.blurSamples = 4;
#ifndef UITEST_NO_MARGIN_API
    {   // the work rectangle API
        Settings s; s.uiProtection = UiProtection::Crosshair; s.uiBoxWidth = s.uiBoxHeight = 11;
        int32_t b[4], w[4];
        Interpolator::boxRect(s, W, H, b); Interpolator::workRect(s, W, H, w);
        expect(Interpolator::boxMargin(s) == 2 && w[0] == b[0] - 2 && w[1] == b[1] - 2 && w[2] == b[2] + 4 && w[3] == b[3] + 4, "workRect: V1 box grown by 2 px on every side");
        s.uiProtection = UiProtection::CrosshairV2;
        Interpolator::workRect(s, W, H, w);
        expect(Interpolator::boxMargin(s) == 1 && w[0] == b[0] - 1 && w[2] == b[2] + 2, "workRect: V2 box grown by 1 px on every side");
        s.uiProtection = UiProtection::WholeScreen; Interpolator::workRect(s, W, H, w);
        expect(Interpolator::boxMargin(s) == 0 && w[0] == b[0] && w[2] == b[2], "workRect: WholeScreen keeps its box");
        s.uiProtection = UiProtection::Crosshair; s.uiBoxWidth = s.uiBoxHeight = 9999; s.uiBoxOffsetX = 5000;
        Interpolator::workRect(s, W, H, w);
        expect(w[0] == 0 && w[1] == 0 && w[2] == W && w[3] == H, "workRect: clamped to the frame");
        s.uiBoxWidth = s.uiBoxHeight = 20; s.uiBoxOffsetX = -5000; s.uiBoxOffsetY = -5000;
        Interpolator::workRect(s, W, H, w);
        expect(w[0] == 0 && w[1] == 0 && w[2] == 22 && w[3] == 22, "workRect: a box in the frame corner is clamped on its outer sides only");
    }
#endif
    struct Case { const char* name; UiProtection mode; int box, ox, oy; bool movingRim; bool fill; };
    const Case cases[] = {
        {"V1 box 11x11 (fits core + rim), static rim", UiProtection::Crosshair, 11, 0, 0, false, false},
        {"V1 box 9x9 (rim right/bottom outside)", UiProtection::Crosshair, 9, 0, 0, false, false},
        {"V1 box 9x9 shifted +1,+1 (rim left/top outside)", UiProtection::Crosshair, 9, 1, 1, false, false},
        {"V1 box 11x11, AA rim (changes every frame)", UiProtection::Crosshair, 11, 0, 0, true, false},
        {"V1 box 9x9, AA rim", UiProtection::Crosshair, 9, 0, 0, true, false},
        {"V2 box 11x11, static rim", UiProtection::CrosshairV2, 11, 0, 0, false, false},
        {"V2 box 9x9, static rim", UiProtection::CrosshairV2, 9, 0, 0, false, false},
        {"V2 box 9x9 shifted +1,+1, static rim", UiProtection::CrosshairV2, 9, 1, 1, false, false},
        {"V2 + fill box 11x11, static rim", UiProtection::CrosshairV2, 11, 0, 0, false, true},
        {"V2 + fill box 9x9, static rim", UiProtection::CrosshairV2, 9, 0, 0, false, true},
        {"V2 + fill box 9x9 shifted +1,+1, static rim", UiProtection::CrosshairV2, 9, 1, 1, false, true},
    };
    for (int pass = 0; pass < 2; pass++) {
        const bool moving = pass == 1;
        Scene sc(W, H, 12, 0, 0, 7, 0);
        tinyCrosshair(sc, moving);
        Scene sn = sc;
        std::fill(sn.ui.begin(), sn.ui.end(), 0);
        sn.uiHook = nullptr;
        auto real = sc.frame(11);
        Result refPlain = runSequence(sn, off, true, nullptr), refBlur = runSequence(sn, offBlur, true, nullptr, nullptr, true);
        Result oPlain = runSequence(sc, off, true, nullptr), oBlur = runSequence(sc, offBlur, true, nullptr, nullptr, true);
        MarginRow mo = marginMeasure(sc, oPlain, oBlur, real, refPlain, refBlur);
        printf(" %s rim, Off:%*sartifact px: t=0.5 %zu, blur %zu\n", moving ? "AA" : "static", 40, "", mo.art05, mo.artBlur);
        for (const Case& cs : cases) {
            if (cs.movingRim != moving) continue;
            Settings s; s.uiProtection = cs.mode; s.uiBoxWidth = s.uiBoxHeight = cs.box; s.uiBoxOffsetX = cs.ox; s.uiBoxOffsetY = cs.oy; s.uiEstimateFill = cs.fill;
            Settings sb = s; sb.blurSamples = 4;
            Result rp = runSequence(sc, s, true, nullptr), rb = runSequence(sc, sb, true, nullptr, nullptr, true);
            MarginRow m = marginMeasure(sc, rp, rb, real, refPlain, refBlur);
            printf("  %-52s UI wrong %zu/%zu (t=0.25 %zu, blur %zu), artifact px: t=0.5 %zu, blur %zu\n", cs.name, m.bad05, m.uiN, m.bad025, m.badBlur, m.art05, m.artBlur);
            char what[160];
            snprintf(what, sizeof what, "%s: every crosshair and rim pixel equals the real one (t = 0.5, 0.25 and blurred)", cs.name);
            expect(m.bad05 == 0 && m.bad025 == 0 && m.badBlur == 0, what);
            // Absolute limits. V1 and V2 + fill take the crosshair out of the motion estimate, so nothing smears around it: at most 4.
            // V2 without the fill still lets the crosshair vote in the estimate (a zero vector in the blocks around it: ~17 px on this
            // smooth scene, Off has 21), which only the fill removes; its warp no longer copies the crosshair (ui_reject.glsl).
            const bool noFillV2 = cs.mode == UiProtection::CrosshairV2 && !cs.fill;
            const size_t lim05 = noFillV2 ? 20 : 4, limBlur = 4;
            snprintf(what, sizeof what, "%s: artifact pixels around it at most %zu (Off: %zu / %zu)", cs.name, lim05, mo.art05, mo.artBlur);
            expect(m.art05 <= lim05 && m.artBlur <= limBlur, what);
        }
    }
}
#endif


// ---------------------------------------------------------------- the "doubled crosshair" (CrosshairV2 tap rejection)
// A CS2-like scene: 1280x960, a tiny opaque cyan plus (8x7 px, 1 px thick, no rim) at the centre, a background that pans with fine
// detail, no moving objects. Without a protection the warp copies the crosshair along the pan: every generated pixel blends a tap of
// prev at p - 2t*h and one of curr at p + 2(1-t)*h, and a tap that lands on the crosshair carries it to crosshair +- pan*t (a second,
// full-strength crosshair next to the real one). V1 inpaints it out of the warped frames; V2 hands the real frames to the warp and
// instead rejects the taps that land on the UI (shaders/ui_reject.glsl). Measured as ARTIFACT PIXELS: non-UI pixels within 24 px of the
// crosshair that differ by more than 24 levels from the same run on the scene WITHOUT the crosshair (protection Off), so the limits are
// absolute: what must be left of the crosshair's effect on its surroundings, not "no more than Off".
// Backgrounds: smooth value noise (the check scene's), hard 3x3 blocks of 20 / 235, and per-pixel grain (CS2-ish fine detail).
// Settings profiles: the library defaults, and the ones the host runs with (hudBonus 2.5, trust 254..255, fallbacks off) where an
// artifact is not hidden by the untrusted-pixel fallbacks.
static void ghostCross(Scene& sc) {
    std::fill(sc.ui.begin(), sc.ui.end(), 0);
    std::fill(sc.uiCol.begin(), sc.uiCol.end(), 0);
    for (int dy = -3; dy <= 3; dy++)
        for (int dx = -4; dx <= 3; dx++) {
            if (dy != 0 && dx != 0) continue;
            const size_t i = (size_t)(sc.cy + dy) * sc.W + sc.cx + dx;
            sc.ui[i] = 1;
            uint8_t* p = &sc.uiCol[i * 4];
            p[0] = 0; p[1] = 255; p[2] = 255; p[3] = 255;
        }
    sc.uiHook = nullptr;
}
static void ghostBackground(Scene& sc, int kind) {   // 0 = the scene's own smooth noise
    for (int y = 0; y < sc.bgH; y++)
        for (int x = 0; x < sc.bgW; x++)
            for (int c = 0; c < 3; c++) {
                uint8_t& v = sc.bg[((size_t)y * sc.bgW + x) * 4 + c];
                if (kind == 1) v = (hash2(x / 3, y / 3, 91 + c) & 1) ? 235 : 20;
                if (kind == 2) {
                    const float base = 60.0f + 120.0f * vnoise(x, y, 24, 5) + 40.0f * vnoise(x, y, 5, 6);
                    const float g = ((hash2(x, y, 7) & 255) / 255.0f - 0.5f) * 70.0f;
                    v = (uint8_t)std::min(255.0f, std::max(0.0f, base * (c == 2 ? 0.8f : 1.0f) + g));
                }
            }
}
static void ghostCheck() {
    const int W = 1280, H = 960;
    printf("\nThe doubled crosshair (%dx%d, 8x7 px cyan plus, artifact px within 24 px of it vs the same run without the crosshair):\n", W, H);
    const char* bgName[] = {"smooth noise", "hard 3x3 blocks", "fine grain"};
    const int pans[][2] = {{14, 6}, {30, 0}};
    const int oldPanX = kPanX, oldPanY = kPanY;
    for (int bgk = 0; bgk < 3; bgk++)
        for (const auto& pn : pans)
            for (int prof = 0; prof < 2; prof++) {
                kPanX = pn[0]; kPanY = pn[1];
                Scene sc(W, H, 12, 0, 0, 7, 0);
                ghostBackground(sc, bgk);
                ghostCross(sc);
                Scene sn = sc;
                std::fill(sn.ui.begin(), sn.ui.end(), 0);
                const auto real = sc.frame(11), f10 = sc.frame(10);
                Settings base;
                if (prof == 1) { base.hudBonus = 2.5f; base.trustLow = 254; base.trustHigh = 255; base.globalFallback = 1.0f; base.sceneCutSad = 255; }
                const Result ref = runSequence(sn, base, true, nullptr);
                struct Row { size_t art05 = 0, art025 = 0, uiBad = 0; };
                auto measureRow = [&](const Settings& st) {
                    const Result r = runSequence(sc, st, true, nullptr);
                    Row m;
                    for (int y = sc.cy - 24; y <= sc.cy + 24; y++)
                        for (int x = sc.cx - 24; x <= sc.cx + 24; x++) {
                            const size_t i = (size_t)y * W + x;
                            int d05 = 0, d025 = 0, dr = 0;
                            for (int c = 0; c < 3; c++) {
                                d05 = std::max(d05, std::abs((int)r.out[0][i * 4 + c] - (int)ref.out[0][i * 4 + c]));
                                d025 = std::max(d025, std::abs((int)r.out[1][i * 4 + c] - (int)ref.out[1][i * 4 + c]));
                                dr = std::max(dr, std::abs((int)r.out[0][i * 4 + c] - (int)real[i * 4 + c]));
                            }
                            if (sc.ui[i]) { m.uiBad += dr > 1; continue; }
                            // a background pixel the two real frames show identically, next to the crosshair: V2's patch pastes it
                            // by design (the mask's grow; it did not change in this pair) -- not a warp artifact (see v2Check 2.)
                            if (!memcmp(&f10[i * 4], &real[i * 4], 3)) continue;
                            m.art05 += d05 > 24;
                            m.art025 += d025 > 24;
                        }
                    return m;
                };
                Settings off = base, v1 = base, v2 = base, v2f = base;
                v1.uiProtection = UiProtection::Crosshair;
                v2.uiProtection = v2f.uiProtection = UiProtection::CrosshairV2;
                v2f.uiEstimateFill = true;
                v1.uiBoxWidth = v1.uiBoxHeight = v2.uiBoxWidth = v2.uiBoxHeight = v2f.uiBoxWidth = v2f.uiBoxHeight = 11;
                const Row mo = measureRow(off), m1 = measureRow(v1), m2 = measureRow(v2), mf = measureRow(v2f);
                char name[160];
                snprintf(name, sizeof name, "%s, pan %d,%d, %s settings", bgName[bgk], kPanX, kPanY, prof ? "host" : "default");
                printf(" %-44s artifact px (t=0.5 / 0.25)  Off %3zu/%3zu  V1 %3zu/%3zu  V2 %3zu/%3zu  V2+fill %3zu/%3zu\n", name,
                       mo.art05, mo.art025, m1.art05, m1.art025, m2.art05, m2.art025, mf.art05, mf.art025);
                char what[256];
                snprintf(what, sizeof what, "%s: V2 and V2 + fill keep the crosshair pixel-exact", name);
                expect(m2.uiBad == 0 && mf.uiBad == 0, what);
                // the fill takes the crosshair out of the estimate too: nothing of it is left around the crosshair
                snprintf(what, sizeof what, "%s: V2 + fill leaves at most 1 artifact px (t = 0.5) and 2 (t = 0.25)", name);
                expect(mf.art05 <= 1 && mf.art025 <= 2, what);
                if (bgk == 0) {
                    // smooth noise: the crosshair's vote in the estimate (not its warp) is what is left; only the fill removes it
                    snprintf(what, sizeof what, "%s: V2 leaves no more artifact px than Off", name);
                    expect(m2.art05 <= mo.art05 && m2.art025 <= mo.art025, what);
                } else {
                    // detailed backgrounds: the warp copy is the whole artifact; rejecting it leaves (almost) nothing, with or without the fill
                    snprintf(what, sizeof what, "%s: V2 leaves at most 1 artifact px (t = 0.5) and 2 (t = 0.25)", name);
                    expect(m2.art05 <= 1 && m2.art025 <= 2, what);
                    snprintf(what, sizeof what, "%s: control, Off shows the doubled crosshair (>= 15 px, V2 %zu)", name, m2.art05);
                    expect(mo.art05 >= 15, what);
                }
            }
    kPanX = oldPanX; kPanY = oldPanY;
}

// ---------------------------------------------------------------- CrosshairV2
#ifndef UITEST_NO_V2
// The scenes of these tests have NO moving objects (the background pans 14, 6 px per frame, the crosshair is static), so the
// interpolation of everything that is not UI is near perfect and a protection artifact stands out. `Scene::uiHook` makes the
// crosshair change: a "tip" or a "right arm" subset changes colour, disappears or comes back at chosen frames.
struct V2Case {
    Settings s;
    bool explicitObserve = true;
};
static Settings v2Settings(int H) {
    Settings s;
    s.uiProtection = UiProtection::CrosshairV2;
    s.uiBoxHeightFrac = 43.0f / H;   // the fitted box of the other tests
    return s;
}
static bool subsetPx(const Scene& sc, int x, int) { return x - sc.cx >= 8; }   // the right arm (and its outline) beyond the centre block
static bool tipPx(const Scene& sc, int x, int y) { return std::max(std::abs(x - sc.cx), std::abs(y - sc.cy)) > 10; }   // the outer arm ends
static void setRgb(uint8_t* p, int r, int g, int b) { p[0] = (uint8_t)r; p[1] = (uint8_t)g; p[2] = (uint8_t)b; p[3] = 255; }
// pixels of `sc.ui == 1` for which pick(x, y) holds that differ from `ref` by more than `tol` levels in `img`
static size_t uiDiffers(const Scene& sc, const std::vector<uint8_t>& img, const std::vector<uint8_t>& ref, int tol, const std::function<bool(int, int)>& pick, size_t* total = nullptr) {
    size_t bad = 0, n = 0;
    for (int y = 0; y < sc.H; y++)
        for (int x = 0; x < sc.W; x++) {
            size_t i = (size_t)y * sc.W + x;
            if (sc.ui[i] != 1 || !pick(x, y)) continue;
            int d = 0;
            for (int c = 0; c < 3; c++) d = std::max(d, std::abs((int)img[i * 4 + c] - (int)ref[i * 4 + c]));
            bad += d > tol;
            n++;
        }
    if (total) *total = n;
    return bad;
}

// A hard-edged background (random 3x3 blocks of 20 / 235 per channel) around and under the crosshair: a smooth fill of
// a removed or hidden crosshair pixel is then far from the truth, which is what makes the artifacts of a clean copy visible.
static void harshBackground(Scene& sc) {
    for (int y = 0; y < sc.bgH; y++)
        for (int x = 0; x < sc.bgW; x++)
            for (int c = 0; c < 3; c++) sc.bg[((size_t)y * sc.bgW + x) * 4 + c] = (hash2(x / 3, y / 3, 91 + c) & 1) ? 235 : 20;
}
static int v2Check(int W, int H) {
    printf("\nCrosshairV2:\n");
    const Settings v2 = v2Settings(H);
    auto all = [](int, int) { return true; };

    // 1. the crosshair of the check scene (objects cross it, background pans) is pixel-exact; implicit observe is bit-identical
    {
        Scene sc(W, H, 12, 0, 0);
        auto real11 = sc.frame(11);
        Result r = runSequence(sc, v2, true, nullptr);
        Result ri = runSequence(sc, v2, false, nullptr);
        size_t n;
        size_t bad = uiDiffers(sc, r.out[0], real11, 1, all, &n), bad2 = uiDiffers(sc, r.out[1], real11, 1, all);
        printf(" check scene: crosshair wrong %zu/%zu (t=0.5), %zu (t=0.25)\n", bad, n, bad2);
        expect(bad == 0 && bad2 == 0, "V2: crosshair pixels equal the real ones (<= 1 level), t = 0.5 and 0.25");
        expect(r.out[0] == ri.out[0] && r.out[1] == ri.out[1], "V2: implicit observe == explicit observe (bit-exact)");
        Settings off;
        Result ro = runSequence(sc, off, true, nullptr);
        auto truth = sc.frame(10.5);
        Metrics mo = measure(sc, ro.out[0], real11, &truth), mv = measure(sc, r.out[0], real11, &truth);
        printf(" background PSNR (away from the UI): Off %.2f dB, V2 %.2f dB\n", mo.psnrBg, mv.psnrBg);
        expect(mv.psnrBg >= mo.psnrBg - 0.3, "V2: background PSNR within 0.3 dB of Off (or better)");
        // the default box (small) around a small crosshair, and an offset box on an off-centre crosshair
        Scene sm(W, H, 12, 0, 0, 7);
        Settings cd; cd.uiProtection = UiProtection::CrosshairV2;
        Result rs = runSequence(sm, cd, true, nullptr);
        expect(uiDiffers(sm, rs.out[0], sm.frame(11), 1, all) == 0, "V2: default-fraction box around a small crosshair is pixel-exact");
        Scene so(W, H, 12, 37, -23);
        Settings co = v2; co.uiBoxWidth = 120; co.uiBoxHeight = 120; co.uiBoxOffsetX = 37; co.uiBoxOffsetY = -23;
        Result ro2 = runSequence(so, co, true, nullptr);
        expect(uiDiffers(so, ro2.out[0], so.frame(11), 1, all) == 0, "V2: offset box protects the off-centre crosshair exactly");
        // contrast gate: a gate above the crosshair's contrast leaves it unprotected (the gate has teeth); 0 disables the gate
        Settings hi = v2; hi.uiContrastThreshold = 255.0f;
        Result rh = runSequence(sc, hi, true, nullptr);
        size_t badHi = uiDiffers(sc, rh.out[0], real11, 1, all);
        Settings lo = v2; lo.uiContrastThreshold = 0.0f;
        Result rl = runSequence(sc, lo, true, nullptr);
        size_t badLo = uiDiffers(sc, rl.out[0], real11, 1, all);
        printf(" contrast gate 255: crosshair wrong %zu; gate 0: %zu\n", badHi, badLo);
        expect(badHi > 0, "V2: uiContrastThreshold 255 admits nothing: the crosshair is not protected");
        expect(badLo == 0, "V2: uiContrastThreshold 0 (no gate): crosshair pixel-exact");
    }

    // 1b. V2 without the estimate fill (the default) runs the estimate on the host's frames exactly as Off does, and the synthesis
    //     differs from Off's only where a warp tap can land on the UI (ui_reject.glsl): the pixels within the motion reach of the work
    //     rectangle. Outside that, the generated frames are bit-identical to Off's. With the fill on (uiEstimateFill) the crosshair
    //     stays exact and the error near a thick crosshair on a hard-edged moving background does not grow (it falls: the crosshair no
    //     longer casts a zero-vector vote in the blocks around it).
    {
        Scene sc(W, H, 12, 0, 0);
        Settings off;
        Result ro = runSequence(sc, off, true, nullptr), rv = runSequence(sc, v2, true, nullptr);
        int32_t wr[4];
        Interpolator::workRect(v2, W, H, wr);
        // reach of a tap: the scene's fastest object moves 20 px / frame, a tap at t = 0.25 is 2 * 0.75 = 1.5 half-vectors away
        // (= 0.75 * 20 px), + 1 px of bilinear footprint + 3 px: the 7x7 trust box spreads a changed mismatch
        const int reach = 15 + 1 + 3;
        size_t outside = 0, inside = 0;
        int far = 0;   // distance (Chebyshev) of the farthest differing pixel from the work rectangle
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                size_t i = (size_t)y * W + x;
                bool differs = memcmp(&ro.out[0][i * 4], &rv.out[0][i * 4], 3) != 0 || memcmp(&ro.out[1][i * 4], &rv.out[1][i * 4], 3) != 0;
                if (!differs) continue;
                const int d = std::max(std::max(wr[0] - x, x - (wr[0] + wr[2] - 1)), std::max(wr[1] - y, y - (wr[1] + wr[3] - 1)));
                far = std::max(far, d);
                if (d > reach) outside++; else inside++;
            }
        printf(" V2 vs Off (t = 0.5 and 0.25): %zu pixels differ within %d px of the %dx%d work rectangle, %zu beyond (farthest %d px)\n", inside, reach, wr[2], wr[3], outside, far);
        expect(inside > 0 && outside == 0, "V2 (no estimate fill): the generated frames are bit-identical to Off beyond the work rectangle plus the motion reach");
        for (int arm : {7, 16}) {
            Scene sh(W, H, 12, 0, 0, arm, 0);
            harshBackground(sh);
            auto truth = sh.frame(10.5, false), real = sh.frame(11);
            Result a = runSequence(sh, off, true, nullptr);
            Settings vf = v2; vf.uiEstimateFill = true;
            Result b = runSequence(sh, vf, true, nullptr);
            auto meanErr = [&](const std::vector<uint8_t>& out) {
                double se = 0; size_t n = 0;
                for (int y = std::max(sh.cy - 24, 0); y <= std::min(sh.cy + 24, H - 1); y++)
                    for (int x = std::max(sh.cx - 24, 0); x <= std::min(sh.cx + 24, W - 1); x++) {
                        size_t i = (size_t)y * W + x;
                        if (sh.ui[i]) continue;
                        int e = 0;
                        for (int c = 0; c < 3; c++) e = std::max(e, std::abs((int)out[i * 4 + c] - (int)truth[i * 4 + c]));
                        se += e; n++;
                    }
                return se / n;
            };
            double eo = meanErr(a.out[0]), ef = meanErr(b.out[0]);
            printf(" estimate fill, hard-edged moving background, arm %d: mean error within 24 px of the crosshair: Off %.2f, V2 + fill %.2f\n", arm, eo, ef);
            expect(uiDiffers(sh, b.out[0], real, 1, all) == 0, "V2 + estimate fill: crosshair pixels equal the real ones");
            // V2 Quality: its blur along the motion depends on how much the taps disagree, i.e. on the field, which the fill changes;
            // on this hard-edged background the field is wrong with and without the fill (Off ~70 levels) and the blur moves the
            // mean by about a level either way (without the blur: Off 64.2, fill 57.6): 2 % of Off's error
            const double tol = gAlgo == MotionVersion::V2 && gMode == V2Mode::Quality ? std::max(0.5, 0.02 * eo) : 0.5;
            expect(ef <= eo + tol, "V2 + estimate fill: the background near the crosshair is not worse than Off");
        }
    }

    // 2. the shrinking / growing mask: V1's known artifacts (inpainted clean copy read by the warp, paste onto changed pixels)
    //    against V2's. Pair (10, 11): the outer arm ends vanish (or appear) at frame 11. Artifact = a pixel that is wrong
    //    by > 32 levels against the true mid frame AND > 24 levels worse than with protection Off, in the 43x43 px around the
    //    crosshair, excluding the crosshair itself and the arm ends (whose true mid frame is ambiguous).
    for (int grow = 0; grow < 2; grow++) {
        Scene sc(W, H, 12, 0, 0, 16, 0);
        harshBackground(sc);
        sc.uiHook = [&](int f, int x, int y, uint8_t* px, const uint8_t* under) {
            if (tipPx(sc, x, y) && (grow ? f < 11 : f >= 11)) memcpy(px, under, 4);
        };
        auto real11 = sc.frame(11);
        auto truth = sc.frame(10.5, false);   // the pan at 10.5, no UI: valid wherever the UI is absent at both frames
        auto changed = [&](int x, int y) { return tipPx(sc, x, y); };
        std::vector<uint8_t> skip((size_t)W * H, 0);   // the crosshair itself, and the arm ends with 2 px around them
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                if (sc.ui[(size_t)y * W + x]) {
                    skip[(size_t)y * W + x] = 1;
                    if (tipPx(sc, x, y))
                        for (int dy = -2; dy <= 2; dy++)
                            for (int dx = -2; dx <= 2; dx++) {
                                int X = x + dx, Y = y + dy;
                                if (X >= 0 && Y >= 0 && X < W && Y < H) skip[(size_t)Y * W + X] = 1;
                            }
                }
        (void)changed;
        Settings off;
        Result ro = runSequence(sc, off, true, nullptr);
        Settings c1 = v2; c1.uiProtection = UiProtection::Crosshair;
        Result r1 = runSequence(sc, c1, true, nullptr);
        Result r2 = runSequence(sc, v2, true, nullptr);
        // A background pixel that the two real frames show identically (the harsh 8-colour pattern repeats, so the pan leaves some
        // pixels unchanged) and that the mask's 1 px grow covers is pasted back by V2's patch by design (it did not change in this
        // pair); its true mid frame differs. Not an artifact of the warp: skipped (the old synthesis was off by as much there with
        // protection Off too, so it never counted).
        const auto f10 = sc.frame(10);
        auto unchanged = [&](size_t i) { return !memcmp(&f10[i * 4], &real11[i * 4], 3); };
        auto artifacts = [&](const std::vector<uint8_t>& out) {
            size_t art = 0;
            for (int y = std::max(sc.cy - 21, 0); y <= std::min(sc.cy + 21, H - 1); y++)
                for (int x = std::max(sc.cx - 21, 0); x <= std::min(sc.cx + 21, W - 1); x++) {
                    size_t i = (size_t)y * W + x;
                    if (skip[i] || unchanged(i)) continue;
                    int eo = 0, ec = 0;
                    for (int c = 0; c < 3; c++) {
                        eo = std::max(eo, std::abs((int)ro.out[0][i * 4 + c] - (int)truth[i * 4 + c]));
                        ec = std::max(ec, std::abs((int)out[i * 4 + c] - (int)truth[i * 4 + c]));
                    }
                    art += ec - eo > 24 && ec > 32;
                }
            return art;
        };
        size_t a1 = artifacts(r1.out[0]), a2 = artifacts(r2.out[0]);
        size_t x1 = uiDiffers(sc, r1.out[0], real11, 1, [&](int x, int y) { return !tipPx(sc, x, y); });
        size_t x2 = uiDiffers(sc, r2.out[0], real11, 1, [&](int x, int y) { return !tipPx(sc, x, y); });
        printf(" arm ends %s at frame 11: artifact px  V1 %zu   V2 %zu   (crosshair body wrong: V1 %zu, V2 %zu)\n", grow ? "appear" : "vanish", a1, a2, x1, x2);
        // The same window as a whole: mean error against the true mid frame and pixels off by > 32, Off vs V2.
        auto windowErr = [&](const std::vector<uint8_t>& out, size_t& bad) {
            double sum = 0; size_t n = 0; bad = 0;
            for (int y = std::max(sc.cy - 21, 0); y <= std::min(sc.cy + 21, H - 1); y++)
                for (int x = std::max(sc.cx - 21, 0); x <= std::min(sc.cx + 21, W - 1); x++) {
                    size_t i = (size_t)y * W + x;
                    if (skip[i] || unchanged(i)) continue;
                    int e = 0;
                    for (int c = 0; c < 3; c++) e = std::max(e, std::abs((int)out[i * 4 + c] - (int)truth[i * 4 + c]));
                    sum += e; n++; bad += e > 32;
                }
            return n ? sum / n : 0.0;
        };
        size_t bo, b2;
        const double mo = windowErr(ro.out[0], bo), m2 = windowErr(r2.out[0], b2);
        printf("   window: mean error / px > 32  Off %.2f / %zu   V2 %.2f / %zu\n", mo, bo, m2, b2);
        // MotionVersion::V2 (both modes): no V1 pixel field, so on this harsh random pattern the static crosshair (a zero-vector vote,
        // the estimate sees it as with Off) leaves the block field around it wrong with or without protection (Off's window mean is
        // 55-70 levels) and the tap rejection only re-deals that error pixel by pixel (Quality: its edge-aware per-pixel field, the
        // coverage weights and the blur along the motion re-deal it too, ~10-30 such pixels against V1's 170-530): judged on the
        // window as a whole instead, V2 must not be worse than Off.
        const bool perf = gAlgo == MotionVersion::V2;
        const char* what = grow ? "V2: no artifact pixel when the arm ends appear" : "V2: no artifact pixel when the arm ends vanish";
        if (perf) what = grow ? "V2: arm ends appear: the window is not worse than Off (mean, pixels > 32)"
                              : "V2: arm ends vanish: the window is not worse than Off (mean, pixels > 32)";
        expect(perf ? m2 <= mo + 0.5 && b2 <= bo : a2 == 0, what);
        expect(a1 > 0, "control: V1 shows artifact pixels in the same pair (the test has teeth)");
        expect(x2 == 0, "V2: the unchanged part of the crosshair stays pixel-exact");
    }

    // 3. hysteresis: the right arm is disturbed for 2 frames (colours cycle: every frame is a change) and comes back; it
    //    must still be protected the very first pair after it is back. leave 4 stays (3 changed frames in a row), leave 1 does not.
    {
        const int NF = 18;   // last pair (16, 17): the arm is back at frame 16
        Scene sc(W, H, NF, 0, 0, 16, 0);
        sc.uiHook = [&](int f, int x, int y, uint8_t* px, const uint8_t*) {
            if (!subsetPx(sc, x, y)) return;
            if (f == 14) setRgb(px, 10, 200, 10);
            else if (f == 15) setRgb(px, 220, 220, 10);
        };
        auto real = sc.frame(NF - 1);
        Settings h4 = v2; h4.uiReentryFrames = 0;   // isolate the hysteresis from fast re-entry
        Settings h1 = h4; h1.uiLeaveFrames = 1;
        Result r4 = runSequence(sc, h4, true, nullptr), r1 = runSequence(sc, h1, true, nullptr);
        size_t n, b4 = uiDiffers(sc, r4.out[0], real, 1, all, &n), b1 = uiDiffers(sc, r1.out[0], real, 1, all);
        Settings c1 = v2; c1.uiProtection = UiProtection::Crosshair;
        Result rv1 = runSequence(sc, c1, true, nullptr);
        size_t bv1 = uiDiffers(sc, rv1.out[0], real, 1, all);
        printf(" hysteresis (2 changed frames, back at 16, pair 16-17): wrong px  leave 4: %zu   leave 1: %zu   V1: %zu   (of %zu)\n", b4, b1, bv1, n);
        expect(b4 == 0, "hysteresis: leave 4 keeps the arm masked through a 2-frame disturbance, exact on the first pair after it");
        expect(b1 > 0, "control: leave 1 loses the arm at the first change and has not got it back");
        expect(bv1 > 0, "control: V1 loses the arm (needs 8 still frames again)");
    }

    // 4. fast re-entry: the right arm cycles colours for 5 frames (14..18: it LEAVES the mask after 4 changed frames), is back
    //    at frame 19 and stays. With re-entry 2 it is protected again two frames later: pair (19, 20); without it only after
    //    uiStillFrames = 8 still frames: pair (26, 27).
    {
        auto runAt = [&](int nf, const Settings& s) {
            Scene sc(W, H, nf, 0, 0, 16, 0);
            sc.uiHook = [&](int f, int x, int y, uint8_t* px, const uint8_t*) {
                if (!subsetPx(sc, x, y)) return;
                static const int pal[5][3] = {{10, 200, 10}, {220, 220, 10}, {10, 10, 220}, {220, 10, 10}, {10, 220, 220}};
                if (f >= 14 && f <= 18) setRgb(px, pal[f - 14][0], pal[f - 14][1], pal[f - 14][2]);
            };
            Result r = runSequence(sc, s, true, nullptr);
            return uiDiffers(sc, r.out[0], sc.frame(nf - 1), 1, [&](int x, int y) { return subsetPx(sc, x, y); });
        };
        Settings re2 = v2, re0 = v2; re0.uiReentryFrames = 0;
        size_t a = runAt(21, re2), b = runAt(20, re2), c = runAt(21, re0), d = runAt(27, re0), e = runAt(28, re0);
        printf(" fast re-entry (arm leaves at 17, back at 19): wrong arm px  re-entry 2: pair(19,20) %zu, pair(18,19) %zu | re-entry 0: pair(19,20) %zu, pair(25,26) %zu, pair(26,27) %zu\n", a, b, c, d, e);
        expect(a == 0, "fast re-entry 2: the arm is protected again on the pair 2 frames after it came back");
        expect(b > 0, "(the pair before it is not: one matching frame is not enough)");
        expect(c > 0, "control: without re-entry the arm is still unprotected there");
        expect(d > 0 && e == 0, "without re-entry the arm needs uiStillFrames = 8 still frames after it came back: protected from pair (26, 27)");
    }

    // 5. the pair gate: the right arm changes colour in the LAST pair only. V2 never pastes a pixel that changed in the pair
    //    (the generated frame shows the previous frame's colour there, not curr's); the rest of the crosshair is exact.
    //    V1's hard grow / half-alpha ring pastes curr onto the changed pixels next to still ones.
    {
        const int NF = 12;
        Scene sc(W, H, NF, 0, 0, 16, 0);
        sc.uiHook = [&](int f, int x, int y, uint8_t* px, const uint8_t*) { if (f == NF - 1 && subsetPx(sc, x, y)) setRgb(px, 10, 200, 10); };
        auto real = sc.frame(NF - 1), prev = sc.frame(NF - 2);
        Result r2 = runSequence(sc, v2, true, nullptr);
        Settings c1 = v2; c1.uiProtection = UiProtection::Crosshair;
        Result r1 = runSequence(sc, c1, true, nullptr);
        Settings vb = v2; vb.blurSamples = 4;
        Result rb = runSequence(sc, vb, true, nullptr, nullptr, true);   // out[1] = the blurred output
        size_t nsub;
        auto subset = [&](int x, int y) { return subsetPx(sc, x, y); };
        auto rest = [&](int x, int y) { return !subsetPx(sc, x, y); };
        // changed pixels equal to curr in the output = pasted
        size_t sameCurr2 = nsub = 0, pasted1 = 0, pasted2 = 0;
        size_t nChanged = 0;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                size_t i = (size_t)y * W + x;
                if (sc.ui[i] != 1 || !subset(x, y)) continue;
                int dp = 0, o1 = 0, o2 = 0;
                for (int c = 0; c < 3; c++) {
                    dp = std::max(dp, std::abs((int)real[i * 4 + c] - (int)prev[i * 4 + c]));
                    o1 = std::max(o1, std::abs((int)r1.out[0][i * 4 + c] - (int)real[i * 4 + c]));
                    o2 = std::max(o2, std::abs((int)r2.out[0][i * 4 + c] - (int)real[i * 4 + c]));
                }
                if (dp <= 3) continue;   // (colours that happen to be close: not a change)
                nChanged++;
                pasted1 += o1 <= 1;
                pasted2 += o2 <= 1;
            }
        (void)sameCurr2;
        size_t restBad = uiDiffers(sc, r2.out[0], real, 1, rest);
        printf(" pair gate: %zu crosshair pixels change in the last pair; pasted (== curr) V1 %zu, V2 %zu; unchanged crosshair wrong V2 %zu\n", nChanged, pasted1, pasted2, restBad);
        expect(nChanged > 0 && pasted2 == 0, "V2: no pixel that changed in the pair is pasted (t = 0.5)");
        expect(restBad == 0, "V2: the unchanged part of the crosshair stays pixel-exact in the same pair");
        {   // recordSynthBlur: the patch after the blend has the same gate
            size_t pastedB = 0, nB = 0;
            for (size_t i = 0; i < (size_t)W * H; i++) {
                if (sc.ui[i] != 1 || !subsetPx(sc, (int)(i % W), 0)) continue;
                int dp = 0, ob = 0;
                for (int c = 0; c < 3; c++) {
                    dp = std::max(dp, std::abs((int)real[i * 4 + c] - (int)prev[i * 4 + c]));
                    ob = std::max(ob, std::abs((int)rb.out[1][i * 4 + c] - (int)real[i * 4 + c]));
                }
                if (dp <= 3) continue;
                nB++;
                pastedB += ob <= 1;
            }
            size_t restBadB = uiDiffers(sc, rb.out[1], real, 1, rest);
            printf(" pair gate, motion blur [0.25, 0.75] x 4: changed pixels pasted %zu/%zu; unchanged crosshair wrong %zu\n", pastedB, nB, restBadB);
            expect(nB > 0 && pastedB == 0, "V2 + motion blur: no pixel that changed in the pair is pasted");
            expect(restBadB == 0, "V2 + motion blur: the unchanged part of the crosshair is pixel-exact");
        }
        expect(pasted1 > 0, "control: V1 pastes curr onto changed pixels (hard grow / half-alpha ring)");
    }

    // 6. runtime switching between the modes (the state layout and the mask format differ) and to Off
    {
        Scene sc(W, H, 12, 0, 0);
        auto real11 = sc.frame(11);
        Settings off;
        Result rOff = runSequence(sc, off, true, nullptr);
        auto sw = [&](const Settings& start, const std::vector<std::pair<int, UiProtection>>& steps) {
            return runSequence(sc, start, true, nullptr, [&](Interpolator& fg, int f) {
                for (auto& st : steps)
                    if (st.first == f) {
                        Settings x = fg.settings();
                        x.uiProtection = st.second;
                        x.uiBoxHeightFrac = 43.0f / H;
                        if (!fg.setSettings(algo(x))) { printf("setSettings failed\n"); exit(1); }
                    }
            });
        };
        Result a = sw(off, {{1, UiProtection::CrosshairV2}});
        expect(uiDiffers(sc, a.out[0], real11, 1, all) == 0, "runtime switch Off -> V2 mid-stream protects the crosshair");
        Result b = sw(v2, {{2, UiProtection::Crosshair}});
        expect(uiDiffers(sc, b.out[0], real11, 1, all) == 0, "runtime switch V2 -> Crosshair (V1): the clean copies are allocated then, the crosshair is exact");
        Result c = sw(v2, {{1, UiProtection::WholeScreen}, {3, UiProtection::CrosshairV2}});
        expect(uiDiffers(sc, c.out[0], real11, 1, all) == 0, "runtime switch V2 -> WholeScreen -> V2: exact");
        Result d = sw(v2, {{9, UiProtection::Off}});
        expect(d.out[0] == rOff.out[0] && d.out[1] == rOff.out[1], "runtime switch V2 -> Off: later output bit-identical to an Off run");
    }

    // 7. validation of the new settings
    {
        Settings bad;
        bad.uiContrastThreshold = -1.0f; expect(!Interpolator::validate(bad), "validate rejects uiContrastThreshold < 0");
        bad = Settings{}; bad.uiContrastThreshold = 256.0f; expect(!Interpolator::validate(bad), "validate rejects uiContrastThreshold > 255");
        bad = Settings{}; bad.uiContrastThreshold = std::nanf(""); expect(!Interpolator::validate(bad), "validate rejects uiContrastThreshold NaN");
        bad = Settings{}; bad.uiLeaveFrames = 0; expect(!Interpolator::validate(bad), "validate rejects uiLeaveFrames 0");
        bad = Settings{}; bad.uiLeaveFrames = 256; expect(!Interpolator::validate(bad), "validate rejects uiLeaveFrames 256");
        bad = Settings{}; bad.uiReentryFrames = 256; expect(!Interpolator::validate(bad), "validate rejects uiReentryFrames 256");
        bad = Settings{}; bad.uiReentryFrames = 0; bad.uiContrastThreshold = 0.0f; expect(Interpolator::validate(bad), "validate accepts uiReentryFrames 0 and uiContrastThreshold 0");
        bad = Settings{}; bad.uiProtection = UiProtection::CrosshairV2; expect(Interpolator::validate(bad), "validate accepts CrosshairV2");
        bad.uiProtection = (UiProtection)4; expect(!Interpolator::validate(bad), "validate rejects mode 4");
    }
    ghostCheck();
    return fails ? 1 : 0;
}
#endif

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
            if (!fg.setSettings(algo(lo))) { printf("setSettings(lower) failed\n"); exit(1); }
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
            if (!fg.setSettings(algo(s))) { printf("setSettings(raise) failed\n"); exit(1); }
        }
        vkDeviceWaitIdle(dev);
        fg.destroy();
        for (auto& r : ring) { vkDestroyImageView(dev, r.view, nullptr); vkDestroyImage(dev, r.img, nullptr); vkFreeMemory(dev, r.mem, nullptr); }
        vkDestroyImageView(dev, out.view, nullptr); vkDestroyImage(dev, out.img, nullptr); vkFreeMemory(dev, out.mem, nullptr);
        Metrics m = measure(sc, res, realB, nullptr);
        printf(" %-22s crosshair wrong %zu/%zu, hud wrong %zu/%zu\n", name, m.xhBad, m.xhN, m.hudBad, m.hudN);
        return res;
    };
    for (UiProtection mode : {UiProtection::Crosshair, UiProtection::WholeScreen UT_V2_MODE}) {
        const char* nm = modeName((int)mode);
        Settings off, s3, s10, s14;
        s3.uiProtection = s10.uiProtection = s14.uiProtection = mode;
        s3.uiBoxHeightFrac = s10.uiBoxHeightFrac = s14.uiBoxHeightFrac = 43.0f / H;   // fitted to the 35 px test crosshair (the default 0.025 is only 9 px at 360 high)
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


// ---------------------------------------------------------------- recordTrack (pass-through keeps the UI counters warm)
#ifndef UITEST_NO_UI
// The UI of the real scene CHANGES when the host stops generating (frame `pre`): the crosshair is recoloured and the HUD
// inverted, then both stay static. A host that does not look at the gap frames sees the old picture on one side of the
// gap and the new one on the other, so its counters start at 0 after the resume; one that tracks them has counted every
// gap frame. Sequence: `pre` observed frames (each pair estimated + synthesised), `gap` frames handled per gapMode
// (0 = nothing, 1 = recordTrack, 2 = recordObserve), then 2 frames observed + the pair (N, N+1) synthesised at t = 0.5:
// the FIRST generated frame after the resume, returned with the real frame N+1 it is judged against.
static std::vector<uint8_t> trackedFrame(const Scene& sc, int f, int altFrom) {
    auto fr = sc.frame(f);
    if (f >= altFrom)
        for (size_t i = 0; i < (size_t)sc.W * sc.H; i++) {
            uint8_t* p = &fr[i * 4];
            if (sc.ui[i] == 1 && (p[0] | p[1] | p[2])) { p[0] = 64; p[1] = 255; p[2] = 64; }
            else if (sc.ui[i] == 2) { p[0] = 255 - p[0]; p[1] = 255 - p[1]; p[2] = 255 - p[2]; }
        }
    return fr;
}
struct TrackRun { std::vector<uint8_t> out, real; bool trackOk = true; };
static TrackRun trackRun(const Scene& sc, const Settings& s, int pre, int gap, int gapMode, int altFrom = -2) {
    if (altFrom == -2) altFrom = pre;   // default: the UI changes at the first gap frame; INT_MAX = never
    Interpolator fg;
    if (!makeInterp(fg, sc.W, sc.H, s)) { fprintf(stderr, "init failed\n"); exit(1); }
    Img ring[4];
    for (auto& r : ring) r = makeImage(sc.W, sc.H, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    Img out = makeImage(sc.W, sc.H, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    begin();
    for (auto& r : ring) imgBarrier(r.img, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    imgBarrier(out.img, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    submit();
    TrackRun res;
    const int total = pre + gap + 2;
    for (int f = 0; f < total; f++) {
        Img& cur = ring[f % 4];
        auto fr = trackedFrame(sc, f, altFrom);
        upload(fr, cur, sc.W, sc.H);
        begin();
        fg.beginFrame();
        const bool inGap = f >= pre && f < pre + gap;
        if (inGap && gapMode == 0) { /* the host does nothing with this frame */ }
        else if (inGap && gapMode == 1) res.trackOk = fg.recordTrack(cb, cur.view) && res.trackOk;
        else if (!fg.recordObserve(cb, cur.view)) { fprintf(stderr, "observe failed\n"); exit(1); }
        if ((f > 0 && f < pre) || f >= pre + gap + 1) {   // both frames of the pair are observed
            Img& prv = ring[(f - 1) % 4];
            if (!fg.recordEstimate(cb, prv.view, cur.view) || !fg.recordSynth(cb, prv.view, cur.view, out.view, 0.5f)) { fprintf(stderr, "synth failed\n"); exit(1); }
        }
        submit();
        if (f == total - 1) { res.out = download(out, sc.W, sc.H); res.real = fr; }
    }
    vkDeviceWaitIdle(dev);
    fg.destroy();
    for (auto& r : ring) { vkDestroyImageView(dev, r.view, nullptr); vkDestroyImage(dev, r.img, nullptr); vkFreeMemory(dev, r.mem, nullptr); }
    vkDestroyImageView(dev, out.view, nullptr); vkDestroyImage(dev, out.img, nullptr); vkFreeMemory(dev, out.mem, nullptr);
    return res;
}

static int trackCheck(int W, int H) {
    Scene sc(W, H, 44, 0, 0);
    printf("track %dx%d: crosshair %zu px, hud box %zu px\n", W, H, (size_t)std::count(sc.ui.begin(), sc.ui.end(), 1), (size_t)std::count(sc.ui.begin(), sc.ui.end(), 2));
    const int PRE = 14, GAP = 20;
    for (UiProtection mode : {UiProtection::Crosshair, UiProtection::WholeScreen UT_V2_MODE}) {
        const bool ws = mode == UiProtection::WholeScreen;
        // V2 keeps a masked pixel through a gap (hysteresis): a recoloured crosshair is protected on the first pair after the gap
        // even without tracking, so the V1 "control" expectations and the exact-count thresholds do not apply to it
        const bool v2 = (int)mode == 3;
        Settings s; s.uiProtection = mode;
        s.uiBoxHeightFrac = 43.0f / H;   // the 35 px test crosshair needs a fitted box (the default 0.025 is 9 px at 360 high)
        printf("%s:\n", modeName((int)mode));
        auto bad = [&](const TrackRun& r) { return measure(sc, r.out, r.real, nullptr); };
        TrackRun ctl = trackRun(sc, s, PRE, GAP, 0);
        TrackRun trk = trackRun(sc, s, PRE, GAP, 1);
        TrackRun obs = trackRun(sc, s, PRE, GAP, 2);
        TrackRun st = trackRun(sc, s, 0, GAP, 1);   // tracking from the very first frame, before any observe
        TrackRun stc = trackRun(sc, s, 0, GAP, 0);
        Metrics mc = bad(ctl), mt = bad(trk), mo = bad(obs), ms = bad(st), msc = bad(stc);
        printf(" control (gap, nothing)    crosshair wrong %zu/%zu, hud wrong %zu/%zu\n", mc.xhBad, mc.xhN, mc.hudBad, mc.hudN);
        printf(" recordTrack x%d            crosshair wrong %zu/%zu, hud wrong %zu/%zu\n", GAP, mt.xhBad, mt.xhN, mt.hudBad, mt.hudN);
        printf(" recordObserve x%d (ref)    crosshair wrong %zu/%zu, hud wrong %zu/%zu\n", GAP, mo.xhBad, mo.xhN, mo.hudBad, mo.hudN);
        printf(" tracked from frame 0      crosshair wrong %zu/%zu, hud wrong %zu/%zu  (control without: %zu/%zu, hud %zu)\n", ms.xhBad, ms.xhN, ms.hudBad, ms.hudN, msc.xhBad, msc.xhN, msc.hudBad);
        // informational: a UI that does NOT change during the gap keeps its counters across it even without tracking
        // (they hold the last frame seen before the gap; unchanged pixels compare equal on the other side)
        {
            Metrics mu = bad(trackRun(sc, s, PRE, GAP, 0, 1 << 30)), mut = bad(trackRun(sc, s, PRE, GAP, 1, 1 << 30));
            printf(" UI unchanged in the gap: no tracking crosshair wrong %zu, hud wrong %zu | tracking crosshair wrong %zu, hud wrong %zu\n", mu.xhBad, mu.hudBad, mut.xhBad, mut.hudBad);
        }
        expect(trk.trackOk && st.trackOk, "recordTrack returns true");
        if (!v2) expect(mc.xhBad > 0, "control: without tracking the first synth after the gap has wrong crosshair pixels (test has teeth)");
        if (ws) expect(mc.hudBad > 0, "control: without tracking the first synth after the gap has wrong HUD pixels");
        expect(mt.xhBad == 0, "recordTrack x20: crosshair pixel-exact (<= 1 level) on the very first synth after the resume");
        if (ws) expect(mt.hudBad == 0, "recordTrack x20: HUD box pixel-exact on the very first synth after the resume");
        expect(ms.xhBad == 0 && (!ws || ms.hudBad == 0), "tracking from the first frame (nothing observed yet): exact after the resume");
        expect(msc.xhBad > 0, "control from frame 0: without tracking the first synth is wrong");
        expect(trk.out == obs.out, "equivalence: first synth after 20 tracked frames is bit-identical to after 20 observed frames");
        // count-exactness: the counters must reach uiStillFrames (8) at the right frame. Gap g: the new crosshair has count g
        // at frame N and g + 1 at N + 1 (the frame the mask is read from): masked from g = 7.
        for (int g : {6, 7}) {
            TrackRun a = trackRun(sc, s, PRE, g, 1), b = trackRun(sc, s, PRE, g, 2);
            Metrics ma = bad(a);
            printf(" gap %d: tracked crosshair wrong %zu/%zu\n", g, ma.xhBad, ma.xhN);
            expect(a.out == b.out, g == 6 ? "equivalence at the threshold, 6 tracked frames == 6 observed (not yet protected)" : "equivalence at the threshold, 7 tracked frames == 7 observed (protected)");
            if (v2) continue;
            if (g == 6) expect(ma.xhBad > 0, "6 tracked frames: counters at 7 of 8, crosshair not yet protected (counts exactly)");
            else expect(ma.xhBad == 0, "7 tracked frames: counters at 8 of 8, crosshair protected (counts exactly)");
        }
        if (v2) {   // tracked from the start (no observe before): the state machine counts the tracked frames like observed ones
            TrackRun a = trackRun(sc, s, 0, 6, 1), b = trackRun(sc, s, 0, 6, 2), c = trackRun(sc, s, 0, 7, 1), d = trackRun(sc, s, 0, 7, 2);
            expect(a.out == b.out && c.out == d.out, "equivalence from the first frame: 6 / 7 tracked frames == 6 / 7 observed");
            expect(bad(a).xhBad > 0 && bad(c).xhBad == 0, "from a cold start the crosshair needs uiStillFrames = 8 frames counted: tracking counts them exactly");
        }
    }
    // Off: recordTrack is a no-op returning true; the output is byte-identical to a run that never calls it
    {
        Settings off;
        TrackRun a = trackRun(sc, off, PRE, GAP, 1), b = trackRun(sc, off, PRE, GAP, 0);
        expect(a.trackOk, "Off: recordTrack returns true");
        expect(a.out == b.out, "Off: output byte-identical with and without recordTrack calls");
    }
    return fails ? 1 : 0;
}

// ---- cost: GPU time per call, back to back (8 per command buffer), moving scene; observe = the same call with recordObserve
static void trackBench(int W, int H) {
    const int NIMG = 16;
    Scene sc(W, H, NIMG, 0, 0);
    printf("\n== %dx%d ==\n", W, H);
    Img ring[NIMG];
    for (auto& r : ring) r = makeImage(W, H, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    begin();
    for (auto& r : ring) imgBarrier(r.img, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    submit();
    for (int i = 0; i < NIMG; i++) upload(sc.frame(i), ring[i], W, H);
    for (UiProtection mode : {UiProtection::Crosshair, UiProtection::WholeScreen UT_V2_MODE}) {
        Settings s; s.uiProtection = mode;
        double ms[2] = {0, 0};
        for (int track = 0; track < 2; track++) {
            Interpolator fg;
            if (!makeInterp(fg, W, H, s)) { fprintf(stderr, "init failed\n"); exit(1); }
            VkQueryPoolCreateInfo qp{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qp.queryType = VK_QUERY_TYPE_TIMESTAMP; qp.queryCount = 2;
            VkQueryPool qpool;
            CK(vkCreateQueryPool(dev, &qp, nullptr, &qpool));
            std::vector<double> v;
            for (int rep = 0; rep < 14; rep++) {
                begin();
                fg.beginFrame();
                vkCmdResetQueryPool(cb, qpool, 0, 2);
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);
                for (int k = 0; k < 8; k++) {
                    bool ok = track ? fg.recordTrack(cb, ring[(rep * 8 + k) % NIMG].view) : fg.recordObserve(cb, ring[(rep * 8 + k) % NIMG].view);
                    if (!ok) { fprintf(stderr, "record failed\n"); exit(1); }
                }
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 1);
                submit();
                uint64_t t2[2];
                CK(vkGetQueryPoolResults(dev, qpool, 0, 2, sizeof(t2), t2, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
                if (rep >= 4) v.push_back((t2[1] - t2[0]) * tsPeriodMs / 8);
            }
            std::sort(v.begin(), v.end());
            ms[track] = v[v.size() / 2];
            vkDestroyQueryPool(dev, qpool, nullptr);
            vkDeviceWaitIdle(dev);
            fg.destroy();
        }
        printf("  %-12s recordObserve %.4f ms   recordTrack %.4f ms   (%.0f %% of an observe)\n", modeName((int)mode), ms[0], ms[1], 100.0 * ms[1] / ms[0]);
    }
    for (auto& r : ring) { vkDestroyImageView(dev, r.view, nullptr); vkDestroyImage(dev, r.img, nullptr); vkFreeMemory(dev, r.mem, nullptr); }
}
#endif

// ---------------------------------------------------------------- bench
#ifndef UITEST_NO_UI
static void bench(int W, int H) {
    const int NIMG = 16;
    Scene sc(W, H, NIMG, 0, 0);
    printf("\n== %dx%d ==\n", W, H);
    struct Mode { const char* name; Settings s; };
    Settings sOff, sC, sW, sWn, sV, sVf;
    sC.uiProtection = UiProtection::Crosshair;
    sV.uiProtection = UT_V2_ENUM;
    sW.uiProtection = UiProtection::WholeScreen;
    sC.uiBoxHeightFrac = sW.uiBoxHeightFrac = sV.uiBoxHeightFrac = 43.0f / H;   // box fitted to the 35 px test crosshair
    sVf = sV;
#ifndef UITEST_NO_V2
    sVf.uiEstimateFill = true;
#endif
    sWn = sW; sWn.uiMaxStillFraction = 1.0f;
    Mode modes[] = {{"Off", sOff}, {"Crosshair", sC}, {"CrosshairV2", sV}, {"CrosshairV2 + fill", sVf}, {"WholeScreen", sW}, {"WholeScreen, no still-skip", sWn}};
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
            if (!strncmp(md.name, "CrosshairV2", 11) && md.s.uiProtection == UiProtection::Off) continue;   // (built without V2)
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

#ifndef UITEST_NO_MOTION
// ---------------------------------------------------------------- the motion version (Settings::motionVersion)
// The pipelines the library holds, counted through the gdpa it loads its device functions with (InitInfo::gdpa): created - destroyed.
static int64_t gLivePipes = 0;
static VKAPI_ATTR VkResult VKAPI_CALL countCreatePipes(VkDevice d, VkPipelineCache c, uint32_t n, const VkComputePipelineCreateInfo* ci,
                                                      const VkAllocationCallbacks* a, VkPipeline* out) {
    VkResult r = vkCreateComputePipelines(d, c, n, ci, a, out);
    if (r == VK_SUCCESS) gLivePipes += n;
    return r;
}
static VKAPI_ATTR void VKAPI_CALL countDestroyPipe(VkDevice d, VkPipeline p, const VkAllocationCallbacks* a) {
    if (p) gLivePipes--;
    vkDestroyPipeline(d, p, a);
}
static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL countingGdpa(VkDevice d, const char* name) {
    if (!strcmp(name, "vkCreateComputePipelines")) return (PFN_vkVoidFunction)countCreatePipes;
    if (!strcmp(name, "vkDestroyPipeline")) return (PFN_vkVoidFunction)countDestroyPipe;
    return vkGetDeviceProcAddr(d, name);
}
static const char* algoName(MotionVersion v) { return v == MotionVersion::V2 ? "V2" : "V1"; }
// The pipelines of one motion version's group (framegen.cpp MotionGroup): V1 = 11 estimate / synth passes + 3 motion-blur passes.
// V2 = both V2Modes: V1's luma / down, its 2 faster estimate passes (medhist, search), the block re-pick and field clean-up (both
// modes), Performance's morph, Quality's per-pixel field, splat, coverage and morph; no motion-blur passes (its morphs blur): 11.
static int64_t versionPipes(MotionVersion v) { return v == MotionVersion::V2 ? 11 : 14; }

// --algo picks the version the run starts in (`from`), the other one is switched to (`to`). Frames 10 and 11 of the check scene,
// protection Off, t = 0.5.
//   V2 needs Rgba8: settingsError() refuses it for every other Format with a reason, init() refuses it with that reason.
//   A switch drops the estimate: recordSynth / recordSynthBlur of the pair return false until a new recordEstimate, also after
//   switching straight back (an estimate of one version is never synthesised by the other); setSettings without a switch keeps it.
//   The new version's output equals a fresh instance's in that version.
//   releaseUnusedMotion(): the first switch to a version creates its pipelines (versionPipes()), the release frees the unselected
//   version's (and nothing while there is none), selecting it again re-creates them and gives the same output as before; destroy()
//   frees all.
//   Settings::v2Mode: validated, runtime (setSettings creates nothing), a switch of the mode alone keeps the estimate; V1 ignores it.
static void motionCheck(const Scene& sc) {
    const int W = sc.W, H = sc.H;
    const MotionVersion from = gAlgo, to = gAlgo == MotionVersion::V1 ? MotionVersion::V2 : MotionVersion::V1;
    printf("\n== motion version: start %s, switch to %s ==\n", algoName(from), algoName(to));
    {   // V2 needs Rgba8
        Settings v1, v2; v2.motionVersion = MotionVersion::V2;
        Settings bad; bad.motionVersion = (MotionVersion)2;
        expect(Interpolator::validate(v2) && !Interpolator::validate(bad), "validate accepts V2 and rejects an unknown motion version");
        const Format others[4] = {Format::Rgb10, Format::RgbaF16, Format::Nv12, Format::P010};
        bool v1All = !Interpolator::settingsError(v1, Format::Rgba8), v2Refused = true;
        for (Format f : others) {
            v1All = v1All && !Interpolator::settingsError(v1, f);
            v2Refused = v2Refused && Interpolator::settingsError(v2, f) != nullptr;
        }
        expect(v1All && !Interpolator::settingsError(v2, Format::Rgba8), "settingsError: V1 valid for every Format, V2 for Rgba8");
        expect(v2Refused, "settingsError: V2 refused for Rgb10, RgbaF16, Nv12 and P010");
        printf("    reason: %s\n", Interpolator::settingsError(v2, Format::Rgb10));
        expect(Interpolator::settingsError(bad, Format::Rgba8) != nullptr, "settingsError refuses what validate refuses");
        Settings mq = v2, mp = v2, mb = v2, m1 = v1;
        mp.v2Mode = V2Mode::Performance; mb.v2Mode = (V2Mode)2; m1.v2Mode = V2Mode::Performance;
        expect(Interpolator::validate(mq) && Interpolator::validate(mp) && !Interpolator::validate(mb) && Interpolator::validate(m1),
               "validate: v2Mode Quality / Performance accepted (also with V1, which ignores it), an unknown mode rejected");
        expect(Settings{}.v2Mode == V2Mode::Quality, "v2Mode defaults to Quality");
        expect(Interpolator::settingsError(mp, Format::Rgb10) && !strcmp(Interpolator::settingsError(mp, Format::Rgb10), Interpolator::settingsError(v2, Format::Rgb10)),
               "settingsError: V2 Performance refused for Rgb10 with the same reason");
        // init() of a non-SDR view format with V2: false before anything is created, with that reason (RgbaF16 also linear)
        const VkFormat vfs[5] = {VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
                                 VK_FORMAT_R8_UNORM, VK_FORMAT_R16_UNORM};
        bool initRefused = true;
        for (int i = 0; i < 5; i++) {
            Interpolator fg;
            InitInfo info;
            info.phys = phys; info.device = dev; info.gdpa = vkGetDeviceProcAddr; info.width = W; info.height = H;
            info.viewFormat = vfs[i]; info.getMemProps = vkGetPhysicalDeviceMemoryProperties; info.settings = v2;
            if (i == 2) info.transfer = Transfer::Linear;
            Format f;
            formatOf(vfs[i], &f);
            const bool ok = fg.init(info);
            const char* e = fg.initError();
            initRefused = initRefused && !ok && e && (!strcmp(e, Interpolator::settingsError(v2, f)) || strstr(e, "not built"));
            fg.destroy();
        }
        expect(initRefused, "init refuses V2 for Rgb10, RgbaF16 (encoded and linear), Nv12 and P010, initError = settingsError's reason");
    }

    Img a = makeImage(W, H, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    Img b = makeImage(W, H, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    Img o = makeImage(W, H, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    begin();
    for (Img* i : {&a, &b, &o}) imgBarrier(i->img, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    submit();
    upload(sc.frame(10), a, W, H);
    upload(sc.frame(11), b, W, H);
    // estimate (optional) + synth at 0.5 into o; returns the synth's result, the output is downloaded only when it succeeded
    auto run = [&](Interpolator& fg, bool estimate, std::vector<uint8_t>* out) {
        begin();
        fg.beginFrame();
        bool ok = (!estimate || fg.recordEstimate(cb, a.view, b.view)) && fg.recordSynth(cb, a.view, b.view, o.view, 0.5f);
        submit();
        if (ok && out) *out = download(o, W, H);
        return ok;
    };
    auto blurRefused = [&](Interpolator& fg) {
        begin();
        fg.beginFrame();
        bool ok = fg.recordSynthBlur(cb, a.view, b.view, o.view, 0.25f, 0.75f);
        submit();
        return !ok;
    };
    Settings sFrom, sTo;
    sFrom.motionVersion = from; sTo.motionVersion = to;
    sFrom.v2Mode = sTo.v2Mode = gMode;   // --mode
    sFrom.blurSamples = sTo.blurSamples = 4;   // recordSynthBlur blurs (else it is recordSynth)
    std::vector<uint8_t> outA, outB, outC, outR, outD;
    {   // the reference: a fresh instance in `to`
        Interpolator ref;
        InitInfo info;
        info.phys = phys; info.device = dev; info.gdpa = vkGetDeviceProcAddr; info.width = W; info.height = H;
        info.viewFormat = fmt(); info.getMemProps = vkGetPhysicalDeviceMemoryProperties; info.settings = sTo;
        if (!ref.init(info) || !run(ref, true, &outR)) { expect(false, "reference instance"); }
        vkDeviceWaitIdle(dev);
        ref.destroy();
    }
    gLivePipes = 0;
    Interpolator fg;
    InitInfo info;
    info.phys = phys; info.device = dev; info.gdpa = countingGdpa; info.width = W; info.height = H;
    info.viewFormat = fmt(); info.getMemProps = vkGetPhysicalDeviceMemoryProperties; info.settings = sFrom;
    if (!fg.init(info)) { expect(false, "init with the counting gdpa"); return; }
    const int64_t p0 = gLivePipes;
    expect(run(fg, true, &outA), "estimate + synth in the start version");
    expect(fg.setSettings(sFrom) && run(fg, false, &outB) && outB == outA, "setSettings without a switch keeps the estimate (same output)");
    expect(fg.setSettings(sTo), "switch accepted");
    const int64_t k = gLivePipes - p0;
    printf("    pipelines: %lld after init, +%lld for the first switch to %s\n", (long long)p0, (long long)k, algoName(to));
    expect(k == versionPipes(to), "the first switch creates the new version's pipelines (V1: 14, V2: 11)");
    expect(!run(fg, false, nullptr) && blurRefused(fg), "after the switch the old estimate is gone: recordSynth and recordSynthBlur return false");
    expect(fg.setSettings(sFrom) && gLivePipes == p0 + k, "switching back creates nothing (the start version kept its pipelines)");
    expect(!run(fg, false, nullptr), "switching straight back does not revive the dropped estimate");
    expect(fg.setSettings(sTo) && run(fg, true, &outC), "a new estimate in the new version synthesises");
    expect(outC == outR, "the new version's output equals a fresh instance's in that version");
    // both V2 modes morph (their own syntheses on V1's block field, computed faster)
    expect(outC != outA, gMode == V2Mode::Quality ? "V2 Quality's output differs from V1's (its own synthesis)"
                                                  : "V2 Performance's output differs from V1's (its own synthesis)");
    vkDeviceWaitIdle(dev);
    fg.releaseUnusedMotion();
    expect(gLivePipes == p0 + k - versionPipes(from), "releaseUnusedMotion frees the unselected version's pipelines");
    fg.releaseUnusedMotion();
    expect(gLivePipes == p0 + k - versionPipes(from), "a second releaseUnusedMotion frees nothing (nothing unused)");
    expect(run(fg, true, &outC) && outC == outR, "the selected version still works after the release");
    expect(fg.setSettings(sFrom) && gLivePipes == p0 + k, "selecting the released version again re-creates its pipelines");
    expect(run(fg, true, &outD) && outD == outA, "the re-created version gives the same output as before the release");
    vkDeviceWaitIdle(dev);
    fg.releaseUnusedMotion();
    expect(gLivePipes == p0, "release after switching back: the other version freed");
    fg.destroy();
    expect(gLivePipes == 0, "destroy frees every pipeline");

    {   // Settings::v2Mode: runtime (no pipelines created, no rebuild), a switch of the mode alone keeps the estimate (the other mode's
        // part is derived from it by its first synth: the same output as with that mode's own estimate); a switch of the version still
        // drops it; V1 ignores the mode. Each mode (and V1) gives its own output.
        Settings q; q.motionVersion = MotionVersion::V2; q.v2Mode = V2Mode::Quality;
        Settings p = q; p.v2Mode = V2Mode::Performance;
        Settings v1p = p; v1p.motionVersion = MotionVersion::V1;
        Settings v1q = v1p; v1q.v2Mode = V2Mode::Quality;
        gLivePipes = 0;
        Interpolator f;
        InitInfo inf;
        inf.phys = phys; inf.device = dev; inf.gdpa = countingGdpa; inf.width = W; inf.height = H;
        inf.viewFormat = fmt(); inf.getMemProps = vkGetPhysicalDeviceMemoryProperties; inf.settings = q;
        std::vector<uint8_t> oq, op, op2, oq2, oq3, o1p, o1q;
        expect(f.init(inf) && run(f, true, &oq), "v2Mode: V2 Quality estimate + synth");
        const int64_t pm = gLivePipes;
        expect(f.setSettings(p) && f.settings().v2Mode == V2Mode::Performance && gLivePipes == pm,
               "v2Mode: setSettings switches Quality -> Performance at runtime and creates no pipeline (no rebuild)");
        expect(run(f, false, &op), "v2Mode: the switch of the mode alone keeps the estimate (Performance synthesises Quality's estimate)");
        expect(run(f, true, &op2) && op2 == op, "v2Mode: ... with the same output as Performance's own estimate");
        expect(op != oq, "v2Mode: Performance's output differs from Quality's (its own synthesis)");
        expect(f.setSettings(q) && run(f, false, &oq2) && oq2 == oq, "v2Mode: Performance -> Quality keeps the estimate too (Quality's output)");
        expect(f.setSettings(p) && run(f, false, &op) && op == op2 && f.setSettings(q) && run(f, false, &oq3) && oq3 == oq,
               "v2Mode: switching back and forth on one estimate keeps each mode's output");
        {   // the same through recordSynthBlur (4 samples): each mode's part derived on demand inside the blur record too
            auto runBlur = [&](bool estimate, std::vector<uint8_t>* out) {
                begin();
                f.beginFrame();
                bool ok = (!estimate || f.recordEstimate(cb, a.view, b.view)) && f.recordSynthBlur(cb, a.view, b.view, o.view, 0.25f, 0.75f);
                submit();
                if (ok && out) *out = download(o, W, H);
                return ok;
            };
            Settings qb = q, pb = p;
            qb.blurSamples = pb.blurSamples = 4;
            std::vector<uint8_t> bq, bp, bq2, bp2;
            expect(f.setSettings(qb) && runBlur(true, &bq) && f.setSettings(pb) && runBlur(false, &bp2) && runBlur(true, &bp) && bp2 == bp &&
                   f.setSettings(qb) && runBlur(false, &bq2) && bq2 == bq && bp != bq,
                   "v2Mode: recordSynthBlur after a switch of the mode alone gives that mode's own blur (both directions)");
            expect(f.setSettings(q), "v2Mode: blur off again");
        }
        Settings bad = q; bad.v2Mode = (V2Mode)2;
        expect(!f.setSettings(bad) && f.settings().v2Mode == V2Mode::Quality, "v2Mode: setSettings refuses an unknown mode and changes nothing");
        expect(f.setSettings(v1p) && !run(f, false, nullptr) && blurRefused(f), "v2Mode: a switch of the version (V2 -> V1) still drops the estimate");
        expect(run(f, true, &o1p) && o1p != oq && o1p != op, "v2Mode: V1's output differs from both V2 modes'");
        expect(f.setSettings(v1q) && run(f, false, &o1q) && o1q == o1p, "v2Mode: V1 ignores the mode (a switch keeps its estimate, same output)");
        vkDeviceWaitIdle(dev);
        f.destroy();
        expect(gLivePipes == 0, "v2Mode: destroy frees every pipeline");
    }
    for (Img* i : {&a, &b, &o}) { vkDestroyImageView(dev, i->view, nullptr); vkDestroyImage(dev, i->img, nullptr); vkFreeMemory(dev, i->mem, nullptr); }
}
#endif

int main(int argc, char** argv) {
    std::vector<std::string> pos;
    bool hash = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bgra")) bgra = true;
#ifndef UITEST_NO_MOTION
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
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dumpDir = argv[++i];
        else if (!strcmp(argv[i], "--hash")) hash = true;
        else pos.push_back(argv[i]);
    }
    if (pos.empty() || (pos[0] != "check" && pos[0] != "bench" && pos[0] != "deep" && pos[0] != "track" && pos[0] != "trackbench")) { fprintf(stderr, "usage: uitest check|bench|deep|track|trackbench [--bgra] [--hash] [--algo v1|v2] [--mode quality|performance] [W H]\n"); return 2; }
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
    if (pos[0] == "track" || pos[0] == "trackbench") {
#ifdef UITEST_NO_UI
        fprintf(stderr, "track needs the UI API\n");
        return 2;
#else
        if (pos[0] == "track") {
            int W = pos.size() >= 3 ? atoi(pos[1].c_str()) : 640, H = pos.size() >= 3 ? atoi(pos[2].c_str()) : 360;
            initStaging((size_t)W * H * 4);
            rc = trackCheck(W, H);
            printf(rc ? "RESULT: FAIL (%d)\n" : "RESULT: PASS\n", fails);
        } else {
            initStaging((size_t)2560 * 1440 * 4);
            if (pos.size() >= 3) trackBench(atoi(pos[1].c_str()), atoi(pos[2].c_str()));
            else { trackBench(1920, 1080); trackBench(2560, 1440); }
        }
#endif
    } else if (pos[0] == "deep") {
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
        initStaging(std::max((size_t)W * H, (size_t)1280 * 960) * 4);   // ghostCheck() runs its own 1280x960 scenes
#ifndef UITEST_NO_MOTION
        printf("motion version %s (--algo), v2Mode %s (--mode)\n", algoName(gAlgo), gMode == V2Mode::Performance ? "performance" : "quality");
#endif
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
