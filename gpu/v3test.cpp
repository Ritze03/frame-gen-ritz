// v3test: the known-motion test of MotionVersion::V3 (gpu/V3.md section 11). Procedural scenes whose motion is known exactly are
// rendered on the CPU (the two real frames, and the true in-between frame R_t for any t); V3's estimated flows (recordDebugV3) and its
// in-between frames (recordSynth) are measured against them, once with V3's own estimate and once with the exact flows injected
// (recordDebugV3Inject).
//   v3test [--size WxH] [--fs 1|2|4] [--scene NAME|all] [--t T1,T2,..] [--report-only] [--selfcheck] [--colour-window] [--detail-split] [--blend 0..kV3BlendMax] [--guard off|guard|v2motion] [--v3-motion 0|1] [--list]
//     --colour-window   Settings::v3ColourWindow on (hook (j)); the pass marks stay the same (information with the hook on)
//     --detail-split    Settings::v3DetailSplit on (hook (l)); the marks stay the same (information with the hook on)
//     --blend N         Settings::v3Blend 0..kV3BlendMax (learned blend, experimental; 1 Fluidity, 2 Consistency); the marks stay the same (information)
//     --v3-motion N     Settings::v3Motion 0 (V3, default) | 1 (V3.5: the regularised motion field); the marks stay the same (information with 1)
//     --guard MODE      Settings::v3Guard off (default) | guard | v2motion (task 104); the marks stay the same (information with it on)
//     --size   frame size (default 1920x1080). The scenes' lengths (shifts, layer sizes) are those of 1080p scaled by min(W/1920, H/1080);
//              the texture (wavelengths 4..128 px) is not.
//     --fs     flowScale (1, 2 or 4; default 2)
//     --scene  shift | zoom | rotate | layers | fade | stripes | border | motion | all (default all; motion = task 160: Settings::v3Motion switched at run time, v3MotionActive(); border = task 109 frame-border injection scenes; stripes = repetitive texture, informational, not in all). shift = 7 shifts, layers = a rectangle and a zooming disc on a background.
//     --t      times of the in-between frames measured (default 0.25,0.5)
//     --report-only   always exit 0 (until the later steps of V3 make the metrics pass)
//     --selfcheck     verify the tool itself instead of V3: (i) the CPU warp of prev by the exact F10 (and of curr by F01) reproduces curr (prev)
//                     on the visible pixels (mean error <= 2.5 levels), (ii) R_0 = prev and R_1 = curr exactly, and no pixel is "hidden" by the
//                     prev side at t = 0 / the curr side at t = 1, (iii) the exact flows injected through recordDebugV3Inject are read back by recordDebugV3
//                     bit-exact (FD), (iv) the ideal gather blend with the exact layer motion reproduces R_t (mean <= 1.0 level) on the non-hidden pixels
// Scenes (positions at time tau in [0, 1], prev: tau = 0, curr: tau = 1): every layer has a band-limited random texture (6 octaves of value
// noise, wavelengths 4..128 px, per channel, plus sharp-edged rectangles and ellipses) and moves by x(tau) = pivot + s(tau) R(theta tau)
// (u - pivot) + tau shift. A frame is the layers' top-most colour at 2x2 sub-samples per pixel, averaged and rounded to 8 bit; the same function
// renders R_t at any t. Exact data: F01(p) / F10(q) at the flow pixels' centres (layer visible there; flow = layer motion at that position / fs),
// the layer maps id0 / id1.
// Visible flow pixel (F01 at p; F10 symmetric): p and p + F01(p) inside the frame, the same layer visible at the partner, and no layer boundary within
// 2 flow px of p or of the partner. Hidden pixel of R_t: the layer there is not the visible layer at x - t m or at x + (1 - t) m (or that place is outside
// the frame), dilated by 3 px, plus pixels within 2 px of a layer boundary at t.
// Metrics: (a) flow: share of visible flow pixels with fs |F_est - F_true| <= 1.0 px (also 0.5, 2.0); pass >= 95% (fade: 90%). (b) in-between: mean over
// the non-hidden pixels of the mean over channels |out - R_t| (levels), 99th percentile, share <= 3 levels; with the injected exact flows pass <= 1.0;
// with V3's own estimate pass <= 2.0 (shift only: informational for the other scenes).
#include "framegen.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d\n", #x, r_); exit(1); } } while (0)

// ================================================================= the scene model (CPU)
struct V2 { double x, y; };

static inline uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t h = a * 374761393u + b * 668265263u + c * 2246822519u + 0x9E3779B9u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h = (h ^ (h >> 16)) * 2246822519u;
    return h ^ (h >> 13);
}
static inline float u01(uint32_t h) { return (float)(h & 0xFFFFFFu) * (1.0f / 16777216.0f); }

// value noise of wavelength lam at (x, y), smooth (smoothstep) interpolation of lattice values in [0, 1)
static inline float vnoise(float x, float y, float lam, uint32_t seed) {
    const float sx = x / lam, sy = y / lam;
    const float fxl = std::floor(sx), fyl = std::floor(sy);
    const int xi = (int)fxl, yi = (int)fyl;
    float fx = sx - fxl, fy = sy - fyl;
    fx = fx * fx * (3.0f - 2.0f * fx);
    fy = fy * fy * (3.0f - 2.0f * fy);
    auto v = [&](int a, int b) { return u01(hash3((uint32_t)(xi + a), (uint32_t)(yi + b), seed)); };
    const float a = v(0, 0) + (v(1, 0) - v(0, 0)) * fx, b = v(0, 1) + (v(1, 1) - v(0, 1)) * fx;
    return a + (b - a) * fy;
}

struct Layer {
    int kind = 0;                                   // mask in layer coordinates: 0 the whole plane, 1 rectangle, 2 ellipse
    double mcx = 0, mcy = 0, mhw = 0, mhh = 0;      // centre and half sizes of the mask
    double px = 0, py = 0;                          // pivot (layer coordinates = screen coordinates at tau = 0)
    double scaleRate = 0, rotDeg = 0, shx = 0, shy = 0;
    double period = 0;                              // > 0: the texture is modulated by vertical stripes of this period (px): a repetitive structure (the probes' test)
    bool solid = false;                             // a flat colour instead of the texture (overlays: kind 3 / 4)
    float srgb[3] = {255, 255, 255};
    double gamma = 1.0;                             // the texture's levels to the power gamma (a nonlinear change of the histogram: the scene-cut test)
    uint32_t seed = 1;
    V2 toScreen(double tau, V2 u) const {
        const double s = 1.0 + scaleRate * tau, th = rotDeg * tau * M_PI / 180.0, c = std::cos(th), sn = std::sin(th);
        const double dx = u.x - px, dy = u.y - py;
        return {px + s * (c * dx - sn * dy) + tau * shx, py + s * (sn * dx + c * dy) + tau * shy};
    }
    V2 toLayer(double tau, V2 x) const {
        const double s = 1.0 + scaleRate * tau, th = rotDeg * tau * M_PI / 180.0, c = std::cos(th), sn = std::sin(th);
        const double dx = (x.x - px - tau * shx) / s, dy = (x.y - py - tau * shy) / s;
        return {px + c * dx + sn * dy, py - sn * dx + c * dy};
    }
    bool inMask(V2 u) const {
        if (kind == 0) return true;
        if (kind == 3) {   // a plus (a reticle): centre (mcx, mcy), arms of half length mhw, half thickness mhh
            const double dx = std::fabs(u.x - mcx), dy = std::fabs(u.y - mcy);
            return (dx <= mhw && dy <= mhh) || (dy <= mhw && dx <= mhh);
        }
        if (kind == 4) {   // a text-like block: rectangle of half sizes (mhw, mhh), rows of 14 px every 18 px, glyph strokes 4 px wide every 6 px, 75% of the columns filled
            const double rx0 = u.x - (mcx - mhw), ry0 = u.y - (mcy - mhh);
            if (rx0 < 0 || ry0 < 0 || rx0 >= 2 * mhw || ry0 >= 2 * mhh) return false;
            const int row = (int)(ry0 / 18.0), col = (int)(rx0 / 6.0);
            if (std::fmod(ry0, 18.0) >= 14.0 || std::fmod(rx0, 6.0) >= 4.0) return false;
            return (hash3((uint32_t)col, (uint32_t)row, 77u) & 255u) < 192u;
        }
        const double dx = (u.x - mcx) / mhw, dy = (u.y - mcy) / mhh;
        return kind == 1 ? (std::fabs(dx) <= 1.0 && std::fabs(dy) <= 1.0) : dx * dx + dy * dy <= 1.0;
    }
    // the layer's texture at layer position u, levels 0..255 per channel
    void texture(V2 u, float rgb[3]) const {
        if (solid) { rgb[0] = srgb[0]; rgb[1] = srgb[1]; rgb[2] = srgb[2]; return; }
        static const float lam[6] = {128, 64, 32, 16, 8, 4}, amp[6] = {0.38f, 0.30f, 0.24f, 0.18f, 0.12f, 0.07f};
        const float x = (float)u.x, y = (float)u.y;
        float v[3];
        for (int c = 0; c < 3; c++) {
            float s = 0.5f;
            for (int o = 0; o < 6; o++) s += amp[o] * (vnoise(x, y, lam[o], seed * 977u + (uint32_t)(c * 31 + o)) - 0.5f);
            if (period > 0) s = 0.35f * s + 0.65f * (0.5f + 0.5f * std::cos(6.2831853f * x / (float)period));
            v[c] = std::min(255.0f, std::max(0.0f, 28.0f + 200.0f * s));
        }
        // sharp-edged rectangles and ellipses, one candidate per 96 px cell (up to 40 px half size: neighbours' too)
        const int cx0 = (int)std::floor(x / 96.0f), cy0 = (int)std::floor(y / 96.0f);
        for (int j = -1; j <= 1; j++)
            for (int i = -1; i <= 1; i++) {
                const uint32_t h = hash3((uint32_t)(cx0 + i), (uint32_t)(cy0 + j), seed * 131u + 7u);
                if ((h & 255u) >= 150u) continue;
                const float ccx = (float)(cx0 + i) * 96.0f + 96.0f * u01(hash3(h, 1, 2)), ccy = (float)(cy0 + j) * 96.0f + 96.0f * u01(hash3(h, 3, 4));
                const float hw = 10.0f + 30.0f * u01(hash3(h, 5, 6)), hh = 10.0f + 30.0f * u01(hash3(h, 7, 8));
                const float dx = (x - ccx) / hw, dy = (y - ccy) / hh;
                const bool in = (h & 256u) ? (std::fabs(dx) <= 1.0f && std::fabs(dy) <= 1.0f) : dx * dx + dy * dy <= 1.0f;
                if (!in) continue;
                for (int c = 0; c < 3; c++) v[c] = 0.25f * v[c] + 0.75f * (20.0f + 210.0f * u01(hash3(h, 11 + c, 9)));
            }
        if (gamma != 1.0) for (int c = 0; c < 3; c++) v[c] = 255.0f * std::pow(v[c] / 255.0f, (float)gamma);
        rgb[0] = v[0]; rgb[1] = v[1]; rgb[2] = v[2];
    }
};

struct Scene {
    std::string name;
    std::vector<Layer> layers;   // bottom first
    bool fade = false;           // colour = T (1 - 0.35 tau) + 12 tau
    int W = 0, H = 0;
    int top(double tau, V2 x) const {
        for (int i = (int)layers.size() - 1; i > 0; i--)
            if (layers[i].inMask(layers[i].toLayer(tau, x))) return i;
        return 0;
    }
    bool inside(V2 x) const { return x.x >= 0.0 && x.y >= 0.0 && x.x < W && x.y < H; }
    // where the content at screen position x of time tau is at time tau2
    V2 moveTo(double tau, V2 x, double tau2, int* layer = nullptr) const {
        const int L = top(tau, x);
        if (layer) *layer = L;
        return layers[L].toScreen(tau2, layers[L].toLayer(tau, x));
    }
};

static void parallelRows(int n, const std::function<void(int)>& fn) {
    const unsigned nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::atomic<int> next{0};
    std::vector<std::thread> th;
    for (unsigned i = 0; i < nt; i++)
        th.emplace_back([&] { for (int r; (r = next++) < n;) fn(r); });
    for (auto& t : th) t.join();
}

// frame at time tau: RGBA8 (alpha 255), 2x2 sub-samples per pixel
static std::vector<uint8_t> renderFrame(const Scene& sc, double tau) {
    const int W = sc.W, H = sc.H;
    std::vector<uint8_t> out((size_t)W * H * 4);
    const float gain = sc.fade ? (float)(1.0 - 0.35 * tau) : 1.0f, off = sc.fade ? (float)(12.0 * tau) : 0.0f;
    parallelRows(H, [&](int y) {
        for (int x = 0; x < W; x++) {
            float acc[3] = {0, 0, 0};
            for (int sy = 0; sy < 2; sy++)
                for (int sx = 0; sx < 2; sx++) {
                    const V2 p{x + 0.25 + 0.5 * sx, y + 0.25 + 0.5 * sy};
                    const int L = sc.top(tau, p);
                    float c[3];
                    sc.layers[L].texture(sc.layers[L].toLayer(tau, p), c);
                    for (int k = 0; k < 3; k++) acc[k] += c[k];
                }
            uint8_t* o = &out[((size_t)y * W + x) * 4];
            for (int k = 0; k < 3; k++) {
                const float v = 0.25f * acc[k] * gain + off;
                o[k] = (uint8_t)std::min(255.0f, std::max(0.0f, std::floor(v + 0.5f)));
            }
            o[3] = 255;
        }
    });
    return out;
}

// layer id at the pixel centres of the frame at time tau
static std::vector<uint8_t> idMap(const Scene& sc, double tau) {
    std::vector<uint8_t> m((size_t)sc.W * sc.H);
    parallelRows(sc.H, [&](int y) { for (int x = 0; x < sc.W; x++) m[(size_t)y * sc.W + x] = (uint8_t)sc.top(tau, V2{x + 0.5, y + 0.5}); });
    return m;
}

// exact flows at the flow pixels (flow px; centre of flow px m = fs m + fs / 2 in continuous pixel coordinates), both grids
struct Flows {
    int fw = 0, fh = 0, fs = 1;
    std::vector<float> f01, f10;        // fw * fh vec2 each
    std::vector<uint8_t> vis01, vis10;  // visible flow pixels
};
static Flows exactFlows(const Scene& sc, int fs) {
    Flows F;
    F.fs = fs;
    F.fw = (int)std::nearbyint(sc.W * (1.0 / fs));
    F.fh = (int)std::nearbyint(sc.H * (1.0 / fs));
    const size_t N = (size_t)F.fw * F.fh;
    F.f01.assign(N * 2, 0.0f); F.f10.assign(N * 2, 0.0f);
    F.vis01.assign(N, 0); F.vis10.assign(N, 0);
    parallelRows(F.fh, [&](int my) {
        for (int mx = 0; mx < F.fw; mx++) {
            const size_t i = (size_t)my * F.fw + mx;
            const V2 c{fs * mx + 0.5 * fs, fs * my + 0.5 * fs};
            for (int dir = 0; dir < 2; dir++) {
                const double t0 = dir ? 1.0 : 0.0, t1 = dir ? 0.0 : 1.0;   // dir 0: F01 on prev's grid, 1: F10 on curr's grid
                int L = 0;
                const V2 p = sc.moveTo(t0, c, t1, &L);
                float* f = dir ? &F.f10[i * 2] : &F.f01[i * 2];
                f[0] = (float)((p.x - c.x) / fs);
                f[1] = (float)((p.y - c.y) / fs);
                bool vis = sc.inside(c) && sc.inside(p) && sc.top(t1, p) == L;
                for (int k = 1; k <= 2 && vis; k++)
                    for (int dy = -1; dy <= 1 && vis; dy++)
                        for (int dx = -1; dx <= 1 && vis; dx++) {
                            if (!dx && !dy) continue;
                            const V2 a{c.x + dx * k * fs, c.y + dy * k * fs}, b{p.x + dx * k * fs, p.y + dy * k * fs};
                            vis = sc.top(t0, a) == L && sc.top(t1, b) == L;
                        }
                (dir ? F.vis10 : F.vis01)[i] = vis;
            }
        }
    });
    return F;
}

// hidden pixels of R_t (see the header comment); `strips` = the first condition alone (before the boundary band), for the t = 0 / 1 check
// sides: bit 0 = the prev side of the first condition is tested, bit 1 = the curr side (the self-check tests one side at t = 0 / 1)
static std::vector<uint8_t> hiddenMask(const Scene& sc, double t, std::vector<uint8_t>* strips = nullptr, int sides = 3, std::vector<uint8_t>* band = nullptr) {
    const int W = sc.W, H = sc.H;
    std::vector<uint8_t> a((size_t)W * H, 0), tmp((size_t)W * H, 0), d((size_t)W * H, 0);
    parallelRows(H, [&](int y) {
        for (int x = 0; x < W; x++) {
            const V2 p{x + 0.5, y + 0.5};
            const int L = sc.top(t, p);
            const V2 u = sc.layers[L].toLayer(t, p), x0 = sc.layers[L].toScreen(0.0, u), x1 = sc.layers[L].toScreen(1.0, u);
            const bool seen0 = !(sides & 1) || (sc.inside(x0) && sc.top(0.0, x0) == L), seen1 = !(sides & 2) || (sc.inside(x1) && sc.top(1.0, x1) == L);
            a[(size_t)y * W + x] = !(seen0 && seen1);
        }
    });
    // dilate by 3 (separable)
    parallelRows(H, [&](int y) {
        for (int x = 0; x < W; x++) {
            uint8_t m = 0;
            for (int i = std::max(0, x - 3); i <= std::min(W - 1, x + 3) && !m; i++) m = a[(size_t)y * W + i];
            tmp[(size_t)y * W + x] = m;
        }
    });
    parallelRows(H, [&](int y) {
        for (int x = 0; x < W; x++) {
            uint8_t m = 0;
            for (int j = std::max(0, y - 3); j <= std::min(H - 1, y + 3) && !m; j++) m = tmp[(size_t)j * W + x];
            d[(size_t)y * W + x] = m;
        }
    });
    if (strips) *strips = d;
    // plus the pixels within 2 px of a layer boundary at t
    const std::vector<uint8_t> id = idMap(sc, t);
    if (band) band->assign((size_t)W * H, 0);
    parallelRows(H, [&](int y) {
        for (int x = 0; x < W; x++) {
            const uint8_t c = id[(size_t)y * W + x];
            for (int j = std::max(0, y - 2); j <= std::min(H - 1, y + 2); j++)
                for (int i = std::max(0, x - 2); i <= std::min(W - 1, x + 2); i++)
                    if (id[(size_t)j * W + i] != c) { d[(size_t)y * W + x] = 1; if (band) (*band)[(size_t)y * W + x] = 1; }
        }
    });
    return d;
}

// ---- the scenes
static Layer plane(uint32_t seed, double px, double py) { Layer l; l.kind = 0; l.px = px; l.py = py; l.seed = seed; return l; }
static std::vector<Scene> makeScenes(const std::string& which, int W, int H) {
    const double k = std::min(W / 1920.0, H / 1080.0);   // lengths of the 1080p definitions
    auto name = [&](const char* base, double sx, double sy) {
        char b[64];
        snprintf(b, sizeof b, "%s(%g,%g)", base, std::round(sx * k * 10) / 10, std::round(sy * k * 10) / 10);
        return std::string(b);
    };
    std::vector<Scene> v;
    auto add = [&](Scene s) { s.W = W; s.H = H; v.push_back(std::move(s)); };
    const double cx = 0.5 * W, cy = 0.5 * H;
    auto want = [&](const char* n) { return which == "all" || which == n; };
    if (want("shift")) {
        const double S[7][2] = {{8, 3}, {40, -20}, {120, 60}, {250, 0}, {350, -300}, {500, 0}, {-480, 200}};
        for (auto& s : S) {
            Scene sc; sc.name = name("shift", s[0], s[1]);
            Layer l = plane(1, cx, cy); l.shx = s[0] * k; l.shy = s[1] * k;
            sc.layers = {l};
            add(sc);
        }
    }
    if (which == "slow") {   // task 105: slow whole-frame motions (not in "all"): the zero-vector bonus must not freeze them
        const double S[4][2] = {{1, 0}, {2, 1}, {3, -2}, {6, 3}};
        for (auto& s : S) {
            Scene sc; sc.name = name("shift", s[0], s[1]);
            Layer l = plane(1, cx, cy); l.shx = s[0] * k; l.shy = s[1] * k;
            sc.layers = {l};
            add(sc);
        }
        { Scene sc; sc.name = "zoom(+0.5%)"; Layer l = plane(1, cx, cy); l.scaleRate = 0.005; sc.layers = {l}; add(sc); }
        { Scene sc; sc.name = "rotate(0.3deg)"; Layer l = plane(1, cx, cy); l.rotDeg = 0.3; sc.layers = {l}; add(sc); }
    }
    if (want("zoom")) {
        Scene sc; sc.name = "zoom(+6%)";
        Layer l = plane(1, cx, cy); l.scaleRate = 0.06;
        sc.layers = {l};
        add(sc);
    }
    if (want("rotate")) {
        Scene sc; sc.name = "rotate(4deg)";
        Layer l = plane(1, cx, cy); l.rotDeg = 4.0;
        sc.layers = {l};
        add(sc);
    }
    if (want("layers")) {
        for (int zoomFg = 0; zoomFg < 2; zoomFg++) {
            Scene sc; sc.name = zoomFg ? "layers(zoom disc)" : "layers(rect)";
            Layer bg = plane(1, cx, cy); bg.shx = -60 * k; bg.shy = 10 * k;
            Layer fg; fg.seed = 2;
            const double fx = 0.38 * W, fy = 0.52 * H;   // the foreground's centre at tau = 0
            if (zoomFg) { fg.kind = 2; fg.mhw = fg.mhh = 170 * k; fg.scaleRate = 0.15; }
            else { fg.kind = 1; fg.mhw = 200 * k; fg.mhh = 150 * k; }
            fg.mcx = fx; fg.mcy = fy; fg.px = fx; fg.py = fy; fg.shx = 180 * k; fg.shy = -40 * k;
            sc.layers = {bg, fg};
            add(sc);
        }
    }
    if (which == "stripes") {   // not in "all": a repetitive texture (vertical stripes of 24 px over the noise) shifted by 2.5 periods; the flow may be wrong (informational), the ambiguity is the point
        Scene sc; sc.name = name("stripes", 60, 8);
        Layer l = plane(1, cx, cy); l.shx = 60 * k; l.shy = 8 * k; l.period = 24;
        sc.layers = {l};
        add(sc);
    }
    if (want("hud")) {   // a static overlay (a red reticle, a block of white text-like strokes) over a panning background: the overlay must stay where it is, the background must move
        Scene sc; sc.name = name("hud", 60, 20);
        Layer bg = plane(1, cx, cy); bg.shx = 60 * k; bg.shy = 20 * k;
        Layer ret; ret.kind = 3; ret.solid = true; ret.srgb[0] = 255; ret.srgb[1] = 30; ret.srgb[2] = 30; ret.mcx = 0.62 * W; ret.mcy = 0.45 * H;
        ret.mhw = std::max(14.0, 44 * k); ret.mhh = std::max(1.5, 2.0 * k); ret.px = ret.mcx; ret.py = ret.mcy; ret.seed = 5;
        Layer txt; txt.kind = 4; txt.solid = true; txt.srgb[0] = txt.srgb[1] = txt.srgb[2] = 250; txt.mcx = 0.22 * W; txt.mcy = 0.84 * H;
        txt.mhw = std::max(40.0, 120 * k); txt.mhh = std::max(18.0, 36 * k); txt.px = txt.mcx; txt.py = txt.mcy; txt.seed = 6;
        sc.layers = {bg, ret, txt};
        add(sc);
    }
    if (want("fade")) {
        Scene sc; sc.name = name("fade", 120, 60); sc.fade = true;
        Layer l = plane(1, cx, cy); l.shx = 120 * k; l.shy = 60 * k;
        sc.layers = {l};
        add(sc);
    }
    return v;
}

// ================================================================= Vulkan plumbing
static VkDevice dev;
static VkPhysicalDevice phys;
static VkPhysicalDeviceMemoryProperties memProps;
static VkQueue queue;
static VkCommandBuffer cb;
static VkFence fence;

static uint32_t memType(uint32_t bits, VkMemoryPropertyFlags f) {
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & f) == f) return i;
    fprintf(stderr, "no memory type\n");
    exit(1);
}
struct Img { VkImage img; VkDeviceMemory mem; VkImageView view; };
static Img makeImage(uint32_t w, uint32_t h, VkImageUsageFlags usage) {
    Img r{};
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
static void destroyImage(Img& i) { vkDestroyImageView(dev, i.view, nullptr); vkDestroyImage(dev, i.img, nullptr); vkFreeMemory(dev, i.mem, nullptr); i = Img{}; }
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
struct HostBuf { VkBuffer b = VK_NULL_HANDLE; VkDeviceMemory m = VK_NULL_HANDLE; uint8_t* map = nullptr; size_t size = 0; };
static HostBuf makeHostBuf(size_t bytes, VkBufferUsageFlags usage) {
    HostBuf r;
    r.size = bytes;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes; bci.usage = usage;
    CK(vkCreateBuffer(dev, &bci, nullptr, &r.b));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, r.b, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size,
                             memType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    CK(vkAllocateMemory(dev, &mai, nullptr, &r.m));
    CK(vkBindBufferMemory(dev, r.b, r.m, 0));
    CK(vkMapMemory(dev, r.m, 0, bytes, 0, (void**)&r.map));
    return r;
}
static void destroyHostBuf(HostBuf& b) { if (b.b) { vkUnmapMemory(dev, b.m); vkDestroyBuffer(dev, b.b, nullptr); vkFreeMemory(dev, b.m, nullptr); } b = HostBuf{}; }

static HostBuf stg;
static void upload(const std::vector<uint8_t>& rgba, Img& img, uint32_t w, uint32_t h) {
    memcpy(stg.map, rgba.data(), rgba.size());
    VkBufferImageCopy rg{};
    rg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; rg.imageExtent = {w, h, 1};
    begin();
    imgBarrier(img.img, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    vkCmdCopyBufferToImage(cb, stg.b, img.img, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
    imgBarrier(img.img, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    submit();
}
static std::vector<uint8_t> download(Img& img, uint32_t w, uint32_t h) {
    VkBufferImageCopy rg{};
    rg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; rg.imageExtent = {w, h, 1};
    begin();
    imgBarrier(img.img, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    vkCmdCopyImageToBuffer(cb, img.img, VK_IMAGE_LAYOUT_GENERAL, stg.b, 1, &rg);
    submit();
    return std::vector<uint8_t>(stg.map, stg.map + (size_t)w * h * 4);
}

// ================================================================= measuring
static int gFails = 0;
static bool gReportOnly = false;
static void verdict(bool ok, bool counts) { if (!ok && counts) gFails++; }

// What one run of V3 left: the estimated flows (a copy of recordDebugV3's dump) and the in-between frames at each t.
struct Run {
    std::vector<uint8_t> dump;
    framegen::Interpolator::DebugV3Layout L{};
    std::vector<std::vector<uint8_t>> out;   // per t
    int mode = 0;
    uint32_t unsure = 0;
    std::vector<double> sShare;              // --sshare: per t, the share of pixels with smear amount s > 0.1 (the weight stage's s, from the dump of every pixel)
    std::vector<std::vector<float>> sMap;    // and s itself (W * H)
    float failMix = 0, histD = 0;   // v3_decide: the unsure count (px of both grids), the soft fallback mix, the scene-cut divergence
};
static framegen::Interpolator gFg;
static bool gSShare = false;   // --sshare
static int gW = 0, gH = 0;

// estimate (or, with flows != null, the injected-flows estimate) + recordDebugV3 in one submission, then one synth per t
static bool runV3(Img& a, Img& b, Img& o, const std::vector<double>& ts, const HostBuf* flows, Run& r) {
    using namespace framegen;
    if (!gFg.debugV3Layout(r.L)) { fprintf(stderr, "debugV3Layout failed\n"); return false; }
    HostBuf dbg = makeHostBuf((size_t)r.L.size, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    begin();
    gFg.beginFrame();
    const size_t n8 = (size_t)r.L.fw * r.L.fh * 8;
    bool ok = flows ? gFg.recordDebugV3Inject(cb, a.view, b.view, flows->b, 0, n8) : gFg.recordEstimate(cb, a.view, b.view);
    ok = ok && gFg.recordDebugV3(cb, dbg.b, 0);
    submit();
    if (!ok) { destroyHostBuf(dbg); fprintf(stderr, "recording the V3 estimate failed\n"); return false; }
    r.dump.assign(dbg.map, dbg.map + r.L.size);
    destroyHostBuf(dbg);
    uint32_t hdr[16];
    memcpy(hdr, r.dump.data(), sizeof hdr);
    r.mode = (int)hdr[10];
    r.unsure = hdr[11];
    memcpy(&r.failMix, &hdr[12], 4);
    memcpy(&r.histD, &hdr[13], 4);
    r.out.clear();
    for (double t : ts) {
        begin();
        gFg.beginFrame();
        ok = gFg.recordSynth(cb, a.view, b.view, o.view, (float)t);
        submit();
        if (!ok) { fprintf(stderr, "recordSynth(%g) failed\n", t); return false; }
        r.out.push_back(download(o, gW, gH));
    }
    if (gSShare && !flows) {   // the weight stage's records of every pixel (recordDebugV3Weights, the dump grows the T-field buffer by ~230 MB at 1080p)
        framegen::Interpolator::DebugV3Weights wl;
        if (!gFg.debugV3WeightsLayout((uint32_t)(gW * gH), wl)) return false;
        HostBuf wb = makeHostBuf((size_t)wl.size, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        for (double t : ts) {
            begin();
            gFg.beginFrame();
            ok = gFg.recordDebugV3Weights(cb, a.view, b.view, o.view, (float)t, wb.b, 0, (uint32_t)(gW * gH));
            VkMemoryBarrier hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, nullptr, 0, nullptr);
            submit();
            if (!ok) { fprintf(stderr, "recordDebugV3Weights(%g) failed\n", t); destroyHostBuf(wb); return false; }
            const float* f = (const float*)wb.map;
            std::vector<float> sm((size_t)wl.count);
            size_t c = 0;
            for (uint32_t i = 0; i < wl.count; i++) { sm[i] = f[(size_t)i * 28 + 26]; c += sm[i] > 0.1f; }
            r.sShare.push_back(100.0 * c / wl.count);
            r.sMap.push_back(std::move(sm));
        }
        destroyHostBuf(wb);
    }
    return true;
}

struct FlowStat { double vis = 0, w05 = 0, w1 = 0, w2 = 0, zero = 0; size_t n = 0; };   // zero: % of visible px whose estimate is zero (< 0.25 px) though the true motion is not (>= 0.25 px)
static FlowStat flowStat(const Flows& F, const float* est, const std::vector<float>& truth, const std::vector<uint8_t>& vis) {
    FlowStat s;
    size_t c05 = 0, c1 = 0, c2 = 0, n = 0, cz = 0;
    const size_t N = (size_t)F.fw * F.fh;
    for (size_t i = 0; i < N; i++) {
        if (!vis[i]) continue;
        n++;
        const double dx = (double)est[2 * i] - truth[2 * i], dy = (double)est[2 * i + 1] - truth[2 * i + 1];
        const double e = F.fs * std::sqrt(dx * dx + dy * dy);
        c05 += e <= 0.5; c1 += e <= 1.0; c2 += e <= 2.0;
        cz += F.fs * std::hypot((double)est[2 * i], (double)est[2 * i + 1]) < 0.25 && F.fs * std::hypot((double)truth[2 * i], (double)truth[2 * i + 1]) >= 0.25;
    }
    s.n = n; s.vis = 100.0 * n / N;
    if (n) { s.w05 = 100.0 * c05 / n; s.w1 = 100.0 * c1 / n; s.w2 = 100.0 * c2 / n; s.zero = 100.0 * cz / n; }
    return s;
}

struct BetStat { double mean = 0, p99 = 0, le3 = 0, frac = 0; size_t n = 0; };
static BetStat betweenStat(const std::vector<uint8_t>& out, const std::vector<uint8_t>& ref, const std::vector<uint8_t>& hidden) {
    BetStat s;
    std::vector<uint64_t> hist(766, 0);   // sum over the 3 channels of |out - ref|
    uint64_t tot = 0, n = 0, le3 = 0;
    const size_t P = hidden.size();
    for (size_t i = 0; i < P; i++) {
        if (hidden[i]) continue;
        int d = 0;
        for (int c = 0; c < 3; c++) d += std::abs((int)out[i * 4 + c] - (int)ref[i * 4 + c]);
        hist[d]++; tot += d; n++; le3 += d <= 9;
    }
    s.n = n; s.frac = 100.0 * n / P;
    if (!n) return s;
    s.mean = (double)tot / n / 3.0;
    s.le3 = 100.0 * le3 / n;
    uint64_t acc = 0;
    for (int d = 0; d < 766; d++) { acc += hist[d]; if (acc * 100 >= n * 99) { s.p99 = d / 3.0; break; } }
    return s;
}

// ---- the static overlay check (scene "hud"): the overlay layers (id >= 1 at time t) against the background around and away from them
static std::vector<uint8_t> dilateBox(const std::vector<uint8_t>& m, int W, int H, int r) {
    std::vector<uint8_t> tmp(m.size(), 0), d(m.size(), 0);
    parallelRows(H, [&](int y) { for (int x = 0; x < W; x++) { uint8_t v = 0; for (int i = std::max(0, x - r); i <= std::min(W - 1, x + r) && !v; i++) v = m[(size_t)y * W + i]; tmp[(size_t)y * W + x] = v; } });
    parallelRows(H, [&](int y) { for (int x = 0; x < W; x++) { uint8_t v = 0; for (int j = std::max(0, y - r); j <= std::min(H - 1, y + r) && !v; j++) v = tmp[(size_t)j * W + x]; d[(size_t)y * W + x] = v; } });
    return d;
}
struct HudStat { double ovl = 0, ring = 0, ringBad = 0, far = 0; size_t nOvl = 0, nRing = 0, nFar = 0; };
// ovl: mean error over the overlay's own pixels (eroded by 1 px: the anti-aliased rim is a blend of the two), ring: over the background 3..40 px from the overlay (where a dragged or doubled
// copy of it would show; the revealed strips are in it: the background moves 60 px), ringBad: the share of the ring with an error above 40 levels, far: the background more than 60 px away
static HudStat hudStat(const Scene& sc, const std::vector<uint8_t>& out, const std::vector<uint8_t>& ref, double t) {
    const int W = sc.W, H = sc.H;
    const std::vector<uint8_t> id = idMap(sc, t), hid = hiddenMask(sc, t);
    std::vector<uint8_t> ov(id.size()), bgm(id.size());
    for (size_t i = 0; i < id.size(); i++) { ov[i] = id[i] >= 1; bgm[i] = id[i] == 0 && !hid[i]; }   // (the content hidden in one real frame is the layers' own test, not this one)
    const std::vector<uint8_t> inner = dilateBox(bgm, W, H, 1), d3 = dilateBox(ov, W, H, 3), d40 = dilateBox(ov, W, H, 40), d60 = dilateBox(ov, W, H, 60);
    HudStat s;
    double so = 0, sr = 0, sf = 0;
    size_t bad = 0;
    for (size_t i = 0; i < id.size(); i++) {
        double e = 0;
        for (int c = 0; c < 3; c++) e += std::abs((int)out[i * 4 + c] - (int)ref[i * 4 + c]);
        e /= 3.0;
        if (ov[i] && !inner[i]) { so += e; s.nOvl++; }
        else if (bgm[i] && d40[i] && !d3[i]) { sr += e; s.nRing++; bad += e > 40.0; }
        else if (bgm[i] && !d60[i]) { sf += e; s.nFar++; }
    }
    s.ovl = s.nOvl ? so / s.nOvl : 0; s.ring = s.nRing ? sr / s.nRing : 0; s.ringBad = s.nRing ? 100.0 * bad / s.nRing : 0; s.far = s.nFar ? sf / s.nFar : 0;
    return s;
}

// ---- the CPU warp of the self-check: dst(c) = bilinear(src, c + fs F(m)) for the pixels whose flow pixel is visible
static double warpError(const Scene& sc, const Flows& F, const std::vector<uint8_t>& src, const std::vector<uint8_t>& dst, const std::vector<float>& flow,
                        const std::vector<uint8_t>& vis, size_t* count, double gain, double offset) {
    const int W = sc.W, H = sc.H;
    std::vector<double> rowSum(H, 0.0);
    std::vector<size_t> rowN(H, 0);
    parallelRows(H, [&](int y) {
        for (int x = 0; x < W; x++) {
            const int mx = std::min(x / F.fs, F.fw - 1), my = std::min(y / F.fs, F.fh - 1);
            const size_t i = (size_t)my * F.fw + mx;
            if (!vis[i]) continue;
            const double px = x + 0.5 + F.fs * flow[2 * i] - 0.5, py = y + 0.5 + F.fs * flow[2 * i + 1] - 0.5;   // pixel-index coordinates
            const double fx = std::floor(px), fy = std::floor(py);
            const double ax = px - fx, ay = py - fy;
            auto at = [&](int xx, int yy, int c) { xx = std::min(W - 1, std::max(0, xx)); yy = std::min(H - 1, std::max(0, yy)); return (double)src[((size_t)yy * W + xx) * 4 + c]; };
            double e = 0;
            for (int c = 0; c < 3; c++) {
                const double v = (1 - ay) * ((1 - ax) * at((int)fx, (int)fy, c) + ax * at((int)fx + 1, (int)fy, c)) +
                                 ay * ((1 - ax) * at((int)fx, (int)fy + 1, c) + ax * at((int)fx + 1, (int)fy + 1, c));
                e += std::fabs(gain * v + offset - dst[((size_t)y * W + x) * 4 + c]);
            }
            rowSum[y] += e / 3.0; rowN[y]++;
        }
    });
    double s = 0; size_t n = 0;
    for (int y = 0; y < H; y++) { s += rowSum[y]; n += rowN[y]; }
    *count = n;
    return n ? s / n : 0.0;
}

// The oracle of the self-check (iv): the ideal gather blend (1 - t) prev(x0) + t curr(x1) with the exact layer motion (x0, x1: where the content at x
// of R_t is at time 0 / 1), bilinear taps, rounded: what V3 with perfect flows and no occlusion handling would make. Its mean error against R_t over
// the non-hidden pixels shows what the "<= 1.0 level" threshold of the injected path can ask of these textures.
static std::vector<uint8_t> oracleBlend(const Scene& sc, const std::vector<uint8_t>& prev, const std::vector<uint8_t>& curr, double t) {
    const int W = sc.W, H = sc.H;
    std::vector<uint8_t> out((size_t)W * H * 4, 255);
    auto tap = [&](const std::vector<uint8_t>& im, V2 p, int c) {
        const double px = p.x - 0.5, py = p.y - 0.5, fx = std::floor(px), fy = std::floor(py), ax = px - fx, ay = py - fy;
        auto at = [&](int x, int y) { x = std::min(W - 1, std::max(0, x)); y = std::min(H - 1, std::max(0, y)); return (double)im[((size_t)y * W + x) * 4 + c]; };
        return (1 - ay) * ((1 - ax) * at((int)fx, (int)fy) + ax * at((int)fx + 1, (int)fy)) + ay * ((1 - ax) * at((int)fx, (int)fy + 1) + ax * at((int)fx + 1, (int)fy + 1));
    };
    parallelRows(H, [&](int y) {
        for (int x = 0; x < W; x++) {
            const V2 p{x + 0.5, y + 0.5};
            const int L = sc.top(t, p);
            const V2 u = sc.layers[L].toLayer(t, p), x0 = sc.layers[L].toScreen(0.0, u), x1 = sc.layers[L].toScreen(1.0, u);
            for (int c = 0; c < 3; c++) out[((size_t)y * W + x) * 4 + c] = (uint8_t)std::min(255.0, std::max(0.0, std::floor((1 - t) * tap(prev, x0, c) + t * tap(curr, x1, c) + 0.5)));
        }
    });
    return out;
}

// --diag: where the wrong flow pixels (visible, error > 1 px) are, per layer: by distance to the nearest layer boundary of the source frame (flow px) and how many of them
// carry another layer's true motion; --errmap PREFIX writes a PPM per direction (visible and right: dark grey, wrong: red, estimate = another layer's motion: yellow, not visible: black)
static bool gDiag = false;
static bool gColourWindow = false;   // --colour-window
static bool gDetailSplit = false;    // --detail-split
static int gMotion = 0;              // --v3-motion N: Settings::v3Motion (0 = V3, 1 = V3.5)
static int gBlend = 0;               // --blend N: Settings::v3Blend (learned blend, experimental)
static framegen::V3Guard gGuard = framegen::V3Guard::Off;   // --guard
static float gHud = -1.0f;            // --hud X: Settings::hudBonus (default: the library default)
static std::string gErrMap;
static bool gProfile = false;   // --profile: the occlusion ramp along the middle row (layers scenes)
static std::string gSave;   // --save PREFIX: the scene's two real frames as raw RGBA (PREFIX-NAME-p.rgba, -c.rgba) for the other tools (fgtest, v3flow.py)
static std::string gFrames;   // --frames PREFIX: PPMs of every in-between frame, its reference and the error (x8, grey)
static void writePpm(const std::string& fn, const std::vector<uint8_t>& rgba, int W, int H, bool err, const std::vector<uint8_t>* ref) {
    FILE* f = fopen(fn.c_str(), "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    std::vector<uint8_t> row((size_t)W * 3);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            const size_t i = ((size_t)y * W + x) * 4;
            for (int c = 0; c < 3; c++) row[x * 3 + c] = err ? (uint8_t)std::min(255, 8 * std::abs((int)rgba[i + c] - (int)(*ref)[i + c])) : rgba[i + c];
        }
        fwrite(row.data(), 1, row.size(), f);
    }
    fclose(f);
}
static int gBlkLevel = -1;
static void diagFlow(const Scene& sc, const Flows& F, const float* est, const std::vector<float>& truth, const std::vector<uint8_t>& vis, int dir, const std::string& nm) {
    const int fw = F.fw, fh = F.fh, fs = F.fs, W = sc.W;
    const std::vector<uint8_t> id = idMap(sc, dir ? 1.0 : 0.0);
    auto idAt = [&](int mx, int my) { return id[(size_t)std::min(sc.H - 1, my * fs + fs / 2) * W + std::min(W - 1, mx * fs + fs / 2)]; };
    const int NL = (int)sc.layers.size();
    const double tau0 = dir ? 1.0 : 0.0, tau1 = dir ? 0.0 : 1.0;
    std::vector<size_t> vis_n(NL, 0), wrong(NL, 0), wrongOther(NL, 0), byDist[4];
    std::vector<double> sEx(NL, 0), sEy(NL, 0), sTx(NL, 0), sTy(NL, 0);
    for (auto& v : byDist) v.assign(NL, 0);
    std::vector<uint8_t> img;
    if (!gErrMap.empty()) img.assign((size_t)fw * fh * 3, 0);
    for (int my = 0; my < fh; my++)
        for (int mx = 0; mx < fw; mx++) {
            const size_t i = (size_t)my * fw + mx;
            if (!vis[i]) continue;
            const int L = idAt(mx, my);
            vis_n[L]++;
            sEx[L] += est[2 * i]; sEy[L] += est[2 * i + 1]; sTx[L] += truth[2 * i]; sTy[L] += truth[2 * i + 1];
            const double dx = est[2 * i] - truth[2 * i], dy = est[2 * i + 1] - truth[2 * i + 1];
            const bool bad = fs * std::sqrt(dx * dx + dy * dy) > 1.0;
            if (!gErrMap.empty()) { uint8_t* o = &img[i * 3]; o[0] = o[1] = o[2] = 60; }
            if (!bad) continue;
            wrong[L]++;
            if (gDiag && getenv("V3TEST_SAMPLES") && L == 0 && wrong[L] % 400 == 0) printf("    wrong bg px (%d,%d): est (%.2f,%.2f) true (%.2f,%.2f)\n", mx, my, est[2 * i], est[2 * i + 1], truth[2 * i], truth[2 * i + 1]);
            int d = 99;   // Chebyshev distance (flow px) to the nearest pixel of another layer (up to 24)
            for (int r = 1; r <= 24 && d == 99; r++)
                for (int j = -r; j <= r && d == 99; j++)
                    for (int k = -r; k <= r; k++) {
                        if (std::max(std::abs(j), std::abs(k)) != r) continue;
                        const int x = mx + k, y = my + j;
                        if (x < 0 || y < 0 || x >= fw || y >= fh) continue;
                        if (idAt(x, y) != L) { d = r; break; }
                    }
            byDist[d < 4 ? 0 : d < 8 ? 1 : d < 16 ? 2 : 3][L]++;
            bool other = false;
            const V2 c{fs * mx + 0.5 * fs, fs * my + 0.5 * fs};
            for (int M = 0; M < NL && !other; M++) {
                if (M == L) continue;
                const V2 q = sc.layers[M].toScreen(tau1, sc.layers[M].toLayer(tau0, c));
                const double ox = est[2 * i] - (q.x - c.x) / fs, oy = est[2 * i + 1] - (q.y - c.y) / fs;
                other = fs * std::sqrt(ox * ox + oy * oy) <= 1.0;
            }
            wrongOther[L] += other;
            if (!gErrMap.empty()) { uint8_t* o = &img[i * 3]; o[0] = 255; o[1] = other ? 255 : 0; o[2] = 0; }
        }
    for (int L = 0; L < NL; L++)
        printf("  diag %-18s %s layer %d: visible %zu, wrong %zu (%.1f%%); wrong by distance to the layer boundary: <4 %zu, 4..8 %zu, 8..16 %zu, >=16 %zu; carrying another layer's motion %zu; mean est (%.2f, %.2f) true (%.2f, %.2f) flow px\n", nm.c_str(),
               dir ? "F10" : "F01", L, vis_n[L], wrong[L], vis_n[L] ? 100.0 * wrong[L] / vis_n[L] : 0.0, byDist[0][L], byDist[1][L], byDist[2][L], byDist[3][L], wrongOther[L],
               vis_n[L] ? sEx[L] / vis_n[L] : 0.0, vis_n[L] ? sEy[L] / vis_n[L] : 0.0, vis_n[L] ? sTx[L] / vis_n[L] : 0.0, vis_n[L] ? sTy[L] / vis_n[L] : 0.0);
    if (!gErrMap.empty()) {
        std::string fn = gErrMap + "-" + nm + (dir ? "-F10" : "-F01") + ".ppm";
        for (char& c : fn) if (c == '(' || c == ')' || c == ',' || c == '%' || c == '+') c = '_';
        if (FILE* f = fopen(fn.c_str(), "wb")) { fprintf(f, "P6\n%d %d\n255\n", fw, fh); fwrite(img.data(), 1, img.size(), f); fclose(f); }
    }
}

// The check of the stats (state G) against the analytic motion of the scene. Expected affine camera = the background layer's motion in flow px (d at the frame
// centre and its slopes over the half frame); the dominant motions = the motion of every layer at its own centre. Tolerances: the block vectors the stats read
// come from level LH, an integer lattice of 2^LH flow px, so a translation is right to 0.75 2^LH (+ 2%), a slope to 1.5 flow px + 6% + 1% of the translation, and a layer's dominant
// motion to 1.5 2^LH (a zooming layer's vector varies over it).
static bool statsCheck(const Scene& sc, const Flows& F, const uint32_t* G, uint32_t nLevels, char* msg, size_t msgLen) {
    float gf[64];
    memcpy(gf, G, sizeof gf);
    const double lat = nLevels >= 3 ? (double)(1 << std::max(1u, nLevels - 3)) : 1.0;
    const int fs = F.fs;
    const double hx = 0.5 * F.fw, hy = 0.5 * F.fh;   // half frame in flow px
    auto motionAt = [&](int layer, V2 c) {           // the motion of layer `layer`'s content at screen position c (px), in flow px
        const Layer& l = sc.layers[layer];
        const V2 p = l.toScreen(1.0, l.toLayer(0.0, c));
        return V2{(p.x - c.x) / fs, (p.y - c.y) / fs};
    };
    auto motion = [&](int layer, double X, double Y) { return motionAt(layer, V2{fs * (hx + X * hx), fs * (hy + Y * hy)}); };
    bool ok = true;
    size_t n = 0;
    msg[0] = 0;
    auto add = [&](const char* fmt, double a, double b, double c, double d) { if (n < msgLen - 120) n += snprintf(msg + n, msgLen - n, fmt, a, b, c, d); };
    if (!(G[6] & 256u)) { ok = false; add("aff INVALID", 0, 0, 0, 0); }
    else {
        const V2 t0 = motion(0, 0, 0), tx0 = motion(0, -1, 0), tx1 = motion(0, 1, 0), ty0 = motion(0, 0, -1), ty1 = motion(0, 0, 1);
        const double ex[6] = {t0.x, 0.5 * (tx1.x - tx0.x), 0.5 * (ty1.x - ty0.x), t0.y, 0.5 * (tx1.y - tx0.y), 0.5 * (ty1.y - ty0.y)};
        double worst = 0;
        for (int k = 0; k < 6; k++) {
            const double tol = (k == 0 || k == 3) ? 0.75 * lat + 0.02 * std::fabs(ex[k]) : 1.5 + 0.06 * std::fabs(ex[k]) + 0.01 * std::hypot(t0.x, t0.y);
            const double e = std::fabs(gf[16 + k] - ex[k]) / tol;
            worst = std::max(worst, e);
            if (e > 1.0) ok = false;
        }
        add("aff worst error %.2f of its tolerance (true %.2f %+.2f X, %+.2f Y)  ", worst, ex[0], ex[1], ex[2]);
    }
    if (sc.layers.size() > 1) {   // every layer's motion is among the valid dominant motions (the foreground at its mask centre, the background at the frame centre)
        for (size_t L = 0; L < sc.layers.size(); L++) {
            if (sc.layers[L].solid) continue;   // (an overlay of a few thousand px is below the dominant-motion mass minimum)
            const V2 c = L ? V2{sc.layers[L].mcx, sc.layers[L].mcy} : V2{0.5 * sc.W, 0.5 * sc.H};
            const V2 e = motionAt((int)L, c);
            double best = 1e9;
            for (int k = 0; k < 4; k++) if (G[6] & (1u << k)) best = std::min(best, std::hypot(gf[8 + 2 * k] - e.x, gf[9 + 2 * k] - e.y));
            if (best > 1.5 * lat) ok = false;
            add(L ? "fg dominant error %.1f (true %.1f,%.1f; tol %.1f)  " : "bg dominant error %.1f (true %.1f,%.1f; tol %.1f)  ", best, e.x, e.y, 1.5 * lat);
        }
    } else {   // a single layer: a pure translation has the layer's motion as dominant motion (a zoom / rotation has no single one: not tested)
        const V2 t0 = motion(0, 0, 0), tx0 = motion(0, -1, 0), tx1 = motion(0, 1, 0);
        if (std::hypot(tx1.x - tx0.x, tx1.y - tx0.y) < 2.0) {
            const double e = (G[6] & 1u) ? std::hypot(gf[8] - t0.x, gf[9] - t0.y) : 1e9, tol = 0.75 * lat + 0.02 * std::hypot(t0.x, t0.y);
            if (e > tol) ok = false;
            add("dom[0] error %.1f flow px (tol %.1f)%.0s%.0s", e, tol, 0, 0);
        }
    }
    return ok;
}

// ================================================================= main
static std::vector<double> parseTs(const char* s) {
    std::vector<double> v;
    for (char* c = strdup(s); c;) {
        char* nx = strchr(c, ',');
        if (nx) *nx++ = 0;
        v.push_back(atof(c));
        c = nx;
    }
    return v;
}

// (task 109) the frame-border scenes (gpu/V3.md 3.15 "As built (user test 2, task 109)"): a textured pan of 40 px (1080p lengths) whose frames carry a
// dark border (6 px of 0), black bars (60 px top and bottom) or a bright border (6 px of 255, the control), with injected vectors (recordDebugV3Inject:
// F01 = v, F10 = -v everywhere, as the search would give for a pan) of 100 .. 1500 px and a diagonal one, t = 0.02 / 0.5 / 0.98, v3Smear 0 and 0.35;
// and with non-finite / out-of-half-range vectors (NaN, +inf, 1e5 flow px) in the centre third (the true pan elsewhere). Checks, inside the frame minus
// a margin (the decoration + 16 px): among the pixels whose two taps (at t, for the injected v) both lie more than 2 px outside the frame, the share
// that takes the border's colour (luma < 8 dark, > 248 bright: the texture never reaches either) is <= 0.002 (worst t); the share over the whole
// interior is printed as information (a wrong vector with one tap inside reads real content, the decoration inside the picture included: no rule can
// tell a black border from black content). In the non-finite region: border colour and the share darker than min(prev, curr) - 40 at the pixel
// <= 0.0005 each. A vector whose taps both leave the picture must not bring the border in; a non-finite vector must not reach the output.
static float lumaOf(const std::vector<uint8_t>& f, size_t i) { return 0.299f * f[4 * i] + 0.587f * f[4 * i + 1] + 0.114f * f[4 * i + 2]; }
static void borderScenes(const framegen::Settings& base, Img& a, Img& b, Img& o, int W, int H) {
    framegen::Interpolator::DebugV3Layout L;
    if (!gFg.debugV3Layout(L)) { printf("border: no V3 layout   FAIL\n"); verdict(false, true); return; }
    const int fw = (int)L.fw, fh = (int)L.fh, fs = (int)L.fscale;
    const size_t N = (size_t)fw * fh;
    const double k = std::min(W / 1920.0, H / 1080.0);
    Scene sc;
    sc.W = W; sc.H = H;
    Layer l = plane(5, 0.5 * W, 0.5 * H);
    l.shx = 40.0 * k;
    sc.layers = {l};
    const std::vector<uint8_t> p0 = renderFrame(sc, 0.0), c0 = renderFrame(sc, 1.0);
    const std::vector<double> ts = {0.02, 0.5, 0.98};
    struct Deco { const char* nm; uint8_t v; int w; bool barsOnly; };
    const int bars = std::max(8, (int)std::lround(60.0 * k));
    const Deco decos[3] = {{"dark border", 0, 6, false}, {"black bars", 0, bars, true}, {"bright border", 255, 6, false}};
    HostBuf fb = makeHostBuf(N * 16, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    float* f01 = (float*)fb.map;
    float* f10 = f01 + 2 * N;
    for (const Deco& d : decos) {
        std::vector<uint8_t> p = p0, c = c0;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                const bool on = d.barsOnly ? (y < d.w || y >= H - d.w) : (x < d.w || y < d.w || x >= W - d.w || y >= H - d.w);
                if (!on) continue;
                for (int ch = 0; ch < 3; ch++) p[4 * ((size_t)y * W + x) + ch] = c[4 * ((size_t)y * W + x) + ch] = d.v;
            }
        upload(p, a, W, H);
        upload(c, b, W, H);
        const int M = d.w + 16;
        // the share of the interior that takes the border's colour, worst over the t's; and (nf) the share of the region darker than min(prev, curr) - 40
        // vFull != 0: also the share of border colour among the pixels whose two taps (prev at x - t v, curr at x + (1 - t) v) both lie more than 2 px
        // outside the frame (the mechanism: nothing of the picture is there, clamp-to-edge reads the border); 0 such pixels: 0
        auto measure = [&](const Run& r, int x0, int y0, int x1, int y1, double& bordShare, double& darkShare, double vxF = 0.0, double vyF = 0.0, double* bothShare = nullptr, size_t* bothN = nullptr) {
            bordShare = darkShare = 0.0;
            if (bothShare) *bothShare = 0.0;
            if (bothN) *bothN = 0;
            auto outside = [&](double px, double py) { return px < -2.5 || py < -2.5 || px > W + 1.5 || py > H + 1.5; };
            for (size_t kt = 0; kt < r.out.size(); kt++) {
                const auto& out = r.out[kt];
                const double t = ts[kt];
                size_t nb = 0, nd = 0, n = 0, mb = 0, mn = 0;
                for (int y = std::max(y0, M); y < std::min(y1, H - M); y++)
                    for (int x = std::max(x0, M); x < std::min(x1, W - M); x++) {
                        const size_t i = (size_t)y * W + x;
                        const float g = lumaOf(out, i);
                        const bool bc = d.v == 0 ? g < 8.0f : g > 248.0f;
                        nb += bc;
                        nd += g < std::min(lumaOf(p, i), lumaOf(c, i)) - 40.0f;
                        n++;
                        if ((vxF != 0.0 || vyF != 0.0) && outside(x - t * vxF, y - t * vyF) && outside(x + (1.0 - t) * vxF, y + (1.0 - t) * vyF)) { mn++; mb += bc; }
                    }
                bordShare = std::max(bordShare, (double)nb / std::max<size_t>(n, 1));
                darkShare = std::max(darkShare, (double)nd / std::max<size_t>(n, 1));
                if (bothShare) *bothShare = std::max(*bothShare, (double)mb / std::max<size_t>(mn, 1));
                if (bothN) *bothN += mn;
            }
        };
        struct Vec { const char* nm; double x, y; };
        const Vec vecs[] = {{"100", 100, 0}, {"300", 300, 0}, {"600", 600, 0}, {"1500 across", 1500, 0}, {"1500 down", 0, 1500}, {"700 diagonal", 700, -700}};
        for (const Vec& v : vecs)
            for (float smear : {0.0f, 0.35f}) {
                const float vx = (float)(v.x * k / fs), vy = (float)(v.y * k / fs);
                for (size_t i = 0; i < N; i++) { f01[2 * i] = vx; f01[2 * i + 1] = vy; f10[2 * i] = -vx; f10[2 * i + 1] = -vy; }
                framegen::Settings st = base;
                st.v3Smear = smear;
                gFg.setSettings(st);
                Run r;
                const bool ok = runV3(a, b, o, ts, &fb, r);
                double bs = 1.0, ds = 0.0, ms = 1.0;
                size_t mn = 0;
                if (ok) measure(r, 0, 0, W, H, bs, ds, v.x * k, v.y * k, &ms, &mn);
                const bool pass = ok && ms <= 0.002;
                char nm[96];
                snprintf(nm, sizeof nm, "border %s v=%s", d.nm, v.nm);
                printf("%-36s smear %.2f inject: border colour where both taps leave the frame %.4f of %zu px, in the whole frame %.4f (informational)   %s (<= 0.002)\n",
                       nm, smear, ms, mn, bs, pass ? "PASS" : "FAIL");
                verdict(pass, true);
            }
        // non-finite vectors in the centre third, the true pan elsewhere
        const float bad[3] = {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), 1e5f};
        const char* badNm[3] = {"NaN", "+inf", "1e5"};
        for (int q = 0; q < 3; q++) {
            const float sx = (float)(40.0 * k / fs);
            for (int y = 0; y < fh; y++)
                for (int x = 0; x < fw; x++) {
                    const size_t i = (size_t)y * fw + x;
                    const bool in = x >= fw / 3 && x < 2 * fw / 3 && y >= fh / 3 && y < 2 * fh / 3;
                    f01[2 * i] = in ? bad[q] : sx; f01[2 * i + 1] = in ? bad[q] : 0.0f;
                    f10[2 * i] = in ? bad[q] : -sx; f10[2 * i + 1] = in ? bad[q] : 0.0f;
                }
            framegen::Settings st = base;
            st.v3Smear = 0.35f;
            gFg.setSettings(st);
            Run r;
            const bool ok = runV3(a, b, o, ts, &fb, r);
            double bs = 1.0, ds = 1.0;
            if (ok) measure(r, fs * (fw / 3), fs * (fh / 3), fs * (2 * fw / 3), fs * (2 * fh / 3), bs, ds);
            const bool pass = ok && bs <= 0.0005 && ds <= 0.0005;
            char nm[96];
            snprintf(nm, sizeof nm, "border %s v=%s", d.nm, badNm[q]);
            printf("%-36s smear 0.35 inject: border colour in the region %.4f, darker than min(prev, curr) - 40 %.4f   %s (<= 0.0005)\n", nm, bs, ds, pass ? "PASS" : "FAIL");
            verdict(pass, true);
        }
    }
    gFg.setSettings(base);
    destroyHostBuf(fb);
}


// Settings::v3Motion at run time (task 160): one interpolator that runs with v3Motion 0, is switched to 1 by setSettings and back, three times in a row, mid-stream (the same pair:
// V3 keeps no state across pairs, every pair's estimate starts from its two frames), then an interpolator created with each value. The frames after switching back are byte-identical to
// the frames of a run that never switched, the frames at 1 to those of a run created with 1; v3MotionActive() follows the setting and the motion version.
static void motionSwitch(const framegen::InitInfo& info, Img& a, Img& b, Img& o, int W, int H) {
    using namespace framegen;
    const double k = std::min(W / 1920.0, H / 1080.0);
    Scene sc; sc.W = W; sc.H = H;
    Layer bg = plane(1, 0.5 * W, 0.5 * H); bg.shx = -60 * k; bg.shy = 10 * k; bg.rotDeg = 1.0;
    Layer fg; fg.seed = 2; fg.kind = 1; fg.mhw = 200 * k; fg.mhh = 150 * k; fg.mcx = fg.px = 0.38 * W; fg.mcy = fg.py = 0.52 * H; fg.shx = 180 * k; fg.shy = -40 * k;
    sc.layers = {bg, fg};
    upload(renderFrame(sc, 0.0), a, W, H);
    upload(renderFrame(sc, 1.0), b, W, H);
    const std::vector<double> ts = {0.5};
    auto settings = [&](int m, V3Guard g) { Settings s = info.settings; s.v3Motion = (uint8_t)m; s.v3Guard = g; return s; };
    auto run = [&](std::vector<uint8_t>& out) { Run r; const bool ok = runV3(a, b, o, ts, nullptr, r); if (ok) out = r.out[0]; return ok; };
    auto check = [&](bool pass, const char* what) { printf("%-20s v3Motion switching: %s   %s\n", "motion", what, pass ? "PASS" : "FAIL"); verdict(pass, true); };
    auto nDiff = [](const std::vector<uint8_t>& x, const std::vector<uint8_t>& y) { size_t n = 0; for (size_t i = 0; i < x.size() && i < y.size(); i++) n += x[i] != y[i]; return n; };
    const V3Guard g0 = info.settings.v3Guard;
    std::vector<uint8_t> A0, A1, out;
    bool ok = gFg.setSettings(settings(0, g0)) && run(A0);
    check(ok && !gFg.v3MotionActive(), "v3Motion 0: runs, v3MotionActive() false");
    ok = gFg.setSettings(settings(1, g0)) && run(A1);
    check(ok && gFg.v3MotionActive(), "setSettings(v3Motion 1): runs, v3MotionActive() true");
    printf("%-20s v3Motion switching: information: %zu of %zu output bytes differ between v3Motion 0 and 1\n", "motion", nDiff(A0, A1), A0.size());
    size_t bad0 = 0, bad1 = 0;
    bool allOk = true;
    for (int cycle = 0; cycle < 3; cycle++) {
        allOk = allOk && gFg.setSettings(settings(0, g0)) && run(out) && !gFg.v3MotionActive();
        bad0 += nDiff(out, A0);
        allOk = allOk && gFg.setSettings(settings(1, g0)) && run(out);
        bad1 += nDiff(out, A1);
    }
    check(allOk && bad0 == 0, "back at 0 (3 switches): every frame byte-identical to the run that never switched");
    check(allOk && bad1 == 0, "at 1 again (3 switches): every frame byte-identical to the first one at 1");
    // an interpolator created with each value
    for (int m = 0; m < 2; m++) {
        gFg.destroy();
        InitInfo i2 = info;
        i2.settings = settings(m, g0);
        const bool inited = gFg.init(i2);
        out.clear();
        const bool rok = inited && run(out);
        const size_t d = nDiff(out, m ? A1 : A0);
        char nm[128];
        snprintf(nm, sizeof nm, "created with v3Motion %d: byte-identical to the switched run (%zu bytes differ)", m, d);
        check(rok && d == 0, nm);
    }
    // the guard modes: V3.5 runs under both (V2Motion: V2's field replaces the dense field, only the statistics of V3.5 show); another motion version ignores v3Motion
    {
        std::vector<uint8_t> G0, G1;
        const bool gok = gFg.setSettings(settings(0, V3Guard::V2Motion)) && run(G0) && gFg.setSettings(settings(1, V3Guard::V2Motion)) && run(G1);
        check(gok && gFg.v3MotionActive(), "v3Guard V2Motion with v3Motion 1: runs, v3MotionActive() true");
        printf("%-20s v3Motion switching: information: v3Guard V2Motion, %zu output bytes differ between v3Motion 0 and 1\n", "motion", nDiff(G0, G1));
        const bool g2 = gFg.setSettings(settings(1, V3Guard::Guard)) && run(out);
        check(g2 && gFg.v3MotionActive(), "v3Guard Guard with v3Motion 1: v3MotionActive() true");
        gFg.destroy();
        InitInfo i3 = info;
        i3.settings = settings(1, g0);
        i3.settings.motionVersion = MotionVersion::V1;
        const bool v1ok = gFg.init(i3);
        check(v1ok && !gFg.v3MotionActive(), "motionVersion V1 with v3Motion 1: v3MotionActive() false");
    }
    gFg.destroy();
    if (!gFg.init(info)) { fprintf(stderr, "init failed after the switching test\n"); exit(1); }
}

int main(int argc, char** argv) {
    int W = 1920, H = 1080, fs = 2;
    std::string which = "all";
    std::vector<double> ts = {0.25, 0.5};
    bool selfcheck = false, list = false;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--size" && i + 1 < argc) { if (sscanf(argv[++i], "%dx%d", &W, &H) != 2) { fprintf(stderr, "--size WxH\n"); return 2; } }
        else if (a == "--fs" && i + 1 < argc) fs = atoi(argv[++i]);
        else if (a == "--scene" && i + 1 < argc) which = argv[++i];
        else if (a == "--t" && i + 1 < argc) ts = parseTs(argv[++i]);
        else if (a == "--report-only") gReportOnly = true;
        else if (a == "--selfcheck") selfcheck = true;
        else if (a == "--diag") gDiag = true;
        else if (a == "--blk" && i + 1 < argc) gBlkLevel = atoi(argv[++i]);
        else if (a == "--errmap" && i + 1 < argc) { gErrMap = argv[++i]; gDiag = true; }
        else if (a == "--frames" && i + 1 < argc) gFrames = argv[++i];
        else if (a == "--save" && i + 1 < argc) gSave = argv[++i];
        else if (a == "--sshare") gSShare = true;
        else if (a == "--profile") gProfile = true;
        else if (a == "--colour-window") gColourWindow = true;
        else if (a == "--detail-split") gDetailSplit = true;
        else if (a == "--v3-motion" && i + 1 < argc) gMotion = atoi(argv[++i]);
        else if (a == "--blend" && i + 1 < argc) gBlend = atoi(argv[++i]);
        else if (a == "--guard" && i + 1 < argc) {
            const std::string g = argv[++i];
            if (g == "off" || g == "0") gGuard = framegen::V3Guard::Off;
            else if (g == "guard" || g == "1") gGuard = framegen::V3Guard::Guard;
            else if (g == "v2motion" || g == "2") gGuard = framegen::V3Guard::V2Motion;
            else { fprintf(stderr, "--guard off|guard|v2motion\n"); return 2; }
        }
        else if (a == "--hud" && i + 1 < argc) gHud = (float)atof(argv[++i]);
        else if (a == "--list") list = true;
        else { fprintf(stderr, "usage: v3test [--size WxH] [--fs 1|2|4] [--scene shift|zoom|rotate|layers|fade|stripes|border|decide|motion|all] [--t T1,T2,..] [--report-only] [--selfcheck] [--diag] [--errmap PREFIX] [--colour-window] [--detail-split] [--blend N] [--guard off|guard|v2motion] [--v3-motion 0|1] [--list]\n"); return 2; }
    }
    if ((fs != 1 && fs != 2 && fs != 4) || W < 16 || H < 16) { fprintf(stderr, "--fs 1|2|4, --size at least 16x16\n"); return 2; }
    for (double t : ts) if (!(t > 0.0 && t < 1.0)) { fprintf(stderr, "--t: times in (0, 1)\n"); return 2; }
    std::vector<Scene> scenes = makeScenes(which, W, H);
    if (scenes.empty() && which != "decide" && which != "border" && which != "motion") { fprintf(stderr, "unknown scene %s (shift zoom rotate layers fade all)\n", which.c_str()); return 2; }
    if (list) { for (auto& s : scenes) printf("%s\n", s.name.c_str()); return 0; }

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

    gW = W; gH = H;
    stg = makeHostBuf((size_t)W * H * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    Img a = makeImage(W, H, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    Img b = makeImage(W, H, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    Img o = makeImage(W, H, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    begin();
    for (Img* i : {&a, &b, &o}) imgBarrier(i->img, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    submit();

    framegen::InitInfo info;
    info.phys = phys; info.device = dev; info.gdpa = vkGetDeviceProcAddr; info.width = W; info.height = H;
    info.viewFormat = VK_FORMAT_R8G8B8A8_UNORM; info.getMemProps = vkGetPhysicalDeviceMemoryProperties; info.getProps2 = vkGetPhysicalDeviceProperties2;
    info.settings.motionVersion = framegen::MotionVersion::V3;
    info.settings.flowScale = (uint32_t)fs;
    info.settings.v3ColourWindow = gColourWindow;
    info.settings.v3DetailSplit = gDetailSplit;
    info.settings.v3Blend = (uint8_t)gBlend;
    info.settings.v3Motion = (uint8_t)gMotion;
    info.settings.v3Guard = gGuard;
    if (gHud >= 0.0f) info.settings.hudBonus = gHud;
    if (!gFg.init(info)) { fprintf(stderr, "init failed: %s\n", gFg.initError() ? gFg.initError() : "?"); return 1; }

    printf("v3test %dx%d flowScale %d on %s%s\n", W, H, fs, props.deviceName, selfcheck ? " (selfcheck)" : "");
    int selfFails = 0;
    if (which == "all" || which == "decide") {
        // the decisions on pairs that are not ordinary: a cut (two unrelated frames) is mode 1, noise against a picture is failMix 1; a lighting change (gamma) is neither
        // (task 103) the scene cut is off by default (Settings::v3SceneCut = 0, never): the cut rows set the old threshold 0.45 explicitly.
        // (task 112) the row "cut (default)" runs the same unrelated pair with the default settings and "cut (all off)" with every fallback
        // setting off (v3SceneCut 0, globalFallback 1.0, trust 254 / 255, sceneCutSad 255): the hard rule of v3_decide (the two frames do not
        // correspond) makes both a cut, and every generated frame is the nearer real frame. (Before task 112 "cut (default)" expected a
        // generated frame: the cross-fade that the owner's mosaic came from.)
        struct Case { const char* name; int expectMode; float expectFail; float thresh; };
        const Case cases[] = {{"cut", 1, -1, 0.45f}, {"noise", 0, 1, 0.45f}, {"gamma 0.5", 0, 0, 0.45f}, {"cut (default)", 1, -1, -1.0f}, {"cut (all off)", 1, -1, -2.0f}};
        for (const Case& cs : cases) {
            framegen::Settings st = info.settings;
            if (cs.thresh >= 0.0f) st.v3SceneCut = cs.thresh;
            if (cs.thresh == -2.0f) { st.v3SceneCut = 0.0f; st.globalFallback = 1.0f; st.trustLow = 254; st.trustHigh = 255; st.sceneCutSad = 255.0f; }
            gFg.setSettings(st);
            Scene A, B;
            A.W = B.W = W; A.H = B.H = H;
            A.layers = {plane(1, 0.5 * W, 0.5 * H)};
            B.layers = {plane(9, 0.5 * W, 0.5 * H)};
            std::vector<uint8_t> pa = renderFrame(A, 0.0), pb = renderFrame(B, 0.0);
            if (!strncmp(cs.name, "cut", 3)) for (size_t i = 0; i < pb.size() / 4; i++) {   // another scene: big dark / bright blocks (the same generator with another seed, or an inverted copy, has the same level statistics: not a cut a histogram could see)
                const size_t x = i % W, y = i / W;
                const uint8_t v = ((x / 240 + y / 135) & 1) ? 235 : 20;
                pb[i * 4] = pb[i * 4 + 1] = pb[i * 4 + 2] = v;
            }
            if (!strcmp(cs.name, "noise")) { for (size_t i = 0; i < pb.size(); i++) pb[i] = i % 4 == 3 ? 255 : (uint8_t)(hash3((uint32_t)i, 3u, 5u) & 255u); }
            if (!strcmp(cs.name, "gamma 0.5")) { pb = pa; for (size_t i = 0; i < pb.size(); i++) if (i % 4 != 3) pb[i] = (uint8_t)(255.0 * std::pow(pa[i] / 255.0, 0.5) + 0.5); }
            upload(pa, a, W, H);
            upload(pb, b, W, H);
            Run r;
            const bool rok = runV3(a, b, o, ts, nullptr, r);
            const bool ok = rok && r.mode == cs.expectMode && (cs.expectFail < 0 || r.failMix == cs.expectFail);
            printf("%-20s decide mode %d  failMix %.2f  histogram D %.3f (v3SceneCut %.2f)   %s (expected mode %d%s)\n", cs.name, r.mode, r.failMix, r.histD, st.v3SceneCut, ok ? "PASS" : "FAIL", cs.expectMode, cs.expectFail < 0 ? "" : ", failMix 1.00 (or 0.00 for the gamma pair)");
            verdict(ok, true);
            if (false) {   // (before task 112) the default on a true cut: one generated frame per t, not a copy of either real frame
                for (size_t k = 0; k < ts.size(); k++) {
                    size_t same0 = 0, same1 = 0;
                    double d0 = 0.0, d1 = 0.0;
                    for (size_t i = 0; i < pa.size(); i++) {
                        if (i % 4 == 3) continue;
                        same0 += r.out[k][i] == pa[i]; same1 += r.out[k][i] == pb[i];
                        d0 += std::abs((int)r.out[k][i] - (int)pa[i]); d1 += std::abs((int)r.out[k][i] - (int)pb[i]);
                    }
                    const double nv = pa.size() * 0.75;
                    const bool okc = d0 / nv > 1.0 && d1 / nv > 1.0;
                    printf("%-20s default output t=%.2f: mean |out - prev| %.1f, |out - curr| %.1f levels (identical values %.1f%% / %.1f%%)   %s (a generated frame, not a held real one)\n", cs.name, ts[k],
                           d0 / nv, d1 / nv, 100.0 * same0 / nv, 100.0 * same1 / nv, okc ? "PASS" : "FAIL");
                    verdict(okc, true);
                }
            }
            if (rok && ts.size() && r.mode == 1) {   // a cut: every generated frame is a real frame (prev for t <= 0.5, curr after)
                for (size_t k = 0; k < ts.size(); k++) {
                    const std::vector<uint8_t>& want = ts[k] <= 0.5 ? pa : pb;
                    size_t bad = 0;
                    for (size_t i = 0; i < want.size(); i++) bad += r.out[k][i] != want[i];
                    printf("%-20s cut output t=%.2f: %zu differing bytes from the real %s frame   %s\n", cs.name, ts[k], bad, ts[k] <= 0.5 ? "previous" : "current", bad == 0 ? "PASS" : "FAIL");
                    verdict(bad == 0, true);
                }
            }
        }
        gFg.setSettings(info.settings);
    }
    if (which == "all" || which == "border") borderScenes(info.settings, a, b, o, W, H);
    if (which == "all" || which == "motion") motionSwitch(info, a, b, o, W, H);
    for (const Scene& sc : scenes) {
        const Flows F = exactFlows(sc, fs);
        const std::vector<uint8_t> prev = renderFrame(sc, 0.0), curr = renderFrame(sc, 1.0);
        upload(prev, a, W, H);
        upload(curr, b, W, H);
        if (!gSave.empty()) {
            std::string base = gSave + "-" + sc.name;
            for (char& ch : base) if (ch == '(' || ch == ')' || ch == ',' || ch == '%' || ch == '+') ch = '_';
            for (int i = 0; i < 2; i++) if (FILE* f = fopen((base + (i ? "-c.rgba" : "-p.rgba")).c_str(), "wb")) { fwrite((i ? curr : prev).data(), 1, prev.size(), f); fclose(f); }
        }
        const size_t N = (size_t)F.fw * F.fh;
        // the injected exact flows: F01 then F10 (fw * fh vec2 each)
        HostBuf fb = makeHostBuf(N * 16, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        memcpy(fb.map, F.f01.data(), N * 8);
        memcpy(fb.map + N * 8, F.f10.data(), N * 8);

        // ---- (iii) injected flows come back bit-exact (FD)
        Run inj;
        const bool injOk = runV3(a, b, o, ts, &fb, inj);
        bool rtOk = false;
        if (injOk) {
            const size_t n8 = N * 8;
            rtOk = !memcmp(inj.dump.data() + inj.L.fd0, F.f01.data(), n8) && !memcmp(inj.dump.data() + inj.L.fd1, F.f10.data(), n8);
        }

        if (selfcheck) {
            printf("selfcheck %-18s ", sc.name.c_str());
            size_t n10 = 0, n01 = 0;
            // (fade: curr = 0.65 prev' + 12 for the same content: the warp is compared after that gain / offset, or its inverse)
            const double e10 = warpError(sc, F, prev, curr, F.f10, F.vis10, &n10, sc.fade ? 0.65 : 1.0, sc.fade ? 12.0 : 0.0);   // prev warped by F10 -> curr
            const double e01 = warpError(sc, F, curr, prev, F.f01, F.vis01, &n01, sc.fade ? 1.0 / 0.65 : 1.0, sc.fade ? -12.0 / 0.65 : 0.0);   // curr warped by F01 -> prev
            const bool okWarp = n10 && n01 && e10 <= 2.5 && e01 <= 2.5;
            const bool okR0 = renderFrame(sc, 0.0) == prev && renderFrame(sc, 1.0) == curr;
            std::vector<uint8_t> s0, s1;
            hiddenMask(sc, 0.0, &s0, 1);   // at t = 0 the prev side shows every pixel: no strip from it
            hiddenMask(sc, 1.0, &s1, 2);
            size_t h0 = 0, h1 = 0;
            for (uint8_t v : s0) h0 += v;
            for (uint8_t v : s1) h1 += v;
            const bool okHid = h0 == 0 && h1 == 0;
            printf("(i) warp err F10 %.2f (%zu px) F01 %.2f (%zu px) %s | (ii) R0==prev,R1==curr %s, prev side at t=0 / curr side at t=1 hidden: %zu / %zu %s | (iii) inject readback bit-exact %s\n",
                   e10, n10, e01, n01, okWarp ? "ok" : "FAIL", okR0 ? "ok" : "FAIL", h0, h1, okHid ? "ok" : "FAIL", rtOk ? "ok" : "FAIL");
            selfFails += !okWarp + !okR0 + !okHid + !rtOk;
            for (double t : ts) {   // (iv) the oracle blend against R_t: mean <= 1.0 (the threshold of V3's injected path must be reachable on these textures)
                const BetStat s = betweenStat(oracleBlend(sc, prev, curr, t), renderFrame(sc, t), hiddenMask(sc, t));
                printf("                                  (iv) oracle gather blend vs R_t at t=%.2f: mean %.2f p99 %.2f <=3 levels %.1f%% (non-hidden %.1f%%) %s\n", t, s.mean, s.p99, s.le3, s.frac, s.mean <= 1.0 ? "ok" : "FAIL");
                selfFails += !(s.mean <= 1.0);
            }
            destroyHostBuf(fb);
            continue;
        }

        // ---- estimated path
        Run est;
        const bool estOk = runV3(a, b, o, ts, nullptr, est);
        const std::string nm = sc.name;
        const double flowThr = sc.fade ? 90.0 : 95.0;
        if (estOk) {
            {   // the pair's decisions (v3_decide): an ordinary pair is no cut and no failed estimate (the fade scene included: a lighting change is not a cut)
                const uint32_t* G = (const uint32_t*)(est.dump.data() + est.L.g);
                const double N2 = 2.0 * F.fw * F.fh, share = 100.0 * est.unsure / N2, shareE = 100.0 * G[31] / std::max((double)G[30], 0.25 * N2);
                const bool ok = est.mode == 0 && est.failMix == 0.0f;
                printf("%-20s decide mode %d  failMix %.2f  histogram D %.3f (v3SceneCut %.2f)  unsure %.1f%% of flow px, %.1f%% of those that could find a partner (globalFallback 15%%)   %s (an ordinary pair: mode 0, failMix 0)\n", nm.c_str(),
                       est.mode, est.failMix, est.histD, info.settings.v3SceneCut, share, shareE, ok ? "PASS" : "FAIL");
                verdict(ok, true);
            }
            if (gSShare) for (size_t k = 0; k < ts.size() && k < est.sShare.size(); k++) {
                printf("%-20s smear  t=%.2f  share of pixels with s > 0.1: %.2f%%\n", nm.c_str(), ts[k], est.sShare[k]);
                if (!gFrames.empty()) {
                    std::vector<uint8_t> img((size_t)gW * gH * 4, 255);
                    for (size_t i = 0; i < (size_t)gW * gH; i++) { const uint8_t v = (uint8_t)std::lround(255.0f * std::min(1.0f, std::max(0.0f, est.sMap[k][i]))); img[i * 4] = img[i * 4 + 1] = img[i * 4 + 2] = v; }
                    std::string base = gFrames + "-" + sc.name + "-t" + std::to_string(ts[k]).substr(0, 4) + "-smap.ppm";
                    for (char& ch : base) if (ch == '(' || ch == ')' || ch == ',' || ch == '%') ch = '_';
                    writePpm(base, img, gW, gH, false, nullptr);
                }
            }
            {   // the stats of the estimate (state G): dominant motions, the camera model, the wide search's counters
                const uint32_t* G = (const uint32_t*)(est.dump.data() + est.L.g);
                float gf[64];
                memcpy(gf, G, sizeof gf);
                printf("%-20s stats  dom", nm.c_str());
                for (int k = 0; k < 4; k++) if (G[6] & (1u << k)) printf(" (%.1f,%.1f)", gf[8 + 2 * k], gf[9 + 2 * k]); else printf(" -");
                if (G[6] & 256u) printf("  aff dx = %.2f %+.2f X %+.2f Y, dy = %.2f %+.2f X %+.2f Y (X, Y in half frames)", gf[16], gf[17], gf[18], gf[19], gf[20], gf[21]);
                else printf("  aff invalid");
                printf("  wide flagged %u replaced %u\n", G[25], G[26]);
                if (est.L.nLevels >= 3) {
                    char msg[600];
                    const bool ok = statsCheck(sc, F, G, est.L.nLevels, msg, sizeof msg);
                    printf("%-20s stats check: %s   %s\n", nm.c_str(), msg, ok ? "PASS" : "FAIL");
                    verdict(ok, true);
                }
            }
            {   // the ambiguity measure of the level-0 blocks (the level-1 probes' result), over the visible blocks: share above 0.5, and the same over the blocks whose vector is right
                const int gw = est.L.lgw[0], gh = est.L.lgh[0];
                printf("%-20s amb   ", nm.c_str());
                for (int d = 0; d < 2; d++) {
                    const float* r = (const float*)(est.dump.data() + est.L.blkLevel[0] + (size_t)(d * 2) * gw * gh * 16);
                    size_t n = 0, hi = 0, nOk = 0, hiOk = 0, hiBad = 0;
                    for (int by = 0; by < gh; by++)
                        for (int bx = 0; bx < gw; bx++) {
                            const int mx = std::min(F.fw - 1, bx * 8 + 4), my = std::min(F.fh - 1, by * 8 + 4);
                            const size_t i = (size_t)my * F.fw + mx;
                            if (!(d ? F.vis10 : F.vis01)[i]) continue;
                            const float* q = r + 4 * (by * gw + bx);
                            const std::vector<float>& tr = d ? F.f10 : F.f01;
                            const bool right = F.fs * std::hypot(q[0] - tr[2 * i], q[1] - tr[2 * i + 1]) <= 1.0;
                            n++; hi += q[3] > 0.5f; nOk += right; hiOk += right && q[3] > 0.5f; hiBad += !right && q[3] > 0.5f;
                        }
                    printf(" %s: %.1f%% of visible blocks above 0.5 (%.1f%% of the right ones, %.1f%% of the wrong ones)", d ? "F10" : "F01", n ? 100.0 * hi / n : 0.0, nOk ? 100.0 * hiOk / nOk : 0.0, n > nOk ? 100.0 * hiBad / (n - nOk) : 0.0);
                }
                printf("\n");
            }
            if (gBlkLevel >= 0 && (uint32_t)gBlkLevel < est.L.nLevels) {   // --blk L: the block records of level L, F01 slot 0 and slot 1 (vx,vy / cost / amb%)
                const int l = gBlkLevel, gw = est.L.lgw[l], gh = est.L.lgh[l];
                for (int slot = 0; slot < 2; slot++) {
                    printf("%s level %d F01 slot %d (vx,vy/cost/amb%%)\n", nm.c_str(), l, slot);
                    const float* r = (const float*)(est.dump.data() + est.L.blkLevel[l] + (size_t)(slot) * gw * gh * 16);
                    for (int y = 0; y < gh; y++) {
                        for (int x = 0; x < gw; x++) { const float* q = r + 4 * (y * gw + x); printf(" %g,%g/%.1f/%d", q[0], q[1], q[2], (int)std::lround(100 * q[3])); }
                        printf("\n");
                    }
                }
            }
            if (gDiag) {   // the level-0 block vectors (before the dense pass) against the exact flow at the block centres, visible ones
                const int gw = est.L.lgw[0], gh = est.L.lgh[0];
                for (int d = 0; d < 2; d++) {
                    const float* r = (const float*)(est.dump.data() + est.L.blkLevel[0] + (size_t)(d * 2) * gw * gh * 16);
                    size_t n = 0, c05 = 0, c1 = 0;
                    for (int by = 0; by < gh; by++)
                        for (int bx = 0; bx < gw; bx++) {
                            const int mx = std::min(F.fw - 1, bx * 8 + 4), my = std::min(F.fh - 1, by * 8 + 4);
                            const size_t i = (size_t)my * F.fw + mx;
                            if (!(d ? F.vis10 : F.vis01)[i]) continue;
                            const float* q = r + 4 * (by * gw + bx);
                            const std::vector<float>& tr = d ? F.f10 : F.f01;
                            const double e = F.fs * std::hypot(q[0] - tr[2 * i], q[1] - tr[2 * i + 1]);
                            n++; c05 += e <= 0.5; c1 += e <= 1.0;
                        }
                    printf("%-20s level-0 blocks %s: visible %zu, within 0.5 px %.1f%%, 1 px %.1f%%\n", nm.c_str(), d ? "F10" : "F01", n, n ? 100.0 * c05 / n : 0.0, n ? 100.0 * c1 / n : 0.0);
                }
            }
            const float* f01 = (const float*)(est.dump.data() + est.L.ff0);
            const float* f10 = (const float*)(est.dump.data() + est.L.ff1);
            const FlowStat s01 = flowStat(F, f01, F.f01, F.vis01), s10 = flowStat(F, f10, F.f10, F.vis10);
            const FlowStat* ss[2] = {&s01, &s10};
            for (int d = 0; d < 2; d++) {
                const bool pass = ss[d]->n && ss[d]->w1 >= flowThr;
                printf("%-20s flow %s    visible %5.1f%%  within 0.5 px %5.1f%%  1 px %5.1f%%  2 px %5.1f%%  zero-though-moving %5.2f%%   %s (>= %.0f%% within 1 px)\n", nm.c_str(), d ? "F10" : "F01",
                       ss[d]->vis, ss[d]->w05, ss[d]->w1, ss[d]->w2, ss[d]->zero, pass ? "PASS" : "FAIL", flowThr);
                verdict(pass, true);
                if (gDiag && (sc.layers.size() > 1 || !pass || !gErrMap.empty())) diagFlow(sc, F, d ? f10 : f01, d ? F.f10 : F.f01, d ? F.vis10 : F.vis01, d, nm);
            }
        } else { printf("%-20s flow: the V3 estimate failed to record\n", nm.c_str()); verdict(false, true); }
        // ---- in-between, V3's own flows and the exact ones
        for (size_t k = 0; k < ts.size(); k++) {
            const std::vector<uint8_t> ref = renderFrame(sc, ts[k]);
            const std::vector<uint8_t> hid = hiddenMask(sc, ts[k]);
            for (int path = 0; path < 2; path++) {
                const Run& r = path ? inj : est;
                if (!(path ? injOk : estOk) || r.out.size() <= k) { printf("%-20s inbetween %s t=%.2f: the run failed\n", nm.c_str(), path ? "exact flows" : "estimated  ", ts[k]); verdict(false, true); continue; }
                const BetStat s = betweenStat(r.out[k], ref, hid);
                if (!gFrames.empty()) {
                    char tb[64];
                    snprintf(tb, sizeof tb, "-t%.2f-%s", ts[k], path ? "exact" : "est");
                    const std::string tn = tb;
                    std::string base = gFrames + "-" + sc.name + tn;
                    for (char& ch : base) if (ch == '(' || ch == ')' || ch == ',' || ch == '%') ch = '_';
                    writePpm(base + "-out.ppm", r.out[k], gW, gH, false, nullptr);
                    writePpm(base + "-err.ppm", r.out[k], gW, gH, true, &ref);
                    if (path == 0) { std::string br = gFrames + "-" + sc.name + "-t" + tn.substr(2, 4) + "-ref.ppm"; for (char& ch : br) if (ch == '(' || ch == ')' || ch == ',' || ch == '%') ch = '_'; writePpm(br, ref, gW, gH, false, nullptr); }
                }
                const bool gated = path == 1 || sc.name.rfind("shift", 0) == 0;
                const double thr = path ? 1.0 : 2.0;
                const bool pass = s.n && s.mean <= thr;
                printf("%-20s inbetween %s t=%.2f  mean %6.2f  p99 %6.2f  <=3 levels %5.1f%%  (non-hidden %5.1f%%)   %s (mean <= %.1f%s)\n", nm.c_str(),
                       path ? "exact flows" : "estimated  ", ts[k], s.mean, s.p99, s.le3, s.frac, gated ? (pass ? "PASS" : "FAIL") : "INFO", thr, gated ? "" : ", informational");
                verdict(pass, gated);
            }
            const bool shiftScene = sc.layers.size() == 1 && sc.name.rfind("shift", 0) == 0;
            if (sc.layers.size() > 1 || shiftScene) {   // (shift scenes: the strip entering at the leading edge, no partner in the other frame; informational, tuning D3)
                // INSIDE the hidden strips (content absent in one real frame; not within 2 px of a layer boundary): ours against the plain time blend
                std::vector<uint8_t> strips, band;
                hiddenMask(sc, ts[k], &strips, 3, &band);
                std::vector<uint8_t> notInside(strips.size());
                for (size_t i = 0; i < strips.size(); i++) notInside[i] = !(strips[i] && !band[i]);   // (betweenStat skips the mask: the strip interior is what is left)
                std::vector<uint8_t> blend(prev.size());
                for (size_t i = 0; i < prev.size(); i++) blend[i] = (uint8_t)std::lround((1.0 - ts[k]) * prev[i] + ts[k] * curr[i]);
                const BetStat sb = betweenStat(blend, ref, notInside);
                for (int path = 0; path < 2; path++) {
                    const Run& r = path ? inj : est;
                    if (r.out.size() <= k) continue;
                    const BetStat so = betweenStat(r.out[k], ref, notInside);
                    printf("%-20s %s %s t=%.2f  mean %6.2f  p99 %6.2f  (%.1f%% of the frame)   plain blend: mean %6.2f  p99 %6.2f\n", nm.c_str(), shiftScene ? "entering edge" : "hidden strips", path ? "exact flows" : "estimated  ", ts[k], so.mean, so.p99, so.frac, sb.mean, sb.p99);
                }
            }
        }
        if (gProfile && estOk && sc.layers.size() > 1) {   // o_eff of F01 along the middle row of the estimate: where it changes, over how many flow px
            const float* f01 = (const float*)(est.dump.data() + est.L.ff0);
            const uint32_t* mp = (const uint32_t*)(est.dump.data() + est.L.map0);
            const int y = est.L.fh / 2;
            std::vector<float> o(est.L.fw);
            for (uint32_t x = 0; x < est.L.fw; x++) {
                const size_t i = (size_t)y * est.L.fw + x;
                const double e = ((mp[i] >> 8) & 255) / 8.0, g = (mp[i] >> 24) / 255.0, F = std::hypot(f01[2 * i], f01[2 * i + 1]);
                double r = e / (0.6 + 0.02 * F);
                r = std::min(1.0, std::max(0.0, (r - 1.0) / 1.5));
                o[x] = (float)(r * r * (3 - 2 * r) * (1 - g));
            }
            printf("%-20s occlusion o_eff (F01, middle row, flow px) transitions (0.05 < o < 0.95 counted):", nm.c_str());
            for (uint32_t x = 1; x < est.L.fw; x++) {
                const bool lo0 = o[x - 1] < 0.05f, hi1 = o[x] >= 0.95f, hi0 = o[x - 1] >= 0.95f, lo1 = o[x] < 0.05f;
                if ((lo0 && !hi1 && !lo1) || (hi0 && !lo1 && !hi1)) {   // a transition starts at x
                    int w = 0;
                    uint32_t q = x;
                    while (q < est.L.fw && o[q] >= 0.05f && o[q] < 0.95f) { w++; q++; }
                    printf("  [x=%u width %d:", x, w);
                    for (int k = 0; k < w + 1 && k < 24; k++) printf(" %.2f", o[x + k]);
                    printf("]");
                    x = q;
                }
            }
            printf("\n");
        }
        if (sc.name.rfind("hud", 0) == 0 && estOk) {
            // a static overlay over a moving background (Settings::v2StaticMask is V3's switch too): with the mask the overlay stays at its place (small error on it, none in the ring around it
            // where a dragged copy would show) and the background still moves (far error small); without it the check has teeth (the ring shows the copies)
            HudStat on[8], off[8];
            for (int maskOn = 1; maskOn >= 0; maskOn--) {
                framegen::Settings st = info.settings;
                st.v2StaticMask = maskOn != 0;
                gFg.setSettings(st);
                Run r;
                const bool rok = runV3(a, b, o, ts, nullptr, r);
                gFg.setSettings(info.settings);
                for (size_t k = 0; rok && k < ts.size() && k < 8; k++) (maskOn ? on : off)[k] = hudStat(sc, r.out[k], renderFrame(sc, ts[k]), ts[k]);
            }
            for (size_t k = 0; k < ts.size() && k < 8; k++) {
                const bool pass = on[k].ovl <= 7.0 && on[k].ring <= 4.0 && on[k].ringBad <= 0.5 && on[k].far <= 0.5;
                const bool teeth = off[k].ringBad >= 2.0 * on[k].ringBad + 0.5 || off[k].ovl >= on[k].ovl + 1.0;
                printf("%-20s static overlay t=%.2f  mask ON: overlay %5.2f (%zu px)  ring %5.2f  ring > 40 levels %5.2f%% (%zu px)  background far %5.2f   %s (overlay <= 7, ring <= 4, ring > 40 <= 0.5%%, far <= 0.5; measured t=0.25 / 0.5: 5.5 / 5.7, 2.8 / 2.95, 0.21 / 0.08, 0.08)\n", nm.c_str(), ts[k],
                       on[k].ovl, on[k].nOvl, on[k].ring, on[k].ringBad, on[k].nRing, on[k].far, pass ? "PASS" : "FAIL");
                printf("%-20s static overlay t=%.2f  mask OFF: overlay %5.2f           ring %5.2f  ring > 40 levels %5.2f%%             background far %5.2f   %s (the check has teeth: OFF worse than ON)\n", nm.c_str(), ts[k],
                       off[k].ovl, off[k].ring, off[k].ringBad, off[k].far, teeth ? "PASS" : "FAIL");
                verdict(pass, true);
                verdict(teeth, true);
            }
        }
        printf("%-20s inject readback (FD == the injected flows, bit-exact): %s\n", nm.c_str(), rtOk ? "PASS" : "FAIL");
        verdict(rtOk, true);
        destroyHostBuf(fb);
    }
    vkDeviceWaitIdle(dev);
    gFg.destroy();
    destroyHostBuf(stg);
    destroyImage(a); destroyImage(b); destroyImage(o);
    if (selfcheck) { printf("selfcheck: %s (%d failed)\n", selfFails ? "FAIL" : "PASS", selfFails); return selfFails ? 1 : 0; }
    printf("v3test: %s (%d metric%s failed)%s\n", gFails ? "FAIL" : "PASS", gFails, gFails == 1 ? "" : "s", gReportOnly ? " (--report-only: exit 0)" : "");
    return gFails && !gReportOnly ? 1 : 0;
}
