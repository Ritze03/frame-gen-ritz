// framegen: image-only frame interpolation (Vulkan compute port of proto/interp.py).
// Plain Vulkan, no gamescope types. Never submits, never touches queues.
//
// Threading / queues: not thread-safe. All records of one Interpolator must be submitted to ONE queue,
// in recording order: its internal buffers are shared between records and only ordered by pipeline
// barriers, which do not order against work on another queue.
#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <vector>

namespace framegen {

// Static-UI protection mode (Settings::uiProtection); the contract is documented at recordObserve().
enum class UiProtection : uint32_t {
    Off = 0,          // nothing is recorded or allocated: output byte-identical to a library without it
    Crosshair = 1,    // full-resolution protection of one box (default: the centre): crosshair, reticle, hit marker
    WholeScreen = 2,  // the box at full resolution, plus every solid static HUD element elsewhere (reduced-res detection)
    CrosshairV2 = 3,  // the crosshair box only, without inpaint or clean copy: contrast-gated detection with hysteresis and fast
                      // re-entry, a pixel is pasted back only if it did not change in the pair being synthesised, and a warp tap
                      // that lands on the UI is rejected so the crosshair is not copied along the background's motion (optionally
                      // the UI is also hidden from the motion estimate: uiEstimateFill). Next to Crosshair (V1, unchanged).
                      // See "CrosshairV2" below.
};

// Motion algorithm (Settings::motionVersion): the motion estimate and the warp / mismatch / resolve passes that turn it into
// generated frames. Values are append-only and never renumbered. Runtime: setSettings() switches between them (see there).
enum class MotionVersion : uint32_t {
    V1 = 0,   // the algorithm of proto/interp.py (the frozen reference), every Format: output byte-identical to a library without the
              // choice. The default.
    V2 = 1,   // the second algorithm, next to V1 (unchanged): a "morphing" interpolator (content is warped along a smooth field and
              // stretched over gaps; no real frame is shown locally, only the whole-frame scene-cut / flick fallbacks). Rgba8 only:
              // settingsError() refuses it with any other Format. Settings::v2Mode picks between its two modes. Its motion estimate
              // computes V1's block field with faster passes (shared by both modes).
};

// The two modes of MotionVersion::V2 (Settings::v2Mode). Values are append-only and never renumbered. Runtime: setSettings() at any
// time, no rebuild, no pipeline creation; a switch keeps the current estimate (the part of the estimate the other mode needs is
// derived by its first recordSynth / recordSynthBlur, with the same output as that mode's own estimate). V1 ignores it.
enum class V2Mode : uint32_t {
    Quality = 0,       // the default: the better picture for more GPU time, the "coverage morph". Performance's cleaned block
                       // field, upsampled per flow pixel with edge awareness (motion edges follow objects instead of ramping over
                       // a block) and zero on static overlays (subtitles, HUD); the field splatted to both real frames to know which
                       // frame still shows each piece of content; one full-resolution gather like Performance's whose blend leans
                       // to that frame (no 50/50 double image at occlusions and at the frame border), whose taps skip static
                       // overlays, blurred along the motion where the two taps still disagree. No per-pixel trust or hold.
                       // RX 7900 XTX, 1080p 2x: ~0.24 ms per pair (V1 0.30, Performance 0.14).
    Performance = 1,   // the "grid morph": the finest block field (16x16 px cells at flowScale 2) re-picked among neighbours,
                       // refined and cleaned (vector median + light smoothing), upsampled bilinearly to full resolution and taken to
                       // time t per pixel; one full-resolution two-tap blend (and a zero-vector candidate where the two frames agree,
                       // for HUD text and static overlays). No per-pixel trust or hold. RX 7900 XTX, 1080p 2x: ~0.15 ms per pair.
};

// Weights of the sub-frames of one motion-blurred output (Settings::blurWeights, recordSynthBlur()).
enum class BlurWeights : uint32_t {
    Even = 0,       // every sub-frame 1/N: a plain box shutter
    Gaussian = 1,   // w_k ~ exp(-2 x_k^2), x_k = (2k + 1 - N) / N, the slice centre mapped so the window ends are x = -+1: a Gaussian whose sigma is 1/4 of the
                    // window, so the ends of the window carry exp(-2) = 13.5 % of the centre's weight
};

// Image formats (README.md / INTEGRATION.md "Formats"). Chosen at init() by InitInfo::viewFormat, which maps to exactly one:
//   Rgba8    B8G8R8A8_UNORM, R8G8B8A8_UNORM                            the default; the shaders and output are unchanged
//   Rgb10    A2B10G10R10_UNORM_PACK32, A2R10G10B10_UNORM_PACK32
//   RgbaF16  R16G16B16A16_SFLOAT                                       linear / scRGB / encoded, see InitInfo::transfer
//   Nv12     R8_UNORM (the Y plane view), chroma R8G8_UNORM            8-bit 4:2:0, two plane views per frame (Frame::chroma)
//   P010     R16_UNORM (the Y plane view), chroma R16G16_UNORM         10-bit 4:2:0 in the high bits of 16-bit words
// Only the format-specific shaders differ per format (the motion estimate runs on luma and is the same code for all).
enum class Format : uint32_t { Rgba8 = 0, Rgb10 = 1, RgbaF16 = 2, Nv12 = 3, P010 = 4 };
// The format class of a view format, false if the library does not know it.
bool formatOf(VkFormat viewFormat, Format* out);

// How RgbaF16 values are encoded (InitInfo::transfer). Only the mismatch / stillness measurement depends on it; the
// frames are always warped and blended in their own space.
enum class Transfer : uint32_t {
    Encoded = 0,   // display-encoded values (sRGB / gamma / PQ / HLG): the level of a channel is value * 255
    Linear = 1,    // linear light (scRGB, or any linear float): 1.0 = InitInfo::linearWhiteNits; measured as PQ codes
};

// The colour matrix of a YUV format (InitInfo::yuvMatrix).
enum class YuvMatrix : uint32_t { Bt601 = 0, Bt709 = 1, Bt2020 = 2 };

// One image as the library takes it: the view, and for the two-plane formats (Nv12, P010) the CbCr plane view. A plain
// VkImageView converts implicitly, so a host that only uses RGB formats passes views exactly as before.
struct Frame {
    VkImageView view = VK_NULL_HANDLE;     // the image view (RGB formats), or the Y plane view (Nv12, P010)
    VkImageView chroma = VK_NULL_HANDLE;   // Nv12 / P010: the CbCr plane view (half resolution); ignored otherwise
    Frame() = default;
    Frame(VkImageView v, VkImageView c = VK_NULL_HANDLE) : view(v), chroma(c) {}
};

// Tuning knobs. The defaults are the tuned values of proto/interp.py (the approved look); with them the
// output is byte-identical to the library without settings. Costs and mismatches are in 8-bit luma/RGB
// levels (0-255) in EVERY format: 10-bit and float input is measured as the levels of an 8-bit picture of the same
// content (10-bit codes / 4, PQ codes, YCbCr differences converted to R'G'B'), so a value means the same perceived
// amount in all of them. Interpolator::validate() states the accepted ranges; anything else is rejected.
struct Settings {
    // ---- runtime: setSettings() at any time; applies to every record*() call recorded after it ----
    // Per-pixel fallback ramp on the 7x7-averaged mismatch |warped prev - warped curr|: below trustLow the
    // pixel is interpolated, above trustHigh it shows the nearer real frame, linear in between.
    // [0, 255], trustHigh >= trustLow + 1. Lower = safer (fewer ghosts/smears, more held pixels, judder on
    // hard areas); higher = smoother but more visible artefacts at occlusions.
    float trustLow = 12.0f;
    float trustHigh = 40.0f;
    // Whole-frame fallback: if more than this share of pixels is untrusted (fast flick, chaos), present the
    // nearer real frame instead. (0, 1]; 1 disables it. Lower = safer, higher = interpolates more.
    float globalFallback = 0.15f;
    // Scene cut: finest-level mean block SAD above this presents the nearer real frame. (0, 255];
    // 255 disables it. Lower = holds more often (safer), higher = tries to interpolate cuts (blends them).
    float sceneCutSad = 30.0f;
    // Static-HUD protection: per-pixel cost bonus for the zero vector, so overlays, crosshairs and text
    // stay put instead of being dragged along with the background. [0, 255]; 0 = off, useful 0-4.
    // Higher = HUD steadier, but slow motion near-still areas may freeze (judder).
    float hudBonus = 1.0f;
    // Per-pixel cost bonus for the smooth (bilinear) block field over the sharper per-block candidates.
    // [0, 255]; useful 0-2. Higher = smoother, fewer speckles, softer motion edges; lower = sharper edges.
    float smoothBonus = 0.5f;
    // Motion search: cost per pixel step away from the predicted vector (and for leaving the parent's
    // vector). [0, 255]; useful 0.25-2. Higher = steadier vectors in flat/repetitive areas (less shimmer),
    // lower = follows small or fast detail more freely.
    float searchPenalty = 0.5f;
    // Sub-pixel (1/2 then 1/4 px) refinement of the block vectors. Off = ~0.05 ms cheaper per real pair,
    // slightly softer/wobblier output on slow pans.
    bool subpixel = true;

    // Static-UI protection: find pixels that have not changed for uiStillFrames real frames (a crosshair, a HUD
    // box, a minimap frame), remove them from what the motion estimate and the warps see (inpainted from their
    // surroundings, in the library's own copy of the frame), and paste the exact real pixels back over every
    // generated frame. The host's frames are only read. Unlike hudBonus it works per pixel: a thin crosshair that
    // shares a block with moving background neither drags the background nor freezes it. Needs recordObserve()
    // (or the implicit observe, see there). Runtime: setSettings() switches it on/off/between modes with no GPU
    // wait (it only ever ALLOCATES, never frees; setSettings returns false if that allocation fails). Off =
    // byte-identical to a library without it.
    // recordTrack() keeps the counters current while the host passes frames through (see there).
    // CrosshairV2 differs: no clean copy, no inpaint; see "CrosshairV2" below for what it does instead, its settings
    // (uiContrastThreshold, uiLeaveFrames, uiReentryFrames, uiEstimateFill) and its limits. Cost (RX 7900 XTX, 1080p and
    // 1440p, GPU time per real frame): observe 0.020 ms + patch 0.002 ms; recordTrack 0.020 ms (it is the whole detect pass).
    // Cost (RX 7900 XTX, 2560x1440, GPU time per real frame): Crosshair ~0.07 ms, WholeScreen ~0.09 ms typical
    // and <= ~0.25 ms in the worst case below the uiMaxStillFraction skip (about half of it is the full-frame copy
    // into the clean image). Memory: uiHistory (default kUiFrames = 3) clean copies of the frame (Rgba8 and Rgb10 4 bytes/px each: 44 MB
    // at 1440p for 3, 14.7 MB per kept frame; RgbaF16 8, Nv12 1.5, P010 3), under 1 MB for the box, and for WholeScreen about 1 byte/px of mask per
    // kept frame plus the 1/2-res cell state (+ ~15 MB at 1440p).
    UiProtection uiProtection = UiProtection::Off;
    // How many observed real frames keep their clean copy + mask (LRU by observe order), [kUiFrames, kUiFramesMax].
    // 3 = the default: the newest pair and the pair before it. A host that synthesises from an OLDER pair (the
    // pacer's lag-spike buffer plays pairs several frames back) sets it to the number of real frames it keeps:
    // pacing.h's Pacer::HistoryDepth() / Report::historyFrames, which follows the buffer. Growing allocates the
    // missing clean copies at once (no GPU wait; false from setSettings if that fails, nothing changed); the new
    // slots are empty until observed. Lowering it evicts the least recently observed frames but KEEPS their memory
    // for a later raise; releaseUnusedUi() frees it. With protection Off, or at the default, memory and output are
    // exactly as before.
    uint32_t uiHistory = 3;
    // The crosshair box: the region where the library LOOKS FOR static UI, at FULL resolution, in all modes (a crosshair is the
    // most noticeable thing if it breaks). What it protects reaches a few pixels past it: Crosshair (V1) and CrosshairV2 treat
    // the box grown by boxMargin() (2 px for V1, 1 px for V2; clamped to the frame) as their working rectangle (workRect()). A
    // pixel can only be DETECTED as UI inside the box, but the anti-aliased rim of what was detected, V1's mask feather, the
    // inpaint's source pixels and the paste-back all extend into that margin. So the box needs to cover the crosshair's own
    // stable pixels, not its whole rim: a box that just fits the reticle works. Sizing it bigger than that has a cost, not a
    // benefit: every box pixel is a candidate for "still", so a box much bigger than the reticle judges background still
    // too, and moving background there shows up as small white speckles. (Why this changed: a user sized the box to "just
    // big enough for the whole crosshair", 11 px at 960 high, around an 8x7 px crosshair; the 2 px mask dilation, the inpaint
    // source and the paste were all clipped at the box edge, so the crosshair's rim was smeared by frame generation and
    // worse by motion blur. With a 192 px box it had been protected.) A host can draw boxRect() / workRect() as a preview
    // while the user adjusts the slider.
    //   uiBoxHeightFrac: (0, 1], the side of the square box as a fraction of the FRAME HEIGHT: side =
    //   round(frameHeight * uiBoxHeightFrac), clamped to [8, min(w, h)]. 0.025 = 24 px at 960, 27 at 1080, 36 at 1440,
    //   54 at 2160. 0 = legacy: take the side from uiCrosshairArea instead (below).
    float uiBoxHeightFrac = 0.025f;
    //   uiBoxWidth/uiBoxHeight: the box size in pixels, [0, 16384]. 0 = take that side from uiBoxHeightFrac.
    //   An explicit size wins over the fraction; it is clamped to [4, frame size].
    uint32_t uiBoxWidth = 0, uiBoxHeight = 0;
    //   uiCrosshairArea: DEPRECATED, kept for API compatibility. Used ONLY when uiBoxHeightFrac == 0: then the box is
    //   a square of side sqrt(uiCrosshairArea * w * h), clamped to [32, min(w, h)] (0.03 = 333 px at 2560x1440).
    //   Range (0, 1] as before. Why: that area box was 192 px on 1280x960 and caught background (see README.md).
    float uiCrosshairArea = 0.03f;
    //   uiBoxOffsetX/Y: where the box is centred, in pixels from the frame centre ([-16384, 16384]; +x right,
    //   +y down; 0 = dead centre). For a game whose reticle is not in the middle. The box is clamped into the frame.
    int32_t uiBoxOffsetX = 0, uiBoxOffsetY = 0;
    // A pixel is "still" if no channel moved more than uiStillThreshold (8-bit levels, [0, 255]) between two
    // consecutive real frames, and is protected after uiStillFrames ([1, 255]) consecutive still frames. Fewer
    // frames = protected sooner (and more of a briefly paused picture is masked, which is harmless);
    // a higher threshold tolerates noisy/animated UI (and masks slowly moving gradients).
    uint32_t uiStillFrames = 8;
    float uiStillThreshold = 3.5f;
    // The inpaint walks at most this many pixels ([1, 16]) in each of 8 directions to the first unprotected pixel.
    // It must exceed half the thickness of the thickest thing to remove (plus the 2 px dilation); beyond that the
    // inside of a big solid box is left as is (it is static, so it costs the estimate nothing).
    uint32_t uiInpaintReach = 12;
    // WholeScreen only: if more than this share of the screen outside the box is still ((0, 1]; 1 = never skip),
    // that frame's whole-screen mask/inpaint/patch passes are skipped on the GPU. A (nearly) static screen needs
    // no generated frame to differ from the real one, and this bounds the worst case of a paused/menu picture.
    float uiMaxStillFraction = 0.75f;
    // CrosshairV2 only (ignored by every other mode); the box, uiStillFrames and uiStillThreshold apply to it as to Crosshair,
    // uiHistory still sizes the kept-frame set (no clean copies are allocated for V2: one small mask per kept frame).
    //   uiContrastThreshold: a pixel can only ENTER the mask if its local contrast c = max channel |I - median7x7(I)| is at
    //   least this ([0, 255], 8-bit-equivalent levels in every format like uiStillThreshold; 0 = no contrast gate). A
    //   crosshair has an edge against its surroundings, a still flat piece of background (sky, a wall, a menu) does not: on
    //   the user's CS2 footage 88-90 % of the still crosshair pixels pass 16 and 77-83 % pass 24 while 0.03-0.08 % (16) / less
    //   (24) of the still background does. Higher = fewer false positives, thin or low-contrast crosshairs are missed.
    float uiContrastThreshold = 20.0f;
    //   uiLeaveFrames: a masked pixel leaves the mask only after this many CONSECUTIVE frames in which it changed more than
    //   uiStillThreshold ([1, 255]). Hysteresis: the mask no longer flickers when the background behind a thin crosshair
    //   shimmers for a frame. 4 default; 1 = leaves at the first change (like V1's mask without the delay).
    uint32_t uiLeaveFrames = 4;
    //   uiReentryFrames: fast re-entry. The colour a pixel had when it was last confirmed is kept; a pixel that left the mask and
    //   shows that colour again (within uiStillThreshold, and contrasty) for this many consecutive frames re-enters at once
    //   instead of after uiStillFrames more still frames: the crosshair is back after a disruption (a flash, a muzzle effect, a
    //   scope) within [0 = off, 255] frames. 2 default.
    uint32_t uiReentryFrames = 2;
    //   uiEstimateFill: also hide the UI from the MOTION ESTIMATE. Off (default): the estimate runs on the host's frames exactly as
    //   with protection Off. On: the flow-resolution luma of every cell that holds a UI pixel is re-averaged over the cell's non-UI
    //   pixels (a wider window where all are UI), so a static crosshair casts no vote for the zero vector in the blocks around it.
    //   The fill exists only in the estimate's luma buffer, never in a colour pass, so it cannot reach an output pixel. The warp's
    //   copy of the crosshair along the background's motion (the "doubled crosshair") is removed by the tap rejection, with or
    //   without the fill; the fill only fixes the flow around the crosshair. Measured on the user's CS2 footage WITH the rejection
    //   (INTEGRATION.md): per frame, 18.4 pixels better than Off and 1.7 worse without the fill, 20.5 better and 3.6 worse with it,
    //   and the same mean error near the crosshair (6.30 / 6.29, Off 6.37): it nets the same (better minus worse: 16.9 and 16.7
    //   px/frame) but doubles the visible defects, some far from the crosshair (re-rolled vectors: in 4 of 1173 frames 80-643
    //   worse pixels there), so the default stays off. On synthetic scenes it
    //   removes what is left of the crosshair's flow vote: smooth background 26 -> 0 artifact pixels around the crosshair; mean
    //   error within 24 px of a thin / thick crosshair on a hard-edged moving background 56.9 / 125.2 for Off, 32.9 / 70.5 with the
    //   fill. Costs ~0.03 ms per pair at 1440p (RX 7900 XTX) and 2 sets of the kUiSets ring per estimate.
    bool uiEstimateFill = false;

    // Interpolation-based motion blur (recordSynthBlur()): one output = the weighted average of blurSamples
    // interpolated sub-frames spread evenly over a window [t0, t1] of the current real pair, a longer, smoother
    // "shutter" than a single instant. 0 or 1 = off (recordSynthBlur then equals recordSynth at the window's end t1).
    // 2..kBlurMaxSamples; any value works: the sub-frames sit at the MIDPOINTS of N equal slices of the window,
    // t_k = t0 + (t1 - t0)(k + 0.5)/N, k = 0..N-1 (a box-filter shutter; no sample on a window end), with the weights of blurWeights. Runtime: setSettings() at any time; the scratch it needs (per sample one
    // field at t and one mismatch image: ~3 B per pixel at flowScale 2, 9 B at flowScale 1) is allocated by the first
    // blurred record and only grows. Off = byte-identical to a library without it.
    // Cost (RX 7900 XTX, per output, on top of the pair's estimate): see "Motion blur" in INTEGRATION.md.
    uint32_t blurSamples = 0;
    BlurWeights blurWeights = BlurWeights::Gaussian;

    // The motion algorithm (see MotionVersion). V1 = the default: output and memory exactly as before. V2 needs Format::Rgba8
    // (settingsError()). Runtime: setSettings() at any time. A switch DROPS the current estimate: recordSynth / recordSynthBlur
    // return false (out untouched, the host shows the real frame, as with no estimate since init/resize) until the next
    // recordEstimate, which runs the new version; an estimate of one version is never synthesised by the other. The first switch to a
    // version creates its pipelines (and later its buffers) at once, no GPU wait (false from setSettings if that fails, nothing
    // changed); switching away keeps them for a later switch back, releaseUnusedMotion() frees them.
    MotionVersion motionVersion = MotionVersion::V1;
    // The mode of MotionVersion::V2 (see V2Mode). Quality = the default. V1 ignores it (any valid value, no effect). Runtime:
    // setSettings() at any time, no rebuild; a switch of the mode alone keeps the current estimate (a switch of motionVersion
    // drops it, see there). validate() takes Quality or Performance.
    V2Mode v2Mode = V2Mode::Quality;

    // ---- structural: init() or reconfigure() only (re-creates buffers; retire rule of resize()) ----
    // Motion estimation runs at 1/flowScale of the frame size: 1, 2 or 4. 2 = quality (default);
    // 4 = performance (1440p: ~40 % less total time at 2x, ~30 % at 4x; can miss thin/fine motion detail);
    // 1 = full res (1080p 2x: ~2.3x the time, not better on the test clips).
    uint32_t flowScale = 2;
};

struct InitInfo {
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkGetDeviceProcAddr gdpa = nullptr;   // all device functions are loaded through this
    uint32_t width = 0, height = 0;           // >= kMinSize each
    // The format of the views you pass, which also picks the Format (see there): B8G8R8A8_UNORM or R8G8B8A8_UNORM
    // (default; 8-bit UNORM, never _SRGB), A2B10G10R10 / A2R10G10B10 UNORM_PACK32, R16G16B16A16_SFLOAT, or for the
    // two-plane formats the format of the Y plane view (R8_UNORM / R16_UNORM). Anything else -> init false.
    VkFormat viewFormat = VK_FORMAT_B8G8R8A8_UNORM;
    // Required (instance-level, cannot come from gdpa): used to put the buffers in DEVICE_LOCAL memory.
    // init() returns false if null.
    PFN_vkGetPhysicalDeviceMemoryProperties getMemProps = nullptr;
    // Optional (instance-level): lets init() check that the device can use a non-Rgba8 format (sampled with linear
    // filtering and storage, for the views you pass and for the library's own clean copies) and refuse it with a reason
    // (Interpolator::initError()). Null = no check, the host vouches for it. Never consulted for Rgba8.
    PFN_vkGetPhysicalDeviceFormatProperties getFormatProps = nullptr;
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    // RgbaF16 only. Fixed at init (it is baked into the pipelines): a stream does not change its encoding.
    Transfer transfer = Transfer::Encoded;
    float linearWhiteNits = 80.0f;   // Transfer::Linear: the luminance of the value 1.0 (80 = scRGB). (0, 10000]
    // Nv12 / P010 only: how the planes encode colour. Defaults: BT.709, limited (video) range.
    YuvMatrix yuvMatrix = YuvMatrix::Bt709;
    bool yuvFullRange = false;
    Settings settings{};                      // init() returns false if Interpolator::settingsError(settings, format) (initError() says why)
};

class Interpolator {
public:
    // false -> the object is not usable (record() is a no-op returning false); skip frame generation
    // for this surface. Fails for sizes below kMinSize, unsupported formats, null getMemProps, or
    // any Vulkan creation/allocation failure.
    // init() on a live object and destroy()/resize() free resources that earlier records use: the
    // caller must guarantee every previous record has retired first (fence/timeline wait, or
    // vkQueueWaitIdle/vkDeviceWaitIdle; resolution changes are rare).
    bool init(const InitInfo&);
    // Why the last init() returned false (a string literal); null after a successful init or before any.
    const char* initError() const { return err_; }
    // The active format (Rgba8 if not initialised).
    Format format() const;
    void destroy();
    // Re-creates the size-dependent buffers. Same retire rule as above. On false the object stays
    // alive but holds no size (record() returns false) until a later resize() succeeds.
    // viewFormat may change only within the same Format (B8G8R8A8 <-> R8G8B8A8, A2B10G10R10 <-> A2R10G10B10): the pipelines
    // belong to the format chosen at init(); another Format needs a new init() (false here).
    bool resize(uint32_t w, uint32_t h, VkFormat viewFormat);

    // Settings (see struct Settings). validate(): true if every field is finite and in range.
    static bool validate(const Settings&);
    // validate() plus the rules that depend on the Format: null if the settings are valid for `format`, else why not (a string
    // literal). init(), setSettings() and reconfigure() refuse what it refuses (init() with this as initError()); a host whose
    // setSettings() returned false can ask it why. Today's only Format rule: MotionVersion::V2 needs Format::Rgba8.
    static const char* settingsError(const Settings&, Format format);
    // The box size (pixels) a frame of w x h gets under these settings: the explicit uiBoxWidth/uiBoxHeight, else the
    // square from uiBoxHeightFrac (or the legacy uiCrosshairArea). The same numbers the library protects; for a host
    // preview. Does not check validity.
    static void boxSize(const Settings&, uint32_t w, uint32_t h, uint32_t* boxW, uint32_t* boxH);
    // The box rectangle {x, y, w, h} in the frame: boxSize() centred on the frame centre plus uiBoxOffsetX/Y, clamped into the frame.
    // The only place UI can be DETECTED. Does not check validity.
    static void boxRect(const Settings&, uint32_t w, uint32_t h, int32_t rect[4]);
    // How many pixels past the box the active mode protects: 2 (Crosshair), 1 (CrosshairV2), 0 (Off, WholeScreen). Depends only on
    // Settings::uiProtection.
    static uint32_t boxMargin(const Settings&);
    // The working rectangle {x, y, w, h}: boxRect() grown by boxMargin() on every side and clamped to the frame. Everything
    // the box protects lies inside it (the mask, the inpaint, the paste-back); detection stays inside boxRect(). For a host
    // preview of the area that is actually protected. Does not check validity.
    static void workRect(const Settings&, uint32_t w, uint32_t h, int32_t rect[4]);
    // Runtime change: applies the runtime fields (the UI-protection ones too) to every record*() recorded after this call (each record
    // copies them into its own push constants, so in-flight and already-recorded work keeps its values;
    // no retire needed). Returns false and changes nothing if the object is not initialised, the settings
    // are invalid (settingsError() for the active format says why), or flowScale differs from the active one (that needs
    // reconfigure()). Switching Settings::motionVersion drops the current estimate (see there).
    bool setSettings(const Settings&);
    // Structural change (flowScale) plus the runtime fields. If flowScale changes it re-creates the
    // size-dependent buffers: same retire rule and failure behaviour as resize(), and the next synth needs a
    // new recordEstimate(). Returns false and changes nothing if the object is not initialised or the
    // settings are invalid (settingsError() for the active format).
    bool reconfigure(const Settings&);
    // The active settings (defaults if not initialised).
    Settings settings() const;

    // Call once per command buffer / submission, before its first record*() call: resets the
    // per-submission counts of both descriptor rings (see "Descriptor rings" below).
    void beginFrame();

    // Frame generation for one real pair (prev, curr) is split in two:
    //   recordEstimate(prev, curr)        passes 1-8 (motion estimate), once per real pair
    //   recordSynth(prev, curr, out, t)   the generated frame at time t, any number of times after one
    //                                     recordEstimate, each into its own out (N x: t = k/N, k = 1..N-1)
    //   record(prev, curr, out, t)        = recordEstimate + recordSynth (2x: t = 0.5)
    //
    // All views: 2D, single mip/layer, the same size as init/resize, format viewFormat (a Frame of the two-plane formats:
    // `view` = the Y plane view at full size, `chroma` = the CbCr plane view at half size, rounded up), i.e. an 8-bit
    // UNORM view for Rgba8 (an _SRGB view gives wrong luma), VK_IMAGE_LAYOUT_GENERAL on entry and exit. The "Formats"
    // section of INTEGRATION.md has the host contract per format (image flags, plane views, device features).
    // prev/curr: SAMPLED usage, already made readable by the caller. out: STORAGE usage, caller-owned (for the two-plane
    // formats both plane views of out are written: Y at full, CbCr at half resolution).
    //
    // recordEstimate ends with a COMPUTE -> COMPUTE barrier on the library's own buffers. recordSynth ends
    // with a COMPUTE_SHADER SHADER_WRITE -> COMPUTE_SHADER SHADER_READ memory barrier (valid on a
    // compute-only queue). Other consumers of out (fragment, transfer) need the caller's own barrier.
    // Synths of one pair share internal scratch (field at t, mismatch, counter) and are serialised by
    // barriers, in recording order; each one's out is complete when its own record ends.
    //
    // t in (0, 1). t == 0.5 is exactly the 2x midpoint path. Every fallback (per pixel, scene cut, more than
    // Settings::globalFallback untrusted, default 15 %) presents the nearer real frame: prev for t <= 0.5, curr for t > 0.5.
    //
    // Descriptor rings: recordEstimate takes the next of kSetRing estimate sets, recordSynth the next of
    // kSetRing synth sets; record() takes one of each. A call returns false and records nothing (out
    // untouched, so the caller must not substitute it) if the object is not initialised, t is outside (0, 1),
    // or its ring already gave out kSetRing sets since the last beginFrame() (asserts in debug builds).
    // The caller must also keep the estimates and the synths of all not-yet-retired submissions
    // <= kSetRing each, in total. With protection on, recordObserve takes 1 (Crosshair) or 2 (WholeScreen)
    // sets of a third ring of kUiSets, and so does every recordSynth (plus an implicit observe in a
    // recordEstimate): keep those per beginFrame() and per not-yet-retired submission <= kUiSets too.
    // recordSynth also returns false if no recordEstimate was recorded since init/resize, or if prev/curr
    // are not the views of the latest recordEstimate (it always uses the latest estimate).
    bool recordEstimate(VkCommandBuffer cb, Frame prev, Frame curr);
    bool recordSynth(VkCommandBuffer cb, Frame prev, Frame curr, Frame out, float t);
    bool record(VkCommandBuffer cb, Frame prev, Frame curr, Frame out, float t = 0.5f);

    // ---- Motion blur ----
    // recordSynthBlur(cb, prev, curr, out, t0, t1): out = the weighted average of Settings::blurSamples sub-frames of
    // the latest recordEstimate's pair, at t_k = t0 + (t1 - t0)(k + 0.5)/N, k = 0..N-1 (the midpoints of N equal slices: windows that tile, e.g. [0, .5] and
    // [.5, 1], share no sample, so no real frame is counted twice), with Settings::blurWeights.
    // 0 <= t0 <= t1 <= 1 (else false). A sub-frame is exactly what recordSynth(t) would have produced (the same
    // fallbacks per sub-frame: per pixel trust ramp, scene cut, more than globalFallback untrusted -> the nearer real
    // frame); t <= 0 / t >= 1 is the real frame prev / curr (with blur on and t0 < t1 no sample is ever exactly 0 or 1, so every sample is a synthesis). Each
    // sub-frame is rounded to 8 bit like a recordSynth output, the average is rounded once.
    //   N >= 2 and t0 < t1: the midpoint rule above. t0 == t1, or blur off (blurSamples < 2): one instant, t1 (NOT the
    //   window centre: that keeps the point case equal to recordSynth(t1)): the same as recordSynth(t1) (t1 = 0 / 1: a copy of
    //   prev / curr). With blur off and t0 < t1 the window collapses to its END t1: a host that centres its window
    //   on the output's t should call recordSynth(t) itself while blur is off.
    // Cost: the estimate is shared by every output of the pair; the per-sample work (field lookup, mismatch) is one
    // dispatch each for all samples, one more resolves and averages them (a sub-frame is never stored) and a last one
    // applies the whole-frame fallback if a sample needs it: 4 dispatches whatever N is (see INTEGRATION.md).
    // UI protection: the patch is applied AFTER the average, so a protected crosshair/HUD is the exact real pixels
    // while the world around it is blurred.
    // Rings: takes ONE synth set (not one per sample) and the same UI sets as recordSynth. Same preconditions and
    // refusals as recordSynth (no estimate, prev/curr not the estimate's, ring full), plus false if the scratch
    // cannot be allocated. Like recordSynth: out in GENERAL, written by the last pass, SHADER_WRITE -> SHADER_READ barrier.
    bool recordSynthBlur(VkCommandBuffer cb, Frame prev, Frame curr, Frame out, float t0, float t1);

    // ---- Static UI protection (Settings::uiProtection != Off) ----
    // The library keeps, per real frame it has "observed", its own clean copy (UI inpainted out) and the
    // mask of the still pixels. recordEstimate/recordSynth then read the clean copies, and every synth ends with
    // out = mix(out, real pixel, mask) over the masked area, reading the real pixels from the `curr` view you pass.
    // Your views are only read; they stay yours.
    //
    // recordObserve(cb, frame): call it ONCE for every new real frame, when `frame` already holds its final
    // pixels (same layout/usage rules as prev/curr), in presentation order, and for EVERY real frame, also those
    // you will not interpolate: the stillness counters compare consecutive real frames, so a skipped frame
    // costs the counters (a gap makes them compare the frames on either side of it: UI that did not change still
    // counts as still, only the exact "N consecutive frames" is lost). Call it after the frame's pixels are
    // written and before the recordEstimate that uses it as curr. Returns true (recording nothing) with
    // protection Off. Returns false and records nothing if the object is not initialised, a descriptor ring is
    // full (see below) or the first-use allocation failed (protection then stays off for that frame: do not
    // call recordEstimate on the frame expecting a protected result; calling it is still valid and unprotected).
    // `frame` is the cache key: when a view is observed again, its earlier entry is replaced, so a host that reuses
    // views (a ring of images) MUST observe each new content, or the library would pair new pixels with a stale mask.
    // At least kUiFrames = 3 observed frames are kept (LRU by observe order; Settings::uiHistory keeps more): the
    // newest pair AND the pair before it stay usable (a host that synthesises an older pair after the newest frame
    // arrived). recordEstimate(prev,
    // curr) / recordSynth need both views to still be among them, and recordSynth that nothing evicted them
    // since the estimate (observing 2 more frames would): else they return false and record nothing.
    //
    // Implicit observe: a host that never calls recordObserve still works. Then recordEstimate observes `prev`
    // if the library has not seen that view, and `curr` if it is not the most recently observed view (so pairs
    // (A,B), (B,A), (A,B)... or a ring of views work). Caveats: a view that receives new content under the same
    // handle is not noticed (stale mask: always call recordObserve then), a frame that is never in an estimate pair
    // is never seen by the counters, and a host that estimates an older pair after a newer frame would re-observe
    // it (so such hosts must observe explicitly). Once recordObserve has been called, estimates no longer
    // observe implicitly: a pair of which a view has not been observed (protection was just switched on, an
    // observe was skipped, the entry was evicted) is simply interpolated unprotected, as with Off.
    bool recordObserve(VkCommandBuffer cb, Frame frame);

    // ---- CrosshairV2 (Settings::uiProtection = CrosshairV2) ----
    // The crosshair box only (no WholeScreen variant), same API and host contract as Crosshair (explicit / implicit observe, recordTrack,
    // uiHistory, the box settings), a different algorithm. V1 stays unchanged next to it. Built from what V1 gets wrong on the user's CS2
    // recordings (white specks and patterns that move with the picture inside the protected area; measurements in INTEGRATION.md), and
    // from the "doubled crosshair" that a first version of V2 (and Off) still showed (point 4):
    //   1. it READS THE INPAINTED CLEAN COPY IN THE COLOUR WARP (~61 % of V1's artifact pixels): a pixel inpainted in prev's copy but real
    //      in curr (a mask that shrinks) is warped as 8-direction streaks. V2 has no inpaint and no clean copy: the estimate and every
    //      colour pass read the host's own frames (and, but for point 4, the generated frame would be bit-identical to Off everywhere but
    //      the pasted pixels).
    //   2. it PASTES curr'S PIXELS ONTO PIXELS THAT CHANGED IN THE PAIR (~37 %): the half-alpha ring and the hard grow. V2 pastes a pixel
    //      only if max channel |curr - prev| <= uiStillThreshold for the pair being synthesised (the pair gate), at full strength.
    //   3. its MASK FLICKERS (879-3378 px flip per frame with a 249 px box, ~20 with 27 px): a pixel that was masked for 8 frames drops
    //      out on one noisy frame. V2 enters a pixel after uiStillFrames still AND contrasty frames, but leaves it only after
    //      uiLeaveFrames consecutive changed frames, and re-enters at once (uiReentryFrames) when it shows its confirmed colour again.
    //   4. THE WARP COPIES THE CROSSHAIR ("doubled crosshair", also in Off and in the first V2). A generated pixel blends a tap of prev at
    //      p - 2t*h with a tap of curr at p + 2(1-t)*h (h = the half-vector); where the background pans, a tap that lands on the static
    //      crosshair carries it to crosshair +- motion*t: a second, full-strength crosshair next to the real one (the user's CS2 report:
    //      "major artifacting ... doubled crosshair", a copy ~16 px to the side). The paste only restores the real crosshair at its own
    //      place. V1 avoids it by warping inpainted copies. V2 REJECTS THE TAP: when a tap lands on a UI pixel of ITS OWN frame (the V2
    //      mask of that frame, bilinear footprint; in 4:2:0 the footprint of the chroma taps) and the other tap does not, both take the other
    //      frame's tap (shaders/ui_reject.glsl, in mismatch() and warpChroma(), also in the blur passes); both on UI or neither: unchanged.
    //      The library binds the pair's masks itself; a host does nothing. Outside the work rectangle plus the motion reach, the generated
    //      frame is still bit-identical to Off (without the fill). Cost: ~0.005 ms per pair at 1280x960 (RX 7900 XTX).
    // Per observed frame, per box pixel (full resolution, one state machine, 4 state words per pixel, no per-frame mask history; the
    // 1 px ring around the box keeps only the previous original and the pair-gate flag, so it can be pasted but never detected):
    //   still      = max channel |new - previous original| <= uiStillThreshold            (the V1 test, in the format's 8-bit-equivalent levels)
    //   contrasty  = max channel |I - median7x7(I)| >= uiContrastThreshold                (a crosshair has an edge, a flat background does not)
    //   out of the mask: counts consecutive still-and-contrasty frames, enters at uiStillFrames; or enters at once after
    //                    uiReentryFrames consecutive (contrasty) frames in the colour it was last confirmed in
    //   in the mask:     counts consecutive changed frames, leaves at uiLeaveFrames; while still it refreshes the confirmed colour
    //   mask = the pixels in the mask plus a hard 1 px grow (8-neighbourhood, binary; the grow may reach 1 px past the box: see
    //   boxMargin()). Each mask pixel is PASTED (out = real curr pixel)
    //   only if it is unchanged in this pair, else it is left to the interpolation (and still counts as UI for the optional fill).
    // The previous original is the frame observed (or tracked) just before, so the pair gate compares each frame with its predecessor in
    // observe order: call recordObserve / recordTrack for every real frame, as always. recordSynthBlur pastes after the average with the
    // same gate. recordTrack runs the very same state-machine pass and nothing else: the state after N tracked frames is the state after N
    // observed ones, hysteresis counters and confirmed colours included. Works for every Format: the thresholds are 8-bit-equivalent levels
    // (YUV: the contrast difference is converted to R'G'B' levels like the mismatch metric; stillness is on the Y/Cb/Cr codes as in V1).
    // Memory: no clean copy (V1: 4 B/px per kept frame); one mask per kept frame (1 B per work-rectangle pixel, workRect()) and 16 B per work-rectangle pixel of shared state.
    // Cost: observe = one pass over the box (a 7x7 median per candidate pixel, only where it can matter) + the mask pass; no inpaint, no
    // whole-frame copy. With uiEstimateFill: two more tiny passes in recordEstimate.
    // Limits: the crosshair must be opaque and static. A semi-transparent crosshair (its pixels change with the background behind it) and a
    // crosshair that changes shape every frame (a spread that follows movement or recoil) stay unprotected while it changes: its changed
    // pixels are never pasted, by design, and it re-enters uiReentryFrames / uiStillFrames frames after it settles. In 4:2:0 formats
    // (Nv12, P010) the chroma of an edge pixel is shared with the moving background, so a few edge pixels count as changed. No clean copy
    // also means nothing hides the crosshair from the optical-flow vote of the blocks around it unless uiEstimateFill is on (what that
    // leaves is flow noise near the crosshair, not a copy of it: the tap rejection removes the copy).
    // Host: add the enum value, set the settings you expose (the defaults are the tuned ones), keep calling recordObserve / recordTrack.

    // recordTrack(cb, frame): the cheap sibling of recordObserve for a real frame the host passes THROUGH without
    // generating (pacer reason GameTooFast / Off / CostGuard ...). It advances ONLY the stillness state (the box's
    // per-pixel counters and, in WholeScreen, the 1/2-res cell counters, with their "previous original" values) with
    // the very same shaders and thresholds an observe uses, and nothing else: no clean copy, no mask, no inpaint, no
    // entry in the kept-frame set (so nothing is evicted and nothing is allocated). The state after N tracked frames
    // is exactly the state after N observed ones.
    // Why: a host that is idle for a while (the game is fast enough, the guard paused generation) and then starts
    // generating again must not find a crosshair or HUD that "has not been still for uiStillFrames frames yet": the
    // user's words were "make sure that it still keeps track of like static UI elements, like the crosshair, even when
    // it's temporarily deactivated because the FPS is high enough. Then it should still keep track of the crosshair so
    // it doesn't start shifting around once it starts generating frames again." Without it a UI element that changed
    // during the idle stretch (or a library that was reset) needs uiStillFrames new real frames after the restart
    // before it is masked again, and until then it can shift.
    // Call it ONCE per new real frame instead of recordObserve while not generating (a frame is tracked OR observed,
    // never both: a second pass over the same pixels would count them twice). Then, when generation resumes, call
    // recordObserve on the next real frames as usual: the counters are warm, so the first pair that is synthesised
    // (both frames observed) already has the full mask. The frame that was only tracked has no clean copy and cannot be
    // a member of a pair; a host that wants to interpolate its very next frame against it observes that one instead.
    // Same contract as recordObserve otherwise: protection Off -> returns true and records nothing; false (and nothing
    // recorded) if not initialised, a descriptor ring is full (it takes the same 1 / 2 sets of the kUiSets ring) or the
    // first-use allocation failed. `frame` is only SAMPLED (the host's own image, GENERAL layout, final pixels); it
    // needs no storage usage and the library keeps no reference to it. A change of frame size / format (resize) or of
    // the box restarts the counters, as for an observe. Works for every Format. Cost, 2560x1440 (RX 7900 XTX, GPU): Crosshair ~0.004 ms, WholeScreen ~0.027 ms (about 3 % / 20 % of recordObserve).
    // Not for hosts that rely on the implicit observe (never call recordObserve): recordEstimate would count a tracked
    // frame a second time.
    bool recordTrack(VkCommandBuffer cb, Frame frame);

    // Frees the clean copies that Settings::uiHistory no longer keeps (they stay allocated, unused, when it is
    // lowered, so a later raise costs nothing). Retire rule of resize()/destroy(): every earlier record must have
    // retired (the freed images may be in a submission still running), so call it where the host already waits for the
    // GPU, e.g. after the depth has stayed lower for a while. No-op if nothing is unused. resize()/reconfigure()
    // with a new size/flowScale/destroy() free everything anyway.
    void releaseUnusedUi();

    // Frees the pipelines and buffers of the motion versions that Settings::motionVersion no longer selects (they stay
    // allocated, unused, when it is switched, so switching back costs nothing). Retire rule of resize()/destroy(): every earlier
    // record must have retired (the freed pipelines and buffers may be in a submission still running), so call it where the host
    // already waits for the GPU, e.g. after the version has stayed switched for a while. No-op if nothing is unused. Selecting a
    // freed version again creates its resources anew (setSettings, see Settings::motionVersion). destroy() frees everything anyway.
    void releaseUnusedMotion();

    // Profiling (test/benchmark only): timestamps after every pass group. recordEstimate (or record)
    // starts a new profile, recordSynth appends its passes to it. Returns false if the query pool cannot
    // be created (profiling then stays off).
    bool setProfiling(bool on);
    // Reads the last profile (waits for results). Fills names + GPU ticks per pass group
    // (multiply by VkPhysicalDeviceLimits::timestampPeriod for ns). Returns false if unavailable.
    // With protection on the profile also holds "ui_detect" (stillness + masks + the clean copy) and
    // "ui_inpaint" (both recorded by recordObserve; a profile opened by an observe is continued by the next
    // recordEstimate) and "ui_patch" (the paste-back, per recordSynth). recordSynthBlur adds "blur lookup", "blur mismatch", "blur resolve"
    // (the average of all samples) and "blur fixup" (the whole-frame fallback check, a no-op in the normal case), then "ui_patch".
    // MotionVersion::V2: its estimate ends after the levels, then "field clean" (once per estimate: re-pick, refinement, clean-up);
    // Quality adds "pixel field", "splat" and "coverage" (once per estimate); per synth "morph" / "blur morph" and "fallback" (its
    // fix-up).
    bool readTimings(std::vector<const char*>& names, std::vector<uint64_t>& ticks);

    // Debug / test only (bench/score), read-only: changes no output and no state. Records a copy of the latest recordSynth's
    // (or record's) whole-frame decision inputs into dst (TRANSFER_DST usage) at dstOffset: 2 uint32 = [untrusted pixels (trust
    // ramp past its midpoint), finest-level SAD sum x 256 (from the estimate)]. Preceded by a COMPUTE write -> TRANSFER read
    // barrier; the caller adds its own TRANSFER -> HOST barrier. The global fallback fires when untrusted > globalFallback x W x H,
    // the scene cut when sadSum / 256 / debugSadBlocks() > sceneCutSad. Both counts are computed whatever those settings are.
    // Not meaningful after recordSynthBlur. V2 (both modes): untrusted = the pixels of the 8x8 tiles whose mean mismatch is past the
    // midpoint, SAD sum = its own (the re-picked, refined block vectors' SADs). False if not initialised.
    bool recordDebugCounters(VkCommandBuffer cb, VkBuffer dst, VkDeviceSize dstOffset);
    uint32_t debugSadBlocks() const;   // the scene-cut mean's block count (finest pyramid level); 0 if not initialised
    // Debug / test only (fgtest --dump), read-only like recordDebugCounters: copies the latest record's / recordSynth's maps into dst at
    // dstOffset, laid out as debugMapLayout() says: the block field the per-pixel choice read (bw x bh vec2, finest level after
    // sub-pixel), the per-pixel half-vector field on the midpoint grid (fw x fh vec2) and the packed full-res mismatch e (one byte per
    // pixel, eStride words per row, 4 pixels per word, low byte first; e of the latest synth, before the 7x7 box). Preceded by a
    // COMPUTE write -> TRANSFER read barrier; the caller adds its own TRANSFER -> HOST barrier. False if not initialised.
    // V2 Performance (same layout): the block slot holds the cleaned block field its morph uses; the per-pixel field and e (it has
    // neither) are zero. V2 Quality (same layout): the cleaned block field, its per-pixel field, and at the start of the e part its
    // coverage, fw x fh uint32 (cP | cC << 11 | static-overlay mask << 22; flowScale >= 2, else zeros), see gpu/fgmaps.py.
    struct DebugMapLayout {
        uint32_t W, H, fw, fh, bw, bh, fscale, eStride;   // eStride in uint32 words
        VkDeviceSize blockOff, pixOff, eOff, size;          // byte offsets in dst (relative to dstOffset) and the total size
    };
    bool debugMapLayout(DebugMapLayout& l) const;
    bool recordDebugMaps(VkCommandBuffer cb, VkBuffer dst, VkDeviceSize dstOffset);

    static constexpr uint32_t kSetRing = 32;
    static constexpr uint32_t kUiSets = 64;    // descriptor sets of the UI-protection passes (see above)
    static constexpr uint32_t kBlurMaxSamples = 16;   // Settings::blurSamples upper bound
    static constexpr uint32_t kUiFrames = 3;   // observed frames kept by default (clean copy + mask each)
    static constexpr uint32_t kUiFramesMax = 65;   // Settings::uiHistory upper bound (= pacing.h's kHistoryMax)
    static constexpr uint32_t kMinSize = 16;
    struct Impl;
private:
    Impl* m = nullptr;
    const char* err_ = nullptr;
};

}  // namespace framegen
