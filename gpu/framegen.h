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
    // The crosshair box: always protected at FULL resolution, in both modes (a crosshair is the most noticeable
    // thing if it breaks). Size it so the whole reticle fits with room for its spread and outline: about 2-3x the
    // crosshair's largest extent. Too small cuts the crosshair's ends off the protection, too big only costs a few
    // more pixels of work (the box is cheap; the default is 333 px at 2560x1440).
    //   uiBoxWidth/uiBoxHeight: the box size in pixels, [0, 16384]. 0 = take that side from uiCrosshairArea.
    //   An explicit size wins over the fraction; it is clamped to [4, frame size].
    uint32_t uiBoxWidth = 0, uiBoxHeight = 0;
    //   uiCrosshairArea: (0, 1]. The area-derived box is a square of side sqrt(uiCrosshairArea * w * h), clamped to
    //   [32, min(w, h)]: 0.03 = 333 px at 2560x1440.
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
    Settings settings{};                      // init() returns false if !Interpolator::validate(settings)
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
    // Runtime change: applies the runtime fields (the UI-protection ones too) to every record*() recorded after this call (each record
    // copies them into its own push constants, so in-flight and already-recorded work keeps its values;
    // no retire needed). Returns false and changes nothing if the object is not initialised, the settings
    // are invalid, or flowScale differs from the active one (that needs reconfigure()).
    bool setSettings(const Settings&);
    // Structural change (flowScale) plus the runtime fields. If flowScale changes it re-creates the
    // size-dependent buffers: same retire rule and failure behaviour as resize(), and the next synth needs a
    // new recordEstimate(). Returns false and changes nothing if the object is not initialised or the
    // settings are invalid.
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
    bool readTimings(std::vector<const char*>& names, std::vector<uint64_t>& ticks);

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
