# frame-gen-ritz

frame-gen-ritz is an image-only frame interpolation library, similar in kind to Lossless Scaling's LSFG or AMD AFMF. It sees only finished frames, with no motion vectors or depth from the game. From two real frames it generates the frames in between, for 2x, 3x or 4x the base frame rate. It is a set of Vulkan compute shaders behind a small C++ API, plus the Python/NumPy reference implementation it was ported from.

It was written for gamescope-ritz, a gamescope fork. There it is meant to run as a compute pass on the game's layer-0 image, before scaling and before the overlay layers (Steam overlay, cursor, performance HUD) are composited on top, so those are never interpolated. This repository contains only the library. The pacing decisions (which frame to show when, when to turn generation off) live here too, in `gpu/pacing.h`. The gamescope side is only the glue that feeds the pacer and presents what it decides. `INTEGRATION.md` is the spec for that integration.

frame-gen-ritz is original code. It is not derived from LSFG, lsfg-vk or FidelityFX source. `FRAMEGEN_RESEARCH.md` holds background research on how such tools work.

## Design goal

"They can be subtle, but worth the compute. It should only make the motion be perceived more smoothly."

The output only has to make motion look smoother. Wherever the motion estimate is not trustworthy, frame-gen-ritz shows the nearer real frame for that pixel (or for the whole frame) instead of guessing. A region that stays still for one frame is not noticeable. A smear or a ghost is.

## How it works

Motion is estimated on luma at half resolution (`flowScale` 2) over an image pyramid. The vectors are *bilateral*: for each 8x8 block of the in-between frame, the search finds a half-vector `h` that minimises the difference between `prev[q - h]` and `curr[q + h]`. The vectors therefore already sit on the output frame's grid, so there is no splatting and there are no holes. Both frames are then warped to time `t`. Where the two warped frames disagree, the output falls back to the nearer real frame.

Per real frame pair (`recordEstimate`, passes 1-8):

1. Luma at flow resolution (`luma.comp`), then a 2x2-average pyramid (`down.comp`).
2. Per level, coarse to fine: seed candidates per block (parents, zero vector, global median vector of the previous level; `medhist.comp`), then an integer search around the best seed, ±6 px at the coarsest level and ±2 px below it, with a penalty per step (`search.comp`).
3. 3x3 median of the block vectors (`median3.comp`).
4. Finest level: sub-pixel refinement at 1/2 and then 1/4 px (`subpix.comp`). A scene cut is detected when the mean block SAD is too high.
5. Per-pixel vector choice from the smooth field, the pixel's own and neighbouring blocks, and the zero vector. A small bonus for the zero vector keeps static HUD elements in place (`pixfield.comp`).

Per generated frame at time `t` (`recordSynth`, cheap):

6. For `t != 0.5`, look up the field at time `t` (`lookup.comp`). Warp both frames and compute the per-pixel mismatch (`error.comp`).
7. Turn the 7x7-averaged mismatch into a trust ramp and blend the warped frames, `(1-t)·prev + t·curr`, falling back to the nearer real frame where trust is low (`resolve.comp`).
8. Whole-frame fallback to the nearer real frame on a scene cut or when more than 15% of pixels are untrusted (`fallback.comp`).

"Nearer real frame" means `prev` for `t <= 0.5` and `curr` otherwise. The docstring of `proto/interp.py` has the full pass description.

## Repository layout

| Path | Contents |
|---|---|
| `proto/interp.py` | Python/NumPy/OpenCV reference implementation of the algorithm (`estimate`, `synth`, `interpolate`, `interpolate_multi`) |
| `proto/interp_v0.py` | First version, kept for comparison |
| `proto/harness.py` | Scores interpolation against the real in-between frames of 120 fps footage |
| `proto/render.py` | Renders review videos (generated frames only, or the full real + generated stream) |
| `proto/NOTES.md` | Experiment log: what was tried, measured and kept |
| `gpu/framegen.h`, `gpu/framegen.cpp` | The Vulkan library (`framegen::Interpolator`) |
| `gpu/pacing.h` | Host-agnostic pacing: which frame to show when, pass-through, cost guard, motion-blur window (`framegen::pacing`, header-only, no Vulkan) |
| `gpu/pacingtest.cpp`, `gpu/minitest.h` | CPU-only tests of the pacing header and the lag-spike buffer (`meson test`), with a tiny Catch2 stand-in |
| `gpu/shaders/*.comp` | GLSL compute passes (`ui_*.comp`: static-UI protection, `blur_*.comp`: motion blur) |
| `gpu/fgtest.cpp` | Test and benchmark tool for raw RGBA frame files |
| `gpu/uitest.cpp` | Validation and benchmark of static-UI protection on a synthetic scene |
| `gpu/blurtest.cpp`, `gpu/blendbench.comp` | Validation and benchmark of the motion blur on a synthetic scene (the shader is the naive blend the benchmark compares against) |
| `gpu/compare.py` | GPU vs Python reference checks, and the benchmark runs |
| `INTEGRATION.md` | Integration spec for gamescope-ritz, full API contract, settings, costs |
| `FRAMEGEN_RESEARCH.md` | Background research on image-based frame generation |

## Results

GPU cost per real frame pair, RX 7900 XTX (RADV), default settings, timestamp medians from `compare.py bench` (one estimate plus N-1 synths, from `INTEGRATION.md`):

| | 2x | 3x | 4x |
|---|---|---|---|
| 1080p | 0.35 ms | 0.42 ms | 0.49 ms |
| 1440p | 0.50 ms | 0.62 ms | 0.71 ms |

With `flowScale = 4`, 1440p 2x drops to about 0.28 ms (roughly 40% less). Each extra generated frame costs about 0.06 ms at 1080p and 0.1 ms at 1440p.

On 10 real frame pairs, the GPU output scores within ±0.01 dB of the Python reference when both are compared with the real middle frame. RGBA and BGRA output are byte-identical.

Quality against the real in-between frames of the test footage (mean PSNR in dB, `harness.py` at full scale, from `proto/NOTES.md`). "dup" shows the nearer real frame, which is what you see without frame generation:

| clip | 2x dup / interp | 3x dup / interp | 4x dup / interp |
|---|---|---|---|
| battlefield | 24.36 / 31.33 | 23.97 / 29.18 | 23.25 / 27.49 |
| obs1 | 25.88 / 30.63 | 26.23 / 29.86 | 25.13 / 28.28 |
| obs2 | 24.38 / 29.86 | 24.19 / 28.37 | 22.99 / 26.02 |

The game in the OBS clips ran at about 90 fps, so their "real" frame timing is uneven. That lowers every method's score on those clips.

## Building

Requirements: Meson, Ninja, the Vulkan headers and loader, and `glslangValidator`.

```sh
cd gpu
meson setup build
ninja -C build
```

This builds the static library `framegen` (the shaders are embedded as SPIR-V headers) and the `fgtest` tool. The library needs only the Vulkan headers, because it loads every device function through `vkGetDeviceProcAddr`.

## Static UI protection

A crosshair, a minimap frame or a health bar is static while the picture behind it moves. In a block-based motion search such a pixel shares its block with moving background, so either the crosshair is dragged along or the background around it freezes. `hudBonus` (zero-vector bias) only trades one for the other. Static UI protection removes the problem at the source, per pixel, and belongs in the library so that any host gets it by flipping one setting:

- **Detect** (every real frame, `recordObserve`): a pixel is *still* when no channel moved more than `uiStillThreshold` (3.5 levels) since the previous real frame. After `uiStillFrames` (8) consecutive still frames it is masked; the mask is dilated by 2 px (1 px fully, the second px at half weight) to catch anti-aliased rims. The comparison is always against the previous *original*, never against the inpainted copy.
- **Inpaint**: the library keeps its own "clean" copy of each real frame. Every masked pixel in it is replaced by the 1/d²-weighted average of the first unmasked pixel along each of 8 directions (reach `uiInpaintReach`, 12 px). The motion estimate and the warps run on the clean copies and never see the UI.
- **Patch back**: after each generated frame, `out = mix(out, real pixel, mask)`. Masked pixels therefore show the exact pixels of the real frame.
- There is deliberately no "is this a crosshair" test. Static for N frames *is* the detector. If the camera is still, static background is masked too, which is harmless: it is patched back with the identical real pixels, and any pixel that changes leaves the mask on the very next pair.

Two modes (`Settings::uiProtection`, a runtime setting, `Off` by default and byte-identical to a library without the feature):

| Mode | What is protected | How |
|---|---|---|
| `Crosshair` | the **box** (default: a centred square of 3% of the frame area, 333 px at 1440p) | full resolution, every pixel |
| `WholeScreen` | the box, plus every solid static HUD element outside it | the box as above; outside it detection at half resolution (2x2 cell means, one counter per cell), the mask is upsampled and dilated at full resolution, and work is skipped for every 32x32 tile that cannot contain a still pixel (GPU-side tile flags); inpaint and patch touch only masked pixels in flagged tiles |

The box is always full resolution in both modes, because the crosshair is the most noticeable thing if it breaks. **Make sure it is big enough for your reticle:** `uiBoxWidth`/`uiBoxHeight` set it in pixels (0 = derive that side from `uiCrosshairArea`, an explicit size wins) and `uiBoxOffsetX`/`uiBoxOffsetY` move its centre from the frame centre (for a game whose reticle is not dead centre). Rule of thumb: 2-3x the crosshair's largest extent, so its spread (shooting, movement) and outline fit. Too small leaves the ends of the crosshair unprotected; too big only costs a few more pixels of work. The box is clamped into the frame.

`WholeScreen` bounds its worst case: if more than `uiMaxStillFraction` (75%) of the cells outside the box are still, that frame's whole-screen passes exit on the GPU (a nearly static picture needs no generated frame to differ from the real one). Costs, memory and measurements are in `INTEGRATION.md`. Limits: only *opaque, solid, static* UI is protected. Semi-transparent UI over moving background, UI whose shape changes every frame (a scrolling kill feed, an animated number) and UI that moves (a floating damage number) are not; they behave as before. The cell-mean detection also treats a very slowly moving gradient (under `uiStillThreshold` per frame) as still, which only means it is shown from the real frame.

Host sketch (details in `gpu/framegen.h`):

```cpp
Settings s; s.uiProtection = UiProtection::WholeScreen;
s.uiBoxWidth = 200; s.uiBoxHeight = 200;        // optional: box sized for this game's reticle
s.uiBoxOffsetX = 0; s.uiBoxOffsetY = -30;       // optional: reticle sits 30 px above the centre
fg.setSettings(s);                              // or Settings::uiProtection in InitInfo

// per NEW real frame (also frames you will not interpolate), after its pixels are written:
fg.recordObserve(cb, frameView);
// per pair, as before:
fg.recordEstimate(cb, prevView, currView);
fg.recordSynth(cb, prevView, currView, outView, t);   // curr must still hold the real frame: it is the patch source
```

## Motion blur

Interpolation-based motion blur, the technique of video tools such as Blur and smoothie-rs: interpolate to a high rate and average groups of sub-frames. In the library it is one primitive, `recordSynthBlur(cb, prev, curr, out, t0, t1)`: **one output image = the weighted average of `Settings::blurSamples` (2-16, any count) synthesised sub-frames at evenly spaced t in [t0, t1] of the current real pair**, with `Even` or `Gaussian` weights (`Settings::blurWeights`). Each sub-frame is exactly what `recordSynth(t)` produces, including every fallback; `t <= 0` / `t >= 1` is the real frame. Off (`blurSamples` 0 or 1) leaves every output byte-identical.

- **Blur only**, no frame generation: show each real frame blurred over the end of the interval that led to it, `recordSynthBlur(prev, curr, out, 1 - s, 1)`. The newest sample is the real frame, so it needs no future frame and adds no delay.
- **With frame generation**, M outputs per real pair at `t = m / M`: output m is blurred over the window that *ends* at its time, `recordSynthBlur(prev, curr, out, (m - s) / M, m / M)`, with the shutter `s` (<= 1) in output intervals. Also no delay; `s = 1` is a 360 degree shutter at the output rate.
- **Per-t work is not shared** (lookup, mismatch, trust and the fallbacks depend on t); the estimate is. The library runs all samples of one output in 4 dispatches and never stores a sub-frame, so it takes one descriptor set per output: 8 outputs x 8 samples in one pair fit the 32-set ring. A sample costs about as much as one generated frame (0.03 ms at 1280x960, 0.05 at 1080p, 0.1 at 1440p).
- **Static UI protection applies after the average**: with `uiProtection` on, a crosshair or HUD stays pixel-exact while the world around it blurs.
- The 60 fps to 280 Hz case with 4 samples per output costs about 0.76 ms per real frame in total at 1280x960 (46 ms of GPU time per second, 4.6 %) and 2.1 ms at 1440p (12.7 %). Tables, shutter conventions and the pass breakdown are in `INTEGRATION.md`.
- Limits: blur spreads interpolation errors instead of hiding them, and it reduces clarity by design, so competitive players will often want it off.

## Testing and benchmarking

Footage is not in the repository. The tools expect 120 fps recordings: `compare.py` reads `battlefield.mp4` and `obs2.mp4` from `footage/` at the repo root (override with `FG_FOOTAGE`). The test clips were a CS2 recording made with OBS and a Battlefield clip. Work files go to `FG_WORK` (default `/tmp/fgwork`). The `fgtest` binary is `gpu/build/fgtest` unless `FGTEST` is set.

```sh
python3 gpu/compare.py check [--bgra] [--split]   # 2x: GPU vs Python vs real middle frame, ~10 pairs + fallback cases
python3 gpu/compare.py multi [--bgra]             # 3x/4x: one estimate + N-1 synths vs interp.synth, plus fallback cases
python3 gpu/compare.py bench [N]                  # fgtest --bench N --mult 2/3/4 at 1080p and 1440p (default N = 100)
python3 gpu/compare.py bench 100 --set flowScale=4,subpixel=0   # any mode: --set k=v,... applies Settings to both sides
```

`fgtest` works on tightly packed RGBA8 files of W*H*4 bytes:

```
fgtest [--bgra] [--split] [--t T1,T2,..] [--bench N [--mult M]] [--set k=v,..] [--then k=v,..] W H prev.rgba curr.rgba out.rgba [prev curr out ...]
```

- `--bgra`: use `B8G8R8A8_UNORM` images (the file data stays RGBA).
- `--split`: `recordEstimate` + `recordSynth` instead of `record`.
- `--t`: generated times (default 0.5). With several values, the k-th output goes to `out.rgba.<k>`.
- `--bench N --mult M`: benchmark N iterations, one estimate + M-1 synths per pair.
- `--set`: `framegen::Settings` fields by name. `--then`: a `setSettings` change applied between two records (runtime-change test).

`blurtest` (needs no footage) validates and benchmarks the motion blur on a synthetic scene (panning textured background, a moving object, a static crosshair):

```sh
gpu/build/blurtest check [--bgra] [W H]  # default 1280x720: blur == recordSynth for one instant, == the CPU average of the
                                         # recordSynth sub-frames (N = 2..8), smear of the object's edge vs the ideal model,
                                         # crosshair pixel-exact, descriptor rings, validation; prints FNV hashes of the blur-off output
gpu/build/blurtest bench [W H]           # GPU cost: one synth, N x recordSynth (+ blend), recordSynthBlur for N = 2..8, M outputs per pair
```

`uitest` (needs no footage) validates and benchmarks static-UI protection on a synthetic sequence:

```sh
gpu/build/uitest check [--bgra] [W H]    # default 1280x720: asserts the exact crosshair / HUD pixels, PSNR, validation
gpu/build/uitest check --hash            # also prints a hash per generated frame of the Off run (compare against the
                                         # library before the feature to prove byte-identical output)
gpu/build/uitest deep [W H]              # Settings::uiHistory: a synth from a pair 10 frames behind the newest is still protected (default 640x360)
gpu/build/uitest bench [W H]             # per-pass GPU times: moving / still / 70% still scenes, all modes
```

If Lossless Scaling's implicit Vulkan layer is installed, disable implicit layers when testing and benchmarking: `VK_LOADER_LAYERS_DISABLE='~implicit~'`.

Python reference (needs NumPy and OpenCV; `render.py` also needs ffmpeg):

```sh
python3 proto/interp.py                                   # self-check: synthetic pan (2x, 3x, 4x) and static HUD
python3 proto/harness.py footage/battlefield.mp4 --mult 3 # score against real in-between frames
python3 proto/render.py footage/obs2.mp4 --start 52 --mult 3 -o gen.mp4 --full full.mp4
```

## Library usage

The library uses plain Vulkan with no gamescope types. It records into a command buffer the caller provides. It never submits work and never touches queues. Condensed calling sequence (see `gpu/framegen.h` and `INTEGRATION.md` for the full contract):

```cpp
#include "framegen.h"

framegen::InitInfo info;
info.phys = physicalDevice;
info.device = device;
info.gdpa = vkGetDeviceProcAddr;
info.width = w; info.height = h;                       // >= kMinSize (16)
info.viewFormat = VK_FORMAT_B8G8R8A8_UNORM;            // or R8G8B8A8_UNORM; never _SRGB, never 10-bit
info.getMemProps = vkGetPhysicalDeviceMemoryProperties; // required
framegen::Interpolator fg;
if (!fg.init(info)) { /* no frame generation for this surface */ }

// per command buffer:
fg.beginFrame();

// 2x:
bool ok = fg.record(cb, prevView, currView, outView);  // t = 0.5

// N x (3x, 4x): estimate once per real pair, then one synth per generated frame
fg.recordEstimate(cb, prevView, currView);
for (int k = 1; k < N; ++k)
    slotOk[k - 1] = fg.recordSynth(cb, prevView, currView, outView[k - 1], float(k) / N);
```

- All views are 2D, single mip/layer, 8-bit UNORM in `viewFormat`, and in `VK_IMAGE_LAYOUT_GENERAL` on entry and exit. `prev`/`curr` need SAMPLED usage and must already be readable. `out` needs STORAGE usage and is owned by the caller.
- Every `record*()` returns `false` and records nothing if the object is not initialised, `t` is outside (0, 1), or more than `kSetRing` (32) estimates or synths were recorded since `beginFrame()`. `recordSynth()` also fails without a matching `recordEstimate()`. Present an output only if its call returned `true`.
- Synths end with a compute-to-compute barrier only. Other consumers of `out` (fragment, transfer) need the caller's own barrier.
- Not thread-safe. All records of one `Interpolator` must go to one queue, in recording order.
- `resize()`, `destroy()`, re-`init()` and a `flowScale` change through `reconfigure()` free resources that earlier records use. Wait until those records have retired first.

## Settings

`framegen::Settings` (in `gpu/framegen.h`) is passed through `InitInfo::settings`. The defaults are the tuned values that were reviewed visually, and with them the output is byte-identical to the library before settings existed. Runtime fields can be changed at any time with `setSettings()`. `flowScale` is structural and changes only through `init()` or `reconfigure()`. `Interpolator::validate()` checks the ranges.

| Field | Default | Effect |
|---|---|---|
| `trustLow`, `trustHigh` | 12, 40 | Mismatch (8-bit levels) where the per-pixel fallback to the nearer real frame starts and where it is complete. Lower = fewer ghosts, more held pixels. |
| `globalFallback` | 0.15 | Share of untrusted pixels above which the whole frame is the nearer real frame. 1 = off. |
| `sceneCutSad` | 30 | Mean block SAD above which a pair counts as a scene cut. 255 = off. |
| `hudBonus` | 1.0 | Zero-vector bonus that keeps static HUD elements in place. Too high can freeze slow motion. 0 = off. |
| `smoothBonus` | 0.5 | Bonus for the smooth vector field. Higher = calmer but softer motion edges. |
| `searchPenalty` | 0.5 | Search cost per pixel step. Higher = steadier on flat or repetitive surfaces, lower = follows small fast detail. |
| `subpixel` | true | 1/2 + 1/4 px refinement. Off saves about 0.04-0.06 ms per pair, slightly softer on slow pans. |
| `uiProtection` | Off | `Off`, `Crosshair` or `WholeScreen`: static-UI protection (see above). Runtime. |
| `uiBoxWidth`, `uiBoxHeight` | 0, 0 | Size of the full-resolution box in pixels; 0 = take that side from `uiCrosshairArea`. Clamped to [4, frame]. Runtime. |
| `uiCrosshairArea` | 0.03 | Box = square of side sqrt(area * w * h), clamped to [32, min(w, h)]. (0, 1]. |
| `uiBoxOffsetX`, `uiBoxOffsetY` | 0, 0 | Box centre offset from the frame centre, pixels (+x right, +y down). The box is clamped into the frame. |
| `uiStillFrames`, `uiStillThreshold` | 8, 3.5 | A pixel is protected after this many consecutive real frames in which no channel moved more than this many levels. |
| `uiInpaintReach` | 12 | Walk length (1-16 px) of the inpaint. |
| `uiHistory` | 3 | Observed frames that keep their UI-protection clean copy (3 to 65). Raise it to `Pacer::HistoryDepth()` when the lag-spike buffer plays older pairs. 4 B/px per kept frame. Runtime. |
| `blurSamples` | 0 | Motion blur: sub-frames averaged by `recordSynthBlur()`. 0 or 1 = off, 2-16 on (see above). Runtime. |
| `blurWeights` | Gaussian | `Even` (1/N each) or `Gaussian` (ends of the window at 2 sigma). Runtime. |
| `uiMaxStillFraction` | 0.75 | WholeScreen: above this still share (of cells outside the box) the whole-screen passes are skipped for that frame. 1 = never. |
| `flowScale` | 2 | Motion estimation at 1/flowScale resolution: 1, 2 or 4. 4 is much cheaper but can miss thin, fine motion. 1 is much slower and was not better on the test clips. (structural) |

`INTEGRATION.md` lists the accepted ranges, the measured cost of each setting, and a suggested user-facing subset.

## Pacing

`gpu/pacing.h` (`framegen::pacing`, header-only, no Vulkan, no OS calls, times passed in as nanoseconds) decides **which frame to show at which moment and whether to generate at all**, so a host only builds its GUI and platform glue. The host feeds it real-frame arrivals (`OnArrival`) and one paint per present tick (`OnPaint`), and applies each `Decision`: show the real frame, or synthesise at `t` between `prevId` and `currId` (`Interpolator::recordSynth`), reset the renderer, skip the tick, or paint again. gamescope-ritz is the reference host.

```cpp
pacing::Pacer pacer;
pacer.OnArrival(frameId, fenceSignalNs, windowId, {w, h, fmt});            // a real frame is usable
pacing::Pacer::Inputs in;  in.now = nowNs;  in.presentNs = predictedPresentNs;
in.frameGen = true;  in.mode = pacing::Mode::Fixed;  in.multiplier = 3;  in.refreshHz = 144.0;
in.blur = true;  in.blurSamples = 4;  in.extraDelayNs = 0;                  // ... see INTEGRATION.md
const auto d = pacer.OnPaint(in, newestFrameId, /*bPresentTick*/ true);     // per present tick
if (d.prevId == 0) { /* show the real frame d.showId */ }
else               { /* recordSynthBlur(cb, prev(d.prevId), curr(d.currId), out, d.t0, d.t1) */ }
```

- **Per-tick fractional pacing**: every output frame is generated for its own content time `tau = V - D` between the two newest real frames. No multiplier is chosen or held; a fixed N sets only the output interval and saturates at the refresh, and a game that speeds up or slows down is followed on the very next pair.
- **Modes**: fixed N (2x-8x) or a target fps; **Low latency** (the pair is the newest, the delay is the smoothed interval minus the output interval plus the measured jitter) or **Smoothness** (about one game frame queued, never hitches on a late frame).
- **Pause at refresh rate**: on, the output is capped at the refresh and generation stops once the game reaches it; off, the cap is gone and `ExtraTimer()` asks the host to paint on its own timer. `Inputs::canExceedRefresh` is a host capability: where a present between ticks cannot be shown, off behaves as on.
- **GPU limit**: generation may use at most 25% of a game interval; otherwise the output rate is lowered, then it passes through, with a probe every 10 s. A switch turns it off.
- **Pass-through** (real frame, renderer inert) only when the game reaches the output rate (a ratio band, 1.02 / 1.10), the renderer is unavailable, or the cost guard trips. **Resets** on a new stream, a format change, a gap over 100 ms, or `Discontinuity()`.
- **Three features, one timeline**: frame generation (which instants get an output), motion blur (what each output shows over time: a trailing window `[t0, t1]` ending at the output's instant, `BlurWindow`) and the lag-spike buffer (how far behind real time we show: `Inputs::lagBuffer`, see below) are separate switches that work alone and together. Every `Decision` carries `{prevId, currId, t0, t1, showId}` and the host makes **one** renderer call, `recordSynthBlur(prev, curr, out, t0, t1)`. With frame generation off, the planner runs real-rate: one output per real frame at its own time, gaps the buffer covers filled with synths.

### Lag-spike buffer

Some games, especially on Linux, hitch: a frame arrives one, two or three intervals late and the picture freezes. The buffer (`Pacer::Inputs::lagBuffer`, off by default) keeps a small adaptive extra delay so that the next hitch is **filled with generated frames instead of frozen**. In the user's words: *"some games on Linux are sometimes unplayable because of lag spikes ... in case one frame is missing, it can just fully generate between the lag spike ... mainly usable for controller games since it will add ... delay. But it could just turn a game from unplayable into a playable one."* It is independent of the multiplier (frame generation off plans real-rate: real frames, gaps filled) and works together with frame generation and motion blur.

- **Detection**: an interval above 1.75x the smoothed interval is a spike; its size is `interval - smoothed` ("missing frames x interval"). *"The starters ... should be like 1, 2, maybe 3 missing frames, so it shouldn't go that high."*
- **Judgement**: a spike larger than `maxBufferMs` is an outlier and never sizes the buffer (*"single, like huge lag spikes, for example, when you're compiling and drops down to like 10 FPS, it shouldn't build a 100 millisecond buffer"*); it is only counted.
- **Recency**: target = max over spikes of `size x w(age)`, `w` falling smoothly (cosine) from 1 to 0.3 over `lookbackSec`, 0 beyond; +10 % +1 ms margin; floored at half the long-run mean spike size while spikes recur. The newest big spike dominates, a game that hitches every 2-3 minutes keeps a steady buffer, and one spike fades away after the window (*"the longer the game has been running, the better it can adjust ... only like the newest biggest lag spike in the last 5 minutes actually affects ... and everything else is just smoothed out"*).
- **Settings**: `enabled` (false), `lookbackSec` 300 (1-600), `maxBufferMs` 50 (0-250; *"something insane like 250 milliseconds"* is allowed, the default is 50), `testMode` Off / ForceMin / ForceMax (pins the target at 0 or at the max, through the same ramp, so the user can feel each end).
- **Smooth**: the buffer ramps at most 3 % of real time (content speed 0.97x-1.03x), never jumps, and shrinks only after the target has stayed low for 2 s.
- **Frames follow the buffer**: the real frames the host keeps (`HistoryDepth()`, `Report::historyFrames`) and, with UI protection, the clean copies (`Settings::uiHistory`) follow the **current** buffer: `ceil((delay) / interval) + 2`, 3 with no buffer, at most 65. Memory is **per buffered frame**: 4 B/px each, 4.9 MB at 1280x960 and 14.7 MB at 1440p (250 ms at 60 fps = 17 frames: 84 / 251 MB, and the same again for UI-protection clean copies).
- **Costs**: the buffer is real latency (at most `maxBufferMs`), so it suits controller games; the host's audio is not delayed, so above ~50-80 ms lip-sync drifts.
- **Report**: `bufferDelayMs`, `bufferTargetMs`, `lastSpikeMs`, `spikesInWindow`, `outliersIgnored`, `historyFrames`, for a line like "buffer 32 ms (last spike 28 ms)".

`INTEGRATION.md` has the full host contract, every field, a minimal loop and the reasoning behind each rule. `meson test -C <build> pacing` runs the CPU-only tests (`gpu/pacingtest.cpp`).

## Limitations

- 8-bit UNORM RGBA/BGRA only. There is no HDR or 10-bit support, so the caller must skip frame generation or convert first.
- HUD drawn into the game image can still show artifacts, especially translucent HUD elements and scrolling or changing text. Solid, static HUD is handled by the optional static-UI protection (`uiProtection`, above); semi-transparent or shape-changing UI is not protected by it either. Overlays composited later by gamescope are not affected.
- Repetitive structures (stairs, ladders, fences) under fast motion can lock onto the wrong period. Both warps then agree, so the mismatch test cannot catch it.
- Very fast flicks below the whole-frame fallback threshold can show seams between warped and held regions.
- Interpolation adds latency, because a generated frame needs the next real frame.
- At a low base frame rate (3x, 4x) motion between real frames is larger, so fallbacks are more frequent.
- Presentation is the caller's job. `gpu/pacing.h` decides the schedule (which frame at which time), the caller owns the clock, the display and the actual present.

## License

MIT, see `LICENSE`.
