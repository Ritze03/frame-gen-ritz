# Frame generation: integration into gamescope-ritz

Root: `/home/mo/Schinken/Programming/GitHubProjects/gamescope-ritz`. `R` = `src/rendervulkan.cpp`, `H` = `src/rendervulkan.hpp`. Line numbers checked 2026-10-03.

## Tap point
- `vulkan_composite(const FrameInfo_t *pCallerFrameInfo, ..., std::unique_ptr<CVulkanCmdBuffer> pInCommandBuffer)`: R:5369. Command buffer: R:5480 (`pInCommandBuffer` or `g_device.commandBuffer()`).
- Layer 0 is substituted only in a private copy of the frame info, through the lambda `SubstituteLayer0` (R:5404-5412). Callers never see the change, so a screenshot re-composite repeats the pass (rationale R:5376-5402).
- Pre-pass gate (R:5551-5562): layer0.tex present, `!bBaseLayerEffectsApplied`, layer0.tex != `g_output.effectsOutput`, `g_nativeEffects.AnyEnabled()`, SDR (not HDR, not PASSTHRU) and `!layer0.isYcbcr()`.
- The output is allocated by `update_effects_image(w,h,drmFormat)` (R:3733-3757). The final dispatch is at R:5926-5932, and the substitution `SubstituteLayer0(g_output.effectsOutput)` is at R:5958. FG goes right before or after that block, still inside the SDR/non-YCbCr branch. It substitutes its own output as layer 0, and scaling (FSR/NIS/blit), colour management and the overlay layers then composite on top. Order to choose: effects then FG (FG reads graded frames, so the previous graded output must be kept as well), or FG on raw layer 0 then effects on the FG output. The second is simpler, because the inputs are commit textures.

## Command recording and pipelines
- Wrapper `CVulkanCmdBuffer` (H:1452). It exposes `rawBuffer()` -> `VkCommandBuffer` (H:1459), plus `bindTexture/setTextureSrgb/setSamplerNearest/setSamplerUnnormalized/bindTarget/uploadConstants/bindPipeline/dispatch` (H:1463-1474) and the public `prepareSrcImage/prepareDestImage/markDirty/insertBarrier` (H:1482-1487).
- gamescope's own system: one global descriptor-set layout and pipeline layout. The layout has no push-constant ranges (R:895-916). `dispatch()` records the barriers, writes a fresh descriptor set from `m_device->descriptorSet()`, binds it and dispatches (R:1802-1950, bind at ~R:1949). Constants go into a uniform buffer at binding 0 through `uploadConstants<T>()` (R:1787-1795), not push constants. `bindPipeline` is plain `vkCmdBindPipeline` COMPUTE (R:1797-1800). Pipelines come from `g_device.pipeline(SHADER_TYPE_...)` (enum entry `SHADER_TYPE_EFFECTS_LAYER0` H:1049).
- Shader binding convention (`src/shaders/descriptor_set.h`): binding 1 `rgba8 writeonly image2D dst`, binding 2 chroma, binding 3 `sampler2D s_samplers[VKR_SAMPLER_SLOTS]`, binding 4 ycbcr samplers, 5/6 LUTs. These are specialization constants 0-7. That is a single output and a few sampled slots, with no SSBOs or extra storage images, so it does not fit a multi-pass pyramid/vector-field algorithm.
- Workgroup: the effects shaders use 8x8x1 (`src/shaders/cs_effects_layer0.comp:46-48`), with dispatch `div_roundup(w,8)` (R:5930-5931).
- Raw handles are available: `g_device.device()` (H:1306), `physDev()` (H:1307), `g_device.vk.*` dispatch table (used e.g. R:1798), `CVulkanTexture::vkImage()` (H:213), `srgbView()/linearView()` (H:199-200), `width()/height()/drmFormat()/format()`. **An external plain-Vulkan library can therefore record straight into `cmdBuffer->rawBuffer()`.** This is the recommended route.

## Images (formats, views, layouts, barriers)
- `layers[0].tex` is `gamescope::Rc<CVulkanTexture>`. For a game it is the commit's imported dmabuf (`commit_t::vulkanTex`, `src/commit.h:42`). Its format follows the client, typically `DRM_FORMAT_XRGB8888/ARGB8888` -> `VK_FORMAT_B8G8R8A8_*`, or `ABGR8888` -> `R8G8B8A8_*` (R:241-243). 10-bit formats are possible. **Assume BGRA8 or RGBA8, and possibly 10-bit.** The library takes `B8G8R8A8_UNORM` / `R8G8B8A8_UNORM` views as before (the default), and, since the "Formats" section below, also 10-bit `A2B10G10R10` / `A2R10G10B10`, `R16G16B16A16_SFLOAT` and the two-plane video formats (NV12, P010) as separate shader variants. gamescope can use them as it likes, or keep skipping FG for anything but 8-bit.
- View naming is inverted. `m_srgbView` is created with `DRMFormatToVulkan(fmt,false)` = **UNORM** (R:2722, R:2736). `m_linearView` uses `DRMFormatToVulkan(fmt,true)` = **_SRGB**, is created only when `bSampled`, and has STORAGE usage stripped (R:2744-2750). The pre-pass reads layer 0 through `setTextureSrgb(0,true)` -> `srgbView()`, which is the UNORM view and gives raw encoded values (R:1905). It samples nearest and unnormalized (`cs` setup ~R:5654-5660). Targets are written through `srgbView()` (UNORM) as a storage image (R:~1933). **FG should sample `srgbView()` (UNORM) with its own linear sampler.** All views passed to FG must be UNORM, never `_SRGB` (`linearView()`): an `_SRGB` view decodes to linear and gives wrong luma. The library cannot check this from a `VkImageView`.
- Layout: every gamescope image is in `VK_IMAGE_LAYOUT_GENERAL` while in use. Sampler and target descriptors use GENERAL (R:1900, R:~1936). Barriers come from `insertBarrier()` (R:2083-2143): UNDEFINED->GENERAL for discarded/dest images, foreign/external queue-family import for dmabufs (`needsImport = externalImage()`, R:2044-2054), and ALL_COMMANDS->ALL_COMMANDS. `markDirty()` makes the next `insertBarrier()` emit a write->read barrier (R:2075-2081).
- Two pitfalls for raw recording:
  1. A dmabuf must be acquired (queue-family import) before you touch it. Call `cmdBuffer->prepareSrcImage(tex)` for both prev and curr, then `cmdBuffer->insertBarrier()`, before handing the handles over.
  2. For the output: call `prepareDestImage(out)` + `insertBarrier()` before the record (UNDEFINED->GENERAL), and `markDirty(out)` after it. The next `dispatch()` then inserts the write->read barrier. The library itself ends with a COMPUTE_SHADER SHADER_WRITE -> COMPUTE_SHADER SHADER_READ barrier only (a FRAGMENT_SHADER dst stage is invalid on the compute-only queue, VUID-vkCmdPipelineBarrier-dstStageMask-06462). Graphics/transfer consumers are covered by gamescope's own ALL_COMMANDS->ALL_COMMANDS `insertBarrier()` after `markDirty(out)`.
- `effectsOutput` is allocated at source size with format `effects_storage_format()`: the input DRM format if it supports optimal STORAGE+SAMPLED, else ARGB8888, i.e. B8G8R8A8 (R:3708-3729). Flags are `bSampled|bStorage|bTransferSrc` (R:3745-3748), created with `BInit(w,h,1,fmt,flags,nullptr)` (R:3750-3751) and pooled until size or format changes (R:3737-3743). An FG output should be allocated the same way, **except for the format**: the FG output must be BGRA8/RGBA8 UNORM (matching the 8-bit input) regardless of the `effects_storage_format()` rule. Note that BGRA8 storage images need `shaderStorageImageWriteWithoutFormat`, or `rgba8` declared on a UNORM BGRA view, which gamescope itself does (`descriptor_set.h` `rgba8` on a B8G8R8A8 view). This works on RADV but is technically a format mismatch. A safe choice for the library is to declare its output image without a format qualifier or to own its output in RGBA8.

## Shader build
- `src/meson.build:55` finds `glslang`/`glslangValidator`. The generator at :59-71 runs `-V @INPUT@ --vn @BASENAME@ -o @BASENAME@.h` (+ depfile), producing a C header with a `uint32_t` array named after the file. Sources are listed in `shader_src` (:73-97), processed at :99 and added via `src += spirv_shaders` (:277). They are included as `#include "cs_effects_layer0.h"` etc. (R:48-57). The library can use the same generator or ship its own headers. If it adds `.comp` files to `shader_src`, they get the same `--vn` naming.

## Holding the previous frame
- `std::array<gamescope::Rc<commit_t>, HELD_COMMIT_COUNT> g_HeldCommits` (`src/steamcompmgr.cpp:1313`, enum at :1305-1309 `HELD_COMMIT_BASE`, `HELD_COMMIT_FADE`). The base commit is painted at `src/steamcompmgr.cpp:2549-2551`.
- Textures are refcounted (`gamescope::Rc<CVulkanTexture>`). Binding one to a command buffer keeps it alive until the submission retires (`m_textureRefs`, R:1735-1740, R:1768-1773). To keep the previous frame, keep an `Rc<CVulkanTexture>` (or `Rc<commit_t>`) of the last distinct layer-0 texture in a static next to `s_bEffectsPassRanLastTime` (R:~5541). Advance it only when `layer0.tex` changes, because re-composites of the same commit must not advance it. Caveat: the client may reuse that dmabuf buffer for a later frame once the commit is released. Holding the Rc of the commit (not only the texture) is what stops the buffer being released back to the client.

## Available device features
- Vulkan 1.2: `shaderFloat16` (only if supported, `m_bSupportsFp16`), `scalarBlockLayout`, `timelineSemaphore` (R:716-722). `shaderInt16` = `m_bSupportsFp16` (R:700-705, detection R:573). Vulkan 1.3: `dynamicRendering` only (R:679-685). `samplerYcbcrConversion`, robustness2 `nullDescriptor` (R:724-734). presentId/presentWait.
- **Not enabled:** `shaderStorageImageReadWithoutFormat` / `WriteWithoutFormat` (the core `features2.features` sets only `shaderInt16`, R:700-705), `storageBuffer16BitAccess`, `shaderSubgroupExtendedTypes`, `synchronization2`. Basic subgroup operations are core in 1.1 and need no feature bit (query `VkPhysicalDeviceSubgroupProperties`). Anything else must be added to device creation in gamescope, or the shaders must avoid it.

## Constraints
- FG only runs where the pre-pass can run: SDR, not YCbCr, base layer not already processed (R:5551-5562).
- Direct scanout bypasses `vulkan_composite`. The DRM backend forces a composite when `vulkan_native_effects_active()` is true (`src/Backends/DRMBackend.cpp:3681-3683`, R:5364-5367). FG needs the same `bNeedsFullComposite |=` hook.
- The composite submits to the compute-only queue (R:1466-1470 comment; `commandBuffer()` uses `queue()/queueFamily()`, R:1395; queue family choice `bTryComputeOnly = true` R:363, R:391). There is no async compute: FG time adds directly to composite latency.
- One queue: all `record()`/`recordEstimate()`/`recordSynth()` calls of one `Interpolator` must be submitted to one queue, in recording order. Its internal buffers are shared between records and ordered only by pipeline barriers. gamescope's single composite queue satisfies this.
- `vulkan_composite` runs once per output frame at display cadence and may run twice per frame (screenshot, R:5376-5395). It does not run once per game frame. Producing the in-between frame needs a cadence change in steamcompmgr. That is out of scope here; see `superdoc/planning/lsfg-in-tree-port.md` §4.
- `effectsOutput` is a single pooled texture, not double-buffered. The FG output must not be the same texture the effects pass writes.

## Library API (as implemented in `gpu/framegen.h`)
Plain Vulkan, no gamescope types. gamescope passes raw handles and does the barriers on its own images.

```cpp
namespace framegen {
struct InitInfo {
  VkPhysicalDevice phys; VkDevice device;
  PFN_vkGetDeviceProcAddr gdpa;   // load via this; gamescope uses its own dispatch table (g_device.vk)
  uint32_t width, height;         // >= kMinSize (16) each
  VkFormat viewFormat;            // default B8G8R8A8_UNORM; R8G8B8A8_UNORM, or another Format (see "Formats"); never _SRGB
  PFN_vkGetPhysicalDeviceMemoryProperties getMemProps;   // REQUIRED: g_device.vk.GetPhysicalDeviceMemoryProperties (H:1166)
  PFN_vkGetPhysicalDeviceFormatProperties getFormatProps;   // optional: lets init() check non-Rgba8 formats (see "Formats")
  VkPipelineCache pipelineCache;  // optional
  Transfer transfer; float linearWhiteNits;   // RgbaF16 only (see "Formats")
  YuvMatrix yuvMatrix; bool yuvFullRange;     // Nv12 / P010 only
  Settings settings;              // tuning knobs, defaults = the approved look (see "Settings" below)
};
class Interpolator {
public:
  bool init(const InitInfo&);     // owns its buffers (DEVICE_LOCAL), descriptor pool, sampler, pipelines
  const char* initError() const;  // why the last init() returned false
  Format format() const;          // the Format of viewFormat (Rgba8, Rgb10, RgbaF16, Nv12, P010)
  void destroy();
  bool resize(uint32_t w, uint32_t h, VkFormat viewFormat);   // re-create on size/format change
  static bool validate(const Settings&);
  bool setSettings(const Settings&);   // runtime knobs, any time, applies to later records
  bool reconfigure(const Settings&);   // also flowScale: re-creates buffers (retire rule of resize)
  Settings settings() const;
  void beginFrame();              // once per command buffer, before its first record*() call
  // All views (Frame; a plain VkImageView converts implicitly): single mip/layer, VK_IMAGE_LAYOUT_GENERAL on entry and exit,
  // 8-bit UNORM for Rgba8, see "Formats" for the others (Nv12 / P010: Frame{Y plane view, CbCr plane view}).
  // prev/curr: SAMPLED usage, made readable by the caller (gamescope: prepareSrcImage + insertBarrier).
  // out: STORAGE usage, caller-owned. Ends with COMPUTE SHADER_WRITE -> COMPUTE SHADER_READ.
  // recordSynth ends with COMPUTE SHADER_WRITE -> COMPUTE SHADER_READ; recordEstimate ends with a COMPUTE -> COMPUTE barrier.
  bool recordObserve(VkCommandBuffer cb, Frame frame);                               // once per NEW real frame, only if Settings::uiProtection != Off (see "Static UI protection")
  bool recordEstimate(VkCommandBuffer cb, Frame prev, Frame curr);            // passes 1-8, once per real pair
  bool recordSynth(VkCommandBuffer cb, Frame prev, Frame curr, Frame out, float t);   // frame at t in (0,1)
  bool record(VkCommandBuffer cb, Frame prev, Frame curr, Frame out, float t = 0.5f); // = estimate + synth
  bool recordSynthBlur(VkCommandBuffer cb, Frame prev, Frame curr, Frame out, float t0, float t1);   // motion blur over [t0, t1] (see "Motion blur")
  void releaseUnusedUi();         // free the UI clean copies Settings::uiHistory no longer keeps (retire rule of resize)
  static constexpr uint32_t kSetRing = 32, kUiSets = 64, kUiFrames = 3, kUiFramesMax = 65, kMinSize = 16;
};
}
```
- **Return values.** `init()`/`resize()` return false below 16x16, for a `viewFormat` that is not one of the supported formats (see "Formats"; `resize()` also for one of another `Format` than `init()` chose), for a null `getMemProps`, or on any Vulkan allocation failure. The object then holds no size and every `record*()` call returns false. gamescope skips frame generation for that surface (shows layer 0 as is). `record()`/`recordEstimate()`/`recordSynth()` return false and record nothing when the object is not initialised, `t` is outside (0, 1), or the per-submission limit of their ring is hit. `recordSynth()` also returns false when there was no `recordEstimate()` since init/resize, or when its `prev`/`curr` are not the views of the latest `recordEstimate()`. `out` is then untouched, so gamescope must not substitute it as layer 0 for that composite (and for N× must not present that slot).
- **Descriptor rings.** Two rings of `kSetRing` = 32 descriptor sets each: every `recordEstimate()` takes the next estimate set, every `recordSynth()` the next synth set, and `record()` takes one of each (it checks both first, so a refused `record()` records nothing). At most 32 estimates and 32 synths may follow one `beginFrame()`; the 33rd of either returns false (and asserts in debug builds), so overflow cannot silently corrupt an earlier record's set. Across submissions the caller must keep the estimates and the synths of all not-yet-retired submissions at <= 32 each, in total. At 4× one real pair uses 1 estimate + 3 synths; gamescope's 1-2 composites per refresh are far below the limit. Call `beginFrame()` once per command buffer, before FG's first record in it.
- **Lifetime.** `resize()`, `destroy()` and `init()` on a live object free buffers that earlier records still use. The caller must make sure every previous record has retired first: wait on the composite timeline point, or `vkQueueWaitIdle`/`vkDeviceWaitIdle` on a resolution or format change (rare). There is no deferred free.
- **`t` and the nearer-frame fallback.** `t` is the time of the generated frame between prev (0) and curr (1). `t == 0.5` is exactly the 2× midpoint path (byte-identical to the 2×-only library). Otherwise the per-pixel field is looked up at time `t` (a tiny flow-res pass), prev/curr are warped by `2t` / `2(1-t)` of it and blended `(1-t)·wa + t·wc`. Every fallback (untrusted pixels, scene cut, more than `Settings::globalFallback` (15 %) untrusted) presents the **nearer real frame**: prev for `t <= 0.5`, curr for `t > 0.5`, so a failed generated frame looks like the base frame rate for one step rather than a jump back to prev. Fallback is decided per `t`: within one pair some generated frames can be held and others not.
- **N× frame generation (3×, 4×).** Per new real frame: one `recordEstimate(prev, curr)`, then N−1 `recordSynth(prev, curr, out[k-1], k/N)` for k = 1..N−1, each into its own output from a pool of N−1 textures allocated like the 2× output (pooled until size/format change). All can go into the one composite command buffer of the frame where the new real frame arrives: synths of one pair share the library's scratch buffers and are serialised by barriers in recording order, and each `out[k]` is complete (COMPUTE write -> read barrier) when its own call ends. `prepareDestImage` + `insertBarrier()` for every `out[k]` before, `markDirty()` for each after. gamescope then presents `out[0..N-2]` on the following vblanks, then the real frame; that cadence (and dropping the rest if the next real frame arrives early) stays gamescope's job. The synths may also be recorded one per composite in later command buffers, as long as no other `recordEstimate()` is recorded in between and prev/curr (and their commit Rc's) stay alive until the last synth retires. A slot must only be presented if its `recordSynth()` returned true. `record(prev, curr, out, 0.5)` remains the 2× call.
- Cost (RX 7900 XTX, game running, timestamp medians, one estimate + N−1 synths per real pair): 1080p 2× 0.33, 3× 0.42, 4× 0.48 ms; 1440p 2× 0.47, 3× 0.60, 4× 0.69 ms. Per extra synth at 1440p: lookup 0.012-0.017 + mismatch 0.037 + resolve 0.050 + fallback 0.005 ms.
- The output is caller-owned: a pooled `CVulkanTexture` like `update_effects_image` (sampled+storage, BGRA8/RGBA8 UNORM matching the 8-bit input) passed as `out->srgbView()`. That lets gamescope substitute it with `SubstituteLayer0()` and keep refcount and barrier tracking.
- The library may allocate its own memory and descriptor pool. It must not submit, must not touch queues, and must not assume RGBA channel order (swizzle is irrelevant for luma if the weights are symmetric, otherwise take a `bgr` flag). It must not require features beyond the list above, and should avoid `shaderStorageImageRead/WriteWithoutFormat` unless gamescope enables them. It should declare formats explicitly (`rgba8`, plus `r16f`/`r32f` etc. for its own internal images).
- Descriptors: the library must bind its own pipeline layout and sets. gamescope rebinds its set in every `dispatch()` (~R:1949), so state clobbering is not a problem.

## Settings (`framegen::Settings`, `gpu/framegen.h`)
A plain struct whose in-class defaults are the tuned values of `proto/interp.py` (the approved look). With the defaults every output is byte-identical to the library before settings existed (2×, 3×, 4×, RGBA/BGRA, record and split). Pass it in `InitInfo::settings`; `Interpolator::validate(s)` tells whether it is accepted.

| Field | Default | Range (else rejected) | Kind | Effect |
|---|---|---|---|---|
| `trustLow`, `trustHigh` | 12, 40 | 0-255, `trustHigh >= trustLow + 1` | runtime | Per-pixel fallback ramp on the 7×7-averaged mismatch between the two warped frames (8-bit levels): below `trustLow` interpolated, above `trustHigh` the nearer real frame. Lower = safer (fewer ghosts/smears, more held pixels); higher = smoother, more artefacts at occlusions. |
| `globalFallback` | 0.15 | (0, 1]; 1 = off | runtime | Share of untrusted pixels above which the whole generated frame is the nearer real frame (fast flicks). Lower = safer, higher = interpolates more chaos. |
| `sceneCutSad` | 30 | (0, 255]; 255 = off | runtime | Mean finest-level block SAD above which the pair counts as a scene cut (nearer real frame). Lower = holds more often. |
| `hudBonus` | 1.0 | 0-255 (useful 0-4) | runtime | Static-HUD protection: per-pixel bonus for the zero vector. Higher = crosshairs/text stay put more reliably, but slow, near-still detail may freeze; 0 = off. |
| `smoothBonus` | 0.5 | 0-255 (useful 0-2) | runtime | Bonus for the smooth (bilinear) vector field. Higher = calmer, softer motion edges; lower = sharper edges, more speckle. |
| `searchPenalty` | 0.5 | 0-255 (useful 0.25-2) | runtime | Motion-search cost per pixel step (interp's `LAMBDA`). Higher = steadier vectors on flat/repetitive surfaces; lower = follows small fast detail more freely. |
| `subpixel` | true | bool | runtime | 1/2 + 1/4 px refinement. Off saves ~0.04 ms (1080p) / ~0.06 ms (1440p) per real pair; ~0.5 dB lower, slightly softer/wobblier on slow pans. |
| `uiHistory` | 3 | 3-`kUiFramesMax` (65) | runtime | Observed real frames that keep their clean copy + mask (LRU). 3 = the newest pair and the one before it. A host that synthesises from an older pair (the lag-spike buffer plays pairs several frames back) sets it to `Pacer::HistoryDepth()`. Memory: 4 B/px per kept frame (4.9 MB at 1280x960, 14.7 MB at 1440p; WholeScreen +1 B/px). See "Static UI protection". |
| `blurSamples` | 0 | 0, 1 (off), 2-16 | runtime | Interpolation-based motion blur: sub-frames averaged by `recordSynthBlur()` (see "Motion blur"). Off = byte-identical. |
| `blurWeights` | Gaussian | Even, Gaussian | runtime | Weights of the sub-frames: 1/N each, or a Gaussian with the window ends at 2 sigma. |
| `flowScale` | 2 | 1, 2, 4 | structural | Motion estimation at 1/flowScale of the frame. 4 = performance mode (~40 % cheaper at 2×, ~30 % at 4×, see below; ~0.1 dB lower on the test clips, can miss thin/fine motion); 1 = full res, much slower (1080p 2×: 0.79 vs 0.35 ms), not better on the test clips. |

- **Runtime** fields: `setSettings(s)` at any time, even between two records of one command buffer. Each record copies the values into its own push constants, so nothing in flight changes and no retire/wait is needed; every `record*()` recorded after the call uses the new values (a synth uses the values current when the synth is recorded, the estimate those current at its record). `setSettings` returns false and changes nothing if `s` is invalid or its `flowScale` differs from the active one.
- **Structural** (`flowScale`): `reconfigure(s)` applies all fields and re-creates the size-dependent buffers when `flowScale` changes. Same rule as `resize()`: every earlier record must have retired (`vkQueueWaitIdle` on a settings change is fine, it is a rare user action), and the next `recordSynth()` needs a new `recordEstimate()`. Invalid `s` -> false, nothing changed; an allocation failure leaves the object without a size like a failed `resize()`. `resize()` keeps the active settings. `settings()` returns the active ones.
- Left fixed (not exposed): block size 8, search radii (2 per level, 6 at the coarsest; with the pyramid they already track up to ~250 px of motion between real frames at 1080p, and larger radii cost LDS/ALU for no measured gain), pyramid stop `MIN_SIDE` 48, per-pixel window 5×5, mismatch box 7×7, sub-pixel step sizes, `LOOKUP_ITERS` 2 (3 was measured worse), time-weighted blend. These are either baked into workgroup/LDS sizes or have a single sensible value; none is something a user can judge.

**Cost** (RX 7900 XTX, `compare.py bench 100 [--set ...]`, profiled medians, one estimate + N−1 synths per real pair, ms):

| config | 1080p 2× | 1080p 3× | 1080p 4× | 1440p 2× | 1440p 3× | 1440p 4× |
|---|---|---|---|---|---|---|
| defaults (flowScale 2, subpixel on) | 0.35 | 0.42 | 0.49 | 0.50 | 0.62 | 0.71 |
| `subpixel = false` | 0.31 | 0.39 | 0.45 | 0.43 | 0.56 | 0.65 |
| `flowScale = 4` | 0.20 | 0.27 | 0.33 | 0.28 | 0.39 | 0.49 |
| `flowScale = 4`, `subpixel = false` | 0.19 | 0.26 | 0.31 | 0.25 | 0.36 | 0.46 |

The settings only change the per-pair estimate; the per-synth cost (~0.06 ms at 1080p, ~0.1 ms at 1440p) is the same in every row. Quality on the test clips (mean PSNR vs the real in-between frame, 10 pairs, 2×): defaults 29.15 dB, `flowScale` 4 29.05, `subpixel` off 28.66, both 28.33, `flowScale` 1 29.14.

**Recommended user-facing subset for gamescope-ritz (convars / profile / overlay):**
1. **Frame generation: Off / 2× / 3× / 4×** — the multiplier (gamescope cadence; `recordSynth` at k/N).
2. **Quality preset: Quality / Performance** — `flowScale` 2 / 4 (Performance also `subpixel = false`). Plain language: "Performance estimates motion at lower resolution: cheaper, slightly less precise on thin or fine moving detail." Changing it calls `reconfigure()` after a queue wait.
3. **Artifact safety: Low / Default / High** — trust ramp, e.g. High = (8, 28), Default = (12, 40), Low = (16, 56); optionally also `globalFallback` 0.10 / 0.15 / 0.25. Plain language: "Higher shows the real frame instead of guessing where motion is hard to track: fewer ghosting/smearing artefacts, but more spots that look like the base frame rate."
4. **Static HUD protection: Off / Normal / Strong** — `hudBonus` 0 / 1 / 2.5. Plain language: "Keeps crosshairs, text and other fixed overlays from wobbling or being dragged along with the scene. Too strong can make slow-moving areas stutter."
5. Expert only (config/convar, no overlay entry): `searchPenalty`, `smoothBonus`, `sceneCutSad`, `globalFallback` on its own, and a "scene-cut detection" toggle (`sceneCutSad` 255 = off). The Low/High preset values above are suggestions within the validated ranges; only the defaults were reviewed visually by the user.

## Formats (`InitInfo::viewFormat`, `framegen::Format`, `framegen::Frame`)

The library works on more than 8-bit RGB: 10-bit RGB, half-float RGB (linear / scRGB / encoded) and the two-plane video formats NV12 and P010. The format is chosen **once at `init()`** by `InitInfo::viewFormat`, which maps to exactly one `Format` (`framegen::formatOf()`, `Interpolator::format()`):

| `Format` | `viewFormat` | What the host passes per frame (`Frame`) | Clean copy (UI protection), per px |
|---|---|---|---|
| `Rgba8` (default) | `B8G8R8A8_UNORM`, `R8G8B8A8_UNORM` | one view (`Frame::view`); a plain `VkImageView` converts implicitly | 4 B (`R8G8B8A8`) |
| `Rgb10` | `A2B10G10R10_UNORM_PACK32`, `A2R10G10B10_UNORM_PACK32` | one view | 4 B |
| `RgbaF16` | `R16G16B16A16_SFLOAT` | one view | 8 B |
| `Nv12` | `R8_UNORM` (the Y plane view) | `Frame{Y plane view, CbCr plane view}`; CbCr is `R8G8_UNORM`, half size rounded up | 1.5 B (`R8` + `R8G8`) |
| `P010` | `R16_UNORM` (the Y plane view) | `Frame{Y plane view, CbCr plane view}`; CbCr is `R16G16_UNORM` | 3 B (`R16` + `R16G16`) |

At 1440p one clean copy is 14.7 MB (Rgba8, Rgb10), 29.5 MB (RgbaF16), 5.5 MB (Nv12), 11.1 MB (P010); the host's own frame buffers (`pacing.h`'s `HistoryDepth()`) are in the host's format and scale the same way.

**Unchanged for 8-bit RGB.** Rgba8 is the default and the shaders it uses compile to the **same SPIR-V as before this section existed** (see "Rgba8 stays byte-identical" below): a host that passes `B8G8R8A8` / `R8G8B8A8` views changes nothing, not even its build.

### Host contract per format

- **All formats:** views are 2D, single mip/layer, the size given to `init()`/`resize()`, `VK_IMAGE_LAYOUT_GENERAL` on entry and exit, `prev`/`curr` with SAMPLED usage and `out` with STORAGE usage, like before. The library samples with linear filtering and writes with `imageStore`: the device must support SAMPLED + SAMPLED_FILTER_LINEAR + STORAGE_IMAGE on the view format (optimal tiling). Pass `InitInfo::getFormatProps` (`vkGetPhysicalDeviceFormatProperties`, instance-level, optional) and `init()` checks this for the view format and the library's own clean-copy format and returns false with `Interpolator::initError()` set to a reason; with null it trusts the host. Rgba8 is never checked (as before).
- **Device feature:** every format except Rgba8 and RgbaF16 writes through a storage image of format `rgb10_a2` / `r8` / `rg8` / `r16` / `rg16`, which needs **`shaderStorageImageExtendedFormats`** enabled at device creation. gamescope enables only `shaderInt16` today, so enabling Rgb10, Nv12 or P010 there is one more line in its `VkPhysicalDeviceFeatures`. RgbaF16 (`rgba16f`) and Rgba8 need nothing extra. No format needs `shaderStorageImageRead/WriteWithoutFormat`.
- **Rgb10:** values are UNORM codes, treated as perceptually encoded (SDR gamma or PQ); alpha (2 bit) passes through. `A2R10G10B10` additionally needs the device to offer storage on it, which many do not (the init check says so).
- **RgbaF16:** `InitInfo::transfer` says how the values are encoded (fixed at init, it is baked into the pipelines): `Transfer::Encoded` (default; sRGB / gamma / PQ / HLG already applied) or `Transfer::Linear` (linear light, scRGB or plain linear; `InitInfo::linearWhiteNits` is the luminance of 1.0, default 80 = scRGB). Values above 1.0 and negative values are kept, the output is never clamped. Only the *measurement* (mismatch, stillness, luma) depends on the transfer; the frames are warped and blended in their own space.
- **Nv12 / P010:** create ONE multi-planar image, `VK_FORMAT_G8_B8R8_2PLANE_420_UNORM` / `VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16`, with `VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT` and usage SAMPLED (+ STORAGE for `out`), then two views, each with a `VkImageViewUsageCreateInfo` of SAMPLED (+ STORAGE for `out`): the **Y view** (`aspectMask = PLANE_0`, format `R8_UNORM` / `R16_UNORM`) and the **CbCr view** (`PLANE_1`, `R8G8_UNORM` / `R16G16_UNORM`), passed as `Frame{y, cbcr}`. `InitInfo::viewFormat` is the Y view's format. Two separate single-plane images (an R8 and an R8G8 image) work just as well, which is also what the library allocates for its own clean copies; `formattest` runs both (`--separate-planes`). **`out`'s two views are both written**: Y at full resolution, CbCr at half resolution. No `VkSamplerYcbcrConversion` is used or needed in the library; the host's device must have the `samplerYcbcrConversion` feature enabled only for creating multi-planar images at all. P010 samples are 10 bits in the high bits of a 16-bit word (`code << 6`): the library reads `code`, writes `code << 6` exactly (code grid of 10 bits), so the low 6 bits of its output are zero. `InitInfo::yuvMatrix` (`Bt601`, `Bt709` default, `Bt2020`) and `InitInfo::yuvFullRange` (default limited/video range 16-235 / 16-240) describe the colour; they only change how mismatch and luma are measured (below), never the warp or the blend, which run on Y'CbCr directly (a linear transform of R'G'B', so blending in it equals blending in R'G'B'). **Chroma is assumed centre-sited** (the chroma texel sits in the middle of its 2x2 block): the CbCr view is sampled at the same normalised coordinate as luma and written from the 2x2 block centre; zero-motion pixels and every fallback therefore round-trip a chroma texel exactly, whatever the real siting (left-sited MPEG-2/H.264 video just has a quarter-texel chroma shift that is the same in the input and in the output). Chroma of a generated frame uses the trust and the vectors of the top-left pixel of its 2x2 block. Odd sizes work (chroma is `(w+1)/2 x (h+1)/2`).

### Thresholds and transfer: one scale for every format

Every level-based setting (`trustLow/High`, `sceneCutSad`, `hudBonus`, `smoothBonus`, `searchPenalty`, `uiStillThreshold`) **keeps its meaning of "8-bit levels" in every format**; nothing is rescaled per format by the host. Inside, the library converts what it measures to the levels of an 8-bit picture of the same content, so a value means the same *perceived* amount:

| Format | Level of a channel (mismatch, stillness) | Luma for the motion estimate |
|---|---|---|
| Rgba8 | the code (as before, rounded) | `(4899 R + 9617 G + 1868 B + 8192) >> 14` as before |
| Rgb10 | `code / 1023 * 255` (unrounded) | the same weights, in floats |
| RgbaF16, `Encoded` | `value * 255` | same weights, floats |
| RgbaF16, `Linear` | `PQ(value * linearWhiteNits / 10000) * 255`: a perceptually uniform code, so a level weighs the same in the shadows and in a 1000-nit highlight | same weights on the PQ levels |
| Nv12 / P010 | the (Y, Cb, Cr) difference is converted to the R'G'B' difference it stands for (matrix and range of `InitInfo`), then the max channel as for RGB | the Y plane, `(Y - black) * 255/219` (limited) so it spans 0-255 like RGB luma; 10-bit codes / 4 first |

Blending, warping and the fallback frame always stay in the source's own space, and the packed mismatch is clamped to 255. Because the metric is in 8-bit levels, the default settings are the tuned ones for every format; only the *source* of the number differs. Stillness for UI protection is measured the same way and stored as 8-bit levels (so a pixel that is bit-identical in two frames is still in every format).

### Shader variants, build, embedded size

Only the shaders that touch colour are format-specific: `luma`, `error`, `resolve`, `fallback`, `ui_detect`, `ui_box_update`, `ui_inpaint`, `ui_patch`, `blur_error`, `blur_resolve` (+ `ui_inpaint_chroma`, `ui_patch_chroma` for the YUV formats, the half-resolution CbCr passes). The motion estimate (`down`, `medhist`, `search`, `median3`, `subpix`, `pixfield`, `lookup`, `ui_mask`, `ui_box_mask`, `blur_lookup`) runs on luma only and is **one shader for all formats**. Each colour shader is compiled once per format **from the same source** with `-DFG_FORMAT_RGB10|F16|NV12|P010` (`shaders/format.glsl`); `init()` creates only the pipelines of the chosen format, so the other variants cost nothing at runtime. `gpu/meson.build` generates `<name>_<fmt>_spv.h` with the array `<name>_<fmt>_spv` (`fmt` = `rgb10`, `f16`, `nv12`, `p010`); **Rgba8 keeps the historical `<name>_spv.h` / `<name>_spv`**. `framegen.cpp` includes each variant set behind `__has_include`: a host that compiles only the historical shaders gets an Rgba8-only library that refuses other formats at `init()` with "the shader variants of this format are not built into the library". The meson option `formats` (array, default all four) picks the variants of the standalone build.

Embedded SPIR-V (words x 4): the Rgba8 set (20 shaders, as before) 205,396 B; plus Rgb10 111,156 B, RgbaF16 120,496 B, Nv12 150,320 B, P010 150,320 B if built. **What gamescope's `src/meson.build` `custom_target` list must do:** for Rgba8, **nothing** (same names, same sources, no `-D`). To offer a format, add the variant shaders of that format to the list with `-DFG_FORMAT_<FMT>=1`, output `<name>_<fmt>_spv.h` and `--vn <name>_<fmt>_spv` for the 10 colour shaders (+ the 2 chroma shaders for `nv12` / `p010`); the library then picks them up through `__has_include`.

### Rgba8 stays byte-identical

With no `-D`, `format.glsl` declares nothing and its macros expand to exactly the tokens the shaders used to contain, so glslang produces **the same SPIR-V**. Verified on the 20 library shaders (`glslangValidator -V --target-env vulkan1.2 --vn <name>_spv`, the exact flags of both builds): the generated headers of this tree and of `f8c9423` are `cmp`-identical for every one. The C++ side adds nothing for Rgba8 (same descriptor layouts, pools and pipelines; no format query). As a smoke test on top of that, `uitest check --hash 640 360` (RGBA and BGRA) prints identical FNV-1a hashes for all 11 generated frames from a build of `f8c9423` and from this tree, and `blurtest check` passes in both.

### Spec constants and layout of the variants (for maintainers)

Specialization constants of the non-Rgba8 colour pipelines (`VkSpecializationInfo` at pipeline creation; there is no spare push-constant space, the block is full at 128 bytes): RgbaF16 `0 = linear?`, `1 = white nits`; YUV `0 YU_SCALE` (sample -> level), `1 YU_QL` (one code in levels), `2 YU_CODES`, `3 YU_BLACK`, `4 YU_YS` (luma gain), `5..8` the R'G'B' per Cb/Cr level (matrix and range folded in). The descriptor set of a YUV `Interpolator` has three more bindings (8, 9: the CbCr planes of `prev`/`curr`, immutable sampler; 10: the CbCr plane of `out`); the UI set two more (4: `srcC`, 5: `dstC`). They are created only for the YUV formats.

### Tests (`gpu/formattest.cpp`, single frames, no footage)

`formattest check [W H]` (default 640x360) runs, per format the device supports, a 12-frame synthetic sequence (the `uitest` scene) and compares against the same scene run through Rgba8 and against the format's own quantised ground truth, all after decoding to 8-bit RGB levels: sane output (not worse than Rgba8 by more than 1.5 dB; YUV 4 dB, because half-resolution chroma loses on the saturated noisy texture), both YUV planes written (a 0-filled sentinel output), a narrow blur window, static crosshair and HUD box exact with `Crosshair` / `WholeScreen` (and wrong without, so the check has teeth), no cost of protection to the background; plus a 10-bit/float gradient (Rgb10 and RgbaF16 give 4x / far more distinct values than Rgba8 on a 0.2-0.3 ramp), linear float highlights above 1.0 kept, and API checks (`_SRGB` refused with a reason, `resize()` across formats refused, a YUV frame without a chroma view refused). `formattest time [W H]` prints one GPU time per format.

## Static UI protection (`Settings::uiProtection`)
The crosshair protection that used to live in the host (gamescope-ritz `FrameGenHost.cpp`: `RecordCrosshairDetect`/`RecordCrosshairPatch`, `cs_fg_crosshair_*.comp`, `cs_fg_inpaint.comp`) is part of the library, in two modes. The reason it is in the library: any host gets it by flipping a setting, the repository is public, and the algorithm only needs the frames.

**Settings** (all runtime, `setSettings()`; defaults = today's host values; `validate()` checks every range):

| Field | Default | Range | Meaning |
|---|---|---|---|
| `uiProtection` | `Off` | `Off`, `Crosshair`, `WholeScreen` | `Off` records nothing and allocates nothing; output byte-identical to the library before this feature (verified: `uitest check --hash` against `e2f3b61`). |
| `uiBoxWidth`, `uiBoxHeight` | 0, 0 | 0-16384 | The full-resolution box, pixels. 0 = that side from `uiCrosshairArea`. An explicit size wins; clamped to [4, frame]. |
| `uiCrosshairArea` | 0.03 | (0, 1] | Area-derived box: square of side `round(sqrt(area*w*h))`, clamped to [32, min(w, h)] (333 px at 2560x1440). |
| `uiBoxOffsetX`, `uiBoxOffsetY` | 0, 0 | -16384..16384 | Box centre relative to the frame centre (+x right, +y down). Clamped so the box stays inside the frame. |
| `uiStillFrames` | 8 | 1-255 | Consecutive still real frames before a pixel is masked. |
| `uiStillThreshold` | 3.5 | 0-255 (8-bit levels, in every format) | A pixel is still if no channel moved more than this since the previous real frame. |
| `uiInpaintReach` | 12 | 1-16 | Inpaint walk length per direction. |
| `uiMaxStillFraction` | 0.75 | (0, 1] | WholeScreen: skip the whole-screen passes of a frame whose still share outside the box exceeds this (1 = never). |

**How to size the box for your crosshair.** The box is always protected at full resolution, in both modes. Make it 2-3x the crosshair's largest extent (the whole reticle including its spread while shooting/moving and its outline), and set `uiBoxOffsetX/Y` if the reticle is not at the frame centre. Too small: the ends of the reticle fall outside the protection (`uitest check` shows this with a 20x20 box on a 35x35 px crosshair: 61 of 305 crosshair pixels wrong). Too big: only costs a few more pixels of work. Size changes at runtime need no GPU wait: the box buffers only ever grow (see "Memory").

**API contract** (full text in `gpu/framegen.h`):
- `recordObserve(cb, frame)`: once per NEW real frame, in order, for every real frame (also those never interpolated), after its pixels are final. Returns true and records nothing when `Off`. The library keys its entries by `frame`, so a host that reuses views (a ring of images) must observe each new content. At least `kUiFrames` = 3 observed frames are kept (LRU; `Settings::uiHistory` keeps more, see below). `recordEstimate(prev, curr)` and `recordSynth` read the library's clean copies of those frames; `recordSynth` additionally reads the real pixels from the `curr` view you pass, so `curr` must still hold the real frame at synth time (this was already required).
- Implicit observe: a host that never calls `recordObserve` is observed by `recordEstimate` (prev if unseen, curr unless it is the newest observed view). Caveat: new content under the same view handle is not noticed, and a frame never used in a pair is never seen by the counters. Once `recordObserve` was called, estimates do not observe implicitly; a pair of which a view was not observed (just switched on, skipped, evicted) is interpolated unprotected like `Off`.
- `recordSynth` returns false if an observe evicted one of the pair's two entries since its `recordEstimate` (observe two newer frames between them) or re-observed one of the views.
- Descriptor sets: a third ring of `kUiSets` = 64 sets. `recordObserve` takes 1 (Crosshair) or 2 (WholeScreen), each `recordSynth` takes up to 2, an implicit observe in `recordEstimate` takes the same as `recordObserve`. Keep them per `beginFrame()` and per not-yet-retired submission <= 64 (a normal 4x frame needs about 3 + 2x3). Exhaustion returns false and records nothing.
- Retire rule: `resize()`, `destroy()` and `reconfigure()` that changes `flowScale` free all UI resources; same rule as the other buffers (all earlier records retired). `setSettings()` only creates (the first time a mode is switched on: `uiHistory` clean images + buffers, WholeScreen: + masks and cell state) and returns false if that allocation fails (settings unchanged). Growing the box allocates a bigger buffer at the next `recordObserve`; the old one goes to a graveyard that is freed by the next `resize`/`destroy`, so no GPU wait is needed. The clean images are in the source's own format, in logical channel order (`R8G8B8A8_UNORM` for Rgba8 as always, `A2B10G10R10_UNORM_PACK32` for Rgb10, `R16G16B16A16_SFLOAT`, and for NV12 / P010 two images, `R8` + `R8G8` or `R16` + `R16G16`), independent of the BGRA/RGBA order of `viewFormat`, and are moved UNDEFINED -> GENERAL by a one-off barrier at the first `recordObserve`.
- Profile names: `ui_detect` (clean copy, stillness, masks) and `ui_inpaint` per `recordObserve`; `ui_patch` per `recordSynth`. An observe opens the profile the next `recordEstimate` continues.

**Algorithm** (shaders `gpu/shaders/ui_*.comp`, common header `ui_common.glsl`, all embedded with the usual `--vn <name>_spv`):

| Pass | File | When | What |
|---|---|---|---|
| clean copy + cell detect | `ui_detect.comp` | every observe | One thread per 2x2 cell (`kUiCell` in `framegen.cpp`; 1, 2 or 4 work): copies the real frame into the clean image. WholeScreen: cell mean (8-bit) vs the stored previous mean, max channel diff <= `uiStillThreshold` -> count+1, else 0; cells overlapping the box never count; a cell with count >= `uiStillFrames` is still: it flags its 32x32 tile (and the neighbours within the 2 px dilation) and is counted into the frame's header word. |
| box counters | `ui_box_update.comp` | every observe | Per box pixel (full res): the stored word holds the previous ORIGINAL pixel + count, updated in place. |
| box mask | `ui_box_mask.comp` | every observe | 5x5 neighbourhood of the counters: 255 within 1 px of a still pixel, 128 within 2 px. One thread per 4 pixels (packed bytes). |
| whole-screen mask | `ui_mask.comp` | WholeScreen | Per flagged 32x32 tile only: the same dilation applied to the upsampled cell mask at full res. Exits at once for unflagged tiles and for a suppressed frame (still share > `uiMaxStillFraction`). |
| inpaint | `ui_inpaint.comp` | every observe, whole screen then box | 16x16 tile, mask + `uiInpaintReach` halo in shared memory, one thread per pixel, 8 walks then 8 independent fetches; 1/d^2 weights over the first unmasked pixel in each direction; masked pixels only. A tile with no masked pixel, or entirely inside the masked area, exits early. Whole screen: unflagged tiles count as unmasked. |
| patch | `ui_patch.comp` | every `recordSynth`, whole screen then box | `out = mix(out, real, mask)` over the mask rectangle (flagged tiles only for whole screen). Real = the `curr` view. |

Box passes are sized to the box (e.g. 333x333 at 1440p); whole-screen passes run per cell (1/4 of the pixels) or per flagged tile. Resolutions: box = full; detection = 1/2 per axis; mask, inpaint, patch = full, restricted to flagged tiles.

**Why the real pixels come from `curr`.** The patch uses the observed frame the synth leads up to (`curr`): pixels are still for >= `uiStillFrames` frames there, so `prev` is the same within the threshold. The library therefore stores no full-resolution copies of originals, only the clean copy.

**Skip rule.** WholeScreen counts the still cells outside the box on the GPU (one atomic per workgroup). If more than `uiMaxStillFraction` of all cells are still, the mask, inpaint and patch passes of that frame exit at the first instruction (the clean copy is still made): a picture that static needs no generated frame to differ from the real one. The counters keep updating, so there is no flicker of the mask, only of whether it is applied. The box is never skipped.

**Deep history (`Settings::uiHistory`, runtime).** The kept-frame count is a setting, 3 (default) to `kUiFramesMax` = 65 (= `pacing.h`'s `kHistoryMax`). With the lag-spike buffer the pacer synthesises from pairs several real frames back, and a synth from a pair whose clean copies were evicted runs **unprotected** (`recordEstimate` with explicit observes then behaves like `Off` for that pair), so the host sets `uiHistory = pacer.HistoryDepth()` (or `Report::historyFrames`): it follows the buffer, so memory scales with the buffer, not with a fixed cap. `uitest deep` checks it: a synth from a pair 10 frames behind the newest has the exact crosshair and HUD with `uiHistory` 14, and is bit-identical to `Off` with 3 or 10 (frame just evicted).
- *Growing* (`setSettings` with a larger `uiHistory`): first re-activates slots that kept their memory, then allocates the missing clean images (and, in WholeScreen, masks) at once, with no GPU wait because nothing is freed. The new slots hold no frame until observed. Returns false, nothing changed, if an allocation fails.
- *Shrinking*: the least recently observed frames are evicted (their pairs are unprotected from then on; the pacer shrinks only past frames the content time has left), but **the memory stays allocated** (in use by in-flight records, possibly), so a later raise is free. It is freed by `releaseUnusedUi()`, which has the retire rule of `resize()` (every earlier record retired): call it where the host already waits for the GPU, e.g. after `HistoryDepth()` has stayed lower for a few seconds. `resize()` / `reconfigure()` (new size or `flowScale`) / `destroy()` free everything. Why not free on shrink: the pacer's depth moves up and down with the buffer, and freeing an image a running submission still reads needs a GPU wait; keeping it makes the common case (depth wobbling) allocation-free.
- At the default (3) memory and output are exactly as before.

**Memory** (4 bytes/px clean copy, 3 kept frames; 1440p / 1080p; **per kept frame** 14.7 MB / 8.3 MB, +1 B/px of mask in WholeScreen (3.7 / 2.1 MB), so it scales with `uiHistory`): Crosshair 44.2 + <1 MB / 24.9 + <1 MB (clean copies + box buffers, 333 px box); WholeScreen adds 3 full-frame masks (1 byte/px: 11.1 MB / 6.2 MB) and the cell state (1/4 of the pixels x 4 bytes: 3.7 MB / 2.1 MB).

**Cost** (RX 7900 XTX, GPU timestamp medians of the pass groups, per real frame: one `recordObserve` + one `recordSynth`; `uitest bench`). Scenes: *moving* = textured background panning 14,6 px/frame with 16 independent moving objects, static crosshair and HUD box; *still* = every frame identical; *70% still* = the left 70% identical, the right 30% as moving (just below the default skip limit). "no skip" = `uiMaxStillFraction = 1`.

| scene / mode (1440p) | ui_detect | ui_inpaint | ui_patch | UI total |
|---|---|---|---|---|
| moving, Crosshair | 0.052 | 0.012 | 0.002 | 0.067 ms |
| moving, WholeScreen | 0.057 | 0.032 | 0.006 | 0.095 ms |
| still, Crosshair | 0.055 | 0.013 | 0.004 | 0.071 ms |
| still, WholeScreen (frame skipped, 100% > 75%) | 0.057 | 0.019 | 0.007 | 0.083 ms |
| 70% still, WholeScreen | 0.114 | 0.110 | 0.027 | 0.251 ms |
| still, WholeScreen, no skip (worst case) | 0.138 | 0.143 | 0.037 | 0.318 ms |

| scene / mode (1080p) | ui_detect | ui_inpaint | ui_patch | UI total |
|---|---|---|---|---|
| moving, Crosshair | 0.038 | 0.012 | 0.002 | 0.052 ms |
| moving, WholeScreen | 0.040 | 0.023 | 0.004 | 0.068 ms |
| still, Crosshair | 0.039 | 0.012 | 0.003 | 0.055 ms |
| still, WholeScreen (skipped) | 0.042 | 0.016 | 0.005 | 0.062 ms |
| 70% still, WholeScreen | 0.073 | 0.081 | 0.017 | 0.172 ms |
| still, WholeScreen, no skip (worst case) | 0.084 | 0.094 | 0.023 | 0.200 ms |

About half of `ui_detect` is the full-frame copy into the clean image (bandwidth). The timestamp sums over-count the marginal cost: back-to-back per pair (8 pairs in one command buffer, estimate + one synth) at 1440p: Off 0.47 ms, Crosshair 0.50, WholeScreen 0.52 (moving); 70% still WholeScreen 0.68 ms.

**Validation** (`uitest check`, synthetic 12-frame sequence: background panning 14,6 px/frame, 16 independent moving objects, a static opaque crosshair at the centre and a static opaque HUD box in a corner; last pair interpolated at t = 0.5 and 0.25): the crosshair pixels (305) and HUD pixels (6960) of the generated frame must equal the real ones within 1 level. Results at 1280x720:

| | crosshair wrong | HUD wrong | background PSNR |
|---|---|---|---|
| Off | 16 / 305 | 182 / 6960 | 29.11 dB |
| Crosshair | 0 | 182 (not protected) | 29.11 dB |
| WholeScreen | 0 | 0 | 29.05 dB |
| Crosshair, explicit 80x60 box | 0 | - | 29.11 dB |
| Crosshair, 20x20 box (smaller than the reticle) | 61 | - | 29.11 dB |
| crosshair offset (+37, -23), 120x120 box with the same offset | 0 | - | 29.27 dB |
| offset crosshair, WholeScreen + offset box | 0 | 0 | 29.20 dB |

At 2560x1440 (same scene): Off 11/305 crosshair and 112/27840 HUD pixels wrong, background PSNR 36.91 dB; Crosshair 0 / 112, 36.80 dB; WholeScreen 0 / 0, 36.75 dB. The 0.1-0.2 dB background difference is the price of showing real pixels where a pixel was judged still (identical background that happens to be still is shown from the real frame).

Also checked: implicit observe is bit-identical to explicit observe; runtime switching Off -> Crosshair -> WholeScreen mid-stream protects both; WholeScreen -> Off mid-stream gives output bit-identical to an Off run; RGBA and BGRA; `validate()` rejections; no Vulkan validation or synchronization-validation messages for RGBA (BGRA output views give the library's existing `rgba8`-on-BGRA storage-format warning, now also for the patch pass that reads `out`).

**What gamescope-ritz must do** (a later task; not done here):
1. Add the six new shaders to the `custom_target` list that compiles the library's shaders (`--vn <name>_spv`, include dir for `ui_common.glsl`): `ui_detect.comp`, `ui_box_update.comp`, `ui_box_mask.comp`, `ui_mask.comp`, `ui_inpaint.comp`, `ui_patch.comp` (+ the header `ui_common.glsl`, a dependency of all six).
2. Set `Settings::uiProtection` (and the box fields from its own settings: size, offset, still frames) in `ToLibrarySettings()`; `setSettings` can fail (allocation), keep the old value then.
3. Call `recordObserve(cb, frameView)` once per new real frame at the point where it currently copies a real frame into its ring slot and runs `RecordCrosshairDetect`, including frames it will not interpolate.
4. Delete `EnsureCrosshair`/`ReleaseCrosshair`/`RecordCrosshairDetect`/`RecordCrosshairPatch`, the per-slot patch textures, `cs_fg_crosshair_detect.comp`, `cs_fg_inpaint.comp`, `cs_fg_crosshair_patch.comp`; the library now owns the clean copies, so the host's ring slots hold real, un-inpainted frames. The host must keep `curr` valid until the last synth of its pair retires (as before).
5. Account for `kUiSets` (64) per submission in its pacing guard, like `kMaxSynthsPerPair` for the 32-set rings.

## Motion blur (`Settings::blurSamples`, `recordSynthBlur`)
Interpolation-based motion blur, the technique of video tools such as Blur and smoothie-rs: interpolate to a high rate, then average groups of sub-frames, which gives a physically plausible blur without the "smear" of a screen-space filter. The idea, in the user's words: *"interpolated motion blur, which people basically achieve by, for example, interpolating a 60 FPS video to 240, and then blending the frames together to get like really accurate motion blur."* In the library it is one primitive: **one output image = the weighted average of N synthesised sub-frames at t in [t0, t1] of the current real pair.** It works without frame generation (blur only) and together with it.

```cpp
Settings s; s.blurSamples = 4; s.blurWeights = BlurWeights::Gaussian;   // runtime; 0 or 1 = off
fg.setSettings(s);
fg.recordEstimate(cb, prev, curr);                          // once per real pair, as before
fg.recordSynthBlur(cb, prev, curr, out, t0, t1);            // 0 <= t0 <= t1 <= 1, one call per output image
```

**Sub-frames and weights.** N = `blurSamples` (any value 2-16; 2-8 are the expected ones, odd counts work). Sample k sits at `t_k = t0 + (t1 - t0) k / (N - 1)`, k = 0..N-1, both window ends included. Weights `w_k` are normalised to sum 1: `Even` = 1/N; `Gaussian` = `exp(-2 x_k^2)`, `x_k = (2k - (N-1)) / (N-1)`, i.e. a Gaussian with sigma = 1/4 of the window (ends at 13.5 % of the centre's weight). Even is a box shutter (what a physical camera does, hard-edged smear); Gaussian is softer and its effective length is shorter (full width at half maximum 0.59 of the window). A sub-frame is **exactly** what `recordSynth(t_k)` produces (same field lookup, warps, trust ramp, scene-cut and `globalFallback` decisions, nearer-real-frame rule, rounded to 8 bit); `t_k <= 0` / `>= 1` is the real frame prev / curr (no synthesis, so a window ending at 1 costs one sample less). The average of the 8-bit sub-frames is rounded once. With `t0 == t1`, or blur off, the result is `recordSynth(t1)` bit for bit (`t1` = 0 / 1: a copy of prev / curr); with blur off and `t0 < t1` the window collapses to its end `t1` (a host that centres its window should call `recordSynth` itself while blur is off).

**Passes (what is shared, what is per t).** *Shared by every output of the pair*: the whole `recordEstimate` (luma, pyramid, search, medians, sub-pixel, per-pixel field, scene-cut SAD sum) and, with UI protection, the observe (clean copies, masks). *Per sub-frame t_k* (nothing here can be shared, the field at t differs): the field lookup (`blur_lookup.comp`, flow res), the full-res mismatch of the two warps (`blur_error.comp`), then the 7x7 trust, the warps and blend and the fallbacks of that sample (`blur_resolve.comp`). `blur_lookup` and `blur_error` are one dispatch for all samples (z = sample); `blur_resolve` is one dispatch that loops the samples per pixel and accumulates in registers, so **no sub-frame is ever written to memory**. The whole-frame "more than `globalFallback` untrusted" decision of a sample needs a count over the whole frame, so `blur_resolve` assumes no sample is flicked and counts each sample's untrusted pixels as it goes, and a second dispatch of the same shader (`fix`) exits at once unless some sample turned out flicked (rare), in which case it redoes the output with the final counts. Result identical to deciding first, one counting pass per sample cheaper. One blurred output is therefore 4 dispatches whatever N is. Timestamps in `readTimings`: `blur lookup`, `blur mismatch`, `blur resolve`, `blur fixup`.

**UI protection.** With `uiProtection != Off` the patch is applied **after** the average (`ui_patch`, once per output, real pixels from `curr`), so a static crosshair or HUD is the exact real pixels while the world around it is blurred. The estimate runs on the clean copies as always.

**Rings and memory.** One synth descriptor set per call (not per sample) plus the UI patch sets, i.e. the same as `recordSynth`; 8 outputs x 8 samples in one pair take 8 sets. Scratch per sample: the field at t (8 B per flow pixel) and the mismatch (1 B per pixel) = 3 B per pixel at `flowScale` 2 (N = 8: 29 MB at 1280x960, 88 MB at 1440p), 9 B at `flowScale` 1. It is allocated by the first blurred call, only ever grows (the old buffer waits in a graveyard until `resize`/`reconfigure`/`destroy`, like the UI buffers; no GPU wait) and `recordSynthBlur` returns false if that allocation fails. Changing `blurSamples` live is free.

### How a host calls it
`s` is the **shutter** as a fraction of an interval and the window always *ends* at the instant the output shows, so it needs no future frame and adds no delay. A window that starts before the pair (negative `t0`) cannot be built from this pair; the two modes below are arranged so that it never happens.

- **Blur only (no frame generation).** Each real frame is shown blurred over the last part of the interval that led to it: `recordEstimate(prev, curr)`, then `recordSynthBlur(prev, curr, out, 1 - s, 1)`, 0 < s <= 1. s = 0.5 is a 180 degree shutter (the film default: smear = half the motion between two real frames), s = 1 a 360 degree one (the whole motion). Because t = 1 is the real frame, N = 4 costs three synths. The newest sample is the real frame: the picture's front edge is sharp and the blur trails behind it.
- **With interpolation (M outputs per real pair).** Output m = 1..M of a pair is at `t = m / M` (m = M is the real frame `curr`, which is why the windows end at t and not centre on it: a centred window would need the next pair). With the shutter `s` (<= 1) measured in **output** intervals, the window is `[t - s/M, t]`, i.e. `recordSynthBlur(prev, curr, out, (m - s) / M, m / M)`. It never leaves [0, 1], keeps the same length for every output and, for s = 1, tiles the pair exactly. Example, 60 fps to 280 Hz (M = 4.67, so the host's cadence decides each output's t; the same formula with that t): s = 1 smears each output over 3.6 ms of motion, what a 280 fps camera with a 360 degree shutter records. That is the setting that hides the stepping of interpolation yet stays sharp.
- **Longer "film look" blur** (shutter measured in *real* frame intervals, `L = s_real` instead of `s/M`): window `[t - L, t]` slid into the pair, `a = clamp(t - L, 0, 1 - L)`, `[a, a + L]`. A window longer than the output interval overlaps its neighbours (and for `t < L` looks slightly ahead of `t`); the blur is as long as a 60 fps camera's and so heavy, and costs the same N sub-frames per output but needs a larger N for the same smoothness (the samples are further apart in pixels).

How many samples: the sub-frames are `L x v / (N - 1)` pixels apart for an object moving `v` px per real frame, and the blur is smooth when that is at most about 2-3 px; fewer samples show a stack of discrete ghost copies instead of a ramp (N = 2 is two ghosts; use it only for a tiny L). Pick N from the longest smear: e.g. v = 16 px/frame and s = 0.5 real frame -> 8 px smear -> N = 4-5.

### Cost
RX 7900 XTX (RADV), back-to-back, minimum of 21 submissions per cell (`blurtest bench`; GPU shared with the desktop, so +-5 %). "1 output" = one estimate plus one blurred output with N samples over `[0.25, 0.75]` (all interpolated; a window ending on a real frame is one synth cheaper); "naive" = the same with N `recordSynth` calls into N images plus a blend pass:

| ms per pair | | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 |
|---|---|---|---|---|---|---|---|---|
| 1280x960 Quality | 1 output | 0.26 | 0.29 | 0.32 | 0.35 | 0.38 | 0.41 | 0.44 |
| | naive | 0.27 | 0.30 | 0.35 | 0.39 | 0.43 | 0.46 | 0.51 |
| 1280x960 Performance | 1 output | 0.17 | 0.19 | 0.22 | 0.25 | 0.28 | 0.30 | 0.34 |
| 1920x1080 Quality | 1 output | 0.36 | 0.42 | 0.46 | 0.52 | 0.58 | 0.62 | 0.68 |
| | naive | 0.37 | 0.43 | 0.51 | 0.56 | 0.66 | 0.71 | 0.77 |
| 1920x1080 Performance | 1 output | 0.22 | 0.27 | 0.31 | 0.35 | 0.41 | 0.46 | 0.50 |
| 2560x1440 Quality | 1 output | 0.56 | 0.66 | 0.75 | 0.85 | 0.96 | 1.07 | 1.20 |
| | naive | 0.58 | 0.69 | 0.83 | 0.94 | 1.09 | 1.22 | 1.38 |
| 2560x1440 Performance | 1 output | 0.33 | 0.42 | 0.50 | 0.59 | 0.69 | 0.76 | 0.85 |

Per extra sample (least squares over N): 0.030 ms (1280x960 Quality) / 0.028 (Performance), 0.052 / 0.048 (1080p), 0.105 / 0.087 (1440p), against 0.035 / 0.032, 0.060 / 0.052, 0.099 / 0.092 ms for a plain `recordSynth`: **a sample costs about as much as a generated frame**, only a little less (no per-frame fallback pass, no blend, no N output images or sets), at 1440p Quality about 5 % more. There is no cheaper shortcut that keeps per-t correctness: the lookup, the mismatch and the trust all depend on t. Blur only with N = 4 and a window ending on the real frame: about 0.29 ms per real frame at 1280x960 Quality (0.19 Performance), 0.65 ms at 1440p (0.41).

**The target case, 60 fps game to 280 Hz (4.67 outputs per real pair, 60 pairs/s), 4 samples per output = 18.7 sub-frames per pair**, the total including the estimate and replacing the plain generated frames, measured as 1 pair in 3 with 4 outputs and 2 with 5:

| | per pair | per second (60 pairs) | of one second of GPU | plain frame generation alone | blur adds |
|---|---|---|---|---|---|
| 1280x960 Quality | 0.76 ms | 46 ms | 4.6 % | 0.36 ms (2.2 %) | 0.40 ms = 24 ms/s (2.4 %) |
| 1280x960 Performance | 0.64 ms | 38 ms | 3.8 % | 0.26 ms (1.5 %) | 0.38 ms = 23 ms/s (2.3 %) |
| 1920x1080 Quality | 1.22 ms | 73 ms | 7.3 % | 0.53 ms (3.2 %) | 0.69 ms = 41 ms/s (4.1 %) |
| 2560x1440 Quality | 2.11 ms | 127 ms | 12.7 % | 0.82 ms (4.9 %) | 1.29 ms = 77 ms/s (7.7 %) |
| 2560x1440 Performance | 1.71 ms | 103 ms | 10.2 % | 0.58 ms (3.5 %) | 1.13 ms = 68 ms/s (6.8 %) |

Per 280 Hz output that is 0.16 ms (1280x960) to 0.45 ms (1440p) of a 3.57 ms budget. N = 2 / N = 8 at the same case: 1280x960 Quality 28 / 82 ms/s, 1440p Quality 69 / 247 ms/s. A cheaper variant that is not implemented: with `s = 1` the windows of neighbouring outputs share an end sample (`M (N-1) + 1` distinct sub-frames per pair instead of `M N`, 25 % fewer at N = 4); it would need a batched call that dedupes the sample times.

### Limits
Blur averages interpolation errors, it does not hide them: an artefact in a sub-frame (a ghost, a held region, a smeared occlusion) is spread over the window, softer but wider. It reduces clarity by design (the blur *is* a loss of sharpness in motion, so competitive players will often want it off, or a short shutter), and the library's usual limits apply per sub-frame (very fast motion, repetitive structures, scene cuts and flicks present the nearer real frame, so a blurred output of such a pair is an average of real frames).

## Pacing (`gpu/pacing.h`, namespace `framegen::pacing`)
Which frame to show at which moment, and whether to generate at all, lives in the library, so every host paces the same way and the host keeps only its UI and platform glue. Header-only C++17, no Vulkan, no OS calls, no clock: every time is passed in as nanoseconds on any monotonic base. It decides; `Interpolator` renders. The host never needs to choose a multiplier `N` per frame, detect "the game is already fast enough", or implement an automatic turn-off: all of that is the `Pacer`. Reference host: gamescope-ritz (`steamcompmgr.cpp` is the thin glue that feeds it and applies its decisions). `gpu/pacingtest.cpp` drives it with a fake clock (`meson test -C <build> pacing`, CPU only, milliseconds).

Vocabulary: a **present tick** is one display refresh (a vblank, a frame callback, a timer standing in for either); its time `V` is when that paint's image is shown. A **frame id** is any host-chosen identity of a real frame (a commit counter, a buffer sequence); the same id is the same image, 0 is none.

### What the host feeds
| Call | When | Arguments |
|---|---|---|
| `OnArrival(frameId, arrivalNs, streamKey, format)` | a real frame became usable | `arrivalNs` = the earliest instant it could be shown (its acquire fence signalled), **not** when the loop noticed it and **not** the tick that latches it. `streamKey` identifies the source (window/game); `format` = `FrameFormat{width, height, format}` (any pixel-format encoding). A new key or format is a reset. |
| `OnPaint(inputs, newestId, bPresentTick, bSkippable)` | before every paint | `newestId` = the real frame the host would show unprocessed. `bPresentTick`: this paint was triggered by a present tick (false for a UI/cursor repaint). `bSkippable`: nothing but frame generation asked for it, so a tick that is not due may be dropped. |
| `Discontinuity()` | the real frames cannot take part now (a fade, a menu on top, no frame) | the next frame starts a fresh sequence |
| `Reset()` | frame generation switched off/on, new session | forget everything |
| `StatusDue(now)` / `TakeStatus(inputs, now)` | for a HUD / log, every `kStatusPeriodNs` (250 ms) | returns `Report` |

**Skippable ticks (real-rate mode).** With frame generation off and a lag buffer or blur (and for video files), a present tick with nothing new to show and no new real frame to ingest comes back `Decision::skip` when `bSkippable` is set, exactly like an FG cadence tick: commit nothing, the display keeps the last buffer. A tick that brings a new real frame is never skipped (its ring copy is needed even when the frame is not shown yet). Without it a 100 ms buffer at 30 fps composited on every tick (about 120/s instead of about 60/s). The host's existing handling of `skip` is all it takes.

**Cold start.** Until the interval estimator has measured the game rate, the lag buffer holds at its current value (0 at start) and the history depth stays at its default; neither is sized from a guessed interval.

`Pacer::Inputs`, read fresh at each paint (frame generation, the buffer and blur each have their own fields, see "How the features combine"): `now`, `presentNs` (V; 0 = now; a predicted present time, or an output timer's target), `mode` (`Off`/`Fixed`/`Target`), `multiplier` (2..`kMaxN` = 8), `targetFps` (0 = the display's refresh), `priority` (`LowLatency`/`Smoothness`), `refreshHz`, `pauseAtRefresh`, `canExceedRefresh`, `costGuard`, `estimateMs`/`synthMs`/`costSeq` (the renderer's last GPU timings and a counter that changes with every new measurement; < 0 = none), `rendererOk` (false when the host cannot run the renderer: HDR, 10-bit, ...); `frameGen` (frame generation on; default true), `lagBuffer` (`LagBufferSettings`: the lag-spike buffer, off by default, see "Lag-spike buffer"), `extraDelayNs` (a fixed extra delay on top of it, default 0), `blur` / `blurSamples` / `blurAmount` / `blurRelative` (motion blur; active when `blur && blurAmount > 0 && blurSamples >= 2`).

### What the host applies (`Pacer::Decision`)
| Field | Meaning |
|---|---|
| `skip` | a cadence-skip tick: paint and commit nothing, the display keeps the last buffer, ask for the next tick. Only set when the paint was `bSkippable`; nothing was consumed. |
| `reset` | reset the renderer (forget its previous real frames) **before** this paint |
| `inert` | pass-through: show the real frame, the renderer does nothing at all |
| `prevId`, `currId`, `t0`, `t1` | **the one renderer call**: if `prevId != 0`, `recordSynthBlur(prev, curr, out, t0, t1)` on that pair. `t0 == t1` (blur off) is a plain `recordSynth(t1)`; a window ending at 1 is the real frame `curr` as its last sample. `prevId == 0`: no renderer work, show the real frame `showId`. |
| `t` | the output's content instant in the pair: `1` a real frame, in (0,1) a point between the frames. `t1 == t` except the film look near the start of a pair. |
| `showId` | the real frame to show when `prevId == 0`: the newest, unless the buffer is delaying (then an older frame the host must still hold; `Pacer::HistoryDepth()` says how many) |
| `newestId` | the newest real frame (the renderer copies it into its ring on first sight) |
| `outId` | the output frame's identity: the same `outId` is the same image (reuse the cached output instead of rendering again) |
| `newOutput` | this paint presents a new output frame (count it for presented fps) |
| `repaintNext` | another output is due: paint again at the next tick (or the next `ExtraTimer()` tick) |

`Pacer::ExtraTimer()`: the output interval is shorter than a tick and the display path can show it (`pauseAtRefresh` false and `canExceedRefresh` true), so the host arms its own timer for each output frame and paints on it with `presentNs` = that timer's target (`NextExtraTargetNs(lastTarget, now, OutputMs())`). `PaintTick(...)` is the optional main-loop rule that lets only the right ticks paint while the pacer is planning (pass `Pacer::Planning()` as `bGenerating`: it is also true for real-rate planning; `Generating()` is frame generation alone). Under VRR a loop would otherwise paint at arrival times and bunch the outputs.

Minimal host loop:

```cpp
#include "pacing.h"
using namespace framegen;

pacing::Pacer pacer;

// a real frame is ready (its fence signalled at fenceNs):
pacer.OnArrival(frame.id, fenceNs, window.id, {w, h, fmt});

// every present tick, and on any other repaint request:
pacing::Pacer::Inputs in;
in.now = nowNs;  in.presentNs = predictedPresentNs;       // V
in.frameGen = cfg.fgOn;                                    // frame generation
in.mode = pacing::Mode::Fixed;  in.multiplier = 3;
in.priority = pacing::Priority::LowLatency;  in.refreshHz = 144.0;
in.pauseAtRefresh = true;  in.canExceedRefresh = false;    // host capability
in.lagBuffer.enabled = cfg.lagOn;  in.lagBuffer.maxBufferMs = 50;     // lag-spike buffer (see "Lag-spike buffer")
in.lagBuffer.lookbackSec = 300;  in.lagBuffer.testMode = pacing::LagTestMode::Off;
in.extraDelayNs = 0;                                       // fixed extra delay (0 = none)
in.blur = cfg.blurOn;  in.blurSamples = 4;                 // motion blur
in.blurAmount = 0.5;   in.blurRelative = pacing::BlurRelative::ShownFrame;
in.costGuard = false;
in.estimateMs = fgTimings.estimateMs;  in.synthMs = fgTimings.synthMs;  in.costSeq = fgTimings.seq;
in.rendererOk = fg.initialised();

const pacing::Pacer::Decision d = pacer.OnPaint(in, newestFrame.id, isPresentTick, !uiAskedForThisPaint);
if (d.skip) { requestNextTick(); return; }
if (d.reset) host.resetRenderer();                         // forget previous real frames
if (d.prevId == 0) present(frameById(d.showId));           // a real frame as it is (inert or not)
else {
    if (pairChanged(d.prevId, d.currId)) fg.recordEstimate(cb, view(d.prevId), view(d.currId));   // once per pair
    if (!cached(d.outId)) fg.recordSynthBlur(cb, view(d.prevId), view(d.currId), out, d.t0, d.t1); // the one call
    present(out);
}
if (d.repaintNext) requestNextTick();
if (pacer.StatusDue(nowNs)) show(pacer.TakeStatus(in, nowNs));
```

### Model: per-tick fractional pacing
No `N` is held anywhere. Every output frame is generated for its own moment between two real frames: a real frame carries its arrival time `A`; the output for the tick at `V` shows the **content time** `tau = V - D` (`D` = the delay, per priority), `t = (tau - A0) / (A1 - A0)`; `t >= 1` (or `>= 0.98` of the newest pair) shows the real frame itself, otherwise a synth at `t`, clamped to 0.02..0.98. `tau` is monotonic. "How many generated frames per real frame" falls out of `t` from the latest arrivals, so a game dropping 120 -> 100 fps against a 240 target gets about 2.4 outputs per real frame on the very next pair, which is per-frame accuracy without a hold or a step. `Why` arrival and not latch time: latch times are quantised to the refresh (about +-7 ms for 60 fps on 144 Hz), which wrecks the interval estimate.

**Output interval `o`.** Fixed N: `max(interval / N, 1 / refresh)`. Target: `1 / min(target, refresh)` (0 = the refresh). When `o` is longer than a tick only the ticks where a phase accumulator crosses `o` get a new image (Bresenham), so the average rate is exact and a fixed N simply saturates at the refresh instead of stepping down. `interval` = median of the last 8 arrival intervals, clamped to 4..50 ms, trusted after 4 (warm-up; real frames are copied meanwhile so the first pair has its previous frame).

**Priority.**
- `LowLatency` (default): `D = (smoothed interval - o) + margin`, floored at 0 (and at one `o` while the ratio is >= 1.1). A genuinely early or late real frame (more than `max(2 ms, 15%)` off the smoothed interval) snaps the content time to the new pair. `margin` = the largest ordinary arrival jitter (at most 4 ms) plus the lead by which the paint wakes before its tick (`presentNs - now`, at most 4 ms): 0 for a perfectly steady game. `Why` the smoothed interval and the margin: with `D` from the latest interval alone, ordinary jitter moved `D` from pair to pair, and the real frame of a pair sits exactly where the next one is expected, so any late arrival left a tick with nothing new and lost an output (a 30 fps game on a 120 Hz window presented about 105/s instead of 120). `Why` the floor of one `o` at ratio >= 1.1: a game just under the output rate (212 fps on a 280 Hz panel) has `interval - o` below one `o`, no generated frame fitted between two real ones and the output had fewer frames than the game.
- `Smoothness`: `D = median interval + max(2 ms, spread)` (spread = second-largest minus median, so one hitch is ignored), at most half an interval. The content time advances evenly, the pair played may be one **older** than the newest (about one game frame queued), a late real frame holds at `t = 1` instead of hitching. `Why` the renderer keeps `kHistory` = 3 real frames: with `D >= one interval` the newest frame has usually already arrived while `tau` is inside the pair before it.

### Modes and switches
- **Fixed N / Target fps**: `Inputs::mode`, `multiplier` 2..8, `targetFps` (0 = refresh). `Inputs::frameGen` false is frame generation off: the cadence becomes one output per real frame (see "How the features combine"); `mode`, `multiplier`, `targetFps`, `priority` and `pauseAtRefresh` are then unused.
- **Pause at refresh rate** (`pauseAtRefresh`, default true): capped at the refresh, generation stops once the game alone reaches it. False: the cap is gone (Fixed `o = interval / N`, floor 0.5 ms; Target not clamped; pass-through only when the game itself reaches the output rate), and `ExtraTimer()` asks the host to paint on its own timer when `o` is under 98% of a tick (back to the tick timer above 102%). `Why` a switch: whether 800 frames a second is wanted depends on the display path and the GPU budget, which the pacer cannot know; the user can.
- **`canExceedRefresh`** is a **host-provided capability**: true only where a present between ticks is shown or discarded without blocking (a nested Wayland window, a tearing flip). Where it is false "off" behaves exactly as "on" (a plain display cannot show more than its refresh and a second flip while one is pending fails).
- **GPU limit** (`costGuard`, "Limit to GPU speed"): the renderer's estimate plus the synths for one pair may use at most 25% of the game interval; otherwise `o` is raised to `interval / (g_max + 1)`, and if not even one generated frame fits it passes through (`Reason::CostGuard`). A probe retries every 10 s with one generated frame per pair, judged on the next measurement (`costSeq` changes), given up after 3 s. No timings, no guard. False: costs are still read, but the guard never caps, never passes through and never probes, and a cap lifts at once. `Why` it is a switch: the guard trades the chosen output rate for the game's own frame rate, and a steady generated output looks better to some users than switching generation on and off.

### Pass-through rules
Real frame at once, renderer inert, **only** when: frame generation is off and neither blur nor the buffer is on (`Reason::Off`), the game's rate reaches the output rate (interval / `o` below 1.02; generating again above 1.10: a ratio band, not a time hold), the renderer is unavailable (`rendererOk` false; the host keeps driving it so it can re-check), or the cost guard trips. With blur or the buffer on, a game that reaches the output rate is not passed through but planned real-rate (below). VRR and tearing are **not** a pass-through reason. `Why` the old "frame rate times N above the refresh" step-down is gone: it passed through a 144 Hz game at 86 fps with 2x. `Report::reason` (`Reason`) names the cause: `Normal`, `Off`, `WarmingUp`, `CostGuard`, `GameTooFast`, `GameStalled`, `RendererUnavailable`, `WaitingForDecoder` (video mode, see below).

### Resets
Frames are keyed by id, never by texture pointer or object address (a host may reuse one buffer for several frames: pointer equality misses re-commits, and without a reset two unrelated images would be blended). The pacer forgets the previous real frames and the interval history and sets `Decision::reset` on: a new `streamKey`, a new `FrameFormat`, a gap over 100 ms between arrivals, a paint whose `newestId` no announced arrival explains, and `Discontinuity()`. A new stream also forgets the cost-guard state, because another game's cost and rate are unrelated. A changed mode, multiplier, target, priority or cap adopts the new target at once.

### How the features combine
Frame generation, motion blur and the lag-spike buffer are separate and can all be on at once; each owns **one dimension of the same content timeline**, so every combination falls out of one planner instead of special cases:

| Feature | Owns | Input | Effect |
|---|---|---|---|
| Lag-spike buffer | how far behind real time we show | `lagBuffer`, `extraDelayNs` | added to the base delay `D` (`tau = V - D`), sized from the game's own spikes and ramped (see "Lag-spike buffer"); gaps between real frames are filled by synthesising between the pair that brackets `tau`; more real frames are kept (`HistoryDepth()`, following the buffer) |
| Frame generation | which instants get an output (the cadence) | `frameGen`, `mode`, `multiplier`, `targetFps`, `priority`, `pauseAtRefresh`, `canExceedRefresh` | fixed N / target / pause at refresh as above; **off: one output per real frame, at its own time** |
| Motion blur | what each output shows over time | `blur`, `blurSamples`, `blurAmount`, `blurRelative` | the window `[t0, t1]` ending at the output's instant, clamped into the pair that brackets it (`BlurWindow`); off: `t0 == t1` |

Every `Decision` carries the pair, `t0`, `t1`, so the host makes **one** renderer call, `recordSynthBlur(prev, curr, out, t0, t1)`, which is a plain synth when `t0 == t1` and the real frame when the window ends at 1.

| Frame gen | Blur | Buffer | What you get |
|---|---|---|---|
| off | off | off | pass-through, renderer inert (`Reason::Off`) |
| on | off | off | generated frames as above (`prevId`/`currId`, `t0 == t1 == t`) |
| off | on | off | one output per real frame, blurred over `[1 - s, 1]`; no delay |
| off | off | on | real frames at their own time plus `D`; a lag spike inside the buffer is filled with synths |
| on | on | off | every output blurred over its own window (they tile at `s = 1` when the output instants sit on the `m/M` grid) |
| on | off | on | delayed, so a gap is interpolated rather than held |
| off | on | on | real-rate output, blurred, gaps filled |
| on | on | on | all of the above on one timeline |

Shared rules:
- **One estimate per real pair**, shared by every output and every blur sample (the host calls `recordEstimate` when the pair changes).
- **Cost guard** (`costGuard`, off by default): the budget is the estimate plus every synth, so each blurred output costs `blurSamples` synths (a conservative count: a window ending at the real frame is one sample cheaper). Over budget: the output rate is lowered (frame generation), then pass-through (`Reason::CostGuard`), with the same probe.
- **Pause at refresh** is frame generation's cadence only. When the game already reaches the output rate, blur and the buffer still work: the planner runs **real-rate** (`Pacer::RealRate()`, `Reason::GameTooFast` while frame generation is on), one output per real frame.
- **Resets** (new stream, format change, a gap over 100 ms, `Discontinuity()`) reset all three together: the history, the interval estimate, the content time.
- **`Report`** exposes each part for its own settings line: frame generation (`fgActive`, `activeN`, `effectiveMultiplier`, `presentedFps`, `targetFps`), blur (`blurActive`, `blurWindowMs`, `blurSamples`), the buffer (`bufferDelayMs`, `bufferTargetMs`, `lastSpikeMs`, `spikesInWindow`, `outliersIgnored`, `historyFrames`; `delayMs` is the total `D`).

**Real-rate planning** (frame generation off, or the game reaches the output rate and blur / the buffer is on). The output is the real frame whose time has come (the newest with `arrival <= tau = V - D`), at its own time; with `D` = the buffer (+ `extraDelayNs`) alone (the base delay is 0 without generation: priority is a cadence concern). When `tau` passes the planned instant of the next output (the last output plus one smoothed game interval) and no real frame is due, but a later real frame is already known, the planner fills with a synth at that instant, between the pair that brackets it. A real frame that arrives within `max(2 ms, 15%)` after a planned instant is waited for instead of being filled before, so a steady game gets no spurious extra outputs. Without a delay nothing is ever bracketed (every real frame is due the moment it exists): exactly one real output per real frame, no added latency. `Why` this is the same planner and not a second path: the buffer and blur must keep working when frame generation is off or paused, and the content-time rules (`tau` monotonic, one pair per output, resets) are the same.

`Why` the buffer deepens the history: with `D` of 50 ms and 16 ms frames `tau` lies about three frames behind the newest, so the pair that brackets it is older than the last two frames; `HistoryDepth()` follows the delay (below) and the host must keep that many real frames addressable by id. Low latency's "play only the newest pair" rule is lifted while a delay is set, because the delay is the explicit choice to play behind. At the very start, before the first delayed frame is due, the host shows the newest real frame.

### Lag-spike buffer (`Inputs::lagBuffer`, `LagBufferSettings`)
The user's request: *"some games on Linux are sometimes unplayable because of lag spikes ... in case one frame is missing, it can just fully generate between the lag spike ... mainly usable for controller games since it will add ... delay. But it could just turn a game from unplayable into a playable one."* A game that hitches (a frame arrives 1, 2 or 3 intervals late) normally freezes for the hitch. With a small extra delay `D_buf` the late frame has already arrived by the time the content time `tau = V - D` needs it, so the gap is **filled with generated frames** (a real-rate fill, or an interpolated cadence under frame generation) instead of frozen. It is independent of the multiplier (there is no "1x fill gaps only" multiplier value; frame generation off simply plans real-rate), of blur and of the priority; with all three on each still owns its own dimension (the buffer owns delay).

**Settings** (`LagBufferSettings`, read fresh at each paint; off by default, and then the pacer is byte-identical to one without the module):

| Field | Default | Range | Meaning |
|---|---|---|---|
| `enabled` | false | | the switch |
| `lookbackSec` | 300 | 1-600 | how far back spikes count: *"the longer the game has been running, the better it can adjust ... it should take mostly the most recent lag spikes into account. So only like the newest biggest lag spike in the last 5 minutes actually affects or like is visible and everything else is just smoothed out."* |
| `maxBufferMs` | 50 | 0-250 | the cap on the buffer, and the line between a spike and an outlier. *"the user should also be able to set a buffer to something insane like 250 milliseconds, but ... default max should be maybe like 50 milliseconds."* |
| `testMode` | `Off` | `Off`, `ForceMin`, `ForceMax` | pins the **target** at 0 or at `maxBufferMs` (needs `enabled`), so the user can feel each end: *"the test mode ... force[s] the minimum and the maximum buffer"*. Reached through the same ramp as everything else, so the change is visible but smooth (no band, no 2 s wait on the way down). |

*"the starters that I'm talking about should be like 1, 2, maybe 3 missing frames, so it shouldn't go that high."* The default cap therefore covers about 3 frames at 60 fps; a bigger cap is for the user to choose.

**Detection** (`OnArrival`, from the arrival intervals the pacer already tracks). An interval above `kLagSpikeRatio` = 1.75 x the smoothed (median) interval is a **spike**; its size is `interval - smoothed`, in ms, i.e. missing frames x interval (a 2-frame hole at 60 fps: 50 - 16.7 = 33.3 ms). A stall longer than the 100 ms gap reset is counted as a spike too, before the history is dropped; with the buffer on the reset threshold is `max(100 ms, maxBufferMs + 50 ms)`, so a spike the buffer is sized for never resets the pipeline. Spikes sit in a ring of `kLagSpikeMax` = 64 (time, size); when full, the entry contributing least right now is replaced. Detection needs `kMinSamples` = 4 intervals first. A change of game rate (60 -> 30 fps) looks like a few spikes for the 4 frames the median takes to follow it; ordinary jitter (+-4 ms at 60 fps) is below the ratio.

**Judgement.** *"single, like huge lag spikes, for example, when you're compiling and drops down to like 10 FPS, it shouldn't build a 100 millisecond buffer. That would be stupid. So there has to be some kind of judgment."* A spike larger than `maxBufferMs` is an **outlier**: never sized for, only counted (`Report::outliersIgnored`) and shown as `lastSpikeMs`. `Why` ignore rather than clamp: a one-off 100 ms drop says nothing about what the next hitch will be, and clamping it would make every compile stall a 50 ms buffer.

**Recency-weighted target.** Weight of a spike of age `a`: `w = 0.3 + 0.7 x (1 + cos(pi a / L)) / 2` for `a <= L` (`L` = `lookbackSec`), 0 beyond: 1 when new, 0.65 at half the window, 0.3 at its end (smooth, flat at both ends so a spike of the last minute counts in full). `base = max over spikes (size x w)`: the newest big one dominates, older ones fade but are never hard-dropped inside the window (*"for some games lag spikes only happen like every 2 maybe 3 minutes, but there should be also evened out"*: a game that hitches every 150 s with a 300 s window always has one at w >= 0.65). Long-run steadiness: while spikes recur (>= 1 inlier in the window, >= 2 seen this session) `base` is at least `kLagFloorFrac` = 0.5 x the session's EWMA of inlier spike sizes (`kLagEwmaAlpha` = 0.15), so a single quiet stretch does not throw everything away; with no spike in the window at all the target is 0 whatever the history. `target = clamp(base x 1.10 + 1 ms, 0, cap)` (`kLagMarginFrac`, `kLagMarginMs`: a repeat of the same spike fits), where `cap = min(maxBufferMs, (kHistoryMax - kHistory - kHistoryMargin) x interval)` so the history can always hold the delay (240 ms at 250 fps, 250 ms at 240 fps and below). `D_buf >= spike` is what makes a hole fill; in frame generation the base delay comes on top, which only adds margin.

**Smooth changes.** `D_buf` never jumps: it moves toward the target by at most `kLagRampEps` = 3 % of real time per unit time (per paint: at most `0.03 x min(dt, 100 ms)`), i.e. the content time plays at 0.97x while the buffer grows and 1.03x while it shrinks (`tau` stays monotonic; it never goes backwards). A 37.7 ms buffer is reached in 1.26 s, 250 ms takes 8.3 s. Growing starts at once. Shrinking starts only once `D_buf` has been more than `kLagShrinkBandMs` = 3 ms above the target for `kLagShrinkDwellMs` = 2 s (the look-back decay is slow; this keeps a flicker of the target from making the speed wobble) and then goes to `target + 1.5 ms` (to 0 when the target is 0). `Why` a ramp and not a jump: a jump would be a visible hold or skip of `D_buf` ms; 3 % playback speed is invisible, and the same ramp serves the test mode.

**History depth follows the buffer.** *"The amount of frames should be dynamically be determined by the buffer size."* `HistoryDepth()` = `kHistory` (3) with no delay, else `clamp(ceil((base delay + max(D_buf, target) + extraDelayNs) / interval) + kHistoryMargin (2), 3, kHistoryMax)`; `Report::historyFrames` is the same number live. It grows **at once and ahead of the ramp** (it counts the target, not only the live `D_buf`), so the pacer never synthesises from a frame the host has already dropped. It shrinks only after the wanted depth has stayed lower for `kDepthShrinkDwellMs` = 2 s **and** the content time has not left the frames that would go (hysteresis: no freeing and reallocating back and forth). `kHistoryMax` = 65 = ceil(250 ms / 4 ms) + 2: the ceiling covers the largest buffer at the fastest supported game rate (250 fps). Examples: a 2-frame spike at 60 fps: 3 -> 5, decays back to 3; ForceMax 250 ms at 60 fps: 17. With the buffer off the depth is exactly 3 and nothing else changes.

**Host contract** for the buffer: (1) set `Inputs::lagBuffer` per paint, call `OnArrival` for **every** real frame even while the pacer is inert (detection runs in `OnArrival`, also while the buffer is 0 and the plan is a pass-through); (2) keep `HistoryDepth()` real frames addressable by id, and grow your ring **when `Report::historyFrames` or `HistoryDepth()` grows** (check it each paint, it can grow by one per paint during a ramp; it only ever shrinks slowly); (3) with UI protection set `Settings::uiHistory = HistoryDepth()` (see "Deep history"); (4) call `releaseUnusedUi()` (and free your own spare ring slots) only where you already wait for the GPU, after the depth has stayed lower for a while; (5) show `Report::bufferDelayMs`, `bufferTargetMs`, `lastSpikeMs` (the newest spike in the window; 0 = none), `spikesInWindow`, `outliersIgnored` (spikes in the window above `maxBufferMs`), e.g. "buffer 32 ms (last spike 28 ms)".

**Costs.**
- *Latency*: `D_buf` ms on top of whatever the priority adds, by design. It is mainly for games where input latency matters less than smoothness (controller games); the default cap of 50 ms is about 3 frames at 60 fps. At most `maxBufferMs`.
- *Audio*: the host's audio is **not** delayed. Up to about 50 ms nobody sees the lip-sync drift; above roughly 50-80 ms it becomes visible (the 250 ms ForceMax is for feeling the buffer, not for watching a cutscene).
- *Memory, per buffered frame*: the host keeps `HistoryDepth()` real frames (4 B/px each at 8-bit RGBA: 4.9 MB at 1280x960, 14.7 MB at 1440p) and, with UI protection, one clean copy per frame as well (4 B/px, WholeScreen +1 B/px of mask). 250 ms at 60 fps = 17 frames: 83.6 MB of host frames at 1280x960 (250.7 MB at 1440p) plus the same again for the clean copies (+ 21 / 63 MB with WholeScreen). The default 50 ms cap (5 frames at 60 fps) is 24.6 / 73.7 MB per set. It scales with the buffer, so it is only paid while the buffer is up.
- *GPU*: the fills are ordinary synths (the cost guard budgets them like any); no work when the buffer is 0.

### Video / timestamps (`Pacer::OnFrame`)
For video the frames carry presentation timestamps (PTS) and the decoder has the next ones ready: nothing has to be guessed from arrival times and no latency is at stake, so the pacer can put **every output at its exact content time**. The flagship case is 24 fps film on a 120 / 144 Hz display: each tick shows `t = (tau - pts_k) / (pts_k+1 - pts_k)` with no clamp, so there is no 3:2-style judder. The default stays arrival timing (`OnArrival`), byte-identical to before; a stream is timestamped from its first `OnFrame` call.

**Host contract**
- **PTS**: announce each decoded frame, in presentation order, with `OnFrame(id, ptsNs, streamKey, format)` instead of `OnArrival`. The origin is the host's; PTS spacing gives the frame interval exactly (the median / 4-50 ms clamp still applies to the *cadence* maths; `t` itself always uses the real PTS spans, but content below 20 fps is out of scope for the cadence).
- **Media clock** (`Inputs::mediaOffsetNs`, `Inputs::playbackRate`, read fresh at each paint): PTS `p` is to appear at `V = mediaOffsetNs + p / playbackRate`. The paint at `V` shows content time `tau = (V - mediaOffsetNs) * playbackRate`. 1.0 normally; a rate change or an A/V trim (`mediaOffsetNs`) takes effect at once. `V` is `presentNs` as always (0 = `now`).
- **Decoder lookahead**: keep at least the next frame announced. Announcing ahead is expected: the pacer holds up to `kVideoLookahead` (8) frames (`OnFrame` returns false when full: announce it again later; `WantsFrame()` / `PendingFrames()`), promotes each when its time comes and reports it in `Decision::newestId` (the renderer copies it there). The id you pass to `OnPaint` is ignored in this mode. If the next frame is missing the pacer **holds the last real frame at `t = 1`** (never extrapolates) and reports `Reason::WaitingForDecoder` (`Pacer::WaitingForDecoder()`); when it arrives the output is the exact `t` again. The same state shows after the last frame of a stream.
- **Pause**: either stop painting, or keep painting and advance `mediaOffsetNs` with `V` so `tau` stays fixed (the same output repeats, no new frame). Re-anchor `mediaOffsetNs` on resume.
- **Seek**: `Discontinuity()` (or `Reset()`), then announce the frames at the new position and set the new `mediaOffsetNs`. The first frame is shown directly (`prevId == 0`, `t == 1`, `reset` set) until the next is announced. A PTS that goes backwards is treated as an implicit seek; an equal PTS is a duplicate and ignored.

**What changes** (`Pacer::Timestamps()`): `D = 0` (`delayMs` is 0); the pacer is **never inert**, because frames are announced ahead and "the newest" is not the due one, so even where it would pass through it schedules: the real frame whose time has come at each tick (`Decision::prevId == 0`, `showId`), a new output (`newOutput`) only when that frame changes. `Reason::GameTooFast` / `Off` / `CostGuard` / `RendererUnavailable` still report why no synth is made. The estimator is trusted after one PTS interval. A synth or real frame within `kPtsSnap` (1e-4) of a frame's time is that frame.

**Unchanged**: frame generation's cadence (Fixed N, Target, "match display" = Target with `targetFps` 0, pause at refresh: no synth when the source reaches the output rate), blur (its window comes from the PTS spacing), the cost guard, `HistoryDepth()` (3).

**Inert**: the lag-spike buffer (`lagBuffer`) and `extraDelayNs` (there are no arrival spikes to size a buffer for, and `D` is 0 on the media clock: use `mediaOffsetNs` to trim A/V sync), the Low latency / Smoothness priorities (they set an arrival-time delay), and gap detection (a PTS jump is the host's `Discontinuity()`).

**Example: 24 fps film on 144 Hz.** `Mode::Target`, `targetFps = 0`, `refreshHz = 144`: a tick every 6.94 ms, a frame every 41.67 ms, so six outputs per frame at `t = 0, 1/6, ... 5/6` (`t = 0` is the real frame itself), each at exactly its tick's content time. At 23.976 fps on 120 Hz the phase slides by 0.005 of a frame per frame and a frame lands on a tick every 200 frames; the content time follows the media clock exactly (no drift over 60 s).

```cpp
pacer.OnFrame(id, ptsNs, streamKey, {w, h, fmt});                    // as the decoder produces them, a few ahead
in.presentNs = vblankNs;  in.mediaOffsetNs = clockOffsetNs;  in.playbackRate = 1.0;
in.mode = pacing::Mode::Target;  in.targetFps = 0;  in.refreshHz = 144.0;
const auto d = pacer.OnPaint(in, 0, /*bPresentTick*/ true);          // d.newestId: copy it into the renderer's ring
// then as for a game: showId when prevId == 0, else recordSynthBlur(prev, curr, out, d.t0, d.t1)
```

### Motion-blur window (`BlurWindow`)
`BlurWindow(t, outputIntervalNs, gameIntervalNs, amount, BlurRelative) -> BlurSpan{t0, t1}` is the arithmetic of the host's motion-blur logic: where inside the current real pair the blur averages. Feed it to `recordSynthBlur(cb, prev, curr, out, t0, t1)` (the interpolation-based motion-blur pass; `pacing.h` does not depend on it). `prev`/`curr` is `Decision::prevId`/`currId` (for a real-frame output with blur: the previous real frame and that frame). The window **always ends at the instant the output shows** (trailing shutter). `Why`: it needs no future frame and so adds no delay; a centred window would need the next pair for the output that is the real frame itself (`t = 1`). It never leaves [0, 1]. `amount` is the shutter, 0..1:
- `ShownFrame`: measured in **output** intervals, window `[t - s * o / g, t]` (`o` the output interval, `g` the game interval). For output `m` of `M` per real pair (`t = m/M`) that is `[(m - s)/M, m/M]`; `s = 1` tiles the pair, `s = 0.5` is a 180 degree shutter. Blur only (no interpolation, `o = g`, `t = 1`): `[1 - s, 1]`. A `t` that is not on `m/M` (a fractional cadence) is cut at 0, never moved.
- `GameFrame`: measured in **real-frame** intervals (the film look), `L = amount`: `[a, a + L]` with `a = clamp(t - L, 0, 1 - L)`, the length kept and the window slid into the pair (neighbouring outputs' windows overlap; the intervals are not used).

`amount <= 0` (or an unusable interval for `ShownFrame`) is a point, `t0 == t1`. The planner calls it for every output (the game interval is the real span of the pair that brackets the output, the output interval is `OutputMs()`, or the game interval in real-rate), so a host normally just reads `Decision::t0`/`t1`; the function is public for hosts that blur on their own.
