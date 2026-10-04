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
- `layers[0].tex` is `gamescope::Rc<CVulkanTexture>`. For a game it is the commit's imported dmabuf (`commit_t::vulkanTex`, `src/commit.h:42`). Its format follows the client, typically `DRM_FORMAT_XRGB8888/ARGB8888` -> `VK_FORMAT_B8G8R8A8_*`, or `ABGR8888` -> `R8G8B8A8_*` (R:241-243). 10-bit formats are possible. **Assume BGRA8 or RGBA8, and possibly 10-bit.** The FG library supports only `B8G8R8A8_UNORM` / `R8G8B8A8_UNORM` views: on a 10-bit layer 0 gamescope must skip FG (or convert to 8-bit first).
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
  VkFormat viewFormat;            // B8G8R8A8_UNORM or R8G8B8A8_UNORM only (never _SRGB, never 10-bit)
  PFN_vkGetPhysicalDeviceMemoryProperties getMemProps;   // REQUIRED: g_device.vk.GetPhysicalDeviceMemoryProperties (H:1166)
  VkPipelineCache pipelineCache;  // optional
  Settings settings;              // tuning knobs, defaults = the approved look (see "Settings" below)
};
class Interpolator {
public:
  bool init(const InitInfo&);     // owns its buffers (DEVICE_LOCAL), descriptor pool, sampler, pipelines
  void destroy();
  bool resize(uint32_t w, uint32_t h, VkFormat viewFormat);   // re-create on size/format change
  static bool validate(const Settings&);
  bool setSettings(const Settings&);   // runtime knobs, any time, applies to later records
  bool reconfigure(const Settings&);   // also flowScale: re-creates buffers (retire rule of resize)
  Settings settings() const;
  void beginFrame();              // once per command buffer, before its first record*() call
  // All three views: UNORM 8-bit, single mip/layer, VK_IMAGE_LAYOUT_GENERAL on entry and exit.
  // prev/curr: SAMPLED usage, made readable by the caller (gamescope: prepareSrcImage + insertBarrier).
  // out: STORAGE usage, caller-owned. Ends with COMPUTE SHADER_WRITE -> COMPUTE SHADER_READ.
  // recordSynth ends with COMPUTE SHADER_WRITE -> COMPUTE SHADER_READ; recordEstimate ends with a COMPUTE -> COMPUTE barrier.
  bool recordObserve(VkCommandBuffer cb, VkImageView frame);                               // once per NEW real frame, only if Settings::uiProtection != Off (see "Static UI protection")
  bool recordEstimate(VkCommandBuffer cb, VkImageView prev, VkImageView curr);            // passes 1-8, once per real pair
  bool recordSynth(VkCommandBuffer cb, VkImageView prev, VkImageView curr, VkImageView out, float t);   // frame at t in (0,1)
  bool record(VkCommandBuffer cb, VkImageView prev, VkImageView curr, VkImageView out, float t = 0.5f); // = estimate + synth
  bool recordSynthBlur(VkCommandBuffer cb, VkImageView prev, VkImageView curr, VkImageView out, float t0, float t1);   // motion blur over [t0, t1] (see "Motion blur")
  static constexpr uint32_t kSetRing = 32, kUiSets = 64, kUiFrames = 3, kMinSize = 16;
};
}
```
- **Return values.** `init()`/`resize()` return false below 16x16, for any format other than BGRA8/RGBA8 UNORM, for a null `getMemProps`, or on any Vulkan allocation failure. The object then holds no size and every `record*()` call returns false. gamescope skips frame generation for that surface (shows layer 0 as is). `record()`/`recordEstimate()`/`recordSynth()` return false and record nothing when the object is not initialised, `t` is outside (0, 1), or the per-submission limit of their ring is hit. `recordSynth()` also returns false when there was no `recordEstimate()` since init/resize, or when its `prev`/`curr` are not the views of the latest `recordEstimate()`. `out` is then untouched, so gamescope must not substitute it as layer 0 for that composite (and for N× must not present that slot).
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
| `uiStillThreshold` | 3.5 | 0-255 (8-bit levels) | A pixel is still if no channel moved more than this since the previous real frame. |
| `uiInpaintReach` | 12 | 1-16 | Inpaint walk length per direction. |
| `uiMaxStillFraction` | 0.75 | (0, 1] | WholeScreen: skip the whole-screen passes of a frame whose still share outside the box exceeds this (1 = never). |

**How to size the box for your crosshair.** The box is always protected at full resolution, in both modes. Make it 2-3x the crosshair's largest extent (the whole reticle including its spread while shooting/moving and its outline), and set `uiBoxOffsetX/Y` if the reticle is not at the frame centre. Too small: the ends of the reticle fall outside the protection (`uitest check` shows this with a 20x20 box on a 35x35 px crosshair: 61 of 305 crosshair pixels wrong). Too big: only costs a few more pixels of work. Size changes at runtime need no GPU wait: the box buffers only ever grow (see "Memory").

**API contract** (full text in `gpu/framegen.h`):
- `recordObserve(cb, frame)`: once per NEW real frame, in order, for every real frame (also those never interpolated), after its pixels are final. Returns true and records nothing when `Off`. The library keys its entries by `frame`, so a host that reuses views (a ring of images) must observe each new content. At least `kUiFrames` = 3 observed frames are kept (LRU), so synthesising the pair before the newest one (gamescope's "Smoothness" priority) still works. `recordEstimate(prev, curr)` and `recordSynth` read the library's clean copies of those frames; `recordSynth` additionally reads the real pixels from the `curr` view you pass, so `curr` must still hold the real frame at synth time (this was already required).
- Implicit observe: a host that never calls `recordObserve` is observed by `recordEstimate` (prev if unseen, curr unless it is the newest observed view). Caveat: new content under the same view handle is not noticed, and a frame never used in a pair is never seen by the counters. Once `recordObserve` was called, estimates do not observe implicitly; a pair of which a view was not observed (just switched on, skipped, evicted) is interpolated unprotected like `Off`.
- `recordSynth` returns false if an observe evicted one of the pair's two entries since its `recordEstimate` (observe two newer frames between them) or re-observed one of the views.
- Descriptor sets: a third ring of `kUiSets` = 64 sets. `recordObserve` takes 1 (Crosshair) or 2 (WholeScreen), each `recordSynth` takes up to 2, an implicit observe in `recordEstimate` takes the same as `recordObserve`. Keep them per `beginFrame()` and per not-yet-retired submission <= 64 (a normal 4x frame needs about 3 + 2x3). Exhaustion returns false and records nothing.
- Retire rule: `resize()`, `destroy()` and `reconfigure()` that changes `flowScale` free all UI resources; same rule as the other buffers (all earlier records retired). `setSettings()` only creates (the first time a mode is switched on: 3 clean images + buffers, WholeScreen: + masks and cell state) and returns false if that allocation fails (settings unchanged). Growing the box allocates a bigger buffer at the next `recordObserve`; the old one goes to a graveyard that is freed by the next `resize`/`destroy`, so no GPU wait is needed. The clean images are always `R8G8B8A8_UNORM` in logical channel order, independent of `viewFormat`, and are moved UNDEFINED -> GENERAL by a one-off barrier at the first `recordObserve`.
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

**Memory** (4 bytes/px clean copy, 3 kept frames; 1440p / 1080p): Crosshair 44.2 + <1 MB / 24.9 + <1 MB (clean copies + box buffers, 333 px box); WholeScreen adds 3 full-frame masks (1 byte/px: 11.1 MB / 6.2 MB) and the cell state (1/4 of the pixels x 4 bytes: 3.7 MB / 2.1 MB).

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
