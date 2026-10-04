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

`Pacer::Inputs`, read fresh at each paint (frame generation, the buffer and blur each have their own fields, see "How the features combine"): `now`, `presentNs` (V; 0 = now; a predicted present time, or an output timer's target), `mode` (`Off`/`Fixed`/`Target`), `multiplier` (2..`kMaxN` = 8), `targetFps` (0 = the display's refresh), `priority` (`LowLatency`/`Smoothness`), `refreshHz`, `pauseAtRefresh`, `canExceedRefresh`, `costGuard`, `estimateMs`/`synthMs`/`costSeq` (the renderer's last GPU timings and a counter that changes with every new measurement; < 0 = none), `rendererOk` (false when the host cannot run the renderer: HDR, 10-bit, ...); `frameGen` (frame generation on; default true), `extraDelayNs` (the lag-spike buffer's delay, default 0), `blur` / `blurSamples` / `blurAmount` / `blurRelative` (motion blur; active when `blur && blurAmount > 0 && blurSamples >= 2`).

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
in.extraDelayNs = buffer.delayNs();                        // lag-spike buffer (0 = none)
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
Real frame at once, renderer inert, **only** when: frame generation is off and neither blur nor the buffer is on (`Reason::Off`), the game's rate reaches the output rate (interval / `o` below 1.02; generating again above 1.10: a ratio band, not a time hold), the renderer is unavailable (`rendererOk` false; the host keeps driving it so it can re-check), or the cost guard trips. With blur or the buffer on, a game that reaches the output rate is not passed through but planned real-rate (below). VRR and tearing are **not** a pass-through reason. `Why` the old "frame rate times N above the refresh" step-down is gone: it passed through a 144 Hz game at 86 fps with 2x. `Report::reason` (`Reason`) names the cause: `Normal`, `Off`, `WarmingUp`, `CostGuard`, `GameTooFast`, `GameStalled`, `RendererUnavailable`.

### Resets
Frames are keyed by id, never by texture pointer or object address (a host may reuse one buffer for several frames: pointer equality misses re-commits, and without a reset two unrelated images would be blended). The pacer forgets the previous real frames and the interval history and sets `Decision::reset` on: a new `streamKey`, a new `FrameFormat`, a gap over 100 ms between arrivals, a paint whose `newestId` no announced arrival explains, and `Discontinuity()`. A new stream also forgets the cost-guard state, because another game's cost and rate are unrelated. A changed mode, multiplier, target, priority or cap adopts the new target at once.

### How the features combine
Frame generation, motion blur and the lag-spike buffer are separate and can all be on at once; each owns **one dimension of the same content timeline**, so every combination falls out of one planner instead of special cases:

| Feature | Owns | Input | Effect |
|---|---|---|---|
| Lag-spike buffer | how far behind real time we show | `extraDelayNs` | added to the base delay `D` (`tau = V - D`); gaps between real frames are filled by synthesising between the pair that brackets `tau`; more real frames are kept (`HistoryDepth()`) |
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
- **`Report`** exposes each part for its own settings line: frame generation (`fgActive`, `activeN`, `effectiveMultiplier`, `presentedFps`, `targetFps`), blur (`blurActive`, `blurWindowMs`, `blurSamples`), the buffer (`bufferDelayMs`; `delayMs` is the total `D`).

**Real-rate planning** (frame generation off, or the game reaches the output rate and blur / the buffer is on). The output is the real frame whose time has come (the newest with `arrival <= tau = V - D`), at its own time; with `D = extraDelayNs` alone (the base delay is 0 without generation: priority is a cadence concern). When `tau` passes the planned instant of the next output (the last output plus one smoothed game interval) and no real frame is due, but a later real frame is already known, the planner fills with a synth at that instant, between the pair that brackets it. A real frame that arrives within `max(2 ms, 15%)` after a planned instant is waited for instead of being filled before, so a steady game gets no spurious extra outputs. Without a delay nothing is ever bracketed (every real frame is due the moment it exists): exactly one real output per real frame, no added latency. `Why` this is the same planner and not a second path: the buffer and blur must keep working when frame generation is off or paused, and the content-time rules (`tau` monotonic, one pair per output, resets) are the same.

`Why` the buffer deepens the history: with `D` of 50 ms and 16 ms frames `tau` lies about three frames behind the newest, so the pair that brackets it is older than the last two frames; `HistoryDepth()` grows with `extraDelayNs` (at most `kHistoryMax` = 16) and the host must keep that many real frames addressable by id. Low latency's "play only the newest pair" rule is lifted while a delay is set, because the delay is the explicit choice to play behind.

What is not here: the lag-spike **detection** (deciding `extraDelayNs` from the frame times, growing and shrinking it) is a later change. This header has the hook it will drive; a host can already set `extraDelayNs` by hand. A change of `extraDelayNs` is followed smoothly (the content time never goes backwards, a larger delay holds the last output until `tau` catches up). At the very start, before the first delayed frame is due, the host shows the newest real frame.

### Motion-blur window (`BlurWindow`)
`BlurWindow(t, outputIntervalNs, gameIntervalNs, amount, BlurRelative) -> BlurSpan{t0, t1}` is the arithmetic of the host's motion-blur logic: where inside the current real pair the blur averages. Feed it to `recordSynthBlur(cb, prev, curr, out, t0, t1)` (the interpolation-based motion-blur pass; `pacing.h` does not depend on it). `prev`/`curr` is `Decision::prevId`/`currId` (for a real-frame output with blur: the previous real frame and that frame). The window **always ends at the instant the output shows** (trailing shutter). `Why`: it needs no future frame and so adds no delay; a centred window would need the next pair for the output that is the real frame itself (`t = 1`). It never leaves [0, 1]. `amount` is the shutter, 0..1:
- `ShownFrame`: measured in **output** intervals, window `[t - s * o / g, t]` (`o` the output interval, `g` the game interval). For output `m` of `M` per real pair (`t = m/M`) that is `[(m - s)/M, m/M]`; `s = 1` tiles the pair, `s = 0.5` is a 180 degree shutter. Blur only (no interpolation, `o = g`, `t = 1`): `[1 - s, 1]`. A `t` that is not on `m/M` (a fractional cadence) is cut at 0, never moved.
- `GameFrame`: measured in **real-frame** intervals (the film look), `L = amount`: `[a, a + L]` with `a = clamp(t - L, 0, 1 - L)`, the length kept and the window slid into the pair (neighbouring outputs' windows overlap; the intervals are not used).

`amount <= 0` (or an unusable interval for `ShownFrame`) is a point, `t0 == t1`. The planner calls it for every output (the game interval is the real span of the pair that brackets the output, the output interval is `OutputMs()`, or the game interval in real-rate), so a host normally just reads `Decision::t0`/`t1`; the function is public for hosts that blur on their own.
