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
- One queue: all `record()`s of one `Interpolator` must be submitted to one queue, in recording order. Its internal buffers are shared between records and ordered only by pipeline barriers. gamescope's single composite queue satisfies this.
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
};
class Interpolator {
public:
  bool init(const InitInfo&);     // owns its buffers (DEVICE_LOCAL), descriptor pool, sampler, pipelines
  void destroy();
  bool resize(uint32_t w, uint32_t h, VkFormat viewFormat);   // re-create on size/format change
  void beginFrame();              // once per command buffer, before its first record()
  // All three views: UNORM 8-bit, single mip/layer, VK_IMAGE_LAYOUT_GENERAL on entry and exit.
  // prev/curr: SAMPLED usage, made readable by the caller (gamescope: prepareSrcImage + insertBarrier).
  // out: STORAGE usage, caller-owned. Ends with COMPUTE SHADER_WRITE -> COMPUTE SHADER_READ.
  bool record(VkCommandBuffer cb, VkImageView prev, VkImageView curr, VkImageView out, float t = 0.5f);
  static constexpr uint32_t kSetRing = 32, kMinSize = 16;
};
}
```
- **Return values.** `init()`/`resize()` return false below 16x16, for any format other than BGRA8/RGBA8 UNORM, for a null `getMemProps`, or on any Vulkan allocation failure. The object then holds no size and `record()` returns false. gamescope skips frame generation for that surface (shows layer 0 as is). `record()` returns false and records nothing when the object is not initialised or the per-submission limit is hit. `out` is then untouched, so gamescope must not substitute it as layer 0 for that composite.
- **Descriptor ring.** Each record takes the next of `kSetRing` = 32 descriptor sets. At most 32 records may follow one `beginFrame()`; the 33rd returns false (and asserts in debug builds), so overflow cannot silently corrupt an earlier record's set. Across submissions the caller must keep the records of all not-yet-retired submissions at <= 32 in total. gamescope's 1-2 composites per refresh are far below that. Call `beginFrame()` once per command buffer, before FG's first record in it.
- **Lifetime.** `resize()`, `destroy()` and `init()` on a live object free buffers that earlier records still use. The caller must make sure every previous record has retired first: wait on the composite timeline point, or `vkQueueWaitIdle`/`vkDeviceWaitIdle` on a resolution or format change (rare). There is no deferred free.
- `t`: only 0.5 is implemented (midpoint interpolator), other values are ignored.
- The output is caller-owned: a pooled `CVulkanTexture` like `update_effects_image` (sampled+storage, BGRA8/RGBA8 UNORM matching the 8-bit input) passed as `out->srgbView()`. That lets gamescope substitute it with `SubstituteLayer0()` and keep refcount and barrier tracking.
- The library may allocate its own memory and descriptor pool. It must not submit, must not touch queues, and must not assume RGBA channel order (swizzle is irrelevant for luma if the weights are symmetric, otherwise take a `bgr` flag). It must not require features beyond the list above, and should avoid `shaderStorageImageRead/WriteWithoutFormat` unless gamescope enables them. It should declare formats explicitly (`rgba8`, plus `r16f`/`r32f` etc. for its own internal images).
- Descriptors: the library must bind its own pipeline layout and sets. gamescope rebinds its set in every `dispatch()` (~R:1949), so state clobbering is not a problem.
