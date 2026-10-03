# Image-Based Frame Generation: Research Notes + Build Guide

Research date: 2026-09-30. Deep-research run: 23 sources fetched, 112 claims extracted,
25 verified by 3-vote adversarial check (22 confirmed, 3 killed).

Legend used below:
- **[verified-high] / [verified-medium]**: claim survived adversarial verification.
- **[engineering]**: my own implementation guidance, *not* verified by the research run.
  Standard graphics-programming practice; treat as a starting point and test it.

---

## 0. The core idea in one paragraph

A purely image-based frame generator only sees finished frames. It takes frame N-1 and
frame N, estimates per-pixel/per-block motion between them **from the pixels alone**
(optical flow), warps both frames toward the midpoint t=0.5, blends them while handling
regions that are visible in only one frame (occlusion/disocclusion), and presents the result
between the two real frames. Everything else — hooking, pacing, HUD handling — is plumbing
around that.

"Pure image-based" (LSFG, AMD AFMF, NVIDIA Smooth Motion) vs. "engine-assisted" (FSR3-FG,
DLSS-FG, XeFG, OptiFG): the latter additionally get depth + motion vectors from the game,
which makes occlusion and HUD handling much easier.

---

## 1. Lossless Scaling (LSFG) and lsfg-vk

- **[verified-high]** LSFG's algorithm lives in `Lossless.dll` and is a **Direct3D 11 program**:
  a chain of DXBC compute shaders plus host-side dispatch logic.
- **[verified-high]** `lsfg-vk` (Linux) is a **Vulkan layer** that hooks Vulkan apps and runs
  **Lossless Scaling's original shaders**, not a reimplementation. The host pipeline
  (resources, dispatches, synchronization, swapchain hooking) was rewritten in native Vulkan.
- **[verified-high]** How it was reverse-engineered (developer's wiki):
  - Hooked all D3D11 calls in Lossless Scaling to dump the DXBC shaders.
  - Ran shaders through DXVK's DXBC→SPIR-V translator in a test harness; output was
    byte-identical on dumped textures (checked for a few shaders, not all).
  - Dispatch math recovered with IDA + self-written D3D11 command-dump tools
    (RenderDoc/Nsight could not capture Lossless Scaling).
  - Rebuilt the pipeline in D3D11 first, then rewrote it in Vulkan, because D3D11 has no
    explicit sync primitives and Vulkan depends on them entirely.
- **[verified-medium]** Derivatives (GameNative PR #1911 — closed, not merged; LSFG-Android)
  parse the PE of the user's own `Lossless.dll`, extract the DXBC blobs, translate to SPIR-V
  with DXVK's `src/dxbc` frontend, cache by DLL hash (FP16 variants where supported).
  LSFG-Android deletes its DLL copy afterwards, so nothing is redistributed.
- **[verified-medium]** Android 12+ blocks injecting code into non-debuggable apps, so
  LSFG-Android does **screen capture (MediaProjection) + overlay** instead of swapchain hooking.
  Author's *unmeasured* estimate: ~50–80 ms extra latency vs. the Linux Vulkan layer.
- **Status:** lsfg-vk GitHub repo archived → moved to lsfg-vk.dev, license changed to
  CC BY-NC-ND 4.0, v2.0 released (Sep 2026).
- **Important limitation:** nobody has published *what LSFG's shaders actually compute*
  (block matching? small neural net?). lsfg-vk treats them as opaque blobs. Studying its
  source teaches you **hooking and orchestration, not the algorithm**.

## 2. OptiScaler: OptiFG and HUDfix

- **[verified-high]** OptiFG = OptiScaler's experimental, DX12-only injected frame generation
  (since v0.7). **Not purely image-based**: OptiScaler intercepts the game's upscaler calls
  (DLSS2+, FSR2+, XeSS), grabs depth + motion vectors from those inputs, and feeds them to an
  existing FG backend (FSR3-FG, XeFG, or FSR4-FG — the latter auto only on RDNA4 since v0.9).
  Doesn't work with XeSS inputs in Unreal games because the UE XeSS plugin passes no depth.
- **[verified-high]** **HUDfix** (the "hotfix"): OptiFG gets no HUD-less color buffer, so
  interpolating the final image smears the HUD/UI. HUDfix tries to identify the HUD-less
  resource by tracking the upscaler's input/output resources (which exist before UI is drawn).
  Success depends on the game's UI rendering; Unreal games tend to work better.
  FSR3-FG always needs HUDfix; XeFG/FSR4-FG (ML models handle HUD) may not.
- Unsettled (voted down, but some verifiers found supporting wiki text): a tunable
  "HUDfix limit" and a "Show Detected UI" debug option.

## 3. AMD: FSR3 frame generation, FidelityFX Optical Flow, AFMF

- **[verified-high]** **FidelityFX Optical Flow** is the purely image-based part and is
  "based on AMD Fluid Motion Frames technology". Algorithm:
  1. Input: scene color only → converted to **luminance**.
  2. Build a **7-level pyramid** (full res + 6 downscales, generated with SPD).
  3. 7 iterations, coarse → fine; each iteration = 3 compute passes:
     1. **SAD block search**: one vector per **8×8 block**, **24×24 search window**.
     2. **3×3 median filter** on the vector field (outlier removal).
     3. **2× upscale** of vectors, choosing between candidates by SAD → seeds next level.
  4. Output: `R16G16_SINT`, one vector per 8×8 block, texture size `ceil(displaySize/8)`.
     Also produces a **scene-change detection** signal.
- **[verified-high]** FSR3 / 3.1 frame interpolation is a **hybrid**: inputs are 2 back buffers
  + dilated depth + dilated game motion vectors + the optical-flow output. It builds two
  vector fields (game MVs, optical flow), interpolates a color from each, then **soft-blends
  per pixel** weighted by how well each vector set makes prev/current colors match.
  Optical flow wins on shadows, reflections, transparencies (which game MVs miss).
- **[verified-medium]** Occlusion: two **disocclusion masks** (interpolated-vs-previous,
  current-vs-interpolated). Pixels hidden in one source are taken from the other.
- Covers FSR 3/3.1 only. FSR4's ML frame generation likely doesn't use this path.
- **Not verified (no surviving claims):** AFMF 2 / NVIDIA Smooth Motion internals.

---

## 4. Building your own image-based frame generator

This is the main deliverable. Sections 4.1–4.3 follow the verified FidelityFX design;
everything else is **[engineering]**.

### 4.1 Architecture overview

```
 game renders frame N ──► vkQueuePresentKHR (hooked by your layer)
                              │
                              ├─ copy swapchain image N → history texture (curr)
                              ├─ compute: luma + pyramid(curr)
                              ├─ compute: optical flow (prev pyramid ↔ curr pyramid)
                              ├─ compute: scene-change check
                              ├─ compute: warp + blend → interpolated frame (t=0.5)
                              ├─ acquire extra swapchain image, write interpolated frame
                              ├─ present interpolated frame
                              ├─ (pace: wait ~half a frame interval)
                              ├─ present real frame N
                              └─ swap(prev, curr)
```

Interpolation **must** wait for frame N before it can show the frame between N-1 and N.
So it always adds latency: at least ~½ real-frame interval + your compute time. That is the
fundamental cost. (Extrapolation avoids it but artifacts much more.)

### 4.2 Getting frames: three capture options

| Option | Platform | Latency | Difficulty | Notes |
|---|---|---|---|---|
| **Vulkan implicit layer** | Linux (native Vulkan, DXVK/VKD3D-Proton games under Proton) | lowest | medium | What lsfg-vk does. Covers almost all Linux gaming because Proton translates D3D→Vulkan. |
| **DXGI Present hook** (`IDXGISwapChain::Present`) | Windows D3D11/12 | low | high | What OptiScaler does (via proxy DLL like `dxgi.dll`). Anti-cheat may object. |
| **Screen capture + overlay** (DXGI Desktop Duplication / WGC / PipeWire / MediaProjection) | anywhere | high (extra frames of delay) | low–medium | What Lossless Scaling on Windows and LSFG-Android do. No injection, works with anything. |

**Recommendation for this machine (Linux/CachyOS): a Vulkan implicit layer.**

#### Vulkan layer skeleton [engineering]

A layer is a `.so` plus a JSON manifest. Implicit layers load into every Vulkan app when an
env var enables them.

`~/.local/share/vulkan/implicit_layer.d/framegen.json`:
```json
{
  "file_format_version": "1.0.0",
  "layer": {
    "name": "VK_LAYER_framegen",
    "type": "GLOBAL",
    "library_path": "/path/to/libframegen.so",
    "api_version": "1.3.0",
    "implementation_version": "1",
    "description": "image-based frame generation",
    "enable_environment": { "ENABLE_FRAMEGEN": "1" },
    "disable_environment": { "DISABLE_FRAMEGEN": "1" }
  }
}
```

Functions to intercept (resolve the next layer's pointers via the chain info in
`vkCreateInstance` / `vkCreateDevice`):

| Hook | What to do |
|---|---|
| `vkGetInstanceProcAddr` / `vkGetDeviceProcAddr` | return your hooks, forward everything else |
| `vkCreateInstance` / `vkCreateDevice` | set up dispatch tables; create your own compute queue/command pools; enable needed extensions |
| `vkCreateSwapchainKHR` | add `VK_IMAGE_USAGE_TRANSFER_SRC_BIT \| TRANSFER_DST_BIT` (or STORAGE), **bump `minImageCount`** so you can hold an extra image; allocate your history/pyramid/flow textures at swapchain size |
| `vkGetSwapchainImagesKHR` | record the real images |
| `vkQueuePresentKHR` | the heart: copy, dispatch compute, acquire an image, present generated, present real |
| `vkDestroySwapchainKHR` | free your resources |

Present-time sequence [engineering]:
1. Wait on the app's `pWaitSemaphores` (chain them into your command buffer submit).
2. Copy the presented image → `curr` texture (convert to a storage-friendly format, e.g.
   `R8G8B8A8_UNORM` or `R16G16B16A16_SFLOAT` for HDR).
3. Record + submit the compute passes (4.3–4.5).
4. `vkAcquireNextImageKHR` for an extra image, blit the interpolated frame into it, present it.
5. Present the original image, waiting on a semaphore signaled after your copy finished.
6. Swap `prev`/`curr` (ping-pong; keep the previous pyramid too, don't rebuild it).

Study `lsfg-vk`'s layer code for exactly this; its wiki page
"Injecting frames into Vulkan apps" describes the approach.

### 4.3 Motion estimation: hierarchical block matching (the FidelityFX recipe)

**Pass A: luminance.** One thread per pixel:
```glsl
float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
```
Store as `R8_UNORM` / `R16_SFLOAT`. Working in luma is ~3× cheaper and robust enough.

**Pass B: pyramid.** 7 levels total (full + 6 halvings). FidelityFX uses SPD (single-pass
downsampler); a naive "one 2×2-average dispatch per level" works fine to start.

**Pass C: per level, coarse → fine** (7 iterations × 3 passes):

1. **SAD block search.** One workgroup per 8×8 block of `curr`. For each candidate offset
   `d` in the search window (±12 px → 24×24 window) around the seed vector
   (from the coarser level, ×2; zero at the coarsest level):
   ```
   SAD(d) = Σ_{p in block} | curr[p] - prev[p + d] |
   ```
   Keep the `d` with the lowest SAD. Load the block + search area into **shared memory**
   first — this pass dominates cost.
   Tip: a small penalty `λ·|d - seed|` in the cost keeps flat/low-texture regions from
   producing random vectors. [engineering]
2. **3×3 median filter** over the block-vector field (component-wise median, or pick the
   neighbour vector minimizing summed distance to the others). Kills outliers.
3. **Upscale ×2 with candidate selection.** For each finer-level block, test a few
   candidates (the parent's vector ×2, neighbouring parents' vectors ×2) and keep the one
   with lowest SAD at the finer level. This fixes blocky errors along motion edges.

Output: `ivec2` (or `vec2`) per 8×8 block, `R16G16_SINT`, at `ceil(size/8)`.

**Scene-change detection** [verified that FidelityFX outputs one; method is engineering]:
if the average best-SAD over the frame (or a histogram difference of the coarsest level)
exceeds a threshold → skip interpolation this frame, just duplicate/present the real frame.

**Make it bidirectional** [engineering, strongly recommended]: also compute flow
`prev → curr`. Forward/backward consistency (`v_fw(p) + v_bw(p + v_fw(p)) ≈ 0`) is your
cheapest occlusion detector when you have no depth.

### 4.4 Interpolation: warp + blend

Convention: `m(p)` = motion of the content at pixel `p` of frame N, pointing from where it
was in N-1 to where it is in N (so content at `p` in N was at `p - m(p)` in N-1).
At `t = 0.5`, content at `q` was at `q - 0.5·m` in N-1 and at `q + 0.5·m` in N.

**Simple version (backward sampling, start here)** [engineering]:
```glsl
vec2 m   = sampleFlowBilinear(q);              // block vectors, filtered to pixel res
vec3 a   = texture(prevColor, (q - 0.5*m) / size).rgb;
vec3 b   = texture(currColor, (q + 0.5*m) / size).rgb;
float e  = length(a - b);                      // mismatch = likely occlusion / bad vector
out = (e < thr) ? mix(a, b, 0.5) : chooseSide(q, a, b);
```
Assumes `m(q) ≈ m` at the in-between position — fine for smooth motion, breaks at
object edges (you'll see "halos").

**Better version (forward splatting, like FSR3 does with its vectors)** [engineering]:
1. For each block/pixel of frame N, compute its position at t=0.5: `p - 0.5·m(p)`.
2. Splat the vector into an "interpolated-frame vector buffer" with `imageAtomicMin`,
   keyed by a priority (e.g. packed `SAD | vector`) so that on collisions one wins
   consistently. Without depth you can't know which object is in front; lower SAD /
   larger motion as foreground is a usable heuristic.
3. Holes (no vector landed) = **disocclusion**. Fill from neighbours, then sample only
   from the frame where the content is visible.
4. Sample prev and curr with the splatted vector, blend.

**Occlusion handling** (verified concept from FSR3; the heuristics are engineering):
- Build two masks, matching FidelityFX: *interpolated vs. prev* and *curr vs. interpolated*.
  With flow only: a pixel is "not visible in prev" where the backward vector fails the
  consistency check or the prev sample mismatches strongly.
- Per pixel: `w_prev`, `w_curr` from the masks → `out = (w_prev·a + w_curr·b)/(w_prev+w_curr)`.
- Soft weights (not binary) avoid flicker.

**Artifact reducers** [engineering]:
- When confidence is low everywhere, fall back to a plain 50/50 blend or skip the frame.
  Bad interpolation looks worse than no interpolation.
- Clamp the output to the min/max of the 3×3 neighbourhood of `a` and `b` (the same trick
  TAA uses) to hide garbage from wrong vectors.
- Large motion beyond the search range (±12 px × 2⁶ at the top pyramid level is plenty,
  but fast camera pans can still exceed it): detect via high SAD → reduce effect.

### 4.5 HUD / UI handling without engine access

This is the unsolved part: no pure image-based tool gets a HUD-less frame (that's why
OptiScaler's HUDfix exists, and it needs engine hooks). Heuristics [engineering]:
- **Static-pixel lock:** where `|prev - curr|` ≈ 0 **and** the zero vector has the lowest SAD,
  output `curr` directly (stops static HUD from being dragged along with a moving background).
- **Edge-with-zero-motion bias:** sharp, high-contrast, non-moving pixels are usually UI.
- **User-defined exclusion rectangles** (minimap, crosshair) — crude but very effective.
- **Screen border bias:** HUD lives near edges; prefer zero vectors there when SAD is close.

None of these are perfect. Semi-transparent HUD over a moving scene will always artifact.

### 4.6 Frame pacing and latency

[engineering; FidelityFX docs on frame pacing and the frame-interpolation swapchain were
among the fetched sources but produced no verified claims]
- Present the generated frame **halfway** between real frames. With a hook you know the
  real frame's arrival time; estimate the frame interval with a moving average and delay
  the real present by ~interval/2 (busy-wait/sleep on a dedicated thread, or use
  `VK_KHR_present_wait` / `VK_KHR_present_id` / `VK_GOOGLE_display_timing` if available).
- Uneven pacing is **more visible than a lower frame rate**. Use a dedicated present thread
  so the game's thread isn't blocked on your timing.
- Works best when the base frame rate is **stable and capped** (e.g. cap at half the refresh
  rate). Recommend users cap FPS.
- Latency budget: ~½ frame interval (inherent) + compute time + any extra queued image.
  Aim for a compute cost of ~1–2 ms at 1440p (my rough target, not a measured figure).
- Consider **multi-frame** (×3, ×4): same flow, different `t` values (1/3, 2/3, …).
  LSFG supports this; cost is one warp+blend pass per extra frame; flow is reused.

### 4.7 Performance tips [engineering]
- Flow on a **downscaled input** (e.g. half-res luma) is usually good enough; warp at full res.
- FP16 math where supported (lsfg-vk ships FP16 SPIR-V variants for this reason).
- Reuse the previous frame's pyramid; only build the new one.
- Run on an async compute queue so you overlap with the game's next frame.
- Profile with RenderDoc (works for your own layer even if it didn't work on Lossless Scaling)
  or `VK_LAYER_KHRONOS_validation` during development.

### 4.8 ML alternative (not verified in this research)
RIFE, IFRNet, FILM are neural interpolators with much better quality on hard cases
(occlusion, non-rigid motion) but far heavier. RIFE-class models run realtime at 1080p on
strong GPUs; at 4K and in a game's frame budget, classical block matching wins.
FSR4-FG, XeFG, DLSS4-FG have moved to ML. A quantitative RIFE-vs-block-matching comparison at
real-time budgets was **not** established by this research.

### 4.9 Suggested milestones [engineering]
1. Offline prototype: Python/NumPy or a standalone compute program that takes two PNGs and
   outputs the midpoint frame. Iterate on flow + warp here, where you can inspect everything.
2. Port passes to GLSL compute, still offline (load PNGs, dispatch, write PNG).
3. Minimal Vulkan layer that only **duplicates** frames (present the same image twice).
   This proves hooking, extra-image acquisition and sync work. Test with `vkcube`.
4. Plug in the interpolation passes.
5. Pacing thread, scene-change skip, HUD heuristics, multi-frame.
6. Test under Proton with a DXVK/VKD3D game.

### 4.10 Code to study

| Project | Learn | Caveat |
|---|---|---|
| **FidelityFX SDK** — `optical_flow` + `frameinterpolation` shaders, frame-interpolation swapchain | the full, documented algorithm (flow, disocclusion, blending, pacing) | HLSL; hybrid design expects game MVs + depth, but OF part is standalone |
| **lsfg-vk** (lsfg-vk.dev) | Vulkan layer, swapchain hooking, extra-frame injection, sync, pipeline orchestration | algorithm itself is opaque DXBC; CC BY-NC-ND license — study, don't copy |
| **OptiScaler** | D3D12/DXGI hooking, upscaler interception, FG backend injection, HUDfix | engine-assisted, not image-only |
| **obs-optical-flow** | small optical-flow implementation on captured frames | — |
| **RIFE** (arXiv 2011.06294) | ML interpolation reference | heavy |
| **Bilateral ME + adaptive OBMC** (Choi et al.) | classic TV-style motion-compensated interpolation, block artifact removal | academic |

---

## 5. Open questions (not answered by the research)
- What LSFG's shaders actually compute (block matching vs. small neural net; HUD handling).
- AFMF 2 / NVIDIA Smooth Motion internals (hardware optical flow accelerators? ML?).
- Which capture API Lossless Scaling uses on Windows, and its pacing strategy.
- Quantitative ML vs. classical comparison at real-time 1080p–4K budgets.

## 6. Refuted / unsettled claims
- lsfg-vk itself does runtime PE parsing + a statically linked DXVK subset — voted 1-2
  (but some verifiers found wiki support; unsettled).
- GameNative runs LSFG as a host-compositor compute pass — 0-3.
- HUDfix works via a "HUDfix limit" + "Show Detected UI" option — 0-3 (contested).

## 7. Sources
Primary:
- https://github.com/PancakeTAS/lsfg-vk (→ https://lsfg-vk.dev)
- https://github.com/PancakeTAS/lsfg-vk/wiki/Porting-LSFG-to-native-Vulkan
- https://github.com/PancakeTAS/lsfg-vk/wiki/Injecting-frames-into-Vulkan-apps
- https://github.com/utkarshdalal/GameNative/pull/1911
- https://github.com/FrankBarretta/LSFG-Android
- https://github.com/optiscaler/OptiScaler
- https://github.com/optiscaler/OptiScaler/wiki/OptiFG
- https://github.com/optiscaler/OptiScaler/wiki/Frame-Generation-Options
- https://gpuopen.com/manuals/fidelityfx_sdk/techniques/optical-flow/
- https://gpuopen.com/manuals/fidelityfx_sdk/techniques/frame-interpolation/
- https://gpuopen.com/manuals/fidelityfx_sdk/techniques/super-resolution-interpolation/
- https://gpuopen.com/manuals/fsr_sdk/techniques/frame-interpolation-swap-chain/
- https://github.com/azagramac/FidelityFX-SDK-FSR3/blob/master/docs/techniques/frame-pacing.md
- https://github.com/qshiqshi/obs-optical-flow
- https://arxiv.org/abs/2011.06294 (RIFE)
- https://arxiv.org/abs/2205.14620
- https://www.researchgate.net/publication/3309052 (Bilateral ME + adaptive OBMC)

Secondary / blog / forum:
- https://www.gamingonlinux.com/2026/09/lossless-scalings-frame-generation-for-linux-lsfg-vk-2-0-is-out-now/
- https://deepwiki.com/optiscaler/OptiScaler/6-frame-generation
- https://hone.gg/blog/lossless-scaling-the-future-of-performance/
- https://sageinfinity.github.io/docs/FAQ/dxgiwgc
- https://steamcommunity.com/app/993090/discussions/0/4848777451365120792/
- https://raphlinus.github.io/ui/graphics/gpu/2021/10/22/swapchain-frame-pacing.html
