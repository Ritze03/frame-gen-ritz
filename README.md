# frame-gen-ritz

frame-gen-ritz is an image-only frame interpolation library, similar in kind to other image-only frame generators such as AMD AFMF. It sees only finished frames, with no motion vectors or depth from the game. From two real frames it generates the frames in between, for 2x, 3x or 4x the base frame rate. It is a set of Vulkan compute shaders behind a small C++ API, plus the Python/NumPy reference implementation it was ported from.

It was written for gamescope-ritz, a gamescope fork. There it is meant to run as a compute pass on the game's layer-0 image, before scaling and before the overlay layers (Steam overlay, cursor, performance HUD) are composited on top, so those are never interpolated. This repository contains only the library. The pacing decisions (which frame to show when, when to turn generation off) live here too, in `gpu/pacing.h`. The gamescope side is only the glue that feeds the pacer and presents what it decides. `INTEGRATION.md` is the spec for that integration.

frame-gen-ritz is original code. It is not derived from other frame-generation projects or FidelityFX source.

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

## Motion version 2

`Settings::motionVersion` picks the algorithm. `MotionVersion::V1` (the default) is everything described above, frozen: output byte-identical to a library without the choice. `MotionVersion::V2` is a second, "morphing" interpolator next to it: image parts are moved along a smoothed motion field and gaps are filled with stretched content. There is **no local hold** (no per-pixel fallback to a real frame); only the whole-frame scene-cut / flick fallback remains. Single generated frames can look distorted; it is tuned for how it looks in motion. Same latency as V1 (a synth reads only the current pair). The earlier occlusion-based V2 is gone (git tag `v2-occlusion`).

`Settings::v2Mode` (runtime, only used by V2) picks one of two modes:

| Mode | What it does |
|---|---|
| `Quality` (default) | Performance's field, plus an edge-aware per-pixel field, a static-overlay (HUD / subtitle) mask, coverage (which real frame still sees each spot) so the blend leans to the visible side, and blur along the motion where the two frames still disagree. |
| `Performance` | The cleaned block field (re-pick among neighbours + vector median + bilateral smoothing), upsampled, and one full-resolution two-tap gather. |

V2 runs every format V1 runs (Rgba8, Rgb10, RgbaF16 encoded or linear / scRGB, Nv12, P010; HDR included), in both modes, with the static mask and motion blur: its synthesis has a shader variant per format and all its thresholds are in the same 8-bit levels of a perceptually encoded picture as V1's (PQ codes for linear light), see "Formats". All UI-protection modes (including `CrosshairV2`) and motion blur work with both modes. `Settings::v2StaticMask` (default on, V2 only) adds a static-structure mask in both modes: HUD, text and static overlays, including semi-transparent panels and inverted-colour or blended crosshairs that `CrosshairV2` cannot lock on, stay in place; off = no such protection (the tests' unprotected reference). A change takes effect at the next estimate. V2 also tries block-vector candidates where its two taps disagree (fewer pasted / outlier artifacts and ring outlines), and a tap outside the frame takes the other tap. Switching `v2Mode` at runtime keeps the current estimate (no hitch); switching `motionVersion` drops it (the next synth returns false until the next estimate).

Cost, RX 7900 XTX, 1080p, back-to-back median ms per real pair (one estimate + N-1 synths):

| flowScale, mult | V1 | V2 Performance | V2 Quality |
|---|---|---|---|
| 2, 2x | 0.302 | 0.174 | 0.261 |
| 2, 3x | 0.380 | 0.247 | 0.364 |
| 2, 4x | 0.438 | 0.298 | 0.432 |
| 4, 2x | 0.181 | 0.120 | 0.159 |
| 4, 3x | 0.243 | 0.190 | 0.272 |
| 4, 4x | 0.302 | 0.241 | 0.344 |

Before the fix round (no static mask): 1440p 2x flowScale 2: 0.448 / 0.227 / 0.392 ms; 4K 2x flowScale 2: 0.897 / 0.462 / 0.849 ms.

Quality, preliminary (reference clips, every 20th window, PSNR vs V1): Performance +0.46 dB, Quality +0.51 dB. The final judgement is the user's blind comparison.

Other formats cost what Rgba8 does (1080p 2x fs 2, Performance / Quality: Rgb10 0.193 / 0.290, Nv12 0.189 / 0.283, P010 0.191 / 0.287 ms), RgbaF16 a little more (encoded 0.217 / 0.314, linear scRGB 0.256 / 0.362 ms: twice the bytes, and linear light is compared as PQ codes); table per format and flowScale in INTEGRATION.md "Motion version 2".

## Learned motion estimate (testing)

`MotionVersion::Learned` is opt-in. It is V1 with its motion search replaced by ml-fg's learned estimator, a 5-level conv net on the flow-res luma, followed by ml-fg's zero-motion test. It writes V1's per-pixel field and scene-cut SAD, and V1's synthesis, fallbacks, motion blur and UI protection are unchanged.

- `Settings::learnedNet` picks the net: `E2v5` (default, width 24, e2v4 fine-tuned on more varied footage), `E1v5` (the small net, width 12, about 2.4x cheaper), or the older `E2v4` / `E1v4` / `E2v3` / `E1v3`.
- `Settings::learnedZeroTest` (default on) sets a vector to zero where the zero vector matches about as well. This keeps static HUD from being dragged by the motion around it. Off gives the net's raw field.
- `Settings::learnedStaticGuard` (default off, opt-in) is a post-synthesis pass that shows the plain time mix where both frames are locally identical and the motion is small.
- `Settings::learnedHalf` stores the activations as fp16.

It needs `flowScale` 2 and runs every format. Two (near-)identical frames (mean |a - c| ≤ 0.6 levels) skip the net and get a zero field. The weights of all six nets are embedded at build time (`gpu/ml/`, `embed_weights.py`). The WebGL backend has the same estimator and the same setting names (`motion: 'learned'`, `motionNet`, `learnedZeroTest`, `learnedStaticGuard`, `webgl/nets/`).

It costs more than V1: 1080p 2x, back-to-back per pair, e2v5 1.35-1.66 ms (half 1.02-1.28), e1v5 0.53-0.54 ms (half 0.49), against V1's 0.30 ms; e2v5 / e1v5 cost what e2v4 / e1v4 do. The zero test adds ~0.005 ms and the static guard ~0.05 ms per generated frame. `ML-FG-TESTING.md` covers parity with ml-fg's test vectors (`gpu/mltest.cpp`, `gpu/ml/synthcheck.py`, `gpu/ml/guardcheck.py`), memory, the API differences to the WebGL side, the HUD finding and the known issues.

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
| `gpu/shaders/*.comp` | GLSL compute passes (`ui_*.comp`: static-UI protection, `blur_*.comp`: motion blur); `format.glsl` selects the per-format variants of the colour passes |
| `gpu/fgtest.cpp` | Test and benchmark tool for raw RGBA frame files |
| `gpu/uitest.cpp` | Validation and benchmark of static-UI protection on a synthetic scene |
| `gpu/formattest.cpp` | Single-frame checks of every image format (Rgb10, RgbaF16, Nv12, P010) against Rgba8 on a synthetic scene |
| `gpu/blurtest.cpp`, `gpu/blendbench.comp` | Validation and benchmark of the motion blur on a synthetic scene (the shader is the naive blend the benchmark compares against) |
| `gpu/compare.py` | GPU vs Python reference checks, and the benchmark runs |
| `gpu/ml/`, `gpu/shaders/ml_*`, `gpu/mltest.cpp` | `MotionVersion::Learned`: the nets' weights (ml-fg export: e2v3, e1v3, e2v4, e1v4, e2v5, e1v5) and `embed_weights.py`, its passes (incl. `ml_zero`, `ml_guard`), its test against ml-fg's test vectors (`ml/synthcheck.py`: end to end; `ml/guardcheck.py`: the static guard vs ml-fg's `static_guard.py`) |
| `ML-FG-TESTING.md` | QC record of the learned motion estimator (both backends): status, parity, cost, integration notes |
| `INTEGRATION.md` | Integration spec for gamescope-ritz, full API contract, settings, costs |

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

This builds the static library `framegen` (the shaders are embedded as SPIR-V headers) and the test tools. `meson setup build -Dformats=rgb10,nv12` builds only some of the non-Rgba8 shader variants (default: all, see "Formats"). The library needs only the Vulkan headers, because it loads every device function through `vkGetDeviceProcAddr`.

## Formats

The library is not limited to 8-bit RGB games. The format is picked once at `init()` by `InitInfo::viewFormat`:

| Format | `viewFormat` | Frame |
|---|---|---|
| `Rgba8` (default) | `B8G8R8A8_UNORM`, `R8G8B8A8_UNORM` | one view |
| `Rgb10` | `A2B10G10R10_UNORM_PACK32`, `A2R10G10B10_UNORM_PACK32` | one view |
| `RgbaF16` | `R16G16B16A16_SFLOAT` (`InitInfo::transfer`: encoded, or linear / scRGB with `linearWhiteNits`) | one view |
| `Nv12` | `R8_UNORM` (Y plane view; chroma `R8G8_UNORM`) | `Frame{Y view, CbCr view}` |
| `P010` | `R16_UNORM` (Y plane view; chroma `R16G16_UNORM`; 10 bits in the high bits) | `Frame{Y view, CbCr view}` |

- **Both motion versions run every format.** V1 and V2 (both modes, static mask, motion blur, every UI-protection mode), SDR and HDR (Rgb10 / RgbaF16 with PQ, RgbaF16 scRGB with values far above 1.0, P010 BT.2020 PQ): highlights are interpolated, never clipped (`formattest`'s HDR check: a 1000-4000-nit object moved to its mid position within 0.25 PQ levels).
- **Rgba8 is untouched.** Each colour shader is built once per format from the same source (`-DFG_FORMAT_*`, `gpu/shaders/format.glsl`); with no define it compiles to the same SPIR-V as before (checked with `cmp` for all 20 shaders when formats landed; since the CrosshairV2 tap rejection the four warp shaders carry two more buffer bindings, and their *output* is checked by hash instead: Off, `Crosshair` and `WholeScreen` equal `origin/main` bit for bit in every format), keeps its header names, and a host that passes plain `VkImageView`s changes nothing. The other variants are only loaded when chosen. The motion estimate runs on luma and is the same code for every format.
- **Thresholds keep their meaning.** Every level-based setting stays in "8-bit levels": the library converts what it measures to the levels of an 8-bit picture (10-bit codes / 4, PQ codes for linear float, YCbCr differences mapped to R'G'B'), so a number means the same perceived amount in every format. Blending and warping stay in the source's own space.
- **Host contract.** NV12 / P010: one multi-planar image created with `MUTABLE_FORMAT | EXTENDED_USAGE`, plane views (`PLANE_0` as `R8`/`R16`, `PLANE_1` as `R8G8`/`R16G16`); `out`'s two views are both written (Y full, CbCr half resolution; chroma assumed centre-sited). Every format except Rgba8 and RgbaF16 needs the device feature `shaderStorageImageExtendedFormats`. `InitInfo::getFormatProps` lets `init()` check device support and report a reason (`initError()`).
- **Memory** of one UI-protection clean copy at 1440p: Rgba8 / Rgb10 14.7 MB, RgbaF16 29.5 MB, Nv12 5.5 MB, P010 11.1 MB. **Embedded SPIR-V** per non-Rgba8 format: Rgb10 418 KB, RgbaF16 442 KB, Nv12 and P010 490 KB each (of which V2's synthesis 251 / 263 / 276 / 276 KB).
- **Sizes:** from `Interpolator::kMinSize` (16 x 16) up to the device's limits, checked with a reason (`Interpolator::sizeError()`, `initError()`, `settingsError()`): RX 7900 XTX, both versions, every format, up to 16384 x 16384 at flowScale 2 / 4 and 12288 x 12288 at flowScale 1; verified with validation and GPU-AV from 16 x 16 to 7680 x 4320 (INTEGRATION.md "Formats").

Full contract, the threshold table per format, build and gamescope notes: INTEGRATION.md "Formats".

## Static UI protection

A crosshair, a minimap frame or a health bar is static while the picture behind it moves. In a block-based motion search such a pixel shares its block with moving background, so either the crosshair is dragged along or the background around it freezes. `hudBonus` (zero-vector bias) only trades one for the other. Static UI protection removes the problem at the source, per pixel, and belongs in the library so that any host gets it by flipping one setting:

- **Detect** (every real frame, `recordObserve`): a pixel is *still* when no channel moved more than `uiStillThreshold` (3.5 levels) since the previous real frame. After `uiStillFrames` (8) consecutive still frames it is masked; the mask is dilated by 2 px (1 px fully, the second px at half weight) to catch anti-aliased rims. The comparison is always against the previous *original*, never against the inpainted copy.
- **Inpaint**: the library keeps its own "clean" copy of each real frame. Every masked pixel in it is replaced by the 1/d²-weighted average of the first unmasked pixel along each of 8 directions (reach `uiInpaintReach`, 12 px). The motion estimate and the warps run on the clean copies and never see the UI.
- **Patch back**: after each generated frame, `out = mix(out, real pixel, mask)`. Masked pixels therefore show the exact pixels of the real frame.
- There is deliberately no "is this a crosshair" test. Static for N frames *is* the detector. If the camera is still, static background is masked too, which is harmless: it is patched back with the identical real pixels, and any pixel that changes leaves the mask on the very next pair.

Three modes (`Settings::uiProtection`, a runtime setting, `Off` by default and byte-identical to a library without the feature). The first two are described here; `CrosshairV2`, the third, is a different algorithm for the box alone and has its own paragraph below the table:

| Mode | What is protected | How |
|---|---|---|
| `Crosshair` | the **box** (default: a centred square, side 2.5% of the frame height: 24 px at 960, 36 at 1440) | full resolution, every pixel |
| `CrosshairV2` | the box only (no WholeScreen variant), same box settings | contrast-gated detection with hysteresis and fast re-entry, **no inpaint and no clean copy**, a pixel is pasted back only if it did not change in the pair being synthesised: see "CrosshairV2" below |
| `WholeScreen` | the box, plus every solid static HUD element outside it | the box as above; outside it detection at half resolution (2x2 cell means, one counter per cell), the mask is upsampled and dilated at full resolution, and work is skipped for every 32x32 tile that cannot contain a still pixel (GPU-side tile flags); inpaint and patch touch only masked pixels in flagged tiles |

The box is the region where the library **looks for** static UI, always at full resolution because the crosshair is the most noticeable thing if it breaks. What it protects reaches a few pixels past it: a pixel can only be *detected* inside the box, but the mask's dilation (V1: 2 px, a 1 px full-strength ring and a 1 px half-strength feather) or grow (V2: 1 px), the inpaint's source pixels and the paste-back all extend into a margin around it (`Interpolator::boxMargin()`: 2 for `Crosshair`, 1 for `CrosshairV2`, 0 for `Off` and `WholeScreen`; `Interpolator::workRect()` is the box grown by it and clamped to the frame). **Size it for your reticle:** `uiBoxHeightFrac` (default 0.025) sets the side of the square as a fraction of the frame height, `round(height * frac)` clamped to [8, min(w, h)]: 24 px at 960, 27 at 1080, 36 at 1440, 54 at 2160. `uiBoxWidth`/`uiBoxHeight` set it in pixels instead (0 = take that side from the fraction; an explicit size wins), and `uiBoxOffsetX`/`uiBoxOffsetY` move its centre from the frame centre (for a game whose reticle is not dead centre). Rule of thumb: the box has to cover the crosshair's own stable pixels and its spread, **not its whole anti-aliased rim**; a box that just fits the reticle works, even when the rim lies a pixel outside it. Bigger is not safer: every box pixel is a candidate for "still", so a box much larger than the reticle also judges background still, which shows as small white speckles. A box that cuts off more than the margin of the crosshair's solid pixels still leaves those ends unprotected (a pixel 2 px past a V1 box's last still pixel gets only the half-strength feather; V2 pastes nothing 2 px past). A host can draw `boxRect()` and `workRect()` as a preview while the user adjusts a slider. The box is clamped into the frame. `Interpolator::boxSize()` returns the size the library will use. `uiCrosshairArea` is deprecated and used only when `uiBoxHeightFrac` is 0 (the old area box).

`Why:` before the margin, the box was also the rim of the protection: V1's 2 px mask dilation was clipped at the box edge, its inpaint could only take source pixels from unmasked pixels inside the box (a box that just fits the reticle has none, so the crosshair stayed in the clean copy the warps read), and the paste covered only the box. A user followed the UI's advice to make the box "just big enough for the whole crosshair" and set 11 px at 960 high (about 1.1%) around an 8x7 px crosshair: its rim smeared under frame generation, worse with motion blur (which averages the leaked pixels along the motion before the paste), while the old 192 px box had protected it. So the box now means "where to look", and the library adds the margin the algorithm needs itself.

`Why:` the former default, a 3% *area* box, was 192 px on 1280x960. It caught moving background around the reticle, which the user saw as small white pixel artifacts, and its 32 px minimum made a small box impossible. The side is now a plain fraction of the height, which is easy to reason about and to put on a slider.

**CrosshairV2** (`UiProtection::CrosshairV2`, next to `Crosshair`, which is unchanged so the two can be compared). Measured on the user's CS2 footage (two 120 fps OBS recordings, the real crosshair stamped in exactly; `INTEGRATION.md` has the method and the table), V1's visible damage inside the protected area had three causes, and V2 removes each of them; a fourth, found later (the "doubled crosshair"), was in V2 and `Off` as well:

1. *The warp reads the inpainted clean copy* (about 61 % of V1's artifact pixels). A pixel that was inpainted in `prev`'s copy but is real in `curr` (every time the mask shrinks) is warped as 8-direction streaks that move with the picture, the "moving patterns". **V2 has no inpaint and no clean copy**: the estimate and every colour pass read the host's own frames, so the generated frame is bit-identical to `Off` except on the pasted pixels and where the tap rejection of point 4 acts (`uitest check` asserts bit-identity beyond the work rectangle plus the motion reach).
2. *The paste lands on pixels that changed in the pair* (about 37 %): the half-alpha ring and the hard grow paste `curr` onto background that moved. **V2 pastes a pixel only if `max channel |curr - prev| <= uiStillThreshold` for the pair being synthesised** (the *pair gate*), at full strength, never a feather. Without the gate the measured artifacts were about 1.8 px/frame in the CPU model, with it about 0.02.
4. *The warp copies the crosshair* (the **doubled crosshair**; V2's own and `Off`'s). A generated pixel blends a tap of `prev` at `p - 2t*h` with a tap of `curr` at `p + 2(1-t)*h`; where the background pans, a tap that lands on the static crosshair carries it to `crosshair +- motion*t`, a second full-strength crosshair next to the real one. The paste restores the real crosshair at its true place only, so the copy stayed; V1 avoids it by warping inpainted copies. **V2 rejects the tap**: when a warp tap lands on a UI pixel of its own frame (that frame's V2 mask, bilinear footprint; for NV12 / P010 the footprint of the chroma taps) and the other tap does not, both take the other frame's sample; both on UI, or neither, are left alone (`shaders/ui_reject.glsl`, in `mismatch()` and `warpChroma()` and the motion-blur passes; the library binds the masks itself). On a CS2-like 1280x960 scene (an 8x7 px plus over a panning fine-grain background, host settings) the artifact pixels around the crosshair go from 28 to 0 (smooth background: 37 to 26, and 0 with `uiEstimateFill`; hard 3x3 blocks 23 to 1); the crosshair stays pixel-exact. Cost about 0.005 ms per pair at 1280x960.
3. *The mask flickers* (879-3378 px flip per frame with the old 249 px box, about 20 with a 27 px one): that shimmer is what triggers the streaks. V2 only lets a pixel **enter** after `uiStillFrames` (8) consecutive frames that are still **and contrasty** (`uiContrastThreshold`, default 20: max channel `|I - median7x7(I)|`, a crosshair has an edge, flat background does not), only **leaves** after `uiLeaveFrames` (4) consecutive changed frames (hysteresis), and **re-enters** within `uiReentryFrames` (2) frames when it shows the colour it was last confirmed in (the crosshair is back after a flash or a recoil).

The mask is the confirmed pixels plus a hard 1 px grow, binary. `recordTrack` runs the same state machine pass, so a crosshair stays warm through pass-through. A mode that hides the UI from the *motion estimate* as well exists as an option, `uiEstimateFill` (**off by default**): the tap rejection of point 4 does not need it, and on the footage, with the rejection, it nets the same as without it (better minus worse than `Off`: 16.9 against 16.7 px/frame, the mean background error near the crosshair 6.29 against 6.30, `Off` 6.37) while doubling the visible defects (3.6 against 1.7 px/frame worse than `Off`, in 4 of 1173 frames 80-643 of them far from the crosshair), so the default leaves the estimate exactly as `Off` runs it. On synthetic scenes the fill is clearly better, because the crosshair's vote in the flow is what the rejection leaves (smooth background 26 to 0 artifact pixels; mean error near a thin / thick crosshair on a hard-edged moving background 56.9 / 125.2 for `Off`, 32.9 / 70.5 with it): turn it on for a game whose crosshair is thick on a fast, hard-edged background. Trade-offs against V1: V1 also hides the crosshair from the estimate and the warp by inpainting it (so a *thick* crosshair on a hard-edged moving background interpolates better around it with V1 unless V2 runs with `uiEstimateFill`: without it the estimate still sees the crosshair as an occluder, exactly as `Off` does); with the defaults V2's output is bit-identical to `Off` beyond the work rectangle plus the motion reach, and inside it the doubled crosshair is gone. On the footage, 12 segments, 1173 frames (measured before the tap rejection; the mask columns are unchanged by it, see `INTEGRATION.md` for the artifact columns after it: V2 1.7 px/frame worse than `Off` and 18.4 better, because the doubled crosshair is part of `Off`'s error): artifact pixels per frame (worse than `Off` by more than 24 levels and 32 off the truth) V1 4.44 with a 27 px box (20.9 with the old 249 px one), V2 0.012; white specks 1.36 against 0.009; mask flips per frame 16.7 against 2.0 (1669 for 249 px); recall of the crosshair after the first 8 frames 0.917 against 0.977; V2's `observe` costs 0.022 ms at 1440p against V1's 0.066. Limits: opaque, static crosshairs only. A semi-transparent crosshair (its pixels change with the background behind it) and one that changes shape (a spread that follows movement) stay unprotected while they change, by design (a pixel that changed in the pair is never pasted); they come back `uiReentryFrames` / `uiStillFrames` frames after they settle. In 4:2:0 formats (NV12, P010) a few edge pixels of a crosshair share their chroma with the moving background and count as changed. No WholeScreen variant of V2 yet.

`Why:` the tap rejection exists because the user, playing CS2 with Crosshair V2, reported *"major artifacting ... doubled crosshair"*: a second full-strength crosshair about 16 px to the side of the real one, along the camera pan. V2 as first shipped (PR #14) had been judged against `Off` ("no more than `Off`"), and `Off` has the same copies, so no test could see it. The fix lives in the warp, not in the mask or the paste, because the copy is made there. The copy does not come from the estimate, so the fill (`uiEstimateFill`) is not what fixes it, and it stays off by default (see above).

`WholeScreen` bounds its worst case: if more than `uiMaxStillFraction` (75%) of the cells outside the box are still, that frame's whole-screen passes exit on the GPU (a nearly static picture needs no generated frame to differ from the real one). Costs, memory and measurements are in `INTEGRATION.md`. Limits: only *opaque, solid, static* UI is protected. Semi-transparent UI over moving background, UI whose shape changes every frame (a scrolling kill feed, an animated number) and UI that moves (a floating damage number) are not; they behave as before. The cell-mean detection also treats a very slowly moving gradient (under `uiStillThreshold` per frame) as still, which only means it is shown from the real frame.

Host sketch (details in `gpu/framegen.h`):

```cpp
Settings s; s.uiProtection = UiProtection::WholeScreen;
s.uiBoxHeightFrac = 0.03f;                      // optional: box side as a fraction of the frame height
// s.uiBoxWidth = 40; s.uiBoxHeight = 40;       // or explicit pixels (wins over the fraction)
s.uiBoxOffsetX = 0; s.uiBoxOffsetY = -30;       // optional: reticle sits 30 px above the centre
fg.setSettings(s);                              // or Settings::uiProtection in InitInfo

// per NEW real frame (also frames you will not interpolate), after its pixels are written:
fg.recordObserve(cb, frameView);
// per pair, as before:
fg.recordEstimate(cb, prevView, currView);
fg.recordSynth(cb, prevView, currView, outView, t);   // curr must still hold the real frame: it is the patch source
```

## Motion blur

Interpolation-based motion blur, the technique of video tools such as Blur and smoothie-rs: interpolate to a high rate and average groups of sub-frames. In the library it is one primitive, `recordSynthBlur(cb, prev, curr, out, t0, t1)`: **one output image = the weighted average of `Settings::blurSamples` (2-16, any count) synthesised sub-frames at the midpoints of N equal slices of [t0, t1] of the current real pair** (`t_k = t0 + (t1 - t0)(k + 0.5)/N`, a box-filter shutter), with `Even` or `Gaussian` weights (`Settings::blurWeights`). Each sub-frame is exactly what `recordSynth(t)` produces, including every fallback; `t <= 0` / `t >= 1` is the real frame. Off (`blurSamples` 0 or 1) leaves every output byte-identical.

- **Blur only**, no frame generation: show each real frame blurred over the end of the interval that led to it, `recordSynthBlur(prev, curr, out, 1 - s, 1)`. The window ends at the real frame, so it needs no future frame and adds no delay.
- **With frame generation**, M outputs per real pair at `t = m / M`: output m is blurred over the window that *ends* at its time, `recordSynthBlur(prev, curr, out, (m - s) / M, m / M)`, with the shutter `s` (<= 1) in output intervals. Also no delay; `s = 1` is a 360 degree shutter at the output rate.
- **Per-t work is not shared** (lookup, mismatch, trust and the fallbacks depend on t); the estimate is. The library runs all samples of one output in 4 dispatches and never stores a sub-frame, so it takes one descriptor set per output: 8 outputs x 8 samples in one pair fit the 32-set ring. A sample costs about as much as one generated frame (0.03 ms at 1280x960, 0.05 at 1080p, 0.1 at 1440p).
- **Why midpoints:** *"interpolating a 60 FPS video to 240, and then blending the frames together"* (the user's reference: FlowFrames/RIFE x4, then export at 60 fps with frame blending) blends 4 distinct frames per output at 0, 1/4, 1/2 and 3/4 of the frame interval, so every frame is used exactly once. Sampling both window ends (the old rule) counted the real frame at a shared end in two neighbouring outputs when windows tile (100 % blur), a faint double-weighting ghost. The midpoint rule tiles with no shared sample; the positions are shifted half a slice, the proper centred box filter. Cost: no sample lands on a real frame any more, so a window ending at t = 1 costs one more synth than before (every one of the N samples is a synth). `t0 == t1` or blur off is still exactly `recordSynth(t1)`.
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
fgtest [--bgra] [--format F] [--split] [--t T1,T2,..] [--bench N [--mult M]] [--set k=v,..] [--then k=v,..] W H prev.rgba curr.rgba out.rgba [prev curr out ...]
```

- `--bgra`: use `B8G8R8A8_UNORM` images (the file data stays RGBA).
- `--format rgba8|rgb10|f16|f16lin|nv12|p010`: the image format of inputs and outputs; the files stay RGBA8 and are converted on upload and back on readback (rgb10 / f16 / p010 as PQ HDR, an 8-bit value v being 1000 (v/255)^2.4 nits; f16lin the same light as scRGB; nv12 BT.709 SDR; p010 BT.2020 PQ), so every format's output compares with the rgba8 one.
- `--split`: `recordEstimate` + `recordSynth` instead of `record`.
- `--t`: generated times (default 0.5). With several values, the k-th output goes to `out.rgba.<k>`.
- `--bench N --mult M`: benchmark N iterations, one estimate + M-1 synths per pair.
- `--set`: `framegen::Settings` fields by name, plus `algo=v1|v2` (= `motionVersion`), `v2Mode=quality|performance`, `ui=...` (e.g. `--set algo=v2,v2Mode=performance,flowScale=4`). `--then`: a `setSettings` change applied between two records (runtime-change test).

`blurtest` (needs no footage) validates and benchmarks the motion blur on a synthetic scene (panning textured background, a moving object, a static crosshair):

```sh
gpu/build/blurtest check [--bgra] [W H]  # default 1280x720: blur == recordSynth for one instant, == the CPU average of the
                                         # recordSynth sub-frames (N = 2..8), smear of the object's edge vs the ideal model,
                                         # crosshair pixel-exact, descriptor rings, validation; prints FNV hashes of the blur-off output
gpu/build/blurtest bench [W H]           # GPU cost: one synth, N x recordSynth (+ blend), recordSynthBlur for N = 2..8, M outputs per pair
```

`uitest` (needs no footage) validates and benchmarks static-UI protection on a synthetic sequence:

```sh
gpu/build/uitest check [--bgra] [W H]    # default 1280x720: asserts the exact crosshair / HUD pixels, PSNR, validation; the box-margin block (a tiny crosshair in a box that just fits it, V1 and V2, plain and motion-blurred); the CrosshairV2 block: exact crosshair, V2 == Off outside the box,
                                         # shrinking/growing arm ends vs V1 (the control), hysteresis, fast re-entry, the pair gate (also under motion blur), mode switching
gpu/build/uitest check --hash            # also prints a hash per generated frame of the Off run (compare against the
                                         # library before the feature to prove byte-identical output)
gpu/build/uitest deep [W H]              # Settings::uiHistory: a synth from a pair 10 frames behind the newest is still protected (default 640x360)
gpu/build/uitest track [W H]             # recordTrack: UI protection kept warm during pass-through; first synth after the resume is exact (default 640x360)
gpu/build/uitest trackbench [W H]        # GPU cost of recordTrack vs recordObserve
gpu/build/uitest bench [W H]             # per-pass GPU times: moving / still / 70% still scenes, all modes
```

`uitest`, `blurtest` and `formattest` take `--algo v1|v2` and `--mode quality|performance` to run their checks with motion version 2.

`formattest` (needs no footage) checks every image format (Rgb10, RgbaF16 encoded and linear, NV12, P010) on the same kind of synthetic scene, single frames only (640x360 by default, a few milliseconds of GPU per format):

```sh
gpu/build/formattest check [--separate-planes] [--algo v1|v2] [--mode quality|performance] [W H]   # per format: output vs the Rgba8 path, YUV planes written, blur, exact crosshair / HUD, HDR highlights, size limits
                                                       # with UI protection, 10-bit gradient precision, float highlights above 1.0, API refusals
gpu/build/formattest time [W H]                        # one GPU time per format (estimate + one synth), default 1920x1080
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
info.viewFormat = VK_FORMAT_B8G8R8A8_UNORM;            // or R8G8B8A8_UNORM, or another format ("Formats"); never _SRGB
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

- All views are 2D, single mip/layer, in `viewFormat` (8-bit UNORM for Rgba8; two plane views per `Frame` for NV12 / P010), and in `VK_IMAGE_LAYOUT_GENERAL` on entry and exit. `prev`/`curr` need SAMPLED usage and must already be readable. `out` needs STORAGE usage and is owned by the caller.
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
| `uiProtection` | Off | `Off`, `Crosshair`, `WholeScreen` or `CrosshairV2`: static-UI protection (see above). Runtime. |
| `uiBoxHeightFrac` | 0.025 | Box = square of side round(frame height * frac), clamped to [8, min(w, h)]. [0, 1]; 0 = legacy area box. Runtime. |
| `uiBoxWidth`, `uiBoxHeight` | 0, 0 | Size of the full-resolution box in pixels; 0 = take that side from `uiBoxHeightFrac`. Clamped to [4, frame]. Runtime. |
| `uiCrosshairArea` | 0.03 | Deprecated; used only when `uiBoxHeightFrac` is 0: square of side sqrt(area * w * h), clamped to [32, min(w, h)]. (0, 1]. |
| `uiBoxOffsetX`, `uiBoxOffsetY` | 0, 0 | Box centre offset from the frame centre, pixels (+x right, +y down). The box is clamped into the frame. |
| `uiStillFrames`, `uiStillThreshold` | 8, 3.5 | A pixel is protected after this many consecutive real frames in which no channel moved more than this many levels. |
| `uiInpaintReach` | 12 | Walk length (1-16 px) of the inpaint. |
| `uiContrastThreshold` | 20 | CrosshairV2: a pixel can only enter the mask if its contrast `max channel \|I - median7x7(I)\|` is at least this many levels (0-255, 8-bit-equivalent in every format; 0 = no gate). |
| `uiLeaveFrames` | 4 | CrosshairV2: consecutive changed frames before a masked pixel leaves the mask (1-255). |
| `uiReentryFrames` | 2 | CrosshairV2: consecutive frames in its last confirmed colour before a left pixel re-enters at once (0-255, 0 = off). |
| `uiEstimateFill` | false | CrosshairV2: also hide the UI from the motion estimate (luma only; see above: the same net on real footage, clearly better on synthetic scenes; the doubled-crosshair fix does not need it). |
| `uiHistory` | 3 | Observed frames that keep their UI-protection clean copy (3 to 65). Raise it to `Pacer::HistoryDepth()` when the lag-spike buffer plays older pairs. 4 B/px per kept frame. Runtime. |
| `blurSamples` | 0 | Motion blur: sub-frames averaged by `recordSynthBlur()`. 0 or 1 = off, 2-16 on (see above). Runtime. |
| `blurWeights` | Gaussian | `Even` (1/N each) or `Gaussian` (ends of the window at 2 sigma, weights taken at the slice centres). Runtime. |
| `uiMaxStillFraction` | 0.75 | WholeScreen: above this still share (of cells outside the box) the whole-screen passes are skipped for that frame. 1 = never. |
| `motionVersion` | V1 | `V1`, `V2` (see "Motion version 2") or `Learned` (see "Learned motion estimate"; flowScale 2 only); all run every format. Runtime; a switch drops the current estimate. |
| `learnedNet`, `learnedHalf` | E2v5, false | Learned only: the net (`E2v5`, `E1v5` = the small net, `E2v4`, `E1v4`, `E2v3`, `E1v3`) and fp16 activation storage. Runtime; the next estimate uses them. |
| `learnedZeroTest`, `learnedStaticGuard` | true, false | Learned only: ml-fg's zero-motion test on the net's field (next estimate) and the opt-in static guard on the generated frames (next synth). Runtime. |
| `v2Mode` | Quality | V2 only: `Quality` or `Performance`. Runtime; a switch keeps the estimate. |
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

- **Detection**: an interval above 1.3x the smoothed interval and at least 2 ms longer is a spike (1.75x until 2026-10-05; a 14 ms gap at 10 ms counts); its size is `interval - smoothed` ("missing frames x interval"). *"The starters ... should be like 1, 2, maybe 3 missing frames, so it shouldn't go that high."*
- **Judgement**: a spike larger than `maxBufferMs` is an outlier and never sizes the buffer (*"single, like huge lag spikes, for example, when you're compiling and drops down to like 10 FPS, it shouldn't build a 100 millisecond buffer"*); it is only counted.
- **Recency**: target = max over spikes of `size x w(age)`, `w` falling smoothly (cosine) from 1 to 0.3 over `lookbackSec`, 0 beyond; +10 % +1 ms margin; floored at half the long-run mean spike size while spikes recur. The newest big spike dominates, a game that hitches every 2-3 minutes keeps a steady buffer, and one spike fades away after the window (*"the longer the game has been running, the better it can adjust ... only like the newest biggest lag spike in the last 5 minutes actually affects ... and everything else is just smoothed out"*).
- **Settings**: `enabled` (false), `lookbackSec` 300 (1-600), `maxBufferMs` 50 (0-250; *"something insane like 250 milliseconds"* is allowed, the default is 50), `testMode` Off / ForceMin / ForceMax (pins the target at 0 or at the max, through the same ramp, so the user can feel each end).
- **Stall credit**: a spike larger than the delay in use (and within the max) grows the buffer by the difference at once, because the display already stood still for it; the repeat is then bridged (60 ms stalls from nothing: freezes of 58, 42, 29, 0 ms with the ramp alone, 58, 0, 0, 0 with the credit). Spikes above the max are still ignored.
- **Smooth**: otherwise (the margin above the credited size, test modes, a raised max, all shrinking) the buffer ramps at most 3 % of real time (content speed 0.97x-1.03x), and shrinks only after the target has stayed low for 2 s.
- **Frames follow the buffer**: the real frames the host keeps (`HistoryDepth()`, `Report::historyFrames`) and, with UI protection, the clean copies (`Settings::uiHistory`) follow the **current** buffer: `ceil((delay) / interval) + 2`, 3 with no buffer, at most 65. Memory is **per buffered frame**: 4 B/px each, 4.9 MB at 1280x960 and 14.7 MB at 1440p (250 ms at 60 fps = 17 frames: 84 / 251 MB, and the same again for UI-protection clean copies).
- **Costs**: the buffer is real latency (at most `maxBufferMs`), so it suits controller games; the host's audio is not delayed, so above ~50-80 ms lip-sync drifts.
- **Report**: `bufferDelayMs`, `bufferTargetMs`, `lastSpikeMs`, `spikesInWindow`, `outliersIgnored`, `historyFrames`, for a line like "buffer 32 ms (last spike 28 ms)".

`INTEGRATION.md` has the full host contract, every field, a minimal loop and the reasoning behind each rule. `meson test -C <build> pacing` runs the CPU-only tests (`gpu/pacingtest.cpp`).

### Video / timestamps

For video, `Pacer::OnFrame(id, ptsNs, streamKey, format)` replaces `OnArrival`: the frames carry presentation timestamps and the decoder has the next ones ready, so every tick's output is generated at its **exact content time** `t = (tau - pts_k) / (pts_k+1 - pts_k)` with `tau = (V - Inputs::mediaOffsetNs) * Inputs::playbackRate`. 24 fps film on a 144 Hz display becomes six outputs per frame at `t = 0, 1/6 ... 5/6`, with no 3:2 judder; 23.976 fps follows the media clock without drift. Host contract: PTS in order, the media clock (`mediaOffsetNs`, `playbackRate`), keep the next frame announced (a late decoder holds the last frame at `t = 1` and reports `Reason::WaitingForDecoder`), `Discontinuity()` on seek. The lag-spike buffer, `extraDelayNs` and the delay priorities are inert (no arrival spikes, `D = 0`); frame generation's cadence, blur and the cost guard work as usual, and the pacer is never inert in this mode (it schedules the due frame). Details in INTEGRATION.md's "Video / timestamps". The default is unchanged arrival timing.

## Limitations

- Supported formats: 8-bit RGBA/BGRA, 10-bit RGB (A2B10G10R10), half-float RGB (linear/scRGB or encoded), NV12 and P010. Not supported: `_SRGB` views, 4:2:2 / 4:4:4 and 12/16-bit YUV, `VkSamplerYcbcrConversion` sampling (chroma is sampled bilinearly, centre-sited), BT.2020 constant luminance, and full-range P010 is treated as 8-bit-full-range gain (0.3 % off).
- HUD drawn into the game image can still show artifacts, especially translucent HUD elements and scrolling or changing text. Solid, static HUD is handled by the optional static-UI protection (`uiProtection`, above); semi-transparent or shape-changing UI is not protected by it either (`CrosshairV2` leaves a crosshair that changes shape or transparency alone while it changes). Overlays composited later by gamescope are not affected.
- Repetitive structures (stairs, ladders, fences) under fast motion can lock onto the wrong period. Both warps then agree, so the mismatch test cannot catch it.
- Very fast flicks below the whole-frame fallback threshold can show seams between warped and held regions.
- Interpolation adds latency, because a generated frame needs the next real frame.
- At a low base frame rate (3x, 4x) motion between real frames is larger, so fallbacks are more frequent.
- Presentation is the caller's job. `gpu/pacing.h` decides the schedule (which frame at which time), the caller owns the clock, the display and the actual present.

## License

MIT, see `LICENSE`.
