# WebGL2 backend

The frame interpolation of this library (`gpu/`, Vulkan) for a WebGL2 context: GLSL ES 3.00 fragment passes and a small JS host.
It runs V1 (the algorithm of `proto/interp.py`) and V2, the morphing interpolator, in both modes (Performance and Quality), with
V2's static-overlay mask, V2's motion blur and RGBA16F (HDR / float) input. It is meant for pages, where WebGL2 is all there is
(no compute shaders, subgroups or atomics): the Firefox extension browser-frame-gen vendors it instead of keeping its own port, and
exposes the settings in its popup.

Every pass is checked against the Vulkan library on the same frames (`tests/`, "Tests" below): the output differs by at most a few
hundredths of a level on average and the whole-frame decisions are identical.

## Vendoring

Copy `webgl/framegen.js` and `webgl/shaders/` (the whole tree; `__fg.shaderFiles` lists the files the engine needs, as paths
relative to `shaders/`). `framegen.js` is a classic script: it attaches `globalThis.__fg` and nothing else, never fetches, never
touches the DOM or a `<video>`. The caller loads the shader texts (for the extension: through its asset relay) and passes them to
`createEngine`. `LAYOUT.md` and `README.md` are the design docs; `tests/` stays here.

## API

Compatible with the extension's `LAYOUT.md` §6 (the engine it replaces): `createEngine(gl, shaderSources, settings)`, `resize`,
`pushFrame(source)`, `estimate()`, `synth(t, framebuffer, viewport, opts)` → bool, `presentFrame`, `setSettings`, `passCounts`,
`debugRead(name)`, `stats()`, `dispose`, `shaderFiles`, `defaultSettings()`, `checkSupport(gl)`, `levelLayout`. `LAYOUT.md` here is
that contract, updated. Additions (nothing removed):

| Addition | What |
|---|---|
| `__fg.version` | `'2'` |
| settings `v2Mode` | `'quality'` (default, as the library) or `'performance'`. Runtime; a switch keeps the estimate (the other mode's part is derived by its first `synth`, with the same output as that mode's own estimate) |
| settings `blurSamples`, `blurWeights` | V2's motion blur (`synthBlur`): 0 / 1 = off, 2..16 sub-frames; `'gaussian'` (default) or `'even'`. Runtime |
| settings `inputFormat` | `'rgba8'` (default) or `'rgba16f'`: frames, warps and output in RGBA16F. Structural (recompiles the programs, drops the frames) |
| settings `transfer`, `linearWhiteNits` | `rgba16f` only: `'encoded'` (default: display-encoded values, PQ / HLG / sRGB; a level is value × 255) or `'linear'` (scRGB / linear light, measured as PQ codes; 1.0 = `linearWhiteNits`, default 80). Structural |
| `engine.synthBlur(t0, t1, framebuffer, viewport, opts)` | the library's `recordSynthBlur` for V2: the weighted average of `blurSamples` sub-frames at the midpoints of N equal slices of [t0, t1], each exactly `synth(t_k)`'s output; `0 <= t0 <= t1 <= 1`; blur off or t0 == t1: the instant t1. V1 with blur on throws (see "Gaps") |
| `__fg.frameSizeError(gl, w, h, settings)`, `__fg.minSize` | why a frame size cannot run (below 16 × 16, the library's minimum, or a texture above `MAX_TEXTURE_SIZE`), else null. `resize` / `pushFrame` throw a `RangeError` with that text and keep the engine at its old size |
| `__fg.checkSupport(gl, frame)` (changed) | returns `{ ok, missing, sizeError, quality, qualityError }`. `missing`: GL feature names only, as before. `sizeError`: the optional second argument `{width, height, flowScale}` checked by `frameSizeError`, else null (it used to be pushed into `missing`; `ok` is still false when it is set). `quality` / `qualityError`: whether V2 Quality can run here (the MIN-blend probe below) and, if not, why; does not affect `ok` (V1 and V2 Performance do not need it) |
| `engine.support` | the `checkSupport(gl)` result the engine was created with |
| V2 Quality refusal | where `checkSupport(gl).quality` is false, `createEngine` / `setSettings` throw a `RangeError` (`framegen: settings.v2Mode 'quality' is not supported on this context (<qualityError>); use v2Mode 'performance' or motionVersion 1`) for any settings with `motionVersion` 2 and `v2Mode` `'quality'`, and keep the current settings. V1 (whatever `v2Mode` says) and V2 Performance run as before |
| `__fg.assemblePrograms(sources, settings)` | the exact text of every program the host compiles (for validation) |
| `engine.setPassHook(fn)` | profiling only: `fn(passName, 0 / 1)` around every draw |

`pushFrame` takes, besides DOM sources, `{width, height, data}` with `data` a `Uint8Array` (rgba8) or a `Float32Array` / `Uint16Array`
of half floats (rgba16f); a DOM source is uploaded as `HALF_FLOAT` for rgba16f. `debugRead` / `stats` are unchanged in form; the
debug names that changed or are new are listed per stage below. `passCounts().clears` is 0 now (no pass needs a clear).

Required, as before: WebGL2, `EXT_color_buffer_float`, `EXT_float_blend`, `OES_texture_float_linear`, 3 draw buffers, a vertex
texture unit (V2 Quality's splat reads textures in its vertex shader). Blend equation `MIN` (core WebGL2) on R32F, for V2 Quality
only: `checkSupport` probes it (once per context, cached; not while the context is lost): a 1 × 1 R32F framebuffer, three
`uintBitsToFloat` words of the splat's range (>= 2^23) drawn as size-1 points in the order big, small, mid with `blendEquation(MIN)`,
read back; the texel must hold the small word bit for bit, else `quality: false` with `qualityError` `'MIN blend on R32F: kept 0x..,
want 0x..'` (or the GL error / incomplete framebuffer / link failure that stopped it). The probe saves and restores every binding and
state it touches and drains pending GL errors before it runs. Nothing else needs a probe: R32F / RGBA32F rendering and float blending
are the required extensions; vertex texture fetch (16 units), integer textures and a point size of 1 are WebGL2 core minimums (the
splat draws size-1 points only; `ALIASED_POINT_SIZE_RANGE` always contains 1); V2 Performance's blur accumulation uses `FUNC_ADD` on
float targets (`EXT_float_blend`).

## Motion versions

### V1

`proto/interp.py` exactly as the extension ported it (`LAYOUT.md`), with three cheaper but equivalent estimate passes: the search's
argmin and the 3×3 median are separate passes (the fused version recomputed 9 argmins per block), the per-level median histogram
is a gather into a banded cumulative histogram with a binary-search `medsel` (the point scatter with additive blending serialised on
the few popular bins), and the first untrusted-count step reduces 8 × 8 instead of 16 × 16.

### V2 Performance ("grid morph")

The extension's port of `framegen-v2` (`v2_search`, `v2_repick`, `v2_clean`, `v2_static`, `v2_enclose`, `v2_morph`), taken over as
it was. Estimate: the static-structure mask (`v2/static_raw`, `static_s`, `static_cell`), V1's seed / search passes on every level
with `v2/search_pick0` at level 0 (raw argmin + the equiangular sub-pixel fit), the re-pick (`v2/repick_*`, 9 candidates on the
16-px quincunx), two refinements (`v2/refine_*`, 0.5 and 0.25 px), the scene-cut sum (`reduce_sad` over the fixed-point per-block
terms), the clean-up (`v2/clean`: vector median + bilateral) and the enclosure / "overlay near" bits (`v2/enclose`). Synthesis per
t: `v2/em` (the taps' mismatch per pixel), `v2/group` (one fragment per 4 × 4 group: the block-vector candidate choice the library
makes with subgroup shuffles), `v2/morph` (MRT: the sub-frame and its mismatch e), `v2/tile` + `reduce_untrusted` (the flick count:
8 × 8 tiles whose mean e passes the trust midpoint), `v2/final` (the sub-frame, or the nearer real frame on a scene cut / flick).
Debug names: `v2.sRaw v2.s v2.s8 v2.vfit v2.rep0 v2.rep1 v2.rep2 v2.vf v2.ec` (estimate), `v2.em v2.grp v2.morph v2.e v2.tile`
(synth; `v2.e` is now R16F with the integer mismatch, unclamped like the library's, for float input).

Subgroup replacements: the library's `subgroupAny(em > 24)` gate of the block-vector candidates becomes "any pixel of the 8 × 8
tile" (the subgroup on the reference GPU, RDNA3 wave64); the 4 × 4 lane sums are `v2/group`; the 16- and 8-lane clusters of the
re-pick and the enclosure are separate SAD / pick passes and a serial loop per cell. Bilinear frame taps are 4 `texelFetch` at
1/32-px positions (the library uses the hardware filter), the source of the remaining ±1-level differences.

### V2 Quality ("coverage morph") — new

Port of `v2_qfield`, `v2_qown`, `v2_qcov` and `v2_qmorph` (`gpu/shaders/v2_quality.glsl`). Per estimate, after Performance's
clean-up (`v2/q_common.inc.glsl` has the constants):

| Pass | Library | Output (debug name) |
|---|---|---|
| `v2/qfield` | `v2_qfield.comp`: the edge-aware per-pixel field (the 4 block vectors around each flow pixel weighted by how well each matches on its 3 × 3) and the static-overlay mask z | `v2.qf` RGBA32F: xy = half-vector, z = mask |
| `v2/qclaim` | the per-location part of `v2_qown.comp`: key (luma mismatch × 8, rounded, ≤ 4095), quantised vector, footprint pixels in prev / curr | `v2.qclaim` RGBA32I |
| `v2/qown_z` + `v2/qown` (×2, one per frame) | `v2_qown.comp`'s `atomicMin` | `v2.qownA`, `v2.qownC` R32F |
| `v2/qvis` | the per-location part of `v2_qcov.comp`: does each frame still show the location's content | `v2.qvis` |
| `v2/qcov` | `v2_qcov.comp`: the 5 × 5 box, quantised to 11 / 11 / 10 bits exactly as the library packs it | `v2.qcov` RGBA32F: cP, cC, z |

Synthesis: `v2/em`, `v2/group`, `v2/morph` compiled with `V2_QUALITY` (programs `v2_qem`, `v2_qgroup`, `v2_qmorph`): the per-pixel
field instead of the block field, the coverage-weighted blend toward the frame that still shows the content, the blur along the
motion where the taps still disagree (4 more tap pairs over ±¼ of the motion), local structure as disagreement in the zero-vector
test; then V2's tile count and final.

**The coverage splat without atomics.** The library keeps per footprint pixel the smallest 32-bit word `key << 20 | vector`
(`atomicMin`). Here every location draws one point at its footprint into an R32F target with blend equation `MIN`, carrying the word
`(min(key, 255) << 20 | vector) + 2^23` as the float with those bits. Positive normal floats order like their bit patterns and a
MIN blend selects without arithmetic, so the target keeps exactly the smallest word (the 2^23 offset keeps the words above the
denormals, which a GPU may flush). Clamping the key at 255 changes no result: `v2/qvis` only asks whether the winning key is above
`Q_WIN × Q_KEY` = 48 (then the location sees its content whatever the vector) and otherwise compares it with the location's own,
unclamped key; a key above 255 never wins against one of 48 or less. The static-overlay claims (key 0 at the location's own place)
are a full-screen pass that also initialises the target. Checked against the library's coverage words: identical on every pair.

Cost and memory: about 56 bytes per flow pixel for Quality's textures (1080p at flowScale 2: 29 MB), allocated the first time
Quality runs.

### Motion blur (`synthBlur`, V2)

The library's V2 blur is its synthesis with n samples; each sample is exactly `recordSynth(t_k)`'s stored output (8-bit or
half-float, with its own scene-cut / flick fallback), averaged with the weights. Here: per sample the full synthesis into `out`,
`v2/blur_accum` adds it × weight into an RGBA32F accumulator (blend ONE, ONE, sample order as the library's register sum),
`v2/blur_resolve` divides by the weight sum and stores. About 7 draws per sample.

## Frame formats

`inputFormat: 'rgba16f'` follows `gpu/shaders/format.glsl` (RgbaF16): the frames are warped and blended in their own values (never
clamped; the output keeps values above 1.0), while every measurement is in levels of a perceptually encoded 8-bit picture, so the
thresholds (trustLow / High, sceneCutSad, the V2 constants) mean the same as for RGBA8: `transfer: 'encoded'` → value × 255,
`'linear'` → the PQ code of value × `linearWhiteNits` / 10000, × 255. The luma of the motion estimate uses the same weights on those
levels, unrounded. In the shaders (`common.glsl` "frame formats"): `fgPx` / `fgTap` read working values (RGBA8: integer levels,
the tap rounded as cv2's uint8 remap), `FG_D` / `FG_G` are the distance / difference in levels (V2's `V2_D` / `V2_G`), `fgStore`
stores (RGBA8: rounded levels). The host defines `FG_F16`, `FG_XF_LINEAR`, `FG_XF_WHITE`; RGBA8 compiles to the same arithmetic as
before. Memory per full-res pixel doubles for the frame, warp and output textures.

The library's V2 is Rgba8-only on `framegen-v2`; its formats work (`v2-formats` @ e215f83, "V2 for every format") uses the same
conventions as this backend (`V2_LEVELS`, `V2_D`, `V2_G`, `V2_LV`, `V2_QUANT` = the half-float store), and the WebGL2 V2 with rgba16f
matches it (see "Results"). `tests/hdr.mjs` asks whichever library it is built against for RgbaF16 V2 and reports "no Vulkan
reference" while that library refuses.

## Tests

`webgl/tests/run.sh [step ...]` runs everything below in order (`setup validate golden xbackend hdr blur bench`; `xbackend-gpu`,
`bench-gpu` with hardware GL; `clean`). Outputs (raw frames, references, logs) go to `$FG_WORK`, default `<repo>/.local-work/webgl`
(gitignored); `clean` deletes it. Needs node (playwright-core is found in `~/.npm/_npx` or via `$PLAYWRIGHT_CORE`, the Chromium
headless shell in `~/.cache/ms-playwright` or `$FG_CHROMIUM`), python3 with numpy + cv2, ffmpeg (footage pairs), glslangValidator,
meson + ninja + the Vulkan headers (the library's build) and a Vulkan GPU with compute subgroups (the reference). Footage:
`$FG_FOOTAGE` (default `<repo>/footage`, the bench footage); missing files are skipped.

| Step | Files | What |
|---|---|---|
| setup | `vkref.mjs --build-only`, `pairs.py`, `vkref.mjs` | builds fgtest in `gpu/build-webgl`; writes the pairs; runs the library (fgtest) on every pair × mode × flowScale × t |
| validate | `validate.mjs` | glslangValidator on every assembled program, all three format variants |
| golden | `ref_gen.py`, `golden.mjs` (+ `compare.mjs`, `page.js`) | V1 vs `proto/interp.py` with per-stage diagnostics (the extension's golden harness) |
| xbackend | `xbackend.mjs` (+ `server.mjs`, `xpage.js`) | WebGL2 (SwiftShader) vs the library: V1, V2 P, V2 Q; block field, per-pixel field and coverage per stage; switch semantics; API checks; the V2 Quality MIN-blend probe (real result, cached, GL state kept; a context with MIN broken by a Proxy: `quality` false, Quality refused, V1 / V2 P run) |
| hdr | `hdr_pairs.py`, `hdr.mjs`, `vkhdr.cpp` | rgba16f (scRGB linear, PQ-encoded, identity) vs the library in R16G16B16A16_SFLOAT / Rgba8 |
| blur | `blur.mjs`, `vkhdr.cpp --blur` | `synthBlur` vs `recordSynthBlur` (V2 P, V2 Q) |
| bench | `bench.mjs` | ms per pair (`--gpu`: hardware GL, timer queries, `--passes`: per pass) |

**Pairs** (`pairs.py`): the V1 golden scenes (pan, hud_odd, objects, noisy → flick, cut), the V2 scenes of the extension (panel:
HUD text + a semi-transparent text panel, cross: crosshair + HUD, zoom: rotation + zoom, fast: large motion past the search window),
uitest-like scenes (square: a moving square over static texture; hudpan: static HUD text over a pan; ui: pan + 16 moving objects + a
static opaque crosshair + HUD box, 1280 × 720), sizes (odd 645 × 383, tiny 97 × 61), footage (battlefield 1080p, two minecraft120
cuts: furnace HUD and forest sprint), bf_odd (1279 × 719 crop), bf_4k (3840 × 2160) and bf_8k (7680 × 4320, hardware GL only).
flowScale 1, 2 and 4 (4K: 2, 4; 8K: 4; ui and bf_odd: 2); t = 0.5 and 0.25.

**Gates (tolerances)**, per pair × mode × flowScale × t:

| Check | Gate | Why this much |
|---|---|---|
| decisions | scene cut identical; flick identical when there is no cut | the whole-frame fallbacks must never differ |
| fallback output | bit-identical (both show the nearer real frame) | |
| output, no fallback | MAE ≤ 0.5 levels and ≤ 1 % of pixels with a channel > 4 levels off | the bilinear taps (4 texelFetch vs the hardware filter's 8-bit weights) and fp32 order differ: ±1 level on some pixels, rarely a few-level step where a soft weight flips. Measured: worst MAE 0.004 |
| golden (V1 vs interp.py) | MAE ≤ 1 level, decisions identical | the extension's D11 gate |
| hdr vs RgbaF16 | MAE ≤ 0.5 levels (encoded levels), ≤ 1 % > 4, decisions identical | as above |
| hdr identity vs Rgba8 | same gate | the float path does not round taps to levels (RGBA8 does): measured MAE 0.2–0.3, < 0.6 % of pixels > 4 |
| blur | MAE ≤ 0.5, ≤ 1 % > 4 | as above |

Diagnostics that do not gate: block-field / per-pixel-field endpoint error and coverage difference vs the library's debug maps
(`fgtest --dump`), the scene-cut sum and untrusted counts.

### Results (2026-10-22, framegen-v2 @ 4bdf179, RX 7900 XTX RADV, Chromium 156 headless)

| Test | Rows | Result |
|---|---|---|
| validate | 143 shaders (3 format variants) | all OK |
| golden (V1 vs interp.py, 640 × 384, flowScale 2) | 5 pairs × 3 t | PASS, MAE ≤ 0.22 (objects t = 0.5), decisions identical |
| xbackend, SwiftShader | 312 (19 pairs, V1 / V2 P / V2 Q, flowScale 1 / 2 / 4, 2 t) | all PASS; decisions identical everywhere (72 fallbacks); mean MAE ≤ 0.0001, worst MAE 0.0036 (V2 P), worst % > 4 levels 0.04 %, worst single pixel 83 levels (mc_sprint, flowScale 1: a block-vector candidate tie) |
| xbackend, hardware GL (ANGLE on RADV) | 318 (+ 8K at flowScale 4) | all PASS; 270 of 318 bit-identical, every other row max 1 level; block field, per-pixel field exact (≤ 0.001 px) |
| per stage (SwiftShader) | | block field identical on synthetic pairs; on footage 0.15–0.56 % of V2's blocks differ (fp ties in the refinement SADs, mean EPE ≤ 0.0005 px); V1 pixel field: ≤ 0.024 % of flow pixels pick another candidate (ties); Quality's coverage: identical except a few locations at key ties (mean ≤ 3e-4) |
| switch semantics, API | | PASS: a motionVersion switch drops the estimate, a v2Mode switch keeps it and gives the same output as that mode's own estimate (bit-identical), sizes refused with the reason |
| hdr vs the library on `framegen-v2` (V1 RgbaF16) | 8 pairs × {scRGB, PQ} × 2 t | PASS: MAE ≤ 0.05 levels, max ≤ 5.4, decisions identical |
| hdr vs `v2-formats` @ e215f83 (V1 + V2 P + V2 Q RgbaF16) | 144 | all PASS: V2 MAE ≤ 0.001 levels, untrusted counts and scene-cut sums identical |
| hdr identity encoding vs Rgba8 | 48 | PASS: MAE 0.2–0.3 (unrounded float taps vs rounded 8-bit taps), ≤ 0.6 % > 4 |
| blur (V2 P / Q, 4 pairs × 3 windows) | 24 | all PASS: MAE ≤ 0.08, max 1 level |

To compare against another library version: build it out of tree (e.g. `git archive <rev> gpu | tar -x -C <dir>`, `meson setup
<dir>/build <dir>/gpu`, `ninja -C <dir>/build fgtest`) and run `FG_VKTAG=<name> FG_GPU=<dir>/gpu FG_GPU_BUILD=<dir>/build node
webgl/tests/hdr.mjs` (or `blur.mjs`); `vkref.mjs` takes `FGTEST=<dir>/build/fgtest` (use a fresh `$FG_WORK` for its references).

## Performance

1080p (battlefield), 2x (one estimate + one synth per pair), flowScale 2 unless noted. GPU times from
`EXT_disjoint_timer_query_webgl2`; "back-to-back" = 20 estimate + synth in one query (the GPU kept busy, like the library's own
benchmark), "per pair" = one pair with a sync before it. The upload of a raw RGBA frame from JS (`pushFrame` with `{data}`, 8 MB)
costs another ~0.7 ms of GPU time; a `<video>` source is a GPU copy instead.

| | V1 fs1 | V1 fs2 | V1 fs4 | V2 P fs1 | V2 P fs2 | V2 P fs4 | V2 Q fs1 | V2 Q fs2 | V2 Q fs4 |
|---|---|---|---|---|---|---|---|---|---|
| WebGL2, RX 7900 XTX (ANGLE / Vulkan / RADV), back-to-back ms | 1.38 | 0.89 | 0.57 | 1.15 | 0.90 | 0.77 | 1.79 | 1.09 | 0.90 |
| — of which estimate / synth | 0.94 / 0.18 | 0.49 / 0.20 | 0.26 / 0.21 | 0.86 / 0.27 | 0.44 / 0.28 | 0.32 / 0.32 | 1.34 / 0.30 | 0.67 / 0.29 | 0.46 / 0.34 |
| WebGL2, one pair with a sync (estimate + synth) | 2.50 | 1.34 | 0.93 | 2.32 | 1.52 | 1.20 | 3.52 | 1.92 | 1.36 |
| Vulkan library (fgtest --bench, incl. luma), back-to-back | | 0.31 | | | 0.20 | | | 0.28 | |
| SwiftShader (CPU), wall ms per pair | | 800 | | | 990 | | | 1730 | |

Where the time goes: ~40–50 draws per pair, each its own render pass in ANGLE. With a timer query around every draw (which adds
its own overhead) no pass stands out any more after the cheap wins (the fused argmin + median was 0.6 ms alone; the histogram
scatter, the serial median scan and the coverage splat's 4 point draws were the next): the rest is many small passes of
0.01–0.15 ms. Further wins would be structural (merging passes: e.g. `seed_sad` + `seed_pick`, the sub-pixel pairs, `em` +
`group`), not shader tweaks. `bench.mjs --gpu --passes` prints the per-pass table.

## Gaps

- V1's motion blur (`blur_*.comp`, `recordSynthBlur` with V1) is not ported: `synthBlur` throws for V1 with blur on.
- UI protection (`uiProtection`: Crosshair, WholeScreen, CrosshairV2) is not ported; V2's own static mask (`v2StaticMask`) is.
- RGBA16F with V2 is checked against `v2-formats` @ e215f83 (not yet on `framegen-v2`, whose library refuses V2 outside Rgba8:
  `hdr.mjs` against this branch's own build reports those rows as "no Vulkan reference"). V2's motion blur with RGBA16F is not
  tested (the library rounds each sub-frame to the nearest half explicitly; here the render target's conversion does).
- Rgb10 and the YUV formats (Nv12, P010) are not offered: a page gets RGBA8 or float textures.
- A frame needs every texture within `MAX_TEXTURE_SIZE`: the widest is the top pyramid level's search tile, 13 × its block count;
  extreme aspect ratios (8192 × 64 at flowScale 1) are refused by `frameSizeError` with the size it needs.
- The per-pass overhead of WebGL (one render pass per program, ~45 per pair) dominates on a fast GPU (see Performance).
