# WebGL2 backend

The frame interpolation of this library (`gpu/`, Vulkan) for a WebGL2 context: GLSL ES 3.00 fragment passes and a small JS host.
It runs V1 (the algorithm of `proto/interp.py`) and V2, the morphing interpolator, in both modes (Performance and Quality), with
V2's static-overlay mask, motion blur (V1, learned and V2) and RGBA16F (HDR / float) input, and, opt-in for V1, ml-fg's learned motion estimator
in place of V1's block search ("Learned motion estimate" below). It is meant for pages, where WebGL2 is all there is
(no compute shaders, subgroups or atomics): the Firefox extension browser-frame-gen vendors it instead of keeping its own port, and
exposes the settings in its popup.

Every pass is checked against the Vulkan library on the same frames (`tests/`, "Tests" below): the output differs by at most a few
hundredths of a level on average and the whole-frame decisions are identical.

## Vendoring

Copy `webgl/framegen.js`, `webgl/nets/` and `webgl/shaders/` (the whole trees; `__fg.shaderFiles` lists the shader files the
engine needs, as paths relative to `shaders/`). `framegen.js` is a classic script: it attaches `globalThis.__fg` and nothing else,
never fetches, never touches the DOM or a `<video>`. The caller loads the shader texts (for the extension: through its asset relay)
and passes them to `createEngine`. `nets/e2v5.js`, `nets/e1v5.js`, `nets/e2v4.js`, `nets/e1v4.js`, `nets/e2v3.js`, `nets/e1v3.js`, `nets/g2v2.js`, `nets/g2v3.js`, `nets/m2v2.js` and `nets/m2v3.js` are classic scripts too (the learned
estimator's weights, 540 KB for each e2 and 162 KB for each e1; each attaches `globalThis.__fg.nets[<name>]`, in either order with `framegen.js`): load them next to
`framegen.js` (for the extension: in the same content-script list), only needed for `motion: 'learned'`. `LAYOUT.md` and
`README.md` are the design docs; `tests/` stays here.

## API

Compatible with the extension's `LAYOUT.md` §6 (the engine it replaces): `createEngine(gl, shaderSources, settings)`, `resize`,
`pushFrame(source)`, `estimate()`, `synth(t, framebuffer, viewport, opts)` → bool, `presentFrame`, `setSettings`, `passCounts`,
`debugRead(name)`, `stats()`, `dispose`, `shaderFiles`, `defaultSettings()`, `checkSupport(gl)`, `levelLayout`. `LAYOUT.md` here is
that contract, updated. Additions (nothing removed):

| Addition | What |
|---|---|
| `__fg.blurMotionVersions` | `[1, 2]` (frozen, read-only): the `motionVersion` values `synthBlur` supports (V1 covers `motion: 'learned'`) |
| `__fg.version` | `'8'` (`'7'`: the real game nets g2v2 / g2v3 / m2v2 / m2v3; `'8'` adds `synthBlur` for V1 and learned; `'6'`: the colour net variant; `'5'`: e2v5 / e1v5 as the default / small net, before the colour and learned-features variants; `'4'`: e2v4 / e1v4 as the default / small net; `'3'`: the learned estimator before e2v4 / e1v4, the zero test, the static guard and `smallNet`; `'2'` before the learned estimator) |
| settings `v2Mode` | `'quality'` (default, as the library) or `'performance'`. Runtime; a switch keeps the estimate (the other mode's part is derived by its first `synth`, with the same output as that mode's own estimate) |
| settings `blurSamples`, `blurWeights` | motion blur (`synthBlur`, V1 / learned / V2): 0 / 1 = off, 2..16 sub-frames; `'gaussian'` (default) or `'even'`. Runtime |
| settings `inputFormat` | `'rgba8'` (default) or `'rgba16f'`: frames, warps and output in RGBA16F. Structural (recompiles the programs, drops the frames) |
| settings `transfer`, `linearWhiteNits` | `rgba16f` only: `'encoded'` (default: display-encoded values, PQ / HLG / sRGB; a level is value × 255) or `'linear'` (scRGB / linear light, measured as PQ codes; 1.0 = `linearWhiteNits`, default 80). Structural |
| `engine.synthBlur(t0, t1, framebuffer, viewport, opts)` | the library's `recordSynthBlur` (V1, learned or V2): the weighted average of `blurSamples` sub-frames at the midpoints of N equal slices of [t0, t1], each exactly `synth(t_k)`'s output; `0 <= t0 <= t1 <= 1`; blur off or t0 == t1: the instant t1. works for every motion version (`__fg.blurMotionVersions`) |
| `__fg.frameSizeError(gl, w, h, settings)`, `__fg.minSize` | why a frame size cannot run (below 16 × 16, the library's minimum, or a texture above `MAX_TEXTURE_SIZE`), else null. `resize` / `pushFrame` throw a `RangeError` with that text and keep the engine at its old size |
| `__fg.checkSupport(gl, frame)` (changed) | returns `{ ok, missing, sizeError, quality, qualityError }`. `missing`: GL feature names only, as before. `sizeError`: the optional second argument `{width, height, flowScale}` checked by `frameSizeError`, else null (it used to be pushed into `missing`; `ok` is still false when it is set). `quality` / `qualityError`: whether V2 Quality can run here (the MIN-blend probe below) and, if not, why; does not affect `ok` (V1 and V2 Performance do not need it) |
| `engine.support` | the `checkSupport(gl)` result the engine was created with |
| V2 Quality refusal | where `checkSupport(gl).quality` is false, `createEngine` / `setSettings` throw a `RangeError` (`framegen: settings.v2Mode 'quality' is not supported on this context (<qualityError>); use v2Mode 'performance' or motionVersion 1`) for any settings with `motionVersion` 2 and `v2Mode` `'quality'`, and keep the current settings. V1 (whatever `v2Mode` says) and V2 Performance run as before |
| `__fg.assemblePrograms(sources, settings)` | the exact text of every program the host compiles (for validation) |
| `engine.setPassHook(fn)` | profiling only: `fn(passName, 0 / 1)` around every draw |
| settings `motion`, `motionNet`, `motionPrecision` | V1's learned motion estimate (below): `motion` `'search'` (default: V1's block search) or `'learned'`; `motionNet` `'e2v5'` (default), `'e1v5'` (the small net), `'e2v4'`, `'e1v4'`, `'e2v3'` or `'e1v3'`, or the opt-in `'g2v2'` (colour variant, 25 inputs), `'g2v3'` (learned-features variant, 27 inputs) and `'m2v2'` (the colour allrounder) and `'m2v3'` (the features allrounder) (LAYOUT.md 10.10, 10.11); `motionPrecision` `'fp16'` (default) or `'fp32'`. Runtime (a change drops the estimate, like a `motionVersion` switch); ignored by V2; `'learned'` needs `flowScale` 2 |
| settings `learnedZeroTest`, `learnedStaticGuard` | learned estimate only (below): the zero-motion test after the net (`true` default; a change drops the estimate) and the static guard after the synthesis (`false` default; runtime, no drop). Same names as the Vulkan library's settings |
| `__fg.nets` | the learned nets' weights, attached by `webgl/nets/<name>.js` |
| `__fg.defaultNet`, `__fg.smallNet` | read-only strings: `'e2v5'` (= `defaultSettings().motionNet`) and `'e1v5'`, the recommended small / fallback net (~1.9x faster at 1080p). A caller that drops to a cheaper net (e.g. at 1080p / high refresh rates) sets `motionNet: __fg.smallNet` instead of naming a net; both follow future nets |
| `checkSupport` `learned`, `learnedError` | whether the learned estimator can run here (WebGL2 core minimums, below); does not affect `ok`. Where false, or where the selected net's script is not loaded, `createEngine` / `setSettings` throw a `RangeError` for learned settings and keep the current ones |
| `stats()` `duplicate`, `dupDiff` | after a learned estimate: whether the pair was a duplicate (skipped) and its mean absolute difference; null otherwise |
| `debugRead('ml.*')`, `debugRead('sg')` | the learned estimator's textures, the zero test's (`ml.pfNet`, `ml.ztCost`, `ml.zt`) and the static guard's filters (`LAYOUT.md` §10.3, §10.8, §10.9) |

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

### Learned motion estimate (V1, opt-in) — new

`motion: 'learned'` replaces V1's block search (the estimate passes of `LAYOUT.md` §3.3) by ml-fg's learned motion estimator,
a small coarse-to-fine conv net on the luma (ml-fg `export/SPEC.md`, `export/reference.py`); V1's synthesis (field lookup, warps,
7 × 7 mismatch, trust, blend, the scene-cut and flick fallbacks) is unchanged and reads the net's output exactly where it reads
the search's: the per-pixel half-vector field `pixfield` and the scene-cut sum `sadSum` (= the net's `sad` × bw × bh, so
`meanSad` is the net's `sad` and the cut stays `sad > sceneCutSad` = 30). `LAYOUT.md` §10 is the pass contract.

```js
engine.setSettings({ motionVersion: 1, motion: 'learned' });                       // e2v5, fp16, zero test on (the defaults)
engine.setSettings({ learnedZeroTest: false });                                    // the net's field without the zero test (A/B)
engine.setSettings({ learnedStaticGuard: true });                                  // the static guard (opt-in)
engine.setSettings({ motion: 'learned', motionNet: __fg.smallNet });               // the small net (e1v5)
engine.setSettings({ motion: 'learned', motionPrecision: 'fp32' });                // the exact float32 port
engine.setSettings({ motion: 'search' });                                          // back to V1's search (the default)
```

| Setting | Values (default first) | |
|---|---|---|
| `motion` | `'search'`, `'learned'` | V1 only (V2 ignores it); `'learned'` needs `flowScale` 2 (the nets were trained at 2; else a `RangeError`) |
| `motionNet` | `'e2v5'`, `'e1v5'`, `'e2v4'`, `'e1v4'`, `'e2v3'`, `'e1v3'`, `'g2v2'`, `'g2v3'`, `'m2v2'`, `'m2v3'` | e2v5: e2v4 fine-tuned on more varied footage (the user's failure recordings, a friend's captures, 9 gameplay videos; same architecture, weight layout and cost), ml-fg's primary (with the zero test 31.23 dB mean on 14 clips at 2x, +0.31 dB over `final`, +0.04 over e2v4, no clip worse); e1v5 (`__fg.smallNet`): e1v4 fine-tuned the same way (with the zero test +0.12 dB over `final`, +0.06 over e1v4); e2v4: e2v3 fine-tuned (same architecture, weight layout and cost), the default before e2v5 (with the zero test +0.27 dB over `final` on 14 clips); e1v4 (the small net, `__fg.smallNet`, before e1v5): e1v3 fine-tuned, width 12, 28,330 weights, same layout and cost as e1v3, with the zero test +0.05 dB over e1v3 (better on 13 of 14 clips); e2v3: width 24, 95,530 weights (+0.23 dB over V1's `final` on its 7 clips, SPEC 9); e1v3: width 12, 28,330 weights, on par with `final` (−0.01 dB), 3.4x fewer MACs |
| `motionPrecision` | `'fp16'`, `'fp32'` | storage of the net's input and activations: half floats (8 per RGBA32UI texel) or RGBA32F; the field, pyramid, weights and all arithmetic are float32 either way |

| `learnedZeroTest` | `true`, `false` | the zero-motion test after the net (below); `false` gives the net's field as is (A/B) |
| `learnedStaticGuard` | `false`, `true` | the static guard after the synthesis (below); opt-in |

A change of `motion`, `motionNet`, `motionPrecision` or `learnedZeroTest` (with V1) drops the current estimate, like a
`motionVersion` switch; `learnedStaticGuard` acts on the next `synth`. The learned programs (22, the zero test's and the
guard's included) are compiled when it is first selected, its textures and weight buffer allocated by its first estimate.

**Zero-motion test** (`learnedZeroTest`, default on; ml-fg `export/SPEC.md` stage 7, `reference.py` `zero_test()`): the net has
no zero-motion prior, so static HUD is dragged by the motion around it and still frames flicker. The test is V1's
`pixel_field` zero candidate applied to the net's field: per flow pixel q, C(h) = the 5 × 5 box mean (replicate border) of
`|ya(q − h) − yc(q + h)|` (true bilinear, the net's own vectors, clamp to the fw × fh grid) and C(0) = the same of
`|ya − yc|`; where `C(0) − 1.0 ≤ C(h)` the vector becomes 0. `ya`, `yc` are the net's own flow-res lumas (0..255). The
net's level 0 writes `ml.pfNet` instead of `pixfield`, `learned/zt_cost` computes d and e per flow pixel, `learned/zt` the box
means, the mask and the final field into `pixfield`, plus per pixel the SAD term `zero ? e : d`, which `reduce_sad` sums into
`sadSum`, so `pixfield`, `meanSad` and the cut all belong to the final field (SPEC's SAD identity; `ml_sad` is skipped).
Window 5 and bonus 1.0 are SPEC's fixed constants (`learned/zt.frag.glsl`). A duplicate pair stays field 0, sad 0. Off: the
estimate is byte-identical to the backend without the test (`fd5c163`).

**Static guard** (`learnedStaticGuard`, default off; ml-fg `scripts/static_guard.py`, authoritative, and
`superdoc/features/static-guard.md`): where prev and curr are locally identical and the motion small, the generated pixel is
mixed towards the plain time mix `(1 − t) prev + t curr`. Per output pixel: s = max over R, G, B of |a − c|; d = 3 × 3 mean of
s; dMin = 9 × 9 min of s; motion = 2 max(t, 1 − t) · flowScale · |field at t| (the field magnitude bilinearly upsampled, cv2
`INTER_LINEAR`); gate = 1 − smoothstep(0.5, 1.0, motion); wLocal = (1 − smoothstep(7.65, 22.95, d)) · (1 − smoothstep(5.1,
7.65, d) · (1 − gate)); wHalo = 1 − smoothstep(5.1, 7.65, dMin); wStatic = max(wLocal, wHalo) · gate; out = out · (1 − wStatic)
+ ((1 − t) a + t c) · wStatic where wStatic > 0.001 (else bit-exact). The constants (static_guard.py `DEFAULTS`, which ml-fg may
still change) live in one marked block at the top of `shaders/learned/sguard.inc.glsl`. Colour differences are in the levels of
the backend's mismatch metric `FG_D` (Rgba8: integer levels; RgbaF16: value × 255 for encoded input, PQ code × 255 for
linear), the mix is in the frame's working values (RGBA8: on the rounded 8-bit output, as static_guard.py; RGBA16F: unrounded).
Passes: `learned/sg_h` + `learned/sg_v` (separable 3-tap sum and 9-tap min, RG16F, once per pair at the first guarded synth)
and `final_sg` (`synth/final.frag.glsl` with `FG_SGUARD`) in place of `final`. The scene-cut and flick fallbacks are untouched;
with the guard off the output is byte-identical.

**Weights**: `webgl/nets/e2v5.js`, `webgl/nets/e2v4.js`, `webgl/nets/e2v3.js` (540 KB each), `webgl/nets/e1v5.js`, `webgl/nets/e1v4.js`
and `webgl/nets/e1v3.js` (162 KB each), plus the game nets `webgl/nets/g2v2.js` (colour, SPEC 10; 607 KB) and `webgl/nets/g2v3.js` (learned features, SPEC 11; 623 KB), classic scripts that attach `__fg.nets.e2v5` / `.e1v5` / `.e2v4` / `.e1v4` / `.e2v3` / `.e1v3` / `.g2v2` / `.g2v3` / `.m2v2` / `.m2v3` (the float32 weights as base64, byte for byte ml-fg's `export/<net>/weights.bin`, sha256 checked by
the generator `tests/make_nets.py`; plus the manifest's layer table). No fetch: the page or extension loads them like
`framegen.js`. Selecting a net whose script is not loaded throws a `RangeError` naming the file (`framegen: settings.motion
'learned': the weights of net 'e2v5' are not loaded (load webgl/nets/e2v5.js next to framegen.js)`); with the default net that
is any `motion: 'learned'` without `nets/e2v5.js`.

Adding a net from ml-fg (same family: 5 levels, 14 input channels, width 12 or 24): `python3 -I webgl/tests/make_nets.py <net>`
(writes `nets/<net>.js`, checks the sha256), add `'<net>'` to `ML_NETS` in `framegen.js` (its first entry is `DEFAULTS.motionNet`;
change both to make it the default; `ML_SMALL_NET` is `__fg.smallNet`), and ship `nets/<net>.js`. The tests pick it up by themselves: `learned.mjs` runs every
`nets/<net>.js` that has an `export/<net>/testvec`, on every pair directory there (cut flags from each `vec.json`), and
The colour (`g2v2`) and features (`g2v3`) game nets pass `learned.mjs` on all their pairs (the extra chroma / feature inputs; max `pf` tolerance 0.15 px colour, 0.25 px features (SPEC 8 / 11.6)); `__fg.version` '7'. `bench.mjs` takes the net's name as a mode (`<net>`, `<net>f`, `...n`, `...g`).

**Duplicate pairs** (required: the net has no zero-motion prior; on two identical frames it still outputs motion, mean 0.3 px,
up to 5 px, and the synthesis would change ~40 % of the pixels): the estimate first applies FGR's duplicate test, the mean
absolute difference over all pixels and R, G, B at most 0.6 levels (`proto/harness.py:29,50` `DUP_DIFF`, the same threshold as
`gpu/compare.py:56,63`), on the GPU; on a duplicate every net pass writes zeros and skips its work, so `pixfield` = 0 and
`sadSum` = 0 and the generated frame is the input. `stats()` reports `duplicate` / `dupDiff`.

**Required features**: nothing beyond the backend's extensions; WebGL2 core minimums only (4 draw buffers and colour
attachments, uniform blocks of 10,416 bytes, 2 fragment uniform blocks, 9 texture units: the static guard's `final_sg`), reported as `checkSupport(gl).learned`
/ `learnedError`.

**Memory** (1080p, 2x; e2v5 = e2v4 = e2v3, e1v5 = e1v4 = e1v3): e2v3 fp16 ~77 MB, e1v3 fp16 ~61 MB, e2v3 fp32 ~140 MB, e1v3 fp32 ~90 MB of textures (1440p: x1.76),
plus the weight buffer (400 / 122 KB); V1's own textures stay allocated.

**Results** (`tests/learned.mjs` vs ml-fg's test vectors, `export/SPEC.md` §8; 2026-10-26, Chromium 156; the same on SwiftShader,
ANGLE / Vulkan / RADV and ANGLE / OpenGL / radeonsi on the RX 7900 XTX, every row PASS): every pair of both nets (e2v3: static,
motion, far, cut, odd 319 × 181, hd 1920 × 1080, overlay; e1v3: motion, cut, odd, overlay). These rows run with
`learnedZeroTest: false` (the net alone, against `pf.f32` / `sad`); the zero test's rows follow. e2v4 (static, motion, far,
cut, cut2, odd, hd, overlay; on e2v4 `cut` is no cut, sad 28.3, and `cut2` is, sad 42.8) is in the e2v4 paragraph below, e2v5 / e1v5
after e1v4's.

| | `fp32` (exact port; gates = reference.py's) | `fp16` (default; judged by the synthesised frame, SPEC §7) |
|---|---|---|
| ya / yc (flow-res luma) | bit-exact | bit-exact |
| coarsest level (in_top, act_top0..3, raw_top), max \|d\| | ≤ 2.5e-5 (gate 5e-5) | ≤ 4.7e-3 |
| pf mean / max \|d\| (px) | ≤ 5.0e-5 / ≤ 1.8e-2 (gates 2e-4 / 5e-2) | ≤ 3.5e-2 / ≤ 4.9 (low-texture pixels) |
| sad relative error | ≤ 1.3e-6 (gate 1e-3) | ≤ 1.1e-3 (gate 1e-2) |
| cut decision | identical (cut pairs: cut) | identical |
| synth(pf) vs synth(expected pf), `proto/interp.py`, t = 0.5 / 0.25 | 82–97 dB (≥ 80 = exact port) | 63.8–78.7 dB (gate 45) |
| PSNR vs the real middle frame (obs1.mp4), vs the expected pf's | equal to 0.001 dB | within 0.002 dB (gate 0.05) |
| the engine's own output (V1 synth on its field) vs synth(expected pf) | MAE ≤ 0.06 levels | MAE ≤ 0.06 |

RGBA16F input (`inputFormat 'rgba16f'`, e2v3 fp16, motion / odd): 52–58 dB, middle-frame PSNR within 0.002 dB (its luma is
unrounded, so ya / yc are not compared). Duplicate skip, both nets × both precisions: two identical frames give `duplicate: true`,
`dupDiff` 0, a zero field, `meanSad` 0 and an output identical to the input at t = 0.5 and 0.25; a near-duplicate (±1 level noise,
mean |a − c| 0.25) is skipped too; none of the test pairs is. V1 and V2 are untouched: `xbackend` (312 rows SwiftShader, 318
hardware GL) and `golden` give results identical in every metric to the backend before this change (19f179b), `hdr` and `blur`
pass.

**Zero-motion test** (`learnedZeroTest: true`, the same pairs estimated again, against the vectors' `pf_zt` / `zero_zt` /
`cost_h_zt` / `cost0_zt` / `sad_zt` / `cut_zt`; whole chain: the engine's own net field through its own test; 2026-10-26, the same
three GLs, every row PASS). Gates: SPEC §8 (fp32: mask ≥ 99.9 %, pf on the agreeing pixels mean ≤ 2e-4 / max ≤ 5e-2 px,
sad_zt ≤ 1e-3, same cut, synth ≥ 70 dB; fp16: sad_zt ≤ 1e-2, synth ≥ 45 dB, mask reported), plus the net field
(`ml.pfNet`) bit-identical to the run without the test.

| | `fp32` | `fp16` |
|---|---|---|
| zero-mask agreement (flipped px) | 99.9983–100 % (0–5; hd: 4–5 of 518,400, overlay: 0–1 of 57,600; reference.py: 0–6) | 99.875–99.998 % (1–428) |
| zeroed share vs the vectors' `zeroed_zt` | equal to 0.1 % on every pair | equal to 0.1 % |
| pf vs pf_zt on the agreeing pixels, mean / max (px) | ≤ 4.7e-5 / ≤ 1.8e-2 | ≤ 3.2e-2 / ≤ 4.9 |
| C(0) / C(h) vs cost0_zt / cost_h_zt, max (levels) | ≤ 1.5e-5 / ≤ 3.7e-2 (C(h) follows the field) | ≤ 1.5e-5 / ≤ 16 |
| sad_zt relative error | ≤ 7.9e-5 | ≤ 1.0e-3 |
| cut_zt | identical (cut pairs: cut, sad 33.0 / 34.2) | identical |
| synth(pf) vs synth(pf_zt), t = 0.5 / 0.25 | 73.7–98.7 dB (e1v3 overlay 73.7: its one flipped pixel, as reference.py's 73.8) | 58.8–80.3 dB |
| vs the real middle frame, vs synth(pf_zt)'s | within 0.001 dB | within 0.003 dB |
| engine output vs synth(pf_zt) | MAE ≤ 0.05 levels | MAE ≤ 0.06 |

RGBA16F input (e2v3 fp16, motion / odd): mask 98.86–99.16 % (its luma is unrounded), 49.9–55.6 dB, middle frame within 0.007 dB.
Duplicates with the test on: field 0, sad 0, output = input, as without it. With `learnedZeroTest: false` the estimate and the
output are byte-identical to the backend before the test (`fd5c163`: `pf.f32`, both outputs and sad / cut of all 24 rows per GL,
SwiftShader, ANGLE / Vulkan and ANGLE / GL compared file by file).

**e2v4** (`learned.mjs`, 2026-10-26, the same gates; SwiftShader, ANGLE / Vulkan / RADV and ANGLE / OpenGL / radeonsi, every row
PASS; 8 pairs x fp32 / fp16 + RGBA16F motion / odd, each without and with the zero test, + 4 duplicate cases per GL):

| | `fp32` | `fp16` |
|---|---|---|
| ya / yc | bit-exact | bit-exact |
| net pf mean / max \|d\| (px) | ≤ 5.0e-5 / ≤ 4.1e-2 | ≤ 6.6e-2 / ≤ 7.8 (cut2) |
| sad relative error | ≤ 1.4e-6 | ≤ 7.3e-4 |
| cut | identical (cut2: cut; cut: not) | identical |
| synth(pf) vs synth(expected pf), t = 0.5 / 0.25 | 82.1–95.4 dB (cut, cut2, far: identical, both fall back) | 64.8–78.6 dB |
| zero test: mask agreement | 99.9985–100 % (hd: 5–8 flipped px) | 99.809–100 % |
| sad_zt relative error | ≤ 1.7e-5 | ≤ 3.6e-4 |
| cut_zt | identical (cut2: cut) | identical |
| zero test: synth(pf) vs synth(pf_zt), t = 0.5 / 0.25 | 78.7–98.1 dB | 59.2–79.4 dB |

RGBA16F input (e2v4 fp16, motion / odd): 51.7–58.2 dB without, 50.2–56.1 dB with the zero test (mask 98.90–99.22 %). Middle frame
within 0.002 dB (RGBA16F: 0.008), engine output vs synth MAE ≤ 0.06 levels (RGBA16F: ≤ 0.49). Cost: as e2v3 (same passes), table below.

**e1v4** (`__fg.smallNet` before e1v5; `learned.mjs`, 2026-10-26, the same gates as e1v3; SwiftShader, ANGLE / Vulkan / RADV and ANGLE /
OpenGL / radeonsi, every row PASS; 5 pairs (cut, cut2, motion, odd 319 × 181, overlay) x fp32 / fp16, each without and with the
zero test, + 4 duplicate cases per GL; on e1v4 both `cut` (sad 43.2) and `cut2` (sad 56.1) are cuts, taken from `vec.json`):

| | `fp32` | `fp16` |
|---|---|---|
| ya / yc | bit-exact | bit-exact |
| net pf mean / max \|d\| (px) | ≤ 2.9e-5 / ≤ 1.1e-2 | ≤ 3.8e-2 / ≤ 5.7 (cut2) |
| sad relative error | ≤ 1.1e-6 | ≤ 4.6e-4 |
| cut | identical (cut, cut2: cut) | identical |
| synth(pf) vs synth(expected pf), t = 0.5 / 0.25 | 86.6–93.5 dB (cut, cut2: identical, both fall back) | 66.3–71.3 dB |
| zero test: mask agreement | 100 % (0 flipped px) | 99.912–99.993 % (1–51 flipped) |
| sad_zt relative error | ≤ 6.7e-7 | ≤ 5.6e-4 |
| cut_zt | identical (cut, cut2: cut) | identical |
| zero test: synth(pf) vs synth(pf_zt), t = 0.5 / 0.25 | 87.1–93.5 dB | 61.6–71.6 dB |

Middle frame within 0.002 dB, engine output vs synth MAE ≤ 0.06 levels. Cost: as e1v3 (same passes), table below.

**e2v5** (the default) and **e1v5** (`__fg.smallNet`) (`learned.mjs`, 2026-10-27, the same gates; SwiftShader, ANGLE / Vulkan /
RADV and ANGLE / OpenGL / radeonsi, every row PASS; e2v5 8 pairs + RGBA16F motion / odd, e1v5 5 pairs, x fp32 / fp16, each without
and with the zero test, + 4 duplicate cases per net per GL; the e2v4 / e1v4 / e2v3 / e1v3 rows re-run, all PASS). Worst over
pairs and the three GLs:

| | e2v5 `fp32` | e2v5 `fp16` | e1v5 `fp32` | e1v5 `fp16` |
|---|---|---|---|---|
| ya / yc | bit-exact | bit-exact | bit-exact | bit-exact |
| net pf mean / max \|d\| (px) | ≤ 7.4e-5 / ≤ 3.9e-2 (hd) | ≤ 7.9e-2 / ≤ 6.5 (cut2) | ≤ 4.6e-5 / ≤ 9.7e-3 | ≤ 3.7e-2 / ≤ 3.7 |
| sad relative error | ≤ 8.1e-6 | ≤ 7.2e-4 | ≤ 3.1e-6 | ≤ 6.1e-4 |
| synth(pf) vs synth(expected pf), t = 0.5 / 0.25 | 81.8–97.5 dB | 63.7–78.3 dB | 87.6–92.1 dB | 66.0–71.2 dB |
| zero test: mask agreement | 99.9965–100 % (0–7 flipped) | 99.79–100 % (0–393) | 100 % (0 flipped) | 99.91–99.99 % (2–51) |
| sad_zt relative error | ≤ 5.1e-5 | ≤ 7.4e-4 | ≤ 3.9e-6 | ≤ 4.2e-4 |
| zero test: synth(pf) vs synth(pf_zt) | 71.7–99.5 dB | 58.7–79.9 dB | 88.7–94.5 dB | 61.3–68.3 dB |

RGBA16F input (e2v5 fp16, motion / odd): 51.9–58.0 dB without, 50.3–55.7 dB with the zero test (mask 99.00–99.33 %). Middle frame
within 0.003 dB, engine output vs synth MAE ≤ 0.06 levels (RGBA16F: ≤ 0.49). The lowest fp32 zero-test row is e2v5 `hd` at t =
0.25 (71.7 dB on SwiftShader and ANGLE / Vulkan, 77.0 on ANGLE / GL; gate 70): 7 of 518,400 mask pixels flip (`reference.py` itself
flips 5 there against torch, SPEC 8), each a whole vector of difference; the net's field alone gives 81.8 dB on the same row.

Cut decisions (from `vec.json`, all identical on every GL and precision): e2v5 `cut` is a cut net-only (sad 32.21) and with the
zero test (sad_zt 30.17, 0.17 over the threshold), `cut2` a cut both ways (45.47 / 42.96); e1v5 `cut` is a cut net-only (31.26) but
**not** with the zero test (29.11), `cut2` a cut both ways (53.05 / 50.79). The engine's sad on the near-threshold rows: e2v5 `cut`
with the zero test fp16 30.164 (SwiftShader) / ≤ 2.3e-4 relative on the hardware GLs (≤ 0.007 sad units, against a margin of 0.17);
e1v5 `cut` fp16 31.264 / 29.124 (margins 1.26 / 0.89). No precision or GL comes near flipping either.

**Static guard** check (`learned.mjs`: e2v3 fp32 with the zero test on overlay t = 0.5 / 0.25, motion t = 0.25, cut t = 0.5,
overlay RGBA16F t = 0.25; the engine's own a, c, unguarded output and field at t through `static_guard.py`): RGBA8 max |d| 0
levels (SwiftShader) / 1 level on 1–4 pixels of 230,400 (hardware GL: rounding ties); RGBA16F 0.29–0.33 levels; the guard
changes 3,694–6,937 pixels of the overlay pair (static_guard.py: the same ± 1), leaves the cut pair's fallback untouched, and
switched off again gives the unguarded output bit for bit.

**Cost** (RX 7900 XTX, ANGLE / Vulkan, `bench.mjs --gpu`, battlefield, 1440p resampled from it; ms of GPU time back-to-back, 20
in one timer query; the estimate alone, then estimate + synth(0.5) per 2x pair):

| | V1 | V2 P | V2 Q | e2v3 fp16 | e1v3 fp16 | e2v3 fp32 | e1v3 fp32 |
|---|---|---|---|---|---|---|---|
| 1080p estimate | 0.68 | 0.67 | 0.78 | **1.47** | **0.79** | 2.75 | 1.10 |
| 1080p estimate + synth | 0.85 | 0.77 | 0.93 | 1.66 | 0.95 | 2.94 | 1.22 |
| 1440p estimate | 0.74 | 0.59 | 0.79 | **2.25** | **1.09** | 4.89 | 1.67 |
| 1440p estimate + synth | 0.90 | 0.95 | 1.19 | 2.49 | 1.36 | 5.15 | 1.88 |

e2v4 costs the same as e2v3 (identical passes; 2026-10-26, `bench.mjs --gpu --modes e2v4,ml2,...`, 1080p estimate, three
runs each: e2v4 fp16 1.53 / 1.57 / 1.70 ms vs e2v3 1.50 / 1.55 / 1.69, e2v4 fp32 2.79 / 2.84 vs e2v3 2.80 / 2.80; per-pair median
3.38 ms both). e1v4 costs the same as e1v3 (2026-10-26, `bench.mjs --gpu --modes v1,e2v4,e1v4,ml1,e2v4f,e1v4f,ml1f`, 1080p,
zero test on, three runs; estimate back-to-back): e1v4 fp16 0.75 / 0.77 / 0.78 ms vs e1v3 0.75 / 0.75 / 0.78 and e2v4 1.50 /
1.51 / 1.55 (about 2.0x); e1v4 fp32 1.03 / 1.04 / 1.04 vs e1v3 1.02 / 1.02 / 1.06 and e2v4 2.75 / 2.78 / 2.77; per-pair median
(push + estimate + synth, a sync per pair) e1v4 fp16 1.93 ms vs e2v4 3.28–3.35 ms (V1 1.26–1.28). e2v5 / e1v5 cost the same as
e2v4 / e1v4 (identical passes: 49 draws fp16, 69 / 49 fp32; 2026-10-27, `bench.mjs --gpu --modes
v1,e2v5,e1v5,e2v4,e1v4,e2v5f,e1v5f,e2v4f,e1v4f`, ANGLE / Vulkan, zero test on, three runs each; per-pair GPU median of estimate +
synth(0.5)): 1080p e2v5 fp16 3.31 / 3.36 / 3.32 ms vs e2v4 3.32 / 3.37 / 3.36, e1v5 1.95 / 1.97 / 1.97 vs e1v4 1.95 / 1.98 / 1.98,
e2v5 fp32 5.29 / 5.33 / 5.36 vs e2v4 5.33 / 5.28 / 5.39, e1v5 fp32 2.36 / 2.41 / 2.40 vs e1v4 2.37 / 2.38 / 2.36; 1440p
(`--size 2560x1440`) e2v5 fp16 4.93 / 4.95 / 4.94 vs e2v4 4.94 / 4.94 / 4.99, e1v5 2.78 / 2.78 / 2.80 vs e1v4 2.79 / 2.79 / 2.80,
e2v5 fp32 7.96 / 8.50 / 8.08 vs e2v4 8.78 / 8.30 / 8.46, e1v5 fp32 3.61 / 3.59 / 3.60 vs e1v4 3.47 / 3.60 / 3.56 (V1 1.27 / 1.79).
Back-to-back estimate + synth at 1080p fp16: e2v5 1.71–1.73 ms, e2v4 1.70–2.59 (one noisy run), e1v5 0.96–2.00, e1v4
1.48–1.77 (the e1 back-to-back figures scatter run to run here; the medians above do not).

47 draws per learned estimate (fp32 e2v3: 67, two passes per 24-wide layer). At level 0 a 24-wide hidden layer is ~0.33 ms
(5,184 MACs per texel over 960 × 544, about half the GPU's fp32 FMA peak): the convs dominate, the coarser levels are mostly
per-pass overhead (~0.02–0.04 ms each). The first `setSettings({ motion: 'learned' })` compiles 22 programs (17 before the zero test and the guard; 94 ms on ANGLE /
Vulkan, 590 ms on ANGLE / OpenGL); the first learned estimate decodes and uploads the weights and allocates (~40 ms of CPU
time). SwiftShader runs it too (the tests), far too slowly for use.

Added cost (1080p 2x, battlefield, `bench.mjs --gpu --passes --modes ml2n,ml2,ml2g,...`, GPU time per pair with a sync per
pair; three runs of e2v3 fp16, one each of e1v3 fp16 and e2v3 fp32): **zero test +0.02–0.08 ms (typically 0.04)** on ANGLE /
Vulkan for every net and precision (`ml_zt_cost` 0.026 + `ml_zt` 0.03 −
`ml_sad` 0.04 + one `reduce_sad` step), +0.02–0.03 ms on ANGLE / GL; **static guard +0.06–0.13 ms (typically 0.11)** on
ANGLE / Vulkan
(`ml_sg_h` 0.04–0.07 + `ml_sg_v` 0.02–0.03 once per pair, `final_sg` 0.06 in place of `final` 0.045), +0.08 ms on ANGLE / GL.
Memory: the zero test 16 MB at 1080p (`ml.pfNet`, `ml.ztCost`, `ml.zt`), the guard 16 MB (`sgH`, `sg`, RG16F full res).

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

### Motion blur (`synthBlur`, V1 / learned / V2)

The library's V2 blur is its synthesis with n samples; each sample is exactly `recordSynth(t_k)`'s stored output (8-bit or
half-float, with its own scene-cut / flick fallback), averaged with the weights. Here: per sample the full synthesis into `out`,
`v2/blur_accum` adds it × weight into an RGBA32F accumulator (blend ONE, ONE, sample order as the library's register sum),
`v2/blur_resolve` divides by the weight sum and stores. About 7 draws per sample (V1 and learned: their `synth` passes, `motionVersion` 1 including `motion: 'learned'`, see `blur.mjs` modes `v1` / `learned`). The library's V1 blur (`blur_*.comp`) is documented bit-identical per sub-frame to `recordSynth(t)`, so the same per-sample loop is its contract.

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
| blur | `blur.mjs`, `vkhdr.cpp --blur` | `synthBlur` vs `recordSynthBlur` (V2 P, V2 Q, V1) and vs the engine's own `synth(t_k)` average (learned e2v5, m2v3) |
| learned | `learned.mjs`, `learned_ref.py` (+ `make_nets.py`) | the learned estimator vs ml-fg's test vectors, every net (`nets/*.js`) x both precisions + RGBA16F input, without and with the zero-motion test (`*_zt` vectors), the duplicate skip, the static guard vs ml-fg's `scripts/static_guard.py` (`learned-gpu`: hardware GL) |
| bench | `bench.mjs` | ms per pair (`--gpu`: hardware GL, timer queries, `--passes`: per pass; modes `v1 v2p v2q ml2 ml1 ml2f ml1f` and any net by name, e.g. `e2v5 e2v5f`, `--size WxH`) |

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
| learned (2026-10-26, this branch: e2v3 + e1v3 x fp32 / fp16 + RGBA16F input, duplicates) | 20 + 4 per GL (SwiftShader, ANGLE Vulkan, ANGLE GL) | all PASS ("Learned motion estimate"); xbackend / golden / hdr / blur re-run on this branch: identical to 19f179b |
| learned e2v4 (2026-10-26, this branch: e2v4 default) | 42 net rows + 42 zero-test rows (e2v4 18 + e2v3 16 + e1v3 8) + 12 duplicate cases + 5 static-guard cases per GL (SwiftShader, ANGLE Vulkan, ANGLE GL) | all PASS; validate 209/209, golden PASS, xbackend SwiftShader 312/312 PASS |
| learned e2v5 / e1v5 (2026-10-27, this branch: e2v5 default, e1v5 = `__fg.smallNet`, `__fg.version` '5') | 80 net rows + 80 zero-test rows (e2v5 18 + e1v5 10 + e2v4 18 + e1v4 10 + e2v3 16 + e1v3 8, each with the zero test) + 24 duplicate cases + 5 static-guard cases per GL (SwiftShader, ANGLE Vulkan, ANGLE GL) | all PASS; validate 209/209, golden PASS, xbackend SwiftShader 312/312 PASS (MAE 0.000 vs Vulkan, decisions identical), hdr 144/144, blur PASS |
| learned e1v4 (2026-10-26, this branch: e1v4 = `__fg.smallNet`, `__fg.version` '4') | 52 net rows + 52 zero-test rows (e2v4 18 + e1v4 10 + e2v3 16 + e1v3 8) + 16 duplicate cases + 5 static-guard cases per GL (SwiftShader, ANGLE Vulkan, ANGLE GL) | all PASS; validate 209/209, golden PASS, xbackend SwiftShader 312/312 PASS |
| learned + zero test + static guard (2026-10-26, this branch) | 24 net rows + 24 zero-test rows + 8 duplicate cases + 5 static-guard cases per GL (SwiftShader, ANGLE Vulkan, ANGLE GL) | all PASS; xbackend (SwiftShader, 312 rows) and golden identical to fd5c163 in every metric, validate 209/209, hdr 144/144, blur PASS |

To compare against another library version: build it out of tree (e.g. `git archive <rev> gpu | tar -x -C <dir>`, `meson setup
<dir>/build <dir>/gpu`, `ninja -C <dir>/build fgtest`) and run `FG_VKTAG=<name> FG_GPU=<dir>/gpu FG_GPU_BUILD=<dir>/build node
webgl/tests/hdr.mjs` (or `blur.mjs`); `vkref.mjs` takes `FGTEST=<dir>/build/fgtest` (use a fresh `$FG_WORK` for its references).

## Performance

1080p (battlefield), 2x (one estimate + one synth per pair), flowScale 2 unless noted. GPU times from
`EXT_disjoint_timer_query_webgl2`; "back-to-back" = 20 estimate + synth in one query (the GPU kept busy, like the library's own
benchmark), "per pair" = one pair with a sync before it. The upload of a raw RGBA frame from JS (`pushFrame` with `{data}`, 8 MB)
costs another ~0.7 ms of GPU time; a `<video>` source is a GPU copy instead.
The learned estimator's costs (1080p and 1440p) are in "Learned motion estimate" above.

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

- UI protection (`uiProtection`: Crosshair, WholeScreen, CrosshairV2) is not ported; V2's own static mask (`v2StaticMask`) is.
- RGBA16F with V2 is checked against `v2-formats` @ e215f83 (not yet on `framegen-v2`, whose library refuses V2 outside Rgba8:
  `hdr.mjs` against this branch's own build reports those rows as "no Vulkan reference"). V2's motion blur with RGBA16F is not
  tested (the library rounds each sub-frame to the nearest half explicitly; here the render target's conversion does).
- Rgb10 and the YUV formats (Nv12, P010) are not offered: a page gets RGBA8 or float textures.
- A frame needs every texture within `MAX_TEXTURE_SIZE`: the widest is the top pyramid level's search tile, 13 × its block count;
  extreme aspect ratios (8192 × 64 at flowScale 1) are refused by `frameSizeError` with the size it needs.
- The per-pass overhead of WebGL (one render pass per program, ~45 per pair) dominates on a fast GPU (see Performance).
- The learned estimator: V1 only (V2 keeps its own estimate: it reads neither `pixfield` nor V1's `sadSum`), `flowScale` 2 only
  (the nets were trained at 2), UI protection not ported (as for V1). Its duplicate test is FGR's harness threshold (mean
  |a - c| <= 0.6): a pair where only a small object moves over a still picture (a mouse pointer on a static desktop) also counts
  as a duplicate and gets no motion (V1's trust then holds the nearer frame where the two differ).
