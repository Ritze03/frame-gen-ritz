STATUS: READY

# ml-fg learned motion estimator: testing branch `ml-fg-testing`

Both ports of ml-fg's learned motion estimator pass ml-fg's test vectors (`export/SPEC.md` §8, all 9 pairs: e2v3 static, motion,
far, cut, odd 319×181, hd 1920×1080; e1v3 motion, cut, odd). These are the Vulkan library (`gpu/`) and the WebGL2 backend
(`webgl/`). The results come from an independent QC re-run on 2026-10-26 (RX 7900 XTX, RADV, Chromium 156).

The estimator is **opt-in**. V1 and V2 stay the defaults, and their output is byte-identical to `19f179b`.

## What was added

The net replaces only V1's motion search. It takes the flow-res luma and writes V1's per-pixel half-vector field and V1's
scene-cut SAD. V1's synthesis, trust ramp, fallbacks and motion blur are unchanged, and so is the cut rule (`sad > 30`,
`fallback.comp` and `webgl/shaders/synth/*` are not touched). The net is a 5-level coarse-to-fine conv net on the luma.
Two weight sets ship:

| net | width | weights | 1080p GMAC | ml-fg score vs V1 `final` (SPEC §9) |
|---|---|---|---|---|
| `e2v3` (default) | 24 | 95,530 | 13.2 | +0.23 dB over 7 clips (+0.62 on the 4 hard holdouts, −0.28 on FGR's own 3) |
| `e1v3` | 12 | 28,330 | 3.9 | −0.01 dB (on par) |

| | Vulkan `gpu/` | WebGL2 `webgl/` |
|---|---|---|
| Passes | `shaders/ml_dup` (per-format variants), `ml_reduce`, `ml_pyr`, `ml_match`, `ml_conv`, `ml_out` (+ `ml_common.glsl`) | `shaders/learned/` (8 files, all listed in `__fg.shaderFiles`) |
| Weights | `gpu/ml/<net>/{manifest.json,weights.bin}` (ml-fg's export, sha256-identical), embedded at build time by `gpu/ml/embed_weights.py` → `ml_weights.h` | `webgl/nets/e2v3.js` (540 KB), `e1v3.js` (162 KB): classic scripts, base64 f32, sha256-identical to the export |
| Luma | V1's `P_LUMA` (FGR's 14-bit luma, also the UI clean copies) | its own pass, cv2 5.0.0's 15-bit luma (bit-exact with the vectors' `ya`/`yc`) |
| Output | `field[fieldPF]` (fw×fh vec2, flow px), `u[1] = roundEven(256·sad·bw·bh)` | `pixfield` (RG32F fw×fh), `sadRed` last texel = `sad·bw·bh` |
| Tests | `gpu/mltest.cpp`, `gpu/ml/synthcheck.py` | `webgl/tests/learned.mjs`, `learned_ref.py`, `make_nets.py`; `run.sh` steps `learned`, `learned-gpu` |

## How to enable it

**Vulkan** (`gpu/framegen.h`):

| Field | Values (default first) | Notes |
|---|---|---|
| `Settings::motionVersion` | `MotionVersion::Learned` (= 2) | needs `flowScale` 2. Otherwise `settingsError()` says `MotionVersion::Learned needs flowScale 2 (its nets were trained at 2)`. Runs in every `Format`. |
| `Settings::learnedNet` | `LearnedNet::E2v3`, `E1v3` | runtime; the next estimate uses the new net |
| `Settings::learnedHalf` | `false` (fp32), `true` (fp16 activations) | runtime; the field, the pyramid and the arithmetic stay fp32 |

```sh
gpu/build/fgtest --set algo=learned[,learnedNet=e2v3|e1v3][,learnedHalf=0|1] W H prev.rgba curr.rgba out.rgba
gpu/build/uitest|formattest|blurtest check --algo learned      # runs, but see Known issues (a)
MLFG_EXPORT=<ml-fg>/export MLTEST_DUMP=<dir> gpu/build/mltest   # test vectors + duplicate tests (needs libpng)
python3 -I gpu/ml/synthcheck.py <dir>                            # end to end via proto/interp.py (reads footage/ for the middle frames)
```

**WebGL** (`engine.setSettings`, or the `createEngine` settings):

| Key | Values (default first) | Notes |
|---|---|---|
| `motion` | `'search'`, `'learned'` | V1 only (V2 ignores it); `'learned'` needs `flowScale` 2 (else `RangeError`) |
| `motionNet` | `'e2v3'`, `'e1v3'` | the matching `webgl/nets/<net>.js` must be loaded; a missing net throws `RangeError` naming the file |
| `motionPrecision` | `'fp16'`, `'fp32'` | activation storage. **The default is fp16** (Vulkan's default is fp32) |

Load `webgl/nets/e2v3.js` and/or `e1v3.js` next to `framegen.js`, in either order. Tests: `webgl/tests/run.sh learned`
runs on SwiftShader and `learned-gpu` on hardware GL (`FG_ANGLE=gl` for ANGLE/OpenGL). Both read the vectors from
`FG_MLFG_EXPORT` (default `~/Schinken/Programming/Local/ml-fg/export`).

## Duplicate skip (both backends, same rule)

The net has no zero-motion prior: on two identical frames it outputs motion (mean ~0.3 px, max ~5 px). The synthesis would
then change ~40 % of the pixels. So the GPU first runs FGR's duplicate test (`proto/harness.py:28` `DUP_DIFF = 0.6`, the same
threshold as `gpu/compare.py`): a pair whose mean |a − c| over the pixels and R, G, B is ≤ 0.6 levels counts as a duplicate.
Both backends compute `Σ|a−c| ≤ 0.6·3·W·H`. On a duplicate, every net pass exits, the field is 0 and the SAD is 0, so the
output is V1's answer for a still pair. WebGL reports `stats().duplicate` / `dupDiff`; Vulkan sets `u[2]`
(`ml_common.glsl` `U_MLDUP`).

**Caveat:** this is a mean over the whole frame, so a small cursor moving on a still desktop also counts as a duplicate, and
its motion is not interpolated. On real footage the rule does fire: `obs1.mp4` frames 3000 → 3002 (mean |a−c| 0.23) is
skipped by both nets and both precisions alike (identical outputs).

## Measured parity (QC re-runs)

Tolerances (SPEC §8) for an exact fp32 port: pf mean ≤ 2e-4 px and max ≤ 5e-2 px, sad relative ≤ 1e-3, the same cut,
synth ≥ 80 dB. With shortcuts (fp16, FGR luma) the gates are synth ≥ 45 dB and middle-frame PSNR within 0.05 dB.

**Vulkan** (`mltest` 12/12 cases, 685 assertions; `synthcheck.py` 36/36 PASS). "ref" = the net fed the reference luma
(the exact-port check); "fgr" = the production path (FGR 14-bit luma). Synth = dB of `interp.synth(pf)` vs
`interp.synth(expected pf)`.

| net pair | ref pf mean / max (px) | ref sad rel | synth dB ref / fgr / fgr+half | cut |
|---|---|---|---|---|
| e2v3 static | 2.1e-6 / 2.4e-4 | 1.3e-6 | 96.1 / 82.0 / 76.9 | same |
| e2v3 motion | 1.0e-5 / 3.2e-3 | 1.4e-7 | 88.3 / 69.0 / 67.7 | same |
| e2v3 far | 3.5e-5 / 5.6e-3 | 2.3e-7 | inf / inf / inf (global fallback, see below) | same |
| e2v3 cut | 4.4e-5 / 8.1e-3 | 6.7e-7 | inf (cut) | true = true |
| e2v3 odd | 4.6e-6 / 3.4e-4 | 2.4e-6 | 90.5 / 70.4 / 68.8 | same |
| e2v3 hd | 4.7e-5 / 2.5e-2 | 3.0e-7 | 84.1 / 77.9 / 68.9 | same |
| e1v3 motion | 8.9e-6 / 1.0e-3 | 8.3e-7 | 87.6 / 69.3 / 68.2 | same |
| e1v3 cut | 3.1e-5 / 4.4e-3 | 2.5e-7 | inf (cut) | true = true |
| e1v3 odd | 3.4e-6 / 1.3e-4 | 3.4e-6 | 90.5 / 70.4 / 69.1 | same |

The ref-path `ya`/`yc` are exact. FGR's luma differs on 0.18–0.52 % of the flow pixels, always by multiples of 0.25. The
coarsest level's `in_top`/`act_top0..3`/`raw_top` max is ≤ 2.0e-5 (gate 5e-5). PSNR against the real middle frame is within
0.001 dB of the expected pf's in every row, fp16 included. Duplicate cases, both nets: identical frames give a zero field,
SAD sum 0 and output = input. Mean |a−c| 0.5 is skipped and 1.0 runs the net.

**WebGL** (`learned.mjs`: 20 rows + 4 duplicate cases, all PASS on SwiftShader, ANGLE/Vulkan and ANGLE/OpenGL):

| | SwiftShader | ANGLE / Vulkan (RADV) | ANGLE / OpenGL (radeonsi) |
|---|---|---|---|
| fp32 pf mean / max (px) | ≤ 5.0e-5 / ≤ 1.5e-2 | ≤ 5.0e-5 / ≤ 1.5e-2 | ≤ 4.9e-5 / ≤ 1.8e-2 |
| fp32 sad rel | ≤ 7.2e-7 | ≤ 1.3e-6 | ≤ 5.5e-7 |
| fp32 synth dB (t = 0.5) | 84.1–95.4 | 84.1–95.7 | 84.3–96.1 |
| fp16 synth dB (t = 0.5) | 69.1–77.8 | 63.8–75.8 | 63.9–75.8 |
| fp16 sad rel | ≤ 1.9e-4 | ≤ 1.1e-3 | ≤ 1.1e-3 |
| middle-frame PSNR vs expected pf | within 0.002 dB | within 0.001 dB | within 0.001 dB |
| cut decisions | identical | identical | identical |

`ya`/`yc` are bit-exact. RGBA16F input (e2v3 fp16) gives 52–58 dB with the middle frame within 0.002 dB. The engine's own
output against `synth(expected pf)` has MAE ≤ 0.06 levels. The `far` and `cut` rows are `inf` on both backends:
`interp.synth` shows the nearer real frame for any pf there (`far` trips the 15 % global fallback, `cut` the scene cut). So on
those two pairs only the pointwise pf/sad checks and the cut decision carry information, and those pass.

**V1 / V2 regression vs 19f179b** (built side by side):
- SPIR-V: all 104 shader headers of 19f179b are byte-identical; there are 10 new `ml_*` headers.
- Hashes: `uitest check --hash` (V1, V1 BGRA, V2 Q, V2 P), `formattest check --hash` (V1, V2 Q, V2 P) and `blurtest check`
  (V1, V2 Q, V2 P) give identical hash lines.
- fgtest: 52 outputs `cmp`-identical over 4 footage pairs. The configurations were V1 at fs 1/2/4, V2 Q/P, t 0.25/0.5/0.75,
  `ui=crosshair`, V2 + `crosshairv2`, and at 1277×719 nv12/f16/rgb10/BGRA.
- WebGL `validate`: 194/194. `xbackend` (SwiftShader, 312 rows): every row identical to a 19f179b run in every metric.

**Validation**: 22 runs of Learned under GPU-AV (12) and synchronization validation (10). They covered:
- formats: RGBA8, BGRA8, NV12, P010, F16, F16-linear, RGB10;
- odd size 1277×719; e1v3, e2v3 and half; `--split` with t = 0.25/0.75; UI crosshair;
- runtime switches `learnedNet`/`learnedHalf` and V1 → Learned; a 1080p duplicate;
- `mltest` (319×181 and 1080p), and `formattest check --algo learned`.

GPU-AV found nothing. Syncval reports two `SYNC-HAZARD-WRITE-AFTER-WRITE` on binding 2 (`u[]`) with `ui=crosshair`. They
are **pre-existing**: V1 with `ui=crosshair` gives the same two on this build and on 19f179b (Known issues (g)).

**Real footage** (fgtest 1080p 2x, PSNR vs the real middle frame, dB):

| pair | V1 | e2v3 | e2v3 half | e1v3 |
|---|---|---|---|---|
| obs2 5000→5002 | 27.14 | 27.01 | 27.01 | 27.03 |
| battlefield 2000→2002 | 36.09 | 35.79 | 35.79 | 35.42 |
| obs1 13100→13102 | 34.80 | 34.09 | 34.09 | 33.26 |
| obs1 3000→3002 (duplicate) | 51.56 | 51.58 | 51.58 | 51.58 |

The outputs look sane: no garbage, and the artefacts at motion edges are of V1's kind. On FGR's own clips the net is a little
below V1, as SPEC §9 predicts. The library's e2v3 output on obs2 5000 agrees with `reference.py` + `interp.synth` at
55.6 dB. One visible difference: **static HUD text is warped** (Known issues (h)).

## Cost (RX 7900 XTX, 1080p 2x, ms of GPU time)

**Vulkan**, `fgtest --bench 100 --mult 2`, battlefield, re-measured. "Pair" = back-to-back median per real pair (estimate +
synth); "estimate" = the sum of the estimate passes from the profiled pair.

| | V1 | V2 P | V2 Q | e2v3 fp32 | e2v3 half | e1v3 fp32 | e1v3 half |
|---|---|---|---|---|---|---|---|
| 1080p pair | 0.29 | 0.18 | 0.26 | **1.30** | **0.98** | 0.51 | 0.46 |
| 1080p estimate | 0.27 | 0.15 | 0.22 | 1.25 | 0.95 | 0.46 | 0.42 |
| 1440p (worker, not re-measured) | 0.40 | 0.24 | 0.37 | 2.59 | 1.62 | 0.79 | 0.70 |

e2v3 fp32 per pass: luma 0.03, duplicate test 0.015, pyramid 0.007, levels 4..0 = 0.03 / 0.05 / 0.11 / 0.25 / 0.80, output
0.02. Level 0 dominates. gamescope's budget for the whole V1 2x pipeline is 0.33 / 0.47 ms (1080p / 1440p, INTEGRATION.md).
e2v3 is about 4x that, e1v3 about 1.5x.

**WebGL**, ANGLE/Vulkan, `bench.mjs --gpu`, battlefield, re-measured twice (GPU timer queries, estimate alone). The worker's
figures are in brackets.

| | V1 | e2v3 fp16 | e1v3 fp16 | e2v3 fp32 | e1v3 fp32 |
|---|---|---|---|---|---|
| 1080p estimate | 0.67–0.86 (0.68) | 1.51–1.72 (1.47) | 0.99–1.11 (0.79) | 2.81–2.95 (2.75) | 1.12–1.65 (1.10) |
| 1440p estimate (worker) | 0.74 | 2.25 | 1.09 | 4.89 | 1.67 |

The order of magnitude is confirmed. The WebGL numbers are noisy run to run (the first run had a CPU-heavy job in the
background). The first `motion: 'learned'` compiles 17 programs: 94 ms on ANGLE/Vulkan and 590 ms on ANGLE/GL (worker figure).

## Memory (on top of V1)

| | 1080p | 1440p |
|---|---|---|
| Vulkan e2v3 fp32 / half (activations) | 100 / 50 MB | 177 / 88 MB |
| Vulkan e1v3 fp32 / half | 58 / 29 MB | 103 / 52 MB |
| Vulkan state (pyramid, fields, partial sums) | 11 MB | 20 MB |
| Vulkan weights (both nets, one buffer) | 0.52 MB | 0.52 MB |
| WebGL textures, e2v3 fp16 / e1v3 fp16 / e2v3 fp32 / e1v3 fp32 (worker) | 77 / 61 / 140 / 90 MB | ×1.76 |

The Vulkan figures are computed from `mlLayout()`. The activation buffer only ever grows (switching e2v3 → e1v3 keeps the
larger one) and is freed with the size or by `releaseUnusedMotion()`.

## Integration notes

**gamescope-ritz (Vulkan):**
- Add the six `ml_*.comp` (+ `ml_common.glsl`) to the shader build. `ml_dup` also needs a `-DFG_FORMAT_<FMT>=1` variant for
  every non-RGBA8 format in use, like the colour shaders.
- Run `gpu/ml/embed_weights.py` (see `gpu/meson.build` `ml_weights`) to produce `ml_weights.h`. Without the headers
  `__has_include` leaves Learned out, and `init()`/`setSettings()` refuse it with a reason (`kMlNotBuilt`). V1/V2 are
  unaffected.
- Learned's sets use descriptor bindings 13 (state), 14 (activations) and **15 (weights)**, all storage buffers in the
  library's own layout.
- The weights are uploaded with `vkCmdUpdateBuffer` inside the **first Learned estimate's command buffer** (and again after
  `releaseUnusedMotion()` freed them). The library's existing contract that every record is submitted covers this. A host
  that discards that command buffer unsubmitted gets an un-uploaded weight buffer from then on.
- Switching to Learned creates 34 pipelines (both nets × both precisions) at once.
- Learned needs `flowScale` 2; any other value is refused.
- UI protection works as with V1: the net reads `P_LUMA`, i.e. the clean copies. But see (a) and (h).

**browser-frame-gen (WebGL):**
- Vendor `webgl/framegen.js`, `webgl/shaders/` (incl. `learned/`) and `webgl/nets/`. The QC loaded `framegen.js` + both nets in
  a bare VM context with no fs, fetch or DOM: all 57 `__fg.shaderFiles` exist under `webgl/shaders/`, and nothing outside
  `webgl/` is needed at runtime.
- Load `nets/<net>.js` as content scripts next to `framegen.js`.
- `__fg.version` is now `'3'`.
- New fields: `checkSupport().learned` / `learnedError`, `stats().duplicate` / `dupDiff`, `debugRead('ml.*')`.
- The default precision is **fp16**.
- `estimate()` with learned leaves a uniform buffer bound (UNIFORM_BUFFER and indexed points 0 and 1; framegen.js GL state
  contract).
- Not tested on Firefox or ANGLE/D3D11. There is no UI protection in the WebGL backend.

**API differences between the backends** (recorded, not changed):

| | Vulkan | WebGL |
|---|---|---|
| How it is selected | its own `MotionVersion::Learned` (= 2) | `motion: 'learned'` inside `motionVersion` 1 (V2 ignores it) |
| Net | `learnedNet` (`E2v3` / `E1v3`) | `motionNet` (`'e2v3'` / `'e1v3'`) |
| Precision | `learnedHalf` (bool, **default false = fp32**) | `motionPrecision` (`'fp16'` **default** / `'fp32'`) |
| Luma | FGR 14-bit (`P_LUMA`) | cv2 5.0.0 15-bit (exact with the vectors) |
| Weights | compiled in | separate scripts, must be loaded |

## Known issues

- (a) `uitest check --algo learned` fails 10 checks, `formattest` 3 and `blurtest` 4 (re-counted). These are V1/V2-tuned
  quality thresholds on synthetic scenes and were not loosened. The failing checks include "CrosshairV2 leaves at most 1 / 8
  artifact px" and "V2 leaves at most 1 artifact px" (uitest, formattest), the blur edge position within 2 px (blurtest) and
  "no highlight left at the object's old places" (formattest). So with Learned, CrosshairV2 leaves more pixels around the
  crosshair than with V1.
- (b) `uitest check --bgra 640 360` fails 8 box-margin checks on 19f179b too (pre-existing, not re-checked here).
- (c) Weights upload in the first estimate's command buffer (see gamescope notes).
- (d) Learned needs the ml shaders and `ml_weights.h` in the host build (gamescope notes).
- (e) uitest's "unknown motion version" probe moved from 2 to 3.
- (f) The duplicate test is whole-frame, so a small moving cursor on a still desktop counts as a duplicate.
- (g) Pre-existing, not Learned: syncval reports 2× `SYNC-HAZARD-WRITE-AFTER-WRITE` on binding 2 (`u[]`) for
  `fgtest --set ui=crosshair` with V1 too, on 19f179b as well. GPU-AV is clean.
- (h) **Static HUD warps.** The net has no zero-motion prior (V1 has `hudBonus`), so static HUD text over moving content is
  warped with the background. On obs2 5000, V1 keeps the "$69" money counter and the menu title readable; e2v3 smears them.
  The reference (`reference.py`) does the same, so this is the net's behaviour, not a port bug. `ui=wholescreen` did not help
  on that pair. Watch for this in the morning test.
- (i) The `far` and `cut` vector pairs cannot test the synthesis end to end (global fallback / cut; see parity).
- (j) WebGL: not tested on Firefox or D3D11, V1 only, no UI protection.
- Vulkan 1440p and WebGL 1440p timings, and the WebGL memory figures, are the workers'; the QC re-measured 1080p only.
