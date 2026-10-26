STATUS: READY

# ml-fg learned motion estimator: testing branch `ml-fg-testing`

Both ports of ml-fg's learned motion estimator pass ml-fg's test vectors (`export/SPEC.md` §8), including round 2's zero-motion
test (`*_zt` vectors) and the two new nets **e2v4** (the new default) and **e1v4** (the new small net). The two ports are the
Vulkan library (`gpu/`) and the WebGL2 backend (`webgl/`). The results below come from an independent QC re-run on
2026-10-26 on an RX 7900 XTX with RADV and Chromium 156, at `bb73df0` (round 2).

The estimator is still **opt-in**. V1 and V2 stay the defaults, and their output is byte-identical to `fd5c163` (round 1).

## What changed since round 1 (for browser-frame-gen and gamescope-ritz)

Round 1 was `fd5c163`. Round 2 is at `bb73df0` and later.

- **New default net `e2v4`.** It is e2v3 fine-tuned by ml-fg, with the same architecture, size, cost and memory. **New small net
  `e1v4`** is e1v3 fine-tuned, with e1v3's cost, and it is the recommended fallback when e2v4 costs too much. e2v3 and e1v3 are
  still selectable.
- **Zero-motion test, on by default.** It is ml-fg's SPEC stage 7, V1's per-pixel zero candidate applied to the net's field.
  Per flow pixel, the vector becomes 0 where `C(0) − 1.0 ≤ C(h)` (5×5 box means of the luma difference). The scene-cut SAD is
  then that of the final field. It largely fixes the round-1 issue "static HUD warps" (see "HUD" below). It costs < 0.01 ms in
  Vulkan and ~0.05 ms in WebGL at 1080p. Off, with the same net, the output is byte-identical to round 1.
- **Static guard, off by default** (ml-fg `scripts/static_guard.py`). It is a post-synthesis pass. Where both real frames are
  locally identical and the motion is small, it shows the plain time mix `(1 − t)·A + t·C`. Off = byte-identical. It costs
  ~0.05 ms per generated frame in Vulkan and ~0.1 ms in WebGL.
- **Vulkan host build (gamescope-ritz) must change, or Learned disappears.** Add `ml_zero.comp` and `ml_guard.comp`, plus
  `ml_guard` per non-RGBA8 format, to the shader build. Add `gpu/ml/e2v4/` and `gpu/ml/e1v4/` to the `ml_weights` inputs
  (meson `ml_nets = ['e2v3', 'e1v3', 'e2v4', 'e1v4']`, in that order). `framegen.cpp` only enables Learned when
  `ml_zero_spv.h`, `ml_guard_spv.h` and `ml_weights.h` all exist. With the old shader list, `init()` / `setSettings()` refuse
  Learned with a reason, and V1/V2 are unaffected.
- **WebGL vendor set (browser-frame-gen) must change.** Add the 5 new shader files (`learned/zt_cost.frag.glsl`,
  `learned/zt.frag.glsl`, `learned/sg_h.frag.glsl`, `learned/sg_v.frag.glsl`, `learned/sguard.inc.glsl`). Load
  `nets/e2v4.js`, because the default `motionNet` is now `'e2v4'`: `motion: 'learned'` without `nets/e2v4.js` throws a
  `RangeError` naming the file. Also load `nets/e1v4.js` for the small net. `__fg.version` is `'4'`. The new read-only
  `__fg.defaultNet` (`'e2v4'`) and `__fg.smallNet` (`'e1v4'`) let a caller pick a net without naming it.
- Learned now needs **9** fragment texture units in WebGL (8 before) and creates **56** pipelines in Vulkan (34 before).
- The duplicate skip, the cut rule (`sad > 30`) and V1's synthesis are unchanged.
- Settings to expose for A/B testing: `learnedZeroTest`, `learnedStaticGuard`, and `learnedNet` / `motionNet` with the new
  names. They are the same names in both backends (see below).

## Settings and defaults

| | Vulkan `gpu/framegen.h` `Settings` | WebGL `setSettings` / `createEngine` |
|---|---|---|
| Select Learned | `motionVersion = MotionVersion::Learned` (= 2) | `motionVersion: 1, motion: 'learned'` (V2 ignores `motion`) |
| Net | `learnedNet`: **`LearnedNet::E2v4`** (= 2, default), `E1v4` (= 3, the small net), `E2v3` (= 0), `E1v3` (= 1) | `motionNet`: **`'e2v4'`** (default, `__fg.defaultNet`), `'e1v4'` (`__fg.smallNet`), `'e2v3'`, `'e1v3'` |
| Net by name | `framegen::learnedNetName()` / `learnedNetFromName()` (`"e2v4"`, ...), `kLearnedNets` = 4 | the strings above; `nets/<name>.js` must be loaded |
| Zero test | `learnedZeroTest` = **true** | `learnedZeroTest: true` |
| Static guard | `learnedStaticGuard` = **false** | `learnedStaticGuard: false` |
| Precision | `learnedHalf` = **false** (fp32 activations) | `motionPrecision: 'fp16'` (**default**) / `'fp32'` |
| Runtime? | all runtime (`setSettings`, no GPU wait); a net, half or zero-test change applies from the next estimate; the guard from the next synth | a change of `motion` / `motionNet` / `motionPrecision` / `learnedZeroTest` **drops the estimate** (synth shows the real frame until the next `estimate()`); `learnedStaticGuard` applies from the next synth |
| Needs | `flowScale` 2 (else `settingsError()`), every `Format` | `flowScale` 2 (else `RangeError`), RGBA8 and RGBA16F |

**A/B toggles**

```sh
# Vulkan (fgtest)
gpu/build/fgtest --set algo=learned W H prev.rgba curr.rgba out.rgba                               # e2v4, zero test on, guard off
gpu/build/fgtest --set algo=learned,learnedZeroTest=0 ...                                         # the net's raw field (+ learnedNet=e2v3: round 1's output, byte-identical)
gpu/build/fgtest --set algo=learned,learnedStaticGuard=1 ...                                      # + static guard
gpu/build/fgtest --set algo=learned,learnedNet=e1v4[,learnedHalf=1] ...                           # the small net
gpu/build/uitest|formattest|blurtest check --algo learned [--zt 0|1] [--guard 0|1] [--net e2v4|e1v4|e2v3|e1v3]
MLFG_EXPORT=<ml-fg>/export MLTEST_DUMP=<dir> gpu/build/mltest && python3 -I gpu/ml/synthcheck.py <dir>
python3 -I gpu/ml/guardcheck.py W H a.rgba c.rgba off.rgba on.rgba DIR/pair_0000.fgmaps [--t T]    # guard vs static_guard.py
```

```js
// WebGL
engine.setSettings({ motionVersion: 1, motion: 'learned' });          // e2v4, fp16, zero test on, guard off
engine.setSettings({ learnedZeroTest: false });                       // A/B: no zero test (drops the estimate)
engine.setSettings({ learnedStaticGuard: true });                     // static guard (next synth)
engine.setSettings({ motionNet: __fg.smallNet });                     // e1v4
```

A gamescope host maps its own settings to `Settings::learnedZeroTest`, `learnedStaticGuard` and `learnedNet`. Use
`learnedNetFromName()` for a string setting.

## Files a host needs

**gamescope-ritz (Vulkan), on top of round 1:**
- Shaders: `gpu/shaders/ml_zero.comp`, `ml_guard.comp`, and `ml_guard` with `-DFG_FORMAT_<FMT>=1` for every non-RGBA8 format in
  use (meson `ml_colour_srcs = ['shaders/ml_dup.comp', 'shaders/ml_guard.comp']`). The full Learned list is now `ml_dup`,
  `ml_reduce`, `ml_pyr`, `ml_match`, `ml_conv`, `ml_out`, `ml_zero`, `ml_guard` (+ `ml_common.glsl`).
- Weights: `gpu/ml/e2v4/` and `gpu/ml/e1v4/` (`manifest.json` + `weights.bin`, sha256-identical to ml-fg's export).
  `embed_weights.py` gets all four nets in `LearnedNet` order: meson `ml_nets = ['e2v3', 'e1v3', 'e2v4', 'e1v4']`.
- The weights buffer (binding 15) now holds all four nets: 1.0 MB, uploaded in the first Learned estimate's command buffer as
  before.

**browser-frame-gen (WebGL):**
- Vendor `webgl/framegen.js`, `webgl/shaders/` (62 files, all in `__fg.shaderFiles`; 13 under `learned/`) and `webgl/nets/`
  (`e2v4.js` 540 KB, `e1v4.js` 162 KB, `e2v3.js`, `e1v3.js`). The QC loaded `framegen.js` and the four nets in a bare VM
  context with no fs, fetch or DOM. Every listed shader exists, no file under `shaders/` is unlisted, and the weights' sha256
  match the export (e2v4 `ba6981cf…`, e1v4 `28128da2…`).
- Load at least `nets/e2v4.js` (the default) and, for a fallback, `nets/e1v4.js`, in any order with `framegen.js`.

## Duplicate skip (unchanged, both backends)

FGR's duplicate test runs before the net (`proto/harness.py` `DUP_DIFF = 0.6`). A pair with mean |a − c| ≤ 0.6 levels (over the
pixels and R, G, B) skips the net. Its field is 0 and its SAD is 0, so the output is V1's answer for a still pair. The zero test
does not change this (duplicates: field 0, sad 0, output = input, with the test on and off). The rule is whole-frame, so a small
cursor moving on a still desktop is also held. `obs1` 3000 → 3002 (mean |a−c| 0.23) is skipped by every net.

## Zero test and static guard

- **Zero test** (SPEC stage 7, `reference.py` `zero_test()`): per flow pixel q, C(h) = the 5×5 box mean (replicate border) of
  `|ya(q − h) − yc(q + h)|` with the net's h, and C(0) = the same of `|ya − yc|`. Where `C(0) − 1.0 ≤ C(h)`, the vector becomes 0.
  The SAD is then accumulated from the final field. Vulkan: `ml_zero.comp` replaces `ml_out`. WebGL: `ml_zt_cost` + `ml_zt`
  (MRT), and `reduce_sad` sums the zero test's per-pixel SAD terms.
- **Static guard** (`static_guard.py`, authoritative): per output pixel, s = max over RGB of |A − C|, d = 3×3 mean of s, dMin =
  9×9 min of s, and motion = 2·max(t, 1 − t)·flowScale·|field at t| (bilinear to full res). Then
  `w = max(wLocal, wHalo) · gate`, where
  - gate = 1 − smoothstep(0.5, 1.0, motion);
  - wLocal = (1 − smoothstep(7.65, 22.95, d)) · (1 − smoothstep(5.1, 7.65, d)·(1 − gate));
  - wHalo = 1 − smoothstep(5.1, 7.65, dMin).

  The output is `mix(out, (1 − t)A + tC, w)` where w > 0.001 and bit-exact elsewhere. Whole-frame fallbacks (cut, flick) are
  untouched. Both backends follow the script's max(wLocal, wHalo)·gate, not the simplified product in the ml-fg brief.
- **Constants are identical in both backends:** Vulkan `gpu/framegen.cpp:369` `kStaticGuard = {7.65, 22.95, 5.1, 7.65, 4, 0.5,
  1.0, 0.001}`, WebGL `webgl/shaders/learned/sguard.inc.glsl` (`SG_*`). Both are `static_guard.py` `DEFAULTS`. ml-fg may still
  send final values; each backend has one place to change them.
- Vulkan: `ml_guard.comp` runs after the fallback pass and before the UI paste-back (`ui_patch`). `recordSynthBlur` applies it
  once to the average. WebGL: `ml_sg_h` + `ml_sg_v` (RG16F, once per pair) and `final_sg` in place of `final`. WebGL has no UI
  protection.

## Measured parity (QC re-runs)

Tolerances (SPEC §8): for an exact fp32 port, pf mean ≤ 2e-4 px, max ≤ 5e-2 px, sad rel ≤ 1e-3, the same cut and synth ≥ 80 dB.
For the zero test: mask agreement ≥ 99.9 %, pf compared where the masks agree, sad_zt rel ≤ 1e-3, the same cut_zt and synth
≥ 70 dB. With shortcuts (fp16, FGR luma), synth ≥ 45 dB and the middle-frame PSNR within 0.05 dB.

Vectors: e2v3 static, motion, far, cut, odd, hd and overlay; e1v3 motion, cut, odd and overlay; e2v4 static, motion, far, cut,
cut2, odd, hd and overlay; e1v4 motion, cut, cut2, odd and overlay. The cut flags come from each `vec.json`. On **e2v4, `cut` is
not a cut** (sad 28.3) and `cut2` is (42.8). On e1v4 both are (43.2 / 56.1).

**Vulkan**: `mltest` 29 cases, 2600 assertions, 0 failed; `synthcheck.py` 216/216 PASS. Column meanings:
- "ref" is the net fed the reference luma (the exact-port check);
- "fgr" is the production path (FGR 14-bit luma);
- "zt" is the same with the zero test;
- "stage" is the zero test alone on the vectors' `pf.f32`.

| net | ref pf max (px) | ref sad rel | zt mask agree: stage / ref / fgr / fgr+half | synth dB ref / refzt / fgrzt / fgrzt+half | cuts |
|---|---|---|---|---|---|
| e2v3 (7 pairs) | ≤ 2.5e-2 | ≤ 2.9e-6 | 100 / ≥ 99.998 / ≥ 99.96 / ≥ 99.94 % | 84.1–96.1 / 82.3–96.5 / 66.4–84.7 / 62.8–79.1 | same (cut: true) |
| e1v3 (4) | ≤ 4.4e-3 | ≤ 3.4e-6 | 100 / ≥ 99.998 / ≥ 99.97 / ≥ 99.97 % | 87.6–90.5 / 73.7–90.5 / 62.3–70.0 / 61.9–65.8 | same (cut: true) |
| **e2v4** (8) | ≤ 4.5e-2 (hd) | ≤ 1.6e-6 | ≥ 99.9994 (hd: 3 px) / ≥ 99.9986 / ≥ 99.95 / ≥ 99.91 % | 83.7–93.1 / 81.1–93.1 / 66.3–82.7 / 63.5–78.5 | same (cut: false, cut2: true) |
| **e1v4** (5) | ≤ 1.1e-2 | ≤ 1.5e-6 | 100 / 100 / ≥ 99.97 / ≥ 99.93 % | 86.8–89.7 / 87.6–90.5 / 66.3–69.9 / 65.1–68.6 | same (cut, cut2: true) |

The lowest refzt synth, e1v3 73.7 dB, is the overlay pair, where one mask pixel flips. It is still above the 70 dB gate.

- ref `ya`/`yc` are exact.
- Coarsest level `in_top`/`act_top*`/`raw_top` ≤ 2.5e-5 (gate 5e-5).
- Zero-test sad_zt rel ≤ 7.7e-5 on the ref path, and the zeroed share equals the vectors' to 0.01 %.
- Middle-frame PSNR is within 0.001 dB of the expected field's (fp16 + zt: 0.003 dB) in every row.
- Duplicates (all 4 nets): identical frames give a zero field, SAD 0 and output = input. Mean |a−c| 0.5 is skipped, 1.0 runs
  the net.
- The tightest margin is e2v4 hd ref pf max 4.47e-2 against the 5e-2 gate. It passes, and its mean (4.9e-5) is far inside.

**WebGL** (`learned.mjs`, QC re-run on SwiftShader and ANGLE/Vulkan, every row PASS on both):
- 52 rows, each without and with the zero test: e2v4 8 pairs, e1v4 5, e2v3 7 and e1v3 4, at fp32 and fp16, plus 4 RGBA16F rows.
- 16 duplicate cases: 4 nets × fp32/fp16 × zero test on/off.
- 5 static-guard rows.

ANGLE/OpenGL was not re-run by the QC; the worker reports PASS there too.

| | SwiftShader | ANGLE / Vulkan (RADV) |
|---|---|---|
| fp32 net pf mean / max (px) | ≤ 5.0e-5 / ≤ 4.0e-2 | ≤ 5.0e-5 / ≤ 4.1e-2 |
| fp32 sad rel / zt sad_zt rel | ≤ 1.8e-6 / ≤ 7.8e-5 | ≤ 2.7e-6 / ≤ 7.9e-5 |
| fp32 zt mask agreement | ≥ 99.998 % (≤ 5 px flipped) | ≥ 99.998 % (≤ 6 px) |
| fp32 synth dB, net / zt (t = 0.5, 0.25) | 82.3–97.0 / 80.1–97.0 | 82.1–95.7 / 73.8–98.8 |
| fp16 synth dB, net / zt | 67.8–78.8 / 61.4–80.3 | 63.9–76.9 / 58.8–77.0 |
| fp16 sad rel / sad_zt rel (gate 1e-2) | ≤ 3.5e-4 / ≤ 5.3e-4 | ≤ 1.1e-3 / ≤ 1.0e-3 |
| fp16 zt mask agreement (reported) | ≥ 99.92 % | ≥ 99.81 % |
| RGBA16F input (e2v4 / e2v3 fp16), net / zt | 51.7–58.3 / 49.9–56.0 dB | 51.7–58.3 / 49.9–56.0 dB |
| cut / cut_zt decisions | identical (e2v4: only `cut2`; e1v4: `cut` and `cut2`) | identical |
| net field with the zero test (`ml.pfNet`) vs without | bit-identical | bit-identical |
| engine output vs synth(expected) | MAE ≤ 0.06 (RGBA16F ≤ 0.36) | MAE ≤ 0.06 (RGBA16F ≤ 0.42) |

`ya`/`yc` are bit-exact, and the middle-frame PSNR is within the 0.05 dB gate in every row. Duplicates: field 0, sad 0, output =
input, also for the near-duplicate.

**Static guard vs `static_guard.py`:**
- Vulkan (`gpu/ml/guardcheck.py`, QC re-run): 13/13 PASS, max |d| ≤ 1 level.
  - Pairs: obs2 5000, battlefield 2000 and obs1 13100 at 1080p, each at t = 0.5 and 0.25 with e2v4 and e1v4, plus a 1277×719
    crop.
  - 0–28 px differ by 1 level. The guard changed 0.12–3.9 % of the pixels (library and script agree to ±4 px).
- WebGL (`learned.mjs` static-guard rows, pinned to e2v3 fp32 + zero test; the guard does not depend on the net): 5/5 PASS on
  both GLs. The rows are overlay t 0.5 / 0.25, motion t 0.25, cut t 0.5 and overlay RGBA16F t 0.25, all at 640×360.
  - RGBA8: max |d| 0 levels on SwiftShader and ≤ 1 on ANGLE/Vulkan.
  - RGBA16F: 0.29 / 0.33 levels.
  - The guard changed 3.7k–6.9k px, exactly the script's count on RGBA8.
- Both: guard off = bit-identical output, and a cut frame is unchanged.

**V1 / V2 / round-1 regression vs fd5c163** (built side by side from `git archive fd5c163`):
- SPIR-V: all 114 shader headers of fd5c163 are byte-identical. There are 6 new ones: `ml_zero`, and `ml_guard` plus its 4
  format variants.
- fgtest: 66 outputs are `cmp`-identical over 4 footage pairs (obs2 5000, battlefield 2000, obs1 13100, obs1 3000). The
  configurations:
  - V1 at fs 1/2/4, `--t 0.25,0.75`, `--split`, `--bgra`, `ui=crosshair`;
  - V2 Q/P, V2 + `crosshairv2`;
  - 1277×719 nv12/f16/rgb10/BGRA;
  - **Learned E2v3 / E1v3 / E2v3 half with `learnedZeroTest=0`** against fd5c163's Learned, also at 1277×719 nv12.
- `uitest check --hash` (V1, V1 BGRA, V2 Q, V2 P), `formattest check --hash` (V1, V2 Q, V2 P) and `blurtest check` (V1, V2 Q,
  V2 P) give identical output (timing lines excluded). So do `--algo learned --zt 0 --net e2v3` against fd5c163's
  `--algo learned`; the only differing line is uitest's pipeline count (65 vs 43 after init: 4 nets instead of 2).
- WebGL (SwiftShader, `run.sh` QC re-run):
  - `validate`: 209/209 OK (194 in round 1; the new programs were added).
  - `golden`: PASS, 5 pairs.
  - `xbackend`: PASS. All **312 rows are identical to a fd5c163 run** (same footage, every metric).
  - `hdr`: 144/144 PASS. `blur`: PASS.
  - **Learned with `learnedZeroTest: false`**: e2v3 and e1v3, every pair, fp32 / fp16 / RGBA16F. All 96 files (`pf.f32`, the
    t 0.5 / 0.25 outputs and the sad / cut in `est.json`) are byte-identical to fd5c163's `learned.mjs` run on SwiftShader.

**Validation (GPU-AV)**: 12 fgtest runs of Learned under GPU-assisted validation found nothing. They covered:
- defaults (e2v4 + zero test) on one pair and on two pairs;
- guard on with `--t 0.25,0.5,0.75`, guard + `--split`;
- e1v4 + half + guard;
- 1277×719 NV12 + guard, 1277×719 P010 + guard, F16-linear + guard at 1080p, 1277×719 RGB10;
- BGRA + guard + `ui=crosshair`;
- a runtime switch `--then learnedZeroTest=0,learnedStaticGuard=1,learnedNet=e1v4`;
- a 1080p duplicate + guard.

`blurtest check --algo learned --guard 1` under the validation layer gave no messages either; it fails its 3 by-design checks,
see Known issues.

## Real footage (fgtest 1080p 2x, PSNR vs the real middle frame, dB; QC re-run)

| pair | V1 | e2v3 zt off (round 1) | e2v3 zt | e2v4 zt off | **e2v4 defaults** | e2v4 + guard | e2v4 half | e1v4 zt off | **e1v4** | e1v4 + guard |
|---|---|---|---|---|---|---|---|---|---|---|
| obs2 5000→5002 | 27.14 | 27.01 | 27.14 | 27.04 | 27.13 | 27.14 | 27.13 | 27.04 | 27.17 | 27.18 |
| battlefield 2000→2002 | 36.09 | 35.79 | 36.16 | 35.59 | 36.33 | 36.34 | 36.33 | 35.37 | 36.31 | 36.31 |
| obs1 13100→13102 | 34.80 | 34.09 | 34.30 | 32.86 | 32.98 | 33.01 | 32.98 | 32.50 | 32.58 | 32.61 |
| obs1 3000→3002 (duplicate) | 51.56 | 51.58 | 51.58 | 51.58 | 51.58 | 52.89 | 51.58 | 51.58 | 51.58 | 52.89 |

This reproduces the worker's table to the 0.01 dB.
- The zero test helps every moving pair with every net, by +0.08 to +0.94 dB.
- With it, e2v4 and e1v4 match V1 on obs2 (27.13 / 27.17 vs 27.14) and beat it on battlefield (+0.2 dB).
- On obs1 13100, e2v4 and e1v4 are 1.8–2.2 dB below V1 and 1.3–1.7 dB below e2v3 with the zero test. That pair is the one
  regression of the new nets against e2v3 on FGR's clips.
- The guard adds 0–0.03 dB on the moving pairs and +1.3 dB on the duplicate pair.

**HUD (round-1 issue (h)): the zero test fixes the smear, mostly.** These are crops of obs2 5000→5002 at 2–4x. The crop PSNR
against the real middle frame is in dB.

| crop | V1 | e2v3 zt off (round 1) | e2v4 zt off | **e2v4 defaults** | e2v4 + guard | e1v4 (zt) |
|---|---|---|---|---|---|---|
| "$69" money counter | 31.15 | 22.91 | 23.82 | **27.22** | 27.23 | 26.93 |
| "ANCIENT MOVEMENT" menu | 35.84 | 28.86 | 30.45 | **34.33** | 34.40 | 34.56 |
| "100" health | 40.68 | 40.40 | 40.63 | 40.83 | 41.07 | 40.70 |
| top timer / score | 29.47 | 29.63 | 29.82 | 30.66 | 30.94 | 30.50 |

Without the zero test, e2v3 and e2v4 smear the HUD into garbage: "$69" becomes an unreadable "S18" or "881" blob, and "ANCIENT
MOVEMENT" and "Toggle arrow" are doubled. With the zero test (the default), all the menu text and "$69" are readable again. The
menu is close to V1. The residue is on "$69": the glyph edges are notched or torn, with a few stray pixels around the "$" and
"9", where V1 is clean (27.2 vs 31.2 dB on that crop). e1v4 shows the same, a little more torn on the "9". **The static guard
does not visibly change this pair.** Its motion gate closes wherever the field moves more than ~1 px, and the field around the
counter still does. So the guard is no HUD fix here.

## Cost (RX 7900 XTX, 1080p 2x, ms of GPU time; QC re-measured)

**Vulkan**, `fgtest --bench 100 --mult 2`, battlefield. "b2b" is the back-to-back median per real pair (estimate + synth, 31
pairs per command buffer). "profiled" is the per-pass sum of one profiled pair.

| | V1 | e2v3 zt off | e2v4 zt off | **e2v4 defaults** | e2v4 + guard | e2v4 half | **e1v4** | e1v4 + guard | e1v4 half |
|---|---|---|---|---|---|---|---|---|---|
| b2b per pair | 0.30–0.33 | 1.42–1.47 | 1.46–1.48 | **1.44–1.48** | 1.48–1.50 | 1.09–1.13 | **0.62** | 0.64–0.68 | 0.49–0.51 |
| profiled pair | 0.33 | 1.37 | 1.35–1.36 | 1.35–1.37 | 1.41 | 1.05 | 0.57 | 0.62 | 0.50 |

- Zero test: the output pass goes from 0.014–0.015 ms (`net output`) to 0.019–0.020 ms (`net output + zero test`), so it adds
  **+0.005 ms** for e2v4 and e1v4.
- Static guard: one pass of **0.045–0.047 ms per generated frame**.
- e2v4 level 0 is 0.79 ms (half 0.59), e1v4 level 0 is 0.25 ms (half 0.22).
- Worker figures: e2v4 1.37 (half 1.06), e1v4 0.57 (half 0.50). These match the profiled-pair row.
- The machine was ~10 % slower today than in round 1. fd5c163's own build gives e2v3 b2b 1.43–1.44 ms today, against round
  1's 1.30. So the gap to round 1 is not a regression of this branch: the new build's e2v3 with the zero test off measures the
  same 1.42–1.47 ms.

**WebGL**, ANGLE/Vulkan, `bench.mjs --gpu`, battlefield, 3 runs (GPU timer queries, back-to-back). The guard rows use the zero
test.

| | V1 | **e2v4 fp16** | e2v4 zt off | e2v4 + guard | e2v4 fp32 | **e1v4 fp16** | e1v4 zt off | e1v4 + guard |
|---|---|---|---|---|---|---|---|---|
| pair (estimate + synth) | 0.68–0.70 | **1.85–1.91** | 1.85–1.94 | 1.89–1.91 | 3.11–3.26 | **1.02–1.10** | 0.97–1.03 | 1.02–1.06 |
| estimate | 0.46–0.50 | 1.63–1.72 | 1.61–1.65 | 1.65–1.69 | 3.01–3.08 | 0.83–0.85 | 0.80–0.83 | 0.83–0.87 |

- Per pass (`--passes`), the zero test replaces `ml_sad` 0.030 with `ml_zt_cost` 0.024 + `ml_zt` 0.028–0.032, and `reduce_sad`
  grows by ~0.03. That is **≈ +0.05 ms**, which is at the noise level of the totals (worker: typically 0.04).
- The guard is `ml_sg_h` 0.058 + `ml_sg_v` 0.022 (once per pair), plus `final_sg` 0.050 in place of `final` 0.035–0.039. That
  is **≈ +0.09 ms** (worker: 0.11).
- Worker figures: e2v4 fp16 ~1.5 and e1v4 0.77 ms estimate. The QC's are ~10 % higher, the same offset as on Vulkan today.
- 1440p was not re-measured.

gamescope's budget for the whole V1 2x pipeline is 0.33 / 0.47 ms (1080p / 1440p, INTEGRATION.md). e2v4 is about 4.5x that
and e1v4 about 1.9x. Memory is as in round 1: e2v4 = e2v3, e1v4 = e1v3. The zero test and the guard allocate nothing in Vulkan.
WebGL adds fw×fh RG32F ×2 + RGBA32F for the zero test (~16 MB at 1080p) and 2 × W×H RG16F for the guard (~16 MB at 1080p).

| | 1080p | 1440p |
|---|---|---|
| Vulkan e2v4 / e2v3 fp32 / half (activations) | 100 / 50 MB | 177 / 88 MB |
| Vulkan e1v4 / e1v3 fp32 / half | 58 / 29 MB | 103 / 52 MB |
| Vulkan state (pyramid, fields, partial sums) | 11 MB | 20 MB |
| Vulkan weights (all four nets, one buffer) | 1.0 MB | 1.0 MB |
| WebGL textures, e2v3 fp16 / e1v3 fp16 / e2v3 fp32 / e1v3 fp32 (worker, round 1; e2v4 / e1v4 the same) | 77 / 61 / 140 / 90 MB | ×1.76 |

## Integration notes

**gamescope-ritz (Vulkan):**
- Shader list and weights: see "Files a host needs". **Without `ml_zero` / `ml_guard`, Learned is refused as a whole**
  (`kMlNotBuilt`), not just the new features.
- Bindings 13 (state), 14 (activations) and 15 (weights) are as in round 1. The weights upload is as in round 1:
  `vkCmdUpdateBuffer` inside the first Learned estimate's command buffer, which must be submitted.
- Switching to Learned creates 56 pipelines (16 + 10 per net) at once.
- `flowScale` must be 2.
- UI protection: the net and the zero test read `P_LUMA` (the clean copies), and the guard reads the frames the synthesis read
  and runs before the UI paste-back.
- Expose `learnedZeroTest` (default on), `learnedStaticGuard` (default off) and `learnedNet` (`e2v4` default, `e1v4` small) for
  the morning A/B test.

**browser-frame-gen (WebGL):**
- Vendor the files listed above and load `nets/e2v4.js` (+ `nets/e1v4.js`).
- `__fg.version` is `'4'`. New fields: `__fg.defaultNet`, `__fg.smallNet`, `debugRead('ml.pfNet' | 'ml.ztCost' | 'ml.zt' |
  'sg')`.
- The first `motion: 'learned'` compiles 22 programs (17 in round 1).
- The default precision is still **fp16**.
- `estimate()` with learned leaves a uniform buffer bound, as in round 1.
- Not tested on Firefox or ANGLE/D3D11. There is no UI protection in the WebGL backend.

**API differences between the backends** (recorded, not changed):

| | Vulkan | WebGL |
|---|---|---|
| How it is selected | its own `MotionVersion::Learned` (= 2) | `motion: 'learned'` inside `motionVersion` 1 (V2 ignores it) |
| Net | `learnedNet` (`E2v4` default / `E1v4` / `E2v3` / `E1v3`) | `motionNet` (`'e2v4'` default / `'e1v4'` / `'e2v3'` / `'e1v3'`) |
| Small net | `LearnedNet::E1v4` (documented in `framegen.h`; no "small net" constant) | `__fg.smallNet` (`'e1v4'`) |
| Precision | `learnedHalf` (bool, **default false = fp32**) | `motionPrecision` (`'fp16'` **default** / `'fp32'`) |
| Zero test / guard | `learnedZeroTest` (true) / `learnedStaticGuard` (false) | the same names and defaults |
| Zero-test switch | applies from the next estimate, keeps the current one | drops the current estimate |
| Luma | FGR 14-bit (`P_LUMA`) | cv2 5.0.0 15-bit (exact with the vectors) |
| Weights | compiled in | separate scripts, must be loaded |

## Known issues

- (a) The synthetic quality tests are V1/V2-tuned and were not loosened. All counts are QC re-runs.
  - `uitest check --algo learned` fails 12 checks with e2v4 (10 with e2v3, 11 with e1v4), and the zero test does not change
    the count.
  - With `--guard 1` it fails 13. The extra one is "without re-entry the arm needs uiStillFrames = 8 still frames ...".
  - The failing checks are "control, Off shows the doubled crosshair" and "V2 [+ fill] leaves at most 1 artifact px".
  - Correction to the worker's note: the "control, Off shows the doubled crosshair" failure is **not** caused by the zero test.
    It fails the same way with `--zt 0`, and with e2v3 in round 1.
  - `formattest --algo learned` fails 3 (+1 with the guard). `blurtest --algo learned` now **passes** with e2v4 (e2v3 failed
    4) and fails 4 with e1v4 (the blur edge position).
- (b) By design, the guard is applied once to a blur average. So `blurtest --guard 1` fails its 3 "blur == average of the
  recordSynth sub-frames" checks (up to 36 levels on 0.5–18k bytes per configuration), and formattest has the matching extra
  failure. Learned's "scene cut forced" variant (sceneCutSad 0.5) is not a cut on that synthetic scene: its guarded differences
  equal the default variant's. So the guard's fallback skip for blur is covered only by the flick variants, which pass.
- (c) e2v4 hd, stage alone: the zero test flips 3 of 518,400 mask pixels (gate 99.9 %, passed by a wide margin). These are
  pixels at the `C(0) − 1 = C(h)` boundary, where fp32 summation order decides.
- (d) Hosts must add the new shaders and the ml dirs, or Learned is refused (Vulkan) or the default net is missing (WebGL).
- (e) The whole-frame duplicate test holds a small moving cursor on a still desktop (unchanged).
- (f) The static guard does not fix the dragged HUD where the field around it moves more than ~1 px (the motion gate,
  `static_guard.py` demo (a) asserts this as a known limit). The zero test is what fixes the HUD.
- (g) Static HUD is still not as clean as V1. Glyph edges on obs2's "$69" stay notched with the zero test (see "HUD").
- (h) obs1 13100: e2v4 and e1v4 are ~1.3 dB below e2v3 and ~2 dB below V1.
- (i) The `far` and `cut` (cut2) vector pairs cannot test the synthesis end to end (global fallback / cut).
- (j) Pre-existing, not Learned: syncval `SYNC-HAZARD-WRITE-AFTER-WRITE` on binding 2 with `ui=crosshair` (round 1 (g)). Syncval
  was not re-run in round 2; GPU-AV was.
- (k) WebGL: not tested on Firefox or D3D11, V1 only, no UI protection.
- (l) 1440p timings and the WebGL memory figures are the workers'. The QC measured 1080p only.
