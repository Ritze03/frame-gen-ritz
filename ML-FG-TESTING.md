STATUS: READY

# ml-fg learned motion estimator: testing branch `ml-fg-testing`

Both ports of ml-fg's learned motion estimator pass ml-fg's test vectors (`export/SPEC.md` §8, §10, §11), including the
zero-motion test (`*_zt` vectors), round 3's nets **e2v5** (the default) and **e1v5** (the small net) and **round 4's two new
net variants** (colour, learned features; stand-in nets `c2smoke` and `c2feat`). The two ports are the Vulkan library (`gpu/`)
and the WebGL2 backend (`webgl/`). The round-4 results come from an independent QC re-run on 2026-10-28 on an RX 7900 XTX with
RADV and Chromium 156, at `eeb12a9` (gpu `6213959`, webgl `74ba982` + `08c9816`). The round-3 results come from an independent
QC re-run on 2026-10-27, at `d4a71e3`; the round-3 sections below still hold for the six luma nets, whose output is
unchanged. The round-2 sections further down (QC of 2026-10-26 at
`bb73df0`) still hold for e2v4 / e1v4 / e2v3 / e1v3, whose output is unchanged; where they say "the default" or "the small
net", read e2v4 / e1v4 as round 2's.

The estimator is still **opt-in**. V1 and V2 stay the defaults, and their output is byte-identical to `6133a4a` (round 2) and
`fd5c163` (round 1). Round 4 changed no default: the default net is still e2v5 and the small net e1v5.

## What changed in round 4 (for browser-frame-gen and gamescope-ritz)

Round 4 adds two **architecture variants** of the learned estimator, each selectable next to the current nets and opt-in only.
This is not a weight swap like rounds 2 and 3: the first conv's input contract changes. Until the real nets finish training
the variants are tested with ml-fg's smoke stand-ins (600 training steps, the real nets' shapes).

| variant | real nets (training) | stand-in (test vectors) | loaded as | first-conv inputs | spec |
|---|---|---|---|---|---|
| luma (rounds 1-3) | e2v5, e1v5, ... | | | 14 | §3-§8 |
| **colour** | `g2v2` (game), `m2v2` (allrounder) | `export/c2smoke-colour/` | **`c2smoke`** | 25 = 14 + 2 colour-mismatch + 9 colour-cost | §10 |
| **learned features** | `g2v3` (game), `m2v3` (allrounder) | `export/c2smoke-features/` | **`c2feat`** | 27 = 14 + 4 feature-diff + 9 feature-cost | §11 |

- **Colour:** per frame, flow-res Cb = 0.564 (B - Y) and Cr = 0.713 (R - Y) from the *unrounded* BT.601 Y, same pad and
  pyramid as the luma; the chroma is warped with the same h; only the first conv of each level grows. The zero test, sad and
  cut stay luma-only.
- **Features:** a shared conv3x3 3 -> 8 -> 4 extractor (clamp01, zero padding) runs on each RGB pyramid level; the four maps
  are matched like the luma (warped difference + a 9-channel cost).
- **Selecting them.** Vulkan: `Settings::learnedNet = LearnedNet::C2Smoke` (= 6) or `C2Feat` (= 7); fgtest / uitest / formattest
  / blurtest take `learnedNet=c2smoke|c2feat` / `--net c2smoke|c2feat`. WebGL: `motionNet: 'c2smoke'` or `'c2feat'` (load
  `nets/c2smoke.js` / `nets/c2feat.js`). `kLearnedNets` is **8**; `__fg.version` is **`'6'`**. The names `g2v2`, `m2v2`, `g2v3`
  and `m2v3` are reserved for the real nets (steps below). The zero test, static guard, fp16 / fp32 and the duplicate skip work
  per net as before.
- **Wrong net, wrong pipeline = loud failure.** Vulkan: a build error (see "Round 4 QC"). WebGL: a `RangeError` from
  `createEngine` / `setSettings` naming the net and the mismatch.
- **WebGL packing:** fp32 uses 7 RGBA32F textures for the 25 / 27-channel first conv, so match runs as two passes (WebGL2 has
  no 7 MRT); fp16 uses 4 RGBA32UI. Feature maps stay RGBA32F even in fp16. A per-frame **features cache** keyed on the
  `pushFrame` count (with `ML_NODUP`, so a duplicate pair cannot poison it) saves about 0.29 ms per steady-state pair; the
  colour planes are not cached. **Vulkan has no features cache** (view handles cannot prove the content is unchanged; it would save
  about 0.03 ms).
- **New files a host needs:** see "Files a host needs". New shaders: Vulkan `ml_planes.comp`, `ml_feat.comp` (and `ml_planes`
  per non-RGBA8 format); WebGL `learned/rgb0.frag.glsl` (63 files now, 14 under `learned/`; `lv0`, `down`, `match`, `conv` and
  `conv_h` changed).
- **Pipelines / memory:** switching to Learned creates `22 + 10 x 8` = **102** Vulkan pipelines (76 before). The embedded
  weights are 2,469,232 bytes (sum of the eight `kMlWeights_*` arrays; 1.56 MB for the six luma nets) and the net files are
  `c2smoke.js` 608 KB, `c2feat.js` 623 KB.

### Round 4 QC summary (independent re-run at `eeb12a9`)

- **Vulkan parity:** `mltest` **64 cases, 5960 assertions, 0 failed**; `synthcheck.py` **495/495** (162 of them c2smoke /
  c2feat), 0 failed. c2smoke and c2feat each ran all 9 pairs on the ref and FGR paths, fp32 and `learnedHalf`, zero test off
  and on, plus the duplicate cases.
- **WebGL parity:** `learned.mjs --nets c2smoke,c2feat` PASS on **SwiftShader** and on **ANGLE/Vulkan (RADV)**: 38 rows each
  (2 nets x 9 pairs x fp32/fp16, plus 2 RGBA16F rows), every row ok with the zero test, 8 duplicate cases PASS. The QC did not run
  ANGLE/OpenGL (worker: PASS).
- **Cuts** match `vec.json` on both backends (table below).
- **Loud failure:** confirmed in both backends (below).
- **Regression:** Vulkan V1 / V2 / V2 Quality / Learned (six luma nets, fp32 and half + zero test, guard, CrosshairV2) are
  **byte-identical** to `4bae33a`: 36 of 36 fgtest outputs over 2 1080p pairs. WebGL V1, V2 Performance, V2 Quality and
  e2v5 / e1v5 / e2v4 / e2v3 (fp32, fp16, with / without the zero test) are **bit-identical** to `4bae33a`: 18 of 18 configs
  x 2 synth times (SHA-1 of the output) on SwiftShader and on ANGLE/Vulkan. The `obs1` pair falls back (the same output for every
  learned net), so the learned part of that comparison rests on the `battlefield` pair.
- **Defaults:** `Settings{}.learnedNet == E2v5` (mltest asserts it; `framegen.h:305`); `__fg.defaultNet` `'e2v5'`,
  `__fg.smallNet` `'e1v5'`, `__fg.version` `'6'`.
- **Cost:** above the spec's +12% (below).

### Round 4 parity (QC re-run; tolerance for the variants: max |pf| 0.15 px in fp32)

**Vulkan** (worst over the 9 pairs, ref path = the vector's own luma, fp32; zero test on gives the same maxima):

| net | pf max (ref, fp32) | pair | fgr (library luma) pf max | ref half pf max | cost sad rel |
|---|---|---|---|---|---|
| c2smoke | **7.25e-2** | cut2 | 4.9 (far, luma-path differences) | 6.3 | <= 1e-6 (ref) |
| c2feat | **2.89e-2** | hd | 1.7 (far) | 5.0 | <= 1e-6 (ref) |

The `colour` and `features` pairs (which a port ignoring chroma / features fails) give pf max 0 (all pixels zeroed by the
luma-only zero test; the net field matches to 4.5e-7 / 5.4e-7 in `act_top0`). The synthcheck gates are all met. The stage alone
reaches inf dB. The thinnest margin is **c2smoke `hd` `refzt`: 76.7 dB against the 70 dB gate** (c2feat `hd` 81.6; every other
refzt row 86-inf; the non-zt ref rows are 83 dB or more against 45).
The worker's 6.2e-2 for c2smoke is 7.25e-2 here (still half of the tolerance): see "Discrepancies".

**WebGL** (SwiftShader / ANGLE-Vulkan; fp32 gates are SPEC §8's, with the variant's pf max 0.15):

| net, precision | pf max | zero-test pf max (agreeing px) | thinnest synth (zt on, vs synth(pf_zt)) | min zero-mask agreement | max sad rel |
|---|---|---|---|---|---|
| c2smoke fp32 | 5.3e-2 / 5.1e-2 (hd) | same | **78.7 / 73.8 dB** (hd, t 0.25; gate 70) | 99.9983 % | 1.6e-6 |
| c2feat fp32 | 2.7e-2 / 4.8e-2 (hd) | 2.4e-2 / 4.8e-2 | **80.2 / 80.4 dB** (hd, t 0.25; gate 70) | 99.9983 % / 99.9994 % | 2.9e-6 |
| c2smoke fp16 | 6.3 / 6.4 (cut2; reported only) | | 59.8 / 56.6 dB (gate 45) | 99.91 % / 99.86 % | 3.3e-4 / 4.8e-4 |
| c2feat fp16 | 5.0 / 6.1 (far; reported only) | | 62.1 / 59.0 dB (gate 45) | 99.91 % / 99.81 % | 5.1e-4 / 4.0e-4 |

Thinnest margin of round 4: **WebGL c2smoke fp32 with the zero test on `hd`, t = 0.25: 73.8 dB on ANGLE/Vulkan (78.7 on
SwiftShader) against the 70 dB gate** (e2v5 had 71.7 there in round 3). All 38 rows per backend pass.

### Round 4 cut decisions (net-only `cut` / with the zero test `cut_zt`, both backends)

| pair | c2smoke (vec.json) | c2feat (vec.json) | Vulkan | WebGL |
|---|---|---|---|---|
| `cut` | True / False | True / False | match | match (SwiftShader and ANGLE) |
| `cut2` | True / True | True / True | match | match |
| every other pair (`colour` / `features`, `far`, `hd`, `motion`, `odd`, `overlay`, `static`) | False / False | False / False | match | match |

### Round 4 loud failure (a net in the wrong pipeline)

- **Vulkan, compile time (read, not run):** `gpu/ml/embed_weights.py` takes the manifest's `variant` (an unknown one fails the
  build: `VARIANTS.index`) and asserts it against `config.colour` / `config.features` and its input channels (25 / 27 / 14); it
  emits `kMlVariant_<net>` / `kMlInCh_<net>` / `kMlExtr_<net>`. `framegen.cpp` has `kMlNetInfo {name, width, variant}`, and
  `static_assert(mlWidthsMatch(), ...)` (line 316) compares each net's embedded variant and input channels with the table's
  `kMlVariants[variant].inCh`. A colour net listed as luma, or the reverse, is a build failure, so it cannot reach a pipeline.
- **WebGL, run time (tested):** a throwaway check (not committed) changed the loaded net definition and called `createEngine`
  and `setSettings({motion:'learned', motionNet})`. Every case threw a `RangeError` from both, the control did not:

  | change | message (after `framegen: settings.motion 'learned': `) |
  |---|---|
  | c2feat, `variant` deleted | `net 'c2feat' (variant luma) has 27 input channels, the variant needs 14` |
  | c2smoke, `inChannels` = 14 | `net 'c2smoke' (variant colour) has 14 input channels, the variant needs 25` |
  | c2feat, `inChannels` = 25 | `net 'c2feat' (variant features) has 25 input channels, the variant needs 27` |
  | c2smoke, `variant` = 'bogus' | `net 'c2smoke' has variant 'bogus', which this framegen.js does not implement` |
  | e2v5, `variant` = 'colour' | `net 'e2v5' (variant colour) has 14 input channels, the variant needs 25` |
  | c2feat, `extractor` deleted | `net 'c2feat' (variant features) has no extractor weights` |
  | c2feat unchanged (control) | no error |

  `motionNet` outside `ML_NETS` is a `RangeError` too (the list now ends `'c2smoke','c2feat'`). The names `g2v2` etc. must be
  appended to `ML_NETS` first (step 3 below).

### Round 4 cost (RX 7900 XTX, ms of GPU time; QC re-measured)

`fgtest --bench 100 --mult 2`, 1080p, battlefield, zero test on, "TOTAL per pair" **min** of 3 runs (median range in brackets):

| net | fp32 | vs e2v5 | half | vs e2v5 half |
|---|---|---|---|---|
| V1 | 0.32 (0.46) | | | |
| e2v5 | 1.32 (1.40-1.48) | | 1.03 (1.24-1.26) | |
| c2smoke | 1.77 (1.95-2.03) | **+34 %** | 1.26 (1.35-1.41) | +22 % |
| c2feat | 1.93 (2.18-2.26) | **+45 %** | 1.41 (1.54-1.63) | +37 % |

The worker's figures (1.35 / 1.76 / 1.91 fp32, 1.03 / 1.24 / 1.40 half) agree within noise. This is **above the spec's +12 %**
(§10 predicts about +12.5 % conv MACs). The worker's explanation is level-0 memory traffic (the 25 / 27-channel first conv
reads more planes) and 36 KB of LDS in the new match shader; it is untuned and nobody has profiled it. The feature extractor
alone costs 0.061 ms. A real net that wants fewer ms needs a tuned match level 0, not new weights.

**WebGL** (worker figures, ANGLE/Vulkan, 1080p, estimate + synth per-pair min, fp16 / fp32; not re-measured by the QC): e2v5
1.72 / 2.92, c2smoke 1.90 / 3.37, c2feat 2.25 / 3.69 ms. These are uncached; the features cache saves about 0.29 ms per
steady-state pair.

### Round 4 limitations

- The **CrosshairV2 luma fill does not patch the chroma or RGB planes** the two variants read: with UI protection the colour
  and features variants still see the unfilled crosshair in their extra planes.
- `formattest check --algo learned --zt 1`: **3 failures with e2v5 and with c2feat, 7 with c2smoke** (QC re-run, matches the
  worker). All are synthetic-scene quality asserts: "CrosshairV2 leaves at most 8 artifact px around the plus, t = 0.5 and
  blurred" (2 with e2v5 / c2feat, 6 with c2smoke) and "no highlight left at the object's old places" (1 each). Not
  investigated; the stand-ins are 600-step nets.
- **Vulkan has no features cache** (WebGL has one).
- The zero test is luma-only by spec, so chroma-only motion is snapped to 0 (the `colour` pair is 100 % zeroed with the test
  on). Whether the real colour nets want `learnedZeroTest` off is for ml-fg's measurements; the toggle works per net.
- The stand-ins are 600-step nets: their quality says nothing about g2v2 / m2v2 / g2v3 / m2v3. Only port correctness is tested.

### Adding the real nets (g2v2, m2v2, g2v3, m2v3; no code change)

**Vulkan (`gpu/`):**
1. Copy ml-fg's `export/<name>/manifest.json` and `weights.bin` into `gpu/ml/<name>/`.
2. Append `<name>` to `ml_nets` in `gpu/meson.build` (same order as the enum).
3. Append a `LearnedNet` value and bump `kLearnedNets` (`gpu/framegen.h`; names in `learnedNetName()`).
4. Append `{"<name>", 24, MLV_COLOUR|MLV_FEATURES}` to `kMlNetInfo` and `ML_NET(<name>)` to `kMlNet` (`gpu/framegen.cpp`). A wrong
   variant is a `static_assert` failure.
5. Add the name to the four `--net` / `learnedNet=` help strings (fgtest, uitest, formattest, blurtest).
6. Add the mltest cases; if the export directory differs from the net name add it to `runDir()` (`gpu/mltest.cpp`) and
   `EXPORT_DIR` (`gpu/ml/synthcheck.py`).

**WebGL (`webgl/`):**
1. `python3 webgl/tests/make_nets.py <ml-fg>/export g2v2 m2v2 g2v3 m2v3` (`NAME=DIR` maps a differently named directory); it
   writes `webgl/nets/<name>.js` with `variant` and the extractor offsets.
2. Append the names to `ML_NETS` in `webgl/framegen.js` (and to the `HDR_NETS` list if they should be HDR-tested).
3. Run `webgl/tests/run.sh learned learned-gpu`.

## What changed in round 3 (for browser-frame-gen and gamescope-ritz)

Round 2 was `6133a4a`. Round 3 is at `d4a71e3` and later (gpu `9cad3e6`, webgl `019bd59`).

- **New default net `e2v5`, new small net `e1v5`.** ml-fg fine-tuned e2v4 / e1v4 further on more varied footage (the user's
  failure recordings, a friend's captures, 9 gameplay videos). The architecture, layout, tensor shapes, cost and memory are
  unchanged: e2 = width 24, 95,530 weights; e1 = width 12, 28,330 weights. ml-fg's 14-clip score at 2x with the zero test:
  e2v5 31.23 dB (+0.31 over `final`, +0.04 over e2v4, no clip worse), e1v5 31.05 dB (+0.12 over `final`, +0.06 over e1v4).
- **e2v4, e1v4, e2v3 and e1v3 stay selectable** for A/B, with byte-identical output (see "Round 3 regression").
- **Vulkan:** `LearnedNet::E2v5 = 4` (the `Settings{}` default) and `LearnedNet::E1v5 = 5` (the small net) are appended.
  `kLearnedNets` is 6, and `learnedNetName()` / `learnedNetFromName()` know `"e2v5"` / `"e1v5"`. Old values keep their numbers.
- **WebGL:** `ML_NETS` = `['e2v5', 'e1v5', 'e2v4', 'e1v4', 'e2v3', 'e1v3']`. The default `motionNet` and `__fg.defaultNet` are
  `'e2v5'`, `__fg.smallNet` is `'e1v5'`, and **`__fg.version` is `'5'`** (the default learned output changed).
- **Browser cost-guard order: e2v5 → e1v5 → V1.** A caller that picks nets through `__fg.defaultNet` / `__fg.smallNet` instead
  of names follows the change without code edits. It only has to load the two new net files.
- **Vulkan host build (gamescope-ritz) must embed the two new weight sets.** meson's `ml_nets` grew to
  `['e2v3', 'e1v3', 'e2v4', 'e1v4', 'e2v5', 'e1v5']` (in `LearnedNet` order). Add `gpu/ml/e2v5/` and `gpu/ml/e1v5/` to the
  `ml_weights` inputs. No shader changed: all 121 generated SPIR-V headers are byte-identical to `6133a4a`'s.
  - The weights buffer (binding 15) now holds six nets: 1.49 MB (1,486,320 bytes; 1.0 MB before).
  - Switching to Learned now creates 76 pipelines (16 + 10 per net; 56 before). uitest reports 85 pipelines after init (65
    before).
- **WebGL vendor set (browser-frame-gen) must add `nets/e2v5.js` (540 KB) and `nets/e1v5.js` (162 KB).** `motion: 'learned'`
  without `nets/e2v5.js` throws a `RangeError` naming the file. No shader file changed (62 files, all in `__fg.shaderFiles`).
- The zero test, the static guard, the duplicate skip, the cut rule (`sad > 30`) and V1's synthesis are unchanged.

### Round 3 QC summary

- **Vulkan parity:** `mltest` 44 test cases, 3990 assertions, 0 failed. `synthcheck.py` 333/333 PASS. This covers e2v5 (8 pairs)
  and e1v5 (5 pairs), each on the ref and FGR paths, fp32 and `learnedHalf`, zero test off and on, plus the duplicate cases of
  all six nets.
- **WebGL parity:** `learned.mjs` 80/80 rows and 24/24 duplicate cases PASS on SwiftShader and on ANGLE/Vulkan (RADV). That is
  all six nets at fp32 and fp16, each with and without the zero test, plus 6 RGBA16F rows (e2v5 / e2v4 / e2v3). `validate`
  209/209 OK. The QC did not re-run ANGLE/OpenGL; the worker reports PASS there.
- **Cut decisions** match `vec.json` on both backends and every path (table below).
- **Thinnest margin:** WebGL e2v5 fp32 with the zero test, `hd` pair, t = 0.25: synth **71.7 dB against the 70 dB gate**. It is
  the same on SwiftShader (71.67) and ANGLE/Vulkan (71.69). 7 of 518,400 mask pixels flip (99.9986 % agreement). e2v4 had 80.6 dB
  on that row (SwiftShader). It passes, but it has the least headroom of any gate. The same row in Vulkan (refzt) is 79.2 dB.
- **Regression:** V1 / V2 / e2v4 / e1v4 / e2v3 output is byte-identical to `6133a4a` (42 of 42 fgtest outputs over 3 pairs).
- **Cost:** equal to e2v4 / e1v4 (same shapes) in both backends.
- **Real footage:** e2v5 recovers most of e2v4's loss on obs1 13100 (34.00 vs 32.98 dB; V1 34.80) and gives up 0.13 dB on
  battlefield. e1v5 is the best of all on battlefield (36.66) and the worst on obs1 13100 (32.09).
- **uitest Learned:** 14 failures with e2v5, 12 with e2v4. The 2 extra are both "smooth noise, pan 30,0: V2 leaves no more
  artifact px than Off" checks. See Known issues (a).

### Settings and A/B for round 3 and 4

```sh
gpu/build/fgtest --set algo=learned,learnedNet=c2smoke[,learnedHalf=1][,learnedZeroTest=0] ...   # round 4: colour stand-in
gpu/build/fgtest --set algo=learned,learnedNet=c2feat ...                                        # round 4: learned-features stand-in
gpu/build/uitest|formattest|blurtest check --algo learned --net c2smoke|c2feat
```
```js
engine.setSettings({ motionVersion: 1, motion: 'learned', motionNet: 'c2smoke' });   // round 4: colour (needs nets/c2smoke.js)
engine.setSettings({ motionNet: 'c2feat' });                                          // round 4: learned features
```

Round 3:

```sh
gpu/build/fgtest --set algo=learned W H prev.rgba curr.rgba out.rgba                    # e2v5, zero test on, guard off
gpu/build/fgtest --set algo=learned,learnedNet=e1v5[,learnedHalf=1] ...                # the small net
gpu/build/fgtest --set algo=learned,learnedNet=e2v4 ...                                # round 2's default, byte-identical to 6133a4a
gpu/build/uitest|formattest|blurtest check --algo learned [--zt 0|1] [--guard 0|1] [--net e2v5|e1v5|e2v4|e1v4|e2v3|e1v3]
MLFG_EXPORT=<ml-fg>/export MLTEST_DUMP=<dir> gpu/build/mltest && python3 -I gpu/ml/synthcheck.py <dir>
webgl/tests/run.sh learned learned-gpu                                                  # SwiftShader, then ANGLE/Vulkan (FG_ANGLE=gl for GL)
```

```js
engine.setSettings({ motionVersion: 1, motion: 'learned' });          // e2v5 (__fg.defaultNet), fp16, zero test on, guard off
engine.setSettings({ motionNet: __fg.smallNet });                     // e1v5
engine.setSettings({ motionNet: 'e2v4' });                            // A/B against round 2's default
```

### Round 3 parity (QC re-run)

**Cut decisions.** `vec.json` gives `sad` / `cut` and `sad_zt` / `cut_zt`. The QC compared them on Vulkan (ref, FGR, ref + half,
FGR + half) and on WebGL (fp32, fp16; SwiftShader and ANGLE/Vulkan).

| net | pair | sad (net) | cut | sad_zt | cut_zt | Vulkan | WebGL |
|---|---|---|---|---|---|---|---|
| e2v5 | cut | 32.208 | true | 30.169 | true | same on every path; closest: ref + half + zt, sad 30.1656 | same |
| e2v5 | cut2 | 45.469 | true | 42.961 | true | same | same |
| e1v5 | cut | 31.257 | true | 29.114 | **false** | same: a cut net-only, **not a cut with the zero test** | same |
| e1v5 | cut2 | 53.052 | true | 50.794 | true | same | same |
| (e2v4) | cut | 28.305 | false | 26.445 | false | same (round 2) | same |
| (e1v4) | cut | 43.177 | true | 40.004 | true | same (round 2) | same |

Every other e2v5 / e1v5 pair (static, motion, far, odd, hd, overlay) is no cut, with and without the zero test, on every path.
e2v5's `cut` with the zero test sits only 0.17 above the threshold of 30, and e1v5's `cut` crosses it (31.3 → 29.1). `cut2` is
the safe cut check for both nets.

**Vulkan:** `mltest` 44 cases, 3990 assertions, 0 failed; `synthcheck.py` 333/333 PASS. The columns are as in the round-2 table
below ("ref+half" is the ref path with `learnedHalf`).

| net | ref pf max (px) | ref sad rel | zt mask agree: stage / ref / fgr / ref+half / fgr+half | synth dB ref / refzt / fgrzt / fgrzt+half | cuts |
|---|---|---|---|---|---|
| **e2v5** (8) | ≤ 3.4e-2 (hd) | ≤ 6.7e-6 | ≥ 99.9998 (hd: 1 px) / ≥ 99.9983 / ≥ 99.90 / ≥ 99.91 / ≥ 99.85 % | 83.6–93.5 / 79.2–98.1 / 64.3–83.7 / 62.6–78.8 | same (cut, cut2: true) |
| **e1v5** (5) | ≤ 7.6e-3 | ≤ 4.5e-6 | 100 / ≥ 99.9983 / ≥ 99.96 / ≥ 99.94 / ≥ 99.93 % | 88.5–91.5 / 81.3–92.7 / 65.5–66.6 / 64.5–65.6 | same (cut: true, cut_zt: false; cut2: true) |

- ref `ya`/`yc` are exact for every pair. ref sad_zt rel ≤ 3.8e-5 (e2v5) / 3.1e-5 (e1v5).
- The lowest mask agreement, 99.85 % (e2v5 `far`, FGR + half, 85 px), is on a shortcut path. Those paths are gated by the synth
  (≥ 45 dB), not by the mask. e2v4 had 99.91 % there.
- Duplicates (all 6 nets): identical frames give a zero field, SAD 0 and output = input. Mean |a−c| 0.5 is skipped, 1.0 runs the
  net.

**WebGL** (`webgl/tests/run.sh learned` and `learned-gpu`; every row PASS on both). Each cell is e2v5 · e1v5.

| | SwiftShader | ANGLE / Vulkan (RADV) |
|---|---|---|
| fp32 net pf mean / max (px) | ≤ 7.4e-5 / ≤ 3.9e-2 · ≤ 4.6e-5 / ≤ 9.7e-3 | ≤ 7.4e-5 / ≤ 3.7e-2 · ≤ 4.1e-5 / ≤ 7.0e-3 |
| fp32 sad rel / zt sad_zt rel | ≤ 8.1e-6 / ≤ 5.1e-5 · ≤ 3.1e-6 / ≤ 3.9e-6 | ≤ 7.0e-6 / ≤ 3.8e-5 · ≤ 2.4e-6 / ≤ 3.1e-6 |
| fp32 zt mask agreement | ≥ 99.9965 % (≤ 7 px) · 100 % | ≥ 99.9983 % (≤ 7 px) · 100 % |
| fp32 synth dB, net / zt (t = 0.5, 0.25) | 81.8–95.4 / **71.7**–96.5 · 88.3–92.1 / 88.8–94.5 | 82.1–94.2 / **71.7**–99.5 · 88.1–91.5 / 88.7–92.7 |
| fp16 synth dB, net / zt | 67.4–78.3 / 62.4–79.9 · 69.9–71.2 / 63.3–68.1 | 63.7–76.3 / 59.2–76.9 · 66.0–68.7 / 61.4–66.6 |
| fp16 sad rel / sad_zt rel (gate 1e-2) | ≤ 2.2e-4 / ≤ 4.6e-4 · ≤ 2.5e-4 / ≤ 4.2e-4 | ≤ 7.2e-4 / ≤ 7.1e-4 · ≤ 5.8e-4 / ≤ 3.2e-4 |
| fp16 zt mask agreement (reported) | ≥ 99.91 % · ≥ 99.93 % | ≥ 99.80 % · ≥ 99.91 % |
| RGBA16F input (e2v5 fp16), net / zt | 51.9–58.0 / 50.4–55.7 dB | 51.9–58.0 / 50.3–55.7 dB |
| cut / cut_zt decisions | identical to `vec.json` | identical |
| net field with the zero test (`ml.pfNet`) vs without | bit-identical | bit-identical |

The 71.7 dB is e2v5 `hd` (1920×1080) at t = 0.25 with the zero test. The same row is 77.0 dB at t = 0.5. The static-guard rows
(5/5) and the rows of the older nets pass as in round 2.

**Weights and defaults.**
- `gpu/ml/e2v5|e1v5/weights.bin` and the base64 in `webgl/nets/e2v5|e1v5.js` decode to the export's sha256 (e2v5 `7ee3da3a…`,
  e1v5 `357811ac…`). The manifests are byte-identical to the export's.
- `framegen.js` with all six net files loads in a bare VM context. It reports `__fg.version` `'5'`, `defaultNet` `'e2v5'`,
  `smallNet` `'e1v5'` and `defaultSettings().motionNet` `'e2v5'`, with 62 shader files listed and present.
- Vulkan: `mltest` "Learned settings" checks `Settings{}.learnedNet == LearnedNet::E2v5`. `framegen.h` names `E1v5` the small
  net (there is still no "small net" constant).

### Round 3 regression vs 6133a4a

Built side by side from `git archive 6133a4a`:
- SPIR-V: all 121 generated shader headers are byte-identical. Only `ml_weights.h` differs (six nets).
- fgtest at 1080p: 42 outputs are `cmp`-identical over obs2 5000, battlefield 2000 and obs1 13100. The configurations:
  - V1 at fs 1/2/4, `--bgra`, `--split`, `ui=crosshair`;
  - V2 Quality / Performance, V2 + `crosshairv2`;
  - Learned **`learnedNet=e2v4` against 6133a4a's default**, e2v4 with `learnedZeroTest=0`, e2v4 + guard, e1v4 + half, e2v3.
- `uitest check --algo learned --net e2v4` is line-for-line identical to 6133a4a's `uitest check --algo learned` (timing and
  pipeline-count lines excluded).
- WebGL: `framegen.js` changed only in the net list, the small net, the default `motionNet` and `__fg.version`.

### Round 3 real footage (fgtest 1080p 2x, PSNR vs the real middle frame, dB; QC re-run)

| pair | V1 | e2v4 | **e2v5** (default) | e2v5 zt off | e2v5 half | e2v5 + guard | e1v4 | **e1v5** | e1v5 zt off |
|---|---|---|---|---|---|---|---|---|---|
| obs2 5000→5002 | 27.14 | 27.13 | 27.11 | 27.00 | 27.11 | 27.11 | 27.17 | 27.19 | 27.06 |
| battlefield 2000→2002 | 36.09 | 36.33 | 36.20 | 35.55 | 36.20 | 36.21 | 36.31 | 36.66 | 35.88 |
| obs1 13100→13102 | 34.80 | 32.98 | 34.00 | 33.88 | 34.00 | 34.03 | 32.58 | 32.09 | 31.97 |

This reproduces the worker's table to the 0.01 dB.
- e2v5 closes most of e2v4's gap to V1 on obs1 13100 (−0.80 dB instead of −1.82).
- e1v5 is the best of all on battlefield (+0.57 over V1) and the worst on obs1 13100 (−2.71).
- The zero test still helps every moving pair, by +0.11 to +0.78 dB.

### Round 3 cost (RX 7900 XTX, ms of GPU time; QC re-measured)

**Vulkan**, `fgtest --bench 100 --mult 2`, battlefield 2000→2002, 3 runs each. At 1440p the frames are scaled to 2560×1440.
"b2b" is the back-to-back median per pair, and "profiled" is the per-pass sum of one profiled pair.

| | V1 | **e2v5** | e2v5 half | e2v4 | e2v4 half | **e1v5** | e1v5 half | e1v4 | e1v4 half |
|---|---|---|---|---|---|---|---|---|---|
| 1080p b2b | 0.30–0.46 | 1.35–1.66 | 1.02–1.28 | 1.33–1.44 | 1.03–1.04 | 0.53–0.54 | 0.49 | 0.56 | 0.49 |
| 1080p profiled | 0.35–0.36 | 1.37–1.40 | 1.06–1.12 | 1.39–1.40 | 1.08–1.09 | 0.57 | 0.51–0.53 | 0.59–0.60 | 0.53–0.54 |
| 1440p b2b | 0.45–0.46 | 2.70–4.20 | 1.75–1.98 | 3.14–3.18 | 1.75–2.04 | 0.90 | 0.80–1.01 | 0.90–0.93 | 0.80–1.02 |
| 1440p profiled | 0.52 | 2.72–3.74 | 1.82–2.32 | 2.83–2.85 | 1.85–1.87 | 0.96–0.98 | 0.82–0.91 | 0.94–0.98 | 0.82–0.87 |

- e2v5 = e2v4 and e1v5 = e1v4 within the run-to-run noise. The wide e2v5 1440p ranges come from single slow runs; the minima
  match.
- The worker's figures (1080p e2v5 1.33 / half 1.02, e1v5 0.53 / half 0.49) are consistent with these.

**WebGL**, ANGLE/Vulkan, `bench.mjs --gpu`, battlefield 1080p 2x, one run, estimate ms (back-to-back): e2v5 fp16 1.50 (e2v4
1.70), e1v5 fp16 0.91 (e1v4 0.94), e2v5 fp32 2.80 (e2v4 2.82), V1 0.76. They are equal within the noise.

Memory is unchanged: e2v5 = e2v4 = e2v3 and e1v5 = e1v4 = e1v3 (memory table below). Only the Vulkan weights buffer grows, to
1.49 MB.

## What changed in round 2 (since round 1; e2v4 / e1v4 were then the default / small net)

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

## Settings and defaults (current, round 4)

| | Vulkan `gpu/framegen.h` `Settings` | WebGL `setSettings` / `createEngine` |
|---|---|---|
| Select Learned | `motionVersion = MotionVersion::Learned` (= 2) | `motionVersion: 1, motion: 'learned'` (V2 ignores `motion`) |
| Net | `learnedNet`: **`LearnedNet::E2v5`** (= 4, default), `E1v5` (= 5, the small net), `E2v4` (= 2), `E1v4` (= 3), `E2v3` (= 0), `E1v3` (= 1), **`C2Smoke` (= 6, colour stand-in), `C2Feat` (= 7, features stand-in)** | `motionNet`: **`'e2v5'`** (default, `__fg.defaultNet`), `'e1v5'` (`__fg.smallNet`), `'e2v4'`, `'e1v4'`, `'e2v3'`, `'e1v3'`, **`'c2smoke'`, `'c2feat'`** |
| Net by name | `framegen::learnedNetName()` / `learnedNetFromName()` (`"e2v5"`, ...), `kLearnedNets` = **8** | the strings above; `nets/<name>.js` must be loaded |
| Zero test | `learnedZeroTest` = **true** | `learnedZeroTest: true` |
| Static guard | `learnedStaticGuard` = **false** | `learnedStaticGuard: false` |
| Precision | `learnedHalf` = **false** (fp32 activations) | `motionPrecision: 'fp16'` (**default**) / `'fp32'` |
| Runtime? | all runtime (`setSettings`, no GPU wait); a net, half or zero-test change applies from the next estimate; the guard from the next synth | a change of `motion` / `motionNet` / `motionPrecision` / `learnedZeroTest` **drops the estimate** (synth shows the real frame until the next `estimate()`); `learnedStaticGuard` applies from the next synth |
| Needs | `flowScale` 2 (else `settingsError()`), every `Format` | `flowScale` 2 (else `RangeError`), RGBA8 and RGBA16F |

**A/B toggles**

```sh
# Vulkan (fgtest)
gpu/build/fgtest --set algo=learned W H prev.rgba curr.rgba out.rgba                               # e2v5, zero test on, guard off
gpu/build/fgtest --set algo=learned,learnedZeroTest=0 ...                                         # the net's raw field (+ learnedNet=e2v3: round 1's output, byte-identical)
gpu/build/fgtest --set algo=learned,learnedStaticGuard=1 ...                                      # + static guard
gpu/build/fgtest --set algo=learned,learnedNet=e1v5[,learnedHalf=1] ...                           # the small net
gpu/build/fgtest --set algo=learned,learnedNet=e2v4 ...                                           # round 2's default (byte-identical to 6133a4a)
gpu/build/fgtest --set algo=learned,learnedNet=c2smoke ...                                         # round 4: colour stand-in (c2feat: features)
gpu/build/uitest|formattest|blurtest check --algo learned [--zt 0|1] [--guard 0|1] [--net e2v5|e1v5|e2v4|e1v4|e2v3|e1v3|c2smoke|c2feat]
MLFG_EXPORT=<ml-fg>/export MLTEST_DUMP=<dir> gpu/build/mltest && python3 -I gpu/ml/synthcheck.py <dir>
python3 -I gpu/ml/guardcheck.py W H a.rgba c.rgba off.rgba on.rgba DIR/pair_0000.fgmaps [--t T]    # guard vs static_guard.py
```

```js
// WebGL
engine.setSettings({ motionVersion: 1, motion: 'learned' });          // e2v5, fp16, zero test on, guard off
engine.setSettings({ learnedZeroTest: false });                       // A/B: no zero test (drops the estimate)
engine.setSettings({ learnedStaticGuard: true });                     // static guard (next synth)
engine.setSettings({ motionNet: __fg.smallNet });                     // e1v5
engine.setSettings({ motionNet: 'e2v4' });                            // A/B: round 2's default
engine.setSettings({ motionNet: 'c2smoke' });                         // round 4: colour variant (nets/c2smoke.js loaded)
engine.setSettings({ motionNet: 'c2feat' });                          // round 4: learned-features variant (nets/c2feat.js loaded)
```

A gamescope host maps its own settings to `Settings::learnedZeroTest`, `learnedStaticGuard` and `learnedNet`. Use
`learnedNetFromName()` for a string setting.

## Files a host needs

**Round 4 additions (read first):**
- **gamescope-ritz (Vulkan):** embed **`gpu/ml/c2smoke/`** and **`gpu/ml/c2feat/`** (`manifest.json` + `weights.bin`; they keep the stand-in
  names until the real nets arrive) and add the shaders **`gpu/shaders/ml_planes.comp`** and **`gpu/shaders/ml_feat.comp`**; changed
  files: `ml_common.glsl`, `ml_match.comp`, `ml_pyr.comp`. `ml_planes` is also built per non-RGBA8 format
  (`ml_colour_srcs = ['shaders/ml_dup.comp', 'shaders/ml_guard.comp', 'shaders/ml_planes.comp']`). meson `ml_nets` is now
  `['e2v3', 'e1v3', 'e2v4', 'e1v4', 'e2v5', 'e1v5', 'c2smoke', 'c2feat']` (the `LearnedNet` order) and `gpu/ml/embed_weights.py`
  is the new variant-aware one. A host that does not embed the two new nets fails the `kMlNet` static_assert (`kLearnedNets` is 8).
  Without `ml_planes` / `ml_feat`, Learned is refused as a whole (`kMlNotBuilt`).
- **browser-frame-gen (WebGL):** vendor **`nets/c2smoke.js`** (608 KB) and **`nets/c2feat.js`** (623 KB) and the new shader
  **`shaders/learned/rgb0.frag.glsl`** with the changed `learned/{lv0,down,match,conv,conv_h}.frag.glsl` (63 files, all in
  `__fg.shaderFiles`; the set is checked: every listed file exists). `__fg.version` is `'6'`. A net that is selected but not loaded throws
  a `RangeError` naming the file; a loaded one that does not match its variant throws one naming the channel counts.
- Everything below is the round-3 list; the net lists there become eight.

**gamescope-ritz (Vulkan), on top of round 1:**
- Shaders (round 2; round 3 changed none): `gpu/shaders/ml_zero.comp`, `ml_guard.comp`, and `ml_guard` with
  `-DFG_FORMAT_<FMT>=1` for every non-RGBA8 format in use (meson `ml_colour_srcs = ['shaders/ml_dup.comp',
  'shaders/ml_guard.comp']`). The full Learned list is `ml_dup`, `ml_reduce`, `ml_pyr`, `ml_match`, `ml_conv`, `ml_out`,
  `ml_zero`, `ml_guard` (+ `ml_common.glsl`).
- Weights: `gpu/ml/e2v4/`, `gpu/ml/e1v4/` (round 2) and **`gpu/ml/e2v5/`, `gpu/ml/e1v5/` (round 3)** (`manifest.json` +
  `weights.bin`, sha256-identical to ml-fg's export). `embed_weights.py` gets all six nets in `LearnedNet` order: meson
  `ml_nets = ['e2v3', 'e1v3', 'e2v4', 'e1v4', 'e2v5', 'e1v5']`. A host that keeps round 2's four-net list builds a library
  whose default net has no weights, so it must embed the two new sets.
- The weights buffer (binding 15) now holds all six nets: 1.49 MB, uploaded in the first Learned estimate's command buffer as
  before.

**browser-frame-gen (WebGL):**
- Vendor `webgl/framegen.js`, `webgl/shaders/` (62 files, all in `__fg.shaderFiles`; 13 under `learned/`) and `webgl/nets/`
  (`e2v5.js` 540 KB, `e1v5.js` 162 KB, `e2v4.js`, `e1v4.js`, `e2v3.js`, `e1v3.js`). The QC loaded `framegen.js` and the six
  nets in a bare VM context with no fs, fetch or DOM. Every listed shader exists, no file under `shaders/` is unlisted, and the
  weights' sha256 match the export (e2v5 `7ee3da3a…`, e1v5 `357811ac…`, e2v4 `ba6981cf…`, e1v4 `28128da2…`).
- Load at least `nets/e2v5.js` (the default) and, for a fallback, `nets/e1v5.js`, in any order with `framegen.js`. Load
  `nets/e2v4.js` / `nets/e1v4.js` too for an A/B against round 2.

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

## Measured parity (round 2 QC re-runs; round 3 above)

Tolerances (SPEC §8, the same in round 3): for an exact fp32 port, pf mean ≤ 2e-4 px, max ≤ 5e-2 px, sad rel ≤ 1e-3, the same
cut and synth ≥ 80 dB. For the zero test: mask agreement ≥ 99.9 %, pf compared where the masks agree, sad_zt rel ≤ 1e-3, the
same cut_zt and synth ≥ 70 dB. With shortcuts (fp16, FGR luma), synth ≥ 45 dB and the middle-frame PSNR within 0.05 dB.

The round-3 run of the same suites (44 mltest cases, 333 synthcheck rows, 80 + 24 `learned.mjs` rows) reproduced every row of
the older nets below as PASS. The figures below are round 2's.

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

## Real footage, round 2 (fgtest 1080p 2x, PSNR vs the real middle frame, dB; QC re-run)

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

## Cost, round 2 (RX 7900 XTX, 1080p 2x, ms of GPU time; QC re-measured)

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
- Switching to Learned creates **102** pipelines (22 + 10 per net, eight nets) at once (76 in round 3, 56 in round 2).
- Round 4: `learnedNet` also takes `C2Smoke` / `C2Feat` (colour / features variants; they need `ml_planes` and `ml_feat`).
  Expose them in the same picker as the other nets. The variant is fixed per net, a host never sets it.
- `flowScale` must be 2.
- UI protection: the net and the zero test read `P_LUMA` (the clean copies), and the guard reads the frames the synthesis read
  and runs before the UI paste-back.
- Expose `learnedZeroTest` (default on), `learnedStaticGuard` (default off) and `learnedNet` (`e2v5` default, `e1v5` small;
  `e2v4` / `e1v4` for an A/B against round 2) for the morning A/B test. A host that falls back to a cheaper net picks `E1v5`.

**browser-frame-gen (WebGL):**
- Vendor the files listed above and load `nets/e2v5.js` (+ `nets/e1v5.js`; `nets/e2v4.js` / `nets/e1v4.js` for an A/B).
- The cost guard steps e2v5 → e1v5 → V1. Through `__fg.defaultNet` / `__fg.smallNet` it needs no code change. For a colour or
  features net pick the fallback the same way: they cost about 20-45 % more than e2v5, so a guard that triggers on e2v5's cost
  triggers sooner on them.
- Round 4: `__fg.version` is `'6'`; `motionNet: 'c2smoke' | 'c2feat'` with `nets/c2smoke.js` / `c2feat.js` loaded.
- `__fg.version` is `'5'` (`'4'` in round 2). The fields `__fg.defaultNet`, `__fg.smallNet` and `debugRead('ml.pfNet' |
  'ml.ztCost' | 'ml.zt' | 'sg')` are unchanged since round 2.
- The first `motion: 'learned'` compiles 22 programs (17 in round 1).
- The default precision is still **fp16**.
- `estimate()` with learned leaves a uniform buffer bound, as in round 1.
- Not tested on Firefox or ANGLE/D3D11. There is no UI protection in the WebGL backend.

**API differences between the backends** (recorded, not changed):

| | Vulkan | WebGL |
|---|---|---|
| How it is selected | its own `MotionVersion::Learned` (= 2) | `motion: 'learned'` inside `motionVersion` 1 (V2 ignores it) |
| Net | `learnedNet` (`E2v5` default / `E1v5` / `E2v4` / `E1v4` / `E2v3` / `E1v3` / `C2Smoke` / `C2Feat`) | `motionNet` (`'e2v5'` default / `'e1v5'` / `'e2v4'` / `'e1v4'` / `'e2v3'` / `'e1v3'` / `'c2smoke'` / `'c2feat'`) |
| Variant check | compile time (`static_assert` + `embed_weights.py`) | run time `RangeError` (`createEngine` / `setSettings`) |
| Features cache | none | per-frame, keyed on `pushFrame` count (c2feat) |
| Small net | `LearnedNet::E1v5` (documented in `framegen.h`; no "small net" constant) | `__fg.smallNet` (`'e1v5'`) |
| Precision | `learnedHalf` (bool, **default false = fp32**) | `motionPrecision` (`'fp16'` **default** / `'fp32'`) |
| Zero test / guard | `learnedZeroTest` (true) / `learnedStaticGuard` (false) | the same names and defaults |
| Zero-test switch | applies from the next estimate, keeps the current one | drops the current estimate |
| Luma | FGR 14-bit (`P_LUMA`) | cv2 5.0.0 15-bit (exact with the vectors) |
| Weights | compiled in | separate scripts, must be loaded |
| Pass structure of the 25 / 27-channel match | one pass | two passes at fp32 (7 RGBA32F textures), one at fp16 (4 RGBA32UI) |

## Known issues

- (a) The synthetic quality tests are V1/V2-tuned and were not loosened. All counts are QC re-runs.
  - Round 3: `uitest check --algo learned` fails **14** checks with e2v5 (the default) and 11 with e1v5. With `--net e2v4` it
    fails 12, line-for-line the same as `6133a4a`'s default. `--zt 0` gives the same counts (14 / 12), so the zero test is not
    the cause.
  - The 2 extra e2v5 failures are exactly "smooth noise, pan 30,0, default settings" and "... host settings": "V2 leaves no
    more artifact px than Off". The worker's explanation (a better Off baseline) is only half of it. The artifact px at t = 0.5
    / 0.25 are:
    - e2v4: Off 5 / 7 (host 5 / 16), V2 1 / 5;
    - e2v5: Off 1 / 8 (host 2 / 14), V2 3 / 4.

    So Off got better at t = 0.5 *and* CrosshairV2 on top of e2v5 leaves 2 more px there (3 vs 1). V2 + fill stays 0 / 0. It is
    a 1–3 px difference on one synthetic scene, inside the "CrosshairV2 leaves more pixels than with V1" limit already known.
  - With `--guard 1`, e2v5 also fails 14: its guard adds no failure. e2v4 + guard still fails 13; the extra one is "without
    re-entry the arm needs uiStillFrames = 8 still frames ...".
  - The other failing checks are "control, Off shows the doubled crosshair" and "V2 [+ fill] leaves at most 1 artifact px".
  - `formattest --algo learned` fails 3 with e2v5, e2v4 and e1v5. `blurtest --algo learned` passes with e2v5 and e2v4 and
    fails 4 with e1v5 (as e1v4).
  - Round 2 counts follow.
  - `uitest check --algo learned` fails 12 checks with e2v4 (10 with e2v3, 11 with e1v4), and the zero test does not change
    the count.
  - With `--guard 1` it fails 13. The extra one is "without re-entry the arm needs uiStillFrames = 8 still frames ...".
  - Correction to the worker's note: the "control, Off shows the doubled crosshair" failure is **not** caused by the zero test.
    It fails the same way with `--zt 0`, and with e2v3 in round 1.
  - `formattest --algo learned` fails 3 (+1 with the guard). `blurtest --algo learned` now **passes** with e2v4 (e2v3 failed
    4) and fails 4 with e1v4 (the blur edge position).
- (b) By design, the guard is applied once to a blur average. So `blurtest --guard 1` fails its 3 "blur == average of the
  recordSynth sub-frames" checks (up to 36 levels on 0.5–18k bytes per configuration), and formattest has the matching extra
  failure. Learned's "scene cut forced" variant (sceneCutSad 0.5) is not a cut on that synthetic scene: its guarded differences
  equal the default variant's. So the guard's fallback skip for blur is covered only by the flick variants, which pass.
- (c) Thin margin (round 3): WebGL e2v5 fp32 with the zero test on `hd`, t = 0.25, is **71.7 dB against the 70 dB gate** on
  SwiftShader and ANGLE/Vulkan (e2v4: 80.6). 7 of 518,400 mask pixels flip there, at the `C(0) − 1 = C(h)` boundary where fp32
  summation order decides, and each flipped vector moves a few output pixels a lot (engine max error 48 levels). The Vulkan
  ref path gives 79.2 dB on that row, and e2v5 hd stage-alone flips 1 px (e2v4: 3). A future net could cross the gate on
  this row without a port bug.
- (d) Hosts must add the new shaders and the ml dirs, or Learned is refused (Vulkan) or the default net is missing (WebGL).
  Round 3 adds only `gpu/ml/e2v5`, `gpu/ml/e1v5` (meson `ml_nets`) and `webgl/nets/e2v5.js`, `e1v5.js`.
- (e) The whole-frame duplicate test holds a small moving cursor on a still desktop (unchanged).
- (f) The static guard does not fix the dragged HUD where the field around it moves more than ~1 px (the motion gate,
  `static_guard.py` demo (a) asserts this as a known limit). The zero test is what fixes the HUD.
- (g) Static HUD is still not as clean as V1. Glyph edges on obs2's "$69" stay notched with the zero test (see "HUD").
- (h) obs1 13100: e2v5 is 0.8 dB below V1 (e2v4 1.8). e1v5 is 2.7 dB below V1, worse than e1v4 (2.2), although it is the best
  net on battlefield.
- (i) The `far` and `cut` (cut2) vector pairs cannot test the synthesis end to end (global fallback / cut).
- (j) Pre-existing, not Learned: syncval `SYNC-HAZARD-WRITE-AFTER-WRITE` on binding 2 with `ui=crosshair` (round 1 (g)). Syncval
  was not re-run in round 2; GPU-AV was.
- (k) WebGL: not tested on Firefox or D3D11, V1 only, no UI protection.
- (l) The WebGL memory figures are the workers'. Round 2's 1440p timings are the workers'; round 3's QC measured Vulkan 1440p
  itself (table above).
- (m) Round 3 QC did not re-run GPU-AV / syncval, the static-guard `guardcheck.py`, the HUD crops or WebGL `golden` /
  `xbackend` / `hdr` / `blur`. Round 3 changed no shader and no engine code besides the net lists and defaults, and V1 / V2 /
  old-net outputs are byte-identical to `6133a4a`. ANGLE/OpenGL was not re-run (worker: PASS).
- (n) e2v5's `cut` vector sits 0.17 above the cut threshold with the zero test, and e1v5's crosses it (cut net-only, no cut with
  the zero test). Both backends agree with `vec.json`, but a cut test built on that pair is fragile; use `cut2`.
- (o) Round 4 cost: the colour and features variants cost 34 % / 45 % more than e2v5 at fp32 (22 % / 37 % at half), above the
  spec's +12 %. The worker's explanation (level-0 traffic, 36 KB LDS) is unverified and nothing is tuned.
- (p) Round 4: the CrosshairV2 luma fill does not patch the chroma / RGB planes; `formattest --algo learned` fails 7 (c2smoke) /
  3 (c2feat) checks against 3 (e2v5); Vulkan has no features cache (WebGL has).
- (q) Round 4 QC did not re-run GPU-AV / syncval, the HUD crops, `guardcheck.py`, WebGL `golden` / `xbackend` / `hdr` / `blur`
  (only the direct bit-diff of V1 / V2 / old nets against `4bae33a`), ANGLE/OpenGL, WebGL cost or the stand-in nets on real
  footage (they are 600-step smoke nets).
- (r) The WebGL fp32 thin margin has moved: `hd`, t = 0.25, zero test on: c2smoke 73.8 dB on ANGLE/Vulkan (gate 70).
- (s) The round-3 text elsewhere says "six nets" and `__fg.version` `'5'`; read eight nets and `'6'`. `webgl/README.md` (not
  edited by this QC) still lists `'5'` as the version and six nets.
