# motion/: measuring a motion estimate (task 155)

Tools that show and measure the motion field of the library's estimators, and the baseline of `MotionVersion::V3` taken with them.
They exist to judge a better estimator (a later "V3.5") with exactly the same instruments: every tool takes `--fgtest PATH` and a
`--set` string, so a changed build or changed settings are one rerun away. Python 3 with numpy and cv2 only (no other dependency;
the readers of `gpu/v3flow.py` and `gpu/fgmaps.py` are imported, not copied). Nothing in `gpu/` was changed.

## 1. How V3 finds motion

Numbers are for 1080p at the default `flowScale` 2 (fs). Section numbers are those of `gpu/V3.md`; shader lines are at this commit.
Everything runs once per real pair, in both directions at once (F01: content at p in prev is at p + F01(p) in curr; F10 the other way).
The search never reads the previous pair (no temporal state; `gpu/V3.md` 3.8 "Temporal state"), only the two frames of the pair.

**Grids.** The flow grid is `fw x fh = 960 x 540` (one flow px = 2x2 px). A pyramid halves it five times: 960x540, 480x270, 240x135, 120x68, 60x34, 30x17
(n = 6 levels, the top level has a short side <= 32; `gpu/V3.md` section 2 "Levels"). The matching unit is a *block*: 8x8 px at level 0 (= 16x16 full-resolution px,
block grid 120x68), 4x4 px at levels >= 1 (level 1: 120x68 blocks, each = 8x8 flow px; level 2 60x34; 3 30x17; 4 15x9; 5 8x5).
What is matched are high-pass features (YCbCr minus a 5x5 mean, plus a small raw-luma term, `gpu/V3.md` section 2 "Matching cost").

**Passes, in order** (`framegen.cpp:3084-3170`, the estimate; `gpu/V3.md` section 1 has the diagram):

| # | pass (shader) | what it outputs | per-block constant or per-pixel |
|---|---|---|---|
| 1 | `v3_planes`, `v3_hist`, `v3_pyr` | colour planes at flow resolution, brightness statistics (lighting gain), the 6-level feature pyramid | per pixel (features), no vectors yet |
| 2 | `v3_search_top` (level 5, 8x5 blocks of 4x4) | exhaustive search over every integer v in [-8, 8]^2 (289 candidates), 1 iteration, both directions; no prior | **per-block constant, integer** (level-5 px = 32 flow px per level px: one step is 32 flow px, i.e. +-256 flow px = +-512 px reach) |
| 3 | `v3_search` levels 4 .. 1 (2 iterations each) | per level: candidates 16 (see below), then a ring of the 8 integer neighbours of the best; vector scaled x2 from the level above | per-block constant, integer in level px (at level 1: integer = 2 flow px = 4 px steps) |
| 3a | `v3_wide` after level 4 | only for blocks whose cost > 7 levels, or forward-backward error > 3, or ambiguous: a second search on the coarse grid v in {-24, -22 .. 24}^2 (625 candidates) at level 4 = +-384 flow px; replaces the vector if clearly better | per-block constant |
| 3b | `v3_stats` after level 3 (LH) | up to 4 dominant motions (histogram + mean shift) and an affine camera model (6 parameters, robust fit), as candidates for the levels below | global |
| 4 | `v3_search` level 0 (8x8 blocks, 1 iteration, no apron) | candidates: own parent x2 and the parents of the 4 neighbours x2, then the integer ring, then **Lucas-Kanade sub-pixel** (2 iterations, clamp +-1 px per step, accepted only if the cost drops) | **per-block constant, sub-pixel (float)**: this is `BLK`, the block field, 120x68 vectors |
| 5 | `v3_dense` (`v3_dense.comp:66`) | tiles of 8x8 flow px *between 4 block centres*: if the 4 block vectors agree to < 0.5 flow px, every pixel gets the bilinear blend of the 4; otherwise per pixel the cost of each of the 4 vectors over a 5x5 window decides a soft pick `w_k = bil_k * exp(-max(0, c_k - c_min) / 1.5)` (`TAU`, line 24, 210) | **per pixel (float), the first per-pixel field: `FD`** (960x540) |
| 6 | `v3_occl` | forward-backward error e = abs(F01(p) + F10(p + F01(p))), occlusion o, "garbage" g (box 9x9 of o) | maps only |
| 7 | `v3_fill` | in occluded strips the vector is replaced by a colour-guided blend of the nearest non-occluded vectors in 4 directions (walks 12 px, then probes out to 96 px); a band at the frame edge is extrapolated | per pixel: **`FF`** = the field the synth uses |
| 8 | `v3_decide`, then per output time `v3_project` | per-pair scene-cut / fallback decision; project: fixed-point gather of FF to time t (2 iterations), smoothing of the vector over a 9-tap ring where o is high, pull of isolated garbage vectors towards the camera model | per pixel, at flow resolution, half floats |

**Precision.** Block vectors are integer in level px down to level 1 (so the integer step of the coarse search is 2 flow px = 4 px at level 1), then
level 0 refines to sub-pixel with a Lucas-Kanade step over the 8x8 window. `FD` and `FF` are float32 per pixel; `v3_project` stores half floats.
**Search range per level**: top level +-8 level px exhaustive (= +-256 flow px); every lower level only a ring of +-1 level px around the best of its candidates (plus
the candidates inherited from above, which carry large motion down); the wide search reaches +-384 flow px at level 4 on flagged blocks.

**Candidates a block tests** (`gpu/V3.md` 3.5, `v3_search.comp`, `v3_search.glsl:98`): at levels 4 .. 1, iteration 0: its own parent's vector x2 (no penalty), the 3 other nearest parents x2,
zero, the camera model at the block centre (only below level 3), the 4 dominant motions (below level 3, capped penalty 0.5), and the *other direction's* reverse vector (-F10 at the target).
Iteration 1: its own result, the 4 neighbours' results (left, right, up, down), the reverse vector at the current target, zero, dominant motions, camera model.
Level 0: parents of itself and the 4 neighbours, then ring, then LK; **no** camera-model / dominant candidates at level 0 ("they put lattice spots into the background"). The previous pair is never a candidate.
Every candidate except its own parent pays `0.3 levels per level px` of distance from the parent vector (`V3_LAMBDA`, `v3_est.glsl:62`).

**Smoothing / regularisation** that exists: the parent prior above (the only regulariser of the search: flat regions tie and take the parent's / neighbours' vector);
the bilinear blend and the soft pick of `v3_dense` (the only place where a vector varies inside a block); the occlusion fill; in `v3_project` a 9-tap box smoothing of the vector only where the occlusion flag is high, and the pull of
isolated garbage vectors to the camera model. There is **no** global smoothness term, no median filter and no edge-aware smoothing of the field itself, and the per-block vector is never changed by its neighbours except through the candidate list and the dense soft pick.

**Settings that influence any of this**: `flowScale` (1, 2, 4; changes every grid size above, block = 8 flow px), `v3ColourWindow` (hook j: colour-similarity weights of the matching windows of search and dense),
`v3Guard` (V2 Quality's motion replaces or corrects the dense field), `hudBonus` (zero-vector reward), `globalFallback` (via the failMix, replaces vectors by the camera model in garbage regions), `v3SceneCut`.
`v3Smear`, `v3OcclSoft`, `v3Unsure`, `v3Blend`, `v3DetailSplit` act only in the synthesis. `searchRadius`, `searchRadiusTop`, `pyramidMinSide`, `subpixel`, `searchPenalty`, `trustLow/High`, `smoothBonus` are ignored by V3.

## 2. Conventions

- **Displacement**: always the full prev -> curr displacement in **full-resolution px**: content at p in prev is at p + flow in curr; +x right, +y down.
  V3's F01 (a `.v3db` field, flow px, on prev's grid) is multiplied by `flowScale`; V1 / V2 / Learned dump half vectors on the *middle* frame's grid, which `common.fields` doubles
  (grid `mid`). The truth is evaluated analytically on the same grid at the flow pixel's centre `fs*m + (fs-1)/2`. Checked with a pure pan of 64 px (`forza_pan_l`, true mean flow (60.0, 21.9) px):
  V3's mean estimate (57.3, 21.3), so sign and scale agree.
- **Test pairs** (`mkpairs.py`): `prev(p) = S(M(+1/2)(p))`, `curr(q) = S(M(-1/2)(q))`, `mid = S` (unresampled) with Lanczos-4 resampling, 1920x1080. The prev / curr frames are therefore a little softer than S.
  `flow.npy` = M(+1/2)(s) - M(-1/2)(s) per pixel s of the middle frame (float32 [1080,1920,2]); `valid.npy` false where the warp sampled outside the source (4 px margin), within 3 px of a cut-out's outline, and where a cut-out hides the pixel in prev or curr.
  Largest per-pixel displacement is set numerically to 8 / 24 / 64 px (`s`/`m`/`l`). 8 cases x 3 strengths x 3 sources (Forza `racing_forza6_a/000340`, cs2 `fps_cs2_a/000840`, Minecraft `minecraft120_02/000964`) = 72 pairs, 3 GB. `spec.json` holds the transform; `truth()` in `mkpairs.py` recomputes the field on any grid.
- **EPE**: end-point error in full-res px over valid pixels; "tex" = only over textured pixels (the 5x5-dilated top decile of the source's Sobel magnitude, as the edge mask of `blend/edge`), where motion is observable at all. A flat sky or road cannot be measured and is not a failure of the picture.
- **Blockiness** (`common.blockiness`): of the field's curvature q(x) = |F(x+1) - 2F(x) + F(x-1)| (vector norm, along x and y), fold the mean of q by position modulo the block size B (V3 level 0: B = 8 flow px = 16 px at `flowScale` 2; also 8 and 32 px): Q[j]. Add a 0.1 px floor to every Q[j]. s = (largest two neighbouring Q[j]) / (sum of Q); blk = (s - 2/B) / (1 - 2/B), clipped at 0.
  Spread-out curvature (a smooth true field, noise, real texture edges unrelated to the grid) gives s = 2/B, blk 0; a field that is constant per block, or bilinear between block centres, puts all curvature on one grid line: blk 1.
  The grid phase is free (V3's dense field kinks at block *centres*, its block field at block boundaries; the phase is reported). Positions near a true motion edge (the cut-outline of `chase` / `orbit`: true step above 0.75 px per flow px, dilated by 8 flow px) and invalid pixels are excluded, so real motion edges do not count.
  `ctrl` = the same at a period B + 3 that is no harmonic (should be near 0 when the structure is the grid). `step_px` = how many px of extra curvature the grid line carries.
  **Reference values** (in every table): the true field scores 0.000 (all strengths), its per-block-constant version (aligned 8x8 flow px blocks, block mean) 0.18 .. 0.83 rising with strength (at `s` the true gradient x 16 px is below the 0.1 px floor, so the check is weak there; 0.80 at `l` for a zoom).
  Caveat: blk is a share, not a size: a few large wrong blocks in a textured field read lower than many small steps; read it with `step_px` and the sheets.
- **PSNR**: as `blend/edge/score_edge.py`: mse pooled over pixels and RGB, `10 log10(255^2/mse)`; the edge pixels are the 5x5-dilated top decile of the real frame's Sobel magnitude (luma 0.299/0.587/0.114, replicate border). Here the picture is the library's final output (fgtest, `v3Blend` 0, t = 0.5), whereas the figure of `blend/probe` (31.24 / 26.75) is the weight stage before clamp / smear / static mask.
- **Colour wheel / sheets**: see `maps.py --help`; hue = direction, saturation = magnitude (white = no motion), a fixed scale per pair (8 / 24 / 64 px for the synthetic strengths; 32 or 64 px for the real scenes), the same for any `--set`.

## 3. Running the tools

All take `--fgtest PATH` (a built `fgtest`, e.g. `meson setup B gpu --buildtype=release; meson compile -C B fgtest`), `--set 'k=v,..'` (an fgtest `--set` string), `--out DIR`, `--pairs LIST.json` (repeatable), `--only a,b` (name filter). fgtest runs as `flock /home/mo/tmp/fg-handoff/gpu.lock ccorral -- fgtest --window 1 --dump ...` in chunks of 4 pairs, the dumps are deleted per chunk. Run them with `ccorral --` and from the repository root; python with numpy and cv2 only.

    PY=/home/mo/Schinken/Programming/Local/ml-fg/.venv/bin/python
    $PY motion/mkpairs.py --out PAIRS                     # 72 pairs, PAIRS/pairs.json (3 GB)
    $PY motion/mkreal.py heldout --out held.json           # the 23 held-out real pairs
    $PY motion/mkreal.py add --out scenes.json NAME CLIP PREV CURR [--mid N] [--scale PX]   # a real scene
    $PY motion/measure.py --fgtest F --set algo=v3 --out OUT --pairs PAIRS/pairs.json --pairs held.json   # OUT/measure_<label>.{md,json}
    $PY motion/maps.py    --fgtest F --set algo=v3 --out OUT --pairs PAIRS/pairs.json --only forza_      # OUT/sheets/*.png, OUT/png/*

A full `measure.py` run over the 72 + 23 pairs takes about an hour on the shared GPU (most of it is fgtest and the files, not the metrics).
`measure.py` JSON holds every number per pair and per stage (V3: `BLK` level-0 block field, `FD` dense, `FF` after fill / the field the synth uses, `L1..L5` the final block fields of the coarser levels).
Dumps: V1 and Learned and V2 Quality write a per-pixel half-vector field (`.fgmaps`), V2 Performance only a block field; V3 writes `.v3db` as well, which holds the stages.

## 4. Baseline: V3 with the defaults

All numbers below are produced with `measure.py --set algo=v3` (1080p, `flowScale` 2, `v3Blend` 0 = the rule; build of this commit's parent, `meson --buildtype=release`).

### Synthetic pairs, per case and strength (mean over the source frames; field = `FF`)

EPE in full-resolution px over valid pixels (mean of per-pair mean | mean of per-pair 95th percentile | mean over the textured pixels only (5x5-dilated top decile of the source's Sobel magnitude, where motion is observable) | share of pixels above 4 px). `blk` = blockiness (0 smooth .. 1 per-block constant) at 16 px blocks, `ctrl` = the same at a period that is no harmonic of the block (period 11 flow px instead of 8: no harmonic, should be near 0). `truth` / `truth-bc` = the true field and its per-block-constant version (the two reference values). PSNR of the generated t = 0.5 frame against the true one: whole / edge pixels, pooled.

| case | str | EPE mean | EPE p95 | EPE tex | bad>4px | blk | ctrl | truth blk | truth-bc blk | PSNR | PSNR edge |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pan | s | 2.92 | 8.13 | 1.78 | 0.302 | 0.133 | 0.006 | 0.000 | 0.000 | 40.23 | 35.46 |
| pan | m | 5.52 | 22.16 | 2.98 | 0.282 | 0.110 | 0.015 | 0.000 | 0.000 | 40.80 | 35.96 |
| pan | l | 5.33 | 26.21 | 3.21 | 0.301 | 0.165 | 0.017 | 0.000 | 0.000 | 37.23 | 32.49 |
| roll_c | s | 2.37 | 10.81 | 1.48 | 0.165 | 0.123 | 0.005 | 0.000 | 0.367 | 40.55 | 35.74 |
| roll_c | m | 4.01 | 17.08 | 1.72 | 0.277 | 0.149 | 0.041 | 0.000 | 0.635 | 39.60 | 34.78 |
| roll_c | l | 5.80 | 22.46 | 2.70 | 0.293 | 0.169 | 0.020 | 0.000 | 0.823 | 36.49 | 31.67 |
| roll_o | s | 1.79 | 8.78 | 1.23 | 0.107 | 0.118 | 0.005 | 0.000 | 0.297 | 40.89 | 36.06 |
| roll_o | m | 2.96 | 13.25 | 1.34 | 0.226 | 0.137 | 0.009 | 0.000 | 0.559 | 40.04 | 35.20 |
| roll_o | l | 4.39 | 19.33 | 1.71 | 0.256 | 0.112 | 0.013 | 0.000 | 0.772 | 37.04 | 32.20 |
| zoom_c | s | 1.79 | 5.29 | 0.95 | 0.164 | 0.152 | 0.007 | 0.000 | 0.368 | 41.59 | 36.79 |
| zoom_c | m | 4.08 | 13.46 | 1.75 | 0.291 | 0.100 | 0.011 | 0.000 | 0.638 | 39.48 | 34.68 |
| zoom_c | l | 8.06 | 30.15 | 4.16 | 0.340 | 0.109 | 0.013 | 0.000 | 0.827 | 36.56 | 31.80 |
| zoom_o | s | 1.47 | 5.26 | 0.89 | 0.117 | 0.155 | 0.001 | 0.000 | 0.314 | 40.73 | 35.82 |
| zoom_o | m | 3.82 | 14.77 | 2.08 | 0.298 | 0.110 | 0.008 | 0.000 | 0.580 | 39.24 | 34.41 |
| zoom_o | l | 7.53 | 28.22 | 4.04 | 0.330 | 0.104 | 0.027 | 0.000 | 0.789 | 36.68 | 31.96 |
| rollzoom | s | 1.72 | 6.52 | 1.23 | 0.132 | 0.143 | 0.006 | 0.000 | 0.331 | 41.42 | 36.58 |
| rollzoom | m | 3.58 | 14.64 | 1.93 | 0.250 | 0.096 | 0.011 | 0.000 | 0.599 | 40.20 | 35.35 |
| rollzoom | l | 4.84 | 17.48 | 3.04 | 0.285 | 0.116 | 0.018 | 0.000 | 0.802 | 36.87 | 31.93 |
| chase | s | 1.60 | 5.52 | 0.98 | 0.170 | 0.142 | 0.005 | 0.000 | 0.331 | 42.48 | 37.71 |
| chase | m | 4.54 | 14.26 | 2.54 | 0.315 | 0.107 | 0.014 | 0.000 | 0.598 | 40.36 | 35.69 |
| chase | l | 10.15 | 35.44 | 5.83 | 0.340 | 0.100 | 0.016 | 0.000 | 0.802 | 36.57 | 31.92 |
| orbit | s | 0.92 | 3.86 | 0.38 | 0.052 | 0.100 | 0.009 | 0.000 | 0.183 | 43.27 | 38.20 |
| orbit | m | 1.58 | 5.94 | 0.54 | 0.140 | 0.120 | 0.011 | 0.000 | 0.403 | 39.15 | 34.09 |
| orbit | l | 1.98 | 9.80 | 1.07 | 0.097 | 0.114 | 0.012 | 0.000 | 0.642 | 35.89 | 31.30 |
| **all** | | 3.86 | 14.95 | 2.06 | 0.230 | 0.124 | 0.012 | 0.000 | 0.486 | 38.81 | 34.01 |

Blockiness at other block sizes (all synthetic pairs; px = full-resolution): blk8 0.193 (ctrl 0.013), blk16 0.124 (ctrl 0.012), blk32 0.065 (ctrl 0.012)

### Stages (EPE mean / p95 / textured / blk16 / step16 px, mean over all strengths and sources, per case)

| case | BLK | FD | FF |
|---|---|---|---|
| pan | 5.33 / 24.09 / 3.37 / 0.883 / 1.923 | 5.02 / 22.18 / 3.02 / 0.287 / 0.169 | 4.59 / 18.83 / 2.66 / 0.136 / 0.162 |
| roll_c | 6.60 / 24.57 / 3.50 / 0.906 / 2.531 | 6.15 / 23.15 / 3.02 / 0.286 / 0.222 | 4.06 / 16.78 / 1.96 / 0.147 / 0.229 |
| roll_o | 4.73 / 17.40 / 2.45 / 0.883 / 1.931 | 4.40 / 16.23 / 2.10 / 0.267 / 0.167 | 3.04 / 13.79 / 1.43 / 0.122 / 0.114 |
| zoom_c | 5.07 / 17.70 / 2.68 / 0.883 / 1.873 | 4.78 / 16.83 / 2.36 / 0.245 / 0.138 | 4.64 / 16.30 / 2.28 / 0.120 / 0.094 |
| zoom_o | 4.92 / 17.61 / 3.12 / 0.879 / 1.856 | 4.62 / 16.74 / 2.76 / 0.237 / 0.139 | 4.27 / 16.09 / 2.34 / 0.123 / 0.093 |
| rollzoom | 3.99 / 15.20 / 2.67 / 0.886 / 1.840 | 3.67 / 14.00 / 2.32 / 0.255 / 0.145 | 3.38 / 12.88 / 2.06 / 0.118 / 0.093 |
| chase | 5.99 / 19.77 / 3.98 / 0.873 / 1.711 | 5.72 / 19.00 / 3.64 / 0.220 / 0.119 | 5.43 / 18.41 / 3.12 / 0.116 / 0.079 |
| orbit | 1.72 / 7.44 / 0.83 / 0.745 / 0.747 | 1.57 / 6.95 / 0.69 / 0.174 / 0.059 | 1.50 / 6.53 / 0.66 / 0.111 / 0.048 |
| all | 4.79 / 17.97 / 2.82 / 0.867 / 1.801 | 4.49 / 16.88 / 2.49 / 0.246 / 0.145 | 3.86 / 14.95 / 2.06 / 0.124 / 0.114 |

### Levels: EPE mean / p95 of each pyramid level's final block field (nearest-filled to the flow grid; level l: blocks of 4 x 2^l flow px), then BLK = level 0

| case | L5 | L4 | L3 | L2 | L1 | BLK |
|---|---|---|---|---|---|---|
| pan | 21.31 / 25.52 | 14.31 / 28.93 | 12.98 / 28.65 | 7.10 / 23.62 | 6.38 / 23.86 | 5.33 / 24.09 |
| roll_c | 16.60 / 29.37 | 14.19 / 32.19 | 10.79 / 29.69 | 8.62 / 25.55 | 7.55 / 24.84 | 6.60 / 24.57 |
| roll_o | 13.73 / 24.67 | 11.59 / 20.37 | 8.79 / 19.92 | 6.74 / 18.53 | 5.73 / 17.88 | 4.73 / 17.40 |
| zoom_c | 17.06 / 25.64 | 12.78 / 22.85 | 9.84 / 21.35 | 7.05 / 18.78 | 6.03 / 18.24 | 5.07 / 17.70 |
| zoom_o | 14.83 / 24.08 | 12.13 / 22.17 | 9.48 / 20.80 | 6.78 / 18.46 | 5.91 / 18.03 | 4.92 / 17.61 |
| rollzoom | 14.14 / 24.13 | 10.23 / 18.85 | 7.65 / 17.06 | 5.95 / 16.26 | 4.95 / 15.58 | 3.99 / 15.20 |
| chase | 13.71 / 25.68 | 11.18 / 24.41 | 8.85 / 21.68 | 7.64 / 20.72 | 6.82 / 20.12 | 5.99 / 19.77 |
| orbit | 11.56 / 23.42 | 8.48 / 18.35 | 5.33 / 13.00 | 3.13 / 9.04 | 2.40 / 8.24 | 1.72 / 7.44 |
| all | 15.37 / 25.31 | 11.86 / 23.51 | 9.21 / 21.52 | 6.63 / 18.87 | 5.72 / 18.35 | 4.79 / 17.97 |

### Real pairs (the 23 held-out pairs)

PSNR whole / edge pixels pooled (mse over pixels and RGB), blk16 / ctrl16 = blockiness of the estimate (no truth: nothing excluded), mean over the pairs.

| group | pairs | PSNR | PSNR edge | blk16 | ctrl16 |
|---|---|---|---|---|---|
| battlefield | 3 | 32.26 | 25.79 | 0.121 | 0.011 |
| cs2 | 4 | 29.18 | 25.61 | 0.182 | 0.008 |
| forza | 12 | 30.71 | 25.97 | 0.171 | 0.018 |
| minecraft120 | 4 | 40.41 | 35.69 | 0.172 | 0.008 |
| all | 23 | 31.27 | 26.70 | 0.167 | 0.013 |

Stages on the real pairs (blk16 / step16 px, mean over the pairs): BLK 0.937 / 3.766; FD 0.233 / 0.280; FF 0.167 / 0.617

| pair | PSNR | PSNR edge | blk16 | ctrl16 |
|---|---|---|---|---|
| racing_forza6_c_000020_30to60_0 | 32.42 | 28.22 | 0.158 | 0.011 |
| racing_forza6_c_000160_30to60_0 | 27.89 | 23.35 | 0.304 | 0.006 |
| racing_forza6_c_000280_30to60_0 | 27.97 | 23.62 | 0.137 | 0.019 |
| racing_forza6_c_000420_30to60_0 | 31.67 | 25.41 | 0.047 | 0.007 |
| racing_forza6_c_000560_30to60_0 | 36.67 | 33.18 | 0.199 | 0.033 |
| racing_forza6_c_000680_30to60_0 | 34.07 | 29.41 | 0.136 | 0.029 |
| racing_forza6_c_000820_30to60_0 | 27.46 | 21.89 | 0.312 | 0.051 |
| racing_forza6_c_000940_30to60_0 | 30.35 | 25.57 | 0.225 | 0.013 |
| racing_forza6_c_001080_30to60_0 | 31.74 | 27.92 | 0.095 | 0.004 |
| racing_forza6_c_001220_30to60_0 | 31.75 | 26.17 | 0.189 | 0.015 |
| racing_forza6_c_001340_30to60_0 | 31.19 | 25.55 | 0.138 | 0.006 |
| racing_forza6_c_001480_30to60_0 | 35.22 | 30.87 | 0.117 | 0.019 |
| fps_cs2_a_001380_30to60_0 | 30.18 | 26.72 | 0.152 | 0.006 |
| fps_cs2_a_001440_30to60_0 | 30.32 | 26.94 | 0.073 | 0.008 |
| fps_cs2_b_001380_30to60_0 | 29.42 | 25.98 | 0.224 | 0.004 |
| fps_cs2_b_001440_30to60_0 | 27.46 | 23.54 | 0.279 | 0.012 |
| minecraft120_02_002760_60to120_0 | 45.46 | 42.13 | 0.155 | 0.008 |
| minecraft120_02_002880_60to120_0 | 41.18 | 36.99 | 0.186 | 0.003 |
| minecraft120_04_002760_60to120_0 | 39.23 | 34.25 | 0.176 | 0.006 |
| minecraft120_04_002880_60to120_0 | 38.53 | 33.30 | 0.169 | 0.015 |
| battlefield_003120_60to120_0 | 30.56 | 23.94 | 0.155 | 0.013 |
| battlefield_003240_60to120_0 | 38.52 | 31.72 | 0.102 | 0.013 |
| battlefield_003360_60to120_0 | 31.16 | 25.14 | 0.106 | 0.008 |

The pooled 23-pair figure is 31.27 dB whole / 26.70 dB on edge pixels; the known figure (`blend/probe`, rule blend at the weight stage) is 31.24 / 26.75. The difference (+0.03 / -0.05 dB) is that fgtest's output also has the clamp, smear and static mask applied; per group (forza 30.71 / 25.97 against 30.86 / 26.23) the same size.

### Real scenes

| pairs | PSNR | PSNR edge | blk16 | ctrl16 |
|---|---|---|---|---|---|
| scene | 12 | - | - | 0.294 | 0.027 |
| all | 12 | - | - | 0.294 | 0.027 |

Stages on the real pairs (blk16 / step16 px, mean over the pairs): BLK 0.971 / 20.583; FD 0.267 / 1.988; FF 0.294 / 10.949

| pair | PSNR | PSNR edge | blk16 | ctrl16 |
|---|---|---|---|---|
| forza_b600_60fps | - | - | 0.067 | 0.013 |
| forza_b600_30fps | 29.46 | 25.65 | 0.166 | 0.007 |
| forza_a1060_60fps | - | - | 0.186 | 0.025 |
| forza_a1060_30fps | 29.07 | 26.51 | 0.257 | 0.012 |
| tp_7753_60fps | 15.28 | 17.66 | 0.431 | 0.017 |
| tp_7753_30fps | 12.05 | 15.13 | 0.453 | 0.020 |
| tp_7278_60fps | 34.20 | 30.87 | 0.167 | 0.065 |
| tp_7278_30fps | 30.99 | 27.73 | 0.225 | 0.061 |
| faroe_8s_15fps | 30.27 | 25.17 | 0.476 | 0.032 |
| faroe_8s_30fps | - | - | 0.448 | 0.044 |
| fps_cs2c_960 | 23.16 | 22.28 | 0.480 | 0.023 |
| mc120_10_1080 | 23.48 | 20.53 | 0.169 | 0.009 |

(`forza_*`, `faroe_*`: pairs without `-` have a true in-between frame; the `60fps` forza pairs and the faroe 30 fps pair have none; see `scenes.json`.)

### Other estimators and settings (common subset, strength m + 6 held-out pairs)

| row | EPE mean | EPE p95 | EPE tex | blk16 | ctrl16 | PSNR syn | PSNR syn edge | real PSNR | real edge | real blk16 |
|---|---|---|---|---|---|---|---|---|---|---|
| V3 default (full set) | 3.76 | 14.45 | 1.86 | 0.116 | 0.015 | 39.82 | 34.98 | 31.30 | 26.50 | 0.189 |
| V3 default | 3.76 | 14.45 | 1.86 | 0.116 | 0.015 | 39.82 | 34.98 | 31.30 | 26.50 | 0.189 |
| V3 flowScale=1 | 3.94 | 15.04 | 1.86 | 0.062 | 0.007 | 39.88 | 35.10 | 31.03 | 26.28 | 0.123 |
| V3 flowScale=4 | 3.46 | 13.91 | 1.86 | 0.188 | 0.018 | 39.31 | 34.44 | 31.48 | 26.67 | 0.183 |
| V3 v3ColourWindow=1 | 3.86 | 14.71 | 1.94 | 0.108 | 0.013 | 38.82 | 33.88 | 31.23 | 26.41 | 0.197 |
| V3 v3Guard=1 | 3.60 | 13.96 | 1.69 | 0.104 | 0.010 | 39.98 | 35.12 | 31.04 | 26.24 | 0.126 |
| V1 | 3.78 | 14.01 | 1.85 | 0.056 | 0.009 | 36.57 | 31.63 | 28.49 | 23.32 | 0.072 |
| V2 Quality | 1.91 | 8.24 | 1.13 | 0.220 | 0.016 | 39.06 | 34.15 | 29.60 | 24.59 | 0.300 |
| Learned | 3.78 | 14.83 | 1.77 | 0.006 | 0.005 | 36.56 | 31.65 | 29.32 | 24.21 | 0.005 |

EPE mean / blk16 per case, strength m (3 source frames):

| case | V3 default | V3 flowScale=1 | V3 flowScale=4 | V3 v3ColourWindow=1 | V3 v3Guard=1 | V1 | V2 Quality | Learned |
|---|---|---|---|---|---|---|---|---|
| pan | 5.52 / 0.110 | 5.66 / 0.082 | 5.20 / 0.198 | 6.07 / 0.112 | 5.18 / 0.095 | 2.99 / 0.033 | 1.31 / 0.191 | 5.11 / 0.006 |
| roll_c | 4.01 / 0.149 | 4.24 / 0.100 | 3.66 / 0.199 | 4.13 / 0.124 | 3.66 / 0.114 | 4.25 / 0.070 | 2.20 / 0.279 | 3.63 / 0.004 |
| zoom_o | 3.82 / 0.110 | 4.02 / 0.038 | 3.47 / 0.194 | 3.81 / 0.100 | 3.70 / 0.104 | 3.94 / 0.060 | 2.01 / 0.210 | 3.59 / 0.006 |
| rollzoom | 3.58 / 0.096 | 3.79 / 0.058 | 3.26 / 0.180 | 3.61 / 0.091 | 3.51 / 0.096 | 4.67 / 0.070 | 2.43 / 0.289 | 3.68 / 0.006 |
| chase | 4.54 / 0.107 | 4.70 / 0.040 | 4.33 / 0.175 | 4.51 / 0.106 | 4.44 / 0.101 | 4.42 / 0.068 | 2.59 / 0.228 | 3.68 / 0.007 |
| orbit | 1.58 / 0.120 | 1.68 / 0.063 | 1.40 / 0.146 | 1.57 / 0.104 | 1.51 / 0.104 | 1.95 / 0.022 | 0.70 / 0.092 | 3.21 / 0.006 |

### Reading the numbers (V3 defaults, `algo=v3`)

**Where the block steps come from.** The level-0 block field `BLK` is constant per 8x8 flow px by construction (blk 0.87 on the synthetic pairs, 0.94 on the real ones; steps of 1.8 px / 3.8 px on the grid lines). `v3_dense` (the soft per-pixel field `FD`) brings blk down to 0.25 (0.23 real) but not to 0: its kinks sit at the block *centres* (phase 3 / 4 of 8; it blends the 4 surrounding block vectors bilinearly, and the soft pick makes the blend steeper where the block vectors disagree), so the field it outputs is still piecewise-bilinear on the block grid. `v3_occl` + `v3_fill` (`FF`, the field the synth projects) leave 0.12 (0.17 real), now with the lines at the block boundaries (phase 0) where the fill replaced pixels. The error comes in earlier than the steps: the mean EPE of each level's block field is 15.4 px (level 5), 11.9, 9.2, 6.6, 5.7 (level 1), 4.8 (level 0 `BLK`), 4.5 (`FD`), 3.9 px (`FF`), but the 95th percentile hardly moves after level 3 (25.3, 23.5, 21.5, 18.9, 18.4, 18.0, 16.9, 15.0 px): a block that locked to the wrong vector in the coarse levels (or flat, repetitive texture) is inherited by every finer level (the finer levels start from the parent's vector and only ring-search +-1 level px around it), and the dense pass only mixes the four neighbouring block vectors, so it cannot recover a vector none of them has. The fill removes the largest part of the remaining error (`FD` 4.5 -> `FF` 3.9 px mean, p95 16.9 -> 15.0) but only where the forward-backward test flags the pixel. So: the blocks are made by the per-block constant search of levels 1 / 0 (its wrong vectors are what one sees as square patches), the dense pass spreads them into bilinear kinks on the same grid instead of removing them, and nothing after it regularises the field.

**1. "Blockiness, connected motion".** The sheets show square patches of wrong or different motion with edges on the 16 px block grid and its multiples (`forza_pan_l`, `forza_roll_c_l`, `faroe_8s_15fps`): on a pure 64 px pan the true field is one colour and V3's has 30 % of the pixels more than 4 px off (EPE mean 5.3, p95 26). The blockiness figure is a share, not a size, so it reads modest (0.12 against 0 for the truth and 0.49 for the per-block-constant truth): the grid structure is real (the control at a period no harmonic of the block reads 0.012) and `step16` is 0.11 px of extra curvature on the grid lines on average, 0.6 px in real pairs. For comparison on the same pairs: V2 Quality is more blocky (0.22) with half V3's error; V1 0.056 and Learned 0.006 are smooth but have the same error as V3 (so smooth is possible without a higher error); V3 with flowScale 1 halves the blockiness (0.062) at a slightly higher error; flowScale 4 raises it (0.19).

**2. Turning objects (roll, orbit, third person).** Pure rotation is as hard as a pan: `roll_c` / `roll_o` EPE 2.4 / 1.8 px at 8 px, 5.8 / 4.4 px at 64 px with blk 0.11 .. 0.17. The `orbit` case (parallax, still figure) has the lowest error (0.9 .. 2.0 px) but the largest errors sit on the figure's outline and the large flat planes next to it (`forza_orbit_l`: blocky patches of 16 .. 48 px beside the figure); the figure itself is held still correctly. The only third-person clip found (`2026-10-26 10-55-24.mkv`, a 240 fps capture of an avatar game, the camera swings about the avatar) gives PSNR 34.2 dB (`tp_7278_60fps`, a calm swing; the estimate is a smooth pan field with a blocky seam on the left wall) and 15.3 dB (`tp_7753_60fps`, a fast whip: the field is chaotic patches, blk 0.43, the generated frame is a smear); there are no true vectors for them.

**3. Driving, things coming closer (zoom, chase).** The error grows with the strength much faster than for a pan: `zoom_c` EPE mean 1.8 / 4.1 / 8.1 px at 8 / 24 / 64 px, `chase` 1.6 / 4.5 / 10.2 px, p95 35 px, a third of the pixels more than 4 px off at 64 px; on the textured pixels only it is 5.8 px for `chase` l. PSNR falls from 42.5 to 36.6 dB over the same strengths. Large flat areas (road, sky) near the focus of expansion and at the frame edge, where the motion is largest, get patches of the wrong vector or none; the real Forza 30 fps pairs (`forza_b600_30fps`, `forza_a1060_30fps`) show the same patches (blk 0.17 / 0.26) and PSNR 29.5 / 29.1 dB; the drone flight (`faroe_8s_15fps`) blk 0.48, 30.3 dB.

**Reference rows** (30 pairs: the 24 synthetic strength-`m` pairs and 6 held-out pairs; `measure_*.md` / `.json` in `/home/mo/tmp/fg-v35/baseline/`): V2 Quality has the best motion error here (EPE 1.9 against V3's 3.8, p95 8.2 against 14.5) but a worse real picture (29.6 against 31.3 dB; blocky and, here, more kinked); V3's `v3Guard=1` (V2's motion where it explains the frames better) lowers V3's error to 3.6 / 14.0 with no loss in the synthetic picture, but loses 0.3 dB on the real pairs; `v3ColourWindow` changes little.


Notes on the tools: at `flowScale` 4 the blockiness is measured at B = 4 flow px (16 px) while V3's blocks are still 8 flow px (32 px): read `blk32` in the JSON for that row. `v3Guard` needs V2's subgroup requirement. The synthetic sources are single frames, so the textures are those of the games (repeated blocks in Minecraft, flat road in Forza) -- the pan error of Forza and Minecraft pairs is dominated by flat or repetitive areas. The third-person clip is not a clean "camera circles a still character" (the avatar walks while the camera swings), so for that complaint the `orbit` synthetic case is the only exactly known ground truth.
