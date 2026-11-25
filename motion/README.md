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


## V3.5 (round 1)

`Settings::v3Motion = 1` (inside `MotionVersion::V3`; 0 = V3 exactly as before, byte-identical) replaces V3's block field with a regularised one. Passes, per pyramid level 5..0, after V3's own search of the level (the search shaders are the unchanged V3 ones):

1. **Camera refit** (`v35_stats`, levels LH..1): V3's dominant-motion / 6-parameter camera fit, with every sample's weight multiplied by its block's texture (`smoothstep(1, 3.5, mean detail)`), redone on this level's blocks. V3's own fit weighs by cost only, so flat blocks (low cost at any vector) outvote textured ones; the level-3 vectors are also integers x 8 flow px, so its model is crude. At level 1 only every 8th sample (jittered) is used so that all stay in registers (cost: 0.86 -> 0.07 ms, EPE +0.03). `v35_camq` writes the share of matching blocks that agree with the model (a quality, unused in the end: no measurable effect).
2. **Regularisation** (`v35_reg`, two Jacobi steps per level, block resolution): each block gets a confidence-weighted vote of its 5x5 neighbours (weight: confidence from cost and ambiguity, x colour similarity `exp(-|dc|/30)`, so a still foreground does not pour its zero into the road behind), plus the camera model's vote where the block is flat. The result is a weighted medoid plus two mean-shift steps. Aperture-aware: a structure tensor per block splits the change into the component across the line (kept by the block's own evidence) and along it (taken from the neighbours). A change is rejected if the new vector explains the block's 8x8 window clearly worse than the old one (`winCost`), which protects thin objects, parallax and textured blocks.
3. **Dense pass** (`v3_dense.comp` compiled with `V35`): where no two adjacent block vectors of the 4x4 neighbourhood differ by more than 2 flow px, a Catmull-Rom interpolation replaces the bilinear one; elsewhere the original code runs.

Pipelines: 4 added, created lazily (`v35Ensure`, like `guardEnsure`) on the first use with `v3Motion != 0`: they compile at that moment (cold cache: a stall of a few shaders); with `v3Motion = 0` nothing changes. `sizeof(Settings)` 164, `offsetof(v3Motion)` 155 (the former padding byte).

Numbers (72 synthetic + 23 held-out real pairs; V3 -> V3.5): mean EPE 3.86 -> 1.27, p95 14.95 -> 5.28, textured EPE 2.06 -> 0.97, pixels above 4 px 23.0% -> 5.1%, falsely still (estimate under 0.5 px where truth above 2 px) 5.3% -> 0.3%, blk16 0.124 -> 0.058, synthetic PSNR 38.81 -> 39.33 (edge 34.01 -> 34.42). EPE is lower in all 24 case/strength rows (chase l 10.15 -> 5.90, orbit l 1.98 -> 1.60). Real PSNR whole / edge: battlefield 32.26/25.79 -> 32.67/26.16, cs2 29.18/25.61 -> 29.19/25.61, forza 30.71/25.97 -> 30.80/26.07, minecraft120 40.41/35.69 -> 40.45/35.73. Cost on the RX 7900 XTX at 1080p, back-to-back per pair (3 runs each, same session): 2x V3 0.95 -> V3.5 1.59 ms (+0.64), 4x 1.42 -> 2.02 ms (+0.60); the budget was +1.5 ms. Not solved (round 1): chase l (5.9 px), the flat road beside a still car at large strength, tp_7753 (chaotic scene, output unchanged), forza real blk16 up (0.171 -> 0.216: larger smooth areas with a few remaining steps), scenes forza_b600_30fps (-0.49 dB) and faroe_8s_15fps (-0.28 dB).


## V3.5 (round 2)

Changes to the round-1 passes (V3's own search is still the unchanged V3 shaders):

- `v35_camq` is gone (it never changed a number); the top pyramid level is no longer regularised (a few blocks, a dispatch costs more than it gains); the window-cost check uses every second pixel of the 8 x 8 window. Together with a register cache in the level-1 camera fit (samples computed once, thinned to 12 x 256) the added cost fell from +0.64 to +0.45 ms.
- **Regional camera models** (`v35_rgn.comp`, 12 workgroups per stats level 3..1): 4 x 3 regions, each (cell plus half a cell around, so neighbours overlap by half) fits its own affine motion by robust weighted least squares from the F01 blocks. A block counts by its match, its texture and the smaller eigenvalue of its structure tensor (an edge proves the motion across it only), pass 0 already weighs by the residual to the global model (a still foreground is no inlier of the background), the parameters are pulled to the global camera (ridge = 60 sample weights), a region with too little weight takes the global model. Each region also stores a reliability (weighted rms residual, capped per sample): where one affine does not describe the region (parallax: a near cliff beside a far road) its vote and its decisions are weak. The model vector of a block is the bilinear blend of the four nearest regions' predictions (`regVote` in `v35_cam.glsl`): a piecewise-affine camera.
- **The model decides where a confident block is not supported** (`v35_reg.comp`): when fewer than 30% of the neighbour weight supports the vector a block has, and the block's window matches the model's vector nearly as well (`REJ2`), the block takes the model's vector. This removes the high-confidence outliers on repeated textures (window grids, signs) and the wrong-direction slabs that were a texture repeat. Flat blocks are *not* handled by this rule: they follow their confident same-coloured neighbours in the vote (colour scale tightened 30 -> 12, which stops a flat still foreground from taking the road's vector and the road from taking the foreground's), with the regional model as one more voter. A tried and dropped variant (flat blocks with few confident neighbours take the model, "R1") lost 0.05 dB on the real Forza pairs and changed no synthetic number.
- Window-cost rejection tightened to 0.05 absolute (was 0.15 + 5%): a changed vector must explain the window at least as well.

Why tp_7753 (and most of the new `tp_n*` pairs) give identical output: V3's last-resort rule (`v3_decide.comp`, task 112, switchable by no setting) fires: more than half of all flow pixels are unsure (uAll 0.81, rule: > 0.5) *and* the luma histograms of the two frames differ (D = 4.79, rule: > 4.5, `V3_HARD_U2/D2`), so the pair is a cut and the nearer real frame is shown. It is not `v3SceneCut` (default 0). The pair is not one global motion: phase correlation of 1/4-size luma has response 0.17 (60 fps) / 0.07 (30 fps) at best, the exhaustive shift search with at least half the frame overlapping finds no minimum (mean abs luma 17.05 at the best shift against 17.23 at zero), and the half-shift blend of the two frames is worse than the nearer frame (14.3 against 15.3 dB). DIS optical flow of prev to the real in-between frame has median 86, p90 297, p99 551 px (and at 4 frames twice that): a whip pan of a near floor and far walls, flow spread over several hundred px, above V3's search range (about 512 px at the top level), plus motion blur. No global candidate can find that, so no fallback change was made. Of the 20 `tp_*` pairs, 15 are in that regime (16.4 dB or less, V3 and V3.5 identical output except the tie flips below); tp_7278 and tp_n182 (4 pairs) are normal and V3.5 equals V3 there (+-0.06 dB); tp_n043_60fps is two identical frames. Pairs at the rule's edge flip between the two real frames (tp_n153: -0.86 / +0.89 dB, tp_n264_30fps in an earlier build -5.7 dB): that is the hard-cut / unsure-share rules choosing prev or curr, not the field.

Numbers (72 synthetic + 23 held-out real; V3 -> V3.5, round 1 in brackets): EPE 3.86 -> 0.97 (1.27), p95 14.95 -> 3.43 (5.28), textured EPE 2.06 -> 0.82 (0.97), above 4 px 23.0% -> 3.9%, falsely still 5.3% -> 0.1%, blk16 0.124 -> 0.058, PSNR 38.81 -> 39.59 (39.33), edge 34.01 -> 34.65. All 24 rows lower; chase l 10.15 -> 4.80 (5.90). Real PSNR whole / edge: battlefield 32.26/25.79 -> 32.90/26.38, cs2 29.18/25.61 -> 29.19/25.61, forza 30.71/25.97 -> 30.80/26.04, minecraft120 40.41/35.69 -> 40.49/35.74; blk16 forza 0.171 -> 0.155 (round 1: 0.216; the rise was model-vote kinks, now smoothed). Cost on the RX 7900 XTX (1080p, back-to-back per pair, 3 runs): 2x V3 0.96 -> V3.5 1.41 ms, 4x 1.43 -> 1.86 ms. First use: the 3 lazily created pipelines add 0.26 s to the first output with an empty shader cache (1.15 -> 1.41 s from process start); warm runs identical (0.048 s).

Known losses of round 2: the rim of a still foreground object (error inside a 16 px band of the cut-out outline, V3 -> V3.5, forza_chase l 0.8 -> 7, forza_orbit l 8.5 -> 13: the straddling blocks of the coarse levels 1 and 2 take the background's vector; skipping the regularisation there restores the rim but undoes most of the gain outside), thin near objects (the lamp post in forza_b600_30fps takes the background's vector: the 0.14 dB loss there), a fast first-person weapon (fps_cs2c_960) partly smoothed into the background.

## V3.5 (round 3)

Round 3 is the last round. Numbers are from the final code (the same V3 reference build and the same pairs as before; `Settings::v3Motion = 1`).

### What changed

- **Level 0 keeps V3's own vectors (slot 0).** The regularisation at level 0 is now one step that reads slot 0 (V3's search result) and writes only the vector of slot 1. It keeps slot 1's z, which holds the still-part evidence of the zero-vector bonus (round 2 overwrote it, so the bonus never saw its evidence under V3.5). The dense pass reads the regularised vectors from slot 1; where a block's regularised and own vector differ by more than 1 flow px it offers both (8 candidates, the per-pixel pick), and a tile with evidence for the zero-vector bonus takes the unchanged code, not the smooth interpolation (the HUD protection works as in V3).
- **Replacement test.** A vote that moves a block is kept only if the window cost at the new vector is at least 3% lower than at the block's own (`REJ = (0, -0.03)`; before: 0.05 higher allowed). Where the image has no evidence for the change the block keeps its own vector. -10% was better for the rims and cost the fast weapon in `fps_cs2c_960` 0.5 dB, so -3% it is.
- **Aperture strips.** At levels 2 and up, an edge-only block (strong structure along one direction only) takes the camera model's vector along the edge and keeps its measured across-edge part, if its window cost does not rise by more than 5% + 0.05. It happens before the vote, so the block does not vote its wrong along-edge part into the flat surface beside it. Cost +0.07 ms (levels 1 and 0 cost +0.04 more and were left out).
- **Beyond-reach whips.** A pair on which V3's own match weights give the camera fit fewer than 10% inliers while the fit's translation is above 90 flow px has no valid dominant motion and no camera in V3.5 (`G_VALID = 0` from `v35_stats`, as V3 would have it for such a pair). Before, V3.5 found a "camera" among the textured blocks (a stone wall's repeated wrong vector), and V3's own later passes (occlusion, fill, project fall-back) then applied it to the flat sky.

### Numbers

Synthetic (72 pairs): mean EPE 3.86 -> 0.95, p95 14.95 -> 3.62, textured 2.06 -> 0.83, above 4 px 23.0% -> 4.2%, false still 5.3% -> 0.2%, blk16 0.124 -> 0.076, PSNR 38.81 -> 39.55, edge 34.01 -> 34.62. chase l EPE 10.15 -> 4.62, zoom_o l 7.53 -> 1.1. Every one of the 24 rows has a lower EPE. Real pairs (23): all +0.11 dB (edge +0.10); forza +0.11 / +0.09, battlefield +0.38, minecraft120 +0.09 / +0.04, cs2 +0.00 (blk16 .182 -> .193). No group loses.

Cost before the cheaper schedule (task 169, see "V3.5 cheaper"), on the forza real pair, 1080p, back-to-back, 3 runs each: 2x V3 0.95 ms, V3.5 1.47 ms (+0.52); 3x 1.18 / 1.69; 4x 1.42 / 1.90.

### Rims of still foreground objects (the known loss)

Band table (16 px bands outside / inside the still cut-out's outline), flow EPE and picture error (mean absolute error of the generated t = 0.5 frame against the true middle frame, levels), V3 -> V3.5:

| pair | EPE outside | EPE inside | picture outside | picture inside |
|---|---|---|---|---|
| forza_chase_m | 10.36 -> 3.60 | 0.39 -> 1.71 | 1.91 -> 1.67 | 0.12 -> 0.21 |
| forza_chase_l | 33.58 -> 14.59 | 0.78 -> 4.78 | 5.32 -> 5.01 | 0.14 -> 0.28 |
| forza_orbit_m | 6.58 -> 1.68 | 0.98 -> 3.25 | 3.53 -> 3.35 | 0.24 -> 0.35 |
| forza_orbit_l | 10.80 -> 3.73 | 8.52 -> 10.55 | 9.36 -> 9.31 | 0.78 -> 0.82 |
| cs2_chase_m | 4.95 -> 1.52 | 0.22 -> 0.59 | 2.05 -> 1.75 | 0.15 -> 0.27 |
| cs2_chase_l | 17.07 -> 5.66 | 0.83 -> 1.93 | 5.10 -> 4.50 | 0.53 -> 0.74 |
| cs2_orbit_m | 2.43 -> 1.13 | 1.04 -> 1.37 | 4.54 -> 4.38 | 0.68 -> 0.78 |
| cs2_orbit_l | 8.61 -> 7.83 | 5.89 -> 6.01 | 8.34 -> 8.16 | 1.68 -> 1.71 |
| mc_chase_m | 10.12 -> 6.30 | 0.13 -> 0.64 | 1.52 -> 1.32 | 0.04 -> 0.10 |
| mc_chase_l | 35.79 -> 29.97 | 0.47 -> 0.63 | 2.96 -> 2.90 | 0.10 -> 0.17 |
| mc_orbit_m | 6.17 -> 0.94 | 1.30 -> 2.77 | 4.50 -> 4.21 | 0.23 -> 0.47 |
| mc_orbit_l | 15.86 -> 7.59 | 16.46 -> 16.99 | 5.75 -> 5.66 | 0.78 -> 1.17 |

The inside-band gate (V3 + 0.3 px, or not above V3 where V3 is above 5 px) is NOT met: 10 of 12 pairs fail (round 2: 9 of 12 at 2.8 to 13 px; the worst inside errors are now about half). The picture side shows why it matters little: the picture error inside the band stays within 0.1 to 0.4 levels of V3's, and the outside band's picture gets better in every pair. The rim's vector error is mostly absorbed (the rim pixels' two candidates match about equally well in the image). What was tried, and what it did (all measured with this table): a keep-own rule for blocks whose same-coloured neighbours share their vector (inside 6.6 -> 3.9 on chase l, but the outside gain falls from 62% to 37%); colour weight made stricter across vector disagreements (no change); own-evidence test against the model before any vote (no change: the rim blocks are flat, both vectors match equally); switching a level's regularisation off (restores the rim, but each level's loss adds up, and all four levels off undo the whole outside gain). The reason is in the flat blocks: a flat still object and a flat road look the same to every test that uses only the image, and the model's pull is exactly what repairs the road.

### Third-person turning (`tp_m*`, new)

Ten moments of the 240 fps recording (`2026-10-26 10-55-24.mkv`), chosen by DIS flow on 1/4-size frames 4 apart (one 60 fps pair), scanned every 0.25 s, median displacement 15 to 117 px per 60 fps pair (the scan's 150 px upper bound was not reached with sideways-dominant flow and the avatar in view), sideways flow dominant, avatar in view (checked on a contact sheet). Pairs `tp_mNN_60fps` (frames +0 / +2 / +4) and `tp_mNN_30fps` (+0 / +4 / +8); cut by `/home/mo/tmp/fg-v35/round3/cut_m.py`. PSNR whole V3 -> V3.5 (a "fall-back" is failMix above 0.5 on either side):

| pair | median px | 60 fps | 30 fps |
|---|---|---|---|
| m01 | 15 | 24.08 -> 24.05 | 23.63 -> 23.60 |
| m02 | 20 | 34.58 -> 34.65 | 36.39 -> 36.38 |
| m03 | 25 | 25.55 -> 25.70 (fall-back) | 23.99 -> 23.87 (fall-back) |
| m04 | 33 | 30.15 -> 31.19 | 22.39 -> 22.36 (fall-back) |
| m05 | 42 | 26.84 -> 26.84 | 25.76 -> 25.77 |
| m06 | 54 | 19.89 -> 19.81 | 21.77 -> 21.79 (fall-back) |
| m07 | 69 | 21.42 -> 21.27 | 21.25 -> 20.35 (fall-back) |
| m08 | 90 | 21.41 -> 21.29 | 21.91 -> 21.86 (fall-back) |
| m09 | 107 | 22.65 -> 22.59 (fall-back) | 22.16 -> 21.94 (fall-back) |
| m10 | 117 | 19.87 -> 20.04 (fall-back) | 20.65 -> 20.45 (fall-back) |

Group means over the pairs where neither version fell back: 60 fps (7 pairs) 25.48 -> 25.59, edge 21.54 -> 21.65, blk16 .194 -> .199; 30 fps (3 pairs) 28.59 -> 28.58, edge 23.02 -> 23.03, blk16 .314 -> .305. The 60 fps gain is m04 (+1.04); the other six are within 0.15 dB, and from 54 px up V3.5 is a little behind (-0.06 to -0.15 dB, blk16 up): both fields are about equally wrong in the high-frequency grass and wall texture there and V3.5's has a few more block steps (m07 sheet). This is information, not a fix. The hard-cut rule `(uAll > 0.5 && D > 4.5)` never tripped on any tp_m pair, for either version: D stays below 0.75 (the unsure share reaches 0.91 on m10 at 117 px, D is 0.34). It trips on the real whips (D 2.5 to 4.8; tp_7753 at 4.79). The whip group (tp_7753, tp_7278, tp_n043, 082, 127, 139, 153, 182, 206, 264): no pair loses more than 0.08 dB (tp_n153_60 -0.08, tp_n182_30 -0.05), tp_n153_30 +1.51, the rest within 0.03.

### HUD and thin objects

`forza_a1060_30fps`: V3.5 +0.13 dB over V3 (29.07 -> 29.20). The speedometer's digits are unchanged from V3's (crop of the tile at x 1600 to 1920, y 768 to 1008); the tile that loses most against V3 (bottom right, MAE 18 -> 23) is the fast road surface behind it, which both versions smear and V3.5 a little more. `forza_b600_30fps` (lamp post) -0.16 dB, `fps_cs2c_960` (weapon) -0.06, `mc120_10_1080` -0.04, `faroe_8s_15fps` +0.08.

## V3.5 as it stands (final summary, pass by pass)

`Settings::v3Motion = 1`, valid with `MotionVersion::V3` only; four lazily created pipelines on top of V3's own (`v35_stats`, `v35_rgn`, `v35_reg`, `v3_dense_35`).

1. **Search** (unchanged V3 passes, per level n-1 down to 0). Level n-2 down to 1 then run the V3.5 passes below after the wide step.
2. **`v35_stats`** (levels LH = max(1, n-3) down to 1): the dominant motions (V3's weights) and a 6-parameter affine camera from the blocks' vectors, weighted by match and by texture (flat blocks do not count). A pair with fewer than 10% V3-weight inliers and a translation above 90 flow px has no camera.
3. **`v35_rgn`** (same levels): 4 x 3 regional affine models, ridge-regularised toward the camera, with a reliability from their residual.
4. **`v35_reg`** (levels n-2 down to 0; two Jacobi steps per level, one at level 0; the top level is skipped): per block, the aperture step (levels 2 and up), then the vote of the neighbours (colour weighted, confidence and aperture aware) together with the regional (else global) model, weighted medoid plus two mean-shift steps; replacement kept only where the window cost is at least 3% lower; the model decides for confident blocks no neighbour supports (R2). Only the vector changes.
5. **`v3_dense_35`**: Catmull-Rom interpolation of the regularised level-0 field where it is smooth; otherwise V3's per-pixel pick, with V3's own level-0 vector as a second candidate where it differs by more than 1 flow px; the zero-vector bonus as in V3.
6. V3's occlusion, fill, decide, project, synth, and the hard-cut / unsure rules are unchanged.

## After I11 (task 160): V3.5 in every format, with the learned blend, and the HUD hold

V3.5 (`v3Motion = 1`) is complete in the library: the four test programs take `--v3-motion N`, every format at 2x / 3x / 4x runs it, the validation layer (core, GPU-assisted, synchronisation; Rgba8, Rgb10, Nv12 at 2x / 4x, `v3Blend` 0 / 1, `v3Guard` on / off) reports nothing, and switching `v3Motion` 0 -> 1 -> 0 at run time reproduces the output of an interpolator initialised with each value.

**One fix.** `v3test --scene hud` (a still overlay over a panning background) failed at `v3Motion` 1 only: overlay error 9.2 / 9.4 levels against V3's 6.1 / 6.4, limit 7. The background's field was better (dominant error 0.0 against 2.8) but `v35_reg` replaced the zero vector of the still blocks with the neighbours'. Now a block with its own vector below 0.25 flow px and a mean detail of at least 2 keeps it while `v3HudBonus > 0` (`HUD_EN` in `v35_reg.comp`). Overlay error after: 6.2 / 6.4. The 72 + 23 gate was run again on the final code: synthetic mean EPE 3.86 -> 0.96, p95 14.95 -> 3.69, false still 0.2%; held-out 23 pairs 31.27 -> 31.38 dB whole, 26.70 -> 26.81 edge, blk16 0.167 -> 0.148; no row with a higher EPE, no group losing more than 0.05 dB (battlefield rose from +0.38 to +0.50 dB). Frames at `v3Motion` 0 are byte-identical to the build before I11 (84 of 84).

### Hybrid table

PSNR whole / edge in dB, pooled per group, V3 (`v3Motion` 0) -> V3.5 (1), for the rule blend, Fluidity (`v3Blend` 1) and Consistency (`v3Blend` 2). "scenes" are the 45 real-scene pairs of `scenes_all.json` and the `tp_m` pairs that have a true in-between frame.

| group (pairs) | rule | Fluidity | Consistency |
|---|---|---|---|
| battlefield (3) | 32.26 / 25.79 -> 32.75 / 26.26 | 32.44 / 25.85 -> 32.89 / 26.28 | 32.42 / 25.82 -> 32.85 / 26.25 |
| cs2 (4) | 29.18 / 25.61 -> 29.18 / 25.61 | 29.40 / 25.68 -> 29.39 / 25.68 | 29.35 / 25.70 -> 29.34 / 25.70 |
| forza (12) | 30.71 / 25.97 -> 30.81 / 26.05 | 30.45 / 25.62 -> 30.53 / 25.69 | 30.29 / 25.45 -> 30.37 / 25.52 |
| minecraft (4) | 40.41 / 35.69 -> 40.50 / 35.73 | 40.45 / 35.69 -> 40.52 / 35.70 | 40.60 / 35.88 -> 40.70 / 35.91 |
| **held-out (23)** | 31.27 / 26.70 -> 31.38 / 26.81 | 31.19 / 26.52 -> 31.28 / 26.61 | 31.08 / 26.42 -> 31.17 / 26.51 |
| scenes (45) | 16.12 / 16.40 -> 16.17 / 16.42 | 16.15 / 16.46 -> 16.19 / 16.50 | 16.19 / 16.52 -> 16.24 / 16.55 |

Verdict: V3.5 gains about +0.09 dB over V3 under either learned blend on the held-out pairs, the same as under the rule blend, so the two improvements add up and do not interfere; the gain sits in battlefield (+0.45 dB) and forza (+0.08 dB), cs2 is unchanged, and V3.5 Hybrid is not worse than V3 Hybrid on any group. The learned blends themselves stay below the rule blend on forza and the pooled 23 pairs in this metric, as before (they are chosen for the look of the motion, not for PSNR). Worst pairs for V3.5 Hybrid (Fluidity): `tp_m07_30fps` -0.52 dB, `tp_n153_60fps` -0.27 dB; best: `tp_n153_30fps` +1.65 dB, `tp_m04_60fps` +0.80 dB.

### Before / after sheets (V3 Hybrid Fluidity against V3.5 Hybrid Fluidity)

`forza_b600_30fps` -0.06 dB (a lamp post loses: a thin near object), `forza_a1060_30fps` +0.05 dB (the kerb strip loses a little, the wall and the road gain), `fps_cs2c_960` -0.06 dB (blk16 0.480 -> 0.516, lost and gained patches along the crate edges cancel), `mc120_10_1080` -0.03 dB (no visible difference), `tp_m04_60fps` +0.80 dB (the grass plane under the camera turn becomes smooth: the block-patchy orange region of V3's field is gone). On the real game scenes the two are equal within 0.06 dB; the clear gain is on the camera-turn scene.

### Formats, sizes, cost

Seven formats (Rgba8, Bgra8, Rgb10, F16, F16Lin, Nv12, P010) x 2x / 3x / 4x x six sizes (1920x1080, 2560x1440, 1440x1080, 1280x720, 1366x768, 1367x769): 252 runs, none failed; Bgra8 equals Rgba8 byte for byte at both values; the float formats produce no non-finite, negative or over-range value. The PSNR of a format against Rgba8 moves by more than 0.3 dB in 113 of 324 comparisons (87 up, 26 down by 0.3 to 0.67 dB, mostly Nv12), because the estimate changed where the formats' own tone and chroma paths disagree most (on `tp_m04` Rgb10 / F16 against Rgba8 rise from 33.6 to 37.3 dB: the formats agree better with V3.5). This is a shift of what the different formats do with a different field, not a defect of one format.

Cost before task 169 (`fgtest --bench 200`, RX 7900 XTX, median of 3; now +0.30 to +0.33 ms at Quality, see "V3.5 cheaper"): V3.5 adds 0.48 to 0.52 ms per pair at 1080p and 0.62 to 0.65 ms at 1440p. 1080p 2x rule blend: Rgba8 1.470 ms, Rgb10 1.496, Nv12 1.502 (a hair above the 1.5 ms limit). First use with an empty shader cache: 1.16 s (V3) -> 1.43 s; the first `setSettings` 0 -> 1 stalls 275 ms with an empty cache and 0.33 ms with a warm one.

### Weak points (stated, not fixed)

Just inside the outline of a still foreground object, strips along lane lines and kerbs, thin near objects, whip pans beyond the search range (the dominant motions are invalid there; at `flowScale` 1 three `v3test` stats checks fail only at V3.5), and on real third-person turning pairs V3.5 is equal to V3.

## The real road (task 161)

The finding: on real driving pairs the plain road gets almost no motion in V3 and in V3.5 (V3.5's field is smoother, but the asphalt in front of, beside and behind the car, and the centre line, stay near white in `debugView = Motion`), although the known-motion pairs (one sharp frame warped two ways) are right. This section measures it, names what leaves a block at zero, and records an attempt at the fix that did not hold.

### The `road` set and the measures

`road` = 30 pairs: the 12 Forza held-out pairs, `forza_b600_*`, `forza_a1060_*`, `faroe_8s_*` (6 scene pairs) and 12 more pairs cut from the three Forza clips at spread positions with plain road in view (frames N, N+1, N+2 of the 60 fps clip: 30 fps spacing with the true in-between frame; `road_a_000380` ... `road_c_001300`). Tools (all GPU runs through the lock; nothing here needs a window):

- `fgtest --dump` now also carries G words 3896..4095 at the end of the `.v3db` (`gpu/v3flow.py` returns them as `g35`): V3.5's 12 regional camera models of the last level that fitted them (words 3900 + 8 r: the six affine parameters, the reliability, the valid flag; all zero at `v3Motion = 0`). `common.model_field` evaluates them at every flow pixel exactly as `v35_cam.glsl` `regPredict` does (the blend of the regions, level-0 flow px). The output frames and every other word of the dump are unchanged.
- **"still against the model"** (`common.still_vs_model`, `measure.py --road`): the share of the pixels with |final field| < 0.5 px among the pixels where the model predicts more than 2 px with reliability above 0.5. V3 is judged against V3.5's model of the same pair.
- **sparse truth of the ground** (`common.road_truth`): the real in-between frame gives displacements that no model is needed for. On a grid of 64 x 64 px patches over the lower half of the frame, the best displacement prev -> mid and mid -> curr (within +-96 px, L2 on luma); a patch counts when the two agree within 3 px (constant velocity), the match is distinct (error at the best below 0.6 of the search's median and below 0.8 of the zero vector's) and the patch has texture. 913 patches in the 24 pairs that have an in-between frame. The field and the model are read at the patch centres. This is the check that does not depend on the model.
- `maps.py --road DIR`: a sheet per pair, 1800 px wide, 3 x 3 panels: prev | V3 | V3.5 today; V3.5 new | model of today | model of new; the still-against-the-model mask of today | of new | the ground-truth patches as arrows (green: V3.5 today is within 4 px). `measure.py --road` writes the per-pair table, `road.json` and the fields the sheets are drawn from.

### What leaves the blocks at zero (diagnosis, V3.5 as in `1ccc602`, the 30 pairs summed)

`v35_reg.comp` has an optional diagnosis (`#define V35_DIAG 1` at the top of the file; the production build has none of it): per level, the blocks of F01 where the model predicts more than 2 px with reliability above 0.5 and the pass leaves the vector under 0.5 px, and which rule did it (exclusive, in this order). 7.65 million pixels are in the model's area; 5.89 % of them are still (V3 12.8 %, the mean over the pairs of today's V3.5 6.6 %).

| level | blocks in the model's area | still (share) | (ii) HUD hold | (iii) vote mass too small | (iii) zero neighbours outvote the model | (iv) the model is no voter | (i) 3 % window-cost test rejected the vote | (v) R2 | (vii) other |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 120332 | 7890 (6.6 %) | 31 % | 1 % | 3 % | 8 % | 58 % | 0 | 0 |
| 1 | 120332 | 24451 (20.3 %) | 36 % | 0 | 6 % | 20 % | 38 % | 0 | 0 |
| 2 | 45654 | 14211 (31.1 %) | 51 % | 0 | 4 % | 24 % | 21 % | 0 | 0 |
| 3 | 12372 | 5681 (45.9 %) | 64 % | 0 | 1 % | 24 % | 11 % | 0 | 0 |

(iv) is counted as "the vote is zero and the model's own weight is under 0.05": a block with texture (`sr1` above 0.6) gives the model no vote (`cq`). (v) rule R2 is last in the order and never the first cause: a block that reaches it has already been kept at zero by the hold, the vote or the 3 % test, so its two counters (neighbours support the vector / the model's window cost too high) are 0 at every level. (vi) the dense pass: 97 200 of the 450 466 still pixels (21.6 %) lie in a level-0 block whose regularised vector moves (at least 0.5 px) and are still in the final field: V3's own zero vector won the per-pixel pick (the pick is a soft mean over up to 8 candidates, TAU = 1.5 levels, so the zero and the moving vector of a tie average to about half). Distribution over the counted blocks (level 0 / 1 / 2 / 3): window cost at the vector (zero) 0.84 / 1.64 / 2.00 / 2.10 levels, at the model's vector 3.61 / 3.08 / 3.30 / 2.79, |difference| / (detail + 2) 0.97 / 0.57 / 0.51 / 0.30; block detail `sen` 2.27 / 2.77 / 3.98 / 4.63; structure tensor sqrt-eigenvalue ratios r1 (along the dominant gradient) 0.41 / 0.53 / 0.67 / 0.80 and r2 0.26 / 0.31 / 0.49 / 0.61; the model's window better than the vector's by more than 3 % + 0.05 in 0 / 12 / 21 / 26 % of the blocks, about equal in 15 / 20 / 15 / 16 %, worse in 83 / 67 / 62 / 56 %; the mean voter confidence of these blocks 0.25 / 0.27 / 0.45 / 0.56. In the fits (`v35_stats`, `v35_rgn`) such a block's weight is its texture only (`smoothstep(1, 3.5, detail)`, times for the region fit the structure tensor's smaller eigenvalue): a zero-vector block with detail counts in full. Level 1 sample weights of one pair (Forza 000560, bottom row of the 4 x 3 regions, still (< 1 flow px) / moving): 39 / 165, 410 / 197, 465 / 132, 198 / 76: the still samples (car, speedometer, zero road) outweigh the moving ones in the middle and right of the bottom row two to three to one.

**Was the reading right?** Half of it. Right: the mechanisms are there as described (the 3 % rule and the HUD hold between them leave 89 / 74 / 72 / 75 % of the zero blocks at levels 0 .. 3 at zero), and the road is indifferent to V3's cost: costs at zero of 0.5 to 2 levels, flat curves (the same on raw luma: 80 x 80 px windows on the asphalt of Forza 000560 have an L2 error of 2.5 at the best vector, 2.7 at zero and a median of 3.1 over +-40 px). Wrong: that the model would carry the road once the blocks follow it. The model is as wrong on the road as the field. Spots of Forza 000560 (full-resolution px per 1/30 s; truth from the real in-between frame, prev -> mid and mid -> curr add up): the road right of the centre line (1500, 800): truth (38, 30), field (1.7, -1.7), model (0.8, -0.4) with reliability 0.99; (1700, 700): truth (22, 7), field (0.5, -0.6), model (0.7, -1.0), reliability 0.97; (1300, 1000): truth (44, 79), field (2.4, 0.3), model (2.5, 0.5), reliability 1.0. The model's reliability measures how well its own samples agree, and its samples there are the car, the speedometer and the zero road. Over the 913 patches (EPE in px, mean / median, share within 4 px): V3 32.5 / 14.8 / 35.0 %, V3.5 30.8 / 13.0 / 36.3 %, **the model of V3.5 38.5 / 27.5 / 13.9 %**.

The aperture problem is total on the road: lane lines run along the motion, so a block on a line measures "no motion across the line", which is true; the motion along it is in no block.

### An attempt that did not hold (not committed)

Rule: a block may refuse the vote or the model only where the image positively prefers its own vector (per-pixel evidence: at least 3 of the 16 window samples clearly better at its own vector by 1.5 + 0.3 detail levels and not fewer than a fifth of those that prefer the other); indifferent blocks neither vote, nor hold their vector, nor refuse the model; the model fits of a region with a substantial still part (a quarter of the weight and at least 60 moving sample weights) are redone on the moving samples only; the dense pass puts a 2.5-level handicap on V3's own vectors. Result: "still against the model" 6.6 % -> 4.2 % (3.1 % against today's model), the sheets show the field following the model, and the model is still the one above. Against the sparse truth: V3.5 median EPE 13.0 -> 16.3 px, patches within 4 px 36.3 % -> 32.3 % (model 27.5 -> 26.4 median): worse. Following a wrong model moves the road to a wrong vector instead of a zero one. The patch is not in the tree.

**What blocks it.** The road carries no evidence at the scale V3 matches (high-pass features, 8 x 8 windows): at 1/30 s the asphalt is smeared, dark and low in contrast, and the lane lines run along the motion. The informative blocks of the lower regions are the car and the overlay, both still. A model fitted from the blocks cannot know the road, and neither can a vote. What would: a model with geometry (the ground plane's motion about the focus of expansion: in Forza 000560 the truth fits d_y ~ (y - 470)^2 within 2 % at two points), or matching the road on raw luma at coarse levels, where the true half-motion is a distinct minimum (error 2.1 against 4.3 at zero and a median of 6.8, at (1500, 800), prev -> mid) while V3's high-pass channels have nothing left. Neither is a rule of the regularisation; neither was started.

## The real ground (task 162)

The question of task 161's last section: which matching can find the motion of a blurred road at the coarse levels? This task measured it offline against the whole frame, then tried the best answer inside V3.5. **The study is committed; the V3.5 change is not** (it meets every number below except one test threshold, see "What blocks it"): `census_candidate/` holds the shader and the patch against the library files, for whoever decides to take the last step.

### The extended truth and the offline study

- `common.frame_truth` is `road_truth` over the whole frame (64 px grid, 64 x 64 px patches, a margin of 32 px, prev -> mid and mid -> curr agree within 3 px, distinct match, texture), `ground.py --truth-only` writes it for every pair of a list that has a real in-between frame. Groups: **ground** = the lower half of the driving pairs (the 12 Forza held-out pairs, the 12 `road` pairs, the Forza scene pairs; 2158 patches in 26 pairs), **drive-up** = their upper half (4704), **tp** = the third-person pairs (1791), **game** = the other real pairs (cs2, Minecraft, Battlefield, first person; 1618).
- `study_match.py` computes, per pair, level (L5 .. L1 of V3's pyramid at `flowScale` 2) and patch, the cost over an exhaustive +-12 level px search of each feature x window variant and reports whether the cost minimum lies within 1 level px of the rounded truth ("findable"). Variants: V3's own feature (raw minus a 5x5 mean) with 4, 8, 12, 16 px windows (`v3_N`), a 9- and 17-tap high-pass (`hp9`, `hp17`, `hpup`), zero-mean SAD with the lighting gain (`zs`), zero-mean normalised correlation (`zncc`), and a census transform (`census`: ternary code of the 8 neighbours at +-2 level px on the raw luma, Hamming cost). `study_table.py` aggregates the result; `--light up|down` repeats the study with the second frame brightened (gain 1.10, +14 levels) or darkened (0.90, -14) to test the lighting tolerance (the rows below are the unchanged case; on the ground pairs alone, with the second frame brightened / darkened, the census with window 8 / 12 still finds 82 / 86 % and 83 / 84 % of the moving patches at L2, V3's feature 63 / 67 % and 55 / 58 %).

Share findable among the moving patches (true vector at least 1 level px), per level, ground / rest (rest = every patch outside the ground group), window 8 and 12 level px:

| feature, window | L4 | L3 | L2 | L1 |
|---|---|---|---|---|
| V3's feature, 4 | 59 / 82 | 58 / 78 | 56 / 76 | 54 / 71 |
| V3's feature, 8 | 64 / 90 | 62 / 92 | 72 / 90 | 71 / 86 |
| V3's feature, 12 | 67 / 90 | 60 / 94 | 69 / 94 | 78 / 91 |
| high-pass 9, 8 | 63 / 91 | 62 / 93 | 72 / 92 | 74 / 88 |
| zero-mean SAD, 8 | 66 / 89 | 72 / 91 | 79 / 90 | 72 / 85 |
| ZNCC, 12 | 69 / 89 | 65 / 93 | 74 / 94 | 78 / 90 |
| census, 8 | 69 / 91 | 73 / 93 | 80 / 91 | 73 / 87 |
| census, 12 | 68 / 92 | 71 / 95 | 80 / 94 | 79 / 91 |

**Reading.** (1) Every variant finds more than half of the moving ground patches at the levels L2 .. L4, so the fallback of the brief (a ground-plane model) is not needed: the evidence exists, V3 does not use it. (2) The reason is not the feature: V3's own feature with an 8 px window finds 72 % at L2; V3 only searches **candidates** (parent x 2, zero, the camera, dominant motions, a ring of +-1), and the true road vector (several level px away from the parent's, from zero and from the camera, because the fits are dominated by the still blocks) is never a candidate. Exhaustive search around zero at the coarse level and around the parent's vector below is what gives the road its motion. (3) Among the features the census is the best or equal at L2 / L3 (80 / 73 % on the ground) and the one that stays best under the lighting change; its confidence (distance to the second-best vector, margin over the parent's vector) is cheap. A window of 8 is as good as 12 and a third of the memory. Choice: **census, 8 x 8 level-px window, exhaustive, a hierarchical chain** (+-12 around zero at L4, +-6 around the parent's vector at L3 and L2, +-2 at L1).

### What was built (the candidate, `census_candidate/`)

- `v35_gsearch.comp` (new, 64 invocations, one block per workgroup, `GMAX` 6): census codes of the window (frame f0) and of the target region (frame f1) in shared memory, the cost of all (2 R + 1)^2 candidates, best vector, confidence `smoothstep(0.03, 0.06, second_best - best) * smoothstep(0, 0.04, cost_at_centre - best)`; writes (vector, confidence, cost) into slot 1 of the level's block records (free until iteration 1).
- `v3_search.glsl` (`#if V35` only): iteration 0 takes the record as candidate 11 (no distance penalty, a bonus of 0.6 x confidence in cost levels); `-DV35` compile of `v3_search.comp` (`v3_search_35`), so V3's SPIR-V is unchanged.
- `framegen.cpp`: the pass runs at levels 3 and 2 (+-6 around the parent's vector x 2), before the level's search; V3.5's `v35_stats` runs only at level 3 (the level that finds the dominant motions) and level 1, `v35_reg` is dropped at levels 2 and 3 (the census vectors are the better evidence there, and the passes cost 0.3 ms); two new lazily created pipelines (`v35_gsearch`, `v3_search_35`; V3.5 has six, V3 itself is unchanged). The unconstrained chain (+-12 at L4, +-6 at L3 and L2, +-2 at L1, all of V3.5's stats and regularisation passes kept) scores better (ground median 3.9 px, within 4 px 50.4 %, still 1.1 %; synthetic mean 0.61 px) at 1.87 ms per pair and does not fit the 1.5 ms mark.
- A first version of the chain (four passes, +-12 / 6 / 6 / 2) cost 1.48 ms more because the pass is latency bound (a block per group, large shared memory, many groups at the fine levels); smaller ranges, a 64-invocation group and a third of the shared memory brought a pass down to 0.05 .. 0.15 ms. Stride-2 sampling of the window lost 2.5 points and was dropped.

### Numbers (the candidate against V3 and today's V3.5)

Ground truth, `ground.py` (`mean px / median px / within 4 px / still where the truth is above 4 px`), patches as above:

| group | V3 | V3.5 today | candidate |
|---|---|---|---|
| ground (2158) | 33.65 / 13.19 / 33.6 % / 10.2 % | 31.80 / 12.10 / 36.0 % / 4.3 % | 23.60 / 4.75 / 46.5 % / 2.6 % |
| drive-up (4704) | 4.22 / 1.11 / 82.1 % / 0.3 % | 3.98 / 1.07 / 84.2 % / 0.1 % | 3.63 / 1.07 / 85.3 % / 0.1 % |
| tp (1791) | 5.16 / 0.82 / 84.8 % / 0.2 % | 4.77 / 0.80 / 87.0 % / 0.2 % | 4.82 / 0.81 / 87.0 % / 0.4 % |
| game (1618) | 7.70 / 1.36 / 75.5 % / 2.0 % | 6.06 / 1.05 / 81.0 % / 0.5 % | 5.75 / 1.05 / 83.6 % / 0.3 % |

The ground median falls from 12.1 to 4.75 px (the aim was 6 px), the share within 4 px rises from 36.0 to 46.5 % (the aim was 55 %: not met), the still share on the ground from 4.3 to 2.6 % (limit 3.7 %). No other group is worse by more than 0.01 px median or 0.4 points of the still share. The 72 synthetic + 23 held-out pairs (`measure.py`, `v3Motion = 1`): end-point error lower than V3 in all 24 rows, mean 0.79 px (today 0.96, V3 3.86); real PSNR against V3 battlefield +0.68, cs2 +0.08, forza +0.36, minecraft +0.34 dB (today +0.50, 0.00, +0.10, +0.09: no group loses against today); per pair against today between -0.08 dB (a cs2 pair) and +0.90 dB (a Minecraft pair). Cost, 1080p Rgba8 2x, `fgtest --bench 200`, median of 3: V3 0.954, today's V3.5 1.471, candidate 1.432 ms per pair.

### Round 2: the candidate after the guard (not committed, `census_candidate/` is the best variant)

Changes to the candidate of round 1 (patch and shader in `census_candidate/` are the state described here):
- **Code**: a thermometer code (four thresholds -2, -0.6, +0.6, +2 levels on the difference to each of the 8 neighbours at +-2 level px, 32 bits) instead of the hard ternary; the cost is still a popcount. Rgb10 against Rgba8 passes the 40 dB mark (50.8 dB; the hard ternary gave 37.6). A 15-step soft code passed too but cost 0.15 ms more (1.55 ms).
- **Still-content guard** in `v35_gsearch`: `w = sum over the 12 x 12 window of detail x exactMatch`, detail = `smoothstep(24, 64, sum of the four neighbour differences)`, exactMatch = `1 - smoothstep(1, 3, |f1 - f0|)` at zero; `conf *= 1 - smoothstep(2, 8, w)`. `v3test --v3-motion 1` `hud(60,20)` static overlay: 6.44 / 6.46 (limit 7; round 1 7.15 / 7.23). A guard on weak detail or a ratio form also stops the road (ground within 4 px 29 .. 36 %), the sharp-detail form costs about one point.
- **Consensus** in `v3_search` (`#if V35`): the census vector is taken only if one of the 8 neighbouring blocks' census vectors is within 1.5 level px of it.
- Confidence ramps 0.01 / 0.03 (second best minus best), smear 0.04, candidate bonus 0.6.

Numbers (`ground.py`, full 64 pairs; mean / median / within 4 px / still share): ground 31.80 / 12.10 / 36.0 % / 4.3 % today -> 20.01 / 4.20 / 48.8 % / 1.8 %; drive-up 3.98 / 1.07 / 84.2 % -> 3.65 / 1.07 / 85.0 %; tp 4.77 / 0.80 / 87.0 % / 0.2 % -> 5.39 / 0.82 / 86.9 % / 0.3 %; game 6.06 / 1.05 / 81.0 % -> 6.21 / 1.05 / 83.4 %. Cost 1080p Rgba8 2x 1.466 ms (limit 1.5; today 1.471). Byte identity at `v3Motion = 0` 84 of 84, validation 49 runs with 0 messages, 252 format runs clean, all test programs as today (the `fs1` and `fs4` marks that fail today fail alike); the rim band is within +0.2 of today (cs2_chase_l +0.17, mc_chase_l +0.17).

**Why it is not committed (gates of the owner's priority: first- and third-person games must look as today's V3.5).**
1. Isolated blocks whose vector differs from the 3x3 median by more than 4 flow px (`blobcmp.py`, 8 px blocks, 35 non-driving pairs): today 50241, candidate 53975, and 28 of 35 pairs are above today's count by more than 2 (worst `tp_m05_60fps` 867 -> 1476, `tp_m02_60fps` 739 -> 1323, `minecraft120_02_002760` 186 -> 388, `battlefield_003120` 893 -> 1254). Part of this is real structure the census finds (the game ground group improves in mean error), but it is what the sheets show as specks.
2. Per-pair PSNR against today's V3.5 on the non-driving real pairs: `tp_n264_30fps` -5.67 dB (its 60 fps pair +7.70; both pairs sit near a scene-cut decision, V3 16 dB), `battlefield_003120` -1.61 dB (the red band at the counter edge). The owner's limit is -0.10 dB.
3. HUD pixels, Forza WholeScreen: `flowScale` 2 5.92 against today's 5.74 and `flowScale` 4 5.83 against 5.68 (limit +0.1); one pair (`racing_forza6_c_000020`) 8.38 -> 9.99.
4. tp ground group: median 0.82 against 0.80, still share 0.3 against 0.2 %.
Restrictions that make the candidate act only where the parent's vector is near zero (the road) were tried: with today's regularisation schedule restored they cost 1.73 ms, fail `hud(60,20)` (9.2) and keep `tp_n264_30fps` at -5.67 dB, `battlefield_003120` at -1.79; without it the blobs fall only from 53975 to 53816. The blobs do not come from the candidate alone: dropping `v35_reg` at levels 2 and 3 (to pay for the census pass) already gives 54546 with the candidate forced almost inert, restoring the schedule gives 51197 (still above today's 50241), so the census chain cannot be paid for inside 1.5 ms without changing what the non-driving scenes look like.
Not tried for lack of a fit in the budget: feeding the confidence into the vote of `v35_reg` and the fit of `v35_rgn`, a level-4 pass, the regional model as the centre of the coarse search. Level 3 / 2 reach (+-6 level px = +-48 / +-24 flow px around the parent) and the missing spreading to flat asphalt beside a found patch are the reasons for the unfound bottom strip and the plain lane asphalt in the sheets.

### Run the extended measure

```
python motion/ground.py --pairs pairs.json --truth TRUTHDIR --truth-only                 # writes TRUTHDIR/<pair>.npy (once)
python motion/ground.py --pairs pairs.json --truth TRUTHDIR --out OUT \
       --run v3=FGTEST:algo=v3 --run new=FGTEST:algo=v3,v3Motion=1                       # OUT/ground.md, OUT/ground.json, OUT/fields/<pair>_<tag>.npz
python motion/maps.py --ground OUT --truth TRUTHDIR --fgtest NEW --ref-fgtest TODAY --out SHEETS --pairs pairs.json --only name,name,...   # sheets 1800 px wide: prev | V3 | today; new | truth arrows (green: within 4 px) of new | of today; real | generated-vs-real error of today | of new
python motion/study_match.py --list pairs.json --truth TRUTHDIR --out study.pkl [--light up|down]; python motion/study_table.py study.pkl --md table.md
```

## V3.5 cheaper (task 169)

V3.5's added GPU time over V3 at flowScale 2 / subpixel on (Quality) was +0.43 ms at 1440x1080 and +0.46 ms at 1080p, at flowScale 4 / subpixel off (Performance) +0.24 / +0.28 ms. The goal was +0.25 / +0.15 ms. Reached: **+0.30 / +0.33 ms Quality, +0.15 / +0.20 ms Performance** (`fgtest --bench 200`, Rgba8 2x, median of 3 alternating runs, noise +-0.03 ms). The target is met only for Performance at 1440x1080; the schedule below is the cheapest one that passes the quality gates.

What was changed (`gpu/V3.md` section 3.18 "Cheaper" has the schedule as built and the cost table for 1440x1080, 1080p and 1440p at 2x / 3x / 4x): `v35_reg` runs one Jacobi step at every level except the one below the top (two), 12 instead of 15 added dispatches (Performance 9 instead of 11); the regularised records of a one-step level stay in slot 0, so `v3_search` is compiled once more with `-DV35=1` (`v3_search_35`) to read its parent from there; `v35_reg` itself got a cache for the blocks' static data, a branch-free window cost and one window cost less, with a bit-identical output (0.32 -> 0.26 ms for the old schedule). The two passes that remain expensive are the single-workgroup `v35_stats` (0.09 ms at Quality) and `v35_rgn` (0.05 ms): latency-bound, and each level of them carries a group (minecraft120 / forza / cs2) of the quality gates. The pass list grows by one compile variant of an existing source (five pipelines instead of four). The debug dump of the level stages L1..L3 shows the field before the regularisation for the one-step levels.

Every variant (cost over V3 at 2x; quality at the preset's own flowScale on the quick set: 72 known-motion pairs for the EPE, 23 held-out + scenes_all pairs for the dB, the blobs of the non-driving real pairs; "gates" = no game group below V3, no non-driving pair more than 0.10 dB below V3, at least 70 % of today's gain per group with a gain of 0.05 dB or more, EPE lower than V3 in all 24 rows, blobs at most today +3 % and no pair above V3's count where today is below it):

### Quality preset (flowScale 2, subpixel on): added cost over V3 and the quality quick-set
| variant | steps per level 0.. / stats levels / rgn levels | dispatches | +ms 1440x1080 | +ms 1080p | EPE mean | battlefield / cs2 / forza / minecraft120 / scene (dB vs V3) | pairs < -0.10 | blobs vs today | blob pairs over V3 | gates |
|---|---|---|---|---|---|---|---|---|---|---|
| today: today: 2 steps at levels 1..4, 1 at 0; stats + rgn at 3, 2, 1 | 122220; 123; 123 | 15 | +0.421 | +0.468 | 0.963 | +0.496 / -0.004 / +0.100 / +0.092 / +0.068 | 0 | 1.000 | 0 | ok |
| v_opt: today's schedule, shaders only optimised (static data cache, branch-free window cost, one window cost less): output bytes identical | 122220; 123; 123 | 15 | +0.357 | +0.386 | same bytes as today |  |  |  |  |  |
| a1: 1 step at every level | 111110; 123; 123 | 11 | +0.291 | +0.303 | 0.988 | +0.409 / +0.006 / +0.087 / +0.077 / +0.066 | 1 | 1.034 | 1 | FAIL |
| a2: no reg at level 4 | 122200; 123; 123 | 13 | +0.354 | +0.397 | 0.999 | +0.422 / -0.002 / +0.096 / +0.034 / +0.065 | 1 | 1.023 | 1 | FAIL |
| a3: no reg at levels 4, 3 | 122000; 123; 123 | 11 | +0.280 | +0.339 | 1.150 | +0.347 / +0.009 / +0.078 / +0.009 / +0.065 | 1 | 1.040 | 2 | FAIL |
| a4: stats + rgn at 2, 1 only | 122220; 12; 12 | 13 | +0.337 | +0.374 | 1.039 | +0.570 / +0.041 / +0.117 / +0.110 / +0.062 | 2 | 0.947 | 2 | FAIL |
| a5: stats + rgn at 1 only | 122220; 1; 1 | 11 | +0.273 | +0.257 | 1.213 | +0.491 / -0.064 / +0.115 / +0.127 / +0.064 | 2 | 0.945 | 3 | FAIL |
| a7: 1 step at levels 1, 2; 2 steps at 3, 4 | 111220; 123; 123 | 13 | +0.294 | +0.334 | 0.980 | +0.492 / +0.002 / +0.088 / +0.087 / +0.068 | 0 | 1.004 | 0 | ok |
| a8: 2 steps at 1, 2; 1 step at 3, 4 | 122110; 123; 123 | 13 | +0.336 | +0.390 | 0.971 | +0.400 / +0.006 / +0.100 / +0.081 / +0.066 | 1 | 1.031 | 1 | FAIL |
| a9: a7 + stats + rgn at 2, 1 only | 111220; 12; 12 | 11 | +0.293 | +0.328 | 1.062 | +0.571 / +0.036 / +0.117 / +0.106 / +0.062 | 2 | 0.950 | 1 | FAIL |
| a10: a7 without the level-0 step | 011220; 123; 123 | 12 | +0.334 | +0.375 | 4.562 | +0.231 / +0.067 / -0.333 / +0.064 / -0.060 | 6 | 1.276 | 17 | FAIL |
| a11: a7 + stats + rgn at 1 only | 111220; 1; 1 | 9 | +0.238 | +0.253 | 1.243 | +0.485 / -0.059 / +0.114 / +0.126 / +0.064 | 2 | 0.948 | 4 | FAIL |
| r1: a7, rgn at 2, 1 only (stats at 3, 2, 1) | 111220; 123; 12 | 12 | +0.304 | +0.344 | 0.963 | +0.590 / -0.007 / +0.071 / +0.033 / +0.068 | 0 | 1.014 | 2 | FAIL |
| r2: a7, rgn at 1 only | 111220; 123; 1 | 11 | +0.302 | +0.306 | 0.965 | +0.532 / +0.008 / +0.043 / +0.031 / +0.068 | 0 | 1.034 | 4 | FAIL |
| z1: a7, rgn at 3 and 1 only | 111220; 123; 13 | 12 | +0.302 | +0.357 | 0.995 | +0.496 / +0.006 / +0.085 / +0.089 / +0.068 | 0 | 1.015 | 1 | ok |
| z2: a7, stats + rgn at 3 and 1 only | 111220; 13; 13 | 11 | +0.256 | +0.311 | 1.031 | +0.434 / +0.006 / +0.061 / +0.089 / +0.068 | 0 | 1.013 | 1 | FAIL |
| m1: 1 step at 1, 2, 3; 2 steps at 4 is replaced by 1 (1 step at 4) | 111210; 123; 123 | 12 | +0.308 | +0.351 | 0.981 | +0.492 / +0.001 / +0.088 / +0.087 / +0.066 | 1 | 1.032 | 1 | FAIL |
| m2: 1 step at 1, 2, 3; 2 steps at level 4 only = THE COMMITTED SCHEDULE | 111120; 123; 123 | 12 | +0.309 | +0.338 | 0.986 | +0.409 / +0.007 / +0.087 / +0.077 / +0.068 | 0 | 1.006 | 0 | ok |
| final: committed build (m2 as built, no knobs) | 111120; 123; 123 | 12 | +0.304 | +0.318 | 0.986 | +0.409 / +0.007 / +0.087 / +0.077 / +0.068 | 0 | 1.006 | 0 | ok |

### Performance preset (flowScale 4, subpixel off)
| variant | steps per level 0.. / stats levels / rgn levels | dispatches | +ms 1440x1080 | +ms 1080p | EPE mean | battlefield / cs2 / forza / minecraft120 / scene (dB vs V3) | pairs < -0.10 | blobs vs today | blob pairs over V3 | gates |
|---|---|---|---|---|---|---|---|---|---|---|
| today: today: 2 steps at levels 1..3, 1 at 0; stats + rgn at 2, 1 | 12220; 12; 12 | 11 | +0.254 | +0.253 | 1.352 | +0.336 / +0.039 / +0.079 / +0.076 / -0.016 | 3 | 1.000 | 0 | FAIL |
| v_opt: today's schedule, optimised shaders only | 12220; 12; 12 | 11 | +0.211 | +0.248 | same bytes as today |  |  |  |  |  |
| p0: same with the env knobs (control) | 12220; 12; 12 | 11 | +0.233 | +0.218 | same bytes as today |  |  |  |  |  |
| p1: 1 step at 1, 2; 2 steps at 3 (committed) | 11120; 12; 12 | 9 | +0.176 | +0.197 | 1.381 | +0.328 / +0.035 / +0.084 / +0.069 / -0.016 | 3 | 1.003 | 1 | FAIL |
| p2: p1 + stats + rgn at 1 only | 11120; 1; 1 | 7 | +0.125 | +0.163 | 1.521 | +0.342 / -0.008 / +0.128 / +0.081 / -0.049 | 4 | 0.991 | 2 | FAIL |
| p3: 1 step at 3 as well | 11110; 12; 12 | 8 | +0.166 | +0.183 | 1.384 | +0.328 / +0.035 / +0.084 / +0.069 / -0.015 | 3 | 1.004 | 1 | FAIL |
| final: committed build | 11120; 12; 12 | 9 | +0.179 | +0.174 | 1.381 | +0.328 / +0.035 / +0.084 / +0.069 / -0.016 | 3 | 1.003 | 1 | FAIL |

Quality-preset gates, committed schedule against today and V3: EPE lower than V3 in 24 / 24 rows, mean 0.986 px (today 0.963; limit 1.20); battlefield +0.41 dB (today +0.50, 82 %), forza +0.087 (+0.100, 87 %), minecraft120 +0.077 (+0.092, 84 %), cs2 +0.007 (-0.004), scene +0.068 (+0.068); no pair more than 0.10 dB below V3; blobs 50554 against today's 50241 (+0.6 %) and V3's 51985, no pair above V3 where today is below. Performance preset: the same as today's Performance within the noise of the gates; today's Performance already has three pairs 0.2 to 0.3 dB below V3 (`battlefield_003240_60t`, `tp_n153_30fps`, `tp_n153_60fps`), and one battlefield pair has 3 blobs more than V3 where today's has not.

What the next 0.1 ms would cost: see the last paragraph of section 3.18 "Cheaper" in `gpu/V3.md`: removing `v35_stats` + `v35_rgn` at level 3 (-0.05 ms) already leaves two pairs 0.10 dB below V3; at levels 3 and 2 (-0.1 ms) cs2 goes to -0.06 dB; dropping the second step at the level below the top loses `fps_cs2c_960` 0.5 dB; dropping the level-0 step breaks every pair.

The census candidate of the previous section (`census_candidate/census_candidate.patch`, not applied) names its search compile variant `v3_search_35` and its pipeline `P_V35_SEARCH` as well: both now exist in the library for the schedule above, so the patch needs these two hunks adapted (its search shader changes would go into the `-DV35` branch of `v3_search.comp` that now exists) before it applies.
