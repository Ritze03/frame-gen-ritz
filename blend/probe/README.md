# blend/probe: does a net that sees neighbouring pixels choose a better V3 mix than a per-pixel net? (task 143)

Offline, Python only, no library change. Scores are at the weight stage (sum of weights x points, before the library's clamp, smear and static mask), like the training scores.

## Pipeline (run in this order; `BLEND` = data root, outputs under `$BLEND/probe/`)
1. `select_pairs.py`: 56 training pairs (16 racing_forza6_a, 16 racing_forza6_b, 8 cs2_a, 8 minecraft120_02 60to120, 8 battlefield 60to120) and 23 held-out pairs (12 racing_forza6_c, 2+2 cs2_a/cs2_b, 2+2 minecraft120_02/_04 60to120, 3 battlefield 60to120): `pairs.json` (here and in `$BLEND/probe`). All real, t = 0.5, manifest split val ones taken from `real_val_ids.json`.
2. `convert.py`: `fgtest --v3weights --v3wmax 2073600` (stride 1, 4 pairs per `flock gpu.lock ccorral` call) -> `$BLEND/probe/pairs/NN/{X,taps,e,y,rule}.npy` (X f16 [1080,1920,34], taps f16 levels, e u8 prev/curr, y u8, rule f16); held-out also `baseC.npy`, `forzaC.npy` (their six weights). Raw record deleted.
3. `oracle_pairs.py`: O4 / O6 colours (oracle.py's `solve`) for the held-out pairs -> `o4.npy`, `o6.npy`.
4. `nets.py` + `train_probe.py` (+ `run_all.sh`): the six nets, 6,000 steps each, Charbonnier vs the real frame, batch 16 random 128x128 crops (read with a 2 px context margin, loss on the centre), Adam 2e-3 -> 1e-4 cosine, seed 0, identical crops in identical order for every net, six choices, logits = log(max(rule,1e-3)) (-5, -5 for prev(x), curr(x)) + net output, last layer zero at the start (as `blend/train/train.py`). Weights in `$BLEND/probe/nets/`.
5. `score.py`, `report.py` (-> `tables.md`), `pictures.py` (-> `$BLEND/probe/img/`), `check_train_loss.py` (in-sample sanity).

## Result (23 held-out pairs; PSNR dB, mse pooled over pixels and RGB per group)
Whole picture / edge pixels (edge mask: luma of the real frame, Sobel magnitude, max over 5x5 >= 90th percentile of the frame's G). Full tables incl. >30-level shares, gains, gap-closure: `tables.md`.

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| rule | 30.86 / 26.23 | 28.79 / 25.23 | 40.05 / 35.26 | 32.36 / 25.93 | 31.24 / 26.75 |
| base C | 30.77 / 26.06 | 29.18 / 25.41 | 40.23 / 35.42 | 32.55 / 25.98 | 31.33 / 26.72 |
| forza-rx9070xt C | 30.71 / 25.96 | 29.26 / 25.51 | 40.40 / 35.61 | 32.59 / 26.01 | 31.33 / 26.69 |
| px8 | 30.51 / 25.88 | 29.06 / 25.33 | 40.75 / 36.02 | 32.41 / 25.88 | 31.14 / 26.59 |
| px32 | 30.52 / 25.87 | 29.05 / 25.32 | 40.69 / 35.94 | 32.43 / 25.89 | 31.15 / 26.58 |
| n3-all | 30.46 / 25.79 | 29.08 / 25.32 | 40.71 / 35.97 | 32.43 / 25.89 | 31.11 / 26.54 |
| n5-all | 30.45 / 25.77 | 29.02 / 25.28 | 40.80 / 36.06 | 32.43 / 25.88 | 31.09 / 26.52 |
| n3-frames | 30.48 / 25.82 | 29.06 / 25.34 | 40.67 / 35.92 | 32.38 / 25.86 | 31.11 / 26.56 |
| n3-taps | 30.43 / 25.76 | 29.08 / 25.34 | 40.77 / 36.04 | 32.36 / 25.83 | 31.09 / 26.52 |
| O4 | 34.89 / 30.34 | 31.54 / 27.49 | 42.53 / 37.64 | 35.27 / 28.65 | 34.69 / 30.02 |
| O6 | 37.37 / 33.45 | 34.35 / 30.31 | 49.45 / 45.49 | 40.98 / 34.40 | 37.65 / 33.41 |

Answer: no network that sees neighbours beats the per-pixel one on held-out pairs in this setup: every gain over px8 is between -0.12 and +0.04 dB (whole and edge), i.e. 0 % of the px8 -> O6 gap closed (between -1.6 % and +0.5 %). In-sample (160 training crops, `check_train_loss.py`) the nets do improve slightly: rule-start 29.38 dB, px8 29.57, px32 29.67, n3-all 29.74. So the neighbourhood buys a little on training crops and nothing on held-out pairs. Caveat: even px8 is not better than the rule on held-out pairs (31.14 vs 31.24 dB), so this training recipe (real-frame target, 6,000 steps, 56 pairs) extracts almost nothing beyond the rule at all; the null result may be a property of the recipe as much as of the information content. See the report for details.

Sanity: rule on the 12 forza held-out pairs is 30.86 dB (stride 1), 30.52 dB on the same pairs at the cache's sparse pixels (stride 64, n=32400 per pair); the stored arrays equal the sparse records at the sampled pixels (taps within f16 rounding 0.0625, rule weights 2.4e-4, target exactly), so the stride-1 path is the cache path; the 29.9 figure is the whole forza val set (74 pairs, 2.79 M sparse rows, `records/measure.md`), and per-pair differences between sparse and stride-1 reach 1.1 dB (sampling noise). O6 <= O4 <= every net (in squared error) holds on all 23 pairs.

## Follow-up (task 144): block-constant ceiling, O6 target, seed noise
Scripts: `block_oracle.py` (Part 1), `make_o6crops.py` + `train_probe2.py` + `score2.py` (Parts 2, 3), `report2.py` -> `tables2.md` (all tables incl. four-candidate rows, per-group, checkerboard, edge shares). New data under `$BLEND/probe/`: `block_oracle/NN.json`, `o6crops/` (crops.json, 56 x [20,128,128,3] f16), `nets/{px8-o6,n3-all-o6,px8-fix,n3-all-fix,px8-s1}.pt`, `scores2.json`. Numbers below: all 23 pairs pooled, whole picture / edge pixels, PSNR dB.

**Part 1: block-constant oracle.** Solver = oracle.py's subset enumeration on the block's Gram data (G = sum c c^T, b = sum c y, yy). Check at B = 1 on pair 56, rows 400..520 (230,400 pixels): colour after float16 rounding equals the stored `o6.npy` and `o4.npy` exactly (max difference 0.0000 levels, MSE 3.7390 = 3.7390 and 5.5799 = 5.5799).

| method (six candidates) | all | gain over rule |
|---|---|---|
| rule | 31.24 / 26.75 | |
| px8 (task 143) | 31.14 / 26.59 | -0.10 / -0.16 |
| whole frame, 1 weight vector | 31.79 / 27.26 | +0.55 / +0.51 |
| block 32 | 32.85 / 28.16 | +1.61 / +1.41 |
| block 16 | 33.41 / 28.65 | +2.17 / +1.90 |
| block 8 | 34.18 / 29.37 | +2.94 / +2.62 |
| block 4 | 35.13 / 30.37 | +3.89 / +3.62 |
| block 2 | 36.32 / 31.74 | +5.08 / +4.99 |
| per-pixel O6 | 37.65 / 33.41 | +6.41 / +6.66 |
| four taps: whole / 32 / 16 / 8 / 4 / 2 | 31.63 / 32.25 / 32.59 / 33.02 / 33.52 / 34.09 (edges 27.07 / 27.51 / 27.80 / 28.19 / 28.70 / 29.33) | +0.39 / +1.01 / +1.35 / +1.78 / +2.28 / +2.85 whole |
| per-pixel O4 | 34.69 / 30.02 | +3.45 / +3.27 |

Checkerboard (fit on even pixels of each block, score on the odd pixels; rule on odd pixels 31.24 / 26.75): B = 16 33.37 / 28.60 (plain fit 33.41 / 28.65), B = 8 34.06 / 29.23 (plain 34.18 / 29.37, i.e. +2.82 / +2.48 over the rule survives), B = 4 34.73 / 29.89 (plain 35.13 / 30.37, +3.49 / +3.14 survives). Four candidates, B = 8: 32.95 / 28.10 (plain 33.02 / 28.19).
Where the gain sits (squared-error reduction vs the rule; edge pixels are 25.8 % of pixels): B = 8 removes 49.1 % of the rule's squared error, 66.8 % of that reduction is at edge pixels and 33.2 % elsewhere; B = 4 removes 59.1 %, 69.4 % at edges / 30.6 % elsewhere; O6 77.1 %, 73.8 % / 26.2 %. Per region B = 8 removes 45.2 % of the squared error inside the edge mask and 59.4 % outside it.

**Part 2: training toward the O6 colour** (fixed crops: 20 random 128x128 per training pair, rng seed 144, 1,120 crops; their O6 colour by the same solver; loss Charbonnier(mix colour, O6 colour), everything else as `train_probe.py`; `-fix` = same crops, real-frame target; scored against the real frame):

| net | all | forza | cs2 | minecraft120 | battlefield |
|---|---|---|---|---|---|
| rule | 31.24 / 26.75 | 30.86 / 26.23 | 28.79 / 25.23 | 40.05 / 35.26 | 32.36 / 25.93 |
| px8 (143) | 31.14 / 26.59 | 30.51 / 25.88 | 29.06 / 25.33 | 40.75 / 36.02 | 32.41 / 25.88 |
| px8-fix | 31.28 / 26.70 | 30.69 / 26.02 | 29.13 / 25.40 | 40.64 / 35.89 | 32.52 / 25.94 |
| px8-o6 | 31.26 / 26.68 | 30.65 / 25.98 | 29.15 / 25.41 | 40.66 / 35.91 | 32.52 / 25.95 |
| n3-all-fix | 31.21 / 26.58 | 30.63 / 25.87 | 29.05 / 25.33 | 40.55 / 35.78 | 32.41 / 25.87 |
| n3-all-o6 | 31.16 / 26.54 | 30.55 / 25.79 | 29.07 / 25.35 | 40.61 / 35.85 | 32.40 / 25.85 |
| px8-s1 (seed 1, 143 recipe) | 31.15 / 26.61 | 30.53 / 25.92 | 29.04 / 25.32 | 40.74 / 36.00 | 32.42 / 25.88 |

`n5-all-o6` was not trained (budget; dropped first as agreed).
**Part 3: seed noise.** px8 seed 0 vs seed 1 (torch init and crop order): 31.14 / 26.59 vs 31.15 / 26.61, differences 0.01 / 0.02 dB (largest per-group difference 0.04 dB, forza whole 30.51 vs 30.53).

## Probe 3 (task 147): can the best 8x8 block mix be predicted by a block-level net?
Scripts: `block3_build.py` (block table), `block3_check.py`, `block3_rulepix.py`, `block3_base.py`, `block3_lib.py`, `block3_train.py`, `block3_all.sh` / `block3_reg.sh` (runs), `block3_report.py` -> `tables3.md` (all tables incl. per group, edge, training set). Data: `$BLEND/probe/block3/` (`NN.npz` block tables, `base.json`, `rule_pix.json`, `res_TAG.json`, `net_TAG.pt`). The pixel data was read once (table build, rule scores); training and scoring use only the block table.

Method. B = 8 (135 x 240 blocks per pair). Per block: Gram data G (6x6), b, yy over 64 px x RGB for the six candidates (0..1 units), the same restricted to edge-mask pixels, mean / std / min / max of the 34 features (136), block-mean rule weights. Error of a block-constant mix w = w^T G w - 2 b^T w + yy (exact). Checks: block oracle from the table equals `block_oracle/NN.json` "6_8" on pairs 56 and 70 (relative difference <= 3e-7 in se and se_e, same edge counts); rule per pixel over the 23 held-out pairs 31.243 / 26.753; block oracle 34.18 / 29.37. Nets: hidden width 64, leaky ReLU 0.1, softmax over logits log(max(block-mean rule, 1e-3)) (taps) and -5 (prev, curr) plus the net output, last layer zero at the start; MLPs are 1x1 convs (2 hidden layers); b-ctx adds the feature means over the 3x3 / 7x7 / 15x15 blocks around (edge-replicated), b-ctx-frame also the frame-wide mean and std of the 34 features, b-conv 3 x (3x3) (receptive field 7x7), b-conv-big 4 x (3x3, dilation 1,2,4,8) (31x31) + frame-wide statistics channels. Inputs standardised with training statistics. Loss: exact mean squared error per pixel from the Gram form (mse); Adam 2e-3 -> 1e-4 cosine, 2,500 steps, 8 random 64x64 block crops (of the 56 training pairs) per step, 3 threads; 1 to 12 minutes per net.

Result (PSNR dB whole / edge; rule held-out 31.24 / 26.75; "train gain" = gain over the per-pixel rule on the 56 training pairs, whole / edge):

| method | held-out all | held-out gain | train gain |
|---|---|---|---|
| (a) rule block-averaged | 31.34 / 26.85 | +0.10 / +0.10 | +0.06 / +0.06 |
| (b) fixed mix, all | 31.36 / 26.83 | +0.12 / +0.07 | +0.34 / +0.28 |
| (c) fixed mix per group | 31.39 / 26.93 | +0.15 / +0.18 | +0.54 / +0.57 |
| (d) block oracle | 34.18 / 29.37 | +2.93 / +2.61 | +4.52 / +3.73 |
| b-mean | 30.94 / 26.43 | -0.30 / -0.32 | +0.64 / +0.49 |
| b-stats | 31.02 / 26.58 | -0.22 / -0.17 | +0.75 / +0.56 |
| b-ctx | 30.46 / 25.91 | -0.78 / -0.85 | +0.81 / +0.58 |
| b-ctx-frame | 30.94 / 26.61 | -0.31 / -0.14 | +1.61 / +1.31 |
| b-conv | 30.25 / 25.88 | -1.00 / -0.87 | +1.02 / +0.71 |
| b-conv-big | 30.17 / 26.19 | -1.07 / -0.57 | +1.37 / +1.17 |
| b-stats, wd 1e-4 + dropout 0.1 | 31.09 / 26.65 | -0.15 / -0.10 | +0.75 / +0.56 |
| b-ctx, wd + dropout | 30.52 / 25.99 | -0.72 / -0.77 | +0.82 / +0.61 |
| b-conv, wd + dropout | 30.68 / 26.17 | -0.57 / -0.59 | +0.90 / +0.65 |
| b-stats, Charbonnier loss | 31.15 / 26.63 | -0.10 / -0.12 | +0.54 / +0.35 |
| b-stats, seed 1 | 31.22 / 26.75 | -0.02 / -0.01 | +0.78 / +0.57 |

Answer: no block net beats the rule on held-out pairs (best -0.02 dB, b-stats seed 1; seed 0 gives -0.22, so the seed noise of this net is about 0.2 dB, far larger than the 0.01 dB of the per-pixel nets). The best net reaches about 0 % of the oracle's +2.93 dB (range -37 % to -1 %). The nets do fit the training pairs (+0.6 to +1.6 dB, more with more context or capacity; the oracle on the training pairs is +4.5 dB) and that does not carry over: held-out forza loses 0.3 to 1.8 dB (held-out forza is another clip, racing_forza6_c, than the training clips _a/_b), minecraft120 gains 0.5 to 0.6 for the 1x1 and context nets, cs2 up to +0.7 for some nets, battlefield about 0. Wider context raises the training gain and lowers the held-out gain (b-stats, b-ctx, b-conv, b-conv-big: held-out -0.22, -0.78, -1.00, -1.07). Frame-wide statistics give the largest training gain (+1.61) and a better held-out figure than the same net without them (b-ctx -0.78, b-ctx-frame -0.31), but still below the rule. Weight decay 1e-4 + dropout 0.1 helps the over-fitting nets a little (b-conv -1.00 -> -0.57, b-ctx -0.78 -> -0.72, b-stats -0.22 -> -0.15), none turns positive. The Charbonnier loss does not change the picture (b-stats -0.10 vs -0.22, inside seed noise). Fixed mixes: the single six-weight mix fitted on the training pairs (0.176 0.217 0.225 0.282 0.033 0.068) gains +0.12 / +0.07 held-out, one per group +0.15 / +0.18, with per-group swings from -0.11 (minecraft120, (c)) to +0.52 (minecraft120, (b)); (a) block constancy alone costs the rule nothing (+0.10). Not done: block size 16, and a second seed of the other nets (only b-stats has seed 1). Training loss values are noisy (random crops); last-100-step means are in `tables3.md`.
