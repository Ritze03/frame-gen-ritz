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
