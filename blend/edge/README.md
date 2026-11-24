# blend/edge: continue the forza-rx9070xt nets with an edge penalty (task 146)

Offline, Python only. Goal: push in the direction the forza-rx9070xt net C went (less spurious edge energy / double contours, a little softer) on purpose. Result: `blend/weights/ar4/{A,B,C}.npz` (scores in `blend/weights/ar4/scores.md`).

## Method
- `common.py`: the library nets' forward pass in torch (features X from `blend/probe` dumps, logits = log(max(rule,1e-3)) [+ -5, -5 for six choices] + net, softmax); checked by `check_forward.py` on 384,000 pixels of held-out pair 60: worst |dw| 8.6e-4 (forza-rx9070xt C vs `forzaC.npy`), 1.1e-3 (base C vs `baseC.npy`).
- `train_edge.py`: loss = Charbonnier(out, real) + lam * mean(relu(G(out) - G(real))); G = Sobel magnitude of luma on the interior of 128x128 crops, colours 0..1 (sqrt eps 1e-8 inside G while training: without it the gradient is NaN on flat areas). Batch 16, 2000 steps, Adam 2e-4 -> 1e-5 cosine, seed 0, same crops in the same order for every run, 56 training pairs of `blend/probe`. Start terms of net C (100 batches): Charbonnier 0.014705, penalty 0.012512, so lam 0.1306 / 0.5037 / 1.763 = 10 / 30 / 60 % penalty share; 1.0 (about 46 %) was added.
- `score_edge.py`, `report_edge.py`: the 23 held-out pairs; measures as in the brief (spurious / missing = mean max(0, +-(G_out - G_real)), whole picture and inside the edge mask of `blend/probe/score.py`, mean w4 + w5 at edge pixels). Control: rule / base C / forza C reproduce 2.880 / 2.463 / 2.355 (spurious) and 5.466 / 5.805 / 6.000 (missing).

## Commands (`BLEND` = data root; outputs in `$BLEND/edge/`)
`sh preread.sh` (page cache), `sh sweep_c.sh` (C: lam 0, 0.1306, 0.5037, 1.763, then reference scores), `sh score_c.sh`, `sh sweep_c2.sh` (lam 1.0), `sh ab.sh 1.0` (nets A, B and their scores), `sh py.sh finalize.py` (writes `blend/weights/ar4`), `sh py.sh report_edge.py NAME[=label] ...`.

## Result (23 held-out pairs, all groups pooled; whole PSNR dB / edge PSNR / spurious whole / missing whole / spurious in mask / w(prev+curr) at edges)
| net C | PSNR | edge PSNR | spur. | miss. | spur. mask | w prev+curr |
|---|---|---|---|---|---|---|
| rule | 31.24 | 26.75 | 2.880 | 5.466 | 5.701 | 0 |
| base C | 31.33 | 26.72 | 2.463 | 5.805 | 5.073 | 0.198 |
| forza-rx9070xt C | 31.33 | 26.69 | 2.355 | 6.000 | 4.781 | 0.223 |
| lam 0 (control) | 31.23 | 26.67 | 2.490 | 5.844 | 5.073 | 0.187 |
| lam 0.1306 | 31.26 | 26.69 | 2.422 | 5.931 | 4.926 | 0.212 |
| lam 0.5037 | 31.27 | 26.67 | 2.306 | 6.138 | 4.653 | 0.261 |
| **lam 1.0 (ar4 C)** | 31.24 | 26.61 | 2.242 | 6.315 | 4.489 | 0.293 |
| lam 1.763 | 31.18 | 26.53 | 2.202 | 6.474 | 4.384 | 0.319 |

Chosen lam = 1.0 (all-pairs whole PSNR -0.088 dB vs forza C, worst group forza -0.150 dB; lam 1.763 gives -0.147 dB and fails). Against forza-rx9070xt C: spurious whole -4.8 %, inside the mask -6.1 %, missing whole +5.3 %. The control (lam 0) drifts back toward base C (the dense recipe alone loses part of what forza C had), so the effect of the penalty is larger against lam 0 (spurious -10 %).
Nets A and B (four taps, no unwarped choice) barely respond: with lam 1.0 spurious whole 2.724 -> 2.714 (A), 2.729 -> 2.719 (B), PSNR +0.02 dB.
