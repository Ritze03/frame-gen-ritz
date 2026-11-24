The library holds five weight sets, all generated into `gpu/shaders/v3_blend_weights.glsl` (15 nets, `Settings::v3Blend` 1..15, value = 1 + 3 * set + net, net 0, 1, 2 = A, B, C):

| set | values | directory | host name | what it is |
|---|---|---|---|---|
| 0 | 1, 2, 3 | `blend/weights/base` | A, B, C | trained on the owner's footage with real and teacher targets, 15,000 steps |
| 1 | 4, 5, 6 | `blend/weights/forza-rx9070xt` | A-A, A-B, A-C | set 0 fine-tuned by a contributor on 50 minutes of their own 120 fps racing footage (real 60->120 targets only), 4,000 steps |
| 2 | 7, 8, 9 | `blend/weights/ar` | AR-A, AR-B, AR-C | trained on the owner's footage with real targets only, from scratch, 15,000 steps |
| 3 | 10, 11, 12 | `blend/weights/ar3` | AR3-A, AR3-B, AR3-C | set 1 continued for 4,000 steps on all of the owner's real rows |
| 4 | 13, 14, 15 | `blend/weights/ar4` | AR4-A, AR4-B, AR4-C | set 1 continued for 2,000 steps on 56 full-resolution frame pairs of the owner's footage with Charbonnier plus `lam * mean(relu(G(out) - G(real)))` (G = Sobel magnitude of luma, lam 1.0, a penalty on spurious edges); net C has about 5 % less spurious edge energy than set 1's C at 0.09 dB lower PSNR, nets A and B barely differ from set 1's (`blend/edge/README.md`) |

Sets 2 to 4 are an experiment for comparison by eye. The held-out scores of sets 2 and 3 are within 0.02 dB of set 0 (all rows, net C: 28.193 base, 28.246 set 1, 28.202 / 28.208 for sets 2 / 3; set 4's net C is 0.09 dB below set 1's by design, see `blend/weights/ar4/scores.md`; the table is in each set's `scores.md`). `blend/weights/ar2` (set 0 fine-tuned with the set-1 recipe on the owner's real 60->120 rows) is no longer in the header: it came out nearly identical to set 0; its directory stays in the repository as a record.

Regenerate the header (from the repository root; the order of the paths is the order of the values, three files per set, and `kV3BlendMax` in `gpu/framegen.h` must equal the new `V3B_NETS`):

    python3 gpu/ml/blend/export_glsl.py blend/weights/base/{A,B,C}.npz blend/weights/forza-rx9070xt/{A,B,C}.npz blend/weights/ar/{A,B,C}.npz blend/weights/ar3/{A,B,C}.npz blend/weights/ar4/{A,B,C}.npz -o gpu/shaders/v3_blend_weights.glsl

Fine-tuned weights from other people go to `blend/weights/<name>/` (A.npz B.npz C.npz, scores.md, run.json); see `blend/FINETUNE.md`.
