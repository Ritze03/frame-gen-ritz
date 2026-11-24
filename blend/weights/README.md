The library holds two nets, both generated into `gpu/shaders/v3_blend_weights.glsl` (`V3B_NETS` = 2, `Settings::v3Blend` 1..2; 0 = the rule). Both are the six-choice shape ("net C": 34 inputs, 6 outputs):

| `v3Blend` | name | weights | character |
|---|---|---|---|
| 0 | the rule | none | the hand-written weights |
| 1 | Fluidity (`kV3BlendFluidity`) | `blend/weights/forza-rx9070xt/C.npz` | smoother in-between motion |
| 2 | Consistency (`kV3BlendConsistency`) | `blend/weights/ar4/C.npz` | fewest ghost edges and the most stable picture, softer edges, motion looks closer to repeated frames |

The values changed with task 153: the old 6 is now 1, the old 15 is now 2 (the earlier build had 1..15 = five sets of three nets).

Measured on 23 held-out pairs (whole / edge PSNR in dB, spurious edge energy, cost per real pair at 1080p 2x on an RX 7900 XTX):

| | whole | edge | spurious edge energy | cost |
|---|---|---|---|---|
| rule | 31.24 | 26.75 | 2.880 | 1.000 ms |
| Fluidity | 31.33 | 26.69 | 2.353 | 1.070 ms |
| Consistency | 31.24 | 26.61 | 2.242 | 1.071 ms |

The other weight sets stay in this folder as training history and are not in the build: `base` (trained on the owner's footage with real and teacher targets), `forza-rx9070xt` (its nets A and B too: `base` fine-tuned by a contributor on 50 minutes of their own 120 fps racing footage), `ar` (real targets only, from scratch), `ar2`, `ar3` (`forza-rx9070xt` continued on all of the owner's real rows) and `ar4` (`forza-rx9070xt` continued with a penalty on spurious edges: `blend/edge/README.md`). The four-choice nets A and B (34 -> 25 inputs, 4 outputs) are no longer in the build. Every folder's `scores.md` holds its held-out table.

Regenerate the header (from the repository root; the order of the paths is the order of the values, one six-choice file per value, and `kV3BlendMax` in `gpu/framegen.h` must equal the new `V3B_NETS`):

    python3 gpu/ml/blend/export_glsl.py blend/weights/forza-rx9070xt/C.npz blend/weights/ar4/C.npz -o gpu/shaders/v3_blend_weights.glsl

Fine-tuned weights from other people go to `blend/weights/<name>/` (A.npz B.npz C.npz, scores.md, run.json); see `blend/FINETUNE.md`.
