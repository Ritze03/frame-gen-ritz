The library holds two weight sets, both generated into `gpu/shaders/v3_blend_weights.glsl`: `base/` = `Settings::v3Blend` 1, 2, 3 (nets A, B, C) and `forza-rx9070xt/` = `v3Blend` 4, 5, 6 (nets A, B, C of a fine-tune of the base nets on other Forza footage). Regenerate the header (from the repository root):

    python3 gpu/ml/blend/export_glsl.py blend/weights/base/{A,B,C}.npz blend/weights/forza-rx9070xt/{A,B,C}.npz

(the first three paths = `v3Blend` 1..3, the next three = 4..6.)
Fine-tuned weights from other people go to `blend/weights/<name>/` (A.npz B.npz C.npz, scores.md, run.json); see `blend/FINETUNE.md`.
