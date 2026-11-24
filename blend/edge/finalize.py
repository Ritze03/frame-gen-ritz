"""Copies the chosen nets to blend/weights/ar4, checks them against the forza-rx9070xt files, writes run.json."""
import json, os, shutil, subprocess
import numpy as np
from common import WORK, WDIR, HERE

LAM = "1.0"
out = f"{WDIR}/ar4"
os.makedirs(out, exist_ok=True)
for n in "ABC":
    src = f"{WORK}/nets/{n}_lam{LAM}.npz"
    shutil.copy(src, f"{out}/{n}.npz")
    a = np.load(f"{out}/{n}.npz", allow_pickle=False)
    b = np.load(f"{WDIR}/forza-rx9070xt/{n}.npz", allow_pickle=False)
    assert sorted(a.files) == sorted(b.files)
    for k in b.files:
        assert a[k].shape == b[k].shape and a[k].dtype == b[k].dtype and np.isfinite(a[k]).all(), (n, k)
    print(n, "ok")
commit = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
json.dump({
    "commit": commit, "task": 146, "start_weights": "blend/weights/forza-rx9070xt (A, B, C)",
    "data": "dense dumps of the 56 training pairs of blend/probe/pairs.json (32 forza 30->60, 8 cs2, 8 minecraft120, 8 battlefield 60->120); scored on the 23 held-out pairs",
    "steps": 2000, "lr": 2e-4, "lr_end": 1e-5, "lr_schedule": "cosine", "optimizer": "Adam", "seed": 0, "batch": 16, "crop": 128,
    "loss": "Charbonnier mean(sqrt((out - real)^2 + 1e-6)) + lam * mean(relu(G(out) - G(real))); colours 0..1, G = Sobel magnitude of luma "
            "(0.299 R + 0.587 G + 0.114 B) on the crop interior, sqrt(gx^2 + gy^2 + 1e-8) while training; output = sum of weights x candidates",
    "lam": float(LAM), "lam_start_share": "about 46 % of the start loss (start terms: Charbonnier 0.014705, penalty 0.012512)",
    "sweep_lam_C": [0, 0.1306, 0.5037, 1.0, 1.763], "choice_rule": "strongest lam with all-pairs whole PSNR at most 0.10 dB and every group at most 0.25 dB below forza-rx9070xt C",
    "scripts": "blend/edge/ (train_edge.py, score_edge.py, report_edge.py, finalize.py)",
}, open(f"{out}/run.json", "w"), indent=1)
