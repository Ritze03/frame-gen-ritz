"""Step 1: the torch forward pass reproduces the stored baseC / forzaC weights of a held-out pair (200,000+ pixels, 2e-3)."""
import sys
import numpy as np, torch
from common import *
k = 60
fr = Frame(k)
rows = slice(400, 600)   # 200 rows x 1920 = 384,000 pixels
X = torch.from_numpy(np.asarray(fr.ld("X")[rows]).astype(np.float32)).permute(2, 0, 1)[None]
R = torch.from_numpy(np.asarray(fr.ld("rule")[rows]).astype(np.float32)).permute(2, 0, 1)[None]
for name, f in (("forza-rx9070xt", "forzaC"), ("base", "baseC")):
    w = weights(load_npz(f"{WDIR}/{name}/C.npz"), X, R)[0].permute(1, 2, 0).numpy()
    ref = np.asarray(fr.ld(f)[rows]).astype(np.float32)
    print(name, "pixels", w.shape[0] * w.shape[1], "worst |dw| %.2e" % np.abs(w - ref).max(), "mean %.2e" % np.abs(w - ref).mean())
