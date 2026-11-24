"""Step 1 checks (task 147): (1) block-oracle error from the block table (gsolve) vs block_oracle/NN.json '6_8' on pairs 56 and 70;
(2) rule PSNR over the 23 held-out pairs computed per pixel from the pixel files (mse pooled)."""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block3_lib import *
from block_oracle import gsolve

for k in (56, 70):
    z = np.load(f"{T3}/{k:02d}.npz", allow_pickle=False)
    G, b, yy = (z[n].astype(np.float64).reshape(-1, *z[n].shape[2:]) for n in ("G", "b", "yy"))
    w, err = gsolve(G, b, yy, 6)
    w = w.reshape(135, 240, 6)
    d = {n: torch.from_numpy(z[n]) for n in z.files}
    e, ee = errs(torch.from_numpy(w), d, True)
    j = json.load(open(f"{ROOT}/block_oracle/{k}.json"))["6_8"]
    print(k, "table: se %.1f se_e %.1f n_e %d | json: se %.1f se_e %.1f n_e %d | rel diff %.2e %.2e" % (
        e.sum(), ee.sum(), z["ne"].sum(), j["se"], j["se_e"], j["n_e"], e.sum() / j["se"] - 1, ee.sum() / j["se_e"] - 1))
se = n = 0
for k in VAL:
    d = f"{ROOT}/pairs/{k:02d}"
    ld = lambda f: np.load(f"{d}/{f}.npy", allow_pickle=False, mmap_mode="r")
    rule, taps, y = ld("rule"), ld("taps"), ld("y")
    for r0 in range(0, 1080, 135):
        c = np.einsum("hwk,hwkc->hwc", np.asarray(rule[r0:r0 + 135]).astype(np.float32), np.asarray(taps[r0:r0 + 135]).reshape(-1, 1920, 4, 3).astype(np.float32))
        se += float(((c - np.asarray(y[r0:r0 + 135]).astype(np.float32)) ** 2).sum()); n += c.size
print("rule per pixel, 23 held-out pairs: %.3f dB" % (10 * np.log10(255.0 ** 2 / (se / n))))
