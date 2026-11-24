"""Per-pixel rule error for all 79 pairs (pixel files, once) -> $BLEND/probe/block3/rule_pix.json {k: {se, n, se_e, n_e}} in levels^2;
prints the held-out pooled PSNR (task 147 check: 31.24 dB whole)."""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import edge_mask
BLEND = os.environ.get("BLEND", "/home/mo/DieGanzeSau/AI_Video_Training_Data/blend")
ROOT = f"{BLEND}/probe"
pairs = json.load(open(f"{ROOT}/pairs.json"))
out = {}
for k in range(79):
    d = f"{ROOT}/pairs/{k:02d}"
    ld = lambda f: np.load(f"{d}/{f}.npy", allow_pickle=False, mmap_mode="r")
    rule, taps, y = ld("rule"), ld("taps"), ld("y")
    em = edge_mask(np.asarray(y))
    se = np.zeros((1080, 1920))
    for r0 in range(0, 1080, 135):
        c = np.einsum("hwk,hwkc->hwc", np.asarray(rule[r0:r0 + 135]).astype(np.float32), np.asarray(taps[r0:r0 + 135]).reshape(-1, 1920, 4, 3).astype(np.float32))
        se[r0:r0 + 135] = ((c - np.asarray(y[r0:r0 + 135]).astype(np.float32)) ** 2).sum(-1)
    out[k] = dict(se=float(se.sum()), n=int(se.size), se_e=float(se[em].sum()), n_e=int(em.sum()))
    print(k, flush=True)
json.dump(out, open(f"{ROOT}/block3/rule_pix.json", "w"))
v = [out[k] for k in range(79) if pairs[k]["split"] == "val"]
print("rule per pixel, held-out pooled: %.3f dB whole, %.3f edge" % (10 * np.log10(255.0 ** 2 / (sum(a["se"] for a in v) / (3 * sum(a["n"] for a in v)))), 10 * np.log10(255.0 ** 2 / (sum(a["se_e"] for a in v) / (3 * sum(a["n_e"] for a in v))))))
