"""Scores methods on the 23 held-out pairs -> $BLEND/edge/scores/NAME.json (per-pair sums).
usage: score_edge.py NAME=SPEC ...   SPEC: rule | file:baseC | file:forzaC (stored six weights) | path/to/net.npz (forward pass of common.py)
Measures: whole / edge-pixel PSNR (edge mask as blend/probe/score.py: 90th percentile of the real frame's Sobel G, max over 5x5), spurious = mean max(0, G_out - G_real),
missing = mean max(0, G_real - G_out) (G = Sobel magnitude of luma, replicate border, levels; as /home/mo/tmp/fg-145/an.py), whole and inside the mask,
mean weight on prev + curr (w4 + w5) at the edge pixels.  Output colour = sum of weights x candidates (weight stage)."""
import json, os, sys, time
import numpy as np, torch
import torch.nn.functional as F
from common import *

torch.set_num_threads(3)
os.makedirs(f"{WORK}/scores", exist_ok=True)
pairs = json.load(open(f"{PROBE}/pairs.json"))
val = [k for k, p in enumerate(pairs) if p["split"] == "val"]
assert len(val) == 23


def lum(c):
    return 0.299 * c[..., 0] + 0.587 * c[..., 1] + 0.114 * c[..., 2]


def sob(L):
    return sobel(torch.from_numpy(np.ascontiguousarray(L))[None], pad=True)[0]


def run(specs):
    res = {n: {} for n in specs}
    for k in val:
        t0 = time.time()
        fr = Frame(k)
        y = np.asarray(fr.ld("y")).astype(np.float32)
        Gr = sob(lum(y))
        thr = np.percentile(Gr.numpy().ravel(), 90)
        em = (F.max_pool2d(Gr[None, None], 5, 1, 2)[0, 0] >= thr).numpy()
        taps = np.asarray(fr.ld("taps")).astype(np.float32).reshape(H, W, 4, 3)
        e = np.asarray(fr.ld("e")).astype(np.float32).reshape(H, W, 2, 3)
        pts = np.concatenate([taps, e], 2)
        for n, spec in specs.items():
            if spec == "rule":
                w = np.concatenate([np.asarray(fr.ld("rule")).astype(np.float32), np.zeros((H, W, 2), np.float32)], 2)
                c = np.einsum("hwk,hwkc->hwc", w, pts)
            elif spec.startswith("file:"):
                w = np.asarray(fr.ld(spec[5:])).astype(np.float32)
                c = np.einsum("hwk,hwkc->hwc", w, pts)
            else:
                w, c = predict_w(load_npz(spec), fr)
                if w.shape[2] == 4:
                    w = np.concatenate([w, np.zeros((H, W, 2), np.float32)], 2)
            d = c.astype(np.float64) - y
            se = (d ** 2).sum(-1)
            Go = sob(lum(c))
            sp = torch.clamp(Go - Gr, min=0).numpy()
            ms = torch.clamp(Gr - Go, min=0).numpy()
            un = w[..., 4] + w[..., 5]
            res[n][k] = dict(group=pairs[k]["group"], se=float(se.sum()), n=int(se.size), se_e=float(se[em].sum()), n_e=int(em.sum()),
                             sp=float(sp.sum()), ms=float(ms.sum()), sp_e=float(sp[em].sum()), ms_e=float(ms[em].sum()), un_e=float(un[em].sum()))
        print(k, "%.0fs" % (time.time() - t0), flush=True)
    for n in specs:
        json.dump(res[n], open(f"{WORK}/scores/{n}.json", "w"))


if __name__ == "__main__":
    run(dict(a.split("=", 1) for a in sys.argv[1:]))
