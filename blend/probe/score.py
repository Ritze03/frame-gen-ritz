"""Scores every method on the held-out pairs -> $BLEND/probe/scores.json (per-pair accumulators) and prints the tables.
usage: score.py [--nets px8,px32,...]
Methods: rule, baseC, forzaC (recorded / stored weights), the trained nets (nets/NAME.pt), O4, O6.  All at the weight stage (Sum weights * points, no clamp,
smear or static mask).  Edge mask: luma of the REAL frame, Sobel gradient magnitude G (edge-replicated border), mask = max of G over the 5x5 window >= the 90th
percentile of G of that frame (np.percentile, linear)."""
import json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nets import *

BLEND = os.environ.get("BLEND", "/home/mo/DieGanzeSau/AI_Video_Training_Data/blend")
ROOT = f"{BLEND}/probe"
torch.set_num_threads(4)


def edge_mask(y):
    """y [H,W,3] u8 -> bool [H,W]."""
    yf = y.astype(np.float32)
    L = torch.from_numpy(0.299 * yf[..., 0] + 0.587 * yf[..., 1] + 0.114 * yf[..., 2])[None, None]
    Lp = F.pad(L, (1, 1, 1, 1), mode="replicate")
    kx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None]
    G = torch.sqrt(F.conv2d(Lp, kx) ** 2 + F.conv2d(Lp, kx.transpose(2, 3)) ** 2)
    thr = np.percentile(G.numpy().ravel(), 90)
    return (F.max_pool2d(G, 5, 1, 2)[0, 0] >= thr).numpy()


def accumulate(c, y, em):
    d = c.astype(np.float64) - y
    se = (d ** 2).sum(-1)
    big = (np.abs(d) > 30).any(-1)
    return dict(se=float(se.sum()), n=int(se.size), se_e=float(se[em].sum()), n_e=int(em.sum()), big=int(big.sum()), big_e=int(big[em].sum()))


def colour_w6(w6, taps, e):
    pts = np.concatenate([taps.reshape(H, W, 4, 3), e.reshape(H, W, 2, 3)], 2).astype(np.float32)
    return np.einsum("hwk,hwkc->hwc", w6.astype(np.float32), pts)


def main():
    names = sys.argv[sys.argv.index("--nets") + 1].split(",") if "--nets" in sys.argv else [n for n in NAMES if os.path.exists(f"{ROOT}/nets/{n}.pt")]
    pairs = json.load(open(f"{ROOT}/pairs.json"))
    nets = {}
    for n in names:
        nets[n] = Net(n)
        nets[n].load_state_dict(torch.load(f"{ROOT}/nets/{n}.pt"))
        nets[n].eval()
    res = {}
    for k, p in enumerate(pairs):
        if p["split"] != "val":
            continue
        t0 = time.time()
        d = f"{ROOT}/pairs/{k:02d}"
        ld = lambda f: np.load(f"{d}/{f}.npy", allow_pickle=False)
        y, taps, e = ld("y"), ld("taps"), ld("e")
        em = edge_mask(y)
        yf = y.astype(np.float32)
        r = {}
        r["rule"] = accumulate(np.einsum("hwk,hwkc->hwc", ld("rule").astype(np.float32), taps.reshape(H, W, 4, 3).astype(np.float32)), yf, em)
        r["baseC"] = accumulate(colour_w6(ld("baseC"), taps, e), yf, em)
        r["forzaC"] = accumulate(colour_w6(ld("forzaC"), taps, e), yf, em)
        pr = Pair(ROOT, k)
        for n, net in nets.items():
            r[n] = accumulate(predict(net, pr), yf, em)
        for o in ("o4", "o6"):
            if os.path.exists(f"{d}/{o}.npy"):
                r[o.upper()] = accumulate(ld(o), yf, em)
        r["_edge_frac"] = float(em.mean())
        res[k] = dict(pair=p, scores=r)
        print(k, p["id"], "%.0fs" % (time.time() - t0), flush=True)
    json.dump(res, open(f"{ROOT}/scores.json", "w"), indent=1)


def psnr(se, n):
    return 10 * np.log10(255.0 ** 2 / (se / (3 * n)))


def tables(order=None):
    res = json.load(open(f"{ROOT}/scores.json"))
    meths = [m for m in next(iter(res.values()))["scores"] if not m.startswith("_")]
    if order:
        meths = [m for m in order if m in meths]
    groups = ["forza", "cs2", "minecraft120", "battlefield", "all"]
    out = {}
    for m in meths:
        for g in groups:
            ks = [v["scores"][m] for v in res.values() if (g == "all" or v["pair"]["group"] == g) and m in v["scores"]]
            a = {f: sum(x[f] for x in ks) for f in ks[0]}
            out[(m, g)] = dict(whole=psnr(a["se"], a["n"]), edge=psnr(a["se_e"], a["n_e"]), big=100 * a["big"] / a["n"], big_e=100 * a["big_e"] / a["n_e"])
    return meths, groups, out


if __name__ == "__main__":
    if "--tables" not in sys.argv:
        main()
    meths, groups, out = tables(["rule", "baseC", "forzaC"] + NAMES + ["O4", "O6"])
    for kind, label in (("whole", "PSNR dB, whole picture"), ("edge", "PSNR dB, edge pixels only"), ("big", "% pixels with error > 30 in any channel (whole)")):
        print("\n" + label)
        print("| method | " + " | ".join(groups) + " |")
        print("|---|" + "---|" * len(groups))
        for m in meths:
            print(f"| {m} | " + " | ".join("%.2f" % out[(m, g)][kind] for g in groups) + " |")
