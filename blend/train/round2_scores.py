"""Round-2 comparison (task 140): PSNR of the rule and of six weight sets on ALL real_val rows, per group and per manifest case, nets A B C;
plus the mean |w - w_start| of the output mix weights on 100000 validation rows.
usage: round2_scores.py OUT.md SET=DIR ...   (SET 'rule' is implicit).  Metric code = finetune_report.colours + diag.Acc (same as scores.md)."""
import sys
import json
import numpy as np
import torch
from feat import *
import train as TR
import finetune_report as FR
import diag

torch.set_num_threads(5)
out, sets = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
va = {k: np.load(f"{CACHE}/real_val_{k}.npy") for k in ("X", "R", "Y", "G", "S")}
man = json.load(open(f"{BLEND}/frames/manifest.json"))["samples"]
case_of = np.array([x["case"] for x in man])[va["S"]]
nets = {n: TR.load_net(d) for n, d in sets}
names = ["rule"] + [n for n, _ in sets]
sel = [("all rows", np.ones(len(va["G"]), bool))]
for g in range(len(GROUPS)):
    m = va["G"] == g
    for c in sorted(set(case_of[m])):
        sel.append((f"{GROUPS[g]} {c}", m & (case_of == c)))
lines = ["# Round 2 scores: PSNR (dB) on all real_val rows, against the real target", "",
         "| rows | net | n | " + " | ".join(names) + " |", "|---|---|---|" + "---|" * len(names)]
for label, m in sel:
    ix = np.nonzero(m)[0]
    y = va["Y"][ix].astype(np.float64) * 255
    taps = va["X"][ix, :12].reshape(-1, 4, 3).astype(np.float64) * 255
    rule = np.einsum("nk,nkc->nc", va["R"][ix].astype(np.float64), taps)
    for v in TR.VARS:
        cols = {"rule": rule}
        for n in nets:
            cols[n] = FR.colours(TR, nets[n], v, va, ix)
        a = diag.Acc()
        a.add(cols, y, np.ones(len(ix), bool))
        f = a.fig()
        lines.append(f"| {label} | {v} | {len(ix)} | " + " | ".join(f"{f[k]['psnr']:.3f}" for k in names) + " |")
    print(label, flush=True)
# weight distances
rng = np.random.default_rng(0)
ix = np.sort(rng.choice(len(va["G"]), 100000, replace=False))
Xv, Rv = torch.from_numpy(va["X"][ix]), torch.from_numpy(va["R"][ix])


def W(n, v):
    NI, K, _ = TR.VARS[v]
    with torch.no_grad():
        return TR.forward(nets[n][v], Xv, Rv, K, NI)[1]


lines += ["", "## Mean |w - w_ref| of the output mix weights (100000 random real_val rows; mean over rows and choices)", "",
          "| set | net | vs base | vs forza-rx9070xt | vs rule weights (R) |", "|---|---|---|---|---|"]
for n, _ in sets:
    for v in TR.VARS:
        w = W(n, v)
        K = w.shape[1]
        r = torch.cat([Rv, torch.zeros(len(Rv), K - 4)], 1)
        d = lambda o: f"{(w - W(o, v)).abs().mean().item():.5f}" if o in nets else "n/a"
        lines.append(f"| {n} | {v} | {d('base')} | {d('forza')} | {(w - r).abs().mean().item():.5f} |")
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
