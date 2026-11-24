"""Continues a library net (npz) with Charbonnier + lam * mean(relu(G(out) - G(real))).
usage: train_edge.py OUT.npz --init START.npz [--lam L] [--steps N] [--startterms]
128x128 crops, batch 16, colours 0..1, G = Sobel magnitude of luma on the crop interior (126x126), Adam 2e-4 -> 1e-5 cosine, seed 0, the same crops in the same order for every run.
--startterms: only prints the two loss terms at the start (mean over the first 100 batches)."""
import json, math, sys, time, resource
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch
from common import *
from nets import Pair

a = sys.argv
out = a[1]
init = a[a.index("--init") + 1]
lam = float(a[a.index("--lam") + 1]) if "--lam" in a else 0.0
steps = int(a[a.index("--steps") + 1]) if "--steps" in a else 2000
LR0, LR1 = 2e-4, 1e-5
torch.set_num_threads(3)
torch.manual_seed(0)
pairs = json.load(open(f"{PROBE}/pairs.json"))
tr = [k for k, p in enumerate(pairs) if p["split"] == "train"]
assert len(tr) == 56
P = {k: Pair(PROBE, k) for k in tr}
B, C = 16, 128
rng = np.random.default_rng(0)
plan = [(tr[rng.integers(len(tr))], int(rng.integers(0, H - C + 1)), int(rng.integers(0, W - C + 1))) for _ in range(2000 * B)]
ex = ThreadPoolExecutor(3)


def batch(s):
    items = [P[k].window(y, y + C, x, x + C) for k, y, x in plan[s * B:(s + 1) * B]]
    return tuple(torch.from_numpy(np.stack([it[j] for it in items])) for j in range(3))


futs = {}


def get(s):
    for i in range(s, min(s + 8, steps)):
        if i not in futs:
            futs[i] = ex.submit(batch, i)
    return futs.pop(s).result()


def terms(p, X, R, Y):
    c = colour(weights(p, X, R), X)
    ch = torch.sqrt((c - Y) ** 2 + 1e-6).mean()
    pen = torch.relu(sobel(luma(c), eps=1e-8) - sobel(luma(Y), eps=1e-8)).mean()
    return ch, pen


p = [t.clone().requires_grad_(True) for t in load_npz(init)]
if "--startterms" in a:
    steps = 100
    cs, ps = [], []
    with torch.no_grad():
        for s in range(100):
            ch, pen = terms(p, *get(s))
            cs.append(ch.item()); ps.append(pen.item())
    print(json.dumps(dict(charb=float(np.mean(cs)), pen=float(np.mean(ps)))))
    sys.exit()
opt = torch.optim.Adam(p, lr=LR0)
print(out, "init", init, "lam", lam, "steps", steps, flush=True)
t0 = time.time()
rc = rp = 0.0
for s in range(steps):
    lr = LR1 + 0.5 * (LR0 - LR1) * (1 + math.cos(math.pi * s / steps))
    for g in opt.param_groups:
        g["lr"] = lr
    ch, pen = terms(p, *get(s))
    loss = ch + lam * pen
    opt.zero_grad(); loss.backward(); opt.step()
    rc = 0.98 * rc + 0.02 * ch.item() if s else ch.item()
    rp = 0.98 * rp + 0.02 * pen.item() if s else pen.item()
    if s % 250 == 0 or s == steps - 1:
        print(s, "charb %.5f pen %.5f" % (rc, rp), "%.0fs" % (time.time() - t0), flush=True)
np.savez(out, **{k: t.detach().numpy().astype(np.float32) for k, t in zip(("W1", "b1", "W2", "b2"), p)})
print("done %.0fs peak rss MB %d" % (time.time() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024), flush=True)
