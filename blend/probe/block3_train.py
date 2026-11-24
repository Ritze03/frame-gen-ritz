"""Train and score one block net (task 147).  Loss = exact mean per-pixel squared error from the Gram form (or --loss char: sqrt(block mse + 1) levels).
Batches of random SxS crops of the block grids (with the net's receptive-field margin and exact context), Adam 2e-3 -> 1e-4 cosine.
usage: block3_train.py NAME [--steps N] [--seed S] [--wd X] [--drop X] [--loss mse|char] [--bs 1|2] [--batch N] [--crop S] [--tag T] [--threads N]
Writes $BLEND/probe/block3/res_TAG.json (train / val PSNR per group, loss curve, timing) and net_TAG.pt."""
import json, sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block3_lib import *

a = sys.argv
name = a[1]
opt = lambda n, d, t=str: t(a[a.index(n) + 1]) if n in a else d
steps, seed, wd, drop, loss_k = opt("--steps", 2000, int), opt("--seed", 0, int), opt("--wd", 0.0, float), opt("--drop", 0.0, float), opt("--loss", "mse")
bs, batch, crop = opt("--bs", 1, int), opt("--batch", 8, int), opt("--crop", 64, int)
tag = opt("--tag", name + ("" if bs == 1 else "-b16"))
torch.set_num_threads(opt("--threads", 3, int))
torch.manual_seed(seed)
rng = np.random.default_rng(seed)
data = {k: load_pair(k, bs) for k in range(79)}
prep = Prep(name, {k: data[k] for k in TRAIN})
net = BNet(prep.nin, prep.layers, 64, drop)
optim = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=wd)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, steps, eta_min=1e-4)
npx = 64 * bs * bs * 3
print(name, tag, "inputs", prep.nin, "radius", prep.rad, "params", sum(p.numel() for p in net.parameters()), flush=True)


def blockloss(w, d, y0, x0, h, wd_):
    """w [1,6,h,wd] -> per-block mean squared error (levels^2 per pixel-channel)"""
    ww = w[0].permute(1, 2, 0)
    sub = {n: d[n][y0:y0 + h, x0:x0 + wd_] for n in ("G", "b", "yy")}
    e = (ww * torch.einsum("...ij,...j->...i", sub["G"], ww)).sum(-1) - 2 * (sub["b"] * ww).sum(-1) + sub["yy"]
    return e.clamp_min(0) * S2 / npx


def predict(d):
    h, w = d["yy"].shape
    x, rl = prep.make(d, 0, 0, h, w)
    return net(x, rl)[0].permute(1, 2, 0)


t0 = time.time()
curve, run = [], 0.0
for step in range(1, steps + 1):
    net.train()
    ls = []
    for _ in range(batch):
        k = TRAIN[rng.integers(len(TRAIN))]
        d = data[k]
        h, w = d["yy"].shape
        S = min(crop, h, w)
        y0, x0 = int(rng.integers(h - S + 1)), int(rng.integers(w - S + 1))
        x, rl = prep.make(d, y0, x0, S, S)
        mse = blockloss(net(x, rl), d, y0, x0, S, S)
        ls.append(mse.mean() if loss_k == "mse" else torch.sqrt(mse + 1.0).mean())
    loss = torch.stack(ls).mean()
    optim.zero_grad(); loss.backward(); optim.step(); sched.step()
    run += loss.item()
    if step % 100 == 0:
        curve.append((step, run / 100)); print(step, "loss %.4f" % (run / 100), "%.0fs" % (time.time() - t0), flush=True); run = 0.0
ttrain = time.time() - t0
net.eval()
res = dict(name=name, tag=tag, steps=steps, seed=seed, wd=wd, drop=drop, loss=loss_k, bs=bs, batch=batch, crop=crop, train_seconds=ttrain, curve=curve)
res["val"] = score(lambda k, d: predict(d), VAL, data, bs)
res["train"] = score(lambda k, d: predict(d), TRAIN, data, bs)
print("val", {g: tuple(round(x, 2) for x in v) for g, v in res["val"].items()}, flush=True)
print("train", {g: tuple(round(x, 2) for x in v) for g, v in res["train"].items()}, flush=True)
json.dump(res, open(f"{T3}/res_{tag}.json", "w"), indent=1)
torch.save(net.state_dict(), f"{T3}/net_{tag}.pt")
