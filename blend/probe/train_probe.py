"""Trains one probe network on the 56 training pairs.  usage: train_probe.py NAME [--steps N] [--bench]
Charbonnier sqrt((C-Y)^2 + 1e-6) (C = sum of weights * points, Y = the real frame, both /255), mean over RGB and pixels; batch 16 random 128x128 crops;
Adam 2e-3 -> 1e-4 cosine; seed 0 (the same crops in the same order for every net).  Saves $BLEND/probe/nets/NAME.pt."""
import json, math, os, queue, sys, threading, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nets import *

BLEND = os.environ.get("BLEND", "/home/mo/DieGanzeSau/AI_Video_Training_Data/blend")
ROOT = f"{BLEND}/probe"
name = sys.argv[1]
steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 6000
BENCH = "--bench" in sys.argv
torch.set_num_threads(4)
torch.manual_seed(0)
pairs = json.load(open(f"{ROOT}/pairs.json"))
tr = [k for k, p in enumerate(pairs) if p["split"] == "train"]
assert len(tr) == 56
P = {k: Pair(ROOT, k) for k in tr}
B, C = 16, 128
rng = np.random.default_rng(0)
plan = [(tr[rng.integers(len(tr))], int(rng.integers(0, H - C + 1)), int(rng.integers(0, W - C + 1))) for _ in range(steps * B)]
from concurrent.futures import ThreadPoolExecutor
ex = ThreadPoolExecutor(3)


def batch(s):
    items = [P[k].window(y - MARGIN, y + C + MARGIN, x - MARGIN, x + C + MARGIN) for k, y, x in plan[s * B:(s + 1) * B]]
    return tuple(torch.from_numpy(np.stack([it[j] for it in items])) for j in range(3))


futs = {}


def get(s):
    for i in range(s, min(s + 8, steps)):
        if i not in futs:
            futs[i] = ex.submit(batch, i)
    return futs.pop(s).result()


net = Net(name)
opt = torch.optim.Adam(net.parameters(), lr=2e-3)
print(name, "params", sum(p.numel() for p in net.parameters()), "steps", steps, flush=True)
t0 = time.time()
run = 0.0
for s in range(steps):
    lr = 1e-4 + 0.5 * (2e-3 - 1e-4) * (1 + math.cos(math.pi * s / steps))
    for g in opt.param_groups:
        g["lr"] = lr
    X, R, Y = get(s)
    c = mix(net(X, R), X)[:, :, MARGIN:-MARGIN, MARGIN:-MARGIN]
    loss = torch.sqrt((c - Y[:, :, MARGIN:-MARGIN, MARGIN:-MARGIN]) ** 2 + 1e-6).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    run = 0.98 * run + 0.02 * loss.item() if s else loss.item()
    if s % 250 == 0 or s == steps - 1:
        print(s, "loss %.5f" % run, "%.0fs" % (time.time() - t0), flush=True)
    if BENCH and s == 40:
        print("sec/step", (time.time() - t0) / 41)
        sys.exit()
os.makedirs(f"{ROOT}/nets", exist_ok=True)
torch.save(net.state_dict(), f"{ROOT}/nets/{name}.pt")
print("done", name, "%.0fs" % (time.time() - t0), "peak rss MB", __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss // 1024, flush=True)
