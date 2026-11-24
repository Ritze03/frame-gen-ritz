"""Task 144 trainer.  usage: train_probe2.py NET OUTNAME --mode {o6,fix,orig} [--seed S] [--steps N]
orig: exactly train_probe.py (random crops from the 56 pairs, real-frame target), seed S (torch init and crop plan; seed 0 = the original).
fix : the fixed 1,120 crops of make_o6crops.py (crops.json), real-frame target.
o6  : same crops, target = the crop's O6 colour ($BLEND/probe/o6crops).
Everything else as train_probe.py: Charbonnier sqrt(d^2+1e-6) /255 scale, batch 16, 6,000 steps, Adam 2e-3 -> 1e-4 cosine, 2 px context margin
(read with the margin, loss on the centre).  For fix / o6 all crops are read into memory once (about 1.5 GB) and the batch order is rng(seed) over the crops."""
import json, math, os, queue, sys, threading, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nets import *

BLEND = os.environ.get("BLEND", "/home/mo/DieGanzeSau/AI_Video_Training_Data/blend")
ROOT = f"{BLEND}/probe"
name, outname = sys.argv[1], sys.argv[2]
mode = sys.argv[sys.argv.index("--mode") + 1]
seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 6000
torch.set_num_threads(4)
torch.manual_seed(seed)
pairs = json.load(open(f"{ROOT}/pairs.json"))
tr = [k for k, p in enumerate(pairs) if p["split"] == "train"]
B, C = 16, 128
rng = np.random.default_rng(seed)
from concurrent.futures import ThreadPoolExecutor
ex = ThreadPoolExecutor(3)
M = MARGIN

if mode == "orig":
    P = {k: Pair(ROOT, k) for k in tr}
    plan = [(tr[rng.integers(len(tr))], int(rng.integers(0, H - C + 1)), int(rng.integers(0, W - C + 1))) for _ in range(steps * B)]

    def batch(s):
        items = [P[k].window(y - M, y + C + M, x - M, x + C + M) for k, y, x in plan[s * B:(s + 1) * B]]
        return tuple(torch.from_numpy(np.stack([it[j] for it in items])) for j in range(3))
else:
    crops = json.load(open(f"{ROOT}/o6crops/crops.json"))
    Xs, Rs, Ys = [], [], []
    t0 = time.time()
    for k in tr:
        p = Pair(ROOT, k)
        o6 = np.load(f"{ROOT}/o6crops/{k:02d}.npy", allow_pickle=False) if mode == "o6" else None
        for i, (y, x) in enumerate(crops[str(k)]):
            X, R, Y = p.window(y - M, y + C + M, x - M, x + C + M)
            if mode == "o6":
                Y[:, M:-M, M:-M] = o6[i].astype(np.float32).transpose(2, 0, 1) / 255.0
            Xs.append(X.astype(np.float16)); Rs.append(R.astype(np.float16)); Ys.append(Y)
    Xs, Rs, Ys = np.stack(Xs), np.stack(Rs), np.stack(Ys)
    print("loaded", Xs.shape, "%.0fs" % (time.time() - t0), flush=True)
    plan = rng.integers(0, len(Xs), steps * B)

    def batch(s):
        ix = plan[s * B:(s + 1) * B]
        return torch.from_numpy(Xs[ix].astype(np.float32)), torch.from_numpy(Rs[ix].astype(np.float32)), torch.from_numpy(Ys[ix])

futs = {}


def get(s):
    for i in range(s, min(s + 8, steps)):
        if i not in futs:
            futs[i] = ex.submit(batch, i)
    return futs.pop(s).result()


net = Net(name)
opt = torch.optim.Adam(net.parameters(), lr=2e-3)
print(name, outname, mode, "seed", seed, "params", sum(p.numel() for p in net.parameters()), "steps", steps, flush=True)
t0 = time.time()
run = 0.0
for s in range(steps):
    lr = 1e-4 + 0.5 * (2e-3 - 1e-4) * (1 + math.cos(math.pi * s / steps))
    for g in opt.param_groups:
        g["lr"] = lr
    X, R, Y = get(s)
    c = mix(net(X, R), X)[:, :, M:-M, M:-M]
    loss = torch.sqrt((c - Y[:, :, M:-M, M:-M]) ** 2 + 1e-6).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    run = 0.98 * run + 0.02 * loss.item() if s else loss.item()
    if s % 250 == 0 or s == steps - 1:
        print(s, "loss %.5f" % run, "%.0fs" % (time.time() - t0), flush=True)
os.makedirs(f"{ROOT}/nets", exist_ok=True)
assert not os.path.exists(f"{ROOT}/nets/{outname}.pt")
torch.save(net.state_dict(), f"{ROOT}/nets/{outname}.pt")
print("done", outname, "%.0fs" % (time.time() - t0), "peak rss MB", __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss // 1024, flush=True)
