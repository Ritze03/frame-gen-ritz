"""Scores the task-144 nets on the 23 held-out pairs against the REAL frame, as score.py does -> $BLEND/probe/scores2.json (per-pair accumulators,
merged with what is already there).  usage: score2.py NAME:NET,NAME:NET ...   e.g. px8-o6:px8,n3-all-o6:n3-all"""
import json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nets import *
from score import edge_mask, accumulate, ROOT

torch.set_num_threads(4)
spec = dict(s.split(":") for s in sys.argv[1].split(","))
f2 = f"{ROOT}/scores2.json"
res = json.load(open(f2)) if os.path.exists(f2) else {}
nets = {}
for n, base in spec.items():
    nets[n] = Net(base)
    nets[n].load_state_dict(torch.load(f"{ROOT}/nets/{n}.pt", weights_only=True))
    nets[n].eval()
pairs = json.load(open(f"{ROOT}/pairs.json"))
for k, p in enumerate(pairs):
    if p["split"] != "val":
        continue
    t0 = time.time()
    d = f"{ROOT}/pairs/{k:02d}"
    y = np.load(f"{d}/y.npy", allow_pickle=False)
    em = edge_mask(y)
    yf = y.astype(np.float32)
    pr = Pair(ROOT, k)
    r = res.setdefault(str(k), {})
    for n, net in nets.items():
        r[n] = accumulate(predict(net, pr), yf, em)
    print(k, "%.0fs" % (time.time() - t0), flush=True)
json.dump(res, open(f2, "w"), indent=1)
