"""Baselines from block data (task 147): rule block-mean, one fixed mix (all / per group, fitted on the training pairs), block oracle.
Writes $BLEND/probe/block3/base.json: {method: {split: {group: [whole, edge]}}} for train and val.
usage: block3_base.py [--bs 1|2]"""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block3_lib import *
from block_oracle import gsolve
torch.set_num_threads(2)
bs = int(sys.argv[sys.argv.index("--bs") + 1]) if "--bs" in sys.argv else 1
data = {k: load_pair(k, bs) for k in range(79)}


def fit(pairs):
    G = sum(data[k]["G"].double().sum((0, 1)) for k in pairs).numpy()
    b = sum(data[k]["b"].double().sum((0, 1)) for k in pairs).numpy()
    yy = sum(data[k]["yy"].double().sum() for k in pairs).item()
    return gsolve(G[None], b[None], np.array([yy]), 6)[0][0]


def oracle(k, d):
    h, w = d["yy"].shape
    wt, _ = gsolve(d["G"].double().numpy().reshape(-1, 6, 6), d["b"].double().numpy().reshape(-1, 6), d["yy"].double().numpy().reshape(-1), 6)
    return torch.from_numpy(wt.reshape(h, w, 6))


out = {}
wall = fit(TRAIN)
wgrp = {g: fit([k for k in TRAIN if PAIRS[k]["group"] == g]) for g in GROUPS}
print("fixed all", np.round(wall, 3), flush=True)
for g in GROUPS:
    print("fixed", g, np.round(wgrp[g], 3), flush=True)
meths = {
    "rule-block": lambda k, d: rule6(d),
    "fixed-all": lambda k, d: torch.from_numpy(wall).float().expand(*d["yy"].shape, 6),
    "fixed-group": lambda k, d: torch.from_numpy(wgrp[PAIRS[k]["group"]]).float().expand(*d["yy"].shape, 6),
    "oracle": oracle,
}
for m, f in meths.items():
    out[m] = {}
    for sp, ks in (("val", VAL), ("train", TRAIN)):
        out[m][sp] = score(f, ks, data, bs)
        print(m, sp, {g: tuple(round(x, 2) for x in v) for g, v in out[m][sp].items()}, flush=True)
out["_fixed"] = {"all": wall.tolist(), **{g: wgrp[g].tolist() for g in GROUPS}}
json.dump(out, open(f"{T3}/base{'' if bs == 1 else bs}.json", "w"), indent=1)
