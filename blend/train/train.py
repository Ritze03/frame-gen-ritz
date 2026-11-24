"""Trains the three learned-blend nets A, B, C together (same batches, one gather per step).
usage: train.py OUTDIR [--teacher] [--steps N] [--seed S] [--threads T] [--lr X] [--lr-end X] [--init DIR] [--mix FILE] [--base DIR] [--case CASE] [--last]
  --last saves the weights of the last step (per net) instead of the best step; log.json / run.json record keep = last (nothing is recorded without --last).
  --case CASE keeps only the local rows (training and validation) of samples whose manifest case is CASE, e.g. 60to120.
  OUTDIR gets A.npz B.npz C.npz (+ log.json). --teacher adds the cached teacher training rows (weight 0.5 per row).
Net = gpu/ml/blend/ref.py: 8 hidden units (max(z, 0.1 z)), softmax(log(max(rule, 1e-3)) [+ -5, -5] + z2), W2 = b2 = 0 at the start.
A: Charbonnier distance of the colour to the four-choice oracle colour; B: to the target; C: six choices, to the six-choice oracle colour.
Fine-tuning: --init DIR starts A, B, C from DIR/{A,B,C}.npz instead of from the rule (and, at the end, writes scores.md + run.json, see finetune_report.py;
  --base DIR = the weights the table compares with, default blend/weights/base). --mix FILE (from export_sample.py) adds rows of other footage: half of every batch
  comes from the local cache (local group balance), half from the file (the file's group balance); the file's validation rows are validated apart and the best step
  is the one with the lowest mean of the local and the file's validation loss (per net). --lr-end = the final rate of the cosine (default 1e-4).
Batches: 65536 rows, 40 % forza, 60 % evenly over the other groups; Adam 2e-3 -> 1e-4 cosine; validation (real rows, group balanced the same way) every 500 steps, best kept.
"""
import sys, os, json, time, math
import numpy as np
import torch
from feat import *

args = sys.argv[1:]
out = args[0] if args else ""
use_teacher = "--teacher" in args
steps = int(args[args.index("--steps") + 1]) if "--steps" in args else 15000
seed = int(args[args.index("--seed") + 1]) if "--seed" in args else 0
torch.set_num_threads(int(args[args.index("--threads") + 1]) if "--threads" in args else 4)
BATCH, H = 65536, 8
LR0 = float(args[args.index("--lr") + 1]) if "--lr" in args else 2e-3
LR1 = float(args[args.index("--lr-end") + 1]) if "--lr-end" in args else 1e-4


def opt(name):
    return args[args.index(name) + 1] if name in args else None


INIT, MIX, CASE = opt("--init"), opt("--mix"), opt("--case")
LAST = "--last" in args   # save the weights of the last step instead of the best step (per net)
VARS = {"A": (25, 4, "O4"), "B": (25, 4, "Y"), "C": (34, 6, "O6")}


def load(name):
    return {k: np.load(f"{CACHE}/{name}_{k}.npy") for k in ("X", "R", "Y", "O4", "O6", "G", "T", "S")}


def concat(a, b):
    return {k: np.concatenate([a[k], b[k]]) for k in a}


def forward(p, X, R, K, NI):
    W1, b1, W2, b2 = p
    z1 = X[:, :NI] @ W1.T + b1
    z2 = torch.where(z1 > 0, z1, 0.1 * z1) @ W2.T + b2
    lg = torch.log(torch.clamp(R, min=1e-3))
    if K == 6:
        lg = torch.cat([lg, torch.full((len(lg), 2), -5.0)], 1)
    w = torch.softmax(lg + z2, 1)
    C = (w[:, :4, None] * X[:, :12].view(-1, 4, 3)).sum(1)
    if K == 6:
        C = C + w[:, 4:5] * X[:, 25:28] + w[:, 5:6] * X[:, 28:31]
    return C, w


def charb(C, Y):
    return torch.sqrt((C - Y) ** 2 + 1e-6).mean(1)


def init(NI, K, rng):
    return [torch.tensor(rng.normal(0, 0.3 / math.sqrt(NI), (H, NI)), dtype=torch.float32, requires_grad=True),
            torch.zeros(H, requires_grad=True), torch.zeros(K, H, requires_grad=True), torch.zeros(K, requires_grad=True)]


def balanced(per):
    """The validation balance of the batches: 40 % Forza, 60 % the mean of the other groups (only one kind present: its mean)."""
    others = [x for g, x in per.items() if g != 0]
    return 0.4 * per[0] + 0.6 * np.mean(others) if 0 in per and others else np.mean(list(per.values()))


def load_net(path_dir):
    z = [np.load(f"{path_dir}/{v}.npz") for v in VARS]
    return {v: [torch.tensor(np.asarray(zz[k], np.float32), requires_grad=True) for k in ("W1", "b1", "W2", "b2")] for v, zz in zip(VARS, z)}


def load_mix(path):
    """export_sample.py file -> (train dict, val dict), G mapped onto this module's GROUPS indices."""
    z = np.load(path)
    names = [str(n) for n in z["names"]]
    assert all(n in GROUPS for n in names), f"{path}: unknown group in {names}"
    gmap = np.array([GROUPS.index(n) for n in names], np.int8)
    out = []
    for sp in ("train", "val"):
        d = {k: z[f"{sp}_{k}"] for k in ("X", "R", "Y", "O4", "O6")}
        d["G"] = gmap[z[f"{sp}_G"]]
        d["T"] = np.zeros(len(d["G"]), np.int8)
        out.append(d)
    return out


def mix_counts(groups_present, B):
    nf = int(round(0.4 * B)) if 0 in groups_present else 0
    others = [g for g in groups_present if g != 0]
    c = {g: (B - nf) // len(others) for g in others}
    if nf:
        c[0] = B - sum(c.values())
    return c


def main():
    tr = load("real_train")
    if use_teacher:
        tr = concat(tr, load("teacher_train"))
    va = load("real_val")
    # leakage guard: every cached row's sample must be a train sample (cache builder asserted per file too)
    man = json.load(open(f"{BLEND}/frames/manifest.json"))["samples"]
    split_of = np.array([s["split"] == "val" for s in man])
    assert not split_of[tr["S"]].any(), "validation row in the training set"
    assert split_of[va["S"]].all()
    if CASE:   # --case: only rows of samples with this manifest case (training and validation alike); pools / counts below follow from G
        keep = np.array([s["case"] == CASE for s in man])
        tr, va = ({k: v[keep[d["S"]]] for k, v in d.items()} for d in (tr, va))
        assert len(tr["G"]) and len(va["G"]), f"--case {CASE}: no rows"
    rows = {}
    for nm, d in (("train", tr), ("val", va)):
        for t in (0, 1):
            rows[(nm, t)] = {GROUPS[g]: int(((d["G"] == g) & (d["T"] == t)).sum()) for g in range(len(GROUPS))}
    print("rows per group", json.dumps({f"{a}/{'teacher' if t else 'real'}": v for (a, t), v in rows.items() if sum(v.values())}), flush=True)
    pools = {g: np.nonzero(tr["G"] == g)[0] for g in range(len(GROUPS)) if (tr["G"] == g).any()}
    vpools = {g: np.nonzero(va["G"] == g)[0] for g in range(len(GROUPS)) if (va["G"] == g).any()}
    cnt = mix_counts(list(pools), BATCH)
    T = {k: torch.from_numpy(v) for k, v in tr.items() if k in ("X", "R", "Y", "O4", "O6", "T")}
    V = {k: torch.from_numpy(v) for k, v in va.items() if k in ("X", "R", "Y", "O4", "O6")}
    rng = np.random.default_rng(seed)
    P = load_net(INIT) if INIT else {v: init(NI, K, rng) for v, (NI, K, _) in VARS.items()}
    mrows = {}
    if MIX:
        mtr, mva = load_mix(MIX)
        mpools = {g: np.nonzero(mtr["G"] == g)[0] for g in range(len(GROUPS)) if (mtr["G"] == g).any()}
        mvpools = {g: np.nonzero(mva["G"] == g)[0] for g in range(len(GROUPS)) if (mva["G"] == g).any()}
        cnt_l, cnt_m = mix_counts(list(pools), BATCH // 2), mix_counts(list(mpools), BATCH - BATCH // 2)
        MT = {k: torch.from_numpy(v) for k, v in mtr.items()}
        MV = {k: torch.from_numpy(v) for k, v in mva.items()}
        mrows = {nm: {GROUPS[g]: int((d["G"] == g).sum()) for g in range(len(GROUPS)) if (d["G"] == g).any()} for nm, d in (("train", mtr), ("val", mva))}
        print("mix file rows per group", json.dumps(mrows), "| batch: local", {GROUPS[g]: c for g, c in cnt_l.items()}, "file", {GROUPS[g]: c for g, c in cnt_m.items()}, flush=True)
    optim = torch.optim.Adam([t for v in P.values() for t in v], lr=LR0)
    best = {v: (1e9, None) for v in VARS}
    log = []

    def validate(VD=None, pl=None):
        VD, pl = (V, vpools) if VD is None else (VD, pl)
        res = {}
        with torch.no_grad():
            for v, (NI, K, tgt) in VARS.items():
                per = {}
                for g, ix in pl.items():
                    s = 0.0
                    for a in range(0, len(ix), 262144):
                        j = torch.from_numpy(ix[a:a + 262144])
                        C, _ = forward(P[v], VD["X"][j], VD["R"][j], K, NI)
                        s += charb(C, VD[tgt][j]).sum().item()
                    per[g] = s / len(ix)
                # same 40 / 60 balance as the batches
                res[v] = balanced(per)
        return res

    t0 = time.time()
    for step in range(steps + 1):
        if step % 500 == 0 or step == steps:
            vl = validate()
            vm = validate(MV, mvpools) if MIX else None
            sel = {v: (vl[v] + vm[v]) / 2 for v in VARS} if MIX else vl   # the best step: mean of the local and the file's validation loss
            for v in VARS:
                if sel[v] < best[v][0]:
                    best[v] = (sel[v], [t.detach().clone() for t in P[v]], step)
            log.append(dict(step=step, val=vl, sec=time.time() - t0, **({"val_mix": vm, "val_sel": sel} if MIX else {})))
            print(step, {k: round(x, 5) for k, x in vl.items()}, *(["mix", {k: round(x, 5) for k, x in vm.items()}] if MIX else []), "%.0fs" % (time.time() - t0), flush=True)
        if step == steps:
            break
        lr = LR1 + 0.5 * (LR0 - LR1) * (1 + math.cos(math.pi * step / steps))
        for gp in optim.param_groups:
            gp["lr"] = lr
        if MIX:
            idx = torch.from_numpy(np.concatenate([rng.choice(pools[g], c) for g, c in cnt_l.items()]))
            midx = torch.from_numpy(np.concatenate([rng.choice(mpools[g], c) for g, c in cnt_m.items()]))
            X, R = torch.cat([T["X"][idx], MT["X"][midx]]), torch.cat([T["R"][idx], MT["R"][midx]])
            wt = torch.cat([1.0 - 0.5 * T["T"][idx].float(), torch.ones(len(midx))])
            tg = {v: torch.cat([T[t][idx], MT[t][midx]]) for v, (_, _, t) in VARS.items()}
        else:
            idx = torch.from_numpy(np.concatenate([rng.choice(pools[g], c) for g, c in cnt.items()]))
            X, R, wt = T["X"][idx], T["R"][idx], 1.0 - 0.5 * T["T"][idx].float()
            tg = {v: T[t][idx] for v, (_, _, t) in VARS.items()}
        loss = 0.0
        for v, (NI, K, tgt) in VARS.items():
            C, _ = forward(P[v], X, R, K, NI)
            loss = loss + (charb(C, tg[v]) * wt).sum() / wt.sum()
        optim.zero_grad()
        loss.backward()
        optim.step()
    if LAST:
        best = {v: (sel[v], [t.detach().clone() for t in P[v]], steps) for v in VARS}
    os.makedirs(out, exist_ok=True)
    for v in VARS:
        W1, b1, W2, b2 = (t.numpy().astype(np.float32) for t in best[v][1])
        np.savez(f"{out}/{v}.npz", W1=W1, b1=b1, W2=W2, b2=b2)
    json.dump(dict(log=log, best={v: [best[v][0], best[v][2]] for v in VARS}, rows={f"{a}/{t}": r for (a, t), r in rows.items()},
                   steps=steps, seconds=time.time() - t0, teacher=use_teacher, batch_counts={GROUPS[g]: c for g, c in cnt.items()},
                   init=bool(INIT), mix_rows=mrows, **({"keep": "last"} if LAST else {}), **({"batch_counts_mix": {GROUPS[g]: c for g, c in cnt_m.items()}} if MIX else {})),
              open(f"{out}/log.json", "w"), indent=1)
    print("best", {v: (round(best[v][0], 5), best[v][2]) for v in VARS})
    if INIT:
        sys.modules.setdefault("train", sys.modules[__name__])
        import finetune_report
        finetune_report.write(out, INIT, opt("--base"), va, mva if MIX else None, rows, mrows, cnt_l if MIX else cnt, cnt_m if MIX else None,
                              {v: best[v][2] for v in VARS}, steps, time.time() - t0, dict(
                                  steps=steps, seed=seed, lr=LR0, lr_end=LR1, threads=torch.get_num_threads(), batch=BATCH,
                                  init=os.path.basename(INIT.rstrip("/")), mix=os.path.basename(MIX) if MIX else None, teacher=use_teacher, **({"keep": "last"} if LAST else {})))


if __name__ == "__main__":
    main()
