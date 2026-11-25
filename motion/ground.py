#!/usr/bin/env python3
"""The extended ground-truth measure (task 162): how far is the final field (V3 `FF`, F01) from the sparse truth of the real in-between frame, per pair group.

Truth: `common.frame_truth` (64 px patches on a 64 px grid over the whole frame, prev -> mid and mid -> curr agree within 3 px, distinct match, texture), computed once per pair
and cached as TRUTHDIR/<pair>.npy (`ground.py --truth-only`).  Groups: `ground` = the lower half (y >= H/2) of the driving pairs (Forza: the held-out pairs, the scene pairs, the `road`
set), `drive-up` = their upper half, `tp` = the third-person pairs (`tp_*`), `game` = every other game pair (cs2, minecraft, battlefield, first person, the drone).  A pair list is a
JSON list as `measure.py --pairs` takes (prev / curr / mid paths); `study_pairs.py` writes the study's.

    python ground.py --pairs study.json --truth TRUTHDIR --out OUT --run v3=FGTEST:algo=v3 --run today=REF:algo=v3,v3Motion=1 --run new=FGTEST:algo=v3,v3Motion=1
        writes OUT/ground.json (per run and group: patches, EPE mean / median, share within 4 px, share still where the truth is above 4 px; per pair the same for the ground),
        OUT/ground.md and OUT/fields/<pair>.npz (the F01 field of every run, float16, for the sheets)
    python ground.py --truth-only --pairs study.json --truth TRUTHDIR          # (re)compute the truth files

The field is read at a patch centre as the mean over the 5x5 flow px around it, as `measure.py --road` does.  Nothing here needs a window; GPU runs take the lock."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

GROUPS = ("ground", "drive-up", "tp", "game")


def group_of(name, y, H=1080):
    if name.startswith(("racing_forza", "road_", "forza_")):
        return "ground" if y >= H / 2 else "drive-up"
    return "tp" if name.startswith("tp_") else "game"


def truth_for(q, tdir):
    f = os.path.join(tdir, q["name"] + ".npy")
    if os.path.exists(f):
        return np.load(f)
    t = C.frame_truth(*(C.read_rgba(q[k], q["W"], q["H"]) for k in ("prev", "mid", "curr")))
    os.makedirs(tdir, exist_ok=True)
    np.save(f, t)
    return t


def summarise(rows):
    """rows: [n, 5] (err, still & big, big, within4, |truth|) -> dict"""
    if not len(rows):
        return dict(n=0)
    big = rows[:, 2].sum()
    return dict(n=int(len(rows)), mean=float(rows[:, 0].mean()), median=float(np.median(rows[:, 0])), within4=float(rows[:, 3].mean()), still=float(rows[:, 1].sum() / max(big, 1)),
                big=int(big))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", action="append", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out")
    ap.add_argument("--run", action="append", default=[], help="TAG=FGTEST:SETSTRING")
    ap.add_argument("--only")
    ap.add_argument("--truth-only", action="store_true")
    ap.add_argument("--no-fields", action="store_true")
    a = ap.parse_args()
    pairs = []
    for p in a.pairs:
        pairs += C.load_pairs(p, a.only)
    pairs = [q for q in pairs if q.get("mid")]
    if a.truth_only:
        for q in pairs:
            print(q["name"], len(truth_for(q, a.truth)), flush=True)
        return
    os.makedirs(os.path.join(a.out, "fields"), exist_ok=True)
    truth = {q["name"]: truth_for(q, a.truth) for q in pairs}
    pairs = [q for q in pairs if len(truth[q["name"]])]
    per = {}   # tag -> group -> list of arrays ; tag -> pair -> array
    perpair = {}
    for spec in a.run:
        tag, rest = spec.split("=", 1)
        exe, st = rest.split(":", 1)
        for r in C.run_fgtest(exe, st, pairs, os.path.join(a.out, "work_" + tag), keep_out=False):
            q = r["pair"]
            n = q["name"]
            fs = r["fg"]["fscale"]
            F = C.fields(r, st)["FF"]["F"]
            t = truth[n]
            v = np.array([F[max(int(y) // fs - 2, 0):int(y) // fs + 3, max(int(x) // fs - 2, 0):int(x) // fs + 3].reshape(-1, 2).mean(0) for x, y, _, _ in t])
            e = np.hypot(v[:, 0] - t[:, 2], v[:, 1] - t[:, 3])
            mag = np.hypot(t[:, 2], t[:, 3])
            rows = np.stack([e, (np.hypot(v[:, 0], v[:, 1]) < 0.5) & (mag > 4), mag > 4, e < 4, mag], 1)
            for g in GROUPS:
                sel = np.array([group_of(n, y, q["H"]) == g for y in t[:, 1]])
                if sel.any():
                    per.setdefault(tag, {}).setdefault(g, []).append(rows[sel])
            perpair.setdefault(tag, {})[n] = rows
            np.savez_compressed(os.path.join(a.out, "fields", "%s_%s.npz" % (n, tag)), F=F.astype(np.float16), fs=fs)
            print("%-6s %s" % (tag, n), flush=True)
            if a.no_fields:
                os.remove(os.path.join(a.out, "fields", "%s_%s.npz" % (n, tag)))
    res = {}
    for tag, gs in per.items():
        res[tag] = {g: summarise(np.concatenate(v)) for g, v in gs.items()}
        res[tag]["pairs"] = {n: {g: summarise(rows[[group_of(n, y) == g for y in truth[n][:, 1]]]) for g in ("ground",)} for n, rows in perpair[tag].items()}
    json.dump(res, open(os.path.join(a.out, "ground.json"), "w"), indent=1)
    md = ["| run | group | patches | EPE mean | EPE median | within 4 px | still where truth > 4 px |", "|---|---|---|---|---|---|---|"]
    for tag in res:
        for g in GROUPS:
            s = res[tag].get(g)
            if s and s["n"]:
                md.append("| %s | %s | %d | %.2f | %.2f | %.1f %% | %.1f %% |" % (tag, g, s["n"], s["mean"], s["median"], 100 * s["within4"], 100 * s["still"]))
    open(os.path.join(a.out, "ground.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
