#!/usr/bin/env python3
"""Numbers for a motion estimate: end-point error, blockiness, PSNR of the generated middle frame (see motion/README.md).

  measure.py --fgtest PATH --set 'algo=v3' --out DIR --pairs pairs.json [--pairs held.json ...] [--only substr,...] [--label NAME]
  measure.py --road --fgtest NEW [--ref-fgtest TODAY] --out DIR --pairs road.json [--only ...]     "still against the model" (below)

Runs `flock gpu.lock ccorral -- fgtest --window 1 --dump ...` on the pairs (chunks of 4, one fresh Interpolator per pair), writes
DIR/measure_<label>.json (every number per pair, the aggregates) and DIR/measure_<label>.md (the tables).
Synthetic pairs (mkpairs.py): EPE (mean / 95th percentile / share above 4 px, full-resolution px, valid pixels), blockiness and PSNR
against the true middle frame.  Real pairs: blockiness of the estimate, PSNR where the real in-between frame is known (held-out
23: whole / edge pixels pooled as blend/edge does).  A row of a stage table is one pass's field (V3: BLK / FD / FF).

Road mode (task 161): `--road` runs V3 (--set algo=v3), V3.5 of the reference build (--ref-fgtest, "today") and V3.5 of --fgtest ("new", --set algo=v3,v3Motion=1) on the pairs
and reports "still against the model": the share of the pixels with |final field| < 0.5 px (FF) among all pixels of the frame where V3.5's regional camera model predicts more than
2 px with reliability above 0.5 (common.still_vs_model; the model is read from the `.v3db` dump).  Columns: V3 and today's V3.5 against today's model, the new V3.5 against its own
model and against today's; the mean |final - model| over the counted pixels.  Writes DIR/road.json, DIR/road.md and DIR/fields/<pair>.npz (F fields, models, for maps.py --road).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CASES = ["pan", "roll_c", "roll_o", "zoom_c", "zoom_o", "rollzoom", "chase", "orbit"]
STRENGTHS = ["s", "m", "l"]


def agg(ms, key=None, stage=None):
    """aggregate of per-pair metric dicts: mean of per-pair values; PSNR pooled (summed squared error)"""
    out = {}
    if not ms:
        return out
    if ms[0].get("se") is not None and all("se" in m for m in ms):
        out["psnr"] = C.psnr(sum(m["se"] for m in ms), sum(m["n"] for m in ms))
        out["psnr_edge"] = C.psnr(sum(m["se_e"] for m in ms), sum(m["n_e"] for m in ms))
    st = stage or ms[0].get("main")
    fl = [m["fields"][st] for m in ms if st in m["fields"]]
    for k in fl[0] if fl else []:
        out[k] = float(np.mean([f[k] for f in fl if k in f]))
    for k in ms[0]:
        if k.startswith("truth"):
            out[k] = float(np.mean([m[k] for m in ms]))
    out["n"] = len(ms)
    return out


def fmt(x, d=2):
    return "-" if x is None else ("%." + str(d) + "f") % x


def tables(allm, setstr, label):
    syn = [m for m in allm if m["group"] == "synthetic"]
    real = [m for m in allm if m["group"] != "synthetic"]
    md = ["# measure: `%s`  (%s)" % (setstr, label), ""]
    res = {"set": setstr, "label": label, "synthetic": {}, "stages": {}, "real": {}}
    if syn:
        stage_names = [k for k in syn[0]["fields"] if not k.startswith("L")]
        level_names = sorted((k for k in syn[0]["fields"] if k.startswith("L")), reverse=True)
        md += ["## Synthetic pairs, per case and strength (mean over the source frames; field = `%s`)" % syn[0]["main"], "",
               "EPE in full-resolution px over valid pixels (mean of per-pair mean | mean of per-pair 95th percentile | mean over the textured pixels only (5x5-dilated top decile of the source's Sobel magnitude, where motion is observable) | share of pixels above 4 px). "
               "`blk` = blockiness (0 smooth .. 1 per-block constant) at 16 px blocks, `ctrl` = the same at a period that is no harmonic of the block "
               "(period 11 flow px instead of 8: no harmonic, should be near 0). `truth` / `truth-bc` = the true field and its per-block-constant version (the two "
               "reference values). PSNR of the generated t = 0.5 frame against the true one: whole / edge pixels, pooled.", "",
               "| case | str | EPE mean | EPE p95 | EPE tex | bad>4px | false still | blk | ctrl | truth blk | truth-bc blk | PSNR | PSNR edge |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in CASES:
            for s in STRENGTHS:
                ms = [m for m in syn if m["case"] == c and m["strength"] == s]
                if not ms:
                    continue
                a = agg(ms)
                res["synthetic"]["%s_%s" % (c, s)] = a
                md.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    c, s, fmt(a.get("epe_mean")), fmt(a.get("epe_p95")), fmt(a.get("epe_tex_mean")), fmt(a.get("bad4"), 3), fmt(a.get("false_still"), 3), fmt(a["blk16"], 3), fmt(a["ctrl16"], 3),
                    fmt(a.get("truth_blk16"), 3), fmt(a.get("truthbc_blk16"), 3), fmt(a.get("psnr")), fmt(a.get("psnr_edge"))))
        a = agg(syn)
        res["synthetic"]["all"] = a
        md.append("| **all** | | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            fmt(a.get("epe_mean")), fmt(a.get("epe_p95")), fmt(a.get("epe_tex_mean")), fmt(a.get("bad4"), 3), fmt(a.get("false_still"), 3), fmt(a["blk16"], 3), fmt(a["ctrl16"], 3),
            fmt(a.get("truth_blk16"), 3), fmt(a.get("truthbc_blk16"), 3), fmt(a.get("psnr")), fmt(a.get("psnr_edge"))))
        md += ["", "Blockiness at other block sizes (all synthetic pairs; px = full-resolution): " +
               ", ".join("blk%d %s (ctrl %s)" % (b, fmt(a["blk%d" % b], 3), fmt(a["ctrl%d" % b], 3)) for b in C.BLOCKS_FULL), ""]
        if len(stage_names) > 1:
            md += ["## Stages (EPE mean / p95 / textured / blk16 / step16 px, mean over all strengths and sources, per case)", "",
                   "| case | " + " | ".join(stage_names) + " |", "|---|" + "---|" * len(stage_names)]
            for c in CASES + ["all"]:
                ms = [m for m in syn if m["case"] == c] if c != "all" else syn
                if not ms:
                    continue
                cells = []
                for st in stage_names:
                    a = agg(ms, stage=st)
                    res["stages"].setdefault(c, {})[st] = a
                    cells.append("%s / %s / %s / %s / %s" % (fmt(a.get("epe_mean")), fmt(a.get("epe_p95")), fmt(a.get("epe_tex_mean")), fmt(a["blk16"], 3), fmt(a["step16"], 3)))
                md.append("| %s | " % c + " | ".join(cells) + " |")
            md.append("")
        if level_names:
            md += ["## Levels: EPE mean / p95 of each pyramid level's final block field (nearest-filled to the flow grid; level l: blocks of 4 x 2^l flow px), then BLK = level 0", "",
                   "| case | " + " | ".join(level_names + ["BLK"]) + " |", "|---|" + "---|" * (len(level_names) + 1)]
            for c in CASES + ["all"]:
                ms = [m for m in syn if m["case"] == c] if c != "all" else syn
                if ms:
                    cells = []
                    for st in level_names + ["BLK"]:
                        a = agg(ms, stage=st)
                        res["stages"].setdefault(c, {})[st] = a
                        cells.append("%s / %s" % (fmt(a.get("epe_mean")), fmt(a.get("epe_p95"))))
                    md.append("| %s | " % c + " | ".join(cells) + " |")
            md.append("")
    if real:
        md += ["## Real pairs", "", "PSNR whole / edge pixels pooled (mse over pixels and RGB), blk16 / ctrl16 = blockiness of the estimate (no truth: nothing excluded), "
               "mean over the pairs.", "", "| group | pairs | PSNR | PSNR edge | blk16 | ctrl16 |", "|---|---|---|---|---|---|"]
        groups = sorted({m["group"] for m in real})
        for g in groups + ["all"]:
            ms = [m for m in real if m["group"] == g] if g != "all" else real
            a = agg(ms)
            res["real"][g] = a
            md.append("| %s | %d | %s | %s | %s | %s |" % (g, len(ms), fmt(a.get("psnr")), fmt(a.get("psnr_edge")), fmt(a["blk16"], 3), fmt(a["ctrl16"], 3)))
        md.append("")
        stn = [k for k in real[0]["fields"] if not k.startswith("L")]
        if len(stn) > 1:
            md += ["Stages on the real pairs (blk16 / step16 px, mean over the pairs): " + "; ".join(
                "%s %s / %s" % (st, fmt(agg(real, stage=st)["blk16"], 3), fmt(agg(real, stage=st)["step16"], 3)) for st in stn), ""]
        if len(real) <= 40:
            md += ["| pair | PSNR | PSNR edge | blk16 | ctrl16 |", "|---|---|---|---|---|"]
            for m in real:
                f = m["fields"][m["main"]]
                md.append("| %s | %s | %s | %s | %s |" % (m["name"], fmt(m.get("psnr")), fmt(m.get("psnr_edge")), fmt(f["blk16"], 3), fmt(f["ctrl16"], 3)))
            md.append("")
    return "\n".join(md), res


def road_main(a):
    pairs = []
    for p in a.pairs:
        pairs += C.load_pairs(p, a.only)
    os.makedirs(os.path.join(a.out, "fields"), exist_ok=True)
    cfgs = [("v3", a.fgtest, "algo=v3")]
    if a.ref_fgtest:
        cfgs.append(("today", a.ref_fgtest, "algo=v3,v3Motion=1"))
    cfgs.append(("new", a.fgtest, "algo=v3,v3Motion=1"))
    res = {}
    for tag, exe, st in cfgs:
        for r in C.run_fgtest(exe, st, pairs, os.path.join(a.out, "work_" + tag), keep_out=False):
            n = r["pair"]["name"]
            fs = r["fg"]["fscale"]
            F = C.fields(r, st)["FF"]["F"]
            m = C.model_field(r["v3"])
            res.setdefault(n, {})[tag] = dict(F=F.astype(np.float16), fs=fs, model=None if m is None else (m[0].astype(np.float16), m[1].astype(np.float16), m[2]))
            print("%-6s %s" % (tag, n), flush=True)
    rows = []
    for n, d in res.items():
        ref = d.get("today", d["new"])["model"]
        row = dict(name=n)
        for tag, mod in (("v3", ref), ("today", ref), ("new", d["new"]["model"]), ("new_vs_today", ref)):
            src = d.get("new" if tag == "new_vs_today" else tag)
            if src is not None and mod is not None:
                fs = src["fs"]
                F = src["F"].astype(np.float32)
                D, Q, ok = mod[0].astype(np.float32) * fs, mod[1].astype(np.float32), mod[2]
                mag = np.hypot(D[..., 0], D[..., 1])
                sel = ok & (mag > 2.0) & (Q > 0.5)
                st_m = sel & (np.hypot(F[..., 0], F[..., 1]) < 0.5)
                err = np.hypot(F[..., 0] - D[..., 0], F[..., 1] - D[..., 1])
                row[tag] = float(st_m.sum() / max(sel.sum(), 1))
                row[tag + "_err"] = float(err[st_m].mean()) if st_m.any() else 0.0
                row[tag + "_sel"] = float(sel.mean())
        rows.append(row)
        np.savez_compressed(os.path.join(a.out, "fields", n + ".npz"), **{k + "_F": v["F"] for k, v in d.items()},
                            **{k + "_D": v["model"][0] for k, v in d.items() if v["model"] is not None},
                            **{k + "_Q": v["model"][1] for k, v in d.items() if v["model"] is not None})
    # the sparse truth of the ground from the real in-between frame (common.road_truth): EPE of the fields and of the model at the patch centres
    pp = {q["name"]: q for q in pairs}
    pool = {}
    for n, d in res.items():
        q = pp[n]
        if not q.get("mid"):
            continue
        rt = C.road_truth(*(C.read_rgba(q[k], q["W"], q["H"]) for k in ("prev", "mid", "curr")), y_from=0.5)
        if not len(rt):
            continue
        rows_n = [r for r in rows if r["name"] == n][0]
        rows_n["truth_patches"] = len(rt)
        for tag, src in d.items():
            fs = src["fs"]
            F = src["F"].astype(np.float32)
            for kind, G in (("F", F), ("model", None if src["model"] is None else src["model"][0].astype(np.float32) * fs)):
                if G is None:
                    continue
                v = np.array([G[max(int(y) // fs - 2, 0):int(y) // fs + 3, max(int(x) // fs - 2, 0):int(x) // fs + 3].reshape(-1, 2).mean(0) for x, y, _, _ in rt])
                e = np.hypot(v[:, 0] - rt[:, 2], v[:, 1] - rt[:, 3])
                still = np.hypot(v[:, 0], v[:, 1]) < 0.5
                big = np.hypot(rt[:, 2], rt[:, 3]) > 4
                pool.setdefault(tag + "_" + kind, []).append(np.stack([e, still & big, big, e < 4, np.hypot(rt[:, 2], rt[:, 3])], 1))
    tm = ["", "Sparse truth patches (real in-between frame, lower half; pooled over the pairs): EPE mean / median / share under 4 px / share still (under 0.5 px) where the truth is above 4 px", "",
          "| field | patches | EPE mean | EPE median | EPE < 4 px | still where truth > 4 px |", "|---|---|---|---|---|---|"]
    tres = {}
    for k, v in pool.items():
        pa = np.concatenate(v)
        tres[k] = dict(n=len(pa), epe_mean=float(pa[:, 0].mean()), epe_median=float(np.median(pa[:, 0])), under4=float(pa[:, 3].mean()), still=float(pa[:, 1].sum() / max(pa[:, 2].sum(), 1)))
        tm.append("| %s | %d | %.2f | %.2f | %.3f | %.3f |" % (k, len(pa), tres[k]["epe_mean"], tres[k]["epe_median"], tres[k]["under4"], tres[k]["still"]))
    md = ["| pair | model>2px share | V3 | V3.5 today | V3.5 new | new vs today's model | mean abs(F - model) today / new (px) |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| %s | %.3f | %.4f | %.4f | %.4f | %.4f | %.1f / %.1f |" % (r["name"], r.get("new_sel", 0), r.get("v3", 0), r.get("today", 0), r.get("new", 0), r.get("new_vs_today", 0), r.get("today_err", 0), r.get("new_err", 0)))
    mean = lambda k: float(np.mean([r.get(k, 0) for r in rows]))
    md.append("| **mean** | %.3f | %.4f | %.4f | %.4f | %.4f | %.1f / %.1f |" % (mean("new_sel"), mean("v3"), mean("today"), mean("new"), mean("new_vs_today"), mean("today_err"), mean("new_err")))
    md += tm
    json.dump(dict(rows=rows, truth=tres), open(os.path.join(a.out, "road.json"), "w"), indent=1)
    open(os.path.join(a.out, "road.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fgtest", required=True)
    ap.add_argument("--set", default="", dest="setstr", help="fgtest --set string, e.g. algo=v3,flowScale=2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", action="append", required=True, help="pairs.json (mkpairs.py / mkreal.py); repeatable")
    ap.add_argument("--only", help="comma list of substrings: only pairs whose name contains one")
    ap.add_argument("--label", help="file name stem (default: the --set string made file-safe)")
    ap.add_argument("--road", action="store_true", help="road mode: 'still against the model' of V3, V3.5 (--ref-fgtest) and V3.5 (--fgtest)")
    ap.add_argument("--ref-fgtest", help="road mode: the reference build ('today')")
    a = ap.parse_args()
    if a.road:
        return road_main(a)
    label = a.label or (a.setstr.replace(",", "_").replace("=", "-") or "default")
    os.makedirs(a.out, exist_ok=True)
    pairs = []
    for p in a.pairs:
        pairs += C.load_pairs(p, a.only)
    allm = []
    for i, r in enumerate(C.run_fgtest(a.fgtest, a.setstr, pairs, os.path.join(a.out, "work_" + label))):
        m = C.pair_metrics(r, a.setstr)
        allm.append(m)
        f = m["fields"].get(m["main"], {})
        print("%3d/%d %-40s epe %s blk16 %s psnr %s" % (i + 1, len(pairs), m["name"], fmt(f.get("epe_mean")), fmt(f.get("blk16"), 3), fmt(m.get("psnr"))), flush=True)
    md, res = tables(allm, a.setstr, label)
    res["pairs"] = allm
    json.dump(res, open(os.path.join(a.out, "measure_%s.json" % label), "w"), indent=1)
    open(os.path.join(a.out, "measure_%s.md" % label), "w").write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
