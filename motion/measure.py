#!/usr/bin/env python3
"""Numbers for a motion estimate: end-point error, blockiness, PSNR of the generated middle frame (see motion/README.md).

  measure.py --fgtest PATH --set 'algo=v3' --out DIR --pairs pairs.json [--pairs held.json ...] [--only substr,...] [--label NAME]

Runs `flock gpu.lock ccorral -- fgtest --window 1 --dump ...` on the pairs (chunks of 4, one fresh Interpolator per pair), writes
DIR/measure_<label>.json (every number per pair, the aggregates) and DIR/measure_<label>.md (the tables).
Synthetic pairs (mkpairs.py): EPE (mean / 95th percentile / share above 4 px, full-resolution px, valid pixels), blockiness and PSNR
against the true middle frame.  Real pairs: blockiness of the estimate, PSNR where the real in-between frame is known (held-out
23: whole / edge pixels pooled as blend/edge does).  A row of a stage table is one pass's field (V3: BLK / FD / FF).
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
               "| case | str | EPE mean | EPE p95 | EPE tex | bad>4px | blk | ctrl | truth blk | truth-bc blk | PSNR | PSNR edge |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in CASES:
            for s in STRENGTHS:
                ms = [m for m in syn if m["case"] == c and m["strength"] == s]
                if not ms:
                    continue
                a = agg(ms)
                res["synthetic"]["%s_%s" % (c, s)] = a
                md.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    c, s, fmt(a.get("epe_mean")), fmt(a.get("epe_p95")), fmt(a.get("epe_tex_mean")), fmt(a.get("bad4"), 3), fmt(a["blk16"], 3), fmt(a["ctrl16"], 3),
                    fmt(a.get("truth_blk16"), 3), fmt(a.get("truthbc_blk16"), 3), fmt(a.get("psnr")), fmt(a.get("psnr_edge"))))
        a = agg(syn)
        res["synthetic"]["all"] = a
        md.append("| **all** | | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            fmt(a.get("epe_mean")), fmt(a.get("epe_p95")), fmt(a.get("epe_tex_mean")), fmt(a.get("bad4"), 3), fmt(a["blk16"], 3), fmt(a["ctrl16"], 3),
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fgtest", required=True)
    ap.add_argument("--set", default="", dest="setstr", help="fgtest --set string, e.g. algo=v3,flowScale=2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", action="append", required=True, help="pairs.json (mkpairs.py / mkreal.py); repeatable")
    ap.add_argument("--only", help="comma list of substrings: only pairs whose name contains one")
    ap.add_argument("--label", help="file name stem (default: the --set string made file-safe)")
    a = ap.parse_args()
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
