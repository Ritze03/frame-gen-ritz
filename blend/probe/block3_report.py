"""Tables for probe 3 (task 147) -> tables3.md (prints it too).  Reads $BLEND/probe/block3/{base.json,base2.json,rule_pix.json,res_*.json}."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block3_lib import PAIRS, TRAIN, VAL, GROUPS, T3, psnr

G5 = GROUPS + ["all"]
rp = {int(k): v for k, v in json.load(open(f"{T3}/rule_pix.json")).items()}


def rule_ref(ks_all):
    r = {}
    for g in G5:
        ks = [k for k in ks_all if g == "all" or PAIRS[k]["group"] == g]
        a = {f: sum(rp[k][f] for k in ks) for f in ("se", "n", "se_e", "n_e")}
        r[g] = (psnr(a["se"], a["n"]), psnr(a["se_e"], a["n_e"]))
    return r


REF = {"val": rule_ref(VAL), "train": rule_ref(TRAIN)}
rows = {"rule (per pixel)": {"val": REF["val"], "train": REF["train"]}}
nm = {"rule-block": "(a) rule weights block-averaged", "fixed-all": "(b) one fixed mix, all groups", "fixed-group": "(c) one fixed mix per group",
      "oracle": "(d) block oracle B=8"}
for f, suffix in ((f"{T3}/base.json", ""), (f"{T3}/base2.json", " [B=16]")):
    if os.path.exists(f):
        b = json.load(open(f))
        for m in nm:
            rows[nm[m] + suffix] = b[m]
for f in sorted(glob.glob(f"{T3}/res_*.json")):
    r = json.load(open(f))
    rows[r["tag"]] = dict(val=r["val"], train=r["train"], meta=r)
b16 = {k: v for k, v in rows.items() if "[B=16]" in k or k.endswith("-b16")}


def refs(sp, bs16=False):
    if bs16:   # B=16 drops the last block row of 135: reference = same pairs, whole-frame rule from pixel data (differs slightly)
        return REF[sp]
    return REF[sp]


out = []
for sp, label in (("val", "Held-out (23 pairs)"), ("train", "Training (56 pairs)")):
    for kind, ki in (("whole picture", 0), ("edge pixels", 1)):
        out.append(f"\n### {label}: PSNR dB, {kind}\n")
        out.append("| method | " + " | ".join(G5) + " |")
        out.append("|---|" + "---|" * len(G5))
        for m, v in rows.items():
            out.append(f"| {m} | " + " | ".join("%.2f" % v[sp][g][ki] for g in G5) + " |")
    out.append(f"\n### {label}: gain over the per-pixel rule, dB (whole / edge)\n")
    out.append("| method | " + " | ".join(G5) + " |")
    out.append("|---|" + "---|" * len(G5))
    for m, v in rows.items():
        if m.startswith("rule (per"):
            continue
        out.append(f"| {m} | " + " | ".join("%+.2f / %+.2f" % (v[sp][g][0] - REF[sp][g][0], v[sp][g][1] - REF[sp][g][1]) for g in G5) + " |")
out.append("\n### Share of the block oracle's gain reached, held-out all pairs (whole / edge)\n")
og = rows[nm["oracle"]]["val"]["all"]
out.append("| method | share |")
out.append("|---|---|")
for m, v in rows.items():
    if m.startswith("rule (per") or "[B=16]" in m or m.endswith("-b16") or m == nm["oracle"]:
        continue
    out.append(f"| {m} | %.0f %% / %.0f %% |" % tuple(100 * (v["val"]["all"][i] - REF["val"]["all"][i]) / (og[i] - REF["val"]["all"][i]) for i in (0, 1)))
out.append("\n### Training runs\n")
out.append("| tag | steps | seed | wd | dropout | loss | batch x crop | train time s | last-100-step loss (levels^2 per px-channel) |")
out.append("|---|---|---|---|---|---|---|---|---|")
for m, v in rows.items():
    if "meta" in v:
        r = v["meta"]
        out.append(f"| {m} | {r['steps']} | {r['seed']} | {r['wd']} | {r['drop']} | {r['loss']} | {r['batch']} x {r['crop']} | {r['train_seconds']:.0f} | {r['curve'][-1][1]:.2f} (step 100: {r['curve'][0][1]:.2f}, step {r['curve'][len(r['curve'])//2][0]}: {r['curve'][len(r['curve'])//2][1]:.2f}) |")
txt = "\n".join(out)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables3.md"), "w").write("# Probe 3 (task 147): block-level nets, tables\n\nB = 8 unless marked. PSNR dB, mse pooled over pixels and RGB per group; edge mask as in `score.py`. Gains are over the per-pixel rule (31.24 / 26.75 held-out). Tables from `block3_report.py`.\n" + txt + "\n")
print(txt)
