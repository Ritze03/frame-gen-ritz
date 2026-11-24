"""Task-144 tables -> prints markdown (redirect to tables2.md).  Sources: scores.json, block_oracle/NN.json, scores2.json.  usage: report2.py [--part1|--part2]"""
import json, os, sys
import numpy as np
BLEND = os.environ.get("BLEND", "/home/mo/DieGanzeSau/AI_Video_Training_Data/blend")
ROOT = f"{BLEND}/probe"
G = ["forza", "cs2", "minecraft120", "battlefield", "all"]
S1 = json.load(open(f"{ROOT}/scores.json"))
grp = {k: v["pair"]["group"] for k, v in S1.items()}
BO = {k: json.load(open(f"{ROOT}/block_oracle/{int(k):02d}.json")) for k in S1 if os.path.exists(f"{ROOT}/block_oracle/{int(k):02d}.json")}
S2 = json.load(open(f"{ROOT}/scores2.json")) if os.path.exists(f"{ROOT}/scores2.json") else {}


def pooled(get, g):
    ks = [k for k in S1 if g == "all" or grp[k] == g]
    ks = [k for k in ks if get(k) is not None]
    return {f: sum(get(k)[f] for k in ks) for f in ("se", "n", "se_e", "n_e")}


def psnr(se, n):
    return 10 * np.log10(255.0 ** 2 / (se / (3 * n)))


def row(label, get, kind="both"):
    cells = []
    for g in G:
        a = pooled(get, g)
        cells.append("%.2f / %.2f" % (psnr(a["se"], a["n"]), psnr(a["se_e"], a["n_e"])))
    return f"| {label} | " + " | ".join(cells) + " |"


def head(title):
    return f"\n**{title}** (whole picture / edge pixels, PSNR dB)\n\n| method | " + " | ".join(G) + " |\n|---|" + "---|" * len(G)


s1 = lambda m: (lambda k: S1[k]["scores"].get(m))
bo = lambda v: (lambda k: BO[k][v] if k in BO else None)
s2 = lambda m: (lambda k: S2.get(k, {}).get(m))
BNAME = {32: "block 32", 16: "block 16", 8: "block 8", 4: "block 4", 2: "block 2", 0: "whole frame (1 weight vector)"}


def reduction_share():
    out = ["\n**Where the gain over the rule sits (squared-error reduction vs the rule, all 23 pairs pooled)**\n",
           "| variant | total reduction (% of rule's squared error) | share at edge pixels | share elsewhere | edge pixels' own reduction (% of rule's se_e) | non-edge pixels' own reduction |", "|---|---|---|---|---|---|"]
    r = pooled(s1("rule"), "all")
    for v in ("6_8", "6_4", "6_2", "4_8", "4_4"):
        a = pooled(bo(v), "all")
        tot, te = r["se"] - a["se"], r["se_e"] - a["se_e"]
        no_r, no_a = r["se"] - r["se_e"], a["se"] - a["se_e"]
        out.append(f"| {v.split('_')[0]} candidates, B={v.split('_')[1]} | {100 * tot / r['se']:.1f} % | {100 * te / tot:.1f} % | {100 * (tot - te) / tot:.1f} % | {100 * te / r['se_e']:.1f} % | {100 * (no_r - no_a) / no_r:.1f} % |")
    a = pooled(s1("O6"), "all")
    tot, te = r["se"] - a["se"], r["se_e"] - a["se_e"]
    out.append(f"| O6 per pixel | {100 * tot / r['se']:.1f} % | {100 * te / tot:.1f} % | {100 * (tot - te) / tot:.1f} % | {100 * te / r['se_e']:.1f} % | {100 * (r['se'] - r['se_e'] - a['se'] + a['se_e']) / (r['se'] - r['se_e']):.1f} % |")
    out.append("\nEdge pixels are %.1f %% of all pixels." % (100 * r["n_e"] / r["n"]))
    return out


def part1():
    o = ["# Task 144 tables: block-constant ceiling, O6-target nets, seed noise", ""]
    o.append(head("Ceiling by block size, six candidates (4 taps + prev + curr)"))
    for m, lab in (("rule", "rule"), ("baseC", "base C"), ("forzaC", "forza-rx9070xt C"), ("px8", "px8")):
        o.append(row(lab, s1(m)))
    for B in (32, 16, 8, 4, 2, 0):
        o.append(row(BNAME[B], bo(f"6_{B}")))
    o.append(row("per-pixel O6", s1("O6")))
    o.append(head("Ceiling by block size, four taps only"))
    o.append(row("rule", s1("rule")))
    for B in (32, 16, 8, 4, 2, 0):
        o.append(row(BNAME[B], bo(f"4_{B}")))
    o.append(row("per-pixel O4", s1("O4")))
    o.append("\n**Checkerboard split**: the block mix is fitted on the even pixels ((y+x) even) of each block and scored on the odd pixels only; 'plain' = fitted on all pixels, scored on the same odd pixels; rule on the same odd pixels for reference.")
    for K in (6, 4):
        o.append(head(f"Checkerboard, {K} candidates, scored on odd pixels"))
        o.append(row("rule (odd pixels)", bo("rule_odd")))
        for B in (16, 8, 4):
            o.append(row(f"block {B} plain fit", bo(f"{K}_{B}_odd")))
            o.append(row(f"block {B} fitted on even pixels", bo(f"{K}_{B}_cb")))
    o += reduction_share()
    return o


def part2():
    o = [head("Nets against the real frame on the 23 held-out pairs")]
    for m, lab in (("rule", "rule"), ("px8", "px8 (task 143, random crops)"), ("n3-all", "n3-all (task 143)"), ("n5-all", "n5-all (task 143)")):
        o.append(row(lab, s1(m)))
    for m in ("px8-fix", "n3-all-fix", "px8-o6", "n3-all-o6", "n5-all-o6", "px8-s1"):
        if any(m in v for v in S2.values()):
            o.append(row(m, s2(m)))
    o.append(row("block 8 (6 cand.)", bo("6_8")))
    o.append(row("per-pixel O6", s1("O6")))
    return o


if __name__ == "__main__":
    out = []
    if "--part2" not in sys.argv:
        out += part1()
    if "--part1" not in sys.argv:
        out += ["", "## Nets (parts 2 and 3)"] + part2()
    print("\n".join(out))
