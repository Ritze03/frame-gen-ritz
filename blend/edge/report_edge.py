"""Markdown tables from $BLEND/edge/scores/NAME.json.  usage: report_edge.py NAME[=label] ...   (--json: also print the 'all' figures as json)"""
import json, sys
import numpy as np
from common import WORK

GROUPS = ["forza", "cs2", "minecraft120", "battlefield", "all"]
names = [a for a in sys.argv[1:] if not a.startswith("--")]
S = {}
for a in names:
    n, _, lab = a.partition("=")
    S[lab or n] = json.load(open(f"{WORK}/scores/{n}.json"))


def agg(res, g):
    ks = [v for v in res.values() if g == "all" or v["group"] == g]
    return {f: sum(v[f] for v in ks) for f in ks[0] if f != "group"}


def psnr(se, n):
    return 10 * np.log10(255.0 ** 2 / (se / (3 * n)))


KINDS = [("PSNR dB, whole picture", lambda a: psnr(a["se"], a["n"]), "%.2f"),
         ("PSNR dB, edge pixels", lambda a: psnr(a["se_e"], a["n_e"]), "%.2f"),
         ("spurious edge energy, whole picture (mean max(0, G_out - G_real), levels)", lambda a: a["sp"] / a["n"], "%.3f"),
         ("missing edge energy, whole picture (mean max(0, G_real - G_out))", lambda a: a["ms"] / a["n"], "%.3f"),
         ("spurious edge energy inside the edge mask", lambda a: a["sp_e"] / a["n_e"], "%.3f"),
         ("missing edge energy inside the edge mask", lambda a: a["ms_e"] / a["n_e"], "%.3f"),
         ("mean weight on prev + curr at edge pixels", lambda a: a["un_e"] / a["n_e"], "%.4f")]
for title, f, fmt in KINDS:
    print(f"\n{title}\n\n| method | " + " | ".join(GROUPS) + " |\n|---|" + "---|" * len(GROUPS))
    for lab, res in S.items():
        print(f"| {lab} | " + " | ".join(fmt % f(agg(res, g)) for g in GROUPS) + " |")
if "--json" in sys.argv:
    print(json.dumps({lab: {t: {g: f(agg(res, g)) for g in GROUPS} for t, f, _ in KINDS} for lab, res in S.items()}))
