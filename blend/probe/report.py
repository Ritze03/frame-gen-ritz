"""Markdown tables from $BLEND/probe/scores.json: scores, gains over px8 / px32, share of the px8 -> O6 gap closed, share of n3-all's gain.  usage: report.py > tables.md"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import tables
from nets import NAMES

meths, groups, out = tables(["rule", "baseC", "forzaC"] + NAMES + ["O4", "O6"])
nb = [m for m in ("n3-all", "n5-all", "n3-frames", "n3-taps") if m in meths]


def tab(title, f, ms, fmt="%.2f"):
    print(f"\n**{title}**\n")
    print("| method | " + " | ".join(groups) + " |")
    print("|---|" + "---|" * len(groups))
    for m in ms:
        print(f"| {m} | " + " | ".join(fmt % f(m, g) if f(m, g) is not None else "-" for g in groups) + " |")


for kind, label in (("whole", "PSNR dB, whole picture"), ("edge", "PSNR dB, edge pixels only (90th-percentile Sobel mask, 5x5 dilated)")):
    tab(label, lambda m, g, k=kind: out[(m, g)][k], meths)
tab("share of pixels with an error above 30 levels in any channel, whole picture (%)", lambda m, g: out[(m, g)]["big"], meths)
tab("share of edge pixels with an error above 30 levels in any channel (%)", lambda m, g: out[(m, g)]["big_e"], meths)
for base in ("px8", "px32"):
    for kind in ("whole", "edge"):
        tab(f"gain over {base} in dB, {kind}", lambda m, g, b=base, k=kind: out[(m, g)][k] - out[(b, g)][k], nb, "%+.2f")
for kind in ("whole", "edge"):
    tab(f"share of the px8 -> O6 gap (dB) each net closes, {kind} (%)",
        lambda m, g, k=kind: 100 * (out[(m, g)][k] - out[("px8", g)][k]) / (out[("O6", g)][k] - out[("px8", g)][k]), ["px32"] + nb, "%.1f")
for kind in ("whole", "edge"):
    tab(f"gain of n3-frames / n3-taps as a share of n3-all's gain over px8, {kind} (%; meaningless when n3-all's gain is near zero)",
        lambda m, g, k=kind: 100 * (out[(m, g)][k] - out[("px8", g)][k]) / (out[("n3-all", g)][k] - out[("px8", g)][k]), [m for m in ("n3-frames", "n3-taps") if m in meths], "%.0f")
