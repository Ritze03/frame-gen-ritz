#!/usr/bin/env python3
"""GPU (fgtest) vs Python reference (proto/interp.py) on real frame pairs, plus the benchmark runs.

  compare.py check [--bgra]   ~10 pairs (f[i], f[i+2]) from battlefield.mp4 and obs2.mp4, PSNR table
  compare.py bench [N]        fgtest --bench N at 1920x1080 and 2560x1440 (upscaled frame)
Work files go to $FG_WORK (default: /tmp/fgwork). fgtest is gpu/build/fgtest unless $FGTEST is set.
"""
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "proto"))
import interp  # noqa: E402  (frozen reference)

FOOT = os.path.join(HERE, "..", "footage")
FGTEST = os.environ.get("FGTEST", os.path.join(HERE, "build", "fgtest"))
WORK = os.environ.get("FG_WORK", "/tmp/fgwork")


def mad(x, y):
    return float(np.abs(x.astype(np.int16) - y.astype(np.int16)).mean())


def triples(clip, starts, need_motion):
    """(name, a, b, c) = f[i], f[i+1], f[i+2]; with need_motion, slide forward until both consecutive
    mean abs diffs > 0.6 (skips the re-encoded duplicate frames in the OBS clips)."""
    cap = cv2.VideoCapture(os.path.join(FOOT, clip + ".mp4"))
    out = []
    for s in starts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        fr = [cap.read()[1] for _ in range(3)]
        i = s
        while need_motion and not (mad(fr[0], fr[1]) > 0.6 and mad(fr[1], fr[2]) > 0.6):
            ok, f = cap.read()
            if not ok:
                break
            fr = fr[1:] + [f]
            i += 1
        out.append((f"{clip}@{i}", *fr))
    return out


def write_rgba(path, bgr):
    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA).tofile(path)


def read_rgba(path, h, w):
    return cv2.cvtColor(np.fromfile(path, np.uint8).reshape(h, w, 4), cv2.COLOR_RGBA2BGR)


def check(bgra):
    os.makedirs(WORK, exist_ok=True)
    sets = triples("battlefield", [300, 900, 1500, 2100, 2700], False)
    sets += triples("obs2", [3000, 9000, 15000, 22000, 30000], True)
    # global-fallback cases (no real middle frame): a scene cut, and a large-stride pair (fast motion)
    cap = cv2.VideoCapture(os.path.join(FOOT, "battlefield.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1200)
    fr = [cap.read()[1] for _ in range(25)]
    sets.append(("cut bf/obs2", sets[0][1], None, sets[5][3]))
    sets.append(("bf@1200 stride24", fr[0], None, fr[24]))
    h, w = sets[0][1].shape[:2]
    args = [FGTEST] + (["--bgra"] if bgra else []) + [str(w), str(h)]
    for k, (name, a, b, c) in enumerate(sets):
        write_rgba(f"{WORK}/a{k}.rgba", a)
        write_rgba(f"{WORK}/c{k}.rgba", c)
        args += [f"{WORK}/a{k}.rgba", f"{WORK}/c{k}.rgba", f"{WORK}/o{k}.rgba"]
    subprocess.run(args, check=True)
    print(f"{'pair':<18} {'GPU-vs-Py':>9} {'Py-vs-real':>10} {'GPU-vs-real':>11} {'delta':>6}  output==a (py/gpu)")
    rows = []
    for k, (name, a, b, c) in enumerate(sets):
        g = read_rgba(f"{WORK}/o{k}.rgba", h, w)
        p = interp.interpolate(a, c)
        cv2.imwrite(f"{WORK}/py{k}.png", p)
        cv2.imwrite(f"{WORK}/gpu{k}.png", g)
        d = np.abs(g.astype(np.int16) - p.astype(np.int16)).max(2)
        cv2.imwrite(f"{WORK}/diff{k}.png", np.clip(d * 8, 0, 255).astype(np.uint8))
        held = ("a" if np.array_equal(p, a) else "-") + "/" + ("a" if np.array_equal(g, a) else "-")
        if b is None:
            print(f"{name:<18} {cv2.PSNR(g, p):9.2f} {'':>10} {'':>11} {'':>6}  {held}")
            continue
        gp, pr, gr = cv2.PSNR(g, p), cv2.PSNR(p, b), cv2.PSNR(g, b)
        rows.append((gp, pr, gr))
        print(f"{name:<18} {gp:9.2f} {pr:10.2f} {gr:11.2f} {gr - pr:+6.2f}  {held}")
    r = np.array(rows)
    print(f"{'mean':<18} {r[:, 0].mean():9.2f} {r[:, 1].mean():10.2f} {r[:, 2].mean():11.2f} {(r[:, 2] - r[:, 1]).mean():+6.2f}")


def bench(n):
    os.makedirs(WORK, exist_ok=True)
    (_, a, b, c), = triples("battlefield", [900], False)
    for w, h in ((1920, 1080), (2560, 1440)):
        aa = cv2.resize(a, (w, h), interpolation=cv2.INTER_CUBIC)
        cc = cv2.resize(c, (w, h), interpolation=cv2.INTER_CUBIC)
        write_rgba(f"{WORK}/ba.rgba", aa)
        write_rgba(f"{WORK}/bc.rgba", cc)
        subprocess.run([FGTEST, "--bench", str(n), str(w), str(h), f"{WORK}/ba.rgba", f"{WORK}/bc.rgba",
                        f"{WORK}/bo.rgba"], check=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        bench(int(sys.argv[2]) if len(sys.argv) > 2 else 100)
    else:
        check("--bgra" in sys.argv)
