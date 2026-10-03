#!/usr/bin/env python3
"""GPU (fgtest) vs Python reference (proto/interp.py) on real frame pairs, plus the benchmark runs.

  compare.py check [--bgra] [--split]   ~10 pairs (f[i], f[i+2]) from battlefield.mp4 and obs2.mp4, PSNR table
                                      (--split: recordEstimate + recordSynth(0.5) instead of record)
  compare.py multi [--bgra]   pairs (f[i], f[i+N]), N = 3, 4, GPU (one estimate + N-1 synths) vs interp.synth at
                              t = k/N, scored against the real in-between frames; plus the fallback cases
  compare.py bench [N]        fgtest --bench N --mult M at 1920x1080 and 2560x1440 (upscaled frame), M = 2, 3, 4
Work files go to $FG_WORK (default: /tmp/fgwork). fgtest is gpu/build/fgtest unless $FGTEST is set.
Footage is ../footage unless $FG_FOOTAGE is set.
"""
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "proto"))
import interp  # noqa: E402  (frozen reference)

FOOT = os.environ.get("FG_FOOTAGE", os.path.join(HERE, "..", "footage"))
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


def check(bgra, split=False):
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
    args = [FGTEST] + (["--bgra"] if bgra else []) + (["--split"] if split else []) + [str(w), str(h)]
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


def windows(clip, starts, n):
    """(name, [f[i] .. f[i+n]]) with every consecutive mean abs diff > 0.6 (clean motion, no duplicates)."""
    cap = cv2.VideoCapture(os.path.join(FOOT, clip + ".mp4"))
    out = []
    for s in starts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        fr = [cap.read()[1] for _ in range(n + 1)]
        i = s
        while not all(mad(x, y) > 0.6 for x, y in zip(fr, fr[1:])):
            ok, f = cap.read()
            if not ok:
                break
            fr = fr[1:] + [f]
            i += 1
        out.append((f"{clip}@{i}", fr))
    return out


def multi(bgra):
    os.makedirs(WORK, exist_ok=True)
    flag = ["--bgra"] if bgra else []
    stats = {}
    for n in (3, 4):
        ts = [k / n for k in range(1, n)]
        sets = windows("battlefield", [300, 900, 1500, 2100], n) + windows("obs2", [3000, 9000, 15000, 22000], n)
        h, w = sets[0][1][0].shape[:2]
        args = [FGTEST] + flag + ["--t", ",".join(f"{t:.9f}" for t in ts), str(w), str(h)]
        for k, (name, fr) in enumerate(sets):
            write_rgba(f"{WORK}/ma{n}_{k}.rgba", fr[0])
            write_rgba(f"{WORK}/mc{n}_{k}.rgba", fr[n])
            args += [f"{WORK}/ma{n}_{k}.rgba", f"{WORK}/mc{n}_{k}.rgba", f"{WORK}/mo{n}_{k}.rgba"]
        subprocess.run(args, check=True)
        print(f"{n}x {'pair':<16} {'t':>5} {'GPU-vs-Py':>9} {'Py-vs-real':>10} {'GPU-vs-real':>11} {'delta':>6}  py-why / gpu==py-fallback")
        for k, (name, fr) in enumerate(sets):
            a, c = fr[0], fr[n]
            est = interp.estimate(a, c)
            for j, t in enumerate(ts):
                g = read_rgba(f"{WORK}/mo{n}_{k}.rgba.{j}", h, w)
                p, why = interp.synth(a, c, est, t, why=True)
                b = fr[j + 1]
                gp, pr, gr = cv2.PSNR(g, p), cv2.PSNR(p, b), cv2.PSNR(g, b)
                stats.setdefault(round(t, 4), []).append((gp, pr, gr))
                print(f"   {name:<16} {t:5.3f} {gp:9.2f} {pr:10.2f} {gr:11.2f} {gr - pr:+6.3f}  {why or '-'} / {np.array_equal(g, p) if why else ''}")
    print(f"per t (mean over pairs): {'t':>6} {'n':>3} {'GPU-vs-Py':>9} {'Py-vs-real':>10} {'GPU-vs-real':>11} {'delta':>7} {'min GvP':>8}")
    for t in sorted(stats):
        r = np.array(stats[t])
        print(f"{'':<24} {t:6.4f} {len(r):3d} {r[:, 0].mean():9.2f} {r[:, 1].mean():10.3f} {r[:, 2].mean():11.3f} "
              f"{(r[:, 2] - r[:, 1]).mean():+7.4f} {r[:, 0].min():8.2f}")
    # global fallbacks at t != 0.5: a scene cut and a large-stride pair; output must be exactly the nearer frame
    cut = windows("battlefield", [300], 1)[0][1][0], windows("obs2", [3000], 1)[0][1][1]
    cap = cv2.VideoCapture(os.path.join(FOOT, "battlefield.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1200)
    fr = [cap.read()[1] for _ in range(25)]
    cases = [("cut bf/obs2", *cut), ("cut inverted", cut[0], 255 - cut[0]), ("bf@1200 stride24", fr[0], fr[24]),
             ("bf@1200 stride16", fr[0], fr[16])]
    ts = [0.25, 1 / 3, 0.5, 2 / 3, 0.75]
    h, w = cases[0][1].shape[:2]
    args = [FGTEST] + flag + ["--t", ",".join(f"{t:.9f}" for t in ts), str(w), str(h)]
    for k, (name, a, c) in enumerate(cases):
        write_rgba(f"{WORK}/fa{k}.rgba", a)
        write_rgba(f"{WORK}/fc{k}.rgba", c)
        args += [f"{WORK}/fa{k}.rgba", f"{WORK}/fc{k}.rgba", f"{WORK}/fo{k}.rgba"]
    subprocess.run(args, check=True)
    print(f"fallback cases: {'pair':<18} {'t':>5} py-why  gpu==a gpu==c gpu==py  GPU-vs-Py")
    for k, (name, a, c) in enumerate(cases):
        est = interp.estimate(a, c)
        for j, t in enumerate(ts):
            g = read_rgba(f"{WORK}/fo{k}.rgba.{j}", h, w)
            p, why = interp.synth(a, c, est, t, why=True)
            print(f"{'':<16}{name:<18} {t:5.3f} {why or '-':<7} {str(np.array_equal(g, a)):<6} {str(np.array_equal(g, c)):<6} "
                  f"{str(np.array_equal(g, p)):<7} {cv2.PSNR(g, p):9.2f}")


def bench(n):
    os.makedirs(WORK, exist_ok=True)
    (_, a, b, c), = triples("battlefield", [900], False)
    for w, h in ((1920, 1080), (2560, 1440)):
        aa = cv2.resize(a, (w, h), interpolation=cv2.INTER_CUBIC)
        cc = cv2.resize(c, (w, h), interpolation=cv2.INTER_CUBIC)
        write_rgba(f"{WORK}/ba.rgba", aa)
        write_rgba(f"{WORK}/bc.rgba", cc)
        for mult in (2, 3, 4):
            subprocess.run([FGTEST, "--bench", str(n), "--mult", str(mult), str(w), str(h), f"{WORK}/ba.rgba",
                            f"{WORK}/bc.rgba", f"{WORK}/bo.rgba"], check=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        bench(int(sys.argv[2]) if len(sys.argv) > 2 else 100)
    elif len(sys.argv) > 1 and sys.argv[1] == "multi":
        multi("--bgra" in sys.argv)
    else:
        check("--bgra" in sys.argv, "--split" in sys.argv)
