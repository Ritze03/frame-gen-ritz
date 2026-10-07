#!/usr/bin/env python3
"""Numeric frame-gen bench: scores contestants (a motion version + settings) against v1 on every clip in footage/.

  taskset -c 6-9,16-19 python3 bench/score/score.py [-c 'name: set-string' ...] [--contestants FILE] [--triples N] ...

See bench/score/README.md. Metric definitions are written into every report (DEFINITIONS below).
"""
import argparse
import csv
import datetime
import glob
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

CPUS = {6, 7, 8, 9, 16, 17, 18, 19}
CPUSTR = "6-9,16-19"
if not os.sched_getaffinity(0) <= CPUS:   # a subset is fine (a cgroup cpuset may narrow it further)
    if os.environ.get("FG_BENCH_REEXEC"):
        sys.exit(f"refusing to run: CPU affinity {sorted(os.sched_getaffinity(0))} is not within {CPUSTR}")
    os.environ["FG_BENCH_REEXEC"] = "1"
    os.execvp("taskset", ["taskset", "-c", CPUSTR, sys.executable] + sys.argv)
os.environ["OMP_NUM_THREADS"] = "8"

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(8)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
GPU = os.path.join(REPO, "gpu")
FGTEST = os.path.join(GPU, "build", "fgtest")
GENRES = ("fps", "racing", "minecraft", "indie", "realworld")
REF = "v1"
FORCED = "globalFallback=1"      # the global fallback off, as the user plays; its would-fire rate is still measured
DEF_GLOBAL_FRAC, DEF_CUT_SAD = 0.15, 30.0   # library defaults: the "would have fired" thresholds
DUP_BMAX = 1.0                   # footage report: max 16x16-block mean abs diff at 640 px wide, gray
HARM_MARGIN = 24                 # proto/harness.py
TIE_DB = 0.01
DISK_CAP = 10e9

DEFINITIONS = f"""\
- **Paired deltas**: per triple, metric(contestant) - metric({REF}) on the same held-out frame. PSNR over RGB (cv2.PSNR),
  SSIM on luma (Gaussian 11x11, sigma 1.5, Wang et al. constants). Win / loss / tie by dPSNR with a tie band of {TIE_DB} dB.
  `worse>0.5dB` = share of triples with dPSNR < -0.5 dB.
- **harm%** (proto/harness.py, proto/NOTES.md): share of pixels whose mean-channel abs error vs the real frame exceeds
  max(error of prev, error of next) + {HARM_MARGIN} levels, i.e. invented content that is far from *both* real frames.
  Mean per clip; `dharm` is the paired delta. Reported, not a gate.
- **flicker (with ground truth)**: for consecutive kept triples k, k+1 of one window, mean over pixels of
  |(g_k - r_k) - (g_k+1 - r_k+1)| on luma (8-bit levels). An error that stays put (a steady smear) scores 0; an error
  that pops in and out between generated frames scores high. Attributed to triple k+1.
- **flicker NR (no reference, 60 fps clips)**: per generated frame g between real frames a, c (60 -> 120):
  mean over pixels of |g - a| + |c - g| - |c - a| on luma (>= 0 per pixel). It is the temporal change the generated
  frame adds on top of the real frames' own change: 0 for any pixel whose value lies between a and c (a blend, a
  monotonic edge), positive where g pops away from both. Correct motion of texture is not 0 either, so read it as a
  delta against {REF}.
- **missing%**: detail in the real frame that the generated one lacks. Sobel gradient magnitude G on luma;
  mask = 7x7 mean abs luma error > 12 (the trust ramp's start); missing% = 100 * sum_mask max(0, G_real - G_gen) /
  sum_all G_real. Lost edges inside wrong regions count; added edges (ghosts) do not (harm% covers those).
- **fallback%**: share of scored pairs where the global fallback would have fired with the default threshold
  (untrusted pixels > {DEF_GLOBAL_FRAC:.0%} of the frame; untrusted = trust ramp past its midpoint), read from the
  library's own counters (Interpolator::recordDebugCounters) while every contestant runs with `{FORCED}` (off).
  **cut%**: the scene cut (finest-level mean block SAD > {DEF_CUT_SAD:g}); NOT disabled (it is the separate
  `sceneCutSad` setting), so a cut pair shows the nearer real frame in every contestant.
- **identical**: triples whose output bytes equal {REF}'s exactly.
- **Dup drop**: a triple is dropped when its held-out frame duplicates a neighbour (bmax < {DUP_BMAX} at 640 px wide,
  INTER_AREA on the Y plane: the footage report's rule). Dropped pairs still run (history stays consecutive).
"""


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def build_and_hash():
    env = dict(os.environ, CCACHE_DISABLE="1")
    if not os.path.isdir(os.path.join(GPU, "build")):
        subprocess.run(["meson", "setup", "build"], cwd=GPU, env=env, check=True)
    subprocess.run(["ninja", "-C", "build", "-j8", "fgtest"], cwd=GPU, env=env, check=True, stdout=subprocess.DEVNULL)
    head = sh(["git", "-C", REPO, "rev-parse", "HEAD"]).strip()
    dirty = sh(["git", "-C", REPO, "status", "--porcelain"]).rstrip()
    h = head + ("-dirty" if dirty else "")
    with open(FGTEST, "rb") as f:
        bh = hashlib.sha256(f.read()).hexdigest()[:16]
    print(f"fgtest built from {h} (binary sha256 {bh})", flush=True)
    return h, bh, dirty


def default_footage():
    try:
        common = sh(["git", "-C", REPO, "rev-parse", "--path-format=absolute", "--git-common-dir"]).strip()
        f = os.path.join(os.path.dirname(common), "footage")
        if os.path.isdir(f):
            return f
    except subprocess.CalledProcessError:
        pass
    return os.path.join(REPO, "footage")


def probe(path):
    o = json.loads(sh(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                       "stream=width,height,r_frame_rate,nb_frames", "-of", "json", path]))["streams"][0]
    n, d = map(int, o["r_frame_rate"].split("/"))
    return o["width"], o["height"], n / d, int(o.get("nb_frames", 0) or 0)


def genre_of(name):
    g = name.split("_")[0]
    return g if g in GENRES else "other"


def discover(footage, test60, only=None):
    clips, skipped = [], []
    paths = sorted(glob.glob(os.path.join(footage, "*.mp4"))) + sorted(glob.glob(os.path.join(footage, "cuts", "*.mp4")))
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        if only and name not in only:
            continue
        try:
            w, h, fps, n = probe(p)
        except (subprocess.CalledProcessError, KeyError, IndexError, ValueError):
            skipped.append(f"{name} (unreadable: still being written?)")
            continue
        if n <= 0:
            skipped.append(f"{name} (no frame count in the header)")
            continue
        top = os.path.dirname(p) == os.path.normpath(footage)
        genre = "fps" if top else genre_of(name)
        if fps < 50:
            skipped.append(f"{name} ({fps:g} fps < 50)")
            continue
        clips.append(dict(name=name, path=p, w=w, h=h, fps=fps, n=n, step=1, genre=genre, ref=fps >= 90))
        if test60 and name == "battlefield":   # self-test stand-in for a 60 fps cut: battlefield's even frames
            clips.append(dict(name="fps_battlefield_even60", path=p, w=w, h=h, fps=fps / 2, n=n // 2, step=2,
                              genre="fps", ref=False))
    return clips, skipped


def decode(clip, start, count):
    """Frames start..start+count-1 of the clip (in its own frame numbering): list of (rgb, y)."""
    step, w, h = clip["step"], clip["w"], clip["h"]
    srcfps = clip["fps"] * step
    t = (start * step - 0.25) / srcfps
    p = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-threads", "8", "-ss", f"{max(t, 0):.6f}", "-i", clip["path"],
                        "-frames:v", str(count * step), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                       check=True, capture_output=True).stdout
    fs = w * h * 3 // 2
    out = []
    for i in range(0, count * step, step):
        b = np.frombuffer(p[i * fs:(i + 1) * fs], np.uint8)
        if b.size < fs:
            raise RuntimeError(f"{clip['name']}: short decode at frame {start + i // step}")
        yuv = b.reshape(h * 3 // 2, w)
        out.append((cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420), yuv[:h]))
    return out


def bmax(ya, yb):
    W = 640
    H = int(round(ya.shape[0] * W / ya.shape[1] / 2)) * 2
    a = cv2.resize(ya, (W, H), interpolation=cv2.INTER_AREA).astype(np.int16)
    b = cv2.resize(yb, (W, H), interpolation=cv2.INTER_AREA).astype(np.int16)
    hb, wb = H // 16, W // 16
    return float(np.abs(a - b)[:hb * 16, :wb * 16].reshape(hb, 16, wb, 16).mean(axis=(1, 3)).max())


def luma(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


def ssim(x, y):
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    g = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)
    mx, my = g(x), g(y)
    sxx, syy, sxy = g(x * x) - mx * mx, g(y * y) - my * my, g(x * y) - mx * my
    return float((((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx * mx + my * my + C1) * (sxx + syy + C2))).mean())


def grad(yf):
    return cv2.magnitude(cv2.Sobel(yf, cv2.CV_32F, 1, 0), cv2.Sobel(yf, cv2.CV_32F, 0, 1))


def err3(x, y):
    return np.abs(x.astype(np.int16) - y).mean(2)


def site(errmap):
    """Centre (x, y) and value of the worst ~64x64 region of a per-pixel error map."""
    s = cv2.resize(errmap, (errmap.shape[1] // 8, errmap.shape[0] // 8), interpolation=cv2.INTER_AREA)
    s = cv2.blur(s, (8, 8))
    yy, xx = np.unravel_index(int(np.argmax(s)), s.shape)
    return int(xx * 8 + 4), int(yy * 8 + 4), float(s[yy, xx])


def crop(img, cx, cy, n=256):
    h, w = img.shape[:2]
    x0, y0 = min(max(cx - n // 2, 0), w - n), min(max(cy - n // 2, 0), h - n)
    return img[y0:y0 + n, x0:x0 + n].copy()


def strip(panels, title):
    out = []
    for lab, im in panels:
        bar = np.full((22, im.shape[1], 3), 32, np.uint8)
        cv2.putText(bar, lab, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        out.append(np.vstack([bar, cv2.cvtColor(im, cv2.COLOR_RGB2BGR)]))
        out.append(np.full((out[-1].shape[0], 4, 3), 255, np.uint8))
    body = np.hstack(out[:-1])
    top = np.full((24, body.shape[1], 3), 0, np.uint8)
    cv2.putText(top, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([top, body])


class Worst:
    """Top-N sites, at most one per window (scene diversity), with their crop strips kept in memory."""
    def __init__(self, n):
        self.n, self.best = n, {}

    def offer(self, key, score, make):
        if key not in self.best or score > self.best[key][0]:
            self.best[key] = (score, make())
            if len(self.best) > 4 * self.n:   # bound memory: drop the weakest
                del self.best[min(self.best, key=lambda k: self.best[k][0])]

    def items(self):
        return sorted(self.best.values(), key=lambda x: -x[0])[: self.n]


def du(path):
    tot = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return tot


def parse_contestants(args):
    lines = []
    if args.contestants:
        with open(args.contestants) as f:
            lines += f.read().splitlines()
    lines += args.c or []
    out = {}
    for ln in lines:
        ln = ln.split("#")[0].strip()
        if not ln:
            continue
        name, _, s = ln.partition(":")
        name, s = name.strip(), s.strip()
        if not re.fullmatch(r"[A-Za-z0-9_.+-]+", name):
            sys.exit(f"bad contestant name {name!r}")
        out[name] = s
    if not out:
        out = {"v1": "algo=v1", "v2": "algo=v2"}
    if REF not in out:
        out = {REF: "algo=v1", **out}
    if out[REF].replace(" ", "") != "algo=v1":
        sys.exit(f"the reference contestant {REF} must be exactly 'algo=v1', got {out[REF]!r}")
    return {REF: out[REF], **{k: v for k, v in out.items() if k != REF}}


def run_fgtest(clip, setstr, triples, window, work, state):
    """triples: [(prev_path, curr_path, out_path)]. Returns per-pair (untrusted_frac, mean_sad)."""
    cmd = [FGTEST, "--stats", "--window", str(window), "--set", setstr + "," + FORCED, str(clip["w"]), str(clip["h"])]
    for t in triples:
        cmd += list(t)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"fgtest failed ({r.returncode}): {r.stderr.strip()}")
    stats = {}
    for ln in r.stdout.splitlines():
        if ln.startswith("STATS "):
            i, un, sad, blocks, px = map(int, ln.split()[1:])
            stats[i] = (un / px, sad / 256.0 / max(blocks, 1))
    if len(stats) != len(triples):
        raise RuntimeError(f"fgtest printed {len(stats)} STATS lines for {len(triples)} pairs")
    state["peak"] = max(state["peak"], du(work))
    if state["peak"] > DISK_CAP:
        raise RuntimeError(f"work dir over {DISK_CAP / 1e9:g} GB")
    return [stats[i] for i in range(len(triples))]


def read_rgb(path, clip):
    return np.fromfile(path, np.uint8).reshape(clip["h"], clip["w"], 4)[:, :, :3]


def score_clip(clip, args, cons, work, state, rows, crops):
    ref = clip["ref"]
    P, warm = args.pairs, args.warm
    Wp = P + warm
    span = (2 * Wp if ref else Wp) + 1     # frames per window
    target = args.triples if ref else args.nr_pairs
    N = clip["n"]
    if N < span + 2:
        print(f"  {clip['name']}: too short, skipped", flush=True)
        return
    rng = np.random.default_rng([args.seed, int(hashlib.md5(clip["name"].encode()).hexdigest()[:8], 16)])
    u0 = float(rng.random())
    taken, starts = [], []
    phi = (math.sqrt(5) - 1) / 2

    def next_window():
        # golden-ratio sequence over the clip: every prefix is spread evenly; no overlapping windows
        for _ in range(1000):
            i = len(starts) + next_window.tries
            next_window.tries += 1
            s = int(((u0 + i * phi) % 1.0) * (N - span - 1))
            if all(abs(s - t) >= span for t in taken):
                taken.append(s)
                return s
        return None
    next_window.tries = 0

    max_windows = max(1, 3 * math.ceil(target / P))
    per_batch = max(1, args.batch // Wp)
    kept = dropped = 0
    widx = 0
    while kept < target and widx < max_windows:
        batch = []
        need = min(per_batch, max(1, math.ceil((target - kept) / P)))
        while len(batch) < need and widx < max_windows:
            s = next_window()
            if s is None:
                break
            batch.append((widx, s))
            widx += 1
        if not batch:
            break
        # decode + write inputs
        bdir = os.path.join(work, "batch")
        os.makedirs(bdir, exist_ok=True)
        pairs = []   # dicts per pair, in fgtest order
        frames = {}
        for wi, s in batch:
            fr = decode(clip, s, span)
            frames[wi] = fr
            stride = 2 if ref else 1
            paths = {}
            for j in range(0, span, stride):
                pth = os.path.join(bdir, f"w{wi}_f{j}.rgba")
                np.dstack([fr[j][0], np.full(fr[j][0].shape[:2], 255, np.uint8)]).tofile(pth)
                paths[j] = pth
            for p in range(Wp):
                a, c = p * stride, (p + 1) * stride
                d = dict(w=wi, s=s, p=p, a=a, c=c, prev=paths[a], curr=paths[c], warm=p < warm)
                if ref:
                    b = a + 1
                    d["b"] = b
                    d["dup"] = min(bmax(fr[a][1], fr[b][1]), bmax(fr[b][1], fr[c][1])) < DUP_BMAX
                else:
                    d["dup"] = bmax(fr[a][1], fr[c][1]) < DUP_BMAX
                pairs.append(d)
        # reference first, then each contestant (its outputs deleted after scoring)
        refmet = {}
        for name, setstr in cons.items():
            trip = [(d["prev"], d["curr"], os.path.join(bdir, f"{name}_w{d['w']}_p{d['p']}.out")) for d in pairs]
            st = run_fgtest(clip, setstr, trip, Wp, work, state)
            prevErr = {}   # (w) -> (k, signed luma error) of the previous kept triple, for flicker
            for d, (_, _, op), (unf, msad) in zip(pairs, trip, st):
                if d["warm"]:
                    continue
                fr = frames[d["w"]]
                g = read_rgb(op, clip)
                a, c = fr[d["a"]][0], fr[d["c"]][0]
                row = dict(clip=clip["name"], genre=clip["genre"], kind="ref" if ref else "noref", window=d["w"],
                           pair=d["p"] - warm, frame=(d["s"] + d["a"]) * clip["step"], contestant=name, dup=int(d["dup"]),
                           untrusted=round(unf, 5), mean_sad=round(msad, 3), fb=int(unf > DEF_GLOBAL_FRAC),
                           cut=int(msad > DEF_CUT_SAD))
                key = (d["w"], d["p"])
                if name == REF:
                    refmet[key] = {"fb": row["fb"]}
                    refmet[key]["out"] = op
                else:
                    with open(op, "rb") as f1, open(refmet[key]["out"], "rb") as f2:
                        row["identical"] = int(f1.read() == f2.read())
                if d["dup"]:
                    rows.append(row)
                    prevErr.pop(d["w"], None)
                    continue
                gy = luma(g)
                if ref:
                    r = fr[d["b"]][0]
                    ry = luma(r)
                    m = dict(psnr=cv2.PSNR(np.ascontiguousarray(g), r), ssim=ssim(gy, ry),
                             harm=float((err3(g, r) > np.maximum(err3(a, r), err3(c, r)) + HARM_MARGIN).mean() * 100))
                    e = gy - ry
                    mask = cv2.blur(np.abs(e), (7, 7)) > 12
                    gr = grad(ry)
                    m["missing"] = float(100 * np.maximum(gr - grad(gy), 0)[mask].sum() / max(gr.sum(), 1e-6))
                    pe = prevErr.get(d["w"])
                    m["flicker"] = float(np.abs(e - pe[1]).mean()) if pe and pe[0] == d["p"] - 1 else float("nan")
                    prevErr[d["w"]] = (d["p"], e)
                    absE = np.abs(e)
                else:
                    ay, cy = luma(a), luma(c)
                    m = dict(flicker_nr=float((np.abs(gy - ay) + np.abs(cy - gy) - np.abs(cy - ay)).mean()))
                row.update({k: round(v, 5) for k, v in m.items()})
                if name == REF:
                    refmet[key].update(m)
                else:
                    for k, v in m.items():
                        row["d_" + k] = round(v - refmet[key][k], 5)
                rows.append(row)
                if ref:   # worst-spot crops
                    title = f"{clip['name']} f{(d['s'] + d['b'])} window {d['w']}"
                    gv1 = g if name == REF else read_rgb(refmet[key]["out"], clip)
                    def mk(cx, cy, what, val, g=g, gv1=gv1, d=d, fr=fr, name=name):
                        pan = [(f"prev f{d['s'] + d['a']}", crop(fr[d["a"]][0], cx, cy)), (REF, crop(gv1, cx, cy))]
                        if name != REF:
                            pan.append((name, crop(g, cx, cy)))
                        pan += [(f"real f{d['s'] + d['b']}", crop(fr[d["b"]][0], cx, cy)),
                                (f"next f{d['s'] + d['c']}", crop(fr[d["c"]][0], cx, cy))]
                        return dict(img=strip(pan, f"{title}  site ({cx},{cy})  {what} {val:.2f}"),
                                    clip=clip["name"], frame=d["s"] + d["b"], x=cx, y=cy, what=what, val=val)
                    cx, cy, v = site(absE)
                    crops[(name, "error")].offer((clip["name"], d["w"]), v, lambda: mk(cx, cy, "local luma error", v))
                    if name != REF:
                        cx2, cy2, v2 = site(absE - np.abs(luma(gv1) - ry))
                        if v2 > 0:
                            crops[(name, "delta")].offer((clip["name"], d["w"]), v2, lambda: mk(cx2, cy2, f"local error minus {REF}", v2))
            if name != REF:
                for d, (_, _, op) in zip(pairs, trip):
                    os.remove(op)
        shutil.rmtree(bdir)   # inputs and the reference's outputs
        nb = [d for d in pairs if not d["warm"]]
        kept += sum(not d["dup"] for d in nb)
        dropped += sum(d["dup"] for d in nb)
        print(f"  {clip['name']}: {widx} windows, {kept} kept / {dropped} dup-dropped", flush=True)
    clip["kept"], clip["dropped"], clip["windows"] = kept, dropped, widx


# ---------------------------------------------------------------- report

def q(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


def f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{nd}f}" if nd < 0 else f"{x:.{nd}f}"


def sgn(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{nd}f}"


def col(rows, k):
    return np.array([r[k] for r in rows if k in r and not (isinstance(r[k], float) and math.isnan(r[k]))], float)


def ref_table(groups, cons):
    out = ["| group | contestant | n | PSNR | dPSNR mean | median | p5 | win/loss/tie | worse>0.5dB | dSSIM mean | median | p5 "
           "| harm% | dharm | flicker | dflicker | missing% | dmissing | fallback% | cut% | identical |",
           "|" + "---|" * 21]
    for g, rs in groups:
        for name in cons:
            r = [x for x in rs if x["contestant"] == name]
            sc = [x for x in r if not x["dup"]]
            fb = col(r, "fb")
            cut = col(r, "cut")
            if name == REF:
                out.append(f"| {g} | {name} (ref) | {len(sc)} | {f(col(sc, 'psnr').mean() if sc else None, 2)} | | | | | | | | "
                           f"| {f(col(sc, 'harm').mean() if sc else None)} | | {f(col(sc, 'flicker').mean() if len(col(sc, 'flicker')) else None)} | "
                           f"| {f(col(sc, 'missing').mean() if sc else None)} | | {f(100 * fb.mean() if len(fb) else None, 1)} "
                           f"| {f(100 * cut.mean() if len(cut) else None, 1)} | |")
                continue
            dp, ds = col(sc, "d_psnr"), col(sc, "d_ssim")
            win = int((dp > TIE_DB).sum()); loss = int((dp < -TIE_DB).sum())
            ident = col(r, "identical")
            fl, dfl = col(sc, "flicker"), col(sc, "d_flicker")
            out.append(
                f"| {g} | {name} | {len(sc)} | {f(col(sc, 'psnr').mean() if sc else None, 2)} | {sgn(dp.mean() if len(dp) else None)} "
                f"| {sgn(np.median(dp) if len(dp) else None)} | {sgn(q(dp, 5))} | {win}/{loss}/{len(dp) - win - loss} "
                f"| {f(100 * (dp < -0.5).mean() if len(dp) else None, 1)}% | {sgn(ds.mean() if len(ds) else None, 5)} "
                f"| {sgn(np.median(ds) if len(ds) else None, 5)} | {sgn(q(ds, 5), 5)} | {f(col(sc, 'harm').mean() if sc else None)} "
                f"| {sgn(col(sc, 'd_harm').mean() if sc else None)} | {f(fl.mean() if len(fl) else None)} "
                f"| {sgn(dfl.mean() if len(dfl) else None)} | {f(col(sc, 'missing').mean() if sc else None)} "
                f"| {sgn(col(sc, 'd_missing').mean() if sc else None)} | {f(100 * fb.mean() if len(fb) else None, 1)} "
                f"| {f(100 * cut.mean() if len(cut) else None, 1)} | {int(ident.sum())}/{len(ident)} |")
    return out


def nr_table(groups, cons):
    out = ["| group | contestant | pairs | scored | flicker NR | dflicker NR mean | median | p95 | fallback% | cut% | identical |",
           "|" + "---|" * 11]
    for g, rs in groups:
        for name in cons:
            r = [x for x in rs if x["contestant"] == name]
            sc = [x for x in r if not x["dup"]]
            fl, dfl = col(sc, "flicker_nr"), col(sc, "d_flicker_nr")
            fb, cut, ident = col(r, "fb"), col(r, "cut"), col(r, "identical")
            out.append(f"| {g} | {name}{' (ref)' if name == REF else ''} | {len(r)} | {len(sc)} | {f(fl.mean() if len(fl) else None)} "
                       f"| {sgn(dfl.mean()) if len(dfl) else ''} | {sgn(np.median(dfl)) if len(dfl) else ''} "
                       f"| {sgn(q(dfl, 95)) if len(dfl) else ''} | {f(100 * fb.mean() if len(fb) else None, 1)} "
                       f"| {f(100 * cut.mean() if len(cut) else None, 1)} | {f'{int(ident.sum())}/{len(ident)}' if len(ident) else ''} |")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", action="append", metavar="'NAME: SET'", help="contestant (repeatable), e.g. 'v2: algo=v2'")
    ap.add_argument("--contestants", metavar="FILE", help="file of 'name: set-string' lines (# comments)")
    ap.add_argument("--triples", type=int, default=200, help="scored (non-dup) triples per >=90 fps clip (default 200)")
    ap.add_argument("--nr-pairs", type=int, default=None, help="scored pairs per 60 fps clip (default = --triples)")
    ap.add_argument("--pairs", type=int, default=10, help="scored consecutive pairs per window (default 10)")
    ap.add_argument("--warm", type=int, default=2, help="unscored warm-up pairs at each window start (default 2)")
    ap.add_argument("--batch", type=int, default=100, help="pairs per fgtest call (default 100)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--crops", type=int, default=8, help="worst-spot crops per contestant and ranking")
    ap.add_argument("--footage", default=None, help="footage dir (default: the main checkout's footage/)")
    ap.add_argument("--clips", default=None, help="comma list of clip names to keep (default: all)")
    ap.add_argument("--test60", action="store_true", help="also score battlefield's even frames as a 60 fps no-reference clip")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    args.nr_pairs = args.nr_pairs or args.triples
    t0 = time.time()
    cons = parse_contestants(args)
    fghash, binhash, dirty = build_and_hash()
    footage = args.footage or default_footage()
    only = set(args.clips.split(",")) if args.clips else None
    clips, skipped = discover(footage, args.test60, only)
    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + fghash[:8]
    out = os.path.join(HERE, "out", run_id)
    work = os.path.join(HERE, "work", run_id)
    os.makedirs(os.path.join(out, "crops"), exist_ok=True)
    os.makedirs(work, exist_ok=True)
    print(f"run {run_id}: {len(clips)} clips, contestants {cons}, pid {os.getpid()}", flush=True)
    state = {"peak": 0}
    rows = []
    crops = {(n, k): Worst(args.crops) for n in cons for k in ("error", "delta")}
    try:
        for clip in clips:
            print(f"{clip['name']} ({clip['genre']}, {clip['fps']:g} fps, {'reference' if clip['ref'] else 'no-reference'})", flush=True)
            score_clip(clip, args, cons, work, state, rows, crops)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        try:
            os.rmdir(os.path.dirname(work))
        except OSError:
            pass

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(out, "per_triple.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    crop_lines = []
    for (name, kind), wst in crops.items():
        for i, (_, it) in enumerate(wst.items()):
            fn = f"{name}_{kind}_{i + 1:02d}_{it['clip']}_f{it['frame']}.png"
            cv2.imwrite(os.path.join(out, "crops", fn), it["img"])
            crop_lines.append(f"| {name} | {kind} | {i + 1} | {it['clip']} | {it['frame']} | ({it['x']},{it['y']}) | {it['val']:.2f} | crops/{fn} |")

    refc = [c for c in clips if c["ref"] and "kept" in c]
    nrc = [c for c in clips if not c["ref"] and "kept" in c]
    L = [f"# frame-gen bench: {run_id}", "",
         f"- **fgtest build**: `{fghash}` (fgtest binary sha256 {binhash}), rebuilt with CCACHE_DISABLE=1 at run start"
         + (f"\n- dirty tree:\n```\n{dirty}\n```" if dirty else ""),
         f"- command: `{' '.join(sys.argv)}`",
         f"- seed {args.seed}; windows of {args.warm} warm-up + {args.pairs} scored consecutive pairs, a fresh Interpolator "
         f"per window (fgtest --window); <= {args.batch} pairs per fgtest call; target {args.triples} kept triples per "
         f"reference clip, {args.nr_pairs} pairs per no-reference clip",
         f"- every contestant runs with `{FORCED}` appended (global fallback off)",
         f"- footage: `{footage}`", f"- runtime {time.time() - t0:.0f} s; peak work dir {state['peak'] / 1e9:.2f} GB", "",
         "## Contestants", "", "| name | settings |", "|---|---|"]
    L += [f"| {n}{' (reference)' if n == REF else ''} | `{s},{FORCED}` |" for n, s in cons.items()]
    L += ["", "## Clips", "", "| clip | genre | fps | kind | windows | triples kept | dup-dropped |", "|---|---|---|---|---|---|---|"]
    L += [f"| {c['name']} | {c['genre']} | {c['fps']:g} | {'reference (held-out middle frame)' if c['ref'] else 'no-reference 60->120'} "
          f"| {c['windows']} | {c['kept']} | {c['dropped']} |" for c in refc + nrc]
    if skipped:
        L += ["", "Skipped: " + ", ".join(skipped)]
    rr = [r for r in rows if r["kind"] == "ref"]
    nr = [r for r in rows if r["kind"] == "noref"]
    if rr:
        groups = [(c["name"], [r for r in rr if r["clip"] == c["name"]]) for c in refc]
        groups += [(f"**genre {g}**", [r for r in rr if r["genre"] == g]) for g in sorted({c["genre"] for c in refc})]
        groups += [("**all**", rr)]
        L += ["", "## Reference clips: paired deltas against v1", "",
              "PSNR = absolute mean (context only). d* = contestant - v1 per triple. flicker / missing / harm: lower is better; "
              "dPSNR / dSSIM: higher is better. fallback% / cut% / identical: over all scored pairs, dup-dropped ones included.", ""]
        L += ref_table(groups, cons)
    if nr:
        groups = [(c["name"], [r for r in nr if r["clip"] == c["name"]]) for c in nrc]
        groups += [(f"**genre {g}**", [r for r in nr if r["genre"] == g]) for g in sorted({c["genre"] for c in nrc})]
        groups += [("**all**", nr)]
        L += ["", "## No-reference clips (60 fps)", "", "flicker NR: lower is better; d = contestant - v1. "
              "pairs = all scored pairs, scored = without dup pairs.", ""]
        L += nr_table(groups, cons)
    L += ["", "## Worst-spot crops", "", "Strips are prev | v1 | contestant | real | next, 256x256 at full resolution. "
          "`error`: worst local (~64 px) luma error of that contestant; `delta`: worst local error minus v1's "
          "(only sites where the contestant is worse). At most one site per window.", "",
          "| contestant | ranking | # | clip | real frame | site | value | file |", "|---|---|---|---|---|---|---|---|"]
    L += crop_lines or ["| (none) | | | | | | | |"]
    L += ["", "## Definitions", "", DEFINITIONS]
    with open(os.path.join(out, "report.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"report: {os.path.join(out, 'report.md')}", flush=True)


if __name__ == "__main__":
    main()
