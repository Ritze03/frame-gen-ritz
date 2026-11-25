"""Shared code of motion/maps.py and motion/measure.py: pair lists, running fgtest, reading its dumps, field metrics, drawing.
Not a tool of its own.  See motion/README.md for the conventions (every field here is in FULL-RESOLUTION px of displacement
prev -> curr, 'flow px' only where said)."""
import json
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gpu"))
sys.path.insert(0, HERE)
import fgmaps   # noqa: E402  gpu/fgmaps.py (reader of .fgmaps)
import v3flow   # noqa: E402  gpu/v3flow.py (reader of .v3db)
import mkpairs  # noqa: E402

LOCK = "/home/mo/tmp/fg-handoff/gpu.lock"
PRESSURE = "/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/claude.slice/memory.pressure"
TMP = "/home/mo/tmp"


# ---------------------------------------------------------------- pair lists
def load_pairs(path, only=None):
    """pairs.json -> list of dicts: name, W, H, prev, curr, mid (or None), dir (synthetic: spec.json / flow.npy / mask.npy), scale (px,
    the fixed magnitude scale of the colour wheel), truth (bool), group.  `only`: comma list of substrings, any must match the name."""
    base = os.path.dirname(os.path.abspath(path))
    out = []
    for e in json.load(open(path)):
        e = dict(e)
        d = e.get("dir")
        if d:
            e["prev"], e["curr"], e["mid"] = (os.path.join(d, n + ".rgba") for n in ("prev", "curr", "mid"))
        e.setdefault("mid", None)
        e.setdefault("truth", False)
        e.setdefault("scale", 32.0)
        e.setdefault("group", "real")
        e.setdefault("W", 1920)
        e.setdefault("H", 1080)
        if only and not any(s in e["name"] for s in only.split(",")):
            continue
        out.append(e)
    return out


def read_rgba(path, w, h):
    return np.fromfile(path, np.uint8).reshape(h, w, 4)[..., :3]


def wait_pressure(limit=10.0):
    while True:
        try:
            v = float(open(PRESSURE).readline().split("avg10=")[1].split()[0])
        except Exception:
            return
        if v <= limit:
            return
        print("memory pressure %.1f > %.0f, waiting" % (v, limit), flush=True)
        time.sleep(15)


# ---------------------------------------------------------------- running fgtest
def run_fgtest(fgtest, setstr, pairs, work, chunk=4, dump=True, keep_out=True):
    """Runs `flock LOCK ccorral -- fgtest --window 1 [--dump D] [--set S] W H prev curr out ...` on chunks of pairs and yields, per
    pair, dict(pair=entry, fg=fgmaps.load dict, v3=v3flow.load dict | None, out=uint8 [H, W, 3] | None).  The files of a chunk
    are deleted when the generator moves past it.  --window 1: a fresh Interpolator per pair (no state between pairs)."""
    os.makedirs(work, exist_ok=True)
    for i in range(0, len(pairs), chunk):
        part = pairs[i:i + chunk]
        wait_pressure()
        d = os.path.join(work, "chunk")
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        cmd = ["flock", LOCK, "ccorral", "--", fgtest, "--window", "1"]
        if dump:
            cmd += ["--dump", os.path.join(d, "dump")]
        if setstr:
            cmd += ["--set", setstr]
        # fgtest takes one W H per call: group by size
        by = {}
        for k, p in enumerate(part):
            by.setdefault((p["W"], p["H"]), []).append(k)
        res = {}
        for (w, h), ks in by.items():
            c = list(cmd)
            if dump:
                c[c.index("--dump") + 1] = os.path.join(d, "dump%dx%d" % (w, h))
            c += [str(w), str(h)]
            for k in ks:
                c += [part[k]["prev"], part[k]["curr"], os.path.join(d, "out%d.rgba" % k)]
            r = subprocess.run(c, capture_output=True, text=True, env=dict(os.environ, TMPDIR=TMP))
            if r.returncode != 0:
                raise RuntimeError("fgtest failed (%d): %s\n%s" % (r.returncode, " ".join(c), (r.stdout + r.stderr)[-2000:]))
            for j, k in enumerate(ks):
                res[k] = os.path.join(d, "dump%dx%d" % (w, h), "pair_%04d" % j)
        for k, p in enumerate(part):
            pre = res[k]
            r = dict(pair=p, fg=fgmaps.load(pre + ".fgmaps") if dump else None, v3=None, out=None)
            if dump and os.path.exists(pre + ".v3db"):
                r["v3"] = v3flow.load(pre + ".v3db")
            if keep_out:
                r["out"] = read_rgba(os.path.join(d, "out%d.rgba" % k), p["W"], p["H"]).copy()
            yield r
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- estimated fields
def estimator(setstr):
    """the motion version named by a --set string: v1 | v2 | learned | v3 (default of fgtest: see its parseSettings; v3 unless named)"""
    for tok in (setstr or "").split(","):
        if tok.startswith("algo="):
            return tok[5:]
    return "v3"


def fields(r, setstr):
    """Fields of one run, each as dict(F=[fh, fw, 2] full-res px prev -> curr, grid='prev'|'mid', fs=flowScale, note).
    Stages in the dict (V3: BLK = level-0 block vectors held constant per 8x8 flow px, FD = dense field, FF = F01 after occlusion
    / fill, the field the synth projects; V1 / V2 / Learned: what their dump holds, see README)."""
    fg, v3 = r["fg"], r["v3"]
    fs = fg["fscale"]
    out = {}
    if v3 is not None:
        fw, fh = v3["fw"], v3["fh"]
        blk = v3["blk"][0][0, :, :, 0:2]   # dir 0 (F01), slot 0 = the final level-0 vectors, flow px
        up = np.repeat(np.repeat(blk, 8, 0), 8, 1)[:fh, :fw]
        out["BLK"] = dict(F=up * fs, grid="prev", fs=fs, block=8)
        out["FD"] = dict(F=v3["fd0"] * fs, grid="prev", fs=fs)
        out["FF"] = dict(F=v3["ff0"] * fs, grid="prev", fs=fs)
        out["main"] = "FF"
        for lv in range(1, v3["levels"]):   # the search's own result of level lv (block = 4 px of the level = 4 * 2^lv flow px), nearest-filled
            blv = v3["blk"][lv][1 if lv < v3["levels"] - 1 else 0][:, :, 0:2]   # dir 0, final slot (iteration 1; the top level has one)
            s = 4 * 2 ** lv
            out["L%d" % lv] = dict(F=np.repeat(np.repeat(blv, s, 0), s, 1)[:fh, :fw] * (2 ** lv) * fs, grid="prev", fs=fs, block=s)
    else:
        algo = estimator(setstr)
        pf, block = fg["pf"], fg["block"]   # half vectors on the midpoint grid, flow px
        fw, fh = fg["fw"], fg["fh"]
        if np.abs(pf).max() > 0:
            out["PF"] = dict(F=2 * pf * fs, grid="mid", fs=fs)
            out["main"] = "PF"
        if np.abs(block).max() > 0:
            bs = fg["W"] // fg["bw"] // fs if fg["bw"] else 8
            up = np.repeat(np.repeat(block, bs, 0), bs, 1)[:fh, :fw]
            out["BLK"] = dict(F=2 * up * fs, grid="mid", fs=fs, block=bs)
            out.setdefault("main", "BLK")
    return out


# ---------------------------------------------------------------- metrics
def epe(F, ftrue, valid):
    e = np.hypot(F[..., 0] - ftrue[..., 0], F[..., 1] - ftrue[..., 1])
    return e, valid & np.isfinite(e)


def blockiness(F, B, ok=None):
    """How much of the field's *curvature* sits on one grid of period B (flow px), rescaled so that a smooth field is 0 and a field that is
    constant per block (or bilinear between block centres: kinks on the grid) is 1.  F [h, w, 2] (px); ok [h, w] bool: pixels that may count
    (a position counts when its 3 pixels do).  q(x) = |F(x+1) - 2 F(x) + F(x-1)| (vector norm, px) along x and along y; Q[j] = mean of q over
    the positions with coordinate = j (mod B), the two axes summed; the grid phase is free: s = max_j (Q[j] + Q[j+1]) / sum_j Q[j]
    (2/B when the curvature is spread evenly, as for noise or a smooth field with texture; 1 when it sits on one grid line, a step
    gives |q| = step at two neighbouring positions); blk = (s - 2/B) / (1 - 2/B) clipped at 0.
    Returns (blk, step_px, phase): step_px = Q[phase] - median(Q) (the extra curvature at the grid line, in px: a pure step of s px
    gives about s), phase = the position in 0..B-1 where the lines are (V3 at flowScale 2: 0 or 7 = block boundary, 3 or 4 = block centre)."""
    h, w, _ = F.shape
    Q = np.zeros(B)
    for ax in (1, 0):
        q = np.hypot(*np.abs(np.diff(F, 2, axis=ax)).transpose(2, 0, 1))     # position x+1 of each triple
        m = np.isfinite(q)
        if ok is not None:
            m &= (ok[:, 2:] & ok[:, 1:-1] & ok[:, :-2]) if ax == 1 else (ok[2:] & ok[1:-1] & ok[:-2])
        qs = np.where(m, q, 0.0).sum(0 if ax == 1 else 1)   # sum over the other axis -> per position
        ns = m.sum(0 if ax == 1 else 1)
        j = (np.arange(len(qs)) + 1) % B
        sm, cn = np.bincount(j, qs, B), np.bincount(j, ns, B)
        Q += np.where(cn > 0, sm / np.maximum(cn, 1), 0.0)
    Q = Q + 0.1   # a floor of 0.05 px of curvature per axis and position (below it the field is smooth to the eye; keeps a smooth field's tiny residues from reading as structure)
    pair = Q + np.roll(Q, -1)
    j = int(pair.argmax())
    s = float(pair[j] / Q.sum())
    ph = j if Q[j] >= Q[(j + 1) % B] else (j + 1) % B
    return max(0.0, (s - 2 / B) / (1 - 2 / B)), float(Q[ph] - np.median(Q)) / 2, ph


def block_constant(F, B):
    """F with every aligned BxB block replaced by its mean (the 'per-block constant version')"""
    h, w, _ = F.shape
    ph, pw = -h % B, -w % B
    P = np.pad(F, ((0, ph), (0, pw), (0, 0)), mode="edge")
    m = P.reshape(P.shape[0] // B, B, P.shape[1] // B, B, 2).mean((1, 3))
    return np.repeat(np.repeat(m, B, 0), B, 1)[:h, :w]


def gradmag(F):
    """magnitude of the Jacobian of F (px of vector change per flow px; central differences)"""
    gy, gx = np.gradient(F, axis=(0, 1))
    return np.sqrt((gx ** 2).sum(-1) + (gy ** 2).sum(-1))


def luma(rgb):
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def edge_mask(real):
    """as blend/probe/score.py / blend/edge/score_edge.py: Sobel magnitude of the luma of the real frame (replicate border),
    pixels whose 5x5 neighbourhood holds a value >= the frame's 90th percentile"""
    L = luma(real.astype(np.float32))
    G = np.hypot(cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE),
                 cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE))
    thr = np.percentile(G.ravel(), 90)
    return cv2.dilate((G >= thr).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


def se_counts(out, real):
    """(sum of squared error over all pixels and RGB, pixel count, same on edge pixels, edge pixel count); PSNR = 10 log10(255^2 / (se / (3 n)))"""
    d = (out.astype(np.float64) - real.astype(np.float64)) ** 2
    se = d.sum(-1)
    em = edge_mask(real)
    return float(se.sum()), int(se.size), float(se[em].sum()), int(em.sum())


def psnr(se, n):
    return 10 * np.log10(255.0 ** 2 / max(se / (3 * n), 1e-12))


# ---------------------------------------------------------------- drawing
def hsv_wheel_rgb(F, scale):
    """direction -> hue, magnitude -> saturation (white = no motion), value 1; F [h, w, 2]; scale px = full saturation. RGB uint8"""
    mag = np.hypot(F[..., 0], F[..., 1])
    hue = ((np.arctan2(F[..., 1], F[..., 0]) / (2 * np.pi)) % 1.0) * 179
    hsv = np.stack([hue, np.clip(mag / scale, 0, 1) * 255, np.full_like(mag, 255)], -1).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def wheel_legend(size, scale):
    """a small colour wheel (radius = scale px) on a dark disc with a label"""
    n = size
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    c = (n - 1) / 2
    v = np.stack([(xx - c) / c * scale, (yy - c) / c * scale], -1)
    rgb = hsv_wheel_rgb(v, scale)
    inside = (np.hypot(xx - c, yy - c) <= c)[..., None]
    img = np.where(inside, rgb, 40).astype(np.uint8)
    cv2.circle(img, (n // 2, n // 2), n // 2 - 1, (90, 90, 90), 1, cv2.LINE_AA)
    return img


def put(img, text, org, scale=0.6, color=(255, 255, 255), thick=1, bg=(0, 0, 0)):
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + bl + 2), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def motion_panel(F, scale, size, label, legend=True):
    """F [fh, fw, 2] (any resolution; vector in px) -> RGB panel of `size` (w, h) with the wheel in the bottom right corner"""
    rgb = cv2.resize(hsv_wheel_rgb(F, scale), size, interpolation=cv2.INTER_AREA)
    if legend:
        n = size[1] // 4
        lg = wheel_legend(n, scale)
        rgb[size[1] - n - 8:size[1] - 8, size[0] - n - 8:size[0] - 8] = lg
        put(rgb, "%g px" % scale, (size[0] - n - 8, size[1] - n - 14), 0.5)
    put(rgb, label, (8, 22), 0.6)
    return rgb


def steps_panel(F, size, label, gs=2.0):
    """magnitude of the field's local gradient, white-on-black, grey = clip(g / gs)^0.6 (g in px of vector change per flow px)"""
    g = np.clip(gradmag(F) / gs, 0, 1) ** 0.6
    rgb = cv2.cvtColor((g * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    put(rgb, label, (8, 22), 0.6)
    return rgb


def heat_panel(e, valid, vmax, size, label):
    """end-point error 0..vmax px, inferno; invalid pixels dark grey"""
    g = (np.clip(e / vmax, 0, 1) * 255).astype(np.uint8)
    rgb = cv2.cvtColor(cv2.applyColorMap(g, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    rgb[~valid] = (45, 45, 60)
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    put(rgb, label, (8, 22), 0.6)
    return rgb


def image_panel(img, size, label):
    rgb = cv2.resize(np.ascontiguousarray(img), size, interpolation=cv2.INTER_AREA)
    put(rgb, label, (8, 22), 0.6)
    return rgb


# ---------------------------------------------------------------- one pair's numbers
BLOCKS_FULL = (8, 16, 32)   # block sizes of the blockiness measure, in full-resolution px; 16 = V3's level-0 block at flowScale 2


def pair_metrics(r, setstr):
    """All numbers of one run (a run_fgtest result): per stage field: EPE mean / p95 / bad share (synthetic pairs), blockiness at
    BLOCKS_FULL (grid-aligned, and 'ctrl' = the grid shifted by half a block), step_px; PSNR whole / edge (and sums)
    against the true middle frame where there is one.  Also the truth's own blockiness (smooth field, per-block-constant field)."""
    p = r["pair"]
    spec = json.load(open(os.path.join(p["dir"], "spec.json"))) if p.get("truth") else None
    m = dict(name=p["name"], group=p.get("group"), case=p.get("case"), strength=p.get("strength"), fields={})
    fl = fields(r, setstr)
    m["main"] = fl.get("main")
    for st in [k for k in fl if k != "main"]:
        f = fl[st]
        fs, F = f["fs"], f["F"]
        d = {}
        ok = None
        if spec is not None:
            ft, valid = mkpairs.truth(spec, p["dir"], fs, f["grid"])
            e, okv = epe(F, ft, valid)
            d["epe_mean"], d["epe_p95"] = float(e[okv].mean()), float(np.percentile(e[okv], 95))
            d["bad4"] = float((e[okv] > 4).mean())
            src = read_rgba(p["prev"] if f["grid"] == "prev" else p["mid"], p["W"], p["H"])   # textured pixels: the 5x5-dilated top decile of the source's Sobel magnitude (edge_mask)
            tx = edge_mask(src)[(fs * np.arange(F.shape[0]) + (fs - 1) // 2)[:, None], (fs * np.arange(F.shape[1]) + (fs - 1) // 2)[None, :]] & okv
            d["epe_tex_mean"], d["epe_tex_p95"] = float(e[tx].mean()), float(np.percentile(e[tx], 95))
            ok = valid & ~mkpairs.truth_step(spec, p["dir"], fs, f["grid"])
        for bf in BLOCKS_FULL:
            B = max(bf // fs, 4)
            b, step, ph = blockiness(F, B, ok)
            d["blk%d" % bf], d["step%d" % bf], d["phase%d" % bf] = b, step, ph
            d["ctrl%d" % bf] = blockiness(F, B + 3, ok)[0]   # a period that is no harmonic of B: should be near 0
        m["fields"][st] = d
    if spec is not None and "main" in fl:   # the truth's own blockiness (once per pair, on the main field's grid)
        f = fl[fl["main"]]
        fs = f["fs"]
        ft, valid = mkpairs.truth(spec, p["dir"], fs, f["grid"])
        ok = valid & ~mkpairs.truth_step(spec, p["dir"], fs, f["grid"])
        for bf in BLOCKS_FULL:
            B = max(bf // fs, 4)
            m["truth_blk%d" % bf] = blockiness(ft, B, ok)[0]
            m["truthbc_blk%d" % bf] = blockiness(block_constant(ft, B), B, ok)[0]
    if p.get("mid") and r["out"] is not None:
        real = read_rgba(p["mid"], p["W"], p["H"])
        se, n, see, ne = se_counts(r["out"], real)
        m.update(se=se, n=n, se_e=see, n_e=ne, psnr=psnr(se, n), psnr_edge=psnr(see, ne))
    return m
