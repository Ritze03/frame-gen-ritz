#!/usr/bin/env python3
"""Test pairs with exactly known motion, made from real 1920x1080 frames (see motion/README.md).

  mkpairs.py --out DIR [--src NAME=FILE.rgba ...] [--cases pan,roll_c,...] [--strengths s,m,l]

For a source frame S and a transform T (a one-parameter group M(a): M(a) after M(b) = M(a+b), M(0) = identity):
  prev(p) = S(M(+1/2)(p))      curr(q) = S(M(-1/2)(q))      mid = S itself (unresampled)
Content at S-position s therefore sits at M(-1/2)(s) in prev and M(+1/2)(s) in curr; the true displacement prev -> curr
is M(+1/2)(s) - M(-1/2)(s) at the middle frame's pixel s, and M(1)(p) - p at prev's pixel p.  Positions are pixel-index
coordinates (pixel i has its centre at i), the displacement is in full-resolution pixels (the convention of fgtest's F01
multiplied by flowScale: content at p in prev is at p + flow in curr).
Writes per pair: prev.rgba curr.rgba mid.rgba flow.npy (float32 [H,W,2], middle frame's grid) valid.npy (bool [H,W], middle
frame's grid), spec.json, mask.npy (still cut-out, chase / orbit only) -- and OUT/pairs.json, the list the other tools read.
Cases (each at three strengths s / m / l = largest per-pixel displacement 8 / 24 / 64 px):
  pan, roll_c, roll_o (rotation about the centre / an off-centre point), zoom_c, zoom_o, rollzoom (both, off-centre point),
  chase (zoom about a point at the horizon, still irregular cut-out at the bottom centre), orbit (the background pans sideways,
  speed changing smoothly from top to bottom; a still figure-shaped cut-out in the middle).
The library-side truth for any grid is truth(spec, ...) below (analytic, used by measure.py / maps.py); the middle-grid flow.npy
is its full-resolution middle-grid sample.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

W, H = 1920, 1080
FRAMES = "/home/mo/DieGanzeSau/AI_Video_Training_Data/blend/frames"
SOURCES = {   # one Forza (60 fps, driving), one first-person shooter, one Minecraft
    "forza": f"{FRAMES}/racing_forza6_a/000340.rgba",
    "cs2": f"{FRAMES}/fps_cs2_a/000840.rgba",
    "mc": f"{FRAMES}/minecraft120_02/000964.rgba",
}
TARGETS = {"s": 8.0, "m": 24.0, "l": 64.0}
CASES = ["pan", "roll_c", "roll_o", "zoom_c", "zoom_o", "rollzoom", "chase", "orbit"]
MARGIN = 4   # px a sample position must stay inside the source (Lanczos support) to count as valid
BAND = 3     # px around a cut-out's outline that are never valid (a pixel on the outline is both)


# ---------------------------------------------------------------- the transform family
def mapping(spec, a, X, Y):
    """M(a) of the spec at coordinates X, Y (float64 arrays) -> X', Y'."""
    m = spec["model"]
    if m == "orbit":   # shift in x that depends on y only
        yy = np.clip(Y / (spec["H"] - 1), 0, 1)
        sm = yy * yy * (3 - 2 * yy)
        return X + a * spec["vmax"] * (spec["v_top"] + (1 - spec["v_top"]) * sm), Y + 0 * Y
    cx, cy = spec["cx"], spec["cy"]
    th, s = a * spec["theta"], np.exp(a * spec["logs"])
    dx, dy = X - cx, Y - cy
    c, sn = np.cos(th), np.sin(th)
    return cx + s * (c * dx - sn * dy) + a * spec["tx"], cy + s * (sn * dx + c * dy) + a * spec["ty"]


def flow_prev(spec, X, Y):
    """true displacement prev -> curr at prev's pixel (X, Y): M(1)(p) - p"""
    x1, y1 = mapping(spec, 1.0, X, Y)
    return np.stack([x1 - X, y1 - Y], -1)


def flow_mid(spec, X, Y):
    """true displacement prev -> curr at the middle frame's pixel (X, Y)"""
    xa, ya = mapping(spec, 0.5, X, Y)
    xb, yb = mapping(spec, -0.5, X, Y)
    return np.stack([xa - xb, ya - yb], -1)


def load_mask(spec_dir):
    p = os.path.join(spec_dir, "mask.npy")
    return np.load(p, allow_pickle=False) if os.path.exists(p) else None


def truth(spec, spec_dir, fs, grid):
    """(flow, valid) of the pair on the flow grid (fw x fh, pixel m at full-res coordinate fs*m + (fs-1)/2) of the frame
    `grid` = "prev" | "mid".  flow is the full prev -> curr displacement in FULL-RESOLUTION px, valid is bool [fh, fw]."""
    w, h = spec["W"], spec["H"]
    fw, fh = int(round(w / fs)), int(round(h / fs))
    X, Y = np.meshgrid(fs * np.arange(fw) + (fs - 1) / 2, fs * np.arange(fh) + (fs - 1) / 2)
    mask = load_mask(spec_dir)
    inside = lambda x, y: (x >= MARGIN) & (x <= w - 1 - MARGIN) & (y >= MARGIN) & (y <= h - 1 - MARGIN)
    if mask is not None:   # edge: the BAND around the outline (both sides); mdil: where a cut-out hides what is behind it
        k = np.ones((2 * BAND + 1, 2 * BAND + 1), np.uint8)
        m8 = mask.astype(np.uint8)
        edge = cv2.dilate(m8, k).astype(bool) & ~cv2.erode(m8, k).astype(bool)
        mdil = cv2.dilate(m8, np.ones((2 * BAND + 3, 2 * BAND + 3), np.uint8)).astype(bool)
    samp = lambda m, x, y: m[np.clip(np.rint(y).astype(int), 0, h - 1), np.clip(np.rint(x).astype(int), 0, w - 1)]
    if grid == "prev":
        f = flow_prev(spec, X, Y)
        q = (X + f[..., 0], Y + f[..., 1])
        valid = inside(*q) & inside(*mapping(spec, 0.5, X, Y))
        if mask is not None:
            fg = samp(mask, X, Y)
            f = np.where(fg[..., None], 0.0, f)
            valid &= ~samp(edge, X, Y) & (fg | ~samp(mdil, *q))
    else:
        f = flow_mid(spec, X, Y)
        p, q = mapping(spec, -0.5, X, Y), mapping(spec, 0.5, X, Y)
        valid = inside(*p) & inside(*q)
        if mask is not None:
            fg = samp(mask, X, Y)
            f = np.where(fg[..., None], 0.0, f)
            valid &= ~samp(edge, X, Y) & (fg | ~(samp(mdil, *p) | samp(mdil, *q)))
    return f.astype(np.float32), valid


def truth_step(spec, spec_dir, fs, grid="prev"):
    """bool [fh, fw] on the flow grid of `grid`: pixels within 8 flow px of a true vector step above 0.75 px per flow px (a real
    motion edge: the cut-out's outline), which blockiness ignores"""
    f, _ = truth(spec, spec_dir, fs, grid)
    dx = np.abs(np.diff(f, axis=1)).max(-1)
    dy = np.abs(np.diff(f, axis=0)).max(-1)
    e = np.zeros(f.shape[:2], np.uint8)
    e[:, 1:] |= (dx > 0.75).astype(np.uint8)
    e[1:, :] |= (dy > 0.75).astype(np.uint8)
    return cv2.dilate(e, np.ones((17, 17), np.uint8)).astype(bool)


# ---------------------------------------------------------------- cut-outs
def blob(shape_seed, pts, jitter):
    """closed polygon through pts (list of (x, y)) with a deterministic irregular outline"""
    rng = np.random.RandomState(shape_seed)
    P = np.array(pts, np.float64)
    out = []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        n = max(2, int(np.hypot(*(b - a)) / 24))
        for k in range(n):
            t = k / n
            out.append(a + (b - a) * t + rng.randn(2) * jitter)
    return np.array(out, np.int32)


def car_mask():
    m = np.zeros((H, W), np.uint8)
    pts = [(560, 1080), (590, 900), (700, 800), (860, 760), (1060, 760), (1220, 800), (1330, 900), (1370, 1080)]
    cv2.fillPoly(m, [blob(7, pts, 5.0)], 1)
    return m.astype(bool)


def figure_mask():
    m = np.zeros((H, W), np.uint8)
    cv2.circle(m, (960, 330), 70, 1, -1)                       # head
    cv2.fillPoly(m, [blob(11, [(880, 420), (1040, 420), (1090, 470), (1120, 700), (1060, 700), (1030, 520),
                               (1030, 1000), (990, 1080), (960, 1080), (950, 760), (930, 1080), (890, 1080),
                               (880, 520), (860, 700), (800, 700), (830, 470)], 4.0)], 1)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(bool)


# ---------------------------------------------------------------- specs
def base_spec(case, W=W, H=H):
    s = dict(W=W, H=H, cx=W / 2, cy=H / 2, theta=0.0, logs=0.0, tx=0.0, ty=0.0, model="affine", mask=False, case=case)
    if case == "pan":
        s.update(tx=0.94, ty=0.34)   # direction (cos 20, sin 20), scaled to the target
    elif case == "roll_c":
        s.update(theta=1.0)
    elif case == "roll_o":
        s.update(theta=1.0, cx=0.3 * W, cy=0.65 * H)
    elif case == "zoom_c":
        s.update(logs=1.0)
    elif case == "zoom_o":
        s.update(logs=1.0, cx=0.35 * W, cy=0.4 * H)
    elif case == "rollzoom":
        s.update(theta=0.6, logs=1.0, cx=0.4 * W, cy=0.45 * H)
    elif case == "chase":
        s.update(logs=1.0, cx=0.5 * W, cy=0.36 * H, mask=True)
    elif case == "orbit":
        s.update(model="orbit", vmax=1.0, v_top=0.2, mask=True)
    return s


def max_disp(spec):
    """largest |true displacement| over the frame outside the cut-out (sampled every 8 px, plus the exact corners)"""
    X, Y = np.meshgrid(np.r_[np.arange(0, W, 8), W - 1].astype(np.float64), np.r_[np.arange(0, H, 8), H - 1].astype(np.float64))
    f = flow_mid(spec, X, Y)
    return float(np.hypot(f[..., 0], f[..., 1]).max())


def calibrate(spec, target):
    """scale the transform's strength until the largest per-pixel displacement is `target` px (a few fixed-point steps: the
    exponent of a zoom and the rotation are not exactly linear)"""
    keys = ["vmax"] if spec["model"] == "orbit" else ["theta", "logs", "tx", "ty"]
    for _ in range(6):
        m = max_disp(spec)
        for k in keys:
            spec[k] *= target / m
    spec["target"] = target
    spec["max_disp"] = max_disp(spec)
    return spec


# ---------------------------------------------------------------- rendering
def warp(src, spec, a):
    """out(p) = S(M(a)(p)) with a Lanczos filter"""
    X, Y = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    x, y = mapping(spec, a, X, Y)
    return cv2.remap(src, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)


def make_pair(S, case, strength, outdir):
    os.makedirs(outdir, exist_ok=True)
    spec = calibrate(base_spec(case), TARGETS[strength])
    mask = {"chase": car_mask, "orbit": figure_mask}.get(case, lambda: None)()
    prev, curr = warp(S, spec, 0.5), warp(S, spec, -0.5)
    if mask is not None:
        prev[mask], curr[mask] = S[mask], S[mask]
        np.save(os.path.join(outdir, "mask.npy"), mask)
    for n, im in (("prev", prev), ("curr", curr), ("mid", S)):
        np.concatenate([im, np.full((H, W, 1), 255, np.uint8)], -1).tofile(os.path.join(outdir, n + ".rgba"))
    X, Y = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    f, valid = truth(spec, outdir, 1, "mid")
    np.save(os.path.join(outdir, "flow.npy"), f)
    np.save(os.path.join(outdir, "valid.npy"), valid)
    json.dump(spec, open(os.path.join(outdir, "spec.json"), "w"), indent=1)
    return spec


def read_rgba(path, w=W, h=H):
    return np.fromfile(path, np.uint8).reshape(h, w, 4)[..., :3].copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", action="append", default=[], help="NAME=FILE.rgba (1920x1080 RGBA8); default: one Forza, one cs2, one Minecraft frame")
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--strengths", default="s,m,l")
    a = ap.parse_args()
    srcs = dict(x.split("=", 1) for x in a.src) or SOURCES
    entries = []
    for sn, path in srcs.items():
        S = read_rgba(path)
        for case in a.cases.split(","):
            for st in a.strengths.split(","):
                name = f"{sn}_{case}_{st}"
                spec = make_pair(S, case, st, os.path.join(a.out, name))
                entries.append(dict(name=name, dir=os.path.join(a.out, name), W=W, H=H, source=sn, case=case, strength=st,
                                    scale=TARGETS[st], truth=True, group="synthetic"))
                print(name, "max displacement %.2f px" % spec["max_disp"], flush=True)
    json.dump(entries, open(os.path.join(a.out, "pairs.json"), "w"), indent=1)
    print(len(entries), "pairs ->", os.path.join(a.out, "pairs.json"))


if __name__ == "__main__":
    main()
