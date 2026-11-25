#!/usr/bin/env python3
"""Calibration of the census search chosen by `study_match.py` (task 162): at level 2, an exhaustive +-12 level px search of a census cost over the 8 x 8 window (V3's window:
block of 4 x 4 plus a 2 px apron) on every patch of the frame-wide truth, with the census variants of the shader (neighbours at +-2 level px, eight of them; `bit = n - c > eps`,
or the ternary form with a second bit `c - n > eps`).  Per patch it writes whether the cost minimum is within 1 level px of the truth, the best cost, the best cost at least
3 px away from it (the second minimum), the cost at zero and the mean over the area: the distinctness the shader turns into a confidence.

    python study_census.py --list pairs_study.json --truth TRUTHDIR --out census.pkl [--only a,b]"""
import argparse
import json
import os
import pickle

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import study_match as S

R = 12
LV = 2
WIN = 8


def census(y, eps, ternary):
    p = np.pad(y, 2, mode='edge')
    H, W = y.shape
    bits = []
    for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (-2, 2), (2, -2), (2, 2)):
        d = p[2 + dy:2 + dy + H, 2 + dx:2 + dx + W] - y
        bits.append((d > eps).astype(np.float32))
        if ternary:
            bits.append((d < -eps).astype(np.float32))
    return np.stack(bits, -1)


VAR = {'c0': (0.0, False), 'c05': (0.5, False), 'c1': (1.0, False), 't05': (0.5, True), 't1': (1.0, True), 't2': (2.0, True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', required=True)
    ap.add_argument('--truth', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--only')
    a = ap.parse_args()
    out = {}
    for e in json.load(open(a.list)):
        n = e['name']
        if a.only and not any(s in n for s in a.only.split(',')):
            continue
        t = np.load(os.path.join(a.truth, n + '.npy'))
        if len(t) == 0:
            continue
        A = np.fromfile(e['prev'], np.uint8).reshape(1080, 1920, 4)[..., :3]
        B = np.fromfile(e['curr'], np.uint8).reshape(1080, 1920, 4)[..., :3]
        ra, rb = S.pyramid(S.planes(A)[0]), S.pyramid(S.planes(B)[0])
        ya, yb = ra[LV][..., 0], rb[LV][..., 0]
        H, W = ya.shape
        rows = []
        cen = {k: (census(ya, *v), census(yb, *v)) for k, v in VAR.items()}
        s = 2.0 ** (LV + 1)
        for pi, (x, y, tx, ty) in enumerate(t):
            vt = np.array([tx, ty]) / s
            vr = np.rint(vt).astype(int)
            if np.abs(vr).max() >= R - 1:
                continue
            cx, cy = x / s - 0.5, y / s - 0.5
            ox = int(np.clip(round(cx - (WIN - 1) / 2), 0, W - WIN))
            oy = int(np.clip(round(cy - (WIN - 1) / 2), 0, H - WIN))
            vx = np.arange(-R, R + 1)
            ok = ((oy + vx >= 0) & (oy + vx <= H - WIN))[:, None] & ((ox + vx >= 0) & (ox + vx <= W - WIN))[None, :]
            if not ok[vr[1] + R, vr[0] + R]:
                continue
            for k, (ca, cb) in cen.items():
                pad = np.pad(cb, ((R, R), (R, R), (0, 0)), mode='edge')
                reg = pad[oy:oy + WIN + 2 * R, ox:ox + WIN + 2 * R]
                Wb = sliding_window_view(reg, (WIN, WIN), axis=(0, 1))
                Wa = np.moveaxis(ca[oy:oy + WIN, ox:ox + WIN], -1, 0)
                c = np.abs(Wa[None, None] - Wb).mean((2, 3, 4))
                c = np.where(ok, c, np.inf)
                pen = c + 1e-5 * (np.abs(vx)[:, None] + np.abs(vx)[None, :])
                j, i = np.unravel_index(np.argmin(pen), pen.shape)
                fin = max(abs(i - R - vr[0]), abs(j - R - vr[1])) <= 1
                yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
                far = (np.maximum(abs(xx - (i - R)), abs(yy - (j - R))) >= 3) & ok
                c2 = c[far].min() if far.any() else np.inf
                rows.append((pi, k, bool(fin), float(c[j, i]), float(c2), float(c[R, R]), float(c[ok].mean()), (i - R, j - R), float(np.hypot(*vt)), float(c[vr[1] + R, vr[0] + R])))
        out[n] = dict(truth=t, rows=rows)
        print(n, len(rows), flush=True)
    pickle.dump(out, open(a.out, 'wb'))


if __name__ == '__main__':
    main()
