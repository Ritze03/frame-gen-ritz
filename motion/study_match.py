#!/usr/bin/env python3
"""Offline study (task 162): which matching makes the true vector the cost minimum on real driving footage?

For every pair of the study list (a JSON list with prev / mid / curr paths: the `road` set, the held-out pairs, the tp_m and the first-person scene pairs) and every patch of the
frame-wide sparse truth (`common.frame_truth`: 64 px patches, truth from the real in-between frame), at each pyramid level 5 .. 1 the block cost of V3's search is emulated over
the integer search area +-R level px for a number of features and window sizes (the V3 feature of `gpu/V3.md` section 2 exactly, with its lighting gain; wider high-passes; raw
luma with the window's own mean removed; normalised correlation; a census transform).  A patch is *findable* when the cost minimum lies within 1 level px of the true vector
(rounded to the level's grid).  Output: one .npz of per-patch records, aggregated by `study_table.py`.

    python study_match.py --list pairs_study.json --truth TRUTHDIR --out out.npz [--light none|up|down] [--only a,b] [--variants a8,zs12]
numpy and cv2 only; one pair at a time (a few hundred MB)."""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

R = 12            # search area +-R level px
LEVELS = (5, 4, 3, 2, 1)


def planes(rgb):
    """Y, Cb, Cr at flow resolution (2x2 cell means), float32"""
    f = rgb.astype(np.float32)
    Y = (4899 * f[..., 0] + 9617 * f[..., 1] + 1868 * f[..., 2]) / 16384.0
    Cb = 0.564 * (f[..., 2] - Y)
    Cr = 0.713 * (f[..., 0] - Y)
    full = np.stack([Y, Cb, Cr], -1)
    H, W = Y.shape
    return full.reshape(H // 2, 2, W // 2, 2, 3).mean((1, 3)), float(Y.mean()), float(Y.std())


def down(a):
    """2x2 mean, an odd last row / column replicated (V3's `v3_pyr`)"""
    h, w = a.shape[:2]
    if h & 1: a = np.concatenate([a, a[-1:]], 0)
    if w & 1: a = np.concatenate([a, a[:, -1:]], 1)
    h, w = a.shape[:2]
    return a.reshape(h // 2, 2, w // 2, 2, -1).mean((1, 3))


def pyramid(raw0, n=6):
    out = [raw0]
    for _ in range(n - 1):
        out.append(down(out[-1]))
    return out


def blur(a, k):
    return cv2.blur(a, (k, k), borderType=cv2.BORDER_REPLICATE).reshape(a.shape)


def hp(raw, k):
    return raw - blur(raw, k).reshape(raw.shape)


def feats(raws, l, kind):
    """feature stack [H, W, C] of level l for the feature `kind`"""
    raw = raws[l]
    Y = raw[..., :1]
    if kind == 'v3':
        return np.concatenate([hp(raw, 5), Y], -1)
    if kind in ('hp9', 'hp17'):
        return np.concatenate([hp(raw, int(kind[2:])), Y], -1)
    if kind == 'hpup':
        if l + 1 < len(raws):
            up = cv2.resize(raws[l + 1], (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_LINEAR).reshape(raw.shape)
        else:
            up = blur(raw, 17)
        return np.concatenate([raw - up, Y], -1)
    if kind in ('zs', 'zncc'):
        return Y
    if kind == 'census':
        y = Y[..., 0]
        p = np.pad(y, 2, mode='edge')
        H, W = y.shape
        bits = []
        for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (-2, 2), (2, -2), (2, 2)):
            bits.append((p[2 + dy:2 + dy + H, 2 + dx:2 + dx + W] > y).astype(np.float32))
        return np.stack(bits, -1)
    raise ValueError(kind)


CW = np.array([0.45, 0.20, 0.20, 0.15], np.float32)

# variant name -> (feature kind, cost kind, window, gain)
VARIANTS = {}
for w in (4, 8, 12, 16):
    VARIANTS[f'v3_{w}'] = ('v3', 'sad', w, True)
for k in ('hp9', 'hp17', 'hpup'):
    for w in (4, 8, 12):
        VARIANTS[f'{k}_{w}'] = (k, 'sad', w, True)
for w in (8, 12, 16):
    VARIANTS[f'zs_{w}'] = ('zs', 'zs', w, False)
for w in (8, 12):
    VARIANTS[f'zsg_{w}'] = ('zs', 'zs', w, True)
    VARIANTS[f'zncc_{w}'] = ('zncc', 'zncc', w, False)
    VARIANTS[f'census_{w}'] = ('census', 'cen', w, False)


def gain(statA, statB):
    """V3's (gm, K) of direction F01: role 0 = prev, role 1 = curr"""
    mu0, s0 = statA
    mu1, s1 = statB
    r = float(np.clip(s0 / max(s1, 1e-2), 0.5, 2.0))
    gm = 1.0 if abs(r - 1.0) < 0.03 else r - np.sign(r - 1.0) * 0.03
    return gm, gm * mu1 - mu0


def cost_map(Fa, Fb, ox, oy, win, ckind, gm, K, use_gain):
    """cost over v in [-R, R]^2 (inf where the window leaves the level), [2R+1, 2R+1]"""
    H, W = Fa.shape[:2]
    ox = int(np.clip(ox, 0, max(W - win, 0)))
    oy = int(np.clip(oy, 0, max(H - win, 0)))
    pad = np.pad(Fb, ((R, R), (R, R), (0, 0)), mode='edge')
    reg = pad[oy:oy + win + 2 * R, ox:ox + win + 2 * R]
    Wb = sliding_window_view(reg, (win, win), axis=(0, 1))          # [2R+1, 2R+1, C, win, win]
    Wa = np.moveaxis(Fa[oy:oy + win, ox:ox + win], -1, 0)           # [C, win, win]
    g = gm if use_gain else 1.0
    if ckind == 'sad':
        d = Wa[None, None] - g * Wb
        d[:, :, 3] += K if use_gain else 0.0
        c = np.einsum('uvcxy,c->uv', np.abs(d), CW) / float(win * win)
    elif ckind == 'cen':
        c = np.abs(Wa[None, None] - Wb).mean((2, 3, 4))
    elif ckind == 'zs':
        a = Wa - Wa.mean((1, 2), keepdims=True)
        b = Wb - Wb.mean((3, 4), keepdims=True)
        c = np.abs(a[None, None] - g * b).mean((2, 3, 4))
    else:   # zncc
        a = Wa - Wa.mean((1, 2), keepdims=True)
        b = Wb - Wb.mean((3, 4), keepdims=True)
        n = float(win * win)
        num = (a[None, None] * b).sum((2, 3, 4))
        c = 1.0 - num / (np.sqrt((a * a).sum() + n) * np.sqrt((b * b).sum((2, 3, 4)) + n))
    # validity: window of candidate v wholly inside the level
    vx = np.arange(-R, R + 1)
    okx = (ox + vx >= 0) & (ox + vx <= W - win)
    oky = (oy + vx >= 0) & (oy + vx <= H - win)
    c = np.where(oky[:, None] & okx[None, :], c, np.inf)
    return c   # indexed [vy + R, vx + R]


def run_pair(name, e, truth, variants, light):
    A = np.fromfile(e['prev'], np.uint8).reshape(1080, 1920, 4)[..., :3]
    B = np.fromfile(e['curr'], np.uint8).reshape(1080, 1920, 4)[..., :3]
    if light != 'none':
        xs = np.linspace(0, 1, 1920, dtype=np.float32)[None, :, None]
        s, add = (1.10, 14.0) if light == 'up' else (0.90, -14.0)
        B = np.clip(B.astype(np.float32) * s + add * xs, 0, 255).astype(np.uint8)
    pa, mua, sa = planes(A)
    pb, mub, sb = planes(B)
    gm, K = gain((mua, sa), (mub, sb))
    ra, rb = pyramid(pa), pyramid(pb)
    rows = []
    for l in LEVELS:
        stacks = {}
        for vn in variants:
            fk = VARIANTS[vn][0]
            if (fk, l) not in stacks:
                stacks[(fk, l)] = (feats(ra, l, fk), feats(rb, l, fk))
        for pi, (x, y, tx, ty) in enumerate(truth):
            s = 2.0 ** (l + 1)
            vt = np.array([tx, ty]) / s                      # true vector in level px (x, y)
            vr = np.rint(vt).astype(int)
            if np.abs(vr).max() >= R - 1:
                continue
            cx, cy = x / s - 0.5, y / s - 0.5
            moving = float(np.hypot(*vt)) >= 1.0
            for vn in variants:
                fk, ck, win, ug = VARIANTS[vn]
                Fa, Fb = stacks[(fk, l)]
                ox, oy = int(round(cx - (win - 1) / 2)), int(round(cy - (win - 1) / 2))
                c = cost_map(Fa, Fb, ox, oy, win, ck, gm, K, ug)
                ct = c[vr[1] + R, vr[0] + R]
                if not np.isfinite(ct):
                    continue
                pen = c + 1e-4 * (np.abs(np.arange(-R, R + 1))[:, None] + np.abs(np.arange(-R, R + 1))[None, :])   # ties: the smaller vector
                j, i = np.unravel_index(np.argmin(pen), pen.shape)
                fin = max(abs(i - R - vr[0]), abs(j - R - vr[1])) <= 1
                c0 = c[R, R]
                fm = c[np.isfinite(c)]
                scale = float(fm.mean()) + 1e-6
                rows.append((pi, l, vn, bool(fin), float((c0 - ct) / scale), moving, float(np.hypot(*vt)), float(ct), float(c0), scale))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', required=True)
    ap.add_argument('--truth', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--light', default='none')
    ap.add_argument('--only')
    ap.add_argument('--variants')
    a = ap.parse_args()
    variants = a.variants.split(',') if a.variants else list(VARIANTS)
    out = {}
    for e in json.load(open(a.list)):
        n = e['name']
        if a.only and not any(s in n for s in a.only.split(',')):
            continue
        t = np.load(os.path.join(a.truth, n + '.npy'))
        if len(t) == 0:
            continue
        rows = run_pair(n, e, t, variants, a.light)
        out[n] = dict(truth=t, rows=rows)
        print(n, len(t), len(rows), flush=True)
    import pickle
    pickle.dump(out, open(a.out, 'wb'))


if __name__ == '__main__':
    main()
