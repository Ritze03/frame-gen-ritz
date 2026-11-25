#!/usr/bin/env python3
"""Statistics of MotionVersion::V3's estimated flows from fgtest --dump DIR files (pair_NNNN.v3db, Interpolator::recordDebugV3's dump).

  v3flow.py pair_0000.v3db [more.v3db ...]

Per file: the header, the length of F01 / F10 (flow px, full prev->curr vectors; mean, p99, max), the number of NaN / inf / absurd (> the frame
diagonal) vectors, the match cost (MAP cost8 / 8, levels) and the forward-backward agreement |F01(p) + F10(p + F01(p))| (bilinear sample of F10;
median, share <= 1 flow px, share > 4 flow px; pixels whose target leaves the frame are not counted). Layout: gpu/framegen.h DebugV3Layout
(16-word header padded to 64 bytes, then FF[0], FF[1], FD[0], FD[1] (fw*fh vec2 f32 each), MAP[0], MAP[1], S (fw*fh uint32 each), BLK,
G words 0..63, the cells; version 2 (task 109): T-field slot 0, load() returns it as gA, gB (flow px), rA, rB (o, hole, u) and its t as tf_t).
"""
import sys

import numpy as np


def load(path):
    raw = open(path, "rb").read()
    h = np.frombuffer(raw[:64], np.uint32)
    assert h[0] == 0x42443356, "not a recordDebugV3 dump"
    W, H, fw, fh = (int(x) for x in h[2:6])
    n = fw * fh
    o = 64
    out = dict(header=h, W=W, H=H, fw=fw, fh=fh, fs=int(h[8]), levels=int(h[9]), mode=int(h[10]))
    for name in ("ff0", "ff1", "fd0", "fd1"):
        out[name] = np.frombuffer(raw, np.float32, n * 2, o).reshape(fh, fw, 2)
        o += n * 8
    for name in ("map0", "map1", "s"):
        out[name] = np.frombuffer(raw, np.uint32, n, o).reshape(fh, fw)
        o += n * 4
    # BLK of every level (4 arrays of gw*gh vec4 per level: dir 0 slot 0, dir 0 slot 1, dir 1 slot 0, dir 1 slot 1), then (step E2) the first 64 words of G
    lw, lh = fw, fh
    blk = []
    for lv in range(out["levels"]):
        bs = 8 if lv == 0 else 4
        gw, gh = (lw + bs - 1) // bs, (lh + bs - 1) // bs
        a = np.frombuffer(raw, np.float32, 4 * gw * gh * 4, o).reshape(4, gh, gw, 4)
        blk.append(a)
        o += 4 * gw * gh * 16
        lw, lh = (lw + 1) // 2 if lw % 2 == 0 else lw // 2 + (lw // 2) % 2, (lh + 1) // 2 if lh % 2 == 0 else lh // 2 + (lh // 2) % 2
    out["blk"] = blk
    if len(raw) >= o + 256:
        out["g"] = np.frombuffer(raw, np.uint32, 64, o)
    o += 256
    n8 = ((fw + 7) // 8) * ((fh + 7) // 8)
    o += n8 * 12   # the cells s8, E, C
    if int(h[1]) >= 2 and len(raw) >= o + 16 * n:   # (version 2, task 109) T-field slot 0 of the latest synth, its t in header word 14
        w = np.frombuffer(raw, np.uint32, 4 * n, o).reshape(fh, fw, 4)
        out["tf_t"] = float(h[14:15].view(np.float32)[0])
        half = lambda x: x.astype(np.uint16).view(np.float16).astype(np.float32)
        out["gA"] = np.stack([half(w[..., 0] & 0xffff), half(w[..., 0] >> 16)], -1)   # flow px
        out["gB"] = np.stack([half(w[..., 1] & 0xffff), half(w[..., 1] >> 16)], -1)
        out["rA"] = np.stack([(w[..., 2] >> (8 * k)) & 255 for k in range(3)], -1).astype(np.float32) / 255   # oA, holeA, uA
        out["rB"] = np.stack([(w[..., 3] >> (8 * k)) & 255 for k in range(3)], -1).astype(np.float32) / 255
        o += 16 * n
        if len(raw) >= o + 800:   # (task 161) G words 3896..4095: g35[8 r + 4 .. 8 r + 11] = region r's model words (V3.5, regions 4 x 3), g35[104:] = the diagnosis counters of a V35_DIAG build
            out["g35"] = np.frombuffer(raw, np.uint32, 200, o)
    return out


def bilinear(f, x, y):
    """f: (h, w, 2); x, y: float arrays (flow px index coordinates, clamped)."""
    h, w, _ = f.shape
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    ax = (x - x0)[..., None]
    ay = (y - y0)[..., None]
    return (f[y0, x0] * (1 - ax) + f[y0, x1] * ax) * (1 - ay) + (f[y1, x0] * (1 - ax) + f[y1, x1] * ax) * ay


def stats(path):
    d = load(path)
    fw, fh = d["fw"], d["fh"]
    print(f"{path}: {d['W']}x{d['H']} flowScale {d['fs']}, {d['levels']} levels, G.mode {d['mode']}")
    diag = float(np.hypot(fw, fh))
    for name, f in (("F01", d["ff0"]), ("F10", d["ff1"])):
        bad = ~np.isfinite(f).all(-1)
        fin = np.where(np.isfinite(f), f, 0)
        ln = np.hypot(fin[..., 0], fin[..., 1])
        absurd = ln > diag
        print(f"  {name}: |v| mean {ln.mean():.2f} p99 {np.percentile(ln, 99):.2f} max {ln.max():.2f} flow px; NaN/inf {int(bad.sum())}, > diagonal {int(absurd.sum())}")
    for name, m in (("F01", d["map0"]), ("F10", d["map1"])):
        c = (m & 255).astype(np.float32) / 8
        print(f"  {name} match cost: mean {c.mean():.2f} p50 {np.median(c):.2f} p95 {np.percentile(c, 95):.2f} levels, share >= 18 {float((c >= 18).mean()) * 100:.1f}%")
    g = d.get("g")
    if g is not None:
        gf = g.view(np.float32)
        doms = [f"({gf[8 + 2 * k]:.1f},{gf[9 + 2 * k]:.1f})" for k in range(4) if g[6] & (1 << k)]
        aff = f"dx = {gf[16]:.2f} {gf[17]:+.2f} X {gf[18]:+.2f} Y, dy = {gf[19]:.2f} {gf[20]:+.2f} X {gf[21]:+.2f} Y" if g[6] & 256 else "invalid"
        print(f"  stats: dominant {' '.join(doms) or 'none'}; affine {aff}; wide search flagged {int(g[25])} replaced {int(g[26])} blocks")
        for dn in range(2):
            amb = d["blk"][0][dn * 2, :, :, 3]
            print(f"  amb F{'01' if dn == 0 else '10'} (level-0 blocks): share > 0.5 {float((amb > 0.5).mean()) * 100:.1f}%, mean {amb.mean():.3f}")
    yy, xx = np.mgrid[0:fh, 0:fw].astype(np.float32)
    for name, a, b in (("F01", d["ff0"], d["ff1"]), ("F10", d["ff1"], d["ff0"])):
        tx, ty = xx + a[..., 0], yy + a[..., 1]
        ok = (tx >= 0) & (tx <= fw - 1) & (ty >= 0) & (ty <= fh - 1)
        e = np.hypot(*(a + bilinear(b, tx, ty)).transpose(2, 0, 1))[ok]
        print(f"  forward-backward {name}: median {np.median(e):.3f} p90 {np.percentile(e, 90):.2f} flow px; <= 1 px {float((e <= 1).mean()) * 100:.1f}%, > 4 px {float((e > 4).mean()) * 100:.1f}%  ({float(ok.mean()) * 100:.1f}% of pixels have a target inside)")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        stats(p)
