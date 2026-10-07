#!/usr/bin/env python3
"""Reads fgtest --dump files (pair_NNNN.fgmaps) and derives v1's per-pixel debug maps.

  fgmaps.py DUMP.fgmaps OUTPREFIX   -> OUTPREFIX_u.png (trust u, 0..255), _e.png (raw mismatch e), _cand.png (pixfield candidate,
                                       colour coded), _flow.png (per-pixel half-vector, HSV coded); a V2 dump (header version 2)
                                       also _class.png (V2's class map, see CLASS_RGB); version 3 also _flowE.png (V2's
                                       edge-aware field) and _flowR.png (V2's refined field), on the scale of _flow.png; version 4 (I10) stores the
                                       refined field packed (2 x int16 in 1/32 px)

Layout (fgtest.cpp, Interpolator::recordDebugMaps): 64-byte header, block field (bw*bh vec2 f32), per-pixel half-vector field
(fw*fh vec2 f32, midpoint grid), packed mismatch e (eStride*H uint32, 4 px per word, low byte first); version 2 (algo=v2): then
V2's class map (eStride*H uint32, same packing; Interpolator::DebugMapLayout::clsOff: bits 0-2 class, bit 3 one-sided side = curr,
bits 4-7 weight x 15); version 3 (algo=v2 since I9): then V2's edge-aware field (fw*fh vec2 f32, flow units) and refined midpoint
field (rw*rh vec2 f32, full-resolution units; header words 14-15 = rw, rh). u here is V1's ramp of the dump header's trustLow / trustHigh (V2's held class may use its preset's ramp).
u = ramp(7x7 box of e (replicate border), trustLow..trustHigh) as resolve.comp computes it; the candidate index is recovered by
matching the stored per-pixel vector against the 7 candidates of pixfield.comp (first match wins, as the first candidate wins ties):
0 bilinear block field, 1 own block, 2 above, 3 below, 4 left, 5 right, 6 zero.
"""
import sys

import numpy as np

CAND_NAMES = ["bilinear", "own", "up", "down", "left", "right", "zero"]
# colours of the candidate map: bilinear dark grey (the smooth default), own white, neighbours saturated, zero black-blue
# V2 class map: trusted black, held red, clean occlusion with the field's vector green, with the background vector blue, with the
# foreground vector magenta, with zero yellow; brightness = weight of the held / one-sided output (dim = mostly blended); one-sided from curr: whitened
CLASS_NAMES = ["trusted", "held", "occ-field", "occ-bg", "occ-fg", "occ-zero"]
CLASS_RGB = np.array([[0, 0, 0], [255, 0, 0], [0, 200, 0], [0, 120, 255], [255, 0, 255], [255, 255, 0]], np.float32)
CAND_RGB = np.array([[60, 60, 60], [255, 255, 255], [255, 60, 60], [60, 255, 60], [60, 120, 255], [255, 220, 0], [0, 0, 140]],
                    np.uint8)


def load(path):
    raw = open(path, "rb").read()
    h = np.frombuffer(raw[:64], np.uint32)
    hf = np.frombuffer(raw[40:56], np.float32)
    assert h[0] == 0x504D4746 and h[1] in (1, 2, 3, 4), "not an fgmaps v1 - v4 file"
    W, H, fw, fh, bw, bh, fscale, es = (int(x) for x in h[2:10])
    o = 64
    block = np.frombuffer(raw, np.float32, bw * bh * 2, o).reshape(bh, bw, 2); o += bw * bh * 8
    pf = np.frombuffer(raw, np.float32, fw * fh * 2, o).reshape(fh, fw, 2); o += fw * fh * 8
    ew = np.frombuffer(raw, np.uint32, es * H, o).reshape(H, es)
    e = np.stack([(ew >> (8 * i)) & 255 for i in range(4)], -1).reshape(H, es * 4)[:, :W].astype(np.float32)
    cls = pfE = pfR = None
    if h[1] >= 2:
        o += es * H * 4
        cw = np.frombuffer(raw, np.uint32, es * H, o).reshape(H, es)
        cls = np.stack([(cw >> (8 * i)) & 255 for i in range(4)], -1).reshape(H, es * 4)[:, :W].astype(np.uint8)
    if h[1] >= 3:
        rw, rh = int(h[14]), int(h[15])
        o += es * H * 4
        pfE = np.frombuffer(raw, np.float32, fw * fh * 2, o).reshape(fh, fw, 2); o += fw * fh * 8
        if h[1] >= 4:   # packed: two int16 in 1/32 px per uint32, x low (shaders/v2_pack.glsl)
            pfR = (np.frombuffer(raw, np.int16, rw * rh * 2, o).reshape(rh, rw, 2) / 32.0).astype(np.float32)
        else:
            pfR = np.frombuffer(raw, np.float32, rw * rh * 2, o).reshape(rh, rw, 2)
    return dict(W=W, H=H, fw=fw, fh=fh, bw=bw, bh=bh, fscale=fscale, block=block, pf=pf, e=e, cls=cls, pfE=pfE, pfR=pfR,
                trustLow=float(hf[0]), trustHigh=float(hf[1]))


def class_rgb(c):
    """V2 class-map bytes -> RGB"""
    k = np.minimum(c & 7, 5)
    w = ((c >> 4) & 15).astype(np.float32) / 15.0
    img = CLASS_RGB[k] * (0.35 + 0.65 * w[..., None])
    sc = ((c & 8) != 0) & (k >= 2)
    img[sc] = img[sc] * 0.6 + 255 * 0.4
    return img.astype(np.uint8)


def box(img, r):
    """(2r+1)^2 box mean, replicate border."""
    p = np.pad(img, r, mode="edge").astype(np.float64)
    c = p.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    n = 2 * r + 1
    s = c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]
    return (s / (n * n)).astype(np.float32)


def trust(d):
    """per-pixel weight of the held nearer real frame (resolve.comp's t)"""
    eb = box(d["e"], 3)
    return np.clip((eb - d["trustLow"]) / (d["trustHigh"] - d["trustLow"]), 0.0, 1.0)


def candidates(d):
    """(7, fh, fw, 2) candidate vectors of pixfield.comp"""
    B, bw, bh, fw, fh = d["block"], d["bw"], d["bh"], d["fw"], d["fh"]
    qy, qx = np.mgrid[0:fh, 0:fw]
    by, bx = qy // 8, qx // 8

    def bv(y, x):
        return B[np.clip(y, 0, bh - 1), np.clip(x, 0, bw - 1)]
    sy, sx = (qy + 0.5) / 8 - 0.5, (qx + 0.5) / 8 - 0.5
    iy, ix = np.floor(sy).astype(int), np.floor(sx).astype(int)
    fy, fx = (sy - iy)[..., None], (sx - ix)[..., None]
    bil = (bv(iy, ix) * (1 - fx) + bv(iy, ix + 1) * fx) * (1 - fy) + (bv(iy + 1, ix) * (1 - fx) + bv(iy + 1, ix + 1) * fx) * fy
    return np.stack([bil, bv(by, bx), bv(by - 1, bx), bv(by + 1, bx), bv(by, bx - 1), bv(by, bx + 1), np.zeros_like(bil)])


def cand_index(d, tol=1e-3):
    c = candidates(d)
    dist = np.abs(c - d["pf"][None]).max(-1)
    idx = np.full(dist.shape[1:], 7, np.uint8)   # 7 = no match (should not happen)
    for k in range(6, -1, -1):
        idx[dist[k] <= tol] = k
    return idx, c


def upsample(m, d):
    """flow res -> full res (nearest)"""
    s = d["fscale"]
    return np.repeat(np.repeat(m, s, 0), s, 1)[:d["H"], :d["W"]]


def flow_rgb(v, scale=None):
    mag = np.hypot(v[..., 0], v[..., 1])
    scale = scale or max(np.percentile(mag, 99), 1e-3)
    ang = (np.arctan2(v[..., 1], v[..., 0]) / (2 * np.pi)) % 1.0
    val = np.clip(mag / scale, 0, 1)
    h6 = ang * 6
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p, q, t = np.zeros_like(val), 1 - f, f
    rgb = np.select([i[..., None] == k for k in range(6)],
                    [np.stack(x, -1) for x in [(1 + 0 * f, t, p), (q, 1 + 0 * f, p), (p, 1 + 0 * f, t), (p, q, 1 + 0 * f),
                                               (t, p, 1 + 0 * f), (1 + 0 * f, p, q)]])
    return (rgb * val[..., None] * 255).astype(np.uint8)


def main():
    from PIL import Image
    d = load(sys.argv[1])
    pre = sys.argv[2]
    Image.fromarray((trust(d) * 255 + 0.5).astype(np.uint8)).save(pre + "_u.png")
    Image.fromarray(np.clip(d["e"] * 4, 0, 255).astype(np.uint8)).save(pre + "_e.png")
    idx, _ = cand_index(d)
    Image.fromarray(CAND_RGB[np.minimum(idx, 6)]).save(pre + "_cand.png")
    mag = np.hypot(d["pf"][..., 0], d["pf"][..., 1])
    sc = max(np.percentile(mag, 99), 1e-3)
    Image.fromarray(flow_rgb(d["pf"], sc)).save(pre + "_flow.png")
    if d["pfE"] is not None:   # the same colour scale (flow units; the refined field is in full-resolution units)
        Image.fromarray(flow_rgb(d["pfE"], sc)).save(pre + "_flowE.png")
        Image.fromarray(flow_rgb(d["pfR"], sc * d["fscale"])).save(pre + "_flowR.png")
    if d["cls"] is not None:
        Image.fromarray(class_rgb(d["cls"])).save(pre + "_class.png")
        k = np.minimum(d["cls"] & 7, 5)
        print("class share:", {CLASS_NAMES[i]: round(float((k == i).mean()), 4) for i in range(6)})
    print("candidate share:", {CAND_NAMES[k]: round(float((idx == k).mean()), 4) for k in range(7)},
          "unmatched", float((idx == 7).mean()))


if __name__ == "__main__":
    main()
