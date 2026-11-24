#!/usr/bin/env python3
"""Numpy reference of the learned-blend net (SPEC section 4, gpu/V3.md 3.16 "As built (learned blend)"): the shader's weights from a record dump.

    blend_weights(rec, npz, e0=None, e1=None) -> (w [n, 4], w45 [n])
    rec: float32 [n, >=28] records of `fgtest --v3weights` (columns CAp CAc CBp CBc (levels), oA holeA uA oB holeB uB t lenA lenB sigma, w0..w3 s pad); the rule's
         weights r (columns 22..25) must come from a dump at v3Blend 0 (a dump at v3Blend >= 1 holds the net's weights there; its inputs are the same).
    npz: W1 [8][NI], b1 [8], W2 [K][8], b2 [K] (NI 34, K 6: six choices, e0 / e1 [n, 3] (levels) = prev(x) / curr(x) are needed; the two nets in the build are of this shape. NI 25, K 4, the old nets A and B, still work here as training history).
    Returns the four tap weights (before any renormalisation) and, for six choices, w4 + w5 (zeros otherwise).
CLI: ref.py RULE_DUMP.bin NET_DUMP.bin WEIGHTS.npz W H prev.rgba curr.rgba STRIDE [rgb10]  -> prints the max |difference| of w0..w3 and w4 + w5 (NET_DUMP = the shader's).
    The optional last argument rgb10: the dump came from `fgtest --format rgb10`, whose upload turns each 8-bit value v into the PQ code of 1000 * (v/255)^2.4 nits
    (rounded to 10 bit); the shader's working values e0 / e1 are code * 255 / 1023 (variant C only).
"""
import sys
import numpy as np

def m(v):
    return 4.0 * np.abs(v).mean(axis=1)

def blend_weights(rec, npz, e0=None, e1=None):
    z = np.load(npz) if isinstance(npz, str) else npz
    W1, b1, W2, b2 = (np.asarray(z[k], np.float32) for k in ("W1", "b1", "W2", "b2"))
    rec = np.asarray(rec, np.float32)
    k = W2.shape[0]
    c = [rec[:, 3 * i:3 * i + 3] / np.float32(255) for i in range(4)]
    x = np.concatenate([np.concatenate(c[:], axis=1), rec[:, 12:19], np.minimum(rec[:, 19:21], 256) / 64,
                        np.stack([m(c[0] - c[1]), m(c[2] - c[3]), m(c[0] - c[2]), m(c[1] - c[3])], axis=1)], axis=1)
    if W1.shape[1] == 34:
        f0, f1 = np.asarray(e0, np.float32) / 255, np.asarray(e1, np.float32) / 255
        x = np.concatenate([x, f0, f1, np.stack([m(f0 - f1), m(c[0] - f0), m(c[3] - f1)], axis=1)], axis=1)
    assert x.shape[1] == W1.shape[1], (x.shape, W1.shape)
    z1 = x @ W1.T + b1
    h = np.maximum(z1, 0.1 * z1)
    z2 = h @ W2.T + b2
    r = rec[:, 22:26]
    p = np.log(np.maximum(r, 1e-3))
    if k == 6:
        p = np.concatenate([p, np.full((len(p), 2), -5.0, np.float32)], axis=1)
    l = p + z2
    e = np.exp(l - l.max(axis=1, keepdims=True))
    w = e / e.sum(axis=1, keepdims=True)
    return w[:, :4], (w[:, 4] + w[:, 5] if k == 6 else np.zeros(len(w), np.float32))

def rgb10_levels(v8):
    """8-bit file values -> the working values (levels) of fgtest --format rgb10's upload."""
    n = 1000.0 * (np.asarray(v8, np.float64) / 255.0) ** 2.4
    p = np.minimum(n / 10000.0, 1.0) ** 0.1593017578125
    code = np.floor(((0.8359375 + 18.8515625 * p) / (1.0 + 18.6875 * p)) ** 78.84375 * 1023.0 + 0.5)
    return (code * (255.0 / 1023.0)).astype(np.float32)

def read_dump(path, n=28):
    return np.fromfile(path, np.float32).reshape(-1, n)

if __name__ == "__main__":
    rule, net, npz, W, H, pf, cf, stride = sys.argv[1:9]
    W, H, stride = int(W), int(H), int(stride)
    R, N = read_dump(rule), read_dump(net)
    e0 = e1 = None
    if np.load(npz)["W1"].shape[1] == 34:
        idx = np.arange(len(R), dtype=np.int64) * stride
        pa = np.fromfile(pf, np.uint8).reshape(H * W, 4)[idx, :3].astype(np.float32)
        ca = np.fromfile(cf, np.uint8).reshape(H * W, 4)[idx, :3].astype(np.float32)
        e0, e1 = (rgb10_levels(pa), rgb10_levels(ca)) if sys.argv[9:] == ["rgb10"] else (pa, ca)
    w, w45 = blend_weights(R, npz, e0, e1)
    print("records %d  max|dw0..3| %.3g  max|dw45| %.3g" % (len(R), np.abs(w - N[:, 22:26]).max(), np.abs(w45 - N[:, 27]).max()))
