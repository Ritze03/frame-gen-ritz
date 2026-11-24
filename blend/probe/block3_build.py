"""Block table at B=8 for all 79 pairs (task 147) -> $BLEND/probe/block3/NN.npz (float32, block grid 135x240).
Keys: G [h,w,6,6], b [h,w,6], yy [h,w] (0..1 units, six candidates tap0..3, prev, curr, all 64 px x RGB); Ge, be, yye, ne (edge-mask pixels only);
feat [h,w,136] = mean(34), std(34), min(34), max(34) of X over the block; rule [h,w,4] block-mean rule weights.
usage: block3_build.py [--procs N]"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BLEND = os.environ.get("BLEND", "/home/mo/DieGanzeSau/AI_Video_Training_Data/blend")
ROOT = f"{BLEND}/probe"
OUT = f"{ROOT}/block3"
H, W, B = 1080, 1920, 8


def build(k):
    import torch
    torch.set_num_threads(1)
    from score import edge_mask
    if os.path.exists(f"{OUT}/{k:02d}.npz"):
        return k
    d = f"{ROOT}/pairs/{k:02d}"
    ld = lambda f: np.load(f"{d}/{f}.npy", allow_pickle=False, mmap_mode="r")
    taps, e, y, rule, X = ld("taps"), ld("e"), ld("y"), ld("rule"), ld("X")
    em = edge_mask(np.asarray(y))
    h, w = H // B, W // B
    G = np.zeros((h, w, 6, 6)); b = np.zeros((h, w, 6)); yy = np.zeros((h, w))
    Ge = np.zeros_like(G); be = np.zeros_like(b); yye = np.zeros_like(yy); ne = np.zeros((h, w))
    feat = np.zeros((h, w, 136), np.float32); rl = np.zeros((h, w, 4), np.float32)
    for br in range(h):
        r0 = br * B
        sl = slice(r0, r0 + B)
        pts = np.concatenate([np.asarray(taps[sl]).reshape(B, W, 4, 3), np.asarray(e[sl]).reshape(B, W, 2, 3)], 2).astype(np.float64) / 255.0
        yv = np.asarray(y[sl]).astype(np.float64) / 255.0
        m = em[sl].astype(np.float64)
        pg = np.einsum("hwic,hwjc->hwij", pts, pts); pb = np.einsum("hwic,hwc->hwi", pts, yv); py = (yv * yv).sum(-1)
        bs = lambda a: a.reshape(B, w, B, *a.shape[2:]).sum((0, 2))
        G[br], b[br], yy[br] = bs(pg), bs(pb), bs(py)
        Ge[br], be[br], yye[br], ne[br] = bs(pg * m[..., None, None]), bs(pb * m[..., None]), bs(py * m), bs(m)
        xb = np.asarray(X[sl]).astype(np.float32).reshape(B, w, B, 34).transpose(1, 0, 2, 3).reshape(w, B * B, 34)
        feat[br] = np.concatenate([xb.mean(1), xb.std(1), xb.min(1), xb.max(1)], 1)
        rl[br] = bs(np.asarray(rule[sl]).astype(np.float64)) / (B * B)
    np.savez(f"{OUT}/{k:02d}.tmp.npz", G=G.astype(np.float32), b=b.astype(np.float32), yy=yy.astype(np.float32), Ge=Ge.astype(np.float32),
             be=be.astype(np.float32), yye=yye.astype(np.float32), ne=ne.astype(np.float32), feat=feat, rule=rl)
    os.rename(f"{OUT}/{k:02d}.tmp.npz", f"{OUT}/{k:02d}.npz")
    return k


if __name__ == "__main__":
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 2
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    with Pool(procs) as pool:
        for k in pool.imap_unordered(build, range(79)):
            print(k, "%.0fs" % (time.time() - t0), flush=True)
