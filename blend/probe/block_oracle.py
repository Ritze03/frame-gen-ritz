"""Block-constant oracle (task 144).  For each held-out pair and block size B (2,4,8,16,32, 0 = whole frame) the best mix with six (or four)
simplex weights constant inside each BxB block: min sum over block pixels and RGB |sum_k w_k c_k - y|^2 (Gram data G = sum c c^T, b = sum c y, yy; same
subset-enumeration solver as blend/data/oracle.py but on stacked rows).  Points = 4 stored taps + prev + curr (the points oracle_pairs.py uses).
Output $BLEND/probe/block_oracle/NN.json: per variant accumulators {se,n,se_e,n_e}.
Variant names: {K}_{B}  plain fit scored on the whole frame, {K}_{B}_odd plain fit scored on odd pixels ((y+x) odd), {K}_{B}_cb fit on even pixels scored on odd.
usage: block_oracle.py [--procs N] | block_oracle.py --check PAIR   (B=1 check vs stored o4/o6)"""
import itertools, json, os, sys, time
from multiprocessing import Pool
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BLEND = os.environ.get("BLEND", "/home/mo/DieGanzeSau/AI_Video_Training_Data/blend")
ROOT = f"{BLEND}/probe"
H, W = 1080, 1920
SUBSETS = {K: [S for r in range(1, K + 1) for S in itertools.combinations(range(K), r)] for K in (4, 6)}
BS = [2, 4, 8, 16, 32, 0]
CB = [4, 8, 16]


def gsolve(G, b, yy, K):
    """G [n,6,6] (first K used), b [n,6], yy [n] -> w [n,K], err [n]."""
    G, b = G[:, :K, :K], b[:, :K]
    n = len(G)
    best = np.full(n, np.inf)
    bestw = np.zeros((n, K))
    scale = np.trace(G, axis1=1, axis2=2) / K + 1.0
    for S in SUBSETS[K]:
        s = len(S)
        ix = np.array(S)
        if s == 1:
            w = np.ones((n, 1))
        else:
            M = np.zeros((n, s + 1, s + 1))
            M[:, :s, :s] = G[:, ix[:, None], ix[None, :]]
            M[:, np.arange(s), np.arange(s)] += (1e-9 * scale)[:, None]
            M[:, :s, s] = 1.0
            M[:, s, :s] = 1.0
            rhs = np.zeros((n, s + 1))
            rhs[:, :s] = b[:, ix]
            rhs[:, s] = 1.0
            w = np.linalg.solve(M, rhs[:, :, None])[:, :s, 0]
        ok = (w >= -1e-9).all(axis=1)
        w = np.where(ok[:, None], np.maximum(w, 0.0), 0.0)
        sw = w.sum(axis=1)
        ok &= sw > 0.5
        w = w / np.maximum(sw, 1e-12)[:, None]
        Gs = G[:, ix[:, None], ix[None, :]]
        err = np.einsum("ni,nij,nj->n", w, Gs, w) - 2 * np.einsum("ni,ni->n", w, b[:, ix]) + yy
        err = np.where(ok, err, np.inf)
        upd = err < best
        best = np.where(upd, err, best)
        full = np.zeros((n, K))
        full[:, ix] = w
        bestw[upd] = full[upd]
    return bestw, best


def load(k):
    d = f"{ROOT}/pairs/{k:02d}"
    ld = lambda f: np.load(f"{d}/{f}.npy", allow_pickle=False)
    taps, e, y = ld("taps"), ld("e"), ld("y")
    pts = np.concatenate([taps.reshape(H, W, 4, 3), e.reshape(H, W, 2, 3)], 2).astype(np.float64)
    return pts, y.astype(np.float64), ld("rule").astype(np.float64), y


def pix_stats(pts, y, m=None):
    """per-pixel Gram data (optionally masked): G [h,w,6,6], b [h,w,6], yy [h,w]"""
    G = np.einsum("hwic,hwjc->hwij", pts, pts)
    b = np.einsum("hwic,hwc->hwi", pts, y)
    yy = (y * y).sum(-1)
    if m is not None:
        G, b, yy = G * m[..., None, None], b * m[..., None], yy * m
    return G, b, yy


def blocksum(a, B):
    h, w = a.shape[:2]
    return a.reshape(h // B, B, w // B, B, *a.shape[2:]).sum((1, 3))


def acc(se, em, m):
    return dict(se=float(se[m].sum()), n=int(m.sum()), se_e=float(se[m & em].sum()), n_e=int((m & em).sum()))


def run_pair(k):
    from score import edge_mask
    import torch
    torch.set_num_threads(1)
    pts, y, rule, yu8 = load(k)
    em = edge_mask(yu8)
    par = ((np.arange(H)[:, None] + np.arange(W)[None]) % 2 == 1)   # odd pixels
    out = {}
    SH = 32
    allm = np.ones((H, W), bool)
    c = np.einsum("hwk,hwkc->hwc", rule, pts[:, :, :4])
    se = ((c - y) ** 2).sum(-1)
    out["rule"], out["rule_odd"] = acc(se, em, allm), acc(se, em, par)
    Gw = {}
    for tag, m in (("p", None), ("e", (~par).astype(np.float64))):
        Gt, bt, yt = np.zeros((6, 6)), np.zeros(6), 0.0
        for r0 in range(0, H, SH):
            G, b, yy = pix_stats(pts[r0:r0 + SH], y[r0:r0 + SH], None if m is None else m[r0:r0 + SH])
            Gt += G.sum((0, 1)); bt += b.sum((0, 1)); yt += yy.sum()
        Gw[tag] = (Gt, bt, yt)
    W_whole = {(K, tag): gsolve(Gw[tag][0][None], Gw[tag][1][None], np.array([Gw[tag][2]]), K)[0][0] for K in (6, 4) for tag in "pe"}
    tot = {}

    def add(name, se, em_, m):
        a = tot.setdefault(name, dict(se=0.0, n=0, se_e=0.0, n_e=0))
        for kk, v in acc(se, em_, m).items():
            a[kk] += v

    for r0 in range(0, H, SH):
        h = min(SH, H - r0)
        P = np.zeros((SH, W, 6, 3)); Y = np.zeros((SH, W, 3))
        P[:h], Y[:h] = pts[r0:r0 + h], y[r0:r0 + h]
        EM = em[r0:r0 + h]; PAR = par[r0:r0 + h]; ALL = np.ones((h, W), bool)
        evenm = np.zeros((SH, W)); evenm[:h] = ~par[r0:r0 + h]
        Gp, bp, yp = pix_stats(P, Y)
        Ge, be, ye = Gp * evenm[..., None, None], bp * evenm[..., None], yp * evenm
        for B in BS:
            for K in (6, 4):
                for tag, (G, b, yy) in (("p", (Gp, bp, yp)), ("e", (Ge, be, ye))):
                    if tag == "e" and B not in CB:
                        continue
                    if B == 0:
                        w = np.broadcast_to(W_whole[(K, tag)], (SH, W, K))
                    else:
                        gb, bb, yb = blocksum(G, B), blocksum(b, B), blocksum(yy, B)
                        nb = gb.shape[0] * gb.shape[1]
                        w = gsolve(gb.reshape(nb, 6, 6), bb.reshape(nb, 6), yb.reshape(nb), K)[0].reshape(gb.shape[0], gb.shape[1], K)
                        w = np.repeat(np.repeat(w, B, 0), B, 1)
                    c = np.einsum("hwk,hwkc->hwc", w[:h], pts[r0:r0 + h, :, :K])
                    se = ((c - y[r0:r0 + h]) ** 2).sum(-1)
                    nm = f"{K}_{B}"
                    if tag == "p":
                        add(nm, se, EM, ALL); add(nm + "_odd", se, EM, PAR)
                    else:
                        add(nm + "_cb", se, EM, PAR)
    out.update(tot)
    os.makedirs(f"{ROOT}/block_oracle", exist_ok=True)
    json.dump(out, open(f"{ROOT}/block_oracle/{k:02d}.json", "w"))
    return k


def check(k):
    """B=1: gsolve on the per-pixel Gram data vs the stored o6 / o4 (rows 400..520 of the pair, all columns)."""
    pts, y, _, _ = load(k)
    sl = slice(400, 520)
    G, b, yy = pix_stats(pts[sl], y[sl])
    n = G.shape[0] * G.shape[1]
    for K, f in ((6, "o6"), (4, "o4")):
        w, _ = gsolve(G.reshape(n, 6, 6), b.reshape(n, 6), yy.reshape(n), K)
        c = np.einsum("nk,nkc->nc", w, pts[sl].reshape(n, 6, 3)[:, :K]).astype(np.float16).astype(np.float64)
        o = np.load(f"{ROOT}/pairs/{k:02d}/{f}.npy", allow_pickle=False)[sl].reshape(n, 3).astype(np.float64)
        d = np.abs(c - o)
        yt = y[sl].reshape(n, 3)
        mse = lambda a: ((a - yt) ** 2).mean()
        print(f, "pair", k, "n", n, "max |colour diff| %.4f levels, share of pixels diff>0.1: %.5f" % (d.max(), (d.max(1) > 0.1).mean()),
              "mse mine %.4f stored %.4f" % (mse(c), mse(o)), flush=True)


if __name__ == "__main__":
    if "--check" in sys.argv:
        check(int(sys.argv[sys.argv.index("--check") + 1])); sys.exit()
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 3
    pairs = json.load(open(f"{ROOT}/pairs.json"))
    ks = [k for k, p in enumerate(pairs) if p["split"] == "val" and not os.path.exists(f"{ROOT}/block_oracle/{k:02d}.json")]
    t0 = time.time()
    with Pool(procs) as pool:
        for k in pool.imap_unordered(run_pair, ks):
            print(k, "%.0fs" % (time.time() - t0), flush=True)
