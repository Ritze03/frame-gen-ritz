"""Stride-1 records for every pair of pairs.json -> compact arrays under $BLEND/probe/pairs/<k>/ (the raw record is deleted).
usage: convert.py [--fgtest PATH] [--only K,K..]
Files (H=1080, W=1920): X.npy f16 [H,W,34] (feat.features), taps.npy f16 [H,W,12] (levels), e.npy u8 [H,W,6] (prev(x), curr(x)), y.npy u8 [H,W,3],
rule.npy f16 [H,W,4]; held-out pairs also baseC.npy / forzaC.npy f16 [H,W,6]: the six weights of blend/weights/{base,forza-rx9070xt}/C.npz
(same maths as gpu/ml/blend/ref.py, whose w0..w3 and w4+w5 they are asserted to equal)."""
import json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
sys.path.insert(0, os.path.join(HERE, "..", "data"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "gpu", "ml", "blend"))
from common import BLEND, W, H
from dump import frame_path, target_path, LOCK
from feat import features
from ref import blend_weights

FG = sys.argv[sys.argv.index("--fgtest") + 1] if "--fgtest" in sys.argv else os.path.expanduser("~/tmp/fg-build-143/fgtest")
ONLY = [int(x) for x in sys.argv[sys.argv.index("--only") + 1].split(",")] if "--only" in sys.argv else None
SCR = os.path.join(HERE, "..", "..", ".tl-scratch", "conv")
os.makedirs(SCR, exist_ok=True)
man = json.load(open(f"{BLEND}/frames/manifest.json"))["samples"]
pairs = json.load(open(f"{BLEND}/probe/pairs.json"))
NETS = {"baseC": f"{HERE}/../weights/base/C.npz", "forzaC": f"{HERE}/../weights/forza-rx9070xt/C.npz"}
CH = 1 << 18


def rgb(p):
    return np.fromfile(p, np.uint8).reshape(-1, 4)[:, :3]


def net6(X, rule, npz):
    """The six weights of a six-choice net file (ref.py's maths) from the 34 inputs X [n,34] f32 and the rule weights [n,4]."""
    z = np.load(npz, allow_pickle=False)
    W1, b1, W2, b2 = (np.asarray(z[k], np.float32) for k in ("W1", "b1", "W2", "b2"))
    z1 = X @ W1.T + b1
    z2 = np.maximum(z1, 0.1 * z1) @ W2.T + b2
    p = np.concatenate([np.log(np.maximum(rule, 1e-3)), np.full((len(X), 2), -5.0, np.float32)], 1)
    lg = p + z2
    e = np.exp(lg - lg.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


CHUNK = 4   # pairs per fgtest call (lock hold about 1.2 s per pair when the GPU is free)
todo = [k for k in range(len(pairs)) if (ONLY is None or k in ONLY) and not os.path.exists(f"{BLEND}/probe/pairs/{k:02d}/DONE")]
for c0 in range(0, len(todo), CHUNK):
    ks = todo[c0:c0 + CHUNK]
    t0 = time.time()
    pre = f"{SCR}/w"
    cmd = ["flock", LOCK, "ccorral", "--", FG, "--window", "1", "--set", "algo=v3", "--t", "0.5", "--v3weights", pre, "--v3wmax", str(W * H), str(W), str(H)]
    for k in ks:
        s = man[pairs[k]["idx"]]
        assert s["id"] == pairs[k]["id"] and s["t"] == 0.5
        cmd += [frame_path(s, s["prev"]), frame_path(s, s["curr"]), "/dev/null"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.time()
    for j, k in enumerate(ks):
        p = pairs[k]
        s = man[p["idx"]]
        out = f"{BLEND}/probe/pairs/{k:02d}"
        os.makedirs(out, exist_ok=True)
        t2 = time.time()
        r28 = np.fromfile(f"{pre}_{j:04d}.bin", np.float32).reshape(-1, 28)
        assert len(r28) == W * H, (len(r28), r.stdout[-300:], r.stderr[-300:])
        rec = np.empty((W * H, 40), np.float32)
        rec[:, :28] = r28
        del r28
        os.remove(f"{pre}_{j:04d}.bin")
        pv, cu, tg = rgb(frame_path(s, s["prev"])), rgb(frame_path(s, s["curr"])), rgb(target_path(s))
        rec[:, 28:31], rec[:, 31:34], rec[:, 34:37] = pv, cu, tg
        X = np.empty((W * H, 34), np.float16)
        for a in range(0, W * H, CH):
            X[a:a + CH] = features(rec[a:a + CH])
        np.save(f"{out}/X.npy", X.reshape(H, W, 34))
        del X
        np.save(f"{out}/taps.npy", rec[:, 0:12].astype(np.float16).reshape(H, W, 12))
        np.save(f"{out}/e.npy", np.concatenate([pv, cu], 1).reshape(H, W, 6))
        np.save(f"{out}/y.npy", tg.reshape(H, W, 3))
        np.save(f"{out}/rule.npy", rec[:, 22:26].astype(np.float16).reshape(H, W, 4))
        if p["split"] == "val":
            for nm, path in NETS.items():
                w = np.empty((W * H, 6), np.float16)
                for a in range(0, W * H, CH):
                    rr = rec[a:a + CH]
                    x = features(rr)
                    w6 = net6(x, rr[:, 22:26], path)
                    if a == 0:
                        w4, w45 = blend_weights(rr, path, rr[:, 28:31], rr[:, 31:34])
                        assert np.abs(w4 - w6[:, :4]).max() < 1e-4 and np.abs(w45 - w6[:, 4] - w6[:, 5]).max() < 1e-4
                    w[a:a + CH] = w6
                np.save(f"{out}/{nm}.npy", w.reshape(H, W, 6))
        open(f"{out}/DONE", "w").write("ok")
        print(k, p["id"], "call %.1fs convert %.1fs" % (t1 - t0, time.time() - t2), flush=True)
