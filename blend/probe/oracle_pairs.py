"""O4 / O6 colours (oracle.py's solve) for the held-out pairs, from the stored arrays -> $BLEND/probe/pairs/<k>/o4.npy, o6.npy (f16 [H,W,3], levels).
Uses the stored f16 taps (levels) and e (u8): the same points every network is scored on. usage: oracle_pairs.py [--procs N]"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "data"))
from common import BLEND, W, H
from oracle import solve

D = f"{BLEND}/probe/pairs"
CH = 1 << 16


def job(args):
    k, a = args
    d = f"{D}/{k:02d}"
    sl = slice(a, a + CH)
    taps = np.load(f"{d}/taps.npy", mmap_mode="r").reshape(-1, 12)[sl].astype(np.float64).reshape(-1, 4, 3)
    e = np.load(f"{d}/e.npy", mmap_mode="r").reshape(-1, 6)[sl].astype(np.float64).reshape(-1, 2, 3)
    y = np.load(f"{d}/y.npy", mmap_mode="r").reshape(-1, 3)[sl].astype(np.float64)
    six = np.concatenate([taps, e], 1)
    o4 = np.einsum("nk,nkc->nc", solve(taps, y)[0], taps)
    o6 = np.einsum("nk,nkc->nc", solve(six, y)[0], six)
    return k, a, o4.astype(np.float16), o6.astype(np.float16)


if __name__ == "__main__":
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 3
    pairs = json.load(open(f"{BLEND}/probe/pairs.json"))
    for k, p in enumerate(pairs):
        if p["split"] != "val" or os.path.exists(f"{D}/{k:02d}/o6.npy") or not os.path.exists(f"{D}/{k:02d}/DONE"):
            continue
        t0 = time.time()
        o4 = np.empty((W * H, 3), np.float16)
        o6 = np.empty((W * H, 3), np.float16)
        with Pool(procs) as pool:
            for _, a, x4, x6 in pool.imap_unordered(job, [(k, a) for a in range(0, W * H, CH)]):
                o4[a:a + CH], o6[a:a + CH] = x4, x6
        np.save(f"{D}/{k:02d}/o4.npy", o4.reshape(H, W, 3))
        np.save(f"{D}/{k:02d}/o6.npy", o6.reshape(H, W, 3))
        print(k, "oracle %.0fs" % (time.time() - t0), flush=True)
