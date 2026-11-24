"""Fixed training crops + their per-pixel O6 colour (task 144).  20 random 128x128 crops per training pair (rng seed 144, one stream, pair order),
O6 colour with oracle.py's solve on the stored taps / prev / curr / real frame -> $BLEND/probe/o6crops/{crops.json, NN.npy (f16 [20,128,128,3], levels)}.
usage: make_o6crops.py [--procs N]"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "data"))
BLEND = os.environ.get("BLEND", "/home/mo/DieGanzeSau/AI_Video_Training_Data/blend")
ROOT = f"{BLEND}/probe"
OUT = f"{ROOT}/o6crops"
H, W, C, NC = 1080, 1920, 128, 20


def job(args):
    from oracle import solve
    k, pos = args
    d = f"{ROOT}/pairs/{k:02d}"
    taps = np.load(f"{d}/taps.npy", mmap_mode="r", allow_pickle=False).reshape(H, W, 4, 3)
    e = np.load(f"{d}/e.npy", mmap_mode="r", allow_pickle=False).reshape(H, W, 2, 3)
    y = np.load(f"{d}/y.npy", mmap_mode="r", allow_pickle=False)
    res = np.empty((len(pos), C, C, 3), np.float16)
    for i, (y0, x0) in enumerate(pos):
        six = np.concatenate([taps[y0:y0 + C, x0:x0 + C], e[y0:y0 + C, x0:x0 + C]], 2).astype(np.float64).reshape(-1, 6, 3)
        yy = np.asarray(y[y0:y0 + C, x0:x0 + C]).astype(np.float64).reshape(-1, 3)
        w = solve(six, yy)[0]
        res[i] = np.einsum("nk,nkc->nc", w, six).reshape(C, C, 3).astype(np.float16)
    np.save(f"{OUT}/{k:02d}.npy", res)
    return k


if __name__ == "__main__":
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 3
    os.makedirs(OUT, exist_ok=True)
    pairs = json.load(open(f"{ROOT}/pairs.json"))
    tr = [k for k, p in enumerate(pairs) if p["split"] == "train"]
    rng = np.random.default_rng(144)
    plan = {k: [(int(rng.integers(0, H - C + 1)), int(rng.integers(0, W - C + 1))) for _ in range(NC)] for k in tr}
    json.dump({str(k): v for k, v in plan.items()}, open(f"{OUT}/crops.json", "w"))
    t0 = time.time()
    todo = [(k, plan[k]) for k in tr if not os.path.exists(f"{OUT}/{k:02d}.npy")]
    with Pool(procs) as pool:
        for k in pool.imap_unordered(job, todo):
            print(k, "%.0fs" % (time.time() - t0), flush=True)
