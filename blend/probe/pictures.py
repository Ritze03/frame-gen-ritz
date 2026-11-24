"""Side-by-side PNGs for 3 held-out racing pairs: real | rule | px8 | best neighbourhood net | O6, cropped to the 640x360 window where the rule's error is largest.
usage: pictures.py  -> $BLEND/probe/img/<id>.png  (the best neighbourhood net = highest whole-picture PSNR in scores.json among the four 3x3 / 5x5 nets)"""
import json, os, struct, sys, zlib
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nets import *
from score import psnr

BLEND = os.environ.get("BLEND", "/home/mo/DieGanzeSau/AI_Video_Training_Data/blend")
ROOT = f"{BLEND}/probe"
torch.set_num_threads(4)


def png(path, a):
    a = np.ascontiguousarray(a, np.uint8)
    h, w, _ = a.shape
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 3)) + chunk(b"IEND", b""))


def main():
    sc = json.load(open(f"{ROOT}/scores.json"))
    cand = ["n3-all", "n5-all", "n3-frames", "n3-taps"]
    tot = {m: sum(v["scores"][m]["se"] for v in sc.values()) for m in cand if m in next(iter(sc.values()))["scores"]}
    best = min(tot, key=tot.get)
    print("best neighbourhood net", best)
    nets = {}
    for n in ("px8", best):
        nets[n] = Net(n)
        nets[n].load_state_dict(torch.load(f"{ROOT}/nets/{n}.pt"))
    forza = [int(k) for k, v in sc.items() if v["pair"]["group"] == "forza"]
    pick = [forza[int(i)] for i in np.linspace(1, len(forza) - 2, 3)]
    os.makedirs(f"{ROOT}/img", exist_ok=True)
    for k in pick:
        d = f"{ROOT}/pairs/{k:02d}"
        y = np.load(f"{d}/y.npy").astype(np.float32)
        taps = np.load(f"{d}/taps.npy").reshape(H, W, 4, 3).astype(np.float32)
        rule = np.einsum("hwk,hwkc->hwc", np.load(f"{d}/rule.npy").astype(np.float32), taps)
        pr = Pair(ROOT, k)
        px, nb = predict(nets["px8"], pr), predict(nets[best], pr)
        o6 = np.load(f"{d}/o6.npy").astype(np.float32)
        se = ((rule - y) ** 2).sum(-1)
        ii = np.pad(se.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        win = ii[360:, 640:] - ii[:-360, 640:] - ii[360:, :-640] + ii[:-360, :-640]   # [H-359, W-639]
        y0, x0 = np.unravel_index(win.argmax(), win.shape)
        cl = lambda a: np.clip(np.rint(a[y0:y0 + 360, x0:x0 + 640]), 0, 255)
        row = np.concatenate([cl(a) for a in (y, rule, px, nb, o6)], 1)
        name = pairs_id(sc, k)
        png(f"{ROOT}/img/{name}.png", row)
        print(name, "window y0 x0", y0, x0, "panels: real | rule | px8 |", best, "| O6")


def pairs_id(sc, k):
    return sc[str(k)]["pair"]["id"]


if __name__ == "__main__":
    main()
