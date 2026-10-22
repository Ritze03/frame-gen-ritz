#!/usr/bin/env python3
"""HDR test pairs (webgl/tests/hdr.mjs) from the 8-bit pairs of pairs.py: each pair converted to RGBA half floats in two encodings,
written as $FG_WORK/hdr/<name>_<enc>/{a,c}.f16 (W*H*4 float16, little endian, row 0 = top) + manifest.json:
  lin   scRGB-like linear light: sRGB EOTF of the 8-bit values x LIN_GAIN (values up to 3.0 = 240 nits at 80 nits per 1.0: HDR
        highlights above 1.0); the engine / library measure it as PQ codes (transfer 'linear', linearWhiteNits 80)
  pq    PQ-encoded (SMPTE ST 2084) values of the same content at PQ_PEAK nits peak; measured as value x 255 (transfer 'encoded')
  id    the 8-bit values / 255 as half floats (transfer 'encoded'): the float path on exactly the Rgba8 content
Usage: hdr_pairs.py [--pairs a,b]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
WORK = os.environ.get("FG_WORK", os.path.join(REPO, ".local-work", "webgl"))
LIN_GAIN = 3.0
PQ_PEAK = 600.0
DEFAULT = "panel,cross,objects,noisy,cut,odd,battlefield,mc_furnace"


def srgb_eotf(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def pq_oetf(nits):
    y = np.clip(nits / 10000.0, 0.0, 1.0)
    p = y ** 0.1593017578125
    return ((0.8359375 + 18.8515625 * p) / (1.0 + 18.6875 * p)) ** 78.84375


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=DEFAULT)
    args = ap.parse_args()
    src = json.load(open(os.path.join(WORK, "pairs", "manifest.json")))
    out = os.path.join(WORK, "hdr")
    man = []
    for p in src["pairs"]:
        if p["name"] not in args.pairs.split(","):
            continue
        W, H = p["width"], p["height"]
        fr = {k: np.fromfile(os.path.join(WORK, "pairs", p["name"], k + ".rgba"), np.uint8).reshape(H, W, 4) for k in "ac"}
        for enc in ("lin", "pq", "id"):
            d = os.path.join(out, f"{p['name']}_{enc}")
            os.makedirs(d, exist_ok=True)
            for k, img in fr.items():
                v = img[..., :3].astype(np.float64) / 255.0
                if enc == "lin":
                    v = srgb_eotf(v) * LIN_GAIN
                elif enc == "pq":
                    v = pq_oetf(srgb_eotf(v) * PQ_PEAK)
                f = np.ones((H, W, 4), np.float16)
                f[..., :3] = v.astype(np.float16)
                f.astype("<f2").tofile(os.path.join(d, k + ".f16"))
            man.append({"name": f"{p['name']}_{enc}", "src": p["name"], "enc": enc, "width": W, "height": H,
                        "transfer": "linear" if enc == "lin" else "encoded"})
            print(f"hdr_pairs: {p['name']}_{enc} {W}x{H}")
    json.dump({"ts": [0.5, 0.25], "pairs": man, "lin_gain": LIN_GAIN, "pq_peak": PQ_PEAK}, open(os.path.join(out, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
