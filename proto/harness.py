"""Score frame interpolation on real footage.

From a high-fps clip take frames (a, b, c), `stride` apart, predict b from a and c, compare
with the real b. stride=1 on 120 fps footage simulates 60 -> 120, stride=2 simulates 30 -> 60.

  dup    = show a again (what you see without frame generation)
  blend  = 50/50 crossfade of a and c
  interp = interp.interpolate(a, c)

harm% = pixels where interp is far off from *both* real frames, i.e. invented garbage
(ghosts, tears) rather than content that's just slightly misplaced.
"""
import argparse
import collections
import pathlib
import time

import cv2
import numpy as np

import interp

HARM_MARGIN = 24  # 8-bit levels beyond the worse of a / c before a pixel counts as harm
DUP_DIFF = 0.6  # re-encoded duplicate frames differ by up to ~0.5 (mean abs diff)


def err(x, y):
    return np.abs(x.astype(np.int16) - y).mean(2)


def triplets(path, stride, every, scale):
    cap = cv2.VideoCapture(str(path))
    win = collections.deque(maxlen=2 * stride + 1)
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            return
        if scale != 1:
            f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        win.append(f)
        i += 1
        if len(win) == win.maxlen and i % every == 0:
            # a duplicated frame (game slower than the recording) would fake the timing: skip
            if all(err(win[k], win[k + 1]).mean() > DUP_DIFF for k in range(len(win) - 1)):
                yield i, win[0], win[stride], win[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clips", nargs="+", type=pathlib.Path)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--every", type=int, default=15, help="test one triplet per N frames")
    p.add_argument("--scale", type=float, default=1.0, help="downscale footage first (faster runs)")
    p.add_argument("--max", type=int, default=200, help="max triplets per clip")
    p.add_argument("--worst", type=int, default=5, help="save the N most harmful triplets as PNG")
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("out"))
    args = p.parse_args()
    args.out.mkdir(exist_ok=True)

    print(f"{'clip':24} {'n':>4} {'dup dB':>7} {'blend dB':>8} {'interp dB':>9} {'worse%':>6} {'harm%':>6} {'ms':>6}")
    for clip in args.clips:
        rows, worst = [], []
        for idx, a, b, c in triplets(clip, args.stride, args.every, args.scale):
            t = time.perf_counter()
            m = interp.interpolate(a, c)
            ms = (time.perf_counter() - t) * 1000
            harm = float((err(m, b) > np.maximum(err(a, b), err(c, b)) + HARM_MARGIN).mean() * 100)
            rows.append((cv2.PSNR(a, b), cv2.PSNR(cv2.addWeighted(a, 0.5, c, 0.5, 0), b), cv2.PSNR(m, b), harm, ms))
            worst = sorted(worst + [(harm, idx, a, b, c, m)], key=lambda w: -w[0])[: args.worst]
            if len(rows) >= args.max:
                break
        if not rows:
            print(f"{clip.name:24} no usable triplets (too short, or every frame duplicated?)")
            continue
        r = np.array(rows)
        worse = (r[:, 2] < r[:, 0]).mean() * 100  # triplets where interp scored below plain dup
        print(f"{clip.name[:24]:24} {len(r):4} {r[:, 0].mean():7.2f} {r[:, 1].mean():8.2f} {r[:, 2].mean():9.2f} "
              f"{worse:6.1f} {r[:, 3].mean():6.2f} {r[:, 4].mean():6.0f}")
        for harm, idx, a, b, c, m in worst:
            heat = cv2.applyColorMap(np.clip(err(m, b) * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            grid = np.vstack([np.hstack([a, c]), np.hstack([b, m]), np.hstack([heat, cv2.addWeighted(a, 0.5, c, 0.5, 0)])])
            cv2.imwrite(str(args.out / f"{clip.stem}_f{idx}_harm{harm:.1f}.png"), grid)


if __name__ == "__main__":
    main()
