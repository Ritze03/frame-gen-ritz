"""Score frame interpolation on real footage.

From a high-fps clip take frames (a, b, c), `stride` apart, predict b from a and c, compare
with the real b. stride=1 on 120 fps footage simulates 60 -> 120, stride=2 simulates 30 -> 60.

--mult N (multi-frame generation): window f[0], f[stride], ..., f[N*stride]; a = f[0] and c = f[N*stride]
are real, the N-1 frames in between are generated at t = k/N and each is scored against the real f[k*stride].
On 120 fps footage N=3 simulates 40 -> 120, N=4 30 -> 120. --mult 2 is the original triplet test.

  dup    = show the nearer real frame (a for t <= 0.5, else c; what you see without frame generation)
  blend  = time-weighted crossfade (1-t) a + t c
  interp = interp.synth(a, c, interp.estimate(a, c), t)
  fb%    = generated frames where a global fallback (scene cut / too much untrusted) held a real frame

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


def windows(path, mult, stride, every, scale):
    cap = cv2.VideoCapture(str(path))
    win = collections.deque(maxlen=mult * stride + 1)
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
                yield i, [win[k * stride] for k in range(mult + 1)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clips", nargs="+", type=pathlib.Path)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--mult", type=int, default=2, help="frame generation factor: N-1 generated frames per real pair")
    p.add_argument("--every", type=int, default=15, help="test one window per N frames")
    p.add_argument("--scale", type=float, default=1.0, help="downscale footage first (faster runs)")
    p.add_argument("--max", type=int, default=200, help="max windows per clip")
    p.add_argument("--worst", type=int, default=5, help="save the N most harmful generated frames as PNG")
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("out"))
    args = p.parse_args()
    args.out.mkdir(exist_ok=True)
    N = args.mult
    ts = [k / N for k in range(1, N)]

    print(f"{'clip':24} {'t':>5} {'n':>4} {'dup dB':>7} {'blend dB':>8} {'interp dB':>9} {'worse%':>6} {'harm%':>6} "
          f"{'fb%':>5} {'ms':>6}")
    for clip in args.clips:
        rows, worst, ms_est, ms_syn = [], [], [], []
        for idx, f in windows(clip, N, args.stride, args.every, args.scale):
            a, c = f[0], f[-1]
            t0 = time.perf_counter()
            est = interp.estimate(a, c)
            ms_est.append((time.perf_counter() - t0) * 1000)
            for k, t in enumerate(ts, 1):
                b = f[k]
                t0 = time.perf_counter()
                m, why = interp.synth(a, c, est, t, why=True)
                ms_syn.append((time.perf_counter() - t0) * 1000)
                near = a if t <= 0.5 else c
                harm = float((err(m, b) > np.maximum(err(a, b), err(c, b)) + HARM_MARGIN).mean() * 100)
                rows.append((k, cv2.PSNR(near, b), cv2.PSNR(cv2.addWeighted(a, 1 - t, c, t, 0), b), cv2.PSNR(m, b),
                             harm, why is not None))
                worst = sorted(worst + [(harm, idx, k, a, b, c, m)], key=lambda w: -w[0])[: args.worst]
            if len(ms_est) >= args.max:
                break
        if not rows:
            print(f"{clip.name:24} no usable windows (too short, or every frame duplicated?)")
            continue
        r = np.array(rows)
        ms = np.mean(ms_est) + np.mean(ms_syn) * len(ts)  # per real pair
        groups = [(f"{t:.2f}", r[r[:, 0] == k]) for k, t in enumerate(ts, 1)] + ([("mean", r)] if N > 2 else [])
        for name, g in groups:
            worse = (g[:, 3] < g[:, 1]).mean() * 100  # generated frames where interp scored below plain dup
            print(f"{clip.name[:24]:24} {name:>5} {len(g):4} {g[:, 1].mean():7.2f} {g[:, 2].mean():8.2f} {g[:, 3].mean():9.2f} "
                  f"{worse:6.1f} {g[:, 4].mean():6.2f} {g[:, 5].mean() * 100:5.1f} {ms:6.0f}")
        print(f"{'':24} windows {len(ms_est)}, ms per pair: estimate {np.mean(ms_est):.0f} + {len(ts)} x synth "
              f"{np.mean(ms_syn):.0f}")
        for harm, idx, k, a, b, c, m in worst:
            heat = cv2.applyColorMap(np.clip(err(m, b) * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            t = ts[k - 1]
            grid = np.vstack([np.hstack([a, c]), np.hstack([b, m]), np.hstack([heat, cv2.addWeighted(a, 1 - t, c, t, 0)])])
            cv2.imwrite(str(args.out / f"{clip.stem}_x{N}_f{idx}_t{k}of{N}_harm{harm:.1f}.png"), grid)


if __name__ == "__main__":
    main()
