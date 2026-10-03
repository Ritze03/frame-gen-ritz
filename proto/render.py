"""Render a video made only of interpolated frames, for judging interpolation quality by eye.

Simulates a 60 fps game doubled to 120 on 120 fps footage: for source frames f[i], f[i+2]
(i = start, start+2, ...) it generates interpolate(f[i], f[i+2]) and writes just those
generated frames as a 60 fps video, so it plays in real time and every frame shown is
a generated one. --compare puts the real in-between frame f[i+1] on the right.

  python proto/render.py footage/obs1.mp4 --start 120 --duration 30 --module interp_v0 -o review/v0_obs1.mp4
"""
import argparse
import collections
import importlib
import multiprocessing as mp
import os
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_mod = None


def _init(name):
    global _mod
    cv2.setNumThreads(1)  # one process per core already
    _mod = importlib.import_module(name)


def _work(a, b, c):
    m = _mod.interpolate(a, c)
    # duplicated source frame (game slower than the recording) -> timing of this triplet is fake
    dup = min(cv2.absdiff(a, b).mean(), cv2.absdiff(b, c).mean()) <= 0.2
    return m, dup


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout
    w, h = map(int, out.strip().split(",")[:2])
    return w, h


def frames(path, start, count, w, h):
    """Exactly-seeked sequential decode via ffmpeg (-ss before -i decodes from the previous
    keyframe and discards up to `start`, so it is frame-accurate, unlike CAP_PROP_POS_FRAMES on HEVC)."""
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{start:.4f}", "-i", path, "-frames:v", str(count),
                          "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         stdout=subprocess.PIPE)
    size = w * h * 3
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                return
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        p.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--duration", type=float, default=30.0, help="seconds of output (generated frames / 60)")
    ap.add_argument("--module", default="interp", help="interpolator module in proto/ (e.g. interp_v0)")
    ap.add_argument("--compare", action="store_true", help="side by side: generated | real middle frame")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--crf", type=int, default=14)
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    w, h = probe(args.clip)
    n = round(args.duration * 60)            # generated frames wanted
    ow = w * 2 if args.compare else w
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{h}",
                            "-r", "60", "-i", "-", "-vf", "scale=out_color_matrix=bt709:out_range=tv",
                            "-c:v", "libx264", "-crf", str(args.crf), "-preset", "medium", "-pix_fmt", "yuv420p",
                            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                            "-movflags", "+faststart", args.out], stdin=subprocess.PIPE)

    t0 = time.perf_counter()
    done = dups = 0
    pending = collections.deque()
    limit = 3 * args.jobs                    # bounded in-flight work, so we never buffer the whole clip

    def drain(block):
        nonlocal done, dups
        while pending and (block or pending[0][0].ready()):
            res, mid = pending.popleft()
            m, dup = res.get()
            dups += dup
            enc.stdin.write((np.hstack([m, mid]) if args.compare else m).tobytes())
            done += 1
            if done % 60 == 0 or done == n:
                el = time.perf_counter() - t0
                print(f"\r{done}/{n} frames  {done / el:5.1f} fps  eta {(n - done) * el / done:4.0f}s",
                      end="", flush=True)
            if block and len(pending) < limit:
                return

    with mp.Pool(args.jobs, _init, (args.module,)) as pool:
        src = frames(args.clip, args.start, 2 * n + 1, w, h)
        prev = next(src, None)
        while prev is not None and done + len(pending) < n:
            b, c = next(src, None), next(src, None)
            if c is None:
                break
            pending.append((pool.apply_async(_work, (prev, b, c)), b if args.compare else None))
            prev = c
            if len(pending) >= limit:
                drain(True)
            drain(False)
        while pending:
            drain(True)

    enc.stdin.close()
    enc.wait()
    el = time.perf_counter() - t0
    print(f"\n{args.out}: {done} generated frames ({done / 60:.2f} s at 60 fps) from {args.clip} "
          f"{args.start:.2f}-{args.start + 2 * done / 120:.2f} s, module {args.module}, "
          f"{dups} triplets with a duplicated source frame, {el:.0f} s total ({done / el:.1f} fps)")
    if done < n:
        print(f"warning: clip ended early, wanted {n} frames", file=sys.stderr)
    return enc.returncode


if __name__ == "__main__":
    sys.exit(main())
