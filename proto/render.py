"""Render a video made only of interpolated frames, for judging interpolation quality by eye.

Simulates a 60 fps game doubled to 120 on 120 fps footage: for source frames f[i], f[i+2]
(i = start, start+2, ...) it generates interpolate(f[i], f[i+2]) and writes just those
generated frames as a 60 fps video, so it plays in real time and every frame shown is
a generated one. --compare puts the real in-between frame f[i+1] on the right.

  python proto/render.py footage/obs1.mp4 --start 120 --duration 30 --module interp_v0 -o review/v0_obs1.mp4

--mult N (multi-frame generation, interp.interpolate_multi): every N-th source frame is treated as real; for
each pair f[i], f[i+N] the N-1 generated frames go to -o at 120 (N-1)/N fps (real time, generated frames only),
and --full writes the stream a player would see, f[i] followed by its generated frames, at 120 fps.
--duration is seconds of source footage consumed (for N=2 the same as seconds of 60 fps output).

  python proto/render.py footage/obs2.mp4 --start 52 --mult 3 -o review/multi/gen_3x_obs2.mp4 --full review/multi/full_3x_obs2.mp4
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


def _work(f):
    a, c = f[0], f[-1]
    ms = [_mod.interpolate(a, c)] if len(f) == 3 else _mod.interpolate_multi(a, c, len(f) - 1)
    # duplicated source frame (game slower than the recording) -> timing of this window is fake
    dup = min(cv2.absdiff(x, y).mean() for x, y in zip(f, f[1:])) <= 0.2
    return ms, dup


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
    ap.add_argument("--duration", type=float, default=30.0, help="seconds of source footage (at 120 fps)")
    ap.add_argument("--mult", type=int, default=2, help="frame generation factor (2 = 60->120, 3 = 40->120, 4 = 30->120)")
    ap.add_argument("--module", default="interp", help="interpolator module in proto/ (e.g. interp_v0)")
    ap.add_argument("--compare", action="store_true", help="side by side: generated | real in-between frame")
    ap.add_argument("--full", help="also write the full stream (real + generated frames) at 120 fps here")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--crf", type=int, default=14)
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    N = args.mult
    w, h = probe(args.clip)
    pairs = round(args.duration * 120 / N)   # real pairs wanted
    n = pairs * (N - 1)                      # generated frames wanted
    gen_fps = 120 * (N - 1) / N
    ow = w * 2 if args.compare else w

    def encoder(path, fps, width):
        return subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{h}",
                                 "-r", f"{fps:g}", "-i", "-", "-vf", "scale=out_color_matrix=bt709:out_range=tv",
                                 "-c:v", "libx264", "-crf", str(args.crf), "-preset", "medium", "-pix_fmt", "yuv420p",
                                 "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                                 "-movflags", "+faststart", path], stdin=subprocess.PIPE)

    enc = encoder(args.out, gen_fps, ow)
    full = encoder(args.full, 120, w) if args.full else None

    t0 = time.perf_counter()
    done = dups = 0
    pending = collections.deque()
    limit = 3 * args.jobs                    # bounded in-flight work, so we never buffer the whole clip

    def drain(block):
        nonlocal done, dups
        while pending and (block or pending[0][0].ready()):
            res, f = pending.popleft()
            ms, dup = res.get()
            dups += dup
            if full:
                full.stdin.write(f[0].tobytes())
            for k, m in enumerate(ms, 1):
                enc.stdin.write((np.hstack([m, f[k]]) if args.compare else m).tobytes())
                if full:
                    full.stdin.write(m.tobytes())
                done += 1
            if done % 60 < N - 1 or done == n:
                el = time.perf_counter() - t0
                print(f"\r{done}/{n} frames  {done / el:5.1f} fps  eta {(n - done) * el / done:4.0f}s",
                      end="", flush=True)
            if block and len(pending) < limit:
                return

    with mp.Pool(args.jobs, _init, (args.module,)) as pool:
        src = frames(args.clip, args.start, N * pairs + 1, w, h)
        prev = next(src, None)
        while prev is not None and done + len(pending) * (N - 1) < n:
            f = [prev] + [next(src, None) for _ in range(N)]
            if f[-1] is None:
                break
            pending.append((pool.apply_async(_work, (f,)), f if (args.compare or full) else None))
            prev = f[-1]
            if len(pending) >= limit:
                drain(True)
            drain(False)
        while pending:
            drain(True)

    enc.stdin.close()
    enc.wait()
    if full:
        full.stdin.close()
        full.wait()
    el = time.perf_counter() - t0
    print(f"\n{args.out}: {done} generated frames ({done / gen_fps:.2f} s at {gen_fps:g} fps) from {args.clip} "
          f"{args.start:.2f}-{args.start + N * done / (N - 1) / 120:.2f} s, {N}x, module {args.module}, "
          f"{dups} windows with a duplicated source frame, {el:.0f} s total ({done / el:.1f} fps)")
    if full:
        print(f"{args.full}: {done // (N - 1) * N} frames at 120 fps (real + generated)")
    if done < n:
        print(f"warning: clip ended early, wanted {n} frames", file=sys.stderr)
    return enc.returncode or (full.returncode if full else 0)


if __name__ == "__main__":
    sys.exit(main())
