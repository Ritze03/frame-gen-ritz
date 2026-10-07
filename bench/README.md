# bench

Testing suite for frame-generation v2. Separate from `gpu/` and `proto/`. Everything below is planned, not built yet.

## bench/score/ (future)

Numeric bench driver. Rebuilds the `gpu` fgtest with `CCACHE_DISABLE=1` and prints its git hash. Samples ~200 seeded triples per >=90 fps clip from `footage/`, holds out the middle frame, and scores v1 vs v2 presets as paired per-triple deltas against v1. Also reports flicker, a missing-content proxy, fallback rate (global fallback off) and worst-spot crops.

## bench/blind/ (future)

Blind-test page. A session builder plus a static local HTML "scoring prompt builder": shows contestants unlabelled in random order (the real frame as a hidden contestant where one exists) and ends with a prompt to paste back to the lead, holding the votes and the reveal.

## Rules

- The core mask is enforced system-wide; the tools cap threads at 8 (`ffmpeg -threads 8`, `OMP_NUM_THREADS=8`, `ninja -j8`).
- Work dirs live under `bench/` on /home, never /tmp (it is a 16 GB RAM disk).
- Footage lives in `footage/` (gitignored). `work/`, `out/` and `blind/sessions/` are gitignored too.
