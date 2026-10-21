# bench/score

`python3 bench/score/score.py [-c 'v2: algo=v2' ...] [--contestants FILE] [--triples 200] [--test60] [--base-set S] [--warm N]`
rebuilds `gpu/build/fgtest` (CCACHE_DISABLE=1) and records its git hash, then scores every contestant (`name: fgtest --set string`,
one per line in FILE or per `-c`; `v1: algo=v1` is always added as the paired reference) on every clip of the main checkout's
`footage/`: clips of >= 90 fps (genre `fps` at the top level, else from the `<genre>_<game>_<x>` name) hold out every other
frame and are scored against it (paired PSNR / SSIM deltas against v1, harm%, flicker, missing-content proxy); `footage/cuts/`
clips of 50-89 fps get no-reference metrics (60 -> 120: flicker NR); clips under 50 fps are skipped. Triples are sampled as
seeded windows of consecutive pairs (`--warm` + `--pairs`), a fresh Interpolator per window, about `--batch` pairs per fgtest
call. Every contestant runs with the **base set** `--base-set` (default `globalFallback=1,ui=crosshairv2`) applied BEFORE its own
set string (so a contestant can override a key, e.g. `v2-noui: algo=v2,ui=off`): the global fallback off (its would-have-fired
rate is read from the library's counters, `fgtest --stats`) and CrosshairV2 UI protection on, as the user plays; fgtest
recordObserves every real frame of a window once, like a host. The report header prints the base set. **Warm-up**: a fresh
Interpolator per window masks a static pixel only after `uiStillFrames` (8) still frames, i.e. from the 8th pair of a window
(index 7), so with UI protection on `--warm` defaults to `uiStillFrames - 1` = 7 (else 2; v2's field history needs 1-2); an
explicit smaller `--warm` prints a warning. Frames are decoded per window and deleted after scoring in `bench/score/work/` (never /tmp, < 10 GB). Output:
`bench/score/out/<run-id>/report.md` (all definitions included), `per_triple.csv` and `crops/` (prev | v1 | contestant | real |
next strips of the worst sites). `--triples 40 --test60` is the quick self-test (`--test60` adds battlefield's even frames as a
60 fps clip). No thread caps (ffmpeg, OpenCV / OpenMP and ninja use their defaults); CPU limits are system-wide.

**Comparing builds in one run.** `FGTEST=/path/to/fgtest python3 bench/score/score.py ...` uses that binary as is: no
rebuild, and the report records the binary's sha256 instead of the commit (the run id ends in that hash). A contestant may
also name its own binary with a `bin=PATH` key in its set string, e.g.

```sh
FGTEST=builds/a/fgtest python3 bench/score/score.py -c 'v1: algo=v1' -c 'v2new: algo=v2,bin=builds/b/fgtest' --triples 40
```

`bin=` is stripped before `--set`; the reference `v1` may carry one too (`v1: algo=v1,bin=...`). The report's contestant
table names every contestant's binary and its sha256 (the console prints them at the start).
