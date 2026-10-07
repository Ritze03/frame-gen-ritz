# bench/score

`python3 bench/score/score.py [-c 'v2: algo=v2' ...] [--contestants FILE] [--triples 200] [--test60]`
rebuilds `gpu/build/fgtest` (CCACHE_DISABLE=1) and records its git hash, then scores every contestant (`name: fgtest --set string`,
one per line in FILE or per `-c`; `v1: algo=v1` is always added as the paired reference) on every clip of the main checkout's
`footage/`: clips of >= 90 fps (genre `fps` at the top level, else from the `<genre>_<game>_<x>` name) hold out every other
frame and are scored against it (paired PSNR / SSIM deltas against v1, harm%, flicker, missing-content proxy); `footage/cuts/`
clips of 50-89 fps get no-reference metrics (60 -> 120: flicker NR); clips under 50 fps are skipped. Triples are sampled as
seeded windows of consecutive pairs (`--warm` + `--pairs`), a fresh Interpolator per window, about `--batch` pairs per fgtest
call; every contestant runs with `globalFallback=1` and the would-have-fired fallback rate is read from the library's counters
(`fgtest --stats`). Frames are decoded per window and deleted after scoring in `bench/score/work/` (never /tmp, < 10 GB). Output:
`bench/score/out/<run-id>/report.md` (all definitions included), `per_triple.csv` and `crops/` (prev | v1 | contestant | real |
next strips of the worst sites). `--triples 40 --test60` is the quick self-test (`--test60` adds battlefield's even frames as a
60 fps clip). The script re-execs itself under the CPU mask if needed.
