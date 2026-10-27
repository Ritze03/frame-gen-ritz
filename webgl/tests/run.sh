#!/usr/bin/env bash
# The WebGL2 backend's tests, in order (README "Tests"). Every output goes to $FG_WORK (default <repo>/.local-work/webgl); each
# step's full log is $FG_WORK/<step>.log, its summary is printed.
#   webgl/tests/run.sh [step ...]     steps: setup validate golden xbackend xbackend-gpu hdr blur learned learned-gpu bench
#                                     bench-gpu clean (default: all but the -gpu ones and clean)
#   setup     build fgtest (gpu/build-webgl, meson + ninja), the test pairs (pairs.py; footage from $FG_FOOTAGE, default
#             <repo>/footage) and the Vulkan references (vkref.mjs)
#   validate  glslangValidator on every assembled program, all frame-format variants
#   golden    V1 vs proto/interp.py (ref_gen.py + golden.mjs)
#   xbackend  WebGL2 (SwiftShader) vs Vulkan: V1, V2 Performance, V2 Quality at each pair's flow scales (xbackend-gpu: hardware GL,
#             adds the 8K pair)
#   hdr       RGBA16F input (scRGB linear, PQ, identity) vs the library's RgbaF16 / Rgba8
#   blur      V2 synthBlur vs recordSynthBlur
#   learned   the learned motion estimator vs the ml-fg test vectors ($FG_MLFG_EXPORT, default
#             ~/Schinken/Programming/Local/ml-fg/export) + the duplicate skip (learned-gpu: hardware GL)
#   bench     ms per pair at 1080p (bench-gpu: hardware GL, also the learned estimator)
#   clean     delete $FG_WORK (raw frames, references, logs)
# Exit: the first failing step's code.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
export FG_WORK="${FG_WORK:-$repo/.local-work/webgl}"
mkdir -p "$FG_WORK"
steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(setup validate golden xbackend hdr blur learned bench)
run() {   # step lines-of-summary cmd...
  local s="$1" n="$2"; shift 2
  "$@" > "$FG_WORK/$s.log" 2>&1
  local rc=$?
  tail -n "$n" "$FG_WORK/$s.log"
  [ $rc -eq 0 ] || { echo "== $s FAILED ($rc), log: $FG_WORK/$s.log"; exit $rc; }
}
for s in "${steps[@]}"; do
  echo "== $s"
  case "$s" in
    setup)
      run setup-build 3 node "$here/vkref.mjs" --build-only
      run setup-pairs 2 python3 "$here/pairs.py"
      run setup-vkref 1 node "$here/vkref.mjs" ;;
    validate) run validate 1 node "$here/validate.mjs" ;;
    golden) run golden-ref 1 python3 "$here/ref_gen.py"; run golden 9 node "$here/golden.mjs" ;;
    xbackend) run xbackend 14 node "$here/xbackend.mjs" --json "$FG_WORK/xbackend.json" ;;
    xbackend-gpu) run xbackend-gpu 14 node "$here/xbackend.mjs" --gpu --json "$FG_WORK/xbackend-gpu.json" ;;
    hdr) run hdr 2 node "$here/hdr.mjs" ;;
    blur) run blur 1 node "$here/blur.mjs" ;;
    learned) run learned 29 node "$here/learned.mjs" --json "$FG_WORK/learned.json" ;;
    learned-gpu) run learned-gpu 29 node "$here/learned.mjs" --gpu --json "$FG_WORK/learned-gpu.json" ;;
    bench) run bench 4 node "$here/bench.mjs" --fs 2 --iters 10 ;;
    bench-gpu) run bench-gpu 10 node "$here/bench.mjs" --gpu --fs 1,2,4
               run bench-learned-gpu 13 node "$here/bench.mjs" --gpu --modes v1,e2v5,e1v5,e2v4,e1v4,ml2,ml1,e2v5f,e1v5f,e2v4f,e1v4f,ml2f,ml1f ;;
    clean) rm -rf "$FG_WORK" ;;
    *) echo "unknown step $s" >&2; exit 2 ;;
  esac
done
