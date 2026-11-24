#!/bin/sh
# step 1 + baselines: checks, per-pixel rule scores, baselines (task 147)
cd "$(dirname "$0")"
export TMPDIR=$HOME/tmp
PY=/home/mo/Schinken/Programming/Local/ml-fg/.venv/bin/python
$PY block3_check.py 2>&1 | grep -v "^rule per pixel"
$PY block3_rulepix.py > .tl-scratch/rulepix.log 2>&1 &
$PY block3_base.py
wait
tail -1 .tl-scratch/rulepix.log
