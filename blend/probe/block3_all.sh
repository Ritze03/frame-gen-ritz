#!/bin/sh
# trains the six block nets in sequence (task 147); logs in .tl-scratch/NAME.log
cd "$(dirname "$0")"
for n in b-stats b-ctx b-conv b-ctx-frame b-conv-big b-mean; do
  sh block3_run.sh block3_train.py $n --steps ${STEPS:-2500} > .tl-scratch/$n.log 2>&1
done
