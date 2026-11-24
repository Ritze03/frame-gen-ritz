#!/bin/bash
# trains the six nets one after the other (waits while less than 6 GB are available)
PY=/home/mo/Schinken/Programming/Local/ml-fg/.venv/bin/python
cd "$(dirname "$0")"
mkdir -p ../../.tl-scratch/logs
for n in px8 px32 n3-all n3-frames n3-taps n5-all; do
  while [ "$(free -g | awk '/Mem:/{print $7}')" -lt 6 ]; do sleep 60; done
  ccorral -- $PY train_probe.py $n > ../../.tl-scratch/logs/$n.log 2>&1
done
echo ALLDONE > ../../.tl-scratch/logs/ALLDONE
