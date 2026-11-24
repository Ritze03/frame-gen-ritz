#!/bin/sh
# regularised retries (weight decay 1e-4, dropout 0.1) and extra runs, one after the other (task 147)
cd "$(dirname "$0")"
for n in b-stats b-ctx b-conv; do
  sh block3_run.sh block3_train.py $n --steps 2500 --wd 1e-4 --drop 0.1 --tag $n-reg > .tl-scratch/$n-reg.log 2>&1
done
sh block3_run.sh block3_train.py b-stats --steps 2500 --loss char --tag b-stats-char > .tl-scratch/b-stats-char.log 2>&1
sh block3_run.sh block3_train.py b-stats --steps 2500 --seed 1 --tag b-stats-s1 > .tl-scratch/b-stats-s1.log 2>&1
