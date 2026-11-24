#!/bin/sh
# sweep for net C (start terms: charb 0.014705, pen 0.012512 -> lam = s/(1-s)*charb/pen)
cd "$(dirname "$0")"
O=/home/mo/DieGanzeSau/AI_Video_Training_Data/blend/edge
mkdir -p $O/nets
for L in 0 0.1306 0.5037 1.763; do
  sh ./py.sh train_edge.py $O/nets/C_lam$L.npz --init ../weights/forza-rx9070xt/C.npz --lam $L > $O/C_lam$L.log 2>&1
done
sh ./py.sh score_edge.py rule=rule baseC=file:baseC forzaC=file:forzaC > $O/score_ref.log 2>&1
