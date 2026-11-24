#!/bin/sh
# usage: ab.sh LAM   nets A and B (four choices) with the chosen strength, then scores of forza-rx9070xt A / B and the new ones
cd "$(dirname "$0")"
O=/home/mo/DieGanzeSau/AI_Video_Training_Data/blend/edge
L=$1
for N in A B; do
  sh ./py.sh train_edge.py $O/nets/${N}_lam$L.npz --init ../weights/forza-rx9070xt/$N.npz --lam $L > $O/${N}_lam$L.log 2>&1
done
sh ./py.sh score_edge.py forzaA=../weights/forza-rx9070xt/A.npz forzaB=../weights/forza-rx9070xt/B.npz newA=$O/nets/A_lam$L.npz newB=$O/nets/B_lam$L.npz > $O/score_ab.log 2>&1
