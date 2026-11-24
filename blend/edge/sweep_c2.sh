#!/bin/sh
# one intermediate strength (share about 46 %), added after the first sweep
cd "$(dirname "$0")"
O=/home/mo/DieGanzeSau/AI_Video_Training_Data/blend/edge
sh ./py.sh train_edge.py $O/nets/C_lam1.0.npz --init ../weights/forza-rx9070xt/C.npz --lam 1.0 > $O/C_lam1.0.log 2>&1
sh ./py.sh score_edge.py C45=$O/nets/C_lam1.0.npz > $O/score_c45.log 2>&1
