#!/bin/sh
cd "$(dirname "$0")"
O=/home/mo/DieGanzeSau/AI_Video_Training_Data/blend/edge
while [ ! -f $O/scores/forzaC.json ]; do sleep 5; done
sh ./py.sh score_edge.py C0=$O/nets/C_lam0.npz C1=$O/nets/C_lam0.1306.npz C3=$O/nets/C_lam0.5037.npz C6=$O/nets/C_lam1.763.npz > $O/score_c.log 2>&1
