#!/bin/sh
cd "$(dirname "$0")"
while ! grep -q "preread done" /home/mo/DieGanzeSau/AI_Video_Training_Data/blend/edge/preread.log; do sleep 5; done
sh ./sweep_c2.sh
