#!/bin/sh
# pulls the training pairs' arrays into the page cache (sequential reads)
R=/home/mo/DieGanzeSau/AI_Video_Training_Data/blend/probe/pairs
for k in $(seq -w 0 55); do cat $R/$k/X.npy $R/$k/rule.npy $R/$k/y.npy > /dev/null; done
echo preread done
