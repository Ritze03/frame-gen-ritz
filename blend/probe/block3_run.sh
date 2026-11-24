#!/bin/sh
# usage: block3_run.sh SCRIPT.py ARGS...   (python of the ml-fg venv, TMPDIR in ~/tmp)
cd "$(dirname "$0")"
export TMPDIR=$HOME/tmp
exec /home/mo/Schinken/Programming/Local/ml-fg/.venv/bin/python "$@"
