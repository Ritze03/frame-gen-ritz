#!/bin/sh
# runs a blend/edge script with the ml-fg python (cwd = this folder)
export TMPDIR=$HOME/tmp
cd "$(dirname "$0")"
exec /home/mo/Schinken/Programming/Local/ml-fg/.venv/bin/python "$@"
