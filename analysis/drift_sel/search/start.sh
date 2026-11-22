#!/bin/zsh
# usage: start.sh OUT_DIR [search args]
cd /home/user/repos/personal/analysis/drift_sel/search
out=$1; shift
mkdir -p $out
env PYTHONPATH=/home/user/repos/ml_genn/ml_genn OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 5 /home/user/repos/personal/derivations/network_exp/.venv/bin/python -u drift_arch_search.py $out "$@" > $out/search.log 2>&1
