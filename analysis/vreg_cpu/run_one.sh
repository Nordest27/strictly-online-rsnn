#!/bin/zsh
n=$1
cd /home/user/repos/personal/analysis/vreg_cpu/$n && env PYTHONPATH=/home/user/repos/ml_genn/ml_genn SNAKE_CONFIG=$PWD/config.json MPLBACKEND=Agg OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BUILD_JOBS=2 nice -n 5 /home/user/repos/personal/derivations/network_exp/.venv/bin/python -u /home/user/repos/ml_genn/examples/eprop/run_headless.py > log.txt 2>&1
echo "$n done $(date +%H:%M) $(ls outputs/*_probe.npz 2>/dev/null | wc -l) probe" >> /home/user/repos/personal/analysis/vreg_cpu/progress.txt
