#!/bin/zsh
cd /home/user/repos/personal/analysis/drift_sel/search
mkdir -p species_mexhat2
env PYTHONPATH=/home/user/repos/ml_genn/ml_genn OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 5 /home/user/repos/personal/derivations/network_exp/.venv/bin/python -u species_search.py species_mexhat2 --hours 10 --max 60 --parallel 2 --moves 1200 --train-steps 200000 > species_mexhat2/search.log 2>&1
