#!/bin/zsh
cd /home/user/repos/personal/analysis/reach_wall
running() { ps -eo args | awk '$1 ~ /python/ && $0 ~ /run_headless.py/' | wc -l; }
while read -r job; do
  while [ "$(running)" -ge 5 ]; do sleep 30; done
  (cd $job && env PYTHONPATH=/home/user/repos/ml_genn/ml_genn SNAKE_CONFIG=$PWD/config.json MPLBACKEND=Agg OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BUILD_JOBS=2 nice -n 5 /home/user/repos/personal/derivations/network_exp/.venv/bin/python -u /home/user/repos/ml_genn/examples/eprop/run_headless.py > log.txt 2>&1; echo "reach_wall/$job done $(date +%H:%M)" >> /home/user/repos/personal/analysis/drift_sel/progress.txt) &
  sleep 20
done < jobs.txt
wait
