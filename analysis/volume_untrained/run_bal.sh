#!/bin/zsh
for n in vol_n50_bal vol_n50_inh vol_n50_toI; do
  cd /home/user/repos/personal/analysis/volume_untrained/$n && env PYTHONPATH=/home/user/repos/ml_genn/ml_genn CUDA_PATH=/usr/local/cuda SNAKE_CONFIG=$PWD/config.json MPLBACKEND=Agg OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 BUILD_JOBS=4 nice -n 5 /home/user/repos/personal/derivations/network_exp/.venv/bin/python -u /home/user/repos/ml_genn/examples/eprop/run_headless.py > log.txt 2>&1
  echo "$n done $(date +%H:%M) $(ls outputs/*_probe.npz 2>/dev/null | wc -l) probe" >> /home/user/repos/personal/analysis/volume_untrained/progress.txt
done
