#!/bin/zsh
cd /home/user/repos/personal/derivations/network_toy
# wait for the Adam rewiring pair to exit (6-job CPU cap)
while ps -eo pid,args | awk '/run_headless.py/ && !/awk|thermal/ {print $1}' | while read p; do readlink /proc/$p/cwd; done | grep -q "/rewire_s"; do sleep 60; done
PY=/home/user/repos/personal/derivations/network_exp/.venv/bin/python
(OMP_NUM_THREADS=1 nice -n 5 $PY -I net_toy.py --variants prop,reservoir --seeds 4 --env-steps 20000 --block 2000 --opt adam > sparse_test/dense.log 2>&1; cp net_toy_last.npz sparse_test/dense.npz) &
sleep 5
(OMP_NUM_THREADS=1 nice -n 5 $PY -I net_toy.py --variants prop,reservoir --seeds 4 --env-steps 20000 --block 2000 --opt adam --in-rate 0.03 --rep 30 > sparse_test/sparse_many.log 2>&1) &
wait
