#!/bin/zsh
cd /home/user/repos/personal/analysis/drift_sel
running() { ps -eo args | awk '$1 ~ /python/ && $0 ~ /run_headless.py/' | wc -l; }
while read -r job; do
  while [ "$(running)" -ge 5 ]; do sleep 30; done
  ./run_any.sh "$job" &
  sleep 20
done < jobs_skip.txt
wait
