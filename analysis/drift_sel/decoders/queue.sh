#!/bin/zsh
# start each job when fewer than 6 run_headless.py processes are running (counted by argv, not by matching this script)
running() { ps -eo args | awk '$1 ~ /python/ && $0 ~ /run_headless.py/' | wc -l; }
while read -r job; do
  while [ "$(running)" -ge 6 ]; do sleep 30; done
  ./run_one.sh "$job" &
  sleep 20
done < "$1"
wait
