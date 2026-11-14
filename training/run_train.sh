#!/usr/bin/env bash
# Run training/train.py with a watchdog: if the log shows no progress for 10 minutes (a GPU hang under
# WSL leaves it spinning at 100% CPU), kill it and its dataloader workers and resume from last.pt.
# Usage: training/run_train.sh LOGFILE train.py-args...   (must include --name)
log=$1; shift
cd "$(dirname "$0")/.."
for attempt in 1 2 3 4 5 6; do
  touch "$log"
  extra=(); [ $attempt -gt 1 ] && extra=(--resume)
  setsid .venv/bin/python training/train.py "$@" "${extra[@]}" >> "$log" 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 30
    if [ $(( $(date +%s) - $(stat -c %Y "$log") )) -gt 600 ]; then
      echo "watchdog: no progress for 10 min, restarting with --resume (attempt $attempt)" >> "$log"
      kill -9 -- -$pid 2>/dev/null; kill -9 $pid 2>/dev/null; sleep 5; break
    fi
  done
  wait $pid 2>/dev/null
  grep -qa "^done:" "$log" && exit 0
done
echo "watchdog: giving up" >> "$log"; exit 1
