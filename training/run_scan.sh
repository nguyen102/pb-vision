#!/usr/bin/env bash
# Run training/scan.py with a watchdog: if the log shows no progress for 10 minutes (a GPU driver reset
# under WSL can leave the process spinning forever), kill it and restart; scan.py resumes from its checkpoint.
# Usage: training/run_scan.sh LOGFILE scan.py-args...
log=$1; shift
cd "$(dirname "$0")/.."
for attempt in 1 2 3 4 5 6; do
  touch "$log"  # the watchdog clock starts now, not when the log was created
  .venv/bin/python training/scan.py "$@" >> "$log" 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 30
    if [ $(( $(date +%s) - $(stat -c %Y "$log") )) -gt 600 ]; then
      echo "watchdog: no progress for 10 min, restarting (attempt $attempt)" >> "$log"
      kill -9 $pid; break
    fi
  done
  wait $pid 2>/dev/null && exit 0
  grep -q "^done:" "$log" && exit 0
done
echo "watchdog: giving up" >> "$log"; exit 1
