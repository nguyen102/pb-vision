#!/usr/bin/env bash
# Writes a progress summary of a training chain every minute, so long runs can be checked at a glance.
#   training/logs/STATUS.txt         current step, progress, time since the active log last changed, state
#   training/logs/status-events.log  one line per state change (step started, STALLED, watchdog restart, DONE, FAILED)
# State is STALLED when the active log hasn't changed for --stall seconds (default 300; the watchdogs restart
# a job after 600 s with no progress, so a STALLED event warns before that happens).
# Usage: training/status.sh [chain-output, default training/logs/round2.out] [stall-seconds]
cd "$(dirname "$0")/.."
chain=${1:-training/logs/round2.out}
stall=${2:-300}
logs=(training/logs/scan-train-r1.log training/logs/labels-r1.log training/logs/train-r2.log training/logs/scan-test-r2.log)
status=training/logs/STATUS.txt
events=training/logs/status-events.log
prev_state="" prev_step="" prev_restarts=0

progress() {  # one-line progress for a log file
  case "$1" in
    *scan-*)
      tail -1 "$1" | awk -F'[/ ,]+' '/frames,/ {printf "%s (%.0f%% of video)\n", $0, 100*$1/$2; next} {print}' ;;
    *train-*)
      local t; t=$(tr '\r' '\n' < "$1" | sed 's/\x1b\[[0-9;]*[A-Za-z]//g')
      local ep; ep=$(grep -aoE "^ +[0-9]+/[0-9]+ " <<<"$t" | tail -1 | tr -d ' ')
      local it; it=$(grep -aoE "[0-9]+/[0-9]+ [0-9.]+it/s" <<<"$t" | tail -1)
      local val; val=$(grep -aE "^ +all +[0-9]" <<<"$t" | tail -1 | awk '{print "P " $4 "  R " $5 "  mAP50 " $6}')
      echo "epoch ${ep:-?} (batch ${it:-?}); last val: ${val:-none yet}" ;;
    *) tail -1 "$1" ;;
  esac
}

while true; do
  now=$(date +%s)
  step=$(grep -aE "^\[[0-9:]+\]" "$chain" 2>/dev/null | tail -1)
  active="" newest=0
  for l in "${logs[@]}"; do
    [ -f "$l" ] && m=$(stat -c %Y "$l") && [ "$m" -gt "$newest" ] && newest=$m active=$l
  done
  age=$(( now - newest ))
  restarts=$(cat "${logs[@]}" 2>/dev/null | grep -ac "^watchdog: no progress")
  if grep -q "ROUND2 DONE" "$chain" 2>/dev/null; then state=DONE
  elif grep -qE "FAILED|giving up" "$chain" 2>/dev/null; then state=FAILED
  elif ! pgrep -f "$(basename "${chain%.out}").sh" >/dev/null; then state=NOT-RUNNING
  elif [ "$age" -gt "$stall" ]; then state=STALLED
  else state=OK; fi
  {
    echo "updated:   $(date '+%Y-%m-%d %H:%M:%S')"
    echo "state:     $state"
    echo "step:      ${step:-?}"
    echo "log:       ${active:-none} (changed ${age}s ago)"
    [ -n "$active" ] && echo "progress:  $(progress "$active")"
    echo "restarts:  $restarts watchdog restart(s)"
    timeout 15 nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null \
      | sed 's/^/gpu:       /' || echo "gpu:       nvidia-smi not responding"
  } > "$status.tmp" && mv "$status.tmp" "$status"
  ts=$(date '+%H:%M')
  [ "$step" != "$prev_step" ] && echo "$ts STEP $step" >> "$events"
  [ "$restarts" -gt "$prev_restarts" ] && echo "$ts RESTART watchdog restarted a hung job ($restarts so far)" >> "$events"
  if [ "$state" != "$prev_state" ]; then
    echo "$ts STATE $state - $(grep -a '^progress' "$status" | cut -c12-)" >> "$events"
  fi
  prev_state=$state prev_step=$step prev_restarts=$restarts
  case $state in DONE|FAILED|NOT-RUNNING) exit 0 ;; esac
  sleep 60
done
