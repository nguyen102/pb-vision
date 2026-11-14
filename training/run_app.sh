#!/usr/bin/env bash
# Run app.py on a video with a watchdog, or report on such a run.
#
#   training/run_app.sh OUTDIR VIDEO [app.py args...]   start app.py --out OUTDIR (or adopt one already running for
#                                                        OUTDIR) and restart it if it dies or stalls before "done"
#   training/run_app.sh --status OUTDIR                  state, stage progress, log age, restarts
#
# Output goes to OUTDIR/run.log, deleted once the run finishes (its final summary line is kept in
# OUTDIR/watchdog.log, which logs every watchdog action). A run is stalled when run.log hasn't
# changed for STALL_S seconds (default 1800: the clothing-colour pass between tracking and analysis prints nothing
# for several minutes). app.py can't resume mid-tracking, but tracking.json is cached once tracking finishes, so a
# restart after that point goes straight to the later stages. Gives up after MAX_RESTARTS (default 3) restarts.
# Example: nohup training/run_app.sh outputs/10-11-v5 ~/yt-vids/10-11-BBB-mb-tim-cut.mkv --no-pose >/dev/null 2>&1 &
cd "$(dirname "$0")/.."
STALL_S=${STALL_S:-1800}
MAX_RESTARTS=${MAX_RESTARTS:-3}

app_pid() { pgrep -f "^[^ ]*python[0-9.]* app.py .*--out $1( |$)" | head -1; }  # not the shell that launched it
log_age() { echo $(( $(date +%s) - $(stat -c %Y "$1/run.log" 2>/dev/null || date +%s) )); }
finished() { grep -aq "^done (" "$1/run.log" 2>/dev/null; }
# the last start/adopt/finish in watchdog.log was a finish (run.log is gone by then)
recorded_done() { grep -aE " (started|adopted|done:) " "$1/watchdog.log" 2>/dev/null | tail -1 | grep -q " done: "; }
stage() {  # last progress bar or final line of run.log (tqdm rewrites one line with \r), else its last line
  local t; t=$(tail -c 20000 "$1/run.log" 2>/dev/null | tr '\r' '\n' | grep -a .)
  { grep -aE '[0-9]+%\||^done \(' <<<"$t" || echo "$t"; } | tail -1 | cut -c1-110
}

if [ "$1" = "--status" ]; then
  out=${2:?usage: $0 --status OUTDIR}
  pid=$(app_pid "$out")
  if [ -n "$pid" ] && [ "$(log_age "$out")" -gt "$STALL_S" ]; then state=STALLED
  elif [ -n "$pid" ]; then state=RUNNING
  elif finished "$out" || recorded_done "$out"; then state=DONE
  else state=NOT-RUNNING; fi
  watch=$(pgrep -f "run_app.sh $out( |$)" | head -1)
  echo "state:     $state${pid:+ (pid $pid)}"
  if [ -f "$out/run.log" ]; then
    echo "progress:  $(stage "$out")"
    echo "log:       $out/run.log, changed $(log_age "$out")s ago"
  else
    echo "progress:  $(grep -a " done: " "$out/watchdog.log" 2>/dev/null | tail -1 | sed 's/.* done: //' | cut -c1-110)"
    echo "log:       none (deleted when the run finished)"
  fi
  echo "watchdog:  $([ -n "$watch" ] && echo "running (pid $watch)" || echo "not running")"
  echo "restarts:  $(cat "$out/watchdog.log" 2>/dev/null | grep -ac "restarting")"
  exit 0
fi

out=${1:?usage: $0 OUTDIR VIDEO [app.py args...]}; shift
mkdir -p "$out"
wlog=$out/watchdog.log
note() { echo "$(date '+%F %T') $*" >> "$wlog"; }

pid=$(app_pid "$out")
if [ -n "$pid" ]; then  # adopt a run started by hand; restarts reuse its exact arguments
  mapfile -d '' cmd < /proc/"$pid"/cmdline
  cmd=("${cmd[@]:1}")  # drop the interpreter; we run .venv/bin/python ourselves
  note "adopted running pid $pid: ${cmd[*]}"
else
  [ $# -ge 1 ] || { echo "usage: $0 OUTDIR VIDEO [app.py args...]" >&2; exit 2; }
  video=$1; shift
  cmd=(app.py "$video" --out "$out" "$@")
fi

for restarts in $(seq 0 "$MAX_RESTARTS"); do
  if [ -z "$pid" ]; then
    : > "$out/run.log"
    .venv/bin/python "${cmd[@]}" >> "$out/run.log" 2>&1 &
    pid=$!
    note "started pid $pid: ${cmd[*]}"
  fi
  while kill -0 "$pid" 2>/dev/null; do
    sleep ${POLL_S:-60}
    if [ "$(log_age "$out")" -gt "$STALL_S" ]; then
      note "no progress for ${STALL_S}s at: $(stage "$out")"
      kill -9 "$pid"
      break
    fi
  done
  wait "$pid" 2>/dev/null
  if finished "$out"; then
    note "done: $(grep -a '^done (' "$out/run.log" | tail -1)"
    rm -f "$out/run.log"
    exit 0
  fi
  note "pid $pid stopped before finishing, last output: $(stage "$out")"
  pid=""
  [ "$restarts" -lt "$MAX_RESTARTS" ] && note "restarting ($((restarts + 1))/$MAX_RESTARTS)"
done
note "giving up after $MAX_RESTARTS restarts"
exit 1
