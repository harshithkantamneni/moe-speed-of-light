#!/bin/bash
# GPU job runner: polls the `gpu` branch for jobs/NNN_name.sh, runs each once (in order),
# and pushes jobs' outputs to results/NNN_name/. A heartbeat with the tail of the current
# job's log is pushed every few minutes. Jobs share a persistent scratch dir ($WORK).
set -u
TOKEN="$(cat ${TOKEN_FILE:-/root/.runner_token} 2>/dev/null)"
URL="${RUNNER_URL:-https://x-access-token:${TOKEN}@github.com/harshithkantamneni/moe-speed-of-light.git}"
BASE=${RUNNER_BASE:-/opt/runner}; REPO=$BASE/repo; export WORK=$BASE/work
mkdir -p "$BASE" "$WORK"
until git clone -q --branch gpu --single-branch "$URL" "$REPO"; do sleep 20; done
cd "$REPO"
git config user.name "Harshith Kantamneni"
git config user.email "harshithkantamneni@users.noreply.github.com"

push() {  # commit everything under results/ and push, retrying on races
  git add -A results >/dev/null 2>&1
  git diff --cached --quiet && return 0
  git commit -q -m "$1" || return 0
  for i in 1 2 3 4 5; do
    git pull -q --rebase && git push -q && return 0
    sleep 5
  done
}

last_hb=0; current=""
heartbeat() {
  local now; now=$(date +%s)
  (( now - last_hb < 180 )) && return
  last_hb=$now
  mkdir -p results
  {
    echo "time: $(date -u +%FT%TZ)"; echo "uptime: $(uptime)"
    echo "current_job: ${current:-idle}"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw --format=csv,noheader 2>/dev/null
    df -h / | tail -1
    [ -n "$current" ] && { echo "--- tail of $current stdout ---"; tail -n 40 "results/$current/stdout.log" 2>/dev/null; }
  } > results/heartbeat.txt
  push "heartbeat" 
}

while true; do
  git pull -q --rebase || true
  for job in $(ls jobs/*.sh 2>/dev/null | sort); do
    name=$(basename "$job" .sh); out="$REPO/results/$name"
    [ -f "$out/DONE" ] && continue
    mkdir -p "$out"; current=$name; last_hb=0; heartbeat
    start=$(date +%s)
    ( cd "$WORK" && OUT="$out" timeout "${JOB_TIMEOUT:-4h}" bash "$REPO/$job" > "$out/stdout.log" 2> "$out/stderr.log" ) &
    pid=$!
    while kill -0 $pid 2>/dev/null; do heartbeat; sleep 20; done
    wait $pid; rc=$?
    echo "rc=$rc seconds=$(( $(date +%s) - start ))" > "$out/DONE"
    current=""
    push "result: $name (rc=$rc)"
  done
  heartbeat
  sleep 30
done
