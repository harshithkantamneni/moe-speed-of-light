#!/bin/bash
# GPU job runner: polls the `gpu` branch for jobs/NNN_name.sh, runs each once (in order),
# and pushes jobs' outputs to results/NNN_name/. A separate heartbeat loop pushes
# results/heartbeat.txt every 2 minutes with the machine state and the tail of the
# current job's logs; every diagnostic it runs has a timeout, so a hung job or a hung
# nvidia-smi cannot silence it. Git access from the two loops is serialised with flock.
set -u
TOKEN="$(cat ${TOKEN_FILE:-/root/.runner_token} 2>/dev/null)"
URL="${RUNNER_URL:-https://x-access-token:${TOKEN}@github.com/harshithkantamneni/moe-speed-of-light.git}"
BASE=${RUNNER_BASE:-/opt/runner}; REPO=$BASE/repo; export WORK=$BASE/work
LOCK=$BASE/git.lock; CUR=$BASE/current_job
mkdir -p "$BASE" "$WORK"
until git clone -q --branch gpu --single-branch "$URL" "$REPO"; do sleep 20; done
cd "$REPO"
git config user.name "Harshith Kantamneni"
git config user.email "harshithkantamneni@users.noreply.github.com"
: > "$CUR"

push() {  # commit everything under results/ and push, retrying on races
  (
    flock -w 120 9 || exit 0
    git add -A results >/dev/null 2>&1
    git diff --cached --quiet && exit 0
    git commit -q -m "$1" || exit 0
    for i in 1 2 3 4 5; do
      timeout 60 git pull -q --rebase && timeout 60 git push -q && exit 0
      git rebase --abort >/dev/null 2>&1
      sleep 5
    done
  ) 9>"$LOCK"
}

heartbeat_loop() {
  while true; do
    local cur; cur=$(cat "$CUR" 2>/dev/null)
    mkdir -p results
    {
      echo "time: $(date -u +%FT%TZ)"; echo "uptime: $(uptime)"
      echo "current_job: ${cur:-idle}"
      timeout 10 nvidia-smi --query-gpu=name,utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw,clocks_throttle_reasons.active --format=csv,noheader 2>&1 || echo "nvidia-smi timed out"
      free -g | head -2; df -h / | tail -1
      echo "--- top processes ---"; timeout 5 ps -eo pid,stat,pcpu,pmem,etime,args --sort=-pcpu | head -8 | cut -c1-200
      echo "--- dmesg ---"; timeout 5 sudo dmesg 2>/dev/null | tail -5
      if [ -n "$cur" ]; then
        echo "--- tail of $cur stdout ---"; tail -n 30 "results/$cur/stdout.log" 2>/dev/null | cut -c1-300
        echo "--- tail of $cur stderr_runs ---"; tail -n 8 "results/$cur/stderr_runs.txt" 2>/dev/null | cut -c1-300
      fi
    } > results/heartbeat.txt 2>&1
    push "heartbeat"
    sleep "${HB_INTERVAL:-120}"
  done
}
heartbeat_loop &

while true; do
  ( flock -w 120 9 && timeout 60 git pull -q --rebase || git rebase --abort >/dev/null 2>&1 ) 9>"$LOCK"
  for job in $(ls jobs/*.sh 2>/dev/null | sort); do
    name=$(basename "$job" .sh); out="$REPO/results/$name"
    [ -f "$out/DONE" ] && continue
    mkdir -p "$out"; echo "$name" > "$CUR"
    start=$(date +%s)
    ( cd "$WORK" && OUT="$out" timeout "${JOB_TIMEOUT:-100m}" bash "$REPO/$job" > "$out/stdout.log" 2> "$out/stderr.log" )
    rc=$?
    echo "rc=$rc seconds=$(( $(date +%s) - start ))" > "$out/DONE"
    : > "$CUR"
    push "result: $name (rc=$rc)"
  done
  sleep 30
done
