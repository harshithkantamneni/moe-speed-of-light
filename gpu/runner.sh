#!/bin/bash
# GPU job runner: polls the `gpu` branch for jobs/NNN_name.sh, runs each once (in order),
# and pushes jobs' outputs to results/NNN_name/ on `gpu`.
#
# Observability does not depend on the job loop: a separate heartbeat loop, with its own
# clone, force-pushes a one-commit branch `gpu-heartbeat` every 2 minutes holding the
# machine state and a snapshot of the running job's results directory. No command in it
# can block (GPU tools are detached, every git call has a timeout, stale locks are cleared),
# so partial results are visible even if a job or the main push hangs.
set -u
export GIT_TERMINAL_PROMPT=0
TOKEN="$(cat ${TOKEN_FILE:-/root/.runner_token} 2>/dev/null)"
URL="${RUNNER_URL:-https://x-access-token:${TOKEN}@github.com/harshithkantamneni/moe-speed-of-light.git}"
BASE=${RUNNER_BASE:-/opt/runner}; REPO=$BASE/repo; HB=$BASE/hb; export WORK=$BASE/work
CUR=$BASE/current_job
mkdir -p "$BASE" "$WORK"
until git clone -q --branch gpu --single-branch "$URL" "$REPO"; do sleep 20; done
until git clone -q --branch gpu --single-branch "$URL" "$HB"; do sleep 20; done
for d in "$REPO" "$HB"; do
  git -C "$d" config user.name "Harshith Kantamneni"
  git -C "$d" config user.email "harshithkantamneni@users.noreply.github.com"
  git -C "$d" config gc.auto 0
done
: > "$CUR"

push() {  # main repo only; the heartbeat never touches it
  cd "$REPO"
  rm -f .git/index.lock
  git add -A results >/dev/null 2>&1
  git diff --cached --quiet && return 0
  git commit -q -m "$1" || return 0
  for i in 1 2 3 4 5 6; do
    rm -f .git/index.lock
    timeout 180 git pull -q --rebase && timeout 180 git push -q && return 0
    git rebase --abort >/dev/null 2>&1
    sleep 10
  done
}

heartbeat_loop() {
  renice -n -5 -p $BASHPID >/dev/null 2>&1
  cd "$HB" && git checkout -q --orphan heartbeat && git rm -rq --cached . >/dev/null 2>&1
  local first=1
  while true; do
    local cur; cur=$(cat "$CUR" 2>/dev/null)
    rm -f "$HB/.git/index.lock"
    find "$HB" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} + 2>/dev/null
    ( timeout -k 2 8 nvidia-smi --query-gpu=name,utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw,clocks_throttle_reasons.active --format=csv,noheader > $BASE/nvsmi.txt 2>&1 & )
    sleep 3
    {
      echo "time: $(date -u +%FT%TZ)"; echo "uptime: $(uptime)"
      echo "current_job: ${cur:-idle}"
      cat $BASE/nvsmi.txt 2>/dev/null; echo "nvidia-smi processes alive: $(pgrep -c -x nvidia-smi)"
      free -g | head -2; df -h / | tail -1
      echo "--- top processes ---"; timeout 5 ps -eo pid,stat,wchan:24,pcpu,pmem,etime,args --sort=-pcpu | head -8 | cut -c1-220
      echo "--- git processes ---"; timeout 5 ps -eo pid,etime,args | grep -E "[g]it " | cut -c1-160
      echo "--- dmesg ---"; timeout 5 sudo dmesg 2>/dev/null | grep -iE "nvrm|xid|oom|hung|blocked" | tail -6
      if [ -n "$cur" ]; then
        echo "--- tail of $cur stdout ---"; tail -n 30 "$REPO/results/$cur/stdout.log" 2>/dev/null | cut -c1-300
        echo "--- tail of $cur stderr_runs ---"; tail -n 8 "$REPO/results/$cur/stderr_runs.txt" 2>/dev/null | cut -c1-300
      fi
    } > "$HB/heartbeat.txt" 2>&1
    if [ -n "$cur" ] && [ -d "$REPO/results/$cur" ]; then  # snapshot of the running job's (small) outputs
      mkdir -p "$HB/snap/$cur"
      find "$REPO/results/$cur" -maxdepth 1 -type f -size -20M -exec cp {} "$HB/snap/$cur/" \; 2>/dev/null
    fi
    ls -d "$REPO"/results/*/ 2>/dev/null | xargs -n1 basename > "$HB/done_jobs.txt" 2>/dev/null
    git add -A . >/dev/null 2>&1
    if [ $first = 1 ]; then git commit -q -m heartbeat && first=0; else git commit -q --amend -m heartbeat; fi
    timeout 90 git push -q -f origin HEAD:gpu-heartbeat >/dev/null 2>&1
    sleep "${HB_INTERVAL:-120}"
  done
}
heartbeat_loop &

while true; do
  cd "$REPO"; rm -f .git/index.lock
  timeout 180 git pull -q --rebase || git rebase --abort >/dev/null 2>&1
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
  push "results sync"
  sleep 30
done
