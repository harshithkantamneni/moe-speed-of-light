#!/bin/bash
# Runs on a rented Vast.ai machine. It holds no credential: the public `gpu` branch was cloned to /w by the launch
# command, the job runs like under gpu/runner.sh (OUT=results dir, cwd=$WORK), and the results directory is printed
# to the container log as base64 lines between markers. The controller (main: gpu/vast.py) reads them back through
# Vast's log API and destroys the machine. Usage: vast_boot.sh <job name without .sh>
JOB="$1"; W=/w; OUT="$W/results/$JOB"; export WORK=/work OUT JOB
mkdir -p "$OUT" "$WORK"
echo "@@START $JOB $(date -u +%FT%TZ) $(nproc) cpus"
( while true; do echo "@@HB $(date -u +%FT%TZ) $(tail -n 1 "$OUT/stdout.log" 2>/dev/null | cut -c1-160)"; sleep 60; done ) &
HBP=$!
start=$(date +%s)
( cd "$WORK" && timeout "${JOB_TIMEOUT:-6h}" bash "$W/jobs/$JOB.sh" > "$OUT/stdout.log" 2> "$OUT/stderr.log" )
rc=$?
echo "rc=$rc seconds=$(( $(date +%s) - start ))" > "$OUT/DONE"
kill $HBP 2>/dev/null
# never ship big files: anything over 8 MB is replaced by a listing line
find "$OUT" -type f -size +8M -printf '%s %p\n' > "$OUT/omitted_large_files.txt"
find "$OUT" -type f -size +8M -delete
tar -C "$W/results" -czf /tmp/r.tgz "$JOB"
sha=$(sha256sum /tmp/r.tgz | cut -d' ' -f1); n=$(stat -c %s /tmp/r.tgz)
base64 -w 0 /tmp/r.tgz | fold -w 1000 > /tmp/r.b64; echo >> /tmp/r.b64   # fold leaves the last line unterminated
lines=$(wc -l < /tmp/r.b64)
emit() {
  echo "@@RESULT_BEGIN $JOB rc=$rc sha256=$sha bytes=$n lines=$lines"
  i=0; while IFS= read -r l; do i=$((i+1)); echo "@@R $i $l"; done < /tmp/r.b64
  echo "@@RESULT_END $JOB"
}
emit
while true; do sleep 1200; emit; done   # re-emit so a late log fetch still finds a complete copy near the tail
