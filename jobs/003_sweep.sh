#!/bin/bash
# Static-offload decode sweep (pre-registered: prereg/PROTOCOL.md on main).
# llama-bench, clocks locked, one randomized order over all configurations,
# 5 repetitions each, three configurations repeated at the end for drift.
set -x
exec 2>&1
B=$WORK/llama.cpp/build/bin/llama-bench
M=$WORK/models
declare -A F=([gpt-oss-20b-mxfp4]=gpt-oss-20b-MXFP4.gguf [gpt-oss-120b-mxfp4]=gpt-oss-120b-MXFP4.gguf
              [qwen3-30b-a3b-q4_k_m]=Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf [qwen3-30b-a3b-q8_0]=Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf)

# re-assert the clock lock and log clocks/throttle reasons for the whole job
MAXSM=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
MAXMEM=$(nvidia-smi --query-gpu=clocks.max.mem --format=csv,noheader,nounits | head -1)
sudo nvidia-smi -pm 1; sudo nvidia-smi -lgc $MAXSM,$MAXSM; sudo nvidia-smi -lmc $MAXMEM,$MAXMEM
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active,memory.used \
  --format=csv -l 5 > "$OUT/gpu_monitor.csv" &
MON=$!

# page-cache warm: read every model once so load time does not depend on order
for f in "${F[@]}"; do cat "$M/$f" > /dev/null; done
free -g

# thread count, chosen once by the pre-registered rule: best of {8, nproc/2, nproc}
NP=$(nproc)
$B -m $M/${F[gpt-oss-20b-mxfp4]} -ngl 99 -ncmoe 24 -fa 1 -p 0 -n 64 -r 3 -t 8,$((NP/2)),$NP -o jsonl > "$OUT/threads.jsonl"
T=$(python3 -c "
import json; rs=[json.loads(l) for l in open('$OUT/threads.jsonl') if l.strip()]
print(max(rs, key=lambda r: r['avg_ts'])['n_threads'])")
echo "threads=$T" | tee "$OUT/threads.txt"

# configurations (must match SWEEP in scripts/preregister.py)
{
  for n in 0 3 6 9 12 15 18 21 24 cpu; do echo "gpt-oss-20b-mxfp4 $n"; done
  for n in 0 6 12 18 24 30 36 42 48 cpu; do echo "qwen3-30b-a3b-q4_k_m $n"; done
  for n in 18 24 30 36 42 48; do echo "qwen3-30b-a3b-q8_0 $n"; done
  for n in 25 28 31 34 36; do echo "gpt-oss-120b-mxfp4 $n"; done
} > configs.txt
SEED=20260926
shuf --random-source=<(yes $SEED) configs.txt > "$OUT/order.txt"
{ cat "$OUT/order.txt"; echo "gpt-oss-20b-mxfp4 0"; echo "qwen3-30b-a3b-q4_k_m 24"; echo "gpt-oss-120b-mxfp4 36"; } > run.txt

i=0
while read -r model n; do
  i=$((i+1))
  if [ "$n" = cpu ]; then args="-ngl 0"; else args="-ngl 99 -ncmoe $n"; fi
  echo "[$i] $model n_cpu_moe=$n $(date -u +%T)"
  $B -m $M/${F[$model]} $args -fa 1 -p 0 -n 128 -r 5 -t $T -o jsonl > run.jsonl 2> run.err
  rc=$?
  python3 - "$model" "$n" "$i" "$rc" >> "$OUT/sweep.jsonl" <<'PY'
import json, sys
model, n, i, rc = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rows = [json.loads(l) for l in open("run.jsonl") if l.strip()]
if not rows:
    print(json.dumps(dict(model=model, n_cpu_moe=n, seq=i, rc=rc, error=open("run.err").read()[-2000:])))
for r in rows:
    r.update(model=model, n_cpu_moe=n, seq=i, rc=rc)
    print(json.dumps(r))
PY
  tail -3 run.err
done < run.txt

kill $MON
nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv
