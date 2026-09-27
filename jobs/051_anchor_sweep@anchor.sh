#!/bin/bash
# Audit anchors, part 2: llama-bench (stock llama.cpp 2145525a) decode sweep over --n-cpu-moe for the configurations
# predicted by job 050 (scripts/anchor_predict.py SWEEP), clocks locked, -fa 1 -p 0 -n 128 -r 5, one randomized order;
# thread count by the phase-1 rule (best of {8, nproc/2, nproc} on one CPU-heavy configuration).
set -x
exec 2>&1
B=$WORK/llama.cpp/build/bin/llama-bench
M=$WORK/models
declare -A F=([mixtral-8x7b-q4_k_m]=Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf [mixtral-8x7b-q8_0]=Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf
              [phi3.5-moe-q4_k_m]=Phi-3.5-MoE-instruct-Q4_K_M.gguf [qwen2-57b-q4_k_m]=qwen2-57b-a14b-instruct-q4_k_m.gguf
              [dsv2lite-q8_0]=DeepSeek-V2-Lite-Chat.Q8_0.gguf [qwen3-30b-a3b-q4_k_m]=Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf)
MAXSM=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
MAXMEM=$(nvidia-smi --query-gpu=clocks.max.mem --format=csv,noheader,nounits | head -1)
sudo nvidia-smi -pm 1; sudo nvidia-smi -lgc $MAXSM,$MAXSM; sudo nvidia-smi -lmc $MAXMEM,$MAXMEM
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active,memory.used --format=csv -l 5 > "$OUT/gpu_monitor.csv" &
MON=$!
for f in "${F[@]}"; do cat "$M/$f" > /dev/null; done; free -g
NP=$(nproc)
$B -m $M/${F[mixtral-8x7b-q4_k_m]} -ngl 99 -ncmoe 32 -fa 1 -p 0 -n 64 -r 3 -t 8,$((NP/2)),$NP -o jsonl > "$OUT/threads.jsonl"
T=$(python3 -c "
import json; rs=[json.loads(l) for l in open('$OUT/threads.jsonl') if l.strip()]
print(max(rs, key=lambda r: r['avg_ts'])['n_threads'])")
echo "threads=$T" | tee "$OUT/threads.txt"
python3 -c "
import sys; sys.path.insert(0, '$(cd "$(dirname "$0")" && pwd)/anchor')
from scripts.anchor_predict import SWEEP
for m, ns in SWEEP.items():
    for n in ns: print(m, n)" > configs.txt
shuf --random-source=<(yes 20260927) configs.txt > "$OUT/order.txt"
{ cat "$OUT/order.txt"; echo "qwen3-30b-a3b-q4_k_m 24"; echo "mixtral-8x7b-q4_k_m 16"; } > run.txt
i=0
while read -r model n; do
  i=$((i+1))
  echo "[$i] $model n_cpu_moe=$n $(date -u +%T)"
  timeout 15m $B -m $M/${F[$model]} -ngl 99 -ncmoe $n -fa 1 -p 0 -n 128 -r 5 -t $T -o jsonl > run.jsonl 2> run.err
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
  tail -2 run.err
done < run.txt
kill $MON
python3 - "$OUT" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1] + "/sweep.jsonl")]
for r in rows:
    print(r["model"], r["n_cpu_moe"], round(r.get("avg_ts", 0), 2), r.get("rc"), (r.get("error") or "")[-120:].replace("\n", " "))
PY
