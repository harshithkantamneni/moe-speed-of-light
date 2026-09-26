#!/bin/bash
# Phase 2 evaluation (pre-registered: prereg/a10_ec/ on main). Per model and VRAM budget, teacher-forced
# decode of the 16 test sequences (prefill <=128 untimed, 192 decode steps timed) with:
#   llama.cpp static layer offload (--ncmoe n, experts of the first n layers on the CPU),
#   expert cache with the same number of GPU-resident experts (slots = E*(L-n)/L per layer):
#   static hot set (profiled on 8 other sequences), LRU, DFA, DFA without CPU/GPU overlap.
set -x
exec 2>&1
J=$(cd "$(dirname "$0")" && pwd)/ec
B=$WORK/llama.cpp-ec/build/bin/llama-ec-bench
M=$WORK/models
T=30
PROF=0/1/2/3/5/6/9/17
COMMON="-t $T --n-prefill 128 --n-decode 192"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active,memory.used \
  --format=csv -l 5 > "$OUT/gpu_monitor.csv" &
MON=$!

eval_model() {  # file corpus L E test_seqs "ncmoe list" allgpu(0/1)
  local f=$1 corp=$J/$2 L=$3 E=$4 TEST=$5 NS="$6" ALLGPU=$7 tag=${1%.gguf}
  local cfg="slots=1:policy=static:stats=$OUT/${tag}_profile.json:seqs=$PROF"
  for n in $NS; do
    local C=$(( (E*(L-n) + L/2) / L ))
    cfg="$cfg;slots=$C:policy=static:init=$OUT/${tag}_profile.json:stats=$OUT/${tag}_C${C}_static.json"
    cfg="$cfg;slots=$C:policy=lru:stats=$OUT/${tag}_C${C}_lru.json"
    cfg="$cfg;slots=$C:policy=dfa:stats=$OUT/${tag}_C${C}_dfa.json"
    cfg="$cfg;slots=$C:policy=dfa:overlap=0:stats=$OUT/${tag}_C${C}_dfa_serial.json"
  done
  echo "[$(date -u +%T)] $tag expert cache"
  $B -m $M/$f --corpus $corp --seqs $TEST --host-experts --ec "$cfg" $COMMON > $OUT/${tag}_ec.jsonl 2>> $OUT/stderr_runs.txt
  for n in $NS $L; do
    echo "[$(date -u +%T)] $tag static n_cpu_moe=$n"
    $B -m $M/$f --corpus $corp --seqs $TEST --ncmoe $n $COMMON > $OUT/${tag}_static_n$n.jsonl 2>> $OUT/stderr_runs.txt
  done
  if [ "$ALLGPU" = 1 ]; then
    $B -m $M/$f --corpus $corp --seqs $TEST --ncmoe 0 $COMMON > $OUT/${tag}_static_n0.jsonl 2>> $OUT/stderr_runs.txt
  fi
  python3 - "$OUT" "$tag" <<'PY'
import json, sys, glob, statistics as st
out, tag = sys.argv[1], sys.argv[2]
def tps(path, label=None):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(r.get("config", ""), []).append(r)
    return {k: sum(x["n_decode"] for x in v) / (sum(x["decode_ms"] for x in v) / 1000) for k, v in by.items()}
for p in sorted(glob.glob(f"{out}/{tag}_*.jsonl")):
    for k, v in tps(p).items():
        print(f"{p.split('/')[-1]:50s} {k[:60]:60s} {v:8.2f} tok/s")
PY
}

eval_model gpt-oss-20b-MXFP4.gguf                  tok_gpt-oss-20b.jsonl  24 32  4,7,8,10,11,12,14,15,16,18,19,20,25,26,27,31 "21 18 12" 1
eval_model Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf tok_qwen3_30b.jsonl    48 128 4,7,8,10,11,12,14,15,16,18,19,20,25,26,27,32 "42 36 24" 1
eval_model Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf   tok_qwen3_30b.jsonl    48 128 4,7,8,10,11,12,14,15,16,18,19,20,25,26,27,32 "42 36 24" 0
eval_model gpt-oss-120b-MXFP4.gguf                 tok_gpt-oss-120b.jsonl 36 128 4,7,8,10,11,12,14,15,16,18,19,20,25,26,27,31 "32 27" 0
kill $MON
