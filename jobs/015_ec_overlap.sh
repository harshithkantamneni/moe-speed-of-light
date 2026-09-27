#!/bin/bash
# After the overlap fix (views in the GPU split no longer mark the CPU split dependent): DFA with zero-copy
# admissions, overlap on/off, admission margin kappa, three budgets; then Q8_0 and gpt-oss-120b.
# Every configuration and its llama.cpp baseline are measured in this job, on this instance.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/ec/setup.sh"
R() { timeout 30m "$@" 2>> $OUT/stderr_runs.txt; }
PROF=0/1/2/3/5/6/9/17
SEQ=4,8,11,14
run_model() {  # file corpus L E kappa "budgets (fractions of E in 1/8 units)"
  local f=$1 corp=$J/$2 L=$3 E=$4 K=$5 FR="$6" tag=${1%.gguf}
  local cfg="slots=1:policy=static:stats=$OUT/${tag}_profile.json:seqs=$PROF"
  for q in $FR; do
    local C=$(( E*q/8 ))
    cfg="$cfg;slots=$C:policy=dfa:stats=$OUT/${tag}_C${C}_dfa.json"
    cfg="$cfg;slots=$C:policy=dfa:overlap=0:stats=$OUT/${tag}_C${C}_dfa_serial.json"
    cfg="$cfg;slots=$C:policy=dfa:kappa=$K:stats=$OUT/${tag}_C${C}_dfa_k.json"
    cfg="$cfg;slots=$C:policy=static:init=$OUT/${tag}_profile.json:stats=$OUT/${tag}_C${C}_static.json"
  done
  R $EC_BIN -m $M/$f --corpus $corp --seqs $SEQ --host-experts --ec "$cfg" -t 30 --n-prefill 128 --n-decode 128 > $OUT/${tag}_ec.jsonl
  grep "run concurrently" $OUT/stderr_runs.txt | tail -3
  for q in $FR; do
    local n=$(( L - L*q/8 ))
    R $EC_BIN -m $M/$f --corpus $corp --seqs $SEQ --ncmoe $n -t 30 --n-prefill 128 --n-decode 128 > $OUT/${tag}_static_n$n.jsonl
  done
}
run_model gpt-oss-20b-MXFP4.gguf tok_gpt-oss-20b.jsonl 24 32 1 "1 2 4"
run_model Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf tok_qwen3_30b.jsonl 48 128 2 "1 2 4"
run_model Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf tok_qwen3_30b.jsonl 48 128 2 "1 2"
run_model gpt-oss-120b-MXFP4.gguf tok_gpt-oss-120b.jsonl 36 128 1 "1 2"
python3 - "$OUT" <<'PY'
import json, sys, glob, os
out = sys.argv[1]
def rate(rows): return sum(r["n_decode"] for r in rows) / (sum(r["decode_ms"] for r in rows) / 1000)
for p in sorted(glob.glob(f"{out}/*.jsonl")):
    by = {}
    for l in open(p):
        if l.strip():
            r = json.loads(l); by.setdefault(r.get("config", ""), []).append(r)
    for k, v in by.items():
        st = k.split("stats=")[-1].split(":")[0] if "stats=" in k else ""
        s = json.load(open(st)) if st and os.path.exists(st) else {}
        print(f"{rate(v):7.1f} tok/s hit {s.get('hit_rate', 0):.3f} adm/step {s.get('admits', 0)/max(1, s.get('steps', 1)):5.1f} {os.path.basename(p)} {k.split(':stats')[0][:60]}")
PY
