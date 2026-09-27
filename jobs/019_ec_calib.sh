#!/bin/bash
# Calibration run on the PROFILE sequences only (0,1,2,3,5,6,9,17; the test sequences are not touched), final
# system configuration: mailbox helpers (28, chunked, AVX-512 MXFP4), DFA with kappa. For every model and budget:
# llama.cpp static layers, the static-layer layout on the helpers, the split-graph cache and the mailbox cache;
# anchors: all experts on the GPU where they fit, all on the CPU (stock and helpers). The step-time model is
# fitted on these runs and its predictions for the test sequences are committed before job 020 runs.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/ec/setup.sh"
R() { timeout 30m "$@" 2>> $OUT/stderr_runs.txt; }
SEQ=0,1,2,3,5,6,9,17
MB="mailbox=1:tdec=1:helpers=28"
alloc() { python3 -c "
L,E,n=$2,$3,$4
print('\n'.join(f'{il} {0 if il < n else E}' for il in range(L)))" > $1; }
run_model() {  # file corpus L E kappa "budgets (1/8)" "extra stock n values"
  local f=$1 corp=$J/$2 L=$3 E=$4 K=$5 FR="$6" XN="$7" tag=${1%.gguf}
  alloc $OUT/${tag}_n$L.txt $L $E $L
  local cfg="slots=1:alloc=$OUT/${tag}_n$L.txt:$MB:stats=$OUT/${tag}_mb_n$L.json"
  for q in $FR; do
    local C=$(( E*q/8 )) n=$(( L - L*q/8 ))
    alloc $OUT/${tag}_n$n.txt $L $E $n
    cfg="$cfg;slots=$C:policy=dfa:kappa=$K:$MB:stats=$OUT/${tag}_mb_C$C.json"
    cfg="$cfg;slots=1:alloc=$OUT/${tag}_n$n.txt:$MB:stats=$OUT/${tag}_mb_n$n.json"
    cfg="$cfg;slots=$C:policy=dfa:kappa=$K:stats=$OUT/${tag}_split_C$C.json"
  done
  R $EC_BIN -m $M/$f --corpus $corp --seqs $SEQ --host-experts --ec "$cfg" -t 30 --n-prefill 128 --n-decode 128 > $OUT/${tag}_ec.jsonl
  echo "$tag rc=$?"
  for n in $XN $L $(for q in $FR; do echo $(( L - L*q/8 )); done); do
    R $EC_BIN -m $M/$f --corpus $corp --seqs $SEQ --ncmoe $n -t 30 --n-prefill 128 --n-decode 128 > $OUT/${tag}_static_n$n.jsonl
  done
}
run_model gpt-oss-20b-MXFP4.gguf tok_gpt-oss-20b.jsonl 24 32 1 "1 2 4" "0"
run_model Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf tok_qwen3_30b.jsonl 48 128 2 "1 2 4" "0"
run_model Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf tok_qwen3_30b.jsonl 48 128 2 "1 2 4" ""
run_model gpt-oss-120b-MXFP4.gguf tok_gpt-oss-120b.jsonl 36 128 1 "1 2" ""
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
        nll = sum(r["nll_sum"] for r in v) / max(1, sum(r["nll_n"] for r in v))
        print(f"{rate(v):7.1f} tok/s nll {nll:.4f} hit {s.get('hit_rate', 0):.3f} {os.path.basename(p)} {k.split(':stats')[0][:50]} {os.path.basename(st)}")
PY
