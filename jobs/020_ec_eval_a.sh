#!/bin/bash
# Pre-registered test run, part 1 (prereg/PROTOCOL.md, phase 3): gpt-oss-20b and Qwen3-30B-A3B Q4_K_M on the 12
# fresh test sequences x 192 decode steps. Mailbox cache, mailbox static layers, split-graph cache, llama.cpp
# static layers at every budget; all experts on the GPU as the reference. Top-5 logits dumped for every run.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/ec/setup.sh"
R() { timeout 40m "$@" 2>> $OUT/stderr_runs.txt; }
MB="mailbox=1:tdec=1:helpers=28"
alloc() { python3 -c "
L,E,n=$2,$3,$4
print('\n'.join(f'{il} {0 if il < n else E}' for il in range(L)))" > $1; }
run_model() {  # file corpus L E kappa "budgets (1/8)" seqs allgpu(0/1)
  local f=$1 corp=$J/$2 L=$3 E=$4 K=$5 FR="$6" SEQ=$7 AG=$8 tag=${1%.gguf}
  local cfg=""
  for q in $FR; do
    local C=$(( E*q/8 )) n=$(( L - L*q/8 ))
    alloc $OUT/${tag}_n$n.txt $L $E $n
    cfg="$cfg;slots=$C:policy=dfa:kappa=$K:$MB:stats=$OUT/${tag}_mb_C$C.json"
    cfg="$cfg;slots=1:alloc=$OUT/${tag}_n$n.txt:$MB:stats=$OUT/${tag}_mb_n$n.json"
    cfg="$cfg;slots=$C:policy=dfa:kappa=$K:stats=$OUT/${tag}_split_C$C.json"
  done
  R $EC_BIN -m $M/$f --corpus $corp --seqs $SEQ --host-experts --ec "${cfg#;}" -t 30 --n-prefill 128 --n-decode 192 --dump $OUT/${tag}_ec.bin > $OUT/${tag}_ec.jsonl
  echo "$tag ec rc=$?"
  for q in $FR; do
    local n=$(( L - L*q/8 ))
    R $EC_BIN -m $M/$f --corpus $corp --seqs $SEQ --ncmoe $n -t 30 --n-prefill 128 --n-decode 192 --dump $OUT/${tag}_static_n$n.bin > $OUT/${tag}_static_n$n.jsonl
    echo "$tag static n=$n rc=$?"
  done
  if [ "$AG" = 1 ]; then
    R $EC_BIN -m $M/$f --corpus $corp --seqs $SEQ --ncmoe 0 -t 30 --n-prefill 128 --n-decode 192 --dump $OUT/${tag}_static_n0.bin > $OUT/${tag}_static_n0.jsonl
    echo "$tag all-GPU rc=$?"
    R $EC_BIN -m $M/$f --corpus $corp --seqs $SEQ --ncmoe 0 -t 1 --n-prefill 128 --n-decode 192 > $OUT/${tag}_allgpu_t1.jsonl
    echo "$tag all-GPU 1 thread rc=$?"
  fi
}
run_model gpt-oss-20b-MXFP4.gguf tok_gpt-oss-20b.jsonl 24 32 1 "1 2 4" 7,10,12,15,16,18,19,20,25,26,27,31 1
run_model Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf tok_qwen3_30b.jsonl 48 128 2 "1 2 4" 7,10,12,15,16,18,19,20,25,26,27,32 1
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
        print(f"{rate(v):7.1f} tok/s nll {nll:.4f} hit {s.get('hit_rate', 0):.3f} n_seq {len(v)} {os.path.basename(p)} {k.split(':stats')[0][:50]} {os.path.basename(st)}")
PY
