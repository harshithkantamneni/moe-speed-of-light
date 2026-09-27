#!/bin/bash
# Pre-registered robustness run (H15): the models' own text. For each model, sample 193-token continuations of
# the 12 test prompts (top-k 40, top-p 0.95, temperature 0.8, seed 42 + sequence, end-of-generation banned), then
# teacher-force llama.cpp static layers and the mailbox cache on that text at the 12.5 % and 25 % budgets.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/ec/setup.sh"
R() { timeout 40m "$@" 2>> $OUT/stderr_runs.txt; }
MB="mailbox=1:tdec=1:helpers=28"
run_model() {  # file corpus L E kappa seqs n_gen(ncmoe used to generate)
  local f=$1 corp=$J/$2 L=$3 E=$4 K=$5 SEQ=$6 NG=$7 tag=${1%.gguf}
  R $EC_BIN -m $M/$f --corpus $corp --seqs $SEQ --ncmoe $NG -t 30 --n-prefill 128 --n-decode 192 --gen-out $OUT/${tag}_gen.jsonl > /dev/null
  echo "$tag gen rc=$? lines=$(wc -l < $OUT/${tag}_gen.jsonl)"
  local GS=$(python3 -c "print(','.join(str(i) for i in range($(wc -l < $OUT/${tag}_gen.jsonl))))")
  local cfg=""
  for q in 1 2; do
    cfg="$cfg;slots=$(( E*q/8 )):policy=dfa:kappa=$K:$MB:stats=$OUT/${tag}_gen_mb_C$(( E*q/8 )).json"
  done
  R $EC_BIN -m $M/$f --corpus $OUT/${tag}_gen.jsonl --seqs $GS --host-experts --ec "${cfg#;}" -t 30 --n-prefill 128 --n-decode 192 > $OUT/${tag}_gen_ec.jsonl
  echo "$tag gen ec rc=$?"
  for q in 1 2; do
    local n=$(( L - L*q/8 ))
    R $EC_BIN -m $M/$f --corpus $OUT/${tag}_gen.jsonl --seqs $GS --ncmoe $n -t 30 --n-prefill 128 --n-decode 192 > $OUT/${tag}_gen_static_n$n.jsonl
    echo "$tag gen static n=$n rc=$?"
  done
}
run_model gpt-oss-20b-MXFP4.gguf tok_gpt-oss-20b.jsonl 24 32 1 7,10,12,15,16,18,19,20,25,26,27,31 0
run_model Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf tok_qwen3_30b.jsonl 48 128 2 7,10,12,15,16,18,19,20,25,26,27,32 0
run_model Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf tok_qwen3_30b.jsonl 48 128 2 7,10,12,15,16,18,19,20,25,26,27,32 36
run_model gpt-oss-120b-MXFP4.gguf tok_gpt-oss-120b.jsonl 36 128 1 7,10,12,15,16,18,19,20,25,26,27,31 27
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
