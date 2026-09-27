#!/bin/bash
# Baseline check (deviation from the phase-3 protocol, reported as such): llama.cpp warns that CPU tensor
# overrides with a memory-mapped model are slow and advises --load-mode none, which reads the CPU experts into
# its own buffers (and may repack them for faster CPU kernels). Job 020 ran llama.cpp with the default mmap.
# Here: llama.cpp static layers with --load-mode none at every budget, same test sequences and steps as 020.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/ec/setup.sh"
R() { timeout 40m "$@" 2>> $OUT/stderr_runs.txt; }
run() {  # file corpus L "budgets" seqs
  local f=$1 corp=$J/$2 L=$3 FR="$4" SEQ=$5 tag=${1%.gguf}
  for q in $FR; do
    local n=$(( L - L*q/8 ))
    R $EC_BIN -m $M/$f --corpus $corp --seqs $SEQ --ncmoe $n --no-mmap -t 30 --n-prefill 128 --n-decode 192 --dump $OUT/${tag}_nommap_n$n.bin > $OUT/${tag}_nommap_n$n.jsonl
    echo "$tag no-mmap n=$n rc=$?"
    grep -iE "repack|model buffer size" $OUT/stderr_runs.txt | tail -4
  done
}
run gpt-oss-20b-MXFP4.gguf tok_gpt-oss-20b.jsonl 24 "1 2 4" 7,10,12,15,16,18,19,20,25,26,27,31
run Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf tok_qwen3_30b.jsonl 48 "1 2 4" 7,10,12,15,16,18,19,20,25,26,27,32
run Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf tok_qwen3_30b.jsonl 48 "1 2 4" 7,10,12,15,16,18,19,20,25,26,27,32
run gpt-oss-120b-MXFP4.gguf tok_gpt-oss-120b.jsonl 36 "1 2" 7,10,12,15,16,18,19,20,25,26,27,31
python3 - "$OUT" <<'PY'
import json, sys, glob, os
out = sys.argv[1]
for p in sorted(glob.glob(f"{out}/*.jsonl")):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    if rows:
        tps = 1000 * sum(r["n_decode"] for r in rows) / sum(r["decode_ms"] for r in rows)
        print(f"{tps:7.1f} tok/s n_seq {len(rows)} {os.path.basename(p)}")
PY
