#!/bin/bash
# Phase 4, 4.5, arm S: as job 025 on the models' own sampled responses (job 022, vLLM, generation-config sampling)
# to the same 12 test prompts. Waits for job 022 (another instance) to push its results.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/ec/setup.sh"
. "$(cd "$(dirname "$0")" && pwd)/ec/windows.sh"
REPO=$(dirname "$(dirname "$OUT")"); G=$REPO/results/022_prov_gen@a100
for i in $(seq 1 120); do [ -f "$G/DONE" ] && break; ( cd $REPO && timeout 120 git pull -q --rebase ) || true; sleep 30; done
cat "$G/DONE" || { echo "job 022 results not available"; exit 1; }
mk() {  # prompts-key unused seqs out: the S responses of the test rows, in --seqs order
  python3 - "$G/corp_$1_S.jsonl" "$J/../prov/data/prompts/$1.jsonl" "$3" > "$4" <<'PY'
import json, sys
corp = {r["corpus_idx"]: r for r in map(json.loads, open(sys.argv[1]))}
rows = {r["tok_row"]: r["corpus_idx"] for r in map(json.loads, open(sys.argv[2])) if r["tok_row"] >= 0}
for s in map(int, sys.argv[3].split(",")):
    c = corp[rows[s]]
    print(json.dumps({"domain": c["domain"], "ids": c["ids"], "prompt_len": c["prompt_len"], "src_row": s}))
PY
}
SG=7,10,12,15,16,18,19,20,25,26,27,31; SQ=7,10,12,15,16,18,19,20,25,26,27,32
mk gpt-oss-20b x $SG $OUT/S_gpt-oss-20b.jsonl; mk gpt-oss-120b x $SG $OUT/S_gpt-oss-120b.jsonl; mk qwen3-30b-a3b x $SQ $OUT/S_qwen3.jsonl
wc -l $OUT/S_*.jsonl
A=0,1,2,3,4,5,6,7,8,9,10,11
run_model gpt-oss-20b-MXFP4.gguf $OUT/S_gpt-oss-20b.jsonl 24 32 1 "1 2 4" $A 1 S
run_model Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf $OUT/S_qwen3.jsonl 48 128 2 "1 2 4" $A 1 S
run_model Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf $OUT/S_qwen3.jsonl 48 128 2 "1 2 4" $A 0 S
run_model gpt-oss-120b-MXFP4.gguf $OUT/S_gpt-oss-120b.jsonl 36 128 1 "1 2" $A 0 S
summary
