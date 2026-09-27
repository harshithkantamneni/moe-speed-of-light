#!/bin/bash
# Phase 4.5 on the same A10 instance as job 038, part 2: arm S. The models' own sampled responses (vLLM,
# generation-config sampling: jobs 031 and 033 (Qwen3 from its FP8 checkpoint); gpt-oss-120b in job 039b) to the 12 test prompts,
# whole prompt prefilled, first <= 192 response tokens decoded; systems and budgets as job 038's arm D.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/ec/setup.sh"
. "$(cd "$(dirname "$0")" && pwd)/ec/windows.sh"
. "$(cd "$(dirname "$0")" && pwd)/ec/s_corpora.sh"
RR=$(dirname "$OUT")
SG=7,10,12,15,16,18,19,20,25,26,27,31; SQ=7,10,12,15,16,18,19,20,25,26,27,32; A=0,1,2,3,4,5,6,7,8,9,10,11
s_corpus $RR/031_gen_small@40gb gpt-oss-20b_S gpt-oss-20b $SG $OUT/S_gpt-oss-20b.jsonl
s_corpus $RR/033_gen_qwen_fp8@40gb qwen3-30b-a3b_fp8_S qwen3-30b-a3b $SQ $OUT/S_qwen3.jsonl
wc -l $OUT/S_*.jsonl
run_model gpt-oss-20b-MXFP4.gguf $OUT/S_gpt-oss-20b.jsonl 24 32 1 "1 2 4" $A 1 S
run_model Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf $OUT/S_qwen3.jsonl 48 128 2 "1 2 4" $A 1 S
run_model Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf $OUT/S_qwen3.jsonl 48 128 2 "1 2 4" $A 0 S
summary
