#!/bin/bash
# Phase 4.5 on the same A10 instance as jobs 038/039a, part 3: arm S for gpt-oss-120b (job 035b's responses). The models' own sampled responses (vLLM,
# generation-config sampling: jobs 031, 033 (Qwen3 from its FP8 checkpoint), 035b) to the 12 test prompts,
# whole prompt prefilled, first <= 192 response tokens decoded; systems and budgets as job 038's arm D.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/ec/setup.sh"
. "$(cd "$(dirname "$0")" && pwd)/ec/windows.sh"
. "$(cd "$(dirname "$0")" && pwd)/ec/s_corpora.sh"
RR=$(dirname "$OUT"); REPO=$(dirname "$RR")
for i in $(seq 1 100); do [ -f "$RR/035b_gen_120b_s@40gb/DONE" ] && break; ( cd $REPO && timeout 120 git pull -q --rebase ) || true; sleep 30; done
cat "$RR/035b_gen_120b_s@40gb/DONE" || { echo "job 035b results not available"; exit 1; }
SG=7,10,12,15,16,18,19,20,25,26,27,31; SQ=7,10,12,15,16,18,19,20,25,26,27,32; A=0,1,2,3,4,5,6,7,8,9,10,11
s_corpus "$RR/035b_gen_120b_s@40gb" gpt-oss-120b_S gpt-oss-120b $SG $OUT/S_gpt-oss-120b.jsonl
wc -l $OUT/S_*.jsonl
run_model gpt-oss-120b-MXFP4.gguf $OUT/S_gpt-oss-120b.jsonl 36 128 1 "1 2" $A 0 S
summary
