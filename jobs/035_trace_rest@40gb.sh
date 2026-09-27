#!/bin/bash
# Phase 4 (prereg/PROTOCOL.md on main). GPU traces (bf16 / MXFP4 checkpoints, fp32 math) of job 033's Qwen3 corpora and job 034's gpt-oss-120b corpora, plus the 120b and Qwen3 dataset token files and job 020c samples (4.1).
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/prov/setup.sh"
. "$(cd "$(dirname "$0")" && pwd)/prov/gen.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
trace_arms qwen3-30b-a3b $(dirname "$OUT")/033_gen_qwen_fp8@40gb qwen3-30b-a3b_fp8 D G S
trc qwen3-30b-a3b $P/data/tok_qwen3_30b.jsonl qwen3-30b-a3b_tok
for f in ${GEN020C[qwen3-30b-a3b]}; do trc qwen3-30b-a3b $P/data/gen020c/$f qwen3-30b-a3b_gen020c_${f%_gen.jsonl}; done
trace_arms gpt-oss-120b $(dirname "$OUT")/034_score_120b@40gb gpt-oss-120b D G
trc gpt-oss-120b $P/data/tok_gpt-oss-120b.jsonl gpt-oss-120b_tok
trc gpt-oss-120b $P/data/gen020c/gpt-oss-120b-MXFP4_gen.jsonl gpt-oss-120b_gen020c_gpt-oss-120b-MXFP4
ls -la $OUT
