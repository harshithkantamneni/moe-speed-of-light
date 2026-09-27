#!/bin/bash
# Phase 4, 4.1: traces that jobs 032/035 could not produce (the MoE sub-block over ~190k tokens at once ran out of
# memory on the 40 GB A100); now chunked (--moe-chunk 16384; exact on the equivalence models). Existing packs are kept.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/prov/setup.sh"
. "$(cd "$(dirname "$0")" && pwd)/prov/gen.sh"
R=$(dirname "$OUT")
trace_arms olmoe $R/031_gen_small@40gb olmoe D G S
trace_arms gpt-oss-20b $R/031_gen_small@40gb gpt-oss-20b D G S
trace_arms qwen3-30b-a3b $R/033_gen_qwen_fp8@40gb qwen3-30b-a3b_fp8 D G S
trc qwen3-30b-a3b $P/data/tok_qwen3_30b.jsonl qwen3-30b-a3b_tok
for f in ${GEN020C[qwen3-30b-a3b]}; do trc qwen3-30b-a3b $P/data/gen020c/$f qwen3-30b-a3b_gen020c_${f%_gen.jsonl}; done
trace_arms gpt-oss-120b $R/034_score_120b@40gb gpt-oss-120b D G
trc gpt-oss-120b $P/data/tok_gpt-oss-120b.jsonl gpt-oss-120b_tok
trc gpt-oss-120b $P/data/gen020c/gpt-oss-120b-MXFP4_gen.jsonl gpt-oss-120b_gen020c_gpt-oss-120b-MXFP4
ls -la $OUT
