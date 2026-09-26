#!/bin/bash
# After PID $1 exits: gpt-oss-120b diagnostic run, then OLMoE bf16 re-trace.
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_PROGRESS_BARS=1
while kill -0 "$1" 2>/dev/null; do sleep 20; done
python3 -m mosl.collect --remote --repo openai/gpt-oss-120b --corpus data/diag_gptoss.jsonl --out data/traces/_diag_gptoss120b > results/diag_gptoss120b.log 2>&1
python3 -m mosl.collect --remote --dtype bfloat16 --repo allenai/OLMoE-1B-7B-0125-Instruct --corpus data/tok_olmoe.jsonl --out data/traces/olmoe-1b-7b-bf16 > results/collect_olmoe_bf16.log 2>&1
