#!/bin/bash
# After the gpt-oss queue (PID $1) finishes: re-trace OLMoE with bf16 compute to measure routing sensitivity to numerics.
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_PROGRESS_BARS=1
while kill -0 "$1" 2>/dev/null; do sleep 30; done
python3 -m mosl.collect --remote --dtype bfloat16 --repo allenai/OLMoE-1B-7B-0125-Instruct --corpus data/tok_olmoe.jsonl --out data/traces/olmoe-1b-7b-bf16 > results/collect_olmoe_bf16.log 2>&1
