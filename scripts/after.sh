#!/bin/bash
# Wait for a PID to exit, then collect gpt-oss-20b.
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_PROGRESS_BARS=1
while kill -0 "$1" 2>/dev/null; do sleep 30; done
python3 -m mosl.collect --repo openai/gpt-oss-20b --corpus data/tok_gpt-oss-20b.jsonl --out data/traces/gpt-oss-20b > results/collect_gpt-oss-20b.log 2>&1
