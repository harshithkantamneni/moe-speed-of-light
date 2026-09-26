#!/bin/bash
# Collect gpt-oss traces one after another, streaming tensors over HTTP (no disk).
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_PROGRESS_BARS=1
for r in openai/gpt-oss-120b openai/gpt-oss-20b; do
  n=$(basename $r)
  python3 -m mosl.collect --remote --repo $r --corpus data/tok_$n.jsonl --out data/traces/$n > results/collect_$n.log 2>&1
done
