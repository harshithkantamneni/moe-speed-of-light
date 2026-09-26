#!/bin/bash
# Collect traces for the remaining models one after another (RAM allows one at a time).
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_PROGRESS_BARS=1
for r in openai/gpt-oss-20b openai/gpt-oss-120b; do
  n=$(basename $r)
  python3 -m mosl.collect --repo $r --corpus data/tok_$n.jsonl --out data/traces/$n > results/collect_$n.log 2>&1
done
