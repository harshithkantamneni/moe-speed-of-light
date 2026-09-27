#!/bin/bash
# Phase 4, 4.5 (prereg/PROTOCOL.md on main), arm D: the phase-3 test re-run with correct windows. The whole prompt
# is prefilled (--n-prefill 640 >= every test prompt), then the first <= 192 tokens of the dataset response are
# teacher-forced as decode steps. Same 12 test sequences, systems, budgets and settings as jobs 020a/020b.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/ec/setup.sh"
. "$(cd "$(dirname "$0")" && pwd)/ec/windows.sh"
run_model gpt-oss-20b-MXFP4.gguf $J/tok_gpt-oss-20b.jsonl 24 32 1 "1 2 4" 7,10,12,15,16,18,19,20,25,26,27,31 1 D
run_model Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf $J/tok_qwen3_30b.jsonl 48 128 2 "1 2 4" 7,10,12,15,16,18,19,20,25,26,27,32 1 D
run_model Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf $J/tok_qwen3_30b.jsonl 48 128 2 "1 2 4" 7,10,12,15,16,18,19,20,25,26,27,32 0 D
run_model gpt-oss-120b-MXFP4.gguf $J/tok_gpt-oss-120b.jsonl 36 128 1 "1 2" 7,10,12,15,16,18,19,20,25,26,27,31 0 D
summary
