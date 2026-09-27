#!/bin/bash
# Phase 4 (prereg/PROTOCOL.md on main). Qwen3-30B-A3B's bf16 checkpoint (61 GB) does not fit the 40 GB A100 that Lambda has available: G and S are generated, and D scored, with the vendor's FP8 checkpoint of the same model (deviation from 4.1, reported). Routing traces (job 035) use the bf16 weights.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/prov/setup.sh"
. "$(cd "$(dirname "$0")" && pwd)/prov/gen.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
( dl qwen3-30b-a3b-fp8 ${HUB[qwen3-30b-a3b-fp8]}; dl qwen3-30b-a3b ${HUB[qwen3-30b-a3b]} ) &
run_gen qwen3-30b-a3b qwen3-30b-a3b-fp8 "--tag _fp8" 60m _fp8
wait
ls -la $OUT
