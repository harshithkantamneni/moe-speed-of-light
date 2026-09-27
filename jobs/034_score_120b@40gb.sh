#!/bin/bash
# Phase 4 (prereg/PROTOCOL.md on main). gpt-oss-120b on a 40 GB A100 with vLLM weight offload (4.4): greedy responses for the 35 traced prompts only (V1 needs its own analysis), D scores for all 160 conversations, format variants V1/V2.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/prov/setup.sh"
. "$(cd "$(dirname "$0")" && pwd)/prov/gen.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
( dl gpt-oss-120b ${HUB[gpt-oss-120b]} ) &
run_gen gpt-oss-120b gpt-oss-120b "--arms G,D,H --gen-rows traced --cpu-offload-gb 36 --gpu-mem 0.92 --batched-tokens 2048" 80m
wait
ls -la $OUT
