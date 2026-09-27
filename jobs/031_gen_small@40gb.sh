#!/bin/bash
# Phase 4 (prereg/PROTOCOL.md on main). Replaces 027 (FlashInfer sampler JIT needed ninja/nvcc). OLMoE and gpt-oss-20b with vLLM: G and S responses to the 160 prompts, D scores, gpt-oss format variants (4.1, 4.4).
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/prov/setup.sh"
. "$(cd "$(dirname "$0")" && pwd)/prov/gen.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
( dl olmoe ${HUB[olmoe]}; dl gpt-oss-20b ${HUB[gpt-oss-20b]} ) &
run_gen olmoe olmoe ""
run_gen gpt-oss-20b gpt-oss-20b ""
wait
ls -la $OUT
