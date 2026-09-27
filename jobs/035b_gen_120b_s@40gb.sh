#!/bin/bash
# Phase 4 (4.1, 4.5): gpt-oss-120b sampled responses (arm S, generation-config sampling, seed = corpus index) for
# the 35 traced prompts, with vLLM weight offload on the 40 GB A100. Needed for the A10 S-arm re-run.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/prov/setup.sh"
. "$(cd "$(dirname "$0")" && pwd)/prov/gen.sh"
( dl gpt-oss-120b ${HUB[gpt-oss-120b]} ) &
run_gen gpt-oss-120b gpt-oss-120b "--arms S --gen-rows traced --cpu-offload-gb 36 --gpu-mem 0.92 --batched-tokens 2048" 80m
wait
ls -la $OUT
