#!/bin/bash
# Retry of job 041 (vLLM install on aarch64). Paper: routing traces of the audit's models (prereg 4.6 uses on-policy S traces where we have the model).
# Mixtral-8x7B-Instruct: generated with its AWQ-INT4 checkpoint (bf16 does not fit 40 GB), traced with bf16. G, S, D as in phase 4.1 (160 prompts, <= 1024 new tokens); GPU traces of D, G and S.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/prov/setup.sh"
. "$(cd "$(dirname "$0")" && pwd)/prov/gen.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv
audit_model mixtral-8x7b mixtral-8x7b-awq mixtral-8x7b
ls -la $OUT
