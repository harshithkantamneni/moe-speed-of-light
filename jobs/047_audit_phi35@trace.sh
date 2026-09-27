#!/bin/bash
# Retry of job 043 (vLLM install on aarch64). Paper: routing traces of the audit's models (prereg 4.6 uses on-policy S traces where we have the model).
# Phi-3.5-MoE-instruct: generated with a community AWQ-INT4 checkpoint, traced with bf16. G, S, D as in phase 4.1 (160 prompts, <= 1024 new tokens); GPU traces of D, G and S.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/prov/setup.sh"
. "$(cd "$(dirname "$0")" && pwd)/prov/gen.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv
audit_model phi3.5-moe phi3.5-moe-awq phi3.5-moe
ls -la $OUT
