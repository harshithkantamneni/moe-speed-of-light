#!/bin/bash
# Paper: routing traces of the audit's models (prereg 4.6 uses on-policy S traces where we have the model).
# Qwen2-57B-A14B-Instruct: generated with the official GPTQ-Int4 checkpoint, traced with bf16. G, S, D as in phase 4.1 (160 prompts, <= 1024 new tokens); GPU traces of D, G and S.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/prov/setup.sh"
. "$(cd "$(dirname "$0")" && pwd)/prov/gen.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv
audit_model qwen2-57b qwen2-57b-gptq qwen2-57b
ls -la $OUT
