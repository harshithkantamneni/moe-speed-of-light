#!/bin/bash
# Paper: routing traces of the audit's models. Qwen2-57B-A14B-Instruct, generated with vLLM FP8 on-the-fly quantization of the bf16
# checkpoint on the GH200 (96 GB; the AWQ/GPTQ kernels of jobs 045-047 need a newer driver: PTX toolchain error),
# traced with the bf16 checkpoint. G, S, D as in phase 4.1.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/prov/setup.sh"
. "$(cd "$(dirname "$0")" && pwd)/prov/gen.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
audit_model qwen2-57b qwen2-57b qwen2-57b "--quantization fp8 --gpu-mem 0.9"
ls -la $OUT
