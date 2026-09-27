#!/bin/bash
# Phase 4 (prereg/PROTOCOL.md on main, 4.1 and 4.4), replacing job 022 (vLLM wheel for CUDA 13 on a CUDA 12.8
# driver; hf download pattern bug). Qwen3-30B-A3B (bf16) and gpt-oss-120b (need 80 GB): with vLLM, greedy (G) and generation-config sampling (S) responses
# to the 160 prompts, teacher-forced NLL of the 160 dataset conversations (D), and for gpt-oss the format variants
# V1/V2 of the 35 traced conversations.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/prov/setup.sh"
. "$(cd "$(dirname "$0")" && pwd)/prov/gen.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
gen qwen3-30b-a3b gpt-oss-120b
ls -la $OUT
