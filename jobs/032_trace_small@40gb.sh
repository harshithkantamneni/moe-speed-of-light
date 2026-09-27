#!/bin/bash
# Phase 4 (prereg/PROTOCOL.md on main). GPU traces of job 031's D, G and S corpora for OLMoE and gpt-oss-20b (4.1). The dataset token files and job 020c's samples were traced in job 028.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/prov/setup.sh"
. "$(cd "$(dirname "$0")" && pwd)/prov/gen.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
G=$(dirname "$OUT")/031_gen_small@40gb
trace_arms olmoe $G olmoe D G S
trace_arms gpt-oss-20b $G gpt-oss-20b D G S
ls -la $OUT
