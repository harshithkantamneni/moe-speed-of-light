#!/bin/bash
# Phase 4, 4.1: GPU routing traces (collector, fp32, TF32 off) of job 027's D/G/S corpora, the committed dataset
# token files (GPU-vs-CPU exactness check) and job 020c's samples, for OLMoE and gpt-oss-20b. GPU equivalence tests first.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/prov/setup.sh"
. "$(cd "$(dirname "$0")" && pwd)/prov/gen.sh"
gpu_tests
trace "$(dirname "$OUT")/027_gen_small@40gb" olmoe gpt-oss-20b
ls -la $OUT
