#!/bin/bash
# Collector exactness on the GPU for all eight tiny checkpoints (seven architectures), with the NLL check added after
# the Phi-3.5 final-norm/lm_head bias fix: selections equal the reference transformers model as prefill and as
# KV-cache decode, and per-token NLL within 1e-3 (previous GPU runs, jobs 023/028, covered four architectures).
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/prov/setup.sh"
. "$(cd "$(dirname "$0")" && pwd)/prov/gen.sh"
gpu_tests
cat $OUT/test_gpu_equivalence.log | grep -E "^OK|Error|assert" 
