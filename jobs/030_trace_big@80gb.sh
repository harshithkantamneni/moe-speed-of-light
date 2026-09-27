#!/bin/bash
# Phase 4, 4.1: as job 028, for Qwen3-30B-A3B and gpt-oss-120b (job 029's corpora).
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/prov/setup.sh"
. "$(cd "$(dirname "$0")" && pwd)/prov/gen.sh"
gpu_tests
trace "$(dirname "$OUT")/029_gen_big@80gb" qwen3-30b-a3b gpt-oss-120b
ls -la $OUT
