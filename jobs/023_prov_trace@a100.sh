#!/bin/bash
# Phase 4, 4.1: exact routing traces of the D, G and S corpora (job 022) for OLMoE, gpt-oss-20b and Qwen3-30B-A3B,
# with the layer-streamed collector on the GPU (fp32, TF32 off). Also re-traces the committed dataset token files
# (the GPU-vs-CPU exactness check) and job 020c's sampled continuations. The equivalence tests run first, on GPU.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/prov/setup.sh"
G=$(dirname "$OUT")/022_prov_gen@a100
( cd $P && MOSL_DEVICE=cuda timeout 15m $WORK/tv/bin/python tests/test_collect_equivalence.py ) > $OUT/test_gpu_equivalence.log 2>&1
echo "gpu equivalence rc=$?"; grep -E "^OK|Error" $OUT/test_gpu_equivalence.log
trc() {  # key hub-id corpus name
  [ -f "$3" ] || { echo "missing $3"; return; }
  ( cd $P && timeout 40m $WORK/tv/bin/python scripts/trace_corpus.py --repo $MD/$1 --hub-id $2 --corpus "$3" --out $OUT/$4.npz --device cuda ) > $OUT/$4.log 2>&1
  echo "$4 rc=$?"; tail -1 $OUT/$4.log
}
run() {  # key hub-id tokfile gen020c-files...
  local key=$1 hub=$2 tokf=$3; shift 3
  wait_dl $key || { echo "$key not downloaded"; return; }
  trc $key $hub $P/data/$tokf ${key}_tok
  for arm in D G S; do trc $key $hub "$G/corp_${key}_$arm.jsonl" ${key}_$arm; done
  for f in "$@"; do trc $key $hub $P/data/gen020c/$f ${key}_gen020c_${f%_gen.jsonl}; done
}
run olmoe allenai/OLMoE-1B-7B-0125-Instruct tok_olmoe.jsonl
run gpt-oss-20b openai/gpt-oss-20b tok_gpt-oss-20b.jsonl gpt-oss-20b-MXFP4_gen.jsonl
run qwen3-30b-a3b Qwen/Qwen3-30B-A3B-Instruct-2507 tok_qwen3_30b.jsonl Qwen3-30B-A3B-Instruct-2507-Q4_K_M_gen.jsonl Qwen3-30B-A3B-Instruct-2507-Q8_0_gen.jsonl
ls -la $OUT
