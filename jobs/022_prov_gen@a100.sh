#!/bin/bash
# Phase 4 (prereg/PROTOCOL.md on main, sections 4.1 and 4.4). For OLMoE-1B-7B-0125-Instruct, gpt-oss-20b,
# Qwen3-30B-A3B-Instruct-2507 and gpt-oss-120b, with vLLM: greedy (G) and generation-config sampling (S)
# responses to the 160 corpus prompts, teacher-forced NLL of the 160 dataset conversations (D), and for gpt-oss
# the format variants V1 (own analysis inserted) and V2 (reasoning low) of the 35 traced conversations.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/prov/setup.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
( for spec in "olmoe allenai/OLMoE-1B-7B-0125-Instruct" "gpt-oss-20b openai/gpt-oss-20b" \
              "qwen3-30b-a3b Qwen/Qwen3-30B-A3B-Instruct-2507" "gpt-oss-120b openai/gpt-oss-120b"; do dl $spec; done ) &
for key in olmoe gpt-oss-20b qwen3-30b-a3b gpt-oss-120b; do
  wait_dl $key || { echo "$key download failed"; tail -3 $WORK/dl_$key.log; continue; }
  du -sh $MD/$key
  X=""; [ $key = gpt-oss-120b ] && X="--gpu-mem 0.86 --batched-tokens 2048"
  ( cd $P && timeout 35m $WORK/vv/bin/python scripts/onpolicy_gen.py --model $key --model-dir $MD/$key --out $OUT $X ) > $OUT/gen_$key.log 2>&1
  echo "$key rc=$?"; grep -E "^(G|S|D|V1|V2) |S params|Error|Traceback" $OUT/gen_$key.log | tail -8
done
wait
ls -la $OUT
