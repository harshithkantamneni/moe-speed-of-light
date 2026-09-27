#!/bin/bash
# Phase 4.5 on one A10 instance, part 1. Job 025 ran arm D on another instance than phase 3 (llama.cpp static
# layers came out 0-17 % faster), so window and machine effects are confounded. Here, on the instance that also
# runs arm S (job 039): (a) host read bandwidth (ec-cpubench); (b) a control with the phase-3 windows
# (--n-prefill 128) at the 25 % budget, every model; (c) arm D with correct windows, as job 025.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/ec/setup.sh"
. "$(cd "$(dirname "$0")" && pwd)/ec/windows.sh"
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket|NUMA node\(s\)|L3"
timeout 10m $EC_CPUBENCH > $OUT/cpubench.txt 2>&1; tail -12 $OUT/cpubench.txt
SG=7,10,12,15,16,18,19,20,25,26,27,31; SQ=7,10,12,15,16,18,19,20,25,26,27,32
ctrl() {  # file corpus L E kappa seqs: phase-3 windows (128-token prefill cap), 25 % budget only
  local f=$1 corp=$2 L=$3 E=$4 K=$5 SEQ=$6 tag=${1%.gguf}_ctrl C=$(( $4*2/8 )) n=$(( $3 - $3*2/8 ))
  R $EC_BIN -m $M/$f --corpus $corp --seqs $SEQ --host-experts --ec "slots=$C:policy=dfa:kappa=$K:$MB:stats=$OUT/${tag}_mb_C$C.json" -t 30 --n-prefill 128 --n-decode 192 > $OUT/${tag}_ec.jsonl
  R $EC_BIN -m $M/$f --corpus $corp --seqs $SEQ --ncmoe $n -t 30 --n-prefill 128 --n-decode 192 > $OUT/${tag}_static_n$n.jsonl
  echo "$tag rc=$?"
}
ctrl gpt-oss-20b-MXFP4.gguf $J/tok_gpt-oss-20b.jsonl 24 32 1 $SG
ctrl Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf $J/tok_qwen3_30b.jsonl 48 128 2 $SQ
ctrl Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf $J/tok_qwen3_30b.jsonl 48 128 2 $SQ
ctrl gpt-oss-120b-MXFP4.gguf $J/tok_gpt-oss-120b.jsonl 36 128 1 $SG
run_model gpt-oss-20b-MXFP4.gguf $J/tok_gpt-oss-20b.jsonl 24 32 1 "1 2 4" $SG 1 D
run_model Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf $J/tok_qwen3_30b.jsonl 48 128 2 "1 2 4" $SQ 1 D
run_model Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf $J/tok_qwen3_30b.jsonl 48 128 2 "1 2 4" $SQ 0 D
run_model gpt-oss-120b-MXFP4.gguf $J/tok_gpt-oss-120b.jsonl 36 128 1 "1 2" $SG 0 D
summary
