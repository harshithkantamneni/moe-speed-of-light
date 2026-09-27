# gen KEY...: vLLM on-policy corpora + scores for each model key (source after setup.sh). Downloads run in the
# background in the given order; each model runs in its own process.
declare -A HUB=([olmoe]=allenai/OLMoE-1B-7B-0125-Instruct [gpt-oss-20b]=openai/gpt-oss-20b
                [qwen3-30b-a3b]=Qwen/Qwen3-30B-A3B-Instruct-2507 [gpt-oss-120b]=openai/gpt-oss-120b
                [qwen3-30b-a3b-fp8]=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
                [mixtral-8x7b]=mistralai/Mixtral-8x7B-Instruct-v0.1 [mixtral-8x7b-awq]=hugging-quants/Mixtral-8x7B-Instruct-v0.1-AWQ-INT4
                [deepseek-v2-lite]=deepseek-ai/DeepSeek-V2-Lite-Chat [qwen1.5-moe]=Qwen/Qwen1.5-MoE-A2.7B-Chat
                [qwen2-57b]=Qwen/Qwen2-57B-A14B-Instruct [qwen2-57b-gptq]=Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4
                [phi3.5-moe]=microsoft/Phi-3.5-MoE-instruct [phi3.5-moe-awq]=danieldk/Phi-3.5-MoE-instruct-AWQ-INT4)
# audit_model KEY GEN-CHECKPOINT TRACE-CHECKPOINT: on-policy corpora (vLLM on GEN, which fits 40 GB), then GPU traces
# of D, G, S with the bf16 TRACE checkpoint; both checkpoints are deleted afterwards (disk)
audit_model() {
  local key=$1 gck=$2 tck=$3
  ( dl $gck ${HUB[$gck]}; [ $tck != $gck ] && dl $tck ${HUB[$tck]} ) &
  run_gen $key $gck "--arms G,S,D" 60m
  wait
  trace_arms $tck $OUT $key D G S
  rm -rf $MD/$gck $MD/$tck
  df -h / | tail -1
}
# vLLM's FlashInfer top-k/top-p sampler JIT-compiles (needs ninja and nvcc; job 027 failed on it): use PyTorch's
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$WORK/vv/bin:$PATH
[ -x $WORK/vv/bin/ninja ] || uv pip install -q --python $WORK/vv/bin/python ninja > /dev/null 2>&1
run_gen() {  # prompts-key checkpoint-key "extra args" timeout
  wait_dl $2 || { echo "$2 download failed"; tail -3 $WORK/dl_$2.log; return; }
  du -sh $MD/$2
  ( cd $P && timeout ${4:-45m} $WORK/vv/bin/python scripts/onpolicy_gen.py --model $1 --model-dir $MD/$2 --out $OUT $3 ) > $OUT/gen_$1$5.log 2>&1
  echo "$1 on $2 rc=$?"; grep -E "^(G|S|D|V1|V2) |S params|Error|Traceback" $OUT/gen_$1$5.log | tail -8
}
declare -A TOK=([olmoe]=tok_olmoe.jsonl [gpt-oss-20b]=tok_gpt-oss-20b.jsonl [qwen3-30b-a3b]=tok_qwen3_30b.jsonl
                [gpt-oss-120b]=tok_gpt-oss-120b.jsonl)
declare -A GEN020C=([gpt-oss-20b]="gpt-oss-20b-MXFP4_gen.jsonl" [gpt-oss-120b]="gpt-oss-120b-MXFP4_gen.jsonl"
                    [qwen3-30b-a3b]="Qwen3-30B-A3B-Instruct-2507-Q4_K_M_gen.jsonl Qwen3-30B-A3B-Instruct-2507-Q8_0_gen.jsonl")
gen() {
  ( for key in "$@"; do dl $key ${HUB[$key]}; done ) &
  for key in "$@"; do
    wait_dl $key || { echo "$key download failed"; tail -3 $WORK/dl_$key.log; continue; }
    du -sh $MD/$key
    local X=""; [ $key = gpt-oss-120b ] && X="--gpu-mem 0.86 --batched-tokens 2048"
    ( cd $P && timeout 45m $WORK/vv/bin/python scripts/onpolicy_gen.py --model $key --model-dir $MD/$key --out $OUT $X ) > $OUT/gen_$key.log 2>&1
    echo "$key rc=$?"; grep -E "^(G|S|D|V1|V2) |S params|Error|Traceback" $OUT/gen_$key.log | tail -8
  done
  wait
}
trc() {  # key corpus name (skipped when an earlier job already produced name.npz)
  [ -f "$2" ] || { echo "missing $2"; return; }
  ls "$(dirname "$OUT")"/*/"$3.npz" > /dev/null 2>&1 && { echo "$3 exists: $(ls "$(dirname "$OUT")"/*/"$3.npz")"; return; }
  # MoE sub-block in chunks of MOE_CHUNK tokens (job 032 ran out of memory on 190k tokens at once; exact, see tests)
  ( cd $P && timeout 40m $WORK/tv/bin/python scripts/trace_corpus.py --repo $MD/$1 --hub-id ${HUB[$1]} --corpus "$2" --out $OUT/$3.npz --device cuda --moe-chunk ${MOE_CHUNK:-16384} ) > $OUT/$3.log 2>&1
  echo "$3 rc=$?"; tail -1 $OUT/$3.log
}
trace_arms() {  # checkpoint-key gen-dir corpus-key arms...: trace the gen job's corpora
  local key=$1 G=$2 ck=$3; shift 3
  wait_dl $key || { echo "$key not downloaded"; return; }
  for arm in "$@"; do trc $key "$G/corp_${ck}_$arm.jsonl" ${ck}_$arm; done
}
trace() {  # G(dir of the gen job) KEY...
  local G=$1; shift
  for key in "$@"; do
    wait_dl $key || { echo "$key not downloaded"; continue; }
    trc $key $P/data/${TOK[$key]} ${key}_tok
    for arm in D G S; do trc $key "$G/corp_${key}_$arm.jsonl" ${key}_$arm; done
    for f in ${GEN020C[$key]}; do trc $key $P/data/gen020c/$f ${key}_gen020c_${f%_gen.jsonl}; done
  done
}
gpu_tests() {
  ( cd $P && MOSL_DEVICE=cuda timeout 15m $WORK/tv/bin/python tests/test_collect_equivalence.py ) > $OUT/test_gpu_equivalence.log 2>&1
  echo "gpu equivalence rc=$?"; grep -E "^OK|Error" $OUT/test_gpu_equivalence.log
}
