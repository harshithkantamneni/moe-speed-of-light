# gen KEY...: vLLM on-policy corpora + scores for each model key (source after setup.sh). Downloads run in the
# background in the given order; each model runs in its own process.
declare -A HUB=([olmoe]=allenai/OLMoE-1B-7B-0125-Instruct [gpt-oss-20b]=openai/gpt-oss-20b
                [qwen3-30b-a3b]=Qwen/Qwen3-30B-A3B-Instruct-2507 [gpt-oss-120b]=openai/gpt-oss-120b)
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
trc() {  # key corpus name
  [ -f "$2" ] || { echo "missing $2"; return; }
  ( cd $P && timeout 40m $WORK/tv/bin/python scripts/trace_corpus.py --repo $MD/$1 --hub-id ${HUB[$1]} --corpus "$2" --out $OUT/$3.npz --device cuda ) > $OUT/$3.log 2>&1
  echo "$3 rc=$?"; tail -1 $OUT/$3.log
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
