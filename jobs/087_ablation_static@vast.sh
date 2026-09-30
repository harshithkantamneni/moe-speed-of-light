#!/bin/bash
# Job 087: the ablation ladder again, with a real static cache. Job 082's "static cache" step loaded no experts: the
# static policy takes its resident set from LLAMA_EC_INIT, which was not set, so every expert ran on the CPU through
# the host-driven path (hits 0 in its counters). Here, on one machine (RTX 5090 + Ryzen 9 9950X3D, listing A, jobs
# 079/083/086), at 25% (C32), AIME-25 problems 0-14, 256 tokens, greedy, session:
#   0  stock llama.cpp (-ncmoe 27 / 36)
#   1a no expert on the GPU, host-driven CPU path (job 082's step, kept to explain it)
#   1  static cache: each layer's 32 most requested experts, profiled on other text (the models' own MATH-500 long
#      outputs of job 082: ec2/counts_mathtext_*_C32.json), host-driven CPU path, CPU sampling
#   1h static cache profiled in hindsight on the measured problems (Table 1 runs: ec2/counts_aime_*_C32.json); not a
#      ladder step, the best any static placement could do
#   2  LRU replacement, 3 decayed frequency, 4 GPU-signalled CPU helpers, 5 slot maps on the GPU, 6 GPU-side sampling,
#   7 FETCH with the fixed table, 8 FETCH with the law's table (= ours), as in job 082.
# Trace simulation (job 084 traces) gives static-hindsight / LRU / decayed-frequency hit rates of 0.750 / 0.775 / 0.785
# (gpt-oss) and 0.719 / 0.737 / 0.750 (Qwen3): per-layer placement, not replacement, should carry most of the gain.
# Predictions, committed before launch:
#   1. the profiled static cache runs >= 1.3x stock llama.cpp on both models;
#   2. replacing experts dynamically adds less than placing them: step 3 / step 1 < step 1 / step 0 on both models;
#   3. the decayed-frequency policy (step 3) beats the profiled static cache (paired CI > 1) on both models;
#   4. every step after the static cache is >= -1%, and ours runs >= 2x stock on both models.
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip >> $OUT/apt.txt 2>&1
CORES=$(usable_cores); echo "usable physical cores $CORES (nproc $(nproc))" | tee $OUT/cores.txt
R() { local t=$1; shift; timeout "$t" "$@"; }
FT=$WORK/ft; export FT_DIR=$FT   # FreeToken's benchmark module supplies the AIME prompt format (not installed or run)
( cd $WORK && git init -q ft && cd ft && git remote add origin https://github.com/FlashML-org/FreeToken &&
  git fetch -q --depth 1 origin 0d652e73a452d014ac5441a15baa75348e9fcb0a && git checkout -q FETCH_HEAD ) || echo "FreeToken clone FAILED"
python3 -m pip install -q --break-system-packages uv huggingface_hub hf_xet > $OUT/pip_uv.txt 2>&1
VE=$WORK/venv
( uv venv -q $VE && . $VE/bin/activate && uv pip install -q torch --index-url https://download.pytorch.org/whl/cpu &&
  uv pip install -q numpy transformers sentencepiece safetensors protobuf huggingface_hub hf_xet ) > $OUT/venv_install.txt 2>&1
echo "venv install rc=$?"
PY=$VE/bin/python
$PY -c "import torch, transformers, huggingface_hub; print(torch.__version__, transformers.__version__)" | tee $OUT/venv_ok.txt
mkdir -p $M
HFD=$M/gpt-oss-120b
( t0=$(date +%s)
  $PY -c "import sys; from huggingface_hub import snapshot_download as s; s('openai/gpt-oss-120b', local_dir=sys.argv[1], allow_patterns=['*.json', '*.jinja', '*.txt'])" "$HFD"
  $PY -c "import sys; from huggingface_hub import hf_hub_download as d; d('ggml-org/gpt-oss-120b-GGUF', 'gpt-oss-120b-MXFP4.gguf', local_dir=sys.argv[1])" "$M"
  echo "gguf hf download rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $M/gpt-oss-120b-MXFP4.gguf
  if [ ! -s $M/gpt-oss-120b-MXFP4.gguf ]; then
    rm -f $M/gpt-oss-120b-MXFP4.gguf
    for i in 1 2 3; do getmodel ggml-org/gpt-oss-120b-GGUF gpt-oss-120b-MXFP4.gguf; [ -f $M/gpt-oss-120b-MXFP4.gguf ] && break; done
  fi
) > $OUT/dl_gguf.txt 2>&1 &
DLG=$!
HFQ=$M/Qwen3-30B-A3B
( t0=$(date +%s); $PY -c "import sys; from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-30B-A3B', local_dir=sys.argv[1])" "$HFQ"
  echo "qwen3 hf download rc=$? in $(( $(date +%s) - t0 )) s"; du -sh "$HFQ" ) > $OUT/dl_qwen3.txt 2>&1 &
DLQ=$!
$PY -c "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('math-ai/aime25','test.jsonl',repo_type='dataset'), '$WORK/aime25.jsonl')" && export FREETOKEN_AIME25_JSONL=$WORK/aime25.jsonl
wc -l $WORK/aime25.jsonl | tee $OUT/aime25_count.txt
$PY - "$WORK/warmup.txt" <<'PY' > $OUT/warmup_source.txt 2>&1
import json, sys
sys.path.insert(0, "/work/ft/benchmarks")
import bench_decode_moe as B
text = None
for repo, fn in (("math-ai/aime24", "test.jsonl"), ("HuggingFaceH4/aime_2024", "train.jsonl")):
    try:
        from huggingface_hub import hf_hub_download
        text, _ = B.load_problem(hf_hub_download(repo, fn, repo_type="dataset"), 0)
        print("warm-up prompt:", repo, fn, "problem 0")
        break
    except Exception as e:
        print("no", repo, fn, repr(e)[:200])
if text is None:
    text = ("Let S be the set of positive integers n at most 2024 such that n^2 + n + 41 is divisible by 43. "
            "Find the number of elements of S.\n" + B.BOXED_INSTRUCTION)
    print("warm-up prompt: fixed fallback text")
open(sys.argv[1], "w").write(text)
PY
cat $OUT/warmup_source.txt
build_tree $WORK/lc-ec "$J/llama.cpp-expert-cache-4da6337.patch" llama-server
build_tree $WORK/lc-stock "" llama-server
EC_SERVER=$WORK/lc-ec/build/bin/llama-server
wait $DLG $DLQ
cat $OUT/dl_gguf.txt $OUT/dl_qwen3.txt
F=$M/gpt-oss-120b-MXFP4.gguf
[ -s "$F" ] && echo "GGUF ok $(stat -c %s $F) bytes" || echo "GGUF MISSING"
GQ=$M/Qwen3-30B-A3B-BF16.gguf
( t0=$(date +%s); PYTHONPATH=$WORK/lc-stock/gguf-py $PY $WORK/lc-stock/convert_hf_to_gguf.py "$HFQ" --outtype bf16 --outfile $GQ
  echo "convert rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $GQ ) > $OUT/convert_qwen3.txt 2>&1
tail -2 $OUT/convert_qwen3.txt
platform
H=$(( CORES > 4 ? CORES - 2 : 2 ))
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
grep -q "concurrent t=" $OUT/concur.txt || { echo "concur incomplete, retrying"; R 15m $WORK/concur > $OUT/concur.txt 2>&1; }
LAWG=$(python3 $J/fetch_table.py $OUT/concur.txt $H 13253760 4 $OUT/fetch_table_law_gptoss.json 37)
LAWQ=$(python3 $J/fetch_table.py $OUT/concur.txt $H 9437184 8 $OUT/fetch_table_law_qwen3.json 48)
CURG=0,1,1,2,3; CURQ=0,1,1,2,3,3,4,5,6
echo "law tables ($(date -u +%FT%TZ)): gpt-oss $LAWG (fixed $CURG); qwen3 $LAWQ (fixed $CURQ)" | tee $OUT/tables.txt
[ -n "$LAWG" ] && [ -n "$LAWQ" ] || { echo "no law table"; exit 3; }
OT="-ot '\\.ffn_(up|down|gate|gate_up)_exps=CUDA_Host'"
GPUS='{"min_p": 0.0, "seed": 1234, "backend_sampling": true}'
CPUS='{"min_p": 0.0, "seed": 1234, "backend_sampling": false}'
waitfree() {
  for i in $(seq 120); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-0}" -lt 1500 ] && return; sleep 2; done
  echo "GPU memory still in use: $u MiB"
}
P15=$(seq -s, 0 14)
cl() {  # model-dir label meta [client args...]
  local hf=$1 lab=$2 meta=$3; shift 3
  waitfree
  R 60m $PY $J/bs1_client.py --model $hf --problems $P15 --decode 256 --greedy --warmup once --warmup-prompt $WORK/warmup.txt \
    --launch 1 --label $lab --meta "$meta" --json $OUT/bs1.jsonl --log $OUT/srv_${lab}.log "$@"
  echo "$lab rc=$?"
}
# ours with a given cache configuration: label table(none|x,y,..) policy mailbox(0|1) mapsgpu(0|1) sampling-json init(none|file) model-dir server-args
oursrun() {
  local lab=$1 t=$2 pol=$3 mb=$4 mg=$5 samp=$6 init=$7 hf=$8 ls=$9
  local e="env LLAMA_EC_POLICY=$pol LLAMA_EC_KAPPA=1 LLAMA_EC_TDEC=1 LLAMA_EC_SLOTS=32 LLAMA_EC_STATS=$OUT/srv_${lab}.json"
  [ "$mb" = 1 ] && e="$e LLAMA_EC_MAILBOX=1 LLAMA_EC_HELPERS=$H"
  [ "$mg" = 0 ] && e="$e LLAMA_EC_MAPS_CPU=1"
  [ "$t" != none ] && e="$e LLAMA_EC_FETCH=$t"
  [ "$init" != none ] && e="$e LLAMA_EC_INIT=$init"
  cl $hf $lab "{\"system\": \"ours\", \"C\": 32, \"table\": \"$t\", \"policy\": \"$pol\", \"mailbox\": $mb, \"maps_gpu\": $mg, \"init\": \"$init\"}" \
    --extra "$samp" --cmd "$e $EC_SERVER $ls $OT --port {port}"
}
stock() {  # label ncmoe model-dir server-args
  cl $3 $1 "{\"system\": \"llama.cpp\", \"ncmoe\": $2}" --extra "$GPUS" --cmd "$STOCK_SERVER $4 -ncmoe $2 --port {port}"
}
ladder() {  # prefix ncmoe table-law table-fixed profiled-counts hindsight-counts model-dir server-args
  local p=$1 n=$2 law=$3 fix=$4 prof=$5 hind=$6 hf=$7 ls=$8
  stock ${p}_abl_0_stock_n$n $n $hf "$ls"
  oursrun ${p}_abl_1a_nocache none static 0 0 "$CPUS" none $hf "$ls"
  oursrun ${p}_abl_1_static none static 0 0 "$CPUS" $prof $hf "$ls"
  oursrun ${p}_abl_1h_static_hindsight none static 0 0 "$CPUS" $hind $hf "$ls"
  oursrun ${p}_abl_2_lru none lru 0 0 "$CPUS" none $hf "$ls"
  oursrun ${p}_abl_3_dfa none dfa 0 0 "$CPUS" none $hf "$ls"
  oursrun ${p}_abl_4_mailbox none dfa 1 0 "$CPUS" none $hf "$ls"
  oursrun ${p}_abl_5_mapsgpu none dfa 1 1 "$CPUS" none $hf "$ls"
  oursrun ${p}_abl_6_gpusample none dfa 1 1 "$GPUS" none $hf "$ls"
  oursrun ${p}_abl_7_fetchfixed $fix dfa 1 1 "$GPUS" none $hf "$ls"
  oursrun ${p}_abl_8_fetchlaw $law dfa 1 1 "$GPUS" none $hf "$ls"
}
LSG="-m $F -ngl 99 -fa on -lm none -t $CORES -np 1 -c 8448 --no-webui"
ladder g 27 $LAWG $CURG $J/counts_mathtext_gptoss_C32.json $J/counts_aime_gptoss_C32.json $HFD "$LSG"
if [ -s "$GQ" ]; then
  LSQ="-m $GQ -ngl 99 -fa on -lm none -t $CORES -np 1 -c 4096 --no-webui --jinja"
  ladder q 36 $LAWQ $CURQ $J/counts_mathtext_qwen3_C32.json $J/counts_aime_qwen3_C32.json $HFQ "$LSQ"
else
  echo "Qwen3 GGUF missing: Qwen3 part skipped"
fi
cd $WORK
for f in $OUT/srv_*.log; do tail -n 40 "$f" > "$f.tail"; done; find $OUT -name 'srv_*.log' -delete
$PY - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, sys, os, statistics as st
out = sys.argv[1]
rows = [json.loads(l) for l in open(f"{out}/bs1.jsonl")] if os.path.exists(f"{out}/bs1.jsonl") else []
by = {}
for r in rows: by.setdefault(r["label"], []).append(r["decode_tok_s"])
for lab, v in by.items():
    s = f"{out}/srv_{lab}.json"
    hit = json.load(open(s))["hit_rate"] if os.path.exists(s) else None
    print(f"{lab:32s} {st.mean(v):7.2f} tok/s (n={len(v)})" + (f"  hit {hit:.3f}" if hit is not None else ""))
PY
