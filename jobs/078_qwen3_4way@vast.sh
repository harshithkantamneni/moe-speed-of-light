#!/bin/bash
# Job 078: second model. Qwen3-30B-A3B in BF16 (48 MoE layers, 128 experts, top-8, 9.44 MB per expert) on one
# RTX 5090 + Ryzen 9 9950X (Zen 5, AVX-512 BF16): stock llama.cpp, our expert cache (ours v2, with and without FETCH),
# FreeToken (offload and hybrid) and KTransformers (kt-kernel + sglang-kt 0.7.0.post4, BF16 CPU experts, exact:
# no deferral), at equal GPU expert memory: 12.5% / 25% / 43.75% of experts
#   llama.cpp --n-cpu-moe 42 / 36 / 27 = ours 16 / 32 / 56 per layer = FreeToken rate 0.125 / 0.25 / 0.4375
#   = KTransformers --kt-num-gpu-experts 16 / 32 / 56 per layer (static, expert ids 0..N-1).
# Same weights everywhere: HF Qwen/Qwen3-30B-A3B, and a BF16 GGUF converted from that snapshot. 30 AIME-25 problems,
# 256 tokens, greedy, thinking on, held-out warm-up, session; one launch per configuration. GPU-side sampling for the
# llama-server systems (074b). Recipe: research_notes/MoE offload system race plan/qwen3_recipe.md.
# Predictions, committed before launch:
#   1. ours (better of FETCH on/off) ahead of FreeToken's better backend at 12.5% and 25%;
#   2. ours ahead of KTransformers at every budget;
#   3. ours >= 1.5x stock llama.cpp at every budget.
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip python3-dev numactl >> $OUT/apt.txt 2>&1
CORES=$(usable_cores); echo "usable physical cores $CORES (nproc $(nproc))" | tee $OUT/cores.txt
grep -o 'avx512_bf16' /proc/cpuinfo | head -1 | tee $OUT/avx512bf16.txt
R() { local t=$1; shift; timeout "$t" "$@"; }
python3 -m pip install -q --break-system-packages uv huggingface_hub hf_xet > $OUT/pip_uv.txt 2>&1
nvcc --version | tail -2 | tee $OUT/nvcc_version.txt
HFD=$M/Qwen3-30B-A3B; mkdir -p $M
( t0=$(date +%s); python3 -c "import sys; from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-30B-A3B', local_dir=sys.argv[1])" "$HFD"
  echo "hf download rc=$? in $(( $(date +%s) - t0 )) s"; du -sh "$HFD" ) > $OUT/dl_hf.txt 2>&1 &
DLH=$!
# FreeToken (pinned commit, its own venv: CUDA 13 torch)
FT=$WORK/ft; export FT_DIR=$FT
( cd $WORK && git init -q ft && cd ft && git remote add origin https://github.com/FlashML-org/FreeToken &&
  git fetch -q --depth 1 origin 0d652e73a452d014ac5441a15baa75348e9fcb0a && git checkout -q FETCH_HEAD ) || echo "FreeToken clone FAILED"
( cd $FT && uv venv -q .venv && . .venv/bin/activate && time uv pip install -q -e ".[accel]" huggingface_hub hf_xet gguf sentencepiece transformers safetensors ) > $OUT/ft_install.txt 2>&1
PY=$FT/.venv/bin/python; FTBIN=$FT/.venv/bin/ft
$PY -c "import freetoken, huggingface_hub" > $OUT/ft_import.txt 2>&1 || { echo "FreeToken not importable, stopping"; tail -20 $OUT/ft_install.txt; exit 3; }
# KTransformers (route A: PyPI wheels, own venv, torch 2.9.1)
( python3 -m venv /opt/kt && . /opt/kt/bin/activate && pip install -q -U pip && time pip install -q "ktransformers[sglang]==0.7.0.post4" &&
  python -c "import kt_kernel as k; print('kt_kernel', k.__version__, getattr(k, '__cpu_variant__', '?'))" &&
  python -c "import torch, sgl_kernel; print('torch', torch.__version__, torch.cuda.get_device_capability())" ) > $OUT/kt_install.txt 2>&1
echo "KTransformers install rc=$?"; tail -3 $OUT/kt_install.txt
KTPY=/opt/kt/bin/python
$PY -c "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('math-ai/aime25','test.jsonl',repo_type='dataset'), '$WORK/aime25.jsonl')" && export FREETOKEN_AIME25_JSONL=$WORK/aime25.jsonl
$PY - "$WORK/warmup.txt" <<'PY' > $OUT/warmup_source.txt 2>&1
import sys
sys.path.insert(0, "/work/ft/benchmarks")
import bench_decode_moe as B
text = None
for repo, fn in (("math-ai/aime24", "test.jsonl"), ("HuggingFaceH4/aime_2024", "train.jsonl")):
    try:
        from huggingface_hub import hf_hub_download
        text, _ = B.load_problem(hf_hub_download(repo, fn, repo_type="dataset"), 0)
        print("warm-up prompt:", repo, fn, "problem 0"); break
    except Exception as e:
        print("no", repo, fn, repr(e)[:200])
if text is None:
    text = "Let S be the set of positive integers n at most 2024 such that n^2 + n + 41 is divisible by 43. Find the number of elements of S.\n" + B.BOXED_INSTRUCTION
open(sys.argv[1], "w").write(text)
PY
build_tree $WORK/lc-ec "$J/llama.cpp-expert-cache-4da6337.patch" llama-server
build_tree $WORK/lc-stock "" llama-server
EC_SERVER=$WORK/lc-ec/build/bin/llama-server
wait $DLH; cat $OUT/dl_hf.txt
[ -f "$HFD/config.json" ] && echo "HF checkpoint ok" || { echo "HF checkpoint MISSING"; exit 3; }
# BF16 GGUF from the same snapshot (identical weights and chat template)
GG=$M/Qwen3-30B-A3B-BF16.gguf
( t0=$(date +%s); PYTHONPATH=$WORK/lc-stock/gguf-py $PY $WORK/lc-stock/convert_hf_to_gguf.py "$HFD" --outtype bf16 --outfile $GG
  echo "convert rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $GG ) > $OUT/convert.txt 2>&1
tail -3 $OUT/convert.txt
[ -s $GG ] || { echo "GGUF conversion failed"; exit 3; }
platform
( cd $FT && R 20m $FTBIN bench bw --dtype bf16 ) > $OUT/ft_bench_bw.txt 2>&1; cp ~/.cache/freetoken/benchbw/*.json $OUT/ 2>/dev/null
H=$(( CORES > 4 ? CORES - 2 : 2 ))
P=$(seq -s, 0 29)
LXB='{"min_p": 0.0, "seed": 1234, "backend_sampling": true}'
LS="-m $GG -ngl 99 -fa on -lm none -t $CORES -np 1 -c 2048 --no-webui --jinja"
OT="-ot '\\.ffn_(up|down|gate|gate_up)_exps=CUDA_Host'"
ECENV="env LLAMA_EC_POLICY=dfa LLAMA_EC_KAPPA=1 LLAMA_EC_MAILBOX=1 LLAMA_EC_TDEC=1 LLAMA_EC_HELPERS=$H"
waitfree() {
  for i in $(seq 120); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-0}" -lt 1500 ] && return; sleep 2; done
  echo "GPU memory still in use: $u MiB"
}
cl() {  # label meta [client args...]
  local lab=$1 meta=$2; shift 2
  waitfree
  R 60m $PY $J/bs1_client.py --model $HFD --problems $P --decode 256 --greedy --warmup once --warmup-prompt $WORK/warmup.txt \
    --launch 1 --label $lab --meta "$meta" --json $OUT/bs1.jsonl --log $OUT/srv_${lab}.log "$@"
  echo "$lab rc=$?"
}
KT="env SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false $KTPY -m sglang.launch_server --host 127.0.0.1 --model-path $HFD --kt-weight-path $HFD \
  --kt-method BF16 --kt-cpuinfer $CORES --kt-threadpool-count 1 --kt-numa-nodes 0 --kt-expert-placement-strategy uniform \
  --kt-max-deferred-experts-per-token 0 --attention-backend triton --sampling-backend pytorch --context-length 2048 \
  --max-total-tokens 2048 --chunked-prefill-size 2048 --max-running-requests 1 --cuda-graph-max-bs 1 --tensor-parallel-size 1 \
  --served-model-name qwen3-30b-a3b"
for row in "12.5:42:16:0.125:0.85:0.9" "25:36:32:0.25:0.85:0.9" "43.75:27:56:0.4375:0.93:0.95"; do
  IFS=: read pct n C r msf mr <<< "$row"
  cl llama_n$n "{\"system\": \"llama.cpp\", \"pct\": $pct, \"ncmoe\": $n}" --extra "$LXB" --cmd "$STOCK_SERVER $LS -ncmoe $n --port {port}"
  cl oursv2_C$C "{\"system\": \"ours v2\", \"pct\": $pct, \"C\": $C, \"fetch\": 0}" --extra "$LXB" \
    --cmd "$ECENV LLAMA_EC_SLOTS=$C LLAMA_EC_STATS=$OUT/srv_oursv2_C$C.json $EC_SERVER $LS $OT --port {port}"
  cl oursv2f_C$C "{\"system\": \"ours v2\", \"pct\": $pct, \"C\": $C, \"fetch\": 1}" --extra "$LXB" \
    --cmd "$ECENV LLAMA_EC_SLOTS=$C LLAMA_EC_FETCH=0,1,1,2,3,3,4,5,6 LLAMA_EC_STATS=$OUT/srv_oursv2f_C$C.json $EC_SERVER $LS $OT --port {port}"
  ( cd $FT; for b in offload hybrid; do
      cl ft_${b}_r$r "{\"system\": \"FreeToken $b\", \"pct\": $pct, \"rate\": $r}" --ft $b --cache-rate $r --mem-ratio $mr
    done )
  cl kt_E$C "{\"system\": \"KTransformers\", \"pct\": $pct, \"gpu_experts_per_layer\": $C}" --ready models \
    --cmd "$KT --kt-num-gpu-experts $C --mem-fraction-static $msf --port {port}"
done
cd $WORK
for f in $OUT/srv_*.log; do tail -n 60 "$f" > "$f.tail"; done; find $OUT -name 'srv_*.log' -delete
$PY - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, sys, os, statistics as st
out = sys.argv[1]
rows = [json.loads(l) for l in open(f"{out}/bs1.jsonl")] if os.path.exists(f"{out}/bs1.jsonl") else []
by = {}
for r in rows: by.setdefault(r["label"], []).append(r)
for lab, v in by.items():
    print(f"{lab:26s} {st.mean(r['decode_tok_s'] for r in v):7.2f} tok/s (n={len(v)})  vram {sorted({round(r.get('vram_used_gib') or 0, 1) for r in v})}  prompt_tokens {sorted({r.get('prompt_tokens') for r in v})[:3]}")
PY
