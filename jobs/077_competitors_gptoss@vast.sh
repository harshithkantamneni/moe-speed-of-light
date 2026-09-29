#!/bin/bash
# Job 077: two more entrants on gpt-oss-120b, at the same budgets and protocol as job 076 (RTX 5090 + Ryzen 9 9950X,
# 30 AIME-25 problems, 256 tokens, greedy, held-out warm-up, session; one launch per configuration):
#   - Pipelined Sharding (MLSys'26 artifact, deepshnv/pipeshard-mlsys26-ae v2.0.3-mlsys26): -pipe-shard with its GPU
#     buffer -mva calibrated so that measured GPU memory equals stock llama.cpp's at -ncmoe 32 / 27 / 22;
#   - leloch's llama.cpp MoE cache (moe-cache-v2-pr @ e3096b0): --cpu-moe --moe-cache N MiB, one pooled LRU of expert
#     projections, CPU misses, background fills; N for 14 / 32 / 51 experts per layer on average.
# Same-session references at each budget: stock llama.cpp master, ours v2 (with and without FETCH), FreeToken
# (offload and hybrid); the faster of each system's two is its entry. Neither new entrant can sample on the GPU (b6097; August base), which costs them
# ~1.6 ms per token against llama-server master (job 074b); reported with the numbers.
# Predictions, committed before launch:
#   1. both new entrants run end to end on gpt-oss-120b and are faster than stock llama.cpp at 25% and 40%;
#   2. ours v2 (the faster of its two settings) is ahead of both at every budget.
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip >> $OUT/apt.txt 2>&1
CORES=$(usable_cores); echo "usable physical cores $CORES (nproc $(nproc))" | tee $OUT/cores.txt
R() { local t=$1; shift; timeout "$t" "$@"; }
FT=$WORK/ft; export FT_DIR=$FT
( cd $WORK && git init -q ft && cd ft && git remote add origin https://github.com/FlashML-org/FreeToken &&
  git fetch -q --depth 1 origin 0d652e73a452d014ac5441a15baa75348e9fcb0a && git checkout -q FETCH_HEAD ) || echo "FreeToken clone FAILED"
python3 -m pip install -q --break-system-packages uv huggingface_hub hf_xet > $OUT/pip_uv.txt 2>&1
nvcc --version | tail -2 | tee $OUT/nvcc_version.txt   # FreeToken builds kernels with nvcc and needs the CUDA 13 toolkit (torch cu130)
( cd $FT && uv venv -q .venv && . .venv/bin/activate && time uv pip install -q -e ".[accel]" huggingface_hub hf_xet ) > $OUT/ft_install.txt 2>&1
echo "FreeToken install rc=$?"
PY=$FT/.venv/bin/python; FTBIN=$FT/.venv/bin/ft
# stop early (and cheaply) if FreeToken is not usable: the comparison is pointless without it (073 attempt 1 ran on a
# CUDA 12.8 image, the install failed and every client run died)
$PY -c "import freetoken, huggingface_hub" > $OUT/ft_import.txt 2>&1 || { echo "FreeToken not importable, stopping"; cat $OUT/ft_import.txt; tail -20 $OUT/ft_install.txt; exit 3; }
$FTBIN --version > $OUT/ft_version.txt 2>&1
HFD=$M/gpt-oss-120b; mkdir -p $M
( t0=$(date +%s); python3 - "$HFD" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("openai/gpt-oss-120b", local_dir=sys.argv[1], ignore_patterns=["original/*", "metal/*"])
PY
  echo "hf download rc=$? in $(( $(date +%s) - t0 )) s"; du -sh "$HFD" ) > $OUT/dl_hf.txt 2>&1 &
DLH=$!
( t0=$(date +%s)
  python3 -c "import sys; from huggingface_hub import hf_hub_download as d; d('ggml-org/gpt-oss-120b-GGUF', 'gpt-oss-120b-MXFP4.gguf', local_dir=sys.argv[1])" "$M"
  echo "gguf hf download rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $M/gpt-oss-120b-MXFP4.gguf
  if [ ! -s $M/gpt-oss-120b-MXFP4.gguf ]; then
    rm -f $M/gpt-oss-120b-MXFP4.gguf
    for i in 1 2 3; do getmodel ggml-org/gpt-oss-120b-GGUF gpt-oss-120b-MXFP4.gguf; [ -f $M/gpt-oss-120b-MXFP4.gguf ] && break; done
  fi
) > $OUT/dl_gguf.txt 2>&1 &
DLG=$!
# AIME-25 (the 30 measured problems) and a held-out warm-up prompt (AIME-24 problem 0, formatted like theirs)
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
build_tree $WORK/lc-ec "$J/llama.cpp-expert-cache-4da6337.patch" llama-ec-bench llama-server
build_tree $WORK/lc-stock "" llama-server
# --- the two community entrants (recipes: research_notes/MoE offload system race plan/{pipeshard,community_cache}_recipe.md)
( git clone -q https://github.com/deepshnv/pipeshard-mlsys26-ae $WORK/pipeshard && cd $WORK/pipeshard && git checkout -q --detach v2.0.3-mlsys26 &&
  git log -1 --format=%H > $OUT/pipeshard_commit.txt &&
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_CUDA_ARCHITECTURES=120 &&
  cmake --build build -j$(nproc) --target llama-server llama-cli concurrent_profiler gpu_profiler ) > $OUT/build_pipeshard.txt 2>&1
echo "pipeshard build rc=$?"; tail -3 $OUT/build_pipeshard.txt
( git clone -q --filter=blob:limit=2m -b moe-cache-v2-pr https://github.com/leloch/llama.cpp $WORK/leloch && cd $WORK/leloch &&
  git checkout -q e3096b046bb809f7f80bc47801f6579aed1cbc60 && git log -1 --format=%H > $OUT/leloch_commit.txt &&
  cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=ON &&
  cmake --build build -j$(nproc) --target llama-server test-moe-cache test-backend-ops ) > $OUT/build_leloch.txt 2>&1
echo "leloch build rc=$?"; tail -3 $OUT/build_leloch.txt
EC_SERVER=$WORK/lc-ec/build/bin/llama-server
wait $DLG $DLH
cat $OUT/dl_gguf.txt $OUT/dl_hf.txt
F=$M/gpt-oss-120b-MXFP4.gguf
[ -f "$HFD/config.json" ] && echo "HF checkpoint ok" || echo "HF checkpoint MISSING"
[ -s "$F" ] && echo "GGUF ok $(stat -c %s $F) bytes" || echo "GGUF MISSING"
# --- the host
platform
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
( cd $FT && R 20m $FTBIN bench bw ) > $OUT/ft_bench_bw.txt 2>&1; cp ~/.cache/freetoken/benchbw/*.json $OUT/ 2>/dev/null
H=$(( CORES > 4 ? CORES - 2 : 2 )); MB="mailbox=1:tdec=1:helpers=$H"
python3 $J/law_predict.py $OUT/concur.txt $H "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" > $OUT/law_prediction.json 2>&1
echo "law prediction written $(date -u +%FT%TZ)"

P=$(seq -s, 0 29)
LXB='{"min_p": 0.0, "seed": 1234, "backend_sampling": true}'
LS="-m $F -ngl 99 -fa on -lm none -t $CORES -np 1 -c 8448 --no-webui"
OT="-ot '\\.ffn_(up|down|gate|gate_up)_exps=CUDA_Host'"
ECENV="env LLAMA_EC_POLICY=dfa LLAMA_EC_KAPPA=1 LLAMA_EC_MAILBOX=1 LLAMA_EC_TDEC=1 LLAMA_EC_HELPERS=$H"
waitfree() {
  for i in $(seq 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-0}" -lt 1500 ] && return; sleep 2; done
  echo "GPU memory still in use: $u MiB"
}
cl() {  # label launch meta problems warmup decode json [client args...]
  local lab=$1 L=$2 meta=$3 probs=$4 wu=$5 dec=$6 js=$7; shift 7
  waitfree
  R 60m $PY $J/bs1_client.py --model $HFD --problems $probs --decode $dec --greedy --warmup $wu --warmup-prompt $WORK/warmup.txt \
    --launch $L --label $lab --meta "$meta" --json $js --log $OUT/srv_${lab}_L$L.log "$@"
  echo "$lab launch $L rc=$?"
}
# --- Pipelined Sharding: profile once (GPU idle), then calibrate -mva per budget and run
export GGML_CUDA_PIPELINE_SHARDING=1 GGML_CUDA_REGISTER_HOST=1
mkdir -p $WORK/ps_prof
( cd $WORK/ps_prof && time R 40m $WORK/pipeshard/build/bin/concurrent_profiler --cold --fast --threads $CORES && time R 40m $WORK/pipeshard/build/bin/gpu_profiler --fast ) > $OUT/ps_profile.txt 2>&1
echo "pipeshard profile rc=$?"; ls -la $WORK/ps_prof | tee -a $OUT/ps_profile.txt
PSS="$WORK/pipeshard/build/bin/llama-server -m $F -pipe-shard --cpu-profile $WORK/ps_prof/concurrent_results.txt --gpu-profile $WORK/ps_prof/gpu_results.txt -t $CORES -np 1 -c 8448 --jinja --no-webui"
for pair in 32:9728 27:17818 22:25907; do
  n=${pair%:*}; target=${pair#*:}; mva=$(( target - 1024 ))
  cl pipeshard_cal_n$n 1 "{\"system\": \"pipeshard\", \"ncmoe_eq\": $n, \"mva\": $mva}" 0 once 16 $OUT/calib.jsonl --extra "$LXB" --cmd "$PSS -mva $mva --port {port}"
  used=$($PY -c "import json,sys; r=[json.loads(l) for l in open('$OUT/calib.jsonl') if json.loads(l)['label']=='pipeshard_cal_n$n']; print(int((r[-1].get('vram_used_gib') or 0)*1024) if r else 0)")
  if [ "$used" -gt 0 ]; then mva=$(( target - (used - mva) )); fi
  echo "pipeshard ncmoe-eq $n: target $target MiB, calibration run used $used MiB -> mva $mva" | tee -a $OUT/ps_calibration.txt
  cl pipeshard_n$n 1 "{\"system\": \"pipeshard\", \"ncmoe_eq\": $n, \"mva\": $mva}" $P once 256 $OUT/bs1.jsonl --extra "$LXB" --cmd "$PSS -mva $mva --port {port}"
  grep -E "PIPELINE PLAN SUMMARY|current_strategy|n_pinned_layers|pinned_weights|max_vram_alloc|Loaded [0-9]+ benchmark entries" $OUT/srv_pipeshard_n${n}_L1.log | head -20 > $OUT/ps_plan_n$n.txt
done
unset GGML_CUDA_PIPELINE_SHARDING GGML_CUDA_REGISTER_HOST
# --- leloch's cache
( cd $WORK/leloch && CUDA_VISIBLE_DEVICES=0 R 10m ./build/bin/test-moe-cache ) > $OUT/leloch_test.txt 2>&1; echo "leloch test rc=$?"
LLS="$WORK/leloch/build/bin/llama-server -m $F -ngl 99 --cpu-moe --no-repack --fit off -c 8448 -np 1 -fa on -t $CORES -lv 4 --no-webui"
for pair in 14:6358 32:14527 51:23150; do
  E=${pair%:*}; N=${pair#*:}
  cl leloch_E$E 1 "{\"system\": \"leloch v2\", \"E_per_layer\": $E, \"moe_cache_mib\": $N}" $P once 256 $OUT/bs1.jsonl --extra "$LXB" \
    --cmd "env GGML_CUDA_MOE_CACHE_STATS=10800 $LLS --moe-cache $N --port {port}"
  grep -E "MoE cache requested|\[moe-cache\].*(capacity|pool\[0\]|enabled|hits=)" $OUT/srv_leloch_E${E}_L1.log | tail -8 > $OUT/leloch_cache_E$E.txt
done
# --- same-session references
for pair in 32:14 27:32 22:51; do
  n=${pair%:*}; C=${pair#*:}
  cl llama_n$n 1 "{\"system\": \"llama.cpp\", \"ncmoe\": $n}" $P once 256 $OUT/bs1.jsonl --extra "$LXB" --cmd "$STOCK_SERVER $LS -ncmoe $n --port {port}"
done
ours_ref() {  # C fetch(0|1)
  local lab=oursv2_C$1; local fe=""
  [ "$2" = 1 ] && { lab=oursv2f_C$1; fe="LLAMA_EC_FETCH=0,1,1,2,3"; }
  cl $lab 1 "{\"system\": \"ours v2\", \"C\": $1, \"fetch\": $2}" $P once 256 $OUT/bs1.jsonl --extra "$LXB" \
    --cmd "$ECENV LLAMA_EC_SLOTS=$1 $fe LLAMA_EC_STATS=$OUT/srv_${lab}_L1.json $EC_SERVER $LS $OT --port {port}"
}
for C in 14 32 51; do ours_ref $C 0; ours_ref $C 1; done
ft_ref() { ( cd $FT; cl ft_$1_r$2 1 "{\"system\": \"FreeToken $1\", \"rate\": $2}" $P once 256 $OUT/bs1.jsonl --ft $1 --cache-rate $2 ); }
for r in 0.111 0.25 0.40; do ft_ref offload $r; ft_ref hybrid $r; done
cd $WORK
for f in $OUT/srv_*.log; do tail -n 60 "$f" > "$f.tail"; done; find $OUT -name 'srv_*.log' -delete
$PY - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, sys, os, statistics as st
out = sys.argv[1]
rows = [json.loads(l) for l in open(f"{out}/bs1.jsonl")] if os.path.exists(f"{out}/bs1.jsonl") else []
by = {}
for r in rows: by.setdefault(r["label"], []).append(r)
for lab, v in by.items():
    print(f"{lab:26s} {st.mean(r['decode_tok_s'] for r in v):7.2f} tok/s (n={len(v)})  vram {sorted({round(r.get('vram_used_gib') or 0, 1) for r in v})}")
PY
