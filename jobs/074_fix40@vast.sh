#!/bin/bash
# Job 074: does removing the per-token overhead put the cache ahead at 40% of experts? Same host type and protocol as
# job 073 (9950X3D + RTX 5090, gpt-oss-120b, 30 AIME-25 problems, 256 tokens, greedy, held-out warm-up, session).
# Fixes in the patch since 073, found in the profiles and the host-side timer (jobs 069c, 072):
#   - the cache's slot maps are allocated on the GPU: one upload per token instead of ~72 synchronous copies of
#     per-layer views before every launch (the profile's 81 host-to-device copies per token against llama.cpp's 9;
#     the timer's 0.5-0.8 ms "launch");
#   - decay powers from a table in the policy loop (post-processing, 0.25 ms per step).
# "ours v2" = the fixed build, FETCH off (FETCH gained nothing in chat in 073). Same-session references: FreeToken's
# better backend per budget from 073, and the unfixed placement (LLAMA_EC_MAPS_CPU=1) at C51.
# Predictions, committed before launch:
#   1. ec-bench, C51: host_us_per_step.launch falls from ~0.7 ms to <= 0.25 ms with the maps on the GPU;
#   2. chat, C51: ours v2 >= 127 tok/s and ahead of FreeToken offload 0.40 in the same session (paired CI > 1);
#   3. chat: ours v2 >= 59 tok/s at C14 and >= 91 tok/s at C32 (073: 57.6 / 88.2 without FETCH);
#   4. half-life 64 instead of 16 at C51: within +-2%.
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

B="policy=dfa:kappa=1:$MB"
# own text: host timing with the maps on the GPU and on the CPU (the old placement), C51
CFG="slots=32:$B"
for c in "C51:slots=51:$B" "C51mc:slots=51:$B:maps_cpu=1" "C32:slots=32:$B" "C51hl64:slots=51:$B:half_life=64"; do
  CFG="$CFG;${c#*:}:stats=$OUT/ref_${c%%:*}_r1.json"
done
ALL="--corpus $J/S_gpt-oss-120b.jsonl --seqs 0,1,2,3,4,5,6,7,8,9,10,11 -t $CORES --n-prefill 640 --n-decode 128 --no-mmap"
R 30m $EC_BIN -m $F $ALL --host-experts --ec "$CFG" > $OUT/ref_ec.jsonl 2> $OUT/ref_ec.err; echo "ref ec rc=$?"
P=$(seq -s, 0 29)
LX='{"min_p": 0.0, "seed": 1234}'
LS="-m $F -ngl 99 -fa on -lm none -t $CORES -np 1 -c 8448 --no-webui"
OT="-ot '\\.ffn_(up|down|gate|gate_up)_exps=CUDA_Host'"
ECENV="env LLAMA_EC_POLICY=dfa LLAMA_EC_KAPPA=1 LLAMA_EC_MAILBOX=1 LLAMA_EC_TDEC=1 LLAMA_EC_HELPERS=$H"
cl() {  # label launch meta problems warmup [client args...]
  local lab=$1 L=$2 meta=$3 probs=$4 wu=$5; shift 5
  R 60m $PY $J/bs1_client.py --model $HFD --problems $probs --decode 256 --greedy --warmup $wu --warmup-prompt $WORK/warmup.txt \
    --launch $L --label $lab --meta "$meta" --json $OUT/bs1.jsonl --log $OUT/srv_${lab}_L$L.log "$@"
  echo "$lab launch $L rc=$?"
}
ours() {  # label launch C extra-env
  cl $1 $2 "{\"system\": \"ours v2\", \"C\": $3}" $P once --extra "$LX" \
    --cmd "$ECENV LLAMA_EC_SLOTS=$3 $4 LLAMA_EC_STATS=$OUT/srv_$1_L$2.json $EC_SERVER $LS $OT --port {port}"
}
ft() {  # backend rate C launch
  ( cd $FT; cl ft_$1_r$2 $4 "{\"system\": \"FreeToken $1\", \"rate\": $2, \"C\": $3}" $P once --ft $1 --cache-rate $2 )
}
for L in 1 2 3; do
  ours oursv2_C14 $L 14 ""; ours oursv2_C32 $L 32 ""; ours oursv2_C51 $L 51 ""
  [ $L = 1 ] && { ft hybrid 0.111 14 1; ft hybrid 0.25 32 1; ft offload 0.40 51 1; }
  [ $L = 3 ] && ft offload 0.40 51 3
done
ours oursv2_C51_mapscpu 1 51 "LLAMA_EC_MAPS_CPU=1"
ours oursv2_C51_hl64 1 51 "LLAMA_EC_HALF_LIFE=64"
ours oursv2f_C51 1 51 "LLAMA_EC_FETCH=0,1,1,2,3"
cd $WORK
for f in $OUT/srv_*.log; do tail -n 40 "$f" > "$f.tail"; done; find $OUT -name 'srv_*.log' -delete
$PY - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, sys, os, statistics as st
out = sys.argv[1]
rows = [json.loads(l) for l in open(f"{out}/bs1.jsonl")] if os.path.exists(f"{out}/bs1.jsonl") else []
by = {}
for r in rows: by.setdefault(r["label"], []).append(r)
for lab, v in by.items():
    per = {}
    for r in v: per.setdefault(r["launch"], []).append(r["decode_tok_s"])
    print(f"{lab:26s} " + " ".join(f"L{L}: {st.mean(x):6.2f} (n={len(x)})" for L, x in sorted(per.items())) + f"  vram {sorted({round(r.get('vram_used_gib') or 0, 1) for r in v})}")
for f in sorted(os.listdir(out)):
    if f.endswith(".json") and (f.startswith("srv_ours") or f.startswith("ref_")):
        s = json.load(open(f"{out}/{f}")); h = s.get("host_us_per_step", {})
        print(f"{f:34s} hit {s.get('hit_rate', 0):.4f}  host us/step " + " ".join(f"{k}={h.get(k, 0):.0f}" for k in ("app", "pre", "inputs", "launch", "issue", "sync", "post")))
PY
