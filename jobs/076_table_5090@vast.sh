#!/bin/bash
# Job 076: the clean comparison table with the fixed build ("ours v2": slot maps on the GPU, tabled decay; GPU-side
# sampling for every llama-server system), 11 / 25 / 40% of experts on one RTX 5090 + Ryzen 9 9950X host.
# gpt-oss-120b, 30 AIME-25 problems, 256 tokens, greedy, held-out warm-up, session protocol (jobs 073-075).
# Equal expert memory: llama.cpp -ncmoe 32 / 27 / 22 = cache 14 / 32 / 51 slots per layer = FreeToken --moe-cache-rate
# 0.111 / 0.25 / 0.40. Variant protocol as job 075: launch 1 runs every variant (ours with and without FETCH, FreeToken
# offload and hybrid), launches 2-3 only each system's faster one per budget (confirmation). llama.cpp: one launch.
# Predictions, committed before launch:
#   1. ours ahead of FreeToken at 11%, 25% and 40%, paired 95% CI above 1 on the confirmation launches;
#   2. ours >= 1.8x llama.cpp at every budget;
#   3. the law, predicted on the machine, within 8% for the cache's own-text runs (base and FETCH, C14/C32/C56).
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
CFG="slots=32:$B"
for c in "C14:slots=14:$B" "C32:slots=32:$B" "C56:slots=56:$B" "C32f:slots=32:$B:fetch=0,1,1,2,3"; do CFG="$CFG;${c#*:}:stats=$OUT/ref_${c%%:*}_r1.json"; done
ALL="--corpus $J/S_gpt-oss-120b.jsonl --seqs 0,1,2,3,4,5,6,7,8,9,10,11 -t $CORES --n-prefill 640 --n-decode 128 --no-mmap"
R 30m $EC_BIN -m $F $ALL --host-experts --ec "$CFG" > $OUT/ref_ec.jsonl 2> $OUT/ref_ec.err; echo "ref ec rc=$?"

P=$(seq -s, 0 29)
LXB='{"min_p": 0.0, "seed": 1234, "backend_sampling": true}'
LS="-m $F -ngl 99 -fa on -lm none -t $CORES -np 1 -c 8448 --no-webui"
OT="-ot '\\.ffn_(up|down|gate|gate_up)_exps=CUDA_Host'"
ECENV="env LLAMA_EC_POLICY=dfa LLAMA_EC_KAPPA=1 LLAMA_EC_MAILBOX=1 LLAMA_EC_TDEC=1 LLAMA_EC_HELPERS=$H"
waitfree() {  # the previous server's GPU memory must be released first (074b: llama.cpp OOM after FreeToken)
  for i in $(seq 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-0}" -lt 1500 ] && return; sleep 2; done
  echo "GPU memory still in use: $u MiB"
}
cl() {  # label launch meta problems warmup [client args...]
  local lab=$1 L=$2 meta=$3 probs=$4 wu=$5; shift 5
  waitfree
  R 60m $PY $J/bs1_client.py --model $HFD --problems $probs --decode 256 --greedy --warmup $wu --warmup-prompt $WORK/warmup.txt \
    --launch $L --label $lab --meta "$meta" --json $OUT/bs1.jsonl --log $OUT/srv_${lab}_L$L.log "$@"
  echo "$lab launch $L rc=$?"
}
llama() { cl llama_n$1 $2 "{\"system\": \"llama.cpp\", \"ncmoe\": $1}" $P once --extra "$LXB" --cmd "$STOCK_SERVER $LS -ncmoe $1 --port {port}"; }
ours() {  # C launch fetch(0|1)
  local lab=oursv2_C$1; local fe=""
  [ "$3" = 1 ] && { lab=oursv2f_C$1; fe="LLAMA_EC_FETCH=0,1,1,2,3"; }
  cl $lab $2 "{\"system\": \"ours v2\", \"C\": $1, \"fetch\": $3}" $P once --extra "$LXB" \
    --cmd "$ECENV LLAMA_EC_SLOTS=$1 $fe LLAMA_EC_STATS=$OUT/srv_${lab}_L$2.json $EC_SERVER $LS $OT --port {port}"
}
ft() { ( cd $FT; cl ft_$1_r$2 $3 "{\"system\": \"FreeToken $1\", \"rate\": $2}" $P once --ft $1 --cache-rate $2 ); }
# launch 1: every variant
for C in 14 32 51; do ours $C 1 0; ours $C 1 1; done
for r in 0.111 0.25 0.40; do ft offload $r 1; ft hybrid $r 1; done
llama 32 1; llama 27 1; llama 22 1
# pick each system's faster variant per budget from launch 1
eval "$($PY - "$OUT/bs1.jsonl" <<'PY'
import json, sys, statistics as st
by = {}
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r["launch"] == 1: by.setdefault(r["label"], []).append(r["decode_tok_s"])
m = {k: st.mean(v) for k, v in by.items()}
for C, rate in ((14, "0.111"), (32, "0.25"), (51, "0.40")):
    f = 1 if m.get(f"oursv2f_C{C}", 0) > m.get(f"oursv2_C{C}", 0) else 0
    b = "offload" if m.get(f"ft_offload_r{rate}", 0) >= m.get(f"ft_hybrid_r{rate}", 0) else "hybrid"
    print(f"OF{C}={f}; FB{C}={b}")
PY
)"
echo "selected: ours fetch C14=$OF14 C32=$OF32 C51=$OF51; FreeToken C14=$FB14 C32=$FB32 C51=$FB51" | tee $OUT/selection.txt
for L in 2 3; do
  if [ $L = 2 ]; then ours 14 2 $OF14; ours 32 2 $OF32; ours 51 2 $OF51; ft $FB14 0.111 2; ft $FB32 0.25 2; ft $FB51 0.40 2
  else ft $FB14 0.111 3; ft $FB32 0.25 3; ft $FB51 0.40 3; ours 14 3 $OF14; ours 32 3 $OF32; ours 51 3 $OF51; fi
done
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
    if f.endswith(".json") and f.startswith("srv_"):
        s = json.load(open(f"{out}/{f}")); h = s.get("host_us_per_step", {})
        print(f"{f:34s} hit {s.get('hit_rate', 0):.4f}  host us/step " + " ".join(f"{k}={h.get(k, 0):.0f}" for k in ("app", "launch", "sync", "post")))
PY
