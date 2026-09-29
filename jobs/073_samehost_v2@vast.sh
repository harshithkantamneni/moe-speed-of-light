#!/bin/bash
# Job 073: the same-machine comparison, redone to the reviews' standard. gpt-oss-120b on one RTX 5090 host; llama.cpp
# (stock llama-server, -ncmoe), the expert cache (+FETCH) and FreeToken (offload and hybrid), each at equal GPU
# memory for experts: -ncmoe 32 / 27 / 22 = cache 14 / 32 / 51 slots per layer = FreeToken --moe-cache-rate
# 0.111 / 0.25 / 0.40 (0.444 does not fit FreeToken's KV cache at memory ratio 0.9, jobs 064/067).
# Changes from jobs 064/067 (one request per prompt, 5 prompts, sampled, same-problem warm-up):
#   - all 30 AIME-25 problems, 256 decode tokens each (FreeToken's decode length and tok/s formula, their client code);
#   - greedy decoding, so every system decodes (nearly) the same text; the outputs' hashes are compared;
#   - one warm-up request on a held-out prompt per server launch, then the 30 problems in order with the cache
#     carried across them (a session), instead of a warm-up on the measured problem itself;
#   - 3 server launches per main configuration, group order rotated between launches;
#   - GPU memory measured (nvidia-smi) at the end of every launch;
#   - FreeToken tuned as its docs direct: `ft bench bw` profile first (hybrid's auto fetch fraction uses it), both
#     backends, and a --moe-hybrid-max-fetch sweep {0, 1, 2} at rate 0.25;
#   - a check of the old protocol: 5 problems with same-problem warm-up for one configuration per system at C32.
# Statistics (paired bootstrap over problems) are computed off the machine: scripts/samehost_stats.py.
# Predictions, committed before launch (from jobs 064/067: +18-21% / +9-11% / +2-3% over FreeToken's best):
#   1. ours+FETCH is ahead of FreeToken's better backend at C14 and C32 by >= 5%, the 95% paired-bootstrap CI
#      excluding 0; at C51 the difference is within +-5%.
#   2. ours+FETCH is >= 1.8x llama.cpp at every budget.
#   3. The old protocol (warm-up on the measured problem) reads 5-15% faster than the session protocol for the two
#      caching systems and within +-2% for llama.cpp (its placement is static).
#   4. Greedy outputs: >= 80% of problems give the same text as llama.cpp's for every system (numerics differ).
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
# --- the host, measured on a quiet machine; the law's prediction for the own-text runs, before them
platform
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
( cd $FT && R 20m $FTBIN bench bw ) > $OUT/ft_bench_bw.txt 2>&1; cp ~/.cache/freetoken/benchbw/*.json $OUT/ 2>/dev/null
H=$(( CORES > 4 ? CORES - 2 : 2 )); MB="mailbox=1:tdec=1:helpers=$H"
python3 $J/law_predict.py $OUT/concur.txt $H "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" > $OUT/law_prediction.json 2>&1
echo "law prediction written $(date -u +%FT%TZ)"
# own text (as jobs 066-072), for the host lottery
B="policy=dfa:kappa=1:$MB"
CFG="slots=32:$B"
for c in "C14:slots=14:$B" "C32:slots=32:$B" "C56:slots=56:$B" "C32f:slots=32:$B:fetch=0,1,1,2,3"; do
  CFG="$CFG;${c#*:}:stats=$OUT/ref_${c%%:*}_r1.json"
done
ALL="--corpus $J/S_gpt-oss-120b.jsonl --seqs 0,1,2,3,4,5,6,7,8,9,10,11 -t $CORES --n-prefill 640 --n-decode 128 --no-mmap"
R 30m $EC_BIN -m $F $ALL --host-experts --ec "$CFG" > $OUT/ref_ec.jsonl 2> $OUT/ref_ec.err; echo "ref ec rc=$?"
# --- the comparison: every system as a server, one client
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
llama_group() {  # launch
  for pair in 32:14 27:32 22:51; do n=${pair%:*}; C=${pair#*:}
    cl llama_n$n $1 "{\"system\": \"llama.cpp\", \"ncmoe\": $n, \"C\": $C}" $P once --extra "$LX" --cmd "$STOCK_SERVER $LS -ncmoe $n --port {port}"
  done
}
ours_group() {  # launch [base]
  for C in 14 32 51; do
    cl oursf_C$C $1 "{\"system\": \"ours+fetch\", \"C\": $C}" $P once --extra "$LX" \
      --cmd "$ECENV LLAMA_EC_SLOTS=$C LLAMA_EC_FETCH=0,1,1,2,3 LLAMA_EC_STATS=$OUT/srv_oursf_C${C}_L$1.json $EC_SERVER $LS $OT --port {port}"
    [ "$2" = base ] && cl ours_C$C $1 "{\"system\": \"ours\", \"C\": $C}" $P once --extra "$LX" \
      --cmd "$ECENV LLAMA_EC_SLOTS=$C LLAMA_EC_STATS=$OUT/srv_ours_C${C}_L$1.json $EC_SERVER $LS $OT --port {port}"
  done
}
ft_group() {  # launch
  ( cd $FT
    for pair in 0.111:14 0.25:32 0.40:51; do r=${pair%:*}; C=${pair#*:}
      for b in offload hybrid; do
        cl ft_${b}_r$r $1 "{\"system\": \"FreeToken $b\", \"rate\": $r, \"C\": $C}" $P once --ft $b --cache-rate $r
      done
    done )
}
for L in 1 2 3; do
  case $L in
    1) llama_group 1; ours_group 1 base; ft_group 1 ;;
    2) ours_group 2; ft_group 2; llama_group 2 ;;
    3) ft_group 3; llama_group 3; ours_group 3 ;;
  esac
done
# FreeToken hybrid with fixed fetch caps (one launch each)
( cd $FT; for mf in 0 1 2; do
  cl ft_hybrid_r0.25_mf$mf 1 "{\"system\": \"FreeToken hybrid\", \"rate\": 0.25, \"C\": 32, \"max_fetch\": $mf}" $P once --ft hybrid --cache-rate 0.25 --hybrid-fetch $mf
done )
# the old protocol (warm-up on the measured problem), 5 problems, one configuration per system at C32
cl old_llama_n27 1 '{"system": "llama.cpp", "ncmoe": 27, "C": 32, "protocol": "same-problem warm-up"}' 0,1,2,3,4 same --extra "$LX" --cmd "$STOCK_SERVER $LS -ncmoe 27 --port {port}"
cl old_oursf_C32 1 '{"system": "ours+fetch", "C": 32, "protocol": "same-problem warm-up"}' 0,1,2,3,4 same --extra "$LX" \
  --cmd "$ECENV LLAMA_EC_SLOTS=32 LLAMA_EC_FETCH=0,1,1,2,3 $EC_SERVER $LS $OT --port {port}"
( cd $FT; cl old_ft_offload_r0.25 1 '{"system": "FreeToken offload", "rate": 0.25, "C": 32, "protocol": "same-problem warm-up"}' 0,1,2,3,4 same --ft offload --cache-rate 0.25 )
cd $WORK
for f in $OUT/srv_*.log; do tail -n 40 "$f" > "$f.tail"; done; find $OUT -name 'srv_*.log' -delete
$PY - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, sys, os, statistics as st
out = sys.argv[1]
rows = [json.loads(l) for l in open(f"{out}/bs1.jsonl")] if os.path.exists(f"{out}/bs1.jsonl") else []
by = {}
for r in rows: by.setdefault(r["label"], []).append(r)
ref = {r["problem"]: r["output_sha1_full"] for r in rows if r["label"] == "llama_n27" and r["launch"] == 1}
print("greedy, 256 tokens, AIME-25, session protocol; tok/s = mean over problems, per launch")
for lab, v in by.items():
    per = {}
    for r in v: per.setdefault(r["launch"], []).append(r["decode_tok_s"])
    same = sum(1 for r in v if ref.get(r["problem"]) == r["output_sha1_full"]) / max(1, len(v))
    vr = sorted({round(r.get("vram_used_gib") or 0, 1) for r in v})
    print(f"{lab:26s} " + " ".join(f"L{L}: {st.mean(x):6.2f} (n={len(x)})" for L, x in sorted(per.items())) + f"  vram {vr}  same text as llama_n27 {same:.0%}")
PY
