#!/bin/bash
# Job 086: FreeToken's own headline model. FreeToken's paper (arXiv 2608.16157) leads with Qwen3.6-35B-A3B in BF16 on an
# RTX 5090 (77-83 tok/s). Here, on an RTX 5090 + Ryzen 9 9950X3D listing (the headline machine's type), Table 1
# protocol (30 AIME-25 problems, 256 tokens, greedy, held-out warm-up, session, GPU-side sampling for llama-server):
#   Qwen/Qwen3.6-35B-A3B (40 layers, 256 experts, top-8, 6.29 MB BF16 experts; hybrid linear/full attention), text only
#   on every system (llama.cpp GGUF converted without the vision tower or the MTP head; FreeToken --text-model-only).
#   Budgets 12.5 / 25 / 37.5% of experts on the GPU: ours C 32 / 64 / 96 = llama.cpp -ncmoe 35 / 30 / 25 = FreeToken
#   --moe-cache-rate 0.125 / 0.25 / 0.375 (43.75% does not fit a 32 GB card with this model's dense weights).
#   Launch 1: ours with the law's table (computed on the machine before any model run; g = 32 us, Qwen3's 48 us scaled by
#   expert bytes, frozen here) and with the fixed table 0,1,1,2,3,3,4,5,6; FreeToken offload and hybrid; llama.cpp.
#   Plus FreeToken as shipped (--moe-strategy auto, automatic cache size) to check our FreeToken setup against the paper.
#   Launch 2: each system's faster variant per budget (the comparison).
# Predictions, committed before launch:
#   1. FreeToken as shipped runs within 15% of its paper's 77-83 tok/s (65-96 tok/s);
#   2. ours leads FreeToken's better backend at 12.5, 25 and 37.5% (launch 2, paired CI > 1);
#   3. ours runs >= 2x llama.cpp at every budget;
#   4. the law's table beats the fixed table at every budget (launch 1, paired CI > 1).
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
nvcc --version | tail -2 | tee $OUT/nvcc_version.txt
( cd $FT && uv venv -q .venv && . .venv/bin/activate && time uv pip install -q -e ".[accel]" huggingface_hub hf_xet gguf sentencepiece transformers safetensors ) > $OUT/ft_install.txt 2>&1
echo "FreeToken install rc=$?"
PY=$FT/.venv/bin/python; FTBIN=$FT/.venv/bin/ft
$PY -c "import freetoken, huggingface_hub" > $OUT/ft_import.txt 2>&1 || { echo "FreeToken not importable, stopping"; cat $OUT/ft_import.txt; tail -20 $OUT/ft_install.txt; exit 3; }
$FTBIN --version > $OUT/ft_version.txt 2>&1
mkdir -p $M
HF6=$M/Qwen3.6-35B-A3B
( t0=$(date +%s); python3 -c "import sys; from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3.6-35B-A3B', local_dir=sys.argv[1])" "$HF6"
  echo "qwen3.6 hf download rc=$? in $(( $(date +%s) - t0 )) s"; du -sh "$HF6" ) > $OUT/dl_qwen36.txt 2>&1 &
DL6=$!
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
wait $DL6
cat $OUT/dl_qwen36.txt
G6=$M/Qwen3.6-35B-A3B-BF16.gguf
( t0=$(date +%s); PYTHONPATH=$WORK/lc-stock/gguf-py $PY $WORK/lc-stock/convert_hf_to_gguf.py "$HF6" --outtype bf16 --no-mtp --outfile $G6
  echo "convert rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $G6 ) > $OUT/convert_qwen36.txt 2>&1
tail -2 $OUT/convert_qwen36.txt
[ -s "$G6" ] || { echo "Qwen3.6 GGUF missing, stopping"; tail -30 $OUT/convert_qwen36.txt; exit 3; }
# --- the host
platform
( cd $FT && R 25m $FTBIN bench bw ) > $OUT/ft_bench_bw.txt 2>&1; cp ~/.cache/freetoken/benchbw/*.json $OUT/ 2>/dev/null
H=$(( CORES > 4 ? CORES - 2 : 2 ))
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
grep -q "concurrent t=" $OUT/concur.txt || { echo "concur incomplete, retrying"; R 15m $WORK/concur > $OUT/concur.txt 2>&1; }
LAW=$(python3 $J/fetch_table.py $OUT/concur.txt $H 6291456 8 $OUT/fetch_table_law_qwen36.json 32)
CUR=0,1,1,2,3,3,4,5,6
echo "law table ($(date -u +%FT%TZ)): qwen3.6 $LAW (fixed $CUR)" | tee $OUT/tables.txt
[ -n "$LAW" ] || { echo "no law table"; exit 3; }
LXB='{"min_p": 0.0, "seed": 1234, "backend_sampling": true}'
OT="-ot '\\.ffn_(up|down|gate|gate_up)_exps=CUDA_Host'"
ECENV="env LLAMA_EC_POLICY=dfa LLAMA_EC_KAPPA=1 LLAMA_EC_MAILBOX=1 LLAMA_EC_TDEC=1 LLAMA_EC_HELPERS=$H"
waitfree() {
  for i in $(seq 120); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-0}" -lt 1500 ] && return; sleep 2; done
  echo "GPU memory still in use: $u MiB"
}
cl() {  # label launch meta [client args...]
  local lab=$1 L=$2 meta=$3; shift 3
  waitfree
  R 60m $PY $J/bs1_client.py --model $HF6 --problems $(seq -s, 0 29) --decode 256 --greedy --warmup once --warmup-prompt $WORK/warmup.txt \
    --launch $L --label $lab --meta "$meta" --json $OUT/bs1.jsonl --log $OUT/srv_${lab}_L$L.log "$@"
  echo "$lab launch $L rc=$?"
}
pick() {  # launch-1 winner among labels
  $PY - $OUT/bs1.jsonl "$@" <<'PY'
import json, sys, statistics as st
by = {}
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r["launch"] == 1: by.setdefault(r["label"], []).append(r["decode_tok_s"])
print(max(sys.argv[2:], key=lambda k: st.mean(by.get(k, [0]))))
PY
}
LS6="-m $G6 -ngl 99 -fa on -lm none -t $CORES -np 1 -c 2048 --no-webui --jinja"
ours() {  # C table-name launch
  local t=$CUR; [ $2 = law ] && t=$LAW
  cl ours_C$1_$2 $3 "{\"model\": \"Qwen3.6-35B-A3B\", \"system\": \"ours v2\", \"C\": $1, \"table\": \"$t\"}" --extra "$LXB" \
    --cmd "$ECENV LLAMA_EC_SLOTS=$1 LLAMA_EC_FETCH=$t LLAMA_EC_STATS=$OUT/srv_ours_C$1_$2_L$3.json $EC_SERVER $LS6 $OT --port {port}"
}
ft() {  # backend rate launch mem-ratio
  ( cd $FT; cl ft_$1_r$2 $3 "{\"model\": \"Qwen3.6-35B-A3B\", \"system\": \"FreeToken $1\", \"rate\": $2}" --ft $1 --cache-rate $2 --mem-ratio $4 --ft-extra=--text-model-only )
}
# FreeToken as shipped: automatic backend and cache size, their benchmark's memory ratio
( cd $FT; cl ft_auto 1 '{"model": "Qwen3.6-35B-A3B", "system": "FreeToken auto (as shipped)"}' --ft auto --ft-extra=--text-model-only )
for C in 32 64 96; do ours $C law 1; ours $C cur 1; done
for rm_ in 0.125:0.9 0.25:0.9 0.375:0.95; do r=${rm_%%:*}; mr=${rm_#*:}; ft offload $r 1 $mr; ft hybrid $r 1 $mr; done
for n in 35 30 25; do cl llama_n$n 1 "{\"model\": \"Qwen3.6-35B-A3B\", \"system\": \"llama.cpp\", \"ncmoe\": $n}" --extra "$LXB" --cmd "$STOCK_SERVER $LS6 -ncmoe $n --port {port}"; done
for trip in 32:0.125:0.9 64:0.25:0.9 96:0.375:0.95; do
  C=${trip%%:*}; rest=${trip#*:}; r=${rest%%:*}; mr=${rest#*:}
  w=$(pick ours_C${C}_cur ours_C${C}_law); t=${w##*_}
  fb=$(pick ft_offload_r$r ft_hybrid_r$r); b=$(echo $fb | cut -d_ -f2)
  echo "qwen3.6 C$C: ours $t, FreeToken $b" | tee -a $OUT/selection.txt
  ours $C $t 2; ft $b $r 2 $mr
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
    per = {}
    for r in v: per.setdefault(r["launch"], []).append(r["decode_tok_s"])
    print(f"{lab:24s} " + " ".join(f"L{L}: {st.mean(x):6.2f} (n={len(x)})" for L, x in sorted(per.items())) + f"  vram {sorted({round(r.get('vram_used_gib') or 0, 1) for r in v})}")
for f in sorted(os.listdir(out)):
    if f.endswith(".json") and f.startswith("srv_"):
        s = json.load(open(f"{out}/{f}")); n = s["steps"]
        print(f"{f:34s} table {s.get('fetch_table')} hit {s['hit_rate']:.3f} misses/token {s['misses'] / n:.1f} fetches {s.get('fetches', 0) / n:.1f}")
PY
