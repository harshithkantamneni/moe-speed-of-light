#!/bin/bash
# Job 081: the headline table again, with the per-host FETCH split from the law (job 080), both models on one machine
# (RTX 5090 + Ryzen 9 9950X3D). The law's tables are computed on the machine from its own probe (fetch_table.py,
# concur.txt) before any model run: gpt-oss-120b (13.25 MB experts, top-4, g = 37 us) and Qwen3-30B-A3B BF16
# (9.44 MB, top-8, g = 48 us). Protocol as jobs 076/078: 30 AIME-25 problems, 256 tokens, greedy, held-out warm-up,
# session, GPU-side sampling for llama-server; equal GPU expert memory, checked on measured VRAM.
#   gpt-oss-120b at 11 / 25 / 40%: llama.cpp -ncmoe 32 / 27 / 22, ours C 14 / 32 / 51, FreeToken rate 0.111 / 0.25 / 0.40.
#   Qwen3 at 12.5%: llama.cpp -ncmoe 42, ours C 16, FreeToken rate 0.125 (25% and 43.75% are in job 080).
# Launch 1 runs every variant (ours: current table and the law's table; FreeToken: offload and hybrid; llama.cpp once);
# each system's faster variant per budget is picked on launch 1 and rerun in launch 2 (confirmation).
# Predictions, committed before launch:
#   1. gpt-oss: the law's table beats the current table (0,1,1,2,3) at every budget (launch 1, paired CI > 1);
#   2. gpt-oss: ours leads FreeToken's better backend at 11, 25 and 40% (launch 2, paired CI > 1);
#   3. Qwen3 12.5%: the law's table beats the current table (0,1,1,2,3,3,4,5,6) (launch 1, paired CI > 1) and ours
#      leads FreeToken's better backend (launch 2, paired CI > 1);
#   4. ours >= 1.8x llama.cpp on gpt-oss at every budget and >= 2x on Qwen3 at 12.5%.
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
( cd $FT && uv venv -q .venv && . .venv/bin/activate && time uv pip install -q -e ".[accel]" huggingface_hub hf_xet gguf sentencepiece transformers safetensors ) > $OUT/ft_install.txt 2>&1
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
HFQ=$M/Qwen3-30B-A3B
( t0=$(date +%s); python3 -c "import sys; from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-30B-A3B', local_dir=sys.argv[1])" "$HFQ"
  echo "qwen3 hf download rc=$? in $(( $(date +%s) - t0 )) s"; du -sh "$HFQ" ) > $OUT/dl_qwen3.txt 2>&1 &
DLQ=$!
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
wait $DLG $DLH $DLQ
cat $OUT/dl_gguf.txt $OUT/dl_hf.txt $OUT/dl_qwen3.txt
F=$M/gpt-oss-120b-MXFP4.gguf
[ -f "$HFD/config.json" ] && echo "HF checkpoint ok" || echo "HF checkpoint MISSING"
[ -s "$F" ] && echo "GGUF ok $(stat -c %s $F) bytes" || echo "GGUF MISSING"
GQ=$M/Qwen3-30B-A3B-BF16.gguf
( t0=$(date +%s); PYTHONPATH=$WORK/lc-stock/gguf-py $PY $WORK/lc-stock/convert_hf_to_gguf.py "$HFQ" --outtype bf16 --outfile $GQ
  echo "convert rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $GQ ) > $OUT/convert_qwen3.txt 2>&1
tail -2 $OUT/convert_qwen3.txt
# --- the host
platform
( cd $FT && R 25m $FTBIN bench bw ) > $OUT/ft_bench_bw.txt 2>&1; cp ~/.cache/freetoken/benchbw/*.json $OUT/ 2>/dev/null
H=$(( CORES > 4 ? CORES - 2 : 2 ))
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
grep -q "concurrent t=" $OUT/concur.txt || { echo "concur incomplete, retrying"; R 15m $WORK/concur > $OUT/concur.txt 2>&1; }
LAWG=$(python3 $J/fetch_table.py $OUT/concur.txt $H 13253760 4 $OUT/fetch_table_law_gptoss.json 37)
LAWQ=$(python3 $J/fetch_table.py $OUT/concur.txt $H 9437184 8 $OUT/fetch_table_law_qwen3.json 48)
CURG=0,1,1,2,3; CURQ=0,1,1,2,3,3,4,5,6
echo "law tables ($(date -u +%FT%TZ)): gpt-oss $LAWG (current $CURG); qwen3 $LAWQ (current $CURQ)" | tee $OUT/tables.txt
[ -n "$LAWG" ] && [ -n "$LAWQ" ] || { echo "no law table"; exit 3; }
LXB='{"min_p": 0.0, "seed": 1234, "backend_sampling": true}'
OT="-ot '\\.ffn_(up|down|gate|gate_up)_exps=CUDA_Host'"
ECENV="env LLAMA_EC_POLICY=dfa LLAMA_EC_KAPPA=1 LLAMA_EC_MAILBOX=1 LLAMA_EC_TDEC=1 LLAMA_EC_HELPERS=$H"
waitfree() {
  for i in $(seq 120); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-0}" -lt 1500 ] && return; sleep 2; done
  echo "GPU memory still in use: $u MiB"
}
cl() {  # model-dir label launch meta [client args...]
  local hf=$1 lab=$2 L=$3 meta=$4; shift 4
  waitfree
  R 60m $PY $J/bs1_client.py --model $hf --problems $(seq -s, 0 29) --decode 256 --greedy --warmup once --warmup-prompt $WORK/warmup.txt \
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
# ---- gpt-oss-120b
LSG="-m $F -ngl 99 -fa on -lm none -t $CORES -np 1 -c 8448 --no-webui"
oursg() {  # C table-name launch
  local t=$CURG; [ $2 = law ] && t=$LAWG
  cl $HFD g_ours_C$1_$2 $3 "{\"model\": \"gpt-oss-120b\", \"system\": \"ours v2\", \"C\": $1, \"table\": \"$t\"}" --extra "$LXB" \
    --cmd "$ECENV LLAMA_EC_SLOTS=$1 LLAMA_EC_FETCH=$t LLAMA_EC_STATS=$OUT/srv_g_ours_C$1_$2_L$3.json $EC_SERVER $LSG $OT --port {port}"
}
ftg() { ( cd $FT; cl $HFD g_ft_$1_r$2 $3 "{\"model\": \"gpt-oss-120b\", \"system\": \"FreeToken $1\", \"rate\": $2}" --ft $1 --cache-rate $2 ); }
for C in 14 32 51; do oursg $C cur 1; oursg $C law 1; done
for r in 0.111 0.25 0.40; do ftg offload $r 1; ftg hybrid $r 1; done
for n in 32 27 22; do cl $HFD g_llama_n$n 1 "{\"model\": \"gpt-oss-120b\", \"system\": \"llama.cpp\", \"ncmoe\": $n}" --extra "$LXB" --cmd "$STOCK_SERVER $LSG -ncmoe $n --port {port}"; done
for pair in 14:0.111 32:0.25 51:0.40; do
  C=${pair%%:*}; r=${pair#*:}
  w=$(pick g_ours_C${C}_cur g_ours_C${C}_law); t=${w##*_}
  fb=$(pick g_ft_offload_r$r g_ft_hybrid_r$r); b=$(echo $fb | cut -d_ -f3)
  echo "gpt-oss C$C: ours $t, FreeToken $b" | tee -a $OUT/selection.txt
  oursg $C $t 2; ftg $b $r 2
done
# ---- Qwen3-30B-A3B BF16, 12.5%
LSQ="-m $GQ -ngl 99 -fa on -lm none -t $CORES -np 1 -c 2048 --no-webui --jinja"
oursq() {  # table-name launch
  local t=$CURQ; [ $1 = law ] && t=$LAWQ
  cl $HFQ q_ours_C16_$1 $2 "{\"model\": \"Qwen3-30B-A3B\", \"system\": \"ours v2\", \"C\": 16, \"table\": \"$t\"}" --extra "$LXB" \
    --cmd "$ECENV LLAMA_EC_SLOTS=16 LLAMA_EC_FETCH=$t LLAMA_EC_STATS=$OUT/srv_q_ours_C16_$1_L$2.json $EC_SERVER $LSQ $OT --port {port}"
}
ftq() { ( cd $FT; cl $HFQ q_ft_$1_r0.125 $2 "{\"model\": \"Qwen3-30B-A3B\", \"system\": \"FreeToken $1\", \"rate\": 0.125}" --ft $1 --cache-rate 0.125 --mem-ratio 0.9 ); }
if [ -s "$GQ" ]; then
  oursq cur 1; oursq law 1; ftq offload 1; ftq hybrid 1
  cl $HFQ q_llama_n42 1 '{"model": "Qwen3-30B-A3B", "system": "llama.cpp", "ncmoe": 42}' --extra "$LXB" --cmd "$STOCK_SERVER $LSQ -ncmoe 42 --port {port}"
  w=$(pick q_ours_C16_cur q_ours_C16_law); t=${w##*_}
  fb=$(pick q_ft_offload_r0.125 q_ft_hybrid_r0.125); b=$(echo $fb | cut -d_ -f3)
  echo "qwen3 C16: ours $t, FreeToken $b" | tee -a $OUT/selection.txt
  oursq $t 2; ftq $b 2
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
for r in rows: by.setdefault(r["label"], []).append(r)
for lab, v in by.items():
    per = {}
    for r in v: per.setdefault(r["launch"], []).append(r["decode_tok_s"])
    print(f"{lab:24s} " + " ".join(f"L{L}: {st.mean(x):6.2f} (n={len(x)})" for L, x in sorted(per.items())) + f"  vram {sorted({round(r.get('vram_used_gib') or 0, 1) for r in v})}")
for f in sorted(os.listdir(out)):
    if f.endswith(".json") and f.startswith("srv_"):
        s = json.load(open(f"{out}/{f}")); n = s["steps"]
        print(f"{f:34s} table {s.get('fetch_table')} misses/token {s['misses'] / n:.1f} fetches {s.get('fetches', 0) / n:.1f}")
PY
