#!/bin/bash
# Job 080: a per-host FETCH split. For a layer with n missed experts, the FETCH table says how many to copy over PCIe
# into GPU slots; the rest run on the CPU helpers. The table so far (0,1,1,2,3,3,4,5,6) was fixed; job 079 found
# FETCH worth +16% on a host whose CPU path is no faster than PCIe (078) and +0.4% on one whose CPU path is faster
# (079). Here the table is chosen per host by the host-DRAM law (fetch_table.py: per-layer time
# max(f S/B_p + g, L_c + (n-f) S/B_c, n S/B_both), bandwidths from this host's concur probe, computed before any model
# run), and checked against a sweep.
#   Host: RTX 5090 + Ryzen 9 9950X3D (the job 079 offer). Qwen3-30B-A3B BF16. Protocol as jobs 078/079: 30 AIME-25
#   problems, 256 tokens, greedy, held-out warm-up, session, GPU-side sampling for llama-server.
#   43.75% of experts on the GPU (C56), launch 1: tables none / current / law / law with f(1)=1 / lite
#     (0,0,1,1,1,2,2,3,3) / all (0,1,2,...,8), and FreeToken offload and hybrid. The best table and the better
#     FreeToken backend are picked on launch 1 and rerun in launch 2 (with the current and law tables).
#   25% (C32), launch 1: none / current / law, FreeToken offload and hybrid.
# Expected law table here, from job 079's FreeToken probe of the same offer (CPU 64.9, PCIe 52.6, both 79 GB/s):
#   0,0,1,1,2,3,3,4,5, i.e. one missed expert runs on the CPU instead of being fetched. Modelled (per-layer, no cache
#   effects) that saves ~1 ms per token against the current table; the cache effect of fetching (a fetched expert is
#   admitted at once) works the other way, hence the sweep.
# Predictions, committed before launch:
#   1. at 43.75%, the law table beats the current table by >= 3% (launch 1, paired, CI lower bound > 1);
#   2. at 43.75%, the law table is within 2% of the best table in the sweep (launch 1);
#   3. at 43.75%, with the best table, ours leads FreeToken's better backend (launch 2, paired CI > 1);
#   4. at 25%, the law table beats the current table (launch 1, paired CI > 1).
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip python3-dev numactl >> $OUT/apt.txt 2>&1
CORES=$(usable_cores); echo "usable physical cores $CORES (nproc $(nproc))" | tee $OUT/cores.txt
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
GG=$M/Qwen3-30B-A3B-BF16.gguf
( t0=$(date +%s); PYTHONPATH=$WORK/lc-stock/gguf-py $PY $WORK/lc-stock/convert_hf_to_gguf.py "$HFD" --outtype bf16 --outfile $GG
  echo "convert rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $GG ) > $OUT/convert.txt 2>&1
tail -3 $OUT/convert.txt
[ -s $GG ] || { echo "GGUF conversion failed"; exit 3; }
platform
( cd $FT && R 20m $FTBIN bench bw --dtype bf16 ) > $OUT/ft_bench_bw.txt 2>&1; cp ~/.cache/freetoken/benchbw/*.json $OUT/ 2>/dev/null
H=$(( CORES > 4 ? CORES - 2 : 2 ))
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
grep -q "concurrent t=" $OUT/concur.txt || { echo "concur incomplete, retrying"; R 15m $WORK/concur > $OUT/concur.txt 2>&1; }
# the law's table for this host, from its own probe, before any model run
LAW=$(python3 $J/fetch_table.py $OUT/concur.txt $H 9437184 8 $OUT/fetch_table_law.json)
echo "law table $LAW ($(date -u +%FT%TZ))"; cat $OUT/fetch_table_law.json
[ -n "$LAW" ] || { echo "no law table"; exit 3; }
LAW1=$(echo $LAW | awk -F, 'BEGIN{OFS=","} {$2=1; print}')
CUR=0,1,1,2,3,3,4,5,6; LITE=0,0,1,1,1,2,2,3,3; ALL=0,1,2,3,4,5,6,7,8
echo "tables: current $CUR law $LAW law_f1 $LAW1 lite $LITE all $ALL" | tee $OUT/tables.txt
LXB='{"min_p": 0.0, "seed": 1234, "backend_sampling": true}'
LS="-m $GG -ngl 99 -fa on -lm none -t $CORES -np 1 -c 2048 --no-webui --jinja"
OT="-ot '\\.ffn_(up|down|gate|gate_up)_exps=CUDA_Host'"
ECENV="env LLAMA_EC_POLICY=dfa LLAMA_EC_KAPPA=1 LLAMA_EC_MAILBOX=1 LLAMA_EC_TDEC=1 LLAMA_EC_HELPERS=$H"
waitfree() {
  for i in $(seq 120); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-0}" -lt 1500 ] && return; sleep 2; done
  echo "GPU memory still in use: $u MiB"
}
cl() {  # label launch meta [client args...]
  local lab=$1 L=$2 meta=$3; shift 3
  waitfree
  R 60m $PY $J/bs1_client.py --model $HFD --problems $(seq -s, 0 29) --decode 256 --greedy --warmup once --warmup-prompt $WORK/warmup.txt \
    --launch $L --label $lab --meta "$meta" --json $OUT/bs1.jsonl --log $OUT/srv_${lab}_L$L.log "$@"
  echo "$lab launch $L rc=$?"
}
ours() {  # label C table|none launch
  local lab=$1 C=$2 t=$3 L=$4 f=""
  [ "$t" != none ] && f="LLAMA_EC_FETCH=$t"
  cl $lab $L "{\"system\": \"ours v2\", \"C\": $C, \"table\": \"$t\"}" --extra "$LXB" \
    --cmd "$ECENV LLAMA_EC_SLOTS=$C $f LLAMA_EC_STATS=$OUT/srv_${lab}_L$L.json $EC_SERVER $LS $OT --port {port}"
}
ft() {  # backend rate launch mem-ratio
  ( cd $FT; cl ft_$1_r$2 $3 "{\"system\": \"FreeToken $1\", \"rate\": $2}" --ft $1 --cache-rate $2 --mem-ratio $4 )
}
# 43.75%, launch 1: the sweep
declare -A T=([none]=none [cur]=$CUR [law]=$LAW [law_f1]=$LAW1 [lite]=$LITE [all]=$ALL)
for k in none cur law law_f1 lite all; do ours ours_C56_$k 56 "${T[$k]}" 1; done
ft offload 0.4375 1 0.95; ft hybrid 0.4375 1 0.95
# pick on launch 1
read BEST FTB <<< $($PY - $OUT/bs1.jsonl <<'PY'
import json, sys, statistics as st
by = {}
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r["launch"] == 1: by.setdefault(r["label"], []).append(r["decode_tok_s"])
m = {k: st.mean(v) for k, v in by.items()}
o = max((k for k in m if k.startswith("ours_C56_")), key=m.get)
f = max((k for k in m if k.startswith("ft_") and k.endswith("0.4375")), key=m.get)
print(o.replace("ours_C56_", ""), f.split("_")[1])
PY
)
echo "launch 1 picks: table $BEST, FreeToken $FTB" | tee $OUT/selection.txt
# 43.75%, launch 2: confirmation
for k in $(echo "$BEST cur law" | tr ' ' '\n' | awk '!s[$0]++'); do ours ours_C56_$k 56 "${T[$k]}" 2; done
ft $FTB 0.4375 2 0.95
# 25%, launch 1
for k in none cur law; do ours ours_C32_$k 32 "${T[$k]}" 1; done
ft offload 0.25 1 0.9; ft hybrid 0.25 1 0.9
cd $WORK
for f in $OUT/srv_*.log; do tail -n 60 "$f" > "$f.tail"; done; find $OUT -name 'srv_*.log' -delete
$PY - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, sys, os, statistics as st
out = sys.argv[1]
rows = [json.loads(l) for l in open(f"{out}/bs1.jsonl")] if os.path.exists(f"{out}/bs1.jsonl") else []
by = {}
for r in rows: by.setdefault((r["label"], r["launch"]), []).append(r)
for (lab, L), v in by.items():
    print(f"{lab:24s} L{L} {st.mean(r['decode_tok_s'] for r in v):7.2f} tok/s (n={len(v)})  vram {sorted({round(r.get('vram_used_gib') or 0, 1) for r in v})}")
for f in sorted(os.listdir(out)):
    if f.startswith("srv_ours") and f.endswith(".json"):
        d = json.load(open(f"{out}/{f}")); n = d["steps"]
        print(f"{f}: table {d.get('fetch_table')} misses/token {d['misses'] / n:.1f} fetches {d.get('fetches', 0) / n:.1f} admits {d['admits'] / n:.1f}")
PY
