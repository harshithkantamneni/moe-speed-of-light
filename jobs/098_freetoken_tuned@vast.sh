#!/bin/bash
# Job 098: FreeToken tuned per cell, as our FETCH table was (review 4a, W7c), with VRAM and each engine's CPU-only
# expert throughput recorded (W7b, W7d). Table 1's protocol (30 AIME-25 problems, 256 tokens, greedy, held-out warm-up,
# session; jobs 080-082/089), both models at the Table 1 budgets, one RTX 5090 host of the 9950X class, one launch.
# Per cell, in a seeded shuffled order: ours with the law's table (the deployed system), and FreeToken in five settings:
# hybrid with its calibrated fetch split (default), hybrid with the fetch cap fixed at 1 and 2, hybrid with 8 CPU
# threads (default: one per physical core), and offload. FreeToken's own `ft bench bw` calibration is run first, as in
# Table 1. Then each engine's CPU-only decode: FreeToken's cpu backend and stock llama.cpp with every MoE layer on the
# CPU. The comparison reported is ours against the best of FreeToken's five settings per cell (chosen on this run, which
# favours FreeToken).
# Predictions, committed before launch (host S, job 089, a 9950X host: ours / FreeToken's Table 1 setting 1.21, 1.20, 1.09
# on gpt-oss and 1.03, 1.05, 0.97 on Qwen3):
#   1. FreeToken's best setting is at most 1.10x its Table 1 setting (calibrated split, default threads, the Table 1
#      backend) at every cell;
#   2. ours leads FreeToken's best by at least 5% at gpt-oss 11% and 25%; ours / FreeToken-best is 0.95-1.15 at gpt-oss
#      40% and 0.90-1.12 at each Qwen3 cell;
#   3. a fixed fetch cap of 1 or 2 is within 8% of the calibrated split at every hybrid run;
#   4. 8 CPU threads are slower than the default, or within 3% of it, at every cell;
#   5. FreeToken's best backend is offload at gpt-oss 40% and Qwen3 43.75% and hybrid at the other four cells;
#   6. FreeToken holds 27-30 GiB of VRAM at every run; ours holds less than FreeToken at every cell;
#   7. CPU-only, stock llama.cpp decodes gpt-oss at least 2x and Qwen3 at least 1.2x FreeToken's cpu backend.
# Budget: the job stops starting steps 6 h after launch; disk about 220 GB (gpt-oss removed before the Qwen3 conversion).
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
T0=$(date +%s); BUDGET=$(( 6 * 3600 )); RESERVE=600
secs() { case $1 in *h) echo $(( ${1%h} * 3600 ));; *m) echo $(( ${1%m} * 60 ));; *s) echo ${1%s};; *) echo $1;; esac; }
left() { echo $(( T0 + BUDGET - RESERVE - $(date +%s) )); }
R() {  # timeout cmd...: bounded by the step's timeout and by the job's deadline; past the deadline the step is skipped
  local t=$(secs $1); shift; local l=$(left)
  if [ "$l" -lt 120 ]; then echo "SKIPPED (deadline): $*" | tee -a $OUT/skipped.txt; return 124; fi
  [ "$t" -gt "$l" ] && t=$l
  timeout -k 30 "$t" "$@"
}
killsrv() { pkill -f "llama-server|llama-batched-bench|freetoken" 2>/dev/null; sleep 5; }
# --- gates, before any download: the device read of job 084b (the memory clock is recorded)
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current,driver_version,power.limit --format=csv > $OUT/gpu.csv 2>&1
cat $OUT/gpu.csv
MEMCLK=$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "memory clock max ${MEMCLK} MHz (listing B reported 17001)" | tee $OUT/gate.txt
nvcc -O3 -arch=sm_$SM "$J/bw.cu" -o $WORK/bw0 && $WORK/bw0 > $OUT/bw_gate.txt 2>&1
nvidia-smi -q > $OUT/nvidia-smi-q-start.txt 2>&1
DR=$(sed -n 's/device read 1 GiB: *\([0-9]*\).*/\1/p' $OUT/bw_gate.txt)
echo "device read ${DR} GB/s" | tee -a $OUT/gate.txt
[ "${DR:-0}" -ge 1500 ] || { echo "GPU reads below 1500 GB/s: throttled card, stopping"; exit 5; }
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip >> $OUT/apt.txt 2>&1
python3 -m pip install -q --break-system-packages uv huggingface_hub hf_xet > $OUT/pip_uv.txt 2>&1
CORES=$(usable_cores); echo "usable physical cores $CORES (nproc $(nproc))" | tee $OUT/cores.txt
# --- downloads first; FreeToken, the conversion venv and the builds run meanwhile
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
FT=$WORK/ft; export FT_DIR=$FT
( cd $WORK && git init -q ft && cd ft && git remote add origin https://github.com/FlashML-org/FreeToken &&
  git fetch -q --depth 1 origin 0d652e73a452d014ac5441a15baa75348e9fcb0a && git checkout -q FETCH_HEAD ) || echo "FreeToken clone FAILED"
nvcc --version | tail -2 | tee $OUT/nvcc_version.txt   # FreeToken builds kernels with nvcc and needs the CUDA 13 toolkit (torch cu130)
( cd $FT && uv venv -q .venv && . .venv/bin/activate && time uv pip install -q -e ".[accel]" huggingface_hub hf_xet gguf sentencepiece transformers safetensors ) > $OUT/ft_install.txt 2>&1
echo "FreeToken install rc=$?"
PY=$FT/.venv/bin/python; FTBIN=$FT/.venv/bin/ft
$PY -c "import freetoken, huggingface_hub" > $OUT/ft_import.txt 2>&1 || { echo "FreeToken not importable, stopping"; cat $OUT/ft_import.txt; tail -20 $OUT/ft_install.txt; kill $DLH $DLG $DLQ 2>/dev/null; exit 3; }
$FTBIN --version > $OUT/ft_version.txt 2>&1
VE=$WORK/venv   # CPU-torch environment for the Qwen3 conversion (job 085's conversion hung in FreeToken's venv)
( uv venv -q $VE && . $VE/bin/activate && uv pip install -q torch --index-url https://download.pytorch.org/whl/cpu &&
  uv pip install -q numpy transformers sentencepiece safetensors protobuf ) > $OUT/venv_install.txt 2>&1 &
VEP=$!
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
EC_SERVER=$WORK/lc-ec/build/bin/llama-server; STOCK_SERVER=$WORK/lc-stock/build/bin/llama-server
ls -la $EC_SERVER $STOCK_SERVER
echo "builds done, waiting for the downloads ($(( $(date +%s) - T0 )) s since launch)"
wait $DLG $DLH $DLQ $VEP
cat $OUT/dl_gguf.txt $OUT/dl_hf.txt $OUT/dl_qwen3.txt; echo "venv install rc: $(tail -1 $OUT/venv_install.txt)"
F=$M/gpt-oss-120b-MXFP4.gguf
[ -f "$HFD/config.json" ] && echo "HF checkpoint ok" || echo "HF checkpoint MISSING"
[ -s "$F" ] && echo "GGUF ok $(stat -c %s $F) bytes" || echo "GGUF MISSING"
# --- the host (idle): platform, FreeToken's calibration, our probe, the law's tables
platform
( cd $FT && R 25m $FTBIN bench bw ) > $OUT/ft_bench_bw.txt 2>&1; cp ~/.cache/freetoken/benchbw/*.json $OUT/ 2>/dev/null
H=$(( CORES > 4 ? CORES - 2 : 2 ))
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
grep -q "concurrent t=" $OUT/concur.txt || { echo "concur incomplete, retrying"; R 15m $WORK/concur > $OUT/concur.txt 2>&1; }
LAWG=$(python3 $J/fetch_table.py $OUT/concur.txt $H 13253760 4 $OUT/fetch_table_law_gptoss.json 37)
LAWQ=$(python3 $J/fetch_table.py $OUT/concur.txt $H 9437184 8 $OUT/fetch_table_law_qwen3.json 48)
echo "law tables ($(date -u +%FT%TZ)): gpt-oss $LAWG (listing B 0,0,1,1,2); qwen3 $LAWQ (listing B 0,0,1,1,2,3,3,4,5)" | tee $OUT/tables.txt
[ -n "$LAWG" ] && [ -n "$LAWQ" ] || { echo "no law table"; exit 3; }
LXB='{"min_p": 0.0, "seed": 1234, "backend_sampling": true}'
OT="-ot '\\.ffn_(up|down|gate|gate_up)_exps=CUDA_Host'"
ECENV="env LLAMA_EC_KAPPA=1 LLAMA_EC_MAILBOX=1 LLAMA_EC_TDEC=1 LLAMA_EC_HELPERS=$H"
waitfree() {
  for i in $(seq 120); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-0}" -lt 1500 ] && return; sleep 2; done
  echo "GPU memory still in use: $u MiB"; killsrv
}
cl() {  # model-dir label launch meta [client args...]
  local hf=$1 lab=$2 L=$3 meta=$4; shift 4
  waitfree
  local t0=$(date +%s)
  R 60m $PY $J/bs1_client.py --model $hf --problems $(seq -s, 0 29) --decode 256 --greedy --warmup once --warmup-prompt $WORK/warmup.txt \
    --launch $L --label $lab --meta "$meta" --json $OUT/bs1.jsonl --log $OUT/srv_${lab}_L$L.log "$@"
  local rc=$?
  echo "$lab launch $L rc=$rc"; echo "$lab $L $rc $(( $(date +%s) - t0 ))" >> $OUT/runs.txt
  [ $rc = 124 ] && killsrv
}
# ---- the run list: per cell, ours (the law's table) and FreeToken in five settings, in a seeded shuffled order
ORDER_SEED=$(( $(date +%s) % 100000 )); echo "order seed $ORDER_SEED" | tee $OUT/order_seed.txt
shuf_seeded() { python3 -c "import random, sys; r = random.Random(int(sys.argv[1])); a = sys.argv[2:]; r.shuffle(a); print(' '.join(a))" "$@"; }
# FreeToken settings: hybrid with its calibrated fetch split (-1, the default) and with the cap fixed at 1 and 2; hybrid
# with 8 CPU threads (its default is one per physical core); offload (no CPU path)
FTSET="h:-1:0 h:1:0 h:2:0 h:-1:8 o:-1:0"
ft_run() {  # model hf label-prefix rate mem-ratio setting
  local mdl=$1 hf=$2 p=$3 r=$4 mr=$5 s=$6; IFS=: read b f th <<< "$s"
  local be=$([ $b = h ] && echo hybrid || echo offload) ex=""
  [ "$th" != 0 ] && ex="--moe-cpu-threads $th"
  local lab=${p}_ft_${be}_r${r}_f${f/-/m}_t${th}
  ( cd $FT; cl $hf $lab 1 "{\"model\": \"$mdl\", \"system\": \"FreeToken $be\", \"rate\": $r, \"hybrid_fetch\": $f, \"threads\": $th}" --ft $be --cache-rate $r --mem-ratio $mr --hybrid-fetch $f ${ex:+--ft-extra "$ex"} )
}
LSG="-m $F -ngl 99 -fa on -lm none -t $CORES -np 1 -c 8448 --no-webui"
oursg() { cl $HFD g_ours_C$1_law 1 "{\"model\": \"gpt-oss-120b\", \"system\": \"ours v2\", \"C\": $1, \"table\": \"$LAWG\"}" --extra "$LXB" \
    --cmd "$ECENV LLAMA_EC_POLICY=dfa LLAMA_EC_SLOTS=$1 LLAMA_EC_FETCH=$LAWG LLAMA_EC_STATS=$OUT/srv_g_ours_C$1_law.json $EC_SERVER $LSG $OT --port {port}"; }
for cell in 14:0.111 32:0.25 51:0.40; do
  IFS=: read C r <<< "$cell"
  for item in $(shuf_seeded $(( ORDER_SEED + C )) ours $FTSET); do
    if [ $item = ours ]; then oursg $C; else ft_run gpt-oss-120b $HFD g $r 0.9 $item; fi
  done
done
# CPU-only expert throughput of each engine on gpt-oss: FreeToken's cpu backend and stock llama.cpp with every MoE layer on the CPU
( cd $FT; cl $HFD g_ft_cpu 1 "{\"model\": \"gpt-oss-120b\", \"system\": \"FreeToken cpu\"}" --ft cpu --cache-rate 0.0 --mem-ratio 0.9 )
cl $HFD g_llama_n36 1 "{\"model\": \"gpt-oss-120b\", \"system\": \"llama.cpp\", \"ncmoe\": 36}" --extra "$LXB" --cmd "$STOCK_SERVER $LSG -ncmoe 36 --port {port}"
# ---- Qwen3-30B-A3B BF16 (gpt-oss files removed first for disk; the conversion runs with no server up)
rm -rf "$HFD" "$F"; df -h $WORK | tail -1
GQ=$M/Qwen3-30B-A3B-BF16.gguf
( t0=$(date +%s); PYTHONPATH=$WORK/lc-stock/gguf-py timeout 30m $VE/bin/python $WORK/lc-stock/convert_hf_to_gguf.py "$HFQ" --outtype bf16 --outfile $GQ
  echo "convert rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $GQ ) > $OUT/convert_qwen3.txt 2>&1
tail -2 $OUT/convert_qwen3.txt
LSQ="-m $GQ -ngl 99 -fa on -lm none -t $CORES -np 1 -c 2048 --no-webui --jinja"
oursq() { cl $HFQ q_ours_C$1_law 1 "{\"model\": \"Qwen3-30B-A3B\", \"system\": \"ours v2\", \"C\": $1, \"table\": \"$LAWQ\"}" --extra "$LXB" \
    --cmd "$ECENV LLAMA_EC_POLICY=dfa LLAMA_EC_SLOTS=$1 LLAMA_EC_FETCH=$LAWQ LLAMA_EC_STATS=$OUT/srv_q_ours_C$1_law.json $EC_SERVER $LSQ $OT --port {port}"; }
if [ -s "$GQ" ]; then
  for cell in 16:0.125:0.9 32:0.25:0.9 56:0.4375:0.95; do
    IFS=: read C r mr <<< "$cell"
    for item in $(shuf_seeded $(( ORDER_SEED + C )) ours $FTSET); do
      if [ $item = ours ]; then oursq $C; else ft_run Qwen3-30B-A3B $HFQ q $r $mr $item; fi
    done
  done
  ( cd $FT; cl $HFQ q_ft_cpu 1 "{\"model\": \"Qwen3-30B-A3B\", \"system\": \"FreeToken cpu\"}" --ft cpu --cache-rate 0.0 --mem-ratio 0.9 )
  cl $HFQ q_llama_n48 1 "{\"model\": \"Qwen3-30B-A3B\", \"system\": \"llama.cpp\", \"ncmoe\": 48}" --extra "$LXB" --cmd "$STOCK_SERVER $LSQ -ncmoe 48 --port {port}"
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
for lab, v in sorted(by.items()):
    print(f"{lab:36s} {st.mean(r['decode_tok_s'] for r in v):7.2f} tok/s n={len(v)} vram {sorted({round(r.get('vram_used_gib') or 0, 1) for r in v})}")
line = []
for p, C, r in (("g", 14, "0.111"), ("g", 32, "0.25"), ("g", 51, "0.40"), ("q", 16, "0.125"), ("q", 32, "0.25"), ("q", 56, "0.4375")):
    o = {x["problem"]: x["decode_tok_s"] for x in by.get(f"{p}_ours_C{C}_law", [])}
    fts = {lab: {x["problem"]: x["decode_tok_s"] for x in v} for lab, v in by.items() if lab.startswith(f"{p}_ft_") and f"_r{r}_" in lab}
    if not o or not fts:
        continue
    best = max(fts, key=lambda k: st.mean(fts[k].values()))
    ratio = st.mean(o[q] / fts[best][q] for q in o if q in fts[best])
    print(f"{p} C{C}: ours {st.mean(o.values()):.1f}; FreeToken best {best} {st.mean(fts[best].values()):.1f}; ours/best {ratio:.3f}")
    line.append(f"{p}{C} {ratio:.2f}")
open(f"{out}/oneline.txt", "w").write("098 ours/FT-best: " + " ".join(line))
PY
ONE=$(cat $OUT/oneline.txt 2>/dev/null || echo "098: no summary")
python3 $J/manifest.py "$OUT" 098_freetoken_tuned@vast "$T0" "$ONE" law_gptoss=$LAWG law_qwen3=$LAWQ memory_clock_mhz=$MEMCLK device_read_gbs=$DR
echo "SUMMARY $ONE"
