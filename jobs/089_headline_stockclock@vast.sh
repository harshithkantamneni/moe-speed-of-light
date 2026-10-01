#!/bin/bash
# Job 089: Table 1 on a stock-clock card, with the LRU attribution and the prefill / batch timings on the same rental.
# Listing B's card (jobs 080-082, Table 1) reported a 17001 MHz memory clock against 14001 MHz on every other RTX 5090 we
# rented. Target: Vast offer 51871552 (RTX 5090, Ryzen 9 9950X, 126 GB RAM, 285 Mb/s download: both models, 126 GB,
# take about an hour, so the downloads start first and the builds run meanwhile).
# Gates before any download: clocks.max.memory in gpu.csv must be 14001 MHz ("WRONG MEMORY CLOCK <value>", exit 4
# otherwise, so the rental can be destroyed within minutes), and the device-read gate of job 084b (bw.cu >= 1,500 GB/s,
# exit 5). Table 1 protocol (30 AIME-25 problems, 256 tokens, greedy, held-out warm-up, session, GPU-side sampling for
# llama-server), both models at the Table 1 budgets: gpt-oss-120b C 14 / 32 / 51 (FreeToken rates 0.111 / 0.25 / 0.40),
# Qwen3-30B-A3B BF16 C 16 / 32 / 56 (0.125 / 0.25 / 0.4375; FreeToken --memory-ratio 0.9 / 0.9 / 0.95 as jobs 080-082).
# The law's tables are computed on the machine from its own probe before any model run (fetch_table.py, concur.txt).
# FreeToken's backend per cell is a pre-registered carry-over of job 081's selection, not picked here: hybrid at gpt-oss
# 11 and 25% and at Qwen3 12.5 and 25%; offload at gpt-oss 40% and Qwen3 43.75%. Stock llama.cpp runs at matched expert
# count: -ncmoe 32 / 27 / 22 (gpt-oss, job 081) and 42 / 36 / 27 (Qwen3: 42 from job 081, 36 and 27 from job 082).
# Run list, per model:
#   launch 1, per cell in this order: ours with the law's table, FreeToken (carried-over backend), stock llama.cpp;
#   launch 2, per cell in the reversed order: FreeToken, then ours (the confirmation and the order check);
#   the attribution: ours with LLAMA_EC_POLICY=lru in place of dfa (the policy of job 087's "LRU replacement" step) and
#     everything else as ours, the law's FETCH table included (FETCH exists only in mailbox mode, so the step's host-driven
#     path cannot carry it), at all three cells;
#   prefill and batch at 25% (gpt-oss C32 vs -ncmoe 27; Qwen3 C32 vs -ncmoe 36): llama-batched-bench (stock tool, built
#     in both trees) at -npp 512,2048 -ntg 128 -npl 1,2,4, -c 8960 (the KV is split per sequence: 2,240 each >= 2,048 +
#     128); ours with its server's -ot placement and LLAMA_EC_* environment (TDEC=1 included, so its multi-token steps
#     run the CPU side on one thread), stock with --n-cpu-moe; pp and tg throughputs recorded (bb_*.jsonl).
# gpt-oss first, then its files are removed and Qwen3 is converted to BF16 GGUF (in a CPU-torch venv, as jobs 085b/087).
# Predictions, committed before launch:
#   1. the card reports a 14001 MHz memory clock and passes the device-read gate (otherwise the job is void);
#   2. ours / FreeToken (launch 2) is within +-0.06 of Table 1's ratio at every cell (1.294, 1.275, 1.154 on gpt-oss;
#      1.032, 1.153, 1.049 on Qwen3);
#   3. ours' absolute speed is 5-20% below listing B's (69.9 / 109.2 / 152.5; 40.0 / 63.1 / 108.2 tok/s) at gpt-oss 40%
#      and Qwen3 43.75%, and within +-12% of it at the other four cells;
#   4. the ours-first (launch 1) and FreeToken-first (launch 2) orders give ratios within 3% of each other at every cell;
#   5. ours with LRU and the law's table is 10-30% slower than ours with decayed frequency at every cell, trails FreeToken
#      at Qwen3 12.5%, and still leads it at gpt-oss 11 and 25%;
#   6. prefill at 2,048 tokens: ours within +-20% of stock llama.cpp; at 2 and 4 parallel sequences ours' decode
#      throughput is below stock's on both models (the cache serves single-token steps only).
# Budget: the job stops starting steps 5 h after launch (every step bounded by its own timeout and by that deadline;
# skipped steps in skipped.txt); disk about 220 GB as job 085 (gpt-oss 120 GB removed before the 57 + 61 GB of Qwen3).
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
T0=$(date +%s); BUDGET=$(( 5 * 3600 )); RESERVE=600
secs() { case $1 in *h) echo $(( ${1%h} * 3600 ));; *m) echo $(( ${1%m} * 60 ));; *s) echo ${1%s};; *) echo $1;; esac; }
left() { echo $(( T0 + BUDGET - RESERVE - $(date +%s) )); }
R() {  # timeout cmd...: bounded by the step's timeout and by the job's deadline; past the deadline the step is skipped
  local t=$(secs $1); shift; local l=$(left)
  if [ "$l" -lt 120 ]; then echo "SKIPPED (deadline): $*" | tee -a $OUT/skipped.txt; return 124; fi
  [ "$t" -gt "$l" ] && t=$l
  timeout -k 30 "$t" "$@"
}
killsrv() { pkill -f "llama-server|llama-batched-bench|freetoken" 2>/dev/null; sleep 5; }
# --- gates, before any download: the memory clock (prediction 1) and the device read of job 084b
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current,driver_version,power.limit --format=csv > $OUT/gpu.csv 2>&1
cat $OUT/gpu.csv
MEMCLK=$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "memory clock max ${MEMCLK} MHz (listing B reported 17001)" | tee $OUT/gate.txt
[ "$MEMCLK" = 14001 ] || { echo "WRONG MEMORY CLOCK $MEMCLK"; echo "WRONG MEMORY CLOCK $MEMCLK" >> $OUT/gate.txt; exit 4; }
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
build_tree $WORK/lc-ec "$J/llama.cpp-expert-cache-4da6337.patch" llama-server llama-batched-bench
build_tree $WORK/lc-stock "" llama-server llama-batched-bench
EC_SERVER=$WORK/lc-ec/build/bin/llama-server; EC_BB=$WORK/lc-ec/build/bin/llama-batched-bench; STOCK_BB=$WORK/lc-stock/build/bin/llama-batched-bench
ls -la $EC_BB $STOCK_BB
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
# the law's prediction for llama.cpp at the matched placements, written before any model run (as jobs 082/085)
python3 - $OUT/concur.txt $CORES > $OUT/law_llama.json <<'PY'
import json, re, statistics, sys, time
txt = open(sys.argv[1]).read(); t = int(sys.argv[2])
by = {}
for k, v in re.findall(r"cpu_read_gbs t=(\d+) ([\d.]+)", txt): by.setdefault(int(k), []).append(float(v))
pts = sorted((k, statistics.mean(v)) for k, v in by.items())
bc = pts[-1][1] if t >= pts[-1][0] else pts[0][1] if t <= pts[0][0] else next(v0 + (v1 - v0) * (t - t0) / (t1 - t0) for (t0, v0), (t1, v1) in zip(pts, pts[1:]) if t0 <= t <= t1)
pred = {f"gptoss_llama_n{n}": round(1e3 / (n * 4 * 13253760 / (bc * 1e9) * 1e3 + 5.063), 2) for n in (32, 27, 22)}
pred.update({f"qwen3_llama_n{n}": round(1e3 / (n * 8 * 9437184 / (bc * 1e9) * 1e3 + 4.3), 2) for n in (42, 36, 27)})
print(json.dumps(dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), B_c=round(bc, 2), threads=t, predicted_tok_s=pred)))
PY
cat $OUT/law_llama.json
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
bb() {  # label command: llama-batched-bench at -npp 512,2048 -ntg 128 -npl 1,2,4 (random tokens, its own warm-up)
  local lab=$1 cmd=$2
  waitfree
  local t0=$(date +%s)
  R 30m bash -c "$cmd -npp 512,2048 -ntg 128 -npl 1,2,4 --output-format jsonl" > $OUT/bb_$lab.jsonl 2> $OUT/bb_$lab.err
  local rc=$?
  echo "batched-bench $lab rc=$rc"; echo "bb_$lab 1 $rc $(( $(date +%s) - t0 ))" >> $OUT/runs.txt
  grep -h '^{' $OUT/bb_$lab.jsonl | cut -c1-300; tail -n 40 $OUT/bb_$lab.err > $OUT/bb_$lab.err.tail; rm -f $OUT/bb_$lab.err
  [ $rc = 124 ] && killsrv
}
# ---- gpt-oss-120b
LSG="-m $F -ngl 99 -fa on -lm none -t $CORES -np 1 -c 8448 --no-webui"
oursg() {  # C launch policy(dfa|lru)
  local pol=${3:-dfa} suf=law; [ $pol = lru ] && suf=lru
  cl $HFD g_ours_C$1_$suf $2 "{\"model\": \"gpt-oss-120b\", \"system\": \"ours v2\", \"C\": $1, \"table\": \"$LAWG\", \"policy\": \"$pol\"}" --extra "$LXB" \
    --cmd "$ECENV LLAMA_EC_POLICY=$pol LLAMA_EC_SLOTS=$1 LLAMA_EC_FETCH=$LAWG LLAMA_EC_STATS=$OUT/srv_g_ours_C$1_${suf}_L$2.json $EC_SERVER $LSG $OT --port {port}"
}
ftg() { ( cd $FT; cl $HFD g_ft_$1_r$2 $3 "{\"model\": \"gpt-oss-120b\", \"system\": \"FreeToken $1\", \"rate\": $2}" --ft $1 --cache-rate $2 ); }
llamag() { cl $HFD g_llama_n$1 1 "{\"model\": \"gpt-oss-120b\", \"system\": \"llama.cpp\", \"ncmoe\": $1}" --extra "$LXB" --cmd "$STOCK_SERVER $LSG -ncmoe $1 --port {port}"; }
CELLS_G="14:0.111:hybrid:32 32:0.25:hybrid:27 51:0.40:offload:22"
for cell in $CELLS_G; do IFS=: read C r b n <<< "$cell"; oursg $C 1; ftg $b $r 1; llamag $n; done
for cell in $CELLS_G; do IFS=: read C r b n <<< "$cell"; ftg $b $r 2; oursg $C 2; done
for C in 14 32 51; do oursg $C 1 lru; done
BBG="-m $F -ngl 99 -fa on -lm none -t $CORES -c 8960"
bb g_ours "$ECENV LLAMA_EC_POLICY=dfa LLAMA_EC_SLOTS=32 LLAMA_EC_FETCH=$LAWG LLAMA_EC_STATS=$OUT/bb_g_ours.json $EC_BB $BBG $OT"
bb g_stock "$STOCK_BB $BBG -ncmoe 27"
# ---- Qwen3-30B-A3B BF16 (gpt-oss files removed first for disk; the conversion runs with no server up)
rm -rf "$HFD" "$F"; df -h $WORK | tail -1
GQ=$M/Qwen3-30B-A3B-BF16.gguf
( t0=$(date +%s); PYTHONPATH=$WORK/lc-stock/gguf-py timeout 30m $VE/bin/python $WORK/lc-stock/convert_hf_to_gguf.py "$HFQ" --outtype bf16 --outfile $GQ
  echo "convert rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $GQ ) > $OUT/convert_qwen3.txt 2>&1
tail -2 $OUT/convert_qwen3.txt
LSQ="-m $GQ -ngl 99 -fa on -lm none -t $CORES -np 1 -c 2048 --no-webui --jinja"
oursq() {  # C launch policy(dfa|lru)
  local pol=${3:-dfa} suf=law; [ $pol = lru ] && suf=lru
  cl $HFQ q_ours_C$1_$suf $2 "{\"model\": \"Qwen3-30B-A3B\", \"system\": \"ours v2\", \"C\": $1, \"table\": \"$LAWQ\", \"policy\": \"$pol\"}" --extra "$LXB" \
    --cmd "$ECENV LLAMA_EC_POLICY=$pol LLAMA_EC_SLOTS=$1 LLAMA_EC_FETCH=$LAWQ LLAMA_EC_STATS=$OUT/srv_q_ours_C$1_${suf}_L$2.json $EC_SERVER $LSQ $OT --port {port}"
}
ftq() {  # backend rate launch mem-ratio (as jobs 080-082)
  ( cd $FT; cl $HFQ q_ft_$1_r$2 $3 "{\"model\": \"Qwen3-30B-A3B\", \"system\": \"FreeToken $1\", \"rate\": $2}" --ft $1 --cache-rate $2 --mem-ratio $4 )
}
llamaq() { cl $HFQ q_llama_n$1 1 "{\"model\": \"Qwen3-30B-A3B\", \"system\": \"llama.cpp\", \"ncmoe\": $1}" --extra "$LXB" --cmd "$STOCK_SERVER $LSQ -ncmoe $1 --port {port}"; }
CELLS_Q="16:0.125:hybrid:42:0.9 32:0.25:hybrid:36:0.9 56:0.4375:offload:27:0.95"
if [ -s "$GQ" ]; then
  for cell in $CELLS_Q; do IFS=: read C r b n mr <<< "$cell"; oursq $C 1; ftq $b $r 1 $mr; llamaq $n; done
  for cell in $CELLS_Q; do IFS=: read C r b n mr <<< "$cell"; ftq $b $r 2 $mr; oursq $C 2; done
  for C in 16 32 56; do oursq $C 1 lru; done
  BBQ="-m $GQ -ngl 99 -fa on -lm none -t $CORES -c 8960"
  bb q_ours "$ECENV LLAMA_EC_POLICY=dfa LLAMA_EC_SLOTS=32 LLAMA_EC_FETCH=$LAWQ LLAMA_EC_STATS=$OUT/bb_q_ours.json $EC_BB $BBQ $OT"
  bb q_stock "$STOCK_BB $BBQ -ncmoe 36"
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
    if f.endswith(".json") and (f.startswith("srv_") or f.startswith("bb_")):
        s = json.load(open(f"{out}/{f}")); n = max(1, s.get("steps", 1))
        print(f"{f:34s} table {s.get('fetch_table')} hit {s.get('hit_rate')} misses/token {s.get('misses', 0) / n:.1f} fetches {s.get('fetches', 0) / n:.1f}")
def mean(lab, L):
    v = [r["decode_tok_s"] for r in by.get(lab, []) if r["launch"] == L]
    return st.mean(v) if v else None
def paired(a, b, L):
    pa = {r["problem"]: r["decode_tok_s"] for r in by.get(a, []) if r["launch"] == L}
    pb = {r["problem"]: r["decode_tok_s"] for r in by.get(b, []) if r["launch"] == L}
    x = [pa[p] / pb[p] for p in pa if p in pb and pb[p] > 0]
    return st.mean(x) if x else None
f2 = lambda x, d=3: None if x is None else round(x, d)
cells = [("g", 14, "0.111", "hybrid", 32, 1.294, 69.9), ("g", 32, "0.25", "hybrid", 27, 1.275, 109.2), ("g", 51, "0.40", "offload", 22, 1.154, 152.5),
         ("q", 16, "0.125", "hybrid", 42, 1.032, 40.0), ("q", 32, "0.25", "hybrid", 36, 1.153, 63.1), ("q", 56, "0.4375", "offload", 27, 1.049, 108.2)]
law = json.load(open(f"{out}/law_llama.json"))["predicted_tok_s"] if os.path.exists(f"{out}/law_llama.json") else {}
line = []
for p, C, r, b, n, t1, b_speed in cells:
    o, ft, lru, ll = f"{p}_ours_C{C}_law", f"{p}_ft_{b}_r{r}", f"{p}_ours_C{C}_lru", f"{p}_llama_n{n}"
    r1, r2 = paired(o, ft, 1), paired(o, ft, 2)
    o2, ft2, l1, lr = mean(o, 2), mean(ft, 2), mean(ll, 1), mean(lru, 1)
    print(f"{p} C{C}: ours L2 {f2(o2, 1)} (listing B {b_speed}, {f2(100 * (o2 / b_speed - 1), 1) if o2 else None}%) FT {b} L2 {f2(ft2, 1)} | ours/FT L2 {f2(r2)} (Table 1 {t1}) L1 {f2(r1)} "
          f"order diff {f2(100 * (r1 / r2 - 1), 1) if r1 and r2 else None}% | LRU {f2(lr, 1)} = {f2(lr / o2, 3) if lr and o2 else None} x ours, {f2(lr / ft2, 3) if lr and ft2 else None} x FT "
          f"| llama n{n} {f2(l1, 1)} (law {law.get(('gptoss' if p == 'g' else 'qwen3') + f'_llama_n{n}')})")
    line.append(f"{p}{C} {f2(r2, 2)}")
for p in ("g", "q"):
    bbs = {}
    for s in ("ours", "stock"):
        f = f"{out}/bb_{p}_{s}.jsonl"
        if os.path.exists(f):
            for l in open(f):
                if l.startswith("{"):
                    d = json.loads(l); bbs[(s, d["pp"], d["pl"])] = (d["speed_pp"], d["speed_tg"])
    for pp in (512, 2048):
        for pl in (1, 2, 4):
            a, b_ = bbs.get(("ours", pp, pl)), bbs.get(("stock", pp, pl))
            if a or b_:
                print(f"{p} batched pp {pp} pl {pl}: ours pp {a and round(a[0], 1)} tg {a and round(a[1], 2)} | stock pp {b_ and round(b_[0], 1)} tg {b_ and round(b_[1], 2)}"
                      + (f" | pp ratio {round(a[0] / b_[0], 3)} tg ratio {round(a[1] / b_[1], 3)}" if a and b_ and b_[0] and b_[1] else ""))
open(f"{out}/oneline.txt", "w").write("089 ours/FT L2: " + " ".join(line) + f" (Table 1 1.29 1.28 1.15 1.03 1.15 1.05); gate {open(f'{out}/gate.txt').read().strip().replace(chr(10), '; ') if os.path.exists(f'{out}/gate.txt') else '?'}")
PY
ONE=$(cat $OUT/oneline.txt 2>/dev/null || echo "089: no summary")
python3 $J/manifest.py "$OUT" 089_headline_stockclock@vast "$T0" "$ONE" law_gptoss=$LAWG law_qwen3=$LAWQ memory_clock_mhz=$MEMCLK device_read_gbs=$DR
echo "SUMMARY $ONE"
