#!/bin/bash
# Job 088: the slow-link comparison done fairly. Job 085 ran gpt-oss-120b next to a Core i9-14900K (RTX 5090, PCIe about
# 25 GB/s) and FreeToken's hybrid backend collapsed (12.0 / 30.5 / 66.0 tok/s at 11 / 25 / 40% against its offload
# backend's 24.6 / 48.2 / 86.0): its CPU executor spread 23 threads over the 8 performance and 16 efficiency cores (its
# log: "CPU MoE executor ready: threads=23 (pinned to cores 0..30) isa=avx2") and its calibration then fetched 65.5% of
# misses over the slow link. Here the same CPU model (Vast offer 48829099: RTX 5090, i9-14900K, 193 GB RAM, PCIe measured
# 25.5 GB/s; 137 Mb/s download, so the two gpt-oss downloads, 120 GB, start first and the builds run meanwhile) with
# FreeToken's CPU executor on 8 threads (--moe-cpu-threads 8: FreeToken pins an explicit count to its physical-core
# representatives in CPU order, cpu 0,2,..,14 on a 14900K, i.e. the 8 P-cores; the mapping it will use is computed and
# recorded in ft_affinity.txt, and the "pinned to cores" line of every FreeToken log is kept) and its calibration run on
# 8 threads (ft bench bw --cpu-threads 8 -o benchbw_t8.json, read by the hybrid-auto runs through FREETOKEN_BENCHBW_PATH).
# gpt-oss-120b only, Table 1 protocol (30 AIME-25 problems, 256 tokens, greedy, held-out warm-up, session, GPU-side
# sampling for llama-server), C 14 / 32 / 51 (FreeToken rates 0.111 / 0.25 / 0.40).
# Run list. Launch 1, per budget in this order:
#   FreeToken offload (default settings; the same-machine anchor for job 085),
#   FreeToken hybrid, --moe-cpu-threads 8, --moe-hybrid-max-fetch automatic from the 8-thread calibration,
#   FreeToken hybrid, --moe-cpu-threads 8, --moe-hybrid-max-fetch 0 (every miss computed on the CPU),
#   ours with the law's table computed on this machine before any model run (as job 085).
# Then once: stock llama.cpp at 25% (-ncmoe 27, GPU sampling, the paper's baseline) with -t 8 pinned one thread per
#   P-core (-C mask --cpu-strict 1 on cpu 0,2,..,14, the pairing of FreeToken's 8 pinned threads; pmask.txt), and with
#   -t $CORES as job 085 ran it; the law's prediction for both thread counts is written before the runs (law_llama.json).
# Launch 2 (confirmation; comparisons use it) in the reversed order: budgets 51, 32, 14, and within each budget ours
#   first, then FreeToken's best configuration of launch 1 by mean speed (selection.txt).
# Note: with an explicit --moe-cpu-threads FreeToken does not reserve a core for its flag coordinator thread (it does in
#   auto mode, which is why job 085's log showed 23 threads on 24 cores); the coordinator then floats.
# Predictions, committed before launch:
#   1. FreeToken's hybrid backend on 8 threads runs faster than job 085's default-thread hybrid (12.0 / 30.5 / 66.0 tok/s)
#      by at least 30% at every budget;
#   2. FreeToken's best configuration improves on job 085's best (24.6 / 48.2 / 86.0 tok/s, its offload backend) at 2 or
#      more of the 3 budgets;
#   3. ours with the law's table leads FreeToken's best configuration at every budget (launch 2, paired CI above 1), by
#      less than in job 085 (1.43-2.03x): between 1.10x and 1.50x at every budget;
#   4. llama.cpp with -t 8 at 25% runs within 10% of the law's prediction written on the machine, and faster than with
#      -t $CORES;
#   5. the law's table computed on this machine is 0,0,0,1,1 (job 085's table on the same CPU model) or fetches less at
#      every entry.
# Budget: the job stops starting steps 4.5 h after launch (every step is bounded by its own timeout and by that deadline;
# skipped steps are listed in skipped.txt); disk needed about 150 GB (HF checkpoint 61 GB + GGUF 59 GB + builds).
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
T0=$(date +%s); BUDGET=$(( 4 * 3600 + 30 * 60 )); RESERVE=600
secs() { case $1 in *h) echo $(( ${1%h} * 3600 ));; *m) echo $(( ${1%m} * 60 ));; *s) echo ${1%s};; *) echo $1;; esac; }
left() { echo $(( T0 + BUDGET - RESERVE - $(date +%s) )); }
R() {  # timeout cmd...: bounded by the step's timeout and by the job's deadline; past the deadline the step is skipped
  local t=$(secs $1); shift; local l=$(left)
  if [ "$l" -lt 120 ]; then echo "SKIPPED (deadline): $*" | tee -a $OUT/skipped.txt; return 124; fi
  [ "$t" -gt "$l" ] && t=$l
  timeout -k 30 "$t" "$@"
}
killsrv() { pkill -f "llama-server|freetoken" 2>/dev/null; sleep 5; }
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip >> $OUT/apt.txt 2>&1
python3 -m pip install -q --break-system-packages uv huggingface_hub hf_xet > $OUT/pip_uv.txt 2>&1
CORES=$(usable_cores); echo "usable physical cores $CORES (nproc $(nproc))" | tee $OUT/cores.txt
# --- downloads first (the slowest part of this rental); builds and installs run meanwhile
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
# --- FreeToken (0d652e7, as every earlier job)
FT=$WORK/ft; export FT_DIR=$FT
( cd $WORK && git init -q ft && cd ft && git remote add origin https://github.com/FlashML-org/FreeToken &&
  git fetch -q --depth 1 origin 0d652e73a452d014ac5441a15baa75348e9fcb0a && git checkout -q FETCH_HEAD ) || echo "FreeToken clone FAILED"
nvcc --version | tail -2 | tee $OUT/nvcc_version.txt   # FreeToken builds kernels with nvcc and needs the CUDA 13 toolkit (torch cu130)
( cd $FT && uv venv -q .venv && . .venv/bin/activate && time uv pip install -q -e ".[accel]" huggingface_hub hf_xet gguf sentencepiece transformers safetensors ) > $OUT/ft_install.txt 2>&1
echo "FreeToken install rc=$?"
PY=$FT/.venv/bin/python; FTBIN=$FT/.venv/bin/ft
$PY -c "import freetoken, huggingface_hub" > $OUT/ft_import.txt 2>&1 || { echo "FreeToken not importable, stopping"; cat $OUT/ft_import.txt; tail -20 $OUT/ft_install.txt; kill $DLH $DLG 2>/dev/null; exit 3; }
$FTBIN --version > $OUT/ft_version.txt 2>&1
# what FreeToken will pin --moe-cpu-threads 8 to on this host (its resolve_threads_and_affinity), and the CPU topology
$PY - <<'PY' > $OUT/ft_affinity.txt 2>&1
import os
from freetoken.moe.cpu_executor import resolve_threads_and_affinity as r, physical_core_cpus as p
print("physical core representatives (cpu ids):", p())
print("--moe-cpu-threads 8 -> threads, cpu ids:", r(8))
print("auto (0) -> threads, cpu ids:", r(0), "(auto also gives the coordinator the last one)")
for c in sorted(os.sched_getaffinity(0)):
    b = f"/sys/devices/system/cpu/cpu{c}/topology/"
    try:
        print(f"cpu {c}: core {open(b + 'core_id').read().strip()} siblings {open(b + 'thread_siblings_list').read().strip()}")
    except OSError as e:
        print(f"cpu {c}: {e}")
PY
lscpu -e >> $OUT/ft_affinity.txt 2>&1
cat $OUT/ft_affinity.txt | head -4
# llama.cpp's -t 8 run is pinned one thread per P-core (strict placement on the first hardware thread of each core that
# has an SMT sibling: cpu 0,2,..,14 on a 14900K, the cores FreeToken's 8 threads use); without a visible hybrid topology
# the 8 threads float over the first 16 CPUs instead
read PMASK PSTRICT PCPUS < <(python3 - <<'PY'
import os
cpus = sorted(os.sched_getaffinity(0)); reps, seen = [], set()
for c in cpus:
    try:
        s = open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read().strip()
    except OSError:
        continue
    if ("," in s or "-" in s) and s not in seen:   # a core with SMT siblings: a P-core on a hybrid Intel part
        seen.add(s); reps.append(c)
if len(reps) >= 8:
    p, strict = reps[:8], 1
else:
    p, strict = cpus[:16], 0
m = 0
for c in p: m |= 1 << c
print(f"{m:x}", strict, ",".join(map(str, p)))
PY
)
echo "llama.cpp -t 8: cpu mask $PMASK strict $PSTRICT (cpus $PCPUS)" | tee $OUT/pmask.txt
# AIME-25 (the 30 measured problems) and the held-out warm-up prompt (AIME-24 problem 0, formatted like theirs)
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
echo "builds done, waiting for the downloads ($(( $(date +%s) - T0 )) s since launch)"
wait $DLG $DLH
cat $OUT/dl_gguf.txt $OUT/dl_hf.txt
F=$M/gpt-oss-120b-MXFP4.gguf
[ -f "$HFD/config.json" ] && echo "HF checkpoint ok" || echo "HF checkpoint MISSING"
[ -s "$F" ] && echo "GGUF ok $(stat -c %s $F) bytes" || echo "GGUF MISSING"
# --- the host (downloads finished, the machine idle): FreeToken's two calibrations, our probe, the law's table
platform
( cd $FT && R 25m $FTBIN bench bw ) > $OUT/ft_bench_bw.txt 2>&1; cp ~/.cache/freetoken/benchbw/*.json $OUT/ 2>/dev/null
( cd $FT && R 25m $FTBIN bench bw --cpu-threads 8 -o $WORK/benchbw_t8.json ) > $OUT/ft_bench_bw_t8.txt 2>&1; cp $WORK/benchbw_t8.json $OUT/benchbw_t8.json 2>/dev/null
grep -h "ceilings\|mxfp4\|hybrid fetches\|cpu " $OUT/ft_bench_bw.txt $OUT/ft_bench_bw_t8.txt | head -20
H=$(( CORES > 4 ? CORES - 2 : 2 ))
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
grep -q "concurrent t=" $OUT/concur.txt || { echo "concur incomplete, retrying"; R 15m $WORK/concur > $OUT/concur.txt 2>&1; }
LAWG=$(python3 $J/fetch_table.py $OUT/concur.txt $H 13253760 4 $OUT/fetch_table_law_gptoss.json 37)
echo "law table ($(date -u +%FT%TZ)): gpt-oss $LAWG (job 085 on this CPU model: 0,0,0,1,1; headline machine 0,0,1,1,2)" | tee $OUT/tables.txt
[ -n "$LAWG" ] || { echo "no law table"; exit 3; }
# prediction 4, written before any model run: llama.cpp -ncmoe 27 reads all 4 routed experts of 27 layers on the CPU;
# T = G + 27 * 4 * S / B_c at the run's thread count (B_c from the probe at that count), G frozen (5.063 ms, jobs 064/066)
python3 - $OUT/concur.txt $CORES > $OUT/law_llama.json <<'PY'
import json, re, statistics, sys, time
txt = open(sys.argv[1]).read(); cores = int(sys.argv[2])
by = {}
for k, v in re.findall(r"cpu_read_gbs t=(\d+) ([\d.]+)", txt): by.setdefault(int(k), []).append(float(v))
pts = sorted((k, statistics.mean(v)) for k, v in by.items())
def bc(t):
    return pts[-1][1] if t >= pts[-1][0] else pts[0][1] if t <= pts[0][0] else next(v0 + (v1 - v0) * (t - t0) / (t1 - t0) for (t0, v0), (t1, v1) in zip(pts, pts[1:]) if t0 <= t <= t1)
pred = {f"gptoss_llama_n27_t{t}": round(1e3 / (27 * 4 * 13253760 / (bc(t) * 1e9) * 1e3 + 5.063), 2) for t in (8, cores)}
print(json.dumps(dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), B_c={str(t): round(bc(t), 2) for t in (8, cores)}, G_ms=5.063, predicted_tok_s=pred)))
PY
cat $OUT/law_llama.json
LXB='{"min_p": 0.0, "seed": 1234, "backend_sampling": true}'
OT="-ot '\\.ffn_(up|down|gate|gate_up)_exps=CUDA_Host'"
ECENV="env LLAMA_EC_POLICY=dfa LLAMA_EC_KAPPA=1 LLAMA_EC_MAILBOX=1 LLAMA_EC_TDEC=1 LLAMA_EC_HELPERS=$H"
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
oursg() {  # C launch (the law's table)
  cl $HFD g_ours_C$1_law $2 "{\"model\": \"gpt-oss-120b\", \"system\": \"ours v2\", \"C\": $1, \"table\": \"$LAWG\"}" --extra "$LXB" \
    --cmd "$ECENV LLAMA_EC_SLOTS=$1 LLAMA_EC_FETCH=$LAWG LLAMA_EC_STATS=$OUT/srv_g_ours_C$1_law_L$2.json $EC_SERVER $LSG $OT --port {port}"
}
ftg() {  # variant rate launch; variants: offload (default settings) | hybrid8 (8 threads, fetch auto from the 8-thread
  # calibration) | hybrid8f0 (8 threads, --moe-hybrid-max-fetch 0). bs1_client appends --ft-extra to their serve command
  # and passes --hybrid-fetch as --moe-hybrid-max-fetch; FREETOKEN_BENCHBW_PATH points the engine at the 8-thread profile.
  local v=$1 r=$2 L=$3 b=offload; local -a x=()
  case $v in
    hybrid8)   b=hybrid; x=("--ft-extra=--moe-cpu-threads 8");;
    hybrid8f0) b=hybrid; x=("--ft-extra=--moe-cpu-threads 8" --hybrid-fetch 0);;
  esac
  ( cd $FT; [ $v = hybrid8 ] && export FREETOKEN_BENCHBW_PATH=$WORK/benchbw_t8.json
    cl $HFD g_ft_${v}_r$r $L "{\"model\": \"gpt-oss-120b\", \"system\": \"FreeToken $b\", \"variant\": \"$v\", \"rate\": $r}" --ft $b --cache-rate $r "${x[@]}" )
}
for pair in 14:0.111 32:0.25 51:0.40; do
  C=${pair%%:*}; r=${pair#*:}
  ftg offload $r 1; ftg hybrid8 $r 1; ftg hybrid8f0 $r 1; oursg $C 1
done
cl $HFD g_llama_n27_t8 1 '{"model": "gpt-oss-120b", "system": "llama.cpp", "ncmoe": 27, "threads": 8}' --extra "$LXB" \
  --cmd "$STOCK_SERVER -m $F -ngl 99 -fa on -lm none -t 8 -C $PMASK --cpu-strict $PSTRICT -np 1 -c 8448 --no-webui -ncmoe 27 --port {port}"
cl $HFD g_llama_n27_tall 1 "{\"model\": \"gpt-oss-120b\", \"system\": \"llama.cpp\", \"ncmoe\": 27, \"threads\": $CORES}" --extra "$LXB" \
  --cmd "$STOCK_SERVER $LSG -ncmoe 27 --port {port}"
# launch 2, reversed: budgets 51, 32, 14; ours first, then FreeToken's best configuration of launch 1
for pair in 51:0.40 32:0.25 14:0.111; do
  C=${pair%%:*}; r=${pair#*:}
  fb=$(pick g_ft_offload_r$r g_ft_hybrid8_r$r g_ft_hybrid8f0_r$r); v=$(echo $fb | cut -d_ -f3)
  echo "gpt-oss C$C: ours law, FreeToken $v" | tee -a $OUT/selection.txt
  oursg $C 2; ftg $v $r 2
done
cd $WORK
# FreeToken logs keep their start-up (thread pinning, calibration line) and their end; the others their tail
for f in $OUT/srv_g_ft_*.log; do [ -f "$f" ] && { head -n 200 "$f"; echo "... [$(wc -l < "$f") lines in all]"; tail -n 60 "$f"; } > "$f.keep"; done
for f in $OUT/srv_*.log; do [ -f "$f.keep" ] || tail -n 40 "$f" > "$f.tail"; done; find $OUT -name 'srv_*.log' -delete
# the settings each FreeToken server reported: strategy, thread count, fetch cap, the executor's pinning line, the calibration line
for f in $OUT/srv_g_ft_*.keep; do
  [ -f "$f" ] || continue
  echo "== $(basename "$f")"
  sed 's/\x1b\[[0-9;]*m//g' "$f" | grep -o "moe_strategy='[a-z]*'\|moe_cpu_threads=[0-9]*\|moe_hybrid_max_fetch=[-0-9]*\|CPU MoE executor ready: threads=[0-9]* (pinned to cores [0-9.]*) isa=[a-z0-9_]*\|moe-hybrid-max-fetch auto: [^;]*" | sort -u
done > $OUT/ft_lines.txt 2>/dev/null
cat $OUT/ft_lines.txt
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
        s = json.load(open(f"{out}/{f}")); n = s["steps"]
        print(f"{f:34s} table {s.get('fetch_table')} misses/token {s['misses'] / n:.1f} fetches {s.get('fetches', 0) / n:.1f}")
def mean(lab, L):
    v = [r["decode_tok_s"] for r in by.get(lab, []) if r["launch"] == L]
    return st.mean(v) if v else None
def paired(a, b, L):  # mean of per-problem ratios a/b on launch L
    pa = {r["problem"]: r["decode_tok_s"] for r in by.get(a, []) if r["launch"] == L}
    pb = {r["problem"]: r["decode_tok_s"] for r in by.get(b, []) if r["launch"] == L}
    x = [pa[p] / pb[p] for p in pa if p in pb and pb[p] > 0]
    return (st.mean(x), len(x)) if x else (None, 0)
sel = {}
if os.path.exists(f"{out}/selection.txt"):
    for l in open(f"{out}/selection.txt"):
        if ":" in l: sel[l.split(":")[0].split("C")[-1]] = l.strip().split()[-1]
line = []
for C, r, ref85 in ((14, "0.111", (24.6, 12.0)), (32, "0.25", (48.2, 30.5)), (51, "0.40", (86.0, 66.0))):
    v = sel.get(str(C)); ft = f"g_ft_{v}_r{r}" if v else None
    o2, f2 = mean(f"g_ours_C{C}_law", 2), (mean(ft, 2) if ft else None)
    h1 = mean(f"g_ft_hybrid8_r{r}", 1)
    rat, n = paired(f"g_ours_C{C}_law", ft, 2) if ft else (None, 0)
    print(f"C{C}: ours L2 {o2 and round(o2, 1)} | FreeToken best = {v} L2 {f2 and round(f2, 1)} (job 085 best {ref85[0]}) | ours/FT paired {rat and round(rat, 3)} (n={n}) "
          f"| hybrid8 L1 {h1 and round(h1, 1)} vs 085 hybrid {ref85[1]} ({h1 and round(100 * (h1 / ref85[1] - 1))}%)")
    line.append(f"C{C} {o2 and round(o2, 1)}/{f2 and round(f2, 1)}={rat and round(rat, 2)}({v})")
l8, la = mean("g_llama_n27_t8", 1), mean("g_llama_n27_tall", 1)
law = json.load(open(f"{out}/law_llama.json"))["predicted_tok_s"] if os.path.exists(f"{out}/law_llama.json") else {}
print(f"llama.cpp n27: -t 8 {l8 and round(l8, 2)} (law {law.get('gptoss_llama_n27_t8')}), -t all {la and round(la, 2)} (law {[v for k, v in law.items() if not k.endswith('_t8')]})")
open(f"{out}/oneline.txt", "w").write("088 ours/FT(L2): " + " ".join(line) + f"; llama -t8 {l8 and round(l8, 1)} (law {law.get('gptoss_llama_n27_t8')}) -tall {la and round(la, 1)}; table {open(f'{out}/tables.txt').read().split(' ')[4] if os.path.exists(f'{out}/tables.txt') else '?'}")
PY
ONE=$(cat $OUT/oneline.txt 2>/dev/null || echo "088: no summary")
python3 $J/manifest.py "$OUT" 088_slowlink_fair@vast "$T0" "$ONE" law_table=$LAWG cpu_mask_t8=$PMASK
echo "SUMMARY $ONE"
