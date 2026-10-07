#!/bin/bash
# Job 107: the time relation with its overlap term (job 106, eq. sum), registered again on machines never rented before,
# with validity gates registered before launch; the admission margin (dk) and MIN's fewest-admission set, loaded either
# way, on the same machines. Engine: patch oracle5, as job 106. Protocol as jobs 100-106: 20 AIME-25 problems x 256
# teacher-forced steps per configuration, configurations in a shuffled order per process, one process per round:
# gpt-oss-120b at C = 14 (11%) and C = 32 (25%) in 2 rounds each; Qwen3-30B-A3B BF16 at C = 16 (12.5%) and C = 32 (25%),
# 1 round each.
#   base        the deployed cache (decayed count, kappa 1, the host's fetch table; admissions copied in the background)
#   dk          base with job 106's admission margin: kappa 3 (gpt-oss 11%), 2 (gpt-oss 25%), 3 (Qwen3 12.5%),
#               2 (Qwen3 25%) (prereg/online_admit.json, chosen on other text)
#   fetchplan   MIN's fewest-admission set (jobs 104-106), its admissions copied in the step; gpt-oss 11%
#   bypassplan  the same set loaded by the CPU, each admission copied in the background (a second read); gpt-oss 11%
# Before any timed run, Nsight Systems profiles base at each model and budget (as job 106): G = the GPU's own kernel time
# per token. The relation, as registered in job 106:
#   T = G + (M + A (1 - G/T)) S / B_host,  solved for T,
# M the misses and A the background admissions per token (the run's counters), S the expert's bytes (13,253,760 and
# 9,437,184), B_host the probe's highest host rate. The plain form drops the (1 - G/T).
# VALIDITY, registered here. A host that fails V0 stops and is replaced by the next offer in the list below.
#   V0 (before anything is installed or downloaded): the GPU's UUID is not in jobs/ec2/known_gpu_uuids.txt (every UUID
#      an earlier launch recorded); host memory in use (free -g, "used") is at most 48 GB (every host that ran stably
#      before showed at most 42; 106c and 106d showed 131-132); device reads >= 1500 GB/s; disk >= 150 GB.
#   V1: the deployed cache's teacher-forced loss per token (mean over problems) is within 2% of 0.190 (gpt-oss) and
#      0.0855 (Qwen3) in every round (every host of jobs 103-106 but 106c was within 0.3%). Checked after the first
#      gpt-oss round, where a failing host stops, and after the Qwen3 rounds.
#   V2: base's time per token differs by at most 2% between the two gpt-oss 11% rounds (max/min - 1). Checked after
#      them, where a failing host stops.
# A host that passes V0 but fails V1 or V2 is reported with what it ran and left out of the predictions below. The test
# needs 3 valid hosts; with fewer it is inconclusive.
# Predictions, committed before launch, over the valid hosts (ratio = the probe's link rate B_p over its CPU rate B_c):
#   1. G_prof: gpt-oss 4.0-5.0 ms and Qwen3 4.5-6.5 ms at both budgets;
#   2. the overlap form is within 6% of base's time (mean over rounds) at every host, model and budget; the median
#      |error| over all of them is at most 4%, and below the plain form's median |error|;
#   3. dk reads fewer host experts per token than base (misses + admissions, counters) at every gpt-oss cell; dk/base
#      speed is >= 0.995 at every cell, and its median over all cells is >= 1.01;
#   4. the overlap form, from each configuration's own counters, predicts dk/base within 0.03 at every cell;
#   5. at gpt-oss 11%, fetchplan/base > 1 and bypassplan/base > 1 on every host whose link reads >= 20 GB/s;
#   6. at gpt-oss 11%, bypassplan beats fetchplan on hosts with ratio < 0.35, and fetchplan beats bypassplan on hosts
#      with ratio >= 0.40 (jobs 104-105: bypassplan ahead at ratios 0.28-0.32, behind at 0.45-1.04); none in between.
# Reported without a prediction: the constant half-discount (A/2 for A (1 - G/T)) and misses-only forms (on job 106's
# stable hosts their median |error| was 1.9% and 4.0%, the overlap form's 2.3%), and predicting no change of speed for dk.
# Offers, in order (none in the rental ledger; replacements are taken in this order): 51952147 (Threadripper 3990X),
# 54579644 (Ryzen 9 5900XT), 52451721 (EPYC 9754), 54581350 (CPU not listed), 50680404 (EPYC 7663), 54519248
# (Threadripper PRO 3000), 51600687 (Xeon E5-2699 v3).
# Budget: the job stops starting steps 2 h 30 min after launch.
# Amended during the job, before host f started (predictions, gates and the job itself unchanged): 107a failed V0
# (50 GB in use) and was replaced by 50680404 (107e); 107c and 107e then failed V2, leaving at most two valid hosts.
# Of the list, 54519248 was no longer offered and 51600687 (Xeon E5, network at $0.0065/GB) would exceed the
# remaining credit; host f is the cheapest verified offer not in the ledger with download >= 1000 Mb/s and network
# at most $0.003/GB: 54055665 (Threadripper PRO 3000, 16 cores), and, as it could not be rented (no_such_ask), the
# next by that rule, 53202662 (CPU not listed, PCIe 5). Every host that started is reported.
# Amended again, before host g started: 107f failed V0 (50 GB in use) and 107b and 107d passed every check, so two
# valid hosts. Host g is the cheapest verified offer not in the ledger, not on a machine already rented in this
# job (52451718 sits beside 107c's EPYC 9754 offer, 54581350 is 107d's), with download >= 900 Mb/s and network at
# most $0.003/GB: 52395190 (Threadripper PRO 5000). This is the last host; the credit allows no other.
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
# --- V0, before anything is installed or downloaded: a machine not rented before, its memory not in use by others
JD="$(cd "$(dirname "$0")" && pwd)"
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1 | tr -d ' ')
free -g > $OUT/free.txt; USEDGB=$(awk '/Mem:/{print $3}' $OUT/free.txt)
echo "GPU UUID $UUID; host memory in use ${USEDGB} GB" | tee $OUT/v0.txt
if grep -qx "$UUID" "$JD/ec2/known_gpu_uuids.txt"; then echo "V0 FAIL: GPU rented before ($UUID), stopping" | tee -a $OUT/v0.txt; exit 8; fi
[ "${USEDGB:-999}" -le 48 ] || { echo "V0 FAIL: ${USEDGB} GB of host memory in use (> 48), stopping" | tee -a $OUT/v0.txt; exit 7; }
echo "V0 identity and memory ok" | tee -a $OUT/v0.txt
. "$JD/ec2/setup.sh"
. "$J/prof.sh"
export BASE; export -f build_tree platform getmodel
export PROF_KEEP_TAIL="G14 G32 Q16 Q32"
PATCH=$J/llama.cpp-expert-cache-4da6337-oracle5.patch
T0=$(date +%s); BUDGET=$(( 150 * 60 )); RESERVE=600
ORDER_SEED=$(( $(date +%s) % 100000 )); echo "order seed base $ORDER_SEED" | tee $OUT/order_seed.txt
NSEQ=20
secs() { case $1 in *h) echo $(( ${1%h} * 3600 ));; *m) echo $(( ${1%m} * 60 ));; *s) echo ${1%s};; *) echo $1;; esac; }
left() { echo $(( T0 + BUDGET - RESERVE - $(date +%s) )); }
R() {  # timeout cmd...: bounded by the step's timeout and by the job's deadline; past the deadline the step is skipped
  local t=$(secs $1); shift; local l=$(left)
  if [ "$l" -lt 120 ]; then echo "SKIPPED (deadline): $*" | tee -a $OUT/skipped.txt; return 124; fi
  [ "$t" -gt "$l" ] && t=$l
  timeout -k 30 "$t" "$@"
}
# --- gates: the card recorded, the device read required, the disk
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current,driver_version,power.limit --format=csv > $OUT/gpu.csv 2>&1
cat $OUT/gpu.csv
MEMCLK=$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "memory clock max ${MEMCLK} MHz" | tee $OUT/gate.txt
nvcc -O3 -arch=sm_$SM "$J/bw.cu" -o $WORK/bw0 && $WORK/bw0 > $OUT/bw_gate.txt 2>&1
DR=$(sed -n 's/device read 1 GiB: *\([0-9]*\).*/\1/p' $OUT/bw_gate.txt)
echo "device read ${DR} GB/s" | tee -a $OUT/gate.txt
[ "${DR:-0}" -ge 1500 ] || { echo "GPU reads below 1500 GB/s: throttled card, stopping"; exit 5; }
RAMGB=$(awk '/Mem:/{print $2}' $OUT/free.txt); echo "host RAM ${RAMGB} GB, in use ${USEDGB} GB" | tee -a $OUT/gate.txt
DISKGB=$(df -BG --output=avail $WORK | tail -1 | tr -dc 0-9); echo "disk free ${DISKGB} GB" | tee -a $OUT/gate.txt
[ "${DISKGB:-0}" -ge 150 ] || { echo "not enough disk for the models, stopping"; exit 6; }
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket" | tee $OUT/cpu.txt
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip >> $OUT/apt.txt 2>&1
python3 -m pip install -q --break-system-packages uv huggingface_hub hf_xet numpy numba > $OUT/pip.txt 2>&1
for i in 1 2 3; do python3 -c "import numba" 2>/dev/null && break; python3 -m pip install -q --break-system-packages numba >> $OUT/pip.txt 2>&1; done
python3 -c "import numba; print('numba', numba.__version__)" | tee -a $OUT/pip.txt
command -v uv >/dev/null || python3 -m pip install -q --break-system-packages uv >> $OUT/pip.txt 2>&1
PV=$WORK/planenv
for i in 1 2 3; do
  ( uv venv -q --clear $PV && uv pip install -q --python $PV/bin/python numpy scipy ) >> $OUT/planenv.txt 2>&1
  $PV/bin/python -c "import scipy" 2>/dev/null && break
done
$PV/bin/python -c "import numpy, scipy; print('plan env: numpy', numpy.__version__, 'scipy', scipy.__version__)" 2>&1 | tee -a $OUT/planenv.txt
$PV/bin/python -c "import scipy" 2>/dev/null || { echo "no plan environment: stopping before any timed run"; exit 4; }
CORES=$(usable_cores); echo "usable physical cores $CORES (nproc $(nproc))" | tee $OUT/cores.txt
# --- downloads first; the build, the probe and the conversion venv run meanwhile
mkdir -p $M
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
VE=$WORK/venv
( uv venv -q $VE && . $VE/bin/activate && uv pip install -q torch --index-url https://download.pytorch.org/whl/cpu &&
  uv pip install -q numpy transformers sentencepiece safetensors protobuf ) > $OUT/venv_install.txt 2>&1 &
VEP=$!
install_nsys &
NSP=$!
build_tree $WORK/lc-ec "$PATCH" llama-ec-bench
wait $NSP; install_nsys
EC_BIN=$WORK/lc-ec/build/bin/llama-ec-bench; ls -la $EC_BIN || { echo "no ec-bench binary"; exit 3; }
sha256sum $PATCH $J/minadm_plan.py $J/known_gpu_uuids.txt | tee $OUT/patch_sha.txt; echo "base $BASE" >> $OUT/patch_sha.txt
# --- the host (idle): platform, our probe, the law's tables
platform
H=$(( CORES > 4 ? CORES - 2 : 2 ))
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
grep -q "concurrent t=" $OUT/concur.txt || { echo "concur incomplete, retrying"; R 15m $WORK/concur > $OUT/concur.txt 2>&1; }
LAWG=$(python3 $J/fetch_table.py $OUT/concur.txt $H 13253760 4 $OUT/fetch_table_law_gptoss.json 37)
LAWQ=$(python3 $J/fetch_table.py $OUT/concur.txt $H 9437184 8 $OUT/fetch_table_law_qwen3.json 48)
echo "law tables ($(date -u +%FT%TZ)): gpt-oss $LAWG; qwen3 $LAWQ" | tee $OUT/tables.txt
[ -n "$LAWG" ] && [ -n "$LAWQ" ] || { echo "no law table"; exit 3; }
echo "builds and probe done, waiting for the gpt-oss download ($(( $(date +%s) - T0 )) s since launch)"
wait $DLG
cat $OUT/dl_gguf.txt
F=$M/gpt-oss-120b-MXFP4.gguf
[ -s "$F" ] || { echo "GGUF MISSING"; exit 3; }
MB="mailbox=1:tdec=1:helpers=$H"
lookahead() {  # tag gguf corpus ncmoe: record the routing of the teacher-forced text on this machine (untimed)
  local tag=$1 g=$2 corpus=$3 n=$4
  local t0=$(date +%s)
  R 30m $EC_BIN -m $g --corpus $corpus -t $CORES --n-seq $NSEQ --n-prefill 1024 --n-decode 256 --ncmoe $n --lookahead $OUT/la_$tag.bin > $OUT/la_$tag.jsonl 2> $OUT/la_$tag.err
  echo "lookahead $tag rc=$? in $(( $(date +%s) - t0 )) s: $(ls -la $OUT/la_$tag.bin | awk '{print $5}') bytes"; cat $OUT/la_$tag.bin.json
  tail -n 5 $OUT/la_$tag.err > $OUT/la_$tag.err.tail; rm -f $OUT/la_$tag.err
}
run_round() {  # tag gguf corpus C law kappa_dk round "names": one process, shuffled order
  local tag=$1 g=$2 corpus=$3 C=$4 law=$5 kdk=$6 rd=$7 names=$8
  local common="--corpus $corpus -t $CORES --n-seq $NSEQ --n-prefill 1024 --n-decode 256 --no-mmap --host-experts"
  local LA=$OUT/la_$tag.bin
  local base="slots=$C:policy=dfa:$MB:fetch=$law:paced=0:overlap=1:oracle_w=0"
  local orc="slots=$C:policy=oracle:oracle=$LA:$MB:kappa=1:fetch=$law:overlap=1:oracle_w=0"
  local st="$OUT/st_${tag}_C${C}_r${rd}"
  local cfg="" n c
  for n in $names; do
    case $n in
      base)      c="$base:kappa=1" ;;
      dk)        c="$base:kappa=$kdk" ;;
      fetchplan)  c="$orc:oracle_bypass=1:oracle_fetch=1:paced=0:oracle_plan=$OUT/plan_${tag}${C}.bin" ;;
      bypassplan) c="$orc:oracle_bypass=1:paced=0:oracle_plan=$OUT/plan_${tag}${C}.bin" ;;
    esac
    cfg="${cfg:+$cfg;}$c:stats=${st}_$n.json"
  done
  local seed=$(( ORDER_SEED + 100 * rd + C ))
  echo "cell $tag C$C round $rd order-seed $seed configs: $cfg" >> $OUT/configs.txt
  local t0=$(date +%s)
  R 100m $EC_BIN -m $g $common --order-seed $seed --ec "$cfg" > $OUT/ec_${tag}_C${C}_r${rd}.jsonl 2> $OUT/ec_${tag}_C${C}_r${rd}.err
  local rc=$?
  echo "cell $tag C$C round $rd rc=$rc in $(( $(date +%s) - t0 )) s"; echo "ec_${tag}_C${C}_r${rd} $rc $(( $(date +%s) - t0 ))" >> $OUT/runs.txt
  grep "\[ec-bench\] config [0-9]*/" $OUT/ec_${tag}_C${C}_r${rd}.err | sed 's/:stats=.*st_/ -> st_/' > $OUT/ec_${tag}_C${C}_r${rd}.order
  grep -i "oracle\|fetch table\|learned\|shuffled\|plan" $OUT/ec_${tag}_C${C}_r${rd}.err | head -40 > $OUT/ec_${tag}_C${C}_r${rd}.err.oracle
  tail -n 30 $OUT/ec_${tag}_C${C}_r${rd}.err > $OUT/ec_${tag}_C${C}_r${rd}.err.tail; rm -f $OUT/ec_${tag}_C${C}_r${rd}.err
}
gprof() {  # the GPU's own compute per token from the named profiles, before any timed run of that model
  python3 - $OUT "$@" <<'PY'
import json, sys
out = sys.argv[1]; r = {}
for lab in sys.argv[2:]:
    try:
        d = json.load(open(f"{out}/prof_{lab}.json"))["median_ms"]
        r[lab] = dict(G_prof_ms=sum(v for k, v in d.items() if k.startswith("cat_") and k not in ("cat_ec_wait", "cat_ec_copy")),
                      wall_ms=d["wall"], wait_ms=d.get("cat_ec_wait"), copy_ms=d.get("cat_ec_copy"))
    except Exception as e:
        r[lab] = dict(error=str(e))
print(json.dumps(r))
PY
}
vcheck() {  # V1: the deployed cache's loss in every round run so far; V2: base's spread over the gpt-oss 11% rounds
  python3 - $OUT "$@" <<'PY'
import glob, json, os, sys
out, what = sys.argv[1], sys.argv[2]
ref = {"g": 0.190, "q": 0.0855}
def base_rows(f):
    rows = [json.loads(l) for l in open(f) if l.strip()]
    return [r for r in rows if r["config"].rsplit("stats=", 1)[-1].endswith("_base.json")]
msg, bad = [], False
if what == "V1":
    for f in sorted(glob.glob(f"{out}/ec_*_r*.jsonl")):
        tag = os.path.basename(f)[3]
        b = base_rows(f)
        if not b:
            msg.append(f"V1 {os.path.basename(f)}: no base rows")
            bad = True
            continue
        nll = sum(r["nll_sum"] / r["nll_n"] for r in b) / len(b)
        ok = abs(nll / ref[tag] - 1) <= 0.02
        bad |= not ok
        msg.append(f"V1 {os.path.basename(f)} loss {nll:.4f} vs {ref[tag]}: {'ok' if ok else 'FAIL'}")
else:
    t = []
    for rd in (1, 2):
        b = base_rows(f"{out}/ec_g_C14_r{rd}.jsonl")
        t.append(sum(r["decode_ms"] / r["n_decode"] for r in b) / len(b) if b else float("nan"))
    v = max(t) / min(t) - 1
    bad = not (v <= 0.02)
    msg.append(f"V2 base ms/token {t[0]:.3f} {t[1]:.3f}: spread {100 * v:.2f}%: {'FAIL' if bad else 'ok'}")
print("\n".join(msg))
sys.exit(9 if bad else 0)
PY
}
# ---- gpt-oss-120b
CG=$J/aime25_owntext_gptoss_084.jsonl; wc -l $CG
lookahead g $F $CG 24
[ -s $OUT/la_g.bin ] || { echo "no gpt-oss lookahead file: oracle runs impossible"; exit 3; }
for C in 14 32; do
  prof_run G$C $EC_BIN -m $F --corpus $CG -t $CORES --n-prefill 1024 --n-decode 24 --no-mmap --host-experts --seqs 1,0 \
    --ec "slots=$C:policy=dfa:$MB:kappa=1:fetch=$LAWG:paced=0:overlap=1:oracle_w=0:seqs=1/0"
done
gprof G14 G32 > $OUT/g_prof.json
echo "G_prof (gpt-oss) written $(date -u +%FT%TZ), before any timed run:"; cat $OUT/g_prof.json
t0=$(date +%s)
( cd $J && R 20m $PV/bin/python -I minadm_plan.py $OUT/la_g.bin 14 $OUT/plan_g14.bin $CORES ) > $OUT/plan_g14.txt 2>&1
echo "plan C14 rc=$? in $(( $(date +%s) - t0 )) s"; cat $OUT/plan_g14.txt
GN14="base dk"; [ -s $OUT/plan_g14.bin ] && GN14="$GN14 fetchplan bypassplan"
FAILED=""
run_round g $F $CG 14 $LAWG 3 1 "$GN14"
vcheck V1 | tee -a $OUT/validity.txt; [ ${PIPESTATUS[0]} -eq 0 ] || { echo "V1 failed: stopping" | tee -a $OUT/validity.txt; FAILED=V1; }
if [ -z "$FAILED" ]; then
  run_round g $F $CG 14 $LAWG 3 2 "$GN14"
  vcheck V2 | tee -a $OUT/validity.txt; [ ${PIPESTATUS[0]} -eq 0 ] || { echo "V2 failed: stopping" | tee -a $OUT/validity.txt; FAILED=V2; }
fi
if [ -z "$FAILED" ]; then
  for rd in 1 2; do run_round g $F $CG 32 $LAWG 2 $rd "base dk"; done
fi
rm -f $F $OUT/la_g.bin
# ---- Qwen3-30B-A3B BF16 (converted here after the gpt-oss runs, so that nothing else runs during a timed step)
if [ -n "$FAILED" ]; then
  kill $DLQ $VEP 2>/dev/null; echo "validity $FAILED failed: Qwen3 part skipped" | tee -a $OUT/skipped.txt
else
wait $DLQ $VEP; cat $OUT/dl_qwen3.txt
GQ=$M/Qwen3-30B-A3B-BF16.gguf
( t0=$(date +%s); export PYTHONPATH=$WORK/lc-ec/gguf-py; R 30m $VE/bin/python $WORK/lc-ec/convert_hf_to_gguf.py "$HFQ" --outtype bf16 --outfile $GQ
  echo "convert rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $GQ ) > $OUT/convert_qwen3.txt 2>&1
tail -2 $OUT/convert_qwen3.txt; rm -rf "$HFQ"
if [ -s "$GQ" ]; then
  CQ=$J/aime25_owntext_qwen3_084b.jsonl
  for C in 16 32; do
    prof_run Q$C $EC_BIN -m $GQ --corpus $CQ -t $CORES --n-prefill 1024 --n-decode 24 --no-mmap --host-experts --seqs 1,0 \
      --ec "slots=$C:policy=dfa:$MB:kappa=1:fetch=$LAWQ:paced=0:overlap=1:oracle_w=0:seqs=1/0"
  done
  gprof Q16 Q32 > $OUT/q_prof.json
  echo "G_prof (Qwen3) written $(date -u +%FT%TZ), before any Qwen3 timed run:"; cat $OUT/q_prof.json
  run_round q $GQ $CQ 16 $LAWQ 3 1 "base dk"
  run_round q $GQ $CQ 32 $LAWQ 2 1 "base dk"
  vcheck V1 | tee -a $OUT/validity.txt
  rm -f $GQ
else
  echo "Qwen3 GGUF missing: Qwen3 part skipped" | tee -a $OUT/skipped.txt
fi
fi
# ---- summary
python3 - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, os, sys, statistics as st
out = sys.argv[1]
line = []
for f in sorted(os.listdir(out)):
    if not (f.startswith("ec_") and f.endswith(".jsonl")):
        continue
    rows = [json.loads(l) for l in open(f"{out}/{f}") if l.strip()]
    by = {}
    for r in rows:
        lab = [p for p in r["config"].split(":") if p.startswith("stats=")]
        lab = os.path.basename(lab[0][6:]).replace(".json", "") if lab else r["config"][:40]
        by.setdefault(lab, {})[r["seq"]] = r["decode_ms"] / r["n_decode"]
    base = next((v for k, v in by.items() if k.endswith("_base")), None)
    for lab, v in sorted(by.items()):
        ms = st.mean(v.values())
        b = st.mean(base[s] for s in v if s in base) if base else None
        stf = f"{out}/{lab}.json"
        s = json.load(open(stf)) if os.path.exists(stf) else {}
        n = max(1, s.get("steps", 1))
        print(f"{lab:26s} {1000/ms:7.2f} tok/s {ms:6.2f} ms n={len(v)} speed vs base {b/ms if b else float('nan'):.3f} "
              f"misses {s.get('misses', 0) / n:.2f} admits {s.get('admits', 0) / n:.2f}")
        if lab.endswith(("_r1_base", "_r1_dk", "_r1_fetchplan", "_r1_bypassplan")):
            line.append(f"{lab.replace('st_', '')} {1000/ms:.1f}")
open(f"{out}/oneline.txt", "w").write("107 " + " ".join(line) + "\n")
PY
ONE=$(cat $OUT/oneline.txt)
python3 $J/manifest.py "$OUT" 107_newhosts@vast "$T0" "$ONE" law_gptoss=$LAWG law_qwen3=$LAWQ device_read_gbs=$DR memory_clock_max_mhz=$MEMCLK
cat $OUT/skipped.txt 2>/dev/null
cat $OUT/validity.txt 2>/dev/null
echo "SUMMARY: $ONE"
