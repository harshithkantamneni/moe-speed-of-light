#!/bin/bash
# Job 109: an online policy built only from the engine's own mechanisms (no foresight), against the engine's oracles,
# on desktop-class machines never rented before, with the population fixed before any machine starts. Engine: patch
# oracle5 (jobs 106-108). gpt-oss-120b at C = 14 (11%) and C = 32 (25%); 20 AIME-25 problems x 256 teacher-forced steps
# (the job 084 own-text corpus); configurations in fresh contexts, in a seeded shuffled order within each process.
#   base    the deployed cache: decayed count, kappa 1, the host's fetch table; admissions copied in the background
#   foa     base with each admission fetched in the step (one read; jobs 097-099)
#   bypass  MIN with bypass, admissions copied in the background (two reads; jobs 096-105)
#   fetch   MIN with bypass, admissions fetched in the step (one read)
#   both3p  fetch with the paced single-read prefetch, lead 3 (job 099)
#   dk      base with job 106's admission margin: kappa 3 at C = 14, 2 at C = 32 (prereg/online_admit.json)
#   pf      base with the layer-ahead PREFETCH: layer l+1's router applied to layer l's MoE input predicts layer l+1's
#           experts, and the top predicted non-resident one is copied into a slot on a side stream (jobs 065-071, 105)
#   R1      the online policy of this job: read once and admit by prediction, i.e. base + fetch-on-admit + PREFETCH
#           (every admission is read once, by the GPU, either in the step or one layer ahead from the prediction)
#   R2      R1 admitting less: R1 with dk's kappa
# R1 and R2 use no foresight. They combine mechanisms that each ran before in this engine; the combination had not run
# before this job. A smoke run first (109s: SMOKE=1, gpt-oss-20b at C = 4 and 8, 3 problems x 64 steps, gates recorded
# but not enforced) must show every configuration completing and R1 and R2 passing V3 before any host below starts.
# Each round is two processes: A = base foa bypass fetch both3p dk pf, B = base R1 R2 (the new combination runs in its
# own process, with its own base). Rounds: C = 14 twice (A then B, twice), then C = 32 once (A then B). Before any timed
# run: this machine's routing of the text (lookahead, untimed) and Nsight profiles of base at each budget (as job 108).
# VALIDITY, registered here (a host that fails stops and is replaced by the next offer in the list below):
#   V0 (before anything is installed or downloaded): the GPU's UUID is in no earlier launch (jobs/ec2/known_gpu_uuids.txt),
#      at most 48 GB of host memory in use;
#   V0c (after the tools are installed, before the model is read): one NUMA node and at most 32 usable physical cores
#      (desktop-class, as job 108); device reads >= 1500 GB/s; disk >= 100 GB;
#   VR (after the probe, before any model run): ratio = B_p / B_c from the host's fetch table (fetch_table.py's reading
#      of the probe, as jobs 099-108) is at least 0.5. This line was drawn after the data of jobs 093-101 (the paper's
#      "fast-link" machines); it is fixed here, before any machine of this job starts;
#   V1: base's loss per token (mean over problems) within 2% of 0.190, in both processes of the first round (a failing
#      host stops) and of the C = 32 round;
#   V2: process A's base time per token differs by at most 2% between the two C = 14 rounds (a failing host stops);
#   V3: every configuration's loss within 1% of base's in the same process (within-process differences on jobs
#      099-107 were at most 0.6%); a configuration that fails is left out of the predictions on that machine.
# A machine is valid if it passes V0, V0c, VR, V1 and V2. Every host that starts is reported.
# Definitions: X/base = base's mean time per token over the problems / X's, both from the same process; per machine and
# budget, the geometric mean over its rounds. Eq. (1) = R* S / B_host (R* = 38.315 at C = 14 and 15.340 at C = 32,
# prereg/speed_limit_v2.json; S = 13,253,760 B; B_host the probe's highest host rate, scripts/reanalysis.py); the gap =
# base - Eq. (1) in ms per token (base's times as means over the rounds of process A); together = (base - fetch) / gap;
# interaction = (base - fetch) - (base - bypass) - (base - foa) in ms; residual = (both3p - Eq. (1) - 2.94 ms -
# (reads - R*) S / B_host) / gap, reads = both3p's misses + admissions + prefetches per token (counters), 2.94 ms the
# smallest profiled T_GPU (scripts/decomp_measured.py); capture = (best/base - 1) / (both3p/base - 1), best the largest
# of dk, pf, R1, R2 vs base.
# Predictions, committed before launch, on every valid machine unless stated:
#   1. at 11%, fetch/base >= max(bypass/base, foa/base) + 0.04; the interaction is positive at both budgets;
#   2. at 11%, foa/base and bypass/base each within [0.96, 1.05];
#   3. together, mean over the valid machines: [0.25, 0.55] at 11% and [0.20, 0.50] at 25%;
#   4. both3p > fetch at 25% on every machine, and at 11% on all but at most one;
#   5. bypass/base >= 1.08 at 25%;
#   6. the residual, mean over the valid machines, within +-0.10 at 11% and +-0.15 at 25%;
#   7. no-foresight policies capture little of the oracle's gain: at both budgets, best/base <= 1.10 and capture
#      <= 0.35 on every machine, and at 11% the mean capture over the machines is at most 0.25;
#   8. pf/base at 11% is below 1.00 on machines with ratio below 0.75, and at least 0.98 on machines with ratio at least
#      0.9 (job 105: 0.81 at ratio 0.45 and 1.04 at 1.04; nothing measured in between);
#   9. at 11%, |R1/base - pf/base| <= 0.04 (reading admissions once adds little to prediction);
#  10. dk/base within [0.98, 1.06] at both budgets, and R2/R1 within [0.97, 1.06];
#  11. R1 and R2 read at least 1.3 R* host experts per token (misses + admissions + prefetches, counters, mean over
#      rounds) at both budgets.
# The test needs at least four valid machines; with fewer it is inconclusive.
# Hosts: RTX 5090, whole machine, from these offers, in order (none in the rental ledger; offers no longer listed are
# skipped): 54598851 (Ryzen 7 9800X3D), 43221030 (Ryzen 9 9950X3D), 54227874 (Ryzen 9 7945HX), 51325952 (Core i9-13900KF),
# 52126290 (Core Ultra 9 285K), 54573923 (Ryzen 9 5950X), 54573925 (Ryzen 9 5950X), 54573659 (Ryzen 9 5950X), 54227914
# (Ryzen 9 7945HX), 54716110 (Ryzen 9 9950X). The first five start together; each host that fails a gate is replaced
# by the next offer, until five machines are valid, nine hosts have started, or the credit left is below $2.
# Budget: the job stops starting steps 125 minutes after launch (45 with SMOKE=1).
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = 1 ]; then
  MREPO=ggml-org/gpt-oss-20b-GGUF; MFILE=gpt-oss-20b-MXFP4.gguf; CA=4; CB=8; NSEQ=3; NDEC=64; BMIN=45
else
  MREPO=ggml-org/gpt-oss-120b-GGUF; MFILE=gpt-oss-120b-MXFP4.gguf; CA=14; CB=32; NSEQ=20; NDEC=256; BMIN=125
fi
KA=3; KB=2
echo "SMOKE=$SMOKE model $MFILE budgets C=$CA,$CB problems $NSEQ steps $NDEC" | tee $OUT/mode.txt
JD="$(cd "$(dirname "$0")" && pwd)"
gate_fail() {  # exit-code message: a failed gate stops the host (recorded only in the smoke run)
  local code=$1; shift
  echo "$*" | tee -a $OUT/v0.txt
  if [ "$SMOKE" = 1 ]; then echo "(smoke run: recorded, not enforced)" | tee -a $OUT/v0.txt; return 0; fi
  exit $code
}
# --- V0, before anything is installed or downloaded
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1 | tr -d ' ')
free -g > $OUT/free.txt; USEDGB=$(awk '/Mem:/{print $3}' $OUT/free.txt)
echo "GPU UUID $UUID; host memory in use ${USEDGB} GB" | tee $OUT/v0.txt
if grep -qx "$UUID" "$JD/ec2/known_gpu_uuids.txt"; then gate_fail 8 "V0 FAIL: GPU rented before ($UUID), stopping"; fi
[ "${USEDGB:-999}" -le 48 ] || gate_fail 7 "V0 FAIL: ${USEDGB} GB of host memory in use (> 48), stopping"
echo "V0 checked" | tee -a $OUT/v0.txt
. "$JD/ec2/setup.sh"
. "$J/prof.sh"
export BASE; export -f build_tree platform getmodel
export PROF_KEEP_TAIL="G$CA G$CB"
PATCH=$J/llama.cpp-expert-cache-4da6337-oracle5.patch
T0=$(date +%s); BUDGET=$(( BMIN * 60 )); RESERVE=300
ORDER_SEED=$(( $(date +%s) % 100000 )); echo "order seed base $ORDER_SEED" | tee $OUT/order_seed.txt
secs() { case $1 in *h) echo $(( ${1%h} * 3600 ));; *m) echo $(( ${1%m} * 60 ));; *s) echo ${1%s};; *) echo $1;; esac; }
left() { echo $(( T0 + BUDGET - RESERVE - $(date +%s) )); }
R() {  # timeout cmd...: bounded by the step's timeout and by the job's deadline; past the deadline the step is skipped
  local t=$(secs $1); shift; local l=$(left)
  if [ "$l" -lt 120 ]; then echo "SKIPPED (deadline): $*" | tee -a $OUT/skipped.txt; return 124; fi
  [ "$t" -gt "$l" ] && t=$l
  timeout -k 30 "$t" "$@"
}
# --- V0c: the class, the card, the disk (before any download)
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket|NUMA node\(s\)" | tee $OUT/cpu.txt
NUMA=$(lscpu | sed -n 's/^NUMA node(s): *\([0-9]*\).*/\1/p'); CORES=$(usable_cores)
echo "usable physical cores $CORES (nproc $(nproc)); NUMA nodes ${NUMA:-?}" | tee $OUT/cores.txt
[ "${NUMA:-9}" -eq 1 ] && [ "${CORES:-99}" -le 32 ] || gate_fail 9 "V0c FAIL: ${NUMA} NUMA nodes, ${CORES} usable cores: not desktop-class, stopping"
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current,driver_version,power.limit --format=csv > $OUT/gpu.csv 2>&1
MEMCLK=$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "memory clock max ${MEMCLK} MHz" | tee $OUT/gate.txt
nvcc -O3 -arch=sm_$SM "$J/bw.cu" -o $WORK/bw0 && $WORK/bw0 > $OUT/bw_gate.txt 2>&1
DR=$(sed -n 's/device read 1 GiB: *\([0-9]*\).*/\1/p' $OUT/bw_gate.txt)
echo "device read ${DR} GB/s" | tee -a $OUT/gate.txt
[ "${DR:-0}" -ge 1500 ] || gate_fail 5 "V0c FAIL: GPU reads below 1500 GB/s: throttled card, stopping"
RAMGB=$(awk '/Mem:/{print $2}' $OUT/free.txt); echo "host RAM ${RAMGB} GB, in use ${USEDGB} GB" | tee -a $OUT/gate.txt
DISKGB=$(df -BG --output=avail $WORK | tail -1 | tr -dc 0-9); echo "disk free ${DISKGB} GB" | tee -a $OUT/gate.txt
[ "${DISKGB:-0}" -ge 100 ] || gate_fail 6 "V0c FAIL: not enough disk, stopping"
echo "V0c checked" | tee -a $OUT/v0.txt
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip >> $OUT/apt.txt 2>&1
python3 -m pip install -q --break-system-packages huggingface_hub hf_xet numpy > $OUT/pip.txt 2>&1
# --- the download first; the build and the probe run meanwhile
mkdir -p $M
( t0=$(date +%s)
  python3 -c "import sys; from huggingface_hub import hf_hub_download as d; d(sys.argv[2], sys.argv[3], local_dir=sys.argv[1])" "$M" "$MREPO" "$MFILE"
  echo "gguf hf download rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $M/$MFILE
  if [ ! -s $M/$MFILE ]; then
    rm -f $M/$MFILE
    for i in 1 2 3; do getmodel $MREPO $MFILE; [ -f $M/$MFILE ] && break; done
  fi
) > $OUT/dl_gguf.txt 2>&1 &
DLG=$!
install_nsys &
NSP=$!
build_tree $WORK/lc-ec "$PATCH" llama-ec-bench
wait $NSP; install_nsys
EC_BIN=$WORK/lc-ec/build/bin/llama-ec-bench; ls -la $EC_BIN || { echo "no ec-bench binary"; exit 3; }
sha256sum $PATCH $J/known_gpu_uuids.txt | tee $OUT/patch_sha.txt; echo "base $BASE" >> $OUT/patch_sha.txt
# --- the host (idle): platform, our probe, the fetch table, the ratio
platform
H=$(( CORES > 4 ? CORES - 2 : 2 ))
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
grep -q "concurrent t=" $OUT/concur.txt || { echo "concur incomplete, retrying"; R 15m $WORK/concur > $OUT/concur.txt 2>&1; }
LAWG=$(python3 $J/fetch_table.py $OUT/concur.txt $H 13253760 4 $OUT/fetch_table_law_gptoss.json 37)
echo "law table ($(date -u +%FT%TZ)): gpt-oss $LAWG" | tee $OUT/tables.txt
[ -n "$LAWG" ] || { echo "no law table"; exit 3; }
RATIO=$(python3 -c "import json, sys; d = json.load(open(sys.argv[1])); print(f'{d[\"B_p\"] / d[\"B_c\"]:.4f}')" $OUT/fetch_table_law_gptoss.json)
echo "VR: B_p / B_c = ${RATIO:-?}" | tee -a $OUT/gate.txt
if ! python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= 0.5 else 1)" "${RATIO:-0}"; then
  if [ "$SMOKE" != 1 ]; then pkill -P $DLG; kill $DLG; fi
  gate_fail 10 "VR FAIL: link/CPU ratio ${RATIO} below 0.5: outside the registered population, stopping"
fi
echo "VR checked" | tee -a $OUT/v0.txt
echo "build and probe done, waiting for the download ($(( $(date +%s) - T0 )) s since launch)"
wait $DLG
cat $OUT/dl_gguf.txt
F=$M/$MFILE
[ -s "$F" ] || { echo "GGUF MISSING"; exit 3; }
MB="mailbox=1:tdec=1:helpers=$H"
PACED="paced=1:chunk=16777216:max_pending=64"
CG=$J/aime25_owntext_gptoss_084.jsonl; wc -l $CG
# --- this machine's routing of the text (untimed), for the oracle configurations
t0=$(date +%s)
R 30m $EC_BIN -m $F --corpus $CG -t $CORES --n-seq $NSEQ --n-prefill 1024 --n-decode $NDEC --ncmoe 24 --lookahead $OUT/la_g.bin > $OUT/la_g.jsonl 2> $OUT/la_g.err
echo "lookahead rc=$? in $(( $(date +%s) - t0 )) s: $(ls -la $OUT/la_g.bin | awk '{print $5}') bytes"; cat $OUT/la_g.bin.json
tail -n 5 $OUT/la_g.err > $OUT/la_g.err.tail; rm -f $OUT/la_g.err
[ -s $OUT/la_g.bin ] || { echo "no lookahead file: oracle runs impossible"; exit 3; }
# --- the GPU's own compute per token, profiled before any timed run (one problem, 24 decode tokens)
for C in $CA $CB; do
  prof_run G$C $EC_BIN -m $F --corpus $CG -t $CORES --n-prefill 1024 --n-decode 24 --no-mmap --host-experts --seqs 1,0 \
    --ec "slots=$C:policy=dfa:$MB:kappa=1:fetch=$LAWG:paced=0:overlap=1:oracle_w=0:seqs=1/0"
done
python3 - $OUT G$CA G$CB > $OUT/g_prof.json <<'PY'
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
echo "G_prof written $(date -u +%FT%TZ), before any timed run:"; cat $OUT/g_prof.json
run_round() {  # C kappa_dk round process "names": one process, the named configurations in a shuffled order
  local C=$1 kdk=$2 rd=$3 pr=$4 names=$5
  local common="--corpus $CG -t $CORES --n-seq $NSEQ --n-prefill 1024 --n-decode $NDEC --no-mmap --host-experts"
  local base="slots=$C:policy=dfa:$MB:fetch=$LAWG:paced=0:overlap=1:oracle_w=0"
  local orc="slots=$C:policy=oracle:oracle=$OUT/la_g.bin:$MB:kappa=1:fetch=$LAWG:overlap=1:oracle_w=0"
  local tag="C${C}_r${rd}${pr}"
  local cfg="" n c
  for n in $names; do
    case $n in
      base)   c="$base:kappa=1" ;;
      foa)    c="$base:kappa=1:fetch_on_admit=1" ;;
      bypass) c="$orc:oracle_bypass=1:paced=0" ;;
      fetch)  c="$orc:oracle_bypass=1:oracle_fetch=1:paced=0" ;;
      both3p) c="$orc:oracle_bypass=0:oracle_lead=3:oracle_fetch=1:$PACED" ;;
      dk)     c="$base:kappa=$kdk" ;;
      pf)     c="$base:kappa=1:prefetch=1" ;;
      R1)     c="$base:kappa=1:fetch_on_admit=1:prefetch=1" ;;
      R2)     c="$base:kappa=$kdk:fetch_on_admit=1:prefetch=1" ;;
      *)      echo "unknown configuration $n"; return 2 ;;
    esac
    cfg="${cfg:+$cfg;}$c:stats=$OUT/st_g_${tag}_$n.json"
  done
  local off=0; [ "$pr" = B ] && off=7
  local seed=$(( ORDER_SEED + 100 * rd + C + off ))
  echo "cell $tag order-seed $seed configs: $cfg" >> $OUT/configs.txt
  local t0=$(date +%s)
  R 45m $EC_BIN -m $F $common --order-seed $seed --ec "$cfg" > $OUT/ec_g_${tag}.jsonl 2> $OUT/ec_g_${tag}.err
  local rc=$?
  echo "cell $tag rc=$rc in $(( $(date +%s) - t0 )) s"; echo "ec_g_${tag} $rc $(( $(date +%s) - t0 ))" >> $OUT/runs.txt
  grep "\[ec-bench\] config [0-9]*/" $OUT/ec_g_${tag}.err | sed 's/:stats=.*st_/ -> st_/' > $OUT/ec_g_${tag}.order
  grep -i "oracle\|fetch table\|fetch-on-admit\|prefetch\|shuffled\|kappa" $OUT/ec_g_${tag}.err | head -40 > $OUT/ec_g_${tag}.err.oracle
  tail -n 30 $OUT/ec_g_${tag}.err > $OUT/ec_g_${tag}.err.tail; rm -f $OUT/ec_g_${tag}.err
}
vcheck() {  # V1 / V2 / V3 on the rounds run so far
  python3 - $OUT $SMOKE $CA "$@" <<'PY'
import glob, json, os, sys
out, smoke, ca, what = sys.argv[1], sys.argv[2] == "1", sys.argv[3], sys.argv[4]
def by_config(f):
    by = {}
    for l in open(f):
        if l.strip():
            r = json.loads(l)
            lab = os.path.basename(r["config"].rsplit("stats=", 1)[-1]).replace(".json", "").rsplit("_", 1)[-1]
            by.setdefault(lab, []).append(r)
    return by
msg, bad = [], False
files = sorted(glob.glob(f"{out}/ec_g_C*_r*.jsonl"))
if what == "V1":
    for f in files:
        b = by_config(f).get("base", [])
        if not b:
            msg.append(f"V1 {os.path.basename(f)}: no base rows"); bad = True; continue
        nll = sum(r["nll_sum"] / r["nll_n"] for r in b) / len(b)
        ok = smoke or abs(nll / 0.190 - 1) <= 0.02
        bad |= not ok
        msg.append(f"V1 {os.path.basename(f)} base loss {nll:.4f} vs 0.190: {'ok' if ok else 'FAIL'}" + (" (smoke: not checked)" if smoke else ""))
elif what == "V2":
    t = []
    for rd in (1, 2):
        b = by_config(f"{out}/ec_g_C{ca}_r{rd}A.jsonl").get("base", []) if os.path.exists(f"{out}/ec_g_C{ca}_r{rd}A.jsonl") else []
        t.append(sum(r["decode_ms"] / r["n_decode"] for r in b) / len(b) if b else float("nan"))
    v = max(t) / min(t) - 1
    bad = not (v <= 0.02)
    msg.append(f"V2 process A base ms/token {t[0]:.3f} {t[1]:.3f}: spread {100 * v:.2f}%: {'FAIL' if bad else 'ok'}")
else:  # V3: every configuration computes the same model as base in its process; reported, never stops the host
    for f in files:
        by = by_config(f)
        if "base" not in by:
            continue
        b = sum(r["nll_sum"] / r["nll_n"] for r in by["base"]) / len(by["base"])
        for k, v in sorted(by.items()):
            nll = sum(r["nll_sum"] / r["nll_n"] for r in v) / len(v)
            ok = abs(nll / b - 1) <= 0.01
            msg.append(f"V3 {os.path.basename(f)} {k}: {len(v)} problems, loss {nll:.4f} vs base {b:.4f} ({100 * (nll / b - 1):+.2f}%): {'ok' if ok else 'FAIL'}")
print("\n".join(msg))
sys.exit(9 if bad else 0)
PY
}
NA="base foa bypass fetch both3p dk pf"; NB="base R1 R2"
FAILED=""
run_round $CA $KA 1 A "$NA"; run_round $CA $KA 1 B "$NB"
vcheck V1 | tee -a $OUT/validity.txt; [ ${PIPESTATUS[0]} -eq 0 ] || { echo "V1 failed: stopping" | tee -a $OUT/validity.txt; FAILED=V1; }
vcheck V3 | tee -a $OUT/validity.txt
if [ -z "$FAILED" ]; then
  run_round $CA $KA 2 A "$NA"; run_round $CA $KA 2 B "$NB"
  vcheck V2 | tee -a $OUT/validity.txt; [ ${PIPESTATUS[0]} -eq 0 ] || { echo "V2 failed: stopping" | tee -a $OUT/validity.txt; [ "$SMOKE" = 1 ] || FAILED=V2; }
fi
if [ -z "$FAILED" ]; then
  run_round $CB $KB 1 A "$NA"; run_round $CB $KB 1 B "$NB"
  vcheck V1 | tee -a $OUT/validity.txt
  vcheck V3 | tee -a $OUT/validity.txt
fi
rm -f $F $OUT/la_g.bin
python3 - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, os, sys, statistics as st
out = sys.argv[1]; line = []
for f in sorted(os.listdir(out)):
    if not (f.startswith("ec_g_") and f.endswith(".jsonl")):
        continue
    by = {}
    for l in open(f"{out}/{f}"):
        if l.strip():
            r = json.loads(l)
            lab = os.path.basename(r["config"].rsplit("stats=", 1)[-1]).replace(".json", "")
            by.setdefault(lab, {})[r["seq"]] = (r["decode_ms"] / r["n_decode"], r["nll_sum"] / r["nll_n"])
    base = next((v for k, v in by.items() if k.endswith("_base")), None)
    for lab, v in sorted(by.items()):
        ms = st.mean(x[0] for x in v.values()); nll = st.mean(x[1] for x in v.values())
        b = st.mean(base[s][0] for s in v if s in base) if base else float("nan")
        bn = st.mean(base[s][1] for s in v if s in base) if base else float("nan")
        stf = f"{out}/{lab}.json"
        s = json.load(open(stf)) if os.path.exists(stf) else {}
        n = max(1, s.get("steps", 1))
        print(f"{lab:24s} {1000 / ms:7.2f} tok/s {ms:6.2f} ms n={len(v)} vs base {b / ms:.3f} loss {100 * (nll / bn - 1):+.2f}% "
              f"misses {s.get('misses', 0) / n:.2f} admits {s.get('admits', 0) / n:.2f} prefetches {s.get('prefetches', 0) / n:.2f} "
              f"useful {s.get('prefetch_useful', 0) / n:.2f} fetches {s.get('fetches', 0) / n:.2f} "
              f"reads/tok {(s.get('misses', 0) + s.get('admits', 0) + s.get('prefetches', 0)) / n:.2f} "
              f"foa {s.get('fetch_on_admit', '-')} pf {s.get('prefetch_q', '-')} kappa {s.get('kappa', '-')}")
        if "_r1" in lab and not lab.endswith("_base"):
            line.append(f"{lab.replace('st_g_', '')} {b / ms:.3f}")
open(f"{out}/oneline.txt", "w").write("109 " + " ".join(line) + "\n")
PY
ONE=$(cat $OUT/oneline.txt)
python3 $J/manifest.py "$OUT" 109_onlinepolicy@vast "$T0" "$ONE" law_gptoss=$LAWG device_read_gbs=$DR memory_clock_max_mhz=$MEMCLK ratio=$RATIO smoke=$SMOKE
cat $OUT/skipped.txt 2>/dev/null
cat $OUT/validity.txt 2>/dev/null
echo "SUMMARY: $ONE"
