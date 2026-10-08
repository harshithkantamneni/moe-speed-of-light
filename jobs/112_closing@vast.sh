#!/bin/bash
# Job 112: the last measurements of the paper. Four questions, one protocol (job 110's, jobs/110_secondcard@vast.sh, with
# the changes below; engine patch oracle5, gpt-oss-120b, 20 AIME-25 problems x 256 teacher-forced steps, the job 084
# own-text corpus, configurations in fresh contexts in a seeded shuffled order within each process):
#   (a) the RTX 4090 at higher link-to-CPU ratios: job 111's three valid hosts all read 0.37-0.40;
#   (b) the 2 x 2's timing: in the arm "MIN's set, read twice" the engine publishes a background copy at the start of the
#       second step after the step that decided it, so an admission for the next step's use lands late and that arm
#       also misses more (reviews 21a and 21b). New arm bypassplanS holds the timing while keeping both reads;
#   (c) a second probe of each host after its timed runs (the bound's B_host rests on one probe per host);
#   (d) two relaunches of job 109's RTX 5090 machines, which add (b) and (c) on the registered fast-link machines.
# Configurations: job 109's (base foa bypass fetch both3p dk pf R1 R2, defined in jobs/109_onlinepolicy@vast.sh) and
#   fetchplan    MIN's fewest-admission set (jobs/ec2/minadm_plan.py on this host's own routing, as jobs 103-107), each
#                admission fetched by the GPU in the step (one read)
#   bypassplan   the same set served by the CPU, each admission copied in the background after the step that decided it
#                (two reads; the copy is published at the start of the step after next)
#   bypassplanS  bypassplan with every record of the plan moved one step earlier (jobs/ec2/plan_shift.py; a record at a
#                sequence's first step stays): the copy for an expert served at step s is issued after step s - 1,
#                overlaps step s, where the expert is still loading and the CPU serves it (the first read), and is
#                published at s + 1, as MIN with bypass assumes. It uses the routing of step s one step early, nothing
#                beyond what the oracle arms already use.
#   fetchplan and bypassplan ran on this engine in job 107; bypassplanS differs from bypassplan only in its plan file.
#   No smoke run: every configuration has run on patch oracle5 (jobs 107 and 109-111).
# Two modes, set by the wrapper (CARD):
#   CARD=4090  new RTX 4090 machines. Process A = base foa bypass fetch both3p dk pf fetchplan bypassplan bypassplanS,
#              process B = base R1 R2; C = 14 twice, then C = 32 once; this machine's routing (lookahead, untimed) and
#              Nsight profiles of base before any timed run, as job 110. Budget 210 minutes.
#   CARD=5090  relaunches of job 109's machines. Process A = base foa bypass fetch fetchplan bypassplan bypassplanS, no
#              process B; C = 14 twice, then C = 32 once; lookahead before any timed run; no profiles. Budget 105 minutes.
# The second probe: after the last round (the model deleted, no process running), the probe again (concur2.txt), read
# as the first (fetch_table.py; B_host2 by scripts/reanalysis.py's rule). The first probe ran while the model downloaded.
# VALIDITY, registered here (a host that fails stops and is replaced by the next offer of its card below):
#   CARD=4090: V0 the GPU's UUID in no earlier launch (jobs/ec2/known_gpu_uuids.txt) except job 091's machine
#     (GPU-406c0b8f-8507-0127-d730-9055591f8cca, offer 49588631, which ran job 091's grid and nothing of jobs 109-111),
#     at most 48 GB of host memory in use; V0c as job 110 (one NUMA node, at most 32 usable physical cores, "4090" in the
#     name, 23,000-25,000 MiB, device reads >= 800 GB/s, disk >= 100 GB); VR ratio = B_p / B_c at least 0.45 (above
#     job 111's 0.37-0.40); V1 base's loss within 2% of 0.1963 (job 111's reference for this card); V2 and V3 as job 109.
#   CARD=5090: V0 the GPU's UUID one of job 109's five valid machines (jobs/ec2/job109_uuids.txt), at most 48 GB in use;
#     V0c as job 109 (one NUMA node, at most 32 usable physical cores, "5090" in the name, device reads >= 1500 GB/s,
#     disk >= 100 GB); VR ratio at least 0.5 (job 109's line); V1 within 2% of 0.190; V2 and V3 as job 109.
#   A host is valid if it passes V0, V0c, VR, V1 and V2. A configuration failing V3 on a host is left out there. If a
#   host's plan cannot be computed, its three plan arms are dropped (recorded) and the host continues.
# Definitions as job 109 (X/base: base's mean time per token over the problems / X's, same process, geometric mean over
# rounds; misses: the engine's counter per decode step, mean over rounds). Earlier data, for each prediction's band:
# job 107's RTX 5090 hosts at gpt-oss 11% (ratios 0.44-0.75) read bypassplan/base 1.03-1.13 and fetchplan/base
# 1.16-1.43; bypassplan missed 50.5-55.6 per token, fetchplan 40.7-42.1. A replay of the engine's publication rule
# (a scratch replay on job 063's gpt-oss-120b trace, another text; not part of this job) gives bypassplanS 0.91x
# bypassplan's misses at C = 14 and at C = 32 (3.9 and 1.7 fewer per token), far less than the engine's 10-15: we
# expect the rest to come from the fetch table's in-step fetches, which every arm keeps (the counters will show it).
# Predictions, committed before launch:
#   On this job's valid RTX 4090 hosts (job 110's predictions 1-6, the frozen RTX 5090 trend of its header):
#   Q1. |ln(X/base) - trend(r)| <= 0.10 for fetch and both3p at 11%; at 25% on all but at most one host-configuration
#       of this job; and <= 0.06 for foa and bypass at 11%;
#   Q2. pf/base and R1/base below 1.00 at 11% on every host with ratio below 0.75;
#   Q3. the best of dk, pf, R1, R2 vs base at most 1.10 at both budgets;
#   Q4. base / Eq. (1) within [1.8, 3.5] at 11% and [2.5, 6.5] at 25%;
#   Q5. R1 and R2 read at least 1.3 R* host experts per token at both budgets;
#   Q6. on hosts with ratio at least 0.5: the interaction (job 109's definition) positive at 11%.
#   On every valid host of either card:
#   T1. counters: bypassplanS's misses per token at most 0.95x bypassplan's, at both budgets (the late landing removed);
#   T2. bypassplanS/base at least bypassplan/base - 0.01, at both budgets;
#   T3. at 11%, on hosts with ratio at least 0.5: fetchplan/base - bypassplanS/base > bypassplanS/base - bypassplan/base
#       (the second read costs more than the late landing);
#   T4. at 11%, on hosts with ratio at least 0.5: bypassplanS/base at most 1.10 (MIN's set read twice gains little even
#       when its copies land on time);
#   T5. the second probe: |B_host2 / B_host1 - 1| <= 0.10.
#   On the RTX 5090 relaunches:
#   T6. process A's base time per token at C = 14 within 5% of the same machine's in job 109 (mean over rounds);
#   T7. job 109's prediction 1 again at 11%: fetch/base >= max(bypass/base, foa/base) + 0.04.
# Q1-Q6 are tested per host (none tested with no valid RTX 4090 host); T1-T5 need at least three valid hosts over the two
# cards, else they are inconclusive. Every host that starts is reported.
# Hosts, from these offers in order (offers no longer listed are skipped; none in the rental ledger except as stated):
#   CARD=4090: 49588631 (Core i5-12400, job 091's machine), 52088382 (Ryzen 7 7800X3D), then 53783561 (Ryzen 9 8945HX),
#     54729181 (Core i7-10700KF);
#   CARD=5090: 51325952 (job 109d, Core i9-13900KF, ratio 0.59), 54227874 (job 109c, Ryzen 9 7945HX, 0.89), then
#     54573923 (job 109f, Ryzen 9 5950X, 0.84).
# The first two of each list start together; a host that fails a gate is replaced by the next offer of its card, until
# two hosts of the card are valid, three have started, or the credit left is below the running hosts' needs plus $2.
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
SMOKE=${SMOKE:-0}
CARD=${CARD:?CARD must be 4090 or 5090}
MREPO=ggml-org/gpt-oss-120b-GGUF; MFILE=gpt-oss-120b-MXFP4.gguf; CA=14; CB=32; NSEQ=20; NDEC=256
case $CARD in
  4090) BMIN=210; NA="base foa bypass fetch both3p dk pf fetchplan bypassplan bypassplanS"; NB="base R1 R2"; PROF=1
        V1REF=0.1963; DRMIN=800; VRMIN=0.45 ;;
  5090) BMIN=105; NA="base foa bypass fetch fetchplan bypassplan bypassplanS"; NB=""; PROF=0
        V1REF=0.190; DRMIN=1500; VRMIN=0.5 ;;
  *) echo "unknown CARD $CARD"; exit 2 ;;
esac
KA=3; KB=2
echo "CARD=$CARD SMOKE=$SMOKE model $MFILE budgets C=$CA,$CB problems $NSEQ steps $NDEC A=[$NA] B=[$NB]" | tee $OUT/mode.txt
JD="$(cd "$(dirname "$0")" && pwd)"
gate_fail() {  # exit-code message: a failed gate stops the host
  local code=$1; shift
  echo "$*" | tee -a $OUT/v0.txt
  if [ "$SMOKE" = 1 ]; then echo "(smoke run: recorded, not enforced)" | tee -a $OUT/v0.txt; return 0; fi
  exit $code
}
# --- V0, before anything is installed or downloaded
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1 | tr -d ' ')
free -g > $OUT/free.txt; USEDGB=$(awk '/Mem:/{print $3}' $OUT/free.txt)
echo "GPU UUID $UUID; host memory in use ${USEDGB} GB" | tee $OUT/v0.txt
if [ "$CARD" = 4090 ]; then
  if grep -qx "$UUID" "$JD/ec2/known_gpu_uuids.txt" && [ "$UUID" != GPU-406c0b8f-8507-0127-d730-9055591f8cca ]; then
    gate_fail 8 "V0 FAIL: GPU rented before ($UUID), stopping"
  fi
else
  grep -qx "$UUID" "$JD/ec2/job109_uuids.txt" || gate_fail 8 "V0 FAIL: not one of job 109's RTX 5090 machines ($UUID), stopping"
fi
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
GNAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); GMEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "card ${GNAME}, ${GMEM} MiB" | tee -a $OUT/gate.txt
case "$GNAME" in *$CARD*) ;; *) gate_fail 4 "V0c FAIL: not an RTX $CARD (${GNAME}), stopping";; esac
if [ "$CARD" = 4090 ]; then
  [ "${GMEM:-0}" -ge 23000 ] && [ "${GMEM:-0}" -le 25000 ] || gate_fail 4 "V0c FAIL: ${GMEM} MiB of device memory (not 24 GB), stopping"
fi
[ "${DR:-0}" -ge $DRMIN ] || gate_fail 5 "V0c FAIL: GPU reads below $DRMIN GB/s: throttled card, stopping"
RAMGB=$(awk '/Mem:/{print $2}' $OUT/free.txt); echo "host RAM ${RAMGB} GB, in use ${USEDGB} GB" | tee -a $OUT/gate.txt
DISKGB=$(df -BG --output=avail $WORK | tail -1 | tr -dc 0-9); echo "disk free ${DISKGB} GB" | tee -a $OUT/gate.txt
[ "${DISKGB:-0}" -ge 100 ] || gate_fail 6 "V0c FAIL: not enough disk, stopping"
echo "V0c checked" | tee -a $OUT/v0.txt
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip python3-venv >> $OUT/apt.txt 2>&1
python3 -m pip install -q --break-system-packages huggingface_hub hf_xet numpy scipy > $OUT/pip.txt 2>&1
# --- the plan's environment (numpy and scipy; HiGHS through scipy), as job 107
PYP=python3
if ! python3 -I -c "import numpy, scipy.optimize" 2>/dev/null; then
  python3 -m pip install -q --break-system-packages uv >> $OUT/pip.txt 2>&1
  PV=$WORK/planenv
  for i in 1 2 3; do
    ( uv venv -q --clear $PV && uv pip install -q --python $PV/bin/python numpy scipy ) >> $OUT/planenv.txt 2>&1
    $PV/bin/python -I -c "import scipy.optimize" 2>/dev/null && break
  done
  PYP=$PV/bin/python
fi
$PYP -I -c "import numpy, scipy; print('plan env: numpy', numpy.__version__, 'scipy', scipy.__version__)" 2>&1 | tee -a $OUT/planenv.txt
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
if [ "$PROF" = 1 ]; then install_nsys & NSP=$!; fi
build_tree $WORK/lc-ec "$PATCH" llama-ec-bench
if [ "$PROF" = 1 ]; then wait $NSP; install_nsys; fi
EC_BIN=$WORK/lc-ec/build/bin/llama-ec-bench; ls -la $EC_BIN || { echo "no ec-bench binary"; exit 3; }
sha256sum $PATCH $J/minadm_plan.py $J/plan_shift.py $J/known_gpu_uuids.txt $J/job109_uuids.txt | tee $OUT/patch_sha.txt; echo "base $BASE" >> $OUT/patch_sha.txt
# --- the host (idle but for the download): platform, our probe, the fetch table, the ratio
platform
H=$(( CORES > 4 ? CORES - 2 : 2 ))
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
grep -q "concurrent t=" $OUT/concur.txt || { echo "concur incomplete, retrying"; R 15m $WORK/concur > $OUT/concur.txt 2>&1; }
echo "probe 1 ended $(date -u +%FT%TZ); download running: $(kill -0 $DLG 2>/dev/null && echo yes || echo no)" | tee $OUT/probe1_when.txt
LAWG=$(python3 $J/fetch_table.py $OUT/concur.txt $H 13253760 4 $OUT/fetch_table_law_gptoss.json 37)
echo "law table ($(date -u +%FT%TZ)): gpt-oss $LAWG" | tee $OUT/tables.txt
[ -n "$LAWG" ] || { echo "no law table"; exit 3; }
RATIO=$(python3 -c "import json, sys; d = json.load(open(sys.argv[1])); print(f'{d[\"B_p\"] / d[\"B_c\"]:.4f}')" $OUT/fetch_table_law_gptoss.json)
echo "VR: B_p / B_c = ${RATIO:-?}" | tee -a $OUT/gate.txt
if ! python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)" "${RATIO:-0}" $VRMIN; then
  if [ "$SMOKE" != 1 ]; then pkill -P $DLG; kill $DLG; fi
  gate_fail 10 "VR FAIL: link/CPU ratio ${RATIO} below $VRMIN, stopping"
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
# --- this machine's routing of the text (untimed), for the oracle configurations, and the plans
t0=$(date +%s)
R 30m $EC_BIN -m $F --corpus $CG -t $CORES --n-seq $NSEQ --n-prefill 1024 --n-decode $NDEC --ncmoe 24 --lookahead $OUT/la_g.bin > $OUT/la_g.jsonl 2> $OUT/la_g.err
echo "lookahead rc=$? in $(( $(date +%s) - t0 )) s: $(ls -la $OUT/la_g.bin | awk '{print $5}') bytes"; cat $OUT/la_g.bin.json
tail -n 5 $OUT/la_g.err > $OUT/la_g.err.tail; rm -f $OUT/la_g.err
[ -s $OUT/la_g.bin ] || { echo "no lookahead file: oracle runs impossible"; exit 3; }
PLANS_OK=1
for C in $CA $CB; do
  t0=$(date +%s)
  ( cd $J && R 20m $PYP -I minadm_plan.py $OUT/la_g.bin $C $OUT/plan_g$C.bin $CORES ) > $OUT/plan_g$C.txt 2>&1
  echo "plan C$C rc=$? in $(( $(date +%s) - t0 )) s"; tail -c 700 $OUT/plan_g$C.txt
  ( cd $J && $PYP -I plan_shift.py $OUT/la_g.bin $OUT/plan_g$C.bin $OUT/plan_g${C}S.bin ) > $OUT/plan_g${C}S.txt 2>&1
  echo "shifted plan C$C rc=$?"; cat $OUT/plan_g${C}S.txt
  [ -s $OUT/plan_g$C.bin ] && [ -s $OUT/plan_g${C}S.bin ] || PLANS_OK=0
done
if [ "$PLANS_OK" != 1 ]; then
  echo "NO PLAN: fetchplan, bypassplan and bypassplanS dropped on this host" | tee -a $OUT/validity.txt
  NA=$(echo $NA | tr ' ' '\n' | grep -v plan | tr '\n' ' ')
fi
sha256sum $OUT/plan_g*.bin > $OUT/plan_sha.txt 2>/dev/null
# --- the GPU's own compute per token, profiled before any timed run (one problem, 24 decode tokens; RTX 4090 only)
if [ "$PROF" = 1 ]; then
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
fi
run_round() {  # C kappa_dk round process "names": one process, the named configurations in a shuffled order
  local C=$1 kdk=$2 rd=$3 pr=$4 names=$5
  local common="--corpus $CG -t $CORES --n-seq $NSEQ --n-prefill 1024 --n-decode $NDEC --no-mmap --host-experts"
  local base="slots=$C:policy=dfa:$MB:fetch=$LAWG:paced=0:overlap=1:oracle_w=0"
  local orc="slots=$C:policy=oracle:oracle=$OUT/la_g.bin:$MB:kappa=1:fetch=$LAWG:overlap=1:oracle_w=0"
  local tag="C${C}_r${rd}${pr}"
  local cfg="" n c
  for n in $names; do
    case $n in
      base)        c="$base:kappa=1" ;;
      foa)         c="$base:kappa=1:fetch_on_admit=1" ;;
      bypass)      c="$orc:oracle_bypass=1:paced=0" ;;
      fetch)       c="$orc:oracle_bypass=1:oracle_fetch=1:paced=0" ;;
      both3p)      c="$orc:oracle_bypass=0:oracle_lead=3:oracle_fetch=1:$PACED" ;;
      dk)          c="$base:kappa=$kdk" ;;
      pf)          c="$base:kappa=1:prefetch=1" ;;
      R1)          c="$base:kappa=1:fetch_on_admit=1:prefetch=1" ;;
      R2)          c="$base:kappa=$kdk:fetch_on_admit=1:prefetch=1" ;;
      fetchplan)   c="$orc:oracle_bypass=1:oracle_fetch=1:paced=0:oracle_plan=$OUT/plan_g$C.bin" ;;
      bypassplan)  c="$orc:oracle_bypass=1:paced=0:oracle_plan=$OUT/plan_g$C.bin" ;;
      bypassplanS) c="$orc:oracle_bypass=1:paced=0:oracle_plan=$OUT/plan_g${C}S.bin" ;;
      *)           echo "unknown configuration $n"; return 2 ;;
    esac
    cfg="${cfg:+$cfg;}$c:stats=$OUT/st_g_${tag}_$n.json"
  done
  local off=0; [ "$pr" = B ] && off=7
  local seed=$(( ORDER_SEED + 100 * rd + C + off ))
  echo "cell $tag order-seed $seed configs: $cfg" >> $OUT/configs.txt
  local t0=$(date +%s)
  R 60m $EC_BIN -m $F $common --order-seed $seed --ec "$cfg" > $OUT/ec_g_${tag}.jsonl 2> $OUT/ec_g_${tag}.err
  local rc=$?
  echo "cell $tag rc=$rc in $(( $(date +%s) - t0 )) s"; echo "ec_g_${tag} $rc $(( $(date +%s) - t0 ))" >> $OUT/runs.txt
  grep "\[ec-bench\] config [0-9]*/" $OUT/ec_g_${tag}.err | sed 's/:stats=.*st_/ -> st_/' > $OUT/ec_g_${tag}.order
  grep -i "oracle\|plan\|fetch table\|fetch-on-admit\|prefetch\|shuffled\|kappa" $OUT/ec_g_${tag}.err | head -60 > $OUT/ec_g_${tag}.err.oracle
  tail -n 30 $OUT/ec_g_${tag}.err > $OUT/ec_g_${tag}.err.tail; rm -f $OUT/ec_g_${tag}.err
}
vcheck() {  # V1 / V2 / V3 on the rounds run so far
  python3 - $OUT $SMOKE $CA $V1REF "$@" <<'PY'
import glob, json, os, sys
out, smoke, ca, ref, what = sys.argv[1], sys.argv[2] == "1", sys.argv[3], float(sys.argv[4]), sys.argv[5]
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
        ok = smoke or abs(nll / ref - 1) <= 0.02
        bad |= not ok
        msg.append(f"V1 {os.path.basename(f)} base loss {nll:.4f} vs {ref}: {'ok' if ok else 'FAIL'}" + (" (smoke: not checked)" if smoke else ""))
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
round_pair() {  # C kappa round: process A, then process B if this mode has one
  run_round $1 $2 $3 A "$NA"
  [ -n "$NB" ] && run_round $1 $2 $3 B "$NB"
  return 0
}
FAILED=""
round_pair $CA $KA 1
vcheck V1 | tee -a $OUT/validity.txt; [ ${PIPESTATUS[0]} -eq 0 ] || { echo "V1 failed: stopping" | tee -a $OUT/validity.txt; FAILED=V1; }
vcheck V3 | tee -a $OUT/validity.txt
if [ -z "$FAILED" ]; then
  round_pair $CA $KA 2
  vcheck V2 | tee -a $OUT/validity.txt; [ ${PIPESTATUS[0]} -eq 0 ] || { echo "V2 failed: stopping" | tee -a $OUT/validity.txt; [ "$SMOKE" = 1 ] || FAILED=V2; }
fi
if [ -z "$FAILED" ]; then
  round_pair $CB $KB 1
  vcheck V1 | tee -a $OUT/validity.txt
  vcheck V3 | tee -a $OUT/validity.txt
fi
rm -f $F $OUT/la_g.bin $OUT/plan_g*.bin
# --- the second probe: the host idle, after the timed runs
if [ -z "$FAILED" ]; then
  sync; sleep 20
  R 15m $WORK/concur > $OUT/concur2.txt 2>&1
  grep -q "concurrent t=" $OUT/concur2.txt || { echo "concur2 incomplete, retrying"; R 15m $WORK/concur > $OUT/concur2.txt 2>&1; }
  echo "probe 2 ended $(date -u +%FT%TZ)" | tee $OUT/probe2_when.txt
  LAWG2=$(python3 $J/fetch_table.py $OUT/concur2.txt $H 13253760 4 $OUT/fetch_table2_law_gptoss.json 37)
  RATIO2=$(python3 -c "import json, sys; d = json.load(open(sys.argv[1])); print(f'{d[\"B_p\"] / d[\"B_c\"]:.4f}')" $OUT/fetch_table2_law_gptoss.json 2>/dev/null)
  echo "second probe: law table $LAWG2; B_p / B_c = ${RATIO2:-?} (first ${RATIO})" | tee -a $OUT/tables.txt
fi
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
              f"plan_already {s.get('plan_already', '-')} refused {s.get('oracle_refused', '-')} kappa {s.get('kappa', '-')}")
        if "_r1" in lab and not lab.endswith("_base"):
            line.append(f"{lab.replace('st_g_', '')} {b / ms:.3f}")
open(f"{out}/oneline.txt", "w").write("112 " + " ".join(line) + "\n")
PY
ONE=$(cat $OUT/oneline.txt)
python3 $J/manifest.py "$OUT" "112_closing@vast" "$T0" "$ONE" card=$CARD law_gptoss=$LAWG law_gptoss2=$LAWG2 device_read_gbs=$DR memory_clock_max_mhz=$MEMCLK ratio=$RATIO ratio2=$RATIO2 smoke=$SMOKE
cat $OUT/skipped.txt 2>/dev/null
cat $OUT/validity.txt 2>/dev/null
echo "SUMMARY: $ONE"
