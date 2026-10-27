#!/bin/bash
# Job 124: job 123's registered test, unchanged, on the population job 123's scope excluded (paper 2; the design and
# every number below: prereg/p2/job124_design.md, computed by scripts/p2/job124_design.py on main). The question: on
# slow-link consumer desktops (r = B_p / B_c below 0.6), where the model's one known systematic failure lies (on 7
# earlier slow-link Intel desktops Eq. (1) put the deployed cache's in-step fetches 7% ahead; they were 4% behind
# (medians over the machines) and lost on 13 of their 14 cells), does job 123's frozen model still predict the decode
# time of the ten configurations, and does it still get the fetch decision wrong? It can fail: one clause (T4) is at
# risk and one (F3) is predicted to fail.
# Population (exact): single-GPU consumer desktops (Ryzen 3/5/7/9 but not Threadripper, Core i3/i5/i7/i9, Core Ultra;
# one NUMA node, at most 32 usable physical cores) with an RTX 5090 and an Intel CPU, or an RTX 4090 and an Intel or
# AMD CPU, whose first probe gives r = B_p / B_c below 0.6 (B_c the CPU read at H helpers, B_p the 16 MB x 64-block
# zero-copy read, as jobs/ec2/fetch_table.py reads concur.txt). Earlier machines below 0.6: Intel RTX 5090 15 of 22,
# Intel RTX 4090 5 of 6, AMD RTX 4090 4 of 5; AMD RTX 5090 desktops are left out (3 of 44).
# Protocol: job 123's (jobs/123_confirm@vast.sh), unchanged: gpt-oss-120b MXFP4; the job 084 own-text corpus, 20
# AIME-25 problems x 256 teacher-forced steps, fresh context per configuration; engine patch oracle7; this machine's
# routing recorded first (untimed) and the Few plan from it (jobs/ec2/minadm_plan.py) at C = 14 and at C = 32; C = 14
# for three rounds, each one process with the ten arms in a seeded shuffled order, then C = 32 for one round; the
# fixed fetch table 0,0,1,2,3 for every arm with in-step fetches; the arms baselaw base lru aa hitopt fetch fetchplan
# both3p bypass0 base0 exactly as job 123's header defines them (H helpers = usable physical cores - 2). Definitions
# (T, counted rounds, counters per token, hit rate, cell, error) as job 123's header.
# What changes, and nothing else: the population (VR inverted; V0c's card and CPU gates); on an RTX 4090 its own V1
# reference loss (0.1963, job 111's) and device-read floor (800 GB/s, job 110's) and its own G; the host budget (135
# minutes, for the slower rounds of slow links); C0's frozen counters (every valid run before this job); the clauses
# after T6.
# The frozen model: job 123's, unchanged (T_pred = max(G + sum over layer-steps of max(f s_p, c s_c, [0 < f < n] n
# s_b), root(G + X_in s_p, A s_p), root(G + (X_c + X_in) s_b, A s_b)), S = 13,253,760 bytes, the split of a
# layer-step's misses and B_c, B_p, B_both from this machine's first probe as its header states). G = 4.413 ms on an
# RTX 5090 (job 123's constant); on an RTX 4090 G = 5.32896 ms, the same definition (the median over machines of the
# Nsight GPU time of gpt-oss-120b, 13 RTX 4090s; paper 2's G table). No constant is fitted to time.
# The scorer: jobs/124/scorer.py on this branch, committed with this header, sha256
# a642e0ebc6b469b252ca60cba943d49c817c22c4dfc4a20996d32e8ed67ddabf. Standard library only; it reads the results
# directory and nothing else, and holds the constants, the thresholds and C0's frozen counters. Self-test, run before
# this commit: pointed at job 123's results with --job 123, it reproduces job 123's rebuilt scorecard (main:
# prereg/p2/scorecard_123.json) in all 8,348 values (8,347 exact, 1 within 1.3e-15), every verdict and the decision
# (confirmed), and job 123's recorded numbers (median |error| 1.7%, 90 cells, T6 40 of 40 pairs);
# prereg/p2/job124_selftest.md on main.
# Clauses (on the valid machines; "slower": larger T, point values; every threshold fixed here, before any host):
#   C0.  counters are not timing: base, aa, fetch, fetchplan, base0 and bypass0 (and baselaw when its table has frozen
#        counters) recur on every valid machine, host reads per token within 1% of the frozen value (the median over
#        the valid runs before this job of the identical configuration, budget, model and text; in the scorer); hitopt
#        and lru agree across the valid machines, max/min - 1 at most 1% at each budget;
#   C0p. (reported, not in the decision) the paced arm both3p within 5% of its frozen value (job 123's tolerance;
#        paced copies issued ahead depend on timing, and on earlier slow-link hosts read up to 4.6% from their
#        configuration's median);
#   T1.  the median |error| pooled over every cell at most 7%;
#   T2.  on every valid machine, the median |error| over its cells at most 11%;
#   T3.  at least 70% of the cells within 10%;
#   T4.  on every machine x budget with at least six counted arms, Spearman correlation of T and T_pred at least 0.6;
#   T5.  of the pairs of arms on one machine and budget whose T differ by more than 1%, at least 85% ordered as
#        T_pred;
#   T6.  reads, not hits: of those pairs on which the hit rate (more than 0.005 apart; higher = faster) and T_pred
#        predict opposite orders and T_pred's gap is at least 10%, at least 85% ordered as T_pred (untested below five
#        pairs);
#   K1.  two paths in the step: of the cells of fetch and fetchplan (each miss either copied over the link or computed
#        on the CPU in the step), at least 70% within 10% (untested on fewer than five cells);
#   K2.  the per-token two-path form (G + max(X_in s_p, X_c s_c, [X_c, X_in > 0] (X_c + X_in) s_b) with the same two
#        overlap roots), registered as the competitor: fewer than 70% of the same cells within 10%, i.e. it fails K1
#        (untested on fewer than five cells);
#   H1.  hits order online pairs no better than the model: of the pairs of online arms (baselaw base lru aa base0) on
#        one machine and budget whose T differ by more than 1%, the share the hit rate orders right (higher = faster)
#        is at most the share T_pred orders right (untested on fewer than ten pairs);
#   F1.  the fetch sign: on the Intel machines, base0 faster than base (the fixed table's in-step fetches lose) in at
#        least two thirds of the machine x budget cells (untested on fewer than three cells);
#   F2.  the deployed table: on the Intel machines, base0 faster than baselaw in at least two thirds of the cells
#        where the machine's own law table is not all zero (untested on fewer than three cells);
#   F3.  the model's fetch decision: Eq. (1)'s sign of the fetches (T_pred of base or baselaw against T_pred(base0))
#        right on at least 80% of the F1 and F2 cells (untested on fewer than three cells);
#   F4.  (exploratory) the same sign on the AMD machines' cells;
#   P1-P4 (secondary, job 123's clauses as its header states them): hitopt slower than both3p at C = 14 and 32 and
#        than fetch at C = 14; aa slower than base at C = 14 where T_pred(aa) / T_pred(base) is at least 1.10; lru
#        slower than base; T_pred's sign of hitopt against base right on at least 80% of the cells.
# Decision: the frozen model holds on the slow-link population if C0, T1-T6 and K1 hold on at least three valid
# machines; refuted (a failure of the frozen model, reported as such, with nothing refitted) if any of them fails;
# inconclusive if none fails but one is untested; untested with fewer than three valid machines. The boundary (the
# model's fetch decision is wrong on slow-link Intel desktops) is confirmed if F1 or F2 holds and F3 fails, not
# confirmed if F3 holds. C0p, K2, H1, F4 and P1-P4 are reported with their own verdicts. Every host that starts is
# reported.
# Expected (prereg/p2/job124_design.md; earlier slow-link evidence: 400 runs on 19 population machines, 334
# configurations; shares of every 3-, 4- and 5-machine subset that pass): T1, T2, T3 and T5 hold (median |error| 3.6%,
# worst machine 8.7%, 91.9% within 10%, 93.6% of pairs ordered; subsets 0.99-1.00); T4 is at risk and the clause most
# likely to fail (lowest earlier cell 0.595; 0.84 / 0.79 / 0.74 of subsets), because the fetch-sign failure inverts
# fetch-on and fetch-off arms that differ by a few percent; T6 holds (98.6% of 370 earlier pairs); K1 holds (96.3% of
# 54 earlier two-path cells) and K2 holds (the per-token form 25.9%; no subset reaches 70%; on job 123's fast-link
# cells it had 77.8%); H1 holds (hit rate 10.4% vs Eq. (1) 87.2% of 297 earlier online pairs); C0 holds; C0p is at
# risk; F1 and F2 hold (the law-table fetches lost on 13 of 14 earlier Intel cells, and the fixed table fetches more);
# F3 fails (Eq. (1) was right on 1 of those 14; with job 123's counters on the earlier Intel machines' probes it puts
# the fixed table behind on 18 of 36 cells and the law table behind on 0 of 34); P1-P4 hold (projected hitopt / both3p
# at least 1.48, aa / base at least 1.39 at C = 14). T1-T5 together: 0.83 / 0.79 / 0.74 of subsets on 3 / 4 / 5
# machines; the per-token form 0.20 / 0.15 / 0.11.
# VALIDITY: V0 the GPU's UUID in no earlier launch (jobs/ec2/known_gpu_uuids124.txt: job 123's list and every UUID in
# results/ up to job 123), at most 48 GB of host memory in use; V0c one NUMA node, at most 32 usable physical cores, a
# consumer desktop CPU (as above; no X-series Core i7/i9), and the population's card: an RTX 5090 with an Intel CPU
# (device reads >= 1500 GB/s) or an RTX 4090 (23,000-25,000 MiB, device reads >= 800 GB/s), at least 90 GB of host
# memory, at least 150 GB of free disk; VR the first probe's ratio B_p / B_c below 0.6 (job 123's VR, inverted); V1
# base's loss within 2% of 0.190 (RTX 5090) or 0.1963 (RTX 4090) in every round where base completed; V2 base's
# complete C = 14 rounds (at least two) within 2% of each other; V3 every arm's loss within 1% of base's in its round
# (an arm-round failing it does not count). A host is valid if it passes V0, V0c, VR, V1 and V2; a GPU that already
# ran in this job does not count again. A host stops at its first failed V0, V0c, VR, V1 or V2. If the plan cannot be
# computed, fetchplan is dropped on that host (recorded; C0 does not count it missing).
# Hosts: single-GPU, whole-machine on-demand offers of the population's cards (an RTX 5090 with an Intel consumer
# desktop CPU at most $0.90/h, an RTX 4090 with a consumer desktop CPU at most $0.80/h), at most $0.010/GB down, at
# least 90 GB of host memory, 160 GB of disk and 300 Mb/s down, on a machine never rented before (no offer of it in
# the rental ledger or in a committed listing of an earlier launch; V0 checks the GPU itself), in order of expected
# cost, $/h x (1.6 h + the hours to download 65 GB at the listed speed), from a listing committed and pushed before
# each launch (jobs/ec2/list124.py). At most two started hosts that did not fail a gate per Vast host; up to 3 run at
# once; each launched with JOB_TIMEOUT=165m and destroyed by the controller when its results are fetched, and at the
# latest 3.0 hours after launch. Budget per host: 135 minutes from the end of the V0 check and the package setup (T0);
# a step that would run past it less five minutes is cut there or skipped (a host whose C = 32 round is skipped counts
# with its C = 14 cells). Host count: at least 3 valid hosts for a decision; the job stops starting hosts when 5 are
# valid, 12 have started, the spend rule below would be broken, or 72 hours after this header was pushed, whichever
# comes first.
# Cost (gpu/vast.py's rule: $/h x hours + GB downloaded x $/GB; prices from the rental ledger, no listing call): the
# ledger's 36 rentals of these card classes (jobs 058-115) cost $0.62/h median (at most $0.87), downloads $0.0047/GB;
# with the earlier outcomes (r below 0.6 on 73% of these classes' machines; 63% of in-scope starts of jobs 119-123
# valid) a valid host costs about $1.42, a VR stop $0.28, another failure $0.58, the smoke run $0.22: about 10.8
# starts and $9.84 expected for 5 valid hosts ($9.63 at the median price of jobs 116-122's listings), $10.16 under the
# $20.00 cap.
# Spend rule (all-in cap $20.00): a rental of this job (the smoke run included) starts only while this job's spend in
# the ledger (every rental named 124*) plus a $3.40 hold for each of its running rentals, the new one included, stays
# at or below $18.00. $3.40 bounds one rental ($0.90/h x 3.0 h + 65 GB x $0.010/GB = $3.35), so the job cannot spend
# more than $18.00, $2.00 under the cap. The controller's VAST_CAP is the ledger's total at the first launch plus the
# cap. A host starts only while the account's credit, less $3.40 for each running rental of this job and the new one,
# stays at or above $1.50.
# A smoke run (SMOKE=1: gpt-oss-20b, 2 problems x 32 steps, budgets 14 and 24, gates recorded, not enforced;
# jobs/124s_slowlink@vast.sh) checks the script on one host of the population before the first host (the card and CPU
# gates, the RTX 4090 path, the inverted VR, lru in ec-bench); it is not a host of this job and is reported.
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
SMOKE=${SMOKE:-0}
MREPO=ggml-org/gpt-oss-120b-GGUF; MFILE=gpt-oss-120b-MXFP4.gguf; NL=36
CA=14; CB=32; NSEQ=20; NDEC=256; ROUNDS="1 2 3"; TAB=0,0,1,2,3
BMIN=135; RAMMIN=90; DISKMIN=150; VRMAX=0.6; V1REF=0.190; DRMIN=1500   # V1REF and DRMIN: the RTX 5090's; an RTX 4090 sets its own in V0c
ARMS="baselaw base lru aa hitopt fetch fetchplan both3p bypass0 base0"
if [ "$SMOKE" = 1 ]; then
  MREPO=ggml-org/gpt-oss-20b-GGUF; MFILE=gpt-oss-20b-MXFP4.gguf; NL=24; NSEQ=2; NDEC=32; DISKMIN=60; RAMMIN=24; CB=24
fi
echo "SMOKE=$SMOKE model $MFILE budgets C=$CA (rounds $ROUNDS), C=$CB (one round) problems $NSEQ x $NDEC table $TAB arms=[$ARMS] VRMAX=$VRMAX (r below)" | tee $OUT/mode.txt
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
if grep -qx "$UUID" "$JD/ec2/known_gpu_uuids124.txt"; then
  gate_fail 8 "V0 FAIL: GPU rented before ($UUID), stopping"
else
  echo "V0: a GPU in no earlier launch" | tee -a $OUT/v0.txt
fi
[ "${USEDGB:-999}" -le 48 ] || gate_fail 7 "V0 FAIL: ${USEDGB} GB of host memory in use (> 48), stopping"
echo "V0 checked" | tee -a $OUT/v0.txt
. "$JD/ec2/setup.sh"
export BASE; export -f build_tree platform getmodel
PATCH=$J/llama.cpp-expert-cache-4da6337-oracle7.patch
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
# --- V0c: the class, the card, the memory, the disk (before any download)
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
CPUN=$(lscpu | sed -n 's/^Model name: *//p' | head -1)
CVEND=other
echo "$CPUN" | grep -Eq 'Intel|Core' && CVEND=Intel
echo "$CPUN" | grep -Eq 'AMD|Ryzen' && CVEND=AMD
CONSUMER=0
if ! echo "$CPUN" | grep -Eq 'Threadripper|EPYC|Xeon|i[79]-[0-9]{4,5}XE?( |$)'; then
  echo "$CPUN" | grep -Eq 'Ryzen [3579] |Core\S* ?(i[3579]|Ultra)|\bi[3579]-[0-9]{4,5}' && CONSUMER=1
fi
echo "CPU ${CPUN}: vendor ${CVEND}, consumer desktop part ${CONSUMER}" | tee -a $OUT/gate.txt
[ "$CONSUMER" = 1 ] || gate_fail 11 "V0c FAIL: ${CPUN} is not a consumer desktop CPU (outside the population), stopping"
case "$GNAME" in
  *5090*) CARD=5090; DRMIN=1500; V1REF=0.190
          [ "$CVEND" = Intel ] || gate_fail 11 "V0c FAIL: an RTX 5090 with a ${CVEND} CPU (outside the population: RTX 5090 hosts are Intel), stopping" ;;
  *4090*) CARD=4090; DRMIN=800; V1REF=0.1963
          [ "${GMEM:-0}" -ge 23000 ] && [ "${GMEM:-0}" -le 25000 ] || gate_fail 4 "V0c FAIL: an RTX 4090 with ${GMEM} MiB (not 23,000-25,000), stopping" ;;
  *) CARD=other; gate_fail 4 "V0c FAIL: not an RTX 5090 or RTX 4090 (${GNAME}), stopping" ;;
esac
echo "card class $CARD: device reads at least $DRMIN GB/s, V1 reference loss $V1REF" | tee -a $OUT/gate.txt
[ "${DR:-0}" -ge $DRMIN ] || gate_fail 5 "V0c FAIL: GPU reads below $DRMIN GB/s: throttled card, stopping"
RAMGB=$(awk '/Mem:/{print $2}' $OUT/free.txt); echo "host RAM ${RAMGB} GB, in use ${USEDGB} GB" | tee -a $OUT/gate.txt
[ "${RAMGB:-0}" -ge $RAMMIN ] || gate_fail 6 "V0c FAIL: ${RAMGB} GB of host memory (< $RAMMIN), stopping"
DISKGB=$(df -BG --output=avail $WORK | tail -1 | tr -dc 0-9); echo "disk free ${DISKGB} GB" | tee -a $OUT/gate.txt
[ "${DISKGB:-0}" -ge $DISKMIN ] || gate_fail 6 "V0c FAIL: ${DISKGB} GB of free disk (< $DISKMIN), stopping"
echo "V0c checked" | tee -a $OUT/v0.txt
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip python3-venv >> $OUT/apt.txt 2>&1
python3 -m pip install -q --break-system-packages huggingface_hub hf_xet numpy scipy > $OUT/pip.txt 2>&1
python3 -c "import huggingface_hub" >> $OUT/pip.txt 2>&1 || echo "huggingface_hub missing" >> $OUT/pip.txt
# --- the plan's environment (numpy and scipy; HiGHS through scipy), as job 112: its own uv environment if needed
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
build_tree $WORK/lc-ec "$PATCH" llama-ec-bench
EC_BIN=$WORK/lc-ec/build/bin/llama-ec-bench; ls -la $EC_BIN || { echo "no ec-bench binary"; exit 3; }
sha256sum $PATCH $J/known_gpu_uuids124.txt $J/fetch_table.py $J/minadm_plan.py $J/la_act.py "$JD/124/scorer.py" | tee $OUT/patch_sha.txt; echo "base $BASE" >> $OUT/patch_sha.txt
# --- the host (downloading, otherwise idle): platform, our probe, the law table, the ratio
platform
H=$(( CORES > 4 ? CORES - 2 : 2 ))
echo "helpers H=$H" | tee -a $OUT/gate.txt
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
grep -q "concurrent t=" $OUT/concur.txt || { echo "concur incomplete, retrying"; R 15m $WORK/concur > $OUT/concur.txt 2>&1; }
echo "probe 1 ended $(date -u +%FT%TZ); download running: $(kill -0 $DLG 2>/dev/null && echo yes || echo no)" | tee $OUT/probe1_when.txt
LAWG=$(python3 $J/fetch_table.py $OUT/concur.txt $H 13253760 4 $OUT/fetch_table_law_gptoss.json 37)
echo "law table ($(date -u +%FT%TZ)): gpt-oss $LAWG; fixed table of the arms $TAB" | tee $OUT/tables.txt
[ -n "$LAWG" ] || { echo "no law table"; exit 3; }
RATIO=$(python3 -c "import json, sys; d = json.load(open(sys.argv[1])); print(f'{d[\"B_p\"] / d[\"B_c\"]:.4f}')" $OUT/fetch_table_law_gptoss.json)
echo "VR: B_p / B_c = ${RATIO:-?}" | tee -a $OUT/gate.txt
if ! python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)" "${RATIO:-9}" $VRMAX; then
  if [ "$SMOKE" != 1 ]; then pkill -P $DLG; kill $DLG; fi
  gate_fail 10 "VR FAIL: link/CPU ratio ${RATIO} at or above $VRMAX (outside the population), stopping"
fi
echo "VR checked" | tee -a $OUT/v0.txt
echo "build and probe done, waiting for the download ($(( $(date +%s) - T0 )) s since launch)"
wait $DLG
cat $OUT/dl_gguf.txt
F=$M/$MFILE
[ -s "$F" ] || { echo "GGUF MISSING"; exit 3; }
df -h $WORK | tail -1 > $OUT/df.txt
MB="mailbox=1:tdec=1:helpers=$H"
PACED="paced=1:chunk=16777216:max_pending=64"
CG=$J/aime25_owntext_gptoss_084.jsonl; wc -l $CG
# ---- this machine's routing of the text (untimed), then the Few plans from it
t0=$(date +%s)
R 30m $EC_BIN -m $F --corpus $CG -t $CORES --n-seq $NSEQ --n-prefill 1024 --n-decode $NDEC --ncmoe 24 --lookahead $OUT/la_g.bin > $OUT/la_g.jsonl 2> $OUT/la_g.err
echo "lookahead g rc=$? in $(( $(date +%s) - t0 )) s: $(ls -la $OUT/la_g.bin | awk '{print $5}') bytes"; cat $OUT/la_g.bin.json
tail -n 5 $OUT/la_g.err > $OUT/la_g.err.tail; rm -f $OUT/la_g.err
python3 $J/la_act.py $OUT/la_g.bin $OUT/route_g.npz
[ -s $OUT/la_g.bin ] || { echo "no lookahead file: oracle runs impossible"; exit 3; }
PLANS_OK=1
for C in $CA $CB; do
  t0=$(date +%s)
  ( cd $J && R 20m $PYP -I minadm_plan.py $OUT/la_g.bin $C $OUT/plan_g$C.bin $CORES ) > $OUT/plan_g$C.txt 2>&1
  echo "plan C$C rc=$? in $(( $(date +%s) - t0 )) s"; tail -c 700 $OUT/plan_g$C.txt
  [ -s $OUT/plan_g$C.bin ] || PLANS_OK=0
done
if [ "$PLANS_OK" != 1 ]; then
  echo "NO PLAN: fetchplan dropped on this host" | tee -a $OUT/validity.txt
  ARMS=$(echo $ARMS | tr ' ' '\n' | grep -vx fetchplan | tr '\n' ' ')
fi
sha256sum $OUT/plan_g*.bin > $OUT/plan_sha.txt 2>/dev/null
echo "arms on this host: $ARMS" | tee -a $OUT/mode.txt
run_round() {  # C round: one process, the arms in a seeded shuffled order
  local C=$1 rd=$2 cfg="" n c
  local common="--corpus $CG -t $CORES --n-seq $NSEQ --n-prefill 1024 --n-decode $NDEC --no-mmap --host-experts"
  local dfa="slots=$C:policy=dfa:$MB:paced=0:overlap=1:oracle_w=0:kappa=1"
  local orc="slots=$C:policy=oracle:oracle=$OUT/la_g.bin:$MB:kappa=1:overlap=1:oracle_w=0"
  local tag="g_C${C}_r${rd}"
  for n in $ARMS; do
    case $n in
      baselaw)   c="$dfa:fetch=$LAWG" ;;
      base)      c="$dfa:fetch=$TAB" ;;
      lru)       c="slots=$C:policy=lru:$MB:fetch=$TAB:paced=0:overlap=1:oracle_w=0:kappa=1" ;;
      aa)        c="$dfa:fetch=$TAB:kappa=-1000000000:fetch_on_admit=1" ;;
      hitopt)    c="$orc:fetch=$TAB:oracle_bypass=0:paced=0" ;;
      fetch)     c="$orc:fetch=$TAB:oracle_bypass=1:oracle_fetch=1:paced=0" ;;
      fetchplan) c="$orc:fetch=$TAB:oracle_bypass=1:oracle_fetch=1:paced=0:oracle_plan=$OUT/plan_g$C.bin" ;;
      both3p)    c="$orc:fetch=$TAB:oracle_bypass=0:oracle_lead=3:oracle_fetch=1:$PACED" ;;
      bypass0)   c="$orc:fetch=0:oracle_bypass=1:paced=0" ;;
      base0)     c="$dfa:fetch=0" ;;
      *)         echo "unknown arm $n"; return 2 ;;
    esac
    cfg="${cfg:+$cfg;}$c:stats=$OUT/st_${tag}_$n.json"
  done
  local seed=$(( ORDER_SEED + 100 * rd + C ))
  echo "cell $tag order-seed $seed configs: $cfg" >> $OUT/configs.txt
  local t0=$(date +%s) lim=45m
  [ "$C" = "$CB" ] && lim=30m
  R $lim $EC_BIN -m $F $common --order-seed $seed --ec "$cfg" > $OUT/ec_${tag}.jsonl 2> $OUT/ec_${tag}.err
  local rc=$?
  echo "cell $tag rc=$rc in $(( $(date +%s) - t0 )) s"; echo "ec_${tag} $rc $(( $(date +%s) - t0 ))" >> $OUT/runs.txt
  grep "\[ec-bench\] config [0-9]*/" $OUT/ec_${tag}.err | sed 's/:stats=.*st_/ -> st_/' > $OUT/ec_${tag}.order
  grep -i "oracle\|plan\|fetch table\|fetch-on-admit\|policy\|helpers\|slots (" $OUT/ec_${tag}.err | head -80 > $OUT/ec_${tag}.err.oracle
  tail -n 30 $OUT/ec_${tag}.err > $OUT/ec_${tag}.err.tail; rm -f $OUT/ec_${tag}.err
}
vcheck() {  # what [rounds...]: V1 (every round file) / V2 (base's C = CA rounds) / V3 / LAYERS on the rounds run so far
  python3 - $OUT $SMOKE $CA $V1REF $NL $NSEQ "$@" <<'PY'
import glob, json, os, sys
out, smoke, ca, ref, nl, nseq, what = (sys.argv[1], sys.argv[2] == "1", sys.argv[3], float(sys.argv[4]), int(sys.argv[5]),
                                      int(sys.argv[6]), sys.argv[7])
def by_config(f):   # rows of a round file by arm (a skipped step leaves a non-JSON line: ignored)
    by = {}
    for l in open(f):
        if l.startswith("{"):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            lab = os.path.basename(r["config"].rsplit("stats=", 1)[-1]).replace(".json", "").rsplit("_", 1)[-1]
            by.setdefault(lab, []).append(r)
    return by
msg, bad = [], False
files = sorted(glob.glob(f"{out}/ec_g_C*_r*.jsonl"))
if what == "V1":   # on base's complete rounds (all problems); an incomplete base is reported, not checked
    for f in files:
        b = by_config(f).get("base", [])
        if len(b) < nseq:
            msg.append(f"V1 {os.path.basename(f)}: base has {len(b)} of {nseq} problems, not checked"); continue
        nll = sum(r["nll_sum"] / r["nll_n"] for r in b) / len(b)
        ok = smoke or abs(nll / ref - 1) <= 0.02
        bad |= not ok
        msg.append(f"V1 {os.path.basename(f)} base loss {nll:.4f} vs {ref}: {'ok' if ok else 'FAIL'}" + (" (smoke: not checked)" if smoke else ""))
elif what == "V2":   # base's complete C = CA rounds, at least two
    t = []
    for rd in sys.argv[8:]:
        p = f"{out}/ec_g_C{ca}_r{rd}.jsonl"
        b = by_config(p).get("base", []) if os.path.exists(p) else []
        if len(b) >= nseq:
            t.append(sum(r["decode_ms"] for r in b) / sum(r["n_decode"] for r in b))
    if len(t) < 2:
        bad = True
        msg.append(f"V2 g C{ca}: {len(t)} complete base rounds (fewer than two): FAIL")
    else:
        v = max(t) / min(t) - 1
        bad = not (v <= 0.02)
        msg.append(f"V2 g C{ca} base ms/token {' '.join(f'{x:.3f}' for x in t)}: spread {100 * v:.2f}%: {'FAIL' if bad else 'ok'}")
elif what == "V3":  # every arm computes the same model as base in its round; reported, never stops the host
    for f in files:
        by = by_config(f)
        if "base" not in by:
            continue
        b = sum(r["nll_sum"] / r["nll_n"] for r in by["base"]) / len(by["base"])
        for k, v in sorted(by.items()):
            nll = sum(r["nll_sum"] / r["nll_n"] for r in v) / len(v)
            ok = abs(nll / b - 1) <= 0.01
            msg.append(f"V3 {os.path.basename(f)} {k}: {len(v)} problems, loss {nll:.4f} vs base {b:.4f} ({100 * (nll / b - 1):+.2f}%): {'ok' if ok else 'FAIL'}")
else:  # LAYERS: the per-layer counters the model needs, in every stats file (reported)
    for p in sorted(glob.glob(f"{out}/st_g_C*_r*.json")):
        try:
            s = json.load(open(p)); lay = s.get("layers") or []
            ok = len(lay) == nl and all(len(l.get("miss_hist") or []) >= 2 and "fetches" in l and "prefetches" in l for l in lay) and s.get("steps", 0) > 0
        except Exception as e:
            ok, lay = False, []
        msg.append(f"LAYERS {os.path.basename(p)}: {len(lay)} layers with miss histograms: {'ok' if ok else 'MISSING'}")
print("\n".join(msg))
sys.exit(9 if bad else 0)
PY
}
FAILED=""; DONE14=""
for rd in $ROUNDS; do
  [ -n "$FAILED" ] && break
  run_round $CA $rd
  grep -q '"decode_ms"' $OUT/ec_g_C${CA}_r${rd}.jsonl 2>/dev/null && DONE14="$DONE14 $rd"
  if [ "$rd" = 1 ]; then
    vcheck V1 | tee -a $OUT/validity.txt; [ ${PIPESTATUS[0]} -eq 0 ] || { echo "V1 failed: stopping" | tee -a $OUT/validity.txt; FAILED=V1; }
  fi
  if [ -z "$FAILED" ] && [ "$rd" != 1 ]; then
    vcheck V2 $DONE14 | tee -a $OUT/validity.txt; [ ${PIPESTATUS[0]} -eq 0 ] || { echo "V2 failed (rounds$DONE14): stopping" | tee -a $OUT/validity.txt; [ "$SMOKE" = 1 ] || FAILED=V2; }
  fi
done
if [ -z "$FAILED" ]; then
  run_round $CB 1
  vcheck V1 | tee -a $OUT/validity.txt
fi
vcheck V3 | tee -a $OUT/validity.txt
vcheck LAYERS | tee -a $OUT/validity.txt
rm -f $F $OUT/la_g.bin $OUT/plan_g*.bin
# ---- summary
python3 - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, os, sys, statistics as st
out = sys.argv[1]; line = []
for f in sorted(os.listdir(out)):
    if not (f.startswith("ec_") and f.endswith(".jsonl")):
        continue
    by = {}
    for l in open(f"{out}/{f}"):
        if l.startswith("{"):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            lab = os.path.basename(r["config"].rsplit("stats=", 1)[-1]).replace(".json", "")
            by.setdefault(lab, {})[r["seq"]] = (r["decode_ms"] / r["n_decode"], r["nll_sum"] / r["nll_n"])
    base = next((v for k, v in by.items() if k.endswith("_base")), None)
    for lab, v in sorted(by.items()):
        ms = st.mean(x[0] for x in v.values()); nll = st.mean(x[1] for x in v.values())
        b = st.mean(base[s][0] for s in v if s in base) if base else float("nan")
        bn = st.mean(base[s][1] for s in v if s in base) if base else float("nan")
        stf = f"{out}/{lab}.json"
        try:
            s = json.load(open(stf))
        except Exception:   # missing, or cut short by the deadline
            s = {}
        n = max(1, s.get("steps", 1))
        hits, mis = s.get("hits", 0), s.get("misses", 0)
        print(f"{lab:26s} {1000 / ms:7.2f} tok/s {ms:6.2f} ms n={len(v)} vs base {b / ms:.3f} loss {100 * (nll / bn - 1):+.2f}% "
              f"misses {mis / n:.2f} admits {s.get('admits', 0) / n:.2f} fetches {s.get('fetches', 0) / n:.2f} "
              f"prefetches {s.get('prefetches', 0) / n:.2f} reads/tok {(mis + s.get('admits', 0) + s.get('prefetches', 0)) / n:.2f} "
              f"hit {hits / max(1, hits + mis):.4f} layers {len(s.get('layers') or [])} plan_records {s.get('plan_records', '-')}")
        if "_r1" in lab and "C14" in lab and not lab.endswith("_base"):
            line.append(f"{lab.replace('st_g_C14_r1_', '')} {b / ms:.3f}")
open(f"{out}/oneline.txt", "w").write("124 " + " ".join(line) + "\n")
PY
ONE=$(cat $OUT/oneline.txt)
python3 $J/manifest.py "$OUT" "124_slowlink@vast" "$T0" "$ONE" card=$CARD cpu_vendor=$CVEND law_gptoss=$LAWG fixed_table=$TAB device_read_gbs=$DR memory_clock_max_mhz=$MEMCLK ratio=$RATIO helpers=$H plans=$PLANS_OK failed=${FAILED:-none} smoke=$SMOKE
cat $OUT/skipped.txt 2>/dev/null
cat $OUT/validity.txt 2>/dev/null
echo "SUMMARY: $ONE"
