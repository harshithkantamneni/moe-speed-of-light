#!/bin/bash
# Job 123: the registered confirmation of the byte model and of "reads, not hits" (paper 2; the plan is
# prereg/p2/confirm_plan.json and prereg/p2/findings_engine.md section 5 on main). The question: on RTX 5090 desktops that
# played no part in building the model, does the frozen per-layer two-path byte model, given only the machine's first
# probe and each configuration's own counters, predict the decode time of ten cache configurations, and are the
# configurations with more hits slower where the model says so? The ratio floor of 0.6 below (VR) is the scope of the
# test (job 122's), not a finding of it.
# Protocol: gpt-oss-120b MXFP4; the job 084 own-text corpus, 20 AIME-25 problems x 256 teacher-forced steps, fresh
# context per configuration; engine patch oracle7; this machine's routing recorded first (untimed), and from it the
# Few plan (jobs/ec2/minadm_plan.py, HiGHS, in its own uv environment as job 112) at C = 14 and at C = 32; then C = 14
# for three rounds, each one process with the ten arms in a seeded shuffled order, then C = 32 for one round, same
# arms. One fetch table for every arm with in-step fetches on every machine, the fixed table 0,0,1,2,3, so that
# counters do not depend on the machine. The arms (H helpers = usable physical cores - 2; MB = mailbox=1:tdec=1:helpers=H):
#   baselaw    the deployed cache (decayed frequency) with this machine's own law table (jobs/ec2/fetch_table.py), as deployed
#   base       the deployed cache, the fixed table
#   lru        LRU admission (every miss admitted, copied after the step), the fixed table
#   aa         always-admit: every miss copied over the link in the step (kappa=-1e9, fetch_on_admit=1), the fixed table
#   hitopt     hit-optimal: Belady prefetch of the whole sequence, unpaced (oracle_bypass=0, paced=0), the fixed table
#   fetch      MIN with bypass, read once: each admission copied in the step (oracle_bypass=1, oracle_fetch=1), the fixed table
#   fetchplan  Few: fetch on MIN's fewest-admission hit-optimal schedule (oracle_plan=this machine's plan), the fixed table
#   both3p     MIN ahead: read once, paced copies issued three steps ahead (oracle_lead=3, paced=1), the fixed table
#   bypass0    MIN with bypass, two reads (CPU, then a background copy), in-step fetches off
#   base0      the deployed cache, in-step fetches off (every miss on the CPU)
# Definitions. A round's time for an arm: the sum of its decode milliseconds over the 20 problems / the decoded tokens;
# T(arm, C): the mean over its counted rounds (an arm-round counts when all 20 problems x 256 steps completed and V3
# held). Counters per token from the arm's own stats file (mean over its counted rounds): misses M, in-step fetches F,
# layer-ahead prefetches P, background copies A (admissions after the step, copies issued ahead); host reads M + A + P;
# hit rate hits / (hits + M); per-layer misses, fetches, prefetches and miss histograms (first counted round). A cell is
# one machine x arm x budget; error = predicted / measured - 1.
# The frozen model (scripts/p2/job123.py on main, committed before the first launch, sha256 51adfba672db5fd5...f3f8599c;
# its constants are in the file):
#   T_pred = max( G + sum over layer-steps of max(f s_p, c s_c, [0 < f < n] n s_b),
#                 root(G + X_in s_p, A s_p), root(G + (X_c + X_in) s_b, A s_b) ),  root(a, v) the larger root of
#   T = a + v (1 - G/T); X_c = M - F, X_in = F + P; s_x = S / B_x, S = 13,253,760 bytes; a layer-step's n misses split
#   into f link copies and c = n - f CPU reads: f = table[n] of the arm's own table (scaled to the fetches the engine
#   made) for dfa and lru without fetch-on-admit, binomially by the layer's fetched share F_l / M_l for the oracles and
#   fetch-on-admit; a layer-ahead prefetch on the previous layer's link side;
#   G = 4.413 ms (the median Nsight GPU time of gpt-oss-120b on 24 RTX 5090s; no constant fitted to time); B_c the CPU
#   read at H helpers (linear in threads), B_p the zero-copy read (16 MB x 64 blocks, the maximum) and B_both the median
#   CPU + zero-copy sum, all from this machine's first probe (concur.txt), read as jobs/ec2/fetch_table.py reads it.
#   The predictions are computed after the runs by that scorer, from these inputs only; counters are deterministic
#   functions of the routing, the policy and the table, not of timing, which C0 checks.
# Clauses (on the valid machines; "slower": larger T, point values):
#   C0. counters are not timing: every arm with frozen counters (base, aa, fetch, fetchplan, both3p; base0 and
#       bypass0; baselaw when its table has frozen counters; values in the scorer, from confirm_plan.json) recurs on
#       every valid machine, host reads per token within 1% of the frozen value (both3p, paced: 5%); hitopt and lru
#       agree across the valid machines, host reads per token max/min - 1 at most 1% at each budget (earlier machines:
#       hitopt's reads within 0.18% at C = 14 and 0.37% at C = 32; one oracle arm of job 112 spread 0.8% across machines);
#   T1. the median |error| pooled over every cell at most 7%;
#   T2. on every valid machine, the median |error| over its cells at most 11%;
#   T3. at least 70% of the cells within 10%;
#   T4. on every machine x budget with at least six counted arms, Spearman correlation of T and T_pred at least 0.6;
#   T5. of the pairs of arms on one machine and budget whose T differ by more than 1%, at least 85% ordered as T_pred;
#   T6. reads, not hits: of those pairs on which the hit rate (more than 0.005 apart; higher = faster) and T_pred predict
#       opposite orders and T_pred's gap is at least 10%, at least 85% ordered as T_pred (untested below five pairs);
#   P1. on every valid machine: hitopt (hit rate 0.98) slower than both3p at C = 14 and at C = 32, and slower than
#       fetch at C = 14 (untested on fewer than three machines with these cells);
#   P2. aa (58 host reads per token, all over the link in the step) slower than base (63, half on the CPU) at C = 14
#       on every valid machine where T_pred(aa) / T_pred(base) is at least 1.10 (untested where none is);
#   P3. (secondary) lru slower than base at C = 14 and at C = 32 on every valid machine (untested on fewer than three);
#   P4. (exploratory) T_pred's sign of hitopt against base right on at least 80% of the machine x budget cells.
# Decision: confirmed if C0, T1-T6 and P1 hold and P2 holds or is untested, on at least three valid machines; refuted
# (a failure of the frozen model, reported as such, with nothing refitted) if any of C0, T1-T6, P1, P2 fails;
# inconclusive if none fails but one of C0, T1-T6, P1 is untested; untested with fewer than three valid machines.
# Every host that starts is reported.
# Pass probabilities (prereg/p2/confirm_plan.json: the 47 launches of the 39 earlier qualifying RTX 5090 machines with
# at least six arms, resampled): T1-T5 together 0.96 / 0.98 / 0.99 on 3 / 4 / 5 machines, T6 1.00 (65 of 66 such pairs
# right in the test's families); the law without background copies passes T1-T5 together 0.57 / 0.50 / 0.44; a world
# where the hit rate decides fails T6, P1 and P2 (the hit rate would pass T5 alone in about two thirds of draws).
# On the 39 earlier probes the model put hitopt above both3p by 1.25-2.07x (C = 14) and 1.14-1.71x (C = 32), above
# fetch by 1.20-1.77x (C = 14), aa above base by at least 10% on 30 of 39, lru above base on all 39.
# VALIDITY: V0 the GPU's UUID in no earlier launch (jobs/ec2/known_gpu_uuids123.txt: every UUID in the known list and in
# results/ up to job 122; no repeat host), at most 48 GB of host memory in use; V0c one NUMA node, at most 32 usable
# physical cores, an RTX 5090 with device reads >= 1500 GB/s, at least 90 GB of host memory, at least 150 GB of free
# disk; VR the first probe's ratio B_p / B_c at least 0.6; V1 base's loss within 2% of 0.190 in every round where
# base completed; V2 base's complete C = 14 rounds (at least two) within 2% of each other; V3 every arm's loss within
# 1% of base's in its round (an arm-round failing it does not count). A host is valid if it passes V0, V0c, VR, V1
# and V2; a GPU that already ran in this job does not count again. A host stops at its first failed V0, V0c, VR, V1 or
# V2. If the plan cannot be computed, fetchplan is dropped on that host (recorded).
# Hosts: single-GPU, whole-machine on-demand RTX 5090 offers at most $1.10/h, at least 90 GB of host memory, 160 GB of
# disk and 300 Mb/s down, a consumer desktop CPU (Ryzen 3/5/7/9 but not Threadripper, Core i3/i5/i7/i9, Core Ultra), on
# a machine never rented before (no offer of it in the rental ledger; V0 checks the GPU itself), in order of expected
# cost, $/h x (1.4 h + the hours to download 65 GB at the listed speed), from a listing committed and pushed before each
# launch (jobs/ec2/list123.py). No repeat machines. At most two started hosts that did not fail a gate per Vast host;
# up to three run at once. The job stops starting hosts when five are valid, ten have started, its hosts' spend in the
# rental ledger reaches $9 (the smoke run counts in neither), or 48 hours after this header was pushed, whichever comes
# first; and a host starts only while the account's credit, less a $2.20 hold for each running rental of this job (the
# smoke run included) and for the new one, stays at or above a $1.50 reserve (no host starts with credit below $3.70;
# a host costs up to about $2: its time plus up to $0.013/GB for the 65 GB download). Budget per host: 110 minutes
# from the end of the V0 check and the package setup (T0); a step that would run past it less five minutes is cut
# there or skipped (a host whose C = 32 round is skipped counts with its C = 14 cells). A smoke run (SMOKE=1: gpt-oss-20b, 2 problems x 32 steps, budgets 14 and 24, gates recorded,
# not enforced) checks the script on one host before launch (the build, the plan, all ten arms, lru for the first time
# in ec-bench); it is not a host of this job.
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
SMOKE=${SMOKE:-0}
MREPO=ggml-org/gpt-oss-120b-GGUF; MFILE=gpt-oss-120b-MXFP4.gguf; NL=36
CA=14; CB=32; NSEQ=20; NDEC=256; ROUNDS="1 2 3"; TAB=0,0,1,2,3
BMIN=110; RAMMIN=90; DISKMIN=150; VRMIN=0.6; V1REF=0.190; DRMIN=1500
ARMS="baselaw base lru aa hitopt fetch fetchplan both3p bypass0 base0"
if [ "$SMOKE" = 1 ]; then
  MREPO=ggml-org/gpt-oss-20b-GGUF; MFILE=gpt-oss-20b-MXFP4.gguf; NL=24; NSEQ=2; NDEC=32; DISKMIN=60; RAMMIN=24; CB=24
fi
echo "SMOKE=$SMOKE model $MFILE budgets C=$CA (rounds $ROUNDS), C=$CB (one round) problems $NSEQ x $NDEC table $TAB arms=[$ARMS] VRMIN=$VRMIN" | tee $OUT/mode.txt
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
if grep -qx "$UUID" "$JD/ec2/known_gpu_uuids123.txt"; then
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
case "$GNAME" in
  *5090*) CARD=5090 ;;
  *) CARD=other; gate_fail 4 "V0c FAIL: not an RTX 5090 (${GNAME}), stopping" ;;
esac
echo "card class $CARD" | tee -a $OUT/gate.txt
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
sha256sum $PATCH $J/known_gpu_uuids123.txt $J/fetch_table.py $J/minadm_plan.py $J/la_act.py | tee $OUT/patch_sha.txt; echo "base $BASE" >> $OUT/patch_sha.txt
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
open(f"{out}/oneline.txt", "w").write("123 " + " ".join(line) + "\n")
PY
ONE=$(cat $OUT/oneline.txt)
python3 $J/manifest.py "$OUT" "123_confirm@vast" "$T0" "$ONE" card=$CARD law_gptoss=$LAWG fixed_table=$TAB device_read_gbs=$DR memory_clock_max_mhz=$MEMCLK ratio=$RATIO helpers=$H plans=$PLANS_OK failed=${FAILED:-none} smoke=$SMOKE
cat $OUT/skipped.txt 2>/dev/null
cat $OUT/validity.txt 2>/dev/null
echo "SUMMARY: $ONE"
