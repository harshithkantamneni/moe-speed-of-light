#!/bin/bash
# Job 121: the fix, a held-out workload and a second engine (reviews 37a/b). Jobs 114-120 found that the engine's
# in-step fetches (a fetch table copies some of a step's misses into GPU slots whatever the policy chose) evict MIN's
# set: each fetch evicts the resident with the lowest decayed count, the deployed policy's order, not MIN's. A replay
# (scripts/fetch_victim_replay.py, scripts/job121_predict.py on main) says that ranking those victims by MIN's own
# order (the resident needed furthest ahead first) removes about two-thirds of the misses the fetches add. This job
# (1) builds that fix (engine patch oracle7 = oracle5 + LLAMA_EC_FETCH_OWN=1, configuration "own": MIN-2R with the
#     fetch table on and its victims in MIN's order) and times it against the deployed cache, MIN-2R with the fetches
#     on (bypass) and off (bypass0);
# (2) repeats it on a workload never used to build or tune anything: the 12 sequences of own sampled text in chat,
#     math, code and other languages (jobs/ec2/S_gpt-oss-120b.jsonl, 448 teacher-forced steps each), process "s";
# (3) times a second, independent engine on the same machine and text: upstream llama.cpp's MoE cache (PR #29887 at
#     2de17c2, one LRU over GPU banks shared by all layers, every miss uploaded and run on the GPU), built with our
#     teacher-forced harness (jobs/ec2/llama.cpp-pr29887-2de17c2-ec-bench.patch, which only adds tools/ec-bench and
#     passes LLAMA_EC_MOE_CACHE_MIB to its context), at the same GPU bytes as 11% and 25% (504 and 1152 slots,
#     pr14 and pr32), and with its cache off (cpu: experts of a small batch on the CPU, upstream's --cpu-moe), process
#     "p". Same weights placement for both engines: every expert in pinned host memory (--host-experts).
# Protocol otherwise as job 119 (gpt-oss-120b MXFP4; the job 084 own-text corpus, 20 AIME-25 problems x 256 steps,
# process "g"; this machine's routing of each text recorded first, untimed; one process per round with its
# configurations in a seeded shuffled order). Three rounds of each process, in the order g s p, p g s, s p g.
# g: base base0 bypass bypass0 own at C = 14 (11%) and base32 (the deployed cache at C = 32, 25%); s: base base0
# bypass bypass0 own at C = 14; p: pr14 pr32 cpu.
# Definitions as job 119: X/Y is Y's mean time per token over the problems / X's (a speed ratio), geometric mean over
# rounds; Eq. (1) is R* S / B_host at the first probe's rate, R* this machine's MIN-with-bypass reads per token per
# layer on the text it times; for the shared cache of PR #29887, Eq. (1) with R* of MIN with bypass over one pool of
# 14 x 36 (or 32 x 36) slots on the layer-interleaved stream (the fewest host reads of any shared cache of that size).
# Replay numbers (job 084c's trace, first 20 problems; prereg/job121_predict.json): per layer at 11%, MIN 38.2 misses
# per token; with table 0,0,1,2,2: score-ordered fetches 44.0, own-ordered 40.2 (0,0,1,1,2: 43.3, 39.8); jobs 119/120's
# engine counters: bypass0 43.5, bypass 50.2 (table 0,0,1,2,2) and 48.6 (0,0,1,1,2). PR #29887's LRU exactly as it
# orders hits and misses: 69.6 misses per token at 504 slots, 31.1 at 1152; MIN over the shared pool 36.1 and 13.9.
# Predictions, committed before launch, on every valid machine whose first probe's fetch table is not all zero
# (11% unless stated):
#   F1. counters (AIME): own makes in-step fetches, and its misses per token are at most bypass0's + 0.5 (bypass's -
#       bypass0's): the fix removes at least half of the misses the fetches add (replay: two-thirds);
#   F2. own/bypass at least 1.02 (AIME);
#   F3. own/base at least 1.04 (AIME);
#   F4. pooled over valid machines (at least three): the mean of ln(own/bypass0) above zero, its 95% t-interval over
#       machines above zero: with the fix, the fetches add to MIN's set instead of costing it;
#   S1. on the held-out text: bypass0/base0 at least 1.07;
#   S2. on the held-out text: own/base at least 1.04 and own/bypass at least 1.02;
#   S3. the deployed cache's fraction of Eq. (1)'s speed (Eq. (1) / base) on the held-out text within 0.10 of its
#       fraction on AIME on the same machine (point values);
# and on every valid machine (the second engine, AIME):
#   P1. counters: PR #29887's misses per decode token (its own log, ubatch <= 8) within 3% of 69.6 at 504 slots and of
#       31.1 at 1152 (its LRU replayed on job 084c's trace);
#   P2. the bound holds for a second engine: pr14 and pr32 each take at least Eq. (1) per token for the shared pool
#       at their budget (point values: a bound is a bound);
#   P3. base/pr14 at least 1.10: the deployed cache beats upstream's at 11%.
# Exploratory (no prediction): base32/pr32, pr14/cpu, each engine's fraction of the bound, the fix's share of the gap.
# Rule for intervals: as job 119 (problems' bootstrap combined with the configuration's variation between rounds, three
# rounds, t with two degrees of freedom). A clause held when the interval lies on the predicted side (pooled: the
# t-interval over machines; counters and P2, S3: the point values), failed otherwise. Analysis: scripts/job121.py on
# main, written after the runs to these definitions.
# VALIDITY: V0 the GPU's UUID in no earlier launch (jobs/ec2/known_gpu_uuids.txt, now with jobs 119/120's), at most 48
# GB of host memory in use; V0c one NUMA node, at most 32 usable physical cores, an RTX 5090 (device reads >= 1500 GB/s)
# or RTX 4090 (>= 800 GB/s), at least 90 GB of host memory, at least 150 GB of free disk; VR the first probe's ratio B_p
# / B_c at least 0.5 (a fetch table that fetches); V1 base's AIME loss within 2% of 0.190 (5090) or 0.1963 (4090);
# V2 base's three AIME rounds within 2% of each other; V3 every configuration's loss within 1% of base's in its process
# (reported; a configuration failing it is left out there); VS base's three held-out rounds within 2% (else the S
# clauses are void there); VP PR #29887 logs 504 and 1152 slots and its loss is within 2% of base's on AIME (else the
# P clauses are void there). A host is valid if it passes V0, V0c, VR, V1 and V2. Every host that starts is reported.
# Hosts: single-GPU, whole-machine (gpu_frac 1) on-demand RTX 5090 offers at most $1.10/h or RTX 4090 offers at most
# $0.70/h, with at least 90 GB of host memory, 160 GB of disk and 300 Mb/s down, whose listed CPU is a consumer desktop
# part (as job 119), not in the rental ledger, in order of expected cost, $/h x (2.6 h + the hours to download 65 GB at
# the listed speed), from a listing committed and pushed before each launch (jobs/ec2/list121.py), skipping machines
# whose GPU failed V0 earlier. At most two valid machines per Vast host. Up to three run at once. The job stops starting
# hosts when six are valid, eleven have started, its spend in the rental ledger reaches $20, or 72 hours after this
# header was pushed, whichever comes first. Budget per host: 250 minutes (later rounds are skipped if the deadline nears;
# recorded). A smoke run (SMOKE=1: gpt-oss-20b, 2 problems x 32 steps, gates recorded, not enforced) checks the
# script on one host before launch; it is not a host of this job.
# Amendment (committed before host b; host a's container never started): consumer desktops with at least 90 GB of host
# memory and a new GPU proved scarce (at host b's turn one such offer was listed, and its machine failed to start).
# When no new offer qualifies and at least two hours have passed since this job's last launch, a launch may take the
# cheapest offer that meets every rule but the ledger's on a machine that was a valid host of jobs 109-120
# (jobs/ec2/repeat_machines121.txt), with V0's new-GPU clause waived for it (its wrapper sets REPEAT=1): a "repeat"
# host. Every clause is scored over all valid hosts and, separately, over the new ones alone; no repeat host on a
# machine already run in this job, and the two-per-provider rule applies to them too. Predictions and validity are
# unchanged.
# Amendment 2 (committed before host d): the job was paused from 2026-10-14T20:27Z to 2026-10-19 (the operator's usage
# limits), so its 72-hour window lapsed on 2026-10-17T18:13Z with no valid host (a never started; b stopped at V0c, a
# throttled card; c's offer vanished before its launch). The window is reopened for 72 hours from this amendment's push.
# The launcher also skips machines that failed V0c or VR in jobs 119-121, as they would fail again. Nothing else changes.
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
SMOKE=${SMOKE:-0}
MREPO=ggml-org/gpt-oss-120b-GGUF; MFILE=gpt-oss-120b-MXFP4.gguf; NL=36
CA=14; CB=32; NSEQ=20; NDEC=256; NSEQS=12; NDECS=448; ROUNDS="1 2 3"
BMIN=250; RAMMIN=90; DISKMIN=150; VRMIN=0.5
NG="base base0 bypass bypass0 own base32"; NS="base base0 bypass bypass0 own"; NP="pr14 pr32 cpu"
if [ "$SMOKE" = 1 ]; then
  MREPO=ggml-org/gpt-oss-20b-GGUF; MFILE=gpt-oss-20b-MXFP4.gguf; NL=24; NSEQ=2; NDEC=32; NSEQS=2; NDECS=32; DISKMIN=60
fi
# PR #29887 sizes its banks from the byte budget: n slots need (n + 1) x 13,219,200 bytes (gate, up and down of one
# MXFP4 expert of n_embd = n_ff = 2880; its biases stay on the host); the largest n that fits is the slot count
# (shell arithmetic: python3 may not be installed yet; the smoke run found both empty, so its pr14 and pr32 ran with
# the cache off)
MIB14=$(( ((CA * NL + 1) * 13219200 + 1048575) / 1048576 ))
MIB32=$(( ((CB * NL + 1) * 13219200 + 1048575) / 1048576 ))
echo "SMOKE=$SMOKE model $MFILE C=$CA (and base32) problems $NSEQ x $NDEC held-out $NSEQS x $NDECS g=[$NG] s=[$NS] p=[$NP] PR MiB $MIB14 $MIB32" | tee $OUT/mode.txt
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
REPEAT=${REPEAT:-0}
if grep -qx "$UUID" "$JD/ec2/known_gpu_uuids.txt" && [ "$REPEAT" = 1 ]; then
  echo "V0: a GPU rented before ($UUID): a repeat host, admitted by the amendment" | tee -a $OUT/v0.txt
elif grep -qx "$UUID" "$JD/ec2/known_gpu_uuids.txt"; then
  gate_fail 8 "V0 FAIL: GPU rented before ($UUID), stopping"
else
  echo "V0: a GPU in no earlier launch" | tee -a $OUT/v0.txt
fi
[ "${USEDGB:-999}" -le 48 ] || gate_fail 7 "V0 FAIL: ${USEDGB} GB of host memory in use (> 48), stopping"
echo "V0 checked" | tee -a $OUT/v0.txt
. "$JD/ec2/setup.sh"
export BASE; export -f build_tree platform getmodel
PATCH=$J/llama.cpp-expert-cache-4da6337-oracle7.patch
PRPATCH=$J/llama.cpp-pr29887-2de17c2-ec-bench.patch
PRSHA=2de17c21b81fab40ba962500deb3ee2247d8e356
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
  *5090*) CARD=5090; DRMIN=1500; V1REF=0.190 ;;
  *4090*) CARD=4090; DRMIN=800; V1REF=0.1963 ;;
  *) CARD=other; DRMIN=99999; V1REF=0.190; gate_fail 4 "V0c FAIL: neither an RTX 5090 nor an RTX 4090 (${GNAME}), stopping" ;;
esac
echo "card class $CARD" | tee -a $OUT/gate.txt
[ "${DR:-0}" -ge $DRMIN ] || gate_fail 5 "V0c FAIL: GPU reads below $DRMIN GB/s: throttled card, stopping"
RAMGB=$(awk '/Mem:/{print $2}' $OUT/free.txt); echo "host RAM ${RAMGB} GB, in use ${USEDGB} GB" | tee -a $OUT/gate.txt
[ "${RAMGB:-0}" -ge $RAMMIN ] || gate_fail 6 "V0c FAIL: ${RAMGB} GB of host memory (< $RAMMIN), stopping"
DISKGB=$(df -BG --output=avail $WORK | tail -1 | tr -dc 0-9); echo "disk free ${DISKGB} GB" | tee -a $OUT/gate.txt
[ "${DISKGB:-0}" -ge $DISKMIN ] || gate_fail 6 "V0c FAIL: ${DISKGB} GB of free disk (< $DISKMIN), stopping"
echo "V0c checked" | tee -a $OUT/v0.txt
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip >> $OUT/apt.txt 2>&1
python3 -m pip install -q --break-system-packages huggingface_hub hf_xet > $OUT/pip.txt 2>&1
python3 -c "import huggingface_hub" >> $OUT/pip.txt 2>&1 || echo "huggingface_hub missing" >> $OUT/pip.txt
# --- the download first; the builds and the probe run meanwhile
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
build_pr() {  # upstream llama.cpp at PR #29887's head 2de17c2 with the harness added
  local d=$WORK/lc-pr
  [ -f $d/.built ] && return 0
  ( cd $WORK && rm -rf $d && mkdir -p $d && cd $d && git init -q . && git remote add origin https://github.com/ggml-org/llama.cpp &&
    { git fetch -q --depth 1 origin $PRSHA || { sleep 20; git fetch -q --depth 1 origin $PRSHA; } || { sleep 60; git fetch -q --depth 1 origin $PRSHA; } ||
      git fetch -q --depth 1 origin refs/pull/29887/head; } &&   # retries: host f's fetch met a DNS failure
    git checkout -q FETCH_HEAD && [ "$(git rev-parse HEAD)" = "$PRSHA" ] && git apply "$PRPATCH" &&
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$SM -DLLAMA_CURL=OFF \
      > $OUT/cmake_lc-pr.txt 2>&1 &&
    cmake --build build -j$NPROC --target llama-ec-bench > $OUT/build_lc-pr.txt 2>&1 && touch .built )
  echo "build lc-pr: $([ -f $d/.built ] && echo ok || echo FAILED) at $(git -C $d rev-parse HEAD 2>/dev/null)"
  grep -E "error" $OUT/build_lc-pr.txt 2>/dev/null | head -20
}
build_pr
PR_BIN=$WORK/lc-pr/build/bin/llama-ec-bench; ls -la $PR_BIN || echo "no PR binary: process p is skipped" | tee -a $OUT/skipped.txt
sha256sum $PATCH $PRPATCH $J/known_gpu_uuids.txt $J/fetch_table.py $J/S_gpt-oss-120b.jsonl | tee $OUT/patch_sha.txt; echo "base $BASE pr $PRSHA" >> $OUT/patch_sha.txt
# --- the host (downloading, otherwise idle): platform, our probe, the fetch table, the ratio
platform
H=$(( CORES > 4 ? CORES - 2 : 2 ))
echo "helpers H=$H" | tee -a $OUT/gate.txt
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
grep -q "concurrent t=" $OUT/concur.txt || { echo "concur incomplete, retrying"; R 15m $WORK/concur > $OUT/concur.txt 2>&1; }
echo "probe 1 ended $(date -u +%FT%TZ); download running: $(kill -0 $DLG 2>/dev/null && echo yes || echo no)" | tee $OUT/probe1_when.txt
LAWG=$(python3 $J/fetch_table.py $OUT/concur.txt $H 13253760 4 $OUT/fetch_table_law_gptoss.json 37)
echo "law table ($(date -u +%FT%TZ)): gpt-oss $LAWG" | tee $OUT/tables.txt
[ -n "$LAWG" ] || { echo "no law table"; exit 3; }
# smoke runs only: a slow link gives an all-zero table, which would leave the fetch path (and own) untried
if [ "$SMOKE" = 1 ] && ! echo "$LAWG" | grep -q "[1-9]"; then LAWG=0,0,1,2,2; echo "smoke: table forced to $LAWG" | tee -a $OUT/tables.txt; fi
RATIO=$(python3 -c "import json, sys; d = json.load(open(sys.argv[1])); print(f'{d[\"B_p\"] / d[\"B_c\"]:.4f}')" $OUT/fetch_table_law_gptoss.json)
echo "VR: B_p / B_c = ${RATIO:-?}" | tee -a $OUT/gate.txt
if ! python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)" "${RATIO:-0}" $VRMIN; then
  if [ "$SMOKE" != 1 ]; then pkill -P $DLG; kill $DLG; fi
  gate_fail 10 "VR FAIL: link/CPU ratio ${RATIO} below $VRMIN, stopping"
fi
echo "VR checked" | tee -a $OUT/v0.txt
echo "builds and probe done, waiting for the download ($(( $(date +%s) - T0 )) s since launch)"
wait $DLG
cat $OUT/dl_gguf.txt
F=$M/$MFILE
[ -s "$F" ] || { echo "GGUF MISSING"; exit 3; }
df -h $WORK | tail -1 > $OUT/df.txt
# --- three more probe runs on the idle machine (recorded; the table and the gate stay the first probe's)
for i in 1 2 3; do
  t0=$(date +%s)
  R 15m $WORK/concur > $OUT/concur_idle$i.txt 2>&1
  echo "idle probe $i rc=$? in $(( $(date +%s) - t0 )) s ($(date -u +%FT%TZ))" | tee -a $OUT/probe_idle.txt
done
MB="mailbox=1:tdec=1:helpers=$H"
CG=$J/aime25_owntext_gptoss_084.jsonl; CS=$J/S_gpt-oss-120b.jsonl; wc -l $CG $CS
lookahead() {  # tag corpus n-seq n-decode: this machine's routing of the teacher-forced text (untimed)
  local tag=$1 corpus=$2 ns=$3 nd=$4 t0=$(date +%s)
  R 30m $EC_BIN -m $F --corpus $corpus -t $CORES --n-seq $ns --n-prefill 1024 --n-decode $nd --ncmoe 24 --lookahead $OUT/la_$tag.bin > $OUT/la_$tag.jsonl 2> $OUT/la_$tag.err
  echo "lookahead $tag rc=$? in $(( $(date +%s) - t0 )) s: $(ls -la $OUT/la_$tag.bin | awk '{print $5}') bytes"; cat $OUT/la_$tag.bin.json
  tail -n 5 $OUT/la_$tag.err > $OUT/la_$tag.err.tail; rm -f $OUT/la_$tag.err
  # the routing alone (int8 experts per step and layer), small enough to ship: the analysis replays MIN and the LRU on it
  python3 $J/la_act.py $OUT/la_$tag.bin $OUT/route_$tag.npz
}
run_round() {  # process round: one process, its configurations in a shuffled order
  local m=$1 rd=$2
  local bin=$EC_BIN corpus la ns nd names cfg="" n c
  case $m in
    g) corpus=$CG; la=$OUT/la_g.bin; ns=$NSEQ; nd=$NDEC; names=$NG ;;
    s) corpus=$CS; la=$OUT/la_s.bin; ns=$NSEQS; nd=$NDECS; names=$NS ;;
    p) corpus=$CG; ns=$NSEQ; nd=$NDEC; names=$NP; bin=$PR_BIN
       [ -x "$bin" ] || { echo "no PR binary: p_r$rd skipped" | tee -a $OUT/skipped.txt; return 0; } ;;
  esac
  local common="--corpus $corpus -t $CORES --n-seq $ns --n-prefill 1024 --n-decode $nd --no-mmap --host-experts"
  local base="slots=$CA:policy=dfa:$MB:fetch=$LAWG:paced=0:overlap=1:oracle_w=0:kappa=1"
  local base0="slots=$CA:policy=dfa:$MB:fetch=0:paced=0:overlap=1:oracle_w=0:kappa=1"
  local orc="slots=$CA:policy=oracle:oracle=$la:$MB:kappa=1:fetch=$LAWG:overlap=1:oracle_w=0:oracle_bypass=1:paced=0"
  local orc0="slots=$CA:policy=oracle:oracle=$la:$MB:kappa=1:fetch=0:overlap=1:oracle_w=0:oracle_bypass=1:paced=0"
  local tag="${m}_C${CA}_r${rd}"
  for n in $names; do
    case $n in
      base)    c="$base" ;;
      base0)   c="$base0" ;;
      bypass)  c="$orc" ;;
      bypass0) c="$orc0" ;;
      own)     c="$orc:fetch_own=1" ;;
      base32)  c="slots=$CB:policy=dfa:$MB:fetch=$LAWG:paced=0:overlap=1:oracle_w=0:kappa=1" ;;
      pr14)    c="moe_cache_mib=$MIB14" ;;
      pr32)    c="moe_cache_mib=$MIB32" ;;
      cpu)     c="moe_cache_mib=0" ;;
      *)       echo "unknown configuration $n"; return 2 ;;
    esac
    cfg="${cfg:+$cfg;}$c:stats=$OUT/st_${tag}_$n.json"
  done
  local off=0; case $m in s) off=7 ;; p) off=13 ;; esac
  local seed=$(( ORDER_SEED + 100 * rd + off ))
  echo "cell $tag order-seed $seed configs: $cfg" >> $OUT/configs.txt
  local t0=$(date +%s)
  R 60m $bin -m $F $common --order-seed $seed --ec "$cfg" > $OUT/ec_${tag}.jsonl 2> $OUT/ec_${tag}.err
  local rc=$?
  echo "cell $tag rc=$rc in $(( $(date +%s) - t0 )) s"; echo "ec_${tag} $rc $(( $(date +%s) - t0 ))" >> $OUT/runs.txt
  grep "\[ec-bench\] config [0-9]*/" $OUT/ec_${tag}.err | sed 's/:stats=.*st_/ -> st_/' > $OUT/ec_${tag}.order
  grep -i "oracle\|fetch table\|own order\|helpers\|moe_cache\|moe cache\|slots (" $OUT/ec_${tag}.err | head -80 > $OUT/ec_${tag}.err.oracle
  tail -n 30 $OUT/ec_${tag}.err > $OUT/ec_${tag}.err.tail; rm -f $OUT/ec_${tag}.err
}
vcheck() {  # what process reference [rounds...]: V1 / V2 / V3 on the rounds run so far
  python3 - $OUT $SMOKE "$@" <<'PY'
import glob, json, os, sys
out, smoke, what, m, ref = sys.argv[1], sys.argv[2] == "1", sys.argv[3], sys.argv[4], float(sys.argv[5])
def by_config(f):
    by = {}
    for l in open(f):
        if l.strip():
            r = json.loads(l)
            lab = os.path.basename(r["config"].rsplit("stats=", 1)[-1]).replace(".json", "").rsplit("_", 1)[-1]
            by.setdefault(lab, []).append(r)
    return by
msg, bad = [], False
files = sorted(glob.glob(f"{out}/ec_{m}_C*_r*.jsonl"))
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
    rounds = sys.argv[6:]
    t = []
    for rd in rounds:
        p = glob.glob(f"{out}/ec_{m}_C*_r{rd}.jsonl")
        b = by_config(p[0]).get("base", []) if p else []
        t.append(sum(r["decode_ms"] / r["n_decode"] for r in b) / len(b) if b else float("nan"))
    v = max(t) / min(t) - 1
    bad = not (v <= 0.02)
    msg.append(f"V2 {m} base ms/token {' '.join(f'{x:.3f}' for x in t)}: spread {100 * v:.2f}%: {'FAIL' if bad else 'ok'}")
else:  # V3: every configuration computes the same model as base in its process; reported, never stops the host
    for f in files:
        by = by_config(f)
        k0 = "base" if "base" in by else ("cpu" if "cpu" in by else None)
        if k0 is None:
            continue
        b = sum(r["nll_sum"] / r["nll_n"] for r in by[k0]) / len(by[k0])
        for k, v in sorted(by.items()):
            nll = sum(r["nll_sum"] / r["nll_n"] for r in v) / len(v)
            ok = abs(nll / b - 1) <= 0.01
            msg.append(f"V3 {os.path.basename(f)} {k}: {len(v)} problems, loss {nll:.4f} vs {k0} {b:.4f} ({100 * (nll / b - 1):+.2f}%): {'ok' if ok else 'FAIL'}")
print("\n".join(msg))
sys.exit(9 if bad else 0)
PY
}
# ---- routing of both texts (untimed), then the timed rounds
lookahead g $CG $NSEQ $NDEC
lookahead s $CS $NSEQS $NDECS
[ -s $OUT/la_g.bin ] || { echo "no AIME lookahead file: oracle runs impossible"; exit 3; }
[ -s $OUT/la_s.bin ] || { echo "no held-out lookahead file: process s skipped" | tee -a $OUT/skipped.txt; NS=""; }
FAILED=""
order_of() { case $1 in 1) echo "g s p" ;; 2) echo "p g s" ;; 3) echo "s p g" ;; esac; }
for rd in $ROUNDS; do
  for m in $(order_of $rd); do
    [ -n "$FAILED" ] && break
    [ "$m" = s ] && [ -z "$NS" ] && continue
    run_round $m $rd
    if [ "$m" = g ] && [ "$rd" = 1 ]; then
      vcheck V1 g $V1REF | tee -a $OUT/validity.txt; [ ${PIPESTATUS[0]} -eq 0 ] || { echo "V1 failed: stopping" | tee -a $OUT/validity.txt; FAILED=V1; }
    fi
  done
  if [ -z "$FAILED" ] && [ "$rd" = 2 ]; then
    vcheck V2 g $V1REF 1 2 | tee -a $OUT/validity.txt; [ ${PIPESTATUS[0]} -eq 0 ] || { echo "V2 failed (two rounds): stopping" | tee -a $OUT/validity.txt; [ "$SMOKE" = 1 ] || FAILED=V2; }
  fi
done
if [ -z "$FAILED" ]; then
  vcheck V2 g $V1REF 1 2 3 | tee -a $OUT/validity.txt
  vcheck V2 s $V1REF 1 2 3 | sed 's/^V2 s/VS s/' | tee -a $OUT/validity.txt
fi
for m in g s p; do vcheck V3 $m $V1REF | tee -a $OUT/validity.txt; done
grep -h "slots (\|MoE cache size\|hits = " $OUT/ec_p_*.err.oracle 2>/dev/null | sort | uniq -c > $OUT/pr_cache_log.txt; cat $OUT/pr_cache_log.txt
rm -f $F $OUT/la_g.bin $OUT/la_s.bin
# ---- summary
python3 - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, os, sys, statistics as st
out = sys.argv[1]; line = []
for f in sorted(os.listdir(out)):
    if not (f.startswith("ec_") and f.endswith(".jsonl")):
        continue
    by = {}
    for l in open(f"{out}/{f}"):
        if l.strip():
            r = json.loads(l)
            lab = os.path.basename(r["config"].rsplit("stats=", 1)[-1]).replace(".json", "")
            by.setdefault(lab, {})[r["seq"]] = (r["decode_ms"] / r["n_decode"], r["nll_sum"] / r["nll_n"])
    base = next((v for k, v in by.items() if k.endswith("_base")), None) or next((v for k, v in by.items() if k.endswith("_cpu")), None)
    for lab, v in sorted(by.items()):
        ms = st.mean(x[0] for x in v.values()); nll = st.mean(x[1] for x in v.values())
        b = st.mean(base[s][0] for s in v if s in base) if base else float("nan")
        bn = st.mean(base[s][1] for s in v if s in base) if base else float("nan")
        stf = f"{out}/{lab}.json"
        s = json.load(open(stf)) if os.path.exists(stf) else {}
        n = max(1, s.get("steps", 1))
        print(f"{lab:26s} {1000 / ms:7.2f} tok/s {ms:6.2f} ms n={len(v)} vs ref {b / ms:.3f} loss {100 * (nll / bn - 1):+.2f}% "
              f"misses {s.get('misses', 0) / n:.2f} admits {s.get('admits', 0) / n:.2f} fetches {s.get('fetches', 0) / n:.2f}")
        if "_r1" in lab and not lab.endswith(("_base", "_cpu")):
            line.append(f"{lab.replace('st_', '')} {b / ms:.3f}")
open(f"{out}/oneline.txt", "w").write("121 " + " ".join(line) + "\n")
PY
ONE=$(cat $OUT/oneline.txt)
python3 $J/manifest.py "$OUT" "121_ownfetch@vast" "$T0" "$ONE" card=$CARD law_gptoss=$LAWG device_read_gbs=$DR memory_clock_max_mhz=$MEMCLK ratio=$RATIO helpers=$H pr_mib14=$MIB14 pr_mib32=$MIB32 failed=${FAILED:-none} smoke=$SMOKE
cat $OUT/skipped.txt 2>/dev/null
cat $OUT/validity.txt 2>/dev/null
echo "SUMMARY: $ONE"
