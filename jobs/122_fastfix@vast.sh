#!/bin/bash
# Job 122: the fix on fast links (reviews 38-41). Job 121 registered F4: pooled over its valid machines, "own" (MIN-2R
# with the engine's in-step fetches on and their victims in MIN's own order, engine patch oracle7, LLAMA_EC_FETCH_OWN)
# faster than "bypass0" (MIN-2R with the fetches off). It failed: 1.04 [0.94, 1.15] over four machines. Three had a
# link-to-CPU ratio of 0.65-0.86 and ran own/bypass0 1.045-1.10; the fourth (ratio 0.55) ran 0.95, and there the
# fetches slowed even the deployed cache. This job asks, on machines that played no part in that result, whether the
# fix beats switching the fetches off where the link is fast. The ratio floor of 0.6 below (VR) was chosen after job 121,
# from its one slow machine: it is the scope of this test, not a finding of it.
# Protocol as job 121's process g, alone: gpt-oss-120b MXFP4 at 11% (C = 14); the job 084 own-text corpus, 20 AIME-25
# problems x 256 teacher-forced steps; this machine's routing recorded first, untimed; three rounds, each one process
# with its configurations in a seeded shuffled order. Configurations: base (the deployed cache, fetch table on), base0
# (the deployed cache, fetches off), bypass0 (MIN-2R, fetches off), own (MIN-2R, fetch table on, victims in MIN's order).
# Definitions as job 121: X/Y is Y's mean time per token over the problems / X's (a speed ratio), geometric mean over
# rounds; intervals combine the problems' bootstrap with the configuration's variation between rounds (three rounds, t
# with two degrees of freedom); a clause held when its interval lies on the predicted side, failed otherwise.
# Predictions, committed before launch, on every valid machine whose first probe's fetch table is not all zero:
#   G1. pooled over the valid machines (at least four; else untested): the mean of ln(own/bypass0) above zero, its 95%
#       t-interval over machines above zero (the primary clause);
#   G2. own/bypass0 above 1 on every machine;
#   G3. base/base0 above 1 on every machine (the mechanism: on a fast link the fetches speed the deployed cache);
#   G4. counters: own's misses per token at most bypass0's (point values).
# Power for G1 (noncentral t, 5% two-sided, planned six machines): with job 121's three fast-link machines (mean of
# ln(own/bypass0) 0.073, standard deviation between machines 0.026), near 1; with that standard deviation doubled, 0.77;
# tripled, 0.44.
# Exploratory (no prediction): own/base, bypass0/base0, the fix's share of the gap, G1 with job 121's three fast-link
# machines added (reported apart). Analysis: scripts/job122.py on main, written after the runs to these definitions.
# VALIDITY: V0 the GPU's UUID in no earlier launch (jobs/ec2/known_gpu_uuids.txt, now with job 121's), at most 48 GB of
# host memory in use; V0c one NUMA node, at most 32 usable physical cores, an RTX 5090 (device reads >= 1500 GB/s) or
# RTX 4090 (>= 800 GB/s), at least 90 GB of host memory, at least 150 GB of free disk; VR the first probe's ratio B_p /
# B_c at least 0.6 (the scope); V1 base's loss within 2% of 0.190 (5090) or 0.1963 (4090); V2 base's three rounds within
# 2% of each other; V3 every configuration's loss within 1% of base's (reported; a configuration failing it is left
# out). A host is valid if it passes V0, V0c, VR, V1 and V2. Every host that starts is reported.
# Hosts: as job 121 (single-GPU, whole-machine on-demand RTX 5090 offers at most $1.10/h or RTX 4090 offers at most
# $0.70/h, at least 90 GB of host memory, 160 GB of disk and 300 Mb/s down, a consumer desktop CPU, not in the rental
# ledger), in order of expected cost, $/h x (1.0 h + the hours to download 65 GB at the listed speed), from a listing
# committed and pushed before each launch (jobs/ec2/list122.py). When no new offer qualifies, a launch may take a
# machine that was a valid host of jobs 109-120 (jobs/ec2/repeat_machines121.txt; its wrapper sets REPEAT=1 and V0's
# new-GPU clause is waived; every clause is also scored without such hosts). No machine of job 121 runs. Machines that
# failed V0c or VR in jobs 119-122 are skipped. At most two valid machines per Vast host; up to three run at once. The
# job stops starting hosts when six are valid, fourteen have started, its spend in the rental ledger reaches $9, the
# account's credit falls below $1.50, or 72 hours after this header was pushed, whichever comes first. Budget per host:
# 120 minutes. A smoke run (SMOKE=1: gpt-oss-20b, 2 problems x 32 steps, gates recorded, not enforced) checks the
# script on one host before launch; it is not a host of this job.
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
SMOKE=${SMOKE:-0}
MREPO=ggml-org/gpt-oss-120b-GGUF; MFILE=gpt-oss-120b-MXFP4.gguf; NL=36
CA=14; NSEQ=20; NDEC=256; ROUNDS="1 2 3"
BMIN=120; RAMMIN=90; DISKMIN=150; VRMIN=0.6
NG="base base0 bypass0 own"
if [ "$SMOKE" = 1 ]; then
  MREPO=ggml-org/gpt-oss-20b-GGUF; MFILE=gpt-oss-20b-MXFP4.gguf; NL=24; NSEQ=2; NDEC=32; DISKMIN=60; RAMMIN=24
fi
echo "SMOKE=$SMOKE model $MFILE C=$CA problems $NSEQ x $NDEC g=[$NG] VRMIN=$VRMIN" | tee $OUT/mode.txt
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
  echo "V0: a GPU rented before ($UUID): a repeat host, admitted by the header's rule" | tee -a $OUT/v0.txt
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
sha256sum $PATCH $J/known_gpu_uuids.txt $J/fetch_table.py | tee $OUT/patch_sha.txt; echo "base $BASE" >> $OUT/patch_sha.txt
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
echo "build and probe done, waiting for the download ($(( $(date +%s) - T0 )) s since launch)"
wait $DLG
cat $OUT/dl_gguf.txt
F=$M/$MFILE
[ -s "$F" ] || { echo "GGUF MISSING"; exit 3; }
df -h $WORK | tail -1 > $OUT/df.txt
MB="mailbox=1:tdec=1:helpers=$H"
CG=$J/aime25_owntext_gptoss_084.jsonl; wc -l $CG
# ---- this machine's routing of the text (untimed)
t0=$(date +%s)
R 30m $EC_BIN -m $F --corpus $CG -t $CORES --n-seq $NSEQ --n-prefill 1024 --n-decode $NDEC --ncmoe 24 --lookahead $OUT/la_g.bin > $OUT/la_g.jsonl 2> $OUT/la_g.err
echo "lookahead g rc=$? in $(( $(date +%s) - t0 )) s: $(ls -la $OUT/la_g.bin | awk '{print $5}') bytes"; cat $OUT/la_g.bin.json
tail -n 5 $OUT/la_g.err > $OUT/la_g.err.tail; rm -f $OUT/la_g.err
python3 $J/la_act.py $OUT/la_g.bin $OUT/route_g.npz
[ -s $OUT/la_g.bin ] || { echo "no lookahead file: oracle runs impossible"; exit 3; }
run_round() {  # round: the configurations in a shuffled order
  local rd=$1 cfg="" n c
  local common="--corpus $CG -t $CORES --n-seq $NSEQ --n-prefill 1024 --n-decode $NDEC --no-mmap --host-experts"
  local base="slots=$CA:policy=dfa:$MB:fetch=$LAWG:paced=0:overlap=1:oracle_w=0:kappa=1"
  local base0="slots=$CA:policy=dfa:$MB:fetch=0:paced=0:overlap=1:oracle_w=0:kappa=1"
  local orc="slots=$CA:policy=oracle:oracle=$OUT/la_g.bin:$MB:kappa=1:fetch=$LAWG:overlap=1:oracle_w=0:oracle_bypass=1:paced=0"
  local orc0="slots=$CA:policy=oracle:oracle=$OUT/la_g.bin:$MB:kappa=1:fetch=0:overlap=1:oracle_w=0:oracle_bypass=1:paced=0"
  local tag="g_C${CA}_r${rd}"
  for n in $NG; do
    case $n in
      base)    c="$base" ;;
      base0)   c="$base0" ;;
      bypass0) c="$orc0" ;;
      own)     c="$orc:fetch_own=1" ;;
      *)       echo "unknown configuration $n"; return 2 ;;
    esac
    cfg="${cfg:+$cfg;}$c:stats=$OUT/st_${tag}_$n.json"
  done
  local seed=$(( ORDER_SEED + 100 * rd ))
  echo "cell $tag order-seed $seed configs: $cfg" >> $OUT/configs.txt
  local t0=$(date +%s)
  R 40m $EC_BIN -m $F $common --order-seed $seed --ec "$cfg" > $OUT/ec_${tag}.jsonl 2> $OUT/ec_${tag}.err
  local rc=$?
  echo "cell $tag rc=$rc in $(( $(date +%s) - t0 )) s"; echo "ec_${tag} $rc $(( $(date +%s) - t0 ))" >> $OUT/runs.txt
  grep "\[ec-bench\] config [0-9]*/" $OUT/ec_${tag}.err | sed 's/:stats=.*st_/ -> st_/' > $OUT/ec_${tag}.order
  grep -i "oracle\|fetch table\|own order\|helpers\|slots (" $OUT/ec_${tag}.err | head -80 > $OUT/ec_${tag}.err.oracle
  tail -n 30 $OUT/ec_${tag}.err > $OUT/ec_${tag}.err.tail; rm -f $OUT/ec_${tag}.err
}
vcheck() {  # what reference [rounds...]: V1 / V2 / V3 on the rounds run so far
  python3 - $OUT $SMOKE "$@" <<'PY'
import glob, json, os, sys
out, smoke, what, ref = sys.argv[1], sys.argv[2] == "1", sys.argv[3], float(sys.argv[4])
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
    rounds = sys.argv[5:]
    t = []
    for rd in rounds:
        p = glob.glob(f"{out}/ec_g_C*_r{rd}.jsonl")
        b = by_config(p[0]).get("base", []) if p else []
        t.append(sum(r["decode_ms"] / r["n_decode"] for r in b) / len(b) if b else float("nan"))
    v = max(t) / min(t) - 1
    bad = not (v <= 0.02)
    msg.append(f"V2 g base ms/token {' '.join(f'{x:.3f}' for x in t)}: spread {100 * v:.2f}%: {'FAIL' if bad else 'ok'}")
else:  # V3: every configuration computes the same model as base; reported, never stops the host
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
FAILED=""
for rd in $ROUNDS; do
  [ -n "$FAILED" ] && break
  run_round $rd
  if [ "$rd" = 1 ]; then
    vcheck V1 $V1REF | tee -a $OUT/validity.txt; [ ${PIPESTATUS[0]} -eq 0 ] || { echo "V1 failed: stopping" | tee -a $OUT/validity.txt; FAILED=V1; }
  fi
  if [ -z "$FAILED" ] && [ "$rd" = 2 ]; then
    vcheck V2 $V1REF 1 2 | tee -a $OUT/validity.txt; [ ${PIPESTATUS[0]} -eq 0 ] || { echo "V2 failed (two rounds): stopping" | tee -a $OUT/validity.txt; [ "$SMOKE" = 1 ] || FAILED=V2; }
  fi
done
[ -z "$FAILED" ] && vcheck V2 $V1REF 1 2 3 | tee -a $OUT/validity.txt
vcheck V3 $V1REF | tee -a $OUT/validity.txt
rm -f $F $OUT/la_g.bin
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
    base = next((v for k, v in by.items() if k.endswith("_base")), None)
    for lab, v in sorted(by.items()):
        ms = st.mean(x[0] for x in v.values()); nll = st.mean(x[1] for x in v.values())
        b = st.mean(base[s][0] for s in v if s in base) if base else float("nan")
        bn = st.mean(base[s][1] for s in v if s in base) if base else float("nan")
        stf = f"{out}/{lab}.json"
        s = json.load(open(stf)) if os.path.exists(stf) else {}
        n = max(1, s.get("steps", 1))
        print(f"{lab:26s} {1000 / ms:7.2f} tok/s {ms:6.2f} ms n={len(v)} vs ref {b / ms:.3f} loss {100 * (nll / bn - 1):+.2f}% "
              f"misses {s.get('misses', 0) / n:.2f} admits {s.get('admits', 0) / n:.2f} fetches {s.get('fetches', 0) / n:.2f}")
        if "_r1" in lab and not lab.endswith("_base"):
            line.append(f"{lab.replace('st_', '')} {b / ms:.3f}")
open(f"{out}/oneline.txt", "w").write("122 " + " ".join(line) + "\n")
PY
ONE=$(cat $OUT/oneline.txt)
python3 $J/manifest.py "$OUT" "122_fastfix@vast" "$T0" "$ONE" card=$CARD law_gptoss=$LAWG device_read_gbs=$DR memory_clock_max_mhz=$MEMCLK ratio=$RATIO helpers=$H failed=${FAILED:-none} smoke=$SMOKE repeat=$REPEAT
cat $OUT/skipped.txt 2>/dev/null
cat $OUT/validity.txt 2>/dev/null
echo "SUMMARY: $ONE"
