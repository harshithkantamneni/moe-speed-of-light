#!/bin/bash
# Job 120: the second-card test again, on new RTX 4090 machines (reviews 35a/b and 36a/b). Job 110 froze a trend, fitted
# on the RTX 5090 panel, that predicts each oracle's speed over the deployed cache from the machine's link-to-CPU ratio;
# jobs 111 and 112 tested it on five RTX 4090 machines, three of them relaunched after a void first round whose results
# were in hand. This job tests the same frozen trend (jobs/110_secondcard@vast.sh, its header) on new RTX 4090 machines,
# none relaunched, and adds the three fetches-off arms of jobs 114/115 (base0 foa0 bypass0), so that the in-step fetches'
# eviction of MIN's set is measured on a second card and on CPU families the RTX 5090 tests lacked.
# Protocol: job 112's CARD=4090 mode (jobs/112_closing@vast.sh: engine patch oracle5, gpt-oss-120b MXFP4, the job 084
# own-text corpus, 20 AIME-25 problems x 256 teacher-forced steps, this machine's routing and Nsight profiles of the
# deployed cache before any timed run, C = 14 twice then C = 32 once, one process per round in a seeded shuffled
# order), with process A = base foa bypass fetch both3p base0 foa0 bypass0 and no process B. Budget 180 minutes.
# Definitions as jobs 110-115 (X/base: base's mean time per token over the problems / X's, geometric mean over rounds;
# trend(r): job 110's frozen line in ln(r) for each configuration; the gap and shares as job 114).
# Predictions, committed before launch, on every valid machine unless stated (11%: C = 14; 25%: C = 32):
#   L1. |ln(X/base) - trend(r)| <= 0.10 for fetch and both3p at 11%, and at 25% on all but at most one
#       machine-configuration of this job; <= 0.06 for foa and bypass at 11% (job 110's predictions);
#   L2. where the first probe's fetch table is not all zero: bypass0/bypass at least 1.05 at 11%;
#   L3. on machines with ratio at least 0.45: bypass0/base0 at least 1.07 at 11%;
#   L4. base / Eq. (1) within [1.8, 3.5] at 11% (job 112's Q4).
# Rule for intervals: as job 119 (problems' bootstrap combined with the configuration's variation between rounds).
# VALIDITY: V0 the GPU's UUID in no earlier launch (jobs/ec2/known_gpu_uuids.txt), at most 48 GB of host memory in use;
# V0c one NUMA node, at most 32 usable physical cores, "4090" in the name, 23,000-25,000 MiB, device reads >= 800 GB/s,
# at least 90 GB of host memory, at least 100 GB of free disk; VR the first probe's ratio at least 0.30 (the trend was
# fitted on 0.28-1.05); V1 base's loss within 2% of 0.1963 (job 111's reference for this card); V2 base's two rounds at
# 11% within 2%; V3 reported. A host is valid if it passes V0, V0c, VR, V1 and V2. Every host that starts is reported.
# Hosts: single-GPU, whole-machine (gpu_frac 1) on-demand RTX 4090 offers with at least 96 GB of host memory and 130 GB
# of disk, at most $0.80/h, at least 100 Mb/s down, whose listed CPU is a consumer desktop part (as job 119), not in the
# rental ledger, in order of expected cost, $/h x (2.0 h + the hours to download 63 GB), from a listing committed and
# pushed before each launch (jobs/ec2/list119.py). At most two valid machines per Vast host. Up to three run at once. A
# host that fails a gate or V1/V2 is replaced by the next. The job stops starting hosts when five are valid, eight have
# started, its spend in the rental ledger reaches $12, or 72 hours after this header was pushed, whichever comes first.
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
SMOKE=${SMOKE:-0}
CARD=4090
MREPO=ggml-org/gpt-oss-120b-GGUF; MFILE=gpt-oss-120b-MXFP4.gguf; CA=14; CB=32; NSEQ=20; NDEC=256
BMIN=180; NA="base foa bypass fetch both3p base0 foa0 bypass0"; PROF=1
V1REF=0.1963; DRMIN=800; VRMIN=0.30; RAMMIN=90; DISKMIN=100
KA=3; KB=2
echo "CARD=$CARD SMOKE=$SMOKE model $MFILE budgets C=$CA,$CB problems $NSEQ steps $NDEC A=[$NA]" | tee $OUT/mode.txt
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
if grep -qx "$UUID" "$JD/ec2/known_gpu_uuids.txt"; then
  gate_fail 8 "V0 FAIL: GPU rented before ($UUID), stopping"
else
  echo "V0: a GPU in no earlier launch" | tee -a $OUT/v0.txt
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
case "$GNAME" in *$CARD*) ;; *) gate_fail 4 "V0c FAIL: not an RTX $CARD (${GNAME}), stopping";; esac
[ "${GMEM:-0}" -ge 23000 ] && [ "${GMEM:-0}" -le 25000 ] || gate_fail 4 "V0c FAIL: ${GMEM} MiB of device memory (not 24 GB), stopping"
[ "${DR:-0}" -ge $DRMIN ] || gate_fail 5 "V0c FAIL: GPU reads below $DRMIN GB/s: throttled card, stopping"
RAMGB=$(awk '/Mem:/{print $2}' $OUT/free.txt); echo "host RAM ${RAMGB} GB, in use ${USEDGB} GB" | tee -a $OUT/gate.txt
[ "${RAMGB:-0}" -ge $RAMMIN ] || gate_fail 6 "V0c FAIL: ${RAMGB} GB of host memory (< $RAMMIN), stopping"
DISKGB=$(df -BG --output=avail $WORK | tail -1 | tr -dc 0-9); echo "disk free ${DISKGB} GB" | tee -a $OUT/gate.txt
[ "${DISKGB:-0}" -ge $DISKMIN ] || gate_fail 6 "V0c FAIL: ${DISKGB} GB of free disk (< $DISKMIN), stopping"
echo "V0c checked" | tee -a $OUT/v0.txt
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip python3-venv >> $OUT/apt.txt 2>&1
python3 -m pip install -q --break-system-packages huggingface_hub hf_xet numpy scipy > $OUT/pip.txt 2>&1
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
sha256sum $PATCH $J/known_gpu_uuids.txt $J/fetch_table.py | tee $OUT/patch_sha.txt; echo "base $BASE" >> $OUT/patch_sha.txt
# --- the host (downloading, otherwise idle): platform, our probe, the fetch tables, the ratios
platform
H=$(( CORES > 4 ? CORES - 2 : 2 ))
echo "helpers H=$H" | tee -a $OUT/gate.txt
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
grep -q "concurrent t=" $OUT/concur.txt || { echo "concur incomplete, retrying"; R 15m $WORK/concur > $OUT/concur.txt 2>&1; }
echo "probe 1 ended $(date -u +%FT%TZ); download running: $(kill -0 $DLG 2>/dev/null && echo yes || echo no)" | tee $OUT/probe1_when.txt
LAWG=$(python3 $J/fetch_table.py $OUT/concur.txt $H 13253760 4 $OUT/fetch_table_law_gptoss.json 37)
echo "law table ($(date -u +%FT%TZ)): gpt-oss $LAWG" | tee $OUT/tables.txt
[ -n "$LAWG" ] || { echo "no law table"; exit 3; }
ratio_of() { python3 -c "import json, sys; d = json.load(open(sys.argv[1])); print(f'{d[\"B_p\"] / d[\"B_c\"]:.4f}')" "$1"; }
RATIO=$(ratio_of $OUT/fetch_table_law_gptoss.json)
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
# --- PROBE: three more runs on the idle machine (recorded; the fetch tables and the gate stay the first probe's)
for i in 1 2 3; do
  t0=$(date +%s)
  R 15m $WORK/concur > $OUT/concur_idle$i.txt 2>&1
  echo "idle probe $i rc=$? in $(( $(date +%s) - t0 )) s ($(date -u +%FT%TZ))" | tee -a $OUT/probe_idle.txt
done
python3 - $OUT > $OUT/probe_compare.txt <<'PY'
import re, sys
out = sys.argv[1]
def best(f):
    t = open(f).read()
    v = [float(x) for x in re.findall(r"cpu_read_gbs t=\d+ ([\d.]+)", t)]
    v += [float(x) for x in re.findall(r"pcie_\w+_gbs [^\n]*? ([\d.]+)\n", t)]
    v += [float(x) for x in re.findall(r"sum ([\d.]+)", t)]
    return max(v) if v else float("nan")
b1 = best(f"{out}/concur.txt")
bi = [best(f"{out}/concur_idle{i}.txt") for i in (1, 2, 3)]
print(f"first probe highest {b1:.2f} GB/s; idle runs {' '.join(f'{x:.2f}' for x in bi)}; idle best / first {max(bi) / b1:.4f}; idle spread {max(bi) / min(bi):.4f}")
PY
cat $OUT/probe_compare.txt
MB="mailbox=1:tdec=1:helpers=$H"
PACED="paced=1:chunk=16777216:max_pending=64"
CG=$J/aime25_owntext_gptoss_084.jsonl; wc -l $CG
lookahead() {  # tag gguf corpus ncmoe: this machine's routing of the teacher-forced text (untimed)
  local tag=$1 g=$2 corpus=$3 n=$4 t0=$(date +%s)
  R 30m $EC_BIN -m $g --corpus $corpus -t $CORES --n-seq $NSEQ --n-prefill 1024 --n-decode $NDEC --ncmoe $n --lookahead $OUT/la_$tag.bin > $OUT/la_$tag.jsonl 2> $OUT/la_$tag.err
  echo "lookahead $tag rc=$? in $(( $(date +%s) - t0 )) s: $(ls -la $OUT/la_$tag.bin | awk '{print $5}') bytes"; cat $OUT/la_$tag.bin.json
  tail -n 5 $OUT/la_$tag.err > $OUT/la_$tag.err.tail; rm -f $OUT/la_$tag.err
}
profiles() {  # model-tag gguf corpus law C...: Nsight profiles of the deployed cache, one problem, 24 decode tokens
  local m=$1 g=$2 corpus=$3 law=$4; shift 4
  [ "$PROF" = 1 ] || return 0
  local labs="" C
  for C in "$@"; do
    prof_run $m$C $EC_BIN -m $g --corpus $corpus -t $CORES --n-prefill 1024 --n-decode 24 --no-mmap --host-experts --seqs 1,0 \
      --ec "slots=$C:policy=dfa:$MB:kappa=1:fetch=$law:paced=0:overlap=1:oracle_w=0:seqs=1/0"
    labs="$labs $m$C"
  done
  python3 - $OUT $labs >> $OUT/g_prof.jsonl <<'PY'
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
  echo "profiles written $(date -u +%FT%TZ), before any timed run of this model:"; tail -1 $OUT/g_prof.jsonl
}
run_round() {  # model-tag C round process "names": one process, the named configurations in a shuffled order
  local m=$1 C=$2 rd=$3 pr=$4 names=$5
  local g corpus la law mb
  case $m in
    g) g=$F; corpus=$CG; la=$OUT/la_g.bin; law=$LAWG; mb=$MB ;;
  esac
  local common="--corpus $corpus -t $CORES --n-seq $NSEQ --n-prefill 1024 --n-decode $NDEC --no-mmap --host-experts"
  local base="slots=$C:policy=dfa:$mb:fetch=$law:paced=0:overlap=1:oracle_w=0"
  local orc="slots=$C:policy=oracle:oracle=$la:$mb:kappa=1:fetch=$law:overlap=1:oracle_w=0"
  local base0="slots=$C:policy=dfa:$mb:fetch=0:paced=0:overlap=1:oracle_w=0"
  local orc0="slots=$C:policy=oracle:oracle=$la:$mb:kappa=1:fetch=0:overlap=1:oracle_w=0"
  local tag="${m}_C${C}_r${rd}${pr}"
  local cfg="" n c
  for n in $names; do
    case $n in
      base)    c="$base:kappa=1" ;;
      base0)   c="$base0:kappa=1" ;;
      foa0)    c="$base0:kappa=1:fetch_on_admit=1" ;;
      bypass0) c="$orc0:oracle_bypass=1:paced=0" ;;
      foa)     c="$base:kappa=1:fetch_on_admit=1" ;;
      bypass)  c="$orc:oracle_bypass=1:paced=0" ;;
      fetch)   c="$orc:oracle_bypass=1:oracle_fetch=1:paced=0" ;;
      both3p)  c="$orc:oracle_bypass=0:oracle_lead=3:oracle_fetch=1:$PACED" ;;
      *)       echo "unknown configuration $n"; return 2 ;;
    esac
    cfg="${cfg:+$cfg;}$c:stats=$OUT/st_${tag}_$n.json"
  done
  local off=0
  local seed=$(( ORDER_SEED + 100 * rd + C + off ))
  echo "cell $tag order-seed $seed configs: $cfg" >> $OUT/configs.txt
  local t0=$(date +%s)
  R 60m $EC_BIN -m $g $common --order-seed $seed --ec "$cfg" > $OUT/ec_${tag}.jsonl 2> $OUT/ec_${tag}.err
  local rc=$?
  echo "cell $tag rc=$rc in $(( $(date +%s) - t0 )) s"; echo "ec_${tag} $rc $(( $(date +%s) - t0 ))" >> $OUT/runs.txt
  grep "\[ec-bench\] config [0-9]*/" $OUT/ec_${tag}.err | sed 's/:stats=.*st_/ -> st_/' > $OUT/ec_${tag}.order
  grep -i "oracle\|plan\|fetch table\|fetch-on-admit\|prefetch\|shuffled\|kappa\|helpers" $OUT/ec_${tag}.err | head -60 > $OUT/ec_${tag}.err.oracle
  tail -n 30 $OUT/ec_${tag}.err > $OUT/ec_${tag}.err.tail; rm -f $OUT/ec_${tag}.err
}
vcheck() {  # what model-tag reference [C rounds...]: V1 / V2 / V3 on the rounds run so far
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
        b = by_config(f).get("base", []) or by_config(f).get("base0", [])
        if not b:
            msg.append(f"V1 {os.path.basename(f)}: no base rows"); bad = True; continue
        nll = sum(r["nll_sum"] / r["nll_n"] for r in b) / len(b)
        ok = smoke or abs(nll / ref - 1) <= 0.02
        bad |= not ok
        msg.append(f"V1 {os.path.basename(f)} base loss {nll:.4f} vs {ref}: {'ok' if ok else 'FAIL'}" + (" (smoke: not checked)" if smoke else ""))
elif what == "V2":
    C, rounds = sys.argv[6], sys.argv[7:]
    t = []
    for rd in rounds:
        p = f"{out}/ec_{m}_C{C}_r{rd}A.jsonl"
        b = by_config(p).get("base", []) if os.path.exists(p) else []
        t.append(sum(r["decode_ms"] / r["n_decode"] for r in b) / len(b) if b else float("nan"))
    v = max(t) / min(t) - 1
    bad = not (v <= 0.02)
    msg.append(f"V2 {m} C{C} base ms/token {' '.join(f'{x:.3f}' for x in t)}: spread {100 * v:.2f}%: {'FAIL' if bad else 'ok'}")
else:  # V3: every configuration computes the same model as base in its process; reported, never stops the host
    for f in files:
        by = by_config(f)
        k0 = "base" if "base" in by else "base0"
        if k0 not in by:
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
# ---- gpt-oss-120b: routing, profiles, then the timed rounds
lookahead g $F $CG 24
[ -s $OUT/la_g.bin ] || { echo "no lookahead file: oracle runs impossible"; exit 3; }
profiles G $F $CG $LAWG $CA $CB
FAILED=""
run_round g $CA 1 A "$NA"
vcheck V1 g $V1REF | tee -a $OUT/validity.txt; [ ${PIPESTATUS[0]} -eq 0 ] || { echo "V1 failed: stopping" | tee -a $OUT/validity.txt; FAILED=V1; }
vcheck V3 g $V1REF | tee -a $OUT/validity.txt
if [ -z "$FAILED" ]; then
  run_round g $CA 2 A "$NA"
  vcheck V2 g $V1REF $CA 1 2 | tee -a $OUT/validity.txt; [ ${PIPESTATUS[0]} -eq 0 ] || { echo "V2 failed: stopping" | tee -a $OUT/validity.txt; [ "$SMOKE" = 1 ] || FAILED=V2; }
fi
if [ -z "$FAILED" ]; then
  run_round g $CB 1 A "$NA"
  vcheck V1 g $V1REF | tee -a $OUT/validity.txt
  vcheck V3 g $V1REF | tee -a $OUT/validity.txt
fi
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
    base = next((v for k, v in by.items() if k.endswith("_base")), None) or next((v for k, v in by.items() if k.endswith("_base0")), None)
    for lab, v in sorted(by.items()):
        ms = st.mean(x[0] for x in v.values()); nll = st.mean(x[1] for x in v.values())
        b = st.mean(base[s][0] for s in v if s in base) if base else float("nan")
        bn = st.mean(base[s][1] for s in v if s in base) if base else float("nan")
        stf = f"{out}/{lab}.json"
        s = json.load(open(stf)) if os.path.exists(stf) else {}
        n = max(1, s.get("steps", 1))
        print(f"{lab:26s} {1000 / ms:7.2f} tok/s {ms:6.2f} ms n={len(v)} vs base {b / ms:.3f} loss {100 * (nll / bn - 1):+.2f}% "
              f"misses {s.get('misses', 0) / n:.2f} admits {s.get('admits', 0) / n:.2f} prefetches {s.get('prefetches', 0) / n:.2f} "
              f"fetches {s.get('fetches', 0) / n:.2f} reads/tok {(s.get('misses', 0) + s.get('admits', 0) + s.get('prefetches', 0)) / n:.2f}")
        if "_r1" in lab and not lab.endswith(("_base", "_base0")):
            line.append(f"{lab.replace('st_', '')} {b / ms:.3f}")
open(f"{out}/oneline.txt", "w").write("120 " + " ".join(line) + "\n")
PY
ONE=$(cat $OUT/oneline.txt)
python3 $J/manifest.py "$OUT" "120_secondcard2@vast" "$T0" "$ONE" card=$CARD law_gptoss=$LAWG device_read_gbs=$DR memory_clock_max_mhz=$MEMCLK ratio=$RATIO helpers=$H failed=${FAILED:-none} smoke=$SMOKE
cat $OUT/skipped.txt 2>/dev/null
cat $OUT/validity.txt 2>/dev/null
echo "SUMMARY: $ONE"
