#!/bin/bash
# Job 118: job 115's protocol (job 114's 2x2 with the in-step fetches off, three idle probe runs after the download) on a
# third CPU family, a Ryzen 7000/8000/9000 (Zen 4 or Zen 5) desktop part (reviews 31a and 31b). Jobs 114 and 115 left
# five valid machines of two kinds: two Ryzen 9 5950Xs (Zen 3, PCIe 4, one provider) and three Intel machines
# (Core i9-14900K, Core Ultra 7 265K and 9 285K). The extra that MIN's set and reading once pay together (the interaction)
# was 21% of the gap on the two 5950Xs and -3% to +3% on the three Intel machines; the paper reads this, post hoc, as
# following the link-to-CPU ratio (0.78-0.79 against 0.54-0.57). The one other machine those jobs ran, job 114's host e
# (Ryzen 9 9950X3D, Zen 5, PCIe 5), had a ratio of 0.87 and an interaction of -2% of the gap over its two rounds, which
# disagreed by 6.1% (it failed V2 and is not in the paper's pooled numbers). A slow link (the 5950Xs' 26 GB/s over
# PCIe 4, against 34-49 GB/s on the others) explains all six machines as well as the ratio explains five. This job
# measures one valid machine of the third family; the protocol is job 115's, unchanged (jobs/115_moremachines@vast.sh:
# the same eight configurations base, base0, foa, foa0, bypass, bypass0, fetch, both3p in one process per round, C = 14
# twice, then C = 32 once; the same engine patch oracle5, model, corpus, gates V0, V0c, VR (ratio >= 0.5), V1, V2, V3,
# the same three idle probe runs), with two changes: the job's step deadline is 120 minutes (job 115: 160), to bound the
# cost; and the V0 list of GPUs rented before now holds jobs 115-117's.
# Definitions as job 114 (scripts/job114.py on main): the gap is base - Eq. (1) at the first probe's rate; the
# interaction is bypass0 + foa0 - base0 - fetch (ms per token); MIN's set alone is base0 - bypass0.
# Predictions, committed before launch, for the valid host (from jobs 114 and 115; where the third family matters, from
# job 114's invalid host e, the only earlier run of it):
#   H1. counters: MIN-1R's misses and fetches per token within 1% of jobs 109-114's (39.7 and 19.8 at 11%, 16.4 and 9.7
#       at 25%); base0 and bypass0 make no in-step fetch and their misses and admissions are within 1% of job 114's
#       (62.1 and 8.2; 43.5 and 20.4 at 11%); where the first probe's fetch table is not all zero, foa0's in-step fetches
#       per token at most 0.5x foa's at 11%;
#   H2. 11%, where the fetch table is not all zero: bypass0/bypass at least 1.05;
#   H3. bypass0/base0 at least 1.07 at 11% and at least 1.20 at 25%;
#   H4. 11%: foa0/base0 within [0.98, 1.10];
#   H5. 11%: the interaction's share of the gap within [-0.05, 0.10]: the 5950Xs' 21% does not recur. If the extra
#       followed the ratio, this machine (H7b: ratio at least 0.70) would show at least 0.10; if it follows a slow link,
#       this PCIe 5 machine shows less. We predict the second, from host e;
#   H6. 11%: base0/base within [0.92, 1.10];
#   H7. the probe: the highest reading of the three idle runs within [0.95, 1.15] of the first probe's highest reading,
#       and the three idle runs' highest readings within 5% of each other (largest over smallest at most 1.05);
#   H7b. the first probe: B_p / B_c at least 0.70 and B_p at least 40 GB/s;
#   H8. pooled with jobs 114 and 115's five valid machines (six in all; five of them already measured): MIN's set alone,
#       read twice on the CPU-only path, closes a mean share of the gap at 11% of at least 0.10, with its 95% t-interval
#       over machines above zero;
#   H9. this host: MIN's set alone closes at least 0.20 of the gap at 11% (the five: 0.13-0.23; host e: 0.47).
# Scoring as jobs 114 and 115: held where the 95% interval over problems (machines, for H8) is on the predicted side,
# held (point) where only the point estimate is, failed otherwise. One valid host is the plan; its results are single-
# machine results, and every host that starts is reported.
# Hosts: single-GPU, whole-machine RTX 5090 offers with at least 120 GB of host memory and 150 GB of disk, at most
# $1.25/h, with at least 140 Mb/s down (job 115: 150; lowered so that both third-family offers in the listing qualify; a
# slower download costs time, not validity), whose listed CPU is a Ryzen 7000, 8000 or 9000 part (not Threadripper or
# EPYC), not from Vast host 716820 and not in the rental ledger, from the listing committed with this header
# (results/118_offers_at_registration.txt, jobs/ec2/list118.py), taken in order of expected cost: $/h x (1.25 h + the
# hours to download 63 GB at the listed speed). That puts 55264264 (Ryzen 9 9950X, 205 Mb/s, $0.696/h: about $1.35)
# before 52187178 (Ryzen 7 9800X3D, 148 Mb/s, $1.067/h: about $2.35), the only two that qualify. The first starts
# alone. If it fails a gate before any timed round (V0, V0c, VR), the next by expected cost in a new listing committed and pushed before that
# launch starts, if the credit then covers its expected cost plus $0.50 (the account holds $3.30); a host stopped by V1
# or V2 is reported and not replaced. At most two start.
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
SMOKE=${SMOKE:-0}
CARD=5090
MREPO=ggml-org/gpt-oss-120b-GGUF; MFILE=gpt-oss-120b-MXFP4.gguf; CA=14; CB=32; NSEQ=20; NDEC=256
BMIN=120; NA="base base0 foa foa0 bypass bypass0 fetch both3p"; NB=""; PROF=0
V1REF=0.190; DRMIN=1500; VRMIN=0.5
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
sha256sum $PATCH $J/minadm_plan.py $J/known_gpu_uuids.txt $J/job109_uuids.txt | tee $OUT/patch_sha.txt; echo "base $BASE" >> $OUT/patch_sha.txt
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
# --- PROBE: three more runs on the idle machine (recorded; the fetch table and the gate stay the first probe's)
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
IB=$(python3 -c "import re,sys; f=sys.argv[1:]; b=lambda p: max([float(x) for x in re.findall(r'(?:cpu_read_gbs t=\d+|sum) ([\d.]+)', open(p).read())] or [0]); print(max(f, key=b))" $OUT/concur_idle1.txt $OUT/concur_idle2.txt $OUT/concur_idle3.txt)
LAWI=$(python3 $J/fetch_table.py $IB $H 13253760 4 $OUT/fetch_table_idle.json 37)
echo "idle law table (from $(basename $IB), recorded only): gpt-oss $LAWI" | tee -a $OUT/tables.txt
MB="mailbox=1:tdec=1:helpers=$H"
PACED="paced=1:chunk=16777216:max_pending=64"
CG=$J/aime25_owntext_gptoss_084.jsonl; wc -l $CG
# --- this machine's routing of the text (untimed), for the oracle configurations
t0=$(date +%s)
R 30m $EC_BIN -m $F --corpus $CG -t $CORES --n-seq $NSEQ --n-prefill 1024 --n-decode $NDEC --ncmoe 24 --lookahead $OUT/la_g.bin > $OUT/la_g.jsonl 2> $OUT/la_g.err
echo "lookahead rc=$? in $(( $(date +%s) - t0 )) s: $(ls -la $OUT/la_g.bin | awk '{print $5}') bytes"; cat $OUT/la_g.bin.json
tail -n 5 $OUT/la_g.err > $OUT/la_g.err.tail; rm -f $OUT/la_g.err
[ -s $OUT/la_g.bin ] || { echo "no lookahead file: oracle runs impossible"; exit 3; }
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
  local base0="slots=$C:policy=dfa:$MB:fetch=0:paced=0:overlap=1:oracle_w=0"
  local orc0="slots=$C:policy=oracle:oracle=$OUT/la_g.bin:$MB:kappa=1:fetch=0:overlap=1:oracle_w=0"
  local tag="C${C}_r${rd}${pr}"
  local cfg="" n c
  for n in $names; do
    case $n in
      base)        c="$base:kappa=1" ;;
      base0)       c="$base0:kappa=1" ;;
      foa0)        c="$base0:kappa=1:fetch_on_admit=1" ;;
      bypass0)     c="$orc0:oracle_bypass=1:paced=0" ;;
      bypassplan0) c="$orc0:oracle_bypass=1:paced=0:oracle_plan=$OUT/plan_g$C.bin" ;;
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
              f"plan_already {s.get('plan_already', '-')} refused {s.get('oracle_refused', '-')} kappa {s.get('kappa', '-')}")
        if "_r1" in lab and not lab.endswith("_base"):
            line.append(f"{lab.replace('st_g_', '')} {b / ms:.3f}")
open(f"{out}/oneline.txt", "w").write("118 " + " ".join(line) + "\n")
PY
ONE=$(cat $OUT/oneline.txt)
python3 $J/manifest.py "$OUT" "118_thirdcpu@vast" "$T0" "$ONE" card=$CARD law_gptoss=$LAWG device_read_gbs=$DR memory_clock_max_mhz=$MEMCLK ratio=$RATIO smoke=$SMOKE
cat $OUT/skipped.txt 2>/dev/null
cat $OUT/validity.txt 2>/dev/null
echo "SUMMARY: $ONE"
