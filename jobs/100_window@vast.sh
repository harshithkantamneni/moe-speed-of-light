#!/bin/bash
# Job 100: the window on the deployed read path, and a probe-only test of the calibrated time model, on new hosts.
#
# Two questions the reviews of job 099 raised. (1) Every window of job 099 extends "admit every miss", so it fetches
# each admission over PCIe in the step, the path that loses on slow links. What do W tokens of foresight buy on the
# deployed path instead (decayed frequency, kappa 1; the CPU serves the miss and the admission is copied into its slot
# in the background)? New engine mode (patch oracle3): LLAMA_EC_ORACLE_HYBRID without ORACLE_FETCH. On a CPU build it
# equals the deployed policy counter for counter with no window (oracle_w = -1) at C = 4 and 12 of the tiny test model,
# and at C = 8 up to one admission at the last step of the trace, where the oracle has no next record.
# (2) Does the time model predict a new host from its probe alone? jobs/ec2/predict_100.py computes, after the probe
# and before any timed run, the time per token of the four in-step states whose counters do not depend on the host,
# with G frozen at the median of the 14 earlier hosts' fits; its output is $OUT/predict_probe.json (timestamped).
# Back-tested on the 15 earlier host-cells: median error 2.0%, 90th percentile 10.4%; aa and w4 within 3.4% except
# one host (+7.7 to +11.2%); w16 and fetch under-predicted by up to 18% where the link is slower than the CPU.
#
# gpt-oss-120b at C = 14 and 32, 20 AIME-25 problems x 256 teacher-forced steps (as job 099), 11 configurations per cell:
#   base foa aa bypass fetch both3p    as job 099 (the deployed cache; single read; admit every miss; MIN 2 reads; MIN
#                                      1 read in the step; MIN prefetched, paced, lead 3)
#   w4 w16                             aa + exact foresight of 4 / 16 steps, fetched in the step (as job 099)
#   b4 b16                             the deployed policy + exact foresight of 4 / 16 steps, background copies (new)
#   b8r5                               the same with 8 steps, each future expert kept with probability 0.5 (new)
# Hosts: Pd of job 099 again (Vast offer 51748728, i9-13900KF, link 28 / CPU 71: a relaunch of the same machine), and
# new RTX 5090 desktop hosts chosen to span the link/CPU ratio (a Ryzen 7 5700X3D, a Ryzen 9 9950X, a Core i5-12400F).
#
# Predictions, committed before launch (per host unless stated):
#   1. engine counters (misses, in-step fetches per step) of aa, w4, w16 and fetch equal the job 099 panel means to
#      within 0.5%;
#   2. probe-only model (predict_probe.json): over all new host-cells, the median |error| of the eight predicted times
#      is <= 5%; aa and w4 within 12% at every host-cell; fetch under-predicted (predicted < measured) at every
#      host-cell whose link/CPU ratio is below 0.7;
#   3. with the same frozen G and each run's own counters, the model gets the sign of fetch/base, foa/base and aa/base
#      right at every host-cell where it predicts |ratio - 1| >= 0.05;
#   4. the window on the deployed path never loses and gains less than MIN with two reads gains plus 0.03:
#      b16/base in [1.00, 1.25], b4/base in [0.98, 1.12], b8r5/base in [0.95, 1.08], b16/base <= bypass/base + 0.03;
#      monotone: b4 <= b16 (time per token b4 >= b16) at every host-cell;
#   5. the path decides which window pays: at gpt-oss 11%, b16/base > w16/base on hosts with link/CPU < 0.5, and
#      w16/base > b16/base on hosts with link/CPU >= 0.9;
#   6. the relaunch of Pd: base time per token within 3% of job 099d's at both cells, and foa/base, aa/base, fetch/base,
#      both3p/base, w4/base, w16/base each within 0.03 of job 099d's;
#   7. host plan at most 150 us per step for every window and for fetch;
#   8. the engine's misses per step of b4, b16 and b8r5 are within 10% of the replay's (value_map.replay, hybrid with
#      kappa 1, which admits at once and has no fetch table; job 099's deployed state was within 2% of its replay).
# Budget: the job stops starting steps 2.5 h after launch.
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
export BASE; export -f build_tree platform getmodel
QWEN=0
T0=$(date +%s); BUDGET=$(( 150 * 60 )); RESERVE=600
ORDER_SEED=$(( $(date +%s) % 100000 )); echo "order seed base $ORDER_SEED; QWEN=$QWEN" | tee $OUT/order_seed.txt
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
free -g > $OUT/free.txt; RAMGB=$(awk '/Mem:/{print $2}' $OUT/free.txt); echo "host RAM ${RAMGB} GB" | tee -a $OUT/gate.txt
DISKGB=$(df -BG --output=avail $WORK | tail -1 | tr -dc 0-9); echo "disk free ${DISKGB} GB" | tee -a $OUT/gate.txt
[ "${DISKGB:-0}" -ge $(( QWEN == 1 ? 190 : 80 )) ] || { echo "not enough disk for the models, stopping"; exit 6; }
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket" | tee $OUT/cpu.txt
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip >> $OUT/apt.txt 2>&1
python3 -m pip install -q --break-system-packages uv huggingface_hub hf_xet numpy numba > $OUT/pip.txt 2>&1
for i in 1 2 3; do python3 -c "import numba" 2>/dev/null && break; python3 -m pip install -q --break-system-packages numba >> $OUT/pip.txt 2>&1; done
python3 -c "import numba; print('numba', numba.__version__)" | tee -a $OUT/pip.txt
CORES=$(usable_cores); echo "usable physical cores $CORES (nproc $(nproc))" | tee $OUT/cores.txt
# --- downloads first; the build and the probe run meanwhile
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
if [ "$QWEN" = 1 ]; then
  ( t0=$(date +%s); python3 -c "import sys; from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-30B-A3B', local_dir=sys.argv[1])" "$HFQ"
    echo "qwen3 hf download rc=$? in $(( $(date +%s) - t0 )) s"; du -sh "$HFQ" ) > $OUT/dl_qwen3.txt 2>&1 &
  DLQ=$!
  VE=$WORK/venv
  ( uv venv -q $VE && . $VE/bin/activate && uv pip install -q torch --index-url https://download.pytorch.org/whl/cpu &&
    uv pip install -q numpy transformers sentencepiece safetensors protobuf ) > $OUT/venv_install.txt 2>&1 &
  VEP=$!
fi
build_tree $WORK/lc-ec "$J/llama.cpp-expert-cache-4da6337-oracle3.patch" llama-ec-bench
EC_BIN=$WORK/lc-ec/build/bin/llama-ec-bench; ls -la $EC_BIN || { echo "no ec-bench binary"; exit 3; }
sha256sum $J/llama.cpp-expert-cache-4da6337-oracle3.patch $J/value_map.py | tee $OUT/patch_sha.txt; echo "base $BASE" >> $OUT/patch_sha.txt
# --- the host (idle): platform, our probe, the law's tables
platform
H=$(( CORES > 4 ? CORES - 2 : 2 ))
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
grep -q "concurrent t=" $OUT/concur.txt || { echo "concur incomplete, retrying"; R 15m $WORK/concur > $OUT/concur.txt 2>&1; }
LAWG=$(python3 $J/fetch_table.py $OUT/concur.txt $H 13253760 4 $OUT/fetch_table_law_gptoss.json 37)
LAWQ=$(python3 $J/fetch_table.py $OUT/concur.txt $H 9437184 8 $OUT/fetch_table_law_qwen3.json 48)
echo "law tables ($(date -u +%FT%TZ)): gpt-oss $LAWG; qwen3 $LAWQ" | tee $OUT/tables.txt
( cd $J && python3 predict_100.py $OUT/concur.txt $H ) > $OUT/predict_probe.json 2>&1
echo "probe-only predictions written $(date -u +%FT%TZ), before any timed run:"; cat $OUT/predict_probe.json
[ -n "$LAWG" ] && [ -n "$LAWQ" ] || { echo "no law table"; exit 3; }
echo "builds and probe done, waiting for the downloads ($(( $(date +%s) - T0 )) s since launch)"
wait $DLG
cat $OUT/dl_gguf.txt
F=$M/gpt-oss-120b-MXFP4.gguf
[ -s "$F" ] || { echo "GGUF MISSING"; exit 3; }
MB="mailbox=1:tdec=1:helpers=$H:kappa=1"
PACED="paced=1:chunk=16777216:max_pending=64"
AA="kappa=-1000000000"
lookahead() {  # tag gguf corpus ncmoe: record the routing of the teacher-forced text on this machine (untimed)
  local tag=$1 g=$2 corpus=$3 n=$4
  local t0=$(date +%s)
  R 30m $EC_BIN -m $g --corpus $corpus -t $CORES --n-seq $NSEQ --n-prefill 1024 --n-decode 256 --ncmoe $n --lookahead $OUT/la_$tag.bin > $OUT/la_$tag.jsonl 2> $OUT/la_$tag.err
  echo "lookahead $tag rc=$? in $(( $(date +%s) - t0 )) s: $(ls -la $OUT/la_$tag.bin | awk '{print $5}') bytes"; cat $OUT/la_$tag.bin.json
  tail -n 5 $OUT/la_$tag.err > $OUT/la_$tag.err.tail; rm -f $OUT/la_$tag.err
}
run_cell() {  # tag gguf corpus C law "names": the named configurations at one cell, in a shuffled order
  local tag=$1 g=$2 corpus=$3 C=$4 law=$5 names=$6
  local common="--corpus $corpus -t $CORES --n-seq $NSEQ --n-prefill 1024 --n-decode 256 --no-mmap --host-experts"
  local LA=$OUT/la_$tag.bin
  local base="slots=$C:policy=dfa:$MB:fetch=$law:paced=0:overlap=1:oracle_w=0"
  local orc="slots=$C:policy=oracle:oracle=$LA:$MB:fetch=$law:overlap=1:oracle_w=0"
  local hyb="$orc:oracle_bypass=1:oracle_fetch=1:paced=0:oracle_hybrid=1:$AA"
  local bgw="$orc:oracle_bypass=1:paced=0:oracle_hybrid=1"
  local st="$OUT/st_${tag}_C${C}"
  local cfg="" n c
  for n in $names; do
    case $n in
      base)   c="$base" ;;
      foa)    c="$base:fetch_on_admit=1" ;;
      aa)     c="$base:fetch_on_admit=1:$AA" ;;
      bypass) c="$orc:oracle_bypass=1:paced=0" ;;
      fetch)  c="$orc:oracle_bypass=1:oracle_fetch=1:paced=0" ;;
      both3p) c="$orc:oracle_bypass=0:oracle_lead=3:oracle_fetch=1:$PACED" ;;
      w1)     c="$hyb:oracle_w=1" ;;
      w2)     c="$hyb:oracle_w=2" ;;
      w4)     c="$hyb:oracle_w=4" ;;
      w16)    c="$hyb:oracle_w=16" ;;
      w8r5)   c="$hyb:oracle_w=8:oracle_recall=0.5:oracle_fill=1:oracle_seed=1" ;;
      allr5)  c="$hyb:oracle_w=0:oracle_recall=0.5:oracle_fill=1:oracle_seed=1" ;;
      b4)     c="$bgw:oracle_w=4" ;;
      b16)    c="$bgw:oracle_w=16" ;;
      b8r5)   c="$bgw:oracle_w=8:oracle_recall=0.5:oracle_fill=1:oracle_seed=1" ;;
    esac
    cfg="${cfg:+$cfg;}$c:stats=${st}_$n.json"
  done
  local seed=$(( ORDER_SEED + C ))
  echo "cell $tag C$C order-seed $seed configs: $cfg" >> $OUT/configs.txt
  local t0=$(date +%s)
  R 100m $EC_BIN -m $g $common --order-seed $seed --ec "$cfg" > $OUT/ec_${tag}_C${C}.jsonl 2> $OUT/ec_${tag}_C${C}.err
  local rc=$?
  echo "cell $tag C$C rc=$rc in $(( $(date +%s) - t0 )) s"; echo "ec_${tag}_C${C} $rc $(( $(date +%s) - t0 ))" >> $OUT/runs.txt
  grep "\[ec-bench\] config [0-9]*/" $OUT/ec_${tag}_C${C}.err | sed 's/:stats=.*st_/ -> st_/' > $OUT/ec_${tag}_C${C}.order
  grep -i "oracle\|fetch table\|forced\|fetch-on-admit\|shuffled\|hybrid\|degraded" $OUT/ec_${tag}_C${C}.err | head -40 > $OUT/ec_${tag}_C${C}.err.oracle
  tail -n 30 $OUT/ec_${tag}_C${C}.err > $OUT/ec_${tag}_C${C}.err.tail; rm -f $OUT/ec_${tag}_C${C}.err
}
replay() {  # tag C E "names": this machine's replay of the hybrid configurations on its own lookahead file
  local tag=$1 C=$2 E=$3 names=$4
  python3 - $OUT/la_$tag.bin $C $E "$names" $J >> $OUT/value_map_replay.jsonl <<'PY'
import json, sys
sys.path.insert(0, sys.argv[5])
from value_map import replay, load_la
la, C, E, names = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4].split()
act, seqs, meta = load_la(la)
spec = {"aa": (-1, 1.0, 0), "w1": (1, 1.0, 0), "w2": (2, 1.0, 0), "w4": (4, 1.0, 0), "w16": (16, 1.0, 0), "w8r5": (8, 0.5, 1), "allr5": (0, 0.5, 1)}
bspec = {"b4": (4, 1.0, 0), "b16": (16, 1.0, 0), "b8r5": (8, 0.5, 1)}
out = {"la": la, "C": C, "records": int(act.shape[0])}
for n in names:
    if n in spec:
        W, r, s = spec[n]
        out[n] = replay(act, seqs, E, C, W, True, r, True, s, kappa=-1e9)[0]
    if n in bspec:
        W, r, s = bspec[n]
        m, a = replay(act, seqs, E, C, W, True, r, True, s, kappa=1.0)
        out[n] = {"misses": m, "admits": a, "reads": m + a}
out["min"] = replay(act, seqs, E, C, 0, False)[0]
out["online_k1"] = replay(act, seqs, E, C, -1, True, kappa=1.0)[0]
print(json.dumps(out))
PY
  tail -1 $OUT/value_map_replay.jsonl
}
# ---- gpt-oss-120b
CG=$J/aime25_owntext_gptoss_084.jsonl; wc -l $CG
lookahead g $F $CG 24
[ -s $OUT/la_g.bin ] || { echo "no gpt-oss lookahead file: oracle runs impossible"; exit 3; }
GN="base foa aa bypass fetch both3p w4 w16 b4 b16 b8r5"
replay g 14 128 "$GN"; replay g 32 128 "$GN"
run_cell g $F $CG 14 $LAWG "$GN"
run_cell g $F $CG 32 $LAWG "$GN"
rm -f $F $OUT/la_g.bin
rm -f $OUT/la_g.bin $OUT/la_q.bin
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
        print(f"{lab:22s} {1000/ms:7.2f} tok/s (harmonic) {ms:6.2f} ms n={len(v)} speed vs base {b/ms if b else float('nan'):.3f} "
              f"reads/tok {(s.get('misses', 0) + s.get('admits', 0) + s.get('prefetches', 0)) / n:.2f} plan {s.get('oracle_plan_us_per_step', 0)} us")
        if lab.endswith(("_base", "_fetch", "_w16", "_b16")):
            line.append(f"{lab.replace('st_', '')} {1000/ms:.1f}")
open(f"{out}/oneline.txt", "w").write("100 " + " ".join(line) + "\n")
PY
ONE=$(cat $OUT/oneline.txt)
python3 $J/manifest.py "$OUT" 100_window@vast "$T0" "$ONE" law_gptoss=$LAWG law_qwen3=$LAWQ device_read_gbs=$DR memory_clock_max_mhz=$MEMCLK
cat $OUT/skipped.txt 2>/dev/null
echo "SUMMARY: $ONE"
