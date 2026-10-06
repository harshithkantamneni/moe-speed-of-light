#!/bin/bash
# Job 103: is the in-step copy's loss on slow links a property of MIN, or of the greedy MIN schedule? Belady's MIN with
# bypass has many hit-optimal schedules. The engine's oracle follows the greedy one, which admits a miss whenever a
# resident is needed later and may evict it again before its next use. jobs/ec2/minadm_plan.py computes, on this host's
# own routing trace, the schedule with the fewest admissions among those with the most hits (an LP over reuse
# intervals under the engine's rule for a copy in the step), and the engine (patch oracle4, LLAMA_EC_ORACLE_PLAN)
# follows it. Same protocol as jobs 100-101 (20 AIME-25 problems x 256 teacher-forced steps, gpt-oss-120b at C = 14
# and 32); six configurations per cell, in a shuffled order: base foa bypass fetch bypassplan fetchplan, i.e. the 2x2
# (what to cache: deployed or MIN; how to load: CPU then background copy, or copy in the step) with MIN's set taken
# both ways (greedy, plan). On the 084c trace the plan admits 0.58x the greedy schedule's experts for 0.2% more misses.
#
# Predictions, committed before launch (ratio = the probe's link-to-CPU ratio B_p / B_c on the host):
#   1. counters, every host and cell: fetchplan's in-step copies per token at most 0.70x fetch's, and its misses
#      within 3% of fetch's; the engine's counters within 2% of the host's replay of each schedule (minadm_plan.py);
#   2. counters: bypassplan's host reads (misses + admissions) per token at most 0.90x bypass's;
#   3. on the host with ratio below 0.35 (103a, Pf again): fetchplan/base above 1 at gpt-oss 11% (fetch/base there was
#      0.875 and 0.86 on two launches; the calibrated model with 101a's counters puts the plan at 1.21-1.29);
#   4. every host with ratio below 0.7: fetchplan/base exceeds fetch/base by at least 0.03 at gpt-oss 11%;
#   5. every host with ratio at least 0.85: fetchplan/base within 0.05 of fetch/base at both cells;
#   6. every host and cell: bypassplan/base at least bypass/base - 0.01;
#   7. the Shapley interaction of the 2x2 (share of the gap to the bound) is smaller with the plan's set than with the
#      greedy set at gpt-oss 11% on every host;
#   8. 103a: base time per token within 3% of job 101a's at both cells, and fetch/base within 0.03 of 101a's.
# Hosts: 103a = Pf again (Core Ultra 9 285K, link 27; offer 40038866); 103b = a second 285K behind a 48 GB/s link
# (offer 50573928); 103c = a Ryzen 7 7800X3D, link 49 (offer 53510760); 103d = a Ryzen 9 5950X behind a 27 GB/s link
# (offer 52273675); 103e = a Threadripper 3970X behind a 27 GB/s link (offer 48418440).
# Budget: the job stops starting steps 2 h after launch.
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
export BASE; export -f build_tree platform getmodel
QWEN=0
T0=$(date +%s); BUDGET=$(( 120 * 60 )); RESERVE=600
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
python3 -m pip install -q --break-system-packages uv huggingface_hub hf_xet numpy numba scipy > $OUT/pip.txt 2>&1
for i in 1 2 3; do python3 -c "import numba" 2>/dev/null && break; python3 -m pip install -q --break-system-packages numba >> $OUT/pip.txt 2>&1; done
for i in 1 2 3; do python3 -c "import scipy" 2>/dev/null && break; python3 -m pip install -q --break-system-packages scipy >> $OUT/pip.txt 2>&1; done
python3 -c "import numba, scipy; print('numba', numba.__version__, 'scipy', scipy.__version__)" | tee -a $OUT/pip.txt
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
build_tree $WORK/lc-ec "$J/llama.cpp-expert-cache-4da6337-oracle4.patch" llama-ec-bench
EC_BIN=$WORK/lc-ec/build/bin/llama-ec-bench; ls -la $EC_BIN || { echo "no ec-bench binary"; exit 3; }
sha256sum $J/llama.cpp-expert-cache-4da6337-oracle4.patch $J/value_map.py $J/minadm_plan.py | tee $OUT/patch_sha.txt; echo "base $BASE" >> $OUT/patch_sha.txt
# --- the host (idle): platform, our probe, the law's tables
platform
H=$(( CORES > 4 ? CORES - 2 : 2 ))
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
grep -q "concurrent t=" $OUT/concur.txt || { echo "concur incomplete, retrying"; R 15m $WORK/concur > $OUT/concur.txt 2>&1; }
LAWG=$(python3 $J/fetch_table.py $OUT/concur.txt $H 13253760 4 $OUT/fetch_table_law_gptoss.json 37)
LAWQ=$(python3 $J/fetch_table.py $OUT/concur.txt $H 9437184 8 $OUT/fetch_table_law_qwen3.json 48)
echo "law tables ($(date -u +%FT%TZ)): gpt-oss $LAWG; qwen3 $LAWQ" | tee $OUT/tables.txt
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
      fetchplan)  c="$orc:oracle_bypass=1:oracle_fetch=1:paced=0:oracle_plan=$OUT/plan_${tag}${C}.bin" ;;
      bypassplan) c="$orc:oracle_bypass=1:paced=0:oracle_plan=$OUT/plan_${tag}${C}.bin" ;;
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
  grep -i "oracle\|fetch table\|forced\|fetch-on-admit\|shuffled\|hybrid\|degraded\|plan" $OUT/ec_${tag}_C${C}.err | head -40 > $OUT/ec_${tag}_C${C}.err.oracle
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
GN="base foa bypass fetch bypassplan fetchplan"
replay g 14 128 "$GN"; replay g 32 128 "$GN"
for C in 14 32; do
  t0=$(date +%s)
  ( cd $J && R 20m python3 minadm_plan.py $OUT/la_g.bin $C $OUT/plan_g$C.bin $CORES ) > $OUT/plan_g$C.txt 2>&1
  echo "plan C$C rc=$? in $(( $(date +%s) - t0 )) s"; cat $OUT/plan_g$C.txt
  [ -s $OUT/plan_g$C.bin ] || echo "NO PLAN for C$C: the plan configurations fall back to greedy MIN (the engine logs it)"
done
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
        if lab.endswith(("_base", "_fetch", "_fetchplan")):
            line.append(f"{lab.replace('st_', '')} {1000/ms:.1f}")
open(f"{out}/oneline.txt", "w").write("103 " + " ".join(line) + "\n")
PY
ONE=$(cat $OUT/oneline.txt)
python3 $J/manifest.py "$OUT" 103_minadm@vast "$T0" "$ONE" law_gptoss=$LAWG law_qwen3=$LAWQ device_read_gbs=$DR memory_clock_max_mhz=$MEMCLK
cat $OUT/skipped.txt 2>/dev/null
echo "SUMMARY: $ONE"
