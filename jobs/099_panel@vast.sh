#!/bin/bash
# Job 099: a panel of rented RTX 5090 hosts (the machine is the statistical unit) measuring, on each host, (a) the 2 x 2
# of admission set (online vs MIN with bypass) x reads (two vs one) with pacing nested under MIN, for an order-free
# accounting, and (b) what foresight of a given horizon and accuracy is worth in seconds: the online policy with every
# miss admitted ("aa", demand fetch with decayed-frequency victims) plus a window of W future steps' routing, exact or
# degraded (each future expert kept with probability r, else replaced by a random one), planned on the host before
# each step exactly as scripts/value_map.py replays it (equal on the CPU build at three budgets, 30 configurations,
# and aa equal to the replay's W = -1). One launch per host; configurations in fresh contexts in a seeded shuffled order.
#
# gpt-oss-120b at C = 14 and 32 (11 and 25% of each layer's experts), 20 AIME-25 problems x 256 teacher-forced steps
# after the prompt (the job 084 own-text corpus), 11 configurations per cell:
#   base     deployed: decayed frequency, kappa 1, the law's FETCH table; CPU serves misses, admissions copied (two reads)
#   foa      the same with admissions fetched in the step (one read)
#   aa       foa with kappa -1e9: every miss fetched into the lowest-score slot (demand fetch; = hybrid W = -1)
#   bypass   MIN with bypass, admissions copied in the background (two reads)
#   fetch    MIN with bypass, admissions fetched in the step (one read)
#   both3p   fetch + the paced single-read prefetch, lead 3 (the best state of jobs 095-097)
#   w1 w4 w16        aa + exact foresight of the next 1 / 4 / 16 steps (LLAMA_EC_ORACLE_HYBRID, kappa -1e9)
#   w8r5 allr5       aa + foresight of 8 steps / the rest of the problem, each future expert kept with probability 0.5
# QWEN=1 (launch env): also Qwen3-30B-A3B BF16 at C = 16 (12.5%) with base, foa, aa, fetch, w2, w8r5.
#
# Replay on the AIME routing of jobs 084b/084c (prereg/value_map.json and the job's own replay, value_map_replay.json,
# computed here from this machine's lookahead file before any timed run), reads per token:
#   gpt-oss C14: aa 58.64, w1 55.07, w4 48.58, w16 39.72, w8r5 53.0, allr5 50.4, MIN 39.26 (online kappa 1: 61.13)
#   gpt-oss C32: aa 28.22, w1 27.34, w4 25.23, w16 20.23, w8r5 26.5, allr5 22.6, MIN 16.03 (29.81)
#   Qwen3 C16:   aa 150.0, w2 124.9, w8r5 128.9, MIN 100.4 (173.1)
# i.e. shares of the aa -> MIN read gap: C14 w1 .18 w4 .52 w16 .98 w8r5 .29 allr5 .43; C32 .07 .25 .66 .14 .46;
# Qwen3 C16 w2 .51 w8r5 .42.
#
# Predictions, committed before launch (per host unless stated; "share" = (t_aa - t_x) / (t_aa - t_fetch) in ms per
# token, means over problems):
#   1. every hybrid configuration's engine reads per token are within 4% of this machine's replay of the same
#      configuration on its own lookahead file, and aa's within 4% of the replay's W = -1;
#   2. the time share tracks the read share: |time share - read share| <= 0.15 at >= 80% of the hybrid host-cells
#      (pooled over hosts), and the time shares increase with W (w1 < w4 < w16) at both cells on >= 80% of hosts;
#   3. exact foresight of 16 steps recovers >= 0.80 of aa -> fetch at C14 and 0.45-0.85 at C32; 4 steps 0.35-0.70 at
#      C14 and 0.12-0.40 at C32; 1 step <= 0.30 at both;
#   4. degraded foresight (r = 0.5): w8r5 0.15-0.45 at C14 and <= 0.30 at C32; allr5 0.30-0.60 at both cells;
#   5. the 2 x 2 (base, foa, bypass, fetch) shows a positive interaction at both cells on every host: MIN's set is
#      worth more with one read (foa -> fetch) than with two (base -> bypass), and reading the online policy's
#      admissions once (base -> foa) changes the time by at most 5% on every host;
#   6. fetch / base 1.10-1.55 and both3p / base 1.25-1.90 at both cells on every host; both3p > fetch > foa (means);
#   7. aa / foa (time): within [0.92, 1.04] on hosts whose probed link rate is >= 40 GB/s; below 0.95 on hosts whose link
#      is below 32 GB/s (every read goes over the link);
#   8. machine variance dominates: across hosts the spread (max - min) of fetch / base exceeds 3x the median within-host
#      95% half-width of fetch / base;
#   9. the host plan costs at most 150 us per step (oracle_plan_us_per_step) in every hybrid and fetch configuration.
#  10. (QWEN=1) Qwen3 C16: w2 share 0.30-0.70, w8r5 0.25-0.60; fetch / base 1.15-1.55.
# Hosts: RTX 5090 with a desktop CPU and >= 90 GB of RAM, one launch each, CPU class recorded; disk 200 GB.
# Budget: the job stops starting steps 2.5 h after launch (3.5 h with QWEN=1).
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
export BASE; export -f build_tree platform getmodel
QWEN=${QWEN:-0}
T0=$(date +%s); BUDGET=$(( QWEN == 1 ? 210 * 60 : 150 * 60 )); RESERVE=600
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
build_tree $WORK/lc-ec "$J/llama.cpp-expert-cache-4da6337-oracle2.patch" llama-ec-bench
EC_BIN=$WORK/lc-ec/build/bin/llama-ec-bench; ls -la $EC_BIN || { echo "no ec-bench binary"; exit 3; }
sha256sum $J/llama.cpp-expert-cache-4da6337-oracle2.patch $J/value_map.py | tee $OUT/patch_sha.txt; echo "base $BASE" >> $OUT/patch_sha.txt
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
out = {"la": la, "C": C, "records": int(act.shape[0])}
for n in names:
    if n in spec:
        W, r, s = spec[n]
        out[n] = replay(act, seqs, E, C, W, True, r, True, s, kappa=-1e9)[0]
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
GN="base foa aa bypass fetch both3p w1 w4 w16 w8r5 allr5"
replay g 14 128 "$GN"; replay g 32 128 "$GN"
run_cell g $F $CG 14 $LAWG "$GN"
run_cell g $F $CG 32 $LAWG "$GN"
rm -f $F $OUT/la_g.bin
# ---- Qwen3-30B-A3B BF16 at C = 16 (QWEN=1)
if [ "$QWEN" = 1 ]; then
  df -h $WORK | tail -1 >> $OUT/df.txt
  wait $DLQ $VEP; cat $OUT/dl_qwen3.txt
  GQ=$M/Qwen3-30B-A3B-BF16.gguf
  ( t0=$(date +%s); PYTHONPATH=$WORK/lc-ec/gguf-py timeout 30m $VE/bin/python $WORK/lc-ec/convert_hf_to_gguf.py "$HFQ" --outtype bf16 --outfile $GQ
    echo "convert rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $GQ ) > $OUT/convert_qwen3.txt 2>&1
  tail -2 $OUT/convert_qwen3.txt; rm -rf "$HFQ"
  if [ -s "$GQ" ]; then
    CQ=$J/aime25_owntext_qwen3_084b.jsonl
    lookahead q $GQ $CQ 36
    if [ -s $OUT/la_q.bin ]; then
      QN="base foa aa fetch w2 w8r5"
      replay q 16 128 "$QN"
      run_cell q $GQ $CQ 16 $LAWQ "$QN"
    else
      echo "no Qwen3 lookahead file: Qwen3 runs skipped" | tee -a $OUT/skipped.txt
    fi
    rm -f $OUT/la_q.bin
  else
    echo "Qwen3 GGUF missing: Qwen3 part skipped" | tee -a $OUT/skipped.txt
  fi
fi
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
        if lab.endswith(("_base", "_aa", "_fetch", "_w4")):
            line.append(f"{lab.replace('st_', '')} {1000/ms:.1f}")
open(f"{out}/oneline.txt", "w").write("099 " + " ".join(line) + "\n")
PY
ONE=$(cat $OUT/oneline.txt)
python3 $J/manifest.py "$OUT" 099_panel@vast "$T0" "$ONE" law_gptoss=$LAWG law_qwen3=$LAWQ device_read_gbs=$DR memory_clock_max_mhz=$MEMCLK qwen=$QWEN
cat $OUT/skipped.txt 2>/dev/null
echo "SUMMARY: $ONE"
