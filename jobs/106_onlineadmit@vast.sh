#!/bin/bash
# Job 106: (1) the sum law with an overlap term, registered on both models at both budgets; (2) online admission rules
# on the deployed path, the fewest-admission lesson made realisable; (3) launch-level intervals on the slow-link
# machines. Engine: patch oracle5 (= oracle4 + the learned order allowed on the deployed path, with a margin).
# Protocol as jobs 100-105: 20 AIME-25 problems x 256 teacher-forced steps per configuration, configurations in a
# shuffled order per process, one process per round ("launch"): gpt-oss-120b at C = 14 (11%) in 3 rounds and C = 32
# (25%) in 2 rounds; Qwen3-30B-A3B BF16 at C = 16 (12.5%) and C = 32 (25%), 1 round each.
#   base   the deployed cache (decayed count, kappa 1, the host's fetch table; misses on the CPU or fetched in the
#          step, admissions copied in the background: each admission a second read)
#   dk     base with the admission margin kappa chosen on other text: kappa 3 (gpt-oss 11%), 2 (gpt-oss 25%),
#          3 (Qwen3 12.5%), 2 (Qwen3 25%) (scripts/online_admit.py: the deployed path's misses + admissions per token
#          minimised over kappa in {0 .. 8} on the model's mixed-domain trace, no AIME text; prereg/online_admit.json)
#   lrn    base with job 097's learned reuse predictor ordering admissions and victims (LLAMA_EC_LEARNED; the model
#          files are recomputed here and must hash to job 097's), admitting a miss when its predicted reuse beats the
#          victim's by a margin chosen the same way: 0.4, 0.4, 0.4, 0.3
#   fetch, fetchplan   MIN with bypass copied in the step, greedy and fewest-admission sets (jobs 103-105), gpt-oss 11%
# Replay on the AIME routing (prereg/online_admit.json; background-admission semantics, no in-step fetch installs):
# reads per token deployed / dk / lrn = gpt-oss 11% 70.8 / 66.7 / 65.5, 25% 40.0 / 35.7 / 34.0; Qwen3 12.5% 196.6 /
# 189.8 / 186.2, 25% 115.1 / 108.9 / 105.6; MIN 39.0, 15.5, 100.4, 44.3.
# Before any timed run, Nsight Systems profiles base at each model and budget (one problem, 24 decode tokens, the
# decode-tail GPU trace kept): G = the GPU's own kernel time per token (every kernel but the helper wait and the
# expert copies). The law, registered here with its overlap term:
#   T = G + (M + A (1 - G/T)) S / B_host,  solved for T,
# M the misses and A the background admissions per token (the run's counters), S the expert's bytes (13,253,760 and
# 9,437,184), B_host the probe's highest host rate. The background copies spread over the next step; the share that
# runs while the GPU computes (G/T) takes no host time. The plain law (job 105) is the same without the (1 - G/T).
# On jobs 093-105 (34 launches, exploratory): plain law within 6% at gpt-oss 11% / 25% on 32 / 18 of 34; the overlap
# form on 29 / 33 of 34 (median -1.2% / +2.2%); on the 10 earlier Qwen3 launches with G = 5.8 ms (job 079's profile at
# 43.75%), both forms within 5%.
# Predictions, committed before launch (ratio = the probe's B_p / B_c):
#   1. G_prof: gpt-oss 4.0-5.0 ms at both budgets on every host; Qwen3 4.5-6.5 ms at both budgets;
#   2. the overlap form is within 6% of the deployed cache's time (base, mean over rounds) at every host, model and
#      budget, and the median |error| over all of them is at most 4%; at gpt-oss 25% its median |error| is at most
#      half the plain law's;
#   3. dk reads fewer host experts per token than base (misses + admissions, counters) at every gpt-oss cell, and
#      dk/base speed is >= 1.01 at gpt-oss 11% and 25% on every host, pooled median >= 1.02; at Qwen3 >= 1.00;
#   4. lrn reads at most as many as dk at every gpt-oss cell; lrn/base >= 1.00 at every gpt-oss cell; its host time
#      learned_us_per_step <= 400 us (gpt-oss) and <= 700 us (Qwen3);
#   5. the overlap form, from each configuration's own counters (lrn: plus its learned_us_per_step), predicts dk/base
#      and lrn/base within 0.03 at every gpt-oss cell;
#   6. on hosts with ratio < 0.4: in every round at gpt-oss 11%, fetch/base < 1 and fetchplan/base > 1, and the
#      launch-level 95% interval of fetchplan/base (bootstrap over rounds, then problems) lies above 1;
#   7. fetchplan/base > 1 at gpt-oss 11% on every host (mean over rounds), and fetchplan copies <= 0.65x fetch's;
#   8. base's time per token varies by at most 2% between rounds on every host.
# Hosts: 106a = the Threadripper 9960X of 105b again (offer 53278552, ratio 0.32); 106b = Pf again (Core Ultra 9
# 285K, x8 link, 40038866); 106c = an EPYC 7302 behind a 13 GB/s link (35010867); 106d = a Xeon Platinum 8347C
# (42405157); 106e = O3's Ryzen 9 9950X3D again (51051777).
# Amended after 106a-d launched, before 106e started: O3's offer was gone; 106e = a Ryzen 9 9950X (54559478).
# Predictions unchanged.
# Budget: the job stops starting steps 3 h 20 min after launch.
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
. "$J/prof.sh"
export BASE; export -f build_tree platform getmodel
export PROF_KEEP_TAIL="G14 G32 Q16 Q32"
PATCH=$J/llama.cpp-expert-cache-4da6337-oracle5.patch
T0=$(date +%s); BUDGET=$(( 200 * 60 )); RESERVE=600
ORDER_SEED=$(( $(date +%s) % 100000 )); echo "order seed base $ORDER_SEED" | tee $OUT/order_seed.txt
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
[ "${DISKGB:-0}" -ge 150 ] || { echo "not enough disk for the models, stopping"; exit 6; }
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket" | tee $OUT/cpu.txt
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip >> $OUT/apt.txt 2>&1
python3 -m pip install -q --break-system-packages uv huggingface_hub hf_xet numpy numba > $OUT/pip.txt 2>&1
for i in 1 2 3; do python3 -c "import numba" 2>/dev/null && break; python3 -m pip install -q --break-system-packages numba >> $OUT/pip.txt 2>&1; done
python3 -c "import numba; print('numba', numba.__version__)" | tee -a $OUT/pip.txt
command -v uv >/dev/null || python3 -m pip install -q --break-system-packages uv >> $OUT/pip.txt 2>&1
PV=$WORK/planenv
for i in 1 2 3; do
  ( uv venv -q --clear $PV && uv pip install -q --python $PV/bin/python numpy scipy ) >> $OUT/planenv.txt 2>&1
  $PV/bin/python -c "import scipy" 2>/dev/null && break
done
$PV/bin/python -c "import numpy, scipy; print('plan env: numpy', numpy.__version__, 'scipy', scipy.__version__)" 2>&1 | tee -a $OUT/planenv.txt
$PV/bin/python -c "import scipy" 2>/dev/null || { echo "no plan environment: stopping before any timed run"; exit 4; }
CORES=$(usable_cores); echo "usable physical cores $CORES (nproc $(nproc))" | tee $OUT/cores.txt
# --- the learned order's model files, recomputed from the committed traces and weights; hashes must equal job 097's
LM=$WORK/learned; mkdir -p $LM
( python3 $J/learned_export.py $J/learned/gpt-oss-120b_S.npz $J/learned/learned_weights.json gpt-oss-120b_C14 $LM/learned_g14.bin
  python3 $J/learned_export.py $J/learned/gpt-oss-120b_S.npz $J/learned/learned_weights.json gpt-oss-120b_C32 $LM/learned_g32.bin
  python3 $J/learned_export.py $J/learned/qwen3-30b-a3b_fp8_S.npz $J/learned/learned_weights.json qwen3-30b-a3b_C16 $LM/learned_q16.bin
  python3 $J/learned_export.py $J/learned/qwen3-30b-a3b_fp8_S.npz $J/learned/learned_weights.json qwen3-30b-a3b_C32 $LM/learned_q32.bin
  python3 - $J/learned/learned_weights.json $LM <<'PY'
import hashlib, json, sys
w = json.load(open(sys.argv[1]))
for key, f in (("gpt-oss-120b_C14", "g14"), ("gpt-oss-120b_C32", "g32"), ("qwen3-30b-a3b_C16", "q16"), ("qwen3-30b-a3b_C32", "q32")):
    h = hashlib.sha256(open(f"{sys.argv[2]}/learned_{f}.bin", "rb").read()).hexdigest()
    print(f, "MATCH" if h == w[key]["sha256"] else f"MISMATCH {h} != {w[key]['sha256']}")
PY
) > $OUT/learned_export.txt 2>&1 &
LMP=$!
# --- downloads first; the build, the probe and the conversion venv run meanwhile
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
( t0=$(date +%s); python3 -c "import sys; from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-30B-A3B', local_dir=sys.argv[1])" "$HFQ"
  echo "qwen3 hf download rc=$? in $(( $(date +%s) - t0 )) s"; du -sh "$HFQ" ) > $OUT/dl_qwen3.txt 2>&1 &
DLQ=$!
VE=$WORK/venv
( uv venv -q $VE && . $VE/bin/activate && uv pip install -q torch --index-url https://download.pytorch.org/whl/cpu &&
  uv pip install -q numpy transformers sentencepiece safetensors protobuf ) > $OUT/venv_install.txt 2>&1 &
VEP=$!
install_nsys &
NSP=$!
build_tree $WORK/lc-ec "$PATCH" llama-ec-bench
wait $NSP; install_nsys
EC_BIN=$WORK/lc-ec/build/bin/llama-ec-bench; ls -la $EC_BIN || { echo "no ec-bench binary"; exit 3; }
sha256sum $PATCH $J/minadm_plan.py $J/learned/learned_weights.json | tee $OUT/patch_sha.txt; echo "base $BASE" >> $OUT/patch_sha.txt
# --- the host (idle): platform, our probe, the law's tables
platform
H=$(( CORES > 4 ? CORES - 2 : 2 ))
wait $LMP; cat $OUT/learned_export.txt
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
grep -q "concurrent t=" $OUT/concur.txt || { echo "concur incomplete, retrying"; R 15m $WORK/concur > $OUT/concur.txt 2>&1; }
LAWG=$(python3 $J/fetch_table.py $OUT/concur.txt $H 13253760 4 $OUT/fetch_table_law_gptoss.json 37)
LAWQ=$(python3 $J/fetch_table.py $OUT/concur.txt $H 9437184 8 $OUT/fetch_table_law_qwen3.json 48)
echo "law tables ($(date -u +%FT%TZ)): gpt-oss $LAWG; qwen3 $LAWQ" | tee $OUT/tables.txt
[ -n "$LAWG" ] && [ -n "$LAWQ" ] || { echo "no law table"; exit 3; }
echo "builds and probe done, waiting for the gpt-oss download ($(( $(date +%s) - T0 )) s since launch)"
wait $DLG
cat $OUT/dl_gguf.txt
F=$M/gpt-oss-120b-MXFP4.gguf
[ -s "$F" ] || { echo "GGUF MISSING"; exit 3; }
MB="mailbox=1:tdec=1:helpers=$H"
lrnok() { grep -q "^$1 MATCH" $OUT/learned_export.txt; }
lookahead() {  # tag gguf corpus ncmoe: record the routing of the teacher-forced text on this machine (untimed)
  local tag=$1 g=$2 corpus=$3 n=$4
  local t0=$(date +%s)
  R 30m $EC_BIN -m $g --corpus $corpus -t $CORES --n-seq $NSEQ --n-prefill 1024 --n-decode 256 --ncmoe $n --lookahead $OUT/la_$tag.bin > $OUT/la_$tag.jsonl 2> $OUT/la_$tag.err
  echo "lookahead $tag rc=$? in $(( $(date +%s) - t0 )) s: $(ls -la $OUT/la_$tag.bin | awk '{print $5}') bytes"; cat $OUT/la_$tag.bin.json
  tail -n 5 $OUT/la_$tag.err > $OUT/la_$tag.err.tail; rm -f $OUT/la_$tag.err
}
run_round() {  # tag gguf corpus C law kappa_dk margin learned_file round "names": one process, shuffled order
  local tag=$1 g=$2 corpus=$3 C=$4 law=$5 kdk=$6 mrg=$7 lf=$8 rd=$9 names=${10}
  local common="--corpus $corpus -t $CORES --n-seq $NSEQ --n-prefill 1024 --n-decode 256 --no-mmap --host-experts"
  local LA=$OUT/la_$tag.bin
  local base="slots=$C:policy=dfa:$MB:fetch=$law:paced=0:overlap=1:oracle_w=0"
  local orc="slots=$C:policy=oracle:oracle=$LA:$MB:kappa=1:fetch=$law:overlap=1:oracle_w=0"
  local st="$OUT/st_${tag}_C${C}_r${rd}"
  local cfg="" n c
  for n in $names; do
    case $n in
      base)      c="$base:kappa=1" ;;
      dk)        c="$base:kappa=$kdk" ;;
      lrn)       lrnok $lf || { echo "lrn skipped at $tag C$C: model file hash mismatch" | tee -a $OUT/skipped.txt; continue; }
                 c="$base:kappa=1:learned=$LM/learned_$lf.bin:learned_margin=$mrg" ;;
      fetch)     c="$orc:oracle_bypass=1:oracle_fetch=1:paced=0" ;;
      fetchplan) c="$orc:oracle_bypass=1:oracle_fetch=1:paced=0:oracle_plan=$OUT/plan_${tag}${C}.bin" ;;
    esac
    cfg="${cfg:+$cfg;}$c:stats=${st}_$n.json"
  done
  local seed=$(( ORDER_SEED + 100 * rd + C ))
  echo "cell $tag C$C round $rd order-seed $seed configs: $cfg" >> $OUT/configs.txt
  local t0=$(date +%s)
  R 100m $EC_BIN -m $g $common --order-seed $seed --ec "$cfg" > $OUT/ec_${tag}_C${C}_r${rd}.jsonl 2> $OUT/ec_${tag}_C${C}_r${rd}.err
  local rc=$?
  echo "cell $tag C$C round $rd rc=$rc in $(( $(date +%s) - t0 )) s"; echo "ec_${tag}_C${C}_r${rd} $rc $(( $(date +%s) - t0 ))" >> $OUT/runs.txt
  grep "\[ec-bench\] config [0-9]*/" $OUT/ec_${tag}_C${C}_r${rd}.err | sed 's/:stats=.*st_/ -> st_/' > $OUT/ec_${tag}_C${C}_r${rd}.order
  grep -i "oracle\|fetch table\|learned\|shuffled\|plan" $OUT/ec_${tag}_C${C}_r${rd}.err | head -40 > $OUT/ec_${tag}_C${C}_r${rd}.err.oracle
  tail -n 30 $OUT/ec_${tag}_C${C}_r${rd}.err > $OUT/ec_${tag}_C${C}_r${rd}.err.tail; rm -f $OUT/ec_${tag}_C${C}_r${rd}.err
}
gprof() {  # the GPU's own compute per token from the named profiles, before any timed run of that model
  python3 - $OUT "$@" <<'PY'
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
}
# ---- gpt-oss-120b
CG=$J/aime25_owntext_gptoss_084.jsonl; wc -l $CG
lookahead g $F $CG 24
[ -s $OUT/la_g.bin ] || { echo "no gpt-oss lookahead file: oracle runs impossible"; exit 3; }
for C in 14 32; do
  prof_run G$C $EC_BIN -m $F --corpus $CG -t $CORES --n-prefill 1024 --n-decode 24 --no-mmap --host-experts --seqs 1,0 \
    --ec "slots=$C:policy=dfa:$MB:kappa=1:fetch=$LAWG:paced=0:overlap=1:oracle_w=0:seqs=1/0"
done
gprof G14 G32 > $OUT/g_prof.json
echo "G_prof (gpt-oss) written $(date -u +%FT%TZ), before any timed run:"; cat $OUT/g_prof.json
t0=$(date +%s)
( cd $J && R 20m $PV/bin/python -I minadm_plan.py $OUT/la_g.bin 14 $OUT/plan_g14.bin $CORES ) > $OUT/plan_g14.txt 2>&1
echo "plan C14 rc=$? in $(( $(date +%s) - t0 )) s"; cat $OUT/plan_g14.txt
GN14="base dk lrn fetch"; [ -s $OUT/plan_g14.bin ] && GN14="$GN14 fetchplan"
for rd in 1 2 3; do run_round g $F $CG 14 $LAWG 3 0.4 g14 $rd "$GN14"; done
for rd in 1 2; do run_round g $F $CG 32 $LAWG 2 0.4 g32 $rd "base dk lrn"; done
rm -f $F $OUT/la_g.bin
# ---- Qwen3-30B-A3B BF16 (converted here after the gpt-oss runs, so that nothing else runs during a timed step)
wait $DLQ $VEP; cat $OUT/dl_qwen3.txt
GQ=$M/Qwen3-30B-A3B-BF16.gguf
( t0=$(date +%s); export PYTHONPATH=$WORK/lc-ec/gguf-py; R 30m $VE/bin/python $WORK/lc-ec/convert_hf_to_gguf.py "$HFQ" --outtype bf16 --outfile $GQ
  echo "convert rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $GQ ) > $OUT/convert_qwen3.txt 2>&1
tail -2 $OUT/convert_qwen3.txt; rm -rf "$HFQ"
if [ -s "$GQ" ]; then
  CQ=$J/aime25_owntext_qwen3_084b.jsonl
  for C in 16 32; do
    prof_run Q$C $EC_BIN -m $GQ --corpus $CQ -t $CORES --n-prefill 1024 --n-decode 24 --no-mmap --host-experts --seqs 1,0 \
      --ec "slots=$C:policy=dfa:$MB:kappa=1:fetch=$LAWQ:paced=0:overlap=1:oracle_w=0:seqs=1/0"
  done
  gprof Q16 Q32 > $OUT/q_prof.json
  echo "G_prof (Qwen3) written $(date -u +%FT%TZ), before any Qwen3 timed run:"; cat $OUT/q_prof.json
  run_round q $GQ $CQ 16 $LAWQ 3 0.4 q16 1 "base dk lrn"
  run_round q $GQ $CQ 32 $LAWQ 2 0.3 q32 1 "base dk lrn"
  rm -f $GQ
else
  echo "Qwen3 GGUF missing: Qwen3 part skipped" | tee -a $OUT/skipped.txt
fi
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
        print(f"{lab:26s} {1000/ms:7.2f} tok/s {ms:6.2f} ms n={len(v)} speed vs base {b/ms if b else float('nan'):.3f} "
              f"misses {s.get('misses', 0) / n:.2f} admits {s.get('admits', 0) / n:.2f} learned_us {s.get('learned_us_per_step', 0)}")
        if lab.endswith(("_r1_base", "_r1_dk", "_r1_lrn", "_r1_fetchplan")):
            line.append(f"{lab.replace('st_', '')} {1000/ms:.1f}")
open(f"{out}/oneline.txt", "w").write("106 " + " ".join(line) + "\n")
PY
ONE=$(cat $OUT/oneline.txt)
python3 $J/manifest.py "$OUT" 106_onlineadmit@vast "$T0" "$ONE" law_gptoss=$LAWG law_qwen3=$LAWQ device_read_gbs=$DR memory_clock_max_mhz=$MEMCLK
cat $OUT/skipped.txt 2>/dev/null
echo "SUMMARY: $ONE"
