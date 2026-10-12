#!/bin/bash
# Job 117: shadow run-ahead, stage B1, on Qwen3-30B-A3B BF16 at C = 16 (12.5% of each layer's experts). Job 116 (stage
# A, offline) found that the model run ahead on its own resident experts forecasts its routing well enough to pass the
# registered bar: driving the window policy, the self forecast closes 0.47 of MIN's gain over admitting every miss at
# W = 4 (precision@k 0.62 over h = 1..4). Two questions remain before building the shadow into the engine (stage B2):
# what that forecast is worth in seconds on the engine's read path, and whether the GPU has the spare time to run the
# shadow during a step. This job answers both with the shadow's cost set to zero: the forecast is computed beforehand
# (Hugging Face, as job 116) and read from a file by the engine's oracle.
# Lag. Stage A scored the forecast made at the end of step g for the decisions of step g. A shadow running inside the
# engine cannot deliver that: the input of step g + 1 is known only when step g ends. A shadow that runs during step g
# (from the end of step g - 1) delivers its forecast one step late (lag 1): seen from step g, the forecast of steps
# g + 1 .. g + 3 (its fourth horizon falls outside the window). Lag 2 is what a shadow a full step behind delivers.
# Method:
#   - Forecasts: jobs/ec2/shadow_runahead.py (job 116's, unchanged) on the first 20 problems of job 084b's AIME-25 own
#     text (jobs/ec2/aime25_owntext_qwen3_084b.jsonl), C = 16, H = 4; jobs/ec2/forecast_file.py writes the self
#     forecast's first k = 8 experts per (step, horizon, layer) for the engine.
#   - Engine: patch oracle6 = oracle5 + LLAMA_EC_ORACLE_FORECAST (the window's future records read from the forecast)
#     and LLAMA_EC_ORACLE_FC_LAG (the forecast seen d steps late). On a CPU build of the tiny Qwen3-MoE (4 layers, 32
#     experts, top-4) the engine's reads equal vmap.replay / vmap.replay_fc on its own trace in 24 configurations (C =
#     4, 8, 12; exact window, random-error window, no window, forecasts at lags 0, 1, 2), and a forecast equal to the
#     trace gives the exact window's counters.
#   - On the machine: Qwen/Qwen3-30B-A3B converted to a BF16 GGUF (as job 099 with QWEN=1); this machine's routing trace
#     (ec-bench --lookahead, every weight on the GPU); the probe and the law's FETCH table (as job 115).
#   - Timed: C = 16, 20 problems x 256 teacher-forced steps after the prompt, every expert in pinned host memory, two
#     rounds (one ec-bench process each, configurations in fresh contexts in a seeded shuffled order):
#       base   the deployed cache: decayed frequency, kappa 1, the law's FETCH table (two reads per admission)
#       aa     every miss fetched in the step into the lowest-score slot (the window policy with no window, one read)
#       fetch  MIN with bypass, admissions fetched in the step (one read): the target
#       w4     aa + the exact routing of the next 4 steps (the forecast's ceiling)
#       f0     aa + the shadow's forecast of the next 4 steps, lag 0 (stage A's yardstick; not deliverable)
#       f1     the same, lag 1 (what a shadow running during the step delivers)
#       f2     the same, lag 2
#       r7     aa + a random-error window of 4 steps, precision = recall = 0.7 (the value map's reference)
#   - Room: T_full = ms per token of the model with every expert on the GPU and no cache (ec-bench, the 20 problems): one
#     shadow token executes as many experts from device memory, so T_full is about the cost of one shadow token. Nsight
#     Systems on problem 0 (256 steps; medians over the last 200 tokens, jobs/ec2/prof_summary.py) for f1 and base:
#     G_prof = the GPU's kernel time per token except the helper wait and the expert copies (as job 105); room = wall
#     time per token - G_prof, the time the GPU spends waiting, in which a shadow could run.
# Scoring (jobs/ec2/score117.py): speed vs base = mean over problems of base's ms per token over the configuration's,
# each problem averaged over the two rounds, 95% interval by bootstrap over problems; shares of aa -> fetch in time and
# in engine reads per token.
# Decision rule (registered here, before launch): GO for stage B2 (the shadow inside the engine) if
#   (a) f1's speed vs base is at least 1.08, AND
#   (b) the room per token in f1 is at least 4 x T_full (a 4-token shadow fits in the GPU's waiting time).
#   If (a) holds and (b) fails, B2 goes ahead only with a cheaper shadow (fewer tokens per step); if (a) fails, NO-GO.
# Predictions:
#   P1. speed f0 > f1 > f2;
#   P2. speed vs base: f0 in [1.00, 1.15], f1 in [0.97, 1.10], f2 in [0.95, 1.06], w4 in [1.08, 1.35], r7 in [1.00, 1.15];
#   P3. time tracks reads: |time share - read share| <= 0.15 for w4, r7, f0, f1, f2;
#   P4. (b) fails: the room per token in f1 is below 4 x T_full;
#   P5. the decision is not GO (our prior for GO is about 0.15).
#   (Basis: on the two problems job 116 shipped, the self forecast closes 0.45, 0.35 and 0.23 of the aa -> MIN read gap
#   at lags 0, 1 and 2 at C = 16, the exact window 0.75; on job 099a's RTX 5090 host at Qwen3 C = 16, aa / base 0.93,
#   fetch / base 1.42, and w2's time share 0.41 against its read share 0.51.)
# Validity conditions (reported apart from the predictions; V0 and VR stop the host): V0 one GPU with at least 79 GB,
# device read at least 1500 GB/s, one NUMA node and at most 32 usable cores, at least 110 GiB of host memory (free -g)
# and 190 GB of disk; VR the probe's link/CPU ratio at least 0.5 (the fetch path loses on slower links, jobs 100a and 100f); V1
# the Hugging Face routing's top-8 overlaps this machine's trace by at least 0.90 (the forecast is aligned with the
# engine's steps; below 0.5 the host stops); V2 the greedy match at least 0.90; V3 every oracle configuration's engine
# reads per token within 3% of this machine's replay of it (jobs/ec2/replay117.py, before any timed run); V4 every
# configuration's teacher-forced loss within 1% of base's; V5 base's ms per token in the two rounds within 3%. If any of
# V1-V5 fails, the results are reported as exploratory.
# Host: the cheapest rentable on-demand offer of one whole GPU with at least 79 GB on a desktop CPU (Core, Core Ultra or
# Ryzen; not Threadripper, EPYC or Xeon), at least 120 GB of RAM (the listing's figure) and 190 GB of disk, at least 300 Mb/s down, CUDA at
# least 12.8, at most $1.60/h, from the listing committed with this header (results/117_offers_at_registration.txt).
# Machines rented before are allowed (one machine, not a panel). A machine that fails V0 or VR or cannot install or
# download is replaced once by the cheapest qualifying offer in a new listing committed before that launch.
# Cheapest qualifying at registration: 55163302 (RTX PRO 6000 WS, Core Ultra 7 265K, $0.91/h; job 116's machine). The account holds about $5.65.
# Budget: the job stops starting steps 150 minutes after launch.
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
JD="$(cd "$(dirname "$0")" && pwd)"
C=16; NSEQ=20; NDEC=256; BMIN=150; DRMIN=1500; VRMIN=0.5; RAMMIN=110; DISKMIN=190
gate_fail() {  # exit-code message: a failed gate stops the host
  local code=$1; shift
  echo "$*" | tee -a $OUT/v0.txt
  exit $code
}
nvidia-smi --query-gpu=name,memory.total,uuid,driver_version --format=csv | tee $OUT/gpu0.csv
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
[ "${MEM:-0}" -ge 79000 ] || gate_fail 7 "V0 FAIL: GPU memory ${MEM} MiB below 79000, stopping"
. "$JD/ec2/setup.sh"
. "$J/prof.sh"
export BASE; export -f build_tree platform getmodel
PATCH=$J/llama.cpp-expert-cache-4da6337-oracle6.patch
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
# ---- V0: the card, the CPU class, memory and disk
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket|NUMA node\(s\)" | tee $OUT/cpu.txt
NUMA=$(lscpu | sed -n 's/^NUMA node(s): *\([0-9]*\).*/\1/p'); CORES=$(usable_cores)
echo "usable physical cores $CORES (nproc $(nproc)); NUMA nodes ${NUMA:-?}" | tee $OUT/cores.txt
[ "${NUMA:-9}" -eq 1 ] && [ "${CORES:-99}" -le 32 ] || gate_fail 9 "V0 FAIL: ${NUMA} NUMA nodes, ${CORES} usable cores: not desktop-class, stopping"
nvcc -O3 -arch=sm_$SM "$J/bw.cu" -o $WORK/bw0 && $WORK/bw0 > $OUT/bw_gate.txt 2>&1
DR=$(sed -n 's/device read 1 GiB: *\([0-9]*\).*/\1/p' $OUT/bw_gate.txt)
MEMCLK=$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "device read ${DR} GB/s; memory clock max ${MEMCLK} MHz" | tee $OUT/gate.txt
[ "${DR:-0}" -ge $DRMIN ] || gate_fail 5 "V0 FAIL: GPU reads below $DRMIN GB/s: throttled card, stopping"
free -g > $OUT/free.txt; RAMGB=$(awk '/Mem:/{print $2}' $OUT/free.txt)
DISKGB=$(df -BG --output=avail $WORK | tail -1 | tr -dc 0-9)
echo "host RAM ${RAMGB} GB; disk free ${DISKGB} GB" | tee -a $OUT/gate.txt
[ "${RAMGB:-0}" -ge $RAMMIN ] || gate_fail 6 "V0 FAIL: host RAM ${RAMGB} GB below $RAMMIN, stopping"
[ "${DISKGB:-0}" -ge $DISKMIN ] || gate_fail 6 "V0 FAIL: disk ${DISKGB} GB below $DISKMIN, stopping"
echo "V0 checked" | tee -a $OUT/v0.txt
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip python3-venv >> $OUT/apt.txt 2>&1
python3 -m pip install -q --break-system-packages uv huggingface_hub hf_xet numpy numba scipy > $OUT/pip.txt 2>&1
for i in 1 2 3; do python3 -c "import numba" 2>/dev/null && break; python3 -m pip install -q --break-system-packages numba >> $OUT/pip.txt 2>&1; done
python3 -c "import numba; print('numba', numba.__version__)" | tee -a $OUT/pip.txt
# ---- the probe on the idle host, the law's table, VR
platform
H=$(( CORES > 4 ? CORES - 2 : 2 ))
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
grep -q "concurrent t=" $OUT/concur.txt || { echo "concur incomplete, retrying"; R 15m $WORK/concur > $OUT/concur.txt 2>&1; }
LAWQ=$(python3 $J/fetch_table.py $OUT/concur.txt $H 9437184 8 $OUT/fetch_table_law_qwen3.json 48)
echo "law table ($(date -u +%FT%TZ)): qwen3 $LAWQ" | tee $OUT/tables.txt
[ -n "$LAWQ" ] || { echo "no law table"; exit 3; }
RATIO=$(python3 -c "import json, sys; d = json.load(open(sys.argv[1])); print(f'{d[\"B_p\"] / d[\"B_c\"]:.4f}')" $OUT/fetch_table_law_qwen3.json)
echo "VR: B_p / B_c = ${RATIO:-?}" | tee -a $OUT/gate.txt
python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)" "${RATIO:-0}" $VRMIN || \
  gate_fail 10 "VR FAIL: link/CPU ratio ${RATIO} below $VRMIN, stopping"
echo "VR checked" | tee -a $OUT/v0.txt
# ---- downloads and environments in the background, the engine build in the foreground
HFQ=$M/Qwen3-30B-A3B; GQ=$M/Qwen3-30B-A3B-BF16.gguf; CQ=$J/aime25_owntext_qwen3_084b.jsonl
mkdir -p $M
( t0=$(date +%s); python3 -c "import sys; from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-30B-A3B', local_dir=sys.argv[1])" "$HFQ"
  echo "qwen3 hf download rc=$? in $(( $(date +%s) - t0 )) s"; du -sh "$HFQ" ) > $OUT/dl_qwen3.txt 2>&1 &
DLQ=$!
VE=$WORK/venv; VC=$WORK/venv_conv
( uv venv -q $VE && . $VE/bin/activate && uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 &&
  uv pip install -q "transformers==5.17.0" accelerate numpy numba safetensors huggingface_hub hf_xet ) > $OUT/venv.txt 2>&1 &
VEP=$!
( uv venv -q $VC && . $VC/bin/activate && uv pip install -q torch --index-url https://download.pytorch.org/whl/cpu &&
  uv pip install -q numpy transformers sentencepiece safetensors protobuf ) > $OUT/venv_conv.txt 2>&1 &
VCP=$!
build_tree $WORK/lc-ec "$PATCH" llama-ec-bench
EC_BIN=$WORK/lc-ec/build/bin/llama-ec-bench; ls -la $EC_BIN || { echo "no ec-bench binary"; exit 3; }
install_nsys
sha256sum $PATCH $J/shadow_runahead.py $J/shadow_score.py $J/vmap.py $J/forecast_file.py $J/replay117.py $J/score117.py \
  $J/prof_summary.py $CQ | tee $OUT/sha.txt; echo "base $BASE" >> $OUT/sha.txt
wait $VEP $VCP
. $VE/bin/activate
python -c "import torch, transformers, numba; print(torch.__version__, torch.version.cuda, transformers.__version__, numba.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))" | tee $OUT/venv_ok.txt
deactivate
[ -s $OUT/venv_ok.txt ] || { echo "install failed"; exit 3; }
echo "build done, waiting for the download ($(( $(date +%s) - T0 )) s since launch)"
wait $DLQ; cat $OUT/dl_qwen3.txt
[ -s "$HFQ/config.json" ] || { echo "download failed"; exit 3; }
# ---- the forecasts (GPU) while the GGUF is converted (CPU)
( t0=$(date +%s); PYTHONPATH=$WORK/lc-ec/gguf-py timeout 40m $VC/bin/python $WORK/lc-ec/convert_hf_to_gguf.py "$HFQ" --outtype bf16 --outfile $GQ
  echo "convert rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $GQ ) > $OUT/convert_qwen3.txt 2>&1 &
CVP=$!
t0=$(date +%s)
R 50m $VE/bin/python $J/shadow_runahead.py --model "$HFQ" --corpus $CQ --out $WORK/run --C $C --H 4 --nprob $NSEQ \
  --budget-min 40 2>&1 | grep -v -i "warn" | tee $OUT/shadow.txt
echo "shadow run in $(( $(date +%s) - t0 )) s" | tee -a $OUT/shadow.txt
cp $WORK/run/run.json $OUT/ 2>/dev/null
wait $CVP; tail -3 $OUT/convert_qwen3.txt
if [ ! -s "$GQ" ]; then   # once more with the forecasts' environment (transformers 5.17.0)
  ( . $VE/bin/activate && uv pip install -q sentencepiece protobuf
    t0=$(date +%s); PYTHONPATH=$WORK/lc-ec/gguf-py timeout 40m $VE/bin/python $WORK/lc-ec/convert_hf_to_gguf.py "$HFQ" --outtype bf16 --outfile $GQ
    echo "convert (second try) rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $GQ ) >> $OUT/convert_qwen3.txt 2>&1
  tail -3 $OUT/convert_qwen3.txt
fi
rm -rf "$HFQ"
[ -s "$WORK/run/shadow.npz" ] || { echo "no forecasts"; exit 3; }
[ -s "$GQ" ] || { echo "Qwen3 GGUF missing"; exit 3; }
FC=$WORK/fc_self.bin
python3 $J/forecast_file.py $WORK/run/shadow.npz $FC self | tee -a $OUT/shadow.txt
cp $FC.json $OUT/fc_self.bin.json
# ---- this machine's routing trace (every weight on the GPU), V1 / V2, the replay before any timed run
t0=$(date +%s)
R 30m $EC_BIN -m $GQ --corpus $CQ -t $CORES --n-seq $NSEQ --n-prefill 1024 --n-decode $NDEC --lookahead $OUT/la_q.bin > $OUT/la_q.jsonl 2> $OUT/la_q.err
echo "lookahead rc=$? in $(( $(date +%s) - t0 )) s: $(ls -la $OUT/la_q.bin | awk '{print $5}') bytes"; cat $OUT/la_q.bin.json
tail -n 5 $OUT/la_q.err > $OUT/la_q.err.tail; rm -f $OUT/la_q.err
[ -s $OUT/la_q.bin ] || { echo "no lookahead file: oracle runs impossible"; exit 3; }
python3 - $OUT/la_q.bin $WORK/la_q.npz $J <<'PY'
import sys, numpy as np
sys.path.insert(0, sys.argv[3])
import vmap as V
act, seq, meta = V.load_la(sys.argv[1])
np.savez(sys.argv[2], act=act, seq=seq, n_expert=int(meta["n_expert"]))
print("trace", act.shape, "problems", int(seq.max()) + 1)
PY
python3 $J/shadow_score.py $WORK/run $WORK/la_q.npz 2>&1 | tee $OUT/score.txt
cp $WORK/run/score.json $OUT/ 2>/dev/null
V1=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get('V1_trace_overlap') or 0)" $OUT/score.json 2>/dev/null)
echo "V1: trace overlap ${V1:-?}" | tee -a $OUT/validity.txt
python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= 0.5 else 1)" "${V1:-0}" || { echo "V1 below 0.5: forecast not aligned with the engine, stopping" | tee -a $OUT/validity.txt; exit 11; }
python3 $J/replay117.py $OUT/la_q.bin $FC $C $OUT/replay117.json $OUT/la_small.npz | tee $OUT/replay117.txt
echo "replay written $(date -u +%FT%TZ), before any timed run"
python3 - "$WORK/run/shadow.npz" "$OUT/shadow_small.npz" <<'PY'
import sys, numpy as np
z = np.load(sys.argv[1]); s = z["seq"]; n = int((s < 1).sum()); k = int(z["k"])
np.savez_compressed(sys.argv[2], R=z["R"][:n], seq=s[:n], match=z["match"][:n], tok_self=z["tok_self"][:n],
                    F_orac=z["F_orac"][:n, :, :, :k], E=z["E"], C=z["C"], H=z["H"], k=k, L=z["L"])
print("subset", n, "steps")
PY
# ---- T_full: every expert on the GPU, no cache
t0=$(date +%s)
R 20m $EC_BIN -m $GQ --corpus $CQ -t $CORES --n-seq $NSEQ --n-prefill 1024 --n-decode $NDEC --no-mmap > $OUT/tfull.jsonl 2> $OUT/tfull.err
echo "tfull rc=$? in $(( $(date +%s) - t0 )) s"; tail -n 5 $OUT/tfull.err > $OUT/tfull.err.tail; rm -f $OUT/tfull.err
# ---- configurations
MB="mailbox=1:tdec=1:helpers=$H"
AA="kappa=-1000000000"
cfg_of() {  # name -> the configuration string
  local base="slots=$C:policy=dfa:$MB:fetch=$LAWQ:paced=0:overlap=1:oracle_w=0:kappa=1"
  local orc="slots=$C:policy=oracle:oracle=$OUT/la_q.bin:$MB:kappa=1:fetch=$LAWQ:overlap=1:oracle_w=0"
  local hyb="$orc:oracle_bypass=1:oracle_fetch=1:paced=0:oracle_hybrid=1:$AA"
  case $1 in
    base)  echo "$base" ;;
    aa)    echo "$hyb:oracle_w=-1" ;;
    fetch) echo "$orc:oracle_bypass=1:oracle_fetch=1:paced=0" ;;
    w4)    echo "$hyb:oracle_w=4" ;;
    f0)    echo "$hyb:oracle_w=4:oracle_forecast=$FC" ;;
    f1)    echo "$hyb:oracle_w=4:oracle_forecast=$FC:oracle_fc_lag=1" ;;
    f2)    echo "$hyb:oracle_w=4:oracle_forecast=$FC:oracle_fc_lag=2" ;;
    r7)    echo "$hyb:oracle_w=4:oracle_recall=0.7:oracle_fill=1:oracle_seed=1" ;;
    *)     echo "unknown configuration $1" >&2; return 2 ;;
  esac
}
# ---- room: Nsight Systems on problem 0 (256 steps; the trace's and the forecast's first records are problem 0)
prof117() {  # label name
  local label=$1 name=$2
  [ -z "$NSYS" ] && { echo "prof $label: no nsys"; return 1; }
  R 15m $NSYS profile -o $WORK/prof_$label --force-overwrite true --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none \
    $EC_BIN -m $GQ --corpus $CQ -t $CORES --seqs 0 --n-prefill 1024 --n-decode $NDEC --no-mmap --host-experts \
    --ec "$(cfg_of $name):stats=$OUT/st_prof_$name.json" > $OUT/prof_$label.stdout 2> $OUT/prof_$label.err
  echo "prof $label rc=$?"
  $NSYS stats --force-export=true --report cuda_gpu_trace,cuda_api_trace --format csv --output $WORK/prof_$label \
    $WORK/prof_$label.nsys-rep > /dev/null 2>> $OUT/prof_$label.err
  python3 $J/prof_summary.py --work $WORK --label $label --out $OUT --tokens 200 --decode-steps $NDEC >> $OUT/prof_summary.txt 2>> $OUT/prof_$label.err
  rm -f $WORK/prof_${label}_*.csv $WORK/prof_${label}.sqlite $WORK/prof_$label.nsys-rep
  tail -n 20 $OUT/prof_$label.err > $OUT/prof_$label.err.tail; rm -f $OUT/prof_$label.err
}
prof117 F1 f1
prof117 BASE base
cat $OUT/prof_summary.txt
# ---- the timed rounds
NAMES="base aa fetch w4 f0 f1 f2 r7"
run_round() {  # round: one process, every configuration in a shuffled order
  local rd=$1 cfg="" n
  for n in $NAMES; do cfg="${cfg:+$cfg;}$(cfg_of $n):stats=$OUT/st_q_r${rd}_$n.json"; done
  local seed=$(( ORDER_SEED + 100 * rd + C ))
  echo "round $rd order-seed $seed configs: $cfg" >> $OUT/configs.txt
  local t0=$(date +%s)
  R 60m $EC_BIN -m $GQ --corpus $CQ -t $CORES --n-seq $NSEQ --n-prefill 1024 --n-decode $NDEC --no-mmap --host-experts \
    --order-seed $seed --ec "$cfg" > $OUT/ec_q_r$rd.jsonl 2> $OUT/ec_q_r$rd.err
  local rc=$?
  echo "round $rd rc=$rc in $(( $(date +%s) - t0 )) s"; echo "ec_q_r$rd $rc $(( $(date +%s) - t0 ))" >> $OUT/runs.txt
  grep "\[ec-bench\] config [0-9]*/" $OUT/ec_q_r$rd.err | sed 's/:stats=.*st_/ -> st_/' > $OUT/ec_q_r$rd.order
  grep -i "oracle\|forecast\|fetch table\|shuffled\|hybrid\|degraded" $OUT/ec_q_r$rd.err | sort | uniq -c | head -40 > $OUT/ec_q_r$rd.err.oracle
  tail -n 30 $OUT/ec_q_r$rd.err > $OUT/ec_q_r$rd.err.tail; rm -f $OUT/ec_q_r$rd.err
}
run_round 1
run_round 2
rm -f $GQ $OUT/la_q.bin
python3 $J/score117.py $OUT 2>&1 | tee $OUT/summary.txt
ONE=$(python3 -c "import json, sys; d = json.load(open(sys.argv[1])); s = d['speed']; print('117 ' + ' '.join(f'{k} {v[\"ratio\"]:.3f}' for k, v in s.items() if v) + ' decision ' + d['gate']['decision'])" $OUT/score117.json 2>/dev/null)
python3 $J/manifest.py "$OUT" "117_shadowb1@vast" "$T0" "${ONE:-117 incomplete}" law_qwen3=$LAWQ device_read_gbs=$DR memory_clock_max_mhz=$MEMCLK ratio=$RATIO
cat $OUT/skipped.txt 2>/dev/null
echo "SUMMARY: ${ONE:-incomplete}"
