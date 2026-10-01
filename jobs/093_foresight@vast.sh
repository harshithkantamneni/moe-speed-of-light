#!/bin/bash
# Job 093: what foresight is worth in the engine, measured. The paper's accounting names missing routing foresight the
# largest cost where the host binds; this job measures it instead of modelling it. The cache gets an oracle policy
# (LLAMA_EC_POLICY=oracle, patch llama.cpp-expert-cache-4da6337-oracle.patch): the routing of the coming W decode steps is
# known (a lookahead file recorded on this machine over the same teacher-forced text, ec-bench --lookahead, record g =
# decode step g), and after every step the policy admits the experts the next W steps of the sequence will route to,
# soonest first, evicting the resident whose next use is furthest (Belady within the window; W = 0 is the rest of the
# sequence). The copies ride the normal admission path; an expert still in flight at its step runs on the CPU as any
# miss. Everything else is Table 1's engine (decayed frequency replaced by the oracle; mailbox helpers, GPU sampling
# not involved, the law's FETCH table computed on this machine from its own probe before any model run).
# Workload: the models' own greedy text on the 30 AIME-25 prompts (the routing-trace corpora of jobs 084/084b:
# ec2/aime25_owntext_gptoss_084.jsonl, ec2/aime25_owntext_qwen3_084b.jsonl), teacher-forced by llama-ec-bench, 256 decode
# steps per sequence after the prompt, timed one token at a time: the same text the speed limit's optimum is computed
# on, so the oracle's reads can be compared with the optimum's. Target: an RTX 5090 host with at least 120 GB of RAM.
# Gates: the device-read gate of job 084b (bw.cu >= 1,500 GB/s, exit 5); the card's clocks are recorded, not gated.
# Runs, per cell (gpt-oss-120b C 14 / 32, Qwen3-30B-A3B C 16 / 32: the four host-bound cells of the accounting; then
# gpt-oss C 51 and Qwen3 C 56, the GPU-bound ones, with three configurations each), one ec-bench process per cell,
# the model loaded once, every configuration a fresh context; labels in the config string:
#   base      Table 1's engine: policy=dfa, kappa=1, the law's FETCH table, copies published two steps later (paced=0)
#   paced     the same with paced copies (paced=1, 16 MB pieces, at most 64 queued): the oracle's copy path, as a control
#   oracle_wN the oracle at W = 2, 4, 16, 64 and 0 (the rest of the sequence), paced copies as above
#   oracle_np the oracle at W = 0 with the unpaced copy path (published two steps later)
#   noovl     base with LLAMA_EC_OVERLAP=0: a layer's CPU misses no longer run concurrently with its GPU hits
#   allcpu    base with no FETCH table: every miss runs on the CPU
# The GPU-bound cells run base, oracle_w0 and noovl only. Before the timed runs of each model, one untimed pass with
# --lookahead records the routing (ec-bench with --ncmoe 24 / 36, as jobs 084c / 084b; the recording syncs per tensor,
# so its timing is discarded). Ratios are of mean speeds over the 30 sequences, paired by sequence, 95% bootstrap
# interval (10,000 resamples), computed offline (scripts/foresight_stats.py in the main repo).
# Predictions, committed before launch (the law on this machine's probe gives the point values; the bands allow for the
# copies being bound by the link rather than the combined rate, and for the oracle's reads exceeding the optimum's):
#   1. at every host-bound cell the oracle at W = 0 is faster than base (paired interval above 1);
#   2. the W = 0 gain over base is 20-80% at every host-bound cell (the law with the optimum's reads at the combined
#      rate: about +38% at gpt-oss 11%, +34% at 25%, +65% at Qwen3 12.5%, +76% at 25%; the link alone bounds the
#      Qwen3 cells near +50%);
#   3. W = 16 captures at least half of the W = 0 gain at every host-bound cell, and W = 4 does at gpt-oss 11% and
#      Qwen3 12.5% (the trace study's W50: 3.1, 9.4, 1.5 and 3.7 tokens at these C/k);
#   4. with the oracle at W = 0 ours reaches 45-65% of this machine's speed limit at the host-bound cells (exact
#      optimum's reads, this host's highest probed rate, the datasheet 1,792 GB/s; scored offline), from 26-46% without;
#   5. the oracle's hit rate at W = 0 is within 3 points of the trace optimum's (gpt-oss 73.4% at 11%, 89.3% at 25%;
#      Qwen3 74.9% at 12.5%, 88.7% at 25%), and its admissions per token are within 30% of the optimum's reads
#      (38.3 / 15.3 / 96.5 / 43.4);
#   6. noovl is 5-25% slower than base at the host-bound cells; allcpu is 0-10% slower than base at every cell where
#      the law's table copies anything;
#   7. at the GPU-bound cells the oracle's gain is below 15%.
# Budget: the job stops starting steps 4.5 h after launch (every step bounded by its own timeout and by that deadline;
# skipped steps in skipped.txt); disk about 200 GB (gpt-oss 61 GB GGUF removed before Qwen3's 57 GB checkpoint is
# converted to a 61 GB GGUF).
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
export BASE; export -f build_tree platform getmodel
T0=$(date +%s); BUDGET=$(( 9 * 1800 )); RESERVE=600
secs() { case $1 in *h) echo $(( ${1%h} * 3600 ));; *m) echo $(( ${1%m} * 60 ));; *s) echo ${1%s};; *) echo $1;; esac; }
left() { echo $(( T0 + BUDGET - RESERVE - $(date +%s) )); }
R() {  # timeout cmd...: bounded by the step's timeout and by the job's deadline; past the deadline the step is skipped
  local t=$(secs $1); shift; local l=$(left)
  if [ "$l" -lt 120 ]; then echo "SKIPPED (deadline): $*" | tee -a $OUT/skipped.txt; return 124; fi
  [ "$t" -gt "$l" ] && t=$l
  timeout -k 30 "$t" "$@"
}
# --- gates: the card recorded, the device read required
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current,driver_version,power.limit --format=csv > $OUT/gpu.csv 2>&1
cat $OUT/gpu.csv
MEMCLK=$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "memory clock max ${MEMCLK} MHz (listing B 17001, the common clock 14001)" | tee $OUT/gate.txt
nvcc -O3 -arch=sm_$SM "$J/bw.cu" -o $WORK/bw0 && $WORK/bw0 > $OUT/bw_gate.txt 2>&1
nvidia-smi -q > $OUT/nvidia-smi-q-start.txt 2>&1
DR=$(sed -n 's/device read 1 GiB: *\([0-9]*\).*/\1/p' $OUT/bw_gate.txt)
echo "device read ${DR} GB/s" | tee -a $OUT/gate.txt
[ "${DR:-0}" -ge 1500 ] || { echo "GPU reads below 1500 GB/s: throttled card, stopping"; exit 5; }
free -g > $OUT/free.txt; RAMGB=$(awk '/Mem:/{print $2}' $OUT/free.txt); echo "host RAM ${RAMGB} GB" | tee -a $OUT/gate.txt
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip >> $OUT/apt.txt 2>&1
python3 -m pip install -q --break-system-packages uv huggingface_hub hf_xet numpy > $OUT/pip_uv.txt 2>&1
CORES=$(usable_cores); echo "usable physical cores $CORES (nproc $(nproc))" | tee $OUT/cores.txt
# --- downloads first; the build and the conversion venv run meanwhile
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
VE=$WORK/venv   # CPU-torch environment for the Qwen3 conversion (as jobs 085b/087/089)
( uv venv -q $VE && . $VE/bin/activate && uv pip install -q torch --index-url https://download.pytorch.org/whl/cpu &&
  uv pip install -q numpy transformers sentencepiece safetensors protobuf ) > $OUT/venv_install.txt 2>&1 &
VEP=$!
build_tree $WORK/lc-ec "$J/llama.cpp-expert-cache-4da6337-oracle.patch" llama-ec-bench
EC_BIN=$WORK/lc-ec/build/bin/llama-ec-bench; ls -la $EC_BIN || { echo "no ec-bench binary"; exit 3; }
# the oracle patch must be the committed one: its hash and the base commit, for the record
sha256sum $J/llama.cpp-expert-cache-4da6337-oracle.patch | tee $OUT/patch_sha.txt; echo "base $BASE" >> $OUT/patch_sha.txt
# --- the host (idle): platform, our probe, the law's tables
platform
H=$(( CORES > 4 ? CORES - 2 : 2 ))
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
grep -q "concurrent t=" $OUT/concur.txt || { echo "concur incomplete, retrying"; R 15m $WORK/concur > $OUT/concur.txt 2>&1; }
LAWG=$(python3 $J/fetch_table.py $OUT/concur.txt $H 13253760 4 $OUT/fetch_table_law_gptoss.json 37)
LAWQ=$(python3 $J/fetch_table.py $OUT/concur.txt $H 9437184 8 $OUT/fetch_table_law_qwen3.json 48)
echo "law tables ($(date -u +%FT%TZ)): gpt-oss $LAWG (listing B 0,0,1,1,2; job 089 0,0,1,2,3); qwen3 $LAWQ" | tee $OUT/tables.txt
[ -n "$LAWG" ] && [ -n "$LAWQ" ] || { echo "no law table"; exit 3; }
# the law's point predictions for the oracle at W = 0 (prediction 2), written from this probe before any model run:
# T = G + max(X_c / B_c, X_p / B_p, (X_c + X_p) / B_cp) with the optimum's reads split evenly between the paths
python3 - $OUT/fetch_table_law_gptoss.json $OUT/fetch_table_law_qwen3.json > $OUT/law_oracle.json <<'PY'
import json, sys, time
g = json.load(open(sys.argv[1])); q = json.load(open(sys.argv[2]))
G = {"g": 4.819346e-3, "q": 4.3e-3}
S = {"g": 13253760, "q": 9437184}
Rstar = {("g", 14): 38.3, ("g", 32): 15.3, ("g", 51): 6.8, ("q", 16): 96.5, ("q", 32): 43.4, ("q", 56): 13.7}   # prereg/speed_limit_v2.json, exact
out = {}
for (m, C), R in Rstar.items():
    t = g if m == "g" else q
    bc, bp, bb = t["B_c"] * 1e9, t["B_p"] * 1e9, t["B_both"] * 1e9
    xc, xp = 0.5 * R * S[m], 0.5 * R * S[m]
    T = G[m] + max(xc / bc, xp / bp, (xc + xp) / bb)
    out[f"{m}_C{C}"] = dict(reads=R, t_ms=round(1e3 * T, 2), tok_s=round(1 / T, 1), link_bound_tok_s=round(1 / (G[m] + R * S[m] / bp), 1))
print(json.dumps(dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), B=dict(gpt=g, qwen=q), predicted=out), indent=1))
PY
cat $OUT/law_oracle.json
echo "builds and probe done, waiting for the downloads ($(( $(date +%s) - T0 )) s since launch)"
wait $DLG $VEP
cat $OUT/dl_gguf.txt; echo "venv install rc: $(tail -1 $OUT/venv_install.txt)"
F=$M/gpt-oss-120b-MXFP4.gguf
[ -s "$F" ] || { echo "GGUF MISSING"; exit 3; }
MB="mailbox=1:tdec=1:helpers=$H:kappa=1"
PACED="paced=1:chunk=16777216:max_pending=64"
run_cell() {  # tag gguf corpus C law full(1|0) ncmoe-for-lookahead
  local tag=$1 g=$2 corpus=$3 C=$4 law=$5 full=$6
  local common="--corpus $corpus -t $CORES --n-prefill 1024 --n-decode 256 --no-mmap --host-experts"
  local LA=$OUT/la_$tag.bin
  local base="slots=$C:policy=dfa:$MB:fetch=$law:paced=0:overlap=1:oracle_w=0"
  local orc="slots=$C:policy=oracle:oracle=$LA:$MB:fetch=$law:overlap=1"
  local cfg="$base:stats=$OUT/st_${tag}_C${C}_base.json"
  if [ "$full" = 1 ]; then
    cfg="$cfg;$base:$PACED:stats=$OUT/st_${tag}_C${C}_paced.json"
    for W in 2 4 16 64 0; do cfg="$cfg;$orc:oracle_w=$W:$PACED:stats=$OUT/st_${tag}_C${C}_oracle_w$W.json"; done
    cfg="$cfg;$orc:oracle_w=0:paced=0:stats=$OUT/st_${tag}_C${C}_oracle_np.json"
    cfg="$cfg;$base:overlap=0:stats=$OUT/st_${tag}_C${C}_noovl.json"
    cfg="$cfg;$base:fetch=0:stats=$OUT/st_${tag}_C${C}_allcpu.json"
  else
    cfg="$cfg;$orc:oracle_w=0:$PACED:stats=$OUT/st_${tag}_C${C}_oracle_w0.json"
    cfg="$cfg;$base:overlap=0:stats=$OUT/st_${tag}_C${C}_noovl.json"
  fi
  echo "cell $tag C$C configs: $cfg" >> $OUT/configs.txt
  local t0=$(date +%s)
  R 70m $EC_BIN -m $g $common --ec "$cfg" > $OUT/ec_${tag}_C${C}.jsonl 2> $OUT/ec_${tag}_C${C}.err
  local rc=$?
  echo "cell $tag C$C rc=$rc in $(( $(date +%s) - t0 )) s"; echo "ec_${tag}_C${C} $rc $(( $(date +%s) - t0 ))" >> $OUT/runs.txt
  tail -n 30 $OUT/ec_${tag}_C${C}.err > $OUT/ec_${tag}_C${C}.err.tail; rm -f $OUT/ec_${tag}_C${C}.err
}
lookahead() {  # tag gguf corpus ncmoe: record the routing of the teacher-forced text on this machine (untimed)
  local tag=$1 g=$2 corpus=$3 n=$4
  local t0=$(date +%s)
  R 30m $EC_BIN -m $g --corpus $corpus -t $CORES --n-prefill 1024 --n-decode 256 --ncmoe $n --lookahead $OUT/la_$tag.bin > $OUT/la_$tag.jsonl 2> $OUT/la_$tag.err
  echo "lookahead $tag rc=$? in $(( $(date +%s) - t0 )) s: $(ls -la $OUT/la_$tag.bin | awk '{print $5}') bytes"; cat $OUT/la_$tag.bin.json
  tail -n 5 $OUT/la_$tag.err > $OUT/la_$tag.err.tail; rm -f $OUT/la_$tag.err
}
# ---- gpt-oss-120b
CG=$J/aime25_owntext_gptoss_084.jsonl; wc -l $CG
lookahead g $F $CG 24
[ -s $OUT/la_g.bin ] || { echo "no gpt-oss lookahead file: oracle runs impossible"; exit 3; }
run_cell g $F $CG 14 $LAWG 1
run_cell g $F $CG 32 $LAWG 1
run_cell g $F $CG 51 $LAWG 0
# ---- Qwen3-30B-A3B BF16 (gpt-oss removed first; conversion in the CPU-torch venv)
rm -f $F $OUT/la_g.bin; df -h $WORK | tail -1 > $OUT/df.txt
wait $DLQ; cat $OUT/dl_qwen3.txt
GQ=$M/Qwen3-30B-A3B-BF16.gguf
( t0=$(date +%s); PYTHONPATH=$WORK/lc-ec/gguf-py timeout 30m $VE/bin/python $WORK/lc-ec/convert_hf_to_gguf.py "$HFQ" --outtype bf16 --outfile $GQ
  echo "convert rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $GQ ) > $OUT/convert_qwen3.txt 2>&1
tail -2 $OUT/convert_qwen3.txt
if [ -s "$GQ" ]; then
  CQ=$J/aime25_owntext_qwen3_084b.jsonl; wc -l $CQ
  lookahead q $GQ $CQ 36
  if [ -s $OUT/la_q.bin ]; then
    run_cell q $GQ $CQ 16 $LAWQ 1
    run_cell q $GQ $CQ 32 $LAWQ 1
    run_cell q $GQ $CQ 56 $LAWQ 0
  else
    echo "no Qwen3 lookahead file: Qwen3 oracle runs skipped" | tee -a $OUT/skipped.txt
  fi
  rm -f $OUT/la_q.bin
else
  echo "Qwen3 GGUF missing: Qwen3 part skipped" | tee -a $OUT/skipped.txt
fi
rm -f $OUT/la_g.bin $OUT/la_q.bin   # 18 / 37 MB: beyond the result channel; the stats files carry what the paper needs
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
        by.setdefault(r["config"], {})[r["seq"]] = r["tok_s"]
    base = None
    for cfg, v in by.items():
        lab = [p for p in cfg.split(":") if p.startswith("stats=")]
        lab = os.path.basename(lab[0][6:]).replace(".json", "") if lab else cfg[:40]
        m = st.mean(v.values())
        if lab.endswith("_base"):
            base = v
        ratio = st.mean([v[s] / base[s] for s in v if base and s in base]) if base else None
        stf = f"{out}/{lab}.json"
        hit = adm = None
        if os.path.exists(stf):
            s = json.load(open(stf)); n = max(1, s.get("steps", 1)); hit = s.get("hit_rate"); adm = s.get("admits", 0) / n
        print(f"{lab:28s} {m:7.2f} tok/s n={len(v)}  vs base {ratio if ratio is None else round(ratio, 3)}  hit {hit if hit is None else round(hit, 3)}  admits/token {adm if adm is None else round(adm, 1)}")
        if "oracle_w0" in lab or lab.endswith("_base"):
            line.append(f"{lab.replace('st_', '')} {m:.1f}")
open(f"{out}/oneline.txt", "w").write("093 " + " ".join(line) + f"; device read {open(f'{out}/gate.txt').read().strip().splitlines()[-2].split()[-2] if os.path.exists(f'{out}/gate.txt') else '?'} GB/s\n")
PY
ONE=$(cat $OUT/oneline.txt)
python3 $J/manifest.py "$OUT" 093_foresight@vast "$T0" "$ONE" law_gptoss=$LAWG law_qwen3=$LAWQ device_read_gbs=$DR memory_clock_max_mhz=$MEMCLK
cat $OUT/skipped.txt 2>/dev/null
echo "SUMMARY: $ONE"
