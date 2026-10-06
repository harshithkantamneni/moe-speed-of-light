#!/bin/bash
# Job 097: a realisable admission order in the engine. Job 096 measured foresight spent in ten ways; this job measures
# what can be had without it: the online policy with its admissions read once (foa, as job 096), and the same with the
# admission and victim order taken from a learned reuse predictor (learned): a logistic model of "requested within the
# next C/k steps" on features the engine has when it plans a step (decayed request counts at five half-lives, steps
# since the last request, requested-now, same-layer and 8-bit cross-layer transition scores from the previous step's
# experts, training popularity), fitted on each model's mixed-domain trace (results/035, 036; no AIME text) and
# replayed offline on the AIME routing (prereg/learned_offline.json on main). The weights are in
# jobs/ec2/learned/learned_weights.json; the model files are recomputed here by jobs/ec2/learned_export.py and must
# hash to the replay's files. On a CPU build the engine's reads and fetches equal a Python replay of the same policy on
# the engine's own routing exactly (three variants). Five configurations per cell, each in a fresh context (cold cache),
# in a seeded shuffled order, at the six cells of Table 1:
#   base     the online policy (decayed frequency, kappa 1, the law's FETCH table; admissions read twice)
#   foa      the same, each admission fetched into its slot in the step that misses it (one read)
#   learned  foa with the learned order (LLAMA_EC_LEARNED)
#   fetch    MIN with bypass, admissions fetched in the step (oracle; one read)
#   both3p   fetch + the paced single-read prefetch with a three-step lead (oracle; the best state of jobs 095/096)
# Same protocol as job 096 (teacher-forced llama-ec-bench over the trace corpora, 30 x 256 steps after a 1,024-token
# prefill, a lookahead pass per model on this machine, the law's table from this machine's probe, one process per cell).
# Gate (prereg/learned_gate_097.md on main, written before the replay and before job 096's results): all three conditions hold. (1) Replayed offline across corpora, the
# learned order with single-read admission closes 47-57% of the deployed policy's read gap to MIN at the four host-bound
# cells (>= 30%). (2) Beyond single-read admission it closes 17-30% of the remaining gap (>= 15% at three of four cells).
# (3) In job 096, foa / base had a lower 95% bound >= 0.98 at all four host-bound cells on O4 (1.020, 1.030, 1.015, 1.016)
# and at three of four on O5 (0.998, 0.974, 1.006, 0.999).
# Predictions, committed before launch (per host unless stated; host-bound = gpt-oss 11/25%, Qwen3 12.5/25%;
# replay = prereg/learned_offline.json, reads per token on the AIME routing: single-read decayed frequency 59.42 / 28.28 /
# 14.30 and 166.25 / 84.22 / 28.84, learned 55.97 / 26.12 / 12.57 and 146.28 / 74.84 / 26.26, i.e. 5.8 / 7.6 / 12.1 and
# 12.0 / 11.1 / 9.0% fewer)
#   1. the six model files written here hash to the replay's (learned_export.txt: six MATCH);
#   2. learned reads fewer experts per token than foa at every cell, and its reduction is within 5 points of the
#      replay's at every cell;
#   3. learned runs 1.01-1.18x foa at the four host-bound cells, with the paired interval above 1.00;
#   4. at the GPU-bound cells (gpt-oss 40%, Qwen3 43.75%) learned / foa >= 0.99;
#   5. learned runs faster than base at the four host-bound cells and recovers 10-45% of the fetch oracle's gain there,
#      (learned/base - 1) / (fetch/base - 1);
#   6. the learned order's host time is at most 0.3 ms per token on gpt-oss and 0.6 ms on Qwen3 (stats learned_us_per_step);
#   7. on the same machine as job 096's O4 (Vast offer 52267630), foa / base is within 0.02 and fetch / base within
#      0.05 of job 096's at every cell (096: foa 1.020 / 1.030 / 1.024 / 1.015 / 1.016 / 1.017; fetch 1.361 / 1.426 /
#      1.335 / 1.438 / 1.513 / 1.375), and both3p / base within 0.08 (1.495 / 1.809 / 1.609 / 1.479 / 1.804 / 1.638).
# Hosts: the O4 machine (RTX 5090 + Ryzen 9 9950X) and an RTX 5090 + Ryzen 9 9950X3D, one launch each.
# Budget: the job stops starting steps 4 h after launch; disk about 200 GB.
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
export BASE; export -f build_tree platform getmodel
T0=$(date +%s); BUDGET=$(( 4 * 3600 )); RESERVE=600
ORDER_SEED=$(( $(date +%s) % 100000 )); echo "order seed base $ORDER_SEED" | tee $OUT/order_seed.txt
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
# --- the learned order's model files, recomputed here from the committed traces and weights; their hashes must equal
# the replay's (learned_weights.json); the bench runs whatever is written, and the hashes go in the record
python3 -m pip install -q --break-system-packages numba > $OUT/pip_numba.txt 2>&1
LM=$WORK/learned; mkdir -p $LM
( for c in 14 32 51; do python3 $J/learned_export.py $J/learned/gpt-oss-120b_S.npz $J/learned/learned_weights.json gpt-oss-120b_C$c $LM/learned_gpt-oss-120b_C$c.bin; done
  for c in 16 32 56; do python3 $J/learned_export.py $J/learned/qwen3-30b-a3b_fp8_S.npz $J/learned/learned_weights.json qwen3-30b-a3b_C$c $LM/learned_qwen3-30b-a3b_C$c.bin; done
  python3 - $J/learned/learned_weights.json $LM <<'PY'
import hashlib, json, sys
w = json.load(open(sys.argv[1]))
for k, v in w.items():
    h = hashlib.sha256(open(f"{sys.argv[2]}/learned_{k}.bin", "rb").read()).hexdigest()
    print(k, "MATCH" if h == v["sha256"] else f"MISMATCH {h} != {v['sha256']}")
PY
) > $OUT/learned_export.txt 2>&1 &
LMP=$!
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
echo "law tables ($(date -u +%FT%TZ)): gpt-oss $LAWG (listing B 0,0,1,1,2; jobs 089/093/094 0,0,1,2,3); qwen3 $LAWQ" | tee $OUT/tables.txt
[ -n "$LAWG" ] && [ -n "$LAWQ" ] || { echo "no law table"; exit 3; }
# the law's point values for the optimum's reads on this host (the accounting's v(F) uses the optimum's own CPU/copy split;
# here an even split), written from this probe before any model run, for the record
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
wait $DLG $VEP $LMP
cat $OUT/learned_export.txt
cat $OUT/dl_gguf.txt; echo "venv install rc: $(tail -1 $OUT/venv_install.txt)"
F=$M/gpt-oss-120b-MXFP4.gguf
[ -s "$F" ] || { echo "GGUF MISSING"; exit 3; }
MB="mailbox=1:tdec=1:helpers=$H:kappa=1"
PACED="paced=1:chunk=16777216:max_pending=64"
run_cell() {  # tag gguf corpus C law: the five configurations at one cell, in a shuffled order
  local tag=$1 g=$2 corpus=$3 C=$4 law=$5
  local mk=$([ "$tag" = g ] && echo gpt-oss-120b || echo qwen3-30b-a3b)
  local common="--corpus $corpus -t $CORES --n-prefill 1024 --n-decode 256 --no-mmap --host-experts"
  local LA=$OUT/la_$tag.bin
  local base="slots=$C:policy=dfa:$MB:fetch=$law:paced=0:overlap=1:oracle_w=0"
  local orc="slots=$C:policy=oracle:oracle=$LA:$MB:fetch=$law:overlap=1:oracle_w=0"
  local st="$OUT/st_${tag}_C${C}"
  local cfg="$base:stats=${st}_base.json"
  cfg="$cfg;$base:fetch_on_admit=1:stats=${st}_foa.json"
  cfg="$cfg;$base:fetch_on_admit=1:learned=$LM/learned_${mk}_C${C}.bin:stats=${st}_learned.json"
  cfg="$cfg;$orc:oracle_bypass=1:oracle_fetch=1:paced=0:stats=${st}_fetch.json"
  cfg="$cfg;$orc:oracle_bypass=0:oracle_lead=3:oracle_fetch=1:$PACED:stats=${st}_both3p.json"
  local seed=$(( ORDER_SEED + C ))
  echo "cell $tag C$C order-seed $seed configs: $cfg" >> $OUT/configs.txt
  local t0=$(date +%s)
  R 90m $EC_BIN -m $g $common --order-seed $seed --ec "$cfg" > $OUT/ec_${tag}_C${C}.jsonl 2> $OUT/ec_${tag}_C${C}.err
  local rc=$?
  echo "cell $tag C$C rc=$rc in $(( $(date +%s) - t0 )) s"; echo "ec_${tag}_C${C} $rc $(( $(date +%s) - t0 ))" >> $OUT/runs.txt
  grep "\[ec-bench\] config [0-9]*/" $OUT/ec_${tag}_C${C}.err | sed 's/:stats=.*st_/ -> st_/' > $OUT/ec_${tag}_C${C}.order
  grep -i "oracle\|fetch table\|forced\|fetch-on-admit\|shuffled\|learned" $OUT/ec_${tag}_C${C}.err | head -30 > $OUT/ec_${tag}_C${C}.err.oracle
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
run_cell g $F $CG 14 $LAWG
run_cell g $F $CG 32 $LAWG
run_cell g $F $CG 51 $LAWG
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
    run_cell q $GQ $CQ 16 $LAWQ
    run_cell q $GQ $CQ 32 $LAWQ
    run_cell q $GQ $CQ 56 $LAWQ
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
            mis = s.get("misses", 0) / n; fe = s.get("fetches", 0) / n; forced = s.get("oracle_forced", 0) / n
        else:
            mis = fe = forced = None
        print(f"{lab:28s} {m:7.2f} tok/s n={len(v)}  vs base {ratio if ratio is None else round(ratio, 3)}  hit {hit if hit is None else round(hit, 3)}  misses/token {mis if mis is None else round(mis, 1)} fetches {fe if fe is None else round(fe, 1)} admits {adm if adm is None else round(adm, 1)} forced {forced if forced is None else round(forced, 1)}")
        if lab.endswith("_learned") or lab.endswith("_base") or lab.endswith("_foa") or lab.endswith("_fetch"):
            line.append(f"{lab.replace('st_', '')} {m:.1f}")
open(f"{out}/oneline.txt", "w").write("097 " + " ".join(line) + f"; device read {open(f'{out}/gate.txt').read().strip().splitlines()[-2].split()[-2] if os.path.exists(f'{out}/gate.txt') else '?'} GB/s\n")
PY
ONE=$(cat $OUT/oneline.txt)
python3 $J/manifest.py "$OUT" 097_learned@vast "$T0" "$ONE" law_gptoss=$LAWG law_qwen3=$LAWQ device_read_gbs=$DR memory_clock_max_mhz=$MEMCLK
cat $OUT/skipped.txt 2>/dev/null
echo "SUMMARY: $ONE"
