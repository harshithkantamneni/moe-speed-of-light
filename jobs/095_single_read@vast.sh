#!/bin/bash
# Job 095: the single-read oracles. Jobs 093 and 094 showed that both oracles as built read an admitted expert twice
# (the CPU serves the miss, then the background copy reads it again), so their host bytes were 1.7-2.0x the optimum's
# R* and neither realised the foresight state the accounting prices; the hit-optimal oracle sat at the DRAM roof at
# 44% of the limit = R*/reads x utilisation. This job runs oracles that read an admitted expert ONCE, in the patch
# llama.cpp-expert-cache-4da6337-oracle.patch (tested on a CPU build against scripts/foresight_single_read_sim.py:
# hits and reads within 2% on a toy model):
#   fetch   LLAMA_EC_ORACLE_FETCH=1 + ORACLE_BYPASS=1: MIN with bypass decided before the step from the trace's record
#           of it; each admitted miss is copied by the GPU's FETCH op into the victim's slot and run from there this
#           step (one read, on the critical path); bypassed misses run on the CPU. Reads = misses, the optimum's up
#           to the within-sequence lookahead (simulated 40.0 / 17.6 / 9.3 and 101.3 / 47.3 / 17.7 per token against
#           R* 38.3 / 15.3 / 6.8 and 96.5 / 43.4 / 13.7). Serialised: the model's foresight-only state.
#   lead2   LLAMA_EC_ORACLE_LEAD=2 (unpaced copies, published two steps after issue): the scheduled single-read
#           prefetch: an expert whose first use is at least 2 steps away is copied ahead, if MIN with bypass would
#           admit it at that use (the victim is needed later than the expert's use after that one, or never); an
#           expert needed sooner misses on the CPU. Foresight with overlap, one read per admitted expert.
#   both2   fetch + lead2: the misses within the lead are fetched, the rest prefetched (simulated reads 48.9 / 23.1 /
#           12.3 and 117.9 / 64.7 / 24.2 per token; hit rates 86.5 / 98.6 / 99.5 and 84.1 / 96.7 / 99.5%).
#   both3p  the same on the paced copy path (16 MB pieces, <= 64 queued) with a 3-step lead.
#   hitopt  job 093's hit-optimal oracle, unpaced, W = all, for the same-host comparison (2x bytes).
# With base (the online policy, the law's FETCH table) these are the measured coalitions of a factorial
# {online, single-read oracle} x {serialised, prefetch-overlapped} at all six cells of Table 1. Same protocol as
# jobs 093/094 (teacher-forced llama-ec-bench over the trace corpora, 30 x 256 steps, a lookahead pass per model on
# this machine with --ncmoe 24 / 36, the law's table from this machine's probe). One process per cell.
# Target: an RTX 5090 next to a Ryzen 9 9950X3D (listing B's CPU, the accounting's host class) or 9950X, >= 120 GB RAM.
# Predictions, committed before launch:
#   1. the fetch oracle's host reads per token (misses: CPU-served or fetched, each read once; no admissions) are
#      within 10% of the simulation's 40.0 / 17.6 / 9.3 (gpt-oss 11 / 25 / 40%) and 101.3 / 47.3 / 17.7 (Qwen3
#      12.5 / 25 / 43.75%); its hit rate within 3 points of the simulation's 72.3 / 87.8 / 93.5 and 73.6 / 87.7 / 95.4%;
#   2. the fetch oracle is faster than base at every cell (paired interval above 1), and the law on its own counters
#      (misses on the CPU, fetches over the link, no admissions) predicts its time within 10% at every cell: it is
#      serialised execution, the law's regime;
#   3. both2's reads (CPU misses + fetches + admissions) are within 25% of the simulation's 48.9 / 23.1 / 12.3 and
#      117.9 / 64.7 / 24.2 per token, and its hit rate within 5 points of 86.5 / 98.6 / 99.5 and 84.1 / 96.7 / 99.5%;
#   4. both2 stands at 55-85% of this host's limit at the four host-bound cells (the hit-optimal oracle stood at
#      44-45% on two hosts at 2x the bytes), and is faster than hitopt at every host-bound cell;
#   5. the law over-predicts both2's time by 15-50% at the four host-bound cells (its copies overlap the step);
#   6. at gpt-oss 11% and Qwen3 12.5%, where no oracle so far gained more than 6%, both2 gains at least 25% over base;
#   7. both3p (paced) is within 10% of both2 at every cell.
# Budget: the job stops starting steps 4.5 h after launch; disk about 200 GB.
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
wait $DLG $VEP
cat $OUT/dl_gguf.txt; echo "venv install rc: $(tail -1 $OUT/venv_install.txt)"
F=$M/gpt-oss-120b-MXFP4.gguf
[ -s "$F" ] || { echo "GGUF MISSING"; exit 3; }
MB="mailbox=1:tdec=1:helpers=$H:kappa=1"
PACED="paced=1:chunk=16777216:max_pending=64"
run_cell() {  # tag gguf corpus C law full(1|0): the six configurations at every cell (full is ignored)
  local tag=$1 g=$2 corpus=$3 C=$4 law=$5 full=$6
  local common="--corpus $corpus -t $CORES --n-prefill 1024 --n-decode 256 --no-mmap --host-experts"
  local LA=$OUT/la_$tag.bin
  local base="slots=$C:policy=dfa:$MB:fetch=$law:paced=0:overlap=1:oracle_w=0"
  local orc="slots=$C:policy=oracle:oracle=$LA:$MB:fetch=$law:overlap=1:oracle_w=0"
  local cfg="$base:stats=$OUT/st_${tag}_C${C}_base.json"
  cfg="$cfg;$orc:oracle_bypass=1:oracle_fetch=1:paced=0:stats=$OUT/st_${tag}_C${C}_fetch.json"
  cfg="$cfg;$orc:oracle_bypass=0:oracle_lead=2:paced=0:stats=$OUT/st_${tag}_C${C}_lead2.json"
  cfg="$cfg;$orc:oracle_bypass=0:oracle_lead=2:oracle_fetch=1:paced=0:stats=$OUT/st_${tag}_C${C}_both2.json"
  cfg="$cfg;$orc:oracle_bypass=0:oracle_lead=3:oracle_fetch=1:$PACED:stats=$OUT/st_${tag}_C${C}_both3p.json"
  cfg="$cfg;$orc:oracle_bypass=0:paced=0:stats=$OUT/st_${tag}_C${C}_hitopt.json"
  echo "cell $tag C$C configs: $cfg" >> $OUT/configs.txt
  local t0=$(date +%s)
  R 80m $EC_BIN -m $g $common --ec "$cfg" > $OUT/ec_${tag}_C${C}.jsonl 2> $OUT/ec_${tag}_C${C}.err
  local rc=$?
  echo "cell $tag C$C rc=$rc in $(( $(date +%s) - t0 )) s"; echo "ec_${tag}_C${C} $rc $(( $(date +%s) - t0 ))" >> $OUT/runs.txt
  grep -i "oracle\|fetch table\|forced" $OUT/ec_${tag}_C${C}.err | head -20 > $OUT/ec_${tag}_C${C}.err.oracle
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
            mis = s.get("misses", 0) / n; fe = s.get("fetches", 0) / n; forced = s.get("oracle_forced", 0) / n
        else:
            mis = fe = forced = None
        print(f"{lab:28s} {m:7.2f} tok/s n={len(v)}  vs base {ratio if ratio is None else round(ratio, 3)}  hit {hit if hit is None else round(hit, 3)}  misses/token {mis if mis is None else round(mis, 1)} fetches {fe if fe is None else round(fe, 1)} admits {adm if adm is None else round(adm, 1)} forced {forced if forced is None else round(forced, 1)}")
        if "both2" in lab or lab.endswith("_base"):
            line.append(f"{lab.replace('st_', '')} {m:.1f}")
open(f"{out}/oneline.txt", "w").write("095 " + " ".join(line) + f"; device read {open(f'{out}/gate.txt').read().strip().splitlines()[-2].split()[-2] if os.path.exists(f'{out}/gate.txt') else '?'} GB/s\n")
PY
ONE=$(cat $OUT/oneline.txt)
python3 $J/manifest.py "$OUT" 095_single_read@vast "$T0" "$ONE" law_gptoss=$LAWG law_qwen3=$LAWQ device_read_gbs=$DR memory_clock_max_mhz=$MEMCLK
cat $OUT/skipped.txt 2>/dev/null
echo "SUMMARY: $ONE"
