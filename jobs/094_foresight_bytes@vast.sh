#!/bin/bash
# Job 094: foresight in bytes. Job 093 measured a hit-optimal oracle (Belady prefetch of the window's experts): it
# recovered 52-103% of the accounting's foresight term at 25% and above but lost 8% at the lowest budgets, where it
# admitted 35-63 experts per token, more host bytes than the policy it replaced, because Belady admits experts used once.
# The accounting's term is the optimum's READS, Belady's MIN with bypass (mosl.cachesim): a miss is served by the CPU and
# admitted only if its next use comes before the furthest next use among the residents. This job runs that oracle in the
# engine (LLAMA_EC_ORACLE_BYPASS=1, patch llama.cpp-expert-cache-4da6337-oracle.patch, tested on a CPU build against the
# simulator: within 6% of its misses and admissions) beside the hit-optimal one, on the same host, at all six cells of
# Table 1. Same protocol as job 093 otherwise (teacher-forced llama-ec-bench over the trace corpora, 30 x 256 steps, a
# lookahead pass per model recorded on this machine with --ncmoe 24 / 36, the law's FETCH table from this machine's
# probe). Configurations per cell, one process per cell:
#   base        Table 1's engine (policy=dfa, kappa=1, the law's FETCH table)
#   bypass_w0   the MIN-with-bypass oracle, window = the rest of the sequence, paced copies (16 MB pieces, <= 64 queued)
#   bypass_w16  the same with a 16-token window
#   prefetch_w0 job 093's hit-optimal oracle at W = 0, for the same-host comparison
# Target: an RTX 5090 host with at least 120 GB of RAM (the 093 host was a 9950X whose memory read 44 GB/s by the CPU
# and 46 over the link; this one is whatever the market offers; the gates of 093 apply).
# Predictions, committed before launch:
#   1. the bypass oracle's host reads per token (misses, each read once by the CPU or by FETCH) are within 15% of the
#      optimum's (38.3 / 15.3 / 6.8 for gpt-oss at 11 / 25 / 40%; 96.5 / 43.4 / 13.7 for Qwen3 at 12.5 / 25 / 43.75%);
#   2. the bypass oracle is faster than base at every cell (paired interval above 1), by 10-45% at the four host-bound
#      cells (the law with the optimum's reads, v(F), says +45-62% on the 093 host; the background admissions it does not
#      charge read the same DRAM);
#   3. its hit rate is within 4 points of the optimum's at every cell (73.4 / 89.3 / 95.3; 74.9 / 88.7 / 96.4);
#   4. the law on the bypass oracle's own counters (misses on the CPU, fetches + admissions over the link) over-predicts
#      its measured time by 10-40% at every host-bound cell: the admissions overlap the step, which the law charges;
#   5. at gpt-oss 11% and Qwen3 12.5% the bypass oracle beats the hit-optimal one by at least 15%; at 25% and above the
#      two are within 15% of each other;
#   6. W = 16 captures at least 70% of the W = 0 gain at every cell where that gain is positive.
# Budget: the job stops starting steps 4 h after launch (every step bounded by its own timeout and by that deadline;
# skipped steps in skipped.txt); disk about 200 GB.
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
export BASE; export -f build_tree platform getmodel
T0=$(date +%s); BUDGET=$(( 4 * 3600 )); RESERVE=600
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
run_cell() {  # tag gguf corpus C law full(1|0): the four configurations at every cell (full is ignored)
  local tag=$1 g=$2 corpus=$3 C=$4 law=$5 full=$6
  local common="--corpus $corpus -t $CORES --n-prefill 1024 --n-decode 256 --no-mmap --host-experts"
  local LA=$OUT/la_$tag.bin
  local base="slots=$C:policy=dfa:$MB:fetch=$law:paced=0:overlap=1:oracle_w=0"
  local orc="slots=$C:policy=oracle:oracle=$LA:$MB:fetch=$law:overlap=1"
  local cfg="$base:stats=$OUT/st_${tag}_C${C}_base.json"
  cfg="$cfg;$orc:oracle_bypass=1:oracle_w=0:$PACED:stats=$OUT/st_${tag}_C${C}_bypass_w0.json"
  cfg="$cfg;$orc:oracle_bypass=1:oracle_w=16:$PACED:stats=$OUT/st_${tag}_C${C}_bypass_w16.json"
  cfg="$cfg;$orc:oracle_bypass=0:oracle_w=0:$PACED:stats=$OUT/st_${tag}_C${C}_prefetch_w0.json"
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
        if "bypass_w0" in lab or lab.endswith("_base"):
            line.append(f"{lab.replace('st_', '')} {m:.1f}")
open(f"{out}/oneline.txt", "w").write("094 " + " ".join(line) + f"; device read {open(f'{out}/gate.txt').read().strip().splitlines()[-2].split()[-2] if os.path.exists(f'{out}/gate.txt') else '?'} GB/s\n")
PY
ONE=$(cat $OUT/oneline.txt)
python3 $J/manifest.py "$OUT" 094_foresight_bytes@vast "$T0" "$ONE" law_gptoss=$LAWG law_qwen3=$LAWQ device_read_gbs=$DR memory_clock_max_mhz=$MEMCLK
cat $OUT/skipped.txt 2>/dev/null
echo "SUMMARY: $ONE"
