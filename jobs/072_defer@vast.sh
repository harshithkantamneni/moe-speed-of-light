#!/bin/bash
# Job 072: deferred admission copies (LLAMA_EC_DEFER_ADMIT=1), A/B against the current cache on an RTX 5090 host.
# Why: the job 069c profile (Core Ultra 7 270K host) shows ~2 ms per token between the output head and the next
# token's first kernel. The admission copies issued after each step (zero-copy kernels on the copy stream) saturate the
# link while the host uploads the next token's inputs (about 81 small H2D copies), so the next launch waits behind them.
# DEFER_ADMIT issues the same copies right after the next graph is launched. Same policy, same bytes.
# Prediction, recorded here before the run (commit time on the public gpu branch is the timestamp):
#   1. host_us_per_step.inputs falls by at least half with defer=1, at C = 32 and 56;
#   2. tok/s with defer=1 over defer=0: C14 0 to +10%, C32 +3 to +15%, C56 +5 to +20%; FETCH at C32 within +-3%
#      (few admissions);
#   3. hit rates and admissions are identical (the policy does not change), top-1 agreement 100%.
# If the gain is below +3% at C32 and C56, the ~2 ms is not the admissions' doing and the boundary is host work.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
. "$J/prof.sh"
export PROF_KEEP_TAIL="C32 C32d"
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip >> $OUT/apt.txt 2>&1
python3 -m pip install -q --break-system-packages huggingface_hub hf_xet > $OUT/pip.txt 2>&1
CORES=$(usable_cores); echo "usable physical cores $CORES (lscpu: $(lscpu -p=CORE | grep -v '^#' | sort -u | wc -l), nproc $(nproc))" | tee $OUT/cores.txt
R() { local t=$1; shift; timeout "$t" "$@"; }
( t0=$(date +%s); mkdir -p $M
  python3 -c "import sys; from huggingface_hub import hf_hub_download as d; d('ggml-org/gpt-oss-120b-GGUF', 'gpt-oss-120b-MXFP4.gguf', local_dir=sys.argv[1])" "$M"
  echo "gguf download rc=$? in $(( $(date +%s) - t0 )) s"
  [ -s $M/gpt-oss-120b-MXFP4.gguf ] || getmodel ggml-org/gpt-oss-120b-GGUF gpt-oss-120b-MXFP4.gguf ) > $OUT/dl_gguf.txt 2>&1 &
DLG=$!
install_nsys
build_tree $WORK/lc-ec "$J/llama.cpp-expert-cache-4da6337.patch" llama-ec-bench
wait $DLG; cat $OUT/dl_gguf.txt
F=$M/gpt-oss-120b-MXFP4.gguf
platform
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
H=$(( CORES > 4 ? CORES - 2 : 2 )); MB="mailbox=1:tdec=1:helpers=$H"
# pre-registered law prediction for this host, from its own probe, before any model run (law_predict.py, frozen constants)
python3 $J/law_predict.py $OUT/concur.txt $H "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" > $OUT/law_prediction.json 2>&1
echo "law prediction written $(date -u +%FT%TZ)"; cat $OUT/law_prediction.json
B="policy=dfa:kappa=1:$MB"
# reference speeds (12 x 128 own text), one process, two repetitions
CFG="slots=32:$B"
for rep in 1 2; do
  for c in "C14:slots=14:$B" "C14d:slots=14:$B:defer_admit=1" "C32:slots=32:$B" "C32d:slots=32:$B:defer_admit=1" "C56:slots=56:$B" "C56d:slots=56:$B:defer_admit=1" \
           "C32f:slots=32:$B:fetch=0,1,1,2,3" "C32fd:slots=32:$B:fetch=0,1,1,2,3:defer_admit=1" "C32pf:slots=32:$B:prefetch=1" "C32pfd:slots=32:$B:prefetch=1:defer_admit=1"; do
    lab=${c%%:*}; cfg=${c#*:}
    CFG="$CFG;$cfg:stats=$OUT/ref_${lab}_r$rep.json"
  done
done
ALL="--corpus $J/S_gpt-oss-120b.jsonl --seqs 0,1,2,3,4,5,6,7,8,9,10,11 -t $CORES --n-prefill 640 --n-decode 128 --no-mmap"
R 60m $EC_BIN -m $F $ALL --host-experts --ec "$CFG" > $OUT/ref_ec.jsonl 2> $OUT/ref_ec.err; echo "ref ec rc=$?"
R 30m $EC_BIN -m $F $ALL --ncmoe 27 > $OUT/ref_static27.jsonl 2> $OUT/ref_static27.err; echo "ref static rc=$?"
ONE="--corpus $J/S_gpt-oss-120b.jsonl -t $CORES --n-prefill 640 --n-decode 24 --no-mmap"
for c in "C32:slots=32:$B" "C32d:slots=32:$B:defer_admit=1"; do
  lab=${c%%:*}; cfg=${c#*:}
  prof_run $lab $EC_BIN -m $F $ONE --seqs 1,0 --host-experts --ec "$cfg:seqs=1/0"
done
python3 - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, os, sys
out = sys.argv[1]
by = {}
for l in open(f"{out}/ref_ec.jsonl"):
    if l.strip():
        r = json.loads(l); by.setdefault(r.get("config", ""), []).append(r)
rate = {k.split("stats=")[-1].split("/")[-1]: sum(r["n_decode"] for r in v) / (sum(r["decode_ms"] for r in v) / 1000) for k, v in by.items()}
for k, v in sorted(rate.items()):
    st = {}
    p = f"{out}/{k}"
    if os.path.exists(p):
        st = json.load(open(p))
    h = st.get("host_us_per_step", {})
    print(f"{k:24s} {v:6.1f} tok/s  hit {st.get('hit_rate', 0):.4f} admits/step {st.get('admits', 0) / max(1, st.get('steps', 1)):5.2f}  host us/step " + " ".join(f"{a}={h.get(a, 0):.0f}" for a in ("app", "pre", "inputs", "launch", "issue", "sync", "post")))
PY
