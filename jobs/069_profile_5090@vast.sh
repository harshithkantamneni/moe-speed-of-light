#!/bin/bash
# Where does the time go? Nsight Systems timelines of gpt-oss-120b decode on an RTX 5090 + Ryzen 9 9950X host (rerun:
# the first run's results were lost when its machine was destroyed after a failed log fetch) of the class of jobs
# 060 / 064 / 066 (physical speed limit 313 tok/s at 25% of experts resident, the same as all resident; measured
# 63-115 tok/s for our cache). One sequence (own text, seq 0), 640 prefilled tokens, 24 decode tokens per profiled
# run; CUDA graphs traced per node. Runs: the cache at C = 14 / 32 / 56, the cache + FETCH and + FETCH + PREFETCH at
# C = 32, and stock llama.cpp placement (--ncmoe 27, inside ec-bench). Every configuration also runs without the
# profiler on 12 sequences x 128 tokens (tok/s reference, profiler overhead check). The host is measured first.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
. "$J/prof.sh"
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip >> $OUT/apt.txt 2>&1
python3 -m pip install -q --break-system-packages huggingface_hub hf_xet > $OUT/pip.txt 2>&1
CORES=$(lscpu -p=CORE | grep -v '^#' | sort -u | wc -l); echo "physical cores $CORES" | tee $OUT/cores.txt
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
B="policy=dfa:kappa=1:$MB"
# reference speeds without the profiler (12 x 128 own text), one process
CFG="slots=32:$B"
for c in "slots=14:$B" "slots=32:$B" "slots=56:$B" "slots=32:$B:fetch=0,1,1,2,3" "slots=32:$B:fetch=0,1,1,2,3:prefetch=1" "slots=32:$B:prefetch=1:prefetch_late=1" "slots=32:$B:llc=1"; do
  CFG="$CFG;$c:stats=$OUT/ref_$(echo $c | cut -d: -f1 | tr = _)$(echo $c | grep -q ':fetch=' && echo _f)$(echo $c | grep -q prefetch= && echo _pf)$(echo $c | grep -q prefetch_late && echo late)$(echo $c | grep -q llc= && echo _llc).json"
done
ALL="--corpus $J/S_gpt-oss-120b.jsonl --seqs 0,1,2,3,4,5,6,7,8,9,10,11 -t $CORES --n-prefill 640 --n-decode 128 --no-mmap"
R 60m $EC_BIN -m $F $ALL --host-experts --ec "$CFG" > $OUT/ref_ec.jsonl 2> $OUT/ref_ec.err; echo "ref ec rc=$?"
R 30m $EC_BIN -m $F $ALL --ncmoe 27 > $OUT/ref_static27.jsonl 2> $OUT/ref_static27.err; echo "ref static rc=$?"
# profiled runs: one sequence, 24 decode tokens, one configuration per process (warm-up on seq 1 first)
ONE="--corpus $J/S_gpt-oss-120b.jsonl -t $CORES --n-prefill 640 --n-decode 24 --no-mmap"
for c in "C14:slots=14:$B" "C32:slots=32:$B" "C56:slots=56:$B" "C32f:slots=32:$B:fetch=0,1,1,2,3" "C32fpf:slots=32:$B:fetch=0,1,1,2,3:prefetch=1" "C32pflate:slots=32:$B:prefetch=1:prefetch_late=1" "C32llc:slots=32:$B:llc=1"; do
  lab=${c%%:*}; cfg=${c#*:}
  prof_run $lab $EC_BIN -m $F $ONE --seqs 1,0 --host-experts --ec "$cfg:seqs=1/0"
done
prof_run static27 $EC_BIN -m $F $ONE --seqs 1,0 --ncmoe 27
python3 - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, os, sys
out = sys.argv[1]
def rate(p):
    by = {}
    for l in open(p):
        if l.strip():
            r = json.loads(l); by.setdefault(r.get("config", ""), []).append(r)
    return {k: sum(r["n_decode"] for r in v) / (sum(r["decode_ms"] for r in v) / 1000) for k, v in by.items()}
for f in ("ref_ec.jsonl", "ref_static27.jsonl"):
    if os.path.exists(f"{out}/{f}"):
        for k, v in rate(f"{out}/{f}").items(): print(f"reference {f}: {k.split('stats=')[-1].split('/')[-1] or 'static/warm-up'} {v:.1f} tok/s")
for f in sorted(os.listdir(out)):
    if f.startswith("prof_") and f.endswith(".stdout"):
        for l in open(f"{out}/{f}"):
            if l.startswith("{"):
                r = json.loads(l); print(f"profiled {f[5:-7]} seq {r['seq']}: {r['tok_s']:.1f} tok/s (under nsys)")
PY
