#!/bin/bash
# The "if the 5090 had 96 GB" reference: RTX PRO 6000 Blackwell (GB202, the RTX 5090's die, 96 GB, 1.8 TB/s) runs
# gpt-oss-120b with every expert resident, own text 12 x 128 (ec-bench, -ngl 99) and llama-bench tg128 at depth 512,
# plus an Nsight Systems timeline of 24 decode tokens. This measures the GPU-side speed that no offloading can beat on
# this die, next to the 313 tok/s physical limit. The host is measured, and the offloaded runs of job 069 (cache at
# C = 32 with and without FETCH, stock --ncmoe 27) are repeated on this host's CPU as a third host type.
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
build_tree $WORK/lc-stock "" llama-bench
wait $DLG; cat $OUT/dl_gguf.txt
F=$M/gpt-oss-120b-MXFP4.gguf
platform
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
ALL="--corpus $J/S_gpt-oss-120b.jsonl --seqs 0,1,2,3,4,5,6,7,8,9,10,11 -t $CORES --n-prefill 640 --n-decode 128 --no-mmap"
R 30m $EC_BIN -m $F $ALL > $OUT/ref_allgpu.jsonl 2> $OUT/ref_allgpu.err; echo "all-GPU rc=$?"
R 30m $STOCK_BENCH -m $F -ngl 99 -fa on -lm none -t $CORES -p 0 -n 128 -d 512 -r 5 -o jsonl > $OUT/llamabench_allgpu.jsonl 2> $OUT/llamabench_allgpu.err; echo "llama-bench rc=$?"
ONE="--corpus $J/S_gpt-oss-120b.jsonl -t $CORES --n-prefill 640 --n-decode 24 --no-mmap"
prof_run allgpu $EC_BIN -m $F $ONE --seqs 1,0
# the offloaded configurations on this host (third host type)
H=$(( CORES > 4 ? CORES - 2 : 2 )); B="policy=dfa:kappa=1:mailbox=1:tdec=1:helpers=$H"
R 60m $EC_BIN -m $F $ALL --host-experts --ec "slots=32:$B;slots=32:$B:stats=$OUT/ref_C32.json;slots=32:$B:fetch=0,1,1,2,3:stats=$OUT/ref_C32_f.json" > $OUT/ref_ec.jsonl 2> $OUT/ref_ec.err; echo "ref ec rc=$?"
R 30m $EC_BIN -m $F $ALL --ncmoe 27 > $OUT/ref_static27.jsonl 2> $OUT/ref_static27.err; echo "ref static rc=$?"
python3 - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, os, sys
out = sys.argv[1]
def rate(p):
    by = {}
    for l in open(p):
        if l.strip():
            r = json.loads(l); by.setdefault(r.get("config", ""), []).append(r)
    return {k: sum(r["n_decode"] for r in v) / (sum(r["decode_ms"] for r in v) / 1000) for k, v in by.items()}
for f in ("ref_allgpu.jsonl", "ref_ec.jsonl", "ref_static27.jsonl"):
    if os.path.exists(f"{out}/{f}"):
        for k, v in rate(f"{out}/{f}").items(): print(f"{f}: {k.split('stats=')[-1].split('/')[-1] or '-'} {v:.1f} tok/s")
if os.path.exists(f"{out}/llamabench_allgpu.jsonl"):
    for l in open(f"{out}/llamabench_allgpu.jsonl"):
        if l.strip().startswith("{"):
            r = json.loads(l); print(f"llama-bench all-GPU tg128@d512: {r.get('avg_ts')} tok/s (sd {r.get('stddev_ts')})")
PY
