#!/bin/bash
# Overlap test on the RTX 5090 + Ryzen 7 9800X3D host (96 MB L3 on one CCD). The host-DRAM law says decode time is
# G + host-DRAM bytes / bandwidth, with nothing overlapping the two terms; host memory is idle during the GPU's own
# work (G, about 4.8 ms per token). Two ways to read future experts' bytes during that window, both driven by the
# next-layer prediction (84% / 97% recall at top-4 / top-8):
#   PREFETCH_LATE: the prefetch copy into GPU slots starts when this layer's helpers finish (i.e. during the next
#                  layer's attention), instead of at this layer's REQUEST (when the helpers are reading host memory);
#   LLC:           idle helpers read the next layer's predicted CPU experts into the CPU's L3, so their computation
#                  later reads the cache instead of DRAM.
# gpt-oss-120b own text, 12 x 128 tokens, C = 14 / 32 / 56, all configurations of a budget in one process, with a
# repeat of the base and of the best candidate at the end (drift check). The law's prediction for each configuration is
# computed from its own byte counts afterwards.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip >> $OUT/apt.txt 2>&1
python3 -m pip install -q --break-system-packages huggingface_hub hf_xet > $OUT/pip.txt 2>&1
CORES=$(lscpu -p=CORE | grep -v '^#' | sort -u | wc -l); echo "physical cores $CORES" | tee $OUT/cores.txt
lscpu | grep -i "L3" | tee $OUT/l3.txt
R() { local t=$1; shift; timeout "$t" "$@"; }
( t0=$(date +%s); mkdir -p $M
  python3 -c "import sys; from huggingface_hub import hf_hub_download as d; d('ggml-org/gpt-oss-120b-GGUF', 'gpt-oss-120b-MXFP4.gguf', local_dir=sys.argv[1])" "$M"
  echo "gguf download rc=$? in $(( $(date +%s) - t0 )) s"
  [ -s $M/gpt-oss-120b-MXFP4.gguf ] || getmodel ggml-org/gpt-oss-120b-GGUF gpt-oss-120b-MXFP4.gguf ) > $OUT/dl_gguf.txt 2>&1 &
DLG=$!
build_tree $WORK/lc-ec "$J/llama.cpp-expert-cache-4da6337.patch" llama-ec-bench
wait $DLG; cat $OUT/dl_gguf.txt
F=$M/gpt-oss-120b-MXFP4.gguf
platform
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
H=$(( CORES > 4 ? CORES - 2 : 2 )); B="policy=dfa:kappa=1:mailbox=1:tdec=1:helpers=$H"
ALL="--corpus $J/S_gpt-oss-120b.jsonl --seqs 0,1,2,3,4,5,6,7,8,9,10,11 -t $CORES --n-prefill 640 --n-decode 128 --no-mmap --host-experts"
for C in 14 32 56; do
  b="slots=$C:$B"
  CFG="$b"
  CFG="$CFG;$b:stats=$OUT/C${C}_base.json"
  CFG="$CFG;$b:prefetch=1:stats=$OUT/C${C}_pf.json"
  CFG="$CFG;$b:prefetch=1:prefetch_late=1:stats=$OUT/C${C}_pflate.json"
  CFG="$CFG;$b:llc=1:stats=$OUT/C${C}_llc.json"
  CFG="$CFG;$b:llc=1:prefetch=1:prefetch_late=1:stats=$OUT/C${C}_llc_pflate.json"
  CFG="$CFG;$b:fetch=0,1,1,2,3:stats=$OUT/C${C}_fetch.json"
  CFG="$CFG;$b:llc=1:fetch=0,1,1,2,3:stats=$OUT/C${C}_llc_fetch.json"
  CFG="$CFG;$b:llc=1:prefetch=1:prefetch_late=1:fetch=0,1,1,2,3:stats=$OUT/C${C}_all.json"
  CFG="$CFG;$b:stats=$OUT/C${C}_base_r.json"
  R 60m $EC_BIN -m $F $ALL --ec "$CFG" --dump $OUT/ab_C$C.bin > $OUT/ab_C$C.jsonl 2> $OUT/ab_C$C.err; echo "ab C=$C rc=$?"
done
python3 - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, os, sys
import numpy as np
out = sys.argv[1]
def top1(p): return np.fromfile(p, dtype=np.int32).reshape(-1, 10)[:, 0] if os.path.exists(p) and os.path.getsize(p) else None
for C in (14, 32, 56):
    p = f"{out}/ab_C{C}.jsonl"
    if not os.path.exists(p): continue
    by, order = {}, []
    for l in open(p):
        if l.strip():
            r = json.loads(l); k = r.get("config", "")
            if k not in by: order.append(k)
            by.setdefault(k, []).append(r)
    base, bt = None, top1(f"{out}/ab_C{C}.bin.1")
    for i, k in enumerate(order):
        if "stats=" not in k: continue
        sp = k.split("stats=")[1].split(":")[0]; s = json.load(open(sp)) if os.path.exists(sp) else {}
        v = by[k]; rate = sum(r["n_decode"] for r in v) / (sum(r["decode_ms"] for r in v) / 1000)
        base = base or rate
        t = top1(f"{out}/ab_C{C}.bin.{i}")
        ag = float((t == bt).mean()) if t is not None and bt is not None and len(t) == len(bt) else float("nan")
        st = max(1, s.get("steps", 1)); L = len(s.get("layers", [])) or 1
        print(f"{os.path.basename(sp)[:-5]:16s} {rate:7.1f} tok/s ({rate/base:.3f}x) hit {s.get('hit_rate', 0):.4f} pf/ls {s.get('prefetches', 0)/st/L:.3f} "
              f"fetch/ls {s.get('fetches', 0)/st/L:.3f} llc MB/token {s.get('llc_bytes', 0)/st/1e6:.0f} top1 {ag:.4f}")
PY
