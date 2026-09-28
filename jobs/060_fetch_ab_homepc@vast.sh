#!/bin/bash
# FETCH A/B on the home-PC-class host (RTX 5090 + Ryzen 9 9950X, as job 059): gpt-oss-120b own text, 12 prompts x 128
# tokens, at C = 14 / 32 / 56 slots per layer. Per budget, in one process on one loaded model: the cache without
# FETCH, with FETCH overlapped with the helpers (tables 0,1,1,2,3 / 0,0,1,1,2 / 0,1,1,2,2), and FETCH copied before
# the hand-off (job 059's form), then the first two again in reverse order as a drift check. Job 059 measured +2-6 %
# for the serial form; the overlapped form should do better if the combined DRAM read (~75 GB/s measured) is the
# limit rather than either path (~62 GB/s CPU, ~57 GB/s PCIe).
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > $OUT/gpu.csv; lscpu > $OUT/lscpu.txt; free -g > $OUT/free.txt
CORES=$(lscpu -p=CORE | grep -v '^#' | sort -u | wc -l)
build_tree $WORK/lc-ec "$J/llama.cpp-expert-cache-4da6337.patch" llama-ec-bench
getmodel ggml-org/gpt-oss-120b-GGUF gpt-oss-120b-MXFP4.gguf
H=$(( CORES > 4 ? CORES - 2 : 2 )); MB="mailbox=1:tdec=1:helpers=$H"
CFG="slots=14:policy=dfa:kappa=1:$MB"
for C in 14 32 56; do
  B="slots=$C:policy=dfa:kappa=1:$MB"
  CFG="$CFG;$B:stats=$OUT/C${C}_base.json"
  CFG="$CFG;$B:fetch=0,1,1,2,3:stats=$OUT/C${C}_f01123.json"
  CFG="$CFG;$B:fetch=0,0,1,1,2:stats=$OUT/C${C}_f00112.json"
  CFG="$CFG;$B:fetch=0,1,1,2,2:stats=$OUT/C${C}_f01122.json"
  CFG="$CFG;$B:fetch=0,1,1,2,3:fetch_overlap=0:stats=$OUT/C${C}_f01123serial.json"
  CFG="$CFG;$B:fetch=0,1,1,2,3:stats=$OUT/C${C}_f01123_r.json"
  CFG="$CFG;$B:stats=$OUT/C${C}_base_r.json"
done
timeout 90m $EC_BIN -m $M/gpt-oss-120b-MXFP4.gguf --corpus $J/S_gpt-oss-120b.jsonl --seqs 0,1,2,3,4,5,6,7,8,9,10,11 -t $CORES \
  --n-prefill 640 --n-decode 128 --no-mmap --host-experts --ec "$CFG" --dump $OUT/ab.bin > $OUT/ab.jsonl 2> $OUT/ab.err
echo "ab rc=$?"
python3 - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, sys, os, glob
import numpy as np
out = sys.argv[1]
by, order = {}, []
for l in open(f"{out}/ab.jsonl"):
    if l.strip():
        r = json.loads(l); k = r.get("config", "")
        if k not in by: order.append(k)
        by.setdefault(k, []).append(r)
def rate(v): return sum(r["n_decode"] for r in v) / (sum(r["decode_ms"] for r in v) / 1000)
def nll(v): return sum(r["nll_sum"] for r in v) / max(1, sum(r["nll_n"] for r in v))
def top1(i): p = f"{out}/ab.bin.{i}"; return np.fromfile(p, dtype=np.int32).reshape(-1, 10)[:, 0] if os.path.exists(p) and os.path.getsize(p) else None
rows = []
for i, k in enumerate(order):
    if "stats=" not in k: continue
    st = k.split("stats=")[1].split(":")[0]; name = os.path.basename(st)[:-5]
    s = json.load(open(st)) if os.path.exists(st) else {}
    rows.append((name, rate(by[k]), nll(by[k]), s.get("hit_rate"), s.get("fetches"), top1(i)))
base = {}
for name, r, n, h, f, t in rows:
    C = name.split("_")[0]
    if name.endswith("_base"): base[C] = (r, t)
for name, r, n, h, f, t in rows:
    C = name.split("_")[0]; b = base.get(C)
    agree = float((t == b[1]).mean()) if b and t is not None and b[1] is not None and len(t) == len(b[1]) else None
    print(f"{name:22s} {r:7.1f} tok/s  nll {n:.4f}  hit {h}  fetches {f}  vs base {r / b[0] if b else float('nan'):.3f}x  top1 vs base {agree}")
PY
