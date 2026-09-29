#!/bin/bash
# PREFETCH (ggml-ec.h), first GPU run: does it work and is it correct? gpt-oss-20b on its own text (12 prompts x 128
# tokens), the cache at C = 8 of 32 experts per layer (mailbox mode), on any CUDA GPU:
#   base / prefetch q=1 with LLAMA_EC_CHECK=1 (the ids the graph used must equal the host maps after the prefetch log)
#   / prefetch q=1 + FETCH / prefetch q=2 / FETCH alone, all in one process on one loaded model, then prefetch q=1
#   again with CUDA graphs off (same dump expected), and compute-sanitizer memcheck on a short prefetch run.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > $OUT/gpu.csv; lscpu > $OUT/lscpu.txt; free -g > $OUT/free.txt
CORES=$(lscpu -p=CORE | grep -v '^#' | sort -u | wc -l)
( getmodel ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-MXFP4.gguf ) > $OUT/download.txt 2>&1 &
build_tree $WORK/lc-ec "$J/llama.cpp-expert-cache-4da6337.patch" llama-ec-bench
wait; cat $OUT/download.txt
F=$M/gpt-oss-20b-MXFP4.gguf
H=$(( CORES > 4 ? CORES - 2 : 2 )); MB="mailbox=1:tdec=1:helpers=$H"
B="slots=8:policy=dfa:kappa=1:$MB"
S="--corpus $J/S_gpt-oss-20b.jsonl --seqs 0,1,2,3,4,5,6,7,8,9,10,11 -t $CORES --n-prefill 640 --n-decode 128 --no-mmap --host-experts"
CFG="$B"
CFG="$CFG;$B:stats=$OUT/base.json"
CFG="$CFG;$B:prefetch=1:check=1:stats=$OUT/pf1_check.json"
CFG="$CFG;$B:prefetch=1:stats=$OUT/pf1.json"
CFG="$CFG;$B:prefetch=1:fetch=0,1,1,2,3:stats=$OUT/pf1_fetch.json"
CFG="$CFG;$B:prefetch=2:stats=$OUT/pf2.json"
CFG="$CFG;$B:fetch=0,1,1,2,3:stats=$OUT/fetch.json"
CFG="$CFG;$B:stats=$OUT/base_r.json"
timeout 60m $EC_BIN -m $F $S --ec "$CFG" --dump $OUT/ab.bin > $OUT/ab.jsonl 2> $OUT/ab.err; echo "ab rc=$?"
GGML_CUDA_DISABLE_GRAPHS=1 timeout 30m $EC_BIN -m $F $S --ec "$B:prefetch=1:stats=$OUT/pf1_nographs.json" --dump $OUT/nog.bin > $OUT/nog.jsonl 2> $OUT/nog.err; echo "no-graphs rc=$?"
timeout 40m compute-sanitizer --tool memcheck --error-exitcode 9 $EC_BIN -m $F --corpus $J/S_gpt-oss-20b.jsonl --seqs 0 -t $CORES \
  --n-prefill 64 --n-decode 24 --no-mmap --host-experts --ec "$B:prefetch=1" > $OUT/memcheck.txt 2>&1; echo "memcheck rc=$?"
tail -n 5 $OUT/memcheck.txt
python3 - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, sys, os
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
def top1(p): return np.fromfile(p, dtype=np.int32).reshape(-1, 10)[:, 0] if os.path.exists(p) and os.path.getsize(p) else None
base_t = top1(f"{out}/ab.bin.1")
for i, k in enumerate(order):
    if "stats=" not in k: continue
    st = k.split("stats=")[1].split(":")[0]; s = json.load(open(st)) if os.path.exists(st) else {}
    t = top1(f"{out}/ab.bin.{i}")
    agree = float((t == base_t).mean()) if t is not None and base_t is not None and len(t) == len(base_t) else None
    steps = max(1, s.get("steps", 1)); L = len(s.get("layers", [])) or 1
    print(f"{os.path.basename(st)[:-5]:14s} {rate(by[k]):7.1f} tok/s  nll {nll(by[k]):.4f}  hit {s.get('hit_rate')}  "
          f"prefetch/layer-step {s.get('prefetches', 0) / steps / L:.3f} useful {s.get('prefetch_useful', 0) / steps / L:.3f}  "
          f"fetches {s.get('fetches')}  check_bad {s.get('check_bad')}/{s.get('check_steps')}  top1 vs base {agree}")
a, b = top1(f"{out}/ab.bin.3"), top1(f"{out}/nog.bin")
if a is not None and b is not None and len(a) == len(b):
    print(f"prefetch q=1, CUDA graphs on vs off: top-1 identical on {float((a == b).mean()):.4f} of steps")
PY
