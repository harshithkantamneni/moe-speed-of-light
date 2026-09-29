#!/bin/bash
# PREFETCH on the home-PC-class host of jobs 060 / 064 (RTX 5090 + Ryzen 9 9950X): gpt-oss-120b own text, 12 prompts x
# 128 tokens, at C = 14 / 32 / 56, in one process per run on one loaded model: the cache without prefetch, prefetch
# q=1 (top-4 predictions), q=1 + FETCH, q=2, FETCH alone, and the first two again in reverse order (drift check). Then
# the same client runs as job 064 (FreeToken's method, AIME-25 problems 0-4, 256 tokens, llama-server) for the cache
# with prefetch at C = 14 / 32 / 51 / 56, to set beside job 064's numbers. The host is measured first (STREAM, bw.cu,
# concur.cu). Job 065 verified prefetch on gpt-oss-20b (ids consistent on 1,536 steps, CUDA graphs on/off identical,
# memcheck clean, hit rate as simulated) on a PCIe 4.0 host where it lost 16 %, as FETCH did there.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip >> $OUT/apt.txt 2>&1
python3 -m pip install -q --break-system-packages huggingface_hub hf_xet > $OUT/pip.txt 2>&1
CORES=$(lscpu -p=CORE | grep -v '^#' | sort -u | wc -l); echo "physical cores $CORES" | tee $OUT/cores.txt
R() { local t=$1; shift; timeout "$t" "$@"; }
( t0=$(date +%s); mkdir -p $M
  python3 -c "import sys; from huggingface_hub import hf_hub_download as d; d('ggml-org/gpt-oss-120b-GGUF', 'gpt-oss-120b-MXFP4.gguf', local_dir=sys.argv[1])" "$M"
  echo "gguf download rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $M/gpt-oss-120b-MXFP4.gguf
  [ -s $M/gpt-oss-120b-MXFP4.gguf ] || getmodel ggml-org/gpt-oss-120b-GGUF gpt-oss-120b-MXFP4.gguf ) > $OUT/dl_gguf.txt 2>&1 &
DLG=$!
( cd $WORK && git init -q ft && cd ft && git remote add origin https://github.com/FlashML-org/FreeToken &&
  git fetch -q --depth 1 origin 0d652e73a452d014ac5441a15baa75348e9fcb0a && git checkout -q FETCH_HEAD ) || echo "FreeToken clone FAILED"
export FT_DIR=$WORK/ft
python3 -c "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('math-ai/aime25','test.jsonl',repo_type='dataset'), '$WORK/aime25.jsonl')" && export FREETOKEN_AIME25_JSONL=$WORK/aime25.jsonl
build_tree $WORK/lc-ec "$J/llama.cpp-expert-cache-4da6337.patch" llama-ec-bench llama-server
wait $DLG; cat $OUT/dl_gguf.txt
F=$M/gpt-oss-120b-MXFP4.gguf
platform
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
H=$(( CORES > 4 ? CORES - 2 : 2 )); MB="mailbox=1:tdec=1:helpers=$H"
COMMON="--corpus $J/S_gpt-oss-120b.jsonl --seqs 0,1,2,3,4,5,6,7,8,9,10,11 -t $CORES --n-prefill 640 --n-decode 128 --no-mmap --host-experts"
for C in 14 32 56; do
  B="slots=$C:policy=dfa:kappa=1:$MB"
  CFG="$B;$B:stats=$OUT/C${C}_base.json;$B:prefetch=1:stats=$OUT/C${C}_pf1.json;$B:prefetch=1:fetch=0,1,1,2,3:stats=$OUT/C${C}_pf1f.json"
  CFG="$CFG;$B:prefetch=2:stats=$OUT/C${C}_pf2.json;$B:fetch=0,1,1,2,3:stats=$OUT/C${C}_fetch.json"
  CFG="$CFG;$B:prefetch=1:stats=$OUT/C${C}_pf1_r.json;$B:stats=$OUT/C${C}_base_r.json"
  R 60m $EC_BIN -m $F $COMMON --ec "$CFG" --dump $OUT/ab_C$C.bin > $OUT/ab_C$C.jsonl 2> $OUT/ab_C$C.err; echo "ab C=$C rc=$?"
done
# the client of job 064 (FreeToken's method) for the cache with prefetch
EC_SERVER=$WORK/lc-ec/build/bin/llama-server
CL="python3 $J/bs1_client.py --model $M/no-hf-checkpoint --problems 0,1,2,3,4 --decode 256 --json $OUT/bs1.jsonl"
LX='{"min_p": 0.0, "seed": 1234}'
LS="-m $F -ngl 99 -fa on -lm none -t $CORES -np 1 -c 8448 --no-webui"
OT="-ot '\\.ffn_(up|down|gate|gate_up)_exps=CUDA_Host'"
for C in 14 32 51 56; do
  ECENV="env LLAMA_EC_SLOTS=$C LLAMA_EC_POLICY=dfa LLAMA_EC_KAPPA=1 LLAMA_EC_MAILBOX=1 LLAMA_EC_TDEC=1 LLAMA_EC_HELPERS=$H"
  R 40m $CL --label ourspf_C$C --meta "{\"system\": \"ours+prefetch\", \"C\": $C}" --extra "$LX" --log $OUT/srv_ourspf_C$C.log \
    --cmd "$ECENV LLAMA_EC_PREFETCH=1 LLAMA_EC_STATS=$OUT/srv_ourspf_C$C.json $EC_SERVER $LS $OT --port {port}"
done
for f in $OUT/srv_*.log; do tail -n 40 "$f" > "$f.tail"; done; find $OUT -name 'srv_*.log' -size +2M -delete
python3 - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, sys, os, statistics as st
import numpy as np
out = sys.argv[1]
def top1(p): return np.fromfile(p, dtype=np.int32).reshape(-1, 10)[:, 0] if os.path.exists(p) and os.path.getsize(p) else None
for C in (14, 32, 56):
    p = f"{out}/ab_C{C}.jsonl"
    if not os.path.exists(p): print(f"C={C}: missing"); continue
    by, order = {}, []
    for l in open(p):
        if l.strip():
            r = json.loads(l); k = r.get("config", "")
            if k not in by: order.append(k)
            by.setdefault(k, []).append(r)
    base_rate, base_t = None, top1(f"{out}/ab_C{C}.bin.1")
    for i, k in enumerate(order):
        if "stats=" not in k: continue
        st_path = k.split("stats=")[1].split(":")[0]; s = json.load(open(st_path)) if os.path.exists(st_path) else {}
        v = by[k]; rate = sum(r["n_decode"] for r in v) / (sum(r["decode_ms"] for r in v) / 1000)
        nll = sum(r["nll_sum"] for r in v) / max(1, sum(r["nll_n"] for r in v))
        if base_rate is None: base_rate = rate
        steps = max(1, s.get("steps", 1)); L = len(s.get("layers", [])) or 1
        t = top1(f"{out}/ab_C{C}.bin.{i}")
        agree = float((t == base_t).mean()) if t is not None and base_t is not None and len(t) == len(base_t) else float("nan")
        print(f"{os.path.basename(st_path)[:-5]:14s} {rate:7.1f} tok/s ({rate / base_rate:.3f}x)  nll {nll:.4f}  hit {s.get('hit_rate', 0):.4f}  "
              f"prefetch/ls {s.get('prefetches', 0) / steps / L:.3f} useful {s.get('prefetch_useful', 0) / steps / L:.3f}  fetches {s.get('fetches')}  top1 vs base {agree:.4f}")
if os.path.exists(f"{out}/bs1.jsonl"):
    by = {}
    for l in open(f"{out}/bs1.jsonl"):
        r = json.loads(l); by.setdefault(r["label"], []).append(r)
    print("bs=1 via the chat API (job 064's client and prompts)")
    for lab, v in by.items():
        print(f"{lab:16s} {st.mean(r['decode_tok_s'] for r in v):7.2f} tok/s (n={len(v)})")
PY
