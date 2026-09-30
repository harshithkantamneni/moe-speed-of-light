#!/bin/bash
# Job 084c: the gpt-oss-120b routing trace of the measured workload (job 084's part B for gpt-oss, which was lost: the
# full lookahead record exceeded the 8 MB result limit). Input: job 084's own greedy text (the 30 AIME-25 prompts as
# llama-server templates and tokenises them, 256 tokens generated all on the GPU by llama-ec-bench, --gen-temp 1e-4),
# committed as ec2/aime25_owntext_gptoss_084.jsonl. Teacher-forced over it with --lookahead; only the router's choices
# are kept (uint8). Any RTX 5090 host: routing does not depend on the machine (experts of 24 of 36 layers on the CPU).
# Feeds the speed-limit column and job 084's prediction 4 (carried over unchanged).
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip >> $OUT/apt.txt 2>&1
CORES=$(usable_cores); echo "usable physical cores $CORES (nproc $(nproc))" | tee $OUT/cores.txt
python3 -m pip install -q --break-system-packages huggingface_hub hf_xet numpy > $OUT/pip.txt 2>&1
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
build_tree $WORK/lc-ec "$J/llama.cpp-expert-cache-4da6337.patch" llama-ec-bench
wait $DLG
cat $OUT/dl_gguf.txt
F=$M/gpt-oss-120b-MXFP4.gguf
nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv > $OUT/gpu.csv; lscpu > $OUT/lscpu.txt
wc -l $J/aime25_owntext_gptoss_084.jsonl
timeout 60m $EC_BIN -m $F --corpus $J/aime25_owntext_gptoss_084.jsonl -t $CORES --n-prefill 1024 --n-decode 256 --ncmoe 24 \
  --lookahead $OUT/la_aime25_gptoss.bin > $OUT/la_aime25_gptoss.jsonl 2> $OUT/la_gptoss.err
echo "lookahead rc=$? $(ls -la $OUT/la_aime25_gptoss.bin)"
tail -n 3 $OUT/la_gptoss.err
python3 - $OUT/la_aime25_gptoss.bin $OUT/route_aime25_gptoss.npz <<'PY'
import json, sys, numpy as np
p, o = sys.argv[1], sys.argv[2]
h = json.load(open(p + ".json")); L, k = h["n_layer"], h["k"]
a = np.fromfile(p, dtype=np.int16).reshape(-1, 2 + L * (2 * k + 24))
act = a[:, 2:].reshape(-1, L, 2 * k + 24)[:, :, :k]
assert act.min() >= 0 and act.max() < 256
np.savez_compressed(o, seq=a[:, 0], step=a[:, 1], act=act.astype(np.uint8), n_expert=h["n_expert"], n_layer=L, k=k)
print("routes", act.shape, "->", o)
PY
rm -f $OUT/la_aime25_gptoss.bin
python3 - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, os, sys
out = sys.argv[1]
v = [json.loads(l) for l in open(f"{out}/la_aime25_gptoss.jsonl") if l.startswith("{")]
v = [r for r in v if "nll_sum" in r]
if v:
    print(f"gptoss: teacher-forced own text, {len(v)} sequences, NLL {sum(r['nll_sum'] for r in v) / sum(r['nll_n'] for r in v):.3f} nats/token")
print("routes file:", os.path.exists(f"{out}/route_aime25_gptoss.npz"))
PY
