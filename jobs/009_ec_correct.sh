#!/bin/bash
# Correctness of the expert cache on the real models: top-5 logits of 3 test sequences x 64 decode steps,
# cache (hits on GPU, misses on CPU; DFA, LRU; overlap on/off) against all experts on the CPU and all on the GPU.
set -x
exec 2>&1
J=$(cd "$(dirname "$0")" && pwd)/ec
B=$WORK/llama.cpp-ec/build/bin/llama-ec-bench
M=$WORK/models
T=30
run() { $B "$@" -t $T --n-prefill 128 --n-decode 64 2>>"$OUT/stderr_runs.txt"; }
for spec in "gpt-oss-20b-MXFP4.gguf tok_gpt-oss-20b.jsonl 24 8" "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf tok_qwen3_30b.jsonl 48 32"; do
  set -- $spec; f=$1; corp=$J/$2; L=$3; C=$4; tag=${f%.gguf}
  run -m $M/$f --corpus $corp --seqs 4,7,8 --ncmoe $L --dump $OUT/${tag}_cpu.bin > $OUT/${tag}_cpu.jsonl
  run -m $M/$f --corpus $corp --seqs 4,7,8 --ncmoe 0 --dump $OUT/${tag}_gpu.bin > $OUT/${tag}_gpu.jsonl
  run -m $M/$f --corpus $corp --seqs 4,7,8 --host-experts \
      --ec "slots=$C:policy=dfa:overlap=1:stats=$OUT/${tag}_dfa1.json;slots=$C:policy=dfa:overlap=0:stats=$OUT/${tag}_dfa0.json;slots=$C:policy=lru:overlap=1:stats=$OUT/${tag}_lru1.json" \
      --dump $OUT/${tag}_ec.bin > $OUT/${tag}_ec.jsonl
done
python3 - "$OUT" <<'PY'
import numpy as np, sys, os, json
out = sys.argv[1]
def load(p):
    a = np.fromfile(p, dtype=np.uint8)
    rec = a.reshape(-1, 40)
    return rec[:, :20].copy().view(np.int32), rec[:, 20:].copy().view(np.float32)
res = {}
for tag in ["gpt-oss-20b-MXFP4", "Qwen3-30B-A3B-Instruct-2507-Q4_K_M"]:
    ref_i, ref_v = load(f"{out}/{tag}_cpu.bin")
    for other in ["gpu", "ec.bin.0", "ec.bin.1", "ec.bin.2"]:
        p = f"{out}/{tag}_{other}" if other.startswith("ec") else f"{out}/{tag}_{other}.bin"
        if not os.path.exists(p): continue
        i, v = load(p)
        n = min(len(i), len(ref_i))
        top1 = float((i[:n, 0] == ref_i[:n, 0]).mean())
        dmax = float(np.abs(v[:n, 0] - ref_v[:n, 0]).max())
        res[f"{tag}:{other}"] = dict(steps=n, top1_agreement=top1, max_abs_top1_logit_diff=dmax,
                                     rel=float(np.abs(v[:n, 0] - ref_v[:n, 0]).max() / np.abs(ref_v[:n, 0]).max()))
        print(tag, other, res[f"{tag}:{other}"])
json.dump(res, open(f"{out}/correctness.json", "w"), indent=1)
PY
