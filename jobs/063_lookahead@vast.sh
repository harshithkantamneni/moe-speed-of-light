#!/bin/bash
# Can a layer's experts be predicted early enough to prefetch them? For gpt-oss-20b and gpt-oss-120b on their own text
# (12 prompts x 128 decode tokens, teacher-forced, as jobs 058-060), ec-bench --lookahead records per decode step and
# layer the experts the router selected and the top-8 that the router of layer il+1 (il+2) would pick from the
# residual after layer il's attention or from layer il's output, plus a host recomputation of each layer's own
# routing as a check. Any GPU: the prediction does not depend on the machine (gpt-oss-20b all on the GPU; 120b with
# every expert in CPU memory, --ncmoe 36). The eval callback syncs at every observed tensor, so no timings here.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > $OUT/gpu.csv; lscpu > $OUT/lscpu.txt; free -g > $OUT/free.txt
CORES=$(lscpu -p=CORE | grep -v '^#' | sort -u | wc -l)
( getmodel ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-MXFP4.gguf; getmodel ggml-org/gpt-oss-120b-GGUF gpt-oss-120b-MXFP4.gguf ) > $OUT/downloads.txt 2>&1 &
build_tree $WORK/lc-ec "$J/llama.cpp-expert-cache-4da6337.patch" llama-ec-bench
wait; cat $OUT/downloads.txt
S="--seqs 0,1,2,3,4,5,6,7,8,9,10,11 -t $CORES --n-prefill 640 --n-decode 128"
timeout 60m $EC_BIN -m $M/gpt-oss-20b-MXFP4.gguf --corpus $J/S_gpt-oss-20b.jsonl $S --lookahead $OUT/la_gpt-oss-20b.bin > $OUT/la_20b.jsonl 2> $OUT/la_20b.err
echo "20b rc=$?"
timeout 120m $EC_BIN -m $M/gpt-oss-120b-MXFP4.gguf --corpus $J/S_gpt-oss-120b.jsonl $S --ncmoe 36 --no-mmap --lookahead $OUT/la_gpt-oss-120b.bin > $OUT/la_120b.jsonl 2> $OUT/la_120b.err
echo "120b rc=$?"
python3 - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, sys, numpy as np
out = sys.argv[1]
for m in ("gpt-oss-20b", "gpt-oss-120b"):
    try:
        h = json.load(open(f"{out}/la_{m}.bin.json"))
    except Exception as e:
        print(m, "missing", e); continue
    L, k = h["n_layer"], h["k"]; per = 2 * k + 24
    a = np.fromfile(f"{out}/la_{m}.bin", dtype=np.int16).reshape(-1, 2 + L * per)[:, 2:].reshape(-1, L, per)
    act, slf = a[..., :k], a[..., k:2 * k]
    same = np.mean([set(act[s, l]) == set(slf[s, l]) for s in range(len(a)) for l in range(L)])
    def recall(off, n, shift):
        return np.mean([len(set(act[s, l + shift]) & set(a[s, l, off:off + n])) / k for s in range(len(a)) for l in range(L - shift)])
    print(f"{m}: {len(a)} steps, self-check {same:.4f}; recall of layer l+1's experts: from mid(l) top-{k} {recall(2*k, k, 1):.3f}, "
          f"top-8 {recall(2*k, 8, 1):.3f}; from out(l) top-{k} {recall(2*k+8, k, 1):.3f}, top-8 {recall(2*k+8, 8, 1):.3f}; "
          f"l+2 from mid(l) top-{k} {recall(2*k+16, k, 2):.3f}, top-8 {recall(2*k+16, 8, 2):.3f}")
PY
