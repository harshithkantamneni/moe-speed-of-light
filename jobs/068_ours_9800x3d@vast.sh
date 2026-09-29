#!/bin/bash
# Job 067's our-cache half, which did not run (the llama.cpp fetch for the patched tree hit a network reset; the fetch
# now retries): RTX 5090 + Ryzen 7 9800X3D, the same machine. The host is measured again, then own text (cache, cache +
# FETCH) and the chat benchmark of jobs 064 / 067 (FreeToken's method, AIME-25 problems 0-4 x 256 tokens, llama-server)
# for the cache and the cache + FETCH at C = 14 / 32 / 51 / 56. llama.cpp and FreeToken were measured on this machine
# in job 067.
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
CFG="slots=14:policy=dfa:kappa=1:$MB"
for C in 14 32 56; do
  CFG="$CFG;slots=$C:policy=dfa:kappa=1:$MB:stats=$OUT/ot_C${C}_base.json;slots=$C:policy=dfa:kappa=1:$MB:fetch=0,1,1,2,3:stats=$OUT/ot_C${C}_f01123.json"
done
R 60m $EC_BIN -m $F $COMMON --ec "$CFG" > $OUT/ot_ec.jsonl 2> $OUT/ot_ec.err; echo "own-text ec rc=$?"
EC_SERVER=$WORK/lc-ec/build/bin/llama-server
CL="python3 $J/bs1_client.py --model $M/no-hf-checkpoint --problems 0,1,2,3,4 --decode 256 --json $OUT/bs1.jsonl"
LX='{"min_p": 0.0, "seed": 1234}'
LS="-m $F -ngl 99 -fa on -lm none -t $CORES -np 1 -c 8448 --no-webui"
OT="-ot '\\.ffn_(up|down|gate|gate_up)_exps=CUDA_Host'"
for C in 14 32 51 56; do
  ECENV="env LLAMA_EC_SLOTS=$C LLAMA_EC_POLICY=dfa LLAMA_EC_KAPPA=1 LLAMA_EC_MAILBOX=1 LLAMA_EC_TDEC=1 LLAMA_EC_HELPERS=$H"
  R 40m $CL --label ours_C$C --meta "{\"system\": \"ours\", \"C\": $C}" --extra "$LX" --log $OUT/srv_ours_C$C.log \
    --cmd "$ECENV LLAMA_EC_STATS=$OUT/srv_ours_C$C.json $EC_SERVER $LS $OT --port {port}"
  R 40m $CL --label oursf_C$C --meta "{\"system\": \"ours+fetch\", \"C\": $C}" --extra "$LX" --log $OUT/srv_oursf_C$C.log \
    --cmd "$ECENV LLAMA_EC_FETCH=0,1,1,2,3 LLAMA_EC_STATS=$OUT/srv_oursf_C$C.json $EC_SERVER $LS $OT --port {port}"
done
for f in $OUT/srv_*.log; do tail -n 40 "$f" > "$f.tail"; done; find $OUT -name 'srv_*.log' -size +2M -delete
python3 - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, sys, os, statistics as st
out = sys.argv[1]
if os.path.exists(f"{out}/bs1.jsonl"):
    by = {}
    for l in open(f"{out}/bs1.jsonl"):
        r = json.loads(l); by.setdefault(r["label"], []).append(r["decode_tok_s"])
    print("bs=1 via the chat API (FreeToken's method, AIME-25 problems 0-4, 256 tokens)")
    for lab, v in by.items(): print(f"{lab:12s} {st.mean(v):7.2f} tok/s (min {min(v):.2f} max {max(v):.2f}, n={len(v)})")
by = {}
for l in open(f"{out}/ot_ec.jsonl"):
    if l.strip():
        r = json.loads(l); by.setdefault(r.get("config", ""), []).append(r)
print("own text (12 prompts x 128 tokens, ec-bench)")
for k, v in by.items():
    name = k.split("stats=")[-1].split("/")[-1] if "stats=" in k else "warm-up"
    print(f"  {name}: {sum(r['n_decode'] for r in v) / (sum(r['decode_ms'] for r in v) / 1000):.1f} tok/s")
for f in sorted(os.listdir(out)):
    if f.startswith("srv_") and f.endswith(".json"):
        s = json.load(open(f"{out}/{f}")); print(f"{f}: hit {s.get('hit_rate')} fetches {s.get('fetches')}")
PY
