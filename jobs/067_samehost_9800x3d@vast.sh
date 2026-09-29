#!/bin/bash
# Job 064's same-host, same-client comparison on a second machine: RTX 5090 + Ryzen 7 9800X3D (8 cores, one CCD), a
# common gaming configuration, to see whether the result carries over. The cache's FETCH runs in llama-server now
# (fixed after 064); FreeToken's 0.444 retry is dropped (its KV cache did not fit at 0.9 or 0.95).
# Same host, same client: llama.cpp, the expert cache and FreeToken on one home-PC-class machine (RTX 5090 + Ryzen 9
# 9950X-class, >= 123 GB DDR5, CUDA 13 image), gpt-oss-120b, at equal GPU memory for experts:
#   llama.cpp -ncmoe 32 / 27 / 20  =  cache 14 / 32 / 56 of 128 experts per layer  =  FreeToken --moe-cache-rate
#   0.111 / 0.25 / 0.444 (0.40 if 0.444 does not fit).
# Every system runs as a server with the OpenAI chat API and is timed by one client, jobs/ec2/bs1_client.py, which
# reuses FreeToken's own benchmark code (their AIME-25 prompts, sampling, warm-up protocol and tok/s formula):
# 5 AIME-25 problems x 256 decode tokens per configuration. FreeToken's unmodified benchmark script also runs once
# (rate 0.25, problem 0) as a check that the client reproduces it. The host is measured first (STREAM, bw.cu,
# concur.cu, `ft bench bw`), and the own-text ec-bench runs of jobs 059/060 are repeated so this host can be
# compared with those.
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip >> $OUT/apt.txt 2>&1
CORES=$(lscpu -p=CORE | grep -v '^#' | sort -u | wc -l); echo "physical cores $CORES" | tee $OUT/cores.txt
R() { local t=$1; shift; timeout "$t" "$@"; }
# --- FreeToken (pinned commit) and the two model downloads, in the background while the llama.cpp trees build
FT=$WORK/ft; export FT_DIR=$FT
( cd $WORK && git init -q ft && cd ft && git remote add origin https://github.com/FlashML-org/FreeToken &&
  git fetch -q --depth 1 origin 0d652e73a452d014ac5441a15baa75348e9fcb0a && git checkout -q FETCH_HEAD ) || echo "FreeToken clone FAILED"
python3 -m pip install -q --break-system-packages uv > $OUT/pip_uv.txt 2>&1
( cd $FT && uv venv -q .venv && . .venv/bin/activate && time uv pip install -q -e ".[accel]" huggingface_hub hf_xet ) > $OUT/ft_install.txt 2>&1
echo "FreeToken install rc=$?"
PY=$FT/.venv/bin/python; FTBIN=$FT/.venv/bin/ft
if [ ! -x $PY ]; then echo "FreeToken venv missing: client on system python"; python3 -m pip install -q --break-system-packages huggingface_hub hf_xet >> $OUT/pip_uv.txt 2>&1; PY=python3; fi
$FTBIN --version > $OUT/ft_version.txt 2>&1
HFD=$M/gpt-oss-120b; mkdir -p $M
( t0=$(date +%s); $PY - "$HFD" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("openai/gpt-oss-120b", local_dir=sys.argv[1], ignore_patterns=["original/*", "metal/*"])
PY
  echo "hf download rc=$? in $(( $(date +%s) - t0 )) s"; du -sh "$HFD"; ls "$HFD" ) > $OUT/dl_hf.txt 2>&1 &
DLH=$!
( t0=$(date +%s)
  $PY -c "import sys; from huggingface_hub import hf_hub_download as d; d('ggml-org/gpt-oss-120b-GGUF', 'gpt-oss-120b-MXFP4.gguf', local_dir=sys.argv[1])" "$M"
  echo "gguf hf download rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $M/gpt-oss-120b-MXFP4.gguf
  if [ ! -s $M/gpt-oss-120b-MXFP4.gguf ]; then
    rm -f $M/gpt-oss-120b-MXFP4.gguf
    for i in 1 2 3; do getmodel ggml-org/gpt-oss-120b-GGUF gpt-oss-120b-MXFP4.gguf; [ -f $M/gpt-oss-120b-MXFP4.gguf ] && break; done
  fi
) > $OUT/dl_gguf.txt 2>&1 &
DLG=$!
# AIME-25 problems for every client run (one fetch, then local)
$PY -c "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('math-ai/aime25','test.jsonl',repo_type='dataset'), '$WORK/aime25.jsonl')" && export FREETOKEN_AIME25_JSONL=$WORK/aime25.jsonl
build_tree $WORK/lc-ec "$J/llama.cpp-expert-cache-4da6337.patch" llama-ec-bench llama-server
build_tree $WORK/lc-stock "" llama-server
EC_SERVER=$WORK/lc-ec/build/bin/llama-server
wait $DLG $DLH
cat $OUT/dl_gguf.txt $OUT/dl_hf.txt
F=$M/gpt-oss-120b-MXFP4.gguf
[ -f "$HFD/config.json" ] && echo "HF checkpoint ok" || echo "HF checkpoint MISSING"
[ -s "$F" ] && echo "GGUF ok $(stat -c %s $F) bytes" || echo "GGUF MISSING"
# --- the host, measured on a quiet machine
platform
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
( cd $FT && R 20m $FTBIN bench bw ) > $OUT/ft_bench_bw.txt 2>&1; cp ~/.cache/freetoken/benchbw/*.json $OUT/ 2>/dev/null
# --- own text (as jobs 059/060): the cache without and with FETCH, and stock placement, inside ec-bench
H=$(( CORES > 4 ? CORES - 2 : 2 )); MB="mailbox=1:tdec=1:helpers=$H"
COMMON="--corpus $J/S_gpt-oss-120b.jsonl --seqs 0,1,2,3,4,5,6,7,8,9,10,11 -t $CORES --n-prefill 640 --n-decode 128 --no-mmap"
CFG="slots=14:policy=dfa:kappa=1:$MB"
for C in 14 32 56; do
  CFG="$CFG;slots=$C:policy=dfa:kappa=1:$MB:stats=$OUT/ot_C${C}_base.json;slots=$C:policy=dfa:kappa=1:$MB:fetch=0,1,1,2,3:stats=$OUT/ot_C${C}_f01123.json"
done
R 60m $EC_BIN -m $F $COMMON --host-experts --ec "$CFG" > $OUT/ot_ec.jsonl 2> $OUT/ot_ec.err; echo "own-text ec rc=$?"
for n in 32 27 20; do
  R 40m $EC_BIN -m $F $COMMON --ncmoe $n > $OUT/ot_static_n$n.jsonl 2> $OUT/ot_static_n$n.err; echo "own-text static n=$n rc=$?"
done
# --- one client, every system as a server
P=0,1,2,3,4; CL="$PY $J/bs1_client.py --model $HFD --problems $P --decode 256 --json $OUT/bs1.jsonl"
LX='{"min_p": 0.0, "seed": 1234}'
LS="-m $F -ngl 99 -fa on -lm none -t $CORES -np 1 -c 8448 --no-webui"
lcrun() {  # label meta server-cmd
  R 40m $CL --label "$1" --meta "$2" --extra "$LX" --log $OUT/srv_$1.log --cmd "$3"; local rc=$?
  if [ $rc -ne 0 ] && grep -q "does not match the expected" $OUT/srv_$1.log 2>/dev/null; then
    echo "$1: chat parser rejected the output; rerun with --no-jinja"
    R 40m $CL --label "$1_nojinja" --meta "$2" --extra "$LX" --log $OUT/srv_$1_nojinja.log --cmd "$3 --no-jinja"
  fi
}
for pair in 32:14 27:32 20:56 0:51; do
  n=${pair%:*}; C=${pair#*:}
  [ $n -gt 0 ] && lcrun llama_n$n "{\"system\": \"llama.cpp\", \"ncmoe\": $n, \"C\": $C}" "$STOCK_SERVER $LS -ncmoe $n --port {port}"
  ECENV="env LLAMA_EC_SLOTS=$C LLAMA_EC_POLICY=dfa LLAMA_EC_KAPPA=1 LLAMA_EC_MAILBOX=1 LLAMA_EC_TDEC=1 LLAMA_EC_HELPERS=$H"
  OT="-ot '\\.ffn_(up|down|gate|gate_up)_exps=CUDA_Host'"
  lcrun ours_C$C "{\"system\": \"ours\", \"ncmoe\": $n, \"C\": $C}" "$ECENV LLAMA_EC_STATS=$OUT/srv_ours_C$C.json $EC_SERVER $LS $OT --port {port}"
  lcrun oursf_C$C "{\"system\": \"ours+fetch\", \"ncmoe\": $n, \"C\": $C}" "$ECENV LLAMA_EC_FETCH=0,1,1,2,3 LLAMA_EC_STATS=$OUT/srv_oursf_C$C.json $EC_SERVER $LS $OT --port {port}"
done
cd $FT
for pair in 0.111:14 0.25:32 0.444:56; do
  r=${pair%:*}; C=${pair#*:}
  for b in offload hybrid; do
    R 60m $CL --ft $b --cache-rate $r --label ft_${b}_r$r --meta "{\"system\": \"FreeToken $b\", \"rate\": $r, \"C\": $C}" --log $OUT/srv_ft_${b}_r$r.log
    rc=$?
    if [ $rc -ne 0 ] && [ $r = 0.444 ]; then
      R 60m $CL --ft $b --cache-rate 0.40 --label ft_${b}_r0.40 --meta "{\"system\": \"FreeToken $b\", \"rate\": 0.40, \"C\": 51.2}" --log $OUT/srv_ft_${b}_r0.40.log
    fi
  done
done
# their benchmark, unmodified, as the check on the client
PYTHONPATH=python:. R 60m $PY benchmarks/bench_decode_moe.py --model $HFD --backend offload,hybrid \
  --cache-rate 0.25 --decode 256 --json $OUT/ft_own_bench.jsonl > $OUT/ft_own_bench.txt 2>&1; echo "their bench rc=$?"
cd $WORK
for f in $OUT/srv_*.log; do tail -n 60 "$f" > "$f.tail"; done; find $OUT -name 'srv_*.log' -size +2M -delete
$PY - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, sys, os, statistics as st
out = sys.argv[1]
rows = [json.loads(l) for l in open(f"{out}/bs1.jsonl")] if os.path.exists(f"{out}/bs1.jsonl") else []
by = {}
for r in rows: by.setdefault(r["label"], []).append(r)
print("bs=1 decode via the chat API, AIME-25 problems 0-4, 256 tokens (FreeToken's method); mean over problems")
for lab, v in by.items():
    t = [r["decode_tok_s"] for r in v]
    srv = [r["server_timings"]["predicted_per_second"] for r in v if r.get("server_timings")]
    print(f"{lab:24s} {st.mean(t):7.2f} tok/s  (min {min(t):.2f} max {max(t):.2f}, n={len(t)})"
          + (f"  server-side {st.mean(srv):.2f}" if srv else "") + f"  vram {v[0].get('vram_used_gib') or 0:.1f} GiB")
for f in sorted(os.listdir(out)):
    if f.startswith("srv_ours") and f.endswith(".json"):
        s = json.load(open(f"{out}/{f}")); print(f"{f}: hit {s.get('hit_rate')} fetches {s.get('fetches')} steps {s.get('steps')}")
def rate(p):
    v = [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []
    by = {}
    for r in v: by.setdefault(r.get("config", ""), []).append(r)
    return {k: sum(r["n_decode"] for r in x) / (sum(r["decode_ms"] for r in x) / 1000) for k, x in by.items()}
print("own text (12 prompts x 128 tokens, ec-bench)")
for k, x in rate(f"{out}/ot_ec.jsonl").items():
    print(f"  ec {k.split('stats=')[-1].split('/')[-1] if 'stats=' in k else 'warm-up'}: {x:.1f} tok/s")
for n in (32, 27, 20):
    for k, x in rate(f"{out}/ot_static_n{n}.jsonl").items(): print(f"  static ncmoe {n}: {x:.1f} tok/s")
PY
cat $OUT/ft_own_bench.jsonl >> $OUT/summary.txt 2>/dev/null
