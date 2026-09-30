#!/bin/bash
# Job 079: where do our 5.44 ms per token go? In job 078 (Qwen3-30B-A3B BF16, RTX 5090 + 9950X with 44.7 GB/s host
# DRAM) our cache cost 5.44 ms per token plus 0.194 ms per miss, and FreeToken led at 43.75% of experts on the GPU
# (94.1 vs 89.2 tok/s). This job profiles both at 43.75% with Nsight Systems, on the decode of one measured request.
# The same host also reruns the unprofiled comparison at 43.75%.
#   Host: RTX 5090 + Ryzen 9 9950X3D (a different machine from job 078; its host bandwidth is measured first).
#   Reference, no profiler: 30 AIME-25 problems, 256 tokens, greedy, held-out warm-up, session, one launch each:
#     llama.cpp -ncmoe 27, ours C56 with and without FETCH, FreeToken offload and hybrid at rate 0.4375.
#   Profiled: the same servers under `nsys launch`. After the warm-up the client starts collection, runs problem 0
#     (256 tokens), stops collection, then runs problems 1-4 unprofiled. CUDA graphs are traced per node. Per-token
#     medians over the last 200 decode tokens are computed on the machine (prof_summary.py): wall, GPU busy,
#     short/long idle gaps, ec_wait, kernels per token, time per kernel category, top kernels.
# Predictions, committed before launch:
#   1. our host-side time per decode step (app + pre + inputs + launch + post, from the cache's own timer, C56+FETCH)
#      is below 1 ms and below 10% of the token time: the fixed cost is on the GPU, not the host;
#   2. ours runs at least 1.5x as many GPU kernels per decode token as FreeToken offload (profile medians);
#   3. on this host, ours (better of FETCH on/off) at 43.75% is within +-5% of FreeToken's better backend
#      (paired ratio 0.95-1.05): job 078's loss depends on the host.
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
. "$J/prof.sh"
export PROF_KEEP_TAIL="oursf_C56 ft_offload_r0.4375" PROF_TOKENS=200
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip python3-dev numactl >> $OUT/apt.txt 2>&1
CORES=$(usable_cores); echo "usable physical cores $CORES (nproc $(nproc))" | tee $OUT/cores.txt
R() { local t=$1; shift; timeout "$t" "$@"; }
python3 -m pip install -q --break-system-packages uv huggingface_hub hf_xet > $OUT/pip_uv.txt 2>&1
nvcc --version | tail -2 | tee $OUT/nvcc_version.txt
HFD=$M/Qwen3-30B-A3B; mkdir -p $M
( t0=$(date +%s); python3 -c "import sys; from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-30B-A3B', local_dir=sys.argv[1])" "$HFD"
  echo "hf download rc=$? in $(( $(date +%s) - t0 )) s"; du -sh "$HFD" ) > $OUT/dl_hf.txt 2>&1 &
DLH=$!
install_nsys
# FreeToken (pinned commit, its own venv: CUDA 13 torch)
FT=$WORK/ft; export FT_DIR=$FT
( cd $WORK && git init -q ft && cd ft && git remote add origin https://github.com/FlashML-org/FreeToken &&
  git fetch -q --depth 1 origin 0d652e73a452d014ac5441a15baa75348e9fcb0a && git checkout -q FETCH_HEAD ) || echo "FreeToken clone FAILED"
( cd $FT && uv venv -q .venv && . .venv/bin/activate && time uv pip install -q -e ".[accel]" huggingface_hub hf_xet gguf sentencepiece transformers safetensors ) > $OUT/ft_install.txt 2>&1
PY=$FT/.venv/bin/python; FTBIN=$FT/.venv/bin/ft
$PY -c "import freetoken, huggingface_hub" > $OUT/ft_import.txt 2>&1 || { echo "FreeToken not importable, stopping"; tail -20 $OUT/ft_install.txt; exit 3; }
# which nsys switches go to `launch` and which to `start` (version dependent)
nsys_selftest $PY && echo "nsys interactive: launch [$NSYS_LOPTS] start [$NSYS_SOPTS]" || echo "nsys interactive collection FAILED"
$PY -c "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('math-ai/aime25','test.jsonl',repo_type='dataset'), '$WORK/aime25.jsonl')" && export FREETOKEN_AIME25_JSONL=$WORK/aime25.jsonl
$PY - "$WORK/warmup.txt" <<'PY' > $OUT/warmup_source.txt 2>&1
import sys
sys.path.insert(0, "/work/ft/benchmarks")
import bench_decode_moe as B
text = None
for repo, fn in (("math-ai/aime24", "test.jsonl"), ("HuggingFaceH4/aime_2024", "train.jsonl")):
    try:
        from huggingface_hub import hf_hub_download
        text, _ = B.load_problem(hf_hub_download(repo, fn, repo_type="dataset"), 0)
        print("warm-up prompt:", repo, fn, "problem 0"); break
    except Exception as e:
        print("no", repo, fn, repr(e)[:200])
if text is None:
    text = "Let S be the set of positive integers n at most 2024 such that n^2 + n + 41 is divisible by 43. Find the number of elements of S.\n" + B.BOXED_INSTRUCTION
open(sys.argv[1], "w").write(text)
PY
build_tree $WORK/lc-ec "$J/llama.cpp-expert-cache-4da6337.patch" llama-server
build_tree $WORK/lc-stock "" llama-server
EC_SERVER=$WORK/lc-ec/build/bin/llama-server
wait $DLH; cat $OUT/dl_hf.txt
[ -f "$HFD/config.json" ] && echo "HF checkpoint ok" || { echo "HF checkpoint MISSING"; exit 3; }
GG=$M/Qwen3-30B-A3B-BF16.gguf
( t0=$(date +%s); PYTHONPATH=$WORK/lc-stock/gguf-py $PY $WORK/lc-stock/convert_hf_to_gguf.py "$HFD" --outtype bf16 --outfile $GG
  echo "convert rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $GG ) > $OUT/convert.txt 2>&1
tail -3 $OUT/convert.txt
[ -s $GG ] || { echo "GGUF conversion failed"; exit 3; }
platform
( cd $FT && R 20m $FTBIN bench bw --dtype bf16 ) > $OUT/ft_bench_bw.txt 2>&1; cp ~/.cache/freetoken/benchbw/*.json $OUT/ 2>/dev/null
H=$(( CORES > 4 ? CORES - 2 : 2 ))
LXB='{"min_p": 0.0, "seed": 1234, "backend_sampling": true}'
LS="-m $GG -ngl 99 -fa on -lm none -t $CORES -np 1 -c 2048 --no-webui --jinja"
OT="-ot '\\.ffn_(up|down|gate|gate_up)_exps=CUDA_Host'"
ECENV="env LLAMA_EC_POLICY=dfa LLAMA_EC_KAPPA=1 LLAMA_EC_MAILBOX=1 LLAMA_EC_TDEC=1 LLAMA_EC_HELPERS=$H"
waitfree() {
  for i in $(seq 120); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-0}" -lt 1500 ] && return; sleep 2; done
  echo "GPU memory still in use: $u MiB"
}
cl() {  # label meta problems [client args...]
  local lab=$1 meta=$2 probs=$3; shift 3
  waitfree
  R 60m $PY $J/bs1_client.py --model $HFD --problems $probs --decode 256 --greedy --warmup once --warmup-prompt $WORK/warmup.txt \
    --launch 1 --label $lab --meta "$meta" --json $OUT/bs1.jsonl --log $OUT/srv_${lab}.log "$@"
  echo "$lab rc=$?"
}
ours_cmd() {  # fetch(0|1) stats-file
  local f=""; [ "$1" = 1 ] && f="LLAMA_EC_FETCH=0,1,1,2,3,3,4,5,6"
  echo "$ECENV LLAMA_EC_SLOTS=56 $f LLAMA_EC_STATS=$2 $EC_SERVER $LS $OT --port {port}"
}
LCMD="$STOCK_SERVER $LS -ncmoe 27 --port {port}"
M27='{"system": "llama.cpp", "pct": 43.75, "ncmoe": 27}'
MO0='{"system": "ours v2", "pct": 43.75, "C": 56, "fetch": 0}'
MO1='{"system": "ours v2", "pct": 43.75, "C": 56, "fetch": 1}'
P30=$(seq -s, 0 29)
# 1) reference, no profiler
cl llama_n27 "$M27" $P30 --extra "$LXB" --cmd "$LCMD"
cl oursv2_C56 "$MO0" $P30 --extra "$LXB" --cmd "$(ours_cmd 0 $OUT/srv_oursv2_C56.json)"
cl oursv2f_C56 "$MO1" $P30 --extra "$LXB" --cmd "$(ours_cmd 1 $OUT/srv_oursv2f_C56.json)"
( cd $FT; for b in offload hybrid; do
    cl ft_${b}_r0.4375 "{\"system\": \"FreeToken $b\", \"pct\": 43.75, \"rate\": 0.4375}" $P30 --ft $b --cache-rate 0.4375 --mem-ratio 0.95
  done )
# 2) profiled: problem 0 under collection, problems 1-4 after it
P5=0,1,2,3,4
MFT='{"system": "FreeToken offload", "pct": 43.75, "rate": 0.4375}'
if [ -n "$NSYS_LOPTS" ]; then
  pw() { echo "$NSYS launch --session-new=s_${1//./_} $NSYS_LOPTS"; }
  pb() { echo "$NSYS start --session=s_${1//./_} $NSYS_SOPTS -o $WORK/prof_$1 --force-overwrite=true"; }
  pa() { echo "$NSYS stop --session=s_${1//./_}"; }
  cl prof_llama_n27 "$M27" $P5 --extra "$LXB" --cmd "$LCMD" \
    --wrap "$(pw llama_n27)" --before-cmd "$(pb llama_n27)" --after-cmd "$(pa llama_n27)" --stop-wait 300
  prof_post llama_n27 255
  cl prof_oursv2f_C56 "$MO1" $P5 --extra "$LXB" --cmd "$(ours_cmd 1 $OUT/srv_prof_oursv2f_C56.json)" \
    --wrap "$(pw oursf_C56)" --before-cmd "$(pb oursf_C56)" --after-cmd "$(pa oursf_C56)" --stop-wait 300
  prof_post oursf_C56 255
  cl prof_oursv2_C56 "$MO0" $P5 --extra "$LXB" --cmd "$(ours_cmd 0 $OUT/srv_prof_oursv2_C56.json)" \
    --wrap "$(pw ours_C56)" --before-cmd "$(pb ours_C56)" --after-cmd "$(pa ours_C56)" --stop-wait 300
  prof_post ours_C56 255
  ( cd $FT; cl prof_ft_offload_r0.4375 "$MFT" $P5 --ft offload --cache-rate 0.4375 --mem-ratio 0.95 \
      --wrap "$(pw ft_offload_r0.4375)" --before-cmd "$(pb ft_offload_r0.4375)" --after-cmd "$(pa ft_offload_r0.4375)" )
  prof_post ft_offload_r0.4375 255
else
  # fallback: profile the whole server run (warm-up + problem 0) for the llama-server systems; SIGINT ends it
  whole() { echo "$NSYS profile -o $WORK/prof_$1 --force-overwrite=true --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none"; }
  cl prof_oursv2f_C56 "$MO1" 0 --extra "$LXB" --cmd "$(ours_cmd 1 $OUT/srv_prof_oursv2f_C56.json)" --wrap "$(whole oursf_C56)" --stop-wait 900
  prof_post oursf_C56 510
  cl prof_llama_n27 "$M27" 0 --extra "$LXB" --cmd "$LCMD" --wrap "$(whole llama_n27)" --stop-wait 900
  prof_post llama_n27 510
fi
cd $WORK
for f in $OUT/srv_*.log; do tail -n 60 "$f" > "$f.tail"; done; find $OUT -name 'srv_*.log' -delete
rm -f $WORK/prof_*.nsys-rep
cat $OUT/prof_summary.txt
$PY - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, sys, os, statistics as st
out = sys.argv[1]
rows = [json.loads(l) for l in open(f"{out}/bs1.jsonl")] if os.path.exists(f"{out}/bs1.jsonl") else []
by = {}
for r in rows: by.setdefault(r["label"], []).append(r)
for lab, v in by.items():
    prof = [r for r in v if r.get("profiled")]
    rest = [r for r in v if not r.get("profiled")]
    s = f"{lab:26s} {st.mean(r['decode_tok_s'] for r in rest):7.2f} tok/s (n={len(rest)})" if rest else f"{lab:26s}"
    if prof:
        s += f"  profiled problem {prof[0]['problem']}: {prof[0]['decode_tok_s']:.2f} tok/s"
    print(s + f"  vram {sorted({round(r.get('vram_used_gib') or 0, 1) for r in v})}")
for f in sorted(os.listdir(out)):
    if f.startswith("srv_") and f.endswith(".json") and "oursv2" in f:
        d = json.load(open(f"{out}/{f}")); n = d["steps"]; h = d.get("host_us_per_step", {})
        print(f"{f}: misses/token {d['misses'] / n:.1f} fetches/token {d.get('fetches', 0) / n:.1f}; host us/step "
              + " ".join(f"{k} {v:.0f}" for k, v in h.items() if k != "n"))
PY
