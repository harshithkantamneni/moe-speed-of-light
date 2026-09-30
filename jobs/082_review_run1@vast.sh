#!/bin/bash
# Job 082: the review's run 1, on the headline machine (RTX 5090 + Ryzen 9 9950X3D, Vast offer 51046112, jobs 080/081).
#   A. Steady state and held-out prompts: 10 MATH-500 problems (never used during development) x 2,048 tokens, greedy,
#      session, per-token window rates (tokens 1-256 and 257-2048). Ours (the law's split, computed on the machine) vs
#      FreeToken's Table 1 backend at every Table 1 budget; FreeToken also with --disable-moe-prefill-overlap at the top
#      budget of each model. Run order alternates between budgets.
#   B. Ablation ladder at 25% (gpt-oss C32 / Qwen3 C32), AIME-25 problems 0-14, 256 tokens (the Table 1 protocol):
#      stock llama.cpp -> static cache -> LRU -> decayed frequency (DFA) -> + GPU-signalled CPU helpers (mailbox) ->
#      + slot maps on the GPU -> + GPU-side sampling -> + FETCH (fixed table) -> + FETCH (law's table) = ours.
#      Steps before the mailbox use the host-driven CPU path; steps before "+ GPU sampling" sample on the CPU.
#   C. Numerical parity (teacher-forced, llama-ec-bench, 12 sequences x 128 decode steps after 640 prefilled tokens):
#      ours (law's split, C32) against stock llama.cpp, with a second stock placement as the baseline disagreement.
#      gpt-oss uses its own-text corpus; Qwen3 a MATH-500 text corpus tokenized on the machine.
#   D. The missing Table 1 cells: stock llama.cpp on Qwen3 at 25% (-ncmoe 36) and 43.75% (-ncmoe 27), 30 AIME-25 problems.
# Predictions, committed before launch:
#   1. steady state (tokens 257-2048, held-out prompts): ours leads FreeToken on gpt-oss at 11, 25 and 40% (paired CI > 1);
#   2. FreeToken gains more from the first 256 tokens to the steady state than ours at every budget
#      (its 257-2048 / 1-256 rate ratio is higher);
#   3. at Qwen3 43.75%, FreeToken's steady-state rate is above ours (ratio ours / FreeToken < 1);
#   4. every ablation step is non-negative within 1%, and ours runs >= 2x stock llama.cpp at 25% on both models;
#   5. parity: ours agrees with stock llama.cpp on >= 98% of teacher-forced next tokens with |dNLL| <= 1% on both models;
#   6. llama.cpp on Qwen3 at 25% and 43.75% runs within 5% of CPU expert bytes / B_c + 4.3 ms (B_c: this machine's
#      probe at the thread count used; the prediction is written before the runs).
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip >> $OUT/apt.txt 2>&1
CORES=$(usable_cores); echo "usable physical cores $CORES (nproc $(nproc))" | tee $OUT/cores.txt
R() { local t=$1; shift; timeout "$t" "$@"; }
FT=$WORK/ft; export FT_DIR=$FT
( cd $WORK && git init -q ft && cd ft && git remote add origin https://github.com/FlashML-org/FreeToken &&
  git fetch -q --depth 1 origin 0d652e73a452d014ac5441a15baa75348e9fcb0a && git checkout -q FETCH_HEAD ) || echo "FreeToken clone FAILED"
python3 -m pip install -q --break-system-packages uv huggingface_hub hf_xet > $OUT/pip_uv.txt 2>&1
nvcc --version | tail -2 | tee $OUT/nvcc_version.txt   # FreeToken builds kernels with nvcc and needs the CUDA 13 toolkit (torch cu130)
( cd $FT && uv venv -q .venv && . .venv/bin/activate && time uv pip install -q -e ".[accel]" huggingface_hub hf_xet gguf sentencepiece transformers safetensors ) > $OUT/ft_install.txt 2>&1
echo "FreeToken install rc=$?"
PY=$FT/.venv/bin/python; FTBIN=$FT/.venv/bin/ft
# stop early (and cheaply) if FreeToken is not usable: the comparison is pointless without it (073 attempt 1 ran on a
# CUDA 12.8 image, the install failed and every client run died)
$PY -c "import freetoken, huggingface_hub" > $OUT/ft_import.txt 2>&1 || { echo "FreeToken not importable, stopping"; cat $OUT/ft_import.txt; tail -20 $OUT/ft_install.txt; exit 3; }
$FTBIN --version > $OUT/ft_version.txt 2>&1
HFD=$M/gpt-oss-120b; mkdir -p $M
( t0=$(date +%s); python3 - "$HFD" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("openai/gpt-oss-120b", local_dir=sys.argv[1], ignore_patterns=["original/*", "metal/*"])
PY
  echo "hf download rc=$? in $(( $(date +%s) - t0 )) s"; du -sh "$HFD" ) > $OUT/dl_hf.txt 2>&1 &
DLH=$!
( t0=$(date +%s)
  python3 -c "import sys; from huggingface_hub import hf_hub_download as d; d('ggml-org/gpt-oss-120b-GGUF', 'gpt-oss-120b-MXFP4.gguf', local_dir=sys.argv[1])" "$M"
  echo "gguf hf download rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $M/gpt-oss-120b-MXFP4.gguf
  if [ ! -s $M/gpt-oss-120b-MXFP4.gguf ]; then
    rm -f $M/gpt-oss-120b-MXFP4.gguf
    for i in 1 2 3; do getmodel ggml-org/gpt-oss-120b-GGUF gpt-oss-120b-MXFP4.gguf; [ -f $M/gpt-oss-120b-MXFP4.gguf ] && break; done
  fi
) > $OUT/dl_gguf.txt 2>&1 &
DLG=$!
HFQ=$M/Qwen3-30B-A3B
( t0=$(date +%s); python3 -c "import sys; from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-30B-A3B', local_dir=sys.argv[1])" "$HFQ"
  echo "qwen3 hf download rc=$? in $(( $(date +%s) - t0 )) s"; du -sh "$HFQ" ) > $OUT/dl_qwen3.txt 2>&1 &
DLQ=$!
# AIME-25 (the 30 measured problems) and a held-out warm-up prompt (AIME-24 problem 0, formatted like theirs)
$PY -c "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('math-ai/aime25','test.jsonl',repo_type='dataset'), '$WORK/aime25.jsonl')" && export FREETOKEN_AIME25_JSONL=$WORK/aime25.jsonl
wc -l $WORK/aime25.jsonl | tee $OUT/aime25_count.txt
# held-out prompts (never used while developing the system): MATH-500, first problems in file order
$PY -c "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('HuggingFaceH4/MATH-500','test.jsonl',repo_type='dataset'), '$WORK/math500.jsonl')" && wc -l $WORK/math500.jsonl | tee $OUT/math500_count.txt
$PY - "$WORK/warmup.txt" <<'PY' > $OUT/warmup_source.txt 2>&1
import json, sys
sys.path.insert(0, "/work/ft/benchmarks")
import bench_decode_moe as B
text = None
for repo, fn in (("math-ai/aime24", "test.jsonl"), ("HuggingFaceH4/aime_2024", "train.jsonl")):
    try:
        from huggingface_hub import hf_hub_download
        text, _ = B.load_problem(hf_hub_download(repo, fn, repo_type="dataset"), 0)
        print("warm-up prompt:", repo, fn, "problem 0")
        break
    except Exception as e:
        print("no", repo, fn, repr(e)[:200])
if text is None:
    text = ("Let S be the set of positive integers n at most 2024 such that n^2 + n + 41 is divisible by 43. "
            "Find the number of elements of S.\n" + B.BOXED_INSTRUCTION)
    print("warm-up prompt: fixed fallback text")
open(sys.argv[1], "w").write(text)
PY
cat $OUT/warmup_source.txt
build_tree $WORK/lc-ec "$J/llama.cpp-expert-cache-4da6337.patch" llama-ec-bench llama-server
build_tree $WORK/lc-stock "" llama-server
EC_SERVER=$WORK/lc-ec/build/bin/llama-server
wait $DLG $DLH $DLQ
cat $OUT/dl_gguf.txt $OUT/dl_hf.txt $OUT/dl_qwen3.txt
F=$M/gpt-oss-120b-MXFP4.gguf
[ -f "$HFD/config.json" ] && echo "HF checkpoint ok" || echo "HF checkpoint MISSING"
[ -s "$F" ] && echo "GGUF ok $(stat -c %s $F) bytes" || echo "GGUF MISSING"
# --- the host
platform
( cd $FT && R 25m $FTBIN bench bw ) > $OUT/ft_bench_bw.txt 2>&1; cp ~/.cache/freetoken/benchbw/*.json $OUT/ 2>/dev/null
H=$(( CORES > 4 ? CORES - 2 : 2 ))
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
grep -q "concurrent t=" $OUT/concur.txt || { echo "concur incomplete, retrying"; R 15m $WORK/concur > $OUT/concur.txt 2>&1; }
LAWG=$(python3 $J/fetch_table.py $OUT/concur.txt $H 13253760 4 $OUT/fetch_table_law_gptoss.json 37)
LAWQ=$(python3 $J/fetch_table.py $OUT/concur.txt $H 9437184 8 $OUT/fetch_table_law_qwen3.json 48)
CURG=0,1,1,2,3; CURQ=0,1,1,2,3,3,4,5,6
echo "law tables ($(date -u +%FT%TZ)): gpt-oss $LAWG (current $CURG); qwen3 $LAWQ (current $CURQ)" | tee $OUT/tables.txt
[ -n "$LAWG" ] && [ -n "$LAWQ" ] || { echo "no law table"; exit 3; }
# prediction 6, written before any model run
python3 - $OUT/concur.txt $CORES > $OUT/law_llama_qwen3.json <<'PY'
import json, re, statistics, sys, time
txt = open(sys.argv[1]).read(); t = int(sys.argv[2])
by = {}
for k, v in re.findall(r"cpu_read_gbs t=(\d+) ([\d.]+)", txt): by.setdefault(int(k), []).append(float(v))
pts = sorted((k, statistics.mean(v)) for k, v in by.items())
bc = pts[-1][1] if t >= pts[-1][0] else next(v0 + (v1 - v0) * (t - t0) / (t1 - t0) for (t0, v0), (t1, v1) in zip(pts, pts[1:]) if t0 <= t <= t1)
S = 9437184
pred = {f"llama_n{n}": round(1e3 / (n * 8 * S / (bc * 1e9) * 1e3 + 4.3), 2) for n in (36, 27)}
print(json.dumps(dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), B_c=round(bc, 2), threads=t, G_ms=4.3, predicted_tok_s=pred)))
PY
cat $OUT/law_llama_qwen3.json
OT="-ot '\\.ffn_(up|down|gate|gate_up)_exps=CUDA_Host'"
GPUS='{"min_p": 0.0, "seed": 1234, "backend_sampling": true}'
CPUS='{"min_p": 0.0, "seed": 1234, "backend_sampling": false}'
waitfree() {
  for i in $(seq 120); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-0}" -lt 1500 ] && return; sleep 2; done
  echo "GPU memory still in use: $u MiB"
}
cl() {  # model-dir label meta problems decode prompts-file [client args...]
  local hf=$1 lab=$2 meta=$3 probs=$4 dec=$5 pf=$6; shift 6
  waitfree
  R 90m $PY $J/bs1_client.py --model $hf --problems $probs --decode $dec --greedy --warmup once --warmup-prompt $WORK/warmup.txt \
    --aime $pf --launch 1 --label $lab --meta "$meta" --json $OUT/bs1.jsonl --log $OUT/srv_${lab}.log "$@"
  echo "$lab rc=$?"
}
P10=$(seq -s, 0 9); P15=$(seq -s, 0 14); P30=$(seq -s, 0 29)
MATH=$WORK/math500.jsonl; AIME=$WORK/aime25.jsonl
# ours with a given cache configuration: label C table(none|x,y,..) policy mailbox(0|1) mapsgpu(0|1) sampling-json problems decode prompts model-dir server-args
oursrun() {
  local lab=$1 C=$2 t=$3 pol=$4 mb=$5 mg=$6 samp=$7 probs=$8 dec=$9 pf=${10} hf=${11} ls=${12}
  local e="env LLAMA_EC_POLICY=$pol LLAMA_EC_KAPPA=1 LLAMA_EC_TDEC=1 LLAMA_EC_SLOTS=$C LLAMA_EC_STATS=$OUT/srv_${lab}.json"
  [ "$mb" = 1 ] && e="$e LLAMA_EC_MAILBOX=1 LLAMA_EC_HELPERS=$H"
  [ "$mg" = 0 ] && e="$e LLAMA_EC_MAPS_CPU=1"
  [ "$t" != none ] && e="$e LLAMA_EC_FETCH=$t"
  cl $hf $lab "{\"system\": \"ours\", \"C\": $C, \"table\": \"$t\", \"policy\": \"$pol\", \"mailbox\": $mb, \"maps_gpu\": $mg}" $probs $dec $pf \
    --extra "$samp" --cmd "$e $EC_SERVER $ls $OT --port {port}"
}
ftrun() {  # label backend rate mem-ratio problems decode prompts model-dir [extra]
  local lab=$1 b=$2 r=$3 mr=$4 probs=$5 dec=$6 pf=$7 hf=$8 x=$9
  ( cd $FT; cl $hf $lab "{\"system\": \"FreeToken $b\", \"rate\": $r, \"extra\": \"$x\"}" $probs $dec $pf --ft $b --cache-rate $r --mem-ratio $mr ${x:+--ft-extra=$x} )
}
stock() {  # label ncmoe problems decode prompts model-dir server-args
  cl $6 $1 "{\"system\": \"llama.cpp\", \"ncmoe\": $2}" $3 $4 $5 --extra "$GPUS" --cmd "$STOCK_SERVER $7 -ncmoe $2 --port {port}"
}
parity() {  # tag model-file corpus C law-table ncmoe-ref ncmoe-alt
  local tag=$1 mf=$2 corpus=$3 C=$4 t=$5 n1=$6 n2=$7
  local COMMON="--corpus $corpus --seqs 0,1,2,3,4,5,6,7,8,9,10,11 -t $CORES --n-prefill 640 --n-decode 128 --no-mmap"
  local cfg="slots=$C:policy=dfa:kappa=1:mailbox=1:tdec=1:helpers=$H:fetch=$t"
  R 40m $EC_BIN -m $mf $COMMON --host-experts --ec "$cfg;$cfg:stats=$OUT/par_${tag}_ours.json" --dump $OUT/par_${tag}_ours.bin > $OUT/par_${tag}_ours.jsonl 2> $OUT/par_${tag}_ours.err; echo "parity $tag ours rc=$?"
  for n in $n1 $n2; do
    R 40m $EC_BIN -m $mf $COMMON --ncmoe $n --dump $OUT/par_${tag}_n$n.bin > $OUT/par_${tag}_n$n.jsonl 2> $OUT/par_${tag}_n$n.err; echo "parity $tag n$n rc=$?"
  done
}
# ================= gpt-oss-120b =================
LSG="-m $F -ngl 99 -fa on -lm none -t $CORES -np 1 -c 8448 --no-webui"
# A. steady state, held-out prompts; order alternates between budgets
oursrun g_long_ours_C14 14 $LAWG dfa 1 1 "$GPUS" $P10 2048 $MATH $HFD "$LSG"; ftrun g_long_ft_hybrid_r0.111 hybrid 0.111 0.9 $P10 2048 $MATH $HFD
ftrun g_long_ft_hybrid_r0.25 hybrid 0.25 0.9 $P10 2048 $MATH $HFD; oursrun g_long_ours_C32 32 $LAWG dfa 1 1 "$GPUS" $P10 2048 $MATH $HFD "$LSG"
oursrun g_long_ours_C51 51 $LAWG dfa 1 1 "$GPUS" $P10 2048 $MATH $HFD "$LSG"; ftrun g_long_ft_offload_r0.40 offload 0.40 0.9 $P10 2048 $MATH $HFD
ftrun g_long_ftnoov_offload_r0.40 offload 0.40 0.9 $P10 2048 $MATH $HFD --disable-moe-prefill-overlap
# B. ablation ladder at 25% (C32), AIME 0-14, 256 tokens
stock g_abl_0_stock_n27 27 $P15 256 $AIME $HFD "$LSG"
oursrun g_abl_1_static 32 none static 0 0 "$CPUS" $P15 256 $AIME $HFD "$LSG"
oursrun g_abl_2_lru 32 none lru 0 0 "$CPUS" $P15 256 $AIME $HFD "$LSG"
oursrun g_abl_3_dfa 32 none dfa 0 0 "$CPUS" $P15 256 $AIME $HFD "$LSG"
oursrun g_abl_4_mailbox 32 none dfa 1 0 "$CPUS" $P15 256 $AIME $HFD "$LSG"
oursrun g_abl_5_mapsgpu 32 none dfa 1 1 "$CPUS" $P15 256 $AIME $HFD "$LSG"
oursrun g_abl_6_gpusample 32 none dfa 1 1 "$GPUS" $P15 256 $AIME $HFD "$LSG"
oursrun g_abl_7_fetchfixed 32 $CURG dfa 1 1 "$GPUS" $P15 256 $AIME $HFD "$LSG"
oursrun g_abl_8_fetchlaw 32 $LAWG dfa 1 1 "$GPUS" $P15 256 $AIME $HFD "$LSG"
# C. parity
parity g $F $J/S_gpt-oss-120b.jsonl 32 $LAWG 27 22
# ================= Qwen3-30B-A3B BF16 (gpt-oss files removed first for disk) =================
rm -rf "$HFD" "$F"; df -h $WORK | tail -1
GQ=$M/Qwen3-30B-A3B-BF16.gguf
( t0=$(date +%s); PYTHONPATH=$WORK/lc-stock/gguf-py $PY $WORK/lc-stock/convert_hf_to_gguf.py "$HFQ" --outtype bf16 --outfile $GQ
  echo "convert rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $GQ ) > $OUT/convert_qwen3.txt 2>&1
tail -2 $OUT/convert_qwen3.txt
[ -s "$GQ" ] || { echo "Qwen3 GGUF missing: Qwen3 part skipped"; }
if [ -s "$GQ" ]; then
LSQ="-m $GQ -ngl 99 -fa on -lm none -t $CORES -np 1 -c 4096 --no-webui --jinja"
# D. the missing Table 1 cells (also the ablation's stock step at 25%)
stock q_stock_n36 36 $P30 256 $AIME $HFQ "$LSQ"
stock q_stock_n27 27 $P30 256 $AIME $HFQ "$LSQ"
# A. steady state, held-out prompts
ftrun q_long_ft_hybrid_r0.125 hybrid 0.125 0.9 $P10 2048 $MATH $HFQ; oursrun q_long_ours_C16 16 $LAWQ dfa 1 1 "$GPUS" $P10 2048 $MATH $HFQ "$LSQ"
oursrun q_long_ours_C32 32 $LAWQ dfa 1 1 "$GPUS" $P10 2048 $MATH $HFQ "$LSQ"; ftrun q_long_ft_hybrid_r0.25 hybrid 0.25 0.9 $P10 2048 $MATH $HFQ
ftrun q_long_ft_offload_r0.4375 offload 0.4375 0.95 $P10 2048 $MATH $HFQ; oursrun q_long_ours_C56 56 $LAWQ dfa 1 1 "$GPUS" $P10 2048 $MATH $HFQ "$LSQ"
ftrun q_long_ftnoov_offload_r0.4375 offload 0.4375 0.95 $P10 2048 $MATH $HFQ --disable-moe-prefill-overlap
# B. ablation ladder at 25% (C32), AIME 0-14 (stock step = q_stock_n36, problems 0-14)
oursrun q_abl_1_static 32 none static 0 0 "$CPUS" $P15 256 $AIME $HFQ "$LSQ"
oursrun q_abl_2_lru 32 none lru 0 0 "$CPUS" $P15 256 $AIME $HFQ "$LSQ"
oursrun q_abl_3_dfa 32 none dfa 0 0 "$CPUS" $P15 256 $AIME $HFQ "$LSQ"
oursrun q_abl_4_mailbox 32 none dfa 1 0 "$CPUS" $P15 256 $AIME $HFQ "$LSQ"
oursrun q_abl_5_mapsgpu 32 none dfa 1 1 "$CPUS" $P15 256 $AIME $HFQ "$LSQ"
oursrun q_abl_6_gpusample 32 none dfa 1 1 "$GPUS" $P15 256 $AIME $HFQ "$LSQ"
oursrun q_abl_7_fetchfixed 32 $CURQ dfa 1 1 "$GPUS" $P15 256 $AIME $HFQ "$LSQ"
oursrun q_abl_8_fetchlaw 32 $LAWQ dfa 1 1 "$GPUS" $P15 256 $AIME $HFQ "$LSQ"
# C. parity on a MATH-500 text corpus (12 sequences of >= 800 tokens, Qwen3 chat template, tokenized here)
$PY - "$HFQ" "$MATH" "$WORK/S_qwen3_math.jsonl" <<'PY' > $OUT/qwen3_corpus.txt 2>&1
import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
probs = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
out, i = [], 100          # problems 100.. (disjoint from the speed runs' 0-9)
while len(out) < 12 and i < len(probs):
    text, ids = "", []
    while len(ids) < 800 and i < len(probs):
        p = probs[i]; i += 1
        text += f"Problem: {p['problem']}\nSolution: {p.get('solution', '')}\n\n"
        ids = tok(text)["input_ids"]
    out.append({"domain": "math", "ids": ids})
with open(sys.argv[3], "w") as f:
    for r in out: f.write(json.dumps(r) + "\n")
print("sequences", len(out), "tokens", [len(r["ids"]) for r in out])
PY
cat $OUT/qwen3_corpus.txt
parity q $GQ $WORK/S_qwen3_math.jsonl 32 $LAWQ 36 27
fi
cd $WORK
for f in $OUT/srv_*.log; do tail -n 60 "$f" > "$f.tail"; done; find $OUT -name 'srv_*.log' -delete
$PY - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, sys, os, statistics as st
import numpy as np
out = sys.argv[1]
rows = [json.loads(l) for l in open(f"{out}/bs1.jsonl")] if os.path.exists(f"{out}/bs1.jsonl") else []
by = {}
for r in rows: by.setdefault(r["label"], []).append(r)
for lab, v in by.items():
    w = [r.get("windows") or {} for r in v]
    ws = {k: st.mean(x[k] for x in w if x.get(k)) for k in ("1-256", "257-end") if any(x.get(k) for x in w)}
    print(f"{lab:32s} {st.mean(r['decode_tok_s'] for r in v):7.2f} tok/s (n={len(v)})" + "".join(f"  [{k}] {x:6.2f}" for k, x in ws.items())
          + f"  vram {sorted({round(r.get('vram_used_gib') or 0, 1) for r in v})}")
def nll(p, measured_only=False):
    v = [json.loads(l) for l in open(p) if l.strip().startswith("{")] if os.path.exists(p) else []
    v = [r for r in v if "nll_sum" in r and (not measured_only or "stats=" in r.get("config", ""))]
    return sum(r["nll_sum"] for r in v) / max(1, sum(r["nll_n"] for r in v)) if v else None
for tag, refs in (("g", ("27", "22")), ("q", ("36", "27"))):
    base = f"{out}/par_{tag}_n{refs[0]}.bin"
    if not os.path.exists(base): continue
    t0 = np.fromfile(base, dtype=np.int32).reshape(-1, 10)[:, 0]
    for name, p in (("ours", f"{out}/par_{tag}_ours.bin.1"), (f"n{refs[1]}", f"{out}/par_{tag}_n{refs[1]}.bin")):
        if os.path.exists(p):
            t1 = np.fromfile(p, dtype=np.int32).reshape(-1, 10)[:, 0]
            n0 = nll(f"{out}/par_{tag}_n{refs[0]}.jsonl")
            n1 = nll(f"{out}/par_{tag}_ours.jsonl", True) if name == "ours" else nll(f"{out}/par_{tag}_n{refs[1]}.jsonl")
            d = f", NLL {n1:.4f} vs {n0:.4f} ({100 * (n1 - n0) / n0:+.2f}%)" if n0 and n1 else ""
            print(f"parity {tag}: {name} vs n{refs[0]} top-1 agreement {float((t1 == t0).mean()):.4f} over {len(t0)} steps" if len(t1) == len(t0)
                  else f"parity {tag}: {name}: length mismatch {len(t1)} vs {len(t0)}" + d)
            if len(t1) == len(t0): print("   " + d)
PY
