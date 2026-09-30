#!/bin/bash
# Job 083: gpt-oss-120b long outputs on hard held-out problems, on the headline machine (RTX 5090 + Ryzen 9 9950X3D,
# Vast offer 51046112). Job 082's long runs used easy MATH-500 problems: the model answered well inside 2,048 tokens and
# ignore_eos then forced post-answer text that differs by engine, so its gpt-oss steady-state numbers are void. Here:
# 10 AIME 2022-2024 problems (AI-MO/aimo-validation-aime, problems 11-15 of each exam, first 10 in file order; never used
# during development) x 2,048 tokens, greedy, held-out warm-up, session. Ours (the law's split, computed on the machine)
# vs FreeToken's Table 1 backend (hybrid, hybrid, offload) at 11 / 25 / 40%; order alternates between budgets. The client
# records reasoning vs answer events and the output tail, so post-answer text can be detected and excluded.
# Predictions, committed before launch:
#   1. both systems are still reasoning at token 2,048 on >= 8 of 10 problems (no answer events for ours);
#   2. ours' rate over all 2,048 tokens is within 15% of its Table 1 rate (69.9 / 109.2 / 152.5 tok/s): no post-answer
#      speed-up on hard problems;
#   3. ours leads FreeToken at all three budgets on the all-token rate (paired CI > 1), and on tokens 257-2,048 wherever
#      both stream about one event per token;
#   4. the lead at 11% and 25% is >= 15%.
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
( cd $FT && uv venv -q .venv && . .venv/bin/activate && time uv pip install -q -e ".[accel]" huggingface_hub hf_xet pyarrow ) > $OUT/ft_install.txt 2>&1
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
# AIME-25 (the 30 measured problems) and a held-out warm-up prompt (AIME-24 problem 0, formatted like theirs)
$PY -c "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('math-ai/aime25','test.jsonl',repo_type='dataset'), '$WORK/aime25.jsonl')" && export FREETOKEN_AIME25_JSONL=$WORK/aime25.jsonl
wc -l $WORK/aime25.jsonl | tee $OUT/aime25_count.txt
# held-out hard problems: AIME 2022-2024 (AI-MO/aimo-validation-aime), problems 11-15 of each exam, first 10 in file order
$PY - "$WORK/aime_hard.jsonl" <<'PY' > $OUT/aime_hard_source.txt 2>&1
import json, re, sys
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
t = pq.read_table(hf_hub_download("AI-MO/aimo-validation-aime", "data/train-00000-of-00001.parquet", repo_type="dataset")).to_pylist()
hard = [r for r in t if re.search(r"Problem_1[1-5]\b", r.get("url", ""))][:10]
with open(sys.argv[1], "w") as f:
    for r in hard:
        f.write(json.dumps({"problem": r["problem"], "answer": r["answer"], "url": r["url"]}) + "\n")
print(len(t), "problems;", len(hard), "selected:", [r["url"].split("/")[-1] for r in hard])
PY
cat $OUT/aime_hard_source.txt
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
build_tree $WORK/lc-ec "$J/llama.cpp-expert-cache-4da6337.patch" llama-server
EC_SERVER=$WORK/lc-ec/build/bin/llama-server
wait $DLG $DLH
cat $OUT/dl_gguf.txt $OUT/dl_hf.txt
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
echo "law table ($(date -u +%FT%TZ)): gpt-oss $LAWG" | tee $OUT/tables.txt
[ -n "$LAWG" ] || { echo "no law table"; exit 3; }
OT="-ot '\\.ffn_(up|down|gate|gate_up)_exps=CUDA_Host'"
GPUS='{"min_p": 0.0, "seed": 1234, "backend_sampling": true}'
waitfree() {
  for i in $(seq 120); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-0}" -lt 1500 ] && return; sleep 2; done
  echo "GPU memory still in use: $u MiB"
}
cl() {  # label meta [client args...]
  local lab=$1 meta=$2; shift 2
  waitfree
  R 90m $PY $J/bs1_client.py --model $HFD --problems $(seq -s, 0 9) --decode 2048 --greedy --warmup once --warmup-prompt $WORK/warmup.txt \
    --aime $WORK/aime_hard.jsonl --launch 1 --label $lab --meta "$meta" --json $OUT/bs1.jsonl --log $OUT/srv_${lab}.log "$@"
  echo "$lab rc=$?"
}
LSG="-m $F -ngl 99 -fa on -lm none -t $CORES -np 1 -c 8448 --no-webui"
ours() {  # C
  cl g_hard_ours_C$1 "{\"system\": \"ours\", \"C\": $1, \"table\": \"$LAWG\"}" --extra "$GPUS" \
    --cmd "env LLAMA_EC_POLICY=dfa LLAMA_EC_KAPPA=1 LLAMA_EC_TDEC=1 LLAMA_EC_MAILBOX=1 LLAMA_EC_HELPERS=$H LLAMA_EC_SLOTS=$1 LLAMA_EC_FETCH=$LAWG LLAMA_EC_STATS=$OUT/srv_g_hard_ours_C$1.json $EC_SERVER $LSG $OT --port {port}"
}
ft() {  # backend rate
  ( cd $FT; cl g_hard_ft_$1_r$2 "{\"system\": \"FreeToken $1\", \"rate\": $2}" --ft $1 --cache-rate $2 )
}
ours 14; ft hybrid 0.111
ft hybrid 0.25; ours 32
ours 51; ft offload 0.40
cd $WORK
for f in $OUT/srv_*.log; do tail -n 60 "$f" > "$f.tail"; done; find $OUT -name 'srv_*.log' -delete
$PY - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, sys, os, statistics as st
out = sys.argv[1]
rows = [json.loads(l) for l in open(f"{out}/bs1.jsonl")] if os.path.exists(f"{out}/bs1.jsonl") else []
by = {}
for r in rows: by.setdefault(r["label"], []).append(r)
for lab, v in by.items():
    w = [r.get("windows") or {} for r in v]
    ws = {k: st.mean(x[k] for x in w if x.get(k)) for k in ("1-256", "257-end") if any(x.get(k) for x in w)}
    ans = sum(1 for r in v if r.get("n_content_events"))
    print(f"{lab:28s} {st.mean(r['decode_tok_s'] for r in v):7.2f} tok/s (n={len(v)})" + "".join(f"  [{k}] {x:6.2f}" for k, x in ws.items())
          + f"  events/token {st.mean(r['events'] / r['completion_tokens'] for r in v):.2f}  answered {ans}/{len(v)}")
PY
