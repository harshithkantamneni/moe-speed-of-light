#!/bin/bash
# Job 085: the contrasting machine. The same GPU and CPU as the headline machine (RTX 5090 + Ryzen 9 9950X3D) but about
# half its PCIe rate (Vast offer 53405183 lists 27.5 GB/s against 54-55 GB/s): the one host parameter that sets how a
# miss should be served changes, everything else stays. Table 1 protocol (30 AIME-25 problems, 256 tokens, greedy,
# held-out warm-up, session, GPU-side sampling for llama-server), both models at the Table 1 budgets:
#   gpt-oss-120b C 14 / 32 / 51 (FreeToken rates 0.111 / 0.25 / 0.40), Qwen3-30B-A3B BF16 C 16 / 32 / 56
#   (0.125 / 0.25 / 0.4375). Launch 1: ours with the law's table computed on this machine before any model run, ours
#   with the fixed table, FreeToken offload and hybrid; at 25% also ours with no fetching and llama.cpp. Launch 2: each
#   system's faster variant per budget (confirmation; comparisons use launch 2).
# Predictions, committed before launch:
#   1. the law's tables fetch less than on the headline machine (every entry <= gpt-oss 0,0,1,1,2 / Qwen3
#      0,0,1,1,2,3,3,4,5, at least one entry lower);
#   2. the law's table beats the fixed table at every budget on both models (launch 1, paired CI > 1), by more than on
#      the headline machine (gpt-oss +7.9 / +5.4 / +3.3%, Qwen3 +8.0 / +6.7 / +4.8%) at >= 4 of the 6 budgets;
#   3. ours leads FreeToken's better backend at every budget on both models (launch 2, paired CI > 1);
#   4. ours' lead over FreeToken is larger than on the headline machine at >= 4 of the 6 budgets;
#   5. llama.cpp at 25% is within 5% of the law's prediction written on the machine before the runs, on both models.
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
nvcc --version | tail -2 | tee $OUT/nvcc_version.txt
( cd $FT && uv venv -q .venv && . .venv/bin/activate && time uv pip install -q -e ".[accel]" huggingface_hub hf_xet gguf sentencepiece transformers safetensors ) > $OUT/ft_install.txt 2>&1
echo "FreeToken install rc=$?"
PY=$FT/.venv/bin/python; FTBIN=$FT/.venv/bin/ft
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
$PY -c "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('math-ai/aime25','test.jsonl',repo_type='dataset'), '$WORK/aime25.jsonl')" && export FREETOKEN_AIME25_JSONL=$WORK/aime25.jsonl
wc -l $WORK/aime25.jsonl | tee $OUT/aime25_count.txt
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
CURG=0,1,1,2,3; CURQ=0,1,1,2,3,3,4,5,6; NONEG=0,0,0,0,0; NONEQ=0,0,0,0,0,0,0,0,0
echo "law tables ($(date -u +%FT%TZ)): gpt-oss $LAWG (fixed $CURG); qwen3 $LAWQ (fixed $CURQ)" | tee $OUT/tables.txt
[ -n "$LAWG" ] && [ -n "$LAWQ" ] || { echo "no law table"; exit 3; }
# prediction 5, written before any model run: llama.cpp -ncmoe n reads all k routed experts of n layers on the CPU;
# T = G + n * k * S / B_c (threads = CORES), G frozen per model (gpt-oss 5.063 ms, jobs 064/066; Qwen3 4.3 ms, job 082)
python3 - $OUT/concur.txt $CORES > $OUT/law_llama.json <<'PY'
import json, re, statistics, sys, time
txt = open(sys.argv[1]).read(); t = int(sys.argv[2])
by = {}
for k, v in re.findall(r"cpu_read_gbs t=(\d+) ([\d.]+)", txt): by.setdefault(int(k), []).append(float(v))
pts = sorted((k, statistics.mean(v)) for k, v in by.items())
bc = pts[-1][1] if t >= pts[-1][0] else pts[0][1] if t <= pts[0][0] else next(v0 + (v1 - v0) * (t - t0) / (t1 - t0) for (t0, v0), (t1, v1) in zip(pts, pts[1:]) if t0 <= t <= t1)
pred = {"gptoss_llama_n27": round(1e3 / (27 * 4 * 13253760 / (bc * 1e9) * 1e3 + 5.063), 2),
        "qwen3_llama_n36": round(1e3 / (36 * 8 * 9437184 / (bc * 1e9) * 1e3 + 4.3), 2)}
print(json.dumps(dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), B_c=round(bc, 2), threads=t, predicted_tok_s=pred)))
PY
cat $OUT/law_llama.json
LXB='{"min_p": 0.0, "seed": 1234, "backend_sampling": true}'
OT="-ot '\\.ffn_(up|down|gate|gate_up)_exps=CUDA_Host'"
ECENV="env LLAMA_EC_POLICY=dfa LLAMA_EC_KAPPA=1 LLAMA_EC_MAILBOX=1 LLAMA_EC_TDEC=1 LLAMA_EC_HELPERS=$H"
waitfree() {
  for i in $(seq 120); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-0}" -lt 1500 ] && return; sleep 2; done
  echo "GPU memory still in use: $u MiB"
}
cl() {  # model-dir label launch meta [client args...]
  local hf=$1 lab=$2 L=$3 meta=$4; shift 4
  waitfree
  R 60m $PY $J/bs1_client.py --model $hf --problems $(seq -s, 0 29) --decode 256 --greedy --warmup once --warmup-prompt $WORK/warmup.txt \
    --launch $L --label $lab --meta "$meta" --json $OUT/bs1.jsonl --log $OUT/srv_${lab}_L$L.log "$@"
  echo "$lab launch $L rc=$?"
}
pick() {  # launch-1 winner among labels
  $PY - $OUT/bs1.jsonl "$@" <<'PY'
import json, sys, statistics as st
by = {}
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r["launch"] == 1: by.setdefault(r["label"], []).append(r["decode_tok_s"])
print(max(sys.argv[2:], key=lambda k: st.mean(by.get(k, [0]))))
PY
}
# ---- gpt-oss-120b
LSG="-m $F -ngl 99 -fa on -lm none -t $CORES -np 1 -c 8448 --no-webui"
oursg() {  # C table-name launch
  local t=$CURG; [ $2 = law ] && t=$LAWG; [ $2 = none ] && t=$NONEG
  cl $HFD g_ours_C$1_$2 $3 "{\"model\": \"gpt-oss-120b\", \"system\": \"ours v2\", \"C\": $1, \"table\": \"$t\"}" --extra "$LXB" \
    --cmd "$ECENV LLAMA_EC_SLOTS=$1 LLAMA_EC_FETCH=$t LLAMA_EC_STATS=$OUT/srv_g_ours_C$1_$2_L$3.json $EC_SERVER $LSG $OT --port {port}"
}
ftg() { ( cd $FT; cl $HFD g_ft_$1_r$2 $3 "{\"model\": \"gpt-oss-120b\", \"system\": \"FreeToken $1\", \"rate\": $2}" --ft $1 --cache-rate $2 ); }
for C in 14 32 51; do oursg $C law 1; oursg $C cur 1; done
oursg 32 none 1
for r in 0.111 0.25 0.40; do ftg offload $r 1; ftg hybrid $r 1; done
cl $HFD g_llama_n27 1 '{"model": "gpt-oss-120b", "system": "llama.cpp", "ncmoe": 27}' --extra "$LXB" --cmd "$STOCK_SERVER $LSG -ncmoe 27 --port {port}"
for pair in 14:0.111 32:0.25 51:0.40; do
  C=${pair%%:*}; r=${pair#*:}
  w=$(pick g_ours_C${C}_cur g_ours_C${C}_law); t=${w##*_}
  fb=$(pick g_ft_offload_r$r g_ft_hybrid_r$r); b=$(echo $fb | cut -d_ -f3)
  echo "gpt-oss C$C: ours $t, FreeToken $b" | tee -a $OUT/selection.txt
  oursg $C $t 2; ftg $b $r 2
done
# ---- Qwen3-30B-A3B BF16 (gpt-oss files removed first for disk; the conversion runs with no server up)
rm -rf "$HFD" "$F"; df -h $WORK | tail -1
GQ=$M/Qwen3-30B-A3B-BF16.gguf
( t0=$(date +%s); PYTHONPATH=$WORK/lc-stock/gguf-py $PY $WORK/lc-stock/convert_hf_to_gguf.py "$HFQ" --outtype bf16 --outfile $GQ
  echo "convert rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $GQ ) > $OUT/convert_qwen3.txt 2>&1
tail -2 $OUT/convert_qwen3.txt
LSQ="-m $GQ -ngl 99 -fa on -lm none -t $CORES -np 1 -c 2048 --no-webui --jinja"
oursq() {  # C table-name launch
  local t=$CURQ; [ $2 = law ] && t=$LAWQ; [ $2 = none ] && t=$NONEQ
  cl $HFQ q_ours_C$1_$2 $3 "{\"model\": \"Qwen3-30B-A3B\", \"system\": \"ours v2\", \"C\": $1, \"table\": \"$t\"}" --extra "$LXB" \
    --cmd "$ECENV LLAMA_EC_SLOTS=$1 LLAMA_EC_FETCH=$t LLAMA_EC_STATS=$OUT/srv_q_ours_C$1_$2_L$3.json $EC_SERVER $LSQ $OT --port {port}"
}
ftq() {  # backend rate launch mem-ratio (as jobs 080/081)
  ( cd $FT; cl $HFQ q_ft_$1_r$2 $3 "{\"model\": \"Qwen3-30B-A3B\", \"system\": \"FreeToken $1\", \"rate\": $2}" --ft $1 --cache-rate $2 --mem-ratio $4 )
}
if [ -s "$GQ" ]; then
  for C in 16 32 56; do oursq $C law 1; oursq $C cur 1; done
  oursq 32 none 1
  for rm_ in 0.125:0.9 0.25:0.9 0.4375:0.95; do r=${rm_%%:*}; mr=${rm_#*:}; ftq offload $r 1 $mr; ftq hybrid $r 1 $mr; done
  cl $HFQ q_llama_n36 1 '{"model": "Qwen3-30B-A3B", "system": "llama.cpp", "ncmoe": 36}' --extra "$LXB" --cmd "$STOCK_SERVER $LSQ -ncmoe 36 --port {port}"
  for trip in 16:0.125:0.9 32:0.25:0.9 56:0.4375:0.95; do
    C=${trip%%:*}; rest=${trip#*:}; r=${rest%%:*}; mr=${rest#*:}
    w=$(pick q_ours_C${C}_cur q_ours_C${C}_law); t=${w##*_}
    fb=$(pick q_ft_offload_r$r q_ft_hybrid_r$r); b=$(echo $fb | cut -d_ -f3)
    echo "qwen3 C$C: ours $t, FreeToken $b" | tee -a $OUT/selection.txt
    oursq $C $t 2; ftq $b $r 2 $mr
  done
else
  echo "Qwen3 GGUF missing: Qwen3 part skipped"
fi
cd $WORK
for f in $OUT/srv_*.log; do tail -n 40 "$f" > "$f.tail"; done; find $OUT -name 'srv_*.log' -delete
$PY - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, sys, os, statistics as st
out = sys.argv[1]
rows = [json.loads(l) for l in open(f"{out}/bs1.jsonl")] if os.path.exists(f"{out}/bs1.jsonl") else []
by = {}
for r in rows: by.setdefault(r["label"], []).append(r)
for lab, v in by.items():
    per = {}
    for r in v: per.setdefault(r["launch"], []).append(r["decode_tok_s"])
    print(f"{lab:24s} " + " ".join(f"L{L}: {st.mean(x):6.2f} (n={len(x)})" for L, x in sorted(per.items())) + f"  vram {sorted({round(r.get('vram_used_gib') or 0, 1) for r in v})}")
for f in sorted(os.listdir(out)):
    if f.endswith(".json") and f.startswith("srv_"):
        s = json.load(open(f"{out}/{f}")); n = s["steps"]
        print(f"{f:34s} table {s.get('fetch_table')} misses/token {s['misses'] / n:.1f} fetches {s.get('fetches', 0) / n:.1f}")
PY
