#!/bin/bash
# Job 090: output parity of our cache against an all-VRAM reference, as KL divergence over the full vocabulary, on one
# machine. Job 082 compared ours with stock llama.cpp's --n-cpu-moe placement (top-1 agreement 98.6 / 99.3%, |dNLL| <=
# 0.31%) but had no all-VRAM reference: a 32 GB card cannot hold either model. Target: an A100 SXM4 80 GB rental (Vast
# offers 46709200 or 41566564: EPYC hosts, 129 GB RAM, AVX2 only). Both models fit in its VRAM, so the reference is
# stock llama.cpp with every layer on the GPU (no --n-cpu-moe, no -ot), teacher-forced on job 082's parity text and step
# counts (llama-ec-bench: 12 sequences, prefill min(640, prompt), 128 decode steps each; gpt-oss its own-text corpus
# ec2/S_gpt-oss-120b.jsonl, Qwen3 a MATH-500 text corpus tokenized on the machine exactly as job 082 did).
# The build targets the A100: setup.sh takes the compute capability from nvidia-smi (sm_80 here) for the trees, concur.cu
# and bw.cu, and the patch has no hard-coded architecture; the expert-cache helpers detect AVX-512 at run time
# (__builtin_cpu_supports) and take the ggml vec_dot path on AVX2 hosts, as on the EPYC 7352 in job 069c.
# llama-ec-bench's --dump holds only the top-5 logits of each step (int32 ids, float32 values), from which no KL over the
# vocabulary and no lumped remainder mass can be computed (the normaliser is unknown), so a second small patch
# (ec2/ec-bench-fulldump.patch, on the bench tool only) adds --dump-full: every step's float32 logit vector. The dumps
# (1.2 GB per gpt-oss run, 0.9 GB per Qwen3 run; ours writes one per config) stay in /work and are deleted after the KL
# is computed on the machine; only parity_kl.json and the top-5 dumps go to the results. The reference binary is
# llama-ec-bench compiled in the stock tree (the tool source is stock-API only); if that build fails, the patched tree's
# binary with the cache off is used and the manifest says so.
# Runs, both models (gpt-oss-120b MXFP4 first, then its file is removed and Qwen3-30B-A3B is converted to BF16 GGUF):
#   (a) all-VRAM reference: stock llama.cpp, every layer on the GPU, dumping logits;
#   (b) ours with the cache at 25% (gpt-oss C32, Qwen3 C32), the law's table computed on the machine first (fetch_table.py
#       from concur.txt), dumping logits (config 0 warms the cache, config 1 is the measured one, dumped as .1);
#   (c) stock llama.cpp with job 082's --n-cpu-moe placement (gpt-oss 27, Qwen3 36), the comparison the paper reports;
#   (a') control: the patched tree's binary with the cache off, every layer on the GPU (expected to match (a) exactly).
# Then, on the machine, from the dumps (ec2/kl_from_dumps.py): per-step KL(reference || system) over the full vocabulary
# (mean, median, 99th and 99.9th percentiles, max), top-1 agreement, mean NLL of the forced text and its difference from
# the reference's; parity_kl.json; dumps deleted.
# Predictions, committed before launch:
#   1. mean KL of ours from the all-VRAM reference is at most 0.01 nats on both models, and the 99.9th percentile at most
#      0.5;
#   2. top-1 agreement with the all-VRAM reference is at least 98% on both models, and stock llama.cpp's --n-cpu-moe
#      placement agrees with the reference within 1 point of ours' agreement;
#   3. ours' mean NLL on the forced text is within +-0.5% of the reference's.
# Budget: the job stops starting steps 2.5 h after launch (every step bounded by its own timeout and by that deadline;
# skipped steps in skipped.txt); disk about 160 GB (gpt-oss GGUF 59 GB removed before Qwen3's 57 GB checkpoint is
# converted to a 57 GB GGUF; full-logit dumps about 6 GB per model, deleted after the KL).
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
T0=$(date +%s); BUDGET=$(( 2 * 3600 + 30 * 60 )); RESERVE=480
secs() { case $1 in *h) echo $(( ${1%h} * 3600 ));; *m) echo $(( ${1%m} * 60 ));; *s) echo ${1%s};; *) echo $1;; esac; }
left() { echo $(( T0 + BUDGET - RESERVE - $(date +%s) )); }
R() {  # timeout cmd...: bounded by the step's timeout and by the job's deadline; past the deadline the step is skipped
  local t=$(secs $1); shift; local l=$(left)
  if [ "$l" -lt 120 ]; then echo "SKIPPED (deadline): $*" | tee -a $OUT/skipped.txt; return 124; fi
  [ "$t" -gt "$l" ] && t=$l
  timeout -k 30 "$t" "$@"
}
killsrv() { pkill -f "llama-ec-bench|llama-server" 2>/dev/null; sleep 5; }
# --- gate, before any download: an 80 GB card (the all-VRAM reference needs it); the architecture is recorded
nvidia-smi --query-gpu=name,memory.total,compute_cap,clocks.max.sm,clocks.max.mem,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current,driver_version,power.limit --format=csv > $OUT/gpu.csv 2>&1
cat $OUT/gpu.csv
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "GPU memory ${VRAM} MiB, compute capability sm_$SM (A100: sm_80)" | tee $OUT/gate.txt
[ "${VRAM:-0}" -ge 75000 ] || { echo "WRONG GPU: ${VRAM} MiB is not an 80 GB card, stopping"; exit 4; }
[ "$SM" = 80 ] || echo "note: compute capability sm_$SM is not the A100's sm_80; the build targets sm_$SM"
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip >> $OUT/apt.txt 2>&1
python3 -m pip install -q --break-system-packages uv huggingface_hub hf_xet > $OUT/pip_uv.txt 2>&1
CORES=$(usable_cores); echo "usable physical cores $CORES (nproc $(nproc))" | tee $OUT/cores.txt
grep -o "avx512[a-z]*\|avx2" /proc/cpuinfo | sort -u | tr '\n' ' ' | tee $OUT/isa.txt; echo
# --- downloads first: the gpt-oss GGUF (no FreeToken here, so no HF checkpoint) and the Qwen3 checkpoint
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
HFQ=$M/Qwen3-30B-A3B
( t0=$(date +%s); python3 -c "import sys; from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-30B-A3B', local_dir=sys.argv[1])" "$HFQ"
  echo "qwen3 hf download rc=$? in $(( $(date +%s) - t0 )) s"; du -sh "$HFQ" ) > $OUT/dl_qwen3.txt 2>&1 &
DLQ=$!
VE=$WORK/venv   # CPU-torch environment: the Qwen3 conversion and tokenizer (as jobs 085b/087), numpy for the KL
( uv venv -q $VE && . $VE/bin/activate && uv pip install -q torch --index-url https://download.pytorch.org/whl/cpu &&
  uv pip install -q numpy transformers sentencepiece safetensors protobuf huggingface_hub hf_xet ) > $OUT/venv_install.txt 2>&1
echo "venv install rc=$?"
PY=$VE/bin/python
$PY -c "import torch, transformers, numpy; print(torch.__version__, transformers.__version__, numpy.__version__)" | tee $OUT/venv_ok.txt
$PY -c "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('HuggingFaceH4/MATH-500','test.jsonl',repo_type='dataset'), '$WORK/math500.jsonl')" && wc -l $WORK/math500.jsonl | tee $OUT/math500_count.txt
# --- builds (sm_$SM): the patched tree's bench with the full-logit dump; the stock tree with the same tool source
build_tree $WORK/lc-ec "$J/llama.cpp-expert-cache-4da6337.patch" llama-ec-bench
( cd $WORK/lc-ec && git apply "$J/ec-bench-fulldump.patch" && cmake --build build -j$NPROC --target llama-ec-bench ) > $OUT/build_fulldump.txt 2>&1
echo "full-dump bench build rc=$?"; grep -E "error" $OUT/build_fulldump.txt | head -5
$EC_BIN --dump-full $WORK/x.bin --corpus /dev/null > $OUT/ecbench_flagcheck.txt 2>&1; cat $OUT/ecbench_flagcheck.txt | head -2
FULL=1; grep -q "unknown argument" $OUT/ecbench_flagcheck.txt && { FULL=""; echo "WARNING: --dump-full missing from the bench (KL will fall back to the top-5 dumps)"; }
fd() { [ -n "$FULL" ] && echo "--dump-full $1"; }   # the full-logit dump option, when the bench has it
build_tree $WORK/lc-stock "" llama
( cd $WORK/lc-stock && mkdir -p tools/ec-bench && cp $WORK/lc-ec/tools/ec-bench/ec-bench.cpp tools/ec-bench/ &&
  printf 'set(TARGET llama-ec-bench)\nadd_executable(${TARGET} ec-bench.cpp)\ntarget_link_libraries(${TARGET} PRIVATE llama ggml ${CMAKE_THREAD_LIBS_INIT})\ntarget_compile_features(${TARGET} PRIVATE cxx_std_17)\n' > tools/ec-bench/CMakeLists.txt &&
  echo 'add_subdirectory(ec-bench)' >> tools/CMakeLists.txt &&
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$SM -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON > $OUT/cmake_lc-stock-ecbench.txt 2>&1 &&
  cmake --build build -j$NPROC --target llama-ec-bench ) > $OUT/build_lc-stock-ecbench.txt 2>&1
echo "stock-tree bench build rc=$?"; grep -E "error" $OUT/build_lc-stock-ecbench.txt | head -5
STOCK_EC=$WORK/lc-stock/build/bin/llama-ec-bench
if [ -x "$STOCK_EC" ]; then REF_BIN=$STOCK_EC; REF_KIND="stock tree (llama.cpp 4da6337 + the bench tool only)"; else REF_BIN=$EC_BIN; REF_KIND="expert-cache tree, cache off (stock-tree bench build FAILED)"; fi
echo "reference binary: $REF_BIN = $REF_KIND" | tee $OUT/reference_binary.txt
echo "builds done, waiting for the downloads ($(( $(date +%s) - T0 )) s since launch)"
wait $DLG $DLQ
cat $OUT/dl_gguf.txt $OUT/dl_qwen3.txt
F=$M/gpt-oss-120b-MXFP4.gguf
[ -s "$F" ] && echo "GGUF ok $(stat -c %s $F) bytes" || echo "GGUF MISSING"
# --- the host (idle): platform, the probe, the law's tables
platform
H=$(( CORES > 4 ? CORES - 2 : 2 ))
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
grep -q "concurrent t=" $OUT/concur.txt || { echo "concur incomplete, retrying"; R 15m $WORK/concur > $OUT/concur.txt 2>&1; }
LAWG=$(python3 $J/fetch_table.py $OUT/concur.txt $H 13253760 4 $OUT/fetch_table_law_gptoss.json 37)
LAWQ=$(python3 $J/fetch_table.py $OUT/concur.txt $H 9437184 8 $OUT/fetch_table_law_qwen3.json 48)
echo "law tables ($(date -u +%FT%TZ)): gpt-oss $LAWG; qwen3 $LAWQ (listing B 0,0,1,1,2 / 0,0,1,1,2,3,3,4,5)" | tee $OUT/tables.txt
[ -n "$LAWG" ] && [ -n "$LAWQ" ] || { echo "no law table"; exit 3; }
D=$WORK/dumps; mkdir -p $D
waitfree() {
  for i in $(seq 120); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-0}" -lt 1500 ] && return; sleep 2; done
  echo "GPU memory still in use: $u MiB"; killsrv
}
run1() {  # label command (teacher-forced bench; stdout = one JSON line per sequence, stderr kept as a tail)
  local lab=$1 cmd=$2
  waitfree
  local t0=$(date +%s)
  R 40m bash -c "$cmd" > $OUT/par_$lab.jsonl 2> $OUT/par_$lab.err
  local rc=$?
  echo "parity $lab rc=$rc in $(( $(date +%s) - t0 )) s"; echo "par_$lab 1 $rc $(( $(date +%s) - t0 ))" >> $OUT/runs.txt
  tail -n 40 $OUT/par_$lab.err > $OUT/par_$lab.err.tail; rm -f $OUT/par_$lab.err
  [ $rc = 124 ] && killsrv
}
parity() {  # tag model-file corpus C law-table ncmoe
  local tag=$1 mf=$2 corpus=$3 C=$4 t=$5 n=$6
  local COMMON="--corpus $corpus --seqs 0,1,2,3,4,5,6,7,8,9,10,11 -t $CORES --n-prefill 640 --n-decode 128 --no-mmap"
  local cfg="slots=$C:policy=dfa:kappa=1:mailbox=1:tdec=1:helpers=$H:fetch=$t"
  run1 ${tag}_ref "$REF_BIN -m $mf $COMMON --dump $OUT/par_${tag}_ref.bin $(fd $D/full_${tag}_ref.bin)"
  run1 ${tag}_ours "$EC_BIN -m $mf $COMMON --host-experts --ec '$cfg;$cfg:stats=$OUT/par_${tag}_ours.json' --dump $OUT/par_${tag}_ours.bin $(fd $D/full_${tag}_ours.bin)"
  run1 ${tag}_n$n "$EC_BIN -m $mf $COMMON --ncmoe $n --dump $OUT/par_${tag}_n$n.bin $(fd $D/full_${tag}_n$n.bin)"
  run1 ${tag}_refec "$EC_BIN -m $mf $COMMON --dump $OUT/par_${tag}_refec.bin $(fd $D/full_${tag}_refec.bin)"
  ls -la $D/ | tee -a $OUT/dumps_ls.txt
  local meta="{\"model\": \"$tag\", \"C\": $C, \"table\": \"$t\", \"ncmoe\": $n, \"reference\": \"$REF_KIND\", \"n_seq\": 12, \"n_prefill\": 640, \"n_decode\": 128}"
  if [ -s $D/full_${tag}_ref.bin ]; then
    R 30m $PY $J/kl_from_dumps.py --ref $D/full_${tag}_ref.bin --sys ours=$D/full_${tag}_ours.bin.1 --sys n$n=$D/full_${tag}_n$n.bin \
      --sys refec=$D/full_${tag}_refec.bin --sys ours_warmup_pass=$D/full_${tag}_ours.bin.0 --out $OUT/parity_kl_$tag.json --meta "$meta" | tee -a $OUT/parity_kl.txt
  else
    echo "no full dump for $tag: KL from the top-5 dumps (approximate, see kl_from_dumps.py)" | tee -a $OUT/parity_kl.txt
    R 10m $PY $J/kl_from_dumps.py --topk --ref $OUT/par_${tag}_ref.bin --sys ours=$OUT/par_${tag}_ours.bin.1 --sys n$n=$OUT/par_${tag}_n$n.bin \
      --sys refec=$OUT/par_${tag}_refec.bin --out $OUT/parity_kl_$tag.json --meta "$meta" | tee -a $OUT/parity_kl.txt
  fi
  rm -f $D/full_${tag}_*; df -h $WORK | tail -1
}
# ================= gpt-oss-120b =================
parity g $F $J/S_gpt-oss-120b.jsonl 32 $LAWG 27
# ================= Qwen3-30B-A3B BF16 (gpt-oss file removed first for disk; the conversion runs with no GPU process up) =================
rm -f "$F"; df -h $WORK | tail -1
GQ=$M/Qwen3-30B-A3B-BF16.gguf
( t0=$(date +%s); PYTHONPATH=$WORK/lc-stock/gguf-py timeout 30m $PY $WORK/lc-stock/convert_hf_to_gguf.py "$HFQ" --outtype bf16 --outfile $GQ
  echo "convert rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $GQ ) > $OUT/convert_qwen3.txt 2>&1
tail -2 $OUT/convert_qwen3.txt
# the parity corpus of job 082: MATH-500 problems 100.. as "Problem/Solution" text, 12 sequences of >= 800 tokens, Qwen3 tokenizer
$PY - "$HFQ" "$WORK/math500.jsonl" "$WORK/S_qwen3_math.jsonl" <<'PY' > $OUT/qwen3_corpus.txt 2>&1
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
cat $OUT/qwen3_corpus.txt; cp $WORK/S_qwen3_math.jsonl $OUT/S_qwen3_math.jsonl
if [ -s "$GQ" ]; then
  parity q $GQ $WORK/S_qwen3_math.jsonl 32 $LAWQ 36
else
  echo "Qwen3 GGUF missing: Qwen3 part skipped"
fi
cd $WORK
$PY - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, sys, os
out = sys.argv[1]
def nll(p, measured_only=False):
    v = [json.loads(l) for l in open(p) if l.strip().startswith("{")] if os.path.exists(p) else []
    v = [r for r in v if "nll_sum" in r and (not measured_only or "stats=" in r.get("config", ""))]
    return (sum(r["nll_sum"] for r in v) / max(1, sum(r["nll_n"] for r in v)), len(v), sum(r["decode_ms"] for r in v) / 1e3,
            sum(r["n_decode"] for r in v) / max(1e-9, sum(r["decode_ms"] for r in v) / 1e3)) if v else None
for tag in ("g", "q"):
    for name in ("ref", "ours", f"n{27 if tag == 'g' else 36}", "refec"):
        r = nll(f"{out}/par_{tag}_{name}.jsonl", name == "ours")
        if r: print(f"parity {tag} {name:6s}: {r[1]} sequences, NLL {r[0]:.4f}, decode {r[2]:.1f} s, {r[3]:.1f} tok/s (bench's own timing)")
allkl = {}
line = []
for tag in ("g", "q"):
    p = f"{out}/parity_kl_{tag}.json"
    if not os.path.exists(p): continue
    d = json.load(open(p)); allkl[tag] = d
    for name, s in d["systems"].items():
        k = s.get("kl_nats") or s.get("kl_nats_top5_renormalised") or {}
        if "error" in s: print(f"KL {tag} {name}: ERROR {s['error']}"); continue
        print(f"KL {tag} {name:16s} mode {s.get('mode')}: mean {k.get('mean', float('nan')):.5f} median {k.get('median', float('nan')):.5f} p99 {k.get('p99', float('nan')):.4f} "
              f"p99.9 {k.get('p999', float('nan')):.4f} max {k.get('max', float('nan')):.4f} | top-1 {s.get('top1_agreement', float('nan')):.4f} "
              f"| NLL {s.get('nll_sys', float('nan')):.4f} vs ref {s.get('nll_ref', float('nan')):.4f} ({s.get('dnll_pct', float('nan')):+.3f}%) | steps {s.get('n_steps')}")
        if name == "ours":
            line.append(f"{tag}: KL mean {k.get('mean', float('nan')):.4f} p99.9 {k.get('p999', float('nan')):.3f} top1 {s.get('top1_agreement', float('nan')):.3f} dNLL {s.get('dnll_pct', float('nan')):+.2f}%")
json.dump({"job": "090_parity_a100@vast", "models": allkl, "reference_binary": open(f"{out}/reference_binary.txt").read().strip() if os.path.exists(f"{out}/reference_binary.txt") else None},
          open(f"{out}/parity_kl.json", "w"), indent=1)
open(f"{out}/oneline.txt", "w").write("090 ours vs all-VRAM reference: " + ("; ".join(line) if line else "no KL computed"))
PY
ls -la $D 2>/dev/null; rm -rf $D
ONE=$(cat $OUT/oneline.txt 2>/dev/null || echo "090: no summary")
python3 $J/manifest.py "$OUT" 090_parity_a100@vast "$T0" "$ONE" law_gptoss=$LAWG law_qwen3=$LAWQ reference_binary="$REF_KIND" sm=$SM
echo "SUMMARY $ONE"
