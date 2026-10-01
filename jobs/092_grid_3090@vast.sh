#!/bin/bash
# Job 092: the paper's grid, card 3 - Table 1's protocol and the probed speed limit on an RTX 3090 (24 GB GDDR6X, datasheet
# 936 GB/s, PCIe 4.0 x16; the cheapest card on Vast, about $0.21 an hour), plus a fourth model that is a real offloading
# case on 24 GB. Target: a Vast RTX 3090 host with at least 128 GB of RAM. Same protocol as job 091 (derived from 089: both
# models downloaded first, FreeToken 0d652e7, the law's tables computed on the machine from its own probe before any model
# run with S = 13,253,760 bytes (gpt-oss, MXFP4 gate_up + down + f32 biases of one expert) and 9,437,184 (Qwen3 BF16, 3 x
# 2048 x 768 x 2) as data/gguf_bytes.json in the main repo accounts the GGUF headers, g 37 / 48 us and L_c 20 us frozen
# from the 5090 profiles; per cell ours -> FreeToken -> llama.cpp in launch 1 and FreeToken -> ours in launch 2; bs1_client
# with 30 AIME-25 problems, 256 tokens, greedy, held-out warm-up, session, GPU-side sampling for llama-server; FreeToken's
# backend per cell the carry-over of job 081's selection: hybrid at gpt-oss 11% and at Qwen3 12.5 and 25%, offload at Qwen3
# 43.75%; stock llama.cpp at matched expert count; results, manifest, the deadline-aware R wrapper, runs.txt, SUMMARY), with:
#   - gates: the card's name, memory, PCIe link, clocks.max.memory and clocks.max.sm recorded in gate.txt; exit 4 unless the
#     name holds "3090" and memory.total is 23,000-25,000 MiB, exit 5 if bw.cu's device read is below 700 GB/s (datasheet
#     936; 089's gate was 84% of the 5090's datasheet, 700 is 75% of this one's); host RAM recorded, a note below 120 GB;
#   - cells: gpt-oss-120b at C 14 only (11%; FreeToken 0.111, llama.cpp -ncmoe 32; both launches). Qwen3-30B-A3B BF16 at
#     C 16 / 32 / 56 (0.125 / 0.25 / 0.4375; -ncmoe 42 / 36 / 27), each cell behind the VRAM arithmetic below: C 56 is
#     25.4 GB of slots alone (56 x 9.44 MB x 48) and is expected to be skipped with a note, leaving Qwen3 at 12.5 and 25%;
#   - the fourth model, after Qwen3: Mixtral-8x7B-Instruct-v0.1 Q4_K_M (mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF,
#     Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf, the file data/gguf_bytes.json names; 26 GB of experts plus 1.1 GB dense, so
#     it does not fit a 24 GB card: a real offloading case; 32 layers, 8 experts per layer, top-2), ours and stock llama.cpp
#     only (FreeToken does not support it), launch 1 only, ours first, at C = 2 and C = 4 experts per layer against
#     --n-cpu-moe at matched expert count (8 experts x (32 - n) GPU layers = C x 32: n = 24 for C 2, n = 16 for C 4). The law's
#     table for it is computed on the machine with S = the mean bytes of one expert over the 32 layers read from the GGUF
#     header (vram_check.py; the Q4_K_M mix puts ffn_down in Q6_K on 16 of the layers: 99,090,432 bytes on the Q4_K layers,
#     114,229,248 on the others, mean 106,659,840, the value data/gguf_bytes.json gives and the fallback if the header read
#     fails), k = 2, and g = 230 us (two experts of 106.66 MB read from VRAM at the 3090's datasheet 936 GB/s = 228 us plus
#     the small ops; a stated estimate, not a profile: no Mixtral profile exists). The llama.cpp law prediction
#     (law_llama.json) has no Mixtral entry: its GPU term G has not been measured;
#   - no LRU attribution and no llama-batched-bench (as 091);
#   - VRAM arithmetic before each cell (vram_check.py, written by this job, reads the GGUF header on the machine): C x
#     expert bytes x layers + the dense weights on the GPU (every tensor that is not an expert and not token_embd, the
#     output head included) + the KV at the run's -c (8448 for gpt-oss and Mixtral, 2048 for Qwen3 as 089; f16, every layer
#     at full context). The cell runs when the sum is <= 23 GB, else it is skipped with a note (the 5090 runs show about
#     1.1 GiB on top of the sum in nvidia-smi). Expected: gpt-oss C14 6.7 + 1.7 + 0.6 = 9.0 GB; Qwen3 C16 9.9, C32 17.1,
#     C56 28.0 (skip); Mixtral C2 6.8 + 1.1 + 1.1 = 9.0 GB, C4 13.7 + 1.1 + 1.1 = 15.8 GB;
#   - FreeToken on 24 GB: as in 091, --memory-ratio 0.9 leaves about 3 GB to the KV at every kept cell (gpt-oss at 0.111:
#     4.3 GB of weights + 512 slots x 13.2 MB = 11 GB; Qwen3 at 0.25: 3.1 + 14.5 GB), so the ratios of jobs 080-089 stay
#     (gpt-oss at the client's default 0.9, Qwen3 0.9 / 0.9 / 0.95). FreeToken on sm_86 is untested in these jobs (its code
#     emulates fp8 on pre-sm_89 cards and names the 3090); if its server fails to start, the run's rc is in runs.txt and the
#     ours / llama.cpp comparison stands;
#   - the speed limit is scored offline (scripts/speed_limit.py, mosl/ecsim_fast and the trace live in the main repo): the
#     job records the probe (concur.txt) and the device read (bw_gate.txt, bw.txt); B_gpu = 936 GB/s for this card.
# Run list, in order:
#   gpt-oss-120b C 14: launch 1 ours (law's table), FreeToken hybrid, llama.cpp -ncmoe 32; launch 2 FreeToken hybrid, ours;
#   gpt-oss files removed; Qwen3 converted to BF16 GGUF (CPU-torch venv);
#   Qwen3 C 16, 32, (56): launch 1 ours, FreeToken hybrid / hybrid / offload, llama.cpp -ncmoe 42 / 36 / 27; launch 2 FreeToken, ours;
#   Mixtral C 2 then C 4: launch 1 ours (law's table), llama.cpp --n-cpu-moe 24 / 16.
# Predictions, committed before launch:
#   1. ours leads FreeToken at every Qwen3 cell (launch 2, paired CI above 1), by 0-15%; at gpt-oss 11% by 10-35%;
#   2. ours runs at least 1.8x llama.cpp at every Qwen3 and gpt-oss cell;
#   3. on Mixtral, ours runs at least 1.3x llama.cpp at C = 2 and at C = 4;
#   4. the law's table on this machine fetches no more than the headline machine's at any entry (gpt-oss 0,0,1,1,2; Qwen3
#      0,0,1,1,2,3,3,4,5);
#   5. the two launch orders agree within 3% at every Qwen3 cell (gpt-oss 11% has both launches too and is reported).
# Budget: the job stops starting steps 4 h after launch (every step bounded by its own timeout and by that deadline;
# skipped steps in skipped.txt); disk about 240 GB (gpt-oss 65 + 63 GB and Qwen3's 57 GB checkpoint and the 26 GB Mixtral
# GGUF at the peak; gpt-oss removed before Qwen3's 61 GB GGUF is written).
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
export BASE; export -f build_tree platform getmodel   # setup.sh steps run under timeout below
T0=$(date +%s); BUDGET=$(( 4 * 3600 )); RESERVE=600
secs() { case $1 in *h) echo $(( ${1%h} * 3600 ));; *m) echo $(( ${1%m} * 60 ));; *s) echo ${1%s};; *) echo $1;; esac; }
left() { echo $(( T0 + BUDGET - RESERVE - $(date +%s) )); }
R() {  # timeout cmd...: bounded by the step's timeout and by the job's deadline; past the deadline the step is skipped
  local t=$(secs $1); shift; local l=$(left)
  if [ "$l" -lt 120 ]; then echo "SKIPPED (deadline): $*" | tee -a $OUT/skipped.txt; return 124; fi
  [ "$t" -gt "$l" ] && t=$l
  timeout -k 30 "$t" "$@"
}
killsrv() { pkill -f "llama-server|freetoken" 2>/dev/null; sleep 5; }
# --- gates, before any download: the card (name, memory, link, clocks recorded; a 24 GB 3090 required) and the device read
nvidia-smi --query-gpu=name,memory.total,compute_cap,clocks.max.sm,clocks.max.memory,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current,driver_version,power.limit --format=csv > $OUT/gpu.csv 2>&1
cat $OUT/gpu.csv
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/^ *//; s/ *$//')
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
MEMCLK=$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits | head -1 | tr -d ' ')
SMCLK=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1 | tr -d ' ')
PCIE=$(nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current --format=csv,noheader | head -1 | tr -d ' ')
RAM=$(free -g | awk '/^Mem:/ {print $2}')
{ echo "gpu $GPU, memory.total ${VRAM} MiB, sm_$SM"
  echo "clocks.max.memory ${MEMCLK} MHz, clocks.max.sm ${SMCLK} MHz"
  echo "pcie gen.max,gen.current,width.max,width.current $PCIE (PCIe 4.0 x16 expected)"
  echo "host RAM ${RAM} GB (128 wanted)"; } | tee $OUT/gate.txt
case "$GPU" in *3090*) ;; *) echo "WRONG GPU $GPU (RTX 3090 expected)" | tee -a $OUT/gate.txt; exit 4;; esac
[ "${VRAM:-0}" -ge 23000 ] && [ "${VRAM:-0}" -le 25000 ] || { echo "WRONG GPU MEMORY ${VRAM} MiB (a 24 GB card expected)" | tee -a $OUT/gate.txt; exit 4; }
[ "${RAM:-0}" -ge 120 ] || echo "note: host RAM ${RAM} GB is below 120 GB" | tee -a $OUT/gate.txt
nvcc -O3 -arch=sm_$SM "$J/bw.cu" -o $WORK/bw0 && $WORK/bw0 > $OUT/bw_gate.txt 2>&1
nvidia-smi -q > $OUT/nvidia-smi-q-start.txt 2>&1
DR=$(sed -n 's/device read 1 GiB: *\([0-9]*\).*/\1/p' $OUT/bw_gate.txt)
echo "device read ${DR} GB/s (datasheet 936; gate 700)" | tee -a $OUT/gate.txt
[ "${DR:-0}" -ge 700 ] || { echo "GPU reads below 700 GB/s: throttled card, stopping"; exit 5; }
DEBIAN_FRONTEND=noninteractive timeout 15m apt-get install -y -qq python3-venv python3-pip >> $OUT/apt.txt 2>&1
timeout 10m python3 -m pip install -q --break-system-packages uv huggingface_hub hf_xet > $OUT/pip_uv.txt 2>&1
CORES=$(usable_cores); echo "usable physical cores $CORES (nproc $(nproc))" | tee $OUT/cores.txt
# --- downloads first (gpt-oss checkpoint and GGUF, Qwen3 checkpoint, Mixtral GGUF); FreeToken, the conversion venv and
# the builds run meanwhile
HFD=$M/gpt-oss-120b; mkdir -p $M
( t0=$(date +%s); timeout 100m python3 - "$HFD" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("openai/gpt-oss-120b", local_dir=sys.argv[1], ignore_patterns=["original/*", "metal/*"])
PY
  echo "hf download rc=$? in $(( $(date +%s) - t0 )) s"; du -sh "$HFD" ) > $OUT/dl_hf.txt 2>&1 &
DLH=$!
( t0=$(date +%s)
  timeout 100m python3 -c "import sys; from huggingface_hub import hf_hub_download as d; d('ggml-org/gpt-oss-120b-GGUF', 'gpt-oss-120b-MXFP4.gguf', local_dir=sys.argv[1])" "$M"
  echo "gguf hf download rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $M/gpt-oss-120b-MXFP4.gguf
  if [ ! -s $M/gpt-oss-120b-MXFP4.gguf ]; then
    rm -f $M/gpt-oss-120b-MXFP4.gguf
    for i in 1 2 3; do timeout 60m bash -c 'getmodel "$@"' _ ggml-org/gpt-oss-120b-GGUF gpt-oss-120b-MXFP4.gguf; [ -f $M/gpt-oss-120b-MXFP4.gguf ] && break; done
  fi
) > $OUT/dl_gguf.txt 2>&1 &
DLG=$!
HFQ=$M/Qwen3-30B-A3B
( t0=$(date +%s); timeout 100m python3 -c "import sys; from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-30B-A3B', local_dir=sys.argv[1])" "$HFQ"
  echo "qwen3 hf download rc=$? in $(( $(date +%s) - t0 )) s"; du -sh "$HFQ" ) > $OUT/dl_qwen3.txt 2>&1 &
DLQ=$!
GMF=Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf; GM=$M/$GMF
( t0=$(date +%s)
  timeout 100m python3 -c "import sys; from huggingface_hub import hf_hub_download as d; d('mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF', sys.argv[2], local_dir=sys.argv[1])" "$M" "$GMF"
  echo "mixtral gguf hf download rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $GM
  if [ ! -s $GM ]; then
    rm -f $GM
    for i in 1 2 3; do timeout 60m bash -c 'getmodel "$@"' _ mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF $GMF; [ -f $GM ] && break; done
  fi
) > $OUT/dl_mixtral.txt 2>&1 &
DLM=$!
FT=$WORK/ft; export FT_DIR=$FT
( cd $WORK && git init -q ft && cd ft && git remote add origin https://github.com/FlashML-org/FreeToken &&
  timeout 15m git fetch -q --depth 1 origin 0d652e73a452d014ac5441a15baa75348e9fcb0a && git checkout -q FETCH_HEAD ) || echo "FreeToken clone FAILED"
nvcc --version | tail -2 | tee $OUT/nvcc_version.txt   # FreeToken builds kernels with nvcc and needs the CUDA 13 toolkit (torch cu130)
( cd $FT && uv venv -q .venv && . .venv/bin/activate && time timeout 45m uv pip install -q -e ".[accel]" huggingface_hub hf_xet gguf sentencepiece transformers safetensors ) > $OUT/ft_install.txt 2>&1
echo "FreeToken install rc=$?"
PY=$FT/.venv/bin/python; FTBIN=$FT/.venv/bin/ft
$PY -c "import freetoken, huggingface_hub" > $OUT/ft_import.txt 2>&1 || { echo "FreeToken not importable, stopping"; cat $OUT/ft_import.txt; tail -20 $OUT/ft_install.txt; kill $DLH $DLG $DLQ $DLM 2>/dev/null; exit 3; }
$FTBIN --version > $OUT/ft_version.txt 2>&1
VE=$WORK/venv   # CPU-torch environment for the Qwen3 conversion (job 085's conversion hung in FreeToken's venv)
( uv venv -q $VE && . $VE/bin/activate && timeout 30m uv pip install -q torch --index-url https://download.pytorch.org/whl/cpu &&
  timeout 15m uv pip install -q numpy transformers sentencepiece safetensors protobuf ) > $OUT/venv_install.txt 2>&1 &
VEP=$!
timeout 10m $PY -c "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('math-ai/aime25','test.jsonl',repo_type='dataset'), '$WORK/aime25.jsonl')" && export FREETOKEN_AIME25_JSONL=$WORK/aime25.jsonl
wc -l $WORK/aime25.jsonl | tee $OUT/aime25_count.txt
timeout 10m $PY - "$WORK/warmup.txt" <<'PY' > $OUT/warmup_source.txt 2>&1
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
# the VRAM arithmetic of ours for one cell, from the GGUF header on the machine (stdlib only; the header reader of
# mosl/gguf_bytes.py in the main repo); its "mean" mode gives Mixtral's expert bytes for the law's table
cat > $WORK/vram_check.py <<'PY'
"""VRAM arithmetic of ours for one cell, from the GGUF header (jobs 091/092).
slots: C x bytes of one expert per layer, summed over the MoE layers (gpt-oss's f32 expert biases included, as in
data/gguf_bytes.json); dense: every other tensor but token_embd (llama.cpp keeps the embedding table on the CPU; the output
head is on the GPU); KV: CTX tokens x layers x kv heads x (key + value length) x 2 bytes (f16; every layer at full
context, an upper bound for gpt-oss's sliding-window layers). The cell fits when the sum is <= 23 GB (the RTX 5090 runs of
jobs 080-089 showed about 1.1 GiB more than this sum in nvidia-smi: the compute buffer and the CUDA context).
    python3 vram_check.py GGUF C CTX TAG   -> one line; exit 0 fits, 2 does not fit, 3 header unreadable
    python3 vram_check.py GGUF mean        -> mean bytes of one expert over the MoE layers (for fetch_table.py)"""
import struct
import sys

GGML = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
        10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292),
        16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82),
        23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8), 29: (256, 56), 30: (1, 2),
        34: (256, 54), 35: (256, 66), 39: (32, 17)}
SC = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]


def rs(f):
    return f.read(rd(f, "Q")).decode("utf-8", "replace")


def val(f, t):
    if t in SC:
        return rd(f, SC[t])
    if t == 8:
        return rs(f)
    if t == 9:
        et, n = rd(f, "I"), rd(f, "Q")
        if et in SC:
            f.read(n * struct.calcsize(SC[et]))
            return None
        return [val(f, et) for _ in range(n)]
    raise ValueError(f"gguf value type {t}")


def header(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF", "not a GGUF file"
    rd(f, "I")
    nt, nkv = rd(f, "Q"), rd(f, "Q")
    kv = {}
    for _ in range(nkv):
        k = rs(f)
        kv[k] = val(f, rd(f, "I"))
    ts = []
    for _ in range(nt):
        name = rs(f)
        nd = rd(f, "I")
        dims = [rd(f, "Q") for _ in range(nd)]
        t = rd(f, "I")
        rd(f, "Q")
        n = 1
        for d in dims:
            n *= d
        be, bb = GGML[t]
        ts.append((name, n // be * bb))
    return kv, ts


def main():
    kv, ts = header(sys.argv[1])
    arch = kv["general.architecture"]
    L, E = kv[f"{arch}.block_count"], kv[f"{arch}.expert_count"]
    per, dense, head = {}, 0, 0
    for name, nb in ts:
        if "_exps" in name:
            li = int(name.split(".")[1])
            per[li] = per.get(li, 0) + nb
        elif name.startswith("token_embd"):
            continue
        else:
            dense += nb
            if name.startswith("output."):
                head += nb
    S = [per[l] / E for l in sorted(per)]
    if sys.argv[2] == "mean":
        print(int(round(sum(S) / len(S))))
        return 0
    C, ctx, tag = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    nh = kv[f"{arch}.attention.head_count"]
    nkv = kv.get(f"{arch}.attention.head_count_kv", nh)
    nh, nkv = (max(x) if isinstance(x, list) else x for x in (nh, nkv))
    hd = kv.get(f"{arch}.attention.key_length") or kv[f"{arch}.embedding_length"] // nh
    vd = kv.get(f"{arch}.attention.value_length") or hd
    slots, kvb = C * sum(S), ctx * L * nkv * (hd + vd) * 2
    tot = slots + dense + kvb
    ok = tot <= 23e9
    print(f"vram {tag} C{C}: slots {slots / 1e9:.2f} GB ({C} x {sum(S) / len(S) / 1e6:.2f} MB mean expert x {len(S)} layers; "
          f"per-layer expert bytes {sorted(set(int(round(s)) for s in S))}) + dense {dense / 1e9:.2f} GB (output head {head / 1e9:.2f}, "
          f"token_embd on the CPU) + KV {kvb / 1e9:.2f} GB (-c {ctx}: {L} layers x {nkv} kv heads x {hd}+{vd} x f16) = {tot / 1e9:.2f} GB "
          f"-> {'fits' if ok else 'DOES NOT FIT'} (limit 23 GB; nvidia-smi will show about 1.1 GiB more)")
    return 0 if ok else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"vram check failed: {e!r}")
        sys.exit(3)
PY
vcheck() {  # tag gguf C ctx: prints the arithmetic (also to vram_check.txt); returns 0 when the cell fits
  local msg rc
  msg=$(python3 $WORK/vram_check.py "$2" "$3" "$4" "$1" 2>&1); rc=$?
  echo "$msg" | tee -a $OUT/vram_check.txt
  [ $rc = 0 ] || echo "SKIPPED (VRAM): $1 C$3 cell (every system at that expert memory): $msg" | tee -a $OUT/skipped.txt
  return $rc
}
timeout 75m bash -c 'build_tree "$@"' _ $WORK/lc-ec "$J/llama.cpp-expert-cache-4da6337.patch" llama-server
timeout 75m bash -c 'build_tree "$@"' _ $WORK/lc-stock "" llama-server
EC_SERVER=$WORK/lc-ec/build/bin/llama-server
ls -la $EC_SERVER $STOCK_SERVER
echo "builds done, waiting for the downloads ($(( $(date +%s) - T0 )) s since launch)"
wait $DLG $DLH $DLQ $DLM $VEP
cat $OUT/dl_gguf.txt $OUT/dl_hf.txt $OUT/dl_qwen3.txt $OUT/dl_mixtral.txt; echo "venv install rc: $(tail -1 $OUT/venv_install.txt)"
F=$M/gpt-oss-120b-MXFP4.gguf
[ -f "$HFD/config.json" ] && echo "HF checkpoint ok" || echo "HF checkpoint MISSING"
[ -s "$F" ] && echo "GGUF ok $(stat -c %s $F) bytes" || echo "GGUF MISSING"
[ -s "$GM" ] && echo "Mixtral GGUF ok $(stat -c %s $GM) bytes" || echo "Mixtral GGUF MISSING"
# --- the host (idle): platform, FreeToken's calibration, our probe, the law's tables
timeout 25m bash -c platform
( cd $FT && R 25m $FTBIN bench bw ) > $OUT/ft_bench_bw.txt 2>&1; cp ~/.cache/freetoken/benchbw/*.json $OUT/ 2>/dev/null
H=$(( CORES > 4 ? CORES - 2 : 2 ))
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
grep -q "concurrent t=" $OUT/concur.txt || { echo "concur incomplete, retrying"; R 15m $WORK/concur > $OUT/concur.txt 2>&1; }
LAWG=$(python3 $J/fetch_table.py $OUT/concur.txt $H 13253760 4 $OUT/fetch_table_law_gptoss.json 37)
LAWQ=$(python3 $J/fetch_table.py $OUT/concur.txt $H 9437184 8 $OUT/fetch_table_law_qwen3.json 48)
echo "law tables ($(date -u +%FT%TZ)): gpt-oss $LAWG (headline machine 0,0,1,1,2); qwen3 $LAWQ (headline machine 0,0,1,1,2,3,3,4,5)" | tee $OUT/tables.txt
[ -n "$LAWG" ] && [ -n "$LAWQ" ] || { echo "no law table"; exit 3; }
# Mixtral's table: S from the GGUF header (mean of the per-layer bytes of one expert), fallback data/gguf_bytes.json's value
SMIX=$(python3 $WORK/vram_check.py "$GM" mean 2>/dev/null) || SMIX=""; SSRC="GGUF header on this machine"
[[ "$SMIX" =~ ^[0-9]+$ ]] || { SMIX=106659840; SSRC="fallback, data/gguf_bytes.json (header read failed)"; }
LAWM=$(python3 $J/fetch_table.py $OUT/concur.txt $H $SMIX 2 $OUT/fetch_table_law_mixtral.json 230)
echo "law table mixtral ($(date -u +%FT%TZ)): $LAWM (S $SMIX bytes, the mean over 32 layers, $SSRC; k 2; g 230 us estimated, see header)" | tee -a $OUT/tables.txt
[ -n "$LAWM" ] || echo "no Mixtral law table" | tee -a $OUT/skipped.txt
# the law's prediction for llama.cpp at the matched placements, written before any model run (as jobs 082/085/089; the GPU
# terms G are the 5090's, 5.063 / 4.3 ms, so the prediction is a check of the host term only; no Mixtral G is known)
python3 - $OUT/concur.txt $CORES > $OUT/law_llama.json <<'PY'
import json, re, statistics, sys, time
txt = open(sys.argv[1]).read(); t = int(sys.argv[2])
by = {}
for k, v in re.findall(r"cpu_read_gbs t=(\d+) ([\d.]+)", txt): by.setdefault(int(k), []).append(float(v))
pts = sorted((k, statistics.mean(v)) for k, v in by.items())
bc = pts[-1][1] if t >= pts[-1][0] else pts[0][1] if t <= pts[0][0] else next(v0 + (v1 - v0) * (t - t0) / (t1 - t0) for (t0, v0), (t1, v1) in zip(pts, pts[1:]) if t0 <= t <= t1)
pred = {"gptoss_llama_n32": round(1e3 / (32 * 4 * 13253760 / (bc * 1e9) * 1e3 + 5.063), 2)}
pred.update({f"qwen3_llama_n{n}": round(1e3 / (n * 8 * 9437184 / (bc * 1e9) * 1e3 + 4.3), 2) for n in (42, 36, 27)})
print(json.dumps(dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), B_c=round(bc, 2), threads=t, G_ms_from_5090=True, predicted_tok_s=pred)))
PY
cat $OUT/law_llama.json
LXB='{"min_p": 0.0, "seed": 1234, "backend_sampling": true}'
OT="-ot '\\.ffn_(up|down|gate|gate_up)_exps=CUDA_Host'"
ECENV="env LLAMA_EC_POLICY=dfa LLAMA_EC_KAPPA=1 LLAMA_EC_MAILBOX=1 LLAMA_EC_TDEC=1 LLAMA_EC_HELPERS=$H"
waitfree() {
  for i in $(seq 120); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-0}" -lt 1500 ] && return; sleep 2; done
  echo "GPU memory still in use: $u MiB"; killsrv
}
cl() {  # model-dir label launch meta [client args...]
  local hf=$1 lab=$2 L=$3 meta=$4; shift 4
  waitfree
  local t0=$(date +%s)
  R 60m $PY $J/bs1_client.py --model $hf --problems $(seq -s, 0 29) --decode 256 --greedy --warmup once --warmup-prompt $WORK/warmup.txt \
    --launch $L --label $lab --meta "$meta" --json $OUT/bs1.jsonl --log $OUT/srv_${lab}_L$L.log "$@"
  local rc=$?
  echo "$lab launch $L rc=$rc"; echo "$lab $L $rc $(( $(date +%s) - t0 ))" >> $OUT/runs.txt
  [ $rc = 124 ] && killsrv
}
declare -A FIT
# ---- gpt-oss-120b: C 14 only (11%), both launches
LSG="-m $F -ngl 99 -fa on -lm none -t $CORES -np 1 -c 8448 --no-webui"
oursg() {  # C launch
  cl $HFD g_ours_C$1_law $2 "{\"model\": \"gpt-oss-120b\", \"system\": \"ours v2\", \"C\": $1, \"table\": \"$LAWG\", \"policy\": \"dfa\"}" --extra "$LXB" \
    --cmd "$ECENV LLAMA_EC_SLOTS=$1 LLAMA_EC_FETCH=$LAWG LLAMA_EC_STATS=$OUT/srv_g_ours_C$1_law_L$2.json $EC_SERVER $LSG $OT --port {port}"
}
ftg() { ( cd $FT; cl $HFD g_ft_$1_r$2 $3 "{\"model\": \"gpt-oss-120b\", \"system\": \"FreeToken $1\", \"rate\": $2}" --ft $1 --cache-rate $2 ); }
llamag() { cl $HFD g_llama_n$1 1 "{\"model\": \"gpt-oss-120b\", \"system\": \"llama.cpp\", \"ncmoe\": $1}" --extra "$LXB" --cmd "$STOCK_SERVER $LSG -ncmoe $1 --port {port}"; }
CELLS_G="14:0.111:hybrid:32"
if [ -s "$F" ]; then
  for cell in $CELLS_G; do IFS=: read C r b n <<< "$cell"; FIT[g$C]=0; vcheck gpt-oss $F $C 8448 && FIT[g$C]=1; done
  for cell in $CELLS_G; do IFS=: read C r b n <<< "$cell"; [ "${FIT[g$C]}" = 1 ] || continue; oursg $C 1; ftg $b $r 1; llamag $n; done
  for cell in $CELLS_G; do IFS=: read C r b n <<< "$cell"; [ "${FIT[g$C]}" = 1 ] || continue; ftg $b $r 2; oursg $C 2; done
else
  echo "gpt-oss GGUF missing: gpt-oss part skipped" | tee -a $OUT/skipped.txt
fi
# ---- Qwen3-30B-A3B BF16 (gpt-oss files removed first for disk; the conversion runs with no server up)
rm -rf "$HFD" "$F"; df -h $WORK | tail -1
GQ=$M/Qwen3-30B-A3B-BF16.gguf
( t0=$(date +%s); PYTHONPATH=$WORK/lc-stock/gguf-py timeout 30m $VE/bin/python $WORK/lc-stock/convert_hf_to_gguf.py "$HFQ" --outtype bf16 --outfile $GQ
  echo "convert rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $GQ ) > $OUT/convert_qwen3.txt 2>&1
tail -2 $OUT/convert_qwen3.txt
LSQ="-m $GQ -ngl 99 -fa on -lm none -t $CORES -np 1 -c 2048 --no-webui --jinja"
oursq() {  # C launch
  cl $HFQ q_ours_C$1_law $2 "{\"model\": \"Qwen3-30B-A3B\", \"system\": \"ours v2\", \"C\": $1, \"table\": \"$LAWQ\", \"policy\": \"dfa\"}" --extra "$LXB" \
    --cmd "$ECENV LLAMA_EC_SLOTS=$1 LLAMA_EC_FETCH=$LAWQ LLAMA_EC_STATS=$OUT/srv_q_ours_C$1_law_L$2.json $EC_SERVER $LSQ $OT --port {port}"
}
ftq() {  # backend rate launch mem-ratio (as jobs 080-089)
  ( cd $FT; cl $HFQ q_ft_$1_r$2 $3 "{\"model\": \"Qwen3-30B-A3B\", \"system\": \"FreeToken $1\", \"rate\": $2}" --ft $1 --cache-rate $2 --mem-ratio $4 )
}
llamaq() { cl $HFQ q_llama_n$1 1 "{\"model\": \"Qwen3-30B-A3B\", \"system\": \"llama.cpp\", \"ncmoe\": $1}" --extra "$LXB" --cmd "$STOCK_SERVER $LSQ -ncmoe $1 --port {port}"; }
CELLS_Q="16:0.125:hybrid:42:0.9 32:0.25:hybrid:36:0.9 56:0.4375:offload:27:0.95"
if [ -s "$GQ" ]; then
  for cell in $CELLS_Q; do IFS=: read C r b n mr <<< "$cell"; FIT[q$C]=0; vcheck qwen3 $GQ $C 2048 && FIT[q$C]=1; done
  for cell in $CELLS_Q; do IFS=: read C r b n mr <<< "$cell"; [ "${FIT[q$C]}" = 1 ] || continue; oursq $C 1; ftq $b $r 1 $mr; llamaq $n; done
  for cell in $CELLS_Q; do IFS=: read C r b n mr <<< "$cell"; [ "${FIT[q$C]}" = 1 ] || continue; ftq $b $r 2 $mr; oursq $C 2; done
else
  echo "Qwen3 GGUF missing: Qwen3 part skipped" | tee -a $OUT/skipped.txt
fi
# ---- Mixtral-8x7B-Instruct-v0.1 Q4_K_M: ours (law's table) and stock llama.cpp --n-cpu-moe at matched expert count, launch 1,
# ours first. The GGUF's own chat template; --model is the client's sampling source only (greedy here), so a bare directory
HFM=$M/Mixtral-8x7B-Instruct-v0.1; mkdir -p $HFM
LSM="-m $GM -ngl 99 -fa on -lm none -t $CORES -np 1 -c 8448 --no-webui"
oursm() {  # C launch
  cl $HFM m_ours_C$1_law $2 "{\"model\": \"Mixtral-8x7B-Instruct-v0.1-Q4_K_M\", \"system\": \"ours v2\", \"C\": $1, \"table\": \"$LAWM\", \"policy\": \"dfa\", \"expert_bytes\": $SMIX}" --extra "$LXB" \
    --cmd "$ECENV LLAMA_EC_SLOTS=$1 LLAMA_EC_FETCH=$LAWM LLAMA_EC_STATS=$OUT/srv_m_ours_C$1_law_L$2.json $EC_SERVER $LSM $OT --port {port}"
}
llamam() { cl $HFM m_llama_n$1 1 "{\"model\": \"Mixtral-8x7B-Instruct-v0.1-Q4_K_M\", \"system\": \"llama.cpp\", \"ncmoe\": $1}" --extra "$LXB" --cmd "$STOCK_SERVER $LSM -ncmoe $1 --port {port}"; }
CELLS_M="2:24 4:16"
if [ -s "$GM" ] && [ -n "$LAWM" ]; then
  for cell in $CELLS_M; do IFS=: read C n <<< "$cell"; FIT[m$C]=0; vcheck mixtral $GM $C 8448 && FIT[m$C]=1; done
  for cell in $CELLS_M; do IFS=: read C n <<< "$cell"; [ "${FIT[m$C]}" = 1 ] || continue; oursm $C 1; llamam $n; done
else
  echo "Mixtral GGUF or table missing: Mixtral part skipped" | tee -a $OUT/skipped.txt
fi
cd $WORK
for f in $OUT/srv_*.log; do tail -n 40 "$f" > "$f.tail"; done; find $OUT -name 'srv_*.log' -delete
$PY - "$OUT" "$LAWG" "$LAWQ" "${LAWM:-none}" "$DR" <<'PY' | tee $OUT/summary.txt
import json, sys, os, random, statistics as st
out, lawg, lawq, lawm, dr = sys.argv[1:6]
rows = [json.loads(l) for l in open(f"{out}/bs1.jsonl")] if os.path.exists(f"{out}/bs1.jsonl") else []
by = {}
for r in rows: by.setdefault(r["label"], []).append(r)
for lab, v in by.items():
    per = {}
    for r in v: per.setdefault(r["launch"], []).append(r["decode_tok_s"])
    print(f"{lab:24s} " + " ".join(f"L{L}: {st.mean(x):6.2f} (n={len(x)})" for L, x in sorted(per.items())) + f"  vram {sorted({round(r.get('vram_used_gib') or 0, 1) for r in v})}")
for f in sorted(os.listdir(out)):
    if f.endswith(".json") and f.startswith("srv_"):
        s = json.load(open(f"{out}/{f}")); n = max(1, s.get("steps", 1))
        print(f"{f:34s} table {s.get('fetch_table')} hit {s.get('hit_rate')} misses/token {s.get('misses', 0) / n:.1f} fetches {s.get('fetches', 0) / n:.1f}")
def speeds(lab, L):
    return {r["problem"]: r["decode_tok_s"] for r in by.get(lab, []) if r["launch"] == L}
def mean(lab, L):
    v = list(speeds(lab, L).values()); return st.mean(v) if v else None
def ratio(a, La, b, Lb, n=2000):
    """ratio of mean speeds, paired by problem, with a 95% bootstrap interval (2,000 resamples of the problems, seed 0)"""
    pa, pb = speeds(a, La), speeds(b, Lb)
    ps = sorted(p for p in pa if p in pb and pb[p] > 0)
    if not ps: return None
    x, y = [pa[p] for p in ps], [pb[p] for p in ps]
    rng = random.Random(0); bs = []
    for _ in range(n):
        idx = [rng.randrange(len(ps)) for _ in ps]
        bs.append(sum(x[i] for i in idx) / sum(y[i] for i in idx))
    bs.sort()
    return st.mean(x) / st.mean(y), bs[int(0.025 * n)], bs[int(0.975 * n) - 1], len(ps)
fmt = lambda r: "None" if r is None else f"{r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}] (n={r[3]})"
f1 = lambda x: None if x is None else round(x, 1)
law = json.load(open(f"{out}/law_llama.json"))["predicted_tok_s"] if os.path.exists(f"{out}/law_llama.json") else {}
cells = [("g", 14, "0.111", "hybrid", 32, 1.294, 69.9),
         ("q", 16, "0.125", "hybrid", 42, 1.032, 40.0), ("q", 32, "0.25", "hybrid", 36, 1.153, 63.1), ("q", 56, "0.4375", "offload", 27, 1.049, 108.2)]
l_ft, l_ll, diffs = [], [], []
for p, C, r, b, n, t1, v5090 in cells:
    o, ft, ll = f"{p}_ours_C{C}_law", f"{p}_ft_{b}_r{r}", f"{p}_llama_n{n}"
    if not (by.get(o) or by.get(ft) or by.get(ll)):
        print(f"{p} C{C}: no runs (cell skipped)"); l_ft.append(f"{p}{C} -"); l_ll.append(f"{p}{C} -"); continue
    r2, r1, rl = ratio(o, 2, ft, 2), ratio(o, 1, ft, 1), ratio(o, 1, ll, 1)
    o2, ft2, l1 = mean(o, 2), mean(ft, 2), mean(ll, 1)
    od = 100 * (r1[0] / r2[0] - 1) if r1 and r2 else None
    if od is not None: diffs.append(abs(od))
    print(f"{p} C{C}: ours L2 {f1(o2)} tok/s (Table 1's 5090: {v5090}) | FT {b} L2 {f1(ft2)} | llama n{n} L1 {f1(l1)} "
          f"(law {law.get(('gptoss' if p == 'g' else 'qwen3') + f'_llama_n{n}')}) | ours/FT L2 {fmt(r2)} (Table 1 {t1}) L1 {fmt(r1)} "
          f"order diff {None if od is None else round(od, 1)}% | ours/llama L1 {fmt(rl)}")
    l_ft.append(f"{p}{C} " + ("-" if r2 is None else f"{r2[0]:.3f} [{r2[1]:.3f}, {r2[2]:.3f}]"))
    l_ll.append(f"{p}{C} " + ("-" if rl is None else f"{rl[0]:.2f}"))
l_m = []
for C, n in ((2, 24), (4, 16)):
    o, ll = f"m_ours_C{C}_law", f"m_llama_n{n}"
    if not (by.get(o) or by.get(ll)):
        print(f"m C{C}: no runs (cell skipped)"); l_m.append(f"m{C} -"); continue
    rl = ratio(o, 1, ll, 1)
    print(f"m C{C}: ours L1 {f1(mean(o, 1))} tok/s | llama n{n} L1 {f1(mean(ll, 1))} | ours/llama L1 {fmt(rl)} (prediction >= 1.3)")
    l_m.append(f"m{C} " + ("-" if rl is None else f"{rl[0]:.3f} [{rl[1]:.3f}, {rl[2]:.3f}]"))
open(f"{out}/oneline.txt", "w").write("092 ours/FT L2: " + " ".join(l_ft) + " | ours/llama L1: " + " ".join(l_ll) + " | mixtral ours/llama L1: " + " ".join(l_m)
    + f" | order diff max {round(max(diffs), 1) if diffs else None}% | tables g {lawg} q {lawq} m {lawm} | device read {dr} GB/s")
PY
ONE=$(cat $OUT/oneline.txt 2>/dev/null || echo "092: no summary")
python3 $J/manifest.py "$OUT" 092_grid_3090@vast "$T0" "$ONE" law_gptoss=$LAWG law_qwen3=$LAWQ law_mixtral=${LAWM:-none} mixtral_expert_bytes=$SMIX device_read_gbs=$DR memory_clock_max_mhz=$MEMCLK sm_clock_max_mhz=$SMCLK gpu="$GPU" vram_mib=$VRAM
echo "SUMMARY $ONE"
