#!/bin/bash
# Job 116: shadow run-ahead, stage A (an offline go/no-go), on Qwen3-30B-A3B BF16.
# The paper finds that policies without foresight recover little, and that forecasters trained on routing histories
# (ridge, GRU) close at most about 15% of MIN's gain on gpt-oss: past routing does not carry the future. A forecaster
# has to read the model's own state. Shadow run-ahead does: in a host-bound step the GPU mostly waits on host reads, so
# it could run the model itself a few tokens ahead, executing only the experts its cache already holds, and read the
# routers' choices as the forecast. It needs no extra weights and no host reads. Its risk is that at 1/8 of the experts
# resident much of each token's computation is approximated, and the errors compound over 48 layers. This job measures
# that offline, before any engine work.
# Method (jobs/ec2/shadow_runahead.py, tested on a small random Qwen3-MoE on CPU, where the patched MoE forward equals
# transformers' own to 5e-8 and cropping the shadow extension restores the real cache):
#   - Hugging Face transformers 5.17.0, Qwen/Qwen3-30B-A3B in BF16 on one GPU with at least 79 GB.
#   - Job 084b's AIME-25 own text (jobs/ec2/aime25_owntext_qwen3_084b.jsonl), teacher-forced: the prompt prefilled,
#     then 256 decode steps per problem, step g's input ids[P+g]; problems in order, as many as start within 150 minutes.
#   - After real step g, an online cache of C = 16 slots per layer (scripts/value_map.py's online policy: always admit,
#     evict the lowest decayed score) takes step g's routing. The model then runs H = 4 tokens ahead on a temporary
#     extension of the KV cache, each MoE layer executing only the top-k among that cache's residents (weights
#     renormalised), and logs each router's unmasked top-11; the extension is then cropped. Shadow inputs: "self" (the
#     model's own greedy choices, the realistic forecast) and "orac" (the true next inputs, to separate token errors
#     from the resident-only approximation).
# Scoring (jobs/ec2/shadow_score.py, with jobs/ec2/vmap.py = main's scripts/value_map.py at 9a93c14): precision@k
# (share of the true top-8 of step g+h among the forecast's first 8) and recall@11 per horizon h = 1..4; and the share
# of MIN's gain over admitting every miss that the paper's window policy closes when driven by the forecast's first 8
# per step (Section 8's yardstick; online = always-admit, kappa = -1e9), for W = 1, 2, 4, with the exact window,
# random-error windows and persistence on the same routing for reference.
# Decision rule (registered here, before launch): GO for stage B if, at C = 16, the self forecast's share at W = 4 is
#   at least 1/3 AND its precision@k averaged over h = 1..4 is at least 0.5. The bar comes from the value map on job
#   084b's routing (prereg/value_map.json): random-error windows of precision = recall = 0.5 at W = 4 close 0.34 of MIN's
#   gain at Qwen3 C = 16. NO-GO otherwise.
# Predictions:
#   P1. the self forecast's precision@k falls with the horizon: lower at h = 4 than at h = 1;
#   P2. the orac forecast's precision@k is at least the self forecast's at every h from 2 to 4;
#   P3. the decision is NO-GO (our prior for GO is about 0.25).
# Validity conditions (reported apart from the predictions): V0 the GPU has at least 79 GB (else the job stops); V1 the
# real routing's top-8 overlaps job 084b's llama.cpp trace by at least 0.90 on average; V2 the model's greedy choice
# after step g equals ids[P+g+1] on at least 0.90 of the steps; V3 at least 6 problems complete. If V1, V2 or V3 fails,
# the results are reported as exploratory.
# Host: the cheapest rentable on-demand offer of one whole GPU with at least 79 GB (A100 80 GB, H100, RTX PRO 6000),
# at most $1.60/h, at least 150 GB disk, at least 300 Mb/s down, CUDA at least 12.8, not in the rental ledger, from the
# listing committed with this header (results/116_offers_at_registration.txt): 55163302 (RTX PRO 6000 WS, Core Ultra 7
# 265K). One machine. A machine that fails V0 or cannot install or download is replaced once by the cheapest
# qualifying offer in a new listing committed before that launch. The account holds about $6.60.
set -x
exec 2>&1
J=/w/jobs/ec2
nvidia-smi --query-gpu=name,memory.total,uuid,driver_version --format=csv | tee $OUT/gpu.csv
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
[ "${MEM:-0}" -ge 79000 ] || { echo "V0 FAIL: GPU memory ${MEM} MiB below 79000, stopping" | tee $OUT/v0.txt; exit 7; }
echo "V0: ${MEM} MiB" | tee $OUT/v0.txt
free -g > $OUT/free.txt; df -h $WORK > $OUT/df.txt; nproc > $OUT/nproc.txt; lscpu > $OUT/lscpu.txt 2>&1
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq > $OUT/apt.txt 2>&1
apt-get install -y -qq python3 python3-venv python3-pip >> $OUT/apt.txt 2>&1
python3 -m pip install -q --break-system-packages uv > $OUT/pip_uv.txt 2>&1
VE=$WORK/venv
uv venv -q $VE && . $VE/bin/activate
( uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 &&
  uv pip install -q "transformers==5.17.0" accelerate numpy numba safetensors huggingface_hub hf_xet ) > $OUT/venv.txt 2>&1
echo "venv rc=$?"
python -c "import torch, transformers, numba; print(torch.__version__, torch.version.cuda, transformers.__version__, numba.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))" | tee $OUT/venv_ok.txt
[ -s $OUT/venv_ok.txt ] || { echo "install failed"; exit 3; }
M=$WORK/models/Qwen3-30B-A3B
t0=$(date +%s)
python -c "import sys; from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-30B-A3B', local_dir=sys.argv[1])" "$M" > $OUT/dl.txt 2>&1
echo "download rc=$? in $(( $(date +%s) - t0 )) s; $(du -sh $M)" | tee -a $OUT/dl.txt
sha256sum $J/shadow_runahead.py $J/shadow_score.py $J/vmap.py $J/aime25_owntext_qwen3_084b.jsonl $J/route_aime25_qwen3_084b.npz | tee $OUT/sha.txt
python $J/shadow_runahead.py --model "$M" --corpus $J/aime25_owntext_qwen3_084b.jsonl --out $WORK/run --C 16 --H 4 \
  --budget-min 150 2>&1 | grep -v -i "warn" | tee $OUT/shadow.txt
cp $WORK/run/run.json $OUT/ 2>/dev/null
python $J/shadow_score.py $WORK/run $J/route_aime25_qwen3_084b.npz 2>&1 | tee $OUT/score.txt
cp $WORK/run/score.json $OUT/ 2>/dev/null
# a subset small enough to ship, for re-checks off the machine: the first two problems, forecasts' first k
python - "$WORK/run/shadow.npz" "$OUT/shadow_small.npz" <<'PY'
import sys, numpy as np
z = np.load(sys.argv[1]); s = z["seq"]; n = int((s < 2).sum()); k = int(z["k"])
np.savez_compressed(sys.argv[2], R=z["R"][:n], seq=s[:n], inp=z["inp"][:n], match=z["match"][:n], tok_self=z["tok_self"][:n],
                    F_self=z["F_self"][:n, :, :, :k], F_orac=z["F_orac"][:n, :, :, :k], E=z["E"], C=z["C"], H=z["H"], k=k, m=k, L=z["L"])
print("subset", n, "steps")
PY
nvidia-smi --query-gpu=name,clocks.max.sm,power.limit --format=csv >> $OUT/gpu.csv
