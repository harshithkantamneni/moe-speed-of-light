#!/bin/bash
# FreeToken (FlashML-org/FreeToken at 0d652e7, 25 Sep 2026) on the home-PC-class host (RTX 5090 + Ryzen 9 9950X):
# its own bs=1 decode benchmark (benchmarks/bench_decode_moe.py: spawns `ft serve`, streams an AIME-25 prompt with
# thinking, times 256 decode tokens) for gpt-oss-120b (HF safetensors, MXFP4 experts) at expert-cache rates equal to
# our budgets (0.111 / 0.25 / 0.444 of all experts = llama.cpp -ncmoe 32 / 27 / 20; 0.40 as a fallback if 0.444 does
# not fit), strategies offload and hybrid, after `ft bench bw` calibrates hybrid. Needs a CUDA 13 image
# (nvidia/cuda:13.0.3-devel-ubuntu24.04) and driver r580+.
set -x
exec 2>&1
export PATH=/usr/local/cuda/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > $OUT/gpu.csv; lscpu > $OUT/lscpu.txt; free -g > $OUT/free.txt; nvcc --version > $OUT/nvcc.txt
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3 python3-venv python3-pip git curl > $OUT/apt.txt 2>&1
cd $WORK && git init -q ft && cd ft && git remote add origin https://github.com/FlashML-org/FreeToken && git fetch -q --depth 1 origin 0d652e7 && git checkout -q FETCH_HEAD
python3 -m pip install -q --break-system-packages uv > $OUT/pip_uv.txt 2>&1
uv venv -q .venv && . .venv/bin/activate
( time uv pip install -q -e ".[accel]" huggingface_hub hf_transfer ) > $OUT/install.txt 2>&1; echo "install rc=$?"
ft --version | tee $OUT/ft_version.txt
M=$WORK/models; mkdir -p $M
( time hf download openai/gpt-oss-120b --local-dir $M/gpt-oss-120b --exclude "original/*" "metal/*" ) > $OUT/download.txt 2>&1; echo "download rc=$?"
du -sh $M/gpt-oss-120b | tee -a $OUT/download.txt
timeout 20m ft bench bw > $OUT/bench_bw.txt 2>&1; echo "bench bw rc=$?"; cp ~/.cache/freetoken/benchbw/*.json $OUT/ 2>/dev/null
for rate in 0.111 0.25 0.444 0.40; do
  PYTHONPATH=python:. CUDA_VISIBLE_DEVICES=0 timeout 60m python benchmarks/bench_decode_moe.py --model $M/gpt-oss-120b \
    --backend offload,hybrid --cache-rate $rate --decode 256 --json $OUT/ft_bench.jsonl > $OUT/ft_bench_r$rate.txt 2>&1
  echo "bench rate=$rate rc=$?"
done
tail -n 40 $OUT/ft_bench_r*.txt > $OUT/summary.txt 2>&1
cat $OUT/ft_bench.jsonl >> $OUT/summary.txt 2>/dev/null
