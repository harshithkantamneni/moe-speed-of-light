#!/bin/bash
# Phase 2 setup on a fresh instance: record the platform, lock clocks, build llama.cpp (pinned) with the
# expert-cache patch (runtime/llama.cpp-expert-cache.patch on main), run the op tests against the CPU
# backend, and download the models.
set -x
exec 2>&1
J=$(cd "$(dirname "$0")" && pwd)/ec
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem,pcie.link.gen.max,pcie.link.width.max,ecc.mode.current,driver_version --format=csv | tee "$OUT/gpu.csv"
lscpu | grep -E "Model name|^CPU\(s\)|NUMA|L3" | tee "$OUT/lscpu.txt"; free -g | tee -a "$OUT/lscpu.txt"
MAXSM=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
MAXMEM=$(nvidia-smi --query-gpu=clocks.max.mem --format=csv,noheader,nounits | head -1)
sudo nvidia-smi -pm 1; sudo nvidia-smi -lgc $MAXSM,$MAXSM; sudo nvidia-smi -lmc $MAXMEM,$MAXMEM
# host bandwidth, same protocol as job 001 (best Triad over thread counts)
mkdir -p $WORK/bench && cd $WORK/bench
curl -fsSL https://raw.githubusercontent.com/jeffhammond/STREAM/master/stream.c -o stream.c
gcc -O3 -march=native -fopenmp -DSTREAM_ARRAY_SIZE=400000000 -DNTIMES=10 -mcmodel=medium stream.c -o stream
for t in 8 $(( $(nproc)/2 )) $(nproc); do echo "== OMP_NUM_THREADS=$t"; OMP_NUM_THREADS=$t OMP_PROC_BIND=spread ./stream | grep -E "Triad"; done | tee "$OUT/stream.txt"

# Nsight (for later jobs) - same install as job 005
export DEBIAN_FRONTEND=noninteractive
if ! command -v ncu >/dev/null; then
  curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update -qq
  sudo apt-get install -y -qq cuda-nsight-compute-12-8 cuda-nsight-systems-12-8 > "$OUT/apt.txt" 2>&1
fi

cd $WORK
[ -d llama.cpp ] || git clone -q https://github.com/ggml-org/llama.cpp
cd llama.cpp && git fetch -q origin 2145525a4081d66ff1a87cf43ef809f95a85ac0c
rm -rf $WORK/llama.cpp-ec; git worktree prune; git worktree add -f $WORK/llama.cpp-ec 2145525a4081d66ff1a87cf43ef809f95a85ac0c
cd $WORK/llama.cpp-ec && git apply "$J/llama.cpp-expert-cache.patch" && echo "patch applied: $(sha256sum "$J/llama.cpp-expert-cache.patch")"
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$SM -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON > "$OUT/cmake.txt" 2>&1
cmake --build build -j$(nproc) --target llama-ec-bench test-backend-ops llama-bench > "$OUT/build.txt" 2>&1; echo "build rc=$?"
grep -E " error|error:" "$OUT/build.txt" | head -20

# models (same files and checksums as job 002)
pip install -q -U "huggingface_hub[hf_transfer]" 2>&1 | tail -1
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p $WORK/models && cd $WORK/models
for spec in "ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-MXFP4.gguf" "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf" \
            "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf" "ggml-org/gpt-oss-120b-GGUF gpt-oss-120b-MXFP4.gguf"; do
  set -- $spec; [ -f "$2" ] || hf download "$1" "$2" --local-dir . > /dev/null 2>&1 || huggingface-cli download "$1" "$2" --local-dir . > /dev/null 2>&1
done
sha256sum *.gguf | tee "$OUT/models.sha256" &
SHA=$!

# op tests: CUDA against CPU, including negative (not-served-here) expert ids
cd $WORK/llama.cpp-ec
./build/bin/test-backend-ops -b CUDA0 -o MUL_MAT_ID_EC > "$OUT/test_mul_mat_id_ec.txt" 2>&1; echo "MUL_MAT_ID_EC rc=$?"; tail -4 "$OUT/test_mul_mat_id_ec.txt"
./build/bin/test-backend-ops -b CUDA0 -o ADD_ID_EC > "$OUT/test_add_id_ec.txt" 2>&1; echo "ADD_ID_EC rc=$?"; tail -4 "$OUT/test_add_id_ec.txt"
./build/bin/test-backend-ops -b CUDA0 -o MUL_MAT_ID > "$OUT/test_mul_mat_id.txt" 2>&1; echo "MUL_MAT_ID rc=$?"; tail -3 "$OUT/test_mul_mat_id.txt"
./build/bin/test-backend-ops -b CUDA0 -o ADD_ID > "$OUT/test_add_id.txt" 2>&1; echo "ADD_ID rc=$?"; tail -3 "$OUT/test_add_id.txt"
grep -c "FAIL" "$OUT"/test_*.txt
wait $SHA
