#!/bin/bash
# Fix-up for 002/003/004: (1) build llama.cpp for the GPU's explicit SM (CMake 3.22 has no
# "native"), (2) install Nsight Compute + Nsight Systems from NVIDIA's CUDA apt repository.
set -x
exec 2>&1
export DEBIAN_FRONTEND=noninteractive
cd $WORK
if ! command -v ncu >/dev/null; then
  curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update -qq
  apt-cache search nsight | head -20
  sudo apt-get install -y -qq cuda-nsight-compute-12-8 cuda-nsight-systems-12-8 > "$OUT/apt.txt" 2>&1 || \
    sudo apt-get install -y -qq nsight-compute nsight-systems >> "$OUT/apt.txt" 2>&1
fi
for d in /usr/local/cuda-12.8/bin /usr/local/cuda/bin /opt/nvidia/nsight-compute/*/ /opt/nvidia/nsight-systems/*/bin; do
  [ -d "$d" ] && ls "$d" | grep -E "^(ncu|nsys)$" | sed "s|^|$d/|"
done | tee "$OUT/nsight_paths.txt"
NCU=$(command -v ncu || grep -m1 '/ncu$' "$OUT/nsight_paths.txt")
NSYS=$(command -v nsys || grep -m1 '/nsys$' "$OUT/nsight_paths.txt")
[ -n "$NCU" ] && sudo ln -sf "$NCU" /usr/local/bin/ncu
[ -n "$NSYS" ] && sudo ln -sf "$NSYS" /usr/local/bin/nsys
ncu --version | tail -1; nsys --version

SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
cd $WORK/llama.cpp && git checkout -q 2145525a4081d66ff1a87cf43ef809f95a85ac0c && rm -rf build
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$SM -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > "$OUT/cmake.txt" 2>&1; echo "cmake rc=$?"
cmake --build build -j$(nproc) --target llama-bench llama-cli > "$OUT/build.txt" 2>&1; echo "build rc=$?"
tail -3 "$OUT/build.txt"
./build/bin/llama-bench -m $WORK/models/gpt-oss-20b-MXFP4.gguf -ngl 99 -fa 1 -p 0 -n 16 -r 1 -o md | tee "$OUT/smoke.md"
sudo ncu --metrics dram__bytes_read.sum --launch-count 3 ./build/bin/llama-bench -m $WORK/models/gpt-oss-20b-MXFP4.gguf -ngl 99 -fa 1 -p 0 -n 1 -r 1 --no-warmup > "$OUT/ncu-check.txt" 2>&1
tail -12 "$OUT/ncu-check.txt"
