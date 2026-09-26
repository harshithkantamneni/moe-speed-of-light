#!/bin/bash
# Build llama.cpp (pinned) with CUDA and download the benchmark models.
set -x
exec 2>&1
LLAMA_COMMIT=2145525a4081d66ff1a87cf43ef809f95a85ac0c
cd $WORK
[ -d llama.cpp ] || git clone -q https://github.com/ggml-org/llama.cpp
cd llama.cpp && git fetch -q origin $LLAMA_COMMIT && git checkout -q $LLAMA_COMMIT
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > "$OUT/cmake.txt" 2>&1
cmake --build build -j$(nproc) --target llama-bench llama-cli llama-server > "$OUT/build.txt" 2>&1; echo "build rc=$?"; tail -5 "$OUT/build.txt"
./build/bin/llama-bench --help | head -5
pip install -q -U "huggingface_hub[hf_transfer]" 2>&1 | tail -1
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p $WORK/models && cd $WORK/models
for spec in "ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-MXFP4.gguf" "ggml-org/gpt-oss-120b-GGUF gpt-oss-120b-MXFP4.gguf" \
            "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf" \
            "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"; do
  set -- $spec; t0=$(date +%s); huggingface-cli download "$1" "$2" --local-dir . > /dev/null 2>&1 || hf download "$1" "$2" --local-dir . > /dev/null 2>&1
  echo "$2 $(du -h "$2" | cut -f1) in $(( $(date +%s)-t0 ))s"
done | tee "$OUT/models.txt"
sha256sum *.gguf | tee "$OUT/models.sha256"
