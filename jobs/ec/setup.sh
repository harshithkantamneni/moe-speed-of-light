# Idempotent environment for the expert-cache jobs (source it). Locks clocks, builds llama.cpp
# 2145525a + jobs/ec/llama.cpp-expert-cache.patch into $WORK/llama.cpp-ec (rebuilt when the patch
# changes), downloads the four GGUFs if missing. Exports EC_BIN, EC_TESTS, M, J.
J=${J:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
export J M=$WORK/models
export GGML_NO_BACKTRACE=1   # a failed assert must exit, not attach gdb to a process holding the GPU (the likely 010 hang)
MAXSM=$(timeout 10 nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
MAXMEM=$(timeout 10 nvidia-smi --query-gpu=clocks.max.mem --format=csv,noheader,nounits | head -1)
sudo timeout 20 nvidia-smi -pm 1 >/dev/null; sudo timeout 20 nvidia-smi -lgc $MAXSM,$MAXSM >/dev/null; sudo timeout 20 nvidia-smi -lmc $MAXMEM,$MAXMEM >/dev/null
PATCH_SHA=$(sha256sum "$J/llama.cpp-expert-cache.patch" | cut -c1-16)
if [ "$(cat $WORK/llama.cpp-ec/.patch_sha 2>/dev/null)" != "$PATCH_SHA" ]; then
  ( cd $WORK && { [ -d llama.cpp ] || git clone -q https://github.com/ggml-org/llama.cpp; } && cd llama.cpp && git fetch -q origin 2145525a4081d66ff1a87cf43ef809f95a85ac0c
    rm -rf $WORK/llama.cpp-ec; git worktree prune; git worktree add -f $WORK/llama.cpp-ec 2145525a4081d66ff1a87cf43ef809f95a85ac0c
    cd $WORK/llama.cpp-ec && git apply "$J/llama.cpp-expert-cache.patch" || exit 1
    SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$SM -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON > $WORK/ec_cmake.txt 2>&1
    cmake --build build -j$(nproc) --target llama-ec-bench llama-ec-cpubench test-backend-ops > $WORK/ec_build.txt 2>&1 && echo "$PATCH_SHA" > .patch_sha )
  echo "ec build: $(cat $WORK/llama.cpp-ec/.patch_sha 2>/dev/null || echo FAILED)"; grep -E "error:" $WORK/ec_build.txt | head
fi
export EC_BIN=$WORK/llama.cpp-ec/build/bin/llama-ec-bench EC_TESTS=$WORK/llama.cpp-ec/build/bin/test-backend-ops EC_CPUBENCH=$WORK/llama.cpp-ec/build/bin/llama-ec-cpubench
if [ ! -f $M/.done ]; then
  pip install -q -U "huggingface_hub[hf_transfer]" > /dev/null 2>&1; export HF_HUB_ENABLE_HF_TRANSFER=1
  mkdir -p $M && ( cd $M
  for spec in "ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-MXFP4.gguf" "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf" \
              "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf" "ggml-org/gpt-oss-120b-GGUF gpt-oss-120b-MXFP4.gguf"; do
    set -- $spec; [ -f "$2" ] || timeout 20m hf download "$1" "$2" --local-dir . > /dev/null 2>&1
  done; ls *.gguf | wc -l | grep -q 4 && touch .done )
fi
ls -la $M/*.gguf | awk '{print $5, $9}'
