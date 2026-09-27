#!/bin/bash
# Audit anchors, part 1 (paper, section "Anchors"): platform microbenchmarks (as job 001), stock llama.cpp at the
# pinned commit, the anchor GGUFs, and the model's predictions for every configuration job 051 will measure
# (scripts/anchor_predict.py: frozen third-party fit + this machine's STREAM/link numbers). The runner commits this
# job's outputs before job 051 starts, so the predictions precede the measurements in git history.
set -x
exec 2>&1
A=$(cd "$(dirname "$0")" && pwd)/anchor
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current,driver_version --format=csv | tee "$OUT/gpu.csv"
lscpu | tee "$OUT/lscpu.txt"; free -g; nproc
MAXSM=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
MAXMEM=$(nvidia-smi --query-gpu=clocks.max.mem --format=csv,noheader,nounits | head -1)
{ sudo nvidia-smi -pm 1; sudo nvidia-smi -lgc $MAXSM,$MAXSM; sudo nvidia-smi -lmc $MAXMEM,$MAXMEM; } > "$OUT/clock-lock.txt" 2>&1
mkdir -p $WORK/bench && cd $WORK/bench
curl -fsSL https://raw.githubusercontent.com/jeffhammond/STREAM/master/stream.c -o stream.c
gcc -O3 -march=native -fopenmp -DSTREAM_ARRAY_SIZE=400000000 -DNTIMES=10 -mcmodel=medium stream.c -o stream
for t in 1 4 8 $(( $(nproc)/2 )) $(nproc); do
  echo "== OMP_NUM_THREADS=$t"; OMP_NUM_THREADS=$t OMP_PROC_BIND=spread ./stream | grep -E "Copy|Scale|Add|Triad"
done | tee "$OUT/stream.txt"
cat > bw.cu <<'CU'
#include <cstdio>
#include <cuda_runtime.h>
__global__ void rd(const float4* __restrict__ a, float* out, size_t n){
  float s=0; for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){float4 v=a[i]; s+=v.x+v.y+v.z+v.w;}
  if(s==-1.f) *out=s; }
int main(){
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); float ms;
  size_t sizes[]={1<<20, 4<<20, 16<<20, 64<<20, 256<<20};
  for(size_t sz: sizes){ void *h,*d; cudaMallocHost(&h,sz); cudaMalloc(&d,sz);
    for(int dir=0;dir<2;dir++){ for(int w=0;w<3;w++) dir?cudaMemcpy(h,d,sz,cudaMemcpyDeviceToHost):cudaMemcpy(d,h,sz,cudaMemcpyHostToDevice);
      int R=20; cudaEventRecord(e0); for(int r=0;r<R;r++) dir?cudaMemcpyAsync(h,d,sz,cudaMemcpyDeviceToHost):cudaMemcpyAsync(d,h,sz,cudaMemcpyHostToDevice);
      cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms,e0,e1);
      printf("%s pinned %8.1f MB  %7.2f GB/s  %7.1f us/copy\n", dir?"D2H":"H2D", sz/1e6, sz*R/(ms*1e6), ms*1e3/R); }
    cudaFreeHost(h); cudaFree(d); }
  size_t n=(1ull<<30)/16; float4* a; float* o; cudaMalloc(&a,n*16); cudaMalloc(&o,4); cudaMemset(a,0,n*16);
  int dev; cudaGetDevice(&dev); cudaDeviceProp p; cudaGetDeviceProperties(&p,dev);
  for(int w=0;w<3;w++) rd<<<p.multiProcessorCount*8,256>>>(a,o,n);
  cudaEventRecord(e0); for(int r=0;r<20;r++) rd<<<p.multiProcessorCount*8,256>>>(a,o,n); cudaEventRecord(e1); cudaEventSynchronize(e1);
  cudaEventElapsedTime(&ms,e0,e1); printf("device read 1 GiB: %7.1f GB/s\n", n*16.0*20/(ms*1e6));
  return 0; }
CU
nvcc -O3 -arch=native bw.cu -o bw && ./bw | tee "$OUT/bw.txt"
# stock llama.cpp at the pinned commit
cd $WORK; [ -d llama.cpp ] || git clone -q https://github.com/ggml-org/llama.cpp
cd llama.cpp && git fetch -q origin 2145525a4081d66ff1a87cf43ef809f95a85ac0c && git checkout -q 2145525a4081d66ff1a87cf43ef809f95a85ac0c
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)   # cmake 3.22 has no 'native'
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$CC -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > "$OUT/cmake.txt" 2>&1
cmake --build build -j$(nproc) --target llama-bench > "$OUT/build.txt" 2>&1; echo "build rc=$?"; tail -3 "$OUT/build.txt"
pip install -q -U "huggingface_hub[hf_transfer]" numpy scipy requests 2>&1 | tail -1
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p $WORK/models && cd $WORK/models
for spec in "mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf" "mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf" \
            "bartowski/Phi-3.5-MoE-instruct-GGUF Phi-3.5-MoE-instruct-Q4_K_M.gguf" "Qwen/Qwen2-57B-A14B-Instruct-GGUF qwen2-57b-a14b-instruct-q4_k_m.gguf" \
            "mradermacher/DeepSeek-V2-Lite-Chat-GGUF DeepSeek-V2-Lite-Chat.Q8_0.gguf" "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"; do
  set -- $spec; t0=$(date +%s); [ -f "$2" ] || timeout 30m hf download "$1" "$2" --local-dir . > /dev/null 2>&1
  echo "$2 $(du -h "$2" | cut -f1) in $(( $(date +%s)-t0 ))s"
done | tee "$OUT/models.txt"
# predictions, from this machine's microbenchmarks only
cd $A && python3 scripts/anchor_predict.py "$OUT" "$OUT" | tee "$OUT/predict.log"
