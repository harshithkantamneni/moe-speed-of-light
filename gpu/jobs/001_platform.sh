#!/bin/bash
# Platform characterization: hardware inventory, clock locking, host DRAM (STREAM),
# host<->device copy bandwidth, device memory bandwidth, Nsight Compute availability.
set -x
exec 2>&1
nvidia-smi -q > "$OUT/nvidia-smi-q.txt"
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current,driver_version --format=csv | tee "$OUT/gpu.csv"
lscpu | tee "$OUT/lscpu.txt"
free -g; nproc
(sudo apt-get install -y -qq numactl hwloc >/dev/null 2>&1; numactl -H; lstopo-no-graphics --of console | head -60) > "$OUT/topology.txt" 2>&1
sudo dmidecode -t memory > "$OUT/dmidecode-memory.txt" 2>&1 || true
cat /etc/os-release | head -3; which nvcc ncu nsys; nvcc --version | tail -2; ncu --version | tail -1
# clock locking (record what the platform allows)
MAXSM=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
MAXMEM=$(nvidia-smi --query-gpu=clocks.max.mem --format=csv,noheader,nounits | head -1)
{ sudo nvidia-smi -pm 1; sudo nvidia-smi -lgc $MAXSM,$MAXSM; sudo nvidia-smi -lmc $MAXMEM,$MAXMEM; } > "$OUT/clock-lock.txt" 2>&1
nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv >> "$OUT/clock-lock.txt"

# STREAM (host DRAM bandwidth) across thread counts
mkdir -p $WORK/bench && cd $WORK/bench
curl -fsSL https://raw.githubusercontent.com/jeffhammond/STREAM/master/stream.c -o stream.c
gcc -O3 -march=native -fopenmp -DSTREAM_ARRAY_SIZE=400000000 -DNTIMES=10 -mcmodel=medium stream.c -o stream
for t in 1 4 8 $(( $(nproc)/2 )) $(nproc); do
  echo "== OMP_NUM_THREADS=$t"; OMP_NUM_THREADS=$t OMP_PROC_BIND=spread ./stream | grep -E "Copy|Scale|Add|Triad"
done | tee "$OUT/stream.txt"

# host<->device and device bandwidth microbenchmark
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
  // small launches: latency floor
  cudaEventRecord(e0); for(int r=0;r<1000;r++) rd<<<1,32>>>(a,o,32); cudaEventRecord(e1); cudaEventSynchronize(e1);
  cudaEventElapsedTime(&ms,e0,e1); printf("tiny kernel launch: %6.2f us each\n", ms);
  return 0; }
CU
nvcc -O3 -arch=native bw.cu -o bw && ./bw | tee "$OUT/bw.txt"
# Nsight Compute counter access check
sudo ncu --metrics dram__bytes_read.sum,gpu__time_duration.sum --kernel-name rd --launch-count 1 ./bw > "$OUT/ncu-check.txt" 2>&1; tail -15 "$OUT/ncu-check.txt"
