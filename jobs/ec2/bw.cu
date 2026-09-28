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
