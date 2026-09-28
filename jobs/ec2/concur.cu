// Host memory bandwidth as the expert cache sees it: CPU threads streaming weights, PCIe host-to-device transfers
// (copy engine, and a kernel reading mapped pinned memory, "zero-copy"), and both at once on the same DRAM; plus the
// critical-path latency of fetching one to four expert-sized blocks. Every buffer rotates through 2 GiB so that no
// level of cache holds it. Output: one "key value" line per measurement.
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

__global__ void zc_copy(const int4 * __restrict__ src, int4 * __restrict__ dst, size_t n) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) dst[i] = src[i];
}

// n_threads CPU readers over [buf, buf+bytes), each on its own share, for `dur` seconds; returns GB/s
static double cpu_read(const uint8_t * buf, size_t bytes, int nt, double dur, std::atomic<bool> * go = nullptr) {
    std::vector<std::thread> th; std::vector<double> gb(nt, 0); std::atomic<bool> stop{false};
    const size_t share = bytes / nt / 64 * 64;
    for (int t = 0; t < nt; ++t) th.emplace_back([&, t] {
        const uint64_t * p = (const uint64_t *) (buf + t * share); const size_t n = share / 8; uint64_t s = 0; double done = 0;
        if (go) while (!go->load()) {}
        while (!stop.load(std::memory_order_relaxed)) {
            for (size_t i = 0; i + 8 <= n; i += 8) s += p[i] ^ p[i + 1] ^ p[i + 2] ^ p[i + 3] ^ p[i + 4] ^ p[i + 5] ^ p[i + 6] ^ p[i + 7];
            done += share;
        }
        gb[t] = done; if (s == 42) printf(" ");
    });
    auto t0 = clk::now(); if (go) go->store(true);
    std::this_thread::sleep_for(std::chrono::duration<double>(dur)); stop = true;
    for (auto & x : th) x.join();
    double tot = 0; for (double g : gb) tot += g;
    return tot / secs(t0, clk::now()) / 1e9;
}

// PCIe host->device for `dur` seconds, chunk bytes per transfer; mode 0 copy engine, 1 zero-copy kernel; GB/s
static double pcie(uint8_t * hbuf, uint8_t * dhost, uint8_t * dbuf, size_t bytes, size_t chunk, int mode, int blocks, double dur, cudaStream_t st) {
    size_t off = 0, moved = 0; auto t0 = clk::now();
    while (secs(t0, clk::now()) < dur) {
        for (int k = 0; k < 8; ++k) {
            if (off + chunk > bytes) off = 0;
            if (mode == 0) CK(cudaMemcpyAsync(dbuf + (off % (256u << 20)), hbuf + off, chunk, cudaMemcpyHostToDevice, st));
            else zc_copy<<<blocks, 256, 0, st>>>((const int4 *) (dhost + off), (int4 *) (dbuf + (off % (256u << 20))), chunk / 16);
            off += chunk; moved += chunk;
        }
        CK(cudaStreamSynchronize(st));
    }
    return moved / secs(t0, clk::now()) / 1e9;
}

int main(int argc, char ** argv) {
    const int ncpu = (int) std::thread::hardware_concurrency();
    std::vector<int> tcs = { 1, 4, 8, 12, 16, ncpu / 2, ncpu };
    const size_t bytes = (size_t) 2 << 30;
    uint8_t * cbuf = (uint8_t *) aligned_alloc(4096, bytes); memset(cbuf, 1, bytes);
    uint8_t * hbuf; CK(cudaHostAlloc(&hbuf, bytes, cudaHostAllocMapped)); memset(hbuf, 2, bytes);
    uint8_t * dhost; CK(cudaHostGetDevicePointer((void **) &dhost, hbuf, 0));
    uint8_t * dbuf; CK(cudaMalloc(&dbuf, 256u << 20));
    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    for (int t : tcs) if (t >= 1 && t <= ncpu) printf("cpu_read_gbs t=%d %.1f\n", t, cpu_read(cbuf, bytes, t, 1.5));
    for (size_t ch : { (size_t) 2 << 20, (size_t) 16 << 20 }) {
        printf("pcie_copyengine_gbs chunk=%zuMB %.1f\n", ch >> 20, pcie(hbuf, dhost, dbuf, bytes, ch, 0, 0, 1.5, st));
        for (int b : { 16, 64, 256 }) printf("pcie_zerocopy_gbs chunk=%zuMB blocks=%d %.1f\n", ch >> 20, b, pcie(hbuf, dhost, dbuf, bytes, ch, 1, b, 1.5, st));
    }
    // both at once: CPU readers on cbuf while PCIe streams hbuf (different DRAM pages, same memory controllers)
    for (int t : { 8, 16, ncpu / 2 }) for (int mode : { 0, 1 }) {
        std::atomic<bool> go{false}; double cg = 0, pg = 0;
        std::thread pt([&] { while (!go.load()) {} pg = pcie(hbuf, dhost, dbuf, bytes, 16u << 20, mode, 64, 1.5, st); });
        cg = cpu_read(cbuf, bytes, t, 1.5, &go); pt.join();
        printf("concurrent t=%d pcie=%s cpu_gbs %.1f pcie_gbs %.1f sum %.1f\n", t, mode ? "zerocopy64" : "copyengine", cg, pg, cg + pg);
    }
    // critical-path fetch latency: k blocks of s bytes, from a rotating host offset, measured with events
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (size_t s : { (size_t) 2860000, (size_t) 13250000 }) for (int k : { 1, 2, 4 }) for (int mode : { 0, 1, 2 }) {
        const size_t s16 = s / 16 * 16; double tot = 0; const int R = 40; size_t off = 0;
        for (int r = 0; r < R + 5; ++r) {
            CK(cudaEventRecord(e0, st));
            for (int j = 0; j < k; ++j) {
                if (off + s16 > bytes) off = 0;
                if (mode == 0) CK(cudaMemcpyAsync(dbuf + j * s16, hbuf + off, s16, cudaMemcpyHostToDevice, st));
                else zc_copy<<<mode == 1 ? 16 : 64, 256, 0, st>>>((const int4 *) (dhost + off), (int4 *) (dbuf + j * s16), s16 / 16);
                off += 64u << 20;
            }
            CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); if (r >= 5) tot += ms;
        }
        printf("fetch_us size=%.2fMB k=%d mode=%s %.1f\n", s / 1e6, k, mode == 0 ? "copyengine" : mode == 1 ? "zc16" : "zc64", 1000 * tot / R);
    }
    return 0;
}
