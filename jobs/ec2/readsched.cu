// Can the bound's host term be reached? Replays MIN's per-layer host reads of gpt-oss-120b (whole 13.25 MB experts;
// counts from scripts/readsched.py, one uint8 per step and layer) through CPU helper threads, which read their experts
// from ordinary host memory as the engine's helpers do, and through the PCIe copy engine, which copies the rest from
// pinned memory into GPU slots. No model runs: this measures only the host reads a schedule needs.
//   layer       per layer, the layer's n experts split whole between the CPU (c) and the link (n - c), c from the
//               probe's rates (argmin of max(c/Bc, (n-c)/Bp, n/Bcp)); the layer waits for both, as an engine must
//   layer_link  per layer, every expert over the link (c = 0), as MIN with one read in the step does
//   layer_cpu   per layer, every expert on the CPU (c = n)
//   token       the same split summed over the token, one wait per token: the bound's view (no layer barriers)
// Usage: readsched counts.bin L helpers Bc Bp Bcp [steps]     (rates in GB/s)
// Output: "readsched mode=<m> ms_per_token <x> reads_per_token <r> gbs <g>" lines.
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#if defined(__x86_64__)
#include <immintrin.h>
static inline void relax() { _mm_pause(); }
#else
static inline void relax() {}
#endif

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
using clk = std::chrono::steady_clock;
static const size_t S = 13253760;               // bytes per gpt-oss-120b expert (MXFP4)
static size_t CAP = (size_t) 4 << 30;           // each host buffer rotates through 4 GiB so no cache holds it (env READSCHED_CAP_MB)

struct Pool {
    int H;
    const uint8_t * base;
    std::atomic<uint64_t> gen{0};
    std::atomic<int> done{0};
    std::atomic<bool> quit{false};
    size_t off = 0, bytes = 0;
    std::vector<std::thread> th;
    std::vector<uint64_t> sink;
    Pool(int h, const uint8_t * b) : H(h), base(b), sink(h * 8, 0) {
        for (int t = 0; t < H; ++t) th.emplace_back([this, t] {
            uint64_t seen = 0;
            while (true) {
                uint64_t g;
                while ((g = gen.load(std::memory_order_acquire)) == seen) { if (quit.load()) return; relax(); }
                seen = g;
                const size_t share = bytes / H / 64 * 64;
                const size_t lo = off + (size_t) t * share, hi = t == H - 1 ? off + bytes : lo + share;
                const uint64_t * p = (const uint64_t *) (base + lo);
                const size_t n = (hi - lo) / 8;
                uint64_t s = 0;
                for (size_t i = 0; i + 8 <= n; i += 8) s += p[i] ^ p[i + 1] ^ p[i + 2] ^ p[i + 3] ^ p[i + 4] ^ p[i + 5] ^ p[i + 6] ^ p[i + 7];
                sink[t * 8] += s;
                done.fetch_add(1, std::memory_order_acq_rel);
            }
        });
    }
    void post(size_t o, size_t b) { off = o; bytes = b; done.store(0); gen.fetch_add(1, std::memory_order_acq_rel); }
    void wait() { while (done.load(std::memory_order_acquire) < H) relax(); }
    ~Pool() { quit = true; for (auto & x : th) x.join(); }
};

int main(int argc, char ** argv) {
    if (argc < 7) { fprintf(stderr, "usage: %s counts.bin L helpers Bc Bp Bcp [steps]\n", argv[0]); return 2; }
    if (const char * e = getenv("READSCHED_CAP_MB")) CAP = (size_t) atoll(e) << 20;
    const int L = atoi(argv[2]), H = atoi(argv[3]);
    const double bc = atof(argv[4]), bp = atof(argv[5]), bb = atof(argv[6]);
    FILE * f = fopen(argv[1], "rb"); if (!f) { perror("counts"); return 2; }
    std::vector<uint8_t> cnt; { uint8_t buf[1 << 16]; size_t r; while ((r = fread(buf, 1, sizeof buf, f)) > 0) cnt.insert(cnt.end(), buf, buf + r); }
    fclose(f);
    size_t T = cnt.size() / L;
    if (argc > 7) T = std::min(T, (size_t) atoll(argv[7]));
    // whole-expert split per count n from the probe's rates
    int split[256];
    for (int n = 0; n < 256; ++n) {
        double best = 1e30; int bc_ = 0;
        for (int c = 0; c <= n; ++c) {
            double t = std::max({ c / bc, (n - c) / bp, n / bb });
            if (t < best - 1e-12) { best = t; bc_ = c; }
        }
        split[n] = bc_;
    }
    printf("split table:"); for (int n = 0; n <= 6; ++n) printf(" %d", split[n]); printf("\n");
    uint8_t * cbuf = (uint8_t *) aligned_alloc(4096, CAP); memset(cbuf, 1, CAP);
    uint8_t * hbuf; CK(cudaHostAlloc(&hbuf, CAP, cudaHostAllocDefault)); memset(hbuf, 2, CAP);
    uint8_t * dbuf; CK(cudaMalloc(&dbuf, 8 * S));
    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    Pool pool(H, cbuf);
    size_t coff = 0, hoff = 0;
    auto cpu_off = [&](size_t bytes) { if (bytes > CAP) { fprintf(stderr, "a task of %zu bytes exceeds the buffer\n", bytes); exit(3); }
                                        if (coff + bytes > CAP) coff = 0; size_t o = coff; coff += bytes; return o; };
    auto copies = [&](int m) {
        for (int j = 0; j < m; ++j) {
            if (hoff + S > CAP) hoff = 0;
            CK(cudaMemcpyAsync(dbuf + (j % 8) * S, hbuf + hoff, S, cudaMemcpyHostToDevice, st));
            hoff += S;
        }
    };
    auto run = [&](const char * mode, size_t t0, size_t t1) {
        const bool tok = !strcmp(mode, "token");
        for (size_t t = t0; t < t1; ++t) {
            const uint8_t * row = cnt.data() + t * L;
            if (tok) {
                int c = 0, p = 0;
                for (int l = 0; l < L; ++l) { c += split[row[l]]; p += row[l] - split[row[l]]; }
                if (c) pool.post(cpu_off(c * S), c * S);
                copies(p);
                CK(cudaStreamSynchronize(st));
                if (c) pool.wait();
                continue;
            }
            for (int l = 0; l < L; ++l) {
                const int n = row[l];
                if (!n) continue;
                const int c = !strcmp(mode, "layer_link") ? 0 : !strcmp(mode, "layer_cpu") ? n : split[n];
                if (c) pool.post(cpu_off(c * S), c * S);
                copies(n - c);
                if (n - c) CK(cudaStreamSynchronize(st));
                if (c) pool.wait();
            }
        }
    };
    double reads = 0; for (size_t t = 0; t < T; ++t) for (int l = 0; l < L; ++l) reads += cnt[t * L + l];
    reads /= T;
    for (int rep = 0; rep < 2; ++rep) {
        for (const char * mode : { "layer", "token", "layer_link", "layer_cpu" }) {
            run(mode, 0, std::min<size_t>(T, 64));   // warm-up
            auto a = clk::now();
            run(mode, 0, T);
            double s = std::chrono::duration<double>(clk::now() - a).count();
            printf("readsched mode=%s rep=%d steps=%zu ms_per_token %.4f reads_per_token %.3f gbs %.2f\n", mode, rep, T, 1e3 * s / T,
                   reads, reads * S * T / s / 1e9);
            fflush(stdout);
        }
    }
    return 0;
}
