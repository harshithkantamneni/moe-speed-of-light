# Job 079: where our time per token goes on Qwen3 at 43.75% — predictions and outcome

The predictions are in the header of `jobs/079_qwen3_profile@vast.sh` (gpu branch commit 48f0529, pushed before
launch). Numbers are in `prereg/qwen3_profile_079.json` and the gpu branch `results/079_qwen3_profile@vast/`
(`prof_*.json`, and a 3-token kernel timeline in `tail3/`).

**Setup.**
- **Host:** RTX 5090 + Ryzen 9 9950X3D. This is a different machine from job 078, which was no longer for rent.
- **Host memory:** CPU STREAM read 66.9 GB/s, PCIe H2D 57.9 GB/s. Job 078's host had 44.7 / 46.7 GB/s.
- **Workload and protocol:** Qwen3-30B-A3B BF16, 43.75% of experts on the GPU. 30 AIME-25 problems unprofiled,
  greedy, 256 tokens, session.
- **Profiles:** Nsight Systems on the decode of one full request (problem 0), under `nsys launch` with start/stop
  around the request. Per-token means are over the last 200 decode tokens; CUDA graphs are traced per node.
- **Profiler cost:** 0–4% on speed (profiled request against the same problem unprofiled).

**Speeds (tok/s):**
- ours 99.7 (without FETCH) / 100.1 (with FETCH);
- FreeToken 98.5 (offload) / 101.0 (hybrid);
- llama.cpp 27.5.

**Where one decode token goes (ms, mean of 200 tokens, profiled):**

| | ours (FETCH) | FreeToken offload |
|---|---|---|
| Missed experts copied over PCIe | 4.93 | 5.20 |
| Resident-expert matrix-vector products | 2.29 | 2.21 |
| Attention projections (Q, K, V, O) | 1.65 (four kernels) | 1.34 (fused QKV + O) |
| Attention | 0.49 | 0.37 |
| Router + top-k | 0.29 | ~0.28 |
| Output head | 0.38 | 0.39 |
| Norms, small ops, cache control, waits | 0.92 | ~0.4 |
| GPU idle between kernels | 0.60 | 0.27 |
| **Token (wall)** | **10.92** | **10.45** |
| Kernels per token | 1436 | 894 |

In ours, 0.63 ms of kernel time overlaps other kernels (concurrent streams; not attributed further), so our rows add up to more
than the wall time. FreeToken's rows add up to its wall time.

1. **Held.** Our host-side time per step is 0.27 ms (app 48, pre 2, inputs 79, launch 103, post 33 µs), 2.7% of the
   token. The fixed cost is on the GPU.
2. **Held.** We launch 1.60× as many GPU kernels per token as FreeToken (1436 vs 894).
3. **Held.** Ours ÷ FreeToken's better backend = 0.991 [0.975, 1.009]: parity on this host. Job 078's −5.2% depends on
   the host. Against FreeToken offload, the backend that led in job 078, ours is +1.7% [+0.4, +3.0].

**What the 5.44 ms "fixed cost" of job 078 is.**
- It is not overhead. It is the GPU work every token needs whatever misses:
  - attention and its projections;
  - the router;
  - the resident experts;
  - the output head.
- On this GPU that work is ~5.3 ms for us and ~5.0 ms for FreeToken.
- **Our extra ~0.4–0.6 ms per token comes from llama.cpp's unfused batch-1 kernels:**
  - separate Q, K and V products;
  - separate norms and small element-wise kernels;
  - 1.6× the kernel count, and 0.3 ms more idle time between kernels.
- **The miss path is the other half of the token in both systems:** 4.9–5.2 ms of PCIe copies.

**Neither system overlaps its PCIe copies with compute.**
- In the 3-token timelines, 0 µs of copy time overlaps any other kernel, in ours and in FreeToken.
- Each layer runs router → copy the missed experts → all 8 experts' products.
- **Bound:** computing a layer's resident experts during its copy would save min(copy, resident compute) per layer.
  From the 3-token sample that is about 0.5–0.9 ms per token (5–9%). Most layers copy nothing, and the layers that
  copy a lot have few resident hits.

**FETCH on this host.**
- FETCH gains only +0.4% here, against +16% on job 078's host.
- The table (0,1,1,2,3,3,4,5,6 per miss count) sends 75% of misses over PCIe (22.4 of 29.8 per token).
- On this host the CPU reads host memory faster than PCIe (66.9 vs 57.9 GB/s), so the split is probably not the best
  one. The fixed table is not tuned per host.
