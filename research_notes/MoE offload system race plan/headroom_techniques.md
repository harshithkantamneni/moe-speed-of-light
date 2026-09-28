# Where batch-1 hybrid CPU/GPU MoE decode loses time against the hardware limit, and which techniques close each gap (state as of 28 Sep 2026)

Scope and method. This note extends four earlier notes: `gpu_initiated_signalling.md`, `academic_systems.md`, `numerics_and_prediction.md` and `time_domain_audit_and_invariance.md`. It does not repeat their novelty findings.
- Paper facts come from full-text greps (pdftotext) of the arXiv/SOSP PDFs, fetched 28 Sep 2026.
- Code facts come from local clones: llama.cpp 4d86b2f (27 Sep 2026), kt-kernel/ktransformers c40722b (23 Sep 2026), ik_llama.cpp (27 Sep 2026).
- Community numbers come from GitHub pages via WebFetch summaries.
- "Bandwidth fraction" figures are my arithmetic: published tok/s multiplied by the per-token weight bytes in `data/gguf_bytes.json`, which the repo derives from GGUF headers. They exclude KV-cache reads and appear under Inferences.
- The fitted A10 system parameters (GPU at 38% of datasheet bandwidth, CPU at 60% of peak, 21 µs per CPU-executed expert, 84 µs/layer host hand-off) are the assignment's. They are not independently sourced.

## Q1. GPU side: batch-1 bandwidth efficiency of llama.cpp's quantized GEMV/MoE kernels against the best available, why it is low, and which fixes work

### Takeaway
On Ada and Blackwell, llama.cpp's batch-1 decode reaches about 60–70% of datasheet DRAM bandwidth on dense Q4_0 and on gpt-oss-20b MXFP4 after its late-2025 fusion work. It reaches only about 38% (RTX 5090) to 52% (RTX 4090) on Qwen3-30B-A3B Q4_K_M, whose experts are tiny. The best published batch-1 systems reach 62% (Cohere's persistent megakernel on a 30B-A3B MoE, H100) and 78% (Hazy's dense Llama-1B megakernel, H100). Kernel-per-op engines (vLLM/SGLang) sit at 39–50%.

The loss is a fixed cost per kernel and per layer, not a GEMV inner-loop problem:
- launch latency and gaps, about 0.8 µs per node even inside a CUDA graph on B200;
- fill/drain of short kernels, which get about 20% of peak for grouped expert GEMMs;
- serialized small auxiliary ops.

These costs are constant in time, so they take a larger share on faster GPUs. That explains why the 5090 scores below the 4090 on the same model. Fusion (upstream llama.cpp: +27–42% tg on MoE on RTX 5090) is the cheap fix. Persistent/megakernels, PDL and fused MoE operators are the expensive fix (1.14–1.58× in 2025–2026 papers).

### Cited Findings
**llama.cpp measured batch-1 decode (tg128) on Ada/Blackwell**
- CUDA scoreboard, Llama 2 7B Q4_0 (3.56 GiB), tg128: RTX 5090 300.40 t/s; RTX PRO 6000 Blackwell 281.11; H100 80GB 280.74; A100 80GB 200.90; RTX 4090 188.96; RTX 5080 184.68; RTX 3090 Ti 172.26. All with FA; commits differ per row. — [llama.cpp Discussion #15013](https://github.com/ggml-org/llama.cpp/discussions/15013)
- gpt-oss guide (Aug 2025 builds, around b6205–b6316):
  - gpt-oss-20b tg128: RTX 5090 282.51 t/s; RTX PRO 6000 286.91; RTX 4090 225.22; RTX 3090 161.95.
  - gpt-oss-120b tg128: RTX PRO 6000 196.31 t/s.
  - — [llama.cpp Discussion #15396](https://github.com/ggml-org/llama.cpp/discussions/15396)
- am17an, "Optimizing Token Generation in llama.cpp's CUDA Backend" (30 Nov 2025). Fusions off → on, tg128:
  - Qwen3-30B-A3B Q4_K_M: RTX 5090 246.96 → 352.06 t/s; RTX 4090 198.39 → 271.04.
  - gpt-oss-20b MXFP4: RTX 5090 329.23 → 419.14; RTX 4090 232.05 → 271.99.
  - PRs: #16715 GEMV fusion (GLU, bias+GLU, bias), #16130 TopK-MoE (softmax + top-k fused), #14800 fused RMS norm (+mul, +add), #15631 fused adds, #14907 softcap, #16991 concurrent streams for Q/K/V (needs `GGML_CUDA_GRAPH_OPT=1`).
  - "none of these PRs increase the TG by more than 10%" individually; "kernel fusion increases register pressure".
  - — [Discussion #17621](https://github.com/ggml-org/llama.cpp/discussions/17621); [Aman's blog](https://am17an.bearblog.dev/new-post/)
- PR #16715 alone (GEMV fusion): +5% tg on gpt-oss-120B MXFP4 (6× 4090), +3% on llama 8B Q4_0, +5% on llama 1B Q4_0 (RTX 4090). "enabling fusion only for Volta and newer" (P40 showed little or negative gain). — [llama.cpp PR #16715](https://github.com/ggml-org/llama.cpp/pull/16715)
- CUDA graphs in llama.cpp (NVIDIA, Aug 2024): "the highest achieved speedup is 1.2x for the smallest Llama 7B model on the fastest NVIDIA H100 GPUs", batch 1. Benefit grows with faster GPUs and smaller models. — [NVIDIA blog, A. Gray](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/)
- PDL (Programmatic Dependent Launch) for llama.cpp: proposed in issue #15479 (21 Aug 2025), implemented in PR #15480, **closed without merge**; the issue is labelled stale. PDL "allows for a dependent secondary kernel to launch before the primary kernel it depends on in the same CUDA stream has finished executing". — [issue #15479](https://github.com/ggml-org/llama.cpp/issues/15479); [PR #15480](https://github.com/ggml-org/llama.cpp/pull/15480)
- GH200, llama.cpp all-GPU: gpt-oss-20b 322.93 t/s; gpt-oss-120b 209.01 t/s (fairydreaming, Dec 2025). — [Discussion #18005](https://github.com/ggml-org/llama.cpp/discussions/18005)
- Project-internal: on a GH200 VM, all-GPU runs were over-predicted by the bytes/bandwidth model (measured/predicted 0.28–0.92) because "at 4 TB/s a small model's decode step is dominated by per-layer fixed costs". — [paper/numbers4.tex (repo)](/home/claude/moe-speed-of-light/paper/numbers4.tex)

**Best-in-class batch-1 kernels and systems**
- Hazy Research megakernel (Llama-1B, dense, 27 May 2025):
  - 78% of H100 bandwidth; vLLM/SGLang use "at most 50%".
  - Under 1 ms per forward on H100, about 600 µs on B200. About 2.5× vLLM and 1.5× SGLang on H100; about 3.5× and 1.5× on B200.
  - Launch "about 2.1 microseconds" on H100, "around 1.3 microseconds" with CUDA graphs. Existing systems use "around a hundred separate kernels" per forward.
  - Uses counter-based synchronization in global memory.
  - — [Hazy Research blog](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- Cohere megakernel (8 Sep 2026; North Mini Code, a 30B MoE with 3.3B active, one H100, batch 1):
  - 292 tok/s at 62% of theoretical bandwidth, against vLLM's 185 tok/s at 39% (1.58×).
  - Named losses: wave quantization ("The smallest kernel, the worse that rounding gets"), false dependencies from full-grid barriers, and launch overhead.
  - Techniques: one persistent kernel, counter-based fine-grained barriers, and prefetching of immutable weights into shared memory "before activation dependencies resolve".
  - Vendor blog, not peer reviewed. — [Cohere blog](https://cohere.com/blog/megakernels)
- MonoMoE (Amazon AGI, arXiv 2609.04244; H200, FP8 block-wise only). The fetched page gives 19 Aug 2026, while the arXiv ID implies a September posting.
  - Fuses router logits, top-k, activation quantization, gate/up, activation, down and reduction into one persistent kernel.
  - 1.54× over vLLM Triton grouped GEMM on the routed-MoE operator. TPOT −9.9% to −18.7% (Qwen3.5-35B), −9.1% to −11.5% (Qwen3.5-122B), −2.9% to −16.7% (GLM-5.2).
  - Baseline expert GEMMs reach "only 17–22% of peak DRAM". "Short grouped-GEMM kernels spend most of their lifetime in fill and drain phases."
  - — [arXiv 2609.04244](https://arxiv.org/html/2609.04244v1)
- MPK, Mirage Persistent Kernel (CMU/NVIDIA et al., arXiv 2512.22219, 22 Dec 2025):
  - 1.0–1.7× over kernel-per-operator systems.
  - On B200, "each launch costs 3.8 µs in eager execution … and 0.8 µs with CUDA Graphs". Qwen3-8B issues 293 launches per token (0.2 ms/token under graphs).
  - Qwen3-30B-A3B has 533 operators.
  - MoE layer of Qwen3-30B-A3B on B200: MPK-Hybrid-MoE is 1.14× faster than SGLang-MoE at batch 1.
  - — [arXiv 2512.22219](https://arxiv.org/abs/2512.22219) (full-text grep)
- Marlin (Frantar, Castro, Chen, Hoefler, Alistarh; PPoPP'25; benchmarked on **A10**):
  - FP16×INT4 GEMM with "close to ideal (4x) speedups", maximum 3.87× after scale overhead, "up to batchsizes around 16-32". Earlier kernels were near-optimal only at batch 1.
  - Techniques: async global weight loads with L2-evict hints, activations from L2, striped partitioning for SM balance, reduction in the output buffer.
  - The public text I fetched gives no batch-1 number or %-of-bandwidth figure.
  - — [Marlin README](https://github.com/IST-DASLab/marlin); [arXiv 2408.11743](https://arxiv.org/abs/2408.11743)
- NVIDIA's low-latency Blackwell recipe (Llama 4 Maverick >1,000 TPS/user, May 2025): PDL "eliminates GPU idle gaps between consecutive kernels"; fusions (AllReduce+RMSNorm+Quantize, SwiGLU into the preceding GEMM); CUDA graphs. No per-technique gain is quantified. Vendor claim. — [NVIDIA blog](https://developer.nvidia.com/blog/blackwell-breaks-the-1000-tps-user-barrier-with-metas-llama-4-maverick)

**Hardware facts for Blackwell targets**
- RTX PRO 6000 Blackwell (GB202):
  - 188 SMs, 128 MB L2. VRAM latency 329 ns; L2 about 130 ns.
  - "dispatches with short-duration waves may struggle to take advantage of Blackwell's scale". The 1:16 GPC-to-SM ratio makes work distribution a bottleneck for small workloads.
  - — [Chips and Cheese, Blackwell](https://old.chipsandcheese.com/2025/06/28/blackwell-nvidias-massive-gpu/)
- KTransformers profiled hybrid DeepSeek-V3 decode on A100: Fiddler issues >7,000 launches per token at 16 µs average (73% of GPU time); llama.cpp about 3,000 at 5 µs (21% of GPU time). — [KTransformers SOSP'25](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)

### Inferences
- **Computed bandwidth fractions** (weights read per token × tok/s ÷ datasheet bandwidth; bytes per token from `data/gguf_bytes.json`: gpt-oss-20b 2.575 GB, gpt-oss-120b 3.594 GB, Qwen3-30B-A3B Q4_K_M 1.920 GB, Llama-2-7B Q4_0 ≈3.75 GB):

  | Model | GPU | tg t/s | GB/s | % of datasheet |
  |---|---|---|---|---|
  | Llama-2-7B Q4_0 (dense) | RTX 4090 | 189.0 | 708 | 70% |
  | Llama-2-7B Q4_0 | RTX 5090 | 300.4 | 1,126 | 63% |
  | Llama-2-7B Q4_0 | RTX PRO 6000 | 281.1 | 1,054 | 59% |
  | gpt-oss-20b MXFP4 (Aug 2025) | RTX 5090 / PRO 6000 / 4090 | 282.5 / 286.9 / 225.2 | 727 / 739 / 580 | 41% / 41% / 58% |
  | gpt-oss-20b MXFP4 (Nov 2025, fused) | RTX 5090 / 4090 | 419.1 / 272.0 | 1,079 / 700 | 60% / 70% |
  | gpt-oss-120b MXFP4 (Aug 2025) | RTX PRO 6000 | 196.3 | 706 | 39% |
  | Qwen3-30B-A3B Q4_K_M (Nov 2025, unfused → fused) | RTX 5090 | 247 → 352 | 474 → 676 | 26% → 38% |
  | Qwen3-30B-A3B Q4_K_M (unfused → fused) | RTX 4090 | 198 → 271 | 381 → 520 | 38% → 52% |
  | gpt-oss-20b / 120b | GH200 (4 TB/s) | 322.9 / 209.0 | 832 / 751 | 21% / 19% |

  The A10's 38% for the system's GPU part matches upstream llama.cpp for small-expert MoE on current GPUs. It is not an outlier.
- **Why efficiency falls as the GPU gets faster.**
  - The fused Qwen3-30B-A3B step on a 5090 is 2.84 ms, or about 59 µs per layer across 48 layers. MPK counts 533 operators for this model, about 11 per layer. At 0.8 µs per graph node (B200 figure) that is about 9 µs per layer of pure launch latency, roughly 15% of the step, before counting kernel ramp/drain.
  - Little's law: at 1.79 TB/s and 329 ns unloaded latency, the bandwidth-delay product is about 0.59 MB. One Qwen3-30B Q4_K_M expert matrix is about 1 MB (2.86 MB for all three), so a single expert GEMV sits near the bandwidth-delay scale. It cannot reach steady-state streaming unless many experts or ops are batched into one kernel or overlapped across kernels. That is exactly what PDL and persistent kernels provide.
  - Expect the system's GPU fraction to **drop** from the A10's 38% on a 5090 or PRO 6000 unless per-layer kernel count and ramp costs are cut. The 4090 → 5090 drop for the same model (52% → 38%) is the empirical precedent.
- **Ranking the fixes by cost in llama.cpp**:
  1. Make sure the branch carries upstream's Nov 2025 fusions and uses fused gate/up+GLU mmvq for the GPU-hit experts. Rebase or cherry-pick; low cost; +27–42% on pure-GPU MoE tg in upstream's own measurement.
  2. Fold the system's own per-layer kernels (cache lookup/slot remap, mailbox publish, spin/merge, weighted sum) into existing kernels: the top-k kernel and the down-projection/weighted-sum kernel. Each removed node saves roughly 1–5 µs per layer.
  3. PDL: an upstream attempt exists (PR #15480) but was not merged. Medium cost.
  4. A persistent MoE kernel or megakernel: 1.14–1.58× in MPK/MonoMoE/Cohere, but weeks of work.
- For hit experts, what matters most is batching all resident top-k experts of a projection into one `mul_mat_id`-style launch rather than one launch per expert. The Little's-law argument says the kernel must see all ~k·(0.9–13) MB of the layer's hit weights at once.

### Gaps
- I found no public llama.cpp tg numbers on an A10 or L4, and no public Q8_0 MoE tg numbers on Blackwell. The fractions above cover Q4_0, Q4_K_M and MXFP4 only.
- No source measures llama.cpp mmvq/mul_mat_id against TensorRT-LLM, FlashInfer or Machete for **batch-1 MoE on RTX-class Blackwell**. The best-in-class numbers are on H100/H200/B200 and mostly FP8/BF16, so comparisons across formats and GPUs are indirect.
- Cohere's 62% and NVIDIA's PDL claims are vendor blog figures. MonoMoE's 17–22% baseline is for vLLM Triton FP8 grouped GEMM on H200, not llama.cpp mmvq.
- The 0.8 µs per-node graph launch cost was measured on B200 (MPK). I found no equivalent figure for the RTX 5090 or PRO 6000.
- I did not verify whether current llama.cpp master contains any PDL code path, beyond the closed PR.

## Q2. CPU side: achieved bandwidth of MoE expert GEMV on server CPUs, thread scaling, NUMA, and per-expert fixed overheads

### Takeaway
Well-tuned CPU MoE decode reaches about 80% of **measured** (STREAM/likwid-class) bandwidth. llama.cpp on a 72-core Grace at 32 threads hits about 80% of likwid on gpt-oss-20b/120b. That is still only about half of datasheet peak, and cloud VMs expose just 46–59% of datasheet in STREAM.

Efficiency collapses from two sources:
- too many threads (Grace: 84.5 t/s at 32 threads, 34.8 t/s with high variance at 72 threads);
- cross-socket traffic (DeepSeek-V3 gains 2–8% from a second socket).

The dominant per-expert overhead is synchronization: barriers take 15% and out-of-GEMV work 30% of time for 7168×2048 matrices on dual Genoa. Serial single-thread ops also cost 6.9 ms/token in one ik_llama.cpp case. Remedies with evidence:
- merge all selected experts into one or two fused task batches (KTransformers);
- atomic work-stealing over small chunks in a spinning pool (kt-kernel, llama.cpp);
- NUMA-local weight slicing (KTransformers +63% on dual socket);
- thread count tuned below core count.

### Cited Findings
**Achieved bandwidth**
- FreeToken measured B_H, "the measured effective bandwidth of the CPU-side MoE expert kernel … on the deployed tensor shapes":
  - 2× Xeon Gold 6459C, DDR5, 6 threads: 77.3 GB/s.
  - 2× Xeon Plat 8358P, DDR4, 6 threads: 63.2 GB/s.
  - 2× Xeon Gold 6330, DDR4, 6 threads: 56.7 GB/s.
  - Ryzen 9 9950X3D, 16 cores: 53.8 GB/s.
  - i9-13900H: 47.5 GB/s.
  - **Xeon Platinum 8559C, 48 threads, DDR5: 178 GB/s** (the RTX PRO 6000 box).
  - Servers were "capped at 6 CPU threads and pinned to the GPU's NUMA node".
  - Workers "form a persistent C++ pool pinned to physical cores" using "architecture-specific SIMD and in-kernel dequantization".
  - — [FreeToken, arXiv 2608.16157](https://arxiv.org/abs/2608.16157)
- llama.cpp on GH200 Grace (72 Neoverse-V2 cores, LPDDR5X, likwid about 267 GB/s sustainable):
  - gpt-oss-20b MXFP4 CPU-only: 84.52 t/s at 32 threads vs 34.83 ± 10.05 t/s at 72 threads.
  - gpt-oss-120b: 59.47 t/s at 32 threads.
  - — [llama.cpp Discussion #18005](https://github.com/ggml-org/llama.cpp/discussions/18005)
- Dual-socket EPYC STREAM-class bandwidth (fairydreaming et al., Feb–Mar 2025):
  - 1× 9374F NPS2: 185.3 GB/s (80.4%).
  - 2× 9175F Turin: 378.3 → 753.6 GB/s (92%).
  - 2× 9654 Genoa: 359.9 → 679.2 GB/s (78% → 73.6%).
  - DeepSeek-V3 Q4_K_S tg single → dual socket: 9.08 → 9.79 t/s (Turin, 108%); 8.48 → 8.67 (Genoa, 102%).
  - Mixtral 8x22B Q8_0 scales 146–185%.
  - — [llama.cpp Discussion #11733](https://github.com/ggml-org/llama.cpp/discussions/11733)
- Chips and Cheese, EPYC 9355P (32 cores as 8 CCDs × 4 enabled cores, 12-channel DDR5-5200, "just under 500 GB/s" theoretical):
  - GMI-Wide gives 99.8 GB/s read per CCD (62.5 GB/s for desktop GMI-Narrow).
  - An NPS4 node reached 117.33 GB/s locally and about 107 GB/s to other nodes.
  - — [Chips and Cheese, EPYC 9355P](https://chipsandcheese.com/p/amds-epyc-9355p-inside-a-32-core)
- KTransformers (SOSP'25):
  - Dual Xeon 8452Y, MLC-measured 220 GB/s intra-socket and 125 GB/s cross-socket.
  - Fiddler's single MoE layer of DeepSeek-V3 takes 6.9 ms on one socket and 5.8 ms on two ("a modest 16% improvement … due to inefficient memory access across NUMA nodes").
  - NUMA-aware tensor parallelism, which slices each expert's matrices per socket followed by reduce-scatter, "improves decoding throughput by up to 1.63×" over NUMA-oblivious execution.
  - For decode ("ARI is four or fewer"), a lightweight AVX-512 kernel beats AMX.
  - — [KTransformers SOSP'25](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- ARCLIGHT (arXiv 2603.07770; 4-NUMA Kunpeng-920 ARM): local memory about 102 GB/s vs remote 23–26 GB/s. llama.cpp's `--numa distribute` leaves placement to the OS, which causes remote accesses. NUMA-aware TP gives "up to 46%" over llama.cpp. — [arXiv 2603.07770](https://arxiv.org/html/2603.07770v1)
- Thread scaling of a Mixtral-8x7B FFN on a Threadripper 7960X with an RTX 4090 over PCIe 4.0 ×16 (2512.16473, Table I): CPU compute time 44.1 / 25.5 / 18.3 / 15.8 / 11.0 / 7.3 ms at 1 / 2 / 4 / 8 / 16 / 24 threads. The GPU takes 0.3 ms, and the transfer ("communication") 28.0 ms. — [arXiv 2512.16473](https://arxiv.org/abs/2512.16473)
- Project-internal: cloud VM hosts reached 46–59% of datasheet in STREAM Triad. On an A100 host (30 vCPUs of an EPYC 7J13), Triad was 93.7 of 204.8 GB/s, scaling 28.5 → 68.4 → 80.7 → 83.2 → 93.7 GB/s at 1 / 4 / 8 / 15 / 30 threads. — [audit_requirements.md (repo)](/home/claude/moe-speed-of-light/research_notes/MoE%20audit%20measurement%20plan/audit_requirements.md)

**Per-op and per-expert fixed overheads, and how systems reduce them**
- Dual-socket small-matrix analysis (fairydreaming, 21 Feb 2025):
  - For 7168×2048 matrices (DeepSeek expert shape), barrier and synchronization take 45.1% of computation time on dual socket vs 27.4% on single socket.
  - Breakdown on 2× 9654: 29.7% outside `ggml_compute_forward_mul_mat()`, 15.4% in barrier code, 54.9% in compute.
  - — [Discussion #11733](https://github.com/ggml-org/llama.cpp/discussions/11733)
- ik_llama.cpp issue #2406 (Threadripper PRO 5975WX, Qwen3.8-Flash-Next):
  - A `[10240,1]` sigmoid ran on one thread while "the others wait at the barrier", costing about 9% of decode time.
  - Element-wise splitting, SIMD and fusion gave +6.7% (13.87 → 14.80 t/s).
  - Time fitted as "61.82 ms + 1.30 ms × expert_count".
  - — [ik_llama.cpp #2406](https://github.com/ikawrakow/ik_llama.cpp/issues/2406)
- KTransformers fused MoE operator: "merging Gate projections across experts into one larger task, similarly fusing Up and Down … and combining Gate and Up … MoE execution reduces to two fused batches, significantly lowering threading overhead". Dynamic task scheduling gives up to 1.83× in prefill; no decode figure is given. — [KTransformers SOSP'25](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- kt-kernel worker pool (code, c40722b):
  - Workers **spin** on an atomic status and park on a condition variable only after >50 ms idle.
  - Tasks are claimed with `curr_.fetch_add(1)` ("omp-guided-style", block = 1). The master thread also executes tasks, then spin-waits on each worker's status.
  - Threads are bound per NUMA node via hwloc.
  - — [worker_pool.cpp](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/cpu_backend/worker_pool.cpp)
- llama.cpp CPU `mul_mat_id` (4d86b2f):
  - One `ggml_barrier` after activation quantization and row counting.
  - Then all threads walk the experts in order, taking 16/64-row chunks from a **per-expert atomic chunk counter**, with no barrier between experts.
  - "disable for NUMA": chunking is turned off when NUMA mode is on.
  - Each graph op (gate, up, GLU, down, weighted sum) still ends in a graph-level barrier.
  - — [ggml-cpu.c](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/ggml-cpu.c)
- ik_llama.cpp enables fused MoE up/gate by default (`-no-fmoe` disables it) and offers `-thp` (transparent huge pages). — [ik_llama.cpp common.cpp](https://github.com/ikawrakow/ik_llama.cpp/blob/main/common/common.cpp)

### Inferences
- Implied efficiencies:
  - Grace llama.cpp: 84.52 t/s × 2.575 GB ≈ 218 GB/s (≈81% of likwid 267); 59.47 × 3.594 ≈ 214 GB/s (≈80%).
  - FreeToken's PRO 6000 host: 178 GB/s on a 48-thread Emerald Rapids. Assuming 8-channel DDR5-5600, peak is about 358 GB/s, so ≈50% of datasheet; the channel count is my assumption.
  - The system's "60% of peak" is therefore at or above typical if "peak" means datasheet, and below best practice (≈80%) if it means STREAM. **Clarify which basis the 60% uses.**
- **Fixed costs dominate on server CPUs.**
  - At a plausible 276 GB/s (60% of a 460.8 GB/s 12-channel DDR5-4800 datasheet), streaming one Qwen3-30B Q4_K_M expert (2.86 MB) takes about 10 µs, and one gpt-oss expert (13.25 MB) about 48 µs.
  - A fixed 21 µs per expert is then 2× the byte time for Qwen3 and 44% of it for gpt-oss.
  - On the A10 host (helpers at 107–143 GB/s) the same 21 µs was relatively smaller. The racing hardware makes the fixed cost the largest CPU-side gap.
- Realistic floor for a per-layer miss batch:
  - One wake of a spinning pool, one barrier between gate/up and down (or none with per-expert completion counters), one completion flag. That is a few µs per layer, not tens of µs per expert.
  - I found no published microbenchmark of this floor. The fairydreaming shares (≈15% barrier, ≈30% outside GEMV) are the closest evidence.
- Thread count must be swept, not maximized:
  - Grace fell by 59% (84.5 → 34.8 t/s) from 32 to 72 threads.
  - FreeToken deliberately used 6 threads per NUMA node on large servers.
  - Per-CCD bandwidth on Turin is about 100 GB/s, so roughly 4–6 busy CCDs saturate a 12-channel socket. Extra threads add barrier cost without adding bandwidth.
- NUMA: if the 30–60-core race host is a dual-socket or multi-NPS VM, expert weights must be partitioned (KTransformers-style row slicing, or whole experts pinned per node with helpers on that node). Otherwise expect DeepSeek-like 2–8% scaling from the second socket.

### Gaps
- No public per-expert (per-call) fixed-cost measurement exists for kt-kernel, ik_llama.cpp or llama.cpp CPU MoE GEMV at batch 1. The 21 µs figure can only be compared with indirect evidence.
- I found no batch-1 MoE decode bandwidth measurements on Intel Granite Rapids (MRDIMM) or on EPYC Turin with llama.cpp or kt-kernel. Only STREAM-class figures (Turin 92%) and FreeToken's Emerald Rapids 178 GB/s are available.
- No measured effect of huge pages on CPU MoE decode was found (ik's `-thp` exists, no benchmark located).
- The KTransformers paper gives no decode-only effect of its two-batch fusion or work stealing, only prefill.

## Q3. CPU–GPU coordination: measured hand-off latencies, and how much intra-layer overlap saves

### Takeaway
The physical floor of a GPU↔CPU mailbox round trip over CPU-attached PCIe is about 1–2 µs: 0.54–0.79 µs per PCIe access, plus a system-scope fence that costs about 1 µs on GB200. Host-driven paths cost 5–16 µs per launch, 6–7 µs per `cudaMemcpy`, and ~84 µs per layer in the system's own llama.cpp measurement.

KTransformers, FreeToken and SeqMoE all keep a host function or host thread in the loop inside CUDA graphs. No public microsecond figure exists for a graph host node's wake-up.

Intra-layer overlap is where coordination pays. KTransformers found only 5% CPU/GPU overlap, with GPU utilization at 28%. HybriMoE's intra-layer scheduling gave 1.46× decode in its ablation, and KTransformers' CUDA graph up to 1.23×. The bigger KTransformers win, Expert Deferral (+33%), changes outputs.

Once the mailbox is in place, remaining coordination headroom is small: a few µs per layer. The priority is keeping the GPU busy while CPU misses run.

### Cited Findings
- PCIe access latency, measured with a GPU reading host-coherent memory (Nvidia T1000 over Vulkan): CPU-attached PCIe about 650 ns (Zen 5 AM5), 785 ns (Arrow Lake), 536 ns (Skylake). Chipset-attached adds 340–920 ns. — [Chips and Cheese, chipset microbenchmarks](https://chipsandcheese.com/p/microbenchmarking-chipsets-for-giggles)
- Speed-of-light latency study on GB200 NVL72 (arXiv 2607.16100):
  - "each barrier will incur more than 1 µs of overhead", about 40% of a ~5 µs small AllReduce.
  - L2 round trip 0.306 µs; remote store 0.792 µs.
  - Recommendation: "Eliminate memory barriers entirely through barrier-free synchronization using low-latency (LL) protocol or sentinel-based polling".
  - — [arXiv 2607.16100](https://arxiv.org/html/2607.16100v1)
- GDRCopy (CPU writes into mapped GPU memory):
  - `gdr_copy_to_mapping` takes 0.088 µs for 4 B and 0.18 µs for 1 KB, against `cudaMemcpy`'s "6-7us overhead" (V100).
  - Reads from GPU memory are slow (1.7–3.1 µs small).
  - Supported on data-center and RTX workstation GPUs, **not GeForce**.
  - — [NVIDIA/gdrcopy README](https://github.com/NVIDIA/gdrcopy)
- Launch costs:
  - B200: 3.8 µs eager, 0.8 µs in CUDA graphs. — [MPK, arXiv 2512.22219](https://arxiv.org/abs/2512.22219)
  - H100: 2.1 µs, 1.3 µs with graphs. — [Hazy Research](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
  - Hybrid DeepSeek-V3 on A100: 16 µs per launch in Fiddler, 5 µs in llama.cpp. — [KTransformers SOSP'25](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- Older and third-party round trips (from companion notes):
  - `cudaMemcpyAsync` 1.2 µs API + 6 µs driver; `cudaStreamSynchronize` 1 µs (HPCA'13 hardware). — [Lustig & Martonosi](https://mrmgroup.cs.princeton.edu/papers/dlustigHPCA13.pdf)
  - NCCL GIN proxy GPU→CPU-proxy→GPU round trip 18.0 µs, network included. — [arXiv 2511.15076](https://arxiv.org/html/2511.15076v1)
  - One forum report of ~12 ms host-callback latency (CUDA 12.6) that NVIDIA could not reproduce. — [NVIDIA forum](https://forums.developer.nvidia.com/t/culaunchhostfunc-overhead-latency-usage-cpu-gpu-signaling/327066)
- KTransformers (SOSP'25; single DeepSeek-V3 layer, BF16):
  - "CPU utilization was 74%, while GPU utilization was merely 28%, with CPU-GPU overlap accounting for only 5% of total execution time"; "synchronization and activation transfers accounted for another 3% idle time".
  - Expert Deferral (3 of 8 experts deferred) "reduced single-layer execution time by 26%, and increased end-to-end decoding throughput by 33%". It **changes outputs**.
  - One decode CUDA graph via `cudaLaunchHostFunc` gives "up to 1.23×".
  - — [KTransformers SOSP'25](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- FreeToken captures the CPU branch into the decode graph: "The device-to-host copies, a host-function submit node, the concurrent GPU path, a synchronization node, and the host-to-device result copy are all captured together". Routing-dependent cache control runs on the GPU, because "a host-controlled cache would reintroduce a costly device synchronization at every MoE layer". — [FreeToken, arXiv 2608.16157](https://arxiv.org/abs/2608.16157)
- HybriMoE decode ablation (A6000 + Xeon 5220R, 10 cores): baseline 0.21 s → +scheduling 0.14 s (1.46×) → +prefetching 0.18 s (1.15×) → +caching 0.15 s (1.38×) → all 0.11 s (1.86×). Scheduling means dynamic intra-layer CPU/GPU balancing. — [HybriMoE, arXiv 2504.05897](https://arxiv.org/abs/2504.05897)
- 2512.16473 needed "two independent CUDA streams. Without these dual streams, data transfers would be serialized". — [arXiv 2512.16473](https://arxiv.org/abs/2512.16473)
- Graph host function semantics: the stream "is considered idle for the duration of the function's execution", and the function "may not call any CUDA APIs". — [CUDA Programming Guide 2.5](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
- colibri (6× RTX 5090): an apparent 54 ms/token of API time was a "summation illusion". CPU expert sweeps (~57 ms/token) were the real bottleneck, and a one-graph rewrite was deprioritized. — [colibri #431](https://github.com/JustVugg/colibri/issues/431)

### Inferences
- The mailbox already removes the ~84 µs/layer host path. What remains per layer is roughly:
  - the GPU publish (a store plus `__threadfence_system`, ~1 µs);
  - the CPU poll-detect (~0.5–1 µs);
  - the CPU result write;
  - the GPU poll/merge (~0.5–1 µs per PCIe read).

  That is a few µs, versus 21 µs × misses on the CPU and tens of µs of GPU work per layer. **Further coordination engineering is low-yield**, with two exceptions:
  - (a) Put a sequence number in each result cache line (LL/LL128-style "flag in data"). The GPU can then consume results without a separate flag load and a fence.
  - (b) Stream results per expert, so the GPU merges early finishers while later misses are still computing.
- The larger coordination gain is **scheduling**, i.e. making the per-layer critical path max(GPU hits, CPU misses) rather than their sum:
  - Issue hit-expert GEMVs before the spin kernel on disjoint SMs (or a second stream).
  - Choose how many misses go to the CPU vs PCIe fill so both finish together (FreeToken's q*, HybriMoE's balancing).
  - HybriMoE's 1.46× came from a KTransformers-based baseline with poor overlap. For a system that already overlaps, expect far less.
- Output-changing overlap (Expert Deferral) is out of scope for an exact-routing speed-of-light race. It should appear only as a disclosed, separate row.

### Gaps
- No public 2024–2026 microbenchmark was found for (a) graph host-node or `cudaLaunchHostFunc` wake-to-completion latency, (b) `cuStreamWaitValue32` on host memory, or (c) a GPU↔CPU mapped-memory ping-pong on PCIe Gen4/Gen5. The system's planned host-node ablation would be the first such number for MoE.
- I found no measurement of how much SM occupancy a spinning wait kernel costs co-running GEMV kernels on Blackwell.

## Q4. Prediction and prefetch: accuracy, realized decode speed-ups, and how much of the Belady gap is future knowledge

### Takeaway
Next-layer expert prediction is accurate: about 90% recall@k beyond the first layers on Qwen3-30B-A3B, 97% with confidence-thresholded prefetch in Fate, and >90% top-(k+3) recall across steps in SeqMoE. Realized decode gains are small wherever PCIe per layer is the bottleneck:
- 5–14% TPOT (Speculating Experts);
- 1.15× in HybriMoE's ablation;
- 1.09× over LRU in llama.cpp (ProMoE);
- 0.3% at 64.7% recall and 5.0% with *perfect* one-layer advice (2608.12103).

Layer compute hides only about 1.4 expert transfers per layer on a 4090 over PCIe 4 (SeqMoE). DALI found that prefetching exactly one expert is best.

For caching, 84–97% of the gap between the best causal policy and Belady is knowing which resident expert will be used last (2608.07911). A learned next-use predictor made misses worse. Prediction is therefore a weak one-week lever for a CPU-miss hybrid.

### Cited Findings
- 2608.07911 (Qwen3-30B-A3B at 40% residency):
  - Best causal policy misses 18.01% vs Belady 9.93%, a 44.85% gap at B = 8 (50.42% at B = 2).
  - "bypass admission accounts for 15.7% of the gap at B=8 and 3.4% at B=2, leaving 84.3% and 96.6% to future-victim knowledge".
  - A trained next-use-distance predictor "worsens misses by 11.4%" and picks the optimal victim 3.39% of the time vs 22.1% for LFRU (per companion note).
  - — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)
- SeqMoE (11 Sep 2026):
  - "on Qwen3-30B-FP8 with an RTX 4090 over PCIe 4.0, one layer's computation overlaps the transfer of only 1.39 experts on average".
  - Top-11 recall >90% for k = 8. 96.97% average hit rate at 45% residency.
  - "Despite an average miss rate of only 3.25%, on-demand loading remains the largest [overhead] component". Predictor inference is 3.46% of time.
  - — [arXiv 2609.12978](https://arxiv.org/abs/2609.12978)
- DALI: "prefetching only one expert—the one with the highest predicted workload—yields the best performance … as more experts are prefetched, the computation time becomes insufficient to overlap the communication cost". — [arXiv 2602.03495](https://arxiv.org/abs/2602.03495)
- Speculating Experts (A6000/A100/GH200):
  - Recall@k "of approximately 90% on average" beyond the first two layers on Qwen3-30B-A3B.
  - "5-14% reduction in time per output token (TPOT) over on-demand CPU expert" loading.
  - Its model bounds the gain by min(t_copy, t_compute): "The maximum achievable speedup is 2×".
  - Transfer was 84–88% of TPOT.
  - — [arXiv 2603.19289](https://arxiv.org/abs/2603.19289)
- Si et al., kernel-managed tiering (2608.12103): "At 64.7% measured recall, router lookahead changes median time by 0.3% … perfect one-layer advice gains 5.0% through the same interface". — [arXiv 2608.12103](https://arxiv.org/abs/2608.12103)
- Fate: cross-layer gate-input prefetch reaches "prefetch accuracy of 97.15%" when transferring experts above the 75th percentile of confidence. The lowest decode accuracy is 76.94%. — [arXiv 2502.12224](https://arxiv.org/abs/2502.12224)
- HybriMoE decode ablation: prefetching alone gives 1.15× over its baseline. — [arXiv 2504.05897](https://arxiv.org/abs/2504.05897)
- ProMoE inside llama.cpp: 1.09× decode over an LRU cache (1.49× over static). — [arXiv 2410.22134](https://arxiv.org/abs/2410.22134) (via companion note academic_systems.md)
- MoE-Beyond: 97.5% accuracy; hit rate 17% → 72% at 10% GPU residency (DeepSeek-V2-Lite). — [arXiv 2508.17137](https://arxiv.org/abs/2508.17137) (via companion note)

### Inferences
- **Breakeven reuse of a fill vs CPU execution on race hardware.** This is my arithmetic, ignoring overlap. Fill cost is expert bytes ÷ PCIe bandwidth; the saving per reuse is (CPU service − GPU hit service), with the GPU hit at 50% of 1,792 GB/s.

  | Expert | PCIe | Host BW | Breakeven reuses (21 µs fixed) | (5 µs fixed) |
  |---|---|---|---|---|
  | gpt-oss, 13.25 MB | Gen5, 52 GB/s | 276 GB/s | 4.7 | 6.7 |
  | Qwen3-30B Q4_K_M, 2.86 MB | Gen5, 52 GB/s | 276 GB/s | 1.9 | 4.5 |
  | Qwen3-30B Q4_K_M, 2.86 MB | Gen4, 25 GB/s | 150 GB/s | 3.1 | 5.5 |

  With a fast server CPU a fill must be reused several times to pay off. Cutting the CPU fixed cost (Q2) raises the threshold further. Prediction's best use is therefore for **admission and bypass** of fills, not aggressive prefetch. The admission threshold (the system's r*) must be retuned whenever the fixed cost or PCIe generation changes.
- Prefetch only helps when PCIe is idle and off the critical path. With admission fills already sharing the copy engine (Q5), adding prefetch traffic can hurt: DALI's "Random" prefetch was worse than none. A one-expert-per-layer prefetch of the top predicted miss is the most that literature supports.
- 2608.07911's decomposition says admission-only (bypass) tuning captures at most 3–16% of the causal-to-Belady miss gap. Closing the rest needs future knowledge that learned predictors have not delivered. Treat "better hit rate" as a multi-week research lever, not a one-week engineering one.

### Gaps
- No paper reports prediction-driven gains for a system whose misses run on a **server-class CPU at ≥150 GB/s** over **PCIe Gen5**. All realized speed-ups are on Gen4 consumer setups or GH200.
- 2608.12103's 5.0% "perfect one-layer" figure comes from a kernel-readahead (page-cache) interface, not a CUDA-copy interface. Transfer to this system is an inference.
- I did not find how much of the *time-domain* gap (as opposed to miss-count gap) future knowledge explains. This remains the project's own open question.

## Q5. PCIe fills: effective host-to-device bandwidth, pinned vs pageable, copy engines, zero-copy kernels, Gen4 vs Gen5

### Takeaway
Pinned H2D reaches about 25 GB/s on Gen4 ×16 and 49–53 GB/s on Gen5 ×16 (FreeToken on RTX 5090/PRO 6000; H20). Pageable transfers can be 3–4× slower. On GPUs with one copy engine per direction, one large copy blocks a later small one in another stream. NVIDIA's advice is small chunks or **custom copy kernels over pinned memory**.

SeqMoE's design addresses the system's "admission copies serialize with critical-path transfers" problem directly:
- critical on-demand loads run as an **SM-driven copy kernel from `cudaHostAllocMapped` memory**, in the graph;
- background fills use `cudaMemcpyAsync` on a separate stream;
- the GPU raises a host-visible flag that **pauses new prefetch submissions** while an on-demand load is in flight.

### Cited Findings
- FreeToken measured B_P (pinned expert-transfer bandwidth): RTX 5090 Gen5 ×16 52.7 GB/s (server) and 49.0 (desktop); PRO 6000 Gen5 ×16 51.5; RTX 4090 Gen4 ×16 25.1; RTX 3090 Gen4 ×16 25.3; 4060 Laptop Gen4 ×8 11.8. The text cites "PCIe 5.0 x16, ∼60 GB/s" and "PCIe 4.0 x16, ∼25 GB/s". — [FreeToken, arXiv 2608.16157](https://arxiv.org/abs/2608.16157)
- Multipath Memory Access (Tencent/Hunan, arXiv 2512.16056 v2, May 2026; dual EPYC 9654 + 8× H20, Gen5 ×16): native H2D about 53 GB/s. Relaying through peer GPUs over NVLink reaches 245 GB/s; the multipath gain starts at ~10 MB transfers, with optimal chunks of ~2.8 MB (H2D). — [arXiv 2512.16056](https://arxiv.org/html/2512.16056v2)
- Pageable vs pinned (hardware unspecified in the README, so treat as illustrative): at 256 MB, pinned H2D 17.0 GB/s vs pageable 4.5 GB/s (3.8×). Large pinned H2D 16.9–17.4 GB/s. — [ak811/cuda-h2d-d2h-bandwidth](https://github.com/ak811/cuda-h2d-d2h-bandwidth)
- Copy-engine head-of-line blocking (NVIDIA forum; RTX 3070 Mobile with `asyncEngineCount == 1`):
  - A large D2H in one stream delayed a small H2D in another by 20 ms.
  - Robert Crovella (NVIDIA) advised limiting transfer sizes (around a 10 KiB threshold, undocumented) or using "custom copy kernels using pinned memory".
  - — [NVIDIA forum](https://forums.developer.nvidia.com/t/cudamemcpyasync-htod-and-dtoh-blocking-each-other/290917)
- SeqMoE runtime, verbatim:
  - "For on-demand loading, we allocate the host expert pool with cudaHostAllocMapped. A custom kernel on the main thread's CUDA stream copies missing experts from host to GPU memory, avoiding host synchronization while remaining graph-capturable."
  - "when a cache miss triggers on-demand loading, the GPU sets a shared flag to pause new prefetch submissions, reducing bandwidth contention."
  - "the CPU prefetcher initiates host-to-device DMA using cudaMemcpyAsync on a dedicated CUDA copy stream."
  - — [arXiv 2609.12978](https://arxiv.org/abs/2609.12978)
- FreeToken fuses each step's fills into "a single, fused transfer" driven by a device-resident source/destination list, "yield[ing] few kernel launches, high PCIe utilization". — [arXiv 2608.16157](https://arxiv.org/abs/2608.16157)
- Zero-copy GPU compute on host-resident experts (llama.cpp feature request #26448, Aug 2026, RTX 4090 over Gen4): Qwen3.6-35B-A3B at 46.8 tok/s with 2.1 GB VRAM; DeepSeek-V2-Lite Q2_K at 37.5 tok/s (vs 65 tok/s all in VRAM). Not merged; community claim. — [llama.cpp #26448](https://github.com/ggml-org/llama.cpp/issues/26448)
- GH200: CPU→GPU 397 GB/s (nvbandwidth, NVLink-C2C), the non-PCIe reference point. — [Discussion #18005](https://github.com/ggml-org/llama.cpp/discussions/18005)

### Inferences
- Fixes for admission copies that serialize with critical-path transfers, in increasing cost:
  1. Keep every critical-path transfer off the copy engine. Activations out, results back and any on-demand weights should be read and written by kernels from mapped pinned memory. The mailbox likely does this for activations and results already. Verify with nsys that no `cudaMemcpyAsync` sits on the critical stream.
  2. Split admission fills into ~1–4 MB chunks (FreeToken/MMA-scale chunks keep ~full link rate). Then a critical DMA waits at most one chunk: about 40–80 µs at 25–50 GB/s rather than a whole 13 MB expert (~250–530 µs).
  3. Gate admissions with a GPU-written "critical transfer pending" flag (SeqMoE) or issue them only after the layer's CPU hand-off completes.
  4. If the GPU exposes ≥2 H2D-capable engines, put admissions on a separate stream so they land on a different engine. This depends on hardware; check `asyncEngineCount`.
- Gen5 doubles fill bandwidth (≈25 → ≈50 GB/s), while the fixed CPU cost stays at 21 µs. On a PRO 6000 or 5090 host, fills become relatively more attractive and the admission threshold should be re-derived (Q4 table).

### Gaps
- Copy-engine counts, and whether multiple H2D streams use separate engines, for the RTX 5090 and RTX PRO 6000 Server Edition were not found. Measure `asyncEngineCount` on the rented box.
- I found no source confirming whether CUDA stream priorities affect copy-engine scheduling. I believe they apply to kernels only, but that is unverified here.
- I found no measured bandwidth for SM-driven zero-copy kernels reading mapped host memory on Gen5 (SeqMoE does not report its copy kernel's GB/s).

## Q6. Synthesis: for a system at ~50–60% of speed-of-light, which 3–5 techniques promise the largest gain, and which fit into one week in llama.cpp?

### Takeaway
Moving from A10 to Blackwell plus a 30–60-core server CPU shrinks every bandwidth term by 2.5–3×, but the fixed terms do not shrink:
- 21 µs per CPU expert;
- about 1–5 µs per GPU kernel node;
- the hand-off.

My illustrative model, using the A10-fitted parameters on gpt-oss-120b with an RTX 5090, predicts the system would fall from 47–64% to roughly **39–46% of speed-of-light**. The biggest one-week levers, in order:
1. **Cut the CPU per-expert fixed cost** (batch a layer's misses into one or two fused, work-stealing jobs): about +10–20%.
2. **Raise CPU streaming efficiency** (NUMA/CCD placement, thread sweep, huge pages): about +10–20%.
3. **Cut GPU per-layer node count** (upstream fusions plus folding cache/mailbox kernels into neighbours): about +5–19%.
4. **Take critical-path transfers off the copy engine and chunk or gate admissions**: removes stalls whose size must be measured first.
5. **Retune the admission threshold for the new bandwidth ratios**: hours of work, 0–10%.

Prediction/prefetch, PDL and persistent megakernels are real but multi-week.

### Cited Findings
- Fixed-cost evidence:
  - Barrier/sync is 15–45% of small-matrix CPU time. — [Discussion #11733](https://github.com/ggml-org/llama.cpp/discussions/11733)
  - One serial op cost 9% of decode. — [ik_llama.cpp #2406](https://github.com/ikawrakow/ik_llama.cpp/issues/2406)
  - Graph-node launch is 0.8 µs on B200. — [MPK](https://arxiv.org/abs/2512.22219)
  - Grouped GEMMs reach "only 17–22% of peak DRAM" at batch 1. — [MonoMoE](https://arxiv.org/html/2609.04244v1)
  - "per-layer fixed costs" dominate at 4 TB/s. — [paper/numbers4.tex (repo)](/home/claude/moe-speed-of-light/paper/numbers4.tex)
- Fusion and thread-pool prior art:
  - KTransformers' "two fused batches". — [SOSP'25](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
  - kt-kernel spin plus atomic work stealing. — [worker_pool.cpp](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/cpu_backend/worker_pool.cpp)
  - llama.cpp's per-expert chunk counters with no inter-expert barrier. — [ggml-cpu.c](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/ggml-cpu.c)
  - ik_llama.cpp fused up/gate. — [common.cpp](https://github.com/ikawrakow/ik_llama.cpp/blob/main/common/common.cpp)
- CPU efficiency prior art:
  - About 80% of likwid at a tuned thread count on Grace, collapsing at full core count. — [Discussion #18005](https://github.com/ggml-org/llama.cpp/discussions/18005)
  - NUMA-aware TP up to 1.63×. — [KTransformers](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
  - About 100 GB/s per Turin CCD. — [Chips and Cheese](https://chipsandcheese.com/p/amds-epyc-9355p-inside-a-32-core)
- GPU fusion prior art:
  - +27–42% tg on MoE on RTX 5090 from six fusion PRs. — [Discussion #17621](https://github.com/ggml-org/llama.cpp/discussions/17621)
  - PDL attempt not merged. — [PR #15480](https://github.com/ggml-org/llama.cpp/pull/15480)
  - Megakernels 1.14–1.58×. — [MPK](https://arxiv.org/abs/2512.22219); [Cohere](https://cohere.com/blog/megakernels); [MonoMoE](https://arxiv.org/html/2609.04244v1)
- Transfer prior art:
  - SM-copy kernel for on-demand loads plus a prefetch-pause flag. — [SeqMoE](https://arxiv.org/abs/2609.12978)
  - Chunking or copy kernels. — [NVIDIA forum](https://forums.developer.nvidia.com/t/cudamemcpyasync-htod-and-dtoh-blocking-each-other/290917)
  - Fused fill lists. — [FreeToken](https://arxiv.org/abs/2608.16157)
- Race competitor datum: FreeToken on an RTX PRO 6000 box (Xeon 8559C, B_H 178 GB/s, B_P 51.5 GB/s) served GLM-5.2 at 14.9 tok/s vs llama.cpp's 7.3. The RTX 5090 is its main platform. — [FreeToken](https://arxiv.org/abs/2608.16157)

### Inferences
**Illustrative sensitivity model.** This is my inference, not a measurement.
- Setup: gpt-oss-120b MXFP4 on an RTX 5090 (1,792 GB/s). Per layer: dense GPU work, then max(GPU hits, CPU misses + 5 µs hand-off). A10-fitted efficiencies: GPU 38%, CPU 60%, 21 µs per CPU expert. Host at 276 GB/s effective (60% of a 460.8 GB/s datasheet); also a 150 GB/s VM-like case.
- Speed-of-light uses datasheet bandwidths with perfect overlap.
- It ignores KV reads, admission traffic and spin occupancy.

| Case (host BW, hit rate) | Baseline, % of SoL | Fixed 21 → 5 µs | GPU 38% → 60% | CPU 60% → 80% | +0.10 hit rate | All three efficiency levers |
|---|---|---|---|---|---|---|
| 276 GB/s, h = 0.3 | 104 tok/s, 40% | 1.20× | 1.10× | 1.14× | 1.12× | 1.63× |
| 276 GB/s, h = 0.5 | 131 tok/s, 39% | 1.18× | 1.14× | 1.13× | 1.15× | 1.62× |
| 276 GB/s, h = 0.7 | 177 tok/s, 39% | 1.14× | 1.19× | 1.10× | 1.20× | 1.60× |
| 150 GB/s, h = 0.5 | 95 tok/s, 45% | 1.12× | 1.09× | 1.18× | 1.18× | 1.53× |

**Ranked technique list**

| Rank | Technique (gap closed) | Expected gain on race hardware | llama.cpp implementation cost | Prior art | Fits in the week? |
|---|---|---|---|---|---|
| 1 | **CPU miss batch as one job per layer.** Build one work list per layer covering all missed experts. Quantize the activation once. Fuse gate+up across experts. Use per-expert completion counters (or at most one barrier) before down. Keep a persistent spinning pool with atomic chunk claiming (~16–64 rows). Write gate-weighted partial sums directly into the mailbox result slot. **Target ≤5 µs fixed per layer, not per expert.** | +10–20% (model). Larger for small-expert models (Qwen3: fixed cost ≈2× byte time) | 1–3 days in the helper-thread code. Microbenchmark first: time vs number of misses, fit intercept and slope. | KTransformers "two fused batches"; kt-kernel work stealing; llama.cpp per-expert chunk counters; ik fused up/gate; fairydreaming barrier breakdown | **Yes** |
| 2 | **CPU bandwidth efficiency.** Sweep thread count (expect the optimum well below core count). Pin helpers per CCD/NUMA node and keep each expert's rows node-local (NPS/SNC-aware slicing). Use 2 MB pages for the expert pool. Avoid SMT siblings. Measure STREAM on the rented host first. | +10–20% if helpers are at 60% of STREAM; ≈0 if already ~80% | 1–2 days (allocation plus affinity; slicing experts across nodes is +1–2 days) | Grace 80% at 32 of 72 threads; KT NUMA TP 1.63×; Turin ~100 GB/s/CCD; ARCLIGHT local vs remote 4× | **Yes** (single-node placement); cross-socket slicing is borderline |
| 3 | **GPU per-layer node diet.** Rebase onto upstream Nov-2025+ fusions (GEMV+GLU, TopK-MoE, fused RMS/add). Run hits as one fused gate/up mmvq-id launch plus one down launch. Fold cache lookup/slot remap and mailbox publish into the top-k kernel, and merge/spin into the weighted-sum kernel. Check the graph node count per layer with nsys. | +5–19% (model: 38% → 60% GPU efficiency gives 1.09–1.19×); upstream pure-GPU fusions gave +27–42% | 2–3 days; kernel fusions touch ggml-cuda | llama.cpp PRs #16715/#16130/#14800/#15631; MPK/Cohere/MonoMoE show the ceiling | **Yes** (fusion); PDL/megakernel **no** |
| 4 | **Copy-engine hygiene.** No `cudaMemcpyAsync` on the critical stream. Split admission fills into 1–4 MB chunks. Add a GPU-written "critical pending" flag that holds admissions. Use a separate engine if `asyncEngineCount` ≥ 2. | Removes admission-induced stalls; size unknown until measured (it was a named A10 gap). Plausibly 2–10% | 1–2 days | SeqMoE (copy kernel plus pause flag); FreeToken fused fill list; NVIDIA forum advice | **Yes** |
| 5 | **Re-derive the admission threshold (r*) and bypass** for Gen5 PCIe and the new CPU fixed cost (Q4 table). | 0–10% | Hours (parameter plus re-sweep) | 2608.12103 breakeven; FreeToken q* | **Yes** |
| 6 | Mailbox micro-tuning (flag-in-data, per-expert streaming merge) | ≤ a few µs per layer, likely <3% | 1–2 days | LL/LL128, 2607.16100; NCCL GIN proxy | Only if 1–5 are done |
| 7 | Next-layer prediction/prefetch (1 expert per layer) | 0–15% in literature (Gen4, fetch-based); likely lower with a fast CPU | 3–5 days | SeqMoE, DALI, HybriMoE, Speculating Experts, Fate | **No** |
| 8 | PDL / persistent MoE kernel / megakernel | 1.14–1.58× on the GPU part (other hardware and formats) | Weeks | MPK, MonoMoE, Cohere, Hazy | **No** |
| — | Expert Deferral (output-changing) | +33% (KTransformers) | — | KTransformers | Exclude from exact-routing race; report separately if at all |

- **Ordering logic.**
  - Items 1–3 attack the terms that do not scale with bandwidth. Their combined effect in the model is about 1.5–1.6×, enough to move a ~40%-of-SoL Blackwell baseline back to ~60–65%. Each can be verified with a null-work or sweep control in hours.
  - Item 4 is ranked by risk rather than expected size: it is a known A10 gap, and Gen5 plus larger fills will make it worse if it is left in place.
- **Measurement first (day 1).** Establish on the rented box:
  - (a) STREAM/likwid host bandwidth per NUMA node and a thread sweep;
  - (b) the per-layer CPU time vs number of misses (intercept = fixed cost);
  - (c) the GPU graph node count and per-layer GPU time with zero misses;
  - (d) `asyncEngineCount` and fill/critical overlap in nsys.

  These four numbers decide whether items 1, 2, 3 or 4 dominate on that hardware.
- **Platform caveat.** gpt-oss-120b MXFP4 (~61 GB) fits entirely in a PRO 6000's 96 GB, so offload is needed there only for larger models (DeepSeek-V4-Flash, GLM-5.2) or an explicit VRAM cap. On the RTX 5090 (32 GB), gpt-oss-120b is the natural offloaded row, and it is FreeToken's and SeqMoE's platform class.

### Gaps
- Every gain in the tables above comes from my analytic model with A10-fitted parameters. None is measured on Blackwell, and the model ignores KV-cache reads, admission traffic, spin-kernel SM occupancy and CPU–GPU memory contention.
- No public source reports a per-expert CPU fixed cost for any hybrid MoE system. The claim that ≤5 µs per layer is achievable is an engineering inference from barrier and launch literature, not a measured result.
- It is unknown whether the system's branch already includes upstream's Nov-2025 CUDA fusions. If it does, item 3's upside shrinks to folding the system's own kernels (+ a few %).
- The rented host's topology (CCDs per VM, NPS/SNC mode, SMT) is unknown and may shift items 2 and 1 in either direction.
