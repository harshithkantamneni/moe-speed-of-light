# Industry-grade CPU/GPU hybrid MoE engines for batch-1 decode (state as of 2026-09-27)

Scope note: "batch-1 decode" = single request, token-by-token generation. Numbers labelled "throughput"/"ShareGPT"/"concurrency N" are batched serving and NOT comparable to batch-1 decode. Source access date for all GitHub code/README reads: 2026-09-27 (KTransformers main at commit c40722b, 2026-09-23; llama.cpp master and SGLang main fetched the same day).

## KTransformers (SOSP 2025 paper + GitHub): mechanisms, placement, synchronization, reported decode speeds

### Takeaway
KTransformers uses static (offline-profiled) GPU expert placement, runs CPU experts with AMX/AVX-512 kernels, uses NUMA-aware tensor parallelism, and fits the whole decode step for one token into a single CUDA graph by issuing CPU "submit" and "sync" through `cudaLaunchHostFunc` stream callbacks (host callbacks, not GPU-signalled polling). Its "Expert Deferral" is a lossy cross-layer overlap trick. Reported batch-1 decode speedups over a Feb-2025, custom-extended llama.cpp are 1.25–1.93x without deferral and 1.66–2.56x with it. It has no per-token dynamic GPU expert cache in the paper.

### Cited Findings
**Paper venue and setup**
- Paper: "KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models", Hongtao Chen et al., SOSP '25, Oct 13–16, 2025, Seoul — [MADSys PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf); [ACM DL](https://dl.acm.org/doi/10.1145/3731569.3764843)
- Evaluation hardware: a dual-socket Intel Xeon Platinum 8452Y (36 physical cores per socket) with 1 TB DDR5. Intel MLC measures 220 GB/s intra-socket and 125 GB/s cross-socket bandwidth. GPUs are an NVIDIA A100 40 GB (BF16/FP16 models) and an RTX 4080 16 GB (quantized models), both on PCIe 4.0 (32 GB/s) — [paper PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- Models:
  - DeepSeek-V3-0324 (671B; 17B on GPU / 654B on CPU; 58 MoE layers, 256 routed experts, top-8)
  - DeepSeek-V2.5-1210 (236B)
  - Qwen2-57B-A14B
  - DS-3 is quantized to Int4 on the RTX 4080; DS-2 and QW-2 to Int8.
  - All runs use batch size 1. The decode workload is a 32-token prompt with up to 512 generated tokens.
  - No gpt-oss or Qwen3 results appear in the paper.
  - Source: [paper PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- Baselines are Fiddler and llama.cpp, both run "Fiddler-style" with routed experts on the CPU. Quotes from the paper:
  - "Since Llama.cpp originally only supports layer-wise offloading and lacks expert-level offloading capability … we extended Llama.cpp with custom code to enable expert-level offloading."
  - llama.cpp ran FP16 because it lacks BF16 CUDA kernels.
  - The llama.cpp reference is "Retrieved Feb 8, 2025".
  - Source: [paper PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)

**CPU kernels and NUMA**
- AMX kernel (prefill): uses an AMX-tile-aligned memory layout, 64-byte cache-line alignment and group-wise Int8/Int4 quantization. It reaches 21.3 TFLOPS on a single socket, 3.98x the oneDNN-based PyTorch baseline — [paper PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- AVX-512 kernel (decode):
  - AVX-512 beats AMX when the arithmetic intensity is ≤4 tokens per expert. KTransformers switches between the two dynamically: up to 1.20x in decode vs pure AMX, and up to 10.81x in prefill vs pure AVX-512.
  - In the decode breakdown, the AVX-512 kernel alone gives 2.22x over the Fiddler base.
  - Source: [paper PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- NUMA-aware tensor parallelism splits every expert's weights across sockets (reduce-scatter) instead of using expert parallelism. It gives up to 1.63x in decode vs a NUMA-oblivious baseline on the dual-socket machine — [paper PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)

**CPU–GPU hand-off and CUDA graph**
- The paper's description, quoted:
  - "A naive design must synchronize twice per MoE layer, once when the GPU sends activations to the CPU (submit) and again when the CPU returns its results (sync). These barriers break CUDA Graphs…"
  - "KTransformers encapsulates both submit and sync in cudaLaunchHostFunc, which lets CUDA invoke the callbacks inside the current stream. The entire decode path for one token now fits in a single CUDA Graph… improving decoding speed by up to 1.23x."
  - Source: [paper PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- The paper's introduction describes the same thing differently: "encapsulates the entire decode phase into a single CUDA Graph instance by leveraging CUDA-based spinning". The intro therefore says "spinning" while §3 says `cudaLaunchHostFunc` — [paper PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- Current code agrees with §3:
  - `CPUInfer::submit_with_cuda_stream` calls `cudaLaunchHostFunc(stream, func, args)` to enqueue CPU work.
  - `sync_with_cuda_stream` calls `cudaLaunchHostFunc(stream, &sync_, args)`, where `sync_` blocks on `task_queue_->sync(allow_n_pending)`.
  - Worker threads live in a `WorkerPool` and `TaskQueue`.
  - Source: [kt-kernel cpu_backend/cpuinfer.h](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/cpu_backend/cpuinfer.h)
- The Python side notes "graph MoE uses `_launch_host_func` + pinned buffers" during graph capture — [kt-kernel python/experts_base.py](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/python/experts_base.py)
- Within a layer, "a CPU control thread (i) pushes routed-expert tasks into a lock-free queue and (ii) launches GPU kernels for the shared experts". This means the shared experts on the GPU overlap with the routed experts on the CPU — [paper PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- Launch-overhead motivation, profiled with DeepSeek-V3 on an A100:
  - Fiddler issues more than 7,000 kernel launches per token at about 16 µs each, which is 73% of GPU time.
  - llama.cpp issues about 3,000 launches per token at about 5 µs each, which is 21% of GPU time.
  - The paper says "Llama.cpp disables CUDA Graph optimizations to avoid repeated capture overhead". This refers to the Feb-2025 llama.cpp.
  - Source: [paper PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)

**Expert placement**
- Placement is static. "Attention layers … are prioritized on the GPU, followed by frequently-used experts identified through offline profiling". Shared experts go on the GPU and routed experts on the CPU. The paper describes no runtime or dynamic expert caching — [paper PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)

**Expert Deferral**
- How it works:
  - Routed experts are split into "immediate" experts, consumed by the next layer's attention, and "deferred" experts, whose output is added one layer later (consumed at layer k+2).
  - The deferred experts' CPU work then overlaps with the next layer's GPU attention.
  - It applies to decode only. In prefill, near-full expert coverage "nearly doubles memory access footprints".
  - Source: [paper PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- Configurations and utilization:
  - DS-3 BF16 uses 5 immediate + 3 deferred experts; the Int4 model defers 6. DS-2 defers 4. QW-2 defers 2 (BF16) or 4 (Int8).
  - The heuristic is to defer the minimum number that saturates the CPU while keeping ≥2 immediate experts.
  - On DS-3, CPU/GPU utilization rises from 74%/28% to 100%/37%, giving +33% decode throughput.
  - The largest gain is up to 45%; the headline figure is "up to 1.45x".
  - Source: [paper PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- Accuracy:
  - The average drop is "within 0.5%". On LiveBench with 6 deferred experts, deferral loses 0.5% vs 13.3% for Expert Skipping.
  - Table 2 examples: DS-3 (8+0) vs (2+6) scores HumanEval 83.0/83.0, MBPP 71.2/70.2, GSM8K 94.8/95.2.
  - Source: [paper PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)

**Reported batch-1 decode results**
- DeepSeek-V3 decode without Expert Deferral "remains limited to 5.87 tokens per second" (BF16, A100 configuration) — [paper PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- Speedups without deferral:
  - Full precision: 2.42–4.09x over Fiddler and 1.25–1.76x over llama.cpp.
  - Quantized: 1.77–1.93x over llama.cpp.
  - GPU invocation overhead falls "from over 20% … to almost zero".
  - Source: [paper PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- With Expert Deferral: 1.66–2.56x over llama.cpp. The abstract gives 1.66–4.90x over "existing methods" in general and 1.25–4.09x decode without deferral — [paper PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- Figure 12's y-axes top out at 25 tok/s for BF16 on the A100 and 40 tok/s for quantized models on the RTX 4080. Exact bar values are not in the text — [paper PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- Decode breakdown vs the Fiddler base: +AVX-512 gives 2.22x, +NUMA-aware up to 1.63x more, +CUDA Graph 1.23x more — [paper PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- Earlier README numbers (Feb 2025, v0.2/v0.3-preview), from [KTransformers DeepSeek-R1/V3 tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md):
  - Hardware and model: DeepSeek-V3/R1 Q4_K_M on 2x Xeon Gold 6454S (32 cores per socket, 1 TB DRAM) with an RTX 4090/4090D.
  - Decode progression: 8.73 tok/s (32 cores) → 11.26 (dual-socket) → 13.69 ("selectively using 6 experts", lossy), vs llama.cpp 4.51 tok/s on 2x32 cores, "up to 3.03x".
  - V0.2.1 table: 8 experts decode 13.4 tok/s with a 1K prompt and 12.4 with an 8K prompt; 6 experts decode 15.9 (1K).
- Production status per the paper: 11,000 lines of C++ plus 2,000 lines of Python, "operates on hundreds of machines" — [paper PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)

**2026 model support**
- README news covers Kimi-K2.5 (Jan 27, 2026), GLM-5 (Feb 12), MiniMax-M2.5 (Feb 13), AVX2-only CPU backend (Mar 26), v0.6.1 (Apr 30), DeepSeek-V4-Flash (May 2), GLM-5.2 (Jun 17), MiniMax-M3 (Jun 21) and GLM-5.3-flash (Aug 26, 2026) — [KTransformers GitHub](https://github.com/kvcache-ai/ktransformers)

### Inferences
- KTransformers' CPU hand-off is host-callback-driven:
  - The CUDA driver's callback thread runs `submit`/`sync`, and the stream blocks inside the `sync` callback until the CPU workers finish.
  - This differs in mechanism from a GPU-written mailbox with a device-side spin-wait kernel.
  - But the goal claimed for the work under review (one CUDA graph per token with CPU experts inside it) was already published and shipped by KTransformers in 2025.
- The paper's llama.cpp baseline is a custom-patched build from Feb 2025. It predates mainline `--n-cpu-moe` and the current per-split CUDA-graph handling (see the llama.cpp section). Its 1.25–1.93x "over llama.cpp" is therefore not a comparison against today's llama.cpp.
- The paper's best decode numbers depend on dual-socket NUMA (up to 1.63x) and on lossy Expert Deferral. Neither applies to a single-NUMA 30-vCPU VM with exact computation.

### Gaps
- Exact per-model decode tok/s from Figure 12 (only bar charts; the text gives ranges plus 5.87 tok/s).
- The per-call latency of KTransformers' `cudaLaunchHostFunc` submit/sync path is not reported in the paper; no source was found.
- No KTransformers batch-1 decode numbers were found for gpt-oss-20b/120b or Qwen3-30B-A3B.
- No gpt-oss model support was found in kt-kernel docs or source. A grep of kt-kernel found MXFP4 MoE kernels, documented for DeepSeek-V4-Flash, and a `swiglu_limit` path, but no "gpt-oss" string.

## SGLang + KTransformers integration (kt-kernel, LMSYS blog Oct 2025) and vLLM CPU/expert offload

### Takeaway
SGLang+kt-kernel (Oct–Nov 2025) brings the KTransformers CPU kernels, a CUDA-graph-compatible submit/compute/sync hybrid MoE layer (CPU experts run concurrently with GPU experts in the same layer) and Expert Deferral into a serving engine. Its only "dynamic" expert placement re-plans GPU experts during long layerwise prefills, not per decode token. vLLM mainline offers static CPU weight offload only. Its dynamic expert-cache RFC fetches misses to the GPU instead of computing them on the CPU, and CPU-computed experts exist only as a draft, out-of-tree effort.

### Cited Findings
**SGLang + KTransformers**
- The LMSYS blog "Accelerating Hybrid Inference in SGLang with KTransformers CPU Kernels" (Oct 22, 2025) lists AMX/AVX-512 switching, NUMA-aware tensor parallelism ("up to 63% decoding throughput improvement on dual-socket servers"), CUDA Graph integration ("kernel-launch overhead from over 20% to nearly zero") and Expert Deferral ("up to 1.45x higher decoding throughput, with accuracy variation below 0.5%") — [LMSYS blog](https://www.lmsys.org/blog/2025-10-22-KTransformers/)
- LMSYS numbers, which are relative or batched:
  - Single-GPU + CPU: "up to 20x" prefill and "up to 4x" decode (the paper's numbers).
  - 8x L20 + Xeon Gold 6454S, int4 DeepSeek-V3: single concurrency is +26% for 8 GPUs vs 1 GPU; 8-way concurrency gives a "264% throughput gain".
  - ShareGPT with 1000 requests on DeepSeek-R1-0528 FP8 (batched): 227.85 total tok/s and 87.58 output tok/s, median ITL 299.18 ms, P99 ITL 1935.13 ms.
  - Source: [LMSYS blog](https://www.lmsys.org/blog/2025-10-22-KTransformers/)
- SGLang PR #12586 "Support Expert Deferral Mechanism in KTransformers" was merged Nov 5, 2025. It adds `--kt-max-deferred-experts-per-token`, and the core logic lives in kt-kernel — [SGLang PR #12586](https://github.com/sgl-project/sglang/pull/12586)
- The SGLang `KTEPWrapperMethod` implements "submit-compute-sync":
  - (1) submit the CPU expert computation, non-blocking, on the current CUDA stream;
  - (2) mask CPU expert IDs to -1 and run the GPU experts with any GPU MoE kernel "in parallel";
  - (3) sync the CPU results and add them to the GPU output.
  - GPU experts are those with ID < `num_gpu_experts`.
  - Source: [sglang kt_ep_wrapper.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/kt_ep_wrapper.py)
- kt-kernel flags:
  - `--kt-num-gpu-experts` sets GPU experts per layer.
  - `--kt-max-deferred-experts-per-token` takes 0 (synchronous), 1–4 (recommended) or 5–7 ("may introduce noticeable accuracy loss").
  - `--kt-threadpool-count` equals the number of NUMA nodes.
  - `--kt-method` takes AMXINT4/AMXINT8/RAWINT4/FP8/FP8_PERCHANNEL/BF16/LLAMAFILE.
  - Source: [kt-kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)
- Placement and dynamic update:
  - Initial placement strategies are `uniform` (default), `frequency` (needs an offline activation-statistics .pt), `front-loading` and `random`.
  - `--kt-enable-dynamic-expert-update` works as follows: "During layerwise prefill, the system collects actual routing statistics and redistributes GPU experts accordingly". It "Requires `--kt-gpu-prefill-token-threshold`… and prefill length must be ≥ the threshold value".
  - Source: [kt-kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)
- The docs say dynamic expert update "may be helpful for workloads with skewed routing distributions, but it is workload-sensitive" — [KTransformers docs: Expert Placement](https://ktransformers.net/zh/docs/optimization-techniques/expert-placement)
- Expert-scheduling benchmark (batched ShareGPT throughput, not batch-1), on Qwen3-Next-80B-A3B-Instruct-FP8, 4x RTX 4090 TP=4, Xeon Gold 6454S, 512 GB DDR5. Throughput in tok/s by strategy — [KT expert-scheduling tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/experts-sched-Tutorial.md):

  | GPU expert ratio | uniform | frequency | dynamic-expert-update |
  |---|---|---|---|
  | 0% | 52.96 | 52.72 | 53.37 |
  | 10% | 56.57 | 58.60 | 70.22 |
  | 30% | 62.08 | 66.50 | 75.55 |
  | 50% | 65.25 | 76.19 | 81.17 |
  | 90% | 81.06 | 107.15 | 95.04 |
  | 100% | 112.32 | 114.26 | 112.99 |

- KTransformers leaderboard decode TPS:
  - Qwen3.5-35B-A3B FP8: 86.2 tok/s on 1x RTX 5090, 93.2 on 2x, 97.5 on 4x.
  - MiniMax-M2.1 FP8: 38.5 on 1x RTX 5090.
  - Kimi-K2.6 RAWINT4: 29.1 on 4x RTX 5090.
  - DeepSeek-V3.2 FP8: 17.3 on 8x RTX 5090.
  - The page gives no CPU/RAM, GPU-expert fraction, concurrency or dates.
  - Source: [KTransformers benchmark leaderboard](https://ktransformers.net/en/benchmarks)

**vLLM**
- vLLM RFC #38256 "Incremental MoE Expert Offloading — GPU Cache + Async Pipeline" (e1n00r, Mar 26, 2026):
  - Design: a GPU cache of hot experts backed by pinned CPU memory, with LFRU eviction plus cross-layer prediction. Misses are fetched over PCIe to the GPU: "No CPU fallback computation."
  - Result: 30 tok/s single-stream for GPT-OSS-20B MXFP4 on an RTX PRO 2000 8 GB, with a claimed 97–100% hit rate.
  - PR 1 is #37190 (~980 lines of Python, synchronous H2D). PR #34535 (merged) is "Static CPU weight offload", where offloaded weights stay on the CPU permanently.
  - Source: [vLLM RFC #38256](https://github.com/vllm-project/vllm/issues/38256)
- vLLM has a UVA-based CPU offloader. PR #54610 is titled "[Bugfix][Offloader] Prioritize sparse MoE experts in UVA CPU offloading" (title only; not read in detail) — [vLLM PR #54610](https://github.com/vllm-project/vllm/pull/54610)
- vLLM PR #56118 "add VLLM_EXPERTS_LOAD_DEVICE=cpu for GPU/CPU mixed expert placement" (Sept 9, 2026, draft):
  - It uses static placement of routed experts on the host.
  - Quote: "The CPU path is what this PR enables"; the actual CPU expert compute is in an out-of-tree package (vllm-xtu-moe).
  - Only prefill numbers are given (DeepSeek-V4-Flash, 2x A100 + EPYC 9654); there are no batch-1 decode numbers.
  - Source: [vLLM PR #56118](https://github.com/vllm-project/vllm/pull/56118)
- The LMCache "10x MoE" blog (Apr 3, 2026) is about sharing KV cache across data-parallel ranks on 8x H100, not expert offload. It is not relevant to hybrid expert decode — [LMCache blog](https://blog.lmcache.ai/en/2026/04/03/lmcaches-new-architecture-boosts-moe-inference-performance-by-10x/)

### Inferences
- The SGLang+KT combination already has (b) a graph-capturable CPU hand-off (host-callback based) and (c) intra-layer CPU||GPU concurrency (CPU routed experts alongside GPU experts or shared experts).
- It lacks (a) a per-decode-token dynamic cache. Its "dynamic expert update" re-plans placement from prefill statistics only when the prefill is at least the threshold length.
- The only published dynamic-vs-static data (ShareGPT, 4x4090) shows the dynamic update helping most at 10–50% GPU expert ratios (up to +24% over `uniform` at 10%). It trails `frequency` at ≥80%.
- vLLM is not a batch-1 hybrid engine in mainline. Its caching proposal is a PCIe-streaming cache, a different design point from CPU-executed misses.

### Gaps
- No SGLang+KT batch-1 decode numbers were found for gpt-oss-120b, gpt-oss-20b or Qwen3-30B-A3B on a single consumer or datacenter GPU with a known CPU.
- Whether the kvcache-ai fork (`sglang-kt`) differs from upstream SGLang in the wrapper, for example in how it applies the GPU expert mask for non-contiguous expert IDs, was not verified.
- vLLM RFC #38256/PR #37190 merge status as of Sept 2026 is unknown; the fetched page showed no maintainer responses.

## ik_llama.cpp: MoE hybrid optimizations and reported speedups vs mainline

### Takeaway
ik_llama.cpp contributed many hybrid-MoE features first: tensor overrides `-ot`, fused MoE FFN `-fmoe`, run-time repack `-rtr`, control over which RAM-resident ops are offloaded, "offload only activated experts" for prompt processing, and auto-fit. Its documented wins are mostly prompt processing and full-GPU or CPU-only token generation. No GPU expert cache, CPU/GPU intra-layer overlap or GPU-initiated hand-off feature was found. Evidence for hybrid token generation beating mainline is mixed, and some 2026 reports show ik slower.

### Cited Findings
- Fork history: ik_llama.cpp forked llama.cpp in June 2024 and "was last synced with upstream in August of 2024" — [ik_llama.cpp README](https://github.com/ikawrakow/ik_llama.cpp/blob/main/README.md)
- Hybrid-relevant features and dates:
  - Fused FFN ops for MoE (`-fmoe`), PR 229, Feb 23 2025.
  - Tensor overrides (`-ot`), PR 232, Feb 25 2025.
  - Smart Expert Reduction (fewer active experts, lossy), PR 239, Mar 1 2025.
  - "Better TG performance for MoE models on CUDA", PR 248, Mar 10 2025.
  - User control over "if/which operations with tensors held in RAM are offloaded to the GPU", PR 405, May 12 2025.
  - "Better GPU offload strategy for MoE models when using hybrid GPU/CPU inference", PR 520.
  - Split mode "graph" for multi-GPU, PR 1022.
  - Auto-fit offloaded tensors to VRAM, PRs 1501/1504.
  - Source: [ik_llama.cpp README](https://github.com/ikawrakow/ik_llama.cpp/blob/main/README.md)
- README warnings:
  - For hybrid MoE "do not use -rtr unless you know what you are doing". Repacked RAM tensors force the matmuls to run "always … on the CPU", typically lowering prompt processing speed; k-quants have no CUDA row-interleaved implementation.
  - With split mode graph plus partial offload (`--cpu-moe`/`--n-cpu-moe`/overrides), users who see gibberish should try `-cuda graphs=0`.
  - Source: [ik_llama.cpp README](https://github.com/ikawrakow/ik_llama.cpp/blob/main/README.md)
- PR #698 "Offload only activated experts to the GPU" (merged Sept 4, 2025):
  - Scope: prompt processing only, above a batch threshold of `32 * total_experts / active_experts`; flag `--offload-only-active-experts`.
  - Results: gpt-oss-120b MXFP4 1.33–1.68x PP (Ryzen-5975WX + RTX-4080); gpt-oss-20b 1.08–1.30x; DeepSeek-Lite roughly 1.0x.
  - No cross-token caching.
  - Source: [ik_llama.cpp PR #698](https://github.com/ikawrakow/ik_llama.cpp/pull/698)
- Discussion #758 "GPT-OSS performance" (Sept 2025, updated Nov 2025), gpt-oss-20b MXFP4:
  - Full GPU on an RTX-4080: TG ~163 vs ~114 t/s at ~30k context (+15%, mainline with `--swa-full`); PP ~6,884 vs ~3,023 t/s.
  - CPU-only on a Ryzen-7950X: TG ~1.5x at 25k context.
  - Nov 2025 update: another +15–25% TG on GPU.
  - No hybrid gpt-oss-120b TG comparison is given.
  - Source: [ik_llama.cpp Discussion #758](https://github.com/ikawrakow/ik_llama.cpp/discussions/758)
- Hybrid report, Qwen3-235B-A22B on an RTX 3090 + Threadripper 2950X (AVX2), 128 GB DDR4-2933:
  - TG ~7.4 t/s, "similar across both" ik and mainline.
  - After removing `-rtr` and `-fmoe`: PP 67.35 t/s, TG 8.19 t/s. The discussion is undated in the fetch, ~2025.
  - Source: [ubergarm HF discussion](https://huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF/discussions/3)
- Issue #1699 (Apr 27, 2026): ik_llama.cpp is slower than mainline for hybrid CPU-MoE on Qwen3.6-35B-A3B IQ4_XS (GTX 1660 Super 6 GB + Ryzen 7 5800X). PP 118.79 vs 239.81 t/s, TG 10.40 vs 15.69 t/s. No maintainer response appeared in the fetch — [ik_llama.cpp issue #1699](https://github.com/ikawrakow/ik_llama.cpp/issues/1699)

### Inferences
- ik_llama.cpp's hybrid-decode path uses the same host-sequenced ggml backend-scheduler model as mainline, with faster kernels and fusions. No evidence was found of dynamic GPU expert caching, device-initiated CPU hand-off, or overlapping CPU expert compute with GPU work in the same layer.
- "ik vs mainline" hybrid TG gains are hardware- and quant-dependent and sometimes negative. They should not be used as a fixed multiplier.

### Gaps
- No credible hybrid (`--n-cpu-moe`/`-ot`) batch-1 TG comparison of ik vs mainline was found for gpt-oss-120b or Qwen3-30B-A3B.
- The ik_llama.cpp source was not checked for a hidden or experimental expert-cache feature; only the README feature list was checked. "On-demand tensor reload" (PR 1989) was not investigated.

## Mainline llama.cpp (2025–2026): --n-cpu-moe/--cpu-moe, CUDA graphs, CPU/GPU overlap, typical gpt-oss-120b tok/s, plus the 2026 community expert-cache PRs

### Takeaway
Mainline llama.cpp (master, Sept 27 2026) places experts statically (`-ot`, `--cpu-moe`, `--n-cpu-moe`, `--fit`). It executes backend splits in order, host-sequenced, with synchronizations between CPU and GPU splits, so there is no CPU/GPU overlap within a layer in batch-1 decode. It captures CUDA graphs per GPU split, not across the CPU work. Its only "dynamic" expert movement is copying just the used CPU-resident experts to the GPU for large prompt-processing batches. In 2026, however, at least four independent community implementations of a dynamic GPU expert cache with CPU-executed misses were posted (an RFC and PRs from June–Aug 2026). They report +7% to 2.2x decode, and none is merged.

### Cited Findings
**Static placement flags and user-facing guides**
- `--n-cpu-moe` existed by Aug 12, 2025. LM Studio 0.3.23's "Force Model Expert Weights onto CPU" uses "the same underlying technology as llama.cpp's --n-cpu-moe" — [LM Studio 0.3.23 blog](https://lmstudio.ai/blog/lmstudio-v0.3.23)
- The Doctor-Shotgun HF guide (Jan 30, 2026) recommends putting all "always active" tensors (attention, dense FFN, shared experts) on the GPU and routed experts on the CPU via `-ot exps=CPU`/`--cpu-moe`/`--n-cpu-moe N`. It explains that for large batches llama.cpp copies the CPU-assigned weights to the GPU to process the prompt (a threshold of ~32 tokens by default in llama.cpp; ik uses an expert-ratio-scaled threshold) — [Doctor-Shotgun HF blog](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide); [gist](https://gist.github.com/DocShotgun/a02a4c0c0a57e43ff4f038b46ca66ae0)

**Scheduler behaviour (master source, Sept 27 2026)**
- In `ggml_backend_sched_compute_splits`, splits run in order. Before a split, it synchronizes the previous backend or waits on events and copies inputs across backends. If `cpy_tensor_async` is unavailable, it falls back to `ggml_backend_synchronize(input_backend)` plus a blocking copy. It then calls `ggml_backend_graph_compute_async(split_backend, …)` — [ggml-backend.cpp (master)](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend.cpp)
- "When offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used": when host-resident expert weights feed a GPU `MUL_MAT_ID` split, the scheduler reads the ids tensor back to the host (with a synchronize) and copies only the used expert ranges. This is the prompt-processing offload path — [ggml-backend.cpp (master)](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend.cpp)

**CUDA graphs (master source, Sept 27 2026)**
- CUDA graphs are keyed per ggml cgraph (`graph_key = cgraph->nodes[0]`), which yields one graph per GPU split. A graph is enabled after a 2-call warmup with unchanged properties. It is disabled if a `MUL_MAT_ID` node needs the synchronizing fallback path (ref PR #18958) — [ggml-cuda.cu (master)](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu)
- A cross-backend event wait through `cudaLaunchHostFunc` is present but `#if 0 // untested`, followed by `GGML_ABORT`. The CUDA backend therefore does not host-callback-wait on CPU events — [ggml-cuda.cu (master)](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu)
- `GGML_CUDA_GRAPH_OPT=1` enables graph reordering and concurrent CUDA streams (for example Q/K/V projections). With full-GPU models this gives:
  - RTX 5090: Qwen3 MoE 30B TG 352.06 vs 246.96 t/s; GPT-OSS-20B 419.14 vs 329.23.
  - RTX 4090: Qwen3 MoE 30B 271.04 vs 198.39; GPT-OSS-20B 271.99 vs 232.05.
  - The blog does not discuss hybrid offload and is undated.
  - Source: [am17an blog](https://am17an.bearblog.dev/new-post/); [llama.cpp PR #16991](https://github.com/ggml-org/llama.cpp/pull/16991)

**Community gpt-oss-120b hybrid numbers**
- carteakey.dev (Sept 21, 2025, updated through Mar 12, 2026):
  - Hardware: RTX 4070 12 GB, i5-12600K, 64 GB DDR5-6000 dual-channel.
  - Settings: `--fit on --fit-ctx 32768` (or `-ngl 37` + `-ot` of layers 5+ experts to CPU), `-fa`, `taskset` P-cores, `GGML_CUDA_GRAPH_OPT=1`.
  - Result: TG 25–28 tok/s, PP 427–429 tok/s at 32k context. Before enabling XMP (RAM at 2000 MT/s) it managed 10–11 tok/s.
  - Source: [carteakey.dev](https://carteakey.dev/blog/local-inference/optimizing-gpt-oss-120b-local-inference/)
- Hardware Corner (Nov 10, 2025):
  - RTX 3090 + EPYC 7343 + 64 GB DDR4-3200 with 27 MoE layers on the CPU: TG 1.41–1.81 t/s.
  - RTX 5090 with `--n-cpu-moe 21`: TG 8.14–9.60 t/s (CPU/RAM unspecified).
  - Source: [Hardware Corner](https://www.hardware-corner.net/gpt-oss-offloading-moe-layers/)
- The llama.cpp gpt-oss guide (Discussion #15396) lists full-GPU TG of 161.77 (RTX 3090), 221.95 (RTX 4090) and 186.51 (4080 SUPER) t/s. These are presumably gpt-oss-20b, since 120b does not fit in 24 GB. The main post has no `--n-cpu-moe` hybrid tables — [llama.cpp Discussion #15396](https://github.com/ggml-org/llama.cpp/discussions/15396)

**Community dynamic expert-cache implementations (all unmerged at fetch time)**

**RFC Discussion #24528 "MoE expert cache, VRAM caching of hot CPU-resident experts with hybrid hit/miss execution"** (leloch, June 12, 2026; fork branches `moe-cache-pr`/`moe-cache-v2-pr`, ~1,700 lines in `moe-cache.cu/.cuh`, CUDA-only). Source for all items: [llama.cpp Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)
- Execution model: an "inverted execution model" keeps `MUL_MAT_ID` on the CPU. CPU thread 0 dispatches one batched matvec over cached (hit) rows to the GPU while the other CPU threads compute the miss rows.
- Admission: decode-only filling; adaptive admission threshold (`GGML_CUDA_MOE_CACHE_ADMIT_AFTER`); eviction on VRAM pressure; a persistent hot-set file.
- Routing skew: "top 10% of experts take ~80% of hits" on Qwen3.5-122B.
- Reported decode results:

  | Model and setup | Cache | Baseline | Change |
  |---|---|---|---|
  | GLM-5.1 754B IQ2_M, 4x3090, EPYC 7R13, DDR4-3200 | 17.49 | 13.96 | +25% |
  | Qwen3.5-397B Q3_K_XL | 30.25 | 28.18 | +7% |
  | DeepSeek-V4-Flash Q8, 2x3090 | 24.4 | 16.7 | +46% |
  | Laguna S 2.1 Q6_K, RTX 5090 + DDR5 | 27.6 | 18.2 | +52% |
  | DeepSeek V4 Q4_K_XL, RTX 5090 + DDR5 | 21.7 | 16.5 | +32% |
  | GTX 1080 Ti, 4 GB cache | 13.25 | 19.32 | −31% |

- Prefill cost: Laguna S prefill fell 980 → 840 t/s (−14%).
- Community summary: "Decode win is real and consistent; prefill cost is real and context-dependent".

**PR #26563 "Expert caching …, use -ehs N"** (miltos22; draft; opened before PR #26824 of Aug 10, 2026 — exact date not extracted). Source for all items: [llama.cpp PR #26563](https://github.com/ggml-org/llama.cpp/pull/26563)
- Design: a heat map with hysteresis and dwell; the top-S experts live in GPU slots; cold misses are computed on the CPU with no transfer; single-token decode only.
- Qwen3.6-35B-A3B: Q2_M 33 → 57 t/s (1.72x); Q5_K 17 → 36 t/s (2.07x).
- DeepSeek-V4-Flash (256 experts): at similar VRAM the heat cache lost to static pinning, "12.77 t/s vs 13.45 for a static 'pin 4 complete layers' baseline".
- Regression on an RTX 2000 laptop: 31 → 26 t/s.

**PR #26824** (miltos22, Aug 10, 2026, open). Source for all items: [llama.cpp PR #26824](https://github.com/ggml-org/llama.cpp/pull/26824)
- Design: "decay-tracked usage counters" per expert; hottest experts on the best device; mmap pinning; CPU↔device swaps with hash-verified handshakes; misses computed on the CPU via a fused op.
- Results on an RTX 3070 8 GB with 32 GB RAM:

  | Model | Baseline t/s | PR t/s | Speedup |
  |---|---|---|---|
  | Qwen3.6-35B IQ2_M | 38.1 | 59.7 | 1.57x |
  | Qwen3.6-35B Q4_K_M | 32.0 | 47.7 | 1.49x |
  | Qwen3.5-122B IQ2_M | 7.18 | 12.5 | 1.74x |
  | Gemma-4-26B | 19.9 | 43.6 | 2.19x |
  | Laguna-S IQ3_XXS | — | — | 1.02x |

**PR #27861 "GPU-resident LRU cache for host-offloaded MoE expert weights"** (csantiago78, Aug 28, 2026, draft). Source for all items: [llama.cpp PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861)
- Design: per-layer `[ne0, ne1, K+1]` cache tensors (slot K zero) with device and host id→slot maps. Hits run through a second GPU `mul_mat_id` chain over the cache tensors; misses use the CPU `mul_mat_id`; the outputs are summed ("exact by construction"). No custom CUDA kernels; decode-only (`n_tokens == 1`). Uploads are asynchronous and throttled by a worker thread and published at decode-boundary sync points.
- Author's result: 2x RTX 3090 on dual Xeon, Qwen3.8-Flash-Next UD-Q4_K_XL, 18.4 → 24.2 t/s (+31%) with 48 slots per layer (~4.1 GiB), 67–81% simulated LRU hit rate.
- Reviewer results:
  - R9700 with DDR5: +15% to +40%.
  - RTX PRO 4500, Qwen4_exp 180B: stock 16.83 vs tuned 41.31 t/s. The tuned run also changed `--fit-target`, so it is not equal-VRAM.
  - RX 7600 8 GB (Vulkan), Qwen3-30B-A3B Q4_K_M: 14.4 → 16.5 t/s (+14.6%).

**Issue #20757 "Two-tier GPU+RAM expert cache"** (e1n00r, Mar 19, 2026, closed). Source: [llama.cpp issue #20757](https://github.com/ggml-org/llama.cpp/issues/20757)
- Proposal: misses would be fetched CPU→GPU, not computed on the CPU.
- Claim: gpt-oss-120b on an RTX PRO 2000 8 GB reaches 12–14 tok/s steady state with a "~98–100%" hit rate, vs "0.5–1 tok/s" for baseline CPU offload.

**Related:** issue #25859 reports that offloaded-MoE prefill leaves the GPU idle waiting on serial expert H2D copies (single-GPU `-ncmoe`) — [llama.cpp issue #25859](https://github.com/ggml-org/llama.cpp/issues/25859)

### Inferences
- Because the scheduler synchronizes before a CPU split consumes GPU outputs, and the CPU backend computes synchronously, mainline batch-1 hybrid decode serializes the GPU attention/shared work and the CPU expert work in each layer. CUDA graphs cover only the GPU-split segments between CPU splits. This is inferred from the source code, not from a maintainer statement.
- The "dynamic GPU expert cache + CPU-executed misses" idea is being actively and independently re-implemented in the llama.cpp community (June–Aug 2026):
  - PR #26824 uses decay-tracked counters, which is conceptually the same family as "decayed-frequency admission".
  - Headline claims of 1.4–2.2x on skewed-routing models (Qwen3.x A3B, Gemma-4) overlap the 1.41–1.81x range of the work under review.
  - Differentiation therefore has to come from the hand-off mechanism, intra-layer concurrency, overheads, equal-VRAM methodology and robustness (flat-routing models, small GPUs, prefill cost), not from the caching concept.
- Of the community designs, only leloch's RFC clearly overlaps CPU miss compute with GPU hit compute within a layer, and it is CPU-initiated: a CPU thread dispatches the GPU matvec. None of the fetched designs describes a GPU-initiated, GPU-polled mailbox inside one CUDA graph.
- Most community speedups are not equal-VRAM. The cache typically uses VRAM beyond the `--n-cpu-moe` baseline, as with "zero-config" or tuned `--fit-target`. The one explicit equal-VRAM comparison (PR #26563 on DeepSeek-V4-Flash) shows a cache losing 5% to static layer pinning. An equal-GPU-memory methodology is a genuine strength if it is applied rigorously.
- Rough hardware context for the claimed gpt-oss-120b baseline of 40 tok/s at 25% of experts on the GPU (A10, ~150 GB/s DRAM): the best-documented consumer datapoint is 25–28 tok/s on an RTX 4070 with dual-channel DDR5-6000. The Hardware Corner DDR4 3090 figures of 1.4–1.8 t/s look anomalous. 64 GB of RAM for a ~60+ GB model suggests paging, which the article does not say. That source is low-confidence.

### Gaps
- Merge status after the fetch date: all of #26563, #26824 and #27861 were open or draft when read, and no core-maintainer (ggerganov, slaren, JohannesGaessler) position on merging an expert cache was found.
- The exact opening date of PR #26563 was not extracted.
- Whether PR #27861's GPU-hit chain and CPU-miss split actually overlap in time was not verified. Under the standard scheduler they are likely sequential (inference).
- Good-quality, dated community llama-bench numbers for gpt-oss-120b with `--n-cpu-moe` on RTX 3090/4090/5090 plus stated DDR4/DDR5 configurations were scarce. Only carteakey (4070) was well specified.
- The claim in issue #20757 of 98–100% hit rate for gpt-oss-120b with 8 GB VRAM, and its 0.5–1 tok/s baseline, look implausible or unrepresentative and were not independently verified.

## Other engines (Ollama, LM Studio, PowerInfer, MLX, TensorRT-LLM, llamafile, academic-but-open HybriMoE)

### Takeaway
None of the consumer engines checked ships dynamic GPU expert caching with CPU-executed misses. LM Studio exposes llama.cpp's static `--n-cpu-moe`, and Ollama has not merged expert-level CPU offload. The closest non-llama.cpp prior art for "dynamic cache + CPU misses + intra-layer scheduling" is academic: HybriMoE (DAC 2025), built on KTransformers.

### Cited Findings
- LM Studio 0.3.23 (Aug 12, 2025) added "Force Model Expert Weights onto CPU", which uses llama.cpp's `--n-cpu-moe` technology. That is static placement, and the release gives no numbers — [LM Studio 0.3.23 blog](https://lmstudio.ai/blog/lmstudio-v0.3.23)
- Ollama issue #11772 requests MoE CPU weight offload; it is "not implemented in Ollama proper" as of the latest comments. PR #12333 is unmerged, and users work around it by running llama.cpp directly — [Ollama issue #11772](https://github.com/ollama/ollama/issues/11772)
- HybriMoE (arXiv Apr 8, 2025; DAC 2025):
  - Built on KTransformers.
  - Mechanisms: "dynamic intra-layer scheduling strategy to balance workloads across CPU and GPU", "impact-driven inter-layer prefetching", and "score-based caching algorithm".
  - Results: average 1.33x prefill and 1.70x decode vs existing hybrid MoE systems.
  - Code: PKU-SEC-Lab/HybriMoE.
  - Source: [arXiv 2504.05897](https://arxiv.org/abs/2504.05897); [DAC 2025](https://dl.acm.org/doi/10.1109/DAC63849.2025.11133274)
- kt-kernel ships a `LLAMAFILE` CPU backend: GGUF-based, for any AVX2+ CPU. llamafile's CPU kernels are thus used as one KTransformers backend — [kt-kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)
- PowerInfer-2 (arXiv 2406.06282, June 2024) targets smartphone inference of models larger than DRAM, using neuron-cluster-level caching and flash I/O. A secondary summary reports Mixtral 8x7B at ~11 tok/s on a phone; this was not verified against the paper — [PowerInfer-2 arXiv](https://arxiv.org/html/2406.06282v1/); [secondary summary on X](https://x.com/rohanpaul_ai/status/1808627118538187127)

### Inferences
- Ollama and LM Studio inherit llama.cpp's static, host-sequenced hybrid path at best, so they are not competitive threats for the dynamic-cache/hand-off contribution.
- MLX runs on Apple unified memory, so the discrete-GPU/host-DRAM expert split this work addresses does not arise in the same form. This is inference; no source was fetched.

### Gaps
- TensorRT-LLM: no source was found or checked on CPU-executed MoE experts or host-memory expert offload for batch-1 decode.
- MLX: not researched beyond the unified-memory inference above.
- fastllm and other Chinese-ecosystem hybrid engines were not researched.
- PowerInfer (v1, desktop) was not checked for MoE expert caching.
- HybriMoE's exact hardware, models, absolute tok/s and CPU/GPU synchronization mechanism were not extracted; only the abstract was read.

## Feature matrix: (a) dynamic GPU expert caching with CPU-executed misses, (b) low-overhead / GPU-initiated CPU hand-off, (c) CPU–GPU concurrency within a layer

### Takeaway
(a) is already implemented in several unmerged llama.cpp community PRs (2026) and in academic HybriMoE. Shipping engines (KTransformers/SGLang, mainline llama.cpp, ik_llama.cpp, vLLM, LM Studio, Ollama) do not do per-token dynamic caching with CPU misses. (b) KTransformers already puts the CPU hand-off inside a single per-token CUDA graph, but via `cudaLaunchHostFunc` host callbacks rather than a GPU-written mailbox with a GPU spin-wait; no engine found uses the latter. (c) KTransformers/SGLang and leloch's llama.cpp RFC overlap CPU and GPU expert work within a layer; mainline llama.cpp does not.

### Cited Findings
Status as of Sept 2026:

**KTransformers (SOSP'25 paper)**
- (a) No. Static placement from offline profiling — [paper](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- (b) Yes, low-overhead: submit and sync run via `cudaLaunchHostFunc` inside one CUDA graph per token, giving up to 1.23x. It is host-callback based, not GPU-polled — [paper](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf); [cpuinfer.h](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/cpu_backend/cpuinfer.h)
- (c) Yes. Shared experts run on the GPU while routed experts run on the CPU, and Expert Deferral adds (lossy) cross-layer overlap — [paper](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)

**SGLang + kt-kernel**
- (a) Partial. Static strategies plus `--kt-enable-dynamic-expert-update`, which redistributes GPU experts only during layerwise prefill ≥ threshold — [kt-kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)
- (b) Same `cudaLaunchHostFunc` submit/sync as the paper — [cpuinfer.h](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/cpu_backend/cpuinfer.h)
- (c) Yes. Submit CPU work, run GPU experts "in parallel", then sync and add — [kt_ep_wrapper.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/kt_ep_wrapper.py)

**Mainline llama.cpp (master)**
- (a) No. Static `-ot`/`--n-cpu-moe`; used-expert copies to the GPU happen only for large-batch prompt processing — [ggml-backend.cpp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend.cpp)
- (b) No. Host-sequenced splits with synchronize/event waits; the `cudaLaunchHostFunc` wait path is `#if 0` — [ggml-cuda.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu)
- (c) No, inferred from the in-order split execution in the same scheduler code.

**llama.cpp community PRs (unmerged)**
- (a) Yes: RFC #24528, PR #26563, PR #26824 and PR #27861 — [#24528](https://github.com/ggml-org/llama.cpp/discussions/24528); [#26563](https://github.com/ggml-org/llama.cpp/pull/26563); [#26824](https://github.com/ggml-org/llama.cpp/pull/26824); [#27861](https://github.com/ggml-org/llama.cpp/pull/27861)
- (b) No GPU-initiated mailbox was found. #24528 is CPU-initiated (CPU thread 0 dispatches the GPU hit matvec), and #27861 uses the standard graph/scheduler.
- (c) #24528: yes, hits on the GPU concurrently with misses on the CPU. #27861: unclear.

**ik_llama.cpp**
- (a), (b) and (c) were not found in the README feature list. PR #698 is prompt-processing-only activated-expert offload — [README](https://github.com/ikawrakow/ik_llama.cpp/blob/main/README.md); [PR #698](https://github.com/ikawrakow/ik_llama.cpp/pull/698)

**vLLM**
- (a) Only as the RFC #38256 cache, and its misses are PCIe-fetched to the GPU rather than computed on the CPU. CPU-computed experts are draft and out-of-tree (#56118) — [RFC #38256](https://github.com/vllm-project/vllm/issues/38256); [PR #56118](https://github.com/vllm-project/vllm/pull/56118)
- (b), (c) No evidence.

**LM Studio / Ollama**
- (a) No: LM Studio uses static llama.cpp `--n-cpu-moe`, and Ollama lacks even that — [LM Studio](https://lmstudio.ai/blog/lmstudio-v0.3.23); [Ollama #11772](https://github.com/ollama/ollama/issues/11772)

**HybriMoE (academic, DAC'25)**
- (a) Yes (score-based caching).
- (c) Yes (dynamic intra-layer CPU/GPU scheduling).
- Source: [arXiv 2504.05897](https://arxiv.org/abs/2504.05897)

### Inferences
- Novelty assessment for the work under review:
  - (a) Dynamic per-layer GPU expert cache with CPU-executed misses and frequency/decay admission is not novel as a concept. It has 2025 academic prior art (HybriMoE) and at least four 2026 llama.cpp community implementations, one with decay-tracked counters.
  - (b) The specific mechanism is a GPU kernel writing layer input to pinned host memory, spinning CPU helper threads, and a GPU spin-wait merge kernel, all inside one CUDA graph. No engine was found that does this. However, "the entire hybrid decode step in one CUDA graph" is KTransformers' published contribution (via `cudaLaunchHostFunc`), and its paper intro even mentions "CUDA-based spinning". Novelty is therefore limited to the device-initiated, polling-based hand-off and whatever latency advantage it has over host callbacks. That advantage would need to be measured directly, as per-layer hand-off latency and end-to-end tok/s vs a `cudaLaunchHostFunc` variant.
  - (c) Intra-layer CPU||GPU concurrency is present in KTransformers/SGLang and in leloch's llama.cpp RFC.
- Competitiveness:
  - The reported 1.41–1.81x over `--n-cpu-moe` at equal GPU memory is in the same band as the community PRs, which are mostly not equal-memory: +7% to 2.2x, with some regressions. It is also in the band of KTransformers' 1.25–1.93x over an older, patched llama.cpp without deferral.
  - The strongest missing comparison is a same-hardware, equal-VRAM head-to-head against SGLang+kt-kernel. That run would use AVX-512 on the Xeon 8358, `--kt-num-gpu-experts` matched to the 25% budget, deferral 0 for an exact comparison, and `uniform` or `frequency` placement.
  - A second missing comparison is against the best llama.cpp community cache PR (#26824 or #27861) at equal VRAM.

### Gaps
- No public head-to-head was found between KTransformers/SGLang and llama.cpp `--n-cpu-moe` on current (2026) versions for gpt-oss or Qwen3-30B-A3B at batch-1.
- There are no published measurements of `cudaLaunchHostFunc` callback latency inside CUDA graphs, per MoE layer, in KTransformers.
- Whether any closed or commercial engine (for example TensorRT-LLM, or vendor forks) implements GPU-initiated CPU expert hand-off could not be determined.
