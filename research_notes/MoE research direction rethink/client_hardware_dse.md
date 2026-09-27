# Client-hardware DSE for MoE batch-1 decode (A) and the batch-1 GPU efficiency gap / CPU+GPU bandwidth aggregation (B): prior art and open gaps, as of 27 Sept 2026

Method note: sources are arXiv abstract/HTML pages, conference PDFs, vendor/benchmark blogs and GitHub pages, fetched 27 Sept 2026. Many summaries came through a small-model page summariser, so treat quoted numbers as "as summarised". Where only a title was seen in search results (not fetched), the entry says "title only". Several of the closest papers are 1–3 weeks old (arXiv 2609.xxxxx), which matters for novelty claims. Venue-deadline and SPCL facts are reused from the sibling note `research_notes/MoE hybrid decode novelty check/venue_bar.md` and cite the same primary URLs.

## (A) Published characterizations and hardware DSE of LLM/MoE inference on client/edge hardware: is MoE-specific DSE with real routing traces across memory hierarchies already done?

### Takeaway
Nobody has published a systematic, validated, MoE-specific DSE across *client* machine classes: dGPU + dual-channel DDR5, unified-memory APUs such as Strix Halo and DGX Spark, Apple, CXL and PCIe generations, all driven by real routing traces. The space next to it is filling up fast, though. In Aug–Sep 2026 alone three arXiv papers each covered part of it. One is a single-machine three-tier RTX 3070 study with Qwen3-30B-A3B traces and a bytes-per-token model (Aug 2026). One is a Huawei DSE of memory provisioning for a "trillion-parameter MoE in a box", driven by expert-occupancy traces (14 Sep 2026). One is an Apple-Silicon SSD-streaming MoE system (Sep 2026). The analytical DSE tools (GenZ, LLMCompass, LLM-Viewer) target datacenter platforms and do not model offload hierarchies with routing locality. The honest verdict is "adjacent and getting crowded; the client cross-platform framing is still open".

### Cited Findings

**Closest MoE + memory-tier + routing-trace work (2026)**
- **"Cacheable by Design? Training MoE Routers for Locality Against the Edge Memory-Bandwidth Wall: A Pre-Registered Negative Result, with a Systems Measurement Study"**, Shriniwas Ramesh Suram (University of the Cumberlands), arXiv Aug 2026.
  - Platform and model: one consumer box (RTX 3070 8 GB, i9-12900, 32 GB DDR4, PCIe 3.0 DRAM-less NVMe) running Qwen3-235B-A22B Q4_K_M (134 GB).
  - Tiers characterised: VRAM ~448 GB/s, DDR4 ~50 GB/s, NVMe ~2.4 GB/s.
  - Model: a first-principles "tok/s = bandwidth / bytes-per-token" model predicts cold-disk performance "within 6%".
  - Routing traces: its tool "llama-moe-trace" captured 32,000 tokens across 4 domains on Qwen3-30B-A3B. At a 13.4% fast-memory budget the hit rates are LRU 65.9%, static pinning 59.2% and Belady 79.1%. 52.5% of experts serve 95% of traffic.
  - Projection: its "Path-Mapped Serving" projects 2–7× gains, and it finds that 10 tok/s "provably requires ~95% cache hit rate".
  - Source: [arXiv 2608.18261](https://arxiv.org/html/2608.18261v1)
- **"Trillion-Parameter MoE in a Box: Decoupling Memory Provisioning with High-Bandwidth Flash"**, Xia et al. (Huawei), arXiv 2609.15636, 14 Sep 2026.
  - Framing: an explicit hardware DSE for a "compact, low-concurrency" appliance serving 1–8 requests. It uses an analytical/simulation methodology built on LLMServingSim 2.0, with expert-occupancy traces from DSV4-Flash (SPEED-Bench) and agentic multi-turn traces.
  - Configurations swept:
    - DRAM "state tier": HBM3e, SOCAMM2 and LPCAMM2
    - number of HBF packages: 1–12
    - host-link quanta: 96 GB/s each
    - near-HBF compute on or off
  - Findings: two "largely orthogonal" knees. The state-tier knee sits at a DRAM bandwidth-to-capacity ratio of "1.4–4.0 s⁻¹, roughly 8–24× below HBM3e's 33.3 s⁻¹". The HBF link knee sits at 384 GB/s per package.
  - This is the closest methodological analogue to direction (A), but it targets trillion-parameter appliances with HBF, not consumer PCs.
  - Source: [arXiv 2609.15636v1](https://arxiv.org/html/2609.15636v1)
- **"The Other Half of the Memory Wall: Serving 35B MoEs from SSD with Trained Routing Prediction" (Edge0)**, Lin et al. (AutoArk), arXiv 2609.18063, Sep 2026.
  - Platforms and model: Mac mini M4 Pro 24 GB and MacBook M2 16 GB, running Qwen3.6-35B-A3B from mmapped int4 experts.
  - Results: 20.4 tok/s at 2.9 GiB peak memory. A fitted per-load cost of "1.17 ms + 1.33 ms × (cold fraction)". "Adjacent tokens agree on only about a quarter of a layer's expert set."
  - Nature: a system paper, not a DSE.
  - Source: [arXiv 2609.18063](https://arxiv.org/html/2609.18063)
- **Other 2025–26 edge-MoE titles (not fetched):**
  - "APEX: Adaptive Expert Prefetching for Memory-Efficient Edge MoE Inference" — [arXiv 2608.11688 (Pith)](https://pith.science/paper/2608.11688)
  - "Enabling MoE on the Edge via Importance-Driven Expert Scheduling" — [arXiv 2508.18983](https://arxiv.org/pdf/2508.18983)
  - "Accelerating Edge Inference for Distributed MoE Models with Latency-Optimized Expert Placement" — [arXiv 2508.12851](https://arxiv.org/abs/2508.12851)
  - "Beyond Independent Optimization: Compression, MoE Routing, and Quantization Interactions in Multimodal Edge Intelligence" — [arXiv 2607.20981](https://arxiv.org/html/2607.20981v1)
- **"Expert Streaming" / FSE-DP**: "a parallelization paradigm specifically architected for low-batch MoE inference on multi-chiplet accelerators", reporting 1.22–2.00× over baselines and up to 78.8% on-chip memory saved. It is an accelerator-architecture paper; authors and venue were not retrieved. — [arXiv 2603.27624](https://arxiv.org/html/2603.27624)

**Analytical models / DSE tools (all datacenter-oriented)**
- **GenZ.** arXiv 2406.01698, v1 3 Jun 2024 and v3 15 May 2025, retitled "Demystifying AI Platform Design for Distributed Inference of Next-Generation LLM models" (Bambhaniya, …, Tushar Krishna).
  - It is an analytical tool linking model architectures (Dense, GQA, MoE, Mamba) to platform compute, memory capacity/bandwidth and network.
  - The fetched abstract shows no offload/memory-hierarchy DSE and no edge focus.
  - It is an arXiv preprint with no venue found.
  - Source: [arXiv 2406.01698](https://arxiv.org/abs/2406.01698); [PyPI genz-llm](https://pypi.org/project/genz-llm)
- **LLMCompass.** "LLMCompass: Enabling Efficient Hardware Design for Large Language Model Inference", ISCA 2024 (Princeton). A hardware evaluation framework for accelerator design; not client or MoE-offload specific. Only the title and venue were seen. — [ISCA'24 PDF](https://parallel.princeton.edu/papers/isca24_llmcompass.pdf)
- **LLM-Viewer.** From "LLM Inference Unveiled: Survey and Roofline Model Insights" (arXiv 2402.16363), a tool to "analyze computation, storage, transmission, and hardware roofline model". — [arXiv 2402.16363](https://arxiv.org/abs/2402.16363); [GitHub LLM-Viewer](https://github.com/hahnyuan/LLM-Viewer)
- **MoE-Lightning.** "High-Throughput MoE Inference on Memory-constrained GPUs", ASPLOS 2025 (arXiv Nov 2024).
  - It uses a hierarchical-roofline performance model to choose a CPU-GPU-I/O pipelining policy for *high-throughput batched* inference, not batch-1 or hardware DSE.
  - The sibling note found that MoE-Lightning's HRM and "Pipelined Sharding" use models to choose policies, not to bound performance.
  - Source: [arXiv 2411.11217](https://arxiv.org/abs/2411.11217); [ACM DL](https://dl.acm.org/doi/10.1145/3669940.3707267); [sibling note venue_bar.md]

**CXL for LLM inference (datacenter, not MoE batch-1)**
- **TRACE** ("Unlocking Effective CXL Bandwidth via Lossless Compression and Precision Scaling", Xie et al.).
  - Dates: arXiv 3 Sep 2025, revised 30 Jan 2026.
  - Scope: a datacenter, long-context setting where "KV spills to CXL", tested at 128k tokens on GPT-OSS-120B.
  - Results: lossless BF16 weight footprint −25.2% and KV −46.9%.
  - No MoE focus.
  - Source: [arXiv 2509.03377](https://arxiv.org/abs/2509.03377)
- **LIA** ("A Single-GPU LLM Inference Acceleration with Cooperative AMX-Enabled CPU-GPU Computation and CXL Offloading"), ISCA 2025. It pairs Intel AMX CPU compute with a GPU and uses CXL for offloading. Partitioning and speedup details were not retrieved. — [ACM DL](https://dl.acm.org/doi/10.1145/3695053.3731092); [GitHub](https://github.com/hyungyokim/LIA_AMXGPU)
- **Other CXL titles (not fetched):**
  - "CXLRAMSim v1.0: System-Level Exploration of CXL Memory Expander Cards" — [awesomepapers 2603.29483](https://awesomepapers.io/systems-efficiency/papers/2603.29483)
  - "Analysis and Optimized CXL-Attached Memory Allocation for Long-Context LLM Fine-Tuning" — [arXiv 2507.03305](https://arxiv.org/html/2507.03305v2)
  - IPDPS'25 "Performance Characterization of CXL Memory and Its Use Cases" — [PDF](http://pasalabs.org/papers/2025/IPDPS25_CXL.pdf)

**Client/edge platform characterizations and measured datapoints (usable for calibration)**
- **DGX Spark (LMSYS review, 13 Oct 2025).**
  - Memory: 273 GB/s LPDDR5X unified memory.
  - Ollama batch-1 results: gpt-oss-20b 49.7 tok/s, Llama 3.1 8B (SGLang FP8) 20.5 tok/s, Llama 3.1 70B FP8 2.7 tok/s.
  - An RTX Pro 6000 Blackwell was "roughly 4× faster" on gpt-oss-20b. Memory bandwidth was called "the key bottleneck".
  - Source: [LMSYS](https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/)
- **llama.cpp DGX Spark discussion #16578** (14 Oct 2025, updates to 5 Feb 2026).
  - gpt-oss-20b: Spark 60 t/s (tg32) vs M4 Max 117.83 t/s (tg128).
  - gpt-oss-120b: Spark 35 t/s vs Ryzen AI Max+ 395 47.49 t/s (tg32).
  - Qwen3-Coder-30B-A3B Q8_0: Spark 44.26 t/s.
  - The summary also quotes "Spark (86 t/s)" for generation, which conflicts with the 60 t/s figure. Treat the Spark gpt-oss-20b number as uncertain in the 60–86 t/s range.
  - Source: [llama.cpp discussion #16578](https://github.com/ggml-org/llama.cpp/discussions/16578)
- **Strix Halo (community guide, 2026).**
  - Memory: Radeon 8060S at 256 GB/s theoretical.
  - Decode on llama.cpp Vulkan/RADV:
    - gpt-oss-120b MXFP4: 55.57 t/s (build b9049, 7 May 2026 campaign)
    - Qwen3.6-35B-A3B: 32.23 t/s with a filled 128K KV cache
    - Nemotron 3 Super 120B-A12B: 18.93 t/s (Jun 2026)
  - The guide computes no bandwidth fractions.
  - Source: [strix-halo-guide](https://github.com/hogeheer499-commits/strix-halo-guide)
- **Apple Silicon studies (titles only):**
  - "Production-Grade Local LLM Inference on Apple Silicon: A Comparative Study of MLX, MLC-LLM, Ollama, llama.cpp, and PyTorch MPS" — [arXiv 2511.05502](https://arxiv.org/abs/2511.05502)
  - "Native LLM and MLLM Inference at Scale on Apple Silicon" — [arXiv 2601.19139](https://arxiv.org/html/2601.19139v1)
  - "BaseRT: Best-in-Class LLM Inference on Apple Silicon via Native Metal" — [arXiv 2607.00501](https://arxiv.org/html/2607.00501v1)
- **Mobile/AI-PC characterizations (titles only):**
  - "LLM Inference at the Edge: Mobile, NPU, and GPU Performance Efficiency Trade-offs Under Sustained Load" — [arXiv 2603.23640](https://arxiv.org/abs/2603.23640)
  - "Energy-Efficient On-Device RAG on a Mobile NPU … Snapdragon X Elite" — [arXiv 2606.11257](https://arxiv.org/html/2606.11257v1)
- **CPU-GPU coupling characterization** ("Characterizing and Optimizing LLM Inference Workloads on CPU-GPU Coupled Architectures", CMU/Samsung, arXiv 2504.11750).
  - Platforms: GH200 against PCIe AMD+A100 and Intel+H100.
  - Launch overhead: 2771.6 ns on GH200 vs 2260.5 / 2374.6 ns.
  - CPU-boundness: GH200 stays CPU-bound "up to 4x larger batch sizes".
  - No decode bandwidth fraction and no analytical model.
  - Source: [arXiv 2504.11750](https://arxiv.org/html/2504.11750v1)
- ISPASS/IISWC program scan: the ISPASS 2025 accepted list and the ISPASS 2026 program exist but were not scanned. — [ISPASS 2025 accepted](https://ispass.org/ispass2025/accepted-papers.php); [ISPASS 2026 program](https://ispass.org/ispass2026/program.php)

### Inferences
- **What is actually open.**
  - (i) A cross-client-class DSE that puts dGPU+DDR5, APU-unified (Strix Halo, Spark, Apple), CXL expanders and PCIe gen 4/5/6 on one validated model.
  - (ii) Real routing locality (exact traces from OLMoE, Qwen3-30B-A3B, gpt-oss-20b) setting the effective bytes/token per tier. This is the "hit-rate knee" that Cacheable-by-Design measured for one box only.
  - (iii) Answers in "speed-of-light fraction" form.
  - The Huawei HBF paper's "bandwidth-to-capacity ratio knee" is the idea to borrow, or at least cite and contrast.
- **Crowdedness, bluntly.** The MoE-on-edge *systems* space is very crowded (see the sibling novelty-check notes: ~35 systems). The *DSE* niche is less crowded but was touched three times in six weeks. A reviewer will call pure roofline sweeps "obvious" unless the DSE gives non-obvious, decision-relevant answers. Candidate answers the researcher's model could test:
  - whether PCIe gen 5/6 matters at batch 1 once hot experts are resident;
  - whether DRAM channels matter more than VRAM size;
  - at what VRAM:DRAM ratio a given routing locality makes dGPU+DDR5 beat a 256–273 GB/s APU;
  - whether CXL helps only capacity, not batch-1 speed.
- **A rough calibration check on the community numbers.** gpt-oss-120b on Strix Halo at ~55 t/s implies roughly 60–70% of 256 GB/s, assuming ~3 GB of active bytes per token. That figure is my estimate and not sourced. The researcher's 52-measurement set probably already covers most of these points.
- **Feasibility.**
  - By 27 Oct 2026, a model-driven DSE preprint is feasible: sweeps are cheap, the traces exist, and published datapoints provide validation.
  - What is *not* feasible without consumer hardware is first-party validation on Strix Halo, Spark or Apple. The paper would rest on 52 published measurements plus cloud A10/L4/L40S runs.
  - Cloud VMs have server CPUs with many DRAM channels, which are unrepresentative of desktop DDR5. That weakens the validation of any dGPU+DDR5 conclusion.
  - By Dec 2026, adding 1–2 owned or borrowed consumer machines would substantially strengthen an ISPASS submission.

### Gaps
- Vidur (MLSys 2024) and any newer "LLM inference simulator for edge" tools were not retrieved, so it is unconfirmed whether any of them models client offload hierarchies.
- ISPASS 2025/2026, IISWC 2025, HPCA 2026, ISCA 2026 and MICRO 2025 programs were not scanned paper by paper. A client-hardware LLM characterization at one of these venues may exist and was missed.
- The full text of the HBF paper was not read for its baseline hardware or validation. It is unknown whether its model is validated against real hardware.
- No paper was found that characterises Qualcomm/Intel AI-PC (NPU+iGPU) MoE decode with routing traces. This is absence of evidence from 2–3 searches, not proof.
- No published work on HBM-on-client proposals or PCIe gen 6 client MoE decode was found; not searched in depth.

## (B) The batch-1 GPU efficiency gap and heterogeneous bandwidth aggregation: is "the CPU helps even when the model fits in VRAM" published?

### Takeaway
The ~50% batch-1 bandwidth gap is well known and actively attacked:
- the Hazy "No Bubbles" megakernel (May 2025: vLLM/SGLang "at most 50%" vs 78% on H100)
- Mirage Persistent Kernel (Dec 2025, up to 1.7×)
- Ada-MK (May 2026)
- a 44-cell cross-GPU study (May 2026: H100 ~27%, L4 ~81%)
- a llama.cpp ROCm issue (Sep 2026: 65.9% single card)

"Aggregate a second memory's bandwidth during decode" is also published, but only in two settings that differ from the researcher's:
- HeteroInfer (SOSP'25): mobile unified SoC, GPU+NPU concurrently, 43.3→59.5 GB/s.
- BOOST (arXiv 11 Sep 2026, Georgia Tech + NVIDIA Research): GH200, where the GPU reads host LPDDR directly over NVLink-C2C. Only +4.3% TPOT at low latency.

FusionML (Jul 2026) reports that CPU+GPU co-execution gives *no* decode benefit on Apple unified memory. I found nothing published on the exact configuration: a discrete PCIe GPU with the CPU computing a slice from its own DRAM, when the model fits in VRAM, at batch 1, for MoE, with gains explained by the GPU's own <100% efficiency. The claim is narrow and contingent on the GPU gap persisting.

### Cited Findings

**Batch-1 efficiency gap: measurements and fixes**
- **Hazy Research, "Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B"** (27 May 2025).
  - Speed: on H100, under 1 ms per forward pass, "almost 2.5x faster than vLLM and over 1.5x faster than SGLang". On B200, under 680 µs.
  - Bandwidth: the megakernel uses "78% of available GPU bandwidth" on H100, while vLLM/SGLang "are only able to use at most 50%".
  - Causes named: kernel-boundary serialization; launch cost of ~2.1 µs, or ~1.3 µs with CUDA graphs; and weight/activation load stalls at kernel starts.
  - Source: [Hazy Research blog](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- **Mirage Persistent Kernel (MPK).** Cheng, Zhang, Zhou et al. (CMU, Tsinghua, NVIDIA, …), arXiv 2512.22219, 22 Dec 2025. It compiles LLMs into a single mega-kernel with "1.0–1.7× speedups" vs SGLang/vLLM with CUDA graphs on A100, H100 and B200. No bandwidth fraction appeared in the fetched text. — [arXiv 2512.22219](https://arxiv.org/html/2512.22219v1); [GitHub](https://github.com/mirage-project/mirage)
- **Ada-MK** ("Adaptive MegaKernel Optimization via Automated DAG-based Search for LLM Inference", arXiv 2605.11581, ~May 2026; title only). It shows megakernel work continuing in 2026. — [arXiv 2605.11581](https://arxiv.org/html/2605.11581)
- **"Memory-Bound but Not Bandwidth-Limited: The Physical AI Inference Gap in Batch-1 LLM Decode"** (Josef Chen, KAIKAKU, arXiv 2605.30571, May 2026).
  - Scope: 44 cells covering H100, A100, L40S and L4 with Qwen2.5-7B, Mistral-7B and Llama-3.1-8B in bf16.
  - Bandwidth: the achieved fraction of peak falls as peak rises, from ~81% on L4 to ~27% on H100.
  - CUDA graphs: 1.259× on H100 (95% CI [1.253, 1.267]) but only 1.028× on L4. The paper attributes the H100 gap mainly to per-kernel launch overhead.
  - Caveat: single-author company preprint; the engine is not clearly identified in the summary.
  - Source: [arXiv 2605.30571](https://arxiv.org/html/2605.30571v1)
- **llama.cpp issue #28863** (13 Sep 2026). On a 7900 XTX (960 GB/s), a Qwen 27B Q4_K_XL batch-1 decode reaches 36.05 t/s, which is "65.9% of DRAM bandwidth (632.5 GB/s)". A two-card tensor split falls to 43.0% per card, a 1.31× total speedup. No maintainer response yet. — [llama.cpp #28863](https://github.com/ggml-org/llama.cpp/issues/28863)
- **"A Systematic Characterization of LLM Inference on GPUs"** (arXiv 2512.01644). It reports decode as memory-dependency-stall dominated (roofline arithmetic intensity ≈1–10) but gives no batch-1 bandwidth fraction. — [arXiv 2512.01644](https://arxiv.org/html/2512.01644v1)

**Concurrent use of a second memory's bandwidth for decode (closest prior art)**
- **HeteroInfer**, "Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference" (Le Chen, …, Haibo Chen; SJTU IPADS/Tsinghua/SenseTime), SOSP '25, 13–16 Oct 2025.
  - Single processor: on Snapdragon 8 Gen 3, "each individual processor (CPU, GPU, or NPU) achieves only 40–45 GB/s" of a 68 GB/s peak.
  - Concurrent: running GPU and NPU together raises decode bandwidth "from 43.3 GB/s (only GPU) to 59.5 GB/s".
  - Partitioning: weight-centric row partitioning of weight tensors between GPU and NPU during decode.
  - This is the unified-memory version of "one engine can't saturate memory; add a second engine".
  - Source: [arXiv 2501.14794](https://arxiv.org/html/2501.14794v2)
- **BOOST**, "Concurrent Access to Host Memory and HBM to Accelerate LLM Inference" (Saxena, Ju, Taneja, Tsai, Jaleel, Kozyrakis, Qureshi; Georgia Tech + NVIDIA Research), arXiv 2609.13592, 11 Sep 2026.
  - Idea: proportional *concurrent demand loads* from HBM and host memory, with no migration.
  - Platform: GH200 over NVLink-C2C (HBM ~3.33 TB/s, host ~350 GB/s, α≈10%), plus H100 emulation.
  - Placement and data: modulo page placement for weights and CTA-level tier specialization. Models are Llama-3.3-70B FP8 and Qwen3-Next-80B MoE.
  - Results:
    - Low-latency TPOT +4.3% vs HBM-only, at batch 10 and 88 requests. Prefetching gives −6% there.
    - Throughput +31%, "host capacity contributes ~27%, concurrent access adds 4%".
    - Microbenchmark 7.4% vs 10.4% ideal.
  - Here the GPU does all the compute; the CPU does not compute.
  - Source: [arXiv 2609.13592](https://arxiv.org/html/2609.13592)
- **FusionML**, "Prefill, Not Decode: Mechanism and Boundaries of CPU+GPU Co-Execution on Unified-Memory Apple Silicon" (Om Mohite, arXiv 2607.22785, 24 Jul 2026).
  - Prefill: CPU+GPU co-execution on M1–M4 Pro gives 1.15–1.38× block-level and 1.18–1.25× TTFT on Qwen2.5-7B.
  - Decode: "no benefit" because decode is "bandwidth-bound because unified-memory bandwidth is shared".
  - Source: [Pith summary of arXiv 2607.22785](https://pith.science/paper/2607.22785)
- **HeteGen** (Zhao et al., NUS/Yang You), MLSys 2024.
  - Mechanism: partitions linear layers between CPU and GPU, runs them concurrently and asynchronously, and targets models that *exceed* GPU memory.
  - Evaluated on an **NVIDIA A10 24 GB** + Xeon over PCIe (~25–30 GB/s), with up to "317%" gains on OPT-30B.
  - Not the model-fits case.
  - Source: [MLSys'24 PDF](https://proceedings.mlsys.org/paper_files/paper/2024/file/5431dca75a8d2abc1fb51e89e8324f10-Paper-Conference.pdf)
- **ATSInfer**, "Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference on Consumer Devices" (Nanjing University, arXiv 2607.10183).
  - Mechanism: fine-grained CPU/GPU tensor scheduling that dynamically promotes CPU-resident tensors "even when the model fits in VRAM".
  - Claims: it does **not** claim the CPU adds bandwidth; it treats CPU execution as a latency cost.
  - Setup: RTX 3060 laptop and RTX 4090 desktop, with MoE models including gpt-oss-20b/120b, Qwen3-Next-80B-A3B and Qwen3.5-122B-A10B.
  - Results: decode up to 3.29× / 3.12× vs llama.cpp.
  - Source: [arXiv 2607.10183](https://arxiv.org/html/2607.10183)
- **Multipath Memory Access (MMA)** (arXiv 2512.16056v2, 13 May 2026). It relays host→GPU transfers through peer GPUs over NVLink on 8×H20, reaching 245 GB/s vs 53 GB/s native. It concerns loading and KV fetch; the CPU does not compute decode. — [arXiv 2512.16056](https://arxiv.org/html/2512.16056v2)
- **Other CPU-GPU heterogeneous decode work** (titles/venues only). All target GPU-memory-constrained or throughput settings, or speculative decoding, rather than the model-fits batch-1 case:
  - APEX ("Asynchronous Parallel CPU-GPU Execution for Online LLM Inference on Constrained GPUs") — [arXiv 2506.03296](https://arxiv.org/abs/2506.03296)
  - FastDecode (heterogeneous high-throughput serving) — [arXiv 2403.11421](https://arxiv.org/pdf/2403.11421)
  - TwinPilots (SYSTOR 2024) — [ACM DL](https://dl.acm.org/doi/10.1145/3688351.3689164)
  - Dovetail (CPU/GPU heterogeneous speculative decoding, EMNLP 2025) — [ACL Anthology](https://aclanthology.org/2025.emnlp-main.879.pdf)
  - Ghidorah (edge speculative decoding with hetero-core parallelism) — [Pith 2505.23219](https://pith.science/paper/2505.23219)
- **Community conventional wisdom** is that spilling layers to the CPU is a cliff: "15× slower the moment layers spill to CPU". This reflects llama.cpp's *layer-serial* split, not an intra-layer concurrent split. — [InventiveHQ "VRAM cliff"](https://inventivehq.com/blog/vram-offload-cliff-gpu-layers-benchmark)
- **KTransformers (SOSP'25)**: CPU-GPU hybrid MoE with CPU experts inside a CUDA graph through `cudaLaunchHostFunc`, for models that do not fit (sibling notes). — [sibling note industry_engines.md; SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)

### Inferences
- **Novelty boundary.** "Use two memories' bandwidth concurrently during decode" is published:
  - unified mobile SoC: HeteroInfer (SOSP'25)
  - coherent superchip: BOOST (Sep 2026, strong group)
  - negative result on Apple unified memory: FusionML
  - The discrete-PCIe-GPU variant is distinct. There the CPU *computes* on its own DRAM because PCIe (~25–32 GB/s) is too slow for BOOST-style direct GPU reads, and the model fits.
- **The central, testable twist.** On the A10, the GPU runs at η_g≈0.5, so effective GPU bandwidth is only ~260 GB/s. The CPU's DRAM share is then effectively ~2× larger than nominal, and CPU help is worthwhile. BOOST's low-latency gain was only 4.3% because α≈10% and the GPU was efficient. This is one mechanism and one equation, and it makes a crisp, falsifiable prediction.
- **The biggest reviewer objection.** "Fix the GPU kernel instead." My arithmetic on the researcher's own numbers, not a sourced result: if a megakernel lifted gpt-oss-20b on the A10 from η_g≈0.5 to ~0.78 (the Hazy H100 figure), GPU-only would be ~139×0.78/0.5 ≈ 217 tok/s. That exceeds the 180 tok/s concurrent bound computed at η_g≈0.5.
  - The paper must present the bound as a function of η_g and of the CPU/GPU bandwidth ratio. It must show that CPU aggregation still adds (1+α_eff) on top of an improved GPU kernel, or else concede it is a "until kernels improve" result.
  - This is exactly the kind of bound-parameterised framing SPCL values, but it also exposes the contingency.
- **The platform question.** A cloud A10 VM has a server CPU. Consumer desktops with dual-channel DDR5 have much lower CPU bandwidth relative to high-end dGPUs. Bandwidth specs were not sourced here, so this is qualitative. The gain could be smaller on an RTX 4090 desktop and larger on a mid-range card.
  - Without consumer hardware, the "Accessible AI" claim rests on the model. Renting cloud L4 instances (300 GB/s, ~81% η per 2605.30571) would test the case where the GPU is already efficient, and the prediction there is a small gain.
- **Crowdedness, bluntly.** The GPU-efficiency-gap half is crowded: megakernels come from well-resourced groups (Stanford, CMU/NVIDIA), and a solo researcher will not beat them on kernels. The aggregation half is thin but just got a heavyweight entrant (BOOST). The PCIe-dGPU+CPU-compute+MoE variant looks open as of 27 Sep 2026 but could be scooped quickly.
- **Feasibility.**
  - By 27 Oct 2026: yes for an arXiv preprint. The mechanism, the A10 results and the bound exist. Adding 2–3 cloud GPU types (L4, L40S, A10) and 3 MoE models is ~1–2 weeks of rented compute.
  - By Dec 2026 (ISPASS): add an η_g sweep (CUDA graphs on/off, and a megakernel-style or better-kernel baseline if possible) and at least one consumer-class machine.

### Gaps
- No full-text check of whether HeteroInfer, BOOST or ATSInfer evaluated a PCIe discrete GPU + CPU compute split at batch 1 when the model fits. The summaries say no, but related-work sections were not exhaustively read.
- Batch-1 bandwidth fractions for TensorRT-LLM and for vLLM on consumer GPUs (RTX 4090/5090) were not found from a primary source.
- MPK's achieved bandwidth fraction at batch 1 was not in the fetched text.
- LIA's (ISCA'25) exact CPU/GPU partition, whether it covers the model-fits case, and its hardware were not retrieved.
- Whether llama.cpp maintainers or ik_llama.cpp have discussed a "tensor-parallel CPU+GPU split when the model fits" option was not checked beyond one search.

## Open gaps, feasibility by 27 Oct and Dec 2026, venue, and fit with SPCL taste (both directions)

### Takeaway
- **(B)** is the sharper, more novel and more feasible paper with the researcher's existing assets. Frame it as a bound-parameterised study, "when does adding CPU DRAM bandwidth beat an imperfect GPU", with BOOST, HeteroInfer and FusionML as the three contrasting prior points. It fits ISPASS 2027 (~mid-Dec 2026, inferred) or EuroMLSys 2027, and an arXiv preprint by 27 Oct is realistic.
- **(A)** is a better long-term "Accessible AI" thesis agenda and matches SPCL's performance-modelling taste. As a stand-alone paper by Oct/Dec 2026 it risks "roofline sweep" criticism and lacks first-party consumer-hardware validation. It is best folded in as the "design implications" section of (B), or pitched as the PhD research proposal.
- No top-venue *acceptance* is possible before 27 Oct 2026.

### Cited Findings
- **Deadlines:**
  - MLSys 2027: paper deadline 30 Oct 2026 (three days after the ETH deadline), 10 pages plus appendix, double-blind, arXiv allowed. — [MLSys 2027 CFP](https://mlsys.org/Conferences/2027/CallForResearchPapers)
  - ISPASS 2026: abstract 8 Dec 2025, paper 15 Dec 2025, 9 pages, with a "tool and benchmark papers" option. ISPASS 2027 is therefore probably mid-Dec 2026 (inferred). — [ISPASS 2026 submission](https://ispass.org/ispass2026/submission.php)
  - EuroMLSys 2026: deadline 24 Feb 2026, 6 pages, archival in ACM DL. — [EuroMLSys](https://euromlsys.eu/)
  - HPCA 2027: 20–24 Mar 2027. Its deadline was not retrieved and has presumably passed. — [casys-kaist deadlines](https://casys-kaist.github.io/)
  - Also passed: EuroSys 2027 fall (24 Sep 2026), ASPLOS 2027 Sept cycle (9 Sep 2026) and PPoPP 2027 (3 Aug 2026). — [sibling note venue_bar.md, citing CFPs](https://ppopp27.sigplan.org/track/PPoPP-2027-papers)
- **SPCL methodology.**
  - Hoefler & Belli's "Scientific Benchmarking of Parallel Computing Systems" (SC'15) rule 11 is "If possible, show upper performance bounds". — [PDF](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf)
  - SPCL's data-movement lineage includes "Data Movement Is All You Need" (MLSys 2021). — [PDF](https://htor.inf.ethz.ch/publications/img/data_movement_is_all_you_need.pdf)
  - SPCL publishes characterization work at IISWC, e.g. "Confidential LLM Inference: Performance and Cost Across CPU and GPU TEEs" (IISWC 2025). — [SPCL IISWC'25 PDF](https://spcl.inf.ethz.ch/Publications/.pdf/IISWC25_confidential_llms.pdf); [SPCL Publications](https://spcl.inf.ethz.ch/Publications/)
- **How prior work was framed:**
  - The closest (B) prior art quantifies gains against *efficient* baselines: BOOST reports a 7.4% microbenchmark gain vs a 10.4% ideal, and a +4.3% low-latency gain. — [arXiv 2609.13592](https://arxiv.org/html/2609.13592)
  - The closest (A) prior art quantifies "knees" in bandwidth-to-capacity ratio: 1.4–4.0 s⁻¹. — [arXiv 2609.15636](https://arxiv.org/html/2609.15636v1)

### Inferences
- **Direction (B): what remains open.**
  - The discrete-PCIe-GPU + CPU-compute split when the model fits, at batch 1, for MoE. MoE experts are natural independent units to place on the CPU; a dense model would need row-splitting.
  - A closed-form bound, expressed in η_g, GPU bandwidth, CPU bandwidth, the hand-off latency and per-layer sync count, that predicts when this helps and when it does not.
  - The bound must reproduce known points: FusionML's Apple zero (shared bandwidth) and HeteroInfer's mobile gain.
  - Validation should span platforms with different η_g, such as L4 (high η) and A10 (η≈0.5).
  - A low-latency GPU-signalled hand-off is a real contribution, since KTransformers uses `cudaLaunchHostFunc` (sibling notes).
- **Direction (B): feasibility.** 27 Oct: arXiv preprint plus "submitted to MLSys 2027", tight but possible. Dec: ISPASS 2027 with the η_g sweep and 1 consumer box is the best fit.
- **Direction (B): SPCL fit.** High, because the paper is organised around a bound (rule 11) and an explanation of data movement.
- **Direction (A): what remains open.** A client-class DSE with real routing traces and a validated model. It should answer which memory mix (VRAM size, DRAM channels, PCIe gen, unified vs discrete, CXL) makes MoE batch-1 fast per dollar.
- **Direction (A): feasibility.** 27 Oct: model-only DSE preprint feasible. Dec: publishable at ISPASS only if validated on ≥2 real client classes, which may require buying or borrowing hardware.
- **Direction (A): SPCL fit.** Good for modelling taste, but a DSE whose numbers are never measured on the target hardware conflicts with SPCL's measurement rigour. Also be blunt: Huawei's HBF paper, Cacheable-by-Design and Edge0 all appeared in Aug–Sep 2026, so the window is closing.
- **Combined framing, probably strongest for the ETH application.** Lead with (B) as the measured, bounded result. Then use the same model to run (A)'s DSE as "implications for client hardware design", which is the Accessible-AI thesis pitch.

### Gaps
- ISPASS 2027, IISWC 2027 and EuroMLSys 2027 CFP dates are not yet posted; the timings above are inferred from 2026.
- It is unknown whether the researcher can access any consumer machine (Strix Halo, Spark, Mac, or a DDR5 desktop with an RTX card). This decides whether (A) can be validated first-party by December.
- The PC-level acceptance appetite at ISPASS/IISWC for a model-driven DSE without first-party hardware was not assessed from reviews. No reviews were retrieved.
