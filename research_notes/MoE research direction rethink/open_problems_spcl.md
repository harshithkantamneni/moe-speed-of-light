# Open problems in efficient LLM/MoE inference on memory-constrained / consumer hardware (Sep 2026) and SPCL (Hoefler, ETH) research directions — candidate directions beyond the excluded seven

Scope note: excluded directions (not re-proposed here) are (1) meta-science/benchmarking of MoE offloading, (2) client-hardware DSE, (3) hardware-aware MoE architecture/routing locality, (4) CPU/GPU numerical divergence, (5) batch-1 GPU efficiency gap / CPU+GPU bandwidth aggregation, (6) next-token expert prediction / spec decoding with offloading, (7) off-policy teacher-forcing pitfalls. Date of research: 27 Sep 2026. Many arXiv items below are from Aug-Sep 2026 and were read via abstract/HTML summaries only; treat numbers as the authors' claims.

## Q1. What is SPCL working on in 2025-2026 (LLM inference, data movement, performance modeling, sparsity, quantization, networking for AI), and what has Hoefler said publicly about open problems?

### Takeaway
SPCL's 2025-2026 output is dominated by networking-for-AI (load balancing, congestion control, collectives, NCCL analysis, network simulation) and performance-analysis tooling, with a Besta-led stream on reasoning LMs (including an Aug 2026 performance model of RLMs) and an Alistarh-collaboration stream on quantization; there is no SPCL paper on consumer-hardware or MoE-offloaded inference on the publications page, which is a gap a candidate can fill in SPCL's idiom (data movement, I/O complexity, performance models, scientific benchmarking). Hoefler's public framing ("Scalable and Efficient AI: From Supercomputers to Smartphones") is closely aligned with the "Accessible AI" mission, but the publicly available statements are 2022-2023 vintage; no 2025-2026 Hoefler "open problems in inference" talk was found.

### Cited Findings
LLM inference / reasoning LMs / quantization (2025-2026):
- "Performance Foundations of Parallel & Distributed Reasoning Language Models" (Besta, ..., Hoefler), arXiv:2608.27046, Aug 2026 — [SPCL publications](https://spcl.inf.ethz.ch/Publications/)
  - Builds a work-depth-memory complexity framework for RLM post-training (PPO, GRPO, DPO) with generation / assessment / training stages; explicitly models KV cache memory (M_KV = 2(S+T)·L·d); notes per-trajectory generation depth O(T·D_f) cannot be parallelized away; targets datacenter clusters, consumer hardware not addressed — [arXiv HTML](https://arxiv.org/html/2608.27046v1)
  - Listed open problems: stage-fusion/placement policies for multi-model PPO pipelines; MoE routing load balancing under evolving policy; communication-aware expert placement; async staleness bounds — [arXiv HTML](https://arxiv.org/html/2608.27046v1)
- "Confidential LLM Inference: Performance and Cost Across CPU and GPU TEEs" (Chrapek, Copik, Mettaz, Hoefler), IISWC, Oct 2025 — the only explicitly LLM-inference performance-characterization paper on the page for 2025 — [SPCL publications](https://spcl.inf.ethz.ch/Publications/)
- "BLaST: High Performance Inference and Pretraining using BLock Sparse Transformers" (Okanovic et al., Hoefler), arXiv:2507.03117, Jul 2025 — [SPCL publications](https://spcl.inf.ethz.ch/Publications/)
- "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on LLMs" (Frantar, Castro, Chen, Hoefler, Alistarh), PPoPP'25, Feb 2025; "QuaRot" NeurIPS'24; "HALO: Hadamard-Assisted Lower-Precision Optimization" NeurIPS'25; "Beyond Outliers: A Study of Optimizers Under Quantization" ICLR'26 — [SPCL publications](https://spcl.inf.ethz.ch/Publications/)
- "Reasoning Language Models: A Blueprint" arXiv:2501.11223 (listed Jun 2025); "Demystifying Chains, Trees, and Graphs of Thoughts" IEEE TPAMI Dec 2025; "Affordable AI Assistants with Knowledge Graph of Thoughts" arXiv:2504.02670; "Process Reward Agents..." ICML'26 workshop; "Multi-Head RAG" — [SPCL publications](https://spcl.inf.ethz.ch/Publications/)
- "Apertus: Democratizing Open and Compliant LLMs for Global Language Environments" ACL'26 (Hoefler among 30+ authors) — [SPCL publications](https://spcl.inf.ethz.ch/Publications/)

Sparsity / precision:
- "STen: Productive and Efficient Sparsity in PyTorch" TACO'26; "Memory-Efficient LLM Training with Dynamic Sparsity" ICML'26; "When Data Is Scarce: Scaling Sparse Language Models with Repeated Training" ICML'26; "What Bloats Your Floats? Right-Sizing Numerical Precision for Scientific Computing" (2026) — [SPCL publications](https://spcl.inf.ethz.ch/Publications/)

Performance modeling / analysis / portability:
- "Cost-Effective Empirical Performance Modeling" (Ritter, ..., Hoefler, Wolf) IEEE TPDS Feb 2026 (Extra-P lineage); "C.A.T.S.: Memory and Control Flow Tracing for Whole-Program Performance Analysis" SC'25; "EDAN: Towards Understanding Memory Parallelism and Latency Sensitivity in HPC" ICS'25; "SecPerf: Demystifying Cost of Confidential HPC" IPDPS'26; "PerfDojo: Automated ML Library Generation for Heterogeneous Architectures" SC'25; "XaaS Containers: Performance-Portable Representation With Source and IR Containers" SC'25 (Best Paper finalist) — [SPCL publications](https://spcl.inf.ethz.ch/Publications/)
- Energy: "Core Hours and Carbon Credits: Incentivizing Sustainability in HPC" SC'25; "Energy-Optimal and Low-Depth Algorithmic Primitives for Spatial Dataflow Architectures" IPDPS'25 — [SPCL publications](https://spcl.inf.ethz.ch/Publications/)

Networking for AI (largest cluster):
- "PICO: Performance Insights for Collective Operations" ISC'26 (Hans Meuer best paper); "REPS" EuroSys'26; "Spritz" IPDPS'26; "EvalNet" IPDPS'26; "In-Network Collective Operations: Game Changer or Challenge for AI Workloads?" IEEE Computer Jan 2026; "Flowcut Switching" ToN 2026; "Network Design for Wafer-Scale Systems..." DAC'26; SC'25: "SDR-RDMA", "Bine Trees", "ATLAHS" (network simulator toolchain for AI/HPC, best student paper finalist), "Uno"; "Demystifying NCCL" HOTI'25; "SMaRTT"; "RailX"; "CrossPipe" ATC'25 — [SPCL publications](https://spcl.inf.ethz.ch/Publications/)
- No SPCL 2025-2026 entry on the publications page mentions MoE offloading, consumer/edge inference, KV-cache offloading, or llama.cpp — [SPCL publications](https://spcl.inf.ethz.ch/Publications/)

Hoefler's public statements:
- ICPP 2023 opening keynote "Scalable and Efficient AI: From Supercomputers to Smartphones": data movement is the bottleneck ("Data Movement Is All You Need", 30% over PyTorch on BERT); quantization of GPT-3 from 700 GiB FP32 to 66 GiB at 3-bit to fit a single A100 with "2-4x faster generative inference"; 95% sparse models at "essentially same quality"; I/O (a single H100 needs ~50 SSDs for random ImageNet access); future: accelerators converging on low precision + sparsity + memory-coupled compute, "performance metaprogramming" via SDFG IRs, Ultra Ethernet — [ICPP keynote slides](https://spcl.inf.ethz.ch/Publications/.pdf/hoefler-keynote-icpp.pdf)
- Same talk abstract (ISI, 12 Jun 2023): "Efficient inference is even more challenging ... These techniques may eventually enable inference with powerful models on hand-held devices." — [ISI event page](https://www.isi.edu/events/3921/scalable-and-efficient-ai-from-supercomputers-to-smartphones/); talk also given at Microsoft Research and elsewhere — [MSR video](https://www.microsoft.com/en-us/research/video/scalable-and-efficient-ai-from-supercomputers-to-smartphones/)
- Axelera interview (22 Feb 2022): "We need to seriously start thinking about data movement... the data-centric view needs to percolate into programming systems and architectures"; advice to young researchers: "Learn about I/O complexity!"; data-centric computing applies from "smallest portable devices to largest supercomputers" — [Axelera interview](https://community.axelera.ai/product-updates/interview-with-torsten-hoefler-axelera-ai-s-scientific-advisor-113)
- Adjacent (not SPCL) framing that SPCL-style reviewers will know: Ma & Patterson (Google), "Challenges and Research Directions for LLM Inference Hardware", arXiv:2601.05047, 8 Jan 2026 (rev. 6 Feb 2026): memory and interconnect, not compute, bound decode; proposes high-bandwidth flash ("10X memory capacity with HBM-like bandwidth"), processing-near-memory, 3D memory-logic stacking, low-latency interconnect; also discusses mobile applicability — [arXiv 2601.05047](https://arxiv.org/abs/2601.05047)

### Inferences
- SPCL taste signals, in rough order of weight: (a) first-principles performance models with validation (Extra-P, work-depth-memory for RLMs, EDAN), (b) data movement / I/O complexity as the organizing lens, (c) networking and collectives for AI, (d) "demystifying X" characterization papers (NCCL, Chains/Trees/Graphs of Thoughts, confidential HPC, higher-order GNNs), (e) quantization/sparsity via the Alistarh line. A consumer-hardware MoE paper framed as "Demystifying ..." or as a data-movement/I/O-complexity model fits (a)+(b)+(d).
- The Aug 2026 RLM performance-foundations paper is the most natural hook for an ETH AI Center / SPCL pitch: it is datacenter-only and explicitly leaves "communication-aware expert placement" and MoE routing open; extending its work-depth-memory framing to a single consumer node with a memory hierarchy (host DRAM / PCIe / VRAM / SSD) is a direct, legible extension.
- The "Supercomputers to Smartphones" talk title and abstract are the best public quote to align "Accessible AI" with Hoefler's own stated trajectory; cite the 2023 abstract rather than claiming a recent statement.

### Gaps
- No 2025-2026 Hoefler talk/slides listing open problems specifically for LLM inference efficiency were found (searches returned conference-agenda noise); a YouTube/ISC 2026/SC'25 keynote check was not completed.
- "Chatbot energy" SPCL work mentioned in the assignment was not found on the publications page listing; could not confirm it exists.
- Hoefler's current extra-university affiliations (e.g., CSCS chief architect, industry advisory roles) not verified in this session.
- "Data Movement Is All You Need" (MLSys'21) and SpQR are pre-2025 and were not in the 2025-2026 listing; not re-verified.

## Q2. What are the community pain points in local LLM inference in 2026 (long-context decode, agentic prefix reuse, prefill-heavy workloads, energy, 1T-param MoE on desktops, SSD offload)?

### Takeaway
The loudest 2026 local-inference pain points are (i) prompt re-processing in agentic/coding sessions, now worse for SWA/hybrid-recurrent models whose state cannot be partially rewound in llama.cpp, (ii) KV cache competing with weights for VRAM at long context (KV offload helps restore but not decode), (iii) the PCIe "VRAM wall" for hybrid offload, and (iv) running 0.3-3 TB MoEs from SSD or across Mac clusters; SSD streaming and energy benchmarking have become crowded in the last two months.

### Cited Findings
Agentic prefix reuse / hybrid models:
- llama.cpp cannot partially rewind SWA or hybrid recurrent (Mamba/linear-attention) state; when no context checkpoint covers the resume position the server sets n_past=0 and re-processes the whole prompt; the log line appears only at verbosity 4 — [Particula blog](https://particula.tech/blog/prompt-reprocessing-swa-hybrid-models-kv-cache)
- Issue #22746 (Jul 2026): one rejected request deleted 15 checkpoints totalling 2.19 GiB on a 65,536-token slot; checkpoints 62.8-213.7 MiB each; TTFT flat across turns (6.03/6.17/6.13 s at 16,946 tokens) i.e. no reuse — [Particula blog](https://particula.tech/blog/prompt-reprocessing-swa-hybrid-models-kv-cache)
- Open: disk-backed checkpoints (issue #20697, no maintainer response as of Jul 2026); on unified-memory systems (Strix Halo, Apple Silicon) `--cache-ram` competes with weights for the same pool; PR #22929 (25 May 2026) replaced `--checkpoint-every-n-tokens` with `--checkpoint-min-step` (eviction floor, default 8192) causing silent checkpoint loss for short agent turns — [Particula blog](https://particula.tech/blog/prompt-reprocessing-swa-hybrid-models-kv-cache)
- vLLM's hybrid KV coordinator supports exactly two attention types; SGLang uses radix caches with int8-compressed recurrent states — [Particula blog](https://particula.tech/blog/prompt-reprocessing-swa-hybrid-models-kv-cache)
- Claude Code against llama.cpp (21 Jun 2026): a changing attribution header in the system prompt broke prefix reuse, forcing re-processing of "tens of thousands of tokens"; Qwen 3.6-27B on RTX 4090 at 170K context; checkpoint restore 511 ms for 212 tokens after fix — [Aleksandrov blog](https://www.mykolaaleksandrov.dev/posts/2026/06/claude-code-llamacpp-prompt-cache-fix/)
- Closest academic prior art on hybrid-model prefix caching: Marconi (MLSys'25, Outstanding Paper honorable mention), datacenter-oriented — [arXiv 2411.19379](https://arxiv.org/abs/2411.19379); [GitHub](https://github.com/ruipeterpan/marconi)

Long-context KV on consumer hardware:
- 70B at 128K context: KV cache ~40 GB vs ~42.5 GB Q4_K_M weights; llama.cpp slot save/restore: 5K-token chat 9.9 s cold re-prefill vs 1.4 s restore from disk (7x); vLLM+LMCache GPU->RAM->NVMe tiering: 3.0x lower mean TTFT, 2.3x more requests on multi-turn agentic workloads; Ollama has no KV-to-disk as of v0.32.5 (27 Jul 2026) — [runaihome KV offload](https://runaihome.com/blog/nvme-kv-cache-offloading-local-llm-consumer-gpu-2026/)
- Unsolved: "During decoding, the active context's KV must be in VRAM"; below ~2K tokens offload round-trip costs more than recompute; SATA SSD adds ~13x stall; KV files are a privacy exposure — [runaihome KV offload](https://runaihome.com/blog/nvme-kv-cache-offloading-local-llm-consumer-gpu-2026/)
- LMCache reported a new architecture boosting MoE inference by 10x (Apr 2026) — [LMCache blog](https://blog.lmcache.ai/en/2026/04/03/lmcaches-new-architecture-boosts-moe-inference-performance-by-10x/); vLLM RFC #38256 proposes incremental MoE expert offloading with GPU cache + async pipeline — [vLLM issue](https://github.com/vllm-project/vllm/issues/38256)

VRAM wall / backends:
- "Silicon Showdown" (Javat & Kazakov, arXiv 2605.00519): CPU offloading via PCIe "reduces throughput by over 90%"; Apple M-series up to 23x tokens/joule advantage; M4 Pro beats dual-die M3 Ultra on MoE; RTX 4090 beats RTX 5090 by >2.2x TTFT in TensorRT-LLM v1.1.0 (software lag); new architectures (GLM-4.7-Flash) lack TRT-LLM support, forcing llama.cpp without NVFP4 — [arXiv 2605.00519](https://arxiv.org/html/2605.00519v1)
- Backend portability is a live community topic: llama.cpp discussions on Vulkan (#10879), ROCm/HIP (#15021), Vulkan vs CUDA (#23109); FOSDEM 2026 talk "Vulkan API for Machine Learning? Competing with CUDA and ROCm in llama.cpp"; Phoronix ROCm 7.1 vs RADV Vulkan on Radeon AI PRO R9700 — [Discussion #10879](https://github.com/ggml-org/llama.cpp/discussions/10879); [#15021](https://github.com/ggml-org/llama.cpp/discussions/15021); [#23109](https://github.com/ggml-org/llama.cpp/discussions/23109); [FOSDEM 2026](https://fosdem.org/2026/schedule/event/CZSPSC-llama-cpp-vulkan/); [Phoronix](https://www.phoronix.com/review/rocm-71-llama-cpp-vulkan)

1T-class MoE on desktops (SSD / clusters):
- SSD-LLaMA (Liang et al., 16 Sep 2026): RTX 5090 + 16 GB DRAM + 9 GiB/s NVMe; Kimi-K3 (2.8 TB) at 0.465 tok/s decode, 1.217 tok/s prefill; expert-pack layout, SSD-RAM-VRAM tiers, lossless rANS with CUDA decompression, async I/O, CPU-GPU expert split; 77.7% of peak SSD bandwidth vs 43.2% baseline; no analytical model; KV cache not addressed — [arXiv 2609.18110](https://arxiv.org/html/2609.18110v1)
- "The Other Half of the Memory Wall" / Edge0 (AutoArk, Sep 2026): trained per-layer "prerouter" whose prediction is used as the routing itself; Qwen3.6-35B-A3B at 20.4 tok/s on Mac mini M4 Pro 24 GB with 2.9 GiB active memory vs 3.9 tok/s fully resident; fitted load cost "1.17 ms + 1.33 ms x (cold fraction)"; AIME gaps of 6-10 points; "adjacent tokens agree on only about a quarter of a layer's experts"; 44 ms CPU graph-build floor — [arXiv 2609.18063](https://arxiv.org/html/2609.18063)
- FlashMoE (Jan 2026): ML-based cache replacement for SSD-offloaded MoE on edge — [arXiv 2601.17063](https://arxiv.org/abs/2601.17063)
- Practitioner post on streaming DeepSeek-V4 from RAM (Sep 2026) — [Luke Osborne](https://lukeosborne.au/2026/09/streaming-massive-moe-models-running-deepseek-v4-in-ram/)
- Mac clusters: macOS Tahoe 26.2 added RDMA over Thunderbolt 5; latency ~300 us -> 5-9 us; 80 Gb/s; exo+MLX: 2 nodes 26.2 tok/s and 4 nodes 31.9 tok/s with RDMA vs 17.2 and 15.2 tok/s without (4 nodes slower than 2 without RDMA); Kimi K2 ~25 tok/s, DeepSeek V3.1 ~32.5 tok/s; single-user only — [runaihome Mac cluster](https://runaihome.com/blog/mac-studio-cluster-rdma-thunderbolt5-trillion-parameter-2026/)

Energy:
- Energy benchmarking of local inference is now crowded: GreenBench (Apple Silicon, arXiv 2608.28667, Aug 2026); "Energy Efficiency of Locally Deployed LLMs: A Preliminary Quantitative GPU Power Benchmark on Consumer Hardware" (arXiv 2608.00008); "Watt Counts" (arXiv 2604.09048); "Scaling Laws for Energy Efficiency of Local LLMs" (arXiv 2512.16531) — [GreenBench](https://arxiv.org/abs/2608.28667); [2608.00008](https://arxiv.org/html/2608.00008v1); [Watt Counts](https://arxiv.org/html/2604.09048v1); [Scaling Laws](https://arxiv.org/html/2512.16531)
- Analytical energy model "From Tokens to Watt-hours" (Vartziotis et al., 29 Jul 2026): H100-class, dense FLOP accounting, decomposes into compute / parameter-access / KV-write / attention-read energy, separates prefill and decode; no MoE or CPU offload mentioned — [arXiv 2607.26571](https://arxiv.org/abs/2607.26571)

### Inferences
- Agentic sessions on local hardware are prefill/reuse-dominated, and the 2026 shift to hybrid (SWA/linear/recurrent) architectures broke the simple "KV prefix cache" story in llama.cpp; this is under-studied academically for consumer hardware (Marconi is datacenter).
- "Decode needs active KV in VRAM" plus MoE expert offload means that at long context KV and experts compete for the same VRAM bytes; practitioners discuss each separately.
- SSD expert streaming went from open to crowded within Sep 2026 (two papers in the same week plus FlashMoE); a standalone "SSD bounds" paper would now need to beat/explain SSD-LLaMA and Edge0.
- The Mac RDMA numbers (4 nodes slower than 2 without RDMA) are a clean example of latency-bound batch-1 communication that an SPCL-style LogGP-type model would explain.

### Gaps
- r/LocalLLaMA threads were not directly fetched (reddit content not retrieved); community evidence is from blogs and llama.cpp issues/discussions.
- Silicon Showdown date: arXiv ID 2605.00519 implies May 2026, while the fetched summary described it as 2025; exact date unconfirmed.
- No quantitative data found on multi-model local setups (e.g., running draft + main + embedding models concurrently).

## Q3. What do recent surveys and papers (2025-2026) list as open challenges for efficient MoE / edge inference, and how crowded is each sub-area?

### Takeaway
Offloaded/edge MoE serving is very crowded at the systems level (expert caching, prefetching, SSD tiers, adaptive precision, CPU-GPU split), and a Berkeley/UT paper (FreeToken, Aug 2026) now publishes a closed-form CPU/GPU split for missed experts, which directly overlaps excluded direction (5). Validated predictive models for MoE on consumer hardware remain thin: the main hardware-agnostic predictor (AMD LIFE) is dense-only and lists MoE as future work.

### Cited Findings
- FreeToken (Yang, Fan, Xu et al., UC Berkeley / UT Austin, 17 Aug 2026): "bandwidth-adaptive execution" divides missed expert accesses between PCIe transfer to GPU cache and direct CPU execution with closed-form split q* ~ m·B_p/B_h (PCIe vs host expert bandwidth); tested RTX 4060 laptop (8 GB) to RTX PRO 6000 (96 GB); DeepSeek-V4-Flash, Qwen3.6-35B-A3B, GLM-5.2; RTX 5090: 77-83 tok/s (Qwen3.6), 22-25 tok/s (DeepSeek-V4-Flash), 1.5-2.3x over baselines; 8 GB laptop 39.3 tok/s on 35B; mean TTFT < 44 s vs > 150 s baselines under agentic workloads — [arXiv 2608.16157](https://arxiv.org/html/2608.16157v1)
- MoE-APEX: adaptive-precision expert offloading, ASPLOS'26 — [ACM DL](https://dl.acm.org/doi/10.1145/3779212.3790187)
- Other edge MoE systems: OD-MoE (on-demand expert loading, cacheless edge-distributed, Dec 2025) — [arXiv 2512.03927](https://arxiv.org/html/2512.03927v1); Fate (cross-layer gate prefetching, 2025) — [arXiv 2502.12224](https://arxiv.org/html/2502.12224v2); latency-optimized expert placement for distributed edge MoE (Aug 2025) — [arXiv 2508.12851](https://arxiv.org/html/2508.12851v4)
- Survey: "A Survey on Inference Optimization Techniques for Mixture of Experts Models", ACM Computing Surveys — [ACM DL](https://dl.acm.org/doi/10.1145/3794845) (full text not retrievable, 403); "Efficient Inference for Edge Large Language Models: A Survey" (Tsinghua Science and Technology) — [SciOpen](https://www.sciopen.com/article/10.26599/TST.2025.9010166); edge-cloud collaborative survey with open challenges (Jul 2025) — [arXiv 2507.16731](https://arxiv.org/html/2507.16731v1)
- LIFE (Patwari, Sirasao, Das, AMD; arXiv 2508.00904): hardware-agnostic analytical forecasting of TTFT/TPOT/TPS; validated on Ryzen 9 HX 370 CPU, Ryzen AI Max+ 395 NPU/iGPU, V100; models KV growth, quantization, LoRA, MHA/GQA/MQA/MLA; MoE and VLMs explicitly future work; no hybrid CPU+GPU offload; mostly Llama2-7B — [arXiv 2508.00904](https://arxiv.org/html/2508.00904)
- A community "inference-predictor" repo (analytical capacity, roofline latency, scheduling simulator) exists — [GitHub](https://github.com/yehfelareborn/inference-predictor); VRAM-calculator style guides are heuristic — [localllm.in](https://localllm.in/blog/llamacpp-vram-requirements-for-local-llms)
- Sparse-attention KV offload is active and datacenter-leaning: HiSparse (hierarchical KV for sparse-attention decoding, arXiv 2608.07009), FlashMemory-DeepSeek-V4 (lookahead sparse attention, KV to 13.5%, arXiv 2606.09079), ScoutAttention (layer-ahead CPU pre-computation for KV offload, arXiv 2603.27138), SparDA (arXiv 2606.04511) — [HiSparse](https://arxiv.org/html/2608.07009); [FlashMemory](https://arxiv.org/abs/2606.09079); [ScoutAttention](https://arxiv.org/abs/2603.27138); [SparDA](https://arxiv.org/pdf/2606.04511)
- Distributed consumer inference: prima.cpp (30-70B on heterogeneous home clusters, arXiv 2504.08791, Apr 2025) — [arXiv 2504.08791](https://arxiv.org/abs/2504.08791); NCCL EP (LL mode for 1-128 token decode all-to-all over RDMA, datacenter H100, arXiv 2603.13606) — [arXiv 2603.13606](https://arxiv.org/abs/2603.13606v3)

### Inferences
- Crowdedness ranking (most to least) for consumer/edge MoE: expert caching/prefetch/SSD tiers (very high) > energy benchmarking (high, 5+ papers in 2026) > KV offload incl. sparse attention (high, datacenter) > CPU/GPU split (now high after FreeToken) > backend portability (high practitioner, low academic) > agentic/hybrid-state reuse on local hardware (medium-low academic) > validated MoE-aware predictors for consumer hardware (low) > distributed consumer MoE over RDMA/TB5 (low academic) > test-time-compute (parallel sampling) on offloaded MoE (low, none found).
- FreeToken should be treated as a must-cite and a threat for excluded direction (5); its q* formula is a special case of a bytes-over-bandwidth model, so any paper using the researcher's decode model must position against it.

### Gaps
- The ACM CSUR MoE inference survey text (and its open-challenges list) could not be retrieved (403).
- Did not verify KTransformers (SOSP'25), MoE-Lightning, MoE-Gen, Fiddler, DFloat11, or HOBBIT details in this session; they are likely prior art for several candidates below and must be checked.
- Workshop CFPs (EuroMLSys 2027, MLArchSys 2027, ESFoMo) were not checked.

## Q4. Candidate directions NOT on the excluded list: concrete ideas, closest prior art (with dates), asset reuse, SPCL fit, crowdedness

### Takeaway
Best fits by (openness x asset reuse x SPCL taste): (A) test-time-compute (parallel sampling / best-of-n) on offloaded MoE, analysed GPU-free with the routing traces; (B) joint KV-vs-expert VRAM budgeting for long-context MoE decode; (C) latency-bound expert-parallel decode across consumer nodes (RDMA over Thunderbolt 5 / Ethernet) with a LogGP-style model; (D) agentic prefix-reuse and hybrid-state checkpoint-vs-recompute on local hardware. Secondary: (E) validated MoE-aware "will it run, how fast" predictor, (F) I/O-complexity lower bound for MoE decode, (G) energy of hybrid CPU+GPU offload, (H) cross-backend portability as fraction-of-SOL, (I) when compression of offloaded experts pays.

### Cited Findings
Prior-art anchors per candidate (all cited above; restated with dates for the writer):
- (A) Test-time compute: SPCL "Reasoning Language Models: A Blueprint" (arXiv 2501.11223) and "Performance Foundations of Parallel & Distributed RLMs" (arXiv 2608.27046, Aug 2026, datacenter only) — [SPCL publications](https://spcl.inf.ethz.ch/Publications/); Edge0 reports adjacent tokens share only ~1/4 of a layer's experts (Sep 2026) — [arXiv 2609.18063](https://arxiv.org/html/2609.18063); an energy/test-time-scaling paper exists in a 2026 Cell Press journal — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2542435126001145)
- (B) KV vs experts: "active context's KV must be in VRAM" during decode — [runaihome](https://runaihome.com/blog/nvme-kv-cache-offloading-local-llm-consumer-gpu-2026/); SSD-LLaMA does not address KV (Sep 2026) — [arXiv 2609.18110](https://arxiv.org/html/2609.18110v1); HiSparse (Aug 2026), ScoutAttention (Mar 2026), FlashMemory-DeepSeek-V4 (Jun 2026) — [HiSparse](https://arxiv.org/html/2608.07009); [ScoutAttention](https://arxiv.org/abs/2603.27138); [FlashMemory](https://arxiv.org/abs/2606.09079); vLLM RFC #38256 — [vLLM](https://github.com/vllm-project/vllm/issues/38256)
- (C) Consumer multi-node: RDMA over TB5 numbers (Jul 2026 article) — [runaihome](https://runaihome.com/blog/mac-studio-cluster-rdma-thunderbolt5-trillion-parameter-2026/); prima.cpp (Apr 2025) — [arXiv 2504.08791](https://arxiv.org/abs/2504.08791); OD-MoE (Dec 2025) — [arXiv 2512.03927](https://arxiv.org/html/2512.03927v1); NCCL EP (2026) — [arXiv 2603.13606](https://arxiv.org/abs/2603.13606v3); SPCL "Demystifying NCCL" (HOTI'25), PICO (ISC'26), ATLAHS (SC'25) — [SPCL publications](https://spcl.inf.ethz.ch/Publications/)
- (D) Agentic/hybrid state: llama.cpp issues #22746, #20697, PR #22929 (May-Jul 2026) — [Particula](https://particula.tech/blog/prompt-reprocessing-swa-hybrid-models-kv-cache); Marconi (MLSys'25) — [arXiv 2411.19379](https://arxiv.org/abs/2411.19379); FreeToken agentic TTFT (Aug 2026) — [arXiv 2608.16157](https://arxiv.org/html/2608.16157v1)
- (E) Predictor: LIFE (AMD, 2025, dense only) — [arXiv 2508.00904](https://arxiv.org/html/2508.00904); crowdsourced llama.cpp benchmark threads usable as validation corpora — [#10879](https://github.com/ggml-org/llama.cpp/discussions/10879), [#15021](https://github.com/ggml-org/llama.cpp/discussions/15021); SPCL "Cost-Effective Empirical Performance Modeling" TPDS'26 — [SPCL publications](https://spcl.inf.ethz.ch/Publications/)
- (F) I/O complexity: Hoefler "Learn about I/O complexity!" — [Axelera](https://community.axelera.ai/product-updates/interview-with-torsten-hoefler-axelera-ai-s-scientific-advisor-113); FlashMoE ML-based cache replacement (Jan 2026) — [arXiv 2601.17063](https://arxiv.org/abs/2601.17063)
- (G) Energy: GreenBench (Aug 2026), Tokens-to-Watt-hours (Jul 2026, dense H100) — [GreenBench](https://arxiv.org/abs/2608.28667); [2607.26571](https://arxiv.org/abs/2607.26571); SSD-LLaMA flags energy as a concern — [arXiv 2609.18110](https://arxiv.org/html/2609.18110v1)
- (H) Portability: community threads + FOSDEM 2026 — [FOSDEM](https://fosdem.org/2026/schedule/event/CZSPSC-llama-cpp-vulkan/); SPCL XaaS containers (SC'25) and PerfDojo (SC'25) — [SPCL publications](https://spcl.inf.ethz.ch/Publications/)
- (I) Compression: SSD-LLaMA lossless rANS + CUDA decompression (Sep 2026) — [arXiv 2609.18110](https://arxiv.org/html/2609.18110v1); MoE-APEX adaptive precision offloading (ASPLOS'26) — [ACM DL](https://dl.acm.org/doi/10.1145/3779212.3790187)

### Inferences
Candidate list (my synthesis; "open" means no directly matching paper found in this search, not a guarantee):

A. Test-time compute on offloaded MoE: "parallel sampling amortizes expert streaming."
- Question: with n parallel samples (best-of-n, self-consistency, tree search), each decode step fetches the union of experts routed by n tokens; how does union size grow with n (sublinear?), and at what n does tokens/s per sample or accuracy-per-second peak on a PCIe/SSD-bound consumer box? Contrast with datacenter where batch amortization is standard.
- Why open: SPCL's RLM performance-foundations paper (Aug 2026) is datacenter-only; offloading systems (SSD-LLaMA, Edge0, FreeToken) optimize single-stream decode. No consumer-hardware test-time-scaling-under-offload model found.
- Asset reuse: very high. Expert-union growth vs n can be computed GPU-free from exact routing traces (need multi-sample traces; self-generated corpora + llama.cpp instrumentation); bytes-over-bandwidth model extends directly (bytes = f(union)); A10 hybrid path measures batched decode cost; cloud runner for validation.
- SPCL fit: very high (RLM blueprint, GoT/ToT, work-depth-memory framing). Good ETH AI Center pitch: "Accessible reasoning."
- Crowdedness: low (to my search). Risk: overlaps conceptually with (6) speculative decoding (both use batch amortization) — keep it about sampling-based reasoning and accuracy per joule/second.
- Deadlines: feasible core result (trace-based union curves + model + A10 validation) for MLSys 30 Oct is tight but plausible; better as ETH AI Center proposal centerpiece and ISPASS/MLSys-workshop paper.

B. Joint VRAM budgeting: KV cache vs resident experts for long-context MoE decode at batch 1.
- Question: given VRAM V, host BW, PCIe BW, and context length L, what split of VRAM between KV and hot experts minimizes step time, and at what L does moving KV (with CPU attention, as llama.cpp `--no-kv-offload` allows) beat moving experts? How do MLA and DeepSeek-style sparse attention (top-k KV reads) shift the crossover?
- Why open: KV-offload papers (HiSparse, ScoutAttention, FlashMemory) are datacenter/dense-oriented; MoE-offload papers (SSD-LLaMA explicitly) ignore KV; practitioner guides treat them separately.
- Asset reuse: high. Add a KV-bytes term to the decode model; A10 measurements sweep context length; routing traces unaffected; speed-of-light placement bound generalizes to a two-resource knapsack.
- SPCL fit: high (data movement, memory hierarchy modeling, EDAN-style latency sensitivity).
- Crowdedness: medium (components crowded, the joint consumer question not found). Risk: someone adds KV tiering to FreeToken/KTransformers.
- Deadlines: model + A10 validation is plausible for MLSys; clean ISPASS characterization.

C. Latency-bound MoE decode across consumer nodes (RDMA over Thunderbolt 5, 10/25 GbE, llama.cpp RPC, exo/MLX).
- Question: per-token cost of pipeline vs tensor vs expert parallelism at batch 1 when each layer's collective is a few KB and link latency dominates (5-9 us RDMA vs ~300 us TCP); explain why 4 nodes w/o RDMA were slower than 2 (15.2 vs 17.2 tok/s) and predict scaling.
- Why open: prima.cpp (2025) and OD-MoE (2025) are systems papers; NCCL EP LL mode is datacenter; no LogGP-style validated model for consumer-interconnect MoE decode found.
- Asset reuse: medium-high. Bytes-over-bandwidth model gains an alpha (latency) term; the per-layer hand-off cost methodology (~84 us CPU hand-off, copy-engine contention) is directly analogous to per-layer network hops; routing traces give per-token all-to-all volume and imbalance; cloud runner can emulate multi-node over Ethernet (not TB5).
- SPCL fit: very high (networking for AI is SPCL's largest cluster: PICO, ATLAHS, Demystifying NCCL, LogGP heritage).
- Crowdedness: low academically, high practitioner activity. Risk: needs 2-4 physical nodes for TB5 claims; cloud-only emulation weakens the consumer story.
- Deadlines: ISPASS-sized characterization; strong for the ETH application narrative.

D. Agentic sessions on local hardware: prefix reuse, hybrid-state checkpoint-vs-recompute, and MoE prefill with offloaded experts.
- Question: characterize coding-agent traces (16K-170K contexts, many short turns) on llama.cpp; model TTFT as min(recompute prefill, restore checkpoint from RAM/SSD) for SWA/recurrent hybrids; for MoE prefill with host-resident experts, model the crossover chunk size at which streaming all experts over PCIe beats CPU compute.
- Why open: Marconi is datacenter; llama.cpp issues show the problem is unsolved (disk-backed checkpoints open, semantics changed May 2026); FreeToken reports agentic TTFT but not a reuse/checkpoint model.
- Asset reuse: medium. llama.cpp instrumentation, self-generated corpora, cloud runner reuse well; prefill cost model is new work; routing traces help for prefill expert coverage.
- SPCL fit: medium-high (workload characterization + model; RLM/agent line).
- Crowdedness: medium-low for local; datacenter agentic characterization is crowded. Risk: fast-moving engine fixes can obsolete findings within months (pre-register versions).

E. Validated MoE-aware "will it run and how fast" predictor for consumer/hybrid setups.
- Question: predict TTFT/TPOT for (model, quant, VRAM, host BW, PCIe gen, backend, context) including partial expert offload, validated out-of-sample against crowdsourced llama-bench threads and first-party runs.
- Why open: LIFE is dense-only (MoE future work); calculators are heuristic.
- Asset reuse: very high (52-measurement validation, 2.9% pre-registered step model).
- SPCL fit: high (Extra-P / empirical performance modeling, scientific benchmarking).
- Crowdedness: low-medium. Risk: overlaps with excluded (2) client-hardware DSE and (1); position it as a tool + validation methodology rather than DSE. Likely a workshop/tool paper, not an MLSys main-track novelty.

F. I/O-complexity lower bound for MoE decode on a two/three-level hierarchy.
- Question: formalize the time-domain speed-of-light placement bound as an I/O lower bound (pebbling-style / Belady-MIN on routing traces) with capacity M, giving a model-and-trace-specific floor any offloading system must respect; report how far SSD-LLaMA/Edge0/FreeToken-reported numbers are from it.
- Why open: offloading papers use empirical caches (FlashMoE uses ML-based replacement); none found stating a formal lower bound.
- Asset reuse: very high (the SOL bound and traces already exist).
- SPCL fit: very high ("Learn about I/O complexity!").
- Crowdedness: low. Risk: reviewers may see it as a re-framing of existing asset/(1); strongest as a theory section inside A or B rather than standalone.

G. Energy of hybrid CPU+GPU MoE offload: energy-optimal vs time-optimal split.
- Question: decompose J/token into GPU HBM reads, host DRAM reads, PCIe transfers, and idle/static power; show the energy-optimal expert placement differs from the latency-optimal one.
- Why open: 2026 energy papers are Apple Silicon/dense/H100-analytical; none found on hybrid offload.
- Asset reuse: medium (model extends with pJ/byte terms; A10 NVML power available).
- SPCL fit: medium (sustainability, energy-optimal primitives).
- Crowdedness: high for energy benchmarking in general. Risk: cloud VMs rarely expose CPU RAPL, so host-side energy needs owned hardware.

H. Cross-backend performance portability of llama.cpp as fraction-of-speed-of-light per op.
- Question: for CUDA, Vulkan, HIP, Metal, SYCL, measure decode matvec / mul_mat_id / attention / prefill GEMM as % of the bytes or FLOP bound; attribute gaps to kernel quality vs API/launch overhead.
- Why open: evidence is forum threads, Phoronix, FOSDEM; no rigorous SOL-attribution paper found.
- Asset reuse: medium (SOL model; GPU runner is NVIDIA-centric, so Vulkan-on-NVIDIA vs CUDA is easy, AMD/Apple need other hardware).
- SPCL fit: medium-high (XaaS performance portability, PerfDojo).
- Crowdedness: high practitioner / low academic. Risk: partially overlaps excluded (5) batch-1 efficiency gap; differentiate by cross-backend scope.

I. When does compressing offloaded experts pay?
- Question: closed-form condition for lossless/lossy expert compression to reduce step time: compression ratio x link BW vs GPU decompression throughput and CPU-side decode; map across PCIe 4/5, SSD, and TB5 links.
- Why open: SSD-LLaMA shows rANS helps empirically but gives no model; MoE-APEX adapts precision.
- Asset reuse: high (one extra term in the bytes model).
- SPCL fit: high (compression line: psit, error-bounded compression, quantization).
- Crowdedness: medium-high and rising. Best as a section of B or E, not standalone.

Overall bluntness:
- Avoid as standalone: SSD streaming bounds (two papers in Sep 2026 week), energy benchmarking (5+ in 2026), sparse-attention KV offload (4+ in 2026), and anything resembling CPU/GPU split (FreeToken, Aug 2026).
- Must-cite threats for the researcher's existing framing: FreeToken (q* ~ m·B_p/B_h closed form) and Edge0 (routing prediction consumed as routing, 20.4 tok/s on 24 GB Mac).

### Gaps
- For candidate A, did not verify batch/multi-sample expert-union studies in prior offloading work (MoE-Lightning, MoE-Gen, Klotski); must be checked before claiming novelty.
- For candidate C, did not find any peer-reviewed paper on exo/MLX RDMA-over-TB5 performance; numbers come from a practitioner blog.
- For candidate D, KTransformers (SOSP'25) prefill-offload details not verified.
- For candidate F, SPCL's own red-blue pebbling / I/O lower-bound papers (e.g., COSMA-era work) were not re-verified in this session.
- Hoefler/Belli "Scientific Benchmarking of Parallel Computing Systems" (SC'15) is a relevant methodology citation for E/H but was not re-verified here.
