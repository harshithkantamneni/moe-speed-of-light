# Prior art and open gaps, as of 27 Sep 2026: (A) numerical divergence in hybrid CPU/GPU inference; (B) next-token expert predictability and foresight caching for MoE offloading, including speculative decoding

Scope: literature and community evidence through late September 2026. Bottom line up front:
- **(A)** The parent topic (LLM inference nondeterminism, differences between backends and between GPUs, mismatch between the RL trainer and the inference engine) is **crowded**. No paper I found measures **divergence caused specifically by hybrid CPU/GPU offload or by different quantized kernels on the two devices**, so that niche is **open**. It is narrow, though. The researcher's own numbers (NLL within 0.1%) invite the reviewer response "so it doesn't matter."
- **(B)** Most of this area is **crowded to taken**. Next-layer prediction is saturated. **Predicting routing for the next token across decode steps was published by SeqMoE on 11 Sep 2026**, on the same models the researcher has traces for. **The Belady gap for miss counts on Qwen3-30B-A3B was published on 4 Aug 2026** (about 45–50% gap, with a decomposition into bypass and victim ranking). Speculative decoding combined with offloading has at least 6 papers. What remains is narrower: bounds that account for time and bandwidth rather than only miss counts, eviction that uses bounded lookahead from a draft or MTP head, and MTP under hybrid offload.

---

## (A) Is divergence caused by hybrid CPU/GPU offload, or by quantized-kernel mismatch between devices, already studied? How crowded is the surrounding area?

### Takeaway
Nondeterminism research has boomed since mid-2025 and now covers batch invariance, GPU count and type, backend choice, differences between GPU architectures, and CPU emulation of tensor cores. Every paper I found either explicitly excludes CPU, hybrid offload and quantized kernels, or holds them constant. The niche of divergence caused by CPU/GPU offload with quantized kernels is unclaimed. The llama.cpp community treats it as a known, benign effect ("near-tie flips, same as changing -ngl").

### Cited Findings
**Core nondeterminism work (crowded parent area)**
- Yuan et al., "Give Me FP32 or Give Me Death? Challenges and Solutions for Reproducible Reasoning", arXiv 2506.09501, v1 June 2025. With BF16 greedy decoding, DeepSeek-R1-Distill-Qwen-7B shows "up to 9% variation in accuracy and 9,000 tokens difference in response length" depending on GPU count, GPU type and batch size. The paper proposes LayerCast (16-bit weights, FP32 compute). It does not study CPU or quantized kernels. — [arXiv 2506.09501v2](https://arxiv.org/abs/2506.09501v2); [v1 HTML](https://arxiv.org/html/2506.09501v1)
  - v2 appears to be retitled "Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference". — [arXiv HTML v2 title in search](https://arxiv.org/html/2506.09501v2)
- Horace He / Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference", **10 Sep 2025**. Identifies the root cause as a failure of **batch invariance**, not "concurrency + floating point". With Qwen3-235B-A22B-Instruct-2507, 1,000 requests at T=0 produced **80 unique completions**, with the first divergence at token 103. Batch-invariant kernels made all 1,000 identical. The cost is 26 s → 55 s unoptimized, or 42 s with an improved attention kernel, on Qwen3-8B. In RL, truly on-policy training kept KL at 0. The post does not discuss CPU, quantization or cross-hardware effects. — [Thinking Machines blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- Cooper, Jeong, Jeon, Young, Kim (Georgia Tech), "Accelerating the Mitigation of LLM Inference Nondeterminism Across GPU Architectures", arXiv 2609.25624, **22 Sep 2026**. On A100, L40S and H100:
  - Unmitigated BF16: 31–100% of problems diverge across GPUs (median about 85%).
  - CastFloat32: under 1% divergence, with residuals up to 0.51%.
  - Their pinned FMA-only GEMM kernels: 0% divergence on linear layers and 0.08% end to end, 1.17–3.1× faster than CastFloat32.
  - It **explicitly does NOT cover CPU inference, hybrid CPU/GPU execution, or quantized kernels.**
  - Related work cited: LLM-42 (Gond et al., 2026, determinism on the same GPU) and Hawkeye (Badash et al., 2026, CPU emulation of tensor-core arithmetic, about 10× slower).
  - — [arXiv 2609.25624](https://arxiv.org/html/2609.25624)
- "Hawkeye: Reproducing GPU-Level Non-Determinism", arXiv 2603.20421 (March 2026). — [arXiv 2603.20421](https://arxiv.org/abs/2603.20421)
- "Deterministic Inference across Tensor Parallel Sizes That Eliminates Training–Inference Mismatch", arXiv 2511.17826 (Nov 2025), title only. — [arXiv 2511.17826](https://arxiv.org/html/2511.17826)

**Divergence across backends and its effect on evals**
- Pape, Evertz, Schönherr, "The Silent Hyperparameter: Quantifying the Impact of Inference Backends on LLM Reproducibility", arXiv 2605.19537, **19 May 2026**.
  - Five engines (vLLM, SGLang, llama.cpp, and others), with weights, decoding settings and hardware held fixed.
  - Backend choice alone shifts benchmark scores by **up to 16.6 percentage points** and produces high disagreement between outputs.
  - Of 35,000 ML papers analyzed, the inference stack is rarely reported.
  - Causes named: prefix caching, CUDA graphs, custom kernels, and defaults in logit processing.
  - — [arXiv 2605.19537](https://arxiv.org/abs/2605.19537)
- Masoudian, Elchafei, Swain, Schedl (JKU Linz), "What We Observe as LLM Behavior Can Be a Side-effect of Inference Backend", arXiv 2608.04714 (August 2026).
  - Compares HF variants, vLLM 0.8.5 and Ollama/llama.cpp on one RTX 3090 in FP16, using 1B–1.5B models.
  - Up to 48% of items flip on StereoSet and 40% on TruthfulQA for Llama-3.2-1B. Cohen's κ for Ollama vs the baseline is 0.45.
  - It **explicitly did not study CPU vs GPU, hybrid offload, quantized kernels or activation quantization**. The authors say "quantization format and compute precision are themselves likely sources of cross-backend disagreement."
  - — [arXiv 2608.04714](https://arxiv.org/html/2608.04714)

**Community evidence specific to hybrid offload (llama.cpp)**
- llama.cpp RFC #24528, "MoE expert cache, VRAM caching of hot CPU-resident experts with hybrid hit/miss execution", **12 Jun 2026**. It states that "decode-path perplexity [is] statistically identical to pure CPU" and that GPU rounding can flip near-tie tokens under greedy decoding, "matching any `-ngl` configuration change". In other words, the community acknowledges the effect and treats it as benign. Nobody has quantified it. — [llama.cpp Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)
- FreeToken (Yang, Fan, Pan et al.; UC Berkeley/MIT/UT Austin), arXiv 2608.16157, **August 2026**. It splits each layer's expert misses between GPU cache-fill and CPU in-place execution with q* ≈ m·(B_P/B_H), "re-evaluated fresh at every layer of every step". A secondary review says both paths "execute the same precision — no precision loss". — [FreeToken technical review blog, 23 Aug 2026](https://www.zhongzhuzhou.org/blog/2026-08-23-freetoken-technical-review-en/) (secondary source; I did not read the primary paper).
- KTransformers (SOSP 2025) is the canonical hybrid CPU/GPU MoE system. I found no numerical-divergence analysis in the sources I saw. — [KTransformers SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)

**Heterogeneous-hardware inconsistency (background)**
- An aggregated research direction lists Schlögl et al. ("Causes and Effects of Unanticipated Numerical Deviations in Neural Network Inference Frameworks"), MMA-Sim (a bit-accurate model of tensor and matrix cores), HSPI (identifying hardware and software platforms from numerical signatures), DiFR and VeriLLM (verifying inference despite nondeterminism), and integer networks for identical CPU/GPU results. **None of these covers hybrid offload or quantized LLM kernels.** — [Lacuna direction page](https://lacuna.tiptreesystems.com/direction/numerical-inconsistency-across-heterogeneous-machine-learning-hardware/txn_5387a6ac5fee413a94580465dd067e65)

### Inferences
- **Status of (A): parent area crowded; the offload-specific niche is open but thin.** No paper found (through 27 Sep 2026) isolates any of these three sources:
  - the mismatch in activation quantization between CPU and GPU (e.g., q8_1 MMQ on CUDA vs q8_0/q8_K vec_dot on CPU);
  - differences in accumulation order between CPU and GPU MoE kernels;
  - divergence that depends on placement or cache state.
- **The most novel framing is "placement invariance", the counterpart of Thinking Machines' batch invariance.**
  - Dynamic hybrid systems decide from runtime state whether each expert runs on CPU or GPU. Examples: the llama.cpp hit/miss expert cache (RFC #24528), and FreeToken's q* split, which is re-evaluated from measured bandwidth at every layer and step.
  - If the CPU and GPU kernels are not bitwise-equal, output then depends on cache history and on timing. That is **run-to-run nondeterminism on the same machine at batch size 1**. It is a crisp, falsifiable claim that no one has published.
  - It also directly contests the "same precision, no precision loss" claim in the FreeToken review: the same weight format does not imply the same numerics once activations are quantized or accumulated differently.
- **Link to (B), and to the researcher's routing traces.** R3 and PR² (see the next question) show that routing flips of even one slot matter for MoE RL. If offload numerics flip the router's top-k choice in later layers, then offload changes routing, which in turn changes cache behavior. Measuring this routing-flip cascade is a small, novel result that only someone with exact routing traces can produce cheaply.
- **The weakness is real.** A 1.6–4.5% greedy top-1 disagreement per teacher-forced step with NLL within 0.1% looks like "statistically identical", which is the RFC's own claim. A paper has to show consequences:
  - compounding in free-running generation (time to first divergence, how far trajectories drift apart);
  - variance in benchmark scores comparable to the 16.6 pp between backends, or the 2–6 pp run-to-run variance on SWE-bench;
  - or placement nondeterminism, which breaks reproducibility even on one machine.
- Without one of those, reviewers will read (A) as "one more source of nondeterminism, of the same kind as -ngl or batch size". Blunt estimate: as a standalone paper it is **workshop-level to mid-tier** unless the placement-invariance angle or measured downstream consequences carry it.

### Gaps
- I could not confirm whether any 2026 paper benchmarks llama.cpp CPU vs CUDA outputs for the same GGUF (KL or flip rates). Search found only guides and performance discussions. I found no llama.cpp GitHub issue that specifically documents q8_1 vs q8_0/q8_K divergence. Absence of evidence rather than proof, since GitHub issue search is weak.
- I did not read the primary FreeToken paper (arXiv 2608.16157). The "same precision" claim comes from a third-party blog.
- Venue and acceptance of "Give Me FP32" v2 (possibly NeurIPS 2025) are not verified.

---

## (A, continued) Does it matter? Downstream accuracy, eval reproducibility, agents and long generations, RL rollouts

### Takeaway
Yes, in general: numerical nondeterminism measurably moves eval scores, agent outcomes and RL stability. None of that evidence comes from hybrid CPU/GPU offload, and RL rollouts are essentially never run with CPU expert offload. So the RL angle is **not a good fit for (A)**. The strongest "it matters" arguments for hybrid offload are consumer and edge evaluation reproducibility, and agentic divergence.

### Cited Findings
- **Agents.** Bjarnason, Silva, Monperrus (KTH), "On Randomness in Agentic Evals", arXiv 2602.07150 (ICLR 2026 workshop, March 2026).
  - Scale: 60,000 SWE-Bench-Verified trajectories, 25 billion tokens.
  - Single-run pass@1 fluctuates by **2.2–6.0 pp**, with standard deviation above 1.5 pp **even at temperature 0**.
  - "Trajectories typically diverge very early, within the first tokens" (median point 0.1–0.5% into the trajectory).
  - The gap between pass@k and pass^k is 24.9 pp.
  - — [arXiv 2602.07150](https://www.arxiv.org/pdf/2602.07150)
- **Reasoning evals.** Up to 9% accuracy swing and 9,000-token differences in length from GPU count, type and batch size. — [arXiv 2506.09501v2](https://arxiv.org/abs/2506.09501v2)
- **Benchmarks across backends.** Up to 16.6 pp. — [arXiv 2605.19537](https://arxiv.org/abs/2605.19537)
- **RL mismatch between trainer and inference engine (crowded):**
  - "Defeating the Training-Inference Mismatch via FP16", arXiv 2510.26788 (Oct 2025). — [arXiv 2510.26788](https://arxiv.org/abs/2510.26788)
  - "Diagnosing Training Inference Mismatch in LLM Reinforcement Learning", arXiv 2605.14220 (May 2026). — [arXiv 2605.14220](https://arxiv.org/html/2605.14220v1)
  - "Beyond Precision: Training-Inference Mismatch is an Optimization Problem…", arXiv 2602.01826 (Feb 2026). — [alphaXiv 2602.01826](https://www.alphaxiv.org/abs/2602.01826)
- **MoE-specific routing mismatch in RL:**
  - R3, "Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers", Ma et al., arXiv 2510.11370, **13 Oct 2025**. Routing differs between SGLang/vLLM and Megatron and can cause "catastrophic RL training collapse". R3 records the inference engine's routing and replays it in training. — [arXiv 2510.11370](https://arxiv.org/abs/2510.11370)
  - PR², "Predictive Routing Replay", arXiv 2606.00395 (the ID implies June 2026; the fetched page said "2025 preprint", so the date conflicts). Baseline GRPO has only 76.9% of tokens with an identical expert set, and 19.6% differ by one slot. PR² gains +12 to +19 AIME24 points over routing replay on Qwen3-30B-A3B-Base. — [arXiv 2606.00395](https://arxiv.org/html/2606.00395v1)
  - Megatron-LM has a feature request for Router Replay. — [NVIDIA/Megatron-LM #4168](https://github.com/NVIDIA/Megatron-LM/issues/4168)

### Inferences
- Routing flips of one slot matter in MoE training (R3, PR²). This makes a measurement of **how often offload numerics flip top-k routing** meaningful and citable. It is still a stretch to claim RL relevance for CPU offload, because RL rollouts run on GPU clusters.
- The most defensible "does it matter" story for (A) is **local and consumer evaluation**. Many open-model evaluations and agent harnesses run on llama.cpp or Ollama with partial offload. If VRAM size changes greedy outputs, then eval results from local runs are not comparable across machines. The agent paper shows that early divergence compounds into different strategies.
- A clean experiment: for the same prompts, run free-running (not teacher-forced) generation at all-GPU vs several -ngl or -ot splits. Measure:
  - the distribution of the first token where outputs diverge;
  - pass-rate variance on a small agentic or maths benchmark;
  - how that variance compares with seed or temperature variance.

  If offload variance is at most the T=0 run-to-run variance already reported (1.5 pp standard deviation on SWE-bench), the honest conclusion is "doesn't matter beyond existing noise". That is still publishable as a measurement note, but it is weak.

### Gaps
- No source found measuring how hybrid offload affects any downstream benchmark or agent task. This is the gap the researcher would fill, and it is also the risk.

---

## (B) Expert prediction, prefetch and caching: how crowded, and is next-token prediction or closing the Belady gap still open?

### Takeaway
**Crowded, and the two gaps the researcher was eyeing were each published in Aug–Sep 2026.** SeqMoE (11 Sep 2026) does learned prediction of **next-step (next-token) routing across the full layer stack**, using a Mamba2 model on Qwen3-30B-A3B and GPT-OSS-120B. Zhang (4 Aug 2026) publishes the **Belady vs best-causal miss gap on Qwen3-30B-A3B (about 45–50%)**, decomposes it into bypass and victim ranking, and shows that a learned next-use-distance predictor fails. What remains: bounds that account for time and bandwidth (the researcher's framing), bounded-lookahead MIN using draft or MTP foresight, and rigorous head-to-head evaluation.

### Cited Findings
**Next-layer prediction by gate lookahead or learned predictors (saturated)**
- An aggregated table dates the main systems:
  - Pre-gated MoE (ISCA 2024), AdapMoE (ICCAD 2024), ProMoE (2024, trained predictor across layers).
  - HOBBIT (2024, falls back to lower-precision experts), ExpertFlow (2024 arXiv; DAC 2026), SiDA-MoE (MLSys 2024).
  - Fate (2025, reuses gate inputs from adjacent layers), fMoE (2025, per-request expert maps retrieved by semantic similarity), MoE-Infinity (activation-matrix trace matching), MoE-Beyond (2025), Mixtral-offloading (2023).
  - The table's stated gap: "no clearly released implementation combines a multi-token horizon with a learned cross-layer or cross-token correlation signal". SeqMoE (below) now appears to do exactly this.
  - — [verdict-sweep MoE offloading survey](https://github.com/animaresearch/verdict-sweep/blob/main/examples/moe-offloading-survey.md)
- Fate: "Accurate Expert Predictions in MoE Inference via Cross-Layer Gate", arXiv 2502.12224 (Feb 2025). — [arXiv 2502.12224](https://arxiv.org/html/2502.12224v1)
- FineMoE / fMoE appears in EuroSys 2026. — [FineMoE EuroSys'26 PDF](https://intellisys.haow.us/assets/pdf/Hanfei_FineMoE_EuroSys26.pdf)
- MoE-Beyond (Gavhane et al.), arXiv 2508.17137, **23 Aug 2025**. A lightweight transformer predictor over activation traces. 97.5% accuracy, F1 86.6%. Cache hit rate rises from 17% to 72% when 10% of experts fit on the GPU (DeepSeek-V2-Lite). — [arXiv 2508.17137](https://arxiv.org/abs/2508.17137)
- "Speculating Experts Accelerates Inference for MoE" (Madan, Singhania, Bhatele, Goldstein, Panda; UMD/Together), arXiv 2603.19289, **9 Mar 2026**.
  - Predicts next-layer experts from a "quasi-hidden state".
  - Up to 14% lower TPOT on an A6000, where CPU-GPU transfer was 84–88% of time.
  - Qwen3-30B-A3B accuracy is sensitive (AIME24, GSM8k), so the authors add neural estimators for problem layers.
  - — [arXiv 2603.19289](https://arxiv.org/html/2603.19289)
- SpecPrefetch (Wang et al.), arXiv 2607.24787 (Pith gives 24 Jun 2026; the arXiv ID implies July 2026). A low-rank adapter predicts next-layer experts. 6.48M parameters vs 207M for baselines. Up to 20% throughput gain on Snapdragon 8 Elite. — [Pith 2607.24787](https://pith.science/paper/2607.24787)
- Edge0, "The Other Half of the Memory Wall: Serving 35B MoEs from SSD with Trained Routing Prediction", arXiv 2609.18063, **16 Sep 2026**. Trained "prerouter" heads predict next-layer routing. 20.4 tok/s for 35B on a 24 GB Mac mini using 2.9 GiB active memory. Accuracy on five benchmarks is 3.9 points below FP16. — [GitHub aggregator issue](https://github.com/jjakimoto/research-issues/issues/1623) (secondary source)

**Next-token and cross-step routing prediction (the researcher's intended gap is now taken)**
- **SeqMoE** (Wang, Wang, Gong, Tang, Lou, Wang, Wang, Zhou; USTC-affiliated authors per the author list), arXiv 2609.12978, **11 Sep 2026**.
  - Recasts expert prediction as **sequence modeling across generation steps**: it encodes the activation state of the whole layer stack per step and trains a **Mamba2** model to predict the **next step's activations, with multi-step lookahead**.
  - Qwen3-30B-A3B: **top-11 recall above 90%** (k=8). GPT-OSS-120B: **top-7 recall above 90%** (k=4).
  - At 45% expert residency: **96.97% average hit rate vs 88.50% for the strongest baseline**, and **80.22% of full-load performance** vs 51.50% for FreeToken.
  - The runtime is compatible with **CUDA graphs**.
  - Models: Qwen3-30B-A3B-FP8, Qwen3.6-35B-A3B, GPT-OSS-120B-MXFP4, DeepSeek-V4-Flash. Baselines: llama.cpp, MoE-Infinity, KTransformers, FreeToken.
  - It cites other cross-step methods: ST-MoE (Zhao 2026), Patterns-MoE (Yu 2026b), Taming-MoE (Yu 2026a), MoE-APEX (Tang 2026).
  - The fetched content does not mention speculative decoding.
  - — [arXiv 2609.12978](https://arxiv.org/html/2609.12978)
- **ST-MoE** (Zhao, Bunescu, Louri, Karanth, Wang), arXiv 2606.15453, **13 Jun 2026**. Explicitly exploits **temporal (cross-token)** correlation ("consecutive decoding tokens often trigger overlapping expert selections") together with cross-layer correlation through correlation and history tables. About 85% prediction accuracy. It is a hardware accelerator in TSMC 40 nm, with no Belady comparison. — [arXiv 2606.15453](https://arxiv.org/html/2606.15453)
- FreeToken: "decode-time routing exhibits strong temporal locality". A shared LRU gives 2–5× lower miss rates than static splits. — [FreeToken review](https://www.zhongzhuzhou.org/blog/2026-08-23-freetoken-technical-review-en/)

**Cache policy vs Belady (the researcher's speed-of-light result overlaps heavily)**
- **Yu Zhang, "Reproducible Evaluation of MoE Expert Caching: Replay Semantics, Workload Contamination, and Operating Regimes"**, arXiv 2608.07911, **4 Aug 2026**.
  - Models: Granite-3.1-3B-A800M, OLMoE-1B-7B-0125, Qwen3-30B-A3B (4-bit).
  - Qwen3-30B-A3B at 40% residency, B=8: best causal policy (LFRU) misses 18.01% vs **Belady 9.93%**, a **recoverable gap of 44.85%**, stable at 44.2–45.9% across 13 workload mixes. At B=2 the gap is **50.42%**.
  - Decomposition of Belady's advantage: **bypass admission accounts for 15.7% (B=8) or 3.4% (B=2); ranking victims by future use accounts for 84.3–96.6%.**
  - A trained next-use-distance predictor **worsens misses by 11.4%**. It picks the optimal victim 3.39% of the time vs 22.1% for LFRU.
  - Replaying accesses one at a time instead of per event inflates recency-based policies by 27–29% and can invert rankings. Normalized miss rates carry across models only when matched on r̄, the ratio of the per-step expert union to per-layer capacity.
  - The paper explicitly separates its result from prefetch prediction: "Expert prefetching predicts which experts a future step will route to; our predictor estimates how long a cached block will remain unused."
  - — [arXiv 2608.07911](https://arxiv.org/html/2608.07911)
- Si, Li, Lin, Zhang (Waterloo), "Who Should Own the Expert Cache? Kernel-Managed Tiering for Trillion-Parameter MoE Inference", arXiv 2608.12103 (August 2026).
  - With 896 experts per layer, kernel LRU "captures approximately 70% of Belady's optimum (44 of 62%)".
  - The production router-lookahead predictor reaches 64.7% recall but changes median time by only 0.3%. Perfect one-layer prediction gains only 5.0%.
  - — [arXiv 2608.12103](https://arxiv.org/html/2608.12103)
- SpecMD (Hoang, Jaiswal, Samragh, Cho; Apple), arXiv 2602.03921, **3 Feb 2026**, **ICML 2026 poster**. Finds "MoE expert access is not consistent with temporal locality assumptions (e.g LRU, LFU)". Its Least-Stale policy cuts collision misses by up to 85× over LRU, with hit rates above 88% on OLMoE. No Belady comparison. — [arXiv 2602.03921](https://arxiv.org/abs/2602.03921); [ICML 2026 poster](https://icml.cc/virtual/2026/poster/62315)
- FlashMoE, arXiv 2601.17063 (Jan 2026). ML-based cache replacement for MoE on SSD-based edge devices, title-level only. — [arXiv 2601.17063](https://arxiv.org/html/2601.17063v1)
- The llama.cpp RFC #24528 reports about 70–80% hit rates, with the top 10% of experts taking about 80% of accesses. Gains are +10% to +57% across 13 models. — [llama.cpp #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)

### Inferences
- **Next-token routing prediction is taken** (SeqMoE, 11 Sep 2026, same models, strong numbers, CUDA graphs, llama.cpp/KTransformers/FreeToken baselines). Pursuing it as the main contribution would read as incremental.
- **The Belady miss-gap result is taken in its count-based form** (Zhang, 4 Aug 2026: about 45–50% on Qwen3-30B-A3B). The researcher's 46–59% at a 25% budget is numerically consistent with it and therefore **not novel on its own**. Zhang's decomposition ("victim ranking is 84–97% of the gap; bypass is small") also preempts a bypass-centric story.
- **What still distinguishes the researcher's asset:** a speed-of-light bound **in the time domain**, i.e. stall time under PCIe bandwidth and overlap with compute, not miss counts. Belady MIN bounds demand misses without prefetching. Systems like SeqMoE report prefetch "hit rates" (about 97%) that a count-based Belady bound cannot adjudicate. A bound on minimum exposed transfer time for a trace, a budget and a link bandwidth, which also allows prefetching, would let the field score SeqMoE, FreeToken and the others against an optimum. Si et al.'s finding (perfect one-layer prediction gains only 5%) suggests the time domain is where the real answer lives.
- **Bounded-lookahead MIN.** Zhang shows victim ranking by future use drives the gap, and that learned next-use predictors fail. A draft model or MTP head supplies **exact future tokens k steps ahead**, and therefore near-exact future routing for k steps. How much of the Belady gap k-token foresight recovers, as a function of k and acceptance rate, is not answered in anything I found. SP-MoE and MoE-SpeQ use drafts for **prefetch** only, not for **eviction or bypass**.

### Gaps
- I could not read arXiv 2511.10676, which the search surfaced as a candidate on cross-token expert patterns. The proxy rate-limited the fetch (HTTP 429), so its content and date are unverified.
- Taming-MoE, Patterns-MoE (Yu 2026a/b) and MoE-APEX (Tang 2026) are cited by SeqMoE but were not retrieved. Their exact scope (cross-step prediction) is unverified.
- Whether any paper computes a time-domain or bandwidth-aware optimal bound for MoE offloading was not confirmed either way. Si et al. and Zhang are count-based.

---

## (B, continued) Speculative decoding (draft models, MTP) under expert offloading

### Takeaway
**Crowded for "use the draft to prefetch experts".** At least SpecMoEOff (Aug 2025), SP-MoE (Oct 2025), MoE-SpeQ (Nov 2025), DraftExpert (Jul 2026) and EcoSpec (Jul 2026) exist, plus MoE-Spec and MoESD. The **expert-union blowup during verification** is well documented. **MTP heads under hybrid CPU/GPU offload** and **drafts as eviction oracles** look open. Community measurements show speculative decoding often *hurts* A3B MoEs on consumer GPUs. There is also a llama.cpp bug-level observation that expert caches silently disengage with drafters placed on the GPU.

### Cited Findings
- SpecMoEOff, "Accelerating Mixture-of-Experts Inference by Hiding Offloading Latency with Speculative Decoding", arXiv 2508.21706 (**Aug 2025**). — [arXiv 2508.21706](https://arxiv.org/abs/2508.21706)
- **SP-MoE** (Chen, Wen, Wu, Zhang, Wu; SYSU/HKU), arXiv 2510.10302, v2 **6 Nov 2025**.
  - Feeds each draft layer's attention output into the **target model's gate** to predict the experts verification will need.
  - About 88–89% top-1 accuracy on DeepSeek-Lite, Mixtral and Phi-3.5-MoE.
  - 1.07–3.5× TPOT gain over prior methods on RTX 4090 over PCIe 4.0.
  - A "cutoff layer" policy prevents cache thrashing.
  - — [arXiv 2510.10302](https://arxiv.org/html/2510.10302v2)
- **MoE-SpeQ** (Wang, Liu, Hou et al.; SJTU per the author list), arXiv 2511.14102, **18 Nov 2025**.
  - An INT4 draft predicts target experts for future tokens.
  - 90.9% "accurate" top-4 prediction (44.1% exact, 46.8% correct set in a different order).
  - An Expert Lookahead Buffer covers k tokens × L layers.
  - Up to 2.34× on A100-40G.
  - It explicitly notes that verification must "load and compute the *union* of all experts activated across all k tokens", which can degrade performance.
  - — [arXiv 2511.14102](https://arxiv.org/html/2511.14102v1)
- DraftExpert, "Expansion-Aware Self-Speculative Decoding for End-Device MoE", arXiv 2607.24434 (July 2026). Title and abstract only; the Pith page returned 403. — [arXiv PDF 2607.24434](https://arxiv.org/pdf/2607.24434)
- **EcoSpec**, "Less Experts, Faster Decoding: Cost-Aware Speculative Decoding for Mixture-of-Experts" (Xie et al.), arXiv 2607.12696, **14 Jul 2026**.
  - Names **"expert scattering"**: high-confidence draft tokens route to disjoint experts, inflating the union during verification.
  - Chooses draft trees that account for marginal expert cost.
  - Up to 1.62× on Qwen3-235B-A22B, 1.50× on GPT-OSS-120B, 1.47× on DeepSeek-V3.1.
  - Offloading is described only as "complementary". It cites MoE-Spec (McDanel et al., 2026) and MoESD (Huang et al., 2026).
  - — [arXiv 2607.12696](https://arxiv.org/html/2607.12696)
- "Utility-Driven Speculative Decoding for Mixture-of-Experts", arXiv 2506.20675 (June 2025), title only. — [arXiv 2506.20675](https://arxiv.org/html/2506.20675v1)
- **Community: speculative decoding hurts an A3B MoE on consumer GPUs.**
  - Setup: Qwen3.6-35B-A3B at Q4_K on an RTX 3090, batch 1, greedy.
  - Baseline 139.9 tok/s. ngram-cache gives 119–121 (−15%); a draft model (Qwen3.5-0.8B) gives 65–86 tok/s (−39% to −54%).
  - "100% draft acceptance cannot rescue it." The cause given is saturation of the expert union: a fresh expert union per drafted token, with a saturation threshold around 94 tokens.
  - Offload configuration unclear; a Q4 35B model may be fully GPU-resident on 24 GB.
  - — [HackMD benchmark](https://hackmd.io/@thc1006/SJly6IE6Wx)
- **Community: MTP helps an all-GPU MoE.** Qwen3.6-35B-A3B at IQ3_S on an RTX 5060 Ti with all experts on the GPU goes from 98 to 144 tok/s (1.47×) with MTP draft-2 at 69–81% acceptance, **14 May 2026**. This is not offloaded. — [njannasch blog](https://njannasch.dev/blog/mtp-speculative-decoding-qwen-3-6-5060ti/)
- **llama.cpp expert cache vs drafters.** In RFC #24528 testing, "the cache composes with an external drafter for a +37% all-time record — but only when the drafter is NOT on a GPU". Cache engagement **fails silently** when a drafter on the GPU reorders the device list. — [llama.cpp #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)

### Inferences
- "Draft predicts target experts → prefetch" is **taken** (SP-MoE, MoE-SpeQ, SpecMoEOff). "Expert union during verification costs bandwidth" is **known** (MoE-SpeQ, EcoSpec, community).
- **MTP heads (DeepSeek-V3, Qwen3-Next, Qwen3.6) under hybrid CPU/GPU expert offload** are not covered by any paper I found. Open questions:
  - Does the per-step union (1 + k tokens) push more experts onto the CPU and erase the gains?
  - Does the MTP token's routing overlap the next real token's?
  - What is the best split between CPU compute and GPU fetch when verifying k tokens? This extends FreeToken's q* to k tokens.

  All three are open and cheap to measure.
- **The most defensible (B) contribution given the assets:** a **time-domain speed-of-light model** for offloaded MoE decode that covers three strategies:
  - demand caching, bounded by Belady;
  - prefetch, bounded by an oracle predictor combined with bandwidth;
  - speculative verification with union cost.

  It would evaluate how far SeqMoE-style next-step prediction, SP-MoE-style draft prefetch and FreeToken-style CPU/GPU splitting each are from the bound, and give a k-lookahead MIN curve showing how much of the gap draft or MTP foresight can close. This is a performance-model paper rather than a systems paper.

### Gaps
- No source found on how MTP heads affect expert locality or cache hit rate under offload.
- No source found that uses speculative tokens for **eviction or bypass** decisions (as opposed to prefetch).
- I found no paper that confirms, analyzes or fixes the llama.cpp drafter/cache device-ordering interaction; the only evidence is the community thread.

---

## Feasibility (27 Oct 2026 and Dec 2026), venue, and fit with SPCL

### Takeaway
MLSys 2027 papers are due **30 Oct 2026, 12:00 PM PDT**, 3 days after the 27 Oct target. (A) is the cheaper and less contested bet for 27 Oct, but it risks being judged "minor". (B) has more intellectual weight, but it now has to be positioned directly against SeqMoE (11 Sep 2026) and Zhang (4 Aug 2026). In its original form ("predict next-token routing" or "the Belady gap") it is effectively scooped.

### Cited Findings
- MLSys 2027 dates:
  - Submission opens 10 Oct 2026; **deadline 30 Oct 2026 12:00 PM PDT**.
  - Reviews 18 Jan 2027; notification 28 Feb 2027.
  - Conference 22–24 Jun 2027, Indio, CA.
  - — [MLSys 2027 Dates](https://mlsys.org/Conferences/2027/Dates)
- The competing work in (B) is recent and fast-moving. SeqMoE (11 Sep 2026), Edge0 (16 Sep 2026), FreeToken (Aug 2026), Zhang (4 Aug 2026), Si et al. (Aug 2026) and EcoSpec (14 Jul 2026) all appeared within about 10 weeks. — [SeqMoE](https://arxiv.org/html/2609.12978); [Zhang](https://arxiv.org/html/2608.07911); [Si et al.](https://arxiv.org/html/2608.12103); [EcoSpec](https://arxiv.org/html/2607.12696)
- The nondeterminism literature also keeps moving: Cooper et al. (22 Sep 2026), Masoudian et al. (Aug 2026), Pape et al. (May 2026). None touches hybrid offload. — [2609.25624](https://arxiv.org/html/2609.25624); [2608.04714](https://arxiv.org/html/2608.04714); [2605.19537](https://arxiv.org/abs/2605.19537)

### Inferences
**(A) Hybrid-offload numerics (OPEN niche, crowded parent)**
- **By 27 Oct:** feasible as a sharp measurement paper, with the infrastructure already in place. Required pieces:
  1. Flip rates per step and routing-flip cascades for all-GPU vs several splits, across 3 models and 2–3 GPUs.
  2. A root-cause ablation (activation quantization vs accumulation order vs attention path).
  3. A placement-invariance demonstration: the same machine giving different outputs under a dynamic expert cache.
  4. One downstream consequence, such as agent or maths pass-rate variance vs a seed baseline.

  Realistic targets: an MLSys 2027 submission (weak accept at best as a standalone), or better a workshop, or as one section of a larger paper.
- **By Dec 2026:** add a mitigation, "placement-invariant" CPU/GPU MoE kernels that are bitwise-matched, or quantizing activations identically on both devices, with a cost measurement. That turns it into a Thinking-Machines-style "measure → root-cause → fix" story. A fix plus a cost gives much stronger standing.
- Risk: the researcher's own NLL result (within 0.1%) supports the "benign" reading. Unless free-running divergence or placement nondeterminism is striking, reviewers will shrug.

**(B) Prediction and foresight caching (CROWDED; core claims taken in Aug–Sep 2026)**
- **By 27 Oct:** only feasible as a repositioned **performance-bound and evaluation paper**, containing:
  - the time-domain speed-of-light bound;
  - where SeqMoE-class next-step predictors, draft prefetch (SP-MoE/MoE-SpeQ) and CPU/GPU splitting (FreeToken) sit relative to the bound;
  - a k-lookahead MIN curve.

  It must cite and differentiate from Zhang 2608.07911 on count-based Belady and SeqMoE 2609.12978 on next-step prediction. Reimplementing SeqMoE in 4 weeks is unrealistic; evaluating an oracle version of its predictor on the researcher's traces is realistic.
- **By Dec 2026:** a system that uses MTP or draft foresight for **eviction/bypass plus prefetch** under hybrid offload in llama.cpp, including a fix for the drafter/cache device-ordering issue. Feasible for one person, but it competes against well-staffed groups (SJTU, USTC, UMD/Together, Apple).
- A pure "next-token expert predictor" paper is **not recommended**: SeqMoE already has better models, larger baselines and CUDA-graph support.

**SPCL fit (my judgement, not sourced)**
- The hedged-bound framing in (B), with an optimal oracle, a time-domain model and a clear gap, matches a measurement- and modeling-heavy HPC style better than another predictor system does.
- The reproducibility framing in (A) (placement invariance, root cause, bitwise fix) also fits a rigorous-benchmarking taste.
- The strongest combined story may be to merge the two: **offload placement changes both numerics and routing; an optimal time-domain bound says how much offload can ever buy; and deterministic, placement-invariant execution says what it costs to keep outputs identical.**

### Gaps
- I did not retrieve sources on SPCL/Hoefler's current (2026) research priorities or recent SPCL MoE/inference papers; the search returned nothing SPCL-specific. The fit assessment above rests on general knowledge, not citations.
- I did not verify EuroSys 2027, ASPLOS 2027 or ATC 2027 deadlines, or whether any venue has a deadline on 27 Oct 2026 specifically.
- I found no source quantifying how much of the Belady or time-domain gap SeqMoE or SP-MoE actually close. Neither compares against an optimal bound in the material fetched.
