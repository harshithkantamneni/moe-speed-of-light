# MoE offloading and CPU-GPU hybrid MoE inference, June 2025 to October 2026: related work that "Where the Seconds Go" must cite or beat

Method note:
- Two arXiv API sweeps were run on 6 Oct 2026, with 46 keyword queries sorted by submission date. They returned 315 in-window hits, which were triaged by title.
- 85 abstracts were pulled. 46 PDFs were converted to text and grepped for Belady, oracle, optimal, lookahead, llama.cpp, `--n-cpu-moe`, GPU names and baselines.
- llama.cpp facts were checked against `git log` of `ggml-org/llama.cpp` master (shallow history since 2025-01-01, head `43fe9c6`, 2026-10-06). The heads of all 257 PRs numbered 29620 to 30041 were also fetched.
- KTransformers release dates come from its git tags.
- Earlier project notes cover up to 28 Sep 2026 in more depth. Items taken from them without re-reading are marked "(prior notes)".
- Tags: "(abs)" means the claim was checked against the arXiv abstract on 6 Oct 2026; "(text)" means it was checked against the PDF text; "(git)" means it was checked against a repository.
- Labels used throughout: (a) speed limit; (b) llama.cpp expert cache; (c) in-engine oracle factorial; (d) nine-model trace study and W50 law; (e) learned admission.

## Q1. Which new systems (June 2025 to Oct 2026) do expert caching, offloading or CPU-GPU hybrid execution for batch-1 or small-batch MoE decode on consumer GPUs? Venue and date, hardware, models, speed-up, baseline

### Takeaway
About 60 relevant arXiv systems appeared in the window, roughly 4 per month, accelerating in Aug–Oct 2026.

Exact-routing systems on consumer GPUs that are close to the paper's setting, and not already cited:
- FlashMoE (Jan 2026; v2 5 Oct 2026)
- the OSDI'26 CPU-GPU hybrid (2606.10493)
- Pipelined Sharding (MLSys'26)
- ATSInfer (Jul 2026)
- WiSP (Jun/Aug 2026)
- SAEM (Aug 2026)
- SSD-LLaMA (Sep 2026)
- Mira (29 Sep 2026)
- OLED-MoE (EuroSys'27, dLLM)
- RapidMoE (EuroSys'27, lossy)
- HiNa-MoE (PACT'26, CPU kernels)

A concurrent upstream llama.cpp PR (#29887, 3 Oct 2026) adds a GPU LRU expert cache.

Most other entries are one of:
- lossy (substitution, mixed precision, router fine-tuning);
- speculative-decoding-coupled (outside a one-token-per-pass bound);
- edge, NPU or SSD-tier systems.

Almost none use an equal-memory llama.cpp `--n-cpu-moe` baseline.

### Cited Findings

#### A. CPU-executes-misses / CPU-GPU hybrid, exact routing (closest to (b))
- **OSDI'26, "Achieving Cloud-Grade SLOs for Local MoE Inference through CPU-GPU Hybrid Design"** (Wang, Hou, Ji, Qu, Zhang; arXiv 2606.10493, v1 9 Jun 2026). (abs) — [arXiv 2606.10493](https://arxiv.org/abs/2606.10493); [USENIX PDF](https://www.usenix.org/system/files/osdi26-wang-wenxin.pdf)
  - Platform: dual-socket commodity server CPUs with 1–2 RTX 5090.
  - Results: 28 tok/s on INT4 DeepSeek-V3/R1 ("1.25× improvement over KTransformers") and 21.5 tok/s on intact FP8 V3. Stream-loading prefill reaches 1,200 tok/s, or 1,800 tok/s on two 5090s.
  - Baselines: KTransformers, llama.cpp, ik_llama.cpp. (text)
  - Its motivation is framed as a fraction of a hardware limit: KTransformers' decode "is only about 50% of the nominal aggregate DDR5 bandwidth". (text)
  - It reports no bound or oracle.
- **ATSInfer, "Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference on Consumer Devices"** (2607.10183, v1 11 Jul 2026). (text) — [arXiv 2607.10183](https://arxiv.org/abs/2607.10183)
  - Mechanism: llama.cpp extended with tensor-granularity placement plus load-aware dynamic transfer.
  - Platforms: laptop with RTX 3060 6 GB, i7-11800H and 32 GB DDR4; desktop with RTX 4090, i7-11700 and 64 GB DDR4, PCIe 4.0. Batch 1.
  - Models include GPT-OSS-120B (MXFP4), Qwen3-30B-A3B and Qwen3-Next-80B.
  - Baselines: llama.cpp ("primary baseline"), vLLM, KTransformers v0.5.2. KTransformers "does not support the MXFP4 format".
  - Claims "decode throughput by up to 3.29×" over llama.cpp (laptop).
  - The llama.cpp config is only "the same offloading policy that fills GPU memory as much as possible"; no `--n-cpu-moe` string appears in the text.
  - It notes that KTransformers "consistently outperforms llama.cpp because expert-granularity offloading…".
- **Pipelined Sharding, "Efficient, VRAM-Constrained xLM Inference on Clients"** (NVIDIA; 2604.26334, v1 29 Apr 2026; MLSys'26 Industry Track). (abs, text) — [arXiv 2604.26334](https://arxiv.org/abs/2604.26334)
  - Mechanism: sub-layer sharding, CPU offload, pipelined copy-compute, built on llama.cpp.
  - Claims "TTFT improves by up to 6.7× and TPS by up to 30× for LLMs" against an "aggressive" llama.cpp baseline that searches the maximum number of layers fitting the VRAM budget.
  - Includes a 5090 client (cli3).
  - It has a planner "oracle comparison across 105 configurations" (selects the best of 3 strategies 105/105). This is a config oracle, not a routing oracle.
- **SAEM** (Zhang, Gao, Mitra; 2608.21614, v1 21 Aug 2026; extended DAC'26). (text) — [arXiv 2608.21614](https://arxiv.org/abs/2608.21614)
  - Mechanism: stage-aware caching plus in-situ CPU execution of infrequent experts.
  - Hardware: one A100 80 GB with a Xeon Gold 6326 (single socket used), PCIe 4.0.
  - Baselines: MoE-OnDemand, Mixtral-Offloading, Fiddler, DAOP.
  - Claims 1.33× average throughput, or 1.54× with matched calibration.
  - Contains an in-engine "Prediction-Oracle Upper-Bound Analysis" (see Q2).
- **OLED-MoE** (2609.33385, v1 27 Sep 2026; EuroSys'27). (abs, text) — [arXiv 2609.33385](https://arxiv.org/abs/2609.33385)
  - Scope: diffusion LLMs. Inter-iteration expert retention plus CPU-GPU cooperative execution of misses.
  - Hardware: RTX 5090 with a Xeon Platinum 8470Q.
  - Baselines: KTransformers, MoE-Infinity, Fiddler, DALI.
  - Claims TPOT 1.23–7.93× better, and at 40% expert memory only "23% higher TPOT" than full residency.
  - Not autoregressive, so it is out of scope for (a), but it is a CPU-miss cache design to cite.
- **RapidMoE** (2610.01265, v1 1 Oct 2026; EuroSys'27). (abs, text) — [arXiv 2610.01265](https://arxiv.org/abs/2610.01265)
  - Mechanism: built on KTransformers; "residual-split" bit-level offload with runtime "importance arbitration" (an accuracy-latency trade-off, so lossy).
  - Models: DeepSeek-V3/R1, Qwen3-235B. Hardware: Xeon with 512 GB DDR4-3200 and an RTX 4090 or A100.
  - Claims "up to 3.5× speedup in decoding" over SOTA offloading systems.
- **HiNa-MoE** (2610.05123, v1 4 Oct 2026; PACT'26). (abs, text) — [arXiv 2610.05123](https://arxiv.org/abs/2610.05123)
  - An AMX CPU MoE operator library on A6000 + Xeon Gold 6430/6448H.
  - Claims "up to 3.37× speedup for FFN kernels and up to 2.09× end-to-end" over SOTA, with KTransformers as the main baseline.
- **Efficient CPU-GPU Collaborative Inference (Huang, Lin, Lee)**: 2512.16473, 18 Dec 2025, ASP-DAC'26; RTX 4090. **Already cited** — [arXiv 2512.16473](https://arxiv.org/abs/2512.16473)
- **DALI** (2602.03495, 3 Feb 2026; RTX 3090 + EPYC 7532): average decode speed-ups of 3.97× over llama.cpp, 2.16× over KTransformers, 1.48× over MoE-Lightning and 1.32× over HybriMoE (prior notes). **Already cited** — [arXiv 2602.03495](https://arxiv.org/abs/2602.03495)
- **FreeToken** (2608.16157, 17 Aug 2026; still v1). Shared LRU cache with a bandwidth-derived fill/CPU split; 77–83 tok/s on Qwen3.6-35B-A3B BF16 on an RTX 5090 server (prior notes). **Already cited** — [arXiv 2608.16157](https://arxiv.org/abs/2608.16157)
- **KTransformers**:
  - SOSP'25: batch 1, decode 1.25–1.76× over a custom-patched llama.cpp (prior notes). **Already cited** — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
  - In-window feature: v0.5.1 (tag dated 2026-01-22) added "CPU-GPU Expert Scheduling", covered in Q3. (git)
- **SMoE** (2508.18983, v1 26 Aug 2025, v3 3 May 2026). (abs, text) — [arXiv 2508.18983](https://arxiv.org/abs/2508.18983)
  - Lossy: substitutes low-importance experts with cached ones.
  - Hardware: A6000 + Xeon Gold 6444Y. Baselines: HybriMoE and llama.cpp.
  - Claims "48% lower decoding latency with over 60% expert cache hit rate".
- **SPICE** (2608.21240, ASP-DAC'27). (abs; hardware from prior notes) — [arXiv 2608.21240](https://arxiv.org/abs/2608.21240)
  - Lossy mode: LoRE surrogates for low-confidence misses, and exact residuals on the CPU.
  - Hardware: RTX 5090 with PCIe 4.0 x8 and 128 GB DDR5; RTX 4060; A800.
  - Claims up to 3.12× TPOT, and 2.04–2.70× over AdapMoE.

#### B. GPU-fetch cache and prefetch systems, exact routing
- **FlashMoE (ML-based cache replacement)** (Kim, Lee, Han, Yoo, Kim; 2601.17063, v1 22 Jan 2026, **v2 5 Oct 2026**). (text) — [arXiv 2601.17063](https://arxiv.org/abs/2601.17063)
  - Tier and hardware: experts on SSD. Ryzen 5 9600X, **RTX 5070 Ti 16 GB on PCIe 5.0**, dual-channel DDR5-6000, PCIe 5.0 NVMe at 7.4 GB/s.
  - Models: OLMoE-1B-7B and Qwen3-30B-A3B.
  - Baselines: Fiddler, DAOP, llama.cpp, HF Transformers.
  - Cache policy: a learned FFN eviction policy over recency and frequency features, trained on Belady-derived labels.
  - Hit rate: +21% over LRU and +51% over LFU (OLMoE); up to 28% over ARC and 21% over LeCaR.
  - Speed over LRU: +22% (OLMoE) and **+7% (Qwen3-30B-A3B)**.
  - "Total speedup of 2.5× over llama.cpp", including load and prefill.
  - It has a learned policy and a Belady comparison, but no seconds bound (see Q2).
- **Mira** (Yadav, Asgari; 2609.38090, v1 29 Sep 2026). (abs, text) — [arXiv 2609.38090](https://arxiv.org/abs/2609.38090)
  - Mechanism: per-layer predictors two layers ahead, a HOT+STAGE two-tier GPU cache driven by routing telemetry, and custom expert compression ("minimally degrading accuracy", i.e., not bit-exact).
  - Hardware: RTX 6000 Ada 48 GB, RTX A5000 24 GB with EPYC 7352, RTX 3070 8 GB.
  - Models: Mixtral-8x7B and 8x22B, DeepSeek-V2-Lite.
  - Claims: "5.71× over MoE-Infinity on a 24GB GPU"; 10.46× over DeepSpeed, 2.86× over MoE-Infinity and 1.55× over Fiddler on 48 GB; TTFT 11.71× vs Fiddler.
  - No llama.cpp baseline.
- **WiSP** (Nokia; 2606.21868, v1 20 Jun 2026, v2 30 Aug 2026). (abs) — [arXiv 2606.21868](https://arxiv.org/abs/2606.21868)
  - A routing-aware expert pager plugged into an unmodified engine, on an RTX 3090 24 GiB.
  - Claims "up to 2.0× the decode throughput of static offload at the same memory budget" (vLLM `--cpu-offload-gb`).
  - Finds prefetch "does not help in single-stream decode: the bottleneck is PCIe bandwidth".
  - Absolute Qwen3-30B-A3B rate is ≈6.5–6.8 tok/s (prior notes).
- **SSD-LLaMA** (2609.18110, v1 16 Sep 2026). (abs) — [arXiv 2609.18110](https://arxiv.org/abs/2609.18110)
  - SSD-native, three-tier storage with CPU-GPU hybrid execution, on a single RTX 5090 with ≤32 GB RAM.
  - Claims decode 2.10–15.58× "over the evaluated baselines" and >1 tok/s on a trillion-parameter model.
  - llama.cpp flags are unstated (prior notes).
- **ExactMoE** (2608.15383, v1 15 Aug 2026). (abs) — [arXiv 2608.15383](https://arxiv.org/abs/2608.15383)
  - W4 routed experts in pinned host memory plus a GPU slot cache, single L4, OLMoE.
  - A 16-slot config keeps "81.85% of BF16 decode throughput" at 1.836 GiB. A 64-slot config reaches 31.923 vs 21.662 tok/s.
- **MoEpic** (2509.08342, v1 10 Sep 2025). (abs, text) — [arXiv 2509.08342](https://arxiv.org/abs/2509.08342)
  - Mechanism: caches the top segment of hot experts and prefetches the next layer's experts.
  - Latency is 37.51–65.73% lower than Pre-gated, AdapMoE and MoE-Infinity baselines (A6000/A100-class GPUs).
- **ExpertFlow (Shen et al.)** (2510.26730, 30 Oct 2025). (abs) — [arXiv 2510.26730](https://arxiv.org/abs/2510.26730)
  - "Continuously adjusts its prediction horizon", plus cache-aware routing.
  - Stall time is "less than 0.1% of the baseline".
- **ExpertFlow (He et al.)**: v2 2 Apr 2026, DAC'26; A40, throughput-oriented. (abs) — [arXiv 2410.17954](https://arxiv.org/abs/2410.17954)
- **PreScope / LayerScope** (2509.23638, v1 28 Sep 2025; ICS'26). (abs) — [arXiv 2509.23638](https://arxiv.org/abs/2509.23638)
  - Multi-batch scheduling on "legacy servers", with a learnable layer-aware predictor and cross-layer scheduling producing "globally optimal plans".
  - Claims "141% higher throughput and 74.6% lower latency".
  - Models: Mixtral, DeepSeek-MoE, Qwen3-30B-A3B, Moonlight.
- **DuoServe-MoE** (2509.07379, v1 9 Sep 2025, v2 9 Apr 2026). (abs) — [arXiv 2509.07379](https://arxiv.org/abs/2509.07379)
  - Phase-specialised prefetch and caching on A5000/A6000.
  - Claims TTFT up to 5.34× and end-to-end 7.55× over baselines.
- **Speculating Experts** (2603.19289, 9 Mar 2026). (abs) — [arXiv 2603.19289](https://arxiv.org/abs/2603.19289)
  - Up to 14% TPOT reduction "over on-demand loading" (A6000, A100, GH200; gpt-oss and Qwen3).
  - Its analytic gain bound is ΔT = Σ min(t_copy, t_compute), "maximum achievable speedup is 2×" (prior notes).
- **Pre-attention expert prediction (Zhu et al.)** (2511.10676): 93.03% / 94.69% / 97.62% accuracy on DeepSeek-V2-Lite / Qwen3-30B / Phi-mini-MoE. **Already cited** — [arXiv 2511.10676](https://arxiv.org/abs/2511.10676)
- **SeqMoE** (2609.12978, still v1). Belady variants are covered in Q2. **Already cited** — [arXiv 2609.12978](https://arxiv.org/abs/2609.12978)
- **MoE-Beyond** (2508.17137, 23 Aug 2025). (abs) — [arXiv 2508.17137](https://arxiv.org/abs/2508.17137)
  - A learned transformer predictor trained on "66 million expert activation traces" (DeepSeek-V2-Lite).
  - Simulated GPU hit rate rises "from 17% to 72% when only 10% of experts fit". Simulation only.
- **DynaExq** (2511.15015, v4 10 Sep 2026). (abs) — [arXiv 2511.15015](https://arxiv.org/abs/2511.15015)
  - Mixed-precision resident set on a single GPU; Qwen3-30B and 80B.
  - Claims "up to 2.73× higher throughput than offloading/prefetch baselines at batch size 32". Lossy.
- **SliceMoE** (2512.12990; DAC'26): bit-sliced caching; decode latency up to 1.81× / 1.64× on DeepSeek-V2-Lite / Qwen1.5-MoE. (abs) — [arXiv 2512.12990](https://arxiv.org/abs/2512.12990)
- **FluxMoE** (2604.02715, v3 10 Sep 2026): datacenter, large batch. (abs) — [arXiv 2604.02715](https://arxiv.org/abs/2604.02715)
  - Up to 7.2× vLLM on 8×H20.
  - 4.3× KTransformers' throughput on Mixtral on 2×L40S.

#### C. Lossy, substitution or precision-switching systems (outside an exact-routing bound)
- **FloE** (2505.05950, ICML 2025; just before the window): 48.7× vs DeepSpeed-MII on an RTX 3090, with 4.4–7.6% degradation. — [arXiv 2505.05950](https://arxiv.org/abs/2505.05950)
- **BuddyMoE** (2511.10054, 13 Nov 2025): substitutes "buddy" experts on prefetch failure. — [arXiv 2511.10054](https://arxiv.org/abs/2511.10054)
- **MoBiLE** (2510.12357, ASP-DAC'26): big-little experts; 1.60–1.72× on a consumer GPU. — [arXiv 2510.12357](https://arxiv.org/abs/2510.12357)
- **DyMoE** (2603.19172): TPOT up to 14.58× on edge. — [arXiv 2603.19172](https://arxiv.org/abs/2603.19172)
- **Low-rank compensation** (2512.17073). — [arXiv 2512.17073](https://arxiv.org/abs/2512.17073)
- **AcceptMoE** (2608.02989): verifier expert restriction; 2.06× under offload in SGLang at batch 1, with a 0.27 pp accuracy drop. — [arXiv 2608.02989](https://arxiv.org/abs/2608.02989)
- **CAEE** (2606.29982): prunes low-importance high-cost experts; 8–18% latency reduction on DeepSeek-R1. — [arXiv 2606.29982](https://arxiv.org/abs/2606.29982)
- **APEX** (2608.11688, CODES'26): simulated edge; its correctness-preserving mode is "up to 26%" lower latency. — [arXiv 2608.11688](https://arxiv.org/abs/2608.11688)
- **Edge0 / "The Other Half of the Memory Wall"** (2609.18063): the predicted routing replaces the true routing; 35B MoE at 20 tok/s in 3 GiB on a 24 GB machine. — [arXiv 2609.18063](https://arxiv.org/abs/2609.18063)

#### D. Speculative-decoding-coupled offload (multi-token verification, outside a one-token-per-pass bound)
- **SpecMoEOff** (2508.21706, v2 31 Oct 2025): "up to 2.5x decode throughput", with roofline-guided CPU/GPU orchestration on an RTX 4090. — [arXiv 2508.21706](https://arxiv.org/abs/2508.21706)
- **SP-MoE** (2510.10302): 1.07–3.5× TPOT. — [arXiv 2510.10302](https://arxiv.org/abs/2510.10302)
- **MoE-SpeQ** (2511.14102): up to 2.34× on Phi-MoE, with an "Amortization Roofline Model". — [arXiv 2511.14102](https://arxiv.org/abs/2511.14102)
- **MoE-SpAc** (2603.09983): "SD … as an informative lookahead sensor"; 42% over llama.cpp-with-SD; 4.04× average; batch 1. — [arXiv 2603.09983](https://arxiv.org/abs/2603.09983)
- **EcoSpec** (2607.12696): up to 1.62×, including GPT-OSS-120B. — [arXiv 2607.12696](https://arxiv.org/abs/2607.12696)
- **DraftExpert** (2607.24434): 1.45× average on RTX 4090 and Snapdragon. — [arXiv 2607.24434](https://arxiv.org/abs/2607.24434)
- **S2-MoE** (2608.15018): in llama.cpp; up to 5.3× vs AR decode. — [arXiv 2608.15018](https://arxiv.org/abs/2608.15018)
- **"Limits of Speculation"** (2609.22156): an SSP oracle for speculation budget. — [arXiv 2609.22156](https://arxiv.org/abs/2609.22156)

#### E. Router-training methods that change routing (cite as orthogonal; they change the trace, so the bound must be recomputed on their routing)
- **Oracle-MoE**: ICML 2025, cited by MELINOE and MaskCoFT. Text, not verified directly — [MaskCoFT arXiv 2609.34077](https://arxiv.org/abs/2609.34077)
- **ReMoE** (2605.27081, ICML 2026): expert reuse +26%; llama.cpp on Jetson Orin NX decode 1.77–1.99×. — [arXiv 2605.27081](https://arxiv.org/abs/2605.27081)
- **StickyMoE** (2607.08780): switch rate −60%. — [arXiv 2607.08780](https://arxiv.org/abs/2607.08780)
- **MELINOE** (2602.11192): 1.2–3× over efficient baselines. — [arXiv 2602.11192](https://arxiv.org/abs/2602.11192)
- **Cache-Aware Joint Router Adaptation** (2609.04895): simulator only. — [arXiv 2609.04895](https://arxiv.org/abs/2609.04895)
- **MaskCoFT** (2609.34077, v2 2 Oct 2026): fetches −23.7% for Mixtral at 4 experts per layer; TPOT −16.4%. — [arXiv 2609.34077](https://arxiv.org/abs/2609.34077)
- **Cacheable by Design** (2608.18261, pre-registered negative result). — [arXiv 2608.18261](https://arxiv.org/abs/2608.18261)

#### F. Edge, NPU, SSD and other tiers (cite only if the related work covers tiers)
- **Budgeting Bytes** (2609.04238): RK3588 eMMC and Apple M4. — [arXiv 2609.04238](https://arxiv.org/abs/2609.04238)
- **Paging the Experts** (2609.29032): iPhone flash. — [arXiv 2609.29032](https://arxiv.org/abs/2609.29032)
- **MawForge** (2607.09686): macOS unified memory. — [arXiv 2607.09686](https://arxiv.org/abs/2607.09686)
- **RotaryQuant** (2608.08081): LRU disk paging on Metal; 9–19 tok/s. — [arXiv 2608.08081](https://arxiv.org/abs/2608.08081)
- **SpecPrefetch** (2607.24787): Snapdragon 8 Elite; +20% decode. — [arXiv 2607.24787](https://arxiv.org/abs/2607.24787)
- **NPUMoE** (2604.18788): Apple ANE. — [arXiv 2604.18788](https://arxiv.org/abs/2604.18788)
- **MoE-CORE** (2610.01950, 1 Oct 2026): NPU appliance; "nonuniform layer-wise cache capacity … routing-history-aware replacement". — [arXiv 2610.01950](https://arxiv.org/abs/2610.01950)
- **OD-MoE** (2512.03927): 10-node edge; "approximately 75% of the decoding speed of a fully GPU-cached MoE deployment while using only 1/3 of the GPU memory". — [arXiv 2512.03927](https://arxiv.org/abs/2512.03927)
- **GPU-NDP scheduling** (2601.03992, DATE'26). — [arXiv 2601.03992](https://arxiv.org/abs/2601.03992)
- **TriMoE** (2603.01058, DAC'26). — [arXiv 2603.01058](https://arxiv.org/abs/2603.01058)
- **CoX-MoE** (2605.17889, DAC'26). — [arXiv 2605.17889](https://arxiv.org/abs/2605.17889)
- **SSD-offload energy** (2508.06978, IEEE CAL): up to ~12× energy vs HBM. — [arXiv 2508.06978](https://arxiv.org/abs/2508.06978)
- **Fermi C2075 hybrid** (2606.24031): CPU decode 2.8 → 8.6 tok/s. — [arXiv 2606.24031](https://arxiv.org/abs/2606.24031)

#### G. Datacenter or multi-GPU items that surfaced but are out of scope
- PROBE (2602.00509) — [arXiv](https://arxiv.org/abs/2602.00509)
- METRO (2512.09277) — [arXiv](https://arxiv.org/abs/2512.09277)
- ThunderEP (2609.40093; EP over PCIe on 4090/5090) — [arXiv](https://arxiv.org/abs/2609.40093)
- VAMP (2609.13537) — [arXiv](https://arxiv.org/abs/2609.13537)
- WriteScope (2609.14507; KV split) — [arXiv](https://arxiv.org/abs/2609.14507)
- MonoMoE (2609.04244; H200 kernel) — [arXiv](https://arxiv.org/abs/2609.04244)

#### Pre-window items whose venue versions fall in the window
- FineMoE / fMoE: EuroSys'26; v2 4 Oct 2025 — [arXiv 2502.05370](https://arxiv.org/abs/2502.05370)
- Fiddler: ICLR 2025, v3 1 May 2025 — [arXiv 2402.07033](https://arxiv.org/abs/2402.07033)
- ProMoE: v3 1 Sep 2025 — [arXiv 2410.22134](https://arxiv.org/abs/2410.22134)
- DAOP: v3 21 Aug 2026 — [arXiv 2501.10375](https://arxiv.org/abs/2501.10375)
- MoE-APEX: ASPLOS'26; evaluation details unverified; ACM DL returned 403 in prior notes — [ACM DL](https://dl.acm.org/doi/10.1145/3779212.3790187)

### Inferences
- **Paper must add to related work (systems).** These are exact-routing, consumer or workstation GPU, batch-1 or small-batch, and not yet cited:
  1. FlashMoE (also central for (e));
  2. OSDI'26 2606.10493;
  3. Pipelined Sharding (MLSys'26);
  4. ATSInfer (gpt-oss-120b on RTX 4090 vs llama.cpp);
  5. WiSP;
  6. SAEM (also central for (c));
  7. SSD-LLaMA;
  8. Mira;
  9. KTransformers' dynamic expert update (v0.5.1);
  10. llama.cpp PR #29887 (concurrent upstream LRU cache; see Q3).

  OLED-MoE and RapidMoE (EuroSys'27) merit one sentence each as "CPU-miss cache, but dLLM / lossy".
- **Head-to-head expectations.** None of these reports gpt-oss-120b on an RTX 5090 against `--n-cpu-moe` at equal memory. ATSInfer is the closest (gpt-oss-120b, RTX 4090, llama.cpp baseline with unstated flags). The FreeToken comparison (11 of 12 cells) remains the paper's only direct system race, unless ATSInfer or Pipelined Sharding (both llama.cpp-based, with public code) are added.
- **Two separate lists.** The large multipliers in the window (3–15×) come almost entirely from lossy designs, speculative coupling, weak baselines (DeepSpeed, MoE-Infinity, vLLM static offload) or SSD tiers. The paper's related-work paragraph should separate "exact-routing, one token per pass" systems (the paper's bound applies) from the rest (it does not).

### Gaps
- The arXiv API searches abstracts, not full text. Papers that use Belady or oracles only in their body (as FlashMoE and 2505.16056 do) could be missed if their abstracts lack the keywords. The catalogue is therefore not provably exhaustive.
- Non-arXiv venue papers were not swept: ATC'26 and EuroSys'27 programmes, ISCA/MICRO'25 proceedings, IEEE early access. Semantic Scholar citation lists were not queried this pass; the earlier pass got HTTP 429.
- Not re-read in full text: BuddyMoE (PDF text extraction failed), Speculating Experts' figures, MoE-APEX.
- Not verified:
  - SPICE, SeqMoE and FreeToken numbers were taken from prior notes.
  - Mira's "lossy" status: it states "minimally degrading accuracy" for its compression.

## Q2. Has anyone published a lower bound / speed-of-light / optimal-cache (Belady MIN) comparison for MoE expert caching in seconds, an in-engine oracle study, or a "how far from optimal" measurement?

### Takeaway
- **No one converts an exact-routing MIN-with-bypass miss count into a seconds-domain speed limit using measured host-read and PCIe rates, and no one runs a randomised in-engine oracle factorial. So (a) and (c) remain unoccupied.** Partial precedents for each must be cited:
  - **(a):** Budgeting Bytes, WiSP, Paging the Experts, Euro-Par'25 layered paging.
  - **(c):** SAEM's in-engine clairvoyant oracle; Budgeting Bytes' in-engine perfect-prefetch oracle; 2608.12103's one-layer oracle advice; WiSP's prefetch-hurts result.
- **(d) is the most exposed:**
  - An ICLR 2026 paper (2505.16056) measures, across 20 MoE models, the hit rate of a lookahead-m oracle cache as a function of cache ratio ρ = cache size / active experts, i.e., C/k, and of window m. It reports LRU and LFU against the clairvoyant optimum.
  - 2608.07911 already reports a 44–46% gap between causal policies and MIN-with-bypass.
  - FlashMoE, 2509.02408, SeqMoE and 2608.12103 each report online-versus-Belady gaps.
  - No paper fits a lookahead scaling law (W50 vs C/k).
- **(e) has a close precedent.** FlashMoE learns a cache policy on causal recency and frequency features against Belady labels and gets +7% speed on Qwen3-30B-A3B. 2608.07911's causal next-use predictor recovers −11.4% of the gap.

### Cited Findings

#### Seconds-domain bounds and hardware-limit framings (threats to (a))
- **Budgeting Bytes** (2609.04238; v1 stamp 29 Jul 2026, announced Sep 2026; workshop draft). (abs, text) — [arXiv 2609.04238](https://arxiv.org/abs/2609.04238)
  - "reduces prefetch scheduling to single-machine feasibility with release times, yielding a closed-form windowed roofline".
  - Earlier notes quote: "prefetch changes when and how bytes are read, never how many … no schedule beats bytes/B".
  - It is a hiding-feasibility / exposed-latency bound for a given fetch set, not a lower bound over caching policies. It has no MIN, no bypass and no CPU execution of misses.
- **WiSP** (2606.21868 v2). (abs; formula from prior notes) — [arXiv 2606.21868](https://arxiv.org/abs/2606.21868)
  - Latency model "T(f) = c_exp m_exp(C) + c_kv m_kv(κ)" with "c_exp = M_e t_byte", where m(C) is the LRU reuse-distance miss curve.
  - This makes it a seconds estimator for LRU, not a bound.
- **Paging the Experts** (2609.29032, v1 24 Sep 2026). (prior notes) — [arXiv 2609.29032](https://arxiv.org/abs/2609.29032)
  - "The relaxed unpinned oracle is a lower miss bound under equal-size, mandatory-admission paging, not an online speed prediction."
  - Oracle vs LRU hit rates: 54.73% vs 0.00% at 512 MiB; 68.31% vs 47.59% at 1,024 MiB (Qwen3.6-35B-A3B).
- **"Cache Management for Mixture-of-Experts LLMs"** (Angelopoulos, Marchal, Obrecht, Simon; arXiv 2509.02408, 2 Sep 2025; Euro-Par 2025 per the author page). (text) — [arXiv 2509.02408](https://arxiv.org/abs/2509.02408); [author page](https://perso.ens-lyon.fr/adrien.obrecht/publications/2025-08-europar/)
  - Defines "a new paging problem that models expert management" (ℓ-layered paging). It proves competitive-ratio lower bounds: deterministic ≥ k − ℓ + 1, and Cr(LRU) ≥ k when ℓ divides k+1. A randomised bound is also given.
  - Proposes a layer-aware LRU (LLRU).
  - Simulates on Mixtral (1,000 prompts) and Llama-MoE traces against "an optimal offline strategy Opt … Belady's rule". It finds "a sizable gap to Opt that can go up to ×2.5" and compares per-layer ("-Dist") with shared caches.
  - Results are fault counts, not seconds; there is no bypass (classical paging) and no hardware.
- **Cacheable by Design?** (2608.18261, 18 Aug 2026). (abs; numbers from prior notes) — [arXiv 2608.18261](https://arxiv.org/abs/2608.18261)
  - Llama.cpp with `-ngl 99 -ncmoe 94` on an RTX 3070 8 GB measures "0.44 tok/s warm, matching a bytes-per-token / bandwidth model".
  - Per-layer simulation at 13.4% budget: LRU 65.9%, LFU 60.1%, static pin 59.2%, Belady 79.1%.
- **OSDI'26 2606.10493**: KTransformers decode "is only about 50% of the nominal aggregate DDR5 bandwidth" (fraction-of-peak framing, no policy bound). (text) — [arXiv 2606.10493](https://arxiv.org/abs/2606.10493)
- **MoE-Lens** (2504.09345, Apr 2025; pre-window): a "theoretical performance upper bound" for high-throughput, batched serving, not batch-1. (prior notes) — [arXiv 2504.09345](https://arxiv.org/abs/2504.09345)
- **Analytical models in system papers** (models, not bounds):
  - FreeToken's q⋆ ≈ m·B_P/B_H. (prior notes) — [arXiv 2608.16157](https://arxiv.org/abs/2608.16157)
  - SpecMoEOff's "theoretical and empirical roofline". (abs) — [arXiv 2508.21706](https://arxiv.org/abs/2508.21706)
  - MoE-SpeQ's "Amortization Roofline Model". (abs) — [arXiv 2511.14102](https://arxiv.org/abs/2511.14102)
  - Speculating Experts' 2× cap. (prior notes) — [arXiv 2603.19289](https://arxiv.org/abs/2603.19289)
  - ATSInfer's per-boundary transfer cost c_i = S_in,i / B_pcie. (text) — [arXiv 2607.10183](https://arxiv.org/abs/2607.10183)

#### In-engine oracle studies (threats to (c))
- **SAEM: "Prediction-Oracle Upper-Bound Analysis"** (2608.21614). (text) — [arXiv 2608.21614](https://arxiv.org/abs/2608.21614)
  - It "construct[s] a clairvoyant oracle that replaces the predictive proxy with perfect knowledge of the upcoming stage while preserving the cache capacity, update events, and cache-enforcement mechanism".
  - Protocol: "Fixed-sequence replay … fixes the prompts, decoding length, and boundary events".
  - Metrics: η_CHR = CHR_SAEM / CHR_Oracle and η_TP = T_SAEM / T_Oracle.
  - On Qwen3 (A100 + Xeon), η_TP is 87.87 / 90.82 / 92.80% at batch 1 and 95.69 / 98.46 / 100.78% at batch 8, at expert cache ratios 50 / 25 / 12.5%.
  - Implication: its oracle is worth at most ≈14% at batch 1. The oracle is a stage-level activation profile (frequency), not MIN admission; one host; not randomised.
- **Budgeting Bytes, trace-driven oracle in the deployed engine.** (text) — [arXiv 2609.04238](https://arxiv.org/abs/2609.04238)
  - Method: "a first pass records the exact experts each token uses, a second (greedy, identical tokens) prefetches them perfectly one token ahead. Perfect prediction yields no speedup (0.12 → 0.13)".
  - Temporal-locality prefetch is "net-negative: 0.19 → 0.12 tok/s". This is on an RK3588 with eMMC, Qwen3-30B-A3B.
  - It also reports an A100 PCIe offload path where "prefetch-overlap tops out at 1.34×" (prior notes).
- **Si et al., "Who Should Own the Expert Cache?"** (2608.12103 v2). (abs) — [arXiv 2608.12103](https://arxiv.org/abs/2608.12103)
  - "At 64.7% measured recall, router lookahead changes median time by 0.3% … perfect one-layer advice gains 5.0% through the same interface and nothing through blocking reads".
  - "At equal enforced memory, kernel recency serves essentially the same demand as an oracle static-frequency policy". This is the SSD/DRAM page-cache tier on GH200, not PCIe→GPU.
- **WiSP**: "speculative transfers compete with demand transfers instead of hiding them". Earlier notes record prefetch "lowers decode throughput by 38–60% at constrained caps" (RTX 3090). (abs; number from prior notes) — [arXiv 2606.21868](https://arxiv.org/abs/2606.21868)
- **Speculating Experts**: prefetch gains of 5–14% TPOT, larger at long context. Not an oracle. (prior notes) — [arXiv 2603.19289](https://arxiv.org/abs/2603.19289)
- **Pipelined Sharding**: "oracle comparison across 105 configurations" for strategy selection. Configuration oracle only. (text) — [arXiv 2604.26334](https://arxiv.org/abs/2604.26334)

#### Online-vs-optimal gaps and lookahead studies (threats to (d))
- **"Not All Models Suit Expert Offloading: On Local Routing Consistency of MoE Models"** (Liang et al.; arXiv 2505.16056; v1 21 May 2025, v4 28 Feb 2026; **ICLR 2026**). (abs, text) — [arXiv 2505.16056](https://arxiv.org/abs/2505.16056); code [moe-lrc](https://github.com/ljcleo/moe-lrc)
  - SCH definition: the "Segment Cache Best Hit Rate (SCH), which measures the hit rate of an expert cache utilizing a length of future information under a cache limit". The oracle "evicts experts that are activated the least times in the next m tokens", with cache size set by "cache ratio ρ … the ratio between the cache size and the number of activated experts".
  - Coverage: 20 MoE LLMs (3B–54B), m ∈ {4, 16, 64, 256}.
  - Corpus: 22,528 samples × 512 tokens from RedPajama plus LMArena, OpenMath, OpenCode and OpenScience. This is teacher-forced dataset text, not model generations.
  - Relative to the clairvoyant optimum (=100), for the model labelled "Baseline" (likely one of its toy models; unverified):
    - LRU = 56.49 / 67.04 / 75.26 at ρ = 1 / 2 / 3;
    - LFU = 61.92 / 70.87 / 78.35;
    - SCH(m=16) = 80.97 / 90.55 / 96.23.
  - Its recommendation: "cache sizes approximately twice the active experts".
  - Differences from (d): mandatory admission (no bypass), hit-rate units (not bytes or seconds), a frequency-in-window oracle rather than a W50 fit, and no ARC or S3-FIFO.
- **"Reproducible Evaluation of MoE Expert Caching"** (Yu Zhang; 2608.07911, v4 25 Aug 2026). (abs) — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)
  - "a stable gap to the offline optimum remains (44.2-45.9% over 13 frozen workload compositions)".
  - "A forced-admission oracle attributes 84.3-96.6% of it to knowing which resident expert is used furthest in the future".
  - "permuting only the temporal order of an identical event stream moves the offline-optimal gap from 44.9% to 30.8%".
  - Three models (Granite-3.1-3B-A800M, OLMoE, Qwen3-30B-A3B MLX 4-bit). Results are block counts only (prior notes).
- **FlashMoE** (2601.17063 v2). (text) — [arXiv 2601.17063](https://arxiv.org/abs/2601.17063)
  - "compared to Belady's optimal algorithm with an 86% hit rate, LRU achieves only around 73%, resulting in nearly 1.9× more I/O operations".
  - "LRU's evicted experts were reused 34.2% of the time [within 5 steps] … significantly higher than Belady's 0.1%".
  - On Qwen3-30B-A3B, "LRU made the better choice approximately 56% of the time" versus LFU.
- **SeqMoE** (already cited). (text) — [arXiv 2609.12978](https://arxiv.org/abs/2609.12978)
  - "Belady uses the complete future activation sequence … Belady-8 limits this oracle knowledge to the next eight steps, while Belady-F8 replaces oracle activations with our eight-step forecasts".
  - Finding: "future-aware policies consistently outperform history-based ones". Hit-rate curves on Qwen3-30B-A3B-FP8 (simulation).
  - This is the only lookahead-limited Belady point found in the window. It is a single window, with no scaling law.
- **Si et al. 2608.12103**: LRU reaches "approximately 70% of Belady's" at one scale (prior notes); recency ≈ oracle static-frequency (abs). — [arXiv 2608.12103](https://arxiv.org/abs/2608.12103)
- **SpecMD** (Apple; 2602.03921): "MoE expert access is not consistent with temporal locality assumptions (e.g LRU, LFU)". Its "Least-Stale" policy cuts collision misses "by up to 85× over LRU". A100 emulation; no Belady comparison (grep found none). (abs, text) — [arXiv 2602.03921](https://arxiv.org/abs/2602.03921)
- **In-depth Analysis on Caching and Pre-fetching** (2511.05814): LFU beats LRU, plus traces. No Belady (grep). (abs, text) — [arXiv 2511.05814](https://arxiv.org/abs/2511.05814)
- **Mixture of Cache-Conditional Experts** (Qualcomm; TMLR 06/2025): treats Belady as "a theoretical upper bound for cache policy". (prior notes) — [arXiv 2412.00099](https://arxiv.org/abs/2412.00099)
- **Older closest prior art (pre-window)**: "Towards MoE Deployment" (Huang et al., arXiv 2303.06182, 2023) compares Expert Buffering to "Belady's MIN", "very close to Belady's MIN". (prior notes) — [arXiv 2303.06182](https://arxiv.org/abs/2303.06182)

#### Learned cache policies (threats to (e))
- **FlashMoE learned policy.** (text) — [arXiv 2601.17063](https://arxiv.org/abs/2601.17063)
  - Inputs are recency r_t and frequency f_t, normalised.
  - "The target data is generated based on the ideal, oracle[1]-based optimal experts to be evicted".
  - The model is a per-layer FFN of about 113 KB, trained on TriviaQA.
  - Results: hit rate +21% vs LRU on OLMoE; speed +22% (OLMoE) and +7% (Qwen3-30B-A3B) over LRU on an RTX 5070 Ti with an SSD tier.
- **2608.07911**: "A causal next-use predictor, used as an eviction rule, recovers -11.4% of the gap; it picks an optimal victim 3.4% of the time, against 2.4% for a random resident block and 20.6-22.1% for LRU and LFRU". (abs) — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)
- **MoE-Beyond**: learned activation predictor; simulated hit rate 17% → 72% at 10% cache. (abs) — [arXiv 2508.17137](https://arxiv.org/abs/2508.17137)
- **SeqMoE**: "probabilistic Belady policy" driven by a learned seq2seq forecast (already cited). — [arXiv 2609.12978](https://arxiv.org/abs/2609.12978)
- **Other non-learned scored-replacement policies**:
  - DALI's "Workload-Aware Cache Replacement" (abs) — [arXiv 2602.03495](https://arxiv.org/abs/2602.03495)
  - Mira's routing-telemetry HOT+STAGE (abs) — [arXiv 2609.38090](https://arxiv.org/abs/2609.38090)
  - MoE-CORE's "routing-history-aware replacement" (abs) — [arXiv 2610.01950](https://arxiv.org/abs/2610.01950)
  - Euro-Par'25 LLRU (text) — [arXiv 2509.02408](https://arxiv.org/abs/2509.02408)

### Inferences

**Threat ranking by contribution**

| Contribution | Status | Most-threatening items | What remains distinctive |
|---|---|---|---|
| (a) speed limit in seconds | Intact; cite 5 | Budgeting Bytes (bytes/B floor, exposed-latency bound); WiSP (LRU misses × per-miss seconds); Paging the Experts (Belady miss bound "not an online speed prediction"); Euro-Par'25 (competitive-ratio lower bounds, OPT gap ×2.5); 2608.18261 (bytes/bandwidth model matches llama.cpp `-ncmoe`) | MIN-with-bypass over all exact-routing policies with C slots per layer; measured host-read *and* PCIe rates; CPU-or-PCIe miss service; result in seconds |
| (c) in-engine oracle factorial | Intact; cite 4 | SAEM (in-engine clairvoyant oracle, fixed-sequence replay, η_TP 88–101%); Budgeting Bytes (in-engine perfect-prefetch oracle, no gain); 2608.12103 (perfect one-layer advice +5.0%); WiSP (prefetch hurts single-stream decode) | MIN-spent foresight (admit only what MIN admits, read once) vs Belady-style prefetch / copy-after-serve; byte accounting (1.6–2.5× MIN's bytes); randomised factorial on three hosts. The *qualitative* "prefetch cannot beat the byte floor" message is already in Budgeting Bytes and WiSP; frame the paper's version as the quantitative, policy-resolved form |
| (d) nine-model traces, online vs MIN, W50 law | **Most exposed**; cite 6 | 2505.16056 (ICLR'26: lookahead-window oracle × cache ratio C/k × 20 models; LRU/LFU at 56–78% of optimal hit rate); 2608.07911 (44–46% causal-vs-MIN gap, with bypass); FlashMoE (Belady vs LRU, 1.9× I/O); Euro-Par'25 (OPT gap up to ×2.5); SeqMoE (Belady-8); 2608.12103 | The W50 ≈ 0.59 (C/k)^1.33 scaling law (no one fits lookahead vs C/k); bypass MIN in bytes across nine current models including gpt-oss; ARC and S3-FIFO included; the model's own generations as traces. "Online policies read 35–110% more than MIN" is **not** new as a qualitative finding |
| (e) learned admission | Exposed; cite 2 | FlashMoE (learned FFN on recency/frequency trained on Belady labels, +7% speed on Qwen3-30B-A3B); 2608.07911 (causal next-use predictor recovers −11.4% of gap) | Admission rather than eviction; logistic regression on causal features; an honest in-engine null (0.90–1.05×) despite 6–18% fewer reads. This agrees with FlashMoE's small +7% on Qwen3-30B-A3B and with 2608.07911 |

**Must add, not in the current related work:**
- 2505.16056 (ICLR'26)
- 2608.07911
- 2509.02408 (Euro-Par'25)
- FlashMoE 2601.17063
- Budgeting Bytes 2609.04238
- SAEM 2608.21614
- 2608.12103
- WiSP 2606.21868
- Paging the Experts 2609.29032
- 2608.18261
- SpecMD 2602.03921
- Speculating Experts 2603.19289
- Older closest prior art: 2303.06182 (Belady MIN comparison, 2023) and 2412.00099 (TMLR'25)

**Suggested positioning sentences:**
- For (d): "2505.16056 shows that a frequency oracle with a 16-token window nearly matches the clairvoyant hit rate at moderate cache ratios. We quantify how much window is needed as a function of C/k (W50 ∝ (C/k)^1.33), in bytes, under bypass."
- Check the paper's W50 numbers against 2505.16056's SCH(m, ρ) curves for any shared model (Mixtral-8x7B, DeepSeek-V2-Lite, Qwen1.5-MoE, OLMoE are in their 20). Agreement or disagreement is a free validation point. This is a suggestion.

### Gaps
- 2505.16056 per-model SCH(m, ρ) values are in figures, not text. Which model "Baseline" denotes in its Table 4 was not confirmed.
- Whether SAEM's oracle replay was run at batch 1 on the same "Qwen3" as its main results (presumably Qwen3-30B-A3B) was not confirmed from the table header.
- No paper was found that reports a W50-style lookahead-to-close-gap metric, but full-text search of all 2025–2026 arXiv was not possible.
- The OpenReview (ICLR'27 submission) cycle was not searched. Submissions that match (d) could appear there after early October 2026.

## Q3. What did llama.cpp, ik_llama.cpp and KTransformers add in 2025–2026 for MoE on consumer GPUs, and what tok/s are reported for gpt-oss-120b / Qwen3-30B-A3B on RTX 5090/4090 + desktop CPU?

### Takeaway
Mainline llama.cpp gained:
- `-ot` (2 Apr 2025);
- `--cpu-moe` (31 Jul 2025);
- `--n-cpu-moe` (4 Aug 2025);
- CUDA graphs with `--n-cpu-moe` (24 Jan 2026).

It still has **no merged expert cache** as of master on 6 Oct 2026. On 3 Oct 2026, Aman Gupta, a llama.cpp CUDA contributor, opened **PR #29887, "llama : add a GPU cache for MoE experts kept in host memory"**:
- a port of the QVAC-fabric LRU cache;
- all misses uploaded to the GPU;
- one LRU per group of same-layout layers;
- `--moe-cache-mib`.

It is a concurrent upstream competitor to (b) and should be cited as such. Community forks and PRs (2026) report +10% to +90% over `--n-cpu-moe`.

KTransformers added per-layer GPU expert placement with prefill-triggered dynamic updates (v0.5.1, Jan 2026). It lacks MXFP4/gpt-oss CPU kernels per ATSInfer.

Public gpt-oss-120b numbers on consumer GPUs with `--n-cpu-moe`:
- ≈22 tok/s on RTX 4090 + i5-13400F;
- ≈36 tok/s on RTX 5090 + 9950X3D (`-ncmoe 25`);
- 47.6 tok/s with a static hot-set fork;
- 127 tok/s for FreeToken at about 40% of experts resident.

### Cited Findings

**Mainline llama.cpp, verified from master `git log` (git)** — [llama.cpp commits](https://github.com/ggml-org/llama.cpp/commits/master)
- 2025-04-02 `e0e912f`: "llama : add option to override model tensor buffers (#11397)" (`-ot`). 2025-04-27: llama-bench `--override-tensors` (#12922).
- 2025-07-31 `a06ed5f`: "add simple option to enable CPU for MoE weights (--cpu-moe) (#14992)".
- 2025-08-04 `ec428b0`: "llama : add --n-cpu-moe option (#15077)". 2025-08-13: draft-model variants (#15191).
- 2025-08-20: "sched : copy only the used experts when offloading prompt processing (#15346)". This is for prefill.
- 2025-09-16: llama-bench `--n-cpu-moe` (#15952).
- 2026-01-05: "CUDA: disable cuda graph when using n-cpu-moe (#18593)".
- 2026-01-24: "ggml-cuda: enable cuda-graphs for `n-cpu-moe` (#18934)". Earlier notes record gpt-oss-120b MXFP4 tg 95.96→100.93 at ncmoe 8 and 40.87→44.90 at ncmoe 64 (GPU per the PR page, not re-checked) — [PR #18934](https://github.com/ggml-org/llama.cpp/pull/18934)
- 2026-03-11: fix `--n-cpu-moe`/`--cpu-moe` for fused gate+up (#20416). 2026-09-11: skip 0-sized ids tensor when offloading selected experts (#28739).
- `--fit` / `params_fit` exists by Dec 2025 (fix #18070, 2025-12-16).
- No master commit through `43fe9c6` (2026-10-06) adds an expert cache, expert prefetch or hot-expert pinning.
- Official gpt-oss guide (Discussion #15396, 2025-08-18) recommends `--n-cpu-moe`, e.g. `--n-cpu-moe 32` for 16 GB VRAM (prior notes) — [Discussion #15396](https://github.com/ggml-org/llama.cpp/discussions/15396)

**New upstream PR**
- **#29887** "llama : add a GPU cache for MoE experts kept in host memory". (git)
  - Author Aman Gupta; head `6b7b03a`, 2026-10-03; parent is "CUDA: fuse shared experts into MMVQ (#29184)".
  - Commit message: "Port of the qvac-fabric MoE cache. MUL_MAT_ID ops on host experts run on the GPU with an LRU cache of experts, only the misses are uploaded. Small batches only (<= 32 tokens) … Layers with different expert layouts get separate banks. Enable with --moe-cache-mib N."
  - Diff: 13 files, +748/−14, including a new `src/llama-moe-cache.cpp` (459 lines). Code comment: "layers with the same expert tensor layout share the banks and the LRU of a group", so the budget is shared across layers and every miss is admitted.
  - Merge state was not visible; GitHub UI and API access were blocked. — [PR #29887](https://github.com/ggml-org/llama.cpp/pull/29887)
  - QVAC Fabric is Tether Data's llama.cpp-based framework (launched Dec 2025 per press) — [crowdfundinsider](https://www.crowdfundinsider.com/2025/12/256112-tether-data-launches-qvac-fabric-llm-to-train-large-language-models-on-hardware/); [HF blog](https://huggingface.co/blog/qvac/fabric-llm-finetune)
- No other new MoE-cache PR among heads #29620–#30041. The others are kernel or Vulkan/SYCL MoE fixes and #29660 (`-ot` output device re-sync, 2026-10-04). (git)

**Earlier community llama.cpp caches** (prior notes; states are from refs, not the UI)
- RFC Discussion #24528 (leloch, 12 Jun 2026): VRAM cache with "thread 0 dispatch[ing] cached (hit) rows to GPU while other threads compute misses"; "+21% to +52%" on an RTX 5090 — [Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)
- #21609 (Apr 2026, closed): LFRU cache plus FATE prefetch, "1.79x decode speedup on GPT-OSS-120B with RTX PRO 2000 8GB (--n-cpu-moe 36 --expert-cache-slots 16)" — [PR #21609](https://github.com/ggml-org/llama.cpp/pull/21609)
- #27861 (GPU-resident LRU, draft, head 2026-08-28) — [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861)
- #28414 `--prefetch-experts-slots` (2026-09-04) — [PR #28414](https://github.com/ggml-org/llama.cpp/pull/28414)
- #26414 `--pin-hot-experts` (2026-09-24) — [PR #26414](https://github.com/ggml-org/llama.cpp/pull/26414)
- #29539 benchmark-only random routing (2026-09-27) — [PR #29539](https://github.com/ggml-org/llama.cpp/pull/29539)
- Earlier notes record maintainer push-back: am17an asked for an RFC because a cache PR was "too large for any maintainer to review"; pwilkin would not consider prefetch PRs unless they "clearly beat plain `--mmap`" (prior notes, from PR threads).

**ik_llama.cpp** (prior notes) — [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)
- Hybrid features: `-fmoe` fused MoE (Feb 2025), PR #698 "offload only activated experts" (prompt processing), `-ooae`.
- #2101 `--prefetch-experts` (MADV_POPULATE_READ, 2026-07-11).
- #2444 (2026-09-15): `-ot` CPU overrides land in pinned host memory.
- No dynamic GPU expert cache.
- Issue #1699 (27 Apr 2026): ik is *slower* than mainline for hybrid Qwen3.6-35B-A3B (TG 10.40 vs 15.69 t/s on a GTX 1660S + 5800X) — [issue #1699](https://github.com/ikawrakow/ik_llama.cpp/issues/1699)

**KTransformers** (tags verified via git; doc read at tag v0.7.1)
- Releases in the window: v0.3.2 (2025-07-01), v0.4.1 (2025-11-04), v0.5.0 (2025-12-24), v0.5.1 (2026-01-22, "add Experts sched tutorial"), v0.5.3 (2026-04-01), v0.6.2 (2026-05-03), v0.6.3 (2026-06-21), v0.7.0 (2026-08-17), v0.7.1 (2026-09-14) — [KTransformers releases](https://github.com/kvcache-ai/ktransformers/releases)
- "CPU-GPU Expert Scheduling" (`doc/en/kt-kernel/experts-sched-Tutorial.md`):
  - `--kt-num-gpu-experts` per MoE layer; placement strategies `uniform`/`frequency`/`front-loading`/`random`.
  - `--kt-enable-dynamic-expert-update` triggered by `--kt-gpu-prefill-token-threshold`.
  - Results on Qwen3-Next-80B-A3B-FP8 with 4×RTX 4090 (TP4), Xeon Gold 6454S, ShareGPT: at 10% GPU experts, dynamic update reaches 70.22 tok/s vs 56.57 for uniform (53.37 at 0%; 112.99 at 100%).
  - This is the "prefill-updated placement" that FreeToken compares against — [KTransformers repo](https://github.com/kvcache-ai/ktransformers)
- ATSInfer reports that KTransformers v0.5.2 "supports only a subset of MoE models because its custom CPU kernels do not support the MXFP4 format", i.e., no gpt-oss. (text) — [arXiv 2607.10183](https://arxiv.org/abs/2607.10183)

**Reported tok/s on consumer GPUs**
- RTX 4090 + Core i5-13400F, gpt-oss-120b GGUF (4 Oct 2025, WebFetch summary) — [zenn.dev](https://zenn.dev/kota_iizuka/articles/d5dbb66008305a)

  | Config | tok/s |
  |---|---|
  | `--n-gpu-layers 12` | 12.98 |
  | `--cpu-moe` | 17.77 |
  | `--n-cpu-moe 28` | 21.96 |
  | Ollama default | 9.8 |

- RTX 5090 + Ryzen 9950X3D + DDR5-6000: moe-autopilot static hot-set fork (prior notes) — [moe-autopilot](https://github.com/JigSawPT/moe-autopilot)
  - gpt-oss-120B `--n-cpu-moe 25`: 36.2 → 47.6 tok/s (+31.6%);
  - Qwen3.6-35B: 115.8 → 128.3.
- RTX 5090 + 128 GB, FreeToken 0.1.2 (community, 2026-08-22): gpt-oss-120b at "127.1 tok/s single-stream" with 40.4% of experts resident, about 29 GB VRAM (prior notes) — [zenn.dev/lifona](https://zenn.dev/lifona/articles/a8606bb95e17e1?locale=en)
- llama.cpp issue #27685 (late Aug 2026): RTX 5090, llama.cpp 22 t/s vs FreeToken 53 t/s. Flags undocumented; summary only (prior notes) — [issue #27685](https://github.com/ggml-org/llama.cpp/issues/27685)
- Qwen3-30B-A3B:
  - SeqMoE Fig. 1 (RTX 4090, FP8, 45% resident): llama.cpp "Static Offload" 30.9, KTransformers 34.2, SeqMoE 104.1, full load 122.7 tok/s (prior notes) — [arXiv 2609.12978](https://arxiv.org/abs/2609.12978)
  - llama-moe-cache fork (RTX 4070 Ti 12 GB): 33.74 → 64.45 t/s against "vanilla" (prior notes) — [llama-moe-cache](https://github.com/ongunm/llama-moe-cache)
  - Budgeting Bytes quotes llama.cpp `--n-cpu-moe` running "Qwen3-30B-A3B at ∼30 tok/s on 6 GB VRAM + 32 GB RAM" (prior notes) — [arXiv 2609.04238](https://arxiv.org/abs/2609.04238)
  - FlashMoE (RTX 5070 Ti, SSD tier): "2.5× over llama.cpp" in total latency, including load and prefill. (text) — [arXiv 2601.17063](https://arxiv.org/abs/2601.17063)
- No peer-reviewed or arXiv paper reports gpt-oss-120b on a GeForce RTX 5090 with llama.cpp `--n-cpu-moe` at a stated equal expert budget. ATSInfer (RTX 4090, gpt-oss-120b) gives llama.cpp numbers only in figures, with flags unstated. (text) — [arXiv 2607.10183](https://arxiv.org/abs/2607.10183)

### Inferences
- **PR #29887 matters for (b) in two ways.**
  - It makes "a llama.cpp GPU expert cache" a moving target that maintainers now own: an LRU, shared across same-layout layers, with mandatory admission and fetch-every-miss.
  - The paper's cache differs on exactly the axes its own study says matter: decayed frequency with bypass, misses served on CPU or PCIe per a per-machine table, per-layer budget.
  - Cite it as concurrent work. If time allows, add it as an arm, since it is a cheap race entrant built on the same engine and its policy is the paper's LRU-online baseline in-engine. This is a suggestion.
- **Community anchors for the 9950X-class RTX 5090 host.** The 4090 + `--n-cpu-moe 28` figure (≈22 tok/s) and the 5090 + `-ncmoe 25` figure (≈36 tok/s) bracket the static baseline the paper's cache must beat.
- **The 127 tok/s community FreeToken number** (at about 29 GB VRAM) is a reminder that the 11-of-12 FreeToken comparison must state equal expert memory and FreeToken version and mode.

### Gaps
- PR #29887's open/merged state, discussion thread and any benchmark numbers were not retrievable: the GitHub UI and API returned 403, and the commit message has no numbers.
- The QVAC-fabric source MoE cache (date, numbers) was not located.
- Hardware for the #18934 gpt-oss-120b numbers was not re-checked.
- No first-party or credible published tok/s for Qwen3-30B-A3B with `--n-cpu-moe` on an RTX 5090 + desktop CPU was found this pass.
- r/LocalLLaMA was not searchable (no Reddit fetch).

## Q4. Measurement and benchmarking papers on MoE inference that set evaluation standards

### Takeaway
The evaluation-standard literature in the window is:
- MoE-CAP (NeurIPS'25 D&B: sparsity-aware bandwidth/FLOP utilisation metrics);
- 2608.07911 (a reporting contract for expert-cache studies: replay semantics, workload contamination, union-to-capacity ratio);
- SpecMD (a standardised cache-policy benchmarking framework);
- Paging the Experts (measurement-scoping language);
- general LLM-serving evaluation anti-patterns (2507.09019);
- dense batch-1 "fraction of the memory floor" studies (2605.30571, LIMINAL).

None evaluates offload systems against a policy-independent seconds bound. That is the slot the paper's (a) plus "% of the limit" reporting fills. These should be cited as the standards the paper's reporting follows or extends.

### Cited Findings
- **MoE-CAP** (arXiv 2412.07067, v6 19 Nov 2025; duplicate 2505.11415; **NeurIPS 2025 Datasets & Benchmarks**). (abs, text) — [arXiv 2412.07067](https://arxiv.org/abs/2412.07067); [arXiv 2505.11415](https://arxiv.org/abs/2505.11415)
  - Introduces "Sparse Memory Bandwidth Utilization (S-MBU) and Sparse Model FLOPS Utilization (S-MFU)".
  - Finds that MoE systems "typically optimize two of the three dimensions" (cost, accuracy, performance).
  - It covers offloading designs (MoE-Infinity, Fiddler) in its taxonomy.
  - It reports utilisation, not a policy bound.
- **2608.07911**: "the per-step expert union relative to per-layer capacity must be reported". (abs; audit detail from prior notes) — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)
  - Replay semantics: "an inconsistent per-access replay inflates recency-based policies by 27-29% … inverting the policy ranking".
  - Contamination: "one instruction template per category produce[s] verbatim-identical generation prefixes".
  - An audit of ten papers found "none … directly reports the measured per-step/per-layer expert union divided by usable per-layer capacity".
- **SpecMD** (Apple, 2602.03921): "a standardized framework for benchmarking ad-hoc cache policies on various hardware configurations". (abs) — [arXiv 2602.03921](https://arxiv.org/abs/2602.03921)
- **Paging the Experts** (2609.29032): "These results establish bounded feasibility and identify limitations that a deployment claim must not hide". (abs) — [arXiv 2609.29032](https://arxiv.org/abs/2609.29032)
- **"On Evaluating Performance of LLM Inference Serving Systems"** (Agrawal et al., 2507.09019, 11 Jul 2025): catalogues evaluation anti-patterns, e.g. conflating engineering effort with algorithmic novelty, and untuned baselines (prior notes). — [arXiv 2507.09019](https://arxiv.org/abs/2507.09019)
- **"Memory-Bound but Not Bandwidth-Limited"** (2605.30571, 28 May 2026; dense 7–8B models, batch 1): "an L4 reaches roughly 81% of its analytic memory floor, while an H100 reaches only 27%". It uses a 10-session bootstrap CI for CUDA-graph effects. This is a methodological analogue for "% of a speed limit" in dense batch-1 decode. (text) — [arXiv 2605.30571](https://arxiv.org/abs/2605.30571)
- **LIMINAL** (2507.14397, v2 13 Nov 2025): an analytical decode-limit model with "mean absolute error of 7.6%". Datacenter and future hardware. (text) — [arXiv 2507.14397](https://arxiv.org/abs/2507.14397)
- **Earlier related items** (prior notes):
  - The KV-cache SoK (2609.30854): "a protocol that strictly separates derived and reported claims" — [arXiv 2609.30854](https://arxiv.org/abs/2609.30854)
  - Qiu et al. (2609.14864): GGUF-metadata prediction of llama.cpp throughput; leave-one-host-out MAPE 11.6–36.0% — [arXiv 2609.14864](https://arxiv.org/abs/2609.14864)
- **MoE-Lens** (2504.09345; pre-window): "theoretical performance upper bound" with "94% accuracy" (throughput regime). (prior notes) — [arXiv 2504.09345](https://arxiv.org/abs/2504.09345)
- **2505.16056 (ICLR'26)**: SRP and SCH as model-level metrics of offloading suitability, proposed for comparing models. (abs) — [arXiv 2505.16056](https://arxiv.org/abs/2505.16056)

### Inferences
- **Reporting checklist.** The paper's reporting should explicitly cover each standard:
  - MoE-CAP's S-MBU (report each system's achieved bandwidth utilisation alongside "% of the speed limit");
  - 2608.07911's union-to-capacity ratio and replay contract (state the trace provenance and C/k for every cell);
  - 2505.16056's ρ = C/k axis (use the same symbol, to make the W50 law directly comparable).

  Doing so pre-empts the most likely reviewer requests.
- **Unclaimed niche.** No measurement paper in the window reports offload systems as a fraction of a seconds-domain, policy-independent limit. The closest are utilisation (MoE-CAP; OSDI'26's "50% of nominal DDR5") and dense memory-floor fractions (2605.30571).

### Gaps
- MoE-CAP's leaderboard contents (whether any offloading engine is benchmarked on consumer GPUs) were not checked.
- MLSys'26 "Demystifying the Mixture of Experts Serving Tax" (listed in prior notes) was not located on arXiv this pass. Its metrics are unknown.
