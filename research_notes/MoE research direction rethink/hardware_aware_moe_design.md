# Hardware-aware / consumer-memory-aware MoE architecture design and routing locality (prior art and open gaps, as of 27 Sept 2026)

## Cache-/offload-friendly routing and training: what is published, and what is already taken?

### Takeaway
Nearly every cheap version of the idea has been published. Pre-gating and early routing for prefetch appeared in 2024 (Pre-gated MoE at ISCA'24, Read-ME at NeurIPS'24) and then in a shipped model, SmallThinker (Jul 2025). Training-free cache-aware routing came in Dec 2024 (Qualcomm). A locality-aware routing architecture appeared at ICML'25 (Oracle-MoE). Router-only fine-tuning for expert reuse appeared in May 2026 (ReMoE), and a temporal-consistency ("sticky") loss during training in Jun/Jul 2026 (StickyMoE). Three things remain open: whether these methods work on modern fine-grained models (Qwen3-30B-A3B, gpt-oss, OLMoE) at realistic scale, a disputed result between ReMoE and StickyMoE, and cross-layer consistency.

### Cited Findings
**Pre-gating / early routing (enables prefetch that overlaps compute)**
- Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference (Hwang et al., ISCA 2024, arXiv 2308.12066, Aug 2023). The gate in one block pre-selects experts for the next block so that expert loading can overlap with compute. — [arXiv](https://arxiv.org/abs/2308.12066); [Microsoft Research page](https://www.microsoft.com/en-us/research/publication/pre-gated-moe-an-algorithm-system-co-design-for-fast-and-scalable-mixture-of-expert-inference/)
- Read-ME (Cai, Ro, Kim, Wang, Ehteshami Bejnordi, Akella, Wang; NeurIPS 2024; arXiv 2410.19123, Oct 2024):
  - Converts a pre-trained dense LLM into a smaller MoE by using activation sparsity to extract experts.
  - Uses a "pre-gating router decoupled from the MoE backbone", which enables "pre-computing and lookahead scheduling" for batching and cache use.
  - Reports up to +10.1% MMLU and up to 6.1% lower end-to-end latency than dense models of similar scale. — [arXiv abs](https://arxiv.org/abs/2410.19123); [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d298cf34e4539f9134db7f38b42f69fe-Abstract-Conference.html); [GitHub](https://github.com/VITA-Group/READ-ME)
- SmallThinker (IPADS SJTU + Zenergize AI, arXiv 2507.20984, Jul 2025):
  - Places the router before the attention block to create "a sufficient time window for prefetching the required expert parameters" from SSD in parallel with attention.
  - This is a shipped model family, not a systems patch (details in the consumer-design section below). — [arXiv](https://arxiv.org/html/2507.20984v2)

**Training-free cache-aware routing at inference**
- Mixture of Cache-Conditional Experts (Skliar, van Rozendaal, Lepert, Boinovski, van Baalen, Nagel, Whatmough, Ehteshami Bejnordi; Qualcomm AI Research / Contextual AI; arXiv 2412.00099, Dec 2024):
  - Method: training-free manipulation of router logits (max-rank, cumulative-probability threshold, cache-prior reranking) that promotes experts already resident in DRAM.
  - Results: more than 50% fewer cache misses and about 2x speedups on mobile hardware with 12 and 16 GB DRAM, at a cost of 0.1–3% higher perplexity and under 0.1% downstream accuracy loss.
  - Models evaluated: Qwen1.5-MoE-A2.7B, DeepSeek-V2-Lite, Phi-3.5-MoE, Mixtral-8x7B.
  - Observation: standard MoEs "lack temporal consistency in expert selection". — [arXiv HTML v2](https://arxiv.org/html/2412.00099v2)

**Locality-preserving routing architecture (requires training)**
- Oracle-MoE (Zhou et al., ICML 2025, PMLR v267):
  - Diagnosis: "temporal inconsistencies of inter-token expert activations" cause "overly frequent expert swapping" that dominates on-device latency.
  - Method: routes tokens in a compact "oracle space" derived from attention scores to preserve semantic locality across consecutive tokens, with a theoretical locality guarantee.
  - Scale: validated only on GPT-2 architectures of 200M, 350M, 790M and 2B parameters. — [PMLR](https://proceedings.mlr.press/v267/zhou25b.html); [OpenReview](https://openreview.net/forum?id=wn6WHREK9k)

**Router-only fine-tuning for expert reuse (post-hoc)**
- ReMoE: Boosting Expert Reuse through Router Fine-Tuning in Memory-Constrained MoE LLM Inference (Zhu, Liao, Jiang, Zhang, Wang, Xiao; Beihang University + Huawei; arXiv 2605.27081, 26 May 2026). A paper-notes aggregator lists it under ICML 2026; I did not verify this on icml.cc.
  - Method: fine-tunes only the gate parameters with three losses: a Trust-KL anchor to the frozen router, a reuse-mass loss, and a composite locality loss (smoothness, lag suppression, working-set compaction). Load-balancing aux loss is set to 0 because "it conflicts with cache locality goals".
  - Models: DeepSeek-V2-Lite and Qwen1.5-MoE-A2.7B only.
  - Results:
    - Expert reuse +26.4% and +27.2% on the two models.
    - +8.4% throughput under vLLM GPU–CPU offloading.
    - 43.6–49.8% lower TPOT on Jetson Orin NX with SSD offload.
    - 1.77–1.99x decode speedup.
  - Stated scope and limitations: B=1 only; mainly adjacent-step locality; "no explicit ablation comparing impact across varying E or K". — [arXiv HTML](https://arxiv.org/html/2605.27081v1); [paper note (ICML2026 listing)](https://en.papernotes.org/ICML2026/llm_efficiency/remoe_boosting_expert_reuse_through_router_fine-tuning_in_memory-constrained_moe/)

**Temporal-consistency ("sticky") loss during training**
- Sticky Routing: Training MoE Models for Memory-Efficient Inference (StickyMoE; Ali Kayyam, BrainChip Inc.; arXiv 2607.08780, Jul 2026, paper dated 12 Jun 2026):
  - Loss: adds L_cons = mean of ||g_t − g_{t−1}||² over consecutive softmax gate vectors, plus a "Soft-Hard" segment-anchor variant. Total loss is L = L_CE + λ·L_cons + μ·L_bal. No architecture change.
  - Results: up to 59% lower switch rate and up to 3.92x fewer cache misses, with utilisation entropy ≥1.92 of 2.0 bits.
  - Scale: toy only, small and medium MoEs trained on WikiText-2.
  - Claims:
    - The Soft-Hard variant "Pareto-dominates post-hoc fine-tuning".
    - In their reproduction, ReMoE "fails to reduce switch rate at either scale".
  - Stated future work: cross-layer routing consistency (Appendix D). — [arXiv HTML](https://arxiv.org/html/2607.08780v1); [arXiv abs](https://arxiv.org/abs/2607.08780)
- **Conflict:** ReMoE reports +26–27% reuse on real 2.7B–16B-class checkpoints ([ReMoE](https://arxiv.org/html/2605.27081v1)), while StickyMoE reports that ReMoE fails at WikiText-2 toy scale ([StickyMoE](https://arxiv.org/html/2607.08780v1)). The two setups are not comparable (different scale, data, and metric: switch rate vs reuse).

**Segment-level routing (the limiting case of stickiness)**
- Lory (Zhong, Xia, Chen, Lewis; Princeton & Meta; arXiv 2405.03133, May 2024). A fully differentiable MoE for autoregressive pretraining that uses causal segment-level routing and expert merging. — [arXiv abs](https://arxiv.org/abs/2405.03133v2)

**Offload-native sparse parameters (prefetchable by construction)**
- Mixture of Lookup Experts (MoLE; ICML 2025 oral; arXiv 2503.15798, Mar 2025):
  - Experts are FFNs during training and are re-parameterized into lookup tables indexed by input ids, which are offloaded to storage.
  - Claims inference speed comparable to dense models and much faster than MoE with experts offloaded, with quality on par with MoE. — [arXiv abs](https://arxiv.org/abs/2503.15798); [ICML oral](https://icml.cc/virtual/2025/oral/47174)
- Engram / "Conditional Memory via Scalable Lookup" (DeepSeek; arXiv 2601.07372, 12 Jan 2026, v2 12 Jul 2026):
  - N-gram-style O(1) lookup scaled to 27B parameters.
  - Claims "deterministic addressing enables runtime prefetching from host memory, incurring negligible overhead". — [arXiv abs](https://arxiv.org/abs/2601.07372); [GitHub](https://github.com/deepseek-ai/Engram)

**The systems side is crowded**
- Examples:
  - ExpertFlow (arXiv 2410.17954): [arXiv](https://arxiv.org/html/2410.17954v1)
  - SMoE, expert substitution for edge (arXiv 2508.18983): [arXiv](https://arxiv.org/html/2508.18983)
  - MoE-SpeQ, speculative decoding with proactive expert prefetching (arXiv 2511.14102): [arXiv](https://arxiv.org/html/2511.14102)
  - FineMoE, fine-grained expert offloading (EuroSys'26): [PDF](https://intellisys.haow.us/assets/pdf/Hanfei_FineMoE_EuroSys26.pdf)
  - High-bandwidth-flash MoE inference (arXiv 2608.14333): [arXiv](https://arxiv.org/html/2608.14333)
- I read titles only for these; their contents were not verified.

### Inferences
- **Taken:**
  - Pre-gating and pre-attention routing: Pre-gated MoE, Read-ME, SmallThinker.
  - Training-free cache-biased routing: Cache-Conditional Experts.
  - Router-only locality fine-tuning: ReMoE.
  - Sticky loss in pretraining at toy scale: StickyMoE.
  - Attention-space locality routing at GPT-2 scale: Oracle-MoE.
  - A paper whose headline is "router fine-tuning improves expert reuse" is no longer novel after May 2026.
- **Open, and cheap:**
  1. Apply ReMoE-style router-only fine-tuning to fine-grained modern models that ReMoE did not test: Qwen3-30B-A3B (128 experts, top-8), gpt-oss-20b, OLMoE-1B-7B. The researcher already has exact traces for these.
  2. Settle the ReMoE-vs-StickyMoE disagreement with a controlled comparison.
  3. Report gains as predicted and measured decode speed against a speed-of-light bound, rather than as hit rate. None of the papers above report locality gains against a validated bytes/bandwidth bound.
- **Open, and more expensive:**
  - Cross-layer consistency. StickyMoE lists it as future work, and it is exactly what enables whole-token prefetch.
  - Sticky or segment losses at modern fine-grained configurations (e.g. 64–128 experts, top-4/8, with or without a shared expert), trained for more than toy token counts.
- The researcher measured ~46–50% same-expert repeat on consecutive tokens. ReMoE reports a +26–27% relative reuse improvement, so fine-tuning seems to have meaningful headroom above the natural baseline. However, the metric definitions (IR_t / EOR vs "fraction of selections repeating") must be aligned before comparing.

### Gaps
- I could not verify that Cache-Conditional Experts was published in TMLR 2025. The fetched arXiv v2 did not show a venue.
- I did not fetch or verify SiDA-MoE (MLSys 2024).
- ReMoE's fine-tuning cost (tokens, GPU-hours) did not appear in the extracted text.
- StickyMoE's exact model sizes were not reported in the extract. Its code is said to be on GitHub, but I did not verify the link.
- I could not fetch "Training-Free Halving of Activated Experts in Fine-Grained MoE" (arXiv 2609.04575, Sep 2026); it may be relevant to reducing k without training.

## Scaling laws and empirical studies: how do expert locality, reuse and popularity change with E, k, granularity, shared experts and training?

### Takeaway
There is one large empirical study, Liang et al. (ICLR 2026), covering 20 MoEs. It found that local routing consistency is hurt by shared experts, by top-k below 4, and by MoE layers placed only every few layers. It is helped by larger expert-combination spaces and by domain-specialized experts. I found **no scaling law for locality**: nothing that models reuse or miss rate as a function of E, k, granularity and training tokens. Meanwhile the on-device scaling-law papers recommend shared experts and fine granularity for loss. This is an unresolved tension between loss-optimal and offload-optimal design.

### Cited Findings
- "Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models" (Liang, Wang, Tian, Li, Tang, Wei; Fudan, USC, Huawei; arXiv 2505.16056, May 2025; listed as ICLR 2026 on ML Anthology):
  - **Metrics:** SRP (segment routing best performance: how well a fixed per-segment expert set approximates the router) and SCH (segment cache best hit rate).
  - **Models:** 20 MoEs from 3B to 57B, including OLMoE, Qwen3, Qwen1.5-MoE, Qwen2, DeepSeekMoE, DeepSeek-V2-Lite, Mixtral-8x7B, Phi-3.5-MoE, GRIN-MoE, Jamba-Mini, JetMoE, MiniCPM-MoE, PowerMoE, LLaMA-MoE v1/v2, Yuan2.0, XVERSE, OpenMoE, NLLB-MoE and SwitchTransformers.
  - **Findings:**
    - Shared experts "significantly harm consistency". Among toy models, Share1 and Share2 show much lower SRP.
    - Top-k below 4 and MoE placed every 2–6 layers correlate with lower consistency.
    - A larger expert-combination space correlates with higher consistency.
    - Domain-specialized experts predict consistency; vocabulary specialization does not, except in later layers.
    - Post-training (SFT/RL) shows "no significant difference".
    - A cache of about 2x the number of active experts is the knee.
    - SCH correlates above 0.90 with LRU/LFU hit rates.
    - "High local routing consistency almost always means low local load balance", but Qwen3 and GRIN-MoE achieve both through domain-specialized experts.
  - **Limitations:** decode phase only; 512-token segments; no systematic study of the load-balance-objective trade-off. — [arXiv v3](https://arxiv.org/html/2505.16056v3); [ML Anthology ICLR 2026](https://mlanthology.org/iclr/2026/liang2026iclr-all/)
- Cache-Conditional Experts:
  - For Mixtral, the router's top-2 "yield the best performance only 28% of the time".
  - In granular architectures, swapping the 3rd-ranked expert and lower costs little.
  - Together these mean routing has slack that can be traded for locality. — [arXiv](https://arxiv.org/html/2412.00099v2)
- SmallThinker uses a "DP-Groups" load-balance loss to promote "expert specialization" and "predictable activation patterns". 70–80% of experts have activation frequency below 0.14. — [arXiv](https://arxiv.org/html/2507.20984v2)
- ReMoE turns the load-balancing aux loss off during locality fine-tuning because it conflicts with cache locality. — [arXiv](https://arxiv.org/html/2605.27081v1)
- StickyMoE frames switch rate as governed by router temporality, "independent of expert count N or k". This is their framing, not a measured sweep. — [arXiv](https://arxiv.org/html/2607.08780v1)
- MobileMoE (Meta, arXiv 2605.27358, 27 May 2026):
  - Sweeps E ∈ {1, 2, 4, 8, 16, 32}, granularity g ∈ {1, 2, 4, 8, 16} and shared expert on/off.
  - Loss findings: E ∈ {4, 8} is "the practical sweet spot in the on-device memory regime"; g>1 helps, with diminishing returns beyond g=8; shared experts reduce loss.
  - Final config: 8 routed experts split 8-way into 64 fine-grained experts, top-4, plus 1 shared expert. — [arXiv PDF](https://arxiv.org/pdf/2605.27358)
- Joint MoE Scaling Laws: "Mixture of Experts Can Be Memory Efficient" (Ludziejewski et al., ICML 2025; arXiv 2502.05172):
  - Joint law over active params, data and number of experts, fitted on more than 280 runs up to 2.7B active and 5B total parameters.
  - Concludes MoE can be more memory-efficient than dense and gives a framework for the optimal config under fixed memory and compute.
  - The extract did not mention decode bandwidth or offloading. — [arXiv](https://arxiv.org/abs/2502.05172); [PMLR](https://proceedings.mlr.press/v267/ludziejewski25a.html)
- Hardware co-design roofline scaling laws (arXiv 2602.10377) find "wider models require sparser MoE (ρ* ∝ d^−1.19)", where ρ = K/E. — [arXiv](https://arxiv.org/html/2602.10377v1)

### Inferences
- The loss-optimal recipe for small on-device MoEs (MobileMoE: fine-grained plus a shared expert, top-4) partly contradicts the locality-optimal recipe (Liang: no shared expert, larger combination space, MoE on every layer, k ≥ 4).
- No paper resolves that tension with a single objective such as "decode tokens/s at iso-quality on platform P when weights exceed fast memory". A bytes/bandwidth decode model that takes the miss rate as an input is the natural arbiter, and this is the researcher's comparative advantage.
- The researcher's traces overlap heavily with Liang et al. (OLMoE and Qwen3 are in their list).
  - gpt-oss-20b is not in the extracted model list and may be uncovered.
  - Cross-domain popularity measured on exact traces may add something.
  - Characterization alone is not a novel contribution after ICLR 2026. It needs the decode-speed and speed-of-light layer on top.
- A "locality scaling law", i.e. reuse or SCH as a function of (E, k, g, shared, tokens trained), appears unpublished. It needs trained model families, not just public checkpoints, because public models vary many factors at once.

### Gaps
- I could not confirm whether Liang et al.'s v3 added gpt-oss.
- I found no study that tracks locality over training checkpoints. OLMoE's intermediate checkpoints would allow this, but I did not verify that anyone has done it.
- I found no study of how QAT/quantization or MTP/speculative decoding change locality, beyond the titles of systems papers.

## Hardware-aware model design for consumer devices: is SmallThinker (or similar) already doing "hardware-constrained MoE design with pre-attention routing and locality"?

### Takeaway
Largely yes, for the "design a MoE for a local memory budget" framing:
- SmallThinker (Jul 2025) co-designs a pre-attention router for SSD prefetch with ReGLU sparsity and hybrid attention, for 1 GB and 8 GB budgets.
- LFM2 (Nov 2025) uses hardware-in-the-loop search on phone and laptop CPUs and ships an 8B-A1.5B MoE.
- MobileMoE (Meta, May 2026) fits a DRAM-aware MoE scaling law and validates on phones.

None of them uses an analytical offload-aware decode model to pick E/k/g, and none optimizes temporal locality with a training objective. SmallThinker's load-balancing loss targets specialization and predictability.

### Cited Findings
- SmallThinker (IPADS/SJTU + Zenergize AI; arXiv 2507.20984, Jul 2025):
  - **Configs:**
    - 4B-A0.6B: 32 experts, top-4, 32 layers, 2.5T tokens.
    - 21B-A3B: 64 experts, 52 layers, 7.2T tokens.
    - Expert hidden size is 768 in both.
  - **Memory budgets:** 1 GB (4B) and 8 GB (21B).
  - **Pre-attention router** for SSD prefetch overlapped with attention.
  - **ReGLU** adds about 60% extra sparsity inside experts.
  - **NoPE–RoPE 1:3 hybrid** with 4096-token sliding window.
  - **Decode speed:**
    - 108 tok/s (4B) and 30 tok/s (21B) on an i9-14900K.
    - 20.3 tok/s for the 21B under an 8 GB memory limit.
  - The extract mentions no explicit hardware cost model for choosing hyperparameters; choices were driven by "constraints into design principles".
  - Stated limitations: smaller corpus, SFT only. — [arXiv HTML](https://arxiv.org/html/2507.20984v2); [HF model card](https://huggingface.co/Tiiny/SmallThinker-4BA0.6B-Instruct)
- LFM2 Technical Report (Liquid AI; arXiv 2511.23404):
  - **Search:** hardware-in-the-loop over local/global mixing blocks (short convolutions, SWA, linear attention, SSMs, GQA) and layout, measured on a Samsung Galaxy S25 (Snapdragon 8 Elite) and an AMD Ryzen HX 370. The metrics were TTFT, ms/token, prefill throughput and peak RSS. MoE hyperparameters were not a search axis per the extract.
  - **LFM2-8B-A1B design:**
    - 8.3B total / 1.5B active.
    - 32 experts, top-4, per-expert FF 1792.
    - First 2 layers dense.
    - Normalized sigmoid router with adaptive biases.
    - Rationale: "compute per token, not weight storage, dominates perceived latency" (i.e. all weights resident).
  - **Decode speed:**
    - 48.6 tok/s at 1K context on the S25 (Q4_0).
    - 74.9 tok/s on the Ryzen HX 370.
  - **Date conflict:** the fetched HTML reported "August 24, 2026", but the arXiv id 2511 implies a Nov 2025 submission.
  - An LFM2.5-8B-A1B follow-up was released around 28 May 2026. — [arXiv HTML](https://arxiv.org/html/2511.23404v1); [Liquid AI blog](https://www.liquid.ai/blog/lfm2-8b-a1b-an-efficient-on-device-mixture-of-experts); [LFM2.5 news](https://www.marktechpost.com/2026/05/28/liquid-ai-releases-lfm2-5-8b-a1b-an-on-device-moe-model-with-8-3b-total-and-1-5b-active-parameters/)
- MobileMoE: Scaling On-Device Mixture of Experts (Chen, Huang, Chang, Szwejbka, Desai, Liu, Chandra, Krishnamoorthi; Meta AI; arXiv 2605.27358, 27 May 2026):
  - **Scaling law:** a generalized on-device law L(N_act, D, E, architecture choices) with a memory constraint M = weights(INT4) + KV cache ≤ budget (≤5 GB). Models compress to 0.68–2.75 GB.
  - **Validation:** Galaxy S25 and iPhone 16 Pro via ExecuTorch/XNNPACK. Decode is 2.2–3.4x faster and prefill 1.8–3.8x faster than MobileLLM-Pro.
  - **Training scale:** about 6T pretraining tokens on 8×8 H100 for 3–4 weeks. Sizes are 272M/1.3B, 528M/2.8B and 922M/5.3B (active/total).
  - **Offloading and locality:** "Not addressed". All weights are assumed resident; there is no prefetch or locality-aware caching.
  - Future work includes "dynamic routing". — [arXiv PDF](https://arxiv.org/pdf/2605.27358)
- Apple Foundation Models 2025 (9 Jun 2025):
  - The on-device model has about 3B parameters, split into two blocks with a 5:3 depth ratio. Block-2 KV is shared from block 1, giving 37.5% less KV memory.
  - The server model uses Parallel-Track MoE (PT-MoE), which synchronizes only at track-block boundaries (87.5% fewer syncs at D=4).
  - There is no on-device MoE. — [Apple ML Research](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates); [arXiv 2507.13575](https://arxiv.org/abs/2507.13575)
- MoLE (ICML 2025) and Engram (DeepSeek, Jan 2026) take a different route to consumer-memory-friendly sparsity: parameters addressed by token id, which makes prefetch or storage offload trivial by construction. — [MoLE](https://arxiv.org/abs/2503.15798); [Engram](https://arxiv.org/abs/2601.07372)

### Inferences
- **Blunt verdict:** "hardware-constrained MoE design for local deployment with pre-attention routing" is SmallThinker's contribution (Jul 2025). "DRAM-constrained MoE scaling law validated on phones" is MobileMoE's (May 2026). "Hardware-in-the-loop on-device architecture search" is LFM2's (Nov 2025). A proposal framed at that level would be seen as derivative.
- **The remaining white space** is the regime between these works: total weights exceed fast memory, so experts stream over PCIe, UMA-to-SSD or DRAM-to-SSD, and decode speed depends on miss rate, which depends on locality. This regime covers:
  - consumer dGPUs with 8–24 GB VRAM and host DRAM;
  - laptops and phones running models larger than DRAM.

  MobileMoE and LFM2 explicitly assume residency. SmallThinker operates in this regime but chose hyperparameters without a model.

### Gaps
- I did not fetch the PowerInfer-2, MobileLLM(-R1/-Pro) or Gemma 3n (per-layer-embedding offload) reports in this pass. I cannot confirm whether any of them uses routing-locality objectives. Given the SmallThinker lineage (same IPADS group as PowerInfer), PowerInfer-2 is likely a systems paper, but this is unverified.
- I could not see whether SmallThinker's full paper reports measured temporal reuse or prefetch-accuracy numbers.

## Does anyone use a validated decode-time model to pick MoE hyperparameters for a target consumer platform, and validate with real training? What is the cheapest credible experiment?

### Takeaway
Yes, but only for the all-resident regime:
- "Hardware Co-Design Scaling Laws via Roofline Modelling for On-Device LLMs" (Feb 2026) couples a roofline latency model with a loss scaling law that includes MoE E and K/E. It trains 170 models for 10B tokens each and targets Jetson Orin.
- MobileMoE (May 2026) does DRAM-constrained selection validated on phones.

**Open:** nobody (that I found) includes a locality- or miss-rate-dependent offload bandwidth term, or a speed-of-light bound for expert placement, in the co-design loop. This is the most defensible gap for the researcher's assets.

### Cited Findings
- Hardware Co-Design Scaling Laws via Roofline Modelling for On-Device LLMs (Sun, Jiang, Ding et al.; Li Auto, CAS, UCL; arXiv 2602.10377, 10 Feb 2026):
  - **Method:** a loss scaling law plus roofline latency, used to find Pareto-optimal architectures.
  - **Scale:** 1,942 candidate architectures profiled; 170 models trained at 10B tokens each (R² 0.975 train, 0.952 validation); target NVIDIA Jetson Orin.
  - **Findings:**
    - "MoE configurations constitute 100% of Pareto-optimal designs under on-device batch-one inference".
    - Optimal designs are wide and shallow.
    - l* ∝ d^−2.
    - ρ* ∝ d^−1.19.
    - 19.42% lower perplexity than Qwen2.5-0.5B at matched latency.
  - **Roofline error:** "10–20% deviations".
  - **Limitations:** they call for future work on "KV-cache locality, and routing overhead in MoE systems". The work excludes SSM and linear attention. — [arXiv HTML](https://arxiv.org/html/2602.10377v1)
- MobileMoE's latency model is qualitative: bandwidth-bound decode that "scales with active parameters". It includes no offload term. — [arXiv PDF](https://arxiv.org/pdf/2605.27358)
- Joint MoE Scaling Laws gives optimal configs under memory and compute budgets from more than 280 runs, but per the extract it does not model decode bandwidth. — [arXiv](https://arxiv.org/abs/2502.05172)
- LFM2 uses measured on-device latency (hardware-in-the-loop) rather than an analytical model, and does not search MoE hyperparameters. — [arXiv HTML](https://arxiv.org/html/2511.23404v1)
- StickyMoE uses only a toy latency expression, τ_t = τ_compute + 1[miss]·τ_load, with no validation against hardware. — [arXiv HTML](https://arxiv.org/html/2607.08780v1)
- ReMoE gives Proposition 3.1, which bounds fetches by the expert overlap ratio under recency caching. That is a cache model, not a platform decode model. — [arXiv HTML](https://arxiv.org/html/2605.27081v1)

### Inferences
- **Cheapest credible experiment** (training-free, about 3–5 weeks):
  1. Take about 10–20 public MoEs spanning E, k, granularity and shared-expert choices (OLMoE, Qwen3-30B-A3B, gpt-oss-20b, DeepSeek-V2-Lite, Qwen1.5-MoE, Mixtral, Phi-3.5-MoE, GRIN, and SmallThinker/LFM2-8B-A1B as on-device-designed controls).
  2. Collect or reuse routing traces and compute reuse and SCH-style metrics.
  3. Feed architecture plus measured locality into the validated bytes/bandwidth decode model and speed-of-light bound to predict offloaded decode tok/s on 2–3 consumer platforms (e.g. a 12–16 GB consumer GPU with PCIe 4.0 host DRAM; an Apple-class UMA; the A10).
  4. Validate on a subset with a real offloading runtime.

  Headline claim: "architecture hyperparameters, plus a locality term, predict offloaded decode speed within X%; architectures optimized for resident-weight on-device regimes (MobileMoE/LFM2 style) rank differently from those optimized for offloaded regimes".

  This extends Liang et al. (hit rates only) and the roofline co-design paper (no locality), and it uses assets the researcher already has.
- **Next-cheapest** (small-scale training, Dec 2026):
  - Train a small grid of MoEs (e.g. ≤0.5B active) that varies E/g/shared expert and a sticky-loss λ at fixed tokens.
  - Fit loss(E, g, shared, λ) and reuse(E, g, shared, λ).
  - Combine them with the decode model to produce an offload-aware Pareto frontier for a named consumer platform.
  - The roofline co-design paper used 170 × 10B-token runs, so a credible student-scale version must be much smaller, with far fewer runs and tokens. The claim should therefore be about trends and rank-ordering, not about a fitted law.
- **Router-only fine-tuning** on OLMoE-1B-7B is the cheapest training intervention for testing whether an architecture's locality headroom predicts fine-tuning gains. It is feasible on a single rented GPU. This is an inference: ReMoE did not report its cost.

### Gaps
- I did not see the roofline co-design paper's appendix-level latency error on Jetson for MoE configs specifically, only the general 10–20% statement.
- I found no public work that validates an offloaded-MoE decode model across many architectures on consumer GPUs. This absence is itself the gap, but I did not do an exhaustive search of MLSys/ISCA 2026 proceedings.
- Compute cost estimates for the small-scale training grid are my inference; no source gives them.

## Per sub-idea: closest prior art, what is open, feasibility by 27 Oct 2026 and by Dec 2026, likely venue, and fit with SPCL taste

### Takeaway
The analysis- and model-driven "offload-aware co-design" framing is the only sub-idea that is both still open and matched to the researcher's assets and to SPCL's data-movement and performance-modeling taste. The following are taken or crowded:
- Router tweaks: ReMoE, StickyMoE, Cache-Conditional.
- Pre-routing: Pre-gated, Read-ME, SmallThinker.
- Characterization: Liang et al.
- On-device scaling laws: MobileMoE, roofline co-design.

### Cited Findings
- **SPCL context:**
  - "Data Movement Is All You Need: A Case Study on Optimizing Transformers" (Ivanov, Dryden, Ben-Nun, Li, Hoefler; MLSys 2021; arXiv 2007.00072) is a canonical SPCL data-movement-centric analysis of transformers. — [arXiv](https://arxiv.org/abs/2007.00072); [SPCL PDF](https://htor.inf.ethz.ch/publications/img/data_movement_is_all_you_need.pdf)
  - "Spatial Mixture-of-Experts" (Dryden & Hoefler, NeurIPS 2022) is SPCL's own MoE work. — [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/4c5e2bcbf21bdf40d75fddad0bd43dc9-Paper-Conference.pdf); [SPCL publications](https://spcl.inf.ethz.ch/Publications/)
- **Sub-idea A: router-only fine-tuning for locality.**
  - Prior art: ReMoE (May 2026), Cache-Conditional Experts (Dec 2024, training-free).
  - Open: fine-grained modern models (Qwen3, gpt-oss, OLMoE); reconciling ReMoE's gains with StickyMoE's report that ReMoE fails. — [ReMoE](https://arxiv.org/html/2605.27081v1); [StickyMoE](https://arxiv.org/html/2607.08780v1); [Cache-Conditional](https://arxiv.org/html/2412.00099v2)
- **Sub-idea B: temporal-consistency loss during training.**
  - Prior art: StickyMoE (Jun/Jul 2026, WikiText-2 toy scale), Oracle-MoE (ICML 2025, GPT-2 200M–2B), Lory (2024, segment routing).
  - Open: modern fine-grained configurations; cross-layer consistency (StickyMoE's own future work). — [StickyMoE](https://arxiv.org/html/2607.08780v1); [Oracle-MoE](https://proceedings.mlr.press/v267/zhou25b.html); [Lory](https://arxiv.org/abs/2405.03133v2)
- **Sub-idea C: pre-attention / early routing for prefetch.**
  - Prior art: Pre-gated MoE (ISCA 2024), Read-ME (NeurIPS 2024), SmallThinker (Jul 2025, shipped models). — [Pre-gated](https://arxiv.org/abs/2308.12066); [Read-ME](https://arxiv.org/abs/2410.19123); [SmallThinker](https://arxiv.org/html/2507.20984v2)
- **Sub-idea D: characterization of locality versus architecture.**
  - Prior art: Liang et al. (ICLR 2026, 20 models, SRP/SCH, shared experts hurt, cache ≈2x active). — [arXiv](https://arxiv.org/html/2505.16056v3)
- **Sub-idea E: hardware-constrained MoE co-design / scaling law.**
  - Prior art:
    - Roofline co-design (Feb 2026, Jetson, 170 trained models, MoE included, no locality or offload).
    - MobileMoE (May 2026, phones, DRAM-resident).
    - Joint MoE Scaling Laws (ICML 2025, memory-constrained, no decode).
    - LFM2 (Nov 2025, measured hardware in the loop, no MoE search). — [Roofline co-design](https://arxiv.org/html/2602.10377v1); [MobileMoE](https://arxiv.org/pdf/2605.27358); [Joint MoE](https://arxiv.org/abs/2502.05172); [LFM2](https://arxiv.org/html/2511.23404v1)
- **Sub-idea F: offload-native sparse parameters (lookup experts, memory tables).**
  - Prior art: MoLE (ICML 2025 oral), Engram (DeepSeek, Jan 2026, scaled to 27B). — [MoLE](https://arxiv.org/abs/2503.15798); [Engram](https://arxiv.org/abs/2601.07372)

### Inferences
**A. Router-only locality fine-tuning** (taken as a headline)
- By 27 Oct: feasible to produce a replication/extension on OLMoE-1B-7B, and possibly Qwen3-30B-A3B's router only, measured in predicted and measured tok/s.
- By Dec: add gpt-oss and cross-domain robustness.
- Venue: workshop, or as a section of a larger paper.
- SPCL fit: low to moderate as a standalone paper (incremental ML); acceptable as the "intervention" arm of E.

**B. Sticky or cross-layer consistency loss in pretraining** (partially open)
- By 27 Oct: only toy-scale (≤100M-class) evidence is possible, which would be no better than StickyMoE.
- By Dec: small but real runs (a few hundred M active, 5–20B tokens) that test cross-layer stickiness plus a decode-speed evaluation are plausible on Lambda.
- Venue: ICML/NeurIPS main track needs a larger scale than is feasible, so the realistic targets are MLSys or a workshop.
- SPCL fit: moderate. It is stronger if framed as reducing bytes moved per token toward the speed-of-light bound.

**C. Pre-attention routing** (taken)
- The only open angle is quantitative: how much routing lookahead a given bandwidth ratio needs, derived from the decode model. That is a paragraph or section, not a paper.
- SPCL fit: fine as analysis.

**D. Characterization** (taken as a standalone contribution)
- The researcher's traces become valuable only as the empirical input to E.
- By 27 Oct: straightforward.

**E. Offload-aware co-design: predict offloaded decode speed from (E, k, expert size, shared, layers, locality) with a validated bytes/bandwidth model and speed-of-light bound; show loss-optimal on-device recipes are not offload-optimal** (most open, best fit)
- By 27 Oct: an arXiv preprint is feasible.
  - Contents: the analytical model; about 10–20 public MoEs with measured locality; validation on 2–3 platforms; a "design rules for offloaded consumer MoE" section, including the shared-expert and granularity tension between MobileMoE and Liang et al.
- By Dec: add a small training grid (E, g, shared, sticky-λ) to show the predicted Pareto shift is real.
- Venues: MLSys 2027, ISPASS/IISWC-style characterization venues, or ICML 2027 with the training grid. I did not verify deadlines.
- SPCL fit: high. It matches SPCL's data-movement-first transformer analysis ("Data Movement Is All You Need") and its prior MoE interest (Spatial MoE).

**F. Lookup / offload-native experts** (big-lab territory)
- Not cheap. Mention only as the design frontier that E's model could evaluate: bytes per token for MoLE/Engram-style parameters versus routed experts.

**Biggest novelty risk for E:** a follow-up from the roofline co-design group or from Meta's MobileMoE team adding an offload/locality term. Both papers name routing overhead and dynamic routing as future work, so speed matters.

### Gaps
- I did not verify venue deadlines (MLSys 2027, ICML 2027, NeurIPS 2026 workshops).
- I did not search SPCL's 2025–2026 publication list in depth for any in-house MoE-offload or on-device work that could overlap or be complementary. Only Spatial MoE (2022) and the 2021 data-movement paper were confirmed.
- I did not check whether any NeurIPS 2026 accepted papers (decisions around late Sept 2026) cover offload-aware MoE co-design. That is a residual scoop risk.
