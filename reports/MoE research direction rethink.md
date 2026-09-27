# Build the yardstick, not another cache

Pursue a **time-domain speed-of-light yardstick for offloaded MoE decode**. It has three parts: a trace-driven lower bound on step time for any exact-routing placement policy; the bytes-over-bandwidth model already validated on 52 third-party measurements; and a bound-normalized audit of published MoE-offloading speedups against equal-VRAM baselines. Fold in a **trace-provenance check** (each model's own samples versus dataset text) as the validity section. Of **32 candidate directions** vetted against 2024–2026 prior art, **13 are taken, 10 partially open, 8 open and 1 an unverified anomaly**. Nearly every "next mechanism" an independent researcher would reach for appeared between May and mid-September 2026: next-token expert prediction, draft-driven prefetch, router fine-tuning for locality, SSD expert streaming and closed-form CPU/GPU splits. The yardstick scores highest (**30/35**) because it is the only candidate that is open in its specific form, reuses almost every built asset, implements Hoefler's own benchmarking rule "show upper performance bounds", and can be a finished arXiv preprint before 27 October. Its closest prior art, 2608.07911 (August 2026), stays in block counts and explicitly declines to re-derive published speedups. Bluntly, this is a measurement paper, not a breakthrough. Its value to the application is that it demonstrates exactly the method SPCL publishes, and it becomes the engine for the PhD proposal: **offload-aware MoE architecture co-design**, which nobody has done because every on-device design study assumes all weights stay resident in memory. The runner-up, parallel test-time compute on an offloaded MoE, has a higher ceiling. It needs new data at scale, though, and faces groups that published adjacent work in April and August 2026. The main scoop threat to the recommendation is a time-domain follow-up by 2608.07911's author, who posted four versions in 17 days and names real-hardware validation as the paper's biggest gap. Post v1 by 16–21 October. No venue can accept anything before 27 October, and the fellowship does not require publications.

## Thirteen of thirty-two candidate directions are already taken

Each candidate below was vetted against papers, llama.cpp issues and PRs, and community benchmarks, as of 27 September 2026. "Open" means no directly matching paper or artifact was found. Many September 2026 arXiv papers were read only through abstracts or HTML summaries, so an "open" verdict records absence of evidence on a date, not proof.

| # | Candidate | Verdict | Closest prior art (date) | What is left |
|---|---|---|---|---|
| 1 | GPU expert cache with CPU-executed misses (the previous direction) | **Taken** | HybriMoE, Apr 2025 ([arXiv](https://arxiv.org/abs/2504.05897)); 2512.16473, 18 Dec 2025 ([arXiv](https://arxiv.org/abs/2512.16473)); FreeToken, 17 Aug 2026 ([arXiv](https://arxiv.org/abs/2608.16157)); llama.cpp RFC #24528, 12 Jun 2026 ([discussion](https://github.com/ggml-org/llama.cpp/discussions/24528)) | Nothing as a concept |
| 2 | Closed-form split of missed experts between PCIe fetch and CPU execution | **Taken** | FreeToken's q* ≈ m·B_P/B_H, Aug 2026 ([arXiv](https://arxiv.org/html/2608.16157v1)) | A special case of the bytes model; cite it |
| 3 | Next-token (cross-step) expert prediction | **Taken** | SeqMoE, 11 Sep 2026 ([arXiv](https://arxiv.org/html/2609.12978)); ST-MoE, 13 Jun 2026 ([arXiv](https://arxiv.org/html/2606.15453)) | Scoring an oracle predictor against a bound |
| 4 | Next-layer expert prediction and prefetch | **Taken** | Pre-gated MoE, ISCA'24 ([arXiv](https://arxiv.org/abs/2308.12066)); Fate, Feb 2025 ([arXiv](https://arxiv.org/html/2502.12224v1)); Speculating Experts, 9 Mar 2026 ([arXiv](https://arxiv.org/html/2603.19289)) | — |
| 5 | Count-based Belady gap; trace-level evaluation contract for expert caching | **Taken** | 2608.07911, Aug 2026, v4 25 Aug ([arXiv](https://arxiv.org/abs/2608.07911)) | The time-domain version (row 7) |
| 6 | Draft model prefetches target experts; expert-union cost in verification | **Taken** | SpecMoEOff, Aug 2025 ([arXiv](https://arxiv.org/abs/2508.21706)); SP-MoE, Nov 2025 ([arXiv](https://arxiv.org/html/2510.10302v2)); MoE-SpeQ, 18 Nov 2025 ([arXiv](https://arxiv.org/html/2511.14102v1)); EcoSpec, 14 Jul 2026 ([arXiv](https://arxiv.org/html/2607.12696)); AcceptMoE, Aug 2026 ([arXiv](https://arxiv.org/html/2608.02989)) | — |
| 7 | **Time-domain speed-of-light bound plus bound-normalized cross-paper audit** | **Open (narrow)** | 2608.07911 (block counts only); MoE-CAP's single-peak S-MBU, NeurIPS 2025 ([arXiv](https://arxiv.org/abs/2412.07067)); 2608.18261 (one configuration), 18 Aug 2026 ([arXiv](https://arxiv.org/abs/2608.18261)); MoE-Lens (own system, throughput) ([arXiv](https://arxiv.org/abs/2504.09345)) | Seconds instead of blocks; many papers; equal-VRAM baselines |
| 8 | End-to-end, time-level reporting contract for offloaded decode | **Open** | 2608.07911's checklist excludes transfer, overlap and bandwidth; "Silent Hyperparameter", May 2026, covers accuracy only ([arXiv](https://arxiv.org/abs/2605.19537)); MLPerf Client has no MoE ([MLCommons](https://mlcommons.org/benchmarks/client/)) | The whole contract |
| 9 | Draft or MTP foresight used for eviction and bypass (k-lookahead MIN); MTP under hybrid offload | **Open** | SP-MoE and MoE-SpeQ use drafts for prefetch only | The whole question |
| 10 | I/O-complexity lower bound for MoE decode | **Open** | FlashMoE's ML-based replacement, Jan 2026 ([arXiv](https://arxiv.org/abs/2601.17063)); no formal bound found | Best as the theory section of row 7 |
| 11 | Off-policy teacher-forced traces vs on-policy traces | **Partially open** | 2608.07911 (chat-template and contamination axes); SeqMoE replays generated tokens but makes no comparison; Hayashi et al., 19 Apr 2026, token identity drives routing ([arXiv](https://arxiv.org/abs/2604.17182)) | The policy-mismatch axis, quantified on conclusions |
| 12 | gpt-oss-120b loss near-uniform on dataset text | **Undocumented, unverified** | Raw-text perplexity of gpt-oss-20b is known to be high: llama.cpp #15155, Aug 2025 ([issue](https://github.com/ggml-org/llama.cpp/issues/15155)); transformers #40990, Sep 2025 ([issue](https://github.com/huggingface/transformers/issues/40990)) | Needs harmony-wrap and third-implementation checks |
| 13 | Hybrid CPU/GPU numerical divergence ("placement invariance") | **Partially open** | Thinking Machines, 10 Sep 2025 ([blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)); Cooper et al., 22 Sep 2026, GPU only ([arXiv](https://arxiv.org/abs/2609.25624)); an anecdote in PR #26563, Aug 2026 ([PR](https://github.com/ggml-org/llama.cpp/pull/26563)) | Measurement, root cause, a bitwise fix |
| 14 | Router-only fine-tuning for expert locality | **Taken** | ReMoE, 26 May 2026 ([arXiv](https://arxiv.org/html/2605.27081v1)); Cache-Conditional Experts, Dec 2024 ([arXiv](https://arxiv.org/html/2412.00099v2)) | Replication on fine-grained models |
| 15 | Temporal or cross-layer consistency loss in training | **Partially open** | StickyMoE, Jul 2026, toy scale ([arXiv](https://arxiv.org/html/2607.08780v1)); Oracle-MoE, ICML 2025, GPT-2 scale ([PMLR](https://proceedings.mlr.press/v267/zhou25b.html)) | Cross-layer consistency at modern scale (expensive) |
| 16 | Pre-attention or pre-gated routing | **Taken** | Pre-gated MoE (ISCA'24); Read-ME (NeurIPS'24) ([arXiv](https://arxiv.org/abs/2410.19123)); SmallThinker, Jul 2025 ([arXiv](https://arxiv.org/html/2507.20984v2)) | Sizing the lookahead from the bandwidth ratio (one paragraph) |
| 17 | Characterizing locality across architectures | **Taken** | Liang et al., ICLR 2026, 20 MoEs ([arXiv](https://arxiv.org/html/2505.16056v3)) | Only as an input to row 18 |
| 18 | **Offload-aware architecture co-design** (locality and offload terms in the design loop) | **Open** | Roofline co-design, 10 Feb 2026 ([arXiv](https://arxiv.org/html/2602.10377v1)); MobileMoE, 27 May 2026 ([arXiv](https://arxiv.org/pdf/2605.27358)); both assume resident weights | The whole question |
| 19 | On-device MoE design, resident-weight regime | **Taken** | MobileMoE; LFM2, Nov 2025 ([arXiv](https://arxiv.org/html/2511.23404v1)); SmallThinker | — |
| 20 | Lookup-table or other offload-native experts | **Taken (big-lab scale)** | MoLE, ICML'25 ([arXiv](https://arxiv.org/abs/2503.15798)); Engram, Jan 2026 ([arXiv](https://arxiv.org/abs/2601.07372)) | Evaluating them with the bound |
| 21 | Client-hardware DSE across machine classes | **Partially open, closing** | HBF "MoE in a box", 14 Sep 2026 ([arXiv](https://arxiv.org/html/2609.15636v1)); 2608.18261; Edge0, 16 Sep 2026 ([arXiv](https://arxiv.org/html/2609.18063)) | A cross-class DSE with measured validation |
| 22 | Batch-1 GPU efficiency gap, closed with better kernels | **Taken** | Hazy "No Bubbles", 27 May 2025 ([blog](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)); MPK, Dec 2025 ([arXiv](https://arxiv.org/html/2512.22219v1)); 2605.30571, May 2026 ([arXiv](https://arxiv.org/html/2605.30571v1)) | — |
| 23 | CPU+GPU bandwidth aggregation when the model fits (PCIe dGPU, MoE) | **Partially open** | BOOST, 11 Sep 2026 ([arXiv](https://arxiv.org/html/2609.13592)); HeteroInfer, SOSP'25 ([arXiv](https://arxiv.org/html/2501.14794v2)); FusionML, 24 Jul 2026, negative result ([Pith](https://pith.science/paper/2607.22785)); ATSInfer, Jul 2026 ([arXiv](https://arxiv.org/html/2607.10183)) | A bound parameterized by η_g (achieved fraction of GPU bandwidth); contingent on kernel quality |
| 24 | Parallel test-time compute on an offloaded MoE | **Partially open** | Hayashi et al., 19 Apr 2026; AcceptMoE, Aug 2026; XShare, Feb 2026 ([arXiv](https://arxiv.org/html/2602.07265v1)); Kinetics, Jun 2025 ([arXiv](https://arxiv.org/abs/2506.05333)); mobile-NPU TTS, EuroSys'26 ([arXiv](https://arxiv.org/pdf/2509.23324)) | Expert union U(n) → bytes → accuracy-per-second frontier |
| 25 | Joint KV-cache vs expert VRAM budgeting at long context | **Open (components crowded)** | HiSparse, Aug 2026 ([arXiv](https://arxiv.org/html/2608.07009)); ScoutAttention, Mar 2026 ([arXiv](https://arxiv.org/abs/2603.27138)); SSD-LLaMA ignores KV ([arXiv](https://arxiv.org/html/2609.18110v1)) | The joint question |
| 26 | Latency-bound MoE decode across consumer nodes (RDMA over Thunderbolt 5, Ethernet) | **Open academically** | prima.cpp, Apr 2025 ([arXiv](https://arxiv.org/abs/2504.08791)); OD-MoE, Dec 2025 ([arXiv](https://arxiv.org/html/2512.03927v1)); NCCL EP, 2026, datacenter ([arXiv](https://arxiv.org/abs/2603.13606v3)) | A LogGP-style model; needs physical nodes |
| 27 | Agentic prefix reuse and hybrid-state checkpointing on local hardware | **Partially open** | Marconi, MLSys'25, datacenter ([arXiv](https://arxiv.org/abs/2411.19379)); llama.cpp #22746/#20697, 2026 ([Particula](https://particula.tech/blog/prompt-reprocessing-swa-hybrid-models-kv-cache)) | A local-hardware model; the engine changes monthly |
| 28 | SSD expert streaming | **Taken (crowded)** | SSD-LLaMA and Edge0, both 16 Sep 2026; FlashMoE, Jan 2026 | Only bounds that explain these systems |
| 29 | Energy of hybrid CPU+GPU offload | **Partially open** | GreenBench, Aug 2026 ([arXiv](https://arxiv.org/abs/2608.28667)); "From Tokens to Watt-hours", 29 Jul 2026, dense models ([arXiv](https://arxiv.org/abs/2607.26571)) | Needs host RAPL, so owned hardware |
| 30 | Validated MoE-aware "will it run, how fast" predictor | **Partially open** | LIFE, Aug 2025, dense models only ([arXiv](https://arxiv.org/html/2508.00904)) | Folds into row 7 |
| 31 | Cross-backend portability as a fraction of speed-of-light | **Open academically** | Community threads; FOSDEM 2026 ([FOSDEM](https://fosdem.org/2026/schedule/event/CZSPSC-llama-cpp-vulkan/)) | Needs AMD and Apple hardware |
| 32 | When compressing offloaded experts pays off | **Partially open** | SSD-LLaMA's rANS coding; MoE-APEX, ASPLOS'26 ([ACM](https://dl.acm.org/doi/10.1145/3779212.3790187)) | One extra term in the bytes model |

The pattern is stark. In the eight weeks before this report, a new paper closed a mechanism or crowded a niche roughly every week:

- **2608.07911** (August): the trace-level contract and the count-based Belady gap.
- **FreeToken** (17 August): the closed-form CPU/GPU split.
- **"Cacheable by Design?"** (18 August): a bytes/bandwidth match for one configuration.
- **AcceptMoE** (August): union control under offload.
- **SeqMoE** and **BOOST** (both 11 September): cross-step prediction, and host/HBM bandwidth aggregation.
- The **Huawei "MoE in a box" DSE** (14 September).
- **SSD-LLaMA** and **Edge0** (both 16 September).
- **Cooper et al.** (22 September): cross-GPU determinism.

What remains open clusters in three places: bounds and measurement in seconds (rows 7–10), design questions that need such a bound as input (rows 18, 24, 25), and validity questions about how the traces themselves were produced (rows 11–13). Mechanism ideas are the wrong ground for a single researcher. The groups shipping them (SJTU, USTC, Berkeley/UT Austin, Georgia Tech with NVIDIA) turn an idea into a paper in weeks.

## A time-domain yardstick outscores the forward-looking alternatives

The scores below are judgments drawn from the vetting, 1 (poor) to 5 (strong) on each axis, unweighted, out of 35. "Venue" means the fit to a realistic 2026–27 venue. Scoop risk is not scored; it is discussed in the last section.

| Rank | Direction (verdict-table rows) | Novelty | Impact | Asset reuse | By 27 Oct | By Dec | Venue | SPCL fit | Total | Realistic venue |
|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **Time-domain yardstick + audit** (7, 8, 10, 30) | 4 | 3 | 5 | 4 | 5 | 4 | 5 | **30** | ISPASS 2027; MLSys 2027 only if it passes a gate |
| 2 | Parallel test-time compute on offloaded MoE (24) | 4 | 4 | 4 | 3 | 4 | 4 | 5 | 28 | MLSys or ISPASS; high scoop risk |
| 3 | Trace provenance, on- vs off-policy (11, 12) | 3 | 2 | 5 | 5 | 5 | 2 | 3 | 25 | Folded into #1; EuroMLSys alone |
| 4 | Offload-aware architecture co-design (18) | 4 | 5 | 3 | 2 | 3 | 3 | 4 | 24 | PhD agenda; ICML/MLSys 2027–28 once training exists |
| 5 | KV vs expert VRAM budgeting (25) | 3 | 3 | 4 | 3 | 4 | 3 | 4 | 24 | ISPASS/IISWC follow-on |
| 6 | Placement invariance (13) | 4 | 2 | 4 | 4 | 4 | 2 | 3 | 23 | EuroMLSys or workshop |
| 7 | CPU+GPU aggregation when the model fits (23) | 3 | 2 | 4 | 3 | 4 | 3 | 4 | 23 | ISPASS/EuroMLSys |
| 8 | Consumer multi-node MoE decode (26) | 4 | 3 | 2 | 1 | 2 | 3 | 5 | 20 | SC workshops/HOTI; needs hardware |

### Why the yardstick ranks first

**The yardstick wins on fit and finishability, not on ceiling.**

- **Novelty (4, not 5).** 2608.07911 already owns the framing that the field lacks "a common evaluation contract". Bytes over bandwidth is a standard model form. The novel parts are:
  - working in seconds rather than blocks;
  - validating against dozens of other groups' measurements rather than one's own system;
  - re-expressing others' claims as a fraction of the bound.

  2608.07911 states that "every result here is a block count". It names real-hardware validation as the limitation it most wants removed. It says its audit "does not support re-labelling published speedups as artifacts" ([arXiv 2608.07911](https://arxiv.org/abs/2608.07911)). The one bytes/bandwidth match in this niche covers a single RTX 3070 configuration ([arXiv 2608.18261](https://arxiv.org/abs/2608.18261)).
- **SPCL fit (5).** Hoefler and Belli's rule 11 is **"If possible, show upper performance bounds"** ([SC'15 PDF](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf)). Hoefler's advice to young researchers is **"Learn about I/O complexity!"** ([Axelera interview](https://community.axelera.ai/product-updates/interview-with-torsten-hoefler-axelera-ai-s-scientific-advisor-113)). SPCL's 2025–2026 publication list has no paper on MoE offloading or consumer inference ([SPCL publications](https://spcl.inf.ethz.ch/Publications/)).
- **Impact (3).** Measurement papers change reporting practice slowly. Their value is as infrastructure other people cite.
- **An honest limit.** With a **16% median leave-one-source-out error**, the audit cannot adjudicate a published speedup smaller than roughly 1.2–1.3×. The headline claims it would examine are mostly larger:
  - DALI's 3.97× over llama.cpp, averaged over batch sizes ([arXiv 2602.03495](https://arxiv.org/abs/2602.03495));
  - SeqMoE's 104.1 versus 30.9 tok/s for llama.cpp static offload ([arXiv 2609.12978](https://arxiv.org/abs/2609.12978));
  - KTransformers' 1.25–1.93× ([SOSP'25](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)).

### Why test-time compute is the runner-up, not the pick

**Test-time compute has the higher ceiling and the worse odds.** The question is how the per-step expert union U(n) grows across n samples of one prompt, and what that does to accuracy per second on a PCIe- or DRAM-bound box.

- **The hypothesis is crisp.** Under independent routing, XShare's law E[N_a] = N(1 − (1 − k/N)^B) ([arXiv 2602.07265](https://arxiv.org/html/2602.07265v1)) implies that best-of-8 amortizes expert bytes by only **1.24× on Qwen3-30B-A3B and 1.52× on OLMoE and gpt-oss-20b**. That is my arithmetic from the formula.
- **Correlation decides the real gain.** Hayashi et al. measured sibling-fork Jaccard of **0.649 for the same token** and 0.179 for different tokens, decaying to 0.294 and 0.086 after 200 steps. They state that "no verification [was] performed translating routing locality into actual offloading throughput" ([arXiv 2604.17182](https://arxiv.org/abs/2604.17182)).
- **SPCL has a direct hook.** SPCL's own work-depth-memory performance model for reasoning LMs (August 2026) is datacenter-only ([arXiv 2608.27046](https://arxiv.org/html/2608.27046v1)).
- **Why it loses.** It needs new multi-sample traces, accuracy runs and batched validation, all inside four weeks. AcceptMoE's authors already run offloaded union control at batch 1 on an RTX 5090, with 2.06× under offloading ([arXiv 2608.02989](https://arxiv.org/html/2608.02989)); best-of-n is a short step for them.
- **Where it goes instead.** It is the best second paper, using the same yardstick.

### Why co-design is the PhD agenda, not the October paper

**Co-design has the highest impact and the mission fit, but it cannot be finished by October.**

- **The gap is real.** The roofline co-design study found that **"MoE configurations constitute 100% of Pareto-optimal designs under on-device batch-one inference"**. Its latency model assumes resident weights, and it names routing overhead as future work ([arXiv 2602.10377](https://arxiv.org/html/2602.10377v1)).
- **There is an unresolved tension.** MobileMoE's loss-optimal recipe uses fine granularity and a shared expert, with offloading "not addressed" ([arXiv 2605.27358](https://arxiv.org/pdf/2605.27358)). Liang et al. found that shared experts "significantly harm" routing consistency ([arXiv 2505.16056](https://arxiv.org/html/2505.16056v3)). Only an objective that includes offloading can arbitrate between the two.
- **Why it waits.** Public checkpoints vary granularity, expert count, data and training all at once, so a credible answer needs a training grid. The roofline study trained **170 models at 10B tokens each**.
- **What goes in the October paper.** A small "design implications" section, as a teaser.

### The rest of the shortlist

- **Provenance (rank 3)** is small but necessary, so it is folded into the primary paper.
- **Placement invariance (rank 6)** runs into the researcher's own data: NLL within 0.1% supports the reading that the effect is benign.
- **Aggregation when the model fits (rank 7)** is contingent. BOOST gained only **+4.3%** in low-latency TPOT on GH200 ([arXiv 2609.13592](https://arxiv.org/html/2609.13592)). A better GPU kernel could erase the effect: a megakernel reaching 78% of H100 bandwidth is the benchmark ([Hazy Research](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)).
- **Consumer multi-node (rank 8)** fits SPCL's networking cluster best of all, but needs 2–4 physical machines.

## Recommendation: publish a validated speed-of-light before 21 October

### The novelty claim must say seconds, many papers, and equal VRAM

The claim below is worded so that each clause names something the closest prior art explicitly does not do. Bracketed quantities are filled in once measured.

> We give (1) a lower bound on batch-1 decode step time, in seconds, for any policy that executes an MoE's exact routing within a given per-layer expert capacity, with a global-capacity variant. The bound holds whether misses are fetched on demand, prefetched, or executed on the CPU, and it is computed from exact on-policy routing traces. (2) We validate the accompanying bytes-over-bandwidth model out of sample: against 52 third-party measurements from [N] independent sources (leave-one-source-out median error 16%), and against pre-registered first-party measurements (2.9% median error on 33 configurations). (3) We re-express [K] published MoE-offloading results as fractions of their own hardware's bound, next to a model-predicted equal-VRAM expert-offload baseline, anchored by re-runs of [M] published configurations. (4) We quantify how teacher-forcing dataset text, instead of the model's own samples, changes routing locality, cache conclusions and the bound.

What the paper must cite and not claim:

- the trace-level contract and the count-based Belady gap ([2608.07911](https://arxiv.org/abs/2608.07911));
- bytes/bandwidth agreement for one configuration ([2608.18261](https://arxiv.org/abs/2608.18261)) or for a system's own branches ([FreeToken](https://arxiv.org/abs/2608.16157));
- sparsity-aware bandwidth utilization. MoE-CAP's S-MBU divides achieved bandwidth by a single B_peak, which is ill-defined when expert bytes come from three tiers at different rates. That reading is from the main-text equation; MoE-CAP's appendix was not checked ([arXiv 2412.07067](https://arxiv.org/abs/2412.07067));
- hierarchical rooflines for batched throughput ([MoE-Lightning](https://arxiv.org/abs/2411.11217));
- teacher-forced replay of generated tokens ([SeqMoE](https://arxiv.org/abs/2609.12978));
- pre-registration, which both 2608.07911 and 2608.18261 already use.

Position the paper as filling 2608.07911's own checklist item D1, "absolute transferred bytes per output token … together with the bandwidth budget". Its author says that field cannot be filled without hardware calibration ([arXiv 2608.07911](https://arxiv.org/abs/2608.07911)). Also report its regime ratio (union over capacity) for every row. That turns the closest competitor into the paper's first citation rather than its first reviewer objection.

The bound needs one extension before it is safe to call it a speed-of-light. FreeToken's shared all-layer cache sits outside a per-layer-capacity class, and it missed only **16% and 39%** of decode expert reads in its own equal-capacity replay ([arXiv 2608.16157](https://arxiv.org/abs/2608.16157)). Experts within one model are equal-sized, so Belady MIN over the interleaved all-layer access stream should give the global-capacity variant. That is my inference, and it must be checked against the repository's bound test. Methods that alter routing (Cache-Conditional Experts, KTransformers' Expert Deferral) stay outside the bound by definition, and the paper's statement must say so.

### Thesis and title

> **Thesis.** On consumer hardware, how fast an offloaded MoE can decode is set by the bytes moved through each memory tier over that tier's sustained bandwidth, and by the model's own routing. Offloading systems should therefore be compared as fractions of a trace-driven speed-of-light on the hardware they ran on, not as speedups over baselines of unstated strength. The paper tests the hypothesis that, measured this way, much of the spread among published speedups reflects bandwidth ratios and baseline configuration rather than policy.

> **Title.** *Seconds, Not Blocks: A Validated Speed-of-Light for Offloaded Mixture-of-Experts Decode*

Pre-register the audit's decision rules so that both outcomes are publishable. If published speedups survive normalization, the paper reports that the literature holds up. The bound and the validated model are the contribution either way.

The baseline evidence makes the hypothesis plausible:

- **Weak or unusual baselines are common:**
  - KTransformers "extended Llama.cpp with custom code" and ran it in FP16 ([SOSP'25](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)).
  - ProMoE's own layer-offload baseline "surpasses ProMoE in the decode stage" on Mixtral ([arXiv 2410.22134](https://arxiv.org/abs/2410.22134)).
  - HOBBIT's 13–19× over llama.cpp on Jetson Orin came from mmap page faults ([arXiv 2411.01433](https://arxiv.org/abs/2411.01433)).
- **Configuration is rarely reported.** Only Fiddler states a llama.cpp build. No academic system paper found configures `--n-cpu-moe` at equal VRAM ([arXiv 2402.07033](https://arxiv.org/abs/2402.07033)).
- **Host bandwidth moves baselines.** FreeToken reports that llama.cpp "keeps only 80% of its rate" when its CPU-resident experts "starve on two DDR5 channels" ([arXiv 2608.16157](https://arxiv.org/abs/2608.16157)).
- **Audits are publishable.** A rigor-matched audit of layer skipping, posted 28 August 2026, reversed the naive ranking of two methods in three of four cells ([arXiv 2608.28846](https://arxiv.org/abs/2608.28846)).

### Four weeks to 27 October

Use Lambda for anything that needs clock locking or Nsight counters, and RunPod consumer GPUs for tok/s-only anchor runs, where the lack of `ncu` access does not matter. GPU-hour figures are my estimates.

| Week | Measure and build | Compute | Write and admin | Exit check |
|---|---|---|---|---|
| **1** (Mon 28 Sep – Sun 4 Oct) | Generate each model's own samples on the 160 corpus conversations, greedy and at T=0.8. For gpt-oss, use full harmony with `analysis` and `final` channels. Classify the gpt-oss-120b anomaly: wrap the dataset text as a proper `final`-channel message, and score it with vLLM `prompt_logprobs` as a third implementation. Extract audit rows for about 12 systems plus the llama.cpp community PRs: batch-1 single-turn numbers, hardware, baseline flags. Digitize figure-only values (KTransformers, HybriMoE, DALI, Fiddler). Tier each row as directly modelable, imputable or weak. Pre-register the provenance metrics, the "conclusion flips" criteria and the audit decision rules. | Lambda H100 80 GB with vLLM/SGLang, a few GPU-hours; existing CPU VM | Rewrite the abstract and intro around the claim; build the related-work table from rows 1–10; secure arXiv endorsement for cs.DC/cs.PF if never posted there ([arXiv endorsement](https://info.arxiv.org/help/endorsement.html)); ask 2–3 referees now | On-policy corpora exist; ≥10 sourced audit rows; 120b anomaly classified |
| **2** (5–11 Oct) | Trace the on-policy corpora with the GPU-free tracer. Cross-check a slice against SGLang `--enable-return-routed-experts`. Compute on both provenances: union/capacity, LRU/LFU/Belady hit rates at 12.5/25/50% capacity, the bound, and predicted tok/s. **Recompute every existing fraction-of-bound number on on-policy traces.** Add the global-capacity bound variant. Add Mixtral-8x7B to the tracer, since it appears in most audited papers. For each audit row, compute the bound, the leave-one-source-out prediction and the model-predicted equal-VRAM `--n-cpu-moe` baseline, with bands for imputed DRAM bandwidth. | A larger CPU instance for tracing; Lambda A10 spot checks | Methods: the bound's statement and scope, the model, the validation | Provenance effect sized; audit numbers carry bands |
| **3** (12–18 Oct) | Pre-registered anchor re-runs of 2–3 published configurations. First, Qwen3-30B-A3B on an RTX 4090 over PCIe 4, at layer offload and at equal-VRAM `--n-cpu-moe` (SeqMoE's 30.9 tok/s static-offload point; DALI's setup). Second, Mixtral-8x7B on an RTX 4090 (ProMoE; 2512.16473). Measure host STREAM/MLC bandwidth on every pod. Then write the design-implications section: speed-of-light for 4–5 architectures on three platform classes. | RunPod RTX 4090 (5090 if available), 1–2 days; Lambda A10 | Results, audit table, discussion. **Post arXiv v1 by Fri 16 Oct, no later than Tue 20 Oct.** | Anchors land inside the model's error band, or the misses are explained |
| **4** (19–25 Oct) | Bootstrap CIs across sequences; nonparametric tests; no averaged ratios. Release code, traces, the 140-row measurement set and a bound calculator. Upstream the ggml scheduler fix as a small standalone PR. | Minimal | arXiv v2. Motivation letter, research proposal, CV, video. Send courtesy notes to audited authors with their rows and invite corrections. | Application complete by Sun 25 Oct |
| **Close** (26–30 Oct) | — | — | Submit ETH by **Tue 27 Oct, 16:00 CET**. MLSys go/no-go on Thu 29 Oct. MLSys deadline **Fri 30 Oct, 12:00 PDT**. | — |

The ETH dates come from the fellowship FAQ: applications close 27 October at 16:00 CET, letters are due 2 November, and at least two referees must be independent investigators or lecturers ([ETH AI Center FAQ](https://ai.ethz.ch/research/phd-and-postdoc-programs/fellowship-faqs.html)). MLSys submissions open 10 October and reviews arrive 18 January 2027 ([MLSys 2027 dates](https://mlsys.org/Conferences/2027/Dates)).

If the schedule slips, cut in this order: design implications, then Mixtral tracing, then the third anchor, then the global-capacity variant. Never cut the on-policy recomputation, the two anchors, or the confidence intervals. Without them the paper repeats the weaknesses it audits.

### MLSys needs a gate; ISPASS needs a consumer machine

Submit to MLSys on 30 October only if four conditions hold:

1. At least two anchors land inside the model's leave-one-source-out band.
2. Every bound number rests on on-policy traces.
3. The audit headline is quantified, in the form "X of Y published batch-1 speedups fall below Z× against the equal-VRAM baseline".
4. Every headline number carries a confidence interval.

MLSys has no benchmarks track that was verified, so a measurement paper there is a gamble either way.

Otherwise target ISPASS 2027. It is inferred at mid-December from its 15 December 2025 deadline, a 9-page limit and a tool-and-benchmark option ([ISPASS 2026](https://ispass.org/ispass2026/submission.php)). For ISPASS, add:

- **One owned or borrowed desktop** with dual-channel DDR5 and a 12–24 GB GPU. Cloud pods run server hosts, and FreeToken's 80% figure shows that host channels move the baseline.
- **Same-machine re-runs of two or three systems with public code.** SGLang with kt-kernel on Qwen3-30B-A3B, with deferral set to zero; and one llama.cpp community cache, #27861 or #24528 force-enabled.
- **The k-lookahead MIN curve** (row 9), showing how much of the bound's gap draft or MTP foresight could close through eviction.
- **A fidelity clause in the reporting contract.** Outputs are bitwise-identical, tolerance-equal or approximate. The 4.5% (gpt-oss-20b) and 1.6% (Qwen3) greedy top-1 disagreement between layer offload and all-GPU is the first published figure for the placement effect that PR #26563 describes only anecdotally ([PR #26563](https://github.com/ggml-org/llama.cpp/pull/26563)).

If the yardstick goes to MLSys instead, the ISPASS slot should carry the test-time-compute characterization, built on the same yardstick.

### The application should propose co-design, not caching

The fellowship requires a motivation letter, CV, video and references, not publications ([ETH AI Center FAQ](https://ai.ethz.ch/research/phd-and-postdoc-programs/fellowship-faqs.html)). SPCL's direct route asks for a "brief description of most important achievement" ([SPCL Jobs](https://spcl.inf.ethz.ch/Jobs/)).

The achievement is the validated yardstick. The proposal is **"Accessible AI through data-movement-optimal MoE"**, with three steps:

1. Measure what consumer hardware can do (the preprint).
2. Design architectures whose routing locality and expert granularity are chosen for offloaded decode rather than resident decode, testing the hypothesis that loss-optimal on-device recipes are not offload-optimal.
3. Spend otherwise idle bandwidth on test-time compute.

Anchor it to Hoefler's own framing. His "Supercomputers to Smartphones" talk abstract says these techniques "may eventually enable inference with powerful models on hand-held devices" ([ISI event page](https://www.isi.edu/events/3921/scalable-and-efficient-ai-from-supercomputers-to-smartphones/)). SPCL's August 2026 reasoning-LM performance model covers clusters but not a single consumer node's memory hierarchy ([arXiv 2608.27046](https://arxiv.org/html/2608.27046v1)).

## Every asset survives, but as instrument rather than contribution

| Asset | What it becomes | What must change | What to drop |
|---|---|---|---|
| (a) GPU-free, layer-streamed, teacher-forced tracer | The instrument for exact on-policy traces of models of any size | Feed it the model's own samples. Add Mixtral. Report exactness against SGLang's routed-experts output. | Any novelty claim for teacher-forced replay: SeqMoE packs generated tokens into a prefill ([arXiv](https://arxiv.org/abs/2609.12978)) |
| (a) Dataset-text traces for OLMoE, Qwen3-30B-A3B, gpt-oss-20b | The off-policy arm of the provenance study | Relabel them as off-policy everywhere | Using them as the sole basis of any bound or locality number |
| (a) gpt-oss-120b trace; NLL 7.3 vs 0.65 nats | The provenance section's hook, if it survives the checks | Add the harmony-wrap and third-implementation checks. OpenAI says gpt-oss "will not work correctly" without harmony ([OpenAI Cookbook](https://developers.openai.com/cookbook/articles/openai-harmony)) | The whole thread, if it proves to be a format artifact |
| (b) Bytes-over-bandwidth model; 52-point leave-one-source-out validation; the 140-row dataset | The audit engine, a predictor of the strong baseline each paper should have run, and a released dataset | Use leave-one-source-out predictions for audited rows. Carry bands for imputed bandwidth. | Any claim that the model form is new |
| (b) Two-run on-platform calibration (4.9%) | The calibration protocol inside the reporting contract | — | — |
| (b) Time-domain bound (Belady MIN-bypass + load accounting + Jensen) | The centerpiece | Add the global-capacity variant; state the scope; recompute on on-policy traces | Belady-with-bypass and r* as novelty claims; r* has the same form as 2608.12103's breakeven count ([arXiv](https://arxiv.org/abs/2608.12103)) |
| (c) A10: static offload affine in CPU layers (R² > 0.996) | Validation of the model's static-offload term | — | — |
| (c) A10: batch-1 GPU at about 0.5 of device bandwidth; ~84 µs/layer hand-off; copy-engine contention; CPU helpers at 107/143 GB/s | The "where the time goes" decomposition of every gap to the bound | Present as overhead terms, measured with null-work controls | — |
| (c) Pre-registered step model, 2.9% median error on 33 configurations | The first-party validation case study | — | "First pre-registered study" as a claim |
| (c) Mailbox cache, 1.41–1.81× over `--n-cpu-moe` at equal VRAM | One system under test, showing the headroom is reachable: 46–63% of speed-of-light, against 30–42% for llama.cpp, both to be recomputed on on-policy traces | Move it to a short section. The host-callback ablation waits for ISPASS. | The cache, decayed-frequency admission and one-graph execution as contributions. HybriMoE, 2512.16473, FreeToken, KTransformers and three llama.cpp contributors got there first. |
| (c) Top-1 disagreement 4.5% / 1.6%, NLL within 0.1% | The fidelity clause; the seed of a later placement-invariance note | Tie it to the root cause: CPU Q8_0/Q8_K activations versus CUDA q8_1 ([ggml-cpu.c](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/ggml-cpu.c); [mmvq.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu)) | — |
| (c) ggml scheduler overlap bug | An upstream PR; the most direct community impact | Show it does not reintroduce the race that upstream fixed in August 2026 | — |
| (d) Lambda job runner, llama.cpp instrumentation, sampled-corpus generator | Unchanged infrastructure; the generator now produces the on-policy input | — | — |
| (d) GPU-signalled CPU hand-off with GPU-decided skip | Kept in the repository as future systems work | Needs a `cudaLaunchHostFunc` variant before any claim | From the October paper |

Two things must go completely. The first is the system-first framing and its title. The second is every novelty claim that the prior-art table in the first section refutes: the cache concept, the admission rule, one-graph execution, r*, Belady with bypass, teacher-forced replay and "first pre-registered". The traces also need honesty up front. The existing bound results rest on teacher-forced dataset responses, and on-policy traces could move them. If the 46–63% figure shifts materially once recomputed, that shift is itself the provenance section's headline.

## The main scoop risk has a name and four versions in seventeen days

| Risk | Evidence | Likelihood | De-risking |
|---|---|---|---|
| 2608.07911's author extends the contract to the time domain | v1 early Aug 2026, v4 on 25 Aug. The paper names "real-hardware validation" as the limitation it most wants removed and leaves D1 unfilled ([arXiv](https://arxiv.org/abs/2608.07911)) | **High** within 1–3 months | Post v1 by 16–21 Oct. Frame the paper as filling D1 and cite it first. Check its arXiv listing weekly. |
| A system group publishes bound comparisons | FreeToken and SeqMoE already model their own branches as bytes over bandwidth | Medium | Their incentive is to benchmark their own system, not to audit others. The cross-paper validation is the moat. |
| MoE-CAP extends S-MBU to tiered memory | Its cost model already includes CPU, PCIe, DRAM and SSD terms ([arXiv](https://arxiv.org/abs/2412.07067)) | Medium-low | Read its appendix and repository in week 1 |
| Concurrent submissions not yet visible | ICLR 2027 and MLSys 2027 submissions are not public; NeurIPS 2026 decisions arrive around late September | Unknown | Search ICLR 2027 submissions on OpenReview once they are listed. The arXiv timestamp is the only defense. |
| The proposal direction gets scooped | The roofline co-design and MobileMoE teams both list routing or dynamic routing as future work | Medium | Stake a small result in the design-implications section |
| The gpt-oss-120b anomaly is an artifact | Raw-text perplexity of gpt-oss-20b is 196–394, from two independent reports ([#15155](https://github.com/ggml-org/llama.cpp/issues/15155); [#40990](https://github.com/huggingface/transformers/issues/40990)); agreement between HF fp32 and llama.cpp does not rule out shared upstream causes | Medium | Harmony wrap plus vLLM `prompt_logprobs` in week 1; drop it if it fails |
| The audit reads as a hit piece | 2608.07911 deliberately avoided calling any speedup wrong | Medium | Report both system and baseline as fractions of the bound. Send courtesy notes and invite corrections for v2. |
| Model error swamps the effects being audited | 16% median leave-one-source-out error | Medium | Adjudicate only effects beyond the error band. The anchors test the counterfactual baselines. |
| Practical blockers | arXiv endorsement for first-time posters; referee letters due 2 Nov | Controllable | Settle both in week 1 |

Several papers need reading in week 1, because each could move a verdict:

- **2608.07911's latest version**, and any follow-up by its author.
- **MoE-CAP's appendix**, for how S-MBU handles multi-tier offload.
- **NVIDIA's MLSys 2026 client-inference paper.** One reading says its baseline is llama.cpp `-cmoe`; a full-text grep found only `-ngl`. This decides whether any strong-baseline precedent exists ([arXiv 2604.26334](https://arxiv.org/abs/2604.26334)).
- **The 2608.18261 repository**, to learn whether its domain traces were teacher-forced.
- **SPCL's 2608.27046**, to confirm it has no consumer or offload term.
- **Unread items that SeqMoE cites:** Taming-MoE, Patterns-MoE and 2511.10676.

The test-time-compute runner-up carries a separate, high scoop risk from the AcceptMoE, EcoSpec and Hayashi groups. If it becomes the December paper, start its trace collection in November, not January.

## Conclusion

The field's bottleneck moved in August and September 2026 from mechanisms to measurement. At least seven groups converged on the same hybrid cache, and four more closed the obvious next steps within weeks of each other. What nobody built is a way to say how far any of them sits from what the hardware allows. That is also where an independent researcher's comparative advantage lies: large groups have no incentive to audit their own speedups, and a validated cross-paper model cannot be scooped by writing faster kernels. The recommendation turns what looked like a late systems project into an early methodology paper, on the one axis where the assets are already ahead.

The deeper payoff is for the mission, not the October deadline. A yardstick in seconds is exactly the objective function that hardware-constrained architecture design has lacked. The on-device design literature optimizes for resident weights, while consumer MoE deployment is dominated by weights that are not resident. The prediction that loss-optimal MoE recipes are not offload-optimal is the most consequential open question the yardstick exposes, and it is the thesis-sized question to put in front of SPCL.
