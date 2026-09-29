# New prior art, 1–28 September 2026 (plus August items an earlier check missed): offloaded MoE decode on one consumer or workstation GPU, checked against the five contributions of "Seconds, Not Blocks"

Contribution labels used below:
- **C1** = seconds-domain lower bound (Belady MIN-with-bypass + load accounting + Jensen; per-layer or shared budget; static or dynamic; demand or prefetch; PCIe fetch or CPU execution; GPU-free teacher-forced traces).
- **C2** = bytes-over-bandwidth model, validated leave-one-group-out on 52 third-party measurements plus a pre-registered test on three first-party platforms.
- **C3** = audit of 41 systems against the bound and a predicted equal-memory llama.cpp `--n-cpu-moe` baseline.
- **C4** = trace provenance (teacher-forced dataset text vs the model's own generations; gpt-oss-120b NLL 7.3 vs 0.65 nats).
- **C5** = a reporting contract in seconds.
- **C6** = the first-party same-machine, same-client comparison (FreeToken's client driving llama.cpp, a llama.cpp GPU expert cache and FreeToken at equal GPU expert memory, 9–34% of the bound).

Method:
- **arXiv sweep.** Full monthly listings for Aug and Sep 2026 in cs.DC, cs.AR, cs.PF, cs.LG, cs.CL, cs.AI and cs.OS, plus pastweek listings through Mon 28 Sep 2026: 12,926 unique IDs. Titles were keyword-filtered, then 47 abstract pages were read. Full PDF text was read (pdftotext) for 2609.04238, 2609.29032, 2608.21614, 2606.21868v2, 2608.21240, 2608.11688, 2609.14507, 2608.23841, 2609.30854, 2609.14864, 2609.18110, 2608.16947, 2609.22156 and 2608.26612.
- **Code repositories.** FreeToken, KTransformers, SGLang and vLLM commits from 25 Aug to 28 Sep were read from git. For llama.cpp, the master commits from 26–28 Sep were read, plus the head commits of PRs #29500–#29628 and of all MoE-cache PRs already known.
- **Not available.** Semantic Scholar returned HTTP 429 every time and the arXiv API returned "Rate exceeded". GitHub REST/HTML pages were blocked, except through WebFetch.

## Q1. Which arXiv papers from September 2026 (and missed August items) address MoE offloading, expert caching, hybrid CPU-GPU MoE decode, consumer-GPU performance bounds or evaluation methodology — and do they overlap C1–C5?

### Takeaway
No new paper takes any of the five contributions whole. Three papers take pieces of C1/C2 and must be cited and differentiated:
- **"Budgeting Bytes" (2609.04238)** has a closed-form "windowed storage roofline" with an exposed-latency bound. It also has a bytes-per-token tok/s planner that explicitly discusses llama.cpp `--n-cpu-moe`. It is edge/storage-focused.
- **"Paging the Experts" (2609.29032, 25 Sep)** computes a Belady-style "lower miss bound" on a shared all-layer expert cache. It says explicitly that this is "not an online speed prediction", i.e. blocks, not seconds.
- **WiSP v2 (2606.21868v2, 30 Aug)** converts LRU miss curves into seconds through a per-miss PCIe cost. It states that prefetching cannot help single-stream decode because PCIe is the bottleneck.

Several new systems (WiSP, SAEM, SPICE, SSD-LLaMA, S2-MoE, APEX) are fresh rows for the C3 audit. None of them uses a llama.cpp `--n-cpu-moe` baseline.

### Cited Findings

**Partial overlaps (must cite; differentiate)**

- **2609.04238 "Budgeting Bytes: A Windowed Storage Roofline and Dual-Budget Architecture Ablations for Storage-Bound LLM Decoding"** (Hanhaodi Zhang, single author).
  - Date: arXiv v1 is stamped Wed 29 Jul 2026, but the 2609.* ID means it was first announced in September 2026. The exact announcement day could not be retrieved. The comment says: "Workshop draft. Deployment numbers are single-run per configuration" — [arXiv 2609.04238](https://arxiv.org/abs/2609.04238)
  - C1 (partial): "The classification reduces prefetch scheduling to single-machine feasibility with release times, yielding a closed-form windowed roofline: a full-hiding criterion and exposed-latency bound via EDF optimality." — [arXiv 2609.04238](https://arxiv.org/abs/2609.04238)
  - C1 (partial): on the time floor, "prefetch changes when and how bytes are read, never how many. A per-token expert read is ∼500–900 MB; no schedule beats bytes/B." — [arXiv PDF 2609.04238v1](https://arxiv.org/pdf/2609.04238)
  - C2 (partial): the paper ships "Strata, a deployment planner that takes a MoE spec and a device (RAM, slow-storage bandwidth, measured granularity efficiency η) and reports predicted tok/s for all-in-RAM / static-offload (llama.cpp -ot) / predict-prefetch via the windowed roofline". Validation is "against a published microcontroller deployment (< 3% calibration error)" and "overlap pipelines match Theorem 1 within 7% (serial 224 vs. predicted 209 ms/token; overlapped 139.6 vs. 130)" — [arXiv PDF 2609.04238v1](https://arxiv.org/pdf/2609.04238)
  - C3-adjacent: "llama.cpp's --n-cpu-moe/--override-tensor places expert FFN tensors in slow memory while keeping attention in fast memory … It runs Qwen3-30B-A3B at ∼30 tok/s on 6 GB VRAM + 32 GB RAM." It claims "the theory that says why this placement is near-optimal and where it breaks" — [arXiv PDF 2609.04238v1](https://arxiv.org/pdf/2609.04238)
  - C3-adjacent: "Why do prefetch predictors (ProMoE et al.) report 1.3–2× where ours yields nothing? Because their win is cache-hit reduction of transferred bytes, not overlap". On an A100 "(PCIe Gen4, 25.4 GB/s H2D measured) under CPU→GPU expert offload: single-stream, transfer dominates compute and prefetch-overlap tops out at 1.34×". The engine used for the A100 run is not stated — [arXiv PDF 2609.04238v1](https://arxiv.org/pdf/2609.04238)
  - Platforms: an 8 GB RK3588 (eMMC) and an Apple M4 16 GB running Qwen3-30B-A3B. It found that a "trace-driven oracle (perfect prediction, 0.12 → 0.13)" does not help. The paper has no Belady/MIN, no per-layer vs shared GPU budget, no CPU execution of misses, no third-party measurement corpus, no audit and no trace-provenance study — [arXiv 2609.04238](https://arxiv.org/abs/2609.04238)

- **2609.29032 "Paging the Experts: A Reproducible Characterization of Flash-Backed MoE Inference on iPhone"** (Musa Shams).
  - Date: v1 Thu 24 Sep 2026 05:05 UTC; announced Fri 25 Sep 2026 in cs.PF — [arXiv 2609.29032](https://arxiv.org/abs/2609.29032); [cs.PF pastweek listing](https://arxiv.org/list/cs.PF/pastweek)
  - C1 (partial, blocks only): "Farthest-next-use replacement is a classical offline paging reference [6]" ([6] = Belady 1966). The paper states: "The relaxed unpinned oracle is a lower miss bound under equal-size, mandatory-admission paging, not an online speed prediction." — [arXiv PDF 2609.29032](https://arxiv.org/pdf/2609.29032)
  - Hit-rate results for the oracle: 54.73% / 57.13% / 68.31% at 512 / 576 / 1,024 MiB, against LRU at 0.00% / 38.58% / 47.59% on a shared all-layer expert cache for Qwen3.6-35B-A3B. The "minimum observed distinct-expert reuse distance is 312 on every trace" — [arXiv PDF 2609.29032](https://arxiv.org/pdf/2609.29032)
  - C4/C5-adjacent: "The apparent capacity cliff is therefore a policy/workload interaction, not a universal memory requirement." Also: "These results establish bounded feasibility and identify limitations that a deployment claim must not hide." — [arXiv 2609.29032](https://arxiv.org/abs/2609.29032)
  - Traces are the model's own recorded 128-token generations ("five later 128-token route captures"). There is no bypass (mandatory admission), no seconds conversion, no PCIe or CPU-execution path (iPhone flash plus unified memory) and no audit — [arXiv PDF 2609.29032](https://arxiv.org/pdf/2609.29032)

- **2606.21868v2 WiSP, "A Working-Set View of Mixture-of-Experts Serving on Extremely Low-Resource Hardware"** (Zhang, Wu, Darbari, Hong; Nokia). This is an August update the earlier check missed.
  - Date: v1 20 Jun 2026; **v2 Sun 30 Aug 2026**. The v2 comment reads "all headline results re-measured on a physically constrained 24 GiB RTX 3090; adds discussion of concurrent work" — [arXiv 2606.21868](https://arxiv.org/abs/2606.21868)
  - C1/C2 (partial, form only): "An expert miss transfers weights over PCIe, with cost c_exp = M_e t_byte … The latency model at split f is T(f) = c_exp m_exp(C) + c_kv m_kv(κ)". Also: "From a reference trace, the reuse-distance histogram determines the miss curve m(C)". The miss curve is LRU's, not Belady's, so this is an estimator, not a lower bound — [arXiv PDF 2606.21868v2](https://arxiv.org/pdf/2606.21868v2)
  - Insight overlap with our bound's message: "We find that this does not help in single-stream decode: the bottleneck is PCIe bandwidth, not prediction quality, so speculative transfers compete with demand transfers instead of hiding them." Turning on prefetch "lowers decode throughput by 38–60% at constrained caps" — [arXiv 2606.21868](https://arxiv.org/abs/2606.21868)
  - C3 row (supports our finding): "naive offload is vLLM's static --cpu-offload-gb". WiSP "achieves up to 2.0x the decode throughput of static offload at the same memory budget". The absolute steady state for Qwen3-30B-A3B on the RTX 3090 is "6.47 vs. 6.82 tok/s" (coherent vs diverse session). llama.cpp appears only in related work — [arXiv PDF 2606.21868v2](https://arxiv.org/pdf/2606.21868v2)
  - It discusses FreeToken as concurrent work: "at decode, serves residual expert misses through a second channel: executing missing experts in place on the CPU" — [arXiv PDF 2606.21868v2](https://arxiv.org/pdf/2606.21868v2)

**No overlap: new systems that are candidate rows for the C3 audit**

- **2608.21614 SAEM** (Zhang, Gao, Mitra). v1 Fri 21 Aug 2026. It is the extended version of a DAC 2026 paper with "a prediction-oracle upper-bound analysis".
  - Claim: "an average 1.33x throughput improvement over the strongest state-of-the-art caching and offloading baselines under constrained GPU memory, rising to 1.54x when calibration data matches the workload".
  - Hardware and baselines: a single A100 80GB, against MoE-OnDemand, Mixtral-Offloading, Fiddler and DAOP. There is no llama.cpp baseline.
  - Its "oracle" is expressed as a cache-hit ratio or throughput, not a hardware bound — [arXiv 2608.21614](https://arxiv.org/abs/2608.21614)
- **2608.21240 SPICE** (Lyu, Li, Jia; ASP-DAC 2027). v1 21 Aug 2026; v2 Mon 7 Sep 2026.
  - Platforms: "an NVIDIA 5090 with PCIe 4.0 x8 and 128 GB DDR5", an RTX 4060 with PCIe 4.0 x8 and 16 GB DDR4, and an A800.
  - Claims: "up to 3.12 speedup in Time Per Output Token" and "2.04–2.70× speedup over AdapMoE". Baselines are Naive, LRU, AdapMoE and CG-MoE; there is no llama.cpp baseline.
  - It is lossy: "low-confidence misses are approximated by the resident shared expert with low rank expert (LoRE) surrogates". Only its exact-residual mode falls under an exact-routing bound — [arXiv 2608.21240](https://arxiv.org/abs/2608.21240)
- **2609.18110 SSD-LLaMA** (16 Sep 2026). RTX 5090 with an NVMe tier.
  - Claim: 1.35×–3.44× over llama.cpp.
  - The PDF text has no occurrence of `n-cpu-moe`, `override-tensor` or `-ngl`, so the llama.cpp configuration is unstated. The paper also reports "77.7% of peak SSD bandwidth" — [arXiv 2609.18110](https://arxiv.org/abs/2609.18110)
- **2608.15018 S2-MoE** (15 Aug; v2 19 Aug 2026). "Implemented in llama.cpp, S2-MoE achieves up to 5.3× speedup (about 2.0× on average) over standard autoregressive decoding … on edge devices". This is self-speculative decoding, so the baseline is plain AR decoding, not an offload config — [arXiv 2608.15018](https://arxiv.org/abs/2608.15018)
- **2608.11688 APEX** (Kanani, Badawi, Ogras; CODES 2026 / TCAD). v1 12 Aug 2026.
  - It is simulated: Ramulator timing, an edge accelerator on "PCIe 6.0 ×16". It has a "correctness-preserving mode that guarantees exact routing semantics" with "up to 26%" lower per-token latency.
  - Baselines are fixed top-k prefetch schemes — [arXiv 2608.11688](https://arxiv.org/abs/2608.11688)
- **2608.02989 AcceptMoE** (4 Aug 2026; already in prior notes): 2.06× throughput under offloading on an RTX 5090 at batch 1 — [arXiv 2608.02989](https://arxiv.org/abs/2608.02989)
- **2608.22643 NeuroPrefetcher** (ICPP 2026). v1 23 Aug 2026. Dense sparse-neuron streaming from NVMe: "7.9-12.0x speedup over [llama.cpp] across constrained memory budgets" on unified-memory edge hardware. This is not MoE — [arXiv 2608.22643](https://arxiv.org/abs/2608.22643)

**No overlap: other September/August items read and dismissed**

- **2609.12978 SeqMoE** (known prior art): still v1 (11 Sep); no update — [arXiv 2609.12978](https://arxiv.org/abs/2609.12978)
- **2609.14864, Qiu et al., "GGUF-Metadata Prediction of Single-Sequence llama.cpp Throughput Across Three Systems"** (known prior art).
  - v1 Mon 14 Sep 2026, no later version. It is ICASSP 2027-bound.
  - It uses first-party measurements only: 318 measurements from 53 host-file configurations on two M4 Max machines and an RTX 5080. Leave-one-host-out test MAPE is "11.6%, 16.8%, and 36.0%". There is no PCIe expert offload.
  - Remains a known partial overlap for C2 — [arXiv 2609.14864](https://arxiv.org/abs/2609.14864)
- **2609.14507 WriteScope/InplaceKVCache** (13 Sep 2026). Single-GPU long-context MoE with the KV cache split between CPU and GPU. It uses "a portable roofline performance model" for the CPU share, on A100/V100. The KV side is out of scope for C1 — [arXiv 2609.14507](https://arxiv.org/abs/2609.14507)
- **2609.13537 VAMP** (11 Sep; vLLM HBM expert↔KV repartitioning, Qwen3-Next-80B, server-class) — [arXiv 2609.13537](https://arxiv.org/abs/2609.13537)
- **2609.13592 BOOST** (11 Sep; GH200 concurrent host+HBM access) — [arXiv 2609.13592](https://arxiv.org/abs/2609.13592)
- **2609.15636 HBF "MoE in a Box"** (14 Sep; simulation DSE) — [arXiv 2609.15636](https://arxiv.org/abs/2609.15636)
- **2609.18063 Edge0** (16 Sep, v2 17 Sep; lossy trained routing, Mac mini) — [arXiv 2609.18063](https://arxiv.org/abs/2609.18063)
- **2609.14643 BigMoMo** (13 Sep; mobile flash + speculative decoding) — [arXiv 2609.14643](https://arxiv.org/abs/2609.14643)
- **2608.14333** (14 Aug; HBF direct paths, simulator) — [arXiv 2608.14333](https://arxiv.org/abs/2608.14333)
- **2609.04895** Cache-aware router adaptation (4 Sep, v2 10 Sep). A trace-driven simulator only; per earlier notes, it establishes "neither wall-clock speedup nor energy savings". This supports C1/C5 motivation rather than overlapping — [arXiv 2609.04895](https://arxiv.org/abs/2609.04895)
- **2609.22156 "The Limits of Speculation: Bounding Speculative Decoding in MoE"**. v1 26 Aug; announced Tue 22 Sep 2026. It is a speculative-decoding SSP oracle on an A100 80GB, with no offload — [arXiv 2609.22156](https://arxiv.org/abs/2609.22156)
- **2609.22471** (Apple). v1 18 Sep; announced 22 Sep. It studies router coactivation for speculative decoding with DRAM→NPU transfers — [arXiv 2609.22471](https://arxiv.org/abs/2609.22471)
- **2609.17940** (15 Sep). Residual predictive structure in OLMoE/JetMoE routing; routing predictability only — [arXiv 2609.17940](https://arxiv.org/abs/2609.17940)
- **2608.16947** (v1 15 Aug, v2 29 Aug). A deterministic O(1)-competitive replica allocation for multi-GPU dynamic MoE serving. It "does not include network topology … or routing decisions", and is not caching — [arXiv 2608.16947](https://arxiv.org/abs/2608.16947)
- **2608.26612 "Launch-Bound and Substitutable"** (27/28 Aug). A100-resident; measures "a measured 1.07x ceiling" for fused kernels; no offload — [arXiv 2608.26612](https://arxiv.org/abs/2608.26612)
- **2608.23841 "Pipeline-Native Transformers"** (24 Aug). CPU-only decode with a roofline. "cflow decodes at 5.94 tokens/s … ahead of llama.cpp (4.75)" on dense comparators, with a disk-expert overlap model "within 1%" — [arXiv 2608.23841](https://arxiv.org/abs/2608.23841)
- **2609.30854 "The KV Cache Is the New Memory Wall"** (SoK). v1 25 Sep; announced Mon 28 Sep 2026.
  - KV cache, datacenter GPUs, no MoE offload.
  - It is a methodological analogue for C3/C5: "This SoK paper unifies the field analytically, with a protocol that strictly separates derived and reported claims." — [arXiv 2609.30854](https://arxiv.org/abs/2609.30854)
- Other August/September items with no relation to our contributions:
  - 2609.06172 AutoUVM (5 Sep; UVM prefetch, "roofline-inspired policy") — [arXiv](https://arxiv.org/abs/2609.06172)
  - 2609.17008 FlexEE (15 Sep; dense early exit under offloading) — [arXiv](https://arxiv.org/abs/2609.17008)
  - 2609.17475 JustFit (15/20 Sep; MLX state management) — [arXiv](https://arxiv.org/abs/2609.17475)
  - 2608.08081 RotaryQuant (8 Aug; LRU expert paging to disk on Metal) — [arXiv](https://arxiv.org/abs/2608.08081)
  - 2608.08910 Tied Trit-Planes (9 Aug; SSD expert streaming on a laptop CPU) — [arXiv](https://arxiv.org/abs/2608.08910)
  - 2608.14385 DeaMoE (14 Aug; architecture) — [arXiv](https://arxiv.org/abs/2608.14385)
  - 2608.05303 EdgeXpert (5 Aug; MICRO 2026 ASIC) — [arXiv](https://arxiv.org/abs/2608.05303)
  - 2609.09241 (8 Sep; dynamic top-k distribution alignment) — [arXiv](https://arxiv.org/abs/2609.09241)
  - 2609.05228 ACE (4 Sep; expert skipping) — [arXiv](https://arxiv.org/abs/2609.05228)
- **2608.30877 and 2609.17620** (Xiao & Xu, 31 Aug and 14 Sep). These are "DeepSeek 175B"/"trillion-parameter" deployments on an RTX 4060 laptop. They claim "100x throughput of an 8-card A100 cluster baseline". The claims are implausible and give no method details; low credibility; no overlap — [arXiv 2608.30877](https://arxiv.org/abs/2608.30877); [arXiv 2609.17620](https://arxiv.org/abs/2609.17620)

**Known prior art: version check (none updated after 25 Sep)**

- FreeToken 2608.16157: still v1 (17 Aug) — [arXiv](https://arxiv.org/abs/2608.16157)
- 2608.07911: last is v4 (25 Aug, metadata only) — [arXiv](https://arxiv.org/abs/2608.07911)
- 2608.12103: v2 30 Aug — [arXiv](https://arxiv.org/abs/2608.12103)
- 2608.18261: v1 only (18 Aug) — [arXiv](https://arxiv.org/abs/2608.18261)
- Single versions only: DALI 2602.03495, Speculating Experts 2603.19289, TriMoE 2603.01058, HybriMoE 2504.05897, 2512.16473, MoE-Lens 2504.09345 — [arXiv DALI](https://arxiv.org/abs/2602.03495)
- Fate: last is v2 (May 2025) — [arXiv](https://arxiv.org/abs/2502.12224)
- ProMoE: last is v3 (Sep 2025) — [arXiv](https://arxiv.org/abs/2410.22134)

### Inferences
- The main novelty risk is Budgeting Bytes (2609.04238), because a reviewer can read it as "a roofline/bound for offloaded MoE decode plus a tok/s predictor that already reasons about `--n-cpu-moe`". What separates us:
  - Its bound is a hiding-feasibility / exposed-latency bound for a *given* fetch set. It is not a lower bound over *all* exact-routing caching policies, and it has no miss minimisation (no MIN, no bypass, no per-layer vs shared budget).
  - Its validation is one published MCU deployment plus its own pipelines, not a 52-row third-party corpus.
  - It has no audit and no trace-provenance study.
  - It should be cited in C1/C2 related work, with a sentence that its model is a special case: the zero-cache, all-miss floor, "no schedule beats bytes/B".
- "Paging the Experts" helps our framing. It computes exactly the block-count Belady bound our title positions against, then disclaims it: "not an online speed prediction". Quote it as evidence that the literature stops at blocks.
- WiSP turns miss counts into seconds with c_exp = M_e·t_byte. That is the same accounting primitive as our load accounting, but applied to LRU (an estimator), not MIN-with-bypass (a bound), and not validated. Cite it as the closest "seconds" precedent, and note that it has no Jensen/overlap step and no CPU-execution path.
- None of the three flips C3's headline. None of the new systems benchmarks against llama.cpp `--n-cpu-moe`: WiSP uses vLLM `--cpu-offload-gb`, SAEM/SPICE/APEX use academic baselines, and SSD-LLaMA gives no flags. Budgeting Bytes *mentions* `--n-cpu-moe` and quotes a "∼30 tok/s on 6 GB VRAM + 32 GB RAM" figure, but reports no speed-up against it on a GPU.
  - Adjust the C3 wording from "no paper … with a llama.cpp baseline configured --n-cpu-moe" to "no speed-up claim is measured against …". Otherwise the Budgeting Bytes mention could be cited against it.
- WiSP's ~6.5–6.8 tok/s for Qwen3-30B-A3B on an RTX 3090 at "2.0× over static offload" is a strong new audit row. A predicted `--n-cpu-moe` baseline on that card would very likely exceed it. This is an inference; our model should compute the number.

### Gaps
- The first public announcement day of 2609.04238 (stamped 29 Jul, ID 2609) could not be retrieved; arXiv OAI and the API were rate-limited or empty.
- Semantic Scholar and Google Scholar could not be queried (HTTP 429), so non-arXiv venues (OpenReview ICLR 2027 submissions, workshop PDFs, IEEE/ACM early access) are not covered.
- Papers submitted on 28 Sep that appear in the 29 Sep listing were not visible.
- 2609.06551 (EStream, mobile NPU prefill) was dismissed on its title only.

## Q2. Did FreeToken publish updates, benchmarks or comparisons after 25 Sep 2026? Did KTransformers, SGLang, vLLM or llama.cpp add or discuss expert caching / prefetch / partial expert offload in September 2026?

### Takeaway
- FreeToken published no new benchmarks or comparisons after 25 Sep. Its last main commit is ROCm RDNA3/4 support (25 Sep 17:49 PDT). The arXiv paper is still v1.
- KTransformers, SGLang and vLLM merged no GPU expert cache or expert prefetch in September. Their offload work was KV/PLE/CPU-kernel work.
- llama.cpp master (26–28 Sep) added nothing for expert caching. The one new relevant-looking PR, #29599 `llama_prefetch_rows` (28 Sep), prefetches per-layer token-embedding (PLE) rows, not experts. The previously known expert-cache PRs (#27861, #28414, #26414, #28545) have no new heads after 24 Sep.

### Cited Findings
- **FreeToken main head**: `0d652e7`, "feat(rocm): add RDNA3 and RDNA4 runtime foundation (#132)", committed 2026-09-25 17:49:45 −0700. No later commit is on any of the ~45 branches; the newest non-main branch tips are `split/row-store` and `split/dsv41-fixes` at 2026-09-24 — [commit 0d652e7](https://github.com/FlashML-org/FreeToken/commit/0d652e73a452d014ac5441a15baa75348e9fcb0a)
- **FreeToken, September changes under `benchmarks/`**: only `bench_load_weight_generic.py`. README/docs changes were models/CLI/install docs. The README's only llama.cpp mention is the acknowledgements ("learned the design and reused code from … llama.cpp"). There are no comparison tables — [FreeToken repo](https://github.com/FlashML-org/FreeToken)
- **FreeToken commit `cee23bf`** (2026-09-23): "fix(runtime): correct overlap token limits and cache safety (#546)". It includes "fix(moe): initialize slots used by inactive hybrid routes" and touches `python/freetoken/moe/offload_cache.py` and the scheduler. It is relevant to pinning the FreeToken version used in C6 — [commit cee23bf](https://github.com/FlashML-org/FreeToken/commit/cee23bf44a86e48d70f8fda3ca3b6d4d9890072e)
- **FreeToken release 0.1.3**: tagged via "chore(release): 0.1.3 (#489)" on 2026-09-15 — [FreeToken repo](https://github.com/FlashML-org/FreeToken)
- **KTransformers**: head `c40722b` (23 Sep, "fix(kt-kernel): cap compressed-tensors below the torch>=2.10 releases"). September work was CPU-side kernels:
  - "make AVX512 MXFP4 decode memory-bound (#2181)" (1 Sep)
  - "packed-int4-resident AVX-VNNI-256 GPTQ INT4 backend (#2187)" (7 Sep)
  - "Ascend NPU backend for single-card DeepSeek-V4-Flash offload (#2157)" (1 Sep)
  - No GPU expert cache or prefetch — [commit c40722b](https://github.com/kvcache-ai/ktransformers/commit/c40722bf04c494f2492b7eb9e86ef01a4ede45b3); [commit 3a19fd6](https://github.com/kvcache-ai/ktransformers/commit/3a19fd6d88fe3895e702d6952b2977b9f35b36c4)
- **SGLang**: offload-related September commits were HiCache (KV) prefetch fixes, PLE embedding offload, and "Support GLM-5.3-Flash hybrid attention CPU offload and PD index mapping (#40310)" (21 Sep). Commits mentioning "expert" were weight-loading fixes only — [commit 00986c8](https://github.com/sgl-project/sglang/commit/00986c81be687d82e66651fb3ce9a096070b06e6)
- **vLLM**: September offload commits were KV-offload connectors and PLE CPU offload, e.g. "[Qwen4Exp][ROCm] PLE n-gram table CPU offload (#57497)" (24 Sep). Commits mentioning "expert" were weight-loading fixes only — [commit 267eee5](https://github.com/vllm-project/vllm/commit/267eee54bf232e4008a7e0b510f19a62e23aaaca)
- **llama.cpp master, 26–28 Sep**: head `fc07d78` (2026-09-29 10:19 +0800). No commit touches expert caching, expert prefetch, `--n-cpu-moe` or MoE offload — [llama.cpp commits](https://github.com/ggml-org/llama.cpp/commits/master)
- **llama.cpp PR #29599 "llama: llama_prefetch_rows"** (Aman Gupta; head 2026-09-29 01:33 +0800 = 28 Sep 17:33 UTC).
  - It adds `madvise(MADV_WILLNEED)` over host pages for gathered rows.
  - It is wired only to `per_layer_tok_embd` (PLE) in `gemma4.cpp`/`qwen4exp.cpp`, not to expert tensors. No overlap — [PR #29599](https://github.com/ggml-org/llama.cpp/pull/29599)
- **Other new llama.cpp PR**: #29609 "ggml-cuda: sanitize post-bias NaNs in MoE selection" (28 Sep) is a correctness fix — [PR #29609](https://github.com/ggml-org/llama.cpp/pull/29609)
- **Known llama.cpp expert-cache PRs, current heads** (from `refs/pull/N/head`; no new heads after 24 Sep):
  - #27861 "MoE expert cache: GPU-resident LRU cache for host-offloaded expert weights" — 2026-08-28
  - #28414 "--prefetch-experts-slots - lookahead H2D prefetch of host-resident MoE experts" — 2026-09-04
  - #28545 "experimental mlock pinning of hot MoE expert slices" — 2026-09-07
  - #26414 "--pin-hot-experts" — 2026-09-24
  - #29539 "LLAMA_MOE_RANDOM_ROUTING=1 for benchmark-only random expert selection" — 2026-09-27
  - Sources: [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861); [PR #28414](https://github.com/ggml-org/llama.cpp/pull/28414); [PR #26414](https://github.com/ggml-org/llama.cpp/pull/26414); [PR #29539](https://github.com/ggml-org/llama.cpp/pull/29539)
- **exllamav3 PR #341** "MoE CPU offload: zero-copy streamed prefill from a page-locked shared arena (~2x long-context prefill)" (Denys Ashikhin). Commits run to 2026-09-13, including a Windows pinned-arena backend. It is about prefill, not a decode bound — [PR #341](https://github.com/turboderp-org/exllamav3/pull/341)
- **truespar/paddock (new engine, repo started 3 Sep 2026)**:
  - Commit `43276ce` (2026-09-04), "engine: MoE expert offload - host-mapped expert planes with a VRAM slot cache", merged as PR #9 on 2026-09-05 — [commit 43276ce](https://github.com/truespar/paddock/commit/43276ce0053b9c7dd69f3f7cfdaa7159ddb5d070)
  - Issue #8 reports Qwen3.6-35B-A3B on a 16 GB RTX 5060 Ti at "62 tok/s" with the cache vs "9 tok/s" zero-copy only, with "12.6 GB/s measured for the fill pattern" on PCIe 4.0 x8. It notes "spec loses on a PCIe-bound cache at this size" — [paddock issue #8](https://github.com/truespar/paddock/issues/8)

### Inferences
- There is no new FreeToken material to reconcile with C6. The version to pin is main at or after `cee23bf`/`0d652e7`, because `cee23bf` changed `moe/offload_cache.py` two days before our 28 Sep runs.
- paddock is a new practitioner engine with a GPU expert slot cache on consumer cards (September). It is not prior art for any contribution, but it is a candidate C3 row or race entrant.

### Gaps
- FreeToken GitHub issues and discussions, SGLang/vLLM open PRs, and llama.cpp Discussions could not be listed: the GitHub API was blocked and robots.txt blocked pull-list pages. Only merged commits and PR head refs were checked.
- vLLM RFC #38256 ("Incremental MoE Expert Offloading — GPU Cache + Async Pipeline") surfaced in search, but its date and activity could not be checked — [vLLM #38256](https://github.com/vllm-project/vllm/issues/38256)

## Q3. Has anyone published a same-machine comparison of these engines, or reported systems as a fraction of a hardware bound?

### Takeaway
Nothing public matches C6. The closest items are:
- an informal llama.cpp issue (#27685, late August) timing FreeToken vs llama.cpp on one RTX 5090 box, with undocumented flags and no equal-memory control;
- papers that compare themselves on the same machine against one weak baseline (WiSP vs vLLM static offload; SSD-LLaMA vs llama.cpp with flags unstated).

Nobody reports several engines as a fraction of a time-domain decode bound. The nearest is bandwidth utilisation: SSD-LLaMA's "77.7% of peak SSD bandwidth", and Budgeting Bytes' "pinned at the eMMC bandwidth ceiling".

### Cited Findings
- **llama.cpp issue #27685 "Research: FreeToken is faster than llama.cpp"**, opened in the week of 24–31 Aug 2026 (it is listed in the Buttondown weekly report for that week).
  - According to a summary of the issue page (not a verbatim quote): RTX 5090 (32 GB) + 128 GB RAM, Qwen 3.8-flash-next Q4, llama.cpp at 22 t/s vs FreeToken at 53 t/s. GPU utilisation was about 20% for llama.cpp vs 100% for FreeToken.
  - llama.cpp flags, FreeToken config and memory parity were not documented.
  - Commenters (am17an) pointed to PR #21067 for prefill.
  - Sources: [llama.cpp #27685](https://github.com/ggml-org/llama.cpp/issues/27685); [Buttondown weekly report, Aug 24–31 2026](https://buttondown.com/weekly-project-news/archive/weekly-github-report-for-llamacpp-august-24-2026-5588/)
- **WiSP**: a same-machine, iso-VRAM comparison only against vLLM `--cpu-offload-gb` on a 24 GiB RTX 3090: "comparisons are iso-VRAM: the arms differ only in where the same bytes go" — [arXiv PDF 2606.21868v2](https://arxiv.org/pdf/2606.21868v2)
- **SSD-LLaMA**: reports "77.7% of peak SSD bandwidth vs 43.2% baseline" (per earlier notes) and a 1.35–3.44× speed-up over llama.cpp on the same PC, with the llama.cpp flags unstated — [arXiv 2609.18110](https://arxiv.org/abs/2609.18110)
- **Budgeting Bytes**: "the device already reads at ∼277 MB/s—the eMMC's sequential ceiling—so it is bandwidth-bound". This is a single-engine statement about llama.cpp on an RK3588 — [arXiv PDF 2609.04238v1](https://arxiv.org/pdf/2609.04238)
- **Snack on AI FreeToken post** (Mohinish S, 24 Aug 2026): per the page summary, it does not compare FreeToken, llama.cpp and KTransformers on one machine and does not normalise to a bound. It quotes FreeToken's own "q* = m·Bp/Bh" split rule — [Snack on AI](https://www.snackonai.com/p/freetoken-or-why-your-laptop-was-never-the-bottleneck)
- **The KV SoK (2609.30854)**: evaluates methods "under a single protocol" and finds gains "approaching the roofline bound". This is for KV-cache techniques on H100/B200/MI300X, not MoE offload engines — [arXiv 2609.30854](https://arxiv.org/abs/2609.30854)

### Inferences
- C6 is still unoccupied: one client driving llama.cpp, a llama.cpp GPU expert cache and FreeToken at equal GPU expert memory, with each system reported as a percentage of a seconds-domain bound.
- #27685's 2.4× FreeToken-over-llama.cpp figure is the kind of unflagged comparison C3/C5 critique. If our C6 numbers differ, the paper can cite #27685 as the informal claim being tested. This is a suggestion, not a finding.

### Gaps
- r/LocalLLaMA could not be searched directly (no Reddit fetch). Search engines returned no September 2026 Reddit threads with multi-engine same-machine tables, but coverage is weak.
- The exact open date and author of #27685 were not retrievable (the GitHub HTML summary omitted them).

## Q4. For each hit: what exactly overlaps which contribution, and what remains ours?

### Takeaway
Every contribution survives. Only C1 and C2 have partial overlaps, all of which must be cited:

| Contribution | Status | Why |
|---|---|---|
| C1 | Partial | Budgeting Bytes (exposed-latency bound; bytes/B floor), Paging the Experts (Belady miss bound in blocks), WiSP (LRU misses × per-miss PCIe seconds) |
| C2 | Partial | Budgeting Bytes (bytes-per-token tok/s planner), Qiu 2609.14864 (known) |
| C3 | Intact | No new audit exists; the audit gains new rows |
| C4 | Intact | Nobody contrasts teacher-forced dataset traces with the model's own generations |
| C5 | Intact | Only methodological analogues (KV SoK, Paging the Experts' measurement-scoping language) |
| C6 | Intact | Only an informal GitHub issue |

Separate from the literature: the project's own GitHub repository is publicly readable and indexed by web search. This matters for "not yet public" and for double-blind review.

### Cited Findings

| Item (public date) | C1 | C2 | C3 | C4 | C5 | C6 | Overlapping sentence | What remains ours |
|---|---|---|---|---|---|---|---|---|
| 2609.04238 Budgeting Bytes (v1 stamp 29 Jul; announced Sep 2026) | partial | partial | no (mentions `--n-cpu-moe`, no audit) | no | no | no | "yielding a closed-form windowed roofline: a full-hiding criterion and exposed-latency bound via EDF optimality"; "no schedule beats bytes/B"; Strata "reports predicted tok/s for all-in-RAM / static-offload (llama.cpp -ot) / predict-prefetch" | MIN-with-bypass miss minimisation over all exact-routing policies; per-layer vs shared budgets; CPU-executed misses; Jensen step; GPU-free teacher-forced traces; 52-row third-party validation and pre-registration; 41-system audit; provenance; contract; C6 |
| 2609.29032 Paging the Experts (v1 24 Sep; announced 25 Sep) | partial (blocks only) | no | no | adjacent | adjacent | no | "The relaxed unpinned oracle is a lower miss bound under equal-size, mandatory-admission paging, not an online speed prediction." | Seconds conversion; bypass; PCIe/CPU paths; everything in C2–C6 |
| 2606.21868v2 WiSP (v2 30 Aug) | partial (seconds form, LRU) | partial (form) | no (new row; vLLM baseline) | no | no | no | "An expert miss transfers weights over PCIe, with cost c_exp = M_e t_byte … T(f) = c_exp m_exp(C) + c_kv m_kv(κ)"; "the bottleneck is PCIe bandwidth, not prediction quality" | A *lower bound* (MIN, not LRU); validation; audit; provenance; contract; C6 |
| 2608.21614 SAEM (21 Aug) | no | no | row | adjacent (calibration match 1.33→1.54×) | no | no | "rising to 1.54x when calibration data matches the workload" | All |
| 2608.21240 SPICE (v1 21 Aug, v2 7 Sep) | no | no | row (lossy) | no | no | no | "low-confidence misses are approximated … (LoRE) surrogates" | All |
| 2609.18110 SSD-LLaMA (16 Sep) | no | no | row (llama.cpp flags unstated) | no | no | no | — | All |
| 2608.15018 S2-MoE, 2608.11688 APEX, 2608.02989 AcceptMoE, 2608.22643 NeuroPrefetcher (Aug) | no | no | rows / out of scope | no | no | no | — | All |
| 2609.14864 Qiu (14 Sep; known) | no | partial (known) | no | no | no | no | "Leave-one-host-out coefficients fitted on the other two systems yield 11.6%, 16.8%, and 36.0% test MAPE." | Third-party corpus; offload; bound |
| 2609.30854 KV SoK (25/28 Sep) | no | no | adjacent method | no | adjacent method | no | "a protocol that strictly separates derived and reported claims" | All (different domain) |
| llama.cpp #27685 (week of 24–31 Aug) | no | no | no | no | no | partial (informal same-box timing) | llama.cpp 22 t/s vs FreeToken 53 t/s on RTX 5090 (summary) | Equal memory, single client, flags, bound |
| llama.cpp PR #29599 (28 Sep) | no | no | no | no | no | no | PLE row prefetch only | — |
| FreeToken repo after 25 Sep | no | no | no | no | no | no | ROCm support only | — |

Sources: [2609.04238](https://arxiv.org/abs/2609.04238); [2609.29032](https://arxiv.org/abs/2609.29032); [2606.21868](https://arxiv.org/abs/2606.21868); [2608.21614](https://arxiv.org/abs/2608.21614); [2608.21240](https://arxiv.org/abs/2608.21240); [2609.18110](https://arxiv.org/abs/2609.18110); [2609.14864](https://arxiv.org/abs/2609.14864); [2609.30854](https://arxiv.org/abs/2609.30854); [llama.cpp #27685](https://github.com/ggml-org/llama.cpp/issues/27685); [PR #29599](https://github.com/ggml-org/llama.cpp/pull/29599); [FreeToken commit 0d652e7](https://github.com/FlashML-org/FreeToken/commit/0d652e73a452d014ac5441a15baa75348e9fcb0a)

**Public status of our own project**:
- A web search for `"speed of light" MoE decode offload PCIe bound` returned "GitHub - harshithkantamneni/moe-speed-of-light" as its first result — [GitHub repo](https://github.com/harshithkantamneni/moe-speed-of-light)
- An anonymous `git ls-remote` on 28 Sep 2026 returned HEAD `092d50ad` and branches `gpu`, `gpu-heartbeat`, `gpu-heartbeat-a10` and `gpu-heartbeat-a100`, so the repository is publicly readable — [GitHub repo](https://github.com/harshithkantamneni/moe-speed-of-light)

### Inferences
- **C1 wording.** Say explicitly that prior bounds are either:
  - block/miss-count bounds that disclaim speed (Paging the Experts);
  - exposed-latency bounds for a *given* prefetch schedule (Budgeting Bytes, via EDF);
  - or LRU-based seconds estimators (WiSP).
  None gives a policy-independent lower bound in seconds over exact-routing policies with per-layer/shared budgets and CPU-executed misses. That sentence is defensible from the quotes above.
- **C2 wording.** Budgeting Bytes validates against one published MCU deployment. "Validated on 52 third-party published measurements (MoE offload)" should stay qualified with "MoE-offload" and "leave-one-group-out" to keep it distinct.
- **C3 wording.** Change it to "no published speed-up is measured against an equal-memory `--n-cpu-moe` baseline", because Budgeting Bytes discusses `--n-cpu-moe` in text. Add WiSP, SAEM, SPICE and SSD-LLaMA as audit rows if the audit is re-run.
- **Public repository.** Check before submission what the public repository exposes (paper draft? results?). A public, indexed repository whose name matches the title undermines double-blind anonymity and means the work is already partly public. This is an inference; the repository contents were not inspected.

### Gaps
- The contents of the public `moe-speed-of-light` repository were not inspected: which branch holds the paper, and whether results are visible.
- Whether any journal or conference versions (not on arXiv) of these works appeared in September could not be checked (Semantic Scholar/Scholar unavailable).
