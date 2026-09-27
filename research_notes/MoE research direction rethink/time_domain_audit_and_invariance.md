# Time-domain audit of MoE-offloading claims, and "placement invariance" in hybrid CPU/GPU MoE runtimes (state as of 27 Sep 2026)

Method note:
- Hardware, baseline and speedup details come from full-text greps (pdftotext) of each paper's PDF, fetched from arxiv.org/pdf, and from the SOSP'25 PDF for KTransformers.
- Determinism papers were fetched as PDFs and grepped for CPU / offload / hybrid / llama.cpp / MoE.
- llama.cpp statements come from the GitHub pages (WebFetch summaries, quoted), plus a local llama.cpp clone at commit 2145525a (26 Sep 2026) for source-level facts.
- "Not stated" means a keyword grep of the full text found nothing. It is not proof of absence.
- The arXiv API (export.arxiv.org) returned empty bodies from this environment, so arXiv discovery used the arxiv.org/search listing pages and web search.

## Q1. Has anyone published a cross-paper, time-domain, bound-normalized reanalysis or baseline audit of MoE-offloading / hybrid decode claims, or of LLM-inference speedups generally?

### Takeaway
Verdict: **open for the specific contribution; partially open for the framing.**
- No paper found re-derives published MoE-offloading or hybrid batch-1 decode numbers from many systems with one bytes-over-bandwidth model, normalizes them to a hardware speed-of-light bound, or audits baseline strength (`-ngl` vs `--n-cpu-moe`/`-ot` at equal VRAM).
- The closest prior art, 2608.07911 (Yu Zhang; v4 25 Aug 2026), already stakes out the "MoE caching lacks a common evaluation contract" framing and audits 10 papers. It stays in block counts, lists real-hardware validation as its main missing piece, and explicitly declines to relabel any published speedup.
- Nothing newer in Sep 2026 closes the gap. Recent "audit" papers target other techniques (layer skipping, 2608.28846) or accuracy and reproducibility (2605.19537, 2608.04714, 2609.04198).

### Cited Findings
**Closest prior art (MoE-specific)**
- 2608.07911, "Reproducible Evaluation of MoE Expert Caching: Replay Semantics, Workload Contamination, and Operating Regimes" (Yu Zhang, single author).
  - Dates: v4 is 25 Aug 2026. v1 was early Aug 2026; companion notes give 4 Aug and 8 Aug, which conflict. — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)
  - It audits ten papers (SpecMD, DALI, Fate, Pre-gated MoE, SiDA-MoE, MoE-Beyond, Mixtral-Offloading, HOBBIT, MoE-Infinity, SP-MoE) and publishes a reporting checklist.
  - It states "every result here is a block count, and the step to service quality passes through overlap, topology and contention we do not model", and names "real-hardware validation" as the limitation it would most want removed.
  - It states that the audit "does not show that any cited speedup or quality result is wrong" and "does not support re-labelling published speedups as artifacts". — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911) (quotes via companion notes benchmarking_and_offpolicy.md / academic_systems.md, which read the v4 PDF)
- 2608.18261, "Cacheable by Design?" (Suram), 18 Aug 2026, pre-registered:
  - Single RTX 3070 8GB serving Qwen3-235B-A22B Q4_K_M with "llama.cpp (-ngl 99 -ncmoe 94)".
  - "measured decode is 0.44 tok/s warm, in exact agreement with a bytes-per-token ÷ bandwidth model".
  - That is one configuration. It is not a cross-paper audit. — [arXiv 2608.18261](https://arxiv.org/abs/2608.18261)
- MoE-CAP (NeurIPS 2025) benchmarks systems the authors run themselves, using a single-peak-bandwidth utilization metric (S-MBU). It does not reconcile other papers' published numbers (per companion notes). — [arXiv 2412.07067](https://arxiv.org/abs/2412.07067)
- FreeToken models branch times as bytes over measured bandwidths (T_fill ≈ qS/B_P; CPU ≈ (m−q)S/(B_H−B_P)) for its own scheduler. It does not bound or audit others. — [arXiv 2608.16157](https://arxiv.org/abs/2608.16157)

**Recent "audit"-type papers found (none is a time-domain cross-paper MoE audit)**
- 2608.28846 (Prateek Kumar Sikdar, 28 Aug 2026), "A rigor-matched audit of periodic-step layer skipping for efficient LLM inference: ConfLayers versus SWIFT":
  - A three-seed re-run of two layer-skipping methods on Qwen2.5-0.5B/1.5B. It finds that "SWIFT's true inference speed is faster than ConfLayers's in all four cells (5-21%), reversing the naive wall-clock ranking in three".
  - It re-runs methods empirically. It uses no bounds, no bandwidth model and no MoE. — [arXiv 2608.28846](https://arxiv.org/abs/2608.28846)
- 2609.04198 (3 Sep 2026), "Clean Engineering, Unstable Measurement": a pre-registered reliability audit of LLM judges on shared endpoints. It is about measurement validity, not speed. — [arXiv 2609.04198](https://arxiv.org/abs/2609.04198)
- "The Silent Hyperparameter" (Pape et al., 2605.19537, v2 20 May 2026) covers backend-induced accuracy variation, up to 16.6 pp, across vLLM, SGLang, llama.cpp and others. The target is accuracy, not speed. — [arXiv 2605.19537](https://arxiv.org/abs/2605.19537)
- Masoudian et al. (2608.04714, 5 Aug 2026): "What We Observe as LLM Behavior Can Be a Side-effect of Inference Backend". Accuracy and behaviour only. — [arXiv 2608.04714](https://arxiv.org/abs/2608.04714)
- Other search hits are characterizations or benchmarks, not reanalyses of published claims:
  - "Prefill/Decode-Aware Evaluation of LLM Inference on Emerging AI Accelerators" (2606.17104) — [arXiv 2606.17104](https://arxiv.org/html/2606.17104v1)
  - InferenceBench (2607.20468), a benchmark for AI agents doing inference optimization — [arXiv 2607.20468](https://arxiv.org/html/2607.20468)
- The systems-security "Benchmarking Crimes" paper (2018) is the classic genre precedent. Searches found no LLM-inference or MoE counterpart. — [arXiv 1801.02381](https://arxiv.org/abs/1801.02381)

**Evidence inside system papers that baselines are weak, or that llama.cpp is stronger than the headline suggests**
- KTransformers (SOSP'25): "Since Llama.cpp originally only supports layer-wise offloading and lacks expert-level offloading capability, to ensure fair comparisons, we extended Llama.cpp with custom code to enable expert-level offloading analogous to Fiddler". It also ran llama.cpp in FP16 because "Llama.cpp lacks BF16 CUDA kernels". — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- ProMoE: "when evaluating the Mixt model, the LO [llama.cpp layer offload] baseline is significantly faster and even surpasses ProMoE in the decode stage". — [arXiv 2410.22134](https://arxiv.org/abs/2410.22134)
- HybriMoE: "llama.cpp demonstrates relatively strong performance in this [decode] stage". Its llama.cpp baseline "statically maps model layers to CPU or GPU". — [arXiv 2504.05897](https://arxiv.org/abs/2504.05897)
- DALI configured llama.cpp and KTransformers as "layer-wise hybrid frameworks" and controlled "the number of MoE layers stored and executed on the GPU" for "comparable memory usage". Against that baseline it reports its largest gain, 3.97×. — [arXiv 2602.03495](https://arxiv.org/abs/2602.03495)
- FreeToken: "Single-stream benchmarks therefore overstate baseline agentic performance". It also reports that llama.cpp "keeps only 80% of its rate as its CPU-resident experts starve on two DDR5 channels" when moving from a many-channel server to a dual-channel desktop. That is direct evidence that host DRAM channels drive the baseline. — [arXiv 2608.16157](https://arxiv.org/abs/2608.16157)
- HOBBIT's 13.0× and 18.9× over llama.cpp on Jetson AGX Orin are attributed to llama.cpp mmap "severe page faults when there is insufficient CPU memory". On the RTX 4090 + CPU setup, HOBBIT is only 0.99× (Mixtral) and 1.46× (Phi-MoE) versus Fiddler. — [arXiv 2411.01433](https://arxiv.org/abs/2411.01433)
- 2512.16473 estimates its Pre-gated MoE baseline "by assuming perfect computation-communication overlap" because the implementation "does not support these newer models". — [arXiv 2512.16473](https://arxiv.org/abs/2512.16473)
- Speed-of-light and roofline framings exist, but only for each paper's own system or for throughput:
  - MoE-Lens ("theoretical performance upper bound", 94% accuracy, batched throughput) — [arXiv 2504.09345](https://arxiv.org/abs/2504.09345)
  - MoE-Lightning (HRM) — [arXiv 2411.11217](https://arxiv.org/abs/2411.11217)
  - CoX-MoE and TriMoE rooflines — [arXiv 2605.17889](https://arxiv.org/abs/2605.17889); [arXiv 2603.01058](https://arxiv.org/abs/2603.01058)
- Hoefler & Belli's SC'15 "Scientific Benchmarking of Parallel Computing Systems" rules include reporting upper performance bounds. None of the MoE-offloading papers checked reports achieved fraction of a hardware bound for batch-1 decode. — [SPCL publications](https://spcl.inf.ethz.ch/Publications/)

### Inferences
- The defensible gap is precise: **seconds, not blocks; many papers, not one config; bound-normalized; baseline-audited.** 2608.07911 is the paper to cite and differentiate. Its own stated missing piece ("real-hardware validation") is exactly what a time-domain audit supplies. The 2608.07911 author is active (v4 in 3 weeks), so a time-domain extension from that author is a real scoop risk over the next 1–3 months.
- Tone risk is real: 2608.07911 deliberately avoided saying speedups are wrong. The audit is safer framed as "fraction of speed-of-light achieved by system and by baseline". The speedup is then a ratio of two bound-fractions, which exposes weak baselines without calling anyone wrong.
- Data availability is itself a finding, and a limit. Only FreeToken (measured B_P and B_H per machine) and KTransformers (MLC-measured 220 GB/s) report the host bandwidth a bytes/bandwidth model needs. Most papers give only the CPU model, or "256GB DDR4" with no channels or speed. The audit will have to impute DRAM bandwidth for most rows, and should carry uncertainty bands.
- Baseline audit: among academic system papers, the llama.cpp baseline is layer offload (`-ngl`; Fiddler, HybriMoE, DALI, ProMoE's "LO", NVIDIA 2604.26334), custom-patched (KTransformers), or unspecified (FreeToken, SeqMoE, ATSInfer, SSD-LLaMA, MoE-Infinity "Ollama"). Only Fiddler reports a llama.cpp build (b2956). No academic system paper found reports `--n-cpu-moe`/`-ot exps=CPU` at equal VRAM. The only academic `-ncmoe` user is the 2608.18261 measurement study.

### Gaps
- Workshop papers (MLSys/NeurIPS/SC'26 workshops) are not systematically indexed. I found no workshop paper doing this audit, but coverage is not exhaustive.
- Companion notes (venue_bar.md) say NVIDIA's MLSys'26 "Efficient, VRAM-Constrained xLM Inference on Clients" (2604.26334) uses llama.cpp `-cmoe` as its baseline. My full-text grep found no "cmoe", "cpu-moe" or "n-cpu" string. The paper says instead that "Llama.cpp requires users to manually specify the number of GPU layers (-ngl parameter)" and speaks of "Finding the optimal ngl value". **This conflicts and should be re-checked in the PDF's figure captions or appendix.** — [arXiv 2604.26334](https://arxiv.org/abs/2604.26334)
- I could not confirm whether ATSInfer's "same GPU VRAM budget" llama.cpp baseline uses `-ngl` or `-ot`/`--n-cpu-moe`. The flags were not found by grep. — [arXiv 2607.10183](https://arxiv.org/abs/2607.10183)

## Q2. Audit input table: hardware, model/quant, llama.cpp baseline config, headline batch-1 decode speedup

### Takeaway
- Twelve academic systems plus the llama.cpp community PRs are tabulated below.
- DRAM channels and speed are stated for almost nothing academic. KTransformers (MLC-measured bandwidth), FreeToken (measured B_H) and DALI (DDR4, no channels) are the partial exceptions.
- PCIe generation is usually stated.
- The llama.cpp build is stated only by Fiddler (b2956). llama.cpp flags are stated by Fiddler (`ngl` 8/16), 2608.18261 and the community PRs.
- Headline speedups are mostly against non-llama.cpp baselines, or against layer-offload llama.cpp.

### Cited Findings
**Academic systems** (batch 1 unless noted; "n/s" = not stated in full text)

| System (venue, date) | GPU / PCIe | CPU / DRAM | Model(s), precision | llama.cpp baseline config | Headline decode speedup (vs what) | Source |
|---|---|---|---|---|---|---|
| Fiddler (ICLR'25; arXiv Feb 2024) | Env1 Quadro RTX 6000 24GB, PCIe Gen3 x16 ("32GB/s"); Env2 RTX 6000 Ada 48GB, Gen4 x16 ("64GB/s") | Env1 Xeon Gold 6126 ("48 core"); Env2 Xeon Platinum 8480+ ("112 core"); DRAM n/s | Mixtral-8x7B 16-bit; 56/256 and 125/256 experts on GPU | **llama.cpp b2956, `ngl`=8 (Env1), 16 (Env2)**, i.e. layer offload | "1.26 times speed up in single batch inference" (avg vs best baseline); appendix 1.81× ShareGPT / 1.56× LMSYS vs llama.cpp | [arXiv 2402.07033](https://arxiv.org/abs/2402.07033) |
| KTransformers (SOSP'25) | A100-40GB or RTX 4080-16GB, PCIe 4.0 (32 GB/s) | 2× Xeon Platinum 8452Y (36c each), 1 TB DDR5; MLC-measured 220 GB/s intra-socket, 125 GB/s cross-socket | DS-V3-0324, DS-V2.5-1210, Qwen2-57B-A14B; BF16 on A100 (llama.cpp FP16); Int4 (DS-3) / Int8 (DS-2, QW-2) on 4080 | **llama.cpp "extended … with custom code to enable expert-level offloading analogous to Fiddler"**; version n/s | 1.25–1.76× vs llama.cpp (full precision), 1.77–1.93× (quantized); 1.66–2.56× with Expert Deferral (which changes outputs); 2.42–4.09× vs Fiddler | [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf) |
| HybriMoE (DAC'25; arXiv Apr 2025) | RTX A6000; PCIe n/s | Xeon Gold 5220R "restricting usage to 10 cores"; DRAM n/s | Mixtral-8x7B-Instruct, DeepSeek-V2-Lite-Chat, Qwen2-57B-A14B-Instruct; Marlin 4-bit; GPU expert cache ratios 25/50/75% | llama.cpp "statically maps model layers to CPU or GPU" (layer offload); flags/version n/s | "average throughput improvement of 1.70×" vs kTransformers; vs llama.cpp only in Fig. 8 | [arXiv 2504.05897](https://arxiv.org/abs/2504.05897) |
| MoE-Infinity (arXiv Jan 2024, v3 Mar 2025) | RTX A5000 24GB; "PCIe 4.0 interface (32GB/s)" in §5. The abstract says "PCIe4.0 (24GB/s)", an internal inconsistency | CPU n/s; host memory 32 GB–1 TB by model | DeepSeek, Mixtral, Switch, NLLB, Arctic | "Llama.cpp (Ollama)", described as "By default, Llama.cpp stores all model parameters in CPUs and offloads computations to GPUs"; flags/version n/s | "3.1–16.7× per-token latency improvements" vs vLLM, Ollama, DeepSpeed, BrainStorm | [arXiv 2401.14361](https://arxiv.org/abs/2401.14361) |
| ProMoE (arXiv Oct 2024, v3 Sep 2025) | RTX 4090 24GB, PCIe 4.0 ("32 GB/s" unidirectional; motivation measures 23.9 GB/s on PCIe 4.0 x8) | i9-14900K, 128 GB DRAM (type/channels n/s) | DS-1, DS-2, QW-1 FP16; QW-2, Mixtral INT4; batch 1–4 | ProMoE built into llama.cpp; baselines are llama.cpp **LO (layer offload: "offloads both parameters and computations")**, static cache, LRU cache; version n/s | In llama.cpp: 1.49× vs static, 1.09× vs LRU (decode); 1.66× (up to 2.07×) vs baselines; LO "surpasses ProMoE in the decode stage" for Mixtral; abstract: decode 2.07× avg (up to 5.02×) | [arXiv 2410.22134](https://arxiv.org/abs/2410.22134) |
| AdapMoE (arXiv Aug 2024) | RTX 4090 (8x7B), A6000 (8x7B, 8x22B); PCIe n/s | n/s | Mixtral-8x7B (HQQ 4+2-bit and 4-bit), Mixtral-8x22B 4-bit | None (GPU-fetch system) | "average 1.35× vs. Mixtral-Offloading" | [arXiv 2408.10284](https://arxiv.org/abs/2408.10284) |
| HOBBIT (arXiv Nov 2024) | RTX 4090 24GB, PCIe 4.0 (32 GB/s); Jetson AGX Orin 32GB unified | 4090 host: "256GB of CPU memory, and 64 CPU cores" (type n/s); Orin: 12 CPU cores | Mixtral-8x7B, Phi-MoE; FP16 on 4090 (int4 miss replacement), int8 on Orin | llama.cpp on Orin (mmap page faults); flags/version n/s | 13.0× / 18.9× vs llama.cpp (Orin); up to 9.93× vs MoE-Infinity; on 4090+CPU 0.99× / 1.46× vs Fiddler | [arXiv 2411.01433](https://arxiv.org/abs/2411.01433) |
| DAOP (DATE'25; arXiv Dec 2024) | RTX A6000 48GB, PCIe 4.0 ("64GB/s") | i9-10980XE (18c @3.0GHz), 130 GB host memory (type/channels n/s) | Mixtral 8x7B, Phi-3.5-MoE | None | "outperforms Fiddler by 40.4%" (4.52 tok/s Mixtral); up to 8.20× vs caching/prefetching | [arXiv 2501.10375](https://arxiv.org/abs/2501.10375) |
| 2512.16473 (ASP-DAC'26; arXiv 18 Dec 2025) | RTX 4090 24GB, PCIe Gen 4.0 x16 | Threadripper 7960X 24-core; DRAM n/s | Mixtral 8x7B, Phi-3.5-MoE (no quantization stated) | None | "4.4× … Mixtral 8x7B and 4.3× for Phi3.5-MoE" vs prefetching; ~1.6× vs Fiddler; up to 4.8 / 10.4 tok/s | [arXiv 2512.16473](https://arxiv.org/abs/2512.16473) |
| DALI (arXiv 3 Feb 2026) | RTX 3090 24GB, PCIe 4.0 x16 | AMD EPYC 7532 64c (16 cores / 32 threads used), 256 GB DDR4 (channels/speed n/s) | DeepSeek-V2-Lite-Chat, Qwen3-30B-A3B, Mixtral-8x7B-Instruct; precision n/s; 50% cache ratio; prompt 64 / gen 64 | llama.cpp as "layer-wise hybrid" with MoE layers on GPU set for "comparable memory usage"; version n/s | Avg decode 3.97× vs llama.cpp, 2.16× KTransformers, 1.48× MoE-Lightning, 1.32× HybriMoE, **averaged over batch sizes** (not batch-1 only) | [arXiv 2602.03495](https://arxiv.org/abs/2602.03495) |
| FreeToken (arXiv 17 Aug 2026) | Six systems, measured B_P: 5090 server PCIe 5.0 x16 52.7 GB/s; 4090 4.0 x16 25.1; 3090 4.0 x16 25.3; 5090 desktop 5.0 x16 49.0; 4060 laptop 4.0 x8 11.8; PRO 6000 5.0 x16 51.5 | Measured B_H: 2× Xeon Gold 6459C, DDR5 180 GiB, 77.3; 2× Xeon Plat 8358P, DDR4 240, 63.2; 2× Xeon Gold 6330, DDR4 180, 56.7; Ryzen 9 9950X3D, DDR5 192, 53.8; i9-13900H, LPDDR5 32, 47.5; Xeon Plat 8559C, DDR5 512, 178. Servers capped at 6 threads and NUMA-pinned | Qwen3.6-35B-A3B BF16; DeepSeek-V4-Flash native MXFP4; GLM-5.2 (753B); 4060 laptop uses NVFP4 | "llama.cpp's routing-blind static split"; flags/version n/s | 77–83 tok/s Qwen3.6, 22–25 DSV4-Flash, "1.8–2.3× and 1.5–1.9× the strongest baseline" (agentic W1–W4 on 5090); W2 cross-hardware 1.3× (3090/4090), 1.9× (5090 server), 2.1× (5090 desktop), 1.8× (4060 laptop); GLM-5.2 on PRO 6000 14.9 vs llama.cpp 7.3 tok/s (2.0×) | [arXiv 2608.16157](https://arxiv.org/abs/2608.16157) |
| SeqMoE (arXiv 11 Sep 2026) | RTX 4090 24GB PCIe 4.0; RTX 5090 32GB PCIe 5.0; RTX PRO 6000 96GB PCIe 5.0 | Xeon Gold 6430, 120 GB (4090); Xeon Plat 8470Q, 120 GB (5090); 8470Q, 256 GB (PRO 6000); DRAM type/channels n/s | Qwen3-30B-A3B-FP8 (4090), Qwen3.6-35B-A3B-BF16 (5090), GPT-OSS-120B-MXFP4 and DSV4-Flash (PRO 6000); 45% expert residency | "Llama.cpp (Static Offload)", which "supports concurrent CPU–GPU computation"; flags/version n/s; FullLoad on vLLM 0.28.0 | Fig. 1 (Qwen3-30B-FP8, 4090): llama.cpp 30.9, KTrans 34.2, MoE-Inf 11.9, SeqMoE 104.1 (≈3.4× vs llama.cpp), FullLoad 122.7 tok/s; abstract: 96.97% hit rate, 80.22% of full-load | [arXiv 2609.12978](https://arxiv.org/abs/2609.12978) |

**Other 2026 academic papers with llama.cpp baselines, surfaced in this pass**

| System | Hardware | llama.cpp config | Headline | Source |
|---|---|---|---|---|
| 2608.18261 Suram (18 Aug 2026) | RTX 3070 8GB | **`-ngl 99 -ncmoe 94`** | 0.44 tok/s warm, matches bytes/bandwidth model (a measurement, not a speedup) | [arXiv 2608.18261](https://arxiv.org/abs/2608.18261) |
| ATSInfer 2607.10183 (v2 14 Jul 2026) | Laptop: i7-11800H, RTX 3060 6GB, 32 GB DDR4, Gen4 x16. Desktop: i7-11700, RTX 4090, 64 GB DDR4, Gen4 x16. Batch 1 | "Compared with llama.cpp under the same GPU VRAM budget"; flags n/s | decode "up to 3.29×" vs llama.cpp | [arXiv 2607.10183](https://arxiv.org/abs/2607.10183) |
| SSD-LLaMA 2609.18110 (16 Sep 2026) | PCIe 5.0 x16, consumer PC, SSD-tier experts | Implemented in llama.cpp; baseline flags n/s | "decode token rate by 2.10×–15.58× over the evaluated baselines"; "1.35×–3.44× speedup over llama.cpp" | [arXiv 2609.18110](https://arxiv.org/abs/2609.18110) |
| NVIDIA pipelined sharding 2604.26334 (MLSys'26) | Client GPUs, VRAM budgets 2G–24G | Baseline described via "-ngl"; see the conflict under Q1 Gaps | TPS speedups averaged over configs; "Exceptional TPS speedups (up to 30×)" | [arXiv 2604.26334](https://arxiv.org/abs/2604.26334) |

**llama.cpp community expert-cache PRs** (community claims, not peer reviewed)
- **RFC/Discussion #24528** (leloch, 12 Jun 2026):
  - Hardware: 4× RTX 3090, EPYC 7R13 48c, **8-ch DDR4-3200 256GB**.
  - Method: "llama-bench tg300, identical command lines both arms" with `--cpu-moe` and `-ngl`.
  - Reported gains: +25% GLM-5.1 IQ2_M; +7% Qwen3.5 397B; +10…+57% across 13 forced-offload models.
  - Independent reports: noonghunna (2× RTX 3090, 8-ch DDR4) and nibor1896 (RTX 5090, DDR5).
  - Sources: [Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528); companion notes llamacpp_community_caches.md.
- **PR #26563** (miltos22, Aug 2026), 8 GB VRAM, Qwen3.6-35B-A3B:
  - Q2_M: 33.25 → 57.2 tok/s (1.72×). Q5_K_P: 17.34 → 35.93 (2.07×).
  - On DeepSeek-V4-Flash a tester reported only parity with static layer pinning: "heat caching can't beat static layer pinning".
  - Regression reported: RTX 2000, 31 → 26 t/s.
  - Source: [PR #26563](https://github.com/ggml-org/llama.cpp/pull/26563)
- **PR #27861** (csantiago78, 28 Aug 2026), with `-ot …exps=CPU` / `-ncmoe`:
  - 2× RTX 3090 + dual-Xeon ("single RAM channel/socket"), Qwen3.8-Flash-Next UD-Q4_K_XL: 18.4 → 24.2 tok/s (+31%, 48 slots/layer).
  - RX 7600 8GB Vulkan, Ryzen 9 5900X, DDR4-3200, Qwen3-30B Q4_K_M: 14.4 → 16.5 tok/s.
  - Source: [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861)
- **PR #21609** (Apr 2026, closed): "96.7% hit rate, 1.79x decode speedup on GPT-OSS-120B with RTX PRO 2000 8GB (--n-cpu-moe 36 --expert-cache-slots 16)". — [PR #21609](https://github.com/ggml-org/llama.cpp/pull/21609)

### Inferences
- For a bytes/bandwidth re-derivation, the rows split into three tiers:
  - **Directly modelable** (bandwidths measured or specified): FreeToken (all six machines), KTransformers, #24528, 2608.18261.
  - **Imputable with assumptions** (CPU SKU known, so maximum channels and DDR generation can be looked up, but population unknown): Fiddler, HybriMoE, DALI, ProMoE, 2512.16473, DAOP, SeqMoE, ATSInfer.
  - **Weakly modelable**: AdapMoE and MoE-Infinity (no CPU stated), and HOBBIT on Orin (a page-fault regime, not bandwidth-bound).
- Many headline numbers are not batch-1 single-turn:
  - DALI averages over batch sizes.
  - FreeToken's headline is agentic multi-turn (W1–W4), but it has single-turn W1.
  - KTransformers' 1.66–2.56× includes Expert Deferral, which changes outputs.
  - The audit must pick the batch-1 single-turn number per paper, often from figures.
- Hardware-condition confounds visible in the table:
  - HybriMoE caps the CPU at 10 cores.
  - FreeToken caps servers at 6 threads.
  - DALI uses 16 of 64 cores.
  - A capped CPU weakens CPU-heavy baselines like llama.cpp `--n-cpu-moe` unless the capped bandwidth is the modelled quantity. FreeToken, to its credit, reports B_H measured under the cap.

### Gaps
- Exact batch-1 decode tok/s for HybriMoE, DALI, KTransformers and Fiddler per model are in figures only. They need digitizing from the PDFs, which was not done here.
- DRAM channel count and speed are not stated for Fiddler, HybriMoE, ProMoE, DAOP, 2512.16473, SeqMoE or DALI (DALI gives "256GB DDR4" only).
- The llama.cpp build or commit is not stated for any paper except Fiddler (b2956).
- MoE-Infinity's PCIe bandwidth is internally inconsistent (24 vs 32 GB/s).
- AdapMoE's venue (ICCAD'24?) is not verified.

## Q3. "Placement invariance": do runtimes that choose CPU vs GPU per expert, step or layer produce nondeterministic outputs, and has anyone documented, measured, claimed bitwise equivalence for, or fixed this?

### Takeaway
Verdict: **partially open, and effectively open as a research contribution.**
- The phenomenon has been **acknowledged anecdotally** in a llama.cpp PR (#26563, Aug 2026). Its author states that output at temp 0 varies with which experts are cached "due to differences in how experts are processed on the CPU vs the GPU", and that the same happens when changing `--n-cpu-moe`. No measurement is given.
- #24528 claims "decode-path perplexity statistically identical to pure CPU". That is a tolerance claim, not bitwise.
- FreeToken claims "exact MoE output without algorithmic approximation". That is algorithmic, not numerical.
- None of the 2025–2026 determinism papers (Thinking Machines Sep 2025; LLM-42; CoRun; MarginGate; Cooper et al.; 2609.11356; Hawkeye; Pape; Masoudian) treats CPU/GPU hybrid placement. None measures placement-induced divergence or builds bitwise-identical CPU/GPU quantized kernels.
- llama.cpp source confirms a structural root cause: the CPU and CUDA backends quantize activations to different formats (Q8_0/Q8_K vs q8_1). Hybrid hit/miss execution is therefore not bitwise placement-invariant by construction.

### Cited Findings
**Statements in hybrid-runtime artifacts**
- PR #26563 (miltos22, Aug 2026): "Due to differences in how experts are processed on the CPU vs the GPU, even at temp 0 when different experts are cached, the output may slightly vary. I have run extended tests and confirmed this is not silent corruption". The author notes the same occurs in stock llama.cpp when adjusting `--n-cpu-moe` (WebFetch summary). — [PR #26563](https://github.com/ggml-org/llama.cpp/pull/26563)
- Discussion #24528 (leloch):
  - 12 Jun 2026: "Quality: decode-path perplexity statistically identical to pure CPU; prompt/batch path bit-untouched; `test-backend-ops` MUL_MAT_ID 789/789 both modes."
  - 6 Aug 2026: "117,656-token retrieval exact; perplexity parity; four concurrent clients exact".
  - noonghunna (9 Aug 2026): "Sampled-decode confirms it's not a greedy artifact".
  - Source: [Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)
- PR #27861 (Aug 2026) claims only "Output coherent over thousands of tokens at 48-81% measured hit rates". There are no bit-identity, perplexity or KL claims. — [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861)
- FreeToken (17 Aug 2026):
  - "the CPU and GPU compute their respective partial sums and merge them, preserving the exact MoE output without algorithmic approximation".
  - "every engine consumes DSV4-Flash's native MXFP4 expert blocks bit-exactly", and GLM-5.2 is compared "with bit-identical expert weights".
  - It says "Agent trajectories diverge across engines so cross-engine wall-clock totals are not compared".
  - No statement on whether FreeToken's own outputs are bitwise stable across cache states or runs was found.
  - Source: [arXiv 2608.16157](https://arxiv.org/abs/2608.16157)
- KTransformers' Expert Deferral deliberately changes computation. It is evaluated for accuracy (HumanEval, MBPP, GSM8K, StrategyQA, LiveBench), a different category from numerical placement effects. — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- Greps of HybriMoE, DALI, 2512.16473, Fiddler, HOBBIT and SeqMoE for "bitwise / identical / numerical / exact same / lossless" found no numerical-equivalence claims about CPU vs GPU expert execution. — [HybriMoE](https://arxiv.org/abs/2504.05897); [DALI](https://arxiv.org/abs/2602.03495); [2512.16473](https://arxiv.org/abs/2512.16473); [Fiddler](https://arxiv.org/abs/2402.07033); [HOBBIT](https://arxiv.org/abs/2411.01433); [SeqMoE](https://arxiv.org/abs/2609.12978)

**Source-level root cause in llama.cpp** (local clone, commit 2145525a, 26 Sep 2026)
- The CPU backend's `vec_dot_type` for Q4_0-family weights is `GGML_TYPE_Q8_0`, and for K-quants (Q4_K, Q6_K) it is `GGML_TYPE_Q8_K`. The CPU therefore quantizes activations to Q8_0 or Q8_K before the dot product. — [ggml-cpu.c](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/ggml-cpu.c)
- The CUDA MMVQ path's dot-product functions take `block_q8_1` activations (`vec_dot_q_cuda_t(... const block_q8_1 * ...)`). CUDA quantizes activations to q8_1, a different block format with different scales and rounding. — [mmvq.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu)
- `test-backend-ops` checks backends against CPU with a normalized-MSE tolerance, not bit-equality. The default `max_nmse_err` is 1e-7, and **`test_mul_mat` and `test_mul_mat_id` override it to 5e-4** (2e-2 for one backend special case). leloch's "MUL_MAT_ID 789/789" therefore certifies closeness, not bitwise identity. — [test-backend-ops.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tests/test-backend-ops.cpp)

**Adjacent llama.cpp evidence that small numeric changes flip greedy tokens** (GPU-internal, not placement)
- Issue #29168 (delebash, 20 Sep 2026): the CUDA MoE weighted-reduction fusion (PR #25952, b10751) "changes the order of floating-point accumulation across experts. Small differences in the target model's logits are enough to flip the argmax on some tokens".
  - Draft acceptance fell 0.823 → 0.481, and output went from byte-identical greedy to diverging.
  - The run config was an RTX 2070 SUPER with `-ngl 31 --n-cpu-moe 21`.
  - Source: [Issue #29168](https://github.com/ggml-org/llama.cpp/issues/29168)
- Issue #27407: "Batched verification alone … perturbs CUDA target logits at near-ties". It is attributed to `calc_nwarps` and the MMVQ threshold changing reduction structure with batch width. A contributor measured "76 to 80 % of requests differ from their baseline" (median fork 23% into the text). CPU/GPU placement is not discussed. — [Issue #27407](https://github.com/ggml-org/llama.cpp/issues/27407)

**Determinism literature (2025–2026): no CPU/GPU hybrid placement treatment found**
- Thinking Machines, "Defeating Nondeterminism in LLM Inference" (10 Sep 2025): the root cause is lack of batch invariance. GPU kernels only. — [Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- LLM-42 (2601.17768, 30 Jan 2026): determinism via verified speculation, same GPU. The full text mentions CPU only as the host (64 cores) and in related work on offloading. — [arXiv 2601.17768](https://arxiv.org/abs/2601.17768)
- CoRun (2608.14376, 14 Aug 2026): "most kernels are not batch-invariant, they are position-invariant". Deterministic via isolated prefill and fixed-shape decode, "improving throughput by 15-324% over batch-invariant approaches". GPU only. — [arXiv 2608.14376](https://arxiv.org/abs/2608.14376)
- MarginGate (2605.30218, 28 May 2026): sparse margin-triggered verification for batch-invariant inference. Zero CPU mentions in the full text. — [arXiv 2605.30218](https://arxiv.org/abs/2605.30218)
- 2609.11356 (Yang, Riasanovsky, Deng, Sarkar; 10 Sep 2026), "Taming Bitwise Behavior in GPU Kernels with Tensor Core":
  - Black-box reconstruction of cuBLAS reduction order.
  - Triton GEMMs that "match NVIDIA cuBLAS in all tested cases on Blackwell and Hopper".
  - "Preserving a fixed order can cost up to 20 percent".
  - Zero CPU mentions in the full text.
  - Source: [arXiv 2609.11356](https://arxiv.org/abs/2609.11356)
- Cooper, Jeong, Jeon, Young, Kim (2609.25624, 22 Sep 2026): kernels bit-identical across GPU architectures ("Across A100, L40S, and H100, all seven cases are bitwise identical"). The CPU appears only as the input seeder ("seeded on the CPU so every GPU sees identical input bits") and via contrast with Hawkeye's CPU replay. No offload or hybrid. — [arXiv 2609.25624](https://arxiv.org/abs/2609.25624)
- Hawkeye (2603.20421; v2 15 May 2026): "anyone can re-execute on a CPU the exact matrix multiplication operations underlying a machine learning" model, "achieving bit-exact equivalence" of tensor-core arithmetic on CPUs. It is for third-party verification. Cooper et al. describe it as "an order of magnitude slower". It is the closest thing to "bitwise-identical CPU reproduction of GPU arithmetic", but it targets FP16/BF16 tensor-core MMA, not quantized ggml kernels, and not serving-time placement. — [arXiv 2603.20421](https://arxiv.org/abs/2603.20421); [arXiv 2609.25624](https://arxiv.org/abs/2609.25624)
- Pape et al., "Silent Hyperparameter" (2605.19537): llama.cpp is one of the backends, with all models in FP16 GGUF. Zero CPU or offload mentions in the full text. — [arXiv 2605.19537](https://arxiv.org/abs/2605.19537)
- Masoudian et al. (2608.04714): Ollama/llama.cpp vs HF on a single RTX 3090, "without quantization". No offload. — [arXiv 2608.04714](https://arxiv.org/abs/2608.04714)
- 2609.04198 (3 Sep 2026) found that "self-hosting on batch-invariant kernels helped only while the server was quiet". Relevant only as evidence that determinism breaks under load. — [arXiv 2609.04198](https://arxiv.org/abs/2609.04198)

**Timing dependence of placement** (relevant to whether nondeterminism is run-to-run or history-only)
- 2512.16473: misses are "offloaded to CPU while experts are asynchronously fetched to GPU for subsequent token generation". — [arXiv 2512.16473](https://arxiv.org/abs/2512.16473)
- PR #27861 uses a worker thread and caps inserts per layer per step. Its thread warns "if upload speed falls behind decode, all slots may be in-flight" (per companion notes). — [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861)

### Inferences
- **Two distinct invariance failures, worth separating in a paper:**
  1. **History dependence** (analogous to batch invariance): with a deterministic cache, the same prompt yields different outputs depending on prior requests or cache warm state, because hit/miss placement depends on history. #26563's statement describes exactly this.
  2. **Run-to-run nondeterminism**: when fills complete asynchronously (2512.16473's async fetch, #27861's worker thread), whether an expert is a GPU hit at step t can depend on wall-clock timing. Then even an identical cold-start run can diverge.
  
  I found no paper or issue that separates or measures either.
- The llama.cpp activation-quantization mismatch (Q8_0/Q8_K on CPU vs q8_1 on CUDA) means bitwise equality cannot come from fixed reduction order alone. A fix needs a shared activation-quantization format and bit-matched integer dot products with a fixed accumulation order on both devices. Integer dot products of quantized blocks are exactly associative in int32, so a bitwise-identical CPU/GPU quantized kernel is plausible (inference). The remaining risk is in float scale application and the final FP32 sum order.
- The researcher's measurements (layer-offload vs all-GPU greedy top-1 disagreement ~4.5% on gpt-oss-20b and 1.6% on Qwen3; NLL within 0.1%) would be the first published quantification of the phenomenon #26563 describes. They are consistent with #24528's "statistically identical perplexity": NLL parity does not imply token-sequence identity, as #27407's 76–80% request divergence under GPU-internal changes shows.
- Scoop risk: moderate. The determinism community is very active (four relevant papers in Aug–Sep 2026) but GPU-centric. The llama.cpp expert-cache authors have noticed the effect but have not measured it.

### Gaps
- The full #26563 thread was only seen through a WebFetch summary. The exact date of the "may slightly vary" comment and its "extended tests" (method, metric) were not retrieved.
- I did not check ik_llama.cpp, KTransformers or SGLang-KT issue trackers for similar reports.
- FreeToken's code (flashml.ai) was not inspected for whether its CPU kernels use the same accumulation or activation format as its GPU kernels.
- Not verified: whether llama.cpp's CUDA MMQ (batched) path and other backends (Vulkan, Metal) use q8_1 in every case. Only MMVQ was checked.
- The Thinking Machines follow-ups surveyed here are those on arXiv. Blog posts, vLLM/SGLang docs, and any Sept 2026 workshop papers on determinism were not exhaustively searched.
