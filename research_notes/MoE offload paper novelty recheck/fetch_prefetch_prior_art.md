# Prior art for FETCH (miss split) and PREFETCH (cross-layer router lookahead), rechecked 28 Sep 2026

Method note:
- Full texts were read from arXiv HTML, converted to text and grepped. Covered: FreeToken 2608.16157, Fate 2502.12224, Speculating Experts 2603.19289, SeqMoE 2609.12978, Si et al. 2608.12103, HybriMoE 2504.05897, DALI 2602.03495, ProMoE 2410.22134, Mixtral-offloading 2312.17238, AdapMoE 2408.10284, HOBBIT 2411.01433, DAOP 2501.10375, MMA 2512.16056, DAK 2604.26074 and QuaRot 2404.00456.
- The FreeToken repository was cloned and its full history fetched. HEAD is `0d652e7` (2026-09-25); the initial open-source commit is `3af9d90` (2026-08-11).
- llama.cpp items come from the companion notes (`MoE hybrid decode novelty check/llamacpp_community_caches.md`, `MoE offload system race plan/llamacpp_ecosystem.md`), plus one WebFetch of PR #16991 and a grep of local master `2145525` (2026-09-26).
- Numbers for "our" system are from the brief and from [prefetch_lookahead.md](/home/claude/moe-speed-of-light/research_notes/MoE%20offload%20system%20race%20plan/prefetch_lookahead.md).

## Q0. Verdict summary: which details are new?

### Takeaway
Almost all of the mechanisms are already published. Most new material is in the measurements rather than the mechanisms:
- FETCH's mechanism is already done, in essentially identical form, by FreeToken. Its public code (11 Aug 2026) also measures the concurrent CPU-plus-DMA bandwidth pair. HybriMoE (Apr 2025) and DALI (Feb 2026) already did split-by-simulation.
- PREFETCH's predictor is already done: the next layer's router on the current state goes back to Mixtral-offloading (Dec 2023). Speculating Experts (Mar 2026) applies the *next layer's norm* to the current residual on gpt-oss-20b/120b.
- Folding the norm-weight ratio into one precomputed matrix is an algebraic rearrangement of that. Folding RMSNorm gains into adjacent linear weights is itself standard (QuaRot).
- New or partly new:
  - (a) Measured sign changes of both features across hosts (+14–19% / +2–6% / −16%), with the explanation in PCIe-to-DRAM terms.
  - (b) Published contention numbers (73–77 GB/s vs 62; 52 vs 47). Only FreeToken's code measures this; it publishes no numbers, and its paper's model assumes full contention.
  - (c) The specific graph plumbing: a zero-copy SM kernel prefetch on an event-forked side stream inside the decode CUDA graph, joined before the next layer's GPU experts.
  - (d) The findings that a late prefetch should be waited for rather than sent to the CPU, and that a main-stream copy gains ~0%.
  - (e) The prefetch benefit depending on context length is **not** new (Speculating Experts reports it). Its interaction with a CPU-miss baseline and the PCIe/DRAM ratio is.

| Detail | Closest prior work (public date) | Verdict |
|---|---|---|
| Split a layer's m misses: some copied and run on GPU, rest on CPU, concurrently | HybriMoE (8 Apr 2025); FreeToken q⋆ (paper 17 Aug 2026, code 11 Aug 2026); DALI (3 Feb 2026) | Already done |
| Table T[m] from measured bandwidths | FreeToken `hybrid_fetch_fraction` × m, rounded "to whichever integer balances the overlap best", ≥1 fill | Already done (a table vs a rounded fraction is cosmetic) |
| Copy by a kernel reading pinned mapped host memory, inside CUDA graph, overlapped with CPU helpers | FreeToken `fast_index_copy` (zero-copy, pinned+mapped, graph-captured); SeqMoE on-demand kernel | Already done |
| Where the split helps vs hurts as a function of PCIe:DRAM ratio | FreeToken: hybrid-vs-offload rule at a 2× CPU:PCIe threshold; q⋆ ≈ m·B_P/B_H. No isolated ablation, no reported loss case | Partly new (the measured −16% on PCIe 4.0 vs CPU-only misses is new) |
| Measured DRAM contention, CPU reads plus PCIe DMA | FreeToken code `measure_overlap_bw` (11 Aug 2026), unpublished numbers; paper Eq. 2 assumes full contention | Partly new (published numbers are new; the qualitative point is in FreeToken's code) |
| Next layer's router applied to current layer's MoE input | Mixtral-offloading (28 Dec 2023); AdapMoE (Aug 2024); HOBBIT (Nov 2024); HybriMoE; Fate (Feb 2025); DALI; Speculating Experts (9 Mar 2026) | Already done |
| Correct for the next layer's RMSNorm weight | Speculating Experts: q_l = LN_{l+1}(d_l + r_l) | Already done in effect |
| Fold the norm ratio into a precomputed per-layer-pair matrix | No exact match. HOBBIT stacks next-layer gate matrices into one GEMV; QuaRot absorbs RMSNorm gain into the following linear | New as an implementation detail only (not claimable as a contribution) |
| q=1 prefetch, lowest-score victim, never evict a predicted expert | DALI: prefetching exactly one expert is best; Mixtral-offloading: speculative loads "do not replace the currently cached experts" | Mostly done; victim rule is a detail |
| Zero-copy prefetch kernel on an event-forked side stream inside the CUDA graph, joined before layer l+1's GPU experts | llama.cpp #16991 (event fork/join inside graphs, for QKV); SeqMoE (prefetch via CPU-thread `cudaMemcpyAsync` on a copy stream, outside the graph); Speculating Experts (PyTorch copy stream) | Partly new (the combination was not found) |
| Copy must overlap attention/GPU work (main-stream copy ≈ 0%) | Speculating Experts' ΔT = Σ min(t_copy, t_compute), gains from "separate CUDA streams" | Already known in principle |
| Wait for a late prefetch rather than falling back to CPU | Not found | New (minor) |
| Prefetch gain depends on context length | Speculating Experts: "greater benefits at longer sequence lengths" (1024 vs 65536) | Already reported |
| Prefetch gain depends on the host's PCIe/DRAM ratio (and turns negative on PCIe 4.0 vs CPU-miss baseline) | Not found. All prefetch papers use an on-demand-copy baseline | New |

### Cited Findings
- FreeToken splits misses "into a cache-fill set F and a CPU-execution set C … These two sets are served concurrently". — [FreeToken, arXiv 2608.16157](https://arxiv.org/abs/2608.16157)
- HybriMoE's scheduling: "argmin over cpu_expert, gpu_expert of max(CPU_TIME(cpu_expert), GPU_TIME(gpu_expert))", with "Transfer Priority … movement of high-load uncached experts from CPU to GPU". — [HybriMoE, arXiv 2504.05897](https://arxiv.org/abs/2504.05897)
- Mixtral-offloading: "it is possible to get an accurate guess of next layer's experts by applying next layer's gating function to previous layer's hidden states — or, more specifically, to the same hidden states that are used by previous MoE layer's gating function" (arXiv v1 28 Dec 2023). — [arXiv 2312.17238](https://arxiv.org/abs/2312.17238)
- Speculating Experts: "The quasi-hidden state q_l is defined as q_l = LN_{l+1}(d_l + r_l), where LN_{l+1}(·) denotes the normalization applied to the residual stream prior to expert routing at layer l+1" (v1 9 Mar 2026). — [arXiv 2603.19289](https://arxiv.org/abs/2603.19289)
- QuaRot: "the linear parts of LayerNorm or RMSNorm are fused into adjacent weight matrices". — [QuaRot, arXiv 2404.00456](https://arxiv.org/abs/2404.00456)

### Inferences
- A paper should present FETCH and PREFETCH as re-implementations inside llama.cpp, with new cross-host measurements. It should not present them as new mechanisms. The defensible claims are:
  - the sign flip across hosts;
  - the measured contention and its effect on the split;
  - the lookahead's interaction with a CPU-miss baseline.

### Gaps
- Klotski, DuoServe-MoE, eMoE, SiDA and MoE-Infinity were not re-read in full text this pass.
  - Companion notes and FreeToken's related work describe them as learned, trace-based or offline predictors.
  - SiDA (per FreeToken) "substitute[s] or skip[s] low-scoring experts".
  - None is known to use the next-layer router with norm correction. This is unverified for Klotski, DuoServe-MoE and eMoE.

## Q1. Who else splits a layer's misses between copying to the GPU and executing on the CPU (FETCH)?

### Takeaway
At least three published systems do this, plus one closely related variant. FreeToken is a near-exact match:
- misses split by a bandwidth-derived count;
- the fill done by a zero-copy gather kernel from pinned mapped host memory;
- all control on the GPU inside a captured CUDA graph;
- the CPU branch run concurrently.

HybriMoE and DALI split by simulation or integer programming. 2512.16473 runs every miss on the CPU and copies it asynchronously for later tokens.

**Verdict: already done.**

### Cited Findings
- **FreeToken** (arXiv 17 Aug 2026; Berkeley/UT; code https://github.com/FlashML-org/FreeToken). — [arXiv 2608.16157](https://arxiv.org/abs/2608.16157)
  - "FreeToken's bandwidth-adaptive execution dynamically divides the m missing experts between PCIe transfer and CPU execution based on the balance of two measured bandwidths: the pinned expert-transfer bandwidth (B_P) and the host-side expert-processing bandwidth (B_H)."
  - Eq. 4: q⋆ ≈ m·B_P/B_H.
  - "In practice, FreeToken rounds q⋆ to an integer … It always retains at least one fill".
  - "FreeToken launches the CPU branch first. It then runs the GPU miss path, which consists of a cache update, a batch copy of F, and a grouped evaluation of the combined GPU execution set G = H ∪ F. Concurrently, the CPU workers process C."
  - "For each MoE layer, one GPU kernel deduplicates the routed experts, classifies them against the residency table, derives the bandwidth-based fetch count q, selects eviction victims, and rewrites logical routed IDs into physical slot IDs or a special CPU-assignment flag."
- **FreeToken code** (initial release `3af9d90`, 2026-08-11). — [FreeToken repo](https://github.com/FlashML-org/FreeToken)
  - `python/freetoken/kernel/pinned.py`: "The offload gather kernel (`fast_index_copy`) reads host memory zero-copy from the GPU, so allocations must be pinned + device-mapped".
  - `offload_cache.py`: "hybrid only: when > 0, replaces the fixed cap with a per-step fraction -- fetch ~fraction * misses experts over PCIe (rounded to whichever integer balances the overlap best), the CPU computes the rest."
  - `docs/cli.md`: `--moe-hybrid-max-fetch` "0 = never fetch (all misses on CPU); large = behaves like plain offload."
- **HybriMoE** (arXiv 8 Apr 2025; DAC 2025). "HybriMoE divides all activated experts into a GPU queue and a CPU queue … Before the actual execution, HybriMoE performs a simulation phase … by iteratively filling the CPU computation, GPU computation, and data transferring timelines". Warm-up collects "CPU and GPU processing speeds and data transfer latency". — [arXiv 2504.05897](https://arxiv.org/abs/2504.05897)
- **DALI** (arXiv 3 Feb 2026). It "dynamically assigns experts to CPU or GPU by modeling assignment as a 0-1 integer optimization problem", with objective min max(T_gpu, T_cpu). — [arXiv 2602.03495](https://arxiv.org/abs/2602.03495) (quotes via companion note [academic_systems.md](/home/claude/moe-speed-of-light/research_notes/MoE%20hybrid%20decode%20novelty%20check/academic_systems.md))
- **2512.16473** (ASP-DAC 2026, arXiv 18 Dec 2025). "Misses are computed on the CPU while the missed expert is asynchronously copied to the GPU for later tokens" (companion-note paraphrase). It uses "two independent CUDA streams". — [arXiv 2512.16473](https://arxiv.org/abs/2512.16473)
- **Fiddler and KTransformers** execute misses on the CPU. HybriMoE describes them thus: "when an expert is not in the GPU cache, the CPU executes the corresponding expert layer instead of loading it from memory". — [HybriMoE, arXiv 2504.05897](https://arxiv.org/abs/2504.05897)
- **llama.cpp community** (dual-chain hit/miss split, none with a copy/CPU miss split):
  - leloch RFC #24528 (12 Jun 2026): "thread 0 dispatches one batched matvec over the cached (hit) rows on the GPU while the other threads compute the miss rows". — [Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)
  - #27861 and #26563 run the dual chains through stock ggml scheduler splits. — [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861); [PR #26563](https://github.com/ggml-org/llama.cpp/pull/26563) (as summarised in companion notes)

### Inferences
- Relative to FreeToken, FETCH differs only in:
  - the integer table T[m] instead of a rounded fraction;
  - running inside llama.cpp with GPU-signalled CPU helpers;
  - its baseline being CPU-executes-all-misses (FreeToken's hybrid always does ≥1 fill).

  None of these is a mechanism-level novelty.
- FETCH's "rest run on CPU in the same step" plus "copied experts stay resident" is FreeToken's F/C split exactly: "Experts in F … remain resident for future reuse; experts in C … leave residency unchanged".

### Gaps
- I found no public llama.cpp PR or fork that implements a per-layer copy/CPU split of misses. Those examined route all misses to the CPU or all to PCIe.

## Q2. Does anyone report where the split helps vs hurts as a function of the PCIe-to-DRAM bandwidth ratio? Which numbers should we compare against?

### Takeaway
FreeToken frames the split as ratio-dependent. Its code's auto-selection picks "hybrid" only when CPU MoE bandwidth exceeds 2× the PCIe gather bandwidth, and otherwise pure offload.

FreeToken publishes no isolated ablation of the split (split vs CPU-only vs fill-only), and no case where fetching loses. Its paper reports whole-system gains on PCIe 4.0 machines (1.3× over the strongest baseline on RTX 3090/4090).

The measured result is therefore partly new: FETCH gains +14–19% at link ≈ DRAM, +2–6% at 57/62, and −16% on PCIe 4.0 (link ≈ half DRAM) against a CPU-only-miss baseline. It is also in tension with FreeToken's always-≥1-fill policy on PCIe 4.0-class hosts.

### Cited Findings
- FreeToken problem statement: "Relying on transfer alone leaves residual host bandwidth and CPU cores idle whenever host memory can deliver more bytes than the link can move. Conversely, relying on CPU execution alone leaves the PCIe link idle and forfeits the future hits that a cache fill would buy. The right mixture depends on the hardware … an RTX 4060 laptop on LPDDR5 and an RTX 5090 desktop on DDR5 sit at opposite ends of the host-to-PCIe balance. This optimal mixture cannot be read from specification sheets." — [arXiv 2608.16157](https://arxiv.org/abs/2608.16157)
- FreeToken Table 1 (B_P = measured pinned H2D expert transfer; B_H = measured CPU MoE kernel bandwidth; GB/s). — [arXiv 2608.16157](https://arxiv.org/abs/2608.16157)

  | Machine | PCIe | B_P | B_H | q⋆/m = B_P/B_H |
  |---|---|---|---|---|
  | 5090 server, 2× Xeon Gold 6459C | 5.0 x16 | 52.7 | 77.3 | 0.68 |
  | 4090, 2× Xeon Plat 8358P | 4.0 x16 | 25.1 | 63.2 | 0.40 |
  | 3090, 2× Xeon Gold 6330 | 4.0 x16 | 25.3 | 56.7 | 0.45 |
  | **5090 desktop, Ryzen 9 9950X3D** | 5.0 x16 | 49.0 | 53.8 | 0.91 |
  | 4060 laptop, i9-13900H | 4.0 x8 | 11.8 | 47.5 | 0.25 |
  | RTX PRO 6000, Xeon Plat 8559C | 5.0 x16 | 51.5 | 178 | 0.29 |

- FreeToken cross-hardware results (whole system, not the split alone). — [arXiv 2608.16157](https://arxiv.org/abs/2608.16157)
  - "FreeToken leads the strongest baseline by 1.3× on the RTX 3090 and 4090, 1.9× on the 5090 server, 2.1× on the 5090 desktop, and 1.8× on the RTX 4060 laptop".
  - "moving from the many-channel server to a dual-channel consumer desktop costs FreeToken 4% of its decode rate, while llama.cpp keeps only 80% of its rate".
  - On the 5090 server, Qwen3.6-35B-A3B BF16 reached "77–83 tok/s … 1.8–2.3× … the strongest baseline".
- FreeToken code, auto backend choice (`benchbw.py`, since 2026-08-11). — [FreeToken benchbw.py](https://github.com/FlashML-org/FreeToken/blob/main/python/freetoken/moe/benchbw.py)
  - "recommend `hybrid` when the CPU MoE kernel bandwidth exceeds `threshold` x the PCIe gather bandwidth (default 2x), otherwise `offload`".
  - The code comment says the MXFP4/gpt-oss geometry is "the reason that format sits near the 2x threshold".
  - `engine.py` states that "a cached `ft bench bw` profile can upgrade the offload default to hybrid when this machine's CPU MoE bandwidth clears its PCIe gather bandwidth by the bench threshold (default 2x)".
- I found no ablation isolating q⋆ in the FreeToken paper. Its breakdown section covers pipelined prefill, cache locality (LRU misses 16%/39% vs KTransformers 41%/59% vs llama.cpp static 62%/89%) and cross-hardware only. — [arXiv 2608.16157](https://arxiv.org/abs/2608.16157)
- HybriMoE ablation (A6000 + Xeon 5220R, 10 cores): "+scheduling" alone gives 0.21 s → 0.14 s per token (1.46×). The baseline is KTransformers-style static mapping. — [arXiv 2504.05897](https://arxiv.org/abs/2504.05897) (numbers via companion note headroom_techniques.md)
- Our measurements (brief, jobs 059/060/065/066). — [prefetch_lookahead.md](/home/claude/moe-speed-of-light/research_notes/MoE%20offload%20system%20race%20plan/prefetch_lookahead.md)
  - FETCH +14–19% on RTX 5090 + 9950X (DRAM ~47, link ~47);
  - +2–6% on the 62/57 host;
  - −16% on RTX 4500 Ada + 7900X (PCIe 4.0).

### Inferences
- **Direct comparison point.** Our 9950X host (link ≈ DRAM ≈ 47) sits where FreeToken's 5090 desktop sits (49.0/53.8).
  - FreeToken's rule would pick pure "offload" there, since CPU < 2× PCIe; i.e., fetch every miss.
  - Our FETCH instead keeps some misses on the CPU and gains +14–19% over CPU-only.
  - The paper should add the missing arm, **fetch-all (FreeToken's offload)** at equal VRAM, or say why it is absent.
- **Where FreeToken's policy may be wrong.** On PCIe 4.0 hosts (B_P/B_H ≈ 0.4) FreeToken still fetches ~40% of misses and "always retains at least one fill". Our −16% on a PCIe 4.0 host against CPU-only misses suggests that fill traffic can be net-negative there, at least for gpt-oss-20b at C = 8/32. That is a publishable contrast, but it rests on one host and one model.
- **FreeToken never reports this regime.** Its PCIe 4.0 wins are against llama.cpp static placement and KTransformers, not against its own `--moe-hybrid-max-fetch 0`.
- **Mechanism for the sign flip (my arithmetic).** A fill costs S/B_P of link time plus DRAM bandwidth taken from the helpers, and pays back only through future hits.
  - When B_P ≈ B_DRAM, one fill costs about the same wall time as one CPU miss and also creates a resident expert, so it wins.
  - When B_P ≈ B_DRAM/2, one fill costs ~2 CPU-miss times on the link, and it contends with the helpers.
  - This matches the breakeven-reuse framing in 2608.12103's r⋆ (see companion notes).

### Gaps
- FreeToken does not publish per-machine q⋆ values or tok/s for hybrid vs cpu vs offload on the same machine. The repo's `benchmarks/bench_decode_moe.py --backend offload,cpu,hybrid` can produce them, but no results file is committed.
- I found no paper that reports a *loss* from splitting misses on any host.

## Q3. Does anyone measure the contention when CPU reads and PCIe DMA hit host DRAM at the same time?

### Takeaway
There are no published numbers. FreeToken's paper models full contention: the CPU keeps only B_H − B_P while the link is saturated. FreeToken's public code (since 11 Aug 2026) explicitly measures the concurrent pair and says the full-contention assumption "over-penalizes" the CPU. That is the same qualitative observation as ours, but no numbers are published.

Our numbers (73–77 GB/s total vs 62 CPU-alone; 52 vs 47) would be the first published measurement for this MoE setting. They also quantitatively refute the paper's Eq. 2 on our host.

**Verdict: partly new.**

### Cited Findings
- FreeToken paper, Eq. 2–3: "Since both expert DMA transfers and CPU execution read from the same host-memory subsystem, a saturated PCIe transfer leaves a residual bandwidth of B_R = max(B_H − B_P, 0) … T_cpu(m−q) ≈ (m−q)S/(B_H − B_P)." — [arXiv 2608.16157](https://arxiv.org/abs/2608.16157)
- FreeToken code, `measure_overlap_bw` in `python/freetoken/moe/benchbw.py` (present in the initial release `3af9d90`, 2026-08-11). — [FreeToken benchbw.py](https://github.com/FlashML-org/FreeToken/blob/main/python/freetoken/moe/benchbw.py)
  - Docstring: "Concurrent achieved bandwidths (GB/s): the CPU MoE GEMV and the PCIe gather running at the same time -- the contention regime hybrid decode's overlap actually lives in."
  - "The standalone numbers cannot predict this split. Assuming full DRAM contention (CPU keeps `cpu_bw - pcie_bw` under DMA) over-penalizes a CPU kernel that never saturated DRAM to begin with -- the DMA then mostly rides the leftover bandwidth; assuming no contention ignores it entirely."
  - Method: a CPU thread loops bs=1 CPU decode steps while the main thread loops full-layer zero-copy gathers for 2 s.
- FreeToken `bench_profile.load_hybrid_fetch_fraction` uses the overlapped pair when present: "fetched/misses = pcie_ov / (pcie_ov + cpu_ov)". Older profiles fall back to "pcie/cpu" (the full-contention formula). — [FreeToken bench_profile.py](https://github.com/FlashML-org/FreeToken/blob/main/python/freetoken/moe/bench_profile.py)
- SeqMoE addresses *link* contention, not DRAM. "when a cache miss triggers on-demand loading, the GPU sets a shared flag to pause new prefetch submissions, reducing bandwidth contention." — [SeqMoE, arXiv 2609.12978](https://arxiv.org/abs/2609.12978)
- Other contention papers are about different resources:
  - DAK (28 Apr 2026) measures "contention between the local and remote data paths" *inside the GPU* (Grace Hopper TMA host reads vs HBM). — [DAK, arXiv 2604.26074](https://arxiv.org/abs/2604.26074)
  - MMA (v2 13 May 2026) notes that with multipath relays "host DRAM bandwidth and the inter-socket interconnect, not PCIe, become the binding constraints". It assumes "the CPU is lightly loaded". — [MMA, arXiv 2512.16056](https://arxiv.org/abs/2512.16056)
- Our measurement (concur.cu), from [prefetch_lookahead.md](/home/claude/moe-speed-of-light/research_notes/MoE%20offload%20system%20race%20plan/prefetch_lookahead.md):
  - host 1: CPU 62 GB/s alone and 37 GB/s alongside the link; link 57 alone and 38 alongside; sum ≈ 75 (brief: 73–77 total vs 62 CPU-alone);
  - host 2: 52 vs 47.

### Inferences
- **FreeToken's two formulas on our host 1** (B_P = 57, B_H = 62; my arithmetic):
  - Paper Eq. 2 predicts the CPU keeps 62 − 57 = 5 GB/s under DMA; we measured 37 GB/s.
  - Paper Eq. 4 gives q⋆/m = 57/62 = 0.92. The code's overlapped formula with our numbers gives 38/(38 + 37) = 0.51.
  - The published closed form would therefore fetch almost all misses, while the measured-contention split fetches about half.
  - This is a concrete, checkable point for the paper. It is the one place where our numbers directly correct the FreeToken paper; FreeToken's own code already moved away from Eq. 2.
- **Total bandwidth exceeds CPU-alone** (75 vs 62). One CPU helper pool does not saturate host DRAM, so DMA partly uses otherwise idle bandwidth. This is the "DMA then mostly rides the leftover bandwidth" statement in FreeToken's code, now quantified.

### Gaps
- I found no HPC or systems paper reporting STREAM-plus-PCIe-DMA aggregate host-DRAM bandwidth on AM5 Zen 5 or Zen 4 desktops. The contention figures can only be compared with FreeToken's unpublished profiles.
- FreeToken does not state whether its CPU kernel saturates DRAM on any of its six machines.

## Q4. Who applies the next layer's gate to the current hidden state, how do they handle the normalization, and does anyone fold the norm into a precomputed matrix?

### Takeaway
Many do. The lineage runs:
- Mixtral-offloading (Dec 2023);
- AdapMoE (Aug 2024);
- HOBBIT (Nov 2024), which stacks several next-layer gates into one matmul;
- HybriMoE (Apr 2025), for the next three layers;
- Fate (Feb 2025), on the CPU;
- DALI (Feb 2026), which adds a learned-free residual vector;
- Speculating Experts (Mar 2026).

Most feed layer l's already-normalized gate input (with layer l's norm weight) to layer l+1's gate, with no correction. Speculating Experts is the exception: it applies **layer l+1's own normalization** to layer l's post-attention residual (plus a "default vector"), evaluated on gpt-oss-20b/120b and Qwen3-30B-A3B.

Our ratio fold computes the same quantity as LN_{l+1}(r_l) without the default vector: W_{l+1}·diag(g_{l+1}/g_l)·s_l = W_{l+1}·LN_{l+1}(r_l). It is an algebraic rearrangement. I found no one who precomputes that product. Absorbing RMSNorm gains into adjacent linear weights is standard (QuaRot), and HOBBIT already fuses multiple next-layer gates into one GEMV.

**Verdict: already done in effect.** The fold is new only as an implementation detail.

### Cited Findings
- **Mixtral-offloading** (Eliseev & Mazur, arXiv v1 28 Dec 2023). — [arXiv 2312.17238](https://arxiv.org/abs/2312.17238)
  - "applying next layer's gating function to previous layer's hidden states — or, more specifically, to the same hidden states that are used by previous MoE layer's gating function. This heuristic relies on the fact that transformer layers are residual".
  - "The speculative expert loading fetches 1−2 most likely experts. The newly loaded experts do not replace the currently cached experts."
  - No normalization correction is described. The public repo (dvmazur/mixtral-offloading) does not contain the speculative path; I checked `src/custom_layers.py`.
- **AdapMoE** (arXiv 19 Aug 2024): "AdapMoE utilizes the gate functions from the subsequent layer directly for predictive prefetching", with prediction for "the next two/three layers". — [arXiv 2408.10284](https://arxiv.org/abs/2408.10284)
- **HOBBIT** (arXiv Nov 2024, v2 6 Nov 2024). — [arXiv 2411.01433](https://arxiv.org/abs/2411.01433)
  - The predictor uses "the current gating input".
  - "we can optimize the process by stacking all p gating modules together and computing them simultaneously … the Stacking Computer module … stacking, matrix multiplication, and top-k selection".
  - No normalization correction is described.
- **HybriMoE** (8 Apr 2025): "HybriMoE predicts expert activations for the next three layers by reusing the gating information from those layers". — [arXiv 2504.05897](https://arxiv.org/abs/2504.05897)
- **Fate** (arXiv Feb 2025; v2 7 May 2025). — [arXiv 2502.12224](https://arxiv.org/abs/2502.12224)
  - It compares Gate_in_{i+1} with Attn_in_{i+1}, Gate_in_i and Attn_in_i and picks Gate_in_i.
  - "the input (intermediate state) is cloned to the CPU for parallel prediction computation".
  - It reports "prefetch accuracy of 97.15%" (with confidence thresholding).
  - No normalization handling was found in the text.
- **DALI** (3 Feb 2026): "h̃^(l) = hidden_states^(l) + res_vec^(l), predict_expert^(l+1) = gate_func^(l+1)(h̃^(l)), where hidden_states^(l) is the input to the l-th MoE gate, res_vec^(l) is the layer-specific residual vector". "obtaining the residual vector requires no fine-tuning or retraining." — [arXiv 2602.03495](https://arxiv.org/abs/2602.03495)
- **Speculating Experts** (arXiv 9 Mar 2026; code in YALIS). — [arXiv 2603.19289](https://arxiv.org/abs/2603.19289)
  - "In the pre-norm MoE architectures considered in this work, expert routing is computed from a normalized residual stream following attention … q_l = LN_{l+1}(d_l + r_l)".
  - The baseline s_l "corresponds to the normalized residual stream produced at layer l and directly fed into the router".
  - "Across the GPT-OSS models, the quasi-hidden states demonstrate higher average cosine similarity than s_l". For Qwen3-30B-A3B, "recall@k of approximately 90% on average" beyond the early layers.
  - A trained estimator reaches "average hit rates of 83% and 88%" for GPT-OSS-120B and GPT-OSS-20B.
- **Si et al. 2608.12103** (v2 30 Aug 2026): "The production engine's one-layer lookahead predictor (PILOT) caches next-layer gate weights and recalls 64.7% of decode-time selections, with a per-layer range of 1–84%". — [arXiv 2608.12103](https://arxiv.org/abs/2608.12103)
- **Learned or trained predictors**:
  - Pre-gated MoE changes and fine-tunes the router (ISCA 2024). — [arXiv 2308.12066](https://arxiv.org/abs/2308.12066)
  - ProMoE uses a learned predictor; it notes "Due to layer normalization, the outputs are numerically smaller than their inputs, leading to a slow change in hidden states across layers". — [arXiv 2410.22134](https://arxiv.org/abs/2410.22134)
  - ExpertFlow uses a trained "Routing Path Predictor". — [arXiv 2410.17954](https://arxiv.org/abs/2410.17954)
  - SeqMoE uses a Mamba2 seq2seq predictor. — [arXiv 2609.12978](https://arxiv.org/abs/2609.12978)
- **QuaRot** (v2 29 Oct 2024): "the linear parts of LayerNorm or RMSNorm are fused into adjacent weight matrices … removing the scaling operation from RMSNorm (diag(α)) and absorbing into the subsequent weight matrices". — [arXiv 2404.00456](https://arxiv.org/abs/2404.00456)
- **llama.cpp**: the closed PR #21609 (7–8 Apr 2026) was titled "expert-cache: N-slot LFRU cache with FATE prefetch". It reported "96.7% hit rate, 1.79x decode speedup on GPT-OSS-120B with RTX PRO 2000 8GB". It was closed under the AI-generated-PR policy. — [PR #21609](https://github.com/ggml-org/llama.cpp/pull/21609) (as summarised in companion notes)

### Inferences
- Our recall numbers (gpt-oss-120b 84%/97% at top-4/top-8; gpt-oss-20b 87%/98%) should be compared with:
  - Speculating Experts' per-layer figures for gpt-oss (Router-PF with q_l; figure-only, no scalar in the text);
  - its trained-estimator 83%/88%;
  - PILOT's 64.7% (a different model and production engine).

  Ours sit at or above all of them, but the router-PF comparison needs reading their Figure 4.
- The ratio fold gives the same result as Speculating Experts' LN_{l+1}(r_l) and avoids one RMSNorm per layer. It saves only a few µs and cannot be claimed as a contribution. It is worth one sentence ("equivalent to applying layer l+1's RMSNorm, cf. Speculating Experts; we fold the gain ratio into the router matrix, as in QuaRot-style norm absorption").

### Gaps
- Speculating Experts' Figure 4 recall values for gpt-oss under s_l vs q_l could not be extracted from text (figure only).
- I did not verify whether any llama.cpp fork (e.g., ongunm/llama-moe-cache "FATE", PR #21609) applies a norm correction. Their code was not read in this pass.

## Q5. Who places prefetch copies on a side stream inside CUDA graphs? What about the slot policy and waiting vs fallback?

### Takeaway
There is no exact match for "zero-copy SM-kernel prefetch on an event-forked side stream captured inside the decode CUDA graph, joined by an event before layer l+1's GPU experts".

Each ingredient exists:
- llama.cpp's own CUDA backend forks and joins streams with events inside CUDA graphs (PR #16991, for Q/K/V branches);
- FreeToken and SeqMoE run zero-copy gather kernels from pinned mapped memory inside graphs, but on the main path for demand misses;
- SeqMoE issues prefetches from a CPU thread with `cudaMemcpyAsync` on a copy stream *outside* the graph;
- Speculating Experts uses a PyTorch copy stream with `wait_stream`.

"Wait for a late prefetch instead of CPU fallback" was not found anywhere. **Verdict: partly new (engineering).**

### Cited Findings
- llama.cpp PR #16991 "CUDA: add stream-based concurrency" (am17an). — [PR #16991](https://github.com/ggml-org/llama.cpp/pull/16991) (via WebFetch summary)
  - "This PR adds support to run concurrent CUDA streams on single GPU setups", using event fork/join. It works inside CUDA graphs with `GGML_CUDA_GRAPH_OPT=1`.
  - Reported gains are 1–9% on RTX 5090.
  - The mechanism (`ggml_cuda_concurrent_event`, `GGML_CUDA_GRAPH_OPT`) is present on master `2145525` (2026-09-26). — [ggml-cuda.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu)
- SeqMoE (11 Sep 2026). — [arXiv 2609.12978](https://arxiv.org/abs/2609.12978)
  - "CUDA Graph captures the main thread's CUDA operations, with prefetcher activity fully overlapped … For on-demand loading, we allocate the host expert pool with cudaHostAllocMapped. A custom kernel on the main thread's CUDA stream copies missing experts from host to GPU memory, avoiding host synchronization while remaining graph-capturable."
  - "Prefetched weights bypass the control region: the CPU prefetcher initiates host-to-device DMA using cudaMemcpyAsync on a dedicated CUDA copy stream … GPU execution never waits for the CPU prefetcher."
  - Slot states: "when the prefetch lands, the slot transitions from Fetch to Compute if the transfer completes successfully. Otherwise, it falls back to Empty".
- FreeToken: decode fills are "a single, fused transfer" driven by a "device-resident source/destination index list", captured in the graph. Prefill uses "a dedicated transfer stream" to load layer l+1's full expert set while layer l computes. It has no cross-layer *decode* prefetch. — [arXiv 2608.16157](https://arxiv.org/abs/2608.16157); [pinned.py](https://github.com/FlashML-org/FreeToken/blob/main/python/freetoken/kernel/pinned.py)
- Speculating Experts runs in YALIS, which "supports … torch.compile, CUDA Graphs". — [arXiv 2603.19289](https://arxiv.org/abs/2603.19289)
  - Prefetch uses `self.copy_stream`, `compute_stream.wait_stream(self.copy_stream)` and "double buffering to alternate GPU expert buffers across layers".
  - "concurrent CPU→GPU transfers are serialized to use the same copy engine".
  - Whether its prefetch is captured in the graph is not stated.
- KTransformers keeps the CPU path in the graph via `cudaLaunchHostFunc` ("up to 1.23×"). — [KTransformers SOSP'25](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf) (via companion note)
- ongunm/llama-moe-cache ("FATE", ~500 LoC): misses fetched over PCIe on a "Dedicated CUDA prefetch stream". On Qwen3-30B-A3B Q4_K_M on an RTX 4070 Ti, 33.74 → 64.45 t/s against "vanilla". — [llama-moe-cache](https://github.com/ongunm/llama-moe-cache) (as summarised in companion notes)
- Victim and count policy:
  - DALI: "prefetching only one expert—the one with the highest predicted workload—yields the best performance … as more experts are prefetched, the computation time becomes insufficient to overlap the communication cost". — [arXiv 2602.03495](https://arxiv.org/abs/2602.03495)
  - Mixtral-offloading: speculative loads "do not replace the currently cached experts". — [arXiv 2312.17238](https://arxiv.org/abs/2312.17238)
- Wrong or late prefetch: HOBBIT notes that "with cudaMemcpy(), we cannot interrupt the memory copy operation until it completes … this can lead to significant penalties when prediction accuracy is low". — [arXiv 2411.01433](https://arxiv.org/abs/2411.01433)

### Inferences
- The paper can claim the graph-native side-stream plumbing inside llama.cpp as an engineering contribution. It should cite #16991 for the event fork/join pattern and SeqMoE/FreeToken for graph-captured zero-copy kernels.
- The SM-kernel (zero-copy) choice has a cost the paper should state: the copy uses SMs that attention and hit experts also use. SeqMoE instead uses the copy engine for prefetch. Neither paper quantifies this trade-off.
- "Waiting beats CPU fallback" is new. It follows from Q3: a CPU fallback adds DRAM demand while the in-flight copy is still reading DRAM. Present it together with the contention measurement.
- No-evict-predicted plus lowest-score victim is a detail consistent with Mixtral-offloading and SeqMoE's slot states. It is not claimable.

### Gaps
- The PR #16991 open and merge dates were not retrieved. The code is on master as of 2026-09-26.
- I did not confirm whether Speculating Experts' prefetch copy is captured in YALIS's CUDA graphs.

## Q6. Does anyone report that prefetch benefit depends on the host's PCIe/DRAM ratio or on context length?

### Takeaway
- **Context length: already reported.** Speculating Experts shows that copy time is constant while compute grows with context, so prefetch gains more at 65,536 than at 1,024 tokens (Qwen3-30B-A3B −9–14% TPOT). This matches our +10–16% at long prompts vs +4–6% at 109-token chat prompts.
- **GPU speed: reported.** A6000 gains 12–14%; A100/GH200 gain 5–8%.
- **PCIe/DRAM ratio: not found.** Every prefetch paper's baseline is on-demand copying, where prefetch cannot turn negative the way it does against a CPU-executes-misses baseline. Our −16% on PCIe 4.0 and the helps/hurts boundary are new.

### Cited Findings
- Speculating Experts. — [arXiv 2603.19289](https://arxiv.org/abs/2603.19289)
  - "Comparing context lengths 1024 and 65536, we observe that copy times remain constant, but compute time increases with context length for all models. This trend indicates that overlapping compute with copy through prefetching yields greater benefits at longer sequence lengths."
  - "For Qwen3-30B-A3B, prefetching yields a 9–14% reduction in TPOT, with bigger gains at longer sequence lengths."
  - "On more powerful GPUs (A100 and GH200 …), the maximum TPOT improvement is limited to 5–8%, compared to 12–14% on the A6000."
  - Model: ΔT = Σ_l min(t_copy, t_compute,l); "The maximum achievable speedup is 2×".
  - Hardware: A6000 on PCIe 4.0; A100 on PCIe 4.0 (GPT-OSS-120B); GH200 on NVLink C2C.
- SeqMoE: "on Qwen3-30B-FP8 with an RTX 4090 over PCIe 4.0, one layer's computation overlaps the transfer of only 1.39 experts on average". — [arXiv 2609.12978](https://arxiv.org/abs/2609.12978) (via companion note)
- Fate: 16 GB/s PCIe 3.0 (RTX 3090) vs 4 GB/s PCIe 1.0 (GTX 1080 Ti) PCs; "even though outdated PCIe severely slows down inference on low-end PCs, Fate is still significantly better than the baselines" (baselines are on-demand loading and activation-path prefetch). — [arXiv 2502.12224](https://arxiv.org/abs/2502.12224)
- DALI: more prefetches hurt because "the computation time becomes insufficient to overlap the communication cost". — [arXiv 2602.03495](https://arxiv.org/abs/2602.03495)
- 2608.12103: "At 64.7% measured recall, the measured prediction plan changes median iteration time by 0.3% … while oracle one-layer advice improves it by 5.0%". Executing either plan through a blocking reader is slower than no prefetch. This concerns the SSD→DRAM page cache via `posix_fadvise`, not PCIe→GPU. — [arXiv 2608.12103](https://arxiv.org/abs/2608.12103)
- Our results (brief; job 066 and chat benchmark). — [prefetch_lookahead.md](/home/claude/moe-speed-of-light/research_notes/MoE%20offload%20system%20race%20plan/prefetch_lookahead.md)
  - PREFETCH +10–16% on the PCIe 5.0 host with long prompts; +4–6% on 109-token AIME prompts (−2–3% at 40–44% residency); −16% on PCIe 4.0.
  - Simulated main-stream copy gains ~0%.

### Inferences
- Frame context-length dependence as confirming Speculating Experts' min(t_copy, t_compute) model. Frame the PCIe/DRAM-ratio dependence as the new part.
  - With CPU-executed misses as the alternative, the prefetch copy's cost is not just link time. It also takes DRAM bandwidth from the helpers (Q3), so on a host whose link is half its DRAM bandwidth a mispredicted or unused prefetch costs more than the CPU miss it replaces.
- The short-prompt result (−2–3% at high residency) is consistent with Speculating Experts' mechanism: less attention time to hide the copy. It is explained, not new.

### Gaps
- No prefetch paper evaluates a CPU-miss baseline with PCIe 5.0 on a consumer desktop, so there is no external number for the PCIe 5.0 consumer case.

## Q7. What realized decode speed-ups are reported on consumer GPUs for gpt-oss or Qwen3-MoE (numbers to compare against)?

### Takeaway
- **gpt-oss on consumer GPUs**, few rigorous numbers:
  - a static hot-set llama.cpp fork gets +31.6% (36.2 → 47.6 tok/s) for gpt-oss-120B on RTX 5090 + 9950X3D over `--n-cpu-moe 25`;
  - closed llama.cpp PR #21609 claims 1.79× on an 8 GB RTX PRO 2000;
  - SeqMoE reports gpt-oss-120B only on an RTX PRO 6000 (79.2% of full-load at 45% residency vs FreeToken 57.6%);
  - Speculating Experts' gpt-oss-120B result is on an A100 (PCIe 4.0), at 5–8% TPOT.
- **Qwen3-MoE on consumer GPUs:**
  - FreeToken 77–83 tok/s (Qwen3.6-35B-A3B BF16, RTX 5090), 1.8–2.3× the strongest baseline;
  - SeqMoE 104.1 vs llama.cpp 30.9 tok/s (Qwen3-30B-A3B-FP8, RTX 4090, 45% residency);
  - RFC #24528 +7% to +57%;
  - llama-moe-cache 1.91×.
- **Prefetch-only increments** (the right comparison for PREFETCH's +10–16%):
  - HybriMoE 1.15×;
  - ProMoE 1.09× over LRU inside llama.cpp;
  - Fate +30% average;
  - Speculating Experts 5–14%.

### Cited Findings
- JigSawPT/moe-autopilot (RTX 5090 + 9950X3D + DDR5-6000; static hot set, dual FFN chain): "gpt-oss-120B `--n-cpu-moe 25`, HOT_N=22: 36.2 → 47.6 tok/s (+31.6%)"; Qwen3.6-35B 115.8 → 128.3 (+10.3%). — [moe-autopilot](https://github.com/JigSawPT/moe-autopilot) (as summarised in companion notes)
- llama.cpp PR #21609 (Apr 2026, closed): "96.7% hit rate, 1.79x decode speedup on GPT-OSS-120B with RTX PRO 2000 8GB (--n-cpu-moe 36 --expert-cache-slots 16)". — [PR #21609](https://github.com/ggml-org/llama.cpp/pull/21609) (as summarised)
- Issue #20757 (RTX PRO 2000 8GB, GPT-OSS-120B): "12–14 tok/s at ~98–100% hit". — [Issue #20757](https://github.com/ggml-org/llama.cpp/issues/20757) (as summarised)
- SeqMoE: "At 45% cache capacity … SeqMoE achieves 84.82%, 75.69%, 79.20%, and 76.65% of FullLoad performance on QW3, QW36, GPT, and DSV4 … FreeToken reaches only 55.15%, 57.60%, and 45.45% on QW36, GPT, and DSV4". GPT-OSS-120B-MXFP4 ran on an RTX PRO 6000 Blackwell and QW36 on an RTX 5090 (PCIe 5.0). llama.cpp "Static Offload" was 30.9 tok/s vs SeqMoE 104.1 tok/s (QW3, RTX 4090). — [arXiv 2609.12978](https://arxiv.org/abs/2609.12978)
- FreeToken (RTX 5090 server): "FreeToken sustains 77–83 tok/s on Qwen3.6 and 22–25 tok/s on DSV4-Flash, 1.8–2.3× and 1.5–1.9× the strongest baseline". The 4060 laptop "sustains 39.3 tok/s". — [arXiv 2608.16157](https://arxiv.org/abs/2608.16157)
- Speculating Experts: 5–14% TPOT over on-demand copying; GPT-OSS-120B on an A100 80 GB (PCIe 4.0); Qwen3-30B-A3B on an A6000. — [arXiv 2603.19289](https://arxiv.org/abs/2603.19289)
- leloch RFC #24528 (4× RTX 3090 / sm86 etc.): v2 ablation on Qwen3.6-35B was off 81.55, cache-only 88.12, full 99.26 tok/s. — [Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528) (as summarised)
- ongunm/llama-moe-cache: Qwen3-30B-A3B Q4_K_M on an RTX 4070 Ti 12GB, "33.74 → 64.45 t/s (1.91x), hit 99.50%", against "vanilla". — [llama-moe-cache](https://github.com/ongunm/llama-moe-cache) (as summarised)
- Prefetch-only increments:
  - HybriMoE ablation "+prefetching" 1.15×. — [arXiv 2504.05897](https://arxiv.org/abs/2504.05897)
  - Fate: "an average speedup of 30% from prefetching alone". — [arXiv 2502.12224](https://arxiv.org/abs/2502.12224)
  - ProMoE: 1.09× decode over an LRU cache inside llama.cpp on an RTX 4090. — [arXiv 2410.22134](https://arxiv.org/abs/2410.22134) (via companion note)

### Inferences
- The fairest external comparisons for our +10–16% (PREFETCH) and +22% (FETCH + PREFETCH) on gpt-oss-120b are:
  - Speculating Experts' gpt-oss-120B 5–8% (A100, PCIe 4.0, copy baseline);
  - HybriMoE's 1.15× and ProMoE's 1.09× prefetch increments.

  Ours is in the same band or above it, and plausible.
- moe-autopilot's 47.6 tok/s for gpt-oss-120B on a near-identical host (5090 + 9950X3D) is the most useful *absolute* anchor for our end-to-end tok/s on the 9950X host. It is a static hot set at unstated VRAM; compare at equal VRAM.
- FreeToken does not report gpt-oss results in the paper, although its code has a gpt-oss MXFP4 path. Running FreeToken on gpt-oss-120b on the 9950X host would give the direct external comparison for FETCH.

### Gaps
- No peer-reviewed or arXiv paper reports gpt-oss-120b decode on a GeForce RTX 5090 or 4090 with expert offloading. SeqMoE uses an RTX PRO 6000 and Speculating Experts an A100. Community numbers (moe-autopilot, #21609, #20757) come from summaries of READMEs/PR pages and use unstated or non-equal VRAM.
- The HybriMoE and DALI absolute tok/s against llama.cpp at batch 1 appear only in figures.
