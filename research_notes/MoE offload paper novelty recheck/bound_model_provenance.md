# Prior-art recheck (28 Sep 2026) for three claims of "Seconds, Not Blocks": (A) time-domain speed-of-light bound, (B) validated decode model, (C) trace provenance

Method note:
- Full texts were read locally: arXiv PDFs fetched with curl and grepped after `pdftotext`. Quotes are verbatim from those texts.
- Dates are arXiv version stamps from the PDFs, or the arXiv abs listing where stated.
- "Not found" means not found by about 35 searches plus full-text greps of about 40 papers. It does not prove absence.
- Earlier companion notes (27 Sep: `MoE research direction rethink/time_domain_audit_and_invariance.md`, `benchmarking_and_offpolicy.md`) were read first. This pass tried to falsify them.

Summary of verdicts:

| Claim | Verdict | Closest prior work (date) | What stays ours |
|---|---|---|---|
| A. Time-domain bound | **Partly new.** Each ingredient has prior art, some of it 25–30 years old. Nobody found combines them for MoE, or reports systems as a fraction of such a bound. | 2608.07911 (MIN-with-bypass on MoE traces, blocks only; 8 Aug 2026). CHOPT, SIGMETRICS 2020 (latency-optimal offline placement where slower tiers can serve directly). Albers–Garg–Leonardi, JACM 2000 (elapsed-time-optimal prefetching/caching via LP). FreeToken (fetch-vs-CPU split from bandwidths; 17 Aug 2026). Angelopoulos et al. (layered paging for MoE; 2 Sep 2025). | A closed-form lower bound in seconds for batch-1 offloaded-MoE decode. It is valid for any exact-routing policy (demand or prefetch, per-layer or global budget, CPU execution of misses) and built from per-layer MIN-with-bypass miss counts, multi-memory load accounting and convexity/Jensen. It is used to express published systems as a fraction of the physical speed-of-light. |
| B. Validated decode model | **Partly new.** Validation of an efficiency-scaled roofline on third-party published measurements is done (GenZ, 2024–25), but for dense GPU serving, from one third-party source, with per-platform efficiency fitted in-sample. Pre-registration is used in two Aug 2026 MoE-caching papers, but not for performance predictions. | GenZ (arXiv 2406.01698; v3 15 May 2025). Qiu et al. 2609.14864 (14 Sep 2026; own measurements, leave-one-host-out). LIMINAL (2507.14397; own 8×H100, 7.6% MAPE with fitted exposed latency). 2608.07911 and 2608.18261 (pre-registered criteria). | Out-of-sample (leave-one-group-out) validation of a hybrid CPU/GPU offloaded-MoE batch-1 decode model on 52 rows harvested from many independent papers and threads. Plus a pre-registered, numerically thresholded prediction of held-out performance on a new platform, from a registered two-run calibration. |
| C. Trace provenance | **Mostly new, with two scoping corrections needed.** No one found compares teacher-forced dataset-text traces with self-generated traces for cache hit rates. Adjacent work shows text provenance, template and model identity change routing. The paper's claim that teacher-forced dataset text is "the norm" is **not accurate for cache-systems papers**, most of which trace their own generations. | 2608.07911 (chat-template/workload contamination reverses conclusions; greedy generated traces). 2604.09780 (10 Apr 2026: routing on the same math problems differs by which model wrote the solution, ~60% overlap; gpt-oss prefill "router collapse"). HF transformers issue #40990 (gpt-oss-20b WikiText-2 PPL 239–394). Self-calibration (NAACL 2025). | The paired D/G/S-arm measurement of hit-rate and bound change on 9 models, pre-registered F1–F4, the direction result (own text is less local), and the gpt-oss-120b NLL 7.3 vs 0.65 nats, reproduced across three implementations. |

## Claim A: Has anyone turned optimal-cache miss counts into a time bound, bounded decode time for CPU–GPU hybrid MoE, or reported systems as a fraction of such a bound?

### Takeaway
**Verdict: partly new.**

The specific object is new as far as I can find: a lower bound in seconds on batch-1 offloaded-MoE decode, derived from exact-trace MIN-with-bypass miss counts plus multi-memory load accounting, and then used to express published systems as a fraction of it.

Every ingredient has prior art that must be cited:
- MIN with bypass on MoE traces, in blocks: 2608.07911.
- Layered (per-layer) paging for MoE: Angelopoulos et al. 2025.
- Time-optimal offline prefetching and caching: Cao 1995, Kimbrel–Karlin, Albers–Garg–Leonardi 2000.
- Latency-optimal offline placement when the slow tier can serve directly (bypass with cost): CHOPT 2020.
- Demand-miss-optimal Belady under prefetching: Jain & Lin 2018.
- The fetch-vs-CPU-execute split from bandwidths: FreeToken 2026, with Fiddler earlier.
- Break-even reuse from bandwidths: 2608.12103.
- Single-bandwidth utilisation for MoE: MoE-CAP.
- Batch-1 "fraction of memory floor": 2605.30571.
- Throughput upper bounds for CPU–GPU MoE: MoE-Lens.

The claim should be phrased as a synthesis plus an application, not as the first time-aware Belady.

### Cited Findings
**MoE-specific: optimal-cache counts, never converted to seconds**
- 2608.07911 (Yu Zhang). v1, 8 Aug 2026, was titled "When Does Trace-Driven Evaluation Mislead MoE Expert Caching?". v4, 25 Aug 2026, is titled "Reproducible Evaluation of MoE Expert Caching: Replay Semantics, Workload Contamination, and Operating Regimes".
  - It already uses MIN with bypass: "Belady (Belady 1966) (offline optimum with bypass admission), and Belady-forced-admit". It supports per-layer and global scope: "Under per-layer scope the budget is divided into quotas c_l ... under global scope a single pool serves all layers."
  - It explicitly stays in blocks: "Miss bursts are reported in block counts, not in time. A policy that transfers fewer blocks here may not be faster in a system ... where engineering units appear they are unit conversions applied to the analytical reference frame of §3.2".
  - Its checklist item D1 asks for "absolute transferred bytes per output token alongside any gap, together with the bandwidth budget implied by the target service rate". Its §10.6 says the checklist "says nothing about ... real host-to-device transfer and its overlap with compute ... achievable sustained bandwidth, kernel time". — [arXiv 2608.07911v4](https://arxiv.org/abs/2608.07911); [v1](https://arxiv.org/abs/2608.07911v1)
- 2608.12103 (Si, Lin, Li, Zhang; v1 12 Aug 2026, v2 30 Aug 2026).
  - It compares Belady with LRU and LFU per-layer and globally: "global and per-layer allocation differ by at most 2.7 pp".
  - It turns Belady into a miss-reduction headroom only: "Belady bounds the remaining miss-reduction potential at roughly one third of the misses LRU still incurs".
  - Its only time-domain rule is a per-expert break-even: "Promoting an expert of S bytes to HBM costs one overlapped transfer S/346 and saves S (1/327 − 1/2938) on each subsequent use ... The breakeven reuse count is r* = ... = 1.06".
  - It also does "capacity planning in the style of the five-minute rule". It gives no lower bound on decode time. — [arXiv 2608.12103](https://arxiv.org/abs/2608.12103)
- Angelopoulos, Marchal, Obrecht, Simon, "Cache Management for Mixture-of-Experts LLMs – extended version" (arXiv 2509.02408, 2 Sep 2025).
  - It introduces "a new paging problem that models expert management optimization. Our formulation captures both the layered architecture of LLMs", and gives "lower bounds on the competitive ratio of both deterministic and randomized algorithms".
  - It uses Belady's rule as the offline optimum ("an optimal offline strategy Opt ... that follow Belady's rule") on Mixtral-8x7B and Llama-MoE traces.
  - The objective is miss counts (competitive ratio), not time.
  - 2608.12103 summarises it as showing "that fixed per-layer cache allocations can have an unbounded competitive ratio". — [arXiv 2509.02408](https://arxiv.org/abs/2509.02408); [2608.12103 related work](https://arxiv.org/abs/2608.12103)
  - **Not in our refs.bib; must cite** (per-layer vs shared budget).
- SAEM (2608.21614, 21 Aug 2026) builds a "clairvoyant oracle" for its own stage-level cache planner. That is an oracle for one policy class, reported as throughput, not a policy-independent bound. — [arXiv 2608.21614](https://arxiv.org/abs/2608.21614)

**Classic caching theory: time-domain offline optima already exist**
- Cao, Felten, Karlin, Li (SIGMETRICS 1995; already cited as `cao1995prefetching`): "We prove that the performance of the conservative approach is within a factor of two of optimal". The measure is elapsed or running time ("can reduce the running time of applications by up to 50%"). — [Cao et al. 1995 PDF](https://homes.cs.washington.edu/~karlin/papers/sigmetrics.pdf)
- Kimbrel & Karlin, "Near-optimal parallel prefetching and caching": "We study the offline problem of constructing an optimal prefetching schedule in this model, for a given request stream." — [Kimbrel & Karlin PDF](https://homes.cs.washington.edu/~karlin/papers/tracy.pdf); [SIAM J. Comput.](https://epubs.siam.org/doi/10.1137/S0097539797326976)
- Albers, Garg, Leonardi, "Minimizing stall time in single and parallel disk systems" (JACM, Nov 2000): "The total elapsed time is the sum of the processor stall times and the length of the request sequence ... an optimum prefetching/caching schedule for a single disk problem can be computed in polynomial time ... formulating the prefetching/caching problems as linear programs." — [JACM 10.1145/355541.355542](https://dl.acm.org/doi/10.1145/355541.355542) (abstract via Crossref)
- Jain & Lin, "Rethinking Belady's Algorithm to Accommodate Prefetching" (ISCA 2018): "while Belady's MIN algorithm minimizes the total number of cache misses— including those for prefetched lines—it does not minimize the number of demand misses. To address this shortcoming, we introduce Demand-MIN". — [ISCA'18 PDF](https://www.cs.utexas.edu/~lin/papers/isca18.pdf)
  - This bears directly on our proof step "a prefetching schedule bridges no more gaps than some demand schedule with bypass". **Not cited; should be.**
- Zhang, Karimi, Ahmad, Vigfusson, "Optimal Data Placement for Heterogeneous Cache, Memory, and Storage Systems" (CHOPT, SIGMETRICS/POMACS 2020):
  - "which data should be cached in faster memory if it could instead be served directly from slower memory? We present Chopt, an offline algorithm for data placement across multiple tiers of memory with asymmetric read and write costs. We show that Chopt is optimal and can therefore serve as the upper bound of performance gain for any data placement algorithm ... optimal data placement decisions could improve average request latency by 8.2%-44.8% when compared with ... Belady and Mattson's offline, evict-farthest-in-the-future optimal algorithms." — [CHOPT PDF](https://geraldleizhang.com/publications/CHOPT_Sigmetrics20.pdf); [ACM 10.1145/3379472](https://dl.acm.org/doi/10.1145/3379472)
  - This is the closest caching-theory analogue of "misses either loaded or served from the slower memory, whichever is cheaper". **Not cited; must be.**
- Mandarapu & Kunkunuru, "Caching for Dollars, Not Hits: An Exact Offline Reference for Cloud-Egress Caching" (arXiv 2606.20539, v2 19 Jun 2026):
  - "Classic caching minimizes the miss rate, the wrong objective ... For uniform-size page caches with heterogeneous miss costs the offline dollar-optimum is exact in polynomial time via an integral interval linear program".
  - Same genre and a strikingly similar title pattern to "Seconds, Not Blocks", in a different domain. — [arXiv 2606.20539](https://arxiv.org/abs/2606.20539)

**MoE and LLM decode: bandwidth models, splits and bounds that stop short of ours**
- FreeToken (2608.16157, 17 Aug 2026; already cited) splits misses between PCIe fill and CPU execution from measured bandwidths:
  - "The optimal split ratio is derived from a residual-bandwidth argument ... a saturated PCIe transfer leaves a residual bandwidth of B_R = max(B_H − B_P, 0) ... T_fill(q) ≈ qS/B_P, T_cpu(m − q) ≈ (m − q)S/(B_H − B_P)", with q* = m·B_P/B_H.
  - This is the per-step, time-optimal fetch-vs-execute split, used as an online policy. It is not combined with an optimal-cache count and is not a bound. It also already accounts for host DRAM serving both the DMA bytes and the CPU-read bytes. — [arXiv 2608.16157](https://arxiv.org/abs/2608.16157)
- MoE-Lens (2504.09345, v1 12 Apr 2025) sets out to answer "what is the upper bound on achievable performance". It reports "our theoretical model predicting performance with an average 94% accuracy". This is batched high-throughput CPU–GPU serving with no expert-cache bound. — [arXiv 2504.09345](https://arxiv.org/abs/2504.09345)
- MoE-Lightning (2411.11217, v1 18 Nov 2024; ASPLOS'25) introduces "HRM, a general performance model for LLM inference which extends the Roofline Model", used to choose throughput pipelines. — [arXiv 2411.11217](https://arxiv.org/abs/2411.11217)
- MoE-CAP (2412.07067 v6, 19 Nov 2025; NeurIPS'25) defines S-MBU = B_achieved/B_peak with B_achieved = (S_activated + S_KV)/TPOT, where "1[l, i] can be achieved by tracing router outputs".
  - This is a fraction-of-peak measure against a single memory. It has no multi-tier or cache bound.
  - Its profiler "runs on representative data until the activation distribution stabilizes". — [arXiv 2412.07067](https://arxiv.org/abs/2412.07067)
- "Memory-Bound but Not Bandwidth-Limited: The Physical AI Inference Gap in Batch-1 LLM Decode" (Josef Chen, 2605.30571, 28 May 2026):
  - "an L4 reaches roughly 81% of its analytic memory floor, while an H100 reaches only 27%".
  - This is a prior "fraction of speed-of-light for batch-1 decode" report, but for dense models on GPU only, with its own measurements. **Not cited; should be**, as the dense analogue of our "systems are far from the physical speed-of-light". — [arXiv 2605.30571](https://arxiv.org/abs/2605.30571)
- LIMINAL (Davies, Crago, Sankaralingam, Kozyrakis; 2507.14397 v2, 13 Nov 2025) is an "analytical performance model" to "explore the limits of LLM inference ... in auto-regressive decoding". It covers GPU and HBM-class systems, expert parallelism for MoE, and no host offload or cache. — [arXiv 2507.14397](https://arxiv.org/abs/2507.14397)
- SeqMoE (2609.12978, 11 Sep 2026; already cited) already reports an offloading system as a fraction of an all-GPU ceiling: "96.97% hit rate and 80.22% of full-load performance". This is an implementation-relative ceiling, which weakens any novelty claim for our implementation-relative speed-of-light (the physical one is unaffected). — [arXiv 2609.12978](https://arxiv.org/abs/2609.12978)
- TriMoE (2603.01058) and CoX-MoE (2605.17889, v2 19 May 2026) use a "GPU-CPU-NDP roofline" and a "GPU and AMX Roofline Model" to choose their own throughput placements ("The optimal single-layer latency (T_opt) aims to minimize..."). These are not bounds over policies. — [arXiv 2603.01058](https://arxiv.org/abs/2603.01058); [arXiv 2605.17889](https://arxiv.org/abs/2605.17889)
- 2608.18261 (Suram, 18 Aug 2026): "measured decode is 0.44 tok/s warm, in exact agreement with a bytes-per-token ÷ bandwidth model". This covers one configuration with experts streamed from NVMe. There is no cache bound. — [arXiv 2608.18261](https://arxiv.org/abs/2608.18261)
- LLM-Viewer ("LLM Inference Unveiled: Survey and Roofline Model Insights", 2402.16363) is a roofline survey tool. Offloading appears only as surveyed technique. — [arXiv 2402.16363](https://arxiv.org/abs/2402.16363)

### Inferences
- **Nobody found does the MoE time-domain step**, meaning Belady-type miss counts turned into seconds for offloaded MoE with CPU execution as the alternative. **But "converting optimal caching into time" is not new in general.** Cao 1995, Kimbrel–Karlin and Albers–Garg–Leonardi 2000 compute or approximate elapsed-time-optimal prefetching and caching. CHOPT computes latency-optimal offline placement where the slow tier may serve directly, and shows Belady is not latency-optimal there.
  - A reviewer from the caching-theory side will ask why we bound rather than solve, since Albers et al. solve the single-disk case exactly by LP and CHOPT by min-cost flow.
  - Suggested answer: CPU and GPU run concurrently with a max(·) per layer-step, bandwidths are shared, and the bound must cover prefetching. A closed-form lower bound that is valid for all policies is the object an audit needs; an exact optimum for one execution model is not.
- The per-miss choice "load over PCIe or read on the CPU, whichever is cheaper" is Fiddler's and FreeToken's online rule. Fiddler is already cited as `kamahori2024fiddler`; FreeToken as `freetoken`. Consequence (i) of our bound says the optimum runs part of even resident experts on the CPU, i.e. CPU+GPU bandwidth aggregation. That is close in spirit to FreeToken's residual-bandwidth argument. The paper should say that what is new is the bound's validity over all policies, not the split.
- **What remains ours for A:**
  1. The proposition itself: a policy-independent lower bound on mean time per token for exact-routing, per-layer or global-budget, demand or prefetch, with fetch or CPU-execute. It combines M* from MIN-with-bypass (with the prefetch-reduction step, which should cite Jain & Lin and Cao et al.), multi-memory byte accounting, convexity/Jensen over layer-steps and link capacity.
  2. The "physical speed-of-light" (all η = 1, datasheet peaks) evaluated on real traces.
  3. Using it to re-express published offloading systems and their llama.cpp baselines as fractions of it. For batch-1 hybrid MoE, nobody found does this. The dense analogue is 2605.30571, and single-memory utilisation is MoE-CAP's S-MBU.
- The title "Seconds, Not Blocks" pointedly answers 2608.07911, and 2606.20539 uses a "Dollars, Not Hits" title. Neither is a problem, but citing 2606.20539 pre-empts a "derivative framing" remark.

### Gaps
- The full text of Albers–Garg–Leonardi (ACM returned 403) was not read. Only the Crossref abstract was used.
- Whether any variant of CHOPT, or the "offline optimal with bypass and costs" literature, handles concurrent service from two tiers (max rather than sum) was not checked. If one does, the Jensen/convexity step has an antecedent too.
- Fiddler's latency model was not re-read in this pass; it is cited in the paper and in companion notes.
- Splitwise and "Mind the Memory Gap" (2503.08311) were checked only superficially. They concern GPU batching and phase splitting, not offload bounds. The full text of "Mind the Memory Gap" was fetched and has no offload bound.
- Workshop papers (MLSys/NeurIPS/SC'26 workshops) and non-arXiv venues are not systematically indexed.

## Claim B: Has anyone validated an analytical LLM/MoE decode model on third-party published measurements, or pre-registered performance predictions in systems research?

### Takeaway
**Verdict: partly new.** The earlier (27 Sep) conclusion that third-party validation is new is **falsified in its general form**:
- GenZ (Georgia Tech/Meta; arXiv 2406.01698, v3 15 May 2025) validated a roofline-with-efficiency-factors LLM inference model against third-party published measurements (Argonne's LLM-Inference-Bench) for SN40L, MI300X and Gaudi2 at 5.82% geomean error.
- Limits of GenZ's validation: dense Llama3-8B at batch 16, one external source, a per-platform efficiency factor that appears fitted on the same data, and no offload.

Pre-registration itself is not new either:
- NeurIPS 2020/2021 pre-registration workshops in ML.
- In this exact literature, 2608.07911 (thresholds frozen before measurement) and 2608.18261 (pre-registered negative result).
- Neither pre-registers numeric performance predictions for a new platform.

What stays ours:
- Leave-one-group-out, out-of-sample validation on 52 rows from many independent sources.
- Hybrid CPU/GPU offloaded MoE at batch 1.
- A registered calibration procedure whose held-out prediction error was pre-committed (≤10%) and met.

### Cited Findings
**Validation on third-party published data (closest precedent)**
- GenZ, "Demystifying AI Platform Design for Distributed Inference of Next-Generation LLM models" (Bambhaniya et al., arXiv 2406.01698, v3 15 May 2025).
  - Model form: "a roofline-based approach combined with separate efficiency factors ... for computation FLOPs and memory BW", i.e. T_op = max(C_op/(FLOPS·Eff_C), M_op/(BW_mem·Eff_mem)). This is the same functional family as our per-memory η.
  - Own-hardware validation: "Fig. 6 compares actual end-to-end serving times against GenZ estimates, yielding a geomean error of 1.43%."
  - **Third-party validation:** "We also validate GenZ against three other popular architectures: (i) 8xSambanova SN40L, (ii) 1xAMD MI300X running vLLM and (iii) 1xIntel Gaudiv2 running deepspeed ... GenZ achieves a geomean error of 5.82% across all different architectures." Footnote: "We were unable to get access to the physical node for these architecture, so we used the number from LLM-Inference-Bench [72]. The raw data was accessed from https://github.com/argonne-lcf/LLM-Inference-Bench."
  - Per-platform efficiencies are stated: "SN40L uses Sambaflow framework (Eff=0.9), MI300X uses vLLM (Eff=0.25) and Gaudi2 uses deepspeed (Eff=0.6)". No held-out protocol is described. — [arXiv 2406.01698](https://arxiv.org/abs/2406.01698)
  - **Not in our refs.bib; must cite**, and differentiate on out-of-sample protocol, number of sources, MoE hybrid offload and batch 1.
- Qiu et al., "GGUF-Metadata Prediction of Single-Sequence llama.cpp Throughput Across Three Systems" (2609.14864, 14 Sep 2026; already cited as `qiu2026gguf`).
  - The model uses per-host, per-quant bandwidth efficiencies: "η_{h,q} is the median of T_d D/B_h over training rows".
  - Evaluation: host-specific held-out sets. "Leave-one-host-out coefficients fitted on the other two systems yield 11.6%, 16.8%, and 36.0% test MAPE". It includes MoE held-out models, and "three pre-specified reference-probe exclusions".
  - All measurements are the authors' own and all-GPU or unified memory. There is no `--n-cpu-moe` hybrid.
  - This is methodologically very close to our model and must be positioned carefully. — [arXiv 2609.14864](https://arxiv.org/abs/2609.14864)
- LIMINAL (2507.14397, v2 13 Nov 2025) validated on its own 8×H100 vLLM runs, with a fitted per-layer exposed latency: "we fit the exposed latency to measured data ... a mean absolute percent error (MAPE) of 7.6%". — [arXiv 2507.14397](https://arxiv.org/abs/2507.14397)
- AMD LIFE, "Forecasting LLM Inference Performance via Hardware-Agnostic Analytical Modeling" (2508.00904, 29 Jul 2025), validated on own hardware: "We validate LIFE's forecasting with inference on AMD Ryzen CPUs, NPUs, iGPUs and NVIDIA V100 GPUs, with Llama2-7B variants". — [arXiv 2508.00904](https://arxiv.org/abs/2508.00904)
- WattGPU (2607.02391, early Jul 2026) predicts ITL on unseen GPUs with "rigorous leave-one-GPU-out and leave-one-LLM-out cross-validation". It is ML-based, and the data is the authors' own Watt Counts dataset (ref. [22], same authors, arXiv 2604.09048). It explicitly excludes MoE: "leave 8 Mixture-of-Experts models as future work, as their inference exhibits different patterns that require separate modeling". — [arXiv 2607.02391](https://arxiv.org/abs/2607.02391)
- MoE-Lens: "our theoretical model predicting performance with an average 94% accuracy". This is on its own system (already noted in brief). — [arXiv 2504.09345](https://arxiv.org/abs/2504.09345)
- LBNL, "Comprehensive Performance Modeling and System Design Insights for Foundation Models" (2410.00273, 30 Sep 2024) validated on own runs: "For sub-optimal configurations, with different TP/PP/DP, the error ranges from 11–26%". It covers training, not decode. — [arXiv 2410.00273](https://arxiv.org/abs/2410.00273)

**Pre-registration**
- In this MoE literature:
  - 2608.07911: "Every threshold presented as a confirmatory decision criterion was pre-registered and frozen before the corresponding measurement was read ... We release the decision documents unchanged, including those in which a criterion failed". Its thresholds are on cache-simulation outcomes, not time predictions. — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)
  - 2608.18261: "A Pre-Registered Negative Result". Its "Pre-registered criteria. RQ1 (locality): ≥30% reduction in LRU miss/token at 25% capacity and validation perplexity within +1%". The bytes/bandwidth tok/s prediction in the same paper is not described as pre-registered. — [arXiv 2608.18261](https://arxiv.org/abs/2608.18261)
  - 2609.14864 uses "pre-specified reference-probe exclusions" (data-cleaning rules, not predictions). — [arXiv 2609.14864](https://arxiv.org/abs/2609.14864)
- In ML generally: NeurIPS 2020 and 2021 Workshops on Pre-registration in Machine Learning (PMLR vols. 148 and 181). — [PMLR v148](https://proceedings.mlr.press/v148/); [PMLR v181](https://proceedings.mlr.press/v181/); [preregister.science](https://preregister.science/)
- Other Sep 2026 pre-registered audit: 2609.04198, an LLM-judge reliability audit (from companion notes). — [arXiv 2609.04198](https://arxiv.org/abs/2609.04198)

### Inferences
- **Blunt:** "first to validate an analytical LLM decode model on third-party published measurements" would be false. GenZ did it in 2024–25, from one third-party dataset. What stays defensible:
  - (i) **Out-of-sample** validation (leave-one-group-out). GenZ's per-platform Eff appears set on the validated platforms.
  - (ii) **Many independent sources** (52 rows harvested from papers and threads), not one benchmark repository.
  - (iii) **Hybrid CPU/GPU offloaded MoE at batch 1**, with host-DRAM and PCIe terms. GenZ, LIMINAL, LIFE and WattGPU are GPU/accelerator-resident. Qiu et al. is all-GPU and unified memory on the authors' own hosts. WattGPU explicitly defers MoE.
  - (iv) A reported median error (16%) at a scale that is honest about third-party noise.
- The **pre-registered on-platform prediction** (two-run calibration registered with a ≤10% pass threshold, then 33 held-out configurations at 2.9% median) has no precedent I could find for performance predictions. Pre-registration as a practice is precedented in this very niche (2608.07911, 2608.18261). The paper should not imply that pre-registration is novel in MoE-offloading research; the novelty is pre-registering numeric performance predictions and their accuracy thresholds.
- Qiu et al. (14 Sep 2026) is the nearest competitor on method: medians of bandwidth-efficiency ratios and leave-one-host-out. A reviewer may see our model as "Qiu et al. plus a CPU term". The differentiators are the hybrid offload terms, third-party data and the pre-registered transfer test.

### Gaps
- GenZ's text does not say whether the SN40L, MI300X and Gaudi2 Eff values were fitted on the same LLM-Inference-Bench rows used for validation. "Appears in-sample" is my reading of the stated per-platform Eff values, not a quote.
- LLMCompass (ISCA 2024) and Vidur were not re-read. From general knowledge they validate on their own hardware runs, but this pass did not verify it.
- The Davies et al. MLPerf longitudinal study, cited by LIMINAL, analyses published MLPerf results. It was not read, so it is unknown whether it fits a performance model to them.
- Software-engineering registered reports (MSR/ESEM) were not surveyed. They are precedent for pre-registration in computing, but not for performance prediction.

## Claim C: Has anyone studied how the text used to collect routing traces (dataset vs self-generated, on/off-policy, teacher forcing vs sampling) changes expert locality, cache hit rates or conclusions?

### Takeaway
**Verdict: the measurement is mostly new; the framing needs two corrections.**

I found no paper that pairs teacher-forced dataset text with the model's own generations on the same prompts and measures the change in LRU, DFA or MIN hit rates or in a time bound.

Adjacent work that must be cited:
- 2608.07911 shows that generation-side provenance choices (chat template, template multiplicity, shared prefixes) reverse cache conclusions. Its traces are greedy self-generated with the chat template.
- 2604.09780 (JHU, 10 Apr 2026) shows that which model wrote the text changes routing a lot. Solutions to the same problems by different models overlap only ~60% in experts, and gpt-oss routing during prompt prefill "collapses" and diverges only once generation proceeds.
- R3 (Oct 2025) and SAEM (Aug 2026) show that replaying the same tokens can give different expert selections across engines and forward passes.
- A 2025 HF transformers issue documents gpt-oss-20b WikiText-2 perplexity of 239–394, attributed to the harmony and chat format.

**Scoping correction:** the paper states that teacher-forcing dataset text is "the norm in this literature" and "as in most trace studies". That holds for **routing-analysis, locality and router-training** papers. **Most cache-systems papers trace the model's own generations**: 2608.07911, 2608.12103, SeqMoE, MoE-Beyond, SAEM, Mixtral-Offloading, Fate and SpecMD.

### Cited Findings
**Teacher-forced dataset text is common in routing and locality analyses**
- Mixtral technical report (2401.04088, 8 Jan 2024): "we measure the distribution of selected experts on different subsets of The Pile validation dataset"; "consecutive tokens are often assigned the same experts ... we observe some degree of positional locality in The Pile datasets". — [arXiv 2401.04088](https://arxiv.org/abs/2401.04088)
- "Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models" (ICLR 2026; 2505.16056 v4 28 Feb 2026).
  - Corpus: "all 7 categories from RedPajama ... arena-human-preference-140k ... OpenMathInstruct-2 ... OpenCodeInstruct ... OpenScienceReasoning-2". "For OpenMath, OpenCode, and OpenScience, we simply concatenate the input and output of each instance ... cutting them into input sequences of 512 tokens". That means teacher-forced text, including other models' outputs.
  - It relates its segment cache hit metric (SCH) to "the optimal cache hit rate given by the clairvoyant replacement algorithm".
  - It is a strong example of the norm and of cache-relevant conclusions drawn from dataset text. **Not cited; should be.** — [arXiv 2505.16056](https://arxiv.org/abs/2505.16056)
- Huang et al., "Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference" (2303.06182 v2, 18 Jun 2023): "For Language Modeling, we use the PILE dataset as the input, which is the validation set used in prior work". It reports "Worst-Case Cache Miss Rate obtained from traces of expert activations". — [arXiv 2303.06182](https://arxiv.org/abs/2303.06182)
- ReMoE (2605.27081, 26 May 2026): "Routing trajectories under teacher forcing (21st MoE layer of DeepSeek-V2-Lite)". It justifies the choice: "teacher forcing fixes the input token sequence and the time index, so any difference between trajectories reflects a change in routing policy rather than generation drift". — [arXiv 2605.27081](https://arxiv.org/abs/2605.27081)
- Mixture of Cache-Conditional Experts (TMLR, 2412.00099 v2 24 Jun 2025) reports cache miss rate with WikiText perplexity: "we concatenate WikiText text into a single blob ... and chunk it". — [arXiv 2412.00099](https://arxiv.org/abs/2412.00099)
- Sticky Routing (2607.08780): "All experiments use the WikiText-2 raw character dataset". Small trained models. — [arXiv 2607.08780](https://arxiv.org/abs/2607.08780)
- 2608.18261 traces Qwen3-30B-A3B "over 8,000 tokens each of prose, code, math, and medical text" with a llama.cpp eval-callback. It does not say whether the text was generated or teacher-forced. — [arXiv 2608.18261](https://arxiv.org/abs/2608.18261)
- DALI uses "C4 with WikiText calibration", per 2608.07911's audit table. — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)

**Cache-systems papers mostly trace their own generations (counter-evidence to "the norm")**
- 2608.07911 §3.3: "Decode routing was collected autoregressively with the model's chat template applied and reasoning mode disabled, greedy decoding ... Applying the chat template prevents the model from continuing the instruction text instead of answering it ... An earlier collection that did neither is retained and re-analysed in §5 as the contaminated condition." Its limitation: "All traces use greedy decoding, which is reproducible but does not sample production generation diversity." — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)
- 2608.12103: "The protocol records only decode ... Greedy decoding makes the recorded access streams reproducible; the results do not establish behavior under stochastic sampling." — [arXiv 2608.12103](https://arxiv.org/abs/2608.12103)
- SeqMoE: "The first pass collects token ID traces through continuous batching. The second packs multiple token ID traces into a single prefill pass to dump expert activations". This is the same generate-then-prefill-replay technique as our G arm. — [arXiv 2609.12978](https://arxiv.org/abs/2609.12978)
- SAEM (2608.21614): "We first execute the model normally and record the generated token sequence ... then replay the recorded sequence". The advantage is "rising to 1.54× when calibration data matches the workload", i.e. calibration provenance matters to its speedup. — [arXiv 2608.21614](https://arxiv.org/abs/2608.21614)
- 2608.07911's audit of ten papers records SpecMD ("autoregressive generation"), Mixtral-Offloading ("OpenAssistant conversations with autoregressive sampling"), Fate ("decode output up to 1024") and MoE-Beyond (Puffin prompts). MoE-Beyond is on-policy per the companion notes. — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)

**Evidence that text provenance changes routing (closest to our finding)**
- "The Myth of Expert Specialization in MoEs" (Wang, Hayou, Nalisnick; 2604.09780, 10 Apr 2026). **Not cited; must be.**
  - "Balunović et al. (2025) benchmarked a wide range of reasoning LLMs on math datasets, enabling us to study expert activation patterns across reasoning trajectories produced by different models on the same problems ... solutions from two different models to the same question show only ∼60% expert overlap on average, comparable to the overlap between solutions from the same model on different questions ... solving the same problem with a different model is as 'foreign' as solving an entirely different problem."
  - On gpt-oss-20b: "during prefilling, semantically unrelated sequences can activate exactly the same experts ... The picture changes once the model begins generating ... the pruned experts are highly consequential for the model output's (conditional) likelihoods, both on- and off-policy ... We hypothesize that router collapse during prefilling is a property arising from gpt-oss's structured reasoning output ... Removing the chat template also reduces the collapse". It also finds "prompt-level routing does not predict rollout-level routing".
  - It does not measure cache hit rates or offload time, and does not pair a model's own text against dataset reference text for caching. — [arXiv 2604.09780](https://arxiv.org/abs/2604.09780)
- 2608.07911, workload contamination: "a matched-pair rendering intervention moves the measured early-window effect by 19.4–31.9 percentage points and reverses which workloads appear most cache-friendly". This is a provenance axis inside generated text (template and prefix), not dataset vs own text. — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)
- R3, "Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers" (2510.11370 v2, 21 Oct 2025): "We analyze the training-inference consistency of MoE models and identify a notable discrepancy in routing behaviors between the two phases. Moreover, even under identical conditions, the routing framework can yield divergent expert selections across repeated forward passes."
  - This is directly relevant to our claim that teacher-forced prefill reproduces decode routing "up to floating-point ties". **Not cited; should be.** — [arXiv 2510.11370](https://arxiv.org/abs/2510.11370)
- SAEM: "Atomic accumulation in the MoE combine operation can introduce small run-to-run numerical differences, causing unconstrained autoregressive generation to diverge". — [arXiv 2608.21614](https://arxiv.org/abs/2608.21614)
- DuoServe-MoE (2509.07379) documents a prefill-vs-decode phase disparity. This is a phase effect, not text provenance (per companion notes). — [arXiv 2509.07379](https://arxiv.org/abs/2509.07379)

**gpt-oss off-distribution on raw dataset text (prior community evidence)**
- HF transformers issue #40990, "Extremely high perplexity on openai/gpt-oss-20b with WikiText-2 (raw)":
  - Reported "perplexity: 394.467" and "PPL of wikitext2: 239.435", and "around 300 ppl" with harmony wrapping attempted.
  - Maintainer explanation: "This model is not a base model ... Gpt-oss-20b is instructed in chat tunned not in plain LM, we can do wrap text in the Harmony format and score assistant completions only."
  - Date not retrieved (GitHub API blocked); likely Sep 2025 from the issue number. — [transformers #40990](https://github.com/huggingface/transformers/issues/40990)
  - This pre-empts any claim that "gpt-oss is off-distribution on raw dataset text" is a discovery. Our specific result is new: 120b, 7.3 vs 0.65 nats with its own reasoning inserted, three implementations, classified before measurement.

**Analogous result in compression: self-generated data represents the model better**
- Williams, Chrysostomou, Aletras, "Self-calibration for Language Model Quantization and Pruning" (2410.17170 v2, 26 Feb 2025; NAACL 2025): "Conventionally, this is randomly sampled web text ... unrepresentative calibration examples can harm model performance ... we propose self-calibration ... leveraging the model itself to generate synthetic calibration data ... frequently outperforming even using real data." — [arXiv 2410.17170](https://arxiv.org/abs/2410.17170); [ACL Anthology](https://aclanthology.org/2025.naacl-long.509/)

### Inferences
- **What stays ours for C:**
  1. The first paired measurement of dataset-teacher-forced vs self-generated (greedy and sampled) routing traces for **cache hit rates (LRU/DFA/MIN-bypass) and for a time bound**, on 9 models, with pre-registered "conclusion-change" thresholds F1–F4.
  2. The direction result: own text is less local, so dataset traces are optimistic at small budgets, while the bound does not move.
  3. The gpt-oss-120b quantification (7.3 vs 0.65 nats), cross-checked in vLLM, an fp32 collector and llama.cpp, with the cause classified before measurement.
- **Must fix before submission:** scope "the norm" to routing-locality, characterization and router-training studies (Mixtral report, ICLR'26 routing consistency, Huang 2023, ReMoE, Cache-Conditional Experts, Sticky Routing, likely 2608.18261, calibration sets like DALI's). Cache-system evaluations mostly generate. Otherwise a reviewer who knows 2608.07911, 2608.12103 or SeqMoE will cite them against the sentence. The impact statement then becomes: "dataset traces, common in locality analyses and cheap GPU-free tracing, overstate hit rates; system papers that generate are safe on this axis, though greedy-only (2608.07911 and 2608.12103 both state greedy-only limits)".
- The "replay own text via prefill equals decode" method is shared with SeqMoE (two-pass) and SAEM (fixed-sequence replay). It is not a contribution. R3 and SAEM show that exact equality is not guaranteed across engines or runs, which supports reporting the replay-vs-decode agreement rate rather than asserting it.
- 2604.09780's finding that ~60% overlap across authoring models is as foreign as a different problem is the strongest mechanistic support for our result. Dataset reference answers are "another author's" text, so lower or different locality is expected. Citing it turns our finding from a surprise into a quantified consequence. It also explains the gpt-oss prefill collapse, which bears on why dataset text might look more local.

### Gaps
- It was not verified whether 2608.18261's 8,000-token domain traces are teacher-forced (the text is ambiguous, and its repo was not inspected).
- MoE-Infinity's and HOBBIT's locality analyses were not checked for text provenance in this pass. Greps found no "teacher" wording.
- The date of HF transformers issue #40990 is unknown (API blocked). Other community reports of gpt-oss perplexity (llama.cpp discussions) were not searched exhaustively.
- Blog posts and workshop papers comparing on-policy and off-policy routing may exist without being indexed.

## Strong prior work we should cite but might have missed (not in paper/refs.bib as of 28 Sep 2026)

### Takeaway
About a dozen works are missing and material. Five are **must-cite** because a reviewer would raise them:
- Angelopoulos et al. 2025 (layered paging for MoE)
- CHOPT (SIGMETRICS 2020)
- GenZ (third-party validation precedent)
- 2604.09780 "Myth of Expert Specialization" (provenance changes routing)
- "Not All Models Suit Expert Offloading" (ICLR 2026)

The rest strengthen specific sentences.

### Cited Findings
- **Must-cite:**
  - Angelopoulos et al., layered paging for MoE, competitive ratios, Belady OPT on MoE traces — [arXiv 2509.02408](https://arxiv.org/abs/2509.02408)
  - CHOPT, offline latency-optimal multi-tier placement with direct service from slower memory, beats Belady on latency — [SIGMETRICS/POMACS 2020](https://dl.acm.org/doi/10.1145/3379472)
  - GenZ, efficiency-factor roofline validated on third-party LLM-Inference-Bench data — [arXiv 2406.01698](https://arxiv.org/abs/2406.01698)
  - The Myth of Expert Specialization, routing depends on which model wrote the text, and gpt-oss prefill collapse — [arXiv 2604.09780](https://arxiv.org/abs/2604.09780)
  - Not All Models Suit Expert Offloading (ICLR'26), locality and clairvoyant-cache metrics from teacher-forced corpus text over 20 MoEs — [arXiv 2505.16056](https://arxiv.org/abs/2505.16056)
- **Should-cite (theory, for the proof):**
  - Albers–Garg–Leonardi, elapsed-time-optimal prefetching/caching by LP — [JACM 2000](https://dl.acm.org/doi/10.1145/355541.355542)
  - Kimbrel & Karlin — [SIAM J. Comput.](https://epubs.siam.org/doi/10.1137/S0097539797326976)
  - Jain & Lin, Demand-MIN — [ISCA 2018](https://www.cs.utexas.edu/~lin/papers/isca18.pdf)
- **Should-cite (fraction-of-bound and decode limits):**
  - Batch-1 dense decode as a fraction of memory floor — [arXiv 2605.30571](https://arxiv.org/abs/2605.30571)
  - LIMINAL — [arXiv 2507.14397](https://arxiv.org/abs/2507.14397)
  - "Caching for Dollars, Not Hits", same genre and title pattern — [arXiv 2606.20539](https://arxiv.org/abs/2606.20539)
- **Should-cite (provenance and replay):**
  - R3, routing mismatch across engines and repeated forward passes — [arXiv 2510.11370](https://arxiv.org/abs/2510.11370)
  - SAEM, generate-then-replay, and calibration mismatch changes speedup — [arXiv 2608.21614](https://arxiv.org/abs/2608.21614)
  - Mixtral report, Pile-based locality — [arXiv 2401.04088](https://arxiv.org/abs/2401.04088)
  - Huang et al. 2023, PILE-based cache miss rates — [arXiv 2303.06182](https://arxiv.org/abs/2303.06182)
  - ReMoE, explicit teacher-forced traces — [arXiv 2605.27081](https://arxiv.org/abs/2605.27081)
  - Self-calibration, self-generated data better represents the model — [arXiv 2410.17170](https://arxiv.org/abs/2410.17170)
  - gpt-oss-20b WikiText PPL issue — [transformers #40990](https://github.com/huggingface/transformers/issues/40990)
- **Optional:**
  - WattGPU, leave-one-GPU-out ITL prediction that excludes MoE — [arXiv 2607.02391](https://arxiv.org/abs/2607.02391)
  - AMD LIFE — [arXiv 2508.00904](https://arxiv.org/abs/2508.00904)
  - NeurIPS pre-registration workshops — [PMLR v148](https://proceedings.mlr.press/v148/)
  - 2608.07911 v1 title, "When Does Trace-Driven Evaluation Mislead MoE Expert Caching?", worth knowing when searching or citing versions — [arXiv 2608.07911v1](https://arxiv.org/abs/2608.07911v1)

### Inferences
- The largest reviewer risks are the following.
  - (1) A caching theorist citing CHOPT and Albers et al. against "first time-domain optimal-cache bound". Mitigation: frame it as a policy-independent lower bound for concurrent CPU/GPU MoE execution, and cite them.
  - (2) An inference-modeling reviewer citing GenZ against "first validation on third-party data". Mitigation: claim out-of-sample, multi-source, hybrid-MoE, batch-1.
  - (3) A cache-systems reviewer citing 2608.07911, 2608.12103 or SeqMoE against "teacher-forcing is the norm". Mitigation: scope the sentence.

### Gaps
- The citation graph was checked only partly:
  - The Semantic Scholar API returned **zero indexed citing papers for 2608.07911** on 28 Sep 2026 — [S2 API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2608.07911/citations)
  - Lookups for 2608.12103 and 2509.02408 were rate-limited (HTTP 429).
  - Indexing lags by weeks, so a September 2026 follow-up that cites 2608.07911 and adds a time domain could still exist and not have surfaced in web search.
