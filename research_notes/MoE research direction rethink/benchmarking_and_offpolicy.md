# Prior art and open gaps for (A) a scientific-benchmarking / meta-science paper on MoE offloading and hybrid CPU/GPU inference claims, and (B) off-policy teacher-forcing pitfalls in trace-based MoE routing studies (as of 27 Sept 2026)

Method note: arXiv PDFs were downloaded and read as text (pdftotext) for 2608.07911 (v4), 2608.18261 (v1), 2608.12103 (v2), 2609.12978 (v1), 2412.07067 (MoE-CAP v6), 2508.17467, 2508.10251, 2605.19537, 2604.17182, 2604.00362, 2508.17137 and 2509.07379. The arXiv abstract pages could not be read through WebFetch in this session, so all quotes come from the PDF text. GitHub issue pages were read through WebFetch summaries. Those summaries did not show issue comments, so maintainers' explanations are **not** available for the gpt-oss perplexity issues. Items marked "per companion notes" come from `research_notes/MoE hybrid decode novelty check/*.md`, written earlier in this project, which cite primary URLs.

## (A1) Existing MoE benchmarks and audits: what 2608.07911 and 2608.18261 did and did not do, and whether anyone has re-derived published MoE-offloading speedups against a common model or hardware bound

### Takeaway
Direction (A) is **partly taken**. 2608.07911 (Yu Zhang, v1 8 Aug 2026, v4 25 Aug 2026) already argues publicly that there is "no common evaluation contract". It audits 10 papers and publishes a 4-group reporting checklist with a fill-in "minimal reporting block". Its scope is explicitly limited to **trace-driven cache-policy evaluation in block counts**:
- no hardware validation;
- no re-derivation of any published speedup;
- it states that its audit "does not support re-labelling published speedups as artifacts".

2608.18261 is not a benchmarking audit. It is a pre-registered negative result about training routers for locality, plus one measured configuration matched by a bytes/bandwidth model. MoE-CAP (NeurIPS 2025) and MoE-Inference-Bench (SC'25 workshops) benchmark systems that the authors run themselves. Neither reconciles other papers' claims. I found **no paper that re-derives published MoE-offloading or hybrid speedups with one common analytical model, normalizes them to a hardware speed-of-light bound, or systematically audits baseline strength (e.g. llama.cpp `-ngl` versus `--n-cpu-moe` at equal VRAM)**. That end-to-end, time-domain, hardware-validated slice is still open.

### Cited Findings

**2608.07911, "Reproducible Evaluation of MoE Expert Caching: Replay Semantics, Workload Contamination, and Operating Regimes" (Yu Zhang, China National Chemical Equipment Co. Ltd.; single author)**
- Dates: the PDF is dated 2026-08-04, v1 was submitted 8 Aug 2026, and v4 is dated 25 Aug 2026 (four versions in 17 days). — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)
- Method: "a trace-driven, event-atomic simulator over three MoE models (40, 64 and 128 experts)". The models are Granite-3.1-3B-A800M, OLMoE-1B-7B and Qwen3-30B-A3B (MLX 4-bit), per companion notes. — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)
- It identifies three evaluation axes that "change conclusions rather than shift numbers":
  - (i) **Replay semantics**: per-access replay of a fused event "inflates recency-based policies by 27–29%", which inverts the policy ranking.
  - (ii) **Workload contamination**: single-template probe sets produce verbatim-identical generation prefixes. A matched-pair intervention "moves the measured early-window effect by 19.4–31.9 percentage points and reverses which workloads appear most cache-friendly".
  - (iii) **Operating regimes**: "normalized miss fractions do not transfer across models, so the per-step expert union relative to per-layer capacity must be reported".
  - Source: [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)
- Headline result: after corrections, the gap to the offline optimum is 44.2–45.9%. Of that gap, 84.3–96.6% is attributed to future-victim knowledge. A causal next-use predictor recovers −11.4% of it. — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)
- The audit (§12.3, Appendix A, audited 2026-08-02) covers ten papers: SpecMD v1, DALI v1, Fate v2, Pre-gated MoE v3, SiDA-MoE v2, MoE-Beyond v1, Mixtral-Offloading v1, HOBBIT v2, MoE-Infinity v3 and SP-MoE v2. By type, that is "seven end-to-end offloading systems, one trace-driven cache simulation, and two architecture/prediction studies". — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)
- Audit findings:
  - "none of the ten directly reports the measured per-step/per-layer expert union divided by usable per-layer capacity".
  - "the number of instruction templates per category, chat-template application and positional synchronization are generally not reported".
  - Conclusion: "the public reports do not expose a common evaluation contract sufficient to compare cache-policy numbers across systems and models".
  - Source: [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)
- What it explicitly does **not** claim: "This audit does not show that any cited speedup or quality result is wrong". Appendix A.3 adds: "It does not support re-labelling published speedups as artifacts, assigning a direction to an unreported implementation choice…". — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)
- Its own scope limit (§10.6): the checklist "says nothing about … real host-to-device transfer and its overlap with compute, interconnect topology and achievable sustained bandwidth, kernel time, memory contention … A study that satisfies every item here has established comparability with other trace-driven studies, not deployability." — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)
- The limitation it most wants removed: "real-hardware validation: every result here is a block count". Checklist item D1 asks for "absolute transferred bytes per output token … together with the bandwidth budget implied by the target service rate". In its self-audit (Appendix B.2), it says it cannot fill that field without "the hardware calibration §11 says we lack". — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)
- Pre-registration: "Every threshold presented as a confirmatory decision criterion was pre-registered and frozen before the [run]". It releases decision records, including failed criteria. — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)
- Release status: it promises to release, "on publication", "the simulator, a diversity-controlled probe set, the contamination diagnostics and a reporting checklist". — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)
- Trace collection: "Decode routing was collected autoregressively with the model's chat template applied and reasoning mode disabled, greedy decoding … up to 384 recorded decode forwards per request". The Qwen3 traces were collected on "a Mac mini with an Apple M4 Pro" under MLX 0.32.0. — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)

**2608.18261, "Cacheable by Design? Training Mixture-of-Experts Routers for Locality Against the Edge Memory-Bandwidth Wall: A Pre-Registered Negative Result, with a Systems Measurement Study" (Shriniwas Ramesh Suram, University of the Cumberlands; single author; 18 Aug 2026, v1)**
- Systems part: one configuration, "a single RTX 3070", serving Qwen3-235B-A22B Q4_K_M. "measured decode is 0.44 tok/s warm, in exact agreement with a bytes-per-token ÷ bandwidth model". A request-batching scheme "collapses at batch 32 due to paging thrash". — [arXiv 2608.18261](https://arxiv.org/abs/2608.18261)
- Its trace tool "llama-moe-trace" is "a ∼120-line addition to llama.cpp's eval-callback". It warns that a flat copy of the non-contiguous top-k view "produced perfectly uniform garbage". — [arXiv 2608.18261](https://arxiv.org/abs/2608.18261)
- Routing results on Qwen3-30B-A3B, traced "over 8,000 tokens each of prose, code, math, and medical text":
  - adjacent-token reuse is 2.0× chance;
  - 95% of traffic goes through 52.5% of experts;
  - an LRU cache holding 13.4% of experts serves 66% of requests.
  - Source: [arXiv 2608.18261](https://arxiv.org/abs/2608.18261)
- The pre-registration covers the **training** experiment only: 137M-parameter MoE LMs with locality and domain router losses, judged by joint criteria (miss reduction plus a ≤1% perplexity gate). "every configuration fails the pre-registered ≤ 1% perplexity gate". — [arXiv 2608.18261](https://arxiv.org/abs/2608.18261)
- It does **not** audit or critique prior offloading papers' baselines or speedups. — [arXiv 2608.18261](https://arxiv.org/abs/2608.18261)

**2608.12103, "Who Should Own the Expert Cache? Kernel-Managed Tiering for Trillion-Parameter MoE Inference" (Si, Lin, Li, Zhang; Waterloo and independent researchers; v2 30 Aug 2026)**
- It evaluates the OS page cache as the expert tier, using router traces from three models (128–896 experts per layer). A production trillion-parameter model is replayed natively on GH200. It is a systems study, not a benchmarking audit. — [arXiv 2608.12103](https://arxiv.org/abs/2608.12103)

**SeqMoE, 2609.12978 (USTC and collaborators; 11 Sep 2026)**
- A predictive, CUDA-graph-compatible offloading system. It labels its llama.cpp baseline "Static Offload", reported at 30.9 tok/s against SeqMoE's 104.1 tok/s on an RTX 4090 at 45% residency (per companion notes). — [arXiv 2609.12978](https://arxiv.org/abs/2609.12978)

**MoE-CAP (Jiang, Fu, Huang et al.; Edinburgh, Microsoft Research, PKU, NetMind.AI, NVIDIA)**
- arXiv 2412.07067 (first posted Dec 2024; v6 dated 19 Nov 2025). Presented as a NeurIPS 2025 poster (San Diego). I did not verify which track it appeared in. A second arXiv id, 2505.11415, carries the same title. — [arXiv 2412.07067](https://arxiv.org/abs/2412.07067); [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/loc/san-diego/poster/121496); [arXiv 2505.11415](https://arxiv.org/abs/2505.11415); [GitHub Auto-CAP/MoE-CAP](https://github.com/Auto-CAP/MoE-CAP)
- Contributions:
  - the "MoE-CAP trade-off": systems "typically optimize two of the three dimensions [cost, accuracy, performance] at the expense of the third";
  - a CAP radar diagram;
  - sparsity-aware metrics S-MBU and S-MFU.
  - Source: [arXiv 2412.07067](https://arxiv.org/abs/2412.07067)
- S-MBU is defined as B_achieved/B_peak, with B_achieved = (S_activated + S_KV)/TPOT, and S_activated counts only experts whose indicator 1[l,i] is set "by tracing router outputs". — [arXiv 2412.07067](https://arxiv.org/abs/2412.07067)
- Its expert-activation profiler has probes in SGLang and HF Transformers. "The model runs on representative data until the activation distribution stabilizes" and the resulting "activation sheets" are stored for reuse. — [arXiv 2412.07067](https://arxiv.org/abs/2412.07067)
- It supports running vLLM, MoE-Infinity, SGLang, K-Transformers and HF itself. Cost equations include CPU, PCIe, DRAM and SSD terms. — [arXiv 2412.07067](https://arxiv.org/abs/2412.07067)

**MoE-Inference-Bench (Chitty-Venkata et al.; Argonne, Cerebras, PNNL)**
- arXiv 2508.17467 (24 Aug 2025), published in the SC '25 Workshops. — [ACM DL](https://dl.acm.org/doi/10.1145/3731599.3767706); [arXiv 2508.17467](https://arxiv.org/abs/2508.17467)
- Scope: H100 only, covering batch size, sequence length, FFN dimension and expert count, pruning, fused MoE, speculative decoding, quantization and parallelism. The PDF text contains **zero** occurrences of "offload". — [arXiv 2508.17467](https://arxiv.org/abs/2508.17467)

**General LLM-inference benchmarking methodology**
- LLM-Inference-Bench (same Argonne group), SC '24 Workshops. It benchmarks LLM inference across AI accelerators. — [ACM DL](https://dl.acm.org/doi/abs/10.1109/SCW63240.2024.00178); [arXiv 2411.00136](https://arxiv.org/abs/2411.00136)
- FMwork, "Meta-Metrics and Best Practices for System-Level Inference Performance Benchmarking" (IBM Research; 14 Aug 2025). It is about reducing the cost of benchmarking sweeps, reporting "up to 24x improvement" against a full sweep. It is not about reconciling published claims. — [arXiv 2508.10251](https://arxiv.org/abs/2508.10251)
- "The Silent Hyperparameter" (CISPA; v2 20 May 2026):
  - surveys 200 inference engines and 35,000 ML publications and finds "the specific inference stack is rarely reported";
  - finds that backend choice alone "can shift benchmark scores by up to 16.6 percentage points";
  - "advocate[s] standardized reporting of inference stacks".
  - The target is accuracy and reproducibility, not speed.
  - Source: [arXiv 2605.19537](https://arxiv.org/abs/2605.19537)
- MLPerf Client v2.0 (page modified 18 Aug 2026) covers Llama 3.1 8B Instruct, Phi 4 Mini, Phi 4 Reasoning 14B, Qwen 3 8B (experimental) and Flux 2 Klein 4B. Metrics are TTFT, TPS and end-to-end duration. **No MoE models**, and no treatment of CPU/GPU hybrid execution or expert offload. Only runs with the approved "configuration tested" setup are valid scores. — [MLCommons MLPerf Client](https://mlcommons.org/benchmarks/client/)
- The LLMPerf leaderboard (Anyscale/Ray) exists for API-endpoint benchmarking. I did not examine its methodology in this session. — [ray-project/llmperf-leaderboard](https://github.com/ray-project/llmperf-leaderboard)

**Evidence that baselines in the offloading literature are weak or incomparable (per companion notes, primary sources linked)**
- KTransformers (SOSP'25): "Since Llama.cpp originally only supports layer-wise offloading and lacks expert-level offloading capability … we extended Llama.cpp with custom code". Its llama.cpp baseline ran in FP16 because llama.cpp lacked BF16 CUDA kernels. — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- The companion survey of about 35 systems found:
  - "almost every 'llama.cpp' baseline is llama.cpp in static layer-wise mode (`-ngl`), or llama.cpp patched with custom expert offload";
  - "Among the system papers checked, none configures the `--n-cpu-moe` / `-ot exps=CPU` expert-only offload at equal VRAM".
  - Source: [companion notes, academic_systems.md]; primary examples include [SeqMoE](https://arxiv.org/abs/2609.12978) and [2608.18261](https://arxiv.org/abs/2608.18261).
- The NVIDIA MLSys 2026 oral "Efficient, VRAM-Constrained xLM Inference on Clients" (Pipelined Sharding) is built on llama.cpp and uses llama.cpp's `-cmoe` as its baseline, reporting an average 3.7× decode TPS gain (per companion notes, venue_bar.md). This is the one strong-baseline precedent found.

### Inferences
- The "no common evaluation contract" framing is **not available as a headline**: 2608.07911 owns it, dated 8 Aug 2026, with a checklist. Any (A) paper must cite it and differentiate by scope. 2608.07911 covers trace-level replay and workload construction in block counts. The researcher's assets cover **time-domain, hardware-validated, end-to-end claims**:
  - a bytes/bandwidth model with 16% median leave-one-source-out error on 52 third-party measurements;
  - a speed-of-light bound;
  - an equal-VRAM protocol;
  - A10 measurements with a pre-registered step-time model.
  This is exactly the gap 2608.07911 names as its top limitation ("real-hardware validation") and its unfilled D1 field (bytes per token plus bandwidth budget).
- 2608.07911 explicitly declines to say published speedups are wrong. A paper that **does** quantify how much of each published speedup is explained by (i) baseline configuration (e.g. `-ngl` versus `--n-cpu-moe` or equal VRAM), (ii) hardware bandwidth ratios and (iii) precision differences would be a clear next step, not a duplicate. This is the defensible novelty for (A). It requires care: accusing specific papers of weak baselines needs reproducible evidence, ideally re-running at least some baselines, and a neutral tone.
- MoE-CAP's S-MBU uses a single B_peak. For hybrid execution, where expert bytes come from HBM, PCIe and CPU DRAM at different rates, a single peak is ill-defined. A tiered speed-of-light (per-tier bytes over per-tier bandwidth, with the overlap structure) generalizes S-MBU. This is a concrete, citable differentiator. It is my inference from the equation as printed. MoE-CAP's appendix on multi-tier cases was not read in full.
- The two August 2026 pre-registered arXiv papers show that pre-registration in this niche is now "done" by others. The researcher's pre-registered A10 model is still good practice, but it is no longer a novelty claim.

### Gaps
- I did not read MoE-CAP's appendix in full to confirm how S-MBU handles offloaded or CPU-executed experts, or which offloading configurations its leaderboard contains.
- The companion notes say 2608.12103 was first posted 12 Aug 2026. In this session I saw only v2 (30 Aug 2026).
- Vidur (MLSys 2024 simulator), "LLMPerf" methodology and "inference benchmarking pitfalls" papers were not examined in this session. Only LLM-Inference-Bench, FMwork, Silent Hyperparameter and MLPerf Client were checked.
- I found no reproducibility study specifically of MoE offloading (e.g. re-running MoE-Infinity or Fiddler). This is based on search absence, not an exhaustive scan.

## (A2) Is there a published "evaluation contract" or reporting standard for offloaded/hybrid LLM inference, or any calls for one?

### Takeaway
One such standard exists, and it is narrow: 2608.07911 §10 plus its Appendix B "minimal reporting block" (Aug 2026). It covers trace-driven MoE cache evaluation only and explicitly excludes transfer, overlap, topology, bandwidth and kernel time. General calls for reporting inference stacks exist for accuracy reproducibility (Silent Hyperparameter, May 2026). MLPerf Client has run rules but no MoE or offload coverage. I found **no** reporting standard for measured end-to-end offloaded/hybrid decode performance. Examples of what such a standard would require: hardware bandwidths per tier, VRAM budget, baseline flags, precision, prompt and generation lengths, warm/cold state, and the bound achieved.

### Cited Findings
- 2608.07911 checklist groups:
  - **A**, Replay and execution model (A1 execution unit, A2 retention at the event boundary, A3 intra-event order, A4 conformance trace);
  - **B**, Workload construction (B1 templates per category; B2 "State whether the model's chat template was applied"; B3 decode steps recorded; B4 position-locked cohorts; B5 positional token agreement; B6 overlap on low-text-overlap pairs; B7 position-band stability; B8 matched-pair control);
  - **C**, Regime and comparability (C1 union/capacity r(s,l); C2 cache scope; C3 miss denominator; C4 dedup credit; C5 service discipline; C6 static-baseline fit source; C7 align on r̄);
  - **D**, Oracle bounds (D1 absolute bytes per output token and bandwidth budget; D2 forced-admission decomposition; D3 no best-of-N per-step gains; D4 scope of negative results).
  - Source: [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)
- 2608.07911 on intent: "The intent is comparability rather than compliance. Most items ask an author to state a choice, not to make a particular one." — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)
- Its evidence that service discipline matters: "Changing only this, at fixed request set, capacity and mean batch size, moved a static pinned set from 7.5% worse than LFRU to 17.0% better." — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)
- Silent Hyperparameter advocates "standardized reporting of inference stacks to improve the reproducibility and interpretability of benchmark comparisons". This concerns accuracy, not speed. — [arXiv 2605.19537](https://arxiv.org/abs/2605.19537)
- ReproEvalCard, "A Reporting Standard for Reproducible Evaluation of LLM Pipelines", is in ACL 2026 (short papers). I saw only its title in search results. It concerns evaluation pipelines, not performance. — [ACL Anthology 2026.acl-short.22](https://aclanthology.org/2026.acl-short.22/)
- MLPerf Client requires approved configurations for valid scores but has no MoE models and no offload rules. — [MLCommons MLPerf Client](https://mlcommons.org/benchmarks/client/)
- Hoefler and Belli's 12 rules (SC'15) include showing upper performance bounds and statistically sound comparisons. SPCL applies these as its methodology bar (per companion notes, venue_bar.md). — [SPCL Publications](https://spcl.inf.ethz.ch/Publications/)

### Inferences
- The unclaimed piece is an **end-to-end performance contract for offloaded/hybrid decode**. It would plausibly include: per-tier bandwidths (HBM, PCIe H2D, CPU DRAM), VRAM budget and what occupies it, baseline engine and exact flags (`-ngl`, `-ot`, `--n-cpu-moe`), weight precision per tier, batch size, prompt and generation lengths, warm/cold state, trace provenance (on-policy or not; see B), and **achieved fraction of a speed-of-light bound**. This fits the 12-rules ethos ("show upper performance bounds") and complements 2608.07911's trace-level checklist rather than competing with it.
- Positioning suggestion: frame (A) as "Scientific benchmarking of MoE offloading: from block counts to seconds". Cite 2608.07911 as the trace-level contract, and contribute the time-level contract with a validated model and bound, plus a retrospective normalization of published claims.

### Gaps
- I did not search SC/PMBS workshop proceedings 2024–2026 for inference-benchmarking position papers. There may be HPC-community calls for LLM performance reporting standards that I missed.
- The ReproEvalCard content was not read.

## (B) Off-policy teacher forcing in trace-based MoE routing studies: is the bias documented or quantified? What about gpt-oss harmony format and perplexity?

### Takeaway
It is documented only **adjacently**, and the specific claim is not quantified in anything I found.
- 2608.07911 documents and quantifies a closely related trace-provenance confound: generation under **raw continuation (no chat template)** yields shared, surface-repeated prefixes. This moves an early-window locality effect by 19.4–31.9 pp and reverses which workloads look cache-friendly. It adds checklist item B2 ("State whether the model's chat template was applied").
- SeqMoE (Sept 2026) already uses the "correct" on-policy procedure: generate token IDs, then replay them as a teacher-forced prefill to dump routing. It does not quantify how conclusions differ from traces on dataset text.
- Several studies trace dataset text directly. 2608.18261 traces "8,000 tokens each of prose, code, math, and medical text"; the wording implies teacher-forced corpus text, but this is not explicit. MoE-CAP profiles on "representative data".

I found **no paper that quantifies how teacher-forced off-policy dataset text versus on-policy generated text changes expert locality, cache hit rates, Belady gaps or offload speed predictions**. That specific measurement looks open, but it is a narrow extension of 2608.07911's workload-contamination axis, and reviewers will see it that way.

For gpt-oss: very high raw-text perplexity for **gpt-oss-20b** is community-known, and OpenAI states that harmony format is required. No public report was found of **gpt-oss-120b** being far worse than 20b on off-format text (the researcher's 7.3 versus 3.3 nats), or of it collapsing to near-uniform loss. That observation is undocumented. It is also a risk: it could be an implementation or conversion artifact.

### Cited Findings

**Trace-provenance practice in 2025–2026 papers**
- 2608.07911 on the mechanism: "Under raw continuation — that is, when the model's chat template is not applied — the model completes that sentence verbatim before producing anything task-specific, and then emits a fixed reasoning preamble." — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)
- 2608.07911 on the effect size: "In our original collection the first 15–25 of 63 recorded steps were near-identical across requests". "4 of 16 requests produced byte-identical 63-token outputs; positional token agreement within that archetype was 20.65%, against 1.04%" for heterogeneous documents. They "withdrew the original conclusion". — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)
- 2608.07911 on its own limitation: "All traces use greedy decoding, which is reproducible but does not sample production generation diversity." Its Qwen3 traces come from an MLX 4-bit snapshot. — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)
- SeqMoE's collection: "The first pass collects token ID traces through continuous batching. The second packs multiple token ID traces into a single prefill pass to dump expert activations". The trace pool is 50K traces from MATH, GSM8K, CodeForces, OpenOrca and ShareGPT (per companion notes). "Teacher forcing" appears in SeqMoE only for **predictor training**, with Scheduled Sampling added afterwards. — [arXiv 2609.12978](https://arxiv.org/abs/2609.12978)
- 2608.18261 traced Qwen3-30B-A3B with a llama.cpp eval-callback "over 8,000 tokens each of prose, code, math, and medical text". The paper does not say explicitly whether this is teacher-forced corpus text or generation. — [arXiv 2608.18261](https://arxiv.org/abs/2608.18261)
- MoE-Beyond (UPenn, arXiv 2508.17137) built "approximately 100 million expert activation trace points by inferencing 6994 prompts from the LDJnr-Puffin dataset using the DeepSeek-V2-Lite MoE model and recording detailed information for each generated [token]". It reports 97.5% predictor accuracy on unseen WebGLM-QA prompts. That makes it on-policy generation, with no off-policy comparison. — [arXiv 2508.17137](https://arxiv.org/abs/2508.17137)
- MoE-CAP's profiler: "The model runs on representative data until the activation distribution stabilizes". It does not say whether the data is generated or teacher-forced. — [arXiv 2412.07067](https://arxiv.org/abs/2412.07067)
- DuoServe-MoE (Sydney; v2 9 Apr 2026) documents a **prefill/decode phase disparity**: "the prefill phase tends to activate experts densely across many tokens, while the decode phase activates only a few experts per step". This is a phase effect, not a text-provenance effect. — [arXiv 2509.07379](https://arxiv.org/abs/2509.07379)

**Why token distribution matters (mechanism support)**
- Hayashi et al. (Nagoya; 19 Apr 2026) studied Qwen3.5-35B-A3B-FP8. "At positions where both sequences generated the same token, Jaccard similarity reaches 0.649 (40× random), while even at positions with different tokens it remains 0.175 (11× random)". Token identity dominates routing in the input layers, and context dominates in the middle layers. They cite OpenMoE's earlier claim that "expert routing is primarily determined by token identity rather than context". — [arXiv 2604.17182](https://arxiv.org/abs/2604.17182)

**gpt-oss format dependence and raw-text perplexity**
- OpenAI harmony guide (Dominik Kundel, 5 Aug 2025): "gpt-oss should not be used without using the harmony format, as it will not work correctly." It defines the channels `analysis` (chain of thought), `commentary` (tool calls and preambles) and `final` (user-facing), and the special tokens <|start|> (200006), <|end|> (200007), <|message|> (200008), <|return|> (200002) and <|call|> (200012). — [OpenAI Cookbook: harmony](https://developers.openai.com/cookbook/articles/openai-harmony)
- llama.cpp issue #15155, "gpt-oss-20b perplexity broken" (opened 7 Aug 2025 by GlasslessPizza):
  - llama-perplexity over 128 chunks gave PPL 195.72 ± 3.46 (mxfp4) and 196.28 ± 3.46 (Q8_0), i.e. about 5.28 nats per token (ln 195.7).
  - The issue is closed. The resolution and comments were not visible in the fetched summary.
  - Source: [llama.cpp #15155](https://github.com/ggml-org/llama.cpp/issues/15155)
- HF transformers issue #40990, "Extremely high perplexity on openai/gpt-oss-20b with WikiText-2 (raw)" (opened 19 Sep 2025 by kuantuna):
  - PPL ≈ 394.467 (≈ 5.98 nats) on the WikiText-2 raw test split, at context 2048, bf16, A100-40GB, transformers 4.56.1, 28,658 tokens.
  - The reporter asks whether harmony formatting is required for corpus perplexity.
  - The issue is closed. Comments and resolution were not visible in the fetched summary.
  - Source: [transformers #40990](https://github.com/huggingface/transformers/issues/40990)
- "In harmony with gpt-oss" (Mavrin; 1 Apr 2026):
  - OpenAI's gpt-oss-20b agentic scores were first reproduced independently only with "a native harmony agent harness … bypassing the lossy Chat Completions conversion": 60.4% on SWE Verified HIGH against a published 60.7%.
  - For gpt-oss-120b it notes "62.4% published versus 26% on the leaderboard" under a different harness.
  - This is evidence that gpt-oss is unusually format- and harness-sensitive, though for task accuracy rather than perplexity.
  - Source: [arXiv 2604.00362](https://arxiv.org/abs/2604.00362)
- The "Silent Hyperparameter" study found backend-induced divergence driven by "prefix caching and CUDA graphs, custom kernels, and engine-specific defaults in logit processing". This is relevant because the routing traces behind cache studies are collected on varied backends and quantizations (e.g. MLX 4-bit in 2608.07911). — [arXiv 2605.19537](https://arxiv.org/abs/2605.19537)
- Searches for reports of gpt-oss-120b raw-text perplexity being higher than 20b's, or of 120b degenerating to near-flat loss on off-format text, returned nothing relevant. Result pages were about task benchmarks, e.g. the HF discussion "Why is the performance worse on the 20B than on the 120B", which concerns task scores. — [HF gpt-oss-120b discussion #84](https://huggingface.co/openai/gpt-oss-120b/discussions/84)

### Inferences
- **Direction (B) is open in its specific form**: "teacher-forced dataset-text traces vs on-policy (self-generated, correctly formatted) traces change cache and offload conclusions, by X". It is **not** open as a general claim that trace provenance matters, because 2608.07911 already establishes that workload construction and chat-template choices reverse conclusions. A (B) paper should present itself as adding a new provenance axis (policy mismatch and teacher forcing) to 2608.07911's checklist, with numbers on metrics like per-layer union/capacity, LRU and Belady hit rates, reuse distance, and predicted tok/s from the bytes/bandwidth model.
- The gpt-oss numbers give (B) a vivid headline, but they need a robustness check before publication. The researcher's 20b figure (3.3 nats ≈ PPL 27 on their text) is much lower than community WikiText numbers (5.3–6.0 nats). So "their text" differs from WikiText, or the setup differs. The 120b figure (7.3 nats ≈ PPL 1,480, against 0.65 nats ≈ PPL 1.9 on its own samples) being **worse than 20b** is not documented anywhere I found.
  - Agreement between HF fp32 and llama.cpp lowers the chance of an implementation bug. It does not rule out shared upstream causes: the same converted weights, the same missing BOS or system/harmony prefix, or a sliding-window, attention-sink or context-length setting.
  - A third reference would strengthen it: OpenAI's reference implementation, or vLLM `prompt_logprobs`.
  - So would a check that 120b's loss falls to normal levels once the same text is wrapped in harmony (e.g. as a `final`-channel assistant message).
- Mechanism plausibility: routing is strongly token-identity-driven (2604.17182). If off-policy text has a different token distribution from what the model would generate, especially for a reasoning model whose natural output lives in `analysis` and `final` channels, expert popularity and reuse statistics measured on dataset text can differ systematically from deployment decode. A model that finds the text near-unpredictable (120b at 7.3 nats) is plausibly in an out-of-distribution hidden-state regime, where routing may be atypical. That is a testable hypothesis, not an established fact.
- Most MoE cache/offload systems papers collect decode traces by generation (MoE-Beyond, 2608.07911, SeqMoE, 2608.12103). The practical impact of (B) therefore falls mainly on (i) corpus-teacher-forced characterization studies (possibly 2608.18261 and MoE-CAP-style profilers), (ii) cheap GPU-free tracing pipelines like the researcher's own, and (iii) predictor-training datasets. The paper must be honest that the bias affects a subset of the literature, and must show which conclusions actually flip.

### Gaps
- The comment threads of llama.cpp #15155 and transformers #40990 could not be read (GitHub API access blocked; the WebFetch summary showed no comments). Whether maintainers attributed the 20b perplexity to harmony format, a bug, or something else is **unknown**.
- I did not check whether the gpt-oss model card or paper reports any raw-text perplexity.
- No quantitative study was found comparing routing statistics on teacher-forced corpus text against self-generated text for any MoE model. This rests on search absence (about 8 queries), not an exhaustive survey. Workshop papers or blog posts may exist.
- It is unconfirmed whether 2608.18261's 8,000-token domain traces were teacher-forced or generated. The PDF wording does not say, and its released repo was not inspected.

## (C) What remains open, fit with the researcher's assets, feasibility by 27 Oct 2026 and by Dec 2026, and likely venues

### Takeaway
- **(A)** Viable only if repositioned. The trace-level evaluation-contract idea is taken (2608.07911, Aug 2026). A **time-domain, hardware-validated re-derivation of published MoE-offloading and hybrid speedups against a common bytes/bandwidth model and speed-of-light bound, plus a baseline-adequacy audit and an end-to-end reporting contract**, is open and matches the researcher's assets and SPCL's "scientific benchmarking" values closely. An arXiv version by 27 Oct 2026 is feasible if it reuses the existing 52-point validation. A full paper fits MLSys 2027 (due 30 Oct 2026) or, better, ISPASS 2027 (around mid-Dec 2026, inferred).
- **(B)** Open in its narrow form. It is a small, fast measurement paper (workshop or short paper) or a strong section inside (A). The gpt-oss-120b anomaly is either its hook or its liability, depending on the robustness checks.

### Cited Findings
- ETH AI Center Doctoral Fellowship 2027 applications close Tue 27 Oct 2026, 16:00 CET. Reference letters are due 2 Nov 2026 (per companion notes). — [ETH AI Center FAQ](https://ai.ethz.ch/research/phd-and-postdoc-programs/fellowship-faqs.html)
- MLSys 2027: paper deadline 30 Oct 2026 (12:00 PM PDT per the conference page); double-blind; 10 pages; arXiv allowed; notification 28 Feb 2027 (per companion notes). — [MLSys 2027 CFP](https://mlsys.org/Conferences/2027/CallForResearchPapers)
- ISPASS 2026 had its paper deadline on 15 Dec 2025, a 9-page limit, and a "tool and benchmark papers" option. ISPASS 2027 is inferred to be around mid-Dec 2026 (per companion notes). — [ISPASS 2026 submission](https://ispass.org/ispass2026/submission.php)
- EuroMLSys 2026: 6 pages, archival (ACM DL), deadline 24 Feb 2026. EuroMLSys 2027 dates were not posted (per companion notes). — [EuroMLSys](https://euromlsys.eu/)
- SPCL publishes characterization work at IISWC, e.g. "Confidential LLM Inference: Performance and Cost Across CPU and GPU TEEs" (IISWC 2025) (per companion notes). — [SPCL IISWC'25 PDF](https://spcl.inf.ethz.ch/Publications/.pdf/IISWC25_confidential_llms.pdf)
- MoE-Inference-Bench and LLM-Inference-Bench show that SC workshops accept LLM and MoE inference-benchmarking papers. — [MoE-Inference-Bench, SC'25 Workshops](https://dl.acm.org/doi/10.1145/3731599.3767706); [LLM-Inference-Bench, SC'24 Workshops](https://dl.acm.org/doi/abs/10.1109/SCW63240.2024.00178)
- MoE-CAP appeared at NeurIPS 2025 (poster), which shows that ML venues take MoE-system benchmark papers. — [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/loc/san-diego/poster/121496)

### Inferences
**(A) What remains open, ranked by defensibility**
1. A retrospective, **hardware-normalized** table of published MoE-offloading and hybrid batch-1 decode results. Each claim gets: its reported tok/s, the bytes/bandwidth prediction, the fraction of the speed-of-light bound, and the speedup re-expressed against a strong equal-VRAM baseline (e.g. llama.cpp `--n-cpu-moe`). The 52-measurement leave-one-source-out validation (16% median error) is the backbone. Nobody has done this. 2608.07911 explicitly stops short of it.
2. A **baseline-adequacy audit**: which engine, which flags, and whether VRAM is equal. Companion notes show that "llama.cpp" baselines are usually `-ngl` layer offload or custom patches, and rarely `--n-cpu-moe` at equal VRAM.
3. An **end-to-end reporting contract**: the time-domain counterpart to 2608.07911's block-count checklist, including achieved fraction of the bound (Hoefler's rule on upper bounds).
4. A generalization of S-MBU to tiered memory, from which the bound follows.

Risks: tone (it must not read as a hit piece); needing to re-run at least a few published systems or baselines on the A10 to show the normalization predicts real gaps; and the fact that 2608.07911 and the companion "speed-of-light" work can make reviewers ask "what is new beyond a table". The answer has to be quantitative: for example, "X of Y published speedups shrink below Z× once normalized to equal VRAM and bandwidth".

**(B) What remains open**
- A controlled measurement on OLMoE-1B-7B, Qwen3-30B-A3B and gpt-oss-20b/120b. For the same prompts, compare traces from (i) teacher-forced dataset continuations, (ii) on-policy greedy and sampled generations, and (iii) on-policy text replayed through the GPU-free layer-streamed prefill; (iii) should equal (ii) exactly, which validates the tracer.
- Metrics: per-layer union/capacity r̄, LRU, LFU and Belady hit rates at fixed residency, reuse-distance distributions, and predicted tok/s through the bytes/bandwidth model.
- Report which cache and offload **conclusions flip** (policy rankings, static-pin versus LRU, predicted speedups), not just distribution distances. This mirrors 2608.07911's "change conclusions rather than shift numbers" standard, which reviewers will expect.
- gpt-oss-specific arm: raw text versus harmony-wrapped text versus self-generated `analysis` and `final` channel text.

**Feasibility by 27 Oct 2026 (about 4 weeks)**
- (B) is the most feasible new result. Traces and the tracer already exist. On-policy generation for OLMoE, Qwen3-30B-A3B and gpt-oss-20b on CPU via llama.cpp at modest token counts (tens of thousands of tokens per model) is plausibly hours per model. This is my estimate, not measured. The 120b arm is the costly one.
- (A) as an arXiv preprint is feasible only if the 52-point dataset, the bound and the A10 data are already paper-ready, and the work is mainly writing plus the baseline audit table.
- A combined preprint could work: (A) as the main contribution with (B) as a "trace provenance" section that extends 2608.07911's checklist. That would be one coherent SPCL-style "scientific benchmarking" story.

**Feasibility by Dec 2026**
- Both are realistic. (A) could be submitted to MLSys 2027 (30 Oct) or ISPASS 2027 (around mid-Dec, inferred; its tool/benchmark option suits it).
- (B) as a stand-alone short paper suits EuroMLSys 2027 (around Feb 2027, 6 pages, inferred) or an SC'27 or NeurIPS 2027 workshop.
- NeurIPS 2027 Datasets & Benchmarks (around May 2027, inferred from typical timing and not verified) would suit a released-trace-dataset framing: exact on- and off-policy routing traces for three or four models.
- IISWC 2027 (spring 2027, not verified) suits a characterization framing, and SPCL publishes there.

**Fit with SPCL**
- (A) aligns directly with Hoefler's 12 rules: bounds, sound comparisons, reproducibility. (B) aligns with the "measurement validity" ethos. Both depend on data movement and performance modeling, which are SPCL's core.
- The strongest application narrative: "published MoE-offloading claims are not comparable; here is a validated model, a bound, and a contract; here is a trace-validity pitfall that silently biases a class of studies."

### Gaps
- ISPASS 2027, EuroMLSys 2027, IISWC 2027, NeurIPS 2027 D&B and SC'27 workshop deadlines are not posted or were not verified. The timings above are inferred from earlier editions.
- It is unknown whether MLSys 2027 reviewers value meta-science or benchmarking contributions in the research track. MLSys has no dedicated benchmarks track that I verified.
- No measurement exists yet of how large the off-policy effect is on cache conclusions for the researcher's models. If it is small, (B) becomes a negative-result note, which is still reportable but weaker.
