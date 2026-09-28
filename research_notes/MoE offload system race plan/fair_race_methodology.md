# Fair, credible head-to-head methodology for batch-1 MoE decode with CPU offloading (as of 28 Sep 2026)

Method note. This extends three earlier notes and does not repeat them: `research_notes/MoE research direction rethink/benchmarking_and_offpolicy.md`, `research_notes/MoE audit measurement plan/audit_requirements.md` and `research_notes/MoE audit measurement plan/system_reproducibility.md`. Facts marked "(prior notes)" come from those files or from `research_notes/MoE hybrid decode novelty check/venue_bar.md` and `research_notes/MoE research direction rethink/numerics_and_prediction.md`, which cite primary URLs. Repo-relative links resolve against `/home/claude/moe-speed-of-light`.

How sources were read:
- Paper and web pages were read with WebFetch. It returns a summary from a small model, so quoted wording is "as summarised" unless stated otherwise.
- llama.cpp claims come from my own read of the source at commit `4d86b2f5` (master, 27 Sep 2026), cloned into the scratchpad.
- Fetches that failed and could not be recovered: Kalibera & Jones PDF (HTTP 504; a secondary summary was used), the SIGPLAN checklist PDF (binary), the vLLM FAQ (navigation only), the Bailey 1991 text (redirect loop), and the detailed SeqMoE §9 text.

Everything under **Inferences** is my recommendation. It is not a published rule.

## Q1. What published rules govern a fair batch-1 comparison, and what must be held equal? (GPU memory accounting, quantization and format across GGUF and safetensors, CPU threads and NUMA, prompts, output length, warm-up, repetitions, confidence intervals, means)

### Takeaway
No published rule set covers equal-VRAM hybrid CPU/GPU MoE decode. The race contract has to be assembled from four sources:
1. **Hoefler & Belli's 12 rules** for statistics and reporting:
   - harmonic mean for rates;
   - no averaged ratios;
   - nonparametric confidence intervals;
   - discard the first iteration;
   - document every factor;
   - show bounds.
2. **MLPerf Inference and Client** for run discipline:
   - fixed output length with no truncation;
   - a warm-up run followed by measured runs;
   - an accuracy gate relative to a reference;
   - "results that cannot be replicated are not valid".
3. **Agrawal et al. 2025's anti-patterns**: tune the baselines, and do not confuse engineering effort with algorithmic gains.
4. **Precedent from the papers themselves**:
   - Pipelined Sharding: an iterative search that fits each system to the same measured VRAM budget.
   - FreeToken: bit-aligned BF16/MXFP4 weights across engines, and 6 threads pinned to the GPU's NUMA node.

Two tool defaults on current stacks silently break "equal GPU memory" unless overridden:
- llama.cpp master now auto-fits to free VRAM by default (`--fit on`, 1 GiB margin).
- SGLang, which is the base of current KTransformers, pre-allocates most of the GPU for weights plus the KV pool by default.

### Cited Findings
**Hoefler & Belli, "Scientific Benchmarking of Parallel Computing Systems" (SC'15)**
- The twelve rules, verbatim:
  - R1: "When publishing parallel speedup, report if the base case is a single parallel process or best serial execution, as well as the absolute execution performance of the base case."
  - R2: "Specify the reason for only reporting subsets of standard benchmarks or applications or not using all system resources."
  - R3: "Use the arithmetic mean only for summarizing costs. Use the harmonic mean for summarizing rates."
  - R4: "Avoid summarizing ratios; summarize the costs or rates that the ratios base on instead. Only if these are not available use the geometric mean for summarizing ratios."
  - R5: "Report if the measurement values are deterministic. For nondeterministic data, report confidence intervals of the measurement."
  - R6: "Do not assume normality of collected data (e.g., based on the number of samples) without diagnostic checking."
  - R7: "Compare nondeterministic data in a statistically sound way, e.g., using non-overlapping confidence intervals or ANOVA."
  - R8: "Carefully investigate if measures of central tendency such as mean or median are useful to report. Some problems, such as worst-case latency, may require other percentiles."
  - R9: "Document all varying factors and their levels as well as the complete experimental setup (e.g., software, hardware, techniques) to facilitate reproducibility and provide interpretability."
  - R10: "For parallel time measurements, report all measurement, (optional) synchronization, and summarization techniques."
  - R11: "If possible, show upper performance bounds to facilitate interpretability of the measured results."
  - R12: "Plot as much information as needed to interpret the experimental results. Only connect measurements by lines if they indicate trends and the interpolation is valid."
  - Source: [Hoefler & Belli SC'15 PDF](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf); [ACM DL](https://dl.acm.org/doi/10.1145/2807591.2807644)
- On means (§3.1.1): "if the denominator has the primary semantic meaning, the harmonic mean provides correct results". "ratios should never be averaged as such an average is meaningless." — [SC'15 PDF](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf)
- On confidence intervals and sample size (§3.1.2–3.1.3):
  - For medians, "nonparametric CIs can be computed and have a similar interpretation".
  - "n > 5 measurements are needed to assess confidence intervals nonparametrically".
  - For non-normal data: "recomputing the 1−α CI after each ni = i·k … measurements and stop the measurement once the required interval is reached".
  - Source: [SC'15 PDF](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf)
- On warm-up and environment (§4.1):
  - "The first measurement iteration should be excluded from the average computation."
  - Cached data "may or may not be representative for the intended use".
  - "If controlling a certain parameter is not possible then we suggest randomization".
  - Source: [SC'15 PDF](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf)
- On comparisons (§3.2): the nonparametric Kruskal–Wallis test for non-normal data; effect size E = (X̄i − X̄j)/√igv; and "If 1−α confidence intervals do not overlap, then one can be 1−α confident that there is a statistically significant difference." — [SC'15 PDF](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf)
- The survey covered 120 HPDC/SC/PPoPP papers from 2011–2014, 95 of them applicable:
  - only 2 of 95 reported confidence intervals;
  - 15 of 39 speedup papers (38%) omitted base-case absolute performance;
  - only 26 of 95 reported RAM specifications.
  - The count of papers reporting any variation measure conflicts between sources: 17 of 95 in this session's summary versus "15 of 95" in the prior notes (venue_bar.md). Check Table 1 of the PDF before quoting it.
  - Source: [SC'15 PDF](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf)

**Multi-level repetition (Kalibera & Jones, ISMM 2013)**
- Repeat at every level where variance enters (builds, process executions, in-process iterations). Choose repetition counts per level from a dimensioning experiment. Give CIs for speed-up ratios by bootstrap or Fieller's method.
- "71 failed to provide any measure of variation" among surveyed papers.
- These points come from a secondary summary, because the primary PDF returned 504. — [secondary summary](https://mt-caret.github.io/papers/2019-06-12-rigorous-benchmarking.html); [primary, ACM DL](https://dl.acm.org/doi/10.1145/2464157.2464160); [Kent PDF](https://kar.kent.ac.uk/33611/45/p63-kaliber.pdf)

**MLPerf Inference rules (MLCommons `inference_rules.adoc`)**
- SingleStream (batch-1 analogue): metric is the "90%-ile early-stopping latency estimate"; minimum duration 600 seconds; the early-stopping appendix uses a binomial or incomplete-beta confidence bound. — [MLPerf Inference Rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
- Output length:
  - Per-model `min_new_tokens` and `max_new_tokens` are fixed (e.g., Mixtral-8x7B: 2 and 1024).
  - "Truncating output tokens to boost performance or meet accuracy is not permitted".
  - Tokens per sample must fall "between 90% and 110% of reference" for Mixtral and Llama3.1-405B.
  - Source: [MLPerf Inference Rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
- Quantization in the closed division: "arbitrary purely mathematical, reproducible quantization using only the calibration data and weight and bias tensors … to any numerical format that achieves the desired quality". Retraining and pruning are prohibited. The accuracy gate is 99% or 99.9% of the FP32/FP16 reference. — [MLPerf Inference Rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
- Run discipline:
  - "Results that cannot be replicated are not valid results" (§2.7).
  - "Caching of any other queries, query parameters, or intermediate results is prohibited" (§5.1).
  - Audits include two days of hardware access.
  - Source: [MLPerf Inference Rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)

**MLPerf Client (page modified 18 Aug 2026)**
- Each test runs four times: "one warm-up run and three performance runs", and the averages are reported.
- TPS is "the average rate to produce all of the rest of the tokens in the response, excluding the first one".
- Weights are int4, gated by MMLU thresholds (e.g., Llama 3.1 8B at 62%).
- "Only scores generated with this 'configuration tested' notice shall be considered valid". No MoE and no offload coverage.
- Source: [MLPerf Client](https://mlcommons.org/benchmarks/client/)

**LLM-inference evaluation anti-patterns (Agrawal, Kedia, Agarwal, Mohan, Kwatra, Kundu, Ramjee, Tumanov; arXiv 2507.09019, 11 Jul 2025)**
- **AP1, "Conflating Implementation and Algorithm."** Checklist: "Are baselines implemented comparably? If not, are microbenchmarks or ablations used to isolate algorithmic gains from system implementation differences?"
- **AP2, "Neglecting Parameter Tuning."** Checklist: "Were baseline parameters tuned appropriately for the evaluation setup?"
- **AP4, "Non-Representative Workloads."** Single-source datasets and short inputs.
- **AP7, "Reporting Only Summary Statistics."** Wants CDFs or P50/P90/P99.
- **AP8, "Obscuring Performance with Normalization."** TPOT = T_d/N_d "mask[s] … fixed or near-fixed overheads" and generation stalls, so TBT variance should be examined.
- Source: [arXiv 2507.09019 HTML](https://arxiv.org/html/2507.09019); [abs](https://arxiv.org/abs/2507.09019)

**GPU-memory accounting: what the tools actually do**
- **llama.cpp master (commit 4d86b2f5, 27 Sep 2026)**:
  - `-fit/--fit [on|off]` means "whether to adjust unset arguments to fit in device memory". Default `fit_params = true`, with `fit_params_target` of 1024 MiB margin per device and `fit_params_min_ctx = 4096`.
  - `-fitt/--fit-target` sets the margin.
  - `-cmoe/--cpu-moe` and `-ncmoe/--n-cpu-moe` remain available.
  - Source: [common/arg.cpp](https://github.com/ggml-org/llama.cpp/blob/4d86b2f5e8a04667337e20bd5051570be4554cfe/common/arg.cpp); [common/common.h](https://github.com/ggml-org/llama.cpp/blob/4d86b2f5e8a04667337e20bd5051570be4554cfe/common/common.h)
- **SGLang** `--mem-fraction-static` is "The fraction of the memory used for static allocation (model weights and KV cache memory pool)". When unset it is computed as "(GPU memory - reserved memory) / GPU memory", falling back to 0.88. SGLang also exposes `--enable-deterministic-inference`. The current KTransformers inference path runs on the kvcache-ai SGLang fork (prior notes). — [SGLang server arguments](https://docs.sglang.io/docs/advanced_features/server_arguments.md); [kt-kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)
- **FreeToken** exposes `--memory-ratio 0.9`, `--moe-cache-size/--moe-cache-rate/--moe-cache-auto` and `--moe-cpu-threads` (default: physical cores) (prior notes). — [FreeToken docs/cli.md](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md)
- **PyTorch's caching allocator**: "the unused memory managed by the allocator will still show as if used in `nvidia-smi`". It distinguishes `memory_allocated()` from `memory_reserved()` and offers a per-process memory-fraction cap that raises OOM when exceeded. — [PyTorch CUDA semantics (2.14)](https://docs.pytorch.org/docs/2.14/notes/cuda.html)
- **Pipelined Sharding (NVIDIA, MLSys'26)**: the budget was enforced empirically. "For our baselines, we performed this search process for each model-budget combination to find the maximal number of layers that fit within the given VRAM budget": "load the model with a candidate ngl value, run inference to trigger graph generation and allocation, measure actual VRAM usage, adjust ngl, and repeat until the budget is met." Baselines used "as many CPU threads as physical cores". Output was "100 tokens" at 1K–64K contexts. The paper does not list which components count toward the budget. — [arXiv 2604.26334 HTML](https://arxiv.org/html/2604.26334)

**Format and precision alignment across GGUF and safetensors**
- FreeToken (§5.1): "Weight formats are exactly aligned: every engine serves Qwen3.6 in BF16, and every engine consumes DSV4-Flash's native MXFP4 expert blocks bit-exactly." — [arXiv 2608.16157 HTML](https://arxiv.org/html/2608.16157)
- llama.cpp conversion:
  - `repack_mxfp4_blocks` is documented as "Lossless - only moves bits". It covers compressed-tensors MXFP4 "also used by DeepSeek-V4".
  - FP8 checkpoints are dequantized to BF16/F16 by default, or stored as Q8_0 with `--fp8-as-q8`.
  - `--outtype bf16` exists, and `auto` picks "the highest-fidelity 16-bit float type".
  - Source: [conversion/base.py](https://github.com/ggml-org/llama.cpp/blob/4d86b2f5e8a04667337e20bd5051570be4554cfe/conversion/base.py); [convert_hf_to_gguf.py](https://github.com/ggml-org/llama.cpp/blob/4d86b2f5e8a04667337e20bd5051570be4554cfe/convert_hf_to_gguf.py)
- KTransformers (SOSP'25): "Since Llama.cpp lacks BF16 CUDA kernels, we run FP16 models as a substitute". This is a precision mismatch inside a published head-to-head. — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- HybriMoE (DAC'25): "we leverage Marlin quantization, a state-of-the-art 4-bit quantization kernel" for its comparisons. — [arXiv 2504.05897 HTML](https://arxiv.org/html/2504.05897)

**CPU threads, NUMA and frequency: what published head-to-heads did**
- FreeToken: "every serving run and bandwidth measurement on [rented servers] is capped at 6 CPU threads and pinned to the GPU's NUMA node". It measured B_P (pinned H2D) and B_H (host expert-processing bandwidth) per machine. — [arXiv 2608.16157 HTML](https://arxiv.org/html/2608.16157)
- HybriMoE: "Intel Xeon Gold 5220R processor, restricting usage to 10 cores". — [arXiv 2504.05897 HTML](https://arxiv.org/html/2504.05897)
- KTransformers: a dual-socket Xeon 8452Y. Bandwidth was measured with Intel MLC at "220 GB/s" intra-socket and "125 GB/s" cross-socket. — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- 2512.16473: Threadripper 7960X. "CPU frequency varies between 4.8-5.3GHz based on active cores". Its throughput metric includes prefill. — [arXiv 2512.16473 HTML](https://arxiv.org/html/2512.16473)
- llama.cpp `--numa` offers `distribute`, `isolate` and `numactl`. The help text says "if run without this previously, it is recommended to drop the system page cache before using this". — [common/arg.cpp](https://github.com/ggml-org/llama.cpp/blob/4d86b2f5e8a04667337e20bd5051570be4554cfe/common/arg.cpp)
- Cloud-VM STREAM Triad reached only 46–59% of datasheet on the project's two anchor hosts (prior notes). — [paper/paper.tex](paper/paper.tex)

**llama-bench's own summarization (commit 4d86b2f5)**
- Defaults: `reps = 5`, warm-up on (`--no-warmup` to skip), `-d/--n-depth` default 0, plus `-C/--cpu-mask`, `--poll`, `--prio` and `--delay`.
- `avg_ts()` is the arithmetic mean of per-repetition rates, computed as `1e9 * n_tokens / t`. The JSON output also emits the raw `samples_ns`.
- Source: [tools/llama-bench/llama-bench.cpp](https://github.com/ggml-org/llama.cpp/blob/4d86b2f5e8a04667337e20bd5051570be4554cfe/tools/llama-bench/llama-bench.cpp)

### Inferences
**The race contract: what to hold equal, and how to measure it**

1. **Machine and software state (R9, R10)**
   - Run on one physical machine, in one boot session, with one driver and CUDA toolkit.
   - Record: `lscpu` (AVX-512/AMX flags), `numactl -H`, `nvidia-smi -q`, the PCIe link generation and width, STREAM Triad at the thread counts used, measured pinned H2D bandwidth, and the GPU clock policy.
   - Lock clocks, or report the unlocked policy.
   - Randomize and interleave the run order of systems (Hoefler §4.1), so that thermal drift and noisy cloud neighbours spread evenly across systems.
   - Record every system's exact commit. For non-llama.cpp systems, record both the paper-era tag and the latest version (prior notes).

2. **Weights: three tiers, never mixed in one headline**
   - **Tier A, bit-identical (headline):**
     - BF16 models, e.g. Qwen3.6-35B-A3B BF16. Converted with `--outtype bf16` from BF16 safetensors, these should be a straight bit copy. This is not verified byte-for-byte, so hash the tensors.
     - MXFP4-expert models (gpt-oss-120b, DeepSeek-V4-Flash). llama.cpp's converter repacks these losslessly.
     - This is exactly the alignment FreeToken used, so no competitor can object to it.
   - **Tier B, byte-matched but not bit-identical:**
     - System-native int4/int8 (Marlin, KT AMXInt8) versus GGUF Q4_0/Q8_0.
     - Match expert bits per weight, report bytes per decoded token, and gate quality MLPerf-style: report KLD and ΔNLL against a BF16 reference, and state the tolerance.
   - **FP8 models:** exclude from the headline. llama.cpp dequantizes FP8 to BF16, doubling expert bytes, or to Q8_0.
   - **ik_llama.cpp:** restrict it to GGUF types that both forks support (BF16, Q8_0, Q4_0, MXFP4). Its IQ_K types change the bytes.

3. **Equal GPU memory, defined as device-level peak used memory for the whole process tree**
   - **What counts:** everything, measured by sampling NVML (`nvidia-smi --query-compute-apps=used_memory` or pynvml) through a full request. That includes the CUDA context, allocator reserve, weights, KV cache, activations, workspace and CUDA-graph pools. This is the quantity that competes with other applications, which is Pipelined Sharding's motivation. It is also the only quantity that is comparable across PyTorch, SGLang and ggml allocators.
   - **Report a breakdown per system:**
     - GPU-resident expert bytes;
     - dense and attention bytes;
     - KV at the fixed maximum context;
     - "other" (context, workspace, graphs).
     This makes an allocator's overhead visible rather than hidden.
   - **Fit each system to the budget by Pipelined Sharding's iterative search:** raise the resident-expert count or cache size until measured peak ≤ budget. Report the chosen knob, e.g. `-ncmoe k`, `--kt-num-gpu-experts n` or `--moe-cache-size s`.
   - **Override auto-sizers explicitly and log their values:**
     - llama.cpp `--fit off` with explicit `-ncmoe`, or a `--fit-target` set to the budget;
     - SGLang/KT `--mem-fraction-static` and `--max-total-tokens`;
     - FreeToken `--memory-ratio`.
   - If a system cannot be capped, allocate a ballast buffer in a separate process so that free memory equals the budget. That process's context then counts against no one, so document it.
   - **Size KV for the same maximum context** in every system. SGLang-style pools otherwise absorb the budget.
   - **Budgets:** use at least three levels: tight, for example about 25% of expert bytes resident; medium; and the whole card. The ranking of systems can change with the budget.

4. **CPU resources: two regimes, both reported**
   - (a) **Equal fixed budget.** Use the same cores, pinned with `numactl --cpunodebind=N --membind=N` to the GPU-local node. Use FreeToken-style 6 threads, or the full socket. Apply the same SMT policy to every system.
   - (b) **Each system at its best thread count within the same core set.** Sweep and report the chosen count. This answers AP2 (untuned baselines).
   - Use a single socket unless every system gets its best NUMA mode. Options are llama.cpp `--numa distribute` or `isolate` plus interleave, and KT's per-NUMA threadpool. Otherwise KTransformers' NUMA-aware tensor parallelism (up to 1.63×, prior notes) or its absence becomes the confound.
   - Report whether the CPU has AMX or AVX-512. KT's paper-level path needs AMX.
   - Record CPU frequency under load, since 2512.16473 notes 4.8–5.3 GHz variation with active cores.

5. **Workload**
   - **Identical token IDs.** Tokenize once with the reference tokenizer and chat template, then feed token IDs to every engine where possible, or verify that tokenization is identical. This avoids the chat-template confound catalogued in 2608.07911 B2 (prior notes).
   - **Prompt set.** Draw prompts from several domains (code, maths, chat). Routing depends on token identity (prior notes, 2604.17182), and routing drives dynamic-cache hit rates.
   - **Separate decode from prefill.** Measure TPOT excluding the first token, as MLPerf Client TPS does, at several fixed context depths (e.g., 128, 1K, 4K, 16K). llama-bench `-d` exists; other engines need a prefill-then-decode timer.
   - **Fixed output length.** Speed runs use a fixed N (e.g., 256 and 1024) with EOS ignored, following MLPerf's min/max new tokens and "no truncation". Natural-EOS runs are used only for output agreement.
   - **Disable prefix and prompt caching**, or use distinct prompts per repetition (MLPerf §5.1).

6. **Warm-up and cache state**
   - Warm the page cache by reading the model files once.
   - Discard one warm-up request per server launch (R-§4.1, MLPerf Client, llama-bench default).
   - For systems with dynamic expert caches (FreeToken, SeqMoE-style predictors, llama.cpp cache branches), report both a cold first request and steady state after a fixed warm-up prompt set. These differ by design. Hoefler's cache-state caveat applies.

7. **Repetitions (Kalibera & Jones levels)**
   - At least 3 fresh process or server launches per system and configuration, times at least 5 measured requests each.
   - Keep adding blocks until the 95% nonparametric CI half-width on the median TPOT is at most 2–3%, per Hoefler's adaptive rule; n > 5 is needed.
   - Report the variance component between launches. On cloud VMs, launch-to-launch variance is plausibly the dominant term.

8. **Summaries**
   - **Per request:** record decode tokens N_d and decode time T_d.
   - **Per system:** report the rate as ΣN_d/ΣT_d, which equals the token-weighted harmonic mean of per-request rates (R3).
   - **Distributions:** report the median and 95% bootstrap CI of per-request TPOT (a cost, so the arithmetic mean or median is fine), plus P90/P99 time-between-tokens (R8, AP7). The latter exposes stalls from misses or PCIe fetches.
   - **Pairwise comparisons:** report absolute tok/s for every system (R1). Give a bootstrap CI for the ratio of rates (Kalibera & Jones), or run Kruskal–Wallis across systems (R7).
   - **Across models:** do not average speed-ups. If a summary is unavoidable, use the geometric mean of ratios and label it (R4).
   - **llama-bench:** its `avg_ts` is an arithmetic mean of rates. Recompute from `samples_ns`.
   - **Per-request means:** FreeToken's "per-request mean tok/s" is also an arithmetic mean of rates. Re-summarize it when comparing.

### Gaps
- I found **no published standard for what counts toward a "VRAM budget"** (context, allocator reserve, KV, graphs) in any offloading paper or benchmark. Pipelined Sharding measures "actual VRAM usage" but does not enumerate components. FreeToken's fetched text had no explicit VRAM-matching statement. The NVML-peak definition above is my proposal.
- I did not verify that `--outtype bf16` from BF16 safetensors is byte-identical, i.e. no rounding path. The MXFP4 path is documented as lossless in code comments only, so hash tensors on the day.
- I did not verify whether `llama-bench` applies `--fit`. It includes `fit.h` with `fit_params_target` defaulting to `{0}`; the behaviour is unread.
- I did not read the Kalibera & Jones primary text; a secondary summary was used.
- I found no rule on SMT or hyperthreading policy for CPU-offload benchmarks.

## Q2. How to verify that systems produce the same outputs (lossless): greedy token agreement, logit and NLL tolerances, and known sources of numerical difference

### Takeaway
"Identical greedy output" is the wrong pass criterion for cross-engine or cross-placement comparisons in BF16. Even mathematically lossless methods match full greedy trajectories in only about 43–45% of prompts in BF16, and in 100% in FP32 (Orthrus, Sep 2026). FreeToken itself concedes that "agent trajectories diverge across engines".

The defensible protocol is layered:
- (i) weight identity, checked by hashes;
- (ii) **teacher-forced** agreement on a fixed token stream: top-1 agreement plus KLD and ΔNLL, compared against a measured **numerical-noise floor**;
- (iii) free-running greedy runs that report match rate, first-divergence position and the logit margin at the divergence;
- (iv) a task-accuracy gate, only for lossy tiers.

Concrete sources of cross-device difference are visible in llama.cpp's code. With the same GGUF weights, CPU matmuls quantize activations to Q8_0 or Q8_K (BF16 for BF16 weights), while CUDA MMQ/MMVQ quantize them to Q8_1. Placement therefore changes the arithmetic.

### Cited Findings
**How large "lossless" differences are**
- **Orthrus** (Koziev, Sinev, Oseledets; arXiv 2609.15504, Sep 2026):
  - In BF16, "exact trajectory matching occurs in only 45% of cases" (43% for their retrained model) over 1,190 prompts across 12 domains. In FP32, "exact matching on all 1,190 evaluated prompts".
  - Divergence correlates with higher response-conditional perplexity (logistic β₁ = −8.10 to −10.92).
  - Recommendation: "evaluations of 'lossless' language-model acceleration should specify both the operational criterion for equivalence and the numerical precision under which it is measured".
  - Source: [arXiv 2609.15504](https://arxiv.org/html/2609.15504v1)
- **"Lossless but Not Free"** (Chordiya, UCSD; arXiv 2607.17283, 19 Jul 2026) uses three verification gates:
  - a 50,000-trial test of the rejection rule;
  - "bit-exact greedy agreement" on deterministic mock models;
  - a χ² two-sample test on real-model token distributions (χ² = 162.5, dof 200, p = 0.976).
  - Source: [arXiv 2607.17283](https://arxiv.org/html/2607.17283v1)
- **Thinking Machines**, "Defeating Nondeterminism": 1,000 T=0 completions of Qwen3-235B produced 80 unique outputs, with the first divergence at token 103. Batch-invariant kernels made all 1,000 identical (prior notes). — [Thinking Machines blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- **Yuan et al.** (arXiv 2506.09501): BF16 greedy decoding shows "up to 9% variation in accuracy and 9,000 tokens difference in response length" across GPU count, type and batch size (prior notes). — [arXiv 2506.09501v2](https://arxiv.org/abs/2506.09501v2)
- **"The Silent Hyperparameter"** (arXiv 2605.19537): backend choice alone shifts scores by up to 16.6 pp. Named causes are prefix caching, CUDA graphs, custom kernels and "engine-specific defaults in logit processing" (prior notes). — [arXiv 2605.19537](https://arxiv.org/abs/2605.19537)
- **FreeToken** (§5.1):
  - Correctness criterion: "the coding runs must produce the reference gold patch, and the W4 runs must complete all thirteen turns".
  - "Agent trajectories diverge across engines so cross-engine wall-clock totals are not compared".
  - Source: [arXiv 2608.16157 HTML](https://arxiv.org/html/2608.16157)
- **KTransformers** reports task accuracy only for its lossy Expert Deferral: "average model accuracy drop remains within 0.5%". It uses greedy decoding for most benchmarks. No cross-engine output-equivalence check was found. — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- **HybriMoE**: no output-equivalence or repetition statement in the evaluation (as summarised). — [arXiv 2504.05897 HTML](https://arxiv.org/html/2504.05897)
- **llama.cpp RFC #24528**: decode perplexity is "statistically identical to pure CPU", and GPU rounding flips near-tie greedy tokens "matching any `-ngl` configuration change" (prior notes). — [Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)
- The researcher's own earlier measurement: 1.6–4.5% greedy top-1 disagreement per teacher-forced step, with NLL within 0.1%, for hybrid versus reference (prior notes). — [numerics_and_prediction.md](research_notes/MoE%20research%20direction%20rethink/numerics_and_prediction.md)

**Tools and tolerances available**
- **llama.cpp `llama-perplexity --kl-divergence-base <file>` / `--kl-divergence`** saves the base model's full logits and then reports, for the test model:
  - mean KLD;
  - ln(PPL ratio);
  - mean Δp;
  - Δp percentiles ("If the percentiles are symmetric then the quantization is essentially just adding noise");
  - RMS Δp;
  - "Same top p", the share of tokens where both models' top-1 agree.
  - Source: [tools/perplexity/README.md](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/perplexity/README.md); [PR #6936](https://github.com/ggml-org/llama.cpp/pull/6936)
- **Reference magnitudes from that README:**
  - BF16 versus FP16 base: "Mean KLD 0.00002515 ± 0.00000020" and "Same top p 99.739 ± 0.013 %".
  - The LLaMA-3-8B scoreboard lists q8_0 at KLD on the order of 1e-3 with about 97.7% same-top-p. The summarised row mapping was ambiguous, so re-read the table before citing these numbers.
  - Source: [tools/perplexity/README.md](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/perplexity/README.md)
- **llama.cpp `test-backend-ops`** checks each backend op against the CPU with a default `max_nmse_err() = 1e-7` (normalized MSE), with per-op overrides such as 2e-7 or 5e-6 for some cases. — [tests/test-backend-ops.cpp](https://github.com/ggml-org/llama.cpp/blob/4d86b2f5e8a04667337e20bd5051570be4554cfe/tests/test-backend-ops.cpp)
- **MLPerf's lossy-tier gate:** 99% or 99.9% of the reference accuracy, plus output length within 90–110% of the reference (see Q1). — [MLPerf Inference Rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
- **SGLang** has `--enable-deterministic-inference` ("deterministic inference and batch invariant operations"). — [SGLang server arguments](https://docs.sglang.io/docs/advanced_features/server_arguments.md)

**Known sources of numerical difference (llama.cpp, verified in code at 4d86b2f5)**
- CPU `vec_dot_type`, i.e. the activation format for each weight type:
  - Q4_0, Q5_0, Q8_0, IQ4_NL, MXFP4 and NVFP4 → **Q8_0**;
  - Q4_1 and Q5_1 → Q8_1;
  - Q2_K–Q6_K and most IQ types → **Q8_K**;
  - BF16 → **BF16**;
  - F16 → F16.
  - Source: [ggml-cpu.c type traits](https://github.com/ggml-org/llama.cpp/blob/4d86b2f5e8a04667337e20bd5051570be4554cfe/ggml/src/ggml-cpu/ggml-cpu.c)
- CUDA quantizes activations to **q8_1** for both MMQ (`quantize_mmq_q8_1_cuda`) and the batch-1 MMVQ path (`quantize_row_q8_1_cuda`). — [ggml-cuda/mmq.cu](https://github.com/ggml-org/llama.cpp/blob/4d86b2f5e8a04667337e20bd5051570be4554cfe/ggml/src/ggml-cuda/mmq.cu); [ggml-cuda/mmvq.cu](https://github.com/ggml-org/llama.cpp/blob/4d86b2f5e8a04667337e20bd5051570be4554cfe/ggml/src/ggml-cuda/mmvq.cu); [ggml-cuda/quantize.cu](https://github.com/ggml-org/llama.cpp/blob/4d86b2f5e8a04667337e20bd5051570be4554cfe/ggml/src/ggml-cuda/quantize.cu)
- ggml exposes a precision-hint API, `ggml_prec_set_src(a, GGML_PREC_Q8 | GGML_PREC_Q4, idx)`, that "allows the implementation to quantize F32, BF16, F16 data of src[1] down to" 8-bit or 4-bit types. How much activations are quantized is therefore a tunable, per-op implementation choice. — [ggml.h](https://github.com/ggml-org/llama.cpp/blob/4d86b2f5e8a04667337e20bd5051570be4554cfe/ggml/include/ggml.h)
- Other engines:
  - KTransformers' CPU backends include AMXInt8/Int4, AVX-512 BF16 and llamafile, each with its own activation handling.
  - FreeToken's CPU kernel has "3 real kernels: scalar/avx2/avx512, no bf16/VNNI variant" (prior notes).
  - Source: [kt-kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md); [FreeToken benchbw.py](https://github.com/FlashML-org/FreeToken/blob/main/python/freetoken/moe/benchbw.py)

### Inferences
**Verification protocol for the race, run before any timing is published**

- **V0. Weight identity (Tier A).**
  - Hash each expert and dense tensor as loaded, or as dumped from the file, in each engine's format.
  - For GGUF BF16 versus safetensors BF16, compare raw bytes.
  - For MXFP4, dequantize both to FP32 and require exact equality.
  - A Tier A claim fails without this.

- **V1. Teacher-forced agreement against a noise floor** (the main criterion).
  - Build a fixed evaluation stream of about 20–50k tokens from the race prompt set, as reference-model greedy continuations under the chat template, i.e. on-policy (see the off-policy caveat in prior notes).
  - Collect full-vocabulary or top-k logprobs from each system over the same token IDs. Tools: llama.cpp `llama-perplexity --kl-divergence`; vLLM/SGLang prompt logprobs; HF; FreeToken and KT through their HF/SGLang paths.
  - **Reference:** HF FP32 on CPU, or all-GPU-resident BF16 if FP32 is too slow for the larger models.
  - **Metrics:** top-1 agreement, mean and P99.9 KLD, and ΔNLL (ln PPL ratio).
  - **Noise floor:** measure the spread among benign re-configurations of one engine, e.g. llama.cpp all-GPU versus two different `-ncmoe` splits, and FP16 versus BF16. The README's BF16-vs-FP16 figures (KLD ≈ 2.5e-5, 99.74% same top-1) show the order of magnitude to expect.
  - **Pass:** a "lossless" system passes if its KLD and top-1 disagreement are within a pre-registered multiple, e.g. ≤ 3×, of the noise floor.
  - **Lossy:** a system is lossy if it sits at Q8_0-quantization-level KLD, about 1e-3, or worse.
  - Pre-register both thresholds.

- **V2. Free-running greedy (descriptive, not pass/fail).**
  - Report the exact-sequence match rate against the reference, the distribution of first-divergence position, and the reference's top-1/top-2 logit margin at each divergence.
  - Claim: divergences concentrate at near-ties. This is testable; for example, over 90% of divergences have a margin below the P5 margin.
  - Following Orthrus, do not expect 100% match in BF16. State the precision.

- **V3. Routing agreement (MoE-specific, optional).**
  - Where engines expose router outputs (llama.cpp eval-callback, HF hooks), report the share of (token, layer) pairs with an identical top-k expert set on the teacher-forced stream.
  - Numerics-driven top-k flips change cache behaviour, so this links correctness to performance (prior notes).

- **V4. Lossy tiers only (Tier B, Expert Deferral, speculative variants).**
  - Apply an MLPerf-style task gate on a small benchmark, e.g. ≥ 99% of the reference on a GSM8K or MMLU subset. Report the result beside the speed.

- **Controls that remove non-numerical differences:**
  - identical token IDs;
  - identical chat template;
  - sampling forced to pure greedy, with no repetition penalty, top-p/min-p or logit processors (engine defaults differ, per the Silent Hyperparameter);
  - prefix caching disabled;
  - the same maximum context.

- **Expected sources of difference to name in the paper:**
  - activation quantization format, CPU Q8_0/Q8_K versus CUDA Q8_1, which is placement-dependent even for identical GGUF;
  - BF16 rounding of activations on CPU BF16 paths;
  - different accumulation order and FMA contraction in AMX, AVX-512 and CUDA kernels;
  - FlashAttention versus unfused attention;
  - batch-shape and CUDA-graph differences;
  - dynamic placement, where cache history decides whether an expert runs on CPU or GPU. The last one can create **run-to-run** nondeterminism at batch 1 (prior notes, "placement invariance").

### Gaps
- I found **no published cross-engine tolerance** (KLD, top-1 agreement or NLL) for claiming "lossless" hybrid CPU/GPU MoE inference. Every threshold above is a proposal.
- The vLLM FAQ on output variation could not be fetched, so vLLM's official statement of causes and mitigations is not quoted.
- I did not verify whether FreeToken, KTransformers (kt-kernel/SGLang), ik_llama.cpp or 2512.16473 expose per-token logprobs over a supplied token stream. Where they do not, V1 must use their HF-compatible or OpenAI-API prompt-logprob paths, or a patch.
- The LLaMA-3-8B scoreboard values in the llama.cpp README (q8_0 KLD) were ambiguous in the summary and need re-reading.

## Q3. How were the named head-to-heads run, and what did critics or later papers say?

### Takeaway
None of the five named evaluations meets the full contract.

| Paper | Precision alignment | Baseline | Threads / cores | VRAM matching | Repetitions | Output check |
|---|---|---|---|---|---|---|
| KTransformers | Mismatched (llama.cpp in FP16) | Patched, not stock | — | — | Not stated | — |
| HybriMoE | Marlin int4 | llama.cpp as a static layer split | Cores capped at 10 | Cache ratio | None stated | None stated |
| Pipelined Sharding | — | llama.cpp found by `-ngl` search, some rows `-cmoe` | Physical cores | Measured-VRAM search | Unclear | Accuracy "preserved" by construction |
| FreeToken | Bit-aligned BF16/MXFP4 | "Static split" llama.cpp | 6 threads pinned to NUMA node | Not stated in fetched text | Per-request arithmetic mean | Task-completion checks only |
| SeqMoE | — | llama.cpp flags not given | — | — | — | — |

SeqMoE also has no public code.

Later papers re-measured the same systems on their own hosts, and got different numbers. SeqMoE measured FreeToken at 86.2 tok/s on a 5090 platform; FreeToken's paper reported 77.1 tok/s at a similar budget. Community reports show single-request results that depend on configuration. No formal published critique of any of the five was found; SeqMoE and FreeToken are weeks to months old.

### Cited Findings
**FreeToken (arXiv 2608.16157, Aug 2026)**
- **Six machines**, with measured B_P (pinned H2D) and B_H:

  | Machine | GPU | CPU | Host memory | B_P (GB/s) | B_H (GB/s) |
  |---|---|---|---|---|---|
  | RTX 5090 server | RTX 5090 | Xeon Gold 6459C | DDR5 180 GiB | 52.7 | 77.3 |
  | RTX 4090 | RTX 4090 | Xeon Platinum 8358P | DDR4 | 25.1 | 63.2 |
  | RTX 3090 | RTX 3090 | Xeon Gold 6330 | — | 25.3 | 56.7 |
  | RTX 5090 desktop | RTX 5090 | Ryzen 9 9950X3D | — | 49.0 | 53.8 |
  | RTX 4060 Laptop | RTX 4060 Laptop | Core i9-13900H | LPDDR5 32 GiB | 11.8 | 47.5 |
  | RTX PRO 6000 | RTX PRO 6000 | Xeon Platinum 8559C | DDR5 512 GiB | 51.5 | 178 |

- Setup: rented servers were capped at 6 threads and pinned to the NUMA node. Engines were llama.cpp, Ollama, KTransformers and MoE-Infinity. Workloads W1–W4 were agentic (AIME, SWE-bench via OpenCode, Claude Code, OpenClaw).
- Metrics are "per-request mean tok/s" and TTFT. Weight formats are "exactly aligned". The RTX 4060 laptop used the "official NVFP4 release".
- Source: [arXiv 2608.16157 HTML](https://arxiv.org/html/2608.16157)
- Its llama.cpp baseline is described only as a "routing-blind static split" that assigns "whole layers to devices". No flags were found (prior notes). — [arXiv 2608.16157 HTML](https://arxiv.org/html/2608.16157)
- Third-party reactions (prior notes):
  - A community test on an RTX 5090 with an in-VRAM model found single-request FreeToken at 196.9 tok/s against llama.cpp Q4_K_M at 296.4 tok/s. llama.cpp flags were not documented. — [zenn.dev](https://zenn.dev/holy_fox/articles/53b82eed45f956?locale=en)
  - A review "did not reproduce the paper's hardware benchmarks". — [Wavect review](https://wavect.io/blog/freetoken-ai-inference-engine-review/)
- SeqMoE ran FreeToken as a baseline on its own RTX 5090 + Xeon 8470Q platform: 86.2 / 46.3 tok/s at 40% / 15% residency (Qwen3.6). FreeToken's own paper reported 77.1 tok/s at 23.8 GB on its 5090 server (prior notes). — [normalized_v2.jsonl](data/audit/normalized_v2.jsonl); [audit.md](prereg/audit/audit.md)

**KTransformers vs llama.cpp (SOSP'25)**
- "Since Llama.cpp originally only supports layer-wise offloading and lacks expert-level offloading capability, to ensure fair comparisons, we extended Llama.cpp with custom code to enable expert-level offloading analogous to Fiddler."
- llama.cpp ran FP16 because it lacked BF16 CUDA kernels.
- Decode used "a fixed prompt length of 32 tokens" and "a maximum of 512 tokens".
- The host was a dual-socket 8452Y (MLC 220/125 GB/s).
- No thread, NUMA or version details were given for the baselines, and no fraction-of-bandwidth figure.
- Source: [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- It received all three SOSP'25 AE badges, with no pinned commit (prior notes). — [sysartifacts SOSP'25 results](https://sysartifacts.github.io/sosp2025/results)
- A later KT tutorial (MiniMax-M2.1, 2× RTX 5090 + EPYC 9355) claims "30% faster decode" against llama.cpp, with the caveat "We made our best effort to optimize llama.cpp performance" (prior notes). — [KT MiniMax-M2.1 tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/MiniMax-M2.1-Tutorial.md)
- The project's audit found that the direction of the KT llama.cpp-baseline verdicts hinges on whether llama.cpp used one or both sockets. The paper does not state this (prior notes). — [audit_requirements.md](research_notes/MoE%20audit%20measurement%20plan/audit_requirements.md)

**Pipelined Sharding vs llama.cpp (NVIDIA, MLSys'26 oral)**
- **Setup:** measured-VRAM `-ngl` search at the same budget; threads equal to physical cores (16 or 8); 100 output tokens; contexts 1K–64K.
- **Platforms:**
  - cli1: RTX 3500, 119.5 GB/s DRAM, 13 GB/s PCIe;
  - cli2: 5070 Ti, 57.6 GB/s;
  - cli3: 5090 + 16-core EPYC, 153.6 GB/s, 50 GB/s PCIe.
- **Accuracy:** "Our optimizations preserve model accuracy and do not change the operations executed". No numerical check was reported.
- **AE validation:** "Speedup within 90% of the paper's value" prints PASS.
- The paper uses a roofline internally for its profiler, not for end-to-end bounds.
- Source: [arXiv 2604.26334 HTML](https://arxiv.org/html/2604.26334)
- The artifact's baselines, derived from its CSV, include `-ngl 20/1/36` as well as `-ngl + -cmoe`. On the 0 GB row, llama.cpp `-cmoe` measured 25.7 tok/s against the project model's prediction of 49.5 [38.4, 65.1] (m ≈ 0.52) (prior notes). — [system_reproducibility.md](research_notes/MoE%20audit%20measurement%20plan/system_reproducibility.md)

**HybriMoE vs KTransformers (DAC'25)**
- "HybriMoE achieves an average speedup of 1.33× in the prefill stage and 1.70× in the decode stage" over kTransformers, on which it is built. — [arXiv 2504.05897](https://arxiv.org/abs/2504.05897)
- Setup: RTX A6000 and a Xeon Gold 5220R restricted to 10 cores. Baselines were llama.cpp ("static layer-to-device mapping"), AdapMoE and kTransformers, all with Marlin 4-bit. Cache ratios were 25%, 50% and 75%. Metrics were TTFT (inputs of about 32–1024 tokens) and TBT. Datasets were MT Bench, Vicuna Bench and ChatGPT Prompts. — [arXiv 2504.05897 HTML](https://arxiv.org/html/2504.05897)
- From prior notes:
  - The released optimize rules enable the HybriMoE cache only for DeepSeek-V2.
  - For DeepSeek-V2-Lite, the project model predicts a properly configured llama.cpp at 81–140 tok/s, against the paper's HybriMoE figure of 14.5–15.3 tok/s.
  - Source: [system_reproducibility.md](research_notes/MoE%20audit%20measurement%20plan/system_reproducibility.md)

**SeqMoE (arXiv 2609.12978, 11 Sep 2026)**
- Evaluates Qwen3-30B-A3B, Qwen3.6-35B-A3B, GPT-OSS-120B and DeepSeek-V4-Flash against llama.cpp, MoE-Infinity, KTransformers and FreeToken. — [arXiv 2609.12978 HTML](https://arxiv.org/html/2609.12978)
- From prior notes:
  - llama.cpp is described only as supporting "concurrent CPU–GPU computation", with no flags.
  - There is no code release.
  - Its "Static Offload" llama.cpp was 30.9 tok/s against SeqMoE's 104.1 tok/s on an RTX 4090 at 45% residency. The model-predicted equal-VRAM llama.cpp is 61.7 tok/s.
  - Source: [system_reproducibility.md](research_notes/MoE%20audit%20measurement%20plan/system_reproducibility.md); [audit_requirements.md](research_notes/MoE%20audit%20measurement%20plan/audit_requirements.md)

**2512.16473 (CPU-GPU collaborative, ASP-DAC'26)**
- Speed-ups are reported over its own CPU-only mode: "15% ~ 35% and 50% ~ 250% performance improvements over the CPU only setup". The throughput metric includes prefill. — [arXiv 2512.16473 HTML](https://arxiv.org/html/2512.16473)
- Its Fiddler baseline, at 0.2 tok/s, was flagged "implausibly low" (prior notes). — [system_reproducibility.md](research_notes/MoE%20audit%20measurement%20plan/system_reproducibility.md)

**ik_llama.cpp and the community**
- A community issue comparing ik_llama.cpp, vLLM, llama.cpp and ktransformers found "no direct, controlled comparisons". It raises confounds: expert count (6 vs 8), quantization differences and mismatched hardware. One comment: "might not be as performant as ktransformers (haven't seen a direct comparison yet)". — [ubergarm/r1-ktransformers-guide #11](https://github.com/ubergarm/r1-ktransformers-guide/issues/11)
- The ik_llama.cpp quick-start shows one flag changing DeepSeek-R1 decode from about 7.35 to about 13.13 tok/s: runtime repacking, `-rtr`. That is an example of how parameter tuning dominates cross-engine results. — [ik_llama.cpp Discussion #258](https://github.com/ikawrakow/ik_llama.cpp/discussions/258)
- A llama.cpp expert-cache RFC author warns: "Cross-engine comparisons systematically flatter baseline by ~4%" (prior notes). — [llama.cpp Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)
- Agrawal et al.'s AP1 and AP2 name exactly these failure modes: "conflate engineering effort with algorithmic novelty" and untuned baselines. — [arXiv 2507.09019 HTML](https://arxiv.org/html/2507.09019)

### Inferences
- **The race's added value over every precedent** is the combination of:
  - (a) one host for all systems, re-measured by a neutral party;
  - (b) measured-VRAM equalization with components enumerated (Pipelined Sharding does the search but not the enumeration);
  - (c) FreeToken-style bit-aligned weights;
  - (d) harmonic summarization with confidence intervals;
  - (e) a speed-of-light score.
  No named paper has all five.
- **Include `llama.cpp -ngl` (layer split) as a labelled "as-published baseline" row next to `-ncmoe` / `-ot exps=CPU` tuned at equal VRAM.** It shows directly how much of each paper's reported gain came from baseline choice. This is AP1/AP2 made quantitative, without accusing anyone.
- **Use FreeToken's own discipline** (6 threads, NUMA-pinned; BF16/MXFP4 bit-aligned) for the Qwen3.6 and DS-V4-Flash rows. FreeToken's authors then cannot object to the setup, and the only change is the equal-VRAM tuned llama.cpp.
- **KTransformers:** run the current kt-kernel + sglang-kt path, with `--mem-fraction-static`, KV size and `--kt-num-gpu-experts` set to the budget, on an AMX host. Also run the archived paper-era path if it builds. Report both, labelled.
- **SeqMoE:** it cannot be raced. Report it only as "reported number, rescaled to the race host by the bound/model", with no measured head-to-head.

### Gaps
- The FreeToken text as fetched gave no explicit VRAM-matching procedure, baseline flags or repetition count. Its appendix or artifact may contain them; not checked.
- The detailed SeqMoE §9 methodology (repetitions, flags, lengths) could not be extracted.
- No formal published critique (paper, comment or blog by maintainers) of the KTransformers, HybriMoE, Pipelined Sharding, FreeToken or SeqMoE comparisons was found. The only evidence is community tests and the cross-paper re-measurements above. This is search absence, not proof.
- No controlled ik_llama.cpp versus KTransformers versus llama.cpp comparison at equal VRAM was found.

## Q4. Standards, reviewer expectations, artifact badges, and how to report a competitor that does not reproduce

### Takeaway
The reference points below converge on one reporting norm. Report the competitor's published number, your measured number with a CI, the exact version and configuration, the steps you took to reproduce, and the fact that you contacted the authors. Frame a mismatch as "not reproduced under stated conditions", in ACM's vocabulary, not as "wrong".

The reference points:
- **Hoefler's 12 rules:** document every factor; explain any subset reported.
- **MLPerf:** "Results that cannot be replicated are not valid results."
- **ACM badging:** a precise vocabulary for "reproduced" versus "replicated".
- **sysartifacts evaluator guide:** give authors a chance to fix issues; functional correctness before numbers.
- **NeurIPS MLRC 2026:** "A careful, well-documented failure to reproduce a result … is a genuine contribution".

ISPASS explicitly solicits "Confirmations or refutations of important prior results". EuroSys and ASPLOS want limitations shown and sound methods; ASPLOS mandates the SIGPLAN empirical-evaluation checklist.

### Cited Findings
**Artifact badges**
- ACM definitions:
  - **Reproducibility:** "obtained with stated precision by a different team using the same measurement procedure, the same measuring system".
  - **Replicability:** "by a different team, a different measuring system".
  - **Results Reproduced:** "The main results of the paper have been obtained in a subsequent study by a person or team other than the authors, using, in part, artifacts provided by the author."
  - **Results Replicated:** the same, "without the use of author-supplied artifacts".
  - **Functional:** "documented, consistent, complete, exercisable".
  - Source: [ACM Artifact Review and Badging (current)](https://www.acm.org/publications/policies/artifact-review-and-badging-current)
- sysartifacts covers ATC 2022–26, CAIS 2026, EuroSys 2021–27, FAST 2024–25, OSDI 2020–25, SC 2021 and SOSP 2019–26. — [sysartifacts](https://sysartifacts.github.io/)
- sysartifacts evaluator guide:
  - "Merely reproducing similar output as the paper, such as performance metrics, is not enough, the artifact must actually do what it claims to do."
  - "Give authors a chance to fix issues by discussing through HotCRP comments before deciding that their artifact should not get a badge."
  - "It is acceptable to deny badges if artifacts require unreasonable effort."
  - It sets no numeric tolerance for "reproduced".
  - Source: [sysartifacts evaluator guide](https://sysartifacts.github.io/evaluator-guide.html)
- The one concrete tolerance precedent in this area: Pipelined Sharding's AE scripts print PASS when "Speedup within 90% of the paper's value". — [arXiv 2604.26334 HTML](https://arxiv.org/html/2604.26334)
- MLSys 2027 AE is voluntary and does not affect acceptance; EuroSys 2027 AE is optional, with Available, Functional and Reproduced badges (prior notes). — [MLSys 2027 CFP](https://mlsys.org/Conferences/2027/CallForResearchPapers); [EuroSys 2027 CFP](https://2027.eurosys.org/cfp.html)

**Non-reproduction norms**
- MLRC 2026, now an official NeurIPS track:
  - It solicits reproductions "whether they confirm, partially replicate, or fail to reproduce prior results".
  - "A careful, well-documented failure to reproduce a result — with a clear account of what was tried and what was found — is a genuine contribution".
  - It values work that "pushed those claims into new settings … and reported back with nuance".
  - No guidance on contacting authors was found on the page.
  - Source: [NeurIPS blog, MLRC 2026](https://blog.neurips.cc/2026/05/04/mlrc-2026-reproducibility-as-an-official-track-at-neurips/)
- MLPerf §2.7: "Results that cannot be replicated are not valid results." — [MLPerf Inference Rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
- Weak-baseline folklore (Pakin, "Ten Ways to Fool the Masses When Giving Performance Results on GPUs", HPCwire, 13 Dec 2011, updating Bailey 1991):
  - "Compare heavily optimized GPU code to unoptimized CPU code."
  - "Compare full (or even multiple) GPU performance to a single CPU core."
  - "Don't time data movement or kernel-invocation overhead."
  - Source: [HPCwire](https://www.hpcwire.com/2011/12/13/ten_ways_to_fool_the_masses_when_giving_performance_results_on_gpus/); Bailey's original: [IEEE Xplore](https://ieeexplore.ieee.org/document/6278631/) and [Hager's commentary](https://blogs.fau.de/hager/archives/5260)

**Venue expectations**
- ISPASS 2026 topics include "Confirmations or refutations of important prior results". Tool/benchmark papers must "open-source their tool/benchmark before the conference". — [ISPASS 2026 CFP](https://ispass.org/ispass2026/cfp.php)
- From prior notes:
  - EuroSys 2027 wants rigorous evaluation showing "both benefits and limitations" against prior work, and accepts experience papers.
  - ASPLOS 2027 papers "must follow SIGPLAN's empirical evaluation guidelines".
  - MLSys 2027 judges "novelty, quality, interest, and impact".
  - Sources: [EuroSys 2027 CFP](https://2027.eurosys.org/cfp.html); [ASPLOS 2027 CFP](https://www.asplos-conference.org/asplos2027/cfp/); [MLSys 2027 CFP](https://mlsys.org/Conferences/2027/CallForResearchPapers)
- SIGPLAN Empirical Evaluation Checklist: a one-page checklist in seven categories. Version dated 26 Oct 2018; committee chair Steve Blackburn, with Hauswirth, Berger and Hicks. — [SIGPLAN Empirical Evaluation](https://www.sigplan.org/Resources/EmpiricalEvaluation/)

**Benchmarks of MoE systems**
- MoE-CAP (NeurIPS 2025) defines sparsity-aware S-MBU/S-MFU and runs vLLM, SGLang, MoE-Infinity, K-Transformers and HF itself. It does not reconcile other papers' claims, and it uses a single B_peak (prior notes). — [arXiv 2412.07067](https://arxiv.org/abs/2412.07067); [GitHub Auto-CAP/MoE-CAP](https://github.com/Auto-CAP/MoE-CAP)
- 2608.07911 (Aug 2026) owns the trace-level "no common evaluation contract" framing and checklist. It explicitly excludes real transfer, overlap and bandwidth (prior notes). — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)

### Inferences
**How to report each competitor, as a per-system status block in the paper and artifact**

- **Status.** Use exactly one of:
  - (a) *Reproduced*: the paper's own configuration on comparable hardware lands within a pre-registered tolerance of the paper's number after hardware rescaling. Pipelined Sharding's AE uses 90%; I suggest the band implied by the bound/model uncertainty.
  - (b) *Runs, not reproduced*: the measured value falls outside the tolerance.
  - (c) *Does not run*: a build or runtime failure after a documented effort budget, e.g. 1 engineer-day.
  - (d) *Not runnable*: no code.

  Use ACM's "reproduced" only when the authors' artifact was used; otherwise use "replicated".

- **Content of each block:**
  - commit or tag, both the paper-era one and the latest;
  - exact command lines and the equal-VRAM knob value;
  - the host's STREAM figure and PCIe bandwidth;
  - the paper's number;
  - the paper's number rescaled to the race host by the bytes/bandwidth model, pre-registered before measuring;
  - measured median and 95% CI;
  - fraction of speed-of-light;
  - effort log and hypotheses for any gap (host bandwidth, AMX absent, single socket, driver).

- **Process:**
  - Email the authors with logs and configurations at least 1–2 weeks before posting, and state in the paper that they were contacted and whether they replied. This mirrors sysartifacts' "chance to fix issues".
  - Invite them to supply a better configuration, then re-run it and report both.
  - Keep all failed or partial systems in the paper with their status (R2: explain subsets). Never silently drop one.

- **Tone:**
  - Attribute gaps to measurable causes (baseline choice, host bandwidth, precision) rather than to authors. Use 2608.07911's stance as the explicit contrast: it declines to say published speedups are wrong. The race goes one step further only where a same-host measurement shows it.
  - Put "as-published baseline" rows beside "tuned equal-VRAM baseline" rows so readers see the decomposition themselves.

- **Pre-registration.** Freeze the protocol, the tolerances, the predicted numbers per system and the analysis script, e.g. as a tagged commit or on OSF, before renting the machine. This is uncommon in systems work but aligned with SPCL's values (prior notes). It also protects against accusations of tuning the harness in one's own favour, because the researcher's own system is a competitor.

- **Own-system conflict of interest.** State in the paper that the author's own system is in the race. Apply the identical protocol to it. Give competitors the same tuning budget, or more, and report the tuning budget per system (AP2).

### Gaps
- The SIGPLAN checklist items and the Bailey 1991 items could not be fetched verbatim.
- No MLSys-, ISPASS- or EuroSys-specific reviewer guideline text on baselines or non-reproduction was found beyond CFP wording. Reviewer expectations above are inferred from CFPs and accepted papers (prior notes).
- The ISPASS 2027 CFP was not found; it is presumably unposted.
- I found no published norm with a specific waiting period for author contact before publishing a non-reproduction.

## Q5. Precedent for reporting performance as a fraction of a speed-of-light or roofline bound

### Takeaway
There is strong precedent at the kernel and hardware level:
- Nsight Compute's "Speed Of Light" percentage;
- the roofline model;
- FlashAttention-2's "50-73% of the theoretical maximum FLOPs/s";
- Databricks' MBU for LLM decode;
- MoE-CAP's sparsity-aware S-MBU.

Hoefler's R11 makes showing bounds a methodological rule, and §2.1.1 argues that speed-up "can almost always be replaced by lower bounds on execution time". The named MoE-offload head-to-heads report speed-ups and tok/s, not fraction of a bound. FreeToken only notes that prefill hits the "practical ceiling of the PCIe 5.0 x16 link", and Pipelined Sharding uses a roofline only inside its profiler. Scoring every raced system against a tiered speed-of-light is well precedented in form and, as far as I found, new in this setting.

### Cited Findings
- **Hoefler R11 and §5.1** give example bounds: ideal linear speed-up, Amdahl's law and overhead lower bounds. §2.1.1: "speedup itself is a rather meaningless measure because it will typically be higher on slow processors and/or less optimized codes" and "can almost always be replaced by lower bounds on execution time." — [SC'15 PDF](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf)
- **Nsight Compute** "SpeedOfLight" section: "the achieved percentage of utilization with respect to the theoretical maximum" for compute and memory, also shown as roofline charts. — [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- **Databricks** (Agarwal et al., 12 Oct 2023):
  - "MBU is defined as (achieved memory bandwidth) / (peak memory bandwidth) where achieved memory bandwidth is ((total model parameter size + KV cache size) / TPOT)".
  - At batch 1, about 55% MBU on A100-40GB and about 60% on H100-80GB.
  - "MBU is also useful to compare different inference systems (hardware + software) in a normalized manner."
  - Source: [Databricks blog](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
- **MoE-CAP**: S-MBU = B_achieved/B_peak, with B_achieved = (S_activated + S_KV)/TPOT, where S_activated counts only experts routed to (prior notes). — [arXiv 2412.07067](https://arxiv.org/abs/2412.07067)
- **FlashAttention-2** (Dao, 2023): "reaching 50-73% of the theoretical maximum FLOPs/s on A100" and "72% model FLOPs utilization". — [arXiv 2307.08691](https://arxiv.org/abs/2307.08691)
- **FreeToken**: prefill reaches "52.7 GB/s … the practical ceiling of the PCIe 5.0 x16 link". Its q* ≈ m·B_P/B_H split uses measured, not datasheet, bandwidths. — [arXiv 2608.16157 HTML](https://arxiv.org/html/2608.16157)
- **Pipelined Sharding** builds "a roofline graph with this chosen benchmark kernel" for its profiler. It reports no end-to-end bound or fraction of bandwidth. — [arXiv 2604.26334 HTML](https://arxiv.org/html/2604.26334)
- **KTransformers** reports tok/s and speed-ups only, with no utilization fraction. — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- **The original roofline model**: Williams, Waterman, Patterson, "Roofline: an insightful visual performance model for multicore architectures", CACM 2009. Not fetched in this session. — [ACM DL](https://dl.acm.org/doi/10.1145/1498765.1498785)
- **Datasheet versus attainable:** cloud-VM STREAM reached 46–59% of datasheet on the project's anchor hosts (prior notes). A bound computed from datasheet numbers therefore understates achieved efficiency by up to about 2× compared with a bound from measured numbers. — [paper/paper.tex](paper/paper.tex)

### Inferences
**Scoring in the race**
- **Two denominators.** For every system × model × budget, report achieved tok/s as a fraction of the speed-of-light at that same GPU budget:
  - SoL_datasheet, from spec-sheet HBM, PCIe and DRAM bandwidths;
  - SoL_measured, from same-day STREAM Triad at the run's thread count, pinned-H2D bandwidth, and a GPU bandwidth microbenchmark.
  The first is comparable across papers; the second separates software inefficiency from a slow cloud host.
- **Per-system placement.** Where the bound depends on which experts are resident, give both:
  - (a) the bound for the system's own placement or cache policy;
  - (b) the optimal-placement bound at that budget.
  The gap between (a) and (b) is policy loss. The gap between measured throughput and (a) is execution loss.
- **Relation to existing metrics.** This is S-MBU generalized to tiered memory (prior notes). State the relationship explicitly so reviewers map it onto MBU and S-MBU.
- **Presentation.** Put the bound on every throughput plot (R11, R12). Make "fraction of SoL" the headline axis rather than speed-up (Hoefler §2.1.1). Keep absolute tok/s and CIs alongside (R1, R5).

### Gaps
- I found no MoE-offloading or hybrid CPU/GPU paper that reports end-to-end decode as a fraction of a derived bound. This is search absence across the named papers and the prior notes, not an exhaustive scan.
- The roofline CACM paper and MoE-CAP's multi-tier handling were not re-read in this session.
