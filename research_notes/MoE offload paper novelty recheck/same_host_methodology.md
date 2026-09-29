# Same-host, same-client benchmarking of MoE-offload systems: prior art and reviewer expectations (as of 28 Sep 2026)

Method note.
- Sources were read with WebFetch, which returns a small-model summary. Quoted wording is therefore "as summarised" unless marked otherwise.
- This note extends, and does not repeat, four earlier notes: `research_notes/MoE offload system race plan/fair_race_methodology.md` (Hoefler's 12 rules verbatim, MLPerf rules, a table of how named head-to-heads were run, ACM badges), `research_notes/MoE hybrid decode novelty check/venue_bar.md`, `research_notes/MoE research direction rethink/benchmarking_and_offpolicy.md` and `research_notes/MoE audit measurement plan/cloud_hardware.md`.
- Facts marked "(prior notes)" come from those files. Their primary URLs are cited here, but I did not re-fetch them this session.
- The study being assessed:
  - one client, FreeToken's `benchmarks/bench_decode_moe.py` at `0d652e7`;
  - three systems: llama.cpp `--n-cpu-moe`, a llama.cpp-based GPU expert cache, and FreeToken;
  - one RTX 5090 + Ryzen host, at 11/25/40/44% of experts in VRAM, on gpt-oss-120b;
  - a host lottery of 19–22%, and paired bit-identical workloads across hosts.

## Q1. Have others published same-machine, same-client head-to-heads of local LLM/MoE engines? What methodology, and what did they find (esp. gpt-oss-120b / Qwen3-MoE on 24–32 GB GPUs)?

### Takeaway
Running every system on the same machine is the norm in academic MoE-offload papers: FreeToken, KTransformers, SeqMoE, Pipelined Sharding, DALI and HybriMoE all run their baselines on their own hosts. So "same machine" alone is not new. What is new, as far as I found:
- (a) a neutral harness that is literally a competitor's own unmodified benchmark script, validated to 1.3%;
- (b) llama.cpp tuned at equal expert memory (best `--n-cpu-moe`), instead of the "static layer split" that FreeToken, SeqMoE and HybriMoE used;
- (c) paired bit-identical workloads across hosts;
- (d) gpt-oss-120b on a 32 GB consumer card.

No published paper races FreeToken, llama.cpp and an expert-cache llama.cpp on gpt-oss-120b on an RTX 5090. FreeToken's paper does not evaluate gpt-oss-120b at all. SeqMoE runs GPT-OSS-120B only on a 96 GB RTX PRO 6000.

### Cited Findings
**FreeToken (arXiv 2608.16157 v1, 17 Aug 2026)**
- Platforms:
  - six discrete-GPU machines: RTX 5090 server (PCIe 5.0 x16, 52.7 GB/s transfer), RTX 4090, RTX 3090, "RTX 5090 Desktop" with a Ryzen 9 9950X3D, RTX 4060 Laptop, and RTX PRO 6000;
  - rented servers were "capped at 6 CPU threads" to emulate edge conditions, and the edge machines ran unthrottled.
  - Source: [arXiv 2608.16157 HTML](https://arxiv.org/html/2608.16157v1)
- Models and baselines:
  - models: DeepSeek-V4-Flash (MXFP4 experts), Qwen3.6-35B-A3B (BF16) and GLM-5.2 (RTX PRO 6000 only). There is no gpt-oss.
  - baselines: llama.cpp, Ollama, KTransformers and MoE-Infinity, with "identical checkpoint formats".
  - Source: [arXiv 2608.16157 HTML](https://arxiv.org/html/2608.16157v1)
- Workloads and metrics:
  - workloads: W1 AIME chain-of-thought (single-turn, decode-dominated), W2/W3 SWE-bench via OpenCode and Claude Code (56–65k-token sessions), and W4 an OpenClaw email/calendar agent (~24.5k-token context);
  - metrics: "mean tokens/second per request" and mean TTFT, averaged over whole trajectories. "No explicit confidence intervals or error bars" were reported.
  - Source: [arXiv 2608.16157 HTML](https://arxiv.org/html/2608.16157v1)
- Results on RTX 5090:
  - Qwen3.6-35B at 77–83 tok/s, 1.8–2.3× the strongest baseline;
  - DeepSeek-V4-Flash at 22–25 tok/s, 1.5–1.9×;
  - cross-hardware gains of 1.3–2.1×;
  - LRU cache miss rate on Qwen3.6 of 16%, against 41% for KTransformers and 62% for llama.cpp at equal capacity.
  - Source: [arXiv 2608.16157 HTML](https://arxiv.org/html/2608.16157v1)
- Its llama.cpp baseline is described only as a "routing-blind static split" that assigns "whole layers to devices". No flags were given (prior notes). — [arXiv 2608.16157 HTML](https://arxiv.org/html/2608.16157)
- Measured per-machine bandwidths (prior notes):
  - 5090 server: B_P = 52.7 GB/s, B_H = 77.3 GB/s;
  - 5090 desktop (9950X3D): B_P = 49.0 GB/s, B_H = 53.8 GB/s.
  - Source: [arXiv 2608.16157 HTML](https://arxiv.org/html/2608.16157)

**KTransformers (SOSP'25, Oct 2025)** (prior notes)
- Baselines were Fiddler and a llama.cpp "extended … with custom code to enable expert-level offloading analogous to Fiddler". llama.cpp ran in FP16 because it "lacks BF16 CUDA kernels".
- All systems ran on the same dual-socket Xeon 8452Y host (Intel MLC: 220 GB/s intra-socket, 125 GB/s cross-socket).
- Decode used "a fixed prompt length of 32 tokens" and "a maximum of 512 tokens".
- No thread, NUMA or version details were given for the baselines, and no repetition count or CI.
- Source: [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf); [ACM DL](https://dl.acm.org/doi/10.1145/3731569.3764843)
- It received all three SOSP'25 AE badges, with no pinned commit (prior notes). — [sysartifacts SOSP'25](https://sysartifacts.github.io/sosp2025/results)

**SeqMoE (arXiv 2609.12978, 11 Sep 2026)** (prior notes)
- Qwen3-30B-A3B-FP8 on an RTX 4090 at 45% expert residency, batch-1 decode:
  - llama.cpp "Static Offload": 30.9 tok/s;
  - KTransformers: 34.2;
  - MoE-Infinity: 11.9;
  - SeqMoE: 104.1;
  - FullLoad (vLLM): 122.7.
- GPT-OSS-120B appears only on the RTX PRO 6000.
- It re-measured FreeToken on its own RTX 5090 + Xeon 8470Q host at 86.2 and 46.3 tok/s (40% and 15% residency, Qwen3.6). No code is released.
- Source: [arXiv 2609.12978](https://arxiv.org/abs/2609.12978)

**Pipelined Sharding (NVIDIA, MLSys'26 oral; arXiv 2604.26334)** (prior notes)
- Built on llama.cpp, with a `-cmoe` baseline on 3 client platforms, including an RTX 5090 + 16-core EPYC (Gen5).
- Baselines were fitted by a measured-VRAM `-ngl` search per model and budget. Output was 100 tokens at 1K–64K context.
- Average decode gain was 3.7× on Qwen3-30B/235B.
- Source: [arXiv 2604.26334 HTML](https://arxiv.org/html/2604.26334)

**DALI (arXiv 2602.03495, 3 Feb 2026)** (prior notes)
- RTX 3090 + EPYC 7532.
- llama.cpp and KTransformers were configured by controlling "the number of MoE layers stored and executed on the GPU … to maintain a comparable memory usage".
- Decode speed-ups were 3.97×, 2.16×, 1.48× and 1.32× over llama.cpp, KTransformers, MoE-Lightning and HybriMoE.
- Source: [arXiv 2602.03495](https://arxiv.org/abs/2602.03495)

**Community same-machine data for gpt-oss-120b on an RTX 5090**
- moe-autopilot, a llama.cpp fork with a static hot-expert set:
  - hardware: RTX 5090 + Ryzen 9 9950X3D + DDR5-6000;
  - gpt-oss-120B `--n-cpu-moe 25`: 36.2 → 47.6 tok/s (+31.6%);
  - Qwen3.6-35B: 115.8 → 128.3 tok/s.
  - This is an A/B within one engine, with its own client (prior notes). It is the closest public analogue to the "expert cache vs stock llama.cpp" comparison in this study.
  - Source: [JigSawPT/moe-autopilot](https://github.com/JigSawPT/moe-autopilot)
- Hardware Corner (10 Nov 2025):
  - gpt-oss-120b on an RTX 5090 with `--n-cpu-moe 21`: 8.14–9.60 tok/s generation;
  - RTX 3090 + EPYC 7343 + 64 GB DDR4-3200 with `--n-cpu-moe 27`: 1.41–1.81 tok/s;
  - contexts 85–5650 tokens; measurement method and llama.cpp version not stated.
  - Prior notes judged these figures anomalous and low-confidence: 64 GB of RAM for a ~60 GB model suggests paging.
  - Source: [Hardware Corner](https://www.hardware-corner.net/gpt-oss-offloading-moe-layers/)
- The official llama.cpp gpt-oss guide (18 Aug 2025):
  - gives only full-VRAM gpt-oss-120b numbers: RTX PRO 6000 at 196.31 ± 0.14 tok/s (tg128, llama-bench) and Max-Q at 170.62 ± 0.47;
  - marks the <64 GB VRAM gpt-oss-120b section as "TODO".
  - Source: [llama.cpp Discussion #15396](https://github.com/ggml-org/llama.cpp/discussions/15396)
- A zenn.dev community test on an RTX 5090, in-VRAM model:
  - single-request FreeToken at 196.9 tok/s against llama.cpp Q4_K_M at 296.4, with llama.cpp flags undocumented;
  - it also reported gpt-oss-120b on FreeToken at 127.1 tok/s, but the offload setting was unclear (prior notes).
  - Source: [zenn.dev](https://zenn.dev/holy_fox/articles/53b82eed45f956?locale=en)
- A community thread comparing ik_llama.cpp, llama.cpp and ktransformers found "no direct, controlled comparisons" (prior notes). — [ubergarm/r1-ktransformers-guide #11](https://github.com/ubergarm/r1-ktransformers-guide/issues/11)

**Same-client practice outside MoE offload**
- Red Hat (15 Jun 2026, updated 13 Jul 2026) drove llama.cpp and vLLM with one client, GuideLLM, on one H200 with dense Llama 3.1 8B at 1–64 concurrent users.
  - It states it is "a production-serving comparison, not a general statement that one engine is better in every environment".
  - The main text gives no warm-up, CI, lengths or repetitions.
  - Source: [Red Hat Developer](https://developers.redhat.com/articles/2026/06/15/llamacpp-vs-vllm-choosing-right-local-llm-inference-engine)
- Agrawal et al. (arXiv 2507.09019, 11 Jul 2025):
  - AP1: "Conflating Implementation and Algorithm";
  - AP2: "Neglecting Parameter Tuning";
  - AP7: summary statistics only;
  - AP8: normalization "mask[s] … fixed or near-fixed overheads" (prior notes).
  - Source: [arXiv 2507.09019](https://arxiv.org/html/2507.09019)

### Inferences
- **Headline novelty.** The novelty is the measurement contract, not "same machine". It has three parts:
  1. A competitor's own client is the referee. That removes the "your harness disadvantages us" objection.
  2. Every system is tuned at equal expert memory, rather than FreeToken's routing-blind layer split, SeqMoE's static offload, or KTransformers' patched FP16 llama.cpp.
  3. Every system is reported against a hardware bound.

  I found no paper or community post that uses a competitor's unmodified benchmark script as the common client. That is search absence, not proof.
- **Consistency with published numbers.**
  - The study's 1.9–3.3× cache-vs-llama.cpp range is higher than moe-autopilot's +31.6% on the same GPU and CPU class. A reviewer will want the stock llama.cpp absolute tok/s at each budget, next to the ~36 tok/s at `--n-cpu-moe 25` community point, to show the baseline is not weak (AP2).
  - Budgets differ, so this is a sanity check, not a contradiction.
- **The +2–21% over FreeToken will be read against FreeToken's claimed 1.3–2.1× over its baselines.** The study's framing, that FreeToken beats a tuned llama.cpp but a tuned expert cache is at parity or better, directly revises FreeToken's baseline story. That is valuable, but it invites scrutiny of FreeToken's configuration. Report both FreeToken modes, their flags, and FreeToken's own `--moe-cache-*` settings at each budget.
- **Conflict of interest.** If the llama.cpp expert cache is the authors' own system, reviewers following AP1/AP2 will expect:
  - a disclosure;
  - an equal or larger tuning budget for competitors;
  - an ablation isolating the algorithmic contribution from engineering, e.g. cache off, same binary.

### Gaps
- FreeToken's paper text did not show whether one client timed all engines, which llama.cpp flags were used, or how VRAM was matched. Its appendix and repo were not checked.
- I found no r/LocalLLaMA thread with controlled, same-client gpt-oss-120b numbers across llama.cpp, ik_llama.cpp, KTransformers and FreeToken on a 24–32 GB GPU. Reddit search returned nothing usable.
- Whether the zenn.dev figure of 127.1 tok/s for gpt-oss-120b on FreeToken was an offloaded run on a 32 GB card is unclear.
- No PowerInfer, Ollama or LM Studio same-client MoE-offload comparison with a stated methodology was found.

## Q2. Is host-to-host variance on cloud GPU rentals for identical GPU models documented, especially variance driven by host DRAM bandwidth or PCIe link?

### Takeaway
Variance across nominally identical hardware is well documented in general:
- **Maricq et al., OSDI'18, CloudLab.** Memory-bandwidth CoV was 14.5–16% on one server type. An unbalanced DIMM configuration made newer servers about 3× slower than older ones in STREAM.
- **Sinha et al., SC'22, identical HPC GPUs.** Average variation was 8%, with a maximum of 22%.

I found **no published characterization of host-to-host variance on consumer-GPU marketplaces (Vast.ai, RunPod)** for the same GPU and CPU model, driven by host DRAM bandwidth. The study's result is 19–22% tok/s at 46.6 vs 62 GB/s DRAM, with 0.1–0.8% re-rent reproducibility. It is directly analogous to Maricq's DIMM-population finding, and appears new for rented consumer hosts and for offloaded LLM decode.

### Cited Findings
- **Maricq, Duplyakin et al., "Taming Performance Variability" (OSDI'18)**
  - Scale: 835 servers across three CloudLab clusters, 892,964 data points over 10 months (May 2017 – Apr 2018).
  - STREAM memory-test CoV of 14.5–16.0% on Clemson c6320 servers.
  - "an unbalanced DIMM configuration caused newer c220g2 servers to underperform older c220g1 servers by a factor of nearly 3 (about 36 GB/s versus 12 GB/s) in multi-threaded benchmarks."
  - Network latency CoV of 16.9–29.2%.
  - Source: [OSDI'18 PDF](https://users.soe.ucsc.edu/~carlosm/dev/publication/maricq-osdi-18/maricq-osdi-18.pdf); [USENIX page](https://www.usenix.org/conference/osdi18/presentation/maricq)
- Maricq et al., statistics:
  - "over 99% of the configurations (710 out of 713)" failed normality tests;
  - repetitions needed, per the CONFIRM tool: about 10 at 0.3% CoV and about 240 at 9.0% CoV;
  - including a single outlier server raised the repetitions needed by 2.1–5.9×;
  - excluding "from two to seven" servers, about 2% of the population, gave the largest reduction in dissimilarity.
  - Source: [OSDI'18 PDF](https://users.soe.ucsc.edu/~carlosm/dev/publication/maricq-osdi-18/maricq-osdi-18.pdf)
- **Sinha et al., "Not All GPUs Are Created Equal" (SC'22; arXiv 2208.11035, Aug 2022)**
  - Findings: "8% (max 22%) average performance variation even though the GPU architecture and vendor SKU are identical within each cluster, with outliers up to 1.5X slower than the median GPU".
  - Scope: Summit, Vortex, Frontera, Longhorn and Corona, with over 18,800 hours of data.
  - Causes: power management and manufacturing variability.
  - The artifact is published as SC AD/AE.
  - Source: [arXiv 2208.11035](https://arxiv.org/abs/2208.11035); [SC22 page](https://sc22.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap186.html); [artifact repo](https://github.com/hal-uw/gpu_variability_sc22_artifact)
- **Vast.ai documentation (host verification)**
  - Verification checks that "PCIe connections provide full bandwidth and are not throttled".
  - It warns against pairing high-end GPUs with under-provisioned CPU/RAM.
  - DLPerf is "estimated GPU performance on typical deep-learning tasks". Raising it requires fixing "PCIe/thermal bottlenecks; maintain clocks; correct drivers".
  - This implicitly acknowledges that the host shapes the performance of an identical GPU SKU. No quantified variance is given.
  - Source: [Vast.ai docs, Understanding Verification](https://docs.vast.ai/documentation/host/understanding-verification)
- **Vast offers API (prior notes, snapshot of 27 Sep 2026)**
  - Listings expose CPU model, RAM, motherboard, PCIe generation and lanes, and measured `pcie_bw`: a 47.8 GB/s median on desktop RTX 5090 hosts and 22.6 GB/s on 4090 hosts.
  - "No provider exposes DIMM count, channel count or memory speed."
  - Source: [Vast API](https://console.vast.ai/api/v0/bundles/); see `research_notes/MoE audit measurement plan/cloud_hardware.md`
- **FreeToken's own two RTX 5090 hosts** differ in host expert-processing bandwidth: B_H = 77.3 GB/s (Xeon server) against 53.8 GB/s (9950X3D desktop). These are different CPUs, so this is not a same-SKU lottery (prior notes). — [arXiv 2608.16157 HTML](https://arxiv.org/html/2608.16157)
- **Cloud VM STREAM** reached only 46–59% of datasheet on the project's own anchor hosts (prior notes). — `paper/paper.tex` (project-internal)

### Inferences
- **Frame the host lottery with Maricq as the primary precedent.** Same CPU SKU, different DIMM population, large bandwidth gap: that is the same mechanism. Sinha shows the problem for GPUs.
- **The study's contribution:**
  1. It is on rented consumer hosts, where the renter cannot see DIMM count or speed before renting.
  2. The effect is on end-to-end offloaded decode, so the host DRAM tier matters directly.
  3. The fix is paired bit-identical workloads. Relative comparisons stay valid across hosts even when absolute tok/s moves 19–22%.
- **Two hosts is a small N for a variance claim.** Maricq needed hundreds of servers to characterize distributions. A reviewer at SC or IISWC will ask for more:
  - at least 5–10 distinct rentals of the same GPU+CPU model;
  - STREAM plus measured pinned H2D per host;
  - `dmidecode` DIMM info where visible;
  - the CoV of tok/s across hosts;
  - a regression of tok/s on measured DRAM bandwidth.

  The within-host reproducibility of 0.1–0.8% on re-rent is strong evidence that the between-host variance is real. Report it next to the cross-host numbers.
- **PCIe link generation and width.** Vast exposes these. Record `nvidia-smi -q` link generation and width, plus measured H2D, per host, to rule out PCIe as a confound. Maricq's lesson is to "match hardware and software" and to characterize every host.

### Gaps
- I found no peer-reviewed or vendor study that quantifies performance variance across Vast.ai, RunPod, TensorDock or similar hosts with identical GPUs. That is search absence (two queries).
- Cloud-IaaS variability classics (e.g., Leitner & Cito on public-IaaS variation; Uta et al. NSDI'20 on cloud reproducibility) were not fetched this session and are not cited.
- Sinha et al.'s split between memory-bound and compute-bound workloads was not in the abstract, and the PDF was not read.

## Q3. What statistical practice do benchmarking guides require (Hoefler & Belli, Georges, Mytkowicz, ACM badges, MLPerf, MLSys)? Repetitions, CIs, variability, paired designs, warm-up.

### Takeaway
The guides converge on five requirements:
1. Many independent repetitions at every level where variance enters.
2. Nonparametric confidence intervals, since performance data is almost never normal.
3. Harmonic means for rates, and no averaged ratios.
4. Discarding warm-up.
5. Randomizing the setup and run order to defeat hidden bias.

The study currently has one warm-up request and one measured request per configuration, from a single client, over AIME-25 prompts. It needs CIs over prompts and over server launches, and randomized interleaving. A reviewer at SC (the venue of Hoefler and Belli) will treat missing CIs as disqualifying.

### Cited Findings
- **Hoefler & Belli (SC'15)** (verbatim rules in prior notes):
  - R3: the harmonic mean for rates;
  - R4: avoid summarizing ratios;
  - R5: CIs for nondeterministic data;
  - R6: do not assume normality;
  - R7: compare via non-overlapping CIs or ANOVA;
  - R9: document all factors;
  - R11: show upper bounds;
  - "n > 5 measurements are needed to assess confidence intervals nonparametrically";
  - adaptive stopping: recompute the CI after each batch of k runs;
  - "The first measurement iteration should be excluded";
  - "If controlling a certain parameter is not possible then we suggest randomization".
  - Only 2 of 95 surveyed HPC papers reported CIs.
  - Source: [SC'15 PDF](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf); [ACM DL](https://dl.acm.org/doi/10.1145/2807591.2807644)
- **Georges, Buytaert, Eeckhout, "Statistically Rigorous Java Performance Evaluation" (OOPSLA'07)**
  - Method:
    - run multiple independent VM invocations and discard the first;
    - build CIs with Student's t for n < 30 and z for n ≥ 30;
    - for steady state, take multiple iterations per invocation and start measuring once the CoV over k consecutive iterations falls below 0.01–0.02;
    - compute CIs across per-invocation means, which keeps the samples independent.
  - Comparisons: a CI on the difference of means for two systems; ANOVA with Tukey HSD for more than two.
  - Evidence against common practice:
    - "Best of N" gave misleading conclusions in up to 16% of comparisons;
    - common steady-state practice misled in more than 20% of cases at θ = 1%.
  - Their JavaStats tool keeps adding runs until the CI half-width reaches a target, e.g. 2–3%.
  - Source: [OOPSLA'07 PDF](https://dri.es/files/oopsla07-georges.pdf)
- **Mytkowicz, Diwan, Hauswirth, Sweeney, "Producing Wrong Data Without Doing Anything Obviously Wrong!" (ASPLOS'09)**
  - Link order alone moved perlbench's O2-vs-O3 speed-up across 0.92–1.10.
  - UNIX environment size moved lbm across 0.88–1.09.
  - Bias was "commonplace" across 12 SPEC CPU2006 benchmarks, 3 microarchitectures and 2 compilers, with a "7% variation in speedup between different experimental setups".
  - Remedies: setup randomization over many setups (484 combinations demonstrated), causal analysis by intervention, and diverse workloads.
  - Source: [ASPLOS'09 PDF](https://users.cs.northwestern.edu/~robby/courses/322-2013-spring/mytkowicz-wrong-data.pdf)
- **Maricq et al. (OSDI'18)**
  - "Use nonparametric confidence intervals to avoid assumptions of normality";
  - "Perform enough repetitions to achieve tight confidence intervals";
  - "Randomize experiment orderings";
  - "Base experiment design on past measurements".
  - The CONFIRM tool estimates the repetitions needed.
  - Source: [OSDI'18 PDF](https://users.soe.ucsc.edu/~carlosm/dev/publication/maricq-osdi-18/maricq-osdi-18.pdf)
- **Kalibera & Jones (ISMM'13)**: repeat at every level (build, process, iteration), size each level by a dimensioning experiment, and put CIs on speed-up ratios by bootstrap or Fieller's method (prior notes, secondary summary). — [ACM DL](https://dl.acm.org/doi/10.1145/2464157.2464160)
- **MLPerf Client (page modified 18 Aug 2026)** (prior notes):
  - "one warm-up run and three performance runs", reporting averages;
  - TPS is "the average rate to produce all of the rest of the tokens in the response, excluding the first one";
  - the study's (completion_tokens − 1)/(t_last − t_first) matches this definition;
  - no MoE models and no offload rules.
  - Source: [MLPerf Client](https://mlcommons.org/benchmarks/client/)
- **MLPerf Inference** (prior notes):
  - SingleStream requires at least 600 s and an early-stopping confidence bound;
  - "Truncating output tokens … is not permitted";
  - tokens per sample must lie within 90–110% of the reference;
  - "Results that cannot be replicated are not valid results".
  - Source: [MLPerf Inference Rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
- **ACM badging** (prior notes): "Results Reproduced" uses the authors' artifacts, and "Results Replicated" does not. — [ACM Artifact Review and Badging](https://www.acm.org/publications/policies/artifact-review-and-badging-current)
- **Venue policies** (prior notes):
  - MLSys 2027 AE is voluntary and does not affect acceptance. — [MLSys 2027 CFP](https://mlsys.org/Conferences/2027/CallForResearchPapers)
  - ISPASS 2026 solicits "Confirmations or refutations of important prior results". — [ISPASS 2026 CFP](https://ispass.org/ispass2026/cfp.php)
  - SC papers ship AD/AE artifacts; see the SC'22 variability paper's "AD/AE for SC 22 submission" repo. — [hal-uw artifact](https://github.com/hal-uw/gpu_variability_sc22_artifact)
- **llama-bench** defaults to `reps = 5` with warm-up and reports the arithmetic mean of per-rep rates, plus the raw `samples_ns` (prior notes). — [llama-bench.cpp](https://github.com/ggml-org/llama.cpp/blob/4d86b2f5e8a04667337e20bd5051570be4554cfe/tools/llama-bench/llama-bench.cpp)
- **Agrawal et al.** AP7 and AP8 ask for CDFs or P50/P90/P99 of time-between-tokens, not just mean TPOT (prior notes). — [arXiv 2507.09019](https://arxiv.org/html/2507.09019)

### Inferences

**Concrete reviewer demands (a checklist to add):**

1. **Repetitions and CIs.** Each (system, budget, host) cell needs two levels of repetition:
   - at least 3 independent server launches (Georges: invocations; Kalibera–Jones: multi-level);
   - within each launch, measured requests over at least 10 distinct AIME-25 prompts after the warm-up.

   Report the median and a nonparametric 95% CI (Hoefler R5–R6; Maricq). Stop adaptively when the CI half-width is at most 2–3% (Hoefler §3.1.3; Georges' JavaStats). The study's 0.1–0.8% re-rent spread suggests this will be cheap.
2. **Aggregation.**
   - Summarize tok/s across prompts as total tokens over total time, or as a harmonic mean (R3). Do not use an arithmetic mean of per-request tok/s, which is what FreeToken reports.
   - Give speed-ups as ratios of aggregated rates, with a paired-bootstrap CI over prompts (R4; Kalibera–Jones).
3. **Paired design.** Bit-identical generated text across hosts and seeds is a real strength: it gives a within-prompt paired comparison. Analyse it that way, with per-prompt ratios and a Wilcoxon signed-rank test or paired bootstrap. Say explicitly that pairing holds across hosts only for the same engine and seed.

   Across engines the texts differ unless `ignore_eos` and the fixed seed happen to align. The prior notes found that bitwise agreement across engines is rare in BF16. So cross-engine comparisons are paired by prompt, not by token stream.
4. **Warm-up.**
   - Keep the discarded warm-up (Hoefler §4.1; MLPerf Client).
   - For the dynamic expert caches (FreeToken, the expert-cache llama.cpp), also report cold first-request and steady-state numbers. Warm-up changes cache state, and the reported value "may or may not be representative".
5. **Randomization.** Interleave and randomize system order across launches (Mytkowicz; Maricq; Hoefler §4.1), and log time and temperature. The study's CPU is a shared-DRAM desktop part, so run order and background load matter.
6. **Variability reporting.** Show per-token inter-token-latency CDFs or P50/P90/P99 per system (AP7/AP8). Stalls from expert misses are exactly what a mean hides.
7. **Measurement validation.**
   - The 1.3% agreement with FreeToken's unmodified script is good. Also cross-check against server-side timings, such as llama-server `timings.predicted_per_second`.
   - Confirm that each engine streams one token per SSE event. If an engine coalesces tokens, t_first and t_last shift. That is an engine-dependent client artifact that a reviewer may raise.
8. **Artifact.**
   - Pin all commits, including FreeToken `0d652e7` and the llama.cpp build.
   - Publish raw per-request logs and the analysis script, and state the host characterization (STREAM, H2D, lscpu, dmidecode).
   - Aim for ACM Available/Functional. "Reproduced" requires a third party.

### Gaps
- I found no MLSys-specific statistical guideline beyond voluntary AE. No IISWC 2026/2027 CFP text was fetched, so IISWC-specific expectations are inferred from its workload-characterization scope, not quoted.
- The current SC26/SC27 AD/AE policy text (mandatory or optional; reproducibility-initiative wording) was not fetched this session.
- MLPerf Client's exact repetition and variance rules beyond "one warm-up and three performance runs" were not re-verified.

## Q4. Has anyone reported offloaded-MoE systems, or LLM decode on consumer hardware generally, as a fraction of a hardware bound? What fractions?

### Takeaway
Fraction-of-bound reporting is established for in-memory decode:
- Databricks' MBU: about 50–60% at batch 1 on A100/H100.
- llama.cpp full-VRAM decode on consumer GPUs: roughly 26–70% of datasheet bandwidth, by the prior notes' computation.
- llama.cpp CPU decode: about 30–63% of STREAM.

For offloaded MoE, I found no paper that reports end-to-end decode as a fraction of a hardware bound. The nearest precedents are:
- SeqMoE's "80.22% of full-load performance", an empirical ceiling, not a hardware bound;
- the MoE-CAP S-MBU metric;
- rooflines used internally for policy search (MoE-Lightning HRM, Pipelined Sharding).

The study's 9–34% of a measured-bandwidth time-domain speed-of-light is therefore a new, and strikingly low, headline. Because the bound uses measured, not datasheet, bandwidths, these fractions would be even lower on a datasheet (MBU-style) denominator.

### Cited Findings
- **Databricks, "LLM Inference Performance Engineering: Best Practices" (12 Oct 2023)**
  - Definition: "MBU is defined as (achieved memory bandwidth) / (peak memory bandwidth) where achieved memory bandwidth is ((total model parameter size + KV cache size) / TPOT)."
  - Reported: about 55% MBU for Llama2-70B on 4× A100-40GB and about 60% on 2× H100-80GB at batch 1, using TensorRT-LLM with 512-token inputs. The MPT-7B 50% figure is illustrative.
  - Source: [Databricks blog](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
- **Prior-note computation (tok/s × weight bytes per token ÷ datasheet bandwidth)** on public llama.cpp numbers:
  - Llama-2-7B Q4_0: 70% on an RTX 4090 and 63% on an RTX 5090;
  - gpt-oss-20b: 41–70%, depending on GPU and kernel fusion;
  - gpt-oss-120b on an RTX PRO 6000: 39%;
  - Qwen3-30B-A3B on an RTX 5090: 26% → 38% (unfused → fused).
  - This is the project's own derivation, not a published figure. — `research_notes/MoE offload system race plan/headroom_techniques.md`, from [llama.cpp Discussion #15396](https://github.com/ggml-org/llama.cpp/discussions/15396) and related discussions
- **llama.cpp CPU decode on Neoverse-N2** reached about 30–63% of STREAM Triad (435 GB/s). The issue asserts x86 reaches 85–95%, which is unverified (prior notes). — [llama.cpp Issue #25976](https://github.com/ggml-org/llama.cpp/issues/25976)
- **SeqMoE** abstract: "With 45% expert residency, SeqMoE averages a 96.97% hit rate and 80.22% of full-load performance". The normalizer is an empirical all-in-VRAM run on vLLM (prior notes). — [arXiv 2609.12978](https://arxiv.org/abs/2609.12978)
- **MoE-CAP (NeurIPS 2025)** defines sparsity-aware S-MBU and S-MFU. It uses a single B_peak and targets vLLM, SGLang, MoE-Infinity and KTransformers (prior notes). — [arXiv 2412.07067](https://arxiv.org/abs/2412.07067)
- **Rooflines used only to choose policies** (prior notes):
  - MoE-Lightning's Hierarchical Roofline Model, [arXiv 2411.11217](https://arxiv.org/abs/2411.11217);
  - MoE-Lens's "theoretical performance upper bound" with 94% prediction accuracy, [arXiv 2504.09345](https://arxiv.org/abs/2504.09345);
  - Pipelined Sharding's profiler roofline, [arXiv 2604.26334](https://arxiv.org/html/2604.26334).
  - KTransformers and FreeToken report tok/s and speed-ups only. FreeToken notes only that prefill hits the "practical ceiling of the PCIe 5.0 x16 link".
- **Hoefler R11**: "If possible, show upper performance bounds". §2.1.1 adds that speed-up "can almost always be replaced by lower bounds on execution time" (prior notes). — [SC'15 PDF](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf)
- **FlashAttention-2** reports "50-73% of the theoretical maximum FLOPs/s on A100" (prior notes). This is a canonical ML-systems use of fraction-of-peak. — [arXiv 2307.08691](https://arxiv.org/abs/2307.08691)

### Inferences
- **Denominators must be stated.** 9–34% of a *measured-bandwidth, time-domain* bound is not comparable with MBU (datasheet peak, in-memory) or with fractions of STREAM. Present a small translation table: SoL from measured bandwidth, SoL from datasheet bandwidth, and the resulting fractions. Prior notes found the non-comparability of "% of bandwidth" across sources to be a real trap.
- **Reviewer asks on the bound:**
  1. A derivation and a sensitivity analysis: how the fraction moves with ±10% in measured DRAM, H2D or VRAM bandwidth.
  2. Evidence that the bound is attainable in principle, e.g. microbenchmarks that reach the per-tier bandwidths inside the same process.
  3. A breakdown of the gap: CPU expert GEMV efficiency, PCIe underuse, synchronization and launch overhead, and cache misses.

  The fact that every system sits at 9–34% becomes the paper's thesis only if the gap is decomposed.
- **The low fractions are credible against outside data.** KTransformers' own profiling found that kernel-launch overheads made up 21% (llama.cpp) to 73% (Fiddler) of GPU time in hybrid decode (prior notes, [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)). Overheads of that size make low end-to-end fractions plausible.

### Gaps
- No published fraction-of-bound figure for offloaded or hybrid MoE decode on consumer GPUs was found. This is search absence across the named papers plus prior notes' searches, not an exhaustive scan.
- No published MBU-style number specifically for gpt-oss-120b with CPU-offloaded experts was found.

## Q5. Do workload choices (sampled reasoning prompts vs teacher-forced text; prompt length) change measured decode speed in published offloading work?

### Takeaway
Published offload papers use very different workloads:
- KTransformers: 32-token prompts, 512 output tokens.
- Pipelined Sharding: 1K–64K context, 100 output tokens.
- HybriMoE: 32–1024-token inputs.
- FreeToken: agentic trajectories up to 65k tokens.

Only FreeToken states a workload-sensitivity result: decode stayed "within 12%" across its W1–W4 workloads. Context length shifts decode through KV and attention cost, and routing locality through the cache. I found no published quantification of sampled or on-policy versus teacher-forced text on measured offload decode speed. The trace-level literature (2608.07911) shows that workload construction and chat templates can reverse cache conclusions.

### Cited Findings
- **FreeToken**: "Decode rate stayed within 12% across single and multi-turn agentic workloads". The workloads span AIME CoT to 56–65k-token Claude Code sessions. — [arXiv 2608.16157 HTML](https://arxiv.org/html/2608.16157v1)
- **KTransformers**: fixed 32-token prompt and a maximum of 512 decoded tokens (prior notes). — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- **Pipelined Sharding**: 100 output tokens at 1K–64K contexts (prior notes). — [arXiv 2604.26334 HTML](https://arxiv.org/html/2604.26334)
- **HybriMoE**: inputs of about 32–1024 tokens on MT Bench, Vicuna Bench and ChatGPT Prompts, at cache ratios of 25/50/75% (prior notes). — [arXiv 2504.05897 HTML](https://arxiv.org/html/2504.05897)
- **Hardware Corner**: gpt-oss-120b generation on an RTX 5090 fell from about 9.6 to about 8.1 tok/s across 85–5650-token contexts. The absolute values are low-confidence (see Q1). — [Hardware Corner](https://www.hardware-corner.net/gpt-oss-offloading-moe-layers/)
- **moe-autopilot** states a break-even prompt:output ratio of about 2.3, because its hot-expert scheme costs about 10% prefill throughput. Workload mix therefore flips whether a scheme wins end to end (prior notes). — [moe-autopilot](https://github.com/JigSawPT/moe-autopilot)
- **FreeToken's LRU miss rate on Qwen3.6** (16% vs 41% for KTransformers and 62% for llama.cpp) is workload-dependent by construction. It is a cache statistic from its agentic traces. — [arXiv 2608.16157 HTML](https://arxiv.org/html/2608.16157v1)
- **2608.07911 (Aug 2026)**:
  - says workload construction and chat-template choices can reverse cache conclusions;
  - requires reporting of "per-step expert union relative to per-layer capacity";
  - says "normalized miss fractions do not transfer across models".
  - Source (prior notes): [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)
- **SeqMoE** collects on-policy traces: it generates token IDs, then replays them as a prefill to dump routing. It does not quantify the difference from dataset text. Prior notes found **no paper quantifying teacher-forced vs on-policy effects** on expert locality or offload speed. — [arXiv 2609.12978](https://arxiv.org/abs/2609.12978); `research_notes/MoE research direction rethink/benchmarking_and_offpolicy.md`
- **MLPerf Inference** fixes min and max new tokens per model and requires 90–110% of the reference tokens per sample (prior notes). — [MLPerf Inference Rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)

### Inferences
- **Reviewers will ask how general 256 decode tokens on AIME-25 is.** AIME reasoning traces run thousands of tokens, so a 256-token window measures only early decode, at short KV. Add a sweep:
  - decode length 256 / 1024 / 4096, or llama-bench-style fixed context depths;
  - one non-reasoning workload, such as chat or code, to show whether the ranking and fraction-of-SoL hold;
  - FreeToken's "within 12%" is the bar to match or refute.
- **Sampled vs greedy vs teacher-forced.** The study's sampled, fixed-seed, on-policy generation is the deployment-realistic choice, and it enables bit-identical pairing across hosts. A teacher-forced replay of the same tokens would be a cheap ablation: it shows whether routing locality, and so the cache advantage, depends on the text being self-generated. No published number exists to compare against, so this would be new, if small.
- **Report per-step expert union relative to capacity at each budget (11/25/40/44%)**, in the spirit of 2608.07911. The cache advantage then becomes interpretable across models. This also partly answers the "second model" request.
- **Second model.** Add Qwen3.6-35B-A3B (FreeToken's own model, which FreeToken reports at 77–83 tok/s on a 5090) or Qwen3-30B-A3B. Either anchors the study to published numbers and shows that the gpt-oss-120b findings generalize. Accepted top-venue offload papers typically use about 3 models and 2–3 platforms (prior notes, `venue_bar.md`).

### Gaps
- None of the named papers reports a sensitivity study of decode tok/s to prompt type (math vs code vs chat) at a fixed context length, beyond FreeToken's 12% statement.
- No published data on gpt-oss-120b routing locality under different reasoning-effort settings or harmony-format prompts was found.
- Whether FreeToken's 12% figure holds under its own `bench_decode_moe.py` (single-turn AIME) versus its agent harnesses was not checked.
