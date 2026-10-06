# Review 8b: "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth"

**Calibration.** MLSys main track. Overall 1 to 10: 5 = borderline reject, 6 = weak accept, 7 = accept, 8 = strong accept, 9 = top 5% of accepted papers, 10 = best-paper level. Sub-scores 1 to 5. I wrote this as a demanding reviewer from a performance-modelling and benchmarking group, and I did not start from generosity.

**Independence.** I read the paper PDF (31 pages, appendices included) and the supplement PDF as text, the scripts, `mosl/` and `tests/` in the repository, the `prereg/*.json` files and `prereg/PROTOCOL.md`, and the job scripts and raw results on the GPU branch. I did not open anything under `reports/`, any `prereg/*outcome*.md`, or any file named like a review, plan or progress log. I read no commit messages: I used only `git log --format='%h %ad'` and `git diff` of job-script content to establish ordering. I modified nothing in either repository except writing this file. All reruns happened on copies in a scratch directory.

## 1. Summary

The paper studies batch-1 decode of two MoE models (gpt-oss-120b MXFP4 and Qwen3-30B-A3B BF16) on RTX 5090 desktop hosts when most experts live in host DRAM. It makes three claims.

1. **A bound.** It derives a lower bound on time per token for any exact-routing system with C GPU slots per layer: the maximum of a GPU term and a host term, where the host term is the reads of Belady's MIN with bypass divided by the highest probed host read rate. The authors' llama.cpp expert cache reaches 25 to 43% of this bound and leads FreeToken at 11 of 12 cells. Scored by one shared procedure, published systems "in the class" reach a median of 13.6%.
   - A new microbenchmark (job 102) replays MIN's per-layer read counts through CPU threads and the copy engine with no model. It reaches 86 to 96% of the host term.
   - An analytic "read-once cap" argues that where the link is slower than the CPU, MIN's own one-read schedule cannot get near the bound.
2. **In-engine oracles.** Oracles run inside the engine cross *what to cache* (the deployed policy or MIN) with *how to load* (CPU then background copy, or a copy in the step). The design runs on two hosts at six budgets and on 13 more hosts at the two host-bound gpt-oss budgets.
   - MIN loaded once is 16 to 51% faster than the deployed cache on three hosts, and 25 to 81% faster when its copies are prefetched.
   - The two choices interact strongly. A Shapley split credits each choice, and a calibrated two-path time model predicts in-step states within a median 2.2%.
   - The copy in the step loses only on the one machine whose link reads at 0.29 of its CPU rate.
3. **A price on foresight.** Windows of W future tokens (exact or noisy) are measured in the engine and on traces of nine models. The window that closes half of the read gap holds about 0.65·C distinct experts per layer, and no online policy, learned order, speculative batch or forecaster tested gets close.

The work comes with an unusually complete artifact: predictions committed before each rental, raw rows, generated macros, and a 565-plus-547-clause scorecard.

## 2. Strengths

- **S1. The core object is correct.** I recomputed R\* (MIN-with-bypass host reads) for gpt-oss at C = 14 and 32 as an exact interval LP, independent of `mosl.cachesim`. It matches the repository's greedy simulation to the read on every layer (38.315 and 15.340 reads per token). With host B's probe, the bound comes out at 172.3 and 430.4 tok/s, as in Table 2. The bound is a valid probe-relative lower bound for the stated class.
- **S2. Measurement discipline is well above the MLSys norm.**
  - Configurations run in seeded random order with cold caches, and the variant used for each system is selected on a separate launch.
  - Comparisons are paired by problem with bootstrap intervals, and absolute speeds sit next to every ratio.
  - Each host's bandwidth is probed in-job. Relaunches are on the same physical GPUs; I checked the GPU UUIDs.
  - Every number I recomputed from raw rows matched to the printed digit (Section 6).
- **S3. The host is treated as the unit of variation, and the evidence for that is shown.** Two hosts with the same CPU differ by 16 to 29% in deployed time, while relaunches stay within 1.9%. Few systems papers do this.
- **S4. The in-engine oracle factorial is the paper's most original contribution.** It measures in time, in a real engine, what trace studies have only measured in hit rates. The finding that what to cache and how to load are strong complements, rather than additive, is useful to designers. Within one host I estimate the interaction at 43.2% of the gap with a 95% interval of [42.4, 44.0].
- **S5. Negative results and failed predictions are reported, not buried.**
  - 142 of 565 clauses failed, plus 39, 27 and 5 in the machine-scored jobs.
  - The time model's systematic over-prediction of background states is stated (64 of 64 too fast).
  - The authors say the "copy pays" sign test is weak because the copy loses on only one machine.
  - The class line at a link-to-CPU ratio of 0.5 is admitted to be post hoc.
- **S6. The "horizon in distinct experts" framing is a good idea.** It ties the empirical W50 to paging-with-lookahead theory and gives forecaster designers a target in units they control.
- **S7. The artifact runs.** The 15 unit tests pass, and `scripts/hostdep_model.py` rerun on a copy regenerates its macro file byte-for-byte.

## 3. Weaknesses (ranked by how much they limit the claims)

### W1. The read-once cap, and the slow-link story built on it, rest on one arbitrary choice among many optimal schedules. (New finding; the most serious problem.)

**What the paper says.**
- Section 3: "admissions are 57% of MIN's reads at 11% (69% at 25%)". This caps MIN's one-read schedule at 53 to 71% (11%) and 43 to 58% (25%) of the host term on slow-link hosts.
- The bound is therefore "loose for MIN's own schedule" where the link is slow.
- Section 4: on slow-link hosts "much of the rest is the link itself", because no schedule that admits what MIN admits can close 25 to 54% of the gap.

**Why that does not hold.** The 57% and 69% are properties of *one* hit-optimal schedule: the eager greedy implementation in `mosl.cachesim`. Hit-optimal caching with bypass has many optimal solutions, and they differ greatly in how many admissions they make.

**How I checked.**
- Per layer, I set up the interval formulation: x_j ∈ [0,1] holds an expert from one use to its next, and a_j ≥ x_j − x_prev(j) counts the start of a residency chain, i.e. an admission.
- Constraints: at most C held intervals across every step boundary.
- Objective: minimise misses, then admissions (Σ(1−x_j) + 10⁻⁴·Σa_j).
- On three layers HiGHS MILP gives the same integral optimum as the LP. Exhaustive search on 40 random small instances matches the LP exactly.

**Result.**

| gpt-oss | Reads (R\*) | Greedy admissions | Min-admission optimum |
|---|---|---|---|
| C = 14 | 38.32 / token | 21.77 / token (57% of reads) | 12.40 / token (32%) |
| C = 32 | 15.34 / token | 10.62 / token (69%) | 6.75 / token (44%) |

On layer 0, for example, greedy makes 7,006 admissions and the optimum 3,572, for an identical 14,449 misses.

**Consequences.** I computed these with the paper's own `readsched.json` rates and `factorial_shapley.json` gaps.
- **The cap almost disappears.** With the min-admission schedule, the read-once cap at 11% is 0.93 to 0.94 on the 0.29-ratio machine and 1.00 on every other host. At 25% it is 0.68 to 0.69 on the 0.29 machine, 0.89 to 0.92 on the 0.40 to 0.41 hosts, and 1.00 elsewhere.
- **A link-aware bound barely moves.** I also computed a valid tighter bound that minimises over the whole misses-versus-admissions frontier (μ swept from 10⁻⁴ to 4), with admissions forced over the link at its highest probed rate. At 11% it is within 1% of the paper's bound on every host of the panel and follow-ups. At 25% it is at most 7% above the paper's bound, on the 0.29 machine.
- **So the bound is not loose for the link constraint.** What is "loose" is the particular schedule the paper chose.
- **The slow-link share of the gap shrinks.** The share of the gap the cap explains on slow-link hosts drops from 25 to 54% to 0 to 13%.
- **The engine oracle inherits the problem.** The engine's "MIN, 1 read" oracle fetches *the misses MIN would admit* (Appendix F), i.e. greedy's eager admissions. Its loss on the 0.29 host (0.86 to 0.875×), and the conclusion that "the copy in the step needs a link not much slower than the CPU", may be partly self-inflicted.
- **The factorial's cache factor is confounded.** "MIN's set" is not a well-defined level, because which optimal set you pick changes the double-read penalty in the 2-read state by about 40% of admissions. The 43-point interaction and the Shapley split depend on that arbitrary choice. Loaded twice, the min-admission set reads 50.7 experts per token on the trace, against 60.1 for greedy MIN.

**Fix.**
1. Report the misses-versus-admissions frontier, and replace the cap with the link-aware bound. A parametric min-cost-flow or LP solves in minutes per model.
2. Add a "minimum-admission MIN" oracle to the engine. Rerun it on the slow-link machines (Pf, Pd, the x8 9950X, the slower-link 9800X3D) and two fast ones, at both host-bound gpt-oss budgets.
3. Restate finding (2) and the "rest" decomposition according to what that shows. My prediction, registered here: the in-step single read stops losing on the 0.29 machine at 11%.

### W2. Attainability is shown for read volume, not for a schedule that runs the model.

Job 102 is a careful microbenchmark:
- 4 GiB rotating buffers and whole 13.25 MB experts;
- a barrier per layer;
- two repetitions within 0.9%;
- results that reproduce exactly from `readsched_C*.txt` and `concur.txt`.

But what it shows is narrow.
- **No compute anywhere.** Its CPU threads only XOR-reduce, which is not an MXFP4 GEMV; the paper's own measurement puts helpers at 1.09 to 1.31× the probe's time per expert. No GPU kernels run concurrently, and no admission constraint is enforced.
- **Its best mode on the slow-link host is not a valid cache schedule.** On Pd the CPU-only mode (92%) beats the per-layer split (86 to 89%), yet reading every one of MIN's reads on the CPU never fills a slot.
- **It mostly reproduces its own probe.** The denominator B_host is the maximum over about 20 samples of the same kind of read loop that the benchmark runs. Showing that "token mode reaches 93 to 97%" largely shows that whole-expert granularity and layer barriers cost little relative to that probe.
- **The registered thresholds were weak.** The token mode was predicted at ≥0.80 when the authors' own analysis already gave 0.85 to 0.99. The one informative clause (≥0.03 below the analytic per-layer fraction) failed on 2 of 3 hosts.
- **The GPU side is open.** The GPU term uses the datasheet rate, which all-in-VRAM decode reaches only 52 to 61% of, and overlap of GPU work with host reads is never measured.

**Fix.** Extend `readsched.cu` with three modes:
- (a) min-admission MIN's admissions forced over the link;
- (b) helpers running the engine's actual expert kernel on the bytes they read;
- (c) GPU kernels sized to the measured all-in-VRAM step running concurrently.

Report the fraction of the bound for (a+b+c). That is the attainability statement a reader needs.

### W3. External validity is narrow, and the machine-dependence claim rests on one or two machines.

- **Workload and hardware.** Two models; AIME-25 prompts (also used during development); 256 tokens; teacher-forced in the oracle jobs. One GPU type for all oracle work, on a convenience sample of Vast desktop rentals.
- **The slow-link class.** It contains two panel hosts, and the copy-in-step crossover rests on one machine (measured twice). Without that machine, the correlation of Fig. 2 at 11% falls from 0.93 to 0.87 (my recomputation with one point per machine), and the crossover itself is unobserved.
- **The class boundary was drawn after the panel ran.** Qwen3 ran on only 3 panel hosts at one budget and not at 25%.

**Fix.**
- Rent at least 5 more hosts with link-to-CPU ratio below 0.5, pre-registering the class boundary and the predicted sign per host.
- Run Qwen3 at 12.5% and 25% on the full panel.
- Add one non-math workload (chat or code) with 2k-token outputs on 5 hosts.

### W4. Host-level statistics are thin for the strength of the wording.

- **The pooled interval is too narrow.** With 10 panel hosts, the percentile crossed bootstrap gives [1.12, 1.30] for MIN-one-read's gain at gpt-oss 11%; I reproduced [1.119, 1.305]. A t-interval on the 10 per-host ratios is [1.11, 1.33]. With n = 10 and a left outlier (0.875), the percentile bootstrap under-covers.
- **It averages ratios against the paper's own checklist.** The pooled statistic is an arithmetic mean of per-host ratios, while Appendix A claims compliance with "do not summarise ratios". A geometric mean, or a mixed model on log time, is the right tool.
- **"Every host of a stated class" is weak evidence.** With 8 hosts all agreeing, the 95% upper bound on the share of hosts that would disagree is about 31%.
- **Within-host intervals use a single launch.** A relaunch moves ratios by up to 0.030 (I reproduced this), more than three times the median within-launch half-width of 0.009. The paper says so, but every Table 3 and Table 4 entry is still a single-launch number.

**Fix.**
- Report the across-host prediction interval (what a new rented host will see), not just a CI of the mean.
- Fit a random-effects model (host, launch nested in host, problem crossed) and report the variance components.
- Relaunch every panel host at least once. That costs about $10.

### W5. The time model's headline accuracy is on the metric that flatters it.

The 2.2% median (I reproduced the macros exactly) is error on *absolute* time per token. One fitted constant G makes up about 28% of the deployed time.
- **On the in-step states it is genuinely good.** In my recomputation, a frozen one-path model gives 7.1% and a null model ("other hosts' median ratio × this host's deployed time") 9.0%. The two-path model therefore has real content there, which the paper undersells; Appendix C's blind test ties the one-path model.
- **On effect sizes it is much weaker.** For the decision the model is used to justify, the gain of the in-step copy, the frozen model's predicted gain divided by the measured gain has median 1.19 and IQR 1.01 to 1.44 across 38 host-cells.
- **Other gaps.** It mispredicts all 64 background or prefetch states in the same direction (median 16%). With constants frozen, it over-predicts Qwen3 by 5 to 21%, and it fails on the server hosts.

**Fix.**
- Report error on gains, i.e. (1 − t_state/t_deployed), and the regret of the decisions it drives (fetch table, copy or not).
- Model background-copy contention, or drop the claim that the model "predicts every state".

### W6. The order-free accounting is mostly convention.

- **The split is not meaningful at this interaction size.** When the interaction is 43 points of a 52-point joint effect (O4, 11%), the Shapley values are an equal split of the interaction, not a finding. The paper says this, but then leads with the Shapley columns.
- **The rest is a residual.** It is not measured, and it absorbs GPU kernels at half the datasheet rate, partial overlap, prefetch imprecision and any looseness in the bound.
- **No uncertainty is shown.** Within-host intervals are tiny (about ±1 point) but the host-to-host spread is large (O4 against O5: Shapley "cache" 26 against 12; rest 36 against 57).
- **"MIN's set" is not unique (W1).** So the cache factor itself is ill-defined.

**Fix.**
- Present the 2×2 cell means with intervals, and make the interaction the primary quantity.
- Fix the cache factor to a well-defined schedule, either minimum-admission or time-optimal.
- Replace "rest" with measured components: Nsight GPU-busy time, measured overlap, and prefetch bytes over R\*.

### W7. The horizon rule is weaker than presented.

- **"One constant" overstates the rule.** It uses the held-out model's *own* D(W) curve, measured on that model's trace, while the power law uses only C/k. Even with that extra information it does not beat the two-constant power law (bootstrap ratio of median errors 0.95 to 1.09).
- **c is not a constant.** It falls with budget in 9 of 9 models, with only three budgets per model.
- **Sub-token points.** 5 of the 26 points sit below one token, and their D/C equals W50 by linear interpolation. Dropping them leaves the median at 0.63, so this is robust but should be stated.
- **The forecaster evidence is weak.** The degraded windows draw independent uniform errors; the forecasters are a ridge model and a 512-unit GRU trained on small out-of-domain text. "No forecaster we know of or trained reaches" this horizon is weak evidence about what is achievable.

**Fix.**
- Call the rule what it is: a normalisation of the horizon by working-set growth.
- Train at least one stronger in-domain forecaster (larger routing corpus, transformer over routing histories) and evaluate it with correlated errors.

### W8. The pre-registration is transparent but not confirmatory for the headline claims.

What is good:
- Each job script carries predictions committed minutes before the rental starts. I checked jobs 096 and 099 to 102: commit times precede the manifests' start times by 1 to 17 minutes.
- Prediction content did not change after launch, except a disclosed amendment adding hosts 100e and 100f before their own launch.

What limits it:
- **The predictions are many, small and often weak**, and the hand scoring is by the author. Of the 473 scored late clauses, only 168 held under the interval rule.
- **The constructs behind the main claims came after the data:** the 0.5 class line, the 0.65·C rule, the read-once cap.
- **The timestamps are self-reported** git dates, not third-party.

**Fix.** Register 3 to 5 primary hypotheses with decision rules and an analysis plan on OSF or AsPredicted before the next round. Mark each main-text claim as confirmatory or exploratory.

### W9. The literature comparison should not be in the abstract.

The "median 13.6% for published systems and 27% for ours" claim has several problems:
- It scores heterogeneous end-to-end numbers (different precisions, sometimes traces of sibling models, datasheet ceilings) against a bound that is "not theirs" for pooled, whole-layer or speculative designs.
- The 20 in-class rows include 4 below 5%, which Table 19's own caption says are "more likely configuration mismatches ... not adjudicated".
- `audit_ours.py` hard-codes the paper's own speeds rather than reading them from results.

**Fix.** Move the comparison to an appendix with the caveats. Keep in the main text only systems reproduced on the same machine (FreeToken, llama.cpp, Pipelined Sharding, the leloch cache, KTransformers).

### W10. Smaller validity and consistency issues

- **The datasheet GPU ceiling is not an upper bound on host B.** Its card runs GDDR7 at 17001 MHz, and its own probe measured device reads of 1847 GB/s against the 1792 GB/s datasheet value used in the bound. Use max(datasheet, probed) per card. This changes GPU-bound cells by about 3%.
- **"Bound, measured GPU" (Table 2) is not a bound.** It is calibrated on stock llama.cpp kernels that other kernels could beat. Call it a reference.
- **Table 2 and the trace decode different text.** Table 2 runs free greedy decoding, so routing diverges from the trace on which R\* is computed (fully identical outputs were 0 to 3% in job 073). It is a minor effect, but state it.
- **Stale and misattributed numbers.**
  - The cost macro says jobs 058 to 101 (88 rentals, $70.8). The ledger now includes job 102 (91 rentals, $71.2), so the "every number is generated" pipeline was not rerun end to end.
  - "MIN caches 5.9× as many experts as the deployed policy" is computed on the in-step state (20.3 forced fetches against 3.46 admissions). The sentence refers to MIN loaded the deployed way, which admits 4.6×.

## 4. Methodology and statistics

**The bound.**
- **Validity.** Valid as a probe-relative lower bound for the stated class. I verified R\* exactly. Using the maximum probe sample, a single 87.5 GB/s concurrent sample on host B against 76.8 to 83.4 for the others, is the conservative choice for a bound.
- **"Speed of light".** The bound is an empirical roofline in the sense of ERT, not a physical ceiling; DIMM configuration and theoretical DRAM peak are unknown on rentals. The paper is honest about this, but appendix wording ("speed of light", "physical speed-of-light model") overreaches.
- **Tightness.**
  - On the host side, the bound is tighter than the paper believes (W1).
  - On the GPU side, it is about 2× loose at batch 1.
  - Overlap is assumed, not measured.
- **Attainability.** Shown only for read volume without compute (W2).

**The microbenchmark (job 102).**
- Well-engineered and reproducible; I recomputed all 24 fractions of Table 17 from raw output.
- The analytic per-layer model was 12 points optimistic on the Intel host. In the concurrent probe, that host's link falls from 28 to 10 to 15 GB/s when the CPU reads, which explains this. The B_cp abstraction does not capture asymmetric contention, and that matters for any per-machine split.

**Statistics with the machine as unit.**
- Right in spirit and better than most of the field.
- **Crossed bootstrap.** Implemented correctly (one problem resample applied to all drawn hosts, in the pigeonhole sense).
- **Remaining issues (W4):**
  - small n with percentile intervals;
  - arithmetic means of ratios;
  - single launches per host;
  - no variance-component model.
- **The 0.009 / 0.154 / 0.030 contrast is the most valuable statistical fact in the paper:** problems < launches < hosts. It deserves a figure.

**The calibrated time model.** Fine as an engineering device to pick a fetch table. As a scientific claim it is a one-constant-per-budget fit evaluated on absolute time (W5). The leave-one-host-out freezing is the right protocol; the metric is not.

**The order-free accounting.** Correctly computed; I reproduced Table 3's O4 rows to 0.1 point. Its interpretation is limited by the interaction size and by the non-uniqueness of "MIN's set" (W1, W6).

**The foresight study.** The engine-side windows are cleanly done; I reproduced every Table 4 entry and the panel time shares. The trace-side W50 and D(W50) statistics reproduce. Their extrapolation to "no forecaster reaches it" is under-powered (W7).

**Pre-registration.** Exemplary bookkeeping. As a method, though, it guards against forking paths only for the many small predictions, not for the paper's main constructs (W8).

## 5. Clarity

**Score: 2 / 5.** The writing is precise sentence by sentence but dense well beyond what an MLSys reader will tolerate. The main text carries job numbers, host codes and configuration nicknames, and the appendices read like lab notebooks.

| Passage | Problem | Fix |
|---|---|---|
| Abstract (about 300 words) | Packs a system result (11 of 12 vs FreeToken), a literature audit, a microbenchmark, a factorial, a model and a horizon rule into one paragraph, with roughly 15 numbers | Three sentences, one per finding, at most one number each |
| Section 2, "Hosts and statistics" | One paragraph mixes averaging conventions ("jobs 093–097 averaged speeds instead"), relaunch variance, panel design, bootstrap and the scorecard | Split into "Design", "Statistics" and "Pre-registration"; move job-specific exceptions to a footnote |
| Terminology for the bound | It is "bound" (Section 3), "limit" (Appendices D to F, Tables 8 to 11), "speed limit" (Fig. 6) and "speed of light" (Fig. 9, Appendix J) | One term throughout |
| Terminology for the time model | Called "the calibrated model" in Appendix C, which insists "not a law", yet "law's table" in Fig. 5 and "the law predicted llama.cpp" in Appendix I | One term |
| Host names | A, B, S, O1 to O6, Pa to Pj, "Pd again"; the same machine (job 100f) is "9950X, x8 (0.54)" in Table 3 and "9950X, slower link" in Table 4 | One host table with stable names, CPU, link/CPU ratio and the jobs run |
| Configuration names leaking from code | foa, nb2, both3p, hit-opt., lead, paced (Tables 9 to 12) | Use Table 1's names everywhere |
| Table 3 | Ten columns, ranges in small type, nested prefetch, a "rest" that is a residual | Show the 2×2 cell times with intervals; put Shapley in an appendix |
| Appendix B | A single, roughly 60-line paragraph | A table by era: clauses, held, held (point), failed, and the three most consequential failures |
| Section 5 | "Window", "deployed + window", "in step" and "deployed path" are all used before the reader has a picture | A small diagram of the two load paths and where a window acts |

## 6. Artifact checks

All recomputations used copies in a scratch directory.

| # | Claim (where) | What I did | Result |
|---|---|---|---|
| 1 | R\* for gpt-oss at C = 14 / 32 (bound, Section 3; Table 2) | Exact interval LP per layer with HiGHS, independent of `mosl.cachesim`; also my own greedy | **Held.** 38.315 / 15.340 reads per token; identical per layer to `mosl` |
| 2 | Bound 172 / 430 tok/s on host B, gpt-oss 11% / 25% (Table 2) | B_host = max of job 081 `concur.txt` = 87.5 GB/s; R\*·S/B_host | **Held.** 172.3 / 430.4 |
| 3 | Ours ÷ FreeToken 1.294 [1.278, 1.312], 1.275, 1.154, 1.032; ours ÷ llama.cpp 2.00 (Table 2, host B) | Own paired bootstrap (10,000 resamples) on job 081 `bs1.jsonl`, launch 2 | **Held.** All to 3 decimals; harmonic-mean ratios within 0.002 |
| 4 | Table 17: per-layer 86 to 96%, per-token 93 to 97%, link-only 35 / 95 / 53%, CPU-only 92 / 86 / 91%, terms 7.04 / 9.70 / 9.56 ms | From `results/102?_readsched/readsched_C*.txt` and `concur.txt` | **Held.** All 24 entries |
| 5 | Job 102 committed before launch; scorecard 33 = 0 / 28 / 5 | `git log %h %ad` vs `manifest.json` start_utc; `scorecard_102.json` | **Held.** Commit 15:14:17Z; machines started 15:15:11Z, 15:15:17Z, 15:19:23Z; only the host name changed afterwards |
| 6 | "Admissions are 57% (69%) of MIN's reads"; read-once cap 53–71% / 43–58% (Section 3) | Reproduced from `readsched.json`. Then solved the min-admissions-at-R\* LP (MILP- and brute-force-checked) | **Failed as a statement about MIN.** True only for greedy MIN. A hit-optimal schedule needs 12.40 / 6.75 admissions per token (32% / 44%). Cap then ≥0.93 at 11% on all hosts; a link-aware bound is within 1% (11%) and 7% (25%) of the paper's |
| 7 | Table 3, O4 gpt-oss 11%: gap 10.6; load 3.9, cache 4.5, both 51.5, interaction 43.2, Shapley 26.1 / 25.5, prefetch 12.7, rest 35.7 | Own computation from job 096a `ec_g_C14.jsonl` and the limit in `foresight_096a.json` | **Held exactly.** Also the O4 25% and O5 11% rows. Within-host CI of the interaction [42.4, 44.0] |
| 8 | Panel: median within-host half-width 0.009, SD across hosts 0.154, two-stage interval [1.12, 1.30] (Section 4) | Own crossed bootstrap on 099a–j | **Held.** 0.0085 / 0.154 / [1.119, 1.305]; t-interval [1.11, 1.33] is wider |
| 9 | Relaunches within 1.9% in deployed time; ratios move by at most 0.030 (Section 2) | Compared 099d / 100a, 099h / 100c, 099f / 101a; matched GPU UUIDs | **Held.** 1.9% / 0.030; same physical GPUs |
| 10 | Time model: median 2.2%, p90 10%, 14 hosts (Section 4) | Reran `hostdep_model.py` on a copy; added frozen one-path and null baselines | **Held.** Macros byte-identical. One-path 7.1%, null 9.0%; but gain over-predicted, median 1.19× |
| 11 | Fig. 2: correlation 0.93 with the log ratio at 11% | One point per machine, 17 machines | **Held.** 0.928; 0.87 without the 0.29 machine |
| 12 | Table 4, job 100: 32 speed ratios | Recomputed from 100a/b/c/f raw rows | **Held.** All entries; also "at worst 0.997×" for the prefetched copy (101a) |
| 13 | Window time shares (Section 5): 0.14–0.17, 0.32–0.47, 0.78–0.93; half-right 8 tokens 0.20–0.26; 25%: 0.21–0.24, 0.49–0.63 | From panel raw rows | **Held** |
| 14 | W50 ≈ 0.67 (C/k)^1.31, r = 0.977; 1.35 without sub-token points; D(W50) median 0.65 C (0.61–0.72), CV 0.16 | Refit from `w50_aa.json` / `w50_distinct.json` | **Held.** Median 0.63 without the 5 interpolated points |
| 15 | Learned order runs 0.90–1.05× the deployed cache (Section 5) | Job 097a/b raw rows | **Held.** 0.896–1.054 |
| 16 | Scorecard 565 = 218 / 193 / 142 / 8 / 4; era counts; 099 = 190 / 42 / 39 / 2; 100 = 41 / 122 / 27 / 26; 101 = 6 / 19 | Tallied the JSONs | **Held** |
| 17 | In-class published median 13.6% (8.1–20.6), n = 20 (Section 3) | Recomputed from `prereg/audit/*.json` | **Held**, but includes 4 rows below 5% that the paper calls likely mismatches |
| 18 | Same-CPU hosts differ by 16–29% (Section 2) | 285K pair and 9950X pair, both budgets | **Held.** 15.5–29% |
| 19 | "MIN caches 5.9× as many experts as the deployed policy" (Section 4) | Engine counters, job 096a | **Partly.** 5.9× in the in-step state; 4.6× in the 2-read state the sentence describes |
| 20 | Datasheet GPU term is an upper bound (Eq. 1, Table 2) | Host B `nvidia-smi -q` and `bw.txt` | **Failed (minor).** Memory at 17001 MHz; device read 1847 GB/s > 1792 |
| 21 | Cost: 88 rentals, 53 machines, $70.8 (Appendix K) | Recomputed from `gpu/vast_ledger.json` | **Held** for jobs 058–101; the macro is stale (with job 102: 91 rentals, $71.2) |
| 22 | Unit tests | `pytest` on a copy (needs `data/`) | 15 passed |

## 7. Scores

| Criterion | Score |
|---|---|
| Overall (1–10) | **5** (borderline reject) |
| Confidence (1–5) | 4 |
| Novelty (1–5) | 3 |
| Soundness (1–5) | 3 |
| Significance (1–5) | 3 |
| Clarity (1–5) | 2 |
| Methodology (1–5) | 4 |

**Rationale.** Methodology is a genuine 4: the measurements are honest and reproduce to the digit. Soundness is a 3 because one central interpretive mechanism (W1) does not survive a check that takes a few minutes of LP time. Novelty is moderate: the bound is roofline plus Belady, close to WiSP, Budgeting Bytes and Zhang [62]; the in-engine factorial and the distinct-expert horizon are new. Significance is real but niche (batch-1 offload on consumer GPUs). Clarity is the main barrier for this venue.

If W1 is resolved with the min-admission oracle (either outcome, honestly reported), and the paper is reorganised around three clean findings, I would move to 6 or 7.

## 8. Would I take this student?

Yes, and I would want them in a measurement group in particular.

What I would bring:
- The habits that are hardest to teach are already here: pre-committed predictions, raw rows for everything, a scorecard that reports 142 failures, and relaunches to separate machine from launch noise.
- The instinct to ask "how fast could this machine possibly go?" before claiming a speed-up.

What I would work on in the first year:
1. **Question the formal objects as hard as the measurements.** Here, *which* optimum MIN denotes when there are many.
2. **Say less, better.** This paper has three good papers' worth of material fighting for one abstract.

Both are teachable. The rigour is not something I could install in someone who lacked it.

## 9. The changes that would most raise the score

| Priority | Change | Effort | GPU-rental cost |
|---|---|---|---|
| 1 | **Fix W1.** Compute the misses-versus-admissions frontier and the link-aware bound for both models and all budgets (parametric LP or min-cost flow). Add a minimum-admission MIN oracle to the engine (one-read and two-read). Rerun the 2×2 at gpt-oss 11% and 25% on the four slow-link machines and two fast ones. Rewrite finding (2), the cap paragraph and the "rest" decomposition from the result | About 1 day LP; 1 week engine + runs | About $6 (6 rentals × about 1.5 h × $0.6/h) |
| 2 | **Attainability with compute (W2).** Extend `readsched.cu` with admissions forced over the link, helpers running the real expert kernel, and concurrent GPU work. Report the fraction of the bound on 3 to 4 hosts | 1 week | About $4 |
| 3 | **Statistics (W4).** Relaunch every panel host once. Fit a host / launch / problem random-effects model. Report across-host prediction intervals and geometric means. Redo Fig. 1 and Table 3 with intervals | 3 to 4 days | About $10 |
| 4 | **Scope (W3).** Pre-register the class line. Add at least 5 slow-link hosts and Qwen3 at both host-bound budgets on the panel. Add one non-AIME workload with 2k-token outputs on 5 hosts | 1 to 2 weeks | About $25 to 35 |
| 5 | **Time model reported on gains and decisions (W5).** Add a background-contention term or narrow the claim | 2 to 3 days | $0 |
| 6 | **Horizon rule (W7).** Restate it honestly. Train one stronger in-domain forecaster and evaluate it with correlated errors in the engine on 3 hosts | 2 to 3 weeks | About $20 to 50 (training) + $5 (engine) |
| 7 | **Clarity (Section 5).** Three-finding abstract; one host table; consistent terms; 2×2 cell tables; appendix scorecard as a table; literature audit out of the abstract (W9); fix the GPU ceiling, stale macros and misattributed numbers (W10) | 2 weeks | $0 |

**Total.** About 6 to 8 weeks of work and roughly $70 to $110 of rentals. Items 1 to 3 alone (about 2.5 weeks, about $20) would address the weaknesses that most limit the claims.
