# Review 9b: "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth"

**Reviewer.** A professor whose group works on performance modelling and scientific benchmarking of parallel and ML systems: rigorous measurement, the Hoefler–Belli rules, roofline-style bounds, and statistics that treat the machine as the unit.

**Calibration.** MLSys main track. Overall: 5 = borderline reject, 6 = weak accept, 7 = accept, 8 = strong accept, 9 = top 5% of accepted papers, 10 = best-paper level. The sub-scores run from 1 to 5. I was asked to be demanding, and I have been.

**Independence.** This is a blind, independent review.
- I read the paper PDF in full, including all appendices, and the supplement (the prediction scorecard).
- In the artifact I read only what the checks needed: scripts/, the job scripts and jobs/ec2/ (minadm_plan.py, readsched.cu, the engine patch), the raw result directories of the GPU branch, and the prereg/*.json data files.
- I did not open anything under reports/, any prereg/*outcome*.md file, research_notes/, or any file named like a review, plan or progress log. The one exception is jobs/ec2/minadm_plan.py, which the brief names. I did not read commit messages. I used `git log --format='%h %ad'` and `%cd` only, to order commits against job start times.
- All recomputation ran in a scratch copy. Before running anything I deleted the outcome notes from that copy without opening them. No file in either repository was modified. The only file written is this review.

---

## 1. Summary

The paper studies batch-1 decode of MoE models whose experts mostly live in host DRAM, on rented RTX 5090 desktops. It covers gpt-oss-120b in MXFP4 and Qwen3-30B-A3B in BF16, decoding AIME-25 prompts. It makes five claims.

1. **A bound on time per token (Eq. 1).** The bound takes the fewest host reads any policy with C slots per layer can make (Belady's MIN with bypass, R\*) over the host's highest probed read rate, combined with a GPU term in a min–max over how many experts run on the CPU.
   - The authors' expert cache (a llama.cpp patch) reaches 25–43% of the bound. It beats FreeToken at 11 of 12 cells.
   - A microbenchmark (job 102) replays MIN's per-layer reads with no model. It reaches 86–96% of the bound's host term.
   - A "link-aware" variant forces every admission over PCIe. It raises the bound by at most 7%.
   - Published systems scored by the same procedure reach a median of 13.6%.
2. **In-engine oracles.** These read the future routing from a recorded trace. Caching MIN's set and copying each admission once, in the step, beats the deployed cache by 16–51% on three hosts.
   - The benefit depends on the host's link-to-CPU read-rate ratio, across 19 machines.
   - The greedy MIN schedule loses on one slow-link machine (0.86×). MIN's fewest-admission schedule, computed by an LP over reuse intervals (job 104), gains there (1.04×). On three machines it adds 0.08–0.17 to the speed ratio.
3. **A 2×2 "order-free (Shapley)" accounting.** The two factors are what to cache and how to load. Either change alone closes at most 26% of the gap; both together close up to 60%.
4. **A calibrated two-path time model (Eq. 2).** It predicts in-step states within a median 2.2% on 14 hosts.
5. **A price on foresight.** Degraded windows and a nine-model trace study find that half of MIN's read saving needs about 0.65·C distinct experts per layer of lookahead. No online policy, learned order, speculative batch or forecaster the authors tried comes close.

The artifact is unusually complete. Every number is script-generated. Every job's predictions sit in its script header, committed before launch, and are scored clause by clause.

## 2. Strengths

**S1. The measurement discipline is well above the venue's norm.**
- Comparisons are paired by problem.
- Configurations run in shuffled order with recorded seeds, each from a cold cache in a fresh context.
- Probes are recorded per job, along with clocks and power limits.
- The Hoefler–Belli checklist (Appendix A) is followed, not just cited.
- I confirmed the relaunch claims against GPU UUIDs. Pd, Pf, Pg and Ph really are the same physical cards across jobs 099–104. Base time per token moved by at most 1.9% and every ratio I checked by at most 0.029.
- Job 103 failed to install its solver and so ran the greedy schedule under the "plan" label. That gives an accidental A/A test: identical configurations agree within 0.008 in speed ratio.

**S2. The host dependence of how foresight should be spent is real.**
- Across 19 machines, the gain of MIN with one in-step read correlates with log(link/CPU) at 0.94. I recomputed this from raw rows: 0.941; 0.913 without the most extreme machine; Spearman 0.886.
- This is the paper's most robust and most useful empirical result.
- The finding that two machines with the same CPU model differ by 16–29% is a good argument for treating the machine as the unit, and the paper does so.

**S3. The 2×2 factorial (what to cache × how to load) is the right experimental design.**
- Its main lesson is practical: MIN's set is worth little when each admission is read twice. Cache-alone and load-alone each close about 4% at O4 gpt-oss 11%; together they close 52%.
- Systems builders can act on this lesson.

**S4. The artifact reproduces.** Every number I recomputed independently, about 20 of them, matched (Section 6).
- Three analysis scripts reproduce from a clean scratch copy: the job 104 scoring, the calibrated-model evaluation, and the job 102 analysis.
- Deterministic counters are bit-identical across hosts.
- An independent LP formulation of minimum misses (scripts/linkaware.py) gives exactly the R\* of the MIN simulator: 38.315 per token at gpt-oss 11%. This is a nice internal check of the bound's main input.

**S5. Failures are reported honestly.**
- Of 565 clauses, 142 failed. The limitations section names the weak points itself: one slow-link machine, a post hoc dividing line at half, a time model that is optimistic for background copies, and no task-accuracy evaluation.

**S6. The work is frugal.** By the ledger, all 99 rentals cost about USD 74 (USD 70.8 for jobs 058–101 as stated, plus USD 3.4 for jobs 102–104). Reproducing it, or extending it as asked below, costs very little.

## 3. Weaknesses, ranked by how much they limit the claims

### W1. The new headline result, fewest-admission versus greedy MIN, rests on three machines, and its slow-link half rests on one.

The abstract and conclusion state that "the greedy one loses on a machine with a slow link, where the set with the fewest admissions gains". The evidence behind this is thin.

- **Sample.** Job 104 ran on three machines, two budgets, one launch each. Only one machine (Pf, a Core Ultra 9 285K, link/CPU 0.29) is in the regime where greedy loses.
  - The "four launches" of the 0.86× figure are four launches of that same card (verified by UUID).
  - Launches replicate the machine, not the phenomenon.
- **Fit to its own predictions.**
  - Job 104 ran on the submission day. By the repository's timestamps its results entered the paper within about half an hour.
  - 24 of its 47 clauses failed.
  - The mechanism-level prediction failed on all three hosts. The plan's set was expected to shrink the interaction; it grew.
  - The calibrated model, given job 101a's counters, predicted 1.21–1.29× for the plan on Pf. The measurement was 1.035×.
- **It is not a comparison of two hit-optimal schedules in the engine.**
  - minadm_plan.py is hit-optimal within sequences. The engine carries residents across sequences.
  - In replay, the plan misses 0.3% more than greedy. In the engine it misses 2.5% more at 11% and 5.3% more at 25%.
  - The plan's misses deviate from the host's own replay by 3.8% and 6.8%; the prediction was at most 2%. This is unexplained.
  - The comparison is therefore "40% fewer in-step copies for 2.5–5.3% more misses". It is not "which of MIN's hit-optimal sets".
- **A confound in the copy mechanism.**
  - The engine's in-step copy defaults to an SM-based zero-copy kernel with 16 blocks (`LLAMA_EC_ZC=1`).
  - On the 285K host (103b/104b) that path reads 36.0 GB/s. The copy engine reads 49.6 GB/s.
  - The paper calls this machine "40 GB/s link"; the job script says "48 GB/s". Part of greedy's in-step penalty may be an implementation choice, not a property of the link.

*Fix:*
- Pre-register a sample of at least 10 further machines spanning link/CPU 0.25–1.0, with at least 4 below 0.4 from both CPU vendors. Run two launches each, at both models.
- Resolve the replay–engine miss discrepancy before running.
- Run in-step copies on the faster of the copy engine and the zero-copy kernel, per host.
- Until then, move the claim out of the abstract, or state it as "on one machine".

### W2. The bound's tightness is argued with a near-tautology, and the bound is relative to a probe, not a physical ceiling.

- **B_host is an empirical maximum, not a ceiling.**
  - On host B it is the largest concurrent CPU+PCIe "sum" sample in the probe: 87.5 GB/s. The audit's own ceiling for the same AM5 platform is 96 GB/s.
  - The paper concedes that "a faster reader than our probe could exceed it". Yet the language used ("speed of light", "no policy, even one that knows the future, reads fewer") gives the result the status of a physical bound.
  - The bound also assumes two tiers. A policy that deliberately keeps a few experts in a large CPU L3 (128 MB on a 9950X3D, about 9 gpt-oss experts) is outside the bound's class. This should be said.
- **Job 102 adds little beyond the probe.**
  - Its CPU threads stream from a buffer and the copy engine copies. That is the same kind of traffic the probe measured.
  - Unsurprisingly, it reaches 86–96% of a term defined by that probe. On the Pd host, "CPU only" alone reaches 92%; on the 9950X, "link only" reaches 95%.
  - The non-trivial content is narrower: per-layer, whole-expert granularity costs at most about 14% (Pd).
  - The benchmark runs no expert compute (the engine's helpers take 1.09–1.31× the probe's time per expert). It has no router dependence. It ignores admission semantics, which the paper itself admits.
  - It also uses a different copy mechanism from the engine (the copy engine, not the 16-block zero-copy kernel).
- **The full bound is never approached.**
  - The best in-engine oracle on O4 and O5 reaches 46–75% of it. The paper says 46–73%; at O4 Qwen3 12.5%, the 2-step-lead oracle reaches 74.7%.
  - "The rest" is 30–62% of the gap. It is attributed to imprecise prefetch, partial overlap and half-datasheet GPU speed, but by argument, not by a measured decomposition.
- **The link-aware bound is computed on a sampled frontier.**
  - linkaware.py interpolates chords between five LP solutions. Chords lie above the convex frontier, so the computed minimum can overstate the true lower bound.
  - With the supporting-line (outer) approximation I get +6.3% (099f) and +6.6% (101a) at 25%, against the paper's +6.9% and +7.2%. "At most 7%" survives.
  - A bound paper should nevertheless use the outer approximation, or the exact parametric frontier.

*Fix:*
- Call it a probe-relative bound everywhere, and add a column at the platform's DRAM peak. Record the DIMM configuration in the probe.
- Test reachability with compute: replay MIN's schedule inside the engine with copy-engine transfers overlapped with GPU work, or add GEMV-rate CPU work to readsched.cu.
- Decompose "the rest" with Nsight timelines on one host.
- Use the outer approximation in linkaware.py.

### W3. The calibrated model and the Shapley accounting are presented as findings, but they are largely book-keeping.

- **The model's inputs come from the run it predicts.** Eq. 2 takes X_c and X_p from the very run whose time it predicts. In host-bound runs, time ≈ bytes/bandwidth almost by construction.
- **G is a sink for model error.**
  - G is a fitted constant that ranges over 4.1–9.2 ms across host-cells. It exceeds the whole all-in-VRAM token (5.17 against 3.92 ms) because it absorbs whatever the model misses.
- **It does not beat a simpler model.**
  - In the blind test it ties a one-path model (median 3.4% against 3.6%).
  - Its one discriminating success is the sign of the in-step gain. That rests on 2 losses out of 39 host-cells, both on Pf.
  - It under-predicts every background-copy state (64 of 64; median 16%).
  - It overshot Pf's fewest-admission speed by 17–25%.
- **So the explanatory sentences overreach.** "How much it gains is a property of the machine that its two read rates largely predict" goes further than this evidence. Fig. 2's raw correlation supports the qualitative statement better than the model does.
- **"Shapley" adds nothing for two factors.**
  - With two factors, the Shapley value is the classical 2^k main effect (Jain's sign table): φ_cache = ½[(base − bypass) + (foa − fetch)]. I verified the arithmetic for O4.
  - The dominant term is the interaction (43% of the gap at O4 gpt-oss 11%, against 4% and 4.5% simple effects). Shapley splits it equally between the two factors by axiom. The "26% and 25.5%" are a convention, not a measurement.
  - The five-player Shapley of Appendix E is weaker still, since two of its players are definitions (the paper says so).
  - The gap denominator uses the datasheet GPU rate, so "the rest" includes known slack in the bound.

*Fix:*
- Report the 2×2 as simple effects plus interaction, with an interaction plot. Drop the Shapley framing in the main text.
- Evaluate the model with byte counts from a policy simulator, not from the measured counters. Test it where it matters: held-out slow-link hosts.

### W4. The horizon rule's "held-out" test leaks information about the held-out model, and the negative result on forecasters is narrow.

- **Leakage.** In w50_rebaseline.py the D(W₅₀) = c·C rule inverts the held-out model's own distinct-experts curve D(W), computed on that model's full trace. The C/k rules use no information about the held-out model. "A one-constant rule that predicts a held-out model as well as a two-constant power law" is therefore not like-for-like. Once one has the trace needed for D(W), W₅₀ can be computed directly.
- **No clear winner.** The median error factors (1.16, 1.19, 1.22) are close. The paper says the rules are "not distinguished".
- **The forecasters are weak.** The "no forecaster we tested approaches it" result uses a ridge model and a 512-unit GRU, trained on small, out-of-domain text. The paper admits this bounds cross-text behaviour only.

*Fix:*
- Compute D(W) for the held-out model on disjoint text, or from its architecture (E, k) alone.
- Report a bootstrap interval on the difference in error between rules.
- Either train an in-domain forecaster or soften the claim to "the forecasters we trained across text".

### W5. The statistics stop at the host level just where the claims need it most.

- **Within-host intervals miss launch variance.** They are percentile intervals over 20–30 problems, with half-widths of about 0.005–0.01. Launch-to-launch variation is up to 0.03. The paper sensibly refuses to read differences below 0.03, but this should be a variance component, not a rule of thumb.
- **The pooled "crossed bootstrap" estimand needs fixing.**
  - It is implemented correctly for a crossed design: one problem draw is applied to every drawn host.
  - But its estimand is the arithmetic mean of per-host ratios over a deliberately heterogeneous convenience sample (8 fast-link and 2 slow-link hosts). It uses 2,000 resamples, whereas Appendix A says 10,000.
  - With 10 hosts, percentile intervals undercover.
- **Almost no population-level inference appears.** The main text reports host ranges (honest) but makes no population-level statement, and gives no prediction interval for a new machine. A new machine is exactly what a practitioner holds.

*Fix:*
- Fit a mixed model on log time per token: configuration as a fixed effect, a host random effect crossed with problem, and a launch component estimated from the relaunched machines.
- Treat link/CPU as a host-level covariate (a meta-regression), which is what Fig. 2 does informally.
- Report the host-level SD of each effect and a 95% prediction interval for a new host.

### W6. The pre-registration is careful forecast logging, not confirmatory design, and the scorecard over-counts.

- **The ordering holds but is not tamper-evident.** For jobs 099–104 the script commit precedes `start_utc` by 1–16 minutes (job 104: commit 17:33:12 UTC, first host started at 17:34:33 UTC). But these are author-controlled git timestamps.
- **The predictions are not hypothesis tests.** They are bands on the authors' own system, set hours after earlier runs on the same machines. Most of the engine evidence in Sections 4–5 (jobs 099–104) was gathered within about 12 hours on one day.
- **The decisive analysis choices were post hoc:**
  - the half line (acknowledged);
  - the D-rule;
  - which hosts count as "slow";
  - the estimand for pooling.
- **Counts are inflated and the "held (point)" category is mostly self-made.** In job 104, 24 of 47 clauses are four deterministic counter comparisons, identical on all three hosts, counted once per host and budget. 15 of the 24 "failures" are the same 5 counter outcomes counted three times. 46 of 47 clauses carry no interval, because none was computed; most of these are speed or difference clauses for which a paired bootstrap would be trivial.

*Fix:*
- Register estimands, host sample sizes, stopping rules and decision thresholds before the panel, in a tamper-evident place (a signed tag, OSF or Zenodo).
- Count each deterministic quantity once.
- Attach intervals to every non-deterministic clause.
- Label follow-ups such as 100–104 as exploratory unless they were planned beforehand.

### W7. The comparison with published systems is not like-for-like, yet it sits in the abstract, introduction and conclusion.

- The in-class median of 13.6% (n = 20) reproduces.
- **The class is chosen by the authors.** Trace rows exclude speculative and lossy systems. The uniform-routing rows, with a median of 34.8%, are left out. Rows below 5% are "not adjudicated" but still count toward the median.
- **The hardware and protocols vary widely.** The rows span a GTX 1080 Ti to an RTX PRO 6000, with different precisions and timing protocols. Sibling-variant traces stand in where the model's own trace is missing.
- Using datasheet ceilings for everyone (ours 18–42%) is a fair procedural choice. But the attainable fraction of datasheet DRAM bandwidth differs by platform, so the percentages are not comparable across platforms.

*Fix:* move this to an appendix, or restrict it to systems run on the same machine.

### W8. The scope sprawls.

The submission is three papers in one:
- an expert cache that beats FreeToken;
- a bound with an audit of published systems;
- an oracle-and-horizon study of foresight.

Each is compressed until the reasoning disappears into numbers (Section 5). The strongest and most original contribution, the host-dependent value of foresight and the 2×2 interaction, has to compete for space with FreeToken tuning, KTransformers, Mixtral, long outputs and a 52-row audit.

*Fix:* centre the paper on the bound plus the price of foresight. Move the system comparison and the audit to appendices or a companion paper.

### W9. Minor errors and inconsistencies

All of these are found in the artifact checks.
- "Even the best oracle reaches only 46–73%". The best oracle reaches 74.7% at O4 Qwen3 12.5%; the 46–73% range is for MIN prefetched.
- The 285K's link is reported as "40 GB/s" in the paper and "48 GB/s" in the job script. The probe shows 49.6 GB/s (copy engine), 39.9 GB/s (zero-copy, 64 blocks) and 36.0 GB/s (the engine's 16-block kernel).
- Appendix A says 10,000 resamples; the pooled bootstrap uses 2,000.
- The cost statement omits jobs 102–104 (USD 3.36, 11 rentals).
- A job 103 rental (103c, 7800X3D, 0.25 h) has no results directory and is not mentioned. Hoefler–Belli ask for every run to be accounted for.
- Fig. 5's legend and Appendix I say "law", while Appendix C insists the model is "not a law".
- The main text's "20 rows, 13.6%" and Table 20's "trace 9.5% (n = 29)" are not reconciled in the main text.

## 4. Methodology and statistics

**The bound (Eq. 1).**
- *Valid under its stated assumptions.* These are exact routing, whole unmodified experts, at most C residents per layer, one token per pass, DRAM as the only host tier, and the probe's rate as the host ceiling.
- *Reproduced.* My own implementation of the min–max reproduces Table 2's bound column on host B: 172, 430, 517 (paper: 519), 96, 214 and 308 tok/s at datasheet GPU rate; 172, 279, 279, 96, 192 and 192 at the measured all-in-VRAM rate.
- *R\* is solid.* It agrees exactly with the minimum-miss LP of linkaware.py, and the LPs come back integral (no fractional layers).
- *The pooled variant is reported honestly.* It lowers reads by 4.5–18.6%, reproduced from speed_limit_v2.json.
- *The GPU term is loose by about 2×* (52% and 61% of datasheet at batch 1). This does not matter at host-bound budgets, and the paper reports both variants.
- *Weakest points:* B_host as an empirical maximum, and the absence of any reachability test with compute (W2).

**Job 102.**
- I recomputed all 24 entries of Table 18 from readsched_C14.txt and readsched_C32.txt on the three hosts; all match. Repetitions agree within 0.9%.
- The result is reproducible but answers a narrower question than the text implies. It shows that per-layer whole-expert granularity costs at most about 14% of a streaming read rate. It does not show that "the bound's read time is nearly reachable" by a decoding system.
- The abstract's phrase "reach at least 86% of this read time" inverts the quantity. It is the bound's term that is 86% of the replay's time.

**Link-aware bound.** The method is sound in spirit. Every admission must cross the link, and the LP is a valid relaxation. It should use outer approximations (W2). The fewest-admission count of 12.4 per token against 21.8 for greedy, at gpt-oss 11%, reproduces.

**Statistics.**
- *Paired ratios of means.* These are consistent with Hoefler–Belli. Ratios of total time agree with ratios of mean rates within 0.001 on host B, which I checked.
- *Host as unit.* The principle is right and the relaunch data support it.
- *Gaps (W5).* There is no variance-component model, no prediction interval for a new host, and the pooled estimand is an arithmetic mean of ratios over a stratified convenience sample.
- *What to do.* For heterogeneous host populations I would ask for a geometric mean of ratios, or a log-scale mixed model, with link/CPU as a covariate.
- *The decisive slow-link conclusions rest on n = 1 machine.* No statistical apparatus can fix that. More machines can.

**Calibrated time model.**
- The frozen-G evaluation reproduces exactly from a scratch rerun of hostdep_model.py: median 2.2%, 90th percentile 10%, 239 predictions on 14 hosts.
- It is a good engineering model for choosing the FETCH table, where it buys 3–34%.
- It is weak as an explanatory or predictive model of the oracles (W3). In particular, "predicts the share of each window within a median 0.03" uses measured counters for every state.

**Order-free accounting.** I recomputed O4's two host-bound gpt-oss rows from raw job 096a data.
- At 11%: gap 10.62 ms; load alone 3.9%; cache alone 4.5%; interaction 43.2%; φ 26.1% and 25.5%; prefetch 12.7%; rest 35.7%.
- At 25%: 4.4%, 22.3%, 17.4%, 31.1%, 13.1%, 21.9% and 34.0%.
- All of these match Table 4. Arithmetic is not the issue; the framing is (W3).

**Greedy versus fewest-admission (jobs 103/104, minadm_plan.py).**
- *The LP is well posed.* It works per layer over within-sequence reuse intervals, maximising hits minus μ·admissions with μ = 1/(n+1), which is lexicographic. It solves to integral optima; no layer needed the MILP fallback.
- *Table 3 reproduces exactly from raw rows*, including Pf's [1.02, 1.05] interval. So do the copy ratios (0.597 and 0.726) and the miss increases (+2.5% and +5.3%).
- *Within a host the comparison is trustworthy.* The in-step difference between the two schedules (0.08–0.17 at 11%) dwarfs the A/A noise (0.008) and the relaunch noise (0.03).
- *What is missing is breadth and a clean definition of what is compared* (W1).

**Foresight horizon.**
- The panel's time shares reproduce: windows of 1, 4 and 16 tokens recover 0.14–0.17, 0.32–0.47 and 0.78–0.93 at 11%, and 4 tokens recover 0.21–0.24 at 25%.
- So do the window losses: a 4-token window loses on 6 of 10 hosts at 11%; a half-right 8-token window loses on 8 of 10; admitting every miss runs 0.42–1.00×.
- The 0.65·C rule is a descriptive regularity with a leaky held-out test (W4).
- Degraded windows draw independent errors. A real forecaster's errors are correlated with cache state (acknowledged).

**Pre-registration.** See W6. This is commendable as a practice. I would like every MLSys group to log forecasts like this. But it should not be described as confirming the post hoc analyses.

## 5. Clarity (score: 2/5)

The paper is hard to read. Its density hides a clear and useful story. Specific passages and fixes follow.

1. **The abstract** carries about nine claims and over a dozen numbers in one paragraph, with internal jargon ("at gpt-oss 11%", "hit-optimal sets", "order-free accounting").
   - "MIN's own reads … reach at least 86% of this read time" is inverted.
   - *Fix:* three sentences — the bound and where systems stand; how foresight must be spent and how this depends on the machine; how much foresight is needed. Use at most four numbers.
2. **Host nomenclature.** Main-text readers meet host A, B, S, O1–O6, Pa–Pj, "Pf again", "Pg again (5950X)", "285K, 40 GB/s link", "9950X, x8", "9800X3D, slower link" and job numbers 093–104.
   - *Fix:* one host table with stable short IDs, CPU, DRAM, link, link/CPU ratio and the experiments each host ran. Keep job numbers out of the main text.
3. **Policy names drift.** The same configurations appear as "MIN, 1 read", "fetch", "fetch oracle", "bytes-optimal oracle", "bypass oracle", "both", "lead", "paced", "nb2", "foa", "aa", "single read" and "deployed + window".
   - *Fix:* use Table 1's names everywhere, including the appendix tables. Put the glossary next to Table 1.
4. **The yardstick has too many names.** "Bound", "limit", "speed limit", "speed of light" (Fig. 9), "law" (Fig. 5, App. I) and "calibrated model" (App. C) are used for overlapping concepts.
   - *Fix:* "bound" for Eq. 1, "time model" for Eq. 2, and nothing else.
5. **Sections 4 and 5 read as streams of ranges.**
   - Example: "it adds 0.08–0.17 to the speed ratio at gpt-oss 11%". Which ratio? Relative to the deployed cache.
   - Example: "on the 2 below that line, loading in the step costs time at 3 of 4 host-budgets, and at 3 of 4 on 2 more (job 103)".
   - *Fix:* one claim per paragraph, its evidence, then the caveat.
6. **Appendix B is one paragraph of several hundred words of nested clauses.** *Fix:* a table of per-job counts plus three or four sentences.
7. **Table 2's caption runs nine lines** and carries results (the pooled-bound fractions, the RTX PRO 6000 speeds) that belong in the text.
8. **Scope.** See W8. A focused paper would let Figures 1–3 carry the argument.

The figures themselves are clean. Fig. 2 in particular is effective.

## 6. Artifact checks

All recomputation used raw rows in `/home/claude/gpu-branch/results/` and scripts copied to a scratch directory.

| # | Claim (location) | What I did | Result |
|---|---|---|---|
| 1 | Table 3 (job 104): speed ratios for greedy/fewest × CPU/in-step, all 12 cells | Parsed `ec_g_C{14,32}.jsonl` on 104a–c; ratio of mean ms/token paired by problem, own bootstrap | **Held** (e.g. Pf 11%: 0.941 / 1.113 / 0.863 / 1.035) |
| 2 | Pf fewest-admission in-step 1.04×, interval [1.02, 1.05] | Own paired bootstrap (10k) | **Held** ([1.023, 1.053]) |
| 3 | Plan copies 0.60–0.73 of greedy, misses +2.5–5.3%; copies per token 19.8/11.8 and 9.7/7.0 | `st_g_C*_fetch{,plan}.json` counters | **Held** (0.597 / 0.726; +2.5% / +5.3%; identical on all 3 hosts) |
| 4 | Greedy in-step loses on Pf "0.86× on four launches" | Raw 099f, 101a, 103a, 104a; GPU UUIDs | **Held** (0.875, 0.860, 0.859, 0.863; same card), but it is one machine |
| 5 | Job 104 scorecard: 47 clauses, 1 held, 22 point, 24 failed | Reran `scripts/job103.py 104` in scratch | **Held**; 15 of 24 failures are 5 counter outcomes repeated across 3 hosts; 46 of 47 lack intervals |
| 6 | Table 2, host B: 1.294 [1.278, 1.312], 1.275 [1.254, 1.295], 1.154 [1.134, 1.172], 1.032 [1.022, 1.042]; 2.00× llama.cpp | `081/bs1.jsonl`, launch 2; own bootstrap | **Held** at all four cells; ratio of total times within 0.001 |
| 7 | Table 2 bound column (Eq. 1) on host B | Own min–max implementation with R\* from `speed_limit_v2.json`, B_host from the 081 probe | **Held** (172 / 430 / 517 vs 519 / 96 / 214 / 308; measured-GPU 172 / 279 / 279 / 96 / 192 / 192) |
| 8 | Table 18 (job 102): 86–96% per layer, and every other mode | Parsed `readsched_C*.txt` on 102a–c; term = R·S/B_host | **Held** for all 24 entries; repetitions within 0.9% |
| 9 | Link-aware: fewest admissions 12.4 vs greedy 21.8; bound raised ≤1% (11%) and ≤7% (25%) | Read `linkaware.json`; recomputed with supporting-line outer approximation | **Held** numerically; rigorous outer bound +6.3–6.6% vs chord +6.9–7.2% (method flaw, W2) |
| 10 | R\* = MIN's minimum misses (bound input) | LP μ→0 misses vs simulator `exact` | **Held** (38.315 = 38.315 at C = 14) |
| 11 | Running example O4 gpt-oss 11%: bound 10.0, deployed 20.6, MIN 1 read 15.2, prefetched 13.8 ms | Raw 096a | **Held** (10.02 / 20.64 / 15.16 / 13.81) |
| 12 | Table 4, O4 rows at 11% and 25% (load/cache alone, interaction, Shapley, prefetch, rest) | Own 2×2 decomposition from raw 096a | **Held** (26.1 / 25.5 / 12.7 / 35.7 at 11%) |
| 13 | "Best oracle reaches only 46–73% of the bound on O4 and O5" | Minimum over all oracle states per cell, raw 096a/b | **Failed (minor)**: best is 45.6–74.7%; "both, 2-step lead" reaches 74.7% at O4 Qwen3 12.5% |
| 14 | Correlation 0.94 of the in-step gain with log(link/CPU) at 11%, 19 machines | Own extraction of ratios and probe rates | **Held** (0.941; 0.913 without Pf; Spearman 0.886) |
| 15 | Relaunches within 1.9% (base) and 0.030 (ratios) | Raw 099d/100a, 099h/100c, 099f/101a/103a/104a, 099g/103d/104c | **Held** (max 1.9%, max 0.029); identity confirmed by GPU UUID |
| 16 | Same-CPU panel hosts differ by 16–29% | Raw 099e vs 099j, 099a vs 099f | **Held** (1.166–1.29) |
| 17 | Panel window shares (W = 1, 4, 16) and losses (4-token on 6 hosts; half-right 8-token on 8 of 10; admit-every-miss 0.42–1.00×) | Raw 099a–j | **Held** |
| 18 | Calibrated model: median 2.2%, 90th 10%, 14 hosts | Reran `hostdep_model.py` in scratch | **Held** (exact macros reproduced) |
| 19 | Audit in-class median 13.6%, quartiles 8.1–20.6, n = 20 | `audit_sol.json` + `audit.json`, own selection | **Held**; selection rule noted (W7) |
| 20 | Scorecard 565 = 218 / 193 / 142 / 8 / 4 | Counted `scorecard_clauses.json` | **Held** |
| 21 | Cost USD 70.8, 88 rentals, 53 machines (jobs 058–101) | Summed `vast_ledger.json` | **Held** (70.76; 53 distinct offers); jobs 102–104 add USD 3.36; 103c rental unreported |
| 22 | Predictions committed before each machine started (jobs 099–104) | `git log --format='%h %ad / %cd'` vs `manifest.json` `start_utc` | **Held** (1–16 min before); not tamper-evident |
| 23 | 285K host described as having a "40 GB/s link" | `concur.txt` of 103b/104b | **Inconsistent**: copy engine 49.6, zero-copy (64 blocks) 39.9, engine's 16-block kernel 36.0; job script says 48 |
| 24 | D(W₅₀) = c·C "predicts a held-out model" | Read `w50_rebaseline.py` | **Numbers held; method leaks** the held-out model's own D(W) (W4) |

## 7. Scores

| Criterion | Score |
|---|---|
| Overall (1–10) | **5** (borderline reject; a high 5) |
| Confidence (1–5) | **4** |
| Novelty (1–5) | **3** |
| Soundness (1–5) | **3** |
| Significance (1–5) | **3** |
| Clarity (1–5) | **2** |
| Methodology (1–5) | **4** |

Rationale.
- **Methodology (4).** The measurement methodology is the best I have reviewed in this area this cycle; it would be a 5 with a variance-component analysis and a confirmatory design.
- **Soundness (3).** What the paper reports is accurate, and 22 of my 24 checks held outright. Soundness stops at 3 because the paper claims more than it shows in three places: the tightness of the bound, the generality of the fewest-admission result, and the predictive status of the horizon rule.
- **Novelty (3).** A MIN-based bound and oracle caches are not new (WiSP, Budgeting Bytes, Zhang 2026b, Liang et al. 2026b). Cost-aware choice among hit-optimal schedules is known in caching theory (CHOPT, Jain & Lin 2018). The new parts are the in-engine time-domain evidence across many hosts and the host-dependence result.
- **Clarity (2).** It currently costs the paper most of its impact.

## 8. Would I take this student?

Yes, and gladly.
- The habits shown here are rare and hard to teach:
  - logging predictions before running;
  - reporting the failures;
  - relaunching machines to measure launch variance;
  - generating every number from a script;
  - spending USD 74 to answer questions others spend thousands on.
- The work also shows the problems of a researcher working without a group:
  - It sprawls. Each new result triggers another job rather than a decision about what the paper is.
  - Exploratory follow-ups run on the day of submission go straight into the abstract.
  - The prose is written for the author's own bookkeeping, not for a reader.
- These are exactly the things a group fixes:
  - someone to say "stop running jobs and write";
  - a statistician to turn ranges into a variance model;
  - a co-author to cut two thirds of the numbers.
- I would want this person to spend the first year learning to state one claim and design the experiment that could refute it. The measurement instincts are already there.

## 9. The changes that would most raise the score

GPU cost below assumes the ledger's typical RTX 5090 price (USD 0.5–0.7/h) and job-104-sized runs (0.6–1.3 h per machine-launch).

| # | Change | Effect on score | Time | GPU cost |
|---|---|---|---|---|
| 1 | Replicate greedy versus fewest-admission on a pre-registered sample of at least 10 more machines (link/CPU 0.25–1.0; at least 4 below 0.4; both vendors), 2 launches each, both models; resolve the plan's replay–engine miss gap first; use the faster copy path per host | Turns W1 from a single-machine anecdote into a result; +1 | 1–1.5 weeks | about USD 15–25 |
| 2 | Reachability with compute: replay MIN's schedule inside the engine with copy-engine transfers overlapped with GPU work (or add GEMV-rate CPU work and router-ordered barriers to readsched.cu); decompose "the rest" with Nsight on one host | Makes "the bound is nearly reachable" a claim about decoding, not streaming; +0.5–1 | 2 weeks | about USD 5–10 |
| 3 | Restructure around one story (bound plus the price of foresight); move the FreeToken/KTransformers comparisons and the audit to appendices; one host table, one policy glossary, one name per concept; rewrite the abstract in three sentences | Clarity 2 → 3–4; +0.5–1 | 1–2 weeks | none |
| 4 | Mixed-effects reanalysis of existing data (log time, configuration fixed, host and problem crossed random effects, launch component, link/CPU covariate); prediction intervals for a new host | Fixes W5 with data already in hand | 2–3 days | none |
| 5 | Present the 2×2 as simple effects plus interaction; evaluate the time model with simulator-predicted bytes on held-out slow-link hosts | Fixes W3's over-claiming | 3–5 days | about USD 5 (2–3 slow-link launches) |
| 6 | Repair the horizon test: D(W) on disjoint text or from (E, k); bootstrap the difference in rule errors; one in-domain forecaster or a softened claim | Fixes W4 | 2–4 days CPU (in-domain training: about USD 5–10 of rental time) | 0–10 USD |
| 7 | Bound hygiene: outer approximation in linkaware.py; a DRAM-peak column (record DIMM configuration in the probe); "probe-relative" wording; correct the minor errors of W9 | Removes easy attacks on a bound paper | 1–2 days | none (probe change rides on item 1) |
| 8 | Tamper-evident registration of estimands, host counts and decision thresholds before item 1; count deterministic clauses once; intervals on every stochastic clause | Lets the pre-registration actually confirm something | hours | none |

With items 1–4, I would expect to score this paper a 7 at MLSys. With all eight, it would be a strong candidate.
