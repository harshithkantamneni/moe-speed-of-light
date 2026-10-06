# Review 7b: professor, performance modelling and scientific benchmarking (6 October, morning)

**Submission:** "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth" (single author, Independent Researcher).
**Calibration:** MLSys main track. This is a blind, independent review. I read the paper (30 pp.) and the supplement (28 pp.), and checked the artifact in `/home/claude/moe-speed-of-light` and `/home/claude/gpu-branch`. I reran the analysis scripts only in a scratch copy, so no file in either repository was modified. I did not read any earlier reviews.

---

## 1. Summary

The paper studies batch-1 decode of MoE models (gpt-oss-120b in MXFP4, Qwen3-30B-A3B in BF16) on RTX 5090 desktop hosts. Only a budget of C experts per layer fits in GPU memory, and every other expert is read from host DRAM. A non-resident expert is read in one of two ways: either the CPU runs it, or it is copied over PCIe into a GPU slot.

1. **A bound.** For any policy that executes exact routing with at most C residents per layer, the paper gives a roofline-style lower bound on time per token, Eq. (1). Its inputs are the host reads of Belady's MIN with bypass (R\*) on the model's own routing trace, the host's best probed read rate, and the GPU's datasheet bandwidth, optimised over how many experts run on the CPU. The author's llama.cpp cache leads FreeToken at 11 of 12 cells and reaches 25–43% of the bound, or 31–56% with the GPU term at the measured all-in-VRAM rate. Fifty-two published measurements are scored against the same bound; their median is 13.6%.
2. **Where the seconds go.** Oracles run inside the engine with recorded routing. Section 4 crosses *what to cache* (the deployed policy's admissions or MIN's) with *how to load* it (CPU then background copy, which reads the expert twice, or a copy in the step, which reads it once), and nests a three-step prefetch under MIN. The 2×2 runs on hosts O4 and O5 and on a 10-host panel (job 099), and the gap is split into Shapley shares. The finding is that MIN's set pays only when each admission is read once. Whether the in-step copy pays depends on the host's ratio of link to CPU read rate. A calibrated time model, T = G + max(X_c/B_c, X_p/B_p, (X_c+X_p)/B_cp), with G frozen at the median of the other hosts' fits, is claimed to predict this from the bandwidth probe. Job 100 tests it prospectively on four launches.
3. **The price of foresight.** In the engine (panel) and on traces of nine models, a W-token window of exact or degraded routing recovers part of MIN's gain over admit-every-miss. The paper proposes that the horizon closing half the read gap is where the window holds about 0.65·C distinct experts per layer. Job 100 also runs the windows on the deployed read path, where they are "safe but small" (at most 1.12×). No online policy, learned admission order, speculative batch or trained forecaster comes close to the needed horizon.

The artifact is unusually complete. Every GPU job's predictions sit in its script header and were committed before launch, every number is macro-generated, and a supplement scores 1,054 prediction clauses.

---

## 2. Strengths

- **S1. Benchmarking discipline at a level I rarely see from applicants, or in published MLSys papers.**
  - Machines are rented and identified, and every job records the card's device read, memory clock and host probe.
  - Configurations run in a seeded shuffled order with a cold cache.
  - Ratios are paired by problem, with bootstrap intervals.
  - FreeToken's backend is selected on a separate launch.
  - The paper applies the Hoefler–Belli checklist explicitly (App. A).
  - Predictions are committed before launch. I verified the timestamps for jobs 099 and 100 (Section 6).
  - Failures are reported in the main text and scored in a supplement. This is the right culture.
- **S2. The machine is treated as the unit, and the author ran the panel to do so.** The panel shows that between-host spread dwarfs within-host problem resampling (SD 0.154 across hosts against a median half-width of 0.009 for MIN-with-one-read at gpt-oss 11%). Relaunches of three machines show that launches agree closely. This is exactly the experiment most systems papers skip.
- **S3. The 2×2 factorial with a worked example is a genuinely informative result.** At O4, gpt-oss 11%, changing either factor alone closes about 4% of the gap, and changing both closes 52%. The engine evidence that foresight must be spent "as MIN spends it" is clean, and it connects to the classical integrated prefetching-and-caching literature (Cao et al.).
- **S4. The bound is correct as stated, and its assumptions are listed and varied.** Table 16 varies no-evict versus exact, pooled slots, the median probe sample, the measured GPU rate and per-layer maxima. I recomputed two cells and they match (Section 6).
- **S5. Honest negative results.** The paper reports the learned admission order, speculative batching, linear and GRU forecasters, and the deployed-path windows. The limitations section is specific.
- **S6. The artifact reproduces.** `panel_099.py`, `job100.py` and `hostdep_model.py`, rerun in a scratch copy, regenerate their macro files byte for byte.

---

## 3. Weaknesses (ranked by how much they limit the claims), each with a concrete fix

**W1. The claim that the probe "predicts which way pays" rests on one machine, and the prospective test never contained a sign change.**
The abstract and Section 4 say the calibrated model predicts "whether MIN with one read in the step beats the deployed cache at all 39 host-budgets" and, from the probe alone, gets "16 of 16" signs right in job 100. The details undercut this:
- Only 2 of the 39 host-budgets are losses, and both are on one host: Pf, a Core Ultra 9 285K with link/CPU 0.29 (0.875 and 0.969). The trivial rule "always copy in the step" scores 37/39, the same as the one-path model the paper dismisses.
- In job 100, every fetch/base ratio is above 1 (1.048–1.484) and every aa/base ratio is below 1 (0.542–0.975). No job-100 host lies between 0.29 and 0.41, the region where the sign flips, so the 16/16 is uninformative about the decision.
- Where the model is near the crossover, it is biased toward the in-step copy. With frozen G it predicts fetch/base 1.17 on Pd at 11% against a measured 1.05; 1.36 against 1.16 on O5; and 1.36 against 1.17 on Pc. Its predicted crossover therefore sits at a lower link/CPU ratio than the real one.
- The probe-only magnitude test is weaker than "4 further launches, median 3.8%" suggests:
  - Two of the four launches are relaunches of panel machines (Pd, Ph). Their own G fits are inside the frozen median.
  - On the three hosts known before the amendment, the median error is 2.8%. On the two genuinely new machines it is 5.5%.
  - On one of those two (a 9950X behind a half-width link) the model is off by 16–20% on aa, w4 and w16 and fails its own preregistered 12% clause.

*Fix:* run the decisive experiment (Section 9). Rent hosts at link/CPU 0.25–0.5 and commit, from the probe alone, both the predicted sign and the predicted crossover ratio. Score against two naive baselines: "always copy" and the one-path model. Report sign accuracy only on cells where the baselines disagree.

**W2. The machine is the unit, but the intervals do not carry the launch.**
- Every within-host interval resamples problems only. The relaunches show launch-to-launch shifts that exceed those intervals. On Ph at gpt-oss 11% the relaunch moved the deployed time by 1.9% and shifted every ratio up together: aa by 0.017 against a combined half-width of 0.004, fetch by 0.024 against 0.011, both3p by 0.030 against 0.020. On Pd at 25%, both3p moved by 0.029.
- So a host-level claim at the 0.01–0.03 level is below the launch noise floor. Examples are b16/base = 1.00–1.03× at 11%, "safe" deployed-path windows, and clauses scored "held" because a problem-only interval excludes a threshold.
- The pooled crossed bootstrap is reasonable and done carefully. But with one launch per machine it folds launch variance into host variance without estimating it.

*Fix:*
- Estimate variance components (problem, launch within machine, machine) from the relaunch pairs: three now, more under Section 9.
- Inflate per-host intervals by the launch component.
- State the smallest effect a single launch can resolve, and mark claims below it as unresolved.

**W3. The price of foresight is measured against a baseline that is the slowest policy on half the hosts, and on the usable path the price buys little.**
- The window shares are fractions of MIN's gain over *admit every miss*. That is the best online policy in *reads*, but in *time* it runs 0.42–1.00× the deployed cache: every read goes over the link. A 4-token window is slower than the deployed cache on 6 of 10 hosts at 11%. The paper says this, but the headline "0.32–0.47 of the oracle's gain at W=4" still reads as a gain.
- On the deployed path (job 100) a 16-token exact window buys 1.00–1.03× at 11%. The realistic degraded window (8 tokens, recall 0.5) *loses* at 11% on all four hosts (0.943, 0.952, 0.960, 0.980). The main text omits this when it calls the spend "safe".
- The comparison of the two read paths uses 4 hosts, two on each side. The threshold of 0.5 was drawn after the panel. Nothing lies between 0.54 and 0.92. The calibrated model cannot price background copies (it misses them by a median 16%), so the deployed-path windows have no model prediction at all. Prediction 5 is a threshold rule, not the model.

*Fix:*
- Report the value of foresight against the deployed cache in time, as the headline, with admit-every-miss as a secondary reference for reads.
- State the recall-0.5 loss in the abstract or Section 5.
- Either extend the model to background copies, which share the link and DRAM with the step (the `bgmode` variant already in `hostdep_model.py` is a start), or drop "the model predicts which way pays" for the windows.

**W4. Focus and clarity (detailed in Section 5).** This is three papers in one: a cache system with a FreeToken comparison and a 52-row literature audit; an engine factorial with a time model; and a foresight study with a trace simulator. The main text is so number-dense that a reader outside offloading cannot extract the one idea per section. I rank this fourth only because it can be fixed without new data. At a general venue it is the most likely reason for rejection.

*Fix:* see Section 9, item 3.

**W5. The bound is a bound, but it is loose where the paper uses it, and "the rest" (30–62% of the gap) is unattributed.**
- At gpt-oss 25% the bound (430 tok/s) is 1.7× faster than the same card decoding with *every* weight in VRAM (255 tok/s). The headline "25–43% of the bound" therefore mixes host-bound cells with cells dominated by an unreachable datasheet GPU term.
- At the host-bound cells the best oracle reaches 46–73%, and the remainder is described qualitatively: imprecise prefetch, partial overlap, the GPU at half its datasheet rate. The two-path tightening explains only 1–7%. Whether the rest is slack in the bound or inefficiency in the engine is never measured.

*Fix:*
- Headline the bound with the measured all-in-VRAM GPU rate (31–56%), and keep the datasheet version as the outer envelope.
- Add an attainability microbenchmark: on one host, replay MIN's per-step transfer and CPU-read schedule (memcpy plus CPU streaming reads), overlapped with a GPU spin kernel of the measured per-step GPU time and no model compute. The time it achieves divides "the rest" into unreachable (bound slack) and reachable (engine).

**W6. "G" is not a constant of the GPU, so the model is calibrated rather than mechanistic, and its error summary is pseudo-replicated.**
- G is described as "the GPU's work and the step's fixed costs". With the same RTX 5090 and the same budget, it ranges from 4.08 to 9.22 ms at gpt-oss 11% (Pi, an i7-14700K with a slow host: 9.22; Pd: 6.16; median 4.63). It therefore absorbs a host-side term, presumably CPU per-token work and the helper shortfall, as App. C half-admits.
- With frozen G the deployed state is under-predicted by 13% on Pi and 9% on Pd.
- The 2.1% median pools 200 rows that are correlated within 14 hosts. Most of them are states whose bytes barely differ from aa (w1, w4, w8r5: median errors of 1.2–1.3%).
- The state that decides the sign, fetch, has a median error of 4.3% and a maximum of 18.3%. Per host, the median error is 9.8% on O5 and 7.9% on Pj.
- The sentence "the deployed state's included" is inaccurate: the 2.1% median is over the eight in-step states only. The deployed state's frozen error is separate, with a median of 2.4% and a maximum of 13%.

*Fix:*
- Add an explicit host term, such as per-token helper time measured from the engine's timers or a CPU-rate-dependent term, so that G becomes nearly host-invariant.
- Report errors per host (median of host medians, and the worst host) and per state.
- Correct the sentence.

**W7. The "one-constant" horizon rule uses the held-out model's own D(W) curve.**
`w50_lomo_boot.py` inverts the held-out model's measured D(W) to predict its W50. That consumes the held-out trace, which is far more information than the power law's C/k. With that trace the simulator gives W50 directly. Table 5's "1, and D(W)" discloses this, but the abstract and contributions present it as beating a two-constant law.

*Fix:* present D(W50) ≈ 0.65·C as a *unit* for stating a forecaster's target, not as a predictor. Alternatively, show that D(W) can be estimated cheaply (from a short prefix or router statistics) and test the rule using only that estimate.

**W8. The scorecard counts clauses, not predictive skill.**
- Clauses differ greatly in information. Examples: "counters equal job 099's" (deterministic), "host plan ≤150 µs", and wide bands such as fetch/base 1.10–1.55 and w1 ≤ 0.30, repeated across 10 hosts.
- "Held (point)" lumps two different situations: an interval that crosses the threshold, and no interval at all. In job 100, 122 of 216 clauses are "held (point)", mostly because no interval was computed for continuous quantities that could have one, such as probe-only errors and relaunch deltas.
- The panel's main result, host dependence of the in-step copy, was *not* predicted. Prediction 6 of job 099 ("fetch/base 1.10–1.55 on every host") failed on Pf and Pd. The paper does say so in App. B. This is the right attitude, but it means the headline finding is exploratory.

*Fix:*
- Separate the preregistered *substantive* hypotheses (perhaps a dozen per job) from the bookkeeping clauses.
- Score the substantive ones with a proper interval score or against naive baselines.
- Bootstrap every continuous clause.
- Label host dependence as discovered in job 099, then tested (partly) in job 100.

**W9. Minor consistency errors.**
- The caption of supplement Table 3 lists the hosts as "a, b, c, d (12400F)", but the rows are a, b, c and f (the 9950X behind a slower link). The `scripts/job100.py` docstring has the same stale list.
- "¿1" and "¡1" appear in the supplement where ">1" and "<1" are meant (a T1 font-encoding problem).
- In Table 3, brackets for panel rows mean ranges over hosts, but everywhere else brackets mean 95% intervals.
- Table 10 and Table 3 use slightly different ms for the same O4 states (20.59 against 20.64 ms for the deployed state), from different scripts.
- The appendix and Fig. 5 still call the model "the law" or "law's table", while App. C says "It is a calibrated model, not a law".
- Host f and prediction 9 of job 100 were added at 12:06Z, after hosts a, b and c had finished (11:17–11:54Z). The header says "amended before their launch", which is true, but the paper should say the replacement host was chosen with the first three results in hand.

---

## 4. Methodology and statistics

**The machine as unit.** The design is mostly right:
- one launch per host, cold cache per configuration, seeded shuffled order;
- within-host paired bootstrap over problems;
- a crossed two-stage bootstrap over hosts and problems for pooled quantities, correctly recognising that hosts and problems are crossed rather than nested (Kalibera and Jones);
- relaunches on three machines (O4 via 097a, Pd, Ph).

Three problems remain:

1. **Launch variance is unmodelled.** See W2. Launch shifts exceed problem-only intervals by up to 4× on Ph at 11%. The relaunch shifts are *common-mode*: all of Ph's ratios moved up together when its deployed time slowed by 1.9%. That is the signature of a launch effect on the denominator configuration. Ratios to a single baseline configuration are fragile to this.
   - Recommend a variance-components analysis: a mixed model with machine as a random effect, launch nested within machine, and problem crossed.
   - Recommend reporting each configuration's absolute time with launch-inflated uncertainty, as well as its ratio.
2. **The pooled mean is not a population quantity.** The panel is a convenience sample of Vast listings. The mean of fetch/base over hosts (1.22 [1.12, 1.30]) depends on how many slow-link hosts happened to be rented. Since the effect is driven by a measured covariate (log link/CPU, r = 0.93), the right summary is a regression with a leave-one-host-out prediction interval for a new host at a given ratio. With n = 10 the percentile bootstrap also under-covers slightly: a t-interval gives about [1.11, 1.33].
   - The correlations 0.93 and 0.87 are leveraged by the two slow-link hosts (0.80 and 0.70 without them). Report the slope with an interval.
3. **The time-model validation is not independent across rows** (W6). Summaries should be per host.

**The order-free accounting.** The arithmetic is sound. I reproduced the O4 gpt-oss 11% row from raw per-problem times. The worked example is the clearest paragraph in the paper. Three cautions:
- With two factors the Shapley value is just main effect plus half the interaction. When the interaction is 43 of 52 points, the "26% versus 25%" attribution is almost entirely the convention. The scientific statement is "the two choices are complements: neither helps alone". Lead with the interaction, and give the Shapley values second, as the paper partly does.
- "Order-free" applies only to the 2×2. The prefetch term is nested and "the rest" is a residual against a bound with known slack, so the full decomposition is a factorial plus a nesting, not an order-free attribution of the gap. Rename it ("a 2×2 factorial decomposition with a nested prefetch term").
- Appendix E's five-factor Shapley is over a *model*. Two of its terms are definitions and one is a per-cell residual. It is not evidence about the machine. Drop it or clearly separate it from the measured accounting.

**The calibrated time model: fitted where, tested where.**
- *Fitted:* G per host-cell on the deployed state, over 14 hosts (jobs 095–099).
- *Tested, retrospectively:* leave one host out on the same 14 hosts, with the model's form fixed earlier (jobs 069–082). The test is of G's transferability. Because it is run on the hosts that revealed the host dependence, it is not a blind test of the sign.
- *Tested, prospectively:* job 100, from the probe alone, with G frozen at 4.56/4.53 ms and counters fixed at the panel means. The median error is 3.8% over 32 predictions. The magnitude holds on machines that resemble the calibration set, and fails by 16–20% where the probe's link rate was not the rate the engine's copies achieved. The probe-only design is sound and its predictions were committed in time. Its weakness is coverage: there was no host in the crossover band, two of the four launches were known machines, and the planned faster-link 13900KF (100e) never ran.
- *Verdict:* predictive to about 5% for the magnitude of in-step states on hosts like the panel. It has not been shown to predict the *decision* near the crossover, and it says nothing about states with background copies. The weakest input is the link rate from the probe. A probe that mimics the engine's copy pattern (pinned buffers, 13 MB transfers, the engine's stream concurrency, concurrent CPU reads) would likely fix the 100f failure and the under-prediction for slow-link hosts.

**The bound.** Within its model class the bound is valid: exact routing, whole experts, at most C residents per layer, one token per pass, and rates no faster than the best probe and the datasheet. I checked the logic of Eq. (1): host reads are at least max(R\*, c)·S, GPU reads are at least D + (Lk − c)·S, overlap is perfect, and the minimum is taken over c. I also checked the published-bound variant: "exact", with eviction allowed after serving, which is the weaker and therefore valid constraint. Two scope caveats belong in the main text:
- R\* is defined on one routing trace. A system whose numerics change the greedy tokens (CPU experts with Q8 activations disagree on about 1.5% of steps) routes differently.
- With the measured GPU rate, the minimum over c puts more experts on the CPU than R\*, at memory speed. That is optimistic for MXFP4 matmuls on a desktop CPU, but harmless for a lower bound.

Tightness is the real issue (W5).

**Workload.** The workload is AIME prompts that were also used during development, teacher-forced, 20 or 30 problems, 256 tokens. The paper discloses this. Development reuse threatens the design choices (fetch tables, κ, half-lives) more than the measurements. A held-out prompt set for at least the panel's headline cells would remove the doubt cheaply (MATH-500 or GPQA, already used in App. I).

---

## 5. Clarity

**Score: 2 / 5.**

The prose is precise and honest, but it is written for the author. Problems:
- Nearly every sentence carries two to five numbers. The abstract alone carries about 20.
- The paper uses a private vocabulary without a glossary in the main text: "budget", "host-bound", "read path", "in the step", "deployed path", "fetch table", "the rest", "MIN, 2 reads", "foa", "aa", "both3p", "law".
- Figure 1 has 54 marker rows across 6 panels.
- Table 3 has 13 columns, with brackets that mean something different from everywhere else.
- Key results live in 21 pages of appendices: the time model, the accounting model, and the grid.

A reader from outside MoE offloading cannot tell from the abstract what the one thing to remember is.

**The three hardest passages, with suggested rewrites:**

1. *Abstract:*
   > "Knowing the future helps only if the cache uses it as MIN does: it caches only the experts MIN would cache, and loads each from host memory once, by a copy over PCIe, instead of running it on the CPU and then copying it. On hosts whose PCIe link reads at least half as fast as their CPU, doing both closes 26–60% of the gap to the bound; doing either alone closes at most 26%."

   Problems: "doing both" refers back across two clauses; "the gap" does not say from what; and "reads at least half as fast as their CPU" does not say the CPU *does what*.

   *Rewrite:* "Foresight helps only when the cache uses it the way the offline optimum (Belady's MIN) does, in two ways at once: it caches only the experts MIN would cache, and it moves each one with a single PCIe copy instead of first running it on the CPU and copying it afterwards. On machines whose PCIe link reads host memory at least half as fast as the CPU does, the two changes together close 26–60% of the gap between our deployed cache and the bound; either change alone closes at most 26%."

2. *Section 4:*
   > "What predicts the sign is the calibrated time model of Appendix C, T = G + max(X_c/B_c, X_p/B_p, (X_c+X_p)/B_cp): the three rates from the host's probe, the bytes the CPU and the link read from the state's counters, and G, the GPU's work and the step's fixed costs, frozen per budget at the median of the other hosts' fits (the fits range from 4.1 to 9.2 ms). It predicts the time of every state that reads in the step, the deployed state's included, within a median 2.1% (90th percentile 10%) on 39 host-budgets of 14 hosts, and whether MIN with one read in the step beats the deployed cache at all 39 of them."

   *Rewrite:* "We model time per token as T = G + max(X_c/B_c, X_p/B_p, (X_c+X_p)/B_cp). Here X_c is the bytes per token the CPU reads to run missed experts, X_p is the bytes copied over PCIe during the step, and B_c, B_p and B_cp are the host's CPU, PCIe and combined read rates from a 30-second probe. G is a per-budget constant for GPU work and fixed overhead. To test the model on a host it has not seen, we take G as the median of the other 13 hosts' fits. On those held-out hosts it predicts the eight in-step configurations within a median 2.1% (90th percentile 10%). It also predicts correctly whether copying in the step beats the deployed cache in all 39 host-budgets, of which 2, both on one host, are losses."

3. *Section 5, "In the engine":*
   > "The panel ran a family of policies between admit every miss and MIN (Table 1): the victim is the lowest-scored resident not seen in the next W tokens' routing, or, if every resident is seen, the one seen furthest ahead, and a miss is admitted unless the window says the victim is needed sooner. With no window this is admitting every miss, the best online policy in reads at these budgets (below); with the whole rest of the problem in view it is MIN."

   *Rewrite as a three-line algorithm box:* "Window policy W, for each miss m of the current token, per layer:
   1. Pick a victim v: the lowest-scored resident that does not appear in the next W tokens; if all appear, the one whose next use is furthest.
   2. Admit m in place of v, copying it over PCIe now, unless v is used again before m within the window, or m does not appear in the window while v does. Otherwise serve m on the CPU without admitting it.
   3. W = 0 is demand fetch (admit every miss); W = ∞ is MIN with bypass."

   Then give the measurement in a separate sentence.

**Further suggestions:**
- Put Table 1 and a five-line glossary on page 2.
- Allow at most two numbers per sentence in the main text, and move ranges to tables.
- Make Figure 2 (gain against link/CPU) the central figure of Section 4, and move Figure 1 to the appendix.
- Use one name for the model throughout.

---

## 6. Artifact checks

All reruns were done in a scratch copy of the repository with data symlinked read-only. Neither repository's tracked files changed (confirmed with `git status`).

| # | Claim | What I did | Result |
|---|---|---|---|
| 1 | **Job 099:** predictions committed before the machines started | `git log` for `jobs/099_panel@vast.sh` against `manifest.json` start times | **Held.** Commit 7b8327a at 07:18:48Z; first host launched 07:29:50Z, last 08:05:32Z. |
| 2 | **Job 099:** scorecard of 273 clauses (190 held, 42 held (point), 39 failed, 2 untested) and the panel macros | Reran `scripts/panel_099.py` | **Held.** The `wsg_panel.tex` it writes is byte-identical to the paper's, with identical clause statuses. 40 of the 42 "held (point)" clauses are "no interval". |
| 3 | **Job 099:** at gpt-oss 11%, MIN-with-one-read gain has mean 1.22 [1.12, 1.30], host SD 0.154, within-host median half-width 0.009 | Same rerun; read the crossed bootstrap code | **Held.** The bootstrap is correctly crossed. With n = 10, a t-interval is about [1.11, 1.33]. |
| 4 | **Job 099:** window shares at 11% are 0.14–0.17 (W=1), 0.32–0.47 (W=4), 0.78–0.93 (W=16), and 0.20–0.26 at recall 0.5, W=8; engine reads within 1.7% of the replay | Same rerun | **Held.** |
| 5 | **Job 099:** the copy in the step loses on the lowest-ratio host and "tracks the logarithm of the link-to-CPU ratio" | Per-host output; `fig_hostdep.py` | **Held as stated.** Both losses are on one machine (Pf: 0.875 and 0.969). Pd (0.39) still gains (1.051). The correlation of 0.93 falls to 0.80 without the two slow hosts. |
| 6 | **Job 100:** predictions in the header committed before launch | `git log` against manifests | **Held.** 752e1c3 at 10:25:59Z against the first launch (100c) at 10:26:44Z. The amendment adding hosts e and f and prediction 9 (718bf41) is at 12:06:35Z, against the 100f launch at 12:07:04Z. Hosts a, b and c had already finished (11:17–11:54Z), so the replacement host was chosen with their results in hand. |
| 7 | **Job 100:** probe-only median error 3.8% over 32 predictions | Recomputed independently from raw `ec_g_C*.jsonl` rows and each host's `predict_probe.json` | **Held** (3.83%). By host: Pd again −18.6% to +0.4% (fetch is the outlier); 5700X3D −4.7% to +7.3%; Ph again −9.1% to +0.6%; 9950X slower link +10.9% to +20.1% on aa, w4 and w16. Median 2.8% on a, b and c; 5.5% on the two new machines. |
| 8 | **Job 100:** "sign of every ratio … 16 of 16" | Same | **Held but uninformative.** Every fetch/base ratio is above 1 (1.048–1.484) and every aa/base ratio below 1 (0.542–0.975). No host was in the crossover band (lowest 0.41). |
| 9 | **Job 100:** Table 4 values | Recomputed the Pd-again and Ph-again rows from raw rows | **Held** (all within 0.005). Not stated in the text: b8r5 on the deployed path is 0.94–0.98 at 11% on all four hosts, and b4 is 0.960 at 25% on the 9950X with the slower link. |
| 10 | **Job 100:** relaunches reproduce the deployed time within 1.9% and every ratio within 0.030 | Reran `scripts/job100.py` (macros and `tab_window100.tex` byte-identical); compared deltas with within-host half-widths | **Held** (maximum 0.0296, Ph both3p at 11%; base within 1.88%). The deltas are up to 4.2× the combined problem-only half-widths (Ph, aa, 11%), so a launch component exists. |
| 11 | **Time model:** frozen-G median 2.1% (90th percentile 10%) on 39 host-budgets of 14 hosts; 39/39 signs; one-path model 37/39 | Reran `scripts/hostdep_model.py` | **Numbers held; one statement wrong.** The 2.1% median excludes the deployed state, whose median is 2.4% and maximum 13% (Pi). "Always copy" also scores 37/39. G ranges from 4.08 to 9.22 ms at one budget on one GPU model. The fetch state's error is 4.3% median and 18.3% maximum. The model over-predicts fetch/base on slower links (Pd 1.17 against 1.05). |
| 12 | **Time model:** frozen constants of the probe-only test | Read `jobs/ec2/predict_100.py` | **Held.** G is 4.56/4.53 ms (the median of the 14 hosts, which include Pd and Ph). Counters are the panel means, and the P1 clauses confirm they are host-invariant within 0.1%. |
| 13 | **Accounting, Table 3, row O4 gpt-oss 11%:** gap 10.6 ms; load alone 4, cache alone 4, interaction 43, Shapley 26/25, prefetch 13, rest 36 | Recomputed from raw job 096a per-problem rows | **Held** (10.62 ms; 3.9, 4.5, 43.2, 26.1, 25.5, 12.7, 35.7). Table 10 lists slightly different ms for the same states (20.59 against 20.64). |
| 14 | **Bound:** host B gpt-oss 11% = 172 tok/s and 25% = 430 tok/s; with the measured GPU rate, 279 | Recomputed from `prereg/speed_limit_v2.json` (R\* = 38.32 and 15.34, S = 13.25 MB, B_host = 87.5 GB/s, D = 1.714 GB) | **Held** (172.3, 430, 279). Note that 430 tok/s exceeds the all-in-VRAM speed (255). |
| 15 | **Horizon rule:** the one-constant rule predicts a held-out model as well as the two-constant power law | Read `scripts/w50_lomo_boot.py` | **Held as computed, but the rule uses the held-out model's own D(W) curve** (see W7). |
| 16 | **Supplement:** consistency | Read the job 100 table | **Minor errors:** the caption names host d (12400F) while the rows are host f; "¿1"/"¡1" glyph errors. |

---

## 7. Scores

| Item | Score |
|---|---|
| Overall | **5 / 10** (borderline; reject in its current form at MLSys, likely accept after a focused rewrite and the crossover experiment) |
| Confidence | **4 / 5** |
| Clarity | **2 / 5** |
| Soundness | **3 / 5** (measurements and bound sound; predictive and statistical claims overstated at the margins) |
| Novelty | **3 / 5** (the bound is incremental over WiSP, Budgeting Bytes and Zhang's MIN gap; the in-engine factorial, the host panel and the distinct-expert unit are new for this sub-area) |
| Significance | **3 / 5** (useful design rules for consumer MoE offloading; narrow scope of batch 1, two models and RTX 5090 desktops) |

---

## 8. Would I take this student?

Yes. Predictions committed before every rental, over a thousand clauses scored with the failures in plain view, every number regenerated by a script I could rerun, and an unprompted Hoefler–Belli checklist: that discipline is rarer than the systems skill, which the 3,500-line llama.cpp patch also shows. What this student needs from a group is restraint and statistical modelling, not rigour: one claim per paper, fewer numbers per sentence, variance components and covariate regressions in place of more clauses, and designing the decisive experiment (here, hosts at the crossover) before the confirmatory ones; all of that is teachable in a first year, while the honesty and stamina visible in this artifact are not.

---

## 9. The changes that would most raise the score (about three weeks, about $8 of GPU rental)

The ledger implies about $0.80 per rental-hour on Vast. $8 buys roughly 10 one-hour rentals.

1. **A prospective crossover test, with launch replication. About $6–7, 8 rentals.** This raises soundness, and with it the overall score by one point.
   - Rent 5 RTX 5090 hosts with probed link/CPU between 0.25 and 0.5. Intel 285K, 13900K or 14900K hosts with fast DDR5 behind x8 or Gen4 slots are the likely candidates; screen by the listing's PCIe figure.
   - Run only gpt-oss at 11% and 25%, with base, fetch, both3p, aa, b16 and w16 on 20 problems (about 45 minutes each).
   - Before each launch, commit from the probe alone: the predicted sign and magnitude of fetch/base, the predicted link/CPU crossover, and b16 against w16. Do this with the current probe and with a second probe that mimics the engine's copies (pinned 13 MB transfers at the engine's concurrency, with concurrent CPU reads).
   - Relaunch 3 of the 5 hosts once to estimate the launch variance component.
   - Score against "always copy" and the one-path model, on the cells where they disagree.

2. **Statistical re-analysis and a bound-attainability microbenchmark. About $1 and one rental, otherwise analysis.** This raises soundness and makes "the rest" interpretable.
   - Fit a mixed model (machine random, launch nested, problem crossed) to all panel and relaunch data. Report launch-inflated per-host intervals, and the smallest effect one launch can resolve.
   - Replace pooled means over hosts with a regression on log(link/CPU) and leave-one-host-out prediction intervals.
   - Report time-model errors per host and per state, and fix the "deployed state's included" sentence.
   - On one of the rentals from item 1, replay MIN's per-step transfer and CPU-read schedule against a GPU spin kernel of the measured step time, to split the 30–62% "rest" into bound slack and engine inefficiency.

3. **A rewrite around one question. $0.** This raises clarity from 2 to 3–4, and the overall score by about one point.
   - Make the paper about Section 4: where the seconds go, and when foresight pays, as a function of the host.
   - Keep Section 3 as the yardstick, at one paragraph plus Table 2 with the measured-GPU bound as the headline.
   - Cut Section 5 to its robust parts: the deployed-path result including the recall-0.5 loss, and D(W50) ≈ 0.65·C stated as a unit.
   - Move the 52-row audit, the FreeToken grid, the learned order and the five-factor model Shapley to appendices or a separate paper.
   - Allow at most five numbers in the abstract.
   - Add a glossary and Table 1 on page 2, and one name for the model.
   - Fix the minor errors in W9.
