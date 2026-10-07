# Review 11b: "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth" (MLSys 2027, main track)

Reviewer profile: performance modelling and scientific benchmarking of parallel and ML systems.

## Materials read

- `paper/paper.pdf`: all 35 pages, including Appendices A to M. I read the prose from `paper.tex`, `app_wsg.tex`, `app_more.tex`, `app_traces.tex` and `app_value.tex`, and the rendered tables (1–3, 5–23) and figures from the PDF.
- `paper/supplement.pdf`: all 38 pages. I read the scoring rule and Tables 1 (jobs 073–098) and 6–9 (jobs 103–106) clause by clause, and skimmed Tables 2–5 (jobs 099–102).
- Artifact code, read only to learn definitions: `scripts/job106.py`, the helpers in `scripts/panel_099.py`, `scripts/reanalysis.py`, `scripts/sumlaw_paper.py`, parts of `scripts/speed_limit.py`, `bandwidths()` in `gpu-branch/jobs/ec2/fetch_table.py`, and the header of `scripts/w50_rebaseline.py`.
- Prereg JSON: `w50_distinct.json`, `linkaware.json`, `scorecard_099/100/101/106.json`.
- Raw data on the gpu branch:
  - job scripts and headers for 104, 105, 106 and 106a–e;
  - result directories for 069c, 081, 084c, 093–106 (rows, engine counters, probes, Nsight summaries and tail traces, `nvidia-smi -q`, manifests);
  - routing traces 084c (AIME, gpt-oss-120b) and the S-arm traces of gpt-oss-120b, Qwen3-30B-A3B (FP8), DeepSeek-V2-Lite and Mixtral-8x7B.

## Independence statement

I did not open anything under `reports/` except to write this file. I did not open any `prereg/*outcome*.md`, anything under `research_notes/`, or any file named like a review, plan, number check or progress log. On those grounds I also avoided:

- `paper/paper_v1_prereview.tex` and `apply/APPLICATION_PLAN.md`;
- the job script `jobs/082_review_run1@vast.sh`;
- `jobs/ec2/minadm_plan.py` and the `plan_g*.{txt,bin,json}` result files.

My automated greps for GPU UUIDs and device-read bandwidth did pass through the result directory `082_review_run1@vast` (its `nvidia-smi-q.txt` and `bw*.txt`); no file there is named like a review.

I did not read commit messages. I used only `git log --format='%h %ad'` to order commits against job start times, and `git diff` of the job-script *content* between two commits to see what an amendment changed.

All recomputation used my own code in the scratchpad (`review11b/r_common.py`, `r_job106.py`, `r_found.py`, `r_min.py`, `r_minadm_run.py`, `r_horizon.py`, plus inline snippets), run on a copy of the repository without `reports/`, `research_notes/` or the outcome notes. I wrote my own MIN-with-bypass simulator, a min-cost-flow LP (scipy HiGHS) for the fewest-admission hit-optimal schedule, a window-policy simulator, the Eq. (3) solver, and the bootstraps. I modified nothing in either repository; the uncommitted change to `gpu/vast_ledger.json` in the main checkout predates my work.

## 1. Summary

The paper studies batch-1 decode of two MoE models (gpt-oss-120b in MXFP4; Qwen3-30B-A3B in BF16) whose experts mostly live in host DRAM, on rented RTX 5090 hosts. It has five parts.

1. **Two lower bounds on time per token.** Eq. (1) is a roofline-style bound in which host memory must supply at least MIN-with-bypass's reads R* at the highest probed host read rate B_host. Eq. (2) is a tighter bound for systems that read on demand: the GPU's non-expert time T_GPU in series with R*·S/B_host.
2. **A relation for the authors' llama.cpp expert cache.** Eq. (3) is T = G + (M + (1 − G/T)·A)·S/B_host. G is the profiled GPU kernel time; M and A are the counted misses and background admissions. It is described as "nothing fitted". The plain form (without 1 − G/T) was found on 30 exploratory launches and failed its registered test at gpt-oss 25% (job 105). The overlap form was then derived on 34 launches and registered for job 106.
3. **In-engine oracles of future routing.** MIN with one read, MIN prefetched, Belady, and windows show that foresight pays when spent as MIN spends it. Among MIN's hit-optimal schedules, the fewest-admission one beats the greedy one where the PCIe link is slow. An online rule carrying this lesson (a larger admission margin κ) gains 0–6%.
4. **A horizon rule.** Half of MIN's saving over "admit every miss" needs a window containing about 0.65·C distinct experts per layer.
5. **A system comparison and a pre-registration record.** Every prediction was committed before the machine started, and failures are reported in a 38-page supplement.

The artifact is exceptional, and every central number I recomputed reproduces. My concerns are about inference, not arithmetic:

- The registered test of the headline relation failed as specified. Its reported success is a post hoc subset of three machines that all contributed to the data from which the relation was derived.
- "Nothing fitted" hides a specification search that the registered test cannot discriminate.
- The "bound" uses a probed rate that the engine itself exceeds on a sizeable minority of runs.
- Several universal claims ("on every machine that ran stably") rest on two to seven machines.

## 2. Strengths

1. **Pre-registration with verifiable ordering.** I checked jobs 104, 105 and 106. Each job script was committed before every one of its hosts started:
   - job 104: 17:33:12 UTC, against host starts from 17:34:33;
   - job 106: 01:43:48 and 01:44:13, against starts from 01:44:57.

   Job 105's second commit (22:09:26) came after three of its four hosts had started, but the diff only adds host substitutions; the predictions are unchanged. Amendments are written into the headers. Clauses are scored by machine, including the failures (48 of 181 clauses failed in job 106). This is far above MLSys norms.
2. **Every number I checked reproduces from raw data** (Section 4; 29 checks). This includes Table 2's paired bootstrap ratios to the third decimal, job 102's 86–96%, and the 12.4 fewest admissions per token from my own LP.
3. **The machine is treated as the unit.** Launches are deduplicated by GPU UUID; I confirmed, for example, that host 106e shares its UUID with panel host 099e. Relaunch stability is measured: base time moved by at most 3.2% for the same machine and by up to 29% for the same CPU model. Cross-machine Spearman intervals resample machines (n = 19), not launches (n = 28). Variance is decomposed into machines (96%) and problems (1–3%).
4. **Bounds come with attainability evidence.** A model-free replay of MIN's per-layer reads reaches 86–96% of the bound's read time (job 102; I reproduced every cell of Table 21). The paper also states what the bound drops: pooled slots, latencies, and the GPU term at datasheet rate.
5. **The "admissions cost reads" insight is real and replicated.** The fewest-admission set beats the greedy set on the two slow-link machines in five separate launches: Pf in 104a, 105e and 106b; the Threadripper 9960X in 105b and 106a. It gains over the deployed cache on all 10 stable launches.
6. **Failures are generally disclosed.** Examples are the failure of the plain law in job 105, the failure of the dk ≥ 1.01 clause, and the learned order losing where memory is fast. The text also concedes that "predicting no change does almost as well (0.016)" for the law's ratio predictions.

## 3. Weaknesses (most important first)

### W1. The registered test of Eq. (3) failed as registered. The reported success is a post hoc subset of machines that were not new.

The job 106 header (prediction 2) registered that "the overlap form is within 6% … at every host, model and budget, and the median |error| over all of them is at most 4%". As specified, the test failed:

- my recomputation gives **12 of 20** cells within 6% and a pooled median |error| of **4.7%**;
- both server machines missed by 10–52% in every cell.

The paper's abstract and introduction lead with "held within 6% … on three machines rented before". All three of those machines are part of the 34-launch exploratory set on which the overlap term was derived:

- Pf appears as 099f, 101a, 103a, 104a and 105e;
- the Threadripper 9960X as 105b;
- Pe as 099e (same GPU UUID as 106e).

So the registered evidence for the relation is a temporal replication on derivation machines. The only out-of-sample machines both failed.

The exclusion of those two machines was not pre-registered. Prediction 8 ("base varies ≤ 2% between rounds") is a prediction, not a gate, and the job has no output-correctness gate. More importantly, **instability does not explain the Xeon misses**:

- In round 1 at gpt-oss 11% the Xeon 8347C was stable: base 14.52 ms, per-problem CV 5.2%, comparable to Pf's 4.3–4.9%. Eq. (3) predicts 12.03 ms (**−17%**).
- At gpt-oss 25% its two rounds agree within 2.8% (10.23 and 9.96 ms). The law misses by **−15%**.

On the EPYC 7302, the wrong outputs (NLL 0.208 in rounds 1–2, 0.42 afterwards, against 0.190 elsewhere) have a plausible effect on the oracle configurations, whose reads relative to base rise to 0.68 against about 0.58 elsewhere. But base itself has 59.7 misses per token, similar to the other hosts, and the law misses it by −29% to −31%.

The statement "we cannot tell how much of that is the relation and how much the machines" is therefore too generous to the relation. On the Xeon, the data say the relation misses even when the machine runs stably. Together with the earlier calibrated-model failures on the EPYC 9655 and the hybrid-core host, the evidence supports Eq. (3) for desktop and workstation CPUs, not for "the machine" in general.

### W2. "Nothing fitted" overstates the evidence: the specification was selected on the exploratory data, and the registered test cannot discriminate it from alternatives.

No continuous parameter is regressed, but three discrete choices were made with the exploratory data in hand:

1. B_host is the maximum over every probe line, including CPU reads at thread counts above the engine's helper count and concurrent CPU+PCIe sums.
2. G is the sum of all kernel categories except the helper wait and the copies.
3. The overlap term 1 − G/T was introduced after the plain form failed at 25%.

I re-scored alternatives on the same data.

| Specification | 34 exploratory launches, gpt-oss 25% (within 6%) | Job 106, 12 stable cells (within 6%) | Median \|error\| (job 106 stable) |
|---|---|---|---|
| Registered overlap form, B_host | 33/34 | 12/12 | 2.3% |
| Constant half-discount, G + (M + A/2)·S/B_host | 32/34 | 11/12 | 1.9% |
| Misses only, G + M·S/B_host | 21/34 | 10/12 | 4.0% |
| Plain form | 18/34 | 8/12 | 4.3% |
| Overlap form with B_cp (median concurrent probe) | 23/34 | 8/12 | 5.6% |
| Overlap form with B_c at the helper count | 12/34 | 6/12 | 6.1% |

The bandwidth choice carries the result. The overlap mechanism is not distinguished from a constant 0.5 discount on admissions.

The pass is also fragile within the inputs' own uncertainty:

- Using the median of the six concurrent probe samples instead of the maximum (2.1–3.0% lower on the three stable hosts) gives 9/12 within 6%, max 7.3%.
- Raising G by 5% gives 9/12, max 8.6%.
- The paper's own Table 20 reports that median-versus-highest probe sample moves host B's bound by 8%.

The ±6% criterion is about the size of the measurement uncertainty in B_host and G, which carry no intervals. I would call Eq. (3) "a relation with no continuous free parameters whose specification was chosen on 34 exploratory launches", and present a specification curve and prediction intervals.

### W3. B_host is not a ceiling, so Eq. (1) is a bound "at the probed rate", not a lower bound for any system.

Using the paper's own accounting (profiled G, counted reads), I computed the engine's implied host-read rate, (M + A(1 − G/T))·S/(T − G), across the 34 exploratory launches at both budgets:

- under the overlap accounting it exceeds B_host by more than 5% in **13 of 68** launch-cells, up to **1.22×** (100f, a 9950X at gpt-oss 25%); the median is 1.00;
- under the plain accounting, where every counted read costs host time, it exceeds B_host in **50 of 68** cells, up to 1.30×.

A system reading MIN's R* at the rate the engine demonstrably achieves would beat "the bound" on those machines. "It reads at the machine's best rate" and "this rate bounds every system" are in tension. In the data, B_host behaves like a central estimate of attainable bandwidth, not a ceiling.

The same issue affects the GPU term on the headline machine. Host B's card measures 1847 GB/s device read (memory clock 17001 MHz), above the 1792 GB/s datasheet rate used as B_gpu. This only matters in the GPU-involved cells (40% and 43.75%), but there Table 2's "datasheet" bound is not a bound.

The Limitations section concedes that "a faster reader than our probe could exceed them". The paper's own engine is such a reader on some machines, and the efficiency figures (38–54% of Eq. (1), 55–70% of Eq. (2)) inherit this.

### W4. Universal claims rest on very few machines and on classes defined after the fact.

"The fewest-admission one gains on every machine that ran stably" covers 7 machines. The slow-link regime that the abstract highlights ("also wins where the PCIe link is slow") rests on **2 machines** with link-to-CPU ratios of 0.28–0.32. The slowest-link machine tested, the EPYC 7302 (ratio 0.14, 13 GB/s link):

- *lost* in all three rounds (fetchplan/base 0.834, 0.878, 0.824; my round-then-problem bootstrap 0.846 [0.819, 0.877]);
- failed registered clauses P6 and P7 there;
- was excluded post hoc.

The exclusion is mechanistically defensible: with wrong outputs, its routing likely diverged from the oracle's trace. Even so, the abstract's statement should be scoped to "on two machines with ratios 0.28–0.32", and a functioning machine below ratio 0.25 is needed before claiming the slow-link regime.

Similarly, "admit less gains 0–6%" is a range over three machines and two budgets with one launch each, with no population-level estimate or interval. The paper should report a hierarchical estimate (machine as a random effect) with its interval, or state the claims as "on these k machines".

### W5. Rounds and launches are not modelled appropriately.

- **Degenerate top-level resampling.** Round-level intervals bootstrap 3 rounds (gpt-oss 11%) or 2 rounds (gpt-oss 25%). That yields only 10 or 3 distinct top-level resamples, so the percentile intervals reflect problem-level variance plus a handful of round configurations. On the Xeon this gives [0.987, 1.557] for dk from rounds 0.984, 1.489 and 1.351: that is the range of the rounds, not an interval.
- **Rounds are not launches.** Rounds are processes within one rental. Launch-to-launch variation of the *paired* ratio is of the same size as the online-rule effects:
  - Pf's fetchplan/base was 1.035, 1.028 and 1.019 across 104a, 105e and 106b (a 1.6-point spread);
  - the Threadripper's was 1.039 and 1.056 (1.7 points);
  - the dk rule's gains at gpt-oss 11% are 0.3–2.4 points, each from a single launch per machine.

  Table 3's intervals such as Pe dk 1.010 [1.009, 1.011] are therefore not intervals for the claim "this rule gains 1% on this machine".
- **What to do.** Kalibera and Jones (cited) prescribe a variance-components pilot (machines × launches × rounds × problems) to size repetitions at each level. The paper has the infrastructure and should do it.

### W6. The horizon rule is weaker than stated.

From `w50_distinct.json`:

1. **Five of the 26 points have W50 below one token.** There the code sets D(W50) = k·W50, so at C/k = 1 the statistic D(W50)/C equals W50 by construction.
2. **D(W50)/C falls with C/k within every model**, for example:
   - Qwen1.5-MoE: 0.99, 0.80, 0.62;
   - Qwen2-57B: 0.91, 0.86, 0.60;
   - DeepSeek-V2-Lite: 0.84, 0.76, 0.63.

   The "constant" is a pooled median over a systematic trend.
3. **Leave-one-model-out, the rule is not better than a two-constant power law in tokens** (median factor 1.16 versus 1.19; the paper says "not distinguished").

My independent simulation of the window policy described in Appendix I (8 points from gpt-oss-120b, DeepSeek-V2-Lite and Mixtral) gives W50 values 5–40% longer than the paper's for every point:

- D(W50)/C of 0.65–0.77 at C/k ≥ 4 (the paper: 0.62–0.72 for the same points);
- 0.60–1.10 at C/k ≤ 2.7 (the paper: 0.54–0.84).

The statistic is sensitive to policy details at small C/k. "≈0.65·C" is a fair description at large budgets and a loose one at small budgets.

### W7. Failed predictions are not always reported where the claim is made.

The supplement is complete, but several main-text claims sit on failed registered clauses without saying so at that point:

- **Section 5, "copies 0.60–0.73 times as many experts".** The registered ≤ 0.70 failed at 25% (0.726 on all three job 104 hosts).
- **Section 5, "gains over the deployed cache on every one".** Registered P6 and P7 failed on the EPYC 7302. This is stated in Section 4 as an exclusion, not here.
- **Section 6, "exact windows of 4 and 16 tokens recover 0.32–0.47 and 0.78–0.93".** Both lower ends are failed registered bands: w4 at 0.32 and 0.34 against 0.35–0.70, and w16 at 0.78 against ≥ 0.80, on Pf and Pd.
- **Abstract and introduction.** They present the relation as having "held" without saying that the registered test, as specified, failed.

More generally, the prediction record should appear in the main text as a calibration statement. In jobs 088 onward, 168 of 473 clauses held with an interval and 90 of 301 bands failed. On job 106's stable hosts, 16 clauses held with an interval and 77 held only on the point estimate. The pre-registration is valuable precisely because it shows how often the authors' quantitative expectations were right, and that number belongs next to the claims.

### W8. Environment control and representativeness are weak, and partly acknowledged.

- **Unlocked clocks.** `clock-lock.txt` on every host reads "does not have permission to change clocks". Power limits differ, and the headline card's memory is overclocked.
- **Shared hosts.** The two excluded hosts had 131–132 GB of memory in use before download (multi-tenancy), with cgroup CPU limits.
- **Narrow workload.** It is one task family, AIME-25, which was also used during development: 20 problems × 256 teacher-forced tokens in the engine runs and 30 in the system comparison. Only two models are timed, at batch 1. The nine-model traces are used only offline.
- **Profiling conditions.** G is profiled under Nsight on one problem (20 tokens). The profiled runs' wall time is 20–35% above the unprofiled time on several hosts (for example 18.7 against 13.9 ms on Pf at 11%). The paper does not show that kernel durations, as opposed to gaps, are unaffected.

### W9. Clarity.

The paper is very hard to read. It uses dozens of host aliases (host A, B and S; O1–O6; Pa–Pj; "Pf again"), job numbers inside the main argument, and overloaded terms ("fetch", "admit", "1 read" and "2 reads", "greedy", "fewest"). Many main-text sentences pack three or four quantitative claims with cross-references.

Two different time models coexist:

- Eq. (3), with profiled G and B_host;
- Eq. (4), with a fitted G = 5.17 ms, larger than the whole all-in-VRAM token, and per-path rates.

They are not reconciled beyond a note that G and the rate factor are collinear. That collinearity is exactly the identifiability issue behind W2 and W3.

Smaller points: table numbering jumps from 3 to 5. Table 4 appears to be the uncaptioned checklist longtable in Appendix A. The "39%" round variation of the Xeon is (max − min)/mean; I get 52% as (max − min)/min, so the definition should be stated.

### W10. Significance is moderate.

The bound is a roofline with Belady's MIN as the traffic term, which is classical. The online payoff of the main lesson is 0–6%, and at most cells 1–2%. The fewest-admission gains require perfect future knowledge, and no tested forecaster approaches the needed horizon. The paper's main value is methodological: a careful accounting of where time goes, and a model for how to run pre-registered systems experiments. That is worth publishing, but the claims should be scoped to what the evidence supports.

## 4. Claims checked (independent recomputation)

"✓" means reproduced within rounding.

| # | Claim | Where | My value (own code) | Verdict |
|---|---|---|---|---|
| 1 | Eq. (1) on host B: 172 / 430 tok/s at gpt-oss 11% / 25% | Table 2, 20 | R* = 38.32 / 15.34 per token (exact MIN with bypass, 30 problems in sequence); B_host = 87.5 GB/s; 172 / 430 tok/s (no-evict: 170 / 427) | ✓ |
| 2 | Eq. (1) on host S: 140 / 351 tok/s | Table 2 | 140 / 351 (B_host 71.3) | ✓ |
| 3 | O4 example: Eq. (1) 10.0 ms, Eq. (2) 13.0 ms | §5 | 10.02 ms; 10.02 + min T_GPU 2.94 = 12.96 ms | ✓ |
| 4 | T_GPU 2.9–3.4 ms on 7 hosts | §3 | 2.94–3.38 ms (069c + 105 profiles) | ✓ |
| 5 | Deployed cache at 38–54% of Eq. (1), 55–70% of Eq. (2), gpt-oss 11% | §3 | 39–55% and 56–71% over 30 launches (no-evict R*) | ✓ (±1 pt) |
| 6 | Job 102: 86–96% of the bound's read time per layer | §3, Table 21 | 86, 89, 96, 96, 95, 93% (my B_host and term) | ✓ exact |
| 7 | Fewest admissions 12.4 per token at gpt-oss 11% (greedy 21.8) | §3 | My LP: 12.40 (hits equal Belady's on every layer); 6.75 at 25%. Greedy by my tie rule: 21.02 (10.45 at 25%) | ✓ fewest; ≈ greedy (definitional) |
| 8 | Link-aware bound raises the bound ≤ 1% / ≤ 7% | §3 | `linkaware.json` covers only 24 hosts of jobs 095–102; the slowest-link hosts of 105/106 (ratios 0.14–0.32) are not included | Scope not stated |
| 9 | Plain form within 6% on 28 of 30 launches at 11%; over-predicts at 25% (median 6%) | §4 | 28/30; median +5.9% | ✓ |
| 10 | Overlap form: 33 vs 18 within 6% at 25%; 29 vs 32 at 11% (34 launches) | §4 | 33 vs 18; 29 vs 32 | ✓ |
| 11 | Job 106 stable hosts: 12/12 within 6%, median 2.3%, max 6.0%; plain max 10.1% | §4, Table 3 | 12/12; 2.34%; 5.99%; 10.08% | ✓ |
| 12 | Job 106 all hosts: pooled median 4.7% vs 4% registered; unstable 10–52% | §4 | 4.67%; 10.2–52.4%; 12/20 cells within 6% | ✓ (registered test failed as specified) |
| 13 | G 4.1–4.7 ms (gpt-oss), 5.4–5.9 ms (Qwen3) on stable 106 hosts | §4 | Same from the summaries; my own sum over the raw Nsight tail traces matches within 1.5% | ✓ |
| 14 | Elasticity of (T − G) to B_host = −0.95 [−1.06, −0.84] | §4 | −0.95 [−1.06, −0.84] (bootstrap over 21 machines) | ✓ |
| 15 | Fig. 1 shares at gpt-oss 11%: GPU 12–47%, MIN reads 31–54%, beyond 22–35%; G 23–68% of gap | §4 | 12–47%, 31–54%, 22–41% (22–34% of law time), 23–68% (23 machines, first launch) | ✓ (≈ for "beyond") |
| 16 | MIN in step 16–51%, prefetched 25–81% over deployed (O3–O5, six budgets) | §5 | 1.160–1.514; 1.253–1.812 from raw rows | ✓ |
| 17 | Fewest-admission set 1.02–1.34× over deployed on 10 stable launches / 7 machines | §5 | 1.019–1.342 (104a–c, 105a/b/e/f, 106a/b/e) | ✓ |
| 18 | +0.08–0.17 over greedy on 5 machines (ratio 0.28–0.63); slow links: greedy 0.85–0.93, fewest 1.02–1.06; round CIs start at 1.01, 1.05 | §5 | 0.082–0.172; 0.853–0.933 / 1.019–1.056; 1.013, 1.049. EPYC 7302 (ratio 0.14, excluded): fewest 0.846 [0.819, 0.877] | ✓ (exclusion matters) |
| 19 | Copy ratio 0.60–0.73; misses +2.5–5.3% | §5, App. J | 0.597–0.726; 1.025–1.053 | ✓ (0.726 failed registered ≤ 0.70) |
| 20 | dk 1.00–1.02× (11%), 1.01–1.06× (25%), Qwen3 1.01–1.04×; reads −4–6% / −6–10% | §5, Table 3 | 1.003 / 1.024 / 1.010; 1.019 / 1.057 / 1.008; 1.010–1.038; −4.4 to −5.8%; −6.3 to −10.5% | ✓ |
| 21 | Learned order 0.98–1.05×, 7–14% fewer reads, 267–439 µs | §5 | 0.982–1.055; 6.6–14.2%; 267–439 µs | ✓ |
| 22 | Spearman of MIN-in-step gain vs link/CPU ratio = 0.89 [0.63, 0.97], 19 machines; paced 0.87 (17), aa 0.95 (13), bypass vs B_p 0.79 (17) | §5, App. J | 0.89 [0.63, 0.98], n = 19 machines (launch-level would be n = 28, ρ = 0.92); 0.87, 0.95, 0.79 | ✓ machine is the unit |
| 23 | Machines carry 96% of log-gain variance, problems 1–3%; Kendall's W 0.77 / 0.82 | §5, App. J | 96.3% / 1.1% (11%), 96.2% / 2.8% (25%); W 0.77 / 0.82 | ✓ |
| 24 | Relaunch within 1.9% (101) and 3.2% (106); same CPU model differs 16–29% | §2 | Max relaunch drift 3.2% (TR 105b→106a), Pf 099f→101a 1.7%; 285K machines 16–29% | ✓ |
| 25 | Table 2, host B gpt-oss: ours/FreeToken 1.294 [1.278, 1.312], 1.275, 1.154; 2.00 / 2.73 / 3.20× llama.cpp | Table 2 | 1.294 [1.278, 1.312], 1.275 [1.255, 1.295], 1.154 [1.134, 1.173]; 2.00 / 2.73 / 3.20 | ✓ |
| 26 | W50 ranges 0.6–34 tokens; D(W50) ≈ 0.65·C (quartiles 0.61–0.72) | §6 | From the JSON: 0.64–33.6; median 0.651; 5 of 26 points are sub-token interpolations; median 0.63 without them. My own simulation: gpt-oss-120b 0.77 / 0.69 / 0.65, DeepSeek 1.10 / 0.81 / 0.65, Mixtral 0.84 / 0.60; my W50 values are 5–40% longer | Partially reproduced (close at C/k ≥ 4, 0.65–0.77; loose below) |
| 27 | Job 106 tally: 181 clauses, 48 failed, 36 on the two excluded hosts, 11 on stable hosts | App. B | 181 / 48 / 36 / 11 (+1 pooled) | ✓ |
| 28 | Predictions committed before each machine started (104–106) | App. B, M | Commit times precede every host start; 105's late commit only changes hosts | ✓ |
| 29 | Host 106e is "panel host Pe, rented before" | §4 | Same GPU UUID as 099e | ✓ |

New findings not stated in the paper:

- **Xeon, stable conditions.** Eq. (3) misses by −17% in the Xeon's stable round 1 at gpt-oss 11% and by −15% at gpt-oss 25% (W1).
- **Engine exceeds B_host.** The engine's implied read rate exceeds B_host by more than 5% in 13 of 68 exploratory launch-cells (W3).
- **Alternative forms pass too.** A constant half-discount scores 11/12 and misses-only 10/12 on the registered cells (W2).
- **Small input changes break the pass.** A 2–3% lower B_host or a 5% higher G turns 12/12 into 9/12 (W2).
- **Above-datasheet GPU on host B.** Host B's card reads 1847 GB/s, above the 1792 GB/s used as the GPU ceiling (W3).

## 5. Methodology against scientific-benchmarking practice

Rules are those of Hoefler and Belli; the multi-level repetition guidance is from Kalibera and Jones.

- **Rules 1 and 4 (base case; do not summarise ratios): met.** Absolute speeds sit next to every ratio. Ratios are of means, paired by problem, and not averaged across cells.
- **Rule 2 (subsets): partly met.**
  - Models, budgets and the 43.75% omission are explained.
  - Host exclusions are documented: the 105d/g bandwidth gate is pre-specified, and the rental ledger lists failed rentals.
  - However, the job 106 exclusions are post hoc, and the "stable" class is defined after seeing the results.
  - The workload subset (AIME only, also the development set) is a real limitation.
- **Rule 3 (means): met.** Time per token is averaged as a cost. I confirmed that ratios of mean speeds and of mean times agree within 0.003 on Table 2.
- **Rule 5 (variability and intervals): partly met.**
  - Within-host problem-level intervals are everywhere, and many are tight.
  - There are no intervals on the relation's errors, on G, on B_host, on the Fig. 1 shares, on D(W50)/C, or on the pooled online-rule effects.
  - The registered ±6% criterion is not compared with the inputs' own uncertainty (W2).
- **Rules 6 and 7 (nonparametric, sound comparison): mostly met.**
  - Paired bootstrap; configurations shuffled within each process with recorded seeds; two-stage machine-then-problem bootstrap for the panel; machine-level Spearman with BH correction across a 16 × 16 feature/outcome search, and the authors disclose that one headline feature was chosen from that search.
  - Two caveats: two-stage bootstraps over 2–3 rounds are degenerate (W5), and problem-level bootstraps ignore serial dependence within a process, since the cache carries across problems (mitigated by pairing).
- **Rule 8 (central tendency): not met.** p50 and p99 per problem are recorded but never analysed. For an interactive single-user setting, tail latency per token is the natural complement to the mean.
- **Rule 9 (document setup): mostly met.** Probes, CPUs, GPU UUIDs, commits and full job scripts are released. Clocks could not be locked; power limits and memory configuration (DIMM count and speed) are not tabulated per host; the headline card is memory-overclocked.
- **Rule 10 (measurement method): met.** One in-process timer is used for the oracle comparisons and one client for the system comparison. Nsight overhead on G is not characterised (W8).
- **Rule 11 (bounds): present, with a validity gap.** Both bounds are shown and tightened four ways in Table 20. The host rate is not a ceiling for the engine itself (W3), and the GPU term is below the measured device rate on host B.
- **Rule 12 (plots): met.**
- **Number of machines.** There are 25 machines for the oracle and engine program, which is commendable for rented hardware. The confirmatory sample for the relation is 3 re-rented machines plus 2 failures. The slow-link conclusions rest on 2 machines. Power analysis is absent: no statement of how many machines would be needed to bound the relation's error across machines.
- **Relaunches.** Relaunch drift (≤ 3.2%) is measured and is a strength. It is not propagated into the intervals of small effects (W5).
- **Threats to validity: partly addressed.** The Limitations section is candid about hardware, workload, bounds and oracles. It misses three threats:
  - the derivation-machine overlap with the registered test (W1);
  - the probe rate being exceeded by the engine (W3);
  - specification selection (W2).

## 6. Questions for the authors

1. Given that Pf, the Threadripper 9960X and Pe all contributed launches to the 34-launch set on which 1 − G/T was derived, why should job 106's success on them count as an out-of-sample test? Would you present 12/20 cells and a 4.7% pooled median as the registered outcome?
2. On the Xeon 8347C, Eq. (3) misses by 15–17% in its stable round and budget. What explains this? Possibilities include B_host measured at more threads than the 34 helpers, sub-NUMA clustering, or memory-controller contention with the zero-copy kernels. Did you examine the per-layer timers there?
3. Which bandwidth definitions, and which G category sets, were tried on jobs 093–105 before registering B_host and the overlap term? Please give the full specification curve.
4. In 13 of 68 exploratory launch-cells the engine's implied read rate exceeds B_host by more than 5% (up to 1.22× on 100f). How can Eq. (1) at B_host be a lower bound there? Would you report the bound at a ceiling, such as the maximum observed effective rate or the theoretical DRAM peak?
5. What are the token-to-token and problem-to-problem standard deviations of G, and how much does Nsight change kernel durations, as opposed to wall time? Profiled wall time is 20–35% above unprofiled on several hosts.
6. How were the round-level 95% intervals computed with 2 or 3 rounds, and what is their coverage? Why not several launches per machine for effects of 1–2%?
7. For the EPYC 7302, can you show that its loss with the fewest-admission set comes from routing divergence, for example by measuring the oracle's plan hit rate against the realised routing? Can you obtain a correctly functioning machine with a link-to-CPU ratio below 0.25?
8. For the horizon rule: what is the slope of D(W50)/C against C/k, with an interval over models, after excluding the five sub-token points? Do the S traces' prompt tokens enter the simulated decode stream, given that prompts bypass the cache in the engine?
9. Your greedy MIN admits 21.8 per token at 11%; mine admits 21.0. Which admission and tie rule defines "greedy"? Does the fewest-versus-greedy comparison depend on it?
10. Why does the headline host B use a memory-overclocked card (17001 MHz, 1847 GB/s), and does Table 2's datasheet GPU term remain a bound there?

## 7. What would move my score (ranked)

1. **Run the relation's registered test on new machines** (this alone would move me to 7 if it passes). Register Eq. (3), as is, on at least five machines never used in derivation, including at least two server-class CPUs. Pre-register the exclusion gates: output NLL within a fixed tolerance, round-to-round variation of base, memory in use before start. Report every machine. If that is impossible, reframe the claim: the registered test failed as specified (12/20 cells, pooled 4.7%); it held on re-runs of three derivation machines; it missed by 15–17% on a stably running server. Scope the relation to the desktop and workstation class tested.
2. **Make the bounds bounds.** Use a host rate that dominates every effective rate the engine achieves (or the theoretical DRAM peak) and the measured GPU peak, and report Eq. (1) and (2) at both the probed rate and the ceiling. Keep "% of bound" figures only for the ceiling version, or label them clearly.
3. **Replace "nothing fitted" with a specification-uncertainty analysis.** Give the specification curve over bandwidth choices, G definitions and admission-discount forms. Give prediction intervals that propagate uncertainty in G (from profile resampling) and in B_host (from probe-sample variability), and judge the 6% criterion against them.
4. **Fix the statistics of small effects.** Run a variance-components pilot over machines, launches, rounds and problems. Use at least three launches per machine for any effect of 3% or less, and a hierarchical model, or at least report between-launch spreads of paired ratios. Drop percentile bootstraps over 2–3 rounds.
5. **Report every failed registered clause where its claim is made** (the list in W7), and add a one-paragraph calibration summary of the prediction record to the main text.
6. **Broaden the slow-link evidence** before claiming "wins where the link is slow": at least three functioning machines below a ratio of 0.25.
7. **Rewrite for clarity.** Put a glossary and one host table in the main text; reduce job numbers in the argument; reconcile Eq. (3) and Eq. (4) or drop one; move the system comparison into a tighter appendix.
8. **Restate the horizon rule with its trend in C/k and an interval**, excluding the interpolated points, and say plainly that applying it needs the target model's routing trace.

## 8. Scores

- **Overall: 5/10** (borderline reject)
- **Soundness: 3/5.** Measurements are sound and reproducible; key inferences (generality of Eq. (3), the bound as a bound, "every stable machine") are overstated.
- **Methodology: 4/5.** Pre-registration, the artifact and machine-level thinking are exemplary. Confirmatory sample size, post hoc exclusion, degenerate round-level intervals and unlocked clocks hold it back.
- **Significance: 3/5**
- **Clarity: 2/5**
- **Confidence: 4/5**

**Overall 5/10.** This is an unusually rigorous and fully reproducible measurement study: every number I recomputed from raw data matched. However, its headline relation failed its registered test as specified and held only on re-runs of the machines it was derived from, and its bound uses a probed host rate the engine itself exceeds on some machines. With the relation re-tested on new machines, the bound computed at a true ceiling, and the claims scoped to what two to seven machines can support, it would be a clear accept.
