# Review 11a: MLSys 2027 main track, PC member

**Submission:** "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth"

## Materials read

- **Paper** (`paper/paper.pdf`, 35 pages). I read all of it: the 9-page main text, the references and Appendices A–M. I worked from the PDF text and the LaTeX sources (`paper.tex`, `app_wsg.tex`, `app_more.tex`, `app_traces.tex`, `app_value.tex`). For exact values I looked up the generated macro and table files (`wsg_job104/105/106.tex`, `wsg_sumlaw.tex`, `wsg_job102.tex`, `tab_job104/105/106/106_all.tex`, `tab_readsched.tex`, `tab_lomo.tex`, `tab_audit.tex`). I also rendered pages 4–6 to inspect Figures 1–3 and Tables 2–3.
- **Supplement** (`paper/supplement.pdf`): the front matter, and the job 105 and job 106 scorecard tables in full (`tab_scorecard_105.tex`, `tab_scorecard_106.tex`).
- **Artifact.** I read some scripts only for definitions, then wrote my own code:
  - `scripts/job106.py`, `sumlaw_paper.py`, `linkaware.py`, `w50_rebaseline.py` and the `panel_099.py` helpers.
  - `gpu/jobs/ec2/fetch_table.py`, for how B_c, B_p and B_host are read from the probe and how G is taken from Nsight categories.
- **Prereg JSON:** `readsched.json`, `linkaware.json`, `foresight/w50_aa.json`.
- **Job script headers** holding the registered predictions for jobs 104, 105 and 106, including the per-host wrappers.
- **Raw results** on the gpu branch for jobs 069c, 081, 084b/084c (the AIME routing traces), 089, 093–106 and the nine models' S traces.
- **Rental ledger:** `gpu/vast_ledger.json`.
- **Git:** I ran `git log --format='%h %ad'` on the job-104/105/106 scripts and `git diff` between the registration commit and the amendment commit of jobs 105 and 106. I did not read any commit messages.

## Independence statement

- **Files I did not open:** anything under `reports/` except to write this file, any `prereg/*outcome*.md`, `research_notes/`, or any file named like a review, plan, number check or progress log. That also rules out `paper/paper_v1_prereview.tex`, `apply/APPLICATION_PLAN.md`, `prereg/ref_check/` and the gpu-branch result directory whose name contains "review".
- **Commit messages:** not read.
- **Where the recomputation was done:** in a scratch copy, `.../scratchpad/review11a/repo` (made without `reports/`, `research_notes/` or the outcome notes), against a scratch copy of the gpu-branch results.
- **My code:** every checking script is my own, in `.../scratchpad/review11a/chk/`:
  - `common.py`, `j106.py`, `j104_105.py`, `explor.py`, `bounds.py`, `xmach.py`
  - `minsim.py`: Belady MIN with bypass at step granularity
  - `fewest_lp.py`: a lexicographic two-stage LP for fewest admissions
  - `horizon.py`: distinct-expert curves
  - `w50_own.py`: an independent window-policy simulator
- **Use of the authors' code:** none of it was imported. I consulted it only for definitions, for example that B_host is the maximum over all probe samples.
- **Other reviews:** I have not seen any other review of this submission.

## Desk-level issues (not scored)

1. **Template year.** The submission uses `mlsys2025.sty`; the chairs should confirm the current year's style.
2. **Anonymity of the artifact.** The PDF is anonymous. The LaTeX source in the artifact, however, has a named author block with an e-mail address, and the appendix points readers to a "public gpu branch". If reviewers receive the artifact as it is, it is not anonymous. The chairs should check.
3. **Reliance on appendices.** The main text fits the page limit, but several main-text claims are backed only by appendix material (23 pages of appendices plus a 38-page supplement). Examples:
   - the two excluded job-106 hosts (Table 23);
   - the microbenchmark (Table 21);
   - the audit of published systems (Appendix L).

   The main text should stand on its own.

## 1. Summary

The paper studies batch-1 decode of MoE models whose experts mostly live in host DRAM: gpt-oss-120b (MXFP4) and Qwen3-30B-A3B (BF16) on rented RTX 5090 hosts, using a 3,500-line expert-cache patch to llama.cpp. It makes five contributions.

1. **Two lower bounds** on time per token for any system with C GPU slots per layer that follows exact routing:
   - Eq. (1): MIN-with-bypass host reads at the machine's best probed read rate, overlapped with GPU work.
   - Eq. (2): for systems that read only after routing, the GPU's non-expert time plus MIN's read time, in series.
   - A microbenchmark replaying MIN's reads reaches 86–96% of the read term.
2. **An accounting relation** for the authors' deployed cache, T = G + (M + (1 − G/T) A) S / B_host, with nothing fitted. G is profiled; M and A are the run's counters.
   - The plain form was found on 30 launches.
   - It failed its registered test at 25% in job 105.
   - The overlap term was then added and registered for job 106. There it held within 6% on 12 cells of three re-rented machines and missed by 10–52% on two new server machines, which the authors exclude as unstable.
3. **In-engine oracles** that read a recorded trace:
   - MIN's set copied once in the step beats the deployed cache by 16–51% on three fast-link hosts.
   - Among MIN's hit-optimal schedules, the one with the fewest admissions avoids the greedy schedule's losses on slow links.
   - Online, raising the admission margin ("admit less") gains 0–6% on gpt-oss.
4. **A trace study of partial foresight.** The window that closes half the read gap between admit-every-miss and MIN is about 0.65 C distinct experts per layer across nine models. No forecaster tested approaches it.
5. **An artifact** in which every job's predictions were committed before the machine started, with a clause-level scorecard.

## 2. Strengths

- **S1. Exemplary empirical discipline.** Predictions are committed before rentals, and I verified the timing:
  - job 104: commit at 17:33:12Z, first rental at 17:33:17Z;
  - job 105: 22:05:31Z against 22:05:36Z, with the amendment at 22:09:26Z before the replacement rentals at 22:09:28Z;
  - job 106: 01:43:48Z against 01:43:58Z, with the amendment at 01:44:13Z before host e at 01:44:16Z.

  The amendments change only the host lists. Re-rented machines are identified by GPU UUID; I checked that "Pe again" is the same GPU as panel host 099e, and the same holds for Pf and the Threadripper 9960X. Failures are reported, including the registered failures of the law in jobs 105 and 106 and of the online rules. This is rare in our community and worth rewarding.
- **S2. A reproducible artifact.** With independent code I reproduced almost every number I tried from raw rows, traces and probes: 29 claims (Section 4). Where I differ, the difference is small and traceable to definitions.
- **S3. A useful framing.** Measuring a system against a MIN-based, machine-probed bound, and splitting its time into GPU compute and reads in excess of MIN, is a better way to report offloading systems than speed-ups over self-chosen baselines.
- **S4. A concrete, well-supported finding.** The paper separates MIN's set from MIN's schedule. On the two slow-link machines tested, the greedy hit-optimal schedule loses 7–15% where the fewest-admission schedule gains, and this is consistent across five launches of those two machines. "An admission costs a link crossing" is a practical lesson for anyone building such caches.
- **S5. Machines treated as the statistical unit.** Variance shares show machines carry 96% of the variance of the gain, against 1% for problems. Two machines with the same CPU model differ by 16–29%. Both support this choice of unit, and it is the right one.

## 3. Weaknesses (most important first)

### W1. The registered test of the relation is weaker than the abstract and introduction suggest

- **The registered success is on machines from the exploratory set.**
  - The paper says the overlap term was chosen after job 105 failed at 25%.
  - The three job-106 machines on which the law "held" were all in the data used to find the form, as their GPU UUIDs confirm. Pf appears in jobs 099, 101, 103, 104 and 105; the Threadripper 9960X in 105b; Pe in 099e.
  - For gpt-oss, the registered test therefore shows that the relation reproduces on new runs of the same machines. It does not show that it generalises to new machines.
  - Qwen3 is the only new axis, and the job-106 header itself says Qwen3 was examined on 10 earlier launches.
- **On the only two new machines the relation failed badly, and the pooled criterion failed.** I get errors of −10.2% to −52.4% on those machines, and a pooled median |error| of 4.67% over all hosts against the registered ≤ 4% (scorecard clause P2-median: failed).
- **The exclusion rule is post hoc.** "Deployed cache varied > 2% between rounds" was registered prediction P8, not a registered exclusion criterion. Using a failed prediction to remove hosts from the summaries (`scripts/job106.py`, `unstable = {... var14(h) > 0.02}`) is what pre-registration is meant to prevent. The fix is not to hide the two hosts (the paper does report them) but to:
  - present the all-host result as the registered outcome;
  - label the three-host summary as post hoc.
- **The wrong-outputs explanation does not cover the law's miss on the EPYC 7302.** In rounds 1–2 on that host:
  - the teacher-forced loss is 0.208 against 0.19 elsewhere;
  - the oracle counters match the other hosts within 1%: fewest-admission misses 41.09 against 40.73, so routing was close to the trace;
  - yet the deployed cache took 17.4 and 18.9 ms against the law's 12.2 ms, about −30%.

  The plausible cause is a shared server where the probe's maximum rate is not delivered during the run. The paper itself notes 131–132 GB of host memory in use at start, and that probe CPU rates fell when more threads read than the container had physical cores. That is a limitation of the method: B_host is the maximum of about 20 noisy probe samples, including concurrent sums. It is not only a property of "unstable machines".
- **The registered success is narrow.**
  - The largest stable-host error is 5.99%, on the Threadripper at Qwen3 25%, at the edge of the 6% band.
  - Qwen3 errors on that host are +4.9% and +6.0%.
- **The abstract understates the failure.** It should say plainly that the registered pooled criterion failed.

### W2. "Fewest admissions also wins where the PCIe link is slow" rests on two machines, omits the slowest-link machine, and conflicts with the paper's own two-read data

- **The supporting evidence is two physical machines.** "Slow link" is Pf (ratio 0.29) and the Threadripper 9960X (0.32): five launches in all.
- **The one slower machine lost in every round.** On the EPYC 7302 (link 13 GB/s, ratio 0.14) the fewest-admission schedule copied in the step ran 0.834, 0.878 and 0.824 times the deployed cache in the three rounds; the round-level value is 0.846 [0.82, 0.88].
  - In rounds 1–2 the plan was followed: 11.19 copies per token against 11.86 elsewhere, with 0.9% more misses.
  - So the loss is not explained by wrong outputs. The copies alone take about 11.4 ms per token at 13 GB/s, against a 17–19 ms baseline.
  - Two estimates bracket the measured 20.9–21.5 ms: copies plus G with the CPU fully overlapped gives 15.7 ms, and a fully serial sum gives 20.0 ms.
  - Below some link-to-CPU ratio, copies in the step lose even with the fewest admissions. The paper's own accounting implies this crossover.
- **The authors computed the result but do not report it.** `job106.py` computes it (`jiVsPlanMin/Max = 0.82/0.88`, `jiVsPlanCopyMs = 10`). The main text does not mention it, and Figure 3's x-axis starts at 0.25, which leaves out both new hosts (ratios 0.14 and 0.23). The Xeon at 0.23 also lost in round 1 (0.996).
- **Two reads with fewer admissions beats one read on slow links.** On the slow-link machines, the fewest-admission set loaded by the CPU with a background copy (two reads) clearly beats the one-read in-step version:

  | Host (launch) | gpt-oss 11%: two reads vs one read | gpt-oss 25%: two reads vs one read |
  |---|---|---|
  | Pf (104a) | 1.113 vs 1.035 | 1.217 vs 1.059 |
  | Pf (105e) | 1.112 vs 1.028 | 1.215 vs 1.063 |
  | Threadripper 9960X (105b) | 1.114 vs 1.039 | 1.159 vs 1.044 |

  The introduction says foresight pays when spent "caching MIN's set and reading each expert once". The data say what matters is few admissions, kept off the critical path. Reading once is better only where the link is fast. The main text should state this and give the crossover.

### W3. Novelty relative to caching theory and prior MoE-offloading analyses is modest

- **The bounds.** They are MIN-with-bypass read counts times probed bandwidth, a roofline-style argument, and Eq. (2) adds a serial GPU term.
  - MoE-Lightning's hierarchical roofline (CPU, GPU and PCIe bandwidth terms for offloaded MoE) is cited only as "out of scope", but it is the closest prior performance-bound model and should be discussed as such.
  - WiSP, Budgeting Bytes and MoE-CAP are cited. The delta over them (per-layer MIN with bypass, probed rates, an on-demand variant) is incremental.
- **The fewest-admission schedule.** It is the interval (FOO-style) formulation with admissions as a second objective; the paper acknowledges this. Several precedents make "admissions are costly, so admit fewer" a well-known principle:
  - Demand-MIN;
  - offline caching with separate admission and bypass costs (the fault model versus the bit model);
  - flash-cache admission control (Flashield, TinyLFU-style admission).

  The new part is the measurement inside an engine.
- **The online rule.** It raises the deployed policy's margin κ from 1 to 2–3, which is parameter tuning, for 0–6%. The deployed policy reads every admission twice by design, so part of every "foresight" gain is a gain over that design choice.
- **The horizon in distinct experts** restates lookahead measured in distinct pages (Albers; Breslauer) and calibrates its constant empirically.

### W4. The relation is close to an accounting identity with per-host measured inputs, and has little shown predictive value

- **What it actually tests.** G is profiled on the same host, budget and engine, and M and A come from the run's own counters. The relation therefore checks that time not spent in GPU kernels equals bytes divided by B_host. That is useful, but it is not a predictive model of a policy change.
- **It barely beats "no change" on the one prediction tested.** When it was used to predict the online rules' speed ratios, "no change" did nearly as well: median 0.016 against 0.013, which I reproduce. The paper concedes "this tests it little".
- **It does not cover the states that matter for foresight.** It does not describe states that copy in the step, where the implied G rises to 7.4–16.4 ms. It is a description of the deployed path only.
- **Figure 1's split follows from the definitions.** "Serialisation plus the reads MIN would not make" is Eq. (3) minus Eq. (1). It is a consequence of the relation's form, not independent evidence.
- **"Nothing fitted" needs a caveat.** It is literally true, but the functional form was selected after a registered failure, and the result depends on the choice of B_host (maximum versus median probe sample) and on G profiled from one problem over 24 tokens. Neither input carries an uncertainty.

### W5. Eq. (2) is presented as a bound "for every system we know of", but its GPU term belongs to this engine

T_GPU (2.9–3.4 ms; I reproduce 2.94–3.38) is this engine's non-expert kernel time. A system with faster attention or dense kernels (fused kernels, CUDA graphs, a different quantised format for dense weights) would not be bounded by it. Limitations notes this ("not a property of the machine alone"), but the introduction and Section 3 should call Eq. (2) an engine-relative bound, or derive T_GPU from a machine property.

### W6. The comparison with published systems is not like-for-like

The main text contrasts "ours at 38–54% of Eq. (1)" (probed host rates) with "published systems at a median 13.6%". The published systems are scored at datasheet DRAM and PCIe ceilings, on the authors' own traces (some of sibling or FP8 variants, some with uniform routing), against published end-to-end timings. These are different procedures. Either score the authors' own cache by the same datasheet procedure, or move the comparison out of the main text.

### W7. Several statistical choices are weaker than presented

- **Round-level intervals with 2–3 rounds.** Resampling three rounds yields at most 10 distinct round multisets, so these intervals are mostly problem-level intervals. For effects of about +2%, the relevant variability is between launches. Re-launches agree within 1.9% (job 100/101 re-launches) and 3.2% (job 106) in deployed time. The registered prediction that would have given launch-level evidence on the slowest link failed.
- **Many registered clauses have no interval or little content.**
  - 112 of 181 job-106 clauses are "held (point)".
  - Several restate earlier measurements or properties of the trace: G between 4.0 and 5.0 ms; copies ≤ 0.65× the greedy schedule's, which is fixed by the trace (0.597 on every host).
  - Of 301 band clauses in jobs 088 onward, 90 failed.

  The scorecard is a strength in transparency, but its "held" counts should not be read as predictive success.
- **Post-hoc classes.** Statements of the form "on every machine that ran stably" use a class defined after seeing the data.

### W8. The horizon rule is less robust than "one constant" suggests

- **It needs the target model's trace.** The rule uses the held-out model's own D(W) curve. Given that trace, W50 can be simulated directly.
- **The leave-one-model-out comparison does not separate the rules.** Medians of 1.16, 1.19 and 1.22 on 26 points from 9 models are not distinguishable.
- **Five of the 26 points have W50 < 1 token,** obtained by linear interpolation between W = 0 and W = 1. With C = k, D(W50)/C equals W50 by construction there. I find olmoe at C = 8: 0.686, gpt-oss-20b at C = 4: 0.649, and similar values.
  - With them, my own D(W) gives a median of 0.650 (IQR 0.608–0.721), matching the paper.
  - Without them, the median is 0.63 and the range 0.52–0.99: qwen1.5-moe at C = 7 gives 0.99, qwen2-57b 0.86–0.91, deepseek-v2-lite 0.84.
- **The constant depends on policy details.** My independent window-policy simulator follows the paper's description of the policy but uses κ = −∞ only, without the authors' minimisation over κ. At five points it gives W50 7–29% larger and D(W50)/C of 0.60–0.82 instead of 0.54–0.72.

### W9. Clarity

The writing is dense, and the paper reads like a lab notebook compressed into nine pages.

- **Terminology.** Readers must track job numbers (073–106), host nicknames (Pf, Pe, Pg, O3–O5, the panel) and many configuration names (deployed, single read, MIN 2 reads, fetch, fetchplan, bypassplan, dk, lrn, window W with recall r).
- **Hard sentences in the abstract.** For example, "Its gap to the bound is that serialisation plus the reads MIN would not make."
- **Inconsistent naming.** "Relation" in the text, "law" in the tables.
- **Missing information in Table 3.** It omits the greedy column, so the claim "over the greedy schedule" for job 106 cannot be read from it.
- **Fragmented claims.** Each claim uses a different subset of machines (3, 7, 10, 19, 21, 23). A table mapping each claim to its machines and launches would help.

### W10. Scope

- **Narrow setup:**
  - two models, one GPU type, batch 1;
  - one prompt set (AIME-25, also used during development);
  - teacher-forced replay, with parity measured by KL divergence rather than task accuracy.
- **Prompts and batches above one bypass the cache.**
- **The realisable gains are small:**
  - the online rules gain 0–6%, with a confidence interval including 1 on one of six stable gpt-oss cells;
  - forecasters close at most 15% of the read gap.

  The paper is mostly a measurement study of one engine, which limits its significance for the broader MLSys audience.

## 4. Claims checked

All values below were recomputed with my own code from raw rows, probes, Nsight JSON, traces or prereg JSON. "✓" means my value matches the paper at its stated precision.

| # | Claim | Where | My value | Verdict |
|---|---|---|---|---|
| 1 | R⋆ (MIN with bypass, reads per token), gpt-oss 11% / 25%, used in the bounds | §3, Tab. 2, Tab. 21 | 38.315 / 15.340 (own step-granularity simulator on the 084c trace); equal to `readsched.json` | ✓ |
| 2 | Fewest admissions among hit-optimal schedules: 12.4 per token at 11%, against 21.8 for greedy | §3 | Own lexicographic LP: hits equal MIN's exactly (38.315 misses), 12.399 admissions, fractional share ≤ 0.05%. Own greedy rule: 21.40 | ✓ (greedy count within 2%; depends on tie-breaking) |
| 3 | Running example O4: Eq. (1) 10.0 ms, Eq. (2) 13.0 ms, deployed 20.6 ms | §5 | 10.02 / 12.96 / 20.64 ms (B_host 50.7) | ✓ |
| 4 | Jobs 093–104 at 11%: 38–54% of Eq. (1), 55–70% of Eq. (2) | §3 | 38.4–54.0% / 55.2–70.3% (30 launches) | ✓ |
| 5 | T_GPU 2.9–3.4 ms; profiled G 4.1–4.7 ms (gpt-oss) | §3, §4 | 2.94–3.38 / 4.07–4.69 ms; medians 4.28 (C14), 4.50 (C32) | ✓ |
| 6 | Table 2, host-bound bound columns (host B / host S) and ours ÷ FreeToken on host B | Tab. 2 | 172, 430, 96, 214 / 140, 351, 78, 174 tok/s; 69.9/54.0 = 1.294, 109.2/85.6 = 1.276, 152.5/132.2 = 1.154 | ✓ |
| 7 | Microbenchmark reaches 86–96% of the read term (per layer) | §3, Tab. 21 | 86, 89, 96, 96, 95, 93% | ✓ |
| 8 | Plain form within 6% on 28 of 30 launches at 11%; median over-prediction 6% at 25% | §4 | 28/30 (errors −10.0 to +13.7%); +5.9% | ✓ |
| 9 | Over 34 launches, overlap form against plain form: 33 vs 18 within 6% at 25%; 29 vs 32 at 11% | §4 | 33 vs 18; 29 vs 32 | ✓ |
| 10 | Job 105 plain form fails at 25% (6.9–9.3% on 3 of 4 hosts, 3.8% on Pf); pooled median 4.2% | App. B | +9.3, +7.0, +6.9, +3.8%; median 4.24% | ✓ |
| 11 | Job 106 stable hosts: 12 of 12 cells within 6%; median 2.3%, largest 6.0%; plain form up to 10.1% | §4, Tab. 3 | 12/12; 2.34%; 5.99% (TR, Qwen3 25%); plain 10.08% | ✓ (largest error at the band edge) |
| 12 | Job 106 all hosts: pooled median 4.7% (registered ≤ 4%); unstable hosts 10–52%; ratio of g25 medians ≤ 0.5 | §4, App. J | 4.67% (fails); −10.2 to −52.4%; 0.355 | ✓ |
| 13 | Elasticity of T − G on B_host: −0.95 [−1.06, −0.84] | §4 | −0.950 [−1.061, −0.841] (machine bootstrap); raw T −0.74 | ✓ |
| 14 | Fig. 1 at 11%: G 12–47%, MIN 31–54%, beyond MIN 22–35%; G is 23–68% of the gap | §4 | 13–47, 31–53, 22–34%; 23–68% | ✓ (within 1 point; normalisation) |
| 15 | 23 machines (jobs 093–105) | Abstract, Fig. 1 | 23 distinct GPU UUIDs over 34 launches | ✓ |
| 16 | Fewest-admission schedule in the step: 10 launches on 7 machines, 1.02–1.34× at 11% | §5 | 10 / 7; 1.019–1.342 | ✓ |
| 17 | Its gain over greedy, 0.08–0.17, on 5 machines with ratio 0.28–0.63; on slow machines greedy 0.85–0.93× and fewest 1.02–1.06×; round-level lower bounds 1.01 and 1.05 | §5 | +0.082 to +0.172 (5 machines, 8 launches); 0.853–0.933; 1.019–1.056; 1.013 / 1.048 | ✓ (two slow machines only) |
| 18 | Fewest-admission copies 0.60–0.73× greedy's; 2.5–5.3% more misses | §5, App. J | 0.597–0.726; +2.52% / +5.26% | ✓ |
| 19 | **Not reported in the main text:** fewest-admission schedule on the EPYC 7302 (ratio 0.14) | Tab. 23 only | 0.834 / 0.878 / 0.824 per round; rounds 1–2 counters match the trace | Contradicts the slow-link claim (W2) |
| 20 | dk: reads −4–6% at 11% and −6–10% at 25%; speed 1.00–1.02 / 1.01–1.06; Qwen3 1.01–1.04 | §5, Tab. 3 | −4.4 to −5.8% / −6.3 to −10.5%; 1.003–1.024 / 1.008–1.057; 1.010–1.038 | ✓ |
| 21 | dk pooled median ≥ 1.02 holds only through the unstable hosts (1.01 on stable) | §5 | 1.041 (all) / 1.014 (stable) | ✓ |
| 22 | lrn: 7–14% fewer reads, 267–439 µs host time, 0.98–1.05× | §5 | −6.6 to −14.2%; 267–439 µs; 0.982–1.055 | ✓ |
| 23 | Law's ratio prediction: median 0.013, max 0.045; "no change" 0.016 | §5 | 0.0135 / 0.0452 / 0.0159 | ✓ |
| 24 | MIN, 1 read, on O3–O5 at six budgets: +16–51%; prefetched +25–81% | §5 | 1.160–1.514; 1.253–1.812 | ✓ |
| 25 | Layer-ahead copy: 1.04× on the fast-link host, down to 0.72× | §6, Tab. 18 | 1.042; 0.722 (Pf), 0.814, 0.909 | ✓ |
| 26 | Spearman of MIN-in-the-step gain against link/CPU ratio: 0.89 [0.63, 0.97], 19 machines; machines 96% of variance, problems 1–3% | §5, App. J | 0.886 [0.61, 0.98] (first launch per machine); 96.3% / 1.1% | ✓ |
| 27 | Same CPU model 16–29% apart; re-launches within 1.9% (job 101) and 3.2% (job 106) | §2 | 15.6–29.1%; max 1.88% (100c vs 099h); 3.2% (106a vs 105b) | ✓ |
| 28 | Job 106 host details: wrong outputs 0.21–0.45 nats against 0.19 / 0.085; 39% round variation; "Pe again" is panel host Pe | §4, App. J | 0.208–0.446; 38.8%; same GPU UUID as 099e (the registration header named it only as "a Ryzen 9 9950X") | ✓ |
| 29 | Horizon: W50 0.6–34 tokens; D(W50) ≈ 0.65 C (quartiles 0.61–0.72) | §6 | 0.64–33.6 (registered JSON); own D(W): 0.650 [0.608, 0.721]. Without the 5 sub-token points: 0.63, range 0.52–0.99. Own window policy: W50 +7–29%, D/C 0.60–0.82 | ✓ as computed; fragile (W8) |

I did not recompute the link-aware bound's +1% / +7% (§3). A rough hand calculation on Pf at 11%, using the LP frontier, gives about +1–1.5%, which is consistent.

## 5. Questions for the authors

1. **Exclusion rules.** Was any exclusion rule registered for job 106? If not, will you present the all-host outcome as the registered result (pooled median 4.7%, failed) and mark the three-host summaries as post hoc?
2. **The EPYC 7302.** In rounds 1–2 the fewest-admission plan was followed (counters within 1%), yet it ran 0.83–0.88×. Do you agree the cause is the 13 GB/s link? Can your accounting predict the link-to-CPU ratio below which copies in the step lose even with the fewest admissions, and will you state it in the abstract's slow-link claim?
3. **Reading once versus reading twice.** On every machine with ratio < 0.35, the fewest-admission set loaded by the CPU (two reads) beats the one-read in-step load by 7–16 points. Why does the main text present "reading each expert once" as how foresight should be spent?
4. **The probe.** How sensitive is Eq. (3) to the definition of B_host? Please report job-106 errors with the median probe sample, and with B_both, instead of the maximum.
5. **G's uncertainty.** How variable is G across problems and rounds? It is profiled from one problem over 24 tokens. Can you attach an interval to each law error?
6. **The overlap term.** Can the (1 − G/T) share be checked directly on the Nsight timelines, as the fraction of background-copy time that overlaps GPU kernels? The paper says this has not been measured.
7. **Eq. (2).** Can T_GPU be replaced by a machine-level quantity, for example the fastest non-expert time among the engines you ran, so that Eq. (2) is a bound for other systems?
8. **The horizon rule.** What is D(W50)/C without the sub-token points, and with κ fixed rather than minimised per window? Does the 0.65 constant survive a different window policy, such as MIN within the window?
9. **The audit.** If your own cache is scored by the audit's datasheet procedure, where does it fall relative to the 13.6% median?
10. **Push times.** The timing of pre-registration rests on local commit timestamps. Can you give server-side push times, for example from the hosting service's event log?

## 6. What would raise my score

1. **Rescope the claims.** In the abstract and introduction:
   - state that the registered pooled criterion of job 106 failed;
   - limit the slow-link claim to the measured ratio range, about 0.28–1.0;
   - report the EPYC 7302 result and the predicted crossover;
   - replace "reading each expert once" with the admission-cost lesson the data support.
2. **A genuinely out-of-sample registered test of Eq. (3):**
   - at least five machines never used in jobs 093–106, including server-class hosts;
   - a third model;
   - exclusion rules registered before launch;
   - errors with uncertainty from G and B_host.
3. **A predictive use of the relation.** Predict a new policy's time from replay counters rather than run counters, before running it, and compare with "no change" and a fitted baseline. Without this, Eq. (3) is accounting.
4. **Bounds.**
   - Recast Eq. (2) as engine-relative, or make T_GPU a machine property.
   - Give a short formal statement of Eq. (1)'s validity class.
   - Discuss MoE-Lightning's hierarchical roofline and offline caching with admission costs as direct precedents.
5. **The horizon rule.** Show robustness to the policy definition, drop or separate the sub-token points, and compare against simulating W50 directly.
6. **Clarity.**
   - Move the job-by-job narrative into an appendix.
   - Add a table mapping each claim to the machines and launches that support it.
   - Use one name ("relation") throughout.
   - Add the greedy column to Table 3.
   - Extend Figure 3's x-axis to include all hosts.

## 7. Scores

| Criterion | Score |
|---|---|
| Overall | **5 / 10** (borderline reject) |
| Soundness | 3 / 5 |
| Significance | 3 / 5 |
| Novelty | 2 / 5 |
| Clarity | 2 / 5 |
| Confidence | 4 / 5 |

The measurement discipline and the artifact are excellent: I reproduced almost every number I checked, and the observation that the choice among MIN's hit-optimal schedules matters is a real, practical finding. But the central relation is close to an accounting identity whose registered test failed on its only new machines and on the pooled criterion, the slow-link claim leaves out a contrary result from the slowest machine, and the conceptual novelty over caching theory and prior offloading analyses is modest.
