# Review: "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth"

MLSys 2027, main track. Reviewer 10b: performance modelling and scientific benchmarking.

## Materials read

- `paper/paper.pdf`: all 32 pages, including Appendices A–L.
- `paper/supplement.pdf`: all 34 pages of the prediction scorecard (Tables 1–8, jobs 073–105).
- Artifact on main:
  - `scripts/`: read `job105.py`, `reanalysis.py`, `sumlaw_paper.py`, `linkaware.py`, `speed_limit.py` (`host_rates`), `wsg_numbers2.py` (audit), `audit_ours.py` and `w50_rebaseline.py` to learn the definitions. I did not run any of them.
  - `prereg/*.json`: `job104.json`, `job105.json`, `scorecard_clauses.json`, `scorecard_0xx.json`/`scorecard_10x.json`, `linkaware.json` (LP frontier), `foresight/w50_distinct.json`, `audit/audit*.json`, `reanalysis.json` (the size of the correlation family only).
  - `gpu/vast_ledger.json`.
- gpu branch (`/home/claude/gpu-branch`):
  - Job scripts `jobs/*.sh`. Prediction headers read in full for 072, 104 and 105; scanned for the word "predict" in every job before 073.
  - Raw results for every launch of jobs 081, 093–105: per-problem `ec_*.jsonl` rows, `st_*.json` counters, `concur.txt` probes, Nsight `prof_C*.json`, `nvidia-smi-q.txt`, `gpu.csv`, `cgroup_cpu_max.txt`, `readsched_C*.txt`.
  - The gpt-oss routing trace `084c_gptoss_trace@vast/route_aime25_gptoss.npz`.
  - Job 072's results.

## Independence statement

I did not open anything under `reports/` except to write this file. I did not open any `prereg/*outcome*.md`, anything under `research_notes/`, or any file named like a review, plan, number check or progress log. I deleted `paper/paper_v1_prereview.tex` and `apply/APPLICATION_PLAN.md` from my scratch copy unread. I did not read commit messages. I used only `git log --format='%h %ad'` / `%at` to put script commits in order against the rental start times in the ledger.

All recomputation used my own code in `/tmp/.../scratchpad/review10b/my/`:
- `load.py`: my own parser for probes, rows and counters.
- `sumlaw.py`, `states.py`, `minadm.py`, `perproblem.py`, `w50.py`.
- `minlp.py`: my own greedy MIN with bypass, plus an exact min-cost-flow LP for the fewest-admission hit-optimal schedule.

These ran on a copy of the repository without `reports/` and without `prereg/*outcome*`. I wrote the probe-rate definitions (B_c, B_p, B_cp, B_host) myself from the paper's text and the job scripts; I did not import them.

---

## 1. Summary

The paper studies batch-1 decode of two MoE models (gpt-oss-120b in MXFP4, Qwen3-30B-A3B in BF16) on rented RTX 5090 desktops, with most experts in host DRAM. It makes four claims.

1. **Two lower bounds on time per token.** Eq. (1) is Belady's MIN-with-bypass host reads over the highest probed host rate, overlapped with the GPU. Eq. (2) puts the same reads in series with the GPU's non-expert compute, for systems that read only after routing. A read-only microbenchmark reaches 86–96% of the host term. A link-aware variant moves the bound by at most 7%. Published systems score a median of 13.6% of Eq. (1).
2. **An empirical relation, Eq. (3).** It says T = G + R·S/B_host, with G the profiled GPU kernel time and R the counted reads. It holds within 6% on 28 of 30 launches at gpt-oss 11%. It was registered before job 105, where it held at 11% and failed at 25%.
3. **An in-engine oracle factorial.** It crosses what to cache (deployed policy or MIN) with how to load (CPU then copy, or copy in the step). Its main findings:
   - Foresight pays when it is spent as MIN spends it.
   - The choice among MIN's hit-optimal schedules matters on slow-link machines; the fewest-admission schedule (jobs 104–105) wins there.
   - The gains track the host's link-to-CPU read ratio across machines.
4. **A "price of foresight".** Half of MIN's read saving over admit-every-miss needs a window holding about 0.65·C distinct experts per layer. No realisable forecaster reaches that.

The artifact is unusually complete:
- one job script per rental, with predictions committed before the machine starts;
- raw per-problem rows and counters;
- a clause-by-clause scorecard of 1,235 registered clauses.

## 2. Strengths

1. **The artifact is unusually reproducible.** I recomputed 25 central numbers from raw rows, probes, counters and traces with my own code. Every one matched to rounding (Section 4). Examples:
   - Table 2's headline ratio is 1.294 [1.278, 1.312]; I got the same value and interval.
   - The Eq. (3) statistics (28/30, median |error| 1.1%, range −10 to +14%) and the elasticity (−0.95 [−1.07, −0.84]) match exactly.
   - My independent min-cost-flow LP gives the fewest-admission count of 12.40 per token.
2. **The pre-registration is real and time-ordered.** For all 82 job scripts with rentals in the ledger, the script's first commit precedes the first rental, by 3 s to 15 min. Job 105's predictions were committed at 22:05:31Z and its first rental started at 22:05:36Z. Its amendment (105f, 105g) was committed 2 s before those replacement rentals. The amendment is stated in the header, with predictions unchanged.
3. **Failures are disclosed at scale.** The supplement lists 142 failed clauses for jobs 073–098, plus every failure in jobs 099–105. Section 4 reports that Eq. (3) failed its registered test at 25%.
4. **Machines are often treated as the statistical unit.** Relaunches are merged by GPU UUID for rank correlations. The elasticity uses a cluster bootstrap over machines. The variance decomposition (machines 96%, problems 1–3%) is reported, and launch-to-launch agreement on the same machine is measured.
5. **The bound is principled and correctly computed.** MIN with bypass maximises hits under C slots, and R\* = 38.32 / 15.34 reads per token is right; I reproduced it with both a greedy simulation and an exact LP. The link-aware refinement and the pooled-slot variant are sensible tightenings. The audit scores our cache by the same datasheet procedure as the published rows (`scripts/audit_ours.py`), which is the right like-for-like choice.
6. **Some ideas are useful beyond this system:**
   - Separating "which hit-optimal schedule" from "hit-optimal".
   - Showing the cost of reading an admission twice (CPU-then-copy).
   - Measuring look-ahead in distinct experts rather than tokens.
7. **One genuine out-of-range success for Eq. (3).** The Threadripper 9960X in job 105 has B_host = 178 GB/s, far outside the 30–96 GB/s of the 30 exploratory launches. Eq. (3) predicted its 11% time within +2.0%.

## 3. Weaknesses (most important first)

### W1. Eq. (3) is much narrower than the abstract and introduction state

Evidence, all recomputed from raw rows:

- **R barely varies where the relation holds.** At gpt-oss 11% the deployed cache's R ranges only 63.1–69.4 reads per token across all 30 launches. The relation is therefore tested almost entirely along the 1/B_host axis. The "host bytes" part of "GPU compute plus host bytes" is not exercised: a log-log regression cannot separate the R coefficient (it came out at 1.3) from noise.
- **It fails for every other engine state.** Solving T − R·S/B_host per launch for the other configurations gives:

  | State | Implied G, median (range) | Spearman with link/CPU ratio | In the paper? |
  |---|---|---|---|
  | MIN copied in the step | 7.4 ms | −0.91 | yes |
  | MIN prefetched | 2.7 ms | −0.85 | yes |
  | MIN, two reads | 3.8 ms | −0.72 | no |
  | Fetch-on-admit | 5.5 ms | −0.58 | no |
  | Admit every miss | 16.4 ms (6.0–24.9) | −0.91 | no |

  The profiled G is 4.3 ms. So the relation describes one configuration at one budget, not "where host memory binds, time is GPU compute plus host bytes".
- **At 25% it fails, in both the exploratory and the confirmatory sets.** On the 30 exploratory launches it is within 6% on only 17 of 30 (median error +5.9%); the paper gives the median but not the count. In job 105 it was within 6% on 1 of 4 hosts, and the pooled median of 4.2% exceeded the registered 4%.
- **The framing is selective.** The introduction and contributions say it held "on all 4 launches where we registered the relation before the runs". That is true only of the 11% half of the registration: the registered clauses covered both budgets, and 4 of 31 clauses failed, all of them the law at 25%. The abstract does not mention the registered failure.
- **The 30-launch result is exploratory.** `scripts/reanalysis.py`, which computes the 30-launch statistics, was first committed at 16:59 CDT on 6 October. Job 105's predictions were committed 6 minutes later. The 28/30 result is the data from which the relation was formulated. Only job 105 (4 launches, 2 new machines) is confirmatory: 2 of 2 new machines passed at 11% and 0 of 2 at 25%.
- **The 25% mechanism is untested.** The explanation for 25% is that background admissions are a larger share of reads and overlap the GPU. It is consistent across the two budgets but not within one. Across 34 launches at 25%, the law's error is uncorrelated with the admissions' share of reads (Spearman −0.00) and with their link copy time (−0.13).
- **"Nothing fitted" hides a design choice.** No constant is fitted to T. But the fit depends on taking B_host as the single highest of about 20 probe readings. With the median of the probe's concurrent-sum readings instead:
  - 11%: 27/30 within 6% (median |e| 2.3%);
  - 25%: 8/30 within 6%.

  With B_cp the counts are 25/30 and 7/30. The choice predates the relation, since it is the bound's rate. But the result's robustness to this measurement choice should be shown.

### W2. The bounds' validity and tightness are not established in a full system

- **Eq. (1) is a bound only if the probe is right.**
  - B_host is the maximum of single readings from one probe run, with no uncertainty.
  - On host 100f (9950X, x8 link), the deployed engine's effective host read rate, R·S/(T − G), is 61.3 GB/s against the probe's maximum of 52.0 GB/s, 18% above it.
  - That machine's STREAM copy rate varied from 38.5 to 53.5 GB/s across repetitions in the same job.
  - So on at least one of 23 machines, a system reading at our engine's rate would beat the "bound".
  - The limitation is stated in the abstract sense ("a faster reader than our probe could exceed it"), but this instance is not reported, and the bound is never given an interval.
- **Eq. (2)'s T_GPU belongs to this engine, not the machine.** It is the smallest Nsight-profiled non-expert kernel time of this llama.cpp build (2.9 ms). The paper says "Eq. (2) is the bound for everything else". A system with faster attention or dense kernels, or fewer launches, could beat it. It is a bound conditional on these kernels.
- **No full system gets close to either bound.** The read-only microbenchmark shows only that the host term's bytes can be moved at 86–96% of B_host without compute. In the engine:
  - An oracle that still reads on demand (MIN copied in the step) reaches at most 85% of Eq. (2) at 11% (median 74%, 32 launches) and at most 79% at 25% (median 67%).
  - The fastest state on any launch reaches at most 75% of Eq. (1)'s host term at 11% and 65% at 25%.
  - Whether the GPU/host overlap that Eq. (1) assumes is achievable is, as the authors say, unmeasured. "Nearly reachable" should be restricted to the read term.

### W3. Uncertainty is quantified over the wrong unit for most single-host claims

- **The interval covers the smallest source of variance.** Within-host intervals bootstrap over 20–30 problems. By the paper's own decomposition (I get the same numbers):
  - machines carry 96.3% of the variance of the log gain of MIN copied in the step;
  - problems carry 1.1–2.8%.

  Same-CPU panel hosts differ by 15.6–29.1% in deployed time.
- **Time and order are confounded with configuration.** Each configuration runs once per launch, as one contiguous block in a shuffled order. Time-position drift is therefore not in any interval. The paper's own launch-order checks exceeded 3% twice (7% and 11%, jobs 091/092, Appendix B).
- **Problems are not exchangeable.** The cache carries over between problems in a fixed order, so per-problem observations are serially dependent. The i.i.d. problem bootstrap further understates uncertainty.
- **The relaunch data set a noise floor that the claims then undercut.** I found 6 machines with relaunches. Base time agreed within 1.9% and every ratio within 0.030, which the paper adopts ("we do not read smaller differences on one host"). Yet several claims and registered "held" outcomes rest on effects of that size or just above it, from one launch:
  - the layer-ahead copy's 1.042 [1.034, 1.049] on O4, the only "fast-link" evidence;
  - the fewest-admission set's 1.028–1.039 over the deployed cache on Pf and the Threadripper;
  - "at most 0.01 behind" clauses (job 104 P6).
- **The point-estimate category is large.** 446 of the 1,235 clauses are "held (point)", with no interval at all.

### W4. Slow-link conclusions rest on two machines that differ in kind

- **The abstract's slow-link claim has two machines behind it.** "The greedy one loses on machines with a slow link, where the set with the fewest admissions gains" rests on Pf (two launches with the plan, five launches in all) and one Threadripper 9960X launch.
- **The two machines are slow for different reasons:**
  - Pf's link is PCIe x8 at 27 GB/s.
  - The Threadripper's link is a fast 53 GB/s; its ratio is low because its 4-channel DRAM reads 166 GB/s.

  The link-to-CPU ratio merges two hardware mechanisms that a two-point class cannot separate.
- **The ratio was found on the data it is then tested on.** It was identified on the panel ("the line at half was drawn after the panel ran"). It is then evaluated on 19 machines that include the panel.
- **The "B_p instead of the ratio" claim is a selected, untested contrast.** "MIN loaded by the CPU follows the link's absolute rate instead (0.79 against 0.45)" is the best feature from a family of 16 outcomes × 16 features (256 tests, `prereg/reanalysis.json`). Neither the search nor its BH correction is described in the paper. The difference between the two correlations is not significant: Δρ = 0.34, 95% bootstrap CI [−0.15, 0.83], 17 machines.

### W5. The prediction log is not complete, and registered predictions are often weaker than the authors' own model

- **Not every prediction is scored.** Section 2 says "Each job's predictions were committed before its machine started; Appendix B scores all of them." Job 072 (`jobs/072_defer@vast.sh`) registered three predictions ("Prediction, recorded here before the run"). The scorecard starts at 073, and the paper never mentions job 072. From its raw rows:
  - Prediction 2 (defer=1 speeds up C14/32/56 by 0–10%, 3–15% and 5–20%) failed: −3.2%, −3.7% and −3.5%.
  - Prediction 1 (input-upload time falls by at least half) failed at C14 (−43%) and C32 (−40%). At C56 it fell by exactly half.
  - Prediction 3 (identical hits) held.

  The anchor jobs 050–054 also carried predictions; they are out of this paper's scope, but the "all of them" sentence should be scoped.
- **Not every rental is reported.** Ledger rentals with no results that the paper does not mention:
  - 100e (37.6 min), on the same offer as an aborted 13-minute 099d attempt;
  - two replaced 099a attempts on different machines.

  The paper names only 100d, 103c, 105c, 105d and 105g. A rule-based exclusion table belongs in the paper.
- **Sign-only registrations make "held" uninformative.** Job 104's header states that the calibrated model put the fewest-admission set at 1.21–1.29× the deployed cache on Pf. The registered clause was only "> 1". The measurement, 1.035, is scored "held", yet the model behind the clause missed by 15–20%.
- **The bands themselves predict poorly.** About 30% of the bands registered from job 088 onward failed (90 of 301). The authors' quantitative models predict effect sizes poorly; the paper should say so where it uses those models as explanations.

### W6. The fewest-admission schedule is not uniformly better than greedy

- **Against greedy, it loses on the fast-link machine at 25%.** Against the deployed cache it gains with every load at both budgets on all 7 launches, every 95% interval above 1 (verified). But on O4 (ratio 1.04) at 25% it is behind the greedy set:
  - in the step: 1.387 vs 1.419;
  - with two reads: 1.096 vs 1.167, a 0.07 deficit;
  - on Pg at 25% with two reads it is 0.047 behind.

  Table 4 shows only the in-step columns, and the conclusion's "gains on every machine it ran on" is relative to the deployed cache only.
- **The engine does not execute the LP plan faithfully at 25%.** The engine's copy ratio is 0.726 and its misses are 5.3% higher. My LP on the 20-problem trace gives a copy ratio of 6.73/10.47 = 0.643 with equal misses.
- **"The greedy schedule" is not unique.** My natural tie-break (keep the C soonest-next-use among residents and requests) admits 21.40 per token at 11%, against the paper's 21.8.

### W7. The workload is narrow and overlaps with development

- **Engine claims cover very little workload.** Every engine claim uses AIME-25 problems (also used during development), the first 256 greedy tokens, teacher forcing, batch 1, and two models.
- **The cross-domain evidence is in simulation only.** The paper itself shows that the hot set moves across text (a static cache profiled elsewhere reads 1.4–13.2× MIN's reads). The nine-model trace study is simulation only.
- **Batches above one bypass the cache.** Prompts and batches above one run at 77–83% of stock llama.cpp.
- **The confound with development is not discussed.** Developing the policy and its FETCH tables on the same prompts used to evaluate them is a threat to validity that the paper does not address.

### W8. The paper is very hard to read

- **Too many named configurations.** The main text uses about 20 of them (deployed, single read, admit every miss, MIN 1/2 reads, fewest-admission, prefetched, Belady 1/2 reads, window W, window W at recall r, deployed + window), plus artifact labels in figures (fetch, foa, aa, both3p, bypassplan).
- **Key definitions are buried.** Example: B_host is "the highest rate by any method", meaning the maximum single reading of one probe run.
- **Sentences carry too many numbers.** Many sentences hold four or more numeric ranges. The abstract cannot be parsed without the paper.
- **Much of the evidence lives in narrative.** Appendix B's prose scorecard is a page of dense narrative.

## 4. Claims checked

All values were recomputed with my own scripts from raw rows, counters, probes and traces unless marked "(prereg JSON)". "✓" means reproduced to the paper's rounding.

| # | Claim | Where | My value | Verdict |
|---|---|---|---|---|
| 1 | MIN-with-bypass host reads R\* per token, gpt-oss 11% / 25%, 30-problem trace: 38.3 / 15.3 | Table 14, Sec. 3 | Greedy simulation and exact min-cost-flow LP both give 38.32 / 15.34 | ✓ |
| 2 | Fewest admissions among hit-optimal schedules: 12.4 per token at 11%, against 21.8 for greedy | Sec. 3 | My LP: 12.40 (25%: 6.75). My greedy: 21.40 (25%: 10.51). The greedy count depends on tie-breaking | ✓ (LP); greedy ≈ |
| 3 | Running example on O4, gpt-oss 11%: Eq. (1) 10.0 ms, Eq. (2) 13.0, deployed 20.6, MIN in step 15.2, prefetched 13.8 | Sec. 4 | 10.02; 12.9 (with T_GPU = 2.9); 20.64; 15.16; 13.81 | ✓ |
| 4 | Table 2 bound, host B 172 / 430 tok/s, host S 140 / 351 | Table 2 | B_host 87.5 and 71.3 GB/s → 172 / 430 / 140 / 351 | ✓ |
| 5 | Ours ÷ FreeToken on host B: 1.294 [1.278, 1.312], 1.275 [1.254, 1.295], 1.154 [1.134, 1.172], 1.032 [1.022, 1.042] | Table 2 | Identical (one upper bound 1.173). Same launch rerun: 1.001 [0.999, 1.003] | ✓ |
| 6 | Eq. (3), 30 launches at 11%: within 6% on 28/30, median abs. error 1.1%, range −10 to 14% | Sec. 4 | 28/30 on 21 machines (19/21 machines); 1.12%; −10.0% (099i, i7-14700K) to +13.7% (100f) | ✓ (exploratory) |
| 7 | Implied G median 4.3 ms, IQR 4.1–4.6, Spearman 0.02 with link ratio | Sec. 4 | 4.32 (4.14–4.58); ρ = 0.02 over launches, −0.11 over machines; **full range 1.80–7.89 ms** | ✓; range not reported |
| 8 | Elasticity of T − G on B_host: −0.95 [−1.06, −0.84] over machines | Sec. 4 | −0.950 [−1.066, −0.840], cluster bootstrap over 21 machines | ✓ |
| 9 | At 25%, Eq. (3) over-predicts by a median 6% | Sec. 4 | +5.9%; **within 6% on only 17/30** | ✓; count not reported |
| 10 | Job 105 (registered): at 11%, −1.5 to +4.7%; at 25%, +6.9 to +9.3% on 3 hosts and +3.8% on Pf; pooled median 4.2% > 4% | Table 4, App. B | Identical, host by host | ✓; registered test failed at 25% |
| 11 | G_prof 4.1–4.7 ms in 14 profiles on 7 hosts, median 4.3 | Sec. 4 | 4.07–4.69; median 4.28 (11%), 4.50 (25%); the 069c-only median is the same at 11% | ✓ |
| 12 | Implied G: MIN in step 7.4 ms (ρ = −0.91 with ratio); prefetched 2.7 ms | Sec. 4 | 7.37 (−0.91); 2.71 (−0.85). Not reported: **admit every miss 16.4 ms (6.0–24.9, ρ −0.91)**, fetch-on-admit 5.5, MIN two reads 3.8 | ✓; selective |
| 13 | Fewest-admission set: copies 0.60–0.73× greedy, misses +2.5–5.3%; +0.08–0.17 over greedy at 11% on 5 machines (6 launches); −0.01 on O4; every launch above 1 with either load | Sec. 4, Table 3 | 0.597 / 0.726; +2.5 / +5.3%; +0.082 to +0.172; −0.014; all 14 launch-budget cells have a CI above 1 (both loads). Not stated: **on O4 at 25% it is 0.03 (in step) and 0.07 (two reads) behind greedy** | ✓; incomplete |
| 14 | Layer-ahead copy: 1.04× on O4, 0.72–0.91× elsewhere; 84% of copies used | Sec. 5, Table 4 | 1.042 [1.034, 1.049]; 0.722–0.909; 83.8–83.9% (25%: 73%) | ✓ |
| 15 | Cross-machine Spearman: MIN in step 0.89 [0.63, 0.97] (19); prefetched 0.87 [0.54, 0.99] (17); admit every miss 0.95 [0.75, 0.99] (13); MIN two reads vs B_p 0.79 [0.49, 0.92], against 0.45 for the ratio | Sec. 4 | 0.88 [0.62, 0.97]; 0.87 [0.57, 0.98]; 0.95 [0.75, 0.99]; 0.79 [0.49, 0.92] vs 0.45 [−0.01, 0.77]. **Δρ = 0.34 [−0.15, 0.83]** | ✓; the contrast is not significant |
| 16 | Kendall's W 0.77 (11%) / 0.82 (25%) over 19 machines × 20 problems; machines 96% of variance, problems 1–3% | Sec. 4 | 0.77 / 0.82; 96.3% / 96.2%; 1.1% / 2.8% | ✓ |
| 17 | Relaunches within 1.9%, ratios within 0.030; same-CPU panel hosts differ by 16–29% | Sec. 2 | 0.0–1.9% and ≤ 0.030 over 6 relaunched machines; 15.6–29.1% | ✓ |
| 18 | Job 102 microbenchmark: per layer 86–96% of the host term | Sec. 3, Table 19 | 86.1–96.3% (per token 93–97%; repeats within 0.9%) | ✓ |
| 19 | Link-aware bound: at most +1% (11%) and +7% (25%) on the 24 probes of jobs 095–102 | Sec. 3 | +0.9% / +7.1% on 25 probes (frontier from `prereg/linkaware.json`, my own min-max). It also holds on the 11 probes of jobs 103–105: +1.0% / +7.3% | ✓ |
| 20 | Held-out W50 (Table 6): proportional 1.22 / 1.57 / 2.42; power law 1.19 / 1.51 / 1.67; distinct-expert rule 1.16 / 1.42 / 1.61; D(W50)/C 0.65 (0.61–0.72) | Sec. 5 (prereg JSON) | 1.20 / 1.61 / 2.44; 1.19 / 1.51 / 1.67; 1.18 / 1.39 / 1.65 (coarser D(W) interpolation); 0.651 (0.610–0.721). Distinct rule beats the power law on 13/26 points | ✓; the rules are indistinguishable |
| 21 | Published in-class median 13.6% (quartiles 8.1–20.6), n = 20 | Sec. 3, App. K (prereg JSON) | 13.6 (8.1–20.6), n = 20 | ✓ |
| 22 | Scorecard: 565 clauses (218 / 193 / 142 / 8 / 4); job 099: 273 (190 / 42 / 39 / 2); jobs 100–105 as stated; 90 of 301 bands failed from job 088 | App. B, supplement (prereg JSON) | Identical | ✓, but **job 072 (2 of 3 predictions failed) is absent** |
| 23 | GPU-in-series share of the gap: 22–51% (11%), 31–55% (25%) | Sec. 4 | 22–51% and 31–55%. This is an identity once G is implied per launch; with the profiled G it is 23–52% and 33–64% | ✓ (definitional) |
| 24 | Table 5 (job 100) ratios on Pd and the 9950X with the slower link | Table 5 | All 16 cells checked on those two hosts match | ✓ |
| 25 | Predictions committed before each machine started | Sec. 2, App. B | All 82 job scripts in the ledger were committed 3 s to 15 min before their first rental | ✓ |

Additional findings from the same scripts, not claims made in the paper:
- Host 100f: effective engine read rate 61.3 GB/s against probe B_host 52.0 GB/s (W2).
- Eq. (3) under other B_host definitions: 27/30 and 8/30 within 6% (W1).
- At 25%, law error against admissions share: ρ = −0.00 (W1).
- Best demand-reading oracle as a share of Eq. (2): ≤ 85% / ≤ 79% (W2).

## 5. Methodology against benchmarking rules

**Hoefler–Belli, rule by rule. The paper's Appendix A claims compliance; my assessment:**

| Rule | Assessment |
|---|---|
| 1. Base case | Satisfied. Absolute speeds appear next to every ratio. |
| 2. Subsets | Partially satisfied. Two models, one prompt set (AIME-25, also the development set), 256 tokens, batch 1. The panel uses the first 20 of 30 problems. There are no selection criteria for which Vast offers became panel hosts. |
| 3–4. Means and ratios | Satisfied. Ratios of means paired by problem; mean times and harmonic speeds are consistent (≤ 0.006, which I confirmed on Table 2). |
| 5. Confidence intervals | Formally satisfied, substantively weak (W3). The interval is over problems within one launch, the smallest variance component (1–3%). It omits machine (96%), launch (up to 0.03 in ratio) and time-position variance. It is nonetheless used to label clauses "held". |
| 6. Normality | Satisfied (percentile bootstrap). Problems are serially dependent through the carried-over cache, so a block or launch-level resampling scheme would be more honest. |
| 7. Comparing nondeterministic data | Good across machines (UUID merging, cluster bootstrap, crossed bootstrap for the panel). Weak within a launch: one contiguous block per configuration, no interleaving or repeats of the baseline. |
| 8. Central tendency | Mean only. p50/p99 are recorded but never analysed, although step times vary 3–7× within a problem. |
| 9. Setup | Well documented in files, but not controlled (see clocks below). |
| 10. Measurement | Adequate: in-process timer for the oracle jobs, streaming client for Table 2. |
| 11. Upper bounds | Present and correctly derived. Two gaps: the bound's rate is a single-run extreme-value statistic with no uncertainty (exceeded on 100f), and Eq. (2)'s GPU term is implementation-specific (W2). |
| 12. Plots | Adequate. |

**Number of machines.** 23 distinct GPUs over 34 engine launches is good for this area. Class-level claims have far smaller n:
- link/CPU < 1/3: 2 machines;
- "fast link" for the layer-ahead copy: 1 machine;
- fewest-admission schedule: 6 machines.

Pf alone contributes 5 launches. Claims about classes should be stated with machine counts and without within-host intervals as their support.

**Relaunches.** 6 machines were relaunched. The data show launch variance of up to 1.9% in time and 0.03 in ratios. This is the right experiment, but it is not propagated into any interval or into the scoring rule.

**Uncontrolled clocks and configuration.** Clock locking failed on every host (`clock-lock.txt`: no permission). Across hosts:
- power limits ranged 400–600 W;
- memory clocks were 14001, 15501 and 17001 MHz (host B, the headline host, is memory-overclocked);
- drivers ranged 580–610;
- PCIe ran at x8 or x16, Gen 4 or 5;
- cgroup CPU quotas ranged from 15.4 to 61.4 CPUs.

All of this is recorded, but none of it enters an analysis as a covariate. For host-memory-bound workloads on rented machines, co-tenancy and DRAM contention during the run (as opposed to the probe at job start) are the main threat. The 100f probe and STREAM variability suggest it is real. Nothing measures host DRAM bandwidth during the timed runs.

**Workload representativeness.** One math prompt set, short generations, greedy, teacher-forced, batch 1, two models in the engine. The cross-model "price of foresight" is simulation on each model's own sampled text, which is good, but it is not checked in an engine.

**Threats to validity the paper should list but does not:**
- probe noise and its effect on the bound and on the law;
- serial dependence across problems;
- time-position confounding within a launch;
- the development and evaluation prompt overlap;
- post hoc covariate selection (256-test family);
- O4 was the substitute host in job 105 although its class membership was already known from jobs 096 and 097. This is harmless, but should be said.

## 6. Questions for the authors

1. **Scope of Eq. (3).** Does it hold for any engine state other than the deployed policy at 11%? Why does admit-every-miss imply G ≈ 16 ms (range 6–25)? Please report implied G for every state in Table 1.
2. **Host 100f.** Its engine read 18% faster than its probe's maximum. How many probe repetitions did each host get? What is the run-to-run dispersion of B_host? Would you consider an upper-confidence B_host for the bound?
3. **25% mechanism.** Can you test the overlap explanation directly? For example, an Nsight overlap measurement of background copy kernels against compute, or a run with background admissions serialised.
4. **Job 072.** Why is it not in the scorecard? Are any other jobs in this study's scope carrying registered predictions outside 073–105? What happened to rentals 100e and the two replaced 099a attempts?
5. **Eq. (2)'s GPU term.** How should a reader apply Eq. (2) to a system with different kernels? Is T_GPU meant as a property of the model and hardware, or of this engine?
6. **"Instead of the ratio".** Was the B_p-versus-ratio contrast for MIN with two reads selected from the 256-test family? Can you test the difference between the two correlations?
7. **Configuration order.** Within a launch, did any configuration run twice (an ABA design) so that within-launch drift can be estimated? If not, what bounds the drift?
8. **Faithfulness of the plan.** Why does the engine follow the LP plan less faithfully at 25% (0.726 copy ratio against 0.643 in replay)?

## 7. What would move my score (ranked)

1. **Rescope the headline claims; no new experiments needed.**
   - State Eq. (3) as an exploratory relation for the deployed configuration at 11%, with its registered test passed at 11% and failed at 25% (abstract and introduction).
   - Report the 17/30 count at 25% and the implied G of every state.
   - Recast Eq. (2) as conditional on the engine's measured GPU kernels.
   - Restrict "nearly reachable" to the read term.
   - Remove or test the "B_p instead of the ratio" contrast.

   With these changes alone I would hold at 6 with more confidence.
2. **Put uncertainty on the bound's rate.** Repeat the probe at least 5 times per host, before and after the timed runs. Report the dispersion. Either use an upper confidence value for B_host in Eq. (1) or show that no engine state on any host exceeds it. Explain 100f.
3. **Grow the slow-link class.** Add at least 4 more distinct machines with link/CPU < 0.4, with both causes represented (narrow links and wide DRAM). Run the fewest-admission and layer-ahead configurations, with quantitative bands registered from the calibrated model rather than signs.
4. **Fix the variance structure for key single-host claims.** Use interleaved repeats within a launch (for example ABAB baseline/variant blocks) or at least 2 launches per machine. Report intervals that include launch variance, and score clauses against them. For cross-machine claims, fit a mixed model with a machine random effect.
5. **Complete the prediction log and the exclusion table.** Add job 072 to the scorecard. Add one row per rental with its outcome and the rule that excluded it.
6. **Test the 25% mechanism directly** (see Question 3), or drop the explanation.
7. **Clarity.**
   - Halve the number of named configurations in the main text.
   - Define B_host operationally in Section 2.
   - Move the narrative scorecard to the supplement.
   - Replace multi-range sentences with one decomposition figure per budget.
8. **Workload.** Add one non-math prompt set (code or chat) and one longer generation to the engine experiments, for at least the deployed cache, the fewest-admission set and Eq. (3).

## 8. Scores

| Item | Score |
|---|---|
| **Overall** | **6 (weak accept)**. Conditional on item 1 of Section 7; without that rescoping, 5. |
| Soundness | 3 / 5 |
| Methodology | 3 / 5 |
| Significance | 3 / 5 |
| Clarity | 2 / 5 |
| Confidence | 4 / 5 |

Justification: I recomputed 25 central numbers from raw data and all of them reproduce. Pre-registration is verifiably ordered before every rental, and failures are reported at a scale this field rarely sees. Against that:
- The headline relation holds for one configuration at one budget and failed its registered test at the other.
- The bounds rest on a single-run probe maximum that the engine itself exceeded on one machine, and no full system comes within 15% of either bound.
- Single-host intervals capture only the smallest variance component.
- The slow-link conclusions rest on two machines.
