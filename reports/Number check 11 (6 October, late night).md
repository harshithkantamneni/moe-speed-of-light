# Number check 11 (6 October, late night)

## Scope

This check covers every change to `paper/paper.tex` and `paper/app_wsg.tex` between 51067fc and HEAD (842157e). It also covers the generated inputs that feed those changes: `wsg_sumlaw.tex`, `wsg_job105.tex`, `tab_job105.tex`, `tab_scorecard_105.tex`, `wsg_hostdep.tex` with `figs/hostdep.pdf`, `prereg/job105.json`, `prereg/scorecard_105.json`, `prereg/reanalysis.json` and `prereg/sumlaw_outcome_105.md`.

I recomputed the numbers with my own code, which is in the scratchpad `nc11/` (`common.py`, `j105.py`, `launches.py`, `sumlaw.py`, `plan.py`, `corr.py`, `perprob.py`). The code works from the raw files on the gpu branch:

- per-problem `decode_ms/n_decode` from `ec_g_C*.jsonl`, paired by problem against `base`;
- counters from `st_g_C*_<config>.json`;
- the probe rates (B_c, B_p, B_cp, B_host), re-implemented from `concur.txt`;
- the profiles' `median_ms` categories;
- machine identity from the GPU UUID in `nvidia-smi-q.txt`;
- MIN with bypass and the deployed decayed-count policy, written independently and run on the 084c trace.

The authors' scripts were used only for comparison. I built the paper in a scratch copy of `paper/` with latexmk.

## Defects (most severe first)

### 1. Medium–high: the central law is stated more broadly than the data support

**Where:**
- Abstract, l. 58–62: "Across \slLaunches{} launches on \slMachines{} rented RTX 5090 machines, our llama.cpp expert cache … takes the GPU's compute plus its host reads at the machine's best rate, with nothing fitted".
- Intro (1), l. 87–89: "across 30 launches on 21 machines its time is the profiled GPU compute (4.1–4.7 ms) plus its counted reads at the machine's best rate, with nothing fitted".
- Contribution 2, l. 119–121: "A law with nothing fitted … on 30 launches on 21 machines and predicted before the runs on 4 more".
- Conclusion, l. 466: "batch-1 MoE decode is the GPU's compute plus host bytes".

**Evidence (own recomputation over the 30 launches, jobs 093–104):**
- **At gpt-oss 11%:**
  - The implied G (T − R·S/B_host) has median 4.33 ms and interquartile range 4.14–4.59, but ranges from 1.80 ms (100f) to 7.89 ms (099i).
  - With G = 4.3 ms, the law is within ±6% on 28 of 30 launches. It misses 100f by +13.8% and 099i by −10.0%.
  - Taking any G in the profiled range 4.07–4.70, 100f is off by +12.5% to +16.0%.
- **At 25%:**
  - The implied G has median 3.85 ms and IQR 3.64–3.99. 25 of 30 launches fall below the lowest profiled value (4.07).
  - With G = 4.4 ms, the law over-predicts by a median of +5.0% (−2.6% to +17.2%) and is within 6% on only 23 of 30. With G = 4.7 it is within 6% on 8 of 30.
- **The pre-registered test (job 105) failed at 25%:**
  - The 6% band failed on three of four hosts.
  - The pooled-median clause failed (4.24% against at most 4%).
- **Other limits:**
  - Only gpt-oss-120b was tested; no Qwen3 launch enters the law.
  - None of the 21 machines was profiled at the time of its runs. The profiles come from the three 069c hosts and the four job-105 launches.

**Fix:**
- Scope the abstract, intro (1), contribution and conclusion to "gpt-oss-120b at 11%". Give the range or the outliers, for example: "within 6% on 28 of 30 launches; implied G 1.8–7.9 ms".
- Say plainly that at 25% the law over-predicts, both in the reanalysis and in the pre-registered test, where 3 of 4 cells and the median clause failed.
- Drop "predicted before the runs on 4 more" from the contribution, or add "held at 11%, failed at 25%".

### 2. Medium: the intro misidentifies which relation the cache sits on

**Where:** Intro (1), l. 86–88: "for a system that reads on demand, the GPU's own compute adds in series. Our cache sits on the second relation itself".

**Evidence:** "The second relation" is Eq. (demand): T_GPU (non-expert work, 2.9–3.4 ms) plus MIN's reads R* at B_host. The cache is far from that bound:
- At 11% it stands at 55–70% of Eq. (demand), as Section 3, l. 207–208, itself says. My recomputation gives 55.2–70.3% by launch and by machine.
- In the running example (O4, 096a), Eq. (demand) is 12.96 ms against 20.64 ms measured.

What the cache follows is Eq. (sum), with its own reads R (63–69 per token, against R* = 38.3) and the full G (4.1–4.7 ms).

**Fix:** "Our cache's time has the same series form with its own reads (Eq. sum): …".

### 3. Medium: the failed clauses are misreported, and the main text omits one

**Where:**
- Section 4, l. 252–253: "at 25% it over-predicted by \jgErrMidMin–\jgErrMidMax\% [3.8–9.3%], beyond the 6% we had registered".
- Appendix B, `app_wsg.tex` l. 118–120: "Every failure is the law at gpt-oss 25%, which over-predicts the deployed cache by 3.8–9.3% against at most 6% on three of the four hosts".
- `prereg/sumlaw_outcome_105.md`: "At 25% it failed on three of four hosts: the law over-predicts by 3.8–9.3%", and "the paper says so in Section 4, Limitations and Appendix B".

**Evidence:** My errors at 25% are:

| Host | Error | Clause |
|---|---|---|
| EPYC 7402P | +9.26% | failed |
| TR 9960X | +7.03% | failed |
| O4 | +6.92% | failed |
| Pf | +3.78% | held |

The three failures therefore span 6.9–9.3%. The 3.8% belongs to the host that held, so "beyond the 6%" is false for one of the four. Two further problems:
- The pooled-median clause (105-P2-median, 4.24% against at most 4%) is not mentioned in Section 4 or in the Limitations. Only Appendix B reports it.
- The Limitations do not say that a registered prediction failed.

**Fix:**
- Section 4: "at 25% it over-predicted by 6.9–9.3% on three of the four hosts (3.8% on Pf), beyond the 6% we had registered, and the pooled median error (4.2%) exceeded the 4% registered".
- Appendix B: "… over-predicts by 6.9–9.3% (against at most 6%) on three of the four hosts …".
- Correct the outcome note to match.

### 4. Low–medium: the serialisation share and the elasticity use the two-path model's fitted G, not the profiled one

**Where:**
- Abstract, l. 61: "(24–55% at the smallest budget)".
- Intro, l. 92.
- Section 4, l. 253–257: "With G removed … elasticity of −0.97 … the GPU's compute in series with the reads, 24–55% of the gap at 11% and 34–65% at 25%".
- Section 4, l. 260–261, where "The calibrated model … extends Eq. sum … T = G + max(…)" reuses the symbol G.

**Evidence:** `scripts/reanalysis.py` `sum_law()` uses G = 4.56 / 4.53 ms. These are the constants frozen in `jobs/ec2/predict_100.py`: "the median of the per-host-cell fits of jobs 095–099", that is, fitted constants of the two-path model. They are neither the profiled kernel time nor the implied G. My shares of the gap:

| G used | At 11% | At 25% |
|---|---|---|
| 4.56 / 4.53 (the script's constants) | 24.0–55.2% | 33.7–64.6% |
| 4.3 | 22.6–52.0% | 32.0–61.3% |
| Each launch's implied G | 21.7–50.7% | 31.5–54.9% |

At 25%, where the paper itself says part of G overlaps (implied 3.85), 34–65% overstates the serial share.

The elasticity is robust: −0.97 with 4.56, and −0.95 [−1.02, −0.87] with 4.3.

In the two-path model, G is "the GPU's work and fixed costs, one constant per budget taken from the other hosts' fits". The old text said so; the new text drops it and uses the same symbol as the profiled G.

**Fix:**
- Compute the shares with the profiled G, or with each launch's implied G (at 25% the implied G is the right choice), or state that 4.56/4.53 ms is the fitted constant of Appendix C.
- Give the two-path constant its own symbol, or restore "one constant per budget fitted on other hosts".

### 5. Low–medium: the Limitations understate the law's error

**Where:** Limitations, l. 457–458: "Eq. (sum) … over-predicts by up to \jgErrMidMax\% [9.3%] at gpt-oss 25%".

**Evidence:**
- 9.3% is the maximum over job 105's four launches only.
- On the 30 launches of the reanalysis, using any G in the profiled range 4.07–4.70, the over-prediction at 25% reaches +14.3% to +19.8% (100f).
- At 11%, 100f is over-predicted by +12.5% to +16.0% and 099i under-predicted by −8.9% to −10.6%.

**Fix:** "over-predicts at 25% (by 3.8–9.3% on the job-105 launches, and by up to 14–20% on one of the 30 earlier launches), and misses two of 30 launches by 10–14% at 11%".

### 6. Low: Section 4 undercounts launches "in all"

**Where:** Section 4, l. 235–236: "and on follow-up launches, \slLaunches{} launches on \slMachines{} machines in all".

**Evidence:** The 30/21 counts come from `reanalysis.json`, which excludes job 105. Section 4 also reports job 105 (Table `tab:job105`, and the fewest-admission set over jobs 104 and 105). With job 105 included, the totals are 34 launches on 23 machines, since 105a and 105b are new GPU UUIDs and 105e and 105f share Pf's and O4's UUIDs.

**Fix:** "34 launches on 23 machines in all (30 on 21 before job 105)", or drop "in all".

### 7. Low: the claim that both signs were predicted covers a host that had no prediction

**Where:** Section 5, l. 403–405: "loses on the other \jgPfLoseN{} [3], down to 0.72× … both signs were predicted from the link-to-CPU ratio before the runs".

**Evidence:** The three losses are on 105a (EPYC, ratio 0.448, 0.81×), 105b (0.32) and 105e (0.28). The header of `jobs/105_sumlaw@vast.sh` registers a loss only for ratios below 0.4 and a gain only for ratios of at least 0.8. 105a had no prediction, as the outcome note itself says.

**Fix:** "… the gain on O4 and the losses on the two hosts below 0.4 were predicted before the runs; the EPYC 7402P (0.45) had no prediction".

### 8. Low: "wherever the link is slower than the CPU" generalises past the tested range

**Where:** Intro (2), l. 102–103: "wherever the link is slower than the CPU it adds 0.08–0.17 to the speed ratio at gpt-oss 11%".

**Evidence:** This was measured on 6 launches on 5 machines with ratios 0.28–0.63. My gains are +0.082 to +0.172. No machine between 0.63 and 1.0 ran the plan.

**Fix:** "on the five machines whose link reads at 0.28–0.63 of the CPU rate".

### 9. Low: "every state that copies over the link" contradicts the next sentence

**Where:** Section 4, l. 305–310: "the gain of every state that copies over the link follows the link-to-CPU ratio … MIN loaded by the CPU, whose copies run in the background, follows the link's absolute rate instead (0.79 …, against 0.45 for the ratio)".

**Evidence:** MIN loaded by the CPU also copies over the link, and its correlation with the ratio is 0.45 [−0.03, 0.77] (n = 17 machines). Two further gaps:
- The sample sizes are not stated: 19 machines for the in-step copy, 17 for the copy ahead, 13 for admitting every miss.
- These correlations exclude job 105's two new machines.

**Fix:** "every state that copies over the link in the step"; add the n's.

### 10. Low: the per-problem paragraph is scoped loosely

**Where:** Section 4, l. 326–330.

**Evidence:**
- The Spearman 0.88 is at 11% only: my value is 0.875, against 0.683 at 25%.
- "Machines carry 93–96%" pools the in-step copy (96.3% and 96.2%) with the copy ahead (92.5% and 93.7%). The sentence is about the in-step copy alone, for which machines carry 96% and problems 1–3%.
- "Problems rank the same on every machine" is too strong for W = 0.77–0.82.

**Fix:**
- "(Spearman 0.88 at 11%, 0.68 at 25%)".
- "machines carry 96% of the variance of its log gain (93–96% with the copies made ahead)".
- "Problems rank largely the same".

### 11. Low: the conclusion overstates three points

**Where:** Conclusion, l. 466–473.

**Evidence:**
- "in series for any system that reads on demand" conflicts with Section 4 and the Limitations. There, the deployed cache, which reads on demand, overlaps background copies with the GPU's work at 25%.
- "Knowing the future removes both" overstates. With the copies made ahead, the implied G is still 2.7 ms, not 0, and copies in the step raise it to 7.4 ms.
- "about 0.65 C distinct experts … to halve the excess reads": W50 halves the read gap between admitting every miss and MIN (Section 5, l. 385–389). It does not halve the deployed cache's excess over MIN, which is what "excess reads" means two sentences earlier.

**Fix:** "in series, apart from background copies, for a system that reads on demand"; "reduces both"; "to halve the read gap between admitting every miss and MIN".

### 12. Low: "predicted before the runs" describes the law loosely

**Where:** Intro, l. 89; Contribution 2; Section 4, l. 249–250; the `tab_job105.tex` caption ("predicted before the runs, nothing fitted").

**Evidence:** G_prof was written before any timed run. I checked this in each `stdout.log`: "G_prof written …, before any timed run". But R in G + R·S/B_host is each timed run's own counters (`st_g_C*_base.json`), as registered. What was fixed in advance is the relation and G, not the predicted time.

**Fix:** "registered before the runs and evaluated with each run's counted reads".

### 13. Low: "reachable" in the intro, "nearly reachable" elsewhere

**Where:** Intro (1), l. 92–93: "The bounds' read time is reachable: … reach 86–96% of it". Contribution 1 and Section 3 say "nearly reachable".

**Fix:** "nearly reachable".

### 14. Low: the elasticity interval resamples launches, not machines

**Where:** Section 4, l. 253–254: "(95% interval −1.04 to −0.88)".

**Evidence:** The bootstrap in `reanalysis.py` resamples the 30 launches, so relaunches of 6 machines enter as independent. Resampling machines gives [−1.08, −0.86] at 11%. The point estimate is unchanged (−0.968).

**Fix:** Report the machine-level interval, or say "over launches".

### 15. Low: two errors in `prereg/sumlaw_outcome_105.md`

- "Loaded by the CPU, the fewest-admission set gains on every host at both budgets (1.02–1.22)". Job 105's range is 1.02–1.21; the maximum is 105e at 25%, 1.2149. The 1.22 appears only if job 104a (1.2169) is included.
- The 3.8–9.3% range and the "the paper says so in Section 4, Limitations" sentence carry the same issue as Defect 3.

## Checked and correct

**Job 105 table and macros**
- Every cell of `tab_job105.tex` matches my recomputation from raw files: G_prof, the law, the measured time, and the layer-ahead, greedy and fewest-admission speeds.
- The `jg*` macros match: G_prof 4.07–4.50 ms; errors at 11% from −1.49% to +4.70% (median 1.72%); errors at 25% from +3.78% to +9.26%; pooled median |error| 4.24%; layer-ahead copy 1.042× on O4 and 0.72–0.91× at a ratio below 0.4.
- Pf-useful share: 83.8–83.9% at 11% and 72.9–73.1% at 25%. Copy ratio 0.597–0.726. Fewest-admission misses +2.5–5.3%.
- Probe rates in `prereg/job105.json`: B_host 60.4, 178.2, 96.1 and 51.1 GB/s, and ratios 0.448, 0.321, 0.281 and 1.038.

**Scorecard (`scorecard_105.json`, `tab_scorecard_105.tex`)**
- 31 clauses: 7 held, 20 held on the point estimate, 4 failed.
- The clause set is complete against the five registered predictions in the header of `jobs/105_sumlaw@vast.sh`: P1 ×8, P2 ×8 plus the median, P3 ×3 (none for 105a at 0.45), P4 ×4, P4g ×3 (ratio below 0.7) and P5 ×4.
- Each status agrees with my values and intervals, for example P3 on O4 at [1.034, 1.050] against the 1.03 threshold.

**Pre-registration timing**
- The header commit 92ff9fd is dated 22:05:31Z. Starts, taken from `manifest.json` and the order seeds (T0 mod 100000): 105a at 22:06:11Z, 105d at 22:06:22, 105e at 22:06:41, 105b at 22:06:48.
- The amendment a8e2f92 is dated 22:09:26Z, before 105g (22:09:57Z) and 105f (22:10:22Z). The predictions are unchanged in that diff.
- Margins are 40 s or more. Commit dates are self-reported.

**Machines and hosts**
- By GPU UUID, 105e is Pf (099f, 101a, 103a, 104a) and 105f is O4 (096a, 097a). 105a and 105b are new.
- 104b is a different 285K from Pf.
- 105d and 105g were stopped by the gate (device read 1072 and 1329 GB/s, against 1500). 105c has no results directory.
- 30 launches on 21 machines in `reanalysis.json`, all RTX 5090.

**Profiles and bounds**
- 14 profiles on 7 hosts: G 4.07–4.69 ms, rounded to 4.1–4.7; non-expert part 2.94–3.38 ms, rounded to 2.9–3.4.
- The 069c profiles give 4.28–4.69 ms, matching the header.
- O4 at 11%: Eq. (limit) 10.0 ms, Eq. (demand) 13.0 ms, base 20.6 ms.
- At 11% the cache stands at 38–54% of Eq. (limit) and 55–70% of Eq. (demand), by launch and by machine. No launch is faster than Eq. (demand).

**Sum-law reanalysis**
- Implied G: median 4.3 ms at 11% (IQR 4.1–4.6), Spearman with the ratio 0.02; median 3.8 ms at 25%.
- Implied G with copies made ahead: 2.7 ms. With copies in the step: 7.4 ms, Spearman −0.91 with the ratio.
- Elasticity −0.97 [−1.04, −0.88], launch bootstrap.
- Admissions are 5.5–11.1% of the counted reads at 11% and 12.6–22.2% at 25%, as the outcome note says.

**Fewest-admission set (jobs 104–105)**
- 7 launches on 6 machines, ratios 0.28–1.04.
- Gain over the greedy set: +0.08 to +0.17 on 6 launches; −0.01 on O4.
- On the two machines below a third: greedy 0.86–0.91×, fewest 1.03–1.04× (3 launches).
- Every fewest-admission interval, at both budgets and with either load, lies above 1. The lowest bound is 1.015.

**Greedy copy and copies made ahead**
- The greedy in-step copy loses (below 1) only on Pf's launches and on the TR 9960X, at both budgets, across all 34 launches.
- The copies made ahead are worst at 0.997 [0.987, 1.007] on Pf (101a).

**Rank correlations and per-problem statistics**
- Rank correlations: 0.89 [0.63, 0.97], 0.87 [0.54–0.56, 0.99], 0.95 [0.75, 0.99], 0.79 [0.49, 0.92], 0.45.
- Kendall's W 0.77 and 0.82 over 19 machines and 20 problems.
- Per-problem Spearman 0.88 at 11%, using my own MIN-with-bypass simulator, which reproduces R* = 38.315 and 15.340, and my own decayed-count simulator.
- Variance shares 93–96% and 1–4%.

**Figure and appendix**
- The host-dependence figure has 6 stars per panel, one per machine (104a, 104b, 104c, 105a, 105b, 105f), and crosses for the layer-ahead copy on the 4 job-105 hosts.
- The changed `hd*` macros are not used in the paper text.
- The appendix's host list, 105c/105d/105g account and "every speed prediction held" are correct.

**Build**
- latexmk on a scratch copy finishes with no errors and no undefined references, citations or macros. No "[pend.]" markers appear.
- The main text ends on page 9 (Conclusion, then References on page 9).
- The supplement builds and includes Table 8 (job 105).
- The duplicate-anchor and overfull-box warnings already occur at 51067fc.
