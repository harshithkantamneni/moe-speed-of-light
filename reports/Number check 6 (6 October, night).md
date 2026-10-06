# Number check 6 (6 October, night)

Independent check of `paper/paper.tex` (abstract to Conclusion, main-text tables and captions), `app_value.tex`,
`app_traces.tex`, and `app_wsg.tex` lines 78-197 (machine scoring, job 099, calibrated time model).

## Method

- Resolved all 308 macros used in scope from `paper/wsg_*.tex` in `\input` order (no duplicate `\newcommand`; no
  `\pend` placeholders resolve in scope; the log has no undefined control sequences).
- Recomputed independently with my own Python, without running the generating scripts:
  - job 099, from the raw files in `/home/claude/gpu-branch/results/099?_panel@vast`: per host and cell, t = mean over
    problems of decode_ms/n_decode; time share (t_aa - t_x)/(t_aa - t_fetch); read share from the st_ counters
    (misses + admits + prefetches)/steps; link share fetches/misses; B_c, B_p, B_cp, B_host from concur.txt and cores.txt;
    engine against value_map_replay.jsonl; within-host and two-stage bootstraps.
  - hosts O3-O6 (jobs 095, 096a/b, 097a/b) from the raw ec_/st_ files; the bound from `prereg/speed_limit_v2.json`
    (R*, S, D) and each host's probe; the 2x2 Shapley accounting from the raw times; the host-dependence model
    T = G + max(X_c/B_c, X_p/B_p, (X_c+X_p)/B_cp) from the counters.
  - `prereg/foresight/*.json`, `policy_aa.json`, `policy_study.json`, `value_map.json`, `forecaster_linear.json`,
    `forecaster_gru.json`, `scorecard_auto.json`, `scorecard_099.json`, `scorecard_clauses.json`,
    `audit/audit_ours.json`, `audit/audit_sol.json`, `learned_offline.json`, `batchk_trace.json`,
    `freetoken_tuned_098.json`, job 063's look-ahead file, `tab_headline` and `tab_limit`.
- Checked prose wording (ranges, "every", "never", "about", orderings, units) against the recomputed values, and the
  three figures in scope (rendered PNGs) against their captions.

## Defects

### Errors

1. **paper.tex:170-172** (How tight is it?). Says the measured-GPU rate "moves the bound at the GPU-bound budgets and
   leaves the host-bound ones unchanged, because there the host term binds". **Data:** `tab_headline` and `tab_limit`
   show it moves at two of the four host-bound budgets. gpt-oss 25% goes from 430 to 279 tok/s on host B and from 351
   to 275 on host S. Qwen3 25% goes from 214 to 192 on B. At the measured rate the minimum of eq. 1 moves off c=0:
   c* = 23.6 (gpt-oss 25%, B), 19.6 (S) and 48.3 (Qwen3 25%, B). Only the 11% and 12.5% budgets, and Qwen3 25% on S,
   stay unchanged. **Severity:** error. **Fix:** "leaves the two smallest budgets unchanged; at gpt-oss 25% (and Qwen3
   25% on host B) the measured GPU term takes over."

### Overclaims

2. **paper.tex:228-232, 84, 309-310, 396** (host-dependence model). Says "the calibrated time model of app:law, given
   each host's probe and each state's own counters, predicts...". **Data:** `scripts/hostdep_model.py` refits G on
   every host-cell from that host's measured deployed time. Across host-cells G is 4.1-9.2 ms; app:law's frozen G is
   5.17 ms. My recomputation of the published numbers matches: N=208, median 2.22%, p90 9.49%, 43/43 signs, 68
   background rows at median 18.0% (all over-predicting speed), 60 window shares at median 0.031 and max 0.100, allr5
   max +0.244 (Pj). With app:law's frozen G = 5.17 ms (gpt-oss, 178 in-step rows) the median is 4.3% and p90 10.9%.
   The deployed state itself is then off by -11% to +8%. The fetch sign is still right 30/30. **Fix:** say G is
   calibrated per host and budget on the deployed state, so the model predicts ratios to the deployed cache. Or quote
   the frozen-G errors.
3. **paper.tex:85-86** (Intro). Says the usual ways "read 1.6-2.5x MIN's bytes and lose at the smallest budgets".
   **Data:** on O4 at gpt-oss 11% Belady 1 read is 1.047x, Belady 2 reads 1.040x and MIN 2 reads 1.024x. At Qwen3
   12.5% Belady 1 read is 1.148x. On the panel at 11%, MIN 2 reads is 0.95-1.03x, with 7 of 10 hosts at or above 1.0.
   The section's own range for the best usual way at the two smallest budgets is 0.94-1.15x (`fxUsualLow`).
   **Fix:** "and at the smallest budgets gain at most 15% or lose".
4. **paper.tex:353-355; app_value.tex:30-31** (linear forecaster). Says it closes "much less than a window with random
   errors at that precision". **Data:** the forecaster's precision decays with horizon: 0.48, 0.34, 0.29, 0.26 at
   h=1-4 and 0.23 at h=8 on gpt-oss. A fill window has precision r at every horizon. At gpt-oss 11% the linear
   forecaster closes 0.130 (W=4) and 0.146 (W=8). A random window at r=0.3 closes 0.117 and 0.146. Interpolated to
   the forecaster's mean precision over h=1-4 (0.34), it closes about 0.14. So at gpt-oss 11% the forecaster equals a
   random-error window of matched precision. The gap only appears against r=0.5, i.e. its 1-step precision. On
   Qwen3 the claim holds: 0.19 against about 0.26 at matched mean precision 0.40. **Fix:** compare at the
   horizon-averaged precision, and restrict the claim and its explanation to Qwen3 (and gpt-oss 25%).
5. **paper.tex:410** (Conclusion). Says "our cache reaches 25-43% of it and published systems less". **Data:** this
   compares our bound (25-43%) with the audit procedure. Under the common procedure ours is 18-42% (median 27%). The
   20 in-class published rows run 2.3-36.8%. Seven of them exceed our minimum (18.6-36.8) and four exceed our median
   (27.9, 31.1, 36.5, 36.8). **Fix:** "published systems a median 13.6% (ours 27% by the same procedure)".
6. **paper.tex:315-316; app_value.tex:20** (fill against drop). Says dropping costs "about the same" as replacing.
   **Data:** this holds for finite windows (|diff| <= 0.033 at r >= 0.5 and W = 4-16; at most 0.057, Qwen3 12.5%
   W=16 r=0.3). For the whole future, drop is clearly better. At r=0.3 fill/drop is 0.17/0.29 (gpt-oss 11%), 0.21/0.30 (25%), 0.22/0.30 (Qwen3 12.5%) and 0.22/0.34
   (Qwen3 25%). At r=0.5 it is 0.43/0.48 at gpt-oss 11%. `vmFillDropMax` = 0.12 is a 66% relative difference.
   **Fix:** "about the same for windows up to 16 tokens at recall 0.5 and above; for the whole future at low recall,
   dropping is better by up to 0.12".

### Imprecise

7. **paper.tex:222, 226, 230-231** ("slowest-link host" = Pf). **Data:** Pf's link is 26.6 GB/s. Pi's (19.0) and
   Pg's (25.5) are slower, and Pi gains 1.34x at 11%. Pf has the lowest link/CPU ratio (0.29) because its CPU path is
   the panel's fastest (93.0 GB/s). The other 285K (Pa) differs in both paths: link 48.9 against 26.6, and CPU 60.0
   against 93.0. "the same CPU on a board with a faster link" (line 223) therefore omits the 35% slower CPU-side
   rate. **Fix:** "the host with the lowest link-to-CPU ratio". For Pa: "the same CPU model with a faster link and a
   slower CPU-side read rate (ratio 0.82) gains 1.33x".
8. **paper.tex:101-102, 204, 394-395** ("factorial on three hosts at six budgets"; "full design"). **Data:** O3 (job
   095) ran base, fetch, both2, both3p, hitopt and lead2. It has no single read (foa), no MIN 2 reads (bypass) and no
   nb2/hitoptp. The crossed 2x2 ran only on O4 and O5, as line 246 says. **Fix:** "the 2x2 on two hosts (O4, O5) at
   six budgets; O3 at six budgets without the single-read and two-read cells".
9. **paper.tex:111** ("Prompts are the 30 AIME-25 problems"). **Data:** job 099 ran 20 problems (`NSEQ=20` in
   jobs/099_panel@vast.sh; steps = 5120 = 20 x 256). The read shares of fig:value and tab_value come from the
   30-problem trace (aa 58.64 there, against 58.37 in the panel's own 20-problem replay). **Fix:** state that the
   panel uses the first 20 problems.
10. **paper.tex:287-288** ("1.6% apart on the panel's GPUs"). **Data:** the engine against replay is at most 1.60% at
    gpt-oss 11%, 1.69% at 25% and 0.23% at Qwen3 12.5%. Host Pd has no replay. **Fix:** "within 1.7%"
    (`pnReplayMaxMid`), or qualify with "at 11%" as app_value does.
11. **paper.tex:143** ("app:prereg scores all 565 of them"). **Data:** 565 covers jobs 073-098. Job 099 adds 273
    clauses, scored separately (838 in total). **Fix:** "all 565 clauses of jobs 073-098, and job 099's 273 by
    machine".
12. **app_wsg.tex:80-83** (automatic re-scoring). (a) Says "36 ... that the rule holds outright because no interval
    exists". The rule stated at app_wsg.tex:26-27 gives "held (point)" when no interval exists. The script follows
    the rule (auto = held (point)) and the hand score says held. So the sentence contradicts the rule, and the hand
    totals (`scHeld` 218) include 36 clauses the written rule would class as held (point). (b) Says "Of the clauses
    held on the point estimate only, 213 have no interval and 7 ... cross". These count the script's 220 held-(point)
    clauses, not the hand total `scHeldPoint` = 193 quoted at line 29. **Fix:** add the deterministic-count exception
    to the rule at line 26. Write "Of the 220 clauses the script scores held (point), ...".
13. **paper.tex:180** (audit quartiles "8.2-23.1%"). **Data:** Q3 is taken as `inclass[3n//4]`, the 16th of 20,
    about the 79th percentile. The standard 75th percentile of the 20 in-class rows is 20.6 (linear) or 21.5
    (midpoint). Q1 is 8.1-8.2 either way. **Fix:** use a standard quartile (about 21%).
14. **app_wsg.tex:149-151** ("sec:gap measures ... over-predicts ... by up to 53%"). **Data:** sec:gap
    (paper.tex:265-266) reports +1 to +38% for MIN prefetched on O3-O5. `fspLawErrHighMax` = 53 is job 094's Belady
    prefetch (prefetch_w0, gpt-oss 25%: +52.7%), which is described in app:foresight, not sec:gap. **Fix:** cite
    +1 to +38% (`fxPacedLawTime`), or point to app:foresight.
15. **paper.tex:215-216** ("MIN's way beats the best usual way by 1.18-2.03x"). **Data:** the macro is MIN
    *prefetched* over the best usual way. The sentence's antecedent is "MIN with one read", and that one loses to
    Belady 2 reads prefetched on O4 at gpt-oss 25% (1.428 against 1.524, 0.94x). MIN 1 read over best usual is
    0.94-1.94x. **Fix:** "MIN prefetched beats the best usual way by 1.18-2.03x".
16. **Method (prereg/foresight_09x.json feeding fx\*, hd\* for O-hosts).** `runs[*].mean` is the arithmetic mean of
    per-problem tok/s, and `ms` = 1000/mean. That contradicts paper.tex:137-138 and rule 3 (harmonic). `ratio_to_base`
    differs from the ratio of mean times by at most 0.0064. The only printed change is `fxMinOverUsualMin` 1.18 ->
    1.19 (1.189). **Fix:** regenerate from mean ms.
17. **paper.tex:90, 312** ("8 tokens at 50% recall buy about what 2 exact tokens do", in the engine). **Data:** W=2
    never ran at gpt-oss in the engine. The time share of W=8 at r=0.5 is 0.20-0.26. The 2-token comparison only
    exists in reads (0.29 against 0.31, tab_value) and on Qwen3 12.5% in the engine (0.37-0.39 against 0.35-0.45).
    **Fix:** cite the Qwen3 engine numbers or say "in reads".
18. **paper.tex:259-260** ("the single read in the step costs time (Shapley -21 to 2%)"). **Data:** Pd at 11% is +2
    (it gains). The four values are Pd +2/-10 and Pf -12/-21. **Fix:** "costs time except at Pd 11% (+2%)".
19. **paper.tex:139-140** ("two rentals of one CPU model differed by 19-22% in speed"). The figure is hard-coded and
    unsourced; the nearest source is prereg/slowlink_outcome_088.md, +19%. **Data:** pairs in this paper differ more:
    Pa against Pf (both 285K) by 29%/16% at 11%/25%, O5 against O6 (both 9950X3D) by 18-34%, Pe against Pj (both
    9950X) by 22%/17%, and O3 against O4 (both 9950X) by 8-18%. **Fix:** make it a macro over the paper's own
    same-CPU pairs (e.g. "8-34%").

### Cosmetic

20. **paper.tex:231.** "over-predicts the states whose copies run in the background" means the model over-predicts
    their *speed* (all 68 errors are positive). Line 266 uses "over-predicts ... time" in the opposite sense. Say
    "speed".
21. **paper.tex:227-228.** "at parity only where the link alone reads as fast as CPU and link together". The parity
    hosts (Pb, Pe; aa/base 0.96-1.00) have B_p/B_cp = 0.92 and 0.91. Say "nearly as fast".
22. **paper.tex:224-225.** `hdFetchCorr` 0.89 pools both budgets (28 points). It is 0.93 at 11% and 0.88 at 25%.
23. **paper.tex:230, 310.** `hmFetchN` 43 and `hmErrStepN` 208 count O4's machine twice (jobs 096a and 097a), giving
    39 distinct host-budgets. `hmWinShareN` 60 counts host-cell-windows (20 host-cells x 3 windows), not "host-cells".
24. **paper.tex:138, 141-142.** The two-stage bootstrap uses 2,000 resamples (panel_099.py), not 10,000.
25. **paper.tex:258.** `\fsBalPacingMin--\fsBalPacingMax` renders "-4-23%". Write "-4 to 23%".
26. **app_traces.tex:41.** `lreUs` 237-585 us includes the GPU-bound budgets. At the host-bound budgets the sentence
    is about, it is 275-585.
27. **paper.tex:149.** R* is defined "per token, per layer", but eq. 1 uses R* per token summed over layers (38.3 at
    gpt-oss 11%).
28. **fig:factorial caption (paper.tex:194-200).** It omits the plotted "O4, job 097" series and the panel ticks at
    Qwen3 12.5% (3 hosts). It says the panel is at "the two gpt-oss host-bound budgets" only.
29. **tab_configs; paper.tex:282-283.** "victim seen furthest ahead, else lowest-scored" reverses the priority stated in
    app_value.tex:6-8 and implemented in the engine. The engine takes the lowest-scored resident *not seen* in the
    window first, and the furthest-seen one only when every resident is seen. Reword.
30. **Hard-coded numbers that should be macros.**
    - tab_headline caption: "52%", "61%", "255", "178" (`gpuEffPctGpt/Qwen`; vram tok/s 255.3/177.8). The values are
      correct.
    - paper.tex:399 "5-10%": the pooled bound gains 4.7-10.2% at host-bound budgets and 0 at GPU-bound ones.
      Qualify it, or use the reads cut 4.5-18.6% (`globalReads*`).
    - paper.tex:325 "26 points" (`rbAaCells`). paper.tex:297 "ten ... three" (`pnHostsLow`, `pnHostsQlow`).
    - paper.tex:318 "33,000-152,000" (32,601-152,470; correct). app_wsg.tex:87 "0.6-1.0" (O3-O6 run 0.58-1.05).
    - app_wsg.tex:169 "19 of the 29", :176 "+-0.2%", :178 "a fifth", app_traces.tex:48 "two to four tokens"
      (not verified).
    - app_wsg.tex:86-88: 20 of the 39 job 099 failures are the planning-time clause (2 on every host), 14 are other
      clauses on Pd/Pf, and 5 are elsewhere (aa/foa band on Pa, Ph, Pj; one ordering on Pi). "concentrate" is fair,
      but a count would help.

## Verified as correct (value and wording, unless listed above)

- **Job 099 panel (pn\*), from raw:**
  - Host counts: pnHosts 10, pnHostsQlow 3.
  - Exact-window time shares: pnShareOneLow 0.14-0.17 (0.141-0.173), pnShareFourLow 0.32-0.47, pnShareSixteenLow
    0.78-0.93, pnShareFourMid 0.21-0.24, pnShareSixteenMid 0.49-0.63, pnShareTwoQlow 0.35-0.45.
  - Recall-0.5 time shares: pnShareEightHalfLow 0.20-0.26, pnShareEightHalfQlow 0.37-0.39, pnShareAllHalfLow
    0.42-0.55 (min 0.4151 rounds up).
  - Reads and link: pnReadFourLowMin 0.52, pnLinkPctFour 97, pnLinkPctMin 50.
  - Time-read gap: pnGapFourLowMin/Max 0.05/0.20, pnGapRatioAtMax 0.29, pnGapFourCorr -0.91.
  - Planning and replay: pnPlanMaxLow 277 (Pg, allr5), pnReplayMaxLow 1.6 (Mid 1.7, Qlow 0.2).
  - MIN 1 read spread: pnFetchLowSd 0.154, pnFetchLowMean 1.22, pnFetchLowHw 0.009 (mine 0.0085), two-stage
    [1.12, 1.30].
  - Probe ranges: pnCpu 25-93, pnLink 19-49, pnHostRate 30-94.
- **hd\*:** hdHosts 14, hdRatioMin/Max 0.29/1.05, hdFetchMin/Max 0.87/1.44, hdFetchCorr 0.89, hdPacedMin/Max
  1.01/1.81 (O4 25%), hdAaMin/Max 0.42/1.00.
- **hm\*:** hmErrStepMed 2.2, Pninety 9, N 208; hmErrBgMed 18.0; hmFetchN 43, all signs right; hmWinShareErrMed/Max
  0.03/0.10, N 60; hmAllHalfOverMax 0.24 (Pj, fast link). The numbers match; the caveat is defect 2.
- **fs\* Shapley, recomputed from raw times and the bound (tab_shapley matches cell by cell):** fsBalBoth 26-60,
  fsBalInter 10-58, fsBalReadsOnline -4..4, fsBalSetTwo -13..26, fsBalSetOne 26-57, fsBalSet 9-35, fsBalReads 1-32,
  fsBalPacing -4..23, fsBalRest 30-62, fsBalHosts 10, fsSlowHosts 2, fsSlowReads -21..2, fsSlowRest 68-99. The
  abstract's "either choice alone at most 26%" is correct.
- **fx\* (O3-O5 raw):** fxFetchAllGain 16-51, fxPacedAllGain 25-81, fxBestFrac 46-73, fxUsualLow 0.94-1.15,
  fxFetchLow 1.16-1.44, fxNbReadsOverOpt 1.6-2.3, fxUsualReadsOverOpt 1.6-2.5, fxMinAdmitsOverOnline 2-11,
  fxReplicateMax 0.024, fxFetchSimDiffMax 3, fsbpReadsOpt 1.18-1.58 (also holds on O4-O6 and the panel).
- **dc\*/rb\*, and tab_lomo:** dcMed/Lo/Hi 0.65/0.61/0.72, dcCv 0.16, dcWMin/Max 0.6/34, dcBeta 0.48-0.78.
  tab_lomo is 1.22/1.57/2.42, 1.19/1.51/1.67, 1.16/1.42/1.61. rbPre 0.67, rbExp 1.31, rbR 0.977, rbExpLo/Hi
  1.10/1.43, rbExpLomo 1.25-1.36, rbExpNoInterp 1.35 (orig 1.51), rbAaBeats 23/26, rbAaSaveMax 6.7, rbWRise median
  9 / max 42, rbPolAaWins 18/27, Ties 5, LossMax 2.7, GainMax 5.8, rbPolBest 35-110. fsPre/fsExp 0.59/1.33.
- **vm\*/la\* and tab_value:** vmHalfMin/Max 2/10 (1.95, 3.75, 10.2, 4.5), vmShareTwoLow 0.31, vmShareFourLow/Mid
  0.52/0.25, vmEightHalfLow 0.29, vmAllHalfLow 0.43, vmFillDropMax 0.12 (wording: defect 6). All tab_value cells are
  correct, including the Linear column. laRecallK 84, laRecallEight 97, recomputed from job 063.
- **lf\*/gr\*:** precisions 0.48/0.26 (gpt-oss) and 0.54/0.32 (Qwen3); decayed signal 0.30/0.28; lfShare 3-15 /
  6-19; grPrec 0.40/0.26 and 0.46/0.31; grRecThree 0.53/0.54; grShare 2-11 / 6-17. k/(k+3) gives 0.57/0.73.
- **sca\*/scb\*:** scaN 565, scaAuto 469, scaAgree 432, scaDisagree 37, scaDet 36 (all reads or hit rates, so
  deterministic), scaOther 1 (096O4-P5a, 1.249), scaNoInterval 213, scaCrosses 7 (wording: defect 12). scbN 273,
  Held 190, Point 42, Failed 39, Untested 2 (both on Pd, the replay).
- **audOurs\* and the audit:** 18-42, median 27 (26.6). audAllN 52, audSources 13, audInclassN 20, Med 13.6, Q1 8.2
  (Q3: defect 13).
- **bothLimit\* and tab_headline:** bothLimitPct 25-43, bothLimitMeasPct 31-56, bothLeadCells 11/12, bothLlamaX
  2.0-4.0. Bounds recomputed for B: 172/430/519/96/214/308 (datasheet) and 172/279/279/96/192/192 (measured).
  limAllPct 33-57 (tab_limit). globalReads 4.5-18.6. gpuEffPct 52/61. fttBestOverDefaultMax 1.04, fttLeadCells 5.
- **Learned, batching and policy study:**
  - lreFewer 6-13, lreOverBase 0.90-1.05, lreOverFoaFour 1.01-1.03, lreOverFoaSix 0.90-0.94 (job 097 raw).
  - lrnFewerFetchHost 6-12, lrnFetchHost 17-30.
  - bkSameNine 8-12, bkSameEightLoss 18-20, bkFree 15-32, bkBelady -7..15, bkUnion 20-79.
  - polBestOnline 35-110, polDfBest 12, polSsfBest 10, polCells 27, polLowSpreadMax 4, polLruMore 2-11 on 8 of 9,
    polStatic 1.4-13.2, drift 22-58 / 33-75.
  - The 33,000-152,000 tokens per model is correct.
- **Law section (app_wsg.tex:93-197):** checked lawRows/Meas 33 = 5+5+5+5+8+1+4 over 7 hosts, lawCfg 29,
  lawLlamaOff +10..+119, the 46% llama.cpp figure (1/2.19), vramMs 3.92, the 24 us per 13 MB at 559 GB/s, and the
  running-example table 0,0,1,1,2 with its description. ftRatioGptMid 1.27 matches tab_headline. The per-row
  law-error macros (lawRowsMedian, loho\*) were not recomputed.
- **Not recomputed:** fxPacedLawTime (law_ms), ftPoolPct/llPoolPct, klMean\* (they agree with scorecard clause
  090-P1), lawMedian/lawDesk\*, lawFrozen\*, and the cputest099 equality claim.
