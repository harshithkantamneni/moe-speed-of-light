# Number check 15 (7 October)

## Scope

This check covers `paper/` and `scripts/` at 3d63e65 against 8aa97f5. In scope:
- the abstract, Table 1 (claims) and its caption;
- the Terms and Statistics paragraphs of Section 2;
- Sections 3, 4.1, 4.2, 4.3 and 5, and the Limitations;
- `app_more.tex`'s new paragraph "The probe's rate, and machines as the unit", with `tab_robust.tex`;
- `app_wsg.tex`'s new "Names and terms" section (`tab_names.tex`, and `tab_glossary.tex`, which this commit puts in the paper for the first time);
- the restructured "Prediction log", with `tab_prereg.tex` and the registration-timing sentence;
- the G_fit renaming in the calibrated-model appendix;
- every macro in `paper/wsg_robust.tex` (from `scripts/robustness.py`).

I recomputed everything from the raw files in `/home/claude/gpu-branch/results`, jobs 093–108. I read `robustness.py`, `tab_prereg.py`, `job106.py` (`boot_rounds`) and `reanalysis.py` (`profiled_G`) only for their definitions; none of my code imports them. The definitions I took from them:
- B_host is the highest of every `cpu_read_gbs`, `pcie_*_gbs` and `concurrent … sum` reading in `concur.txt`;
- the second-highest rate is the next value in that list; the CPU-only rate is the highest `cpu_read_gbs`;
- the implied rate is (M + A(1 − G/T))·S/(T − G), and its share is that rate over the rate in question;
- G is the profiled `G_prof_ms` where it exceeds 1 ms, and otherwise the `sum_law` median (4.2835 / 4.4979 ms, the median of the 069c and 105 profiles);
- a launch's speed ratio is the mean of base's per-round means over the mean of the configuration's (paired by problem);
- machines are keyed by GPU UUID, and a machine's value is the mean over its launches.

I set validity independently: base loss within 0.005 of the reference in every round (0.2068 for jobs 093–097, 0.190 after), and rounds at gpt-oss 11% within 2% (max/min − 1). This gives the same 42 valid launches as `prereg/job108_families.json`. The invalid ones are 106c (loss 0.208/0.420, rounds 11.5% apart), 106d (52.2%), 107c (3.5%), 107e (6.0%) and 108b (7.0%). The 42 valid launches ran on 28 GPUs: 25 consumer and 3 server.

Other sources:
- **Registered predictions:** the headers of `jobs/073_…` to `jobs/108_…` on the GPU branch.
- **Scorecards:** `prereg/scorecard_clauses.json` and `scorecard_099.json` to `scorecard_108.json`.
- **Timing:** `git log` on the GPU branch, every push event on `refs/heads/gpu` from the GitHub activity API (258 events, 26 September to 7 October), `gpu/vast_ledger.json` (`start` is written just after the rental `PUT` returns, per `gpu/vast.py`), and each launch's `manifest.json` `start_utc`.
- **Build:** `latexmk -pdf -g` on a `git archive` of 3d63e65.

My scripts are in `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/nc15/`:
- `common15.py` (the relation, the bound, file parsing) and `readings.py` (probe readings);
- `inv15.py`: every launch, with its rounds, loss, UUID, the three rates, own or fallback G, and the error and implied rate under each rate;
- `robust15.py`: every cell of `tab_robust` and the counts;
- `plans15.py`: fetchplan, fetch, bypassplan, dk and lrn against base, per round and pooled;
- `boot15.py`: the machine-level means and intervals, and the found-on counts;
- `share15.py`: the bound's share;
- `timing15.py`: pushes against rentals (`pushes.json` holds the push events).

Every cell of `tab_robust.tex` and `tab_prereg.tex` reproduces, and so does every `rb*` macro the text uses. The defects are about what some of the new numbers are said to show, one false sentence in Section 3, and labels and descriptions.

## Defects (most severe first)

### 1. Moderate: the new one-sided-error sentence holds at gpt-oss 11% only, and its 25 machines are not "the launches it was found on"

**Text.**
- **Section 4.2:** "The error leans the same way on the launches it was found on: over 25 consumer machines its mean at gpt-oss 11% is −2.2% (95% interval over machines −3.7 to −0.4), so Eq. (3) misses a small cost."
- **`app_more`:** "…Eq. (3)'s error at gpt-oss 11% is a mean −2.2% (−3.7 to −0.4%) over the 25 consumer machines: it under-predicts slightly but consistently, on 25 of the 30 launches it was found on as well."

**Evidence** (`boot15.py`; first round, as `robustness.py` computes it; the mean over rounds agrees to 0.01 points).

| Set | gpt-oss 11% | gpt-oss 25% |
|---|---|---|
| 25 consumer machines, mean (95% over machines) | −2.16% [−3.73, −0.41]; 21 of 25 negative | **+1.58% [+0.21, +3.08]; 8 of 25 negative** |
| The 21 machines it was found on (jobs 093–104) | −1.54% [−3.25, +0.38] | +2.22% [+0.77, +3.81] |
| The 30 launches it was found on | 25 under-predicted, median −1.36% | **7 under-predicted**, median +1.85% |

- **The 25 machines are not the found-on set.** Four of them (the Threadripper 9960X of 105b/106a, 107b, 108d and 108f) were first rented for the registered tests. On the 21 machines it was found on, the interval at 11% includes zero.
- **The sign depends on the budget.** At 25% the relation over-predicts on the same machines, with an interval above zero. "So Eq. (3) misses a small cost" is therefore true only at 11%, and "consistently" holds only within that budget.
- **The new-machine sentence just before it covers both budgets.** That sentence ("under-predicted every gpt-oss cell") includes 25%. So "leans the same way" is false at 25%.

**Fix.**
- **Section 4.2:** "At gpt-oss 11% the error leans the same way over the 25 consumer machines (mean −2.2%, 95% interval over machines −3.7 to −0.4; on the 21 machines it was found on, −1.5%, −3.3 to +0.4). At 25% it leans the other way (+1.6%, +0.2 to +3.1), so what Eq. (3) leaves out depends on the budget."
- **`app_more`:** "it under-predicts at 11% on 25 of the 30 launches it was found on, and over-predicts at 25% on 23 of them".
- **Script:** generate the 25% figures in `robustness.py` beside the 11% ones.

### 2. Minor: "The two hosts' intervals do not overlap" is false at one of the six configurations

**Text (Section 3).** "The two hosts' intervals do not overlap: the ratio is a property of the machine as much as of the systems."

**Evidence** (`tab_headline.tex`, Ours ÷ FreeToken).

| Configuration | Host B | Host S | Overlap |
|---|---|---|---|
| gpt-oss 11% | [1.278, 1.312] | [1.195, 1.219] | no |
| gpt-oss 25% | [1.254, 1.295] | [1.163, 1.224] | no |
| gpt-oss 40% | [1.134, 1.172] | [1.077, 1.111] | no |
| **Qwen3 12.5%** | **1.032 [1.022, 1.042]** | **1.027 [1.018, 1.036]** | **yes** |
| Qwen3 25% | [1.134, 1.173]† | [1.037, 1.059] | no |
| Qwen3 43.75% | [1.036, 1.064] | [0.962, 0.987] | no |

† The selection launch, which the table's own footnote says favours FreeToken.

**Fix.** "At five of the six configurations the two hosts' intervals do not overlap (at Qwen3 12.5% they do): …". Generate the count.

### 3. Minor: "more than 5% lower on 1 of 42 machines": the 42 are launches, on 28 machines

**Text (`app_more`).** "Its second-highest reading is a median 0.8% lower and more than 5% lower on 1 of 42 machines, the dual-socket server, by 35%."

**Evidence.**
- `rbLaunches` counts launch directories. The 42 valid launches ran on 28 GPUs: Pf ran six times, the 5950X (b5fd1878) and the 9950X O4 (3077be42) three times each, five other machines twice, and 20 once.
- The median gap (0.80%), the one launch over 5% (107d) and its gap (35.1%) all reproduce. On 107d the highest reading is `concurrent t=128 pcie=zerocopy64 … sum 204.5` and the second is `cpu_read_gbs t=16 151.4`.

**Fix.** "…on 1 of 42 launches (28 machines)". Also generate "the dual-socket server" from the data, since it is hand-typed.

### 4. Minor: the fewest-admission row and Section 5. On 2 of the 9 machines, "beats the deployed cache" was not a registered clause; the means are at gpt-oss 11% only; and "behind slow links" does not separate the losses

**Text.**
- **Table 1:** "MIN's fewest-admission schedule gains on every machine that ran stably | registered; held on 9 stable machines, lost on two of four unsteady ones".
- **Section 5:** "…beat the deployed cache on every machine that ran stably, a mean 1.22× over 9 machines (95% interval over machines 1.14–1.31). On two of the four that ran unsteadily, both behind slow links, it lost."

**Evidence** (`plans15.py`, `boot15.py`; the job headers).

The numbers reproduce.
- **The 9 stable machines** (fetchplan/base at gpt-oss 11%, merged by GPU UUID):

  | Machine | Launches | Value |
  |---|---|---|
  | Pf | 104a, 105e, 106b | 1.035, 1.028, 1.019 → 1.028 |
  | Core Ultra 9 285K (second) | 104b | 1.180 |
  | Ryzen 9 5950X | 104c | 1.338 |
  | EPYC 7402P | 105a | 1.149 |
  | Threadripper 9960X | 105b, 106a | 1.039, 1.056 → 1.047 |
  | O4 | 105f | 1.342 |
  | Pe | 106e | 1.339 |
  | Ryzen 9 5900XT | 107b | 1.419 |
  | Dual-socket server | 107d | 1.157 |

  Mean 1.222, 95% [1.134, 1.307] with my seed. Unmerged, the 12 launches average 1.175. Job 103's "fetchplan" runs (no plan loaded) and job 108 (which never ran fetchplan) are correctly left out.
- **The four unsteady machines.** All four failed the rounds check.

  | Launch | Machine | fetchplan/base (rounds) | Link/CPU |
  |---|---|---|---|
  | 106c | EPYC 7302 | **0.846** (0.834, 0.878, 0.824) | 0.14 |
  | 106d | Xeon 8347C | 1.326 (0.996, 1.489, 1.487) | 0.23 |
  | 107c | EPYC 9754 | 1.175 (1.167, 1.183) | 0.51 |
  | 107e | EPYC 7663 | **0.974** (0.948, 0.9995) | 0.21 |

  106c also computed wrong outputs.

But:
- **On 104b and 104c "beats the deployed cache" was not registered.** Job 104's header registers fetchplan/base > 1 only "on the host with ratio below 0.35" (104a, Pf). For 104b and 104c it registers fetchplan/base ≥ fetch/base + 0.03, a comparison with the greedy set. That held (1.180 against 1.053, and 1.338 against 1.256), so on these two machines "beats the deployed cache" follows only indirectly. The direct clause was registered on 7 of the 9 machines: in jobs 104 (Pf), 105 (P4), 106 (P7) and 107 (P5).
- **The budget is unstated.** 1.22 [1.14, 1.31] is at gpt-oss 11% only. Jobs 106–107 ran fetchplan only there. The admission margin's 1.020 [1.012, 1.031] is a different average (each machine's mean over 11% and 25%), and the text does not say so either.
- **"Both behind slow links" suggests a cause the data does not support.** 106d sits behind a link nearly as slow (0.23) and gained 1.33×, and the stable machines at 0.28–0.32 all gained.

**Fix.**
- **Table 1:** "registered on 7 machines, held on all 9 stable ones (two by implication); lost on two of four unsteady ones".
- **Section 5:** "a mean 1.22× at gpt-oss 11%". Drop "both behind slow links", or add "as was a third, which gained".
- **`app_more`:** say that the margin's per-machine value averages both gpt-oss budgets.

### 5. Minor: the registration-timing parenthetical. For four rentals GitHub records the push *after* the rental was created, not in the same second

**Text (`app_wsg`, Prediction log).** "…committed before the machine started; each machine cloned the published branch at boot and ran the header as committed, so git history orders every prediction before its measurement (for a few rentals the push and the start fall in the same second)."

**Evidence** (`timing15.py`; 71 rentals of jobs 093–108).
- **Commits.** Every header commit, and every per-host script, was committed at least 2.7 s before its rental (the closest is 105f).
- **Pushes within a second of the rental.** Seven rentals have their push within a second of the rental's ledger `start`:

  | Rental | Push (UTC) | Rental `start` | Push − start |
  |---|---|---|---|
  | 107f | 06:37:00 | 06:37:00.444 | −0.44 s |
  | 107g | 06:41:20 | 06:41:20.097 | −0.10 s |
  | 108c | 07:39:37 | 07:39:37.848 | −0.85 s |
  | **108a** | **07:39:37** | **07:39:36.165** | **+0.83 s** |
  | **108b** | **07:39:37** | **07:39:36.976** | **+0.02 s** |
  | **108d's per-host script** (779dfab) | **07:46:01** | **07:46:00.952** | **+0.05 s** |
  | **108f's per-host script** (a6f7a05) | **08:18:31** | **08:18:30.515** | **+0.48 s** |

  GitHub's push timestamps have one-second resolution.
- **What "start" means.** `vast.py` writes `start` only after the rental `PUT` has returned. So for 108a, 88e954c reached GitHub at least 0.8 s after the machine was rented, in the next second, not the same one.
- **The ordering the paper relies on still holds.** The machine clones at boot (`boot_cmd`), and every launch's job started 26 s or more after its push. The earliest is 100f; 108a started at 07:40:37, 60 s after the push.

**Fix.** "…(for seven rentals the push and the rental fall within a second of each other, and for four GitHub records the push up to 0.8 s after the rental was created; every machine cloned the branch and started its job at least 26 s after the push)."

### 6. Minor: `tab_prereg`'s "what it tested" column. Several descriptions do not match the job headers; jobs 105–107 are described by one of their questions; one row is out of order

All counts are right (see What reproduces). The issues are in the hand-typed `WHAT` dictionary of `tab_prereg.py`.

| Job | Table says | Header says | Suggested |
|---|---|---|---|
| 081 | "the calibrated model, blind" | the headline table again on host B, with the model's per-host fetch table; P2–P4 are ours against FreeToken and llama.cpp | "system table on host B; the model's fetch table" |
| 082 | "rerun for the first review" | new experiments the review asked for: steady state on held-out prompts, the ablation ladder, parity, the missing cells | "the first review's experiments" |
| 083 | "gpt-oss, harder and longer prompts" | harder problems, 2,048-token *outputs* | "gpt-oss, hard problems, long outputs" |
| 084, 084b | "traces in VRAM" | all-in-VRAM speeds *and* routing traces | "all-in-VRAM speed; routing traces" |
| 085, 085b | "half-width PCIe link" | a PCIe 4.0-class link (half the rate, not half the lanes) | "half-rate link" |
| 090 | "output parity on an A100" | the scorecard's host is an RTX PRO 6000 + Xeon 6960P; the A100 was stopped at the VRAM gate | "output parity against an all-VRAM reference" |
| 105 | "Eq. 3, plain form; layer-ahead copy" | also MIN's fewest-admission set (P4–P5, 11 of 31 clauses) | add "; fewest-admission set" |
| 106 | "Eq. 3 with overlap; admitting less" | also the slow-link fetch and fetchplan tests (P6–P7, 38 of 181 clauses, 8 failed) | add "; MIN's sets on slow links" |
| 107 | "Eq. 3 on new machines" | also dk (P3–P4, 21 clauses) and fetchplan/bypassplan (P5–P6, 6) | add "; admitting less; fewest-admission set" |

Two further points:
- **Row order.** The rows run 096, **098, 097**, 099, because `scorecard_clauses.json` lists 098 before 097O4.
- **The caption's "Held".** The caption defines it as "held with a 95% interval on the predicted side". By the scorecard's own rule, 14 of the 218 clauses held in jobs 073–098 are deterministic `equality` clauses that held outright, with no interval. Add "(or outright, for deterministic clauses)".

### 7. Minor: the G_fit renaming stops halfway. Eight bare `$G$` still denote the fitted constant

**Text.** Section "The calibrated time model" now says G_fit "is neither the profiled G of Eq. (3) nor T_GPU". The same section and its subsections then use `$G$` for the fitted constant:
- l. 201: "a budget-dependent bias whose $G$ was borrowed from llama.cpp";
- l. 210: "for Qwen3, $G{=}4.3$ ms, the mean residual of three llama.cpp runs";
- l. 216 and l. 261: "the additive form $G+X_c/B_c+X_p/B_p$";
- l. 226: "a fitted rate factor is collinear with $G$";
- l. 260: "the token-level max of Eq. (law) with $G$ refit";
- l. 274: "with $G$ refit";
- l. 282: "with $G$ refit under leave-one-host-out validation".

**Fix.** Replace each with `$G_{\mathrm{fit}}$`.

### 8. Minor: "Names and terms" leaves out codes the supplement uses, and the glossary points to a Setting that no longer names the hosts

**Missing codes.** The new section says `tab_names` "maps the configuration codes used in the appendix tables, the supplement and the scripts". Codes those tables use but `tab_names` does not map:
- `w1`, `w2`, `w8r5` and `allr5` (`tab_scorecard_099.tex`);
- `b8r5` (`tab_scorecard_100.tex`);
- `learned` (job 097's foa with the learned order, in `tab_scorecard.tex`; a different configuration from `lrn`).

Their definitions are in the 099 and 097 headers.

**The glossary.** `tab_glossary` enters the paper for the first time in this commit.
- **A stale cross-reference.** It defines "hosts B, S, A, O1–O5 | rented RTX 5090 machines (Section 2)", but Section 2 no longer names any host. Host B and host S are named only in Table 3's rows.
- **Undefined names.** The appendix also uses the panel names Pf, Pe and Pg (`app_more` l. 24; `app_wsg` l. 126–151), and neither table defines them.

**Fix.** Add the six codes to `tab_names`. In the glossary, point "hosts B, S" to Table 3 and add "Pa–Pj: the panel of job 099".

### 9. Minor: the Statistics paragraph overstates the machine-level intervals and describes a different resampling

**Text.** "These intervals are within one machine; the main claims also carry intervals over machines (Appendix …)." Earlier in the same paragraph: "pooled intervals resample machines and then problems".

**Evidence.**
- **Only three claims have them.** Intervals over machines exist for three claims: the fewest-admission gain, the admission margin and Eq. (3)'s error at 11%. The bound's share (the abstract's 31–54%), foresight on three hosts, and the horizon have none.
- **They resample machines only.** The new intervals resample machines, one point estimate each, and never resample problems.
- **Five machines are too few for a percentile bootstrap.** For the admission margin the 95% percentile interval [1.012, 1.031] is narrower than a t-interval on the same five values, [1.004, 1.035]. For the fewest-admission gain (9 machines) the t-interval is [1.113, 1.331].

**Fix.** "…three of the main claims also carry intervals over machines (a bootstrap of per-machine means; Appendix …)". Either report t-intervals for n ≤ 9, or say that the percentile interval is narrow at this n.

### 10. Minor: what "the median of the profiles" means, and when it is used

**Text (Section 4.1, new).** "G … in an Nsight profile of the same engine (4.1–4.7 ms for gpt-oss; where a machine was not profiled, the median of the profiles)."

**Evidence.**
- **108a and 108b were profiled.** Their profiles captured no decode kernels (G = 0.0015 and 0.017 ms), and they fall back to the median too.
- **There are two different medians.**
  - `robustness.py` and Fig. 2 use the `sum_law` median of the 069c and 105 profiles, 4.2835 / 4.4979 ms.
  - `tab_job108` and Section 4.2 use "the median of earlier profiles", 4.29 / 4.47 (`jmFallbackGLow/Mid`).
  
  The effect on 108a's error is under 0.1 points, but these are two definitions of one term.
- **A generated contrast goes unreported.** `wsg_robust.tex` computes a median |error| of 3.2% on the 9 consumer launches with their own G against 1.7% on the 30 that use the median (`rbProfErrMed`, `rbMedGErrMed`). The new sentence "the others use the median profile" does not mention it. The contrast is confounded with which machines were new, but a reader weighing the median-G launches should see it.

**Fix.**
- **Section 4.1:** "…where a machine was not profiled or its profile captured no decode kernels, the median of the profiles of jobs 069c and 105".
- **`tab_job108`:** use that same median.
- **`app_more`:** add "(median |error| 3.2% on the launches with their own G, 1.7% on the others)".

### 11. Minor: "On the 5 machines that ran stably", carried over from Number check 14 (defect 13)

**Text (Section 5).** "On the 5 machines that ran stably it runs a mean 1.020×…"

**Evidence.** 28 machines ran stably; 5 of them ran dk.

**Fix.** "On the 5 machines that ran it stably".

### 12. Minor: hand-typed values in the changed text

| Where | Hand-typed | Source |
|---|---|---|
| Section 3 | "The two hosts' intervals do not overlap" | a count from `tab_headline` (defect 2) |
| `app_more` | "the dual-socket server" (the launch with the largest gap) | the data |
| Section 5 | "both behind slow links" | link ratios (defect 4) |
| `app_wsg` | "for a few rentals" | the ledger and the push log (defect 5) |
| `tab_prereg.py` | the `WHAT` descriptions | job headers (defect 6) |
| Table 1 | "(3 hosts)", "(9 models)" | counts (carried over from Number check 14) |

### 13. Build note: the main text now ends on page 10

- **Page count.** `paper.tex` builds to 40 pages, with no undefined references, citations or macros, and no `[pend.]`.
- **Where the main text ends.** The Conclusion starts on page 9 and its last three lines (to line 497) run onto page 10, where the References begin. In Number check 14 (df85610) the main text ended on page 9. Page 10 is still within a 10-page main text.
- **No main-text overfull boxes.** The only overfull box is `tab_audit.tex` in the appendix (70.4 pt, lines 4–68), as before.
- **Table placement.** `tab_robust` (Table 26) floats to page 35, three pages after the paragraph that cites it (page 32).
- **Warnings.** There are 39 hyperref "duplicate destination" warnings, against 35 at df85610.

## What reproduces

**`tab_robust.tex`, every cell** (`robust15.py`; 42 valid launches × 2 budgets = 84 launch-budgets).

| Rate | Faster than the rate by >5% (max) | Within 6% (consumer, 11%) | Within 8% | Median \|error\| |
|---|---|---|---|---|
| Highest reading | 12 (1.223×) | 31/39 | 37/39 | 1.69% |
| Second-highest | 15 (1.257×) | 34/39 | 37/39 | 1.48% |
| Best CPU-only | 46 (1.533×) | 22/39 | 29/39 | 5.55% |

- **The 12 launch-budgets above the highest reading.** Eleven are at 25%: 095, 096a, 097a, 097b (1.0502), 099a, 099b, 100f, 103b, 104b, 105a and 105f. The twelfth is 100f at 11% (1.164).
- **The same counts with the mean over rounds.** Identical.

**The `rb*` macros the text uses** (`boot15.py`).
- **The probe's readings.** The second-highest reading's gap has a median of 0.80% and a maximum of 35.1% (107d), and is over 5% on that launch only.
- **Machine-level means.**
  - Fewest-admission (9 machines; defect 4): 1.2221 [1.134, 1.307], against the paper's 1.22 [1.14, 1.31].
  - Admission margin (5 machines, each the mean of its 11% and 25% cells): Pf 1.024/1.057, TR 9960X 1.003/1.019, Pe 1.010/1.008, 107b 1.021/1.019, 107d 1.049/0.987. Result 1.0197 [1.0115, 1.0305], against the paper's 1.020 [1.012, 1.031].
  - Eq. (3)'s error at 11% (25 consumer machines): −2.16% [−3.73, −0.41], against the paper's −2.2 [−3.7, −0.4].
- **Merging by UUID.** Duplicates by GPU UUID are merged in all three. In the error mean, for example: Pf 6 launches; 5950X b5fd1878 and 9950X 3077be42, 3 each; five machines 2 each.
- **Profiled G.** G was profiled on 9 of the 39 consumer launches: 105b, e, f; 106a, b, e; 107b; 108d, f.
- **The found-on launches.** 25 of the 30 found-on launches are under-predicted at 11%.

**The unsteady machines** (defect 4).
- **Who ran the fewest-admission schedule.** 106c, 106d, 107c and 107e, with ratios 0.846, 1.326, 1.175 and 0.974.
- **Who lost.** 106c (link/CPU 0.14) and 107e (0.21), so "two of four" and "both behind slow links" are literally right.

**The bound's share** (abstract and Section 3; one launch per consumer machine, the first by job order).
- **25 consumer machines:** 31.2–54.0% at 11%. The minimum is 105b (Threadripper 9960X) and the maximum 100f.
- **Other scopes.**
  - The 21 machines of jobs 093–104: 38.4–54.0%.
  - All 39 consumer launches: 30.6–54.0%.
  - The servers: 105a 46.1%, 107d 12.4%, 108a 26.7%.

**Eq. (3) and its forms** (Section 4.1).
- **The closed form.** Multiplying T = G + t_M + t_A(1 − G/T) by T gives T² − (G + t_M + t_A)T + t_A·G = 0; T is the larger root. Correct.
- **Over the 34 launches of jobs 093–105.** The overlap form puts 29 within 6% at 11% and 33 at 25%; the plain form 32 and 18. This matches the text and job 106's header.
- **Over the 30 found-on launches.** The plain form puts 28 within 6% at 11% and 17 at 25%.

**`tab_prereg.tex`, every row against the scorecards.**
- **Jobs 073–095.** Every row matches `scorecard_clauses.json`.
- **The merged rows.** 096 = O4 (13/48/5/0) + O5 (16/48/2/0) + the pooled clauses (0/1/5/0) = 29/97/12/0. 097 = O4 (10/36/5) + O6 (1/17/15) = 11/53/20/0.
- **Jobs 099–108.** Every row matches its machine scorecard.
- **The totals.** 495/597/308/83 reproduce, and 1,483 clauses in all.
- **084c.** It has no clauses and is correctly absent.

**`tab_names` against the job headers.** These codes match their definitions:
- `foa` (096), `aa` (099), `dk` and `lrn` (106), `pf` (105);
- `fetch` and `bypass` (094–096), `fetchplan` and `bypassplan` (103–104);
- `lead2` and `both2` (095), `both3p` (095–097);
- `hitopt` and `hitoptp` (095–096), `nb2` (096);
- `w4`, `w16`, `b4` and `b16` (099–100).

The "one in flight" pacing in the glossary agrees with the engine description in `app_wsg` (one 16 MB piece at a time, up to 64 queued).

**Registration timing.** Every prediction was committed before its rental. Every launch that produced data started its job at least 26 s after the push. The amended headers of 105, 106 and 107 changed hosts, not predictions.

**Unchanged numbers in changed sentences.** These are still right:
- 11 of 12 configurations;
- FreeToken trails at 5 of 6 tuned cells;
- the new consumer machines' 0.9–7.1%;
- 37 of 39 consumer launches within 8%;
- dk gaining up to 6% and losing 1% at one of 10 cells;
- the slow-link ratios 0.28–0.32, with greedy 0.85–0.93 and CPU-loaded 1.11–1.22.
