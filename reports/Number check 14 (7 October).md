# Number check 14 (7 October)

## Scope

This check covers `paper/` at df85610 against e4fdc32. The range includes c6bca98, which scored job 108. In scope:
- the main text from the abstract to the Conclusion, including the new claims table (Table 1) and the new step schematic (Fig. 1);
- `tab_regtests.tex` (Table 4 in the build);
- three paragraphs of `app_more.tex`: "Job 108, every host", "Admitting less, in detail" and "The slowest link", together with `tab_job108.tex`;
- the job 108 paragraph of `app_wsg.tex`;
- `figs/decomp.png`, `figs/hostdep.png` and `figs/step.png`, which I looked at;
- every macro in `wsg_job108.tex` and `wsg_decomp.tex` that the text uses.

I recomputed everything from the raw files in `/home/claude/gpu-branch/results`, jobs 093–108. I read the authors' scripts only for their definitions; none of my code imports them. The definitions I took from them:
- B_host is the highest of every `cpu_read_gbs`, `pcie_*_gbs` and `concurrent … sum` reading in `concur.txt`;
- B_c is interpolated at H = cores − 2 helpers;
- B_p is the highest 16 MB zero-copy reading;
- Eq. (3) is solved in closed form;
- the implied rate is (M + A(1 − G/T))·S/(T − G);
- MIN's reads are R* = 38.315 per token at 11% and 15.340 at 25% (`prereg/speed_limit_v2.json`).

What I checked, and against what:
- **Registered predictions.** The headers of `jobs/105_sumlaw@vast.sh`, `106_onlineadmit@vast.sh`, `107_newhosts@vast.sh` (at 346bd34) and `108_smallhosts@vast.sh` (88e954c, never amended), plus `prereg/scorecard_*.json` and `scorecard_clauses.json`.
- **Commit and rental times.** `git log` on the GPU branch against `gpu/vast_ledger.json`.
- **"New" machines.** GPU UUIDs from every earlier launch directory: `nvidia-smi-q.txt`, or `v0.txt` for launches stopped at the gate.
- **Build.** I built `paper.tex` and `supplement.tex` with latexmk in a scratch copy. To compare warnings, I also built e4fdc32.

My scripts are in `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/nc14/`:
- `common14.py`: rates, the relation, the bound and file parsing;
- `inventory.py`: every launch of jobs 093–108 at both gpt-oss budgets, with own or fallback G, round spread, error, implied rate and whether its UUID is new;
- `regtests.py`: every cell of Table 4, including the plain and half-admission forms;
- `decomp.py`: Fig. 2 and `wsg_decomp.tex`;
- `plans.py`: fetchplan, fetch, bypassplan, pf, dk and lrn against base, per round and pooled;
- `extras.py`: the processor split, the dk and lrn read cuts, the fallback-G sensitivity, the elasticity, the bound's share and the ledger lookup.

Every cell of `tab_regtests.tex` and `tab_job108.tex` reproduces, and so does every value in `wsg_decomp.tex` and every `jm*` macro the text uses. The defects are about what the numbers are said to show, about how the registered verdicts and evidence are labelled, and about scope that the rewrite dropped.

## Defects (most severe first)

### 1. Major: "server processors read well below their probed rate", but one of the three valid server machines read at 1.04 of it and fits the relation within 3%

**Text.**
- **Abstract:** "…and by up to 59% on server processors, which read well below their probed rate."
- **Introduction:** "…and on server processors the engine reads well below the probed rate."
- **Table 1:** "Server processors read well below the probed rate | after the fact".
- **Section 4.3:** "What separates the machines the relation fits from those it misses is the processor…". The same paragraph then says "the three valid ones read at AMD engineering sample 0.27, EPYC 7543 0.58, EPYC 7402P 1.04 of it."
- **Section 4.4:** "On the server processors that read below their probed rate, that shortfall is a third part (EPYC 7402P 0%; EPYC 7543 34%; AMD ES, 2 sockets 59% of the time at 11%)."
- **Limitations:** "The split by processor that separates its small misses from its large ones…"
- **Conclusion:** "on server processors it reads below the machine's rate".

**Evidence** (`inventory.py`, `extras.py`).
- **The EPYC 7402P (105a)** has one socket, one NUMA node and 24 usable cores.
  - Its B_host of 60.4 GB/s comes from `concurrent t=8 pcie=copyengine … sum 60.4`; its best CPU-only reading is 57.0.
  - Implied rate: 1.038 at gpt-oss 11% and 1.061 at 25%.
  - The relation's error is +3.0% and +4.0%, within 6% at both budgets.
  - Loss 0.1905, so its outputs are correct.
  - It is one of the three valid server launches in the paper's own split (`jmServList`), and Fig. 2 shows it below the dashed line with no hatching and its bars past the dot.
- **So the claim holds for 2 of 3 valid server machines:** 107d at 0.27 (−58.9%) and 108a at 0.58 (−34.2%).
- **Server processors overall:** 1 of 3 valid launches is within 8% (`jmServWithinEight` = 1).
- **Consumer processors:** 8 of 39 launches miss by more than 6%: 099d −7.5, 099i −10.6, 100a −6.2, 100f +12.6, 103e −7.0, 107b −7.3, 108d −6.9 and 108f −6.5. A server launch at +3.0% fits better than all eight. The split therefore does not "separate its small misses from its large ones", and the processor is not "what separates the machines the relation fits from those it misses".
- **The Section 4.4 sentence contradicts itself.** It lists "EPYC 7402P 0%" among "the server processors that read below their probed rate".

**Fix.**
- **Table 1:** "Two of three valid server-processor machines read at 0.27–0.58 of the probed rate; the third, a 24-core EPYC 7402P, at 1.04 | after the fact".
- **Abstract:** "…and by up to 59% on two of the three server-processor machines that passed our checks, which read at 0.27–0.58 of their probed rate".
- **Introduction:** "on two of three server machines the engine reads well below the probed rate".
- **Section 4.3:** "The relation is within 8% on 37 of 39 consumer launches. Of the three valid server launches it fits one (an EPYC 7402P, +3%) and misses two, by 34% and 59%. We have not established why."
- **Section 4.4:** "On the two server machines that read below their probed rate, that shortfall is a third part (34% and 59% of the time at 11%); the EPYC 7402P has none." Generate this from a list that leaves out the 0% entries.
- **Limitations:** "separates most of its large misses from its small ones".
- **Conclusion:** "on two of three server machines it read well below the machine's rate".

### 2. Moderate: job 107 is "failed" in Table 4, the abstract, Table 1 and the contributions, but its registration makes it inconclusive, as `app_wsg` itself says

**Text.**
- **Table 4:** job 107, "As registered: failed".
- **Abstract:** "every registered test that included machines new to it failed".
- **Table 1:** "exploratory; failed 4 registered tests".
- **Contributions:** "including four failed tests".
- **`app_wsg`, by contrast:** "the test had two valid machines of the three it required, which by its registered rule makes it inconclusive; its per-cell clauses failed on both."

**Evidence.**
- **The registered rule.** The header of `107_newhosts@vast.sh` (1a0616a, unchanged by f9ac8c7, ddec94f and 346bd34) says "The test needs 3 valid hosts; with fewer it is inconclusive." Two hosts were valid.
- **The script does not apply the rule.** `tab_regtests.py`'s `verdict()` checks only the band and the median, never the required count.
- **The per-cell clause cannot be rescued.** It failed at 7 of 8 cells, and no third host could undo that. So "none passed as registered" (Section 4.2, Introduction) is true, but "failed" for job 107 is not what was registered.
- **The consumer-processor claim's own record.** On consumer processors, the relation held as registered in job 106: 12 of 12 cells within 6%, median 2.3%, and the registered clause at 25% also held (overlap median 1.6% against half the plain form's, 3.0%). "Failed 4 registered tests" therefore also overstates the record of the claim as Table 1 words it.

**Fix.**
- **Table 4, job 107:** "inconclusive (2 of 3 valid); band missed at 7/8".
- **Abstract:** "no registered test that included machines new to it passed".
- **Table 1:** "exploratory; registered 4 times: 3 failed, 1 inconclusive (held on the consumer machines of one)".
- **Contributions:** "three failed tests and one inconclusive".
- Add the required-count rule to `verdict()`.

### 3. Moderate: Table 1 labels the fewest-admission claim "registered; held", but as registered in job 106 it failed on two hosts

**Text.**
- **Table 1:** "MIN's fewest-admission schedule gains on every valid machine | registered; held".
- **Section 5:** "Which of MIN's schedules is followed matters (registered)… beat the deployed cache on every machine that passed our checks".

**Evidence** (`plans.py`; `prereg/scorecard_106.json`).
- **What job 106 registered.** P6 required fetchplan/base > 1 "in every round" with the launch interval above 1 on slow-link hosts. P7 required fetchplan/base > 1 "on every host". Job 106 registered no validity gates; 106c's wrong outputs and 106d's round spread were judged after the runs.
- **The scorecard.** It records six failed clauses:
  - 106c's P6 in rounds 1–3 (0.834, 0.878, 0.824);
  - 106c's P6 launch interval and P7 (0.846 [0.820, 0.877]);
  - 106d's P6 in round 1 (0.996).
- **Every machine that passed checks did gain.** I recomputed all of them:
  - jobs 104–106: Pf 1.035, 1.028 and 1.019 on its three launches; 285K (second) 1.180; 5950X 1.338; EPYC 7402P 1.149; TR 9960X 1.039 and 1.056; 9950X (O4) 1.342; 9950X (Pe) 1.339. That is the 7 machines of `jkMachines`.
  - job 107: 107b 1.419 and 107d 1.157.
  - The job 103 "fetchplan" runs equal fetch to the third decimal (no plan loaded), and the scorecard marks them untested.
- **The other set-aside host is excluded legitimately.** 107e lost (0.974) but was excluded by the registered V2 gate.

**Fix.** Table 1: "registered; held on all 9 machines that passed checks (as registered in job 106, failed on the two hosts set aside afterwards)". Say the same in one clause in Section 5.

### 4. Moderate: the abstract's "reaches 38–54% of that bound" is the range for the 21 machines of jobs 093–104; Section 4.4 gives 31–54% for the same quantity on the 25 consumer machines

**Text.**
- **Abstract:** "Our llama.cpp expert cache… reaches 38–54% of that bound at the smallest gpt-oss budget."
- **Section 3:** "Across 21 machines at gpt-oss 11%, our cache stands at 38–54% of Eq. (1)." The old text said "On the 21 machines of jobs 093–104"; the rewrite dropped that scope.
- **Section 4.4:** "MIN's reads, which are the bound, 31–54%".

**Evidence** (`extras.py`). At gpt-oss 11%, the bound is MIN's read time. Its share of the measured time over Fig. 2's rows:

| Machines | Bound / measured time |
|---|---|
| The 21 machines of jobs 093–104 | 38.4–54.0% (reproduces `slMaxEff`) |
| The 25 consumer machines | 31.2–54.0% (Threadripper 9960X 31.2%) |
| All 28 stable machines | 12.4–54.0% (EPYC 7543 26.7%; dual-socket server 12.4%) |

The elasticity sentence in Section 4.4 also lost its "machines of jobs 093–104" scope. On one launch per machine I get −0.97 [−1.08, −0.80] on those 21 and −0.98 [−1.05, −0.90] on the 25 consumer machines, so only the wording needs to change.

**Fix.**
- **Abstract:** "reaches 31–54% of that bound at the smallest gpt-oss budget on machines with consumer processors", or give the full 12–54% with the servers.
- **Section 3:** restore "of jobs 093–104", or switch to the Fig. 2 set.
- **Section 4.4:** say which machines the elasticity is over.

### 5. Moderate: "the fastest of three systems we compare" and "FreeToken still trails" omit the configurations where FreeToken is faster

**Text.**
- **Abstract:** "Our llama.cpp expert cache, the fastest of three systems we compare".
- **Table 1:** "Our cache leads FreeToken and llama.cpp at equal GPU memory | measured (3 hosts)".
- **Section 3:** "tuned on a third host, FreeToken still trails". The old text said "at 5 of six cells".

**Evidence.**
- **Host S.** `tab_headline`, Qwen3 43.75%: FreeToken 98.3 against ours 95.8 tok/s, ratio 0.974 [0.962, 0.987]. FreeToken is faster with its interval below 1, at 1 of 12 configurations (`bothLeadCells` = 11).
- **The tuned host.** `scorecard_clauses` 098-P2 at Qwen3 43.75%: 0.983.
- **The label understates the evidence.** The comparison was registered and held at most cells: 081-P2 [1.294, 1.275, 1.154], 081-P3b, 091-P1a–d, 098-P2.

**Fix.**
- **Abstract:** "the fastest of three systems we compare at 11 of 12 configurations".
- **Section 3:** restore "at 5 of six cells".
- **Table 1:** "registered; held at 11 of 12 configurations".

### 6. Minor: Table 1's other evidence labels

- **Undefined labels.** "derivation" and "measured" are not defined in the caption, which defines only registered, exploratory and after the fact.
- **Row 2 has no verdict.** "That bound's read time is nearly reachable | registered (3 hosts)" says registered but not held or failed. In `scorecard_102` the thresholds (token mode ≥ 0.80, layer mode ≥ 0.50) held at all 6 cells. "At least 0.03 below the analytic fraction" failed at 4 of 6, and "lower at 32 than at 14" failed on 1 of 3 hosts; both failures are in the favourable direction.
- **Fix.** "registered; held (thresholds met at 6/6 cells)", and define every label in the caption.

### 7. Minor: "the three consumer machines new to these tests" is narrower than Table 4's definition of new

**Text.**
- **Section 4.2:** "On the three consumer machines new to these tests the relation under-predicted every gpt-oss cell, by 0.9–7.1%."
- **Limitations:** "on new consumer machines it under-predicts slightly".

**Evidence** (`regtests.py`).
- **The numbers reproduce:** 107b −7.15/−2.28, 108d −6.94/−4.05, 108f −6.46/−0.95.
- **But Table 4 counts more new machines.** Its "New" column includes 105b, a Threadripper 9960X that was new to job 105's test. On 105b the overlap form is −0.9% at 11% and +1.0% at 25%, and the registered plain form +2.0% and +7.0%. So not every cell on a new consumer machine was under-predicted.

**Fix.** "On the three consumer machines of the last two tests…", and the same in the Limitations.

### 8. Minor: "narrowly on consumer processors"

**Text (abstract).** "…failed, narrowly on consumer processors…".

**Evidence.**
- On the Ryzen 9 5900XT (107b) the relation missed by 11.3% at Qwen3 12.5% and by 7.1% at gpt-oss 11%, against a 6% band.
- In job 105 the registered plain form missed the new Threadripper 9960X by 7.0%.

**Fix.** "by up to 11% on consumer processors".

### 9. Minor: the Xeon 8347C's round spread is 52% in the main text and 39% in the appendix

**Text.**
- **Section 4.2:** "the other ran its rounds 52% apart" (`jiUnstableSpreadMax`).
- **`app_more`, "Job 106, every host":** "varied by 39% between rounds at gpt-oss 11%" (`jiUnstableVarMax`).
- **Code:** `job107.py`'s `INVALID` labels 106d "rounds 39% apart".

**Evidence.**
- 106d's base rounds at gpt-oss 11% were 14.521, 22.095 and 21.930 ms.
- max/min − 1 = 52.2%. That is the definition registered as V2 in jobs 107 and 108, and the one the main text uses.
- 39% is (max − min)/mean.
- `tab_job106_all`'s caption also says "between launches" for what are rounds.

**Fix.**
- Use `\jiUnstableSpreadMax` in `app_more` and update the `INVALID` label.
- In the caption, write "between rounds".

### 10. Minor: three of job 108's eight registered offers were already in the rental ledger; the appendix names one

**Text (`app_more`).** "The registered offer list said none of its offers was in the rental ledger; one, the 9950X3D's, had been rented for a few minutes in job 099 without producing data…"

**Evidence** (`vast_ledger.json`, `extras.py`). Three listed offers appear before job 108:

| Offer | CPU | Earlier rental |
|---|---|---|
| 54156078 | Ryzen 9 9950X3D | 099a, 247 s |
| 53424353 | Ryzen 7 5700X3D | 100b, 4,764 s; this is the 108c host the gate stopped as "GPU rented before" |
| 51325952 | Core i9-13900KF | 099d, 806 s, and 100e, 2,256 s; not reached in job 108 |

The UUID gate made this harmless.

**Fix.** "…said none of its offers was in the rental ledger; three were (the 5700X3D's, from job 100, which the gate stopped; the 9950X3D's, rented for a few minutes in job 099 without data; and an i9-13900KF's that was not reached)."

### 11. Minor: the slow-link sentence quotes one budget for the greedy set and two for the CPU-loaded set

**Text (Section 5).** "…the greedy schedule lost (0.85–0.93×) while the fewest-admission one gained in every round. There, its copies belong in the background: loaded by the CPU, the same set gains 1.11–1.22×."

**Evidence** (`plans.py`; launches 104a, 105b, 105e, 106a and 106b).

| Budget | Greedy, in the step | Fewest-admission, loaded by the CPU |
|---|---|---|
| 11% | 0.853–0.933 | 1.112–1.114 |
| 25% | 0.964–0.983 | 1.159–1.217 |

**Fix.** Either "at 11%: lost 0.85–0.93×; loaded by the CPU, gains 1.11×", or quote both budgets for both.

### 12. Minor: the reading-below-the-rate part is described as a server effect, but it appears on most consumer rows of Fig. 2

This is carried over from Number check 13, defect 8.
- **What Fig. 2 shows.** At gpt-oss 11%, 21 of the 25 consumer rows have a hatched segment: 6 of them at 3% or more of the time, and the i7-14700K at 10.6%. `\dcDeskShortLowMax` (11) is generated but never used.
- **The caption's G description.** It says "or the median profile where none was taken". The EPYC 7543 took a profile that captured 3 kernels.
- **Fix.**
  - Add "(up to 11% of the time on consumer machines)" to the Section 4.4 sentence.
  - Caption: "or the median profile where none was taken or it captured no decode kernels".

### 13. Minor: job numbers, nicknames and loose wording left in the main text

- **Job numbers.** Table 4 has a "Job" column (105–108), and its caption mentions "job 108", "Jobs 107 and 108" and "job 106". Use "Test 1–4" instead. The "Host B" and "Host S" labels in Table 3 predate this rewrite.
- **"Little host memory in use by other tenants"** (Section 4.2), carried over from Number check 13, defect 10. V0 checked at most 48 GB in use; whose memory it was is not measured.
- **"On the 5 machines that passed our checks it runs…"** (Section 5). The three valid machines of job 108 also passed the checks but did not run dk. Say "On the 5 valid machines that ran it".
- **"On server processors, 5 of 8 launches failed a validity check"** (Section 4.3). Job 106 registered no validity check; 106c and 106d were set aside afterwards. Say "failed a validity check (two of them judged after the run)".
- **Definition gap.** The Setting defines server processors as "EPYC and Xeon parts", but the dual-socket "AMD engineering sample" is also counted as a server. Add "and AMD engineering samples".

### 14. Minor: hand-typed counts in the changed text

| Where | Hand-typed | Source |
|---|---|---|
| Table 1 | "(3 hosts)" twice, "4 registered tests", "(9 models)" | counts |
| Introduction | "four later experiments" | number of rows of Table 4 |
| Contributions | "four failed tests" | see defect 2 |
| Section 3 | "on two hosts", "a third host" | counts |
| Section 4.2 | "three more experiments", "two pooled clauses" | counts |
| Section 4.2 | "the two server machines new to it" | `\jiUnstableN` exists |
| Section 5 | "less than a third of the CPU rate" | threshold drawn from the data (`jkSlowRatioMax` 0.32) |
| `app_more` | "two of the three stable ones" | count of P3 failures |
| `app_more` | "the two unstable machines" | `\jiUnstableN` |
| `app_more` | "On both EPYC hosts" | count |
| `app_more` | "driver 570", "drivers 580 and later" | from `gpu.csv` |
| `app_more` | "the EPYC 7302 host" | `\jiVsName` exists |

The registered constants (6%, 8%, 4%, 2%, 1.01, 1.02, 0.995, 0.03, κ = 3/2) and the class threshold `\jlClassCores` are acceptable as typed.

### 15. Minor (build): Table 4 is 25.5 pt wider than the text block

It is the only overfull box in the main text: `tab_regtests.tex`, at lines 4–11. In the PDF, its last column runs into the right margin of page 5.

## What reproduces

**Table 4 (`tab_regtests.tex`), every cell** (`regtests.py`).

| Job | Machines (new) | Cells within | New machines' cells | Median \|error\| |
|---|---|---|---|---|
| 105, plain form | 4 (2: 105a, 105b) | 5/8 | 2/4 | 4.24% |
| 106 | 5 (2: 106c, 106d) | 12/20 | 0/8 | 4.67% |
| 107 | 2 (2) | 1/8 | 1/8 | 31.75% |
| 108 | 3 (3) | 4/6 within 8% | 4/6 | 6.70% |

- **New machines.** 106e's GPU (eadc6e4d) is 099e's. 108d's GPU matches no earlier launch.
- **Job 106 without 106c and 106d:** 12/12 within 6%, median 2.34%. The registered clause at 25% also holds there (overlap 1.64% against half the plain form's, 2.99%), so "held" is right.
- **Job 108 without 108a:** 4/4 within 8%, but 2 cells beyond 6% and a median of 5.25%, so "failed" is right.
- **The other verdicts** follow from the registered band and median clauses, except job 107 (defect 2).

**Job 108** (`tab_job108.tex`, `wsg_job108.tex`).
- **Gates.** 108c's GPU (25e4da5b) is 100b's. 108e had 323 GB in use. 108b's base rounds were 60.972 and 65.245 ms, a 7.01% spread.
- **Valid hosts.** Spreads 0.13% (a), 0.10% (d) and 0.02% (f). Loss 0.1893–0.1904 in every round.
- **G.** 4.186/4.241 ms (d) and 4.488/4.568 ms (f).
- **Profiles and drivers.** The 7543 profile captured 3 kernels, the 7K62's 1 (G 0.0015 and 0.017 ms). Drivers: 570.211 and 570.133 on the EPYCs; 580.178 and 595.91 on the Ryzens.
- **Fallback G.** The median of 8 profiles: 4.292 and 4.469 ms. With G at the range ends of those profiles, the EPYC 7543's error is 26.7–35.1%.
- **Every table cell reproduces:** B_host 22/36/65/84; Eq. (3) and measured time; implied rate 0.71, 0.92, 0.99, 0.91, 0.94, 0.58 and 0.61.
- **Macros.**
  - Errors: `jmErra` 34.2/27.9, `jmErrb` 26.8, `jmProfOnlyMed` 5.3 (5.255), `jmNewConsAbs` 0.9–7.1.
  - Counts: 6 launched, 2 gated, 1 invalid, 3 valid.
  - Scorecard: 21 clauses, 13 held on the point estimate, 6 failed, 2 untested. The six failures are the ones `app_wsg` lists.
- **Timing.** 88e954c was committed at 07:39:32 UTC; 108a–c were rented at 07:39:36–38, d at 07:46:01, e at 08:15:58 and f at 08:18:31. Hosts were taken in list order, and the header was never amended.

**The processor split** (Section 4.3).
- Consumer: 39 valid launches on 25 GPUs, implied rate 0.880–1.164, 37 within 8% and 31 within 6%.
- Server: 8 launches, 5 invalid. The valid ones are 0.269 (107d), 0.585 (108a) and 1.038 (105a), with errors from −58.9% to +3.0% (see defect 1).

**Fig. 2 and `wsg_decomp.tex`** (`decomp.py`; own G where profiled, otherwise the reanalysis median).
- **Rows.** 28: 25 consumer and 3 server (105a, 107d, 108a). Own G is used on 105b, 107b, 108d and 108f among the consumer rows.
- **Shares at 11%:** G 11.9–45.4%, MIN 31.2–54.0%, beyond MIN 22.5–34.9%, G as a share of the gap 22.51–66.0%. Median law error −1.5%.
- **Shares at 25%:** 22.3–64.1%, 22.8–36.1%, 9.9–41.3% and 33.4–87.8%. Median law error +2.0%.
- **Server shortfall:** 0 / 34.3 / 58.9% at 11% and 0 / 27.7 / 52.2% at 25%.
- **Consumer shortfall maximum:** 10.6% at 11% and 4.6% at 25%.
- **The image.** 7 stars, all on machines first rented for a registered test. Numbering is consistent across the two panels.

**Section 4** (other numbers).
- **The relation's history.**
  - The plain form is within 6% on 28 of 30 launches at 11%.
  - Over 34 launches at 25%, the overlap form puts 33 within 6% and the plain form 18.
  - At 11% the overlap form puts 29 within 6% and the plain form 32. The text quotes 25% only, correctly.
- **Job 106.**
  - Re-rented machines: 12 cells, median 2.3%.
  - Half-admission form: 11/12.
  - The two server machines: 10.2–52.4%.
- **Job 107.**
  - Dual-socket server: 52.2–58.9%.
  - 5900XT: up to 11.3%.
- **Probe threads.** 118 GB/s at 128 threads against 151 GB/s at 16.
- **Bound exceeded.** "More than 5% faster than the probe on 13 of 68 launch-budgets" reproduces with the median G. With job 105's own profiles it is 12, because 105b at 25% drops from 1.073 to 1.026.

**Section 5 and `app_more`.**
- **Fewest-admission set.** The 7 earlier machines and the 2 valid machines of job 107 all gained (values in defect 3). Fig. 3 has 9 stars at 11% and 6 at 25%. Copy ratio 0.597–0.726.
- **Slowest link.** On the EPYC 7302 (B_p 13.0) fetchplan was 0.824–0.878 per round. Its loss was ≥ 0.208 in every round and model, so "wrong outputs throughout" is right.
- **dk pooled over 5 machines** (106a, b, e; 107b, d): median 1.019, largest gain 5.7%, one loss of −1.3% of 10 cells.
- **dk host-read cuts:** 4.3–5.8% at 11% and 6.3–10.5% at 25%. The lrn cut is 6.6–14.2%.
- **dk on job 106's stable hosts:**
  - 1.003–1.024 at 11% and 1.008–1.057 at 25%; Qwen3 1.010–1.038.
  - It fell short of 1.01 on 106a (11%) and 106e (25%).
  - The pooled median is 1.04 over all 5 hosts and 1.01 on the stable ones.
- **dk on job 107:** 0.987–1.074, median 1.021.
- **"A learned admission order gained no more."** On the stable hosts lrn's median is 1.012 against dk's 1.0145 at gpt-oss, and its maximum 1.055 against 1.057. It is ahead of dk at 3 of 6 gpt-oss cells, by at most 0.004.

**Section 6.** Layer-ahead copy at 11%: 1.042 (9950X, ratio 1.04), 0.814, 0.909 and 0.722. Registered P3 of job 105 held.

**Figures.** Fig. 1's colours and roles match its caption and the Setting. Fig. 3's caption matches its points.

**Build.**
- `paper.tex` builds to 37 pages and `supplement.tex` to 41.
- There are no undefined references, citations or macros, and no `[pend.]`.
- No appendix text references the removed main-text `tab:job106`. The `tab_job106.tex` file is still generated but no longer input.
- The Conclusion and the start of the References are both on page 9, so the main text now ends on page 9 (it ended on page 10 in Number check 13).
- The overfull boxes are `tab_regtests.tex` (main text, defect 15) and `tab_audit.tex` (appendix, as before).
- The hyperref "duplicate destination" warnings (35) are not new: the e4fdc32 build gives 33.
