# Number check 12 (7 October)

## Scope

I checked the main text of `paper/paper.tex` at HEAD (3dc8b21), from the abstract through the Conclusion. I also checked the changed appendix text: `app_more.tex`, the job 105/106 paragraphs of `app_wsg.tex`, and the additions to `app_value.tex` and `app_traces.tex`. The tables and figures in scope were `tab_job106.tex`, `tab_job106_all.tex` and `figs/decomp.pdf`, with the macros in `wsg_job106.tex`, `wsg_decomp.tex` and the new `sl*` macros in `wsg_sumlaw.tex`.

All job 106 numbers were recomputed from raw files on the GPU branch (`results/106?_onlineadmit@vast`):
- time per token: per-problem `decode_ms/n_decode`, paired by problem against base within each round;
- counters: `st_*.json`;
- rates: B_c, B_p and B_host parsed from `concur.txt` on my own;
- G: from `g_prof.json`/`q_prof.json`, cross-checked against the `prof_*.json` medians;
- the law: solved in closed form, as the larger root of T² − (G + t_M + t_A)T + t_A·G = 0.

The same approach was used for the exploratory check over jobs 093–105, the decomposition figure, the combined fewest-admission set (jobs 104–106), and the relaunch comparisons by GPU UUID.

My scripts are in `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/nc12/`: `j106.py`, `sumlaw.py`, `decomp.py`, `relaunch.py`, `ratiopred.py` and `j104105.py`. The paper was built with latexmk in a scratch copy.

Every macro value in `wsg_job106.tex`, `wsg_decomp.tex` and the new `sl*` macros reproduces. The defects below are about validity, scope, wording and missing reports of failures, not about arithmetic.

## Defects (most severe first)

### 1. Major: the EPYC 7302 host (106c) computed wrong outputs, and the paper draws a finding from it

**Text (Section 5).** "Below some link rate even the fewest-admission set must not be copied in the step: on the EPYC 7302 host, whose link moves 13 GB/s (a ratio of 0.14), it lost in each of its three launches (0.82–0.88×, greedy 0.65–0.71×), since copying its 10.2 admissions per token takes 10 ms of link time against the deployed cache's 18 ms step."

The same result supports three other passages:
- Introduction: "gains wherever the link is not extremely slow".
- Conclusion: "the fewest-admission one gains wherever the link is not extremely slow".
- Limitations: "the fewest-admission set loses on one more with a much slower link".

**Evidence.** The teacher-forced NLL per token in the `ec_*.jsonl` rows shows the host's outputs were corrupted in its timed runs:

| Run | NLL per token on 106c | Every other host, jobs 099–106 |
|---|---|---|
| gpt-oss 11%, rounds 1–2 | 0.208 | 0.189–0.191 |
| gpt-oss 11%, round 3 | 0.420 | 0.189–0.191 |
| gpt-oss 25%, both rounds | 0.420 | 0.189–0.191 |
| Qwen3, both budgets | 0.446 | 0.085–0.086 |
| Untimed lookahead run (`la_g.jsonl`) | 0.190 | 0.190 |

- Because the lookahead run was normal, MIN's plan was computed from correct routing that the timed runs then did not follow.
- The counters on 106c change between rounds; on every other host they are identical round to round. In round 3:
  - base misses were 58.16 per token, against 60.40 in rounds 1–2;
  - the fewest-admission set had 55.03 misses and 8.26 copies per token, against 41.09 and 11.19 in rounds 1–2 (40.73 and 11.86 on the other hosts);
  - `plan_already` was 6726 against 3814, and `oracle_refused` 128 against 68.
- The "10.2 admissions per token" is the mean of 11.19, 11.19 and 8.26.

**Fix.** Treat 106c as a failed run, not as a slow-link data point:
- Remove the "Below some link rate…" sentence.
- Remove the "not extremely slow" qualifiers in the Introduction and Conclusion, or state that no valid run shows the fewest-admission set losing.
- Remove the 106c clause from Limitations.
- In `app_more` and the Table job106all caption, say that 106c's model outputs diverged (the NLL figures above).
- Add an NLL-parity gate to `scripts/job106.py`.

### 2. Major: the registered test of the law is presented as passing, but every host it passed on was already in the exploratory set, and both new machines failed

**Text.**
- Abstract: "registered before the runs, this relation held within 6% at both models and two budgets on every host that ran stably."
- Introduction: "…the relation came within 6% on all 12 model-budget cells of the 3 hosts that ran stably (median error 2.3%)."
- Contributions: "found on 34 launches and tested, with its predictions committed before the runs, on new machines and a second model."

**Evidence.**

(a) All three stable hosts are machines rented again, by GPU UUID:
- 106a is 105b's Threadripper 9960X (GPU-9eba03f6).
- 106b is Pf (GPU-b8316d93).
- 106e is the panel host Pe (099e; GPU-eadc6e4d, identical `lscpu.txt`, B_host 51.7 against 51.9). The paper calls it "Ryzen 9 9950X" without "again", and `scripts/fig_hostdep.py` calls it "its one new machine".
- The only new machines in job 106, the EPYC 7302 and the Xeon 8347C, are where registered prediction 2 failed on all 8 cells: the law under-predicted by 10–52%.
- So "tested … on new machines" is false for the evidence that passed.

(b) The stability filter does not explain the failures:
- It reuses registered prediction 8, which was a prediction, not an exclusion rule, and it is evaluated only at gpt-oss 11%.
- At gpt-oss 25% the EPYC 7302's deployed cache varied by only 1.6% between rounds (clause 106c-P8-g25 held), yet the law under-predicted there by 28.7%.
- Even the fastest round at 11% is 28% slower than predicted on the EPYC 7302 (16.92 ms against 12.24 ms) and 17% slower on the Xeon (14.52 ms against 12.03 ms).

(c) The registered pooled clause, median |error| ≤ 4% over all host-model-budget cells, failed at 4.67%. The main text does not say so; only the appendix does.

**Fix.**
- Abstract and Introduction: "registered before the runs, it held within 6% on all 12 cells of three re-rented machines whose deployed cache ran stably, and failed on all 8 cells of the two new machines (10–52% under), which we left out after the runs".
- Contributions: drop "on new machines".
- Label 106e "Pe again" in Tables 3 and job106all, in the scorecard caption and in the outcome note.
- Report the failed pooled clause in Section 4.

### 3. Major: "the relation predicts by how much" is not supported at the size of the effects

**Text.**
- Abstract: "…gains a few percent (1.00–1.06×), and the relation predicts by how much."
- Introduction: "the relation predicts each rule's gain within 0.045 in speed ratio".
- Section 4: "predicted how much faster the online admission rules … ran on the stable hosts within a median 0.013 in speed ratio (at most 0.045)".
- Conclusion: "the same relation predicts how much".

**Evidence** (12 gpt-oss rule/host/budget pairs on the stable hosts; `ratiopred.py`).

| Predictor | Median error | Largest error |
|---|---|---|
| The law | 0.0135 | 0.045 |
| "No change" (ratio = 1) | 0.0159 | 0.057 |

- The rank correlation of predicted with measured ratios is 0.52 (Spearman, p = 0.08).
- On Pf at 25% the law predicted 1.027 and 1.010 where 1.057 and 1.055 were measured.
- On the 9950X it predicted 1.022–1.027 where 1.008–1.014 were measured.
- The registered tolerance of 0.03 failed for both rules on Pf at 25% (0.030 and 0.045), and the main text does not say so.
- Ratio predictions were registered and computed only for gpt-oss, but the text does not say so.

**Fix.**
- Replace "predicts by how much" with "predicted each rule's ratio at gpt-oss within 0.045 (median 0.013), about as close as assuming no change (median 0.016); the registered 0.03 failed on Pf at 25%".
- Drop the claim from the abstract and Conclusion.

### 4. Moderate: failed registered clauses of job 106 are missing where the claims are made

**Text (Section 4, learned order).** "…costs 267–439 μs of host time per token on gpt-oss, and runs 0.98–1.05×: it pays where host memory is slow and loses where it is fast."

**Text (Section 4, the margin rule).** "We had registered at least 1.01 at gpt-oss on every host; it fell short on two of the three stable ones."

**Text (Section 4, MIN's schedules).** "Where the link reads at under a third of the CPU rate, the greedy schedule loses…"

**Evidence** (`prereg/scorecard_106.json`, matching my recomputation).
- Learned order: lrn/base ≥ 1.00 failed on the 9960X (0.982 and 0.988). The host-time caps failed on the 9950X: 435 and 439 μs against 400 μs on gpt-oss, and 739 and 1030 μs against 700 μs on Qwen3. None of this appears in Section 4.
- Margin rule: the registered pooled median ≥ 1.02 "held" at 1.04 only because of the unstable hosts. On the stable hosts it is 1.014. Section 4 does not say so.
- MIN's schedules: predictions 6 and 7 failed on 106c, where the fewest-admission set was predicted to gain. On 106d, whose ratio of 0.23 is under a third, the greedy copy gained in rounds 2 and 3 (1.27 and 1.26), and the fewest-admission set lost in round 1 (0.996).

**Fix.** Add one clause at each claim, for example: "(registered ≥ 1.00 failed on the 9960X; the 400/700 μs caps failed on the 9950X)".

### 5. Moderate: "launch" means two different things, and the launch-level intervals are narrower in meaning than they read

**Text.**
- Section 2: "machines rented again came within 1.9% of their first launch". Here a launch is a rental.
- Section 3: "Found on 34 launches". Also rentals.
- Section 4: "Repeated as three separate launches on both of those machines in job 106 … the launch-level 95% intervals (resampling launches, then problems)". Here a launch is a round.
- Section 4: "varied … between launches". Rounds.
- Table 3 and Table job106all captions: "launch-level 95% intervals (launches, then problems)".

**Evidence.**
- In job 106 a "launch" is a round: a separate process within one rental, run minutes apart (`jobs/106_onlineadmit@vast.sh`: "one process per round ('launch')").
- The same Section 4 paragraph counts each job 106 host as one launch ("10 launches on 7 machines").
- Variation between rentals is larger than variation between rounds. On the 9960X, base at gpt-oss 11% varied 0.1% between rounds but rose from 9.12 ms (105b) to 9.41 ms (106a), +3.2%, between rentals. So the intervals do not capture re-rental variation.
- Qwen3 cells ran one round and gpt-oss 25% ran two. Table 3's "launch-level" intervals for Qwen3 are therefore intervals over problems only.

**Fix.**
- Call them "rounds (separate processes on one rental)".
- State that the intervals cover process-to-process variation only.
- Mark the Qwen3 intervals as over problems only.

### 6. Moderate: the relaunch bound is out of date

**Text (Section 2, repeated in Limitations).** "machines rented again came within 1.9% of their first launch".

**Evidence** (`relaunch.py`, matched problems).
- `jcRelaunchBaseMax` (`scripts/job101.py`) covers relaunches in jobs 100–104 only.
- 106a against 105b: +3.2% at gpt-oss 11% and +2.6% at 25%.
- 106b against 099f: −2.6% at 11%.

**Fix.** Recompute over all re-rentals through job 106. The bound becomes 3.2%.

### 7. Moderate: the exploratory comparison of the two forms uses different denominators

**Text (Section 4).** "with G the profiles' median, it is within 6% … on 28 of them at 11% … with it, 29 and 33 of the 34 launches of jobs 093–105 are within 6% at the two budgets."

**Evidence** (`sumlaw.py`, all 34 launches).

| Form | Within 6% at 11% | Median at 11% | Within 6% at 25% | Median at 25% |
|---|---|---|---|---|
| Plain | 32 of 34 | 0.0% | 18 of 34 | +6.0% |
| With overlap term | 29 of 34 | −1.2% | 33 of 34 | +2.2% |

The overlap term makes the 11% budget worse and fixes 25%. The macro `slPlWithinLow` (32) is computed but not used. The job 106 header itself states "32 / 18 of 34".

**Fix.** Report both forms on the same 34 launches at both budgets.

### 8. Moderate: the slow-link statements are wider than the evidence

**Text.**
- Section 5: "Where the link reads at under a third of the CPU rate, the greedy schedule loses (0.85–0.93×) and the fewest-admission one gains (1.02–1.06×)."
- Abstract: "the one with the fewest admissions also wins where the PCIe link is slow."

**Evidence.**
- The statement rests on 5 launches of 2 machines with ratios 0.28–0.32.
- Job 106's other two hosts under a third, the Xeon at 0.23 and the EPYC at 0.14, either contradict it (defect 4) or are invalid (defect 1).

**Fix.** "on the two stable machines whose link reads at 0.28–0.32 of their CPU rate".

### 9. Minor: the serialisation share does not match Figure 1's machines

**Text (Section 4).** "Figure 1 splits each machine's time by Eq. (3) … The first is serialisation: the GPU's compute in series with the reads, 22–51% of the gap to Eq. (1)."

**Evidence.**
- `slGGap*` is computed over the 30 launches of jobs 093–104, using each launch's implied G from the plain form.
- On Figure 1's 23 machines, with the profiled G and the overlap form, the share at 11% is 23–68% (`decomp.py`).
- On 105b (the 9960X) it is 4.28/(9.12 − 2.85) = 68%, or 63% with the implied G.

**Fix.** Recompute on Figure 1's machines, or scope the sentence to jobs 093–104.

### 10. Minor: 106e (Pe) is counted twice in Figure 3

**Evidence.**
- `scripts/fig_hostdep.py` adds 106e's greedy-copy point at gpt-oss 11% (1.348 at ratio 1.02). Pe's own point (099e: 1.348 at 0.995) is already plotted.
- `hdFetchNLow` went from 21 to 22 and `hdHosts` from 26 to 27 for a machine that is not new.

**Fix.** Keep only 106e's fewest-admission point, and correct the script comment.

### 11. Minor: the margin rule's range in the abstract and Introduction lacks its scope

**Text.**
- Abstract: "gains a few percent (1.00–1.06×)".
- Introduction: "gains 1.00–1.06× on gpt-oss".

**Evidence.**
- The range covers gpt-oss on the three stable hosts only.
- Its lower end is no gain: the 9960X at 11% was 1.003 [0.99, 1.02].

**Fix.** "0–6% at gpt-oss on the three stable hosts".

### 12. Minor: the stable-host rule is applied but not stated in several summaries

**Evidence.**
- Section 4: "5.4–5.9 ms for Qwen3" covers the stable hosts only. On all five hosts it is 5.3–6.3 ms.
- Section 5: the learned order's "267–439 μs" is stable hosts only. It is up to 693 μs on the unstable hosts.
- Figure 1 caption ("One launch per machine (23 machines)") does not say that job 106's two new machines are left out.
- The rule's cell, gpt-oss 11%, is never stated anywhere ("varied … between launches").

**Fix.** Add "on the stable hosts" to each, and "at gpt-oss 11%" where the rule is defined.

### 13. Minor: app_more's description of the unstable hosts is loose

**Text.** "Before our job started, 131–132 GB of their host memory was already in use … and their probes' CPU read rates fell when more threads than physical cores read."

**Evidence.**
- `free.txt` is written at the job's gate, after the job had started.
- On the EPYC 7302, which has 2 × 16 physical cores and 15 usable in its container, the rate peaked at 16 threads (107.7 GB/s) and fell at 32. Thirty-two threads is not more than its physical cores, so the stated cause does not fit this host.
- The stable 9960X also falls slightly, from 167.4 to 162.6 GB/s at 48 threads on 24 cores.

**Fix.** "measured at the job's gate", and give the thread counts at which the rate fell.

### 14. Minor: wording

- "it pays where host memory is slow and loses where it is fast" (learned order): with three hosts, the largest gain is on the mid-speed host (Pf, 94 GB/s: 1.028 and 1.055). The slowest host (9950X, 52 GB/s) gains only 1.014 and 1.010. Say "it loses on the host with the fastest memory".
- "reads less still (7–14% fewer)": the 7–14% is relative to the deployed cache, not to the margin rule.
- "the two schedules tie" (fast link): the fewest-admission set is 0.009 behind in every round on 106e and 0.014 behind on 105f. Say "within 0.01".
- "median error 2.3%" in the Introduction and Section 4 means median |error|. The exploratory medians nearby are signed.
- Contributions, "on 23 machines": counting job 106's new machines, 25 distinct GPUs ran the oracles.

## Checked and correct

**Timeline of the predictions.**
- Job 106's header was committed in 525a9c9 at 01:43:48Z.
- 106a–d were rented at 01:43:58–01:44:02Z.
- The amendment 89c5c3b, at 01:44:13Z, only replaces host e and adds the note; it changes no prediction.
- 106e was rented at 01:44:16Z.
- Job starts were 01:44:57–01:48:03Z (`manifest.json`).
- G was profiled before every timed run on every host (`stdout.log` order).
- The learned model files' hashes matched on every host.
- `online_admit.json` was committed at 01:45:01Z, 4 s after 106c started, but κ, the margins and the replay figures are already in the header.
- κ and margin choices (3, 2, 3, 2 and 0.4, 0.4, 0.4, 0.3) are the minima of the tuning on other text.

**Scorecard.** 181 clauses: 21 held, 112 held on the point estimate, 48 failed. Of the failures, 36 are on the unstable hosts, 11 on the stable hosts and 1 pooled. The appendix's list of the 11 stable failures is accurate.

**Rates and G on all five hosts.** B_c, B_p and B_host reproduce, and so do the link-to-CPU ratios (0.32, 0.29, 0.14, 0.23, 1.02). G from the files equals G recomputed from the `prof_*.json` medians.

**Every number in Table 3 and Table job106all.** G, law, measured time, dk, lrn and the fewest-admission speed all reproduce; intervals agree within bootstrap noise.

**Law errors.**

| Hosts | Form | Range or summary |
|---|---|---|
| Stable | With overlap term | −3.3 to +6.0% (106a Qwen3 25% is 5.99%); median \|error\| 2.34% |
| Stable | Plain | gpt-oss 11% −0.5 to 3.5; 25% 4.6 to 8.1; Qwen3 12.5% −4.1 to 6.6; 25% 0.1 to 10.1 |
| Unstable | With overlap term | 10–52% under |

**Stable-host summaries.**
- `jiUnstableVar` 11% and 39%; base variation on the stable hosts 0.0–0.9%.
- Host memory in use: 131–132 GB on the unstable hosts, 12–15 GB on the stable ones.
- Margin rule speed: 1.00–1.02 and 1.01–1.06 at gpt-oss; 1.01–1.04 on Qwen3.
- Margin rule read cuts: 4–6% and 6–10%.
- Learned order: speed 0.98–1.05, read cuts 7–14%, host time 267–439 μs.
- Ratio prediction: median 0.013, largest 0.045.
- Fewest-admission on the slow-link hosts: lower interval ends 1.01 and 1.05; every round above 1 and every greedy round below 1 on Pf and the 9960X.
- EPYC 7302 per-round values: 0.82–0.88 and 0.65–0.71; 13 GB/s; 10.2 copies; 10 ms; 18 ms. The numbers reproduce, but see defect 1.

**Combined fewest-admission set (`jk*`).** 10 launches on 7 machines by UUID; 1.02–1.34. Over the greedy schedule: 0.08–0.17 on 8 launches and 5 machines with ratios 0.28–0.63. Under a third: greedy 0.85–0.93, fewest-admission 1.02–1.06, on 5 launches and 2 machines. Raw recomputation of jobs 104 and 105 confirms every fewest-admission state there gains with its interval above 1, and is 0.07 behind the greedy set when loaded by the CPU on the fastest link at 25%.

**Exploratory macros.**
- `slOlN` 34; `slOlWithin` 29 and 33; `slOlMed` −1.2 and +2.2.
- `slPlWithin` 32 and 18; `slLawWithinLow` 28 of 30; `slLawErrMidMed` 6.
- The medians of the profiled G are 4.28 and 4.50 ms; the profiles' range is 4.1–4.7 ms over 14 profiles on 7 hosts.

**Figure 1 and `wsg_decomp.tex`.** 23 machines at both budgets. Shares at 11%: G 12–47%, MIN 31–54%, beyond MIN 22–35%. Median law error −1% and +3%.

**Online admission tuning.** `prereg/online_admit.json`'s argmins match the header.

**Build.**
- `latexmk` builds `paper.tex` and `supplement.tex` cleanly.
- No undefined references, citations or macros, and no `[pend.]`.
- The only overfull boxes are in the appendix's `tab_audit.tex`.
- The Conclusion and the start of the References are on page 9 of 35, so the main text ends by page 10.

**Other appendix text.** The `app_traces.tex` sentence about Table lomo agrees with the table. The `app_value.tex` and `app_wsg.tex` job 105 wording ("the plain form of Eq. (3)") is consistent with the main text.
