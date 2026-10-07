# Number check 16 (7 October)

## Scope

The main text of `paper/paper.tex` at HEAD `4da309d` (abstract through Conclusion, with the macros of the `wsg_*.tex` inputs, Table 1 claims, Table 3 `tab_dm.tex`, Figure 2 `figs/decomp_measured.pdf`), the Section 4 summary of `paper/app_relation.tex`, and a clean build. The working tree equals HEAD. I did not read `reports/` and modified nothing except this file.

## Method

My own scripts are in `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/nc16/`:

- `dm_recompute.py` recomputes the decomposition from the raw rows `results/<job>/ec_g_C{14,32}*.jsonl`. It labels each row by the basename of its `stats=` path, computes per-problem ms per token as `decode_ms/n_decode`, and averages over the problems common to the five states. It takes the GPU UUID from `nvidia-smi-q.txt`. The only external inputs are the per-launch bound (`limit_ms` in `prereg/reanalysis_hosts.json`) and T_GPU (2.9, and the unrounded 2.94). The arguments select which launch to keep per GPU (first or last). `dm_recompute20.py` restricts every machine to the first 20 problems.
- `plan_recompute.py` covers the fewest-admission schedule (`fetchplan`) and the admission margin (`dk`) against `base`, from the raw rows, pooled over rounds. It keeps one value per GPU UUID and summarises with a geometric mean and a percentile bootstrap over machines.
- `bound_share.py` computes the bound over the deployed cache's time on the machine set of `fig_decomp.py`. The bound is R*·S/B_host, with B_host the highest probe reading.
- `relaunch.py` compares the deployed cache's time across relaunches of the same GPU.
- `dc_check.py` computes D(W50)/C on the AIME traces with my own sliding-window distinct count (every start position).
- Inline checks covered:
  - `prereg/value_map.json` shares and W50, recomputed from raw reads.
  - The registered tests and the claims table, against the job headers in `gpu-branch/jobs/*.sh`.
  - The implied read rate on 39 consumer launches.
  - The bound share of the headline systems in `tab_headline.tex`.
- Build: `git archive HEAD paper` into `nc16/build`, then `latexmk -pdf -g`.

## Defects (most severe first)

### 1. The Conclusion claims every measured system runs at a third to a half of the bound
**Text (Conclusion):** "The systems we measured, ours included, run at a third to a half of it."

**Evidence:** Table 2 (`tab_headline.tex`) gives these shares of the datasheet bound:
- Stock llama.cpp runs at 8–21% at every cell on hosts B and S.
- FreeToken runs at 20–42%; it is below a third at 8 of 12 cells.
- Our own cache runs at 25–29% at gpt-oss 25% and 40%.
- Over the 25 consumer machines, our cache's bound share is 31–54% at gpt-oss 11% and 17–36% at 25% (`bound_share.py`).
- The published systems in the audit sit at a median of 13.6%.

The "third to a half" range holds only for our cache at the smallest budget.

**Fix:** "Our cache runs at 31–54% of it at the smallest gpt-oss budget and 17–36% at 25%. FreeToken and llama.cpp run lower, and published systems at a median of 13.6%."

### 2. Figure 2 caption and Section 4 misstate the two slow-link machines at 11%
**Text:**
- Figure 2 caption: "On the two machines with slow links (orange), reading once in the step is slower than reading twice."
- Section 4: "On the two machines whose link reads at most 0.40 of their CPU rate, reading once in the step is slower than reading twice."

**Evidence:** On 099d (i9-13900, ratio 0.3997) at gpt-oss 11%, MIN read once is *faster* than MIN read twice: 16.91 against 17.74 ms, a ratio of 0.953. Its Shapley "twice" share is +2.3%, so reading once saves time there. The figure itself shows that orange line falling from "MIN, 2 reads" to "MIN, 1 read".

The statement holds in these cases:
- At 11% only on 099f (ratio 0.29): 16.07 against 14.80 ms, ratio 1.086.
- At 25% on both machines: 1.075 and 1.173.
- For the deployed admissions (foa against base), on both machines at both budgets: 1.019 and 1.056 at 11%, 1.085 and 1.173 at 25%.

The choice of relaunch (100a for 099d's GPU) does not change this; that launch gives 0.954.

**Fix:** "...reading MIN's admissions once in the step is slower than reading them twice at 25%, and at 11% on the slower of the two." Alternatively, state it for the deployed admissions.

### 3. The Section 4 summary of the closed-form account, and Table 1's row for it, misstate the registered tests
**Text:**
- Section 4: "We registered this account before four experiments on machines new to it, and it failed all of them, by up to 11% on a new consumer machine."
- Table 1: "exploratory; failed four registered tests."

**Evidence:** `tab_regtests.tex` and `wsg_regtests.tex` show the following:
- The four tests ended **three failed and one inconclusive**: job 107 registered that it is inconclusive with fewer than three valid machines. The macros `\rtFailedN` = three and `\rtInconclN` = one exist but are unused. The appendix itself says "None passed as registered" and "not passed", not "failed".
- One of the four (job 105) registered the plain form, not the overlap form the appendix calls the relation.
- Jobs 105 and 106 also ran machines rented before. In job 106 the relation held on the three stable ones (12 of 12 cells), so the tests were not "on machines new to it".
- The summary describes the plain form ("kernel time plus its counted host reads at B_host"). The quoted 0.88–1.16 is the overlap form's implied rate, and it is at gpt-oss 11% only. I reproduce 0.880–1.164 over 39 launches on 25 machines; the plain form gives 0.886–1.18.

**Fix:**
- Section 4: "We registered it, first without and then with the overlap term, before four experiments that each added machines new to it. It passed none: three failed and one was inconclusive by its own rule..."
- Section 4: add "at gpt-oss 11%" to the 0.88–1.16 range.
- Table 1: "failed three registered tests; one inconclusive".

### 4. A slow-link result is labelled "(registered)" although no job registered it
**Text (Section 5):** "There the fewest-admission set pays when the CPU serves it and the copies run in the background: 1.11–1.22× (registered)."

**Evidence:** The values come from launches 104a, 105b and 105e (two GPUs, ratios 0.28–0.32), at both budgets: 1.113, 1.217, 1.114, 1.159, 1.112, 1.215. What the jobs registered:
- Jobs 103 and 104 registered for this configuration only counter clauses and "bypassplan/base at least bypass/base − 0.01" (prediction 6). At 104a that threshold is about 0.93; it does not say the set pays.
- Job 105 registered nothing about bypassplan.
- Job 107 registered "bypassplan/base > 1" (prediction 5) and "bypassplan beats fetchplan below 0.35" (prediction 6). Its only machine below 0.35 (107e, ratio 0.21) ran unsteadily.

The previous revision did not carry the label; the restructure added it.

**Fix:** Mark it "(observed)" and keep the registered label for the fetchplan result.

### 5. "Four measured parts" mislabels the decomposition
**Text:**
- Abstract: "Oracles built into the engine then remove the gap to the bound one part at a time."
- Abstract: "...a fifth..., a fifth..., a sixth..., and a quarter..."
- Intro: "The gap has four measured parts... Removing them one at a time in the engine shows..."
- Table 1: "The gap to the bound splits into four measured parts."

**Evidence:**
- Table 3 has five rows.
- The GPU-in-series part is not measured or removed. It is the constant T_GPU = 2.94 ms, the smallest of 14 Nsight profiles (range 2.94–3.38). Section 4 says "No oracle we built removes it", and the Table 3 caption says "T_GPU is profiled".
- A fifth row, "left above Eq. (2) with full foresight", is 20% [15, 25] of the gap at 11% and 11% [7, 15] at 25%. It is not one of the four named parts.
- The abstract's four fractions sum to about 0.82 of the gap, so a reader would take them as the whole.

**Fix:** "Oracles remove three parts one at a time; the GPU's own work in series (profiled) and a remainder of about a fifth make up the rest." In Table 1: "split derived; T_GPU profiled".

### 6. Several range claims hold only at the smallest budget but are not scoped to it
**Text:**
- Abstract: "runs at 31–54% of that bound's speed on 25 machines with consumer processors" (no budget).
- Intro: "runs at roughly a third to a half of the bound's speed."
- Rule 1: "the best system sits at a third to a half of it."
- Intro: "caching the wrong experts is only about a fifth of the gap."
- Section 4 headings: "Caching the wrong experts is about a fifth of the gap" and "Reading admitted experts twice costs as much."

**Evidence:**
- Bound share over the same 25 machines: 31–54% at gpt-oss 11%, 17–36% at 25%. On hosts B and S, ours is 25–27% at gpt-oss 25% (datasheet bound).
- Caching the wrong experts: 20% [16, 23] at 11%, but 29% [27, 31] at 25%.
- Reading twice: 19% at 11% against 7% [3, 10] at 25%, where caching is 29%.

**Fix:** Add "at the smallest gpt-oss budget" to each, or give both budgets.

### 7. Overgeneralisations about the fewest-admission schedule and where foresight pays
**Text:**
- Conclusion: "Foresight removes about half of the gap when each admitted expert is read once, with the fewest admissions, on the read path the machine favours."
- Rule 3: "Among equally good schedules, the one with the fewest admissions wins; online, a higher admission margin gains about 1.021× at no cost."
- Intro, in the same spirit: foresight helps "when the schedule admits as rarely as optimality allows".

**Evidence:**
- The roughly half (54% [49, 58] at 11%, 56% [52, 60] at 25%) was measured with the *greedy* MIN schedule prefetched, on the 13 machines whose link reads at least 0.5 of the CPU rate. The fewest-admission schedule is not part of the decomposition.
- On slow links the favoured path (the fewest-admission set served by the CPU) closes only 15–23% of the gap: 104a 0.17 and 0.23, 105b 0.15 and 0.17, 105e 0.16 and 0.23.
- On the two machines with link-to-CPU ratio about 1, the fewest-admission schedule is about 1% *slower* than the greedy one: 105f 1.342 against 1.356, 106e 1.339 against 1.348. The previous text said "the two schedules tie" there.
- The margin result:
  - "at no cost": `dk` lost on 107d at 25% (0.987). The 25% interval over the five machines is [1.000, 1.039].
  - "about 1.021×" is the 11% value.

**Fix:**
- Conclusion: "removes about half of the gap where the link is fast and each admitted expert is read once".
- Rule 3: "...wins where the link is slower than the CPU and ties where it is not".
- Rule 3: drop "at no cost", or say "1.02× at both budgets; it lost on one of five machines at 25%".

### 8. Next-token and horizon claims are not scoped to the workload they were measured on
**Text:**
- Abstract: "Exact routing of the next token would recover at most a third of what full foresight saves in host reads; half of it needs the routing of about 0.65 C..."
- Intro: "...on 9 models. Exact routing of the next token recovers at most a third of it."
- Conclusion: "...far beyond the next token."
- Rule 5: "target the next ≈0.65 C".

**Evidence:**
- "At most a third" (0.31, Qwen3 12.5%) is measured on the two engine models' AIME routing at the four host-bound budgets.
- In the nine-model set (`prereg/foresight/w50_distinct.json`), 5 of 26 model-budget points have W50 below one token (0.64–0.91: olmoe C8, gpt-oss-20b C4, mixtral C2, qwen2-57b C8, phi3.5 C2). There, next-token routing recovers more than half.
- "About 0.65 C" is the nine-model median. On the routing the engine actually runs it is 0.66–0.81 C, and 0.81 C at gpt-oss 11%. I reproduce 0.806, 0.693, 0.775 and 0.658.

**Fix:**
- Scope "at most a third" to "on our two models' routing at the host-bound budgets".
- Say "0.65–0.8 C" in the abstract and Rule 5.
- Qualify "far beyond the next token" as "at the host-bound budgets".

### 9. "Barely beats" overstates MIN's set with two reads
**Text (Section 4):** "with two reads, MIN's set barely beats the deployed one at 11%."

**Evidence:** On the 13 fast-link machines, MIN read twice against the deployed cache's time ranges 0.968–1.019 (mean 0.994). It is slower on 6 of 13 (1.003–1.019). The causal clause checks out: admissions rise from 3.5 to 15.8 per token, and total reads rise from about 63 to 67 per token.

**Fix:** "is within 3% of the deployed one either way (faster on 7 of 13 machines)".

### 10. The Limitations still lean on the failed relation
**Text (Limitations):** "the engine's implied read rate exceeded the probe's highest reading by more than 5% on 12 of 84 launch-budgets (tab:robust)."

**Evidence:** `robustness.py` defines the implied rate as (M + A(1 − G/T))·S/(T − G)/B. That is the overlap form of Eq. (3), which now lives in the failed-relation appendix. This is the only remaining main-text number that depends on that relation.

**Fix:** Define it without Eq. (3), for example counted reads over (T − G), or say it is the rate the appendix's account implies.

### 11. The ratio to Eq. (2) is not scoped to the fast-link machines
**Text (Section 4):** "With all three parts removed, the engine's time is 1.07–1.38 times Eq. (2) at 11%."

**Evidence:** The range covers only the 13 fast-link machines; the two slow-link machines are at 1.59–1.67. With T_GPU = 2.9 as stated in the paper, the maximum is 1.386, which rounds to 1.39. The 1.38 comes from the unrounded 2.94.

**Fix:** "on those 13 machines". Optionally state T_GPU as 2.94.

### 12. The abstract's machine count lacks its selection criterion
**Text (Abstract):** "At the smallest gpt-oss budget on 13 machines..."

**Evidence:** 15 machines ran all the states; 13 is the subset whose link reads at least half the CPU rate. The criterion appears only later, as "the PCIe link is fast".

**Fix:** "on the 13 machines whose link reads at least half their CPU rate".

### 13. "For the same hits" contradicts the Limitations
**Text (Section 5):** "makes 0.60–0.73 of the greedy one's copies for the same hits."

**Evidence:** The Limitations say that in the engine the fewest-admission schedule misses 2.5–5.3% more than the greedy one.

**Fix:** "for the same hits in replay".

### 14. A registered clause that failed on one machine is labelled only "(registered)"
**Text (Section 6):** "an exact 16-token window recovers 0.78–0.93 of the gain in time at 11% (registered)."

**Evidence:** Job 099 registered ≥ 0.80. That clause failed on Pf (0.781) and held on 9 of 10 machines (`prereg/scorecard_099.json`).

**Fix:** "(registered; held on 9 of 10)".

### 15. Rule 5 uses the wrong horizon as its target
**Text (Rule 5):** "Predict far enough ahead to hide the GPU's own work... target the next ≈0.65 C distinct experts."

**Evidence:** The 0.65 C horizon is what halves foresight's *read* savings. It is not what hiding T_GPU needs: the "late" part was removed by copies issued three steps ahead, and no oracle removed the serial part.

**Fix:** Separate the two targets. Hiding T_GPU needs reads issued before routing; the read savings need about 0.65–0.8 C ahead.

## Item-by-item results

### 1. Measured decomposition (Section 4, Table 3, Figure 2): reproduces from raw rows
**Machine set:**
- 18 launches ran base, bypass, foa, fetch and both3p at both gpt-oss budgets; 15 have distinct GPU UUIDs.
- 13 of those have a link-to-CPU ratio of at least 0.5. The slow ones are 099d (0.3997) and 099f (0.286).
- 096a and 096b time 30 problems and the others 20. Restricting all machines to the first 20 changes no rounded share.
- Keeping the later relaunch per GPU (100a, 100c, 101a) changes no rounded value.
- My raw means match `ms_*` in `reanalysis_hosts.json` exactly.

**Mean shares over the 13 machines** (95% bootstrap interval over machines; T_GPU = 2.94):

| Part | gpt-oss 11% | gpt-oss 25% |
|---|---|---|
| What is cached | 19.5 [15.8, 22.9] | 28.6 [26.6, 30.5] |
| How it is read (twice) | 19.3 [15.8, 22.9] | 6.7 [3.3, 9.9] |
| When it is read (late) | 14.6 [10.5, 18.3] | 20.5 [18.2, 22.3] |
| GPU's work in series | 26.9 [24.0, 29.5] | 33.4 [30.5, 36.0] |
| Left above Eq. (2) | 19.6 [14.7, 25.0] | 10.9 [6.6, 15.2] |

- Every Table 3 cell matches after rounding.
- With T_GPU = 2.9, the serial part is 26.5 and the remainder 20.0 at 11%. Table 3's rounded values are unchanged.
- The parts sum to the gap on every machine, and the means sum to 100.0%.

**Other Section 4 quantities:**
- The best oracle closes a mean of 53.8% [49.4, 58.0] at 11% and 55.7% [51.6, 59.8] at 25%. It is the prefetched oracle on 14 of 15 machines.
- On the slow-link machines the best oracle closes at most 16.8% (the paper's 17%) at 11% and 32% at 25%.
- Prefetched over Eq. (2) is 1.066–1.384 at 11% (paper: 1.07–1.38) and 0.991–1.32 at 25%.
- Defect 2 is the only statement in this section that fails as worded.

### 2. Verbal quantities
**Supported:**
- "about a fifth / a fifth / a sixth / a quarter" in the abstract, at 11%: 20, 19, 15, 27.
- "about half": 54 and 56.
- "a quarter to a third" (Rule 5): 27 and 33.
- "largest part": the serial part is largest at both budgets.
- "at most 17%".
- "at most 0.40": 0.3997.
- "most of the rest": 61 of 80 points at 11%.
- "Machines differ far more than problems": over the 13 panel and window machines, machines carry 92–96% of the log-variance of the deployed cache's time and of the MIN-1-read gain; problems carry 1–8%.
  - The supporting evidence does reproduce: 099a against 099f gives a 29% same-CPU difference, and relaunches agree within 3.2% (the maximum is 105b against 106a; all 9 relaunched GPUs are within 3.2%).
  - That evidence, however, compares machines with relaunches, not with problems. Citing the variance split would fit the claim better.

**Not supported at the stated scope:** defects 1, 5, 6, 7, 8 and 9.

### 3. Table 1 against the job headers
**Correct as worded:**
- "registered thresholds held (3 hosts)": job 102's token mode ≥ 0.80 and layer mode ≥ 0.50 held. Its five failed clauses all failed because layer mode did *better* than predicted.
- "registered and held on host B; host S's registered bands failed".
  - Jobs 080 and 081 registered leads at gpt-oss 11/25/40%, Qwen3 12.5% and Qwen3 43.75%, and ≥ 1.8×/2× over llama.cpp; all held.
  - Job 089's ±0.06 bands missed at 5 of 6 cells.
  - Note: host B's Qwen3 25% FreeToken cell was selection-only, not registered.
- "states registered; split derived": jobs 096, 099, 100 and 101 named all five states, with predictions.
- "registered on 7, held; two losses on unsteady machines": the registered launches were 104a, 105a, 105b, 105e, 105f, 106a, 106b, 106e, 107b and 107d. These cover 7 GPUs; 107b and 107d have links of 26.2 and 51.6 GB/s, above job 107's 20 GB/s condition.
- "registered; partly held" (admitting less): job 106's dk/base ≥ 1.01 failed at 106a 11% (1.003) and 106e 25% (1.008).

**Wrong:** the closed-form row (defect 3).

### 4. Fewest-admission schedule and admission margin (Section 5): reproduces from raw rows
**Fewest-admission schedule at 11%:**
- Nine stable GPUs give a geometric mean of 1.215 [1.130, 1.304]; the minimum is 1.028 (GPU b8316d93, three launches).
- Adding 106c, 106d, 107c and 107e gives 1.166 [1.075, 1.257] over 13.
- The two losses are 106c (0.846, ratio 0.144) and 107e (0.974, ratio 0.205): the two slowest links.

**Admission margin** over five machines (106a, 106b, 106e, 107b, 107d):
- 11%: 1.0214 [1.009, 1.036].
- 25%: 1.0176 [1.000, 1.039].
- Per machine, the 25% values include 0.987 on 107d.
- Paper: 1.021 / 1.018.

### 5. Section 6: reproduces
- Next-token share: 0.184 and 0.073; the maximum over the four host-bound budgets is 0.308 (Qwen3 12.5%).
- W50: 3.75 and 10.18 tokens (paper: 4 and 10).
- An 8-token window right half the time recovers 0.290; two exact tokens recover 0.313.
- D(W50)/C on the AIME routing (my own distinct count over all start positions):
  - gpt-oss: 0.806 at 11%, 0.693 at 25%.
  - Qwen3: 0.775 at 12.5%, 0.658 at 25%.
  - This reproduces 0.66–0.81.
- Nine-model median 0.651, quartiles 0.610–0.721; W50 range 0.64–33.6 tokens.

### 6. Dependence on the failed relation
- The bound shares (31–54%) are the bound over *measured* time. They do not depend on Eq. (3), even though `fig_decomp.py` generates them.
- T_GPU comes from the profiles, not the relation.
- `\cref{eq:sum}` and `fig:decomp` are not referenced in the main text.
- Remaining dependence: the implied read rate in the Limitations (defect 10).
- The Section 4 summary is inaccurate in the ways listed in defect 3. Its numbers do reproduce: 0.88–1.16 over 39 launches, and 11% on a new consumer machine (107b; 105b's errors were 2% and 7%).

### 7. Build
`latexmk -pdf -g` from `git archive HEAD` completes: 42 pages, no undefined references or citations, no `[pend.]` placeholders.

**Box warnings:**
- No overfull boxes in the main text.
- Underfull rows (badness 10000) in Table 1, at `paper.tex` lines 138–139 (narrow p-columns); cosmetic.
- The only overfull box is in the appendix: `tab_audit.tex`, 70.4 pt too wide.
- Duplicate hyperref destinations in the appendix: table.28–30 and figure.12.

**Layout:** The main text ends at the top of page 9 with Section 10 (Conclusion), left column. References fill pages 9–12, and the appendix starts on page 13.
