# Review 23a: "Where the Seconds Go" (MLSys 2027, main track, blind PC review)

## Materials read

- `paper/paper.pdf` (44 pages): the whole main text (pp. 1-11), the references, and these appendices in full: A (map), B (benchmarking checklist), C (names, Tables 8-9) and D (prediction log, Tables 10-12). I skimmed E-N for definitions and table pointers.
- The LaTeX source of the main text and its generated tables: `main_body.tex`, `abstract_body.tex`, `tab_headline.tex`, `tab_dm.tex`, `app_wsg.tex`, `app_prereg.tex`. I read the macro files (`wsg_numbers*.tex`, `wsg_*.tex`) only to see which job each number comes from.
- `paper/supplement.pdf`: page count and structure only. I did not use its clause tables.
- `paper/paper_ieee.pdf` (39 pages): skimmed for the IEEE section, plus `paper_ieee.tex`.
- Artifact (`/home/claude/moe-speed-of-light`). I read these scripts for definitions only: `decomp_measured.py` (states, gap and attributed parts), `job109.py` (loading and validity definitions), `fig_decomp.py` (`best_rates`, `dep_bound`), `speed_limit.py` (`host_rates`) and `sumlaw_paper.py` (`profiles`, the definition of non-expert time). I also read `gpu/vast_ledger.json` (rental start times).
- Raw results (`/home/claude/gpu-branch`):
  - `results/*/ec_*.jsonl`, `st_*.json`, `concur.txt`/`concur2.txt`, `fetch_table_law_gptoss.json`, `prof_*.json`, `g_prof.json`, `bs1.jsonl`, `readsched_C*.txt` and `validity.txt` for jobs 081, 089 and 093-112;
  - the job script headers of 081, 089, 102, 109, 110, 111 and 112;
  - `git log --format='%h %ad'` / `%ct` for the job scripts and result directories, and `git diff` of the job 100/102/105/106/111 headers between their commits.

## Independence statement

I did not open anything under `reports/` except to write this file. I did not open `prereg/*outcome*.md`, `research_notes/`, `apply/`, `paper/paper_v1_prereview.tex`, or any review, number-check, plan or progress-log file. I did not read commit messages; git was used only for hashes, dates and file contents. I modified nothing in either repository.

All my scripts and outputs are in `scratchpad/rev23a/fresh/`. One note on that folder: when I created `scratchpad/rev23a/` it already held files written by an earlier session about an hour before mine (`common.py`, `j109.py`, `panel.py`, `trend.py`, `supp.txt`, `h111_first.txt` and others). I did not open any of them, and I worked only in the new `fresh/` subfolder. Every number in the table below comes from my own code reading the raw rows.

---

## Summary

The paper studies batch-1 decode of MoE models whose experts mostly live in host DRAM, on rented single-GPU machines: mainly RTX 5090, plus a few RTX 4090s. It makes four contributions.

1. **A bound.** Eq. (1) multiplies MIN-with-bypass host reads per token by the machine's highest probed read rate. Two variants follow: a demand bound (Eq. 2) and an ordered bound (Eq. 3). The authors' own llama.cpp expert cache is placed against it at 31-54% of Eq. (1) on consumer machines at gpt-oss 11%.
2. **A measured decomposition of the gap**, using oracles built into that engine. It changes what is cached (MIN's set vs the deployed set) and how an admitted expert is read (once, fetched in the step, vs twice, served by the CPU and then copied in the background). The headline: at gpt-oss 11% on fast links, each change alone closes about 0% of the gap and both together close 33% (15-machine panel) or 35% (5 new machines, registered). A read-ahead oracle closes another 15-19%.
3. **Policies without foresight** built from the engine's own mechanisms (larger margin, layer-ahead copy) recover at most 6% of the read-ahead oracle's gain. A log-linear trend in the link-to-CPU ratio, fitted on RTX 5090s, predicts the oracles' speed on five RTX 4090s within 7%.
4. **A price on foresight** from offline traces: half of MIN's gain over admit-every-miss needs about 0.65 C distinct experts of lookahead.

The paper is unusually explicit about registration: every job header carries predictions committed before the machine starts, and Appendix D scores 565 + several hundred clauses.

## Strengths

1. **The numbers are real and reproduce exactly.** I recomputed 33 quantitative claims from the raw per-problem rows, counters and probes with my own code (table below). All of these match to the stated rounding: every cell of Table 4 (panel and new, both budgets, measured and attributed rows), Tables 5 and 6, the Table 3 speeds, and the running example. Only one is a stale statistic and one is a wording overstatement. This is rare, and it makes the artifact genuinely useful.
2. **Registration is real and auditable.** Job headers were committed seconds to minutes before each rental. My own commit-vs-ledger check finds the minimum lead is 2.7 s (job 105f), matching Appendix D. The header edits made after launch (jobs 100, 102, 105, 106, 107, 111) are all disclosed and, apart from job 100's added prediction 9, change no prediction. Failures are reported in the main text, not buried: the panel's interaction failed on one machine, the job 112 control failed twice, three of four tests of the closed-form account failed, and host S's ratios fell outside ±0.06 at five of six cells.
3. **Machine-as-unit statistics with honest scoping.** The paper reports same-CPU variation (29%) against relaunch variation (3.2% / 5.5%), uses t-intervals over machines, keeps a fast/slow-link split that it admits was drawn post hoc, and keeps a separate post-hoc label for the consumer/server split.
4. **Oracles inside a real engine** rather than a simulator. This separates set, read count and read timing in a way prior offloading papers do not, and the 2x2 interaction is a clean, useful result *for this engine*.
5. **Practical, testable guidance.** The link-to-CPU ratio tells a builder which read path to use, and the second-card test is a genuine out-of-sample check: RTX 4090 deviations of at most 6.6% (11%) and 5.0% (25%), against up to 46% / 69% for a no-change prediction.

## Weaknesses (most important first)

1. **Generality: engine-specific findings are stated as general rules.**
   - The decomposition and the "pay only together" result come from one engine: the authors' patch, with its serve-then-copy path, a background copy published two steps after the admission, and a per-machine fetch table that fetches 17-24 misses per token regardless of the schedule. They also come from one model (gpt-oss-120b) at two budgets, one prompt set (20 AIME-25 problems × 256 teacher-forced tokens) and batch 1.
   - The text says "on this engine", but Section 8 turns the result into builder rules ("Change what is cached and how it is read together").
   - The finding is also specific to the *greedy* MIN schedule. MIN's fewest-admission set read twice (Few-2R) gains 1.03× on fast links (Table 5) and 1.11-1.22× on links at ratio 0.28-0.32 (I get 1.112-1.217 from jobs 104a, 105b and 105e).
   - The Qwen3 side of the decomposition is absent.
2. **The confirmatory weight of "registered" is overstated in the abstract and main text.**
   - The abstract leads with "the two changes together close 35% of the gap … in a test registered before its machines were rented". But job 109's registered band for that number was [0.25, 0.55]. Appendix D itself says such bands "could hardly have failed"; the only clauses that could fail were the sign of the interaction and the per-machine no-foresight bounds.
   - Similarly, "the bound's read time is nearly reachable (registered: 50% or more per layer; held)" rests on a threshold far below the header's own analytic expectation of 85-99%. A ≥50% test cannot confirm "nearly".
   - The Few-1R claim is scoped to machines that "ran it stably". In the data, the two machines where it lost (106c at ratio 0.144 and 107e at 0.205) are also the two slowest links, so the stability condition removes exactly the counter-examples a link-based scope would predict.
   - These caveats belong in Table 1 and the abstract, not only in Appendix D.
3. **The landing-time control (job 112) is confounded, and the conclusion drawn from it is not supported.**
   - The registered manipulation check failed: Few-2R-early's misses are 0.968-0.980 of Few-2R's at 11%, against a registered ≤0.95.
   - The raw counters show something the paper does not report. The early arm makes **1.9× the admissions** of Few-2R at 11% (11.7 vs 6.2 per token on every RTX 5090; 1.5× at 25%). That is about 7% more host reads per token (66.3 vs 61.9).
   - T2 (no slowdown) failed on 7 of 8 machine-budgets: the early arm is 0.006-0.053 slower.
   - T3 ("the second read costs more than the landing") is reported as *held*, but it held only because the "landing" gain was negative.
   - The text nevertheless concludes that "it does show that landing the copies in time is not what holds this engine's two-read arms back". With a failed manipulation and an extra-reads confound, the experiment shows only that *this plan shift* does not help. The Table 1 row should read "inconclusive: manipulation failed".
4. **The "bound" is probe-relative and loose exactly where the headroom headline comes from.**
   - Eq. (1) is labelled "derived" and described in the abstract as "how fast such decoding can be". But the paper's own data show the engine implying rates more than 5% above the probe's best reading on 12 of 84 launch-budgets. Pooled slots save 4.5-18.6% of MIN's reads. The probe was usually taken while the model downloaded.
   - The low end of "31-54%" comes from one high-end-desktop Threadripper 9960X (B_host 176-178 GB/s, ratio 0.32). There, Eq. (3) is 1.56× Eq. (1), and the cache is at 49% of Eq. (3). Without that machine the consumer range is 38-54%.
   - Conceptually, Belady-MIN reads × probed bandwidth is a modest step beyond a roofline. The contribution is the measurement around it, not the bound.
5. **The abstract and introduction overstate how much of the time is explained.**
   - "Oracles inside the engine account for the rest of the time" is not what Table 4 shows. Measured oracle steps close about 48-54% of the gap. The remaining 46-52% is split *by attribution* (subtracting a fixed 2.94 ms T_GPU and the extra reads at B_host), not by measurement.
   - On the two slow-link panel machines, 35-47% of the gap is unexplained (I get 35.5% and 47.4%), and 31-36% on the slow RTX 4090s.
   - Similarly, "the link-to-CPU ratio predicts the oracles' speed on a second card" sits next to a 5-of-18-cell miss (up to 17.6%) on new machines of the *same* card.
6. **Smaller reporting issues found in the raw data.**
   - Spearman 0.89 [0.63, 0.97] over 19 machines (§6.2) is a stale statistic generated before job 109. Over all 28 RTX 5090 machines with MIN-1R at 11% I get 0.84; over the 15 panel + 5 job-109 machines, 0.75. The machine set is not stated.
   - "They still read 1.66-1.68 R*" (§6.1) is true for LA-1R and LA-1R-margin only. Margin reads 1.58 R* and LA 1.75 R* (deployed 1.65-1.66).
   - The engine's MIN-1R reads 1.037 R* (39.7 vs 38.3 per token), so the oracle is not exactly MIN on R*'s trace. R*_eng is defined only in Table 9.
   - The 15-machine panel pools 096a/b (30 problems, a different text, loss 0.207) with 13 machines on the 20-problem own-text corpus (loss 0.190).
   - Intervals over 4-5 machines are t-intervals on possibly skewed per-machine shares.
7. **Format.** The MLSys main text runs about five lines onto page 11, past the 10-page limit. The main text points into a 30-page appendix (Tables 28-30, job numbers) and a 55-page supplement for things a reader needs, such as the fetch table, the publication rule and the probe definition.

## Clarity

Earlier rounds scored clarity 2, then 3 several times. This revision is cleaner, but I would still score it 3.

**What works**

- Table 1 (claim, test and outcome, machines, section) is an excellent map, and its evidence labels are mostly accurate.
- Table 2 plus the systematic names (MIN-1R / MIN-2R / Dep-1R / Few-1R …) remove the code soup of earlier versions. Table 8 maps codes to names for the appendix.
- Fig. 1 makes the two-read path concrete.
- The running example in §5 (10.0 / 13.0 / 20.6 / 15.2 / 13.8 ms, all reproduced) anchors the ratios.
- Section titles are claims, and §8 condenses the paper into five rules.
- Every number is macro-generated, and I found no internal inconsistencies between text, tables and raw data.

**What still makes it hard to read**

1. **Too many baselines at once.**
   - There are five yardsticks: speed vs deployed, share of Eq. (1), share of the gap, capture, and value relative to admit-every-miss. There are also three bounds (Eqs. 1-3), and Fig. 2 plots two of them.
   - The reader has to re-anchor in almost every paragraph. Fig. 5's caption has to open with "Baseline here: *admit every miss*, not the deployed cache", which is a symptom.
   - "The bound we use throughout" is Eq. (1), yet §4's slow-link discussion switches to Eq. (3).
2. **Vocabulary load and an overloaded "panel".**
   - The reader must hold 13 configuration names plus about 15 coined terms (fetch, admit, read ahead, fetch table, publication delay, host-bound, fast/slow link, launch, ran stably, gap, capture, value, cell, R*_eng, hosts B/S/O3-O5/Pa-Pj).
   - "Panel" means 10 machines in Figs. 3 and 5 and in §4's registration sentence ("the 10 machines of the panel's main job"), but 15 in Table 4, Fig. 2 and §6.2.
3. **The abstract is a chain of nested appositives.** "…the two changes together close 35% of the gap, the deployed cache's time beyond the bound, in a test registered before its machines were rented; there, four policies … recover at most 6% of the oracles' gain." It uses undefined terms ("the oracles' gain", "the smallest budget we test") and needs two readings.
4. **The job 112 control paragraph in §4 is the hardest passage in the paper.** It interleaves three arms, a registered threshold, a replay, the fetch table and a double negative ("The control thus failed to give the two-read path MIN's set, so it cannot say … ; it does show …"). A reader cannot tell what was learned. Weakness 3 suggests the honest answer is "nothing conclusive".
5. **Status parentheticals and appendix pointers interrupt the argument.**
   - Paragraph headings carry registration status: "(registered: gain bands per budget on two machines; held at most budgets)" and "(registered: faster than the deployed cache; held on every stable machine, failed on two unsteady ones)".
   - The text cites Tables 28-30 and Appendices H, I and N for basic definitions.
   - Mechanisms the results depend on (the per-machine fetch table, the two-step publication) get one sentence each.
   - It would read better to keep the status in Table 1 and give the mechanisms a short paragraph with Fig. 1.

**Smaller clarity issues**

- Fig. 3's x-axis label is clipped at the right edge ("…grey ticks: panel" with no closing text or parenthesis).
- Fig. 2's second x position mixes two states ("MIN-2R (or Dep-1R)") with two median lines, which takes effort to decode.
- The Table 3 caption is about 10 lines and mixes protocol, bound definitions, pooled bounds and the RTX PRO 6000 numbers.
- Table 6's subscript ± and bracketed trend predictions are dense. A separate "trend − observed" column would help.

## Questions for the authors

1. In job 112, why does Few-2R-early make 11.7 admissions per token against Few-2R's 6.2 (at 11%)? Does the shifted plan re-admit experts the fetch table evicted? Given this, would you relabel the Table 1 row "inconclusive"?
2. Which 19 machines give Spearman 0.89? Do you agree that it drops to about 0.84 with all RTX 5090 machines, and to about 0.75 for panel + job 109?
3. Why does MIN-1R in the engine read 1.037 R*? Is R* computed on the same trace the oracle replays?
4. Does Table 4's panel column change if 096a/b, run on a different 30-problem text, are excluded?
5. Can you show the 2x2 decomposition for Qwen3 at 12.5% and 25%, or on a second engine path, for example without the fetch table?
6. How much of the "pay only together" result depends on the fetch table and on the two-step publication delay? Does it survive with the fetch table disabled for the two-read arms?
7. For §8 rule 2: is there any evidence from another system (FreeToken, KTransformers, DALI) that a better set read twice gains nothing there?

## What would raise my score

- Relabel the job 112 row, and remove or soften the "landing in time is not what holds … back" sentence. Report the admission and read counts of the early arm.
- In the abstract and Table 1, say which registered tests were sign tests and which bands were too wide to fail. Replace "Oracles inside the engine account for the rest of the time" with what was measured and what was attributed.
- Scope §8's rules to the engine and schedule, or add one more engine path or model to the decomposition (Qwen3 would do).
- Refresh the Spearman statistic and state its machine set. Fix the "1.66-1.68 R*" sentence.
- Clarity: one baseline per figure where possible; a single "panel" definition; a plain abstract; the job 112 paragraph rewritten in three sentences.
- Fix the page-limit overflow.

## Scores

| Overall | Soundness | Significance | Novelty | Clarity | Confidence |
|---|---|---|---|---|---|
| 6 (weak accept) | 4 / 5 | 3 / 5 | 3 / 5 | 3 / 5 | 4 / 5 |

The measurements are excellent and fully reproducible, and the registration discipline is a model for the field. The score is held back by the narrow scope (one engine and one model for the central decomposition), by a few conclusions that go beyond what the registered tests could test, and by a still-dense presentation.

---

## Claims checked against raw data

All values come from my own code on `results/*/ec_*.jsonl` (time = mean over problems of decode_ms / n_decode), `st_*.json`, `concur.txt` (B_host = the highest reading of any kind), `fetch_table_law_gptoss.json` (ratio = B_p / B_c), `prof_*.json`, `bs1.jsonl` and `readsched_C*.txt`. R* = 38.315 / 15.340 comes from the job 109 header.

| # | Claim | Location | My value | Verdict |
|---|---|---|---|---|
| 1 | Panel, gpt-oss 11%: MIN-2R 0 [−2,3]; Dep-1R 0 [−2,2]; Both 33 [21,44]; read-ahead 15 [11,19]; left 52 [43,62] | Table 4, §4, Intro | 0.5 [−1.8,2.8]; −0.0 [−1.9,1.8]; 32.6 [21.0,44.3]; 14.9 [10.7,19.1]; 52.5 [42.6,62.3] (15 unique GPUs: 096a/b, 099a-j, 100b, 100f, 101b) | ✓ |
| 2 | Panel 11%, attributed: T_GPU 27, extra reads 21, residual 4 [−6,13] | Table 4 | 27.4; 21.4; 3.7 [−6.0,13.3] | ✓ |
| 3 | Panel 25%: 21 / −3 / 31 / 21 / 48; attributed 34 / 19 / −5 | Table 4, §4 | 20.6 / −3.2 / 31.3 / 20.6 / 48.1; 34.0 / 19.1 / −5.0 | ✓ |
| 4 | New (job 109), 11%: 3 [−1,8] / 0 / 35 [22,48] / 19 / 46; 28 / 18 / 1 | Table 4, abstract | 3.4 [−0.9,7.7] / −0.4 / 34.7 [21.9,47.5] / 19.0 / 46.3; 28.2 / 17.5 / 0.5 | ✓ |
| 5 | New, 25% (4 machines): 20 / −2 / 27 / 24 / 48; 38 / 16 / −5 | Table 4 | 19.8 / −2.4 / 27.3 / 24.4 / 48.3; 37.9 / 15.6 / −5.1 | ✓ |
| 6 | B_host +10% (+22%) moves fast-link shares by ≤5 (9) points for Both, ≤6 (11) for left | Table 4 caption | 5.1 (9.3); 5.8 (10.5) | ✓ |
| 7 | Slow-link panel residual 35-47% of the gap at 11% | §4 | 35.5% (099d), 47.4% (099f) | ✓ |
| 8 | Interaction > 0 on 9 of 10 (job 099) and on 5 of 5 new machines | §4, Table 1 | 099f fails (−0.48 ms); job 109: 5/5 at 11%, 4/4 at 25% | ✓ |
| 9 | Table 6, RTX 5090 rows (MIN-1R, ahead, none, capture), both budgets | Table 6 | e.g. 285K 1.215 / 1.460 / 1.021 / 4.5%; 9800X3D 25%: 1.216 / 1.546 / 1.011 / 2.1%; all cells match | ✓ |
| 10 | Best no-foresight variant 1.007-1.024×; capture ≤ 6% | §6.1, abstract | 1.007-1.024; max 5.7% | ✓ |
| 11 | The four variants "still read 1.66-1.68 R*, as many as the deployed cache" | §6.1 | LA-1R 1.665-1.667, LA-1R-margin 1.675-1.678; but Margin 1.575-1.588 and LA 1.751-1.755 (deployed 1.645-1.660) | partly: holds for 2 of 4 |
| 12 | Margin cuts host reads 4-6% at 11% | §6.1 | 4.3% (job 109) | ✓ |
| 13 | Table 6, RTX 4090 rows (jobs 111, 112) | Table 6 | e.g. 14900KF 0.963 / 1.128 / 1.033 / 25.7%; i5-12400 1.312 / 1.465 / 1.013 / 2.9%; all cells match | ✓ |
| 14 | LA-based variants on slow 4090s 0.64-0.72×; Margin 1.033-1.041×; capture ≤ 26% | §6.1 | 0.644-0.724; 1.033-1.041; 25.7% | ✓ |
| 15 | 4090s within 7% (11%) and 5% (25%) of the frozen trend; no-change misses by up to 46% / 69% | §6.2, §8, Table 1 | max 6.6% / 5.0%; 46.5% / 68.9%. Also the unreported Q1 clause for Dep-1R/MIN-2R (≤0.06): max 0.034, held | ✓ |
| 16 | Trend misses 5 of 18 new-RTX-5090 cells, by up to 18% | §6.2, Table 1 | 5 of 18 (109a MIN-1R ×2, 109d/109e read-ahead ×3); max 17.6% | ✓ |
| 17 | MIN-1R gain vs ratio: Spearman 0.89 [0.63, 0.97], 19 machines | §6.2 | set not identifiable. Pre-109 consumer stable machines (20): 0.90; any 19-machine subset of those: 0.886-0.893; all 28 RTX 5090: 0.84; panel + 109: 0.75 | plausible but stale |
| 18 | Table 5: speeds 1R / 2R / early and misses per token | Table 5 | e.g. 5950X 1.406 / 1.030 / 0.977 and 40.7 / 55.6 / 54.5; 7800X3D 1.208 / 1.027 / 0.995 and 40.9 / 54.7 / 53.0; all match | ✓ |
| 19 | Early arm's misses 0.97-0.98 of Few-2R; fetch table still fetches 17.8-23.7 misses per token | §4 | 0.968-0.980; 17.8-23.7. Unreported: the early arm's admissions are 1.9× Few-2R's (11.7 vs 6.2) and its host reads are +7% | ✓ numbers; conclusion not supported (W3) |
| 20 | T2 "no slowdown" failed | Table 1, App. D | early − 2R: −0.006 to −0.053; fails on 7 of 8 machine-budgets | ✓ |
| 21 | Few-1R beats deployed on all 13 stably-run machines, geometric mean 1.24×; loses on the 2 unsteady slowest | §5, Table 1 | 13 machines (104a/b/c, 105a/b/f, 106e, 107b/d, 112a/b/d/g by GPU); geometric mean 1.244; losers 106c (0.144) and 107e (0.205) | ✓ (stability coincides with link speed) |
| 22 | Few-2R 1.11-1.22× at ratios 0.28-0.32 | §6.2 | 1.112-1.217 (104a, 105b, 105e) | ✓ |
| 23 | Deployed at 31-54% of Eq. (1) on 25 consumer machines; 40-54% of Eq. (3); server 12-46%; 5 new machines 41-48% | §3, abstract, Table 1 | 30.6/31.2-54.0% (25 unique RTX 5090 GPUs, jobs 093-108); 40.5-54.0%; 12.4-46.1%; 41.5-48.0%. Low end is only the Threadripper 9960X | ✓ |
| 24 | Eq. (3)/Eq. (1) is 1.00 on the median consumer machine, up to 1.56 | §3 | median 1.000, max 1.557 (TR 9960X) | ✓ |
| 25 | Microbenchmark reaches 86-96% of the bound's read time per layer | §3, Table 30 | layer mode 0.861-0.963 (token mode 0.93-0.97); the registered threshold was only ≥0.50 | ✓ |
| 26 | Running example (9950X, 11%): Eq. (1) 10.0, Eq. (2) 13.0, deployed 20.6, MIN-1R 15.2, read-ahead 13.8 ms | §5 | 10.02, 12.96, 20.64, 15.16, 13.81 (096a) | ✓ |
| 27 | Table 3: absolute speeds on hosts S and B; ours ÷ FreeToken 1.294 etc.; bound 172 / 140 tok/s | Table 3 | host S, all 18 speeds exact (e.g. ours 57.9 / 94.1 / 133.6 / 33.5 / 53.9 / 95.8); host B gpt-oss + Qwen3 12.5% exact; 69.88 / 53.99 = 1.294; 5.80 ms → 172, 7.12 ms → 140 tok/s | ✓ |
| 28 | Host S ratios outside ±0.06 of host B's at 5 of 6 cells; ours faster at 11 of 12 | Table 1 | differences −0.087, −0.079, −0.061, −0.005, −0.105, −0.075; Qwen3 43.75% on S: 0.974 | ✓ |
| 29 | Same-CPU panel machines differ by up to 29%; relaunches within 3.2% (5090) and 5.5% (4090) | §2 | 285K 18.14 vs 14.06 ms (+29.0%); 105b→106a +3.2%; 110c→111g +5.4% | ✓ |
| 30 | Second probe within 4.9% on 4 machines; ratio moved by up to 12% between rentals | §10, App. D | +0.8, +4.9, −0.4, −0.3%; 109f→112g ratio 0.837→0.738 (−11.8%) | ✓ |
| 31 | 16-token window recovers 0.78-0.93 of the gain in time at 11%; deployed-path W=16 0.89-1.32× on 4 machines, below 1 on the 2 lowest ratios | §7 | 0.781-0.935 (099 panel); 0.891-1.320 (100a/b/c/f), below 1 on 100a and 100f | ✓ |
| 32 | T_GPU = 2.9 ms, the smallest Nsight profile | §3, Table 4 | 2.940 ms (105e, C14) among the base profiles; RTX 5090 median about 3.2; RTX 4090 3.4-3.6 | ✓ |
| 33 | Every commit preceded its rental by ≥2.7 s; post-launch header edits disclosed | App. D | min lead 2.71 s (105f); post-launch edits in jobs 100 (+prediction 9), 102, 105, 106 (machine substitutions), 107 and 111 (amendments): all disclosed | ✓ |
| 34 | Read-ahead oracle reads 1.22-1.31 R* at 11% | Table 2 | 1.218-1.236 (RTX 5090, job 109); 1.268-1.289 (RTX 4090) | ✓ |

**Not checkable from the raw rows:**

- the trace-based foresight results (0.65 C, the forecasters, next-token ≤0.31), because the routing traces are not in `results/`;
- the published-systems audit (median 13.6%);
- the KL parity values.

**Registered tests against job headers.** Job 109 P1, P2, P3 and P7; jobs 110/111/112 Q1-Q6 and T1-T7; job 102 P2; job 089 P2; and job 081 P2-P4 all match how the paper states them.

- Table 1's "pre-specified ratios on host B" is slightly inaccurate: job 081 registered orderings (paired CI > 1, ≥1.8× llama.cpp), not ratios.
- Job 110 is void as registered. Job 111 reran job 110's unchanged predictions after a corrected loss reference. That reference was taken from job 110's own first round, which the paper discloses.

## IEEE format (`paper_ieee.pdf`, skimmed)

1. **Run-in paragraph headings get double punctuation.** IEEEtran appends a colon to `\paragraph`, so headings that already end in a period or parenthesis render as "a) Contributions.:", "b) The ordered bound: reads foresight cannot free.:", "…(registered: 50% or more per layer; held).:".
2. **Appendix cross-references read "Section D", "Section H", "Section J"** (e.g. the Table I caption "Section D scores every one"). Under `\appendices` they should read "Appendix D".
3. **Length and appendices.**
   - The main text plus references runs to page 12, followed by about 27 pages of single-column appendices after `\onecolumn` (39 pages in all).
   - IEEE conference proceedings normally have a fixed page budget, often 6-10 pages including references, and do not carry long single-column appendices. The appendices should go to a separate supplement.
4. **Float placement.**
   - Pages 5 and 6 are almost entirely double-column floats (Table III + Fig. 2; Table IV + Fig. 3) with only a few lines of text each, and page 8 is about two-thirds floats.
   - Table III is first cited in Section II (page 2) but sits on page 5.
   - Table III is shrunk with `\resizebox`, so its font falls below IEEE's minimum readable size, about 8 pt.
5. **Captions.** "TABLE I" with the caption above, and "Fig. n." below, are correct. But several table captions are 6-10 lines of method notes (Tables III, IV and VI). IEEE style expects short captions, with the notes as table footnotes.
6. **References.** These are numbered and in order of citation (IEEEtranN), which is correct. However:
   - several entries truncate after two authors with "et al." (e.g. [4] "L. Xue, Y. Fu et al.", [5], [6], [11]-[13]); IEEE uses "et al." only for more than six authors;
   - [63] gives surnames without initials ("Zhang, Gao, and Mitra");
   - [38] has no author;
   - [40]-[43] use GitHub handles as authors.
7. **Front matter.** The Index Terms are present but not alphabetical, and capitalisation is inconsistent ("Mixture of Experts" capitalised, the others lower case). The abstract (about 230 words) contains inline math ("0.65 C", "C experts"), which IEEE discourages in abstracts. The anonymous author block uses `\IEEEauthorblockN/A` correctly.
