# Review 19b: "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth"

Reviewer role: professor in a systems group that works on scientific benchmarking (Hoefler and Belli, SC'15). This is written as a fellowship and workshop referee report.
Date: 7 October 2026.

## Materials read

- `paper/paper.pdf` (37 pages). I read the main text (Sections 1–10) in full and Appendices A–E, H and I in full. I read Appendices F, G, J and K in part, and Appendix L in full. I also rendered pages 1–8 to check the figures and tables.
- LaTeX sources: `paper/paper.tex` (Sections 2–5), `app_relation.tex`, `app_more.tex`, parts of `app_wsg.tex` (Appendices J, K and L), `tab_dm.tex`, and a few generated macro files (`wsg_decomp.tex`, `wsg_robust.tex`, `wsg_pending.tex`), read to match macro names to scripts.
- `paper/supplement.pdf`: the opening section and the headers of the scorecard tables. I did not read the clause-by-clause rows in full.
- Analysis scripts, read only to learn definitions (which states, which files, which constants): `scripts/decomp_measured.py`, `scripts/fig_decomp.py` (the machine-selection part), `scripts/sumlaw_paper.py` (`profiles()`), `scripts/reanalysis.py` (`features()`), `scripts/headline_stats.py`, the headers of `scripts/robustness.py` and `scripts/wsg_numbers2.py`, and `bandwidths()` in `gpu-branch/jobs/ec2/fetch_table.py`.
- `prereg/reanalysis_hosts.json` (the launch list, GPU UUIDs and link-ratio values; its times matched my raw recomputation exactly), `prereg/speed_limit_v2.json` (R★ only), `prereg/gpu_pushes.json` (GitHub push records), `gpu/vast_ledger.json`, and the "Reproduce" section of `README.md`.
- In the `gpu-branch` checkout: the job-script headers for jobs 090, 096, 099, 102, 104 and 105 (prediction blocks), `results/<job>/` for every job from 093 to 108 (`ec_*.jsonl`, `st_*.json`, `concur.txt`, `g_prof.json`, `prof_*.json`, `readsched_*.txt`, `bs1.jsonl`, `parity_kl.json`, `free.txt`, `lscpu.txt`, `v0.txt`, `stdout.log`, `manifest.json`), and `git log --format='%h %at %ct'` / `git diff` on `jobs/` (file contents only).

## Independence statement

- **What I did not open.** Nothing under `reports/` (apart from writing this file), no `prereg/*outcome*.md`, nothing in `research_notes/`, no file named like a review, number check, plan or progress log, not `paper/paper_v1_prereview.tex`, and nothing in `apply/`. I read no commit messages. All `git` calls printed hashes and timestamps only, or file diffs.
- **My own code.** I wrote every check myself, in a new folder (`scratchpad/rev19b/`: `raw.py`, `decomp.py`, `decomp2.py`, `bound25.py`, `headline.py`, `window.py`, `plan.py`, `plan2.py`, `relaunch.py`, `spear.py`, `varcomp.py`, `implied.py`, `regtime.py`, `pushcheck.py`). My checks parse the raw per-problem rows, counters, probe output and profiles, and use my own bootstrap seeds.
- **What I took from the authors.** I did not recompute two inputs, which I took from the authors' files:
  - R★, MIN's reads per token (38.315 at C=14 and 15.340 at C=32 for gpt-oss), from `prereg/speed_limit_v2.json`. Recomputing it needs the routing traces.
  - The link-to-CPU ratio. I took it from `reanalysis_hosts.json`. I also recomputed it from `concur.txt` with a "best over best" definition, and it puts no machine on the other side of the 0.5 split.
- **What I left unchecked.** I did not recheck the 9-model horizon rule (Section 6), the published-systems audit (13.6% and 27%), or the supplement's earlier analyses.
- **Scope.** I modified nothing in either repository and committed nothing.

## Summary

- **Setting.** The paper studies batch-1 decode of two MoE models (gpt-oss-120b in MXFP4 and Qwen3-30B-A3B in BF16) on rented RTX 5090 machines, with most experts kept in host DRAM. Its engine is a 3,500-line llama.cpp expert-cache patch.
- **Bound (Section 3).** It defines a bound, Eq. (1): MIN-with-bypass host reads at the machine's highest probed read rate. Two tighter variants follow: one for systems that read on demand (Eq. 2) and one that respects layer order (Eq. 3). A microbenchmark reaches 86–96% of the bound's read time. The authors' cache runs at 31–54% of Eq. (1) at gpt-oss 11% on 25 consumer machines, and leads FreeToken at 11 of 12 configurations on two machines.
- **Decomposition (Section 4).** Oracles built into the engine split the gap between the deployed cache and Eq. (1):
  - Caching MIN's set while still reading each admission twice closes about 1%.
  - Reading once while keeping the deployed set closes about 1%.
  - Doing both closes 39% on the 13 machines whose link reads at least half as fast as their CPU, a subset drawn after the data. On all 15 machines it closes 33%.
  - Copying ahead closes a further 15%.
  - The remaining 47% is attributed to the GPU's non-expert time in series with the reads (27%) and to the prefetching oracle's reads beyond MIN's (22%).
- **How foresight must be spent (Section 5).** Read each admitted expert once. Among hit-optimal schedules, prefer the one with the fewest admissions (1.21× over the deployed cache on 9 stable machines). Choose the read path by the link-to-CPU ratio (Spearman 0.89 over 19 machines).
- **How much foresight is needed (Section 6).** On trace replays, the value of a window of future routing is priced: half of MIN's gain over admitting every miss needs about 0.65 C distinct experts of lookahead, across 9 models.
- **The record.** Every GPU job's predictions sit in the header of its job script, committed before the rental. 1,483 clauses are scored, failures included, and every number is generated by a script.

## Strengths

1. **The numbers are what the raw data say.** I recomputed 36 quantitative or methodological claims from raw per-problem rows, counters, probes and profiles (table below). Every one reproduces, most of them exactly:
   - all sixteen cells of Table 3, with their bootstrap intervals;
   - all ten rows of Table 4 that I could locate, to the third decimal of the ratios and their intervals;
   - the running example;
   - the fewest-admission geometric mean and its interval;
   - the Spearman correlation;
   - the admission-margin gain;
   - the window shares.

   I found no fabricated, mis-transcribed or cherry-picked number. In my experience this level of agreement between a paper and its raw data is rare.
2. **The registration and validity record is real and checkable.**
   - **Timing.** For all 71 rentals of jobs 093–108, the job-script commit precedes the rental's creation, by at least 2.71 s. The GitHub push record is within 1 s of rental creation for 7 rentals and up to 0.8 s after it for 4, and every machine started its job at least 26 s after the push. Those are exactly the paper's figures.
   - **Validity gates.** The gates run as registered, and their outcomes are logged in `v0.txt` and `stdout.log`:
     - job 107: 7 rented, 3 stopped at the gate, 2 failed the round check (spreads 3.54% and 5.98%), 2 valid;
     - job 108: 6 rented, 2 stopped at the gate (one GPU already rented, one host with 323 GB of memory in use), 1 failed the round check (7.01%), 3 valid.
   - **Failed hosts.** Hosts that failed are reported, not dropped silently. For example, the EPYC 7302 in job 106 computed wrong outputs (teacher-forced loss 0.21–0.45 nats against 0.19), and the Xeon's rounds were 52% apart. Both appear in the appendix tables, and the fewest-admission result is also reported "counting the unsteady machines" (1.17× over 13, with two losses).
3. **The statistical unit is the right one and is stated.** Section 2 argues from data that machines differ far more than problems; I reproduced both halves of that argument:
   - two panel machines with the same CPU model differ by 29%, while relaunches of the same machine stay within 3.2% (worst case);
   - machines carry 95–96% of the variance of the single-read gain in a two-way decomposition.

   Summaries over machines are bootstrapped over machines. Table 4 states that its intervals are over problems within one launch, and launch-to-launch drift there is ≤1.3%.
4. **Bounds are used the way Rule 11 intends.**
   - Eq. (1) is a clean, assumption-scoped lower bound, and nothing measured beats it: the best state of each of the 15 decomposition machines is at least 1.33× Eq. (1) time at 11% and at least 1.54× at 25%.
   - The microbenchmark shows the read term is nearly reachable (layer mode 86–96%, which I reproduced from `readsched_*.txt`).
   - The worry that a CPU could re-read experts from its last-level cache is answered with MIN's shortest re-read distance (0.5 GB). I also tested whether the X3D parts read faster than the probe suggests, and they do not: the median implied rate relative to B_host is 0.991 on X3D parts and 0.993 on the others.
5. **Building oracles into the real engine is a valuable methodological contribution.** Changing one mechanism at a time inside the deployed engine, with teacher-forced routing so that every configuration sees identical text, is the right experimental design for "where does the time go". Counter-level agreement between engine and replay (copies 0.597–0.726 of the greedy schedule's) makes the oracle states interpretable.
6. **Honesty about failure.** The closed-form account failed three registered tests and one was inconclusive. The paper keeps it, labels it exploratory, and says the measured decomposition does not depend on it. Table 8 adds up: 565 = 218 + 193 + 142 + 12, and the per-job totals for 099–108 match the text.

## Weaknesses (most important first)

### W1. The title result is a post-hoc average of a quantity that depends on the machine, and its "what is left" split is an attribution, not a measurement

- **Post hoc throughout.** Section 4 and Table 3 carry the paper's title. Only the sign of the 2×2 interaction was registered (job 099, prediction 5). Everything else in Section 4 is post hoc:
  - the shares;
  - the 0.5 cut on the link-to-CPU ratio;
  - the "left" split.

  `scripts/decomp_measured.py` was first committed at 19:18 UTC on 7 October, about ten hours after the last rental of job 108 ended. The paper flags the subset, but the abstract still leads with the subset number (39%).
- **The mean hides a strong dependence on the machine, even inside the subset.** Within the 13 "fast-link" machines:
  - The "together" share ranges from 14% to 56% at 11% and rises with the link ratio (Spearman ρ = 0.83; 0.89 over all 15).
  - The residual is not noise. It correlates with the ratio (ρ = −0.73, p = 0.005 at 11%; ρ = −0.80, p = 0.001 at 25%). Per machine it runs from −13% to +12% of the gap at 11% and from −21% to +2% at 25%.
  - The two "named parts" therefore over-account on fast-link machines and under-account on slower ones. On O4 at 25%, the prefetching oracle runs at 0.991× Eq. (2), so the full T_GPU it is charged as "in series" cannot all be in series there.
- **T_GPU is assumed, not measured on these machines.** It is the smallest non-expert time among 14 Nsight profiles (2.94 ms; the range is 2.94–3.38 ms), and none of those profiles was taken on any of the 15 decomposition machines. The table caption gives the profile range (27–31% of the gap), but that range is not carried into the intervals.
- **The abstract overstates the evidence.** It says "Oracles built into the engine then account for the rest of its time" and "Most of the rest is the GPU's own work". The oracles remove 54% of the gap. The rest is an accounting with a structured residual.
- **What I ask for.**
  - Present the decomposition as a function of the link ratio: a per-machine table and a regression with intervals, as Fig. 4 already does for speed-ups.
  - Profile the decomposition machines, or propagate the profile range.
  - Reword the abstract so that measured removal and attribution are distinguishable.

### W2. The 2×2 does not cleanly separate "what is cached" from "how it is read"

- **The "MIN's set, still read twice" arm does not realise MIN's set at 11%.** From the counters (identical across panel hosts to 0.1), it makes 50.8 misses and 15.8 admissions per token. That is 66.6 host reads, more than the deployed cache (63.1) and 68% more than MIN read once (39.7). Its misses are 28% above the one-read arm's, because the paced background path (one copier thread, 16 MB pieces, a 64-copy cap) lands admissions late.
- **So "pay only together" partly describes this engine's copy path.** The paper says the arm "lands late", but the table row is still labelled "MIN's set alone". The finding that the two changes pay only together is therefore partly a statement about how this engine implements background copies, not about caching versus reading in general.
- **The registered prediction was weaker than Table 1's claim.** Job 099 prediction 5 was:
  - a positive interaction at both budgets on every host (held 9/10 at 11%, failing on the 0.29-ratio host, and 10/10 at 25%);
  - a "one read alone" effect of at most 5% (held 9/10 and 8/10).

  "MIN's set alone closes about nothing" was not registered, and it holds only at 11% (at 25% that arm closes 21%). Table 1's "registered on the panel, held on 9 of 10" is accurate for the sign at 11% only.
- **What I ask for.**
  - Report achieved misses and reads for each arm against R★.
  - Rename the arm after what it does.
  - If feasible, add a two-read arm whose background copies are timed by the oracle to land before reuse, so that "second read" can be separated from "late copy".

### W3. The bound's denominator is a single probe run taken under background load

- **How B_host is measured.** B_host is the highest single reading of one run of `concur`. In the panel script (job 099, lines 84–117), the probe runs while the gpt-oss download, the Qwen3 download and a torch/pip install proceed in the background. The paper's Limitations section says this.
- **The load biases the probe low, and the data quantify by how much.** Three GPUs were re-probed with no download in job 102:

  | GPU | Probe during download | Quiet probe (job 102) | Difference |
  |---|---|---|---|
  | i9-13900KF | 71.7 (099d), 70.6 (100a) | 72.5 | +1.1% / +2.7% |
  | 9950X, O4 | 50.7 (096a) | 52.6 | +3.7% |
  | 9800X3D | 51.0 (101b) | 53.4 | +4.7% |

  A low B_host makes Eq. (1) too slow, which inflates every "share of the bound" (31–54%) and shrinks the gap by a few points.
- **On some machines Eq. (1) may not be a bound.** By the closed-form account, the engine's reads imply a rate above the probe's best by more than 5% on roughly 12 of 84 launch-budgets. I get 11 of 93 rows, up to 1.22× on host 100f, which is the machine at the top of the 31–54% range.
- **"Highest single reading" is an order statistic of noisy, mixed-mode readings.** The appendix says the median reading would lower the limits by 8%.
- **What I ask for.**
  - Report the probe-under-load bias, with its sign, wherever a share of the bound is quoted.
  - Use a quiet probe where one exists.
  - Prefer a robust estimator (for example the best of repeated concurrent runs) over the maximum across modes.

### W4. Registration is an exemplary record but a weak confirmatory design

- **What the 1,483 clauses look like.** Across 39 jobs:

  | Outcome | Clauses | Share |
  |---|---|---|
  | Held with a 95% interval on the predicted side | 495 | 33% |
  | Held on the point estimate only | 597 | 40% |
  | Failed | 308 | 21% |
  | Untested or void | 83 | 6% |

  Of the clauses held on the point estimate in the hand-scored set, 213 have no interval at all.
- **What the design lacks.** There are no designated primary endpoints and no multiplicity accounting. Bands were often set from pilot data on the same machines; the panel's bands came from four earlier hosts, none of them link-starved, and failed on the two that were.
- **The headline numbers were not what was registered.** Most of the headline numbers are post hoc or derived:
  - 39% (the subset share);
  - 31–54% (shares of the bound);
  - the 0.32 crossover;
  - 0.65 C (the horizon rule).
- **Mid-job edits.**
  - Two prediction-relevant amendments were made after some machines had started: job 100's ninth prediction and job 107's replacement rules. Both are disclosed.
  - The headers of jobs 102, 105 and 106 were edited after some of their machines had started (+10 s, +229 s and +14 s). These are host substitutions; the predictions are unchanged, and the 105 and 106 headers say so. Appendix D does not mention them.
- **This is fine as a lab notebook, and it is far more than the field does.** But Table 1's word "registered" sometimes stands for a weaker registered form (W2), and readers will read it as confirmation.
- **What I ask for.** For each Table 1 row, quote the registered clause, its scoring and its interval status. Separate "registered sign" from "registered magnitude".

### W5. Scope and external validity are narrower than the abstract's phrasing

- **The test bed is narrow.** One GPU model, batch 1, two models, and AIME prompts that were also used during development. Teacher-forced replay is good for internal validity, but it is not free generation.
- **The machine classes were shaped by the results.**
  - The consumer/server split was drawn after the closed-form account failed on servers.
  - The four machines excluded from the fewest-admission headline as "unsteady" are all servers, and they include both of its losses. The exclusion follows the registered gate, so this is legitimate, but the claim's scope is "machines that pass our gate", and passing the gate correlates with being a desktop.
- **The system comparison is thin.** It rests on two machines (plus one tuned-FreeToken run) with intervals over problems within one launch. Each system decodes its own text: 0 of 30 outputs are identical between our cache and FreeToken. Routing therefore differs between the compared runs. That is acceptable for an end-to-end comparison, but it is not a controlled one.
- **Parity was measured on a different platform.** The KL parity (0.0019 nats) comes from an A100 next to a Xeon 6960P, not an RTX 5090 host. The teacher-forced loss is, however, consistent (0.189–0.191 nats) across all 15 decomposition machines and all five states, which partly mitigates this.

### W6. Section 6 prices foresight in reads on replays, against a different baseline

- **Different yardstick.** Section 6's value of a window is a share of MIN's gain over "admit every miss", measured mostly in host reads on trace replays. The engine markers are lower than the replay values at gpt-oss 11%:
  - 8 tokens at recall 0.5: 0.20–0.26 in the engine against 0.29 in the replay;
  - 1 token: 0.14–0.17 against 0.18.
- **Different baseline.** In time, admit-every-miss is up to 2.2× slower than the deployed cache's one-read variant on slow-link machines (aa/foa = 2.23 on host 099f), so the baseline also changes meaning across machines.
- **Unrealistic errors.** Degraded windows draw independent errors.
- **Too much weight in the abstract.** The 9-model rule is exploratory and replay-only, but it gets the abstract's last sentence with the same weight as measured results.

### W7. The artifact pipeline is incomplete in places

- **Missing scripts.** The README's "Reproduce" pipeline runs six scripts. None of them generates Section 4, Appendix H or the job 104–108 macros: `decomp_measured.py`, `fig_decomp.py`, `robustness.py`, `job10x.py` and `panel_099.py` are not in the pipeline.
- **An inaccurate claim.** Appendix L says `wsg_tables.py` and `wsg_numbers2.py` "generate every table, figure and number".
- **A stale docstring.** The docstring of `decomp_measured.py` says T_GPU is the median profile, while the code (and the paper) use the smallest.
- These are small fixes, but they matter for a paper whose fourth contribution is "an open record".

## Clarity

Earlier rounds scored clarity 2/5 and then 3/5 five times. The restructuring helps, and the main text is now navigable by an expert. It is still hard to read in specific, fixable ways.

**What works**

- **Structure.** Section titles are questions ("How fast could it be?", "Where the seconds go", "How foresight must be spent", "How much foresight is needed"). Section 2 has a "Yardsticks" paragraph that names the one measure each section uses. This is the single best change.
- **Vocabulary.** Fig. 1 together with the plain words "read once / read twice / late / ahead" makes the core mechanism understandable without the appendix. Table 2's "Reads" column is excellent.
- **Running example.** Section 5's example (10.0 / 13.0 / 20.6 / 15.2 / 13.8 ms) anchors every ratio that follows. More sections should have one.
- **Claim map.** Table 1 is the right device, and the paragraph-level labels in Section 5, such as "(registered)", "(exploratory)" and "(registered, partly held)", show readers what kind of evidence they are reading.
- **Limitations.** The section is specific and quantitative.

**What still makes it hard to read**

1. **The abstract is a list of numbers.** It contains 11 numbers, three parenthetical qualifiers and two post-hoc caveats, and it uses terms the reader cannot yet parse:
   - "relative to that probe";
   - "the gap", given as an appositive with no definition;
   - "0.65 C distinct experts per layer ahead";
   - "(0.66–0.81 C on our own)".

   A reader cannot tell from it which result is measured, which is registered and which is a model.
2. **The same numbers recur without adding information.** 39%, 31–54%, 0.65 C and 15% each appear in the abstract, the bolded paragraphs of the introduction, the contributions list, Table 1, the section itself, Section 7 and the conclusion: five to seven times each. That space should go to definitions (what "foresight", "window", "share of MIN's gain" and "host-bound" mean) before first use.
3. **One configuration has many names.** The best oracle is called:
   - "MIN, prefetched" (Fig. 3, Table 2);
   - "copies issued a few steps early" (Table 3);
   - "and read ahead" (Fig. 2);
   - "MIN read once and ahead" (Section 5);
   - "the prefetching oracle" (Section 4);
   - `both3p` and "both, paced" (appendix);
   - "MIN fewest admissions" vs "MIN greedy" (Fig. 4).

   Appendix C maps the codes, but the main text and its own figures should use one name per configuration.
4. **Four yardsticks, two baselines.** Section 5 is relative to the deployed cache, while Section 6 switches to "admit every miss", a different online policy whose time behaviour depends on the link. Figure 5's y-axis ("share of MIN's gain over admit every miss") is unreadable without that context. Say in the Section 6 lead why the baseline changes, and what the numbers would be against the deployed cache.
5. **The rate definitions are loose and slightly inconsistent.**
   - Section 2 defines B_c, B_p and B_host, but the link-to-CPU ratio uses B_c interpolated at the helper count and B_p from one zero-copy reading, while Eq. (3) says "best rate".
   - Table 4's caption reads "CPU 72, link 53, together 78, highest 88", where "highest" exceeds "together". The reader has to learn from the code that "together" is a median of concurrent sums and "highest" is a maximum.
6. **Table captions do the work of paragraphs.**
   - Table 3's caption (about 90 words) must be parsed to understand the row arithmetic: "together, ahead and left sum to the gap, and the last three rows sum to left".
   - Table 4's caption (about 140 words) carries bound definitions, backend selection, a pooled-bound aside, an all-VRAM reference and a footnote.
   - Figure 2's legend ("median: MIN's set first (read twice)", "median: one read first (deployed set)") and its two-marker "one change alone" category are hard to decode at print size.
7. **Labels are uneven within Section 4.** Unlike Section 5, the paragraph headings of the title section carry no evidence labels ("What is cached and how it is read pay only together.", "Reading ahead.", "What is left."), even though their content mixes a registered sign, post-hoc shares and an assumed attribution. Readers have to go back to Table 1.
8. **The main text leans on the appendix for key claims.** It cites Tables 20 and 24–26 and Appendices H, I and J for evidence it does not summarise. Appendix D's "Failures" paragraph is a single run-on block of about 30 lines of clause outcomes, and the appendix's host nomenclature (O1–O6, Pa–Pj, "Pf of job 099", "105f", "the Threadripper 9960X of job 105") reads as a lab notebook. A fellowship committee member will not get through Appendices D–I.
9. **Some sentences mix sources.** "The fewest-admission schedule makes 0.60–0.73 of the greedy one's copies for the same hits in replay (in the engine it misses 2.5–5.3% more)": the 0.60–0.73 is measured in the engine, not in replay (I recomputed it from the engine's counters). "A machine rented again came within 3.2%" is the worst case over nine relaunched machines, so it should be plural.
10. **An exploratory model in the title section.** The closed-form account paragraph in Section 4 gives a failed exploratory model's statistics (0.88–1.16 over 39 launches) in the title section. It is honest, but it costs the reader attention and adds a fourth "Eq." to track. It belongs in the appendix, with a one-sentence pointer.

On balance, clarity has improved from a 2 to a solid 3. It is not yet a 4, because a careful non-specialist still cannot extract from the main text alone what was measured, what was assumed and what was registered.

## Claims checked against raw data

"Raw" means `ec_*.jsonl` rows (decode_ms/n_decode, nll_sum/nll_n), `st_*.json` counters, `concur.txt`, `prof_*.json`/`g_prof.json`, `readsched_*.txt`, `bs1.jsonl`, the ledger and git timestamps. ✓ means reproduced; ≈ means reproduced to within rounding or a stated difference in the set of rows.

| # | Claim (location) | Paper | My recomputation | Verdict |
|---|---|---|---|---|
| 1 | Table 3, gpt-oss 11%, all eight rows with intervals over 13 machines | 1 [−1,3]; 1 [0,2]; 39 [32,45]; 15 [11,18]; 47 [42,51]; 27 [24,29]; 22 [20,24]; −2 [−6,2] | 1.2 [−0.7,3.0]; 0.9 [−0.2,2.0]; 38.8 [31.9,45.3]; 14.6 [10.6,18.3]; 46.5 [42.1,51.1]; 26.9 [24.0,29.5]; 21.8 [20.2,23.7]; −2.2 [−6.4,2.5] | ✓ |
| 2 | Table 3, gpt-oss 25% | 21 [19,22]; −1 [−4,1]; 35 [30,40]; 20 [18,22]; 44 [40,48]; 33 [30,36]; 20 [19,21]; −9 [−13,−5] | 20.8; −1.1; 35.3; 20.5; 44.3; 33.4; 20.0; −9.1, all intervals within 0.5 points | ✓ |
| 3 | Abstract: "33% together on 15 machines" | 33% | 32.6% [21.6, 42.0] | ✓ |
| 4 | Best oracle closes a mean 54% (49–58) on 13 machines; at most 17% on the 2 slow-link machines | 54 (49–58); ≤17 | 53.8 [49.4, 58.1]; 16.8 and 1.2 | ✓ |
| 5 | All three changes: time 1.07–1.38× Eq. (2) at 11% | 1.07–1.38 | 1.07–1.38 | ✓ |
| 6 | Prefetching oracle reads 1.22–1.31× R★ at 11% (1.42–1.43 at 25%) | as stated | panel 1.22–1.31 / 1.42–1.43; over the 13 machines 1.42–1.44 at 25% | ≈ (computed on the panel, not the 13) |
| 7 | Registered interaction (job 099 prediction 5), "held on 9 of 10" | 9/10 | interaction > 0: 9/10 at 11% (fails on 099f), 10/10 at 25%; one read alone ≤5%: 9/10 and 8/10 | ✓ (the registered form covers both budgets; see W2) |
| 8 | 31–54% of Eq. (1), 55–70% of Eq. (2), 25 consumer machines, gpt-oss 11% | 31–54; 55–70 | 31.2–54.0; 55.2–70.3 (25 consumer and 3 server machines, same selection rule) | ✓ |
| 9 | T_GPU = 2.9 ms, the smallest Nsight profile | 2.9 | non-expert 2.94–3.38 ms over 14 profiles; G 4.07–4.69 ms | ✓ |
| 10 | Table 4, host S, six rows (tok/s, ratio and 95% interval) | e.g. 57.9 / 48.0 / 28.9; 1.207 [1.195, 1.219] | identical in every cell, including 0.974 [0.962, 0.987] at Qwen3 43.75% | ✓ |
| 11 | Table 4, host B, gpt-oss 11/25/40% and Qwen3 12.5% | e.g. 1.294 [1.278, 1.312] | identical | ✓ |
| 12 | Leads FreeToken at 11 of 12; 2.0–4.0× llama.cpp | 11/12; 2.0–4.0 | 11/12 (one loss at host S, Qwen3 43.75%); 2.00–4.02 | ✓ |
| 13 | Table 4 bound: host B 172 tok/s, host S 140 | 172; 140 | B_host 87.5 → 172.3; 71.3 → 140.5 | ✓ |
| 14 | Rule 3: ratios of total time agree with ratios of mean rates within 0.003 | ≤0.003 | max difference 0.002 | ✓ |
| 15 | Running example (Ryzen 9 9950X, O4) | 10.0 / 13.0 / 20.6 / 15.2 / 13.8 ms | 10.02 / 12.96 / 20.64 / 15.16 / 13.81 | ✓ |
| 16 | MIN read once beats the deployed cache by 16–51% at all budgets on O3–O5; up to 81% read ahead | 16–51; 81 | 1.160–1.514; 1.812 | ✓ |
| 17 | No bypass, or two reads, at the smallest budgets "gain at most 15% or lose" | ≤15% | maximum 1.148 (nb2, Qwen3 12.5%, O3) | ✓ |
| 18 | Fewest-admission schedule: 9 stable machines, geometric mean 1.21× [1.13, 1.30]; 1.17× over 13; 2 losses, the slowest links | as stated | 1.215 [1.128, 1.301]; 1.166; losses 0.845 (ratio 0.13), 0.974 (0.22) | ✓ |
| 19 | Fewest-admission copies 0.60–0.73 of greedy, misses +2.5–5.3% | as stated | 0.597–0.726; +2.5% / +5.3% (engine counters) | ✓ (measured in the engine, not in replay) |
| 20 | Ratio 0.28–0.32: greedy single read 0.85–0.93×; fewest-admission served by the CPU 1.11–1.22× | as stated | 0.853–0.936; 1.112–1.217 | ✓ |
| 21 | Spearman 0.89 [0.63, 0.97], 19 machines | 0.89 [0.63, 0.97] | 0.88–0.89 [0.62, 0.97] (depending on how relaunches are merged) | ✓ |
| 22 | Machines carry 96% of the variance, problems ≤3% | 96 / ≤3 | panel two-way: 95.4 / 1.1 (11%), 96.1 / 2.7 (25%) | ≈ |
| 23 | Admission margin: 1.021× / 1.018× on 5 stable machines; reads −4–6% at 11% | as stated | 1.0215 / 1.0177; −4.3 to −5.8% | ✓ |
| 24 | Layer-ahead copy 1.04× where the link ≈ the CPU, down to 0.72× | 1.04; 0.72 | 1.042 (105f); 0.722 (105e) | ✓ |
| 25 | Exact 16-token window recovers 0.78–0.93 at 11%; registered ≥0.80, missed by the lowest machine | as stated | 0.78–0.93; 099f = 0.78 | ✓ |
| 26 | Same CPU model differs by up to 29%; relaunch within 3.2% | 29; 3.2 | 29.0% (285K pair); worst relaunch 3.2% (TR 9960X), others ≤2.8% | ✓ |
| 27 | Registration timing over 71 rentals | ≥2.7 s; 7 within 1 s; 4 pushes ≤0.8 s after; start ≥26 s after the push | 2.71 s; 7; 4 (0.8, 0.5, 0.0, 0.0 s); 26 s (100f) | ✓ |
| 28 | Amendments after launch | job 100 (ninth prediction), job 107 | as stated; jobs 102, 105 and 106 also changed headers after some of their machines started (host substitutions, predictions unchanged), not mentioned in Appendix D | ✓ (minor omission) |
| 29 | Job 107 gate: 7 rented, 3 at the gate, 2 failed the round check, 2 valid; spreads 3.5% and 6.0%; 50–96 GB in use | as stated | identical (`v0.txt`, `stdout.log`) | ✓ |
| 30 | Job 108: 6 rented, 2 at the gate, 1 round failure, 3 valid; EPYC 7K62 7.0%; 323 GB; GPU rented before (= 100b) | as stated | identical | ✓ |
| 31 | Job 106: EPYC 7302 loss 0.21–0.45 nats against 0.19 / 0.085; Xeon rounds 52% apart; 131–132 GB in use | as stated | 0.208–0.446; gpt-oss 0.189–0.190, Qwen3 0.0855; 14.52 vs 22.09 ms; 132 / 131 GB | ✓ |
| 32 | Probe's second-highest reading a median 0.8% below the highest; one launch more than 5% below (107d, 26%) | as stated | median 0.82%; 1 (107d, 26.0%) | ✓ |
| 33 | Implied read rate 0.88–1.16 of B_host at 11%; more than 5% above on 12 of 84 launch-budgets, up to 1.22× | as stated | 0.88–1.16; 11 of 93 (my row set), maximum 1.224 (100f, 25%) | ≈ |
| 34 | Microbenchmark reaches 86–96% of the bound's read time | 86–96 | layer mode 86–96% (token mode 93–97%) | ✓ |
| 35 | Table 8 arithmetic | 565; totals 495/597/308/83 | 218+193+142+12 = 565; totals match; per-job totals for 099–108 match the text | ✓ |
| 36 | Parity: mean KL 0.0019 nats (gpt-oss) | 0.0019 | 0.00186 (stock offload: 0.00171); measured on an A100 + Xeon 6960P | ✓ (platform caveat, W5) |

**New observations from the same data, not in the paper**

- **A.** The Table 3 residual correlates with the link ratio within the 13 machines (ρ = −0.73 / −0.80), and the "together" share rises with the ratio (ρ = 0.83). See W1.
- **B.** The "MIN's set, read twice" arm reads 66.6 experts per token at 11%, more than the deployed cache's 63.1. See W2.
- **C.** Quiet re-probes read B_host 1.1–4.7% higher than probes taken during downloads, on three GPUs. See W3.
- **D.** No measured state beats Eq. (1) (minimum 1.33× at 11%). The prefetching oracle dips to 0.991× Eq. (2) on O4 at 25%, which the paper allows, because it reads ahead.
- **E.** All 15 decomposition machines produced consistent outputs in all five states (loss 0.189–0.191 nats), so the decomposition set is valid by the paper's own output criterion.

## Questions for the authors

1. Can you show Table 3 per machine, and as a regression of each share on the link-to-CPU ratio, with intervals? Given that the residual depends on the ratio (ρ ≈ −0.75), do you still want to say that two named parts "account for" what is left?
2. What are the achieved misses and reads of each 2×2 arm, relative to R★? Would an arm whose background copies are oracle-timed (issued early enough to land before reuse) separate "the second read" from "the late copy"? If it closes much more than 1% at 11%, "pay only together" becomes a statement about the copy path.
3. Can you quantify the bias from probing under download on more than the three job-102 GPUs, or re-probe a sample of the 25 machines quietly? How do the 31–54% shares move?
4. Why use the minimum T_GPU from profiles taken on other machines? With per-machine G, or with the median, does the residual's dependence on the ratio go away?
5. Which three claims would you call the paper's primary confirmatory results, and where is each one's pre-specified analysis (statistic, machine class, threshold) in a job header?
6. For Table 4, how much of the ours-vs-FreeToken difference could come from the systems decoding different texts? A teacher-forced comparison on identical text for one cell would settle it.
7. Section 6: how does the 0.65 C rule change if window errors are correlated with cache contents (for example, errors drawn from a real forecaster's confusion pattern instead of independently)? Can you give the time-domain values next to the read-domain values in the main text?
8. Of all offers rented, what fraction passed the gate and round check, by CPU class? Does "every machine that ran it stably" amount to "desktop parts"?
9. Is the `gpu` branch public, and at which URL, so that a third party can verify push times independently of `prereg/gpu_pushes.json`?

## What would raise my score

- **To 7 (accept as a workshop paper).**
  - Rewrite the abstract and Section 4 so that measured oracle removal, assumed attribution and post-hoc subsetting are visibly distinct.
  - Report the decomposition as a function of the link ratio, per machine, including the residual's dependence.
  - Rename the read-twice arm and report each arm's achieved reads against R★.
  - Quantify the probe-under-load bias using the job-102 re-probes.
  - Fix the README pipeline and Appendix L so that every number regenerates.
  - Add evidence labels to Section 4's paragraph headings, matching Section 5.
- **To 8 (strong).**
  - Profile T_GPU on the decomposition machines, or bootstrap over the profile range.
  - Re-run the decomposition on a fresh, pre-specified set of machines, with its shares registered in advance (one job), so that the title result becomes confirmatory.
  - Add an oracle-timed two-read arm (Q2).
  - Cut the main text's repetition, and use one name per configuration across text, figures and tables.

## Scores

| Criterion | Score | Note |
|---|---|---|
| Overall | 6 / 10 | Acceptable with revisions |
| Soundness | 3 / 5 | The measured numbers deserve 4–5. The interpretation of the title section (post-hoc averaging, a structured residual, a 2×2 arm that does not realise its label) and the overstated abstract pull it to 3. |
| Methodology | 4 / 5 | Exemplary record-keeping, gates, statistical unit and bounds. Not exemplary as confirmatory design, and the probe runs under load. |
| Significance | 3 / 5 | Useful, careful characterisation of a narrow but practical setting. The oracle-in-engine method is the most transferable part. |
| Clarity | 3 / 5 | Clearly improved structure. Still number-dense, repetitive, inconsistent in naming, and dependent on a notebook-like appendix. |
| Confidence | 4 / 5 | I recomputed 36 claims from raw data. I did not recompute R★ from traces or check the 9-model and audit analyses. |
