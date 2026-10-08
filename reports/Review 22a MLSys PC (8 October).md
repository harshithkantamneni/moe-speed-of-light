# Review 22a: MLSys 2027 main track, PC review

**Submission:** "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth" (blind)
**Reviewer:** PC member 22a. **Date:** 8 October 2026

---

## Materials read

- `paper/paper.pdf` (44 pages, built 8 Oct 04:00 CDT). I read all of it: the main text (pp. 1–9), the references (pp. 9–12), Tables 1–6 (pp. 13–16, discussed below), and Appendices A–L (pp. 17–44). Pages were rendered as images so the tables could be read.
- `paper/paper.tex`, `paper/app_wsg.tex`, `paper/app_prereg.tex`: the source of the main text and of the prediction-log appendix.
- `paper/supplement.pdf`: its structure, the scorecard header and totals, and the clause tables for jobs 109–112.
- Artifact, read for definitions only: `scripts/job109.py` (lines 1–260: ratio, capture, gap-share and bootstrap definitions), `scripts/decomp_measured.py` (header and panel rule), `scripts/speed_limit.py` (`host_rates`, `limit`, `MODELS`: D, S, B_gpu), `scripts/sumlaw_paper.py` (`profiles()`: the non-expert-time definition).
- Data inputs: `prereg/speed_limit_v2.json` (only for the R⋆ values 38.315 and 15.340), `data/gguf_bytes.json`, `data/published_measurements.json` (record format only), `data/traces/gpt-oss-120b` (array shapes only), and `gpu/vast_ledger.json`.
- `/home/claude/gpu-branch`:
  - Job script headers 099, 102, 109, 110, 111 and 112, plus the per-host wrappers.
  - The bodies of the post-launch amendments to jobs 073, 100, 102, 105, 106, 107 and 111 (`git show --format=`, so no commit messages).
  - Commit times (`git log --format='%h %ad'`, `%ct`).
  - Raw results for jobs 089, 090, 093–112: `ec_*.jsonl`, `st_*.json`, `concur.txt`/`concur2.txt`, `fetch_table*_law_gptoss.json`, `g_prof.json`, `prof_*.json`, `readsched_C*.txt`, `validity.txt`, `v0.txt`, `bs1.jsonl` and `summary.txt`.
- My own scripts, all in `scratchpad/rev22a/`: `rvlib.py`, `inventory.py`, `decomp.py`, `table6.py`, `few1r.py`, `boundshare.py`, `spearman.py`, `regtiming.py` and `regtiming_early.py`, plus several inline checks.

## Independence statement

- I did not open anything under `reports/` except to write this file.
- I did not open any `prereg/*outcome*.md`, `research_notes/`, any review, number-check, plan or progress-log file, `paper/paper_v1_prereview.tex` or `apply/`. Directory listings showed some of these file names; I did not open them.
- I read no commit messages; every `git log` and `git show` call used a format without the subject or body.
- I read the authors' scripts only for definitions. Every number in my claims table comes from my own code run on the raw per-problem rows, counters and probe files.
- I modified nothing in either repository apart from writing this file, and committed nothing.
- The job 112 header says in passing that its timing control answered earlier reviews ("reviews 21a and 21b"). I did not read those reviews.

---

## Summary

The paper studies batch-1 decoding of large MoE models (gpt-oss-120b and Qwen3-30B-A3B) on consumer GPUs (mostly RTX 5090) that keep most experts in host DRAM.

1. **The bound (Eq. 1).** It is a roofline-style bound: the host reads of Belady's MIN with bypass (R⋆, for C GPU slots per layer) at the machine's highest probed host read rate. A "demand" bound (Eq. 2) adds the GPU's non-expert time in series. An "ordered" bound (Eq. 3) recognises that only link copies can be issued ahead.
2. **Where their cache stands.** The authors' llama.cpp expert cache, which is faster than FreeToken and stock llama.cpp at most configurations, runs at 31–54% of Eq. (1) at gpt-oss 11% on 25 consumer-processor machines.
3. **The decomposition.** In-engine oracles split the deployed cache's gap to Eq. (1) with a 2×2 factorial: MIN's set vs the deployed set, crossed with one read vs two reads per admission. A read-ahead oracle follows. Only the combination pays (33% of the gap on a 15-machine panel, 35% on 5 machines rented for a registered test).
4. **Policies without foresight.** Four variants built from the engine's own mechanisms (larger admission margin, layer-ahead copy) recover at most 6% of the read-ahead oracle's gain.
5. **Second card.** A log-linear trend in the link-to-CPU read ratio, frozen on the RTX 5090 panel, predicted the oracles' speeds on five RTX 4090 machines within 7%.
6. **Horizon (exploratory).** Trace replays suggest that half of foresight's read savings needs the routing of about 0.65 C distinct experts per layer ahead.

Every GPU job's predictions were committed to git before its machine started, and the paper scores all of them, failures included.

## Strengths

1. **Evidential discipline well beyond the field's norm, and it checks out.**
   - All 92 rentals of jobs 093–112 in the ledger started after their job script was committed; the minimum lag is 2.7 s (105f), exactly as Appendix D states. All 29 rentals of jobs 073–092 did too.
   - The post-launch amendments I inspected (jobs 100, 102, 105, 106, 107, 111) substitute machines or add hosts. The one added prediction (job 100, #9) is disclosed.
   - Job 111's header keeps job 110's predictions verbatim. The void job 110 is reported as void.
   - 321 failed clauses are published, and the paper concedes when a registered band "could hardly have failed" (job 109).
2. **The raw data reproduce the paper.** From per-problem rows and counters, with my own code, I reproduced every cell of:
   - Table 4: all four columns, both budgets, the attributed rows and the t-intervals;
   - Table 5: speeds and misses;
   - Table 6: all 50 values, the capture percentages and the bracketed trend predictions;
   - Table 29;
   - Table 3's host-S block;
   - about 25 further quantities in the text (see the claims table).

   The frozen trend coefficients in job 110's header refit to four decimals from my own per-machine panel ratios. I found no number that the data contradict.
3. **A real methodological contribution.** The paper uses oracles inside a production engine and a factorial over what is cached and how it is read, with the machine as the statistical unit. That is the right way to ask "where does the time go", and it yields a non-obvious, replicated interaction.
4. **The transfer test is meaningful.** It was registered, the model was frozen, and the card was different. A ratio-free null (the panel's geometric mean) would have missed the RTX 4090s by up to 20.5% (my check), against the trend's 6.6%. So the ratio carries real information, beyond the weak "no change" null the paper uses.
5. **Measured negative results that builders can use.** A layer-ahead prediction moves reads earlier but does not remove them (the LA-1R variants read 1.67 R⋆, the same as the deployed cache's 1.65–1.66). One-token foresight buys little.
6. **Careful reporting of variance and validity.** Relaunches came within 0.7% (RTX 5090, jobs 109→112) and 5.5% (RTX 4090). A second probe agreed within 4.9%. The paper reports unsteady and wrong-output rentals, and is candid about server processors.
7. **A specific Limitations section** that names its own deviations, including the RTX 4090 reference-loss error and the job 112 replacement machine.

## Weaknesses (most important first)

**W1. The abstract, Section 8's rules and the Conclusion generalise beyond what was tested.** The body is mostly well scoped; the summary layers are not.

- *"Policies without foresight recover at most 6% of the oracles' gain"* (abstract; rule 3).
  - What was tested: four variants of the authors' own deployed cache (Margin, LA, LA-1R, LA-1R-margin), on 5 RTX 5090s, at two budgets of one model.
  - Section 6.1 says exactly that; the abstract says "policies without foresight".
  - Section 6.1's explanation, "they still read 1.66–1.68 R⋆ … as many as the deployed cache", holds for LA-1R and LA-1R-margin only. By my counters Margin reads 1.58 R⋆, LA reads 1.75 R⋆, and the deployed cache 1.65–1.66 R⋆.
- *"Half of foresight's value needs … about 0.65 C distinct experts per layer ahead"* (abstract; rule 5).
  - It is exploratory and post hoc (Table 1 says so).
  - "Value" here is a share of MIN's read savings over admit-every-miss, in host reads on traces, not time against the deployed cache.
  - The abstract says none of this.
- *"A trend in that ratio … predicted the oracles' speed on a second card within 7%"* (intro; rule 4; Conclusion).
  - Rule 4 and the Conclusion omit that the same trend missed 5 of 18 cells, by up to 17.6%, on new RTX 5090s, the very card it was fitted on.
  - Three of the five RTX 4090s sit at ratios 0.37–0.40, where the trend predicts MIN-1R ≈ 1.0, the same as the no-change predictor (which misses by only 0.5–3.7% for MIN-1R there).
  - The discriminating evidence is the read-ahead oracle plus two machines at 0.57 and 0.77. The test is still informative (see Strength 4), but the rule should state its reach honestly.
- *"31–54% of the bound's speed on machines with consumer processors"* (abstract; §3; rule 1; Conclusion).
  - Appendix H.3 says the consumer/server split was "drawn after the tests". The main text never labels this restriction post hoc, and Table 1 has no row for the headline number.
  - Server machines that passed every check fall at 12.4% (AMD engineering sample), 26.7% (EPYC 7543) and 46.1% (EPYC 7402P).
  - The 31% floor is a Threadripper 9960X (B_host 176–178 GB/s, link-to-CPU 0.32), classified as consumer. On that machine the paper's own tighter Eq. (3) gives 47–49%.

**W2. The 2×2's "set" factor is not cleanly manipulated on the two-read path, so "pay only together" is established for this engine's read paths, not as a general law.**

- MIN-2R ("MIN's set alone") does not hold MIN's set on fast-link machines. At gpt-oss 11% on the job 109 machines (job 099b is similar):

  | Arm | Misses per token |
  |---|---|
  | Deployed cache | ≈ 60 |
  | MIN-2R | 48.6–50.8 |
  | MIN-1R | 39.7 |

  The arm therefore realises only about half of MIN's miss reduction. The authors' own counters explain why: the machine's fetch table fetches 16–24 misses per token in the step, into slots the schedule did not choose.
- The registered control (Few-2R-early) failed its manipulation check: misses fell only to 0.97–0.98 of Few-2R's, against a registered ≤0.95. So it cannot isolate the copies' landing time either. The conclusion "the late landing is not what holds the two-read arms back" is an inference from a manipulation that did not take.
- Where MIN-2R does keep MIN's set (099f: fetch table nearly inactive, 43.9 misses per token), the machine is slow-linked, which confounds the read-count factor.
- The abstract states the general form: "caching the experts MIN would keep pays only if each admitted expert is also read once". A MIN-2R arm with the fetch table forced to all-CPU, on fast-link machines, would separate set from read path. Without it, the claim should be scoped to this engine's two read paths.

**W3. The bound is relative and partly unverifiable from the artifact.**

- *R⋆ cannot be recomputed from the artifact.* R⋆ is the backbone of Eq. (1), Table 4 and the 31–54% figure. It comes from the AIME routing trace (`route_aime25` npz / `la_g.bin`), which the artifact does not include: the job directories list the `.bin` files as omitted, and `data/traces` holds a different, mixed-domain trace. I could only check consistency: the engine's MIN-1R reads 1.037 R⋆, in line with R⋆_eng = 39.3.
- *B_host is soft.*
  - It is the maximum of one probe run, often taken during the model download.
  - The paper's own Eq. (4) implies engine read rates above it on 12 of 84 launch-budgets.
  - The probed link-to-CPU ratio of one machine moved by −11.8% between rentals (job 109f → 112g) while its decode time moved by 0.7%.

  The paper discloses all of this. Still, the "headroom" numbers (31–54%, and the audit's 13.6%) inherit this uncertainty, and the audit uses a different method again (datasheet rates, uniform routing for some rows, a different routing granularity).
- *"Nearly reachable" is a microbenchmark result.* The read-time microbenchmark replays MIN's reads with the same kind of code that defines B_host. Reaching 86–96% shows that per-layer granularity costs little. It does not show that a decoding engine can reach the bound. The registered threshold (≥50% per layer) was very lax for the "nearly reachable" wording.

**W4. "What is left" is attributed with a single RTX 5090 constant.**

- The decomposition uses T_GPU = 2.94 ms, the smallest RTX 5090 profile, on every machine, including the RTX 4090s.
- The RTX 4090 profiles in the artifact give 3.41–3.55 ms of non-expert time (G = 5.1–5.6 ms, against 4.05–4.67 ms on the 5090).
- With the card's own value, the "unexplained" slow-link residual on the three RTX 4090s falls from 31.5–35.8% to 26.6–30.7%.
- Minor: "the smallest of our engine's Nsight profiles" is no longer true. The job 109d profiles give 2.87–2.91 ms; the registered value is still fine to use.

**W5. The evidence base is narrow relative to the title.**

- Nearly all registered results are gpt-oss-120b at two budgets, batch 1, in one engine (the authors' llama.cpp patch), with AIME prompts that were also used during development.
- Qwen3 appears only in the factorial and the trace study.
- The two "rented for the test" samples have 5 machines each, so the headline 35% has a 95% interval of [22, 48].
- The novelty of the bound itself is modest: Eq. (1) is a roofline over MIN's reads, and Related Work already includes MIN-with-bypass gaps (Zhang 2026b), oracle lookahead hit rates (Liang 2026b), WiSP and Budgeting Bytes. The in-engine time decomposition is the new part.
- The published-systems audit compares across hardware with datasheet ceilings. It is useful context, but weak as evidence.

**W6. Format and page budget.** All six main-text tables are floated past the references, to pages 13–16, each nearly alone on a page. With them in the body, the main text would very likely exceed MLSys's 10-page limit. The chairs should check compliance. This is also the top clarity problem below.

**Minor issues.**
- "Registered that their speed … would fall within 0.10 of it in log" (§6.2) simplifies the registered bands. They were 0.06 for Dep-1R and MIN-2R at 11%, and "all but at most one host-configuration" at 25%. State them exactly.
- Table 5's caption says every 95% interval is within ±0.01. My paired bootstrap, resampled jointly across rounds, gives up to ±0.012 for Few-1R on the 5950X and i5-12400.
- The "13 machines" for Few-1R (§5) include the two RTX 4090s of job 112, but §2 says machines are RTX 5090s except in §6.2.
- Job 112's i5-12400 RTX 4090 had run job 091 before. The job header discloses this; the paper does not.
- Builder rule 2's "about a third" for fast links: the 15-machine panel figure (33%) includes both slow-link machines. On fast-link panel machines only it is 39% [31, 47]; on the new machines it is 35%.

## Clarity

Earlier rounds gave 2/5 and then 3/5. This version reads better than its density would suggest, but two mechanical problems and one conceptual one keep it at 3.

**What works**

- Table 1, as a concept: claim → test → outcome → number of machines → section, with "pre-specified / post hoc / derived" labels. The labels I audited are accurate (see the registration audit table).
- Paragraph leads that carry their evidence status inline, such as "(registered: a positive interaction on every machine)" and "(exploratory)". Together with "attributed, not measured" in Table 4, this is exemplary.
- The naming scheme, set + reads (MIN-1R, MIN-2R, Dep-1R, Few-1R), and Table 2.
- The running example (the 9950X: 10.0 / 13.0 / 20.6 / 15.2 / 13.8 ms) grounds the abstractions.
- Fig. 2 (the staircase) is the clearest single picture of the result.
- The "Yardsticks" paragraph names the denominators up front.
- Section 8 turns the results into usable rules.
- The Limitations are concrete.

**What still makes it hard to read**

1. **The main-text tables are not in the main text.** Tables 1–6 render after the references, on pp. 13–16. Table 1 is the paper's roadmap, cited on p. 1 and printed on p. 13. Table 4 carries Section 4 and sits on p. 14, ten pages after the paragraph that discusses it. In practice a reader cannot follow Sections 3–6 without flipping back and forth. This is a float-placement problem (two-column `[t]` tables that are too tall, inputs placed after long figures), and it is fixable.
2. **Too many denominators, often side by side.**

   | Measure | Denominator |
   |---|---|
   | Speed (×) | the deployed cache on the same machine |
   | Bound share (%) | the speed of Eq. (1), or of Eq. (3) |
   | Gap share (%) | the deployed cache's time beyond Eq. (1) |
   | Capture (%) | the read-ahead oracle's gain over the deployed cache |
   | Window value | MIN's gain over *admit every miss*, in host reads or in time |

   The abstract alone has four of these: 31–54%, 35%, 6% and "half of foresight's value". Fig. 5's caption has to open with "Baseline here: admit every miss, not the deployed cache", which is a symptom. A single summary panel, or a fixed symbol per measure (e.g. ×, %bound, %gap, %capture, %MIN-gain), carried through all text and tables, would help a lot.
3. **Vocabulary and indexing load.**
   - Table 2 has 18 configurations.
   - The main-text captions cite job numbers ("job 109", "job 111 (job 110's predictions …)").
   - The appendices switch to host codes (O1–O6, Pa–Pj, "Pd again", "9950X, slower link", hosts A/B/S) and to script codes (fetch, bypass, foa, both3p, dk, pf, R1, R2) that differ from the main-text names. Appendix C maps them, but the reader has to keep using it.
   - The main text needs at most about 8 names. Belady-1R/2R and the window variants could live in Section 7 and the appendix.
4. **Number-dense sentences.** Many sentences stack three or four ranges, each with its own denominator, plus parentheticals. Examples:
   - the §4 control paragraph, "On four machines (both cards, ratios 0.57–0.88) at 11%, Few-2R-early runs 0.98–1.00× … against 1.21–1.41× … (0.97–0.98 of Few-2R's, where we had registered at most 0.95 from a replay) … by 12.1–13.8 per token";
   - the §6.2 paragraph after the footnote.

   Negation chains ("closes almost none of the gap; … closes none either") slow reading further. Move ranges into tables and keep one number per claim in prose.
5. **Several bounds, unclear which one is "the bound".**
   - The headline uses Eq. (1), "the bound we use throughout", while Eq. (3) is "the tighter bound" and "the bound is the larger of the two".
   - Eq. (2) appears in Fig. 2 and in the running example; Eq. (4) (the closed-form account) appears in §3, §4 and the Limitations, although its registered tests failed.
   - The reader has to track four equations to follow the headline percentages.
   - The abstract is not self-contained: "MIN with bypass", "the gap", "the oracles' gain" and C are undefined there.

**Smaller points**
- The intro's bold summary paragraphs run in a different order from the sections (§3, 4, 6.1, 5+6.2, 7).
- Section 6 bundles two unrelated questions. Section 5 is two paragraphs and a dense 7×6 figure (Fig. 3) whose Belady arms are not otherwise used in the argument.
- The intro says things three times: bold paragraphs, a Contributions list and Table 1. That space would be better spent bringing the tables back into the body.
- Fig. 4 overlays four fits and four marker types on two budgets. It is legible, but at the limit.
- The 28-page appendix is thorough but written as a lab notebook (job by job). A short "where each main-text number comes from" index would help more than the map in Appendix A.

## Claims checked against raw data

"Reproduced" means my independent computation from the raw files matches the paper to its stated rounding.

| # | Claim | Location | My value (raw data) | Verdict |
|---|---|---|---|---|
| 1 | Panel 11%: set alone 0 [−2,3], once alone 0 [−2,2], both 33 [21,44], read-ahead 15 [11,19], left 52 [43,62] | Table 4, §4, Intro | 0.5 [−1.8,2.8]; −0.0 [−1.9,1.8]; 32.6 [21.0,44.3]; 14.9 [10.7,19.1]; 52.5 [42.6,62.3] (15 machines: 096a/b, 099a–j, 100b, 100f, 101b) | Reproduced |
| 2 | Panel 25%: 21 / −3 / 31 / 21 / 48 | Table 4 | 20.6 / −3.2 / 31.3 / 20.6 / 48.1, intervals match | Reproduced |
| 3 | New (job 109) 11%: 3 / 0 / 35 [22,48] / 19 / 46 | Table 4, abstract | 3.4 / −0.4 / 34.7 [21.9,47.5] / 19.0 / 46.3 | Reproduced |
| 4 | New 25% (4 machines): 20 / −2 / 27 / 24 / 48 | Table 4 | 19.8 / −2.4 / 27.3 / 24.4 / 48.3 | Reproduced |
| 5 | Attributed rows: T_GPU 27/28/34/38; extra reads 21/18/19/16; residual 4/1/−5/−5 | Table 4 | 27.4/28.2/34.0/37.9; 21.4/17.5/19.1/15.6; 3.7/0.5/−5.0/−5.1 | Reproduced |
| 6 | Interaction > 0 registered; held on 9 of 10 (job 099) and 5 of 5 new | §4, Table 1 | Only failure: 099f (ratio 0.29), −0.48 ms at 11% (positive at 25%); all 5 job-109 machines positive at both budgets | Reproduced |
| 7 | Read-ahead oracle reads 1.22–1.31 R⋆ at 11% | Table 2 | 1.218–1.312 | Reproduced |
| 8 | Slow-link residual 35–47% (panel) and 31–36% (RTX 4090) | §4 | 35.5, 47.4; 31.5–35.8 (26.6–30.7 with the 4090's own non-expert time, 3.41–3.55 ms) | Reproduced; sensitive to T_GPU (W4) |
| 9 | Read-ahead leaves 83–99% (panel slow) and 73–81% (4090) of the gap to Eq. (3); median 45% on fast links | §4 | 83.1, 98.6; 72.7–80.6; 45.5 | Reproduced |
| 10 | Table 6, job 109: MIN-1R, ahead, best none, capture (5 machines × 2 budgets) | Table 6 | All values match to 0.01 and ±1 point; e.g. 285K 1.22/1.46/1.02/4%; 7945HX 25%: 0.99, −1% | Reproduced |
| 11 | Table 6, RTX 4090 (jobs 111, 112), with bracketed trend predictions | Table 6 | All match; e.g. 14900KF 0.96 [0.99], 1.13 [1.11], 26±3% | Reproduced |
| 12 | MIN-1R and ahead within 7% (11%) and 5% (25%) of the trend on 5 RTX 4090s; every 95% interval inside the band | §6.2, abstract, rule 4 | Max 6.6% and 5.0%; worst \|dev\| + half-width 0.079 < 0.10 | Reproduced |
| 13 | Frozen trend (job 110 header) fitted on the 15 panel machines | job 110 header, §6.2 | My refit: identical coefficients to 4 decimals; max residuals 0.081 and 0.094 | Reproduced |
| 14 | "No change" misses by up to 46% / 69%; slow-panel mean by up to 38% / 34% | §6.2 | 46.5 / 68.9; 38.2 / 34.2. (My panel-mean null: up to 20.5%.) | Reproduced |
| 15 | Trend misses 5 of 18 new RTX 5090 cells by more than the band, by up to 18% | §6.2, Table 1 | 5 of 18 (109e ahead ×2, 109d ahead, 109a MIN-1R ×2); max +17.6% | Reproduced |
| 16 | Few-2R-early 0.98–1.00×, Few-2R 1.03×, Few-1R 1.21–1.41× (4 machines) | §4, Table 5 | 0.977–0.997; 1.027–1.030; 1.208–1.406 | Reproduced |
| 17 | Early misses 0.97–0.98 of Few-2R's; above Few-1R by 12.1–13.8; fetch table fetches 17.8–23.7 of 53.0–54.6 misses | §4, Table 5 | 0.968–0.980; 12.1–13.8; 17.8–23.7 of 53.0–54.6 | Reproduced |
| 18 | Table 5: every 95% interval within ±0.01 | Table 5 caption | Few-1R half-width up to 0.012 (5950X, i5-12400) | Minor discrepancy |
| 19 | Registered timing clauses T1 (misses ≤ 0.95×) failed 5/8; T2 failed 7/8 | App. D, Table 1 | T1 fails 4/4 at 11% and 1/4 at 25%; T2 fails 7/8 (only 7945HX 25% passes) | Reproduced |
| 20 | Variants without foresight: best 1.007–1.024× at 11%, capture ≤ 6% | §6.1, abstract | 1.007–1.024; max 5.7% | Reproduced |
| 21 | "They still read 1.66–1.68 R⋆, as many as the deployed cache" | §6.1 | LA-1R 1.665–1.667, LA-1R-margin 1.674–1.678; Margin 1.575–1.588; LA 1.751–1.755; deployed 1.645–1.660 | True for 2 of the 4 variants |
| 22 | Margin cuts host reads by 4–6% at 11% | §6.1 | 4.2–5.2% | Reproduced |
| 23 | RTX 4090: LA-based variants 0.64–0.72×, Margin 1.033–1.041×, capture up to 26% | §6.1 | 0.644–0.724; 1.033–1.041; 25.7% | Reproduced |
| 24 | MIN-1R on 4090s: 0.96–1.01× (slow), 1.13–1.31× (higher ratios) | §6.2 | 0.963–1.007; 1.133–1.312 | Reproduced |
| 25 | Few-2R paid at RTX 5090 ratios 0.28–0.32: 1.11–1.22× | §6.2 | 1.112–1.217 (104a, 105b, 105e) | Reproduced |
| 26 | Few-1R beats deployed on all 13 stable machines, geomean 1.24×; loses on the two slowest-link unsteady ones | §5, App. I | 13 (11 RTX 5090 + 2 RTX 4090), 1.240, t-interval of mean 1.17–1.32; losses 0.845 (ratio 0.14), 0.974 (0.21) | Reproduced (2 of the 13 are RTX 4090s) |
| 27 | 31–54% of Eq. (1) on 25 consumer machines at 11%; 40–54% of Eq. (3); new machines 41–48% | Abstract, §3, Conclusion | 39 launches on 25 machines: 30.6–54.0% (per machine 31.2–54.0); 40.5–54.0%; 41.5–48.0% | Reproduced; consumer scoping is post hoc (W1) |
| 28 | Eq. (3)/Eq. (1): median 1.00, up to 1.56 | §3 | 1.00; 1.56 (TR 9960X) | Reproduced |
| 29 | Server launches: 3 of 8 passed every check | §10 | 3 of 8 (105a, 107d, 108a), at 46.1%, 12.4% and 26.7% of Eq. (1) | Reproduced |
| 30 | Running example (9950X, 11%): Eq. (1) 10.0, Eq. (2) 13.0, deployed 20.6, MIN-1R 15.2, read-ahead 13.8 ms | §5 | 10.02, 12.96, 20.64, 15.16, 13.81 | Reproduced |
| 31 | Spearman (MIN-1R gain vs ratio) 0.89 [0.63, 0.97] over 19 machines | §6.2 | 0.89 [0.63, 0.98], same 19 machines; MIN-2R vs ratio 0.45; admit-every-miss 0.95 | Reproduced |
| 32 | Read-time microbenchmark: 86–96% of the bound's read time, per layer | §3, Table 29 | 86/89 (13900KF), 96/96 (9950X), 95/93 (9800X3D); per token 93–97% | Reproduced |
| 33 | Table 3, host S: ours, FreeToken, ratio with CI, ×llama.cpp, bound share | Table 3 | Launch 2 of job 089 reproduces all six rows (e.g. 57.9 / 48.0 / 1.207 [1.195, 1.219] / 2.00× / 140 tok/s, 41%) | Reproduced |
| 34 | Host S vs B ratio over FreeToken outside ±0.06 at 5 of 6; still faster at 11 of 12 | Table 1 | Differences −0.087, −0.079, −0.061, −0.005, −0.105, −0.075; only Qwen3 43.75% on S below 1 | Reproduced |
| 35 | Same-CPU panel machines differ by up to 29% in deployed time | §2 | 285K: 18.14 vs 14.06 ms (099a vs 099f) = 29.0% | Reproduced |
| 36 | Relaunches within 3.2% (RTX 5090) and 5.5% (RTX 4090); 109→112 within 0.7% | §2, App. D | TR 9960X 3.2%; 110c→111g 5.5%; +0.42% and −0.71% | Reproduced |
| 37 | Second probe within 4.9% on 4 machines; link-to-CPU ratio moved by up to 12% between rentals | §10 | +4.9% max (112b); 0.837→0.738 (−11.8%) | Reproduced |
| 38 | Exact 16-token window recovers 0.78–0.93 of the gain in time at 11% (registered ≥ 0.80; lowest missed) | §7 | 0.78–0.93, 1 of 10 below 0.80 | Reproduced |
| 39 | Same window copied in the step: 0.89–1.32× at 11% on 4 machines | §7, Table 20 | 0.89, 0.89, 1.20, 1.32 | Reproduced |
| 40 | G = 4.1–4.7 ms (gpt-oss, RTX 5090); T_GPU = 2.9 ms, the smallest profile | App. H, §3 | 4.05–4.67; 2.94 (among the 069c/105 profiles); later 109d profiles 2.87–2.91 | Reproduced (nit) |
| 41 | RTX 4090 loss 3% higher than the RTX 5090 reference | Footnote 1 | 0.1963 vs 0.190 (+3.3%) | Reproduced |
| 42 | KL 0.0019 / 0.0005 nats (parity) | §10, App. J | 0.00186 / 0.00053 in the on-machine summary (raw logit dumps not re-scored) | Consistent |
| 43 | 92 rentals of jobs 093–112, each after its commit, by ≥ 2.7 s | App. D | 92; minimum 2.7 s (105f); also 29 of 29 for jobs 073–092 | Reproduced |
| 44 | 124 rentals (74 offers) of jobs 058–108 cost $87.2 | App. L | 124, 74, $87.2 | Reproduced |
| 45 | Published systems: median 13.6% of their bound over 20 rows | §3, Table 1 | 13.6 (median of Table 31's 20 in-class rows); per-row bounds not recomputed | Arithmetic reproduced |

**Not checkable from the artifact:** R⋆ (38.315 / 15.340 reads per token), the trace-replay numbers of §7 and Tables 16–19, and the "−0.4% on the first 20 problems" figure all need the AIME routing trace, which is not shipped. As a consistency check, the engine's MIN-1R reads 1.037 R⋆.

## Registration and evidence-label audit (Table 1)

| Table 1 row | Job(s) | Header commit vs first machine | Registered clause vs paper | Label correct? |
|---|---|---|---|---|
| Read time nearly reachable | 102 | Committed before every rental (minimum 3.3 s after the amendment; 27 s after the first commit) | ≥ 0.80 per token and ≥ 0.50 per layer held; "≥ 0.03 below analytic" failed on 2 machines × 2 budgets; budget order failed on the Intel host: 5 of 33 failed | Yes. The ≥ 50% threshold is too lax to support "nearly reachable"; that wording is post hoc. |
| Faster than FreeToken / llama.cpp | 081, 089 | All 29 rentals of jobs 073–092 after their commits | Verified on host S | Yes |
| Set × read interaction | 099; 109 | 099 header before its rentals; 109 header 22:14Z, machines 22:35Z | 099 registered "positive at both cells on every host": 9/10 at 11%, 10/10 at 25%. 109: P1 held on 5/5 | Yes. The "fast links" scoping is post hoc (the 0.5 line was drawn after the panel), and the paper discloses it in §2. |
| Landing copies does not rescue 2R | 112 | Header 05:41:41Z; first rentals 05:42:54Z | T1 failed 5/8, T2 failed 7/8, T3 and T4 held | Yes; but see W2 on the failed manipulation |
| Variants without foresight recover little | 109 | As above | P7 (≤ 1.10×, capture ≤ 0.35, mean ≤ 0.25) held; max 5.7% | Yes. The wide bands "could hardly have failed", as the paper itself says. |
| Ratio predicts a second card | 110 (void), 111, 112 | 110 header 22:39:06Z, rentals from 22:39:15Z; 111 header 23:52:00Z, rentals from 23:52:10Z; amendment 00:10:51Z, relaunches from 00:11:02Z | Q1/P1 held on all 5; the band was 0.06 for Dep-1R and MIN-2R at 11% | Yes; the main text simplifies the band |
| Few-1R gains on every stable machine | 104–107, 112 | Committed before | Held | Yes |
| Published systems median 13.6% | none | n/a | n/a | "post hoc", correct |
| 0.65 C horizon | none | n/a | n/a | "post hoc" in Table 1, "exploratory" in §7; **not flagged in the abstract or rule 5** |
| Closed-form account | 105–108 | Committed before | 3 failed, 1 inconclusive (Table 21) | Correct. It is nonetheless used in §3 and §10 to discount the 54% top and to argue that the probe underestimates. |
| *(missing)* 31–54% on consumer processors | panel, 103–112 | n/a | The consumer/server split was drawn after the tests (App. H.3) | **Missing row; post hoc scoping not labelled in the main text** |

## Questions for the authors

1. Can you add a MIN-2R arm with the fetch table forced to all-CPU (so that MIN's residency survives the two-read path) on fast-link machines? If MIN-2R still gains nothing there, the "pay only together" claim becomes general. If it gains, the interaction is partly a property of the fetch table.
2. Will you ship the AIME routing traces (`route_aime25_*.npz` / `la_g.bin`) so that R⋆ and the §7 trace numbers can be recomputed independently?
3. Why does the headline use Eq. (1) rather than max(Eq. 1, Eq. 3), when the paper itself argues that Eq. (3) is a valid, tighter bound? With Eq. (3) the consumer range is 40–54%.
4. Will you recompute the RTX 4090 attribution (Table 4's logic, §4's 31–36% residual) with the card's own non-expert time?
5. Was the consumer/server restriction registered anywhere before job 107? If not, please label it post hoc wherever the 31–54% appears, and add it to Table 1.
6. For the second-card test, please report a ratio-free null (e.g. the panel mean). By my count the trend beats it by a wide margin, which would strengthen the claim more than the "no change" comparison does.
7. Have you built any realisable multi-token predictor *into the engine*, rather than evaluated it offline on host reads? Section 7's "nothing realisable reaches it" rests on offline replays.
8. How sensitive are the 2×2 shares to κ and to the fetch table, both of which belong to the deployed baseline the gap is measured against?
9. Does the main text, with Tables 1–6 in place, fit the 10-page limit?

## What would raise my score

- Put Tables 1–6 in the body and meet the page limit; consolidate the yardsticks; cut the main-text vocabulary to about 8 configuration names. This alone would take clarity to 4.
- Rescope the abstract, the rules and the Conclusion:
  - "four variants built from our engine's mechanisms", not "policies without foresight";
  - label the 0.65 C rule exploratory and in host reads;
  - in rule 4, mention the 5-of-18 misses on new RTX 5090s;
  - label the consumer-only restriction post hoc and add a Table 1 row for 31–54%.
- Either the fetch-table-off MIN-2R arm, or "pay only together" scoped explicitly to this engine's two read paths.
- Ship the routing traces behind R⋆.
- Report the stronger null for the transfer test; recompute the RTX 4090 attribution with that card's own T_GPU.
- Beyond this round: an in-engine, realisable multi-token predictor would raise significance more than any further oracle.

## Scores

| | Score |
|---|---|
| **Overall** (1–10; 6 = weak accept, 8 = strong accept) | **6** |
| Soundness (1–5) | 4 |
| Significance (1–5) | 3 |
| Novelty (1–5) | 3 |
| Clarity (1–5) | 3 |
| Confidence (1–5) | 4 |

**Justification.**
- *Soundness.* The measurements are as solid as I have seen in a systems submission: every number I recomputed from raw rows matches, and the registrations are genuine and git-ordered. Soundness is held at 4 by interpretation and scope: W1's abstract-level over-generalisation, W2's confounded "set alone" arm, and W3's unshippable R⋆.
- *Significance and novelty.* Both are moderate. The bound is a simple idea; the in-engine factorial and the second-card transfer are the real contributions; the scope is one engine and essentially one model.
- *Clarity.* It improved in structure (Table 1, inline evidence labels, naming scheme, running example), but deferring all the main tables past the references, the many denominators, and the dense, number-stacked prose still make it a hard read.
