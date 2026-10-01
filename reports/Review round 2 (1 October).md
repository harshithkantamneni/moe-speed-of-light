# Review round 2 — after tranche 1 and the grid (1 October 2026)

Two independent reviews of `paper/paper.pdf` at commit b01d16e (main text 10 pages; jobs 073–092 scored), each written
without seeing the other or the work in progress. The previous round (30 September, six personas, before the
accounting revision) averaged 4.8/10.

| Reviewer | Overall | Recommendation |
|---|---|---|
| MLSys PC member (systems for ML) | 6/10 | weak accept; a clarity-focused reviewer would say borderline |
| Performance-modeling professor (prospective supervisor) | 6.5/10 | borderline accept / major revision; "would take this student today" |

**Where the two agree (the grade is held down by these, in order of cost):**

1. **Writing hides the contribution.** A ~430-word abstract of ranges; a contributions list that is a results dump;
   Section 5 a chronological tour of jobs with job IDs in prose; reviewers will not reach Sections 6–7, the best parts.
   Fix: lead with the limit, the accounting and the traces; compress the system to ~1.5 pages plus Table 8; abstract
   ≤ 200 words; job IDs to the appendix. $0, the hardest edit.
2. **The accounting's terms are partly definitional and the text does not say so.** The "GPU efficiency" fix is a
   per-cell factor calibrated so the measured time is reproduced (a residual by construction); "overlap" is the
   limit's min-over-c form, i.e. overlap plus rebalancing 13–30 experts per token onto the CPU; the Nsight profile's
   "overlap would save 5–10%" sits unreconciled beside Fig. 1's 28–30%. ~10 points of "policy and paths" is the gap
   between the probe's max sample and the median pair. Fix: say so in Section 4, put the gross-GPU variant in the
   figure, and measure v(none) and one interior state with a serialised-execution flag (~$2).
3. **Abstract-versus-body overclaims.** "Bounds any exact-routing system" (the body scopes it to per-layer C; the
   pooled optimum in Table 3 beats it by 5–10%); "never largest in any order" (the body concedes 40 and 15 of 120
   orders when the all-in-VRAM shortfall is charged in full); "median 9.5% of their own limit" (datasheet host ceilings,
   imputed DRAM bands) next to "ours 25–42%" (probed), which Section 7 itself calls not like for like; "169 clauses
   scored" without the 46 failures; a 3.7% median that is per-configuration while the LOHO note uses 3.3% per row.
   Fix: scope the sentence, state the hit rate, bracket the audit number. $0.
4. **The law is validated more narrowly than the abstract implies, and its fitted G (4.82–5.17 ms) exceeds the whole
   all-in-VRAM token (3.92 ms)** because it absorbs a byte-proportional helper shortfall; the max-vs-additive contest
   rests on three fast-link FETCH rows; 19 of 29 configurations use one path; FreeToken is never predicted; median error
   is the wrong scorer for a law whose job is to choose the split on hard machines (Eq. 3 overstates differences 1.8×).
   Fix: p90 and per-regime error, state the regime split or promote variant E, predict FreeToken if its counters exist.
   Mostly $0.
5. **Foresight is named the largest fixable cost and never measured.** The engine records routing lookahead; an
   oracle-lookahead replay at W = 1, 4, 16 on three cells (teacher-forced) would turn the Shapley foresight share into
   a measurement. ~$5–8, the one experiment both reviewers want.

Also: the headline machine is the overclocked rental (1.21× the memory clock of 25 of 26 others) while the stock-clock
host, where FreeToken wins a cell, is demoted; intervals cover problem sampling, not machine-to-machine variance (two
more rentals of the headline cells, ~$6–10, would give a machine-level interval); the workload is 30 AIME prompts also
used in development (one non-math prompt family on the six headline cells, ~$3–5); the Table 3 "all" column prints 41
where 69.9/168 = 41.6; the W50 exponent without the six sub-token points is 1.51, outside its own 1.09–1.47 interval.

**What would make it a 9, per the professor:** scope the bound; measure foresight; lead with the representative
machine; cut a third of the text — "that is an 8. Close the loop on foresight with measurement and it is a 9." Not
reachable by 30 October: a foresight mechanism that recovers the gap, locked clocks, a second 100B-class family,
batch > 1.

---

## Review 1 — MLSys PC member

(verbatim)

1. SUMMARY
The paper studies batch-1 decode of MoE models whose experts live in host DRAM behind one consumer GPU and proposes a per-token "host-memory law", T = G + max(Xc/Bc, Xp/Bp, (Xc+Xp)/Bcp), with bandwidths from an on-machine probe, tested blind on 7 hosts (29 configurations, 3.7% median error) and against additive and per-layer forms under leave-one-host-out validation. From Belady's optimum with bypass on recorded routing traces, two host read paths and probed ceilings, it derives a speed limit in seconds for any exact-routing system at a given expert budget, and reports stock llama.cpp, FreeToken and the authors' llama.cpp expert cache at 9-20%, 20-40% and 25-42% of it on the headline machine. A Shapley attribution over five "fixes" says the remaining gap is mostly missing routing foresight where the host binds (37-53%) and un-overlapped host reads where the GPU binds (39-46%), never kernel efficiency, policy or host work. Trace replays of nine models show online policies read 35-110% more than the optimum, static placement 1.4-13x, four tokens of lookahead beat every online policy, and batched speculative verification recovers little, so foresight must come from predicted routing. The instrument is a 3,700-line llama.cpp expert cache with a per-machine CPU/PCIe miss split from the law; it leads FreeToken by 15-29% on gpt-oss-120b and 3-15% on Qwen3 on one machine (less, or reversed, on a second), 52 published measurements are placed at a median 9.5% of their own limit, and 169 pre-registered clauses are scored (91 held, 25 on the point estimate, 46 failed).

2. SCORES
Novelty 3 | Technical soundness 3 | Experimental rigor 4 | Clarity/writing 2 | Significance 3 | Reproducibility 5
Overall: 6/10. Recommendation: weak accept (a clarity-focused reviewer will say borderline/5).

3. STRENGTHS
- Protocol rigor rare at MLSys: predictions committed before each machine starts, paired bootstrap CIs on every ratio, selection launch separated from confirmation launch, a bound beside every measurement (Hoefler-Belli rule 11), a cost ledger, and every number generated from prereg/*.json. I recomputed Table 2's medians/maxes from prereg/perlayer_model.json, the limits from speed_limit_v2.json (38.32 reads x 13.25 MB / 87.5 GB/s = 5.80 ms = 172 tok/s), the Shapley shares from shapley_gap.json, and Table 1's ratios and percentages: all consistent.
- The limit-in-seconds plus the audit is a yardstick the community lacks; "published trace-scored systems sit at a median tenth of their machine" is a message worth publishing even with its caveats.
- The nine-model trace study is clean and actionable, particularly the negative results (speculative batching is not foresight; the hot set turns over 33-75% per 1,000 tokens so static placement fails).
- Honest reporting: the "What we do not claim" paragraph, failed clauses listed and explained, the confounded FreeToken comparison rerun on pinned cores, ties on FreeToken's own model stated plainly.

4. WEAKNESSES (ranked by cost)
(1) Clarity: the paper hides its contribution. Fixable, $0. The abstract is ~430 words with ~40 numbers; the contributions list is a results dump; Section 5 is a chronological tour of jobs 077-092 with job IDs in the main text ("job 079", "085b", "job 089"). Example: "Ours runs 1.21, 1.20 and 1.09x FreeToken on gpt-oss and 1.03, 1.05 and 0.97x on Qwen3: it leads at five cells and FreeToken leads at Qwen3 43.75% (interval 0.962-0.987). Every ratio is below listing B's, by 0.01-0.11, beyond the +-0.06 we predicted at five of the six cells (one by 0.001)." Most of Section 5 belongs in Table 8 and the appendix. Reviewers will not reach Sections 6-7, which are the best parts.
(2) The accounting's terms are partly definitional, and the text does not say so. Fixable (a paragraph and a figure variant); structural if a measured decomposition is wanted. In scripts/shapley_gap.py the "GPU efficiency" fix is a per-cell factor eta = 1 + resid/gpu_ds "calibrated so that the measured time is reproduced with no fix applied": the fifth term is the law's residual (0.8-1.6 ms per cell), so the five shares close the gap by construction. The "overlap" fix is the limit's min-over-c form, i.e. overlap plus optimally rebalancing 13-30 experts per token onto the CPU, not overlap alone. The paper's own Nsight profile says "neither system overlaps its PCIe copies with GPU compute; doing so would save 5-10%", while Fig. 1 charges 28-30% of the token to "no overlap" in the same cells. Unreconciled, a reader concludes one of the two is wrong. State both facts in Section 4 and put the gross-GPU variant (26-28%) in the figure rather than a sentence.
(3) The law's validation is narrower than the abstract implies. Partly fixable. The 29 configurations are gpt-oss-120b on the authors' engine (19 of them exercising one path), Qwen3 enters only through two llama.cpp rows, and FreeToken, whose split the law is lifted from, is never predicted, though the paper claims the law governs "every mechanism in this design space". "Beats the additive alternative (3.1% against 4.5% median error)" is a median over 33 host-clustered rows with no interval; in-sample the forms are 2.8 vs 3.3, and the body concedes the additive form is closer on the slow-link copy rows. The honest claim is "not worse, and exact on the three fast-link copy rows". G = 4.82 ms exceeds the all-in-VRAM token (3.92 ms) and, per the paper, absorbs helper-path slack collinear with a rate factor, so the "fixed GPU term" reading of the law is not what was fitted.
(4) The system result is modest and the comparison asymmetric. Structural. 15-29% over FreeToken on gpt-oss, 3-15% on Qwen3, -3 to +5% on the second host, a tie on FreeToken's headline model; "FreeToken's fetch cap and thread count were left at their defaults, while our FETCH table was swept"; the Qwen3 25% row is scored on the selection launch. The paper diagnoses foresight as the largest cost and then builds no foresight. Calling the system "the instrument" is right, but it then should not occupy three pages.
(5) Headline numbers the paper itself says are biased. Fixable, $0. The abstract's "median 9.5%" uses datasheet host ceilings and the authors' own traces, which Section 7 says understates published systems by the achieved-to-nominal host ratio; the headline card runs 1.21x the memory clock of 25 of 26 other rentals; "169 clauses are scored" omits that 46 failed (28% of scored; 33% in the post-review jobs). Put the bracketed audit number and the hit rate in the abstract.
(6) Workload: 30 AIME prompts x 256 tokens, also used during development; held-out checks are 10 problems. Partly fixable: one non-math prompt family on the six headline cells is a $3-5 job at this ledger's ~$0.77 per rental.
(7) Small: Table 3 "all" column prints 41 where 69.9/168 = 41.6; "at 50% about 34" is the measured W50 (33.6 in w50_exact.json) while the fitted law gives 24 at C/k=16, say which; the exponent without the six interpolated sub-token points is 1.51, outside the 1.09-1.47 interval, which undercuts the "law" language; Fig. 4's "lead grows with CPU-to-link ratio" rests on two x-clusters (1.0-1.3 and 2.96, five hosts) and omits the 4090 point (ratio ~1.5, lead 1.30-1.37) that would help.

5. QUESTIONS
Q1. If the calibrated residual is left as an unattributed sixth term, what are the Shapley shares, and does "foresight or overlap is largest in every order" survive?
Q2. Can the law predict FreeToken's six headline cells from its own fetch/miss counters? If its counters are unavailable, say so in Section 3.
Q3. How much of the "overlap" value is CPU rebalancing versus overlap proper, given the profile's 5-10%?
Q4. Were half-life 16, kappa, g and L_c chosen on the same 30 AIME prompts the headline table reports?
Q5. For the second-host rerun, would a prediction written from the law with that host's own probe have landed inside the bands that failed (5 of 6 ratio cells, 4 of 6 speed cells)?

6. SINGLE MOST VALUABLE CHANGE
A rewrite, not more data: lead with Fig. 1 and Table 1 as the story (limit, accounting, what traces say), state in Section 4 that GPU-efficiency is the closing residual and overlap includes rebalancing and reconcile with the profile, compress Section 5 to one page plus Table 8, remove job IDs from the main text, and put the prereg hit rate and a bracketed audit figure in the abstract. Estimated grade after: 7/10 (clear accept range), with clarity 2 -> 4 and soundness 3 -> 4 from the honest framing alone.

7. VERDICT
Not a 9. It is a 6. What separates it from a 9: (i) a law validated across engines and a second model, not one engine on one model, with FreeToken predicted; (ii) an accounting whose terms are measured rather than closed by a per-cell residual, agreeing with the profiler; (iii) a system that closes a meaningful part of the gap it diagnoses (a foresight mechanism moving 25-42% of the limit toward 60%), instead of a cache 3-29% faster than FreeToken on one model and tied on another; (iv) a second workload; (v) prose a committee member can read in forty minutes. Items (ii), (iv) and (v) are reachable before 30 Oct with the credit left; (i) and (iii) are not, and they are what a best-paper needs.

---

## Review 2 — performance-modeling professor

(verbatim)

**1. Summary.** The paper models batch-1 decode of host-offloaded MoE models on one consumer GPU as a fixed GPU term plus host bytes over the probed rate of the read paths in use (Eq. 1), tests the law blind on 7 hosts (3.7% median error) and leave-one-host-out against additive and per-layer alternatives. It derives a per-layer Belady-with-bypass speed limit in seconds (Eq. 2) from a max-of-six-samples probed host rate and a datasheet GPU rate, tightens it four ways (Table 3), and Shapley-decomposes the gap between its llama.cpp expert cache and that limit over five modelled "fixes". Nine-model trace studies show online policies read 35–110% more than the optimum, four tokens of foresight beat all of them, and batched speculative verification is not foresight; an audit scores 52 published measurements against their own limit (trace-scored median 9.5%). The cache leads FreeToken by 15–29% (gpt-oss) and 3–15% (Qwen3) on the headline machine, ties on Qwen3.6 and Mixtral, and 169 pre-registered clauses are scored (91 held, 25 on the point, 46 failed).

**2. Scores (1–5).** Modeling rigor 3.5 — one fitted constant plus probed inputs, tested blind and out-of-sample, is real modeling; but the constant carries a byte-proportional shortfall and the form is contested in the two-path regime. Bound quality 3 — well-defined for the per-layer-C class, honestly tightened, but overclaimed in the abstract, loose by ~2× at ≥25% budgets (datasheet GPU), and "exceedable" by construction (probed host max). Benchmarking practice 4 — base cases with absolute numbers, paired bootstrap, harmonic-mean check, selection launch separate from scoring launch, everything released; intervals cover the wrong variance for cross-machine claims. Insight per page 4. Writing 2 — dense, number-choked, job IDs in the main text, a 400-word abstract that is a list of ranges. **Overall 6.5/10. Borderline accept / major revision at MLSys.**

**3. Three attacks.**

*(a) The law's structure, not its error.* Section 3: "G is one constant per engine and model, fitted once … (4.82 ms)" and then "G (5.17 ms refit) exceeds the whole all-in-VRAM token (3.92 ms) because it absorbs a shortfall the probe does not see: the helpers read at 1.09–1.31× the probe's time per expert". A "GPU term" larger than the entire all-in-VRAM token means Eq. 1's attribution is wrong: a byte-proportional host shortfall is carried by a constant, which only holds over the narrow X_c range tested. The paper's own data supports the attack and admits it cannot be resolved at n=33 ("a fitted rate factor is collinear with G"; LOHO A×k gives k=0.97). The max-vs-additive contest (3.1 vs 4.5%) rests on 3 fast-link FETCH rows, since 19 of 29 configurations use one path; on slow links the max form over-predicts +11.7/+11.4% (frozen) and the paper's own asymmetric-sharing variant E has the best p90 (6.3 vs 9.4%, prereg/perlayer_model_outcome.md) but is relegated to one clause. Median error is the wrong scorer for a law whose job is to choose the split on hard machines; Eq. 3 "overstates differences about 1.8×". Verdict: rank-correct, magnitude-loose; "the form is settled" is not supported.

*(b) The bound's class and ceilings.* Abstract: "bounds any exact-routing system at a given expert budget." Body: "at most C experts of each layer resident." Table 3's pool column (182 vs 172, 474 vs 430, 101 vs 96, 234 vs 214 tok/s) shows a same-memory cross-layer allocation beats the published limit by 5–10%, so the abstract's sentence is refuted by the paper's own table; the body is scoped correctly. Also Section 4: "It can exceed the all-in-VRAM speed, because the CPU's bandwidth adds to the GPU's" — the CPU adds ≈9% (279 vs 255 tok/s under the measured GPU rate); the published 519 vs 255 at gpt-oss 40% is the datasheet assumption (52–61% achieved at batch 1). The bound mixes a probed host ceiling (max of six samples, 8% above their median; a second rental probed 90.3) with a datasheet GPU ceiling on a card that actually runs 1.21× datasheet. The author's own outcome note says at ≥25% "% of limit mostly restates the batch-1 kernel efficiency".

*(c) The accounting and the audit juxtaposition.* Abstract: "GPU kernel efficiency, replacement policy and host work are never largest in any order." Section 4 concedes that charging the all-in-VRAM shortfall in full makes GPU efficiency largest in 40 and 15 of 120 orders at the GPU-bound cells — I recomputed both from prereg/shapley_gap.json, and at those same cells foresight is still largest in 55/120 orders against overlap's 65/120, so the "two regimes" are softer than the text reads. Deeper: all 30 interior coalition values are model outputs (scripts/shapley_gap.py); eta is calibrated per cell so v(none) reproduces the measurement, making "GPU efficiency" a residual by construction; and ~10 points of "policy and read paths" (split6 variant: policy 0–4.5%, paths 6.6–12.7%) is the gap between the probe's max sample (87.5) and the median zero-copy pair (77.7) — not a fixable item. No interior state is measured. Finally the abstract places "median 9.5% of their own limit" (datasheet ceilings; imputed DRAM bands — 45 of 57 audit rows have no measured host bandwidth) beside "our cache 25–42%" (probed), which Section 7 itself calls "not like for like".

*Numbers checked:* Shapley shares in exact_max (foresight 43.3/36.6/52.7/44.7% → "37–53%"; overlap 39.4/45.7% → "39–46%"; policy/paths 12.0–15.6; host work 2.9–9.7; GPU 4.4–15.8; 0.141 ms tie at gpt-oss 25%); Table 3 vs speed_limit_v2 outcome ("all" 41/39/55/43/33/57); scorecard totals (20+23+43+83=169; 91/25/46/3/4 sum per block); grid 25–46% (min gpt-oss 25% listing B, max Qwen3 12.5% on the 4090); audit class stats (9.5, 4.9–17.8, 36.8, 34.8); W50 fit from w50_exact.json (0.589, 1.328, r 0.971, CI 1.09–1.47, LOMO 1.24–1.37, 6 sub-token points, 1.51 without them — note the exponent CI nearly includes the natural null of 1). One basis slip: the abstract's 3.7% is the per-configuration median (29); the per-measurement median (33 rows) is 3.3%, the number the LOHO note uses.

**4. Taste.** Largely yes. Asking "how fast could it go" before "how fast does mine go", a bound beside every number, a model tested blind on machines the author had never touched, 169 committed clauses with 46 failures printed, "speculation is not foresight" as a negative, a Mixtral tie reported as scope, all for $40.6 — that is the Hoefler–Belli ethic executed alone. Where it falls short: (i) foresight is named the largest fixable cost and never measured, although the engine records routing lookahead and could replay a W-token oracle — the loop is left open at the one place that matters; (ii) editorial judgment — seven contributions, job numbers in prose, an overclocked rental as the headline machine, the stock-clock host (where FreeToken wins a cell) demoted; (iii) the body's honest caveats do not reach the abstract — all three attacks are abstract-vs-body; (iv) choosing median error and datasheet ceilings where p90 and measured ceilings are the harder test.

**5. To a 9.** Doable in 4 weeks, ≤$17: oracle-lookahead replay in the engine at W=1,4,16 on three cells (teacher-forced, instrumentation exists) — turns the Shapley foresight share into a measurement (~$5–8); a serialised-execution flag to measure v(none) and one interior state (~$2); pooled optimum and measured-GPU basis as headline Table 1 columns, abstract sentence fixed ($0); use the card's measured device bandwidth (1694 GB/s already recorded, 089 P1b) and account for listing B's 1.21× clock ($0); headline on the stock-clock host ($0); p90 and per-regime law error, promote variant E or state the regime split ($0); audit on an achieved-bandwidth basis using the author's own 8-host probe ratios, drop the 9.5-vs-25–42 juxtaposition ($0); restructure law → limit → accounting → traces → instrument, system to 1.5 pages, Qwen3.6/Mixtral/competitors to appendix, abstract ≤200 words ($0, hardest); two more rentals of the headline cells for a machine-level interval (~$6–10). Not doable: a foresight mechanism (decoupled router/forecaster) that recovers the gap; locked clocks; a second 100B-class model family; batch>1.

**6. Verdict.** 6.5/10, up from 4.8 on substance (LOHO, Shapley, Table 3, scorecard are real), held down by overclaims the paper itself contradicts and by writing that hides a good modeling paper inside a compendium. Scope the bound, measure foresight, lead with the representative machine, and cut a third of the text: that is an 8. Close the loop on foresight with measurement and it is a 9. As a supervisor I would take this student today; as a reviewer I would not accept this draft without those changes.
