# Review 22b: "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth"

Referee: professor, systems group (scientific benchmarking, in the spirit of Hoefler and Belli, SC'15). Date: 8 October 2026.

## Materials read

- `paper/paper.pdf` (44 pages, built 8 Oct 04:00 CDT). I read the main text in full (Sections 1 to 11 and Tables 1 to 6). I read Appendices A to D in full and Appendices E to L in the parts that bear on the claims below: E, G, H, I and J in detail, F and K partly.
- `paper/supplement.pdf`: its structure, the scorecard rules, and the clause tables for jobs 109 and 112. I did not read every clause.
- `paper/paper.tex`: structure only (inputs and table placement), plus a grep for "i5-12400"/"091" across `paper/*.tex`.
- Artifact files used as inputs:
  - `gpu/vast_ledger.json` (145 rentals);
  - `prereg/gpu_pushes.json`;
  - `prereg/decomp_measured.json`, used only to find the cause of a mismatch that turned out to be my own loader bug (see check 10);
  - `prereg/value_map.json`;
  - `prereg/scorecard_clauses.json` and `prereg/scorecard_0xx/1xx.json`, for tallies only.
- Raw-data checkout `/home/claude/gpu-branch`:
  - job-script headers for jobs 099, 102, 105, 106, 109, 110, 111 and 112, with their per-host stubs;
  - `git log` dates and file-content diffs of those scripts;
  - `results/<job>/` for jobs 089, 096a/b, 099a–j, 100b/f, 101b, 102a–c, 103–112. From these I used `ec_*.jsonl` rows (decode_ms, n_decode, nll_sum, nll_n, config/stats path), `st_*.json` counters, `concur.txt`/`concur2.txt` probes, `fetch_table*_law_gptoss.json`, `g_prof.json`/`prof_*.json` (Nsight), `manifest.json`, `DONE`, `v0.txt`/`gate.txt`, `bs1.jsonl` (job 089) and `readsched_*.txt` (job 102).
- GitHub's own repository activity API (`gh api repos/.../activity?ref=refs/heads/gpu`): push timestamps and before/after SHAs only. It returns no commit messages.
- My scripts are in `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/rev22b/` (`load.py`, `regtime.py`, `tab6.py`, `tab5.py`, `tab4new.py`, `panel.py`, `trend.py`, `v1check.py`, `rerent.py`, `few1r.py`, `eq3.py`, `implied.py`, `win.py`, `tab3.py`). I ran them all from that folder.

## Independence statement

- **What I did not open.** Nothing under `reports/` except to write this file. No `prereg/*outcome*.md`, nothing in `research_notes/`, no file named like a review, number check, plan or progress log, not `paper/paper_v1_prereview.tex`, and nothing in `apply/`.
- **No commit messages.** I used `git log` only with date/hash formats, and file-content diffs (`git log -p` on job scripts). The GitHub activity call returns timestamps and SHAs only.
- **Incidental exposure.**
  - One grep across `paper/*.tex` printed two lines from `paper_v2_prerewrite.tex`, a file not on the excluded list.
  - The job 112 header, which is file content, mentions "reviews 21a and 21b". I did not open those reviews.
- **Own recomputation.** I wrote every number check myself from raw rows, counters and probes. I did not run the authors' analysis scripts.
- **No changes.** I modified nothing in either repository and committed nothing.

---

## 1. Summary

The paper studies batch-1 decode of two MoE models whose experts mostly live in host DRAM: gpt-oss-120b (MXFP4) and Qwen3-30B-A3B (BF16). The workload is AIME-25, teacher-forced. The experiments run on rented RTX 5090 machines, with RTX 4090s used in two tests. It makes four claims.

1. **A bound.** It defines a per-machine speed bound, Eq. (1): Belady's MIN-with-bypass host reads (R*) at the bandwidth probe's highest reading, max'ed with a GPU term. It adds a "demand" bound, Eq. (2), and an "ordered" bound, Eq. (3). The authors' llama.cpp expert cache runs at 31–54% of Eq. (1) at gpt-oss 11%. A microbenchmark reaches 86–96% of the bound's read time.
2. **A measured decomposition of the gap.** A 2×2 of admission set (deployed vs MIN) × reads per admission (two: CPU-serve-then-copy; one: fetch in step) is built from oracles inside the engine, with a read-ahead oracle on top. MIN's set and single reads "pay only together": a positive interaction that closes 33% of the gap on 15 panel machines and 35% on 5 new machines in a registered test. The remainder is attributed to the GPU's non-expert time in series with reads, plus the oracle's extra reads.
3. **No-foresight variants and a second card.** Four variants built from the engine's own mechanisms (larger admission margin, layer-ahead copy) capture ≤6% of the oracle's gain on fast-link machines. A log-linear trend in the link-to-CPU ratio, frozen on the RTX 5090 panel, predicts the oracles' speed on five RTX 4090s within ±0.10 in log.
4. **Price of foresight.** In trace replay, half of MIN's gain over admit-every-miss needs about 0.65·C distinct experts of lookahead per layer. This part is post hoc, across nine models.

Methodologically, the paper commits predictions in each job script's header before the machine starts and scores every clause (1,672 in all). It reports every rental, including failures, in a ledger, and treats the machine as the statistical unit.

## 2. Strengths

1. **The numbers are real and reproducible from raw data.** I checked 34 quantitative and methodological claims (table in Section 9). Every table cell I recomputed matched within rounding:
   - Table 3 (host S); Tables 4, 5 and 6; Table 29;
   - the frozen trend coefficients and their bracketed predictions;
   - the residual and Eq. (3) shares, the scorecard tallies, the ledger cost.

   I found two small mismatches and four minor imprecisions, listed in Section 9. None changes a conclusion. This is the best number-to-raw-data fidelity I have seen in a systems submission.
2. **The registration is verifiable, not just asserted.**
   - Commit before rental: for all 145 Vast rentals of jobs 058–112, the job script's last commit precedes the rental. For the 92 rentals of jobs 093–112 the minimum lead is 2.7 s, exactly as Appendix D states.
   - Push before job start: every machine whose manifest records a start began its job at least 26 s after the push. Seven pushes fall within 1 s of the rental's creation, and four are up to 0.8 s after it, both as stated.
   - Independent push record: all 268 push timestamps in `prereg/gpu_pushes.json` match GitHub's own activity log to the second.
   - Amendments after launch: I diffed the main scripts of jobs 102, 105, 106 and 111. Each amendment only substitutes or relaunches a machine, says "predictions unchanged", and is disclosed in Appendix D. Job 100's ninth prediction and job 107's mid-job replacement rule are disclosed as post-data amendments.
3. **Gates are code, and failed hosts are reported.**
   - The validity gates (V0, V0c, VR, V1–V3) are enforced in the scripts. Exit codes and gate messages in `results/*/DONE` and `v0.txt`/`gate.txt` match Appendix D's account:
     - job 107: 3 gate stops, 2 round-check failures, 2 valid;
     - job 108: 2, 1 and 3;
     - job 110a stopped at ratio 0.2499;
     - jobs 105d and 105g stopped at the throttled-card gate.
   - Unsteady machines are kept in the tables with markers rather than dropped. The "lost on two" sentence for Few-1R is exactly right: the two unsteady machines with the slowest links (ratios 0.14 and 0.21).
4. **A proper factorial inside a real engine.**
   - The 2×2 is measured, not modelled. Both orders are drawn in Fig. 2, and the interaction is registered and replicated on never-rented machines.
   - Using oracles inside the engine to split the time is the right instrument for this question. The paper is careful to say that the parts left after the oracles are attributed rather than measured.
5. **Sound statistics for the setting.**
   - The machine is the unit: t-intervals over machines, and paired bootstraps over problems within a machine. Per-machine values are always shown beside summaries.
   - The paper quantifies why the machine is the unit: machines carry 96% of the variance of MIN-1R's log gain, and same-CPU machines differ by up to 29%, which I confirmed. It also shows within-machine re-rental stability: 0.7% on the two re-rented RTX 5090s, which I confirmed.
6. **Honesty about failures.** The paper reports:
   - that 321 of 1,672 clauses failed;
   - that Eq. (4), the closed-form account, failed three registered tests and was inconclusive in a fourth;
   - that the registered timing-control clauses failed;
   - that the trend misses 5 of 18 cells on new RTX 5090s;
   - that job 109's bands were too wide to fail.

   The Limitations section is concrete and quantitative.
7. **Useful framing.** Distance to a machine-specific bound is a better yardstick for builders than speed-up over a chosen baseline (Hoefler–Belli rule 11). The link-to-CPU ratio is a cheap, actionable predictor.

## 3. Weaknesses, most important first

**W1. The bound, and every share of the gap, rests on a single probe that the engine sometimes beats.**
- Bhost is the highest reading of one probe run, which on most machines ran while the model was downloading. I recomputed Eq. (4)'s implied read rate on the panel and job 109.
- On 100f, a 9950X with a "slower link" and the 54% top of the paper's 31–54% range, the deployed cache reads at 1.16× (11%) and 1.22× (25%) the probe's best. Four other panel launch-budgets exceed it by 4–7%.
- The paper discloses this (12 of 84 launch-budgets over 5%; Table 27). However, every number in Table 4 (shares of the gap) and every "% of bound" is computed against this Bhost, and Table 4 has no sensitivity analysis.
- The second probe exists only for the last four machines (job 112). It agreed within 4.9%, which is reassuring but covers only 4 of roughly 40 machines.
- Calling Eq. (1) a "bound" is defensible only as "relative to the probe". The abstract and Section 8 rule 1 ("Report the distance to the machine's bound") do not carry that qualifier.

**W2. The registration's confirmatory weight is uneven, and Table 1's "pre-specified" label does not show it.**
- Of 1,672 scored clauses, 588 (35%) held with an interval, 679 (41%) held on the point estimate only, and 321 (19%) failed. I reproduced these tallies.
- Several headline rows of Table 1 rest on weak or failed tests:
  - *"MIN's set and one read pay only together (fast links)."* Job 099 registered a positive interaction "on every host". It failed on the panel: 099f had an interaction of −0.48 ms at 11% (9/10). The 0.5 "fast-link" line was drawn after the panel data. The confirmatory evidence is therefore 5/5 at 11% (and 4/4 at 25%) on job 109's machines. That is fine, but Table 1 should say "failed as registered on the panel (9/10); population restricted post hoc; confirmed 5/5". The "15 + 5" count also includes the two slow-link panel machines that the claim excludes.
  - *"The bound's read time is nearly reachable."* Registered at ≥0.50 per layer and ≥0.80 per token. No clause had an interval (all 28 "held (point)"). Three machines. The claim's wording ("nearly", 86–96%) is descriptive, not confirmed.
  - *"Policies without foresight recover little."* Registered as capture ≤0.35 and best ≤1.10. The authors themselves note that this "could hardly have failed".
- The bold run-in labels in Sections 4 and 5 ("registered: a positive interaction on every machine"; "registered: gain bands per budget") state what was registered, not what happened. A skimming reader infers that they held. Section 7's label ("registered at 0.80 or more; the lowest machine missed") is the right model.

**W3. The second-card test is less blind than Table 6 suggests.**
- **Relaunched machines.** Three of the five RTX 4090 points (job 111) are relaunches of job 110 machines, made after their first-round results were in the repository. The paper says so in a footnote. V0 of job 111 even requires the GPU to be one of job 110's.
- **The machine not relaunched.** Job 110e was "left out, an oversight". It had the worst first-round read-ahead deviation from the trend: −5.6%, against −0.0 to +6.2% for the others. Including it would not have failed the band, but it should be in Table 6 or Appendix D.
- **A machine rented before.** Job 112a, the Core i5-12400 at ratio 0.77 that carries the "higher ratios" half of the test, is job 091's machine (offer 49588631). Its header says so, but the paper does not. Table 6's caption says "machines rented for them".
- **What the trend does on same-card new machines.** Applied to new RTX 5090s, the same card it was fitted on, the trend misses 5 of 18 cells by up to 18% (I reproduced 17.6%). The fair reading is: "the ratio sets the sign of the gain and, on these five RTX 4090s, its size within ±10%". The paper's "a trend in that ratio predicted the oracles' speed on a second card" (abstract, Section 8 rule 4, Conclusion) is stronger than that.

**W4. The attribution of "what is left" uses one RTX 5090 constant for every machine and both cards.**
- **The constant.** T_GPU = 2.94 ms is the smallest RTX 5090 non-expert profile. From the Nsight categories I estimate the non-expert time as G minus expert GEMV minus ec_ctl. That gives 2.91–3.28 ms on RTX 5090s, consistent with 2.94 as the minimum, and 3.41–3.56 ms on the RTX 4090s. The RTX 4090 profiles' G, 5.1–5.4 ms, is not reported anywhere I could find.
- **Effect on the RTX 4090 residual.** With each RTX 4090's own estimate, the residual falls from 31–36% to about 27–31% (111d 30.7%; 111f/g 26.6%).
- **Over-attribution at 25%.** The residual is significantly negative there: −5 [−7, −3] on the new machines and −5 [−12, 2] on the panel. The named parts over-explain, which the main text does not mention ("two named parts account for it").
- **Using the minimum.** The minimum is the right choice for a bound but biases an attribution. Per-machine G is available for every machine from job 105 on.

**W5. Generalisation runs ahead of the evidence in the abstract and Section 8.**
- "Where the link reads at least half as fast as the CPU, policies without foresight recover at most 6%" (abstract). The test covered four variants built from two engine mechanisms, on 5 machines, at two budgets. Section 6.1 states the scope correctly; the abstract and rule 3 ("Do not expect a policy without foresight to get there") do not.
- "Pay only together" holds at 11%. At 25%, MIN's set alone closes 21% of the gap (Table 4).
- The "two-read" arm is this engine's serve-then-copy path, with its fetch table. The interaction is a property of that design as much as of caching in general.
- Everything is measured in one engine, on two models, at batch 1, on a prompt set also used during development.

**W6. The timing control failed its manipulation check.**
- Few-2R-early was meant to land copies in time. In the engine its misses fell only to 0.97–0.98 of Few-2R's against a registered ≤0.95, because the fetch table keeps fetching 17.8–23.7 misses per token into slots the schedule did not choose. I reproduced both numbers.
- The experiment therefore did not realise "landing in time". The conclusion "the late landing is not what holds the two-read arms back" (Section 4; Table 1 row 5) rests on the post-hoc explanation through the fetch table, not on the control.
- An arm with the fetch table forced to all-CPU, or with the plan's slots pinned, would test the intended manipulation.

**W7. Smaller accuracy and reporting items.**
- Section 6.1: "they still read 1.66–1.68 R⋆ host experts per token". That holds only for LA-1R and LA-1R-margin. Margin, the best variant on 3 of 5 machines, reads 1.57–1.59 R⋆, and LA reads 1.75 R⋆ (deployed: 1.65–1.66).
- Appendix L's cost covers jobs 058–108: 124 rentals, $87.2, which I confirmed. The full ledger is 145 rentals and $105.8 at GPU-hour prices.
- `manifest.json` does not record the git HEAD the machine cloned. The ordering argument rests on timestamps, which do support it, but one line in the boot script would make it airtight.
- Table 3 host S uses launch 2 throughout. Launch 1 differs by up to 0.012 in the ratio over FreeToken, which is comparable to the CI half-widths. The choice is not stated.
- The footnote's "first rounds … within 6% of the trend": the largest deviation is 6.2% (110b read-ahead; 0.061 in log).

## 4. Clarity

Earlier rounds rated clarity 2 and then 3. My independent assessment is still 3. The main text is now compact: about 6,900 words, with a median sentence of 17 words. The section titles are questions that map onto the contributions, and the prose is precise. What keeps it at 3 is the reading apparatus around that prose.

**What works**
- The question-shaped section titles, and the Introduction's one-paragraph answers keyed to them.
- The explicit "Terms" and "Yardsticks" paragraphs in Section 2. These are the right idea.
- Fig. 1's schematic of the two reads; Table 2 with a "Reads" column.
- The running example in Section 5 (9950X: 10.0 / 13.0 / 20.6 / 15.2 / 13.8 ms), which grounds the shares. I verified its numbers.
- Run-in labels "(registered …)", "(post hoc)" and "(exploratory)" on paragraph heads. Readers can see the evidence type in the flow.
- A concrete, quantitative Limitations section; Section 8's five rules are actionable.

**What still makes it hard to read** (concrete)

1. **The main-text tables are after the bibliography.** Tables 1–6 sit on pages 13–16, after the references (pages 9–12). Table 1, the claims map, is cited in the Introduction's second paragraph. Tables 4–6 carry the paper's central results. A reader must flip four or more pages for every result. This is the single largest readability cost in the current version, and possibly a page-limit issue as well.
2. **Two naming systems, and too many names.** The main text uses about 15 configuration names: deployed cache, Dep-1R, admit every miss, Margin, LA, LA-1R, LA-1R-margin, MIN-2R, MIN-1R, Few-1R, Few-2R, Few-2R-early, read-ahead oracle, Belady-1R/2R, window W, recall r and deployed + window. The appendices add roughly 30 codes: base, foa, aa, dk, pf, R1, R2, fetch, bypass, fetchplan, bypassplan, bypassplanS, both2, both3p, hitopt(p), nb2, lead2, w1…w16, b4, b16, b8r5, lrn, learned. There are also host labels A, B, S, O1–O6, Pa–Pj and "Pf again". Table 8 maps the codes, but appendix tables still mix them (Table 24's "lrn"), and the prediction log quotes code names inside clause text.
3. **Five yardsticks with three baselines.**
   - Share of the bound's speed (31–54%); share of the gap (35%); speed vs the deployed cache (1.22×); capture, a share of the read-ahead oracle's gain (≤6%); and in Section 7, share of MIN's gain over *admit every miss* (0.31, 0.29).
   - The same token "35%" or "0.31" means different things a page apart.
   - Fig. 5's caption leads with "Baseline here: admit every miss", which helps, but the text of Section 7 also quotes the deployed-cache yardstick ("0.89–1.32× at 11%").
   - A one-row-per-yardstick table (formula, baseline, where used) would remove most of this.
4. **Four equations and seven bandwidth symbols.** Eq. (1)–(3) bounds and Eq. (4)'s closed-form account; B_c, B_p, B_cp, B_host, B̂_c, B̂_p and B_gpu, the last at datasheet or measured rate. Table 3's caption adds a "pooled bound". Section 3 introduces Eq. (3) with B̂_c/B̂_p in one dense paragraph, but Eq. (3) then matters only for one slow-link paragraph in Section 4.
5. **Run-in labels state the registration, not the outcome** (see W2). Where a registered claim failed and was rescoped, as with the interaction, the reader needs the outcome in the label.
6. **Table 1 is typeset in a narrow column with wrapped cells.** The "Test and outcome" column mixes registration, outcome and post-hoc caveats in one sentence fragment. "Mach." means machines in one row, hosts B and S in another, and "15 + 5" in a third. I would put Table 1 full-width on page 2, with separate columns for "registered test", "outcome as registered" and "post hoc qualification".
7. **Section 4's Few-2R-early paragraph** packs into one paragraph:
   - five ratios; two counters; a registered threshold and its miss;
   - a mechanism explanation (the fetch table);
   - and the conclusion.

   It is the hardest paragraph in the paper. The result is in Table 5, so the text could give the logic in three sentences.
8. **Appendix D is chronological and exception-laden.** It is honest but hard to use. Two tallies follow different scoring rules: hand-scored 565 with deterministic counts held outright, and script-scored with those counts "held (point)". Per-job narratives interleave reruns, voids and amendments. There is no index from each Table 1 row to its clause IDs (job/P#), so a reader cannot go from a claim to its test in one step.
9. **Section 5 is two short paragraphs that act as Fig. 3's caption, and Section 6 joins two unrelated studies.** Moving the Few-1R paragraph into Section 4, after the 2×2, and making Section 6 two sections would match the Introduction's ordering.
10. **Prose that should be tables.** Running text carries many paired ranges, e.g. "17.8–23.7 of each token's 53.0–54.6 misses" and "0.98–1.00× … 1.03× … 1.21–1.41×". Where a table already holds them, cite it and give one number.

What would move clarity to 4:
- tables in the body (fixing item 1);
- one naming scheme throughout, appendices included;
- a yardstick table;
- outcome-bearing run-in labels;
- a claim-to-clause index for Table 1.

## 5. Methodology assessment against the brief

- **Statistical unit.** Correct and well argued: machine as unit, t-intervals over 4–15 machines, paired bootstraps over problems within a machine.
  - The t-interval assumes a population. The machines are a convenience sample of the cheapest offers that passed the gates, so the intervals describe "machines like these on Vast this week". This is stated.
  - Pooling 30-problem and 20-problem runs, and three engine versions, in the panel is disclosed.
- **Registration practice.** Exemplary in its mechanics, verified as described in Strength 2. The weaknesses are in content, not mechanics:
  - wide bands;
  - a large "held (point)" share;
  - "every host" predictions rescoped after failure;
  - relaunches with knowledge of first-round results;
  - one previously rented machine in a "rented for the test" population (W2, W3).
- **Validity gates and failed hosts.** Registered and enforced in code. Failed and unsteady hosts are reported with their numbers (Tables 24–26, Fig. 7 markers), not dropped. I confirmed every gate outcome I checked.
- **Evidence labels.** Table 1, Tables 5 and 6 and the prediction log are internally consistent with the data. Table 1's labels need the refinements in W2 and W3.
- **Separating post hoc from registered.** Good in the run-in labels and in Table 1's evidence column. Two places blur it:
  - the interaction claim's "fast links" scope (post hoc) under a "pre-specified" label;
  - the 13-machine Few-1R summary, which pools registered-on-7 with other launches and with both cards.

## 6. Questions to the authors

1. How do Table 4's shares change if B_host is replaced, per machine, by max(probe, the rate Eq. (4) implies), or by an idle repeated probe? What does 100f's row become?
2. Why does Table 4 and the RTX 4090 residual use T_GPU = 2.94 ms rather than each machine's (or each card's) profiled non-expert time? Please report the RTX 4090 profiles (G ≈ 5.1–5.4 ms).
3. For job 111: was there a pre-stated rule for relaunching after a card-specific V1 failure? Would you have relaunched if the first rounds had missed the trend? Please add 110e's first-round numbers to Table 6 or Appendix D.
4. Please state in the paper that job 112's i5-12400 machine had run job 091. Does its prior use (probe and grid known) affect the claim that it was "rented for" the test?
5. The abstract's claim about "policies without foresight" covers four variants of two mechanisms. Would you narrow it, or add the learned reuse predictor (job 106) and the forecaster-driven window to the same capture table?
6. Few-2R-early did not land MIN's set in time. Can you run it with the fetch table at all-CPU, so that the landing time is the only change?
7. For each Table 1 row, which clause IDs carry it? Would a naive predictor (no change, or the panel mean) also have passed each band?
8. Why is the "fast-link" threshold 0.5? How sensitive is the interaction claim to 0.45 or 0.55, given that 099f (0.29) and 099d (0.40) are the only panel machines below it?
9. Can each `manifest.json` record `git rev-parse HEAD` and the job script's sha256?

## 7. What would raise my score

- **Soundness to 5.**
  - A Table 4 sensitivity analysis to B_host, and an idle second probe as standard.
  - Per-machine or per-card T_GPU in the attribution, with the 25% over-attribution discussed.
  - The Few-2R-early manipulation repaired or reframed.
- **Methodology to 5.**
  - Table 1 relabelled with outcome-as-registered and post-hoc scope.
  - 110e disclosed in Table 6, and 112a's prior rental disclosed.
  - A claim-to-clause index.
  - A one-line count of informative versus "could hardly fail" clauses per Table 1 row.
  - HEAD SHA in manifests.
- **Clarity to 4.** The five changes listed at the end of Section 4.
- **Significance.** A second engine, even a minimal one, or a batch-2 point would show the interaction and the ratio trend are not artefacts of this one serve-then-copy design. Alternatively, frame the paper explicitly as a case study of one engine family.
- With W1–W3 addressed and the clarity changes made, I would move to 8.

## 8. Scores

| Criterion | Score |
|---|---|
| Overall (1–10) | **7** (accept with revisions; above the bar on methodology and reproducibility, held back by scope, presentation and some over-generalised claims) |
| Soundness (1–5) | **4** |
| Methodology (1–5) | **4** |
| Significance (1–5) | **3** |
| Clarity (1–5) | **3** |
| Confidence (1–5) | **4** (I recomputed the central tables from raw data; I did not re-derive the trace study of Section 7 or the published-systems audit) |

## 9. Claims checked against raw data

Data sources:
- **R**: `results/<job>/` rows, counters, probes and profiles;
- **J**: job-script headers and diffs;
- **L**: the ledger;
- **G**: git dates and GitHub's activity log.

Verdict key:
- ✓ matches within rounding;
- ≈ close but not exact;
- ✗ incorrect or imprecise as written.

| # | Claim (location) | Paper says | My recomputation (source) | Verdict |
|---|---|---|---|---|
| 1 | Registration timing (App. D) | 92 rentals of 093–112; commit ≥2.7 s before rental; job start ≥26 s after push; 7 pushes within 1 s of rental; 4 up to 0.8 s after | Exactly: 92; min 2.71 s; min 26 s; 7; 4 (max 0.83 s). Also 53 rentals of 058–092: min 3.5 s (G, L, R manifests) | ✓ |
| 2 | Push records are genuine | `gpu_pushes.json` | 268/268 timestamps identical to GitHub's own activity API (G) | ✓ |
| 3 | Post-launch amendments leave predictions unchanged (App. D) | 102, 105, 106 substitutions; 111 relaunch | Diffs touch only host lists and relaunch text; "Predictions unchanged" (J) | ✓ |
| 4 | Gate outcomes (App. D, I) | 107: 3 gate, 2 round, 2 valid; 108: 2/1/3; 110a ratio gate; 105d/g throttled | `DONE` rc codes and gate messages match (R) | ✓ |
| 5 | Table 6, job 109 (5 machines × 2 budgets) | e.g. 285K 1.22/1.46/1.02/4%; 5950X 1.37/1.50/1.01/2% | 1.215/1.460/1.021/4.5%; 1.366/1.500/1.010/1.9%. All 38 cells match (R) | ✓ |
| 6 | Table 6, RTX 4090 jobs 111/112 | 14900KF 0.96/1.13/1.03/26%; i5 1.31/1.46/1.01/3% … | 0.963/1.128/1.033/25.7%; 1.312/1.465/1.013/2.9%. All cells match (R) | ✓ |
| 7 | Table 6 bracketed trend predictions | e.g. [0.99],[1.11]; [1.25],[1.37] | exp(a + b ln r) with job 110 header coefficients and probe ratios: identical (J, R) | ✓ |
| 8 | Frozen trend fitted on the 15 panel machines (job 110 header) | 8 coefficient pairs | Refit from raw rows: identical to 4 decimals (R) | ✓ |
| 9 | Trend on RTX 4090s: within 7% (11%), 5% (25%); all inside ±0.10 log (Sec 6.2) | 7%, 5% | Max 6.6%, 5.0%; 20/20 cells inside (R) | ✓ |
| 10 | Table 4 panel (15), both budgets, 8 rows | 11%: 0/0/33/15/52; 27/21/4. 25%: 21/−3/31/21/48; 34/19/−5 | 0.5/−0.0/32.6/14.9/52.5; 27.4/21.4/3.7. 20.6/−3.2/31.3/20.6/48.1; 33.9/19.1/−5.0. CIs match. My first pass mismatched at 25% because my loader merged Qwen3 C32 rows into gpt-oss C32 in jobs 096a/b; after fixing it, all match (R) | ✓ |
| 11 | Table 4 new (5; 4 at 25%) | 3/0/35/19/46; 28/18/1. 20/−2/27/24/48; 38/16/−5 | 3.4/−0.4/34.7/19.0/46.3; 28.2/17.5/0.5. 19.8/−2.4/27.3/24.4/48.3; 37.9/15.6/−5.1 (R) | ✓ |
| 12 | Interaction sign (Sec 4) | 9 of 10 on job 099; 5 of 5 new | 099f −0.48 ms, others +1.2 to +8.9; job 109: +1.5 to +7.5 ms (11%), +0.5 to +1.0 (25%) (R) | ✓ |
| 13 | Table 5 speeds | 1R 1.41/1.25/1.21/1.35; 2R 1.03; early 0.98/1.00/1.00/0.98 | 1.406/1.247/1.208/1.346; 1.027–1.030; 0.977/0.997/0.995/0.980 (R) | ✓ |
| 14 | Table 5 misses and Sec 4 counters | early/2R 0.97–0.98; early − 1R 12.1–13.8; fetches 17.8–23.7 of 53.0–54.6 | 0.968–0.980; 12.1–13.8; 17.8–23.7 of 53.0–54.6 (st_*.json) | ✓ |
| 15 | Job 112 T1/T2 failure counts (App. D) | failed 5/8 and 7/8 | 5/8 (four at 11%, 0.953 at 25%); 7/8 (only 7945HX at 25%, −0.007, passes) (R) | ✓ |
| 16 | Sec 6.1: best no-foresight variant 1.007–1.024×, capture ≤6% | as stated | 1.007–1.024; max capture 5.7% (R) | ✓ |
| 17 | Sec 6.1: RTX 4090 LA variants 0.64–0.72×; Margin 1.033–1.041×; up to 26% | as stated | 0.644–0.724; 1.033–1.041; 25.7% (R) | ✓ |
| 18 | Sec 6.1: "they still read 1.66–1.68 R⋆" | 1.66–1.68 | LA-1R/LA-1R-margin 1.66–1.68; Margin 1.57–1.59; LA 1.75; deployed 1.65–1.66 (st_*.json) | ✗ (imprecise) |
| 19 | Margin cuts reads 4–6% at 11% (Sec 6.1) | 4–6% | 4.2–4.8% (job 109) (R) | ✓ |
| 20 | RTX 4090 loss 3% higher; V1–V3 passed on valid hosts | 3% | 0.1963–0.1966 vs 0.190 (+3.3%); V2 ≤1.11%; V3 ≤0.68% (R) | ✓ |
| 21 | Job 110 first rounds "within 6% of the trend" (footnote) | 6% | Max 6.2% (110b read-ahead); 110e, not relaunched, −5.6% (R) | ≈ |
| 22 | Re-rentals (Sec 2; App. D) | RTX 5090 within 0.7%; RTX 4090 5.5%; ratio moved up to 12% | 0.71%; 5.44%; 11.8% (R) | ✓ |
| 23 | Second probe within 4.9% on all 4 (Limitations) | 4.9% | +0.8, +4.9, −0.4, −0.3% (concur2.txt) | ✓ |
| 24 | Few-1R on 13 stable machines, geometric mean 1.24×; lost on the two slowest-link unsteady ones (Sec 5; App. I) | 1.24× | 13 distinct GPUs (UUIDs), gm 1.240; losses 0.845 (r 0.14) and 0.974 (r 0.21) (R) | ✓ |
| 25 | Slow-link residual 35–47% (panel), 31–36% (RTX 4090); read-ahead leaves 83–99% / 73–81% of the gap to Eq. (3); fast median 45% (Sec 4) | as stated | 35.5/47.4%; 31.5–35.8%; 83/99%; 73–81%; 45.5% (R). With the 4090's own T_GPU the 4090 residual is 26.6–30.7% | ✓ (but see W4) |
| 26 | 5 new machines at 41–48% of Eq. (1); the 54% top is a machine the engine out-reads (Sec 3) | 41–48%; top too high | 41.5–48.0%; 100f at 54.0% reads at 1.16× (11%) and 1.22× (25%) the probe (R) | ✓ |
| 27 | Running example (Sec 5) | 10.0 / 13.0 / 20.6 / 15.2 / 13.8 ms | 10.02 / 12.96 / 20.64 / 15.16 / 13.81 (096a) (R) | ✓ |
| 28 | T_GPU = 2.94 ms is the smallest profile; G 4.1–4.7 ms (Sec 3; App. H) | as stated | Non-expert minimum 2.91–2.94 ms; G 4.07–4.54 ms on RTX 5090. RTX 4090 G 5.08–5.39 ms, unreported (prof_*.json) | ✓ / gap |
| 29 | Table 3 host S, all six rows | e.g. 28.9/48.0/57.9, 1.207 [1.195, 1.219], 2.00× | Identical for all 6 rows from launch 2; launch 1 differs by up to 0.012 (bs1.jsonl) | ✓ (launch choice unstated) |
| 30 | Rule 3: ratio of total time within 0.003 of ratio of mean rates (App. B) | ≤0.003 | max 0.003 (R) | ✓ |
| 31 | Host S vs host B: outside ±0.06 at 5 of 6; still faster at 11 of 12 (Table 1) | 5/6; 11/12 | 5/6 (only Qwen3 12.5% inside); 11/12 (R + Table 3's host B values) | ✓ |
| 32 | Table 29 microbenchmark: term (ms) and per-layer 86–96% | 7.04/9.70/9.56; 86–96% | 7.04/9.70/9.56 from the probe; 86.1–96.3% (readsched_*.txt) | ✓ |
| 33 | Sec 7: engine W = 16 recovers 0.78–0.93; next token ≤0.31; W = 8 at r = 0.5 0.29 ≈ W = 2 0.31 | as stated | 0.78–0.93 (job 099); replay 0.308 (Qwen3 12.5%), 0.290, 0.313 (R, value_map.json) | ✓ |
| 34 | Scorecard tallies (App. D; Table 10); ledger cost (App. L); same-CPU spread 29% (Sec 2) | 1,672: 588/679/321/84; 565: 218/193/142/8/4; 124 rentals, 74 offers, $87.2; 29% | Identical; identical; identical (stale: the full ledger is 145 rentals, $105.8); 285K 099a vs 099f 1.29× | ✓ |

Disclosure gaps found while checking (not numerical errors):
- 112a was job 091's machine (J, L).
- 110e's first round is not in the paper (R).
- The RTX 4090 Nsight profiles (G 5.1–5.4 ms) are not reported (R).
- No git HEAD is recorded in `manifest.json` (R).
