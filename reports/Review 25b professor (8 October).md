# Review 25b: "Where the Seconds Go" (professor, scientific benchmarking)

## Materials read

- `paper/paper.pdf` (45 pages): all of the main text, and Appendices A–N. I read it as extracted text, checking against `paper/main_body.tex`, `paper/paper.tex` and `paper/abstract_body.tex`.
- `paper/supplement.pdf` (the scorecard): the front matter, and the clause tables for jobs 109 and 113 (Table S19 and the 109 rows), which I reached by search.
- `paper/ieee-paper.pdf` (10 pages) and `paper/ieee-supplement.pdf` (30 pages): the full extracted text, plus rendered pages (contact sheets of every page, and pages 2 and 5 of the paper and 14 and 23 of the supplement at higher resolution). I also read `ieee-paper.tex`, `ieee-supplement.tex`, `ieee_preamble.tex` and the warnings in `ieee-paper.log` and `ieee-supplement.log`.
- Artifact files:
  - `gpu/vast_ledger.json`, the 147 rentals.
  - `prereg/gpu_pushes.json`. I cross-checked it against GitHub's own activity API for the `gpu` branch, which I fetched myself.
  - The docstring and the definitions in `scripts/decomp_measured.py`, the `profiles()` function in `scripts/sumlaw_paper.py`, and the docstring of `scripts/reg_timing.py`. I read these only to learn the definitions. Every number below comes from my own code.
- Raw results (`/home/claude/gpu-branch`):
  - Job headers `jobs/102, 105, 106, 109, 110, 111, 112, 113_*.sh`, with their per-host stubs.
  - `results/*/ec_*.jsonl`, `st_*.json`, `concur.txt`, `fetch_table_law_gptoss.json`, `g_prof.json`, `prof_*.json`, `bs1.jsonl`, `readsched_C*.txt`, `validity.txt`, `v0.txt`, `manifest.json` and `113_offers_at_launch.txt` for jobs 081, 089, 093–113.
  - `git log --format='%h %ct'` dates for every job script, and `git show --format=` diffs (file contents only) of the four amendments made after launch.
- My scripts are in `scratchpad/rev25b/`: `ec.py`, `c113.py`, `c109.py`, `c4090.py`, `cpanel.py`, `cfew.py`, `ctab3.py`, `cshare.py`, `ceq3.py`, `cprof.py`, `cnoise.py`, `regtime.py`, `cpush.py`.

## Independence statement

- I opened nothing under `reports/` except to write this file.
- I opened no `prereg/*outcome*.md`, nothing under `research_notes/` or `apply/`, no `paper_v1_prereview.tex`, and no file named like a review, number check, plan or progress log.
- I read no commit messages. I used only hashes and dates, plus `git show --format=` diffs.
- I modified nothing in either repository and made no commits.
- All analysis code is mine. It was written in a new scratch folder and run from there. I used the authors' scripts only to confirm definitions: the bound's R*, T_GPU, the gap shares and the push-time source. I did not use them to produce any number.

---

## Summary

The paper studies batch-1 decode of two MoE models with most experts in host DRAM, on rented RTX 5090 (and RTX 4090) machines. It does four things.

1. **A bound.** It defines a per-machine bound, Eq. (1): Belady-MIN-with-bypass host reads at the probe's best read rate. It adds two tighter bounds: Eq. (2) for reads that wait on the router, and Eq. (3) for reads that foresight cannot free.
2. **A decomposition.** It places its own llama.cpp expert cache at 31–54% of Eq. (1) on "consumer" machines. It then builds oracles into the engine to split the gap into:
   - *what* is cached (MIN's set);
   - *how* it is read (one read in the step, or a CPU read plus a background copy);
   - *when* it is read (copies issued ahead).
3. **Controls and a transfer test.** It registers predictions before each rental and runs a sequence of controls. The latest (job 113, two machines) shows that the earlier finding, "MIN's set pays only if read once," is caused by the engine's in-step fetches displacing MIN's set. It shows that four policies built from the engine's own mechanisms, without foresight, capture at most 6% of the read-ahead oracle's gain. A link-to-CPU-ratio trend frozen on RTX 5090s predicts the oracles' speed on five RTX 4090s within 7%.
4. **A horizon rule (exploratory).** From trace replays on 9 models, half of foresight's value needs about 0.65 C distinct experts of lookahead.

Everything is backed by a public registration branch, a rental ledger, validity gates, and a clause-by-clause scorecard of 1,687 predictions.

## Strengths

1. **The numbers are real and they reproduce.** I recomputed about 40 tables, cells and ranges from the raw rows, counters and probes with my own code (table below). Every number I checked matched to the printed rounding:
   - Tables 3, 4, 5, 6, 12 and 30;
   - the panel and new-machine gap shares;
   - the trend predictions in brackets;
   - the Few-1R pooled mean and its t-interval;
   - the noise statistics.

   This is rare and creditable.
2. **The registration can be verified, and I verified it.**
   - Every one of the 94 rentals of jobs 093–113 has a job-script commit before the rental was created; the smallest gap is 2.7 s.
   - GitHub's own push records, which I fetched independently, match the authors' push file to the second.
   - The push came after rental creation in four cases (up to 0.8 s). In all 94, the job started at least 26 s after the push.
   - The commits made after launch (jobs 100, 102, 105, 106, 107, 111) are all disclosed. The four diffs I inspected change hosts or add relaunch notes, not predictions.
3. **The paper reports its failures.** Gates were enforced as registered:
   - job 110a stopped at a ratio of 0.2499 against a 0.25 gate;
   - 107c and 107e stopped at round spreads of 3.5% and 6.0%;
   - 108c stopped on a known GPU.

   Failed hosts are in the ledger, and voided jobs (084, 110) are kept and explained. The failed clauses of jobs 109–113 are printed, H3 and H6 of job 113 among them.
4. **Most claims carry an explicit evidence label.** Table 1 separates derived, registered, loose and post hoc. Section headings flag "(post hoc)" and "(exploratory)". Table 4 separates *measured* rows from *attributed* rows.
5. **The authors corrected their own headline in public.** Job 113 is the control that the counters of job 112 called for. The paper now says that "pays only together" is a property of this engine's two-read path. Few groups re-run the experiment that undercuts their earlier narrative.
6. **The second-card test is a genuine out-of-sample check.** The trend was frozen in the header of job 110 before any RTX 4090 data, and it held (largest deviation 6.6% at 11%). Predicting no change would have missed by up to 46%.
7. **The statistical unit is argued for, not assumed.** Machines carry 96% of the variance of the log gain and problems 1–3%. I reproduced this on the 15 panel machines. Claims are framed as "on every machine of a class", which suits a convenience sample.

## Weaknesses, most important first

**W1. The headline decomposition still uses a two-read arm that does not hold MIN's set.**
- Table 4's first measured row is "MIN's set alone (MIN-2R)" = 0% at 11%. Fig. 2 and the abstract's sentence on "what is cached and how it is read" rest on the same arm.
- The counters show that this arm never holds MIN's set. Few-2R misses 54.7–55.7 experts per token with the fetch table, against 43.0 with fetches off and 40.7 for Few-1R. Job 113 confirms it.
- So the row measures the fetch table's interference, not "what is cached". The 2×2 is not a clean factorial: the read-path factor changes the level of the set factor.
- The paper now concedes this in prose. But Table 4, Fig. 2, the bold claim in the introduction ("What is cached pays only if the read path keeps it") and the abstract still lead with the confounded decomposition. The corrected mechanism rests on two machines.
- The two analyses also use different sets:
  - Table 4's decomposition uses MIN's **greedy** set.
  - The corrective headline (1.23–1.26×) uses the **fewest-admission** set.
  - The greedy set read twice with fetches off gains only 1.09–1.12× over the fetch-off deployed cache (I reproduce 1.094 and 1.122).
- So the reader still does not know what share of the gap "what is cached" closes for the greedy set once the set is actually held. That is the quantity Table 4 claims to report.

**W2. Several rules rest on very few machines, near a threshold that is itself noisy.**
- Job 113 has n = 2, and one of the two machines fails two of its six clauses: H3 (deployed 1.07× with fetches off) and H6 (one-read margin +0.02).
- "Where the link reads at under a third of the CPU rate, Few-2R pays instead of MIN-1R (1.11–1.22×)" rests on one stable machine, Pf (I reproduce 1.113 and 1.217).
- The fast/slow line at 0.5 was drawn after the panel's data, and the probe ratio is not a stable property of a machine. On the same Ryzen 9 5950X host, I find 0.837 in job 109 and 0.738 in job 112, a change of −11.8%.
- Several machines sit within that distance of 0.5:
  - Core Ultra 9 285K: 0.54 and 0.56;
  - the second 285K: 0.49;
  - Ryzen 9 9950X: 0.54.
- The paper should report how many machines would change class under a ±12% probe change. Rule 4 for builders should be stated with this uncertainty.

**W3. The bound and the headline range depend on one probe run and on a class drawn after the data.**
- Eq. (1) uses the single highest reading of one probe run, often taken while the model downloads. By the paper's own closed-form account, the engine reads more than 5% faster than the probe on 12 of 84 launch-budgets (up to 1.22×). So on those machines, Eq. (1) is not a bound.
- In the headline "31–54% on consumer machines", both ends are the least reliable points:
  - The top (54%, a 9950X with B_host = 52.0 GB/s) is admitted to be probably too high.
  - The bottom (31.2%) is a single Threadripper 9960X with 178 GB/s of quad-channel memory. The post hoc "consumer" scope includes it. It is also the machine where Eq. (3) is 1.56× Eq. (1).
  - Without it, the lower end is about 38% (I scanned 51 consumer launches).
- The paper should report the median and IQR over machines, report the second probe's repeatability per machine, and either justify putting HEDT parts in the consumer class or report them separately.

**W4. The registration practice is genuine but uneven in what it proves.**
- (a) Several headline thresholds were loose, as the paper admits: the share band 0.25–0.55 against an earlier 33%, the ≥ 0.50 floor against an analytic 0.85–0.99, capture ≤ 0.35, and Margin 0.98–1.06.
  - In Table 1, the reachability claim cites the loose P2 of job 102 (≥ 50%). The stricter registered P1 (token mode ≥ 0.80) also held, at 93–97%, and would be the better citation.
- (b) Table 1's "held" means *point estimate only*, while Appendix D's "held" means the interval excludes the threshold. Overall, 685 of 1,687 clauses (41%) are "held (point)". Often this is only because derived quantities got no interval.
  - I bootstrapped the 2×2 interaction of job 109. Every machine's 95% interval excludes zero (e.g., 109a at 11%: 1.53 ms [1.45, 1.60]), yet the scorecard records "held (point)".
  - The paper under-claims here. More importantly, the reader cannot tell strong "held" from weak "held".
- (c) Scopes were set between tests: fast links, consumer CPUs, and "stable" machines. The stability exclusion removes exactly the two machines on which Few-1R lost (EPYC 7302 at 0.85×, EPYC 7663 at 0.97×). Both exclusions are defensible (wrong outputs; a registered 2% rule), and the paper says so. But the "> 1 on every machine" claim is conditional on them.
- (d) Analysis definitions are in the headers, which is good. The analysis scripts themselves (`scripts/job113.py` and the rest) were written afterwards.
- (e) No results directory records the commit hash the machine cloned. Ordering rests on timestamps alone.
- (f) `results/113_offers_at_launch.txt` was transcribed after the job ("the query's output was not saved at the time").
- (g) Appendix D lists "rentals that produced no results". The list omits some rentals that are in the ledger: the first rental of job 095 (31 min), two short rentals of job 098 (4 and 29 min, one of which left `098_..._attempt1`), and the first rental of 100a (9 min). None of these changes a result, but the enumeration claims to be complete.

**W5. The attributed rows of Table 4 use an ad hoc T_GPU.**
- T_GPU = 2.94 ms is "the smallest of the 14 Nsight profiles we use". All 14 come from older jobs (069c, 105).
- The five machines of job 109, whose shares form Table 4's "New" column, were profiled in the same job. Their own non-expert times are 2.87–3.28 ms. About 30 other RTX 5090 profiles exist (jobs 106–109), and one of them (109d at 25%, 2.87 ms) is below the "smallest".
- The effect is small: about ±4 points on the residual. But per-machine T_GPU is available and more principled. As a lower bound in Eq. (2), the minimum over all profiles should be used.

**W6. The per-machine intervals understate the uncertainty, and pooling mixes classes.**
- Per-machine intervals are a paired bootstrap over 20 problems. But the cache carries over between problems within a run, so problems are not independent units.
- Intervals also omit variance between rentals. That variance (median 0.5%, up to 5.3%) is as large as several registered bands (±0.06, capture of a few percent) and as Margin's whole effect (1.02×).
- The authors' rescoring with widened intervals is the right instinct. A block or round-level bootstrap should be the default.
- Pooled summaries mix classes:
  - Few-1R's 1.26× [1.19, 1.33] pools RTX 4090s, server CPUs, and machines with one to three launches.
  - Table 4's panel pools runs of 20 and 30 problems from three engine versions.

  The "> 1 on every machine" statement is meaningful. The pooled geometric mean is not a population estimate.

**W7. Scope and significance are modest, and one comparison is over-promoted.**
- Scope: one engine (the authors'); one prompt set (AIME-25, also used while developing the system); teacher-forced replay of each model's own greedy text; most findings at two budgets of one model.
- Eq. (1) is a roofline with MIN's read count, so the contribution is chiefly the measurement methodology and the engine findings.
- The published-systems audit is post hoc and heterogeneous, and scored at datasheet rates:
  - Its median of 13.6% (which I reproduce from Table 32's 20 in-class rows) includes three rows below 5%. Table 32's own caption says rows below 5% are "more likely configuration mismatches … and are not adjudicated". Without them, the median is 14.6%.
  - Yet the introduction states "published systems reach a median 13.6%" with no caveat.
- The bold claim in the introduction, "Without foresight, little of it is reachable", is broader than Section 6's evidence. That evidence is four variants built from the engine's own two mechanisms; Section 6's title is correctly narrower.

## Claims checked against raw data

My code is in `scratchpad/rev25b/`. "✓" means equal at the printed precision.

| # | Claim (location) | My check (raw source) | Result |
|---|---|---|---|
| 1 | Every job-093–113 prediction committed before its rental; minimum gap 2.7 s (App. D) | `git log` commit times of stub and main script vs ledger `start`, 94 rentals | ✓ minimum 2.7 s (105f); none missing |
| 2 | Push ≤ 0.8 s after rental creation in 4 cases; job start ≥ 26 s after push | GitHub activity API (fetched by me) + `manifest.json` start_utc | ✓ 108a/b/d/f; minimum 26 s (100f). Authors' push file matches GitHub to 0 s on 95 overlapping events |
| 3 | Post-launch header edits only substitute hosts, predictions unchanged (jobs 102, 105, 106, 111) | `git show --format=` diffs | ✓ host and relaunch notes only. 100 and 107 amendments exist and are disclosed |
| 4 | Job 113 registered before launch | commit f174e0a 18:18:48Z; rentals 18:19:25Z; job start 18:21:05Z | ✓ |
| 5 | Table 5, every cell (ratios, ±, misses 55.6/43.0, 54.7/43.0) | `113b`, `113d` ec rows + st counters | ✓ (e.g. Few-2R0/base0 1.259, 1.231; Few-1R/base 1.397, 1.344) |
| 6 | One read adds +0.02–0.16 | fetchplan/base − bypassplan0/base | ✓ 0.023, 0.162. Table 5's rounded cells imply 0.17 for the 5950X |
| 7 | "Few-2R missed 0.77–0.79 times as often … as the deployed cache without them" | counters | ✗ as worded. 0.773–0.786 is Few-2R0 vs Few-2R *with* fetches. Few-2R0 vs the fetch-off deployed cache is 0.692. Reads 0.785 ✓ |
| 8 | Read twice gets 59–93% of what reading once gains | (Few-2R0 − 1)/(Few-1R − 1) | ✓ 0.59, 0.93 |
| 9 | 25%: fetch-free Few-2R 1.33–1.36× vs Few-1R 1.29–1.43×; two-read arm faster on 285K | `C32_r1A` | ✓ (285K: Few-1R/Few-2R0 = 0.952) |
| 10 | Job 113 validity V1–V3 | loss and round spread | ✓ spreads 0.60%, 0.11%; config losses within 0.46% |
| 11 | Table 6, RTX 5090 rows (MIN-1R, ahead, best variant, capture; both budgets) | `109a,c,d,e,f` | ✓ all cells (e.g. 5950X 1.366 / 1.500 / 1.010 / 1.9%) |
| 12 | Table 4 "New" column, 16 cells with t-intervals | own gap shares; R* 38.315 / 15.340; B_host from `concur.txt` | ✓ (e.g. Both 34.7 [21.9, 47.5]; resid 0.5 [−3.1, 4.2]) |
| 13 | Table 4 "Panel" column, 16 cells; 39% on 13 fast-link machines | 15 panel launches (096a/b, 099a–j, 100b/f, 101b) | ✓ (Both 32.6 [21.0, 44.3]; fast-link 38.8%) |
| 14 | Interaction positive on 9 of 10 job-099 machines; on all 5 new ones | own interaction in ms | ✓ (099f fails). New: bootstrap CIs exclude 0 on every machine, yet scored "held (point)" |
| 15 | RTX 4090 rows of Table 6 and the bracketed trend predictions | frozen coefficients from job 110 header × each probe ratio | ✓ all cells |
| 16 | Trend within 7% (11%) and 5% (25%); no-change misses 46% / 69% | max \|ln dev\| | ✓ 6.6%, 5.0%; 46.5%, 68.9% |
| 17 | Layer-ahead variants 0.64–0.72× on slow 4090s; Margin up to 26% capture | `111d,f,g` | ✓ 0.644–0.724; 25.7% |
| 18 | Table 12 speeds and misses; early/2R misses 0.97–0.98 (11%), 0.95 (25%) | `112a,b,d,g` | ✓ |
| 19 | Relaunch: deployed time within 0.7%; ratio moved up to 12% | 112g vs 109f, 112d vs 109c | ✓ −0.71%, +0.42%; ratio −11.8% |
| 20 | Table 3, host S: all 18 speeds and 6 paired CIs; host B gpt-oss rows and Qwen3 12.5% | `089/bs1.jsonl`, `081/bs1.jsonl` | ✓ (e.g. 1.207 [1.195, 1.219]; 1.294 [1.278, 1.312]). Bounds 172 and 140 tok/s ✓ |
| 21 | 31–54% of Eq. (1) on consumer machines; 41–48% on the 5 new machines | scan of 51 consumer launches | ✓ 31.2–54.0%. Low end is a Threadripper 9960X (178 GB/s); next lowest about 38%. New 41.5–48.0% ✓ |
| 22 | Eq. (3)/Eq. (1): median 1.00, up to 1.56 | own minimisation over f | ✓ median 1.00; 1.56 is the Threadripper 9960X |
| 23 | T_GPU = 2.9 ms, smallest of 14 profiles; RTX 4090 profile 3.5 ms | `prof_C14/C32.json` (069c, 105); 110–112 profiles | ✓ 2.940; 4090 3.37–3.56. But job 109's own profiles give 2.87–3.28 |
| 24 | Slow-link residual 35–47% (panel), 31–36% (4090), 27–31% with 3.5 ms | own attribution | ✓ |
| 25 | Few-1R 1.26× (t-interval 1.19–1.33) on 15 stable machines; lost on the two unsteady ones | 22 launches with fetchplan, one value per machine | ✓ 1.257 [1.186, 1.332]; losses 0.85 (EPYC 7302), 0.97 (EPYC 7663) |
| 26 | Same-CPU panel machines differ by up to 29% | deployed ms at C = 14 | ✓ 1.291 (two 285Ks, ratios 0.29 vs 0.81) |
| 27 | Machines carry 96% of variance in log gain; problems 1–3%; Kendall's W 0.77 / 0.82 | two-way decomposition on 15 × 20 | ✓ 95.8–96.1%, 1.2–2.9%; W 0.78 / 0.80 (paper: 19 machines) |
| 28 | Spearman 0.89 (MIN-1R gain vs ratio) | panel | ✓ 0.89 on 15 machines |
| 29 | Round-to-round median 0.2% (max 1.8%); median half-width 0.6% | 108 machine-configurations, two rounds | ✓ 0.17%, 1.77%, 0.61% |
| 30 | Table 30 (read-schedule microbenchmark), 86–96%; repetitions within 0.9% | `readsched_C*.txt` + probe | ✓ all 24 cells |
| 31 | Variants read 1.57–1.76 R*; Few-1R copies 0.60–0.73 of greedy, +2.5–5.3% misses | st counters (109, 104) | ✓ |
| 32 | Exact 16-token window recovers 0.78–0.93 in time; lowest machine misses 0.80 | panel aa / w16 / fetch | ✓ (099f 0.78) |
| 33 | Unsteady servers: Xeon 8347C rounds 52% apart; EPYC 7302 loss 0.21–0.45; EPYC 9754 / 7663 at 3.5% / 6.0% | 106c/d, 107c/e | ✓ |
| 34 | Table 10 tallies (1,687 clauses; 565 hand-scored) | arithmetic on the table; per-job clause counts | ✓ |
| 35 | Published median 13.6% (20 in-class rows) | Table 32 rows | ✓ (14.6% without the three rows the caption calls likely mismatches) |
| 36 | 147 rentals cost $107.5 | ledger | ✓ $107.48 |
| 37 | Few-2R pays 1.11–1.22× where the link is under a third of the CPU rate | 104a (Pf) | ✓, but one machine |
| 38 | Gates enforced as registered (110a at 0.2499; 107a/f/g, 108c/e) | `v0.txt`, `validity.txt` | ✓ |

## Clarity

**What works**

- Fig. 1, the step schematic, puts the two read paths and the second read in one picture. It is the best single aid in the paper.
- Table 2, the names table, together with the *Terms* paragraph, finally gives the configurations one vocabulary: 1R/2R, greedy vs fewest-admission, the yardsticks.
- The five bold claims in the introduction each point to a section, and Table 1 maps claims to evidence. A reader can find the evidence for each claim.
- Table 4 separates measured rows from attributed rows, and Section 5's running example (O4: 10.0, 13.0, 20.6, 15.2, 13.8 ms) grounds the abstractions.
- Headings that announce evidence status ("(post hoc)", "(exploratory)") and the Limitations bullets are good practice.

**What still makes it hard to read**

1. **Abstract and introduction sentences carry too many qualifiers.** Example: "At that budget, on machines whose link reads at least half as fast as the CPU, the experts MIN's greedy schedule keeps pay in the engine as deployed only when each admitted expert is read once, not twice: then they close 35% of the gap, in a test registered before its machines were rented." Five scoping clauses come before the reader knows the finding. The abstract has about a dozen numbers and assumes MIN, greedy, fewest-admission, in-step fetches and 1R/2R are already known.
2. **There are too many yardsticks and bounds.**
   - Speed relative to the deployed cache, share of the bound, share of the gap, capture, and value of foresight.
   - Three bounds (Eqs. 1–3), each with datasheet and measured GPU variants, plus a pooled bound and a datasheet-ceiling bound for the audit.
   - Fig. 5's caption has to open with "Baseline here: admit every miss, not the deployed cache", which shows the reader is expected to get lost.
   - Phrases like "0.02–0.16 more of the deployed cache's speed" (a difference of two speed ratios) are non-standard. They should be written as "Few-1R/Dep − Few-2R₀/Dep = 0.02–0.16".
3. **Some comparators are ambiguous.** Item 7 in the table above: "Few-2R missed 0.77–0.79 times as often and read 0.78 times as many host experts as the deployed cache without them". The two ratios have different denominators; read literally, the first is wrong (it is 0.69). "Once" is overloaded ("read once", "once kept", "once held") in "MIN's fewest-admission set read twice, once kept, got 59–93%…".
4. **The narrative still leans on the retracted framing.** Section 4 first states "pay only together", then says it is an engine artefact. Section 5 interrupts its own argument to cite the job-113 result again. A reader has to hold the confounded decomposition and its correction at once. Table 4 and Fig. 2 should show the corrected picture, or carry the caveat in their captions.
5. **Orphan parentheticals and compressed table cells.**
   - Orphan: "(Registered as gain bands per budget on two machines; held at most budgets.)" at the end of a Section 5 paragraph, with no indication of what was registered.
   - Compressed cells: Table 1's "positive on 9 of 10 panel machines (failed), 5 of 5 new ones (held; fast-link scope set in between); share 35%".
   - Table 1's "held" means something different from Appendix D's "held".
6. **The appendices are organised around jobs, not claims.**
   - Appendix D is many pages of prose keyed to job numbers (073–113) and host codes (Pa–Pj, O1–O6, "Pf again"). The appendix tables use a second vocabulary: base, foa, bypass, both3p, "ahead only", "read-ahead, paced". The scorecard supplement uses only code names (bypassplan0).
   - A reader checking one claim must cross Table 1 → Table 9 → Table 10/11 → prose → scorecard. A one-page claim → job → clause → number map would help more than more prose.
7. **Tables 3, 5 and 6 are hard to scan.** Host-level metadata is placed inside header rows that span the table. Table 5 nests "vs" headers, and Table 6 mixes "±" half-widths, bracketed predictions and percentages in one cell.

On balance clarity has improved since the 2/5 rounds: the names table, the terms paragraph and the claims table work. But the density per sentence and the number of parallel vocabularies keep it at 3/5.

## Questions for the authors

1. Can the 2×2 be re-run with the in-step fetches off on every arm, or with a fetch table that respects the plan, on the five machines of job 109? What share of the gap does MIN's **greedy** set read twice close once it is actually held?
2. How many machines change fast/slow class under the observed ±12% movement of the probe ratio between rentals? Would Rule 4 survive classification by a repeated probe?
3. Why is T_GPU taken from the 069c/105 profiles rather than from each job-109 machine's own profile? Why not the minimum over all RTX 5090 profiles for Eq. (2)?
4. Does your bootstrap over problems account for the cache carrying over between problems? Have you tried a block bootstrap, or resampling over rounds?
5. Why is the Threadripper 9960X (quad-channel, 178 GB/s) in the "consumer" class that defines the 31% end of the headline range?
6. Should Table 1's "held" match Appendix D's definition (an interval excluding the threshold)? Can you compute intervals for the derived quantities (interaction, capture, differences) so that "held (point)" is reserved for genuinely untestable clauses?
7. Will each results directory record the cloned commit hash (`git rev-parse HEAD`) and the sha256 of the job script in future?
8. Will you exclude the three published rows below 5% from the 13.6% median, as Table 32's own caption suggests, or else drop the number from the introduction?

## What would raise my score

- A clean decomposition, for a soundness of 4: re-run the 2×2 with every arm on a read path that holds the scheduled set (fetch=0, or a plan-respecting fetch table) on at least five machines, and rebuild Table 4 and Fig. 2 from it. Report greedy and fewest-admission sets separately.
- Probe repeatability: a second idle-machine probe on every machine, reporting the bound's and the ratio's repeatability; a median and IQR beside the 31–54% range; HEDT parts reported separately.
- Statistics: intervals for every derived registered quantity; "held" used one way throughout; intervals resampled by round or block by default.
- Per-machine T_GPU where it exists.
- Clarity:
  - An abstract of at most 150 words with at most 5 numbers.
  - One vocabulary from the main text through the appendices to the scorecard.
  - A one-page "claim → job → clause → raw file" map replacing most of Appendix D's prose.
  - Rewording the ambiguous comparators noted above.
- Removing the published-systems median from the introduction, or caveating it there.

## IEEE format (ieee-paper.pdf, ieee-supplement.pdf)

1. **Front matter.** The title, anonymous author block, abstract (no citations) and Index Terms are all present. However:
   - The Index Terms are ad hoc rather than drawn from the IEEE taxonomy.
   - The supplement is announced by an unnumbered footnote that names a file ("ieee-supplement.pdf"). IEEE convention is "supplementary material available online".
2. **Cross-references between the two documents.**
   - The paper cites "Appendix C/D/G/J/L/N" with no indication that these live in a separate PDF. The footnote is the only cue; "Appendix D (supplementary material)" or S-prefixed appendix labels would be conventional.
   - The supplement's Appendix A also lists "The supplement: every clause scored…", which is a *third* document (the MLSys-formatted scorecard, `supplement.pdf`). Inside a file already titled "Supplementary Material", this is a name collision. There is no IEEE-formatted scorecard.
   - The supplement's phrases "(Table VI, the supplement)" and "(the supplement)" are ambiguous for the same reason.
3. **Reference lists.**
   - The paper and the supplement have separate numbered lists, and the same work gets different numbers: FreeToken is [9] in the paper and [31] in the supplement; Hoefler and Belli is [1] in the supplement and absent from the paper.
   - IEEE style truncates author lists of seven or more with "et al.". Ref. [7] lists all 17 authors, while [15] and [35] use "et al.", which is inconsistent.
4. **Build hygiene.** Both IEEE logs carry 30 "Label … multiply defined" warnings. These are bibliography keys imported across documents by `xr`. The printed citation numbers looked correct where I checked, but the build is not clean. There are also font-substitution warnings (OT1/ptm/m/scit, small-caps italic in Times).
5. **Captions.**
   - In the paper, "TABLE I" sits above each table in small caps and "Fig. n." below each figure; this is correct.
   - In the supplement, the table-caption macro is overridden to the body font. This deliberately departs from IEEE style.
   - Table S26 (a longtable) prints "TABLE S26: The audit's rows…" inline, unlike every other table.
6. **Floats and page use.**
   - Table III, cited in Section III, floats to page 5, after Section IV's text has begun.
   - Pages 3–6 are dominated by full-width tables and figures (Tables I and II together on page 3), which pushes the text one or two pages away from its floats.
   - In the supplement, tables scaled to the line width have inconsistent font sizes. Tables S11, S23, S24 and S25 print at roughly twice the body size; Tables S7 and S10 are tiny.
   - Several supplement pages are largely empty around Fig. S4 and Table S26.
7. **Page count.** The paper is 10 pages including references (references start on page 9). Whether this fits depends on the venue; many IEEE conferences allow 8–10 pages including references. The 30-page one-column supplement is unusual in length.
8. **Typography.**
   - Configuration names (MIN-1R, Few-2R) are set in sans-serif inside Times text.
   - Run-in headings lettered a) to l) inside Appendix D are deep fourth-level heads. Subsections would read better.

## Scores

- **Overall: 6 / 10.** Acceptable with revisions. An exemplary record of measurement and registration, with every checked number reproducing. But the central decomposition still rests on a confounded arm, and the corrective mechanism stands on two machines.
- **Soundness: 3 / 5.**
- **Methodology: 4 / 5.**
- **Significance: 3 / 5.**
- **Clarity: 3 / 5.**
- **Confidence: 4 / 5.**
