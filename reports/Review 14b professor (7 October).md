# Review 14b: "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth"

Referee role: professor in a systems group that works on scientific benchmarking (Hoefler and Belli, SC'15). The review is written for a fellowship panel and a workshop programme committee. Date: 7 October 2026.

## Materials read

- `paper/paper.pdf` (37 pages). I read the whole main text (pages 1 to 9) and checked it against `paper/paper.tex`. I read these appendices in full: A (checklist), B (prediction log), C (calibrated model), J (MIN's schedules and the machine), K (the system in detail) and M (artifact). I skimmed D to I and L. I looked at the main-text pages (1 to 9) and one page of Appendix B rendered as images, to judge layout.
- `paper/supplement.pdf` (41 pages, the clause-by-clause scorecard). I read the tables for jobs 105 to 108 and the job 106 table in detail.
- LaTeX sources: `paper.tex`, `app_wsg.tex`, `app_more.tex`, `app_traces.tex`, `app_value.tex`, `tab_regtests.tex`, `tab_configs.tex`, `tab_job107.tex`, `tab_job108.tex`, `supplement.tex`, and the number macros in `wsg_*.tex`. I used the macros only to see which value the text prints.
- The job script headers in `/home/claude/gpu-branch/jobs/`, which hold the registered predictions: 081, 089, 102, 105, 106, 107 (and its host stubs a to g), 108 (and its host stubs a to f). I also read the content diffs of every amendment commit to the 100, 102, 105, 106 and 107 scripts.
- Raw results in `/home/claude/gpu-branch/results/`. I parsed directly: `ec_*.jsonl` (per-problem `decode_ms`, `n_decode`, `nll_sum`, `nll_n`, and the config's `stats=` path), `st_*.json` (engine counters), `concur.txt` (bandwidth probe), `g_prof.json` and `prof_C*.json` (Nsight G), `v0.txt`, `validity.txt`, `cores.txt`, `cpu.txt`, `nvidia-smi-q.txt` (GPU UUID and driver), `readsched_C*.txt`, `manifest.json` and `bs1.jsonl` (server-level rows behind Table 3). This covered jobs 069c, 081, 089, 093 to 108 and 098.
- `gpu/vast_ledger.json` (rental start and stop times) and `gpu/vast.py` (how a rented machine gets the job: it clones the public `gpu` branch).
- The `gpu` branch history: `git log --format='%h %at %ct'` per file and `git diff` of file contents. I also pulled GitHub's server-side push log for `refs/heads/gpu` (`gh api repos/.../activity`), which gives push times that the client cannot set.
- To match definitions only, I read parts of the analysis code: `scripts/job106.py` (`law`, `load`), `scripts/job108.py` (gate and fallback G), `scripts/reanalysis.py` (the exploratory launch set and the elasticity), `scripts/sumlaw_paper.py` (profile categories), `scripts/factorial_shapley.py` (`limit1_of`), `jobs/ec2/fetch_table.py` (`bandwidths`), plus `prereg/speed_limit_v2.json` for MIN's reads per token, R*. All recomputations below use my own code, in `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/rev14b/`: `relation.py`, `table4.py`, `explore.py`, `cons.py`, `plan.py`, `spear.py`, `headline.py`, `decomp.py`, `reg_times.py`, `reg_main.py` and `push_check.py`.

## Independence statement

- I did not open anything under `reports/` (except to write this file), any `prereg/*outcome*.md`, `research_notes/`, `apply/`, `paper/paper_v1_prereview.tex`, `paper/paper_v2_prerewrite.tex`, or any file named like a review, number check, plan or progress log.
- I did not read commit messages. I used only hashes, timestamps and file-content diffs. The GitHub activity API returns no messages.
- I recomputed every figure in the table below from raw rows with my own parsers. The authors' scripts served only to resolve definitions: which probe line counts as `B_host`, which thread count defines `B_c`, and which profiles feed the fallback G. Where my number differs from the paper's, I say so.
- I modified nothing in either repository and committed nothing.

## Summary

The paper studies batch-1 decode of two Mixture-of-Experts models whose experts mostly live in host DRAM: gpt-oss-120b and Qwen3-30B-A3B. All runs are on rented RTX 5090 hosts. It makes four contributions.

1. **Two bounds** (Eqs. 1 and 2). The first is Belady's MIN-with-bypass host reads per token, multiplied by the expert size and divided by the machine's highest probed host read rate, with a GPU term. The second, for systems that read on demand, adds the GPU's non-expert time in series. A read-only microbenchmark reaches 86 to 96% of the read term (registered).
2. **A descriptive relation** (Eq. 3) for the authors' llama.cpp expert cache: time per token = profiled GPU time + counted host reads at the probed rate, with a share G/T of background admissions hidden. It was found on 30 launches. It then **failed all four registered tests**: jobs 105 to 108, Table 4. After the fact, the authors split machines by processor brand. The relation fits consumer processors within 8% on 37 of 39 stable launches, and misses two of the three stable server machines by 28 to 59%.
3. **An in-engine oracle study** of foresight. MIN's admissions read once (copied "in the step") gain 16 to 51% over the deployed cache on three hosts. The usual two-read and Belady-prefetch uses of foresight gain little or lose. Among MIN's hit-optimal schedules, the fewest-admission schedule gains on every machine that passed the checks. Which load path pays depends on the link-to-CPU bandwidth ratio. Raising the deployed admission margin, an online version of "admit less", gains a median 2%.
4. **A price on foresight**, exploratory on nine models' traces. Half of MIN's read saving over "admit every miss" needs a window whose distinct experts per layer number about 0.65 C. No linear or recurrent forecaster tested closes more than 15% of the gap.

The artifact is unusually complete. Each GPU job's predictions sit in the header of the script the machine clones and runs. Every rented host, including those turned away by gates, is in a ledger. Raw rows and counters are released, and a clause-level scorecard covers jobs 073 to 108.

## Strengths

1. **Registration is real and verifiable, not decorative.** For every launch of jobs 093 to 108, the commit carrying the predictions precedes the rental in the ledger, by 2 to 16 s for the decisive jobs 105 to 108. Server-side GitHub push times confirm that the header was public before the machine cloned and ran the job, 0.4 to 47 min before the on-machine start in `manifest.json`. Every amendment I diffed changes only host lists or replacement rules, with two exceptions in job 100: prediction 6 was reworded before any of that job's hosts started, and prediction 9 was added before 100e and 100f. Each amendment was pushed before the hosts it concerns started. This is a better registration practice than I have seen in any systems paper.
2. **Failures are reported as failures.** Table 4 says "failed / inconclusive / failed" for the relation the paper is partly built on. Hosts turned away by gates are listed with reasons. The supplement scores 181 clauses for job 106 alone, including 48 failures. The abstract states that no test of the relation on new machines passed. A paper willing to report this deserves credit in a fellowship context.
3. **The numbers reproduce.** I recomputed 40 quantitative claims from raw rows (table below), and nearly all matched to the printed precision. The exceptions: two counts are off by one at a boundary (rows 18 and 20), one Spearman coefficient differs in the second decimal (row 32), the cost line is stale (row 40), and the supplement defines round spread differently from the main text (row 10). Table 3's ratios and bootstrap intervals reproduce to the third decimal, and so do all four rows of Table 4.
4. **The machine is taken seriously as a source of variance.** The paper shows that machines carry most of the variance, and that two hosts with the same CPU model differ by up to 29% (I get 29.1% within the panel). It shows that gains depend on the link-to-CPU ratio (Spearman 0.89; I get 0.88 [0.63, 0.97] on 19 machines). It refuses to pool across machine classes. This is the right lesson from Hoefler and Belli's rules 5 and 7.
5. **The oracle-in-the-engine design is good experimental practice.** Teacher-forced replay gives every configuration the same routing. The configurations of Table 2 are a clean 2×2 of *what to cache* against *how many host reads per admission*. The "Reads" column makes the mechanism visible. The finding that read-once is what makes foresight pay is clear and actionable. On the three oracle hosts, MIN read twice ranges from −6% to +22%, MIN read once gains 16 to 51%, and Belady prefetch can lose.
6. **The Limitations section is specific and quantitative.** It gives the number of launch-budgets where the engine outran the probe, the confounds of the server/consumer split, and why the overlap term is not identified.

## Weaknesses (most important first)

1. **The "bound" is a probe-relative reference, and the paper's own data show the engine exceeding its rate.**
   - The abstract says the reads "at the machine's measured read rate bound every system that executes the exact routing". But `B_host` is the single largest of about 20 probe lines in `concur.txt`, including concurrent sums.
   - By Eq. 3's own accounting, the deployed cache's implied read rate exceeds 1.05×`B_host` on 13 of 68 exploratory launch-budgets (I get 12 of 68, maximum 1.22×). On those machines Eq. 1 is not a bound on host read time. It only looks like one because the engine reads far more than R*.
   - On the dual-socket host (107d), `B_host` = 204.5 GB/s comes from a concurrent line whose CPU part (181.4 GB/s) exceeds every CPU-only line (best 151.4). This anomaly appears in none of the other 83 probes. It moves that host's implied rate from about 0.37 to 0.27.
   - The derivation is correct given the definitions. The word "bound" in the abstract, Section 3 and Table 1 ("derivation") still claims more than the measurement supports. Either rename it a probe-relative speed of light and show its sensitivity (median probe line, CPU-only best, theoretical DRAM and PCIe peaks), or raise `B_host` to the best rate any system has been observed to sustain.
2. **The central descriptive claim is post hoc, and its out-of-sample errors are one-sided.**
   - Eq. 3 failed all four registered tests. The consumer/server split that now carries it is the third class line drawn after seeing data. Job 107 used new machines; job 108 used one NUMA node with at most 32 usable cores, a line drawn after 107. The processor-brand line was drawn after 108. Job 108's band was also widened from 6% to 8%, citing job 107's 7.1% miss.
   - On every consumer machine new to a registered test, the relation under-predicts every gpt-oss cell: −0.9 to −7.1%, 6 of 6 cells (my recomputation). On Qwen3 the miss reaches −11.3% (5900XT). The 30 exploratory launches, by contrast, centre near −1%. This is the signature of an in-sample fit, not a law.
   - The overlap term is not identified. Discounting admissions by half passes 11 of 12 stable cells with a better median than the registered form (1.9% against 2.3%, my numbers).
   - The paper says much of this. Still, the abstract's second sentence, Section 4.4 ("the deployed cache's time *is* the sum of three parts") and Figure 2 present the relation as the description of consumer machines. Section 4.4 carries no evidence label at all.
3. **The statistical unit is declared but not delivered.** Section 2 states "the machine is the unit".
   - In practice almost every comparison is one launch with a bootstrap over problems.
   - The "round-level" intervals of jobs 106 to 108 resample 2 to 3 processes inside one rental. The bootstrap is close to degenerate: with two rounds there are three distinct resamples. It does not capture rental-to-rental variance, although job scripts and the supplement call it "launch-level".
   - Table 3 shows the consequence. Host B's 1.294 [1.278, 1.312] and host S's 1.207 [1.195, 1.219] are the same comparison with non-overlapping intervals. Job 089 registered a ±0.06 reproduction band and missed it.
   - The 30 problems are also not exchangeable units: the cache carries over between problems in a fixed order.
   - The honest summary is: point estimates per machine, with an across-machine range, and intervals that describe within-launch noise only.
4. **Validity gates partly follow the outcome and are partly post hoc, and Table 1 smooths this over.**
   - Job 106 registered no gates. Its round-stability clause was a *prediction* that failed on 106c and 106d, and those hosts were then set aside. Table 4 shows this transparently.
   - Table 1's "MIN's fewest-admission schedule gains on every valid machine: registered; held on valid machines" depends on that post-hoc exclusion. Job 106's registered clause `c-P7` ("fetchplan/base > 1 at gpt-oss 11% on every host") failed at 0.846 on 106c.
   - On the two slow-link hosts that failed a check, 106c (ratio 0.14) and 107e (ratio 0.21, gated by a registered rule), the fewest-admission schedule *lost*: 0.846 and 0.974. Only Appendix J and Table 21 show this.
   - All five server launches that failed a check are also missed by the relation, by 21 to 38%. The one server that fits, the EPYC 7402P, had a single round, so the round check was never applied to it.
   - "Valid" here means "stable", which is a host property, not a measurement defect. Call it that, and report claims both with and without the gate.
5. **Evidence labels are incomplete and "held" is mostly point-held.**
   - Table 1 gives "registered (3 hosts)" with no verdict for the system comparison. Job 089's reproduction clause and the "≥1.8× llama.cpp" clauses on slow hosts failed.
   - Several quantitative main-text claims carry no label: 38 to 54% of Eq. 1, the published-systems median of 13.6%, the Section 4.4 shares, "The machine decides how much" (exploratory, through job 104), and the forecaster results.
   - Under Appendix B's rule, "held" requires the interval to exclude the threshold. Most registered clauses are scored "held (point)": job 106 has 21 held, 112 held (point) and 48 failed; jobs 073 to 098 have 218, 193 and 142. Table 1 should say which.
6. **Registration timing is right in substance but overstated in wording.**
   - Appendix B says each header was "committed and pushed before the machine started".
   - Commits do precede rental start. But GitHub's server-side push log puts the pushes for 107f, 107g, 108a, 108b, 108d and 108f 0 to 1 s *after* the ledger's start times. The controller pushes and rents almost simultaneously.
   - What is true and verifiable is stronger and simpler: the machine clones the public branch after it boots, so the run necessarily used the published header. `manifest.json` should record the cloned SHA so readers need not reconstruct this.
   - Job 107's rule for replacing machines that failed the round check was added mid-job, after four hosts' results were in. It was disclosed and committed before the hosts it added, but it is an outcome-informed amendment and belongs in Table 4's caption.
7. **Significance is moderate.**
   - The study covers one GPU model, two models, AIME prompts also used during development, the first 256 tokens and batch 1.
   - The system's lead over FreeToken is 3 to 29% and model-dependent. It only ties on FreeToken's own headline model at two of three budgets (1.010 and 1.003), and trails on host S at Qwen3 43.75%.
   - The bound is a roofline-style construct close to prior Belady-based analyses (Zhang 2026, Liang 2026). The descriptive relation did not survive prospective testing.
   - The durable contributions are the read-once lesson, the schedule-choice result and the distinct-expert horizon rule. The rule is exploratory, and no realisable forecaster approaches it.
8. **Artifact hygiene.**
   - The cost line in Appendix M is stale: "88 rentals (53 distinct machines) of jobs 058 to 101 cost 70.8 US dollars". The ledger through job 108 has 124 rentals, 74 distinct offers and $87.2. The registered tests that matter most are outside the stated range.
   - The ledger's hourly prices look like placeholders: every 102 to 105 rental is $0.50 and every 106 rental $0.70.
   - The supplement describes the job 106 server as varying "39%" between rounds; the main text says 52% for the same rounds. These are different spread definitions.
   - R* from the 30-problem trace (38.3 reads per token) is applied to 20-problem runs. Job 106's own header gives 39.0 for the 20-problem routing. This is a 2% bias in every engine-run fraction of the bound; small, but it should be consistent.

## Clarity

Earlier rounds scored clarity 2/5. The rewrite is a real improvement; I would now give 3/5.

**What works.**
- The introduction is organised around three questions, each answered with a pointer to a section. Table 1 maps every headline claim to its kind of evidence. A reader can stop after page 2 and know what is claimed and how firmly.
- Paragraph headings carry their evidence tag, as in "(registered)" and "(registered, partly held)". This is the right device; it is applied inconsistently (below).
- Figure 1 (one decode step) and Table 2 (configurations with a "Reads" column) make the mechanism concrete. The read count per admission is what the whole of Section 5 turns on, and Table 2 shows it.
- Figure 2 is an honest, legible decomposition. Each machine is a row, the measured time is a dot, and the shortfall below the probed rate is hatched. Server rows sit below a dashed line.
- The running example in Section 5 (one Ryzen 9 9950X at gpt-oss 11%: 10.0 / 13.0 / 20.6 / 15.2 / 13.8 ms) is the single most useful device in the paper. I checked it: 10.02 / 20.64 / 15.16 / 13.81.
- Sentences are mostly short and declarative, and the hedging is precise.

**What still makes it hard to read.**
1. **Vocabulary load.**
   - The main text coins or uses at least 14 configuration names: deployed, single read, admit every miss, MIN 1 read, MIN 2 reads, fewest-admission, greedy, prefetched, Belady, window W, recall r, deployed + window, layer-ahead copy, margin κ.
   - On top of these come "in the step" and "by the CPU", "host-bound", "launch", "round", "cell", "valid", "new", "consumer", "server", and B_c, B_p and B_host.
   - Only Table 2 defines the configurations. The repository has a `tab_glossary.tex` that the paper never includes. Put a short glossary on page 2, and drop names a reader meets only once, such as "layer-ahead copy" and "learned admission order".
2. **Three different "G"s.**
   - G in Eq. 3 is all GPU kernels (4.1 to 4.7 ms).
   - T_GPU in Eq. 2 is non-expert GPU work only (2.9 to 3.4 ms).
   - G in Appendix C's Eq. 4 is a fitted constant that "exceeds the whole all-in-VRAM token".
   - A reader moving between Section 3, Section 4 and Appendix C will conflate them. Use distinct symbols and say once, in Section 2, how they relate.
3. **Section 4.2 is a job-number narrative.**
   - The reader must hold plain against overlap, rented-again against new, gated against unsteady against valid, consumer against server against "desktop-class", and 6% against 8% bands across jobs 105 to 108, all in about 30 lines.
   - Table 4's columns ("Cells within", "New machines'", "Set aside") are cryptic, and its caption is 6 lines.
   - Smaller inconsistencies compound this. The Introduction says the relation was "registered before four later experiments", Section 4.2 says "three more", and Table 4's title says "the four registered tests of Eq. (3)" though row 105 tests a different (plain) form.
   - Suggested fix: one table with a row per new machine, giving class, gate outcome, error at each cell and verdict. The prose then only interprets.
4. **Number density without the set over which a range is taken.**
   - Most results are min–max ranges ("16 to 51%", "0.85 to 0.93×", "1.11 to 1.22×", "0.60 to 0.73") whose population is left implicit: which hosts, budgets and models. Some ranges mix budgets; others are over hosts at one budget.
   - Say "over the three oracle hosts and six budgets" or "over the five slow-link launches at gpt-oss 11%" every time. Hoefler and Belli's rule 9 applies to summaries as well as setups.
5. **Appendix B, the paper's most important methodological appendix, is a single paragraph about two pages long.** It interleaves rules, totals, job narratives and exceptions. The supplement already holds the clause tables. Appendix B should be a table (job, hypothesis, registered threshold, measured value, verdict, deviation), with the prose cut to the deviations.

Other clarity points:
- Script paths appear as evidence in the main text, for example "raises the bound by at most 1% at 11% (`scripts/linkaware.py`)". The reader cannot evaluate a path; give the quantity and an appendix pointer.
- Table 3's caption runs to 8 lines and smuggles in two further results: pooled bounds of 18 to 40% and 8 to 20%, and all-in-VRAM speeds.
- Figure 3 has 7 rows × 6 panels, three marker types and grey panel ticks. Its row labels ("Belady, 1 read", "Belady, 2 reads") do not match Table 2's single "Belady" row.
- Section 6 measures against "admit every miss", while Sections 4 and 5 measure against "deployed". Say explicitly at the start of Section 6 why the baseline changes.
- "Its copies" and "the same set" in Section 5's second paragraph have ambiguous antecedents.
- "Launch" and "round" are defined carefully in Section 2, then used loosely elsewhere. Table 20's caption says the cache "varied by more than the registered 2% between launches" where it means rounds, and the job scripts and the supplement call round-level intervals "launch-level".

## Claims checked against raw data

All values are recomputed by my scripts from the raw files named. ✓ = matches to printed precision; ≈ = within rounding or one boundary case; ✗ = does not match or is stale.

| # | Claim (location) | Paper | My recomputation (source) | Verdict |
|---|---|---|---|---|
| 1 | Table 4, job 105 (plain form): cells within 6%, new-machine cells, median \|err\| | 5/8, 2/4, 4.2% | 5/8, 2/4, 4.24% (105a/b/e/f: `ec_g_C*.jsonl`, `st_*_base.json`, `concur.txt`, `g_prof.json`) | ✓ |
| 2 | Table 4, job 106 (overlap); stable hosts | 12/20, 0/8, 4.7%; 12/12, 2.3% | 12/20, 0/8, 4.67%; 12/12, 2.34% (106a Qwen3 25% sits at +5.99%, on the band edge) | ✓ |
| 3 | Table 4, job 107 | 1/8, 31.7%, inconclusive (2 valid) | 1/8, 31.75%; 2 valid (107b, 107d) | ✓ |
| 4 | Table 4, job 108; without the EPYC 7543 | 4/6 within 8%, 6.7%; 4/4, 5.3% | 4/6, 6.70%; 4/4, 5.25%; only 2/6 within 6% | ✓ |
| 5 | EPYC 7543 error with fallback G | 34.2%, 27.9%; fallback 4.29 and 4.47 ms from 8 hosts | −34.23%, −27.91%; 4.292 and 4.469 ms (8 hosts). With G as literally profiled (≈0), −52% and −55%: outcome unchanged | ✓ |
| 6 | Dual-socket server; Ryzen 9 5900XT (job 107) | 52 to 59%; up to 11% | −52.2 to −58.9%; −11.3% (Qwen3 12.5%) | ✓ |
| 7 | New consumer machines, gpt-oss cells | under-predicted, 0.9 to 7.1% | −0.95 to −7.15%; all 6 cells negative | ✓ |
| 8 | Job 107 gate accounting | 7 rented, 3 gated, 2 unsteady, 2 valid | `v0.txt`: 107a 50 GB, 107f 50 GB, 107g 96 GB; `validity.txt`: 107c 3.5%, 107e 6.0% spread | ✓ |
| 9 | Job 108 gate accounting | 6 / 2 / 1 / 3 | 108c GPU UUID = 100b's; 108e 323 GB in use; 108b spread 7.01% | ✓ |
| 10 | Job 106 servers: wrong outputs; rounds 52% apart | as stated | 106c loss 0.208 to 0.420 against 0.190; 106d gpt-oss 11% rounds 14.52 / 22.10 / 21.93 ms = 52.2% (supplement says 39%) | ✓ (definition inconsistent) |
| 11 | Machines rented again came within 3.2% | 3.2% | maximum 3.17% (TR 9960X, 105b to 106a); other relaunches ≤1.9% | ✓ |
| 12 | Same CPU model differs by up to 29% | 29% | Core Ultra 9 285K: 099a 18.14 against 099f 14.06 ms = 29.1% (31.5% against the fastest 285K launch overall) | ✓ |
| 13 | Half-discount passes 11 of 12 stable cells | 11/12 | 11/12; median 1.9% (overlap form 2.3%, misses-only 4.0%) | ✓ |
| 14 | Server launches: 5 of 8 fail a check; implied rate of the valid ones | 0.27, 0.58, 1.04 | 0.268 (107d), 0.584 (108a), 1.038 (105a); failing: 106c, 106d, 107c, 107e, 108b | ✓ |
| 15 | Consumer class: launches/machines, implied rate, within 8% | 39/25; 0.88 to 1.16; 37 | 39/25; 0.880 to 1.164; 37 | ✓ |
| 16 | Dual-socket probe threads | 118 against 151 GB/s | 117.6 (128 threads) against 151.4 (16 threads). `B_host` = 204.5 comes from a concurrent line with CPU part 181.4, unique among 84 probes | ✓ (probe anomaly) |
| 17 | Exploratory set: size and fraction of Eq. 1 at gpt-oss 11% | 30 launches, 21 machines; 38 to 54% | 30 / 21; 38.4 to 54.0% (R* = 38.32 from `speed_limit_v2.json`) | ✓ |
| 18 | Plain form within 6%: 28/30 at 11%; at 25%, 18 against 33 of 34 for overlap | 28/30; 18 vs 33 | 28/30; 17 vs 33 (one launch at 6.0%) | ≈ |
| 19 | Elasticity of T−G to B_host at gpt-oss 11% | −0.95 [−1.06, −0.84] | −0.950 [−1.068, −0.837] (machine bootstrap, 21 machines) | ✓ |
| 20 | Engine reads faster than the probe | 13 of 68 launch-budgets, up to 1.22× | 12 of 68, maximum 1.223 | ≈ |
| 21 | Nsight: G; non-expert GPU time T_GPU | 4.1 to 4.7 ms; 2.9 to 3.4 ms | 4.07 to 4.69; 2.94 to 3.38 (069c and 105 `prof_C14/C32.json`) | ✓ |
| 22 | EPYC profiles empty under driver 570 | as stated | 108a G14 = 0.0015 ms, G32 missing; drivers 570.211 / 570.133 on the EPYCs, 580 and 595 on the Ryzens | ✓ |
| 23 | Table 3, host B: ours ÷ FreeToken with 95% intervals | 1.294 [1.278, 1.312], 1.275 [1.254, 1.295], 1.154 [1.134, 1.172], 1.032 [1.022, 1.042] | identical (launch 2, 10,000 paired resamples over 30 problems); ratio of total time 1.293 | ✓ |
| 24 | Table 3: bound at datasheet GPU; fraction | 172 (host B) and 140 (host S) tok/s; 41% | 172.3, 140.4; 40.5% | ✓ |
| 25 | Leads FreeToken at 11 of 12; 2.0 to 4.0× llama.cpp; tuned FreeToken trails at 5 of 6 | as stated | only loss: host S Qwen3 43.75% at 0.974; ×llama.cpp 2.00 to 4.02; job 098: 1.17, 1.17, 1.09, 1.02, 1.04, 0.98 | ✓ |
| 26 | Read-schedule microbenchmark, waiting per layer (job 102) | 86 to 96% | 0.860 to 0.963 of `B_host` (`readsched_C*.txt`) | ✓ |
| 27 | MIN read once (O3, O4, O5); paced; "usual ways" at the smallest budgets | +16 to 51%; up to +81%; ≤15% or lose | 1.160 to 1.514; 1.812; at most 1.148 | ✓ |
| 28 | Running example (9950X, gpt-oss 11%) | 10.0 / 20.6 / 15.2 / 13.8 ms | 10.02 / 20.64 / 15.16 / 13.81 | ✓ |
| 29 | Fewest-admission schedule beats deployed on every valid machine (7 earlier + 2 new) | held | 9 valid machines at 1.019 to 1.419. **Loses** on invalid 106c (0.846) and 107e (0.974) | ✓ (with caveat) |
| 30 | Fewest-admission: copies and extra misses relative to greedy | 0.60 to 0.73; +2.5 to 5.3% | 0.597 to 0.726; +2.5 to 5.3% | ✓ |
| 31 | Slow links: greedy at 11%; plan loaded by the CPU | 0.85 to 0.93× at ratio 0.28 to 0.32; 1.11 to 1.22× | 0.853 to 0.933 at 0.28 to 0.32; 1.112 to 1.217 | ✓ |
| 32 | Spearman of MIN-in-step gain against link-to-CPU ratio | 0.89 [0.63, 0.97], 19 machines | 0.88 [0.63, 0.97], 19 machines | ≈ |
| 33 | Admission margin κ: median, best, worst; reads saved | 1.02×; up to +6%; −1% at 1 of 10; −4 to 6% | 1.019; 1.057; 0.987 (107d at 25%); −4.3 to −5.8% | ✓ |
| 34 | Layer-ahead copy (job 105) | 1.04×; down to 0.72× at 11% | 1.042; 0.722 (0.69 at 25% on the same host) | ✓ |
| 35 | 16-token window on the deployed path; exact window's share of MIN's gain at 11% | 1.00 to 1.03×; 0.78 to 0.93 | 0.9995 to 1.035; 0.781 to 0.935 | ✓ |
| 36 | Figure 2 shares at gpt-oss 11%, consumer machines; server shortfall | G 12 to 45%, MIN 31 to 54%, beyond 22 to 35%, G 23 to 66% of gap, shortfall ≤11%; servers 0 / 34 / 59% | 11.9 to 45.4, 30.6 to 54.0, 22.1 to 34.9, 22.5 to 66.0, ≤10.6; −3 / 34 / 59% | ✓ |
| 37 | Predictions committed (and pushed) before the machine started | all jobs | every commit precedes its rental; server push precedes on-machine start in every case. Pushes for 107f, 107g, 108a, 108b, 108d, 108f land 0 to 1 s after the ledger start | ✓ in substance; wording ≈ |
| 38 | Amendments made before the affected hosts, predictions unchanged | as stated | diffs for jobs 100, 102, 105, 106 and 107 (three) touch host lists and replacement rules only. Exceptions in job 100: prediction 6 reworded before any 100 host started, prediction 9 added before 100e/f. Each pushed before its hosts started | ✓ |
| 39 | "New" machines: GPU never recorded before | as stated | UUIDs of 107b/d and 108a/d/f appear in no earlier result | ✓ |
| 40 | Cost (Appendix M) | 88 rentals, 53 machines, $70.8 (jobs 058 to 101) | matches for 058 to 101; through job 108: 124 rentals, 74 offers, $87.2 | ✗ stale |

Not checked: the nine-model trace study (W50 and the 0.65 C rule), the forecasters, the published-systems audit (13.6%), KL parity, the Shapley accounting, and the long-output and Qwen3.6 results. These rest on trace corpora or literature data rather than the raw GPU rows I was given.

## Questions for the authors

1. Why is `B_host` the maximum over every probe line, including concurrent sums, rather than a robust statistic? On 107d the maximum comes from a line whose CPU component exceeds every CPU-only measurement. How do the bound fractions and implied rates change with the median line, the best CPU-only line, or theoretical DRAM and PCIe peaks?
2. On the 12 or 13 of 68 launch-budgets where the implied rate exceeds 1.05×`B_host`, what does the engine do that the probe does not: access pattern, thread placement, or page size? Do you still call Eq. 1 a bound on those machines?
3. The consumer/server split is the third class line drawn after data. Will you register the brand split, with a fixed band and at least three valid machines per class, on a fresh sample? If that is not possible, will you present Eq. 3 as an accounting identity with a residual, and drop it as a predictive relation?
4. Every new consumer machine is under-predicted at every gpt-oss cell, while the exploratory launches centre near zero. Is there a mechanism, such as DDR4 platforms or Zen 3 parts, or is this regression to the mean from fitting the form on those launches?
5. Was the exclusion of 106c from the fewest-admission claim registered anywhere? Will Table 1 show the claim with and without it, and mention 107e?
6. With 2 or 3 rounds per launch, what does the round-level bootstrap estimate, and why is it called "launch-level" in the job scripts and the supplement? Would you consider re-renting a subset of machines so that launches, not rounds, are the top-level resampling unit for at least the headline comparison?
7. Can `manifest.json` record the commit SHA the machine cloned? Can Appendix B say "published before the machine cloned and ran the job", which your data prove, rather than "pushed before the machine started", which they do not resolve?
8. R* is computed on the 30-problem trace (38.3 reads per token). The engine runs use 20 problems, for which job 106 gives 39.0. Which value does each fraction of the bound use?

## What would raise my score

- **Soundness.**
  - Qualify or replace "bound". Report Eq. 1 at a robust and at a theoretical host rate, and state in the abstract that it is probe-relative.
  - Report the cases where the engine beats the probe in Section 3, not only in Limitations.
- **The relation.**
  - Either run one more registered test restricted to the post-hoc consumer class, with the threshold, band and machine count fixed in advance, or demote Eq. 3 to an accounting device.
  - Label Section 4.4 "after the fact", and soften "is the sum of three parts".
- **Statistics.**
  - Replace or annotate within-launch intervals where a claim is about machines.
  - Give at least one headline comparison a genuine multi-launch interval.
  - State the population of every range.
- **Table 1.**
  - Add a verdict to every registered row, and tag the untagged claims (Section 4.4, the Spearman correlation, 38 to 54%, 13.6%, the forecasters).
  - Distinguish "held" from "held on the point estimate", and registered gates from post-hoc exclusions.
- **Clarity.**
  - Put a glossary in the main text and use one symbol per GPU-time quantity.
  - Turn the new-machine story of Section 4.2 into a single table.
  - Turn Appendix B into a table.
- **Artifact.** Fix the cost line, unify the definition of round spread, record the cloned SHA, and make the ledger's hourly prices real or label them as caps.

## Scores

| Criterion | Score |
|---|---|
| Overall (1 to 10; 6 = acceptable with revisions, 8 = strong) | **6** |
| Soundness (1 to 5) | **3** |
| Methodology (1 to 5) | **4** |
| Significance (1 to 5) | **3** |
| Clarity (1 to 5) | **3** |
| Confidence (1 to 5) | **4** |

**Rationale.** The methodology is exemplary in the parts that are hardest to fake. Predictions sit in the scripts the machines clone. Server-side timestamps confirm the ordering. Every host is reported, failures are tabulated, and the raw rows reproduce. That earns a 4 on methodology and is the main reason I would accept the paper at a workshop and look on it favourably for a fellowship. Soundness is held to 3 for three reasons: the headline "bound" is exceeded by the engine's own implied rates, the descriptive relation survives only on a post-hoc class, and the stated unit of analysis (the machine) is not the unit of the reported intervals. Significance is moderate: one GPU, two models, short outputs, modest system gains. The read-once and schedule-choice findings are the lasting results. Clarity improved from 2 to 3, but Section 4 and Appendix B still demand that the reader hold too many named things at once.
