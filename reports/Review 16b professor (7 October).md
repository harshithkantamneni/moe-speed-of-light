# Review 16b: "Where the Seconds Go" (referee: professor, scientific benchmarking), 7 October 2026

## Materials read

- **Main paper.** `paper/paper.pdf`: the whole main text (Sections 1-9) and Tables 1-4, Figures 1-5. I read it both as rendered pages (pages 1-8 viewed as images) and as LaTeX (`paper/paper.tex`).
- **Appendices.**
  - `paper/app_wsg.tex` in full: Appendices A-H, L-N.
  - `paper/app_more.tex` (Appendix K), `paper/app_traces.tex` (Appendix I), `paper/app_value.tex` (Appendix J).
  - The rendered appendix tables: 6-9, 20-29.
  - `paper/tab_prereg.tex` (Table 8).
- **Supplement.** `paper/supplement.pdf`: its preamble, the scoring rule and the first clause tables (jobs 073-078).
- **Artifact (scripts).** I read these only to learn file conventions and definitions, not their outputs:
  - `scripts/job108.py`
  - the `law`/`load` functions of `scripts/job106.py`
  - `host_info` in `scripts/panel_099.py`
  - `bandwidths()` in `gpu-branch/jobs/ec2/fetch_table.py`
- **Artifact (ledger).** `gpu/vast_ledger.json`, all 124 rentals.
- **Raw results** (`/home/claude/gpu-branch/results/`), for jobs 081, 084c, 089, 090, 093-108:
  - `ec_*.jsonl` per-problem rows;
  - `st_*.json` counters;
  - `concur.txt` probes;
  - `g_prof.json`, `q_prof.json` and `prof_C*.json` Nsight summaries (including 069c);
  - `bs1.jsonl` client rows;
  - `readsched_C*.txt`;
  - `v0.txt`, `validity.txt`, `free.txt`, `cpu.txt`, `lscpu.txt`, `cores.txt`;
  - `route_aime25_gptoss.npz` (the AIME routing trace).
- **Job scripts.** The headers of `jobs/089, 102, 104, 105, 106, 107, 108`. For 100, 102, 105, 106 and 107 I read the header at its first commit and diffed it against later commits.
- **Git.** `git log --format='%h %ad %cd'` on every job script of jobs 093-108. I did not read commit messages.

## Independence statement

I did not open anything under `reports/` (other than writing this file). I also did not open:
- `prereg/*outcome*.md` or any other file under `prereg/`;
- `research_notes/`, `apply/`, `paper/paper_v1_prereview.tex`;
- any review, number-check, plan or progress-log file.

I did not read commit messages. Every number in the "Claims checked" table comes from my own code. That code is in `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/rev16b/` (`rvload.py`, `c105`-`c108`, `cplan.py`, `call.py`, `cmin.py`, `creg.py`, `cunit.py` and inline scripts), and it reads only the raw files listed above. I wrote my own implementations of:
- Eq. (3), in closed form;
- MIN with bypass on the AIME trace;
- the paired and machine-level bootstraps;
- the probe parser.

I borrowed two definitions from the authors' code because the paper does not state them precisely enough to reimplement: the Nsight category sum that defines G, and the helper-count interpolation that defines B_c. I modified nothing in either repository.

---

## Summary

The paper studies batch-1 decoding of two Mixture-of-Experts models (gpt-oss-120b, Qwen3-30B-A3B) on rented RTX 5090 hosts whose experts mostly live in host DRAM. It makes four contributions.

1. **Two lower bounds on time per token.**
   - Eq. (1): Belady's MIN-with-bypass reads (R*), timed at the machine's highest probed host-read rate (B_host).
   - Eq. (2): a tighter bound for systems that read on demand, which adds the GPU's non-expert time in series.

   A pure-read microbenchmark reaches 86-96% of the bound's read time. Against Eq. (1), the authors' llama.cpp cache runs at 31-54% of the bound's speed on 25 consumer machines. It leads FreeToken at 11 of 12 cells and stock llama.cpp everywhere.
2. **A time relation, Eq. (3).** It is G (profiled GPU kernel time) plus counted misses and background admissions at B_host, with an overlap term. It describes the deployed cache on consumer machines in-sample (median |error| 1.7%). It passed none of four registered tests:
   - plain form: failed;
   - overlap form, job 106: failed;
   - job 107: inconclusive;
   - job 108: failed.

   Two of three valid server machines read far below the probe's rate.
3. **An in-engine oracle study.** Foresight pays when MIN's admissions are read once. Among MIN's hit-optimal schedules, the one with the fewest admissions beats the deployed cache at gpt-oss 11% on all 9 stable machines (mean 1.22x). How much any configuration gains tracks the link-to-CPU read ratio.
4. **A trace study.** The window that closes half of MIN's read gap corresponds to about 0.65 C distinct experts per layer, a horizon no tested forecaster approaches.

The artifact registers predictions in job-script headers committed before each rental. It scores 1,483 clauses and reports every failed and gated host.

## One-paragraph verdict

This is the most carefully registered and most honestly reported performance study I have refereed. I recomputed 30+ quantitative claims from raw rows, counters, probes and profiles. Every one reproduced, most to the printed digit (table below). The registration timeline holds:
- 71 rentals;
- every header commit at least 2.7 s before its rental;
- later header edits limited to host substitutions, plus one added prediction.

What keeps it from "strong" is not dishonesty but what survives the honesty:
- The title claim, "where the seconds go", rests on a relation that failed every registered test and is biased in a budget-dependent way.
- Several evidence labels are more generous than the registrations behind them.
- The bound is defined relative to a single maximum probe reading that is fragile on at least one machine.
- The paper remains hard to read.

It is a sound, methodologically exemplary measurement study of moderate significance. It needs revision mainly in framing, labels and presentation.

---

## Strengths

1. **Registration practice that actually binds.** I verified the following against the ledger and git:
   - Every one of the 71 rentals of jobs 093-108 was preceded by a commit to its header. The minimum lag is 2.7 s (rental 105f).
   - Gates V0/V1/V2 and the "fewer than three valid hosts makes the test inconclusive" rule were in job 107's header at its first commit, before any machine started.
   - 22 rentals saw later commits to their headers. I diffed every one. Jobs 102, 105, 106 and 107 changed only the host lists, and each amendment is annotated "predictions unchanged". The exception is job 100, which added prediction 9 for its replacement hosts (see Weakness 6).
2. **Negative results are in the abstract.** Writing "this relation passed none" of four registered tests in the abstract, with the size of the misses, is rare and admirable. Table 4 is a model of how to report a registered test series.
3. **Failed hosts are reported, not dropped.**
   - Every gated, unsteady or wrong-output host appears in Tables 23-25 with its numbers.
   - The ledger lists rentals that produced nothing.
   - Job 108's registered offer list said none of its offers was in the ledger. Three were (I confirmed offers 53424353, 54156078 and 51325952 in the ledger), and the paper says so.
4. **Numbers are generated, and they reproduce.** Every table row I recomputed matched (Tables 3, 22-25, 26, 28; Table 8's sums). So did the running example and the abstract's 59%/51%/11% figures.
5. **The statistical unit is taken seriously.** The paper shows that machines, not problems, carry the variance. I recomputed this:
   - Machines carry 96.5-98.3% of the variance of the log gain, problems 1.1-2.9%.
   - Two panel hosts with the same CPU differ by 29.1% in time.
   - Relaunches of the same machine stay within 3.2%.

   It then reports machine-level intervals for its main pooled claims. It also flags its own Hoefler-Belli departures (Appendix A, row 3).
6. **The bound is well posed, and its scope is stated:** exact routing, whole experts, C per-layer slots, one token per pass, relative to the probe. It is applied with the same procedure to the system and to 52 published rows. MIN's R* is easy to recompute: I got 38.32 / 15.34 reads per token at 11% / 25%. My result also confirms the stated -0.4% change on the first 20 problems.
7. **The oracle study is an in-engine factorial, not a simulation.** It shows two things:
   - how foresight is spent (one read versus two, greedy versus fewest-admission) matters as much as having it;
   - the machine's link-to-CPU ratio decides the sign of several effects.

   This is a useful corrective to "speed-up over a baseline of its choosing" papers.

## Weaknesses (most important first)

### W1. The title claim (the decomposition) is exploratory and the relation behind it is misspecified

Figure 2 and §4.4 split each machine's time into three parts: G, MIN's reads at B_host, and the excess reads. These three parts are outputs of Eq. (3), not separate measurements. Eq. (3)'s record:

- **Plain form (job 105): failed.** I recomputed 5/8 cells within 6% and a median 4.2%. The 25% cells are off by +6.9 to +9.3%.
- **The overlap term came after the failure and is not identified.** It was added after job 105 failed. On job 105's own data it gives 8/8 within 6% (median 2.6%), which is in-sample. The paper concedes that discounting admissions by half fits as well.
- **Job 106: failed as registered.** 12/20 cells within 6%, 0/8 on the two new machines.
  - The "held (12/12, median 2.3%)" result is on three machines rented before (I confirm 12/12 and 2.3%; one cell is at +5.99%).
  - Two of those three, Pf and the Threadripper 9960X, ran in job 105, whose failure motivated the overlap term. The result is therefore not out-of-sample.
- **Job 107: inconclusive.** One of eight cells within 6%.
- **Job 108: failed.**
  - 4/6 within 8%, 2/6 within 6%, median 6.7%.
  - Without the unprofiled EPYC: 4/4 within 8% but a median of 5.3%, so the pooled clauses still fail.
- **The bias is systematic.**
  - On every new consumer machine the relation under-predicts every gpt-oss cell: -0.9 to -7.1%, and -11.3% for Qwen3 on the 5900XT.
  - Over the 25 consumer machines, the mean error is -2.2% [-3.7, -0.4] at 11% and +1.6% [+0.2, +3.1] at 25%. Of the 30 found-on launches, 25 are under-predicted at 11% and 23 over-predicted at 25%.
  - Something budget-dependent is missing from the model, and the residual is not noise.

The paper says all of this; I am not alleging concealment. But the consequence is that the decomposition is an accounting identity under a model that is known to be wrong by a few percent, with a sign that flips. It is further weakened because G itself is a fallback median, not a profile, for 30 of the 39 consumer launches. The title and contribution 2 should be framed accordingly. A direct measurement would strengthen it considerably. One option is the Nsight per-token `ec_wait` time already in `prof_*.json`, which measures how long the GPU waits on reads independently of Eq. (3).

### W2. Validity, exclusions and post-hoc populations

The gates are sensible and, for jobs 107 and 108, registered. Three things still need care.

- **Job 106 had no validity gate.** "Base varies by at most 2% between rounds" was registered as prediction 8, not as a gate; I read the header at commit 525a9c9. Table 4's "Set aside" column and the §4.2 heading ("The first test: held on machines rented before, failed on new ones") still turn a failed prediction into an exclusion criterion for a different prediction. The column is labelled after the fact, but the heading puts the post-hoc result first.
- **The processor-family split is post hoc, and it is not the class that was registered.** Job 108's registered class was "one NUMA node, at most 32 usable cores". It admitted two EPYCs, and it failed. The paper then explains the misses by family (consumer versus server), after the fact, and says "we have not established why".
- **Populations defined by passing checks.** Of the 13 new rentals in jobs 107 and 108, only 5 were valid: 5 stopped at the gate, and 3 failed the round check. The host-memory gate (≤ 48 GB in use) was set after the failing hosts of job 106 showed 131-132 GB in use. Claims of the form "on every machine that ran stably" (the fewest-admission gain, the relation's fit) are therefore made on a population defined by checks that correlate with the outcome. For example, the fewest-admission schedule lost on 2 of 4 unsteady machines (0.846x and 0.974x).

**Recommendation.** Report each machine-class claim also over all machines that produced timed data (intention-to-treat), with the stable subset as the secondary analysis.

### W3. B_host is a fragile estimator, and the paper does not say how its rates are defined

B_host is the single highest of roughly 20 heterogeneous readings from one probe run: CPU reads at several thread counts, PCIe copies by several methods, and concurrent sums. Its fragility shows on one machine:

- On the dual-socket engineering sample (107d), B_host = 204.5 GB/s is one concurrent zero-copy reading at 128 threads. Its CPU component (181.4 GB/s) exceeds every CPU-only reading (≤ 151.4 GB/s), and the same configuration with the copy engine gives 123.5. I believe this is a probe artefact.
- That one reading produces the abstract's "59%". With the second-highest reading the miss is 51%, which the abstract does mention.

Across the 47 launches, the median gap between the highest and second-highest reading is only 0.82%, so the problem is local. But the estimator is a maximum over noisy samples, it has no repeat runs, and it reports no dispersion.

The paper also does not define its rates in a reproducible way:
- B_c is the CPU read rate interpolated at the helper count (cores - 2).
- B_p is one zero-copy mode (16 MB chunks, 64 blocks).
- B_cp is the median of the concurrent sums.

These definitions are in `fetch_table.py`, not in the paper. The link-to-CPU threshold that appears in the abstract ("links at least 0.28") depends on them. My naive max/max ratios differ by up to 0.06 (for example, 0.34 against the paper's 0.32 for the Threadripper). Table 3's header ("together 78, highest 88") cannot be decoded without the code: 78 is the t=16 zero-copy sum and 88 the t=8 one.

Finally, the paper presents "the engine read more than 5% faster than the probe on 12 of 84 launch-budgets" as evidence that a faster reader could beat the bound. I reproduce 12/84 and a maximum of 1.22x. But 11 of the 12 are at the 25% budget, exactly where Eq. (3) is known to over-predict. They mostly measure Eq. (3)'s bias, not the probe's.

### W4. Statistical summaries

- **Within-launch intervals are sound; their scope is not.** I tested serial dependence between problems, since the cache carries over between them. The median lag-1 autocorrelation of the per-problem log gain is 0.00 over 81 launch-configurations, and a block bootstrap does not widen the intervals (median width ratio 0.87). So the paper's own worry about non-exchangeable problems is not material. The real limit is that these intervals describe one rental. O4 and O5 give MIN-in-step gains of 1.36 and 1.16 at 11%, with problem-level intervals of about ±0.01.
- **Averaging ratios contradicts the paper's checklist.**
  - The admission-margin claim averages each machine's ratio "over the two gpt-oss budgets". Appendix A, row 4 says ratios "are never averaged across budgets or models".
  - The 1.22x fewest-admission figure is an arithmetic mean of ratios over a bimodal set of machines: 1.03-1.18 on slow links, 1.34-1.42 on fast links.
  - Use geometric means or, better, the conditional summary that Fig. 4 already supports.
- **Mixed unit for the elasticity.** The quoted -0.95 matches a per-launch fit (-0.950, 30 launches), while its interval is described as "over machines". A machine-level fit gives -0.97 [-1.08, -0.82].
- **Small samples.** Machine-level intervals exist for three claims, on 5-25 machines of a convenience sample (the cheapest offers that passed the gates). The paper says this; the abstract's "every machine that ran it stably" should carry the n (9) and the class.

### W5. Evidence labels are sometimes more generous than the registrations

| Location | What the paper says | What the registration supports |
|---|---|---|
| Table 1, row 3 | "registered on host B, held; host S observed" | Host S is job 089, which registered ours/FreeToken within ±0.06 of host B's ratio. That failed at 5/6 cells (I recomputed -0.087, -0.079, -0.061, -0.005, -0.105, -0.075). At Qwen3 43.75% the registered band [0.989, 1.109] implied a lead; the measurement is 0.974 [0.962, 0.987]. Suggested label: "registered; replication bands failed at 5/6". |
| Table 1, row 2 | "nearly reachable: registered thresholds held" | Job 102 registered token mode ≥ 0.80 and per-layer ≥ 0.50. "Nearly reachable" (86-96%) is the observation, not the registration. The one tight prediction (per-layer waiting costs ≥ 0.03) failed at 4 cells. |
| §5 | "on the other two it follows from a registered comparison with the greedy schedule" | Job 104 registered only fewest - greedy ≥ 0.03 on hosts with ratio < 0.7. Fewest > deployed on 104b and 104c also needs the unregistered greedy/deployed values (1.05, 1.26). Table 1's "observed on two" is the right label; the text overstates. |
| Table 1 caption | "Held is mostly on the point estimate" | True, and worth stating more strongly. Over all jobs, 495 clauses held with an interval and 597 on the point estimate only (job 106: 21 against 112). With 1,483 clauses and no designated primary outcomes, the held/failed counts are hard to interpret. |

### W6. Remaining gaps in the registration record

- **Undisclosed mid-job amendment (job 100).** Job 100's header gained prediction 9 at 12:06:35Z, after 100a-d had launched (10:26Z) and after 100a had finished. The prediction concerns the replacement hosts 100e/f, which had not started, so it is legitimate. But it compares them with Pd's values, one of which (100a) was already measured, and the paper does not mention the amendment. Job 107's comparable amendment (the replacement rule added after the first four results) is disclosed.
- **Bands and rules drawn from data.** Job 108's 8% band was drawn from the found-on data (35/37 within 8%). G's fallback, a deviation from the registration, was decided after the EPYC traces came back empty. Both are disclosed; together they show how many analysis choices sit around a "registered" test.
- **No designated primary outcomes.** Every job registers many clauses (job 099: 273; job 106: 181) with no designation of which ones carry the claim. I would ask for 1-3 primary outcomes per job, with the rest marked secondary.

### W7. Scope and significance

The study covers:
- one GPU model;
- two main models (three counting Qwen3.6);
- AIME prompts that were also used during development;
- teacher-forced decoding at batch 1.

The lead over FreeToken is host- and model-dependent: 1.03-1.29x on host B, 0.97-1.21x on host S, a tie on Qwen3.6. The bound is a natural roofline-plus-Belady construction, so novelty there is incremental. The most original results are:
- the "read once" requirement;
- the fewest-admission schedule measured in an engine;
- the horizon rule in distinct experts, which is exploratory and trace-level.

The decomposition, the paper's headline, is the least confirmed part.

### Minor

- **CPU last-level cache.** Eq. (1) assumes CPU-run experts are read from DRAM. On X3D parts (96-128 MB of L3) a policy could keep about 7-9 gpt-oss experts in L3 as a second resident tier. The paper checks MIN's re-read distance (0.5 GB) but not whether a different policy could exploit L3. State this in the bound's scope, or address it.
- **Fig. 2 launch choice.** The paper does not say which launch represents a machine that ran several times. Using the latest launch, I get G 12-44% (paper: 12-45%) and serialisation 22-63% of the gap (paper: 23-66%).
- **KL parity is not recomputable.** The full logit dumps are not in the repository; only the on-machine summary and top-5 dumps are. I could recompute the loss change from the rows (+0.16% / +0.19%) but not the KL.
- **V1 on older jobs.** V1's 0.190 reference applies to the 20-problem protocol. The 30-problem launches 093-098 have a mean loss about 9% higher. The after-the-fact correctness check for those jobs must use a different reference; say which.

---

## Clarity

Earlier rounds scored clarity 2, 3 and 3. The current version is readable for a determined expert, but not yet clear.

### What works

- **The abstract** states the bound, the registered failure (with its sizes) and the foresight findings in eight sentences, and it does not oversell.
- **Navigation tables.** Table 1 (claim, evidence kind, section) is the right device; it lets a reader calibrate each claim before reading it. Table 4 compresses four registered tests into four rows with "as registered" and "set aside" kept apart. Table 2 names the configurations with their read counts, and Table 6 maps appendix codes to those names.
- **Figures and the running example.** Fig. 1's step schematic and Fig. 2's per-machine stacked bars (hatched area = residual, dot = measurement) communicate the model well. The running example (10.0 / 13.0 / 20.6 / 15.2 / 13.8 ms, all of which I reproduced) anchors §5.
- **Few codes in the main text.** It is largely free of job and host codes; only O3-O5, B, S and jobs 105-108 appear.
- **Specific limitations.** Section 8 names the failure modes rather than offering boilerplate.

### What still makes it hard to read, most important first

1. **Number density.** The main text, with its tables, carries roughly 1,000 numerals in about 6,200 words, about one number every six words. Many paragraphs are runs of ranges with parenthetical qualifications.
   - The first paragraph of §4.3 carries about fifteen numbers and three hedges.
   - §4.2's "Machines never rented before" paragraph switches between the band, the class, the probe's second-highest reading and two hosts within three sentences.

   A reader cannot tell which number is the finding and which is context. Suggested rule: one claim per paragraph, one headline number with its interval, and a pointer to the table.
2. **Organised by chronology, not by claim.** §4 tells the history:
   - the plain form, which failed;
   - the overlap form;
   - "the first test", "the next test", "a last test";
   - a post-hoc split;
   - the decomposition;
   - the server anomalies.

   The reader has to assemble the verdict. Better: state Eq. (3), show Table 4, give one paragraph on what it means (biased by budget, fails on servers, untested out-of-sample on consumers at scale), then the decomposition labelled exploratory. Move the history to the appendix.
3. **Quantities that look alike, and rates defined only in code.**
   - There are three GPU times: T_GPU (Eq. 2), G (Eq. 3) and G_fit (Eq. 4). There are four host rates: B_c, B_p, B_cp, B_host.
   - The rates' operational definitions are only in code (see W3), yet the abstract's threshold "0.28" depends on them.
   - Table 3's "together 78, highest 88" invites the question of how "highest" can exceed "together".
   - Fix: define each rate in one sentence in §2 and drop G_fit from anything the main text cites.
4. **Shifting baselines and hedged negations.**
   - §5 measures against the deployed cache. §6 switches to "admit every miss" in mid-paragraph: "Measured instead against admitting every miss, the online policy that reads least at these budgets (and not against the deployed cache, as in Section 5)...". Fig. 5's y-axis then uses that second baseline.
   - Sentences such as "The processor separates most of the machines the relation fits from those it misses, not all; we drew this split after the tests" make the reader parse three qualifications before learning the point.
   - Fix: pick one baseline per section and state it in the section's first sentence.
5. **An appendix that reads as a lab notebook.** The appendix runs to 27 pages (pages 14-40) and holds most of the paper's 29 tables; "Failures, jobs 073-098" is a single paragraph of about 25 lines. This history is valuable as an audit trail, but it belongs in the supplement. The appendix should hold the evidence for the main claims.
6. **Mixed summary conventions, with the reader left to track them.**
   - Table 3 averages per-problem rates arithmetically, a declared departure.
   - Jobs 093-097 report ratios of mean speeds; later jobs report ratios of mean times.
   - Machine-level summaries are arithmetic means of ratios, one of them across budgets.
7. **The text and Table 1 disagree on labels** (W5): host S, and the two machines where the gain is said to "follow from" a registered comparison. When a careful reader finds the label in the text stronger than the label in the table, trust in both drops.
8. **Fig. 3.** Its 95% intervals are invisible at the plotted scale. The ten panel hosts are thin grey ticks that overlap the O3-O5 markers. Two kinds of evidence (three hosts at all budgets; a panel at two budgets) share one plot without a visual hierarchy.

Overall: 3/5. The scaffolding (Tables 1, 2, 4 and 6, Figs. 1 and 2, the running example) is now good. The prose inside it has not been thinned enough to reach 4.

---

## Claims checked against raw data

All recomputations use my own code on raw rows, counters, probes, profiles or traces. "✓" means it reproduced to the printed precision; "≈" means it reproduced up to an unstated choice that I describe.

| # | Claim (location) | Paper | My recomputation | |
|---|---|---|---|---|
| 1 | Table 25, job 108: Eq. (3) / measured / implied rate, 7 rows | e.g. EPYC 7543 11%: 14.9 / 22.7 / 0.58; 5950X 25%: 16.5 / 16.7 / 0.99 | All 7 rows identical (14.94 / 22.73 / 0.58; 16.52 / 16.68 / 0.99; ...) | ✓ |
| 2 | Job 108 pooled (Table 4) | 4/6 within 8%, 2/6 within 6%, median 6.7%; without the 7543: 4/4, 5.3% | 4/6, 2/6, 6.7%; 4/4, 5.25% | ✓ |
| 3 | Abstract and §4.2: largest misses | 59% (51% vs the second-highest reading); 11% on a new consumer machine | -58.9% (-51.4%); -11.3% (5900XT, Qwen3 12.5%) | ✓ |
| 4 | Table 24, job 107: 8 Eq. (3) cells; 1/8 within 6%; median | 31.7% | Same cells; 1/8; 31.75% | ✓ |
| 5 | Job 107 gates and validity | 3 gated at 50-96 GB; rounds 3.5% and 6.0% apart; under-predicted by 31% and 21% | 50 / 50 / 96 GB; 3.54% and 5.98%; -31.1% and -20.9% | ✓ |
| 6 | Job 106 stable hosts (Table 4, "set aside") | 12/12 within 6%, median 2.3% | 12/12 (one cell +5.99%), median 2.34% | ✓ |
| 7 | Job 106 as registered (Table 4) | 12/20; new machines 0/8; median 4.7% | 12/20; 0/8; 4.67% | ✓ |
| 8 | Job 106 unstable hosts | Loss 0.21-0.45 nats (EPYC 7302); rounds 52% apart (Xeon); 131-132 GB in use against 12-15 GB elsewhere | 0.208-0.446; 52.2%; 132 / 131 against 12 / 15 / 12 | ✓ |
| 9 | Job 105, plain form (Table 22) | 5/8 within 6%, median 4.2%; +6.9 to +9.3% at 25% | 5/8, 4.24%; +6.9, +7.0, +9.3% (Pf +3.8%). The overlap form on the same data: 8/8, median 2.6% (in-sample) | ✓ |
| 10 | Fallback G (median of 7 profiles, jobs 069c and 105) | 4.28 / 4.50 ms; profiled G 4.1-4.7 ms | 4.284 / 4.498; range 4.07-4.69 | ✓ |
| 11 | Fewest-admission schedule, machine level | Mean 1.22x over 9 machines, 95% 1.14-1.31, t-interval 1.11-1.33 | 1.222, [1.135, 1.307], t [1.113, 1.331]; values bimodal (1.03-1.18 and 1.34-1.42) | ✓ |
| 12 | Fewest-admission on unsteady machines | Lost on 2 of 4 | 0.846 (EPYC 7302), 0.974 (EPYC 7663); gained 1.326 (Xeon), 1.175 (EPYC 9754) | ✓ |
| 13 | Admission margin (dk) | 1.020x (1.012-1.031), t 1.004-1.035; up to +6%, -1% at 1 of 10 cells | 1.0197 [1.0115, 1.0302], t [1.0042, 1.0353]; max 1.057, min 0.987, 1 losing cell | ✓ |
| 14 | Fewest versus greedy counters | 0.60-0.73 of greedy's copies; 2.5-5.3% more misses | 0.597-0.726; +2.5 to +5.3% | ✓ |
| 15 | R* on the first 20 problems (§3) | -0.4% at 11% and at 25% | My MIN-with-bypass on the 084c trace: R* = 38.315 → 38.170 (-0.38%); 15.340 → 15.282 (-0.38%) | ✓ |
| 16 | Table 3 bound at gpt-oss 11% | Host B 172 tok/s, host S 140 tok/s | 172.3, 140.4 (B_host 87.5 and 71.3) | ✓ |
| 17 | Table 3, host B gpt-oss | Ours/FreeToken 1.294 [1.278, 1.312], 1.275, 1.154; llama.cpp 34.9 / 40.0 / 47.7 | Identical (launch 2); 34.9 / 40.0 / 47.7 | ✓ |
| 18 | Table 3, host S, against job 089's registered bands | 1.207 ... 0.974; Table 1 says "observed" | Identical ratios. The registered ±0.06-of-host-B bands fail at 5/6 cells. | ✓ numbers; label issue (W5) |
| 19 | Running example (O4, 9950X, gpt-oss 11%) | Eq. (1) 10.0 ms; deployed 20.6; MIN 1 read 15.2; prefetched 13.8 | 10.02; 20.64; 15.16; 13.81 | ✓ |
| 20 | MIN with one read on O3-O5, all 6 budgets | Gains 16-51%; up to 81% copied ahead; "usual ways" at most +15% at the smallest budgets | 1.160-1.514; max 1.812; max 1.148 | ✓ |
| 21 | Consumer launches (§4.3) | 39 valid launches on 25 machines; implied rate 0.88-1.16; 37/39 within 8%; G profiled on 9 | 39 / 25; 0.88-1.16; 37/39; 9 | ✓ |
| 22 | Server launches | 8 launches, 5 invalid; valid ones at 0.27 / 0.58 / 1.04 | 8 / 5; 0.27 (AMD ES), 0.58 (7543), 1.04 (7402P) | ✓ |
| 23 | Error over consumer machines | -2.2% (-3.7, -0.4) at 11%; +1.6% (0.2, 3.1) at 25%; found-on launches: 25/30 under at 11%, 23/30 over at 25% | -2.16 [-3.73, -0.42]; +1.57 [+0.22, +3.09]; 25/30; 23/30 | ✓ |
| 24 | Table 26 | 31/39 within 6%; median 1.7%; 12 of 84 launch-budgets above B_host by more than 5% (max 1.22x) | 31/39; 1.69%; 12/84, max 1.223 (11 of the 12 at 25%) | ✓ (interpretation, W3) |
| 25 | Probe readings | Second-highest a median 0.8% lower; more than 5% lower on 1 launch (107d, 26%); 118 against 151 GB/s with 128 against 16 threads | 0.82%; 1 launch (107d, 26.0%); 117.6 / 151.4 | ✓ |
| 26 | Elasticity of T - G on B_host | -0.95 (-1.06 to -0.84) "over machines" | Per-launch fit -0.950 (n=30); machine-level -0.973 [-1.08, -0.82] | ≈ (unit mixed) |
| 27 | Fig. 2 / §4.4 shares at 11% | G 12-45%; MIN 31-54%; beyond MIN 22-35%; serialisation 23-66% of the gap; below-rate ≤ 11%; servers 0 / 34 / 59% | Latest launch per machine: 12-44; 31-54; 22-35; 22-63; 11; servers -3 (0) / 34 / 58% | ≈ (launch choice unstated) |
| 28 | Spearman of MIN-in-step gain against link ratio | 0.89 [0.63, 0.97] over 19 machines | 0.88 [0.62, 0.97] over 19 machines | ✓ |
| 29 | Variance decomposition (§5, App. K) | Machines 96%, problems ≤ 3% | 98.3% / 96.5% machines, 1.1% / 2.9% problems (19 machines × 20 problems) | ✓ |
| 30 | Machines as the unit (§2) | Same-CPU panel hosts differ by up to 29%; relaunches within 3.2% | 29.1% (two 285K hosts); 3.2% (Threadripper 105b against 106a) | ✓ |
| 31 | Table 28, read microbenchmark (job 102) | 86-96% per layer, 93-97% per token | 86, 89, 96, 96, 95, 93% per layer; 93-97% per token | ✓ |
| 32 | Registration timing (App. C) | 71 rentals of jobs 093-108; every commit ≥ 2.7 s before its rental | 71; minimum 2.7 s (105f). 22 rentals saw later header commits: host swaps only, except job 100's prediction 9 (W6) | ✓ |
| 33 | Table 8 totals | 495 / 597 / 308 / 83; jobs 073-098: 565 clauses (218 / 193 / 142 / 8 + 4) | Rows sum to the same; per-job counts match the appendix | ✓ |
| 34 | Output parity: loss change | +0.16% / +0.19% | +0.160% / +0.192% from the rows; KL not recomputable (full dumps not released) | ✓ / unverifiable |
| 35 | Job 108 offer list against the ledger | Three listed offers were in the ledger | 53424353 (100b), 54156078 (099a), 51325952 (099d, 100e) | ✓ |
| 36 | My own test, behind the Limitations remark that problems are not exchangeable | (paper's worry) | Lag-1 autocorrelation of per-problem log gain: median 0.00 (n = 81); block-bootstrap intervals not wider | Worry not material |

Not checked: the trace-level horizon study (W50 and 0.65 C across nine models), the audit of 52 published rows, and the 099/100 panel windows beyond the variance decomposition.

---

## Questions to the authors

1. **The probe.** Why is B_host the single highest reading of one probe run? What do repeated probes on the same machine give (spread)? Please recompute the bound-relative numbers (31-54%, 59%, the implied rates) with a robust statistic, such as the median of the top three readings, and state B_c, B_p, B_cp and the ratio operationally in §2.
2. **Measuring the decomposition directly.** Can the decomposition be measured rather than computed from Eq. (3)? The Nsight summaries already split GPU kernel time from `ec_wait`. Does G + waited time match T on consumer machines, and what does it say about the 25% over-prediction?
3. **Job 106's round criterion.** Was "base varies by at most 2% between rounds" intended as a gate when job 106 was registered? If not, why is "held (12/12)" a column of Table 4 and the heading of §4.2's first paragraph?
4. **Intention-to-treat.** What do the fewest-admission and relation claims look like over every machine that produced timed data, not only those that ran stably?
5. **Host S.** Will you relabel host S in Table 1 as registered, with the replication bands failed at 5/6? And will you change "follows from a registered comparison" in §5 to "observed"?
6. **CPU cache.** Could a policy that keeps hot non-resident experts in a large L3 (X3D parts) beat Eq. (1) as stated? If so, should the bound's scope exclude CPU-cache residency explicitly?
7. **Held-out text.** The engine experiments replay AIME text that was also used during development. Do the read-once and fewest-admission results hold on held-out text (e.g. the MATH-500 or mixed-domain traces you already have)?
8. **Job 108's band.** The 8% band was drawn from the found-on launches. Given the relation's budget-dependent mean error (-2.2% at 11%, +1.6% at 25%), was a bias-corrected prediction considered for registration? What band would the found-on residuals imply at 95%?
9. **Primary outcomes.** Which 1-3 clauses per job would you designate, in hindsight, as primary? How would the scorecard read if restricted to them?

## What would raise my score

- **Reframe around the confirmed results** (to 7): the bound and its reachability, read-once foresight, the fewest-admission schedule, and machine dependence. Present the decomposition as exploratory accounting with its budget-dependent bias quantified, and adjust the title or subtitle accordingly.
- **Fix the labels:** host S, job 102's loose thresholds, and the two "follows from" machines. Designate primary outcomes per job and summarise only those in the main text.
- **Make the probe robust:** repeated probes with dispersion and a robust B_host, plus a sensitivity line for every bound-relative claim. Define the rates and the ratio in the text.
- **Add intention-to-treat** summaries next to the stable-only ones.
- **Fix the summaries:** use geometric means or conditional summaries of ratios, and remove the cross-budget averaging (or amend checklist row 4).
- **Clarity** (to 4): halve the numbers in the main text, put one claim per paragraph, move the job-by-job history to the supplement, and use one baseline per section.
- **Directly measured parts** (toward 8): validate the decomposition with measured per-part times on new machines, or register and pass a bias-corrected relation on a fresh consumer sample of at least five machines.

## Scores

| Criterion | Score |
|---|---|
| Overall | **6 / 10** (acceptable with revisions; a strong 6) |
| Soundness | **4 / 5** |
| Methodology | **4 / 5** |
| Significance | **3 / 5** |
| Clarity | **3 / 5** |
| Confidence | **4 / 5** |

Reasons:
- **Overall.** The registration and reporting are exemplary. The headline relation failed its tests, and the presentation remains heavy.
- **Soundness.** Every number I checked reproduces, and the conclusions as worded are mostly supported. Some labels and one framing overreach.
- **Methodology.** Registration is near-exemplary. The deductions are for the max-of-readings B_host, the post-hoc exclusions, and the absence of primary outcomes.
- **Significance.** A careful measurement study with incremental bounds and some original oracle findings, on one GPU model.
- **Clarity.** Good scaffolding; prose still too dense.
- **Confidence.** I recomputed 30+ claims from raw data. I did not check the trace-level horizon study or the audit.
