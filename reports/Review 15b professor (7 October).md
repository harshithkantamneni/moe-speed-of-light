# Review 15b: "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth"

Referee role: professor in a systems group, refereeing for a fellowship and a workshop, with scientific benchmarking (Hoefler and Belli, SC'15) as the yardstick. Date: 7 October 2026.

## Materials read

- **Main paper.** `paper/paper.pdf` (40 pages, built 7 Oct 12:24 CDT). I read the text via `pdftotext` and looked at rendered pages 1–2 and 4–8. I also read the LaTeX source `paper/paper.tex` in full, and the appendices in full: `app_wsg.tex` (Appendices A–N), `app_more.tex`, `app_traces.tex` and `app_value.tex`.
- **Tables and macros.** `tab_regtests.tex`, `tab_configs.tex`, `tab_names.tex`, `tab_prereg.tex`, and the macro files `wsg_regtime`, `wsg_regtests`, `wsg_decomp`, `wsg_robust`, `wsg_sumlaw`, `wsg_job105`–`108`, `wsg_pending` and `wsg_rebase`.
- **Supplement.** `paper/supplement.tex` and the structure of `paper/supplement.pdf`. These are the clause-by-clause scorecards; I did not audit them clause by clause.
- **Artifact.**
  - `gpu/vast_ledger.json` (all 124 rentals).
  - From `prereg/`: `gpu_pushes.json`, `online_admit.json`, `speed_limit_v2.json`, `reanalysis.json` (the G values only), `reanalysis_hosts.json` (structure) and `foresight/w50_distinct.json`.
  - From `scripts/`, only to learn definitions: `speed_limit.py:host_rates`, `factorial_shapley.py:limit1_of`, the header and `profiles()` of `sumlaw_paper.py`, and the first 130 lines of `fig_decomp.py`.
- **Raw results** (`/home/claude/gpu-branch`).
  - **Job script headers:** jobs 089, 102, 104, 105, 106, 107 and 108 (the templates and the per-host scripts).
  - **Header diffs** between commits for 105, 106, 107, 073, 061, 066 and 069.
  - **Result files:** `results/<job>/` for every launch of jobs 093–108 (`ec_*.jsonl`, `st_*.json`, `concur.txt`, `g_prof.json`/`q_prof.json`, `prof_*.json`, `validity.txt`, `gate.txt`, `cpu.txt`, `nvidia-smi-q.txt`, `manifest.json`), plus 069c's profiles, 081/089 `bs1.jsonl`, and 102's `readsched_*.txt` and `rates.txt`.
  - **Git metadata:** `git log --format='%h %at %ct'` and `git diff` content only.

My own analysis scripts are in `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/rev15b/`: `lib.py`, `regtests.py`, `survey.py`, `explo.py`, `fs.py`, `plan.py`, `online.py`, `spear.py`, `varshare.py`, `headline.py`, `push.py`, `regtime.py`, `robust.py`, `gmed.py` and `es.py`. All of them read the raw rows and counters directly. Apart from the macro definitions listed above, they share no code with the artifact's analysis scripts.

## Independence statement

- I did not open anything under `reports/` before writing this file.
- I did not open `prereg/*outcome*.md`, `research_notes/`, `apply/`, `paper/paper_v1_prereview.tex` or `paper_v2_prerewrite.tex`, or any file named like a review, number check, plan or progress log.
- I did not read commit messages. I used only hashes and timestamps from `git log`, plus file contents and diffs.
- I modified nothing in either repository except writing this file, and I committed nothing.
- I read four artifact scripts only to pin down definitions: how B_host is taken, where R⋆ comes from, how T_GPU is defined, and which launches enter Fig. 2. All the numbers below come from my own code reading the raw data.

---

## Summary

The paper studies batch-1 decode of two MoE models (gpt-oss-120b in MXFP4 and Qwen3-30B-A3B in BF16) whose experts mostly live in host DRAM, on rented RTX 5090 machines. It makes four contributions.

1. **A bound.** Eq. (1) is the host reads of Belady's MIN with bypass on the routing trace, divided by the machine's highest probed host read rate B_host. Eq. (2) is a tighter, engine-specific bound for systems that read only after routing.
2. **An exploratory decomposition** of the authors' llama.cpp expert cache. Eq. (3) writes time per token as the profiled GPU kernel time G plus counted reads at B_host, with an overlap discount on background admissions. Fig. 2 then splits each machine's time into G, MIN's reads and the reads beyond MIN's.
3. **An in-engine oracle study.** Spending foresight "as MIN spends it" (each admission read once) pays. Among MIN's hit-optimal schedules, the one with the fewest admissions beats the deployed cache at gpt-oss 11% on every stable machine (mean 1.22×). Its online analogue, a larger admission margin, gains about 2%.
4. **A horizon rule.** Half of MIN's saving needs the routing of about 0.65 C distinct experts per layer.

The paper's distinctive feature is its registration practice. Every job's predictions sit in the header of a script committed to a public branch before the rental started. The paper reports 1,483 scored clauses (Table 8), a rental ledger, validity gates and four registered tests of Eq. (3), all of which failed or were inconclusive (Table 4). The paper says so in the abstract.

I recomputed 31 quantitative and methodological claims from raw data; the table is at the end. Twenty-seven reproduce exactly; one of these carries a percentage misstated in the appendix text, although the arithmetic checks out. Three reproduce within rounding or definition tolerance. One, an evidence label in Table 1, does not match the clause that was actually registered. The registration timeline checks out for all 71 rentals of jobs 093–108, and every rental from job 058 on had its script committed before it started. On numerical integrity and registration discipline this is the most careful benchmarking submission I have refereed at workshop level.

My reservations are four:

- **Framing.** The central explanatory claim is exploratory and failed every prospective test, yet it still frames the paper.
- **The bound depends on one probe reading.** The "bound" is relative to the single highest reading of one probe run, which the engine itself exceeded on 12 of 84 launch-budgets.
- **Evidence labels.** Some labels in Table 1 credit registration more than the registered clauses warrant.
- **Clarity.** It has improved but is still the main barrier to the paper being read.

## Strengths

1. **Numerical integrity is excellent.**
   - Every headline number I recomputed from `ec_*.jsonl`, `st_*.json`, `concur.txt` and the Nsight JSONs reproduced: Table 3's ratios and intervals to the third decimal, Table 4's cell counts and medians, Fig. 2's ranges, the fewest-admission and admission-margin intervals over machines, and the oracle factorial.
   - No result file of jobs 093–108 was modified after its first commit.
   - The artifact's numbers really are generated by scripts. `wsg_pending.tex` holds placeholders, but none survives into the PDF.
2. **The registration practice is exemplary and verifiable.**
   - All 71 rentals of jobs 093–108 had their script committed at least 2.7 s before the rental was created (105f is the minimum).
   - GitHub's push log puts 7 pushes within 1 s of rental creation, and 4 at or up to 0.8 s after it.
   - Every machine's own manifest shows the job starting at least 26 s after the push (100f is the minimum).
   - Amendments live in the headers, say what changed, and change only which host is rented, never the predictions. For example, 107f's amendment was committed 3.4 s before 107f started and 107g's 4.1 s before 107g.
   - In jobs 058–092, the four scripts that changed between a first and a second rental (061, 066, 069, 073) changed install steps or descriptions only.
3. **Negative results are reported prominently.**
   - Table 4 states plainly that none of the four tests of Eq. (3) passed.
   - The abstract gives the miss sizes, and the limitations name the confounds.
   - Failed hosts are listed rather than dropped: Tables 23–25, the ledger, and the gate and validity outcomes.
4. **The machine is identified as the main source of variance, with evidence.**
   - I reproduced the variance split: machines carry 96% of the variance in the log gain of MIN copied in the step, problems 1.1–2.8%.
   - I reproduced the 5-of-6 non-overlapping host intervals in Table 3.
   - Repeat rentals of the same GPU UUID came within 3.2%, while same-CPU machines differ by up to 29%.
   - Machine-level bootstraps, with t-intervals beside them where n is small, are the right response.
5. **The oracles are causal interventions inside the real engine,** not trace simulations alone. A 2×2 (what to cache × how to load) and the fewest-admission schedule separate effects that a model alone could not.
6. **Table 1 (claim, evidence type, section) and Table 2 (configurations with a "Reads" column)** are good devices that other benchmarking papers should copy.

## Weaknesses (most important first)

### W1. The paper is framed around an exploratory relation that failed every prospective test

The abstract's second finding, contribution 2, Fig. 2 and Section 4.4 all present Eq. (3), "time ≈ G + counted reads at B_host", as where the seconds go on consumer processors. I recomputed all four registered tests and they fail as stated:

| Job | Form | Cells within band | Median |err| | Notes |
|---|---|---|---|---|
| 105 | plain | 5/8 | 4.2% | |
| 106 | overlap | 12/20 | 4.7% | |
| 107 | overlap | 1/8 | 31.7% | inconclusive by its own rule |
| 108 | overlap | 4/6 | 6.7% | without the EPYC 7543: 4/4 within 8%, but two cells beyond 6% and median 5.3% |

On new consumer machines the relation under-predicts every gpt-oss cell (−0.9% to −7.1%) and misses by up to −11.3% on Qwen3 (5900XT). Over the 25 consumer machines the mean error at 11% is −2.2% [−3.7, −0.4], and at 25% it is +1.6% [0.2, 3.1]: a small but budget-dependent bias.

The consumer/server split that keeps the relation within ±8% is drawn after the fact and confounded with core count, NUMA, co-tenancy and driver version (the paper says so).

The relation is, structurally, the statement "the engine's realised read rate is about B_host". The implied rate is 0.88–1.16 × B_host on consumer launches. Given that, the elasticity of −0.95 is the same fact restated, not independent support.

What survives is a useful descriptive accounting: on desktop hosts this engine reads at roughly the probe's best rate and does not overlap reads with GPU compute. What does not survive is a validated predictive model. I would reframe the paper accordingly. Report the realised-rate factor B_eff/B_host per machine as the finding, with its machine-level distribution and the prospective misses. Present Fig. 2 as an accounting under that factor, not as "where the seconds go" in a causal sense.

### W2. The "bound" depends on one maximum probe reading and is exceeded by the engine itself

B_host is the highest single reading in one run of `concur.txt`, an extreme-value statistic. I confirmed both of the appendix's own robustness numbers:

- **The engine exceeds the probe.** By Eq. (3)'s accounting, the engine reads more than 5% faster than B_host on 12 of 84 launch-budgets (up to 1.22×). Machine 100f is the top of the "31–54% of the bound" range, and its implied rate is 1.16 × B_host.
- **One probe maximum is an outlier.** On the dual-socket ES server (107d), B_host = 204.5 GB/s comes from a single "concurrent, 128 threads, zero-copy" reading whose CPU part (181 GB/s) far exceeds the CPU-only reading at 128 threads (118 GB/s). The next-highest reading is 151.4 GB/s. With 151.4, the "59%" miss in the abstract becomes 46–51%. The conclusion survives, but the headline number is inflated by about 8 points.

Two consequences follow:

- Eq. (1) is a bound relative to the probe, not a bound on the machine. The paper concedes this in Section 3 and the limitations, but the abstract and Table 1 still call it a bound "at the machine's measured read rate".
- Every percentage of the bound inherits an uncertainty the paper never quantifies.

A benchmarking paper should repeat the probe, report the median and the spread, and either use a statistic with known behaviour or carry the probe's uncertainty into an interval on the bound.

### W3. Some evidence labels in Table 1 credit registration more than the registered clauses do

- **"Our cache leads FreeToken and llama.cpp at most configurations — registered; leads at 11 of 12."** Job 089, the host-S rerun, registered only that ours/FreeToken would land within ±0.06 of host B's ratio at every cell. That clause failed at 5 of 6 cells (my values: −0.087, −0.079, −0.061, −0.005, −0.105, −0.075). No sign-of-lead clause for host S is in the header I read. The 11-of-12 lead is an observed outcome of registered experiments, not a registered prediction that held.
- **"Fewest-admission … held on all 9 stable ones (two by implication)."** Job 104 registered "beats the deployed cache" only for Pf. For 104b and 104c it registered "beats the greedy schedule by at least 0.03". The deployed-cache comparison there follows only by combining that clause with an unregistered observation (greedy/deployed = 1.053 and 1.256). "Two by implication" should read "two observed, not registered".
- **The bold heading "Machines rented again: held."** It titles a post-hoc subset of a test that failed as registered. The 2% round-spread criterion used to set aside the two job 106 servers was a prediction there (job 106 header, prediction 8), not a gate. The paper says the checks were "applied after the fact", but a bold "held" is the wrong typographic signal.
- **"Held" mostly means the point estimate.** The intervals used for "held with an interval" are within-launch problem bootstraps, which by the paper's own argument miss the dominant variance (machines).

The fix is mechanical and would make the paper's best feature auditable. For every Table 1 row, give the job and clause numbers it rests on, each clause's scored outcome, and the type of interval.

### W4. The statistical unit is declared but not applied consistently

- **Counts are over launches.** Section 2 says the machine is the unit, but many claims count launches: "within 6% on 28 of 30 launches", "33 instead of 18 of 34", "within 8% on 37 of 39 launches on 25 machines". In the 39, one GPU (Pf, UUID …6e1a114a) appears six times and another (…38c8aecf) three times, so re-rented machines are overweighted.
- **Found-on machines are mixed in.** The "over all 25 consumer machines" error estimate includes the 21 machines on which the relation was found. The paper does also give the 21 separately.
- **Problems are not exchangeable.** The cache carries over between problems ("as in one user's session"), yet the bootstrap resamples problems as if they were exchangeable. A block or launch-level resampling, or at least a check of serial dependence, is needed.
- **Too few rounds and machines.** Only the last experiments have rounds, and only 2–3 of them, so within-machine run-to-run variance is barely estimated. The admission-margin claim rests on n = 5 machines, where the percentile interval (1.012–1.030) is noticeably narrower than the t-interval (1.004–1.035).
- **The sample is a convenience sample.** The machines are the cheapest verified Vast offers that pass the gates, and the paper does not say what population an interval "over machines" generalises to.
- **Threshold counts are knife-edge.** At gpt-oss 25%, eight of the 34 exploratory launches have plain-form errors between 5.7% and 6.3%. Moving G by 0.05 ms moves the "within 6%" count by about six. I reproduce the paper's 18 only to within one. Report error distributions (median |error| with a machine-level interval) rather than counts against a band.

### W5. The scope of the fewest-admission result is set by the validity filter

"Gains on every machine that passed our checks" is literally true: on the 9 stable machines I get a mean of 1.222, bootstrap [1.136, 1.309] and t-interval [1.113, 1.331]. But the two unsteady machines where it lost, the EPYC 7302 (0.846) and the EPYC 7663 (0.974), are also the two with the slowest links relative to their CPUs. Their link-to-CPU ratios are 0.13 and 0.22 (B_p/B_c from my reading of the probe), while every stable machine is at 0.30 or above.

The exclusions were legitimate: wrong outputs, and a registered 2% round gate. Still, the abstract and Table 1 should scope the claim to link-to-CPU ratios of 0.3 and above. The paper already says, in Appendix K, that below some link rate the set "should not be copied in the step"; that belongs in the main claim.

### W6. Significance is moderate

- **Narrow scope.** One GPU model, batch 1, two models, and AIME prompts that were also used during development.
- **The online lesson is small.** The fewest-admission result is an oracle result; its online lesson (admit less) is worth about 2%, and its registered thresholds failed on two of three stable job 106 hosts.
- **Eq. (1) is not new in kind.** It is a roofline-style bytes-over-bandwidth bound with MIN's reads as the byte count.
- **Eq. (2) is not a bound on other systems.** It is presented as a bound "for systems that read on demand", but T_GPU is this engine's kernel time (2.9–3.4 ms by my recomputation from the Nsight categories). It bounds only this engine's kernels.
- **The system comparison is effectively two machines.** Table 3's intervals show that machine choice moves the FreeToken ratio more than the intervals' width, so the 4090/3090 grid in Appendix E deserves more weight.

The horizon rule (0.65 C distinct experts) is the most transferable finding. I reproduced it: median 0.651, quartiles 0.61–0.72. Excluding the five sub-token points, where D/C equals W50 by interpolation, gives a median of 0.63 [0.60, 0.72], so the rule is robust to that choice. It does, however, require each model's routing trace.

### W7. Smaller points

- **Mis-stated percentage.** Appendix K says 107d's second-highest reading is "more than 5% lower … by 35%". It is 26% lower; the highest reading is 35% above it.
- **Arithmetic means of rates.** Table 3 averages per-problem tok/s arithmetically (69.9 against a harmonic 69.7 at host B, gpt-oss 11%). This is negligible here, since ratios differ by about 0.001, but rule 3 of the paper's own checklist asks for harmonic means.
- **R⋆ is not matched to the problem set.** The bound uses R⋆ = 38.32 reads per token from the 30-problem session trace, while jobs 099–108 time only the first 20 problems. The paper should state that R⋆ is not recomputed per run.
- **External anchoring of the registration.** The registration evidence rests on the authors' commit timestamps plus a copy of the GitHub push log (`prereg/gpu_pushes.json`). GitHub's events API keeps events for about 90 days, so the pushes should be archived externally (Software Heritage, Zenodo or OpenTimestamps) to stay checkable.
- **Ambiguous count in the introduction.** "We registered it … before four later experiments" sits beside Section 4.2's "before three more experiments". Both are right, but the reader has to work out that the first test used the plain form.

## Clarity

I score clarity 3/5. The paper is now organised around the right devices, but it still asks the reader to carry too many numbers, terms and identifiers at once.

**What works**

- The introduction's three questions, each mapped to a section, give a usable spine.
- Table 1 tells the reader up front what kind of evidence backs each claim. Paragraph headings repeat the status ("registered", "exploratory", "after the fact"), which keeps registered and post-hoc material visibly apart through Sections 4–6.
- Fig. 1 makes the "admission read twice" mechanism concrete. Table 2's Reads column carries the paper's main idea in one glance, and Table 6 maps the appendix codes.
- The running example (Eq. (1) 10.0 ms; Eq. (2) 13.0; deployed 20.6; MIN in the step 15.2; MIN prefetched 13.8) is the clearest passage in the paper. I reproduced all five numbers from 096a's raw rows.
- Table 4 is compact and honest. Fig. 2 is information-dense but readable.

**What still makes it hard to read**

1. **Number density.** Section 4.2 has about 30 numbers in two paragraphs, many inside nested parentheses. For example: "At gpt-oss 11% the error leans the same way over all 25 consumer machines (mean −2.2%, 95% interval over machines −3.7 to −0.4; on the 21 it was found on, −1.5%, −3.2 to 0.4)". Put the per-test details in Table 4 and keep one takeaway sentence per test.
2. **The abstract makes the reader resolve a contradiction.** It says the time "is close to" G plus reads, then that "no registered test that included machines new to it passed", then qualifies the server misses with "machines that read well below their probed rate". State the bound, the 25-machine result and the prospective result in three plain sentences with numbers.
3. **Too many terms and near-synonyms.**
   - Units: machine, host, launch, rental and round; budget and cell.
   - Read paths: fetch, copy, load, admit, "in the step", "by the CPU", "background".
   - Undefined: "state", as in "the gain of every state that moves reads from the CPU to the link".
   - The model goes by several names: "the relation", "plain form" and "overlap term"; "the calibrated model" (Appendix D); and "the law" in appendix prose and tables ("law against lite", "the law predicted llama.cpp on the Ryzen").
   - "Limit" and "bound" are used interchangeably.
   - There are three different G's (G, T_GPU and G_fit). The paper warns about this, which helps, but one term per concept would help more.
4. **Internal identifiers leak into the main text.** Job numbers ("the median of the 7 profiles of jobs 069c and 105"), host codes in figure legends (O3/O4/O5 in Fig. 3, never introduced in the main text) and script paths (`scripts/linkaware.py`) appear in the body. The appendices add hosts A/B/S, O1–O6 and Pa–Pj.
5. **The baseline switches between sections.** Section 5 and Fig. 3 measure against the deployed cache; Section 6 and Fig. 5 measure against admit-every-miss. The text flags it once in parentheses; the figure axes and captions should carry it too.
6. **The appendices read like a lab notebook.** There are 14 appendices, 29 tables and 40 pages, plus a 41-page supplement. Appendix C's failures paragraph is a single block of about 350 words. For a workshop paper, keep what supports Table 1 and move the rest to the supplement.
7. **Figure details.**
   - Fig. 2's labels, such as "R9 9950X (6)", need the numbering explained.
   - Its caption explains the hatched under-prediction but not the bars that run past the dot (over-prediction, e.g. R9 9950X (6) at 11%).
   - Fig. 3's 95% intervals are invisible at print size, and the grey panel ticks crowd the markers.
   - Fig. 4 does not say whether the non-star markers include machines that failed the checks.
8. **Layout.**
   - The "Contributions." heading sits orphaned at the foot of page 1, with its list on page 2.
   - Table 1 is set in `\scriptsize` with wrapped evidence cells, so its most important column is the hardest to read.
   - The main text points to appendix Tables 22 and 26–28 for core claims.

## Questions for the authors

1. Why use the single highest probe reading as B_host? How do the 31–54% range, Table 4 and Fig. 2 change if you use the median of repeated probe runs? Is 107d's 181 GB/s CPU reading, taken concurrently with PCIe at 128 threads, physically plausible on that two-socket part, or a timing artefact?
2. For each Table 1 row, which registered clauses (job, number) does "registered" refer to, and how did each score? On host S specifically, was a lead over FreeToken registered at each cell, or only the ±0.06 band?
3. Does the problem-level bootstrap account for the cache state carried between problems? What happens to the Table 3 intervals with contiguous-block resampling?
4. Do you have a stable machine with a link-to-CPU ratio below 0.3 on which the fewest-admission schedule was run in the step? If not, will you scope the claim?
5. The machine-level intervals are over rented offers chosen by price and gates. What population do you intend them to describe?
6. Is R⋆ recomputed for the 20-problem runs, or is the 30-problem value used throughout?
7. Would you accept reframing Eq. (3) as "T = G + reads·S/B_eff" and reporting B_eff/B_host as the finding? That is what the data support.

## What would raise my score

1. **Reframe Section 4 and the abstract** around the measured read-efficiency factor and the prospective misses. Present the decomposition as an exploratory accounting, and scope it to desktop-class hosts as defined in job 108.
2. **Make Table 1 auditable:** clause numbers, scored outcomes and interval type. Relabel the host-S lead, the two "by implication" machines, and the "Machines rented again: held" heading.
3. **Treat the probe as a measurement.** Repeat it, report median and spread, put an interval on the bound, and move the existing sensitivity results (Table 26) into the main text.
4. **Apply the machine as the unit everywhere.** Use one value per machine in counts, report error distributions instead of threshold counts, and use block or launch-level resampling. State the sampling frame.
5. **Scope the fewest-admission claim** by link-to-CPU ratio, or add a stable slow-link machine.
6. **Fix the clarity problems listed above,** especially the abstract, Section 4.2's number density, terminology, and pruning the appendices.
7. **Archive the push timestamps externally.**

Items 1–3 together would move my overall score to 7–8. The data and the registration discipline already support a strong benchmarking paper; the framing and presentation do not yet show it.

## Scores

| Criterion | Score |
|---|---|
| Overall | **6 / 10** (acceptable with revisions) |
| Soundness | **4 / 5** |
| Methodology | **4 / 5** |
| Significance | **3 / 5** |
| Clarity | **3 / 5** |
| Confidence | **4 / 5** |

- **Soundness.** The numbers are right and the claims are mostly hedged correctly. The bound's dependence on one probe reading and the overstated evidence labels keep it from 5.
- **Methodology.** Registration, ledger, gates and machine-level analysis are exemplary for systems work. The probe statistic, the unit inconsistencies and the post-hoc promotion of a prediction to a gate keep it from 5.
- **Significance.** Narrow scope, an oracle-only main insight with a small online gain, and a roofline-style bound.
- **Clarity.** Better organised, still dense; see the Clarity section.
- **Confidence.** I recomputed most claims from raw data, but did not audit the supplement's 1,483 clauses or the literature audit (Appendix M).

---

## Claims checked against raw data

Data sources:

- **Rows:** `results/<job>/ec_*.jsonl`. T is the mean over problems of decode_ms/n_decode, averaged over rounds where a job has rounds.
- **Counters:** `st_*.json`, giving M = misses/steps and A = admits/steps.
- **Probe:** `concur.txt`. B_host is the maximum of every cpu, pcie and concurrent-sum reading.
- **GPU time:** `g_prof.json`, or the median of the 069c and 105 profiles where no profile exists or the profile caught no decode kernels.
- **Bound:** R⋆ = 38.315 and 15.340 reads per token at gpt-oss 11% and 25% (`prereg/speed_limit_v2.json`, "exact").

Ratios are of means paired by problem, with 4,000–20,000 bootstrap resamples.

| # | Claim (location) | Paper | My recomputation | Verdict |
|---|---|---|---|---|
| 1 | Job 105, plain form (Table 4) | 5/8 within 6%; new machines 2/4; median 4.2% | 5/8; 2/4; median 4.25% (errors +4.7/+9.3, +2.0/+7.0, −1.5/+3.8, +1.4/+6.9) | ✓ |
| 2 | Job 106, overlap form (Table 4, §4.2) | 12/20; new 0/8; median 4.7%; set aside: 12/12, 2.3%; servers miss by 10–52%; rounds 52% apart | 12/20; 0/8; 4.65%; 12/12 (worst cell +5.99%), 2.35%; servers −10.2 to −52.4%; 106d spread 52.2% | ✓ (one cell at 5.99%) |
| 3 | Job 107 (Table 4) | 2 valid of 7; 1/8 within; median 31.7%; inconclusive | Valid: 5900XT and ES (spread 0.3%); 107c and 107e failed V2 (3.5%, 6.0%); 3 gated; 1/8; 31.75% | ✓ |
| 4 | Job 108 (Table 4, §4.2) | 4/6 within 8%; median 6.7%; EPYC 7543 −34.3%/−27.7%; without it 4/4, 5.3%, two pooled clauses fail | −34.3, −27.7, −6.9, −4.0, −6.5, −0.9; median 6.7%; without the EPYC median 5.25%, two cells beyond 6% | ✓ |
| 5 | Abstract: "up to 11% on new consumer, up to 59% on servers" | 11%, 59% | 5900XT Qwen3 12.5%: −11.3%; ES gpt-oss 11%: −58.9%. With the second-highest probe reading, ES: −46 to −51% | ✓ (the 59% depends on one outlier reading; see W2) |
| 6 | New consumer machines under-predicted on every gpt-oss cell (§4.2) | 0.9–7.1% | 107b −7.1/−2.3, 108d −6.9/−4.0, 108f −6.5/−0.9 | ✓ |
| 7 | Server implied read rates (§4.3) | ES 0.27, EPYC 7543 0.58, EPYC 7402P 1.04 | 0.27, 0.58, 1.04 | ✓ |
| 8 | Server time below the probe's rate (§4.4) | 0%, 34%, 59% | 0% (+3.0% error), 34.3%, 58.9% | ✓ |
| 9 | Launch and machine counts (§4.3, Fig. 2) | 39 valid consumer launches on 25 machines; 5 of 8 server launches invalid | 47 launches with gpt-oss C14: 39 consumer on 25 GPU UUIDs, all valid; 8 server, 5 invalid (106c, 106d, 107c, 107e, 108b) | ✓ |
| 10 | Consumer implied rate and fit (§4.3) | 0.88–1.16; within 8% on 37/39; G profiled for 9 | 0.88 (099i) to 1.16 (100f); 37/39; 9 | ✓ |
| 11 | Shares of time at gpt-oss 11% (§4.4, abstract) | G 12–45%, MIN 31–54%, excess 22–35%; serialisation 23–66% of the gap; below-probe at most 11% | 12–45; 31–54; 22–35; 22.5–66; 10.6 | ✓ |
| 12 | Exploratory plain form (§4.1) | 28/30 within 6% at 11% | 28/30 (G = 4.28 or 4.30) | ✓ |
| 13 | Overlap against plain over 34 launches (§4.1) | 25%: 33 vs 18; 11%: 29 vs 32 | 25%: 33 vs 17; 11%: 29 vs 32. The 25% plain count swings 13–22 as G moves 4.45–4.60 | ≈ (off by one; knife-edge; see W4) |
| 14 | Mean error over machines (§4.2) | −2.2% [−3.7, −0.4]; found-on 21: −1.5 [−3.2, 0.4]; 25%: +1.6 [0.2, 3.1] | −2.16 [−3.7, −0.4]; −1.50 [−3.2, 0.4]; +1.57 [0.2, 3.1] | ✓ |
| 15 | Elasticity of T−G on B_host (§4.4) | −0.95 [−1.06, −0.84] | −0.965 [−1.08, −0.81] (21 machines, first launch) | ≈ |
| 16 | Fallback G and G range (§4.1, App. K) | median of 7 profiles 4.28/4.50 ms; G 4.1–4.7 | 4.284/4.498; 4.07–4.67 | ✓ |
| 17 | T_GPU of Eq. (2) (§3) | 2.9–3.4 ms | 2.94–3.38 (G minus expert GEMV and cache control; includes activation quantize) | ✓ |
| 18 | Variability (§2) | same CPU model differs by up to 29%; re-rentals within 3.2% | 285K pair 18.14/14.06 = 1.290; worst re-rental 9.12/9.41 (TR 9960X) = 3.2% | ✓ |
| 19 | MIN, 1 read, against deployed, 3 hosts × 6 budgets (§5) | +16–51%; prefetched up to +81%; usual ways at most +15% at the smallest budgets | +16.0 to +51.4%; +81.2%; at most +4.7% (gpt-oss 11%) and +14.8% (Qwen3 12.5%) | ✓ |
| 20 | Running example on O4 (§5) | 10.0 / 13.0 / 20.6 / 15.2 / 13.8 ms | 10.02 / 13.02 (T_GPU = 3.0) / 20.64 / 15.16 / 13.81 | ✓ |
| 21 | Fewest-admission schedule over machines (§5, App. K) | 9 stable machines, mean 1.22× [1.14, 1.31], t-interval 1.11–1.33; lost on 2 of 4 unsteady | 1.222 [1.136, 1.309], t [1.113, 1.331]; lost on 106c (0.846) and 107e (0.974), the two slowest links (ratios 0.13 and 0.22) | ✓ (scope; see W5) |
| 22 | Fewest-admission counters and slow links (§5) | copies 0.60–0.73 of greedy; misses +2.5–5.3%; greedy 0.85–0.93× on slow links; CPU-loaded 1.11–1.22× | 0.60 at C14; +2.5% at C14; 0.853–0.933; bypassplan 1.112–1.217 | ✓ |
| 23 | Registration of the fewest-admission claim (Table 1) | registered on 7, two by implication | Job 104 registered "beats deployed" for Pf only; for 104b and 104c only "beats greedy by at least 0.03" | Mislabelled (see W3) |
| 24 | Admit less, κ tuned (§5, App. K) | 5 machines, mean 1.020× [1.012, 1.031]; up to +6%; −1% at 1 of 10 cells; reads −4–6% | 1.020 [1.012, 1.030]; +5.7%; 0.987 (107d, 25%); reads −4.3 to −5.8% | ✓ |
| 25 | Job 106 registered "admit less at least 1.01" (App. K) | short on 2 of 3 stable hosts; pooled stable median 1.01 | 106a 1.003, 106e 1.008; stable median 1.015 | ✓ |
| 26 | Layer-ahead copy (§6) and 16-token window (§6) | 1.04× on a fast link, down to 0.72×; window 1.00–1.03×, half-right windows lose up to 6% | 1.042 and 0.722; 1.000–1.035; worst 0.943 | ✓ |
| 27 | Machine dependence of the gain (§5) | Spearman 0.89 [0.63, 0.97] over 19 machines; machines 96% of variance, problems at most 3% | 0.91 [0.69, 0.98]; 96%; 1.1% and 2.8% | ≈ / ✓ |
| 28 | Table 3, host B (and S) | 1.294 [1.278, 1.312], 1.275 [1.254, 1.295], 1.154 [1.134, 1.172], 1.032 [1.022, 1.042]; 5 of 6 host intervals disjoint | Identical (launch 2; launch 1 was the selection launch); 5/6 disjoint. Arithmetic mean of tok/s, not harmonic (69.88 vs 69.70) | ✓ |
| 29 | Read-schedule microbenchmark (§3) | waiting per layer reaches 86–96% of the bound's read time | 0.861 (i9), 0.963 (9950X), 0.950 (9800X3D) | ✓ |
| 30 | Horizon rule (§6) | 0.65 C (0.61–0.72); W50 from 0.6 to 34 tokens | 0.651 (0.610–0.721); 0.64–33.6; without sub-token points 0.63 (0.60–0.72) | ✓ |
| 31 | Registration timing (App. C) and probe robustness (App. K, Table 26) | 71 rentals; commit ≥2.7 s before; 7 pushes within 1 s, 4 at or after rental (≤0.8 s); job start ≥26 s after push. Engine >5% above the probe on 12 of 84, up to 1.22×; second-highest reading median 0.8% lower, 107d "by 35%" | 71; 2.7 s (105f); 7; 2 strictly after and 2 at the same second, max 0.8 s; 26 s (100f); no amendment changed a prediction; no result file edited after first commit. 12/84, 1.22; 0.8%; 107d is 26% lower (the highest is 35% above the second) | ✓ timing / wording error |

**Tally:** 27 exact, 3 within tolerance (#13, #15, #27), 1 mislabelled evidence claim (#23), and 1 wording error inside #31. The mislabels and wording issues are discussed under W2, W3 and W7.
