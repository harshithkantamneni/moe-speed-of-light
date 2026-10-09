# Review 27b: "Where the Seconds Go" (professor, scientific-benchmarking referee)

## Materials read

- `paper/paper.pdf`, all 45 pages. I read the main text (pp. 1–10) closely. I read Appendices A–E, H, I, J, L, M and N in full, and skimmed F, G and K.
- `paper/main_body.tex` (source of Table 1). `paper/tab_job109.tex` and `paper/tab_audit.tex`, read only to parse table cells the PDF text extraction scrambled.
- `paper/supplement.pdf` (the scorecard): front matter, structure and totals.
- `paper/ieee-paper.pdf`, all 10 pages rendered. `paper/ieee-supplement.pdf`: front matter, numbering and reference list. `ieee-*.log` and `ieee-*.bbl`, checked for box warnings and reference entries.
- Artifact files:
  - `gpu/vast_ledger.json`;
  - `prereg/speed_limit_v2.json`, `prereg/scorecard_clauses.json`, and `prereg/scorecard_{099,102,109,111,112,113,114}.json`;
  - `data/published_measurements.json` (structure only).
- Raw results in `/home/claude/gpu-branch/results/` for jobs:
  - 081, 084c (routing trace), 089, 096a/b, 099a–j, 100b/f, 101b, 102a–c;
  - 104a–c, 105a/b/e/f, 106a–e, 107a–g, 108a–f, 109a/c/d/e/f;
  - 110a–e (gates), 111b–g, 112a/b/d/g, 113b/d, 114a–g.
  - I used `ec_*.jsonl`, `bs1.jsonl`, `st_*.json`, `concur.txt`/`concur2.txt`, `g_prof.json`/`prof_*.json`, `fetch_table_law_gptoss.json`, `v0.txt`, `validity.txt`, `probe1_when.txt` and `readsched_C*.txt`, plus the offer listings for jobs 113 and 114.
- Job script headers in `/home/claude/gpu-branch/jobs/` for jobs 100, 102, 105, 106, 107, 109, 110, 111 and 114. `git log --format='%h %ad'` and file diffs (`git show --format=`) for the job scripts and offer listings.

## Independence statement

- I did not open anything under `reports/` except to write this file.
- I did not open `prereg/*outcome*.md`, `research_notes/`, `apply/`, `paper/paper_v1_prereview.tex`, or any file named as a review, number check, plan or progress log.
- I did not read commit messages. Every `git log`/`git show` call used `--format='%h %ad'` or `--format=` (empty), so only hashes, dates and file contents were shown. Some job-script headers mention earlier reviews in passing (e.g. job 114's first line); I read those as file content.
- I did not use the authors' analysis scripts. Every recomputation below comes from my own code, in a new folder (`scratchpad/rev27b/`: `load.py`, `t3*.py`, `t4*.py`, `t5.py`, `t6.py`, `panel.py`, `inter.py`, `few1r.py`, `minsim.py`, `regtime.py`, `attr.py`, `bhost.py`).
- I modified nothing in either repository and committed nothing.

---

## Summary

The paper studies batch-1 decode of MoE models whose experts mostly live in host DRAM: gpt-oss-120b and Qwen3-30B-A3B on rented RTX 5090 hosts, plus RTX 4090 hosts for one test. It makes four contributions.

1. **A bound.** Belady's MIN-with-bypass gives the fewest host reads any C-slot-per-layer cache can make on the exact routing trace (R⋆). Divided by the host's best probed read rate, this gives a time floor, Eq. (1). Two refinements follow: a demand bound (reads wait for the router, Eq. 2) and an ordered bound (only link copies can be issued early, Eq. 3).
2. **An engine with oracles.** A 3,500-line llama.cpp patch adds a decayed-frequency expert cache, a CPU/PCIe split for misses set by a per-machine "fetch table", and oracle policies that read a recorded routing trace. The oracles remove one part of the gap at a time:
   - *what is cached*: MIN's set;
   - *how it is read*: once in the step, rather than CPU-served and then copied in the background;
   - *when it is read*: ahead of use.
3. **Findings.**
   - The cache reaches 31–54% of the bound on consumer hosts at gpt-oss 11% (post hoc).
   - On the deployed read path, the fetch table's in-step copies displace MIN's set, so the set pays only together with single reads.
   - With in-step fetches off (3 hosts), MIN's set closes about 20% of the gap alone and 44% together with single reads.
   - Online variants recover at most 6% of the read-ahead oracle's gain.
   - The link-to-CPU bandwidth ratio predicts oracle speed on a second card.
   - In trace replay, half of foresight's value needs about 0.65·C distinct experts of lookahead (post hoc).
4. **A registration practice.** Each GPU job's predictions are written in its script header and committed before the machine is rented. Validity gates are registered, and every clause is scored, failures included. The paper reports 154 rentals at $112.7.

## Strengths

1. **The numbers reproduce from raw data.** I checked 37 quantitative and procedural claims with my own code (table below). All the headline tables reproduce from the per-problem rows and counters, to the printed digit:
   - Table 3 (36 speeds and 10 bootstrap intervals);
   - Table 4 (Panel, New and CPU-only columns at both budgets, attribution rows within one point);
   - Tables 5, 6, 12, 13 and 31;
   - the Few-1R pooled estimate (1.257×, [1.186, 1.332] over 15 machines);
   - the round-to-round noise statistic (129 comparisons on 17 machines, median 0.21%, maximum 1.77%);
   - the clause tallies.

   My own MIN-with-bypass replay of the routing trace matches the artifact's count exactly (39.03 reads/token at C=14, keep-for-step semantics). That level of reproducibility is rare in systems papers and is the paper's strongest asset.
2. **The registration is real and checkable.**
   - All 101 rentals of jobs 093–114 started after the last commit of their job script, by at least 2.71 s (job 105f).
   - Every post-launch header edit I diffed (jobs 102, 105, 106, 107, 111) substitutes a host and marks the predictions as unchanged.
   - The one prediction added after launch (job 100's ninth) is disclosed.
   - The RTX 4090 trend was frozen as explicit coefficients in job 110's header before any RTX 4090 ran. I recomputed every bracketed prediction in Table 6 from those coefficients.
   - Gate failures are logged per host (`v0.txt`, `validity.txt`) and the paper's counts match them: job 107 (7 rented: 3 gated, 2 failed the round check, 2 valid), job 108 (6: 2, 1, 3), job 110's void, and job 114's 3 gated plus 1 unstable.
3. **The statistical design follows Hoefler and Belli.**
   - The machine is declared the unit. Claims are per-machine orderings or t-intervals over machines.
   - Ratios are ratios of mean times, and rounds are combined by geometric mean.
   - Round-to-round and rental-to-rental variation are measured and used to re-score the clauses (Appendix D).
   - The paper flags its own departure from the harmonic-mean rule in Table 3 and shows it matters by at most 0.003; I confirmed this.
4. **Oracles inside a real engine, and a control that changed the story.** Measuring foresight in the time domain inside a production-derived engine, not as hit rates in a simulator, is the novel methodological contribution. Jobs 113 and 114 test the authors' own earlier reading: the counters showed that the fetch table displaced MIN's set, and the paper was reorganized around the two read paths. That is how a measurement study should behave.
5. **Failures are reported, not buried.**
   - 142 of 565 hand-scored clauses failed.
   - The closed-form account, Eq. (4), failed all its registered tests and was moved to an appendix.
   - The timing control (job 112) is called "uninformative".
   - The post-hoc application of the trend to new RTX 5090s misses by up to 18%; I reproduced 17.6% on 5 of 18 cells.
6. **Actionable output.** Section 9's rules are scoped ("rules 2–4 from one model at two budgets") and each traces to a measurement.

## Weaknesses (most important first)

### W1. The central decomposition is largely about this engine's own read path; the read-path-independent test rests on three machines that are not independent.

The abstract's main mechanistic claim is that "with [the fetch table] the set pays only if each admitted expert is also read once; without it, on three machines, the set alone closes 20% of the gap and both changes together 44%". The first half describes an artifact of the authors' engine: a table that copies misses MIN bypasses into slots MIN keeps. The second half is the generalizable result, and it comes from job 114's three valid hosts.

- Per machine, the shares are 21.8 / 15.1 / 21.9% (set alone) and 55.5 / 22.4 / 55.5% (both). The "44%" is the mean of two near-identical values and one very different one, with a t-interval of [−3, 92].
- Two of the three hosts, 114b and 114f, are Ryzen 9 5950X machines from the same Vast provider: host 716820, machine IDs 153462 and 153463, identical offer specs. Their probes are nearly identical (B_c 33.3 vs 33.5 GB/s; B_p 26.2 vs 26.2), and so are their Table 5 rows.
- 114f is the same GPU (UUID f90ec646) as 109f and 112g. It had already been measured in the closely related deployed-path 2×2: interaction 7.5 ms, MIN-1R 1.37×.
- A third box from the same provider is job 113's 113b.

So 3 of the 5 CPU-only launches in jobs 113 and 114 come from one provider's identical fleet, and one was a re-rental with known results. The Limitations section says only that "two of its three are Ryzen 9 5950Xs".

The per-machine sign claims (the interaction is above zero on every machine) are well supported. The *means* quoted in the abstract and in rule 2 ("15–22% … and 22–56%") should not be read as population estimates.

### W2. The bound and the baseline both hang on a single, sometimes contended probe, and that uncertainty is not in the headline ranges.

- B_host is the single highest reading of one probe run. Where the job logs it (jobs 112–114), the probe ran while the model was still downloading on 7 of 10 launches.
- By the paper's own accounting, the engine reads more than 5% faster than the probe on 12 of 84 launch-budgets, up to 1.22×.
- Between two rentals of one machine, the link-to-CPU ratio moved by up to 12%.
- The same probe also fixes the deployed cache's fetch table, and that mapping is knife-edge. 114b and 114f differ by 0.8 GB/s in the concurrent reading (32.3 vs 33.1). Their tables came out all-zero on 114b and 0,0,1,2,3 on 114f. This all-zero table made two registered clauses of job 114 fail (H1c, H2), and it is why 114b carries the ‡ mark in Table 5.

The result is that both the denominator (Eq. 1) and the baseline (the deployed cache) carry systematic error that no interval in the paper reflects. "31–54% of the bound" and the gap shares in Table 4 would be more honest with a probe-uncertainty band: second-highest reading, idle second probe, ±12%. Appendix L gives a ±10–22% sensitivity for Table 4. That should be carried into the main text, and the 31–54% should be presented as a band, not a range of point values.

### W3. Sampling and selection: convenience sample, clustering and re-rentals, plus a replacement-rule deviation whose justification is inaccurate.

The machines are "a convenience sample of cheap offers", screened by gates the authors designed, with repeated machines (Pf, Pg, Pe, O4, 109f/112g/114f) and provider clusters (above). The t-intervals over 3–5 machines then estimate the mean of an undefined population.

Job 114's replacement deviation matters here. At the first replacement (`results/114_offers_at_replacement1.txt`, 21:50 UTC):

- registered host e (51782800) was unlisted;
- a cheaper qualifying Core Ultra 9 285K (offer 54905524, $0.669/h) was listed;
- the authors rented 54573923 ($0.802/h), which is 109f's machine.

The paper discloses the skip but says "the choices were made before any of these machines produced data, so they could not select on outcomes". That is not accurate for 114f, whose GPU had produced deployed-path 2×2 data in jobs 109 and 112. We cannot know what the skipped host would have shown. The only Intel hosts on the CPU-only path, 113d (285K) and 114c (265K), showed the smallest single-read benefits (Few-1R over the fetch-free Few-2R: +0.02; interaction 0.36 ms; MIN-1R share 22%). So the substitution plausibly raised the CPU-only means the abstract quotes.

Separately, the tables print per-problem interval half-widths of ±.00–±.01. Rental-to-rental variation is a median 0.5% and up to 5.3%. The widening analysis in Appendix D is good, but the main tables should print the widened intervals.

### W4. Table 1's evidence labels are applied inconsistently, and excluded hosts are not reported consistently.

- **"Loose" is missing where the paper's own rule calls for it.** Job 114's H3 thresholds (MIN-2R ≥ 1.05× at 11% and ≥ 1.15× at 25%) sit below the job 113 values listed as "earlier data" in the same header: 1.09–1.12 and 1.28–1.36. By the paper's definition ("a threshold the earlier data made easy to meet") this row should be marked loose, as the job 102, 109 and 110 rows are.
- **"Held (point)" understates job 109.** Job 109's interaction clauses are labelled held (point) because the scoring script emits no interval for them (`scorecard_109.json`: `ci: None`). My per-problem bootstrap gives intervals far above zero on every machine and round; the lowest bound is 1.43 ms. The label reflects a scoring gap, not weak evidence.
- **Excluded hosts.** For Few-1R, the paper reports what happened on hosts that failed a validity check: "lost on two". For job 114 it does not. The excluded 9950X3D2 (114e, rounds 6.1% apart) had interactions of −2.09 ms and +1.64 ms in its two rounds (mean −0.22 ms). Excluding it follows the registered rule, but "every host that starts is reported" should include its numbers.
- **Table 1 row 3.** The registered prediction on host S failed (five of six ratios outside ±0.06 of host B's). The row is headed with a weaker, unregistered claim ("faster … at most configurations"). The Result column is honest, but the Claim column should state what was registered.

### W5. Registered and post-hoc results are not fully separated in the running text.

The abstract ends: "Registered predictions were committed before their machines were rented; the consumer range and the replay result are post hoc". That implies everything else is registered. It is not:

- The introduction's "39% of the gap on fast-link machines" uses a fast-link subset defined after the panel's data. The panel's own registered interaction clause failed (9 of 10).
- Table 4's Panel column and the attribution rows ("of what is left") are not registered results, yet they sit beside registered columns with no per-column status.
- The forecaster and speculative-decoding results in Section 8 are post hoc or exploratory.

Table 1 is a good device, but it covers selected claims only. Every quantitative statement in the main text should carry its status in place.

### W6. Narrow scope relative to the framing.

- One engine, the authors' own.
- Batch 1, AIME-25 prompts used during development, 20 problems × 256 teacher-forced tokens.
- The decomposition and rules 2–4 come from gpt-oss-120b at two budgets.
- One GPU model, except for two RTX 4090 tests.
- Output parity is shown by KL divergence only.

The title and abstract speak of MoE decode from host memory in general. The limitations are listed, but the claims should be scoped in the abstract.

### W7. The reachability evidence does not exercise the engine's copy path.

The microbenchmark (Table 31: 86–96% per layer, reproduced exactly) copies from pinned memory with the copy engine and runs no GPU compute. Appendix E says the engine's paced path copies "from pageable host memory on a single copier thread". So "the bound's read time is nearly reachable" is shown for an idealized reader. It also passed a registered threshold of ≥ 50%, while the header's own pre-launch estimate was 85–99%.

### W8. Weaker points.

- **The published-systems comparison** (13.6% vs 27%) scores other papers' hardware at datasheet rates, sometimes on sibling-model traces, in a post-hoc subset. It is acceptable as an appendix, but too weak for Table 1 and the introduction.
- **Hoefler–Belli rule 1 is only partly met.** Tables 5, 6, 12 and 13 report only ratios. Absolute ms/token of the deployed cache and of Eq. (1) per machine should sit beside them.
- **T_GPU = 2.9 ms** is "the smallest of the 14 Nsight profiles", but the kernel categories and the 14 profiles are not specified. I get 2.94 ms as the minimum if activation quantization is counted and 2.60 ms if it is not.

---

## Clarity

**Assessment: 3/5.** The paper is readable by a determined specialist. A first-time MLSys reader still cannot reconstruct the main result from the abstract and introduction. The revision has improved the structure. The remaining problems are vocabulary, density and shifting yardsticks.

### What works

- **Section headings that state answers** ("How fast could it be?", "Where the seconds go", "Without foresight, the engine recovers little"), with lead sentences that state each paragraph's finding ("The in-step fetches displace the set the policy chose.").
- **Table 2** gathers configurations, machine sets and yardsticks in one place, and its read twice / read once / read ahead grid is the right mental model.
- **Fig. 1** explains the "second read" in one picture.
- **Table 1** as a claims ledger with registered thresholds beside results. This is an excellent idea; see below for execution.
- **The running example** (Ryzen 9 9950X, gpt-oss 11%: 10.0, 13.0, 20.6, 15.2 and 13.8 ms) makes the yardsticks concrete.
- **Limitations and the rules for builders** are short, scoped and specific.

### What still makes it hard to read

1. **The abstract and introduction depend on terms defined later.** "Fetch table", "in-step fetches", "MIN's set", "read once", "cache-fulls of distinct experts", "fast-link" and "greedy schedule" are defined in Sections 2 and 4. The abstract's central sentence cannot be parsed without them: "The engine's fetch table, which copies misses into cache slots the policy did not choose, hides the value of caching MIN's set: with it, the set pays only if each admitted expert is also read once…". The introduction's five bold "answer" paragraphs then state every number before any definition or evidence, so the reader meets each number twice.
2. **"Fetch" means five different things**, and the key finding depends on telling two of them apart:
   - the FETCH operation, i.e. reading once in the step;
   - the fetch table's in-step copies of unchosen misses;
   - the code `fetch`, which is MIN-1R;
   - `fetch=0`, the CPU-only path;
   - "fetched" in Dep-1R.

   The effect of the table's copies is what displaces MIN's set. Renaming them (e.g. "table copies" or "forced copies") would remove most of the confusion in Section 4.
3. **There are too many labels.**
   - Table 2 alone has ~15 configuration names, 6 machine sets and 5 yardsticks.
   - The appendices add ~25 codes (Table 7), host letters (A, B, S, O1–O6, Pa–Pj) and job numbers used as identifiers, even in main-text captions ("the 15 machines of jobs 096–101").
   - "The panel" (15 machines) and "the panel's main job" (10) are different sets with nearly the same name.
   - "New machines" (Section 4) and "machines never rented before" (jobs 107/108) also overlap in name.
4. **Tables 1 and 4 are too dense to read unaided.**
   - Table 1's Result cells nest parentheses ("5 of 5 new ones (held (point); fast-link scope set in between)") and mix evidence labels into prose. The Mach. column reads "10 + 5".
   - In Table 4, each budget has three columns with different machine sets and different read paths. Footnote ∗ changes what a row means on two of the three columns.
   - The CPU-only column has a hidden baseline mix: numerators are relative to the CPU-only deployed cache, but the denominator is the gap of the deployed cache *with* its table. I had to read job 114's header to reproduce it.
   - A 2×2 diagram per read path would do better than this table.
5. **The yardstick changes between sections.** Sections 5–7 use speed relative to deployed; Section 4 uses share of the gap; Section 6 uses capture; Section 8 uses share of MIN's gain over admit-every-miss. Fig. 5's caption has to warn "Baseline here: admit every miss, not the deployed cache", which shows the problem. Bounds multiply in the same way: Eqs. (1)–(3), Eq. (4) in an appendix, a datasheet-GPU and a measured-GPU column in Table 3, and six tightenings in Table 30.
6. **Load-bearing evidence sits 20–30 pages away.** Section 3's reachability claim relies on Table 31, tightness on Table 30, and the probe caveat on Table 29.
7. **Appendix D reads as a lab notebook.** It is about ten pages of job-by-job narrative, plus a 58-page scorecard. It is valuable as an artifact, but a reader cannot find the test behind a claim without Table 9. A one-page "registered tests at a glance" table would serve better: claim, job, date, n machines, threshold, outcome.
8. **Figures are crowded.**
   - Fig. 2 encodes path, link class, machine and set through colour, line weight and four marker shapes. Its second x-position means different things on the two paths.
   - Fig. 4 overlays four configurations, two card types and two fit lines.
   - Fig. 5's tick labels are tiny at column width.

### Concrete fixes

- Put a five-line vocabulary box before the claims.
- Remove numbers from the introduction except one per question.
- Rename the table copies.
- Use one yardstick (share of gap) throughout, with speed in parentheses.
- Split Table 4 by read path.
- Tag every main-text number as [registered], [post hoc] or [exploratory].
- Move the running example to Section 3.

---

## Questions to the authors

1. **Table 4 at 25% with the measured GPU rate.** At the measured batch-1 GPU rate (52% of datasheet), Eq. (1) becomes GPU-bound at gpt-oss 25% on fast-memory hosts. For 109a (B_host 83.3 GB/s), using `B_gpu_effective` from `speed_limit_v2.json`, I get 3.59 ms against a host term of 2.44 ms. How do the 25% shares, and especially "left after all three" (48%), change under that bound?
2. **Job 114's replacements.** Why was 54573923 (109f's machine) chosen over the cheaper listed Core Ultra 9 285K (54905524) at the first replacement? Did the authors know at the time that it was the 109f/112g machine?
3. **Job 114e.** Will you report its values (rounds −2.09 and +1.64 ms), and run the CPU-only 2×2 on further hosts from other providers and CPU families?
4. **Fetch-table stability.** Is the fetch table derived with any hysteresis? 114b and 114f differ by 0.8 GB/s in one probe reading and get opposite tables. Would a second, idle probe or a stability margin make "the deployed cache" a fixed baseline across machines?
5. **R⋆ semantics.** My replay keeping served experts for the step matches your `pol_084` exactly (39.026). My "free to evict after serving" replay gives 38.63 against your `exact` of 38.32. What within-step request order or semantics does `exact` use?
6. **T_GPU.** Which 14 profiles and which kernel categories define T_GPU = 2.9 ms?
7. **The microbenchmark's copy path.** Does it still reach 86–96% with pageable host memory and a single copier thread, as the engine's paced path uses?
8. **Registration provenance.** Can the GitHub push times behind `scripts/reg_timing.py` be archived by a third party (e.g. GH Archive or Software Heritage)? Can each machine record the commit it cloned? Commit dates are set by the author's own clock.
9. **Job 109 intervals.** Why is job 109's interaction scored without an interval when per-problem paired intervals are straightforward and exclude zero?

## What would raise my score

- **More hosts for the read-path-independent claim.** Replicate the CPU-only 2×2 (job 114's protocol) on at least three further hosts from different providers and CPU families, with a 285K-class host included. Follow the registered replacement rule mechanically and report every started host's numbers. This would move significance and soundness.
- **Mechanical evidence labels.** Apply the labels by a scripted rule, including "loose" whenever the header's own listed earlier data already meet the threshold. Compute intervals for every clause that has per-problem data. Report excluded hosts' values beside the registered outcome.
- **Probe uncertainty in the headline numbers.** Carry probe uncertainty into "31–54%" and Table 4, using the idle second probe or second-highest reading, and stabilize the fetch table.
- **Absolute numbers beside ratios.** Add ms/token of the deployed cache and Eq. (1) per machine to Tables 5, 6, 12 and 13.
- **The clarity fixes above,** especially the vocabulary box, renaming the table copies, one yardstick, and in-place status tags. With those I would expect clarity to reach 4/5.

## Scores

| | Score |
|---|---|
| Overall (1–10; 6 = acceptable with revisions, 8 = strong) | **7** |
| Soundness (1–5) | **4** |
| Methodology (1–5) | **4** |
| Significance (1–5) | **3** |
| Clarity (1–5) | **3** |
| Confidence (1–5) | **4** |

**Rationale.** Every number I checked reproduces from raw data, and the registration practice is verifiable and better than most published systems work, so methodology and soundness are high. They are not 5 because:

- the generalizable mechanism rests on three non-independent hosts;
- the bound's denominator and the baseline carry unquantified probe error;
- the evidence labels are applied inconsistently;
- the post-hoc/registered boundary is not marked in the running text.

Significance is moderate: the setting is batch-1 decode on one engine, with one model carrying most findings. Clarity improved in structure but not in vocabulary or density.

---

## Claims checked against raw data

✓ means reproduced to the printed digits; ≈ means within rounding or about 1%.

| # | Claim (location) | Paper | My recomputation (source) | Verdict |
|---|---|---|---|---|
| 1 | Registration before rental (App. D) | 101 rentals of jobs 093–114; every commit ≥ 2.7 s before rental | 101 rentals; minimum margin 2.71 s (105f); latest job-script commit vs ledger `start` | ✓ |
| 2 | Post-launch header edits (App. D) | 102, 105, 106 host substitutions, predictions unchanged; job 100 gained a 9th prediction after 4 machines started; 107/111 amendments before each new host | Diffs confirm. 102b/c and 106a–d were already running when the edits landed (+9 to +14 s) | ✓ (disclosed) |
| 3 | Job 114 gates (App. D) | 7 started; 3 stopped at V0; one 9950X3D2 failed the round check at 6.1% | 114a/d/g V0 "GPU rented before"; 114e V2 spread 6.10% | ✓ |
| 4 | Table 5, all cells | e.g. 5950X‡ 1.00 / 1.12 / 1.12 / 1.06 / 1.07 / 1.39 / 3.2 ms | Identical for all three hosts (114b, 114c, 114f) | ✓ |
| 5 | Counters (Sec. 4) | MIN-2R misses 48.6–50.8 vs 43.5; Dep-1R fetches 23.4–32.0 vs 8.2 admissions; 20.4 vs 8.2 admits | Same (`st_*.json`) | ✓ |
| 6 | CPU-only interaction (Sec. 4, Table 1) | 0.4–3.2 ms at 11%, CI above 0 on all 3; −0.1 to 0.1 at 25% | 0.36 [0.26, 0.47], 3.17 [3.01, 3.35], 3.10 [2.93, 3.28]; 25%: 0.13, −0.09, 0.04 | ✓ |
| 7 | Interaction share on both paths (Sec. 4) | 31–52% → 3–21% where the table is nonzero; 21% either way | 31→3, 52→21; 22/21 on 114b | ✓ |
| 8 | Table 4 Panel column (10 cells) | e.g. 33 [21, 44], 52 [43, 62] | Identical (15 machines, R⋆ from the artifact, B_host = max probe reading) | ✓ |
| 9 | Table 4 New column (10 cells) | 3 [−1, 8] … 46 [38, 55]; 25%: 20 [18, 21] … | Identical (25% MIN-2R: 19.8 [18.5, 21.1]) | ✓ |
| 10 | Table 4 CPU-only column (12 cells) | 20 [10, 29], 10 [−2, 22], 44 [−3, 92], 14 [5, 22], 39 [13, 65], 3 [−10, 16]; 25% likewise | Identical, but only with the gap of the deployed cache *with* its table as denominator (job 114 header) | ✓ (baseline mix, see Clarity 4) |
| 11 | Table 4 attribution rows (panel) | T_GPU 27, beyond MIN 21, residual 4 (11%); slow-link residual 35–47% | 27 [24, 30], 21 [20, 23], 4 [−6, 14]; slow-link 35.9 / 47.9% | ≈ |
| 12 | MIN-1R on fast-link panel machines (Sec. 1) | 39% (13 machines) | 39 [31, 47] | ✓ (post-hoc subset) |
| 13 | Deployed-path interaction (Table 1) | Positive on 9/10 panel main-job machines; 5/5 new, "held (point)" | 9/10 (099f: −0.48 [−1.06, −0.14]); 5/5 new, every per-round CI ≥ 1.43 ms; `scorecard_109.json` has no CI | ✓ (label understates) |
| 14 | Table 3 speeds (36 values) | e.g. 34.9 / 54.0 / 69.9 | Identical (`bs1.jsonl`, jobs 081 and 089, launch 2) | ✓ |
| 15 | Table 3 ours ÷ FreeToken ratios + CIs | 1.29 [1.28, 1.31] … 0.97 [0.96, 0.99] | Identical (10,000 paired resamples) | ✓ |
| 16 | Rule 3 (App. B) | Ratios of total time within 0.003 of ratios of mean rates | Max difference 0.003 | ✓ |
| 17 | Host B/S rates; bound share (Table 3) | Highest 88 / 71 GB/s; bound 172 tok/s; ours 41% | 87.5 / 71.3; 172.3 tok/s; 40.6% | ✓ |
| 18 | R⋆ from the trace (Sec. 3, App. N) | MIN with bypass, per layer | My replay of `route_aime25_gptoss.npz`: keep-for-step 39.026 = artifact `pol_084`; free-evict 38.63 vs artifact `exact` 38.32 | ≈ (0.8%; Q5) |
| 19 | Table 6, RTX 5090 rows | All cells | Identical (job 109; capture computed in speed) | ✓ |
| 20 | Table 6, RTX 4090 rows | All cells | Identical (jobs 111d/f/g, 112a/b) | ✓ |
| 21 | Trend predictions [brackets] (Table 6, Sec. 7) | 20 bracketed values; within 7% / 5%; no-change misses up to 46% / 69% | All 20 recomputed from the coefficients in job 110's header: identical | ✓ |
| 22 | Trend on new RTX 5090s (Sec. 7) | Misses 5 of 18 cells, up to 18% | 5/18 outside ±0.10 in log, max 17.6%; systematic sign by ratio | ✓ |
| 23 | Variants without foresight (Sec. 6) | Best 1.007–1.024×; capture ≤ 6%; reads 1.57–1.76 R⋆; Margin cuts reads 4–6% | 1.007–1.024; ≤ 5.7%; 1.575–1.755 R⋆; 4.3% | ✓ |
| 24 | Few-1R (Sec. 5, App. H) | 15 stable machines, 1.26× [1.19, 1.33]; losses at ratios 0.14 and 0.21 | 1.257 [1.186, 1.332]; 106c (0.845) and 107e (0.974) lost | ✓ |
| 25 | Table 12 (job 112) | 1R / 2R / early speeds and misses; misses 0.97–0.98 (11%), 0.95 (25%); admits 11.7–11.8 vs 6.2–7.4 | Identical | ✓ |
| 26 | Table 13 (job 113) | 0.98 / 1.12 / 1.26 / 1.23 / 1.40; misses 0.77–0.79 and 0.72–0.74 | Identical | ✓ |
| 27 | Table 31 (job 102) | 86–96% per layer, 93–97% per token; terms 7.04 / 2.84 / 9.70 / 3.91 / 9.56 / 3.85 ms | Identical | ✓ |
| 28 | Nsight G (App. H) | 4.1–4.7 ms for gpt-oss; EPYC traces in job 108 nearly empty | 4.05–4.65 ms; 108a/b G ≈ 0 | ✓ |
| 29 | T_GPU (Sec. 3) | 2.9 ms, the smallest of 14 profiles | Minimum non-expert time over the 069c/105 profiles: 2.94 ms (with quantize), 2.60 ms (without) | Partly (definition unstated) |
| 30 | New machines' share of bound (Sec. 3) | 41–48% | 41.5–48.0% | ✓ |
| 31 | Eq. (3) vs Eq. (1) (Sec. 3) | Median 1.00; up to 1.56 on slow links | Median 1.00 on the 20 hosts I loaded; max 1.18 in that subset | Partly |
| 32 | Round-to-round noise (Sec. 2, App. D) | 129 comparisons, 17 machines, median 0.2%, max 1.8% | 129, 17, 0.21%, 1.77% | ✓ |
| 33 | Second probe (Sec. 11) | Within 4.9% on 4 machines | +0.8, +4.9, −0.4, −0.3% | ✓ |
| 34 | Same-CPU spread (Sec. 2) | Up to 29% | 285K: 18.14 vs 14.06 ms → 29.0% | ✓ |
| 35 | Clause tallies (App. D, Table 10) | 565: 218 / 193 / 142 / 8 / 4; job 099: 273 (190/42/39/2); 102: 33 (0/28/5); 109: 90 (47/43/0); 111–114 as Table 10 | Identical (+1 validity row per job from 107 on) | ✓ |
| 36 | Audit medians (Sec. 3, App. M) | 13.6% in-class (20 rows), 9.5% trace (29), 14.5% all (52) | Identical from Table 33's rows; sources not re-derived | ✓ (table-level) |
| 37 | Cost and ledger (App. N); gates of jobs 107/108/110 (App. D) | 154 rentals, 90 offers, $112.7; 107: 7/3/2/2; 108: 6/2/1/3; job 110 void | Identical; job 110 V1 loss 0.1963 vs 0.190; 110a ratio 0.2499 | ✓ |

### Additional findings from the raw data, not stated in the paper

- **Sibling machines.** 114b, 114f, 114g and 113b are four identical Ryzen 9 5950X machines of one provider (host 716820). 114f is 109f's and 112g's GPU.
- **Job 114 replacement.** At the first replacement a cheaper listed Core Ultra 9 285K (54905524) was skipped for 109f's machine.
- **Excluded host.** 114e's two rounds gave interactions of −2.09 and +1.64 ms; it is excluded by its registered gate and its values are unreported.
- **Knife-edge fetch table.** Nearly identical probes on 114b and 114f gave all-zero vs 0,0,1,2,3 tables.
- **Probe timing.** In jobs 112–114 the first probe ran during the model download on 7 of 10 launches.

---

## IEEE format (`ieee-paper.pdf`, `ieee-supplement.pdf`)

The IEEEtran conference build is mostly conventional: Roman-numeral sections and tables, "Fig. N." captions below figures, table captions above in small caps, and numbered references in order of citation. Points that break or strain IEEE conventions:

1. **Front matter.** Title, anonymous author block, bold abstract and Index Terms are all present. The Index Terms are alphabetized. The abstract is about 241 words, at the top of IEEE's usual 150–250 range. It uses unexplained jargon ("fetch table", "cache-fulls"), which IEEE abstracts should avoid.
2. **Captions.**
   - Table I's caption runs three lines of small caps carrying six definitions (derived, post hoc, loose, held, held (point), Mach.). IEEE expects a short title, with definitions in a table footnote.
   - In small-caps captions, the configuration names lose the case that distinguishes them ("Margin" becomes MARGIN, "LA-1R-margin" becomes LA-1R-MARGIN; Table VI).
   - Tables III–VI also have long notes. That is acceptable as footnotes, but long.
3. **Cross-document references.**
   - The main paper's claims table (Table I) depends on supplement objects: "Eq. (S1)" in its first row and "Table S3" in its caption. The main text cites Tables S23–S25 for load-bearing claims.
   - The first-page footnote explains the S-numbering, which is correct practice.
   - The supplement calls the scorecard "supplement.pdf", which collides with the supplementary material itself. Rename the scorecard (e.g. `scorecard.pdf`).
4. **References.**
   - Main [1]–[39] and supplement [S1]–[S52] are separate numbered lists, so shared works get two numbers (e.g. Roofline [36] and [S46]).
   - Two supplement entries are incomplete: [S49] "Zhang, Gao, and Mitra" lacks initials, and ATSInfer lists no authors. The same two entries are incomplete in the MLSys version.
5. **Float placement and page layout.**
   - Double-column floats sit at page tops, which is fine.
   - Page 6's left column has a large stretched vertical gap before "Admit as rarely as optimality allows". The log shows `Underfull \vbox (badness 10000) … while \output is active`.
   - The last page (10) is unbalanced: the references fill only the left column. Use `\balance` or `\IEEEtriggeratref`.
6. **Page count.** 10 pages including references. Check the target venue's limit; many IEEE conferences allow 8 + 2 for references.
7. **Supplement.** It is set one-column (`[conference,onecolumn]`), which is acceptable for supplementary material. Its 11 underfull-hbox warnings come from the wide checklist and log tables.
