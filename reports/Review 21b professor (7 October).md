# Review 21b: "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth"

Reviewer role: professor in a systems group, reviewing as a fellowship and workshop referee in the spirit of Hoefler and Belli (SC'15). Date: 7 October 2026.

## Materials read

- `paper/paper.pdf`, all 38 pages: main text pp. 1-10, references, Appendices A-L pp. 14-38. I read it as extracted text and checked the layout of the tables and figures.
- `paper/paper.tex` (the main-text source) and two table sources, `paper/tab_dm.tex` (Table 4) and `paper/tab_regtests.tex` (Table 19). I read these only to confirm captions and labels.
- `paper/supplement.pdf`, read selectively. I read Tables 16-17 (job 110/111 clause scoring) in full and searched the rest.
- Scripts, read only for definitions, not results:
  - `scripts/decomp_measured.py` (its docstring and `rows()`);
  - the `B_host`/`host_rates` definitions in `scripts/reanalysis.py` and `scripts/hostdep_model.py`.
- Infrastructure and bound inputs:
  - `gpu/vast_ledger.json`;
  - the clone-at-boot logic in `gpu/runner.sh` and `gpu/vast.py`;
  - `prereg/speed_limit_v2.json`, for R\* and the host-B limits.
- Second checkout `/home/claude/gpu-branch`:
  - job scripts for 093-111: headers of 107, 108, 109, 110 and 111 in full, and the diffs of every commit made after a rental had started (jobs 100, 102, 105, 106, 107, 111);
  - `jobs/ec2/job110_uuids.txt`;
  - `results/` for job 081 and jobs 093-111: `ec_*.jsonl`, `st_*.json`, `concur.txt`, `fetch_table_law_gptoss.json`, `g_prof.json`, `validity.txt`, `gate.txt`, `v0.txt`, `nvidia-smi-q.txt`, `lscpu.txt`, `manifest.json`, `summary.txt`, `bs1.jsonl`;
  - `git log --format='%h %ct %at'`, for commit times only.
- My own reanalysis code, all in `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/rev21b/`: `common.py`, `regtime.py`, `job109.py`, `panel.py`, `broad.py`, `claims2.py`, plus inline checks.

## Independence statement

- I opened nothing under `reports/` other than to write this file.
- I did not open `prereg/*outcome*.md`, `research_notes/`, `apply/`, `paper/paper_v1_prereview.tex`, or any review, number-check, plan or progress-log file.
- I did not read commit messages. I used only hashes and timestamps from `git log`, and `git diff` of job-script contents.
- Every number in the claims table below comes from my own code reading the raw per-problem rows, counters, probes and profiles. I did not use the authors' derived JSON, with one exception: R\* and the host-B limit come from `prereg/speed_limit_v2.json`, because recomputing MIN needs the routing traces.
- I modified nothing in either repository except writing this file, and I made no commits.

---

## Summary

The paper studies batch-1 decode of gpt-oss-120b (MXFP4) and Qwen3-30B-A3B (BF16) when most experts live in host DRAM, on rented RTX 5090 desktops. It contributes five things.

1. **A bound and how close systems get to it.**
   - Eq. (1) bounds the time per token by MIN-with-bypass's host reads at the machine's highest probed host read rate. Eq. (2) is a "demand" bound for reads that wait for the router; Eq. (3) is an "ordered" bound for CPU reads that cannot be issued ahead.
   - A microbenchmark replays MIN's per-layer reads at 86-96% of the bound's read time.
   - The authors' llama.cpp expert cache runs at 31-54% of Eq. (1)'s speed at gpt-oss 11% on 30 machines with consumer processors. It beats FreeToken at 11 of 12 configurations and stock llama.cpp at all 12, on two hosts.
   - Scored against datasheet bounds, published systems reach a median 13.6%.
2. **A measured decomposition of the gap.** Oracles inside the engine split the deployed cache's time beyond Eq. (1).
   - Caching MIN's set while still reading each admission twice, or reading once with the deployed set, each close about 0% of the gap.
   - Doing both closes about a third: 33% on 15 panel machines, and 35% on 5 new fast-link machines under a registered test (job 109).
   - A read-ahead oracle closes another 15-19%, and about half remains. That half is *attributed* to the GPU's non-expert time in series with the reads and to the oracle's reads beyond MIN.
3. **Online policies recover little.** Policies built from the engine's admission margin and layer-ahead prediction capture at most 6% of the read-ahead oracle's gain (registered bound: 35%), because they read as much as the deployed cache.
4. **A second-card test.** A trend in the link-to-CPU ratio, fitted on the 15 RTX 5090 panel machines and frozen in a job header, predicted MIN-1-read and the read-ahead oracle on three RTX 4090 hosts within 6% (registered). The same trend misses new fast-link RTX 5090s by up to 18%.
5. **The value of foresight (post hoc, on traces).** Half of foresight's value needs about 0.65 C distinct experts per layer of look-ahead, far more than the next token.

The paper also includes a full prediction log of 1,614 clauses over jobs 073-111, of which 309 failed, scored in the appendix and supplement.

## Overall assessment

This is the most auditable benchmarking submission I have refereed in some time.

- I re-derived 35 quantitative and procedural claims from raw rows, counters, probes, profiles, the ledger and git timestamps. Every number the paper prints that I recomputed matched to its printed precision.
- The registration mechanics hold up: every one of the 87 rentals of jobs 093-111 started after its job script was committed.
- Failed hosts, voided jobs and failed clauses are reported rather than buried.

The weaknesses are not errors. They are places where a claim is worded more generally than its evidence:

- the second-card transfer;
- "two named parts account for" what the oracles leave;
- the literature audit.

There are also residual design and statistics issues:

- the 2x2's "how it is read" factor is confounded with timing;
- the bound rests on a single probe run taken during the model download;
- percentile bootstraps are used over 4-5 machines;
- many registered bands are lenient.

Clarity has improved, but it is still the main obstacle for a reader.

---

## Strengths

**S1. Reproducible from raw data, to the digit.** I reproduced the following from `ec_*.jsonl`, `st_*.json`, `concur.txt`, `fetch_table_law_gptoss.json` and `g_prof.json`, using only the definitions in the job-109 header:

- every entry of Table 5 (36 RTX 5090 entries and 24 RTX 4090 entries, with the bracketed predictions);
- all 32 entries of Table 4, both the Panel and the New columns;
- the frozen RTX 5090 trend coefficients in job 110's header (all eight intercept/slope pairs, and the stated maximum residuals 0.081/0.094 and 0.027/0.060), refitted from the 15 named panel machines;
- Table 3's host-B gpt-oss row and the Qwen3 12.5% cell (1.294 [1.278, 1.312], 1.275, 1.154, 2.00x llama.cpp, 1.032 [1.022, 1.042]) and the 172 tok/s bound;
- the Eq. (4) predictions of Tables 23-24 and job 108's verdict (median error 6.7%; 5.3% without the EPYC 7543);
- the fewest-admission geometric mean of 1.215x over 9 machines, and the Spearman correlation of 0.886 [0.63, 0.97] over 19 machines.

This is what the artifact claims, and it delivers.

**S2. Registration is real, and its deviations are disclosed.**

- *Timing.* For all 87 rentals of jobs 093-111, the ledger's start time follows the last commit of the job script; the smallest lead is 2.71 s (105f), matching Appendix D's "at least 2.7 s".
- *Post-start edits.* Every commit made after a rental started touches only host substitutions (jobs 102, 105, 106, 107) or a relaunch rule (job 111), with the predictions unchanged. The one exception is job 100's added ninth prediction, which the paper discloses together with its weakness.
- *Ledger slips.* Two factual slips in headers are also disclosed, and I confirmed both: job 109's claim that none of its offers was in the ledger (offer 51325952 had been rented for 099d and 100e), and job 108's three re-listed offers.
- *The RTX 4090 reference-loss error.* The registered output check in job 110 used the RTX 5090's reference loss. The relaunch kept the predictions and only corrected the reference. Both are stated in the main text and in Limitations, which is the right place.

**S3. The statistical unit is argued from the data.**

- Within a launch, the V2 spread between rounds is 0.01-0.42%.
- Relaunches of the same GPU drift up to 3.2% (RTX 5090) and 5.5% (RTX 4090).
- Two machines with the same CPU differ by up to 29%.
- Per-problem times show negligible serial dependence (lag-1 autocorrelation between -0.12 and -0.26). The paired bootstrap over problems within a host is therefore defensible, and making the machine the unit for cross-machine claims is correct.

**S4. Honest evidence labels and separation of claims.**

- Table 1 states failures in-line: 5 of 33 clauses failed; held on host B but failed on host S; "misses new fast-link RTX 5090s by up to 18%"; three of four tests of the closed-form account failed.
- The closed-form account has been demoted to an appendix and is labelled "exploratory, not confirmed".
- Table 4 separates *Measured* from *Attributed* rows.
- Paragraph headings carry "(registered)" or "(exploratory)".

**S5. Validity gates are registered before launch and applied as registered.**

- The gates are V0, V0c, VR and V1-V3. Every host that started is reported, including the server machines that failed (5 of 8 server launches failed a check).
- Job 107's "inconclusive" verdict, under its own rule that fewer than three valid machines makes the test inconclusive, is reported as such, not rescued.

**S6. Useful, actionable conclusions for builders.**

- Read each admission once.
- Admit as rarely as optimality allows: the fewest-admission schedule makes 0.60-0.73 of the greedy schedule's copies.
- Choose the read path by the link-to-CPU ratio.
- Measure look-ahead in distinct experts rather than tokens.
- Report distance to the machine's bound.

The Hoefler-Belli checklist in Appendix B is mapped rule by rule.

---

## Weaknesses (most important first)

### W1. The second-card claim rests on one point of the ratio axis, and its agreement is about the size of relaunch noise

The abstract and Conclusion say that a trend fitted on RTX 5090s "predicted the oracles' speed" on a second card. The evidence is narrower.

- **One ratio point, two distinct configurations.** The three valid RTX 4090 hosts sit at ratios 0.37-0.40. Two are the same CPU model (Core Ultra 9 285K, separate offers), and their numbers are almost identical: deployed cache 20.44 vs 20.40 ms per token; MIN-1-read/deployed 1.007 vs 1.005. There is in effect one point on the ratio axis, with two distinct host configurations.
- **The prediction for MIN, 1 read is "no gain".** At ratio 0.38 the trend predicts MIN-1-read/deployed of 0.99-1.01. A null model that predicts 1.00 passes the registered 0.10 log band equally well (worst case |ln 0.963| = 0.038). Only the read-ahead oracle's prediction (1.11-1.13, measured 1.13-1.20) is informative.
- **The agreement is no tighter than relaunch drift.** Relaunching the same GPU moved MIN-1-read/deployed from 0.953 (110c, round 1) to 1.005 (111g), and the deployed time by 5.5%. "Within 6%" is thus the same size as launch-to-launch drift on a single machine.
- **The trend misses on its own card.** On five new fast-link RTX 5090s, I confirm misses beyond the band in 5 of 18 cells, the largest +17.6% (109e, read-ahead, 11%).
- **Selection of relaunched machines.** Job 111's header lists four offers (b, c, d, e), but `jobs/ec2/job110_uuids.txt` holds only three UUIDs, and the 111e wrapper was never launched. The ledger shows 110e still running (stopped at t = 1791417710) when job 111 was committed (t = 1791417120), which makes the "oversight" plausible. I checked 110e's single round: deviations of -0.036 (MIN, 1 read) and -0.057 (read-ahead) in log, inside the band. The omission therefore does not flip the 11% result, and the paper should say so.

What the data support is this: at a link-to-CPU ratio near 0.38, an RTX 4090 host behaves as an RTX 5090 host at that ratio would. The general sentence in the abstract, Section 7 rule 4 and the Conclusion should be narrowed, or the test repeated across a spread of ratios and CPU models with a naive baseline predictor alongside.

### W2. "Two named parts account for" what the oracles leave holds only on fast links

Section 4 says the remaining half of the gap is accounted for by T_GPU in series with the reads and by the oracle's reads beyond MIN. With the paper's own definitions (T_GPU = 2.94 ms; extra reads at B_host), the residual at gpt-oss 11% is:

- 35% of the gap on panel host 099d (ratio 0.40) and 47% on 099f (ratio 0.29);
- 32-36% on the three RTX 4090 hosts (22-25% at 25%).

Table 4 averages the two slow-link panel machines into "4 [-4, 13]" and reports no RTX 4090 decomposition.

Two further problems:

- On the RTX 4090 the profiled G is 5.1-5.3 ms against 4.1-4.5 ms on the RTX 5090, so a single constant T_GPU taken from RTX 5090 profiles is not the right attribution there.
- Eq. (3) is the natural explanation for a slow-link residual, but it is not used in the decomposition.

Please qualify the sentence to fast links, report the slow-link and RTX 4090 decompositions, and state whether Eq. (3) closes the residual.

### W3. The 2x2's "how it is read" factor is confounded with timing

In Table 2 the two levels of the factor differ in two ways at once:

- "1 read" means fetched into the slot *in the step*, on the critical path.
- "2 reads" means served by the CPU and then copied *in the background*, overlapped with compute.

So the factor changes the number of reads and also when and on which path they happen. The paper's own mechanism ("read twice, its admissions also land late, so that arm misses more") concedes the point. "What is cached and how it is read pay only together" is therefore an interaction between MIN's set and two engine paths, not a clean read-count effect.

For the record, on slow links the "together" cell is about zero (RTX 4090s: -6, +1, +1%; panel 099d 8%, 099f -23%). The interaction is a fast-link phenomenon, which the abstract does acknowledge.

A cell that holds timing fixed would make the factorial interpretable: MIN's set read once, issued in the background or ahead. The read-ahead oracle could then be treated as a third level of a "when" factor rather than a separate step. Without that cell, the wording should be "the two engine paths" rather than "reads".

### W4. The bound depends on a single probe run, mostly taken during the model download

The first yardstick ("% of the bound") divides by the highest reading of one probe run. On most machines that run happened while the model downloaded in the background.

- By the paper's own closed-form account, the engine's implied read rate exceeds the probe's best reading by more than 5% on 12 of 84 launch-budgets, by up to 1.22x. Using the probe's second-highest reading changes the fit (Table 25).
- The abstract's "31-54%" spans two flagged points. The low end (31.2%) is a Threadripper 9960X with 178 GB/s, an HEDT part counted as "consumer" under the paper's definition; without it the low end is 38.4%. The high end (54.0%, 100f) is the value the paper itself calls "probably too high".

For a paper about upper bounds (Hoefler-Belli rule 11), the bound's rate should come from a dedicated measurement on an idle machine, repeated, with both the maximum and the spread reported. At minimum, re-probe a subset of machines and show how the headline range moves.

### W5. Small-n summaries use percentile bootstraps and lack intervals

- **Table 4 New column.** Its intervals are percentile bootstraps over 5 and 4 machines. With n = 4 there are only 35 distinct resamples. Against t-intervals on the same values:

  | Table 4 New cell | Percentile bootstrap | t-interval |
  |---|---|---|
  | "Both" at 11% (n = 5) | [27.6, 43.7] | [21.9, 47.5] |
  | "MIN's set alone" at 11% (n = 5) | [0.5, 6.0] (printed "3 [0, 6]") | [-0.9, 7.7], which includes zero |

  Appendix I already notes this for the 9-machine geometric mean. Apply the same correction everywhere n ≤ 10, or print the per-machine values.
- **Table 5** has no intervals.
- **The second-card test.** "Within 6%" is a maximum over six point estimates, and the transfer clauses are scored "held (point)" because no interval exists (supplement, Table 17). A within-host interval, from problems or rounds, for each ratio-to-prediction would cost nothing to add.

### W6. Many registered tests are lenient, and the paper does not distinguish severe from lenient ones

- **Overall.** Of 1,614 clauses, 671 (42%) "held on the point estimate": they had no interval, or an interval that crossed the threshold.
- **Job 109.** Its 11 predictions all hold; I recomputed each one. But several bands are wide relative to what earlier jobs had shown:
  - "together" in [0.25, 0.55];
  - dk/deployed in [0.98, 1.06];
  - capture ≤ 0.35, when job 106 had already shown margin gains of about 1-2%.
- **Job 110.** Clauses Q3-Q5 are range checks, for example deployed/Eq. (1) in [1.8, 3.5].
- **Job 109's population.** The rule "link-to-CPU ratio at least 0.5" was drawn after the data of jobs 093-101, which the header honestly says. That makes job 109 a confirmatory replication of a known effect on a population chosen where it holds. That is legitimate, but it is not a risky prediction.

I suggest marking each prediction with the value earlier data implied, so readers can see which tests could plausibly have failed.

### W7. The literature audit is a headline claim, but Table 1 omits it and nothing labels it post hoc

The Introduction states that published systems reach a median 13.6% of their bound, and I confirm that median from Table 29's 20 in-class rows. But:

- the claim is not in Table 1;
- it uses datasheet ceilings rather than probes;
- it applies a per-layer, one-token bound to designs that pool caches, pin whole layers or verify several tokens per pass;
- it uses uniform routing for 11 of the 52 rows (Table 29's medians are over all 52; the 13.6% headline itself uses only trace-scored rows).

"Ours, scored the same way, 27%" compares different hardware. Add the audit to Table 1 as post hoc, and move Appendix K's three caveats into the paragraph in Section 3.

### W8. The registration timestamps are self-reported

- Commit times are author-controlled. The paper cites GitHub push records, but the artifact contains no export of them.
- No result directory records the commit each machine ran: `manifest.json` and `patch_sha.txt` hash the patch and the UUID list, not `git rev-parse HEAD` of the boot clone.

Recording HEAD at boot is a one-line fix. Depositing headers with an external timestamping service (OSF, AsPredicted, or a signed tag) would make the ordering independently verifiable.

Separately, the URLs in `gpu/runner.sh` and `gpu/vast.py` name the author's GitHub account. That matters if the venue is double-blind.

### W9. Scope

The paper measures one engine, so the decomposition is of this engine's gap. It covers two models, AIME prompts that were also used during development, teacher-forced text, and in practice consumer desktops only, since most server hosts failed the gates. All of this is disclosed in Limitations. The title and abstract still read as more general than the measured population.

### W10. Minor inconsistencies

- Table 9 lists job 110 as "1 failed, 1 untested", while Appendix D calls job 110 "void as registered".
- Section 2 defines "ran stably" as processes agreeing within 2%, but gate V2 compares *rounds* of process A.
- The Section 5 paragraph tagged "(registered)" opens with two post-hoc statements: the Spearman correlation over 19 machines, and "on fast links the ratio sets the direction of the gain, not its size".
- Table 1 lists "A closed-form account of the time" in the *Claim* column. It should read "not confirmed", as the main text does.
- Section 2 defines a budget as C/E but never states that 11% means C = 14 (and 25% means C = 32).

---

## Clarity

Earlier rounds scored clarity 2, then 3 several times. This version is better organised, but the sentence-level and naming burden remain the main obstacle. I score it **3/5**.

### What works

- **Table 1.** Claim, test and outcome, machines and section, with failures stated in the table. A reader can triage the paper from it.
- **The "Yardsticks" paragraph in Section 2.** It names the four denominators up front, which was the single largest source of confusion before.
- **Table 2.** Configurations are grouped into "No foresight" and "Oracles", with a reads column.
- **The figures.** Fig. 1's schematic of the two reads and Fig. 2's staircase make the 2x2 visible.
- **The running example in Section 5.** Concrete milliseconds on one 9950X: 10.0 / 13.0 / 20.6 / 15.2 / 13.8. I confirmed it from job 096a.
- **Table 4's split into Measured and Attributed rows,** with an explicit sum identity in the caption.
- **Paragraph tags** "(registered)" and "(exploratory)", and the demotion of the closed-form account to an appendix.
- **Section 7's rules and the Limitations list,** which are concrete and quantified.

### What still makes it hard to read

**C1. Four yardsticks, switching by section, and mixed even within the abstract.** "35%" is a share of the gap. "6%" is a share of the read-ahead oracle's gain. "0.78-0.93 of the gain" is a share of MIN's gain over *admit every miss*. "31-54%" is a share of the bound's speed.

The Section 6 baseline is especially misleading in time. *Admit every miss* runs up to 2.36x slower than the deployed cache on slow-link hosts (099f), so "a 16-token window recovers 0.78-0.93 of the gain" says little about what a builder gains over the deployed cache. On the two lowest-ratio hosts of job 100 (ratios 0.41 and 0.54), the exact 16-token window copied in the step runs 0.89x the deployed cache at gpt-oss 11% (Table 18).

Pick one primary unit (ms per token, or time relative to Eq. (1)), put the others in parentheses, and in Section 6 also give speed relative to the deployed cache.

**C2. Configuration names with internal commas, used as nouns in running prose.** "The deployed cache takes 20.6, MIN, 1 read, 15.2, and the read-ahead oracle 13.8." "MIN, 1 read, is faster than the deployed cache at every budget." These sentences must be parsed twice. Use typographic names such as MIN-1R, MIN-2R and Deployed-1R, or small caps.

**C3. Too many names.**

- Table 2 has 12 configurations, and Table 7 has about 28 codes.
- Host names come in four schemes: A/B/S; O1-O6; Pa-Pj; and descriptive tags such as "Pd again" or "285K, second". Appendix tables also use job letters (109a...).
- Appendix tables mix codes (foa, dk, lrn) with the main text's names.

A single host table, with stable IDs used everywhere and one naming scheme, would remove much of the friction.

**C4. Qualifier stacking.**

- The abstract is 236 words with about nine quantitative statements. Many carry three or four qualifiers ("at the smallest gpt-oss budget on machines with consumer processors"; "on fast-link machines rented for a test we registered before they ran").
- The Section 5 paragraph on the second card runs about 25 lines. It combines the fit, the freeze, the test, the out-of-sample miss, an unrun path, and the story of the failed V1 check and the relaunch.

Split such paragraphs: first the result, then the caveat, then process detail in a footnote.

**C5. Section 5 mixes three questions under "How Foresight Must Be Spent":** how to spend foresight, transfer to a second card, and online policies *without* foresight. Split it into three subsections, or move the online policies next to Section 4.

**C6. Process narrative and job numbers in results text.** The main text refers to jobs 104, 105, 107, 108 and 109, and the RTX 4090 reference-loss mishap appears in the middle of a result. The reader needs Appendix D to decode these. Keep job numbers in Table 1 and the appendix only.

**C7. The appendices are walls of text.**

- Appendix D's "Failures, jobs 073-098" is about 85 lines of continuous prose listing failures.
- The paragraphs on jobs 099-104 and 105-108 are similar.

The supplement already has the clause tables. The appendix should give one table (job, clause, predicted, measured, why it failed) and point to the supplement, rather than paraphrase it. Appendix E and Appendix I have the same problem.

**C8. Fig. 7 (Appendix H) reuses the paper's title.** Its caption, "Where the seconds go, machine by machine", sits on a split implied by the *unconfirmed* closed-form account. It is easy to mistake for the measured decomposition of Section 4. Retitle it, for example "The split implied by Eq. (4) (exploratory)".

**C9. Terms with unusual meanings.**

- "Read once/twice" encodes *where and when* a read happens as well as how many reads there are (see W3).
- "Fast/slow link" is a ratio, not a link speed: a 55 GB/s link counts as "slow" next to a Threadripper 9960X whose host memory probes at 178 GB/s (ratio 0.32).
- "Consumer" includes HEDT Threadripper parts.

Each of these needs a short sentence wherever the term first carries weight.

**C10. Length.** Ten pages of main text, 25 pages of appendices and a 53-page supplement is a heavy load for a workshop paper. The main text is now self-sufficient for the headline claims, which is good. The appendices could be cut by half by turning prose into tables.

---

## Registration and reporting audit (summary of what I checked)

- **Timing.** All 87 rentals of jobs 093-111 started after their job script's last pre-start commit, with a minimum lead of 2.71 s. Push times, as opposed to commit times, cannot be checked offline.
- **Post-start amendments.**
  - Jobs 102, 105, 106 and 107 changed hosts only, with predictions unchanged.
  - Job 100 added a ninth prediction after hosts 100a-d had started (disclosed).
  - Job 111 added a rule for relaunching after the model download failed (disclosed).
- **Failed hosts.**
  - 110a stopped at the ratio gate with 0.2499.
  - 110b-e failed V1 at losses of 0.1963-0.1965 against 0.190.
  - 111b and 111c failed the model download and were relaunched as 111f and 111g.
  - 109f's 25% round was cut by the deadline (pf has n = 18).

  All of these are reported. The paper calls job 110 "void as registered", while Table 9 counts it as "1 failed, 1 untested" (see W10).
- **Labels.** Table 1's evidence labels are accurate. Its "Mach." column is right for every row I checked: 15 + 5, 5, 15 + 3, and 9.
- **Table 4.** The Measured/Attributed split is correct by construction, and the identity Both + read-ahead + left = gap holds in my reanalysis.
- **Table 19.** The "As registered" column is honest (failed, failed, inconclusive, failed). The adjacent "Set aside" column re-scores after the fact and, for job 106, turns "failed" into "held (12/12, 2.3%)". It is labelled "after the fact", but it should be separated visually, for example in grey or as a footnote, so it cannot be read as a verdict.
- **Separation of post hoc from registered.** This is good at the paragraph level and imperfect within paragraphs (W10). The audit and the horizon rule are post hoc. Only the horizon rule is labelled so in Table 1.

---

## Table of claims checked against raw data

Data sources: `ec` = per-problem rows (decode_ms/n_decode, nll_sum/nll_n); `st` = counters; `probe` = `concur.txt`; `ft` = `fetch_table_law_gptoss.json`; `G` = `g_prof.json`; `ledger` = `gpu/vast_ledger.json`; `git` = commit timestamps.

| # | Claim (location) | Data and method | My result | Verdict |
|---|---|---|---|---|
| 1 | Every rental of jobs 093-111 started after its job script was committed; minimum lead 2.7 s (App. D) | ledger start vs `git log %ct` of `jobs/NNN_*.sh` | 87 rentals, all after the commit; minimum lead 2.71 s (105f) | Confirmed (commit, not push, times) |
| 2 | Post-start header edits only substitute hosts; predictions unchanged (App. D) | `git diff` of every post-start commit | 102, 105, 106, 107: hosts only; 100: ninth prediction added after 100a-d started; 111: download-relaunch rule | Confirmed; 100 and 111 disclosed |
| 3 | Job 111 relaunched 3 of 4 eligible hosts; e omitted by oversight (§9, App. D) | job 111 header, `job110_uuids.txt`, ledger | header lists e's offer; UUID file has b, c, d only; 111e never launched; 110e still running at job 111's commit | Confirmed; 110e round 1 inside band (-0.036, -0.057 log) |
| 4 | Job 109's "none in ledger" was wrong for one offer (App. D) | ledger | 51325952 rented in 099d, 100e, then 109d | Confirmed |
| 5 | Job 108's list held three offers already in the ledger (App. I) | ledger | 53424353 (100b), 54156078 (099a), 51325952 | Confirmed |
| 6 | 110a stopped at ratio 0.2499; 110b-e failed V1 after one round | `gate`, `validity` | as stated (0.1963-0.1965 vs 0.190) | Confirmed |
| 7 | RTX 4090 computes a loss about 3% higher in every configuration (§5) | `ec` nll | 0.1957-0.1968 vs 0.1895-0.1910 on RTX 5090 (+3.1 to +3.4%) | Confirmed |
| 8 | All five job 109 hosts passed every gate; the deadline cut the 5950X's 25% round | `validity`, files | V1/V3 ok, V2 spreads 0.05-0.42%; 109f C32 partial | Confirmed |
| 9 | Table 5, job 109 (1 read, ahead, online, capture; 2 budgets) | `ec`, geometric mean over rounds, definitions from header | all 36 entries match to printed precision | Confirmed |
| 10 | Table 5, job 111, including bracketed trend predictions | `ec` + frozen coefficients | all 24 entries match | Confirmed |
| 11 | Trend within 6% at 11% and 5% at 25% on RTX 4090s (§5) | as above | max deviation 5.9% (both3p, 111g) and 4.7% (fetch, 111d) | Confirmed; same size as relaunch drift (W1) |
| 12 | Trend misses 5 of 18 new RTX 5090 cells, by up to 18% (§5) | `ec`, 109a-f | 5 of 18 beyond 0.10 log; max +17.6% | Confirmed |
| 13 | Trend frozen on the 15 panel machines (job 110 header) | refit ln(X/deployed) on ln(ratio) | all 8 (a, b) pairs and max residuals reproduce exactly | Confirmed |
| 14 | Table 4, Panel column (15 machines) | `ec`, `st`, `probe`, R\* | all 16 entries within rounding; intervals match | Confirmed |
| 15 | Table 4, New column (5/4 machines) | same | all 16 entries match | Confirmed; percentile intervals narrower than t (W5) |
| 16 | Interaction sign held on 9 of 10 panel (099) and 5 of 5 new machines (§4) | `ec` ms | 099f negative (-0.48 ms); 109a-f +1.5 to +7.5 ms | Confirmed |
| 17 | Best online policy 1.007-1.024x; capture at most 6% (§5) | `ec` | 1.007-1.024; maximum 5.7% | Confirmed |
| 18 | Online policies read 1.66-1.68 R\* (§5) | `st` | 1.665-1.678 | Confirmed |
| 19 | A higher margin cuts host reads by 4-6% at 11% (§5) | `st` | 4.3-4.4% (RTX 5090), 5.2% (RTX 4090) | Confirmed |
| 20 | RTX 4090: layer-ahead policies 0.64-0.72x; margin best at 1.033-1.041x; capture up to 26% (§5) | `ec` | 0.644-0.724; 1.033-1.041; 25.7% | Confirmed |
| 21 | Relaunched machines within 3.2% (RTX 5090) and 5.5% (RTX 4090) (§2) | deployed ms by GPU UUID | 3.17% (TR 9960X); 5.49% (110c to 111g) | Confirmed |
| 22 | Same CPU model differs by up to 29% (§2) | `ec` | 285K: 099a 18.14 vs 099f 14.06 ms (1.29x) | Confirmed |
| 23 | Fewest-admission gains on all 9 stable machines, geometric mean 1.21x (1.13-1.30; t 1.11-1.33); lost on the two slowest-link unstable hosts (§5, App. I) | `ec` fetchplan | 1.215 [1.131, 1.303], t [1.110, 1.329]; losses on 106c (0.845, ratio 0.14) and 107e (0.974, ratio 0.21) | Confirmed |
| 24 | 31-54% of Eq. (1) on consumer machines; 41-48% on the 5 new ones; 40-54% of the max of Eq. (1) and Eq. (3); Eq. (3)/Eq. (1) median 1.00, max 1.56 (§3) | `ec` + `probe` | 31.2-54.0% (n = 30); 41.5-48.0%; 40.5-54.0%; 1.000 and 1.557 | Confirmed; low end is HEDT (W4) |
| 25 | Spearman 0.89 [0.63, 0.97] over 19 machines (§5) | `ec` + `ft`, first launch per GPU, jobs ≤ 104 | 0.886 [0.63, 0.97], n = 19 | Confirmed |
| 26 | Running example: 10.0 / 13.0 / 20.6 / 15.2 / 13.8 ms (§5) | 096a | 10.02 / 12.96 / 20.64 / 15.17 / 13.82 | Confirmed |
| 27 | G = 4.1-4.7 ms for gpt-oss; EPYC traces held almost no kernels (App. H, I) | `G` | 4.05-4.65 ms on RTX 5090; 108a 0.00, 108b 0.02; RTX 4090 5.08-5.65 | Confirmed |
| 28 | Eq. (4) vs measured (Tables 23-24); job 108 median 6.7% (5.3% without the 7543); 7543 off by 34.3% | `st` + `G` + `probe` | all reproduce (7543: 14.94 vs 22.73 ms; AMD ES -58.9%) | Confirmed |
| 29 | Table 3 host B: ours/FreeToken 1.294 [1.278, 1.312], 1.275, 1.154; 2.00x llama.cpp; bound 172 tok/s; Qwen3 12.5% 1.032 [1.022, 1.042] | `bs1.jsonl` (job 081, launch 2), `probe` | identical (B_host 87.5 GB/s, 172.3 tok/s) | Confirmed |
| 30 | An exact 16-token window recovers 0.78-0.93 of the gain at 11%; the lowest machine missed 0.80 (§6) | `ec` 099a-j | 0.78-0.93; 099f 0.78 | Confirmed; baseline up to 2.36x slower than deployed (C1) |
| 31 | Table 9 totals; Appendix D subtotals | arithmetic over the table rows | 550/671/309/84; 218/193/142/12 (8 untested + 4 void) | Confirmed; job 110 "failed" vs "void" (W10) |
| 32 | 124 rentals (74 offers) of jobs 058-108 cost $87.2; 87 rentals in jobs 093-111 (App. D, L) | ledger | 124, 74, $87.22; 87 | Confirmed |
| 33 | In-class audit median 13.6% over 20 rows (§3) | Table 29 values | 13.6% | Confirmed (bounds not recomputed) |
| 34 | Job 109: 11 of 11 predictions held (App. D) | my per-machine values against the header bands | all hold; P8's ratio ≥ 0.9 clause had no machine | Confirmed |
| 35 | Not claimed: residual on slow-link machines | `ec`, `st`, `probe` | 099d 35%, 099f 47%, RTX 4090 32-36% of the gap at 11% | Missing from paper (W2) |

---

## Questions for the authors

1. Was 110e's UUID left out of `job110_uuids.txt` because 110e was still running when job 111 was committed? Will you report 110e's single round? I find it inside the band.
2. Can you report Table 4 separately for slow-link machines (099d, 099f) and the RTX 4090s, with a card-appropriate T_GPU? Does Eq. (3) account for the residual there?
3. On the RTX 4090 test, how do two simple predictors score: X/deployed = 1 for MIN, 1 read, and the nearest RTX 5090 host in ratio (099d at 0.40, 103e at 0.40)? Can the test distinguish the trend from either of them?
4. Is there a configuration with MIN's set read once *in the background* (no in-step fetch)? With it, does the 2x2 interaction survive when the timing is held fixed?
5. Can you re-probe a subset of machines, idle and repeated, and show how the "31-54%" range and Table 4's denominators move?
6. Can you export the GitHub push events for the gpu branch, and add `git rev-parse HEAD` to each machine's manifest?
7. Why are Threadripper parts counted as consumer? How do the summaries change if HEDT parts are a separate class?
8. For each job-109 prediction, what value did the panel data imply before you set the band? Which predictions did you consider risky?

## What would raise my score

- **The second-card claim (W1).** Narrow it to what the data show, or rerun the test across a spread of ratios (at least 0.5 and at most 0.3) and at least five distinct CPU models, with a null predictor scored alongside.
- **The residual (W2).** Report the slow-link and RTX 4090 decompositions and qualify "two named parts account for it". Use Eq. (3) to explain the slow-link residual, if it does.
- **The 2x2 (W3).** Either add a timing-matched cell, or rename the factor so it no longer claims to be about the number of reads.
- **The probe (W4).** Re-measure idle and repeated on a subset of machines, and report the headline range's sensitivity in the main text.
- **Intervals (W5).** Use t or BCa intervals, or per-machine values, wherever n ≤ 10. Add within-host intervals to Table 5 and to the transfer clauses.
- **The audit (W7).** Add it to Table 1 as post hoc and bring its caveats into Section 3.
- **Clarity.**
  - Use one primary yardstick.
  - Give configurations typographic names without commas.
  - Use one host-ID scheme.
  - Split Section 5.
  - Tabulate Appendix D's failures.
  - Retitle Fig. 7.
  - Move process narrative out of results paragraphs.
- **Verifiability (W8).** Record the commit each machine ran, and timestamp the headers externally.

## Scores

| Criterion | Score | Reason |
|---|---|---|
| Overall | 7 / 10 | Acceptable, near strong; exemplary auditability, with claims that need narrowing and prose that needs work |
| Soundness | 4 / 5 | Every number I recomputed matches; the issues are scope and framing (W1, W2, W3), not errors |
| Methodology | 4 / 5 | Registration, ledger, gates and machine-as-unit are exemplary; the probe, small-n intervals, lenient bands and factorial confound keep it from 5 |
| Significance | 3 / 5 | Useful, actionable rules and a reporting practice; narrow population (one engine, two models, batch 1, consumer desktops) |
| Clarity | 3 / 5 | Better scaffolding (Table 1, yardsticks, labels), but naming, qualifier density, yardstick switching and appendix prose remain hard |
| Confidence | 4 / 5 | Extensive raw-data checks; I did not recompute MIN/R\* from traces, the trace-based horizon results, or the audit's bounds |
