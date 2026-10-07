# Review 20b: "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth"

Referee role: systems professor, fellowship and workshop referee, reviewing in the spirit of Hoefler and Belli (SC'15). Date: 7 October 2026.

## Materials read

- `paper/paper.pdf` (37 pages): the whole main text (Sections 1 to 10, Tables 1 to 4, Figures 1 to 5) and all appendices A to L (Tables 6 to 28, Figures 6 to 9). I also read `paper/paper.tex`, `paper/tab_dm.tex` and `paper/tab_prereg.tex` for clean text.
- `paper/supplement.pdf` (48 pages): the structure, the scorecard rules and the first clause tables (jobs 073 to 078). I did not read every clause row.
- Artifact, `/home/claude/moe-speed-of-light`. Scripts read to learn definitions, not results: `scripts/decomp_measured.py`, `scripts/reanalysis.py` (`features()`), `jobs/ec2/fetch_table.py` (`bandwidths()`), `scripts/sumlaw_paper.py` (`profiles()`), `scripts/audit_ours.py` and the docstring of `scripts/reg_timing.py`. Prereg data files (JSON only): `reanalysis_hosts.json` (used for the launch list, GPU UUIDs and link-to-CPU ratios, all three of which I re-derived or cross-checked), `speed_limit_v2.json` (R* values, D, S), `gpu_pushes.json`, `scorecard_099.json` to `scorecard_108.json`. Also `gpu/vast_ledger.json`.
- Raw results, `/home/claude/gpu-branch`: `results/<job>/ec_*.jsonl` (per-problem `decode_ms`, `n_decode`, `nll_sum`, `nll_n`, config with `stats=`), `st_*.json` counters, `concur.txt`, `cores.txt`, `lscpu.txt`, `nvidia-smi-q.txt`, `g_prof.json` and `prof_*.json` (Nsight), `readsched_C*.txt`, `bs1.jsonl` (system comparison, jobs 081 and 089), `v0.txt`, `validity.txt`, `gate.txt`, `manifest.json`, `parity_kl*.json` (job 090), and the routing trace `084c_gptoss_trace@vast/route_aime25_gptoss.npz`. Job scripts: the headers of jobs 097, 099, 100, 102, 105, 106 and 107, and the content diffs of every header edit made after a job's first rental.
- Git history of the gpu branch, read only through `git log --format='%h %ct'` and content diffs (`git show --format=`). I also queried the GitHub Activity API (read-only) for the branch's push events, to check the authors' push log independently.
- My own analysis code is in `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/rev20b/`: `lib.py` (loader), `minsim.py`, `decomp.py`, `readsched.py`, `headline.py`, `fewest.py`, `spearman.py`, `varcomp.py`, `relaunch.py`, `eq3.py`, `eq3b.py`, `eq4.py`, `plain105.py`, `dk.py`, `factorial.py`, `shares.py`, `distinct.py`, `regtime.py`, `pushtime.py`, plus a few inline one-off checks.

## Independence statement

I did not open anything under `reports/` except to create this file. I did not open `prereg/*outcome*.md`, `research_notes/`, `paper/paper_v1_prereview.tex`, `apply/`, or any file named like a review, number check, plan or progress log. I did not read commit messages: I used only `git log --format='%h %ct'` and content diffs with an empty format string. Comments inside job-script headers are file contents, and I read them as such. I ran every computation from the raw rows with my own code in a new scratch folder. I used the authors' scripts only to learn definitions where the paper is silent: how B_c and B_p are defined, which profiles make up T_GPU, and the launch list. I did not use their outputs as results. I modified nothing in either repository and committed nothing.

---

## Summary

The paper studies batch-1 decoding of gpt-oss-120b and Qwen3-30B-A3B on rented RTX 5090 hosts, with most experts kept in host DRAM. It makes four contributions.

1. **A bound (Eq. 1).** MIN-with-bypass host reads per token (R*), divided by the machine's highest probed host read rate, bound any exact-routing system with C slots per layer, relative to that probe. Two refinements follow: Eq. 2, for systems that read only after routing, and Eq. 3, which respects the order of layers. A replay microbenchmark reaches 86 to 96% of the bound's read time. The authors' llama.cpp expert cache runs at 31 to 54% of Eq. 1 on 25 consumer machines at gpt-oss 11%. It leads FreeToken at 11 of 12 configurations and stock llama.cpp at all of them.
2. **A measured decomposition of the gap** (deployed time minus Eq. 1). Oracles inside the engine run a 2x2 design: what is cached (the deployed set or MIN's set) by how it is read (twice or once), followed by reading ahead. At gpt-oss 11%, either change alone closes about 1% of the gap; both together close 39% on the 13 machines whose link-to-CPU ratio is at least 0.5 (33% on all 15). Reading ahead closes another 15%. The remaining 47% is attributed to non-expert GPU time in series with the reads, and to the oracle's reads beyond MIN's.
3. **How foresight should be spent.** Read each admitted expert once, use the fewest-admission hit-optimal schedule (1.21x over the deployed cache on 9 stable machines), and choose the read path by the link-to-CPU ratio.
4. **A price on foresight.** Next-token routing recovers at most about a third of what full foresight saves in reads. Half of it needs about 0.65 C distinct experts per layer of lookahead across 9 models.

Each job's predictions are committed to a public branch before its machine starts. There are 1,483 scored clauses, failures included. An exploratory closed-form account of the time (Eq. 4) failed its registered tests and is reported as such.

## Strengths

1. **The numbers are what the raw data say.** I re-derived 38 quantitative or methodological claims from raw rows, counters, probes, profiles and git or GitHub records. Every one reproduces to the printed precision, or within bootstrap noise (table below). Table 3 reproduces cell by cell, with its intervals. So do Table 4, Table 26, the per-host Eq. 4 predictions in Tables 20 to 23, and the 1.21x [1.13, 1.30] fewest-admission result. The generated-numbers pipeline is real.
2. **The registration is genuinely verifiable, and the record is complete in the ways that matter.** The authors' push log (258 events) is identical to what GitHub's Activity API returns for the branch, with no force pushes. For every rental of jobs 093 to 108, the last header or wrapper commit before the rental precedes it by at least 2.7 s, and every machine started its job at least 26 s after the push. Header edits made after launch are exactly the ones the paper discloses. I checked their diffs:
   - Jobs 102, 105 and 106 changed host lines only. The 102 and 106 edits landed 9 to 14 s after rentals 102b and 102c, and 106a to 106d, were created, but before those machines cloned the branch.
   - Job 100 added a ninth prediction 100 minutes after its first rental.
   - Job 107's two amendments were committed 3 to 4 s before hosts f and g started.

   All 124 rentals ($87.22) are in the ledger.
3. **The statistical unit is argued from data, not assumed.** Machines carry 96% of the variance of the log gain and problems 1 to 3% (I get 96.3% and 1.1% at 11%, 96.2% and 2.8% at 25%). Machine-level summaries are bootstrapped over machines, one launch per GPU (duplicates are removed by GPU UUID; I confirmed that three launches are dropped). The paper even reports t-intervals next to percentile intervals when n is 9.
4. **The engine oracles and the 2x2 are a real methodological contribution.** Measuring "what is cached" against "how it is read", including their interaction, inside a production engine is much more convincing than trace-only gaps (Zhang 2026b; Liang 2026b). The interaction is large (37% of the gap at 11%) and holds in sign on 9 of 10 panel machines, as registered. It is robust: a ratio-of-sums version gives 40%, and dropping the 30-problem launches gives 38.9%.
5. **Failures are reported, not buried.** Eq. 4 is labeled "exploratory, not confirmed" in the main text, and Table 18 shows every test. Validity gates are recorded on the machine (`v0.txt`, `validity.txt`), and the machines they stopped are listed. The 5 of 8 server launches that failed a check are counted.
6. **Hoefler-Belli hygiene is mostly followed.** Absolute speeds sit next to every ratio, and the base case is named. Bootstrap intervals are used, and harmonic and arithmetic summaries are reconciled (within 0.003, as claimed; I find at most 0.0019). Configuration orders are shuffled per cell with recorded seeds, every configuration starts from a cold cache, and versions and flags are released.
7. **The builder rules in Section 7 follow from the evidence.** "Read once" and "a better set read twice closes almost nothing" are exactly what Table 3 shows.

## Weaknesses (most important first)

### W1. The "measured decomposition" measures three of its five parts; the headline remainder is a model attribution on a post-hoc subset
The oracles remove "what", "how" and "ahead". The remainder, "left" (47%), is split by assignment, not measurement. T_GPU is a constant 2.94 ms, the smallest of 14 Nsight profiles, applied to every machine. The prefetching oracle's extra reads are priced at B_host. The residual is -2% at 11% and -9% [-13, -5] at 25%, so at 25% the named parts demonstrably over-account. Yet the abstract states as a finding that "most of the rest is the GPU's own work, which waits in series with reads that depend on routing", and Section 7 rule 5 builds on it. The summary that the abstract and introduction lead with (39%) is over a subset drawn after the data, by a 0.5 threshold on the link-to-CPU ratio. This is flagged in the abstract and in Table 3, but not in the introduction's "doing both closes 39%" or in rule 2. The full, pre-specifiable sample gives 33% [21, 42] (my value), and that should be the primary number.

*Remedy:* measure the serialization term with an oracle. Two options: a run that stages CPU-served experts ahead with lookahead, or a timing-only run with the non-expert kernels elided. Alternatively, label the last three rows of Table 3 "modelled", and drop "measured" from the abstract sentence about the rest.

### W2. The yardstick is relative to one probe run taken during the model download, and B_c and B_p have two definitions
B_host is the single highest reading of one probe run. On most machines that run overlapped the gpt-oss-120b download, as the paper says (`106e` prints "builds and probe done, waiting for the gpt-oss download"; the download took 3,215 s). The paper's own Eq. 4 accounting implies that the engine reads more than 5% faster than this probe on 12 of 84 launch-budgets, up to 1.22x (Table 24). On launch 107d the highest reading is 26% above the second highest (I reproduce this). The gap's denominator, Table 3's shares, the 31 to 54% headline and the audit all inherit this. The read-time microbenchmark (86 to 96%) suggests the probe is roughly right, but "bound" deserves a cleaner ceiling.

Separately, the link-to-CPU ratio uses B_c = the CPU read rate at the helper count (interpolated) and B_p = the zero-copy 16 MB x 64-block reading (`fetch_table.py`). Eq. 3 instead needs "best" rates. I reproduce the paper's Eq. 3 numbers (median 1.00, max 1.56, 40 to 54%) only with the maximum readings. With the ratio's definitions I get a maximum of 1.595 and 41 to 56%. The two definitions share one symbol in Section 2. They also change classifications near the 0.5 threshold that the paper uses after the fact: the second Core Ultra 9 285K is 0.49 under one definition and 0.60 under the other.

*Remedy:* re-probe after the download, with N repetitions, and report the distribution. Define the rates once, and state which reading each equation uses.

### W3. "Registered" is a forecast log, not a confirmatory design, and Table 1's labels convey more than they deliver
There are 1,483 clauses: 33% held with an interval, 40% held on the point estimate only, and 21% failed. Predictions are written seconds before launch, by authors with full access to earlier runs on the same machines. Several thresholds are far looser than the claims later made from them. For example, the microbenchmark registered layer mode at 0.50 or more and token mode at 0.80 or more, while the text claims "nearly reachable" at 86 to 96%. The interaction clause was registered as "on every host" and failed on Pf at 11%. Table 1 reports this accurately ("sign held on 9 of 10"), but a reader sees "registered" next to the claim.

I credit the honesty of the scorecard: it is a valuable calibration record. It is not a substitute for a small set of pre-specified primary hypotheses, each with a fixed statistic, unit, population, subset and interval method. No such set is identified. The multiplicity across 1,483 clauses also means that individual "held" clauses carry little evidential weight.

### W4. External validity is narrow, and the populations behind some claims are tiny
The study uses one GPU model, two LLMs, and one prompt set (AIME-25, which was also used during development) in teacher-forced replay. The machines are a convenience sample of the cheapest rentals that pass the gates. The read-path crossover rests on 2 to 4 slow-link machines: Pf and the TR 9960X give the 0.85 to 0.93x and 1.11 to 1.22x claims, which I reproduce. Server processors mostly fail the checks (5 of 8). The "rules for system builders" should name this population ("stable consumer desktops with an RTX 5090, AIME prompts") in Section 7 itself, not only in Limitations.

### W5. The system comparison's intervals ignore the unit the paper itself establishes
Table 4's intervals are over problems within one launch on one machine. I reproduce them, for example 1.294 [1.278, 1.311]. But the host B to host S difference in the same ratios is 0.06 to 0.11, which is 4 to 6 times the half-width. By the paper's own argument (machines dominate), these intervals cannot describe the lead in general. The paper has five to seven same-protocol hosts for this comparison: B, S, the job 098 host, the slow-link hosts, and the 4090 and 3090 hosts in the supplement. It should report the lead with an interval over machines, or say plainly that Table 4 shows per-host precision only. The FreeToken backend was also selected on host B and carried over to S, which the paper discloses.

### W6. The closed-form account (Eq. 4) takes up space and inference out of proportion to its status
Eq. 4 failed or was inconclusive in all four registered tests, and the processor split that "explains" it was drawn after the tests. It is still used for Figure 7's per-machine split, for the caveat that the engine reads faster than the probe, and for the "consumer processors" population used in Section 3. I reproduce its predictions exactly; for example, the EPYC 7543 comes out at 14.9 against 22.7 ms. The exploratory label is correct, but a reader will over-weight a model that occupies most of Appendices H and I.

### W7. The audit comparison is not like-for-like
"Published systems reach a median 13.6%, ours 27%" compares two different sets. The first is 20 heterogeneous rows (systems, models, hardware), including rows below 5% that Table 28's caption says are "more likely configuration mismatches ... not adjudicated". The second is 12 cells of the authors' own two models on two hosts (`audit_ours.py`). I confirm the 13.6% median from Table 28's rows. Report the distributions side by side, and drop the rows the paper itself does not adjudicate (or report both medians).

### W8. Minor reporting gaps and inconsistencies
- Appendix D lists the rentals that produced no results ("the first two for 099a, 100d, 100e, 103c"). The ledger also has two such rentals for job 097 (offers 54239237 and 53405183) and a first 099d rental (offer 51325952). The ledger is public, but the enumeration claims completeness.
- "Ran stably" means different things across jobs. Jobs 104 and 105 ran one process per configuration and so could never fail the 2% round check, while jobs 106 to 108 could. This should be stated where the stable set (9 machines) is defined.
- The same runs carry two different numbers. Table 10 uses 1000/mean(tok/s), while Section 5 and Table 3 use mean ms per token. For host O4 at gpt-oss 11%, Table 10 prints 15.13 ms for MIN read once, while the running example says 15.2 (raw values: 15.135 and 15.163).
- At gpt-oss 25% on the TR 9960X (B_host 176 to 178 GB/s), Eq. 1 is GPU-bound at the datasheet rate: 1.84 ms, against a host term of 1.14 ms. "Host-bound on nearly every machine" is right, but this case should be named.
- R*: my sequential MIN-with-bypass replay gives 38.63 and 15.39 reads per token, against the paper's "exact" 38.32 and 15.34. The paper's no-evict count (39.03 and 15.47) matches mine exactly. The difference is in the safe direction (it makes the bound more optimistic), but the within-step ordering used by `mosl.cachesim` should be documented.

---

## Clarity

**Verdict: 3/5, at the top of the band.** The main text alone is close to a 4. What holds it back is a handful of concrete, fixable problems, plus appendices that still read like a lab notebook.

### What works
- **Claims as paragraph headings, with the evidence status in the heading:** "(registered)", "(exploratory)", "(exploratory, not confirmed)". Together with Table 1's claim, evidence and section map, a reader always knows which section supports which sentence.
- **Table 2 with a "Reads" column, and the Figure 1 schematic.** Together they make "read twice", the paper's key mechanism, concrete. The main text now uses descriptive names ("MIN, 1 read", "single read"), not codes, and Table 6 maps the codes.
- **The "Yardsticks" paragraph** declares up front that each section has its own metric. **The running example in Section 5** (10.0 / 13.0 / 20.6 / 15.2 / 13.8 ms on one machine) grounds the ratios in milliseconds.
- **Figure 2** shows one line per machine with medians and colours the two slow-link machines. It shows the heterogeneity honestly and makes "pay only together" visible at a glance.
- **Limitations are specific and quantified.** Section 7 turns the results into actionable rules.

### What still makes it hard to read (most damaging first)
1. **"The bound" is ambiguous.** The abstract and Section 3's headline (31 to 54%) use Eq. 1. Section 3 then says that Eq. 3 "and Eq. (1) holds as well, so the bound is the larger of the two", which gives 40 to 54%. Eq. 2 is "a tighter bound" for demand systems. Table 4 has a datasheet column and a measured-GPU column. The audit uses datasheet ceilings, the pooled bound appears in Table 4's caption, and the appendix tables call the bound "the limit". A reader cannot tell which number to carry away. Name one bound as primary, and give the others as named variants in a small table.
2. **The same symbols have different definitions** (W2). B_c and B_p mean one thing in the link-to-CPU ratio and another ("best rate") in Eq. 3. The paper never says which probe reading is "the link".
3. **There are four yardsticks with two baselines.** Sections 4 and 5 measure against the deployed cache; Section 6 measures against "admit every miss". Section 6 also mixes read-based shares (the lines in Figure 5) with time-based shares (the markers). These switches are declared, but each one costs the reader. Figure 5's caption should say in its first clause that the baseline changed.
4. **Table 3 mixes measured and assigned rows under a "measured" heading.** The caption sentence "the last three rows sum to *left*: T_GPU as in Eq. (2) (2.9 ms; 27–31% of the gap at 11% over the profiles' range), and the oracle's reads beyond R* at B_host" must be decoded to see that two rows are model attributions with a constant T_GPU. Split the table, or visually separate those rows.
5. **The abstract is overloaded.** One sentence carries two conditions, two sample sizes, a post-hoc flag and a further increment: "...close almost none of the gap alone but 33% together on 15 machines, and 39% on the 13 whose PCIe link reads at least half as fast as their CPU (a subset we drew after the data); copying ahead closes another 15%." Say "read twice" once and explain why, then give one number.
6. **The evidence vocabulary needs an appendix to decode.** "Registered on the panel (interaction's sign held on 9 of 10); shares over 13 after the fact", "registered on 7, held", "partly held", and "held is mostly on the point estimate" all send the reader to Table 8 and Appendix D. A two-column split, "pre-specified test (outcome)" and "post-hoc", would be clearer.
7. **Some sentences ask too much.** For example, Section 3: "These shares are relative to one probe run: its second-highest reading is a median 0.8% below the highest, and by the closed-form account of Appendix H, on the machine at the top of the range the engine's reads imply a rate above the probe's best (Table 24)." This puts three ideas into one sentence, one of them resting on a model the paper says is unconfirmed.
8. **The appendices remain a lab notebook.** Appendix D is about 1.5 pages of dense prose of job numbers and failure lists. Host codes (A, B, S, O1 to O6, Pa to Pj, "Pf again", "285K, second", "Pd again") change from table to table. A reader cannot reconstruct any one claim's evidence from the appendices without the scripts. One table per main-text claim (hosts, launches, registered clause, outcome, figure or table) would help more than more prose.
9. **The same run appears with two values** (Table 10 against Section 5; see W8), and Figure 3's 95% intervals are invisible at the plotted scale, even though the caption promises them.

---

## Questions for the authors

1. Can you re-run the bandwidth probe after the download on a few machines (say Pf, Pe and O4), with repetitions, and report how B_host, Eq. 1 and Table 3's shares move?
2. Which probe readings define B_c and B_p? Why does the ratio use the zero-copy 16 MB x 64 reading and the CPU rate at the helper count, while Eq. 3 uses maxima? With best rates the second 285K moves from 0.49 to 0.60. Do any Section 5 statements change?
3. Can the non-expert serialization be removed by an oracle, so that "most of the rest is the GPU's own work" becomes measured?
4. Why lead with the post-hoc 0.5 subset (39%) rather than all 15 machines (33%)? Were other thresholds looked at before 0.5 was chosen?
5. Which few hypotheses would you call primary and confirmatory? Could Table 1 list only those as "registered", and move the rest to "forecasts"?
6. What within-step request order does `mosl.cachesim` use for "exact" MIN? A plain sequential replay gives 0.8% more reads at C = 14.
7. Can you give the FreeToken lead with an interval over the hosts that ran the same protocol (B, S, the job 098 host, the slow-link hosts, and the 4090 and 3090 hosts)?
8. How were band widths chosen for the 1,483 clauses? How many were sign predictions, and what fraction of those held?
9. For "ran stably": would any 104 or 105 machine have failed the 2% round check had it run more than one process?

## What would raise my score

- **Soundness and methodology (to 5, and overall 8):** re-probe hygiene and a single definition of the rates (W2). Either a measured serialization term or relabeling of Table 3's remainder rows as modelled (W1). The full-sample decomposition as primary. Machine-level intervals for the system lead (W5). A short "confirmatory core" table separate from the forecast log (W3).
- **Clarity (to 4):** name one bound and tabulate its variants. One definition per symbol. Flag the baseline switch in Section 6 and Figure 5. Split Table 3 into measured and assigned rows. Cut the abstract's densest sentence in half. Turn Appendix D into per-claim evidence tables with stable host names.
- **Significance (to 4):** one more GPU class or one more workload (non-AIME, or real generation rather than teacher forcing) to show that the decomposition and the read-path rule travel.

## Scores

| Criterion | Score |
|---|---|
| Overall (1–10; 6 = acceptable with revisions, 8 = strong) | **7** |
| Soundness (1–5) | **4** |
| Methodology (1–5) | **4** |
| Significance (1–5) | **3** |
| Clarity (1–5) | **3** |
| Confidence (1–5) | **4** |

The work is a careful, unusually transparent benchmarking study whose numbers survive an independent recomputation from raw data. It falls short of "strong" because:
- the headline decomposition's remainder is assigned rather than measured, and its lead summary is post hoc;
- the yardstick rests on a single probe taken under contention;
- "registered" is used for a forecast log that is not a confirmatory design;
- the generality is narrow.

None of these is hidden; each is fixable.

---

## Claims checked against raw data

Re-derived with my own code from the raw files named in the materials list. "Repro" means it matches to the printed precision, or within bootstrap noise. Notes give my values.

| # | Claim (location) | Paper | My recomputation (source) | Verdict |
|---|---|---|---|---|
| 1 | R*, MIN with bypass, gpt-oss C = 14 and 32 (Sec. 3, Table 25) | exact 38.32 and 15.34; no-evict 39.03 and 15.47 | My MIN replay on `route_aime25_gptoss.npz`: no-evict **39.025 and 15.466 (exact match)**; sequential exact 38.63 and 15.39 (paper 0.8% and 0.3% lower, the safe direction) | Repro (no-evict); small documented-order question (exact) |
| 2 | R* on the first 20 problems changes by -0.4% (Sec. 9) | -0.4% | -0.39% (same replay) | Repro |
| 3 | Table 3, gpt-oss 11%: set alone, once alone, together, ahead, left, T_GPU | 1 [-1,3]; 1 [0,2]; 39 [32,45]; 15 [11,18]; 47 [42,51]; 27 [24,29] | 1.2 [-0.7,3.0]; 0.9 [-0.2,2.0]; 38.8 [31.8,45.3]; 14.6 [10.6,18.3]; 46.5 [42.1,51.0]; 26.5 [23.8,29.1] (`ec_*.jsonl`, `concur.txt`) | Repro |
| 4 | Table 3, gpt-oss 25% | 21; -1; 35; 20; 44; 33 (with intervals) | 20.8 [19.1,22.3]; -1.1 [-4.0,1.3]; 35.3 [30.0,40.3]; 20.5 [18.2,22.4]; 44.3 [40.1,48.3]; 32.9 [30.0,35.5] | Repro |
| 5 | 15 machines ran every state (one launch per GPU), 13 at ratio 0.5 or more; 33% together on all 15 (abstract) | 15 and 13; 33% | 15 and 13 (3 duplicate-GPU launches dropped); 32.6% [21.4, 42.0] | Repro |
| 6 | Best oracle closes 54% (49–58) at ratio 0.5 or more; at most 17% on the two slow machines (Sec. 4) | 54 (49–58); 17% or less | 53.8 [49.3, 58.0]; 16.8% and 1.2% | Repro |
| 7 | Prefetching oracle at 1.07–1.38x Eq. 2 at 11% (Sec. 4) | 1.07–1.38 | 1.069–1.386 | Repro |
| 8 | Interaction sign held on 9 of 10 panel machines (Table 1) | 9/10 | 9/10 at 11% (Pf negative); 10/10 at 25% | Repro |
| 9 | Prefetching oracle reads 1.22–1.31x R* (1.42–1.43 at 25%) (Sec. 4) | as stated | 1.22–1.31; 1.42–1.43 (`st_g_C*_both3p.json`) | Repro |
| 10 | Robustness of "together" (reviewer's own check) | – | Ratio of summed ms: 40.1% / 36.2%; without the 30-problem launches: 38.9% / 35.2% | Robust |
| 11 | Microbenchmark reaches 86–96% of the read term; two repetitions within 0.9% (Table 26) | all 24 cells | All 24 cells identical; maximum repetition difference 0.88% (`readsched_C*.txt`, `concur.txt`) | Repro |
| 12 | Table 4, host B: 69.9 / 54.0 / 34.9 tok/s; 1.294 [1.278, 1.312]; 1.275; 1.154; 1.032 | as stated | 69.88 / 53.99 / 34.93; 1.294 [1.278, 1.311]; 1.275 [1.255, 1.295]; 1.154 [1.134, 1.172]; 1.032 [1.022, 1.042] (`081/bs1.jsonl`) | Repro |
| 13 | Table 4, host S ratios 1.207, 1.196, 1.093, 1.027, 1.048, 0.974; leads FreeToken at 11 of 12 | as stated | Same from `089/bs1.jsonl` (launch 2); 11 of 12 | Repro |
| 14 | Ratio of mean rates against total time agree within 0.003 (App. B, rule 3) | 0.003 or less | Maximum difference 0.0019 | Repro |
| 15 | Table 4 bound, host B: 172 / 430 / 519 tok/s | as stated | 172.3 / 430 / 519 from R*, S, D, B_host 87.5 and datasheet B_gpu | Repro |
| 16 | Fewest-admission read once at 11%: 1.21x [1.13, 1.30] on 9 stable machines; 1.17x on 13; lost on the 2 slowest links (Sec. 5) | as stated | 1.215 [1.13, 1.30]; 1.166; losses EPYC 7302 0.845 and EPYC 7663 0.974 (jobs 104–107) | Repro |
| 17 | Spearman (ratio, MIN read-once gain) 0.89 [0.63, 0.97], 19 machines | as stated | 0.881 [0.62, 0.97], 19 machines (0.907 with the ratio of maximum readings) | Repro (within noise) |
| 18 | Machines 96% of log-gain variance, problems at most 3%; Kendall W 0.77 / 0.82 | as stated | 96.3% / 1.1% (11%), 96.2% / 2.8% (25%); W 0.770 / 0.817 | Repro |
| 19 | At ratio 0.28–0.32 greedy read once loses 0.85–0.93x; fewest-admission by CPU 1.11–1.22x | as stated | 0.853–0.933; 1.112–1.217 (Pf, TR 9960X) | Repro |
| 20 | Fig. 3: MIN 1 read +16–51% at every budget on O3–O5; up to +81% ahead; the usual ways at most +15% at the smallest budgets | as stated | 1.160–1.514; 1.812; maximum 1.148 | Repro |
| 21 | Running example, 9950X at 11%: Eq. 1 10.0, Eq. 2 13.0, deployed 20.6, MIN 1 read 15.2, ahead 13.8 ms | as stated | 10.02 / 12.96 / 20.64 / 15.16 / 13.81 (096a). Table 10 prints 20.59 / 15.13 / 13.78 for the same runs (harmonic) | Repro; inconsistent with Table 10 |
| 22 | Same-CPU panel machines differ up to 29%; a relaunch within 3.2% (Sec. 2) | 29%; 3.2% | 29.0% (285K Pa against Pf); 3.2% (TR 9960X, 105b against 106a) | Repro |
| 23 | Admission margin 1.021x at 11%, 1.018x at 25% on 5 stable machines; per machine 1.020 (1.011–1.030) | as stated | 1.0214; 1.0176; 1.020 [1.012, 1.030] | Repro |
| 24 | Exact 16-token window recovers 0.78–0.93 of the gain in time at 11%; the lowest machine missed 0.80 | as stated | 0.781–0.935; Pf 0.781 | Repro |
| 25 | Our cache at 31–54% of Eq. 1 and 55–70% of Eq. 2 on 25 consumer machines at 11% (Sec. 3) | as stated | 25 machines / 39 launches; 30.6–54.0%; 55.2–70.3% | Repro |
| 26 | Eq. 3 / Eq. 1 median 1.00, up to 1.56; 40–54% of Eq. 3 | as stated | With maximum readings: 1.00, 1.557, 40.5–54.0%. With the ratio's own B_c and B_p: max 1.595, 40.5–55.8% | Repro only under "best rate"; definitions differ (W2) |
| 27 | Probe's second-highest reading a median 0.8% below the highest; more than 5% only on 107d (26%) | as stated | 0.84% over 48 launches; only 107d, 26.0% | Repro |
| 28 | T_GPU 2.9 ms (smallest profile); G 4.1–4.7 ms; median profile 4.28 / 4.50 ms | as stated | 2.94; 4.07–4.69; 4.28 / 4.50 (`prof_C*.json`, 14 profiles) | Repro |
| 29 | Job 105 plain form: 5/8 cells within 6%, median 4.2%; over-prediction at 25% of 6.9–9.3% on 3 hosts | as stated | 5/8; 4.24%; +6.9, +7.0, +9.3% (Pf +3.8%) | Repro |
| 30 | Tables 21–23, Eq. 4 against measured (for example EPYC 7543 14.9 against 22.7 ms, -34.3%, implied 0.58; AMD ES 8.2 against 20.0) | as stated | All gpt-oss cells within 0.1 ms; EPYC 7543 -34.3%, 0.58 | Repro |
| 31 | Validity gates: 107: 7 rented, 3 stopped at the gate, 2 failed rounds (3.5%, 6.0%); 108: 2 at the gate, 1 failed rounds (7.0%); Xeon 8347C 52% apart; EPYC 7302 loss 0.21–0.45 | as stated | `v0.txt` and `validity.txt` agree; rounds 3.54 / 5.98 / 7.01 / 52.2%; EPYC 7302 NLL 0.279 / 0.420 against 0.19 | Repro |
| 32 | Registration timing: commit at least 2.7 s before rental; job at least 26 s after push; 7 within 1 s; pushes up to 0.8 s after rental | as stated | 2.7 s; 26 s; 7; 108a +0.8 s, 108f +0.5 s. Authors' 258 push events are identical to the GitHub Activity API; no force pushes | Repro |
| 33 | Post-launch header edits: 100 (ninth prediction), 102 / 105 / 106 (host substitutions, predictions unchanged), 107 (amendments before f and g) | as stated | Diffs confirm. 107 amendments committed 3 s and 4 s before 107f and 107g started | Repro |
| 34 | Table 8 totals 495 / 597 / 308 / 83; per-job counts 099–108 | as stated | Same from `tab_prereg.tex` sums and `scorecard_*.json` | Repro (internal) |
| 35 | 124 rentals, 74 offers, $87.22; no-result rentals enumerated (App. D) | as stated | 124 / 74 / $87.22. **Two 097 rentals and a first 099d rental with no results are not in App. D's list** | Partly; minor omission |
| 36 | KL parity: 0.0019 / 0.0005 nats, top-1 98.5% (App. J) | as stated | `parity_kl.json` 0.00186 / 0.00053, 0.985 (on-machine summary, not raw logits) | Repro (summary level) |
| 37 | Audit in-class median 13.6% over 20 rows (Sec. 3) | 13.6% | 13.6 from Table 28's rows (trace rows less Fate, SP-MoE and MoE-SpeQ) | Repro (internal) |
| 38 | Distinct-expert horizon on the AIME routing 0.66–0.81 C (Sec. 6) | 0.66–0.81 | gpt-oss: 0.80 C (11%) and 0.68 C (25%), from Table 16's W50 and my window count on the trace | Consistent |

Overall: 36 of 38 checks reproduce as printed. One reproduces only under a definition the paper does not state (#26). One shows a minor omission in an enumeration (#35). No check contradicts a claim.
