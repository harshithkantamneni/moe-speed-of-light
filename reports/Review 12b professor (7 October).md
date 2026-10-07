# Review 12b: "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth"

Referee stance: a systems professor reviewing for a fellowship or workshop, applying the standards of Hoefler and Belli (SC'15). Date: 7 October 2026.

## Materials read

- `paper/paper.pdf` (37 pages; built 7 Oct 2026 02:12 CDT): the whole main text, Appendices A–M (checklist, prediction log, calibrated model, grid, accounting, oracles, trace study, partial foresight, MIN's schedules, cache detail, audit rows, artifact), and Tables 1–24.
- `paper/supplement.pdf`: the scorecard tables for jobs 105, 106 and 107 (Tables 8–10). Other job tables were skimmed only.
- Job scripts on the GPU branch: the headers (registered predictions, gates, amendments) of `jobs/105_sumlaw@vast.sh`, `jobs/106_onlineadmit@vast.sh`, `jobs/107_newhosts@vast.sh` and their per-host stubs. I also read the code of the 107 script (gates V0/V1/V2, run loop, `gprof`, `vcheck`).
- Git history of `/home/claude/gpu-branch`: commit hashes and dates only (`--format='%h %cd'`), plus file diffs between the registration commits and their amendments. I read no commit messages.
- GitHub's own push-activity log for the `gpu` branch (`gh api /repos/.../activity`). I printed only the SHA, timestamp and event type.
- `gpu/vast_ledger.json`: rental start and stop times.
- Raw results in `/home/claude/gpu-branch/results/`:
  - `ec_*.jsonl` per-problem rows, `st_*.json` counters, `concur.txt` probes, `g_prof.json`/`q_prof.json`/`prof_*.json` Nsight summaries;
  - `v0.txt`, `validity.txt`, `free.txt`, `numa.txt`, `cores.txt`, `cpu.txt`, `nvidia-smi-q.txt`;
  - for jobs 093–107, the headline job 081 (`bs1.jsonl`), the read-schedule job 102 (`readsched_*.txt`, `rates.txt`) and the parity job 090 (`parity_kl*.json`).
- Authors' code, read only to learn definitions after computing my own values:
  - `scripts/job107.py` (first 200 lines);
  - `scripts/job106.py` (grep for `law`, `boot_rounds` and ratio definitions);
  - `scripts/panel_099.py::host_info`, `scripts/speed_limit.py::host_rates`, `jobs/ec2/fetch_table.py::bandwidths`.
- Authors' derived files, used only for counting: `prereg/reanalysis.json` (the median G constants) and `prereg/scorecard.json`, `scorecard_105/106/107.json` (clause status counts).

## Independence statement

- I did not open anything under `reports/` except to write this file.
- I did not open:
  - any `prereg/*outcome*.md`;
  - `research_notes/`;
  - `apply/`;
  - `paper/paper_v1_prereview.tex`;
  - any file named like a review, number check, plan or progress log.
- I read no commit messages.
- Every number in the claims table comes from my own code, written from the raw rows. The code is in `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/rev12b/`: `common.py`, `eq3_105_107.py`, `classes.py`, `checks2.py`, `checks3.py`, `sens.py`, `rounds.py`, `spearman.py`, `oracle.py`, `table2.py`, `ledger.py`.
- I consulted the authors' scripts only for definitions: which probe readings make up B_host, B_c and B_p, and the median G. I did that after computing my own values, to explain two small discrepancies.
- I modified nothing in either repository and committed nothing.

---

## 1. Summary

The paper studies batch-1 decode of two offloaded MoE models (gpt-oss-120b MXFP4 and Qwen3-30B-A3B BF16) on rented RTX 5090 hosts. Experts live in host DRAM and C per layer are cached on the GPU, in a 3,500-line llama.cpp patch. It makes four contributions.

1. **Two lower bounds.** Both use MIN-with-bypass host reads and the probed host rate:
   - Eq. (1): fully overlapped;
   - Eq. (2): reads in series with the GPU's non-expert work, for read-on-demand systems.

   A microbenchmark reaches 86–96% of the read term.
2. **A time relation, Eq. (3).** T = G + (M + (1−G/T)A)·S/B_host, with "nothing fitted":
   - G is profiled with Nsight;
   - M and A are the run's own miss and admission counters;
   - B_host is the probe's highest reading.

   It was explored on jobs 093–105, registered and tested in job 106 (held on the 3 previously rented stable hosts, failed on 2 new servers), and registered again in job 107 on never-rented machines with validity gates. There it failed: −7% to −11% on a new desktop and −52% to −59% on a dual-socket server. A post-hoc "desktop-class" line (one NUMA node, ≤32 cores) separates the small errors from the large ones.
3. **An in-engine oracle study of foresight.**
   - MIN with one read gains 16–51% over the deployed cache.
   - Among MIN's hit-optimal schedules, the fewest-admission one beats the greedy one on slow links.
   - An online "admit less" rule (larger κ) gains a median 1.02×.
4. **A price on foresight.** The window W50 that closes half the read gap corresponds to about 0.65·C distinct experts per layer, and realisable forecasters fall far short.

The artifact is extensive: a predictions-in-script-header registration practice, a rental ledger, a 565+ clause scorecard and per-problem raw rows.

## 2. Strengths

- **S1. The registration is real and verifiable against an external clock.**
  - For every host of jobs 105, 106 and 107, the ledger start time follows both the commit time of the job header and **GitHub's own recorded push time** for that commit. Start minus push ranged from +0.1 s to +12.4 s; 107f and 107g are effectively ties, but only the host identity changed and both stopped at V0.
  - Every amendment I diffed only substituted hosts or offers; predictions, gates and code were unchanged:
    - 105: 92ff9fd→a8e2f92;
    - 106: 525a9c9→89c5c3b;
    - 107: 1a0616a→9f01f98→f9ac8c7→ddec94f→346bd34.

  This is better registration hygiene than almost any systems paper I referee.
- **S2. The numbers reproduce.** I recomputed 37 quantitative claims from raw per-problem rows, counters and probes (table in Section 8). Thirty-one match to rounding. Four match approximately or once a stated method choice is known. Two are worded loosely: the "16–29%" same-model range and the round-spread definition. I found no fabricated or unsupported number.
- **S3. Failures are reported, including against interest.**
  - Job 107 is called "failed" even though its own registered rule (fewer than 3 valid hosts) would allow "inconclusive".
  - All three V0 stops and both V2 failures are reported with what they ran; Table 20 includes the V2-failed hosts with a dagger.
  - Every clause of every scored job is in the supplement, with 142 failures among the 565 for jobs 073–098.
  - The overlap term's weak specific support (half-discount and misses-only forms do about as well) is stated in the paper.
- **S4. Job 107's validity gates are well designed and registered before launch.**
  - V0 requires a GPU UUID not seen before (I confirmed none of 107b–e is in the committed `known_gpu_uuids.txt`) and at most 48 GB of host memory in use.
  - V1 is a correctness check on teacher-forced loss.
  - V2 caps round-to-round spread.

  The thresholds are motivated by the job-106 failures, which is the right way to learn.
- **S5. The oracle-in-engine methodology is genuinely informative.** The observation that *which* of MIN's equally hit-optimal schedules is followed decides whether foresight pays on slow links is well supported. The fewest-admission set gains on 10 of 10 launches; the greedy set loses at link/CPU ratios ≤0.32. The cross-machine dependence on the link/CPU ratio (Spearman ≈0.87–0.89 over 19 machines) is a useful, generalisable finding.
- **S6. Within-host comparisons follow Hoefler–Belli practice.** They use:
  - arithmetic means of time;
  - ratios of means paired by problem;
  - bootstrap percentile intervals;
  - absolute speeds next to every ratio (Table 2);
  - a bound column;
  - a cost ledger.

## 3. Weaknesses (most important first)

### W1. The abstract and contributions describe the relation's registered record more favourably than the registered outcomes support

Each registered test of the time relation, with its verdict as the job header states the rule:

| Job | Form registered | Hosts started → analysed | Registered verdict | What the abstract and contributions say |
|---|---|---|---|---|
| 105 | plain, B_host max | 7 → 4 (2 gated, 1 never started) | **Failed.** Pooled median 4.2% > 4%; 3 of 8 cells > 6% (all at 25%). Recomputed: 105a +9.3%, 105b +7.0%, 105f +6.9%. | "failed its registered test there" (body only) |
| 106 | overlap | 5 → 5 (no registered gates) | **Failed.** P2 required "within 6% … at every host"; 8 cells on 106c and 106d miss by 10–52%; the pooled median is 4.67% > 4%. | "the relation held when registered before new runs of 3 of them" |
| 107 | overlap | 7 → 2 valid | **Inconclusive** by its own rule (fewer than 3 valid hosts). Per-cell: 7 of 8 beyond 6%; pooled 31.8%; overlap-vs-plain also failed. | "it failed" |

- **Job 106 had no registered validity gates.** Round spread was *prediction 8*, not an exclusion rule. Wrong outputs (106c) are a defensible post-hoc exclusion. 106d (Xeon 8347C), however:
  - computed correct outputs (loss 0.1901 and 0.0856, matching every other host);
  - is under-predicted by 17% even in its *best* gpt-oss 11% round (14.52 ms measured against 12.03 ms by the relation).

  So "held when registered" describes a post-hoc subset of a registered test that failed as written.
- The phrase "new runs of 3 of them" (3 of the 23 machines where the relation was *found*) hides that 5 machines were in the registered test.
- **No machine new to a registered relation test passed every cell under the form registered at the time:**
  - 105a and 105b failed at 25% (plain form);
  - 106c and 106d failed;
  - 107b failed at 3 of 4 cells and 107d at 4 of 4;
  - 107c and 107e were invalid but missed by 31% and 21%.

  The clean passes are on Pf, Pe and the TR 9960X, machines the relation was developed on. At the machine level the registered record is 3 of 5 in job 106 and 0 of 2 valid in job 107.

The body is honest about each of these facts: the 4.7% against 4%, the server under-prediction, and that 107 is inconclusive by rule. The headline framing is not. The paper also applies its vocabulary unevenly: 107 is held to its registered rule (and then called "failed", conservatively), while 106 is reported on an unregistered subset and called "held". The abstract and contributions should lead with the registered verdicts and present the 3-host result as what it is: a reproduction on development machines.

### W2. A post-hoc machine class does too much work in the framing

- **The line was drawn after the data were in.** "Desktop-class" (one NUMA node and ≤32 usable cores) was drawn after job 107 from 5 server launches, 4 of which failed a validity check. The threshold has degrees of freedom:
  - 32 cores separates the TR 3970X (32 cores, −7.1%) from the Xeon 8347C (36 cores);
  - the NUMA criterion is what removes the 15-core EPYC 7302.
- **The name misleads.** "Desktop-class" includes an EPYC 7402P server; "single-NUMA, ≤32-core hosts" would be honest.
- **The framing leans on the class without flagging it as post hoc.** The abstract's first quantitative claim ("On the 23 rented RTX 5090 machines, each with one NUMA node and at most 32 cores, on which we found it … within 6% on 32 of 37 launches") and the Conclusion's first claim are both stated on this class, with no "post hoc" flag. Section 4's "What the failures share", the Section 2 "Terms" note and the Limitations do flag it, which I appreciate.
- **Inside the class, the one prospective test failed.** The single prospectively tested new member, the Ryzen 9 5900XT, failed at 3 of 4 cells (−7.1%, −2.3%, −11.3%, −6.4%). The class is therefore a hypothesis, not a scope on which the relation has been confirmed.
- **The 32-of-37 count mixes evidence types.** It combines 34 exploratory launches (jobs 093–105) with 3 confirmatory ones. One Core Ultra 9 285K (Pf) contributes 6 of the 37 launches. The machine-level count is 19 of 23 machines within 6% on every launch.

### W3. The central relation is tested without any uncertainty, and its ±6% verdicts lie within the measurement uncertainty of its inputs

- **No interval on the error.** Every "within 6%" clause is scored "held (point)" with no interval. Yet T, G and B_host all carry sampling and measurement error. In the paper's own Table 21, the median probe sample instead of the highest moves the host-bound limits by 8%.
- **The verdicts depend on which probe statistic is used.** I recomputed Eq. (3) with two defensible alternatives to "the highest of about 20 probe readings across methods":

  | Hosts | Highest reading (registered) | Median concurrent CPU+link (zero-copy) | Highest CPU-only |
  |---|---|---|---|
  | Job 106 stable hosts | 12 of 12 within 6% | 8 of 12 | 9 of 12 |
  | New desktop 107b | 1 of 4 | 3 of 4 | 3 of 4 |

  Pass and fail on the desktops hinge on the probe statistic. Only the server failure (−51% even at the server's best CPU-only rate) is robust.
- **The max-of-readings statistic is not a true peak.** By the paper's own accounting, the engine read more than 5% faster than "the machine's best rate" on 13 of 68 launch-budgets (11 with per-launch G), up to 1.22× on 100f. That also undermines calling Eq. (1)/(2) a *bound* on those machines (Hoefler–Belli rule 11 asks for a true upper bound). Either use a defensible peak (datasheet or channel bandwidth) for the bound, or call it a "probe-referenced limit".
- **The residuals have structure.** Over the 37 desktop launches of jobs 093–106:
  - at gpt-oss 11% the median error is −1.2%, and 78% of launches are under-predicted;
  - at 25% the median is +1.8%, and 73% are over-predicted.

  A budget-dependent sign flip is a missing term, not noise. It is consistent with the overlap form being *worse* than the plain form at 11% (32 against 35 of 37 within 6%), which the Limitations mention only briefly.

### W4. The statistical unit and the intervals do not match the paper's own stated unit

- **The stated unit is not the counted unit.** Section 2 says "the machine is therefore the unit". The headline counts are over launches (32/37) or over cells: 12/12 cells on 3 machines, and pooled medians over machine × model × budget cells. Report machine-level tallies alongside.
- **Two or three rounds cannot support a 95% interval.** The "round-level 95% intervals (rounds, then problems)" are built from 2–3 rounds inside a single rental:
  - with 3 rounds there are only 10 distinct round resamples, and with 2 only 3, so no meaningful 95% interval over rounds exists;
  - the rounds do not sample the relaunch variance that matters. On the same machines the deployed time moved by up to 3.2% between rentals, against 0.03–0.9% between rounds within a rental.

  Report the per-round values, which are tight and convincing (for example fetchplan on Pf: 1.018, 1.022, 1.018), and use relaunches for intervals.
- **Problems are not independent.** The cache carries over between problems in a fixed sequence, so per-problem times are serially dependent. A problem-level iid bootstrap understates uncertainty somewhat; a block or sequence-aware bootstrap, or a check, is warranted.
- **Rounding a lower bound up to the threshold.** In Table 3, Pf's "dk" interval at gpt-oss 11% is printed as "[1.01, 1.04]", but the computed bound is 1.0051 (scorecard clause 106b-P3-speed-g11, scored "held (point)"). Printing 1.01 next to a registered threshold of 1.01 makes a point-only success look like an interval success. Round lower bounds down, or give three decimals.

### W5. Clarity

- The paper is very dense: 37 pages, a main text that reads as a sequence of job numbers and numeric ranges, and an abstract with roughly 20 numbers. It mixes exploratory, registered and post-hoc results in single sentences.
- Terms such as host-bound, deployed path, fewest-admission, single read and launch versus round are defined in passing.
- A reader cannot tell at a glance which claims are confirmatory.
- A one-page "ledger of registered tests" in the main text would fix most of this: job, hypothesis, hosts launched/valid, registered verdict, post-hoc notes.
- The foresight findings, arguably the most generalisable part, are crowded out by the relation's bookkeeping.

### W6. Significance of the relation and the bounds is modest; the online gain is small and its threshold moved

- **The relation is closer to accounting than prediction.** Eq. (3) uses each run's own M and A counters. Its predictive content is "the engine reads at the probe's rate, in series with G": a roofline-style serial sum. The bounds are Belady reads × bandwidth.
- **The online gain is small and the thresholds moved:**
  - the online "admit less" result is a median 1.02× (range 0.987–1.057 over 10 cells);
  - the registered thresholds failed in job 106 (≥1.01 per gpt-oss cell, short on 2 of 3 stable hosts);
  - in job 107 the per-cell threshold was relaxed from ≥1.01 to ≥0.995 (and the median from ≥1.02 to ≥1.01), and still failed at one cell (0.987);
  - the text does not say that the threshold was lowered between registrations.
- **The bigger contributions are elsewhere.** They are:
  - the engine (2–4× stock llama.cpp, ahead of FreeToken at 11 of 12 configurations);
  - the schedule-choice finding;
  - the distinct-expert horizon rule.

  The positioning should follow that.

### W7. Minor inconsistencies (do not affect conclusions)

- **Round spread is defined two ways.** "Varied by 39%" for 106d is (max−min)/mean. Under job 107's registered V2 definition (max/min − 1) it is **52%**. Use one definition.
- **G is chosen differently across analyses:**
  - Tables 18–20 use per-host profiles;
  - Fig. 1, "13 of 68" and the Limitations use the reanalysis median (4.28/4.50 ms).

  With per-host G, the server's "third part" is 58.9% and 52.2% rather than 58% and 50%, and the >5% count is 11 rather than 13.
- **The same-model range is loose.** "Two hosts with the same CPU model differ by 16–29%" covers the largest spreads only. Same-model machine pairs range from 0.1% to 29%; the two 9800X3D machines differ by 1.6%.
- **A borderline count.** "33 instead of 18 within 6%" at 25% (plain form, 34 launches): I get 33 against 19. 099f's plain error is 5.99%, a rounding boundary.
- **A misleading directory name.** The parity job directory is named `…a100` but ran on an RTX PRO 6000 (the paper text is correct).
- **No interval on a "won" claim.** In job 107, "copying in the step won on both" on the server rests on 1.157 against 1.126, with overlapping per-arm intervals and no interval on the paired difference.

## 4. Specific checks of the methodology questions I was asked to examine

- **Statistical unit.** See W4. Within-host claims are paired by problem and sound. Cross-machine claims are stated as counts over launches and cells. The "machine is the unit" principle is not carried through to the headline numbers.
- **Registration practice, jobs 105–107.** Verified; see S1. The commit-to-start and push-to-start orderings hold for all 19 hosts.
  - Amendments were made before each substituted host started and changed only host identity.
  - One caveat: the job-107 amendment that added *replacement of V2-failed hosts* was committed 35 s after 107c/107d's full results were committed. The authors therefore knew the relation had failed badly on 107d.
  - Predictions were unchanged, and both added hosts failed V0, so this had no consequence. It is nonetheless a protocol change made after unblinding, and the text should say so.
- **Validity gates of job 107 and how failing hosts are reported.** The gates are well designed and were applied as registered:
  - V0 (memory in use): 50, 50 and 96 GB for 107a, 107f and 107g;
  - V2 (round spread): 3.54% and 5.98% for 107c and 107e;
  - V1 (loss): passed everywhere.

  The failing hosts are reported in text, in Table 20 with daggers, and in the server range "0.27–0.77 on all 5 launches". Including them would only strengthen the "relation fails on servers" finding, so the gates are not self-serving here.
- **Separation of the post-hoc machine-class analysis.** It is labelled "found after job 107 and not tested in advance" in Section 4, Section 2 and the Limitations, but it is unlabelled where it matters most: the abstract's first claim and the Conclusion. See W2.
- **Do the paper's claims about Eq. (3) match the registered outcomes?** In the body, yes, including the 4.7% against 4% and the inconclusive-or-failed status of job 107. In the abstract and contributions, no, for job 106. See W1.

## 5. Questions for the authors

1. **Job 106 verdict.** Would you accept rewording the abstract to "registered on five machines in job 106, it failed as registered (pooled median 4.7% against 4%); on the three machines it was developed on it held at 12 of 12 cells"? If not, which pre-launch rule justifies excluding 106d, whose outputs were correct and whose best round is still under-predicted by 17%?
2. **Probe repeatability.** How repeatable is B_host when the probe is rerun on the same host minutes apart? What are the relation's errors, with intervals, when B_host is taken as a median or as a repeat-probe statistic rather than the maximum over methods?
3. **The class threshold.** Why 32 cores? Would you register the class hypothesis prospectively, with ≥5 new single-NUMA hosts of ≤32 cores and ≥3 servers under the 107 gates?
4. **Spearman launch choice.** For the Spearman 0.89 [0.63, 0.97] over 19 machines, which launch per machine did you use? Taking each machine's first launch through job 104, I get 0.87 [0.58, 0.97].
5. **Round intervals.** Why bootstrap over 2–3 rounds rather than report the per-round ratios and use relaunches for between-launch variance?
6. **Problem order.** Is the problem order fixed within a process? Have you checked the problem-level intervals against a block bootstrap, given the cache carry-over?
7. **The dk threshold.** Please state in the text that the dk threshold was lowered (≥1.01 → ≥0.995 per cell; ≥1.02 → ≥1.01 pooled) between jobs 106 and 107, and why.
8. **Which form to keep.** The half-discount form had a lower median error than the registered overlap form on job 106's stable hosts (1.9% against 2.3%), and the plain form beats the overlap form at gpt-oss 11% on the development launches (35 against 32 of 37). What, beyond the mechanism argument, favours keeping the overlap term?

## 6. What would raise my score

1. **Lead with the registered verdicts.** Rewrite the abstract and contributions to give the registered verdict of each relation test (105 failed; 106 failed as registered and held on 3 development hosts; 107 inconclusive by rule and failed per cell) before any post-hoc subset or class. Add a main-text ledger table of registered tests.
2. **Put uncertainty on the relation's errors.** Use problem, relaunch and probe-repeat variability. Score the ±6% clauses against intervals. Report the sensitivity to the B_host statistic and to per-host versus median G.
3. **Fix the bound's peak.** Either base the bound on a true peak rate, or rename it a probe-referenced limit and quantify how often it is exceeded.
4. **Rename the class and test it prospectively.** Rename "desktop-class", flag it as post hoc wherever it is used (abstract and Conclusion included), and ideally run one prospective, gated test of it.
5. **Report at the stated unit.** Report machine-level summaries throughout. Replace 2–3-round bootstrap intervals by per-round values plus relaunch-based intervals.
6. **Restructure for readability.** Move job-by-job narrative to the appendices, foreground the foresight and schedule findings, and cut the abstract's numbers by half.

With items 1, 2 and 5 done I would move to a 7. With a successful prospective test of the class hypothesis (item 4), to an 8.

## 7. Scores

| Criterion | Score |
|---|---|
| Overall (1–10; 6 = acceptable with revisions, 8 = strong) | **6** |
| Soundness (1–5) | **3**: the numbers are correct and reproducible, but the headline framing of the relation outruns its registered record, and its verdicts carry no uncertainty |
| Methodology (1–5) | **4**: the registration, gates, ledger and artifact are exemplary; the unit of analysis, round-level intervals and probe-statistic sensitivity are the gaps |
| Significance (1–5) | **3** |
| Clarity (1–5) | **2** |
| Confidence (1–5) | **4** |

## 8. Claims checked against raw data

"My value" is computed by my own scripts from `results/<job>/ec_*.jsonl` (decode_ms/n_decode, nll_sum/nll_n), `st_*.json` (misses, admits, fetches, steps), `concur.txt`, `g_prof.json`/`q_prof.json`, and the other raw files listed.

Conventions used in the table:
- Eq. (3) uses S = 13,253,760 B (gpt-oss) and 9,437,184 B (Qwen3), B_host = max of all probe readings, and G = per-host profile unless noted.
- Ratios are ratios of mean times, paired by problem.
- Verdicts:
  - ✓: matches to rounding;
  - ≈: matches approximately, or once a method choice is known;
  - ⚠: wording or definition issue;
  - ✗: does not match. I found none.

| # | Claim (paper) | Location | My value | Verdict |
|---|---|---|---|---|
| 1 | Job 106 stable hosts: Eq. (3) within 6% on all 12 cells, median error 2.3%, largest 6.0% | §4, Table 3 | 12/12; median 2.34%; max 5.99% (106a Qwen3 25%) | ✓ (max is 0.01 pp inside the band) |
| 2 | Plain form missed by up to 10.1% on those cells | §4 | 10.08% (106a Qwen3 25%) | ✓ |
| 3 | Job 106 registered pooled median error 4.7% against 4% registered | §4, App. B | 4.67% over all 20 cells | ✓ (registered P2 failed) |
| 4 | 106c and 106d under-predicted by 10–52% | §4, App. J | 106c −16.9 to −30.9%; 106d −10.2 to −52.4% | ✓ |
| 5 | 106c teacher-forced loss 0.21–0.45 against 0.19 and 0.085 elsewhere | §4, App. J | 0.208, 0.208, 0.420 (g11 rounds), 0.420 (g25), 0.446 (q16), 0.446 (q32); others 0.1893–0.1907 and 0.0854–0.0856 | ✓ |
| 6 | 106d deployed cache "varied by 39% between rounds" | §4, App. J, supplement | rounds 14.52, 22.10, 21.93 ms: (max−min)/mean = 38.8%; max/min−1 = **52.2%** (107's registered V2 definition) | ⚠ definition inconsistent with 107's V2 |
| 7 | 106c/106d had 131–132 GB of host memory in use at the gate (12–15 GB elsewhere) | App. J | 132, 131; others 12, 15, 12 | ✓ |
| 8 | Half-discount passes 11 of 12 stable cells, misses-only 10 | §4 | 11/12, 10/12 (medians 1.88% and 4.02%) | ✓ |
| 9 | Job 107: 7 rented, 3 stopped at V0 with 50–96 GB in use, 2 ran rounds 3.5–6.0% apart | §4, Table 20 | V0: 50, 50, 96 GB; V2: 3.54% (107c), 5.98% (107e); V1 passed on 107b, 107d (0.1899–0.1903, 0.0854–0.0856) | ✓ |
| 10 | 107: under-predicted 8 of 8 cells, 7 beyond 6%; 5900XT up to 11% (1 of 4 within); server 52–59% | Abstract, §4 | 107b −7.15, −2.28, −11.27, −6.42%; 107d −58.95, −52.22, −58.25, −56.31% | ✓ |
| 11 | 107 pooled median; "could not have come below 6.8%" with any third host | §4 | pooled 31.75%; with 4 zero-error cells added 6.78% | ✓ |
| 12 | On the desktop the plain form is closer: median 6.0% with 2 within, against 6.8% with 1 | §4 | plain 5.99% with 2/4 within; overlap 6.78% with 1/4 | ✓ |
| 13 | Server: every form misses by 56–58%; against the 151 GB/s CPU-only rate still 51% | §4 | medians: overlap 57.3, plain 56.3, half 57.3, misses-only 58.2%; at 151.4 GB/s: −51.4% | ✓ |
| 14 | Server's implied read rate 0.27 of B_host; probe reads 118 GB/s at 128 threads against 151 at 16 | Abstract, §4 | 0.268; concur.txt 117.6 @128, 151.4 @16; B_host 204.5 (concurrent zero-copy sum) | ✓ |
| 15 | 107c and 107e under-predicted by 31% and 21% | App. J | −31.1%, −20.9% | ✓ |
| 16 | Desktop-class launches of jobs 093–106: 37 launches, within 6% on 32, implied rate 0.88–1.16 of B_host; all 5 server launches 0.27–0.77 | Abstract, §4 | 37 launches, 32 within (plain form: 35), 0.880–1.163; servers 0.268, 0.584, 0.624, 0.738, 0.766 | ✓ (post-hoc class) |
| 17 | "23 rented RTX 5090 machines" with one NUMA node and ≤32 cores | Abstract | 23 distinct GPU UUIDs among those 37 launches (Pf alone accounts for 6 launches) | ✓ |
| 18 | Elasticity of (T−G) on B_host: −0.95 (machines −1.06 to −0.84), jobs 093–104 | §4 | −0.949 over 30 launches; one launch per machine (21): −0.965 [−1.08, −0.81] | ✓ / ≈ interval |
| 19 | Job 105 plain law over-predicts at 25% by 6.9–9.3% on 3 of 4 hosts, pooled 4.2%; at 11% −1.5 to 4.7% | App. B, Table 18 | 25%: +9.26, +7.03, +3.78, +6.92%; 11%: +4.70, +2.00, −1.49, +1.44%; pooled median 4.24% | ✓ |
| 20 | Layer-ahead copy: 0.72 (Pf), 0.91 (TR 9960X), 0.81 (EPYC 7402P), 1.04 (O4) at 11% | Table 18 | 0.722, 0.909, 0.814, 1.042 | ✓ |
| 21 | Fewest-admission set copies 0.60–0.73× the greedy set's in-step copies | §5, App. J | 0.597–0.599 (11%), 0.726 (25%) | ✓ |
| 22 | Fewest-admission in step beats deployed on 10 launches / 7 machines (1.02–1.34× at 11%); beats greedy by 0.08–0.17 at ratio 0.28–0.63; greedy loses 0.85–0.93× and fewest gains 1.02–1.06× at ratio 0.28–0.32; by the CPU 1.11–1.22× | §5 | 104–106 launches: 1.019–1.342; margin over greedy 0.082–0.172; greedy on Pf/TR 0.853–0.933; fewest 1.019–1.056; bypassplan on Pf/TR 1.112–1.217 | ✓ |
| 23 | Server with a 13 GB/s link: fewest-admission set lost in every round (0.82–0.88×) | §5 | 0.834, 0.878, 0.824 | ✓ |
| 24 | dk ("admit less"): median 1.02× on 5 machines, up to 6%, lost 1% at one of 10 cells | Abstract, §5 | 10 gpt-oss cells 0.987–1.057, median 1.019 | ✓ (post-hoc pooling of 106-stable and 107-valid) |
| 25 | dk in 107 ran 0.987–1.074× (median 1.021); law missed ratios by 0.041 and 0.067; median 0.016 against 0.021 for "no change" | §5, App. J | 0.987–1.074, median 1.021; \|Δ\| 0.041, 0.067; median 0.016 against 0.021 | ✓ |
| 26 | dk stable-host pooled median 1.01; with all hosts 1.04 (registered ≥1.02 "held only through the unstable machines") | §5 | 1.0145 (6 cells) against 1.0405 (10 cells) | ✓ |
| 27 | 107 valid hosts: fewest-admission in step 1.16–1.42×, by the CPU 1.04–1.13× | §5 | 1.157 and 1.419; 1.126 and 1.038 | ✓ |
| 28 | MIN with one read 16–51% faster than deployed on O3–O5 at six budgets; prefetched 25–81% | Abstract, §5 | 1.160–1.514; both3p 1.253–1.812 | ✓ |
| 29 | Table 2, host B: ours ÷ FreeToken 1.294 [1.278, 1.312], 1.275, 1.154; Qwen3 12.5% 1.032; llama.cpp 34.9/40.0/47.7 tok/s | Table 2 | 1.294 [1.279, 1.311], 1.275 [1.255, 1.295], 1.154 [1.133, 1.172], 1.032 [1.023, 1.043]; llama.cpp 34.93, 39.98, 47.73 | ✓ |
| 30 | Read-schedule microbenchmark reaches 86–96% of the bound's read term (per layer) | §3, Table 22 | 86.1% (13900KF), 96.3% (9950X), 95.1% (9800X3D) at C=14; per-token, link-only and CPU-only columns also match | ✓ |
| 31 | 21 machines of jobs 093–104 at gpt-oss 11%: 38–54% of Eq. (1), 55–70% of Eq. (2) | §3 | with R* = 38.3 and T_GPU = 2.9: 38.3–54.0% and 55.1–70.1% (the 54% is 100f, where the engine beats the probe by 16%) | ✓ |
| 32 | Engine read >5% faster than the probe on 13 of 68 launch-budgets (up to 1.22×) | Limitations | 13/68 with the reanalysis median G; 11/68 with per-launch G; max 1.222 (100f, 25%) | ≈ (depends on G choice) |
| 33 | Plain form within 6% on 28 of 30 launches of 093–104 at 11%, over-predicts at 25% (median 6%) | §4 | 28/30; median +5.78% | ✓ |
| 34 | On the 34 launches of 093–105 the overlap term brings 33 instead of 18 within 6% at 25% | §4 | 33 against **19** (099f plain = 5.99%) | ≈ rounding boundary |
| 35 | Gain of MIN in the step follows link/CPU ratio: Spearman 0.89 [0.63, 0.97] over 19 machines | §5, App. J | 19 machines, first launch each: 0.868 [0.58, 0.97] | ≈ (launch selection unspecified) |
| 36 | Two hosts with the same CPU model differ by 16–29% | §2 | Largest spread per model at 11%: 9950X 25.6%, 7950X 22.5%, 9950X3D 28.4%, 285K 29.1%, 9800X3D **1.6%**; many same-model pairs differ by under 3% | ⚠ only the largest spreads |
| 37 | Parity: mean KL 0.0019 and 0.0005 nats, top-1 98.5% and 99.3%, loss +0.16% and +0.19%; stock placement 0.0017 and 0.0008 | App. K | on-machine `parity_kl.json`: 0.00186, 0.00053, 0.9850, 0.9935, +0.160%, +0.192%; n27 0.00171, n36 0.00084 | ✓ (summary file; not recomputed from binary dumps) |

### Registration timeline check

Times are CDT. Push times come from GitHub's activity API.

| Job (hosts) | Header commit | Push recorded by GitHub | Rental start (ledger) | Start − push |
|---|---|---|---|---|
| 105 a–e | 17:05:31 | 17:05:33 | 17:05:36–39 | +3.4 to +6.6 s |
| 105 f, g (amendment: hosts only) | 17:09:26 | 17:09:27 | 17:09:28–29 | +1.7, +2.4 s |
| 106 a–d | 20:43:48 | 20:43:51 | 20:43:58–20:44:02 | +7.6 to +11.1 s |
| 106 e (amendment: host only) | 20:44:13 | 20:44:15 | 20:44:16 | +1.9 s |
| 107 a–d | 00:17:31 | 00:17:35 | 00:17:44–47 | +9.9 to +12.4 s |
| 107 e (V0 replacement, per registered rule) | 00:24:30 | 00:24:33 | 00:24:36 | +3.7 s |
| 107 f (amendment, after 107c/d results committed at 01:35:53) | 01:36:28 / 01:36:57 | 01:36:32 / 01:37:00 | 01:37:00 | +0.4 s |
| 107 g (amendment, after 107b results) | 01:41:16 | 01:41:20 | 01:41:20 | +0.1 s |

I diffed every amendment. Each changed only header comments naming hosts or offers, and the stub that names the host. No prediction, gate or code changed. Host 106e, a substitute, is by GPU UUID the same card as panel host Pe (099e), as the paper states.
