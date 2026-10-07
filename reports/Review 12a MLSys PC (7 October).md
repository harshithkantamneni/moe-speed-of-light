# Review 12a: MLSys 2027 main track, PC review (blind)

**Submission:** "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth"
**Date:** 7 October 2026

---

## Materials read

- `paper/paper.pdf` (37 pp.). I read the full main text (abstract to Conclusion), using `pdftotext` and the LaTeX source `paper/paper.tex` for exact wording.
  - Read: Appendices A (checklist), B (prediction log), D and J in full; C, E, F and G in part; the forecaster paragraph of I; Tables 3, 5–10 and 16–20.
  - Not read in detail: H, K, L and M. The 0.65·C horizon rule and the audit of published systems are therefore not verified here.
- `paper/app_more.tex`, `paper/tab_job106.tex`, `paper/tab_job107.tex` (source of Appendix J and Tables 3 and 20).
- `paper/supplement.pdf`: its structure, plus Table 9 (job 106 scorecard) and Table 10 (job 107 scorecard).
- The analysis scripts, **for definitions only** (none of them was run):
  - `scripts/panel_099.py` (`cell_data`, `host_info`, `label_of`, `_status`);
  - `scripts/speed_limit.py` (`host_rates`) and `gpu-branch/jobs/ec2/fetch_table.py` (`bandwidths`);
  - `scripts/job106.py` (`law`, `load`, clause logic) and `scripts/job107.py` (`v0`, `losses`, `machine_classes`, `CLASS_CORES`);
  - `scripts/reanalysis.py` (`profiled_G` and the signature of `sum_law`) and the header of `scripts/headline_stats.py`.
- `prereg/reanalysis.json`: only the `sum_law` entry, to confirm how G0 is defined. I then recomputed G0 from the raw profiles.
- `gpu/vast_ledger.json`: rental start and stop times for jobs 106 and 107.
- Second checkout `/home/claude/gpu-branch`:
  - `jobs/106_onlineadmit@vast.sh` and `jobs/107_newhosts@vast.sh`, as first committed and with every later amendment (`git diff`), plus the per-host stubs 106a–e and 107a–g;
  - `git log --format='%h %ad'` / `%cd` and `git show --stat` with a custom format, for commits since 5 October.
- Raw results in `results/` for jobs 093–107:
  - `ec_*.jsonl`, `st_*.json`, `concur.txt`, `cores.txt`, `cpu.txt`, `numa.txt`/`lscpu.txt`, `free.txt`, `v0.txt`, `validity.txt`, `g_prof.json`, `q_prof.json`, `nvidia-smi-q.txt`, `gpu.csv`;
  - also `069c_*/prof_C{14,32}.json`, `105?_*/prof_C{14,32}.json`, `102?_readsched/readsched_C*.txt`, `081_headline_law/bs1.jsonl` and `106a/plan_g14.txt`.

## Independence statement

All recomputation was done with my own scripts in `scratchpad/rev12a/`:

- `common.py`, `j106_107.py`, `classes.py`, `c32.py`, `foresight.py`, `minadm.py`, `spear.py`, plus a few inline checks.
- I re-implemented the paper's definitions from reading the scripts: B_host, B_c, B_p, time per token, counters per token, Eq. (3) solved by fixed point, round-level bootstrap and the class rule. I did not run the authors' scripts.

I did not open any of the following:

- anything under `reports/`, other than writing this file;
- `prereg/*outcome*.md`, `research_notes/`, `apply/`, or `paper/paper_v1_prereview.tex`;
- any review, number-check, plan or progress-log file;
- `prereg/job106.json`, `job107.json`, or the `scorecard_*.json` files (I scored job 107 myself from its header).

I did not read commit messages; every git command used a format string without the subject or body. Nothing in either repository was modified apart from creating this file, and nothing was committed.

---

## 1. Summary

The paper studies batch-1 decode of two MoE models whose experts mostly sit in host DRAM:

- gpt-oss-120b in MXFP4;
- Qwen3-30B-A3B in BF16.

Decode runs on rented RTX 5090 machines through a llama.cpp patch that keeps a per-layer GPU expert cache. The paper makes five contributions.

1. **Two bounds.**
   - Eq. (1): MIN-with-bypass host reads at the probe's highest host rate, overlapped with GPU work.
   - Eq. (2): the same for on-demand systems, with the GPU's non-expert time added in series.
   - A microbenchmark replaying MIN's reads reaches 86–96% of the bound's read time.
2. **An accounting relation, Eq. (3).** It states T = G + (M + (1−G/T)A)·S/B_host, using profiled G, the run's own miss and admission counters, and B_host.
   - It was found on 34 exploratory launches.
   - Registered before job 106, it held on 3 re-rented machines.
   - Registered before job 107 on new machines, it failed, by up to 11% on a desktop and 52–59% on a dual-socket server.
   - Afterwards the authors draw a "desktop-class" boundary (one NUMA node, at most 32 usable cores) that separates the small misses from the large ones.
3. **In-engine oracles.**
   - MIN with each admission read once is 16–51% faster than the deployed cache (25–81% with copies issued ahead).
   - Belady prefetch and two-read MIN can lose.
   - Among MIN's hit-optimal schedules, the fewest-admission one beats the greedy one where the PCIe link is slow relative to the CPU.
   - An online "admit less" rule gives a median 1.02×.
4. **A price on foresight.** Half of MIN's saving needs about 0.65·C distinct experts of lookahead per layer, and the linear and GRU forecasters tested close at most 15% of the gap.
5. **A pre-registration artifact.** About 1,460 scored clauses over jobs 073–107, with failures reported.

## 2. Strengths

- **Unusual transparency, and it checks out.** Every one of the roughly 40 quantitative claims I recomputed from raw rows reproduced, a few with trivial rounding differences (table below). The registered predictions of jobs 106 and 107 sit in the job-script headers. The registration order also checks out against the git and ledger timestamps:
  - each registration commit precedes its rental start by 3–13 s (for example, `1a0616a` at 00:17:31 and hosts 107a–d starting at 00:17:44–47);
  - the mid-job amendments to 107 changed only how hosts were replaced, never a prediction or a gate.
  - My own clause-by-clause scoring of the job 107 header gives 46 clauses and 13 failures, identical to the supplement.
- **The job 107 failure is reported, not buried.** The paper states that the test was inconclusive by its own rule (2 of the 3 valid hosts it required) but treats it as a failure anyway. It also reports that the registered advantage of the overlap term over the plain form failed. The abstract says "it failed". This is better practice than most systems papers.
- **Sound measurement methodology.**
  - The machine is the unit of analysis; there are relaunch checks (within 1.7–3.2%) and round-level intervals.
  - Validity gates were registered before job 107.
  - Ratios are of means paired by problem.
  - The paper candidly notes that machines differ far more than problems (96% of the variance of the log gain).
- **A useful framing for an active area.**
  - Bounds are stated relative to the same machine and the same GPU memory.
  - The gap is split into serialisation and reads beyond MIN.
  - The oracles show that *how* foresight is spent matters (Belady prefetch and double reads can lose).
  - The fewest-admission schedule's advantage tracks the link-to-CPU ratio (Spearman 0.88–0.89 over 19 machines, reproduced).
  - Together this is informative for the many papers that report speed-ups over self-chosen baselines.
- **A strong engineering baseline.** At equal GPU expert memory on host B the system is 1.03–1.29× FreeToken and 2.0–3.2× stock llama.cpp, reproduced exactly from `bs1.jsonl`.

## 3. Weaknesses (most important first)

### W1. Eq. (3), the "where the seconds go" relation, has never passed a registered test on a machine outside its discovery set, and the abstract and contributions present it selectively.

The registered record is as follows.

- **Job 105.** The plain form failed its pooled criterion: 4.2% against 4% (I recompute 4.25%; it over-predicts gpt-oss 25% by 6.9–9.3% on 3 of 4 hosts).
- **Job 106.** The overlap form was registered on **five** hosts and held on the three re-rented discovery machines (12 cells, median 2.34%, maximum 5.99%, a knife-edge pass on 106a at Qwen3 25%).
  - It failed on both new machines (EPYC 7302 and Xeon 8347C, under-predicted by 10–52%).
  - Its registered pooled median failed (4.67% against 4%).
  - Excluding those two hosts was a post-hoc judgement. Clause P8 ("base varies ≤2% between rounds") was a prediction, not a registered exclusion rule, and the loss check was first registered in job 107.
- **Job 107.** Both valid new machines failed.
  - The relation under-predicted all 8 of 8 cells and missed the 6% band at 7.
  - The overlap-versus-plain clause failed.

The abstract ("the relation held when registered before new runs of 3 of them") and the contributions bullet ("held when registered before new runs and a second model") omit that the job 106 registration covered two other machines and failed its pooled test. The "second model" success also holds only on the re-rented machines: on the new desktop, Qwen3 errors were −11.3% and −6.4%. The body is accurate; the headline summaries are not.

In addition:

- The test cannot single out the overlap term. A half-discount passes 11 of 12 stable cells and misses-only passes 10 of 12; both are reproduced, and the paper says so.
- Eq. (3) uses the run's own counters M and A, so it is an accounting consistency check, not a forecast.

Honestly stated, Eq. (3) is an in-sample description of desktop launches with errors from −10.6% to +14.0% across both gpt-oss budgets, not a validated predictive relation.

### W2. The "desktop-class" scope is post hoc, tightly fitted to five server launches (one of them valid), mislabelled, and not flagged as post hoc where it matters most.

The rule is "one NUMA node and at most 32 usable cores". It needs **both** features to separate the five server launches:

- the EPYC 7302 has only 15 usable cores but 2 NUMA nodes;
- the Xeon 8347C has 1 NUMA node and 36 usable cores.

The 32-core threshold sits exactly between the largest discovery machine (Threadripper 3970X, 32 usable cores, itself a −7.0% miss) and the smallest excluded one (36). "Usable cores" is the container's allocation, not hardware: the EPYC 9754 shows 61 of 128.

The "desktop-class" set also contains an EPYC 7402P and two Threadrippers, so "desktop" is a misnomer.

The Setting, Section 4 ("Found after job 107 and not tested in advance") and Limitations do say the line is post hoc. The abstract, however, *defines the found set by this class* ("On the 23 rented RTX 5090 machines, each with one NUMA node and at most 32 cores, on which we found it…"), and the introduction and conclusion use it the same way, without the qualifier. A reader of the abstract would take the scope to have been specified in advance. Timestamps also support that the rule came after the data: the PDF's creation time (02:12) is 28 minutes after the last job 107 result commit (01:43:59).

The pattern itself is fairly consistent. All five server launches read at 0.27–0.77 of B_host, against 0.88–1.16 on the 37 desktop launches (both reproduced). Even so, the one valid new in-class machine (Ryzen 9 5900XT) also failed 3 of its 4 cells, so the class does not rescue out-of-sample accuracy.

### W3. Over-general phrasing of what Eq. (3) shows, and "nothing fitted" is overstated.

- **Unscoped statements.** Section 4's heading ("Time is GPU compute plus host reads.") and the first sentence of the Conclusion ("With experts in host memory, batch-1 MoE decode is the GPU's compute plus host bytes.") are general. The valid server leaves 58% (gpt-oss 11%) and 50% (gpt-oss 25%) of its time unexplained (reproduced).
- **The functional form was itself selected.** The overlap term was introduced after the plain form failed job 105's registered test.
- **G is not per-host for most launches.** For the 30 discovery launches of jobs 093–104, G is a cross-host median of 7 profiles (4.2835 ms, reproduced).
- **B_host is a maximum over heterogeneous probe modes.** On the server it is a single concurrent 128-thread CPU+PCIe reading (204.5 GB/s); the CPU-only readings are 151 GB/s at 12–16 threads and 118 GB/s at 128.
- **Partly circular evidence.** "The engine reads at the machine's best rate" means the same as "Eq. (3) fits", since implied rate ≈ 1 is equivalent to error ≈ 0. The elasticity of −0.95 is computed with one fixed G across machines and is not independent evidence.

### W4. The deployable gains are small, and the large numbers are oracle upper bounds.

The online "admit less" rule gives:

- a median of 1.019× over 10 gpt-oss cells, ranging 0.987–1.057;
- several within-host intervals that include 1 (for example, 106a at gpt-oss 11%: [0.991, 1.017]);
- a registered "≥1.01 on every host" that failed on 2 of the 3 stable job 106 hosts.

The abstract leads with "gains up to 6%". The 16–51% and 25–81% oracle gains are upper bounds that, by the paper's own Section 6, no realisable forecaster approaches (at most 15% of the read gap). The system's lead over FreeToken is 0.97–1.29×. The contribution is therefore mainly characterisation, and the abstract should weight it that way.

### W5. External validity is narrow, and the "lower bounds" are relative to the probe, not the hardware.

- **Narrow setting.**
  - One GPU model (RTX 5090) carries all the relation and oracle work.
  - The workload is a single domain: AIME, teacher-forced, also used during development, with 20 problems × 256 tokens per configuration in jobs 100–107.
  - Batch is 1.
  - Eq. (3) for Qwen3 was registered on only 4 desktop machines, with errors from −11.3% to +6.0%.
- **The probe is not a ceiling.** By Eq. (3)'s own accounting the engine reads faster than B_host on 12 of 68 launch-budgets (paper: 13), up to 1.22× on host 100f. Either the probe is not the ceiling or the accounting is off by about 20% on some desktops.
- **Consequence for the bounds.** Eq. (1) and Eq. (2) are bounds given the probe; the paper should say "probe-relative" in the abstract.

### W6. The slow-link claim about the fewest-admission schedule is scoped beyond the valid data.

The abstract says the fewest-admission schedule "also wins where the PCIe link is slow". The two slowest-link machines measured, both excluded as invalid, show otherwise:

- 106c, link-to-CPU ratio 0.14: 0.82–0.88 per round;
- 107e, ratio 0.21: 0.97 [0.94, 1.01], with rounds at 0.948 and 0.999.

The valid evidence covers ratios ≥0.28. On 107e the CPU-loaded variant does gain (1.13), consistent with the registered crossover. The claim should be scoped to ratios ≥0.28, or should name the CPU-loaded variant for slower links.

### W7. Clarity.

- **Density.** The paper is extremely dense. Sentences routinely carry 5–10 macro-generated ranges, and the argument depends on job numbers and host nicknames (Pf, Pe, O3–O5, "job 104", "panel host"). Neither the job numbers nor the nicknames are tabulated in one place.
- **Appendix weight.** The appendices run to about 25 pages, and much of the evidence for main-text claims lives there.
- **The abstract.** It is hard to parse (for example, "Carried online, that lesson, admit less, gains up to 6%…").
- **The scorecard.** The prediction scorecard is admirable, but its roughly 1,460 clauses (25% failed in jobs 073–098) are not distilled into what they say about the authors' model of the system.

### W8. Minor numerical and wording issues (none changes a conclusion).

| Claim | Paper | My recomputation |
|---|---|---|
| Launch-budgets where the engine reads more than 5% faster than the probe | "13 of 68" | 12 of 68 |
| Job 101 relaunch difference | 1.9% | 1.7% |
| Fraction of Eq. (2) | "55–70%" | 55–72%, depending on T_GPU |
| Measured time beyond Eq. (3) on desktops | "at most 11%" | 11.8% of Eq. (3), or 10.6% of T |
| Server error for every form | "56–58%" | these are per-form medians; per cell the errors are 49.7–60.0% |
| Running example, MIN in the step (O4) | 15.2 ms in the text | Table 9 gives 15.13 (speed-averaged) |
| Job 106 round spread | "39%" | (max−min)/mean; 52% as max/min−1 |

## 4. Questions for the authors

1. Why does the class rule use *usable* (container-allocated) cores, and why 32 rather than, say, NUMA nodes alone plus a memory-channel criterion? Did you evaluate any alternative rule? How would you classify a 128-core EPYC rented with all cores usable?
2. On the dual-socket server, the probe's CPU rate peaks at 151 GB/s with 12–16 threads, while the engine ran 126 helpers. Did you run the engine with about 14 helpers and memory bound to one node (`numactl --membind`/`--interleave`)? A short pre-registered run of that kind would test the causes you list.
3. Would you state in the abstract that the job 106 registration covered five hosts and failed its pooled criterion (4.67% against 4%), and that the 106c/106d exclusions were not registered?
4. What explains host 100f (Ryzen 9 9950X)? It runs 18.07 ms against about 20.5 ms on other 9950X hosts and implies 1.16–1.22× the probe's best rate. Is the probe under-reading there, or is G over-stated?
5. Qwen3 errors change sign across desktops (+6.0% on the Threadripper 9960X, −11.3% on the 5900XT). Is the G profile (one problem, 24 tokens) representative for Qwen3? What is Eq. (3)'s error with G profiled over more tokens?
6. Can Eq. (3) be used *before* a run, with M and A from a trace replay instead of the run's own counters? If so, what is its accuracy on jobs 105–107?
7. How sensitive are the bound fractions and the foresight value to the workload (AIME, which was also used during development)? Is there any non-math or sampled-decode check in the engine?
8. Is the 0.65·C horizon constant evaluated leave-one-model-out on the nine traces, and what is the held-out error?

## 5. What would raise my score

- **Re-scope the headline claims.**
  - State in the abstract, introduction and contributions that Eq. (3) passed registered tests only on re-rented discovery machines and failed on every new machine (jobs 106 and 107).
  - Mark the desktop-class line as post hoc wherever it scopes a claim, and call it by its operational definition rather than "desktop".
  - Remove or scope the first sentence of the Conclusion and the Section 4 heading.
- **Report Eq. (3)'s accuracy as a distribution.** Use one table separating in-sample from out-of-sample launches, by model and budget (for example, gpt-oss desktop −10.6% to +14.0%; Qwen3 −11.3% to +6.0%), instead of "within 6% on 32 of 37".
- **Run a new pre-registered test** with the class rule fixed in advance: at least 3 new single-NUMA machines with ≤32 cores, on both models. Ideally add a pre-registered server run that varies helper count and NUMA binding.
- **Make the bounds hardware-anchored**, or label them probe-relative. Also explain why the engine exceeds the probe by up to 22% on some hosts.
- **Scope the slow-link claim** about the fewest-admission schedule to link-to-CPU ratios ≥0.28, or show valid data below that.
- **Improve clarity substantially:**
  - one table that maps every host nickname to its CPU, NUMA nodes, cores, link-to-CPU ratio and jobs;
  - a single "registered tests and outcomes" table;
  - fewer numbers per sentence.
- **Broaden the evidence** (optional, but it would raise significance): one more GPU, such as the RTX 4090 already in the grid, and a non-math or sampled workload for the engine experiments.

## 6. Scores

| Criterion | Score |
|---|---|
| **Overall** (1–10; 6 = weak accept, 8 = strong accept) | **5** (borderline, leaning reject) |
| Soundness (1–5) | 3 |
| Significance (1–5) | 3 |
| Novelty (1–5) | 3 |
| Clarity (1–5) | 2 |
| Confidence (1–5) | 4 |

**Justification.**

- **Data and integrity.** These are excellent. Every number I recomputed reproduces, and the registration order is verified.
- **The central relation.** It is presented more strongly in the abstract, introduction and contributions than the registered evidence allows (W1–W3).
- **Practical gains.** These are small (W4).
- **Writing.** It is very hard to follow (W7).
- **Novelty.** The ingredients are known (MIN with bypass, a roofline-style bound, Demand-MIN/FOO-style tie-breaking). The novel part is the in-engine, time-domain measurement across many machines. That is useful, but incremental for the main track.
- **Path to acceptance.** If the claims are re-scoped as in §5 and clarity is fixed, I would move to 6–7. The problems are mostly framing problems, not data problems.

---

## 7. Claims checked against raw data

Definitions:

- **Errors** are Eq. (3) prediction ÷ measured − 1, with the measured time per token averaged over problems and over rounds.
- **B_host** is the maximum of every CPU, PCIe and concurrent-sum reading in `concur.txt`.
- **G** is `g_prof.json`/`q_prof.json` where present, otherwise the median of the 069c and 105 profiles (4.2835 ms at gpt-oss 11%, 4.4979 ms at 25%).
- **R\*** is 38.505 and 15.52 per token, from job 102's replay.

**Verdicts:** ✓ reproduced; ≈ approximately reproduced (difference noted); ⚠ reproduced but framing or scope issue.

| # | Claim | Where | My recomputed value | Verdict |
|---|---|---|---|---|
| 1 | The relation was found on 23 RTX 5090 machines, each with one NUMA node and ≤32 cores | Abstract | 23 distinct GPU UUIDs over the 37 desktop-class launches of jobs 093–106; all RTX 5090; NUMA = 1; 8–32 usable cores | ✓ ⚠ (class drawn post hoc, W2) |
| 2 | Within 6% on 32 of 37 launches (gpt-oss 11%) | Abstract, §4 | 32/37; misses: 099d −7.5%, 099i −10.6%, 100a −6.2%, 100f +12.6%, 103e −7.0% | ✓ |
| 3 | Implied read rate 0.88–1.16 of B_host on desktops; 0.27–0.77 on all 5 server launches | Intro, §4 | 0.880–1.164; servers 0.269, 0.584, 0.624, 0.738, 0.766 (first round) | ✓ |
| 4 | Plain form within 6% on 28 of 30 launches (093–104) at 11%; over-predicts at 25% (median 6%) | §4 | 28/30; median +5.87% | ✓ |
| 5 | On 34 launches (093–105) the overlap term brings 33 instead of 18 within 6% at 25%, at the cost of 29 instead of 32 at 11% | §4 | 33 vs 18; 29 vs 32 | ✓ |
| 6 | Job 106, re-rented hosts: all 12 cells within 6%, median 2.3%, largest 6.0% | Intro, §4, Table 3 | median 2.34%, maximum 5.99% (106a Qwen3 25%) | ✓ (knife-edge) |
| 7 | Plain form missed those cells by up to 10.1% | §4 | 10.08% | ✓ |
| 8 | Half-discount passes 11/12 cells; misses-only passes 10/12 | §4 | 11; 10 | ✓ |
| 9 | 106c loss 0.21–0.45 nats per token against 0.19 (gpt-oss) and 0.085 (Qwen3) elsewhere | §4, App. J | 0.208–0.446; other hosts 0.1894–0.1904 and 0.0854–0.0856 | ✓ |
| 10 | 106d deployed cache varied by 39% between rounds | §4, Limitations | 38.8% as (max−min)/mean (rounds 14.52, 22.10, 21.93 ms) | ✓ |
| 11 | On 106c/106d the relation under-predicted by 10–52%; registered pooled median 4.7% against 4% | §4 | −10.2% to −52.4%; pooled median over 20 cells 4.67% | ✓ ⚠ (abstract omits, W1) |
| 12 | Job 107: 7 rented; 3 stopped at the gate with 50–96 GB in use; 2 ran rounds 3.5–6.0% apart | §4, App. J | 50, 50, 96 GB; 3.54% (EPYC 9754), 5.98% (EPYC 7663) | ✓ |
| 13 | Job 107: under-predicted 8/8 cells, missed 6% at 7; 5900XT up to 11% with 1 of 4 within | §4 | 5900XT −7.1, −2.3, −11.3, −6.4%; AMD ES −58.9, −52.2, −58.3, −56.3% | ✓ |
| 14 | Dual-socket server under-predicted by 52–59% | Abstract, Intro | 52.2–58.9% | ✓ |
| 15 | With a third valid host, the pooled median could not have come below 6.8% | §4 | 6.75% (adding four zero errors) | ✓ |
| 16 | New desktop: plain median 6.0% (2 cells within) against registered overlap form 6.8% (1 within) | §4 | 5.99% (2) vs 6.75% (1) | ✓ |
| 17 | On the server every form misses by 56–58% | §4 | Per-form medians 56.3–58.3%; per cell 49.7–60.0% | ✓ (these are medians; wording) |
| 18 | Server B_host 204 GB/s, a CPU+link reading; CPU-only 151 GB/s at 16 threads and 118 at 128; still misses by 51% | §4 | 204.5 (t=128 zero-copy sum); 151.4 and 117.6; −51.4% at gpt-oss 11% (−46% to −50% at the other cells) | ✓ |
| 19 | Elasticity of T−G to B_host −0.95 [−1.06, −0.84] over machines (093–104) | §4 | −0.950 [−1.066, −0.844]; 21 machines, 30 launches | ✓ ⚠ (not independent of the fit, W3) |
| 20 | Fig. 1 shares at gpt-oss 11%: G 12–47%, MIN 31–54%, beyond MIN 22–35%; serialisation 23–68% of the gap to Eq. (1); server shortfall 58% / 50% | §4, Fig. 1 | 12–47, 31–54, 22–34, 23–68; server 58.3% / 50.7% | ✓ |
| 21 | On desktops, measured time exceeds Eq. (3) by at most 11% at 11% | §4 | 11.8% of Eq. (3), or 10.6% of T (099i) | ≈ |
| 22 | G 4.1–4.7 ms (gpt-oss) and 5.4–5.9 ms (Qwen3) on job 106's stable hosts; T_GPU 2.9–3.4 ms on 7 hosts | §3, §4 | 4.15–4.65; 5.42–5.93; non-expert kernels 2.94–3.36 | ✓ |
| 23 | Margin rule (dk) cuts reads 4–6% (11%) and 6–10% (25%); runs 1.00–1.02× and 1.01–1.06× (Qwen3 1.01–1.04×); ≥1.01 fell short on 2 of 3 hosts; pooled ≥1.02 held only through the unstable hosts (1.01 on the stable ones) | §5, Table 3 | 4.4–5.8% / 6.3–10.5%; 1.003–1.024 / 1.008–1.057; Qwen3 1.010–1.038; pooled 1.04 all vs 1.015 stable | ✓ |
| 24 | Eq. (3) predicts the dk/lrn ratios within a median 0.013 (maximum 0.045); predicting no change gives 0.016 | §5 | 0.0135 (maximum 0.045, Pf 25% lrn); 0.016 | ✓ |
| 25 | Learned rule (lrn): 7–14% fewer reads; 267–439 µs host time; 0.98–1.05×; caps exceeded on Pe | §5 | 6.6–14.2%; 267–439 µs; 0.982–1.055; Pe 435/439 µs (cap 400) and 739/1030 µs (cap 700) | ✓ |
| 26 | Job 107 dk 0.987–1.074 (median 1.021); law misses 0.041 and 0.067; median 0.016 against 0.021 for no change | §5, App. J | Identical | ✓ |
| 27 | "Admit less" gains up to 6% on gpt-oss (median 1.02× over 5 machines; lost 1% at 1 of 10 cells) | Abstract | Maximum 1.057, median 1.019, minimum 0.987 | ✓ ⚠ (leads with the maximum, W4) |
| 28 | Job 107 fewest-admission set: in the step 1.16–1.42×, by the CPU 1.04–1.13×; in the step wins on both hosts | §5 | 1.157 and 1.419; 1.126 and 1.038 | ✓ |
| 29 | MIN with 1 read is 16–51% faster on O3–O5 at all six budgets; 25–81% prefetched; O4 example 20.6 / 15.2 / 13.8 ms | Intro, §5 | 1.160–1.514; 1.253–1.812; 20.64 / 15.16 / 13.81 | ✓ |
| 30 | The usual ways of spending foresight gain at most 15% or lose at each model's smallest budget on O4/O5 | §5 | Maximum 1.148 (nb2, Qwen3 12.5%, O4); others 0.54–1.05 | ✓ |
| 31 | Fewest-admission on 10 launches / 7 machines: 1.02–1.34×; +0.08–0.17 over greedy (ratios 0.28–0.63); on the 2 slowest, greedy 0.85–0.93× and fewest 1.02–1.06×; loaded by the CPU 1.11–1.22×; copies 0.60–0.73×; misses +2.5–5.3% | §5, App. J | 1.019–1.342; +0.082–0.172; 0.853–0.933 and 1.019–1.056; 1.112–1.217; 0.597–0.726; +2.5–5.3% | ✓ |
| 32 | Round-level intervals on the slow-link hosts start at 1.01 and 1.05 | §5 | [1.013, 1.025] (Pf), [1.048, 1.063] (TR 9960X) | ✓ |
| 33 | On the 13 GB/s-link server the fewest-admission set lost in each round (0.82–0.88×) | §5 | 0.824, 0.878, 0.834 | ✓ ⚠ (slowest links not covered by valid data, W6) |
| 34 | Spearman of MIN-in-the-step gain against link-to-CPU ratio: 0.89 [0.63, 0.97] over 19 machines | §5, App. J | 0.88–0.89 [0.61, 0.97], 19 machines | ✓ |
| 35 | Same-CPU hosts differ by 16–29%; relaunches within 1.9% (job 101) and 3.2% (job 106) | §2 | 15.6–29.1% (panel pairs); 1.7% (101a vs 099f); 3.1–3.2% (106a vs 105b) | ≈ (job 101: 1.7%) |
| 36 | Job 102 replay reaches 86–96% of the bound's read time | Intro, §3 | 86.1–96.3% (layer mode, three hosts, two budgets) | ✓ |
| 37 | On the 21 machines of jobs 093–104 at gpt-oss 11%, the cache stands at 38–54% of Eq. (1) and 55–70% of Eq. (2) | §3 | 38.5–54.3%; 55–72% depending on T_GPU (2.94–3.36 ms) | ✓ / ≈ |
| 38 | Table 2, host B: 69.9 / 54.0 / 34.9 tok/s; ours ÷ FreeToken 1.294 [1.278, 1.312]; 2.00× llama.cpp (also the 25%, 40% and Qwen3 12.5% rows) | Table 2 | Identical for all 4 rows checked | ✓ |
| 39 | The engine read more than 5% faster than the probe's highest rate on 13 of 68 launch-budgets (up to 1.22×) | Limitations | 12 of 68; maximum 1.223 (100f, 25%) | ≈ |
| 40 | Fewest admissions 12.4 per token against 21.8 for greedy at gpt-oss 11% | §3 | Engine trace (20 problems): 12.2–12.8 against 20.2 (the paper's figure is on the 30-problem bound trace) | ≈ (consistent) |
| 41 | Predictions committed before each machine started; job 107 amendments changed only host replacement | §2, App. B | Each registration commit precedes its ledger start by 3–13 s; amendment diffs are header comments on replacement rules only | ✓ |
| 42 | Job 107: of 46 clauses, 13 failed (relation 7 cells, pooled median, overlap vs plain, dk at one cell, dk prediction at 2 cells, count of valid hosts) | App. B, Supplement Table 10 | My own scoring of the header gives 46 clauses and 13 failures, the same ones | ✓ |

### On the four specific points I was asked to examine

1. **Registered tests of jobs 106 and 107 against the headers and git history.**
   - Both headers were committed before their rentals started: job 106 at 20:43:48, with hosts starting 20:43:58–20:44:02; job 107 at 00:17:31, with hosts starting 00:17:44–47.
   - The job 106 amendment (20:44:13) replaced only host e. The job 107 amendments (01:36:28, 01:36:57 and 01:41:16) replaced only hosts f and g, under stated rules, and changed no predictions or gates. I compared each version with `git diff`.
   - The paper's descriptions of each registered clause match the headers.
2. **Honesty about job 107.** The outcome is described honestly and in full in Section 4, Appendix B and Appendix J, and the abstract says it failed.
3. **The post-hoc "desktop-class" boundary.** It is labelled post hoc in the Setting, Section 4 and Limitations, but not in the abstract, introduction or conclusion, which use it to scope the relation's success (W2).
4. **Scope of the Eq. (3) claims.** The body scopes them carefully. The abstract, contributions, the Section 4 heading and the Conclusion's first sentence over-state them: they omit the failed pooled clause of job 106 and the new-host failures there, and they generalise beyond the machines where the relation was found (W1, W3).
