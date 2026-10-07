# Review 17a: MLSys 2027 main track, PC review

**Submission:** "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth"
**Reviewer role:** program committee member, blind review
**Date:** 7 October 2026

---

## Materials read

- `paper/paper.pdf`, 42 pages. I read the main text (Sections 1 to 10) closely. I read Appendices A to O in full, with closer attention to C (prediction log), F (order-free accounting), H, I, J, K, L and M.
- `paper/paper.tex`, plus targeted greps of `app_wsg.tex`, `app_traces.tex`, `app_more.tex` and `tab_job105/106/107.tex`, to check labels and cross-references.
- `paper/supplement.pdf`: the scorecard. I skimmed the structure and the jobs 073–081 rows, and checked the totals against Table 8.
- I read these artifact scripts for definitions only: `scripts/decomp_measured.py`, the relevant parts of `scripts/reanalysis.py`, `factorial_shapley.limit1_of`, `speed_limit.host_rates`, `jobs/ec2/fetch_table.bandwidths`, `panel_099.cell_data`, `sumlaw_paper.profiles`, `value_map.py` (the policy definitions), `dc_aime.py`, and the header of `audit_ours.py`. I also read the oracle-planning part of the engine patch, `jobs/ec2/llama.cpp-expert-cache-4da6337-oracle5.patch`.
- Artifact data:
  - `gpu/vast_ledger.json`;
  - `prereg/speed_limit_v2.json`, used only to compare my own R\*;
  - `prereg/dc_aime.json`, opened only after I had computed W50 myself.
- Raw results in `/home/claude/gpu-branch`:
  - `results/<job>/ec_*.jsonl` (per-problem `decode_ms`, `n_decode` and config/stats paths), `st_*.json` counters, `concur.txt`, `cores.txt`, `nvidia-smi-q.txt` and `lscpu.txt` for jobs 093–108;
  - `g_prof.json` and `prof_C*.json` (Nsight) for jobs 069c and 105–108;
  - `readsched_C*.txt` for job 102;
  - `bs1.jsonl` for jobs 081, 089 and 098;
  - the routing traces `084c/route_aime25_gptoss.npz` and `084b/route_aime25_qwen3.npz`.
- Job-script headers with the registered predictions: 099, 102, 104, 105, 106 and 107.
- `git log --format='%h %ad'` / `%at` on both repositories, for timing only.

## Independence statement

- I did not open anything under `reports/` except to write this file.
- I did not open `prereg/*outcome*.md`, `research_notes/`, `apply/`, `paper/paper_v1_prereview.tex`, or any file named like a review, number check, plan or progress log.
- I did not read commit messages; I used only hashes and timestamps.
- Every number in the claims table below comes from my own code, in `/tmp/claude-0/.../scratchpad/rev17a/`. That code reads the raw rows, counters, probes, profiles and routing traces. For example:
  - R\* comes from my own MIN-with-bypass simulator run on the routing trace.
  - The window policies come from my own replay, written from the definitions in Appendix J.
  - The pooled MIN comes from my own implementation.
- I used the authors' scripts only to pin down definitions, such as the parse rule for B_host and the decomposition formula.
- I modified nothing in either repository. I made no commits.

---

## Summary

The paper studies batch-1 decode of MoE models whose experts mostly live in host DRAM. The main experiments use gpt-oss-120b (and Qwen3-30B-A3B in some) on rented single-RTX-5090 machines. It makes four contributions:

1. **A bound (Sec. 3).**
   - Eq. (1): Belady's MIN with bypass gives the fewest host reads R\* for C slots per layer. R\*·S at the machine's highest probed host read rate bounds any system that executes the exact routing.
   - Eq. (2): for systems that read only after routing, the bound tightens by adding T_GPU, the GPU's non-expert time.
   - A microbenchmark shows that the bound's read time is 86–96% reachable without the model.
   - The authors' llama.cpp expert cache leads FreeToken at 11 of 12 cells and runs 2.0–4.0× stock llama.cpp.
   - At gpt-oss 11% it reaches 31–54% of Eq. (1) on 25 consumer machines.
2. **A measured decomposition of the deployed cache's gap to Eq. (1) (Sec. 4, Table 3, Fig. 2).** Oracles inside the engine remove three parts. The cache-set and read-path parts are split order-free with Shapley values. The fourth part is a profiled constant, T_GPU, and the fifth is the residual. Results come from 15 machines, 13 of them "fast-link".
3. **How foresight must be spent (Sec. 5).**
   - Read each admission once.
   - Choose the fewest-admission hit-optimal schedule.
   - Choose the read path by the link-to-CPU ratio.
4. **What foresight is worth (Sec. 6).**
   - Exact next-token routing recovers at most 0.31 of MIN's read gain.
   - Half the gain needs a window of about 0.65·C distinct experts per layer (9 models) or 0.66–0.81·C (AIME workload).
   - Realisable forecasters close at most 15%.

Throughout, the paper keeps an "open record": every job's predictions sit in a script header that was committed before the rental started. Appendix C and the supplement score about 1,480 clauses, including the failures.

---

## Strengths

1. **Unusually reproducible numbers.** I recomputed 33 quantitative claims from the raw rows, counters, probes, profiles and routing traces with my own code (table below). Every one reproduces, almost always to the printed digit.
   - Table 3 matches exactly, means and 95% intervals alike.
   - Table 4 matches exactly for host B (gpt-oss rows and Qwen3 12.5%) and for all six host-S cells, intervals included.
   - My replay reproduces Table 19 exactly for both models. That replay is independent: I wrote it from the text of Appendix J.
   - Table 29, the pooled-MIN range of 4.5–18.6%, and the Eq. (5) implied-rate statistics (0.88–1.16; 31/39 within 6%; median |err| 1.7%) all match.

   For an empirical systems paper, this is the main reason to trust it.
2. **Registration is real and checkable.**
   - Every one of the 71 rentals of jobs 093–108 started at least 2.7 s after its job script was first committed. No script was modified after its rental began, judged by commit times.
   - The ledger confirms the 124 rentals, 74 offers and $87.2 that the paper reports.
   - Failures are reported prominently. Examples: three failed tests and one inconclusive for Eq. (5); the w16 ≥ 0.80 clause, missed on one machine; and host S's bands in Table 1.
3. **The machine is the unit, and that is justified with data.**
   - Machines carry 96% of the variance of the fetch gain and problems 1–3% (I reproduce 96.3/1.1 at 11% and 96.2/2.8 at 25%).
   - Relaunches of the same GPU agree within 3.2% in every pair I found. Same-CPU panel machines differ by up to 29%.
   - Summaries are bootstrapped over machines.
4. **Oracles run inside a real engine, not only in replay.** This lets the paper separate *which* experts are cached, *how many times* each is read, and *when* the read is issued. Its clearest finding is that MIN's set is worth almost nothing when read twice and about 39% of the gap when read once. Trace-only studies cannot show this.
5. **Honest statement of the bound's caveats.** The paper states that the bound is relative to the probe, assumes whole experts and exact routing, and that the probe's rate is exceeded by more than 5% on 12 of 84 launch-budgets by Eq. (5)'s own accounting. It also tightens the bound four ways (Table 28).
6. **Useful, concrete artifacts.**
   - A per-machine probe.
   - A fewest-admission schedule, which beats the deployed cache on all 9 stable machines (geometric mean 1.21×).
   - A horizon rule expressed in distinct experts, which connects to paging theory with lookahead.

---

## Weaknesses (most important first)

### W1. The headline decomposition splits a pure interaction 50/50 but is presented as separable contributions.

The abstract says "about a fifth of the gap is caching the wrong experts, a fifth reading admitted experts twice". Section 4 says that reading admitted experts twice "accounts for 19% of the gap at 11% … Any design that serves a miss on the CPU and then copies it pays this cost". Rule 2 of Section 7 says reading twice "costs about as much as caching the wrong experts".

On the raw data for the 13 fast-link machines at gpt-oss 11%:

| Change | Share of the gap closed |
|---|---|
| Changing only what is cached (MIN, 2 reads vs deployed) | **1.2%** (range −4 to 6) |
| Changing only how it is read (single read vs deployed) | **0.9%** (range −2 to 4) |
| Both together | **38.8%** |
| Interaction | **36.8%** |

So the two "fifths" are each half of an interaction. The deployed cache admits only about 3.5 experts per token, against about 60 misses per token (from `st_g_C14_base.json` on O4, O5 and Pa). Reading those admissions once is worth about 1%. The expensive double read is the double read of *MIN's* roughly 16 admissions per token, which no online policy makes.

Shapley crediting is a legitimate convention. The 2×2 is in Appendix F / Table 11, and Section 9 says the parts "interact". But:

- the abstract, Section 4's paragraph headings and Rule 2 all invite the reading that fixing either one alone recovers about a fifth;
- Fig. 2 (cache first, then read path) shows the opposite picture to Table 3 without saying why.

This is the paper's central result. Table 3 should carry the "alone" columns and the interaction. The abstract and Rule 2 should say that the read-once and MIN-set gains exist only jointly.

### W2. Two of the "five parts" are not measured, and the "largest part" is a constant inserted by definition.

- The "GPU's own work in series" (27% / 33%) is T_GPU = 2.94 ms. It is the *minimum* non-expert kernel time over 14 Nsight profiles taken on 7 hosts, most of which are not among the 15 decomposition machines. I reproduce 2.94 ms; the range is 2.94–3.38 ms and the median 3.16 ms.
- Its share is just 2.94/gap. With the median profile it becomes 29% at 11%, and the residual "left above Eq. (2)" falls from 20% to 18%. At 25% the numbers are 36% and 8%.
- The residual can go negative. On O4 at gpt-oss 25%, MIN prefetched runs at 0.99× Eq. (2) (residual −0.7%), because prefetching already overlaps part of T_GPU. In the prefetched state, then, "T_GPU fully in series" is not true. The serial and late parts overlap conceptually.

The text says "a profiled constant, not removed in the engine", which is honest. Even so, "The GPU's own work in series is the largest part" (heading), "the largest part" (Section 4) and Rule 5 ("A quarter to a third of the gap is the GPU waiting in series") present a definitional term as a finding. Relative to Eq. (2), which applies to every system the paper compares, the deployed cache runs at 55–70% (my computation, 39 consumer launches at 11%), not 31–54%. The "headroom is large" framing rests on Eq. (1), which no demand-reading system can approach.

### W3. Some evidence labels and scopes are generous, and the decomposition is post hoc without being called so.

1. **The decomposition was produced after all the data were in.**
   - `scripts/decomp_measured.py` and `paper/tab_dm.tex` were first committed 2026-10-07 14:18 −0500. The last data they use is from 101b, committed 2026-10-06.
   - The "fast link ≥ 0.5 of the CPU rate" threshold, which defines the 13-machine subset used in the abstract, was drawn after job 099's registered bands failed on exactly the two link-starved hosts (Appendix C).
   - Table 1 labels this "states registered; split derived". I would label it **exploratory (post hoc subset and split of registered runs)**.
   - In fairness, the numbers are robust. All 15 machines give 17/16/15/27/25% with 48% closed. Thresholds of 0.55, 0.6 and 0.7 give 21–24 / 20–23 / 11–14 / 26 / 15–19%.
2. **"Foresight pays when MIN's set is read once, where the link is fast" is labelled "registered; held at most cells".**
   - Job 099 registered "fetch/base 1.10–1.55 at both cells on every host" (prediction 6). It failed on the slow-link hosts; I measure 0.875 on Pf.
   - The qualifier "where the link is fast" is the post hoc rescoping that rescued the claim, and the label should say so.
3. **"The bound's read time is nearly reachable" is labelled "registered thresholds held".**
   - The registered thresholds were ≥ 0.80 (per-token) and ≥ 0.50 (per-layer). The 86–96% figure in the text is descriptive.
   - The second half of the same registered clause (per-layer at least 0.03 below the analytic split) failed on all hosts (Appendix C). "Thresholds held; the overhead clause failed" would be accurate.
4. **Rule 4 is stated as a builder rule without its exploratory status.**
   - The rule: "at a ratio of 0.32 or less, serve them on the CPU".
   - It rests on two *stable* machines, Pf (0.28–0.29) and the TR 9960X (0.32). Job 107 registered the crossover, but only machines at 0.44 and 0.75 were valid. The only sub-0.35 machine there (EPYC 7663, 0.21) ran unsteadily.
   - The introduction states the 0.32 threshold as a result. Section 9's "rests on few points" should travel with the rule.
5. **Section 6's unlabelled paragraphs are deterministic replays.** "Next-token prediction buys little" and "Accuracy matters as much as horizon" are replay results (Table 19), and the values were already printed in job 099's header before launch. They are correct (I reproduce them exactly) but are neither registered nor engine measurements. A "replay (derived)" tag would keep Table 1's taxonomy consistent.

### W4. One comparison in the introduction mixes two scorings.

The introduction says "[Our cache] runs at roughly a third to a half of the bound's speed … **Scored the same way**, published batch-1 systems reach a median 13.6% of theirs." The two are not scored the same way:

- The "third to a half" is against the *probed* rate on the authors' machines.
- The 13.6% is against *datasheet* ceilings.

Scored the same way, at datasheet rates, the authors' cache reaches a median of 27% (range 18–42%), as Section 3 correctly states. The introduction should quote 27% against 13.6%.

The audit also scores pooled-cache, whole-layer-pinning and speculative systems against a per-layer one-token bound that is not theirs. Appendix N acknowledges this, but uniform-routing rows still enter the "all rows" median. I verified the 20-row in-class median of 13.6% only from Table 31's printed percentages, not from the raw audit.

### W5. Narrow empirical scope behind general "rules for system builders".

- **Hardware:** one GPU model (RTX 5090), with a 4090/3090 grid in the appendix.
- **Decomposition:** one model at two budgets, on fast-link machines only.
- **Workload:** batch 1, teacher-forced 256-token AIME decode, with prompts that were also used during development.
- **Prompts and batches above 1** bypass the cache and run at 77–83% of stock (Appendix M).
- **Server processors:** Eq. (5) fails there, and only 3 of 8 server launches passed the checks (confirmed).
- **TR 9960X:** this 24-core HEDT part counts as "consumer" and sets the 31% low end of the 31–54% range (30.6% on its 106a launch).

Section 7 should carry the scope of each rule, for example "on RTX 5090 desktop hosts with fast links, gpt-oss".

### W6. The practical pay-off without foresight is small, and the big gains need foresight nobody has.

The 39% joint gain needs MIN's set. The realisable moves the paper measures are:

- the admission margin, at 1.021× and 1.018× (I reproduce both geometric means);
- the layer-ahead copy, which loses on slow links, down to 0.72× at 11% and 0.69× at 25%;
- forecasters, which close at most 15% of the read gap.

The paper says this candidly ("Nothing realisable reaches it"). Still, it limits the significance of the "how foresight must be spent" contribution to a statement about oracles. The fewest-admission result (1.21×) is likewise an oracle result.

### W7. Smaller technical points

- **"Bound" is a misnomer when the engine can exceed the probe.** By Eq. (5)'s accounting, which the paper itself does not consider confirmed, the deployed cache's implied read rate reaches up to 1.22× B_host at 25%. A one-run probe maximum is a reference rate, not a hardware limit. I suggest "probe-relative bound" in the abstract.
- **Eq. (1) assumes whole-expert, fixed-precision reads.** Systems that read partial experts, keep compressed CPU copies, or hit in a large L3 (X3D parts) are outside it. The authors state this; the abstract's "bound every system" is broader.
- **Fig. 3's caption is ambiguous.** It says "MIN, 1 read gains at every budget on these machines" but also covers the panel ticks, and one panel machine (Pf) is at 0.875 at gpt-oss 11%. The statement holds only for O3–O5.
- **Section 4's closed-form paragraph interrupts the section.** It sits inside the measured decomposition, which "does not rely on it". It belongs in Appendix K alone.

---

## Clarity

Earlier rounds scored clarity 2, then 3 three times. The restructure helped: the main text now has a clear arc.

### What works

- **The introduction asks the right questions in plain words.** How fast could it be? Where does the time go? What is foresight worth? The four bold "answer" paragraphs map one-to-one onto Sections 3–6.
- **Table 1 (claims × evidence × section) is a genuinely good device.** I wish more papers had one. Its labels need the fixes in W3.
- **Table 2 gives one vocabulary for every configuration.** Appendix B maps the codes. The main text uses the names consistently ("MIN, 1 read", "fewest-admission", "deployed").
- **The running example in Section 5 is the most readable passage.** It gives one machine, one budget, and five times in ms (10.0 / 13.0 / 20.6 / 15.2 / 13.8, all of which I reproduce).
- **Fig. 1 makes the double read visible.** Fig. 2 (the staircase), Fig. 4 (gain against link ratio) and Fig. 5 (share against W) are each readable on their own.
- **Limitations are grouped by topic** (hardware, workload, bounds, decomposition, oracles, outputs), and the related-work positioning is concise.

### What still makes it hard to read (most important first)

1. **Fig. 2 and Table 3 tell different stories about the same data.** Fig. 2 removes the parts in one order: in it, MIN's set alone does nothing at 11% and reading once does most of the work. Table 3 reports Shapley shares that give each about 20%. A reader meets the interaction (the actual finding) only in Appendix F. Put "alone / alone / interaction" next to the Shapley values, or draw both orders in Fig. 2.
2. **Number density and mixed scopes.** Nearly every sentence carries two to four numbers, and adjacent numbers often differ in more than one way at once:
   - reads (replay) versus time (engine);
   - all machines versus fast-link versus stable;
   - 11% versus 25%;
   - geometric versus arithmetic means.

   Examples: "0.60–0.73 of the greedy one's copies for the same hits in replay (in the engine it misses 2.5–5.3% more)". And "lost at 11% (0.85–0.93×). There the fewest-admission set pays … 1.11–1.22×", where the second range spans *both* budgets (1.11 at 11%, up to 1.22 at 25%) but reads as 11%.
3. **Four GPU-time constants with different meanings:**
   - T_GPU = 2.9 ms (non-expert, minimum profile);
   - G = 4.1–4.7 ms (all GPU kernels, Eq. (5));
   - G_fit = 5.17 ms (Eq. (3), fitted);
   - g = 37–48 µs per layer.

   Appendix D has to warn that G_fit "is neither the profiled G of Eq. (5) nor T_GPU". "Bound" and "limit" are also used interchangeably in the appendices (Table 7 concedes this).
4. **Label and cross-reference errors that a careful reader trips on:**
   - Tables 23, 24 and 25 head their prediction column "Eq. 3" (hard-coded in `tab_job105/106/107.tex`). The captions say the column is Eq. (5), and Eq. (3) is a different, fitted model.
   - Appendix M says "Section 3 gives the link-aware bound". Section 3 does not.
   - Appendix I sends the reader to Section 6 for the link dependence of admit-every-miss. It is in Section 5 and Appendix J.
   - Appendix L cites Section 4 for job 107's server being under-predicted. Section 4 does not say that.
   - The abstract's "about 0.65 C distinct experts … across 9 models" and the conclusion's "0.66–0.81 C … on our workload" are two different quantities written with near-identical phrasing.
5. **The appendix sprawl is not reviewable in finite time.** There are about 30 pages, 31 tables and 12 figures. Machine codes multiply: hosts A, B and S; O1–O6; Pa–Pj; "Pf again"; and job numbers. Appendix C is a 1.5-page run-on paragraph of clause failures.
   - The handful of tables a reader needs (11, 19, 22, 29) are buried among job-by-job logs that belong in the supplement.
   - The main text cites Table 23, Table 28 and Appendix L for claims stated there, so the reader is pulled into this material constantly.
6. **The deployed cache's per-machine FETCH split is a key mechanism that the main text never explains.** Each machine's table sends some misses over PCIe in the step and runs the rest on the CPU. Fig. 1 shows only "misses: read and run". Several main-text results depend on the split: the slow-link behaviour, the "single read" configuration, and host S using host B's FreeToken backend. The split is defined only in Appendix D, as Eq. (4).
7. **Table 4 is hard to scan.** It has 12 numeric columns, two bound variants, inline backend labels, a † footnote, a pooled-bound range in the caption, and a header with each host's five probe rates. It could be split: speeds and ratios in one table, bounds in another.
8. **Small items:**
   - Table 3 is placed before Section 4 begins.
   - Fig. 3's caption promises "95% interval" bars that are invisible at print size.
   - The abstract's fractions ("a fifth, a fifth, a sixth, a quarter, a fifth") are hard to hold in mind; a five-segment bar would do better.

**Net:** a clearly better paper to read than its description in earlier rounds suggests, but still not effortless, and item 1 is misleading rather than merely dense. **Clarity 3/5.** Fixing items 1, 3 and 4 and moving most of Appendices C, H and L to the supplement would earn a 4.

---

## Questions for the authors

1. Will you report, in Table 3 and the abstract, the "alone" effects of MIN's set (1.2% of the gap at 11%) and of reading once (0.9%), next to the interaction (36.8%)? For a builder *without* foresight, is there any configuration in which reading the deployed admissions once pays more than about 2%?
2. Why the *minimum* of 14 profiles for T_GPU, rather than each machine's own profile or the median? How do you interpret the negative residual on O4 at 25%, where MIN prefetched runs at 0.99× Eq. (2)? Does it not show that the prefetched state already overlaps part of T_GPU, so that the "serial" and "late" parts are not disjoint?
3. When was the 0.5 fast-link threshold chosen, relative to the job-099, 100 and 101 results? Would you relabel Section 4's split as exploratory in Table 1?
4. The introduction compares a probed-rate fraction (31–54%) with a datasheet-rate median (13.6%) as "scored the same way". Will you replace this with 27% against 13.6%?
5. Rule 4 (ratio ≤ 0.32 → CPU then background copy) rests on two stable machines. Do you have any stable machine between 0.33 and 0.43 that ran both fewest-admission loads?
6. Eq. (5) implies that the engine reads above B_host by up to 1.22× on some launches. If that is real, Eq. (1) is not a bound on those machines. If it is not, Eq. (5) is wrong there. Which is it, and what would a longer or multi-run probe give?
7. Why are the server-class failures (implied rate 0.27–0.72 of B_host) unexplained, given that the per-layer timers exist? Is it NUMA, the container's core limit, or the helper count?

## What would raise my score

- **Fix the presentation of the decomposition (W1).** Show the 2×2 alone and interaction effects in Table 3. Make Fig. 2 consistent with Table 3. Reword the abstract and Rule 2 so they do not imply separable gains.
- **Treat T_GPU as a sensitivity (W2).** Use per-machine or median profiles with a band. Drop "largest part" as a finding, or support it with a state in which the GPU work is actually overlapped. Report Eq. (2) fractions alongside Eq. (1) in the headline.
- **Correct the evidence labels (W3).** Mark the split and the fast-link subset as post hoc. Mark the link qualifier on "foresight pays" as a post hoc rescoping. Say "thresholds held; overhead clause failed" for reachability. Tag replay results as derived.
- **Fix the introduction's "scored the same way" comparison (W4)**, and scope each Section 7 rule to the evidence behind it (W5).
- **Fix the label and cross-reference errors** (the "Eq. 3" headers, and the Section 3, 4 and 6 references), and **move the job-by-job material** of Appendices C, H and L to the supplement.

If the first three were done, I would move to 7.

---

## Claims checked against raw data

All values are from my own code on the raw files. "11%" and "25%" mean gpt-oss C = 14 and 32. Eq. (1) uses my own R\* and the highest single probe reading in each machine's `concur.txt`.

| # | Claim | Location | My value (raw) | Verdict |
|---|---|---|---|---|
| 1 | R\* (MIN with bypass, cache carried over) and the −0.4% change on the first 20 problems | Sec. 3, Sec. 9 | 38.315 (C14), 15.340 (C32), 30 problems; first 20: −0.38% at both | ✔ (matches the artifact's "exact" value) |
| 2 | T_GPU = 2.9 ms, "the smallest of our engine's Nsight profiles" | Sec. 3, Eq. (2) | min 2.94 ms over 14 profiles (range 2.94–3.38, median 3.16) | ✔ (min is a choice; see W2) |
| 3 | G = 4.1–4.7 ms; median of the 7 profiles 4.28 / 4.50 ms | App. K, L | 4.07–4.69; medians 4.28 (C14), 4.50 (C32) | ✔ |
| 4 | Table 3, 11%: 20 [16,23], 19 [16,23], 15 [11,18], 27 [24,29], 20 [15,25] | Table 3, abstract | 19.5 [15.8,22.9], 19.3 [15.8,22.8], 14.6 [10.6,18.3], 26.9 [24.0,29.5], 19.6 [14.6,25.1] | ✔ exact |
| 5 | Table 3, 25%: 29 [27,31], 7 [3,10], 20 [18,22], 33 [30,36], 11 [7,15] | Table 3 | 28.6 [26.6,30.5], 6.7 [3.3,9.9], 20.5 [18.2,22.3], 33.4 [30.5,36.0], 10.9 [6.6,15.2] | ✔ exact |
| 6 | 15 machines with all states, 13 fast; slow-link ratios ≤ 0.40 | Sec. 4, Fig. 2 | 15 (one launch per GPU, jobs 093–104); 13 at ≥ 0.5; slow 0.29, 0.40 | ✔ |
| 7 | Best oracle closes a mean 54% (49–58) on fast links; ≤ 17% on slow links at 11% | Sec. 4 | 53.8% [49.4, 58.1]; slow max 16.8% | ✔ |
| 8 | All three parts removed: 1.07–1.38× Eq. (2) at 11% | Sec. 4 | 1.07–1.38 (at 25%: 0.99–1.32, O4 below Eq. (2)) | ✔ (25% note: W2) |
| 9 | "With two reads, MIN's set is no faster than the deployed one at 11%" | Sec. 4 | cache alone 1.2% of gap; load alone 0.9%; joint 38.8%; interaction 36.8% | ✔, but shows W1 |
| 10 | Ours at 31–54% of Eq. (1) at 11% on 25 consumer machines (39 launches) | Abstract, Sec. 3 | 39 launches, 25 GPUs; 30.6–54.0% (min = TR 9960X) | ✔ |
| 11 | Table 4, host B: 69.9/54.0/34.9, ratio 1.294 [1.278,1.312], …; host S all six cells | Table 4 | identical speeds, ratios and CIs (host B gpt-oss ×3 and Qwen3 12.5%; host S ×6) | ✔ exact |
| 12 | Leads FreeToken at 11 of 12; 2.0–4.0× llama.cpp; bounds 172/430 (B), 140/351 (S) | Sec. 3, Table 4 | 11/12 (only Qwen3 43.75% on S, 0.974); 2.00–4.02×; 172.3/430.4, 140.4/350.7 | ✔ |
| 13 | FreeToken tuned trails at 5 of 6 cells | Sec. 3, App. M | 1.172, 1.168, 1.094, 1.021, 1.044, 0.983 | ✔ |
| 14 | Read time reachable: 86–96% per layer (Table 29) | Sec. 3, Table 29 | per layer 86/89/96/96/95/93%; per token 93–97% | ✔ (registered bars were 0.80/0.50) |
| 15 | Running example: Eq. (1) 10.0, Eq. (2) 13.0, deployed 20.6, MIN 1 read 15.2, ahead 13.8 ms | Sec. 5 | 10.02, 12.96, 20.64, 15.16, 13.81 | ✔ |
| 16 | MIN 1 read +16–51% at every budget on O3–O5; up to +81% read ahead | Sec. 5 | +16.0% to +51.4%; max ahead +81.2% (O4, 25%) | ✔ |
| 17 | Usual ways (2 reads / no bypass) gain at most 15% or lose at the smallest budgets | Sec. 5 | max 1.148 (no bypass 1 read, O4 Qwen3 12.5%) | ✔ |
| 18 | Fewest-admission: 0.60–0.73 of greedy copies; +2.5–5.3% misses in the engine | Sec. 5, App. L | engine 0.597–0.726 copies, +2.5% (11%) / +5.1–5.3% (25%) misses | ✔ |
| 19 | Fewest-admission 1 read beats deployed on all 9 stable machines, geometric mean 1.21×; 1.17× over 13; losses = 2 slowest links | Sec. 5, Table 1 | 9 stable all > 1 (1.03–1.42), geometric mean 1.215; 13 → 1.166; losses EPYC 7302 (0.845, ratio 0.14), EPYC 7663 (0.974, 0.21) | ✔ (7 registered: checked headers 104–107) |
| 20 | Spearman 0.89 [0.63, 0.97] over 19 machines (gain against link ratio) | Sec. 5 | 0.89 [0.62, 0.98], n = 19 | ✔ |
| 21 | Machines carry 96% of the gain's variance, problems ≤ 3% | Sec. 5, App. L | 96.3% / 1.1% (11%), 96.2% / 2.8% (25%) | ✔ |
| 22 | Ratio 0.28–0.32: greedy in step lost (0.85–0.93×); fewest by CPU 1.11–1.22× | Sec. 5 | greedy 0.853–0.933 at 11%; fewest by CPU 1.11 (11%), 1.16–1.22 (25%) | ✔ (range spans both budgets) |
| 23 | Admission margin: 1.021× (11%), 1.018× (25%), geometric means over 5 stable machines | Sec. 5 | 1.0213, 1.0177 | ✔ |
| 24 | Layer-ahead copy 1.04× on fast link, down to 0.72× at 11% | Sec. 6 | 1.042 (O4); 0.722 (Pf, 11%); 0.691 at 25% | ✔ (25% is lower) |
| 25 | Exact next-token share 0.18 / 0.07, ≤ 0.31 at any host-bound budget; W50 = 4 and 10 tokens | Sec. 6, Table 19 | own replay: aa 58.6 / MIN 39.3, shares 0.18/0.31/0.52/0.81/0.98; 25%: 0.07/…/0.66; Qwen3 0.31 max; W50 3.7, 10.2 | ✔ exact |
| 26 | Engine: exact 16-token window recovers 0.78–0.93 at 11%; registered ≥ 0.80, lowest missed | Sec. 6 | 0.78–0.93 over 10 panel machines; one at 0.78 | ✔ |
| 27 | D(W50)/C on the AIME routing = 0.66–0.81 | Sec. 6, 10 | 0.66–0.80 (window-boundary handling) | ✔ (≈) |
| 28 | Pooling slots lowers MIN's reads by 4.5–18.6% | Sec. 9 | gpt-oss 5.5/9.2/12.6%; Qwen3 4.5/8.5/18.6% | ✔ |
| 29 | Implied read rate 0.88–1.16 of B_host over 39 consumer launches; Eq. (5) within 6% on 31/39, within 8% on 37/39, median \|err\| 1.7%; servers 0.27 / 0.58 / 1.04 | Sec. 4, App. K, Table 27 | 0.88–1.16; 31/39; 37/39; 1.7%; ES 0.27, 7543 0.58, 7402P 1.04; up to 1.22× at 25% | ✔ |
| 30 | Same-CPU panel machines differ by up to 29%; relaunch within 3.2% | Sec. 2 | 285K pair 18.14 vs 14.06 ms (1.29); max relaunch spread 3.2% (TR 9960X) | ✔ |
| 31 | Every commit preceded its rental by ≥ 2.7 s; 124 rentals, 74 offers, $87.2 | App. C, O | min gap 2.71 s over 71 rentals of jobs 093–108; no script modified after its rental started; 124 / 74 / $87.2 | ✔ (git timestamps only; push records not checkable offline) |
| 32 | 8 server launches, 3 passed every check; jobs 107/108 rented 7 / 6 | Sec. 9, App. K | 8 server launches, valid: 7402P, ES, 7543; 7 and 6 rentals in the ledger | ✔ |
| 33 | Clause totals: 565 for 073–098 (218/193/142); overall Table 8 | App. C, Table 8 | rows sum to 218/193/142/12 and 495/597/308/83 (1,483) | ✔ internally consistent |
| – | "Scored the same way, published systems reach a median 13.6%" against ours at a third to a half | Sec. 1 | 13.6% (20 in-class rows, from Table 31's percentages) is datasheet-scored; ours datasheet-scored = 27% | ✘ wording (W4) |
| – | Tables 23–25 prediction column "Eq. 3" | App. L | the column is Eq. (5) by caption and data (my Eq. (5) errors reproduce Table 27) | ✘ label |
| – | Fig. 3 caption: "MIN, 1 read gains at every budget on these machines" | Fig. 3 | true on O3–O5; panel Pf 0.875 at 11% | ✘ scope |

**On scoping and post hoc labels:**
- **Correctly labelled:** Section 4's closed-form paragraph ("exploratory, not confirmed"); Section 5's link-ratio claim ("exploratory"); Section 6's horizon rule ("exploratory").
- **Under-labelled:** the Section 4 split and its fast-link subset (post hoc, created after all data); the link qualifier on "foresight pays"; the 0.32 rule in Section 7; and the replay-based Section 6 paragraphs.

---

## Scores

| Criterion | Score |
|---|---|
| **Overall** | **6 / 10** (weak accept) |
| Soundness | 3 / 5 |
| Significance | 3 / 5 |
| Novelty | 3 / 5 |
| Clarity | 3 / 5 |
| Confidence | 4 / 5 |

**Justification.**
- **Soundness.** The empirical record is the strongest I have audited at this venue: every number I checked reproduces from raw data, and registration is verifiable. I hold soundness at 3 because the headline decomposition is interpreted beyond what a Shapley split of a near-pure interaction supports (W1), its largest part is an inserted constant (W2), and several labels and scopes are generous (W3, W4).
- **Significance.** Real for builders of consumer MoE offloading, but narrow in scope (W5). The large gains require foresight that nothing realisable provides (W6).
- **Novelty.** Moderate. The bound is Belady plus roofline. The in-engine oracle accounting, the fewest-admission schedule and the distinct-expert horizon rule are new applications with good precedent.
