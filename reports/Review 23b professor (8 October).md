# Review 23b: "Where the Seconds Go" (systems professor, scientific benchmarking)

Reviewed as a workshop and fellowship referee would, in the spirit of Hoefler and Belli (SC'15).

## Materials read

- `paper/paper.pdf` (44 pages, MLSys style). I read the main text (pp. 1–11) in full and Appendices A–N almost in full (map, checklist, names, prediction log, oracle jobs, trace study, partial foresight, closed form, schedules, cache detail, related work, limitations, audit, artifact).
- LaTeX sources: `paper.tex`, `paper_ieee.tex`, `main_body.tex`, `abstract_body.tex`, `macros.tex`, `names.tex`, plus a grep of `wsg_dm.tex` and `wsg_job109.tex` for macro definitions.
- `paper/supplement.pdf` (55 pages): skimmed for the structure of the scorecard and the section list.
- `paper/paper_ieee.pdf` (39 pages): extracted the text, and rendered and inspected pages 1, 4, 9, 13 and a 12-page overview. Read `paper_ieee.log` for font warnings.
- Artifact, read to learn definitions only (no results were reused):
  - `gpu/vast_ledger.json` and `prereg/gpu_pushes.json`;
  - parts of `scripts/reanalysis.py`, `speed_limit.py`, `decomp_measured.py`, `sumlaw_paper.py` and `hostdep_model.py`, and a grep of `job109.py`;
  - `jobs/ec2/fetch_table.py`, `gpu/vast_boot.sh` and `jobs/ec2/job110_uuids.txt`.
- Raw data on the gpu branch (`/home/claude/gpu-branch`): job script headers 109, 110, 111 and 112 and their wrappers, and `results/` for jobs 081, 089, 090, 096a/b, 099a–j, 100b/f, 101b, 102a–c and 103–112. From these I used:
  - `ec_*.jsonl` (decode_ms, n_decode, nll_sum, nll_n, config `stats=`);
  - `st_*.json` counters;
  - `concur.txt` and `concur2.txt`;
  - `fetch_table_law_gptoss.json`;
  - `prof_*.json` and `g_prof.json`;
  - `bs1.jsonl` (jobs 081 and 089) and `parity_kl.txt` (job 090);
  - `v0.txt`, `validity.txt`, `DONE`, `skipped.txt` and `manifest.json`.
- Git: `git log --format='%h %ct'` per job script (no messages), `git show --name-only --format=''`, and `git diff` of the five post-launch amendment commits (content only).
- All my code is in `scratchpad/rev23b/` (`lib.py`, `j109.py`, `j109b.py`, `j111.py`, `panel.py`, `fewone.py`, `share.py`, `headline.py`, `timing.py`, `pushes.py`).

## Independence statement

- I did not open anything under `reports/` except to write this file.
- I did not open:
  - `prereg/*outcome*.md` (I saw their names in a directory listing only);
  - `research_notes/` or `apply/`;
  - `paper/paper_v1_prereview.tex`;
  - any file named like a review, number check, plan or progress log.
- I read no commit messages. The job 112 header mentions "reviews 21a and 21b"; I did not open them.
- All scripts are mine, written in a new folder and run from it.
- I modified nothing in either repository and committed nothing.
- Every value in the claims table below was recomputed by my own code from the raw rows, counters, probes or profiles. Where I needed a definition (for example $B_{\text{host}}$, the fetch-table reading of $B_c$ and $B_p$, or how the non-expert time $T_{\text{GPU}}$ is computed from a profile), I read the authors' function and then reimplemented it.

## Summary

The paper studies batch-1 decode of host-offloaded MoE models (mainly gpt-oss-120b at 11% and 25% of experts resident) on rented RTX 5090 machines. It has three parts.

- **A bound (Eq. 1).** MIN-with-bypass host reads $R^\star$ at the machine's highest probed rate, with a demand variant (Eq. 2) and an ordered variant (Eq. 3).
- **A decomposition of the authors' llama.cpp cache's time beyond that bound,** using oracles inside the engine. "What is cached" (MIN's set) and "how it is read" (once, in the step, versus served by the CPU and then copied) are changed alone and together in a 2×2. The read-ahead oracle is then added and the remainder is attributed.
- **Several follow-ons:**
  - variants without foresight recover ≤6% of the oracle gain;
  - a link-to-CPU-ratio trend fitted on RTX 5090s predicts RTX 4090 oracle speeds within 7%;
  - a horizon rule of about 0.65·C distinct experts per layer;
  - a scoring of 20 published measurements against their own bounds.

Every experiment from job 073 on carries predictions committed before its machines started. The paper scores all of them (1,672 clauses: 588 held, 679 held on the point estimate, 321 failed, 84 untested or void).

## Strengths

1. **The registration practice is exemplary and checkable.** I verified these against git, the ledger and GitHub push records:
   - Of the 92 rentals of jobs 093–112, every relevant job-script commit precedes its rental creation, by at least 2.7 s.
   - Every machine started its job at least 26 s after the GitHub push.
   - Seven rentals fall within 1 s of the push, and four pushes are stamped at or after the rental creation. This is harmless, because the machine clones at boot.
   - Every post-launch header edit I found is disclosed in Appendix D:
     - machine substitutions in jobs 102, 105 and 106, with predictions unchanged;
     - the added prediction 9 in job 100;
     - the gate and replacement amendments of job 107;
     - the relaunch amendment of job 111.
   - Failed predictions are reported in full, with a hand-versus-script re-scoring check. This is rarer than it should be in systems papers.
2. **The artifact is complete and the numbers are faithful.** I made 31 checks covering several hundred printed values (table below). Every table cell I recomputed reproduced to the printed precision: Tables 3 (host S, and host B's gpt-oss rows), 4 (both columns), 5, 6 and 30. Two cases are imprecise wording rather than wrong numbers (rows 7 and 17). The whole campaign cost $105.8 by the ledger, which I also confirmed.
3. **Validity gates are fixed in advance, and failed hosts are reported, not hidden.**
   - The gates are: GPU UUID novelty, host memory in use, NUMA and core class, device read rate, the loss reference, round-to-round agreement, and per-configuration loss parity.
   - The raw records match the paper's accounting:
     - job 107: 3 stopped at the gate, 2 failed the round check, 2 were valid;
     - job 108: 2, 1 and 3;
     - job 110: one machine at a ratio of 0.2499 against the 0.25 gate, and four V1 failures at 0.1963–0.1965;
     - 112c: in the ledger, with no results.
   - Job 110 is reported as void rather than quietly re-run.
4. **The machine is the statistical unit, and the choice is justified with data.** Two Core Ultra 9 285K panel machines differ by 29% in the deployed cache's time (18.14 vs 14.06 ms; reproduced), while relaunches agree within about 5%. Summaries over machines use t-intervals, and the figures plot every machine.
5. **The framing is useful.** The bound needs only a routing trace and a probe. The oracles run inside a real engine rather than in a simulator. Table 1 is an explicit claim/evidence ledger that separates pre-specified, post hoc and derived claims. Appendix B walks through Hoefler and Belli's 12 rules, including where the paper departs from them (rule 3).

## Weaknesses (most important first)

### W1. The central 2×2 is not a clean factorial, and the control meant to repair it failed its manipulation

"MIN's set read twice" (MIN-2R or Few-2R) does not hold MIN's set in this engine. On the four job-112 machines at 11%:

- Few-2R misses 54.7–55.7 experts per token against Few-1R's 40.7–40.9 (my counters), i.e. 34–37% more.
- The cause is that background copies publish two steps late, and the per-machine fetch table keeps fetching 17.8–23.7 other misses per token into slots.

The registered timing control (Few-2R-early) was designed to remove the late landing, but it barely changed the misses:

- registered: at most 0.95 of Few-2R's; the replay predicted 0.91;
- measured: 0.968–0.980 at 11%;
- T1 failed on 5 of 8 machine-budgets and T2 on 7 of 8, as the paper says.

The text concedes this candidly in Section 4: "the control thus failed to give the two-read path MIN's set, so it cannot say what that set would gain read twice". The Limitations section says it too. However:

- the abstract says "caching the experts MIN would keep pays only if each admitted expert is also read once";
- Table 1 lists "Landing the copies in time does not rescue the two-read path" as a claim.

What the data support is narrower: on this engine's two-read path, with its publication delay and fetch table, MIN's admission decisions gain nothing. The Table 1 row rests on a control whose manipulation check failed. It should read "inconclusive for the intended question", and the abstract should say "on this engine".

### W2. About half of "where the seconds go" is attributed, not measured

The measured oracle states remove 48–54% of the gap (Table 4). The rest is split arithmetically, using:

- $T_{\text{GPU}}$ = 2.94 ms;
- the read-ahead oracle's extra reads at $B_{\text{host}}$.

The abstract says "Oracles inside the engine account for the rest of the time", which overstates this. Two details matter:

- 2.94 ms is the smallest of a fixed set of 14 profiles (jobs 069c and 105). It is not "the smallest of our engine's Nsight profiles": over all 47 RTX 5090 profiles the minimum is 2.87 ms (109d) and the median 3.16 ms.
- On slow links the residual is 31–47% of the gap (reproduced), so the attribution does not close there.

The paper should:

- say "oracles remove about half; the rest is attributed" in the abstract;
- report the attributed rows of Table 4 with a $T_{\text{GPU}}$ band (minimum, median, per-machine), as it already does for the probe.

### W3. Uncertainty within a machine is understated, and the scoring rule relies on it

Per-machine intervals are paired bootstraps over problems only, which leaves out two larger sources of variation:

- **Process to process.** Ratios can differ by up to 0.02 between rounds on the same rental (109d MIN-1R: 1.237 vs 1.257). That exceeds Table 6's "±.01".
- **Rental to rental.** The deployed cache's time moves by up to 5.4% (110c to 111g), and the probe's link-to-CPU ratio by up to 11.8% (109f to 112g: 0.837 to 0.738).

Because "held with an interval" uses the problem bootstrap, "held" is easier to reach than readers will assume. The ratio's test-retest error also matters for the fast/slow line at 0.5: the two Core Ultra 9 285K machines read 0.49 and 0.54. Two-stage intervals (rounds, then problems), which the paper already used for job 106, should be standard wherever rounds exist.

### W4. A universal claim is scoped by a filter that coincides with the confound

"Few-1R gains on every machine that ran it stably" (13 machines, geometric mean 1.24×, t-interval 1.16–1.32; reproduced):

- The two machines excluded as unsteady are exactly the two slowest links: EPYC 7302 (ratio 0.14, 0.845×) and EPYC 7663 (ratio 0.21, 0.974×). The paper's own read-path argument predicts losses there.
- The 13 machines include two RTX 4090s and two server CPUs. Section 5 does not say so.

The honest statement is: "gains on every machine with a ratio of at least about 0.28 that passed the checks; lost on the two slowest links, both of which also failed stability". The paper discloses the facts, but the headline is phrased universally.

### W5. The registration is tight in mechanism but loose in content for the headline numbers

- The abstract's 35% was registered as a band of [0.25, 0.55]. The ≤6% capture was registered as ≤35%. The read-time claim was registered as ≥50% and measured at 86–96%. The paper itself admits several of these "could hardly have failed". The confirmatory content of job 109 is mostly the interaction's sign on every machine and the per-machine bounds.
- The 0.5 fast-link line and the consumer/server scope were drawn after the data.
- The RTX 4090 test:
  - three of its five machines are relaunches of machines whose first-round results were already known;
  - 110e was dropped by an oversight. Its UUID is absent from `job110_uuids.txt`. Its single round shows the read-ahead oracle at only 1.056× and Margin capturing 78% of that gain, far outside every reported machine's range, and Table 6 does not show it.

The abstract should present its numbers as estimates from registered experiments, not as the registered predictions themselves.

### W6. Generality is narrow for the significance claimed

- The decomposition covers two budgets of one model, one engine (a 3,500-line llama.cpp patch) and one workload. The workload is AIME prompts, also used during development, teacher-forced for 256 tokens. The machines are a convenience sample of the cheapest rental offers.
- Several "rules for builders" depend on engine-specific mechanisms: the two-step publication delay, the fetch table and the queue cap.
- The ratio trend misses new fast-link RTX 5090s in 5 of 18 cells, by up to 17.6% (reproduced).
- The 0.65·C horizon is a trace-level result in host reads, not in time.

### W7. The bound depends on one probe, and the literature audit compares unlike rates

- $B_{\text{host}}$ is the maximum of one probe run, taken while the model downloaded on some machines. The engine implies a rate more than 5% above it in 12 of 84 launch-budgets.
- The low end of "31–54%" comes from a Threadripper 9960X probed at 178 GB/s (31.2%; reproduced). Counting it as "consumer" is a post hoc choice.
- The audit scores published systems against datasheet rates, with uniform routing for 11 rows, and puts the result next to the authors' probed numbers. It is labelled post hoc, but it is weak support for "the field leaves most of the machine unused".

### W8. Smaller accuracy and provenance points

- **Section 6.1 wording.** "They still read 1.66–1.68 R*" is computed from LA-1R and LA-1R-margin only. Margin reads 1.575–1.588 R* and LA 1.751–1.755 R*.
- **Provenance.** The results do not record which gpu-branch commit each machine cloned (no `git rev-parse` in manifests or stdout), and do not hash the job script. Provenance is inferred from timing alone.
- **Rounding.** I get a 5.4% RTX 4090 relaunch difference against the paper's 5.5%.
- **Naming.** The parity job directory is named `parity_a100` but ran on an RTX PRO 6000.
- **Page limit.** The MLSys build's main text overflows the 10-page limit: the Conclusion's last six lines are on page 11.

## Claims checked against raw data

Unless stated, X/base is the deployed cache's mean time per token divided by X's, from the same process, as a geometric mean over rounds. $R^\star$ is 38.315 or 15.340, S = 13,253,760 B, and $B_{\text{host}}$ is the maximum probe reading.

| # | Claim (where) | Raw data and my computation | Verdict |
|---|---|---|---|
| 1 | Table 6, job 109: MIN-1R, ahead, best variant without foresight, and capture, at 11% and 25% | `ec_g_C*_r*{A,B}.jsonl`. At 11%: 285K 1.215 / 1.460 / 1.021 (Margin) / 4.5%; 13900KF 1.247 / 1.458 / 1.007 / 1.5%; 5950X 1.366 / 1.500 / 1.010 / 1.9%; 9800X3D 1.154 / 1.342 / 1.014 / 4.1%; 7945HX 1.241 / 1.416 / 1.024 / 5.7%. At 25%, e.g. 7945HX 1.283 / 1.626 / 0.995 / −0.8% | Reproduced (all 36 values) |
| 2 | Link-to-CPU ratios 0.54–0.89 (Table 6) | `concur.txt` read by the fetch-table rule ($B_c$ interpolated at the helper count, $B_p$ the maximum zero-copy 16 MB reading) | Reproduced |
| 3 | Table 4, New column (8 rows × 2 budgets, t over 5 and 4 machines) | Both 34.7 [21.9, 47.5]; left 46.3 [37.7, 54.8]; T_GPU 28.2 [20.3, 36.1]; residual 0.5 [−3.1, 4.2]; at 25%: Both 27.3 [23.0, 31.5], residual −5.1 [−7.4, −2.9] | Reproduced |
| 4 | Table 4, Panel column (15 machines) | 096a/b, 099a–j, 100b/f, 101b: Both 32.6 [21.0, 44.3]; left 52.5 [42.6, 62.3]; residual 3.7 [−6.0, 13.3]; all 25% rows too | Reproduced |
| 5 | Interaction positive on 9 of 10 (job 099) and 5 of 5 (job 109) | 099f (285K, ratio 0.29) −0.48 ms; every job-109 cell positive (0.5–7.5 ms) | Reproduced |
| 6 | Best variant without foresight 1.007–1.024×, capture ≤6% at 11% | Job 109 | Reproduced (maximum 5.7%) |
| 7 | "They still read 1.66–1.68 R*" (Section 6.1) | Counters: LA-1R 1.665–1.667; LA-1R-margin 1.675–1.678; Margin 1.575–1.588; LA 1.751–1.755; deployed 1.645–1.660 | Range is LA-1R and LA-1R-margin only (wording) |
| 8 | Table 6, RTX 4090 rows and bracketed trend predictions | 111d/f/g and 112a/b, with the frozen coefficients from the job 110 header | Reproduced |
| 9 | Trend within 7% at 11% and 5% at 25%; Dep-1R and MIN-2R within 0.06 | Maximum \|Δln\| 0.064 and 0.049; Dep-1R and MIN-2R at most 0.034 | Reproduced |
| 10 | Trend misses new RTX 5090s in 5 of 18 cells, by up to 18% | 285K ahead +17.6% and +13.2%; 13900KF ahead +14.6%; 9800X3D MIN-1R −11.3% and −10.7% | Reproduced |
| 11 | RTX 4090: layer-ahead variants 0.64–0.72×, Margin 1.033–1.041×, capture ≤26% | 111d/f/g | Reproduced |
| 12 | Residual 31–36% (27–31% with 3.5 ms) on the slow-link RTX 4090s; 35–47% on the two slow-link panel machines | 111d 35.8 / 30.8; 111f and 111g 31.5 / 27.0; 099d 35; 099f 47 | Reproduced |
| 13 | Table 5 speeds: Few-1R 1.21–1.41, Few-2R 1.03, early 0.98–1.00 | 112a/b/d/g | Reproduced |
| 14 | Table 5 misses; early/Few-2R 0.97–0.98; in-step fetches 17.8–23.7 per token | `st_*_bypassplan*.json`: misses/steps; ratio 0.968–0.980; fetches 17.8–23.7 | Reproduced |
| 15 | T1 failed on 5 of 8, T2 on 7 of 8 | Misses ratios at 25%: 0.9496, 0.9491, 0.9530, 0.9490. Δspeed at 11%: −0.053 to −0.031; at 25%: −0.027 to −0.006 | Reproduced |
| 16 | Running example: 10.0, 13.0, 20.6, 15.2, 13.8 ms | 096a: 10.02, 12.96, 20.64, 15.16, 13.81 | Reproduced |
| 17 | $T_{\text{GPU}}$ 2.9 ms is "the smallest of our engine's Nsight profiles"; G 4.1–4.7 ms; RTX 4090 about 3.5 ms | `prof_*.json`: 2.94 is the minimum of 14 profiles (069c, 105). Over all 47 RTX 5090 profiles: minimum 2.87, median 3.16. G 4.05–4.69; RTX 4090 3.37–3.56 | Values right; "smallest" holds only within that set |
| 18 | Few-1R geometric mean 1.24× [1.16, 1.32] on 13 stable machines; lost on 2 unsteady ones | `fetchplan` across jobs 104–107 and 112: 1.240 [1.163, 1.323]. Losers: EPYC 7302 0.845, EPYC 7663 0.974. The 13 include 2 RTX 4090s and 2 server CPUs | Reproduced (scope, W4) |
| 19 | Same CPU model, up to 29% apart | 099a vs 099f: 18.14 vs 14.06 ms | Reproduced |
| 20 | Relaunches within 5.5% (RTX 4090) and 0.7% (job 112); ratio moved by up to 12% | 110c→111g +5.44%; 110d→111d +4.1%; 109f→112g −0.71%, ratio −11.8% | Reproduced (5.4 vs 5.5) |
| 21 | Second probe within 4.9% on 4 machines | `concur2.txt`: +0.8, +4.9, −0.4, −0.3% | Reproduced |
| 22 | Table 3: host S (all 6 rows), host B gpt-oss rows; bounds 172/430 and 140/351 tok/s | `bs1.jsonl` (launch 2 for ours and FreeToken; paired bootstrap): e.g. host S 1.207 [1.194, 1.219]; host B 1.294 [1.278, 1.312] | Reproduced |
| 23 | 31–54% of the bound on consumer machines; 41–48% on the job 109 machines | 40 consumer RTX 5090 launches: 31.2% (TR 9960X, 178 GB/s) to 54.0% (100f); job 109: 41.5–48.0% | Reproduced (W7) |
| 24 | Table 30: 86–96% of the read time | `readsched_C*.txt`: all 24 cells | Reproduced |
| 25 | Registration timing (92 rentals, ≥2.7 s, ≥26 s, 7 within 1 s, 4 pushes after rental creation) | `git log %ct` against ledger `start`; `gpu_pushes.json`; manifest `start_utc` | Reproduced |
| 26 | Post-launch header edits only substitute machines or add disclosed amendments | Diffs of 3701cea, a8e2f92, 89c5c3b, 718bf41, 58d5454, and the job 107 commits | Reproduced; all disclosed |
| 27 | Gate outcomes for jobs 107, 108 and 110; 112c without results | `v0.txt`, `validity.txt`, `DONE`, ledger | Reproduced |
| 28 | 110e left out of the relaunch "by an oversight" | Its UUID is absent from `job110_uuids.txt`. Round 1: ahead 1.056 (trend 1.12), Margin 1.043, capture 78% | Disclosed; capture not reported |
| 29 | Job 109 validity (V1, V2) and "11 of 11 predictions held" | Losses 0.1902–0.1906; rounds within 0.4%; P2, P4, P5, P7 (mean capture 3.5%) and P8–P11 recomputed | Reproduced |
| 30 | Parity KL 0.0019 and 0.0005 nats | Job 090 `parity_kl.txt`: 0.00186 and 0.00053 (an on-machine summary; logits not shipped) | Matches the summary |
| 31 | Cost $105.8 over 145 rentals | Ledger | Reproduced |

## Clarity

Earlier rounds scored clarity 2 and then 3. This revision is easier to navigate than a typical heavily hedged empirical paper, but it remains hard to read for anyone who did not run the experiments.

**What works**

- **Table 1** is a genuinely helpful map: claim, test, outcome, machine count and section.
- **The "Terms" and "Yardsticks" paragraphs** (Section 2) define the vocabulary up front, and the naming scheme (set + number of reads: MIN-1R, MIN-2R, Dep-1R) is systematic.
- **Fig. 1** makes the "second read" concrete.
- **The running example** (9950X: 10.0 / 13.0 / 20.6 / 15.2 / 13.8 ms) anchors the shares in milliseconds.
- **Section 8** turns the results into five actionable rules, and the Limitations section is specific.
- **Section headings carry their registration status,** which helps transparency.

**What still makes it hard**

1. **Too many named things for ten pages.**
   - Fifteen configuration names in the main text: deployed, Dep-1R, admit every miss, Margin, LA, LA-1R, LA-1R-margin, MIN-2R, MIN-1R, Few-1R, Few-2R, Few-2R-early, read-ahead oracle, Belady-2R, Belady-1R.
   - Separate code names in the appendix (base, foa, fetch, bypass, both3p, bypassplanS, …).
   - At least eight machine sets: panel (15), new (5), factorial O3–O5, hosts A, B and S, P-letters, "13 stable", "25 consumer", and the RTX 4090 jobs 111 and 112.

   A reader has to keep Table 2 and Appendix C open. Suggestions:
   - keep at most six configurations in the main text;
   - drop Belady-1R and Belady-2R, LA-1R-margin and Few-2R-early from the narrative;
   - use one machine vocabulary throughout.
2. **Five denominators and three bounds.**
   - The denominators are: speed relative to the deployed cache; share of Eq. (1)'s speed; share of the gap; capture (share of the read-ahead oracle's gain); and share of MIN's gain over admit-every-miss.
   - Fig. 5's caption has to open with "Baseline here: admit every miss, not the deployed cache", which is a symptom.
   - "The bound" means Eq. (1) "throughout", yet Eq. (3) is "tighter" and "the bound is the larger of the two".
   - Pick one bound for the narrative and at most two shares.
3. **Clause-stacked sentences and double negations.**
   - The abstract's fourth sentence runs to about 70 words with five embedded qualifiers ("At that budget, on machines whose link reads…, caching … pays only if … and the two changes together close 35% of the gap, the deployed cache's time beyond the bound, in a test registered…; there, four policies…").
   - The Section 4 control paragraph chains "does not rescue … fall only to … because … The control thus failed to give … so it cannot say … it does show that … is not what holds … back."
   - Split these, and state the result first and the caveat second.
4. **Cryptic evidence notation.** Table 1's evidence column reads like log output: "misses ≤ 0.95× and no slowdown, failed and failed; second read costs more than the landing, held". Inline headings such as "(registered: gain bands per budget on two machines; held at most budgets)" do not tell the reader what the threshold was against the estimate. One column for "registered threshold" and one for "estimate [CI]" would help.
5. **Contribution sprawl.**
   - The paper is a bound, a decomposition, policies without foresight, a second-card transfer test, a horizon rule, a literature audit, a closed-form model and a prediction log.
   - Section 6 joins two unrelated topics.
   - The audit and the closed form each get main-text paragraphs that end in "post hoc" or "not confirmed".
   - The narrative would be clearer with three contributions in the main text and the rest in appendices. That would also fix the page overflow.
6. **Definitions with exceptions.**
   - "Host-bound" is defined for "the two smallest budgets", then excepted at gpt-oss 25%, one of the two budgets most results use, on some machines and at the measured GPU rate. The definition also uses Eq. (1) before Eq. (1) is introduced.
   - "Fast link" means a link-to-CPU ratio of at least 0.5, not a fast PCIe link. The paper warns about this, but the word still misleads, and the class is unstable across rentals (W3).
7. **Self-discounting forward references.** For example: "31–54% … The top of that range is probably too high: on that machine the engine's reads imply, by the closed-form account of Appendix H, a rate above the probe's best reading (Table 28)". If the top of the range is doubtful, report the range without it, or with it flagged in the range itself.
8. **The figures are dense.**
   - Fig. 2 mixes measured states and two bounds on one categorical axis, labels a state "MIN-2R (or Dep-1R)", and overlays three colour classes plus two median lines.
   - Fig. 3 has 7 rows × 6 panels of small markers.

   Both work in a talk but not at column width.

## IEEE format (paper_ieee.pdf, skimmed)

1. **Length and layout.** The PDF is 39 pages: about 9 pages of body, about 3.5 of references, then 27 pages of appendices after `\clearpage\onecolumn`. IEEE conference papers are two-column throughout and usually limited to 6–10 pages including references. The appendices belong in separate supplementary material.
2. **Appendix references.** Appendices are referred to as "Section D", "Section H", "Section N" and "Sections D and N", including in Table I's caption and in the appendix map ("Section B: the scientific benchmarking checklist…"). This is cleveref with `\appendices` in IEEEtran; it should read "Appendix D".
3. **Run-in headings.** They render with doubled punctuation, e.g. "a) Contributions.:" and "a) Models and workload.:". Several are whole sentences, e.g. "c) The read time is nearly reachable (registered: 50% or more per layer; held).:". IEEE level-4 headings are short phrases ending in a colon.
4. **Table captions.** Multi-sentence captions with definitions are set in IEEE's small-caps caption style (e.g. TABLE I, TABLE IV, TABLE VI). Italic terms inside them fall back: the log reports "Font shape OT1/ptm/m/scit undefined". IEEE expects a short title above the table, with notes as table footnotes.
5. **Citation style.** natbib with `sort&compress` gives "[3–5]" and "[1, 2]" rather than IEEE's "[3]–[5]" and "[1], [2]". Some entries cut the author list to "et al." after one or two names (e.g. "L. Xue, Y. Fu et al."), where IEEE lists up to six authors.
6. **Index Terms.** They are not alphabetical and are inconsistently capitalised ("Mixture of Experts, offloading, caching, performance bounds, foresight").
7. **Float placement.** Fig. 5 is first cited in Section VII on page 7 but appears at the top of page 9, above the Limitations, the Conclusion and the start of the References. Tables run to XXXII in Roman numerals, which is legal but unwieldy.
8. **Front matter.** The anonymous author block and the abstract (221 words) are fine for a double-blind submission.

## Questions for the authors

1. Can you run the two-read arm with the fetch table forced to MIN's plan, or disabled, so that MIN-2R actually holds MIN's set (misses close to Few-1R's 40.7 per token)? Without that, how much of "pay only together" survives beyond this engine?
2. What are the round × problem (two-stage) intervals for Tables 5 and 6? How many per-machine clauses move from "held" to "held (point)" under them?
3. How do Table 4's attributed rows change with $T_{\text{GPU}}$ set to the median profile (3.16 ms) or each machine's own profiled non-expert time?
4. Why is the Few-1R claim stated as "every stable machine" rather than as a function of the link-to-CPU ratio, given that both losses are on the two slowest links?
5. With the probe's test-retest variation of up to 12%, how many machines would change between fast and slow classes on a second rental?
6. Could each manifest record `git rev-parse HEAD` of the cloned branch and the sha256 of the job script, so that "ran the header as committed" is recorded, not inferred?
7. Can you add 110e's first round to Table 6, marked as from the voided job? Its capture (78% of a 1.056× gain) is outside every other machine's range.
8. What is the audit's median when the published systems are scored with conservative measured rates (or your probe on comparable hardware) instead of datasheet rates?

## What would raise my score

- Either make MIN-2R hold MIN's set, or rescope the headline 2×2 claim and the Table 1 timing-control row to "this engine" and "inconclusive".
- Abstract: "oracles remove about half the gap; the rest is attributed", with a $T_{\text{GPU}}$ sensitivity band in Table 4.
- Two-stage intervals for every multi-round experiment, and rental-level reproducibility stated next to every per-machine claim.
- Restate the Few-1R and fast-link claims in terms of the ratio, with its measurement error.
- Cut the main text to the bound, the decomposition and the read-path/ratio result. Move the audit, closed form and horizon rule to appendices. Use one bound and at most two denominators, and fit in ten pages.
- Optionally, a registered decomposition on a second model (Qwen3 at 12.5% and 25%) would materially raise significance.

## Scores

| Criterion | Score |
|---|---|
| Overall | **7 / 10** (accept for a workshop; strong evidence of rigour for a fellowship) |
| Soundness | **4 / 5** |
| Methodology | **4 / 5** |
| Significance | **3 / 5** |
| Clarity | **3 / 5** |
| Confidence | **4 / 5** |

- **Overall.** The numbers and the registration practice are as good as I have seen at this scale. The headline causal reading, how much is measured, and how general the results are all claim more than the evidence carries.
- **Soundness.** Every number I recomputed reproduces. The deductions are W1 (the 2×2 manipulation), W2 (attribution versus measurement) and W4 (scope by filter).
- **Methodology.** Exemplary registration, gates and artifact. The deductions are the problem-only intervals, the convenience sample, and registered bands too wide to be informative for the headline numbers.
- **Significance.** Useful framing and practical rules, but one engine, essentially one model, and engine-specific mechanisms behind the main effect.
- **Clarity.** It has improved and is navigable through Table 1, but it is still dense with names, denominators and stacked qualifiers. The MLSys build overflows the page limit.
- **Confidence.** I checked 31 claims against raw data and read the main text and most appendices closely. I did not re-derive the MIN and LP plans or the trace study.
