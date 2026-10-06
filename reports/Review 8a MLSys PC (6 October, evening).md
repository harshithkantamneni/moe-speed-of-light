# Review 8a: MLSys 2027 PC review

**Submission:** *Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth* (31-page PDF: 10 pages of main text, then appendices A–K; 30-page supplement with the prediction scorecard).

**Materials read**
- `paper/paper.pdf`, read in full with `pdftotext -layout`, including every appendix. I also checked the generated macros and tables it is built from (`paper.tex` preamble, `wsg_job102.tex`, `wsg_readsched.tex`, `wsg_auditours.tex`, `wsg_hostdepmodel.tex`, `tab_headline.tex`).
- `paper/supplement.pdf`: I skimmed the scorecard tables and read the job 099–102 sections.
- Scripts: `readsched.py`, `job102.py`, `hostdep_model.py`, `value_map.py` (the policy core), `audit_ours.py`, the audit and cost functions of `wsg_numbers2.py`, and the headline function of `wsg_tables.py`.
- `prereg/*.json` (not the outcome notes): `readsched.json`, `factorial_shapley.json`, `panel_099.json`, `policy_aa.json`, `foresight/w50_distinct.json`, `speed_limit_v2.json`, `run1_082.json`, `scorecard_clauses.json`, `scorecard_09[9]/10[0-2].json`, `audit/*.json`, and the ledger `gpu/vast_ledger.json`.
- GPU branch: the `jobs/102_readsched@vast.sh` header, and raw results of jobs 080, 081, 082 (partly), 084c (the routing trace), 089, 090, 096a, 099a–j, 100a/b/c/f, 101a/b and 102a/b/c.
- Two web look-ups confirming that the main baseline (FreeToken, arXiv:2608.16157) and the closest trace-level prior work (Zhang, arXiv:2608.07911) exist.

**Independence statement.**
- I did not open anything under `reports/` (I only tested that this file's path did not already exist), any `prereg/*outcome*.md`, or any file named like a review, plan or progress log.
- I did not read commit messages. I used `git log --format='%h %ad'` for commit order. I also ran one `git diff` of the job-102 script between its two commits, to check whether its predictions changed (they did not; only the host description changed).
- I modified no file in either repository. All recomputation used my own scripts in the scratchpad (`review8a/py/`). The one repository script I re-ran (`hostdep_model.py`) was a copy whose outputs I redirected to the scratchpad.

**Desk-level issues for the chairs (not scored)**
1. **Not anonymised.** The title page carries the author's name and "Independent Researcher". MLSys main track is double-blind.
2. **Not the MLSys template.** The paper uses `article`, two columns, 0.75 in margins. That text block is about 7.0 × 9.5 in, against roughly 6.75 × 9.0 in for the MLSys/ICML style. The main text ends on page 10 in this larger block, so in the official style it would very likely exceed the 10-page limit.
3. **Appendices outsize the main text.** There are 21 pages of appendices and a 30-page supplement. Several claims in the main text are justified only there (the time model, the bound's tightening, the 52-row audit).
4. **Artifact references need anonymising.** Appendix K and Appendix B refer to "the public gpu branch", the rental ledger and per-job hosts. That is fine for artifact evaluation, but it must be anonymised for review.

---

## 1. Summary

The paper studies batch-1 decode of MoE models whose experts mostly live in host DRAM on a consumer GPU (gpt-oss-120b in MXFP4 and Qwen3-30B-A3B in BF16, on rented RTX 5090 desktop hosts). It makes five contributions.

**(i) A bound on time per token.** For any exact-routing system with C GPU slots per layer, the bound is Belady's MIN-with-bypass host reads divided by the host's best probed read rate, combined in a roofline-style max/min with a GPU term (Eq. 1).
- The authors' 3,500-line llama.cpp expert cache uses decayed-frequency admission, CPU helpers for misses, and a per-machine CPU/PCIe "fetch table" from a calibrated model. It beats FreeToken at 11 of 12 cells and runs 2.0–4.0× stock llama.cpp.
- It reaches 25–43% of the bound, or 31–56% with a measured GPU rate.
- A microbenchmark (job 102) replays MIN's per-layer reads without the model and reaches 86–96% of the bound's host term.
- A "read-once link cap" shows that MIN's own single-read schedule cannot get near the bound on hosts with a slow link.
- 52 published measurements are scored the same way: a median of 13.6% for the 20 in-class rows, against 27% for the authors' cache.

**(ii) An in-engine oracle factorial.** It crosses *what to cache* (deployed admissions vs MIN's) with *how to load* (CPU then background copy, versus one in-step copy).
- Either change alone closes at most 26% of the gap to the bound. Together they close 14–60%: a large interaction.
- Whether an in-step copy pays depends on the host's link-to-CPU read ratio. On the one machine at a ratio of 0.29 it loses, and only copies made ahead break even.
- Gaps are split with Shapley values.

**(iii) A calibrated two-path time model.** Given byte counts, it predicts in-step states within a median 2.2% on 14 hosts.

**(iv) A price on foresight.**
- Exact windows of W tokens, degraded windows, and real forecasters are tested in the engine and on nine models' traces.
- Half of MIN's read saving needs 2–10 exact tokens. The window that closes half the read gap holds about 0.65·C distinct experts.
- No online policy, learned admission order, speculative batch or trained forecaster comes near that horizon.

**(v) Pre-registration.** Every prediction was committed before its machine started, and 565 + 547 clauses are scored, failures included.

## 2. Strengths

1. **A real system against a strong, current baseline, measured carefully.**
   - The cache leads FreeToken on 11 of 12 cells on two hosts, with paired bootstrap intervals; I reproduced all 12 exactly.
   - It holds over held-out long outputs (2,048 tokens) and against FreeToken tuned over five settings.
   - Output parity is checked by teacher-forced KL against an all-VRAM reference.
   - The FreeToken comparison on FreeToken's own headline model, where the two tie, is reported rather than hidden.
2. **Unusually rigorous experimental practice.**
   - The host is treated as the unit of variation: a 10-host panel, a crossed bootstrap over hosts and problems, relaunches, and seeded random configuration orders.
   - This is the right design for rented consumer hardware. The paper shows that machines differ by 16–29% even with the same CPU model, while relaunches agree within 1.9%. I verified both.
3. **One genuinely useful, non-obvious finding: what to cache and how to load interact.**
   - MIN's admission set is worth almost nothing when each admission is read twice: 4.5% of the gap on O4 at gpt-oss 11%.
   - It is worth 51.5% when each admission is read once.
   - Whether reading once pays is decided by the host's link-to-CPU ratio, and the familiar ways of spending foresight (Belady prefetch, CPU-then-copy) read 1.6–2.5× MIN's bytes.
   - This is a crisp, actionable lesson for expert-cache designers.
4. **Honest treatment of foresight's value.**
   - The window study covers exact and degraded windows, two baselines, engine and trace, linear and recurrent forecasters, learned admission, and speculative batches.
   - It reports negative results plainly: no realisable source of foresight reaches the needed horizon.
5. **The paper criticises its own bound.**
   - It tightens the bound four ways (Table 16).
   - It microbenchmarks the bound's read time, adds a cap showing where the bound is loose, and states which predictions failed and why.
6. **An exceptional artifact.** Every number is generated by a script from raw rows. 24 of my 25 checks held, many to the last printed digit, including an independent re-implementation of MIN with bypass and of the window policy (Section 6).

## 3. Weaknesses (ranked)

**W1. The paper is too dense to read, and it tries to make too many contributions at once.**
- The ten main pages carry a system, a bound, a microbenchmark, a link cap, a 52-row literature audit, a factorial with Shapley accounting, a calibrated model with three validation regimes, a window study on two baselines, a nine-model trace study, a horizon rule, four realisable-foresight attempts, and a pre-registration protocol.
- Almost every sentence carries two or three ranges, and terminology drifts (Section 4 below).
- A PC member reading only the main text will struggle to say what the paper's one main result is.
- *Fix:* restructure around three findings: (a) the bound and how close systems and oracles get; (b) the what × how interaction and its host dependence; (c) the price of foresight. Move the audit, most of the time-model validation and the pre-registration detail to the appendix. Add one host table and one glossary. Cut each paragraph to one quantitative claim and one central tendency.

**W2. The bound is loose exactly where the paper needs it, and the claim that its read time is "nearly reachable" rests on an infeasible relaxation.**
- The job-102 microbenchmark lets *every* MIN read take either path. But an expert that MIN admits must cross the link to land in its slot. A schedule that reads it on the CPU either never makes it resident (more misses later) or reads it twice.
- The abstract's "reaches at least 86% of the bound's read time" is therefore a statement about raw host bandwidth at expert granularity. It is not a statement about any feasible caching schedule.
- The paper's own cap shows that on slow-link hosts MIN's single-read schedule is limited to 53–71% (11%) and 43–58% (25%) of the host term.
- In addition, the GPU term uses the datasheet rate, which batch-1 decode reaches only 52–61% of.
- The best oracle reaches 46–73% of the bound, and 30–62% of the gap ("the rest") is explained only qualitatively.
- Consequently "time is host bytes" is supported as a lower bound, not as a tight description of where the seconds go.
- *Fix:* replace the relaxation with a link-aware, read-once lower bound. Choose per layer which misses to admit, so that admissions go over the link and bypasses may go to either path; this is a small ILP or min-cost-flow per layer, or a Lagrangian relaxation. Then replay *that* schedule in the microbenchmark with admissions forced over PCIe. Report the bound with the measured GPU rate as the primary number, and measure read/compute overlap with Nsight to bound "the rest".

**W3. The novelty is incremental relative to classical caching theory and to the 2026 MoE-caching analyses the paper cites.**
- Each tool is established:
  - "Bytes over bandwidth with MIN's reads" is a roofline plus Belady with bypass.
  - Measuring lookahead in distinct items is paging-with-lookahead theory [2, 6].
  - Shapley attribution of performance gaps has precedent [35].
- Each claimed finding has a close relative in prior work:
  - The causal-vs-MIN gap is in Zhang [62] (44–46%).
  - Hit rate against m-token lookahead versus C/k is in Liang et al. [37].
  - In-engine perfect foresight of the coming stage is in SAEM [57].
  - Cross-hardware prefetch-policy benchmarking is in SpecMD [19].
- The paper's own analysis shows the "0.65·C distinct experts" rule is statistically indistinguishable from the two-constant power law (error ratio 0.95–1.09). It is also not constant within a model: it falls with budget in 9 of 9 models. Yet it appears in the abstract and conclusion as a headline.
- *Fix:* state the delta precisely: time domain, in a real engine, the what × how interaction, the host dependence, and the link cap. Demote the 0.65·C rule to a descriptive observation.

**W4. Narrow scope, a workload reused from development, and few slow-link points.**
- Scope is batch-1, two models of the authors' choosing, and 30 AIME-25 problems that the Limitations say were also used while developing the system. The panel uses the first 20 problems and 256 tokens.
- Generalisation is limited:
  - The oracle factorial ran at every budget on two hosts.
  - The crossover where the in-step copy loses rests on one machine (two launches), and the 0.5 class line was drawn after the panel ran.
  - On FreeToken's own headline model the two systems tie (1.00–1.03×).
  - Prefill and batch>1 bypass the cache and regress to 77–83% of stock llama.cpp. That matters for interactive use (TTFT) and is mentioned only in Appendix I.
- *Fix:* re-run the Table 2 protocol and the 2×2 factorial on held-out chat and code prompts (e.g. LMSYS-Chat or HumanEval) with ≥1k-token outputs, a third model with many small experts (e.g. Qwen3.6-35B or DeepSeek-V2-Lite), and 3–4 more hosts with a link-to-CPU ratio below 0.5. Report TTFT and end-to-end request latency.

**W5. The practical upshot is mostly negative.**
- The large gains (16–51%, and 25–81% with prefetch) require future routing that the paper shows no realisable source provides.
- What is deployable gains little: 16-token windows on the deployed path gain at most 12%; the learned admission order runs 0.90–1.05×; single-read online admissions run 0.97–1.03×.
- For an MLSys systems paper, readers will ask what system to build next. The answer offered ("a forecaster for ≈0.65·C distinct experts") is a target, not a mechanism.
- *Fix:* either demonstrate one deployable mechanism derived from the findings, or reframe the paper explicitly as an evaluation-methodology paper and lead with that contribution. An example mechanism: read-once admissions with a link-aware split plus short-horizon prefetch from a layer-ahead predictor on fast-link hosts, with a measured net gain on the panel.

**W6. The literature audit is fragile but sits in the abstract and introduction.**
- It scores 52 rows at datasheet ceilings, using the authors' traces of other text (sometimes a sibling precision), with imputed C for 4 of the 20 in-class rows.
- 3 of those 20 rows are below 5%, which Table 19's caption itself calls "more likely configuration mismatches … not adjudicated". The headline median nonetheless includes them; without them it is 14.6% rather than 13.6%.
- The authors' own cache is scored on its own tuned workload. "Published systems reach a median 13.6% and ours 27%" invites dispute from the cited groups and does not need to be a headline.
- *Fix:* move the audit to an appendix with per-row sensitivity (probed rather than datasheet ceilings; excluding imputed-C and below-5% rows), or drop the comparison sentence from the abstract.

**W7. The time model predicts modestly, and its framing overstates it.**
- The fitted constant G (4.1–9.2 ms per host-cell) absorbs a large share of each token's time. It is the median of other hosts' fits, not a quantity derived from the probe.
- With G fixed, the model ties a one-path model everywhere except the two losses of a single machine.
- It over-predicts the speed of every background/prefetch state, 64 of 64, by a median 16%.
- With frozen constants it over-predicts Qwen3 on later hosts by +5 to +21%.
- "Given a host's bandwidth probe and the bytes each path carries" in the abstract omits the fitted constant.
- *Fix:* say "with one constant per budget fitted on other hosts" in the abstract. Either model link occupancy and background copies, or restrict the claims to in-step states.

**W8. Smaller inconsistencies that a careful reader will trip on.**
- **(a)** Section 5's lead says a 16-token window on the deployed path "gains on every host". In job 100 it is 0.9995 [0.995, 1.004] on the 5700X3D at gpt-oss 11%, and 1.008 [0.996, 1.022] on the 9950X with a slower link. The abstract's "break even or gain" is the correct wording.
- **(b)** The cap ranges in Section 3 hold only for the 21 hosts of jobs 095–101 that `scripts/readsched.py` selects, and the text does not say so. On job 085's host (ratio 0.48) the cap at 11% is 84%, and the EPYC 7352 (ratio 0.66) binds at 11%.
- **(c)** MIN's admission count is not unique among optimal schedules. My tie-break gives 21.39 admissions per token (55.8%), not 21.77 (56.8%), at C=14, so the cap depends on the tie-break.
- **(d)** Section 2 says jobs 093–097 averaged speeds. The factorial accounting (`factorial_shapley.json`) averages ms per token. The effect is negligible but the statement is inaccurate.
- **(e)** Table 7 contains a blind-test row with n=0.

## 4. Clarity: 2 / 5

Specific passages that are hard to follow, and what would fix them:

- **Abstract, sentence 3.** It mixes three results in one sentence: the cache vs FreeToken, the cache vs the bound, and the microbenchmark vs the bound's read time. "Each free to take the CPU or the PCIe link" hides the relaxation discussed in W2. *Fix:* one sentence per result, and state the relaxation.
- **Section 2, "Hosts and statistics."** A single paragraph covers the four probes, averaging, the bootstrap, host-to-host variation, relaunch variation, the panel design, the crossed bootstrap and the clause counts. *Fix:* a protocol box, plus a table of every host. The table should list the label used in the text (A, B, S, O1–O6, Pa–Pj, "the new machines"), CPU, B_c, B_p, B_cp, link/CPU ratio, and the jobs it ran.
- **Section 3, "The free split ignores one constraint."** One sentence carries four ranges, two budgets and two host classes, with the host set unstated. *Fix:* a small table (host class × budget → cap), with the hosts named.
- **Section 4, "An order-free accounting" and Table 3.** The reader must reconstruct the 2×2 design from prose; the caption defines seven derived columns. *Fix:* show the four measured cell times of the 2×2 for one host-budget as a tiny diagram, then define *alone*, *interaction* and *Shapley* on it.
- **Section 5.** The section alternates between two baselines (deployed vs admit every miss), two units (reads vs time) and two venues (trace vs panel), and Fig. 3 overlays them. *Fix:* use one baseline per subsection, and put the unit in every sentence that quotes a share.
- **Terminology drift.**
  - bound / limit / "speed of light" (Fig. 9's axis);
  - model / "law" (Fig. 5's legend says "law's table"; Appendix I says "the law predicted");
  - copy in the step / single read / fetch / FETCH / foa;
  - hit-optimal / bytes-optimal / bypass / fetch / lead / both / paced / nb2 oracles (Tables 9–10);
  - **undefined in the main text:** host A (Fig. 8; Appendix I), "Ph" (Table 4) and O6 (Fig. 2); O1 and O2 appear only in the appendix.
- **Appendix B.** A paragraph about 60 lines long enumerates failed clauses. *Fix:* a table of failure types with counts, and a pointer to the supplement.
- **Table 9.** About 25 columns; unreadable at print size. Split it into two tables.

## 5. Novelty against related work: 3 / 5

**What is classical or already published:**
- The bound is the roofline [51] with Belady's MIN with bypass [5, 10, 25] as the byte count. Budgeting Bytes [58] already bounds storage-bound decode by bytes over bandwidth, and WiSP [59] by PCIe turnover.
- The gap between causal policies and MIN in MoE expert caching is quantified by Zhang [62].
- Oracle hit rate as a function of lookahead and C/k across 20 models is in Liang et al. [37].
- SAEM [57] gives an in-engine planner perfect stage foresight.
- Lookahead measured in distinct pages comes from Albers [2] and Breslauer [6].
- Integrated prefetching and caching trade-offs come from Cao et al. [8], and Belady with prefetching from Jain and Lin [26].

**What is new:**
1. Oracles run inside a real engine in the time domain, with a factorial separating *what to cache* from *how to load*, and the resulting interaction.
2. The dependence of which load pays on the host's link-to-CPU read ratio, with a model that predicts it.
3. The read-once link cap.
4. The degraded-window and forecaster valuations in time on a 10-host panel.
5. A uniform audit of published systems against a bound.

These are new empirical results. The analytical machinery is not new, and the horizon rule adds little beyond the power law the paper also fits. I score 3 for the new empirical insight (mainly items 1 and 2); the conceptual delta alone would be a 2.

## 6. Soundness and the artifact checks: 4 / 5

The measurement methodology is sound and the artifact is excellent. I recomputed 25 claims, mostly from raw `results/*/` rows with my own code: 24 held and 1 failed (wording). I deduct from 5 for three reasons:
- The "reachable read time" framing relies on a relaxation (W2).
- Several scoping and tie-break issues remain (W8).
- The development workload is reused for evaluation (W4).

| # | Claim (where) | What I did | Result |
|---|---|---|---|
| 1 | Table 2, host B, gpt-oss 11/25/40% and Qwen3 12.5%: speeds, ours÷FreeToken with 95% CI, ours÷llama.cpp | Own paired bootstrap (10,000 resamples) on `081_headline_law@vast/bs1.jsonl`, ours launch 2 vs FreeToken launch 2 | **Held exactly:** 1.294 [1.278, 1.312], 1.275 [1.254, 1.295], 1.154 [1.134, 1.172], 1.032 [1.022, 1.042]; 2.00/2.73/3.20/2.05× |
| 2 | Table 2, host B, Qwen3 25% (†) and 43.75% | Means from `080_fetch_split@vast/bs1.jsonl`; checked that the GPU UUID is the same in 080/081/082 | **Held:** 63.12/54.75 = 1.153; 108.24/103.14 = 1.049 |
| 3 | Table 2, host S, all six rows (job 089); "11 of 12"; "2.0–4.0× llama.cpp" | Own bootstrap on `089_.../bs1.jsonl` | **Held exactly**, including the one loss of 0.974 [0.962, 0.987] and 4.02× |
| 4 | R*, MIN-with-bypass host reads per token (the core input of the bound) | My own Belady-with-bypass implementation on `084c.../route_aime25_gptoss.npz` | **Held exactly:** 38.315 (C=14), 15.340 (C=32), 6.785 (C=51). On the first 4,000 steps it gives 38.505 and 15.520 (job 102's `reads_per_token`), and the per-layer counts equal `jobs/ec2/minreads_g{14,32}.bin` element for element |
| 5 | Bound on host B: 172 / 430 / 519 tok/s; cache at 25–43% (31–56% with measured GPU) | Eq. 1 by hand from R*, S, D = 1.714 GB, B_host = 87.5, B_gpu = 1,792 (at 40%, c* = 12.7 CPU experts) | **Held** |
| 6 | Table 17 (job 102): per-layer 86–96%, per-token 93–97%, link-only, CPU-only; repetitions within 0.9%; model within 0.93–1.14× | Parsed `102{a,b,c}_readsched@vast/readsched_C*.txt`; B_host = max concurrent sum in `concur.txt` (e.g. 72.5 on 102a); recomputed the analytic modes from my own MIN counts | **Held (all 24 cells):** e.g. 102a C14 term 7.04 ms, 86/93/35/92%; worst repetition gap 0.87% |
| 7 | Job 102 scoring and pre-registration | Counted `scorecard_102.json` and re-derived the failing clauses; `git log --format='%h %ad'`; diff of the script's two commits | **Held:** 0 held / 28 point / 5 failed (P2-below on 102b and 102c at both C; P3 order on the Intel host). Script committed at 10:14:17, before the first result at 10:15; the 10:14:41 edit changed only the host description |
| 8 | Read-once link cap: admissions 57% / 69% of MIN's reads; cap 53–71% / 43–58% on slow-link hosts, ≥86% / ≥70% on others; binds up to ratio 0.61 / 0.77 | Recomputed from per-host `readsched.json` and checked `readsched.py` | **Held, with caveats.** The ranges hold only for the 21 hosts of jobs 095–101 that the script filters to, and the text does not say so: job 085 (ratio 0.48) has cap 84% at 11%, and the EPYC 7352 (0.66) binds. My MIN with a different tie-break admits 55.8% / 68.5%, not 56.8% / 69.3%; the admission set is not unique |
| 9 | Shapley accounting, O4 gpt-oss 11% (§4 text, Table 3 row 1) | Own computation from `096a_factorial_9950x@vast/ec_g_C14.jsonl` | **Held exactly:** gap 10.62 ms; load alone 3.9%, cache alone 4.5%, both 51.5%, interaction 43.2; Shapley 26.1 / 25.5; prefetch 12.7; rest 35.7 |
| 10 | Panel (job 099), MIN with one read at gpt-oss 11%: SD 0.154, two-stage interval [1.12, 1.30], correlation 0.93 with log link/CPU | Own per-host ratios and crossed host × problem bootstrap from `099{a..j}` raw rows | **Held:** mean 1.222, SD 0.154, [1.125, 1.302], r = 0.936; worst host 0.875 |
| 11 | Panel window claims (§5): 16-token in-step window 0.71–1.31×; 4-token slower on 6 (8) hosts; 16-token on 2 (3); admit every miss 0.42–1.00× | Same data | **Held** |
| 12 | Machine vs launch variation (§2): relaunches within 1.9%; same-CPU pairs differ 16–29% | 099d/100a, 099h/100c, 099f/101a; the 285K pair and the 9950X pair | **Held:** −0.3, −1.8, +1.9, +0.3, −1.7, −0.0%; pairs 15.6–29.1% |
| 13 | Table 4 (job 100), all 64 entries; §5's "a 16-token window gains on every host" | Recomputed from `100{a,b,c,f}`; bootstrap for b16 | **Table: held exactly. Sentence: failed (wording).** The 5700X3D at 11% is 0.9995 [0.995, 1.004]; the 9950X slower link at 11% is 1.008 [0.996, 1.022] |
| 14 | MIN prefetched never loses beyond its interval (worst 0.997×); MIN one read loses on both launches of the 0.29 machine | `101a` raw rows | **Held:** 0.9968 [0.987, 1.007]; one read 0.875 and 0.86 |
| 15 | O3–O5: MIN one read +16–51%, prefetched +25–81%, best oracle at 46–73% of the bound | Table 10 values (the O4 column checked against raw rows in #9) | **Held** |
| 16 | Time model "median 2.2% (90th percentile 10%) on 14 hosts" | Re-ran a copy of `hostdep_model.py` with outputs redirected to scratch | **Held:** macros identical. Note: G = 4.1–9.2 ms; background states over-predicted in speed 64/64, median 16% |
| 17 | Audit: 20 in-class rows median 13.6% (Q1–Q3 8.1–20.6); ours 18–42%, median 27% | `audit_sol.json` + `audit.json` filters; `audit_ours.json` | **Held.** 3 of the 20 rows are below 5% (median 14.6% without them); 4 have imputed C |
| 18 | W50 ≈ 0.67 (C/k)^1.31, r = 0.977 (1.35 without points below one token); D(W50) median 0.65·C (Q1–Q3 0.61–0.72, CV 0.16); W50 0.6–34 | Refit from `prereg/foresight/w50_distinct.json` | **Held** (I refit the points; I did not regenerate them) |
| 19 | Table 15, gpt-oss 11%: admit every miss 58.6; window shares 0.18 / 0.52 / 0.98 at W = 1 / 4 / 16 | My own replay of the window policy, written from Appendix H's text, on the 084c trace | **Held:** 58.64; 0.185 / 0.520 / 0.978 against the paper's MIN of 39.3 |
| 20 | Online policies (§5): admit every miss best at 18/27 points, 5 ties, worst loss 2.7%; best online policy 35–110% above MIN | `policy_aa.json` | **Held** |
| 21 | Pooling slots across layers lowers reads by 4.5–18.6% | My own pooled MIN (token-atomic, L·C slots) | **Held (approximately):** gpt-oss 5.9 / 9.4 / 12.7%, against Table 16's implied 5.5 / 9.3% |
| 22 | Scorecard totals: 565 = 218/193/142/8/4; job 099: 190/42/39/2; job 100: 41/122/27/26; job 101: 6/19/0/0 | Counted `scorecard_clauses.json` and `scorecard_10x.json` | **Held** |
| 23 | Cost: $70.8 for 88 rentals on 53 machines, jobs 058–101 | `gpu/vast_ledger.json` | **Held** ($70.76; job 102 adds 3 rentals and $0.42, outside the stated range) |
| 24 | Parity: mean KL 0.0019 / 0.0005; top-1 agreement 98.5 / 99.3%; ΔNLL +0.16 / +0.19%; reference on an RTX PRO 6000 | `090.../summary.txt`, `gpu.csv` | **Held** (from the job's summary; not recomputed from the logit dumps) |
| 25 | Predictions committed before runs (jobs 081, 096, 099, 101, 102) | `git log --format='%h %ad'` on job scripts vs result directories | **Held** |

## 7. Questions for the authors

1. **Admission tie-breaks.** MIN's admission set is not unique. With my tie-break, admissions are 55.8% of reads at C=14, not 56.8%. Which rule does the engine and the cap use? Would choosing, among optimal (or near-optimal) schedules, the one with the fewest admissions change the slow-link conclusions? This would be a step toward the link-aware bound in W2.
2. **Host set of the cap.** Please state which hosts the cap statistics in Section 3 cover. Is it intended that job 085's host (ratio 0.48, cap 84% at 11%) and the EPYC 7352 (ratio 0.66, binding at 11%) are excluded?
3. **Overlap assumption.** Have you measured whether the GPU's work overlaps host reads in the deployed cache as Eq. 1 assumes, for example with an Nsight timeline at a host-bound budget? How much of "the rest" (30–62%) is serialisation rather than imprecise prefetch?
4. **Tuning on AIME-25.** Which hyperparameters were tuned on AIME-25: the half-life of 16, κ = 1, the fetch-table model's g and L_c? How does the deployed cache fare against FreeToken on chat or code prompts that were never used in development?
5. **Prefill and batching.** Prefill and batch-2/4 decode run at 77–83% of stock llama.cpp. Could resident experts serve prefill? What is the TTFT penalty for typical 1–4k-token prompts?
6. **Background copies.** The model over-predicts the speed of every background/prefetch state by a median 16%. Is the cause link contention with in-step copies, the paced copier's queue cap, or DRAM interference? Would a model with link occupancy close it?
7. **Audit sensitivity.** Why are the three in-class rows below 5% included in the 13.6% median, when Table 19 calls such rows likely configuration mismatches? How does the median move with probed rather than datasheet ceilings, for the rows where you could probe comparable hardware?
8. **Lookahead within a sequence.** The panel ran 20 problems × 256 tokens with lookahead limited to the current sequence. How much do the window shares and W50 change with longer outputs, where the cache's turnover is no longer dominated by problem boundaries?
9. **CPU cache reuse.** On the X3D hosts (96–128 MB of L3), is any CPU-served expert ever re-served from cache within one token? Your reuse-distance analysis says no (0.5 GB minimum). Is the same true for the second, background read of an admitted expert immediately after the CPU served it?
10. **Release.** Will the engine patch be released (or upstreamed) in a form that runs the deployed policy without the oracle instrumentation?

## 8. Scores

| Criterion | Score | Rationale |
|---|---|---|
| **Overall** | **5 / 10** (borderline reject) | A rigorous, honest, fully reproducible empirical study with a real system that beats a strong recent baseline. Against that, the main pages are overloaded and hard to read. The bound is loose where it matters, and its "reachable" claim rests on a relaxation. The delta over classical caching theory and recent MoE-caching analyses is mostly empirical. The scope is narrow: two models, batch 1, AIME. The deployable gain from the analysis is small. A focused rewrite with a link-aware bound and a held-out workload could make this a clear accept. |
| Confidence | 4 / 5 | I read the whole paper and verified 25 claims against raw data. I am familiar with expert offloading and caching theory. I did not rerun GPU experiments. |
| Novelty | 3 / 5 | New empirical findings: the what × how interaction, host dependence, the link cap and in-engine foresight pricing. The machinery is classical (roofline, Belady, lookahead in distinct pages, Shapley). |
| Soundness | 4 / 5 | Careful statistics, the host as the unit of variation, pre-registration, and an artifact that reproduces. Deductions: the relaxation in the microbenchmark claim, unscoped cap statistics, tie-dependent admissions, an overclaiming sentence in §5, and a workload reused from development. |
| Significance | 2 / 5 | Relevant to local MoE inference, and the bound could improve evaluation practice. But the results cover two models at batch 1 on consumer hosts, the large gains need unavailable foresight, and the prefill/batch regressions limit the system's practical reach. |
| Clarity | 2 / 5 | Extremely dense prose, terminology drift, undefined host labels, and a 31-page PDF plus a 30-page supplement whose essentials do not stand alone in 10 pages. |

## 9. What would move the score up one or two points

GPU cost is estimated from the paper's own ledger: about $0.80 per rental, with RTX 5090 desktop hosts at about $0.35–0.70 per hour on Vast.

**+1 point (to 6, weak accept).** Items 1–3, about 3 weeks and about $30–50 of rentals.

| # | Change | Time | GPU cost |
|---|---|---|---|
| 1 | **Rewrite for clarity (W1).** Three findings; one host table; one glossary; consistent terms; audit, time-model validation and pre-registration moved to the appendix; MLSys template, anonymised, 10 pages. | 1–2 weeks | $0 |
| 2 | **A link-aware, read-once lower bound (W2).** Per layer, choose which misses MIN-style admissions go over the link and which bypasses go to the CPU. Report it beside Eq. 1, and replay that schedule in the job-102 microbenchmark with admissions forced over PCIe on the same three hosts plus two slow-link hosts. | ~1 week | ~$5 |
| 3 | **A held-out workload (W4).** Re-run Table 2 and the 2×2 factorial at the two host-bound gpt-oss budgets on chat and code prompts with ≥1k-token outputs, on host B or S and 3 panel-class hosts. Report TTFT. | 3–4 days | ~$20–40 |

**+2 points (to 7, accept).** Add the following; about 6–8 weeks of total effort and under $150 of rentals.

| # | Change | Time | GPU cost |
|---|---|---|---|
| 4 | **A third model family with many small experts.** For example Qwen3.6-35B-A3B or DeepSeek-V2-Lite, through the oracle factorial on 3 hosts, plus 3–4 more hosts with a link-to-CPU ratio below 0.5, so the crossover rests on more than one machine. | ~1 week | ~$25–40 |
| 5 | **A deployable spend of foresight that gains (W5).** For example read-once admissions with a link-aware per-machine split plus layer-ahead or short-horizon prefetch. It should show a statistically clear net gain over the deployed cache on the fast-link panel hosts, and break even on slow-link ones. | 2–3 weeks | ~$20–30 |
| 6 | **Overlap measurement.** Nsight timelines of the deployed cache and of MIN-prefetched at the host-bound budgets, so "the rest" is decomposed quantitatively rather than by narrative. | 2–3 days | ~$5 |
| 7 | **Task-accuracy parity.** Full AIME-25 decode to completion at 25% against an all-VRAM reference, on one host. | 1 day | ~$10 |

Fixing W6–W8 (audit sensitivity, the §5 wording, the cap's host set, the admission tie-break, the averaging statement) costs no GPU time and about a day of editing.
