# Review 9a: MLSys 2027 program committee, main track

**Submission:** *Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth* (anonymous)

**Materials read**
- `paper/paper.pdf`: 32 pages, built 6 October 2026, 13:48 CDT. I read the main text (pp. 1–9) and all of Appendices A–K (pp. 13–32) from `pdftotext -layout`, with `paper.tex` and the generated `wsg_*.tex` and `tab_*.tex` files open alongside to trace where each number comes from.
- `paper/supplement.pdf`: the prediction scorecard. I read its front matter, the tables for jobs 073–081, and the job 104 table. I checked the clause totals of every job against `prereg/scorecard*.json`.
- The artifact:
  - `scripts/job100.py`, `job101.py`, `job103.py`, `linkaware.py`, `factorial_shapley.py`, `factorial_paper.py`, `panel_099.py`, `hostdep_model.py`, `w50_rebaseline.py`, `wsg_tables.py`, `wsg_numbers2.py` and `speed_limit.py`;
  - `mosl/cachesim.py`;
  - `jobs/104_minadm@vast.sh` (predictions header) and `jobs/ec2/readsched.cu`, `fetch_table.py`;
  - raw rows in `results/081`, `089`, `096a/b`, `099a–j`, `100a/c`, `101a`, `102a–c`, `103a/b/d/e`, `104a–c`, and the trace `084c/route_aime25_gptoss.npz`;
  - the JSON files under `prereg/` that hold the scorecards, the link-aware bound, the read schedule, the W50 rule and the audit.

**Independence statement.** I wrote this review without opening anything under `reports/`, any `prereg/*outcome*.md` file, or any file named like a review, plan or progress log. I did not read commit messages. To order predictions against measurements I used only `git log --format='%h %ad'`. I modified no file in either repository. Every rerun used a copy in `…/scratchpad/review9a/repo`, made without `reports/` and without the outcome notes. Every independent recomputation used my own scripts in `…/scratchpad/review9a/chk`. The one exception is the review file itself, which I was asked to write to this path.

**Desk-level issues for the chairs (not scored)**
1. **Template.** The paper is built with `mlsys2025.sty`. The 2027 style file should be used.
2. **Length.** The main text ends on page 9, and the references start in the second column of that page. This appears to be inside the usual 10-page limit. The remaining 20 pages of appendices are allowed. However, the main text leans on appendix tables for claims it makes itself: §3 cites Tables 17 and 18, and §4 cites Appendix C.
3. **Anonymity of the PDF.** The PDF is anonymised: the author field reads "Anonymous Authors", and the affiliation and email are placeholders.
4. **Anonymity of the artifact.** The artifact as I received it is not anonymised. `paper/paper.tex` carries the author's name and a GitHub no-reply email in `\mlsysauthor` and `\mlsyscorrespondingauthor`. The paper also refers to "the public gpu branch" and to "the ledger in the repository", and its preregistration claims can only be checked through commit timestamps on that public history. The chairs should make sure that any artifact link given to reviewers is an anonymised mirror, or that a third party attests to the timestamps.
5. **Reference hygiene.** Zhang et al. 2020a and 2020b give the same DOI (10.1145/3379472), so one of them is wrong. ATSInfer is cited in the text as "(ats, 2026)" because its bibliography entry has no author. "Zhang, Gao, and Mitra" has no given names. I spot-checked three 2026 preprints (FreeToken arXiv:2608.16157, Zhang arXiv:2608.07911, WiSP arXiv:2606.21868) and all three exist.

---

## 1. Summary

The paper studies batch-1 decode of MoE models whose experts live mostly in host DRAM, on rented RTX 5090 desktop machines. It runs gpt-oss-120b (MXFP4) and Qwen3-30B-A3B (BF16) on AIME-25 prompts, with a llama.cpp-based expert cache that the authors built.

The paper has four parts.

1. **A bound (§3).** Eq. (1) is a roofline-style lower bound on time per token. Its inputs are:
   - the fewest host reads any exact-routing policy with C slots per layer can make (Belady's MIN with bypass, measured on a routing trace);
   - the highest host read rate the machine's probe measured;
   - the GPU's datasheet bandwidth.

   Measured against this bound, the authors' cache reaches 25–43%, and FreeToken and stock llama.cpp sit lower. The cache leads FreeToken at 11 of 12 cells and runs 2.0–4.0× stock llama.cpp. On the paper's single scoring procedure, 20 published "in-class" measurements reach a median 13.6% of their own bounds.

   Two checks support the bound's host term. A no-compute microbenchmark (job 102) replays MIN's per-layer reads and reaches 86–96% of the host term. A linear-program variant that forces every admission over PCIe raises the bound by at most 1% at the 11% budget and 7% at 25%.
2. **In-engine oracles (§4).** The engine reads future routing from a recorded trace and crosses two choices: *what to cache* (the deployed policy's set or MIN's set) and *how to load* (CPU then background copy, two reads; or a copy in the step, one read). This runs on about 15 hosts.
   - MIN's set read once beats the deployed cache by 16–51% on three hosts.
   - A two-player Shapley split shows the two choices mostly interact.
   - The gain depends on the host's link-to-CPU read ratio.
   - The newest result (job 104): among MIN's hit-optimal schedules, the one with the fewest admissions (computed per host by an LP) beats the greedy one by 0.08–0.17 at gpt-oss 11% on three machines. It turns a loss into a gain on the one machine whose link reads at 0.29 of its CPU rate.
3. **The price of foresight (§5).** The oracle is limited to a W-token window, either exact or with random errors.
   - In the engine, short windows are worth little against the deployed cache.
   - In a trace simulator across nine models, the window that closes half of MIN's read saving holds about 0.65·C distinct experts per layer. In leave-one-model-out prediction this rule does about as well as a two-constant power law.
   - Learned admission orders, speculative batching and forecasters (ridge, GRU) do not approach that horizon.
4. **Methodology.** Every job's predictions were committed before the machine started, 1,204 clauses (jobs 073–104) are scored in a supplement, and every number in the paper is generated by a script.

## 2. Strengths

- **S1. Unusually disciplined measurement.**
  - Each job's predictions are committed before launch. I confirmed this from timestamps for jobs 102, 103 and 104, where the job started 1–2 minutes after the commit.
  - Failures are reported openly: 142 of 565 clauses for jobs 073–098 failed, and 24 of 47 for job 104.
  - The host is treated as the unit of analysis. The paper reports relaunch stability and crossed bootstraps, and follows Hoefler and Belli's twelve rules (Appendix A).
  - This is far above the norm for MLSys systems evaluations, and the paper is a useful model of how to run cloud-rented performance studies.
- **S2. The artifact is excellent.** Every one of my 19 checks held, apart from the four noted caveats (§6). Two of them were reruns:
  - `scripts/job103.py 104` in a scratch copy reproduced `scorecard_104.json`, `tab_job104.tex` and `wsg_job104.tex` byte for byte.
  - `scripts/hostdep_model.py` reproduced `wsg_hostdepmodel.tex` byte for byte.

  My own MIN-with-bypass simulator reproduces R⋆ exactly: 38.315 reads per token at C=14 and 15.340 at C=32. My own two-stage linear program reproduces the fewest-admission count (12.399 per token, with an integral solution).
- **S3. A clear, honest yardstick.** Eq. (1) is simple and its assumptions are stated (§3, Limitations). The authors test its tightness in four ways:
  - a read-schedule microbenchmark;
  - a link-aware LP bound;
  - Table 17's one-assumption-at-a-time tightenings;
  - a two-path split.

  Reporting systems as "% of the machine's bound" is a better habit than speed-ups over self-chosen baselines.
- **S4. The time domain, inside a real engine.** Trace-level MIN gaps and lookahead value have been reported before. Measuring oracle states inside the engine shows several things trace studies cannot:
  - the second read wastes foresight;
  - Belady prefetch without bypass loses where host memory binds hardest;
  - the outcome depends on the link-to-CPU ratio;
  - which hit-optimal schedule is used matters.

  These findings are actionable for anyone building an offloading cache.
- **S5. A strong system baseline.** The cache runs 2–4× stock llama.cpp at equal GPU expert memory and leads a recent competitor (FreeToken) at 11 of 12 cells. FreeToken was also given a five-setting tuning job and a long-output test on 2,048 tokens of held-out problems.

## 3. Weaknesses (ranked)

**W1. The title's question is only partly answered, and nothing realisable captures the oracle gains.**
- After every oracle, the best state reaches 46–73% of the bound on O4/O5. "The rest" is still 30–62% of the gap on fast-link hosts (Table 4).
- §4 names three "visible parts" of the rest: an imprecise prefetch, partial overlap, and a GPU at about half its datasheet rate. None of them is measured separately in the engine.
- The "accounting as a model" (Appendix E) closes by construction: the GPU term is the residual, and "overlap" is defined by Eq. (1)'s form.
- On the realisable side the paper is negative: "No online policy, learned admission order, speculative batch or forecaster we tested comes near it."
- So the paper tells a cache designer where *not* to look, and roughly how large the prize is, but not how much of the prize is reachable.

*Fix:* close the rest with measurements rather than definitions.
1. Extend `readsched.cu` with synthetic batch-1 GPU work per layer, at the per-layer kernel times measured with all experts in VRAM. This shows whether the overlap the bound assumes is achievable.
2. Replace the datasheet GPU term with a measured per-layer kernel floor.
3. Run an oracle whose copies land at the ideal step, with no queue cap. Appendix F already simulates this at d=1; doing it in the engine isolates prefetch timing.

**W2. The oracles are hit-optimal, not time-optimal, so "what foresight is worth" is a lower bound tied to one oracle family.**
- Job 104 shows that the choice *among* hit-optimal schedules moves the speed ratio by up to 0.17.
- The authors already trace the misses-versus-admissions frontier in `scripts/linkaware.py`. At C=14 it runs from (38.3 misses, 12.4 admissions) to (50.7, 1.7). Yet no frontier point that trades hits for fewer link crossings was run in the engine.
- On slow links, the time-optimal oracle almost certainly lies off MIN's hit-optimal face. The paper itself cites CHOPT as "a different problem".
- The price of foresight (W50 ≈ 0.65·C) is defined on *reads* relative to admit-every-miss. Yet §5 reports that the share of time gained lags the share of reads saved by 0.05–0.20 on slow links. So the forecaster target in the conclusion is a read-domain target presented as a time-domain one.

*Fix:*
1. Run a cost-aware plan in the engine, for example the LP at a per-host μ chosen from B_p/B_c, or an LP that minimises Eq. (2) time directly. Use the hosts of jobs 101–104.
2. Restate W50 in time, using the panel windows (Fig. 3 markers) where the data allows.
3. Rename the §5 quantity, for example "read horizon".

**W3. The newest headline claim rests on thin evidence that was gathered hours before submission.**
- The abstract and the conclusion lead with "the greedy one loses on a machine with a slow link, where the set with the fewest admissions gains".
- That rests on three machines at two budgets, one launch each (job 104, run 17:34–18:25 UTC on the day the PDF was built). The loss-to-gain crossover rests on one machine, Pf at a ratio of 0.29.
- Pf's loss under the greedy schedule is stable: I recomputed fetch/base as 0.875, 0.860, 0.859 and 0.863 across jobs 099f, 101a, 103a and 104a.
- The fewest-admission gain on that machine (1.035, interval [1.023, 1.052]) was seen once.
- 24 of the 47 preregistered clauses failed. One of them is the predicted *direction* of the interaction (it grew instead of shrinking). Loaded by the CPU at gpt-oss 25% on Pg, the fewest-admission set is 0.047 *behind* the greedy one.
- On the positive side, job 103 provides an unplanned A/A test, which the authors do not use. Its "plan" configurations silently ran the greedy schedule, and they agree with the true greedy runs within 0.008 in speed ratio on all 16 comparisons. This supports reading the 0.08–0.17 differences as real.

*Fix:*
1. Rerun job 104 on at least four further machines below a link-to-CPU ratio of 0.5, with two launches each, and add Qwen3 12.5%.
2. Report the job 103 A/A agreement.
3. Until then, move the fewest-admission result out of the abstract's lead sentence.

**W4. Scope.**
- Every in-engine result is batch-1 decode on desktop hosts with RTX 5090s, for two models, on the first 20–30 AIME-25 problems × 256 teacher-forced tokens. The RTX 4090 and 3090 appear only in the system grid.
- These prompts "were also used while developing the system". The deployed policy's constants (half-life 16, κ=1) and the fetch-table model were tuned on this workload.
- Output parity is shown by teacher-forced KL divergence, not task accuracy. Free-running greedy outputs are almost never identical to stock llama.cpp's over 256 tokens: 0–3% in job 073 P4. As a result, the Table 2 systems decode slightly different text.
- The quantitative takeaways ("25–43% of the bound", "0.65·C", "16–51%") are tied to this setting.

*Fix:*
1. Add a held-out workload, such as chat or code, to Table 2.
2. Add a third model in the engine. Qwen3.6-35B-A3B already runs in job 086.
3. Report one task-accuracy parity number (AIME-25 accuracy at full reasoning length).

**W5. The "% of bound" numbers are measured against a loose bound.**
- B_host is the highest of all probe samples. On host B that is 87.5 GB/s, while the concurrent CPU+link median is 78 (Table 2 header), about 12% higher. The GPU term uses the datasheet rate, although batch-1 decode with everything in VRAM reaches 52% of it on gpt-oss and 61% on Qwen3. Slots are per layer.
- The paper is explicit about all of this, but the headline percentages use the loose version. "Nearly reachable" holds only for the host-read term in a microbenchmark with no compute and no admission constraint.
- §3's "with every assumption of Table 17 relaxed at once, our cache stands at 33–57% … on host B" is the more defensible headline. Note too that this sentence says "relaxed" where the assumptions are tightened.

*Fix:*
1. Make the tightest defensible bound the primary column of Table 2 and of the abstract: two paths, median probe sample, measured GPU rate, and the link-aware LP.
2. Keep the loose bound as the "speed of light" in the appendix.

**W6. The audit of published systems is too heterogeneous to headline.**
- The "median 13.6%" of 20 in-class rows (Introduction, §3, Conclusion) mixes several things:
  - datasheet ceilings, not probed ones;
  - traces of sibling model variants;
  - per-layer bounds applied to pooled designs.
- 4 of the 20 in-class rows (ProMoE) also have an imputed per-layer capacity. I confirmed this from `prereg/audit/audit.json`.
- Appendix J lists these caveats itself. Even so, the number reads as a performance comparison with 13 named systems on hardware the authors never ran.

*Fix:*
1. Move the audit to the appendix, or report it only with a sensitivity band (datasheet vs probed, imputed rows excluded).
2. Drop it from the abstract and the conclusion.

**W7. Presentation** is a major barrier to the paper's impact; see §4 below.

*Fix:*
1. Cut the main text to three findings with a glossary up front.
2. Move the prediction log to the supplement.

**W8. Smaller issues.**
- **Two summaries of the same runs.** Table 11 reports times as the inverse of mean speed (1000/mean tok/s), while the text reports mean ms per token. The running example's "15.2 ms" (from raw 096a: 15.163) therefore appears as 15.13 in Table 11, and "20.6" as 20.59.
- **"Even the best oracle reaches only 46–73%".** This is computed for MIN prefetched (`both3p`) only. The unpaced two-step-lead state (`both2`) reaches 74.7% at O4 Qwen3 12.5% (24.06 ms against a 17.97 ms bound).
- **The greedy admission count depends on tie-breaking.** 21.8 is the artifact's tie-break. An incumbent-preferring greedy rule that I implemented gives 21.4, so the greedy-versus-fewest contrast is slightly overstated.
- **The link-aware bound interpolates linearly between only five μ samples.** For Pf at C=32, the frontier vertices alone give a 2.8%/11.1% raise; interpolated chords give 0.7%/6.9%. The true LP frontier lies on or below the chords, so "at most 7%" is still a valid upper statement. The computation should state this, or solve the frontier parametrically.
- **Host attrition is not reported in the paper or supplement.**
  - Job scripts exist for `100d` (12400F) and `103c` (7800X3D), but neither produced results.
  - Neither appears in the paper or the supplement. Job 103's scorecard lists four hosts, and job 100's lists five, including 100e (13900KF) with two untested clauses.
  - For a preregistered study, launched hosts without data should be listed.
- **Clause counts double-count deterministic counters.** 15 of job 104's 24 failures are five deterministic counter facts, each repeated on three hosts with identical values. Counts of this kind overstate the information in "N of M clauses".
- **The cost line covers jobs 058–101 only.** It omits jobs 102–104 and storage costs.

## 4. Clarity: 2 / 5

The writing is precise sentence by sentence. Read as a whole, it is very hard going: almost every sentence carries a range, a host nickname or a job number, and the reader must hold about 15 configuration names, 20 host labels and 30 job numbers in mind. Specific passages:

- **Abstract.** It nests three claims in one sentence ("MIN's own reads … reach at least 86% of this read time, yet our llama.cpp expert cache, faster than FreeToken at 11 of 12 configurations, stands at 25–43% of the bound"). It then introduces "MIN's many hit-optimal sets", "the greedy one" and "an order-free accounting" before defining any of them. *Fix:* use one sentence per finding, define MIN with bypass once, and drop the Shapley term from the abstract.
- **Two vocabularies.** Table 1 and Figures 1–2 use descriptive names (deployed, single read, MIN 1 read, MIN prefetched). Tables 10–13 and the scripts use internal ones (online, foa, bypass, fetch, both, both3p, paced, nb2, hit-opt, lead). Table 11 mixes the two. *Fix:* use one name set throughout. A `tab_glossary.tex` exists in `paper/`, but it is not in the PDF; include it.
- **Host labels.** Host A, B and S; O1–O6; Pa–Pj; "Pf again"; "285K, 40 GB/s link"; "9950X, x8 (0.54)". Readers have no index. *Fix:* add one host table (CPU, B_c, B_p, B_p/B_c, jobs) and refer to hosts by their link-to-CPU ratio.
- **§4, "An order-free accounting" and Table 4.** The caption runs to eight lines and defines "interaction" asymmetrically ("how much more MIN's set is worth loaded once than loaded twice"). The table reports load alone, cache alone, interaction, two Shapley values, prefetch and rest, with panel ranges in small type. The key message is "the interaction dominates", and it is buried. *Fix:* show a 2×2 bar chart per host class, with Shapley values as a footnote.
- **§4, "The rest."** This paragraph lists three causes without a number per cause and then moves on to slack in the bound. *Fix:* give a measured or bounded share for each cause, or say plainly that the rest is unexplained.
- **§5.** It switches baselines mid-section, from "against the deployed cache" to "against admitting every miss". It also reports shares in time and in reads side by side. The horizon sentence uses "number" as a verb and is split by a float ("…the experts a layer requests within W50 number about 0.65 C"), so it reads like a typo. "Its value is the unit" is cryptic. *Fix:* use one baseline per subsection and state the rule as an equation.
- **§3, "Published systems".** "Inside the bound's class" is not defined in the main text. The definition (trace-scored, not speculative, not lossy) is only in the code.
- **Appendix B.** It is a page-long paragraph listing failed clauses. A table would serve better, and the supplement already has one.
- **Figures 1 and 3.** These are 6- and 4-panel strips with small markers and "grey ticks". In print the intervals are unreadable.
- **Wording.** "Every assumption of Table 17 relaxed at once" should be "tightened".

## 5. Novelty against related work: 3 / 5

- **The bound.** It combines a roofline (Williams et al.) with MIN-with-bypass reads, which are classical (Belady; Carlisle and Lloyd; Jain and Lin), over probed rates. Close predecessors, as the paper itself describes them:
  - Budgeting Bytes, bytes over bandwidth for a given fetch set;
  - WiSP, PCIe turnover;
  - Zhang 2026b, a trace-level contract with a 44–46% causal-versus-MIN-with-bypass gap;
  - Liang et al. 2026b, an oracle cache with m-token lookahead against C/k;
  - SAEM, a perfect-knowledge planner worth ≤14%.

  The bound is therefore a careful application of known pieces rather than a new technique.
- **What is new:**
  1. Oracles inside a real engine, run as a factorial and treated as a measurement across about 15 machines.
  2. The finding that *how* foresight is spent (bypass, a single read, which hit-optimal schedule) decides whether it pays, and that this depends on the link-to-CPU ratio.
  3. The fewest-admission schedule as an oracle.
  4. A horizon rule stated in distinct experts.

  These are useful but incremental. The distinct-expert rule beats the power law on exactly 13 of 26 held-out points by my recomputation, which matches the paper's own "not distinguished". It also uses each held-out model's own D(W) curve.
- **Method.** Preregistration with a machine-scored scorecard is novel for this venue, but it is a methodological contribution rather than a systems one.

## 6. Soundness and the artifact checks: 4 / 5

The methodology is careful, and every number I checked reproduces. I do not give 5 for four reasons:
- the oracles are not time-optimal (W2);
- the newest crossover rests on one machine (W3);
- the gap accounting closes partly by definition (W1);
- the headline percentages use the loosest bound (W5).

None of these is an error.

| # | Claim (where) | What I did | Result |
|---|---|---|---|
| 1 | Table 3 (job 104): all 24 speed ratios and copies per token | Parsed `ec_g_C{14,32}.jsonl` and `st_*.json` in `104a–c` with my own script. Speed = mean(base ms)/mean(X ms), paired by problem. Link/CPU from `concur.txt` (zero-copy 16 MB link rate; CPU rate interpolated at the helper count). | **Held.** For example, Pf at 0.288: 0.941 / 1.113 / 0.863 / 1.035, and copies 19.81 → 11.84. Pg at 0.629, 25%: CPU-loaded fewest set 1.151 < greedy 1.198. |
| 2 | Fewest-admission set in the step on Pf, 1.04× with interval [1.02, 1.05] (§4) | 10,000-resample paired bootstrap | **Held:** 1.035 [1.023, 1.052] |
| 3 | 0.60–0.73× the in-step copies, 2.5–5.3% more misses, +0.08–0.17 at gpt-oss 11% (§4) | Counters from `st_*.json`, then fetchplan/base minus fetch/base | **Held:** 0.597 and 0.726; +2.52% and +5.26%; +0.172, +0.127, +0.082 |
| 4 | R⋆ and the 12.4 vs 21.8 admissions per token at gpt-oss 11% (§3, `linkaware.py`) | Wrote my own MIN-with-bypass simulator on the 084c trace. Wrote my own two-stage LP (maximise hits, then minimise admissions at equal hits; HiGHS, 36 layers). | **Held, with a caveat.** R⋆ = 38.315 (and 15.340 at C=32), exact match. Fewest admissions 12.399, integral solution. My greedy gives 21.39 against 21.8, a tie-break dependence. |
| 5 | Link-aware bound raises the bound by ≤1% at 11% and ≤7% at 25% over 24 probes (§3) | Recomputed T_link for Pf (two launches) and Pd from `linkaware.json` and the probes, with chords and with vertices only | **Held** with chords (0.73–0.87% and 6.87–7.13%). Vertices only give 2.8% and 11.1%. The claim is conservative because the true frontier lies on or below the chords. The 24 probes include at least five relaunches of the same machine, so they cover at most 19 distinct machines; the text correctly says "probes". |
| 6 | Table 18 (job 102): every share, "86–96%", and repetitions agreeing within 0.9% | Parsed `readsched_C*.txt` and recomputed B_host from `concur.txt`. Term = R·S/B_host. Read `readsched.cu` to confirm whole-buffer CPU reads and 4 GiB rotation. | **Held** exactly: Pd 7.04 ms, 86/93/35/92%; largest repetition gap 0.88% |
| 7 | Table 2, host B: 1.294 [1.278, 1.312], 1.275, 1.154, Qwen3 1.032; host S: 1.207 and 0.974 [0.962, 0.987]; leads at 11 of 12; 2.0–4.0× llama.cpp | Paired bootstrap on `bs1.jsonl` (jobs 081, 089), keyed by label and launch | **Held.** Note that ours is launch 2 and llama.cpp launch 1, so the ratio mixes launches. |
| 8 | Bound of 172 / 430 tok/s on host B; 279 with the measured GPU (Table 2) | Eq. (1) by hand: B_host 87.5 GB/s from `081/concur.txt`, S and D from `speed_limit.py` | **Held:** 172.3 / 430.4; 281 at a rounded 52% |
| 9 | Running example on O4: 10.0 / 20.6 / 15.2 / 13.8 ms; shares 4.5 / 3.9 / 51.5 / 26.1 / 25.5 / 12.7 / 35.7% (§4) | Raw `096a ec_g_C14.jsonl` plus my own Shapley arithmetic | **Held** exactly. Table 11 gives 20.59 / 15.13 / 13.78 for the same runs because it uses a different summary (W8). |
| 10 | Job 103: loading in the step costs time at 3 of 4 host-budgets on two more slow-link machines; greedy one read 1.02–1.05× at 0.40 and 0.48 | Sign of the load Shapley value from raw 103b and 103e | **Held:** −, −, − and + (103b at 11%); 1.019 and 1.052 |
| 11 | Relaunches: 4 machines within 1.9% of their first launch, every ratio within 0.030 (§2) | Pd, Ph, Pf and Pg relaunched (jobs 100a, 100c, 101a, 103a, 104a, 103d, 104c) against job 099 | **Held:** largest gaps 1.9% and 0.0296 (100c, both3p). Pf loses with the greedy schedule on all four launches (0.86–0.875). |
| 12 | Panel hosts with the same CPU differ by 16–29% (§2) | Deployed ms on 099a/099f (285K) and 099e/099j (9950X) | **Held:** 15.5–29.0% |
| 13 | Audit: 20 in-class rows, median 13.6%, quartiles 8.1–20.6% (§3) | Re-filtered `prereg/audit/audit_sol.json` with `audit.json` flags | **Held:** 13.63 [8.08, 20.62]. 4 of the 20 rows impute C. |
| 14 | Table 6 and D(W50)/C = 0.65 (0.61–0.72) (§5) | Re-ran leave-one-model-out from the stored per-model W50 values and D(W) curves in `w50_distinct.json`. I did not recompute W50 from the traces. | **Held:** 1.22/1.57/2.42, 1.19/1.51/1.67, 1.16/1.42/1.61. The distinct rule beats the power law at 13 of 26 points. |
| 15 | Scorecard totals (Appendix B): 565 → 218/193/142/8/4; 273, 216, 25, 33, 45, 47 | Counted statuses in `prereg/scorecard*.json`. Re-ran `scripts/job103.py 104` in scratch. | **Held.** The job 104 outputs are byte-identical. |
| 16 | Calibrated model within a median 2.2% (90th percentile 10%) on 14 hosts (§4) | Re-ran `scripts/hostdep_model.py` in scratch | **Held:** `wsg_hostdepmodel.tex` byte-identical |
| 17 | Table 5, Pd rows (job 100) | Raw `100a` rows | **Held:** 0.66/0.89/1.02/1.03/0.94/1.00/1.05/1.11 and the 25% row |
| 18 | "Even the best oracle reaches only 46–73% of the bound on O4 and O5" (§4) | Fastest state per cell from raw 096a and 096b, against the stored limits | **Partly held.** MIN prefetched spans 45.6–73.0%, but the unpaced `both2` reaches 74.7% at O4 Qwen3 12.5%. |
| 19 | Predictions committed before each machine started (§2, Appendix B) | `git log --format='%h %ad'` on the job scripts against `start_utc` in each manifest | **Held.** Job 102: commit 15:14:17Z, start 15:15:11Z. Job 103: 16:50:58Z, 16:52:18Z. Job 104: 17:33:12Z, 17:34:33Z. |

## 7. Questions for the authors

1. Why is the oracle family limited to MIN's hit-optimal face? Have you run, in the engine, a frontier point with μ>0 (fewer admissions, more misses), or a plan that minimises Eq. (2) time on the 0.29 and 0.49 machines? What would you expect it to add over the fewest-admission set?
2. How many further launches of Pf, and how many further machines below a link-to-CPU ratio of 0.5, do you plan for the fewest-admission result? Will you report job 103's unplanned A/A agreement (within 0.008)?
3. What do the headline percentages become if B_host is the concurrent median (78 rather than 88 GB/s on host B) and the GPU term is measured? Why not make the tightest bound the primary one?
4. Can the "rest" (30–62% of the gap) be split by measurement: microbenchmark with compute, in-engine copies with ideal timing, and a per-layer kernel floor?
5. Which tie-break rule produces 21.8 greedy admissions? A rule that keeps incumbents on equal next use gives 21.4.
6. Is the link-aware frontier solved exactly, or interpolated between the five μ values? With vertices only, Pf's raise at 25% is 11%.
7. Jobs `100d` (12400F) and `103c` (7800X3D) have scripts but no results, and neither the paper nor the supplement mentions them. What happened to them?
8. How sensitive are the deployed constants (half-life 16, κ=1, the fetch-table model) to the AIME development set? Is there a held-out workload other than AIME 2022 and MATH-500?
9. In free-running Table 2 runs the systems' outputs differ (0–3% identical to llama.cpp over 256 tokens). How much do hit rates vary with the decoded text, and does teacher-forcing the system comparison change Table 2?
10. Can W50 be stated in time per token, using the panel windows, rather than in reads?
11. Will the artifact be anonymised? `paper.tex` contains an author name and email, and the timestamps that back the preregistration claims are on a public branch.

## 8. Scores

| Criterion | Score | Rationale |
|---|---|---|
| **Overall** | **5 / 10** (borderline reject) | The rigour and the artifact are of top-paper quality. The contribution is narrower: a known-pieces bound, oracle measurements whose realisable lesson is mostly negative, a crossover resting on one machine, and a paper that is very hard to read. At a 20–25% acceptance rate this is just below the bar as written. A focused rewrite plus W2/W3 evidence would make it a clear accept. |
| Confidence | 4 / 5 | I recomputed 19 claims from the raw data and the scripts, including the newest jobs. I am less sure how the MLSys audience will weigh a characterisation study against systems novelty. |
| Novelty | 3 / 5 | The time-domain, in-engine oracle factorial and the "how and which MIN" findings are new. The bound and the horizon study extend Budgeting Bytes, WiSP, Zhang 2026b and Liang 2026b. |
| Soundness | 4 / 5 | Every check held, with four minor caveats. Conceptual limits: the oracles are not time-optimal, the accounting partly closes by definition, the headline bound is the loose one, and the crossover has n=1 machine. |
| Significance | 3 / 5 | Local MoE inference is a real use case, and the yardstick plus the "spend foresight once and sparingly" lessons are useful. But the results are tied to batch-1 RTX 5090 desktops and two models, and no realisable policy captures the measured headroom. |
| Clarity | 2 / 5 | Dense prose, two vocabularies, about 20 host labels and job numbers in the main text, key definitions only in the appendix or the code, and Table 11 inconsistent with the text. |

## 9. What would move the score up one or two points

Costs below use the paper's own ledger: 88 rentals cost US$70.8, about $0.80 per rental at roughly $0.4–0.7 per hour for an RTX 5090 host. Storage is extra.

1. **Rewrite and focus (+1 on its own).**
   - State three findings in the main text: the bound and its tightness; that foresight pays only when spent as MIN spends it, and which MIN depends on the link; and the read horizon.
   - Add a glossary and a single host table.
   - Use one name per configuration.
   - Move the prediction log, the audit and the FreeToken grid to the appendix or supplement.
   - Make the tightest bound primary.

   *About 1–2 weeks; $0.*
2. **A time-aware oracle and a sturdier job 104 (+1).**
   - Run the LP frontier at a per-host μ, or a plan that minimises Eq. (2) time, beside the greedy and fewest-admission sets.
   - Use about 8 machines spanning link-to-CPU ratios 0.25–1.0, including at least 4 below 0.5, with two launches each and gpt-oss at 11% and 25% plus Qwen3 at 12.5%.
   - Report the job 103 A/A check.

   *3–5 days of engineering, since `minadm_plan.py` and `linkaware.py` already exist; about 16–20 rentals × ~1.5 h ≈ $15–30.*
3. **Close "the rest" by measurement (+0.5).**
   - Add synthetic per-layer GPU work to `readsched.cu` (3 hosts).
   - Add an in-engine copy path with no queue cap and ideal landing.
   - Add a per-layer kernel floor from all-in-VRAM profiles.

   *About 1 week; about 6–8 rentals ≈ $5–10.*
4. **Generality (+0.5).**
   - Add a third model to the in-engine factorial (Qwen3.6-35B-A3B is already supported).
   - Add a held-out workload (chat or code) to Table 2 on hosts B and S.
   - Add one task-accuracy parity number (AIME-25 at full reasoning length, ours against stock llama.cpp with every expert on the CPU).

   *About 1–2 weeks; about 10–15 rentals plus 2 × ~3 h of long generation ≈ $15–30.*
5. **A realisable step toward the oracle (+1, if it works).** Build an online admission rule informed by the fewest-admission finding: admit only when reuse before eviction is predicted, in the spirit of Antoniadis et al.'s eviction-before-reuse predictions. Evaluate it in the engine on the panel. Even a 5–10% realisable gain on slow-link hosts would change the paper's message from diagnostic to constructive.

   *2–4 weeks; about $20–40.*

Items 1 and 2 together, at about 3 weeks and under $30 of GPU time, would move this review to a 6–7. Adding item 5 with a positive result would move it to 7–8.
