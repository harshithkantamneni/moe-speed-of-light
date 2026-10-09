# Review 28a: MLSys 2027 PC review of "Where the Seconds Go" (9 October)

## Materials read

- `paper/paper.pdf` (47 pp.): the main text, pp. 1–10, read in full; the appendices A–N (pp. 15–47) read in full except some of the long tables in Appendix D and M, which I only skimmed. I read `paper/main_body.tex` and `paper/paper.tex` alongside, to check how the macros feed the text.
- `paper/supplement.pdf` (the scorecard): I read the front matter and the job 073–075 block, and used Tables S1–S3 for reference.
- `paper/ieee-paper.pdf` and `paper/ieee-supplement.pdf`: skimmed for the IEEE-format section. I rendered pp. 1–3 and 6, plus a montage of pp. 3–10.
- Job-script headers on the gpu branch (`/home/claude/gpu-branch/jobs/`): 099, 102, 109, 110, 113, 114 and 115, plus per-host wrappers. I also read the file diffs of the eight commits that touched a job script after one of its rentals had started: 89fe229, 718bf41, 3701cea, a8e2f92, 89c5c3b, f9ac8c7/346bd34, 58d5454 and 3649594.
- Raw results (`/home/claude/gpu-branch/results/`):
  - `ec_*.jsonl` rows, `st_*.json` counters and `concur.txt` probes for jobs 081, 089, 096a/b, 099a–j, 100b/f, 101b, 102a–c, 103–115;
  - the routing trace `084c_gptoss_trace@vast/route_aime25_gptoss.npz`;
  - Nsight `prof_C*.json` for jobs 069c and 105;
  - `stream.txt`, `concur_idle*.txt`, `v0.txt`, `validity.txt` and `fetch_table_law_gptoss.json`, where relevant.
- `gpu/vast_ledger.json`, and git history of the gpu branch (`%h %ct %ad` and `--name-only`).
- Authors' scripts, read for definitions only:
  - `scripts/job114.py` and `scripts/job109.py` (`load()` and its definitions);
  - `host_rates()` in `scripts/speed_limit.py`;
  - the header and `rows()` of `scripts/decomp_measured.py`;
  - `machines()` in `scripts/fig_decomp.py`;
  - the docstring of `scripts/audit_ours.py`.
- `prereg/reanalysis_hosts.json`: I used it only for the list of directories that make up the 15-GPU panel.
- All my own code is in `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/rev28a/`:
  - `lib.py`: loader;
  - `j114.py`: Tables 4 and 5;
  - `panel.py`: Table 4 panel and new columns;
  - `t6.py` and `trend2.py`: Table 6 and the trend;
  - `t12.py`: Tables 12 and 13;
  - `t3.py`: Table 3;
  - `t31.py`: Table 31;
  - `share.py`: the share of the bound, and Eq. (3);
  - `few1r.py`: Few-1R pooled;
  - `rstar.py`: R⋆ from the trace;
  - `dw.py`: distinct experts;
  - `noise.py`: noise between rounds;
  - `tgpu.py`: T_GPU;
  - `rule4.py`: read once vs read twice;
  - `regtime.py`: registration timing.

## Independence statement

- I did not open anything under `reports/` except to write this file.
- I did not open `prereg/*outcome*.md`, `research_notes/`, `apply/`, `paper/paper_v1_prereview.tex`, or any file named like a review, number check, plan or progress log.
- I did not read any commit messages. My git commands used only `%h`, `%ct`, `%at` and `%ad` formats, file lists and file diffs.
- Every number in the claims table below was recomputed by my own code from the raw rows, counters, probes and trace. I read the authors' scripts only to learn definitions: how a speed ratio is averaged, B_host, the stability gate, which launches form the panel, and the consumer/server split.
- I modified nothing in either repository and made no commits.

---

## Summary

The paper studies batch-1 decode of MoE models whose experts mostly live in host DRAM. Its main model is gpt-oss-120b on rented RTX 5090 machines; Qwen3-30B-A3B appears for the system comparison and the factorial. It makes five contributions.

1. **A bound.** Eq. (1) prices MIN-with-bypass host reads (R⋆) at the machine's highest probed host rate. A "demand" refinement (Eq. 2) and an "ordered" refinement (Eq. 3) follow.
2. **A llama.cpp expert cache.** It is faster than FreeToken at 11 of 12 configurations and than stock llama.cpp at all of them. At gpt-oss 11% it reaches 31–54% of Eq. (1) on 25 consumer machines (post hoc scope).
3. **A decomposition of the gap with oracles inside the engine.** The oracles change:
   - what is cached: MIN's set;
   - how each admission is read: once instead of twice;
   - when copies are issued: read-ahead.
   This is done on two read paths: the deployed one, with a per-machine "fetch table" of in-step copies, and a CPU-only one with those fetches off. The authors find that the fetch table displaces MIN's set. With the fetches off, MIN's set alone closes 19% of the gap and MIN-1R 36%. About half the gap remains even with read-ahead.
4. **Policies without foresight recover ≤ 6% of the read-ahead oracle's gain,** and a trend in the link-to-CPU ratio transfers from RTX 5090s to RTX 4090s.
5. **A trace study (post hoc).** Half of foresight's value needs a window holding about 0.65·C distinct experts.

Every experiment is pre-registered in job-script headers. Appendix D and the scorecard report all clauses, including failures.

## Strengths

1. **Data integrity is exceptional.** I recomputed more than 30 quantitative claims from the raw per-problem rows, counters, probes and routing trace with my own code (table below). Every one reproduced to the reported precision, including:
   - R⋆ from my own MIN-with-bypass simulator: 38.315 and 15.340 reads per token;
   - every cell of Tables 4, 5, 6, 12, 13 and 31, and the gpt-oss/Qwen3 12.5% cells of Table 3;
   - the frozen RTX 5090 trend constants of job 110, which my own least-squares refit on the 15 panel machines reproduces to four decimals.
2. **Pre-registration is real and auditable.** For all 136 rentals of jobs 073–115 in the ledger, the main job script and the host wrapper were committed before the rental started (minimum lead 2.7 s, as stated). For jobs 093–115 the count is 107, matching the paper. The eight later commits to job scripts are comment-only amendments: host substitutions and relaunch rules. The one exception is job 100's added ninth prediction, which the paper discloses. Registered thresholds in Table 1 match the headers of jobs 099 (P5), 102 (P2), 109 (P1, P3, P7), 110 (P1), 113 (H3–H6), 114 (H2–H6) and 115 (H3–H8).
3. **Failures are reported, not buried.** The paper reports the timing control that failed (Table 12), the panel interaction that failed on one machine, the closed-form account that failed on new machines (Appendix H), and job 110, which was voided. Thresholds that were easy to meet are marked "loose". The noise analysis across rounds and rentals (Appendix D) is a model of good practice.
4. **The bound-as-yardstick idea is useful and simple.** Eq. (1) needs only a routing trace and one probe. The audit of published systems (median 13.6% of their bound) is a provocative and useful framing for the offloading literature. I re-derived the 13.6% from Table 33's rows.
5. **The decomposition's main qualitative findings are well supported and actionable.**
   - The read-ahead oracle still leaves 43–52% of the gap.
   - Policies without foresight read about 1.6–1.75·R⋆, about as much as the deployed cache.
   - The engine's own fetch table can erase the benefit of a better admission set.
6. The micro-benchmark (Table 31, 86–96% of the read time) usefully separates "the bound's read time is reachable" from "a model run reaches it".

## Weaknesses (most important first)

### 1. The central decomposition is specific to this engine's read path, and the builder rules over-generalise it

- The 2×2 that carried earlier versions ("MIN's set pays only when read once") turns out to be mostly an artifact of the engine's probe-derived fetch table. The paper now says this itself.
- That table is fragile. Machines 114b and 114f are two Ryzen 9 5950Xs from one provider with near-identical probes (B_c 33.33 vs 33.48 GB/s, B_p 26.2 both). Their tables are `0,0,0,0,0` vs `0,0,1,2,3`, and MIN-2R on the deployed path goes from 1.12× to 1.01×.
- On the fetches-off path, the "together > apart" interaction averages 9% of the gap with a t-interval over machines of [−5, 23]. It is concentrated on the two 5950Xs (21%); on the three Intel machines it is −3 to 3%.
- What generalises is the narrower claim: "MIN's set pays read twice when the read path holds it". It rests on five machines. Two of them are twins from one provider, and one of those (114f) is the same GPU as 109f and 112g, rented for the third time.

### 2. New finding: Rule 4 and the Section 5 heading are contradicted by the authors' own registered data at 25%

On the three Intel machines of jobs 114 and 115 (link-to-CPU 0.54–0.57, all "fast" by the paper's 0.5 line), MIN's set read twice with the fetches off is faster than MIN-1R at gpt-oss 25%. MIN-1R ÷ MIN-2R(off), with paired 95% bootstrap intervals over problems:

| Machine | gpt-oss 25% | gpt-oss 11% |
|---|---|---|
| Core Ultra 7 265K | 0.972 [0.965, 0.979] | 1.05 |
| Core i9-14900K | 0.962 [0.957, 0.969] | 1.02 |
| Core Ultra 9 285K | 0.960 [0.955, 0.966] | 1.03 |
| Ryzen 9 5950X (twins) | — | 1.24 |

- Job 113's Core Ultra 9 285K (ratio 0.56) shows the same at 25%: Few-2R with fetches off ran 1.36× the deployed cache, against Few-1R's 1.29×.
- This contradicts Rule 4 ("Where the link is fast, copy admitted misses into their slots in the step"), which is stated without a budget scope.
- It also contradicts the Section 5 heading "Reading each admitted expert once pays most". That heading rests on the factorial (Fig. 3), whose two-read arms ran with the fetch table and so did not hold MIN's set; Section 4 shows exactly this confound.
- The single read wins clearly only where both of these hold:
  - the link is fast relative to the CPU, as on the 5950Xs (0.78–0.79);
  - the budget is the smallest.
- The 0.5 threshold is not where the crossover lies at 25%. This should be stated in Sections 5, 7 and 9, and in the abstract's "reading each admission once pays most".

### 3. Scope is narrow, and it is narrower than the title and abstract suggest

- One engine (the authors'), one prompt set (AIME-25, also used in development), teacher-forced decode of 256 tokens, batch 1, and gpt-oss at two budgets for every decomposition result.
- A convenience sample of cheap rented machines, with few slow links.
- The rules are explicitly "hypotheses" (Section 9), but the abstract and conclusion state them as findings.
- The 25 consumer machines span very different processors, yet the per-machine spread (31–54%) is the headline number, and its scope was set post hoc.

### 4. "Bound" is relative to a probe that the engine itself appears to exceed

- By the paper's own Eq. (4) accounting, the engine's reads imply a rate more than 5% above the probe's highest reading on 12 of 84 launch-budgets (up to 1.22×).
- Either the probe underestimates achievable bandwidth, or Eq. (4) is wrong. Eq. (4) failed all of its registered tests on new machines (Appendix H).
- In both cases, "MIN's reads at the probed rate bound any exact-routing cache" is not established as a bound on the machine. Table 1 labels it "derived".
- The math is derived, but the claim as worded ("bound any … cache") needs "relative to this probe" in the claim column, not only in the threshold column.

### 5. Some registered tests are weaker than Table 1's labels suggest

- Job 115's pooled clause H8 ("mean share ≥ 10%, interval above zero") pools job 114's three machines. Their shares (22, 15 and 22%) were already known when H8 was written. Table 1 lists "pooled share ≥ 10%" as a registered threshold without "loose" or "partly known".
- Job 115's per-machine thresholds (1.07 and 1.20) were set from job 114's 1.10–1.12 and 1.28–1.36, a day apart, on machines of the same class. The test is sequential, not independent.
- Under the paper's own rental-noise rescoring (Appendix D), the second-card trend checks on the RTX 4090s drop from "held" to "held (point)". Table 1 still shows "held".
- The pooled Few-1R result (1.26× on 15 machines) mixes 7 registered machines with machines from jobs 112 and 113, where Few-1R was a comparator. The registered part is labelled correctly in Table 1, but the headline 1.26× in Section 5 is a post hoc pool.

### 6. Small and non-independent samples carry pooled intervals

- The fetches-off column of Table 4 is a t-interval over n = 5, of which two are near-duplicates. The panel is single-round: one process per configuration, so the 2% round-stability gate defined in Section 2 cannot be applied to it.
- The panel's interaction "failure" on 099f, and its 15-machine intervals, rest on one round per machine.

### 7. The attribution of "what is left" is a modelling choice presented next to measurements

- T_GPU is the *minimum* of 14 profiles: 2.94 ms, against a range of 2.94–3.38 ms. That makes the attributed "residual" row in Table 4 depend on the choice.
- With the median profile (about 3.2 ms), the fast-link residual at 11% becomes slightly negative.
- The table separates "Measured" from "Attributed", which is good. But the text ("On fast links two named parts account for it") reads as a measurement.

---

## Claims checked against raw data

All values below are from my own code on raw rows, counters, probes or traces.

| # | Claim | Location | My value | Verdict |
|---|---|---|---|---|
| 1 | R⋆ = 38.32 (C=14), 15.34 (C=32) reads/token | §3, App. N | 38.315, 15.340 (my MIN-with-bypass on `route_aime25_gptoss.npz`) | ✓ |
| 2 | R⋆ on first 20 problems changes −0.4% | App. L | −0.4% at both budgets | ✓ |
| 3 | T_GPU = 2.9 ms, smallest of 14 Nsight profiles | §3 | 2.94 ms (range 2.94–3.38; 069c ×3, 105a/b/e/f) | ✓ |
| 4 | Deployed cache at 31–54% of Eq. (1), 25 consumer machines, median 47, quartiles 44–48 (gpt-oss 11%) | Abstract, §3, Table 1 | n=25; 31.2–54.0%, median 47.0, IQR 44.3–48.0 | ✓ (post hoc scope) |
| 5 | Server machines 12–46% | Table 1 | 12.4, 26.7, 46.1 | ✓ |
| 6 | Eq. (3)/Eq. (1) median 1.00, max 1.56; 40–54% of the ordered bound | §3 | 1.00, 1.56; 40–54% | ✓ |
| 7 | Microbenchmark reaches 86–96% per layer; reps within 0.9% | §3, Table 31 | 86, 89, 96, 96, 95, 93%; max rep diff 0.88% | ✓ |
| 8 | Table 3 host S (all 6 rows): e.g. gpt-oss 11% 57.9 vs 48.0, 1.21 [1.19, 1.22]; Qwen3 43.75% 0.97 [0.96, 0.99] | Table 3 | 1.207 [1.195, 1.219] … 0.974 [0.962, 0.987]; all speeds match | ✓ |
| 9 | Table 3 host B, gpt-oss 3 cells + Qwen3 12.5% (job 081) | Table 3 | 1.294, 1.275, 1.154, 1.032 | ✓ (Qwen3 25/43.75% on B not checked) |
| 10 | Host S vs B: 5 of 6 ratios outside ±0.06 | Table 1 | 5 of 6 (40% cell at −0.061) | ✓ |
| 11 | Table 5, all 5 machines × 10 columns | Table 5 | every cell reproduces, e.g. 285K: 1.08, 0.99, 1.09, 1.00, 1.02, 1.13, +0.3 [0.2, 0.3] | ✓ |
| 12 | Core i9-14900K interaction −0.3 ms [−0.7, 0.1]; rounds +0.6, −1.2 | §4, App. D | −0.31 [−0.7, +0.1]; +0.6, −1.2 | ✓ |
| 13 | Table 4 fetches-off column, 11% and 25% (12 values with intervals) | Table 4 | 19 [13.5, 24.6], 8 [1.8, 13.7], 36 [12.7, 58.5], 18, 43, 3; 25%: 33, 3, 36, 23, 42, −1 | ✓ |
| 14 | Table 4 panel column (15 GPUs), both budgets; 39% on 13 fast-link | Table 4, §4 | 0.5 [−1.8, 2.8], 0.0, 32.6 [21.0, 44.3], 14.9, 52.5, 27.4, 21.4, 3.7; fast-link 38.8 | ✓ |
| 15 | Table 4 new-machine column | Table 4 | 34.7 [21.9, 47.5], 19.0, 46.3; 25%: 19.8, −2.4, 27.3, 24.4, 48.3 | ✓ |
| 16 | Panel interaction > 0 on 9 of 10 main-job machines | Table 1, §4 | 9/10; 099f (285K, ratio 0.29) −0.48 ms [−1.07, −0.14] | ✓ |
| 17 | MIN-2R misses 48.1–50.8 with the table vs 43.5 without; admissions 20.4 vs 8.2; Dep-1R in-step fetches 20.8–32.0 | §4 | 48.1–50.8 / 43.5; 20.4 / 8.2; 20.8–32.0 | ✓ |
| 18 | Interaction share 19–52% → −3 to 21% without fetches; all-zero-table machine 22% / 21% | §4 | 19–52 → −3 to 21; 22 / 21 | ✓ |
| 19 | New machines on deployed path at 41–48% of the bound | §3 | 41, 42, 43, 44, 48% | ✓ |
| 20 | Table 6, RTX 5090 rows (MIN-1R, ahead, none, capture), both budgets | Table 6 | all cells reproduce; best none 1.007–1.024, capture ≤ 6% | ✓ |
| 21 | Table 6, RTX 4090 rows incl. bracketed trend predictions; within 7% at 11%, 5% at 25% | Table 6, §7 | all cells reproduce; max dev 6.6% at 11%; foa/bypass ≤ 0.034 in log | ✓ |
| 22 | Frozen trend constants (job 110 header) fitted on the 15-GPU panel | §7 | my refit: identical to 4 decimals | ✓ |
| 23 | Trend misses 5 of 18 new-5090 cells, up to 18% | §7 | 5 of 18 beyond 0.10 in log; max +17.6% | ✓ |
| 24 | Table 12, all cells; T1 failed 5/8, T2 7/8; early arm admits 11.7–11.8 vs 6.2–7.4, reads 4–7% more | Table 12, App. D | all reproduce; 5/8, 7/8; 11.7–11.8 vs 6.2–7.4; +4.2–7.1% at 11% | ✓ |
| 25 | Table 13, all cells; one-read margin +0.02–0.16; 25%: Few-2R(off) 1.33–1.36 vs Few-1R 1.29–1.43 | Table 13, App. D | 0.98/1.07, 1.12/1.09, 1.26/1.23, 1.23/1.32, 1.40/1.34; +0.166/+0.022; as stated | ✓ |
| 26 | Few-1R geomean 1.26× (1.19–1.33) on 15 stable machines; lost on 2 of 4 unstable (ratios 0.14, 0.21) | §5, App. I | 1.257 [1.186, 1.332], n=15 (2 RTX 4090, 2 server); 0.845 (0.14), 0.974 (0.21) | ✓ (pool is post hoc) |
| 27 | Running example: Eq. (1) 10.0, Eq. (2) 13.0, deployed 20.6, MIN-1R 15.2, read-ahead 13.8 ms | §5 | O4/096a: 10.02, 12.96, 20.64, 15.17, 13.82 | ✓ |
| 28 | Exact 16-token window recovers 0.78–0.93 of the gain at 11% (registered ≥ 0.80; lowest missed) | §8 | 0.78–0.93; 099f = 0.78 | ✓ |
| 29 | Variants read 1.57–1.76·R⋆ | §6 | 1.575–1.755 | ✓ |
| 30 | Speed ratio moves median 0.2% between rounds (max 6.6%), 143 cases on 19 machines | §2, App. D | 143 / 19; median 0.21%; max 6.9% (6.6% in log) | ✓ |
| 31 | Idle probe reruns within 1.0% of the first | §2 | best idle/first 0.998–1.010; spread ≤ 0.5% | ✓ (max single run 1.04%) |
| 32 | Same CPU model differs by up to 29% in deployed time | §2 | 285K panel pair: 18.14 vs 14.06 ms (29%) | ✓ |
| 33 | 107 rentals of jobs 093–115; every commit before its rental by ≥ 2.7 s | App. D | 107; min lead 2.7 s (commit timestamps vs ledger) | ✓ |
| 34 | 160 rentals, 96 offers, $115.3 | App. N | 160, 96, $115.3 | ✓ |
| 35 | In-class published median 13.6% (20 rows) | §3, App. M | 13.6 from Table 33's percentages | ✓ (not re-derived from sources) |
| 36 | D(W50) = 0.66–0.81·C on the AIME routing | §8 | gpt-oss: 0.81·C (11%), 0.68·C (25%) at W50 interpolated from Table 21 | ✓ (gpt-oss only; 9-model 0.65·C not checked) |
| 37 | *New:* MIN-1R vs MIN's set read twice with fetches off, 25%, fast-link Intel | (not in paper) | 0.960–0.972, CIs exclude 1 | ✗ contradicts Rule 4 / §5 heading |
| 38 | *New:* fetches-off interaction across machines | §4 | mean 8.8% of gap, t-interval [−5.3, 22.9] | consistent with text, but not significant over machines |

Not checked: "12 of 84 launch-budgets" (needs the Eq. (4) accounting on every launch); the pooled-slot 4.5–18.6%; KL parity (logit dumps); and the 9-model trace study beyond gpt-oss.

### Registered tests, evidence labels, post hoc labelling

- **Registered.** Tables 5 and 6, 12 and 13, and the prediction log match the job headers in threshold and direction. The headers predate the rentals by commit time. The clause tallies for jobs 114 (33 clauses: 17 held, 14 held (point), 2 failed) and 115 (28 clauses: 12 / 15 / 1) are what I get by applying the stated rule to my recomputed values.
- **Labels I would change:**
  - row 6: "pooled share ≥ 10%" uses three already-known machines, so it should be marked loose or partly known;
  - row 10: should read "held (point under rental noise)";
  - row 1: the claim column should say "relative to the probe".
- **Post hoc labelling** is otherwise careful:
  - the consumer scope, the fast-link subset, the trend on new RTX 5090s, the audit and the horizon rule are all flagged;
  - one exception: the Section 5 heading and Rule 4 generalise a factorial comparison that Section 4 shows to be confounded.
- **Scope.** Claims are mostly scoped ("gpt-oss 11%", "fast links"), but Rules 2–5 and the conclusion are not.

---

## Clarity

**Score: 3/5.** The vocabulary is better than the codes of earlier versions, but the added read path and machine sets have made Sections 3 and 4 denser.

### What works

- **Table 2.** One table names every configuration (MIN-1R, MIN-2R, Dep-1R, Few-1R/2R, LA, Margin, Belady-1R/2R), every machine set and every yardstick. Names of the form "set + number of reads" are a real improvement over codes.
- **Fig. 1.** The schematic of one decode step makes "read twice" concrete at once.
- **The intro's bolded result paragraphs.** Each maps to a section and states the post hoc status of each claim.
- **Table 5.** It is a clean per-machine 2×2 on both read paths. It is the clearest artefact in the paper.
- **The running example in §5.** A 9950X with 10.0 / 13.0 / 20.6 / 15.2 / 13.8 ms anchors the ratios.
- **Section 9's rules.** They are short and actionable (though see Weakness 2).
- **Separate "Measured" and "Attributed" blocks** in Table 4.

### What still makes it hard to read

1. **Table 1 is the hardest thing in the paper, and it comes first.** It has 13 rows and six columns of telegraphic shorthand. The "Registered threshold" cells pack two tests' thresholds into comma lists, for example: "MIN-2R ≥ 1.05 (loose), 1.07× at 11%, ≥ 1.15, 1.20× at 25%; Dep-1R 0.95, 0.98 to 1.10×". The first row cites Eq. (4), which lives in Appendix H, before Eq. (1) has been introduced. A reader cannot parse this table until they have read Sections 2–7.
2. **Concept load.**
   - About 15 configuration names, at least 8 machine sets (panel, panel main job, new machines, fetches-off machines, O3–O5, hosts A/B/S, RTX 4090s, 25 consumer, 15 stable), and 3–4 bounds (Eqs. 1–3, plus Eq. 4 in the appendix).
   - 5 yardsticks with two different baselines: the deployed cache vs admit-every-miss. Fig. 5's caption has to open with "Baseline here: admit every miss, not the deployed cache".
   - Two read paths with five near-synonyms: "deployed read path", "fetch table", "in-step fetches", "fetches off", and "CPU only" / `fetch=0`.
3. **Row and series labels that are false in some columns.**
   - Table 4's first row is "What is cached: MIN's set, read twice", but its footnote says that in two of the three columns it does not measure the set held.
   - Fig. 2's grey "MIN's set, read twice" state is "MIN's schedule with the fetch table, not its set held".
   - Figs. 3 and 4 plot MIN-2R and Belady-2R on the deployed path without saying so.
4. **Sentences carry too many numbers.** A typical example is the §4 paragraph "With the fetches off, MIN's set pays alone": about 15 numbers, mixing ranges over machines, means with t-intervals over machines, and paired-bootstrap intervals over problems, plus post hoc asides in parentheses. The abstract has the same problem: "A per-machine table that also copies misses into slots the cache did not choose hides the first change: with it, MIN's set pays only when read once; without it …". A reader needs Section 4 to decode the abstract.
5. **The evidence trail is long and scattered.**
   - Appendix D is about 12 pages of procedural history: gates, replacements, amendments, two tallies and a noise rescoring.
   - The claim index (Table 9), the job tallies (Table 10), the failures (Table 11), the scorecard (a separate PDF) and the job headers each hold a piece.
   - The main text points into appendix tables (Tables 29, 30, 31) for evidence behind headline claims: the probe sensitivity, the tightness of the bound, and the reachability of the read time.
6. **The decomposition's arithmetic is hard to follow.** On the fetches-off path the shares are of the gap *with* the fetch table, and they add up only together with an extra row ("The in-step fetches themselves†"). A reader has to work this out from a footnote.

---

## Questions for the authors

1. Given the 25% results on the three Intel machines (MIN-1R 3–4% slower than MIN's set read twice with the fetches off, CIs excluding 1), and job 113's 285K at 25%: how do you scope Rule 4 and the §5 heading? Is the crossover ratio budget-dependent? Can you plot MIN-1R ÷ MIN-2R(off) against ratio and budget?
2. Can the factorial of §5 (Fig. 3) be rerun on the CPU-only path, so that Belady-2R and MIN-2R hold their sets? As it stands, the "reading once pays most" conclusion inherits the fetch-table confound.
3. Why does the fetch table flip from all-zero to `0,0,1,2,3` between two 5950Xs whose probes differ by under 1%? Would a deployed cache with a smoothed or hysteretic table change Table 4's deployed-path columns?
4. For job 115's H8, what is the pooled share and interval on the two new machines alone (my estimate is about 18%, n=2), and would you relabel the pooled clause?
5. Can you test the probe against an independent bandwidth measurement? For example, an engine-free replay at the engine's helper count on the 12 launch-budgets where Eq. (4) implies a faster rate. That would show whether the bound or Eq. (4) is the problem.
6. How sensitive are the "attributed" rows of Table 4 to using the median T_GPU (about 3.2 ms) rather than the minimum (2.94 ms)?
7. Do the decomposition's conclusions hold for Qwen3 (BF16, top-8, smaller experts)? Every decomposition result is gpt-oss.

## What would raise my score

- Scope Rule 4, the §5 heading and the abstract's "reading each admission once pays most" to the smallest budget and the links where the data support it. Report the 25% fetches-off result in the main text. *(+1)*
- Rerun the factorial, or at least MIN-2R and Belady-2R, on the CPU-only path. *(+0.5)*
- Report the decomposition for a second model.
- Fix the Table 1 labels: row 6 (partly-known pooled clause), row 10 (point under rental noise), row 1 (claim wording "relative to the probe").
- A substantial rewrite for clarity:
  - a Table 1 that shows only claim, verdict and section, with the thresholds moved to Table 9;
  - one name per read path;
  - row labels that are true in every column;
  - an abstract that does not need the fetch table to be understood.
  *(+1 to clarity; +0.5 overall)*
- A probe validation that settles whether the engine really exceeds it.

## Scores

| Criterion | Score |
|---|---|
| Overall | **5 / 10** (borderline; the rigour and reproducibility argue for acceptance, while the narrow, engine-specific scope, over-general rules and density argue against it in this form) |
| Soundness | **3 / 5** (numbers and registration are impeccable; construct validity and generalisation are the issues) |
| Significance | **3 / 5** |
| Novelty | **3 / 5** (the bound itself is Belady plus bandwidth; the in-engine oracle decomposition, the audit and the distinct-expert horizon rule are new) |
| Clarity | **3 / 5** |
| Confidence | **4 / 5** |

---

## IEEE format

These comments come from skimming `ieee-paper.pdf` (10 pp.) and `ieee-supplement.pdf` (31 pp.).

### Conforming

- IEEEtran front matter: title, anonymous author block, bold abstract, Index Terms (alphabetical).
- Roman section numerals.
- Table captions in small caps above tables; "Fig. n." captions below figures.
- Numbered references; embedded Type 1 fonts.
- A first-page footnote explains the S-numbering.

### Departures from IEEE conventions and cross-reference problems

1. **Page count.** There are 9 full pages of body text, and the references continue onto p. 10. Many IEEE conferences cap papers at 6–8 pages including references, or 8 plus 1–2 for references. This needs checking against the target venue's limit.
2. **Numbering collides across the two supplementary documents.** The IEEE supplement numbers its tables S1–S27. The artifact's scorecard (`supplement.pdf`), which the IEEE supplement explicitly points to, also uses "Table S1…S21" with different contents. For example, TABLE S3 in `ieee-supplement.pdf` is the claim index, while Table S3 in the scorecard is "The grid". The IEEE supplement's TABLE S3 also sends readers to "clause identifiers in the scorecard's tables". The scorecard refers to main-text items by MLSys numbers ("Table 3's protocol"), not the IEEE "Table III".
3. **Inconsistent equation references.** The supplement mostly writes "Eq. (S1)", but three table headers write "Eq. S1" without parentheses. Prose mixes "Figure 3" and "Fig. 5" in the same sentence (Appendix B, rule 12).
4. **References.** The supplement has its own list [S1]–[S62] that duplicates main-paper works under different numbers (e.g. Kalibera & Jones is [21] in the paper and [S2] in the supplement). Within the lists:
   - "et al." truncation is inconsistent (some 10-author lists are spelled out, others truncated);
   - [16] (llama.cpp) has a year range "2023–2026";
   - [19] lacks issue and pages;
   - [25] (BRICS tech report) has no report number.
5. **Floats.** Tables IV and V together take about 60% of p. 6. Table I takes about half of p. 2 and carries a long paragraph of notes. IEEE style favours short table footnotes.
6. **Run-in heading punctuation.** "How tight is the bound?:" in Appendix J should be one or the other.
7. **Appendix references from the main text.** These are by letter only ("Appendix L"). A first-page footnote explains this, but IEEE practice would write "Appendix L of the supplementary material" at least at first use.
