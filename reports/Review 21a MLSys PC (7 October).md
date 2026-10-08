# Review 21a: MLSys 2027 PC, main track (7 October 2026)

**Submission:** "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth"

## Materials read

- `paper/paper.pdf`: all 38 pages. I read the main text both as extracted text and as `paper/paper.tex`. I rendered and inspected pages 1–8 for the figures and tables. I read Appendices A–L in full.
- `paper/supplement.pdf`: its structure, the start of the jobs 073–098 scorecard, and the section list (S2–S4). I did not read every clause row.
- Generated paper inputs, for macro names only: `paper/wsg_job109.tex`, `paper/tab_prereg.tex`, `paper/tab_regtests.tex`.
- Author scripts, read **only for definitions**:
  - `scripts/job109.py`
  - `scripts/decomp_measured.py`
  - `scripts/speed_limit.py` (`host_rates`)
  - `scripts/speed_limit_v2.py` (the `exact` read count)
  - `mosl/cachesim.py` (step semantics of MIN)
  - `scripts/sumlaw_paper.py` (`profiles`)
  - `scripts/wsg_tables.py` (which launches feed Table 3)
  - `scripts/job106.py` (lines 255–270: the slow-link `bypassplan` range)
  - `scripts/value_map.py` (docstring only)
  - `gpu-branch/jobs/ec2/fetch_table.py` (definitions of B_c and B_p)
- Job script headers (the registered predictions): 089, 096, 099, 100, 102, 104, 105, 106, 107, 109, 110, 111, plus the start of 081 and 090.
- Raw data in `/home/claude/gpu-branch/results/`:
  - `ec_*.jsonl` rows: jobs 095, 096a/b, 099a–j, 100a–f, 101a/b, 103–111
  - `bs1.jsonl` rows: jobs 080, 081, 082, 089
  - counters `st_*.json`
  - probes `concur.txt`
  - `g_prof.json` and `prof_C*.json` (Nsight summaries)
  - `readsched_C*.txt` (job 102)
  - `parity_kl.json` (job 090)
  - routing traces `route_aime25_{gptoss,qwen3}.npz` (jobs 084b/084c)
- `gpu/vast_ledger.json`.
- Git history of the gpu branch: `git log --format='%h %cd'`, file lists per commit, and diffs of the job headers.
- All my analysis code is in `/tmp/.../scratchpad/rev21a/`:
  - shared readers: `common.py`
  - job and panel checks: `j109.py`, `panel.py`, `trend.py`, `spear.py`, `plan.py`, `fact.py`, `w16.py`
  - bounds and probe checks: `rstar2.py`, `eq3.py`, `eq4.py`, `rs.py`, `scan.py`, `t3.py`
  - trace replays: `window.py`, `window2.py`, `minplan.py`
  - registration timing: `timing.py`

## Independence statement

I did not open anything under `reports/` other than to write this file. I did not open `prereg/*outcome*.md`, `research_notes/`, `apply/`, `paper/paper_v1_prereview.tex`, or any file named like a review, number check, plan or progress log. A directory listing of `prereg/` showed me file names only. I read no files from `prereg/`.

Every number in the claims table below was recomputed by my own code from raw rows, counters, probes, profiles or traces. For a few items that is not possible, and the table marks them:

- KL: the logit dumps were deleted on the machine, so only the on-machine summary exists.
- The audit median: it is built from published numbers.

I read the authors' scripts only to fix definitions (which probe readings form B_host, step semantics of MIN, which launches feed Table 3). I took no results from them.

One deviation: a `git show 58d5454 -- jobs/111_secondcard@vast.sh` printed that commit's message along with the diff. The message only restated the header amendment shown in the diff. I used `--format=` for every later git command. I modified nothing in either repository and committed nothing.

---

## Summary

The paper studies batch-1 decode of MoE models whose experts mostly live in host DRAM. It uses gpt-oss-120b and Qwen3-30B-A3B, a llama.cpp expert cache, and rented RTX 5090 hosts. It makes four contributions.

1. **A bound.** Eq. (1) gives a per-machine speed bound from Belady's MIN-with-bypass read count R* and the probe's best host read rate. Eq. (2) is a demand bound and Eq. (3) an ordered bound. The authors' cache reaches 31–54% of Eq. (1) at gpt-oss 11% on 25 consumer machines. Published systems, scored the same way, reach a median of 13.6%.
2. **A measured decomposition of the gap.** Oracles built into the engine split the gap. "MIN's set" and "read once" each close about 0% of the gap at gpt-oss 11%. Together they close 33% on a 15-machine panel and 35% on 5 new machines in a registered replication. A read-ahead oracle closes another 15–19%. The remainder is attributed to serial GPU work and the oracle's extra reads.
3. **How to spend foresight.**
   - Read each admission once.
   - Use MIN's fewest-admission schedule.
   - Pick the read path by the link-to-CPU ratio. A trend in that ratio, fitted on RTX 5090s, predicted the oracles' speed on three RTX 4090 machines within 6% in a registered test.
   - Online policies built from the engine's own mechanisms recover at most 6% of the oracle's gain on fast links.
4. **A horizon rule.** Half of foresight's read savings needs a window holding about 0.65 C distinct experts per layer. This is post hoc, over 9 models' traces.

Throughout, predictions were committed to a public branch before each rental, and failures are reported.

## Strengths

1. **Every number I checked reproduces exactly from raw data.** That is 40 checks against rows, counters, probes, profiles and traces (table below). The checks include:
   - R* itself, recomputed by an independent MIN-with-bypass implementation on the raw trace (38.3154 and 15.3404 per token, exact).
   - All 32 entries of Table 4.
   - Every cell of Table 5, including the bracketed RTX 4090 predictions.
   - The frozen RTX 5090 trend coefficients, refit from the 15 panel machines (identical to four decimals).
   - The Section 6 window replays (58.64 / 55.07 / 48.58 / 39.72 reads per token).

   This is a rare level of computational reproducibility.
2. **The registration is real.**
   - Every rental of jobs 093–111 in the ledger started at least 2.7 s after its scripts' first commit.
   - The header edits made after launch are host substitutions or amendments with predictions unchanged, as the paper says. I diffed jobs 100, 102, 105, 106, 107 and 111.
   - Job 100's added prediction 9 is disclosed.
   - The job 109 replication of the 2×2 (11 predictions, all held, interaction positive on 5/5) was registered with a population rule fixed before launch. It is a genuine out-of-sample confirmation.
3. **Honest negative results.**
   - The closed-form account (Eq. 4) failed three of four registered tests, and the paper demotes it to "exploratory, not confirmed".
   - The trend's out-of-sample miss on new fast-link RTX 5090s (up to 18%) is in the main text.
   - The prediction log reports 309 failed clauses out of 1,614.
4. **A useful framing for builders.** It separates *what* is cached, *how often* each admission is read, and *when* it is read, and measures each part in a real engine rather than a simulator. The interaction finding (each lever alone ≈0 at 11% on fast links, about a third together) is non-obvious and is backed by counters: on fast-link machines MIN's set admits 4–5× more than the deployed policy (15.8–17.5 vs 3.5–3.6 per token at 11%), so a second read per admission erases the benefit.
5. **The machine is the unit of analysis.** Machines differ far more than problems (same-CPU panel machines differ by up to 29%). Relaunches agree within 3.2% (RTX 5090) and 5.5% (RTX 4090). The statistics follow this.

## Weaknesses (most important first)

1. **The second-card test cannot test a trend, and it is barely a test of generalisation.**
   - It has three valid machines, two of the same CPU model (Core Ultra 9 285K), at link-to-CPU ratios of 0.37, 0.39 and 0.40. A single x-value can check the trend's *level* there, not its slope.
   - The three are relaunches of job 110's machines. Job 110's first-round results for them were committed in the same commit as the job 111 script (443bcba). On those first rounds the trend was already within 6.2% (the generated macro `cxFirstDevLowMax` is 6, unused in the text). The authors therefore knew the outcome direction before relaunching.
   - A trivial predictor does nearly as well for MIN, 1 read. Using the mean of the two slow-link panel machines gives a maximum error of 4.5% at 11% and 2.6% at 25%, against the trend's 2.7% and 4.7%. "No change" gives 3.7% and 6.0%. The trend clearly beats a panel-median predictor for the read-ahead oracle (5.9% vs 18%).
   - Relaunch variability on these same machines is up to 5.5% (110c vs 111g: deployed time 19.35 vs 20.41 ms; MIN, 1 read 0.953 vs 1.005×). That is the same size as the claimed precision.
   - Table 1's "predicts the oracles' speed, on slow links on two cards" is also stretched. On the RTX 5090 the slow-link evidence is in-sample: only 099d and 099f are below 0.5 in the fit.

   The claim should be "the trend's level at ratio ≈0.39 transferred to a second card on three machines".
2. **"What the oracles leave" is accounted for only on fast links, and the text does not say so.** Section 4 says "Two named parts account for it". Recomputed residuals (left − T_GPU − extra reads, as a share of the gap):

   | Machines | 11% | 25% |
   |---|---|---|
   | Fast-link panel, mean | −2% | −9% |
   | Slow-link panel 099d / 099f | +35% / +47% | +18% / +25% |
   | All three RTX 4090s (not reported) | +31 to +36% | +22 to +25% |

   Table 4's panel mean (4 [−4, 13]) hides this bimodality. On slow links the named parts leave a third of what is left unexplained, presumably the ordered-bound effect (Eq. 3). That should be shown, not averaged away.
3. **The 2×2 does not hold the cached set fixed, so "what is cached" and "how it is read" are confounded with *when* copies land.**
   - In the "MIN's set, still read twice" arm the background copies land late. That arm misses 22–28% more than MIN read once on fast-link machines (per-token misses 50.8 vs 39.7 on 099b, 48.6 vs 39.7 on 109a at 11%). Its cache contents are therefore not MIN's set.
   - The paper mentions late landing in one clause, yet the headline "pay only together" reads as a statement about the set and the read count.
   - Table 1's row is not scoped. At 25% MIN's set alone closes 21%. On slow links "together" closes 8% (099d) or −23% (099f) at 11%. On the RTX 4090s it closes −6 to +1% at 11% and 2–8% at 25%.
4. **The bound is probe-relative and can be exceeded, and the 25% "host-bound" label depends on the datasheet GPU term.**
   - On the machine at the top of the 31–54% range (100f, Ryzen 9 9950X), the engine's counted reads imply 1.15–1.20× the probe's best rate, depending on G. The paper says this range top is "probably too high".
   - On most machines the probe ran while the model was downloading.
   - With the measured batch-1 GPU rate (Tables 3 and 26), gpt-oss 25% on host B is GPU-bound: 279 vs 430 tok/s. More generally the measured-GPU bound binds at 25% whenever B_host > ~57 GB/s. That is 5 of the 15 panel machines and all 4 new 25% machines. On those machines the 25% gap would shrink by 4–19% (099f the most) and its shares would rescale.
   - Section 3's "the GPU term … binds only at the largest budgets" is therefore true only under the datasheet rate.
5. **The "(registered)" tags sometimes certify less than the sentence they head.**
   - "The read time is nearly reachable (registered)": job 102 registered token mode ≥ 0.80 and per-layer mode ≥ 0.50. "Nearly reachable" (86–96%) is a post hoc reading of a weak registered threshold.
   - The fast/slow line at ratio 0.5 was drawn after jobs 093–101 (job 109 header; `decomp_measured.py` docstring). Section 2 states it as a definition without saying so.
   - "Slow-link RTX 5090s … 1.11–1.22×" is computed with a ratio < 1/3 cut (`job106.py`), not the paper's < 0.5. Under the paper's own definition the range is 1.08–1.22×.
   - Table 1's 2×2 row says "pre-specified sign: held on 9 of 10". Job 099 registered "on every host", so as registered that clause failed (on 099f, the slowest link).
6. **Scope and generalisation of the builder rules.**
   - The evidence covers one engine's deployed policy and gpt-oss alone for the decomposition, the 2×2, the online policies and the second card.
   - The workload is AIME prompts (also used during development), 256 teacher-forced tokens, batch 1, mostly one GPU model on rented hosts.
   - Rule 3, "Do not expect an online policy to get there", generalises from two engine mechanisms (margin, layer-ahead copy) plus offline forecasters trained on small out-of-domain text, which the paper itself says bounds cross-text, not in-domain, performance.
   - The abstract's "far beyond the next token" is not supported in general. W50 spans 0.6–34 tokens across models, and at gpt-oss 11% it is about 4 tokens.
7. **Small-n intervals are overconfident.** Table 4's "New (4)" and "New (5)" columns use percentile bootstrap over 4–5 machines. Examples: together at 11% is 35 [28, 44], and the t-interval is about [22, 47]. Residual at 25% is −5 [−7, −4]. Appendix I acknowledges this for one statistic, but the table does not.
8. **Novelty is moderate.** MIN-based bounds and oracle gaps for expert caching appear in cited concurrent work (WiSP, Budgeting Bytes, Zhang 2026b's 44–46% causal-vs-MIN gap, Liang et al.). The new part is the in-engine time-domain decomposition and the registration discipline, not the bound itself. The system's own lead over FreeToken is modest at some cells (1.03–1.05× on Qwen3; 0.97× at Qwen3 43.75% on host S).

## Claims checked against raw data

"✓" means my value matches the paper's to its printed precision. "✓ (scope)" means the number is right but the sentence generalises beyond it. "△" marks a minor discrepancy. "✗" means the claim as worded is not supported.

| # | Claim (paper) | Location | My value (from raw) | Verdict |
|---|---|---|---|---|
| 1 | R* = 38.315 (C=14), 15.340 (C=32) reads/token | Eq. 1 inputs; job 109 header | 38.3154 and 15.3404 with step-atomic MIN-with-bypass on `route_aime25_gptoss.npz`; an event-ordered variant gives +0.8% | ✓ (bound depends mildly on within-step semantics) |
| 2 | R* on the first 20 problems changes by −0.4% | §9 | −0.38% at both budgets | ✓ |
| 3 | Table 4 panel, 11%: set alone 0 [−2,2], once alone 0 [−2,1], both 33 [22,42], ahead 15 [11,18], left 52 [45,62], T_GPU 27, extra 21, residual 4 [−4,13] | Table 4 | 0.5 [−1.7,2.4], 0.0 [−1.8,1.5], 32.6 [21.6,41.9], 14.9 [11.1,18.5], 52.5 [44.5,62.0], 27.4, 21.4, 3.7 [−4.1,12.8] | ✓ |
| 4 | Table 4 panel, 25%: 21, −3, 31, 21, 48, 34, 19, −5 | Table 4 | 20.6, −3.2, 31.3, 20.6, 48.1, 34.0, 19.1, −5.0 | ✓ |
| 5 | Table 4 New(5) 11%: 3, 0, 35, 19, 46, 28, 18, 1; New(4) 25%: 20, −2, 27, 24, 48, 38, 16, −5 | Table 4 | 3.4, −0.4, 34.7, 19.0, 46.2, 28.2, 17.5, 0.5; 19.8, −2.4, 27.3, 24.4, 48.4, 37.9, 15.6, −5.2 | ✓ |
| 6 | Interaction sign registered: held on 9 of 10 panel (099) and 5 of 5 new | §4, Table 1 | 9/10 (fails on 099f, ratio 0.29); 5/5 at 11% and 4/4 at 25% | ✓ (registered as "every host": strictly failed once) |
| 7 | "Two named parts account for" what is left | §4 | Fast-link panel residual −2% (11%) / −9% (25%); slow panel +35%/+47% (11%); RTX 4090 +31 to +36% (11%) | ✗ (holds only on fast links) |
| 8 | Table 5, RTX 5090 rows: 1 read 1.15–1.37, ahead 1.34–1.50, online 1.01–1.02, capture 2–6% at 11%; and the 25% columns | Table 5 | e.g. 285K 1.215/1.460/1.021/4.5%; 7945HX 1.241/1.416/1.024/5.7%; 25% 7945HX best 0.995, capture −0.8% | ✓ (all 36 cells) |
| 9 | Table 5, RTX 4090 rows with bracketed predictions, e.g. 14900KF 0.96 [0.99], 1.13 [1.11] | Table 5 | 0.963 [0.989], 1.128 [1.111]; 285K 1.007 [1.010], 1.198 [1.133]; 25%: 1.019 [1.069] … | ✓ |
| 10 | Trend predicted RTX 4090 within 6% (11%) and 5% (25%) | Abstract, §5 | Max \|dev\| 5.9% (11%), 4.7% (25%); registered log bands 0.10 / 0.06 all pass | ✓ (n=3, one ratio point; see W1) |
| 11 | Same trend misses 5 of 18 new RTX 5090 cells, by up to 18% | §5 | 5 cells with \|ln dev\| > 0.10; max 17.6% (109e, ahead at 11%) | ✓ |
| 12 | Frozen trend coefficients fitted on the 15 panel machines | job 110 header | Refit: identical to 4 decimals for all 8 lines | ✓ |
| 13 | Online best at 11%: 1.007–1.024×; capture at most 6% | §5 | dk 1.007 (13900KF) to R2 1.024 (7945HX); max capture 5.7% | ✓ |
| 14 | Online policies read 1.66–1.68 R*, "as many as the deployed cache" | §5 | R1/R2 1.665–1.678 R*; deployed 1.645–1.660 R* | ✓ |
| 15 | RTX 4090: layer-ahead policies 0.64–0.72×; admitting less 1.033–1.041×, capture up to 26%; MIN, 1 read 0.96–1.01× | §5 | 0.644–0.724; 1.033–1.041; 25.7%; 0.963–1.007 | ✓ |
| 16 | RTX 4090 loss 3% higher | §5, App. D | 0.1963–0.1965 vs 0.190 (+3.3%) | ✓ |
| 17 | Relaunches within 3.2% (RTX 5090), 5.5% (RTX 4090) | §2 | TR 9960X 9.12 vs 9.41 ms (3.2%); 110c vs 111g 19.35 vs 20.41 ms (5.5%) | ✓ |
| 18 | Same-CPU panel machines differ by up to 29% | §2 | 285K: 18.14 vs 14.06 ms (29%) | ✓ |
| 19 | 31–54% of Eq. 1 at gpt-oss 11% on 25 consumer machines | Abstract, §3 | 25 GPUs, first launch each: 0.312 (TR 9960X) to 0.540 (100f) | ✓ |
| 20 | 5 new machines at 41–48% | §3 | 0.415–0.480 | ✓ |
| 21 | Eq. 3 / Eq. 1: median 1.00, up to 1.56; 40–54% of the larger | §3 | median 1.000, max 1.557 (TR 9960X); 0.405–0.540 | ✓ |
| 22 | Top of the range "probably too high": implied rate above the probe | §3 | 100f: implied 59.6–62.3 GB/s vs probe best 52.0 (1.15–1.20×) | ✓ (and the bound is beaten there) |
| 23 | Table 3 host B and S speeds and ratios (24 speeds, 12 ratios), e.g. 1.294 [1.278, 1.312], 0.974 | Table 3 | 69.88/53.99 = 1.294; S Qwen3 43.75% 95.78/98.32 = 0.974; all rows match. Host B rows pool jobs 080, 081, 082 (same GPU UUID, different rentals) | ✓ (caption's "within one launch" is not true of the ×llama.cpp column) |
| 24 | Bound 172/430 tok/s (B) and 140/351 (S); shares 41/25/41/27% | Table 3 | 172.3/430.4 and 140.4/350.7; 0.406/0.254/0.412/0.268 | ✓ |
| 25 | Ratio of mean rates vs ratio of total times agree within 0.003 | App. B | max \|diff\| 0.0030 over 10 cells | ✓ |
| 26 | Microbenchmark reaches 86–96% of the bound's read time; reps within 0.9% | §3, Table 27 | per-layer 86, 89, 96, 96, 95, 93%; reps 0.88% max | ✓ (registered threshold was ≥ 0.50) |
| 27 | Running example: Eq. 1 10.0, Eq. 2 13.0, deployed 20.6, MIN 1 read 15.2, read-ahead 13.8 ms | §5 | O4 (096a): 10.02, 12.96, 20.64, 15.17, 13.81 | ✓ |
| 28 | T_GPU = 2.9 ms, the smallest profile; G = 4.1–4.7 ms | §3, App. H | min non-expert 2.940 (105e), median 3.16; G 4.07–4.69 over 14 profiles | ✓ (RTX 4090 G is 5.2–5.6 ms, unreported) |
| 29 | Spearman 0.89 [0.63, 0.97] over 19 machines, MIN 1 read vs ratio | §5 | 0.886 [0.63, 0.98], n = 19 (one launch per GPU, jobs 093–104) | ✓ |
| 30 | Fewest-admission: geomean 1.21× on 9 stable machines [1.13–1.30]; lost on the two slowest unstable | §5, App. I | 1.215 [1.128, 1.304]; EPYC 7302 0.845 and EPYC 7663 0.974 | ✓ |
| 31 | Fewest-admission copies 0.60–0.73 of greedy; 2.5–5.3% more misses | §5, §9 | fetches 0.597 / 0.726; misses +2.47 to +5.26% | ✓ |
| 32 | Slow-link RTX 5090s: fewest-admission set via CPU 1.11–1.22× | §5 | 1.112–1.217 only with ratio < 1/3; with the paper's < 0.5: 1.081–1.217 | △ (threshold mismatch) |
| 33 | MIN, 1 read beats deployed at every budget on O3–O5; the "usual ways" gain at most 15% or lose at the smallest budgets | §5 | fetch 1.16–1.51 at all 18 cells; max of bypass/nb2/hitopt(p) at 11%/12.5% = 1.148 | ✓ |
| 34 | In-engine 16-token window recovers 0.78–0.93 at 11%; lowest machine missed 0.80 | §6 | 0.781 (099f) to 0.935; 099d 0.80 | ✓ |
| 35 | Next-token routing recovers at most 0.31; half the gain at gpt-oss 11% needs 4 tokens | §6, Table 17 | Independent replay: aa 58.64, W1 55.07, W4 48.58, W16 39.72; engine-semantics MIN 39.26 → 0.18 / 0.52 / 0.98; Qwen3 12.5% aa 150.03, W1 134.72, MIN 100.35 → 0.31 | ✓ (this "MIN" is 39.3, not R* = 38.3) |
| 36 | Distinct experts at the half-gain window: 0.66–0.81 C on the AIME routing | §6 | gpt-oss C14: D(3)/C = 0.68, D(4)/C = 0.85 around W50 ≈ 3.5–4 | ✓ (plausible); "far beyond the next token" ✗ in general |
| 37 | Admitting less cuts deployed reads 4–6% (11%), 6–10% (25%) | §5, App. I | 4.3–5.8% and 6.3–10.5% | ✓ |
| 38 | Read-ahead oracle reads 1.22–1.31 R* at 11% | Table 2 | 1.218–1.312 (panel), 1.218–1.236 (job 109) | ✓ |
| 39 | Eq. 4 rows of Table 24 (e.g. EPYC 7543 14.9 vs 22.7 ms, implied 0.58; 5950X 27.2 vs 29.1, 0.92) | App. H | 14.94 / 22.73 / 0.58; 27.24 / 29.12 / 0.92; all rows within rounding | ✓ |
| 40 | Prediction timing: every commit at least 2.7 s before its rental; post-launch header edits were substitutions with predictions unchanged; job 100 gained prediction 9 after launch | App. D | ledger vs `git log`: min 2.7 s (105f); diffs of 100/102/105/106/107/111 confirm; job 100's Ph edit committed 9 s before its rental | ✓ |
| 41 | Job 111 relaunched 3 of 4; host e left out by oversight | App. D | `job110_uuids.txt` lacks 110e's UUID although the 111 header lists e's offer; 110e's results were committed after the 111 script | ✓ |
| 42 | Table 9 tallies: 565 clauses for 073–098 (218/193/142/12), 1,614 overall | App. D | Rows sum to 218/193/142/12 and 550/671/309/84 | ✓ |
| 43 | 124 rentals, 74 offers, US$87.2 (jobs 058–108) | App. L | 124, 74, $87.22 | ✓ |
| 44 | KL 0.0019 (gpt-oss), top-1 98.5%, ΔNLL +0.16% | §9, App. J | on-machine `parity_kl.json`: 0.00186, 0.985, +0.160% (dumps deleted, not recomputable) | ✓ (summary-level) |
| 45 | Published in-class median 13.6% (20 rows) | §3, App. K | Median of Table 29's 20 in-class rows = (12.8 + 14.4)/2 = 13.6 | ✓ (from published values, not raw) |
| 46 | "The GPU term … binds only at the largest budgets" | §3 | With the measured GPU rate (Tables 3, 26) gpt-oss 25% on host B is 279 (GPU) vs 430 (host); binds wherever B_host > ~57 GB/s | ✗ (true only for the datasheet rate) |

**Evidence labels in Table 1.**
- "Derived": correct for the bound's logic, but it is a bound relative to the probe (items 22, 46).
- "Pre-specified bands" for read-time reachability: correct, though the registered threshold (≥ 0.50 per layer) is far weaker than "nearly reachable".
- The 2×2 row: correct counts, but the registered clause said "every host".
- "Pre-specified (≤ 35%)" for the online policies: correct.
- The trend row: correctly says "fitted post hoc" on the RTX 5090s, but "predicts … on slow links on two cards" counts an in-sample fit as one of the two cards.
- Fewest-admission: correct (7 registered, 9 stable).
- Horizon rule and closed-form account: correctly post hoc.
- "Failed on host S" for the system lead: misleading. What failed was a ±0.06 band on the ratios. The lead itself held at 5 of 6 cells on S.

**Post hoc results.** These are well labelled for Eq. (4), the horizon rule and the trend fit. They are unlabelled for the fast/slow threshold at 0.5 and for the slow-link selection (< 1/3) behind "1.11–1.22×".

---

## Clarity

The paper has improved. A specialist can now follow the main line on one careful read. Readability still lags the quality of the evidence.

### What works

- **Table 1 (claims × evidence × machines × section)** and the bold claim paragraphs of the introduction give a clear map. Each claim says what kind of evidence backs it. The approach is good and worth keeping.
- **The "Yardsticks" paragraph** names each baseline up front. Figure 5's caption opens with "Baseline here: admit every miss, not the deployed cache". This prevents the commonest misreading.
- **The running example** (10.0 / 13.0 / 20.6 / 15.2 / 13.8 ms) anchors §5. I verified it to the decimal.
- **Table 4's split into "Measured" and "Attributed"** rows, with a caption stating that the attributed rows are not measured, is exactly right.
- **Figure 1** (one decode step, two reads of an admitted expert) makes the "read twice" mechanism concrete in one glance.
- **Section 7's five builder rules** are concrete and mostly scoped (rule 2 says "where the link reads at least half as fast as the CPU").
- **Limitations** is specific and quantitative: deviations of the second-card test, gate failures, probe interference, R* on 20 vs 30 problems.

### What still makes it hard to read

1. **Name collisions remain the biggest barrier.**
   - *"Online policy"* means the deployed cache throughout Appendix E (Tables 10–13: "the online policy's speed"). In §5 and Table 5 it means the four no-foresight policies built *on top of* the deployed cache.
   - *"MIN"* denotes two different counts:
     - R* = 38.3 per token: step-atomic, seeing across problems (§3–4).
     - The engine oracle's 39.3: within-problem, served experts not evictable (§6 and Table 17's "MIN with one read").

     I needed to reimplement both to see that they differ.
   - *Host/machine* and *limit/bound* still alternate between the main text, the appendix tables and Figure 3's axis label.
   - The same configuration carries three or four names:
     - "MIN, 1 read" = fetch = "MIN's set read once" = "Both".
     - "Read-ahead oracle" = both3p = "MIN prefetched" = "copied ahead".

     Table 7 helps but sits on page 15.
2. **There are too many baselines.** The paper uses five yardsticks: share of the bound's speed, share of the gap, speed vs the deployed cache, capture as a share of the read-ahead oracle's gain, and a window's share of MIN's gain over admit-every-miss. Section 5 alone uses three, and "capture" changes its denominator between machines (an "oracle gain that is smaller there"). A reader cannot easily compare "35% of the gap", "1.22×", "6% capture" and "0.31 of the gain".
3. **The abstract and Table 1 compress past the point of legibility.**
   - Most abstract sentences carry three or four qualifiers ("at the smallest gpt-oss budget on machines with consumer processors"; "on fast-link machines rented for a test we registered before they ran").
   - The abstract uses *bypass*, *admitted*, *read once instead of twice*, *the gap* and *C* before any definition.
   - In Table 1, "held on host B, failed on host S" reads as if the lead failed. "15 + 3" machines and "on slow links on two cards" need the reader to reconstruct which part is in-sample.
4. **Section 5 is overloaded.** It bundles read-once, fewest-admission, the second-card registered test, and the online-policy test. `\label{sec:secondcard}` and `\label{sec:online}` are paragraph anchors that the text cites as "Section 5". The "(registered)" tags on paragraph heads do not say what was registered (threshold, sign or band), so the reader cannot tell a strong confirmation (job 109) from a weak one (job 102's ≥ 0.50).
5. **The appendices are sprawling.**
   - Appendix D is several pages of dense prose plus Table 9, with two tallies (565 clauses for 073–098, 1,614 overall) and different "held" semantics for hand and script scoring.
   - Appendix E's Tables 10–13 use the older vocabulary (online / fetch / both / paced / hit-opt.) and "limit".
   - In Figure 3 the O3/O4/O5 markers and their 95% intervals are too small to distinguish at print size. Several rows have only panel ticks.
   - In Figure 4 crosses cannot be "hollow", so the RTX 4090 layer-ahead points are identifiable only by position.
6. **Small slips.**
   - "…after one round on these machines. two of its first launches failed" (lowercase sentence start, App. D).
   - The template and PDF metadata say "mlsys 2025".
   - The < 1/3 vs < 0.5 slow-link cut (item 32).
   - Table 3's caption claims intervals "within one launch on one machine", while host B's rows mix three rentals (jobs 080/081/082).

**Clarity score: 3/5.** It is better than the earlier 2s. It is not yet a 4, because a careful expert still has to reverse-engineer which "online policy" and which "MIN" a sentence means and which baseline a percentage is against.

---

## Questions for the authors

1. Please report Table 4's decomposition for the three RTX 4090 machines and separately for the two slow-link panel machines. Do you agree the named parts leave +31 to +47% of the gap unexplained there, and that Eq. (3) rather than Eq. (1) is the relevant reference on slow links?
2. In the 2×2, the read-twice MIN arm misses 22–28% more than MIN read once. How much of "MIN's set alone ≈ 0" is late landing rather than the second read? Could you run MIN's set read twice with copies issued one step ahead (no foresight beyond the step), to separate the two?
3. When job 111 was committed, job 110's first rounds on the same machines were already in the repository (trend within 6.2%). Would you have relaunched if those rounds had missed the band? Please state this in §5/§9 and report job 110's first-round deviations.
4. Why was the RTX 4090 test run only on hosts with ratio 0.37–0.40? Was any RTX 4090 with a ratio ≥ 0.5 available? Without a second x-value the slope of the trend is untested on that card.
5. The 0.5 fast/slow threshold: was it chosen after jobs 093–101, as the job 109 header says? If so, please say so in §2.
6. Table 17's "MIN with one read" (39.3) differs from R* (38.3). Please name them differently and give the semantics: within-problem, served experts kept for the step.
7. With the measured batch-1 GPU rate, how do the 25% column of Table 4 and the "host-bound" classification change?
8. For rule 3 (no online policy gets there): have you tried any multi-token online predictor *inside the engine*, beyond the layer-ahead copy?

## What would raise my score

- **Scope the decomposition claim to fast links.** Report the slow-link and RTX 4090 decompositions, with Eq. (3) as the slow-link reference. Scope Table 1's 2×2 row to "gpt-oss 11%, fast links" and note that at 25% the set alone closes 21%.
- **Restate the second-card result as what it is.** Call it a level check at one ratio on three machines, disclose that job 110's first rounds were known before the relaunch, and add a comparison against simple baselines (no-change, slow-panel mean). Better still, add RTX 4090 hosts at a second ratio.
- **Fix the 2×2's confound.** Add a counter-level table (misses, admissions, landing lag) per arm showing whether MIN's set is realised in each. Or soften "what is cached and how it is read" to include *when the copy lands*.
- **Make each "(registered)" tag precise.** Say what was registered (threshold, sign or band), and state the post hoc choices (the 0.5 line; the 1/3 cut).
- **Fix the clarity issues above.** Use one name per configuration in the main text and appendix tables, rename "online policy" in Appendix E to "deployed cache", give the two MIN counts different symbols, cut the abstract to one qualifier per sentence, and give Table 5/Table 4's New columns t-intervals.

With weaknesses 1–3 addressed in text and analysis (no new GPU runs needed except optionally for 1 and 3), I would move to 7.

## Scores

| | Score |
|---|---|
| Overall (1–10; 6 = weak accept, 8 = strong accept) | **6** |
| Soundness (1–5) | **4** |
| Significance (1–5) | **3** |
| Novelty (1–5) | **3** |
| Clarity (1–5) | **3** |
| Confidence (1–5) | **4** |

Rationale:
- **Soundness 4:** the empirical record is exceptionally reproducible and honestly registered. The interpretation overreaches in a few scoped places (W1–W5), all fixable by rewording or one more table.
- **Significance 3:** practical guidance for batch-1 MoE offload on consumer GPUs, but for one engine and essentially one model and card class.
- **Novelty 3:** the in-engine oracle decomposition and the registration discipline are new. The MIN-based bound and the oracle gap are close to concurrent work.
