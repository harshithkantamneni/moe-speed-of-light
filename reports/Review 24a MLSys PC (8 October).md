# Review 24a: MLSys 2027 (main track), PC review

**Submission:** "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth"
**Reviewer date:** 8 October 2026

## Materials read

- `paper/paper.pdf`, all 44 pages: the 10-page main text in full, Appendices A–N (pp. 15–44) read in full or skimmed table by table. I rendered pages 1–10, 22 and the Figure 3 region to check layout.
- `paper/supplement.pdf`: I checked its structure (clause scorecards, S1–S4) and read pages 18–23 (S1–S4 prose) in detail.
- `paper/ieee-paper.pdf` (10 pp.) and `paper/ieee-supplement.pdf` (29 pp.): skimmed, rendered and checked for IEEE conventions (see the "IEEE format" section).
- LaTeX source, read only to see what the PDF renders: `main_body.tex` (the Table 1 block), `tab_headline.tex`, `tab_value.tex`, `tab_prereg.tex` and the head of `table_audit_full.tex`.
- Raw results on the second checkout `/home/claude/gpu-branch/results/`: `ec_*.jsonl` rows, `st_*.json` counters, `concur.txt`, `concur2.txt`, `fetch_table_law_*.json`, `g_prof.json`, `prof_C*.json`, `readsched_C*.txt`, `bs1.jsonl`, `manifest.json`, `v0.txt`, `validity.txt` and `cpu.txt` for jobs 081, 084b/084c (routing traces), 089, 093–112, plus the `069c_profile_*` directories.
- Job script headers on the gpu checkout: 081, 089, 099, 102, 109, 110, 111 and 112 in full. For 093–112 I compared each per-machine script's commit time with its machine start time. I used `git log --format='%h %cd'` and one `git diff` of the 111 header's file content. I read no commit messages.
- `gpu/vast_ledger.json`, for rental creation times and cost.
- Authors' scripts, read **only for definitions**: `scripts/job109.py` (header: definitions of S, R⋆, T_GPU, ratios, capture), `scripts/decomp_measured.py` (the definitions of the parts of the gap), `scripts/speed_limit.py` (`host_rates` and the GPU-term constants), `scripts/sumlaw_paper.py` (`profiles()`: how the non-expert time is taken from a profile).
- From `prereg/` I read two files. From `prereg/reanalysis_hosts.json` I took only the list of launch directories that make up the 15-machine panel; every value was recomputed from raw rows. From `prereg/speed_limit_v2.json` I glanced at the header (the B_host definition). R⋆ was recomputed independently from the traces.

All my code is in `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/rev24a/` (`rstar.py`, `rstar_eng2.py`, `load.py`, `panel.py`, `j109.py`, `j112.py`, `share.py`, `eq3.py`, `fewplan.py`, `headline.py`, `regtime*.py`). I ran it from that folder.

## Independence statement

I did not open anything under `reports/` except to write this file. I did not open `prereg/*outcome*.md`, `research_notes/`, `apply/`, `paper/paper_v1_prereview.tex`, or any review, number-check, plan or progress-log file, and I read no commit messages. I took the authors' scripts only for definitions (listed above) and wrote my own code to compute every number in the claims table from the raw rows, counters, probes, traces and profiles. I modified nothing in either repository apart from writing this file, and I made no commits.

---

## Summary

The paper studies batch-1 decode of MoE models (gpt-oss-120b in MXFP4 and Qwen3-30B-A3B in BF16) whose experts mostly live in host DRAM on single consumer GPUs (RTX 5090, plus RTX 4090 for one test). It makes five contributions:

1. **A bound.** Eq. (1) is the time per token implied by the host reads of Belady's MIN with bypass at the machine's highest probed read rate. Two tighter, conditional variants follow: Eq. (2), which charges the GPU's non-expert time in series ("demand"), and Eq. (3), in which only link copies can go ahead ("ordered").
2. **A system.** A 3,500-line llama.cpp expert cache that beats FreeToken at 11 of 12 configurations and stock llama.cpp at all of them. On consumer machines at gpt-oss 11% it reaches 31–54% of Eq. (1).
3. **A decomposition.** Oracles inside the engine split the gap to Eq. (1) along three axes: what is cached (MIN's set), how it is read (once in the step versus CPU-served then copied), and when (read ahead).
   - At gpt-oss 11%, either single change closes about 0% of the gap. Both together (MIN-1R) close 33% (15 panel machines) and 35% (5 machines rented later for a registered test). The read-ahead oracle closes 15–19% more.
   - The remaining ~half is attributed, not measured, to T_GPU and to the oracle's extra reads.
4. **Policy results.**
   - Four online variants built from the engine's own mechanisms recover at most 6% of the read-ahead oracle's gain.
   - MIN's fewest-admission schedule read once gains 1.24× on 13 stable machines.
   - The link-to-CPU read-rate ratio sets which read path pays. A trend fitted on RTX 5090s predicted RTX 4090 machines within 7%.
5. **A horizon rule (post hoc).** Half of foresight's read saving needs the routing of ≈0.65 C distinct experts per layer.

The study has 145 rentals, and every job's predictions were committed to a git branch before its machines started. Appendix D scores 1,672 clauses: 588 held with an interval, 679 held on the point estimate only, 321 failed, 84 were untested or void.

## Strengths

1. **The numbers check out against the raw data, exactly.** I recomputed 40 quantitative claims (table below) from the per-problem rows, the counters, the probes, the traces and the profiles, and every one reproduces within rounding:
   - Table 4 (all 32 entries);
   - Tables 5 and 6;
   - the Table 3 rows I checked;
   - R⋆ from the traces;
   - the frozen RTX 5090 trend coefficients;
   - the registration timestamps.

   Only two departures are interpretive (T3 and the Section 8 window range), plus one small interval difference. This is rare and valuable.
2. **The registration is real and well documented.**
   - Every per-machine script of jobs 093–112 was committed before that machine started (minimum 29 s by commit time), and every one of the 92 rentals came at least 2.7 s after a commit.
   - Amendments made after launch (jobs 100, 105, 107, 111) are disclosed in Appendix D, and the one I inspected (111) left the predictions unchanged.
   - Failures are reported prominently: Table 1, Table 11, Appendix D.
   - Thresholds the earlier data made easy to meet are labelled "loose".
3. **Method.**
   - Building oracles *into a real engine* and replaying the same teacher-forced routing makes configuration comparisons paired and low-noise. Rounds move a median 0.17% (max 1.74%, n=98 comparisons), against much larger effects.
   - Treating the machine as the unit, with t-intervals over machines, is the right call. Machines carry most of the variance (two Core Ultra 9 285K panel machines differ by 29% in deployed time).
4. **The bound framing is useful to the community.** "Report the distance to the machine's bound" (Rule 1) is a good norm for an area where every paper reports speed-ups over its own baseline. The microbenchmark showing that 86–96% of the bound's read time is reachable makes the bound credible as a target.
5. **The negative results are useful.**
   - Online policies built from the engine's mechanisms read 1.57–1.76 R⋆, about as much as the deployed cache's 1.65 R⋆. So without foresight there is little to win in this engine.
   - Next-token routing buys at most 0.31 of MIN's read saving.
6. **The artifact is unusually complete.** It includes job scripts, raw rows, probes, profiles, the rental ledger ($105.8 over 145 rentals, which I reproduce) and a clause-level scorecard.

## Weaknesses (most important first)

1. **The title's question is answered only half by measurement.** At gpt-oss 11% the measured parts (MIN-1R plus the read-ahead oracle) close 48% (panel) and 54% (new machines). The rest is split by *attribution*: a fixed T_GPU = 2.94 ms plus the oracle's extra reads charged at B_host.
   - **The T_GPU used is the smallest of 14 profiles.** It comes from 7 other machines, three of them EPYC servers; the median is ≈3.16 ms. No machine in Table 4 was profiled for it.
   - **The residual depends strongly on the link.** It is +35–47% of the gap on the two slow-link panel machines and +31–36% on the slow-link RTX 4090s. On fast-link machines at 25% it is negative.
   - **"Slightly overshoot" understates the 25% result.** My fast-link-only recomputation of the panel gives a residual of −9.1 points [−14.0, −4.3] at 25%; the new machines give −5.1 [−7.4, −2.9]. Both intervals exclude zero.
   - **So the attribution is not closed.** The two named parts do not add up to what is left even on the machines where the paper says they account for it.

   The paper says "attributed, not measured", which is honest. But the abstract's "we attribute the rest" and Section 4's "two named parts account for it" read stronger than the residuals support.
2. **The central 2×2 finding is engine-specific, its mechanism is unresolved, and Table 1 overclaims the control.**
   - **Why the interaction appears.** It arises because this engine's two-read path both re-reads each admission and publishes its copy two steps late.
   - **What the control did.** The registered control (Few-2R-early, job 112) was meant to separate those two causes and failed its manipulation check. Misses fell only to 0.968–0.980 of Few-2R's, while admissions rose from 6.2–7.4 to 11.7–11.8 per token and host reads by 4–7%.
   - **What Section 4 says.** The text correctly concludes that the control "cannot say how much of the two-read arms' loss is late landing".
   - **What Table 1 and Appendix D say.** Both report clause T3 as "the second read costs more than the late landing (held)". T3 compares Few-1R minus Few-2R-early with Few-2R-early minus Few-2R. Few-2R-early was *slower* than Few-2R on all four machines (by 0.031–0.053), so the right-hand side is negative and T3 holds trivially. It is not evidence for the stated conclusion.
   - **Generality.** The transferable lesson ("don't read an admitted expert twice") is long known from integrated prefetching and caching (Cao et al. 1995; Jain & Lin 2018, both cited).
3. **The scope is narrow and the key explanatory variable is noisy.**
   - **Scope.** One engine (the authors'), batch-1, teacher-forced AIME-25 (also used in development), two models, and a convenience sample of cheap rentals dominated by Ryzen 9950-class desktops.
   - **A noisy classifier.** The link-to-CPU ratio decides "fast" versus "slow" at a 0.5 line drawn after the panel's data. It moves by up to 12% between rentals of one machine: I reproduce 0.837 → 0.738 on the Ryzen 9 5950X of jobs 109f and 112g. Several registered machines sit at 0.54–0.59, so a single probe can move a machine across the line.
   - **Rule 2 over-generalizes.** It says that "a better set read twice … closes almost none of the gap". That holds for MIN's *greedy* set. But the paper's own fewest-admission set read twice gains 1.11–1.22× at ratios 0.28–0.32 (jobs 104 and 105, which I reproduce) and 1.08–1.13× at 0.44–0.49.
4. **The bound is empirical, not physical, and the headline range is post hoc.**
   - **Relative to one probe.** Eq. (1) is relative to a single probe run, mostly taken while the model downloaded. By the paper's own account the engine reads more than 5% faster than the probe on 12 of 84 launch-budgets, up to 1.22×.
   - **The top of the range.** The 54% end of "31–54%" (job 100f) is flagged as likely inflated.
   - **The bottom of the range.** The 31% end comes from a Threadripper 9960X with a 178 GB/s probe.
   - **The consumer/server split was set after the data.** Table 1 labels this, but the abstract states 31–54% and the 0.65 C rule without any post hoc qualifier.
5. **The registered tests have limited diagnostic power.**
   - **The interaction test.** Its retest was registered after narrowing the scope to fast links, where 13 of 13 panel machines had already shown a positive interaction.
   - **The trend test.** The second-card band (±0.10 in log) equals the in-sample maximum residual (0.081 at 11%, 0.094 at 25%). Three of the five RTX 4090s sat at ratios 0.37–0.40, where the trend predicts MIN-1R ≈ 1.0 and "no change" is nearly the prediction.
   - **"Held" usually means the point estimate.** In Table 1, "held" means the point estimate meets the threshold. Across all jobs only 588 of 1,588 scored clauses (37%) held with an interval. The authors disclose all of this, but a reader of Table 1 will over-read "held".
6. **Section 8 is selective in two places.**
   - **The in-step window range.** Section 8 reports that the exact 16-token window copied in the step runs "0.89–1.32× at 11% on 4 machines" (job 100). Job 099's 10 panel machines ran the same configuration, and from their raw rows I get 0.71–1.31× against the deployed cache (0.71× on the 0.29-ratio machine).
   - **The horizon rule.** It is post hoc, statistically indistinguishable from a power law (Table 17), and can only be applied to a model whose routing trace is already in hand. That is fine as an exploratory remark, but it is in the abstract unqualified.
7. **There are presentation defects in the final PDFs** (details under Clarity and IEEE format):
   - Figure 3's x-axis label is clipped;
   - Table 1's Mach. and § columns run together;
   - Table 10 is placed after Table 11;
   - the supplement prints broken cross-document references such as "Table 3paper.pdf", "Eq. (4paper.pdf)" and "Appendix Jpaper.pdf";
   - Type 3 fonts appear in all four PDFs.

## Claims checked against raw data

Values are mine, computed from the raw files (method in the notes column). ✓ means agreement within rounding.

| # | Claim | Location | My value | Verdict |
|---|---|---|---|---|
| 1 | R⋆ = 38.315 (C=14) and 15.340 (C=32) host reads/token, gpt-oss | §3, job headers | 38.3154, 15.3404 (my MIN-with-bypass on `route_aime25_gptoss.npz`) | ✓ exact |
| 2 | R⋆ on the first 20 problems changes by −0.4% | App. L | 38.170 (−0.38%) | ✓ |
| 3 | R⋆_eng (engine semantics) 39.3 / 16.0 / 100.3 / 44.9 | Table 19 | 39.25 / 16.02 / 100.33 / 44.85 (lookahead ends at problem end; served expert kept for the step) | ✓ |
| 4 | Table 3, host S: all six rows (llama.cpp, FreeToken, ours, ratio with CI, ×llama.cpp) | Table 3 | e.g. gpt-oss 11%: 28.9 / 48.0 / 57.9 / 1.207 [1.195, 1.219] / 2.00×; Qwen3 43.75%: 0.974 [0.962, 0.988] | ✓ all six |
| 5 | Table 3, host B: gpt-oss 11/25/40% and Qwen3 12.5% rows | Table 3 | 1.294 [1.278, 1.312], 1.275, 1.154, 1.032; ×llama.cpp 2.00 / 2.73 / 3.20 / 2.05 | ✓ |
| 6 | Host B B_host 88 GB/s; bound 172 tok/s; ours 41% at gpt-oss 11% | Table 3 | 87.5 GB/s; 172.3 tok/s; 40.6% | ✓ |
| 7 | On S, five of six ratios fall outside ±0.06 of B's | Table 1 | Δ = −0.087, −0.079, −0.061, −0.005, −0.105, −0.075 | ✓ (one by 0.001) |
| 8 | Table 4, panel (15): 0/0/33/15/52 and 27/21/4 at 11%; 21/−3/31/21/48 and 34/19/−5 at 25% | Table 4 | 0.5[−1.8,2.8] / −0.0 / 32.6[21.0,44.3] / 14.9 / 52.5 and 27.4 / 21.4 / 3.7 at 11%; 20.6 / −3.2 / 31.3 / 20.6 / 48.1 and 34.0 / 19.1 / −5.0 at 25% | ✓ all 16 |
| 9 | Table 4, new (5 and 4): 3/0/35/19/46 and 28/18/1 at 11%; 20/−2/27/24/48 and 38/16/−5 at 25% | Table 4 | 3.4 / −0.4 / 34.7[21.9,47.5] / 19.0 / 46.3 and 28.2 / 17.5 / 0.5 at 11%; 19.8 / −2.4 / 27.3 / 24.4 / 48.3 and 37.9 / 15.6 / −5.1 at 25% | ✓ all 16 |
| 10 | Interaction positive on 9 of 10 panel main-job machines and 5 of 5 new | §4, Table 1 | 9/10 (099f: −5.6 points of the gap); 5/5 | ✓ |
| 11 | Residual 35–47% of the gap on the two slow-link panel machines | §4 | 35.5%, 47.4% | ✓ |
| 12 | Read-ahead oracle leaves 83–99% of the gap to Eq. (3) on slow links | §4 | ≈83% (099d), ≈99% (099f) | ✓ |
| 13 | Probe +10% (+22%): shares move ≤5 (9) points for Both and ≤6 (11) for left | Table 4 note | per-machine max 5.1 (9.3) and 5.8 (10.5) | ✓ |
| 14 | Table 6, RTX 5090 job 109 (MIN-1R, ahead, best none, capture; 9 cells) | Table 6 | e.g. 285K: 1.215 / 1.460 / 1.021 / 4.5%; 7945HX: 1.241 / 1.416 / 1.024 / 5.7% | ✓ |
| 15 | Table 6, RTX 4090 rows with the bracketed trend predictions | Table 6 | e.g. 14900KF 11%: 0.963 [0.989], 1.128 [1.111], 1.033, 25.7%; i5-12400: 1.312 [1.249], 1.465 [1.375] | ✓ |
| 16 | Trend frozen in the job 110 header was fitted on the 15 panel machines | job 110 header, §7 | refit from raw: fetch 0.3053 + 0.3184 ln r, both3p 0.3935 + 0.2901 ln r (all 8 pairs match to 4 dp) | ✓ exact |
| 17 | RTX 4090s within 7% (11%) / 5% (25%); "no change" would miss by up to 46% / 69% | §7 | max 6.6% / 5.0%; 46.5% / 68.9% | ✓ |
| 18 | Trend misses 5 of 18 cells on the new RTX 5090s, by up to 18% | §7 | 5/18, max 17.6% | ✓ |
| 19 | Table 5: speeds and misses (24 entries); misses 0.97–0.98×; admissions 11.7–11.8 vs 6.2–7.4; reads +4–7%; 0.03–0.05 slower; T1 failed 5/8, T2 7/8 | Table 5, §4 | all 24 match; 0.968–0.980; 11.72–11.75 vs 6.20–7.37; +4.2–7.1%; −0.031 to −0.053; 5/8; 7/8 | ✓ |
| 20 | T3 "second read costs more than the late landing (held)" | Table 1, App. D | Early arm slower than Few-2R on 4/4 machines, so T3's right side is negative and the clause holds trivially | **Holds numerically; the stated conclusion is not supported** |
| 21 | No-foresight variants: best 1.007–1.024×, capture ≤ 6%, reads 1.57–1.76 R⋆; Margin cuts reads 4–6% | §6 | 1.007–1.024; max 5.7%; 1.575–1.755 R⋆ (deployed 1.65); −4.3% on job 109 | ✓ |
| 22 | RTX 4090s: layer-ahead variants 0.64–0.72×; Margin 1.033–1.041×; capture up to 26% | §6 | 0.644–0.724; 1.033–1.041; 25.7% | ✓ |
| 23 | 31–54% of the bound on consumer machines at 11%; new machines 41–48%; servers 12–46% | §3, Table 1 | consumer launches 30.6–54.0% (30.6 is the TR 9960X's second launch); 41.5–48.0; 12.4–46.1 | ✓ |
| 24 | Running example (9950X, O4): Eq. (1) 10.0, Eq. (2) 13.0, deployed 20.6, MIN-1R 15.2, read-ahead 13.8 ms | §5 | 10.02 / 12.96 / 20.64 / 15.17 / 13.85 | ✓ |
| 25 | Two panel machines with the same CPU differ by up to 29% | §2 | 285K: 18.14 vs 14.06 ms (29.0%) | ✓ |
| 26 | Eq. (3)/Eq. (1): median 1.00, up to 1.56 | §3 | 1.000 / 1.557 over 32 consumer launches (the max is the TR 9960X) | ✓ |
| 27 | Microbenchmark reaches 86–96% of the bound's read time; repetitions within 0.9% | §3, Table 29 | 0.861–0.963 (per-layer mode); 0.88% | ✓ |
| 28 | Few-1R 1.24× (geometric mean) on 13 stable machines, mean 1.24 [1.16, 1.32]; loses on ratios 0.14 and 0.21 | §5, App. I | geometric mean 1.240; arithmetic mean 1.247 [1.168, 1.325]; 0.845 and 0.974 | ✓ (interval differs by ≤0.01) |
| 29 | Fewest-admission schedule copies 0.60–0.73× greedy's; +2.5–5.3% misses | §5, App. I | 0.60 / 0.73; +2.5% / +5.3% (job 104) | ✓ |
| 30 | Spearman 0.89 between MIN-1R gain and ratio over 19 machines (jobs 093–104) | §7 | 0.89 over 19 unique GPUs | ✓ |
| 31 | MIN-1R beats the deployed cache at every budget on O3–O5; double-read or no-bypass ≤ +15% at the smallest budgets | §5, Fig. 3 | min MIN-1R 1.16; max 1.15 (Belady-1R, Qwen3 12.5%, O4) | ✓ |
| 32 | Exact 16-token window recovers 0.78–0.93 of the gain in time at 11% | §8 | 0.78–0.93 (job 099, 10 machines) | ✓ |
| 33 | The same window copied in the step: 0.89–1.32× vs the deployed cache "on 4 machines" | §8 | job 100: matches; job 099's 10 machines: 0.71–1.31× | **True as scoped; omits the wider panel** |
| 34 | Half-gain window ≈ 0.66–0.81 C distinct experts on the AIME routing | §8 | 0.65–0.80 (my D(W) at log-interpolated W50 from Table 19) | ✓ (approximate) |
| 35 | T_GPU = 2.9 ms, the smallest of 14 profiles; G 4.1–4.7 ms | §3, App. H | 2.940 (range 2.94–3.38, median ≈3.16); G 4.07–4.69 | ✓ |
| 36 | Round-to-round speed-ratio change: median 0.2%, at most 1.8%, 98 comparisons | §2, App. D | n = 98, median 0.17%, max 1.74% | ✓ |
| 37 | 92 rentals of jobs 093–112, each ≥ 2.7 s after a commit; machines start after their script is pushed | App. D | 92; min 2.71 s; each per-machine script committed ≥ 29 s before its manifest start | ✓ |
| 38 | Second probe within 4.9% on 4 machines; relaunched deployed time within 0.7%; ratio moved up to 12% | §11, App. D | 4.9%; 0.7%; 11.8% | ✓ |
| 39 | Job 109: 11 of 11 predictions held | App. D | P1–P11 all hold on my numbers (P8's ≥ 0.9 arm untestable) | ✓ |
| 40 | Few-2R pays instead of MIN-1R at ratios 0.28–0.32 (1.11–1.22×) | §7 | 1.112–1.217 (jobs 104a, 105b, 105e) | ✓ |

What I did not verify: the 9-model trace study behind "0.65 C"; the audit of published systems (median 13.6%); the rental-to-rental 0.5% / 5.3% figures; Qwen3 25% and 43.75% on host B; and the clause-by-clause hand scores of jobs 073–098.

### Registered tests, evidence labels, post hoc labelling, scope

- **The registered rows of Table 1 match the job headers.** Job 081 P2–P4 and job 089 P2 cover the system claim. Job 099 P5 and job 109 P1 and P3 cover the 2×2. Job 109 P7 and Q3 of jobs 111/112 cover the online policies. Jobs 104 P3, 105 P4, 106 P6/P7 and 107 P5 cover Few-1R. Job 110 Q1 covers the second card, and job 112 T1–T4 the timing control. The headers predate each machine's start.
- **Two things in the headers are disclosed in the text but deserve emphasis.**
  - Job 109's population gate (ratio ≥ 0.5) is explicitly "drawn after the data of jobs 093–101".
  - Job 111 changed the V1 reference after job 110 had produced one round on those machines. The authors handle this transparently (job 110 is reported void, and the first rounds are given).
- **The evidence labels in Table 1 are mostly fair.** The "loose" labels on the microbenchmark floor, the capture bound and the MIN-1R share are correct. I would add that the sign-of-interaction retest followed 13 of 13 positive fast-link panel machines, and that the trend band equals the in-sample maximum residual. The T3 wording should change (row 20 above).
- **Post hoc labelling is good in Table 1 but missing in the abstract.** In Table 1 and the Section 8 header ("exploratory") it is done well. The abstract states the 31–54% range (consumer scope set after the data) and the 0.65 C rule without qualification, and §3 does not mark the 31–54% as post hoc in the running text.
- **Scoping is careful in the abstract and §9's preamble ("elsewhere they are hypotheses to test").** Rule 2's first sentence is broader than the evidence, and §4's "two named parts account for it" is stronger than the fast-link 25% residual allows.

## Clarity

The paper is clearer than an MLSys reader would expect from 112 jobs of history, and clearer than a version graded 2/5 would have been. It is still hard work.

**What works**

- **Table 1 as a claims ledger.** It gives the claim, the registered threshold, the result, the machine count and the section, with "derived", "post hoc", "loose" and "held" labels. A reviewer can audit the paper from it. This is the single best addition.
- **Table 2.** It organizes the configurations on a grid of admitted set against read twice / read once / read ahead, with the machine sets and the five yardsticks beside it. The 1R/2R naming is mnemonic.
- **Figure 1.** One schematic step, which makes "read twice" concrete.
- **The introduction.** Five bold lead-in findings, each pointing to its section, then the rules in §9.
- **The running example in §5** (9950X: 10.0 / 13.0 / 20.6 / 15.2 / 13.8 ms). It anchors the abstractions in milliseconds.
- **Limitations (§11) and Appendix D.** These are frank about deviations, failed gates and amendments.
- **Table 6.** The bracketed predictions next to the measurements make the second-card test legible at a glance.

**What still makes it hard to read**

1. **The vocabulary is large and has to be held in memory across sections.**
   - **Main text:** fifteen configuration names (Dep-1R, Margin, LA, LA-1R, LA-1R-margin, admit every miss, MIN-1R/2R, Few-1R/2R, Few-2R-early, Belady-1R/2R, the read-ahead oracle, the deployed cache); five machine sets; five yardsticks; three bounds (plus Eq. (4) in the appendix); and the terms launch, round, ran stably, host-bound, fast/slow link, R⋆ versus R⋆_eng, B_c, B_p, B_host, B̂_c, B̂_p, T_GPU versus G.
   - **Appendices:** they add 25 configuration codes (Table 7) and host labels O1–O6, Pa–Pj and host A.
   - **Where it hurts most.** §4's argument needs MIN-2R, Dep-1R, MIN-1R, Few-2R, Few-2R-early and Few-1R within one paragraph.
2. **The yardsticks and baselines change between sections.**
   - Fig. 2 plots *time* relative to the deployed cache (lower is better). Figs. 3–4 and Tables 5–6 plot *speed* relative to the deployed cache (higher is better).
   - Table 4 uses shares of the gap to Eq. (1). §6 and Table 6 use capture of the read-ahead oracle's gain.
   - §8 and Fig. 5 switch to shares of MIN's read saving over *admit every miss*, a different baseline. The caption flags this, but the reader still has to re-anchor.
   - Choosing one primary yardstick for the main text (share of the gap, say) and putting the rest in the appendix would help.
3. **The prose packs too many numbers into each sentence.** By my count the running text of the main body carries roughly one numeric value every 20 words, with nested parentheticals.
   - **§4 "What is left".** In 8 lines it gives six ranges, two bounds and a median.
   - **§2's "Statistics and registration" paragraph.** "Widened by its round range, each of the 89 per-machine registered clauses … 49 do, and 29 of the other 40 are bands of ±0.06 or narrower" is audit scaffolding sitting in the Setting section. One sentence plus a pointer to Appendix D would do.
4. **Table 1 has readability defects.**
   - **The Mach. and § columns run together** ("10 (15) + 5 4", "13 5", "25 3").
   - **The Result column does not always state the result.** For "Our cache reaches 31–54%…" it reads only "server machines: 12–46%".
   - **"Held" means the point estimate only.** Most readers will assume statistical support unless they read the caption closely.
   - **The T3 phrasing contradicts §4.**
5. **"Host-bound" is defined three ways in one paragraph** (§2, at datasheet versus measured GPU rates, with exceptions). It matters for reading the 25% results.
6. **The appendices read as a lab notebook.** Job numbers and machine letters carry the narrative (for example, "Job 103's 5950X had turned out to be Pg's second launch"). Table 11 precedes Table 10, and Table 10 floats alone on a mostly blank page. Grouping Appendix D by claim rather than by job, as Table 9 starts to do, would make it navigable.
7. **There are figure and production defects.**
   - Fig. 3's x-axis label is clipped at "grey ticks: panel" (the text layer has "panel machines)").
   - Fig. 2 puts two different states on one x position ("MIN-2R (or Dep-1R)").
   - Fig. 4 overlays four marker families and fitted lines in two small panels.
8. **The supplement has broken cross-document references and clashing numbers.**
   - It prints the document name after the reference, as in "Table 3paper.pdf", "Eq. (4paper.pdf)", "Eq. (1paper.pdf)", "Appendix Jpaper.pdf" and "Section 5paper.pdf" (pp. 18–23).
   - Its own Table 3 and Eqs. (1)–(2) collide with the main paper's (S1's "Eq. (1)" is a calibrated time model, not the bound).
9. **The abstract is hard going.** It is 238 words of dense, specialized sentences: MIN with bypass, the greedy schedule, "machines whose link reads at least half as fast as the CPU", "an oracle that reads ahead". A general MLSys reader will not parse it on a first pass.

On balance the paper is understandable to a motivated specialist, with effort. I score clarity **3/5**. It is better than the earlier 2, but the density, the vocabulary load and the defects keep it from 4.

## Questions for the authors

1. **Table 4 with other T_GPU values.** Can you recompute Table 4 with each machine's own profiled non-expert time, or at least the median of the 14 profiles, and report the fast-link-only means? Does the 25% overshoot (−9 points on fast-link panel machines by my count) persist?
2. **A cleaner timing control.** Why does Few-2R-early admit nearly twice as many experts as Few-2R? Can you design a control that holds the admission set fixed (for example, excluding planned experts from fetch-table fills) so that late landing and the second read can be separated? Until then, will you reword T3 in Table 1 and Appendix D?
3. **Ratio drift.** How robust are the fast/slow classification and the second-card predictions to the ±12% drift of the ratio between rentals? Would the second probe change any machine's class or any registered outcome?
4. **The probe and the bound.** Does the 31–54% range change if B_host is replaced by a rate the engine's access pattern actually reaches (for example, the microbenchmark's token mode)? Why use the maximum over all probe readings rather than the median of the concurrent readings (Table 28 shows the median lowers the bound by about 8%)?
5. **Rule 2.** How do you reconcile Rule 2 with Few-2R gaining 1.08–1.22× at ratios ≤ 0.49?
6. **Wider panel in §8.** Why does §8 report the in-step window on job 100's 4 machines rather than job 099's 10, which ran the same configuration and include a 0.71× case?
7. **Generality.** Is there any evidence beyond your engine, for example the read-twice penalty in another offloading system, or batch > 1 such as speculative verification?

## What would raise my score

- Measure T_GPU on each machine (or report the attribution with a band from the profile range), and soften "account for it" where the residual excludes zero.
- Correct the T3 row in Table 1 and Appendix D. Better still, run a timing control that holds admissions fixed.
- Label the post hoc results (31–54%, 0.65 C) in the abstract. Scope Rule 2 to MIN's greedy set. Report the job 099 window range alongside job 100's.
- Clarity:
  - one primary yardstick in the main text;
  - move the rounds and rentals paragraph out of §2;
  - fix Table 1's columns and its result for the 31–54% row;
  - fix the Fig. 3 label and the supplement's cross-references;
  - reorder Tables 10 and 11.
- Any replication outside this engine (another offloading engine, or batch > 1) would raise significance substantially.

## IEEE format (ieee-paper.pdf and ieee-supplement.pdf)

Skimmed against IEEE conference conventions:

- **Front matter.** The IEEEtran title and author block are anonymous ("Anonymous Author / Anonymous Institution"), which is fine for review. The abstract is bold, as IEEEtran sets it. **Index Terms** are present and alphabetical with only the first term capitalized: correct.
- **Captions.** Main-paper tables are captioned above in small caps ("TABLE I") and figures below ("Fig. 1."): correct. In the supplement the long table is captioned "TABLE S5: Failed predictions…" (colon, sentence case, different layout), unlike the other supplement tables ("TABLE S2" on its own line). Supplement tables use Arabic S-numbers while the main paper uses Roman numerals. That is acceptable, but the mix should be deliberate.
- **References.** The main paper's references are numbered in citation order and formatted in IEEE style: correct. The supplement has its **own reference list restarting at [1]** (Hoefler and Belli), while the main paper's [1] is Ivanov et al. The same label therefore names different works in the two documents. Either continue the numbering or make the supplement's list visibly separate (for example, "[S1]").
- **Float placement.** Table I spans the top of page 2, and Table III and Fig. 2 share page 5: fine. **The last page's columns are not balanced.** The references end in the left column and the right column of page 10 is empty; IEEE practice is to balance them (`\IEEEtriggeratref` or `balance`).
- **Page count.** 10 pages including references. That fits venues with a 10-page-inclusive limit but exceeds the common "8 + references" limit; check the target venue's CFP.
- **Cross-references between the two documents.** The main paper's references to "Appendix D", "Table S21–S23" and "Eq. (S1)" resolve correctly, and the page-1 footnote names `ieee-supplement.pdf`. But the IEEE supplement still refers to a third document, "the supplement (paper/supplement.pdf)", for the clause-by-clause tables. That document is the MLSys-format supplement and is not part of the IEEE package; it should be merged in or renamed consistently, and a repository path should not appear in the paper.
- **Fonts.** `pdffonts` finds Type 3 fonts in `ieee-paper.pdf` (7, DejaVu from matplotlib figures) and `ieee-supplement.pdf` (4). IEEE PDF eXpress typically rejects Type 3 fonts; re-export the figures with `pdf.fonttype: 42` or as Type 1. All fonts are embedded. The same applies to `paper.pdf` (11 Type 3) and `supplement.pdf` (3).
- **Figures.** Fig. 3's x-axis label is clipped at "grey ticks: panel" here too.

## Scores

| | Score |
|---|---|
| Overall (1–10) | **6** (weak accept) |
| Soundness (1–5) | **4** |
| Significance (1–5) | **3** |
| Novelty (1–5) | **3** |
| Clarity (1–5) | **3** |
| Confidence (1–5) | **4** |

**Justification.**

- **Soundness: 4.** The data reproduce exactly and the registration is real and honest. I hold it at 4 because half of the decomposition is attribution with a non-closing residual, the central mechanism is unresolved after a failed control, and Table 1 overstates that control.
- **Significance: 3.** The bound-as-yardstick norm and the oracle-in-engine methodology are valuable. The specific policy findings come from one engine at batch 1 on one workload.
- **Novelty: 3.** It is moderate. Each element (a roofline-style bound, Belady with bypass, the cost of double reads, paging with lookahead) is known; the time-domain, in-engine combination is new.
- **Clarity: 3.** Improved and auditable, but still dense, name-heavy and marked by several production defects.
