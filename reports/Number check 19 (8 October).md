# Number check 19 (8 October)

Independent number check of "Where the Seconds Go" at HEAD 9da42d9, covering the revision 18ef2d7..HEAD.

## Scope

- **Main text:** the abstract, the introduction's claims, Table 1 (`tab:claims`), the Setting (budgets C=14/32, host-bound at datasheet and measured GPU rates, terms, fast/slow, statistics with t-intervals), the Section 3 audit paragraph, all of Section 4 (the 2x2 paragraph, the new timing-control paragraph, "What is left" with the new Eq. (3) sentence), Section 5, Section 6 (6.1, 6.2, the simpler-predictor baselines, the footnote), the new deployed-cache yardstick sentence in Section 7, the Section 8 rules, Limitations and the Conclusion.
- **Tables:** every cell of `tab:gap` (tab_dm.tex), every row of `tab:job109` (tab_job109.tex), including the bracketed trend predictions and subscripts; every cell of `tab:timing` (tab_job112.tex); tab_configs.tex and tab_names.tex against the job scripts' `--ec` strings.
- **Job 112 in full:**
  - per-host recomputation for 112a, 112b, 112d and 112g;
  - all 58 registered clauses (Q1–Q6, T1–T7), compared with prereg/scorecard_112.json and paper/wsg_job112.tex;
  - registration timing against the ledger, git and the push records;
  - the 112c/112g narrative.
- **Appendix:**
  - app_prereg.tex: the job 112 paragraph, the job 109 severity note, the scoring and tally paragraphs, and the registration-timing macros;
  - tab_failures.tex: 25 or more rows spot-checked;
  - tab_prereg.tex: every row and the totals.
- **Statistics:** the t-interval macros in wsg_dm.tex and wsg_robust.tex.
- **Macro rule:** typed measured numbers in the changed text.
- **Build:** a clean build from `git archive HEAD paper`.

## Method

All code is in `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/nc19/`, run with that folder as the working directory. Everything was written from the job headers' definitions. The repository scripts were read for definitions only, and none of their functions are imported.

| Script | What it does |
|---|---|
| `hosts.py` | Independent loader. X/base = base's mean ms/token over the problems ÷ X's, within the same process, then the geometric mean over rounds. Counters are per step and averaged over rounds. B_host is the highest probe reading of any method. ratio = B_p/B_c from the fetch table. Also holds my own paired bootstrap over problems (one resample per draw, shared across configurations and rounds). |
| `j112.py` | Job 112 per host, tab:timing and the job 112 macros. |
| `score112.py` | All 58 job 112 clauses scored, with my own intervals, and diffed against scorecard_112.json. |
| `tab5.py`, `border.py` | Every row of tab:job109: values, half-widths and bracketed trend predictions (job 110's frozen coefficients, typed from its header). |
| `panel.py`, `cards.py` | The 15 panel machines and job 109's machines from raw jsonl and counters. Eq. (1) is the full min–max with the datasheet GPU rate. Shares and 95% t-intervals (mean ± t₀.₉₇₅,ₙ₋₁·s/√n). Also residuals and Eq. (3) shares on the RTX 4090s, and a refit of the frozen trend (it reproduces job 110's header coefficients exactly). |
| `j109m.py` | Job 109/110/111-derived macros. |
| `robust.py` | rbPlan* and rbDk* from raw jsonl: geometric t-intervals on the log scale. |
| `j100.py` | The w16/base yardstick of job 100. |
| `regtime.py` | Ledger starts against git commit times and the push records, for all 92 rentals of jobs 093–112. |
| `rerent.py` | Link-to-CPU ratio across re-rentals of the same GPU. |
| `macros.py` | Resolves every macro used in paper.tex and app_prereg.tex. |

The tally checks read prereg/scorecard_*.json and scorecard_clauses.json directly. The build check unpacked `git archive HEAD paper` into `nc19/build/` and ran `latexmk -pdf -interaction=nonstopmode paper.tex` there.

## Items checked

Verdicts: **OK** means my independent value matches. **DEFECT** means the text or value is wrong or misleading. **NOTE** covers boundary cases, wording issues and items that could not be checked.

| # | Location | Claim (as printed) | My value | Verdict |
|---|---|---|---|---|
| 1 | Build | clean `git archive` copy, `latexmk -pdf` | exit 0; 0 lines starting "! "; 0 undefined refs; 0 multiply defined; 44 pages, 661,508 bytes (same size as the committed PDF); one hyperref "empty anchor" warning | OK |
| 2 | Abstract | 31–54% of the bound's speed (\dcShareMin*) | not recomputed (unchanged wsg_decomp macro) | NOTE |
| 3 | Abstract | the two changes together close 35% of the gap in a registered test | new machines, together share: 35% | OK |
| 4 | Abstract | "Policies without foresight recover at most 6% of the oracles' gain." | 6% (5.7%) is job 109's fast-link maximum; on job 111's RTX 4090s Margin captures 20–26% at 11% | **DEFECT D1** |
| 5 | Intro ¶2 | 33% on 15 panel machines; 35% on 5 rented later | 33 [21, 44] on 15; 35 on 5 | OK |
| 6 | Intro ¶3 | four variants recover ≤6% "on those machines" | 5.7% max (job 109) | OK |
| 7 | Intro ¶4 | trend predicted five RTX 4090s at ratios 0.37–0.77 within 7% | 5 machines; 0.371–0.771; max deviation 6.57% | OK |
| 8 | Intro ¶4 | new RTX 5090s scatter more | up to 17.6% (18) | OK |
| 9 | Table 1 caption | "``15+5''" typed | equals \dmMachines + \olN but typed | **DEFECT D10** |
| 10 | Table 1, row 2 | ≥50% per layer; 5 of 33 clauses failed; 3 machines | job 102 header ≥0.50; scorecard_102: 33 clauses, 5 failed; 3 hosts | OK |
| 11 | Table 1, row 3 | ±0.06; outside at five of six; faster at 11 of 12 | 089-P2 a, b, c, e, f failed (5 of 6); tab:headline: only Qwen3 43.75% on host S < 1 | OK |
| 12 | Table 1, row 4 | interaction > 0 on every machine; held on 9 of 10 panel, 5 of 5 new; 15+5 | 9/10 (099f negative at 11%); 5/5 new (4/4 at 25%) | OK; NOTE N7 |
| 13 | Table 1, row 5 | misses ≤0.95× and no slowdown: failed, failed; second read > landing: held; 4 | T1 failed 5/8; T2 failed 7/8; T3 held 4/4 (with CI); 4 valid | OK |
| 14 | Table 1, row 6 | capture ≤35%; at most 6%; 5 | 5.7%; 5 | OK |
| 15 | Table 1, row 7 | 0.37–0.77; within 0.10 log, held (7%); new 5090s up to 18%; 5 | all Q1 clauses of jobs 111/112 held; 6.57%; 17.6%; 5 | OK |
| 16 | Table 1, row 8 | Few-1R gains on every machine that ran it stably; 7 registered; Mach. 9 | 9 = jobs 104–107 only; job 112 ran Few-1R stably on 4 more machines (1.21–1.41× at 11%) | **DEFECT D6** |
| 17 | Table 1, rows 9–11 | 13.6%; ≈0.65 C on 9 models; 3 of 4 failed | macros not recomputed; 3 failed + 1 inconclusive of 4 matches Section 4 | NOTE |
| 18 | Setting | 11% and 25% are C=14 and C=32 of 128 | 10.9%, 25.0% | OK |
| 19 | Setting | at the datasheet rate host-bound except gpt-oss 25% on the fastest host memory | GPU term binds at C=32 when B_host > ≈106 GB/s (e.g. 105b/106a at 178 GB/s) | OK |
| 20 | Setting | at the measured batch-1 rate the GPU term also binds at gpt-oss 25% on both hosts of tab:headline | yes (430→279, 351→275); it also binds at Qwen3 25% on host B (214→192), not mentioned | **DEFECT D8** |
| 21 | Setting | fast ≥0.5, drawn after the panel's data, before the registered tests | job 109 header fixes 0.5 "after the data of jobs 093–101" | OK |
| 22 | Statistics | same-CPU panel machines differ up to 29% | 29.1% (two Core Ultra 9 285K) | OK |
| 23 | Statistics | re-rented within 3.2% (5090) and 5.5% (4090) | 4090: 5.49% (111g/110c); job 112's 5090 relaunches 0.42% and 0.71% (consistent) | OK |
| 24 | Statistics | "summaries over machines carry 95% t-intervals … with 3 to 15 machines" | t-intervals also over 21 and 25 machines (rbErrFound, rbErr); "3 to 15" typed. The Spearman CI in 6.2 is a percentile bootstrap over machines | **DEFECT D10**; NOTE N5 |
| 25 | Yardsticks | "(``1.22×'')" typed | illustrative example | NOTE N3 |
| 26 | Section 3 audit ¶ | 20 measurements, median 13.6%; ours 27%; three caveats | macros not recomputed (audit data, not raw GPU results); the caveat logic is consistent | NOTE |
| 27 | Section 3 | job 109 machines at 41–48% | 41.49–47.95% | OK |
| 28 | Section 4 2x2 ¶ | MIN-2R 0%, Dep-1R 0%, MIN-1R 33% (panel) and 35% (new) | 0 [−2,3], 0 [−2,2], 33, 35 | OK |
| 29 | Section 4 timing ¶ | four machines, both cards, ratios 0.57–0.88 | 0.572, 0.738, 0.771, 0.883 | OK |
| 30 | Section 4 timing ¶ | Few-2R-early 0.98–1.00×; Few-2R 1.03×; Few-1R 1.21–1.41× | 0.978–0.997; 1.027–1.030; 1.209–1.406 | OK |
| 31 | Section 4 timing ¶ | "leaves the two-read arm where it was" | early arm slower by 0.031–0.053 on all 4 machines; whole intervals < −0.01 | **DEFECT D7** |
| 32 | Section 4 timing ¶ | misses 0.97–0.98 of Few-2R's (registered ≤0.95) | 0.968–0.980; T1 threshold 0.95 | OK |
| 33 | Section 4 timing ¶ | above Few-1R's by 12.1–13.8 per token | 12.066–13.848 | OK |
| 34 | Section 4 timing ¶ | fetch table fetches 17.8–23.7 of 53.0–54.6 misses | 17.84–23.68 of 52.97–54.58 | OK |
| 35 | Section 4 | registered on 10 machines, held on 9; new: 5 of 5 | 10 / 9; 5 | OK |
| 36 | Section 4 | at 25% MIN's set alone closes 21% | 21 [19, 22] | OK |
| 37 | Section 4 "What is left" | ahead 15%, left 52% | 15, 52 | OK |
| 38 | Section 4 "What is left" | slow panel residual 35–47% | 35.48, 47.41 | OK |
| 39 | Section 4 "What is left" | "31–36% on the RTX 4090s of §6.2" | 31.5–35.8% on job 111's three only; job 112's RTX 4090s: 5.4% and 14.4% | **DEFECT D3** |
| 40 | Section 4 "What is left" | Eq. (3) left 83–99% (two machines), 73–81% (RTX 4090s), median 45% fast | 83.1, 98.6; job 111: 72.7–80.6 (job 112's: 41.7, 57.4); fast median 45.49 (13 machines) | values OK; scope **DEFECT D3** |
| 41 | Section 4 closed form | 4 tests, 3 failed, 1 inconclusive | macros consistent with Table 1 | OK |
| 42 | Section 5 | Few-1R beat the deployed cache on all 9 stable machines, geometric mean 1.21× (registered on 7) | 9 machines, 1.2147 [1.1104, 1.3288] from raw (jobs 104–107); job 112 adds 4 stable machines → 13, ≈1.24 [1.16, 1.32] | value OK; **DEFECT D6** |
| 43 | Section 5 | lost on two unsteady machines, the slowest links | 106c 0.845, 107e 0.974 lost; 106d, 107c gained | OK |
| 44 | Section 6.1 | Margin cuts reads 4–6% at 11% | 4.3–5.8% (jobs 106–107) | OK |
| 45 | Section 6.1 | registered ≤1.10× and ≤35% capture on any machine | job 109 header P7 | OK |
| 46 | Section 6.1 | best 1.007–1.024×; at most 6% | 1.007–1.024; 5.7% | OK |
| 47 | Section 6.1 | read 1.66–1.68 R*, as many as the deployed cache | 1.6648–1.6777 (deployed 1.645–1.660) | OK |
| 48 | Section 6.1 | slow-link RTX 4090s: layer-ahead 0.64–0.72×; Margin best 1.033–1.041×; ≤26% | 0.644–0.724; 1.033–1.041 (dk on all 3); 25.7% | OK |
| 49 | Section 6.2 | frozen trend on 15 panel machines | refit from raw reproduces all 16 coefficients and the max residuals (0.081, 0.094) | OK |
| 50 | Section 6.2 | first test 3 machines at 0.37–0.40; second added 2 at 0.57–0.77 | 0.371, 0.392, 0.396; 0.572, 0.771 | OK |
| 51 | Section 6.2 | within 7% at 11% and 5% at 25%; every 95% interval inside the band | 6.57%, 4.99%; max \|log CI bound\| 0.074 < 0.10 | OK |
| 52 | Section 6.2 | interaction positive on the two fast-link machines | 6.60 ms, 2.98 ms (Q6 held) | OK |
| 53 | Section 6.2 footnote | loss 3% higher; first rounds within 6% | 0.1963/0.190 = +3.3%; 6.23% (110b–e) | OK; NOTE N8 |
| 54 | Section 6.2 | no change misses up to 46% / 69% | 46.5% (112a ahead 1.4649), 68.9% (1.6893) | OK |
| 55 | Section 6.2 | slow-link panel mean (two machines) misses up to 38% / 34% | slow means: fetch 0.963/1.046, ahead 1.060/1.259 → 38.2%, 34.2% | OK |
| 56 | Section 6.2 | trend misses five of 18 cells, up to 18% | 5 of 18; 17.59% | OK |
| 57 | Section 6.2 | "At the RTX 4090s' ratios MIN-1R runs 0.96–1.01×" | 0.963–1.007 on job 111 only; job 112's RTX 4090s: 1.133, 1.312 | **DEFECT D4** |
| 58 | Section 6.2 | slow 5090s 0.28–0.32, Few-2R 1.11–1.22× | not recomputed (wsg_job106, unchanged) | NOTE |
| 59 | Section 6.2 | registered "within 0.10 of it in log" for all four configurations | registration: 0.10 for MIN-1R and ahead (25%: all but one), 0.06 for Dep-1R and MIN-2R | NOTE N9 |
| 60 | Section 7 new sentence | w16 vs deployed 0.89–1.32× at 11% on 4 machines | 0.8914, 1.3200, 1.2048, 0.8919 | OK |
| 61 | Section 7 new sentence | "below 1 on a slow-link one" | below 1 on two: 100a (ratio 0.41) and 100f (ratio 0.54, fast by §2's definition) | **DEFECT D5** |
| 62 | Section 8, rule 1 | 31–54% | unchanged macro | OK (not recomputed) |
| 63 | Section 8, rule 2 | fast links: "about a third" | fast panel 39% [31, 47]; new 35% | NOTE N4 |
| 64 | Section 8, rule 4 | within 7% over 0.37–0.77 | as #51 | OK |
| 65 | Limitations | second card for two registered tests | jobs 110/111 and 112 | OK |
| 66 | Limitations | one machine never finished downloading; its replacement started before it was given up | ledger: 112c 05:42:56Z–07:44:15Z (7,279 s), 112g created 06:54:28Z; no results/112c | OK |
| 67 | Limitations | second probe within 4.9% on all 4 machines, idle machine | B_host2/B_host − 1: +0.81, +4.90, −0.44, −0.28%; probe 2 after the model was deleted | OK |
| 68 | Limitations | ratio moved up to 12% between two rentals | 112g vs 109f −11.8%; 112d vs 109c −0.6% | OK; NOTE N6 |
| 69 | Conclusion | 31–54%; registered and replicated; trend predicted second card | consistent with the above | OK |
| 70 | tab:gap, all 32 cells | e.g. Panel 11%: 0[−2,3], 0[−2,2], 33[21,44], 15[11,19], 52[43,62], 27[25,30], 21[20,23], 4[−6,13]; New 25%: 20[18,21], −2[−3,−1], 27[23,32], 24[21,27], 48[44,53], 38[33,43], 16[15,17], −5[−7,−3] | all 32 cells reproduced exactly with t-intervals; machine counts 15/5/15/4 | OK |
| 71 | tab:gap caption | 15 machines of jobs 096–101; 5 of job 109; one 25% round cut; T_GPU 2.9 ms | 15; 5 (4 at 25%); 109f's C32 B round skipped; 2.940 ms | OK |
| 72 | tab:job109, job 109 rows (5 machines) | ratios, MIN-1R, Ahead, None, Capture, subscripts | every value and subscript reproduced (e.g. 109e 1.215→1.22±.01, capture 4.48→4±0) | OK |
| 73 | tab:job109, job 111 rows (3 machines) | values, brackets, subscripts | all reproduced; brackets e.g. 0.37: [0.99][1.11][1.07][1.29] | OK; NOTE N2 (111g ±.00 vs 0.0052) |
| 74 | tab:job109, job 112 rows | 7800X3D 0.57: 1.13[1.14] 1.30[1.26] 1.01 4%; 1.17[1.21] 1.48[1.47] 1.02 5%. i5-12400 0.77: 1.31[1.25] 1.46[1.37] 1.01 3%; 1.35[1.31] 1.69[1.61] 1.02 3% | 1.133[1.136] 1.299[1.261] 1.013 4.2%; 1.166[1.208] 1.484[1.472] 1.023 4.8%; 1.312[1.249] 1.465[1.375] 1.013 2.9%; 1.351[1.314] 1.689[1.609] 1.021 3.0%; subscripts match | OK; NOTE N2 (i5-12400 Ahead 11% half-width 0.0144–0.0151 across seeds, printed ±.01) |
| 75 | tab:job109 caption | geometric mean of rounds; brackets from the frozen trend; subscripts half the paired bootstrap interval | matches my method | OK |
| 76 | tab:timing, all 28 cells | 5950X 0.74: 1.41 1.03 0.98 / 40.7 55.6 54.5; 7945HX 0.88: 1.25 1.03 1.00 / 40.7 55.7 54.6; 7800X3D 0.57: 1.21 1.03 1.00 / 40.9 54.7 53.0; i5-12400 0.77: 1.35 1.03 0.98 / 40.8 55.7 54.5 | identical | OK |
| 77 | tab:timing caption | every 95% interval within ±0.01 | max half-width 0.012 (112b Few-1R 0.0119; 112g 0.0117) | NOTE N1 |
| 78 | tab_configs / tab_names vs `--ec` | base, foa, bypass, fetch, both3p, dk (κ 3/2), pf, R1, R2, fetchplan, bypassplan, bypassplanS, aa, w*, b* | all consistent. both3p's `oracle_bypass=0` + `oracle_lead=3` admits by MIN-with-bypass in the engine (patch line ≈3234), so "MIN's admissions copied ahead" holds | OK; NOTE N12 |
| 79 | names.tex | Few-2R-early = bypassplanS (job 112) | matches header | OK |
| 80 | Job 112 per host | 112a i5-12400 (4090): ratio 0.7713 / ratio2 0.7667; B_host 37.1 / B_host2 37.4. 112b 7800X3D: 0.5723 / 0.5594; 49.0 / 51.4. 112d 7945HX (5090): 0.8833 / 0.8805; 68.3 / 68.0. 112g 5950X (5090): 0.7383 / 0.7416; 36.1 / 36.0 | recomputed from concur.txt, concur2.txt and the fetch tables | OK |
| 81 | wsg_job112 macros | tcRatio*, tcEarly/Two/One*, tcMiss*, tcMissRatio*, tcLag*, tcRead*, tcSlow*, tcEarlyMinusFetchMiss*, tcFetchesEarly*, tcProbeDevMax 4.9, tcProbeDevMean +1.2, tcRatioDevMax 2.3, tcRelaunchMax 0.7, tcRelaunchRatioDevMax 12, tcFourDev 7/5, tcFourDevSmall 1/2, tcFourBest 1.013 / 1.021–1.023, tcFourCap 4/5, tcFourFetch, tcFourOracle, tcQ* 6/6/0, tcT*Status/N/Failed, tcClauses 58 (22/24/12) | all reproduced (e.g. ReadMidMin 0.1055→0.11; ProbeDevMean 1.248) | OK |
| 82 | wsg_job112 | tcStarted = 4 | five machines rented (112c had no results directory); macro unused in text | NOTE N10 |
| 83 | Q1–Q6 | Q1 9 clauses held (8 with CI); Q2 2 held; Q3 4; Q4 2.196/3.284/2.179/3.449 in band; Q5 1.664–2.196 ≥1.3; Q6 6.60, 2.98 ms; pooled 25% misses 0 | 25 clauses, all statuses equal to scorecard_112.json; values within ±0.0001 | OK |
| 84 | T1–T7 | T1 0.979/0.949/0.968/0.953/0.980/0.949/0.980/0.950 (5 failed); T2 −0.047/−0.019/−0.032/−0.020/−0.031/−0.006/−0.053/−0.027 (7 failed); T3 0.41/0.24/0.28/0.48 held; T4 0.98–1.00 held; T5 point; T6 +0.42%, −0.71%; T7 0.21, 0.35; 112-valid 4 | 33 clauses; all statuses match the scorecard | OK |
| 85 | Registration timing (job 112) | header committed and pushed before each rental | commit 92849de 05:41:41Z, push 05:41:43Z; rentals 112a/b/c/d 71–77 s after the push; 112g 4,365 s after; job starts 123–153 s after the push (112g 5,257 s) | OK |
| 86 | wsg_regtime macros | 92 rentals; commit ≥2.7 s before; start ≥26 s after push; 7 within 1 s; 4 pushed after (max 0.8 s) | 92; 2.71; 26; 7; 4 (all job 108; max 0.835) | OK |
| 87 | app_prereg, job 112 ¶ | 109d's machine never finished downloading in two hours and was destroyed without results | 112c rental 7,279 s; no results/112c | OK |
| 88 | app_prereg, job 112 ¶ | "its gates passed" | no artefact (results or git) records it | NOTE N11 |
| 89 | app_prereg, job 112 ¶ | 109f's machine started while the download ran, once two rounds no longer fit | 112g created 71.5 min after 112c; ≈30 min of 112c's 105-min budget left versus ≈36 min for lookahead plus two rounds (109d timings) | OK (plausible) |
| 90 | app_prereg, job 112 ¶ | misses 0.97–0.98 (≤0.95 registered); slowed up to 0.05 (registered 0.01); T1 failed 5/8, T2 7/8; T3 and T4 held | as #32, #84 | OK |
| 91 | app_prereg, job 112 ¶ | first probe during the download on some machines, after it on others | probe1_when: download running yes (112a, 112b), no (112d, 112g) | OK |
| 92 | app_prereg, job 112 ¶ | relaunch time within 0.7%, ratio up to 12% | 0.71%, 11.8% | OK |
| 93 | app_prereg, job 109 severity note | bands 0.25–0.55 (panel 33%), 0.98–1.06 (Margin 1.020×), capture ≤0.35 | header P3/P10/P7; rbDkMean 1.0195 | OK; NOTE N13 |
| 94 | app_prereg, jobs 109–111 ¶ | five started, all five valid; one offer skipped; 13900KF offer rented twice (099d, 100e); 110: five started, one gated at 0.2499; 111: two failed downloads | ledger and results agree | OK |
| 95 | app_prereg, tallies | 565 = 218 / 193 / 142 / 8 / 4; eras 7/19, 19/23 (11 sign), 24/38, 168/473 | recomputed from tab_prereg rows and scorecard_clauses.json | OK |
| 96 | app_prereg, supplement list | "\cxClauses{} for jobs 110 and 111" = 38 | job 111 has 38; job 110 has 2 more (tab_scorecard_110) | **DEFECT D9** |
| 97 | tab_prereg, all 43 rows and totals | 588 / 679 / 321 / 84 | every row equals the JSON tallies (096 = 096 + O4 + O5; 097 = O4 + O6); totals equal | OK |
| 98 | tab_failures, 099–111 tallies | 099 190/42/39/2; 100 41/122/27/26; 101 6/19/0/0; 102 0/28/5/0; 103 4 pt/41 unt; 104 1/22/24; 105 7/20/4; 106 21/112/48; 107 11/22/13; 108 13 pt/6/2 unt; 109 47/44/0; 111 24/14/0 | all equal to the scorecard JSONs | OK |
| 99 | tab_failures, hand-scored rows | 085b 1 of 3; 088 1.51–1.58; 089 ±0.06 at five cells 0.06–0.11; ±12% 14–17%; LRU 6–8%; LRU behind; 091/092 1.56–1.78, FETCH tables, Mixtral tie; 093 43.7–44.8%, 3/2 cells, 0%, 15%, 48/56%; 095 13–17%, 17%, 44/51%, 7/13%, 12–13%; 104 0.73×, 5.3%, 0.047; 105 6.9–9.3% on 3, 4.2%; 106 0.030 / 0.045 | all agree with scorecard_clauses.json and scorecard_10x.json | OK |
| 100 | tab_failures, 091/092 row | launch orders "7% and 11%" | only 092-P5a (7.0%) is a scored clause; the gpt-oss 11% −11% was "reported only" | NOTE (part of **D11**) |
| 101 | tab_failures, job 112 | (no line) | 12 failed clauses (T1 5, T2 7) | **DEFECT D2** |
| 102 | tab_failures, header comment | "Measured values are macros generated by the scoring scripts" | most hand-scored measured values are typed | **DEFECT D11** |
| 103 | wsg_dm t-intervals | all 32 tab:gap cells, plus dmNewInteractionPos, dmRegHeld | reproduced exactly as mean ± t·s/√n | OK |
| 104 | wsg_robust t-intervals | rbPlanMean/Lo/Hi 1.21 [1.11, 1.33]; rbDkMean/Lo/Hi 1.020 [1.004, 1.035]; rbDkLow 1.021 [1.000, 1.043]; rbDkMid 1.018 [0.987, 1.049]; rbPlanAllLo/Hi 1.06/1.28 | 1.2147 [1.1104, 1.3288]; 1.0195 [1.0042, 1.0351]; 1.0215 [1.0001, 1.0433]; 1.0176 [0.9869, 1.0493]; 1.0631/1.2756 (log-scale geometric t-intervals) | OK |
| 105 | wsg_robust | rbPlanAllMean 1.17 | 1.1645 (its own Lo/Hi centre ≈1.165); macro unused in text | NOTE N14 |
| 106 | wsg_robust | rbErr*, rbErrFound* | not recomputed (closed-form law per launch) | NOTE |
| 107 | Statistics claim | panel machines rented again within 3.2% / 5.5% | see #23 | OK |
| 108 | Job 112 narrative vs results | no results/112c; 112c in ledger; 112g a replacement | consistent | OK |
| 109 | Fig. hostdep caption | "the registered second-card test" | both tests' RTX 4090s are plotted (fig_hostdep.py reads job 111 and job 112) | NOTE N15 |

**Totals:** 109 items checked (many rows bundle several numbers), 11 defects.

## Defects, by severity

**D1. The abstract's limit on policies without foresight lost its qualifier (moderate–high).**
- Text: "Policies without foresight recover at most \olCapLowMax\% of the oracles' gain." (renders as 6%)
- Correct value: 6% (5.7%) holds only on job 109's fast-link RTX 5090s. On the slow-link RTX 4090s of job 111, Margin recovers 25.7, 19.6 and 20.5% at 11% (\cxCapLowMax = 26). This revision dropped the earlier qualifier "there".
- Fix: "Policies without foresight recover at most \olCapLowMax\% of the oracles' gain on fast-link machines." The intro, Table 1 and rule 3 are already qualified.

**D2. tab_failures has no line for job 112 (moderate).**
- Text: app_prereg says the table "gives … the families of failures of jobs 099–112", and the table caption says "from job 099 on, each job's line gives its tally". The table ends at job 111.
- Correct value: job 112 has 58 clauses: 22 held, 24 held (point), 12 failed. The failures are T1 on 5 of 8 machine-budgets (misses 0.97–0.98 of Few-2R's at 11%, against ≤0.95) and T2 on 7 of 8 (slower by up to 0.05, against −0.01).
- Fix: add `\tfjob{112}{… \tcClauses{} clauses; \tcClausesHeld{} held, \tcClausesPoint{} held (point), \tcClausesFailed{} failed.}` with the two rows: T1 (≤0.95×, measured \tcMissRatioLowMin–\tcMissRatioLowMax at 11%; the fetch table's in-step fetches) and T2 (≥ −0.01, slowed by up to \tcSlowLowMax).

**D3. "What is left": the RTX 4090 ranges cover job 111 only (moderate).**
- Text: "…and \cxResidLowMin--\cxResidLowMax\% on the RTX 4090s of \cref{sec:secondcard}" (31–36%) and "\dmEqThreeLeftCardLowMin--\dmEqThreeLeftCardLowMax\% on the RTX 4090s" (73–81%).
- Correct value: both ranges come from job 111's three slow-link machines only. Section 6.2's RTX 4090s now include job 112's two fast-link machines, where the residual is 5.4% (i5-12400) and 14.4% (7800X3D), and the share of the Eq. (3) gap left is 41.7% and 57.4%.
- Fix: "on the three slow-link RTX 4090s of \cref{sec:secondcard} (job 111)" in both places.

**D4. Section 6.2: "the RTX 4090s' ratios" now includes job 112 (moderate).**
- Text: "At the RTX 4090s' ratios \MinOne{} runs \cxFetchLowMin--\cxFetchLowMax$\times$ the deployed cache at 11%" (0.96–1.01×).
- Correct value: this is job 111 only (ratios 0.37–0.40). On job 112's RTX 4090s (0.57, 0.77) MIN-1R runs 1.13× and 1.31× (\tcFourFetchLowRng).
- Fix: "At the slow-link RTX 4090s' ratios (\cxRatioMin–\cxRatioMax) …".

**D5. Section 7's new yardstick sentence miscounts the machines below 1 (moderate).**
- Text: "runs \jaWSixteenLowMin--\jaWSixteenLowMax$\times$ at 11\% on \jaHosts{} machines, below 1 on a slow-link one: admitting every miss is a weak base on slow links."
- Correct value: w16/base is below 1 on two machines: 100a (ratio 0.41, 0.891×) and 100f (ratio 0.54, 0.892×). By the paper's own definition 100f is fast-linked.
- Fix: "below 1 on the two with the slowest links (ratios 0.41 and 0.54)". Soften or drop "on slow links".

**D6. The fewest-admission claim excludes job 112's machines (moderate).**
- Text (Section 5): "it beat the deployed cache on all \rbPlanMachines{} machines that ran it stably, by a geometric mean of \rbPlanMean$\times$" (9 machines, 1.21×).
- Text (Table 1, row 8): "…gains on every machine that ran it stably", Mach. 9.
- Correct value: rbPlan* uses jobs 104–107 only. Job 112 ran Few-1R (fetchplan) at gpt-oss 11% on four more machines that passed every check: 1.35, 1.21, 1.25 and 1.41×. Pooled, that is 13 machines at ≈1.24× [1.16, 1.32] (95% geometric t-interval, from my per-launch values).
- Fix: add job 112 to robustness.py's Few-1R set, or scope the claim to "in jobs 104–107".

**D7. Section 4: the early copies did not leave the two-read arm "where it was" (low–moderate).**
- Text: "Issuing each copy of MIN's fewest-admission set one step earlier, so that it lands for the next use, leaves the two-read arm where it was."
- Correct value: Few-2R-early is slower than Few-2R by 0.031–0.053 of the deployed cache's speed at 11% on all four machines. Every paired interval lies below −0.01, and the registered T2 failed on 7 of 8. The appendix says "slowed the arm by up to \tcSlowLowMax".
- Fix: "…makes the two-read arm slightly slower (by \tcSlowLowRng{} of the deployed cache's speed)".

**D8. Setting: the measured-rate sentence omits Qwen3 25% on host B (low).**
- Text: "At the GPU's measured batch-1 rate the GPU term also binds at gpt-oss 25\% on both hosts of \cref{tab:headline}."
- Correct value: tab:headline also shows the GPU term binding at Qwen3 25% on host B (bound 214 → 192 tok/s). That is one of Qwen3's two "host-bound" budgets.
- Fix: "…at gpt-oss 25% on both hosts and at Qwen3 25% on host B".

**D9. app_prereg: \cxClauses is job 111's count only (low).**
- Text: "\cxClauses{} for jobs 110 and 111" (38).
- Correct value: 38 is job 111's count. Job 110 has 2 more (tab_scorecard_110: 2 clauses), so the two jobs total 40.
- Fix: "\cxClauses{} for job 111 and 2 for job 110".

**D10. Typed measured counts in changed text (low).**
- Table 1 caption: "``15${}+{}$5''". Use `\dmMachines${}+{}$\olN`.
- Setting/Statistics: "with 3 to 15 machines" (also repeated in app_wsg.tex, line 27). t-intervals over machines are also reported over 21 machines (\rbErrFoundMachines) and 25 machines (\rbErrMachines) in app_more and app_relation; the smallest n is 4. Use macros, or "4 to 25".

**D11. tab_failures types its measured values despite its source comment (low).**
- Text: the source comment says "Measured values are macros generated by the scoring scripts".
- Typed measured values in the hand-scored block include: "1 of 3", "1.56--1.78×", "7% and 11%", "43.7--44.8%", "48 and 56%", "15%", "13--17%", "17%", "44 and 51%", "7 and 13%", "12--13%", "0.047 behind" and "off by 0.030".
- Correct value: every one I spot-checked agrees with the scorecard JSONs. However, in "7% and 11%", only 7% is a scored clause (092-P5a, Qwen3 12.5%). The gpt-oss 11% −11% was "reported only".
- Fix: generate these from macros or correct the comment. Mark the 11% as reported, not scored.

## Notes (not defects)

- **N1.** tab:timing caption "within ±0.01": the largest half-width is 0.012 (it rounds to 0.01). Consider "±0.012".
- **N2.** The tab:job109 subscripts at the rounding boundary are bootstrap noise, not errors:
  - i5-12400, Ahead at 11%: half-width 0.0144–0.0151 across six seeds (printed ±.01);
  - Core Ultra 9 285K (0.39), MIN-1R at 11%: 0.0049–0.0052 (printed ±.00).
- **N3.** "(``1.22×'')" in Yardsticks is a typed illustrative value.
- **N4.** Rule 2's "about a third" is typed. Fast-link panel 39% [31, 47]; new machines 35%.
- **N5.** The Statistics paragraph says summaries over machines carry t-intervals, but the Spearman interval in 6.2 ([0.63, 0.97] over 19 machines) is a percentile bootstrap over machines (reanalysis.spearman_ci). State the exception for correlations.
- **N6.** The Limitations "ratio moved by up to 12% between two rentals" covers only job 112's two RTX 5090 relaunches. Other re-rentals: 110d→111d 10.0%, 101b→102c 10.2%. GPU-eecf00ca read 0.98 (072), 0.59 (085b) and 0.67 (099c), probably with older probe versions. Scope the sentence to "the two relaunched machines".
- **N7.** Table 1 row 4 shows "held on 9 of 10" for a clause registered as positive "on every host". At 11% it failed on 099f (ratio 0.29). Correct given the "fast links" qualifier, but it reads as a hold.
- **N8.** The 6.2 footnote's "these machines" follows the job 112 sentence but refers to job 111's machines.
- **N9.** "registered … within 0.10 of it in log" for all four configurations. The registration had 0.06 for Dep-1R and MIN-2R, and "all but one" at 25%.
- **N10.** \tcStarted/\tcStartedWord = 4 counts result directories, but five machines were rented (112c). Unused in the text.
- **N11.** "its gates passed" (112c) is not supported by any artefact. Replacing a machine that did not fail a gate is outside the header's replacement rule; the Limitations acknowledge it.
- **N12.** tab_names: "hitopt, hitoptp: Belady-2R (paced copies)". Only hitoptp is paced.
- **N13.** The severity note says the capture bound "could hardly have failed" but the "per-machine bounds on the variants without foresight" could. The capture bound is itself per machine. Reword.
- **N14.** rbPlanAllMean is 1.17 against my 1.1645; its own Lo/Hi (1.06/1.28) centre at ≈1.165. The macro is unused in the text.
- **N15.** Fig. hostdep caption: "the registered second-card test" should be plural (job 111 and job 112 machines are plotted). tab_failures' job 109 line still says "four online policies", against the renamed "policies without foresight".
- **N16.** The moved appendix text (app_foresight.tex, app_extra.tex) contains many typed measured numbers (e.g. "+3.6%, +4.0%", "G_fit = 4.3 ms", "45.7%", "98–99%", "29–47%", "1.24–1.38×", "4.93 (5.20) ms"). This is outside the listed scope and was not checked.
- **N17.** Not recomputed (unchanged macros from other scripts): dcShare*, dcMed, dcAime*, aud*, jd*, fs*, sl*, vm*, pnShare*, jk*, rc*, lf*, la*, kl*, global*, rs*, rbErr*.
