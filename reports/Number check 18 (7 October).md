# Number check 18 (7 October)

Scope: every number and claim in the parts of the main text changed since `c369d01` (abstract, introduction, Table 1,
Contributions, Setting, Sections 3–7 changed sentences, Limitations, Conclusion), Tables 4 (`tab_dm`, both columns) and
5 (`tab_job109`), `tab_configs`, `tab_names`, the appendix paragraph on jobs 109–111, and the `app_traces` addition.
I did not open review files, `prereg/*outcome*.md`, or other reports. Nothing in either repository was modified. My
scripts and outputs are in `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/nc18/`.

## Method

* **Raw rows.** My own loader (`load.py`, `compute.py`) reads `ec_g_C{14,32}_r{1,2}{A,B}.jsonl` for 109a,c,d,e,f and
  111d,f,g. It labels each configuration by the `stats=` path, takes the mean of `decode_ms/n_decode` over the 20 problems,
  and computes X/base = base/X within the same process (A for foa, bypass, fetch, both3p, dk, pf; B for R1, R2), then the
  geometric mean over rounds. B_host is the highest `cpu_read_gbs`, `pcie_*_gbs` or `sum` reading in `concur.txt`.
  Eq. (1) is R* S / B_host with the registered R* and S. The gap is base − Eq. (1), using process A's base as the mean over
  rounds. Shares, the interaction (ms), the residual (with 2.94 ms), capture and reads follow the job 109 header. Reads
  come from `st_*.json`: (misses + admits + prefetches)/steps, mean over rounds. Ratio = B_p/B_c from
  `fetch_table_law_gptoss.json`.
* **Intervals.** I used a percentile bootstrap over machines (20,000 resamples, my own seed), the same method as
  `scripts/decomp_measured.py`.
* **Trend.** I applied job 110's frozen coefficients, copied from its header, to the 111 machines and the 109 machines.
  I also refitted the trend from `prereg/reanalysis_hosts.json` to confirm the frozen coefficients.
* **Panel.** I spot-checked `prereg/reanalysis_hosts.json` against raw `ec_g_C14/C32.jsonl` and `concur.txt` of 096a,
  099d, 099f, 099i, 100f and 101b: 72 values (5 states and Eq. (1) at 2 budgets on 6 machines). I then recomputed the
  whole Panel column from that JSON, with Eq. (1) recomputed from the raw probes and the read-ahead oracle's reads from
  the raw counters.
* **Registration.** I compared commit times (`git log`) with rental creation times in `gpu/vast_ledger.json`. I fetched
  push times with `gh api repos/.../activity?ref=refs/heads/gpu`, because `prereg/gpu_pushes.json` ends before job 109.
  I compared GPU UUIDs in `v0.txt` with `jobs/ec2/job110_uuids.txt` and `known_gpu_uuids.txt`.
* **Build.** I ran `git archive HEAD paper` into a clean folder and then `latexmk -pdf` (and `-f` after it failed). I read
  `paper.log` and extracted the text with pdftotext. I also rendered pages 6 and 8.

## Registration timing (UTC; the push comes from GitHub's activity API, the rental from the ledger's `start`)

| Commit | What | Committed | Pushed | First rental after it | Lag after push |
|---|---|---|---|---|---|
| 02ff39d | job 109 header and wrappers a–j, s | 22:14:01 | 22:14:03 | 109s 22:14:26.9 (hosts a–f 22:35:16–26, after the smoke run ended 22:28:22) | 23.9 s |
| bb95f58 | smoke host offer | 22:14:19 | 22:14:21 | 109s 22:14:26.9 | 5.9 s |
| ca719f2 | job 110 header (frozen trend, Q1–Q6) | 22:39:06 | 22:39:08 | 110a 22:39:15.3 (110e 22:54:15) | 7.3 s |
| 443bcba | job 111 (V1 = 0.1963), wrappers b–e | 23:52:00 | 23:52:03 | 111b 23:52:10.6 | 7.6 s |
| 58d5454 | job 111 amendment, wrappers f, g | 00:10:51 | 00:10:54 | 111f 00:11:02.0 | 7.1 s |

Every prediction was committed and pushed before the machine it concerns was rented. The smoke run finished before any
job 109 host started, as its header requires. I refitted the frozen trend from the 15 panel machines and it reproduces
all eight lines of job 110's header exactly. The largest residuals in the refit are 0.081 (11%) and 0.094 (25%), also as
stated.

## Checked items

Verdicts: ✓ matches; ≈ matches within rounding or bootstrap noise; ✗ defect (numbered in the next section).

| # | Location | Paper | Mine (raw) | Verdict |
|---|---|---|---|---|
| 1 | Abstract: "31–54% of that bound's speed" | 31–54, no machine scope | 31–54 holds on the 25 consumer machines only; valid server machines 12.4% (eng. sample 107d), 26.7% (EPYC 7543 108a), 46.1% (EPYC 7402P 105a) | ✗ (D4) |
| 2 | Abstract: "close almost none of the gap alone" (new machines) | — | set alone 3.4% [0.5, 6.0], one read alone −0.4% [−1.5, 0.8] | ✓ |
| 3 | Abstract: "35% together" | 35 | 34.71 | ✓ (scope: D6) |
| 4 | Abstract: "registered before they ran" | — | pushes 7–24 s before every rental | ✓ |
| 5 | Abstract: online policies "at most 6%" | 6 | max capture 5.68% (109c, R2) | ✓ |
| 6 | Abstract: trend "predicted the oracles' gains … within 6%" | 6 | speed-ratio error ≤ 5.94%; gains themselves off by up to ~50% relative | ✗ (D2) |
| 7 | Intro: "runs at 31–54%" | 31–54 | as #1 | ✗ (D4) |
| 8 | Intro: "33% of the gap on 15 panel machines" | 33 | 32.64 (15 machines) | ✓ |
| 9 | Intro: "35% on 5 machines rented later" | 35, 5 | 34.71, 5 | ✓ |
| 10 | Intro: "What the oracles leave is mostly the GPU's own work" | — | T_GPU/left: panel 11% median 0.54, < 0.5 on 6/15; RTX 4090s 0.32–0.33 | ✗ (D7) |
| 11 | Intro: online "at most 6% … on those machines" | 6 | 5.68 | ✓ |
| 12 | Intro: "they still read as many experts as the deployed cache" | — | R1 1.665–1.667 R*, R2 1.675–1.678, base 1.645–1.660 (dk 4.3% fewer than base) | ✓ |
| 13 | Intro: trend "within 6%" (gains) | 6 | as #6 | ✗ (D2) |
| 14 | Intro: next token "at most a third" | 0.31 | max of 0.18, 0.07, 0.31, 0.16 is 0.31 | ✓ |
| 15 | Table 1 row 4: sign held on 9 of 10 panel machines | 9/10 | interaction (ms) negative only on 099f among 099a–j; 14/15 on the full panel | ✓ |
| 16 | Table 1 row 4: 5 of 5 new machines; machines 15+5 | 5/5 | 1.53, 2.97, 2.90, 3.55, 7.49 ms, all > 0 (25%: 4/4) | ✓ |
| 17 | Table 1 row 5: ≤35% registered; at most 6%; 5 machines | — | P7 as registered; 5.68%; 5 | ✓ |
| 18 | Table 1 row 6: "predicts the oracles' gains … within 6%"; 19+3 | — | see #6; 19 is the Spearman set and 15 the frozen fit | ✗ (D2, D25) |
| 19 | Contributions: 15 machines, replicated on 5 | 15, 5 | 15, 5 | ✓ |
| 20 | Contributions: "a rule that carried over to a second card" | — | fewest-admission and 2-read paths not run on the RTX 4090s | ✗ (D8) |
| 21 | Setting: host-bound "except at gpt-oss 25% on … workstation and server processors" | — | GPU term binds at 25% when B_host > ~107 GB/s: Threadripper 9960X (classed consumer), EPYC 7302/7663/9754, Xeon 8347C, eng. sample; EPYC 7543/7402P/7352/7K62 stay host-bound | ✗ (D11) |
| 22 | Setting: B_c at the helper count, B_p the engine's copy rate, B_host the highest reading | — | `fetch_table.py`: B_c interpolated at helpers, B_p the zero-copy rate | ✓ |
| 23 | Statistics: "jobs 104 and 105 ran one process per configuration" | — | each ran one process per budget containing all configurations (`ec_g_C14.jsonl`) | ✗ minor (D15) |
| 24 | Statistics: "a machine rented again came within 3.2%" | 3.2 | RTX 4090 relaunches: 110c→111g +5.5%, 110d→111d +4.1%, 110b→111f +0.05% | ✗ minor (D15) |
| 25 | Sec. 3: leads llama.cpp "at all of them" | — | \bothLlamaXMin = 2.0 > 1 | ✓ |
| 26 | Sec. 3: "binds only at the largest budgets" | — | contradicts #21 | ✗ (D11) |
| 27 | Sec. 4 ¶1: 15 panel machines, 5 rented, registered before they started | — | ✓ (timing table) | ✓ |
| 28 | Sec. 4: MIN's set alone at 11% (panel) | 0 | 0.45 | ✓ |
| 29 | Sec. 4: one read alone at 11% (panel) | 0 | −0.05 | ✓ |
| 30 | Sec. 4: together, panel / new | 33 / 35 | 32.64 / 34.71 | ✓ |
| 31 | Sec. 4: interaction registered on 10, held on 9; new 5 of 5 | 10, 9, 5/5 | 10, 9, 5/5 | ✓ |
| 32 | Sec. 4: MIN's set alone at 25% | 21 | 20.59 | ✓ |
| 33 | Sec. 4: read-ahead oracle "another 15%"; "52% remains" | 15, 52 | 14.91, 52.45 | ✓ |
| 34 | Sec. 4 "What is left": two named parts account for it (panel) | — | 27 + 21 of 52, residual 4 [−4, 13] | ✓ |
| 35–42 | Table 4 Panel 11% (8 rows) | 0[−2,2]; 0[−2,1]; 33[22,42]; 15[11,18]; 52[45,62]; 27[25,30]; 21[20,23]; 4[−4,13] | 0.45[−1.6,2.5]; −0.05[−1.8,1.5]; 32.6[21.4,42.0]; 14.9[11.1,18.4]; 52.5[44.6,62.0]; 27.4[24.8,29.8]; 21.4[19.9,23.1]; 3.7[−3.8,12.8] | ≈ (means exact; one endpoint ±1, bootstrap noise) |
| 43–50 | Table 4 Panel 25% (8 rows) | 21[19,22]; −3[−7,0]; 31[24,38]; 21[18,22]; 48[42,55]; 34[31,36]; 19[18,20]; −5[−11,2] | 20.6[19.0,22.1]; −3.2[−7.1,0.2]; 31.3[23.8,37.9]; 20.6[18.4,22.5]; 48.1[42.3,54.7]; 34.0[31.2,36.3]; 19.1[17.7,20.5]; −5.0[−11.1,1.9] | ≈ (means exact; endpoints ±1) |
| 51–58 | Table 4 New 11% (8 rows, n = 5) | 3[0,6]; 0[−2,1]; 35[28,44]; 19[16,22]; 46[40,52]; 28[23,33]; 18[17,19]; 1[−2,3] | 3.40[0.47,6.00]; −0.38[−1.51,0.75]; 34.71[27.6,43.8]; 19.02[15.5,22.0]; 46.26[40.3,51.6]; 28.21[23.0,33.3]; 17.53[16.6,19.2]; 0.52[−2.0,2.6] | ✓ |
| 59–66 | Table 4 New 25% (8 rows, n = 4) | 20[19,20]; −2[−3,−2]; 27[25,30]; 24[23,26]; 48[46,51]; 38[35,41]; 16[15,16]; −5[−7,−4] | 19.78[19.1,20.5]; −2.42[−2.85,−1.89]; 27.26[25.0,29.6]; 24.39[23.1,26.2]; 48.35[46.5,50.9]; 37.86[35.4,40.9]; 15.62[15.1,16.2]; −5.14[−6.5,−4.1] | ✓ |
| 67 | Table 4 caption: 15 / 5 / 4 machines; deadline cut one 25% round; 2.9 ms | — | 109f's C32_r1A timed out (rc 124, pf 18 rows) and C32_r1B was skipped | ✓ |
| 68 | Panel spot-check: 72 raw values (096a, 099d, 099f, 099i, 100f, 101b) | JSON | identical to 3 decimals, Eq. (1) included | ✓ |
| 69 | Panel spot-check: interaction share mean | 32 [24, 39] | 32.2 [24.2, 39.3] | ✓ |
| 70 | Sec. 5 "Admit as rarely": "for the same hits" | — | holds only in replay; engine misses 2.5–5.3% more (Limitations) | ✗ minor (D17) |
| 71 | Sec. 5 second card: frozen trend, one line per configuration | — | coefficients reproduced exactly from 15 panel machines | ✓ |
| 72 | Sec. 5: "before any of them ran" | — | ca719f2 pushed 22:39:08, first RTX 4090 rental 22:39:15 | ✓ |
| 73 | Sec. 5: three passed every check, ratios 0.37–0.40 | 3; 0.37–0.40 | 111d 0.3707, 111g 0.3917, 111f 0.3957; V0–V3 rechecked (V2 ≤ 0.30%, V1 ≤ 0.03% off 0.1963) | ✓ |
| 74 | Sec. 5: within 6% (11%) and 5% (25%) | 6, 5 | max |obs/pred−1|: 5.94% (111g read-ahead), 4.66% (111d MIN 1 read; 4.9% in ln) | ✓ |
| 75 | Sec. 5: MIN 1 read 0.96–1.01× at 11% | 0.96–1.01 | 0.963, 1.005, 1.007 | ✓ |
| 76 | Sec. 5: fewest-admission "pays instead … (1.11–1.22× on the slow-link RTX 5090s)" | 1.11–1.22 | 104a/105b/105e (ratios 0.28–0.32): 11% 1.113/1.114/1.112, 25% 1.217/1.159/1.215; pooled over budgets | ✗ (D10) |
| 77 | Sec. 5: RTX 4090 loss 3% higher in every configuration | 3 | per configuration 2.4–3.8% (ratio of means 3.1–3.6%); base 0.1963–0.1965 vs 0.1900–0.1908 | ✓ |
| 78 | Sec. 5: "first launch stopped after one round" | — | 110b–e after one round (V1); 110a at the ratio gate before any round | ✗ minor (D24) |
| 79 | Sec. 5 online: admitting less cuts reads 4–6% at 11% | 4–6 (job 106) | job 109: 4.3–4.4% | ✓ |
| 80 | Sec. 5 online: five machines | five | five | ✓ |
| 81 | Sec. 5 online: best 1.007–1.024× at 11% | 1.007–1.024 | 1.00689 (dk) to 1.02364 (R2) | ✓ |
| 82 | Sec. 5 online: at most 6% of the read-ahead oracle's gain; registered ≤ 35% | 6 | 5.68 | ✓ |
| 83 | Sec. 5 online: "The online policies still read 1.66–1.67 R*" | 1.66–1.67 | R1 1.6648–1.6671; R2 1.6747–1.6777 | ✗ minor (D12) |
| 84 | Sec. 5 online: RTX 4090 layer-ahead "policies … 0.64–0.68× at 11%" | 0.64–0.68 | R1 0.644–0.681; pf 0.682–0.709; R2 0.691–0.724 | ✗ (D9) |
| 85 | Sec. 5 online: admitting less best, 1.033–1.041× | 1.033–1.041 | dk 1.0329, 1.0385, 1.0406 (best on all three) | ✓ |
| 86 | Sec. 5 online: "up to 26%" | 26 | 25.71 (111d) | ✓ |
| 87 | Fig. 4: hollow markers are job 111's valid machines, not in fits | — | `fig_hostdep.py`: job111.json, valid only, plotted after the fits | ✓ |
| 88–92 | Table 5, RTX 5090 rows (ratio + 8 cells each; 109e, d, f, a, c) | see table | 0.54 1.215 1.460 1.021 4.5% 1.241 1.640 1.027 4.2% / 0.59 1.247 1.458 1.007 1.5% 1.256 1.612 1.010 1.7% / 0.84 1.366 1.500 1.010 1.9% – / 0.88 1.154 1.342 1.014 4.1% 1.216 1.546 1.011 2.1% / 0.89 1.241 1.416 1.024 5.7% 1.283 1.626 0.995 −0.8% | ✓ (all 41 numbers) |
| 93–95 | Table 5, RTX 4090 rows (ratio, 8 cells, 4 bracketed predictions each) | see table | 111d 0.37 0.963[0.989] 1.128[1.111] 1.033 25.7% 1.020[1.069] 1.305[1.292] 1.039 12.9%; 111g 0.39 1.005[1.007] 1.196[1.129] 1.039 19.6% 1.057[1.086] 1.364[1.314] 1.043 11.7%; 111f 0.40 1.007[1.010] 1.198[1.133] 1.041 20.5% 1.060[1.089] 1.366[1.318] 1.045 12.4% | ✓ (all 39 numbers) |
| 96 | Table 5 caption: "the two built from both" | — | R1 = 1 read + layer-ahead copy (κ = 1); only R2 uses both mechanisms | ✗ minor (D16) |
| 97 | Sec. 6: next token at most 0.31; half of gain at 11% needs 4 tokens | 0.31, 4 | macros consistent (max over 4 budgets) | ✓ |
| 98 | Sec. 6: 0.66–0.81 C on AIME | 0.66–0.81 | min/max of \dcAime{Low,Mid,Qlow,Qmid} = 0.66–0.81 | ✓ |
| 99 | Sec. 6: speculative "about a tenth … cost reads at 80%" | — | \bkSameNine 8–12%, \bkSameEightLoss 18–20% | ✓ |
| 100 | Sec. 6: layer-ahead online "at most 6%" | 6 | best layer-ahead policy (pf/R1/R2) capture 5.7% (109c R2); negative on RTX 4090s | ✓ |
| 101 | Rule 1: "on our machines … 31–54%" | — | as #1 | ✗ (D4) |
| 102 | Rule 2: "a better set read once closes about a third of it" (unscoped) | ~33 | fast links 33–35%; slow links: 111d −6.4%, 111f 1.1%, 111g 0.8%, panel 099d 8.1%, 099f −23.2% | ✗ (D6) |
| 103 | Rule 3: at most 6%; layer-ahead loses on slow links | 6 | 5.68; RTX 4090 pf/R1/R2 0.64–0.72 | ✓ |
| 104 | Rule 4: "on any card … where it is slow, serve them on the CPU and copy in the background" | — | not tested on RTX 4090s; there MIN 2 reads 0.98–1.01× at 11%, read-ahead oracle 1.13–1.20× | ✗ (D8) |
| 105 | Conclusion: 31–54% | — | as #1 | ✗ (D4) |
| 106 | Conclusion: "on either card we tested" | — | RTX 4090s cover ratios 0.37–0.40 only; trend misses new RTX 5090s by up to 18% | ✗ (D3, D8) |
| 107 | Conclusion: "mostly the GPU's own work" | — | as #10 | ✗ (D7) |
| 108 | Limitations: "deviated from its registration once" | once | V1 replaced; 111b/c relaunch amendment; 110e's machine not relaunched (UUID missing from V0 list) | ✗ (D5) |
| 109 | Limitations: "in the two tests on new machines …" | 3/2 gated, 2/1 unsteady | jobs 109 (0 gated, 0 unsteady of 5) and 110 (1 of 5 gated) are also tests on new machines | ✗ (D13) |
| 110 | P1 (fetch ≥ max(bypass, foa) + 0.04 at 11%; interaction > 0 at both budgets) | held | margins 0.075–0.318; interaction > 0 on 5/5 and 4/4 | ✓ |
| 111 | P2 (foa, bypass in [0.96, 1.05] at 11%) | held | foa 0.989–1.007; bypass 0.991–1.039 | ✓ |
| 112 | P3 (together mean in [0.25, 0.55] / [0.20, 0.50]) | held | 0.347 / 0.273 | ✓ |
| 113 | P4 (both3p > fetch: 25% on every machine, 11% on all but one) | held | 4/4 and 5/5 | ✓ |
| 114 | P5 (bypass ≥ 1.08 at 25%) | held | 1.159–1.179 | ✓ |
| 115 | P6 (residual mean within ±0.10 / ±0.15) | held | +0.005 / −0.051 | ✓ |
| 116 | P7 (best ≤ 1.10, capture ≤ 0.35 per machine; mean capture at 11% ≤ 0.25) | held | best ≤ 1.027; capture ≤ 0.057; mean 0.035 | ✓ |
| 117 | P8 (pf < 1.00 for ratio < 0.75; ≥ 0.98 for ratio ≥ 0.9) | held | 109d 0.968, 109e 0.852; no machine at ≥ 0.9 (max 0.889), so the second clause is untested | ≈ (D14) |
| 118 | P9 (\|R1 − pf\| ≤ 0.04 at 11%) | held | 0.0006–0.0103 | ✓ |
| 119 | P10 (dk in [0.98, 1.06]; R2/R1 in [0.97, 1.06]) | held | dk 0.994–1.027; R2/R1 1.010–1.030 | ✓ |
| 120 | P11 (R1, R2 ≥ 1.3 R*) | held | ≥ 1.665 (11%), ≥ 2.195 (25%) | ✓ |
| 121 | \olPredHeld/\olPredN = 11/11; clauses 91 = 43 + 48 + 0 | 11/11 | 11/11 (P8 partial); scorecard tally 43 + 48 = 91 | ≈ (D14) |
| 122 | Q1 on job 111 (fetch, both3p ≤ 0.10 at 11%; at 25% all but one; foa, bypass ≤ 0.06 at 11%) | held | max 0.058 / 0.048 / 0.034 (ln) | ✓ |
| 123 | Q2 (pf, R1 < 1.00 at 11%, ratio < 0.75) | held | pf 0.682–0.709, R1 0.644–0.681 | ✓ |
| 124 | Q3 (best online ≤ 1.10) | held | ≤ 1.045 | ✓ |
| 125 | Q4 (base/Eq.1 in [1.8, 3.5] / [2.5, 6.5]) | held | 2.488–2.539 / 4.106–4.121 (scorecard 4.117 vs my 4.118 for 111d: rounding) | ✓ |
| 126 | Q5 (R1, R2 ≥ 1.3 R*) | held | 1.671–1.687 / 2.204–2.228 | ✓ |
| 127 | Q6 (interaction > 0 on ratio ≥ 0.5 hosts) | not mentioned | no such host: untested, not in the scorecard or the text | ≈ (D14) |
| 128 | \scDevLowMax 6, \scDevMidMax 5, medians 2 / 3 | 6, 5, 2, 3 | 5.94, 4.9, 2.1, 3.1 | ✓ |
| 129 | \scCapLow 20–26, \scCapMid 12–13, \scBestMid 1.039–1.045, \scPfLow 0.68–0.71, \scFetchMid 1.02–1.06, \scOracle 1.13–1.20 / 1.30–1.37 | — | identical | ✓ |
| 130 | \olCap{Low,Mid}{Min,Mean,Max}; \olBestMid; \olDk, \olPf, \olRone, \olRtwo, \olFetch, \olOracle ranges | — | identical at the printed precision | ✓ |
| 131 | \dmNew* (Max/Min, Pos, PrefReads 1.22–1.24 / 1.42–1.43, Closed 54 / 52, interaction 32 [24, 41], 10 [8, 12]) | — | identical | ✓ |
| 132 | \scFirstDevLowMax 6 (job 110 first rounds, unused in the text) | 6 | 6.2 | ✓ |
| 133 | Appendix: 51325952 rented for 099d (first) and 100e, no result; 109d's GPU never measured | — | ledger: 099d (id 54448691, replaced by offer 51748728) and 100e (54479314); 109d UUID GPU-1f8fba46… not in `known_gpu_uuids.txt` | ✓ |
| 134 | Appendix: one listed offer skipped (109b) | — | no 109b rental in the ledger | ✓ |
| 135 | Appendix: job 110 five started, one ratio-gated at 0.2499, others failed V1 after one round | — | 110a 0.2499; 110b–e V1 FAIL at 0.1963–0.1965 vs 0.190 | ✓ |
| 136 | Appendix: V1 corrected to the first-round loss of one machine | — | job 111 header: 0.1963 = 110c base, first round | ✓ |
| 137 | Appendix: two first launches failed to download; relaunched once by an amendment committed before | — | 111b/c `dl_gguf.txt` rc 1 after 460/468 s; amendment pushed 00:10:54, relaunch 00:11:02 | ✓ ("before any measurement": D23) |
| 138 | Appendix: "Job 111 relaunched those machines" | — | only b, c, d relaunched; 111e has a wrapper but no rental; its UUID is not in `job110_uuids.txt` | ✗ (D5) |
| 139 | Appendix: job 110/111 clause counts; `tab_prereg` rows and new totals 550 / 671 / 309 / 84 | — | 110: 1 failed, 1 untested; 111: 12 + 26 = 38; totals recomputed | ✓ |
| 140 | Appendix: "…, \scClauses{} for jobs 110 and 111" | 565 (rendered) | 38 | ✗ (D1) |
| 141 | `tab_configs`: read-ahead oracle reads "1.2–1.4 R*" | typed | panel 1.22–1.31 / 1.42–1.43; new 1.22–1.24 / 1.42–1.43; RTX 4090 1.27–1.29 / 1.43 | ✓ value, ✗ typed (D19) |
| 142 | `tab_configs` / `tab_names` names | — | see D18 | ✗ minor |
| 143 | Config flags in raw rows (dk κ = 3 / 2; R1, R2; both3p lead 3 paced) | — | match the job 109 header | ✓ |
| 144 | `app_traces`: "Across the nine models … 0.65 C (quartiles 0.61–0.72)" | — | macros \dcMed, \dcLo, \dcHi; "nine" typed | ✗ minor (D19) |
| 145 | Build: clean `latexmk -pdf` | — | exits 12 after the first pdflatex pass (`! LaTeX Error: Command \scClauses already defined.`), so no bibtex or reruns | ✗ (D1) |
| 146 | Build with `-f`: undefined references or citations; overfull boxes | — | 0 and 0 (underfull only); 38 pages; text identical to the committed PDF | ✓ |
| 147 | Main text ends by page 10 | — | Conclusion ends in the right column of p. 10, then References | ✓ |

That is 147 table rows, with grouped rows counted once. They cover about 330 individual numbers: Table 4 has 64 values
with intervals, Table 5 has 80, the panel spot-check has 72, and the rest are text numbers, macros and verdicts.

## Defects (most serious first)

1. **`\scClauses` is defined twice; the build errors and the appendix prints a wrong number.**
   * `wsg_numbers2.tex:299` sets it to 565 (jobs 073–098). `wsg_job109.tex:71` sets it to 38 (jobs 110–111).
   * LaTeX keeps the first definition and raises `! LaTeX Error: Command \scClauses already defined.`
   * As a result, a clean `latexmk -pdf` from `git archive HEAD` exits 12 after the first pass.
   * The appendix sentence in `app_wsg.tex` (line 216) renders "565 for jobs 110 and 111" in `paper.pdf`; it should read
     "38 for jobs 110 and 111".
   * Fix: rename the job-109 macro (e.g. `\scbClauses`/`\tcClauses`) in `scripts/job109.py` and in that sentence.

2. **"predicted the oracles' gains … within 6%" (Abstract; Introduction ¶4; Table 1 row 6) states the wrong quantity.**
   * The 6% is the error in speed relative to the deployed cache (|obs/pred − 1| ≤ 5.94%). The gains themselves
     (X/base − 1) were missed by up to about half.
   * Read-ahead oracle at 11% on the two Core Ultra 9 285Ks: predicted +13.3% / +12.9%, observed +19.8% / +19.6%.
   * MIN 1 read at 25% on the 14900KF: predicted +6.9%, observed +2.0%.
   * Recommend: "predicted the oracles' speed relative to the deployed cache within 6%" (Section 5's own sentence,
     "came within 6% of the prediction", is correct).

3. **The frozen trend's error on the five new RTX 5090 machines is not reported, and it is larger than on the RTX 4090s.**
   * Job 109 also ran after the trend was frozen. On those machines the same trend misses 5 of 18 oracle cells by more
     than Q1's 0.10 (ln) tolerance:
     * 109e read-ahead 11%: 1.460 observed vs 1.242 predicted (+0.162).
     * 109d read-ahead 11%: +0.136.
     * 109a MIN 1 read: −0.120 at 11% and −0.113 at 25%.
     * 109e read-ahead 25%: +0.124.
   * Under Q1's rule, the trend would fail on these five machines. Across them, MIN 1 read's Spearman with the ratio is
     −0.10.
   * The RTX 4090 evidence spans ratios 0.37–0.40 only. "The link-to-CPU ratio, not the card, predicts the oracles'
     gains" (Table 1), the paragraph heading and "on either card we tested" (Conclusion) therefore overstate it.
   * Recommend adding one sentence in Section 5: "the same trend missed the five new RTX 5090 machines by up to 18%
     (0.16 in log)". Scope the claim to "three RTX 4090 machines at ratios 0.37–0.40 fell within 6%".

4. **The "31–54% of the bound's speed" claim lost its scope** (Abstract; Introduction "The headroom is large"; Rule 1 "on
   our machines"; Conclusion).
   * The range comes from the 25 consumer machines (`fig_decomp.py` excludes server processors).
   * Valid server machines run at 12.4% (AMD engineering sample, 107d), 26.7% (EPYC 7543, 108a) and 46.1% (EPYC 7402P,
     105a).
   * Recommend restoring "on machines with consumer processors" in all four places; Section 3 keeps it.

5. **The second-card test's deviations are understated, and one machine is unaccounted for.**
   * Limitations says the test "deviated from its registration once". It deviated in three ways:
     1. V1's reference was replaced (job 111).
     2. An amendment relaunched 111b/c, against the header's "One launch per machine".
     3. Job 110 host e's machine (offer 53029117, Core i9-14900KF, UUID GPU-ada7a477…) was listed in job 111's header and
        has a `111e` wrapper, but it was never rented. Its UUID is also missing from `jobs/ec2/job110_uuids.txt`, which
        lists only b, c and d, so it would have failed V0 had it started.
   * The appendix's "Job 111 relaunched those machines" is therefore 3 of 4.
   * Recommend: "relaunched three of the four (host e's offer was not relaunched, and its GPU was not on the V0 list)",
     and "deviated from its registration twice (the reference loss, and one relaunch of two machines whose download
     failed)".

6. **Rule 2 ("a better set read once closes about a third of it") and the abstract's 35% omit the link scope.**
   * Job 109's population was restricted to ratios ≥ 0.5.
   * On slow links, MIN's set read once closes −6.4%, 1.1% and 0.8% on the three RTX 4090s, and 8.1% and −23.2% on the
     panel's two slow-link machines.
   * Recommend adding "where the link reads at least half as fast as the CPU" to Rule 2. Have the abstract say the
     registered machines had link-to-CPU ratio ≥ 0.5.

7. **"What the oracles leave is mostly the GPU's own work" (Introduction ¶2, Conclusion) is reinstated after number check 17
   removed it.**
   * It is attributed, not measured.
   * At 11% T_GPU is under half of what is left on 6 of 15 panel machines (median 0.54), and about a third on all three
     RTX 4090s, where the residual is 22–36% of the gap.
   * Recommend the `c369d01` wording: "the GPU's own work in series with the reads and the oracle's own reads beyond
     MIN's" (or "about half … by attribution").

8. **"On any card" (Rule 4), "a rule that carried over to a second card" (Contributions) and "on either card we tested"
   (Conclusion) claim more than was run.**
   * The slow-link half of the rule (serve on the CPU and copy in the background, with fewest admissions) was never run
     on the RTX 4090s; no fewest-admission configuration ran there.
   * On those machines, MIN 2 reads gives 0.98–1.01× at 11%, while the read-ahead oracle, which copies over the slow
     link, gains most (1.13–1.20×).
   * Recommend: "the RTX 5090 trend for copying in the step and ahead carried over to three RTX 4090s; the
     background-copy path was measured on RTX 5090s only".

9. **Section 5: "the policies built on the layer-ahead copy lose time (0.64–0.68× at 11%)".**
   * `\scRoneLow` covers R1 only. pf runs at 0.682–0.709× and R2 at 0.691–0.724×.
   * Recommend "0.64–0.72×" (or "R1 at 0.64–0.68×").

10. **Section 5: "…at 11%; MIN's fewest-admission set pays instead … (1.11–1.22× on the slow-link RTX 5090s)".**
    * The range pools 11% (1.11× on all three) with 25% (1.16–1.22×).
    * It comes from machines at ratios 0.28–0.32, not "these ratios" (0.37–0.40); the old text said "over the two budgets".
    * Recommend "(1.11× at 11% and 1.16–1.22× at 25% on RTX 5090s at ratios 0.28–0.32)".

11. **The Setting paragraph's host-bound exception is inconsistent with the rest of the paper.**
    * It says the GPU term binds at gpt-oss 25% "on the machines with the fastest host memory (workstation and server
      processors)". Section 3 says the GPU term "binds only at the largest budgets".
    * The Machines paragraph classes Threadripper as consumer, yet the Threadripper 9960X (B_host 176–178) is GPU-bound
      at 25%. "Workstation" is never defined.
    * EPYC 7543, 7402P, 7352 and 7K62 (B_host 22–84 GB/s) stay host-bound at 25%. The exact criterion is B_host above
      about 107 GB/s.
    * Recommend "on machines whose probe reads above about 107 GB/s", and amend the Section 3 sentence to match.

12. **Section 5: "The online policies still read 1.66–1.67 R*".** R2 reads 1.675–1.678 R*, so the range for both online
    policies is 1.66–1.68. `\olReadsRone` is R1 only.

13. **Limitations: "in the two tests on new machines, \jlGated{} and \jmGated{} machines stopped at the gate …" is stale.**
    Jobs 109 (5 started, 0 gated, 0 unsteady) and 110 (1 of 5 gated) are also tests on new machines. Recommend "in the
    tests of jobs 107–110" with all four counts, or scope the sentence to jobs 107 and 108.

14. **"Of its 11 predictions 11 held" needs a note, and Q6 is missing.**
    * P8's clause for ratio ≥ 0.9 had no machine (highest ratio 0.889) and is not in the scorecard.
    * Job 110's Q6 (ratio ≥ 0.5 hosts) had no RTX 4090 and is neither scored nor mentioned.
    * Recommend: "11 held (P8 only on its clause for ratios below 0.75; no machine reached 0.9)", and "Q6 did not apply".

15. **Statistics paragraph:**
    * "a machine rented again came within 3.2% of its earlier run" is now contradicted by the RTX 4090 relaunches under
      the same protocol: 110c→111g deployed cache 19.35→20.41 ms (+5.5%), 110d→111d +4.1%.
    * "jobs 104 and 105 ran one process per configuration" reads as one process for each configuration. They ran all
      configurations of a budget in one process; say "a single process per budget".

16. **Table 5 caption: "the two built from both" is inaccurate.** R1 combines the layer-ahead copy with one read (κ = 1);
    only R2 adds admitting less. Recommend "the layer-ahead copy with one read, and that policy admitting less".

17. **Section 5 "Admit as rarely": "makes 0.60–0.73 of the greedy one's copies for the same hits"** dropped "in replay".
    In the engine it misses 2.5–5.3% more. Restore "in replay".

18. **Configuration names are still mixed.**
    * `tab_configs` uses "MIN, fewest adm." and "online, 1 read + layer-ahead"; the agreed names are "MIN, fewest
      admissions" and "online policies".
    * The Figure 4 legend uses "MIN greedy, 1 read (in the step)", "MIN greedy, 2 reads (CPU, then copy)" and "deployed +
      layer-ahead copy".
    * The main text says "MIN read once in the step" where Table 2 and Figure 3 say "MIN, 1 read".
    * `app_wsg.tex:85` still says "the prefetching oracle".

19. **Typed numbers that should be macros:**
    * `tab_configs` "reads 1.2–1.4 R*" (the values are in `\dmPrefReads*`/`\dmNewPrefReads*`: 1.22–1.31 at 11%,
      1.42–1.43 at 25%).
    * `app_traces` "Across the nine models" (`\fsModels`).

20. **Figure 2 legend and caption.** The legend says "new machines, registered (5)", but only 4 lines exist at 25%. The
    caption does not explain the thick median lines or which machines they summarise.

21. **Self-reference.** Section 5 says "On the slow-link RTX 4090s of Section 5"; `sec:secondcard` is a paragraph label
    inside Section 5. Use "above".

22. **Appendix grammar.** `\olGated`/`\olUnstable` = "no" renders "no stopped at a gate, no failed a round check".
    Recommend "none … none".

23. **"before any measurement" (appendix and job 111 header) for 111b/c.** The bandwidth probe ran (ratios 0.396 and
    0.395 are recorded). Say "before any model run".

24. **"The RTX 4090s' first launch stopped after one round" (Section 5).** 110a stopped at the ratio gate before any round.
    Recommend "the four that passed the ratio gate stopped after one round".

25. **Minor scope and labels:**
    * Table 1 row 6 counts "19+3" machines. 19 is the Spearman set; the frozen trend was fitted on 15.
    * "each configuration's gain": the frozen trend covers four configurations.
    * `\dcMachinesLow` (25) now leaves out job 109's five consumer machines. Their shares (41.5–48.0%) fall inside 31–54%.
    * The Yardsticks paragraph does not list capture, which Section 5 now uses.

26. **The registration-timing record (`prereg/gpu_pushes.json`, `scripts/reg_timing.py`, `\rt*`) stops at job 108.**
    GitHub's activity API shows every 109–111 commit pushed 5.9–23.9 s before the first rental it governs, so nothing
    is wrong. Extend the push log and macros to jobs 109–111 so that the appendix's timing sentence covers them.
