# Number check 23 (9 October)

## Scope

Everything that changed in `paper/`, `scripts/` and `prereg/` between 215d10e and 919dd15, with emphasis on:

1. Job 114 (registration, host rule and replacements, raw results of 114a-g, scorer `scripts/job114.py`, `paper/wsg_job114.tex`, `paper/tab_job114.tex`, `prereg/scorecard_114.json`, `prereg/job114.json`).
2. The CPU-only column of Table 4 (`scripts/decomp_measured.py`, `paper/tab_dm.tex`, `dmCpu*` in `paper/wsg_dm.tex`) and the redrawn Fig. 2.
3. Where job 114 entered the paper: abstract, introduction, Table 1, Section 4 (all three new paragraphs), Fig. 2 caption, Section 5 qualifier, builder rules 2 and 4, Limitations, Conclusion, `app_prereg.tex`, `app_limits.tex`, `tab_failures.tex`, `tab_prereg.tex`, the scorecard's job-114 table.
4. The other round-26 fixes (like-for-like sentence, "at most 6%", the trend's miss, post hoc marks, Table 1's bound and Few-1R rows, `app_more.tex` geometric mean, 110e capture, "scorecard" renaming, Table 6 groups, cost and registration-timing macros).
5. IEEE build (`ieee-paper.pdf`, `ieee-supplement.pdf`, `refs_ieee.bib`/`refs_wsg_ieee.bib` from `scripts/ieee_bib.py`).
6. Builds of `paper.pdf` and `supplement.pdf`.

## Method

- All job-114 numbers were recomputed with my own code (`nc23/j114.py`, `nc23/cpucol.py`, `nc23/noise114.py` in the session scratchpad) directly from `results/114[a-g]_decomp0@vast/ec_g_C*_r*A.jsonl` (time per token = decode_ms / n_decode, mean over problems; ratio X/Y = geometric mean over rounds of Y's mean time / X's; 4,000-draw paired bootstrap over problems, the same problem indices for every round and configuration), `st_g_*.json` (counters / steps, mean over rounds), `fetch_table_law_gptoss.json` (B_p/B_c, table), `concur.txt` (B_host = max of every cpu_read, pcie and concurrent-sum reading, as `speed_limit.host_rates` defines it) and `v0.txt`/`validity.txt`. Eq. (1) = R* x S / B_host with R* = 38.315 / 15.340 reads per token and S = 13,253,760 bytes; gap = deployed (with table) - Eq. (1); t-intervals over machines.
- Scripts were read only for definitions. My bootstrap seeds differ from the scorer's; all intervals agree to the printed precision.
- Host rule: the header of `jobs/114_decomp0@vast.sh` (cf0112c), the three offer listings `results/114_offers_at_*.txt`, the wrappers 114a-g, `git log` on the gpu branch, `prereg/gpu_pushes.json` and `gpu/vast_ledger.json` in the main repository.
- Other recomputations from raw data: job 109's trend deviations and captures (`results/109*`), 110e's first round (`results/110e*`), the Few-1R speeds of the unsteady machines (106c, 106d, 107c, 107e), the engine-above-probe count over the 84 launch-budgets, the rental/cost totals from the ledger.
- Builds: `git archive 919dd15 paper scripts` into the scratchpad (plus the untracked author blocks), `latexmk` for paper and supplement, `build_ieee.sh` for the IEEE pair; text of every rebuilt PDF compared with the committed PDF (`pdftotext -layout`).

## Items checked

### A. Job 114 registration, hosts and gates

| # | Location | Claim | My value | Verdict |
|---|---|---|---|---|
| 1 | gpu cf0112c | header, predictions H1-H6 and offers listing committed before any rental | commit 21:40:53Z, push 21:40:55Z; rentals a-d 21:41:08-21:41:11Z (13-16 s after push) | OK |
| 2 | gpu b8152c9 | wrappers f, g and listing 1 before f's rental | commit 21:50:18Z, push 21:50:20Z; f rented 21:50:26.4Z | OK |
| 3 | gpu 875d285 | listing 2 before e and g | commit 22:02:43Z, push 22:02:45Z; e 22:02:47.6Z, g 22:02:48.3Z (their wrappers pushed at 21:40:55Z and 21:50:20Z) | OK |
| 4 | manifests | machines start after the push | b 21:42:40Z, c 21:44:51Z, f 21:51:01Z, e 22:03:45Z (>= 41 s after the governing push) | OK |
| 5 | header | "none in the rental ledger" (the five listed offers) | d's offer 49539124 is in the ledger (089_headline_stockclock); its GPU then failed V0 | FALSE (D1) |
| 6 | listing 1 (21:50:18Z) vs header rule | replacements = next listed offers (f, g) | cheapest further whole-machine desktop Ryzen/Core offer with >= 300 Mb/s: 54905524 (Core Ultra 9 285K, 921 Mb/s, $0.669/h, not in ledger), then 54573923 (f, $0.802); g's 54573924 is not in this listing | DEVIATION (D1) |
| 7 | listing 2 (22:02:42Z) | e next in registered order; g "the cheapest further offer (tied with 52268221)" | e correct; 40038866 (Core Ultra 9 285K, 904.7 Mb/s, $0.675/h) is cheaper than g ($0.802); g's offer is in the ledger (108f) | DEVIATION (D1) |
| 8 | app_prereg Job 114 | "the replacements were the next listed offers, as registered" | see 6-7 | WRONG (D1) |
| 9 | 114a/d/g v0.txt | three stopped at V0, GPUs rented before | a: job 071's GPU; d: job 089's; g: job 108f's; all in known_gpu_uuids.txt at cf0112c | OK |
| 10 | 114e | failed V2, rounds 6.1% apart, not scored | 17.768 vs 16.747 ms = 6.10%; V0/V0c/VR/V1/V3 passed (42 GB in use <= 48) | OK |
| 11 | 114b/c/f gates | valid | NUMA 1; cores 16/20/16; device read 1695/1695/1694 GB/s; disk 150 GB; 3/2/3 GB in use; VR 0.786/0.566/0.783; V1 0.1894/0.1902/0.1909 (C=14), 0.1899/0.1900/0.1900 (C=32); V2 0.82/0.30/0.52%; V3 max loss deviation 0.67/0.45/0.77% | OK |
| 12 | 114f | job 109f's machine | v0 "one of job 109's machines"; offer 54573923 = 109f and 112g in ledger | OK |
| 13 | fetch tables | b all zero | b 0,0,0,0,0; c 0,0,1,1,2; f 0,0,1,2,3; e 0,0,1,1,2 | OK |
| 14 | app_limits | started seven, 3 V0 + 1 unsteady, 3 valid of 5 aimed | 7 started (a, b, c, d, f, e, g), 3 + 1 + 3 | OK |
| 15 | prereg/job114.json | host records | values match mine; `known` is True for a, d, g (substring bug) | DEFECT (D13) |

### B. Table 5 (tab_job114), gpt-oss 11%

| # | Location | Claim | My value | Verdict |
|---|---|---|---|---|
| 16-24 | Ryzen 9 5950X (b, ‡) | 0.79; 1.00±.00; 1.12±.01; 1.12±.01; 1.06±.01; 1.07±.00; 1.39±.02; 3.2 [3.0, 3.3] | 0.786; 1.0000 (hw .002); 1.1209 (.012); 1.1235 (.012); 1.0646 (.005); 1.0682 (.004); 1.3899 (.016); 3.175 [3.006, 3.348] | OK (9) |
| 25-33 | Core Ultra 7 265K (c) | 0.57; 1.06±.01; 1.00±.01; 1.10±.01; 1.01±.00; 1.03±.00; 1.16±.01; 0.4 [0.3, 0.5] | 0.566; 1.0558 (.007); 0.9985 (.007); 1.1020 (.010); 1.0065 (.002); 1.0260 (.005); 1.1596 (.011); 0.360 [0.256, 0.466] | OK (9) |
| 34-42 | Ryzen 9 5950X (f) | 0.78; 1.00±.01; 1.01±.01; 1.12±.01; 1.01±.00; 1.07±.01; 1.39±.01; 3.1 [2.9, 3.3] | 0.783; 1.0003 (.008); 1.0079 (.006); 1.1240 (.011); 1.0092 (.002); 1.0691 (.005); 1.3877 (.015); 3.101 [2.926, 3.275] | OK (9) |
| 43 | Table 5 note | ‡ machine's deployed path had no in-step fetches | b's base fetches 0.000/step | OK |
| 44 | Table 5 MinOne column | "same on both" paths | column is fetch/base0; vs the deployed cache with table c is 1.22, not 1.16 | UNCLEAR (D5) |

### C. Job 114 macros used in the text

| # | Macro | Paper | Mine | Verdict |
|---|---|---|---|---|
| 45 | dzGreedyZeroLowRng | 1.10-1.12 | 1.1020-1.1240 | OK |
| 46 | dzGreedyZeroMidRng | 1.28-1.36 | 1.2763-1.3572 | OK |
| 47 | dzOnceZeroLowRng | 1.03-1.07 | 1.0260-1.0691 | OK |
| 48 | dzInterMsLowRng | 0.4-3.2 ms | 0.360, 3.101, 3.175 | OK |
| 49 | dzInterMsMidMin/Max | -0.1 to 0.1 | -0.085, 0.043, 0.132 | OK |
| 50 | dzInterPosCIMid | one | b [0.019, 0.255]; c [-0.173, 0.005]; f [-0.030, 0.117] | OK |
| 51 | dzMissGreedyTabLowRng | 48.6-50.8 | 48.624 (c), 50.750 (f) | OK |
| 52 | dzMissGreedyZeroLowRng | 43.5 | 43.504 (all three) | OK |
| 53 | dzFetchOnceTabLowRng | 23.4-32.0 | 23.420, 31.964 | OK |
| 54 | dzFetchOnceZeroLowRng | 8.2 | 8.194 | OK |
| 55 | dzAdmGreedyZeroLowRng | 20.4 | 20.387 | OK |
| 56 | dzAdmBaseZeroLowRng | 8.2 | 8.202 | OK |
| 57 | dzInteractionTabLowRng | 31-52 % of gap | 30.7 (c), 52.2 (f) | OK |
| 58 | dzInteractionZeroTabLowRng | 3-21 % | 3.2, 20.8 | OK |
| 59 | dzTabNWord / dzN / dzNWord | two / 3 / three | 2 / 3 | OK |
| 60 | dzStarted / dzGated / dzRoundFailN | 7 / three / one | 7 / 3 / 1 | OK |
| 61 | dzRoundFailCpu / Spread | Ryzen 9 9950X3D2 / 6.1 | 6.10% | OK |
| 62 | dzRatioRng | 0.57-0.79 | 0.566-0.786 | OK |
| 63 | dzNoTabCpu / FetchRatio / GreedyHeld | Ryzen 9 5950X / 1.00 / 1.00 | 8.194/8.194; 1.0023 | OK |
| 64 | dzClauses / Held / Point / Failed | 34 / 18 / 14 / 2 | correct as row counts, but 18 includes the validity condition | DEFECT (D3) |
| 65 | unused dz macros | Min/Max/Rng consistent | dzAhead*, dzTogether*, dzTogetherZero*: Rng = speed ratios, Min/Max = gap shares | DEFECT (D14) |

### D. Clause statuses (scorecard_114.json), recomputed

| # | Clauses | Registered | Mine | Verdict |
|---|---|---|---|---|
| 66 | H1a g11/g25, b c f (6) | fetch misses, fetches within 1% of 39.7/19.8 and 16.4/9.7 | 39.732/19.814: dev 0.0008; 16.420/9.685: dev 0.0015; held (point) | OK |
| 67 | H1b base0, bypass0 (6) | no fetch; within 1% of 62.1/8.2 and 43.5/20.4 | 0 fetches; 0.0004, 0.0007; held (point) | OK |
| 68 | H1c (3) | foa0 fetches <= 0.5 x foa's | b 1.000 failed; c 0.350, f 0.256 held (point) | OK |
| 69 | H2 (3) | bypass0/bypass >= 1.03 | b 1.0023 [1.0012, 1.0033] failed; c 1.1652 [1.1586, 1.1718], f 1.1155 [1.1112, 1.1198] held | OK |
| 70 | H3 g11 (3) | bypass0/base0 >= 1.05 | 1.1235 [1.1119, 1.1353], 1.1020 [1.0916, 1.1122], 1.1240 [1.1133, 1.1343] held | OK |
| 71 | H3 g25 (3) | >= 1.15 | 1.3500 [1.3309, 1.3687], 1.2763 [1.2596, 1.2930], 1.3572 [1.3397, 1.3741] held | OK |
| 72 | H4 (3) | foa0/base0 in [0.95, 1.10] | 1.0682, 1.0260, 1.0691, all intervals inside | OK |
| 73 | H5 (3) | interaction > 0 (ms) | 3.175 [3.006, 3.348], 0.360 [0.256, 0.466], 3.101 [2.926, 3.275] held | OK |
| 74 | H6 (3) | base0/base in [0.92, 1.10] | 1.0000, 1.0558, 1.0003, intervals inside | OK |
| 75 | 114-valid | >= 2 valid | 3, met; scored "held" with no interval | DEFECT (D3) |
| 76 | registered interpretation | "If H3 and H5 hold, the paper says that MIN's set pays alone too, and more together with one read" | H3, H5 held; Section 4 heading says so | OK |

### E. Table 4 CPU-only column (shares of base - Eq. (1), mean [95% t-interval], 3 machines)

| # | Row | Paper 11% / 25% | Mine 11% / 25% (per machine b, c, f at 11%) | Verdict |
|---|---|---|---|---|
| 77 | MIN's set, read twice | 20 [10, 29] / 37 [20, 54] | 19.6 [9.9, 29.3] (21.8, 15.1, 21.9) / 37.2 [20.3, 54.1] | OK |
| 78 | deployed set, read once | 10 [-2, 22] / 7 [-13, 28] | 9.9 [-2.5, 22.2] (12.6, 4.1, 12.9) / 7.4 [-13.1, 27.9] | OK |
| 79 | Both | 44 [-3, 92] / 45 [5, 84] | 44.5 [-3.0, 92.0] (55.5, 22.4, 55.5) / 44.7 [5.2, 84.2] | OK |
| 80 | read-ahead oracle | 14 [5, 22] / 21 [16, 26] | 13.7 [5.3, 22.1] / 21.1 [16.0, 26.2] | OK |
| 81 | left | 39 [13, 65] / 37 [11, 63] | 38.8 [12.6, 64.9] / 36.9 [11.1, 62.8] | OK |
| 82 | in-step fetches themselves | 3 [-10, 16] / -3 [-17, 12] | 3.0 [-10.0, 16.0] / -2.7 [-17.1, 11.6] | OK |
| 83 | T_GPU (2.9402 ms) | 22 [12, 31] / 29 [15, 43] | 21.8 [12.3, 31.4] / 28.9 [15.2, 42.6] | OK |
| 84 | oracle's extra reads | 21 [12, 29] / 20 [12, 29] | 20.6 [12.1, 29.1] / 20.1 [11.6, 28.6] | OK |
| 85 | residual | -4 [-29, 21] / -12 [-32, 8] | -3.7 [-28.8, 21.4] / -12.1 [-32.5, 8.3] | OK |
| 86 | note: Both + ahead + left (+ †) = gap | sums | 100.0 per machine, both budgets | OK |
| 87 | dmCpuSetaloneZeroLow/Mid, OncealoneZeroLow, TogetherZeroLow (text) | 20, 37, 10, 44 | as above | OK |
| 88 | dmCpuInteractionZeroLow/Mid (macro) | 15 [-10, 40] / 0 [-3, 3] | 15.0 [-10.4, 40.5] / 0.2 [-2.5, 2.8] | OK |
| 89 | B_host, Eq. (1) | from probe | b 34.5, c 63.0, f 34.4 GB/s; Eq. (1) 14.719/8.061/14.762 ms (11%) | OK |
| 90 | app_limits probe sensitivity "fast-link shares ... at most 5 (9) and 6 (11) points" | covers Table 4 | computed on panel + new only; for the CPU-only column I get <= 4.6 (8.4) and 5.6 (10.2): within | OK |

### F. Fig. 2 (decomp_measured.pdf)

| # | Claim | Mine | Verdict |
|---|---|---|---|
| 91 | blue lines relative to the deployed cache with table; diamonds = base0/base | c diamond 0.947 (11%), f 1.061 (25%), b ~1.00 | OK |
| 92 | squares = Dep-1R; legend counts 18 fast, 2 slow, 3 CPU-only | 13 panel fast + 5 new; 2; 3 | OK |
| 93 | blue median at MIN-1R ~0.72 at 11% | b, f fetch/base 0.720, 0.720; c 0.817 | OK |
| 94 | x-label "MIN's set, read twice" for grey lines | Table 4 note: on the deployed path this point is MIN's schedule with the table, not the set held | UNCLEAR (D16) |

### G. Where job 114 entered the text

| # | Location | Claim | Verdict |
|---|---|---|---|
| 95 | Abstract | "on three machines, the set alone closes 20% ... both 44%" | OK (means; scoped) |
| 96 | Abstract | "with it, the set pays only if each admitted expert is also read once" | OK |
| 97 | Abstract | "about half of the gap"; "On five fast-link machines ... at most 6%" | OK (46-52%; 5.7% max) |
| 98 | Introduction | "on three machines registered for the test ... 20%, 10%, 44%" | numbers OK; "registered for the test" loose (D1) |
| 99 | Introduction | consumer range not marked post hoc while next sentence is | DEFECT (D15) |
| 100 | Table 1, job 114 row | thresholds = H3, H4, H5; 1.10-1.12x, 1.28-1.36x, 1.03-1.07x, +0.4-3.2 ms (held); two removal checks failed where table all zero; Mach. 3 | OK |
| 101 | Table 1, deployed-path row | "pays only together with one read"; 9 of 10 (failed), 5 of 5 (held (point)) | OK (unchanged numbers) |
| 102 | Table 1, job 113 row | "(each failed on the Core Ultra 9 285K, at 11%)" | OK (113d-H3 1.0732, 113d-H6 0.0238, both 11%) |
| 103 | Sec. 4 intro | 15 panel + 5 new; 3 of a second registered test | OK |
| 104 | Sec. 4 "displace" | 48.6-50.8 vs 43.5; 23.4-32.0 vs 8.2; MinOne same on either path; third machine all zero | OK |
| 105 | Sec. 4 "pays alone" | 1.10-1.12x, mean 20%; 1.03-1.07x, 10%; 44%; 0.4-3.2 ms, all above zero; 20.4 vs 8.2 admissions; largest on the two 5950X; 25%: 37%, -0.1 to 0.1 ms | OK |
| 106 | Sec. 4 deployed path | "Most of it is the table's: ... 31-52% to 3-21%" | numbers OK; generalised from two machines (b: 22% with no table) (D6) |
| 107 | Sec. 4 | nothing claims the interaction belongs only to the table | OK (no such claim left anywhere) |
| 108 | Sec. 5 qualifier | "With the in-step fetches off, MIN's set gains read twice as well" | true (1.09-1.12x on 5 machines) but unscoped (D7) |
| 109 | Builder rule 2 | "closes 20% on its own and 44% ..." | unscoped means (D7) |
| 110 | Builder rule 4 | copy admitted misses in the step on fast links | consistent with Tables 4-6 |
| 111 | Limitations | "the 2x2 ran on three machines, two rounds at 11% and one at 25%" | OK; "two tests deviated" now three (D1) |
| 112 | Conclusion | "with them off, it pays by itself, and more when read once" | true on 3 (5) machines; unscoped (D7) |
| 113 | app_limits | "CPU-only column rests on three machines, two of its three are Ryzen 9 5950Xs" | OK |
| 114 | app_prereg claim index | job 114 row, clause IDs 114x-H1a..H6 | OK (scorecard prints them without the job prefix, as for every job) |
| 115 | app_prereg Job 114 | protocol, predictions source, offers committed before start, 7/3/1, ratios 0.57-0.79, f = 109f, MinOne within 1%, H1c/H2 failed on b only | OK |
| 116 | app_prereg Job 114 | "Of the job's 34 clauses, 18 held with their intervals" | DEFECT (D3) |
| 117 | app_prereg Job 114 | interaction 0.4-3.2 ms, above zero on all 3; 25%: -0.1 to 0.1, above zero on one | OK |
| 118 | app_prereg Job 113 | offers file transcribed after the job; job 114 saved its listing at registration | OK (cf0112c contains it) |
| 119 | tab_failures 114 rows | 1.00 on the 5950X (H1c); 1.00 (H2); thresholds 0.5 and 1.03 | OK |
| 120 | tab_failures heads 107-114 | per-job tallies | include validity conditions, unlike Table 10 (D3) |
| 121 | tab_prereg rows | 107 11/22/12/0 not met; 108 0/12/6/2 met; 109 47/43/0 met; 110 0/0/0/1 not met; 111 24/13/0 met; 112 22/23/12 met; 113 6/6/2 met; 114 17/14/2/0 met | OK |
| 122 | tab_prereg totals | 611 / 695 / 323 / 84; 6 met, 2 not | 595-1+17; 685-4+14; 323-2+2; 84; met 108,109,111,112,113,114; not 107,110 | OK |
| 123 | Scorecard Table (job 114) | rows and values | all 34 rows match my values | OK |
| 124 | Scorecard Table (job 114) caption | hosts "b, c, e and f" | e failed V2 and has no rows | DEFECT (D4) |
| 125 | app_prereg "Rounds and rentals" | noise rescoring | covers 109-113 only; job 114 not rescored (D8) | OMISSION |

### H. Other round-26 fixes

| # | Location | Claim | Mine | Verdict |
|---|---|---|---|---|
| 126 | Intro | like-for-like: ours median 27%, published 13.6% at datasheet rates | matches Sec. 3 and wsg_auditours / wsg_numbers2 (macros unchanged since 215d10e; not recomputed) | OK |
| 127 | Abstract, intro, Table 1, rule 3 | "at most 6%" scoped to five fast-link RTX 5090s | capture at 11%: 4.1, 5.7, 1.5, 4.5 (+f); max 5.7 -> 6; best 1.007-1.024 | OK |
| 128 | Intro, Table 1 | trend missed new RTX 5090s by up to 18% | max 17.6% (109e, both3p, 11%); 5 of 18 cells beyond 0.10 | OK |
| 129 | Table 1 bound row | engine >5% faster than probe on 12 of 84 launch-budgets | 12 of 84, max 1.22 | numbers OK; wording (D9) |
| 130 | Table 1 Few-1R row, Sec. 5 | of four unsteady/invalid, lost on two (ratios 0.14, 0.21), gained on others | 106c 0.845 @0.14; 107e 0.974 @0.21; 106d 1.302 @0.23; 107c 1.175 @0.51 | OK |
| 131 | app_more | Few-1R geometric mean, log-scale t-interval | robustness.py: GMEAN, t-interval on logs | OK |
| 132 | app_more | Margin "each machine's mean over the two gpt-oss budgets" | also a geometric mean (log scale), and averages a ratio across budgets | DEFECT (D10) |
| 133 | app_prereg 110e | capture 78% of a 5.6% gain | dk 1.0432, both3p 1.0557: 77.6%, 5.57% | OK |
| 134 | app_prereg 110e | first round -5.6% from trend | both3p -5.6%, fetch -3.6% | OK |
| 135 | app_prereg 110e | sentence structure | two unrelated clauses joined by a semicolon | DEFECT (D11) |
| 136 | "scorecard" renaming | supplement.pdf = "The prediction scorecard"; no stray "the supplement" meaning it | OK (EPYC 9655 rows, grid, v(F) all in supplement.pdf) |
| 137 | Table 6 groups | RTX 5090 rented for the no-foresight test; RTX 4090 second-card test; at higher ratios | matches jobs 109, 111, 112 | OK |
| 138 | wsg_numbers2 | 154 rentals, 90 offers, $112.7, jobs 058-114 | 154; 90; $112.743 at GPU prices; 058-114 | OK |
| 139 | wsg_regtime | 101 rentals of jobs 093-114 | 101 (7 of job 114); 2.7 s, 26 s, seven, four unchanged by job 114 | OK |

### I. IEEE build

| # | Check | Result | Verdict |
|---|---|---|---|
| 140 | Rebuild with build_ieee.sh | 0 errors, 0 undefined references (30 expected multiply-defined citation labels via xr); paper 10 pp., supplement 30 pp. | OK |
| 141 | Cross-references both ways | paper -> Tables S3, S7, S23-S25, Appendices C-N correct; supplement -> Tables IV, V, I, Sections III-XI, Fig. 2/3 correct | OK |
| 142 | Run-in heads | "Models and workload:", "The in-step fetches displace ...:", "Two tallies:" unlettered | OK |
| 143 | Table II placement | Section II starts p. 2, Table II on p. 3 | OK |
| 144 | Footnote | "Appendices A-N ... S1, S2 ... supplementary material available online" | 14 appendices, A-N | OK |
| 145 | Supplement opening note | S-numbering; scorecard is a separate document (supplement.pdf) | OK |
| 146 | Long-table captions | TABLE S5, S27 stacked like the other tables | OK |
| 147 | Abbreviated venues | no title touched; "Journal of Mach. Learn. Research", "IBM Syst. Journal", "ACM Trans. on Storage" half-abbreviated; a note field edited ("(per arXiv v2)" dropped) | DEFECT (D12) |
| 148 | ieee-paper.pdf current | rebuilt text identical to committed | OK |
| 149 | ieee-supplement.pdf current | committed PDF lacks the app_prereg sentence on boot-script commit recording | STALE (D2) |

### J. Builds (MLSys)

| # | Check | Result | Verdict |
|---|---|---|---|
| 150 | paper.pdf rebuild | 0 errors, 45 pages; Conclusion ends p. 10, References start p. 11 | OK |
| 151 | committed paper.pdf current | lacks "The results of jobs up to 114 do not record which commit ... writes it into the results." (app_prereg.tex edited at 18:32:42, PDF built 18:31:51) | STALE (D2) |
| 152 | supplement.pdf rebuild | 0 errors, 58 pages; text identical to committed | OK |

Items checked: 152 rows, 186 individual values or statements (rows 16-42 hold 27 values; rows 66-74 hold 33 clauses).

## DEFECTS

1. **Job 114's replacements did not follow the registered host rule, and the paper says they did** (`paper/app_prereg.tex`, Job 114 paragraph: "the replacements were the next listed offers, as registered"; also `results/114_offers_at_replacement2.txt` note and commit b8152c9's message). At the first replacement the committed listing (21:50:18Z) shows offer 54905524 (Core Ultra 9 285K, whole machine, 921 Mb/s, $0.669/h, not in the ledger; possibly job 113d's machine) as the cheapest qualifying further offer, cheaper than f's 54573923 ($0.802/h); g's offer 54573924 is not in that listing at all. At the second, 40038866 ($0.675/h, qualifying) was cheaper than g, so g was not "the cheapest further offer". The header's "none in the rental ledger" was also false: d's 49539124 (job 089), f's 54573923 (jobs 109f, 112g) and g's 54573924 (job 108f) are all in `gpu/vast_ledger.json` (d and g then failed V0). **Fix:** replace the clause with "The replacements departed from the registered order: at the first, f (54573923, $0.802/h, job 109f's machine) and g (54573924, which the listing does not show) were started instead of the cheapest qualifying offer, 54905524 ($0.669/h); at the second, g was restarted ahead of a cheaper qualifying offer (40038866). The header's 'none in the rental ledger' was false for d (job 089), f (jobs 109f and 112g) and g (job 108f); d and g stopped at V0." Add the same to `app_limits.tex`'s job-114 sentence, change the Limitations' "two tests deviated from their registration" to "three tests", and in the introduction write "on three machines of a registered test" instead of "registered for the test".
2. **Committed PDFs are stale** (`paper/paper.pdf`, `paper/ieee-supplement.pdf`). Both were built before the last edit of `app_prereg.tex` and lack its sentence "The results of jobs up to 114 do not record which commit of the branch a machine ran; its boot script now writes it into the results." **Fix:** rebuild (`latexmk` for paper and supplement, `build_ieee.sh`) and recommit. The rebuilt paper still ends the Conclusion on p. 10 with References on p. 11.
3. **Clause tallies contradict Table 10 now that validity conditions are counted apart** (`app_prereg.tex` Job 114; `tab_failures.tex` heads for jobs 107-114; scorecard captions). `app_prereg.tex` says "Of the job's 34 clauses, 18 held with their intervals". The 18 includes `114-valid`, which has no interval, while Table 10 gives 17/14/2 plus "met". The `tab_failures` heads still count conditions the same way: 107 (11/22/13 against 11/22/12), 108 (13 point against 12), 109 (44 point against 43), 111 (14 against 13), 112 (24 against 23), 113 (7 held against 6) and 114 (18 against 17). `113-valid` and `114-valid` are scored "held" with no interval, while 109, 111 and 112's are "held (point)". **Fix:** write "Of the job's 33 clauses, 17 held with their intervals, 14 on the point estimate (the counter checks, which have no interval) and 2 failed; its validity condition (at least two valid machines) was met". Generate the `tab_failures` heads from the same counts as `tab_prereg.py` (excluding `*-valid`), with the condition named separately. Score the `*-valid` rows "met"/"not met" (or "held (point)") in `job113.py` and `job114.py`.
4. **The scorecard's job-114 caption lists e as a scored host** (`scripts/scorecard_auto.py`, which generates `paper/tab_scorecard_114.tex`). It reads "Hosts b, c, e (Ryzen 9 9950X3D2) and f", but e failed V2 and has no rows. **Fix:** "Hosts b (Ryzen 9 5950X), c (Core Ultra 7 265K) and f (Ryzen 9 5950X, job 109f's machine); e (Ryzen 9 9950X3D2) failed the round check (V2: its deployed cache's rounds 6.1% apart) and is not scored; a, d and g stopped at the first gate ..."
5. **Table 5 does not say which baseline the MinOne column uses** (`tab_job114.tex`, written by `scripts/job114.py`). The column is fetch/base0, relative to the CPU-only deployed cache. Against the deployed cache with its table the Core Ultra 7 265K is 1.22, not 1.16, so the note "the same on both" reads wrongly. **Fix:** in the note, write "MinOne: relative to the CPU-only deployed cache (its forced plans replace the table, so its own time is the same on both paths)".
6. **"Most of it is the table's" generalises from two machines** (Section 4, "On the deployed path ..."). On the third machine (no table) the deployed-path interaction is 22% of its gap, none of it the table's. **Fix:** "On the two job-114 machines with a nonzero table, most of it was the table's: it fell from 31-52% of the gap to 3-21% without the in-step fetches; on the machine without one it was 21-22% either way."
7. **Three new claims are unscoped** (builder rule 2, the Conclusion, Section 5's qualifier). Each states a three-machine (five with job 113) gpt-oss-11% result without scope. The 44% is a mean whose t-interval is [-3, 92], with per-machine values of 22-56%. **Fix:**
   - Rule 2: "without them, on three machines, it closes a mean 20% (15-22%) on its own and 44% (22-56%) when each admission is also read once".
   - Conclusion: "with them off, on the five machines tested, it pays by itself ...".
   - Section 5: "With the in-step fetches off (gpt-oss 11%, the five machines of jobs 113 and 114), MIN's set gains read twice as well".
8. **Job 114 is missing from the noise rescoring** (`app_prereg.tex`, "Rounds and rentals"; `scripts/noise.py`). The rescoring covers jobs 109-113, but job 114 ran two rounds at 11% and Table 1 marks its row "(held)". I widened each interval by the round range and the 5.3% largest rental difference. Under that widening b-H3-g11, b-H4, c-H3-g11, c-H6 and f-H4 would hold only on the point estimate; none fails. **Fix:** add job 114 to `noise.py` and report the counts, or state that job 114 was not rescored.
9. **Table 1's bound row states an inference as a measurement** (`main_body.tex`). "the engine read >5% faster than the probe on 12 of 84 launch-budgets" is a rate implied by Eq. (4), as `app_more` ("By Eq. (4)'s own accounting") and the Limitations ("imply") say. **Fix:** "by Eq. (4)'s accounting, the engine's reads imply a rate >5% above the probe's on 12 of 84 launch-budgets".
10. **The geometric-mean fix was applied to Few-1R but not to \Margin** (`app_more.tex`). "\Margin, each machine's mean over the two gpt-oss budgets, 1.020x (1.004-1.035)" is also a geometric mean with a log-scale t-interval. It also averages a ratio across budgets, which checklist rule 4 says is never done. **Fix:** report the two budgets separately (1.021 [1.000, 1.043] at 11%, 1.018 [0.987, 1.049] at 25%; the `rbDkLow`/`rbDkMid` macros), or call it a geometric mean and note the exception in rule 4.
11. **A run-on sentence in the 110e passage** (`app_prereg.tex`, Jobs 110 and 111). "...where the oracle gains little, capture is unstable; the relaunched machines' first rounds were in the repository when the relaunch was committed." joins two unrelated statements. **Fix:** end the sentence at "unstable." and start a new one: "The relaunched machines' first rounds were in the repository when the relaunch was committed."
12. **Venue abbreviations are half-applied, and the script edits a note field** (`scripts/ieee_bib.py`, giving `refs_ieee.bib` and `refs_wsg_ieee.bib`). "Journal of Mach. Learn. Research", "IBM Syst. Journal" and "ACM Trans. on Storage" are only partly abbreviated. The script also drops "(per arXiv v2)" from ExpertFlow's "Accepted at DAC 2026", although `refs_wsg.bib` records that the DAC DOI returned 404. Its own docstring says it changes booktitle and journal only. **Fix:** add ("Journal of Machine Learning Research", "J. Mach. Learn. Res."), ("IBM Systems Journal", "IBM Syst. J.") and ("Transactions on ", "Trans. ") to WORDS ahead of the single words, and remove the note replacement.
13. **The `known` flag in `scripts/job114.py` is wrong for three hosts.** The test `"one of job 109's machines" in v0` also matches "not one of job 109's machines", so `prereg/job114.json` records known=True for a, d and g. The paper is unaffected (`dzKnown` counts valid hosts only). **Fix:** test for "V0: one of job 109's machines".
14. **Colliding macros in `wsg_job114.tex` (unused, but a trap)** (`scripts/job114.py`, `macros()`). `dzAhead*`, `dzTogether*` and `dzTogetherZero*` hold speed ratios in `...Rng` (for example `dzAheadLowRng` = 1.40-1.51) but gap shares in `...Min`/`...Max` (12/18). **Fix:** give the share macros their own prefix (for example `dzShare...`).
15. **The introduction's consumer range has no post hoc mark** (`main_body.tex`, "The headroom is large"). Table 1 and the abstract mark it post hoc, and the next sentence is marked. **Fix:** "... on consumer machines (a scope set post hoc)".
16. **Fig. 2's label "MIN's set, read twice" contradicts Table 4's note for the deployed-path lines** (Fig. 2 caption). Table 4's ∗ note says that on the deployed path this state is MIN's schedule with the fetch table, not the set held. **Fix:** add to the caption "on the deployed path this state is MIN's schedule with the fetch table, which does not hold the set (Table 4)".
