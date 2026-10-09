# Number check 24 (9 October)

## Scope

Everything that changed in `paper/`, `scripts/`, `prereg/` and `gpu/` between 734fd54 and HEAD (cf75681; commits 50bb6fa,
fa02f37, cf75681), with the raw results on the `gpu` branch checkout (`/home/claude/gpu-branch`, HEAD 3a196aa):

1. Job 115: the header (H1-H8, host rule, AMENDMENT; registered in 4b69328), wrappers 115a-f, the five committed offer
   listings, the ledger and push record, each machine's `branch_sha.txt`, gates, and every ratio, interval, counter,
   interaction, probe comparison and clause status, plus the pooled H8 over jobs 114 and 115.
2. The scorer's outputs: `paper/wsg_job114.tex` (dz/dq/dp), `paper/tab_job114.tex` (Table 5), `prereg/scorecard_115.json`,
   `prereg/job114.json`; Table 4's fetches-off column (`dmCpu*`, `paper/tab_dm.tex`) and Fig. 2.
3. Scoring changes in `scripts/job109.py` (job 109 P1-int and P1 margin, job 112 Q6 now with paired-bootstrap intervals),
   the tallies (`tab_prereg.tex`, `wsg_job109.tex`, `wsg_job112.tex`) and the round/rental rescoring (`wsg_noise.tex`) and
   their description in `app_prereg.tex`.
4. Every place job 115 and the scoring changes entered the text (abstract, introduction, Table 1, Sections 2, 3, 4, 5,
   7, 9, Limitations, Conclusion, Appendix D, Appendix L, Appendix N, Table 11, cost and registration-timing macros,
   `fig_decomp.py`'s new median/quartile/envelope macros).
5. The four PDFs.

## Method

All numbers were recomputed with my own code in
`/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/nc24/` (run from that
folder); the repository's scripts were read only for definitions (which ratio, which counters, which rows):

- `recompute.py`: jobs 114/115 from `ec_g_*.jsonl` rows and `st_g_*.json` counters: per-round speed ratios against the
  same process's deployed cache, geometric mean over rounds, paired bootstrap over problems (20,000 draws, one problem
  resample per draw shared by every configuration and round), the interaction t_bypass0 + t_foa0 - t_base0 - t_fetch with
  its bootstrap interval and per-round values, counters per token, Eq. (1) at the first probe's highest reading, the
  gap decomposition, the probe comparison from the four `concur*.txt` files, and the pooled H8 t-interval.
- `macros_cmp.py`: rebuilds every range/count/share macro of `wsg_job114.tex` (dz, dq, dp) from those values and diffs
  them (1,250 macros); the remaining clause/status macros were checked by hand.
- `dm_cpu.py`: Table 4's fetches-off column (mean and 95% t-interval over machines, with T_GPU = 2.94 ms and the
  read-ahead oracle's reads from its counters).
- `recompute109.py`, `widen_check.py`: job 109 P1 margin and P1-int, job 112 Q6, and the round and rental widening of every
  clause new to the rescoring.
- `offers.py`: the job 115 host rule applied to each committed listing (physical cores from the CPU model, ledger as of
  the listing's timestamp).
- `regtime.py`, `consumer.py`, `slow.py`, `mincheck.py`, `colbal.py`: registration timing, the consumer share (median,
  quartiles, envelope), the slow-link Few-2R values, MIN-with-bypass against my own exhaustive search (300 random traces),
  and the IEEE last-page column bottoms.
- Builds: `git archive HEAD paper scripts mosl` into `nc24/build` and rebuilt all four PDFs there (`build_ieee.sh`,
  `latexmk`); the rebuilt text is identical to the committed PDFs. Push times were cross-checked against GitHub's
  activity API for `refs/heads/gpu`.

Bootstrap intervals agree with the committed ones to within Monte Carlo noise (third decimal); verdicts below say
"matches" when the paper's rounded value equals mine.

## Items checked

| # | Location | Claim | My value | Verdict |
|---|---|---|---|---|
| **A** | **Job 115 registration and hosts** | | | |
| 1 | gpu 4b69328 | header + wrappers 115a/b + registration listing committed and pushed before a, b rented | commit 04:47:48Z, push 04:47:50Z; rentals 04:47:52.6Z, 04:47:53.5Z | OK |
| 2 | gpu 94f6761 | 115c wrapper + replacement-1 listing before c | 04:52:37Z / push 04:52:39Z; a stopped 04:52:27Z; c rented 04:52:41.4Z | OK |
| 3 | gpu 5d62eb3 | amendment: 115d wrapper + listing before d | 05:30:40Z / push 05:30:42Z; d rented 05:30:44.1Z | OK |
| 4 | gpu 3649594 | AMENDMENT header text added after d started | push 05:31:01Z, 17 s after d's rental; changes comments only | OK |
| 5 | gpu 43c5910 | 115e wrapper + replacement-2 listing before e | 05:40:06Z / 05:40:08Z; d stopped 05:39:49Z; e rented 05:40:10.0Z | OK |
| 6 | gpu c8c6408 | 115f wrapper + replacement-3 listing before f | 05:55:11Z / 05:55:13Z; e stopped 05:50:46Z; f rented 05:55:19.1Z | OK |
| 7 | prereg/gpu_pushes.json | nine new push records | identical to GitHub activity API | OK |
| 8 | registration listing | only two qualify, 50915279 (TR 3970X) and 41738200 (i9-14900K) | same; cheaper whole machines 53278552, 54741988 are in the ledger | OK |
| 9 | replacement-1 listing | only qualifying offer 52126327 (Core Ultra 9 285K) | same | OK |
| 10 | amendment listing | cheapest qualifying 53904494; list 53904494, 54895344, 52126368, 53294830 | same (51363468 at $0.668 is in the ledger, job 099a) | OK |
| 11 | replacement-2 listing | cheapest qualifying 54895344 (Ryzen 9 9950X) | same | OK |
| 12 | replacement-3 listing | cheapest qualifying 53294830 (EPYC 7352) | same (47802004 at the same price: 37.8 Mb/s down) | OK |
| 13 | gpu/vast_ledger.json | six rentals, all <= $1.50/h, offers as wrappers | a 0.54, b 0.70, c 0.87, d 0.54, e 0.67, f 1.338; offers match | OK |
| 14 | branch_sha.txt | commit each machine ran | a 4b69328, b 94f6761, c 94f6761, d 3649594, e 43c5910, f c8c6408 | OK |
| 15 | app_prereg commit sentence | for two machines the commit was pushed after the rental and changed no executed line | b (94f6761, +4 m 46 s: c's wrapper, a listing, a's results) and d (3649594, +17 s: comments) | OK |
| 16 | v0/gate/validity | a, d, f VR fail; e V0; b, c valid | ratios 0.3932, 0.3931, 0.2642; e's GPU is 106e's; b, c pass V0, V0c, VR, V1, V2 (0.58%, 0.05%), V3 | OK |
| 17 | header V0 | known list holds job 114's seven GPUs | all seven present at 4b69328 | OK |
| 18 | header AMENDMENT; app_prereg Job 115 | amendment pushed "while b and c ran their first rounds, before any of their timed results existed" | c's round 1 ran 05:05:55-05:21:36Z (V1/V3 written), c in round 2 at the push; b in round 1 since 05:26:30Z | **Defect 1** |
| **B** | **Job 115 per-host results (115b Core i9-14900K / 115c Core Ultra 9 285K)** | | | |
| 19 | scorecard H1a-g11 | 0.0008 / 0.0008 | 0.0008 / 0.0008 (MIN-1R 39.73 misses, 19.81 fetches) | OK |
| 20 | scorecard H1a-g25 | 0.0015 / 0.0015 | 0.0015 / 0.0015 (16.42, 9.69) | OK |
| 21 | scorecard H1b-base0 | 0.0004, no fetch | 0.0004, 0 fetches (62.12, 8.20) | OK |
| 22 | scorecard H1b-bypass0 | 0.0007, no fetch | 0.0007, 0 fetches (43.50, 20.39) | OK |
| 23 | scorecard H1c | 0.394 / 0.3499 | 8.19/20.80 = 0.394; 8.19/23.42 = 0.350 | OK |
| 24 | scorecard H2 | 1.0999 [1.089, 1.110] / 1.1828 [1.177, 1.189] | 1.0999 [1.089, 1.110] / 1.1828 [1.177, 1.189] | OK |
| 25 | scorecard H3-g11 | 1.1595 [1.133, 1.185] / 1.092 [1.083, 1.101] | 1.1595 [1.134, 1.185] / 1.0920 [1.083, 1.101] | OK |
| 26 | scorecard H3-g25 | 1.238 [1.227, 1.250] / 1.2698 [1.257, 1.283] | 1.2380 [1.226, 1.250] / 1.2698 [1.257, 1.283] | OK |
| 27 | scorecard H4 | 1.0404 [1.016, 1.065] / 1.0159 [1.013, 1.019] | 1.0404 [1.017, 1.065] / 1.0159 [1.013, 1.018] | OK |
| 28 | scorecard H5 | -0.310 [-0.722, 0.113] failed / 0.259 [0.208, 0.314] held | -0.3105 [-0.729, 0.110] / 0.2591 [0.207, 0.314] | OK |
| 29 | scorecard H6 | 0.9755 [0.958, 0.994] / 1.0759 [1.071, 1.081] | 0.9755 [0.958, 0.994] / 1.0759 [1.071, 1.081] | OK |
| 30 | probe_compare.txt, H7a | 1.0104 / 0.9984 | first 86.6, idle 87.5/87.1/87.1; first 63.4, idle 63.3/63.3/63.2 | OK |
| 31 | H7b | 1.0046 / 1.0016 | 1.0046 / 1.0016 | OK |
| 32 | tables.txt, fetch_table_idle.json | idle table differs on no machine | 0,0,1,1,1 and 0,0,1,1,2 both unchanged | OK |
| 33 | probe1_when.txt | first probe overlapped the download on one machine | b yes, c no | OK |
| 34 | interaction rounds (b) | +0.6 and -1.2 ms | +0.574, -1.195 | OK |
| 35 | dqInterNegBaseZeroMove | fetch-free deployed cache moved 6% between rounds | base/base0 1.0084 -> 0.9436 (6.4%); time 15.05 -> 15.99 ms | OK |
| 36 | scorecard 115 totals | 28 clauses: 11 held, 16 point, 1 failed; validity met | same | OK |
| 37 | H8a/H8b | mean 0.1903, t-interval lower end 0.1346 over five machines | shares 0.218, 0.151, 0.219, 0.231, 0.133; mean 0.1903 [0.1346, 0.2461] | OK (status: see Defect 4) |
| **C** | **Scorer outputs** | | | |
| 38 | wsg_job114.tex | 1,250 range/count/share macros (dz, dq, dp) | all 1,250 identical to mine | OK |
| 39 | wsg_job114.tex | dq/dp clause and status macros used in the text (dqClauses*, dqHEight*, dqInterNeg*, dpInterPosCILow four, dpVerdict mixed, dqVRFailN three, dqVZeroFailN one, dqStartedWord six, dqGated four, dqValidStatus met, dpTabNWord four) | same | OK |
| 40 | Table 5, 114 rows | ratio, Dep. ms, Eq.(1) ms, six ratios with half-widths, interaction | 0.79, 29.7, 14.7, ...; 0.57, 19.3, 8.1, ...; 0.78, 29.7, 14.8, ... (all cells) | OK |
| 41 | Table 5, 115 rows | Core i9-14900K 0.54, 15.1, 5.9, 0.98+-.02, 1.03+-.01, 1.16+-.03, 0.99+-.00, 1.04+-.02, 1.19+-.02, -0.3 [-0.7, 0.1] | same | OK |
| 42 | Table 5, 115 rows | Core Ultra 9 285K 0.54, 19.4, 8.0, 1.08+-.01, 0.99+-.01, 1.09+-.01, 1.00+-.00, 1.02+-.00, 1.13+-.01, 0.3 [0.2, 0.3] | same | OK |
| 43 | Table 5 note | Dep. column | the deployed cache with its fetch table | **Defect 16** (unclear) |
| 44 | prereg/job114.json | 13 hosts, valid 114b, 114c, 114f, 115b, 115c | same | OK |
| 45 | Table 4 fetches-off, 11% | 19 [13,25], 8 [2,14], 36 [13,58], 18 [10,25], 43 [31,56], 3 [-5,12]; T_GPU 25 [18,31], extra 18 [14,23], residual 0 [-11,12] | same (exact 19.0 [13.5, 24.6], ...) | OK |
| 46 | Table 4 fetches-off, 25% | 33 [24,42], 3 [-8,13], 36 [15,56], 23 [19,27], 42 [29,55], -1 [-7,5]; 32 [24,40], 18 [13,23], -8 [-18,2] | same | OK |
| 47 | Table 4 | header "Fetches off (5)" both budgets; sum rows to the gap with dagger | 35.57 + 17.54 + 43.50 + 3.39 = 100.0 | OK |
| 48 | Fig. 2 | five blue lines; medians (11%): MIN-2R 0.88, MIN-1R 0.82, ahead 0.69, Eq.(2) 0.58, Eq.(1) 0.42; diamonds base0/base | same from my times | OK |
| 49 | T_GPU | smallest of 14 profiles, 2.94 ms (105e, C14) on 7 distinct GPUs | 2.9402 ms; 14 profiles; 7 UUIDs | OK |
| **D** | **Scoring changes (jobs 109, 112) and tallies** | | | |
| 50 | scorecard_109 P1-g11 | 109a 0.1154 [0.111, 0.120], c 0.2123, d 0.2105, e 0.2113, f 0.3582 with intervals | 0.1154 [0.111, 0.120], 0.2123 [0.205, 0.219], 0.2105 [0.204, 0.217], 0.2113 [0.206, 0.217], 0.3582 [0.350, 0.365] | OK |
| 51 | scorecard_109 P1-int-g11 | 1.525, 2.966, 2.904, 3.551, 7.487 ms, all held | 1.525 [1.453, 1.598], 2.966 [2.843, 3.096], 2.904 [2.779, 3.041], 3.551 [3.380, 3.737], 7.487 [7.199, 7.797] | OK |
| 52 | scorecard_109 P1-int-g25 | 0.502, 1.022, 0.780, 0.809 ms; none for 109f | same; 109f's 25% round lacks foa/fetch rows | OK |
| 53 | scorecard_112 Q6 | 112a 6.604 [6.345, 6.890], 112b 2.978 [2.820, 3.141] | 6.604 [6.339, 6.883], 2.978 [2.814, 3.149]; job 111 machines below ratio 0.5 (no Q6) | OK |
| 54 | tab_prereg, wsg_job109 | job 109: 61 held, 29 point | 47 + 14, 43 - 14 | OK |
| 55 | tab_prereg, wsg_job112, tab_scorecard_112 | job 112: 24 held, 21 point | 22 + 2, 23 - 2 | OK |
| 56 | tab_prereg | job 115: 11, 16, 1, 0, met | same | OK |
| 57 | tab_prereg all row | 638, 695, 324, 84; 7 met, 2 not | 611 + 27, 695, 323 + 1, 84 | OK |
| 58 | app_prereg Scoring | one draw resamples problems once for every configuration and round, incl. the interaction and job 109's margin | matches `boot_draws`/`inter_draws` | OK |
| 59 | wsg_noise | nzRoundN 143, nzRoundMachines 19, nzRoundMax 6.6, OverOne 7, OverTwo 3 | +14 configurations from 115b/c; max 6.64% (115b base0); +3/+3 | OK |
| 60 | wsg_noise | nzHeldCI 139, nzStillHeld 136, moved three: 115b H3-g11, H4, H6 | 112 + 27 new; widened: only those three move | OK |
| 61 | wsg_noise | rental widening: 68 still, 71 point; narrow 37, other 34 | new clauses: 20 move (8 P1-int on 109a/c/d/e, 109a margin, 112b Q6, 115b H2, H3 x4, H4 x2, H6 x2, 115c H5), 7 stay | OK |
| 62 | app_prereg Rounds and rentals | list of what moved under rental widening | matches item 61 | OK |
| 63 | app_prereg Rounds and rentals | "the derived intervals (capture, job 112's differences, the interaction in ms of jobs 114 and 115)" | noise.py also widens jobs 109/112's interaction and job 109's margin | **Defect 11** |
| 64 | app_prereg Rounds and rentals | "job 115's H3 (1), H4 (1), H6 (1)" | garbled macro format | **Defect 12** |
| **E** | **Text** | | | |
| 65 | abstract | 31-54%; five machines; 19% and 36%; about half left; five fast-link machines, 6%; 0.65 | same | OK |
| 66 | introduction | 39% (post hoc subset); five machines of two registered tests; 19%, 8%, 36% | same | OK |
| 67 | introduction | "reading it once adds most where the link is fast" | larger one-read gain only on the two Ryzen 9 5950Xs (one provider) vs three Intel parts; all five "fast"; post hoc | **Defect 6** |
| 68 | Table 1, fetches-off row | thresholds by test: 1.05/1.07, 1.15/1.20, 0.95/0.98-1.10, >0, pooled >= 10% | match job 114/115 headers | OK |
| 69 | Table 1, fetches-off row | MIN-2R 1.09-1.16x and 1.24-1.36x; Dep-1R 1.02-1.07x; interaction > 0 on four of five (failed on the i9-14900K); two checks failed (all-zero table); mixed; 5 | same | OK |
| 70 | Table 1, fetches-off row | "share 19% (held)" | scorecard 115-H8a/b: held (point) | **Defect 4** |
| 71 | Table 1, FreeToken row | "loose: after a pilot there"; mixed | job 081 ran on job 080's listing (pilot); held on B, failed on S | OK |
| 72 | Table 1, deployed-path row | 9 of 10 (failed), 5 of 5 new (held); mixed | P1-int-g11 held with intervals on all five | OK |
| 73 | Table 1, Few-1R row | verdict "held" with "(one registered: failed)" | 106c-P6/P7 fetchplan/base 0.846 [0.820, 0.877], failed | **Defect 5** |
| 74 | Table 1, row 1 | with the larger rate, 31-52% | 31.2-52.3% | OK |
| 75 | Table 2 | fetches-off machines: 5, two later tests | same | OK |
| 76 | Section 2, Machines | idle reruns within 1.0% of the first | 1.04% (115b), 0.16% (115c) | OK number; **Defect 9** (scope) |
| 77 | Section 2, Statistics | "ran stably ... its rounds agreed within 2%" | V2 tests the deployed cache only; 115b's fetch-free deployed cache 6.4% apart | **Defect 10** |
| 78 | Section 3 | R*: each step's requests served as one set (Appendix N) | matches mosl/cachesim.py | OK |
| 79 | Section 3 | 25 consumer machines, 31-54%, median 47%, quartiles 44-48% | 25; 31.2, 44.3, 47.0, 48.0, 54.0 | OK |
| 80 | Section 3 | envelope: higher on 4 machines, up to 1.16x, range 31-52% | 4; 1.164 (100f); 31.2-52.3 | OK |
| 81 | Section 3 | "the 5 machines added in Section 4 fall at 41-48%" | true for job 109's five; the five fetches-off machines fall at 38.7-49.7% | **Defect 14** |
| 82 | Section 4 intro | 5 machines (two tests), 15 panel, 5 earlier | same | OK |
| 83 | Section 4, fetches off | 1.09-1.16x; 19% (13-25%); 1.02-1.07x; 8%; 36% | same | OK |
| 84 | Section 4, fetches off | four of five; i9-14900K -0.3 [-0.7, 0.1] | same | OK |
| 85 | Section 4, fetches off | 21% on the two 5950Xs (0.78-0.79, one provider: Vast host 716820), -3 to 3% on the others (0.54-0.57), post hoc | 21.1, 20.8; 3.2, -3.4, 2.3; ratios 0.786, 0.783 / 0.566, 0.540, 0.540 | OK |
| 86 | Section 4, fetches off | admits 20.4 vs 8.2; at 25%, 33% and -0.1 to 0.1 ms | 20.39, 8.20; 33.1; -0.09 to 0.13 | OK |
| 87 | Section 4, displacement | four machines with a table; MIN-2R misses 48.1-50.8 vs 43.5; Dep-1R 20.8-32.0 copies, 8.2 admissions | same | OK |
| 88 | Section 4, fetch table on | interaction 19-52% -> -3-21% | 19.1-52.2 -> -3.4 to 20.8 | OK number; **Defect 8** (typesetting) |
| 89 | Section 4, fetch table on | "on those whose table was all zero, it was 21% on both paths" | one machine (114b): 22.2% with the table, 21.1% without | **Defect 7** |
| 90 | Section 5 | MIN's set gains read twice on the five machines | bypass0/base0 1.09-1.16 on all five | OK |
| 91 | Section 7 | "On the one valid machine whose link reads at under a third of the CPU rate, Few-2R paid ... (1.11-1.22x)" | two machines: Core Ultra 9 285K (104a 0.288, 105e 0.281) and Threadripper 9960X (105b 0.321); Few-2R 1.11-1.22, MIN-1R 0.86-0.98 | **Defect 2** |
| 92 | Section 9, rule 2 | 13-23% (five machines, 11%) | 13.3-23.1 | OK |
| 93 | Limitations | without the fetches, five machines | five, of two kinds | OK number; **Defect 17** (scope) |
| 94 | Conclusion | 31-54%; set pays where the read path keeps it | same | OK |
| 95 | app_prereg commit sentence | 107 rentals of jobs 093-115; >= 2.7 s; >= 26 s; seven; four | 107 (101 + 6); job 115 commit->rental 4.0-8.1 s, push->rental 2.0-6.1 s, push->start 36-479 s | OK |
| 96 | app_prereg Job 115 | first gate admits only never-rented GPUs; thresholds from job 114; listing committed first; two offers together; replacements cheapest in a pushed listing; six | same | OK |
| 97 | app_prereg Job 115 | three at the ratio gate "(EPYC 7352, Ryzen Threadripper 3970X; 0.26-0.39)", one at V0; valid i9-14900K and Ultra 9 285K at 0.54 | numbers right; two names for three machines | OK numbers; **Defect 13** |
| 98 | app_prereg Job 115 | 28 clauses 11/16/1; -0.3 [-0.7, 0.1]; +0.6 and -1.2; 6%; five machines 19%, lower end 13%; 0.998-1.010; 0.5%; table changed on no machine | same | OK |
| 99 | app_prereg claim index | job 115, H1-H8: the same, 115x-H7a/b, 115-H8a/b | ids exist in scorecard_115.json | OK |
| 100 | app_limits, hardware | job 115 started six (its limit), four gated, valid ratio 0.54 | same | OK |
| 101 | app_limits, bounds | T_GPU definition; 14 profiles of jobs 069c and 105 on 7 machines | categories excluded: cat_expert_gemv, cat_ec_ctl, cat_ec_copy, cat_ec_wait; cat_quantize included; 7 distinct GPUs | OK |
| 102 | app_limits, decomposition | five machines of two kinds: two 5950Xs (0.78-0.79), three Intel desktop (0.54-0.57) | same | OK numbers; **Defect 15** (column name) |
| 103 | app_wsg, Traces | MIN with bypass: hits first, then misses, admit if used sooner than the furthest resident, which it evicts; served experts may be evicted; checked by exhaustive search | matches cachesim.py; my exhaustive search agrees on 300 traces; keeping served experts never reads fewer | OK; **Defect 18** (warm-up case unstated) |
| 104 | tab_failures, job 115 row | 28/11/16/1; interaction -0.3, [-0.7, 0.1], i9-14900K; rounds +0.6/-1.2; 6% | same | OK |
| 105 | wsg_numbers2 | 160 rentals, 96 offers, jobs 058-115, $115.3 | 160, 96, 115.251 | OK |
| 106 | wsg_regtime | rtRentals 107 | 107 | OK |
| 107 | refs_wsg(_ieee).bib | atsinfer2026 title and authors | arXiv 2607.10183: Liu, Ye, Li, Li; same title | OK |
| **F** | **Builds** | | | |
| 108 | paper.pdf | 0 errors; Conclusion ends p. 10; References start p. 10; Tables 1-6 and Figs 1-5 before the references | 0 errors, 0 undefined; as stated (47 pages); rebuild text identical | OK |
| 109 | ieee-paper.pdf | last page's reference columns balanced (IEEEtriggeratref{26}) | left column ends at 356.5 pt, right at 437.2 pt (80.7 pt, about 9 lines) | **Defect 3** |
| 110 | ieee-supplement.pdf / ieee-paper.pdf | cross-references both ways resolve | 0 undefined references in both (rebuild), no "??"; 30 multiply-defined citation labels from xr, pre-existing and documented | OK |
| 111 | supplement.pdf | scorecard with job 115's table | Table S21, 28 clauses, statuses as scorecard_115.json; 0 errors | OK |

111 items; items 38 and 45-46 cover 1,250 and 20 generated values respectively.

## Defects

1. **Amendment timing, app_prereg.tex (Job 115 paragraph) and the job 115 header.** "One more machine was added by an
   amendment pushed while the first two ran their first rounds, before any of their timed results existed" is wrong. The
   amendment was pushed at 05:30:42Z. By the machines' own epoch stamps (stdout.log), 115c ran its first round from
   05:05:55Z to 05:21:36Z (its rows and the V1/V3 lines were written) and was nine minutes into its second round; 115b had
   been in its first round since 05:26:30Z. "The first two" also reads as a and b, the two that started together, and a
   had already stopped. The header's AMENDMENT text makes the same claim; it cannot be edited now, so the paper should
   correct it. **Fix:** "One more machine was added by an amendment pushed while the two running machines were measuring (b in
   its first round, c in its second), before either had returned any result (a machine's results come back only when it
   finishes; the header's wording, 'before any of their timed results existed', is wrong for c). Its text reached the header
   17 s after that machine was rented; the machine booted from that commit."
2. **Section 7, main_body.tex l. 321-322.** "On the one valid machine whose link reads at under a third of the CPU rate,
   Few-2R paid instead of MIN-1R (1.11-1.22x)" is wrong. `\jkSlowBypassPlan` spans two machines under ratio 1/3, and my
   recomputation from the raw rows gives the same values:
   - the Core Ultra 9 285K, GPU-b8316d93, in 104a (ratio 0.288) and 105e (0.281): Few-2R 1.113/1.217 and 1.112/1.215;
   - the Threadripper 9960X, GPU-9eba03f6, in 105b (0.321): Few-2R 1.114/1.159.

   MIN-1R ran 0.86-0.98 on all three launches, and `prereg/sumlaw_outcome_105.md` calls 105b "the second slow-link machine".
   **Fix:** "On the two machines whose link reads at under a third of the CPU rate (ratios 0.28-0.32, three launches), Few-2R
   paid instead of MIN-1R (1.11-1.22x)." Take the count from the same hosts as `jkSlowBypassPlan`.
3. **ieee-paper.pdf, last page.** The reference columns are not balanced: the left column ends at 356.5 pt and the right at
   437.2 pt, a difference of 80.7 pt (about nine lines). The \IEEEtriggeratref{26} trigger predates HEAD's abstract trim.
   **Fix:** set `\IEEEtriggeratref{27}` in ieee-paper.tex and rebuild. In my rebuild this leaves 26.9 pt (three lines;
   one reference is three lines) and 10 pages; 28 gives the same imbalance on the other side.
4. **Table 1, fetches-off row: "share 19% (held)" against the scorecard.** `scorecard_115.json` scores 115-H8a and H8b
   "held (point)" (no interval). The Job 115 paragraph counts "the pooled prediction" among the 16 point clauses. Job 109's
   pooled clauses, by contrast, carry t-intervals over machines and are scored "held" (P3-g25, P6-g11/g25, P7-mean). H8a's own
   t-interval, [0.135, 0.246], lies above 0.10. **Fix (preferred):** give 115-H8a its t-interval in
   `clause_pooled` (as `job109.tint`). That scores it "held" and changes these values:

   | Quantity | Now | After |
   |---|---|---|
   | Job 115 held / point | 11 / 16 | 12 / 15 |
   | Overall held / point | 638 / 695 | 639 / 694 |
   | `\nzPooled` | 4 | 5 |

   In the Job 115 paragraph, write "15 on the point estimate (the counter checks, the probe and the interval's lower end)".
   **Alternative:** keep the scoring and write "share 19% (held (point))" in Table 1.
5. **Table 1, Few-1R row: verdict "held".** The row itself reports "(one registered: failed)": 106c (EPYC 7302,
   unsteady) failed its registered clauses 106c-P6/P7, with fetchplan/base 0.846 [0.820, 0.877]. Table 1's note defines
   *held* as meeting the threshold on every machine and *mixed* as held in part and failed in part. **Fix:** set the verdict
   to "mixed". Alternatively, scope the claim column to stable machines and state that the registered "> 1 on every machine"
   failed on one unsteady machine.
6. **Introduction, main_body.tex l. 76: "and reading it once adds most where the link is fast."** This is mis-scoped. The
   larger one-read gains (interaction 21% of the gap against -3 to 3%; Dep-1R 12.6-12.9% against 2.5-6.6%) are on the two
   Ryzen 9 5950Xs of one provider, against three Intel desktop parts. All five machines are "fast" by the paper's own
   0.5 line, and the split is post hoc. **Fix:** "...and reading it once added most on the two Ryzen 9 5950Xs, the
   machines with the highest link-to-CPU ratio (post hoc; two kinds of machine)."
7. **Section 4, main_body.tex l. 247-248: "on those whose table was all zero, it was 21% on both paths."** Only one machine
   (114b) had an all-zero table, and its deployed-path interaction is 22.2% of the gap (fetch-free 21.1%). **Fix:** "on the
   one machine whose table was all zero (a Ryzen 9 5950X), it was 22% with the table and 21% without." This needs a dp macro
   for the deployed-path share on the NoTab machine.
8. **Section 4: "shrank the interaction from 19-52% of the gap to -3-21%"** typesets as "−3–21%". **Fix:**
   `\dpShInteractionZeroTabLowMin{} to \dpShInteractionZeroTabLowMax\%` ("−3 to 21%"), as the paragraph above does.
9. **Section 2, Machines: "Rerun three times on the idle machine in the last test, its best reading came within 1.0% of the
   first's."** This is mis-scoped and unclear. "The last test" is undefined in Section 2. The evidence is two machines, and
   only one (115b, +1.0%) had a first probe that overlapped the download. **Fix:** "In job 115, on its two valid machines,
   three reruns on the idle machine came within 1.0% of the first run's best reading (the first had overlapped the model
   download on one of them)."
10. **Section 2, Statistics: "a machine ran stably when its outputs matched the reference and its rounds agreed within
    2%."** The round check (V2) compares only the deployed cache. On the valid Core i9-14900K, the fetch-free deployed
    cache's two rounds differed by 6.4% (15.05 against 15.99 ms). **Fix:** "...and its deployed cache's time per token agreed
    within 2% between rounds."
11. **app_prereg.tex, Rounds and rentals, l. 78-79.** "the derived intervals (capture, job 112's differences, the
    interaction in ms of jobs 114 and 115) accordingly" omits what `noise.py` now also widens: the interaction in ms of jobs
    109 and 112 (P1-int, Q6) and job 109's margin. These clauses are listed among the moves two sentences later. **Fix:**
    "(capture, job 112's differences, job 109's margin of MIN-1R, and the interaction in ms of jobs 109, 112, 114 and 115)".
12. **app_prereg.tex, Rounds and rentals: "job 115's H3 (1), H4 (1), H6 (1) on the Core i9-14900K".** These are the
    `\nzMovedKinds` counts printed raw. **Fix:** "job 115's H3 at 11%, H4 and H6 on the Core i9-14900K" (write the macro as
    prose, or drop the counts).
13. **app_prereg.tex, Job 115: "three stopped at the ratio gate (EPYC 7352, Ryzen Threadripper 3970X; ratios
    0.26-0.39)".** Two names for three machines, because the macro de-duplicates them. **Fix:** "two Ryzen Threadripper 3970Xs
    (0.39) and an EPYC 7352 (0.26)". In `job114.py`, count the CPU models instead of using a set.
14. **Section 3: "the 5 machines added in Section 4 fall at 41-48%".** Section 4 now adds two sets of five. The range holds
    for job 109's new machines, but the five fetches-off machines fall at 38.7-49.7% (115b 38.7%, 114f 49.7%). **Fix:** "the
    5 new machines of Section 4's first registered test fall at 41-48%".
15. **app_limits.tex, decomposition: "The CPU-only column rests on five machines".** Table 4's column is now headed
    "Fetches off". **Fix:** "The fetches-off column of Table 4 rests on five machines of two kinds..."
16. **Table 5 note: "Dep.: the deployed cache".** The new ms column sits next to off/on ratios, and its value is the
    deployed cache with its fetch table. **Fix:** "Dep.: the deployed cache with its fetch table (on)".
17. **Limitations, main text: "without the in-step fetches, five machines".** It does not say what Appendix L and Section 4
    say: two kinds, two Ryzen 9 5950Xs of one provider and three Intel desktop parts at ratio 0.54-0.57. **Fix:** "without the
    in-step fetches, five machines of two kinds". Check that the Conclusion still ends on page 10.
18. **app_wsg.tex, Traces: MIN's semantics.** "each admitted only if it is used again sooner than the resident used
    furthest ahead" omits the warm-up case: while a slot is free, `cachesim.py` admits any miss that is used again. **Fix:** add
    "(while a slot is free, any miss used again is admitted)".
