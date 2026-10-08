# Number check 22 (8 October)

## Scope

Everything that changed in `paper/`, `scripts/` and `prereg/` between commit 7ddf613 and HEAD (210d7ab):

1. Job 113, new: the header of `jobs/113_heldset@vast.sh` (gpu branch, commit f174e0a) with its predictions H1-H6, the raw results in `results/113b_heldset@vast` and `results/113d_heldset@vast`, `scripts/job113.py`, `paper/wsg_job113.tex`, `paper/tab_job113.tex`, `paper/tab_scorecard_113.tex` and `prereg/scorecard_113.json`.
2. Where job 113 entered the paper: the abstract, the introduction, Table 1, the Section 4 paragraphs, the Section 5 qualifier, builder rule 2, the Limitations, the Conclusion, the Job 113 paragraph of `app_prereg.tex`, the job 113 rows of `tab_failures.tex` and `tab_prereg.tex`, `tab_names.tex`, the note of `tab_configs.tex`, and the derived counts (`wsg_robust.tex`, `wsg_hostdep.tex`, `wsg_numbers2.tex`, `wsg_regtime.tex`).
3. The clarity edits: Sections 2, 3, 6 and 7, the "What is left" paragraph (with the Eq. (3) details moved to `app_limits.tex`), and the timing-control table moved to `app_prereg.tex`.
4. Production fixes: the axis label of Fig. 3, the figure fonts, the supplement's cross-references, the order of Tables 10 and 11, and the IEEE PDF keywords.
5. Builds: `paper.pdf`, `supplement.pdf`, `ieee-paper.pdf` and `ieee-supplement.pdf`.

## Method

- **Registration.** I compared the gpu-branch reflog and `prereg/gpu_pushes.json` with `gpu/vast_ledger.json` and the manifests' `start_utc`. Commit f174e0a is at 18:18:48Z and the GitHub push at 18:18:53Z. Both rentals were created at 18:19:25Z, and the jobs started at 18:20:30Z (113d) and 18:21:05Z (113b). No later commit touches `jobs/113*`.
- **Gates.** I checked the gates from each host's `v0.txt`, `gate.txt`, `cores.txt`, `fetch_table_law_gptoss.json` and `validity.txt`. I checked both GPU UUIDs against `known_gpu_uuids.txt` and `job109_uuids.txt` as they stood at f174e0a.
- **Job 113 numbers.** My own code (`scratchpad/nc22/recompute113.py`) reads every `ec_g_C*_r*A.jsonl` row and every `st_g_*.json` counter. I took definitions from the job header and `scripts/job109.py` only:
  - A ratio X/Y is Y's mean time per token over the 20 problems divided by X's, within one process, with the geometric mean taken over rounds.
  - Intervals come from my own paired bootstrap over problems (4,000 draws, a different seed).
  - Counters are per decode step and averaged over rounds; host reads are misses plus admits.
  - I scored the clause statuses under the paper's rule.
- **Few-1R over 15 machines.** I recomputed fetchplan/base from the raw rows of all 18 launches that feed it (jobs 104-107, 112 and 113). I grouped them by GPU UUID and took the geometric mean and the t-interval on the log scale. For the intention-to-treat figure I added the four unsteady launches (106c, 106d, 107c and 107e) from their raw rows.
- **Cost and timing.** The cost, rental and offer counts come from the ledger directly. I checked the registration timing count with the scripts' rule (jobs 093-113).
- **Paper text.** I expanded every macro in place (`scratchpad/nc22/expand.py`), so that I read each sentence with its numbers.
- **Builds.** I rebuilt all four PDFs from `git archive HEAD` in the scratchpad and diffed the extracted text against the committed PDFs: no difference in any of the four. I read page boundaries and float placement with `pdftotext`, fonts with `pdffonts`, metadata with `pdfinfo`, and cross-reference labels from the `.aux` files.

## Items checked

| # | Location | Claim | My value | Verdict |
|---|---|---|---|---|
| 1 | gpu f174e0a, `gpu_pushes.json`, ledger | header committed and pushed before the rentals | commit 18:18:48Z, push 18:18:53Z, rentals 18:19:25Z (both), starts 18:20:30Z and 18:21:05Z; header not edited after | OK |
| 2 | ledger, V0 | 113b offer 54573925 and 113d offer 54807524 are new; GPUs are in no earlier launch | neither offer is in the ledger before; neither UUID is in either list at f174e0a | OK |
| 3 | 113b gates | V0 to V3 | 3 GB used; 1 NUMA node; 16 cores; 1694 GB/s; 150 GB; ratio 0.7443; loss 0.1903; V2 0.60%; V3 at most 0.46% | valid |
| 4 | 113d gates | V0 to V3 | 4 GB used; 1 NUMA node; 23 cores; 1692 GB/s; 150 GB; ratio 0.5557; loss 0.1902; V2 0.11%; V3 at most 0.46% | valid |
| 5 | `tab_job113`, 5950X | Dep. CPU only 0.98±.01 | 0.9806 [0.9741, 0.9878] | OK |
| 6 | `tab_job113`, 5950X | MIN-2R CPU only 1.12±.01 | 1.1218 [1.1092, 1.1336] | OK |
| 7 | `tab_job113`, 5950X | Few-2R CPU only vs Dep. CPU only 1.26±.01 | 1.2594 [1.2462, 1.2719] | OK |
| 8 | `tab_job113`, 5950X | Few-2R CPU only vs Dep. 1.23±.01 | 1.2350 [1.2279, 1.2422] | OK |
| 9 | `tab_job113`, 5950X | Few-1R 1.40±.01 | 1.3973 [1.3869, 1.4084] | OK |
| 10 | `tab_job113`, 5950X | Few-2R misses 55.6 / 43.0 | 55.64 / 42.99 | OK |
| 11 | `tab_job113`, 285K | 1.07, 1.09, 1.23, 1.32, 1.34 (each ±.01) | 1.0732, 1.0945, 1.2306, 1.3206, 1.3445; half-widths 0.005-0.015 | OK |
| 12 | `tab_job113`, 285K | misses 54.7 / 43.0 | 54.71 / 42.99 | OK |
| 13 | `tab_job113` | ratios 0.74 and 0.56 | 0.7443 and 0.5557 | OK |
| 14 | `wsg_job113` | Few-2R (with fetches) vs Dep. at 11%: 1.03 | 1.0295, 1.0280 | OK |
| 15 | `wsg_job113` | MIN-2R CPU only vs Dep.: 1.10-1.17 (11%), 1.27-1.30 (25%) | 1.1001, 1.1745; 1.2674, 1.2991 | OK |
| 16 | `wsg_job113`, 25% | base0/base 0.93-1.02; bypass0/base0 1.28-1.36; bypassplan0/base0 1.34-1.42 | 0.9347, 1.0174; 1.2768, 1.3560; 1.3358, 1.4178 | OK |
| 17 | `wsg_job113`, 25% | bypassplan0/base 1.33-1.36; fetchplan/base 1.29-1.43; bypassplan/base 1.11-1.13 | 1.3251, 1.3591; 1.2934, 1.4275; 1.1125, 1.1272 | OK |
| 18 | `wsg_job113`, 11% counters | base 59.7-60.0 misses, 3.3-3.6 admits, 21.3-27.2 fetches | 59.69, 59.98; 3.34, 3.63; 21.34, 27.24 | OK |
| 19 | `wsg_job113`, 11% counters | base0 62.1 + 8.2; bypass0 43.5 + 20.4; bypassplan 54.7-55.6 + 6.2-7.4 with 18.6-24.3 fetches; bypassplan0 43.0 + 12.2; fetchplan 40.7, 11.8 fetches | identical on both hosts where fetch=0; all as stated | OK |
| 20 | `wsg_job113`, 25% counters | base 29.6-29.8, 4.4-4.9, 8.0-9.7; base0 30.7 + 9.1; bypass0 17.6 + 10.0; bypassplan 23.8-24.6 + 5.0-5.5, 5.7-7.2; bypassplan0 17.6 + 7.3; fetchplan 17.3, 7.0 | as stated | OK |
| 21 | H1 (four clauses) | bypassplan0 misses at most 0.88x bypassplan's | 0.7726, 0.7180 (5950X); 0.7858, 0.7424 (285K): held (point) | OK |
| 22 | H2 (two) | host reads at most 0.90x base0's | 0.7845 on both: held (point) | OK |
| 23 | H3 (two) | base0/base within [0.80, 1.00] | 0.981 held; 1.073 failed (285K) | OK |
| 24 | H4 (two) | bypassplan0/base0 at least 1.10 | 1.259 and 1.231, both held | OK |
| 25 | H5 (two) | bypass0/base0 at least 1.03 | 1.122 and 1.095, both held | OK |
| 26 | H6 (two) | fetchplan/base at least bypassplan0/base + 0.05 | 0.162 [0.153, 0.171] held; 0.024 [0.017, 0.030] failed (285K) | OK |
| 27 | `scorecard_113.json`, `tab_scorecard_113` | 15 clauses: 7 held, 6 held (point), 2 failed | 14 per-machine clauses (6 held, 6 point, 2 failed) plus the count clause (held) | OK |
| 28 | `wsg_job113` | hsMissRatio 0.77-0.79 / 0.72-0.74; hsReadsRatio 0.78 / 0.63; hsReadsRstar 1.44 / 1.63 | as stated | OK |
| 29 | `wsg_job113` | hsOneReadMargin 0.02-0.16 (11%), -0.07 to 0.10 (25%) | 0.0238, 0.1623; -0.0657, 0.1023 | OK |
| 30 | `wsg_job113` | hsRatioRng 0.56-0.74; hsN 2; clause macros 14 / 6 / 6 / 2; failed on the 285K | as computed | values OK; see defect 11 |
| 31 | Abstract | 35% of the gap (registered test) | dmNewTogetherLow = 35 | OK |
| 32 | Abstract | MIN's set read twice 1.23-1.26x without fetches; reading once adds only 0.02-0.16 | the 1.23-1.26 is the fewest-admission set (the greedy set: 1.09-1.12); the margin is in the with-fetch yardstick (0.02-0.17 in the fetch-free one) | defects 1, 2 |
| 33 | Abstract | "In the engine as deployed, the experts MIN keeps pay only when each admitted expert is read once" | the scope (11%, fast links, greedy) has been dropped; Few-2R with the fetch table gains 1.03x at 11% and 1.11-1.13x at 25% | defect 1 |
| 34 | Intro | doing both closes 39% on 13 fast-link panel machines | 38.8% (`decomp_measured.json`, 13 of 15 with ratio at least 0.5) | OK |
| 35 | Intro | 35% on 5 machines; second test 1.23-1.26x, adds only 0.02-0.16 | as in item 32 | defect 2 |
| 36 | Intro | "reading each admission once pays most on the machines we tested" | 104a (ratio 0.29): Few-2R 1.113 / 1.217 against Few-1R 1.035 / 1.059; 104b at 25%: 1.206 against 1.175; 113d at 25%, fetch-free Few-2R ahead | defect 3 |
| 37 | Intro | "recover at most 6% ... on those machines" | the antecedent is now ambiguous | defect 20 |
| 38 | Table 1, 2x2 row | as deployed; 9 of 10, 5 of 5, 35% | unchanged values | OK |
| 39 | Table 1, control row | misses 0.97-0.98x, 0.03-0.05 slower, failed on 5 and 7 of 8 | tc macros | OK |
| 40 | Table 1, job 113 row | registered thresholds and results; Mach. 2 | as in items 21-26 | OK (defect 12 for the index) |
| 41 | Table 1, Few-1R row | 1.26x on 15 stable machines, 7 registered | 15 UUIDs; geometric mean 1.2568 | OK |
| 42 | Table 1, share row | 31-54% on 25 consumer machines | unchanged macros | OK |
| 43 | Sec. 4, 2x2 paragraph | 0%, 0%, 33% (39% on 13), 35%; 9 of 10; 5; 21% at 25% | as stated | OK |
| 44 | Sec. 4, new paragraph | Few-2R fetched 18.6-24.3 and missed 54.7-55.6, against Few-1R's 40.7 | 18.65 / 24.35; 54.71 / 55.64; 40.73 | OK |
| 45 | Sec. 4, new paragraph | 0.77-0.79 as many misses; 0.78 host reads; 1.23-1.26x; MIN-2R 1.09-1.12x; held | as stated | OK |
| 46 | Sec. 4, new paragraph | one read paid 0.02-0.16 more; at 25% on the 285K the two-read arm was faster | 0.024 / 0.162; -0.066 at 25% (one round) | OK; the yardstick is unstated (defect 2) |
| 47 | Sec. 4, new paragraph | deployed cache moved 0.98-1.07x (registered: at most 1.00) | the registered band is [0.80, 1.00] | defect 5 |
| 48 | Sec. 4, new paragraph | "a two-read path that keeps the set gets most of what reading once gets" | fewest-admission set: 59% (5950X) and 93% (285K) of Few-1R's gain over the deployed cache at 11%; the greedy set read once did not run in job 113 | defect 4 |
| 49 | Sec. 4, What is left | 15% closed by reading ahead; about half left | 15; 52% | OK |
| 50 | Sec. 4, What is left | residual 35-47% and 31-36% | 11% values; the "at 11%" qualifier was dropped | defect 19 |
| 51 | `app_limits` | 83-99%, 73-81%, median 45% against Eq. (3); 8 server launches, three passed | moved intact | OK |
| 52 | Sec. 5 | Few-1R beat the deployed cache on all 15, geometric mean 1.26x, 7 registered, two RTX 4090s, two server processors; lost on two (0.14 and 0.21) | 1.2568 [1.1858, 1.3320]; min 1.0276 | OK |
| 53 | Sec. 5 qualifier | "with the in-step fetches off, though, the two-read path keeps most of the gain" | two machines, gpt-oss 11%, fewest-admission set only; the sentence it qualifies is about O3-O5 at each model's smallest budget | defect 6 |
| 54 | `wsg_robust` | rbPlanMachines 15, mean 1.26, Lo/Hi and TLo/THi 1.19 / 1.33, min 1.03, implied eight | 15; 1.2568; 1.1858 / 1.3320; 1.0276; 15 - 7 = 8 | OK |
| 55 | `wsg_robust` | AllN 19, AllMean 1.21 [1.13, 1.30] | 1.2123 [1.1326, 1.2976] | OK |
| 56 | `wsg_hostdep` | hdHosts 31 (from 29), hdFetchPlanMax 1.43 | adds labels 113b and 113d; max 1.4275 (113b, 25%) | OK |
| 57 | `wsg_numbers2` | 147 rentals, 86 offers, jobs 058-113, $107.5 | 147; 86; 107.48 | OK |
| 58 | `wsg_regtime` | 94 rentals of jobs 093-113; minima 2.7 s and 26 s unchanged | 94; job 113: 37 s commit to rental, 97 s and 132 s push to start | OK |
| 59 | `tab_prereg` | 113: 7 / 6 / 2 / 0; all: 595 / 685 / 323 / 84 | column sums 595 / 685 / 323 / 84 | OK |
| 60 | `tab_failures`, job 113 header | "14 clauses; 6 held with an interval, 6 held (point), 2 failed" | Table 10 and the supplement say 15 and 7 | defect 11 |
| 61 | `tab_failures`, job 113 rows | 0.98-1.07, failed on the 285K; 0.02-0.16, failed on the 285K | as stated | OK |
| 62 | `tab_failures`, job 112 T2 row | "The early copies admitted and read more" | the supporting numbers (11.7-11.8 against 6.2-7.4 admissions; 4-7% more host reads) are no longer anywhere in the paper | defect 18 |
| 63 | `app_prereg`, Job 113 paragraph | offers skipped; both machines new and valid, ratio 0.56-0.74; 0.77-0.79 / 0.72-0.74; 1.23-1.26 / 1.09-1.12 held; 285K failures; at 25% 1.33-1.36 against 1.29-1.43 | all values OK | values OK; defects 9, 10, 23 |
| 64 | `app_prereg` | "14 for job 113" clause-table rows | supplement Table 19 has 15 rows | defect 11 |
| 65 | `app_prereg`, claim index (Table 9) | every pre-specified claim of Table 1 indexed | no row for the job 113 claim | defect 12 |
| 66 | `app_prereg`, Job 112 paragraph | T3 held but carries no evidence | follows from T3's definition once Few-1R > Few-2R > Few-2R-early | OK |
| 67 | Sec. 2, statistics | median 0.2% between rounds and 0.5% between rentals; "app:prereg rescores the registered clauses with both" | `noise.py` covers jobs 109, 111 and 112 only; rental widening would move 113b-H3 (upper bound 1.04) and 113d-H5 (lower bound 1.022) to held (point) | defect 16 |
| 68 | Sec. 2 / Sec. 3 | host-bound exceptions now in Sec. 3; copy published two steps on | moved intact | OK |
| 69 | Sec. 3 | where systems stand | the 41-48% share of job 109's five machines has been dropped from the paper | defect 17 |
| 70 | Sec. 6 | 4-6%; 1.007-1.024x; 6%; 1.57-1.76 R*; up to 26% | as before | OK; layer-ahead loss magnitude dropped (defect 17) |
| 71 | Sec. 7 | Spearman 0.89 over 19 (interval kept in `app_more`); five RTX 4090s at 0.37-0.77 within 7% and 5%; 5 of 18 cells up to 18%; 1.11-1.22x | as stated | OK |
| 72 | Sec. 7 | "Predicting no change would have missed by up to 46%" | 46% at 11%, 69% at 25% | defect 15 |
| 73 | Sec. 7 | the registered interaction on the two fast-link RTX 4090s; every interval inside the band | dropped from the paper | defect 17 |
| 74 | Builder rule 2 | with fetches MIN's set read twice closes none; without them 1.23-1.26x; one read adds 0.02-0.16 | "none" holds at 11% only (21% at 25%); 1.23-1.26 is the fewest-admission set; two machines | defect 7 |
| 75 | Limitations | "the control without them ran on two machines" | scoped | OK |
| 76 | Conclusion | "what is cached pays only when the read path keeps it" | rests on two machines; not scoped | defect 8 |
| 77 | `app_extra` (supplement) | the 2x2 interacts "because MIN caches 4.6 times as many experts ... each is read twice" | still blames the second read | defect 14 |
| 78 | `app_foresight` | MIN-2R's excess misses explained by copy lag (d = 2.8-4.7) | job 113: fetch=0 cuts Few-2R's misses 55.6 to 43.0 at 11%; the fetch table is not mentioned | defect 21 |
| 79 | `tab_names`, `tab_configs` note | base0, bypass0, bypassplan0; "CPU only" | match the header's configuration strings | OK |
| 80 | Fig. 3 | axis label | "speed relative to the deployed cache on the same machine"; legible at 300 dpi; interval note moved to the caption | OK |
| 81 | Figures | no Type 3 fonts | none in the 12 included figures or in the four PDFs; `audit_sol.pdf` embeds Latin Modern as CID Type 0C (OT), with a pdffonts type mismatch warning | OK for Type 3; defect 22 |
| 82 | `supplement.pdf` | no reference prints a file name | none ("paper.pdf" appears nowhere) | OK; defect 13 (number collisions) |
| 83 | `supplement.tex` | the list of scorer scripts | omits `scripts/job113.py` | defect 24 |
| 84 | `paper.pdf` appendix | tables in number order | Table 10 p. 19, Table 11 pp. 20-24, Table 12 p. 25 | OK |
| 85 | `ieee-paper.pdf`, `ieee-supplement.pdf` | PDF keywords | "Caching, foresight, mixture of experts, offloading, performance bounds", same as Index Terms | OK |
| 86 | `paper.pdf` | main text within 10 pages | 45 pages; Conclusion ends p. 10; References start p. 11; main floats on pp. 2-9 (Tables 1-6, Figures 1-5); 0 errors, 0 undefined; rebuild from HEAD identical in text | OK |
| 87 | IEEE builds | cross-references both ways | 0 errors, 0 undefined; paper to supplement: Appendix C-N and Tables S22-S24 match the supplement's labels; supplement to paper resolves; rebuild identical | OK |
| 88 | `supplement.pdf` | build | 0 errors; rebuild identical | OK |
| 89 | Whole paper | nothing still says the second read holds MIN's set back | main text and appendices clean except `app_extra` (and, in part, `app_foresight`) | defects 14, 21 |
| 90 | Scripts | `job113.py`, `fig_hostdep.py`, `robustness.py`, `scorecard_auto.py`, `tab_prereg.py`, `reg_timing.py` | definitions consistent with the header; the inputs added are the right hosts and cells | OK |

90 rows. Counting each number or status inside the grouped rows, about 230 values were checked.

## Defects

1. **Abstract: scope dropped and MIN's set mislabelled.**
   - "In the engine as deployed, the experts MIN keeps pay only when each admitted expert is read once, not twice" lost the scope that the previous abstract and Table 1 carry: the smallest budget, fast links, the greedy schedule. Few-2R, MIN's fewest-admission set read twice, gains 1.03x at 11% and 1.11-1.13x at 25% with the fetch table.
   - "MIN's set read twice runs 1.23-1.26 times" quotes the fewest-admission set, while "MIN's set" elsewhere means the greedy one, which ran 1.09-1.12x.
   - The second test is not scoped to two machines.
   - **Fix:** "In the engine as deployed, at that budget and where the link reads at least half as fast as the CPU, the experts MIN's greedy schedule keeps pay only when each admitted expert is read once, not twice: then they close 35% of the gap, in a test registered before its machines were rented. A second registered test, on two machines, shows why: the engine's in-step fetches push MIN's set out of the cache. With them off, MIN's fewest-admission set read twice runs 1.23-1.26 times as fast as a deployed cache without them (its greedy set 1.09-1.12), and reading each admission once adds 0.02-0.17 more."
2. **Abstract, introduction (lines 30-32) and Sec. 4: "reading once adds (only) 0.02-0.16".**
   - The margin is fetchplan/base minus bypassplan0/base, in units of the deployed cache with its fetches. The number beside it (1.23-1.26) is against the fetch-free cache; in that yardstick the margin is 0.02-0.17.
   - "Only" understates the 5950X, where reading once was 1.13x faster than the fetch-free two-read arm (more than three times the registered 0.05).
   - **Fix:** drop "only". Either quote 0.02-0.17 in the fetch-free yardstick, or keep 0.02-0.16 and add "of the deployed cache's speed". In the introduction, also scope the sentence ("on two machines") and say "MIN's fewest-admission set".
3. **Introduction (lines 38-40): "reading each admission once pays most on the machines we tested" is contradicted by the paper's own data.**
   - On 104a (ratio 0.29), Few-2R ran 1.113 / 1.217x against Few-1R's 1.035 / 1.059x.
   - On 104b at 25%: 1.206 against 1.175.
   - On the 285K at 25%, the fetch-free two-read arm beat Few-1R.
   - **Fix:** "reading each admission once pays most where the link is fast, at the smallest budget".
4. **Sec. 4, last sentence of "The in-step fetches...".** "The interaction above is therefore a property of this engine's two-read path, not of MIN's set: a two-read path that keeps the set gets most of what reading once gets" generalises from two machines. The interaction concerns the greedy set, whose read-once arm did not run in job 113; the "most" (59% and 93% of Few-1R's gain) is the fewest-admission set's.
   - **Fix:** "On these two machines, then, the interaction above belongs to this engine's two-read path with its fetch table rather than to MIN's set: without the fetches, the fewest-admission set read twice got 59-93% of what reading it once gained at 11%."
5. **Sec. 4: "(registered: at most 1.00; failed on the Core Ultra 9 285K)" misstates H3.** The registered band was [0.80, 1.00], as Table 1, `tab_failures` and the header say.
   - **Fix:** "(registered: 0.80-1.00; ...)".
6. **Sec. 5 qualifier: "with the in-step fetches off, though, the two-read path keeps most of the gain" is unscoped.** It comes from two other machines, at gpt-oss 11% only, and for the fewest-admission set only. The sentence it qualifies is about O3-O5 at each model's smallest budget.
   - **Fix:** "; on two other machines at gpt-oss 11%, with the in-step fetches off, MIN's fewest-admission set read twice kept 59-93% of what reading it once gained (Sec. 4)".
7. **Builder rule 2.**
   - "With them, MIN's set read twice closes none of the gap" holds at 11% only; at 25%, MIN-2R closes 21%.
   - "Without them it runs 1.23-1.26x" is the fewest-admission set on two machines; the greedy set ran 1.09-1.12x.
   - **Fix:** "...at gpt-oss 11%, with them, MIN's set read twice closes none of the gap in our engine; on two machines without them, MIN's fewest-admission set read twice ran 1.23-1.26x a cache without them (the greedy set 1.09-1.12x), and reading each admission once added 0.02-0.16 of the deployed cache's speed, depending on the machine."
8. **Conclusion: "In our engine, what is cached pays only when the read path keeps it" replaces a claim tested on 15 + 5 machines with one that rests on two, and does not say so.**
   - **Fix:** "In our engine as deployed, what is cached pays only when each admitted expert is read once; on two machines, a read path that keeps the set let it pay read twice as well, and policies without foresight recover little of the rest."
9. **`app_prereg` Job 113: "With two machines the job is a pair of single-machine results, as its header says, not a test over a class" misquotes the header.** The header says single-machine results apply "with fewer than two valid hosts". Two hosts met its minimum, and the scorecard scores "at least two valid hosts" as held.
   - **Fix:** "Two valid machines met the header's minimum for a test; two machines cannot speak for a class."
10. **`app_prereg` "and the credit left allowed no third" and `app_limits` "Job 113 ran on two machines, all the remaining credit allowed" are unsupported, and they contradict the registered stop rule.**
    - The header stops "until two hosts are valid". Both were valid, so no third host would have started.
    - The ledger puts the two rentals at $1.69 of the $4.14 the header records.
    - **Fix:** "the job stopped, as registered, once two machines were valid". In `app_limits`: "Job 113 ran on two machines, the number its registration required."
11. **Job 113's clause counts are inconsistent.**
    - `tab_failures` prints "14 clauses; 6 held with an interval, 6 held (point), 2 failed".
    - `app_prereg` says "14 for job 113".
    - Table 10, `prereg/scorecard_113.json` and the supplement's Table 19 have 15 clauses, 7 of them held, counting the pooled "at least two valid hosts" clause. Jobs 109 and 112 count their pooled clauses in these macros.
    - **Fix:** in `scripts/job113.py`, `macros()`, count all clauses (`pm = cl`), giving 15 / 7 / 6 / 2, and regenerate. Alternatively, write "14 per-machine clauses and one count; 7 held ...".
12. **The claim index (Table 9, `app_prereg`) has no row for the new Table 1 claim** "The in-step fetches, not the second read, keep MIN's set from paying read twice".
    - **Fix:** add the row "job 113, H1-H6: `113x-H1-g11/g25`, `113x-H2`, `113x-H3`, `113x-H4`, `113x-H5`, `113x-H6`".
13. **`supplement.pdf`: with the file name gone, references into the paper collide with the supplement's own numbers.**
    - The supplement's Table 3 (grid) and the paper's Table 3 (headline) both print "Table 3" ("Table 3: The grid: Table 3's protocol...", "Table 3 lists every cell of Table 3's protocol").
    - The supplement's Eq. (1) (`eq:law`) and the paper's Eq. (1) (`eq:limit`) both print "Eq. (1)".
    - Its Eq. (2) (`eq:split`, "Eq. (2) below") and the paper's Eq. (2) (`eq:demand`) both print "Eq. (2)".
    - **Fix:** number the supplement's floats and equations S1, S2, ..., as `ieee-supplement.tex` does: `\renewcommand{\thetable}{S\arabic{table}}`, `\thefigure` and `\theequation`. Its appendix is already S1.
14. **`app_extra.tex` (supplement, Shapley paragraph) still blames the second read.** It says the choices interact "because MIN caches 4.6 times as many experts per token as the deployed policy and, loaded by the CPU, each is read twice".
    - **Fix:** "...because on the deployed read path the in-step fetches displace MIN's set (Sec. 4 of the paper, job 113), and MIN caches 4.6 times as many experts per token as the deployed policy, each read twice when the CPU loads it."
15. **Sec. 7: "Predicting no change would have missed by up to 46%" understates.** 46% is the 11% figure; at 25% the miss is 69% (`cxNoChangeDevMidMax`). The budget qualifier was dropped in the rewrite.
    - **Fix:** "...by up to 46% at 11% and 69% at 25%".
16. **Sec. 2: "app:prereg rescores the registered clauses with both" no longer covers every registered clause.** `noise.py` covers jobs 109, 111 and 112, not job 113. By my estimate, round widening changes no job 113 clause, but the largest rental difference would move 113b-H3 (upper bound 1.04) and 113d-H5 (lower bound 1.022) to held (point).
    - **Fix:** either extend `scripts/noise.py` to job 113 and report it, or write "rescores the registered clauses of jobs 109-112 with both".
17. **Clarity pass dropped material.**
    - Sec. 3: the share of job 109's five new machines (41-48%) is now nowhere in the paper. Restore "the five machines added in Sec. 4 fall at 41-48%".
    - Sec. 6 and builder rule 3: the layer-ahead variants' loss on slow-link RTX 4090s (0.64-0.72x at 11%) and "the layer-ahead copy loses time on slow links" are gone. Restore "(0.64-0.72x at 11%)" in Sec. 6.
    - Sec. 7: the registered replication of the interaction on the two fast-link RTX 4090s, and the statement that every machine's interval lies inside the band, are gone. Restore one clause, "every machine's 95% interval lies inside the band, and the interaction is positive on the two with fast links, as registered".
18. **`tab_failures` job 112 T2 row: "The early copies admitted and read more" has lost its evidence.** The numbers (11.7-11.8 admissions per token against Few-2R's 6.2-7.4; 4-7% more host reads) left the main text and are not in `app_prereg` or `tab_job112`.
    - **Fix:** add them to `app_prereg`'s Job 112 paragraph, or as an admissions column in `tab_job112`.
19. **Sec. 4 "What is left": "a residual of 35-47% of the gap stays unexplained ... and 31-36% on the three slow-link RTX 4090s" lost its "at 11%".** The preceding sentence now ends on 25%, so the budget reads as ambiguous. The pointer to Sec. 7 for the RTX 4090s was also dropped.
    - **Fix:** "...35-47% of the gap at 11% ... on the three slow-link RTX 4090s of Sec. 7".
20. **Introduction: "recover at most 6% of the read-ahead oracle's gain on those machines" now follows a second registered test, so "those machines" is ambiguous.**
    - **Fix:** "on the five machines of the first registered test".
21. **`app_foresight`, "Why the misses of MIN-2R stay above R*", explains the excess misses by the copy path's lag alone (d = 2.8-4.7 steps).** Job 113 shows the in-step fetches cause most of them: Few-2R's misses fall from 55.6 to 43.0 at 11% with fetch=0, against Few-1R's 40.7.
    - **Fix:** add "Most of the excess on the two-read path comes from the in-step fetches, which put misses the schedule did not choose into slots: with them off (job 113), Few-2R's misses fall from 55.6 to 43.0 per token at 11%, against Few-1R's 40.7."
22. **Figure fonts: `figs/audit_sol.pdf` (Appendix, `app_wsg.tex`) is not TrueType.** It embeds Latin Modern Roman as CID Type 0C (OpenType CFF), and pdffonts warns "Mismatch between font type and embedded font file" for it and for `paper.pdf` and `ieee-supplement.pdf`. It has no Type 3 fonts.
    - **Fix:** in `scripts/fig_audit_sol.py`, set `font.serif` to a TrueType face (e.g. DejaVu Serif, or a .ttf Computer Modern) and regenerate.
23. **`app_prereg` and the supplement's Table 19 caption say offers a and c "were no longer listed"; no record in either repository shows it.** The ledger only shows that b and d started together, which is consistent with it.
    - **Fix:** commit the launcher's offer-check output (e.g. `gpu/launch_113.txt`) and cite it, or say "were not rented (the launcher found them unlisted)".
24. **`supplement.tex` intro lists the scorer scripts through "job 112 by scripts/job112.py" and omits job 113.**
    - **Fix:** append "; job 113 by `scripts/job113.py`".
