# Number check 25 (9 October)

## Scope

Everything that changed in `paper/`, `scripts/` and `prereg/` between 75cfa37 and HEAD (31bfb6c):

1. MIN's set read once over read twice, both with the in-step fetches off (`fetch/bypass0`), on the five valid machines
   of jobs 114 and 115 (114b, 114c, 114f, 115b, 115c), at gpt-oss 11% and 25%. This covers the new macros
   `dp/dq/dzOneOverTwoZero*`, `dp/dq/dzOneBeatsTwoZero*` and `dp/dq/dzOneLosesTwoZero*` (paper/wsg_job114.tex), and the
   text that uses them: the new paragraph heading in Section 5, that paragraph's last sentence, and builder rule 4.
2. The direct check of Eq. (1) (`dmBound*`, paper/wsg_dm.tex), and where Table 1's first row and Section 3 use it.
3. Table 1: the new row order, the changed first row, "mostly in-sample", each row against its section, and each
   verdict against its scorecard.
4. Appendix text: the pooled in-sample sentence in app_prereg.tex, the fetches-off speed sentence in app_limits.tex,
   the shortened Limitations bullets and rule 1, the C-numbering in supplement.tex, and the note in ieee-supplement.tex.
5. Builds: paper.pdf, supplement.pdf, ieee-paper.pdf and ieee-supplement.pdf.

## Method

All values were recomputed with my own code from the raw files on the gpu branch (`results/<job>/`), in
`/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/nc25/`:

- `oneovertwo.py`: speed ratios from `ec_g_C{14,32}_r{1,2}A.jsonl`. Each ratio is the mean time per token over
  problems (decode_ms / n_decode), as a ratio of means. The 11% budget has two rounds, combined by geometric mean; 25%
  has one round. Intervals are 95% paired bootstraps over problems (4000 draws, the same resample in every round and
  configuration).
- `shares.py`: Eq. (1) = R*·S / B_host. R* (exact) is from prereg/speed_limit_v2.json; B_host is the highest reading
  in each machine's concur.txt. The script then computes the shares of the gap (base − Eq. (1)), the pooled H8 mean
  with its t-interval, and bypass0/base0, foa0/base0 and base0/base.
- `eq1check.py`: every configuration in the raw rows of the 25 launches behind Table 4, at gpt-oss 11% and 25%, over
  Eq. (1). Eq. (1) is evaluated exactly (min over m of the max of the GPU term at B_gpu datasheet and the host term).
  The script also reproduces the selection that `decomp_measured.py` makes.
- `eq1other.py`: the other budgets measured on the same launches (gpt-oss 40%; Qwen3 12.5%, 25% and 43.75%) over Eq. (1).
- `implied.py`: the rate implied by the closed-form account (Table 1, row 1), over the 84 valid launch-budgets.
- `job113_25.py`: the fewest-admission set read once over read twice with the fetches off (job 113), as a cross-check
  of rule 4.
- Builds: I exported HEAD's `paper/`, `scripts/` and `prereg/` with `git archive` into the scratch folder and ran a
  clean latexmk build and `build_ieee.sh` there, with the untracked author files copied in. I compared the text of
  the rebuilt PDFs with the committed ones, read the logs, and inspected page layout with pdftotext, bbox output and a
  page-2 render. One build tested `\IEEEtriggeratref{28}` in the scratch copy only.

I read scripts only for definitions. I modified nothing in either repository except this report.

## Items checked

| # | Location | Claim | My value | Verdict |
|---|---|---|---|---|
| 1 | job114.json / macros, 114b 11% | fetch/bypass0 | 1.2371 [1.2308, 1.2429] | OK |
| 2 | 114c 11% | fetch/bypass0 | 1.0523 [1.0484, 1.0561] | OK |
| 3 | 114f 11% | fetch/bypass0 | 1.2347 [1.2282, 1.2406] | OK |
| 4 | 115b 11% | fetch/bypass0 | 1.0226 [1.0144, 1.0315] | OK |
| 5 | 115c 11% | fetch/bypass0 | 1.0338 [1.0303, 1.0373] | OK |
| 6 | 114b 25% (one round) | fetch/bypass0 | 1.1268 [1.1163, 1.1380] | OK |
| 7 | 114c 25% | fetch/bypass0 | 0.9716 [0.9645, 0.9792] | OK |
| 8 | 114f 25% | fetch/bypass0 | 1.1235 [1.1149, 1.1328] | OK |
| 9 | 115b 25% | fetch/bypass0 | 0.9624 [0.9575, 0.9683] | OK |
| 10 | 115c 25% | fetch/bypass0 | 0.9602 [0.9551, 0.9655] | OK |
| 11 | prereg/job114.json (diff) | only `fetch/bypass0` entries added; no other value changed | point values and intervals match items 1–10 to 3–4 decimals | OK |
| 12 | wsg_job114 `dpOneOverTwoZeroLow{Min,Max,Rng}` | 1.02, 1.24, 1.02–1.24 | 1.0226–1.2371 | OK |
| 13 | `dpOneOverTwoZeroLowHiRatio*` | 1.23–1.24 | 1.2347, 1.2371 (5950Xs, ratio 0.783, 0.786) | OK |
| 14 | `dpOneOverTwoZeroLowLoRatio*` | 1.02–1.05 | 1.0226–1.0523 (ratios 0.539–0.566) | OK |
| 15 | `dpOneOverTwoZeroMid*` | 0.96–1.13 | 0.9602–1.1268 | OK |
| 16 | `dpOneOverTwoZeroMidHiRatio*` | 1.12–1.13 | 1.1235, 1.1268 | OK |
| 17 | `dpOneOverTwoZeroMidLoRatio*` | 0.96–0.97 | 0.9602–0.9716 | OK |
| 18 | `dpOneBeatsTwoZeroLow` / `dpOneLosesTwoZeroLow` | five / no | 5 lower ends > 1; 0 upper ends < 1 | OK |
| 19 | `dpOneBeatsTwoZeroMid` / `dpOneLosesTwoZeroMid` | two / three | 2 (5950Xs) / 3 (Intel) | OK |
| 20 | `dzOneOverTwoZero*` (job 114: b, c, f) | Low 1.05–1.24; LowHi 1.23–1.24; LowLo 1.05; Mid 0.97–1.13; MidHi 1.12–1.13; MidLo 0.97 | same | OK |
| 21 | `dzOneBeats/LosesTwoZero*` | Low three/no; Mid two/one | same | OK |
| 22 | `dqOneOverTwoZero*` (job 115: b, c) | Low 1.02–1.03 (LoRatio same); Mid 0.96 (LoRatio same); no HiRatio macros | same; job 115 has no ratio ≥ 0.7 machine | OK |
| 23 | `dqOneBeats/LosesTwoZero*` | Low two/no; Mid no/two | same | OK |
| 24 | `dpRatioHiRatioRng`, `dpRatioLoRatioRng` | 0.78–0.79; 0.54–0.57 | 0.783–0.786; 0.539–0.566 | OK |
| 25 | main_body l.281, paragraph heading (Section 5) | "Reading each admitted expert once pays most at the smallest budget" | Fetches off: read once beats read twice on 5 of 5 machines at 11% and 2 of 5 at 25%. On factorial machine 096a at 25%, another configuration, `hitoptp`, is faster than MIN-1R (2.04× vs 2.18× Eq. (1)), so the restriction is supported. Section 5's own title is unchanged. | OK |
| 26 | main_body l.287 | "With the in-step fetches off, MIN's set gains read twice as well" | bypass0/base0 1.09–1.16 at 11% and 1.24–1.36 at 25%, all 5 machines | OK in number; scope lost (D5) |
| 27 | main_body l.288 | "read once, it ran 1.02–1.24× that at 11%" | 1.0226–1.2371, all intervals above 1 | OK |
| 28 | main_body l.288–289 | "at 25% gained only on the Ryzen 9 5950Xs (1.12–1.13×) and lost on the others (0.96–0.97×; post hoc)" | 2 gain, 3 lose, all intervals exclude 1; one round per machine | OK (D5 on scope) |
| 29 | main_body l.390–392, rule 4 | "(at ratios of 0.54–0.57, only at the smallest budget)" | Fetches off at ratios 0.54–0.57: read once/twice is 1.02–1.05 at 11% and 0.96–0.97 at 25%. Job 113d (ratio 0.556), fewest-admission set: 1.018 [1.013, 1.023] at 11%, 0.952 [0.943, 0.960] at 25%. | OK in substance; unmarked post hoc, no pointer (D6) |
| 30 | main_body l.34, Table 1 row 4 | MIN-2R 1.09–1.16× and 1.24–1.36× | 1.0920–1.1595; 1.2380–1.3572 | OK |
| 31 | Table 1 row 4 | Dep-1R 1.02–1.07× | 1.0159–1.0691 | OK |
| 32 | Table 1 row 4 / app_prereg | pooled share 19%, interval lower end 13% (H8a held, H8b held (point)) | 19.03% [13.46, 24.61], n = 5 | OK |
| 33 | Table 1 row 4 | "interaction > 0 on four of five (failed on the Core i9-14900K); two checks failed where the fetch table was all zero" | scorecard_115: 115b-H5 failed; scorecard_114: 114b-H1c and 114b-H2 failed; 114b table 0,0,0,0,0 | OK |
| 34 | Table 1 row 4 | verdict mixed, Mach. 5 | 3 failed clauses; 5 valid | OK |
| 35 | Table 1 row 4 | "(held, mostly in-sample)" | 3 of 5 pooled machines are job 114's; job 115's header: "Predictions … from job 114's three valid machines" | OK |
| 36 | app_prereg l.275–277 | "three of its five machines are job 114's, whose results set its threshold" | as item 35 | OK |
| 37 | app_prereg l.277 | job 115's two machines alone: 13–23% | 13.3 (115c), 23.1 (115b) | OK |
| 38 | app_prereg l.277 | job 114's: 15–22% | 15.1 (114c), 21.8 (114b), 21.9 (114f) | OK |
| 39 | app_limits l.31 | deployed cache's speed fetches off/on 0.98–1.08 at 11% (`dpBaseZeroLowRng`) | 0.9755–1.0759 (114b 1.000, 114f 1.000, 114c 1.056, 115c 1.076, 115b 0.976) | OK |
| 40 | app_limits l.32–33 | "where it ran faster without them, part of the measured gap is the fetch table's setting, which the probe sets per machine but does not tune" | faster on 114c and 115c; the gap is measured from the deployed cache with its table | OK |
| 41 | wsg_dm `dmBoundMinOver` | 1.33, the read-ahead oracle on 114f at 11% | 1.3290 (both3p, 114f, C14); per round 1.327; next 114b C14 1.336, 100b C14 1.331 | OK |
| 42 | `dmBoundBelow` = 0 | none below 1 | 0 of all 490 configuration times at gpt-oss 11%/25% on the 25 launches (including the panel's window configurations, job 109's R1/R2, and 109f's cut 25% round) | OK |
| 43 | Eq. (1) per launch-budget | the host term binds | GPU term at B_gpu datasheet (1.908 ms at C = 32) is below the host term on all 50 launch-budgets | OK |
| 44 | other budgets on the same launches | (implied by "no measured time … is below it") | gpt-oss 40% and Qwen3: fastest 1.339× (096a, Qwen3 12.5%); none below 1 | OK |
| 45 | `dmBoundMachinesN` = 25 (Table 1: "25 launches") | 25 launches | 15 panel + 5 job 109 + 5 jobs 114/115 = 25 launch directories (24 GPUs: 114f re-rents 109f's GPU f90ec646) | OK (macro misnamed, D3) |
| 46 | `dmBoundLaunches` = 49 | launch-budgets | 49 (109f's 25% round has no complete cell) | OK (misnamed, D3) |
| 47 | `dmBoundRuns` = 312 (Section 3) | 312 times per token measured on Table 4's machines | 150 (panel, 5 states × 30) + 72 (job 109: process A's 7 and process B's base × 9 cells) + 90 (jobs 114/115: 8 + `baseB` × 10). For jobs 114/115 `baseB` is a copy of `base` (no process B ran; job109.load substitutes A's base), so there are 10 duplicates and 302 distinct times. The launches measured 490 times at these budgets. | DEFECT (D3) |
| 48 | Table 1 row 1 | "fastest 1.33× it" | 1.329 | OK |
| 49 | Table 1 row 1 | "a fitted model of the engine's reads implies a rate > 5% faster on 12 of 84 launch-budgets" | 12 of 84 (max 1.223×) recomputed. "Fitted" contradicts App. H ("No constant is fitted to the times"), and "faster" has no referent. | DEFECT (D2) |
| 50 | main_body l.154–156 | Section 3's sentence on the check | numbers as items 41–47 | DEFECT in count/scope (D3) |
| 51 | Table 1 order | registered and derived rows in section order, post hoc rows last | §3, 3, 3, 4, 4, 4, 4, 5, 6, 7, then post hoc 3, 3, 8 | OK |
| 52 | Table 1 move | nothing lost | the moved rows are identical to 75cfa37's apart from "mostly in-sample" | OK |
| 53 | Table 1 row 2 → §3 | 86–96%, 5 of 33 failed, held (point) | 102x-P2floor: 8 held (point); 33 clauses, 5 failed (P2below ×4, 102a-P3) | OK |
| 54 | Table 1 row 3 → §3 | mixed; on S five of six ratios outside ±0.06 | 081: 6 held; 089-P2a/b/c/e/f failed, P2d held | OK |
| 55 | Table 1 row 5 → §4 | FewTwo 1.23–1.26, MIN-2R 1.09–1.12 (held); one read +0.02–0.16, deployed 0.98–1.07 (failed on the 285K); mixed | 1.2594/1.2306; 1.1218/1.0945; scorecard 113d-H3 and H6 failed | Numbers OK; § wrong (D4) |
| 56 | Table 1 row 6 → §4 | 9 of 10 (failed), 5 of 5 (held); mixed | 099 P5a: 19 held, 1 failed (099Pf-g11); 109 P1-int 9 held | OK |
| 57 | Table 1 row 7 → §4 | failed on 5 and 7 of 8 machine-budgets; failed | 112 T1: 5/8 failed; T2: 7/8 failed (T3, T4 held; outside the row's threshold) | Verdict OK; § wrong (D4) |
| 58 | Table 1 row 8 → §5 | mixed | 106-P6/P7 failures; 104, 105, 107 held | OK |
| 59 | Table 1 row 9 → §6 | held | 109-P7 19 held; 111/112-Q3 all held | OK |
| 60 | Table 1 row 10 → §7 | held | 111/112-Q1 held; the two held (point) clauses are pooled counts without an interval | OK |
| 61 | Table 1 rows 11–13 → §3, §3, §8 | post hoc | sections match | OK |
| 62 | Table 1 note | "Mach. … (for the first 2×2 row, the panel's main job and the new machines)" | after the reorder the first 2×2 row is the fetches-off row (Mach. 5), not the deployed-path row (10 + 5) | DEFECT (D1) |
| 63 | Table 1 note → Table 9 (tab:claimindex) | points to each claim | all 9 registered rows present; order still the old one | minor (D10) |
| 64 | main_body l.415–417, Limitations bullet 1 | shortened | nothing lost; "batch-1, teacher-forced decode" now also covers the system comparison, which decodes freely (§2) | minor (D7) |
| 65 | main_body l.418–420, bullet 2 | shortened | all old content kept (exact routing; per-layer slots, pooling 4.5–18.6%; no latencies; probe; ratio moved up to 12%; T_GPU) | OK |
| 66 | main_body l.383–384, rule 1 | "needs only a routing trace and a probe" | same content as "the routing trace and one probe run" | OK |
| 67 | supplement.pdf | tables C1, C2, …; no "Table S" | Tables C1–C21 consecutive; Figures C1–C3; Eqs. C1–C2; sections C1–C4; 0 "Table S", 0 other S-numbers | OK |
| 68 | supplement.pdf | appendix numbering | the supplement's sections C1–C4 sit beside the paper's Appendix C (Names) | minor (D8) |
| 69 | ieee-supplement.tex l.46 | "(paper/supplement.pdf, tables C1, C2, …)" | matches supplement.pdf | OK |
| 70 | paper.pdf | 0 errors; Conclusion ends on page 10; all floats before the references | 47 pp.; 0 errors, 0 undefined refs, 0 overfull. Conclusion ends p.10; references start p.11; Tables 1–6 and Figures 1–5 on pp.2–10. Table 4 = tab:gap (as the prompt assumes). | OK |
| 71 | supplement.pdf build | 0 errors | 59 pp.; 0 errors, 0 undefined refs; 29 overfull hboxes, all ≤ 4.6 pt | OK |
| 72 | ieee-paper.pdf | 0 errors, references resolve | 10 pp.; 0 errors, 0 undefined, no "??"; 30 "multiply defined" (the xr-imported citation labels, as documented) | OK |
| 73 | ieee-paper.pdf, last page | reference columns roughly balanced | left column ends after 34 lines ([13]–[26]), right after 40 ([27]–[39]): 6 lines (≈ 54 pt) apart (75cfa37: 30 vs 40). `\IEEEtriggeratref{28}` gives 37/37 (scratch build: 0 errors, 10 pp.). | minor (D9) |
| 74 | ieee-supplement.pdf | 0 errors, references resolve, balance | 31 pp.; 0 errors, 0 undefined, no "??"; the reference list is one column, so balance does not apply | OK |
| 75 | all four PDFs | committed PDFs come from HEAD | the clean rebuild from `git archive HEAD` (with the untracked author files) gives text identical to the committed PDFs (0 differing lines each) | OK |

## Defects

1. **Table 1 note now points to the wrong row (main_body.tex l.46).** "Mach.: machines that passed every check (for
   the first 2×2 row, the panel's main job and the new machines)" described the deployed-path row when that row came
   first. After the reorder, the first 2×2 row is "With the in-step fetches off …" (Mach. 5). Fix: replace "for the
   first 2$\times$2 row" with "for the row on the deployed read path".

2. **Table 1 row 1 calls the closed-form account "a fitted model" (main_body.tex l.31).** App. H says "No constant is
   fitted to the times", and "a rate > 5% faster" names nothing to compare with. The numbers (12 of 84) are right.
   Fix: replace "a fitted model of the engine's reads implies a rate $>$\,5\% faster on" with "by the closed-form
   account (\cref{app:relation}), the engine's reads imply a rate $>$\,5\% above the probe's highest reading on".

3. **The Eq. (1) check's count is inflated and mis-scoped (wsg_dm `dmBoundRuns` = 312; main_body.tex l.154–156;
   scripts/decomp_measured.py l.404–409).**
   - Ten of the 312 entries are jobs 114/115's `baseB`. No process B ran there, so `job109.load` copies A's `base`
     into it: 302 distinct times.
   - The 312 are not "the times per token measured on the machines of Table 4". They are Table 4's five states on the
     panel, job 109's process A (with its online policies) and process B's base, and jobs 114/115's eight
     configurations. The same launches measured 490 configuration times at these two budgets.
   - All conclusions still hold: none of the 490 is below 1, nor any at the other budgets, and the minimum is 1.33.
   - The macro names are also swapped in meaning: `dmBoundMachinesN` (25) counts launches, and `dmBoundLaunches` (49)
     counts launch-budgets.

   Fix, in decomp_measured.py:
   - Build `over` without the copied `baseB` of the CPU rows, e.g. `for k, tt in r["t"].items() if not (r in CPU and
     k == "baseB")`, which gives 302.
   - Rename `dmBoundMachinesN` to `dmBoundLaunchesN` and `dmBoundLaunches` to `dmBoundLaunchBudgets`.
   - Correct the comment to name the configurations actually included.

   In main_body.tex l.154–155, write: "None of the \dmBoundRuns{} times per token behind \cref{tab:gap}
   (\dmBoundLaunchesN{} launches, gpt-oss 11\% and 25\%) is below it; …". In Table 1 row 1, use the renamed
   `\dmBoundLaunchesN`.

4. **Table 1 rows 5 and 7 point to Section 4, which does not discuss them (main_body.tex l.35, l.37).** "Without the
   fetches, the fewest-admission set pays read twice" (job 113) and "Control: issuing the two-read arm's copies a step
   earlier" (job 112) have no text in Section 4. Their results are only in Table 1, App. D and Tables 12–13. Fix: change
   their § cells from `\ref{sec:gap}` to `\ref{app:prereg}`, or add one sentence on each to Section 4.

5. **Section 5's last sentence lost its machine scope (main_body.tex l.287–289).** The old text had "on the five
   machines tested". The new sentence says "MIN's set gains read twice as well" with no count, and "the Ryzen 9 5950Xs"
   and "the others" are unquantified. Fix: "With the in-step fetches off, on the \dpNWord{} machines of
   \cref{sec:gap}, MIN's set gains read twice as well; read once, it ran \dpOneOverTwoZeroLowRng$\times$ that at 11\%,
   but at 25\% gained only on the \dpHiRatioN{} Ryzen 9 5950Xs of one provider (ratio \dpRatioHiRatioRng;
   \dpOneOverTwoZeroMidHiRatioRng$\times$) and lost on the \dpLoRatioN{} Intel machines (ratio \dpRatioLoRatioRng;
   \dpOneOverTwoZeroMidLoRatioRng$\times$; one round at 25\%; post hoc)."

6. **Rule 4's new clause is post hoc but not marked, and has no pointer (main_body.tex l.391).** "(at ratios of
   0.54–0.57, only at the smallest budget)" rests on the post hoc 25% split of Section 5, from three machines with one
   round each. Job 113d (ratio 0.556) agrees: 0.95× at 25%, 1.02× at 11%. Fix: "(at ratios of \dpRatioLoRatioRng, only
   at the smallest budget; post hoc, \cref{sec:foresight})".

7. **The Limitations bullet makes "teacher-forced" cover all decode (main_body.tex l.416–417).** "batch-1,
   teacher-forced decode of two models" now applies to the system comparison too. Per §2, that comparison decodes
   through each system's server, and only the engine experiments are teacher-forced. Fix: "batch-1 decode of two models
   on AIME prompts also used in development (teacher-forced in the engine experiments)".

8. **The supplement's C-numbered sections can be read as parts of the paper's Appendix C (supplement.tex l.53–74).**
   The supplement's sections are now C1–C4, and its text says "(Appendix C1)" next to references to the paper's
   lettered appendices ("Appendix J", "Appendix E"). The paper's own Appendix C is "Names". Nothing in the supplement
   says which numbers are its own. Fix: add to the supplement's opening paragraph "Sections, tables, figures and
   equations numbered C1, C2, \ldots{} are this document's; other numbers, and the appendix letters A--N, are the
   paper's."

9. **The IEEE last page's reference columns are 6 lines apart (ieee-paper.tex l.28).** The left column ends at 34
   lines and the right at 40. Fix: `\IEEEtriggeratref{27}` → `\IEEEtriggeratref{28}`. In a scratch build this gives
   37/37 lines, 10 pages and 0 errors.

10. **The claim index still has the old order (app_prereg.tex l.36–46, tab:claimindex).** Table 1's note sends readers
    to Table 9, whose rows no longer follow Table 1. Fix: reorder its rows as readsched, FreeToken, fetches off
    (114/115), fewest-admission read twice (113), deployed path (099/109), Control (112), fewest-admission read once,
    policies, second card; keep the closed-form account last.
