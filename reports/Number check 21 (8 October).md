# Number check 21 (8 October)

## Scope

Everything that changed in `paper/` and `scripts/` since commit 9726d68, as uncommitted working-tree changes. That covers:

- the rewritten abstract;
- Table 1, restructured into Claim, Registered threshold, Result;
- Table 2, which is now a names table;
- the round-to-round and rental-to-rental analysis (`scripts/noise.py`, `paper/wsg_noise.tex`, `prereg/noise.json`, the Statistics paragraph of Section 2, and the "Rounds and rentals" paragraph of `app_prereg.tex`);
- Section 4's 2x2, timing-control and "What is left" paragraphs, and the closed-form paragraph removed from Section 4;
- Sections 6 and 7, now split, the builders' scope lead-in, and the first item of the Limitations;
- the four generated tables whose captions were split into a caption plus a note (`\tabnote`), and the other changes to Table 3;
- the IEEE build (`ieee-paper`, `ieee-supplement`, `ieee_preamble`, `build_ieee.sh`);
- the MLSys build `paper.pdf`.

## Method

All of my own code is in `scratchpad/nc21/`:

- **`raw.py` and `machines.py`** read every `ec_g_*.jsonl` row of jobs 099–112. From those rows they compute:
  - each process's speed ratio (the deployed cache's mean time over a configuration's mean time, on shared problems);
  - the geometric mean over rounds;
  - 95% paired bootstrap intervals over problems (4,000 draws, my own seed, paired across configurations and rounds);
  - the engine counters from the `st_g_*.json` files.
- **`rescore.py`** does three things:
  - It re-derives every clause of jobs 109, 111 and 112 that carries an interval, from the job headers. My statuses agree with `prereg/scorecard_109/111/112.json` on all 105 clauses.
  - It recomputes the round ranges.
  - It applies the widening both ways: as `noise.py` does, and as the paper describes it (all intervals).
- **`rental.py`** pairs launches by GPU UUID from `nvidia-smi-q.txt`. It also compares the fetch tables of each pair (`fetch_table_law_gptoss.json`).
- **Other checks:**
  - Job 102's fractions come from `readsched_C*.txt` and `concur.txt`, using the probe's highest reading.
  - The panel and job 109 interactions come from process-A times.
  - The fewest-admission gains come from jobs 104–112.
  - The trend deviations use job 110's frozen constants.
  - The gap shares of job 109 use B_host from `concur.txt` and R* from the header.
- **Builds.** Macros were resolved with my own parser (`macros.py`, `expand.py`). The PDFs were read with `pdftotext`, and the cross-references with my own `.aux` parser (`xref.py`). The IEEE PDFs are newer than every source except `refs.bib` and `refs_wsg.bib`, which another agent is editing, so I did not rebuild them.

I did not recompute from raw data the macros that are unchanged and were checked in number check 20: the consumer and server shares (31–54%, 12–46%, 25 machines), the audit median of 13.6%, and 0.65 C on 9 models. They are listed below as consistent only.

## Items checked (102)

| # | Location | Claim | My value | Verdict |
|---|---|---|---|---|
| 1 | Abstract | 31–54% of the bound on consumer machines | same macro as Section 3 and Table 1 | consistent (not recomputed) |
| 2 | Abstract | faster than two other systems at most configurations | FreeToken 11/12, llama.cpp 12/12 (Table 3) | OK |
| 3 | Abstract | oracles close "about half" of the gap | panel 33+15=48; new 34.7+19.0=54 (recomputed) | OK |
| 4 | Abstract | the rest is attributed, not measured | Table 4's attributed rows | OK |
| 5 | Abstract | caching what MIN keeps pays only if read once | MIN-2R: 0% (panel) and 3% (new) of the gap; Few-2R 1.03× on all 4 control machines | scope: D8 |
| 6 | Abstract | the two changes close 35% in the registered test | 34.7% (job 109, 5 machines) | OK |
| 7 | Abstract | no-foresight policies recover at most 6% | 5.8% (109c) on job 109; 26% on job 111's RTX 4090s | value OK, scope: D9 |
| 8 | Table 1 r2 | registered ≥50% per layer | job 102 P2 "layer mode ≥ 0.50" | OK |
| 9 | Table 1 r2 | "(loose)" | header's analytic estimate 85–99%; app_prereg is silent | D12 |
| 10 | Table 1 r2 | 86–96% | 86.0–96.3% (layer mode, 3 hosts, both budgets) | OK |
| 11 | Table 1 r2 | 5 of 33 clauses failed; "held" | scorecard_102: 5 failed, 28 held (point); P2floor held (point) | count OK; status wording: D13 |
| 12 | Table 1 r2 | Mach. 3 | 102a, 102b, 102c | OK |
| 13 | Table 1 r3 | registered "speed ratios on host B" | job 081: ours > FreeToken (CI > 1), ≥1.8× / ≥2× llama.cpp | vague: D14 |
| 14 | Table 1 r3 | held on B | 081-P2, P3b, P4a, P4b held | OK |
| 15 | Table 1 r3 | 5 of 6 outside ±0.06 on S | differences 0.088, 0.078, 0.061, 0.006, 0.105, 0.075 | OK |
| 16 | Table 1 r3 | faster at 11 of 12 | Qwen3 43.75% on S is 0.974 | OK |
| 17 | Table 1 r4 | interaction > 0 on every machine | job 099 P5, job 109 P1 headers | OK |
| 18 | Table 1 r4 | 0.25–0.55 share "(loose)" | job 109 P3; named in app_prereg | OK |
| 19 | Table 1 r4 | positive on 9 of 10 panel machines (failed) | raw: only 099f is negative (−0.48 ms) at 11%; 10/10 at 25% | OK |
| 20 | Table 1 r4 | 5 of 5 new machines (held) | +1.5 to +7.5 ms; scored held (point) | count OK; wording: D13 |
| 21 | Table 1 r4 | share 35% | 34.7% | OK |
| 22 | Table 1 r4 | Mach. 15+5 | interaction tested on 10 of the 15 | unclear: D15 |
| 23 | Table 1 r5 | misses ≤ 0.95×; "no slowdown" | T1 ≤ 0.95×; T2 allows −0.01 | D10 |
| 24 | Table 1 r5 | misses 0.97–0.98× | 0.968–0.980 at 11%; 0.949–0.953 at 25% | scope: D10 |
| 25 | Table 1 r5 | failed on 5 of 8; 7 of 8 | T1 fails 4/4 at 11%, 1/4 at 25%; T2 fails 4/4 and 3/4 | OK |
| 26 | Table 1 r5 | 0.03–0.05 slower | 0.031–0.053 at 11%; 0.006–0.027 at 25% | scope: D10 |
| 27 | Table 1 r5 | second read costs more than the late landing (held) | T3 held on 4/4 (0.24–0.48, intervals > 0) | OK |
| 28 | Table 1 r5 | Mach. 4 | 112a, 112b, 112d, 112g | OK |
| 29 | Table 1 r6 | ≤ 1.10×; capture ≤ 35% (loose) | job 109 P7; capture named in app_prereg | OK |
| 30 | Table 1 r6 | best 1.007–1.024× | 109d 1.007 … 109c 1.024 | OK |
| 31 | Table 1 r6 | capture at most 6% (held) | 5.8%; all P7 clauses held with intervals | OK |
| 32 | Table 1 r7 | > 1 on every machine | 104a P3, 105 P4, 106 P6/P7, 107 P5 | OK |
| 33 | Table 1 r7 | 1.24× geometric mean on 13 stable machines | 1.240 on 13 UUIDs (raw) | OK |
| 34 | Table 1 r7 | 7 registered | Pf, 105a, Threadripper, 105f, Pe, 107b, 107d | OK |
| 35 | Table 1 r7 | lost on the two unsteady ones | 106c 0.85 (ratio 0.14), 107e 0.97 (0.21); 106c's registered clause failed | D11 |
| 36 | Table 1 r8 | within 0.10 in log | job 110 Q1 | OK |
| 37 | Table 1 r8 | within 7% at ratios 0.37–0.77 | 6.6% at 11% (5.0% at 25%); ratios 0.371–0.771 | OK |
| 38 | Table 1 r8 | new RTX 5090s off by up to 18%; 5 of 18 cells | 17.6%; 5 of 18 | OK |
| 39 | Table 1 r8 | Mach. 5 | 111d, 111f, 111g, 112a, 112b | OK |
| 40 | Table 1 r9–r11 | 31–54 / 12–46 / 25; 13.6%; 0.65 C, 9 models | same macros as the text | consistent (not recomputed) |
| 41 | Table 1 | caption's "loose" definition, and labels on r2, r4, r6 | r4 and r6 are backed by app_prereg; r2 is not | D12 |
| 42 | Table 2 | deployed κ=1; margin κ=3 (11%) and 2 (25%) | job 109 header: KA=3, KB=2 | OK |
| 43 | Table 2 | layer-ahead copy, footnote a | engine copies the top predicted *non-resident* expert | D19 |
| 44 | Table 2 | LA in the "read twice" column only | correctly predicted admissions are read once | D20 |
| 45 | Table 2 | LA-1R and LA-1R-M (footnote b) | R1 = base + fetch_on_admit + prefetch; R2 adds dk's κ | OK |
| 46 | Table 2 | admit every miss is read once | aa: foa with κ = −1e9 | OK |
| 47 | Table 2 | MIN-2R, MIN-1R, read-ahead oracle (footnote c) | bypass, fetch; both3p uses lead=3 gated by the MIN test, with oracle_fetch | OK |
| 48 | Table 2 | Few-1R, Few-2R, Few-2R-early (footnote d) | fetchplan, bypassplan, bypassplanS | OK |
| 49 | Table 2 | Belady-2R and Belady-1R | ORACLE_NOBYPASS with lead | OK |
| 50 | Table 2 | panel: 15 machines, 10 ran its main job | 096a, 096b, 099a–j, 100b, 100f, 101b | OK; wording: D22 |
| 51 | Table 2 | new machines: 5, ratio ≥ 0.5 | job 109's VR gate | OK |
| 52 | Table 2 | O3–O5: three machines | matches Fig. 3's caption | OK |
| 53 | Table 2 | RTX 4090s: 5 | 3 (job 111) + 2 (job 112) | OK |
| 54 | Table 2 | yardsticks (speed, share of bound, share of gap, capture, value) | match job 109's definitions and Section 8 | OK |
| 55 | Table 2 | read-ahead oracle reads 1.22–1.31 R* | unchanged macro | consistent |
| 56 | Section 2 | "Table 2 lists every configuration, machine set, yardstick and bound" | no bounds; window policies missing | D21 |
| 57 | Noise | 98 machine-budget-configurations on 12 machines | 98 on 12, all at 11% | OK |
| 58 | Noise | rounds: median 0.2%, at most 1.8% | 0.17%, 1.75% | OK |
| 59 | Noise | median half-width 0.6% | 0.60% (11% intervals) | OK |
| 60 | Noise | 69 pairs from "10 machine pairs" | 69 pairs; 10 launch pairs on 9 machines | wording: D4 |
| 61 | Noise | rentals: median 0.5%, at most 5.3%, 2 above 5% | 0.53%, 5.32%, 2 (both 110c/111g) | OK |
| 62 | Noise | "same configuration" in both launches | fetch tables differ in 4 of 10 pairs, including the two largest | D3 |
| 63 | Noise | "job 099's panel and its relaunches" | relaunches in jobs 104–106 left out (adding them: median 0.53%, max 5.3%) | D4 |
| 64 | Noise | 93 "per-machine" clauses held with an interval | 89 per machine + 4 pooled (t-intervals over machines) | D2 |
| 65 | Noise | all 93 still hold after round widening | reproduced: 0 move. 25% cells are not widened; widening them by the 11% range also moves 0 | OK; wording: D1, D6 |
| 66 | Noise | rental widening: 60 / 33 / 25 | reproduced as coded; with the 25% intervals widened too: 53 / 40 / 29 | D1 |
| 67 | Noise | no clause changes to failed | none | OK |
| 68 | Noise | soundness of the widening (both sides, full range, interval arithmetic for derived intervals) | conservative, as a sensitivity check | OK apart from D1 and D3 |
| 69 | Section 2 | the main-text noise sentence | numbers match `wsg_noise.tex`; no job scope | D5 |
| 70 | Section 2 | "two or three processes, rounds, per launch" | jobs 109–112: 2 rounds at 11%, 1 at 25% | D6 |
| 71 | Section 4 | MIN-2R 0%, Dep-1R 0%, MIN-1R 33% / 35% | Table 4; new machines recomputed 3.4, −0.4, 34.7 | OK |
| 72 | Section 4 | Few-2R-early misses 0.97–0.98 of Few-2R's | 0.968–0.980 | OK |
| 73 | Section 4 | admits 11.7–11.8 vs 6.2–7.4; +4–7% host reads | 11.7–11.8 vs 6.2–7.4; +4.2% to +7.1% | OK |
| 74 | Section 4 | 0.98–1.00×, Few-2R 1.03×, Few-1R 1.21–1.41× on four machines | raw values match Table 5 | OK |
| 75 | Section 4 | 9 of 10 (failed), all 5 new (held) | as item 19; the new ones are held (point) | D13 |
| 76 | Section 4 | Few-2R 1.03× on the control's machines | 1.027–1.030 (MIN-2R there: 1.000–1.032) | OK |
| 77 | Section 4 | "more on slow links (Section 7)" | 1.11–1.22× (104a, 105b, 105e; ratios 0.28–0.32) | OK |
| 78 | Section 4 | 25%: MIN's set alone closes 21% | panel 21; new 19.8 | OK |
| 79 | Section 4 | read-ahead closes 15% more, 52% left | Table 4 (new machines: 19, 46) | OK |
| 80 | Section 4 | Eq. 3 sentence (83–99%, 73–81%, median 45%) | values unchanged; their pairing is lost | D23 |
| 81 | Section 4 | removed closed-form paragraph is in App. H | relation, consumer-processor scope, "does not rely on it"; 4 tests (3 failed, 1 inconclusive) in tab_regtests | OK; stale pointers: D16 |
| 82 | Sections 6–7 | split, no stale refs to `sec:transfer` | none left; intro and Setting cite both | OK |
| 83 | Section 6 | four variants; second threshold loose | job 109 header; app_prereg | OK |
| 84 | Section 7 | footnote replaced by "App. D reports the deviation" | the first-round deviation (≤6%) is no longer given | D24 |
| 85 | Section 2 | 0.5 line drawn before the tests of Sections 6 and 7 | the first test using it is also Section 4's | D28 |
| 86 | Section 9 | scope lead-in: rules 2–4 from one model at two budgets | gpt-oss 11% and 25% | OK |
| 87 | Section 11 | first item: one engine for every engine result | OK | OK |
| 88 | Table 4 | caption + note = old caption | verbatim | OK |
| 89 | Table 6 | caption + note = old caption | verbatim | OK |
| 90 | Table 5 | caption + note = old caption | verbatim (±0.01 kept) | OK |
| 91 | Table 3 | caption + note | old text kept; h/o key and model abbreviations added; pooled and VRAM sentences moved | OK |
| 92 | Table 3 | 2-decimal intervals (12 ratios, 24 bounds) | match prereg JSON unrounded values (for example 1.2747 → 1.27, 1.1537 → 1.15) | OK |
| 93 | Table 3 | shortened host rows | same values (17001/72/53/78/88; 14001/56/53/66/71) | OK |
| 94 | App. J | pooled-bound sentence; VRAM speeds 255.3 and 177.8 tok/s | vram_084.json: 255.28, 177.76 | OK |
| 95 | IEEE | PDFs newer than the sources | yes (except `refs*.bib`) | not rebuilt |
| 96 | IEEE | paper 10 pages; supplement 30 pages | 10; 30; floats I–VI and Figs. 1–5 before the references | OK |
| 97 | IEEE | 0 errors, 0 undefined references, no "??" | yes | OK |
| 98 | IEEE | cross-document references (23 checked) | App. C, D, F–N; Tables S16, S22, S23, S24; Eqs. (1) and (2); Sections III, IV, VII; Tables I, II, III, VI; Fig. 3 | all correct targets |
| 99 | IEEE | citations in IEEE form | [3]–[5], [6]–[8]; Eliseev = [3] in the paper, Kalibera = [2] in the supplement; author-name citations render | OK; warnings: D27 |
| 100 | IEEE | supplement float placement | Tables S5, S7–S25 fall after the references | D25 |
| 101 | IEEE | supplement numbering S1… | no Table S1 | D26 |
| 102 | MLSys | Conclusion ends p.10; References from p.11; floats before them; 0 errors | Tables 1–6 and Figs. 1–5 on pp. 2–10; current content | OK |

## DEFECTS

**D1. The rental widening skips every 25% interval.**

- **Where:** `scripts/noise.py`, `widen()`; app_prereg "Rounds and rentals"; Section 2 Statistics.
- **Problem:** `widen()` adds `extra` only to (budget, configuration) keys that have a round range. In jobs 109–112 every configuration ran in one round at 25%, so 21 per-machine clauses at 25% are never widened in either step:
  - P5, P7-best-g25, P7-cap-g25 and P10-dk-g25 on 109a, 109c, 109d and 109e;
  - Q3-g25 on 111d, 111f, 111g, 112a and 112b.

  The text says "each machine's interval for each configuration". Widened as stated:
  - 53 clauses still hold (49 per machine + 4 pooled);
  - 40 hold on the point estimate only;
  - 29 of those 40 are the narrow bands (109a, c, d, e P10-dk-g25 and 111d, f, g Q3-g25 move in addition).
- **Fix:**
  1. In `widen()`, use `d = sp.get((C, n), 0.0) + extra` for every interval, and the same for `best`.
  2. Regenerate: nzRentalStillHeld 53, nzRentalMoved 40, nzRentalMovedNarrow 29. Update the text's 60 / 33 / 25 accordingly.
  3. In app_prereg, after "round-to-round range on that machine", add: "(at 11%; each configuration ran once at 25%, so the round step leaves those intervals as they are, and widening them by the same configuration's 11% range moves no clause)".

**D2. "93 per-machine clauses" includes 4 pooled clauses.**

- **Where:** app_prereg "Rounds and rentals".
- **Problem:** Four of the 93 are pooled clauses with t-intervals over machines: 109-P3-g25, 109-P6-g11, 109-P6-g25 and 109-P7-mean. The widening does not touch them.
- **Fix:** Replace the sentence with: "All 93 clauses of these jobs that held with an interval still hold: 89 per machine, whose intervals were widened, and 4 pooled over machines, whose $t$-intervals already include the machines' variation."

**D3. The rental pairs did not all run the same configuration.**

- **Where:** app_prereg "Rounds and rentals"; Section 2.
- **Problem:** Each launch derives its fetch table from its own probe. The tables differed in 4 of the 10 launch pairs (Ph/100c, Pg/103d, 110d/111d, 110c/111g). Those pairs give the largest differences: 5.3% (110c/111g, table 0,0,0,0,0 against 0,0,0,1,1) and 4.1% (110d/111d), both for fetch and both3p on slow-link RTX 4090s. Over the 40 same-table comparisons the median is 0.3% and the largest 2.7%.
- **Fix:**
  - **app_prereg:** replace "every machine measured with the same configuration in two launches" with "every machine measured with the same configurations in two launches (each launch derives its fetch table from its own probe; the tables differed in 4 of the 10 launch pairs, which give the two largest differences, 5.3% and 4.1%; with the same table the largest is 2.7%)".
  - **Section 2:** write "…and 0.5% (at most 5.3%, where a relaunch's probe changed the fetch table) between rentals of one machine".

**D4. The rental-pair set is mislabelled.**

- **Where:** app_prereg "Rounds and rentals".
- **Problem:** The pairs are launch pairs, not machine pairs (Pf is in two). And the panel's relaunches in jobs 104–106 are left out; including them leaves the median at 0.53% and the maximum at 5.3%.
- **Fix:**
  - "(69 pairs from 10 machine pairs: …)" → "(69 comparisons from 10 pairs of launches on 9 machines: …)".
  - "job 099's panel and its relaunches" → "job 099's panel and its relaunches in jobs 100, 101 and 103".

**D5. The main-text rescoring sentence has no job scope.**

- **Where:** Section 2, Statistics.
- **Problem:** "all 93 clauses that held with an interval" reads as all clauses of jobs 073–112.
- **Fix:** "Widened by each machine's round-to-round range, all 93 clauses of jobs 109–112 that held with an interval still do; …".

**D6. Wrong round count for jobs 109–112.**

- **Where:** Section 2, Statistics.
- **Problem:** "each configuration ran in two or three processes, rounds, per launch" is wrong at 25% for jobs 109–112, which is where the noise analysis applies. Those jobs ran 2 rounds at 11% and 1 at 25%.
- **Fix:** "…ran in two or three processes, \emph{rounds}, per launch at 11\% (one at 25\% in jobs 109–112)".

**D7. The 8 non-narrow clauses that move are not named.**

- **Where:** app_prereg "Rounds and rentals"; Section 2.
- **Problem:** They include Table 1's second-card check and the no-foresight bound. As coded they are:
  - 109d-P8;
  - Q1-both3p on 111f and 111g;
  - Q3-g11 on 111f and 111g;
  - Q1-fetch and Q1-both3p on 112a;
  - 112b-T3.

  After D1 there are 11, adding Q3-g25 on 111d, 111f and 111g.
- **Fix:** Add to app_prereg: "the others are the trend checks of \MinOne{} and the read-ahead oracle on three RTX 4090s, the no-foresight bound (Q3) on three, job 109's P8 on one machine and T3 on one".

**D8. The abstract's 2x2 claim is wider than the main text's.**

- **Where:** abstract.
- **Problem:** "caching the experts MIN would keep pays only if each admitted expert is read once" is not limited to the greedy schedule. Section 4 and Table 1 now are: MIN's fewest-admission set read twice runs 1.03× on all four control machines and 1.11–1.22× on slow links.
- **Fix:** "caching the experts MIN's greedy schedule keeps pays only if each admitted expert is read once, not twice".

**D9. The abstract's 6% claim lost its scope.**

- **Where:** abstract.
- **Problem:** "Policies without foresight … recover at most 6%" dropped its "There,". On job 111's slow-link RTX 4090s the best variant captures up to 26% (Table 6; cxCapLowMax).
- **Fix:** "There, policies without foresight, built from the engine's own mechanisms, recover at most 6% of what an oracle that reads ahead gains."

**D10. Table 1 row 5 mixes budgets and paraphrases T2.**

- **Where:** Table 1 row 5.
- **Problem:**
  - The ranges are at 11% only, while the counts are over both budgets. At 25% the misses are 0.949–0.953× (3 of 4 met ≤ 0.95) and the slowdown is 0.006–0.027.
  - T2's threshold is "at least Few-2R's − 0.01", not "no slowdown".
- **Fix:**
  - Threshold: "misses ≤ 0.95× \FewTwo's; slowdown ≤ 0.01".
  - Result: "at 11\%, misses 0.97–0.98× and 0.03–0.05 slower (failed on 5 and 7 of 8 machine-budgets); …".

**D11. Table 1 row 7 hides a failed registered clause.**

- **Where:** Table 1 row 7.
- **Problem:** "held on the 7 registered; lost on the two unsteady ones" omits that the registered clause failed on 106c: 106c-P7-speed failed at 0.85, and the P6 rounds failed. 107e was excluded by job 107's validity rule.
- **Fix:** "… held on the 7 stable machines it was registered on; failed on the unsteady 106c and lost on 107e, the two with the slowest links".

**D12. The "(loose)" label on the 50% threshold is unsupported.**

- **Where:** Table 1 row 2, and the "loose" wording in Section 3.
- **Problem:** app_prereg's list of thresholds that "could hardly have failed" names only job 109's bands and job 110's predictions 3–5. Job 102's header does justify it: its analytic estimate was 85–99% before launch.
- **Fix:** Add to app_prereg's Job 109 paragraph, or near the claim index: "Job 102's floor of 0.50 for the per-layer mode was loose in the same way: its header's analytic estimate was 85–99%."

**D13. Table 1 and Section 4 say "held" for clauses scored "held (point)".**

- **Where:** Table 1 rows 2 and 4; Section 4 "(held)".
- **Problem:** The clauses involved are 102x-P2floor and 109x-P1-int. Both are scored "held (point)" because they have no interval.
- **Fix:** Write "held (point)" in those places, or add to Table 1's caption: "held: at least on the point estimate; \cref{app:prereg} gives each clause's interval status".

**D14. Table 1 row 3 does not state the registered threshold.**

- **Where:** Table 1 row 3.
- **Fix:** Replace "speed ratios on host B" with "on B: ours > FreeToken (CI > 1), ≥ 1.8× llama.cpp (≥ 2× Qwen3 12.5%); on S: within ±0.06 of B's ratios".

**D15. Table 1 row 4's "15+5" is no longer explained.**

- **Where:** Table 1 row 4, Mach.
- **Problem:** The caption no longer explains "15+5", and the interaction was tested on 10 of the 15 panel machines.
- **Fix:** Write "10 (15)${}+{}$5", or restore the caption clause: "\dmMachines${}+{}$\olN: the panel and, separately, the machines rented for the registered test".

**D16. Two stale pointers to the removed closed-form paragraph.**

- **Where:** `app_relation.tex` l.2; `app_wsg.tex` map, l.8.
- **Fix:**
  - **app_relation:** "that the main text summarises in one paragraph (\cref{sec:gap})" → "that the main text uses once (\cref{sec:limit})".
  - **app_wsg map:** "summarised in \cref{sec:gap}" → "used in \cref{sec:limit}".

**D17. The appendix map refers to an "evidence column" that Table 1 no longer has.**

- **Where:** `app_wsg.tex` map, l.5.
- **Fix:** "behind \cref{tab:claims}'s evidence column" → "behind \cref{tab:claims}'s Registered threshold and Result columns".

**D18. The claim index no longer matches Table 1.**

- **Where:** tab:claimindex in `app_prereg.tex`.
- **Problem:** It still lists "The closed-form account (post hoc, then tested)" as a claim of Table 1, but that row was removed. Its "Landing the copies in time does not rescue…" also differs from Table 1's "Issuing the copies a step earlier…".
- **Fix:** Drop the closed-form row, or relabel it "(App. H; no longer in \cref{tab:claims})". Rename the timing row "Issuing the copies a step earlier does not rescue the two-read path".

**D19. Table 2's footnote a drops "non-resident".**

- **Where:** Table 2, footnote a.
- **Problem:** The engine copies the top predicted *non-resident* expert (job 109 header, `pf`).
- **Fix:** "A copy, one layer ahead, of the non-resident expert that the next layer's router, applied to this layer's input, ranks highest."

**D20. Table 2 shows LA as read twice only.**

- **Where:** Table 2.
- **Problem:** LA appears only under "read twice", but an admission the layer-ahead copy predicted correctly is read once. Section 6 says so, and the old table said "2 or 1".
- **Fix:** Append to footnote a: "; an admission it predicted is read once".

**D21. Section 2 overstates what Table 2 lists.**

- **Where:** Section 2, Terms.
- **Problem:** "lists every configuration, machine set, yardstick and bound": Table 2 lists no bounds, and the window policies of Section 8 are not in it (the old caption pointed to App. G).
- **Fix:** "\Cref{tab:configs} lists every configuration (the window policies of \cref{sec:horizon} are in \cref{app:value}), machine set and yardstick".

**D22. Table 2's panel definition is ambiguous.**

- **Where:** Table 2, panel row.
- **Problem:** "rented before the registered tests" is ambiguous, because jobs 096–101 were registered too.
- **Fix:** "the \dmMachines{} machines of jobs 096–101 that ran every state; \dmRegN{} ran job 099, its main job".

**D23. The Eq. 3 sentence no longer says which range belongs to which machines.**

- **Where:** Section 4, "What is left".
- **Problem:** The shortening dropped which range goes with which machines, and "of that gap".
- **Fix:** "Against \cref{eq:dep}, the tighter bound there, the read-ahead oracle still leaves 83–99% of the gap to it on the two slow-link panel machines and 73–81% on the three RTX 4090s (median 45% on fast links): …".

**D24. The deviation that Section 7 points to is no longer reported anywhere.**

- **Where:** Section 7, parenthesis; app_prereg, Jobs 110 and 111.
- **Problem:** The removed footnote's fact, that the relaunched machines' first rounds came within 6% of the trend, is now in neither place. Section 7 still says App. D "reports the deviation". Recomputed from job 110's first rounds:
  - MIN-1R and the read-ahead oracle are within 6.2%;
  - MIN-2R on 110c is 6.4% off (log 0.062), outside Q1's ±0.06 band.
- **Fix:** Add to app_prereg's Jobs 110 and 111 paragraph: "Their first rounds had come within 6% of the trend for \MinOne{} and the read-ahead oracle; \MinTwo{} on one of them was 6.4% off, outside its ±0.06 band."

**D25. Twenty supplement tables come after its reference list.**

- **Where:** IEEE supplement.
- **Problem:** Table S5 (`tab_prereg`) is a `[p]` float. In the one-column supplement it waits for a float page and is flushed at `\end{document}`, after the reference list. It holds back every later table: S5 and S7–S25 land on pp. 23–30, after the references on p. 20. Example: S16 is cited in App. H on p. 12; S23 in App. J on p. 15.
- **Fix:** Make `tab_prereg` `[!tp]` (or add `\usepackage{placeins}` and a `\FloatBarrier` at the end of each appendix). In `ieee-supplement.tex`, add `\clearpage` before `\bibliographystyle`.

**D26. The supplement has no Table S1.**

- **Where:** IEEE supplement.
- **Problem:** The uncaptioned checklist `longtable` in App. B steps the table counter, so the tables start at S2. The paper's first-page note promises "S1, S2, …". The MLSys build has the same gap at Table 7.
- **Fix:** Put `\addtocounter{table}{-1}` after the checklist `longtable` in `app_wsg.tex`.

**D27. Both IEEE logs carry 43 harmless but unreported warnings.**

- **Where:** `ieee-paper.log`, `ieee-supplement.log`, `build_ieee.sh`.
- **Problem:** Each log has 43 "Label … multiply defined" warnings. xr v5.06 also imports the other document's `\bibcite`. Each document's own `.aux` is read later, so its own citation numbers win: the citations are correct (checked). But `build_ieee.sh` does not report these warnings.
- **Fix:** Optional:
  - Add `$(grep -c 'multiply defined' $f.log) multiply-defined (bibcite, expected)` to the summary line.
  - Or clear the imported `\bibcite` entries, for example `\makeatletter\def\XR@test#1#2#3#4\XR@{…}` limited to `\newlabel`.

**D28. The 0.5-line sentence leaves out Section 4.**

- **Where:** Section 2, Machines.
- **Problem:** "before the registered tests of \cref{sec:online,sec:secondcard}" leaves out Section 4. The test that first used the line, job 109, is Section 4's (Table 1: "the fast-link scope was drawn between the two").
- **Fix:** "…before the registered tests of \cref{sec:gap,sec:online,sec:secondcard}".
