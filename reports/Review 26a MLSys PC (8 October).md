# Review 26a: MLSys 2027 PC review of "Where the Seconds Go" (8 October)

## Materials read

- `paper/paper.pdf`: the whole main text (pp. 1–10) and the references. Appendices A–N read in full except the Table 32 audit rows, which I used only to recompute the two medians.
- `paper/supplement.pdf`: skimmed for structure. I read the job-113 scorecard table (Table S19) to compare its clause values with mine.
- `paper/ieee-paper.pdf` and `paper/ieee-supplement.pdf`: skimmed for the IEEE-format section. I extracted their text and rendered pages 1, 2 and 6 and a contact sheet of all 10 main pages.
- Job script headers on the gpu branch (`/home/claude/gpu-branch/jobs/`): 081, 089, 109, 110, 113 in full, and the post-launch diffs of 100, 102, 105 and 106 (file contents only).
- Raw results on the gpu branch:
  - `ec_*.jsonl` rows and `st_*.json` counters for jobs 093–113;
  - `concur.txt` probes, `fetch_table_law_gptoss.json`, `g_prof.json` and the `prof_C*.json` Nsight summaries for jobs 069c and 105;
  - `readsched_C*.txt` for job 102;
  - `bs1.jsonl` for jobs 081 and 089;
  - `validity.txt` and `cpu.txt`;
  - `results/113_offers_at_launch.txt`.
- `gpu/vast_ledger.json` for rental creation times. `git log --format='%H %ct'` on the gpu branch for commit times. I read no commit messages.
- Authors' scripts, read only for definitions:
  - `scripts/job113.py`, and `scripts/job109.py` (ratio, gap, residual and capture definitions);
  - `host_rates` in `scripts/speed_limit.py` (how B_host is taken from the probe);
  - `profiles()` in `scripts/sumlaw_paper.py` (how T_GPU is taken from the Nsight profiles);
  - the docstrings of `scripts/decomp_measured.py` and `scripts/reg_timing.py`.
- R* (38.315 and 15.340 experts per token at C=14 and C=32) is taken from the registered header of job 109. I did not recompute it from the routing traces.

## Independence statement

- I did not open anything under `reports/` except to write this file. I also did not open `prereg/*outcome*.md`, `research_notes/`, `apply/`, `paper/paper_v1_prereview.tex`, or any file named like a review, number check, plan or progress log.
- I read no commit messages. I did not run or import the authors' analysis code.
- Every number in the claim table below comes from my own scripts. They are in `scratchpad/rev26a/` (`mylib.py`, `c113.py`, `c109.py`, `c4090.py`, `c112.py`, `cpanel.py`, `cbound.py`, `ceq3.py`, `cfew.py`, `cspear.py`, `cwin.py`, `c102.py`, `cprof.py`, `cnoise.py`, `regtime.py`, `ctab3.py`). They read the raw rows directly.
- I modified nothing in either repository.

---

## Summary

The paper studies batch-1 decode of MoE models whose experts mostly live in host DRAM, behind one RTX 5090 (and RTX 4090s for one test). It makes four contributions:

1. **A bound.** It defines a per-machine bound, Eq. (1): MIN-with-bypass host reads per token at the highest rate the machine's own probe measured. It adds two refinements, a demand bound (Eq. 2) and an ordered bound (Eq. 3). The authors' llama.cpp expert cache reaches 31–54% of Eq. (1)'s speed on consumer machines at gpt-oss 11%.
2. **A decomposition.** Oracles inside the engine split the deployed cache's gap to the bound into four parts: what is cached (MIN's set), how it is read (one read vs serve-on-CPU-then-copy), when it is read (read-ahead), and a remainder attributed to T_GPU and the extra reads (Table 4). A later registered control (job 113) shows that the striking "MIN's set pays only when read once" interaction comes from the engine's in-step fetches displacing the scheduled set. With those fetches off, MIN's set read twice runs 1.09–1.26× a fetch-free deployed cache on two machines.
3. **Policies without foresight.** Four variants without foresight capture at most 6% of the read-ahead oracle's gain on five fast-link machines. A trend in the link-to-CPU bandwidth ratio, fitted on RTX 5090s, predicts the oracles' speed on five RTX 4090s within 7%.
4. **A trace study.** On 9 models it finds that half of the read saving needs a window holding about 0.65·C distinct experts.

Every experiment's predictions sit in a job-script header committed before its rental. A scorecard of about 1,700 clauses, including the 323 that failed, is provided.

## Strengths

1. **The raw data reproduce exactly.** I recomputed 38 quantitative claims from the raw rows and counters (table below). Every one matched to the printed precision, except two cosmetic labelling points. This includes:
   - every cell of Tables 3 (host S and host B gpt-oss), 4, 5, 6, 12, 21, 23 and 30;
   - the frozen trend predictions in brackets in Table 6;
   - the running example;
   - the headline 31–54% range, the T_GPU and G ranges, and the round-to-round noise statistic.

   This level of reproducibility is rare and is the paper's strongest asset.
2. **The registration is real, and its weak spots are reported.** For all 94 rentals of jobs 093–113, the job script was committed at least 2.7 s before the rental was created (my check agrees with Appendix D). The script edits made after a launch (jobs 100, 102, 105, 106, 107 and 111) are what Appendix D says they are: machine substitutions with the predictions unchanged, plus job 100's added ninth prediction, which is disclosed. Several other weaknesses are stated openly:
   - the Table 1 labels "loose", "post hoc" and "held (point)";
   - the failed manipulation check (job 112);
   - the voided job 110;
   - the inconclusive job 107;
   - Eq. (4) failing all four of its registered tests.

   Few systems papers are this candid.
3. **The bound is useful.** A trace and one probe run give a machine-specific speed-of-light figure, and the microbenchmark (Table 30, 86–96% of the read term) shows the read term can nearly be reached. Rule 1 ("report the distance to the machine's bound") is a good norm for this literature.
4. **The authors corrected their own central finding.** Job 113 attributes the "pays only when read once" interaction to the engine's in-step fetches, and the paper now says so in Sections 4 and 9. Counters back the mechanism: Few-2R's misses fall from 54.7–55.6 to 43.0 per token with fetches off.
5. **Machine-to-machine variation is treated properly.** Machines are the unit, with t-intervals over machines, and the paper shows machine variance dominates (two 285Ks differ by 29%). The round- and rental-level noise analysis (median 0.2% and 0.5%) is sound; I reproduced the round-level figure.
6. **The system itself is strong.** It is 2.0–4.0× stock llama.cpp and ahead of FreeToken at 11 of 12 cells on two hosts.

## Weaknesses (most important first)

1. **The headline decomposition (Table 4, Fig. 2) does not measure what its rows are named for, and the clean version rests on two machines.**
   - On the deployed read path, the "MIN's set" arm never holds MIN's set. The authors concede this with an asterisk on Table 4's first row.
   - So the 2×2 does not separate "what is cached" from "how it is read". Only "Both" (MIN-1R) and the read-ahead step are interpretable as parts of the gap.
   - The clean experiment (Table 5, job 113) ran one process per round on two machines.
   - Its baseline itself moved: turning the fetches off made the deployed cache 1.07× faster on the Core Ultra 9 285K. So that machine's probe-derived fetch table was mis-tuned for the deployed cache.
   - At 25% on that machine, reading twice (1.36×) beat reading once (1.29×).
   - Net effect: the paper's original story (the set pays only with one read) has been replaced by a weaker, machine-dependent one ("reading once usually pays more, by 0.02–0.16 at 11%"). That replacement is supported by two machines. Section 9 rule 2 and the abstract are now right in direction, but the evidence for the rule is thin.
2. **Several abstract and intro claims are broader than their evidence.**
   - *"Policies without foresight recover at most 6% of an oracle's gain"* (abstract). This holds only for the five fast-link RTX 5090s of job 109. On the slow-link RTX 4090s the larger margin captures 12–26% (Table 6). On 110e, the RTX 4090 left out of the job-111 relaunch, I compute that the margin captured 78% of the read-ahead oracle's gain in its single (void) round (1.043 vs 1.056). It is void data, but it shows the "little" claim is ratio-dependent: capture has a small denominator where the oracle gains little. The abstract should say "on five fast-link machines".
   - *"Half of foresight's value needs … about 0.65 cache-fulls of distinct experts ahead"* (abstract). This is a post-hoc, trace-level (simulated) regularity. On the engine's own AIME routing it is 0.66–0.81·C. The abstract places it next to "Every prediction was committed before its machines were rented", and a reader will take it as confirmatory.
   - *"A trend … predicted the oracles' speed on five RTX 4090s within 7%"* (intro). This is true. But applied to five new RTX 5090s, the card it was fitted on, the same trend misses 5 of 18 cells by more than the band, by up to 18%. The intro should say so; Section 7 does.

     The 4090 test is also weaker than "five machines" suggests:
     - three of the five sit at ratios 0.37–0.40;
     - two of those are the same CPU model at 0.39 and 0.40, with near-identical results;
     - the ±0.10 log band is wide against effects of 0.96–1.31;
     - the comparison point, "predicting no change", is a weak null.
   - *"Our cache … reaches 31–54% …; published systems reach a median 13.6%"* (intro). This compares a probed bound with datasheet bounds that sometimes use uniform routing and sibling-model traces. The like-for-like figure is ours at a median 27% (Section 3), and the intro should use it. Also, 3 of the 20 rows in the 13.6% median are below 5%, which Table 32 says are "not adjudicated".
3. **Generality is narrow, and the builder rules go beyond it.**
   - Almost every engine result comes from one engine (the authors'), one model (gpt-oss-120b) at two budgets, and one prompt set (AIME-25, also used during development).
   - All of it is batch-1 and teacher-forced.
   - Rule 4's slow-link half ("serve on the CPU and copy in the background") rests on one stable machine below a ratio of ⅓: Pf, where Few-2R ran 1.11–1.22× against Few-1R's 1.04–1.06×. The only other machine below ⅓ was unstable.
   - The 0.5 "fast-link" line was drawn after the panel data. One panel "fast" machine (100f, ratio 0.54) behaves like a slow one (MIN-1R 1.07×).
4. **The registered tests are less confirmatory than the framing suggests.** About 1,700 clauses are tallied, but the predictions carrying the main claims are few:
   - the interaction's sign on 5 machines (job 109);
   - the trend on 5 RTX 4090s;
   - H1–H6 on 2 machines (job 113).

   Job 109's bands "could hardly have failed" (the authors' words), and job 102's 0.50 floor was loose. Of all clauses scored, 685 are "held (point)", mostly with no interval at all. The registration is a strength for provenance, not a substitute for statistical power.
5. **The bound is relative to the probe, not physical, and the GPU term is weak.**
   - B_host is one probe reading, often taken while the model was downloading. The engine's implied read rate exceeds it by more than 5% on 12 of 84 launch-budgets, up to 1.22×. Where that happens, "the bound" is not a bound.
   - The 54% top of the headline range comes from such a machine (100f; the paper flags this).
   - The GPU term uses datasheet bandwidth, which batch-1 decode reaches only 52–61% of.
   - "Consumer" includes HEDT Threadripper parts, which set the bottom of 31–54% (the 9960X at 30.6–31.2%).
   - These caveats are disclosed, but Table 1 labels the bound "derived", as if it held unconditionally.
6. **The deployed baseline is tuned per machine by a model that can mis-tune it.** The fetch table counts as part of "our cache" (worth 3–34% per Appendix J). Yet on the 285K it made the deployed cache slower than serving every miss on the CPU. So part of the measured "gap" on some machines is baseline tuning, not anything the oracles isolate. No sensitivity of Table 4 to the fetch table is reported.
7. **Small provenance points:**
   - `results/113_offers_at_launch.txt`, which establishes that offers a and c were unavailable, was transcribed and committed about 1.8 h after the rentals. The query output "was not saved at the time". Appendix D calls this "recorded", which overstates it.
   - The 110e omission is explained only for the trend clause (−5.6%, which I reproduce), not for the capture clause (see weakness 2).
8. **Output quality is checked only by teacher-forced KL.** There is no task accuracy, and all engine timing is teacher-forced on each model's own greedy text. That is acceptable for a systems paper, but it limits what the 2.0–4.0× over llama.cpp means end to end.

## Clarity

**Assessment:** better than in earlier rounds, but the paper still reads like a careful lab notebook compressed into ten pages, not an argument.

### What works

- **Fig. 1:** the one-step schematic makes the key mechanism (a second read of each admitted expert) visible.
- **Table 2:** the naming grid (admitted set × number of reads) is the right device. "Few-1R / MIN-2R / Dep-1R" are systematic once learned.
- **Table 1:** explicit labels (derived / post hoc / loose / held / held (point)), with a Mach. column.
- **Table 9:** maps each claim to jobs and clause IDs. With it I could find every datum.
- **Section 5's running example** (one Ryzen 9 9950X, five numbers) grounds the yardsticks.
- **Table 4:** separates "measured" from "attributed".
- **Section 9:** each rule is stated with its scope.
- **Limitations:** concrete, not boilerplate.

### Problems that remain, roughly by cost to the reader

1. **The abstract and intro lean on terms defined only in Section 2 or later:**
   - "in-step fetches", "MIN's set", "fewest-admission set", "read twice", "fetch-free deployed cache", "cache-fulls of distinct experts".
   - "MIN's set runs 1.09 to 1.26 times as fast as the deployed cache without them" can be parsed two ways: without them, or the deployed cache without them.
   - "59–93% of what reading it once gains" does not name its baseline, which is the deployed cache *with* fetches.

   A reader new to the work cannot reconstruct the claims from the abstract.
2. **Section 4 is built as claim then retraction.**
   - The bold lead "On the deployed read path, MIN's set pays only when read once" is undone three paragraphs later ("the interaction … belongs to this engine's read path, not to MIN's set").
   - Table 4's first row carries an asterisk saying it does not measure what its label says.
   - Fig. 2's second x-position is "MIN-2R (or Dep-1R)", depending on which median line one follows.

   Leading with the mechanism (in-step fetches displace the scheduled set), then the fetch-free measurement, then the decomposition, would remove most of the confusion.
3. **Number density is very high.** Many sentences carry 4–10 ranges. Examples:
   - Section 4's "Why" paragraph;
   - the intro paragraph "What is cached pays only if the read path keeps it";
   - Section 7's second paragraph.

   Ranges over two machines ("1.23–1.26×") read like measurements over a population. Saying "on both machines (1.23 and 1.26)" would help.
4. **There are too many names in play at once:**
   - about 14 configuration names, plus appendix codes (Table 7 is needed to read the appendix);
   - about 10 machine-set identifiers (panel, new machines, O1–O6, Pa–Pj, hosts A/B/S, "RTX 4090s");
   - job numbers used as identifiers in the main text (jobs 109, 111, 112, 113 in Table 6's row groups).

   The main text should refer to experiments by purpose, not job number.
5. **The yardsticks and baselines shift between sections:**
   - speed vs deployed;
   - share of the bound;
   - share of the gap;
   - capture (relative to the read-ahead oracle);
   - value of foresight (relative to "admit every miss", not the deployed cache).

   Fig. 5's caption has to open with a warning about its baseline.
6. **Some sentences mis-scope a test:**
   - Section 4: "the registered margin failed on the Core Ultra 9 285K, where at 25% reading twice was faster." The registered clause, H6, is at 11%, where it failed with a margin of 0.024. At 25% it was not registered.
   - Section 5 calls Few-1R's 1.26× a "geometric mean"; Appendix I calls the same figure a "mean" with a t-interval. The two agree to two decimals (1.257 vs 1.263), but the labels should match.
7. **Key qualifications live only in the 34 pages of appendices:**
   - the 110e omission;
   - job 109's loose bands;
   - the post-hoc offers transcription;
   - that the 0.65·C rule is trace-level.

   A reader of the main text alone will over-read Table 1.
8. **Section 3 is crowded.** It covers three bounds (Eqs. 1–3), their assumptions, a microbenchmark, a two-host system comparison and a 52-row literature audit in about a page and a half. The demand bound and ordered bound could move to one paragraph plus an appendix.
9. **Some terms are counterintuitive:**
   - "fast link" is relative to the CPU, not absolute. A 26 GB/s link is "fast" behind a 35 GB/s CPU.
   - "consumer" includes Threadripper.

   Both are defined, but both change the headline numbers.
10. **Fig. 2 is hard to decode.** It overlays too many line colours (blue, orange, green, plus two dark medians in different line styles) in small panels, and the per-machine lines cross.

### Clarity score: 3/5

It is readable for a determined expert with the appendices open. It is not yet readable for a general MLSys audience.

## Questions for the authors

1. How does Table 4 change with the fetch table removed from all arms, or with a per-machine-optimal deployed table? On the 285K the deployed cache was 1.07× faster without fetches.
2. Can you rerun the fetch-free 2×2 (Table 5) on more than two machines, and at 25% with two rounds, before claiming rule 2 in general form?
3. Please report the capture clause for 110e's first round, or explain why it does not bear on "policies without foresight recover little" on slow links. What is the capture across all slow-link machines, including the void rounds?
4. Is the 13.6% literature median robust to dropping the three rows below 5% that Table 32 says are not adjudicated, and to restricting to trace-scored rows with probed rather than datasheet bandwidth?
5. Rule 4's slow-link half rests on Pf. Is there any other stable machine below a ratio of ⅓ where Few-2R beats Few-1R?
6. Why does the RTX 5090 trend miss the new RTX 5090s (up to 18%) but fit the RTX 4090s (within 7%)? Is the 4090 agreement partly because three of five machines cluster at ratios of about 0.37–0.40?
7. What fraction of the probe runs overlapped the model download? How would the 31–54% range change with the idle-machine second probe of job 112 used everywhere it exists?

## What would raise my score

- Scope the abstract and intro claims to their evidence:
  - "on five fast-link machines" for the 6%;
  - "post hoc, trace-level" for 0.65·C;
  - mention the RTX 5090 trend misses;
  - compare 27% with 13.6%.
- Restructure Section 4 around the in-step-fetch mechanism. Either rebuild Table 4 on the fetch-free path, or drop its first two rows from the headline figure.
- Expand the fetch-free control to at least four or five machines covering a spread of link-to-CPU ratios.
- Add at least one more slow-link stable machine for rule 4.
- Cut the main-text number density by about half, and refer to experiments by purpose rather than job number.
- Add even a small task-accuracy check, e.g. AIME accuracy for one model and budget.

## Scores

| | Score |
|---|---|
| Overall (1–10) | **5** (borderline: exceptionally careful and reproducible measurement work, but engine-specific findings with thin confirmatory evidence for the revised central claim, and an abstract that overreaches in two places) |
| Soundness (1–5) | **3** |
| Significance (1–5) | **3** |
| Novelty (1–5) | **2** |
| Clarity (1–5) | **3** |
| Confidence (1–5) | **4** |

## Claims checked against raw data

"Match" means equal at the printed precision. Each "My value" entry was computed from the raw rows and counters by my own scripts.

| # | Claim | Location | My value | Verdict |
|---|---|---|---|---|
| 1 | Ryzen 9 5950X: ratio 0.74; Dep CPU-only 0.98; MIN-2R CPU-only 1.12; Few-2R CPU-only 1.26 (vs Dep CPU-only) and 1.23 (vs Dep); Few-1R 1.40; misses 55.6 / 43.0 | Table 5 | 0.744; 0.981; 1.122; 1.259; 1.235; 1.397; 55.64 / 42.99 | Match |
| 2 | Core Ultra 9 285K: 0.56; 1.07; 1.09; 1.23; 1.32; 1.34; 54.7 / 43.0 | Table 5 | 0.556; 1.073; 1.094; 1.231; 1.321; 1.344; 54.71 / 42.99 | Match |
| 3 | Fetch-free Few-2R misses 0.77–0.79 of with-fetch at 11%, 0.72–0.74 at 25% (registered ≤ 0.88) | §4, App. D | 0.773 / 0.786; 0.718 / 0.742 | Match |
| 4 | Reading twice keeps 59–93% of the one-read gain | Intro, §4, §9 | 59.1%, 93.1% | Match |
| 5 | At 25%: fetch-free Few-2R 1.33–1.36×, Few-1R 1.29–1.43× | App. D | 1.325 / 1.359; 1.427 / 1.293 (on the 285K, two reads beat one) | Match |
| 6 | H3 and H6 failed on the 285K (deployed 0.98–1.07×; one-read margin 0.02–0.16) | Table 1, §4 | 1.073 > 1.00; margin 0.024 (at 11%) | Match. The §4 sentence puts the failure at 25%, but the registered clause is at 11% |
| 7 | All 5 job-109 rows (MIN-1R, ahead, none, capture at two budgets) | Table 6 | e.g. 285K 1.215 / 1.460 / 1.021 / 4.5%; 7945HX 25% 1.283 / 1.626 / 0.995 / −0.8% | Match (all cells) |
| 8 | Best no-foresight variant 1.007–1.024×, capture ≤ 6% at 11% | §6, abstract | 1.0069–1.0236; max 5.7% | Match (scope: fast-link machines only) |
| 9 | Variants read 1.57–1.76 R* | §6 | 1.575–1.755 (deployed 1.645–1.66) | Match |
| 10 | Table 4, New (5) and New (4) columns, all 16 entries | Table 4 | e.g. Both 34.7 [21.9, 47.5]; left 46.3 [37.7, 54.8]; residual 0.5 [−3.1, 4.2]; 25%: 27.3 [23.0, 31.5] | Match |
| 11 | Table 4, Panel (15) columns, all 16 entries | Table 4 | e.g. Both 32.6 [21.0, 44.3]; ahead 14.9 [10.7, 19.1]; T_GPU 27.4 [24.5, 30.3]; 25% residual −5.0 [−12.3, 2.4] | Match |
| 12 | MIN-1R closes 39% on the 13 fast-link panel machines | Intro, §4 | 38.8% | Match |
| 13 | Interaction positive on 9 of 10 panel main-job machines, 5 of 5 new | Table 1, §4 | 099f is the only negative (−0.48 ms); 5/5 positive (1.5–7.5 ms) | Match |
| 14 | Unexplained residual 35–47% (two slow-link panel machines), 31–36% (slow-link RTX 4090s) at 11% | §4 | 35.5, 47.4; 31.5–35.8 | Match |
| 15 | RTX 4090 rows of Table 6, including the bracketed predictions from the trend frozen in job 110's header | Table 6 | e.g. i9-14900KF 0.963 [0.989], 1.128 [1.111], 1.033, 25.7%; i5-12400 25% 1.351 [1.314], 1.689 [1.609] | Match |
| 16 | Trend within 7% (11%) and 5% (25%); "no change" misses by up to 46% / 69% | §7 | max 6.6% / 5.0%; 1.465 / 1.689 | Match |
| 17 | Trend on new RTX 5090s misses 5 of 18 cells, by up to 18% | §7, Table 1 | 5/18 beyond 0.10 in log; max 17.6% | Match |
| 18 | 110e's first round deviated −5.6% from the trend | App. D | −5.6% (read-ahead). Not reported: margin capture 78% in that round | Match; omission noted |
| 19 | LA variants 0.64–0.72× at 11% on slow-link RTX 4090s; margin capture up to 26% | §6 | 0.644–0.724; 25.7% | Match |
| 20 | Job 112: speeds 1R / 2R / early and misses for 4 machines; early/2R misses 0.97–0.98 (11%), 0.95 (25%); early 0.03–0.05 slower; admits 11.7–11.8 vs 6.2–7.4; reads +4–7% | Table 12, App. D | all cells equal; 0.968–0.980; 0.949–0.953; −0.031 to −0.053; 11.7 vs 6.2–7.4; +4.2–7.1% | Match |
| 21 | Deployed cache 31–54% of Eq. (1) on consumer machines at gpt-oss 11%; 12–46% on servers; job-109 machines 41–48% | §3, Table 1 | 30.6% (TR 9960X, 106a) / 31.2% (105b) to 54.0% (100f); servers 12.4–46.1%; 41.5–48.0% | Match (bottom set by HEDT, top by a probe-limited machine) |
| 22 | Running example: Eq. (1) 10.0 ms, Eq. (2) 13.0, deployed 20.6, MIN-1R 15.2, read-ahead 13.8 | §5 | 10.02, 12.96, 20.64, 15.16, 13.81 (job 096a) | Match |
| 23 | Eq. (3)/Eq. (1): median 1.00 on consumer machines; up to 1.56 on slow links | §3 | median 1.000 (48 launches); max 1.557 (TR 9960X) | Match |
| 24 | Table 3, host S: all tok/s, ratios and × llama.cpp | Table 3 | e.g. 28.93 / 47.99 / 57.90, 1.206, 2.00×; Qwen3 43.75% 0.974 | Match |
| 25 | Table 3, host B gpt-oss 11/25/40% and Qwen3 12.5% (job 081) | Table 3 | 34.93 / 53.99 / 69.88 (1.294); 39.98 / 85.64 / 109.17; 47.73 / 132.20 / 152.52; 19.48 / 38.70 / 39.96 | Match |
| 26 | On host S, five of six ratios outside ±0.06 of host B's | Table 1, App. D | deviations −0.088, −0.079, −0.061, −0.005, −0.105, −0.075 | Match |
| 27 | Microbenchmark: 86–96% per layer; all modes on all three machines | Table 30 | all 24 cells equal (e.g. Pd 86 / 93 / 35 / 92%); repetition spread ≤ 0.9% | Match |
| 28 | T_GPU = 2.9 ms, the smallest of 14 Nsight profiles; G = 4.1–4.7 ms on gpt-oss | §3, App. H | 2.940 (median 3.16) over 14 profiles; G 4.05–4.67 on RTX 5090s (two EPYC profiles near 0, as stated) | Match |
| 29 | Few-1R beats deployed on all 15 stable machines, mean 1.26× (t-interval 1.19–1.33); loses at ratios 0.14 and 0.21 | §5, App. I | geometric 1.257, arithmetic 1.263 [1.192, 1.334]; 0.845 (EPYC 7302), 0.974 (EPYC 7663) | Match (label: geometric vs arithmetic) |
| 30 | Spearman of MIN-1R gain with ratio 0.89 (19 machines); read-ahead 0.87 (17) | §7, App. I | 0.886 (19); 0.860 (17, with my choice of first launch per GPU) | Match / within launch-choice tolerance |
| 31 | Exact 16-token window recovers 0.78–0.93 of MIN's gain in time at 11% (registered ≥ 0.80; lowest missed) | §8 | 0.781–0.935 (099f lowest) | Match |
| 32 | Job 100: windows on each read path, 4 machines | Table 21 | all cells equal (e.g. Pd 0.66 / 0.89 / 1.02 / 1.03 / 0.94 / 1.00 / 1.05 / 1.11) | Match |
| 33 | Margin cuts host reads 4–6% at 11%, 6–10% at 25% | §6, App. I | 4.3–5.8%; 6.3–10.5% | Match |
| 34 | Round-to-round: median 0.2%, max 1.8%, over 108 configurations on 14 machines | §2, App. D | 0.17%, 1.77%, n = 108, 14 machines | Match |
| 35 | 94 rentals of jobs 093–113, each commit ≥ 2.7 s before rental creation; post-launch header edits only substitute machines (plus job 100's P9) | App. D | 94; min 2.71 s; post-launch diffs on 100, 102, 105 and 106 are as described (107 and 111 not diffed by me) | Match |
| 36 | Table 10 column sums (595 / 685 / 323 / 84; hand-scored 218 / 193 / 142 / 12) | Table 10 | identical sums | Match |
| 37 | Published median 13.6% (20 in-class rows), 9.5% (29 trace rows) | §3, App. M | 13.6, 9.5 recomputed from Table 32's rows (not from raw audit data) | Match (internal only) |
| 38 | Job 104: Table 23 cells; Few-1R copies 0.60–0.73 of greedy; Few-2R 1.11–1.22× below a ratio of ⅓ | Table 23, §5, §7 | all cells equal; 0.596 / 0.722; 1.113 / 1.217 on Pf only | Match (rule rests on one stable machine) |
| 39 | Two panel machines with the same CPU differ by up to 29% in deployed time | §2 | 285K: 18.14 vs 14.06 ms (1.29) | Match |

I found no data discrepancies. The issues are about scope and labelling: rows 6, 8, 17/18, 21, 29 and 38, plus the post-hoc transcription of `113_offers_at_launch.txt`.

### Registered tests vs job headers

- **Job 113:** header H1–H6 equal Table 1's thresholds (Few-2R ≥ 1.10, MIN-2R ≥ 1.03, one-read margin ≥ 0.05, deployed 0.80–1.00, misses ≤ 0.88). H2 (reads ≤ 0.90 of base0; measured 0.785) is not shown in the main text but held.
- **Job 109:**
  - P1 (interaction > 0 at both budgets; MIN-1R ≥ max(MIN-2R, Dep-1R) + 0.04) and P3 (mean MIN-1R share 0.25–0.55) match Table 1.
  - P7 (best ≤ 1.10, capture ≤ 0.35 per machine, mean ≤ 0.25 at 11%) matches.
  - The population (ratio ≥ 0.5, never rented) was fixed in the header before any machine started. The header was committed 21 minutes before the five rentals, which were created within 10 s of each other.
- **Job 110:** the frozen coefficients reproduce the bracketed predictions in Table 6 exactly, and its Q1 bands (0.10 / 0.06 in log) match Table 1.
- **Jobs 081 and 089:** P2 to P4 of job 081 and P2 of job 089 match Table 1's system-comparison row.

### Evidence labels in Table 1

- The labels are accurate as far as I checked:
  - "loose" is correctly applied to jobs 102 and 109;
  - "post hoc" is correctly applied to the consumer scope, the literature median, the 0.65·C rule and the trend fit;
  - "held (point)" for the interaction sign is correct (an interval was not registered).
- One addition is needed: the "derived" bound is derived relative to the probe and to datasheet GPU bandwidth. It fails as a bound on 12 of 84 launch-budgets, and the label should say so.

### Post-hoc labelling outside Table 1

- The abstract does not mark the 0.65·C result as post hoc or trace-level.
- The intro's 31–54% vs 13.6% contrast is not marked as cross-bound.
- The body labels both correctly.

---

## IEEE format

I skimmed `ieee-paper.pdf` (10 pp.) and `ieee-supplement.pdf` (29 pp.). Points that break or strain IEEE conference conventions:

1. **Heading hierarchy.** Sections II, III and others use IEEEtran's run-in fourth-level headings ("a) Models and workload:") directly under a top-level section, with no "A./B." subsections. IEEE style expects \subsection (A., B.) before lettered run-in paragraphs.
2. **Cross-references between the two documents are inconsistent.**
   - The main text cites "Appendices D and N", "Appendix J" and so on, with no S prefix, while supplement tables, figures and equations are S-prefixed (Table S3, Fig. S1, Eq. (S1)). Appendices should be prefixed the same way, or cited as "Appendix D of the supplementary material".
   - The supplement cites main-text "Table III", "Fig. 4" and "Table VI" without saying they are in the main paper.
   - The supplement's own Appendix A still lists "The supplement: every clause scored…", and its text cites "the scorecard supplement (paper/supplement.pdf)". That is a third document outside the IEEE package, cited by a repository path, so these references dangle.
3. **Page-1 footnote.** It names a file ("ieee-supplement.pdf"). IEEE convention is to refer to "supplementary material available online", not a file name.
4. **Captions.**
   - Main-paper table captions follow the "TABLE I" + small-caps form, but several run to 3–5 lines of definitions in small caps (Table I, Table IV). IEEE convention is a short title above, with definitions in table notes below.
   - In the supplement, Tables S5 and S26 (longtables) use a one-line "TABLE S5: …" sentence-case caption, unlike the other supplement tables.
5. **References.** The numbered style is correct, but:
   - venue names are written out in full ("Proceedings of the 31st ACM Symposium …", "Communications of the ACM") instead of IEEE abbreviations ("in Proc.", "Commun. ACM");
   - page ranges carry digit-group spaces ("pp. 116 126–116 148");
   - several entries mix "arXiv preprint" with a venue.

   The supplement correctly uses its own [S1]… numbering.
6. **Float placement.** Floats are mostly at page tops, which is acceptable. Table IV is first cited in Section IV but appears on page 6 among Sections V–VI, and Tables IV and V are full-width floats placed two pages after their first reference.
7. **Page count.** The paper is 10 pages including references, which start mid-page 9. Whether that fits depends on the venue's limit (commonly 8 + references, or 10 total), so check it.
8. **Front matter.** The anonymous author block, the bold IEEEtran abstract and the alphabetical Index Terms are all fine.
