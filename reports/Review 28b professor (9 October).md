# Review 28b: "Where the Seconds Go" (professor, scientific-benchmarking referee), 9 October 2026

## Materials read

- `paper/paper.pdf` (47 pages): the whole main text (pages 1–9), references, and Appendices A–N in full. I read Appendix D (prediction log), E, H, I, J, L and N closely and skimmed F, G, K and M.
- LaTeX sources: `paper/main_body.tex`, `abstract_body.tex`, `paper.tex`, `tab_job113.tex`, `ieee-paper.tex` and `ieee_preamble.tex`.
- `paper/supplement.pdf` (the scorecard): the front matter and the first job blocks (073–084), plus a check of its table numbering.
- `paper/ieee-paper.pdf` (10 pages): read in full, with pages rendered to check layout. `paper/ieee-supplement.pdf` (31 pages): skimmed, including its front matter, numbering and reference list. Build logs: `ieee-paper.log` and `ieee-supplement.log`.
- Artifact:
  - `gpu/vast_ledger.json`, all 160 rows.
  - `prereg/speed_limit_v2.json`, for R\* and the trace path. I also recomputed R\* independently from the trace.
  - The first 120 lines of `scripts/job114.py`, read only to learn the field semantics of the raw rows.
  - Directory listings of `scripts/` and `prereg/`, names only.
- Raw results (`/home/claude/gpu-branch`):
  - `results/` for jobs 081, 084c, 089, 093–115. I used `ec_*.jsonl` (`decode_ms`, `n_decode`, `nll_*`, `config`/`stats=`), `st_*.json` counters, `concur.txt` and `concur_idle*.txt` probes, `fetch_table_law_gptoss.json`, `prof_C*.json` and `g_prof.json` (Nsight), `readsched_C*.txt`, `bs1.jsonl`, `v0.txt`, `gate.txt`, `tables.txt`, `cpu.txt`/`lscpu.txt` and `nvidia-smi-q.txt`.
  - The headers of `jobs/109_onlinepolicy@vast.sh`, `110_secondcard@vast.sh`, `113_heldset@vast.sh`, `114_decomp0@vast.sh` and `115_moremachines@vast.sh`.
  - Content diffs of the post-launch header edits of jobs 106, 111 and 115.
  - `git log --format='%h %ad'` / `%ct` for every job script of jobs 093–115.
- My own scripts are in `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/rev28b/`:

  | Script | What it checks |
  |---|---|
  | `minbypass.py` | My own MIN-with-bypass implementation |
  | `load.py` | Raw-row loader and probe parser |
  | `t5.py` | Table 5 |
  | `t4.py` | Table 4, fetches-off column |
  | `panel.py` | Table 4, panel column |
  | `t6.py`, `t12.py` | Tables 6 and 12 |
  | `t13.py` | Table 13 |
  | `t3.py` | Table 3, host S |
  | `noise.py` | Round-to-round noise |
  | `share.py` | Share of the bound over all launches |
  | `few1r.py` | Few-1R across launches |
  | `boundcheck.py` | Does any run beat Eq. (1)? |
  | `regtime.py` | Commit times against rental starts |

## Independence statement

- I did not open anything under `reports/` (I only write this file there).
- I did not open `prereg/*outcome*.md`, `research_notes/`, `apply/`, `paper/paper_v1_prereview.tex`, or any file named like a review, number check, plan or progress log.
- I did not read commit messages. I used only `git log` with hash/date formats, and `git diff` for file contents.
- I modified nothing in either repository except writing this file. I did not commit.
- My one reading of an analysis script (`scripts/job114.py`) was to learn the definitions of the raw fields. Every number in the claim table was computed by my own code from raw rows, counters, probes or traces.
- Where I used a paper table value instead of raw data, the claim table says so: 3 of the 15 Few-1R values.

---

## Summary

The paper studies batch-1 decode of MoE models whose experts mostly live in host DRAM: gpt-oss-120b and Qwen3-30B-A3B on rented RTX 5090 (and RTX 4090) machines.

**The bound.** The authors define a per-machine lower bound on time per token (Eq. 1): the host reads of Belady's MIN with bypass on the model's routing trace, divided by the machine's highest probed host read rate. Two tighter variants follow: a demand bound in series with the GPU's non-expert time (Eq. 2), and an "ordered" bound in which CPU reads cannot be issued early (Eq. 3).

**The engine and the decomposition.** The authors' llama.cpp expert cache beats FreeToken and stock llama.cpp at most settings. It reaches 31–54% of Eq. (1)'s speed on consumer machines (a post hoc scope). Oracles in the engine split the gap into three parts:
- what is cached (MIN's set);
- how it is read (once, copied in the step, or twice, CPU-served and then background-copied);
- when it is read (read-ahead).

**The central finding of Section 4.** The deployed read path has a per-machine "fetch table" that copies misses into slots regardless of the policy. On that path MIN's set alone gains nothing. With the table off (5 machines, two registered jobs), MIN's set alone closes 19% [13, 25] of the gap and both changes together 36%. About half of the gap remains even under the read-ahead oracle.

**The other results.**
- Online variants built from the engine's own levers capture at most 6% of the oracle's gain (registered, 5 machines).
- A trend in the link-to-CPU ratio fitted on RTX 5090s predicted RTX 4090 oracle speeds within 7% (registered).
- In trace replay, half of foresight's value needs a window holding about 0.65·C distinct experts (post hoc).

**Process.** Every experiment's predictions were committed to a public branch before the rentals started. The appendices score all 1,741 clauses (639 held with an interval, 694 held on the point estimate, 324 failed, 84 untested or void).

**Verdict.** The study is sound and unusually reproducible: every number I checked against raw data reproduced. Its main limits are three:
- **Generality:** one engine, and one model at two budgets for the decomposition.
- **The statistical unit:** small, clustered machine samples summarized with t-intervals.
- **Weak registered tests:** several are loose by the authors' own account.

Clarity has improved again but remains the paper's biggest practical obstacle.

## Strengths

1. **Registration practice is exemplary and verifiable.**
   - For all 107 rentals of jobs 093–115, the job's header (and the host's wrapper) was committed before the ledger's start time. By local commit time the minimum lead is 4 s for headers and 3 s for wrappers.
   - Every header edit after launch (jobs 100, 102, 105, 106, 107, 111, 115) exists in the history and is disclosed in Appendix D. This includes the awkward one: job 115's amendment text landed about 15 s (commit time) after host d started.
   - Failed and voided tests (jobs 107, 110) are reported, not dropped.
   - Table 10's tallies sum correctly: 639/694/324/84, with 565 hand-scored clauses.
2. **The numbers reproduce exactly from raw data.** In Tables 3 (host S), 4, 5, 6, 12, 13 and 31 every cell I recomputed matched to the printed precision, intervals included. The counters behind the mechanism claims (48.1–50.8 vs 43.5 misses; 20.4 vs 8.2 admissions) match too. My independent MIN-with-bypass implementation gives R\* = 38.315 (C=14) and 15.340 (C=32), identical to the artifact's.
3. **The bound survives falsification.** Across 1,297 gpt-oss configuration-runs in jobs 093–115 (both budgets, every policy and oracle), no measured time per token is below Eq. (1). The closest is the read-ahead oracle at 1.33× the bound, on the two Ryzen 9 5950Xs. The paper does not report this direct check (see Weakness 4).
4. **The engine is used as a measurement instrument.** Oracles inside a real engine, with counters that verify each manipulation, are the right way to attribute time:
   - MIN-1R's counters are identical across 19 machines (39.73 misses and 19.81 fetches per token).
   - Few-2R's misses fall to 0.77–0.79 with the table off, so the manipulation demonstrably held.
   - The timing control of job 112 is reported as uninformative, not spun.
5. **Hoefler–Belli practices are taken seriously.**
   - Absolute numbers sit beside every ratio.
   - Ratios are of mean times, with paired bootstrap intervals over problems.
   - Round-to-round and rental-to-rental variance is quantified: median 0.21% over 143 comparisons on 19 machines, which I reproduced.
   - Both bounds appear in the main figure.
   - A rule-by-rule checklist (Appendix B) admits its one departure (Table 3's arithmetic mean of rates).
6. **The negative results are reported prominently:**
   - the panel interaction failed on one of 10 machines;
   - Eq. (4) failed all of its registered tests;
   - the timing control failed;
   - the trend misses on new RTX 5090s.

## Weaknesses (most important first)

### 1. The headline mechanism is a property of this engine, and the "deployed" baseline is not tuned

Section 4's central story is that the deployed path's in-step fetch table displaces MIN's set. That is a fact about this engine's read path, not about MoE offloading. The decomposition rests on one model (gpt-oss-120b) at two budgets and one prompt set (AIME-25, also used in development).

The raw data show that the deployed fetch table is itself counterproductive on several machines. With the table off, the deployed cache ran:
- 1.056× on the Core Ultra 7 265K (114c);
- 1.076× on the Core Ultra 9 285K (115c);
- 1.073× on job 113's Core Ultra 9 285K.

So on Intel hybrid desktops the "gap" includes a self-inflicted 9–12% of the gap (Table 4's "in-step fetches themselves" row, my values +9.1% and +12.0%). The baseline against which every share is computed is therefore partly an artefact of a mis-calibrated fetch-table model.

The rules for builders acknowledge one engine. But the abstract and Table 1 present "MIN's set pays only when read once" as a finding about the problem, when it is a finding about this read path.

### 2. The statistical unit: small, clustered, partly overlapping machine samples summarized with t-intervals

The paper rightly makes the machine the unit, but the samples do not behave like independent draws.

- **Clustering.** The fetches-off column (n=5) consists of two near-identical Ryzen 9 5950X boxes and three Intel desktops. The 5950Xs come from one provider: nearby offer IDs 54573659 and 54573923, plus 54573925 for job 113's machine and 54573924 for the GPU of 108f/114g. They share bus 06:00.0, 125 GB of RAM and Bhost 34.4–36.4 GB/s, and their Table 5 rows are nearly identical.
  - The per-machine "Both" shares are 55.5, 55.5, 22.4, 26.3 and 18.1%, a bimodal set summarized as 36 [13, 58].
  - The interaction shares are 21.1, 20.8, 3.2, −3.3 and 2.3%.
  - A t-interval over five values from two builds is not a meaningful population statement. The two RTX 4090 Core Ultra 9 285Ks (111f/111g) are likewise twins (Bhost 63.1, speeds 1.198 vs 1.196).
- **Overlap.** One GPU (UUID f90ec646) is "new machine" 109f, job 112's 112g and "fetches-off machine" 114f. Pf (b8316d93) appears in jobs 099, 101, 103, 104, 105 and 106. Appendix D discloses this, but Table 2 presents the machine sets as disjoint.
- **"Ran stably" is undefined for most machines.** Stability is defined as the deployed cache's time agreeing within 2% between rounds. Every launch of jobs 093–105 that I scanned, including all 15 panel machines, ran one round. So the "stable" qualifier in Table 1 (Few-1R on "15 stable machines") and the "25 consumer machines that ran stably" of Fig. 7 cannot mean what Section 2 says.
- **The fast/slow split is not robust to the probe.** The 0.5 line was drawn after the panel. Two of the five fetches-off machines (0.54) and one new machine (0.54) sit within 8% of it, and the paper reports that the ratio moved by up to 12% between rentals of one machine. I reproduced that: 109f gave 0.837, 112g 0.738 and 114f 0.783 on the same GPU.

### 3. Many registered tests could hardly fail, and some "held" verdicts are weaker than they read

The paper is honest about loose thresholds, but Table 1 still counts them as "held":

| Registered threshold | What was already known before registration |
|---|---|
| ≥ 50% of the read time | The header's own pre-launch estimate was 0.85–0.99 |
| MIN-1R's share 0.25–0.55 | The panel had shown 33% |
| Capture ≤ 35% | Typical capture is 2–6% |

Further:
- **Job 115's H8 is largely in-sample.** It pools the two new machines with job 114's three. Those three supplied the threshold's basis, and they carry 3 of the 5 values (21.8, 15.1 and 21.9%).
- **The second-card trend.** It "held" on RTX 4090s: my deviations are at most 6.6% at 11% and 5.0% at 25%. But it missed 5 of 18 cells (up to 17.6%) on the new RTX 5090s, the same card it was fitted on. Three of the five RTX 4090s sit at ratios 0.37–0.40, where MIN-1R's predicted change is about 1.0 and "no change" also falls inside the ±0.10 band. The verdict "held" is correct for the registered clause, but the out-of-sample same-card result is the more informative one.
- **The aggregate record.** Of 1,741 clauses, 19% failed and only 37% held with an interval. The paper reports this, but it should temper the impression that the claims follow from confirmatory tests.

### 4. Two problems with the bound: its validity check and its interpretation

- **The validity check rests on a failed model.** Table 1's first row tests the bound's premise (no reader beats the probe) with Eq. (4)'s accounting. Eq. (4) failed every registered test (Appendix H). Its finding that the engine's reads exceed the probe's rate on 12 of 84 launch-budgets may be a model error rather than a probe deficit. The direct, model-free check is missing from the paper: no measured configuration goes below Eq. (1), minimum 1.33× over 1,297 runs.
- **The headroom is overstated by an unknown amount.** Since even the read-ahead oracle stays ≥ 1.33× above Eq. (1), "31–54% of the bound" mixes engine inefficiency with headroom that may be unreachable by any design.
- **The low end of the range is a definitional choice.** It comes from a Threadripper 9960X with 176–178 GB/s of host bandwidth (my values 30.6–31.2%), counted as "consumer" by a definition set post hoc. Without the HEDT parts, the consumer RTX 5090 range in the raw data is about 38–54%. The abstract's headline range therefore depends on that classification.

### 5. The validity gates do not cover the configurations the inference uses

- **V2 checks only the deployed cache's round-to-round spread.**
  - The 9950X3D2 (114e) was excluded for a 6.1% spread in `base`. Its interaction would have been −0.22 ms [−0.95, 0.51] (rounds −2.09 and +1.64), a second failure.
  - The Core i9-14900K (115b) was kept although its `base0`, the reference for every CPU-only ratio, moved 6.2% between rounds (15.05 vs 15.99 ms).

  The paper reports the 14900K's failure honestly. But the gate is asymmetric, and the decision of which machine counts depends on which configuration happened to wobble.
- **Job 114's replacements deviated from the registered order.** They skipped a cheaper qualifying offer and chose a machine measured before (109f). Appendix D discloses this, but it also changed the composition of the n=5 sample.

### 6. Smaller issues

- **TGPU** (2.9 ms) is the minimum of 14 profiles on machines other than those of Table 4. My range is 2.94–3.38 ms. Using each machine's own value would shift the "attributed" rows by a few points.
- **Table 1's "post hoc" rows** (consumer range, published median, horizon rule) are given the same visual weight as registered rows, and the abstract leads with one of them.
- **Two small mismatches.** Round-to-round noise has a maximum of 6.4% by my definition |r2/r1 − 1| against 6.6% in the paper, probably a definitional difference. "Within 1.0%" for the idle probes is 1.04% on 115b.

## Clarity

Earlier rounds scored 2/5 and then 3/5. This version is better organized, and still a 3.

### What works

- **The introduction leads with answers.** Each bold paragraph states a result and points to its section, and Table 1 gives the evidence trail.
- **Table 2** is a real improvement: configurations by admitted set × number of reads, plus machine sets and yardsticks in one place.
- **One yardstick throughout.** "Speed relative to the deployed cache on the same machine" is used consistently, and the running example in Section 5 (10.0 / 13.0 / 20.6 / 15.2 / 13.8 ms) anchors the numbers.
- **Table 5 is clean:** one row per machine, both paths, intervals as subscripts and the interaction in ms with its interval. Fig. 4 is clear.
- **Post hoc labels appear consistently** in the abstract, Table 1 and Section 8, and the limitations are concise.
- **The appendices are navigable.** Appendix A maps them, and Appendix C maps the code names.

### What still makes it hard

1. **Table 1 is overloaded.** It has 13 rows at scriptsize.
   - Threshold cells merge two jobs' thresholds: "MIN-2R ≥ 1.05 (loose), 1.07× at 11%, ≥ 1.15, 1.20× at 25%; Dep-1R 0.95, 0.98 to 1.10×; interaction > 0; pooled share ≥ 10%". A reader cannot tell which number is a threshold, which job it belongs to, or which result answers which threshold.
   - The rows are not in section order (3, 3, 3, 4, 4, 4, 4, 6, 5, 7, 3, 3, 8).
   - The first row cites an Eq. (4) that the main text never defines.
2. **The vocabulary load is still very high.** The main text has about 16 configuration names (MIN-1R/2R, Dep-1R, Few-1R/2R, Few-2R-early, Belady-1R/2R, Margin, LA, LA-1R, LA-1R-margin, read-ahead oracle, admit every miss, deployed, "CPU only"). The appendix tables and scorecard switch to a second vocabulary: `base`, `foa`, `bypass0`, `fetchplan`, `both3p`, `dk`, `pf`, R1… On top of that come:
   - six machine sets (panel, new, fetches-off, O1–O6, Pa–Pj, hosts A/B/S);
   - five yardsticks;
   - four equations;
   - several count units ("launch-budgets", "machine-budgets", "cells", "clauses").
3. **The pivotal mechanism is under-explained.** That the fetch table copies unchosen misses into slots, so that "MIN-2R on the deployed path" is not MIN's set, gets two sentences in Section 2. It is then explained through Table 4's footnote ∗, Table 5's caption and three Section 4 paragraphs. Fig. 1 does not show it. A small worked example, one layer and one step with and without the table, would let a reader follow Section 4 the first time.
4. **Fig. 2 is overloaded.** It has four line styles, two marker types plotted at a shared x position (Dep-1R squares sit at "MIN's set, read twice"), and an x axis whose second state means different things on the two paths. It also mixes exploratory panel data with two registered tests in one picture.
5. **The same quantity appears with different numbers for different subsets, and the subset is not always marked:**
   - MIN-1R's share of the gap: 33% (panel), 39% (fast-link panel), 35% (new) and 36% (fetches off);
   - the share of the bound: 31–54%, 31–52%, 40–54% and 41–48%;
   - "set alone": 19% (mean), 13–23% (per-machine range, rule 2) and [13, 25] (interval).

   The rule 2 range and the interval look alike and are easily confused.
6. **Section 4 prose is dense with nested qualifiers.** An example: "The extra was 21% of the gap on the two Ryzen 9 5950Xs (ratio 0.78–0.79, one provider) and −3 to 3% on the others (0.54–0.57; post hoc)."
7. **Float and section order break the narrative.** Table 6 (in Section 6) carries Section 7's RTX 4090 test. Section 5 reports results of Section 4's fetches-off test. Appendix D is a valuable but very long narrative in which important deviations are buried mid-paragraph.

### What would make it a 4

- Split Table 1 into a registered table with one threshold per cell and one test per row, and a short post hoc list.
- Add a one-figure explanation of the fetch table's displacement.
- Use one naming system in all tables.
- Give each repeated quantity a single headline subset, with the others in parentheses labelled by subset.
- Simplify Fig. 2: separate panels for the two read paths, and Dep-1R as its own state.

## IEEE format

I reviewed `paper.pdf` above. Skimming `ieee-paper.pdf` (10 pages) and `ieee-supplement.pdf` (31 pages):

**Correct as built:**
- IEEEtran conference class.
- Anonymous author block.
- "Abstract—" in bold, about 230 words, no citations.
- Index Terms present.
- Roman-numeral section headings.
- "TABLE I" small-caps captions above tables; "Fig. n." captions below figures.
- Numbered, citation-ordered references.
- No floats on page 1. The full-width Table I/II/III floats sit at page tops.

**Departures from IEEE conventions:**
1. **The main paper is not self-contained.** Table I's first row cites "Eq. (S1)" and its note "Table S3". The text cites Tables S23–S25 and "Appendix D/N", which live only in the supplement. IEEE supplementary material is optional for reviewers, and key definitions and evidence (Eq. (4), the claim index, the prediction log) should not depend on it.
2. **Numbering collision.** The IEEE supplement numbers its tables S1, S2, … The scorecard (`paper/supplement.pdf`), which the IEEE supplement tells readers to consult, also numbers its tables S1, S2, … (it is "numbered as the MLSys version"). "Table S1" therefore names two different tables.
3. **Page count.** Ten pages including about one page of references is at or above the limit of most IEEE conference formats (often 8 + refs or 10 including refs). Check it against the target venue.
4. **Font size and typography.**
   - Table I is set at about 7 pt (scriptsize), and Tables III–VI are similarly dense. Many IEEE venues require ≥ 8 pt in tables and figures.
   - Configuration names are set in a sans-serif font inside Times text and tables (the `\cfg` macro), which is unusual in IEEE typography.
5. **Two separate reference lists** (main [1]–[39], supplement [S1]…). The same work gets two numbers, for example Kalibera & Jones is [21] in the paper and [S2] in the supplement. Acceptable, but state it.
6. **Build hygiene.** `ieee-paper.log` has 34 "Label … multiply defined" warnings from `xr` importing the supplement's citation-key labels. The cross-document links may resolve ambiguously, and a camera-ready build should be clean.
7. **Reference details.**
   - The llama.cpp entry gives a bare URL without "[Online]. Available:".
   - Some entries mix venue year and arXiv year ("2026, arXiv:2506.20675 (2025)").
   - Several GitHub pull requests and branches are cited by user handle in the supplement.
8. **Run-in headings.** The Introduction's bold run-in paragraph heads ("The headroom is large (Section III).") are not IEEE's run-in style (italic, with a colon). The rest of the body converts correctly.

## Questions for the authors

1. Can you report the model-free check of the bound, the minimum of T/Eq.(1) over all configurations and machines, in place of, or beside, Table 1's Eq. (4)-based row? If Eq. (4) failed its registered tests, why is it the arbiter of the probe's validity?
2. Are the Ryzen 9 5950X machines (offers 54573659/23/24/25) and the two RTX 4090 Core Ultra 9 285Ks separate physical hosts, or GPUs of shared hosts? How do the fetches-off summaries change if each build counts once (n = 4, or n = 3 builds)?
3. What does "ran stably" mean for the single-round launches of jobs 093–105 that enter the 15 Few-1R machines, the 25 consumer machines and the 15 panel machines?
4. Why does the per-machine fetch table slow the deployed cache on Intel hybrid desktops (+6–8% speed with the table off on 265K/285K)? Is the "gap" on those machines still a fair target for the decomposition?
5. Why not extend V2 to the configurations a job's inference uses (`base0` in jobs 113–115), registered in advance?
6. Would you register a fetches-off decomposition on Qwen3, or on a second engine path, to test whether "the set pays alone" generalizes beyond gpt-oss on this engine?
7. The second-card trend misses 5 of 18 same-card cells. What machine features beyond the ratio explain the RTX 5090 misses (for example, 109d/e at +0.14 and +0.16 in log)?

## What would raise my score

- **Generality (biggest effect on significance):** the fetches-off decomposition on a second model (Qwen3 12.5% and 25%) and, ideally, on a second engine or a corrected fetch table, with predictions registered before launch.
- **Statistics:**
  - report per-build (or per-provider) aggregation alongside per-machine aggregation;
  - replace t-intervals for n ≤ 5 bimodal sets with the per-machine values plus a range or a cluster bootstrap;
  - define "stable" for single-round launches, or restrict the "stable" claims to multi-round machines.
- **Registered tests with teeth:**
  - one fresh fetches-off replication whose pooled test uses only new machines;
  - at least two machines below a ratio of 0.5 and two above 0.8;
  - thresholds placed where the earlier data could plausibly fail them.
- **The bound:** report the direct check (no run beats Eq. (1), minimum 1.33×), and show the headline share also against Eq. (3) or a measured achievable floor. Give the consumer range with and without HEDT parts.
- **Clarity:** the changes listed under "What would make it a 4" above.

## Scores

| | Score |
|---|---|
| Overall | 6 / 10 (acceptable with revisions) |
| Soundness | 4 / 5 |
| Methodology | 4 / 5 |
| Significance | 3 / 5 |
| Clarity | 3 / 5 |
| Confidence | 4 / 5 |

## Claims checked against raw data

All checks were done with my own code from raw rows, counters, probes, traces or git and ledger metadata. ✓ = reproduced to the printed precision; ≈ = reproduced with a minor, explained difference; † = new check not in the paper.

| # | Claim (where) | Paper | My result (source) | Verdict |
|---|---|---|---|---|
| 1 | R\* of MIN with bypass, gpt-oss (Eq. 1; L limits) | 38.3 (C=14), 15.3 (C=32); first 20 problems −0.4% | 38.315, 15.340; first 20: 38.170 (−0.38%) (own MIN implementation on `084c/route_aime25_gptoss.npz`) | ✓ |
| 2 | Eq. (1) per machine (Table 5) | 14.7, 8.1, 14.8, 5.9, 8.0 ms | 14.72, 8.06, 14.76, 5.86, 8.01 (R\*·S / max probe reading in `concur.txt`) | ✓ |
| 3 | Table 5 ratios, 5 machines × 7 columns | e.g. 5950X‡: 1.00, 1.12, 1.12, 1.06, 1.07, 1.39 | All 35 cells match, e.g. 1.000, 1.121, 1.124, 1.065, 1.068, 1.390 (`ec_*.jsonl` of 114b/c/f, 115b/c; geometric mean over rounds) | ✓ |
| 4 | Table 5 interaction (ms, 95% CI) | 3.2 [3.0, 3.3]; 0.4 [0.3, 0.5]; 3.1 [2.9, 3.3]; −0.3 [−0.7, 0.1]; 0.3 [0.2, 0.3] | 3.17 [3.01, 3.35]; 0.36 [0.26, 0.47]; 3.10 [2.93, 3.27]; −0.31 [−0.73, 0.11]; 0.26 [0.21, 0.31] (paired bootstrap over problems) | ✓ |
| 5 | Core i9-14900K rounds; base0 moved (Table 11; D) | +0.6 / −1.2 ms; 6% | +0.57 / −1.20 ms; base0 15.05 → 15.99 ms (6.2%) | ✓ |
| 6 | Table 4 fetches-off column (12 cells) | 11%: 19 [13, 25], 8 [2, 14], 36 [13, 58], 18 [10, 25], 43 [31, 56], 3 [−5, 12]; 25%: 33, 3, 36, 23, 42, −1 | 19.0 [13.5, 24.6], 7.7 [1.8, 13.7], 35.6 [12.7, 58.5], 17.5 [10.4, 24.7], 43.5 [30.7, 56.3], 3.4 [−5.1, 11.9]; 25%: 33.1, 2.9, 35.9, 23.0, 42.2, −1.1 (t over 5 machines) | ✓ |
| 7 | Interaction share by machine kind (Section 4) | 21% on two 5950Xs; −3 to 3% on the others | 21.1, 20.8; −3.3, 2.3, 3.2 | ✓ |
| 8 | MIN-2R misses with / without the table (Section 4) | 48.1–50.8 vs 43.5 | 48.06–50.75 vs 43.50 (`st_*.json`) | ✓ |
| 9 | Dep-1R in-step copies; own admissions | 20.8–32.0; 8.2 | 20.80–31.96; 8.19 | ✓ |
| 10 | Admissions per token, MIN's set vs deployed | 20.4 vs 8.2 | 20.39 vs 8.20 | ✓ |
| 11 | Table 4 panel column (15 machines) | 0 [−2, 3]; 0 [−2, 2]; 33 [21, 44]; 15 [11, 19]; 52 [43, 62]; fast-link 39% | 0.5 [−1.8, 2.8]; −0.0 [−1.9, 1.8]; 32.6 [21.0, 44.3]; 14.9 [10.7, 19.1]; 52.5 [42.6, 62.3]; 38.8% (13 fast) | ✓ |
| 12 | Panel interaction > 0 (Table 1) | 9 of 10 main-job machines | 9 of 10 (099f negative); all panel machines single-round | ✓ |
| 13 | Same CPU model, deployed time differs (Section 2) | up to 29% | 099a vs 099f (Core Ultra 9 285K): 18.14 vs 14.06 ms = 1.29× | ✓ |
| 14 | Excluded 9950X3D2 (D, job 114) | base rounds 6.1% apart; interaction −2.1 / +1.6 ms | 17.77 vs 16.75 ms (6.1%); −2.09 / +1.64 ms; pooled −0.22 [−0.95, 0.51]† | ✓ |
| 15 | Table 6, RTX 5090 rows (job 109) | e.g. 285K: 1.22 / 1.46 / 1.02 / 4%; best 1.007–1.024×; capture ≤ 6% | All cells match; best 1.007–1.024; capture max 5.7% (7945HX) | ✓ |
| 16 | Table 6, RTX 4090 rows and trend brackets (jobs 111, 112) | e.g. 14900KF 0.96 [0.99], 1.13 [1.11], 1.03, 26%; within 7% (11%) and 5% (25%) | All cells match; frozen job-110 trend gives [0.99], [1.11], …; max deviation 6.6% (11%), 5.0% (25%) | ✓ |
| 17 | Trend applied post hoc to new RTX 5090s (Section 7) | misses 5 of 18 cells, up to 18% | 5 of 18 outside ±0.10 in log; max 17.6% | ✓ |
| 18 | Table 12 timing control (job 112) | 1R/2R/early speeds; misses 40.7/55.6/54.5…; early misses 0.97–0.98×; admissions 11.7–11.8 vs 6.2–7.4; reads +4–7% | All 24 cells match; ratios 0.968–0.980; 11.72–11.75 vs 6.20–7.40; +4.2–7.1% | ✓ |
| 19 | Table 13 and job 113 text | 0.98 / 1.12 / 1.26 / 1.23 / 1.40; 1.07 / 1.09 / 1.23 / 1.32 / 1.34; misses 0.77–0.79 (11%), 0.72–0.74 (25%) | 0.981 / 1.122 / 1.259 / 1.235 / 1.398; 1.073 / 1.094 / 1.231 / 1.321 / 1.344; 0.773–0.786; 0.718–0.743 | ✓ |
| 20 | Idle probe reruns, job 115 (Section 2; D) | within 1.0%; best idle / first 0.998–1.010; spread within 0.5% | 1.0104 and 0.9984; spread 0.46%, 0.16% (`concur_idle*.txt`) | ≈ (1.04% printed as "within 1.0%") |
| 21 | TGPU and G from Nsight (Eq. 2; App. H) | 2.9 ms = smallest of 14 profiles; G 4.1–4.7 ms | TGPU 2.94 (105e C14), range 2.94–3.38 over 14 profiles; G 4.07–4.69 (`prof_C*.json`, `g_prof.json`) | ✓ |
| 22 | Read-time microbenchmark (Table 31) | 86–96% per layer; repetitions within 0.9% | 86, 89, 96, 96, 95, 93%; repetitions within 0.88% (`readsched_C*.txt`) | ✓ |
| 23 | Table 3, host S (job 089) | 28.9 / 48.0 / 57.9, 1.21 [1.19, 1.22] … 0.97 [0.96, 0.99]; bound 140 tok/s; leads at 11 of 12; 5 of 6 outside ±0.06 | All 6 rows match (launch 2), e.g. 1.207 [1.194, 1.219]; Bhost 71.3 → 140.4 tok/s; leads at 11 of 12; 5 of 6 | ✓ |
| 24 | Running example on O4 (Section 5) | Eq. 1 = 10.0; Eq. 2 = 13.0; deployed 20.6; MIN-1R 15.2; read-ahead 13.8 ms | 10.02; 12.96; 20.64; 15.16; 13.81 | ✓ |
| 25 | Round-to-round noise (D) | median 0.2% over 143 comparisons, 19 machines; max 6.6% | 143 comparisons, 19 machines, median 0.21%; max 6.4% | ≈ (max definition) |
| 26 | Share of Eq. (1), consumer machines at 11%; new machines (Section 3) | 31–54%; 41–48% | Consumer RTX 5090 launches 30.6/31.2% (TR 9960X) to 54.0% (100f); new 41.5–47.9% | ✓ (low end is HEDT) |
| 27 | Few-1R geometric mean over 15 stable machines (Section 5; App. I) | 1.26× (1.19–1.33) | 1.257× (1.186–1.332); 12 values from raw rows, 3 (Pe, 5900XT, AMD ES) from Tables 26–27 | ≈ |
| 28 | Registration timing (D) | every commit before its rental | 107 rentals of jobs 093–115: header commits lead the ledger start by ≥ 4 s, wrappers by ≥ 3 s (commit time); post-start header edits in 100, 102, 105, 106, 107, 111, 115, all disclosed; job 115's amendment text 15 s after host d started | ✓ |
| 29 | Ledger totals (N) | 160 rentals, 96 offers, $115.3 | 160, 96, $115.3 | ✓ |
| 30 | Gates: job 115 (D) | two TR 3970X (0.39) and EPYC 7352 (0.26) stopped at the ratio gate; one at the first gate | 115a 0.3932, 115d 0.3931 (TR 3970X), 115f 0.2642 (EPYC 7352); 115e at V0 | ✓ |
| 31 | Gates: job 114 (D) | three stopped at V0; one is job 089's machine | 114a, 114d, 114g at V0; 114d's GPU e9fe1cab = 089's (host S) | ✓ |
| 32 | Machine identity (D) | 114f is 109f's machine; two 5950Xs from one provider | 114f = 109f = 112g (GPU f90ec646); 113b, 114b, 109f and 108f are one build (nearby offer IDs, bus 06:00.0, 125 GB) | ✓ (clustering larger than stated) |
| 33 | Table 10 tallies (D) | 639 / 694 / 324 / 84; hand 218 / 193 / 142 (+12) = 565 | Row sums give the same | ✓ |
| 34† | Does any run beat Eq. (1)? (not in paper) | (Eq. 4 implies > 5% above the probe on 12 of 84) | 0 of 1,297 gpt-oss configuration-runs (jobs 093–115) below Eq. (1); minimum T / Eq. (1) = 1.33 | New check, supports the bound |
| 35† | Deployed fetch table counterproductive on Intel hybrid (Table 4 † row) | 3 [−5, 12] (mean) | base0/base = 1.056 (265K), 1.076 (285K), 1.073 (113d 285K); share of gap +9.1% and +12.0% | New observation |
