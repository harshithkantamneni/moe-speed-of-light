# Review 26b: "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth"

Reviewer role: systems-group professor, refereeing for a fellowship or workshop in the spirit of Hoefler and Belli (SC'15). Date: 8 October 2026.

## Materials read

- `paper/paper.pdf`: all 44 pages, read as extracted text. That covers the main text (Sections 1–12), References, and Appendices A–N, including Tables 1–32 and Figs. 1–9.
- `paper/supplement.pdf`, the scorecard supplement: the front matter, the scoring rule, the job 073–075 rows, and Table S19 (job 113) in full.
- `paper/ieee-paper.pdf` (10 pp.) and `paper/ieee-supplement.pdf` (29 pp.): skimmed as text. I also rendered IEEE p. 1 and supplement p. 12 at low resolution to check layout.
- Raw data on the gpu branch (`/home/claude/gpu-branch`):
  - `results/<job>/ec_*.jsonl`, `st_*.json`, `concur.txt`, `fetch_table_law_gptoss.json`, `g_prof.json` / `prof_G*.json`, `validity.txt` / `v0.txt` / `gate.txt`, `bs1.jsonl` (Table 3) and `readsched_C*.txt` (job 102).
  - Jobs read: 081, 089, 093, 095, 096a, 099a–j, 100b, 100f, 101b, 102a–c, 104a–c, 105a/b/e/f, 106a–e, 107a–g, 108a–f, 109a/c/d/e/f, 110a–e, 111d/f/g, 112a/b/d/g and 113b/d.
  - The headers of `jobs/099`, `109`, `110` and `113` in full.
  - The diffs of every header commit made after a job's first rental (jobs 100, 102, 105, 106, 107 and 111).
  - The trace `results/084c_gptoss_trace@vast/route_aime25_gptoss.npz`.
- Artifact repository:
  - `gpu/vast_ledger.json` and `gpu/runner.sh`, the start of each.
  - `prereg/speed_limit_v2.json` and `prereg/decomp_measured.json`, both JSON data, not outcome notes.
  - `mosl/cachesim.py`, the first 120 lines, read only to explain a 0.8% difference between my own MIN implementation and the paper's R⋆.
- Git metadata: `git log --format='%h %at %ct'` on the job scripts, and `git show --format=''` for header diffs. I read no commit messages.
- My scripts are in `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/rev26b/`: `lib.py`, `t113.py`, `t109.py`, `t099.py`, `panel15.py`, `t4090.py`, `t3.py`, `share.py`, `regtime.py`, and `minr.py` / `minr3.py`.

## Independence statement

- I did not open anything under `reports/` except to create this file.
- I did not open any `prereg/*outcome*.md`, `research_notes/`, `apply/`, `paper/paper_v1_prereview.tex`, or any file named like a review, number check, plan or progress log.
- I did not read commit messages; I used only hashes, timestamps and diffs.
- I modified nothing in either repository apart from writing this file, and I committed nothing.
- I wrote every number below that I attribute to "my computation" with my own code, from raw per-problem rows, counters, probe logs or traces. There is one exception: I ran the repository's `cachesim` once to explain a semantic difference, after first writing and running my own implementation.

---

## Summary

The paper studies batch-1 decode of MoE models whose experts mostly live in host DRAM: gpt-oss-120b and Qwen3-30B-A3B on rented RTX 5090 hosts, plus RTX 4090s for one test.

**The bound (Eq. 1).** It defines a per-machine yardstick: R⋆ × S / B_host. R⋆ is the host reads per token of Belady's MIN with bypass on the recorded routing trace, and B_host is the highest reading of a bandwidth probe. It adds two tighter forms: a "demand" bound (Eq. 2), and an "ordered" bound for CPU reads that cannot be issued early (Eq. 3).

**The engine and the decomposition.** The authors' llama.cpp expert cache beats FreeToken at 11 of 12 configurations and stock llama.cpp at all 12. It reaches 31–54% of Eq. 1 on 25 consumer machines at gpt-oss 11%. Oracles inside the engine split the remaining gap:
- what is cached: MIN's set instead of the deployed set;
- how it is read: one read instead of CPU-serve-then-copy;
- when it is read: issued ahead of use.

**Main findings.**
- On the deployed read path, MIN's set pays only together with reading once.
- A late control (job 113, two machines) traces this to the engine's own "in-step fetches", which put misses the policy did not choose into slots. With those fetches off, MIN's set pays even when read twice: Few-2R runs 1.23–1.26× and MIN-2R 1.09–1.12× a fetch-free deployed cache.
- Online policies without foresight recover at most 6% of the read-ahead oracle's gain.
- A link-to-CPU trend fitted on RTX 5090s predicts the oracles on RTX 4090s within 7% (registered test).
- A trace study suggests that half of foresight's value needs a window holding about 0.65 C distinct experts per layer.

**Registration.** Every experiment from job 073 on registered predictions in its job-script header before launch. The paper scores 1,687 clauses, including 323 failures.

---

## Strengths

1. **Transparency well above the norm.** Every job script carries its predictions, validity gates and host list, committed before the rental.
   - The ledger records every rental (147 rentals, 86 offers, $107.48; I recomputed all three).
   - Raw per-problem rows, engine counters, probe logs and Nsight summaries are released.
   - Hosts that failed gates are reported; I checked jobs 107, 108 and 110 host by host.
   - Voided jobs, deviations and the 323 failed clauses are tabulated. The paper also flags its own loose thresholds ("could hardly have failed"). This is the most complete prediction log I have seen in a systems benchmarking paper.

2. **The numbers reproduce.** I re-derived 33 groups of quantitative claims from raw data (table below), covering several hundred individual numbers.
   - Every one matched to the printed rounding. This includes all of Table 5, Table 6, Table 12 and Table 30, both columns of Table 4, the Table 3 ratios with their bootstrap intervals, and the running example.
   - R⋆ itself (38.315 and 15.340 reads per token) came out exactly from my own step-atomic implementation of MIN with bypass.

3. **A useful yardstick and a sensible decomposition method.** The paper asks "how far is this machine from what any cache of size C could do?" rather than "how much faster than baseline X?". That is the right question, and Eqs. 1–3 are simple enough for others to adopt.
   - Putting oracles inside a real engine, rather than only in simulation, and removing one cause at a time is good experimental design.
   - The microbenchmark in Table 30 shows that the read part of the bound is approachable: 86–96% of it on three machines.

4. **Honest self-correction.** The paper found that its own 2×2 was confounded by the engine's in-step fetches, registered a control (job 113), and reframed the headline accordingly. It also kept an earlier control that failed its manipulation check (job 112) and called it uninformative.

5. **Hoefler–Belli hygiene.**
   - The machine is treated as the unit, and the paper quantifies that machines vary more than problems (96% of log-gain variance).
   - Variation between rounds and between rentals is quantified. I reproduced the round-to-round figures: 108 comparisons, median 0.17%, max 1.77%.
   - Paired bootstrap within machines, a checklist appendix, and an output-parity check (KL) are all present.
   - One known departure, the arithmetic mean of rates in Table 3, is acknowledged.

6. **A practical engineering contribution.** The cache runs about 2–4× stock llama.cpp, and its lead over FreeToken holds on long outputs and on FreeToken's own headline model (where the two tie).

---

## Weaknesses (most important first)

### W1. The paper's central decomposition is still confounded, and the repair covers 2 machines and half the design

The title question is "where the seconds go", answered by Table 4 and Fig. 2. Its first row, "MIN's set, deployed read path (MIN-2R)", does not measure what its name says.

- The footnote now concedes that the in-step fetches displace MIN's set.
- My counters confirm it. On job 113's machines, Few-2R with fetches misses 54.7–55.6 per token and fetches 18.6–24.3 misses into slots the schedule did not choose. Few-1R misses 40.7.
- So the 0% "what is cached" row and the dominant "interaction" (Both = 33%) are not attributable to "set" versus "reads". They are largely a property of this engine's fetch table.

Job 113 shows the set does pay once the fetches are off. But:
- it ran only the two-read arms plus a fetch-free deployed cache;
- it ran on two machines;
- it has no fetch-free Dep-1R/MIN-1R arm;
- it ran at only one round at 25%.

There is therefore still no unconfounded 2×2 on a panel. The Section 4 heading, the first builder rule, and the Table 1 row "On the deployed read path, MIN's set pays only when read once" describe an engine artifact, not a property of caching.

The confound also cuts the other way:
- On the 5950X, one read still gains +0.16 over fetch-free Few-2R. On the 285K it gains only +0.02, and at 25% reading twice was faster there.
- So the paper cannot currently say how much of "one read" is worth anything once the set is held. Two machines with opposite answers is not a finding.

This is fixable with one more registered panel job. Run all four 2×2 arms with fetch=0, or with a fetch table restricted to misses the policy admits, on five or more machines. Until then, Section 4 should be presented as "this engine's read path interacts with the set", which is mostly what the revised abstract says, and Table 4's first two rows should be renamed.

### W2. The "bound" is relative to a single probe run, and the paper's own data violate it on about 14% of launch-budgets

B_host is the highest reading of one probe run, which was often taken while the model downloaded. My check of `probe1_when.txt` on job 113 shows some runs were idle, but the paper says the probe was "mostly taken while the model downloaded".

- By the paper's own accounting, the engine reads more than 5% faster than B_host on 12 of 84 launch-budgets, up to 1.22× (Table 28).
- The top of the headline range, 54%, is host 100f, where the engine implies a rate above the probe. The paper says the top is "probably too high", yet keeps 54% in the abstract and conclusion.
- Between two rentals of one machine, the link-to-CPU ratio moved by 12% (I confirmed 0.84 → 0.74 for GPU-f90ec646, jobs 109f and 112g). That same ratio is the x-axis of the paper's main predictive claim and the 0.5 class boundary.

Eq. 1 is therefore a probe-relative yardstick, not a physical bound. Calling it a "bound", and calling the repository "speed of light", overclaims. A best-of-N probe on an idle machine, or an envelope that takes the maximum of the probe and the engine-implied rate, would make the 31–54% claim robust. Only 4 machines had a second probe.

### W3. Registered and exploratory results are not separated where readers look first

Table 1 labels evidence carefully, and the Section 7, 8 and Appendix H texts say "post hoc" or "exploratory". The abstract does neither.
- It reports 31–54% of the bound, "about half" closed by oracles, and ≈0.65 C. The first and third are post hoc in Table 1.
- It then closes with "Every prediction was committed before its machines were rented."
- A reader will take the abstract's numbers as confirmatory. Several are not, and the consumer/server split behind "31–54%" was drawn after the data.
- The same applies to the fast-link threshold of 0.5. It was drawn after the panel, and the failed "every machine" interaction clause was then retested only on machines above it. Table 1 does say "fast-link scope set in between", which is honest but easy to miss.

The abstract sentence is also stronger than what can be verified externally:
- Commit timestamps are self-reported.
- Appendix D concedes that for four rentals the GitHub push came up to 0.8 s *after* the rental was created. This is benign, because machines clone at boot about 26 s or more later.
- Result directories do not record the gpu-branch SHA each machine actually executed. `manifest.json` has the patch SHA but not the branch SHA, which would close this gap cheaply.

### W4. The evidence base is narrow relative to the generality of the "rules for system builders"

- One engine.
- One model, gpt-oss, at two budgets for the decomposition and rules 2–4.
- One prompt family (AIME-25) that was also used during development, with 20 problems × 256 teacher-forced tokens in the later jobs.
- A convenience sample of cheap Vast.ai offers, mostly fast links.
- Registered tests with 2–5 machines. Table 4's "New" column is a t-interval over 5 machines; Both = 35 [22, 48].

The predictive claim is the clearest example:
- The link-to-CPU trend, frozen on the panel, held on the RTX 4090s.
- Applied to the 5 new RTX 5090s of job 109, it misses 5 of 18 cells by more than its band, up to 17.6%. I reproduced this exactly.
- The misses are systematic by host. It under-predicts both Intel machines by 8–16% at the read-ahead oracle and over-predicts the 9800X3D by 11–12%. So the ratio is not a sufficient statistic.
- The ±0.10 log band is also wide. For MIN-1R on the three slow-link RTX 4090s, a "no change" model passes it too (ln 0.963, 1.007, 1.005). The test's power comes from the read-ahead oracle.
- "The ratio predicts the oracles' speed on a second card" should be softened to "predicts the direction, and the size within about 20%".

### W5. Statistical reporting has gaps that matter for the scorecard's meaning

- **No interval on the key clause.** The interaction, the key Section 4 clause, is scored "held (point)" because no interval exists. It is a difference of four paired means over 20 problems, so a bootstrap interval is easy.
- **Many point-only holds.** 685 of 1,687 clauses (41%) held only on the point estimate. 213 hand-scored clauses had no interval at all.
- **Operational conditions counted as predictions.** Validity conditions such as "at least two valid hosts" (job 113) and "at least three valid hosts" (job 112) are counted among held clauses. They are operational conditions, not predictions.
- **Interval scope.** Within-machine intervals exclude round and rental variance. The post-hoc widening analysis in Appendix D is reassuring, but it is post hoc, and the 4 pooled clauses are not widened.
- **Pooled summaries mix conditions.** Several summaries pool across three engine versions, 20 versus 30 problems, and repeated rentals:
  - Few-1R's 1.26× over 15 machines: I reproduced it after deduplicating GPUs. Pf ran three times; the TR 9960X twice.
  - Table 4's 15-machine panel.
  - The paper says this, but the reader cannot tell from Table 1 which numbers are pooled across protocols.
- **Table 3** averages per-problem rates arithmetically; this is acknowledged, and the ratios are unaffected within 0.003.

### W6. Smaller soundness points

- **Table 1, Few-1R row.** It says the set "lost on the two unsteady ones". At least four machines that ran Few-1R were unsteady or invalid: EPYC 7302 0.85, EPYC 7663 0.97, Xeon 8347C 1.30, EPYC 9754 1.16 (all from my computation). Section 5's wording ("counting the machines that ran unsteadily as well, it lost on two") is correct; Table 1's is not.
- **Job 110e.** Its trend deviation (−5.6%) is cited to justify not relaunching it. Its capture in that same void round was 78%, of a small 5.6% oracle gain, which the paper does not mention. Capture is unstable when the oracle's gain is small; on the slow-link RTX 4090s it reaches 26%. Report the denominator alongside capture.
- **Job 113's offers file.** `results/113_offers_at_launch.txt`, which justifies skipping offers a and c, was "transcribed … committed after the job; the query's output was not saved at the time". This is minor, but it is post-hoc provenance for a registered host-selection step.
- **Two R⋆ semantics.** R⋆ depends on letting an expert served in a step be evicted later in the same step (step-atomic semantics). My independent sequential-within-step implementation gives 38.62 instead of 38.315 (+0.8%); a step-atomic version reproduces 38.315 exactly. Table 29 discloses the no-evict variant, so this is fine, but the main text should define which semantics R⋆ uses.
- **Abstract vs. Section 2.** The abstract's "every cached expert is read twice" is not true of the deployed cache's normal path, which fetches some misses in-step and reads them once. The Section 2 definitions are clear; the abstract is not.

---

## Clarity

Earlier rounds scored clarity 2, then 3 several times. This revision is better organised: each section opens with its answer, and Fig. 1 (the schematic), Table 2 (names), Table 1 (the claim ledger), the running example in Section 5, and the scoped rules in Section 9 all help. It is still hard to read. The difficulty is conceptual density and naming, not sentence length: my rough count is about 18 words per sentence in the main text.

### What still makes it hard

1. **Section 4 asserts a finding and then withdraws its interpretation.**
   - The paragraph heading says "On the deployed read path, MIN's set pays only when read once".
   - Two paragraphs later: "So the interaction above belongs to this engine's read path, not to MIN's set."
   - Table 4's first row is labelled "MIN's set, deployed read path∗ (MIN-2R)", and its footnote says the row "does not measure MIN's set held".
   - A table row whose label its own footnote contradicts is the single biggest clarity problem. Rename the arm everywhere, for example "MIN-2R (with in-step fetches)" or "MIN's schedule, fetch table on", and rewrite Section 4 so the confound comes first and the fetch-free result second.

2. **The abstract is overloaded.**
   - It has about ten numeric claims in 200 words.
   - "1.09 to 1.26 times" merges two schedules: greedy 1.09–1.12, fewest-admission 1.23–1.26.
   - "Even when every cached expert is read twice" reads as a concession when it is the definition of the arm.
   - "The deployed cache without them" requires the reader to know that the comparator changed.
   - Cut it to the bound, the decomposition, the confound, and the no-foresight result, and mark the post-hoc numbers.

3. **Naming load.** The main text uses:
   - MIN-1R/2R, Few-1R/2R, Dep-1R, LA, LA-1R, LA-1R-margin, Margin, Belady-1R/2R, Few-2R-early, "read-ahead oracle", "admit every miss" and "CPU only";
   - greedy versus fewest-admission schedules;
   - the panel, the new machines, O3–O5, hosts A/B/S and Pa–Pj;
   - job numbers 093–113 (9 job references in the main text).

   Some terms are also inconsistent:
   - "Panel" means 15 machines in Table 4 and Table 2, but the 10 machines of job 099 in Table 1's "10 + 5" and in Appendix C.
   - "Fast link" means a link-to-CPU ratio of at least 0.5, not a fast link.
   - Table 7 (codes to names) is necessary for reading the appendices, which shows how heavy the scheme is.

4. **Table 1 is a dense ledger rather than a readable summary.**
   - Its cells hold compound results ("held on B; on S, five of six ratios fell outside ±0.06 of B's; faster at 11 of 12 in all").
   - The caption doubles as a glossary.
   - Some labels mislead. "Trend fitted post hoc" sits on what was a prospective test. "Held (point)" sits on 86–96% against a 50% threshold, because no interval exists by rule.
   - The "Mach." column mixes counts, host letters and "10 + 5".
   - Splitting it into "registered tests" and "exploratory findings", with one short result per row, would help.

5. **Table 5's header is hard to parse.** It has stacked "vs Dep. / vs Dep. CPU only / vs Dep." spans, "CPU only" is defined only in the notes, and half-widths are printed as "±.01". Give each ratio as an explicit column title, for example "Few-2R(fetch off) ÷ Dep(fetch off)".

6. **Appendix D is a wall of narrative.** Job-by-job prose about amendments, relaunches, gates and offers makes the registration record auditable in principle but tiring in practice. A per-job table would serve better, with columns: committed, first rental, amendments (what changed, before or after which host), gates failed, and outcome.

7. **Smaller points.**
   - Fig. 2's legend ("MIN-2R first (MIN's set, deployed read path)") inherits the naming problem.
   - Numbers appear in the text without a table reference, for example "1.57–1.76 R⋆" in Section 6.
   - Table 2 combines configurations, machine sets and yardsticks, with four footnotes.
   - "Capture" and "value of foresight" use different baselines: the deployed cache versus admit-every-miss. Fig. 5's caption flags this, but the text does not.

### What works

- The question-shaped section titles with one-sentence answers.
- The single running example (Ryzen 9 9950X at 11%: 10.0 / 13.0 / 20.6 / 15.2 / 13.8 ms), which I verified exactly.
- Fig. 1.
- The explicit scope paragraph before the rules in Section 9.
- The limitations up front.
- The consistent use of "speed relative to the deployed cache on the same machine".

My clarity score stays at 3/5. It is a better 3 than before, but the Section 4 / Table 4 framing problem alone keeps it from 4.

---

## Questions to the authors

1. Can you run the full 2×2 with in-step fetches off (base0, foa0, bypass0, fetch) on five or more registered machines, so that Table 4's "what is cached" row measures MIN's set held? Alternatively, restrict the fetch table to misses the policy admits.
2. How do the bound shares change with a best-of-N probe on an idle machine? Can you report Eq. 1 against max(probe, engine-implied rate), so that it never sits below an observed rate?
3. Can you give per-machine bootstrap intervals for the 2×2 interaction (in ms) and rescore that clause?
4. What explains the systematic residuals of the link-ratio trend on the new RTX 5090s (Intel under-predicted 8–16%, 9800X3D over-predicted about 12%)? Is there a second probe feature, such as B_both / B_c or the concurrent sum, that removes them?
5. Would the decomposition change on non-mathematical prompts or longer contexts, where routing locality, and therefore R⋆ and the horizon, differ?
6. In the 15-machine Few-1R summary, how were repeated launches of one machine combined? I took the geometric mean per GPU and obtained 1.257.
7. Why do result directories not record the gpu-branch commit each machine executed? Can the runner write `git rev-parse HEAD` into `manifest.json` from now on?
8. Do validity conditions ("≥2 valid hosts") count as predictions in the 595 held clauses? If so, how many such clauses are there?

---

## What would raise my score

- An unconfounded, registered decomposition on a panel of at least 5 machines (W1). This alone would move soundness to 4 and the overall score to 7.
- A probe protocol that makes Eq. 1 an envelope (W2), and an abstract that drops or qualifies the 54% top end.
- An abstract and Table 1 that separate registered from exploratory claims, plus executed SHAs in the results (W3).
- The Section 4 and Table 4 renaming, and a simplified naming scheme with the "panel" definition fixed (Clarity 1 and 3).
- Interval-based scoring of the interaction, and removing operational validity conditions from the "held" tally (W5).

---

## Scores

| Criterion | Score |
|---|---|
| Overall (1–10) | **6** (acceptable with revisions) |
| Soundness (1–5) | **3** |
| Methodology (1–5) | **4** |
| Significance (1–5) | **3** |
| Clarity (1–5) | **3** |
| Confidence (1–5) | **4** |

**Methodology (4).** The registration, gating, failure reporting and raw-data release are exemplary. The score is not 5 because of W2, W3 and W5.

**Soundness (3).** Every number I checked reproduces. But the central decomposition is confounded and the bound is relative to the probe.

---

## Claims checked against raw data

Legend: ✓ = matches to the printed precision; ≈ = consistent, with the minor difference explained; ! = the claim reproduces but has a framing issue.

| # | Claim (location) | My computation (source) | Result |
|---|---|---|---|
| 1 | R⋆ = 38.315 (C=14) and 15.340 (C=32) reads/token (job 109 header; Eq. 1 throughout) | My own MIN with bypass on `084c…/route_aime25_gptoss.npz`. Step-atomic: 38.315 / 15.340 exactly. Sequential within a step: 38.62 / 15.39 (+0.8%) | ✓ (≈ semantics; see W6) |
| 2 | Table 5, 5950X: Dep CPU-only 0.98; MIN-2R CPU-only 1.12; Few-2R CPU-only 1.26 (vs Dep CPU-only) and 1.23 (vs Dep); Few-1R 1.40; misses 55.6/43.0 | `113b` jsonl and st: 0.981, 1.122, 1.259, 1.235, 1.397; 55.64/42.99. Intervals over problems about ±0.01 | ✓ |
| 3 | Table 5, 285K: 1.07, 1.09, 1.23, 1.32, 1.34; misses 54.7/43.0 | `113d`: 1.073, 1.094, 1.231, 1.321, 1.344; 54.71/42.99 | ✓ |
| 4 | Fetch-off manipulation: Few-2R misses 0.77–0.79× at 11% and 0.72–0.74× at 25% (registered ≤0.88) | 0.773, 0.786; 0.718, 0.742 | ✓ |
| 5 | Few-2R fetched 18.6–24.3 misses per token in-step; Few-1R missed 40.7 (Section 4) | Fetches 24.35 / 18.65; Few-1R 40.73 | ✓ |
| 6 | At 25%: fetch-free Few-2R 1.33–1.36× deployed, against Few-1R 1.29–1.43×; reading twice faster on the 285K | 1.325 / 1.359 against 1.427 / 1.293 | ✓ |
| 7 | H6: one read +0.02–0.16 over fetch-free Few-2R; failed on the 285K | 0.162 and 0.023 (scorecard: 0.024) | ✓ |
| 8 | Table 6, job 109 (5 machines × MIN-1R, Ahead, None, Capture at 11% and 25%) | e.g. 285K 1.215 / 1.460 / 1.021 / 4.5%; 5950X 1.366 / 1.500 / 1.010 / 1.9%; 7945HX at 25% 1.283 / 1.626 / 0.995 / −0.8%. All 36 cells | ✓ |
| 9 | Best no-foresight variant 1.007–1.024× at 11%; capture ≤6% (Section 6) | 1.007–1.024; capture 1.5–5.7% | ✓ |
| 10 | Link-to-CPU ratios 0.54 / 0.59 / 0.84 / 0.88 / 0.89 (Table 6) and 0.74 / 0.56 (Table 5) | B_p/B_c from `fetch_table_law_gptoss.json`, consistent with `concur.txt` | ✓ |
| 11 | Table 4, "New" column (11%: 3, 0, 35, 19, 46; 25%: 20, −2, 27, 24, 48, with t-intervals) | 3.4 [−0.9, 7.7]; −0.4 [−2.2, 1.4]; 34.7 [21.9, 47.5]; 19.0 [13.9, 24.2]; 46.3 [37.7, 54.8]; 25%: 19.8, −2.4, 27.3, 24.4, 48.3 | ✓ |
| 12 | Table 4, "Panel (15)" column (11%: 0 [−2, 3], 0 [−2, 2], 33 [21, 44], 15 [11, 19], 52 [43, 62]; 25%: 21, −3, 31, 21, 48) and 39% on 13 fast-link machines | 15 distinct GPUs (096a/b, 099a–j, 100b, 100f, 101b): 0.5 [−1.8, 2.8], 0.0 [−1.9, 1.8], 32.6 [21.0, 44.3], 14.9 [10.7, 19.1], 52.5 [42.6, 62.3]; 25%: 20.6, −3.2, 31.3, 20.6, 48.1; fast-link 38.8% | ✓ |
| 13 | Interaction positive on 9 of 10 panel main-job machines and on 5 of 5 new ones | Job 099: negative only on 099f at 11% (−0.48 ms). Job 109: +1.5 to +7.5 ms at 11% on all 5 | ✓ |
| 14 | Table 6, RTX 4090 rows with bracketed trend predictions | 111d 0.963 [0.99], 1.128 [1.11], 1.033, 26%; 111f/g 1.007/1.005 [1.01], 1.198/1.196 [1.13]; 112b 1.133 [1.14], 1.299 [1.26]; 112a 1.312 [1.25], 1.465 [1.37]; 25% likewise | ✓ |
| 15 | Trend within 7% (11%) and 5% (25%) on the RTX 4090s; "no change" would miss by 46% / 69% | max \|Δln\| = 0.064 / 0.049; 112a read-ahead 1.465 / 1.689 | ✓ |
| 16 | Trend applied after the fact to job 109's RTX 5090s misses 5 of 18 cells, by up to 18% | 5 of 18 outside ±0.10; worst Δln 0.162 (17.6%); systematic by host | ✓ ! (W4) |
| 17 | Layer-ahead variants on the slow-link RTX 4090s 0.64–0.72× at 11%; Margin up to 26% capture; variants read 1.57–1.76 R⋆ | pf/R1/R2 0.644–0.724; dk 1.033–1.041, capture 20–26%; reads 1.57–1.76 R⋆ on job 109 (pf 1.78 on the RTX 4090s) | ✓ |
| 18 | Table 12 (job 112): 1R / 2R / early speeds and misses on 4 machines; early admits 11.7–11.8 against 6.2–7.4; reads +4–7%; early misses 0.97–0.98× (11%), 0.95× (25%) | e.g. 112g 1.406 / 1.030 / 0.977, misses 40.7 / 55.6 / 54.5; admits 11.72–11.75 against 6.20–7.37; reads +4.2–7.1% | ✓ |
| 19 | Table 3, host B: 34.9 / 54.0 / 69.9, 40.0 / 85.6 / 109.2, 47.7 / 132.2 / 152.5, 19.5 / 38.7 / 40.0 tok/s; Ours ÷ FT 1.29 [1.28, 1.31], 1.27 [1.25, 1.29], 1.15 [1.13, 1.17], 1.03 [1.02, 1.04] | `081/bs1.jsonl`, launch 2, paired bootstrap over 30 problems | ✓ |
| 20 | Table 3, host S: all six cells; 1.21 [1.19, 1.22] at gpt-oss 11%; 0.97 [0.96, 0.99] at Qwen3 43.75% | `089/bs1.jsonl` | ✓ |
| 21 | Host B and S probes: CPU 72 / 56, link 53 / 53, together 78 / 66, highest 88 / 71 GB/s; bound 172 / 140 tok/s; ours 41% | Probe files: 71.6 / 56.2, 53.2 / 52.8, 77.7 / 66.1, 87.5 / 71.3; 172.4 / 140.4 tok/s | ✓ |
| 22 | Running example (Ryzen 9 9950X, 11%): Eq. 1 10.0, Eq. 2 13.0, deployed 20.6, MIN-1R 15.2, read-ahead 13.8 ms; Table 14 O4 row 20.59 / 20.18 / 20.11 / 19.78 / 22.03 / 19.64 / 15.13 / 14.06 / 13.78 / 10.02 | `096a`: 10.02, 12.96, 20.64, 15.16, 13.81; as 1000 / mean speed, every O4 entry matches | ✓ |
| 23 | Table 13 (job 095), gpt-oss 11%: deployed 55.5; MIN-1R 1.29 [1.28, 1.29]; read-ahead 1.39; Belady-2R 0.91 [0.90, 0.92]; ahead only 1.26; paced 1.45 | 55.5; 1.286 [1.279, 1.294]; 1.394; 0.910 [0.899, 0.921]; 1.257; 1.453 | ✓ |
| 24 | Table 15 (job 093): 47.4 → 43.7, 0.92 [0.91, 0.93]; 77.6 → 108.5, 1.40 [1.38, 1.41]; 27.1 → 24.8; misses 47 / 11 / 132; admissions 35 / 20 / 63 | `oracle_w0` rows and counters | ✓ |
| 25 | Table 30 (job 102): every per-layer, per-token, link-only and CPU-only share on 3 machines × 2 budgets; terms 7.04 / 2.84 / 9.70 / 3.91 / 9.56 / 3.85 ms; repetitions within 0.9% | `readsched_C*.txt` and `concur.txt`; repetition spread ≤ 0.87% | ✓ |
| 26 | Rounds: 108 comparisons on 14 machines, median 0.2% (max 1.8%), median half-width 0.6% | n = 108; median 0.17%; max 1.77%; half-width 0.61% | ✓ |
| 27 | Relaunched RTX 5090s: deployed time within 0.7% of job 109; ratio moved by up to 12% | 0.7% (5950X) and 0.4% (7945HX); 0.84 → 0.74 | ✓ |
| 28 | G (Nsight) 4.1–4.7 ms for gpt-oss; EPYC traces held almost no kernels; T_GPU = 2.9 ms is the smallest profile | `g_prof.json`: 4.05–4.67 ms on the RTX 5090s; 108a/b G ≈ 0.00–0.02; `decomp_measured.json` T_GPU range 2.94–3.38 ms | ✓ |
| 29 | Eq. 3 = Eq. 1 on the median consumer machine; up to 1.56× where the link is slow | Median 1.000; consumer maximum 1.557 (TR 9960X); server maximum 1.85 (AMD ES, excluded from that claim) | ✓ |
| 30 | Ours at 31–54% of Eq. 1 on consumer machines (12–46% on server ones); the 5 new machines at 41–48% | Launch-level: consumer 30.6–54.0% (TR 9960X and 100f); server 12.4–46.1%; job 109 41.5–48.0% | ✓ ! (W2: the 54% is on a probe-exceeding host) |
| 31 | Few-1R beats deployed on all 15 stable machines, geometric mean 1.26× (t-interval 1.19–1.33); lost on two unsteady machines | 15 distinct GPUs from jobs 104–113: geometric mean 1.257, arithmetic 1.263 [1.19, 1.33]; unsteady or invalid: 0.85, 0.97, 1.30, 1.16 | ✓ ! (Table 1 wording; W6) |
| 32 | Published median 13.6% (20 in-class rows); 9.5% over 29 trace rows | Medians of the Table 32 rows: 13.6 and 9.5 | ✓ |
| 33 | Scorecard: Table 10 totals 595 / 685 / 323 / 84; per-job clause counts; hand tally 565 = 218 + 193 + 142 + 8 + 4 | Column sums and per-job totals all match; job 113 rows of Table S19 equal my H1–H6 values | ✓ |
| 34 | Registration timing: 94 rentals of jobs 093–113; every commit at least 2.7 s before its rental; post-launch header edits only for substitutions, with predictions unchanged except job 100's added P9 | `git log %ct` against the ledger `start`: minimum lead 2.7 s (105f). Post-start commits on jobs 100, 102, 105, 106, 107 (×3) and 111; their diffs show host substitutions and one added prediction (job 100 P9), as disclosed | ✓ |
| 35 | Gates: job 107, 3 stopped at the gate and 2 failed the round check (3.5%, 6.0%); job 108, 2 at the gate and 1 failed the round check (7.0%); job 110, one ratio gate at 0.2499 and four V1 failures (loss 3% high) | `v0.txt` and `validity.txt` per host: exactly as stated; RTX 4090 loss 0.1963–0.1965 against 0.190 | ✓ |
| 36 | Job 110e's first round off the trend by −5.6% | Read-ahead Δln −0.057 (−5.6%). Unreported: capture 78% in that round | ✓ ! (W6) |
| 37 | Cost: 147 rentals, 86 offers, $107.5 | Ledger: 147, 86, $107.48 | ✓ |
| 38 | Horizon: 0.66–0.81 C distinct experts on the AIME routing | gpt-oss trace: D(W50) ≈ 0.81 C at 11% and 0.69 C at 25%, with W50 log-interpolated from Table 20 | ≈ ✓ |
| 39 | Machines with the same CPU differ by up to 29% in deployed time | 285K panel machines 099a 18.14 ms and 099f 14.06 ms (1.29×) | ✓ |

I found no number that failed to reproduce. The issues are interpretive (W1–W4) and in labelling (Table 1).

---

## IEEE format

I reviewed `paper.pdf`; these notes are from skimming `ieee-paper.pdf` (10 pp.) and `ieee-supplement.pdf` (29 pp.).

- **Front matter.** The IEEEtran title and anonymized author block, the bold abstract, and the Index Terms (alphabetical) are correct. The page-1 footnote names a file ("ieee-supplement.pdf"); IEEE style would say "the supplementary material" without a filename.
- **Section structure.** Sections run straight into `\paragraph`-level run-in heads ("a) Models and workload:"), skipping the A./B. subsection level. IEEEtran allows this, but it is unusual for a conference paper.
- **Captions.**
  - Tables are captioned above with Roman numerals, and figures below as "Fig. n.", which is correct.
  - Table I's caption is a five-line all-caps definitional paragraph. IEEE convention keeps table captions short and puts definitions in table notes.
  - In the supplement, the long tables use a different caption form ("TABLE S5: Failed …", "TABLE S26: …") from the stacked "TABLE S1" form used elsewhere.
- **Reference style.** The main paper has numbered IEEE references ([1]–…), which is correct. The supplement has its own [S1]–[S64] list, also correct, though it duplicates many main-paper entries. A few entries keep non-IEEE annotations, such as "Accepted at DAC 2026 (per arXiv v2)" and "accessed September 2026".
- **Float placement.**
  - Table II is first referenced in Section II (pp. 2–3) but appears on p. 4.
  - Tables IV and V share p. 6.
  - Table I is a two-column float on p. 2, as LaTeX requires.
  - Otherwise the floats are near their first reference.
- **Page count.** The paper is 10 pages: about 8.5 of text and 1.5 of references. That fits a "10 pages including references" limit but not an "8 + references" one; check the target venue. The supplement is single-column and 29 pages, which is fine for supplementary material.
- **Cross-references between documents.**
  - Main → supplement uses "Appendix X", "Table S#", "Fig. S#" and "Eq. (S1)" consistently. Eq. 4 is correctly renamed Eq. (S1) in both documents.
  - Supplement → main uses bare "Table I/III/VI", "Fig. 3/4/5" and "Section V" without saying "in the main paper". The S-prefix avoids collisions, but checklist row 12 reads awkwardly ("Figure 3 … Fig. 5 … Fig. S3 … Figs. S4 and 4"), mixing "Figure" and "Fig." in one cell.
  - **A third document is referenced.** Inside the IEEE supplement, "the supplement" refers ambiguously to itself and to the separate scorecard (`paper/supplement.pdf`), for example "the supplement lists every clause" and "(the supplement)" in Tables S2 and S5. Appendix A's last bullet ("The supplement: every clause scored …") points to a document that is not one of the two IEEE PDFs. Either merge the scorecard into the IEEE supplement or call it "the scorecard (separate artifact)" throughout.
