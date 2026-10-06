# Number check 5 (6 October 2026)

This is an independent check of `paper/paper.tex` (abstract to Conclusion, with the main-text captions and tables) and
of appendix sections F (`app:foresight`) and G (`app:system`). The other appendix sections were checked only for
consistency with these.

**Method.**
- Every macro used in the main text (215) was resolved from `paper/wsg_*.tex`.
- About 150 of them were recomputed directly from `prereg/*.json`, using my own snippets rather than the generating
  scripts:
  - `foresight_095/096a/096b`
  - `accounting_measured`
  - `learned_offline` (the `q8` entries)
  - `freetoken_tuned_098`
  - `policy_study`, `batchk_trace`, `audit/audit_sol`
  - `law_frozen_later`, `shapley_gap`, `speed_limit_v2`, `foresight_single_read_sim`, `scorecard_clauses`
- The rest were traced to their scripts and spot-checked.
- Raw results were used where needed:
  - the AIME routing traces in `gpu-branch/results/084{b,c}`;
  - the job configs (`configs.txt`) and the job 081 server stats.
- The deployed policy was re-simulated with `mosl.ecsim_fast` to check the learned-replay baseline.
- `figs/factorial.pdf` was re-rendered from the current JSON in a scratch copy. It is pixel-identical to the committed
  PNG.

Conventions: ratio = paired ratio of mean speeds to `base` on the same host; reads = misses + admissions (+ prefetches);
R* = `opt_reads_per_token`.
- Host-bound cells (HB): G11, G25, Q12, Q25.
- GPU-bound cells: G40, Q43.
- Hosts: O3 = 095, O4 = 096a, O5 = 096b.
- Oracle configs: base, foa, fetch, lead2, both2, both3p (paced), nb2 (Belady, one read), hitopt (Belady, two reads,
  unpaced), hitoptp (the same, paced), bypass (MIN, serve-then-copy).

**Result.**
- No macro is mis-generated. Every recomputed macro equals its JSON value at the printed precision.
- The defects are in the prose:
  - one universal claim that is false (defect 1);
  - the paper's central contrast, which the data contradict at the 25% host-bound cells on three hosts (defect 2);
  - a baseline mismatch in Section 5 (defect 3);
  - a claim that the measured accounting "confirms" foresight is the largest cost, which O5 contradicts (defect 4);
  - scope and wording slips.

---

## Defects (most serious first)

### 1. "the states fall in the same order on both hosts at every host-bound cell" is false. HIGH

- **Where.** Section 4, `paper.tex` 246–247: "Every interval is within ±0.037, and the states fall in the same order on
  both hosts at every host-bound cell."
- **Data.** The order of the ten states differs between O4 and O5 at every host-bound cell. It also differs for the six
  states in the figure. Ordered by speed, fastest first:

  | Cell | O4 | O5 |
  |---|---|---|
  | G11 | both3p > fetch > nb2 > hitopt > bypass > foa > base | both3p > fetch > base > foa > bypass > nb2 > hitopt |
  | G25 | both3p > fetch > nb2 > hitopt > bypass > foa > base | both3p > fetch > bypass > base > foa > nb2 > hitopt |
  | Q12 | … fetch > nb2 > foa > base > bypass > hitopt | … fetch > foa > base > bypass > nb2 > hitopt |
  | Q25 | … nb2 > hitopt > bypass > foa > base | … bypass > base > foa > nb2 > hitopt |

  Only the top of the order is shared: best prefetch > fetch > foa. That is prediction 4 of
  `foresight_outcome_096.md`, which held at all eight host-cells. The Belady and serve-then-copy states sit above the
  online policy on O4 and below it on O5.
- **Fix.** "…and the best prefetching state, fetch and foa fall in that order on both hosts at every host-bound cell;
  the Belady and serve-then-copy states sit above the online policy on O4 and below it on O5."

### 2. "Spent the usual ways, foresight gains little": the data contradict this at the 25% host-bound cells. HIGH

- **Where the claim appears.**
  - Abstract, lines 34–36: "spent as caches usually spend it … gains little or loses where host memory binds".
  - Thesis, lines 52–54: "recovers that excess only when it is spent as MIN spends it".
  - Section 4: the header "Spent the usual ways, it does not" (line 250), and lines 264–266: "knowing it pays only when
    spent on what MIN admits … Our first two oracle jobs (O1, O2) spent it only in the usual ways and found it worth
    little where host memory is saturated".
  - Conclusion, lines 432–437: "recovered only when … spent as the optimum spends it … gains little or loses where host
    memory binds".
- **Data.** G25 and Q25 are host-bound on every oracle host (App. F line 255). At those two cells the usual ways gain
  substantially on three hosts:

  | Host | State | G25 | Q25 |
  |---|---|---|---|
  | O1 | hit-optimal, paced | 1.40 [1.38, 1.41] | 1.25 |
  | O2 | hit-optimal, paced | 1.63 | 1.41 |
  | O4 | nb2 | 1.29 | 1.26 |
  | O4 | hitopt | 1.27 | 1.25 |
  | O4 | hitoptp | **1.53** [1.51, 1.55] | 1.31 |
  | O4 | fetch (for comparison) | 1.43 | 1.51 |

  - On O4 at G25 the paced Belady prefetch with two reads (hitoptp) beats MIN fetched once.
  - The claim holds at G11 and Q12 on every host, and at all four host-bound cells on O5 (nb2 0.64–0.83; hitopt
    0.54–0.82; hitoptp 0.83–1.16).
  - hitoptp is one of the ten configurations but appears nowhere in the main text, `fig:factorial` or
    `tab:accounting_measured`. That table's "hit-opt." column also mixes paced O1/O2 runs with unpaced O3–O5 runs.
- **Fix.**
  - Scope every instance to "at the two lowest budgets, and on the slow-link host O5".
  - State that at 25% the usual ways gained 25–63% on O1, O2 and O4, always below MIN's paced single-read prefetch on
    the same host.
  - Drop "only".
  - Report hitoptp, or say why it is left out.

### 3. The learned-order replay's "deployed policy" is not the engine's deployed policy. MEDIUM–HIGH

- **Where.** Section 5, lines 340–343: "Counted from the deployed policy, which reads each admitted expert twice,
  single-read admission alone closes 28–48% of the gap to MIN and the learned order 47–57%". Also the `tab:learned`
  "deployed" column.
- **Reads.** The column (70.8 / 40.0 / 191.6 / 108.9 at the host-bound cells) is 6–21% above the engine's measured
  online reads at the same cells (63.4–64.0 / 34.0–34.9 / 160.0–167.0 / 89.8–94.1 on O3–O5).
- **Admissions.** Re-simulating the replay's deployed policy on the same AIME traces gives these admissions per token:

  | Cell | Replay | Engine |
  |---|---|---|
  | G14 | 8.2 | 3.3–3.6 |
  | G32 | 9.0 | 4.2–4.9 |
  | Q16 | 10.7, at κ=2 | 5.0–6.3 |
  | Q32 | 11.0, at κ=2 | 5.2–6.8 |

  - The replay uses κ=2 for Qwen3 (`scripts/provenance.MODELS`). The engine runs κ=1 for both models (`configs.txt`
    of 096; the job 081 stats show `kappa 1` for Qwen3).
- **What the engine shows.** Reading the deployed policy's admissions once (foa) closes 16 / 25 / 10 / 13% (O4) and
  16 / 27 / 11 / 16% (O5) of the deployed-to-R* gap, not 28–48%. This matches Section 4's own "removes 4–22% of its host
  reads".
- The single-read column of the table is close to the engine's foa (within about 8%), so the 17–30% "closed" figure is
  not affected.
- **Fix.**
  - Rebuild the "deployed" column with κ=1 and the engine's admission rate, or replace it with the engine's measured
    online reads.
  - Restate the sentence with the engine's numbers, or drop it.

### 4. "The engine confirms" that foresight is the largest cost: O5 contradicts it. MEDIUM–HIGH

- **Where.**
  - Line 289: "The accounting names foresight across tokens as the largest cost where the host binds, and the engine
    confirms it."
  - Conclusion, line 431: "Where the host binds, the largest piece of the gap is the policy's ignorance of future
    routing".
- **Data** (`accounting_measured.json`, host-bound cells):
  - **O5.** At every host-bound cell, "the rest" (57 / 52 / 62 / 53%) exceeds bytes and overlap combined (43 / 48 / 39 /
    47%).
  - **Bytes alone.** Bytes alone are below the rest at 6 of the 12 host-cells: O3 G11 (41 against 42), O3 G25 (34
    against 43), and all four on O5.
  - **O4.** The claim holds on O4, and on O3 once overlap is counted.
- **Fix.** "On O3 and O4 the bytes and overlap foresight recovers are most of the gap; on O5 the rest, which no oracle
  touches, is the largest share (52–62%)."

### 5. The range "1.6–2.3× the optimum's bytes" for the usual ways omits the Belady prefetch with two reads. MEDIUM

- **Where.**
  - Abstract, line 35: "by Belady prefetch or by copying what the CPU has just served, it reads
    \fxUsualReadsOverOptMin–Max [1.6–2.3]× the optimum's bytes".
  - Conclusion, line 436 (same macro).
- **Data.**
  - The macro covers only nb2 and bypass on O4/O5.
  - Belady prefetch through serve-then-copy (hitopt) reads 1.77–2.53× R* on O4/O5 (Q12 O4 2.53; Q12 O5 2.45), and
    1.77–2.47× on O3.
  - hitoptp reads 2.01–2.19×.
  - The existing macro `\fsExcessReadsOptMin–Max` already gives 1.7–2.5 for jobs 093–095.
- **Fix.** "1.6–2.5×", taken over nb2, bypass, hitopt and hitoptp.

### 6. "most published systems stand at a sixth to a quarter of a latency-free ceiling" is false. MEDIUM

- **Where.** Section 3, line 202.
- **Data** (`audit_sol.json`, the 20 in-class rows):
  - 4 rows lie in [16.7%, 25%];
  - 12 lie below a sixth;
  - 4 lie above a quarter.
  - Median 13.6%; quartiles 8.2–23.1%.
- **Fix.** "half of them stand below a seventh, the middle half at a twelfth to a quarter".

### 7. "Running every expert on the CPU, the two engines are within 1.13× … so the lead is not CPU kernel quality." MEDIUM

- **Where.** Section 2, lines 150–151.
- **Data** (`freetoken_tuned_098.json`).
  - **Which engines.** The comparison is stock llama.cpp against FreeToken's CPU backend, not our cache.
  - **gpt-oss.** llama.cpp is 1.13× [1.126, 1.130] faster. That is comparable to our gpt-oss lead on the same host
    (1.09–1.17×). At host-bound cells most misses run on the CPU, so a 13% CPU-path advantage is not excluded as a
    contributor.
  - **Qwen3.** Only on Qwen3 (0.99×) does the data support the claim.
- **Fix.** "With every expert on the CPU, stock llama.cpp runs 1.13× (gpt-oss) and 0.99× (Qwen3) FreeToken's CPU
  backend, so CPU kernels explain none of the Qwen3 lead and at most part of the gpt-oss lead."

### 8. "The rest is three things": the split is not measured, and the cited evidence points the other way. MEDIUM

- **Where.** Section 4, lines 274–278: "The CPU misses that remain are served in sequence with the step (the law
  over-predicts the prefetching states' time per token by \fsbpLawErrMin to \fsbpLawErrMax [+5 to +36]%)".
- **Logic.** The law prices host reads in sequence with G. A positive over-prediction of time means the states run
  faster than serial reads allow. That is evidence of overlap, not of CPU misses being served in sequence.
- **Measurement.** No measured state isolates any of the three parts of "the rest". The decomposition is asserted.
- **Scope.** The range is O3 only, while the paragraph describes O3–O5. For both3p on O3–O5 the error is +1% (O5 Q12)
  to +43% (O5 Q43).
- **Fix.**
  - Present the three as candidate causes.
  - Drop the parenthetical, or cite evidence that the CPU misses are serial (for example, per-layer timers).
  - Give the O3–O5 range.

### 9. "the GPU runs at 52–61% of the datasheet rate the limit counts, most of the gap at the GPU-bound cells". MEDIUM

- **Where.** Lines 278–279.
- **Data.** The all-in-VRAM time minus the datasheet GPU term is 1.90 ms (gpt-oss) and 2.22 ms (Qwen3). That is 24–33%
  of the gap at the GPU-bound cells on O3–O5, and 45–58% of "the rest".
- **The cited profile.** The Nsight profile cited in the next breath has serialised PCIe copies as the largest single
  item (4.93 of 10.92 ms).
- **Fix.** "about half of the rest at the GPU-bound cells" (approximate: the limit's GPU term at c\*>0 is somewhat below
  the all-GPU datasheet time).

### 10. "Our deployed hysteresis costs 1–8% … and up to 22% on Qwen3" uses κ=2, which the engine does not run. MEDIUM

- **Where.** Section 5, lines 314–315. The `tab:policy` row "with hysteresis κ" has the same problem.
- **Data.** `policy_study.json` uses κ=2 for Qwen3 (κ=k/4). The engine runs κ=1 (Section 2: "margin κ=1"; job configs).
  The Qwen3 figure (4.6 / 11.5 / 22.4%) is therefore not the deployed setting.
- **Fix.** Recompute Qwen3 at κ=1, or say that the trace study uses κ=k/4.

### 11. Limitations: "its gains differ between them by up to 0.26". MEDIUM

- **Where.** Lines 406–407.
- **Data.**
  - 0.26 is fetch's O4–O5 spread only (`\fxFetchSpreadMax`).
  - Other states differ between hosts by up to: nb2 0.49 (Q25: 1.257 against 0.763), hitopt 0.49 (Q25), both3p 0.35
    (Q25), hitoptp 0.37 (G25).
- **Fix.** "fetch's gain differs between them by up to 0.26, and other states' by up to 0.49".

### 12. "the union of K positions' expert sets is 20–79% of Kk …, several times C". MEDIUM–LOW

- **Where.** Section 5, lines 356–357.
- **Data** (`batchk_trace.json`, K=8).
  - For the "same" variant (the one the sentence is about), the union is 0.37–4.3× C. It is below 2C in 54 of 81 rows,
    and below C for gpt-oss-120b at the larger budgets.
  - The 20–79% range also pools the "free" variant, in which rejected drafts are not counted.
- **Fix.** "…up to four times C at small budgets".

### 13. "within 3% of what the simulator predicts". LOW

- **Where.** Line 241, hard-coded.
- **Data.** Engine/simulation − 1 for fetch's reads is +0.5% to +3.6% (largest at G40). App. F says "0 to 4%" and
  App. B says "within 4%".
- **Fix.** "within 4%" (use `\fsfSimReadsDiffMax`).

### 14. Intro: "on three hosts and in randomised order". LOW

- **Where.** Lines 56–57.
- **Data.** O3 (job 095) ran in a fixed order (Fig. 2 caption, `tab:accounting_measured` caption). Only O4 and O5 were
  shuffled.
- **Fix.** "on three hosts, two of them in randomised order".

### 15. Scope mismatch in "removes 4–22% of its host reads but moves its speed by only 0.97–1.03×". LOW

- **Where.** Lines 261–263.
- **Data.**
  - The speed range covers the host-bound cells only.
  - The reads range covers all six cells. At the host-bound cells foa removes 4–15%.
  - At all six cells foa runs 0.95–1.03× (O5 G40 0.947, Q43 0.966).
- **Fix.** Use one scope for both, for example "4–15% … 0.97–1.03× at the host-bound cells".

### 16. "(of which reading the online policy's admissions once is −4 to 4% and foresight 26–57%)" covers O4/O5 only. LOW

- **Where.** Lines 269–271. The sentence covers O3–O5, but job 095 has no foa run.
- **Fix.** Add "on O4 and O5".

### 17. FreeToken's VRAM is given two ways in Section 2. LOW

- **Where.** Line 135: "preallocates 28.4–28.6 GB". Line 149: "reserves 28.4–30.1 GB … at every budget".
- **Data.** The job 098 JSON gives 28.4 (gpt-oss) and 28.6–30.1 (Qwen3; 30.1 at 43.75%). The values are GiB.
- **Fix.** Reconcile the two (note the host for each), and use GiB.

### 18. App. G "FreeToken tuned": "the calibrated hybrid with 8 threads at four cells and offload at gpt-oss 25 and 40% and Qwen3 43.75%". LOW

- **Where.** `app_wsg.tex` line 442, hard-coded.
- **Data.** That makes 4 + 3 = 7 of 6 cells. FreeToken's best setting was 8-thread hybrid at three cells (G11, Q12, Q25)
  and offload at three.
  - Also, "the fixed caps were never better than its own split": at G40 cap 1 is 1.003× the calibrated split (110.94
    against 110.66 tok/s).
- **Fix.** Write "three cells", and "never better by more than 0.3%".

### 19. "Tuned per cell on a third 9950X host". LOW

- **Where.** Line 146.
- **Data.** Of the Table 1 hosts, only S is a 9950X (B is a 9950X3D). The job 098 host is a 9950X.
- **Fix.** "on a third host (Ryzen 9 9950X)".

### 20. `tab:learned`'s "MIN" is not R*. LOW

- **Data.**
  - The column (39.0 / 15.5 / 6.8 / 100.4 / 44.3 / 13.8) is the no-evict MIN (`speed_limit_v2` `pol_084`).
  - R* is the exact MIN (38.3 / 15.3 / 6.8 / 96.5 / 43.4 / 13.7), which the glossary, Section 4 and the limit use.
  - Against R*, the learned order closes 16–29% at the host-bound cells, not 17–30%.
- **Fix.** Use R*, or label the column "MIN (no-evict)".

### 21. Two other ranges are wider than the cells the sentence covers. LOW

- **"it still reads 1.43–1.90× MIN".** Line 343. The sentence follows host-bound ranges, but the macro spans all six
  cells (1.84 and 1.90 are GPU-bound). At the host-bound cells the range is 1.43–1.69.
- **"on O3 its reads are 1.18–1.58× R*".** Line 275 (`\fsbpReadsOpt`). The paragraph is about host-bound cells, but
  the range spans all six. At the host-bound cells it is 1.18–1.43.
- **Fix.** State the scope, or use the host-bound values.

### 22. Thesis: "online caches read 35–110% more". LOW

- **Where.** Line 52.
- **Data.** 35–110% is the best online policy on the traces. The deployed cache in the engine reads 1.66–2.88× R*
  (66–188% more) on O3–O5.
- **Fix.** "the best online policy reads 35–110% more on the traces; our deployed cache, with its second read,
  66–188%".

### 23. Causal statements in Section 4 given as fact. LOW–MEDIUM

- **"a fetch in the step waits on the link, and behind O5's slow one it gains less".** Line 247. O3 and O4 have similar
  links (43 and 45 GB/s), yet fetch differs between them by 0.11–0.13 at G25 and Q25. O5 also differs in CPU (X3D).
  The link is plausible but not isolated.
- **"rationed by its margin".** Line 261. The simulator with the same κ admits 2.1–2.4× more than the engine at G14 and
  G32, so the engine's admission path also rations.
- **"since their background copies hide behind later steps".** Line 263. This explanation was not measured.
- **"The second read matters because foresight admits often".** Line 259. bypass also loses hits to copy latency (G11
  O4: hit rate 0.645 against fetch's 0.723; reads 67.1 against 39.9 + 20.3 forced).
- **Fix.** Hedge these as explanations ("consistent with").

### 24. "fewer reads are worth time only where host memory is the binding term". LOW

- **Where.** Line 326.
- **Data.** fetch gains 1.20–1.37× at the GPU-bound cells on O3–O5.
- **Fix.** Scope the statement to the engine's serial host term, or drop "only".

### 25. Section 5 says the copy-path lead is "two to three steps in ours". LOW

- **Where.** Line 326.
- **Data.** App. F (lines 342–343) measures the paced path's lag at 2.8–4.7 steps ("three to five steps").
- **Fix.** "two to five steps".

### 26. "against the deployed serve-then-copy form with hysteresis, \cref{tab:policy}". LOW

- **Where.** Lines 318–319.
- **Data.** `tab:policy`'s "with hysteresis κ" row is the single-read `dfk`. The serve-then-copy `dfa-deployed` is not
  in the table.
- **Fix.** Correct the cross-reference, or add the row.

### 27. Wording that overstates small effects. LOW

- **"gains nothing on Mixtral-8x7B".** Line 155. The ratio is 1.015 [1.008, 1.024]; App. D says "ties".
- **"a quarter to two fifths".** Conclusion, line 429. The range is 25–43%, so "two fifths" is exceeded.
- **"changes nothing at gpt-oss".** Line 254. The two Belady variants differ by 1–2.5%.

### 28. Appendix B is stale for jobs 096 and 098. LOW

- **Job range.** `app_wsg.tex` line 25 says "jobs 073–\scLastJob [097]". The scorecard ends at 098, and 097 has no
  clauses.
- **Equality failures.** Line 57 says "the 3 equality failures are the 4090 host's FETCH tables". Two are; the third is
  098's "best backend is hybrid at gpt-oss 25%".
- **Threshold failures.** Lines 53–60 list 11 of the 20 threshold failures. They omit 096's five between-host fetch
  spreads and nb2 ≤ 1.05× at Q12 on O4, and 098's 8-thread clause and its two CPU-only clauses.
- **Fix.** Set `scLastJob` from the last scored job, and complete the lists.

### 29. App. F's list of 096 failures is incomplete. LOW

- **Where.** Lines 241–245.
- **Omitted.** nb2/lead2 reads below the 1.25 band at G11 on O4 (1.249) and at Q12 on O5 (1.14); foa reads at O5 G40
  (22.3% against ≤22%).
- **Fix.** Add them, or say "among them".

---

## Notes (checked and holding, or minor)

### Abstract, Introduction and Conclusion

- **The cache against the limit and the audit.**
  - 25–43% of the limit (B and S).
  - 13.6% in-class median, with quartiles 8.2–23.1 (exact order statistics).
  - 52 rows from 13 sources; 21 imputed; 11 i.i.d.
- **The oracle ranges.**
  - 16–51% (fetch, 18 host-cells on O3–O5, all positive).
  - 25–81% (both3p, all positive).
  - 3 hosts.
- **The learned order.** 17–30% closed (the q8 entries, host-bound cells).
- **The instrument.** 2.0–4.0× llama.cpp; 11 of 12 cells ahead.
- **Counts.** The scorecard has 481 clauses (207 held, 140 held on the point, 122 failed, 8 untested, 4 void). W50 =
  0.59 (C/k)^1.33 (cross-check macros `fsPreX`, `fsExpX` agree).

### Section 2

- **Host B, host S and the law.**
  - B's memory clock: 1.21× (17,001/14,001).
  - S's CPU: 22% slower (56 against 72 GB/s).
  - Law: 3.7% median on 7 hosts (33 cache rows).
  - Frozen law: 30 measurements on 4 hosts (089, 093–095; 096 is not included), median 5.6%, gpt-oss −5.8% to +3.5%,
    Qwen3 +5.0% to +21.4% (speed; the signs follow the stated convention).
- **Against FreeToken.**
  - Lead on host B: 15–29% (gpt-oss) and 3–15% (Qwen3).
  - Host S: ahead at 5 of 6 cells.
  - Job 098: best/default 1.00–1.04; ours/best 0.98–1.17, ahead at 5 cells by the lower bound; within 0.034 of host S
    (macro 0.04, ceiling rounding).
  - LRU 6–21% slower; behind FreeToken at four cells.
  - 4090/3090: 1.6–2.6× llama.cpp.
  - Qwen3.6: 1.030, so "within 3%" holds.
- **Expert sizes.** 13.25 MB and 9.44 MB (BF16 Qwen3 expert = 3 × 2048 × 768 × 2 B).
- **The patch.** 3,717 added lines; "3,500-line" is approximate.

### Section 3

- **Limit variations.**
  - Pool: +5–10% at the host-bound cells; reads 4.5–18.6% fewer.
  - Median sample: −7.3% to −7.9% (stated as 8%).
  - GPU efficiency: 52% and 61% (0.516, 0.606).
  - 279 and 192 tok/s, 9% and 8% above 255 and 178 tok/s.
  - Running example: 430 and 109 tok/s.
  - With every variation: 33–57%.
- **Shapley.** Foresight 37–53% at B's host-bound cells; overlap largest at G40 and Q43.

### Section 4

- **Factorial macros.** Every fx\* macro matches the JSON.
  - fetch: 1.36–1.51 (O4), 1.16–1.26 (O5); reads 1.04–1.14× R*.
  - both3p: 1.48–1.81 (O4), 1.25–1.50 (O5).
  - Fraction of the limit: 46–75% against 31–49%.
  - Largest half-width: 0.037 (O4 Q43 hitoptp).
  - Spread: 0.13–0.26 (fails the 0.15 prediction at 5 of 6 cells).
  - nb2: 1.6–2.3× R*; 1.05–1.29 (O4), 0.64–0.83 (O5); below lead2 at all 8 host-bound host-cells.
  - nb2/hitopt: 1.01–1.03 at gpt-oss and 1.41–1.54 at Q12.
  - bypass: 1.7–2.1× R*; 0.94–1.18.
  - MIN admissions 2–11× the online policy's (fetch's forced fetches over base admissions); online admissions
    3.5–6.8.
- **Accounting macros.** All match the stated definitions.
  - Bytes 26–60, overlap 7–23, rest 30–62.
  - Single read −4 to 4; foresight given single read 26–57.
  - Model multipliers 1.1–2.1, 1.1–6.4 and 2.1–8.4.
- **Other numbers.**
  - Nsight figures match `qwen3_profile_outcome_079.md`.
  - No-overlap: 0.994–1.005.
  - W50 on O1: 9.2 (log interpolation between W=4 at 30.5% and W=16 at 42.1% of the limit).
- **Figure 2.** The six rows, three hosts and xlim cover every point (0.54–1.81). The caption's link rates (27, 43, 45)
  match the probes.

### Section 5

- **Policy study.**
  - Best online policy 35–110%.
  - Best policy by cell: decayed frequency 12, S3-FIFO 10 (one tie with LRU resolved to LRU), LRU 3, LFU 2.
  - Within 1% of the best: 20 and 18.
  - C/k ≤ 1.5: 7 cells, spread 4.1%.
  - LRU 2–11% worse than decayed frequency on 8 models (Qwen2-57B −0.4 and +1.2).
  - LFU max 6.15; static 1.43–13.24.
- **Drift and skew.** Drift 22–58 and 33–75 (median 51); top-quarter share 31–56 (at E/4).
- **Foresight.** W4 beats every online policy in all 27 cells; median 1.130.
- **Batched verification.** 15–32% (free); 48%; 8–12% (α=0.9); −18 to −20% (α=0.8); Belady −7 to +15%; union 20–79%.
- **Learned order.**
  - 6–12% fewer reads.
  - Without cross-layer scores 14–26%.
  - Ahead of W=1 at the 3 cells with C/k ≥ 7 (G32, G51, Q56).
  - From the simulator's deployed policy (see defect 3): 28–48% and 47–57%.
- **Trace lengths.** 32.6k–152.5k tokens.

### Appendix F (numbers in the text)

- **Probes.** O1–O5 rates: 44/46/49/50.5, 44/46/50/50.6, 53/43/60/61.5, 43/45/51, 47/27/51.
- **FETCH tables.** 0,0,1,2,2 and 0,0,1,1,2,3,4,4,5 on O3; 0,0,1,2,3 and 0,0,1,2,3,4,5,6,7 on O1/O2.
- **Unpaced hit-optimal on O1.** Hits 98.1, 98.7, 74.9 and 99.3%.
- **Admissions.** 16.0 and 30.2 per token (212 and 285 MB).
- **Hit-optimal over bytes-optimal.** 1.245–1.382 ("1.24–1.38").
- **Clause counts.** 36 (9/2/21/4), 38 (12/1/24/1), 56 (44/12).
- **"the trace's 3.9".** This is the nine-model trace study (`foresight_S_exact` 3.86). The AIME trace's own value is
  3.7. Not a defect, but worth saying which trace.

### Appendix G (numbers in the text)

- **Ablation.** Steps 2.0–8.9%; law table 5.1% and 8.2%.
- **Job 077.** 1.28–1.45 and 1.42–2.34.
- **KTransformers.** kt-kernel 0.7.0.post4.
- **Long outputs.** At most 1 of 10 reach an answer.
- **CPU-only.** 1.13× and 0.99×.

### Other observations

- **The deployed-policy simulation's hit rate at Q16.** It is 0.529–0.533 against the engine's 0.581–0.596, a gap of
  4.8–6.7 points, slightly beyond Section 3's "within 5.3 points" (measured against host B there). Worth re-checking
  against O3–O5.
- **"desktop CPUs" (Intro, line 65).** The blind test (Appendix C) includes EPYC server hosts.

## Summary

The numbers are in good shape mechanically. Every macro recomputed from `prereg/*.json` matches at the printed
precision, the factorial figure is current, and the tables agree with their JSON.

The problems are in what the prose claims about the new job 096 factorial and the learned order:
1. **The order of the states.** The sentence that the states fall in the same order on both hosts is false (defect 1).
2. **The central contrast.** "Only when spent as MIN spends it" and "the usual ways gain little where host memory
   binds" appear in the abstract, thesis, Section 4 and Conclusion. They hold at the two lowest budgets and on O5. At
   the 25% host-bound cells the usual ways gained 25–63% on O1, O2 and O4. hitoptp even beats MIN-fetch on O4, and it is
   omitted from the text and figure (defect 2).
3. **The learned replay's baseline.** The replay's "deployed policy" admits about twice as often as the engine and uses
   κ=2 for Qwen3 where the engine runs κ=1. So the 28–48% and 47–57% "counted from the deployed policy" disagree with
   the engine's own foa measurement (10–27%) in Section 4 (defect 3).
4. **The accounting.** "The engine confirms" that foresight is the largest cost does not hold on O5, where the rest
   exceeds bytes and overlap combined at every host-bound cell (defect 4).

The remaining defects are a too-narrow reads range in the abstract, a false "most at a sixth to a quarter" in Section 3,
an over-reaching CPU-kernel inference, an unmeasured three-part "rest", and about fifteen low-severity scope, wording
and stale-appendix slips. Fixing defects 1–4 means changing sentences, not data: each fix can be made with numbers
already in the JSON.
