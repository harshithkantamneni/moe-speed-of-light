# Number check 4 (2 October 2026)

An independent check of the numbers in `paper/paper.tex` (main text) and of the appendix section "Foresight in the
engine" (`app:foresight` in `paper/app_wsg.tex`); the rest of the appendix was skimmed. Every macro was resolved in
`paper/wsg_*.tex` and re-derived with short Python snippets from `prereg/*.json` (not by re-running the generating
scripts). The oracle definitions were checked against `prereg/foresight_outcome_09{3,4,5}.md`, the job script
`gpu-branch/jobs/095_single_read@vast.sh` and the oracle code in `jobs/ec2/llama.cpp-expert-cache-4da6337-oracle.patch`.
`paper/figs/factorial.pdf` was rendered and compared with `foresight_095.json`. Raw stats files, probe outputs, gpu.csv,
gate.txt and bs1.jsonl on the gpu branch were used where a hard-coded number needed a source.

## Summary

- **Statements checked:** about 285 quantitative statements in the main text, `app:foresight`, the Fig. 2 caption and
  the captions of the oracle tables. Each is a macro value, a hard-coded number, or a quantified prose claim ("every",
  "at most", "within", "a third to a half" and the like).
- **Tables and figure checked entry by entry:**
  - `tab_single_read`: 144 entries.
  - `tab_accounting_measured`: 18 rows.
  - `tab_foresight` and `tab_foresight_bytes`: 6 rows each.
  - `tab_headline`: 12 rows.
  - `tab_limit`: 6 rows.
  - `factorial.pdf`: 30 bars and 12 marks.
- **Result:**
  - **29 failed.** These are the 22 defects below; some defects recur in more than one sentence.
  - **About 255 held as written.** The notes below attach to about 40 of them.
  - **All table entries and all figure bars and marks match the JSON** at the printed precision. Every defect is in
    the prose, in captions, or in one table header. No macro is mis-generated.
- **What holds:** the range claims are almost all true minima and maxima of the sets they describe. The exceptions
  are defects 4, 7, 12, 13 and 17, plus note 36.
- **What fails, by theme:**
  1. **The mechanism given for the hit-optimal oracle.** The text says it reads "each admitted expert twice, by the
     CPU and again by its copy". Its own counters contradict this (defect 1).
  2. **Universal claims that one or two cells break.** These are defects 2, 3, 6, 11, 12, 14 and 19.
  3. **Ranges whose scope differs from the sentence that states them.** These are defects 4, 7 and 17.

Sign conventions used below: "law error" = law_ms/ms − 1 (positive = the law over-predicts time); gains = ratio − 1.
O1 = job 093, O2 = job 094, O3 = job 095. Cells: G11/G25/G40 = gpt-oss 11/25/40%, Q12/Q25/Q43 = Qwen3 12.5/25/43.75%.

---

## Defects (wrong or unsupported), most serious first

### 1. The hit-optimal oracle is not a double-reading oracle: its excess bytes are extra admissions, not CPU-plus-copy reads

**Where it appears.** The same claim recurs in several places:
- Main text, line 343–345: "through the background admission path, which serves the miss on the CPU and then copies
  the expert, so that each admitted expert is read twice".
- Line 363: "The hit-optimal oracle on the same host, with its second read, …".
- Lines 366–367 and 373–375: "ran only double-reading oracles … The double read, not the foresight, was what made
  foresight look worthless".
- Fig. 2 caption ("each admitted expert read by the CPU and again by its copy") and its legend ("hit-optimal (two
  reads)").
- Captions of `tab_accounting_measured` ("Belady prefetch (two reads)"), `tab_foresight` and `tab_foresight_bytes`
  ("background admissions, each read a second time by the copy").
- `app:foresight` line 208: "because each admission is read twice".
- The abstract and introduction attribute "foresight look[ing] worthless" to this path.

**What the data say.** An admitted expert can only be read twice (CPU, then copy) if it was first served as a CPU
miss. So the CPU-served misses per token (misses − fetches) are an upper bound on the double reads.
- **Job 095 (O3, hitopt, unpaced), per token:**

  | Cell | CPU-served misses | Admissions | Hit rate |
  |---|---|---|---|
  | G11 | 2.06 | 65.0 | 98% |
  | G25 | 1.27 | 30.5 | 99% |
  | G40 | 0.78 | 13.8 | 99% |
  | Q12 | 44.2 | 151.2 | 77% |
  | Q25 | 1.52 | 84.0 | 99% |
  | Q43 | 0.98 | 29.5 | 99.5% |

  - At most 2–6% of its admissions were read twice at five cells, and at most 29% at Q12.5.
  - Its 1.77–2.47× R* comes from the admissions themselves: admissions alone are 1.57–2.16× R* per cell.
  - The cause is Belady prefetch without bypass, which copies experts the optimum would bypass and re-copies experts
    it evicted.
  - The raw stats file `st_g_C32_hitopt.json` agrees: 234,377 admissions against 13,903 misses over 7,680 steps.
- **Jobs 093 and 094 (paced).** The bound is weaker but still far from "each": at G25, 6.8 CPU misses against 20.1
  admissions (093) and 4.5 against 23.2 (094).
- **The appendix's own evidence.** The unpaced variant on O1 reached 98–99% hits and still read 1.77× R* at G11 and
  lost 4%. The appendix concludes "the extra bytes cause it", which is over-admission, not a second read.
- **Where "two reads" is right.** It holds for the bytes-optimal oracle (094, MIN with bypass with background
  admission): it admits only this step's CPU-served misses, so every admission is a second read. It also holds for
  the online policy.

**Correction.**
- Describe the hit-optimal oracle as "Belady prefetch without bypass, which copies 1.6–2.2× R* experts per token (it
  admits experts the optimum bypasses and re-copies evicted ones)".
- Reserve "two reads per admitted expert" for the bytes-optimal oracle and the online policy, in the captions and the
  legend as well.
- Scope "the double read … made foresight look worthless" to the bytes-optimal oracle. For the hit-optimal oracle the
  cause is over-admission.

### 2. "slower than every single-read variant at every cell" is false at both GPU-bound cells

**Claim.** Main text line 363–364: "The hit-optimal oracle on the same host … is slower than every single-read variant
at every cell". The same sentence appears in `foresight_outcome_095.md`.

**Data** (`foresight_095.json`): at the two GPU-bound cells hitopt is faster than the fetch oracle.

| Cell | hitopt | fetch |
|---|---|---|
| G40 | 162.97 tok/s, 1.266 [1.254, 1.278] | 161.42 tok/s, 1.254 [1.242, 1.265] |
| Q43 | 116.42 tok/s, 1.274 [1.262, 1.285] | 115.50 tok/s, 1.264 [1.251, 1.276] |

- The paper's own `tab_accounting_measured` shows hit-opt below fetch in ms at these cells: 6.14 against 6.20, and
  8.59 against 8.66.
- Fig. 2 shows the orange bar above the light-blue one at both cells.
- Hitopt is slower than every single-read variant at the four host-bound cells, and slower than the three prefetching
  variants (lead, both, paced) at all six.

**Correction.** "…is slower than every prefetching single-read variant at every cell and than the fetch oracle at the
four host-bound cells (it ties fetch at the two GPU-bound cells: 1.27 against 1.25 and 1.26)".

### 3. "44–45% of the limit at every window" is false; the paper's own figure contradicts it

**Claim.**
- Main text line 371: "the hit-optimal oracle's speed at the host-bound cells was R* over its reads times the roof's
  utilisation, \fsmHostFracMin--\fsmHostFracMax [44–45]\% of the limit at every window".
- `fig:foresight` caption: "stays at 44–45% of the limit at every window".

**Data.** The macros are taken from W = all only (`oracle_w0`). Across the window sweep (`foresight_093.json`, % of
limit):

| Cell | W = 2 | W = 4 | W = 16 | W = 64 | W = all |
|---|---|---|---|---|---|
| G11 | 37.4 | 42.0 | 44.0 | 43.9 | 44.0 |
| G25 | 25.2 | 30.5 | 42.1 | 43.7 | 43.7 |
| Q12.5 | 44.3 | 44.7 | 44.8 | 44.8 | 44.8 |
| Q25 | 35.4 | 38.1 | 44.5 | 44.5 | 44.5 |

- Fig. `foresight.pdf` plots exactly these rising curves.
- What is constant across windows is the utilisation, 88.5–94.0% of the probe's best rate at every window. The
  fraction of the limit varies with the reads (2.0–3.7× R*).
- "R*/reads × utilisation" is an identity at host-bound cells (see note 38).

**Correction.** "…at 88–94% of the probe's best rate at every window, which with the whole sequence in view is 44–45%
of the limit".

### 4. The two oracle-gain ranges are in the wrong order, and the hit-optimal range drops host O1

**Claim.** Main text lines 366–368: "Our first two oracle jobs, on hosts O1 and O2, ran only double-reading oracles,
this one and MIN with bypass with background admissions, and found foresight worth
\fsbGainHighMin--\fsbGainHighMax [13–23]\% and \fspGainHighMin--\fspGainHighMax [41–63]\% at the four cells of 25\%
and above".

**Data.**
- The order is reversed: "this one" is the hit-optimal oracle, but 13–23% is the bytes-optimal (bypass) oracle on O2,
  and 41–63% is the hit-optimal oracle on O2 only.
- The hit-optimal oracle on O1 gained 40, 48, 25 and 56% at those cells. On both hosts it gained 25–63%; the macros
  `\fsHitGainMin`–`\fsHitGainMax` (25–63) already exist.
- "Nothing" at the lowest budgets: the bytes-optimal oracle gained +3.3% [2.9, 3.8] at G11 (see note 32).

**Correction.** "…found the hit-optimal one worth \fsHitGainMin--\fsHitGainMax\% and the bytes-optimal one
\fsbGainHighMin--\fsbGainHighMax\% at the four cells of 25\% and above, and at most 3\% or a loss at the two lowest".

### 5. The conclusion's "that piece is real only when each expert is read once" contradicts the body

**Claim.** Conclusion line 516–517: "in the engine that piece is real only when each expert is read once".

**Data.**
- The double-reading oracles of jobs 093 and 094 gained 13–63% at the four cells of 25% and above (the body says so,
  lines 367–369).
- In job 095 the hit-optimal oracle gained 6–27% at four cells: G25 +14%, Q25 +6%, G40 +27%, Q43 +27%.
- They failed only at the two lowest budgets.

**Correction.** "…and at the lowest budgets that piece shows up only when each expert is read once".

### 6. "foresight edges overlap at the GPU-bound cells too" is false on O3, the host of the measured factorial

**Claim.** Main text lines 314–315: "recomputed for the oracle hosts, whose host memory is weaker, foresight edges
overlap at the GPU-bound cells too".

**Data** (`model_terms.shapley_share` in the three JSONs), foresight against overlap, % of the gap:

| Host | G40 | Q43 |
|---|---|---|
| O1 (093) | 33.8 vs 32.3 (foresight ahead) | 33.0 vs 36.4 (overlap ahead) |
| O2 (094) | 36.1 vs 28.6 (foresight ahead) | 37.5 vs 34.8 (foresight ahead) |
| O3 (095) | 30.2 vs 34.6 (overlap ahead) | 31.1 vs 39.4 (overlap ahead) |

Foresight leads at 3 of the 6 GPU-bound host-cells, and at neither cell on O3. The sentence was true for O1 and O2 at
3 of 4 cells, before job 095.

**Correction.** "…on O2 foresight edges overlap at both GPU-bound cells, on O1 at one, and on O3, the factorial's
host, at neither".

### 7. The contribution bullet's accounting ranges cover four cells but the sentence says six

**Claim.** Line 73–76: "…on one host at six cells, beside the model's values:
\amBytesHostMin--\amBytesHostMax [34–49]\% of the gap is the bytes foresight saves,
\amOverlapHostMin--\amOverlapHostMax [8–23]\% the overlap it allows, and the \amRestHostMin--\amRestHostMax [39–43]\%
that remains is named."

**Data** (recomputed from `foresight_095.json` with the stated definitions; matches `accounting_measured.json`):
- These are the four host-bound cells.
- At all six cells the shares are bytes 27–49, overlap 8–23 and rest 39–56%. The two GPU-bound cells add bytes 27 and
  30 and rest 56 and 52. The macros `\amBytesMin` etc. already hold these.

**Correction.** Insert "at the four host-bound cells", or use `\amBytesMin`–`\amBytesMax`, `\amOverlapMin`–`\amOverlapMax`
and `\amRestMin`–`\amRestMax`.

### 8. The running example's lead over FreeToken uses the gpt-oss 11% cell's value

**Claim.** Line 214–216: "the lead over FreeToken at that cell, \ftRatioGptMax [1.29]$\times$, is largely the CPU path…"
("that cell" = the running example, gpt-oss 25% on host B).

**Data.**
- `\ftRatioGptMax` is the gpt-oss maximum, 1.294 at 11%.
- At 25% the lead is 1.275 [1.254, 1.295] (`headline_081.json`), which prints as 1.27 or 1.28.

**Correction.** Use the gpt-oss 25% ratio: 1.27× (add a macro, e.g. `\ftRatioGptMid`).

### 9. "from three misses the law copies some": the table copies from two misses

**Claim.** Line 214: "the table is \texttt{\scTableGptB} [0,0,1,1,2]: one missed expert runs on the CPU …; from three
misses the law copies some".

**Data.**
- The table is f(0..4): the CUDA code indexes `f.table[m]` with m = the number of misses, and `tab_split` shows the
  same zero-based convention.
- So f(2) = 1: with two misses, one is copied. The same is true of every law table in the paper: host S 0,0,1,2,3,
  O3 0,0,1,2,2, and 4090 0,0,1,2,2.

**Correction.** "from two misses the law copies one".

### 10. "1.21× the bandwidth the limit assumes" contradicts the probe the paper itself cites

**Claims.**
- §Setting line 103–104: "whose card reports a \memClkHeadline [17001]\,MHz memory clock, \memClkRatio [1.21]$\times$
  the bandwidth the limit assumes".
- Limitations line 489–490: "host B's memory runs \memClkRatio$\times$ faster than the other cards'".

**Data.**
- 1.21 = 17001/14001 is the clock ratio.
- Host B's card read 1,847.1 GB/s (`081_headline_law/bw.txt`). That is 1.03× the 1,792 GB/s the limit assumes, which
  §4 itself states ("read 1,847 GB/s … against the 1,792 the bound assumes").
- The common-clock RTX 5090s read 1,685–1,697 GB/s (jobs 059–095 `bw.txt`/`gate.txt`; O1's card 1,554–1,558), so host
  B's card is about 1.09× the other cards.

**Correction.**
- "…a 17001 MHz memory clock (1.21× the common clock); it read 1,847 GB/s, 3% above the datasheet rate the limit
  assumes".
- In Limitations: "host B's card reads 9% faster than the other cards'".

### 11. "the law over-predicts overlapped execution by up to 36%": the oracle jobs show up to 53%

**Claim.** §3 line 201–202: "…which the law does not price and \cref{sec:gap} measures (there the law over-predicts
overlapped execution by up to \fsbpLawAbsMax [36]\%)".

**Data.**
- `\fsbpLawAbsMax` covers only the paced single-read oracle of job 095 (max +35.6%).
- The overlapped hit-optimal runs that §4 also discusses are over-predicted by +37.1% to +52.6% on O2 (094
  `prefetch_w0` at the four cells of 25% and above; +52.65% at G25) and by +23.6% to +39.1% on O1. `app:foresight`
  gives "37–53%".

**Correction.** "…by up to 53%" (`\fspLawErrHighMax`), or scope the clause to "the single-read prefetching oracles".

### 12. "a second path adds bandwidth on every host we measured but the 12-channel EPYC": the 3090 host is a counterexample

**Claim.** Line 197–199: "…on every host we measured but the 12-channel EPYC ($B_{cp}$ against $\max(B_c,B_p)$:
\secondMin--\secondMax [8–17]\% on the Ryzen hosts, \secondUltra [1]\% on the Core Ultra, \secondEpyc [33]\% on the
Zen 2 EPYC)".

**Data** (from `fetch_table_law_*.json` and `law_prediction.json` on the gpu branch):
- **The RTX 3090 host (i9-11900KF, job 092)** has B_c 43.88, B_p 24.5, B_both 41.6: −5.2%. `app:grid` says so itself
  ("CPU and link together read *less* than the CPU alone").
- **The 8–17% covers only the five blind-test Ryzen hosts (jobs 072–077).** Ryzen hosts measured later read higher:
  host S +17.7%, host A +22.7% (087) and +18.4% (083), job 086 +17.1%. The oracle hosts read +7.9/+9.0/+13.6%. Across
  all Ryzen hosts the range is 8–23%.

**Correction.** "…on every host of the blind test but the 12-channel EPYC (8–17% on its Ryzen hosts …); the RTX 3090
host's Core i9-11900KF is a second exception (−5%)".

### 13. The model-against-measured multipliers understate the extremes

**Claim.** Lines 383–384: "it over-prices the bytes by a third to a half and the overlap by one and a half to three
times, and under-prices what remains three to five times".

**Data** (model share / measured share per host-bound cell, `accounting_measured.json`):

| Share | G11 | G25 | Q12.5 | Q25 | Range | Text says | Verdict |
|---|---|---|---|---|---|---|---|
| Bytes (model/measured) | 1.34 | 1.51 | 1.34 | 1.44 | 1.34–1.51 | a third to a half | holds |
| Overlap (model/measured) | 1.95 | 1.57 | 3.46 | 1.62 | 1.57–3.46 | 1.5–3× | top is 3.5 |
| Rest (measured/model) | 3.34 | 3.44 | 5.56 | 4.48 | 3.3–5.6 | 3–5× | top is 5.6 |

**Correction.** "the overlap by one and a half to three and a half times, and under-prices what remains three to five
and a half times".

### 14. "the max beats the sum only on the three rows … and ties elsewhere" is not what the per-row errors show

**Claim.** Lines 186–187.

**Data** (leave-one-host-out errors per row, `perlayer_model.json` models A and B):
- **The max does beat the sum by 11 points on the three 53 GB/s FETCH rows.** This part holds.
- **The EPYC 9655 rows are a second set where the max wins.** Max +16.1/+28.6/+22.3% against additive
  +31.1/+41.6/+32.4%: the max is 10–15 points closer.
- **On the two slow-link FETCH rows the additive form is closer.** 270K: −2.0 against +8.8. EPYC 7352: −4.7 against
  +8.5.
- **On three prefetch rows the max is 2.7–4.1 points closer.**
- **The LOHO medians differ:** 3.1% against 4.5% (p90 9.4 against 11.3).

The appendix sentence ("the max beats the sum on three fast-link rows and ties elsewhere") has the same problem.

**Correction.** "the max beats the sum by 11 points on the three rows that copy misses on demand behind a 53 GB/s link
and by 10–15 points on the EPYC 9655, where both fail; the sum is closer by 4–7 points on the two slow-link copy rows;
elsewhere they are within 4 points".

### 15. Limitations "power limits differed (350–575 W)": three rentals had 600 W

**Data.** `gpu.csv` gives 600 W for jobs 088 (slow-link host), 094 (O2) and 095 (O3), and for the RTX PRO 6000
all-VRAM runs. The 3090 had 350 W.

**Correction.** "350–600 W".

### 16. Limitations "130–186-token prompts" are the two models' medians, not the range

**Data** (`prompt_tokens` in `081_headline_law/bs1.jsonl`, identical in 089):
- gpt-oss: 109–879 tokens, median 186.
- Qwen3: 52–853 tokens, median 130.

**Correction.** "prompts of 52–879 tokens (median 130 for Qwen3, 186 for gpt-oss)".

### 17. `app:foresight` "the prefetching single-read oracles at 71–82%" is the paced oracle only

**Claim.** Line 175–176: "…the prefetching single-read oracles at \fsbpHostUtilMin--\fsbpHostUtilMax [71–82]\%".

**Data.** Utilisation at the host-bound cells = reads × S × tok/s / 61.5 GB/s:

| Oracle | Utilisation |
|---|---|
| paced (both3p) | 71.0–81.7% |
| both2 | 64.4–81.5% |
| lead2 | 66.2–85.0% |

**Correction.** "…the prefetching single-read oracles at 64–85% (the paced one 71–82%)". Note also that the online
policy itself runs at 66–82%; see note 37.

### 18. `app:foresight` "so its gain is the bytes it saves" (bytes-optimal oracle) does not hold at the 25% cells

**Claim.** Line 215–217: "the law on its own counters predicts its time within 9% at every cell, so its gain is the
bytes it saves, at most the 21 points of 23 that the law accounts for at a cell".

**Data.** Gain predicted by the law from the oracle's own counters (base_ms/law_ms − 1) against the measured gain:

| Cell | Law-predicted gain | Measured gain |
|---|---|---|
| G25 | 12 | 18 |
| G40 | 21 | 23 |
| Q25 | 4 | 13 |
| Q43 | 18 | 19 |

At Qwen3 25% the bytes explain a third of the gain.

**Correction.** "…the bytes it saves account for 4–21 points of its 13–23% gain (most of it at gpt-oss 40% and Qwen3
43.75%, a third at Qwen3 25%)".

### 19. `app:foresight` "the engine's misses are at or below it [the ideal simulation]" is false at Qwen3 43.75%

**Data** (`foresight_latency_sim.json`): at Q43 the engine's bypass misses are 18.98 per token against 17.58 for d = 1
(8% above). The fitted latency is d = 1.39, which the same paragraph reports ("at or below 1.4 at the GPU-bound cells").
At G40 the engine is below the ideal (8.77 against 9.31).

**Correction.** "…and the engine's misses are below it at gpt-oss 40% and 8% above it at Qwen3 43.75%".

### 20. Scorecard: 16 timed clauses without an interval are scored "held", against the stated rule

**The rule** (Appendix B and `scorecard_clauses.json` `ci_rule`): a clause with no interval is "held (point)";
deterministic clauses are excepted.

**Data.** These clauses have `ci: None`, are timing-based (not deterministic) and are scored "held":
- Job 095, 14 clauses:
  - P2g, P2j: law on fetch within 10%.
  - P4a, P4c: both2 at 55–85% of the limit.
  - P4e–P4h: both2 faster than hitopt.
  - P5b, P5d: law over-predicts both2 by 15–50%.
  - P7a, P7c, P7d, P7f: paced within 10% of unpaced.
- Job 094, 2 clauses: P6a and P6e, the 16-token window keeping ≥70%.

Other jobs score such clauses "held (point)".

**Effect.** Reclassified, strict held is 140/287 = 49% rather than the introduction's \scHeldStrictPct [54]%. Most
would very likely hold if intervals were computed (the margins are wide).

**Correction.** Compute the paired-bootstrap intervals for these 16 clauses, or score them "held (point)" and update
`\scHeldStrictPct` and the 095 count ("44 of 56 held").

### 21. "25 of the 26 other RTX 5090 rentals" counts job directories, not rentals or cards

**Claim.** §Setting line 105.

**Data.**
- `wsg_numbers2.clocks()` globs `results/0[6-8]*/gpu.csv`, one entry per job.
- Those 26 jobs ran on 17 distinct cards by GPU UUID; 16 of them are at 14001 MHz.
  - Host A's card (UUID 7b36e882) appears in six jobs: 073, 073a, 079, 083, 086 and 087.
  - 073 and 073a are one rental ("two launches on one machine" in `wsg_tables.py`).
  - 074 and 074b share a card, as do 064 and 066.
- The glob also excludes the 14001-MHz rentals of jobs 059 and 093–095.

**Correction.** Count distinct cards (or rentals) over all jobs, e.g. "on 18 of the 19 other RTX 5090 cards that
recorded one" (jobs 059–095), or say "jobs".

### 22. `tab_split` header "$f(1..8)$" heads nine-entry tables, f(0..8)

**Data.** Every row lists nine entries starting with f(0) = 0, e.g. "Law, fetch a single miss" =
`0,1,1,1,2,3,3,4,5`, where f(1) = 1.

**Correction.** Header "$f(0..8)$".

---

## Notes (true or approximately true, but worth rephrasing)

1. **Abstract "\fsbpGainMin--\fsbpGainMax [41–67]\% with the overlap it allows".** The range is the paced oracle only.
   The accounting defines the overlapped state as the faster of both2 and paced, and by that definition the range is
   42–67% (both2 gains 42.1% at Q12.5, paced 41.4%). Use one definition throughout.
2. **"At measured ceilings" (abstract; intro line 50, "over the rates the machine's own probe reached").** Only the
   host rate is measured; the GPU term is the datasheet 1,792 GB/s (§4 says so). Suggest "at the probed host rate and
   the GPU's datasheet rate".
3. **Abstract "A Shapley accounting names routing foresight the largest missing piece".** This holds at the four
   host-bound cells only. At the two GPU-bound cells overlap is largest, on host B (39 and 46% against 23%) and on O3.
   Suggest "the largest missing piece where host memory binds".
4. **Abstract "which speculative batching does not supply".** With free rejected drafts batching saves a median 15–32%
   of reads (window-equivalent about 1–4 tokens, which is W50 for C/k ≤ 2), and with costs at α = 0.9 it still saves
   8–12%. The claim holds once rejected drafts are charged at α ≤ 0.8. Suggest "does not supply once the rejected
   drafts' experts are paid for".
5. **Abstract "52 published measurements a median 14.5%".** This pools 11 i.i.d. rows that the body calls
   overestimates (median 34.8%) and 8 no-capacity rows (median 49%). The trace-scored median, which the body
   emphasises, is 9.5%.
6. **"the two tie within 3% at three budgets" (Qwen3.6) against the appendix "ours runs 1.030 [1.023, 1.038] at
   12.5% and ties at 25 and 37.5%".** At 12.5% the interval excludes 1, so it is a 3% lead.
7. **Mixtral "ties llama.cpp" at 1.015 [1.008, 1.024].** The interval excludes 1. Say "within 2%".
8. **"with the measured GPU rate it is 9% above all-in-VRAM".** gpt-oss 279/255.3 = +9.3%, Qwen3 192.1/177.8 = +8.0%.
   Say "8–9%".
9. **"the fit gives 9 and 24".** These come from the rounded constants (0.59, 1.33). The full-precision fit
   (0.589, 1.328) gives 9.3 and 23.4, i.e. 9 and 23.
10. **"two rentals of one CPU model differed by 19–22% in decode speed, 24–32% in host bandwidth".** The 24% and 32%
    are one pair (47 against 62 GB/s) seen from both directions, not a range. Say "62 against 47 GB/s".
11. **"hit-optimal … the one our first oracle job ran".** Job 095's hitopt is unpaced (`oracle_bypass=0:paced=0`).
    Job 093's main runs were paced; its unpaced variant ran at four cells. State "unpaced" in the text and the Fig. 2
    caption: the paced and unpaced versions behave differently. At O1's four host-bound cells the paced oracle hit
    66–92% and the unpaced one 75–99%.
12. **`app:foresight` "its reads within −17 to −0%".** `\fsfSimReadsDiffMax` prints "−0" for −0.2%. Use "−17 to 0%".
13. **Line 357–358 "…a per-layer latency the law … does not price, and the rest of v(F)".** The clause is garbled. It
    presumably means "and that latency is the rest of v(F)" (the 25–34% of v(F)'s gain not recovered).
14. **The Nsight split (line 389–392).** The numbers match `qwen3_profile_outcome_079.md`, and job 079 did run on
    host A (GPU UUID 7b36e882). But it profiled an earlier engine with the fixed FETCH table (0,1,1,2,3,3,4,5,6), not
    the law's table. Say so.
15. **"a 3,700-line patch".** `llama.cpp-expert-cache-4da6337.patch` adds 3,487 lines (4,196 diff lines). The oracle
    patch adds 3,860. Say "about 3,500 lines added".
16. **"Our deployed hysteresis costs 1–8% reads … (up to 22% on Qwen3)".** The 1–8% is the other eight models (0.6–8.4%).
    On Qwen3 the cost is 4.6–22.4%. Say "1–8% on eight models and up to 22% on Qwen3".
17. **Conclusion "a speed limit … that real systems reach a tenth to two fifths of".** Our cache reaches 25–43% (above
    two fifths at 43%). The trace-scored published rows span 2.3–36.8% (quartiles 4.9–17.8%), and all rows 2.3–76.9%.
    Say "from a twentieth to two fifths, with a median of a tenth".
18. **`app:foresight` "the milliseconds per token of every engine state the three jobs realised".** The table omits
    lead2, job 093's paced online run, its window sweep and unpaced oracle, and job 094's 16-token window.
19. **`app:foresight` "An unpaced copy path … still loses at gpt-oss 11%".** It also loses 28% at Qwen3 12.5%
    (0.717 [0.706, 0.727]), the fourth cell, where its hit rate was 75%. Mention it.
20. **`app:foresight` "at 16–30 admissions per token (210–285 MB)".** These are the bypass oracle's two lowest budgets
    only (G11 16.0, Q12 30.2). At G25 and Q25 it admits 8.8 and 21.3 per token. Scope the sentence.
21. **§5 "foresight is worth these reads only where the host binds; where the bus is already saturated … it is worth
    nothing".** The two halves read as contradicting each other: a saturated bus is a host-bound case. The fetch oracle
    also gains 25–26% at the two cells that are GPU-bound at the limit. Rephrase.
22. **"tightened four ways" (contribution 1; Appendix A rule 11).** Of the variants, pool loosens the limit (+0–10%)
    and median, GPU and layers tighten it. Say "changed four ways".
23. **Line 175 "(\cref{tab:law}; all of its rows are our cache on gpt-oss-120b on RTX 5090 hosts)".** The host-B row
    of `tab_law` has no cache rows, only the two llama.cpp Qwen3 errors.
24. **"the helpers read at 1.09–1.31× the probe's time per expert".** This is the desktop hosts only. On the EPYC 7352
    the factor is 0.95–1.02 and on the EPYC 9655 1.16–1.32 (`perlayer_model.json` `helper_fit`).
25. **Appendix B "106 of 207 in jobs 088 onward, where 33 of 36 sign clauses held".** The era count is strict, while
    the 33 includes 3 "held (point)"; strict is 30 of 36. Use one basis.
26. **"the equivalent of a window of two to four tokens".** At α = 0.6 the free-draft saving (14.5–17%) matches W = 1
    (13–19% in `policy_study.json`). Say "one to four".
27. **Contribution 3 "how many tokens of foresight are worth half the optimum".** W50 closes half the gap between the
    deployed online policy and the optimum. Say so.
28. **"half of the hit-optimal oracle's gain … (9.2 …), near the 10 its trace gives".** Not like for like: one is a
    speed-gain W50 of a paced, over-admitting oracle; the other is a read-count W50 of single-read Belady. At W = 2 and
    4 the engine oracle ran at 0.81× and 0.98×. Present the two numbers as different quantities.
29. **"GPU clocks cannot be locked on these rentals; we report each card's power limit".** The paper reports only a
    range, and that range is wrong (defect 15). Either tabulate per host or drop "each card's".
30. **Appendix C (iii) "no desktop host has [a floor] (their intercepts are at or below zero)".** The Ryzen 9 7900's
    prefetch configuration has an intercept of +113.5 µs (`helper_fit`, 072 C32_pf). Add "in the configurations
    without prefetch".
31. **Line 384–385 "its reads are 1.18–1.58× R*, the lead's victims and the experts used once that it leaves to the
    CPU".** Single-use experts are read once by the optimum too, so they cannot be excess reads. Explain which reads
    are excess (in-lead misses that are not admitted and recur, and the lead's evictions).
32. **"nothing, or a loss" (line 369) and "had gained nothing at the lowest ones" (conclusion).** The bytes-optimal
    oracle gained +3.3% [2.9, 3.8] at gpt-oss 11% (+5.9% with a 16-token window). Say "at most 3–6%".
33. **`tab_single_read` caption "host reads per token (CPU misses + fetches + admissions, each expert read once)".**
    The same column is given for hit-opt. Reword to "each read counted once".
34. **Line 280–281 "The ceiling is a latency-free datasheet one, which our own probed ceilings are not".** Our limit is
    also latency-free ("it drops all latencies") and uses the datasheet GPU rate; only its host rate is probed.
35. **Fig. 2 caption "Every ratio's interval is within ±0.03 (\cref{tab:single_read})".** True (maximum half-width
    0.025, paced at Q43), but the table prints no intervals for lead and paced, so the reader cannot check it there.
36. **Line 359 "fetch + prefetch runs \fsbtRatioGLow--\fsbtRatioGMid [1.39–1.49]$\times$ … on gpt-oss".** The true
    minimum is gpt-oss 40% (1.3918), not 11% (1.3944). Both print as 1.39, so the text is right, but the macro choice
    is fragile; use a Min/Max macro.
37. **"drives host memory at 71–82% of the probe's best rate at the host-bound cells" (hit-optimal).** The online
    policy runs at 66–82% at the same cells (base util 75.9/66.5/82.1/74.2%). The utilisation by itself does not
    distinguish them; the reads do. Consider giving both.
38. **"was R* over its reads times the roof's utilisation".** At host-bound cells this is an identity (frac =
    tok/s × R*·S/B_host), not a finding. Present it as a decomposition.
39. **Contribution 2 "a factorial of oracle policies … foresight with or without a second read, with or without the
    overlap it allows".** Job 095 has no serialised second-read state, and the second-read oracles are the 093/094 runs
    on other hosts (and see defect 1 for the hit-optimal one). Describe the design as it is: online, fetch, both, paced
    and hit-opt on one host.
40. **Data hygiene.** `foresight_093/094/095.json` carry `w50_trace` values (G25 9.4, Q25 3.7) that differ from the
    paper's trace W50s (10.46 and 3.86, from `foresight_S_exact.json`). The paper consistently uses the latter.

---

## What was verified and held (abridged)

**Abstract and introduction.**
- Our cache at 25–43% of the limit on both hosts; audit 52 rows from 13 sources.
- Fetch 25.4–37.9% and paced 41.4–66.8% gains.
- W50 = 0.589 (C/k)^1.328, r = 0.971 over 26 points.
- 54% of 287 (as scored; see defect 20).
- 2.0–4.0× llama.cpp; leads FreeToken at 11 of 12 cells.

**Setting.**
- Expert sizes and layers; budgets; matched expert counts within 2.4%.
- Host S's CPU 22% slower; prefill 79–83%, batch decode 77–79%.
- O1, O2 and O3 CPUs (lscpu) and probes.

**Peers.**
- Host B leads 15–29% (gpt-oss) and 3–15% (Qwen3); 80% of the combined rate at Qwen3 12.5%.
- Host S: five cells, 9–21%, 1.03 and 1.05, 0.974 [0.962, 0.987]; drops of 0.006–0.105.
- LRU 6–21% slower, trailing at four cells; 1.56–2.62× llama.cpp on the 24 GB cards.
- KL 0.0019 and 0.0005, agreement 98.5 and 99.3%.

**Law.**
- G = 4.819 and 5.063 ms.
- Blind test: 29 configurations, 7 hosts, 33 measurements, median 3.7%; desk 3.3% over 24, p90 7.5%.
- llama.cpp +3.7, +3.6, +4.0 on the Ryzen hosts; +10/+16/+119 elsewhere.
- Frozen-constant test: 24 measurements, median 7.0, p90 11.9, worst 16.2% at gpt-oss 40% on host S.
- LOHO 3.1/9.4, 4.5/11.3, 3.2/10.2; 46% of the law; G refit 5.17 against 3.92 ms.
- g = 37 and 48 µs, L_c = 20 µs; law gains 3.3–8.0% and 21–34% / 9–18% on 27–30 GB/s links.

**Limit.**
- 87.5 GB/s; 13 and 30 CPU experts; pool +0–10%.
- Re-read distance 0.52 GB; 90.3 GB/s; 1,847 against 1,792 GB/s, +3%.
- No-evict +0.5–4.0% reads; global −4.5 to −18.6%; warm 1.4%; median 80.7 GB/s, −8%.
- 52 and 61% of datasheet; 279 and 192 tok/s; per-layer sum 1–5% of a token.
- 18–40% and 8–20% of the pooled bound; all-tightening 33–57%; 0.997 and 1.047; simulated hits 1.4 and 0.6–5.3 points.
- Every cell of `tab_limit`.

**Audit.** All class statistics (29 at 9.5% with Q1–Q3 4.9–17.8% and best 36.8%; 11 at 34.8%; 52 at 14.5%; 4 at 4.0%);
24 of 29 trace rows below 20%.

**Shapley (host B).** 120 orders; foresight 37–53% of the gap and 26–31% of the token at host-bound cells; overlap
39–46% and foresight second at 23% at GPU-bound cells; policy 12–16, host 3–10, GPU 4–16%.

**Foresight measured (job 095).**
- O3 probe 52.55/43.2/59.7/61.5; limits 121–512 and 68–303; online at 25–48%.
- Fetch: 1.04–1.14× R*; 17 against 34 and R* = 15.3; hits within 1.1 points; all six ratios with intervals within
  ±0.013; 32–64% of the limit; 66–75% of v(F); law −17 to −8%.
- Both2 1.39–1.49 / 1.41–1.48; paced 1.45–1.66 / 1.41–1.67, 38–68% of the limit, 56–61% of the gap closed, and
  1.66 / 50 / 57 at the running example.
- Hit-opt 1.14 and 0.91/0.69; 1.77–2.47× R*; 71–82% utilisation.
- Oracle reads 1.73–2.21× R*; 4–5% more reads and 0–3% gain.

**Accounting measured.**
- Bytes 34–49, overlap 8–23, rest 39–43 (host-bound); running example 34/23/43; model 52–66/27–36/8–13.
- Prefetch reads 1.18–1.58× R*; law +5 to +36%, least at the lowest budgets; 19 and 46 CPU misses.
- Nsight figures; within-layer overlap under 1%; W50 9.2 between W = 4 and 16 (log interpolation reproduced).
- The GPU shortfall is about 51–53% of the remaining gap at the GPU-bound cells ("most", marginally).

**Traces.**
- 9 traces of 32,601–152,470 tokens.
- Best online 35–110%; decayed frequency best in 12 of 27 and S3-FIFO in 10 (within 1%: 20 and 18); 7 low cells
  within 4.1%.
- LRU +2.0–10.6% on 8 models; LFU 6.15×; static 1.43–13.24×.
- Drift 22–58/33–75, median 51; top quarter 31–56%; W = 4 beats every online policy in all 27 cells; 1.13×.
- Savings 26–56%, median 41%; exponent CI 1.09–1.47, LOMO 1.24–1.37, 1.51 without 6 points; W90 2.7 C/k.
- C/k ≤ 2 → W50 0.64–1.72; 10.5 and 33.6 tokens.
- Batching 15–32% (48% best model), 8–12%, −18 to −20%, union 20–79%, Belady −7 to +15%.

**`app:foresight`.**
- Host probes and law tables of O1, O2 and O3; host-bound classification.
- Fetch and lead semantics match the oracle code (`oracle_plan_fetch`, the lead's `second_use` test) and the job
  configurations.
- Hit-opt ratios on O1 and O2 and at the GPU-bound cells; W50 9.2 and 6.1 against 3.9.
- Losses of 3–8% and 0.78–0.92 at every window on O1; 82/195 against 63/160; 2.02–2.18× R*.
- Unpaced hits 98–99% at three cells; bytes-optimal gains 13–23, 0–3 (6 with W = 16); misses 30→20 and 85→64;
  1.73–2.04× R*; law within 9%.
- 1.01–1.13× bytes; 41–63%; law +37 to +53 (O2) and +24 to +39 (O1); paced online 0.997–1.005; all-CPU 8–15%.
- No-overlap +0.5 to −0.6%; W16 share 34–176% at five cells; misses 1.31–1.47× R* at host-bound cells; ideal 1.29–1.37×.
- d = 1 within 1.5–14%; fits 2.6–3.6 and 1.4; simulated admissions 1.3–2.0×.
- Clause counts 36 (9/2/21/4) and 38 (12/1/24/1); every listed 094 failure and the five 095 failure groups in
  Appendix B.

**Tables and figure.**
- Every entry of `tab_single_read`, `tab_accounting_measured`, `tab_foresight`, `tab_foresight_bytes`, `tab_headline`
  and `tab_limit`.
- All bars of `factorial.pdf` (online, fetch, both, paced and hit-opt as % of limit) and its v(F) and v(O,F) marks;
  the daggers on the four host-bound cells.
