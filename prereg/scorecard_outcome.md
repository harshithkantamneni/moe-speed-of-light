# Prediction scorecard, jobs 073–087 — clause-level tallies under one interval rule

Every prediction committed in the headers of `jobs/073_*.sh` … `jobs/087_*.sh` on the public `gpu` branch, split
into its separable clauses and scored under one stated rule. The clauses, with the quoted header text, the quantity,
the threshold, the source of every number and a note, are hand-encoded in `prereg/scorecard_clauses.json`;
`scripts/scorecard.py` reads that file and writes `prereg/scorecard.json` (the tallies) and `paper/tab_scorecard.tex`
(the appendix table). Where a job's prereg JSON has no interval for a clause (jobs 074–077 and single-mean checks),
one was computed for this scorecard from the job's raw rows (`results/<job>/bs1.jsonl`) with the same paired
bootstrap the paper uses (`scripts/samehost_stats.boot_ratio`, 10,000 resamples, seed 0); the clause's source field
says so.

**The rule, in one sentence.** A clause is *held* only when its point estimate is on the predicted side and its 95%
paired-bootstrap interval excludes the threshold (for a band, lies entirely inside it); a clause whose point estimate
is on the predicted side but whose interval includes the threshold, or for which no interval exists, is *held
(point)*; a clause whose point estimate is on the wrong side is *failed*; *untested* means the clause could not be
evaluated; *void* means the job was voided; deterministic clauses (tables, counts of problems, whether a run
completed) have no interval and are held or failed outright.

## Tallies

86 clauses from 69 prediction entries (the paper's log counts 60: 084's four predictions are repeated unchanged on
084b and 085's five on 085b; job 084c has none of its own). Job 084's four are void (throttled card) and are scored
on 084b, as the paper's log already does; job 085's Qwen3 half is 085b.

| Era | Clauses | Held | Held (point) | Failed | Untested | Void | Held / scored |
|---|---|---|---|---|---|---|---|
| 073–075 (073, 074, 074b, 075) | 20 | 7 | 3 | 9 | 1 | 0 | 7 of 19 (10 of 19 with point) |
| 076–081 | 23 | 19 | 4 | 0 | 0 | 0 | 19 of 23 (23 of 23) |
| 082–087 (incl. 084b, 084c, 085b) | 43 | 24 | 4 | 10 | 1 | 4 | 24 of 38 (28 of 38) |
| **All** | **86** | **50** | **11** | **19** | **2** | **4** | **50 of 80 (61 of 80)** |

By clause type (held / held (point) / failed / untested / void):

| Era | Sign | Threshold | Band | Equality |
|---|---|---|---|---|
| 073–075 | 3: 2 / 0 / 1 / 0 / 0 | 11: 5 / 2 / 3 / 1 / 0 | 5: 0 / 1 / 4 / 0 / 0 | 1: 0 / 0 / 1 / 0 / 0 |
| 076–081 | 11: 11 / 0 / 0 / 0 / 0 | 8: 5 / 3 / 0 / 0 / 0 | 3: 2 / 1 / 0 / 0 / 0 | 1: 1 / 0 / 0 / 0 / 0 |
| 082–087 | 13: 8 / 0 / 4 / 1 / 0 | 14: 9 / 1 / 4 / 0 / 0 | 14: 5 / 3 / 2 / 0 / 4 | 2: 2 / 0 / 0 / 0 / 0 |

Per job (held / held (point) / failed / untested / void): 073 3/0/4/0/0 · 074 0/2/4/0/0 · 074b 2/1/0/1/0 ·
075 2/0/1/0/0 · 076 2/1/0/0/0 · 077 3/0/0/0/0 · 078 3/0/0/0/0 · 079 1/3/0/0/0 · 080 4/0/0/0/0 · 081 6/0/0/0/0 ·
082 4/0/3/1/0 · 083 5/0/0/0/0 · 084 0/0/0/0/4 · 084b 2/2/1/0/0 · 084c — · 085 5/0/1/0/0 · 085b 5/0/1/0/0 ·
086 1/1/2/0/0 · 087 2/1/2/0/0.

**The era pattern the reviews pointed at, at clause level.** Jobs 073–075 fail 9 of 19 scored clauses; jobs
076–081 fail none of 23, and 11 of those 23 are sign clauses (a direction only) whose direction had been measured on
an RTX 5090 + Ryzen 9 9950X-class host in jobs 064/067/073, while the re-used thresholds (≥ 1.8× llama.cpp in 076
and 081) had been met in 073 at 1.83× on such a host; 3 of the era's 4 "held (point)" clauses are timer or profile means with no
interval (079). Jobs 082–087, which moved to held-out prompts, other machines, a third model and an ablation with
a real static cache, fail 10 of 38 scored clauses.

## What changes against the paper's prediction log

The paper's log (`paper/app_wsg.tex`, one verdict per prediction: 073–081 27 of 33 held; 082 2 of 5 with one not
scorable; 083–087 15 held, 1 half, 5 failed) counts predictions; this scorecard counts clauses, so the denominators
differ. Clause by clause, these are the differences:

**Downgraded (logged held, scored otherwise).**
- **074b P2b** — "stock llama.cpp ≥ 3% faster [with backend sampling]": *untested*. The run without backend sampling
  failed to load (out of GPU memory; FreeToken had not released its memory), so only `llama_n22_bs` exists. The log
  scores prediction 2 as "Held: +11.7%", which is its ours-only clause (074b P2a, held).
- **085b P4** — "ours' lead over FreeToken is larger than on the headline machine at ≥ 2 of the 3 budgets" (the
  job's own restatement of 085's "≥ 4 of 6"): *failed*, 1 of 3 (12.5%: 1.067 [1.058, 1.076] > 1.032; 25%: 1.076
  [1.063, 1.089] < 1.153; 43.75%: 1.038 [1.017, 1.060] below 1.049 with the interval covering it). The log scores
  "Held: 4 of 6", which pools the three gpt-oss budgets of 085 (on a Core i9-14900K) with this one Qwen3 budget
  (on a Ryzen 9 7900). The pooled reading is recorded as a note on 085 P4 and is not scored: the prediction was
  registered for one machine. 085 P4 itself (gpt-oss, 3 of 3 larger, every interval excluding the headline value)
  is held under the same two-thirds threshold applied to the three budgets that ran there.

**Held only on the point estimate (logged held; the interval includes the threshold, or there is none).**
- **087 P1** — profiled static cache ≥ 1.3× stock: gpt-oss 1.307 [1.236, 1.379] (Qwen3 1.889 [1.803, 1.969]).
- **084b P3** — ours with every expert in its slots within 5% of stock all in VRAM: Qwen3 1.047 [1.043, 1.052], the
  upper bound crossing 1.05 (gpt-oss 0.997 [0.991, 1.004]).
- **086 P1** — FreeToken as shipped within 15% of its paper's 77–83 tok/s: 93.8 [91.3, 96.5] against a 65–96 band.
- **076 P3** — the law within 8% on own-text runs: the 12-sequence intervals are wider than the band.
- **074 P4** — half-life 64 within ±2%: 1.012 [0.997, 1.027].
- **074 P1, 074b P1, 079 P1a, 079 P1b, 079 P2** — timer or profile means with no interval (0.201 ms, 1.60 ms,
  0.266 ms, 2.7%, 1.61×).

**A half resolved (logged "Half").**
- **084b P4a** — simulated hit rate within 5 points of the engine's at every Table 1 budget: *failed*, Qwen3 12.5%
  misses by 5.3 points (the other five cells are within 2.6 points).
- **084b P4b** — ours at 20–50% of the speed limit at every budget: *held (point)*, 26–43%, no interval on the
  fraction.

**Sub-clauses that hold inside predictions the log scores as one failed unit** (the log's verdict is unchanged for the
prediction; the split makes the held part visible).
- **073 P1a, P1b** — ours+FETCH ≥ 5% ahead of FreeToken at 11% and 25%: 1.109 [1.098, 1.120], 1.084 [1.076, 1.092]
  (P1c, within ±5% at 40%, failed at 0.941 [0.933, 0.951]).
- **082 P4b** — ours ≥ 2× stock at 25% on both models: 2.658 [2.566, 2.749], 2.800 [2.743, 2.858] (P4a, every
  step ≥ −1%, failed on the step that loaded no experts).
- **087 P4b** — ours ≥ 2× stock on both models: 2.680 [2.595, 2.763], 2.815 [2.754, 2.877] (P4a failed on the
  Qwen3 LRU step, 0.877 [0.835, 0.924]).
- **085b P5** — llama.cpp at 25% within 5% of the law on Qwen3: 0.974 [0.972, 0.976] (the gpt-oss half, 085 P5,
  failed at 0.818 on the hybrid-core Intel CPU; the log scores the two halves as one failed prediction).

**Same verdict, corrected figure.**
- **073 P4** — "≥ 80% of greedy outputs identical to llama.cpp's": *failed*, as logged, but the log's 27–47% is
  agreement on the first 160 characters (`text_head`); fully identical 256-token outputs are 0–3% for ours (0 to 1
  of 30 problems) and 0% for FreeToken (`same_text_as_llama_n27` in `prereg/samehost_v2.json`). llama.cpp's own
  other placements agree with it on 40–70% of full outputs.

**Scored against the committed margin rather than the bare direction.**
- **080 P1** — "the law table beats the current table by ≥ 3% (CI lower bound > 1)": held against 1.03, not 1
  (1.048 [1.041, 1.055]).

**Unchanged.** Every other clause keeps the log's verdict: 62 clauses held or failed as logged (073 P4 and 080 P1
among them), 082 P1 untested (the log's "not scorable"), 084's four void and scored on 084b.

## Files

- `prereg/scorecard_clauses.json` — the encoding: per clause the job, prediction number, quoted clause, type
  (sign / threshold / band / equality), quantity, threshold, source (prereg JSON key, raw-row computation or results
  file), measured value, interval, status, the paper log's verdict and a note.
- `scripts/scorecard.py` — reads the encoding, checks it, writes the two outputs below; it computes no statistics.
- `prereg/scorecard.json` — tallies by job, by era, by type and by era × type; the prediction-level comparison with
  the paper's log; the flat clause list.
- `paper/tab_scorecard.tex` — the appendix longtable (job, clause, type, measured, status; booktabs, footnotesize),
  `\label{tab:scorecard}`; it is not yet input from `paper/app_wsg.tex`.
