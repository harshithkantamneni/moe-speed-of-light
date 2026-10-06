# Job 099: the host panel — predictions and outcome

Predictions: header of `jobs/099_panel@vast.sh` (gpu branch, commit 7b8327a, pushed before the first launch); the ten
wrappers `jobs/099a..j_panel@vast.sh` (QWEN=1 for a, b and c). Results on the gpu branch: `results/099{a..j}_panel@vast`
(commits 7b8327a, fe5f47b, 1e4e0f3, ae4b50e). Each host: one RTX 5090 with a desktop CPU and at least 90 GB of RAM,
rented on Vast, one launch, configurations in fresh contexts in a seeded shuffled order. Scoring is by machine:
`scripts/panel_099.py` (clauses `prereg/scorecard_099.json`, statistics `prereg/panel_099.json`, macros
`paper/wsg_panel.tex`), under the interval rule of the scorecard (held if the whole 95% interval satisfies the clause,
held (point) if only the point does or no interval exists, failed otherwise).

**Tally: 190 held, 42 held (point) (40 with no interval, 2 whose interval crosses), 39 failed, 2 untested.**

## The hosts and the main ratios (gpt-oss-120b; speed relative to the deployed state; shares of aa -> MIN in time)

| Host | CPU | link / CPU / both (GB/s) | 11%: foa | aa/foa | fetch | both3p | w4 share (reads .52) | w16 (.98) | 25%: fetch | both3p | w16 (.66) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Pa | Ultra 9 285K | 49 / 60 / 61 | 1.00 | 0.90 | 1.33 | 1.51 | 0.43 | 0.89 | 1.33 | 1.69 | 0.57 |
| Pb | Ryzen 7 9800X3D | 46 / 44 / 50 | 1.00 | 0.95 | 1.34 | 1.46 | 0.47 | 0.93 | 1.40 | 1.71 | 0.62 |
| Pc | Ryzen 9 7900 | 27 / 40 / 46 | 1.00 | 0.66 | 1.17 | 1.33 | 0.36 | 0.84 | 1.24 | 1.54 | 0.54 |
| Pd | i9-13900KF | 28 / 71 / 70 | 0.98 | 0.57 | 1.05 | 1.11 | 0.34 | 0.80 | 1.12 | 1.31 | 0.52 |
| Pe | Ryzen 9 9950X | 46 / 46 / 50 | 1.01 | 0.97 | 1.35 | 1.46 | 0.47 | 0.93 | 1.40 | 1.71 | 0.63 |
| Pf | Ultra 9 285K | 27 / 93 / 93 | 0.95 | 0.45 | 0.87 | 1.01 | 0.32 | 0.78 | 0.97 | 1.20 | 0.49 |
| Pg | Ryzen 9 5950X | 26 / 40 / 39 | 1.01 | 0.72 | 1.25 | 1.38 | 0.38 | 0.83 | 1.32 | 1.63 | 0.54 |
| Ph | Ryzen 9 7950X | 49 / 51 / 63 | 0.99 | 0.85 | 1.25 | 1.42 | 0.43 | 0.89 | 1.29 | 1.63 | 0.59 |
| Pi | i7-14700K | 19 / 25 / 30 | 1.02 | 0.77 | 1.34 | 1.31 | 0.39 | 0.84 | 1.44 | 1.58 | 0.56 |
| Pj | Ryzen 9 9950X | 46 / 52 / 62 | 1.00 | 0.88 | 1.27 | 1.43 | 0.44 | 0.90 | 1.31 | 1.66 | 0.61 |

Pooled over hosts (two-stage bootstrap, hosts then problems): fetch / base 1.22 [1.12, 1.30] at 11% and 1.28
[1.19, 1.36] at 25%; both3p / base 1.34 [1.24, 1.43] and 1.57 [1.46, 1.65]; Qwen3 12.5% (three hosts) fetch / base
1.36 [1.21, 1.44].

## Predictions

1. **Engine reads within 4% of this machine's replay.** Held (point) at 18 of 20 host-cells: 1.6-1.7% at every one
   (no interval; the engine and the replay are deterministic). **Untested on Pd**: the replay step failed because
   `numba` did not install on that host (`stdout.log`), so no replay file was written. Pd's engine reads per token
   equal Pa's to 0.01 at every configuration (both hosts produced the same routing), and Pa's were within 1.6%.
2. **Time share tracks read share.** (a) |time share - read share| <= 0.15 at 94% of hybrid host-cells (threshold
   80%): held (point). (b) w1 < w4 < w16 at every host-cell: held (point). The 6% outside 0.15 are w4 and w16 at 11% on
   the three hosts with the slowest link relative to the CPU (see below).
3. **Exact foresight shares** (w16 >= 0.80 and w4 0.35-0.70 at C14; w16 0.45-0.85, w4 0.12-0.40 at C32; w1 <= 0.30).
   Held at 56 of 60, held (point) at 1; **failed at 3**: w4 at 11% on Pd (0.34) and Pf (0.32), w16 at 11% on Pf
   (0.78).
4. **Degraded foresight shares** (w8r5 0.15-0.45 at C14, <= 0.30 at C32; allr5 0.30-0.60). Held at all 40.
5. **The 2 x 2.** (a) MIN's set worth more with one read than two: held at 19 of 20; **failed on Pf at 11%** (-0.48 ms:
   on the most link-starved host, one read in the step is worse than two). (b) Reading the online policy's admissions
   once changes time by at most 5%: held at 17 of 20; **failed on Pd at 25% (0.92) and on Pf at both cells (0.95,
   0.85)**.
6. **fetch / base 1.10-1.55, both3p / base 1.25-1.90, both3p > fetch > foa.** Held at 34, held (point) at 18;
   **failed at 8**: fetch and both3p below the band on Pd at 11% (1.05, 1.11) and on Pf at both cells (0.87, 1.01;
   0.97, 1.20); the ordering on Pf at 11% (fetch is slower than foa) and on Pi at 11% (both3p 1.31 < fetch 1.34).
7. **aa / foa in [0.92, 1.04] on links >= 40 GB/s; below 0.95 on links < 32 GB/s.** The slow-link half held at all 12
   (0.45-0.82). The fast-link half held at 3 and held (point) at 1 of 8; **failed at 4**: Pa 11% (0.90), Ph 11% (0.85)
   and 25% (0.90), Pj 11% (0.88). On a 46-49 GB/s link, reading every miss over the link in the step still costs
   up to 15% where the CPU is at least as fast as the link.
8. **Machine variance dominates.** The spread of fetch / base across hosts is 56x (11%) and 37x (25%) the median
   within-host 95% half-width (threshold 3x): held (point) at both.
9. **Host plan at most 150 us per step.** **Failed at all 20 host-cells**, through one configuration: the whole-future
   window at 50% recall (allr5) plans in 193-277 us per step, because every one of its steps scans the rest of the
   problem. Every other configuration stays below 150 us at gpt-oss (exact windows 8-30 us, fetch 46-101 us); at
   Qwen3 the MIN plan takes 117-151 us.
10. **Qwen3 C16: w2 share 0.30-0.70, w8r5 0.25-0.60, fetch / base 1.15-1.55.** Held at all 9 (three hosts).

## What it says

The engine reproduces the replay to within 2% of reads on every host where the replay ran; differences in time
therefore come from where the reads go. Exact windows recover more as W grows on every host; the paced single read
is faster than the deployed cache on every host-cell but one (1.11-1.71), and matches it on Pf at 11% (1.01, interval
0.98-1.06); the 2 x 2's positive interaction holds on 19 of 20 host-cells. The failures concentrate on the hosts whose link is slow relative to their CPU (Pf 0.29, Pd 0.40): there,
any change that moves reads from the CPU to the link loses (foa, aa, fetch), and the time share of a window falls
below its read share, because a short window still reads 97% of its misses over the link in the step where MIN reads
50%. The calibrated time model of `scripts/hostdep_model.py` predicts this from each state's counters: in-step states
within 2.2% (median) of measured time, the sign of fetch / base on 43 of 43 host-cells, and the exact windows' time
shares within 0.03 (median; at most 0.10). The pre-registered bands did not encode host dependence; they were set from
the four hosts of jobs 095-097, all with link/CPU of 0.6-1.0.

Two outcomes we did not predict: on fast-link hosts with a CPU at least as fast as the link (Pa, Ph, Pj), admitting
every miss costs 10-15% at 11%; and the model over-predicts the whole-future noisy window's time share by up to 0.24 on
fast-link hosts, which the plan cost (1-4% of the aa -> MIN gap) does not explain.
