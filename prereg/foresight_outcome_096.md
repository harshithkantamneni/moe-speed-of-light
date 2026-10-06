# Job 096: the randomised oracle factorial on two hosts — predictions and outcome

Predictions: header of `jobs/096_factorial@vast.sh` (gpu branch, commit 767e07f, pushed before launch). Results:
`results/096a_factorial_9950x@vast` (host O4) and `results/096b_factorial_9950x3d@vast` (host O5) on the gpu branch
(commit 10d2de0). Statistics: `scripts/foresight_stats.py` -> `prereg/foresight_096a.json`, `prereg/foresight_096b.json`
(paired bootstrap over the 30 problems, 10,000 resamples). Clauses: `scripts/scorecard_096.py` ->
`prereg/scorecard_clauses.json` (jobs 096O4, 096O5, 096).

## Setup

- **O4:** RTX 5090 (14,001 MHz, device read 1,695 GB/s) + Ryzen 9 9950X, 123 GB RAM. Probe: CPU 43 GB/s at 14 helpers,
  link 45, both 48; highest 50.7 GB/s.
- **O5:** RTX 5090 (14,001 MHz, 1,696 GB/s) + Ryzen 9 9950X3D, 123 GB RAM. Probe: CPU 47, **link 27**, both 50; highest
  51.1 GB/s. A slow-link host: in-step fetches cost it more than they cost O4.
- Ten configurations per cell, each in a fresh context (cold cache), in an order shuffled per cell with a recorded
  seed (`configs.txt`, `ec_*.order`). Six cells, 30 AIME-25 problems x 256 steps after a 1,024-token prefill, one launch
  per host. Both hosts rc 0, nothing skipped (O4 3.4 h, O5 4.2 h).

## Predictions

1. **fetch's reads within 6% and both2's within 15% of the corrected simulation at every cell; fetch's hit rate within
   2 points.** Held at every cell on both hosts (fetch reads within +1.5 to +3.1%, hit within 0.4 points; both2 within
   the band). No intervals (counts), so "held (point)" under the scorecard rule.
2. **fetch 1.15-1.45x base at every host-bound cell.** O5: held at all four (1.16, 1.23, 1.20, 1.26). O4: held at three;
   **failed high at Qwen3 25% (1.51)**. O4's fetch gains are larger than O3's (1.29-1.38).
3. **foa reads 3-22% fewer than base; runs 0.97-1.12x base at the host-bound cells.** Reads 4-22% fewer (O5 gpt-oss 40%
   22.3%: failed by 0.3 point). Speed: O4 1.015-1.030 at every cell; O5 0.998 / 0.974 / 1.006 / 0.999 at the host-bound
   cells and 0.95-0.97 at the GPU-bound ones. Held at the host-bound cells on both hosts.
4. **Best prefetch > fetch > foa at the four host-bound cells, each by more than the larger half-width.** Held at all
   eight host-cells.
5. **nb2 reads 1.25-1.9x lead2's; at gpt-oss 11% and Qwen3 12.5% nb2 slower than lead2 and at most 1.05x base.** Reads:
   held at 9 of 12 host-cells; failed at Qwen3 12.5% on both hosts (1.09, 1.14) and at gpt-oss 11% on O4 (1.249).
   nb2 slower than lead2 at both cells on both hosts (held). nb2 <= 1.05x base: held on O5 (0.64, 0.77) and at gpt-oss
   11% on O4 (1.049); **failed at Qwen3 12.5% on O4 (1.149)**.
6. **Qwen3 12.5%: nb2 faster than hitopt; gpt-oss: nb2 within 10% of hitopt.** Held everywhere: at gpt-oss nb2 is
   1.01-1.03x hitopt on both hosts (the second read costs nothing there); at Qwen3 12.5% 1.41x (O4) and 1.54x (O5).
7. **bypass gains at most 8% at the two lowest budgets; lead2 faster than bypass at every host-bound cell.** Held
   everywhere (bypass 0.94-1.02x at the two lowest budgets; lead2 ahead by 1.02-1.35x, by 0.4% at Qwen3 25% on O5).
8. **hitoptp faster than hitopt at the two 25% cells.** Held on both hosts.
9. **Bytes share (base -> fetch) 25-55% of the gap; overlap share (fetch -> best prefetch) 5-30%.** Held at seven of
   eight host-cells; **bytes failed at Qwen3 12.5% on O4 (60%)**. Overlap 6-21%.
10. **fetch/base differs by at most 0.15 between the hosts.** **Failed at five of six cells (0.13-0.26).** The hosts
    differ in their link (45 vs 27 GB/s): fetching admissions in the step pays less where the link is slow.

Tally: O4 66 clauses (13 held, 48 held on the point, 5 failed); O5 66 (16, 48, 2); between hosts 6 (0, 1, 5).

## What was not predicted, and what it changes

- **Reading the online policy's admissions once is worth almost nothing in time (0.97-1.03x),** although it removes
  4-22% of its host reads. Its admissions are rare (3.5-6.8 per token, rationed by kappa) and their background copies
  overlap later steps. The same second read is expensive for MIN's admissions: at the host-bound cells MIN with bypass admits 2-11 times
  as often,
  and served-then-copied (bypass) it gains 0.94-1.18x where fetched in the step it gains 1.16-1.51x. The thesis
  sentence stands for foresight; for the online policy the second read is not the cost.
- **Over-admission, not the second read, is what made the Belady prefetch look worthless at gpt-oss:** nb2 (Belady, one
  read) and hitopt (Belady, two reads) are within 1-3% of each other at every gpt-oss cell on both hosts, both far
  below lead2 (MIN-with-bypass prefetch). At Qwen3 12.5%, where admissions are most frequent (119-152 per token), the
  second read costs too (nb2 1.41-1.54x hitopt).
- **Host dependence is large** (prediction 10): O5's slow link makes every in-step fetch dearer; fetch 1.16-1.26x
  there against 1.36-1.51x on O4. The ordering of the states is the same on both hosts at every host-bound cell.
- The model's accounting, now measured on O3, O4 and O5 at the host-bound cells: bytes 26-60% of the gap (of which
  single reads of the online policy's admissions -4 to +4%, foresight 26-57%), overlap 7-23%, the rest 30-62%; the
  model over-prices bytes 1.1-2.1x and overlap 1.1-6.4x and under-prices the rest 2.1-8.4x
  (`prereg/accounting_measured.json`).
- Gate condition 3 for job 097 (`prereg/learned_gate_097.md`): foa / base lower bound >= 0.98 at four of four host-bound
  cells on O4 and three of four on O5 (gpt-oss 25% on O5: 0.974 [0.971, 0.978]). Passed; job 097 launched.
