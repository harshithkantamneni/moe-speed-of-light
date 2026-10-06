# Job 097: the learned admission order in the engine — predictions and outcome

Predictions: header of `jobs/097_learned@vast.sh` (gpu branch, commit 26713ad, pushed before launch); the gate:
`prereg/learned_gate_097.md` (with its addendum on condition 1's baseline). Results on the gpu branch:
`results/097a_learned_9950x@vast` (the O4 machine of job 096, Vast offer 52267630: RTX 5090 + Ryzen 9 9950X, host
memory 50 GB/s at best, link 44) and `results/097b_learned_9950x3d@vast` (host O6, offer 48592284: RTX 5090 + Ryzen 9
9950X3D, 69 GB/s at best, link 48). Two earlier 9950X3D rentals were given 57 GB of disk instead of the 200 requested
and stopped at setup within minutes (no model ran; about $0.50). Statistics: `scripts/foresight_stats.py` ->
`prereg/foresight_097a.json`, `foresight_097b.json`; `scripts/scorecard_097.py` (paired bootstrap of learned / foa over
the 30 problems; clauses 097O4 and 097O6 in `prereg/scorecard_clauses.json`; macros `paper/wsg_learned_engine.tex`).

| Cell | O4: learned / foa [95% CI] | learned / base | O6: learned / foa [95% CI] | learned / base | fetch / base (O4, O6) |
|---|---|---|---|---|---|
| gpt-oss 11% | 1.011 [1.001, 1.020] | 1.030 | 0.942 [0.937, 0.947] | 0.936 | 1.36, 1.24 |
| gpt-oss 25% | 1.030 [1.018, 1.041] | 1.053 | 0.939 [0.933, 0.945] | 0.926 | 1.41, 1.26 |
| gpt-oss 40% | 1.045 [1.037, 1.052] | 1.075 | 0.969 [0.964, 0.974] | 0.942 | 1.34, 1.20 |
| Qwen3 12.5% | 1.018 [1.015, 1.022] | 1.032 | 0.915 [0.911, 0.919] | 0.910 | 1.44, 1.29 |
| Qwen3 25% | 1.034 [1.026, 1.040] | 1.052 | 0.905 [0.899, 0.911] | 0.896 | 1.51, 1.31 |
| Qwen3 43.75% | 1.024 [1.017, 1.030] | 1.045 | 0.933 [0.927, 0.939] | 0.912 | 1.37, 1.21 |

1. **Six model files hash to the replay's.** Held on both hosts (six MATCH each).
2. **learned reads fewer than foa at every cell; within 5 points of the replay's reduction.** Fewer at every cell on
   both hosts (6-18%). Within 5 points at four of six cells per host; Qwen3 12.5% saved less than replayed (5.6-7.8%
   against 12.0%) and Qwen3 43.75% more (17-18% against 9.0%).
3. **learned 1.01-1.18x foa at the host-bound cells, interval above 1.** Held on O4 (1.011-1.034, all intervals above
   1). **Failed on O6 at all four** (0.905-0.942).
4. **learned >= 0.99x foa at the GPU-bound cells.** Held on O4 (1.045, 1.024); **failed on O6** (0.969, 0.933).
5. **learned faster than base at the host-bound cells, recovering 10-45% of fetch's gain.** On O4 faster at all four
   (1.030-1.053), recovering 8.4, 12.8, 7.4 and 10.1% (two of four below the band). **On O6 slower at all four**
   (0.896-0.936).
6. **Host time at most 0.3 ms per token (gpt-oss) and 0.6 ms (Qwen3).** Held at 11 of 12 runs; gpt-oss 11% on O4 took
   315 us.
7. **On the O4 machine, foa / base within 0.02, fetch / base within 0.05 and both3p / base within 0.08 of job 096.**
   Held at every cell: the largest difference is 0.024 (both3p at gpt-oss 40%); fetch within 0.013, foa within 0.007.
   A second launch, in a different random order, reproduced job 096 on the same machine.

## What it says

The learned order does what it was built to do with reads (6-18% fewer than single-read decayed frequency, as the
replay predicted on average) but not with time. It saves CPU reads by fetching more misses in the step (1.1-1.6x as
many fetches), each a copy its layer waits on, and its logits cost the host 0.24-0.59 ms per token. On the O4 machine,
whose host memory is weaker (50 GB/s), that trade pays 1-5%; on O6 (69 GB/s, the class of host B) it loses 3-10%. The
online policy's own single-read form also loses slightly on O6 (0.97-0.995x base) and gains slightly on O4. A
realisable order from routing history is therefore worth a few percent at best on these engines and nothing on a
stronger host: it does not reach the third of the oracle's gain that the reviews set as the bar. The value the oracles
measure needs foresight. Not predicted: the host dependence of the sign.
