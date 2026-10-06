# Job 101: the in-step copy's loss, relaunched, and a 9800X3D behind a slower link — predictions and outcome

Predictions: header of `jobs/101_crossover@vast.sh` (gpu branch, commit 8b6c67a, pushed before launch). Results:
`results/101a_crossover@vast` (Pf of job 099 relaunched: Core Ultra 9 285K, link 26.6 / CPU 92.0 / both 93.5 GB/s)
and `results/101b_crossover@vast` (a Ryzen 7 9800X3D, link 26.6 / CPU 43.9 / both 46.6; job 099's Pb, the same CPU
model, had link 46 / CPU 44). Same engine and protocol as job 100 (patch oracle3, 20 problems), six configurations per
cell. Scoring by machine: `scripts/job101.py` (`prereg/scorecard_101.json`, `prereg/job101.json`,
`paper/wsg_job101.tex`). Spend: $1.32 of Vast credit (8.18 before, 6.86 after).

**Tally: 6 held, 19 held (point), 0 failed, 0 untested (25 clauses).**

| Host | budget | foa | aa | fetch | both3p | bypass |
|---|---|---|---|---|---|---|
| 101a (Pf again) | 11% | 0.92 | 0.42 | **0.86** [0.85, 0.87] | 1.00 | 0.93 |
| 101a (Pf again) | 25% | 0.85 | 0.52 | **0.97** | 1.20 | 1.14 |
| 101b (9800X3D, slow) | 11% | 0.99 | 0.62 | 1.14 | 1.29 | 0.98 |
| 101b (9800X3D, slow) | 25% | 0.97 | 0.69 | 1.22 | 1.47 | 1.16 |

1. **Pf again: fetch/base below 1 at 11%; every ratio within 0.03 of job 099f; base time within 3%.** Held: fetch/base
   0.86 [0.85, 0.87] (job 099f: 0.875); every ratio within 0.026 (foa at 11%); base time within 1.7%.
2. **9800X3D behind the slower link: fetch/base at 11% below Pb's 1.34 and above 1.** Held: 1.14 [1.13, 1.14].
3. **Model sign of fetch/base (frozen G, run counters) right at both cells of both hosts; probe-only aa within 12%;
   fetch under-predicted on Pf.** Held: predicted 0.92 and 0.95 on Pf (measured 0.86, 0.97), 1.36 and 1.39 on the
   9800X3D (measured 1.14, 1.22); aa within 2.2%; fetch under-predicted on Pf by 10% and 5% (and on the 9800X3D by 13%
   and 11%, not predicted).

## What it says

The only machine on which the in-step copy loses loses again on a second launch, by the same amount, and the copy
made ahead breaks even again (1.00 at 11%). The model gets the sign on both launches but over-predicts the in-step
gain where the link is slower than the CPU (here by 0.06-0.22), as it did on job 100's slow-link hosts. The same CPU
model behind a 27 GB/s link instead of 46 gains 1.14x from the in-step copy instead of 1.34x.
