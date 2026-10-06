# Job 098: FreeToken tuned per cell — predictions and outcome

Predictions: header of `jobs/098_freetoken_tuned@vast.sh` (gpu branch, commit 73bf256, pushed before launch). Attempt 1
stopped at setup (CUDA 12.8 image; `results/098_freetoken_tuned@vast_attempt1`); attempt 2 ran on the CUDA 13.0.3 image,
Vast offer 52711021 (RTX 5090 at 14,001 MHz + Ryzen 9 9950X, 255 GB RAM), rc 0 in 2.9 h. Results:
`results/098_freetoken_tuned@vast` (gpu branch, commit 685823b). Statistics: `scripts/freetoken_tuned.py` ->
`prereg/freetoken_tuned_098.json` (paired by problem, 10,000 bootstrap resamples); clauses under job 098 in
`prereg/scorecard_clauses.json`.

| Cell | ours (tok/s) | FreeToken's best setting | its tok/s | ours / best [95% CI] | best / Table 1 setting | host S (job 089) |
|---|---|---|---|---|---|---|
| gpt-oss 11% | 59.3 | hybrid, calibrated split, 8 threads | 50.6 | 1.173 [1.157, 1.187] | 1.042 | 1.207 |
| gpt-oss 25% | 93.4 | offload | 80.0 | 1.167 [1.147, 1.189] | 1.030 | 1.196 |
| gpt-oss 40% | 133.6 | offload | 122.1 | 1.094 [1.078, 1.112] | 1.000 | 1.093 |
| Qwen3 12.5% | 34.3 | hybrid, calibrated split, 8 threads | 33.6 | 1.020 [1.012, 1.028] | 1.011 | 1.027 |
| Qwen3 25% | 55.0 | hybrid, calibrated split, 8 threads | 52.7 | 1.044 [1.032, 1.056] | 1.014 | 1.048 |
| Qwen3 43.75% | 96.3 | offload | 98.0 | 0.983 [0.970, 0.996] | 1.000 | 0.974 |

1. **FreeToken's best at most 1.10x its Table 1 setting.** Held at every cell (1.00-1.04).
2. **Ours leads FreeToken's best by >= 5% at gpt-oss 11/25%; within 0.95-1.15 at gpt-oss 40%, 0.90-1.12 on Qwen3.**
   Held at every cell. Tuning FreeToken changes the comparison by at most 4% at any cell; ours leads at five of six
   cells and FreeToken at Qwen3 43.75%, as on host S.
3. **A fixed fetch cap of 1 or 2 within 8% of the calibrated split.** Held at 10 of 12; at Qwen3 12.5% (cap 1, -9.4%)
   and 25% (cap 2, -8.7%) the fixed caps are slower than FreeToken's calibrated split: its own split is its best.
4. **8 threads slower than the default or within 3%.** Held at five of six; at gpt-oss 11% 8 threads are 4.2% faster.
   Eight threads were FreeToken's best hybrid setting at four cells (by 0.2-4%).
5. **Offload best at the two highest budgets, hybrid at the other four.** Held at five; at gpt-oss 25% offload (80.0)
   beat hybrid (77.7-79.0).
6. **FreeToken 27-30 GiB at every run; ours less.** FreeToken 28.4 GiB (gpt-oss) and 28.6-30.1 (Qwen3); ours 9.5-27.3
   GiB, below FreeToken at every cell. Failed only on the 30 GiB bound at Qwen3 43.75% (30.1).
7. **CPU-only: llama.cpp >= 2x FreeToken's cpu backend on gpt-oss and >= 1.2x on Qwen3.** **Failed:** 1.13x and 0.99x.
   FreeToken's CPU executor is close to llama.cpp's on this host; the 13.7 tok/s of job 088 was a hybrid-core host's
   eight performance cores. Kernel quality on the CPU does not explain our lead.

Tally: 44 clauses, 22 held, 15 held on the point, 7 failed.
