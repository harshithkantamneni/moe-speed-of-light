# Job 083: gpt-oss-120b long outputs on hard held-out problems — predictions and outcome

The predictions are in the header of `jobs/083_gptoss_hard_long@vast.sh` (gpu branch commits b20eb88 and bee66ce,
pushed before launch). Statistics: `scripts/hard_stats.py`; numbers in `prereg/hard_083.json`.

**Setup.**
- **Host:** RTX 5090 + Ryzen 9 9950X3D, Vast offer 51051777 (listing A, job 079's machine). Offer 51046112 was rented
  out, and the change was committed before launch.
- **The law's table,** computed on the machine: gpt-oss 0,0,1,1,2 (the same as listing B).
- **Workload:** 10 held-out problems. These are problems 11–15 of AIME 2022 I/II (AI-MO/aimo-validation-aime), the
  first 10 in file order, never used in development.
  - Each is decoded for 2,048 tokens, greedy, after a held-out warm-up, in a session.
  - Budget order alternates between systems.
- **Comparison:** ours (law's split) against FreeToken's Table 1 backend: hybrid at 11% and 25%, offload at 40%.

**Result (tok/s; ours ÷ FreeToken, paired by problem, 95% CI):**

| Budget | Ours: all / 1–256 / 257–2,048 | FreeToken: all / 1–256 / 257–2,048 | Ours ÷ FreeToken, all tokens | Tokens 257–2,048 |
|---|---|---|---|---|
| 11% | 73.1 / 68.6 / 73.8 | 53.9 / 52.7 / 54.0 | **1.358 [1.338, 1.378]** | **1.366 [1.343, 1.389]** |
| 25% | 113.2 / 103.6 / 114.8 | 85.7 / 80.9 / 86.4 | **1.322 [1.303, 1.342]** | **1.328 [1.310, 1.350]** |
| 40% | 157.5 / 143.8 / 159.6 | 132.2 / 123.1 / 133.7 | **1.191 [1.172, 1.212]** | **1.194 [1.175, 1.215]** |

- **Both engines stream exactly one event per token** (1.00) on every request, so the window rates are valid for both.
- **Neither system reaches an answer:** at most 1 of 10 problems has answer text or a `\boxed` in its tail.
- **Both systems speed up after the first 256 tokens,** ours slightly more.
- **Our lead on long outputs is larger than in Table 1** (1.29 / 1.28 / 1.15 over the first 256 tokens, on listing B).
  The first-256 window does not favour us on gpt-oss either.

1. **Held.** Both systems are still reasoning at 2,048 tokens on ≥ 8 of 10 problems (ours 9/9/10, FreeToken 10/9/10).
2. **Held.** Ours runs within 15% of its Table 1 rate: +4.6 / +3.7 / +3.3%.
3. **Held.** Ours leads FreeToken at all three budgets, on all tokens and on tokens 257–2,048 (every CI above 1).
4. **Held.** The lead is ≥ 15% at 11% and 25% (+35.8%, +32.2%).

This replaces the void gpt-oss long-output runs of job 082.
