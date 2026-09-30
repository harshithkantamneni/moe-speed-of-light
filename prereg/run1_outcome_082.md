# Job 082: the review's run 1 — predictions and outcome

The predictions are in the header of `jobs/082_review_run1@vast.sh` (gpu branch commit cb97196, pushed before launch).
Statistics: `scripts/run1_stats.py`; numbers in `prereg/run1_082.json`.

**Host:** the headline machine (RTX 5090 + Ryzen 9 9950X3D, Vast offer 51046112, as jobs 080/081).

**Law tables** (computed on the machine, as in 081):
- gpt-oss 0,0,1,1,2
- Qwen3 0,0,1,1,2,3,3,4,5

## A. Steady state on held-out prompts (10 MATH-500 problems × 2,048 tokens)

| Model, budget | ours ÷ FreeToken, all tokens | tokens 1–256 | tokens 257–2048 |
|---|---|---|---|
| Qwen3 12.5% | 1.041 [1.021, 1.063] | 1.023 | **1.043 [1.020, 1.070]** |
| Qwen3 25% | 1.052 [1.033, 1.072] | 1.028 | **1.056 [1.036, 1.077]** |
| Qwen3 43.75% | 1.048 [1.019, 1.080] | 1.011 | **1.054 [1.025, 1.086]** |

- **Qwen3: our lead does not shrink on long outputs; it grows.**
  - Both systems speed up after the first 256 tokens, and ours more so:
    - 12.5%: 1.038 vs 1.017;
    - 25%: 1.094 vs 1.065;
    - 43.75%: 1.117 vs 1.071.
  - Speeds (tok/s, ours / FreeToken): 40.0 / 38.4, 65.1 / 61.9, 109.9 / 104.9.
- **Turning off FreeToken's prefill overlap changes nothing measurable.**
  - Its overall speed: 143.5 vs 144.0 tok/s (gpt-oss 40%) and 103.9 vs 104.9 (Qwen3 43.75%).
  - Its first-256 rate at Qwen3 43.75%: 92.8 vs 98.9.
  - So lending slots to prefill does not hold FreeToken back, and the review's main threat (the first-256 window
    favours us) is refuted where it can be measured.
- **The gpt-oss long runs cannot be compared.**
  - Easy MATH-500 problems are answered well inside 2,048 tokens. After the answer, the client's `ignore_eos` forces
    each engine to keep going on text of its own choosing.
  - Ours then speeds up 1.6–2.1×:
    - its hit rate at C14 rises from 58% (Table 1) to 77%;
    - on the easy problems it reaches 224 tok/s at 11%, near the all-in-VRAM speed;
    - it does not speed up on the one hard problem (p9: 66 tok/s).
  - FreeToken streams far fewer events than tokens there (as few as 431 for 2,048), so its window rates can't be
    measured either.
  - We report neither the gpt-oss steady-state ratios (1.36–2.22×) nor anything derived from them.
  - The fix is a rerun on hard held-out problems (AIME 2022–2024), which are still mid-reasoning at 2,048 tokens.

## B. Ablation ladder at 25% (AIME-25 problems 0–14, 256 tokens; tok/s, each step's paired ratio to the previous one)

| Step | gpt-oss-120b | Qwen3-30B-A3B |
|---|---|---|
| Stock llama.cpp (`-ncmoe 27` / `36`) | 40.5 | 22.0 |
| Static cache (fixed experts per layer, host-driven CPU path, CPU sampling) | 29.3 (0.72) | 16.4 (0.74) |
| + LRU replacement | 67.7 (2.31) | 36.2 (2.21) |
| + decayed-frequency policy | 81.2 (1.20) | 46.1 (1.27) |
| + GPU-signalled CPU helpers (mailbox) | 87.7 (1.08) | 49.7 (1.08) |
| + slot maps on the GPU | 92.6 (1.06) | 50.5 (1.02) |
| + GPU-side sampling | 100.2 (1.08) | 52.3 (1.04) |
| + FETCH, fixed table | 103.6 (1.04) | 57.6 (1.10) |
| + FETCH, law's table (= ours) | **107.6 (1.04)** | **61.7 (1.07)** |
| Ours ÷ stock | **2.66×** | **2.80×** |

Every step after the static cache helps, and every interval is above 1. Two things drive most of the gain:
- **Dynamic caching itself:** 2.2–2.3× over the static cache.
- **The decayed-frequency policy:** +20–27%.

The static cache is slower than stock llama.cpp, because stock pins whole layers and runs them on the CPU without
per-layer hand-offs.

## C. Teacher-forced parity (12 × 128 steps after 640 prefilled tokens)

| | top-1 agreement with stock | NLL change | stock's own second placement: agreement / NLL change |
|---|---|---|---|
| gpt-oss-120b (own-text corpus; ref `-ncmoe 27`) | 98.6% | −0.22% | 99.6% / −0.38% |
| Qwen3-30B-A3B (MATH-500 text; ref `-ncmoe 36`) | 99.3% | +0.31% | 99.9% / +0.02% |

- Ours disagrees with stock on the top token 3.5–5× as often as stock disagrees with itself across placements.
- The NLL does not move beyond stock's own spread on gpt-oss, and it moves +0.31% on Qwen3.
- This is numerical difference, not quality loss. It explains the early divergence of free-running greedy text.

## D. The missing Table 1 cells

- llama.cpp on Qwen3 at 25%: **22.09 tok/s**, against 22.87 predicted (+3.6%).
- llama.cpp on Qwen3 at 43.75%: **28.39 tok/s**, against 29.53 predicted (+4.0%).
- The predictions were written on the machine before the runs.
- With job 080's speeds on the same machine, ours is 2.86× (25%) and 3.81× (43.75%) llama.cpp.

## Predictions

1. **Not scorable.** The gpt-oss steady-state lead "held" (CI > 1 at all three budgets), but the comparison is invalid
   (section A).
2. **Failed.** FreeToken did not gain more than ours after the first 256 tokens. On Qwen3 ours gained more at every
   budget, and gpt-oss can't be scored.
3. **Failed, favourably.** At Qwen3 43.75% ours leads FreeToken in steady state, by 5.4% [2.5, 8.6].
4. **Failed.** The static-cache step is slower than stock (0.72× / 0.74×). Every later step is positive, and ours is
   2.66× / 2.80× stock.
5. **Held.** Top-1 agreement is 98.6% / 99.3% and |ΔNLL| is 0.22% / 0.31%.
6. **Held.** The law is within 3.6% and 4.0% on both missing cells.

## Erratum (found by an independent check of the paper, 30 September)

**The "static cache" step of section B loaded no experts.**
- The static policy takes its resident set from `LLAMA_EC_INIT`, which the job did not set.
- Its counters show 0 hits in both models (`srv_{g,q}_abl_1_static.json`), so every expert ran on the CPU through the
  cache's host-driven path.
- The step measures "no expert resident", not a static cache.
- The conclusions drawn from it are withdrawn: "dynamic caching gives 2.2–2.3× over a static cache" and "the static
  cache is slower than stock".

**Prediction 4 stays failed as scored,** but not for the reason given.

**Trace simulation says per-layer placement carries most of the gain.** On the AIME-25 traces of job 084, a hindsight
static cache hits 75.0% / 71.9% of expert uses, against 77.5% / 73.7% for LRU (gpt-oss / Qwen3).

**Job 087 reruns the ladder** with a real static cache: one profiled on other text, and one in hindsight.
