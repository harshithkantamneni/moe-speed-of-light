# Job 087: the ablation ladder with a real static cache — predictions and outcome

The predictions are in the header of `jobs/087_ablation_static@vast.sh` (gpu branch commit 83dde3f, pushed before
launch). Statistics: `scripts/ablation_stats.py`; numbers in `prereg/ablation_087.json`.

**Why.** Job 082's "static cache" step loaded no experts: `LLAMA_EC_INIT` was not set, and its counters show 0 hits
(erratum in `run1_outcome_082.md`).

**Setup.**
- **Host:** listing A (RTX 5090 + Ryzen 9 9950X3D, Vast offer 51051777).
- **Workload:** 25% of experts on the GPU (C32), AIME-25 problems 0–14, 256 tokens, greedy, session.
- **Static cache, profiled:** each layer's 32 most requested experts in the models' own MATH-500 outputs (job 082's
  long runs).
- **Static cache, in hindsight:** the same, counted in the Table 1 runs on the measured problems themselves.

**Ladder (tok/s; each step ÷ the one before, paired, 95% CI; hit rate from the engine):**

| Step | gpt-oss-120b | Qwen3-30B-A3B |
|---|---|---|
| Stock llama.cpp (`-ncmoe 27` / `36`) | 39.2 | 21.8 |
| Static cache, profiled on other text | 51.3 (1.307 [1.236, 1.379]; hit 52%) | 41.1 (1.889 [1.803, 1.969]; hit 67%) |
| + LRU replacement | 66.8 (1.301 [1.229, 1.392]; hit 77%) | 36.0 (0.877 [0.835, 0.924]; hit 73%) |
| + decayed-frequency policy | 80.8 (1.211 [1.184, 1.239]; hit 78%) | 46.2 (1.282 [1.260, 1.304]; hit 74%) |
| + GPU-signalled CPU helpers | 87.3 (1.080) | 49.7 (1.076) |
| + slot maps on the GPU | 91.4 (1.047) | 50.7 (1.020) |
| + GPU-side sampling | 96.3 (1.054) | 52.0 (1.027) |
| + FETCH, fixed table | 100.0 (1.038) | 56.6 (1.089) |
| + FETCH, the law's table (ours) | **105.2** (1.051) | **61.3** (1.082) |
| Ours ÷ stock | **2.68×** | **2.81×** |

**Outside the ladder:**
- **Static cache profiled in hindsight** (hit 75% / 72%): 79.2 / 46.7 tok/s. Against the decayed-frequency step it is
  0.980 [0.953, 1.008] / 1.011 [0.982, 1.040]: the online policy matches the best static placement without a profile.
- **No expert resident** (job 082's step): 29.0 / 16.4 tok/s, 0.74 / 0.75× stock, as in job 082.

**Predictions.**
1. **Held.** The profiled static cache runs ≥ 1.3× stock on both models: 1.307× (gpt-oss, barely) and 1.889× (Qwen3).
2. **Failed on gpt-oss.**
   - The prediction was that dynamic replacement adds less than placement.
   - On gpt-oss the profile from MATH-500 text matched the AIME workload poorly (hit 52%), so the decayed-frequency
     step added more (1.58×) than placement did (1.31×).
   - It held on Qwen3 (1.12× against 1.89×).
3. **Held.** The decayed-frequency policy beats the profiled static cache on both models (CI above 1).
4. **Failed on Qwen3.**
   - The prediction was that every step after the static cache is ≥ −1%.
   - The LRU step is 0.88× the static cache on Qwen3: LRU admits every miss, and the copies cost host reads.
   - Ours ≥ 2× stock held on both models (2.68×, 2.81×).

**Reading.**
- The gain over llama.cpp comes from placing hot experts per layer, and from a replacement policy that finds the
  right placement online.
- The decayed-frequency policy matches a static placement chosen in hindsight on the measured problems.
- A static placement profiled on other text loses 20–36% against it.
- LRU is the wrong online policy here: it can be slower than a mismatched static cache.
