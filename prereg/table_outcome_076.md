# Job 076: the clean comparison table (fixed build) — predictions and outcome

The predictions are in the header of `jobs/076_table_5090@vast.sh` (gpu branch commit a8ddf7a, pushed before launch).

**Setup.**
- **Host:** RTX 5090 + Ryzen 9 9950X. Host memory: CPU / link / both = 62.3 / 53.0 / 73.0 GB/s.
- **Workload and protocol:** gpt-oss-120b, 30 AIME-25 problems, 256 tokens, greedy, session. GPU-side sampling for
  every llama-server system.
- **Selection:** launch 1 picked each system's faster variant, and launches 2–3 confirm it.
  - Ours: FETCH at every budget.
  - FreeToken: hybrid at 11%, offload at 25% and 40%.

| Experts on the GPU | llama.cpp | ours v2 | FreeToken | ours ÷ FreeToken, launches 2–3 (95% CI) | ours ÷ llama.cpp |
|---|---|---|---|---|---|
| 11% | 31.5 | 60.5 | 50.2 | **1.205 [1.190, 1.220]** | 1.93× |
| 25% | 36.4 | 97.7 | 80.3 | **1.217 [1.201, 1.234]** | 2.68× |
| 40% | 43.5 | 138.7 | 124.1 | **1.118 [1.108, 1.129]** | 3.19× |

- **Measured GPU memory (GiB):**
  - ours: 9.5 / 17.5 / 25.9
  - llama.cpp: 9.6 / 17.4 / 25.3
  - FreeToken: 28.4 at every rate
- **Launch-to-launch spread:** ≤ 0.5%.
- **Unused variants in launch 1:**
  - ours without FETCH: 57.0 / 91.6 / 135.2
  - FreeToken offload at 11%: 45.5; hybrid at 25% and 40%: 79.2 / 109.8

1. **Held.** Ours is ahead at 11%, 25% and 40%, with every CI above 1.
2. **Held.** Ours ≥ 1.8× llama.cpp at every budget.
3. **Held.** The law, predicted on the machine, is within 8% on its own-text configurations: −0.9 / −0.6 / −5.0%, and
   FETCH +4.1%.
