# Job 075: 40% and 60% of experts on an RTX PRO 6000 — predictions and outcome

The predictions are in the header of `jobs/075_pro6000_40_60@vast.sh` (gpu branch commit 0fc6622, pushed before
launch).

**Setup.**
- **Host:** RTX PRO 6000 Blackwell (96 GB) + Ryzen 9 9950X. Host memory: CPU / link / both = 62.3 / 53.0 / 71.3 GB/s.
- **Workload and protocol:** gpt-oss-120b, 30 AIME-25 problems, 256 tokens, greedy, session.
- **Sampling:** GPU-side sampling for every llama-server system.
- **Selection:** launch 1 picked each system's faster variant, and launches 2–3 confirm it.
  - Ours: FETCH at 40%, no FETCH at 60%.
  - FreeToken: offload at both budgets.

| Experts on the GPU | llama.cpp | ours v2 | FreeToken | ours ÷ FreeToken, launches 2–3 (95% CI) |
|---|---|---|---|---|
| 40% (51 per layer) | 43.7 | 137.5 (FETCH) | 122.6 (offload) | **1.122 [1.107, 1.137]** |
| 60% (77 per layer) | 63.0 | 189.7 | 169.7 (offload) | **1.118 [1.109, 1.127]** |
| 100% (all in VRAM) | 261.4 | — | — | — |

- **Measured GPU memory (GiB):**
  - ours: 26.0 / 37.5
  - llama.cpp: 25.4 / 38.0, and 59.6 with everything in VRAM
  - FreeToken: 85.6 (it reserves 90% of the card)
- **Launch-to-launch spread:** under 0.5%.
- **Unused variants in launch 1:** ours without FETCH at 40%, 132.4; with FETCH at 60%, 187.9. FreeToken hybrid,
  111.5 / 145.7.

1. **Held.** 40%: ours ahead, CI above 1: +12.2%.
2. **Failed, on the favourable side.** 60%: "within −3% to +10%" — it is +11.8% [+10.9, +12.7].
3. **Held.** At 60%, ours reaches 72.7% [71.8, 73.6] of llama.cpp's all-in-VRAM speed (≥ 55% predicted). FreeToken
   reaches 64.9%.

**vs llama.cpp at equal expert memory:** 3.16× at 40%, 3.01× at 60%.
