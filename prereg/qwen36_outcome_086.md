# Job 086: FreeToken's own headline model, Qwen3.6-35B-A3B BF16 — predictions and outcome

The predictions are in the header of `jobs/086_qwen36@vast.sh` (gpu branch commit a065076, pushed before launch).
Statistics: `scripts/samemachine_stats.py --job 086`; numbers in `prereg/qwen36_086.json`.

**Setup.**
- **Host:** RTX 5090 + Ryzen 9 9950X3D, listing A (Vast offer 51051777; jobs 079/083).
- **Model:** Qwen/Qwen3.6-35B-A3B in BF16. It has 40 layers of 256 experts, top-8, 6.29 MB experts, and hybrid
  linear/full attention.
  - llama.cpp and ours use a GGUF converted on the machine without the vision tower or the MTP head.
  - FreeToken reads the HF checkpoint with `--text-model-only`.
- **Workload:** Table 1 protocol (30 AIME-25 problems, 256 tokens, greedy, held-out warm-up, session).
- **The law's table,** computed on the machine with g = 32 µs frozen in the header: 0,0,1,1,2,3,3,4,4. The fixed
  table is 0,1,1,2,3,3,4,5,6.

**Result (tok/s; ours = the law's table, FreeToken = hybrid, both picked on launch 1; launch 2 unless marked):**

| Experts on GPU | llama.cpp (L1) | FreeToken hybrid | Ours | Ours ÷ FreeToken (95% CI) | Ours ÷ llama.cpp | Law ÷ fixed (L1) |
|---|---|---|---|---|---|---|
| 12.5% (C32, -ncmoe 35) | 30.2 | 57.9 | **59.7** | **1.030 [1.023, 1.038]** | 1.98× | 1.071 [1.068, 1.074] |
| 25% (C64, -ncmoe 30) | 34.4 | 78.7 | **79.5** | 1.010 [0.998, 1.021] | 2.31× | 1.060 [1.057, 1.063] |
| 37.5% (C96, -ncmoe 25) | 39.8 | 98.4 | **98.7** | 1.003 [0.994, 1.013] | 2.48× | 1.057 [1.052, 1.061] |

- **FreeToken's offload backend,** launch 1: 46.8 / 68.3 / 94.6.
- **FreeToken as shipped** (automatic backend and cache size): 93.8 tok/s at 28.9 GiB.
- **Measured VRAM:**
  - ours 12.6 / 20.1 / 27.7 GiB;
  - llama.cpp the same;
  - FreeToken 28.7–30.2 GiB at every rate.
- **Our hit rates** 0.608 / 0.763 / 0.852; misses per token 125 / 76 / 47.

**Predictions.**
1. **Held.** FreeToken as shipped runs at 93.8 tok/s, within 15% of its paper's 77–83. It is at the top of the band,
   so our setup does not handicap it.
2. **Failed.** Ours leads FreeToken at 12.5% (+3.0%, CI above 1). At 25% and 37.5% the two tie (+1.0% and +0.3%, CIs
   include 1).
3. **Failed.** Ours is 1.98× llama.cpp at 12.5%, just below the 2× predicted; 2.31× and 2.48× at the others.
4. **Held.** The law's table beats the fixed table at every budget (+5.7% to +7.1%, every CI above 1).

**Reading.**
- On FreeToken's own headline model, a hybrid-attention MoE with small experts and 256 per layer, our cache ties
  FreeToken, or leads it by 3% at the smallest budget.
- The lead on gpt-oss-120b (15–36%) and Qwen3-30B-A3B (3–15%) does not transfer to this model.
- It has 125 misses per token at 12.5%, three times gpt-oss's count, of experts half the size, so per-miss costs weigh
  more.
