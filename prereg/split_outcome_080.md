# Job 080: a per-host FETCH split from the host-DRAM law — predictions and outcome

The predictions are in the header of `jobs/080_fetch_split@vast.sh` (gpu branch commit b2ef1a1, pushed before
launch). Statistics: `scripts/split_stats.py`; numbers in `prereg/split_080.json`.

**Setup.**
- **Host:** RTX 5090 + Ryzen 9 9950X3D, a different offer from job 079 with the same CPU and GPU.
- **Host bandwidth** (its own concur probe): CPU 70.9, PCIe zero-copy 53.2, both at once 80.7 GB/s.
- **Workload and protocol:** Qwen3-30B-A3B BF16, 30 AIME-25 problems, 256 tokens, greedy, session.
- **The rule** (`fetch_table.py`): for a layer with n missed experts, fetch the f that minimises
  max(f·S/B_p + g, L_c + (n−f)·S/B_c, n·S/B_both), with g = 48 µs and L_c = 20 µs frozen.
  - It was computed on the machine before any model run.
  - Result: **0,0,1,1,2,3,3,4,5**. The current fixed table is 0,1,1,2,3,3,4,5,6.
  - The one qualitative change: a single missed expert runs on the CPU instead of being fetched.

**43.75% of experts on the GPU, launch 1 (tok/s; paired ratio to the current table):**

| Table | tok/s | vs current |
|---|---|---|
| lite 0,0,1,1,1,2,2,3,3 | 109.0 | 1.057 [1.052, 1.062] |
| **law 0,0,1,1,2,3,3,4,5** | **108.0** | **1.048 [1.041, 1.055]** |
| law with f(1) = 1 | 104.1 | 1.010 [1.007, 1.012] |
| current 0,1,1,2,3,3,4,5,6 | 103.1 | 1 |
| none (CPU only) | 100.7 | 0.977 [0.970, 0.984] |
| all (fetch every miss) | 98.7 | 0.957 [0.953, 0.962] |
| FreeToken offload / hybrid | 103.2 / 96.5 | |

**Launch 2 (confirmation of the launch-1 picks):**
- lite 108.2, law 108.2, current 103.2, FreeToken offload 103.1.
- Ours (law or lite) ÷ FreeToken = **1.049 [1.035, 1.064]**; current ÷ FreeToken = 1.000 [0.987, 1.014].
- law ÷ current = 1.049 [1.043, 1.056].

**25%, launch 1:**
- law 63.1, current 59.2, none 54.3; FreeToken hybrid 54.8, offload 48.7.
- law ÷ current = **1.067 [1.062, 1.071]**; ours (law) ÷ FreeToken's better backend = **1.153 [1.134, 1.173]**.

1. **Held.** At 43.75% the law table beats the current table by 4.8% [4.1, 5.5] (launch 1) and 4.9% (launch 2).
2. **Held.** The law table is within 0.9% of the sweep's best in launch 1 (lite, 0.991 [0.986, 0.996]). In launch 2 the
   two are equal (108.2 each).
3. **Held.** With the best table, ours leads FreeToken's better backend at 43.75% by 4.9% [3.5, 6.4] on the
   confirmation launch. With the current table it was a tie.
4. **Held.** At 25% the law table beats the current table by 6.7% [6.2, 7.1].

**What decides it: one missed expert goes to the CPU.**
- law vs law-with-f(1)=1 differ only in that entry: 108.0 vs 104.1 (+3.7%).
- A layer with exactly one miss is 24% of layer steps at 43.75% (job 079 histogram). There, running the expert on the
  CPU (~155 µs, hidden behind the GPU's other expert products) beats copying 9.4 MB over PCIe and then computing
  (~235 µs, on the critical path).
- The cost is visible: misses rise from 29.8 to 32.7 per token, because an expert run on the CPU is not admitted at
  once. The time saved outweighs it.

**How well the per-layer model tracks the sweep.**
- Each run's modelled MoE time uses its own miss histogram, so the table's effect on the cache is included.
- Measured token time = 6.12 ms + 0.56 × modelled MoE time, with r = 0.958 over nine runs.
- The model ranks the tables correctly apart from law vs lite, which are within 1% of each other.
- It overstates the differences between tables by about 1.8×. The frozen g and L_c are rough, and the model treats
  the CPU and PCIe paths as independent within a layer.
- Fetching everything (FreeToken offload's strategy in our engine) is the worst table here. Running everything on the
  CPU is second worst.
