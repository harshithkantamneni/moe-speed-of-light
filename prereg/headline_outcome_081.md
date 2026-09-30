# Job 081: the headline table with the law's per-host FETCH split — predictions and outcome

The predictions are in the header of `jobs/081_headline_law@vast.sh` (gpu branch commits 2f1ac92 and 79b9cd8, pushed
before launch). Statistics: `scripts/headline_stats.py`; numbers in `prereg/headline_081.json`.

**Setup.**
- **Host:** RTX 5090 + Ryzen 9 9950X3D (Vast offer 51046112, the same listing as job 080).
- **Host bandwidth** (its own probe): CPU 71.6, PCIe zero-copy 53.2, both at once 77.7 GB/s. FreeToken's probe gives
  CPU 69.8, PCIe 54.4 GB/s.
- **The law's tables,** computed on the machine before any model run:
  - gpt-oss-120b 0,0,1,1,2 (current 0,1,1,2,3);
  - Qwen3 0,0,1,1,2,3,3,4,5 (current 0,1,1,2,3,3,4,5,6).
- **Protocol:** 30 AIME-25 problems, 256 tokens, greedy, held-out warm-up, session, GPU-side sampling for llama-server.
  - Launch 1 runs every variant. Each system's faster variant is picked on launch 1 and rerun in launch 2.
  - Headline comparisons use launch 2. llama.cpp runs once.
- **Equal memory:** equal GPU expert memory, checked on measured VRAM.
  - Ours matches llama.cpp within 0.6 GiB at every budget.
  - FreeToken reserves 28.4–28.6 GiB at every rate, as in every earlier job.

**Headline (tok/s; ours = the law's table, picked on launch 1 at every budget; launch 2 unless marked):**

| Model | Experts on GPU | llama.cpp | FreeToken (better backend) | ours | ours ÷ FreeToken (95% CI) | ours ÷ llama.cpp |
|---|---|---|---|---|---|---|
| gpt-oss-120b | 11% | 34.9 | 54.0 (hybrid) | **69.9** | **1.294 [1.278, 1.312]** | 2.00× |
| gpt-oss-120b | 25% | 40.0 | 85.6 (hybrid) | **109.2** | **1.275 [1.254, 1.295]** | 2.73× |
| gpt-oss-120b | 40% | 47.7 | 132.2 (offload) | **152.5** | **1.154 [1.134, 1.172]** | 3.20× |
| Qwen3-30B-A3B BF16 | 12.5% | 19.5 | 38.7 (hybrid) | **40.0** | **1.032 [1.022, 1.042]** | 2.05× |
| Qwen3-30B-A3B BF16 | 25% (job 080, launch 1) | — | 54.7 (hybrid) | **63.1** | **1.153 [1.134, 1.173]** | — |
| Qwen3-30B-A3B BF16 | 43.75% (job 080, launch 2) | — | 103.1 (offload) | **108.2** | **1.049 [1.036, 1.064]** | — |

**The law's table against the current one (launch 1, paired):**
- gpt-oss: +7.9% [7.2, 8.5] at 11%, +5.4% [4.1, 6.6] at 25%, +3.3% [2.1, 4.4] at 40%.
- Qwen3 12.5%: +8.0% [7.3, 8.4].
- Job 080 adds Qwen3 at 25% (+6.7%) and 43.75% (+4.8%).

1. **Held.** On gpt-oss the law's table beats the current table at every budget, with every CI above 1.
2. **Held.** Ours leads FreeToken's better backend on gpt-oss at 11, 25 and 40% on the confirmation launch.
3. **Held.** On Qwen3 at 12.5% the law's table beats the current one (+8.0%), and ours leads FreeToken (+3.2%
   [2.2, 4.2]).
4. **Held.** Ours runs 2.00 / 2.73 / 3.20× llama.cpp on gpt-oss and 2.05× on Qwen3 at 12.5%.

**Notes.**
- **Faster than job 076:** on this host our gpt-oss speeds are 69.9 / 109.2 / 152.5, against 60.5 / 97.7 / 138.7 in
  job 076 (a 9950X host with the current table). FreeToken's are about the same (54.0 / 85.6 / 132.2 vs 50.2 / 80.3 /
  124.1). The difference is the table plus the host.
- **Smallest lead: Qwen3 at 12.5%.** Our miss path carries most of the token there, and FreeToken's hybrid backend also
  splits misses between CPU and PCIe.
- **The Qwen3 25% and 43.75% rows come from job 080** on the same listing. At 25% the FreeToken backend was picked on
  the same launch it is scored on, which biases that comparison in FreeToken's favour.
