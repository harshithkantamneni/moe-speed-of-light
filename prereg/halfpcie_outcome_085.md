# Jobs 085 and 085b: the contrasting machines (PCIe 4.0-class links) — predictions and outcome

The predictions are in the header of `jobs/085_half_pcie@vast.sh` (gpu branch commits 7a4f502 and 4f0ef91, pushed before
launch). Job 085b carries them over for Qwen3 (commit of `jobs/085b_half_pcie_qwen3@vast.sh`, before its launch).
Statistics: `scripts/samemachine_stats.py --job 085 / 085b`; numbers in `prereg/halfpcie_085.json` and
`prereg/halfpcie_085b.json`; cross-machine figure `scripts/fig_ratio.py` → `prereg/ratio_points.json`.

**Hosts.**
- **085:** RTX 5090 + Core i9-14900K (8P+16E, DDR5), Vast offer 46880906.
  - Our probe: CPU 58.0 GB/s at the thread count, PCIe zero-copy 28.0 GB/s, both at once 52.7–58.3.
  - FreeToken's probe: 57.8 / 23.4.
  - Its gpt-oss half ran. The Qwen3 GGUF conversion then hung with the CPU idle until the job's 6-hour timeout
    (rc 124), so it has no Qwen3 rows.
- **085b:** RTX 5090 + Ryzen 9 7900 (DDR5), Vast offer 53537780. It ran the Qwen3 half with the conversion in a
  separate environment.
  - Our probe: CPU 44.75 GB/s, PCIe 26.6, both 49.3.
  - FreeToken's probe: 43.4 / 28.9.
- **Deviation:** the two halves ran on different machines, both with PCIe 4.0-class links. Every comparison below is
  within one machine.

**The law's tables** (computed on each machine before any model run):
- gpt-oss 0,0,0,1,1 (headline machine 0,0,1,1,2; fixed 0,1,1,2,3);
- Qwen3 0,0,1,1,2,2,3,3,4 on the 7900 (headline 0,0,1,1,2,3,3,4,5). On the 14900K the unused table was
  0,0,0,1,1,2,2,3,3.

**Result (tok/s; launch 2 for ours ÷ FreeToken, launch 1 for law ÷ fixed; paired, 95% CI):**

| Model, budget | Ours (law) | FreeToken (better backend) | Ours ÷ FreeToken | Headline machine | Law ÷ fixed | Headline machine |
|---|---|---|---|---|---|---|
| gpt-oss 11% | 49.9 | 24.6 (offload) | **2.030 [1.993, 2.068]** | 1.294 | **1.340** [1.327, 1.355] | 1.079 |
| gpt-oss 25% | 81.9 | 48.2 (offload) | **1.699 [1.662, 1.737]** | 1.275 | **1.274** [1.256, 1.292] | 1.054 |
| gpt-oss 40% | 122.9 | 86.0 (offload) | **1.428 [1.391, 1.463]** | 1.154 | **1.211** [1.187, 1.235] | 1.033 |
| Qwen3 12.5% | 26.4 | 24.8 (hybrid) | **1.067 [1.058, 1.076]** | 1.032 | **1.184** [1.177, 1.190] | 1.080 |
| Qwen3 25% | 43.1 | 40.1 (hybrid) | **1.076 [1.063, 1.089]** | 1.153 | **1.149** [1.142, 1.157] | 1.067 |
| Qwen3 43.75% | 79.6 | 76.8 (hybrid) | **1.038 [1.017, 1.060]** | 1.049 | **1.094** [1.086, 1.101] | 1.048 |

- **FreeToken's hybrid backend on the 14900K** ran gpt-oss at 11–12 tok/s at 11%, against 24.6 for its offload
  backend.
- **llama.cpp at 25%:**
  - gpt-oss (14900K) 27.5 tok/s, against 33.62 predicted (+22%);
  - Qwen3 (7900) 14.98, against 15.38 predicted (+2.7%).
- **Ours without fetching** (Qwen3 25%, 7900): 36.4 tok/s, against 43.1 with the law's table.

**Predictions.**
1. **Held.** The law's tables fetch less than on the headline machine: every entry ≤, some lower, on both machines and
   both models.
2. **Held.** The law's table beats the fixed table at every budget of both models, and by more than on the headline
   machine at all 6 (+9% to +34%, against +3% to +8%).
3. **Held.** Ours leads FreeToken's better backend at every budget of both models, with every CI above 1.
4. **Held, exactly.** Ours' lead is larger than on the headline machine at 4 of 6 budgets: all three gpt-oss budgets
   and Qwen3 at 12.5%.
5. **Failed.** llama.cpp is within 5% of the law on Qwen3 (+2.7%, Ryzen) but not on gpt-oss (+22%, Core i9-14900K).
   This repeats job 069c's finding on a hybrid-core Intel CPU: llama.cpp splits each matrix evenly across P- and
   E-cores.

**Reading.**
- On slower links, the per-machine split matters more (+9–34%) and our lead over FreeToken grows on gpt-oss (to
  1.4–2.0×).
- FreeToken's offload backend sends every miss over the slow link. Its hybrid backend was worse still on the Intel
  host.
- On Qwen3 the lead stays in the 4–8% range. The split is what makes the difference: with the fixed table ours would
  have trailed FreeToken at 12.5% (22.3 vs 24.8).
