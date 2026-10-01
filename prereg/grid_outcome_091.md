# Job 091: the grid, card 2 — Table 1's protocol on an RTX 4090 — predictions and outcome

The predictions are in the header of `jobs/091_grid_4090@vast.sh` (gpu branch commit 4a69c56, pushed before launch).
Statistics: `scripts/grid_stats.py --job 091 --out prereg/grid_091.json`. The limit is scored offline from the job's
probe, as the header says (`scripts/speed_limit.py`'s rule: the exact optimum's reads per token from
`prereg/speed_limit_v2.json`, the highest host rate any probe method reached, the card's datasheet 1,008 GB/s).

**Host.** RTX 4090 (24,564 MiB, memory clock 10,501 MHz, 400 W, PCIe 4.0 x16; device read 957 GB/s, 95% of the
datasheet) next to a Core i5-12400 (6 cores, 12 threads; the job's helper count 4) with 125 GB of RAM, Vast offer
49588631, 3.5 h at $0.54/h. Our probe: CPU 33–34 GB/s at 4–12 threads, zero-copy PCIe 25.2 GB/s, both together
36–37 GB/s (the highest sample, 37.4, is the limit's host rate); FreeToken's probe 35.9 / 24.6. Both gates passed.
The law's tables: gpt-oss `0,0,1,2,2`, Qwen3 `0,0,1,1,2,3,4,4,5`.

**Cells.** gpt-oss-120b at 11 and 25% (C 14 / 32; 40% does not fit: 24.3 GB of slots alone), Qwen3-30B-A3B at 12.5 and
25% (C 16 / 32); the VRAM arithmetic skipped Qwen3 43.75% as expected (28.05 GB against a 23 GB limit). FreeToken's
backend per cell carried over from job 081 (hybrid at all four). 20 runs, 600 rows, nothing else skipped.

**Result (tok/s; launch 2 ratios, paired by problem, 95% CI).**

| Model | Budget | Ours (law) | FreeToken | llama.cpp | Ours ÷ FreeToken | Table 1 (listing B) | Ours ÷ llama.cpp | Limit (tok/s) | Ours, % of limit | FreeToken | llama.cpp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b | 11% | 32.7 | 23.8 | 18.1 | **1.371 [1.346, 1.395]** | 1.294 | 1.81 [1.76, 1.86] | 74 | 44 [43, 46] | 32 | 25 |
| gpt-oss-120b | 25% | 55.0 | 42.5 | 21.0 | **1.296 [1.265, 1.326]** | 1.275 | 2.62 [2.52, 2.74] | 184 | 30 [29, 31] | 23 | 11 |
| Qwen3-30B-A3B | 12.5% | 18.8 | 18.2 | 10.5 | **1.033 [1.021, 1.045]** | 1.032 | 1.78 [1.76, 1.81] | 41 | 46 [45, 46] | 44 | 26 |
| Qwen3-30B-A3B | 25% | 31.3 | 30.0 | 12.3 | **1.045 [1.028, 1.063]** | 1.153 | 2.54 [2.49, 2.61] | 91 | 34 [34, 35] | 33 | 14 |

Every system runs at about half its listing-B speed (ours −50 to −53%, FreeToken −45 to −56%, llama.cpp −44 to −48%):
the host reads memory at 37 GB/s against 88, and every cell is host-bound at the limit (the optimum runs nothing on
the CPU; the GPU alone would allow 278 / 165 tok/s). The launch orders agree within 2.0%.

**Predictions.**
1. **Held** that ours leads FreeToken at every cell (four intervals above 1). The bands: gpt-oss 25% (29.6%, in
   10–35%) **held**; gpt-oss 11% (37.1% [34.6, 39.5]) **failed** the 10–35% band by 2 points; Qwen3 12.5% (3.3%)
   and 25% (4.5%) **held** the 0–15% band.
2. Ours ≥ 1.8× llama.cpp at every cell: gpt-oss 25% (2.62) and Qwen3 25% (2.54) **held**; gpt-oss 11% 1.81 [1.76,
   1.86] **held (point)**; Qwen3 12.5% 1.78 [1.76, 1.81] **failed** by 0.02. llama.cpp loses less than we do on this
   host because its whole-layer offload reads nothing over the slower link.
3. **Failed.** The law's tables fetch *more* than the headline machine's at one entry each (gpt-oss `0,0,1,2,2` against
   `0,0,1,1,2`; Qwen3 `0,0,1,1,2,3,4,4,5` against `0,0,1,1,2,3,3,4,5`). The prediction reasoned from the link alone
   (PCIe 4.0 at 25 GB/s); the CPU is slower in the same proportion (33 against 72 GB/s), so the CPU-to-link ratio
   (1.32) is the same as listing B's (1.35), and with every rate halved the per-layer form hides one more copy
   behind the CPU's experts. The split follows the ratio, not the link.
4. Ours at 25–45% of the limit at the host-bound cells: gpt-oss 11% 44.4% [43.3, 45.5] **held (point)** (the interval
   reaches 45.5), gpt-oss 25% 29.9% **held**, Qwen3 25% 34.3% **held**, Qwen3 12.5% 45.7% [45.1, 46.3] **failed** by
   0.7 points above the band.
5. **Held (point).** The two launch orders agree within 2.0% at every cell (+1.6, +2.0, −0.6, +0.1%).

**Reading.** On a 24 GB card next to a 37 GB/s host the picture of Table 1 holds with the host, not the card, setting
the scale: ours leads FreeToken by 30–37% on gpt-oss and 3–5% on Qwen3 (listing B: 15–29% and 3–15%; at gpt-oss 11%
FreeToken's hybrid backend lost 56% against our 53%, so the lead grew a little), runs 1.8–2.6× llama.cpp, and stands
at 30–46% of this machine's limit (listing B: 25–42%; job 089: 26–43%), a band that has now held on three hosts and
two cards. The 4090's own bandwidth never enters: every cell is host-bound, the regime the limit predicts for a host
this slow.
