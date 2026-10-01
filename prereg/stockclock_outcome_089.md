# Job 089: Table 1 on a stock-clock card, with the LRU attribution and the prefill / batch timings — predictions and outcome

The predictions are in the header of `jobs/089_headline_stockclock@vast.sh` (gpu branch commit c70e311, pushed before
launch). Statistics: `scripts/stockclock_stats.py --out prereg/stockclock_089.json`. The first host launched for this
job (Vast offer 51871552, instance 53621198, 2.0 h at $0.61/h) passed both gates and then had no working network: every
model download failed and nothing was measured (results/089a_headline_stockclock_nonet@vast); the job was relaunched
unchanged on Vast offer 49539124 (instance 53633140, 3.0 h at $0.79/h).

**Host.** RTX 5090 (memory clock 14,001 MHz, SM clock max 3,105 MHz, power limit 400 W; listing B: 17,001 / 3,420 /
575 W) next to a Ryzen 9 9950X with 126 GB of DDR5. Both gates passed: memory clock 14,001 MHz, device read
1,694 GB/s. Our probe at the law's helper count (14 threads): CPU 56.2 GB/s against listing B's 71.6 (−22%), zero-copy
PCIe 52.8 against 53.2, both together 66.1 against 77.7 (−15%). FreeToken's own probe: 56.1 / 53.3 (listing B:
69.8 / 54.4). The law's tables computed on this machine: gpt-oss `0,0,1,2,3` and Qwen3 `0,0,1,2,2,3,4,5,6` (listing B:
`0,0,1,1,2` and `0,0,1,1,2,3,3,4,5`): with the CPU slower relative to the link, the law copies more of each layer's
misses.

**Protocol.** Table 1's, with FreeToken's backend per cell carried over from job 081 (hybrid at gpt-oss 11 / 25% and
Qwen3 12.5 / 25%, offload at gpt-oss 40% and Qwen3 43.75%). Launch 1: ours (law table), FreeToken, stock llama.cpp per
cell; launch 2: FreeToken then ours, the comparison; then ours with `LLAMA_EC_POLICY=lru` and the law's table at every
cell; then llama-batched-bench at 25% (gpt-oss C32 vs `-ncmoe 27`, Qwen3 C32 vs `-ncmoe 36`) at pp 512 / 2,048, tg 128,
1 / 2 / 4 sequences. All 36 runs completed (1,080 rows); no step was skipped.

**Result (tok/s; launch 2 ratios, paired by problem, 95% CI).**

| Model | Budget | Ours (law) | FreeToken | llama.cpp | Ours ÷ FreeToken | Table 1 (listing B) | Ours vs B | FT vs B | llama vs B | LRU ÷ ours | LRU ÷ FreeToken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b | 11% | 57.9 | 48.0 (hybrid) | 28.9 | **1.207 [1.195, 1.219]** | 1.294 | −17% | −11% | −17% | 0.815 | 0.973 [0.962, 0.986] |
| gpt-oss-120b | 25% | 94.1 | 78.7 (hybrid) | 33.6 | **1.196 [1.163, 1.224]** | 1.275 | −14% | −8% | −16% | 0.872 | 1.045 [1.016, 1.068] |
| gpt-oss-120b | 40% | 133.6 | 122.2 (offload) | 40.1 | **1.093 [1.077, 1.111]** | 1.154 | −12% | −8% | −16% | 0.922 | 1.015 [0.997, 1.032] |
| Qwen3-30B-A3B | 12.5% | 33.5 | 32.7 (hybrid) | 16.1 | **1.027 [1.018, 1.036]** | 1.032 | −16% | −16% | −17% | 0.791 | 0.814 [0.808, 0.821] |
| Qwen3-30B-A3B | 25% | 53.9 | 51.4 (hybrid) | 18.5 | **1.048 [1.037, 1.059]** | 1.153 | −15% | −6% | −16% | 0.846 | 0.884 [0.874, 0.894] |
| Qwen3-30B-A3B | 43.75% | 95.8 | 98.3 (offload) | 23.8 | **0.974 [0.962, 0.987]** | 1.049 | −12% | −5% | −16% | 0.936 | 0.913 [0.900, 0.926] |

Ours runs 2.0–4.0× llama.cpp (listing B: 2.0–4.6×). Launch 1 (ours first) against launch 2 (FreeToken first): the
ratios differ by at most 1.0% (−1.0, +0.2, +0.7, +0.3, −0.2, +0.1%).

**Prefill and batch (llama-batched-bench, 25%, ours ÷ stock llama.cpp).** Prefill at 512 and 2,048 tokens: 0.79–0.80
(gpt-oss), 0.82–0.83 (Qwen3). Decode at 2 and 4 parallel sequences: 0.77–0.78 (gpt-oss), 0.77–0.79 (Qwen3). The tool's
single-sequence decode: 3.98–5.40× (gpt-oss), 3.69–4.40× (Qwen3) stock.

**Predictions.**
1. **Held.** Memory clock 14,001 MHz; device read 1,694 GB/s ≥ 1,500. The job is valid.
2. **Failed** (4 of 6 cells). Ours ÷ FreeToken within ±0.06 of Table 1: the differences are −0.088, −0.078, −0.061 on
   gpt-oss and −0.006, −0.105, −0.075 on Qwen3; only Qwen3 12.5% (−0.006) is inside the band, and gpt-oss 40% (−0.061)
   misses it by 0.001. Every cell is below Table 1's ratio. At Qwen3 43.75% FreeToken leads (interval below 1).
3. **Half held.** Ours' absolute speed 5–20% below listing B's at the GPU-heavy cells (gpt-oss 40%: −12%; Qwen3
   43.75%: −12%): **held**. Within ±12% at the other four cells: **failed** (−17, −14, −16, −15%). The host-bound cells
   lost more than the GPU-bound ones, the opposite of the prediction, because this host's CPU is 22% slower and its
   card only 18% lower-clocked (and FreeToken's offload cells, which run no expert on the CPU, lost 5–8%).
4. **Held.** Order effect at most 1.0% at every cell (predicted within 3%).
5. **Partly failed.** LRU with the law's table 10–30% slower than decayed frequency at every cell: 18.5, 12.8, 7.8%
   (gpt-oss) and 20.9, 15.4, 6.4% (Qwen3): **failed** at gpt-oss 40% and Qwen3 43.75% (below 10%; their intervals
   exclude 10%), held at the other four. LRU trails FreeToken at Qwen3 12.5% (0.814): **held**. LRU still leads
   FreeToken at gpt-oss 11 and 25%: **failed** at 11% (0.973 [0.962, 0.986], FreeToken leads), **held** at 25%
   (1.045 [1.016, 1.068]).
6. **Held (point).** Prefill at 2,048 tokens within ±20% of stock: Qwen3 0.825, gpt-oss 0.800 (0.80002, at the band's
   edge; the 512-token and multi-sequence prefills on gpt-oss are 0.789–0.799). Decode at 2 and 4 sequences below
   stock's on both models: 0.77–0.79. One llama-batched-bench run per configuration, so no intervals.

**The limit on this host.** With the exact optimum's reads (the same trace), this host's best probed rate (71.3 GB/s;
listing B 87.5) and the 5090's datasheet 1,792 GB/s, the host-bound limits are 19% lower than on listing B (gpt-oss 11%:
140 vs 172 tok/s; 25%: 351 vs 430; Qwen3 12.5%: 78 vs 96; 25%: 174 vs 214) and the GPU-bound ones about the same
(514 vs 519; 305 vs 308). Ours stands at 41 / 27 / 26% (gpt-oss) and 43 / 31 / 31% (Qwen3) of this host's limit, against
41 / 25 / 29% and 42 / 30 / 35% on listing B; FreeToken at 34 / 22 / 24% and 42 / 30 / 32%, against 31 / 20 / 25% and
40 / 26 / 34%. The limit moves with the host; the fractions do not.

**Reading.** The ratios of Table 1 do not carry over unchanged to a second host: ours leads FreeToken at five of the six
cells, by 9–21% on gpt-oss and 3–5% at Qwen3 12.5 and 25%, and trails by 2.6% at Qwen3 43.75%. The ratios are 0.01 to
0.11 below listing B's, more than the ±0.06 we predicted. The cause is the host, not the card: FreeToken's offload
backend, which runs no expert on the CPU, lost 5–8% (the card's share); every system that uses the CPU lost more — ours
12–17%, FreeToken's hybrid 6–16%, llama.cpp 16–17% — and the law's tables moved from `0,0,1,1,2` / `0,0,1,1,2,3,3,4,5` to
`0,0,1,2,3` / `0,0,1,2,2,3,4,5,6`. Our lead over FreeToken is the CPU path, and it is worth less where the CPU is slower
relative to the link (fig:ratio places both hosts on that axis). The LRU attribution: the admission policy is 77–84% of
the lead at gpt-oss 25 and 40% and more than the whole lead at gpt-oss 11% and Qwen3 12.5 and 25%; with LRU in place
of decayed frequency the cache trails FreeToken at four of six cells and leads it at one (gpt-oss 25%). Prefill and multi-sequence decode run 77–83% of stock llama.cpp's, which the
paper now reports as a cost of the cache rather than leaving unmeasured. The paper's abstract, contributions,
head-to-head paragraph, ablation paragraph, design paragraph and limitations are updated accordingly.
