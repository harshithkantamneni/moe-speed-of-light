# Job 092: the grid, card 3 — Table 1's protocol on an RTX 3090, plus Mixtral-8x7B — predictions and outcome

The predictions are in the header of `jobs/092_grid_3090@vast.sh` (gpu branch commit 4a69c56, pushed before launch).
Statistics: `scripts/grid_stats.py --job 092 --out prereg/grid_092.json`; the limit is scored offline from the job's
probe (the exact optimum's reads from `prereg/speed_limit_v2.json`, the highest host rate any probe method reached,
the card's datasheet 936 GB/s).

**Host.** RTX 3090 (24,576 MiB, memory clock 9,751 MHz, 350 W, PCIe 4.0 x16; device read 888 GB/s, 95% of the
datasheet) next to a Core i9-11900KF (8 cores, 16 threads; helper count 6) with 125 GB of RAM, Vast offer 53295594,
3.9 h at $0.42/h. Our probe: CPU 42–45 GB/s (highest at 4 threads, 44.9, the limit's host rate), zero-copy PCIe
24.5 GB/s, both together 41–43 GB/s — *less* than the CPU alone, so a copy only takes bandwidth from the CPU's experts;
FreeToken's probe 42.7 / 25.3. Both gates passed. The law's tables copy nothing: gpt-oss `0,0,0,0,0`, Qwen3
`0,0,0,0,0,0,0,0,0`, Mixtral `0,0,0`.

**Cells.** gpt-oss-120b at 11% (C 14; both launches), Qwen3-30B-A3B at 12.5 and 25% (C 16 / 32; 43.75% skipped by
the VRAM arithmetic, 28.05 GB against 23), FreeToken's hybrid backend carried over from job 081 at all three; Mixtral-8x7B
Q4_K_M at C 2 (ours with the law's table and llama.cpp `--n-cpu-moe 24`, 30 problems each); at C 4 ours was started
with 4 minutes left before the job's 4 h deadline and completed no problem (rc 124), and llama.cpp `--n-cpu-moe 16` was
skipped. 17 complete runs, 510 rows.

**Result (tok/s; launch 2 ratios, paired by problem, 95% CI).**

| Model | Budget | Ours (law) | FreeToken (hybrid) | llama.cpp | Ours ÷ FreeToken | Ours ÷ llama.cpp | Limit (tok/s) | Ours, % of limit | llama.cpp |
|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b | 11% | 37.3 | 8.4 | 22.8 | 4.46 [4.25, 4.69] | **1.64 [1.60, 1.67]** | 88 | 42 [41, 43] | 26 |
| Qwen3-30B-A3B | 12.5% | 19.2 | 9.0 | 12.3 | 2.12 [2.03, 2.21] | **1.56 [1.48, 1.63]** | 49 | 39 [37, 41] | 25 |
| Qwen3-30B-A3B | 25% | 30.1 | 11.2 | 13.8 | 2.68 [2.53, 2.83] | **2.18 [2.09, 2.27]** | 110 | 27 [26, 29] | 13 |
| Mixtral-8x7B | C 2 (25%) | 8.34 | — | 8.22 | — | **1.015 [1.008, 1.024]** | — | — | — |

Launch 1 against launch 2: FreeToken's gpt-oss run fell from 9.4 to 8.4 tok/s and ours' Qwen3 runs from 20.5 / 31.7 to
19.2 / 30.1, so the two orders' ratios differ by −11.0% (gpt-oss 11%), +7.0% (Qwen3 12.5%) and +2.6% (Qwen3 25%): the
noisiest host of the study. Ours runs at 47–52% of its listing-B speed.

**FreeToken on this host is not compared at its best.** Its hybrid backend, the pre-registered carry-over of listing B's
selection, ran at 8–11 tok/s, below stock llama.cpp. Its own calibration on this machine (`ft bench bw`) measured its
CPU executor at 29.9 GB/s for MXFP4 and 41.0 for BF16 against a 24.6 GB/s link and recommends its offload backend for
every format (ratio below its 2.0 threshold); we did not run offload here. The ours ÷ FreeToken cells are reported but
not counted as a comparison; ours ÷ llama.cpp is the 3090 comparison.

**Mixtral-8x7B (8 experts per layer, top-2; 26 GB of Q4_K_M experts, a real offloading case on 24 GB).** With 2 of 8
experts resident per layer (25%), the decayed-frequency cache hits 31.5% of expert reads (engine counters: 43.9 misses
per token, 2.5 admissions) against the 25% that whole-layer pinning gives llama.cpp; with the law's table every miss
runs on the CPU, as llama.cpp's CPU layers do. The two systems tie: 8.34 against 8.22 tok/s, 1.015 [1.008, 1.024].

**Predictions.**
1. Ours leads FreeToken at every Qwen3 cell and at gpt-oss 11% (intervals above 1): **held** (3 of 3); the bands
   (0–15% on Qwen3, 10–35% on gpt-oss) **failed** at all three — 112%, 168%, 346% — because FreeToken's carried-over
   backend ran below llama.cpp on this host (above).
2. Ours ≥ 1.8× llama.cpp at every cell: Qwen3 25% (2.18) **held**; gpt-oss 11% (1.64 [1.60, 1.67]) and Qwen3 12.5%
   (1.56 [1.48, 1.63]) **failed**. llama.cpp's whole-layer offload loses less on a slow host than we do (its CPU layers
   are the same CPU path, with no admissions): the ours ÷ llama.cpp ratio at gpt-oss 11% is 2.00 on listing B, 1.81 on
   the 4090's host, 1.64 here.
3. Ours ≥ 1.3× llama.cpp on Mixtral at C 2 and C 4: **failed** at C 2 (1.015); **untested** at C 4 (deadline).
4. **Held.** The law's tables fetch no more than the headline machine's at any entry (they fetch nothing).
5. Orders within 3% at every Qwen3 cell: 25% (2.6%) **held (point)**; 12.5% (7.0%) **failed**. gpt-oss 11%, reported
   only: −11.0%.

**Reading.** The 3090 adds a third host where ours stands at 27–42% of the machine's limit (listing B 25–42%, job 089
26–43%, the 4090 host 30–46%), every cell host-bound, and where the law's answer to a host whose CPU alone saturates
memory is to copy nothing. The Mixtral tie is the cache's scope stated by a measurement: with 8 experts per layer a
25% cache hits 31% of reads where whole-layer pinning hits 25%, so there is little locality to earn, and ours matches
llama.cpp rather than beating it; the gain of Table 1 belongs to models with many experts per layer (128 and 128
here, top-4 and top-8), where a per-layer cache hits 60–90%. Two of the five predictions failed outright and FreeToken's
cells are void as a comparison; the paper reports all of it.
