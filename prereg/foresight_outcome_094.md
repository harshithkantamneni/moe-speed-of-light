# Job 094: foresight in bytes — the MIN-with-bypass oracle in the engine — predictions and outcome

The predictions are in the header of `jobs/094_foresight_bytes@vast.sh` (gpu branch commit 8b64c98, pushed before
launch). Statistics: `scripts/foresight_stats.py --job 094_foresight_bytes@vast --out prereg/foresight_094.json`
(paired bootstrap over the 30 sequences; the limit and the accounting's foresight-only term v(F) recomputed for this
host from its probe and its base measurement); the copy-latency simulation: `scripts/foresight_latency_sim.py --out
prereg/foresight_latency_sim.json`.

**Host.** RTX 5090 (memory clock 14,001 MHz; device read 1,692 GB/s) next to a Ryzen 9 7950X (16 cores, 14 helpers),
124 GB of RAM, Vast offer 53039899, 2.8 h at $0.61/h ($1.68). Host memory as weak as the 093 host's: 44.0 GB/s by the
CPU at the helper count, 45.7 over the link, 49.8 with both, 50.6 at best. The law's tables: gpt-oss `0,0,1,2,3`,
Qwen3 `0,0,1,2,3,4,5,6,7` (the same as on the 093 host). All 24 runs completed (6 cells x 4 configurations, 30
sequences x 256 steps each); nothing skipped.

**The two oracles.** Both read the routing of the rest of the sequence from a lookahead file recorded on this machine.
*bypass* (MIN with bypass, the policy whose reads define the limit): a miss runs on the CPU and is admitted only if its
next use comes before the furthest next use among the residents, which it then evicts; experts used once are bypassed.
*prefetch* (job 093's hit-optimal oracle): admits the experts the coming steps will route to, soonest first, whether or
not they are missed now. Both use the paced copy path (16 MB pieces, one in flight at a time, at most 64 queued).

**Result** (tok/s; ratios to base paired by sequence, 95% CI; hit rate; host reads per token as the engine counts
them: misses = experts run on the CPU or fetched on demand, admits = background copies; the law evaluated on the run's
own counters against its measured time; share of the gap to this host's limit closed; share of the accounting's
foresight-only term recovered).

| Cell | base tok/s | bypass W=all | bypass W=16 | prefetch W=all |
|---|---|---|---|---|
| gpt-oss 11% | 48.0 (hit 58%, misses 60.0, admits 3.5) | 49.6, **1.033 [1.029, 1.038]**, hit 65% (opt. 73), misses 50.1 (opt. reads 38.3), admits 16.0, law +7%, closed 6%, recovered 10% | 50.8, **1.059 [1.054, 1.063]** | 46.4, **0.966 [0.957, 0.976]**, hit 71%, misses 42.1, admits 39.0, law +19% |
| gpt-oss 25% | 78.6 (79%, 29.8, 4.2) | 92.8, **1.181 [1.175, 1.187]**, 86% (89), 20.1 (15.3), 8.8, +6%, 22%, 40% | 87.6, **1.114 [1.108, 1.120]** | 128.3, **1.632 [1.609, 1.655]**, 95%, 7.4, 23.2, +53% |
| gpt-oss 40% | 113.8 (89%, 15.3, 3.7) | 140.2, **1.232 [1.220, 1.243]**, 94% (95), 8.8 (6.8), 4.4, +2%, 24%, 52% | 122.9, **1.080 [1.069, 1.091]** | 178.1, **1.565 [1.537, 1.595]**, 98%, 2.6, 12.3, +39% |
| Qwen3 12.5% | 27.3 (60%, 155.0, 5.0) | 27.3, **1.002 [0.997, 1.006]**, 64% (75), 138.3 (96.5), 30.2, +5%, 0%, 1% | 27.9, **1.022 [1.017, 1.026]** | 25.8, **0.946 [0.939, 0.952]**, 67%, 127.7, 64.6, +11% |
| Qwen3 25% | 45.0 (78%, 84.6, 5.2) | 50.8, **1.131 [1.122, 1.139]**, 83% (89), 63.6 (43.4), 21.3, +9%, 18%, 30% | 50.5, **1.122 [1.116, 1.129]** | 63.3, **1.408 [1.388, 1.428]**, 91%, 34.3, 51.8, +37% |
| Qwen3 43.75% | 83.2 (92%, 32.3, 5.4) | 98.9, **1.189 [1.178, 1.200]**, 95% (96), 19.0 (13.7), 8.8, +1%, 22%, 43% | 91.6, **1.101 [1.090, 1.111]** | 134.7, **1.619 [1.584, 1.651]**, 99%, 3.5, 26.7, +47% |

**Why the bypass oracle's reads stay above the optimum's.** Its misses are 1.29-1.47x the optimum's reads (the two
GPU-bound cells 1.29 and 1.39; the host-bound ones 1.31, 1.31, 1.43, 1.47). The same policy simulated on the trace
with the engine's own semantics (the lookahead within the sequence, residents carried over) and an immediate copy
gives 39.4 / 17.5 / 9.3 and 98.0 / 46.5 / 17.6 misses per token (the limit's optimum, 38.3 / 15.3 / 6.8 and 96.5 /
43.4 / 13.7, also sees across sequences); with a copy that lands d decode steps after it is issued, during which the
expert misses again and is not re-admitted, the misses rise to 44.3 / 19.2 / 10.2 and 124.0 / 57.0 / 21.1 at d = 2
and 48.3 / 20.6 / 10.8 and 144.8 / 65.0 / 23.7 at d = 3. The engine's 50.1 / 20.1 / 8.8 and 138.3 / 63.6 / 19.0 fit
d = 3.6 / 2.6 / <1 and 2.7 / 2.8 / 1.4: two to four steps at the host-bound cells, one to two at the GPU-bound ones,
where the admissions are few. The paced copy path copies one 16 MB piece at a time from pageable host memory on a
single copier thread, behind the demand copies, and publishes a landed copy at the start of the next step; at 16-30
admissions per token (210-280 MB) it lags the admissions by steps. An expert still in flight is a miss the CPU reads
again.

**Predictions.**
1. The bypass oracle's host reads per token (misses) within 15% of the optimum's: **failed** at all six cells (+31,
   +31, +29%; +43, +47, +39%). Cause above: the copy latency.
2. Faster than base at every cell: **held** at gpt-oss 11% (1.033), 25% (1.181), 40% (1.232), Qwen3 25% (1.131) and
   43.75% (1.189); **held (point)** at Qwen3 12.5% (1.002 [0.997, 1.006]). By 10-45% at the four host-bound cells:
   **held** at gpt-oss 25% (18%) and Qwen3 25% (13%); **failed** at gpt-oss 11% (3%) and Qwen3 12.5% (0%).
3. Hit rate within 4 points of the optimum's: **held** at gpt-oss 25% (-3.2), 40% (-1.4) and Qwen3 43.75% (-1.3);
   **failed** at gpt-oss 11% (-8.2), Qwen3 12.5% (-10.9) and Qwen3 25% (-5.3).
4. The law on the bypass oracle's counters over-predicts its time by 10-40% at every host-bound cell: **failed** at
   all four (+7, +6, +5, +9%): the admissions are few enough that the law, which charges them in sequence, stays
   within 9% of the measured time. (On the prefetch oracle the same law over-predicts by 11-53%.)
5. At gpt-oss 11% and Qwen3 12.5% bypass beats prefetch by at least 15%: **failed** at both (+7%, +6%); at 25% and
   above the two within 15% of each other: **failed** at all four (bypass runs 0.72, 0.79, 0.80, 0.73x the prefetch
   oracle's speed).
6. W = 16 captures at least 70% of the W = all gain where that gain is positive: **held** at gpt-oss 11% (W = 16 is
   faster than W = all: 176%) and Qwen3 25% (94%); **failed** at gpt-oss 25% (63%), 40% (34%) and Qwen3 43.75% (53%);
   **untested** at Qwen3 12.5% (the W = all gain, 0.2%, is not distinguishable from zero).

**Reading.** The foresight the accounting prices, the optimum's bytes, is worth 13-23% in this engine at budgets of
25% and above and 0-3% at the two lowest, and the law predicts those runs within 1-9%: fewer host bytes, the
mechanism the accounting's foresight term describes, and no more. The bytes-optimal oracle does not reach the
optimum's reads because its copies land two to three steps after they are issued and the expert misses again
meanwhile; a copy published within the step would take its reads to the optimum's (the simulation at d = 1). The
hit-optimal oracle on the same host gains 41-63% at the same four cells and loses 3-5% at the two lowest, reading as
many host bytes as the online policy: its gain is the overlap of copies issued ahead of their use with the step, which
the law charges in sequence (over-prediction 37-53%). So in the engine the two largest terms of the accounting are one
mechanism: foresight is what makes the overlap possible, and its measured value where the link has slack is mostly
that overlap, not the bytes. Where the host binds hardest neither oracle gains more than 3%. Of 38 clauses, 12 held, 1
held on the point estimate, 24 failed, 1 untested.
