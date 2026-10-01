# Job 088: the slow-link comparison done fairly — predictions and outcome

The predictions are in the header of `jobs/088_slowlink_fair@vast.sh` (gpu branch commit c70e311, pushed before
launch). Statistics: `scripts/slowlink_stats.py --out prereg/slowlink_088.json`.

**Host.** RTX 5090 + Core i9-14900K (8P+16E, DDR5), Vast offer 48829099: the CPU model of job 085, a second rental.
Our probe: CPU 63.5 GB/s at 8 threads, 74.8 at 24; zero-copy PCIe 29.9 GB/s. FreeToken's probe: 76.4 / 25.8.
This rental is faster than 085's: ours at 11% runs 59.3 against 49.9 tok/s (+19%).

**What changed from job 085.** FreeToken's CPU executor on `--moe-cpu-threads 8`, which FreeToken pins to cpu 0,2,..,14
(the eight performance cores; its log: "threads=8 (pinned to cores 0..14)"), with its calibration run on 8 threads
(it then copies 51.7% of misses over PCIe, against 65.5% with 23 threads in 085); a variant with no fetching
(`--moe-hybrid-max-fetch 0`); llama.cpp with `-t 8` pinned one thread per P-core (`-C 5555 --cpu-strict 1`) beside
`-t 24`. gpt-oss-120b only. Launch 1 ran every variant; launch 2 (the comparison) reran each system's better
configuration per budget in the reversed order.

**Result (gpt-oss-120b, tok/s; launch 2 ratios, paired, 95% CI).**

| Budget | FreeToken offload (L1) | hybrid, 8 threads (L1) | hybrid, 8 threads, no fetch (L1) | Ours, law table (L2) | FreeToken best (L2) | Ours ÷ FreeToken | Job 085 |
|---|---|---|---|---|---|---|---|
| 11% | 27.7 | 37.5 | 13.8 | 59.3 | 37.5 (hybrid) | **1.582 [1.564, 1.603]** | 2.030 |
| 25% | 52.8 | 62.1 | 13.7 | 94.3 | 62.1 (hybrid) | **1.519 [1.472, 1.559]** | 1.699 |
| 40% | 92.6 | 92.5 | 13.7 | 139.3 | 92.5 (offload) | **1.506 [1.476, 1.537]** | 1.428 |

- **FreeToken's hybrid on 8 threads:** 3.1× / 2.0× / 1.4× job 085's hybrid (12.0 / 30.5 / 66.0); its best configuration
  improves on 085's best by 1.52× / 1.29× / 1.08× (cross-rental means; this host is about 19% faster).
- **FreeToken on the CPU alone** (8 threads, no fetching): 13.7 tok/s at every budget, independent of the hit rate.
- **llama.cpp at 25%:** 40.0 tok/s with `-t 8` (the law predicted 36.2, −9.4%) against 32.8 with `-t 24` (the law
  predicted 41.3, +25.9%); `-t 8` / `-t 24` = 1.22. The hybrid-core failure is a thread-count choice.
- **The law's table:** 0,0,0,1,1, the same as job 085 on this CPU model.

**Predictions.**
1. **Held (point).** Hybrid on 8 threads ≥ 1.3× job 085's hybrid at every budget: 3.1, 2.0, 1.4. Cross-rental means,
   no interval; scaled by this host's +19%, the 40% value would be 1.18.
2. **Held (point).** FreeToken's best improves on 085's best at 3 of 3 budgets (1.52, 1.29, 1.08); the 40% value is
   within the host difference.
3. **Held** that ours leads FreeToken's best at every budget with the interval above 1, and by less than in 085 at 11
   and 25%; **failed** the band 1.10–1.50×: 1.58, 1.52, 1.51 (the 40% interval covers 1.50). At 40% the lead is
   larger than 085's 1.43, because FreeToken's offload backend gained only 8% on this rental while ours gained 13%.
4. **Held.** llama.cpp `-t 8` within 10% of the law (−9.4%) and faster than `-t 24` (1.22 [paired CI above 1]).
5. **Held.** The table is 0,0,0,1,1.

**Reading.** With FreeToken's executor on the performance cores the slow-link lead is 1.51–1.58× rather than 1.4–2.0×,
and it is a property of the link: FreeToken's better configuration copies half of its misses over a 26–30 GB/s link,
ours copies at most one of a layer's misses (table 0,0,0,1,1) and runs the rest on 74 GB/s of CPU bandwidth. The paper
reports job 088 and marks job 085's gpt-oss comparison as confounded.
