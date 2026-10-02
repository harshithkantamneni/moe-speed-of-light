# Job 093: what foresight is worth in the engine, measured — predictions and outcome

The predictions are in the header of `jobs/093_foresight@vast.sh` (gpu branch commit 4b25772, pushed before launch).
Statistics: `scripts/foresight_stats.py --out prereg/foresight_093.json` (paired bootstrap over the 30 sequences; the
limit and the model's foresight-only term v(F) recomputed for this host from its probe and its base measurement with
`scripts/shapley_gap.value_function`).

**Host.** RTX 5090 (memory clock 14,001 MHz; device read 1,554 GB/s) next to a Ryzen 9 9950X (16 cores, helper count
14) with 123 GB of RAM, Vast offer 48822557, 3.5 h at $0.70/h. A weak host memory: our probe reads 43.7 GB/s by the
CPU at the helper count, 45.8 over the link (zero-copy), 49.4 with both, and 50.5 at best by any method (listing B:
72 / 53 / 78 / 87.5; the stock-clock host of Table 1: 56 / 53 / 66 / 71). The law's tables: gpt-oss `0,0,1,2,3`,
Qwen3 `0,0,1,2,3,4,5,6,7`. Every cell is host-bound at the limit except gpt-oss 40% and Qwen3 43.75% (the optimum
runs 7.5 and 17.7 experts per token on the CPU there).

**The oracle.** `LLAMA_EC_POLICY=oracle`: after every decode step the policy knows the routing of the next W steps of
the sequence (a lookahead file recorded on this machine over the same teacher-forced text) and admits the experts they
will route to, soonest first, evicting the resident whose next use is furthest (Belady within the window); the copies
ride the normal admission path (paced, 16 MB pieces, at most 64 queued), and an expert still in flight at its step runs
on the CPU as any miss. This is the hit-optimal oracle: it admits an expert for a single future use as readily as for
ten. 46 runs (4 host-bound cells x 10 configurations, 2 GPU-bound cells x 3), teacher-forced over the trace corpora
(30 sequences x 256 steps), the lookahead pass per model untimed. All 46 runs completed; nothing skipped.

**Result (tok/s; ratios to base paired by sequence, 95% CI; hit rate and host reads per token from the engine's counters;
% of this host's limit).**

| Cell | base | oracle W=0 | ratio | hit base → oracle (optimum) | reads/token base → oracle (optimum) | % of limit base → oracle | model v(F) | foresight recovered | gap closed |
|---|---|---|---|---|---|---|---|---|---|
| gpt-oss 11% | 47.4 | 43.7 | **0.923 [0.914, 0.933]** | 58 → 67% (73%) | 63 → 82 (38) | 48 → 44% | 68.8 | −27% | −16% |
| gpt-oss 25% | 77.6 | 108.5 | **1.398 [1.384, 1.412]** | 79 → 92% (89%) | 34 → 31 (15) | 31 → 44% | 124.4 | 76% | 41% |
| gpt-oss 40% | 112.4 | 166.9 | **1.485 [1.462, 1.509]** | 89 → 98% (95%) | 19 → 15 (7) | 22 → 33% | 174.0 | 92% | 42% |
| Qwen3 12.5% | 27.1 | 24.8 | **0.916 [0.911, 0.922]** | 60 → 66% (75%) | 160 → 195 (97) | 49 → 45% | 40.2 | −28% | −18% |
| Qwen3 25% | 44.0 | 54.9 | **1.247 [1.235, 1.259]** | 78 → 89% (89%) | 90 → 88 (43) | 36 → 45% | 71.2 | 52% | 31% |
| Qwen3 43.75% | 78.7 | 123.1 | **1.565 [1.533, 1.592]** | 92 → 99% (96%) | 38 → 30 (14) | 26 → 41% | 121.3 | 103% | 49% |

Reads per token = misses + admissions (an admitted miss is read twice in this engine: by the CPU at its step and by
the copy). The window sweep (ratio to base): gpt-oss 25%: W = 2 0.81, 4 0.98, 16 1.35, 64 1.40, all 1.40 — half of the
full-window gain at W50 = 9.2 tokens (the trace study's W50 for this C/k: 9.4); Qwen3 25%: 0.99, 1.07, 1.25, 1.25,
1.25 — W50 = 6.1 (trace: 3.7). At gpt-oss 11% and Qwen3 12.5% every window loses (0.78-0.92 and 0.91-0.92). The
unpaced copy path (published two steps later), run at the four host-bound cells, reaches 98-99% hits at three of them
(75% at Qwen3 12.5%) and is slower than the paced oracle at three (0.96, 1.18, 0.72, 1.14 x base); at gpt-oss 11% it
is the faster of the two (1.04 x the paced oracle) and still loses to the online policy. The law evaluated on each oracle run's
own counters over-predicts its time by 24-39% at the four cells where the oracle gains (12.25 vs 9.22 ms at gpt-oss
25%; 7.89 vs 5.99 at 40%; 22.5 vs 18.2 and 11.3 vs 8.1 on Qwen3): the admissions' reads overlapped the step, which the
law charges in sequence. **No-overlap** (`LLAMA_EC_OVERLAP=0`, a layer's CPU misses no longer concurrent with its GPU
hits): 0.99-1.01 x base at every cell — the engine's within-layer overlap is worth nothing in mailbox mode. **All-CPU**
(no FETCH table): 0.92 / 0.91 x base on gpt-oss, 0.85 / 0.85 on Qwen3.

**Predictions.**
1. The oracle at W = 0 faster than base at every host-bound cell: **held** at gpt-oss 25% and Qwen3 25% (intervals
   above 1), **failed** at gpt-oss 11% and Qwen3 12.5% (intervals below 1).
2. The W = 0 gain 20-80% at every host-bound cell: **held** at gpt-oss 25% (40%) and Qwen3 25% (25%); **failed** at
   gpt-oss 11% (−8%) and Qwen3 12.5% (−8%). The law's point values (+38, +34, +65, +76%) were right in sign at the
   25% cells and wrong by 30-50 points everywhere; it does not charge the admissions' DRAM traffic or their lateness.
3. W = 16 captures at least half of the W = 0 gain: **held** at gpt-oss 25% (87%) and Qwen3 25% (100%); **untested**
   at gpt-oss 11% and Qwen3 12.5% (no gain to capture). W = 4 does at gpt-oss 11% and Qwen3 12.5%: **untested** for
   the same reason.
4. Ours with the oracle at 45-65% of this host's limit at the host-bound cells: **failed** at all four, each just
   below the band (44.0, 43.7, 44.8 and 44.5%; the intervals of the last two reach 45.5 and 45.8). At gpt-oss 11% and
   Qwen3 12.5% base already stood at 48-49% of this host's limit, because the limit's host rate is this host's weak
   50.5 GB/s; the oracle lowered it.
5. Hit rate within 3 points of the optimum's: gpt-oss 25% 92.4 vs 89.3 **failed** (3.1 points, above), Qwen3 25% 89.1
   vs 88.7 **held**, gpt-oss 11% 67.5 vs 73.4 and Qwen3 12.5% 65.7 vs 74.9 **failed** (copies late). Admissions per
   token within 30% of the optimum's reads: gpt-oss 11% 34.9 vs 38.3 **held**, 25% 20.1 vs 15.3 **failed** (+31%),
   Qwen3 12.5% 62.9 vs 96.5 **failed** (−35%), 25% 46.4 vs 43.4 **held**.
6. No-overlap 5-25% slower at the host-bound cells: **failed** at all four (0.0, 0.2, 0.0, 0.1% slower). All-CPU
   0-10% slower at every cell where the table copies: **held** at gpt-oss 11% (7.8%) and 25% (9.3%), **failed** at
   Qwen3 12.5% (14.7%) and 25% (14.6%).
7. The oracle's gain below 15% at the GPU-bound cells: **failed** at both (48%, 56%).

**Reading.** Foresight is worth a great deal in this engine where the misses per token are few enough for the link to
carry their copies ahead of time: +25 to +57% at 25% and above, closing 31-49% of the gap to the limit and recovering
52-103% of the accounting's foresight term; the W50 of the trace study is reproduced on gpt-oss (9.2 against 9.4
tokens). Where the host binds hardest (11-12.5%) a hit-optimal oracle is the wrong foresight: it reads more bytes than
the online policy (82 and 195 per token against 63 and 160) because Belady admits experts used once, and the optimum's
38 and 97 reads need bypass, which this oracle lacks; the copies it issues cannot be carried in time (hit rates 66-67%
against the optimum's 73-75%) and compete with the helpers for the same DRAM, and the unpaced path shows that
copies arriving in time (98% hits at gpt-oss 11%) do not remove the loss: the bytes do. The accounting's term is the optimum's
reads, foresight in bytes; job 094 runs that oracle. Two of the measured interior states say something about the
accounting's other terms: the engine's within-layer overlap knob is worth nothing (the "overlap" term is about host
reads serialised with GPU work across the step, not within a layer), and the law's over-prediction of the oracle runs
by 24-39% is the overlap foresight buys for free. Of 36 clauses, 9 held, 2 held on the point estimate, 21 failed, 4 untested.
