# Job 095: the single-read oracles — predictions and outcome

The predictions are in the header of `jobs/095_single_read@vast.sh` (gpu branch commit f9405c1, pushed before launch).
Statistics: `scripts/foresight_stats.py --job 095_single_read@vast --out prereg/foresight_095.json` (paired bootstrap
over the 30 sequences; the limit and the accounting's foresight-only term v(F) recomputed for this host from its probe
and its base measurement). The simulated predictions: `scripts/foresight_single_read_sim.py`
(`prereg/foresight_single_read_sim.json`).

**Host.** RTX 5090 (memory clock 14,001 MHz; device read 1,691 GB/s) next to a Ryzen 9 9950X (16 cores, 14 helpers),
123 GB of RAM, Vast offer 46402211, 2.7 h at $0.50/h ($1.35; a first instance on a 9950X3D stalled while loading its
image and was destroyed after 17 minutes, $0.16). A stronger host memory than the 093/094 hosts': 52.5 GB/s by the CPU
at the helper count, 43.2 over the link, 59.7 with both, 61.5 at best. The law's tables: gpt-oss `0,0,1,2,2`, Qwen3
`0,0,1,1,2,3,4,4,5`. Host-bound at the limit: gpt-oss 11% and 25%, Qwen3 12.5% and 25% (the optimum runs 9.1 and
21.5 experts per token on the CPU at gpt-oss 40% and Qwen3 43.75%). All 36 runs completed (6 cells x 6
configurations, 30 sequences x 256 steps each); nothing skipped.

**The oracles** (all read the routing of the rest of the sequence from a lookahead file recorded on this machine).
*fetch*: MIN with bypass decided before the step from the trace's record of it; each admitted miss is copied by the
GPU's FETCH op into the victim's slot and run from there this step (one read, on the critical path); bypassed misses
run on the CPU; no background copies. *lead2*: the scheduled single-read prefetch: an expert whose first use is at
least two steps away is copied ahead (unpaced, published two steps after issue) if MIN with bypass would admit it at
that use (the victim is needed later than the expert's use after that one, or never); an expert needed sooner misses
on the CPU. *both2*: fetch and lead2 together. *both3p*: both on the paced copy path (16 MB pieces, at most 64
queued) with a three-step lead. *hitopt*: job 093's hit-optimal oracle (Belady prefetch of every coming expert, two
reads per admitted expert), unpaced. *base*: the online policy with the law's table.

**Result** (tok/s; ratios to base paired by sequence, 95% CI; hit rate; host reads per token = CPU misses + fetches +
admissions, each expert read once, as a multiple of the optimum's R*; the law on the run's own counters against its
measured time; % of this host's limit; share of the gap to the limit closed; share of the accounting's foresight-only
term v(F) recovered).

| Cell (limit) | run | tok/s | ratio | hit % | reads (×R*) | law vs measured | % of limit | closed | v(F) recovered |
|---|---|---|---|---|---|---|---|---|---|
| gpt-oss 11% (121) | base | 55.5 | — | 58 | 63.4 (1.66) | — | 46 | — | — |
| | fetch | 71.4 | **1.286 [1.279, 1.294]** | 72 | 39.9 (1.04) | −8% | 59 | 41% | 75% |
| | lead2 | 69.8 | 1.257 [1.245, 1.268] | 86 | 53.8 (1.40) | +13% | 58 | 38% | 68% |
| | both2 | 77.4 | **1.394 [1.386, 1.403]** | 86 | 47.0 (1.23) | +13% | 64 | 52% | 95% |
| | both3p | 80.7 | **1.453 [1.441, 1.465]** | 83 | 47.0 (1.23) | +17% | 67 | 58% | 104% |
| | hitopt | 50.5 | 0.910 [0.899, 0.921] | 98 | 67.8 (1.77) | +24% | 42 | −18% | −33% |
| gpt-oss 25% (302) | base | 90.3 | — | 79 | 34.2 (2.23) | — | 30 | — | — |
| | fetch | 118.8 | **1.316 [1.307, 1.326]** | 89 | 16.5 (1.08) | −13% | 39 | 34% | 66% |
| | lead2 | 131.2 | 1.453 [1.444, 1.463] | 97 | 23.4 (1.53) | +29% | 43 | 44% | 86% |
| | both2 | 134.4 | **1.489 [1.479, 1.499]** | 97 | 22.2 (1.45) | +26% | 44 | 47% | 90% |
| | both3p | 150.2 | **1.664 [1.648, 1.681]** | 96 | 21.9 (1.43) | +36% | 50 | 57% | 110% |
| | hitopt | 102.6 | 1.137 [1.120, 1.154] | 99 | 32.3 (2.11) | +36% | 34 | 17% | 33% |
| gpt-oss 40% (512) | base | 128.8 | — | 89 | 19.0 (2.80) | — | 25 | — | — |
| | fetch | 161.4 | **1.254 [1.242, 1.265]** | 95 | 7.7 (1.14) | −14% | 32 | 27% | 62% |
| | lead2 | 177.8 | 1.381 [1.364, 1.399] | 99 | 11.3 (1.67) | +21% | 35 | 37% | 85% |
| | both2 | 179.2 | **1.392 [1.375, 1.410]** | 99 | 10.8 (1.60) | +20% | 35 | 38% | 87% |
| | both3p | 192.3 | **1.493 [1.472, 1.516]** | 99 | 10.6 (1.56) | +26% | 38 | 44% | 102% |
| | hitopt | 163.0 | 1.266 [1.254, 1.278] | 99 | 15.0 (2.20) | +31% | 32 | 28% | 65% |
| Qwen3 12.5% (68) | base | 32.3 | — | 59 | 165.5 (1.72) | — | 48 | — | — |
| | fetch | 43.4 | **1.345 [1.338, 1.351]** | 74 | 100.8 (1.04) | −9% | 64 | 49% | 75% |
| | lead2 | 39.3 | 1.216 [1.207, 1.224] | 82 | 140.9 (1.46) | +7% | 58 | 34% | 52% |
| | both2 | 45.9 | **1.421 [1.413, 1.429]** | 84 | 115.6 (1.20) | +7% | 68 | 57% | 87% |
| | both3p | 45.7 | **1.414 [1.405, 1.422]** | 80 | 114.0 (1.18) | +5% | 68 | 56% | 85% |
| | hitopt | 22.5 | 0.695 [0.683, 0.706] | 77 | 238.6 (2.47) | +7% | 33 | −84% | −128% |
| Qwen3 25% (150) | base | 51.8 | — | 77 | 93.4 (2.15) | — | 34 | — | — |
| | fetch | 71.4 | **1.379 [1.368, 1.390]** | 88 | 45.2 (1.04) | −15% | 48 | 42% | 70% |
| | lead2 | 72.8 | 1.406 [1.393, 1.421] | 96 | 66.3 (1.53) | +24% | 48 | 44% | 73% |
| | both2 | 76.4 | **1.476 [1.463, 1.488]** | 96 | 61.5 (1.42) | +21% | 51 | 49% | 82% |
| | both3p | 86.3 | **1.667 [1.650, 1.684]** | 94 | 58.4 (1.35) | +27% | 57 | 61% | 101% |
| | hitopt | 55.0 | 1.063 [1.048, 1.079] | 99 | 86.8 (2.00) | +29% | 37 | 9% | 15% |
| Qwen3 43.75% (303) | base | 91.4 | — | 92 | 39.1 (2.86) | — | 30 | — | — |
| | fetch | 115.5 | **1.264 [1.251, 1.276]** | 96 | 14.7 (1.08) | −17% | 38 | 30% | 62% |
| | lead2 | 127.5 | 1.395 [1.377, 1.412] | 99 | 23.4 (1.71) | +21% | 42 | 41% | 84% |
| | both2 | 128.6 | **1.407 [1.388, 1.424]** | 99 | 22.1 (1.62) | +20% | 42 | 41% | 86% |
| | both3p | 137.3 | **1.503 [1.478, 1.526]** | 99 | 21.5 (1.58) | +25% | 45 | 48% | 99% |
| | hitopt | 116.4 | 1.274 [1.262, 1.285] | 100 | 31.2 (2.29) | +34% | 38 | 31% | 64% |

The simulation's predictions (`foresight_single_read_sim.json`): fetch reads 40.0 / 17.6 / 9.3 and 101.3 / 47.3 /
17.7 per token, hits 72.3 / 87.8 / 93.5 and 73.6 / 87.7 / 95.4%; both2 reads 48.9 / 23.1 / 12.3 and 117.9 / 64.7 /
24.2, hits 86.5 / 98.6 / 99.5 and 84.1 / 96.7 / 99.5%.

**Predictions.**
1. The fetch oracle's reads within 10% of the simulation's: **held** at gpt-oss 11% (−0.3%), 25% (−6%), Qwen3 12.5%
   (−0.5%) and 25% (−4%); **failed** at gpt-oss 40% (−17%) and Qwen3 43.75% (−17%), where the engine read fewer than
   simulated (its cache is warm from the previous configuration's run; the simulation starts cold). Its hit rate
   within 3 points of the simulation's: **held** at all six (0.0, +0.7, +1.1, +0.1, +0.5, +0.8 points).
2. The fetch oracle faster than base at every cell: **held** at all six (1.25–1.38, every interval above 1.24). The
   law on its own counters within 10%: **held** at gpt-oss 11% (−8%) and Qwen3 12.5% (−9%); **failed** at gpt-oss
   25% (−13%), 40% (−14%), Qwen3 25% (−15%) and 43.75% (−17%): the oracle runs slower than the law's bandwidth-only
   account of its reads, by more at the cells where the fetched experts are a larger share of them — the fetch's
   per-layer latency (the copy must land before the layer's experts run), which the law does not price.
3. both2's reads within 25% of the simulation's: **held** at all six (−4, −4, −12, −2, −5, −9%). Its hit rate within 5
   points: **held** at all six (−0.8, −1.6, −0.4, +0.2, −0.4, −0.3).
4. both2 at 55–85% of this host's limit at the four host-bound cells: **held** at gpt-oss 11% (64%) and Qwen3 12.5%
   (68%); **failed** at gpt-oss 25% (44%) and Qwen3 25% (51%), below the band. Faster than hitopt at every host-bound
   cell: **held** at all four (1.39 vs 0.91, 1.49 vs 1.14, 1.42 vs 0.69, 1.48 vs 1.06).
5. The law over-predicts both2's time by 15–50% at the host-bound cells: **held** at gpt-oss 25% (+26%) and Qwen3 25%
   (+21%); **failed** at gpt-oss 11% (+13%) and Qwen3 12.5% (+7%), below the band: at the lowest budgets the CPU
   misses that remain (17 and 42 per token) keep more of the step serialised.
6. both2 gains at least 25% over base at gpt-oss 11% and Qwen3 12.5%: **held** at both (39% and 42%).
7. both3p within 10% of both2 at every cell: **held** at gpt-oss 11% (+4%), 40% (+7%), Qwen3 12.5% (−0.5%) and
   43.75% (+7%); **failed** at gpt-oss 25% (+12%) and Qwen3 25% (+13%), where the paced path is the faster one.

**Reading.** Read once, foresight pays everywhere. The fetch oracle realises the accounting's foresight-only state:
its host reads are 1.04–1.14× the optimum's R* at every cell, its hit rates are the simulation's within a point, and
it runs 25–38% faster than the online policy at every cell, including the two lowest budgets where the double-read
oracles of jobs 093 and 094 lost (−8%) or gained 0–3%; it reaches 32–64% of the limit and 62–75% of the speed-up the
foresight-only term v(F) predicts. The rest of v(F) is the fetch's own latency, which the law omits: the serialised
oracle runs 8–17% slower than the law's bandwidth-only prediction of it. Given the same foresight two to three steps
ahead, so that the copy lands before the use, the single-read prefetch adds the overlap: both2 runs 1.39–1.49× base
unpaced and both3p 1.41–1.67× on the paced path, 38–68% of the limit, 44–61% of the gap closed, 85–110% of v(F). The
hit-optimal oracle on the same host, with two reads per admitted expert, is slower than every single-read variant at
every cell and loses at the two lowest budgets (0.91, 0.69); the double read, not the foresight, was what made
foresight look worthless there. What remains between both3p and the limit (33–50% of the limit at the host-bound
cells, more at the GPU-bound ones) is the prefetch's own imprecision (its reads are 1.18–1.58× R*: the lead's victims
and the single-use experts it leaves to the CPU), the CPU misses still served in sequence, and the GPU work the limit
counts at the datasheet rate. Of 56 clauses, 44 held, 12 failed; the failures are of size (the law's error on the
serialised oracle, the band for both2's share of the limit at the 25% cells, the paced path faster than the unpaced
one, the warm cache at the GPU-bound cells); every sign clause held.

**Correction (5 October).** The explanation given above for the two failed reads clauses (P1c, P1f: "the engine's cache
is warm from the previous configuration's run") is wrong: `llama-ec-bench` creates a fresh context per configuration
and the cache starts empty. The cause was in the simulator the predictions were registered against
(`scripts/foresight_single_read_sim.py` and `foresight_latency_sim.py`): a resident carried over from the previous
sequence kept the next use it had at the end of that sequence, "never", and so was evicted first, where the engine's
`oracle_plan_fetch` recomputes every expert's next use within the current sequence. Corrected (next uses recomputed at
each sequence start), the simulation gives fetch reads 39.3 / 16.0 / 7.5 and 100.4 / 44.9 / 14.5 per token against the
engine's 39.9 / 16.5 / 7.7 and 100.8 / 45.2 / 14.7 (within 0-4% at every cell; hits within 0.4 points), and both-2
46.3 / 22.3 / 12.0 and 115.2 / 61.4 / 23.2 against 47.0 / 22.2 / 10.8 and 115.6 / 61.5 / 22.1. The clauses stay scored
against the registered values (the prediction was the registered simulator's number); the two failures are the
simulator's, not the engine's. The job-094 latency simulation changes likewise: the ideal (d = 1) misses are now within
0-4% of the optimum at the host-bound cells and 6-10% above it at the GPU-bound ones, and the engine's misses fit
d = 2.8-4.7 steps at the host-bound cells and 3.0-3.5 at the GPU-bound ones.
