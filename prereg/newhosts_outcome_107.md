# Job 107 outcome: the time relation on machines never rented before

Script: `jobs/107_newhosts@vast.sh` on the gpu branch.

- **Commits:**
  - 1a0616a holds the predictions and the gates, committed before launch.
  - f9ac8c7, ddec94f and 346bd34 are amendments, each committed before the host it adds started. The predictions,
    the gates and the job were not changed.
- **Scoring:** `scripts/job107.py` scores it into `prereg/scorecard_107.json`, `prereg/job107.json` and
  `prereg/job107_classes.json`.
- **Engine:** patch oracle5, as in job 106.

## Hosts

| Job | Offer | Machine | Outcome |
|---|---|---|---|
| 107a | 51952147 | (stopped before lscpu) | V0: 50 GB of host memory in use; stopped before any download |
| 107b | 54579644 | Ryzen 9 5900XT, 16 cores, 1 NUMA node | **valid**: loss 0.1903 / 0.0855, rounds 0.3% apart |
| 107c | 52451721 | EPYC 9754, 61 usable cores | V2: rounds 3.5% apart; stopped after the gpt-oss 11% rounds |
| 107d | 54581350 | AMD engineering sample, 128 cores, 2 NUMA nodes | **valid**: loss 0.1899 / 0.0855, rounds 0.3% apart |
| 107e | 50680404 | EPYC 7663, 56 usable cores | V2: rounds 6.0% apart; stopped (replaced 107a from the registered list) |
| 107f | 53202662 | (CPU not listed) | V0: 50 GB in use (added by amendment) |
| 107g | 52395190 | (Threadripper PRO 5000) | V0: 96 GB in use (added by amendment; the last host the credit allowed) |

- **No new GPU was turned away.** None of the seven GPUs matched an earlier launch's UUID.
- **Not enough valid hosts.** The test needed 3 valid hosts and got 2, so by its own rule it is inconclusive. In
  substance it failed (below).

## Score

46 clauses on the two valid hosts plus the pooled ones: 11 held with an interval, 22 held on the point estimate, 13 failed.

1. **Profiled GPU compute.** Held on all 8 cells:
   - gpt-oss: 4.1–4.5 ms;
   - Qwen3: 5.6–5.9 ms.
2. **The overlap form within 6%. Failed:** 7 of 8 cells missed; all 8 under-predict.
   - On the 5900XT the errors were −7.1%, −2.3%, −11.3% and −6.4%.
   - On the engineering sample they were −52% to −59%.
   - Pooled median |error| was 31.7% against ≤ 4%, and the overlap form was no better than the plain one (30.4%).
   - Half-discount and misses-only did no better (31.9%, 33.4%).
3. **dk reads fewer than base:** held at every gpt-oss cell.
   - **dk/base ≥ 0.995 at every cell:** failed at one cell, the engineering sample at gpt-oss 25% (0.987, interval
     [0.978, 1.000]). Elsewhere it was 1.018–1.074.
   - **Median ≥ 1.01:** held (1.021).
4. **The relation's prediction of dk/base within 0.03:** held at 6 of 8 cells. It failed at two cells on the
   engineering sample (0.041, 0.067), where the relation itself is far off. The median error was 0.016, against 0.021
   for predicting no change.
5. **fetchplan/base > 1 and bypassplan/base > 1 at gpt-oss 11%:** held on both hosts, both with intervals.
   - fetchplan: 1.42 and 1.16.
   - bypassplan: 1.04 and 1.13.
6. **The crossover:** held on both hosts. Both have a ratio ≥ 0.40 (0.75 and 0.44), and on both fetchplan beat
   bypassplan. No valid host had a ratio below 0.35. The two hosts that failed V2 ordered the two loads the same way
   (EPYC 7663, ratio 0.21: bypassplan ahead; EPYC 9754, 0.51: fetchplan ahead), but they are outside the test.

## Found after the fact (not registered)

- **Measure.** For every launch of jobs 093–107 at gpt-oss 11%, the implied read rate (M + A(1 − G/T)) S / (T − G),
  taken over the probe's best rate.
- **Desktop-class launches** (one NUMA node and at most 32 usable cores): 0.88–1.16, median 0.98, over 38 launches on
  24 machines. The overlap form is within 6% on 32 of the 38.
- **Server launches:** 0.27–0.77 on all 5 (106c, 106d, 107c, 107d, 107e). Only 107d passed every validity check.
- **Cause unknown.** Why the engine does not reach the probe's rate on servers is not established. Candidates are memory
  placed on one NUMA node and synchronising up to 126 helper threads per layer (13–126 across the five servers).
- **Source:** `prereg/job107_classes.json`.

## What changes in the paper

- **Sec. 4.**
  - The relation is now a description of desktop-class machines.
  - Its registered test on new machines failed.
  - The desktop/server split is reported as found afterwards.
  - Fig. 1 adds the valid new machines and a hatched segment for reading below the machine's rate.
- **Abstract, introduction and contributions:** the claim is narrowed to match.
- **Sec. 5.**
  - The margin rule is reported on five machines: up to 6% at gpt-oss, median 1.02, one cell lost 1%.
  - The fewest-admission set is reported on the two new valid machines.
- **Limitations:** rented machines are often shared or unsteady, and the class boundary was drawn after the test.
