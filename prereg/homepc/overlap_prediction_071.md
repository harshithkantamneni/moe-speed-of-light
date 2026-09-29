# Prediction for job 071 (overlap test on an RTX 5090 + Ryzen 9 9950X3D2 host), recorded before its results

Made on 29 Sep 2026, 00:50 America/Chicago. Job 071 was launched at 00:45 and had not produced any measurement when this
file was committed.

## Basis

The host-DRAM law (`scripts/law_hostdram.py`) was fitted on the 9950X host (jobs 064/066). It predicted the other two
hosts out of sample at 2.2% and 3.2% median error. Its form is

  T = G + max(cpu/B_c, link/B_p, (cpu + link)/B_both),  with G = 4.82 ms per token for the expert cache.

It has no overlap between G and the host-memory term. A mechanism that reads future experts' bytes while the GPU
works (PREFETCH_LATE, LLC) can at best reach T = max(G, host bytes / B). This file writes B for this host's measured
bandwidth (concur.cu), and B_both for the both-paths figure.

## Predictions (own text, 12 × 128, per budget C = 14 / 32 / 56)

1. **The law holds for the configurations it already covers.** For base, PREFETCH (early), FETCH and LLC-off
   configurations, the law with this host's measured bandwidths and G = 4.82 ms predicts tok/s within 8% for each
   configuration and within 4% median. Nothing is refitted.
2. **Upper bound.** No configuration exceeds 1 / max(G, host bytes / B_both), computed from its own byte counts.
3. **PREFETCH_LATE is at least as fast as early PREFETCH** at C = 32 and C = 56. The late copy reads host memory
   while it would otherwise be idle.
4. **LLC helps only through bytes it moves into the idle window.** The gain of LLC alone over base is between 0% and
   +25% at C = 32. The cap is the law's max-form bound with the idle window: per layer about G / 36 ≈ 134 µs × B.
   A gain above 25% would mean the L3 also saves compute or latency beyond bandwidth. A loss would mean the warm-up
   interferes with the critical path.
5. **Accuracy.** Each configuration's top-1 agreement with base is at least 98% (these mechanisms change only where
   an expert runs), and NLL is within 1%.

Decision rule: if 3 and 4 both fail (no gain from either overlap mechanism), the overlap dividend is not reachable
with next-layer lookahead on this hardware. The paper then states the dividend as a bound-level finding only.

## Outcome (added 29 Sep 2026, 06:00 UTC, after the run; nothing above was changed)

Job 071 ran on an RTX 5090 + Ryzen 9 9950X3D2 host (192 MB L3 in two CCDs; CPU read 69.6 GB/s, CPU + copy engine
86 GB/s). Results: `results/071_overlap_x3d@vast/` on the `gpu` branch.

| Config | C = 14 | C = 32 | C = 56 |
|---|---|---|---|
| base (tok/s) | 55.8 | 84.0 | 121.2 |
| PREFETCH (early) | +13.9% | +11.0% | +3.7% |
| PREFETCH_LATE | +8.3% | +3.4% | −3.0% |
| LLC | −1.8% | −2.5% | −5.0% |
| FETCH | +7.2% | +7.6% | +4.4% |
| LLC + FETCH | −8.4% | −10.1% | −13.6% |

1. **Mostly held.** Covered configurations: median error 1.8%, but C14 FETCH is off by 11.2%, above the 8% limit.
2. **Held.** No configuration exceeds its max-form bound; every one is far below it.
3. **Failed.** PREFETCH_LATE is slower than early PREFETCH at every budget (3–7 points).
4. **Failed.** LLC loses 2–5% alone and 8–14% with FETCH.
5. **Held.** Top-1 agreement with base is ≥ 98.4% for every configuration; mean NLL is within 0.6%.

By the decision rule, the overlap dividend is not reachable with next-layer lookahead on this hardware, and the paper
states it only as a bound-level quantity.

Why the prediction was wrong. The premise was that host memory is idle while the GPU works. It is not:
- the cache's own background admissions already read 100–145 MB per token from host memory, and the copy engine
  schedules them in exactly those windows;
- early PREFETCH already moves predicted experts in the same windows. That is where its +11–14% comes from.

So what remained to overlap was small, and both new mechanisms added reads that compete with the critical path:
- the late copy starts after the CPU's wait, so it runs inside the next layer's short attention window (~130 µs per
  layer, about half an expert at this bandwidth);
- LLC warms the L3 of whichever CCD the helper runs on, with ~100 MB per token of extra reads.
