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
