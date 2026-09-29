# Job 072: deferred admission copies — prediction and outcome

The prediction is in the header of `jobs/072_defer@vast.sh` on the public `gpu` branch (commit 98c8398, pushed
before the launch). Outcome added 29 Sep 2026 after the run. Host: Ryzen 9 7900 (12 cores, 10 helpers) + RTX 5090.

| Config | defer = 0 (tok/s, 2 reps) | defer = 1 | change |
|---|---|---|---|
| C14 | 39.5 / 39.5 | 38.2 / 38.3 | −3.2% |
| C32 | 61.4 / 61.7 | 59.2 / 59.4 | −3.6% |
| C56 | 93.9 / 94.0 | 91.0 / 90.5 | −3.4% |
| C32 FETCH | 73.3 / 73.3 | 72.9 / 72.9 | −0.5% |
| C32 PREFETCH | 70.8 / 71.0 | 69.1 / 69.1 | −2.5% |

1. **Moot.** "Inputs time falls by half" did not apply: `inputs` was only 14–41 µs per step in steady state.
   - Under the profiler, the input uploads looked slow (81 small copies behind the admission copies). That was an
     artefact of tracing.
   - The real host-side boundary is about 1 ms: graph launch 0.5–0.8 ms, post-processing 0.25 ms.
2. **Failed.** Deferring loses 3–4%. It moves the admission reads from the boundary into the CPU phases (sync time
   +0.8 ms at C32), where they compete with the helpers for host memory. The boundary itself saves only 0.2–0.4 ms.
3. **Held.** Hit rates and admissions are identical.

Per the job's decision rule, the boundary is host work (launch and bookkeeping), not the admissions. Admission copies
cost less where they are now than overlapped with CPU phases: the 071 lesson again.
