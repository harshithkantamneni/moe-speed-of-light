# Job 108 outcome: the time relation on new machines of the class it was found on

Script: `jobs/108_smallhosts@vast.sh` on the gpu branch.

- **Commits:**
  - 88e954c holds the predictions, the gates and the ordered offer list, committed before launch.
  - The wrappers for hosts d, e and f were committed before each started. Each host followed the registered list and
    the registered replacement rule; no prediction, gate or code changed.
- **Scoring:** `scripts/job108.py` scores it into `prereg/scorecard_108.json`, `prereg/job108.json` and
  `prereg/job108_families.json`.

## Hosts

| Job | Offer | Machine | Outcome |
|---|---|---|---|
| 108a | 43575030 | EPYC 7543, 30 usable cores, 1 NUMA node, driver 570 | **valid**; Nsight trace empty, so G is the median of earlier profiles |
| 108b | 32984223 | EPYC 7K62, 23 usable cores, driver 570 | V2: rounds 7.0% apart; stopped |
| 108c | 53424353 | Ryzen 7 5700X3D | V0: GPU rented before (job 100b); stopped before any download |
| 108d | 54156078 | Ryzen 9 9950X3D, 16 cores | **valid** |
| 108e | 45485624 | EPYC 9354 | V0: 323 GB of host memory in use; stopped |
| 108f | 54573924 | Ryzen 9 5950X, 16 cores | **valid**; the third valid machine, so renting stopped (credit $0.75) |

**The Nsight failure.** On both EPYC hosts (driver 570, CUDA 12.8) the trace held only 1–7 kernels, so no G could be
taken. The registration does not cover this case.
- For 108a we use the median G_prof of the 8 earlier profiled hosts of jobs 105–107 that passed every check (4.29 and 4.47 ms), as was
  done for 30 of the 34 launches the relation was found on.
- Any G in the range of those profiles leaves 108a off by 26.7–35.1%.

**A false statement in the registration.** The header said none of the listed offers was in the rental ledger.
Offer 54156078 (108d) had been rented for 4 minutes in job 099a, which produced no data. Its GPU UUID matched no earlier
launch, so gate V0 let it through.

## Score

21 clauses: 13 held on the point estimate, 6 failed, 2 untested.

1. **G_prof 4.0–5.0 ms.** Held on d and f (4.19–4.57 ms). Untested on a.
2. **Within 8% at every cell. Failed:** the EPYC 7543 was off by −34.2% and −27.9%.
   - **The Ryzens:** 9950X3D −6.9% and −4.0%; 5950X −6.5% and −0.9%.
   - **At most one cell beyond 6%:** failed, with 4 beyond.
   - **Median |error| ≤ 4%:** failed at 6.7%. Without the EPYC it is 5.3% over 4 cells, all within 8%.
3. **Implied read rate 0.85–1.25 of B_host.** Failed on the EPYC (0.58 and 0.61). The Ryzens were 0.91–0.99.

The plain form (no prediction): median 5.9%, 4 of 6 cells within 6%.

## Found after the fact (not registered)

- **The class line drawn after job 107 does not hold.** It was one NUMA node and at most 32 usable cores, and the EPYC
  7543 is inside it.
- **What separates the machines is the processor family.**
  - **Consumer parts** (Ryzen, Core, Threadripper): on 39 valid launches over 25 machines at gpt-oss 11%, the implied
    rate is 0.88–1.16 of the probe's best rate. The relation is within 8% on 37 and within 6% on 31.
  - **Server parts** (EPYC, Xeon): 5 of 8 launches failed a validity check. The 3 valid ones read at 0.27 (AMD
    engineering sample), 0.58 (EPYC 7543) and 1.04 (EPYC 7402P).
- **The new consumer machines of jobs 107–108** (5900XT, 9950X3D, 5950X) were within 8% at all 6 gpt-oss cells, and
  every one was under-predicted (−7.1% to −0.9%).
- **Source:** `prereg/job108_families.json`.

## What changes in the paper

- **Abstract, introduction, contributions and conclusion:** the relation is stated for consumer processors, the
  record of four registered tests is given, and none passed as registered.
- **Sec. 4:** job 108 is added, with the failed class line and the processor split (after the fact).
- **Table 3:** gains job 108.
- **Fig. 1:** adds the job 108 machines; the EPYC 7543 shows a large hatched segment.
- **Appendix:** gains Table "job 108" and the prediction log entry.
- **Supplement:** gains the scorecard.
