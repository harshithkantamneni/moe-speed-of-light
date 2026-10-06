# Job 102 outcome: can the bound's read time be reached?

Script: `jobs/102_readsched@vast.sh` (gpu branch, predictions committed in e089377 and amended before launch to name
Pd as host 102a, Pf's offer being gone). Microbenchmark: `jobs/ec2/readsched.cu`. Scored by machine by
`scripts/job102.py` (`prereg/scorecard_102.json`, `prereg/job102.json`).

Hosts: 102a = i9-13900KF (Pd of job 099 again; link/CPU 0.40), 102b = Ryzen 9 9950X (1.02), 102c = Ryzen 7 9800X3D
(0.55). gpt-oss MIN reads at C = 14 (11%) and C = 32 (25%), first 4,000 steps of the AIME routing, two repetitions.

## Result

Share of the bound's host term (R* S / B_host) reached by each mode:

| Host | C | per layer | per token | link only | CPU only | analytic per layer |
|---|---|---|---|---|---|---|
| Pd again | 14 | 0.861 | 0.933 | 0.354 | 0.921 | 0.984 |
| Pd again | 32 | 0.889 | 0.929 | 0.355 | 0.921 | 0.980 |
| 9950X | 14 | 0.963 | 0.974 | 0.946 | 0.865 | 0.941 |
| 9950X | 32 | 0.955 | 0.966 | 0.942 | 0.864 | 0.920 |
| 9800X3D | 14 | 0.950 | 0.955 | 0.531 | 0.906 | 0.951 |
| 9800X3D | 32 | 0.933 | 0.939 | 0.529 | 0.906 | 0.935 |

- Waiting per layer reaches 0.86-0.96 of the host term; one wait per token 0.93-0.97. The bound's read time is
  nearly reachable in host reads alone.
- The analytic model of the same schedules (probe rates, no latency) predicts each mode's time within 0.93-1.14x
  (24 mode-cells); it is optimistic on the Intel host (per-layer mode 10-14% slower than modelled, link-only 11% slower) and,
  on the two AMD hosts, pessimistic for the link (copies run faster than the probe's link rate).
- On the Intel host (fast CPU, slow link) reading everything on the CPU (0.92) beats the probe-rate split (0.86-0.89):
  the per-layer wait on the slow link costs 4-7 points there.

## Predictions

- held (point) 102a-P1-C14 token mode reaches >= 0.80 of the bound's host term (Pd again, C = 14) 0.9331
- held (point) 102a-P2floor-C14 layer mode reaches >= 0.50 of it (Pd again, C = 14) 0.8607
- held (point) 102a-P2below-C14 layer mode at least 0.03 below the analytic per-layer fraction 0.98 (Pd again, C = 14) 0.1228
- held (point) 102a-P4-C14 layer_link > 10% slower than layer (CPU path 2.49x the link; Pd again, C = 14) 1.4294
- held (point) 102a-P5-C14 two repetitions agree within 3%, worst mode (Pd again, C = 14) 0.0088
- held (point) 102a-P1-C32 token mode reaches >= 0.80 of the bound's host term (Pd again, C = 32) 0.9294
- held (point) 102a-P2floor-C32 layer mode reaches >= 0.50 of it (Pd again, C = 32) 0.8893
- held (point) 102a-P2below-C32 layer mode at least 0.03 below the analytic per-layer fraction 0.98 (Pd again, C = 32) 0.091
- held (point) 102a-P4-C32 layer_link > 10% slower than layer (CPU path 2.49x the link; Pd again, C = 32) 1.5074
- held (point) 102a-P5-C32 two repetitions agree within 3%, worst mode (Pd again, C = 32) 0.0014
- failed 102a-P3 layer mode's fraction lower at C = 32 than at 14 (Pd again) -0.0286
- held (point) 102b-P1-C14 token mode reaches >= 0.80 of the bound's host term (9950X, C = 14) 0.9744
- held (point) 102b-P2floor-C14 layer mode reaches >= 0.50 of it (9950X, C = 14) 0.9632
- failed 102b-P2below-C14 layer mode at least 0.03 below the analytic per-layer fraction 0.94 (9950X, C = 14) -0.0221
- held (point) 102b-P4-C14 layer_link within 10% of layer (paths match, link/CPU 1.02; 9950X, C = 14) 0.0178
- held (point) 102b-P5-C14 two repetitions agree within 3%, worst mode (9950X, C = 14) 0.001
- held (point) 102b-P1-C32 token mode reaches >= 0.80 of the bound's host term (9950X, C = 32) 0.9663
- held (point) 102b-P2floor-C32 layer mode reaches >= 0.50 of it (9950X, C = 32) 0.9553
- failed 102b-P2below-C32 layer mode at least 0.03 below the analytic per-layer fraction 0.92 (9950X, C = 32) -0.0351
- held (point) 102b-P4-C32 layer_link within 10% of layer (paths match, link/CPU 1.02; 9950X, C = 32) 0.0136
- held (point) 102b-P5-C32 two repetitions agree within 3%, worst mode (9950X, C = 32) 0.0006
- held (point) 102b-P3 layer mode's fraction lower at C = 32 than at 14 (9950X) 0.0079
- held (point) 102c-P1-C14 token mode reaches >= 0.80 of the bound's host term (9800X3D, C = 14) 0.9547
- held (point) 102c-P2floor-C14 layer mode reaches >= 0.50 of it (9800X3D, C = 14) 0.9498
- failed 102c-P2below-C14 layer mode at least 0.03 below the analytic per-layer fraction 0.95 (9800X3D, C = 14) 0.0012
- held (point) 102c-P4-C14 layer_link > 10% slower than layer (CPU path 1.82x the link; 9800X3D, C = 14) 0.7885
- held (point) 102c-P5-C14 two repetitions agree within 3%, worst mode (9800X3D, C = 14) 0.0008
- held (point) 102c-P1-C32 token mode reaches >= 0.80 of the bound's host term (9800X3D, C = 32) 0.9386
- held (point) 102c-P2floor-C32 layer mode reaches >= 0.50 of it (9800X3D, C = 32) 0.9328
- failed 102c-P2below-C32 layer mode at least 0.03 below the analytic per-layer fraction 0.94 (9800X3D, C = 32) 0.0024
- held (point) 102c-P4-C32 layer_link > 10% slower than layer (CPU path 1.82x the link; 9800X3D, C = 32) 0.7629
- held (point) 102c-P5-C32 two repetitions agree within 3%, worst mode (9800X3D, C = 32) 0.0004
- held (point) 102c-P3 layer mode's fraction lower at C = 32 than at 14 (9800X3D) 0.017

Prediction 2's second half failed on both AMD hosts: we expected the per-layer wait to cost at least 0.03 of the host
term against the analytic schedule; measured, the per-layer mode matches or beats the model there. Prediction 3
failed on the Intel host (the per-layer fraction is higher at C = 32). Every other clause held on its point value
(no clause has an interval: each is one deterministic run, and repetitions agree within 0.9%).

## What the paper takes from it

The split ignores one constraint: read once, every expert MIN admits must cross the link to reach its slot.
`scripts/readsched.py` computes, per probed host, the cap this puts on MIN's one-read schedule,
min(1, (R*/A*) B_link / B_host), with A* MIN's admissions per token (57% of its reads at C = 14, 69% at C = 32) and
B_link the probe's highest link rate (number check 9: the first version used the probe's zero-copy line and B_cp,
which counted the combined-rate slack twice): 0.53-0.71 of the host term at 11% and 0.43-0.58 at 25% on hosts whose
link reads at less than half their CPU rate, at least 0.86 and 0.70 on the others. On the slow-link hosts of the
accounting this cap alone is 25-54% of the gap between the deployed cache and the bound; on the fast-link hosts it
binds at 9 of 26 host-budgets, at most 22% (`prereg/readsched_gap.json`).
