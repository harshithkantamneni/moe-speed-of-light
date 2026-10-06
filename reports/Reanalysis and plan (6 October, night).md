# Re-analysis from first principles, and the plan (6 October, night)

Scripts: `scripts/reanalysis.py` (machine table, rank correlations, sum law, per-problem structure; writes
`prereg/reanalysis.json` and `prereg/reanalysis_hosts.json`), `scripts/reanalysis_perlayer.py`
(`prereg/reanalysis_perlayer.txt`), `scripts/reanalysis_idlebus.py` (`prereg/reanalysis_idlebus.txt`). Every number below
comes from those outputs.

## What went in

- **Engine runs.** Every engine launch of jobs 093–104 that has a probe and gpt-oss runs at C = 14 and 32 (11% and 25%
  budgets):
  - 30 launches on 21 machines.
  - Relaunches are merged by GPU UUID. Pf ran 4 times; Pg 3; O4, Pd, Ph and the second 285K twice each.
- **Features per machine,** all from the job's own probe:
  - CPU rate at the helpers' thread count, B_c;
  - link rate, zero-copy (B_p) and copy engine;
  - both paths together, B_cp, and the best rate, B_host;
  - single-thread CPU rate and how it scales with threads;
  - per-copy fixed latency and per-MB cost;
  - STREAM triad, cores and RAM.
- **Outcomes:**
  - the deployed cache's time per token;
  - every state's speed relative to it;
  - efficiency against the bound;
  - counted host reads, and per-problem times.
- **Statistics.** Spearman ρ with a 95% bootstrap interval over machines, a permutation p-value, Kendall's τ, and
  Benjamini–Hochberg q across the 16 features × 16 outcomes.

## Finding 1: time per token is GPU compute **plus** host reads, and our engine already reads at the machine's best rate

Write T for the deployed cache's time per token, G for a fixed per-token cost, R for counted host reads per token and S
for the expert size. The data fit **T = G + R·S/B_host** with one constant G.

- **The constant G is measured, not fitted.** Summing every non-wait, non-copy kernel in the earlier Nsight profiles
  (job 069c) gives the GPU's own compute per token:
  - Core Ultra 7 270K: 4.28 (C14) and 4.53 ms (C32);
  - EPYC 7352: 4.28 and 4.52 ms;
  - EPYC 9655: 4.47 and 4.69 ms.

  The time model's "fitted" G (4.56 / 4.53 ms) is this quantity. For Qwen3 the profile gives 5.8 ms, and the implied G
  is 4.9–7.2 ms.
- **Implied G per state (T minus counted reads at B_host), deployed state over all 30 launches:**

  | Budget | Median | IQR | ρ with link/CPU ratio |
  |---|---|---|---|
  | 11% | 4.32 ms | 4.14–4.58 | 0.02 |
  | 25% | 3.85 ms | 3.64–3.99 | −0.31 |

  It does not depend on the machine.
- **The deployed cache's reads run at B_host.** Effective reads, (T − G)·B_host/S, over counted reads (misses plus
  admissions):

  | Budget | Median | IQR | Range |
  |---|---|---|---|
  | 11% | 0.98 | 0.97–1.00 | 0.83–1.12 |
  | 25% | 0.91 | 0.89–0.94 | |

  At 25% some background copies overlap the GPU's work.
- **Elasticity to host bandwidth.** The elasticity of T to B_host is −0.74 [−0.81, −0.66] at 11%. Once G is removed,
  T − G has elasticity **−0.97 [−1.04, −0.88]**, exactly what pure bandwidth-bound reads give.
- **Consequence.** Our cache wastes nothing in executing the reads it decides to make. Its whole gap to the bound is:
  1. **Serialisation:** the GPU's ~4.3 ms of compute is not overlapped with host reads. This is 24–55% of the gap at
     11% and 34–64% at 25% (13–33% and 22–48% of the time).
  2. **Excess reads:** 63–69 experts per token against MIN's 38.3 at 11%, each at S/B_host.

  This is the "rest" the reviewers say is never measured: mostly serialisation. The Shapley 2×2 becomes a detail.
- **Why the bound is a max but the engine is a sum.** At batch 1, layer l's missed experts cannot be read before layer
  l's router has run, and layer l+1 cannot start before layer l's experts are done. A demand-driven system therefore
  pays GPU compute and host reads in series.
  - This gives a valid, tighter **demand bound**: T ≥ T_GPU,non-expert + R\*·S/B_host. It holds for every system that
    reads an expert only after it is requested: ours, FreeToken, llama.cpp.
  - The paper's max-form bound is reachable only by reading before routing, that is, with foresight.
  - On O4 at 11% the demand bound is about 13.3 ms against the max bound's 10.0. Our cache (20.6 ms) is at about 65% of
    the demand bound, and the oracle with its copies made ahead (13.8) is at it.
- **Why the old "efficiency falls with CPU bandwidth" signal was an artefact.** Efficiency (bound/T) correlates with B_c
  at ρ = −0.92 [−0.97, −0.75] at 25%, the strongest correlation in the table. It disappears once G is removed (ρ = −0.22 at 11%, 0.00 at 25%, on bound/(T − G)):
  faster memory shrinks the read term but not the fixed GPU term.
- **In-step copies pay a link-dependent latency.** Their implied G is higher and tracks the ratio:
  - MIN copied once: 7.4 ms median, ρ = −0.91 with the ratio;
  - admit every miss: 16 ms, ρ = −0.91;
  - copies made ahead: G falls to 2.7 ms, because they overlap the GPU's work.

## Finding 2: two kinds of foresight, and the cheap one is available today

- **Long horizon to read less.** Already in the paper: about 0.65·C distinct experts, which nothing realisable reaches.
- **One layer of horizon to overlap reads with GPU compute.** The paper dismissed layer-ahead predictors because they
  "see less than one token ahead", but one layer is exactly what overlap needs.
  - Layer l's output predicts layer l+1's experts with recall **0.89 at top-4 and 0.99 at top-8** (job 063 records).
  - During the GPU's non-expert compute (about 3.3 ms per token, about 92 µs per layer) both host paths are idle.
  - Simulated on the job 063 trace: CPU helpers prefetch the predicted, non-resident experts of layer l+1 while the
    GPU computes, limited to the bytes the window allows. The prefetch hides:

    | Budget | Hidden per token | Share of T (B_host 30 → 90 GB/s) |
    |---|---|---|
    | 11% | 2.6 ms | 8% → 19% |
    | 25% | 1.7 ms | 9% → 18% |

    The share is largest on fast-memory hosts, where the cache is now furthest from the bound.
- **Why this is a candidate deployable result.**
  - No oracle, no long horizon, and the gain has a predicted ceiling from finding 1.
  - It also reframes the paper's foresight story: long-horizon foresight reduces reads, one-layer foresight removes
    serialisation.

## Finding 3: what the machine's probe predicts (21 machines, rank correlations)

| Outcome | Best predictor | ρ [95% CI] | q |
|---|---|---|---|
| Deployed time at 11% / 25% | B_host | −0.95 [−0.99, −0.83] / −0.92 [−0.99, −0.77] | 0.002 |
| MIN copied once / deployed, 11% | link/CPU ratio | +0.89 [+0.63, +0.97] | 0.002 |
| MIN copied ahead / deployed, 11% | link/CPU ratio | +0.87 [+0.54, +0.99] | 0.002 |
| Admit every miss / deployed | link/CPU ratio | +0.95 [+0.75, +0.99] | 0.002 |
| **MIN loaded twice / deployed, 11%** | **absolute link rate B_p** | **+0.79 [+0.49, +0.92]** (ratio only +0.45, q 0.19) | 0.002 |
| Deployed set copied in the step / deployed, 11% | B_host | −0.80 [−0.96, −0.45] | 0.002 |

- **New:** background copies (two reads) depend on how fast the link is, not on the link/CPU balance. In-step copies
  depend on the balance.
- **New:** fetch-on-admit hurts on fast-memory machines. There the CPU path is cheap, so moving reads to the critical
  link path loses.
- **Link latency matters for background copies.** The probe's per-MB copy cost correlates at ρ = −0.77 with the
  two-read state's gain.

## Finding 4: per-problem structure is real but small

- **Problems rank the same on every machine.** For the oracle gains, Kendall's W across machines is:
  - MIN copied once: 0.77 (19 machines × 20 problems) at 11%, 0.82 at 25%;
  - copies made ahead: 0.75.

  For the deployed set copied in the step it is 0.24, which is noise.
- **The routing trace predicts which problems gain.** A problem's gain correlates with its trace excess, the deployed
  policy's reads over MIN's, at ρ = +0.88 (MIN copied once, 11%) and +0.89 (copies made ahead).
- **Machines dominate anyway.** In the log gain, machines explain 93–96% of the variance and problems 1–4%. Per-problem
  gains span only 1.18–1.23×.

## Dead ends (stop pursuing)

- **Non-uniform slots per layer.** Allocated by marginal reads on half the problems and tested on the other half, they
  cut the deployed policy's reads by only:
  - gpt-oss: 0.3% at 11%, 0.8% at 25%;
  - Qwen3: 0.4% at 12.5%, 1.5% at 25%;
  - MIN at most 3.5%.

  Layers differ, but not enough to matter.
- **Tuning the deployed policy's own reads further.** By finding 1 its reads already run at B_host. The only levers are
  fewer reads (needs foresight) or overlap (needs one layer of prediction).

## What this changes in the paper (no GPU needed)

1. **New central result.** Section 4 becomes "time = GPU compute in series with host reads". State it as one law with
   G measured from profiles, not fitted, and validate it on 30 launches. It replaces the Shapley accounting as the
   headline (Shapley moves to the appendix).
   - This answers the reviewers' three standing objections: clarity, an unmeasured "rest", and a model that is
     "bookkeeping with a fitted constant".
2. **Two bounds.** Add the demand bound (sum form) beside the max-form bound. Demand-driven systems are judged against
   the first; the second is what foresight can reach. The bound-tightness objection is then answered directly: our
   cache is at about 65% of the demand bound, and the oracle with copies made ahead is at it.
3. **Reframe foresight.** Present long-horizon foresight (fewer reads) and one-layer foresight (overlap) as separate
   resources, with the layer-ahead recall above.

## Plan (credit about $16.4: the $15 added plus $1.41 left)

| # | Step | Time | Rental | Kill / success criterion |
|---|---|---|---|---|
| 1 | Rewrite Sections 1, 3 and 4 around the sum law and the demand bound; Shapley to the appendix | 1 day | $0 | — |
| 2 | Pre-registered test of the sum law: Nsight profile of G on 2 hosts with the current engine, then on 3 new machines predict T for the deployed and prefetched states from probe + replayed reads + profiled G before any timed run | ½ day + 3 h rental | ~$2 | Hold if the median error is ≤ 5% |
| 3 | **Idle-bus prefetch:** compute layer l+1's router on layer l's output (one small GEMV), send the top predicted non-resident experts to the CPU helpers, and have them read those experts into LLC while the GPU computes. Commit the simulated gain per host (8–19%) as the prediction, then run on 4 hosts spanning B_host 30–90 GB/s at gpt-oss 11% and 25% | 2–3 days engineering + CPU tests on the tiny model | ~$4–5 | Stop if the first host recovers < 30% of its simulated gain. Success: a deployable speed-up with a predicted ceiling |
| 4 | Fewest-admission set on 3–4 more machines (2 slow-link), plus the fewest-admission set with copies made ahead (the best oracle with the fewest admissions) | 1 day | ~$4 | Widens finding 2 from 3 machines to 6–7 |
| 5 | Number check, two blind reviews, grade | ½ day | $0 | — |

- **Order:** 1 → 2 → 3 → 4 → 5. Step 3 is the one that can lift significance from 2–3 to 3–4. Steps 1–2 are what make
  the paper clearer and the model non-circular.
- **Reserve:** about $5 for one failed host or a re-run, as jobs 103 and 103c needed.
