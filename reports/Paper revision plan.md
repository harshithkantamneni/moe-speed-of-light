# Paper revision plan

*29 September 2026. Built from the novelty recheck, two independent reviews (both "weak reject as it stands, accept
after the fixes") and the night's measurements. Paths: `reports/MoE offload paper novelty recheck.md`,
`research_notes/Paper review 28 Sep/`.*

## The paper after the revision, in one paragraph

On a PC, a large Mixture-of-Experts model whose experts do not fit in the GPU decodes at a speed set by one number you
can measure in a second: the bandwidth of host memory. We show this three ways:
- **A bound.** A lower bound in seconds, valid for every exact-routing policy.
- **A law.** Time per token = a GPU constant + the expert bytes read from host memory ÷ that bandwidth. Using each
  host's own microbenchmarks it predicts 42 configurations of llama.cpp, an expert cache, FreeToken-style miss splitting
  and prefetching, on four machines, within 2–6% median error out of sample.
- **An audit** of the published speed-ups. Measured against the bound and against a properly configured llama.cpp,
  most published gains are baseline gains.

The law explains:
- why "identical" rentals differ by 20%;
- why miss splitting helps on one PC and hurts on another;
- why prefetching never reduces the bytes that matter.

It also points to the one thing no system does: overlapping the host-memory reads with the GPU's own work, worth
+40–55% at a quarter of the experts resident.

Working title: **"Your RAM Decides: A Bound, a Law and an Audit for Offloaded Mixture-of-Experts Decode on PCs"**
(alternative, neutral: *"How Fast Could It Be? Bounding and Explaining Offloaded MoE Decode on Consumer Hardware"*).

## Contributions (revised; each narrowed as the novelty recheck requires)

1. **A lower bound in seconds** for batch-1 exact-routing decode with a per-layer or pooled GPU expert budget, misses
   fetched or run on the CPU, demand or prefetching.
   - In resource form it is valid for every policy. A layer-structured refinement and a demand-only corollary follow.
   - It draws on Belady/MIN-bypass, Cao 1995, Albers et al. 2000, Jain & Lin 2018 and CHOPT, and is set apart from
     "Budgeting Bytes", "Paging the Experts" and WiSP.
   - Physical (datasheet) bounds only; measured-bandwidth figures are called references.
2. **The host-memory law**, which is new since 29 Sep. It uses one fitted constant per engine and no fitted bandwidths,
   and is validated out of sample across hosts. It gives the first published numbers for CPU + DMA contention on host
   memory, correcting FreeToken's full-contention model (5 vs 37 GB/s left to the CPU).
3. **The audit.** Its claims are narrowed:
   - "no speed-up is measured against an equal-memory `--n-cpu-moe` baseline";
   - era-aware labels;
   - three tiers of verdict;
   - the pooled bound for pooled systems;
   - new rows (WiSP, SAEM, SPICE in exact mode).
4. **Trace provenance**, scoped to routing and locality studies and citing "Myth of Expert Specialization". The
   gpt-oss-120b result is stated as 6.40 vs 3.00 nats.
5. **A same-machine protocol.** A competitor's unmodified benchmark client is the referee, at measured equal memory,
   against the bound, with the host measured. It is demonstrated on llama.cpp, the expert cache and FreeToken.
6. **The overlap result**, if job 071 or a follow-up shows it: moving host-memory reads into the GPU's own time, as a
   test of the law's prediction, not as a system claim. FETCH and PREFETCH are credited to FreeToken, HybriMoE, DALI
   and Speculating Experts.

## Done (28–29 Sep)

- **Novelty recheck:** 4 researchers plus a report. **Two reviews:** about 22k words, line-level.
- **The bound, corrected** (`mosl/bounds.py`, 5 tests):
  - resource form, layered refinement, demand-only corollary;
  - pooled and warm-start M*; exact LP solutions.
- **Erratum:** the home-PC analysis counted dense bytes at 16 bits. Fixed; the progress log is corrected.
- **The law**, validated on four hosts (the table above).
- **All-in-VRAM reference:** RTX PRO 6000, 265 tok/s.
- **Built:** PREFETCH (a correct implementation, measured +10–16% on long text), `PREFETCH_LATE` and `LLC`
  (overlap), and llama-server support (`-ot` pinned host memory, fit-pass fix).

## Experiments still needed, in order

| # | Experiment | Why | Cost | Status |
|---|---|---|---|---|
| 1 | Profile of the offloaded configurations (job 069, rerun) | Break G into kernels, host gaps and EC overhead; see whether the host-memory phases really idle during G | ≈ $1 | running |
| 2 | Overlap test (job 071) | Test the law's max-form prediction; decides contribution 6 | ≈ $1 | running |
| 3 | Chat comparison v2: 30 AIME problems × 3 launches, greedy decoding, warm-up on a different problem, measured VRAM, FreeToken tuned (`ft bench bw` profile, `--moe-hybrid-max-fetch` sweep incl. 0), paired bootstrap CIs | The comparison as it stands has one request per prompt and no CIs | ≈ $6–9 | next |
| 4 | Speed limit on the evaluated text: trace the AIME outputs and compute the pooled bound for every system | The current denominators use the own-text trace | CPU only | next |
| 5 | Second model: Qwen3.6-35B-A3B (FreeToken's own) or Qwen3-30B-A3B BF16 | One-model objection | ≈ $4–5 | after 3 |
| 6 | Host lottery: 5–8 rentals, predictions registered per host from the law before each run | Turns the law's host prediction into a registered test | ≈ $5–7 | after 5 |
| 7 | PCIe 4.0 host with the law's prediction, where FETCH and PREFETCH lose | Shows the law predicts the sign flip | ≈ $1 | with 6 |
| 8 | CPU DRAM read counters during an A100-anchor-style run | Settles the A1 "beaten floor" deviation (a bandwidth-basis error, not a byte-count error) | ≈ $1 | later |

Items 1–4 fit in the rest of the first $25 tranche. Items 5–7 need part of the second.

## Writing, in order

1. Fix the three false sentences and five imprecise ones; add the 33 missing references (recheck, table
   "Three draft sentences…").
2. New section order:
   1. Introduction (the host-memory thesis)
   2. Bound
   3. Law and host measurements
   4. Audit
   5. Provenance
   6. Same-machine study (with overlap)
   7. Reporting contract, adding concurrent bandwidth, client and tok/s formula, measured VRAM, cold/warm state, decode
      length and host measurement
   8. Limitations

   The A10 case study moves to an appendix, replaced by the consumer-PC study.
3. Figures:
   - the ladder (bound → demand-only → own-decisions ideal → measured) per budget;
   - measured vs law scatter across hosts;
   - host-memory bytes per token for each mechanism;
   - fraction-of-bound bars for the audit.
4. Send the audit notes to the 13 audited teams now: the reviews' strongest non-experimental recommendation.

## Decisions for Harshith

1. **Headline:** the host-memory thesis ("your RAM decides") or the neutral "how fast could it be". Both reviews say
   neutral wording and "same-machine study", not "race".
2. **The public GitHub repo** breaks double-blind review at MLSys. Options:
   - make it private (the rented-machine runner then needs a read token on the machines);
   - post the arXiv preprint first under the venue's preprint rules.
3. **The second $25 tranche:** items 5–7 need about $10–15.
