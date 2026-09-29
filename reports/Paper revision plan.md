# Paper revision plan

*29 September 2026. Built from the novelty recheck, two independent reviews (both "weak reject as it stands, accept
after the fixes") and the night's measurements. Paths: `reports/MoE offload paper novelty recheck.md`,
`research_notes/Paper review 28 Sep/`.*

## The paper after the revision (updated 29 Sep, late morning)

**Working title:** "Where the Seconds Go: Offloaded Mixture-of-Experts Decode on PCs, from Speed of Light to
Measured".

On a PC, a large MoE model whose experts do not fit in the GPU decodes 4–5× below its speed of light. The paper
accounts for every millisecond of that gap, then asks which parts any system could recover.

1. **A lower bound in seconds** (the speed of light), valid for every exact-routing policy.
2. **A host-memory law.** Time per token = GPU-side constant + expert bytes read from host memory ÷ that bandwidth.
   - It was predicted blind, on the machine and before any run, for four new hosts (plus four earlier ones). Desktop-
     class hosts come within 2–8%.
   - Its failures mark the regime boundaries:
     - on a 12-channel server (559 GB/s), the CPU phases become latency-bound;
     - FETCH carries a per-layer copy latency;
     - llama.cpp's even split on hybrid Intel cores runs at 45% of the law.
3. **Where the seconds go**, per budget (figure `figures/gap_decomposition.pdf`). At C = 32 the gap from 226 to
   62 tok/s splits into:
   - no overlap (2.0 ms);
   - no foresight (3.6 ms, the largest);
   - the deployed policy (2.2 ms; FETCH recovers it);
   - batch-1 GPU work (1.6 ms);
   - host launch (1.0 ms);
   - residual (1.4 ms).
4. **The value of routing foresight** (figure `figures/foresight.pdf`).
   - Across 9 models, the optimum reads 40% fewer expert bytes than the best online policy (median).
   - Recovering half of that needs about C/k tokens of foresight: W50 ≈ 0.5 (C/k)^1.4, r = 0.975.
   - Next-layer prediction, the field's main tool, reduces no bytes.
   - This extends 2608.07911, which splits the gap but does not study a limited horizon.
5. **Pre-registered negative results, each explained by the decomposition:**
   - overlap by late copies and L3 warming (job 071, −2 to −14%);
   - deferring admissions (job 072, −3 to −4%).
6. **Audit and provenance** (supporting): the published speed-ups against the bound and an equal-memory llama.cpp
   baseline; own-text versus dataset-text traces.

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
| 1 | Profile of the offloaded configurations (job 069c) and host-side timing (job 072) | Break G into kernels, host gaps and EC overhead | ≈ $3 | done: G = GPU work 3.65 ms + host 1 ms + contention 0.5–1 ms |
| 2 | Overlap test (job 071) | Test the law's max-form prediction; decides contribution 6 | ≈ $1 | done: predictions 3 and 4 failed |
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
