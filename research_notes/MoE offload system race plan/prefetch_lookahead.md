# Next-layer prediction and prefetch (28 Sep 2026)

## Question

Our cache's largest remaining gap is its hit rate. At the three home-PC budgets it misses 1.8 / 1.0 / 0.5 experts per
layer and token, while the best possible policy (MIN-bypass) misses 0.95 / 0.39 / 0.14. Can a layer's experts be
predicted one layer early, accurately enough and early enough to copy them into the GPU before the layer needs them?

## Measurement (job `063_lookahead@vast`)

`ec-bench --lookahead` records every decode step's selected experts per layer, and the top-8 experts that layer l+1's
router would pick from three earlier states. Own text, 12 prompts × 128 decode tokens, teacher-forced.

The states:
- `mid(l)`: the residual after layer l's attention, available when layer l's MoE starts;
- `out(l)`: layer l's output;
- layer l+2's router applied to `mid(l)`.

A host recomputation of each layer's own routing from its own residual matches the router's selection on 100% of
layer-steps for both models. That checks the extraction of router, norm and bias weights and the host arithmetic.

Share of layer l+1's selected experts found among the top-n predictions:

| Model | from mid(l), top-4 | from mid(l), top-8 | from out(l), top-4 | from out(l), top-8 | l+2 from mid(l), top-4 | l+2, top-8 |
|---|---|---|---|---|---|---|
| gpt-oss-20b (24 layers, 32 experts, top-4) | 87.1% | 97.7% | 92.0% | 99.2% | 81.3% | 94.7% |
| gpt-oss-120b (36 layers, 128 experts, top-4) | 84.2% | 96.7% | 89.3% | 98.6% | 77.1% | 92.5% |

This agrees with the literature for other models: about 90% recall at k beyond the first layers on Qwen3-30B-A3B
(Speculating Experts, arXiv 2603.19289) and 97% with confidence thresholds (Fate, arXiv 2502.12224). It is not new; it
is the input the simulation below needs.

## Simulated effect on the deployed cache (`scripts/sim_prefetch.py`)

**The cache simulation** is the deployed per-layer DFA policy: κ = 1, half-life 16, publication two steps later, CPU
misses. It reproduces the measured hit rates of job 060 on gpt-oss-120b:
- simulated 0.542 / 0.757 / 0.876;
- measured 0.541 / 0.758 / 0.875.

**The prefetch rule:** before layer l runs, copy up to q of the top-n `mid(l-1)` predictions that are not resident
into the lowest-score slot. It never evicts an expert predicted for this step, and takes over an admission already in
flight.

gpt-oss-120b, misses per layer-step (CPU-executed):

| C of 128 | Deployed | q=1 from top-4 | q=2 from top-4 | q=1 from top-8 | Oracle q=1 | Oracle q=2 |
|---|---|---|---|---|---|---|
| 14 | 1.832 | 1.014 (−45%) | 0.609 (−67%) | 0.979 (−47%) | 0.864 (−53%) | 0.308 |
| 32 | 0.970 | 0.454 (−53%) | 0.293 (−70%) | 0.414 (−57%) | 0.323 (−67%) | 0.086 |
| 56 | 0.497 | 0.209 (−58%) | 0.151 (−70%) | 0.172 (−65%) | 0.114 (−77%) | 0.023 |

Full rows are in `prereg/homepc/prefetch_sim_120b.json`. At C = 32 with q = 1 from the top-4, 0.55 experts are
prefetched per layer-step, of which 0.42 are used in that step.

Gating a prefetch on the decayed score (only if the expert's score is at least 0.5× or 1× the victim's) halves the
benefit. Ungated is better.

### Time: a layer timeline with the measured home-PC constants

**Fitted constants (host 1, job 059).** Measured tok/s against simulated misses is linear:
- 4.89 ms per token plus 215 µs per miss;
- 215 µs is 13.25 MB / 62 GB/s, the helpers' DRAM bandwidth.

Host 2 (job 060) gives 5.72 ms plus 286 µs, i.e. about 46 GB/s: slower memory.

**The timeline per layer:**
- attention 80 µs;
- then the MoE phase: GPU hits 48 µs, plus CPU misses at 62 GB/s, or 37 GB/s while the link also reads DRAM.
- The link carries the prefetch at 57 GB/s alone, 38 GB/s alongside the helpers (all measured by `concur.cu`).
- The prefetch for layer l+1 is issued when layer l's MoE phase starts. Layer l+1 waits for it (a stall) if it has not
  landed.
- The timeline reproduces the measured base within 2.7%.

gpt-oss-120b, host 1, tok/s change against the deployed cache:

| C | Prefetch copy on a second stream, q=1 top-4 | q=2 top-4 | q=1 top-8 | Oracle q=1 | Copy on the main stream, q=1 top-4 |
|---|---|---|---|---|---|
| 14 | +19% | +18% | +17% | +30% | +7% |
| 32 | +18% | +16% | +8% | +32% | +1% |
| 56 | +13% | +11% | −2% | +27% | −2% |

**Readings:**
- **The copy must overlap GPU work** (attention, hits), i.e. run on a second stream. On the main stream, placed where
  FETCH's copy runs, it overlaps only the helpers and gains nothing: the DRAM is already busy then.
- **q = 1 from the top-4 is the best practical setting.** More or broader prefetches cost more link time than they save.
- **Letting a late prefetch fall back to the CPU is worse than waiting for it**: the copy keeps competing for DRAM.
- **For comparison, FETCH measured +2–6% on host 1.** Prefetch is predicted to be worth about 3× more there.
- **Not modelled:** FETCH and prefetch together; prefetch two layers ahead (it would give more lead time at lower
  accuracy); the extra GPU work of the prediction (one 2880 × 128 GEMV and a top-k per layer, a few µs).

## Implementation (llama.cpp branch `ec-4da6337`, commit 6dd2d1f)

- **`LLAMA_EC_PREFETCH=q`**, with `LLAMA_EC_PREFETCH_TOP` (default k) and `LLAMA_EC_PREFETCH_BLOCKS` (default 64).
  Mailbox mode only.
- **Prediction.** Layer l+1's router applied to layer l's MoE input. Both layers normalize the same residual, so layer
  l+1's input is layer l's times the ratio of the two norm weights. That ratio is folded into a precomputed F32 matrix
  per layer pair (36 × 2880 × 128 × 4 B = 53 MB of GPU memory for gpt-oss-120b, 0.2% of the largest expert budget).
  Then the bias and a top-k.
- **`PREFETCH` (plan, main stream).** Takes the host's maps and eviction candidates for layer l+1 and picks victims as
  in the simulation. Writes updated maps and candidates, which layer l+1 then uses for its lookups and for FETCH, plus
  a log for the host.
- **`PREFETCH_COPY`**, placed after layer l's REQUEST so the helpers start first. Forks a side stream with an event and
  runs the zero-copy kernel from pinned host memory there.
- **`PREFETCH_JOIN`**, in layer l+1 just before its GPU experts. The main stream waits on the copy's event. The plan
  tensor is a source of the join, so its memory stays allocated until the side stream has read it.
- **Host.** Applies the prefetch logs before counting hits, so a used prefetch counts as a hit. Stats: `prefetches`,
  `prefetch_useful`.
- **Fix found while doing this.** A background admission whose expert was meanwhile fetched (FETCH) or prefetched into
  another slot is now dropped when it would be published. Before, the same expert could occupy two slots.

**Verification so far:**
- It compiles for sm_120.
- The CPU path is unchanged: identical dumps against stock on the tiny model; prefetch stays off without the mailbox.
- **Job 065** (gpt-oss-20b, C = 8 of 32, a cheap GPU) is the first GPU run. It checks:
  - `LLAMA_EC_CHECK=1`: the ids the graph used must equal the host maps after the prefetch log;
  - top-1 agreement and NLL against the cache without prefetch;
  - CUDA graphs on and off giving the same output;
  - `compute-sanitizer` memcheck.
- **Simulation's prediction for job 065:** hit rate 0.639 → 0.816 with q = 1 (0.72 prefetches per layer-step, 0.61
  used), 0.885 with q = 2.
