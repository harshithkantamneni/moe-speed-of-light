# Research proposal (ETH AI Center Doctoral Fellowship)

*Draft, about 1,000 words. Adapt the length to the form's limit. Suggested co-supervision: Prof. Torsten Hoefler
(SPCL, systems and performance modelling) with a PI working on language-model architecture or efficient ML from a
different department. Check the current AI Center faculty list and each PI's recent papers before naming one.*

## Title

**Hardware-constrained sparse models: from a validated speed-of-light to offload-aware Mixture-of-Experts design**

## Motivation

Mixture-of-Experts (MoE) models activate a few billion parameters per token but store tens to hundreds of billions.
On the machines most people own, one GPU with 8–32 GB and a host with 32–256 GB of DRAM, the stored parameters do not
fit next to the compute, so every decoded token moves expert weights across a memory hierarchy. Whether frontier-level
models become usable on such hardware is therefore a data-movement question as much as a modelling one.

The systems literature answers it with mechanisms (expert caches, CPU execution of misses, prefetching), and reports
speed-ups over baselines. My preliminary work shows that this evidence is weaker than it looks. In an audit of 147
published measurements from 41 systems against a validated decode model and a new lower bound on decode time, 16 of
20 adjudicable llama.cpp baselines fell below the band of the predicted equal-memory llama.cpp configuration, only 8
of 22 claimed gains survived against that prediction (4–10 across the predictor's error on our own hardware), and the median system reached 24% of its hardware's physical
speed-of-light. Routing
traces taken on dataset text, the standard input of these studies, overstated cache locality on all nine models I
traced. We lack a trustworthy yardstick, and without one we cannot tell which model or system designs actually help.

## Preliminary work

*Seconds, Not Blocks* (paper, code, traces and audit set released): (i) exact MoE routing traces collected by
streaming one decoder layer at a time, verified token-for-token against reference decode on seven architectures;
(ii) a bytes-over-bandwidth decode model with 16% cross-validated median error on third-party data from nine
sources, tested on pre-registered first-party and anchor runs; (iii) a lower bound on
decode time for any exact-routing placement policy, per layer or pooled; (iv) a pre-registered audit and a
trace-provenance study.

## Research questions and aims

**Aim 1: Bounds and models for the regimes that matter next.** Extend the model and the bound from one request on a
PCIe machine to (a) batched and speculative decoding, where expert reuse across requests and draft tokens changes the
data-movement structure; (b) unified-memory machines (GH200/GB10-class superchips, high-bandwidth APUs, Apple
silicon) and CXL-attached memory, where the "GPU budget" becomes a bandwidth split rather than a capacity split.
Validation follows the pre-registration protocol of my preliminary work, on cloud instances and on CSCS Alps.
*Question:* how close can a placement or scheduling policy come to the bound, and which hardware parameter moves the
bound most per unit cost?

**Aim 2: Offload-aware model design.** Invert the model: treat expert count and size, top-k, shared experts and
routing locality as design variables, and the bound as the objective, subject to a quality constraint. Three steps of
increasing cost: (a) *analysis* of released models: which architectural choices (granularity, shared experts, layer
count) predict decode speed under a memory budget, using exact traces and the validated model; (b) *training-free
interventions* (expert reordering, merging, router temperature, cross-layer placement) evaluated by bound and by
quality; (c) *small-scale training* of MoE models with locality- or placement-aware routing objectives, with scaling
laws fitted to both loss and bound-predicted speed. A recent pre-registered study ("Cacheable by Design?") found that locality losses on
the router of 137M-parameter MoE models could not cut misses within a 1% perplexity budget. The bound gives a sharper
target than miss counts, because it prices CPU and GPU bandwidth jointly and shows when aggregation, not caching, is
optimal, so a design can trade a little locality for a lot of speed, or the reverse.
*Question:* at equal quality and memory, how much faster can a model designed for a consumer hierarchy decode than one
designed for a datacenter?

**Aim 3: Scientific benchmarking for sparse inference.** Turn the reporting contract of my preliminary work into a
community benchmark: released traces of models' own text, a calculator that turns any reported result into a
fraction of the speed-of-light and a strong baseline, and a maintained, re-checked audit of new systems. This follows
the methodology Hoefler and Belli proposed for HPC, adapted to ML systems, where evaluation contracts are still
informal.

## Methods

Performance modelling (roofline-style bounds, offline-optimal caching, convexity arguments); exact trace collection;
trace-driven simulation; kernel- and runtime-level measurement on GPUs and CPUs; pre-registered hypotheses with
committed predictions; small-scale MoE training (up to a few billion parameters) on shared infrastructure.

## Expected outcomes and timeline

- **Year 1.** Aim 1(a–b) models and bounds; multi-platform validation; the benchmark release of Aim 3.
  Venues: MLSys, SC, ISPASS.
- **Year 2.** Aim 2(a–b): analysis of released MoE families and training-free interventions. Venues: ICML/NeurIPS
  (efficient ML), ASPLOS.
- **Years 3–4.** Aim 2(c): offload-aware MoE training and scaling laws; a model family designed for a consumer memory
  hierarchy, released openly.

## Why ETH and the AI Center

SPCL's work on data movement, sparsity and scientific benchmarking is the direct foundation of Aims 1 and 3; the
AI Center's cross-department structure is what Aim 2 needs, since it sits between systems and model design. CSCS
Alps offers exactly the unified-memory superchips Aim 1 targets. The long-term goal is accessible AI: frontier-level
intelligence on hardware that individuals and small institutions can own.
