# Admission and miss service on the home PC: trace-driven simulation

*28 Sep 2026. Code: `R:scripts/sim_homepc.py` (CPU only). Traces: the S, G and D packs of `R:data/traces_manifest.json`.*

**Labels.** Every tok/s, % of bound and gain below is **[sim]**. Measured routing traces drive an exact cache
simulation, and an explicit time model turns its counts into time. The model's machine constants are
**estimates**: nothing was measured on the target machine. Numbers derived from the constants alone, such as r* and
the per-expert costs, are marked **[est]**. Hit rates and copy counts are exact for the traces, but the tok/s values
are only as good as the constants. Treat the *ratios* between options as the result, not the absolute speeds.

---

## 1. Answer

**What to change first (one change, in two stages).** Serve each step's misses the way FreeToken's hybrid does, but
keep our per-layer cache and decayed-score ranking:
- In every MoE layer-step with m misses, **fetch T[m] of them over PCIe into cache slots**. Take the highest decayed
  score first, run them on the GPU, and cache them at once: no κ barrier and no two-step publication.
- The helpers run the other m − T[m] misses. The existing background DFA admission (κ = 1, half-life 16) stays on
  for those CPU-run misses.
- Evict the resident with the lowest decayed score that is not selected this step.
- The table is the argmin of `max(T·(s/B_P + τ_x) + a, f + (m−T)·(s/B_h + τ_e), m·s/B_D)` from measured constants. At
  the estimated constants:
  - **gpt-oss-120b** (k = 4): **T = [0, 1, 1, 2, 3]** for m = 0…4.
  - **Qwen3-30B-A3B** (k = 8): **T = [0, 1, 1, 2, 3, 3, 4, 5, 6]**.
- Stage 1 is **fetch every miss** (T[m] = m) with decayed-score eviction. It needs only the synchronous fetch path, but
  it is **not** robust. Stage 2 adds the table.

| model | budget | current rule, tok/s | stage 1: fetch every miss, gain | **stage 2: split + background κ = 1, gain** | stage 2, tok/s | % of SoL, current → stage 2 (default constants) |
|---|---|---:|---:|---:|---:|---:|
| gpt-oss-120b | ncmoe32 | 40.8 [37.9–43.6] | +20% [+11%, +28%] | **+32% [+23%, +44%]** | 53.8 [51.8–56.2] | 33% → 44% |
| gpt-oss-120b | ncmoe27 | 63.8 [60.0–68.0] | +17% [+11%, +23%] | **+23% [+16%, +30%]** | 78.4 [74.7–82.5] | 40% → 48% |
| gpt-oss-120b | ncmoe20 | 94.2 [88.5–101.7] | +11% [+6%, +15%] | **+12% [+8%, +16%]** | 105.5 [98.8–114.8] | 58% → 65% |
| qwen3-30b-a3b | 11% | 57.2 [52.3–62.1] | +28% [+16%, +39%] | **+50% [+37%, +67%]** | 85.5 [81.2–90.7] | 29% → 43% |
| qwen3-30b-a3b | 25% | 91.0 [84.5–98.5] | +25% [+16%, +34%] | **+36% [+27%, +47%]** | 123.9 [116.2–132.6] | 38% → 50% |
| qwen3-30b-a3b | 44% | 135.2 [125.6–147.0] | +16% [+11%, +21%] | **+19% [+14%, +24%]** | 160.7 [148.3–178.0] | 56% → 66% |

Gains are the Monte Carlo median [P10–P90] over the expected constant ranges (§6.3). The **P10–P90 range is the
uncertainty to quote**. Stage 2 beat the current rule in 200/200 draws and in every one-at-a-time setting of §6.1;
the worst case was ×1.04, gpt-oss-120b at ncmoe20 with PCIe at 30 GB/s. Stage 1 alone falls to ×0.81–0.96 when PCIe
delivers only 30 GB/s, and is barely break-even on gpt-oss-120b at 40 GB/s (×1.01–1.02). So ship stage 1 only after
measuring B_P ≥ 45 GB/s, or go straight to stage 2.

**Second: leave the admission rule alone.**
- **κ.** κ = 1 is already best, or within 1–2 % of best, for gpt-oss-120b at every budget and in 99–160 of 200
  draws. For Qwen3, κ = 2 → 1 is worth +0–2 % [sim] at 11–25 % and +2 % at 44 %: set κ = 1 on both models for
  simplicity, but expect nothing measurable.
- **Pooling.** A pooled cache adds −2 to +3 % (gpt-oss-120b) and −5 to +6 % (Qwen3) [sim]. It is not worth the rewrite
  in either design.
- **Lower κ.** Do **not** lower κ toward "admit more". On this desktop every admission copy reads the same DRAM the
  helpers need. κ = 0 or admit-all costs up to −10 % / −24 %, unless admissions are paced into idle DRAM time.

**Why.** On this desktop:
- **Fetching costs about the same as CPU execution.** Per expert, a fetch costs p ≈ 270 µs and CPU execution
  b ≈ 300 µs [est, gpt-oss-120b], so r* ≈ 0.8–1.0 against ≈ 11 on the A10 host.
- **A fetch leaves the expert on the GPU for free.** Our CPU path reads each admitted expert from DRAM twice: once by
  the helpers and once by the admission copy.
- **The split aggregates DRAM bandwidth.** CPU helpers alone read about 46 GB/s and PCIe alone about 50 GB/s, but
  their sum is capped by one ~57 GB/s DRAM [est]. Splitting a layer's misses between them gets a layer with ≥ 2
  misses from 46–50 GB/s up toward the DRAM ceiling.

**Against FreeToken's policy** (pooled cache + LRU + balanced split, given our fixed costs):
- It is 2–7 % *below* stage 2 at the small and middle budgets. Decayed-score eviction beats LRU when slots are scarce.
- It is 1–2 % above stage 2 at the largest budget, where pooling helps a little [sim, MC medians].
- With stage 2 the policy gap to FreeToken closes. What remains is fixed cost per layer and per call, which this
  study holds equal.

**What is left.**
- Stage 2 reaches 43–66 % of the speed-of-light bound, against 29–58 % for the current rule [sim].
- The offline per-layer Belady-fetch ceiling is 14–22 % above stage 2 on gpt-oss-120b and 2–12 % on Qwen3; the pooled
  one is 17–27 % and 9–17 % above [sim]. That headroom needs prediction or prefetch, not a better admission rule.

---

## 2. Setup

### 2.1 Traces and budgets

- **Routing.** Response tokens only; conversations back to back; cache state carried across (as in BASE §6).
  - gpt-oss-120b: S 32.6k tokens / 35 conversations (`036_trace_retry@40gb`), G 33.1k / 35, D 86.2k / 160.
  - Qwen3-30B-A3B (FP8-checkpoint own text; routing is precision-independent to first order): the first ≈ 40k
    response tokens of S (45 conversations), G (45) and D (68), capped for runtime.
  - S (own sampled text) is the headline. G is a replicate, D is a dataset-text contrast.
- **Budgets.**
  - gpt-oss-120b: llama.cpp `-ncmoe 32 / 27 / 20` of 36 layers leaves 4 / 9 / 16 layers' worth of experts on the GPU
    (512 / 1152 / 2048 slots). That is **C = 14 / 32 / 57 per layer** (11 / 25 / 44 %). Pooled designs get L·C slots.
  - Qwen3 uses the same fractions: C = 14 / 32 / 57 of 128.
- **Time-model sizes.** gpt-oss-120b MXFP4: s = 13.25 MB per expert, D = 1.69 GB dense + LM head. Qwen3-30B-A3B
  Q4_K_M: s = 2.86 MB, D = 0.82 GB. Both from `R:data/gguf_bytes.json`.

### 2.2 Options simulated

One engine (`_engine` in the script, numba, lazy heaps) runs every option on the interleaved stream (row = token ×
layer). Its partitions are either per-layer (C slots each) or one pool (L·C).

- **Policies:**
  - LRU;
  - DFA: decayed score with half-life h; admit iff score(e) > score(victim) + κ; victims exclude experts selected
    this step and experts still loading;
  - MIN: next use, as MIN-bypass or Belady.
- **Miss service:**
  - (a) *CPU*: the helpers run every miss; admissions are background copies published two steps later, which is the
    deployed system.
  - (b) *fetch every miss*: copy, then run on the GPU; demand paging.
  - (c) *split*: fetch T[m], CPU the rest. Fetched experts are cached immediately. CPU-run ones are either not cached,
    or admitted in the background by DFA κ = 1 ("split + background").
- **References:** per-layer LFU (cumulative counts, `cachesim`, no publication delay) and the hindsight top-frequency
  static set.

**Validation** (`cachesim` / `ecsim_fast` on an 8k-token prefix of gpt-oss-120b S, C = 14 and 32). The engine
reproduces:
- `ecsim_fast` DFA (κ = 1, 2-step delay) misses and admissions **exactly**;
- `cachesim` MIN-bypass, Belady and pooled MIN-bypass **exactly**;
- LRU-fetch within 0.05 % (tie order only).

### 2.3 Time model (per decode token)

```
T  = TD + Σ_layers [ O + max(G, C, F, Dr) ] + n_adm·(τ_host + θ·s/B_P)          per-row max, summed over layers
TD = D/(η_g·B_G) + L·t_lay + t_tok        a = s/(η_g·B_G) + t_eg
G  = (k − mc)·a                           GPU: hits + fetched experts
C  = [mc>0]·(f + mc·(s/B_h' + τ_e))       helpers: per-request fixed + per-expert bytes + per-expert fixed
F  = [mf>0]·(mf·(s/B_P' + τ_x) + a)       critical fetches, each GEMV pipelined behind its copy
Dr = (mc + mf)·s/B_D'                     helpers and PCIe DMA read the same DRAM
```

- **Counts.** mc and mf are the CPU-run and fetched misses of that (token, layer), from the simulation. The
  (mc, mf) histogram makes the mean exact.
- **Background admissions.** They move A bytes per token. Default (**"uniform"**): the copies are spread over the
  token and take r = A/T from DRAM and the link, so B_D' = B_D − r, B_h' = min(B_h, B_D') and B_P' = min(B_P − r, B_D'),
  solved as a fixed point. **"Optimistic"**: the copies fit into idle DRAM time (r = 0 inside layers).
- **Hard limits,** always: T ≥ link bytes / B_P and T ≥ DRAM bytes / B_D.
- **Pessimism knob.** θ is the fraction of each admission copy's link time that lands on the critical path; the A10's
  β was ≈ 0.47 of a copy, and GPU-side interference is otherwise not modelled.
- **The bound.** SoL is `sol_a10.bound` with these constants, zero fixed CPU and link costs, perfect prefetch, and the
  extra shared-DRAM term L·(mc + x)·s/B_D. It takes per-layer MIN-bypass M* (pooled M* reported alongside). Its
  perfect-prefetch assumption makes it loose: no option exceeds 72 % of it.

| constant | default | swept one-at-a-time | Monte Carlo range | basis (all estimates) |
|---|---|---|---|---|
| B_G, GPU memory bandwidth | 1792 GB/s | – | – | RTX 5090 datasheet |
| η_g, batch-1 GPU efficiency | 0.5 | 0.4, 0.6 | U(0.4, 0.6) | task brief |
| t_lay, non-GEMV GPU time per layer | 43 µs (gpt-oss), 34 µs (Qwen3) | – | – | A10 nsys × 0.7 (BASE §5.4 assumption) |
| t_eg, GPU fixed time per expert | 2.9 µs / 0.4 µs | – | – | A10 nsys × 0.7 (BASE §5.4) |
| t_tok, per-token host/launch | 150 µs | – | – | assumption |
| B_D, DRAM read ceiling for all agents together | 57 GB/s | 40, 50, 57, 70 | U(52, 62) | dual-channel DDR5, 54–60 GB/s peak read (brief) |
| B_h, helper read rate | 0.8·B_D = 45.6 GB/s | follows B_D | U(40, 50) | helpers at 78–81 % of read BW on the A10 (BASE §4.4); brief says 40–50 |
| B_P, pinned H2D | 50 GB/s | 30, 40, 50, 52 | U(45, 52) | brief: 49–52; FreeToken's 9950X3D measured 49.0 |
| τ_e, fixed CPU cost per expert | 10 µs | 5, 10, 21 | U(5, 21) | brief; M4's 20.7 µs is the top |
| f, helper per-request fixed | 30 µs (gpt-oss), 15 µs (Qwen3) | 10, 30, 50 | ×U(0.33, 1.67) | A10 50 / 19–26 µs × ~0.6 for 16 fast cores |
| O, per-MoE-layer cache overhead | 24 µs | 10, 24, 38 | U(10, 38) | A10 31–38 µs; brief's 10–38 µs |
| τ_x, fixed cost per critical fetch | 5 µs | – | U(2, 10) | one fused, device-initiated copy per expert (assumption) |
| τ_host, host cost per admission | 3 µs | – | – | policy loop + launch (assumption) |
| θ, admission link time on the critical path | 0 | 0, 0.25, 0.5 | U(0, 0.25) | A10 β/p ≈ 0.47 is the pessimistic end |
| interference model | uniform | uniform vs optimistic (§6.2) | 50/50 | – |

---

## 3. The reuse threshold r*, re-derived for these constants [est]

A copy pays off after r* reuses, **r* = p/(b − a)**, where a, b and p are the GPU, CPU and copy time of one expert.
On a desktop, a copy's real cost when the helpers are DRAM-bound is its DRAM time, which gives
**r*_DRAM = (s/B_D)/(b − a)**.

| model | expert bytes s (MB) | a: GPU per expert (µs) | b: CPU per expert s/B_h + τ_e (µs) | p: fetch s/B_P + τ_x (µs) | r*_link = p/(b−a) | r*_DRAM = (s/B_D)/(b−a) | r*_link over the grid | r*_DRAM over the grid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss-120b | 13.25 | 17.7 | 300.7 | 270.1 | 0.95 | 0.82 | 0.81–1.99 | 0.79–0.85 |
| qwen3-30b-a3b | 2.86 | 3.6 | 72.7 | 62.1 | 0.90 | 0.73 | 0.70–1.91 | 0.60–0.79 |

r*_link for gpt-oss-120b over the sensitivity grid (the two values are τ_e = 5 / 21 µs):

| r*_link (τ_e 5 / 21 µs) | B_P 30 | B_P 40 | B_P 50 | B_P 52 |
|---|---:|---:|---:|---:|
| B_D 40 GB/s | 1.11 / 1.07 | 0.84 / 0.81 | 0.84 / 0.81 | 0.84 / 0.81 |
| B_D 50 GB/s | 1.40 / 1.34 | 1.06 / 1.01 | 0.85 / 0.81 | 0.85 / 0.81 |
| B_D 57 GB/s | 1.61 / 1.52 | 1.21 / 1.14 | 0.97 / 0.92 | 0.93 / 0.88 |
| B_D 70 GB/s | 1.99 / 1.86 | 1.50 / 1.40 | 1.21 / 1.13 | 1.16 / 1.08 |

**Reading.**
- r* ≈ 0.8–1.0 at the expected constants, against ≈ 11 on the A10 (BASE §5.4) and the synthesis's 1.1–1.3.
- r*_DRAM ≈ η_h ≈ 0.8 almost regardless of the constants, because the helpers and the copy both read DRAM.
- r* only exceeds 1.3 when PCIe is ≤ 30–40 GB/s and DRAM is fast.

**What r* means for each design:**
- **CPU-execute design.** An admission is an *extra* copy on top of the miss's CPU execution. It pays only if the
  admitted expert earns about one more reuse than the victim it displaces. In DFA units (decayed count, horizon about
  23 steps at half-life 16), that is **κ ≈ r* ≈ 1**. This is why the κ sweep (§4, §5) peaks at κ = 0.5–1 and why
  κ = 0 or admit-all lose.
- **Fetch and split designs.** r* is moot for fetched misses: the copy *is* the execution path, so caching the
  fetched expert costs nothing and every fetched expert should be cached.

---

## 4. gpt-oss-120b MXFP4 (headline), S arm, default constants [sim]

| option | ncmoe32 (C=14) tok/s | % SoL | vs current | ncmoe27 (C=32) tok/s | % SoL | vs current | ncmoe20 (C=57) tok/s | % SoL | vs current |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *Per-layer, CPU executes every miss (admission rule varies)* | | | | | | | | | |
| **current: per-layer DFA κ=now, CPU misses** | 41.9 | 33% | – | 66.2 | 40% | – | 97.6 | 58% | – |
| DFA, admit every miss (κ=−∞) | 32.4 | 26% | -23% | 58.8 | 35% | -11% | 97.1 | 58% | -0% |
| DFA κ=0 | 41.9 | 33% | -0% | 59.6 | 36% | -10% | 97.1 | 58% | -0% |
| DFA κ=0.5 | 42.1 | 34% | +0% | 64.8 | 39% | -2% | 97.3 | 58% | -0% |
| DFA κ=1 | 41.9 | 33% | +0% | 66.2 | 40% | +0% | 97.6 | 58% | +0% |
| DFA κ=2 | 41.5 | 33% | -1% | 65.0 | 39% | -2% | 93.6 | 56% | -4% |
| DFA κ=4 | 40.3 | 32% | -4% | 58.6 | 35% | -11% | 81.7 | 49% | -16% |
| DFA κ=0.5, half-life 8 | 41.0 | 33% | -2% | 57.4 | 34% | -13% | 96.3 | 58% | -1% |
| DFA κ=0.5, half-life 32 | 41.9 | 33% | +0% | 67.0 | 40% | +1% | 99.3 | 59% | +2% |
| LRU (admit every miss) | 29.7 | 24% | -29% | 54.3 | 32% | -18% | 95.6 | 57% | -2% |
| LFU (cumulative, no publish delay) | 26.3 | 21% | -37% | 39.3 | 24% | -41% | 67.0 | 40% | -31% |
| static top-frequency set (hindsight, no admissions) | 28.7 | 23% | -32% | 40.4 | 24% | -39% | 63.2 | 38% | -35% |
| MIN-bypass (offline; policy ceiling for CPU misses) | 57.1 | 46% | +36% | 92.4 | 55% | +40% | 120.9 | 72% | +24% |
| *Pooled all-layer cache (L·C slots), CPU executes every miss* | | | | | | | | | |
| pooled LRU | 28.7 | 23% | -32% | 55.0 | 33% | -17% | 99.2 | 59% | +2% |
| pooled DFA κ=0 | 41.4 | 33% | -1% | 59.6 | 36% | -10% | 100.3 | 60% | +3% |
| pooled DFA κ=0.5 | 42.2 | 34% | +1% | 64.9 | 39% | -2% | 100.3 | 60% | +3% |
| pooled DFA κ=1 | 42.1 | 34% | +0% | 67.2 | 40% | +2% | 99.8 | 60% | +2% |
| pooled MIN-bypass (offline) | 58.7 | 47% | +40% | 95.8 | 57% | +45% | 123.8 | 74% | +27% |
| *Fetch every miss over PCIe, run it on the GPU (demand paging)* | | | | | | | | | |
| fetch every miss, LRU evict | 46.7 | 37% | +12% | 72.8 | 44% | +10% | 105.5 | 63% | +8% |
| fetch every miss, evict lowest decayed score | 50.1 | 40% | +19% | 76.5 | 46% | +16% | 106.5 | 64% | +9% |
| fetch every miss, Belady (offline) | 63.2 | 50% | +51% | 97.1 | 58% | +47% | 123.7 | 74% | +27% |
| pooled, fetch every miss, LRU | 45.6 | 36% | +9% | 73.6 | 44% | +11% | 108.3 | 65% | +11% |
| pooled, fetch every miss, Belady (offline) | 68.3 | 55% | +63% | 101.5 | 61% | +53% | 126.6 | 76% | +30% |
| *Per-step split: fetch table[m] of the m misses, CPU runs the rest* | | | | | | | | | |
| split, LRU evict | 52.5 | 42% | +25% | 77.6 | 46% | +17% | 107.4 | 64% | +10% |
| split, decayed-score evict, CPU misses not cached | 54.8 | 44% | +31% | 79.9 | 48% | +21% | 108.0 | 65% | +11% |
| split, fetched cached only if score > victim | 53.9 | 43% | +29% | 79.9 | 48% | +21% | 108.0 | 65% | +11% |
| **split + background DFA κ=1 for CPU-run misses** | 54.7 | 44% | +31% | 79.9 | 48% | +21% | 108.1 | 65% | +11% |
| pooled split, LRU (FreeToken-like) | 51.9 | 41% | +24% | 78.3 | 47% | +18% | 109.7 | 66% | +12% |
| pooled split, decayed-score evict | 53.6 | 43% | +28% | 80.6 | 48% | +22% | 110.2 | 66% | +13% |
| pooled split + background DFA κ=1 | 53.6 | 43% | +28% | 80.5 | 48% | +22% | 110.2 | 66% | +13% |
| *Bounds* | | | | | | | | | |
| SoL, per-layer M* (the denominator) | 125.3 | 100% |  | 167.2 | 100% |  | 167.2 | 100% |  |
| SoL, pooled M* | 133.2 | |  | 167.2 | |  | 167.2 | |  |

Hit rate and traffic per token (CPU-run experts, critical fetches, background admission copies):

| option | ncmoe32: hit | CPU experts/tok | fetches/tok | bg copies/tok | ncmoe27: hit | CPU experts/tok | fetches/tok | bg copies/tok | ncmoe20: hit | CPU experts/tok | fetches/tok | bg copies/tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **current: per-layer DFA κ=now, CPU misses** | 0.583 | 60.0 | 0.0 | 8.5 | 0.795 | 29.5 | 0.0 | 8.7 | 0.916 | 12.1 | 0.0 | 4.6 |
| DFA κ=0 | 0.587 | 59.4 | 0.0 | 21.3 | 0.799 | 28.9 | 0.0 | 24.5 | 0.920 | 11.6 | 0.0 | 10.5 |
| DFA, admit every miss (κ=−∞) | 0.570 | 61.9 | 0.0 | 53.9 | 0.799 | 28.9 | 0.0 | 25.9 | 0.920 | 11.6 | 0.0 | 10.5 |
| LRU (admit every miss) | 0.520 | 69.1 | 0.0 | 59.0 | 0.775 | 32.5 | 0.0 | 28.6 | 0.916 | 12.0 | 0.0 | 10.9 |
| MIN-bypass (offline; policy ceiling for CPU misses) | 0.762 | 34.3 | 0.0 | 20.7 | 0.904 | 13.9 | 0.0 | 9.9 | 0.967 | 4.8 | 0.0 | 3.8 |
| pooled DFA κ=0.5 | 0.587 | 59.4 | 0.0 | 14.0 | 0.802 | 28.5 | 0.0 | 16.5 | 0.926 | 10.6 | 0.0 | 9.6 |
| pooled MIN-bypass (offline) | 0.776 | 32.3 | 0.0 | 21.0 | 0.913 | 12.6 | 0.0 | 9.5 | 0.972 | 4.1 | 0.0 | 3.4 |
| fetch every miss, LRU evict | 0.590 | 0.0 | 59.0 | 0.0 | 0.801 | 0.0 | 28.6 | 0.0 | 0.924 | 0.0 | 10.9 | 0.0 |
| fetch every miss, evict lowest decayed score | 0.628 | 0.0 | 53.6 | 0.0 | 0.820 | 0.0 | 25.9 | 0.0 | 0.927 | 0.0 | 10.5 | 0.0 |
| fetch every miss, Belady (offline) | 0.742 | 0.0 | 37.1 | 0.0 | 0.900 | 0.0 | 14.4 | 0.0 | 0.966 | 0.0 | 4.9 | 0.0 |
| **split + background DFA κ=1 for CPU-run misses** | 0.623 | 16.9 | 37.5 | 0.1 | 0.816 | 6.6 | 19.9 | 0.0 | 0.925 | 1.9 | 8.9 | 0.0 |
| pooled split, LRU (FreeToken-like) | 0.589 | 18.5 | 40.7 | 0.0 | 0.807 | 6.9 | 20.8 | 0.0 | 0.930 | 1.6 | 8.4 | 0.0 |
| M* per layer-step, per-layer / pooled | 0.953 / 0.897 | | |  | 0.386 / 0.349 | | |  | 0.133 / 0.113 | | |  |
| split table, fetched for m = 0…k misses | [0, 1, 1, 2, 3] | | |  | [0, 1, 1, 2, 3] | | |  | [0, 1, 1, 2, 3] | | |  |

**Reading.**
- **The current rule is well tuned for a CPU-execute design.**
  - No κ, half-life or pooling variant moves it by more than ±2 %.
  - LRU, admit-all DFA and LFU lose up to 29 %, 23 % and 41 % respectively, because every admission copy competes
    with the helpers for DRAM.
  - Even the offline MIN-bypass policy is only +24–40 %.
- **Changing *where* misses run is worth more than changing *which* experts are admitted.**
  - Fetch-every-miss gains +8–19 % while its hit rate is almost the same (0.59–0.63 vs 0.58 at ncmoe32).
  - The split gains +11–31 %.
  - At ncmoe32 the split reaches ~96 % of the offline MIN-bypass ceiling of the CPU design, with an online rule.
- **Decayed-score eviction beats LRU once misses are fetched:** +1–7 %, most at small budgets. It also keeps a
  higher hit rate (0.628 vs 0.590 at ncmoe32).

**Own text vs dataset text (S vs G vs D).** G reproduces S within 1 %. D (dataset text) runs 3–10 % faster for every
option, because its hit rates are higher, and the stage-2 gain is 1–5 points smaller. The S/G
arms are therefore the conservative basis.

| budget | arm (decode tokens, convs) | current: per-layer DFA κ=now, CPU misses | DFA κ=0 | MIN-bypass (offline; policy ceiling for CPU misses) | fetch every miss, LRU evict | split, decayed-score evict, CPU misses not cached | split + background DFA κ=1 for CPU-run misses | pooled split, LRU (FreeToken-like) | SoL | split+bg / current |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ncmoe32 | S (32601, 35) | 41.9 | 41.9 | 57.1 | 46.7 | 54.8 | 54.7 | 51.9 | 125.3 | 1.31 |
| ncmoe32 | G (33072, 35) | 42.2 | 42.1 | 57.4 | 47.0 | 55.1 | 55.0 | 52.1 | 126.3 | 1.31 |
| ncmoe32 | D (86156, 160) | 46.1 | 45.6 | 60.6 | 48.5 | 58.2 | 58.2 | 54.3 | 135.0 | 1.26 |
| ncmoe27 | S (32601, 35) | 66.2 | 59.6 | 92.4 | 72.8 | 79.9 | 79.9 | 78.3 | 167.2 | 1.21 |
| ncmoe27 | G (33072, 35) | 66.6 | 59.9 | 92.7 | 72.9 | 80.2 | 80.1 | 78.4 | 167.2 | 1.20 |
| ncmoe27 | D (86156, 160) | 72.0 | 64.8 | 96.8 | 76.6 | 84.1 | 84.1 | 83.1 | 167.2 | 1.17 |
| ncmoe20 | S (32601, 35) | 97.6 | 97.1 | 120.9 | 105.5 | 108.0 | 108.1 | 109.7 | 167.2 | 1.11 |
| ncmoe20 | G (33072, 35) | 98.0 | 97.4 | 121.1 | 105.6 | 108.3 | 108.3 | 110.0 | 167.2 | 1.11 |
| ncmoe20 | D (86156, 160) | 101.7 | 102.9 | 123.3 | 109.7 | 111.8 | 111.8 | 114.4 | 167.2 | 1.10 |

---

## 5. Qwen3-30B-A3B (Q4_K_M sizes), S arm, default constants [sim]

| option | 11% (C=14) tok/s | % SoL | vs current | 25% (C=32) tok/s | % SoL | vs current | 44% (C=57) tok/s | % SoL | vs current |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *Per-layer, CPU executes every miss (admission rule varies)* | | | | | | | | | |
| **current: per-layer DFA κ=now, CPU misses** | 59.3 | 29% | – | 94.5 | 38% | – | 139.3 | 56% | – |
| DFA, admit every miss (κ=−∞) | 45.2 | 22% | -24% | 88.4 | 35% | -6% | 143.9 | 58% | +3% |
| DFA κ=0 | 59.1 | 29% | -0% | 95.2 | 38% | +1% | 144.0 | 58% | +3% |
| DFA κ=0.5 | 59.3 | 29% | -0% | 95.4 | 38% | +1% | 143.9 | 58% | +3% |
| DFA κ=1 | 59.4 | 29% | +0% | 95.4 | 38% | +1% | 142.7 | 57% | +2% |
| DFA κ=2 | 59.3 | 29% | +0% | 94.5 | 38% | +0% | 139.3 | 56% | +0% |
| DFA κ=4 | 58.7 | 29% | -1% | 90.0 | 36% | -5% | 127.2 | 51% | -9% |
| DFA κ=0.5, half-life 8 | 58.6 | 29% | -1% | 93.4 | 37% | -1% | 142.5 | 57% | +2% |
| DFA κ=0.5, half-life 32 | 59.0 | 29% | -1% | 94.8 | 38% | +0% | 143.2 | 57% | +3% |
| LRU (admit every miss) | 43.2 | 21% | -27% | 80.3 | 32% | -15% | 140.1 | 56% | +1% |
| LFU (cumulative, no publish delay) | 42.7 | 21% | -28% | 56.4 | 23% | -40% | 84.6 | 34% | -39% |
| static top-frequency set (hindsight, no admissions) | 38.0 | 19% | -36% | 48.4 | 19% | -49% | 69.7 | 28% | -50% |
| MIN-bypass (offline; policy ceiling for CPU misses) | 81.1 | 40% | +37% | 130.7 | 52% | +38% | 169.5 | 68% | +22% |
| *Pooled all-layer cache (L·C slots), CPU executes every miss* | | | | | | | | | |
| pooled LRU | 40.5 | 20% | -32% | 79.5 | 32% | -16% | 145.5 | 58% | +4% |
| pooled DFA κ=0 | 58.7 | 29% | -1% | 95.8 | 38% | +1% | 148.0 | 59% | +6% |
| pooled DFA κ=0.5 | 59.1 | 29% | -0% | 96.2 | 39% | +2% | 147.8 | 59% | +6% |
| pooled DFA κ=1 | 59.4 | 29% | +0% | 96.6 | 39% | +2% | 146.8 | 59% | +5% |
| pooled MIN-bypass (offline) | 83.0 | 41% | +40% | 135.2 | 54% | +43% | 173.4 | 69% | +25% |
| *Fetch every miss over PCIe, run it on the GPU (demand paging)* | | | | | | | | | |
| fetch every miss, LRU evict | 70.7 | 35% | +19% | 110.4 | 44% | +17% | 157.5 | 63% | +13% |
| fetch every miss, evict lowest decayed score | 75.1 | 37% | +27% | 116.9 | 47% | +24% | 160.1 | 64% | +15% |
| fetch every miss, Belady (offline) | 87.8 | 43% | +48% | 140.9 | 56% | +49% | 175.5 | 70% | +26% |
| pooled, fetch every miss, LRU | 66.6 | 33% | +12% | 109.9 | 44% | +16% | 162.0 | 65% | +16% |
| pooled, fetch every miss, Belady (offline) | 99.0 | 49% | +67% | 147.9 | 59% | +57% | 179.0 | 72% | +29% |
| *Per-step split: fetch table[m] of the m misses, CPU runs the rest* | | | | | | | | | |
| split, LRU evict | 83.4 | 41% | +41% | 122.3 | 49% | +29% | 162.6 | 65% | +17% |
| split, decayed-score evict, CPU misses not cached | 86.7 | 43% | +46% | 125.9 | 50% | +33% | 163.7 | 66% | +18% |
| split, fetched cached only if score > victim | 83.3 | 41% | +40% | 124.4 | 50% | +32% | 163.7 | 66% | +18% |
| **split + background DFA κ=1 for CPU-run misses** | 86.5 | 43% | +46% | 125.9 | 50% | +33% | 163.8 | 66% | +18% |
| pooled split, LRU (FreeToken-like) | 80.4 | 40% | +36% | 122.6 | 49% | +30% | 165.8 | 66% | +19% |
| pooled split, decayed-score evict | 82.6 | 41% | +39% | 125.1 | 50% | +32% | 166.6 | 67% | +20% |
| pooled split + background DFA κ=1 | 82.5 | 41% | +39% | 125.1 | 50% | +32% | 166.6 | 67% | +20% |
| *Bounds* | | | | | | | | | |
| SoL, per-layer M* (the denominator) | 202.6 | 100% |  | 249.8 | 100% |  | 249.8 | 100% |  |
| SoL, pooled M* | 213.4 | |  | 249.8 | |  | 249.8 | |  |

| option | 11%: hit | CPU experts/tok | fetches/tok | bg copies/tok | 25%: hit | CPU experts/tok | fetches/tok | bg copies/tok | 44%: hit | CPU experts/tok | fetches/tok | bg copies/tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **current: per-layer DFA κ=now, CPU misses** | 0.562 | 168.3 | 0.0 | 7.8 | 0.793 | 79.5 | 0.0 | 8.8 | 0.922 | 29.9 | 0.0 | 4.9 |
| DFA κ=0 | 0.562 | 168.3 | 0.0 | 30.6 | 0.799 | 77.1 | 0.0 | 37.5 | 0.933 | 25.6 | 0.0 | 20.9 |
| DFA, admit every miss (κ=−∞) | 0.490 | 196.0 | 0.0 | 154.9 | 0.796 | 78.3 | 0.0 | 65.8 | 0.933 | 25.6 | 0.0 | 21.8 |
| LRU (admit every miss) | 0.460 | 207.3 | 0.0 | 161.4 | 0.762 | 91.4 | 0.0 | 74.5 | 0.926 | 28.2 | 0.0 | 23.8 |
| MIN-bypass (offline; policy ceiling for CPU misses) | 0.744 | 98.5 | 0.0 | 58.8 | 0.906 | 36.1 | 0.0 | 25.4 | 0.973 | 10.2 | 0.0 | 8.2 |
| pooled DFA κ=0.5 | 0.561 | 168.7 | 0.0 | 22.5 | 0.802 | 76.1 | 0.0 | 25.8 | 0.940 | 23.1 | 0.0 | 16.4 |
| pooled MIN-bypass (offline) | 0.757 | 93.5 | 0.0 | 59.7 | 0.915 | 32.6 | 0.0 | 24.3 | 0.978 | 8.3 | 0.0 | 7.0 |
| fetch every miss, LRU evict | 0.578 | 0.0 | 162.0 | 0.0 | 0.806 | 0.0 | 74.6 | 0.0 | 0.938 | 0.0 | 23.8 | 0.0 |
| fetch every miss, evict lowest decayed score | 0.614 | 0.0 | 148.4 | 0.0 | 0.829 | 0.0 | 65.7 | 0.0 | 0.943 | 0.0 | 21.9 | 0.0 |
| fetch every miss, Belady (offline) | 0.698 | 0.0 | 116.1 | 0.0 | 0.899 | 0.0 | 38.9 | 0.0 | 0.972 | 0.0 | 10.6 | 0.0 |
| **split + background DFA κ=1 for CPU-run misses** | 0.610 | 49.1 | 100.5 | 0.7 | 0.824 | 20.5 | 47.2 | 0.1 | 0.941 | 5.7 | 17.1 | 0.1 |
| pooled split, LRU (FreeToken-like) | 0.561 | 55.1 | 113.6 | 0.0 | 0.809 | 22.3 | 50.9 | 0.0 | 0.945 | 4.9 | 16.0 | 0.0 |
| M* per layer-step, per-layer / pooled | 2.051 / 1.948 | | |  | 0.753 / 0.679 | | |  | 0.214 / 0.173 | | |  |
| split table, fetched for m = 0…k misses | [0, 1, 1, 2, 3, 3, 4, 5, 6] | | |  | [0, 1, 1, 2, 3, 3, 4, 5, 6] | | |  | [0, 1, 1, 2, 3, 3, 4, 5, 6] | | |  |

**Reading.**
- **The same pattern holds, with larger gains.** The Qwen3 split gains +18–46 % [sim] because:
  - 8 experts per layer-step give more layers with m ≥ 2, where the split aggregates DRAM bandwidth;
  - 48 layers pay more helper per-request cost f, which fetching avoids.
- **At 11 % the split beats the offline MIN-bypass ceiling of the CPU design** (86.5 vs 81.1 tok/s).
- **Pooling matters more for Qwen3,** but only in the CPU design at 44 %: +5–6 %.
- **Current κ = 2 is within 0–3 % of the best κ** (0–1).

| budget | arm (decode tokens, convs) | current: per-layer DFA κ=now, CPU misses | DFA κ=0 | MIN-bypass (offline; policy ceiling for CPU misses) | fetch every miss, LRU evict | split, decayed-score evict, CPU misses not cached | split + background DFA κ=1 for CPU-run misses | pooled split, LRU (FreeToken-like) | SoL | split+bg / current |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 11% | S (40225, 45) | 59.3 | 59.1 | 81.1 | 70.7 | 86.7 | 86.5 | 80.4 | 202.6 | 1.46 |
| 11% | G (40364, 45) | 59.2 | 58.8 | 80.9 | 70.3 | 86.4 | 86.2 | 80.0 | 201.8 | 1.46 |
| 11% | D (40482, 68) | 62.2 | 62.2 | 84.7 | 73.8 | 89.9 | 89.7 | 83.8 | 217.1 | 1.44 |
| 25% | S (40225, 45) | 94.5 | 95.2 | 130.7 | 110.4 | 125.9 | 125.9 | 122.6 | 249.8 | 1.33 |
| 25% | G (40364, 45) | 94.2 | 94.8 | 130.5 | 109.8 | 125.6 | 125.6 | 122.0 | 249.8 | 1.33 |
| 25% | D (40482, 68) | 97.2 | 98.2 | 133.7 | 114.4 | 128.5 | 128.5 | 125.8 | 249.8 | 1.32 |
| 44% | S (40225, 45) | 139.3 | 144.0 | 169.5 | 157.5 | 163.7 | 163.8 | 165.8 | 249.8 | 1.18 |
| 44% | G (40364, 45) | 138.8 | 143.5 | 169.3 | 157.1 | 163.5 | 163.5 | 165.7 | 249.8 | 1.18 |
| 44% | D (40482, 68) | 141.1 | 146.2 | 170.9 | 159.9 | 165.1 | 165.1 | 167.2 | 249.8 | 1.17 |

---

## 6. Sensitivity

### 6.1 One constant at a time (others at default) [sim]

"best κ" is the best CPU-design κ among −∞, 0, 0.5, 1, 2 and 4. The ratio columns are tok/s relative to the current
rule under the same constants.

**gpt-oss-120b**

| setting (others at default) | ncmoe32: current tok/s | best κ (CPU misses) | fetch-all DFA-evict / current | split+bg / current | pooled split LRU / current | SoL tok/s | ncmoe27: current tok/s | best κ (CPU misses) | fetch-all DFA-evict / current | split+bg / current | pooled split LRU / current | SoL tok/s | ncmoe20: current tok/s | best κ (CPU misses) | fetch-all DFA-evict / current | split+bg / current | pooled split LRU / current | SoL tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DRAM B_D 40 GB/s (B_h 32) | 32.0 | κ=0.5 | 1.33 | 1.34 | 1.26 | 88 | 53.3 | κ=1 | 1.27 | 1.27 | 1.24 | 166 | 85.2 | κ=1 | 1.16 | 1.16 | 1.19 | 166 |
| DRAM B_D 50 GB/s (B_h 40) | 38.0 | κ=0.5 | 1.32 | 1.34 | 1.26 | 110 | 61.3 | κ=1 | 1.25 | 1.25 | 1.23 | 167 | 93.1 | κ=1 | 1.14 | 1.14 | 1.16 | 167 |
| DRAM B_D 57 GB/s (B_h 46) | 41.9 | κ=0.5 | 1.19 | 1.31 | 1.24 | 125 | 66.2 | κ=1 | 1.16 | 1.21 | 1.18 | 167 | 97.6 | κ=1 | 1.09 | 1.11 | 1.12 | 167 |
| DRAM B_D 70 GB/s (B_h 56) | 48.5 | κ=0 | 1.03 | 1.23 | 1.20 | 154 | 74.0 | κ=1 | 1.03 | 1.11 | 1.10 | 168 | 104.2 | κ=0.5 | 1.02 | 1.04 | 0.99 | 168 |
| PCIe B_P 30 GB/s | 41.9 | κ=0.5 | 0.81 | 1.23 | 1.20 | 122 | 66.2 | κ=1 | 0.86 | 1.10 | 1.10 | 167 | 97.6 | κ=1 | 0.91 | 1.04 | 0.98 | 167 |
| PCIe B_P 40 GB/s | 41.9 | κ=0.5 | 1.01 | 1.24 | 1.20 | 125 | 66.2 | κ=1 | 1.02 | 1.11 | 1.10 | 167 | 97.6 | κ=1 | 1.02 | 1.04 | 0.99 | 167 |
| PCIe B_P 50 GB/s | 41.9 | κ=0.5 | 1.19 | 1.31 | 1.24 | 125 | 66.2 | κ=1 | 1.16 | 1.21 | 1.18 | 167 | 97.6 | κ=1 | 1.09 | 1.11 | 1.12 | 167 |
| PCIe B_P 52 GB/s | 41.9 | κ=0.5 | 1.23 | 1.31 | 1.25 | 125 | 66.2 | κ=1 | 1.18 | 1.22 | 1.19 | 167 | 97.6 | κ=1 | 1.10 | 1.12 | 1.13 | 167 |
| τ_e 5 µs | 42.4 | κ=0.5 | 1.18 | 1.29 | 1.22 | 125 | 66.8 | κ=1 | 1.14 | 1.19 | 1.17 | 167 | 98.1 | κ=1 | 1.09 | 1.10 | 1.12 | 167 |
| τ_e 10 µs | 41.9 | κ=0.5 | 1.19 | 1.31 | 1.24 | 125 | 66.2 | κ=1 | 1.16 | 1.21 | 1.18 | 167 | 97.6 | κ=1 | 1.09 | 1.11 | 1.12 | 167 |
| τ_e 21 µs | 40.8 | κ=0.5 | 1.23 | 1.34 | 1.27 | 125 | 64.8 | κ=1 | 1.18 | 1.23 | 1.21 | 167 | 96.3 | κ=1 | 1.11 | 1.12 | 1.14 | 167 |
| O 10 µs/layer | 42.8 | κ=0.5 | 1.20 | 1.31 | 1.24 | 125 | 68.5 | κ=1 | 1.16 | 1.22 | 1.19 | 167 | 102.6 | κ=1 | 1.10 | 1.11 | 1.13 | 167 |
| O 24 µs/layer | 41.9 | κ=0.5 | 1.19 | 1.31 | 1.24 | 125 | 66.2 | κ=1 | 1.16 | 1.21 | 1.18 | 167 | 97.6 | κ=1 | 1.09 | 1.11 | 1.12 | 167 |
| O 38 µs/layer | 41.0 | κ=0.5 | 1.19 | 1.30 | 1.23 | 125 | 64.1 | κ=1 | 1.15 | 1.20 | 1.18 | 167 | 93.0 | κ=0.5 | 1.09 | 1.10 | 1.12 | 167 |
| f 10 µs/request | 43.0 | κ=0.5 | 1.17 | 1.27 | 1.21 | 125 | 67.9 | κ=1 | 1.13 | 1.18 | 1.15 | 167 | 99.4 | κ=1 | 1.07 | 1.09 | 1.10 | 167 |
| f 30 µs/request | 41.9 | κ=0.5 | 1.19 | 1.31 | 1.24 | 125 | 66.2 | κ=1 | 1.16 | 1.21 | 1.18 | 167 | 97.6 | κ=1 | 1.09 | 1.11 | 1.12 | 167 |
| f 50 µs/request | 40.9 | κ=0.5 | 1.22 | 1.34 | 1.27 | 125 | 64.6 | κ=1 | 1.18 | 1.24 | 1.21 | 167 | 95.8 | κ=1 | 1.11 | 1.13 | 1.14 | 167 |
| η_g 0.4 | 40.9 | κ=0.5 | 1.18 | 1.29 | 1.23 | 125 | 63.2 | κ=1 | 1.14 | 1.19 | 1.17 | 145 | 90.0 | κ=0.5 | 1.08 | 1.09 | 1.11 | 145 |
| η_g 0.5 | 41.9 | κ=0.5 | 1.19 | 1.31 | 1.24 | 125 | 66.2 | κ=1 | 1.16 | 1.21 | 1.18 | 167 | 97.6 | κ=1 | 1.09 | 1.11 | 1.12 | 167 |
| η_g 0.6 | 42.6 | κ=0.5 | 1.20 | 1.32 | 1.24 | 125 | 68.4 | κ=1 | 1.17 | 1.22 | 1.19 | 187 | 103.4 | κ=1 | 1.10 | 1.12 | 1.14 | 187 |
| θ 0.00 | 41.9 | κ=0.5 | 1.19 | 1.31 | 1.24 | 125 | 66.2 | κ=1 | 1.16 | 1.21 | 1.18 | 167 | 97.6 | κ=1 | 1.09 | 1.11 | 1.12 | 167 |
| θ 0.25 | 40.9 | κ=2 | 1.22 | 1.34 | 1.27 | 125 | 63.8 | κ=2 | 1.20 | 1.25 | 1.23 | 167 | 94.8 | κ=1 | 1.12 | 1.14 | 1.16 | 167 |
| θ 0.50 | 40.0 | κ=2 | 1.25 | 1.37 | 1.30 | 125 | 61.5 | κ=2 | 1.24 | 1.30 | 1.27 | 167 | 92.1 | κ=1 | 1.16 | 1.17 | 1.19 | 167 |

**Qwen3-30B-A3B**

| setting (others at default) | 11%: current tok/s | best κ (CPU misses) | fetch-all DFA-evict / current | split+bg / current | pooled split LRU / current | SoL tok/s | 25%: current tok/s | best κ (CPU misses) | fetch-all DFA-evict / current | split+bg / current | pooled split LRU / current | SoL tok/s | 44%: current tok/s | best κ (CPU misses) | fetch-all DFA-evict / current | split+bg / current | pooled split LRU / current | SoL tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DRAM B_D 40 GB/s (B_h 32) | 46.9 | κ=1 | 1.38 | 1.45 | 1.32 | 142 | 78.7 | κ=0.5 | 1.34 | 1.37 | 1.32 | 248 | 125.4 | κ=0 | 1.22 | 1.22 | 1.24 | 248 |
| DRAM B_D 50 GB/s (B_h 40) | 54.6 | κ=1 | 1.38 | 1.46 | 1.34 | 178 | 88.6 | κ=0.5 | 1.32 | 1.36 | 1.31 | 249 | 134.4 | κ=0 | 1.19 | 1.20 | 1.22 | 249 |
| DRAM B_D 57 GB/s (B_h 46) | 59.3 | κ=1 | 1.27 | 1.46 | 1.36 | 203 | 94.5 | κ=0.5 | 1.24 | 1.33 | 1.30 | 250 | 139.3 | κ=0 | 1.15 | 1.18 | 1.19 | 250 |
| DRAM B_D 70 GB/s (B_h 56) | 67.1 | κ=1 | 1.12 | 1.44 | 1.37 | 245 | 103.5 | κ=0.5 | 1.13 | 1.29 | 1.27 | 251 | 146.4 | κ=0 | 1.09 | 1.14 | 1.15 | 251 |
| PCIe B_P 30 GB/s | 59.3 | κ=1 | 0.89 | 1.39 | 1.34 | 187 | 94.5 | κ=0.5 | 0.96 | 1.24 | 1.23 | 250 | 139.3 | κ=0 | 1.01 | 1.11 | 1.09 | 250 |
| PCIe B_P 40 GB/s | 59.3 | κ=1 | 1.09 | 1.43 | 1.35 | 203 | 94.5 | κ=0.5 | 1.12 | 1.30 | 1.26 | 250 | 139.3 | κ=0 | 1.09 | 1.15 | 1.16 | 250 |
| PCIe B_P 50 GB/s | 59.3 | κ=1 | 1.27 | 1.46 | 1.36 | 203 | 94.5 | κ=0.5 | 1.24 | 1.33 | 1.30 | 250 | 139.3 | κ=0 | 1.15 | 1.18 | 1.19 | 250 |
| PCIe B_P 52 GB/s | 59.3 | κ=1 | 1.30 | 1.46 | 1.35 | 203 | 94.5 | κ=0.5 | 1.26 | 1.34 | 1.30 | 250 | 139.3 | κ=0 | 1.16 | 1.18 | 1.19 | 250 |
| τ_e 5 µs | 62.5 | κ=1 | 1.20 | 1.38 | 1.29 | 203 | 98.2 | κ=0.5 | 1.19 | 1.28 | 1.25 | 250 | 142.2 | κ=0 | 1.13 | 1.15 | 1.17 | 250 |
| τ_e 10 µs | 59.3 | κ=1 | 1.27 | 1.46 | 1.36 | 203 | 94.5 | κ=0.5 | 1.24 | 1.33 | 1.30 | 250 | 139.3 | κ=0 | 1.15 | 1.18 | 1.19 | 250 |
| τ_e 21 µs | 53.5 | κ=1 | 1.40 | 1.62 | 1.50 | 203 | 87.3 | κ=0.5 | 1.34 | 1.44 | 1.40 | 250 | 133.2 | κ=0 | 1.20 | 1.23 | 1.25 | 250 |
| O 10 µs/layer | 61.8 | κ=1 | 1.28 | 1.49 | 1.38 | 203 | 100.9 | κ=0.5 | 1.26 | 1.36 | 1.32 | 250 | 153.7 | κ=0 | 1.17 | 1.20 | 1.21 | 250 |
| O 24 µs/layer | 59.3 | κ=1 | 1.27 | 1.46 | 1.36 | 203 | 94.5 | κ=0.5 | 1.24 | 1.33 | 1.30 | 250 | 139.3 | κ=0 | 1.15 | 1.18 | 1.19 | 250 |
| O 38 µs/layer | 57.1 | κ=1 | 1.25 | 1.43 | 1.34 | 203 | 88.8 | κ=0.5 | 1.22 | 1.31 | 1.27 | 250 | 127.4 | κ=0 | 1.13 | 1.16 | 1.17 | 250 |
| f 10 µs/request | 60.1 | κ=1 | 1.25 | 1.44 | 1.34 | 203 | 96.0 | κ=0.5 | 1.22 | 1.31 | 1.28 | 250 | 141.0 | κ=0 | 1.14 | 1.16 | 1.18 | 250 |
| f 30 µs/request | 57.0 | κ=1 | 1.32 | 1.51 | 1.41 | 203 | 90.3 | κ=0.5 | 1.29 | 1.39 | 1.35 | 250 | 134.4 | κ=0 | 1.19 | 1.22 | 1.23 | 250 |
| f 50 µs/request | 54.2 | κ=1 | 1.39 | 1.57 | 1.46 | 203 | 85.3 | κ=0.5 | 1.37 | 1.44 | 1.40 | 250 | 128.5 | κ=0 | 1.25 | 1.26 | 1.28 | 250 |
| η_g 0.4 | 58.5 | κ=1 | 1.26 | 1.45 | 1.35 | 203 | 91.6 | κ=0.5 | 1.22 | 1.32 | 1.28 | 222 | 131.5 | κ=0 | 1.13 | 1.16 | 1.17 | 222 |
| η_g 0.5 | 59.3 | κ=1 | 1.27 | 1.46 | 1.36 | 203 | 94.5 | κ=0.5 | 1.24 | 1.33 | 1.30 | 250 | 139.3 | κ=0 | 1.15 | 1.18 | 1.19 | 250 |
| η_g 0.6 | 59.9 | κ=1 | 1.27 | 1.46 | 1.36 | 203 | 96.5 | κ=0.5 | 1.25 | 1.34 | 1.31 | 273 | 145.0 | κ=0 | 1.16 | 1.19 | 1.20 | 273 |
| θ 0.00 | 59.3 | κ=1 | 1.27 | 1.46 | 1.36 | 203 | 94.5 | κ=0.5 | 1.24 | 1.33 | 1.30 | 250 | 139.3 | κ=0 | 1.15 | 1.18 | 1.19 | 250 |
| θ 0.25 | 58.9 | κ=2 | 1.27 | 1.47 | 1.36 | 203 | 93.4 | κ=1 | 1.25 | 1.35 | 1.31 | 250 | 137.9 | κ=1 | 1.16 | 1.19 | 1.20 | 250 |
| θ 0.50 | 58.6 | κ=2 | 1.28 | 1.47 | 1.37 | 203 | 92.3 | κ=2 | 1.27 | 1.36 | 1.33 | 250 | 136.6 | κ=1 | 1.17 | 1.20 | 1.21 | 250 |

**Reading.**
- **Stage 2 is robust.** Its gain over the current rule stays at ×1.04–1.37 (gpt-oss-120b) and ×1.11–1.62 (Qwen3)
  across every one-at-a-time setting.
- **It is smallest when:**
  - PCIe is slow (30–40 GB/s);
  - DRAM is fast (70 GB/s, where the helpers alone reach 56 GB/s).
  In both cases the table sends lone misses back to the CPU, and background admission is what keeps the cache warm
  there. Without it the split drops to ×0.97–0.98 at ncmoe20 (JSON: `split DFA-evict`).
- **It is largest when the CPU path's fixed costs are high** (f, τ_e) or when admissions interfere (θ).
- **Stage 1 (fetch every miss) is sensitive to PCIe.** At B_P = 30 GB/s it falls to ×0.81–0.91 (gpt-oss-120b) and
  ×0.89–1.01 (Qwen3). At 40 GB/s it is only ×1.01–1.02 on gpt-oss-120b.
- **The best CPU-design κ stays at 0–2 in every setting,** and retuning κ never gains more than 2 % on gpt-oss-120b
  or 5 % on Qwen3.

### 6.2 Admission interference model [sim]

Uniform (default) → optimistic, where admissions are paced into idle DRAM time:

| model | budget | current: per-layer DFA κ=now, CPU misses | DFA, admit every miss (κ=−∞) | DFA κ=0 | DFA κ=0.5 | LRU (admit every miss) | MIN-bypass (offline; policy ceiling for CPU misses) | split + background DFA κ=1 for CPU-run misses |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss-120b | ncmoe32 | 41.9 → 41.9 | 32.4 → 37.1 | 41.9 → 42.1 | 42.1 → 42.1 | 29.7 → 33.6 | 57.1 → 60.6 | 54.7 → 54.8 |
| gpt-oss-120b | ncmoe27 | 66.2 → 66.2 | 58.8 → 66.6 | 59.6 → 66.6 | 64.8 → 66.6 | 54.3 → 62.4 | 92.4 → 92.9 | 79.9 → 79.9 |
| gpt-oss-120b | ncmoe20 | 97.6 → 97.6 | 97.1 → 98.7 | 97.1 → 98.7 | 97.3 → 98.7 | 95.6 → 97.4 | 120.9 → 120.9 | 108.1 → 108.1 |
| qwen3-30b-a3b | 11% | 59.3 → 59.3 | 45.2 → 51.8 | 59.1 → 59.1 | 59.3 → 59.3 | 43.2 → 49.7 | 81.1 → 83.2 | 86.5 → 86.7 |
| qwen3-30b-a3b | 25% | 94.5 → 94.5 | 88.4 → 93.7 | 95.2 → 95.2 | 95.4 → 95.4 | 80.3 → 86.0 | 130.7 → 130.7 | 125.9 → 125.9 |
| qwen3-30b-a3b | 44% | 139.3 → 139.3 | 143.9 → 143.9 | 144.0 → 144.0 | 143.9 → 143.9 | 140.1 → 140.1 | 169.5 → 169.5 | 163.8 → 163.8 |

**Reading.**
- **The current rule (κ = 1 / 2) is insensitive to the model.** Its admission traffic (4.6–8.8 copies per token)
  fits in the DRAM slack between the helpers' 46 GB/s and the 57 GB/s ceiling.
- **Admit-more rules do depend on it.** κ = 0 and admit-all only match κ = 1 if admissions are paced; LRU stays
  below. On
  this machine, pacing (sprint item 6) is a precondition for admitting more, not a gain by itself.
- **Stage 2 is unaffected:** it makes almost no background copies.

### 6.3 Monte Carlo over the expected ranges (200 draws, ratios to the current rule, median [P10–P90]) [sim]

| model | budget | current tok/s | DFA κ=0 | pooled DFA κ=0.5 | fetch every miss, LRU evict | fetch every miss, evict lowest decayed score | split, decayed-score evict, CPU misses not cached | split + background DFA κ=1 for CPU-run misses | pooled split, LRU (FreeToken-like) | MIN-bypass (offline; policy ceiling for CPU misses) | split+bg tok/s | split+bg % of SoL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss-120b | ncmoe32 | 40.8 [37.9–43.6] | 0.98 [0.93–1.00] | 1.00 [0.99–1.00] | 1.12 [1.04–1.20] | 1.20 [1.11–1.28] | 1.32 [1.23–1.44] | 1.32 [1.23–1.44] | 1.25 [1.17–1.37] | 1.39 [1.26–1.44] | 53.8 [51.8–56.2] | 0.43 [0.41–0.45] |
| gpt-oss-120b | ncmoe27 | 63.8 [60.0–68.0] | 0.95 [0.85–1.00] | 0.99 [0.93–1.01] | 1.11 [1.05–1.17] | 1.17 [1.11–1.23] | 1.23 [1.16–1.30] | 1.23 [1.16–1.30] | 1.20 [1.14–1.27] | 1.38 [1.34–1.43] | 78.4 [74.7–82.5] | 0.47 [0.44–0.51] |
| gpt-oss-120b | ncmoe20 | 94.2 [88.5–101.7] | 0.98 [0.95–1.01] | 1.01 [0.99–1.03] | 1.09 [1.05–1.13] | 1.11 [1.06–1.15] | 1.12 [1.08–1.16] | 1.12 [1.08–1.16] | 1.14 [1.10–1.18] | 1.24 [1.21–1.27] | 105.5 [98.8–114.8] | 0.64 [0.61–0.68] |
| qwen3-30b-a3b | 11% | 57.2 [52.3–62.1] | 0.99 [0.98–0.99] | 0.99 [0.98–0.99] | 1.20 [1.09–1.31] | 1.28 [1.16–1.39] | 1.50 [1.37–1.67] | 1.50 [1.37–1.67] | 1.40 [1.28–1.56] | 1.35 [1.28–1.41] | 85.5 [81.2–90.7] | 0.42 [0.40–0.45] |
| qwen3-30b-a3b | 25% | 91.0 [84.5–98.5] | 0.98 [0.96–1.00] | 1.01 [1.00–1.02] | 1.18 [1.10–1.26] | 1.25 [1.16–1.34] | 1.36 [1.27–1.47] | 1.36 [1.27–1.47] | 1.33 [1.24–1.43] | 1.37 [1.32–1.42] | 123.9 [116.2–132.6] | 0.50 [0.46–0.54] |
| qwen3-30b-a3b | 44% | 135.2 [125.6–147.0] | 1.02 [1.00–1.03] | 1.05 [1.03–1.06] | 1.14 [1.09–1.19] | 1.16 [1.11–1.21] | 1.19 [1.14–1.24] | 1.19 [1.14–1.24] | 1.21 [1.15–1.26] | 1.22 [1.18–1.26] | 160.7 [148.3–178.0] | 0.65 [0.60–0.71] |

**Which CPU-design κ wins across the 200 draws.**
- gpt-oss-120b: κ = 1 wins 99–160 draws, and κ = 0.5–2 covers the rest.
- Qwen3: κ = 2 wins 134 draws at 11 %; κ = 1 wins 182 at 25 %; κ = 0.5 and 1 split 44 %.

---

## 7. Recommendation, with expected gain

1. **Synchronous fetch-and-split for current misses.**
   - **Mechanism.**
     - Decided on the GPU in the existing `ec_req` path: m, then T[m], then which experts to fetch, by highest
       decayed score (the host already keeps the scores).
     - The fetch is a device-initiated zero-copy read into the slot. One launch should cover all of a layer's fetched
       experts, so that τ_x stays at a few µs.
     - The mailbox gets only the CPU share. Fetched experts are published into the slot maps this step.
   - **Eviction:** lowest decayed score (half-life 16), not LRU.
   - **Background admission:** keep DFA κ = 1 for CPU-run misses.
   - **Tables at the estimated constants:** gpt-oss-120b T = [0, 1, 1, 2, 3]; Qwen3 T = [0, 1, 1, 2, 3, 3, 4, 5, 6].
     Recompute them from measured B_P, B_h, B_D, f, τ_e and τ_x.
   - **Expected gain [sim, MC median, P10–P90]:**

     | model | small budget | middle budget | large budget |
     |---|---|---|---|
     | gpt-oss-120b (ncmoe32 / 27 / 20) | +32 % [+23, +44] | +23 % [+16, +30] | +12 % [+8, +16] |
     | Qwen3-30B-A3B (11 / 25 / 44 %) | +50 % [+37, +67] | +36 % [+27, +47] | +19 % [+14, +24] |

   - **Staging.**
     - Stage 1 (T[m] = m) captures 56–92 % of the stage-2 gain, more at larger budgets: +11–20 % on gpt-oss-120b and
       +16–28 % on Qwen3 [sim, MC medians].
     - It regresses if the link delivers ~30 GB/s, and gains little on gpt-oss-120b at 40 GB/s.
2. **Admission rule: keep the per-layer DFA; set κ = 1 on both models.**
   - Half-life stays 16 (32 is −1 to +3 %, not worth a knob change).
   - No pooling.
   - Expected effect: +0–3 % [sim]. It is below race noise and is justified only as a simplification.
   - Do not lower κ unless admissions are paced.

**What the gain is not.**
- These are policy and miss-service gains at *our current* per-layer fixed costs (O = 24 µs, f = 15–30 µs). The fixed
  costs themselves (sprint items 3–5) are separate and add on top: O 38 → 10 µs is worth +4–10 % on gpt-oss-120b [sim, §6.1].
- **Fetch latency.** The one assumption stage 2 depends on that §6 does not sweep widely is τ_x, only 2–10 µs. If a
  critical fetch costs a host round trip instead (≈ 30 µs), the result changes [est]:
  - gpt-oss-120b loses about 0.5 ms per token at ncmoe27 (−4 %);
  - Qwen3 at 11 % loses up to about 2.5 ms per token (up to ≈ −18 %), because it makes about 100 fetches per token of
    small experts.
  So on Qwen3 the fetch must be device-initiated and batched per layer.

---

## 8. What to measure on the real machine before implementing

1. **Concurrent DRAM read (decides split vs fetch-all).**
   - Measure helper read throughput with `llama-ec-cpubench` at 16 threads, one per core, while a pinned H2D stream
     copies expert-sized buffers. Also measure each alone.
   - This gives B_h, B_P and the combined ceiling B_D. The split's edge over fetch-all is roughly
     B_D/max(B_h, B_P) − 1 on miss-heavy layers. If the combined read ≈ B_P, stage 1 is enough.
2. **Critical-path fetch cost.** Measure the time from the router's top-k to a usable slot for one 13.25 MB and one
   2.86 MB expert, and for a batch of 2–4 per layer. Compare the zero-copy kernel (`LLAMA_EC_ZC`, 16 vs 64 blocks)
   with the copy engine. This gives the effective B_P and τ_x.
3. **Helper service time vs m.** Fit `mb_busy_us[m]/mb_count[m]` = f + m·c_e on the 9950X. This gives f and τ_e, and
   whether c_e is bandwidth-bound.
4. **O, and admission interference θ.** Run the mailbox cache at two budgets with stats (O), plus DFA κ = 1 against a
   static INIT with no admissions at the same hit rate (θ and the uniform/optimistic question).
5. **Model check before trusting ratios.** Run the current system at ncmoe32/27/20 on the S prompts. The model predicts
   41.9 / 66.2 / 97.6 tok/s at default constants (MC P10–P90: 38–44 / 60–68 / 89–102) [sim].
   - Refit the constants from items 1–4.
   - If the refitted prediction is still off by more than ~10 %, find out why before building stage 2.
   - Then recompute T[m] with `split_table()` and the gains with `sim_homepc.py`. It takes the constants as arguments,
     so this needs code edits only in `DEF`.

---

## 9. Limitations

- **Time model.**
  - Mean-field per token.
  - No overlap across layers (no prefetch).
  - KV-cache growth ignored (short contexts).
  - GPU-side interference of admission kernels is only covered by θ.
- **Unmeasured constants.** All machine constants are unmeasured on the target, and t_lay, t_eg and t_tok are not
  swept. They shift absolute tok/s and the SoL, but barely the ratios: η_g ±0.1 moves the stage-2 ratio by ≤ 0.03.
- **Trace coverage.**
  - gpt-oss-120b S and G cover only 35 conversations each.
  - Qwen3 uses ≈ 40k-token prefixes.
  - The Qwen3 routing comes from the FP8 checkpoint, while the time model uses Q4_K_M sizes.
- **The split is idealised.**
  - The table is per layer-step and static.
  - Which misses to fetch uses the decayed score.
  - There is no per-step reordering across layers.
  - FreeToken's actual split and pool management may differ from the "FreeToken-like" option.
  - Both designs are charged the same O and f here, so this compares *policies*, not implementations.
- **The bound is loose.** It assumes perfect prefetch, so every option's % of SoL is low.

## 10. Reproduce

```
python scripts/sim_homepc.py --models gpt-oss-120b  --out homepc_120b.json    # ≈ 4 min on 2 cores
python scripts/sim_homepc.py --models qwen3-30b-a3b --out homepc_qwen.json    # ≈ 16 min (40k-token prefixes)
```

The two runs are independent, so run them in parallel to stay under 20 minutes. The JSON holds, for each arm and
budget:
- every option's hit rate, CPU / fetch / admission counts per token, tok/s and % of SoL, under both interference
  models;
- the one-at-a-time sensitivity;
- all 200 Monte Carlo draws.
