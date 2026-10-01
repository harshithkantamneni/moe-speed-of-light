# Policy study: recency against frequency for the expert cache on nine models, drift of expert popularity, and intervals on the W50 exponent

Trace-driven, no GPU: `scripts/policy_study.py` (numbers in `prereg/policy_study.json` and
`prereg/foresight/w50_intervals.json`), figure `scripts/fig_policy.py` → `paper/figs/policy.pdf`. Not a registered
job: no predictions were written before the run. Revised once: the Belady reference is now the exact MIN with
bypass of `mosl/cachesim.py` (see the first caveat); the first version used `scripts/foresight.py`'s `_pol` at
W = ∞, which reads 0.5–9.5% more.

**Why.** The literature disagrees on whether recency (LRU) or frequency should replace experts in an offloaded MoE
cache: arXiv 2608.12103 and llama.cpp PR #27861 favour LRU at practical budgets; FlashMoE (2601.17063), Mira
(2609.38090) and SpecMD favour frequency; 2608.07911 warns that a flattened (non-atomic) replay inflates LRU's misses
by 27–29%. Nobody reports nine models at batch 1 in reads or bytes per token, and nobody measures how fast expert
popularity drifts, which is what decides the question.

**Setup.**
- **Traces:** own sampled text (arm S) of the 9 models in `data/traces_manifest.json`, response tokens of every
  conversation back to back, cache carried across conversations (the same streams as `prereg/foresight/foresight_S.json`;
  32,601–152,470 tokens per model, 450–950 tokens per conversation).
- **Budgets:** C = E/8, E/4, 3E/8 slots per layer, rounded half up (Qwen1.5-MoE, E = 60: 8, 15, 23). Mixtral (E = 8)
  would get C = 1, 2, 3; it is run at C = 2, 3, 4 instead, so its rows are not the same fractions as the others.
- **Replay:** event-atomic, per layer. A token's k experts are one request set; hits are served first; for the online
  policies an expert of the current token is never evicted to serve that token. Every policy is scored on the last
  90% of each trace; the first 10% warms the online policies and is the static cache's profile.
- **Policies:** LRU and LFU (classic, every miss copied; `mosl/cachesim.py`); decayed frequency, half-life 16
  tokens, with hysteresis κ = 0 and with the deployed κ (0.5–2 per model; a miss is copied over the lowest-scored
  slot if its score beats the victim's by κ, else run where it lives, one read either way; `_pol` with W < 0);
  ARC and S3-FIFO (new numba ports, checked against pure-Python references on 150 random traces; a resident of the
  current token is skipped as a victim; S3-FIFO's small queue is max(1, round(0.1 C)) slots, one slot for C < 15);
  static top-C by count over the first 10% of the trace; Belady within a window of W ∈ {1, 2, 4, 8, 16} future tokens
  with decayed frequency (κ = 0) beyond it (`_pol`'s rule, given the optimum's freedom to evict a resident already
  served this step, so that W → ∞ is the optimum); and `opt`, the exact Belady MIN with bypass over the whole future
  (`mosl/cachesim.py`, verified against exhaustive search). Three reference rows are in the JSON only: the engine's
  policy as shipped (CPU runs every miss, background admissions are separate copies; `mosl/ecsim_fast.py`), a static
  cache chosen in hindsight over the whole trace, and `opt-pol`, `_pol` at W = ∞ (job 084's optimum).
- **Metric:** expert reads per token from host memory (a bypassed miss is a read), relative to `opt`; bytes per token
  with the expert size from `data/gguf_bytes.json` where a GGUF header was read (7 models; OLMoE and Qwen1.5-MoE have
  none, so they are reported in reads only); hit rate; and the union-to-capacity ratio, the mean number of distinct
  experts a layer requests in 16 consecutive tokens divided by C (above 1: the cache turns over inside the window).
- **Runtime:** 10 minutes for everything on the full traces (2 cores); nothing was truncated.

## 1. Reads per token relative to Belady (`opt` reads per token in the last column)

| model | C | C/k | union/C | LRU | LFU | DF κ=0 | DF κ_dep | ARC | S3-FIFO | static | W=1 | W=2 | W=4 | W=8 | W=16 | Belady (reads/tok) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| olmoe | 8 (E/8) | 1.00 | 4.47 | 1.44 | 1.44 | 1.42 | 1.44 | 1.44 | 1.44 | 1.92 | 1.11 | 1.03 | 1.00 | 1.00 | 1.00 | 47.5 |
| olmoe | 16 (E/4) | 2.00 | 2.24 | 1.75 | 1.90 | 1.68 | 1.75 | 1.74 | 1.65 | 2.74 | 1.38 | 1.26 | 1.11 | 1.01 | 1.00 | 25.5 |
| olmoe | 24 (3E/8) | 3.00 | 1.49 | 2.03 | 2.40 | 1.88 | 2.00 | 1.99 | 1.87 | 3.48 | 1.60 | 1.48 | 1.31 | 1.11 | 1.02 | 14.9 |
| gpt-oss-20b | 4 (E/8) | 1.00 | 4.06 | 1.43 | 1.43 | 1.44 | 1.46 | 1.43 | 1.43 | 2.03 | 1.10 | 1.02 | 1.00 | 1.00 | 1.00 | 35.8 |
| gpt-oss-20b | 8 (E/4) | 2.00 | 2.03 | 1.79 | 2.02 | 1.70 | 1.75 | 1.77 | 1.69 | 3.11 | 1.36 | 1.23 | 1.08 | 1.01 | 1.00 | 17.4 |
| gpt-oss-20b | 12 (3E/8) | 3.00 | 1.35 | 2.03 | 2.60 | 1.83 | 1.93 | 1.99 | 1.87 | 4.39 | 1.56 | 1.43 | 1.26 | 1.09 | 1.01 | 8.8 |
| qwen3-30b-a3b | 16 (E/8) | 2.00 | 2.43 | 1.71 | 2.13 | 1.65 | 1.73 | 1.69 | 1.60 | 3.70 | 1.36 | 1.25 | 1.11 | 1.01 | 1.00 | 86.5 |
| qwen3-30b-a3b | 32 (E/4) | 4.00 | 1.22 | 2.07 | 3.84 | 1.92 | 2.14 | 2.01 | 1.90 | 7.43 | 1.70 | 1.60 | 1.45 | 1.26 | 1.07 | 35.4 |
| qwen3-30b-a3b | 48 (3E/8) | 6.00 | 0.81 | 2.26 | 6.15 | 2.08 | 2.55 | 2.23 | 2.11 | 13.24 | 1.96 | 1.88 | 1.75 | 1.56 | 1.32 | 15.7 |
| gpt-oss-120b | 16 (E/8) | 4.00 | 1.68 | 1.77 | 2.31 | 1.66 | 1.74 | 1.71 | 1.65 | 3.43 | 1.50 | 1.43 | 1.31 | 1.16 | 1.03 | 30.3 |
| gpt-oss-120b | 32 (E/4) | 8.00 | 0.84 | 2.06 | 3.39 | 1.88 | 2.03 | 1.98 | 1.91 | 5.59 | 1.81 | 1.76 | 1.67 | 1.53 | 1.34 | 13.7 |
| gpt-oss-120b | 48 (3E/8) | 12.00 | 0.56 | 2.22 | 4.38 | 2.10 | 2.28 | 2.17 | 2.10 | 7.95 | 2.06 | 2.02 | 1.95 | 1.84 | 1.66 | 6.9 |
| mixtral-8x7b | 2 | 1.00 | 3.75 | 1.42 | 1.42 | 1.46 | 1.47 | 1.42 | 1.42 | 1.66 | 1.11 | 1.02 | 1.00 | 1.00 | 1.00 | 28.2 |
| mixtral-8x7b | 3 | 1.50 | 2.50 | 1.63 | 1.64 | 1.64 | 1.65 | 1.63 | 1.62 | 1.96 | 1.25 | 1.09 | 1.01 | 1.00 | 1.00 | 19.6 |
| mixtral-8x7b | 4 | 2.00 | 1.87 | 1.84 | 1.87 | 1.80 | 1.83 | 1.84 | 1.82 | 2.31 | 1.40 | 1.21 | 1.04 | 1.00 | 1.00 | 13.3 |
| deepseek-v2-lite | 8 (E/8) | 1.33 | 5.02 | 1.47 | 1.49 | 1.47 | 1.50 | 1.45 | 1.44 | 1.82 | 1.23 | 1.13 | 1.02 | 1.00 | 1.00 | 74.2 |
| deepseek-v2-lite | 16 (E/4) | 2.67 | 2.51 | 1.77 | 1.93 | 1.73 | 1.79 | 1.76 | 1.69 | 2.45 | 1.49 | 1.37 | 1.20 | 1.03 | 1.00 | 46.8 |
| deepseek-v2-lite | 24 (3E/8) | 4.00 | 1.67 | 2.03 | 2.36 | 1.94 | 2.04 | 2.02 | 1.94 | 3.11 | 1.71 | 1.59 | 1.40 | 1.17 | 1.01 | 30.4 |
| qwen1.5-moe | 8 (E/8) | 2.00 | 4.26 | 1.51 | 1.51 | 1.45 | 1.46 | 1.48 | 1.45 | 1.71 | 1.30 | 1.22 | 1.10 | 1.01 | 1.00 | 47.8 |
| qwen1.5-moe | 15 (E/4) | 3.75 | 2.27 | 1.80 | 1.85 | 1.69 | 1.70 | 1.78 | 1.69 | 2.18 | 1.54 | 1.45 | 1.30 | 1.11 | 1.00 | 32.1 |
| qwen1.5-moe | 23 (3E/8) | 5.75 | 1.48 | 2.10 | 2.26 | 1.94 | 1.96 | 2.08 | 1.97 | 2.75 | 1.79 | 1.69 | 1.53 | 1.30 | 1.05 | 20.6 |
| qwen2-57b | 8 (E/8) | 1.00 | 5.73 | 1.41 | 1.41 | 1.41 | 1.42 | 1.41 | 1.41 | 1.43 | 1.18 | 1.09 | 1.01 | 1.00 | 1.00 | 110.5 |
| qwen2-57b | 16 (E/4) | 2.00 | 2.87 | 1.71 | 1.61 | 1.71 | 1.74 | 1.70 | 1.67 | 1.74 | 1.45 | 1.32 | 1.13 | 1.00 | 1.00 | 71.5 |
| qwen2-57b | 24 (3E/8) | 3.00 | 1.91 | 1.99 | 1.84 | 1.97 | 2.00 | 1.99 | 1.94 | 2.00 | 1.69 | 1.54 | 1.31 | 1.05 | 1.00 | 47.6 |
| phi3.5-moe | 2 (E/8) | 1.00 | 4.50 | 1.37 | 1.37 | 1.35 | 1.36 | 1.37 | 1.38 | 2.01 | 1.07 | 1.02 | 1.00 | 1.00 | 1.00 | 26.3 |
| phi3.5-moe | 4 (E/4) | 2.00 | 2.25 | 1.66 | 1.95 | 1.58 | 1.61 | 1.64 | 1.59 | 3.03 | 1.30 | 1.19 | 1.06 | 1.01 | 1.00 | 14.4 |
| phi3.5-moe | 6 (3E/8) | 3.00 | 1.50 | 1.88 | 2.53 | 1.75 | 1.79 | 1.86 | 1.78 | 4.20 | 1.49 | 1.37 | 1.20 | 1.06 | 1.01 | 8.4 |

**Medians over the 9 models (Mixtral's C = 2, 3, 4 counted as its three classes):**

| budget | LRU | LFU | DF κ=0 | DF κ_dep | ARC | S3-FIFO | static | W=1 | W=2 | W=4 | W=8 | W=16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E/8 | 1.44 | 1.44 | 1.45 | 1.46 | 1.44 | 1.44 | 1.92 | 1.18 | 1.09 | 1.01 | 1.00 | 1.00 |
| E/4 | 1.77 | 1.93 | 1.70 | 1.75 | 1.76 | 1.69 | 2.74 | 1.45 | 1.32 | 1.13 | 1.01 | 1.00 |
| 3E/8 | 2.03 | 2.40 | 1.94 | 2.00 | 1.99 | 1.94 | 3.48 | 1.69 | 1.54 | 1.31 | 1.11 | 1.01 |
| all 27 | 1.77 | 1.93 | 1.70 | 1.75 | 1.76 | 1.69 | 2.74 | 1.45 | 1.32 | 1.13 | 1.01 | 1.00 |
| all 27, min–max | 1.37–2.26 | 1.37–6.15 | 1.35–2.10 | 1.36–2.55 | 1.37–2.23 | 1.38–2.11 | 1.43–13.24 | 1.07–2.06 | 1.02–2.02 | 1.00–1.95 | 1.00–1.84 | 1.00–1.66 |

No policy reads fewer experts than `opt` in any cell (the JSON's `summary.below_opt` is empty). Job 084's optimum
(`_pol` at W = ∞, which never evicts a resident of the current token) reads 0.5–9.5% more than `opt` (median 4.1%):
4.3–9.5% at C/k ≤ 1.5, 0.5–3.1% at C/k ≥ 4 (JSON `summary.opt_pol_over_opt`).

Fewest reads among the online policies: decayed frequency (κ = 0) in 12 of 27 cells, S3-FIFO in 10, LFU in 2
(Qwen2-57B at E/4 and 3E/8), and LRU in 3, each an exact tie with LFU and ARC at C = k. Within 1% of the best:
decayed frequency 20 cells, S3-FIFO 18, LFU 5, LRU 4, ARC 4.

**LRU against decayed frequency (κ = 0), reads per token; + means LRU reads more (unchanged by the reference):**

| model | E/8 | E/4 | 3E/8 | sign |
|---|---|---|---|---|
| olmoe | 68.3 vs 67.4 (+1.4%) | 44.6 vs 42.9 (+4.0%) | 30.2 vs 28.0 (+7.8%) | + + + |
| gpt-oss-20b | 51.4 vs 51.8 (−0.8%) | 31.1 vs 29.5 (+5.6%) | 18.0 vs 16.2 (+10.6%) | − + + |
| qwen3-30b-a3b | 148.2 vs 143.0 (+3.7%) | 73.2 vs 68.0 (+7.7%) | 35.5 vs 32.8 (+8.4%) | + + + |
| gpt-oss-120b | 53.7 vs 50.2 (+6.9%) | 28.3 vs 25.8 (+9.9%) | 15.3 vs 14.4 (+5.7%) | + + + |
| mixtral-8x7b (C = 2, 3, 4) | 40.2 vs 41.2 (−2.5%) | 32.1 vs 32.1 (−0.1%) | 24.5 vs 24.0 (+2.0%) | − 0 + |
| deepseek-v2-lite | 108.7 vs 109.0 (−0.3%) | 82.7 vs 81.0 (+2.1%) | 61.9 vs 59.1 (+4.9%) | 0 + + |
| qwen1.5-moe | 72.2 vs 69.3 (+4.2%) | 57.9 vs 54.3 (+6.7%) | 43.4 vs 39.9 (+8.6%) | + + + |
| qwen2-57b | 155.5 vs 156.0 (−0.3%) | 122.1 vs 122.5 (−0.4%) | 94.9 vs 93.7 (+1.2%) | 0 0 + |
| phi3.5-moe | 36.1 vs 35.4 (+1.9%) | 24.0 vs 22.8 (+5.0%) | 15.9 vs 14.7 (+7.7%) | + + + |

**Bytes per token (MB) where the expert size is known: LRU / decayed frequency κ=0 / Belady.**

| model | expert MB | file | E/8 | E/4 | 3E/8 |
|---|---|---|---|---|---|
| olmoe | n/a (Q4_K_M-equivalent estimate 3.81 MB from the parameter count) | — | reads only | reads only | reads only |
| gpt-oss-20b | 13.25 | gpt-oss-20b-MXFP4.gguf | 681 / 686 / 475 | 413 / 391 / 230 | 238 / 215 / 117 |
| qwen3-30b-a3b | 2.86 (per layer, 2.65–3.06) | Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf | 426 / 411 / 249 | 212 / 196 / 102 | 103 / 95 / 46 |
| gpt-oss-120b | 13.25 | gpt-oss-120b-MXFP4.gguf | 711 / 665 / 402 | 376 / 342 / 182 | 202 / 191 / 91 |
| mixtral-8x7b (C = 2, 3, 4) | 106.66 | Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf | 4292 / 4398 / 3013 | 3427 / 3429 / 2098 | 2614 / 2562 / 1422 |
| deepseek-v2-lite | 9.19 (Q8_0: the only header on file) | DeepSeek-V2-Lite-Chat.Q8_0.gguf | 999 / 1002 / 682 | 760 / 744 / 430 | 569 / 543 / 280 |
| qwen1.5-moe | n/a (Q4_K_M-equivalent estimate 5.24 MB) | — | reads only | reads only | reads only |
| qwen2-57b | 16.67 | qwen2-57b-a14b-instruct-q4_k_m.gguf | 2595 / 2600 / 1845 | 2039 / 2042 / 1193 | 1584 / 1562 / 794 |
| phi3.5-moe | 47.62 | Phi-3.5-MoE-instruct-Q4_K_M.gguf | 1726 / 1690 / 1255 | 1147 / 1090 / 691 | 761 / 705 / 405 |

Hit rates (LRU / DF κ=0 / Belady, %) at E/8, E/4, 3E/8: olmoe 47/47/63, 65/66/80, 76/78/88; gpt-oss-20b 46/46/63,
68/69/82, 81/83/91; qwen3 61/63/77, 81/82/91, 91/91/96; gpt-oss-120b 63/65/79, 80/82/90, 89/90/95; mixtral 37/36/56,
50/50/69, 62/63/79; deepseek 30/30/52, 47/48/70, 60/62/80; qwen1.5 25/28/50, 40/43/67, 55/58/79; qwen2 31/30/51,
46/45/68, 58/58/79; phi 44/45/59, 63/64/77, 75/77/87.

## 2. Drift of expert popularity

Per layer, the top-C experts by request count in consecutive windows of 100 and of 1,000 tokens; the entry is the
fraction of the set that changes from one window to the next, averaged over layers. `share` is the fraction of a
layer's requests that its C most requested experts of the whole trace receive (uniform routing would give C/E:
12.5%, 25%, 37.5%). `random` is 1 − C/E, the change between two unrelated sets.

| model | E/k | C | share | change / 100 tok | change / 1000 tok | random |
|---|---|---|---|---|---|---|
| olmoe | 64/8 | 8 | 31% | 31% | 58% | 88% |
| olmoe | 64/8 | 16 | 50% | 28% | 48% | 75% |
| olmoe | 64/8 | 24 | 63% | 26% | 40% | 62% |
| gpt-oss-20b | 32/4 | 4 | 25% | 39% | 66% | 88% |
| gpt-oss-20b | 32/4 | 8 | 45% | 29% | 48% | 75% |
| gpt-oss-20b | 32/4 | 12 | 61% | 22% | 35% | 62% |
| qwen3-30b-a3b | 128/8 | 16 | 26% | 35% | 70% | 88% |
| qwen3-30b-a3b | 128/8 | 32 | 43% | 28% | 57% | 75% |
| qwen3-30b-a3b | 128/8 | 48 | 58% | 22% | 44% | 62% |
| gpt-oss-120b | 128/4 | 16 | 36% | 40% | 60% | 88% |
| gpt-oss-120b | 128/4 | 32 | 56% | 33% | 45% | 75% |
| gpt-oss-120b | 128/4 | 48 | 71% | 29% | 34% | 62% |
| mixtral-8x7b | 8/2 | 2 | 29% | 49% | 60% | 75% |
| mixtral-8x7b | 8/2 | 3 | 42% | 40% | 50% | 62% |
| mixtral-8x7b | 8/2 | 4 | 55% | 32% | 40% | 50% |
| deepseek-v2-lite | 64/6 | 8 | 17% | 53% | 75% | 88% |
| deepseek-v2-lite | 64/6 | 16 | 31% | 45% | 64% | 75% |
| deepseek-v2-lite | 64/6 | 24 | 44% | 38% | 53% | 62% |
| qwen1.5-moe | 60/4 | 8 | 19% | 58% | 73% | 87% |
| qwen1.5-moe | 60/4 | 15 | 32% | 50% | 62% | 75% |
| qwen1.5-moe | 60/4 | 23 | 46% | 41% | 51% | 62% |
| qwen2-57b | 64/8 | 8 | 30% | 52% | 46% | 88% |
| qwen2-57b | 64/8 | 16 | 45% | 48% | 42% | 75% |
| qwen2-57b | 64/8 | 24 | 58% | 38% | 33% | 62% |
| phi3.5-moe | 16/2 | 2 | 22% | 36% | 67% | 88% |
| phi3.5-moe | 16/2 | 4 | 39% | 33% | 56% | 75% |
| phi3.5-moe | 16/2 | 6 | 53% | 27% | 46% | 62% |

## 3. Intervals on the W50 exponent

From `prereg/foresight/foresight_S.json` with the fitting code of `scripts/fig_foresight.py` (W50 = a (C/k)^b in
log–log over the 26 model × budget points; 9 of them have W50 < 1, interpolated between W = 0 and W = 1 because one
token of foresight already closes more than half the gap). These use the foresight sweep's own `_pol` optimum and
are unaffected by the revision above.

| fit | exponent b | prefactor a | r | n |
|---|---|---|---|---|
| all points (the paper's fit) | 1.39 | 0.51 | 0.975 | 26 |
| bootstrap over models, 2,000 resamples of the 9 models with replacement, seed 0 | 95% [1.15, 1.52], median 1.38, sd 0.095 | 95% [0.45, 0.61] | | |
| bootstrap over the 26 points (reference; ignores the grouping) | 95% [1.21, 1.53] | | | |
| leave one model out | 1.30 (without gpt-oss-120b) to 1.43 (without Qwen1.5-MoE); jackknife SE 0.11 | 0.49–0.54 | 0.964–0.978 | 23–24 |
| interpolated points excluded | 1.51 | 0.43 | 0.971 | 17 (C/k 1.75–16) |
| … with the bootstrap over models | 95% [1.08, 1.77], median 1.51, sd 0.17 | 95% [0.31, 0.69] | | |

No resample of the full set gives b ≤ 1 (minimum 1.07); without the interpolated points 0.6% of resamples do.

## Reading

- **Recency beats frequency, but the winning policy is recency-weighted frequency.** Classic LFU is never better
  than LRU except on Qwen2-57B, and reads 3–173% more at C/k ≥ 3 (Qwen3 at 3E/8: 6.2× Belady against LRU's 2.3×):
  whole-history counts lock in the experts of the first conversations. Decayed frequency with κ = 0 and S3-FIFO,
  which both forget, read the fewest experts (0.97–1.02 of each other in every cell); ARC reads 0–4% fewer than LRU.
- **Where LRU ties.** At C/k ≤ 1.5 (seven cells, union-to-capacity 2.5–5.7: the 16-token working set is several
  times the cache) LRU, LFU, ARC, S3-FIFO and decayed frequency are within 4% of one another (LRU against decayed
  frequency within ±2.5%) — the cache turns over every token or two, there is nothing to remember, and every online
  policy sits at 1.35–1.65× Belady. From C/k = 2 on, LRU reads 2–11% more than decayed frequency on eight models.
  The exception is Qwen2-57B, where LRU is within 1.2% of decayed frequency at all three budgets and LFU reads 5–8%
  fewer than either at E/4 and 3E/8: its expert popularity is stable over the whole trace (the only model whose
  1,000-token drift is below its 100-token drift, and where a static cache profiled on the first 10% ties LRU:
  1.43 / 1.74 / 2.00 against 1.41 / 1.71 / 1.99).
- **The hysteresis deployed in the engine costs reads.** The deployed κ reads 1–8% more than κ = 0, and 12–22%
  more on Qwen3 at E/4 and 3E/8; κ was tuned for the engine's time model (copies run in the background, the CPU runs
  the miss), not for the read count, and the two objectives differ most where the cache is large. The engine's
  policy as shipped, whose admissions are separate copies, is 1.46–3.25× Belady (JSON row `dfa-deployed`).
- **A profiled static cache is the wrong answer on eight of nine models:** 1.7–13× Belady, and 1.1–5.9× LRU; a
  static cache chosen in hindsight is still 1.6–10× Belady on those eight (JSON row `static-hindsight`). Only on
  Qwen2-57B (stable popularity, flat routing: 46 distinct experts of 64 in 16 tokens at k = 8) does it match LRU.
- **Drift.** Between consecutive 100-token windows 22–58% of a layer's top-C set changes (median 33% at E/4);
  between 1,000-token windows 33–75% (median 50%). On eight of nine models the 1,000-token drift is *larger* than
  the 100-token drift, although the longer window has a quarter of the sampling noise: popularity moves with the
  conversation (mean response 450–950 tokens, so consecutive 1,000-token windows are different conversations). The
  skew is mild: the top E/4 experts take 31–56% of requests (uniform: 25%), the top E/8 17–36% (uniform: 12.5%);
  DeepSeek-V2-Lite and Qwen1.5-MoE are the flattest and drift most (73–75% per 1,000 tokens at E/8, close to the
  87–88% of unrelated sets), which is why frequency has least to offer there.
- **Foresight dominates policy.** Seeing four tokens ahead is within 13% of Belady in the median (1.13) and beats
  every online policy in every cell; W = 16 is within 2% of Belady at C/k ≤ 3, 3–7% short at C/k = 4–6 (Qwen3 E/4
  1.07, gpt-oss-120b E/8 1.03, Qwen1.5 3E/8 1.05) and 32–66% short at C/k ≥ 6 (Qwen3 3E/8 1.32, gpt-oss-120b E/4
  1.34 and 3E/8 1.66), as the W50 law says. The spread between the best and worst reasonable online policy (S3-FIFO
  against LRU) is 0–9%; the spread between the best online policy and Belady is 35–110%.
- **The W50 exponent is superlinear but not sharp.** 1.39 with the paper's fit; 1.15–1.52 under a bootstrap over
  models, 1.30–1.43 leaving one model out (gpt-oss-120b, the only model at C/k = 16, anchors the upper end), and
  1.51 (1.08–1.77) when the nine interpolated small-cache points are dropped: those points flatten the slope,
  lying above the law at C/k = 1 (W50 0.57–0.73 against 0.51) and below it at C/k = 2 (0.94–0.98 against 1.34).
  Report the exponent as about 1.4 with an interval of roughly ±0.2, and the claim that survives every refit as
  "W50 grows faster than C/k".

**Caveats.**
- `opt` is the exact MIN with bypass of `mosl/cachesim.py` (optimal against exhaustive search for set requests): it
  serves a token's hits first and may then evict an expert already served this step to admit another of the same
  token. `scripts/foresight.py`'s `_pol`, the optimum of job 084 and of the foresight sweep, never evicts a resident
  of the current token and reads 0.5–9.5% more (most at C = k). The first version of this study used `_pol` as the
  reference, and its windowed policies, which inherited the same restriction, undercut it by up to 1.4% in 12
  cells; the windowed policies now share the optimum's freedom (hits served first, any resident evictable) and
  nothing undercuts `opt`. The online policies keep the stricter rule, which is what a real cache does.
- ARC and S3-FIFO were adapted to atomic replay (a resident requested by the current token is never the victim,
  so a step's misses never evict its hits); with C ≥ k this never forces a bypass. S3-FIFO's small queue is one slot
  at C < 15, which is what its 10% rule gives on caches this small; the algorithm follows the reference
  implementation (eviction from main only when it is over its share).
- Drift counts a set change whether it is real or sampling noise; ties in the per-window counts are broken by
  expert index. The comparison of 100- against 1,000-token windows is robust to this (noise alone would make the
  longer window drift less), the absolute levels less so.
- Bytes for DeepSeek-V2-Lite are at Q8_0 (the only header on file); the other six are at the quantisation the
  paper uses (MXFP4 for gpt-oss, Q4_K_M otherwise). OLMoE and Qwen1.5-MoE have no GGUF header on file.
- Mixtral's budgets (2, 3, 4 of 8 experts) are 25/37.5/50%, not 12.5/25/37.5%.
