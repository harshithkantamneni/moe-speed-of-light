# Turn the oracle into a forecaster specification

As of 6 October 2026, the two measurement contributions of "Where the Seconds Go" are still unoccupied. No one converts an exact-routing MIN-with-bypass miss count into a seconds-domain limit from measured host and PCIe rates, and no one runs a randomised oracle factorial inside a real engine. The trace study and the learned admission order, however, are now heavily precedented:

- an ICLR 2026 lookahead-window oracle over 20 models (2505.16056);
- a 44–46% gap between causal policies and MIN-with-bypass (2608.07911);
- FlashMoE's learned eviction trained to imitate Belady;
- SAEM's in-engine clairvoyant oracle;
- MoE-SpAc's use of draft tokens as a lookahead sensor;
- SeqMoE's eight-step routing forecasts.

An upstream llama.cpp PR opened on 3 October (#29887) also adds a GPU expert cache that admits every miss into an LRU. The 9/10 that both round-5 reviewers tie to "a realisable mechanism that recovers at least a third of the oracle's 16–51%" is unlikely before the 30 October MLSys deadline; I put it at about **10%**. A forecaster needs roughly **0.7 precision and recall four or more decode steps ahead at C = E/8** to close a third of the read gap. Every realisable source I could measure or find falls short. Layer-ahead predictors reach only 0.08 token. A linear full-stack forecaster trained here on the paper's own traces closes at most **7% (gpt-oss-120b) and 14% (Qwen3-30B-A3B)**.

Checks run for this report found something reviewers would also find: **admitting every miss beats the paper's online baseline in 24 of 26 trace cells**, by up to 6.7% fewer reads (9.8% on the AIME workload the engine runs). On Qwen3 at C = 16 it also captures 81% of the learned order's offline gain. The W50 law survives the stronger baseline, refitting to 0.67 (C/k)^1.31 with a tighter interval. W50 also collapses onto a window that holds about **0.61·C distinct experts**. That one-constant rule predicts held-out models better than either the power law or a naive proportional model.

The realistic target is therefore a 7, not a 9. Fix the five cheap weaknesses: order-free Shapley accounting, the machine as the statistical unit, one measured bound, an honestly named calibrated model, and a rewrite. Recast W50 as a specification for forecasters. Spend the $25 on a ten-host variance study, one long-shot experiment (a cache-only shadow run-ahead), and a third model family.

## The bound and the in-engine factorial survive; the trace study does not

The field kept growing fast through the window. Two arXiv sweeps on 6 October returned 315 hits dated June 2025 to October 2026. About 60 of them are relevant offloading systems, appearing at roughly four a month and accelerating through August–October. The large multipliers (3–15×) come almost entirely from four sources:

- lossy designs, such as Mira's expert compression ([arXiv 2609.38090](https://arxiv.org/abs/2609.38090)) and RapidMoE's importance arbitration ([arXiv 2610.01265](https://arxiv.org/abs/2610.01265));
- speculative coupling, such as MoE-SpeQ, S2-MoE and DraftExpert;
- weak baselines (DeepSpeed, MoE-Infinity, vLLM static offload);
- SSD tiers.

The related-work section should therefore keep two lists. One holds "exact-routing, one-token-per-pass" systems, to which the bound applies. The other holds everything else.

The exact-routing consumer-GPU systems that are not yet cited are listed below.

- **OSDI'26 CPU-GPU hybrid.** It reaches **28 tok/s** on INT4 DeepSeek-V3 with one or two RTX 5090s, a 1.25× improvement over KTransformers. Its motivation is that KTransformers' decode "is only about 50% of the nominal aggregate DDR5 bandwidth" ([arXiv 2606.10493](https://arxiv.org/abs/2606.10493)). It frames results as a fraction of peak but offers no policy bound.
- **ATSInfer.** It extends llama.cpp and runs **gpt-oss-120b on an RTX 4090** against a llama.cpp "primary baseline", claiming up to 3.29×. The baseline's configuration is only "the same offloading policy that fills GPU memory as much as possible", and no `--n-cpu-moe` setting appears ([arXiv 2607.10183](https://arxiv.org/abs/2607.10183)).
- **Pipelined Sharding (MLSys'26).** It claims TPS up to 30× over an "aggressive" llama.cpp baseline. Its "oracle comparison across 105 configurations" is a configuration oracle, not a routing oracle ([arXiv 2604.26334](https://arxiv.org/abs/2604.26334)).
- **WiSP.** It reaches 2.0× vLLM static offload and finds that prefetch "does not help in single-stream decode" ([arXiv 2606.21868](https://arxiv.org/abs/2606.21868)).
- **SSD-LLaMA** ([arXiv 2609.18110](https://arxiv.org/abs/2609.18110)).
- **KTransformers v0.5.1** dynamic expert placement: 70.22 vs 56.57 tok/s at 10% GPU experts on Qwen3-Next-80B with four RTX 4090s ([KTransformers](https://github.com/kvcache-ai/ktransformers)).

None of these reports gpt-oss-120b on an RTX 5090 against an equal-memory `--n-cpu-moe` baseline. The paper's same-machine race against FreeToken is still the only one of its kind.

The most consequential new item for the system contribution is llama.cpp **PR #29887**, "llama : add a GPU cache for MoE experts kept in host memory". Aman Gupta, a llama.cpp CUDA contributor, opened it on 3 October 2026. It ports the QVAC-fabric cache:

- an LRU in which "only the misses are uploaded", so every miss is admitted;
- banks shared across layers with the same expert layout;
- a `--moe-cache-mib` flag;
- a diff of +748/−14 lines over 13 files ([PR #29887](https://github.com/ggml-org/llama.cpp/pull/29887)).

Its merge state could not be read. Either way, maintainers now own a design that differs from the paper's cache on exactly the axes the paper says matter: admission filtering, CPU-or-PCIe service of misses, and per-layer budgets. It must be cited as concurrent work. Because it builds on the same engine, it is a cheap extra arm for the race.

| Contribution | Status, 6 Oct 2026 | Closest new prior art | What remains distinctive |
|---|---|---|---|
| (a) Speed limit in seconds | **Intact**; cite five | Budgeting Bytes (exposed-latency bound for a *given* fetch set, "no schedule beats bytes/B") ([arXiv 2609.04238](https://arxiv.org/abs/2609.04238)); WiSP (LRU miss curve × seconds per miss, an estimator); Paging the Experts ("not an online speed prediction") ([arXiv 2609.29032](https://arxiv.org/abs/2609.29032)); Euro-Par'25 layered paging (OPT gap "up to ×2.5", in faults) ([arXiv 2509.02408](https://arxiv.org/abs/2509.02408)); 2608.18261 (bytes over bandwidth matches llama.cpp `-ncmoe`) ([arXiv 2608.18261](https://arxiv.org/abs/2608.18261)) | MIN with bypass over all exact-routing policies; measured host and PCIe rates; misses served by CPU or PCIe; result in seconds |
| (b) llama.cpp expert cache | **Contested** | PR #29887 (upstream, LRU, admit every miss); community caches reporting gains over `--n-cpu-moe`, e.g. "+21% to +52%" on an RTX 5090 ([Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)) and 1.79× on gpt-oss-120b ([PR #21609](https://github.com/ggml-org/llama.cpp/pull/21609)); FreeToken ([arXiv 2608.16157](https://arxiv.org/abs/2608.16157)) | Decayed frequency, a CPU/PCIe split chosen per machine, per-layer budget. Single-read bypass loses in reads (next section), so lead with the split, not the filter |
| (c) In-engine oracle factorial | **Intact**; cite four | SAEM's in-engine clairvoyant oracle, η_TP 87.9–92.8% at batch 1, i.e. the oracle is worth at most about 14% ([arXiv 2608.21614](https://arxiv.org/abs/2608.21614)); Budgeting Bytes' trace oracle (0.12 → 0.13 tok/s); 2608.12103 (perfect one-layer advice +5.0%) ([arXiv 2608.12103](https://arxiv.org/abs/2608.12103)); WiSP (prefetch hurts) | Foresight spent as MIN spends it vs spent as Belady prefetch, a byte accounting (1.6–2.5× MIN's bytes), randomised and cold on three hosts |
| (d) Nine-model traces, online vs MIN, W50 | **Most exposed**; cite six | 2505.16056 (ICLR'26) ([arXiv 2505.16056](https://arxiv.org/abs/2505.16056)); 2608.07911 ([arXiv 2608.07911](https://arxiv.org/abs/2608.07911)); FlashMoE ([arXiv 2601.17063](https://arxiv.org/abs/2601.17063)); Euro-Par'25; SeqMoE's Belady-8 ([arXiv 2609.12978](https://arxiv.org/abs/2609.12978)); 2608.12103 | The W50 scaling (no one fits lookahead against C/k); MIN with bypass in bytes over current models including gpt-oss; the model's own generations as traces; the distinct-expert rule below |
| (e) Learned admission | **Exposed**; cite two | FlashMoE (learned eviction from recency and frequency features with Belady labels, +7% speed on Qwen3-30B-A3B); 2608.07911 (a causal next-use predictor "recovers -11.4% of the gap") | An honest in-engine null (0.90–1.05×); much of its offline gain is admission alone (next section) |

The trace study needs the most careful repositioning.

**2505.16056 (ICLR 2026).** It measures, across 20 MoE models, the hit rate of an oracle cache that "evicts experts that are activated the least times in the next m tokens". It plots this against the cache ratio ρ, "the ratio between the cache size and the number of activated experts", which is the paper's C/k. For one reference model it reports LRU at 56–75% and LFU at 62–78% of the clairvoyant optimum, and it recommends caches "approximately twice the active experts" ([arXiv 2505.16056](https://arxiv.org/abs/2505.16056)). It differs from the paper in four ways: mandatory admission, hit-rate units, teacher-forced corpus text, and no fitted horizon law. The paper should use the same ρ symbol and check W50 against SCH(m, ρ) for the overlapping models (Mixtral, DeepSeek-V2-Lite, Qwen1.5-MoE, OLMoE). That check is free validation.

**2608.07911.** It already reports "a stable gap to the offline optimum … (44.2-45.9%)" with bypass. It attributes "84.3-96.6% of it to knowing which resident expert is used furthest in the future" ([arXiv 2608.07911](https://arxiv.org/abs/2608.07911)). The qualitative claim that online policies read 35–110% more than MIN is therefore not new. Only its time-domain and multi-model form is.

**FlashMoE (v2, 5 October).** Belady reaches an 86% hit rate against LRU's 73%, "resulting in nearly 1.9× more I/O operations" ([arXiv 2601.17063](https://arxiv.org/abs/2601.17063)).

**MoE-SpAc.** It "repurpose[s] Speculative Decoding … as an informative lookahead sensor for memory management". On Qwen3-30B-A3B with a Qwen3-4B draft at batch 1, it gains 42% over llama.cpp with speculative decoding ([arXiv 2603.09983](https://arxiv.org/abs/2603.09983)). The paper's finding that batched verification "does not substitute for foresight" must be contrasted with this result explicitly. The contrast is that MoE-SpAc's baseline already pays for verification, while the paper charges rejected drafts against a non-speculative cache.

The bound's theory needs citations rather than a new theorem. Review 5a asks whether MIN with bypass is optimal when all k experts of a layer arrive at once ([Review 5a](/home/claude/moe-speed-of-light/reports/Review%205a%20MLSys%20PC%20%286%20October%29.md)). The argument has three steps:

- With bypass, each reuse interval is cached or not independently. Maximising hits is then the maximum C-colourable subgraph of an interval graph, which greedy solves exactly when unweighted ([Carlisle & Lloyd](https://www.martincarlisle.com/publications/kcoloring.pdf)). Simultaneous requests only tie left endpoints.
- Hawkeye's OPTgen computes OPT this way and "assumes that misses will bypass the cache" ([Jain & Lin, ISCA 2016](https://cs.utexas.edu/~akanksha/isca16.pdf)).
- When bypass and admission cost different amounts, the exact optimum becomes a min-cost flow and is no longer Belady. CHOPT beats Belady on latency "by 8.2%-44.8%" when the slow tier can serve directly ([CHOPT](https://geraldleizhang.com/publications/CHOPT_Sigmetrics20.pdf)). That is why the paper uses MIN's miss count as an input to a bound rather than as a time-optimal policy.

The finding that Belady-style prefetch reads 1.6–2.5× MIN's bytes also has classical twins: Cao et al.'s Conservative vs Aggressive integrated prefetching ([SIGMETRICS 1995](https://homes.cs.washington.edu/~karlin/papers/sigmetrics.pdf)), and Jain & Lin's MIN vs Demand-MIN, where traffic rises from 45.4 to 79.4 transfers per kilo-instruction (+75%) ([ISCA 2018](https://cs.utexas.edu/~akanksha/isca18.pdf)). Citing them makes the result look expected rather than anomalous, which is what a theory reviewer wants.

## Admitting every miss beats the paper's own online baseline

The W50 law and the in-engine oracle gains are normalised to an online policy: decayed frequency with a bypass threshold κ, keeping whichever of κ = 0 or the model's κ reads less. Simply setting κ = −∞ is stronger. Under that rule every miss is copied, and the lowest-scored resident is evicted. I re-ran the paper's own simulator (`scripts/foresight.py::_pol`) in three frames ([always_admit_S.json](/home/claude/moe-speed-of-light/research_notes/MoE%20paper%20path%20to%20a%209/checks_2026-10-06/always_admit_S.json); [aa_policy_study.json](/home/claude/moe-speed-of-light/research_notes/MoE%20paper%20path%20to%20a%209/checks_2026-10-06/aa_policy_study.json); [aa_aime.txt](/home/claude/moe-speed-of-light/research_notes/MoE%20paper%20path%20to%20a%209/checks_2026-10-06/aa_aime.txt)):

| Frame | Always-admit beats the baseline | Largest read saving | Largest share of the online-to-MIN gap closed, with no foresight |
|---|---|---|---|
| Foresight study (9 models, own sampled text, baseline `dfa-fetch`) | **24 of 26 cells**; loses only at C/k = 1 (OLMoE −1.1%, Phi-3.5 −1.6%) | 6.7% (Qwen3 C=16; OLMoE C=16) | 17.9% (Qwen3 C=16) |
| Policy study (best of LRU, LFU, ARC, S3-FIFO and two decayed-frequency variants; last 90% of each trace) | 19 of 27 cells, 3 ties, 5 losses of at most 2.7% | 5.8% (OLMoE C=24) | n/a |
| AIME-25 routing that the engine runs (gpt-oss-120b C=14/32/51; Qwen3 C=16/32/56) | 5 of 6 cells, 1 tie | **9.8%** (Qwen3 C=16) | 24.6% (Qwen3 C=16) |

For the two engine models on their own text, the saving is **3.3% / 0.3% / 0.0%** for gpt-oss-120b at C = 16/32/64, and **6.7% / 5.0% / 0.1%** for Qwen3 at C = 16/32/64. So the effect is the 3–7% expected at low budgets, and it vanishes by C = E/2.

The paper's headline range survives. With always-admit added, the best online policy still sits **1.35–2.10× above MIN**, because the two extreme cells are ones where always-admit does not win. The internal story does not survive: always-admit becomes the best online policy in 19 of 27 cells, displacing the reported "decayed frequency best in 12 cells and S3-FIFO in 10" ([Review 5a](/home/claude/moe-speed-of-light/reports/Review%205a%20MLSys%20PC%20%286%20October%29.md)).

The finding matters most for the learned admission order. On the AIME routing, against the single-read online policy, always-admit closes:

- **24.6% of the gap at Qwen3 C = 16, where the learned order closes 30.2%** (81% of its gain);
- 11.6% at Qwen3 C = 32, against 23.6%;
- at most 3.8% on gpt-oss-120b, where the learned order closes 16.8–23.1% ([learned_offline.json](/home/claude/moe-speed-of-light/prereg/learned_offline.json); [aa_aime.txt](/home/claude/moe-speed-of-light/research_notes/MoE%20paper%20path%20to%20a%209/checks_2026-10-06/aa_aime.txt)).

On Qwen3, then, most of what the logistic regression "learned" is to admit more. On gpt-oss-120b it did learn a better eviction ranking. That gain still bought nothing in the engine (0.90–1.05×), which fits FlashMoE's small +7% on Qwen3-30B-A3B.

The mechanism is simple. At low budgets almost every resident is churned within a few steps. The newest expert has just been used, and same-layer reuse between adjacent tokens runs at 0.31 (gpt-oss-120b) to 0.47–0.50 (the other models) ([provenance.md](/home/claude/moe-speed-of-light/prereg/provenance/provenance.md)). Admitting it is therefore a good bet, and the κ filter discards that bet. In the pilot simulator, a "persistence" forecast (next steps reuse this step's experts) gives exactly the same reads as always-admit, to 0.01 ([pilot_forecast_baselines.py](/home/claude/moe-speed-of-light/research_notes/MoE%20paper%20path%20to%20a%209/pilot_forecast_baselines.py)). This tunes admission; it is not foresight.

Two caveats keep this a "may". First, these are read counts under single-read admission. The deployed engine serves every miss on the CPU and copies admissions in the background, reading them twice: 191.6 against 166.3 reads per token at Qwen3 C = 16 on AIME ([aa_aime.txt](/home/claude/moe-speed-of-light/research_notes/MoE%20paper%20path%20to%20a%209/checks_2026-10-06/aa_aime.txt)). There the κ filter saves real second reads. Second, single-read admission (`foa`) in the engine ran **0.97–1.03× the deployed cache at the host-bound cells** ([Review 5a](/home/claude/moe-speed-of-light/reports/Review%205a%20MLSys%20PC%20%286%20October%29.md)). Whether always-admit wins in seconds is untested. It costs one extra arm in the next rentals.

Upstream PR #29887 admits every miss. This result says that is a defensible choice, not a naive one.

The W50 law survives the stronger baseline and becomes slightly more robust. I added κ = −∞ to the set of κ values the paper already minimises over, then re-ran the exact pipeline (`scripts/foresight_exact.py` functions with `scripts/policy_study.w50_intervals`) ([w50_A_min.json](/home/claude/moe-speed-of-light/research_notes/MoE%20paper%20path%20to%20a%209/checks_2026-10-06/w50_A_min.json); [w50_exact.json](/home/claude/moe-speed-of-light/prereg/foresight/w50_exact.json)):

| Fit property | Original (n = 26) | With κ = −∞ added |
|---|---|---|
| Fitted law | 0.589 (C/k)^1.328, r = 0.971 | **0.667 (C/k)^1.311**, r = 0.977 |
| Model-bootstrap exponent interval | 1.09–1.47 | 1.10–1.43 |
| Leave-one-model-out exponents | 1.24–1.37 | 1.25–1.36 |
| Exponent without interpolated points | 1.51 | **1.35** (n = 21) |

Per point, W50 rises by a median of 9% (maximum 42%), and the fitted value at C/k = 8 moves from 9.3 to 10.2 tokens. Review 5b's W9 objected that the exponent "rises from 1.33 to 1.51 when the six interpolated points below one token are dropped" ([Review 5b](/home/claude/moe-speed-of-light/reports/Review%205b%20professor%20%286%20October%29.md)). Most of that fragility disappears under the honest baseline.

W50 also has a more physical form than a power law in tokens. Classical paging theory says lookahead helps only when measured in *distinct* future pages ("strong lookahead") ([Albers, Algorithmica 1997](https://link.springer.com/doi/10.1007/PL00009158)). Breslauer adds that "the competitive ratio is a function of k + l", meaning an extra page of cache and an extra page of foresight are worth the same ([Breslauer](https://tidsskrift.dk/brics/article/download/19951/17604/45315)). So I measured D(W), the mean number of distinct experts per layer in a W-token window, on all nine models. I then evaluated D at each point's W50 ([distinct_orig.json](/home/claude/moe-speed-of-light/research_notes/MoE%20paper%20path%20to%20a%209/checks_2026-10-06/distinct_orig.json)).

W50 in tokens varies about 50-fold across the 26 points, from 0.64 to 33.6. Yet **half the gap closes when the window holds 0.61·C distinct experts** (IQR 0.57–0.66, coefficient of variation 0.16; 0.65·C under the always-admit baseline). The per-model working-set exponent β (D ∝ W^β over W = 1–16) ranges from 0.48 to 0.78. Since W50 ∝ (C/k)^(1/β) under this rule, that range explains why the token exponent sits near 1.3 and wanders with the model mix.

Tested leave-one-model-out, the rule beats both alternatives ([lomo.py](/home/claude/moe-speed-of-light/research_notes/MoE%20paper%20path%20to%20a%209/checks_2026-10-06/lomo.py)):

| Predictor of a held-out model's W50 | Parameters | Median error factor | 90th-percentile error factor | Maximum |
|---|---|---|---|---|
| Naive proportional, W50 ∝ C/k | 1 | 1.24× | 1.69× | 2.83× |
| Paper's power law in C/k | 2 | 1.20× | 1.57× | 1.82× |
| **Distinct-expert rule, D(W50) = 0.61·C** | 1 + the model's D(W) curve | **1.15×** | **1.38×** | 1.61× |

This answers in advance the "law that does not beat a naive model" objection, applied to W50. It also ties the result to 2608.07911's demand that "the per-step expert union relative to per-layer capacity must be reported" ([arXiv 2608.07911](https://arxiv.org/abs/2608.07911)): D(W)/C is exactly a windowed union-to-capacity ratio. It turns W50 into a forecaster target stated in the units a forecaster controls: forecast the next ~0.6·C distinct experts per layer. Present the power law as a descriptive fit and the distinct-expert rule as the predictive statement.

## No realisable forecaster reaches four steps at 0.7 precision

The pilot value map answers Review 5b's Q12 ("Can you bound what a predictor of a given precision and recall would buy?") and Review 5a's Q5 almost word for word. It drives the paper's windowed policy with forecasts instead of the true future, and normalises to always-admit so foresight is never credited for bad admission tuning ([pilot_forecast_sim.py](/home/claude/moe-speed-of-light/research_notes/MoE%20paper%20path%20to%20a%209/pilot_forecast_sim.py); [pilot_forecast_gpt-oss-120b.json](/home/claude/moe-speed-of-light/research_notes/MoE%20paper%20path%20to%20a%209/pilot_forecast_gpt-oss-120b.json); [pilot_forecast_qwen3-30b-a3b.json](/home/claude/moe-speed-of-light/research_notes/MoE%20paper%20path%20to%20a%209/pilot_forecast_qwen3-30b-a3b.json)).

The table gives the share of the read gap closed on about 21,000 decode steps of dataset text per model. "Noisy r" keeps each true expert with probability r and fills the rest with random ones, so precision = recall = r.

| Forecast | gpt-oss-120b C=16 (E/8) | gpt-oss-120b C=32 (E/4) | Qwen3 C=16 (E/8) | Qwen3 C=32 (E/4) |
|---|---|---|---|---|
| Exact, W = 1 / 2 / 4 / 8 | 0.15 / 0.28 / 0.48 / 0.74 | 0.06 / 0.14 / 0.26 / 0.42 | 0.32 / 0.55 / 0.83 / 0.99 | 0.06 / 0.21 / 0.42 / 0.67 |
| Noisy r = 0.9 / 0.7 / 0.5 at W = 4 | 0.42 / **0.31** / 0.19 | 0.23 / 0.16 / 0.09 | 0.73 / 0.54 / 0.35 | 0.36 / 0.24 / 0.12 |
| Noisy r = 0.9 / 0.7 / 0.5 at W = 8 | 0.64 / 0.47 / 0.30 | **0.37** / 0.28 / 0.18 | 0.88 / 0.68 / 0.47 | 0.58 / 0.42 / 0.27 |
| Token table with a perfect drafter, W = 4 | −0.02 | −0.02 | −0.10 | −0.18 |

The one-third bar sits at different places for the two models:

- **gpt-oss-120b, C = E/8:** precision and recall of about 0.7 over 4–8 steps.
- **gpt-oss-120b, C = E/4:** about 0.9 over 8 steps.
- **Qwen3, C = E/8:** about 0.5 over 4 steps.
- **Qwen3, C = E/4:** about 0.7 over 8 steps.

These are read shares, not seconds. The bytes term dominates the measured seconds (next section), so the conversion is favourable but not one to one.

Most realisable forecasters fail that bar by a wide margin. I trained a ridge regression from the full all-layer routing vector at step t to the routing at t+h, folded by conversation, on the same traces ([ridge_full.py](/home/claude/moe-speed-of-light/research_notes/MoE%20paper%20path%20to%20a%209/checks_2026-10-06/ridge_full.py); [ridgefull_gpt-oss-120b.json](/home/claude/moe-speed-of-light/research_notes/MoE%20paper%20path%20to%20a%209/checks_2026-10-06/ridgefull_gpt-oss-120b.json); [ridgefull_qwen3.json](/home/claude/moe-speed-of-light/research_notes/MoE%20paper%20path%20to%20a%209/checks_2026-10-06/ridgefull_qwen3.json)). This is the simplest test of SeqMoE's premise that the joint full-stack state predicts future routing.

Its precision@k is **0.40 at h = 1 falling to 0.29 at h = 4** for gpt-oss-120b, and **0.55 falling to 0.37** for Qwen3. That beats persistence (0.31 and 0.46 at h = 1), but by h = 2 it is worse than a plain decayed-frequency ranking. Decayed frequency holds near 0.35 and 0.45 at every horizon, and the online policy already uses it. Plugged into the policy, the ridge forecasts close at most **0.07 (gpt-oss-120b) and 0.135 (Qwen3)** of the gap.

Its recall@(k+3) at h = 1 is **0.52 and 0.63**, against decayed frequency's 0.50 and 0.56 ([ridge_recall_k3.txt](/home/claude/moe-speed-of-light/research_notes/MoE%20paper%20path%20to%20a%209/checks_2026-10-06/ridge_recall_k3.txt)). SeqMoE reports **0.91 on both models with a Mamba2 sequence model** ([arXiv 2609.12978](https://arxiv.org/abs/2609.12978)). Its claimed lead of 0.28–0.39 recall over a linear forecaster on the same two models is the single most important unverified number for this paper.

| Source | Cross-token horizon | Best reported or measured accuracy on these models | Verdict for W ≈ 4–16 tokens at 0.7 |
|---|---|---|---|
| Layer-ahead gates and predictors (Mixtral-offloading through Speculating Experts, DuoServe, pre-attention) | 1–3 layers, at most 0.08 token | 84% / 97% top-4 / top-8 recall in the engine for gpt-oss-120b; ~90% recall@k on Qwen3 ([arXiv 2603.19289](https://arxiv.org/abs/2603.19289)); ExpertFlow fits asymptotes of 26–33% (gate reuse) and 60–65% (trained) by ~20 layers ([arXiv 2510.26730](https://arxiv.org/abs/2510.26730)) | **No.** Within-token paced prefetch, which the engine already has |
| Persistence and first-order reuse | 1 step | Adjacent-token reuse "2.0× chance" on Qwen3 ([arXiv 2608.18261](https://arxiv.org/abs/2608.18261)); 0.31 on gpt-oss-120b | **No.** Equivalent to always-admit |
| Token identity → expert table | Draft depth | Precision@k 0.35 even with perfect future tokens | **No.** Worse than no forecast |
| Linear full-stack (this report) | 1–8 steps | Precision@k 0.40 → 0.29 (gpt-oss-120b), 0.55 → 0.37 (Qwen3) | **No.** Closes ≤ 7% / 13.5% |
| SeqMoE (Mamba2 over full-stack vectors, 45K training traces) | 1–8 steps | Recall@(k+3) 90.8% (GPT-OSS-120B) and 90.9% (Qwen3-30B-A3B) at 1 step; "average recall drop of only 3.58 percentage points" over 8 steps; 3.46% overhead | **Maybe**, at C = E/8 only, if it reproduces. Precision is capped at 4/7 and 8/11 by the k+3 candidate set. Unreproduced, no code found |
| Draft or self-draft routing (SP-MoE, MoE-SpeQ, DraftExpert, MoE-SpAc) | γ draft tokens | 88–89% top-1 with twinned drafts ([arXiv 2510.10302](https://arxiv.org/abs/2510.10302)); 90.9% top-4 when teacher-forced ([arXiv 2511.14102](https://arxiv.org/abs/2511.14102)); gpt-oss-120b EAGLE3 accepts 2.10–2.75 of 3 drafted tokens ([NVIDIA](https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-short-context)) | **Unlikely.** W = 8 at 0.75 recall needs per-token acceptance ≥ 0.977 |
| MTP heads | 1 token | DeepSeek-V3 85–90% acceptance on the second token ([aman.ai](https://aman.ai/primers/ai/deepseek-v3/)); Qwen3.6-35B-A3B in llama.cpp 55% ([Jarvislabs](https://jarvislabs.ai/blog/qwen36-mtp-llamacpp-rtxpro6000)) | **No.** Horizon 1 caps it at 0.15–0.32, and neither engine model has MTP |
| Decoupled routers (Read-ME, SiDA) | 0 tokens (all layers of the current token) | Exact by construction, but Read-ME needs 1.04B tokens of refactoring ([arXiv 2410.19123](https://arxiv.org/abs/2410.19123)) | **No** for unmodified models |
| Cache-friendly routing (Skliar; StickyMoE; Temporally Extended MoE) | n/a | >50% fewer misses at +0.1–3% perplexity ([arXiv 2412.00099](https://arxiv.org/abs/2412.00099)); gpt-oss-20b switch rate from >50% to <5% at up to 90% of base accuracy ([arXiv 2604.20156](https://arxiv.org/abs/2604.20156)) | Changes the model; out of scope for an exact-routing bound |

Drafts fail for structural reasons, not for lack of accuracy. Target routers predict well only on the target's own hidden states, or on those of a structurally twinned model (SP-MoE's draft/target pairs reach cosine similarity 0.57–0.95) ([arXiv 2510.10302](https://arxiv.org/html/2510.10302v2)). Every published system obtains those states by **verification**. Verification reads the union of experts across drafted tokens, and that union is exactly what makes batch-1 speculation lose under offload. Cascade measured slowdowns of up to 1.5× (Mixtral on math, −54% at K = 3) ([arXiv 2506.20675](https://www.alphaxiv.org/abs/2506.20675.md)). A community benchmark on Qwen3.6-35B-A3B at batch 1 measured −15% with n-gram drafting and −39% to −54% with a 0.8B draft ([HackMD](https://hackmd.io/@thc1006/SJly6IE6Wx)).

Foresight is also only valid while the draft is right, so its value decays as α^j. With α = 0.85, the chance that the draft is still right is 0.52 at j = 4 and 0.27 at j = 8. MoE-SpAc already claims "drafts as a lookahead sensor" for eviction and prefetch ([arXiv 2603.09983](https://arxiv.org/abs/2603.09983)). The open slot is narrower: foresight from drafts *without verification reads*, scored against MIN in seconds.

The only unpublished candidate is a **cache-only shadow run-ahead** ([innovation_directions.md](/home/claude/moe-speed-of-light/research_notes/MoE%20paper%20path%20to%20a%209/innovation_directions.md)). In a host-bound step the GPU idles: on host 1 at C = 14, about 66 CPU misses × 215 µs ≈ 14 ms against about 4.6 ms of GPU work. The idea is to spend that idle time running the target model itself 1–4 tokens ahead, using only GPU-resident experts and renormalising the top-k, and to read the routers' logits as the forecast. It adds no host bytes and no extra weights, and the shadow tokens are never emitted. Its main risk is error compounding: at C = E/8 about 46% of gpt-oss-120b's selected experts are absent, and the errors compound over 36–48 layers. No source measures routing accuracy under resident-only execution, so the first test is offline.

Theory says how to use any such forecaster. Use decision-level information ("will this expert be needed within the window?"), which has exact consistency and a per-error cost ([Antoniadis et al., ICML 2023](https://arxiv.org/abs/2210.02775)), rather than next-use times. Wrap it in a follow-the-better fallback to the online policy ([Chłędowski et al., ICML 2021](https://arxiv.org/abs/2106.14693v1)). Report false negatives and false positives separately, since the two error types carry very different cost constants. SpecMD's warning that "higher prediction accuracy does not guarantee better cache performance" for expert caches makes the same point empirically ([arXiv 2602.03921](https://www.arxiv.org/pdf/2602.03921)).

## Reviewers grade the unit of analysis, not the decimal places

Both round-5 reviews scored 5/10 (weak reject), with novelty 3, soundness 3, rigour 4, clarity 2 and significance 3 ([Review 5a](/home/claude/moe-speed-of-light/reports/Review%205a%20MLSys%20PC%20%286%20October%29.md); [Review 5b](/home/claude/moe-speed-of-light/reports/Review%205b%20professor%20%286%20October%29.md)). The rigour score is already the paper's strength.

The SPCL standard the reviewers invoke is Hoefler and Belli's twelve rules ([SC15](https://www.unixer.de/publications/index.php?pub=222)). The rules that matter here are:

- Rule 3: summarise costs with the arithmetic mean and rates with the harmonic mean.
- Rule 4: "Avoid summarizing ratios".
- Rule 5: confidence intervals for nondeterministic data.
- Rule 7: compare "using non-overlapping confidence intervals or ANOVA".
- Rule 9: "Document all varying factors and their levels".
- Rule 11: "show upper performance bounds".
- Rule 12: plot what is needed.

Wording is taken from [Hoefler's slides](https://blogs.fau.de/hager/files/2017/02/hoefler-scientific-benchmarking-isc-pe-invited.pdf) and a [lecture reproduction](https://wr.informatik.uni-hamburg.de/_media/teaching/sommersemester_2020/siw20-benchmarking-kordt.pdf). Rule 11 makes the speed limit a positive. Rules 5, 7 and 9 together make the rented machine a varying factor, so it must be the unit of analysis.

The SPCL exemplar at MLSys pairs a bound with a realised gain. Ivanov et al.'s "Data Movement Is All You Need" won the 2021 Outstanding Paper Award by diagnosing memory-boundedness and then delivering 1.30× on a BERT encoder layer ([arXiv 2007.00072](https://arxiv.org/abs/2007.00072v3); [MLSys 2021](https://mlsys.org/Conferences/2021/BestPapers)). Pope et al. won in 2023 with a "simple analytical model" plus a new latency/MFU frontier ([MLSys 2023](https://mlsys.org/virtual/2023/poster/2463)). Pure characterisation succeeded only when it came with a field-wide survey, a checklist and a tool ([Blalock et al., MLSys 2020](https://arxiv.org/pdf/2003.03033)). That is why both reviewers tie a 9 to a mechanism.

MLSys 2027 logistics constrain what fits. Submissions open on 10 October, and **the deadline is 30 October 2026 at 12:00 PDT**. Reviews arrive on 18 January and notifications on 28 February ([mlsys.org Dates](https://mlsys.org/Conferences/current/Dates)). The 2026 rules allow 10 pages, say that appendices exist but "reviewers are not required to read these", and require double-blind anonymisation ([MLSys 2026 CFP](https://mlsys.org/Conferences/2026/CallForPapers)). The machine-level intervals, the order-free accounting and the measured bound must therefore sit in the main text. The project's public GitHub repository must be anonymised before submission.

| Round-5 weakness | What the bar requires | Concrete fix | Cost |
|---|---|---|---|
| **Clarity 2/5** (2–6 numeric ranges per sentence; private state names such as "foa", "both3p"; a 17-page scorecard) | One question per exhibit, factors and levels in one table, numbers in figures rather than prose (Rules 9 and 12; Blalock's structure) | Use 5b's thesis: "the gap to the host-bandwidth bound is dominated by admission decisions that need about W50 tokens of routing foresight; online policies, learned orders and speculative batching cannot supply it". Build three figures: limit vs measured across hosts, the order-free accounting, and W50 with the value map. Use plain state names, give every range its scope, fix the six factual slips in 5a §7a, and move the scorecard to the artifact with machine-scored clauses | $0; 3–5 days |
| **Intervals cover only within-machine variance** (±1.5% intervals vs a between-host spread 5–10× wider) | The machine is the top level of a random-effects model, and var(Ȳ) = Σσᵢ²/∏nₖ, so more problems cannot shrink the host term ([Kalibera & Jones](https://arxiv.org/pdf/2007.10899)). At least 5 top-level units (Kalibera's floor); n ≥ 10 is "a practical floor" ([arXiv 2605.00428](https://arxiv.org/pdf/2605.00428)). CPU model explains much of inter-instance spread ([Leitner & Cito](https://arxiv.org/pdf/1411.2429)) | 5 AM5 Zen 5 hosts + 5 of a second class. Hierarchical bootstrap, resampling hosts then problems ([Saravanan et al.](https://arxiv.org/pdf/2007.07797)), plus a mixed model `config × cell + class + (1|host) + (1|problem) + (1|host:config)` and a t-interval with m−1 degrees of freedom, since 5 hosts give few bootstrap resamples. Report two interval types: absolute s/token (host-dominated) and paired within-host effects (oracle gains, Shapley shares), which should be tight. Benjamini–Hochberg over the 565 clauses | ≈$12–14 shared with the next row; 2.5 days |
| **Order-dependent accounting** (bytes-first gives bytes 26–60% / overlap 7–23%; the other order gives overlap −161% to +32%) | Sequential ablation is "a greedy approach" that "may produce suboptimal results" ([Fawcett & Hoos](https://www.cs.ubc.ca/labs/beta/Projects/Ablation/papers/FawcettHoos-joh2016-ablationAnalysis.pdf)). Use Shapley values over all coalitions ([HyperSHAP, AAAI 2026](https://arxiv.org/pdf/2502.01276)) or a 2^k factorial with interaction terms ([Jain, 2^k r designs](https://classes.engineering.wustl.edu/~jain/cse567-15/ftp/k_182kr.pdf)) | Today, from existing data: report both orders and their average, a two-player Shapley that 5b computes as bytes 37–115% and overlap −77% to +27% ([Review 5b](/home/claude/moe-speed-of-light/reports/Review%205b%20professor%20%286%20October%29.md)). Then add the missing corner, "the online policy's own admissions executed as a paced prefetch", to get a clean 2×2 of {admission set: online vs MIN} × {timing: in-step vs paced}. Report main effects and the interaction with host-level intervals; φ_bytes = main + ½·interaction. Present the strong negative interaction (overlap without MIN's set hurts) as a finding | $0 now; ≈1 day of engine work plus the rentals above |
| **Mixed probed and datasheet ceilings** (GPU term at datasheet, which batch-1 reaches only 52–61% of; host term at the highest probe; [Review 5a](/home/claude/moe-speed-of-light/reports/Review%205a%20MLSys%20PC%20%286%20October%29.md)) | Ceilings must be measured, because datasheet peaks "are theoretical maximums and there may exist no code that can achieve them" ([LBNL ERT](https://crd.lbl.gov/divisions/amcr/computer-science-amcr/par/research/roofline/software/ert)). Roofs should be profiled, as in MoE-Lightning's hierarchical roofline ([arXiv 2411.11217](https://arxiv.org/html/2411.11217v1)) | Probe on every host: GPU memory rate, pinned H2D by copy engine and by zero-copy, CPU DRAM rate at the engine's thread count, and the *concurrent* CPU+DMA total (73–77 GB/s exceeded CPU-alone 62 GB/s on one host, so a joint constraint calibrated alone would be invalid; [Progress log](/home/claude/moe-speed-of-light/reports/Progress%20log.md)). Make the measured-GPU ("all") bound the headline and show datasheet as an envelope. Classify regimes under the realistic bound. Score the own engine with the audit procedure (5b's like-for-like: 22–42% vs the 13.6% median). Show that no run beats the bound; the A100 anchor's Triad floor was beaten by up to 16% ([numbers4.tex](/home/claude/moe-speed-of-light/paper/numbers4.tex)) | $0 for rescoring; probes run inside the host study |
| **Calibrated "law" not beating a naive model** (held-out: law 5.6% / 15.5% / 21.4% median / p90 / max vs naive 6.1% / 10.5% / 16.0%; [Review 5b](/home/claude/moe-speed-of-light/reports/Review%205b%20professor%20%286%20October%29.md)) | Fit on training configurations and report held-out error against the noise floor, as in performance-influence models ([Siegmund et al., FSE 2015](https://www.se.cs.uni-saarland.de/publications/docs/SGA+15.pdf)); claim only where the model wins | Rename it "calibrated time model". Report the naive baseline on every held-out set, leave-one-host-out. Measure G directly (expert-free GPU time per token; G = 4.8 ms currently exceeds a whole 3.9 ms all-in-VRAM token). Report error on the variable part. Keep its practical claim: the per-machine FETCH table gains 3.3–8.0% on host B and 21–34% on slow-link hosts. For W50, show the distinct-expert rule's leave-one-model-out win over the naive model as the template | $0; 1 day |
| **No realisable mechanism** | Turn the diagnosis into a design result (the Ivanov and Pope pattern) | Either a mechanism recovering at least a third (the long-shot rows in the plan), or a quantitative necessary condition: the value map plus the distinct-expert rule, with every published and piloted forecaster placed on it and shown outside the one-third region | $0 for the specification; $3–7 for the long shots |

The order-free fix deserves emphasis, because its data mostly exist and the result is more interesting than the current one. A measured Shapley decomposition of a performance gap across optimisation switches appears to be rare in systems papers; the canonical sources are in economics and AutoML. With an explicit interaction term, it says that "overlap" has no stand-alone value at low budgets. It pays only on top of MIN's admission set. That is a sharper version of the paper's own thesis.

## Spend the $25 on host statistics and one long shot

The ledger shows 66 rentals, 91.8 hours and about $55.9 so far. Recent single-RTX-5090 jobs cost $0.50–0.70 per hour. On 28 September there were 294 offers from $0.39/h, but only 4 Ryzen 9 9950X hosts with at least 120 GB of RAM ([innovation_directions.md](/home/claude/moe-speed-of-light/research_notes/MoE%20paper%20path%20to%20a%209/innovation_directions.md)). Host availability, not money, is the binding risk for the variance study.

Hardware levers offer nothing new: the link already runs at 57 of 64 GB/s, and host DRAM bandwidth shared by CPU experts and DMA is the binding resource. Larger models such as Qwen3-235B and DeepSeek-V3 need roughly 2-bit weights and 100–200 GB downloads per rental. So the money should buy statistics, one long-shot forecaster test, and breadth via **Qwen3-Next-80B-A3B**. That model has 512 experts, top-10 routing, a shared expert, 3.1M-parameter experts and an MTP head ([Qwen3-Next config](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)), and the engine's llama.cpp base already defines its architecture ([innovation_directions.md](/home/claude/moe-speed-of-light/research_notes/MoE%20paper%20path%20to%20a%209/innovation_directions.md)).

The table ranks the items by expected change in the reviewers' scores per dollar and hour. P(success) is my estimate: items 1–4 and 7 come from checks already run here, and the rest come from the innovation notes' priors.

| Rank | What | Cost | P(success) | What it changes in the paper |
|---|---|---|---|---|
| 1 | **Re-baseline every foresight, policy, W50 and learned-order number with κ = −∞ added to the κ set.** Scripts are already written ([checks_2026-10-06](/home/claude/moe-speed-of-light/research_notes/MoE%20paper%20path%20to%20a%209/checks_2026-10-06/)) | $0; 3–4 h | 0.95 | Removes a flaw a reviewer can find in an afternoon. The W50 law becomes 0.67 (C/k)^1.31 with a tighter interval. The "35–110%" endpoints stay. The learned-order section must say that admission alone yields 81% of its offline gain on Qwen3 at C = 16 |
| 2 | **Order-free accounting from existing job-096 data**: both orders, the two-order Shapley average, and the interaction term | $0; 3 h | 0.95 | Answers 5a W2 and 5b W4 immediately; the interaction becomes a headline finding |
| 3 | **Foresight value map**: precision × recall × horizon contours with correlated error models (recall decaying with depth, α-truncation, false positives drawn from popular experts). Place persistence, the token table, the ridge forecaster, SeqMoE's claims, EAGLE3/MTP acceptance and layer-ahead recall on it | $0; 8–10 h | 0.95 | Answers 5b Q12 and 5a Q5 directly and turns W50 into a specification: "a forecaster needs ≥0.7 precision and recall over 4–8 steps at C = E/8 (≥0.9 over 8 at E/4 on gpt-oss-120b); no published or piloted forecaster is there" |
| 4 | **Distinct-expert restatement of W50**, with leave-one-model-out errors against the naive model and the power law | $0; 4 h | 0.9 | Fixes 5b W9 (fragile fit, untested on held-out models). Connects to Albers' strong lookahead and to 2608.07911's union-to-capacity contract |
| 5 | **One bound definition and honest modelling**: measured-GPU bound as the headline, own engine scored by the audit procedure, Eq. 1 renamed, naive baseline on every held-out set, G measured directly | $0; 6–8 h | 0.9 | Fixes 5a W5/W6 and 5b W5/W6. Soundness 3 → 4 is plausible |
| 6 | **Ten-host variance study plus the missing factorial corner plus ceiling probes plus an always-admit engine arm.** gpt-oss-120b at the 2 host-bound cells; 4 corners × 2 seeded orders; 12 × 256 teacher-forced steps; ~1.5 h per rental | ≈$13–15; ~20 GPU-h; 2.5 days | 0.7 (host availability) | Host-level intervals and variance components, a clean 2×2 Shapley with intervals, measured ceilings, and whether always-admit wins in seconds. Rigour 4 → 5 |
| 7 | **Rewrite** around one thesis and three exhibits; plain names; scoped ranges; scorecard to artifact; anonymised repository | $0; 3–5 days | 0.85 | Clarity 2 → 3–4. 5a: a rewrite plus order-free accounting plus scoped ranges "would make it a solid weak accept" |
| 8 | **Cache-only shadow run-ahead, Stage A (offline go/no-go).** On Qwen3 first, replay the online cache state at C = E/8 and E/4, run positions t+1..t+4 with resident experts only, and record router top-k. Pre-registered go rule: precision@k ≥ 0.7 averaged over h ≤ 4 and pilot read-gap closure ≥ 0.33 | $3–5; 1.5 days. Stage B on go: ≈$5 and 2–3 days | Go 0.25 (Qwen3) / 0.15 (gpt-oss-120b); in-engine ≥1/3 by 30 Oct ≈ 0.10 | The only realistic path to a 9 (significance 3 → 4–5). A clean negative ("even the model run ahead on its own cache cannot forecast") still becomes a value-map point |
| 9 | **SeqMoE-style sequence forecaster on the paper's own traces** (GRU or Mamba over 36×128 and 48×128 vectors, scheduled sampling, evaluated on the value map) | $0–2 (CPU or 1–2 GPU-h); 1.5–2 days | 0.15 that recall@(k+3) at h = 1 rises from the linear 0.52/0.63 toward 0.8+ and closes ≥ 1/3 at C = E/8 | Either opens a second mechanism path, or makes "no forecaster reaches the region" a measured claim rather than a literature claim. It also tests SeqMoE's unreproduced 0.91 |
| 10 | **Qwen3-Next-80B-A3B as a third family**: own-text trace, W50 point and distinct-expert check, bound, and 2 engine cells; MTP acceptance as a value-map point | $4–6; ~4 GPU-h; 2 days | 0.55 (the 4.5k-line patch may break on the shared expert or gated-DeltaNet layers) | Breadth (5b item 7) and a genuinely held-out test of the W50 and distinct-expert rules at E = 512 |
| 11 | **llama.cpp PR #29887 as a race arm** inside the rentals | $1–2; 0.5 day | 0.6 | Pre-empts "upstream already has this". Its policy is always-admit LRU, the natural in-engine baseline after item 1 |

The budget works out to: item 6 ≈ $14, item 8 ≈ $4, item 10 ≈ $5, and $2 for items 9 and 11, about $25 in total. The schedule is:

- **6–9 October:** the $0 items (1–5) run first, because they set the go thresholds for item 8.
- **10–16 October:** rentals for items 6 and 8 run in parallel. The go/no-go for item 8 falls around 16 October.
- **17–22 October:** item 10, plus Stage B only on a go.
- **23–30 October:** the rewrite closes.

On this plan, the expected outcome is a 6.5–7 with near certainty, and an 8–9 only if Stage A says go and Stage B lands.

Several directions are unlikely to work, and the paper should not spend time on them.

- **A realisable mechanism recovering a third of the oracle gain in the engine by 30 October.** About 10%.
- **External draft routing.** gpt-oss-20b as a drafter for 120b takes about 13 GB of VRAM, most of a 25% cache budget, and needs a learned map across 24 vs 36 layers. A Qwen3-0.6B or 1.7B drafter has foreign hidden states, and its token-level information is worth less than nothing even with perfect tokens. Its probability is about 0.08.
- **MTP-head foresight.** One token of horizon, and absent from both engine models.
- **Persistence, transition tables, linear full-stack forecasters and further learned admission orders.** All sit at or below decayed frequency beyond one step.
- **Read-ME-style decoupled routers.** They need retraining on about 1B tokens and still give zero cross-token horizon.
- **Cache-friendly rerouting.** It changes model outputs, so it leaves the exact-routing bound's scope.
- **Hardware levers** (full PCIe 5.0, pinned memory, GPU-initiated copies) beyond what CUDA graphs already delivered.
- **DeepSeek-V3- or Qwen3-235B-class models** on this budget.
- **Claim language.** The paper should stop calling W50 or Eq. 1 a "law", stop claiming the qualitative online-vs-MIN gap or learned admission as firsts, and stop presenting the learned order as foresight when most of its Qwen3 gain is admission.

## Conclusion

The paper's most defensible identity has shifted. It is no longer "a speed limit plus a cache". It is a measured specification of what routing foresight must look like before it pays. The in-engine factorial shows that foresight pays only when spent on MIN's admission set. The value map shows how good a forecast must be. The distinct-expert rule states the horizon in units a forecaster controls: about 0.6·C distinct experts ahead per layer. The always-admit check is a warning about the paper's own baselines: a one-line policy change recovered most of what a learned model was credited with on Qwen3, and the same kind of check, run by a reviewer, would have cost credibility.

Seen this way, the reviewers' demand for a mechanism has two acceptable answers. A shadow run-ahead or a reproduced SeqMoE that enters the one-third region would be a systems result. A rigorous demonstration that every realisable source sits outside it would be a measurement result. The second is nearly in hand for $0, and it is likely worth a 7. The first is a 10% chance at a 9, and it is cheap enough to try in parallel.

The deeper lesson is about where foresight lives in an MoE. A linear full-stack model barely beats decayed frequency, so any predictable structure in future routing is either strongly nonlinear in the routing history or absent from it. If it is absent, only the model's own computation, run ahead on what is already resident, can supply it.
