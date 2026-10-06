# Innovation directions: which new result most raises the impact of "Where the Seconds Go" on about $25 of RTX 5090 time (6 Oct 2026)

Scope: rank 4–6 candidate new results by (expected reviewer impact × probability of success) / cost. Each gets an
experiment that fits in about $25 and 2–3 days. The notes also answer the five key questions: draft routing, simpler
predictors, community numbers, larger models, and hardware levers.

A $0 pilot was run on the repository's own traces to turn literature priors into numbers for these two models. It is
reproducible from files in this folder:
- `pilot_forecast_sim.py`: the forecast-driven version of `scripts/foresight.py::_pol`.
- `pilot_forecast_baselines.py`.
- `pilot_forecast_gpt-oss-120b.json` and `pilot_forecast_qwen3-30b-a3b.json`.

Pilot inputs: the T-arm traces (dataset text, token ids known), about 21k decode steps per model:
- `results/035_trace_rest@40gb/{gpt-oss-120b,qwen3-30b-a3b}_tok.npz` on the `gpu` branch;
- token ids from `data/tok_*.jsonl`.

The pilot counts reads (host bytes in expert units), not seconds. Pilot numbers are labelled "pilot" below.

---

## Q0. Ranked candidates: which concrete new result, with what design, at what odds?

### Takeaway
The "realisable foresight recovering at least 1/3 of the oracle gain" result that both reviewers tie to a 9 is
**unlikely within $25 and 2–3 days: about 10% for an in-engine demonstration.**

The pilot shows why:
- Only forecasts that reproduce the target's own routing with precision and recall of at least ~0.7, 2–4 decode steps
  ahead, reach 1/3 of the read gap, and only at the lowest budget (C = E/8).
- At C = E/4, even 0.9-accurate 4-step forecasts close 0.23 (gpt-oss-120b) to 0.36 (Qwen3) of the gap.
- Token-level drafting plus a token→expert table is worse than a re-tuned online policy even with *perfect* future
  tokens.

The best expected value per dollar is a bundle:
1. **(#1, $0)** a precision–recall–horizon "foresight value map" that answers reviewer 5b's Q12 verbatim and turns
   W50 into a forecaster specification;
2. **(#2, ≈$12)** a host-as-unit variance study combined with the missing factorial corner for an order-free (Shapley)
   accounting;
3. **(#3, ≈$4)** a go/no-go offline test of the only forecaster with a plausible path to the bar: a "cache-only shadow"
   run-ahead of the target on the experts the GPU already holds, during the GPU's idle time in host-bound steps.

A cheap third family (Qwen3-Next-80B-A3B, 512 experts) is the fourth item. External draft models and MTP heads are
likely to fail and should be reported as negative results or used as points on the value map, not as the headline.

### Cited Findings

**Pilot (this folder; repository traces; $0)**

*Gap closed by forecasts, normalised to the best no-foresight single-read policy (always-admit, decayed-LFU victim):*
- **gpt-oss-120b (E=128, k=4)**, share of the gap closed at C=16:
  - exact foresight: 0.15 (W=1), 0.29 (W=2), 0.48 (W=4), 0.74 (W=8);
  - noisy forecasts at W=4 (precision = recall = r): 0.42 (r=0.9), 0.31 (r=0.7), 0.19 (r=0.5).
  - At C=32, exact W=4 gives 0.26 and r=0.7 at W=4 gives 0.16. — [pilot_forecast_gpt-oss-120b.json](pilot_forecast_gpt-oss-120b.json); [pilot_forecast_sim.py](pilot_forecast_sim.py)
- **Qwen3-30B-A3B (E=128, k=8)**, share of the gap closed at C=16:
  - exact: 0.32 (W=1), 0.55 (W=2), 0.83 (W=4);
  - noisy at W=4: 0.73 (r=0.9), 0.54 (r=0.7), 0.35 (r=0.5).
  - At C=32: exact W=4 0.42, r=0.7 W=4 0.24, r=0.7 W=8 0.42. — [pilot_forecast_qwen3-30b-a3b.json](pilot_forecast_qwen3-30b-a3b.json)
- **Token-ID table forecaster.** Inputs: the true future tokens (a perfect drafter); the top-k experts for that token id
  per layer, fitted on the other half of the conversations.
  - Mean precision@k: 0.348 (gpt-oss-120b) and 0.354 (Qwen3). It is about 0.55 in the first layers.
  - Against the best no-foresight policy it closes −0.02 to −0.10 (gpt-oss-120b, C=16) and −0.10 to −0.27 (Qwen3,
    C=16) of the gap. **It is worse than having no forecast.**
  - With i.i.d. 75% per-token draft acceptance it is slightly worse again. — [pilot_forecast_*.json](pilot_forecast_qwen3-30b-a3b.json)
- **"Persistence" forecast (next steps = this step's experts).** In this policy it is identical to always-admit with
  decayed-LFU eviction: same reads to the 0.01. It is a tuning of admission, not foresight. — [pilot_forecast_baselines.py](pilot_forecast_baselines.py)

*Baseline check on the paper's own S-arm traces with `scripts/foresight.py::_pol`, no foresight:*
- Always-admit (κ=−∞) reads fewer experts than the better of κ=0 and the model's κ:
  - gpt-oss-120b: 3.3% fewer at C=16, 0.3% at C=32, 0.0% at C=64;
  - Qwen3-30B-A3B: 6.7% fewer at C=16, 5.0% at C=32, 0.1% at C=64.
- Qwen3 C=16 raw reads per token:
  - κ=0: 143.53. This equals the "online" value in `prereg/foresight/foresight_S.json`.
  - κ=2: 150.19.
  - always-admit: 133.97.
- Run in this session; the command is in the report. — [scripts/foresight.py](../../scripts/foresight.py); [prereg/foresight/foresight_S.json](../../prereg/foresight/foresight_S.json)

**Literature priors**

*Draft-based expert prediction:*
- SP-MoE feeds each draft layer's attention output into the target's gate, with no projection, and gets about 88–89%
  top-1 expert accuracy. Its draft/target pairs are structurally twinned:
  - Mistral-7B → Mixtral, cosine similarity of attention outputs 59.8%;
  - Phi-mini-MoE → Phi-3.5-MoE, 56.6%;
  - DeepSeek-Lite-AWQ → DeepSeek-Lite (its own quantized copy), 94.6%.
  - Acceptance is 97–98% (RTX 4090, PCIe 4.0). — [SP-MoE arXiv 2510.10302v2](https://arxiv.org/html/2510.10302v2)
- **MoE-SpAc** (arXiv 2603.09983, Mar 2026) "repurpose[s] Speculative Decoding … as an informative lookahead sensor for
  memory management". It unifies prefetching **and eviction** in one utility space.
  - Setup: Qwen3-30B-A3B target, Qwen3-4B-FP8 draft, γ=8; RTX 4090 on PCIe 4.0; 17% expert cache ratio; batch 1.
  - "42% improvement in TPS over the SOTA SD-based baseline" (llama.cpp with SD), and hot/cold prediction accuracy
    "around 0.85" at layer 47.
  - The draft costs about 8% of VRAM.
  - It verifies drafts (batched verification).
  - — [MoE-SpAc on HF Papers](https://arxiv.org/html/2603.09983) (read via hf://papers/2603.09983/paper.md)
- DraftExpert (27 Jul 2026): a self-speculative drafter (shared + top-1 + one trained draft expert per layer) on
  DeepSeek-V2-Lite and Moonlight. Acceptance 84–87%, router-agreement prefetch hit 86–88%, 1.45× average decode speed-up
  (RTX 4090 CPU→GPU and phone NPU). — [DraftExpert alphaXiv 2607.24434](https://www.alphaxiv.org/abs/2607.24434.md)

*Speculation at batch 1:*
- Speculation slows batch-1 MoE decode by up to 1.5×:
  - Mixtral on math: 25% slower at K=1, 54% at K=3;
  - verification moves up to 3× more data at K=7;
  - n-gram and EAGLE drafts; RTX 6000 Ada.
  - — [Utility-Driven SD for MoE (Cascade), arXiv 2506.20675](https://www.alphaxiv.org/abs/2506.20675.md)
- Qwen3-Coder-30B-A3B with an EAGLE3 draft at B=1 (A100): average accepted length saturates around 2.1 tokens; the
  oracle limits drafts to about 2.8 tokens. — [The Limits of Speculation, arXiv 2609.22156](https://arxiv.org/pdf/2609.22156)

*Measured costs in the engine:*
- From the repository notes: prefetch of the next layer gave +14% / +16% / +10% on gpt-oss-120b at C = 14 / 32 / 56.
  The GPU part of a step is about 36 × (80 µs attention + 48 µs hits) ≈ 4.6 ms, against 215 µs per CPU miss on
  host 1. — [prefetch_lookahead.md](../MoE%20offload%20system%20race%20plan/prefetch_lookahead.md)
- Rentals in `gpu/vast_ledger.json`:
  - 66 rentals, 91.8 h, about $55.9 of compute (computed this session);
  - recent jobs at $0.50–0.70/h;
  - a factorial launch (job 096) took 4.78 h ($2.4–3.0). — [gpu/vast_ledger.json](../../gpu/vast_ledger.json)
- Vast snapshot, 28 Sep 2026:
  - single-RTX-5090 offers from $0.39/h, median $0.65/h (294 offers);
  - with ≥120 GB host RAM: from $0.43/h, median $0.73/h (69 offers);
  - only 4 Ryzen 9 9950X hosts and 7 EPYC 7C13 hosts with ≥120 GB at that moment.
  - — [vast_snapshot_2026-09-28_blackwell.json](../MoE%20offload%20system%20race%20plan/vast_snapshot_2026-09-28_blackwell.json)

### Inferences

**Ranking.** Impact is the expected movement of the reviewers' "significance" or "rigor" scores (0–5 scale). Cost is
GPU dollars plus working days.

| # | New result | Impact if it works | P(success) | Cost | Value per cost |
|---|---|---|---|---|---|
| 1 | Foresight value map: share of oracle gain vs (precision, recall, horizon), with published and realisable forecasters placed on it; baseline fixed to include always-admit | 2 (turns the diagnosis into a design spec; answers 5b Q12) | 0.95 | $0, 1 day | Highest |
| 2 | Host-as-unit variance (≥5 rentals × 2 CPU classes) plus the missing factorial corner, giving a 2-player Shapley accounting with host-level intervals | 2.5 (two of the reviewers' 9-list items: statistics, order-free accounting) | 0.7 (host availability is the risk) | ≈$12–14, 2.5 days | High |
| 3 | Cache-only shadow run-ahead, Stage A: offline go/no-go on routing accuracy at h = 1–4 | 4.5 if it then works in the engine; 1.5 as a clean negative ("even the model run ahead on its own cache cannot forecast") | Stage A go ≈0.25 (Qwen3, C=E/8), ≈0.15 (gpt-oss-120b); in-engine ≥1/3 within the window ≈0.10 | ≈$3–5 for Stage A, 1.5 days; Stage B ≈$5 plus 2–3 days | Medium (a lottery ticket, the only path to a 9) |
| 4 | Third family at E=512: Qwen3-Next-80B-A3B trace, W50 point, bound and 2 engine cells (plus MTP acceptance as a value-map point) | 1.5–2 (reviewer 5b item 7, "Broaden") | 0.55 | ≈$4–6, 2 days | Medium |
| 5 | External draft routing, foresight only (Qwen3-0.6B/1.7B or the gpt-oss-120b EAGLE3 head → target routers via a learned map) | 4 if it works | ≤0.08 | ≈$4, 2–3 days | Low |
| 6 | MTP-head foresight; hardware-side tweaks | ≤1 | ≤0.1 | ≈$3 | Lowest |

**Suggested $25 split:** #2 ≈ $13, #3A ≈ $4, #4 ≈ $5, reserve ≈ $3. #1 costs nothing and should be done first,
because it sets the go/no-go thresholds for #3.

**Design #1: foresight value map ($0, 1 day)**
- Extend `scripts/foresight.py` with `_pol_fc` (as in `pilot_forecast_sim.py`). Fix two things the pilot exposed:
  - **(a)** use always-admit (κ=−∞) both as the no-foresight baseline and as the beyond-window rule, so foresight is
    never charged for bad admission tuning;
  - **(b)** use correlated error models, not only i.i.d. noise:
    - per-layer recall decaying with depth;
    - per-step "draft broke" truncation with acceptance α;
    - false positives drawn from the layer's popular experts.
- Report, for each model and budget, the (precision, recall, W) contour where 1/3 and 1/2 of the gap closes. Convert to
  seconds with the existing law/timeline constants.
- Place these on it:
  - persistence;
  - layer-ahead router (84–97% within a step; it gives no next-step foresight);
  - token table (≈0.35);
  - SeqMoE's next-step recall claims;
  - EAGLE3/MTP acceptance turned into a horizon, e.g. EAGLE3 on gpt-oss-120b accepts 2.0–2.75 tokens per 3-token
    draft (Q1).
- Deliverable: one figure and one sentence. For example: "a forecaster must supply ≥0.7 precision and recall over the
  next 4 steps to recover a third of the gap at C=E/8, and ≥0.9 over 8 steps at C=E/4; no published forecaster
  does".
- **Also re-check the W50 law and the "online reads 35–110% more than MIN" range against the always-admit baseline**
  (pilot: −3.3% to −6.7% reads at the low budgets). Do it before a reviewer does.

**Design #2: variance plus order-free accounting (≈$12–14, 2.5 days, 10 rentals)**
- **Engine work (0.5–1 day, CPU-tested first): the missing corner "overlap without byte savings".** The online policy's
  own admissions are executed as a paced prefetch d steps early.
  - Record the online policy's admission log in a first pass (the simulator matches the engine within 0.001 hit rate,
    per prefetch_lookahead.md).
  - Feed the log through the existing lookahead-file oracle-lead machinery.
  - This gives four corners {base, fetch, online-lead, both3p}, so a 2-player Shapley value (bytes, overlap) can be
    computed per host with no order choice.
- **Protocol per rental (~1.5 h at $0.6–0.7/h):**
  - gpt-oss-120b only;
  - the 2 host-bound cells;
  - 4 corners × 2 seeded orders;
  - 12 × 256 teacher-forced steps;
  - probe run first.
- **Hosts:** 5 AM5 Zen 5 desktops (9950X/9950X3D/9900X, CPU model as a covariate) and 5 of a second class (EPYC 7C13
  or 7950X).
- **Analysis:** a hierarchical bootstrap or mixed model with host as the unit, reporting variance components (host /
  launch / problem). Pre-register the band for the between-host SD; the paper's "19–22%" is a prior.
- **Risk:** at most 4 same-model 9950X hosts with ≥120 GB were listed at once on 28 Sep. The study may need to run
  over 2–3 days, or pool a "desktop DDR5 AM5" class.

**Design #3A: cache-only shadow run-ahead, offline go/no-go (≈$3–5, 1.5 days)**
- **Idea.** In a host-bound step the GPU sits idle while CPU experts run. With C=14 on host 1 there are about 66
  misses × 215 µs ≈ 14 ms of CPU time against about 4.6 ms of GPU work (repository constants). Spend that idle time
  running the *target itself* 1–4 tokens ahead with **only GPU-resident experts**:
  - the router logits over all experts give the forecast;
  - the token's hidden state is computed with resident experts only, top-k renormalised.
  - It adds no host bytes and no extra weights, and output is unaffected because the shadow is never emitted.
- **Measurement.** Qwen3-30B-A3B first: k=8, the most forgiving W-curve. Use the T-arm corpus. Run HF transformers on
  one rented 48–80 GB GPU, or llama.cpp with a mask on the 5090 host.
  - For each step t, replay the online cache state from the simulator at C=E/8 and E/4.
  - Run positions t+1..t+4 with (i) true tokens, (ii) shadow-greedy tokens, (iii) a small draft's tokens.
  - Record the shadow routers' top-k per layer.
- **Output.** Precision and recall per horizon and per layer depth. Feed them into `_pol_fc` to get the fraction of the
  gap closed.
- **Go rule (pre-register):** precision@k ≥ 0.7 averaged over h ≤ 4 at C=E/8 with (ii) or (iii) tokens, and pilot
  read-gap closure ≥ 0.33.
- **Stage B, only on go:** a second sequence id in the same `llama_context` for the shadow positions, an expert mask
  from the slot map, KV clean-up, scheduling in the CPU-miss gaps, and admission decisions from the forecast.
- **Literature for:**
  - routing from the target's own approximately computed states predicts well (SP-MoE 88–89% top-1, 94.6% cosine for
    the quantized self-draft);
  - substituting missing experts with resident "buddies" costs negligible accuracy when gated ([BuddyMoE arXiv 2511.10054](https://arxiv.org/html/2511.10054v1)).
- **Against:**
  - at C=E/8 about 46% of gpt-oss-120b's selected experts are missing (hit rate 0.54), and errors compound over 36–48
    layers;
  - k=4 is less redundant than k=8;
  - token agreement over 4 steps decays roughly as α⁴ (α ≈ 0.55–0.75 for MTP/EAGLE-class drafts; Q1);
  - no source measures routing accuracy under resident-only execution.
- **Novelty check:** MoE-SpAc already uses drafts plus **verification** for eviction and prefetch. The distinct claim
  here would be **foresight without verification reads** (a GPU-only shadow) scored against MIN in seconds.

**Design #4: Qwen3-Next-80B-A3B as a third family (≈$4–6, 2 days)**
- The engine's llama.cpp base (`4da6337`, 27 Sep 2026) defines `LLM_ARCH_QWEN3NEXT`, `QWEN35MOE`, `GLM4_MOE`,
  `DEEPSEEK2`, `LLAMA4`, `KIMI_K` and `MINIMAX_M2/M3`, plus NEXTN (MTP) tensors. Checked in `/home/claude/scratch/lc-ec`.
- One rental: a Q4 GGUF (~45–50 GB; size unverified), routing trace via the engine's dump on own text, W-curves and
  a W50 point, then base vs fetch-oracle at 2 cells.
- E=512, top-10, a shared expert and tiny 3.1M-parameter experts test the law at an extreme C/k. They also test
  per-expert copy overhead (≈1.7 MB per expert at about 4.5 bits; an estimate).
- Add-on: MTP acceptance at batch 1 via llama.cpp's Qwen3-Next MTP, to place MTP on the value map.
- **Risk:** the 4.5k-line patch's assumptions (no shared expert, attention type) may break on gated-DeltaNet layers.

### Gaps
- The pilot is read-count only, on T-arm text (about 21k steps), with the window policy of `foresight.py`. In the pilot
  the beyond-window rule used the model's κ, so negative values at C=E/2 are partly a policy artefact. Seconds and the
  S-arm are not covered.
- No source measures routing accuracy of a cache-only (resident-experts-only) forward pass; that is the unknown #3A
  measures.
- MoE-SpAc's per-benchmark tables, and whether its llama.cpp-SD baseline used `--n-cpu-moe`, were not extracted (the
  table images were not in the text).

---

## Q1. Draft-model routing as foresight: does running the target's routers on draft tokens predict future experts? What do speculative-MoE papers report on expert overlap and acceptance at batch 1?

### Takeaway
Target routers predict well only when fed the *target's own* (or a structurally twinned model's) hidden states for
the drafted tokens:
- SP-MoE about 88% top-1;
- MoE-SpeQ 90.9% top-4 set;
- DraftExpert 86–88% prefetch hits;
- MoE-SpAc hot/cold about 0.85.

Every published system obtains those states through **verification**, and verification's expert union is exactly
what makes batch-1 speculation lose under offload. Token identity alone, even with perfect future tokens, predicts
only about 35% of experts (pilot). Acceptance at batch 1 is short: about 2–2.75 accepted tokens per 3-token EAGLE3
draft on gpt-oss-120b, and about 2.1 average accepted on Qwen3-Coder-30B-A3B with EAGLE3. No batch-1 acceptance
numbers were found for gpt-oss-20b→120b or for Qwen3-0.6B/1.7B→30B-A3B.

### Cited Findings

**Prediction from draft states**
- SP-MoE (above): about 88–89% top-1. It works because the drafts share hidden size and architecture with the target
  (Mistral-7B/Mixtral at 4096), or are the target quantized. — [SP-MoE arXiv 2510.10302v2](https://arxiv.org/html/2510.10302v2)
- MoE-SpeQ: an INT4 draft predicts target experts with 90.9% "accurate" top-4 (44.1% exact, 46.8% correct set in a
  different order). It notes verification must "load and compute the *union* of all experts activated across all k
  tokens". — [MoE-SpeQ arXiv 2511.14102v1](https://arxiv.org/html/2511.14102v1) (as recorded in the 27 Sep notes)
- MoE-SpAc: drafts plus verification as a "lookahead sensor"; Qwen3-30B-A3B with a Qwen3-4B-FP8 draft, γ=8. Hot/cold
  accuracy about 0.85 at layer 47. +42% TPS over llama.cpp with SD; the draft costs about 8% of VRAM. — [MoE-SpAc](https://arxiv.org/html/2603.09983)
- DraftExpert: acceptance 84–87%. Unique routed experts per block: about 88–90 with its drafter vs 84–86 for a top-1
  path and 244–248 for top-3. — [DraftExpert](https://www.alphaxiv.org/abs/2607.24434.md)

**The cost of verification at batch 1**
- Cascade: up to 1.5× slowdown at batch 1 from the expert union (Mixtral math −25% at K=1, −54% at K=3).
  Utility-driven K limits the worst case to −5% and gains 7–14% over static K. — [Cascade](https://www.alphaxiv.org/abs/2506.20675.md)
- MoESD: speculative decoding is ineffective at batch 1 for MoE because of extra expert loads, and becomes effective
  at moderate batch "when … all experts are already activated". Up to 2.29× on Qwen2-57B-A14B. — [MoESD arXiv 2505.19645v4](https://arxiv.org/html/2505.19645v4)
- Limits of Speculation: Qwen3-Coder-30B-A3B with EAGLE3 at B=1 on an A100. Per-expert verification cost jumps about
  3.2× between 4 and 5 positions. Accepted length saturates around 2.1. — [arXiv 2609.22156](https://arxiv.org/pdf/2609.22156)
- Community, RTX 3090, Qwen3.6-35B-A3B at Q4_K_XL, batch 1, all on the GPU:
  - baseline 135.7–139.9 tok/s;
  - Qwen3.5-0.8B draft: 65.0 tok/s (−54%);
  - ngram-cache: 119.1 tok/s;
  - DFlash: −44.6%.
  - "Each drafted token pulls a fresh expert slice … the verification pass pays for the union."
  - — [hackmd thc1006](https://hackmd.io/@thc1006/SJly6IE6Wx)
- SpecMoEOff: draft-model speculation raises per-expert work to hide offloading latency on Mixtral-8x7B (A30 and
  4090D). Up to 2.5× over MoE-Lightning (a throughput setting). — [SpecMoEOff arXiv 2508.21706](https://arxiv.org/abs/2508.21706)

**Acceptance for these model pairs**
- gpt-oss-120b drafts:
  - NVIDIA's EAGLE3 head (0.8B parameters; TensorRT-LLM) accepts 2.10 (writing), 2.57 (reasoning), 2.75 (math) and
    2.67 (coding) tokens on MT-Bench with draft length 3. — [nvidia/gpt-oss-120b-Eagle3-short-context](https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-short-context)
  - Snowflake's LSTM speculator (1.76B parameters, 3 tokens): 44–50% acceptance; 1.7–1.8× on gpt-oss-120b at
    concurrency 1 with TP=4 on H200 (not offloaded). — [Snowflake Arctic Inference blog, 25 Aug 2025](https://www.snowflake.com/en/engineering-blog/faster-gpt-oss-reasoning-arctic-inference)
- llama.cpp added EAGLE3 support in release b9606 (PR #18039), about 3 months ago. Whether the gpt-oss-120b EAGLE3
  heads convert and run is unverified. — [newreleases b9606](https://newreleases.io/project/github/ggml-org/llama.cpp/release/b9606)
- In the llama.cpp expert-cache RFC #24528, the cache "composes with an external drafter … only when the drafter is
  NOT on a GPU"; it disengages silently otherwise. — [llama.cpp #24528](https://github.com/ggml-org/llama.cpp/discussions/24528) (from the 27 Sep notes)
- Pilot: a token-ID→expert table given the true future tokens reaches precision@k 0.348 (gpt-oss-120b) and 0.354
  (Qwen3). That is about 0.55 in the first layers and 0.33–0.56 in the last three. As a forecast it does worse than
  always-admit. — [pilot_forecast_*.json](pilot_forecast_gpt-oss-120b.json)
- OpenMoE reports "context-independent specialization": routing follows token id more than semantics. Measured on
  OpenMoE, not on gpt-oss or Qwen3. — [OpenMoE arXiv 2402.01739](https://arxiv.org/html/2402.01739v1)

### Inferences
- **"Run the target's routers on draft tokens" needs the target's hidden states for those tokens.** Getting them
  costs either:
  - a verification pass, which reads the union of experts (the paper already shows this is a loss once rejected
    drafts are charged); or
  - an approximation: a twinned draft whose states fit the target's gates (SP-MoE), or a resident-only shadow pass
    (Design #3).
- For gpt-oss-20b→120b: the two share hidden size 2880 and the tokenizer, but differ in depth (24 vs 36 layers) and
  were trained separately. Direct gate reuse would need a learned map; SP-MoE's cosine similarity was only 0.57–0.60
  even for upcycled or pruned pairs. gpt-oss-20b also takes about 13 GB of VRAM, which is most of the 25% cache
  budget. Verdict: **will fail at equal VRAM.**
- For Qwen3-0.6B/1.7B→30B-A3B: VRAM is cheap (about 0.5–2 GB), but the hidden states are foreign (1024 or 2048 wide,
  28 layers vs 48), so a learned map is needed. The pilot's ceiling for token-identity information is negative.
  Verdict: **likely to fail; at most a value-map point.**
- **Novelty.** "Drafts as a lookahead sensor for eviction and prefetch" is taken (MoE-SpAc, Mar 2026; the earlier
  27 Sep notes missed it). What is open is foresight from drafts **without verification reads**, scored against a
  time-domain MIN bound. The paper should cite MoE-SpAc and contrast its +42% (against llama.cpp with SD) with the
  paper's finding that batched verification loses once rejected drafts are charged.

### Gaps
- No batch-1 acceptance numbers were found for gpt-oss-20b→gpt-oss-120b or for Qwen3-0.6B/1.7B→Qwen3-30B-A3B
  (searched; only the generic Qwen3.5-0.8B→Qwen3.6-35B-A3B community loss).
- MoE-SpeQ's per-horizon (k tokens ahead) accuracy breakdown was not re-extracted.
- "Utility-driven" EAGLE numbers on Mixtral were only summarised (drafting overhead about 5% per unit of K).

---

## Q2. Simpler multi-token predictors: is token t+1's routing predictable from token t's states, or from MTP heads? Published accuracies

### Takeaway
Next-*layer* prediction within a step is accurate (84–97% top-4/top-8 in the repository; up to about 97% in Fate) but
gives no next-*step* foresight. Next-*step* signals are weak:
- persistence of the expert set is 0.31 for gpt-oss-120b and 0.47–0.50 for the others (repository);
- token identity alone is about 0.35 (pilot);
- learned sequence predictors claim much more (SeqMoE: top-11 recall above 90% for Qwen3 at k=8; top-7 above 90% for
  gpt-oss-120b at k=4). That implies precision of only about 0.51–0.65 at h=1, which the value map puts at ≤0.15 of
  the gap.

MTP heads give about one extra token at 55–90% acceptance (DeepSeek-V3 85–90%; Qwen3.6-35B-A3B in llama.cpp 55%).
That is horizon ≈1, worth ≤0.15–0.3 of the gap even if routing were exact.

### Cited Findings
- Repository, teacher-forced, job 063: share of layer l+1's experts among the top-n predictions from layer l's MoE
  input:
  - gpt-oss-120b: 84.2% (top-4) and 96.7% (top-8);
  - gpt-oss-20b: 87.1% and 97.7%;
  - two layers ahead (120b): 77.1% and 92.5%.
  - — [prefetch_lookahead.md](../MoE%20offload%20system%20race%20plan/prefetch_lookahead.md)
- Repository provenance study: temporal reuse |S_t ∩ S_{t−1}|/k on the D arm is 0.484 (OLMoE), 0.499 (gpt-oss-20b),
  0.470 (Qwen3-30B-A3B) and 0.313 (gpt-oss-120b). — [prereg/provenance/provenance.md](../../prereg/provenance/provenance.md)
- SeqMoE (11 Sep 2026): a Mamba2 sequence model of whole-stack activations predicts the next step(s). Recall above 90%
  at top-11 (Qwen3-30B-A3B, k=8) and top-7 (GPT-OSS-120B, k=4); 96.97% hit rate at 45% residency vs 88.50%. The
  arXiv PDF was unreadable through the fetch tool this session; the numbers are from the 27 Sep notes. — [SeqMoE arXiv 2609.12978](https://arxiv.org/pdf/2609.12978)
- ST-MoE: temporal and cross-layer correlation tables, about 85% prediction accuracy (hardware accelerator, no Belady
  comparison). — 27 Sep notes, [arXiv 2606.15453](https://arxiv.org/html/2606.15453)
- DoMoE (IJCAI 2026): multi-layer-ahead semantic prediction; "extending the prediction horizon reduces accuracy". Hit
  ratios 64–72% (Qwen3-30B-A3B among its models). — [DoMoE IJCAI 2026](https://www.ijcai.org/proceedings/2026/0657.pdf)
- ExpertFlow: multi-layer-ahead token-aware prediction, +21.79% on average over pre-gating (DeepSeek-V2-Lite,
  Qwen1.5/2 MoE). — [ExpertFlow arXiv 2510.26730](https://arxiv.org/html/2510.26730v1)
- DeepSeek-V3 MTP: "The acceptance rate of the second token prediction ranges from 85% to 90%, enabling … 1.8 times
  Tokens Per Second". — [aman.ai DeepSeek-V3 primer](https://aman.ai/primers/ai/deepseek-v3/)
- Qwen3.6-35B-A3B MTP in llama.cpp (all on the GPU, RTX PRO 6000, 19 May 2026): 55% acceptance at draft-n-max 3;
  193.36 → 225.48 tok/s (1.17×). The dense Qwen3.6-27B: 65%, 1.73×. — [Jarvislabs blog](https://jarvislabs.ai/blog/qwen36-mtp-llamacpp-rtxpro6000)
- llama.cpp MTP for Qwen3-Next landed in release b10238 (PR #25589), about 2 months ago. A later "MTP clean-up" PR
  reports mixed throughput (one user −15–20%). — [newreleases b10238](https://newreleases.io/project/github/ggml-org/llama.cpp/release/b10238); [llama.cpp PR #23269](https://github.com/ggml-org/llama.cpp/pull/23269)
- GLM-4.5-Air ships one MTP layer (`num_nextn_predict_layers: 1`). — [zai-org/GLM-4.5-Air config](https://huggingface.co/zai-org/GLM-4.5-Air)
- Pilot: on these two models, exact foresight of the next 1 step closes 0.15 (gpt-oss-120b) / 0.32 (Qwen3) of the gap
  at C=E/8 and 0.06 / 0.06 at C=E/4 (best-baseline normalisation). — [pilot_forecast_*.json](pilot_forecast_qwen3-30b-a3b.json)

### Inferences
- Applying layer l's router to token t's residual to predict token t+1 at layer l is the persistence signal (0.31–0.50
  overlap). The decayed-frequency cache already exploits it. **Likely to fail as a foresight source.**
- MTP heads are capped by horizon. Even at 100% routing accuracy, a 1-step horizon closes ≤0.15–0.32 of the gap at
  C=E/8 and about 0.06 at C=E/4. With 55–90% acceptance and imperfect routing for the MTP token, ≥1/3 is very
  unlikely. MTP is also absent from gpt-oss and Qwen3-30B-A3B, so it needs a new model (Q4).
- SeqMoE-class learned forecasters are the strongest published next-step predictors. Their implied h=1 precision
  (about 0.5–0.65) still sits low on the value map, which is consistent with the paper's finding that a learned order
  recovers 7–13%.

### Gaps
- No published accuracy for routing of the *MTP-predicted* token, or for MTP under expert offload.
- SeqMoE's recall as a function of horizon h > 1 could not be read this session (the PDF fetch returned no text).

---

## Q3. What do consumer users get today with --n-cpu-moe (gpt-oss-120b, Qwen3-30B-A3B, Qwen3-Next-80B, GLM-4.5-Air on a 4090/5090)?

### Takeaway
Community numbers for gpt-oss-120b on one RTX 5090 span a factor of about 5 depending on build date and host:
- 8.1–9.6 tok/s on a Nov 2025 build with `--n-cpu-moe 21`;
- 36–47.6 tok/s on 2026 builds with Zen 5 and DDR5.

That spread is itself a framing point: host memory and build matter more than the GPU. No reliable single-5090 offload
numbers were found for Qwen3-Next-80B-A3B or GLM-4.5-Air.

### Cited Findings
- gpt-oss-120b, RTX 5090, `--n-cpu-moe 21`: 9.60 tok/s at 87 tokens and 8.14 tok/s at 5,650 tokens, "from around 3.4
  t/s" without offload tuning. The same article: RTX 3090 with EPYC 7343 and DDR4-3200, `--n-cpu-moe 27`: 1.41–1.81
  tok/s. Article dated 10 Nov 2025. — [hardware-corner](https://www.hardware-corner.net/guides/gpt-oss-offloading-moe-layers/)
- gpt-oss-120b MXFP4, RTX 5090 with Ryzen 5 9600 and DDR5-5600, `--n-cpu-moe 20`: 46.5 tok/s peak decode, 588 tok/s
  prefill at 2K. Single synthetic request, 26–29 Sep 2026 (witcheer dataset). — [contextstudios witcheer](https://www.contextstudios.ai/local-ai/witcheer--gpt-oss-120b-mxfp4-cpumoe)
- moe-autopilot (RTX 5090, 9950X3D, DDR5-6000), gpt-oss-120b with `--n-cpu-moe 25`: 36.2 → 47.6 tok/s with a static
  hot set (+31.6%). — [moe-autopilot](https://github.com/JigSawPT/moe-autopilot) (from earlier notes)
- CUDA graphs work with `--n-cpu-moe` since b7821 (24 Jan 2026), PR #18934. Before that, "cuda graphs get disabled
  when there are splits". — [llama.cpp PR #18934](https://github.com/ggml-org/llama.cpp/pull/18934) (from earlier notes)
- Qwen3-30B-A3B: "∼30 tok/s on 6 GB VRAM + 32 GB RAM" with `--n-cpu-moe`, quoted by Budgeting Bytes. SeqMoE measured
  llama.cpp static offload at 30.9 tok/s vs SeqMoE 104.1 (RTX 4090). — [arXiv 2609.04238](https://arxiv.org/pdf/2609.04238); [arXiv 2609.12978](https://arxiv.org/abs/2609.12978) (from earlier notes)
- Qwen3-Next-80B-A3B after the CUDA speed-ups (Dec 2025), multi-GPU users only:
  - RTX 3080 + 5060 Ti at IQ2_S: 33.7 tok/s;
  - 5070 Ti + 5060 Ti at IQ3_XS with partial CPU offload: 36.6 → 40.4 tok/s.
  - — [reddit snapshot 5 Dec 2025](https://reddit.sentinel-team.org/posts/1pec8hz/snapshots/2025-12-05T08%3A30%3A58.532537Z)
- Unsloth's Qwen3-Next guide gives no tok/s, only "comparable to a 3B–5B model". — [Unsloth Qwen3-Next guide](https://unsloth.ai/docs/models/tutorials/qwen3-next)

### Inferences
- The gap between 8–10 tok/s (Nov 2025) and 36–47 tok/s (Sep 2026) on the same GPU is consistent with the CUDA-graph
  fix of Jan 2026 and faster hosts. Use it in the introduction to motivate a host-rate-normalised bound rather than
  quoting raw tok/s.
- The witcheer number (46.5 tok/s on a 6-core 9600 with DDR5-5600) is a peak, not a mean. Treat it as an upper anchor.

### Gaps
- No single-RTX-5090 `--n-cpu-moe` numbers found for GLM-4.5-Air or Qwen3-Next-80B-A3B with stated host RAM speed.
- r/LocalLLaMA posts could not be fetched directly; only mirrors and aggregators were reachable.

---

## Q4. Which newer, larger many-expert models are realistic on a 5090 with 128–192 GB DRAM, and would one strengthen the evidence cheaply?

### Takeaway
**Qwen3-Next-80B-A3B** is the cheapest model that adds something new:
- 512 experts, top-10, a shared expert, 48 layers, tiny 3.1M-parameter experts, and an MTP head;
- llama.cpp support is in the engine's base;
- it fits easily in 128 GB.

GLM-4.5-Air (128 experts, top-8, a shared expert, 46 layers, MTP) is realistic but resembles Qwen3-30B-A3B. Qwen3-235B
needs about 2-bit weights in 128 GB. DeepSeek-V3.x and Kimi K2 do not fit at useful precision.

### Cited Findings
- Qwen3-Next-80B-A3B-Instruct config: `num_experts` 512, `num_experts_per_tok` 10, `shared_expert_intermediate_size`
  512, `moe_intermediate_size` 512, `hidden_size` 2048, 48 layers, `full_attention_interval` 4 (hybrid linear
  attention). — [Qwen/Qwen3-Next-80B-A3B-Instruct config.json](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
- GLM-4.5-Air config: `n_routed_experts` 128, `num_experts_per_tok` 8, `n_shared_experts` 1, `moe_intermediate_size`
  1408, `hidden_size` 4096, 46 layers, `first_k_dense_replace` 1, `num_nextn_predict_layers` 1. — [zai-org/GLM-4.5-Air config.json](https://huggingface.co/zai-org/GLM-4.5-Air)
- The engine's llama.cpp base `4da6337` (committed 27 Sep 2026) defines `LLM_ARCH_QWEN3NEXT`, `LLM_ARCH_QWEN35MOE`,
  `LLM_ARCH_GLM4_MOE`, `LLM_ARCH_DEEPSEEK2`, `LLM_ARCH_LLAMA4`, `LLM_ARCH_KIMI_K`, `LLM_ARCH_MINIMAX_M2/M3`, plus
  `LLM_TENSOR_NEXTN_*`. Checked with `git log` and grep of `src/llama-arch.h`. — [local engine worktree](/home/claude/scratch/lc-ec/src/llama-arch.h)
- Qwen3.6-35B-A3B (256 routed experts) at Q4_K_XL reaches 139.9 tok/s on one 24 GB RTX 3090. The benchmark mentions
  no CPU offload, so it appears to run fully on the GPU. That would make it a weak offload case on a 32 GB 5090 unless
  run at Q8. — [hackmd thc1006](https://hackmd.io/@thc1006/SJly6IE6Wx)

### Inferences
- **Qwen3-Next-80B-A3B per-expert size is an estimate:** 3 × 2048 × 512 ≈ 3.1M parameters ≈ 1.7–1.8 MB at about 4.5
  bits per weight. Per-copy latency and launch overhead, not bandwidth, may dominate fetches. This is a genuine test of
  whether the bound and W50 law hold at E=512 and fine granularity, the direction DeepSeek-style models take.
  - One rental can trace own text, run the simulator and 2 engine cells: about 3 h and $2–3 plus a ~50 GB download
    (GGUF size unverified).
  - **Risk:** the 4.5k-line patch may assume no shared expert and standard attention.
- **GLM-4.5-Air size is an estimate:** about 106B parameters, roughly 60–70 GB at 4–5 bits. Its experts are 3 × 4096
  × 1408 ≈ 17.3M parameters. It would add a fourth W50 point close to Qwen3's regime (C/k similar), so less
  informative per dollar.
- **Qwen3-235B-A22B and DeepSeek-V3.x / Kimi K2 sizes are estimates:** about 470 GB and 377+ GB at ≥4 bits. They need
  ~2-bit quants for 128–192 GB hosts, and downloads of 100–200 GB per rental (at ~130 GB of net budget per launch in
  past jobs). Not cheap; skip.
- Llama 4 Scout (16 experts, top-1 plus shared) is a degenerate cache case: few, large experts. It would mainly test
  the bound at C/k near 1. Low priority.

### Gaps
- GGUF file sizes for Qwen3-Next-80B-A3B and GLM-4.5-Air quants were not retrieved (the HF listing was not fetched).
- Whether the expert-cache patch handles Qwen3-Next's shared expert and gated-DeltaNet graph was not checked.
- No information found on any gpt-oss successor model.

---

## Q5. Any cheap hardware-side lever (full PCIe 5.0 x16, pinned memory, GPU-initiated copies, CUDA graphs) that materially changes batch-1 offload speed in 2025–2026?

### Takeaway
No. The engine already uses the reported levers: zero-copy kernels from pinned memory, CUDA graphs, and side-stream
copies. The binding resource is host DRAM bandwidth, shared by CPU experts and PCIe reads, not the link.

The hardware-side result worth adding is a measurement: host-level variance in PCIe and DRAM rates across rentals,
which feeds Design #2.

### Cited Findings
- Repository `concur.cu`: the link carries 57 GB/s alone and 38 GB/s alongside the CPU expert threads. DRAM serves
  CPU experts at 62 GB/s alone and 37 GB/s while the link also reads. — [prefetch_lookahead.md](../MoE%20offload%20system%20race%20plan/prefetch_lookahead.md)
- Vast API: measured `pcie_bw` median 47.8 GB/s on desktop RTX 5090 hosts (PCIe 5.0 x16) and 22.6 GB/s on 4090
  hosts. — [Vast API snapshot, earlier notes](https://console.vast.ai/api/v0/bundles/)
- KTransformers (SOSP'25) attributes 1.23× of its decode speed-up to CUDA Graph, 2.22× to its AVX-512 kernel and up to
  1.63× to NUMA-aware tensor parallelism. — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf) (from earlier notes)
- paddock issue #8 (Qwen3.6-35B-A3B, 16 GB RTX 5060 Ti, PCIe 4.0 x8): 62 tok/s with a VRAM slot cache vs 9 tok/s with
  zero-copy only; "spec loses on a PCIe-bound cache at this size". — [paddock issue #8](https://github.com/truespar/paddock/issues/8) (from earlier notes)
- exllamav3 PR #341: zero-copy streamed *prefill* from a page-locked arena, about 2× long-context prefill. Prefill, not
  decode. — [exllamav3 PR #341](https://github.com/turboderp-org/exllamav3/pull/341) (from earlier notes)

### Inferences
- Links already run at about 89% of PCIe 5.0 x16's 64 GB/s alone (57 GB/s), so "use PCIe 5 fully" leaves little.
  Overlap is limited by DRAM contention (38 + 37 GB/s concurrently).
- Lossless compression of expert bytes on the wire would cut bytes only for BF16/FP8 weights. MXFP4 and Q4_K weights
  are near-incompressible (an inference; no source checked). Likely to fail for these models.
- NUMA or multi-channel servers (EPYC) change the host rate, not the mechanism. They belong in the variance study as a
  second CPU class, not as an innovation.

### Gaps
- No 2026 source reports a GPU-initiated-copy or pinned-memory change that moves batch-1 *decode* by more than a few
  percent on a 5090 beyond what llama.cpp's CUDA-graph fix already delivered.
- The Vast `pcie_bw` field's measurement method is not documented.
