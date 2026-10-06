# Realisable sources of MoE routing foresight: horizon, accuracy, overhead, training cost

Scope: methods that predict which experts future layers or future tokens will route to, for batch-1 or small-batch MoE decode. Each entry gives the method's horizon (in layers and tokens), its reported accuracy (with the exact metric, model and setting), its overhead, and whether it needs training.

Token-equivalent conversion used throughout: gpt-oss-120b has 36 MoE layers, so 1 layer = 1/36 ≈ 0.028 token. Qwen3-30B-A3B has 48 layers, so 1 layer ≈ 0.021 token.

Method: arXiv PDFs were downloaded and text-searched for every paper listed. Dates are arXiv first-version dates from the arXiv API unless noted otherwise. Numbers marked "figure-only" could not be read from text.

## Q1. Layer-ahead / pre-gating predictors: how accurate are they at 1, 2 and 3+ layers ahead?

### Takeaway
Within a token, next-layer prediction is a solved problem:
- Training-free gate lookahead gives roughly 65–97% per-layer accuracy, depending on model and metric.
- Small trained predictors give 85–95%.

Accuracy falls with layer distance, though. Most papers stop at 2–3 layers, about 0.06–0.08 token. ExpertFlow (Oct 2025), the only paper that reports out to 20 layers, fits an exponential decay with asymptotes of about 26–33% for gate reuse and about 60–65% for a trained predictor.

**No layer-ahead method reaches even one token of horizon**, so this family cannot supply the 4–15-token horizon (W50) the paper needs.

### Cited Findings
**Training-free gate lookahead (apply layer l+1's router to an earlier hidden state)**
- **Mixtral-offloading** (arXiv 2312.17238, v1 28 Dec 2023) introduced the heuristic: "applying next layer's gating function to previous layer's hidden states". It fetches only 1–2 likely experts and reports no accuracy figure. — [arXiv 2312.17238](https://arxiv.org/abs/2312.17238) (quote verified in companion notes `MoE offload paper novelty recheck/fetch_prefetch_prior_art.md`)
- **AdapMoE** (arXiv 2408.10284, v1 19 Aug 2024).
  - Mechanism: "utilizes the gate functions from the subsequent layer directly for predictive prefetching", plus a "predictive gate dedicated to the first layer's expert selection".
  - Accuracy: "most layers maintain a high level of expert activation prediction accuracy, achieving rates around 90%" (Mixtral-8x7B, Fig. 9b).
  - Horizon: adaptive gating lets it "prefetch experts for the next two or three layers".
  - Overall speedup is 1.35×. Prediction is described as working "without modification and finetuning of models".
  - — [arXiv 2408.10284](https://arxiv.org/abs/2408.10284)
- **HOBBIT** (arXiv 2411.01433, v2 6 Nov 2024), on Mixtral-8x7B:
  - "the top-1 expert prediction accuracy for the next layer is very high, averaging 96% across layers. Even for the next two or three layers, the accuracy remains around 90% on average".
  - It stacks the p next-layer gates into one GEMV ("Stacking Computer").
  - It warns that low accuracy causes penalties because a cudaMemcpy cannot be interrupted.
  - — [arXiv 2411.01433](https://arxiv.org/abs/2411.01433)
- **ProMoE** (arXiv 2410.22134, v1 29 Oct 2024) measured the "skip-based" (gate-reuse) predictor: high accuracy on DeepSeek-MoE but "only 66.9%" average on Qwen2-57B-A14B (QW-2), "because the gate function in the QW-2 model is sensitive to input variations". — [arXiv 2410.22134](https://arxiv.org/abs/2410.22134)
- **Fate** (arXiv 2502.12224, Feb 2025; v2 7 May 2025): cross-layer gate, "prefetch accuracy of 97.15%" (with confidence thresholding), plus an expert hit rate of 99.08% using its shallow-favoring cache. The pre-attention paper says Fate's "prediction part contributes 78.8% to accuracy". — [arXiv 2502.12224](https://arxiv.org/abs/2502.12224); [arXiv 2511.10676](https://arxiv.org/abs/2511.10676)
- **Speculating Experts** (Madan et al., arXiv 2603.19289, v1 9 Mar 2026).
  - Mechanism: a quasi-hidden state q_l = LN_{l+1}(d_l + r_l).
  - Qwen3-30B-A3B: "most of the inter-layer drift occurs within the first two layers … Beyond these layers … recall@k of approximately 90% on average".
  - Optional trained estimators: Qwen3-30B-A3B about 90% after 4M tokens; GPT-OSS-120B and GPT-OSS-20B reach "average hit rates of 83% and 88%" after 5M tokens.
  - TPOT is 5–14% lower than on-demand loading on an A6000, where CPU–GPU transfer is 84–88% of TPOT.
  - — [arXiv 2603.19289](https://arxiv.org/abs/2603.19289)
- **DALI** (arXiv 2602.03495, v1 3 Feb 2026): "Residual-Based Prefetching" adds a per-layer residual vector to the gate input, with "no fine-tuning or retraining". I found no standalone accuracy figure in the text. — [arXiv 2602.03495](https://arxiv.org/abs/2602.03495)
- **PROBE** (arXiv 2602.00509, v1 31 Jan 2026, expert-parallel serving, not batch-1).
  - Untrained one-layer "Lookahead Gating" "suffers from feature drift with only around 70%–80% accuracy".
  - Online distillation of a residual raises "Top-K Accuracy to 87%–94% across layers".
  - — [arXiv 2602.00509](https://arxiv.org/abs/2602.00509)
- **Si et al.** (arXiv 2608.12103, v1 12 Aug 2026): a production one-layer router-lookahead predictor (PILOT) "recalls 64.7% of decode-time selections, with a per-layer range of 1–84%". — [arXiv 2608.12103](https://arxiv.org/abs/2608.12103)
- **Our own engine** (companion notes): layer-ahead recall for gpt-oss-120b is 84% / 97% at top-4 / top-8 candidates, and 87% / 98% for gpt-oss-20b. — `research_notes/MoE offload paper novelty recheck/fetch_prefetch_prior_art.md`

**Trained next-layer predictors**
- **Pre-gated MoE** (Hwang et al., arXiv 2308.12066, v1 23 Aug 2023; ISCA 2024).
  - Mechanism: "the pre-gate function in the N-th MoE block selects the experts to activate for the (N+1)-th MoE block". The model is fine-tuned, so routing is exact by construction (the pre-gate *is* the router).
  - Pre-gates trained 2 or 3 blocks ahead (N=2/3) were also evaluated: "the model accuracy gradually decreases as the pre-gate function's activation level increases (from N=1 to 3)".
  - Models: Switch-Base/Large (T5-family encoder-decoders).
  - — [arXiv 2308.12066](https://arxiv.org/abs/2308.12066)
- **ProMoE** learned predictor (MLP on intermediate states).
  - "average accuracy of 84.7%".
  - Stride prefetching raises the prediction distance by 1 (2 layers), and accuracy "only declines by 5%".
  - Models: DeepSeek-MoE, DeepSeek-V2-Lite, Qwen1.5-MoE, Qwen2-57B-A14B, Mixtral-8x7B, in llama.cpp and Transformers.
  - — [arXiv 2410.22134](https://arxiv.org/abs/2410.22134)
- **ExpertFlow (Shen et al.)** (arXiv 2510.26730, v1 30 Oct 2025): an adaptive "step size S" for cross-layer prediction, with accuracy measured out to about 20 layers ("numSteps", Fig. 8).
  - Accuracy is fitted as f(t)=a·e^(−bt)+c. Asymptotic accuracy c:
    - trained predictor: 63.44% (DeepSeek), 65.31% (Qwen1.5), 60.45% (Qwen2);
    - pre-gate (gate reuse): 26.43%, 33.29% and 29.61% respectively.
  - "improved accuracy by an average of 21.79% (up to 30.36%) compared to the pre-gate method".
  - The predictor uses token IDs plus recent expert activation states, and requires training.
  - — [arXiv 2510.26730](https://arxiv.org/abs/2510.26730)
- **DuoServe-MoE** (arXiv 2509.07379, v1 9 Sep 2025): a trained layer-level next-layer predictor. Training uses "only 2.5% of the dataset"; trace collection "takes about 5 to 8 hours". Table III, read column-wise (my reading of the extracted table):

  | Model | Dataset | Exact top-k: DuoServe | Exact top-k: MIF | ≥ half correct: DuoServe | ≥ half correct: MIF |
  |---|---|---|---|---|---|
  | Qwen3-30B-A3B | Orca | 56.12% | 38.45% | 98.14% | 82.33% |
  | Qwen3-30B-A3B | Squad | 55.33% | 38.22% | 98.10% | 81.59% |
  | Mixtral-8x7B | Orca | 66.85% | 42.33% | 95.45% | 81.32% |

  MIF is presumably MoE-Infinity. — [arXiv 2509.07379](https://arxiv.org/abs/2509.07379)
- **Pre-attention expert prediction** (Zhu et al., arXiv 2511.10676, v1 10 Nov 2025).
  - Mechanism: two linear layers (intermediate size 2048) on the activations *before* the attention block of the same layer, trained with a ranking-aware loss.
  - Accuracy: "93.03% … DeepSeek V2 Lite, 94.69% on Qwen3-30B, and 97.62% on Phi-mini-MoE", described as "about 15% improvement on absolute accuracy over … FATE".
  - Prediction cost is 0.075–0.129 ms ("prediction computation"), which fits within the attention window.
  - Horizon is less than one layer (the attention block), but it covers layer 0.
  - — [arXiv 2511.10676](https://arxiv.org/abs/2511.10676)
- **MoE-Beyond** (arXiv 2508.17137, v1 23 Aug 2025).
  - Mechanism: a transformer over token embeddings plus layer IDs, trained on 66M traces from DeepSeek-V2-Lite.
  - Metrics: 97.5% "position-wise accuracy" (exact set match) and 86.6% macro-F1.
  - Simulated GPU cache hit rate rises from 17% to 72% with 10% of experts resident.
  - This is same-token, all-layer prediction; it has no cross-token horizon.
  - — [arXiv 2508.17137](https://arxiv.org/abs/2508.17137)

**Trace-matching predictors**
- **MoE-Infinity** (arXiv 2401.14361, Jan 2024): request-level and iteration-level "Expert Activation Matrices", with the activation state modelled "by a Markov Chain". It is training-free. Reuse skew "is only present at the request level; after processing multiple requests, the skew dis[appears]". I found no standalone recall figure in the text. — [arXiv 2401.14361](https://arxiv.org/abs/2401.14361)
- **FineMoE / fMoE** (Yu et al., arXiv 2502.05370, v1 7 Feb 2025; EuroSys 2026): "expert maps" (per-layer probability trajectories) plus semantic embeddings, searched by similarity. Fig. 4 plots hit rate against "prefetch distance" from 0 to 30 layers (figure-only). Reports "improves expert hit rate by 39%" and "reduces inference latency by 47%". — [arXiv 2502.05370](https://arxiv.org/abs/2502.05370)
- **SeqMoE's characterisation of the whole family** (arXiv 2609.12978): "These methods rely on residual-induced cross-layer similarity, causing accuracy to degrade rapidly with prediction distance and limiting lookahead to only one to three layers". Trace-matching methods (MoE-Infinity, FineMoE) are "again limiting lookahead to at most three layers". It also notes that on Qwen3-30B-FP8 with an RTX 4090 over PCIe 4.0, "one layer's computation overlaps the transfer of only 1.39 experts on average". — [arXiv 2609.12978](https://arxiv.org/abs/2609.12978)
- **MoE-APEX** (Tang et al., ASPLOS 2026, the HOBBIT authors) "directly feeds shallow-layer hidden states into future routers" (as cited by SeqMoE). I did not retrieve the paper itself. — [SeqMoE reference list, arXiv 2609.12978](https://arxiv.org/abs/2609.12978)

### Inferences
- The longest layer-ahead horizon with a quoted accuracy near 90% is 2–3 layers (HOBBIT on Mixtral, about 90%; ProMoE at distance 2, −5 points).
  - That is 0.06–0.08 token on our models, **50–240× shorter than W50 = 4–15 tokens**.
  - Past roughly 10 layers, accuracy approaches ExpertFlow's asymptotes: about 30% for gate reuse and about 63% for a trained predictor.
  - Even a perfect layer-ahead predictor only feeds *paced prefetch within the current token*. Our engine already captures that (+14–16%). Si et al. likewise measured only +5.0% from *oracle* one-layer advice on an SSD tier.
- Reported metrics are not comparable as given. They include:
  - top-1 accuracy (HOBBIT, SP-MoE);
  - recall@k (Speculating Experts);
  - exact-set match (DuoServe, MoE-Beyond);
  - "≥ half correct" (DuoServe);
  - batch-level accuracy (ExpertFlow 2024).

  The same predictor class on Qwen3-30B-A3B reads 55% exact-set but 98% "≥ half". The paper should report recall@k and recall@(k+m) at a fixed candidate budget so its numbers line up with SeqMoE.
- Our layer-ahead recall (84% / 97% for gpt-oss-120b) sits at or above every published training-free number for gpt-oss. The engine has no headroom left in this family.

### Gaps
- I found no per-distance accuracy table for 2, 3 and 4+ layers on gpt-oss-120b or Qwen3-30B-A3B in any paper. HOBBIT (Mixtral) and ExpertFlow (DeepSeek/Qwen1.5/Qwen2) are figure-only or on other models.
- fMoE's hit rate against prefetch distance is figure-only.
- MoE-APEX (ASPLOS 2026) and Lina (ATC 2023) were not retrieved.
- I could not confirm whether AdapMoE's first-layer "predictive gate" is trained; the text implies a dedicated predictor.

## Q2. Cross-token expert prediction (t+1 … t+W) at batch 1, and measured temporal locality

### Takeaway
Same-layer expert persistence between adjacent tokens is weak: about **2× random chance**, measured independently on Qwen3-30B-A3B and across models by ST-MoE. On Qwen3-30B-A3B (top-8 of 128) that means only about 1 of 8 experts carries over.

Persistence-style and first-order (Markov) cross-step predictors are therefore inaccurate. The only published method with a genuine multi-token horizon and high recall on our two models is **SeqMoE (11 Sep 2026)**. It trains a Mamba2 sequence model over full-layer-stack activation vectors and reports:
- recall@(k+3) of 90.9% on Qwen3-30B-A3B and 90.8% on GPT-OSS-120B for the next step;
- an average drop of only 3.58 points when rolled out recursively to 8 steps;
- predictor cost of 3.46% of execution time.

It is trained on 45K traces. Its recall uses k+3 candidates, so precision is capped at 4/7 for gpt-oss and 8/11 for Qwen3. It has not been independently reproduced.

### Cited Findings
**Measured cross-token locality**
- **Mixtral 8x7B** (arXiv 2401.04088, v1 8 Jan 2024): "consecutive tokens are often assigned the same experts".
  - Random baseline: 12.5% for "First choice" and about 46% for "First and second choice". "Repetitions at the first layer are close to random, but are significantly higher at layers 15 and 31".
  - Extracted first-choice values, by my column reading of Table 5: layer 0 is 13.6–14.9% across domains; the next blocks are 23.6–28.4% and 19.7–22.7%, which I take to be layers 15 and 31.
  - — [arXiv 2401.04088](https://arxiv.org/abs/2401.04088)
- **Cacheable by Design?** (arXiv 2608.18261, v1 18 Aug 2026) traced Qwen3-30B-A3B (48 layers, 128 experts, top-8) over 8,000 tokens each of prose, code, math and medical text:
  - "Expert reuse between adjacent tokens is 2.0× chance (temporal locality)";
  - "95% of traffic flows through 52.5% of experts";
  - per-domain expert sets overlap 0.11–0.16 for code against 0.33–0.42 among prose, math and medical.
  - — [arXiv 2608.18261](https://arxiv.org/abs/2608.18261)
- **ST-MoE** (Zhao et al., arXiv 2606.15453, v1 13 Jun 2026): under random routing, expected overlap between consecutive tokens is E = K²/N, and "Across multiple LLMs and datasets, the observed overlap consistently exceeds this baseline by nearly 2 times". Cross-layer co-activation is non-independent (chi-squared p < 0.01). — [arXiv 2606.15453](https://arxiv.org/abs/2606.15453)
- **Local routing consistency** (Liang et al., arXiv 2505.16056, v1 21 May 2025; ICLR 2026), covering 20 MoE LLMs.
  - SRP is the best possible F1 of a "segment router" that picks one fixed expert set per m-token segment, an oracle upper bound. At m = 16:
    - Group 1 (LLaMA-MoE-v2 … OLMoE): above 0.5;
    - Group 2 (Mixtral-8x7B …): about 0.48;
    - Group 3 (… DeepSeekMoE): about 0.36;
    - Group 4 (NLLB, Switch): below 0.31.
  - "Qwen3 and GRIN-MoE show high SRP".
  - SCH is the hit rate of an oracle cache that evicts experts least used in the next m tokens. The authors conclude "cache sizes 2x the size of active parameters achieve the best segment caching results on most models".
  - — [arXiv 2505.16056](https://arxiv.org/abs/2505.16056)
- **Temporally Extended MoE** (arXiv 2604.20156, v1 22 Apr 2026): MoE models "switch experts at nearly every token". On gpt-oss-20b the baseline switch rate is "over 50%". — [arXiv 2604.20156](https://arxiv.org/abs/2604.20156)
- **Mixture of Cache-Conditional Experts** (Skliar et al., arXiv 2412.00099, v1 27 Nov 2024; TMLR 2025): "state-of-the-art MoEs often lack strong temporal locality in expert selection, leading to inefficient … LRU caching". — [arXiv 2412.00099](https://arxiv.org/abs/2412.00099)
- **MoE-Infinity**: experts "exhibit skewed reuse patterns when continuously decoding tokens within a single request". — [arXiv 2401.14361](https://arxiv.org/abs/2401.14361)
- **FreeToken** (arXiv 2608.16157, Aug 2026): "During decode, routing shows strong temporal expert locality … a routing consistency measured across model families (Liang et al., 2025)". — [arXiv 2608.16157](https://arxiv.org/abs/2608.16157)

**Cross-step predictors**
- **SeqMoE** (Wang et al., USTC; arXiv 2609.12978, v1 11 Sep 2026). Full details:
  - Predictor: "we encode the activation state of the full layer stack per generation step" and train Mamba2 (Linear → Mamba2 → Linear) on next-step prediction, with scheduled sampling for recursive rollout.
  - Evidence of learnable structure: "training fails to converge after shuffling … predictability stems from sequence-level dependence".
  - Next-step recall: "Selecting the top k+3 experts yields recalls of 90.90%, 88.39%, 90.76%, and 85.81% on QW3, QW36, GPT, and DSV4" (Qwen3-30B-A3B, Qwen3.6-35B-A3B, GPT-OSS-120B, DeepSeek-V4-Flash).
  - Multi-step recall: "Seq-8 recursively predicts eight steps, with an average recall drop of only 3.58 percentage points relative to Seq-1". Recall on sequences longer than 4,096 tokens "remains comparable".
  - Deployment: k' = k+3 prefetch tasks per prediction; c = 8 recursive steps for the cache policy; "two predictors, triggered at l' = L/2 and l' = L".
  - Overhead: "Predictor inference accounts for 3.46% of execution time on average". "Increasing the number of predictors extends the overlap window but introduces additional prediction overhead and slightly reduces accuracy".
  - Training data: 45K traces, plus 5K for testing.
  - Setting: batch size 1. Hardware: RTX 4090 (Qwen3-30B-A3B-FP8), RTX 5090, RTX PRO 6000 (GPT-OSS-120B).
  - Hit rates (prefetch hits count as hits) at 15/25/35/40/45% capacity: 84.56/91.72/95.09/96.11/96.97%, against 60.02/74.24/84.49/86.37/88.50% for LRU.
  - It dismisses first-order cross-step methods (ST-MoE, Patterns-MoE) as "memoryless, first-order Markov model[s] … too coarse for precise scheduling".
  - — [arXiv 2609.12978](https://arxiv.org/abs/2609.12978)
- **ST-MoE**: correlation tables and history tables combining cross-layer correlation with the previous token's same-layer experts, updated online (no offline training). "85% expert prediction accuracy". It is a TSMC 40 nm accelerator design. — [arXiv 2606.15453](https://arxiv.org/abs/2606.15453)
- **Patterns behind Chaos / Patterns-MoE** (Yu et al., UCSD; arXiv 2510.05497, v1 7 Oct 2025; ISCA 2026): profiles token-level correlation ("expert selection relations for the same layer between adjacent tokens", Ob2), layer-level correlation and prefill–decode correlation, then builds a "data-driven predictor" for wafer-scale multi-unit systems. I could not extract numeric accuracy. — [arXiv 2510.05497](https://arxiv.org/abs/2510.05497)
- **eMoE** (arXiv 2503.06823, v1 10 Mar 2025): a task-aware predictor run "after every p prompts" to choose which experts to load, using 0.24–1.3% of model memory. Its horizon is at prompt level (coarse working-set foresight), not per-token routing. — [arXiv 2503.06823](https://arxiv.org/abs/2503.06823)
- **Cache-Aware Joint Router Adaptation** (arXiv 2609.04895, v2 10 Sep 2026) trains a "Temporal Router" whose objective uses "the next-token native distribution" to supervise same-layer coverage.
  - Models: Qwen3-30B-A3B-Instruct-2507 and GPT-OSS-20B.
  - On Qwen3 it gains 1.15–18.03 points of adjusted hit rate over the strongest prefetcher (ProMoE, FineMoE, Least-Stale, Temporally Extended MoE). GPT-OSS results are "competitive but task-dependent".
  - Evaluated in simulation: "They establish neither wall-clock speedup nor energy savings".
  - — [arXiv 2609.04895](https://arxiv.org/abs/2609.04895)
- **Zhang** (arXiv 2608.07911, Aug 2026): a causal next-use-distance predictor (log recency, frequency, previous gap, gate mass, layer and so on) "recovers −11.4% of the gap", choosing the optimal victim 3.4% of the time vs 20.6–22.1% for LRU/LFRU, despite a transfer R² of 0.237. — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)

### Inferences
- Expected same-layer overlap between adjacent tokens:
  - Qwen3-30B-A3B: chance is k²/N = 0.5 expert, so "2× chance" means about 1 of 8 experts per layer carries over (about 12.5%).
  - gpt-oss-120b: chance is 16/128 = 0.125 expert, so 2× would be 0.25 of 4. **This applies the 2× factor to gpt-oss by extrapolation; no source measures gpt-oss.**

  Persistence predictors ("the expert you just used will be used again") therefore have low recall at any horizon. That is consistent with our logistic-regression reuse predictor recovering only 7–13%, and with Zhang's −11.4%.
- SeqMoE's claim differs in kind. Instead of per-expert reuse statistics, it models the *joint* full-stack activation vector over time, and its shuffle test shows the information is in the sequence. That is the most credible explanation for why per-expert features (decayed counts, recency, transitions) fail where a sequence model succeeds.
- Projected from SeqMoE's numbers, recall@(k+3) at 8 steps is about 87% on both Qwen3-30B-A3B and GPT-OSS-120B (90.9 − 3.6 and 90.8 − 3.6). It is ambiguous whether the 3.58-point drop averages over steps 1–8 or applies at step 8.
- Recall is quoted at k+3 candidates, so precision is at most 4/7 ≈ 57% for gpt-oss-120b and 8/11 ≈ 73% for Qwen3. For admission or bypass decisions, wrong candidates cost bandwidth and pollute the cache; SeqMoE itself notes that wrong fetches "can also evict cached experts that would otherwise be reused".
- SRP below 0.5 at m = 16 for most model groups means even an oracle that commits to one expert set per 16-token window gets F1 ≤ 0.5. **Window-set foresight is weak; per-step trajectory foresight (SeqMoE-style) is required.**

### Gaps
- SeqMoE gives no per-step recall curve (steps 1…8), no exact top-k recall (only top-(k+3)), no predictor parameter count and no wall-clock training time in the text I extracted. I found no code link. Its Belady-8 vs Belady-F8 comparison (Fig. 3) is figure-only.
- No independent reproduction of SeqMoE exists, and it postdates nearly everything else here (11 Sep 2026).
- I found no published adjacent-token reuse statistic for gpt-oss-120b (top-4 of 128).
- Patterns-MoE token-level correlation values could not be extracted.

## Q3. Draft-model / speculative-decoding foresight: how accurate is routing on draft tokens versus actual tokens?

### Takeaway
Draft-based foresight gives all-layer routing for the next γ drafted tokens. Given the same token, routing fidelity is about 88–91%:
- SP-MoE: 88–89% top-1 expert accuracy, using draft attention output fed to the target gate.
- MoE-SpeQ: 90.9% top-4 for an INT4 self-draft on identical inputs.
- DraftExpert: 86–88% prefetch hit with a trained draft expert.

For a non-speculative decode engine, though, foresight for token t+j is only valid if the draft got tokens t+1…t+j right. Its value therefore decays roughly as (acceptance)^j. **No paper reports routing recall against the tokens actually generated as a function of draft depth j**, and none does so for gpt-oss-120b or Qwen3-30B-A3B.

If the engine itself runs speculative decoding, verification consumes exactly the drafted tokens' routing. But the union of experts across the k+1 verified tokens inflates bandwidth, and community measurements show speculative decoding *slowing* A3B MoEs on consumer GPUs.

### Cited Findings
- **SP-MoE** (Chen et al., SYSU/HKU; arXiv 2510.10302, v1 11 Oct 2025; v2 6 Nov 2025).
  - Mechanism: "in each layer l during drafting, the attention output is fed into our predictor, which directly reuses the gating network of the l-th target layer".
  - Similarity of draft and target hidden states reaches up to 94.59% for DeepSeek-Lite, against 59.82% for Mixtral 8×7B and 56.59% for Phi-3.5-MoE.
  - Top-1 expert prediction accuracy is "88.94% … on average across layers" for DeepSeek-Lite, with "Mixtral and Phi-3.5-MoE maintaining approximately 88%".
  - Draft acceptance on HumanEval is 97.42%, 98.15% and 97.01% for the three draft/target pairs.
  - A "cutoff layer" bounds prefetch depth. Hardware: RTX 4090, PCIe 4.0. The pre-attention paper summarises SP-MoE as "more than 70% of prediction accuracy in most layers".
  - — [arXiv 2510.10302](https://arxiv.org/abs/2510.10302); [arXiv 2511.10676](https://arxiv.org/abs/2511.10676)
- **MoE-SpeQ** (Wang et al., SJTU; arXiv 2511.14102, v1 18 Nov 2025).
  - "The INT4 draft model achieves a 90.9% total accurate prediction rate", made up of 44.1% exact plus 46.8% correct set in a different order (companion notes). Measured "between a full-precision FP16 Qwen-MoE model (the target) and a 4-bit quantized (INT4) version (the draft) on the same input sequences" — i.e. teacher-forced, not on draft-generated continuations.
  - It "outperforms a specialized, one-layer-ahead predictor, which only reaches 84.7% accuracy [ProMoE] … and this quantized predictor can predict all layers simultaneously in a single pass".
  - Its Expert Lookahead Buffer covers k tokens × L layers. It uses an "amortization roofline" with I_amort(k) = E[accepted tokens]·S_token / E[synchronous I/O bytes], and T_cycle(k) = max(T_draft(k), T_pcie,init) + T_pcie,new(k) + T_verify(k+1).
  - Up to 2.34× on A100-40G.
  - — [arXiv 2511.14102](https://arxiv.org/abs/2511.14102)
- **DraftExpert** (arXiv 2607.24434, v1 27 Jul 2026).
  - A self-speculative drafter (shared + top-1 + a *trained* "draft expert") with target-expert prefetching. On DeepSeek-V2-Lite and Moonlight-16B-A3B it "raises draft acceptance to 84–87%, and achieves 86–88% prefetch hit rates", for 1.45× decode throughput.
  - Without the trained draft expert, raising r "raises acceptance from 22% to 31% and 42%"; a top-1 drafter causes "hidden-state drift, lower token agreement, and weaker router agreement".
  - "as draft length or r grows, the cumulative unique routed experts used by drafting expands rapidly".
  - — [arXiv 2607.24434](https://arxiv.org/abs/2607.24434)
- **SPICE** (arXiv 2608.21240, v1 21 Aug 2026): a draft model "aligned with the target MoE architecture" with "confidence-aware adaptive lookahead".
  - Minimum lookahead depth to hide a transfer: H_min = P·M_e / (B_PCIe·T_comp), with T_comp = T_attn + K·T_expert.
  - Up to 3.12× TPOT on DeepSeek-V2-Lite and Qwen2-57B-A14B, but with a 3.0–3.5-point GSM8K drop from low-rank surrogates on misses.
  - — [arXiv 2608.21240](https://arxiv.org/abs/2608.21240)
- **EcoSpec** (arXiv 2607.12696, 14 Jul 2026) names "expert scattering": high-confidence draft tokens route to disjoint experts and inflate the verification union. It fine-tunes a target-specific expert predictor. — [arXiv 2607.12696](https://arxiv.org/abs/2607.12696) (companion notes)
- **AcceptMoE** (arXiv 2608.02989, v1 4 Aug 2026): at batch size 1 in SGLang, "Natural routing incurs expert-weight traffic for draft branches that are ultimately discarded". Restricting verifier expert sets costs −0.27 points of mean accuracy against EAGLE-3. — [arXiv 2608.02989](https://arxiv.org/abs/2608.02989)
- **SpecMoEOff** (arXiv 2508.21706, Aug 2025) uses speculative decoding to raise compute per offloaded byte in throughput (large-batch) settings. It is not a routing predictor. — [arXiv 2508.21706](https://arxiv.org/abs/2508.21706)
- **MoE-CORE** (arXiv 2610.01950, v1 1 Oct 2026; abstract only): DeepSeek-V4-Flash on an NPU at 84 GB reaches 21.5 ms TPOT "with approximate expert substitution and multi-token prediction (MTP) at depth 2". It reports no routing-foresight accuracy for MTP tokens. — [arXiv 2610.01950](https://arxiv.org/abs/2610.01950)
- **Community measurement** (companion notes): on Qwen3.6-35B-A3B Q4_K, RTX 3090, batch 1:
  - ngram speculative decoding gives −15%; a 0.8B draft model gives −39% to −54%;
  - "100% draft acceptance cannot rescue it", attributed to expert-union saturation.
  - — [HackMD benchmark](https://hackmd.io/@thc1006/SJly6IE6Wx)

### Inferences
- **Compounding bound (my arithmetic).** If routing foresight for token t+j is useful only when the draft matches the real tokens t+1…t+j:
  - P(valid at j) = Π α_i, where α is per-token acceptance.
  - For recall ≥ 0.75 at W = 8 with about 0.9 routing fidelity, you need α^8 ≥ 0.83, i.e. α ≥ 0.977 per token. At W = 4 you need α ≥ 0.955.
  - With α = 0.85 (DraftExpert's range), P(valid) is 0.52 at j = 4 and 0.27 at j = 8. With α = 0.97 (SP-MoE on HumanEval), 0.97^8 ≈ 0.78, so about 0.70 after routing fidelity.
  - Draft-token foresight can therefore meet the bar only on highly predictable text (code, templated output), not on general chat.
  - Partial credit from a wrong draft token's routing is roughly the base adjacent-token overlap (about 2× chance), which adds little.
- The teacher-forced fidelity figures (MoE-SpeQ 90.9%, SP-MoE 88%) are **upper bounds** on draft foresight. The decisive measurement — target routing on the actually generated future tokens against routing predicted from a W-token draft — is missing from the literature and cheap to run on our traces.
- For our engine, foresight from drafted tokens needs *target-model* routing for those tokens, which normally requires running the target's layers. Cheap options:
  - the target's own router weights on draft hidden states (SP-MoE, which needs a layer-aligned draft);
  - a quantized self-draft (MoE-SpeQ, which costs a second copy of the weights);
  - a Read-ME-style token-level router (Q4).

### Gaps
- No paper reports routing recall against draft depth j on actually generated text. None reports draft-based routing accuracy on gpt-oss-120b or Qwen3-30B-A3B.
- I found no measurement of MTP-head token routing overlap with the realised next token for DeepSeek-V3/V4 or Qwen3-Next.
- SP-MoE accuracy by layer and by draft position is figure-only.

## Q4. Decoupled, pre-computable and cache-friendly routers

### Takeaway
Decoupled routers make routing exactly known before a token's forward pass, but only at a horizon of 0 tokens (they still need the token):
- Read-ME: one small router shared across layers, so all layers are known at once.
- SiDA-MoE: an offline LSTM "hash" predicting all layers.

Both require refactoring or retraining the model (Read-ME uses about 1B tokens on Llama-2-7B), so neither applies to off-the-shelf gpt-oss-120b or Qwen3-30B-A3B.

"Cache-friendly" routing changes model outputs:
- *Training-free* rerouting to cached experts (Skliar) cuts misses by more than 50% at +0.1–3% perplexity.
- *Trained* locality (StickyMoE, Temporally Extended MoE, Cacheable by Design, Cache-Aware Router Adaptation) cuts switches or misses by 60–90%, at measurable quality cost. One pre-registered study failed a ≤ 1% perplexity gate.

These make future routing trivially predictable by restricting it. They are not foresight about the original model.

### Cited Findings
- **Read-ME** (Cai et al., arXiv 2410.19123, v1 24 Oct 2024; NeurIPS 2024).
  - "a novel pre-gating router, decoupled from the MoE backbone". The router is a 1-layer, 4-head transformer of "only 18 million additional parameters" over x≤t, and "the implicit knowledge learned by each router is extensively shared across layers".
  - Built by refactoring Llama2-7B-chat with "1.04 billion tokens".
  - Belady caching with batched requests (Table 4, cache hit ratio for Random / LRU / Belady):
    - cache 3: 50.14% / 52.42% / 61.82%;
    - cache 4: 67.52% / 66.95% / 77.21%;
    - cache 5: 82.91% / 83.48% / 88.03%.
  - The foresight comes from *queued tokens across requests*: "Prior caching methods are limited by layer-wise routing and lack of foresight into future requests".
  - Latency: −6.1% mean and −10% tail.
  - — [arXiv 2410.19123](https://arxiv.org/abs/2410.19123)
- **SiDA-MoE** (arXiv 2310.18859, Oct 2023; MLSys 2024): an offline-trained LSTM "hash function" that "will determine experts to be activated for each token at each layer". It runs in a separate hash-building thread ahead of the inference thread. "up to 99% prediction accuracy" on Switch Transformers; others cite "more than 90% on Switch-base128". — [arXiv 2310.18859](https://arxiv.org/abs/2310.18859); [Fate, arXiv 2502.12224](https://arxiv.org/abs/2502.12224)
- **ExpertFlow (He et al.)** (arXiv 2410.17954, Oct 2024; DAC 2026): a transformer "routing path predictor that estimates expert usage across all MoE layers in a single forward pass". The predictor is 7.21 MB. It reaches "up to 95% expert prediction accuracy" with a 5–10% cross-domain drop, and is batch-oriented (token scheduler). — [arXiv 2410.17954](https://arxiv.org/abs/2410.17954)
- **ProMoE**: a "token-based predictor" (iteration-wise, all layers from the input token, as in prior work) averages "only 58.3%" accuracy. — [arXiv 2410.22134](https://arxiv.org/abs/2410.22134)
- **Pre-gated MoE** (Q1): fine-tuned one-block-ahead router, exact by construction, with model accuracy falling for 2–3-block-ahead variants. — [arXiv 2308.12066](https://arxiv.org/abs/2308.12066)
- **Mixture of Cache-Conditional Experts** (Skliar et al., Qualcomm; arXiv 2412.00099, v1 27 Nov 2024; TMLR June 2025): "reduces cache miss rates by over 50%, with negligible impact on perplexity (0.1%–3%) and downstream task accuracy (<0.1%)", and "speed-up of up to 2×" on two phones. "All of our methods are general and training-free". — [arXiv 2412.00099](https://arxiv.org/abs/2412.00099)
- **StickyMoE / Sticky Routing** (Kayyam, BrainChip; arXiv 2607.08780, v1 12 Jun 2026): an ℓ2 routing-consistency loss on consecutive gate distributions. It "reduces the expert switch rate by up to 60% with less than 4% perplexity degradation" (abstract), with cache misses down up to 3.92× (body). Only small models trained on WikiText-2 were tested. — [arXiv 2607.08780](https://arxiv.org/abs/2607.08780)
- **Cacheable by Design?** (arXiv 2608.18261, v1 18 Aug 2026), a pre-registered negative result on 137M-parameter models.
  - "locality training cuts cache misses up to 60% … but every configuration fails the pre-registered ≤ 1% perplexity gate".
  - "training-free cache-aware rerouting stacks with trained locality — together reaching ∼80% cache-miss reduction at ≤ 3.4% perplexity".
  - — [arXiv 2608.18261](https://arxiv.org/abs/2608.18261)
- **Temporally Extended MoE** (Shen & Henderson, arXiv 2604.20156, v1 22 Apr 2026): an option-critic controller per layer "learns when to switch expert sets and which to load". Applied to gpt-oss-20b with LoRA, it "reduces switch rates from over 50% to below 5% while retaining up to 90% of base-model accuracy on MATH, MMLU, and MMMLU". — [arXiv 2604.20156](https://arxiv.org/abs/2604.20156)
- **Cache-Aware Joint Router Adaptation** (arXiv 2609.04895, Sep 2026): on Qwen3-30B-A3B, +1.15–18.03 points of adjusted hit rate and −4.6–53.3% traffic vs the strongest prefetcher. "Temporally Extended MoE attains 100% hit by restricting routing to its option set, yet has less favorable traffic and accuracy". Simulation only. — [arXiv 2609.04895](https://arxiv.org/abs/2609.04895)
- **eMoE** (a task-aware routing-tolerance study, arXiv 2503.06823): QA tasks keep similarity above 80% with "only 50% of the MoE layers" routed accurately, while summarization falls below 80% at 50%. — [arXiv 2503.06823](https://arxiv.org/abs/2503.06823)

### Inferences
- Read-ME, SiDA and ExpertFlow-2024 extend horizon *across layers*, not *across tokens*: they know all layers of token t once token t exists. At batch 1 the next token does not exist until sampled, so their cross-token horizon is 0. They only become multi-token foresight sources when composed with a draft (Q3).
  - A "Read-ME router + draft tokens" design would make drafted-token routing exact and cheap (an 18M-parameter router, not the target's layers), leaving only draft-token correctness as uncertainty.
  - It needs a refactored, retrained model, so it cannot be applied to unmodified gpt-oss-120b or Qwen3-30B-A3B.
- Cache-friendly routers change the model and win by making routing predictable, not by predicting it. For a paper whose oracle is defined on the *unmodified* model, they are a separate axis (a quality–locality frontier). Including one would require reporting accuracy deltas. Temporally Extended MoE's "up to 90% of base-model accuracy" means at least a 10% loss on the best tasks.
- A reviewer's request for "a decoupled router (Read-ME style)" therefore implies retraining. Among exact, off-the-shelf mechanisms, only learned cross-step sequence predictors (SeqMoE) and draft or self-draft foresight qualify.

### Gaps
- Lina (ATC 2023) was not retrieved, so its horizon and accuracy are unverified.
- I found no Read-ME-style decoupled router retrofitted onto gpt-oss or Qwen3 with fewer than 1B tokens of training.
- StickyMoE's abstract (60%, < 4% perplexity) and body (59%, perplexity *improved* on the medium model) differ slightly; both are on sub-25M or small models.

## Q5. Theory and measured curves: benefit vs prediction horizon and recall

### Takeaway
Classical paging theory says lookahead only helps under "strong lookahead", a window containing l *distinct* future pages, and that naively trusting a learned oracle can be arbitrarily bad. Robust schemes degrade gracefully, from near-OPT down to O(log k).

Empirically:
- Most of the Belady gap comes from *victim ranking by future use*: 84–97% of it, with bypass contributing 3–16% (Zhang, Qwen3-30B-A3B).
- Oracle *one-layer* prediction is worth only about 5% (Si et al.).
- SeqMoE reports that eviction driven by its 8-step *forecasts* (Belady-F8) "closely approaches" 8-step *oracle* eviction (Belady-8) (figure-only).
- Formulas for the minimum *layer* lookahead to hide a transfer exist (SPICE). Speculative-decoding roofline models exist (MoE-SpeQ).

I found no published benefit-vs-horizon curve in *tokens* for MoE expert caching other than SeqMoE's single Belady-8 point and the SCH metric. The paper's W50 ≈ 0.59 (C/k)^1.33 scaling appears to be new.

### Cited Findings
- **Albers** ("On the Influence of Lookahead in Competitive Paging Algorithms", Algorithmica 18, 1997): "A paging algorithm is on-line with strong lookahead l if it sees the present request and a sequence of future requests that contains l pairwise distinct pages". Strong lookahead improves competitive factors. Ordinary lookahead does not help for caching (from the 1992 tech-report abstract: no "improvement" for "caching, the k-server problem and metrical task systems"). — [Springer, Algorithmica 1997](https://link.springer.com/doi/10.1007/PL00009158); [MPI-I-92-143 report page](https://domino.mpi-inf.mpg.de/internet/reports.nsf/c125634c000710d0c12560400034f45a/470f3e826a5f094bc12560400053710f)
- **Lykouris & Vassilvitskii** ("Competitive caching with machine learned advice", arXiv 1802.05399, v1 15 Feb 2018; ICML 2018): "naively following the oracle's recommendations may lead to very poor performance, even when the average error is quite low". Predictive Marker's competitive ratio "decreases as the oracle's error decreases, and … is always capped by O(log k)". — [arXiv 1802.05399](https://arxiv.org/abs/1802.05399)
- **Jiang, Panigrahi & Sun** (arXiv 2006.09509, v1 16 Jun 2020; ICALP 2020): for weighted paging, "neither a fixed lookahead nor knowledge of the next request for every page is sufficient … a combination of the two … suffices to give a 2-competitive algorithm". — [arXiv 2006.09509](https://arxiv.org/abs/2006.09509)
- **Zhang** (arXiv 2608.07911): on Qwen3-30B-A3B at 40% residency and B = 8, LFRU misses 18.01% vs Belady 9.93%. "bypass admission accounts for 15.7% of the gap at B=8 and 3.4% at B=2, leaving 84.3% and 96.6% to future-victim knowledge". — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911)
- **SeqMoE**, Fig. 3: "Belady-8 limits this oracle knowledge to the next eight steps, while Belady-F8 replaces oracle activations with our eight-step forecasts … Belady-F8 closely approaches Belady-8" and "future-aware policies consistently outperform history-based ones". Values are figure-only. — [arXiv 2609.12978](https://arxiv.org/abs/2609.12978)
- **Local routing consistency (SCH)**: the hit rate of an oracle cache using the next m tokens' activation frequency, plotted against m and the cache ratio ρ. Group-1 models show "turning points near ρ = 2". Their claim: "the required information is easier to learn and predict (future activation frequency vs. precise time of next activation)". — [arXiv 2505.16056](https://arxiv.org/abs/2505.16056)
- **Si et al.**: "At 64.7% measured recall, the measured prediction plan changes median iteration time by 0.3% … while oracle one-layer advice improves it by 5.0%" (SSD→DRAM page cache). — [arXiv 2608.12103](https://arxiv.org/abs/2608.12103)
- **SPICE**: minimum layer lookahead H_min = P·M_e / (B_PCIe·(T_attn + K·T_expert)). **SeqMoE** gives one empirical value: 1.39 experts transferable per layer of compute (Qwen3-30B-FP8, RTX 4090, PCIe 4.0). — [arXiv 2608.21240](https://arxiv.org/abs/2608.21240); [arXiv 2609.12978](https://arxiv.org/abs/2609.12978)
- **MoE-SpeQ amortization roofline**: Θ(k) = E[accepted tokens] / T_cycle(k), with a profiled-PCIe T_cycle that includes the union of new experts |E_new(k)|·S_expert/B_PCIe. — [arXiv 2511.14102](https://arxiv.org/abs/2511.14102)
- **HOBBIT** shows qualitatively that the benefit of prefetching flips to a penalty at low accuracy, because non-interruptible copies of wrong experts block the link (Fig. 9). — [arXiv 2411.01433](https://arxiv.org/abs/2411.01433)

### Inferences
- The paper's W50 result connects naturally to Albers. Because the oracle gain is dominated by victim ranking (Zhang), the useful lookahead is "the next C/k-ish *distinct* expert uses per layer", which is strong lookahead in Albers' sense. That explains why the required horizon in tokens grows with C/k.
- The reviewers' "recall ≥ 0.75" is a proxy. Lykouris–Vassilvitskii and Zhang both show that a predictor with decent average accuracy can make eviction *worse* if used naively. A realisable mechanism should be evaluated by the fraction of oracle gain recovered, not by recall alone. It should also be wrapped in a robust fallback (Predictive-Marker-style, or "trust the forecast only above a confidence threshold").
- Si's oracle one-layer value (5%) against our in-engine oracles (16–51% MIN admission, 25–81% paced prefetch at a 3-token lead) suggests that almost all the value sits at multi-token horizons. That is consistent with W50 ≈ 4–15 tokens.

### Gaps
- The exact competitive ratios in Albers (1997) for strong lookahead could not be verified from the abstract. I believe it shows a deterministic k − l and a randomized ratio around 2H_{k−l}, but this is unverified.
- No paper gives benefit (speed-up or fraction of the Belady gap) as a function of recall *and* token horizon for MoE expert caches. SeqMoE's Fig. 3 is the closest, but its numbers are figure-only.

## Q6. Dense comparison, and which sources could reach W = 4–16 tokens with recall ≥ 0.75 at batch 1

### Takeaway
Of all realisable, exact (model-preserving) sources, only two can plausibly provide a 4–16-token horizon:
- **(a) A learned cross-step sequence predictor of the full-stack activation vector (SeqMoE)**. This is the only one with published multi-token recall ≥ 0.75 on our exact models: about 87% recall@(k+3) at 8 steps.
- **(b) Draft-model or self-draft routing**. This needs per-token acceptance of about 0.95–0.98 to stay above 0.75 at W = 4–8, which holds only on very predictable text.

Layer-ahead methods top out at 2–3 layers (≤ 0.08 token). Decoupled routers give 0-token cross-token horizon unless combined with drafts. Cache-friendly routers change the model.

**Recommendation:** replicate a SeqMoE-style predictor trained on the engine's own traces. Feed its 8–16-step forecasts into the existing MIN-with-bypass admission and paced-prefetch machinery, and keep exact layer-ahead routing for the current token. Report the fraction of oracle gain recovered against W. Measure draft-routing recall against j on real generations as a second source.

### Cited Findings
Master comparison. Horizon is given as layers ahead (L) and tokens ahead (T); "0 T" means the current token only. "Exact" means the model's outputs are unchanged.

| Method (arXiv v1; venue) | Signal | Horizon | Reported accuracy (metric; model) | Overhead | Training-free? | Exact? | Batch-1 decode? | Source |
|---|---|---|---|---|---|---|---|---|
| Mixtral-offloading (28 Dec 2023) | next gate on current hidden state | 1 L | none reported | ~0 | Yes | Yes | Yes | [2312.17238](https://arxiv.org/abs/2312.17238) |
| Pre-gated MoE (23 Aug 2023; ISCA'24) | fine-tuned pre-gate | 1 L (2–3 L variants hurt accuracy) | exact by construction; Switch-Base/Large | ~0 | No (fine-tune) | No (new model) | Yes | [2308.12066](https://arxiv.org/abs/2308.12066) |
| AdapMoE (19 Aug 2024) | next-layer gate reuse | 1 L (2–3 L adaptive) | ~90% per layer; Mixtral-8x7B | small | Mostly (layer-0 predictor) | Partly (adaptive gating drops experts) | Yes (edge) | [2408.10284](https://arxiv.org/abs/2408.10284) |
| HOBBIT (Nov 2024) | stacked next-p gates | 1–3 L | top-1: 96% (1 L), ~90% (2–3 L); Mixtral-8x7B | one stacked GEMV | Yes | No (mixed precision) | Yes | [2411.01433](https://arxiv.org/abs/2411.01433) |
| ProMoE (29 Oct 2024) | learned MLP | 1–2 L | 84.7% avg; −5 pts at distance 2; gate reuse 66.9% (Qwen2-57B) | small | No | Yes | Yes | [2410.22134](https://arxiv.org/abs/2410.22134) |
| MoE-Infinity (Jan 2024) | request/iteration activation-matrix match | ≤3 L (per SeqMoE) | as "MIF" in DuoServe: Qwen3-30B-A3B exact-set ~38%, ≥ half ~82% | CPU-side matching | Yes | Yes | Yes | [2401.14361](https://arxiv.org/abs/2401.14361), [2509.07379](https://arxiv.org/abs/2509.07379) |
| ExpertFlow-He (Oct 2024; DAC'26) | transformer routing-path predictor, all layers | all L, 0 T | up to 95% (batch-level) | 7.21 MB model | No | Yes | Batched | [2410.17954](https://arxiv.org/abs/2410.17954) |
| ExpertFlow-Shen (30 Oct 2025) | token IDs + recent activations | up to ~20 L | asymptote 60–65% (predictor) vs 26–33% (gate reuse) | not quantified | No | Yes | Yes | [2510.26730](https://arxiv.org/abs/2510.26730) |
| FineMoE (7 Feb 2025; EuroSys'26) | semantic + trajectory expert maps | up to ~30 L plotted | +39% hit rate (figure-only vs distance) | similarity search per iteration | Yes (no NN training) | Yes | Serving | [2502.05370](https://arxiv.org/abs/2502.05370) |
| Fate (Feb 2025) | adjacent-layer gate input | 1 L | 97.15% prefetch accuracy (thresholded) | CPU-side | Yes | Yes | Yes (edge) | [2502.12224](https://arxiv.org/abs/2502.12224) |
| DuoServe-MoE (9 Sep 2025) | trained layer predictor | 1 L | Qwen3-30B-A3B exact top-8 55–56%, ≥ half 98% | 5–8 h trace collection | No | Yes | Yes | [2509.07379](https://arxiv.org/abs/2509.07379) |
| Pre-attention (10 Nov 2025) | 2 linear layers on pre-attention activations | <1 L (same layer) | 94.69% Qwen3-30B; 93.03% DS-V2-Lite | 0.075–0.129 ms | No | Yes | Yes | [2511.10676](https://arxiv.org/abs/2511.10676) |
| MoE-Beyond (23 Aug 2025) | transformer on token embeddings + layer ID | all L, 0 T | 97.5% exact-set / 86.6% F1; DS-V2-Lite | small | No | Yes | Yes (sim) | [2508.17137](https://arxiv.org/abs/2508.17137) |
| Speculating Experts (9 Mar 2026) | LN_{l+1}(post-attention residual) | 1 L | ~90% recall@k Qwen3-30B-A3B; estimators 83% / 88% gpt-oss-120b / 20b | negligible | Yes (estimators optional) | Optional speculative execution | Yes | [2603.19289](https://arxiv.org/abs/2603.19289) |
| PROBE (31 Jan 2026) | distilled lookahead gate | 1 L | 70–80% untrained → 87–94% top-K | off critical path | Distilled online | Yes | No (EP serving) | [2602.00509](https://arxiv.org/abs/2602.00509) |
| PILOT (Si et al., 12 Aug 2026) | cached next-layer gate | 1 L | 64.7% recall (1–84% per layer) | ~0 | Yes | Yes | Production | [2608.12103](https://arxiv.org/abs/2608.12103) |
| ST-MoE (13 Jun 2026) | correlation + history tables (cross-layer + previous token) | 1 L / 1 T (first-order) | ~85% accuracy | HW tables | Yes (online) | Yes | Yes (accelerator) | [2606.15453](https://arxiv.org/abs/2606.15453) |
| Patterns-MoE (7 Oct 2025; ISCA'26) | token- and layer-level correlation stats | 1 T (first-order) | not extracted | — | Yes | Yes | Large-scale | [2510.05497](https://arxiv.org/abs/2510.05497) |
| **SeqMoE (11 Sep 2026)** | Mamba2 over full-stack activation vectors | **1–8 T** (all L) | recall@(k+3), 1 T: 90.90% Qwen3-30B-A3B, 90.76% GPT-OSS-120B; 8 T: avg −3.58 pts | 3.46% of exec time | No (45K traces) | Yes | Yes | [2609.12978](https://arxiv.org/abs/2609.12978) |
| eMoE (10 Mar 2025) | task-aware periodic predictor | prompt-level | n/a (task similarity) | 0.24–1.3% memory | No | No (loads a subset) | Serving | [2503.06823](https://arxiv.org/abs/2503.06823) |
| SP-MoE (11 Oct 2025) | draft attention output → target gate | γ draft T (verify step) | top-1 88.94% DS-Lite, ~88% Mixtral/Phi-3.5; acceptance 97–98% (HumanEval) | draft model | Yes (needs aligned draft) | Yes (SD) | Yes | [2510.10302](https://arxiv.org/abs/2510.10302) |
| MoE-SpeQ (18 Nov 2025) | INT4 self-draft routing | k draft T × all L | 90.9% top-4 (teacher-forced); Qwen-MoE | INT4 copy of model | Yes | Yes (SD) | Yes | [2511.14102](https://arxiv.org/abs/2511.14102) |
| DraftExpert (27 Jul 2026) | self-draft + trained draft expert | draft T | prefetch hit 86–88%, acceptance 84–87%; DS-V2-Lite, Moonlight | draft expert | No | Yes (SD) | Yes (edge) | [2607.24434](https://arxiv.org/abs/2607.24434) |
| SPICE (21 Aug 2026) | aligned draft, confidence-adaptive lookahead | adaptive L | 3.12× TPOT; −3.0 to −3.5 pts GSM8K (surrogates) | draft + LoRE | Partly | No (surrogates) | Yes | [2608.21240](https://arxiv.org/abs/2608.21240) |
| Read-ME (24 Oct 2024; NeurIPS'24) | decoupled 18M-parameter pre-gating router, shared across layers | all L, 0 T (queued tokens across requests) | exact by construction; Belady hit 61.8–88.0% | 18M parameters | No (1.04B tokens) | No (refactored model) | Batched | [2410.19123](https://arxiv.org/abs/2410.19123) |
| SiDA-MoE (Oct 2023; MLSys'24) | offline LSTM hash | all L, 0 T | up to 99% (Switch) | separate thread | No | No (predicted routing and scaling replace the router) | Batched | [2310.18859](https://arxiv.org/abs/2310.18859) |
| Cache-conditional (27 Nov 2024; TMLR'25) | reroute to cached experts | n/a | >50% fewer misses, +0.1–3% perplexity | ~0 | Yes | No | Yes (phones) | [2412.00099](https://arxiv.org/abs/2412.00099) |
| Temporally Extended MoE (22 Apr 2026) | option-critic controller, LoRA | expert set persists until switch | switch rate >50% → <5%; ≤90% of base accuracy (gpt-oss-20b) | controller | No | No | Yes | [2604.20156](https://arxiv.org/abs/2604.20156) |
| Cache-Aware Router Adaptation (Sep 2026) | router trained on next-token coverage | 1 T | +1.15–18.03 points adjusted hit (Qwen3-30B-A3B), simulated | ~0 | No | No | Sim | [2609.04895](https://arxiv.org/abs/2609.04895) |

### Inferences
Feasibility for W = 4–16 tokens with recall ≥ 0.75 at batch 1 on gpt-oss-120b and Qwen3-30B-A3B (my assessment):

- **Layer-ahead family (rows 1–16): No.** The horizon is ≤ 3 layers, about 0.08 token. Even with infinite accuracy these cannot inform victim ranking at W50 = 4–15 tokens. Their role is the within-token paced prefetch the engine already has.
- **First-order cross-token (ST-MoE, Patterns-MoE, per-expert reuse features, our logistic regression): No.** Adjacent-token overlap is about 2× chance (about 1 of 8 experts for Qwen3). SeqMoE reports these as "too coarse", and Zhang's learned next-use predictor is net negative. This matches our 7–13% oracle-gain recovery.
- **SeqMoE-style sequence predictor: plausible yes, and the only published candidate.**
  - Recall@(k+3) is about 87% at 8 steps on both target models, with 3.46% time overhead at batch 1.
  - Its CUDA-graph-compatible runtime matches our engine class.
  - Risks:
    - (i) Precision is capped at 57% (gpt-oss) or 73% (Qwen3) at k+3 candidates, so admitting all candidates is wasteful. Use forecast probabilities for *ranking* (as its "probabilistic Belady" does), not for blind admission.
    - (ii) It is unreproduced, with no public code found.
    - (iii) It needs about 45K training traces per model and a domain-shift check.
    - (iv) Recall beyond 8 steps is unreported, yet W50 reaches about 15 tokens at our larger budgets.
  - Prediction: forecasts of 8–16 steps should recover well over one third of the oracle gain wherever W50 ≤ 8. That is consistent with SeqMoE's "Belady-F8 closely approaches Belady-8". This is a hypothesis to test, not a cited result.
- **Draft or self-draft routing: conditional.**
  - Teacher-forced routing fidelity is about 88–91%. Foresight at depth j decays roughly as α^j, so W = 8 at recall ≥ 0.75 needs α ≥ 0.977 per token, achievable on code or templated outputs only (SP-MoE's 97–98% is on HumanEval).
  - A quantized self-draft of gpt-oss-120b (MoE-SpeQ-style) also needs extra memory for the draft weights. That memory competes with the expert cache budget C, which is the very quantity being optimised.
  - Best use: a *hybrid*. Draft-derived routing for the first 1–3 tokens (where α^j stays high) plus a SeqMoE-style forecast for longer horizons.
- **Decoupled routers (Read-ME, SiDA, ExpertFlow-He): No for unmodified models.** Their cross-token horizon is 0 without drafts, and they need retraining or refactoring.
- **Cache-friendly routers (Skliar, StickyMoE, Temporally Extended, Cache-Aware Adaptation): out of scope for an exact-model bound.** They are worth one paragraph as the "change the model" alternative, with their quality costs (+0.1–3% perplexity; ≤ 90% accuracy retained; ≤ 1% perplexity gate failed).

What would satisfy the reviewers' bar, as a concrete experiment:
- Train a small sequence model on our own decode traces: per step, the 36×128 (gpt-oss) or 48×128 (Qwen3) binary or probability vectors.
- Use it to produce W-step forecasts.
- Plug the forecasts into the existing bounded-lookahead MIN-with-bypass and paced-prefetch oracles, replacing oracle routing.
- Report the recovered fraction of oracle gain against W in {1, 2, 4, 8, 16} and against candidate budget k+m, alongside the 7–13% logistic-regression baseline.
- Add a draft-routing curve (recall against j on real generations) as a second realisable source.

### Gaps
- No independent replication of SeqMoE. Its recall beyond 8 steps, exact top-k recall, predictor size, training cost and code availability are not established.
- No published measurement exists of draft-routing recall against draft depth on actually generated text for gpt-oss-120b or Qwen3-30B-A3B. None exists for MTP-head tokens either.
- MoE-Infinity's, Patterns-MoE's and MoE-APEX's standalone accuracies were not extracted. Lina was not retrieved.
- Several table entries rely on my reading of PDF-extracted tables (DuoServe Table III; Mixtral Table 5) and should be spot-checked against the PDFs before quoting in the paper.
