# Test-time compute on offloaded / memory-constrained MoE inference: novelty check (as of 27 Sep 2026)

Bottom line: **partially open.** The building blocks are published: batch-union growth, union cost during speculative verification, routing correlation across forks of one prompt, beam search on an offloaded MoE, and test-time scaling (TTS) on edge hardware for dense models only. What I found no paper doing is putting them together. That means measuring U(n) for n samples of the *same* prompt against the i.i.d. baseline, turning it into bytes, cache hit rate and tok/s under PCIe/SSD/CPU offload, and closing the loop to accuracy-per-second and accuracy-per-joule against a single long chain or a smaller dense model. The nearest neighbours are Hayashi et al. (19 Apr 2026: correlation across forks, no throughput) and AcceptMoE (Aug 2026: offloaded union control, but for speculative trees at batch 1). The area moves fast, so the window is weeks, not months.

## Q1. Expert activation under batching / multi-sample decoding: how does the union of experts grow with batch size, especially for samples of the same prompt?

### Takeaway
"Union of activated experts grows with batch and sets memory-bound decode latency" is well established. XShare (Feb 2026) gives the closed-form i.i.d. law, and Opportunistic Expert Activation (Nov 2025) and XShare re-route tokens to shrink the union. Union inflation during speculative verification is also taken (EcoSpec Jul 2026, AcceptMoE Aug 2026, plus XShare's spec-decoding analysis). Every batching paper I found uses *different-prompt* batches or *one sequence's draft tokens*. None measures the union across n independent samples of one prompt, and none ties it to accuracy.

### Cited Findings
**Batch-union law and memory-bound decode (datacenter, different prompts)**
- XShare (arXiv 2602.07265, Feb 2026) models expected activated experts as **E[N_a] = N(1 − (1 − k/N)^B)**.
  - For DeepSeek-R1 (N=256, k=8) it reports ~57 experts at B=8 (22%) and ~163 at B=32 (64%), and says activation reaches ~95% at B=64.
  - It states "all activated experts must be loaded from memory regardless of their utilization, making inference memory-IO-bound." — [XShare](https://arxiv.org/html/2602.07265v1)
- XShare targets different-prompt production batching and tests robustness with requests drawn "from heterogeneous datasets". — [XShare](https://arxiv.org/html/2602.07265v1)
  - Its speculative-decoding analysis finds that consecutive speculative tokens *from the same request* show 2–3× stronger expert-preference overlap than independent tokens. That is the closest published statement that intra-request tokens are correlated.
  - There is also a vLLM RFC, "Batch-Aware Expert Pruning for MoE Decode (XShare)". — [vLLM issue #35550](https://github.com/vllm-project/vllm/issues/35550)
- Opportunistic Expert Activation / batch-aware expert routing (arXiv 2511.02237, Nov 2025):
  - "MoE latency is governed by the number of activated experts". Because expert load grows more slowly than an equivalent dense layer, models "often enter a memory-bound regime even for moderate batch sizes".
  - Letting tokens "piggyback experts that have already been loaded" cuts decode latency by 39% (Qwen3-30B) and 15% (Qwen3-235B) at batch 16.
  - It does not discuss same-prompt samples. — [arXiv 2511.02237](https://arxiv.org/pdf/2511.02237)

**Throughput-oriented batched offloading (the "amortize expert streaming over a big batch" idea is old)**
- MoE-Lightning (arXiv 2411.11217, Nov 2024): CPU-GPU-I/O pipelining with a Hierarchical Roofline Model. Reports up to 10.3× throughput for Mixtral 8x7B on a single T4. — [arXiv 2411.11217](https://arxiv.org/abs/2411.11217)
- Klotski (arXiv 2502.06888, Feb 2025): expert-aware multi-batch pipeline, up to 85.12× throughput. — [arXiv 2502.06888](https://arxiv.org/abs/2502.06888)
- MoE-Gen (arXiv 2503.09716, Mar 2025): module-based batching, 8–31× over FlexGen, MoE-Lightning and DeepSpeed. At batch 1 it "defaults to on-demand fetching". — [arXiv 2503.09716](https://arxiv.org/abs/2503.09716)
- These summaries come from a sibling note in this repo. I did not re-read the papers this session.

**Beam search on an offloaded MoE (closest existing systems data point)**
- Fiddler (arXiv 2402.07033, Feb 2024; ICLR 2025):
  - Evaluates beam widths 4, 8, 12 and 16 (input 32, output 64 tokens) on a Quadro RTX 6000 (24 GB) and an RTX 6000 Ada (48 GB).
  - Reports 11.57× over llama.cpp on average, and notes "multiple inputs can be processed simultaneously, even for a single request".
  - Compared only against llama.cpp because other baselines "do not support beam search".
  - Does **not** report unique experts across beams, cache behaviour, or accuracy. — [Fiddler v3](https://arxiv.org/html/2402.07033v3)

**Union inflation during speculative verification (single request; taken)**
- AcceptMoE (arXiv 2608.02989, Aug 2026): the verifier-tree expert union "can substantially exceed" per-token top-k.
  - Cites a system with "74.7% fewer verified tokens than EAGLE-3 but only 32.5% fewer activated experts" on Qwen3-30B.
  - Proposes self-sizing verifier expert sets, with "residency-aware pruning" under offload.
  - Results: 2.06× throughput under offloading on an **RTX 5090** and a 73.6–77.1% cut in host-to-device traffic, at batch size one, with Qwen3-30B, Qwen3-Coder-30B and GPT-OSS-120B. — [AcceptMoE](https://arxiv.org/html/2608.02989)
- EcoSpec (arXiv 2607.12696, 14 Jul 2026) names "expert scattering": high-confidence draft tokens route to disjoint experts and inflate the verification union. — [EcoSpec](https://arxiv.org/html/2607.12696)
  - This is from the sibling note numerics_and_prediction.md. The same note records MoE-SpeQ stating that verification must "load and compute the union of all experts activated across all k tokens".

### Inferences
- **The i.i.d. null model shows how much correlation has to buy.** Plugging the three target models into XShare's formula (my arithmetic; see the table below): best-of-8 on an offloaded MoE amortizes expert bytes per sample-token by only ~1.2–1.5×. Best-of-16 reaches ~1.55–2.3×, and n≈32–64 saturates to "stream the whole layer every step".
  - Any measured sublinearity beyond the i.i.d. curve comes from same-prompt correlation, and that is the new quantity.

| Model (N experts, top-k) | U(8) i.i.d. (share of experts) | per-sample amortization at n=8 | U(16) i.i.d. (share of experts) | per-sample amortization at n=16 |
|---|---|---|---|---|
| OLMoE (64, 8) | 42.0 (66%) | 1.52× | 56.4 (88%) | 2.27× |
| Qwen3-30B-A3B (128, 8) | 51.6 (40%) | 1.24× | 82.4 (64%) | 1.55× |
| gpt-oss-20b (32, 4) | 21.0 (66%) | 1.52× | 28.2 (88%) | 2.27× |

- The two ends of the curve are clean bounds: U = k (perfectly correlated samples) and U = N (saturation).
  - At saturation, offloaded best-of-n reduces to FlexGen/MoE-Lightning-style whole-layer streaming, amortized over n.
  - That gives a time-domain speed-of-light bound of the form t_step(n) ≥ max(bytes(U(n) \ resident)/BW, compute(n)). This fits the researcher's existing model directly.
- Prior batching work optimizes *throughput over unrelated prompts* or re-routes tokens, which is lossy (XShare, Opportunistic Expert Activation). No paper treats n as an accuracy knob whose cost is set by U(n).

### Gaps
- I did not re-read the full texts of MoE-Lightning, MoE-Gen, Klotski, Lina, ExFlow or DuoServe. I cannot rule out a figure somewhere that plots unique experts versus batch for same-prompt batches. Nothing in the summaries or search snippets suggested one.
- I could not see the AcceptMoE PDF body, so I could not check whether it ever varies batch or number of drafts beyond batch 1.
- Semantic Scholar was not queried directly; web search was the proxy.

## Q2. Test-time scaling / inference-time compute on edge or consumer hardware, with MoE; hardware-cost-aware TTS laws

### Takeaway
Hardware-aware TTS exists in two forms: memory-aware scaling laws on datacenter GPUs (Kinetics, Jun 2025) and best-of-N / beam search on phone NPUs with *dense* models (EuroSys 2026). Neither models MoE expert streaming under offload. I found no paper that gives accuracy-per-second or accuracy-per-joule for best-of-n on an offloaded MoE, nor one comparing it with a single long chain or a smaller dense model on consumer hardware.

### Cited Findings
- **Kinetics: Rethinking Test-Time Scaling Laws** (Sadhukhan, Chen, Zheng, Zhou, Strubell, Beidi Chen; CMU; arXiv 2506.05333, Jun 2025; v3 exists):
  - Adds memory access to the cost model and finds "attention, rather than parameter count, emerges as the dominant cost factor" in TTS.
  - Says this is "further exacerbated by MoE architectures…which reduce active parameter count without alleviating attention overhead".
  - Covers models from 0.6B to 32B (Qwen3).
  - My fetch found nothing on offloading, consumer hardware, or how expert weights are amortized across best-of-N samples. — [Kinetics arXiv](https://arxiv.org/html/2506.05333v3); [project page](https://infini-ai-lab.github.io/Kinetics/)
- **Scaling LLM Test-Time Compute with Mobile NPU on Smartphones** (Hao, Wei et al.; Tsinghua and MSR; arXiv 2509.23324, Sep 2025; EuroSys '26, Apr 2026):
  - Best-of-N and step-level beam search with reward models, on Qwen2.5 1.5B/3B/7B and Llama 3.2 1B/3B.
  - Hardware: Snapdragon NPUs (V73/V75/V79).
  - Core argument: decode "degenerates into GEMV", so the matrix unit's idle rows absorb extra samples almost for free.
  - Smaller model + TTS matched or beat larger baselines, at under 5 W during decode.
  - **No MoE.** — [arXiv 2509.23324](https://arxiv.org/pdf/2509.23324); [ACM DL](https://dl.acm.org/doi/abs/10.1145/3767295.3769382)
  - A related workshop paper exists: "WiP: From Wasted Compute to Quality Gains: LLM Test-Time Scaling on Mobile NPUs". — [ACM DL](https://dl.acm.org/doi/10.1145/3737902.3768361)
- **MoE-specific TTS algorithms** (accuracy side, not systems):
  - "MoEs Are Stronger than You Think: Hyper-Parallel Inference Scaling with RoE" (arXiv 2509.17238, Sep 2025). Stochastic routing turns one MoE into "a dynamic ensemble", with "an efficient batching strategy and a specialized KV-caching mechanism". It claims a 7B MoE matches a 10.5B MoE with 30% less compute. No offloading or consumer hardware in the abstract. — [arXiv 2509.17238](https://arxiv.org/abs/2509.17238)
  - "Certain Head, Uncertain Tail: Expert-Sample for Test-Time Scaling" (arXiv 2602.02443, Feb 2026). Title and search listing only; not read. — [arXiv 2602.02443](https://arxiv.org/html/2602.02443v1)
- **Energy and local-AI efficiency framing** exists, but at the platform or datacenter level, not offloaded-MoE TTS:
  - "Intelligence per Watt: Measuring Intelligence Efficiency of Local AI" (arXiv 2511.07885, Nov 2025). — [arXiv](https://arxiv.org/html/2511.07885v6)
  - "Energy use of AI inference, efficiency pathways, and test-time scaling" (Joule, 2026). — [Joule](https://www.cell.com/joule/fulltext/S2542-4351(26)00114-5)
  - I saw titles and snippets only; neither was read in depth.

### Inferences
- Kinetics' conclusion that attention and KV dominate TTS cost is a *datacenter HBM* conclusion. On an offloaded MoE, bytes per step are dominated by non-resident expert weights over PCIe/SSD, often an order of magnitude slower than HBM, while KV for n short samples stays on the GPU.
  - The offload regime could therefore *reverse* Kinetics' ranking: parallel sampling amortizes the dominant cost term, where a single long chain does not.
  - This is a crisp, testable hypothesis, and I found no paper that tests it.
- The phone-NPU paper supplies the dense analogue ("idle compute absorbs samples"). The MoE-offload analogue is "idle PCIe bytes absorb samples up to U(n)". The asymmetry is that for MoE the extra samples *do* add bytes, through union growth, so the free-lunch argument holds only as far as correlation reaches.

### Gaps
- I did not verify whether Kinetics v3 includes a best-of-N cost term that shares weight loads across samples. The fetch returned no such detail.
- I did not verify whether "Intelligence per Watt" includes MoE models or parallel sampling.
- I found no source for "test-time scaling on the edge with MoE". Searches for that phrase returned edge-MoE offloading systems (SMoE, OD-MoE, MobileMoE, CoMoE) with no TTS component.

## Q3. Has anyone measured correlation of expert routing across parallel samples or beams of the same prompt?

### Takeaway
**Yes, partly, in one paper:** Hayashi et al. (19 Apr 2026). It measured strong routing overlap between sibling forks of one shared-prefix prompt, for one model and one code task. The authors explicitly did not turn this into offloading throughput, and they report no union-vs-n curve, cache hit rates, tok/s or energy. RAD (22 Jun 2026) uses routing agreement across rollouts as an *answer selector*, which is accuracy only. The systems question is still open.

### Cited Findings
- **Hayashi, Mukunoki, Hoshino, Katagiri (Nagoya U.), "Layer-wise MoE Routing Locality under Shared-Prefix Code Generation: Token-Identity Decomposition and Compile-Equivalent Fork Redundancy"**, arXiv 2604.17182, submitted **19 Apr 2026**, cs.SE/cs.AI. — [arXiv abs](https://arxiv.org/abs/2604.17182); [HTML](https://arxiv.org/html/2604.17182); [Pith (date)](https://pith.science/paper/2604.17182)
  - Setup:
    - Tree-search branching generation from a single C sorting-function prompt, with forks where top-30 cumulative probability ≤ 0.40.
    - 5,745 forks produced 851 completed codes, a branching ratio of about 6.
    - Model: Qwen3.5-35B-A3B-FP8 (256 experts, top-8, 40 layers).
    - Routing captured with SGLang `--enable-return-routed-experts`.
  - Metric: Jaccard similarity of per-token expert sets between sibling forks (random baseline ≈ 0.016).
  - Results:
    - All-layer Jaccard: same token 0.649 (40× random); different token 0.179 (11× random).
    - Different-token overlap across non-equivalent code groups is still 0.175.
    - 200 steps after the fork: same-token 0.294 and different-token 0.086.
    - Layer pattern: L0 is dominated by token identity (same 0.828, different 0.087). Middle layers L14–20 are context-dependent (different-token 0.223, 14× random).
  - Systems claims:
    - Suggests keeping a small set of middle-layer experts GPU-resident is "particularly effective for parallel generation from shared prefix".
    - States as a limitation: "No verification performed translating routing locality into actual offloading throughput."
    - Other limitations: single model, single task, and a thinking-skip condition.
  - Also finds 67% of compiled outputs fall into 3 assembly-equivalent groups, and that "99.6% of within-group differences consist of comments and blank lines". In other words, much sample diversity is redundant.
- **Nian et al., "Does the Same Token Mean the Same State? MoE Routing as Signal for Reasoning Control"** (RAD), arXiv 2606.22798, **22 Jun 2026**. — [Pith](https://pith.science/paper/2606.22798)
  - Represents each rollout by its MoE routing states at anchor windows and selects outputs by routing agreement: 73.9% versus 73.6% for majority voting across 10 MoE models and 6 datasets.
  - Improves best-of-16 SWE-bench patch selection.
  - Nothing on offload cost.
- **Within-sequence (temporal) routing consistency** is well studied. For example, "Not All Models Suit Expert Offloading: On Local Routing Consistency of MoE Models" (arXiv 2505.16056, May 2025; ICLR 2026) is the single-chain baseline the cross-sample question must be compared against. — [arXiv 2505.16056](https://arxiv.org/abs/2505.16056); [ML Anthology](https://mlanthology.org/iclr/2026/liang2026iclr-all/)
- XShare's 2–3× overlap for consecutive speculative tokens of one request is intra-request correlation. It is along the draft axis, not across independent samples. — [XShare](https://arxiv.org/html/2602.07265v1)

### Inferences
- Hayashi et al. removes the claim "nobody knows whether same-prompt samples share experts". A new paper cannot sell the correlation itself as the discovery; it has to cite them.
- What they leave open fits the researcher's assets exactly:
  - (a) U(n) growth curves against the i.i.d. law and the k lower bound, per layer and over decode steps;
  - (b) 3 models (OLMoE, Qwen3-30B-A3B, gpt-oss-20b) and reasoning/math tasks, not one code prompt;
  - (c) turning U(n) into bytes/step, cache hit rate at fixed residency (union/capacity ratio r̄), and tok/s under a validated bytes-over-bandwidth model;
  - (d) accuracy-per-second and accuracy-per-joule frontiers.
- Their Jaccard decay with distance from the fork (0.294 → 0.086 at 200 steps) implies U(n) *rises over the generation*: samples decorrelate. For long reasoning chains the per-step union may approach i.i.d. late in decoding. That is a result worth measuring, because it decides whether best-of-n on an offloaded MoE pays off for long chains.

### Gaps
- Hayashi et al. publishes Jaccard for *pairs*. Union growth for n > 2 cannot be derived from their numbers.
- I found no study of beam search or MCTS specifically, as opposed to i.i.d. sampling or tree forks, measuring expert overlap. Fiddler ran beams without reporting it.

## Q4. Verdict (open / partially open / taken), exact remaining slice, and feasibility by 27 Oct 2026

### Takeaway
**Partially open.** Taken: the batch-union law, union inflation in speculative verification (including under consumer-GPU offload), beam-search throughput on an offloaded MoE, same-prompt routing correlation (one model, one task), and hardware-aware TTS for dense and datacenter cases. Open: the *offload-regime TTS cost model and frontier*. That covers measured same-prompt U(n) across models and tasks, what it does to cache hit rate and tok/s on PCIe/SSD/CPU-hybrid consumer boxes, and when best-of-n beats a single long chain or a smaller dense model per second or per joule. It is feasible by 27 Oct only as a tightly scoped trace-plus-model paper with light GPU validation.

### Cited Findings
**Closest prior art, ranked by overlap (with dates)**
1. Hayashi et al., arXiv 2604.17182 (19 Apr 2026): same-prompt fork routing correlation; no throughput. — [arXiv](https://arxiv.org/abs/2604.17182)
2. AcceptMoE, arXiv 2608.02989 (Aug 2026): union control under offload on an RTX 5090, but speculative-tree verification at batch 1, not TTS. — [arXiv](https://arxiv.org/html/2608.02989)
3. XShare, arXiv 2602.07265 (Feb 2026), and Opportunistic Expert Activation, arXiv 2511.02237 (Nov 2025): the batch-union law and batch-aware re-routing, for different-prompt serving. — [XShare](https://arxiv.org/html/2602.07265v1); [OEA](https://arxiv.org/pdf/2511.02237)
4. Fiddler, arXiv 2402.07033 (Feb 2024; ICLR 2025): beam search 4–16 on an offloaded MoE; throughput only. — [Fiddler](https://arxiv.org/html/2402.07033v3)
5. Kinetics, arXiv 2506.05333 (Jun 2025): memory-aware TTS laws, datacenter, MoE mentioned without offload. — [Kinetics](https://arxiv.org/abs/2506.05333)
6. Mobile-NPU TTS, arXiv 2509.23324 (Sep 2025; EuroSys '26): edge TTS, dense only. — [arXiv](https://arxiv.org/pdf/2509.23324)
7. EcoSpec, arXiv 2607.12696 (14 Jul 2026): expert scattering in verification. — [arXiv](https://arxiv.org/html/2607.12696)
8. RAD, arXiv 2606.22798 (22 Jun 2026): cross-rollout routing for answer selection. — [Pith](https://pith.science/paper/2606.22798)

**Tooling note:** SGLang exposes per-token routed experts through `--enable-return-routed-experts`, which Hayashi et al. used. That means multi-sample routing traces for Qwen3-class models can be captured on a cloud GPU without custom instrumentation. — [Hayashi et al.](https://arxiv.org/html/2604.17182)

### Inferences
**The exact remaining slice.** A defensible claim would be the first offload-regime cost model and measurement of parallel test-time compute for MoE. It has five parts:
1. **Measured U(n, t, layer)** for n ∈ {1, 2, 4, 8, 16, 32} same-prompt samples, bracketed by the lower bound k, the i.i.d. law N(1 − (1 − k/N)^n) and saturation N. Include how U grows with distance from the shared prefix, and how it differs between i.i.d. sampling, beam search and tree search.
2. **Cache consequences.** Hit rate at fixed residency, versus union/capacity r̄, for n samples against one chain. A sibling note records that 2608.07911 calls for reporting exactly this ratio.
3. **Speed-of-light tok/s per sample and per prompt** from the bytes-over-bandwidth model, for PCIe 4/5, SSD and CPU-hybrid (llama.cpp-style) placements, validated on the A10 and a cloud GPU.
4. **Accuracy-vs-time and accuracy-vs-energy frontiers.** Best-of-n / self-consistency on the offloaded MoE, against a single long chain of equal wall-clock, against a smaller dense model that fits in VRAM, and against the same MoE fully resident.
5. **The crossover n\*** where amortization beats the union and cache-thrash penalty. Also test whether the Kinetics ranking (long chain vs parallel) flips in the offload regime.

**What is not novel and must be cited, not claimed:**
- that union grows with batch;
- that union hurts speculative verification;
- that same-prompt forks share experts (Hayashi);
- that beam search works on offloaded MoE (Fiddler);
- that edge decode has spare capacity for TTS (EuroSys '26 NPU).

**Feasibility by 27 Oct 2026 (about 4 weeks):** tight but plausible for a focused workshop or short paper.
- The GPU-free core fits the existing assets:
  - Generate n-sample corpora with routing capture for OLMoE (A10) and Qwen3-30B-A3B / gpt-oss-20b (cloud GPU, SGLang or llama.cpp instrumentation) on GSM8K/MATH500-class prompts.
  - Compute U(n) curves and cache replays offline.
  - Push them through the validated bytes/bandwidth model.
- Accuracy numbers for best-of-n do not depend on offloading, so they come from the same generation runs. Majority vote needs no reward model.
- The risk items:
  - (i) energy measurement on consumer hardware. Model it, or measure it only on the A10.
  - (ii) validating batched offloaded decode tok/s at several n. llama.cpp's parallel slots with CPU-resident experts is the cheapest path.
  - (iii) beam search and MCTS variants. Drop them to future work if time runs short.
- **Scoop risk is high.** EcoSpec (Jul), AcceptMoE (Aug) and Hayashi (Apr) all appeared within six months, and AcceptMoE's authors already have offloaded union control on an RTX 5090. Extending that to best-of-n is a small step for them.

### Gaps
- ICLR 2027 and MLSys 2027 submissions (deadlines around late Sep and late Oct 2026) are not yet public. A concurrent submission on exactly this slice cannot be ruled out.
- I did not verify the content of the SPCL "Performance Foundations of Parallel & Distributed RLMs" (arXiv 2608.27046), per instructions. If it contains a MoE-offload term it would move the verdict toward "taken" for the cost-model part.
- I did not find, and so cannot cite, any work on self-consistency or best-of-n accuracy-per-joule on consumer GPUs with MoE. That absence rests on web search and keyword queries, not an exhaustive Semantic Scholar crawl.
