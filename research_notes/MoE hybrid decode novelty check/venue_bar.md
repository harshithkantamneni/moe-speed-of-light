# Publication bar for MoE / LLM-inference offloading systems papers at top venues (as of 27 Sept 2026)

Scope: where comparable MoE-offloading / CPU-GPU hybrid inference papers were published, what their evaluations contained, what the venues' CFPs require, deadlines from late 2026 on, SPCL/Hoefler methodology preferences, the ETH AI Center fellowship requirements, and whether any systems work has been pre-registered. The work being assessed (single author, one A10 VM, llama.cpp patch, 4 models, 1.41-1.81x over `--n-cpu-moe`, bound + model within 2.9%) is used only as the yardstick in the Inferences sections.

## Q1. Which MoE-offloading / hybrid-inference papers appeared at which venues, and how broad were their evaluations?

### Takeaway
Every comparable paper at a top systems venue from 2025-2026 (KTransformers at SOSP'25, MoE-Lightning at ASPLOS'25, FineMoE at EuroSys'26, MoE-APEX at ASPLOS'26, Pipelined Sharding at MLSys'26 as an oral, HybriMoE at DAC'25, Fiddler at ICLR'25) used **at least 2 hardware configurations, usually 3 models, and 2-4 named system baselines, with llama.cpp almost always one of them**. The closest precedent is the NVIDIA-authored MLSys 2026 oral "Efficient, VRAM-Constrained xLM Inference on Clients" (Pipelined Sharding). It is **built on llama.cpp, uses llama.cpp's `-cmoe` CPU-MoE offload as its baseline, runs Qwen3-30B/235B MoE on 3 client platforms, and reports an average 3.7x decode TPS gain**. Reviewers will read any new llama.cpp MoE-offload paper against it.

### Cited Findings

**KTransformers: SOSP 2025 (the current SOTA reference point)**
- Venue: Proceedings of ACM SIGOPS 31st SOSP (2025), DOI 10.1145/3731569.3764843. Authors are from Tsinghua (MADSys) and Approaching.AI. — [ACM DL](https://dl.acm.org/doi/10.1145/3731569.3764843); [MADSys page](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/)
- Author list: 15 authors across 5-6 institutions (Tsinghua, Approaching.AI, Hangzhou Dianzi, UESTC, BUPT, BIT). — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- Hardware: **one host** (dual-socket Intel Xeon Platinum 8452Y, 36 cores/socket with AMX, 1 TB DDR5 per socket, 220 GB/s intra-socket, 125 GB/s cross-socket) with **two GPU configurations**: an A100 40 GB for full-precision runs and an RTX 4080 16 GB for quantized runs, both over PCIe 4.0. — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- Models: DeepSeek-V3 671B, DeepSeek-V2.5 236B and Qwen2-57B-A14B, in BF16/FP16 on the A100 and Int4 (V3) or Int8 (V2.5, Qwen2) on the 4080. — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- Baselines: **Fiddler and llama.cpp**. The llama.cpp baseline was "extended with expert-level offloading for fair comparison". — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- Metrics: prefill throughput over prompts of 32-8,192 tokens, and decode throughput with a 32-token prompt and up to 512 output tokens, all at **batch size 1**. — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- Headline results: prefill 4.62-19.74x, decode 1.25-4.09x without Expert Deferral and 1.66-4.90x with it. Expert Deferral alone adds up to 1.45x. — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- Accuracy evaluation: needed because Expert Deferral is approximate. It ran HumanEval, MBPP, GSM8K, StrategyQA and LiveBench, with an average accuracy drop of at most 0.5%. — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- Ablations: a per-optimization breakdown covering AMX/AVX-512 kernel choice, dynamic scheduling (up to 1.83x prefill), NUMA-aware tensor parallelism (up to 1.63x decode) and CUDA Graphs (up to 1.23x decode). — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- SGLang integration: KTransformers CPU kernels were integrated into SGLang for hybrid inference, announced 22 Oct 2025. "SGLang offload" is therefore now effectively KTransformers kernels. — [LMSYS blog](https://www.lmsys.org/blog/2025-10-22-KTransformers/); [SGLang issue #11425](https://github.com/sgl-project/sglang/issues/11425)
- Other SOSP 2025 MoE papers: KTransformers is the only MoE-inference paper at SOSP 2025 in a curated per-venue list. Other LLM-inference papers there (Jenga, PrefillOnly, IC-Cache, Aegaeon) are not about MoE offloading. — [Awesome Papers SOSP 2025](https://paper.lingyunyang.com/reading-notes/conference/sosp-2025)

**Efficient, VRAM-Constrained xLM Inference on Clients ("Pipelined Sharding"): MLSys 2026 oral (the closest precedent)**
- Venue and authors: an MLSys 2026 oral by Aditya Ukarande, Deep Shekhar, Marc Blackstein and Ram Rangan. — [MLSys 2026 oral page](https://mlsys.org/virtual/2026/oral/3802); [arXiv 2604.26334](https://arxiv.org/abs/2604.26334)
- Implementation: built on **llama.cpp** (branch 6097) and open-sourced. The authors are "working to upstream" it to llama.cpp. They had a public artifact-evaluation repo and Zenodo records. — [arXiv HTML](https://arxiv.org/html/2604.26334v1); [AE repo](https://github.com/deepshnv/pipeshard-mlsys26-ae); [Zenodo](https://zenodo.org/records/19436383)
- Hardware: **3 client platforms**: a laptop with an RTX 3500 (12 GB, PCIe Gen3), a desktop with an RTX 5070 Ti (16 GB, PCIe Gen5) and a high-end desktop with an RTX 5090 (32 GB) and a 16-core EPYC (PCIe Gen5). — [arXiv HTML](https://arxiv.org/html/2604.26334v1)
- Models: dense Mistral-NeMo-Minitron 4B/8B, **MoE Qwen3 30B and 235B**, and the Nemotron Vision 4B and Cosmos-Reason1 vision-language models. — [arXiv HTML](https://arxiv.org/html/2604.26334v1)
- Baselines: a manually tuned "llama-cpp-baseline" that includes the `-cmoe` and `-kvo` CPU-offload flags, plus vLLM for the vision-language models. — [arXiv HTML](https://arxiv.org/html/2604.26334v1)
- Results: at batch 1, TTFT improves 2x on average (up to 6.7x), TPS 3.7x on average (up to 30x) and end-to-end latency 2x on average (up to 4.3x). Batched throughput improves up to 8.2x. — [arXiv HTML](https://arxiv.org/html/2604.26334v1)
- Scheduling method: "benchmark-profile-guided" scheduling that picks GPU-only, static or dynamic strategies using roofline analysis and profiled kernel timings. — [arXiv HTML](https://arxiv.org/html/2604.26334v1)

**MoE-Lightning: ASPLOS 2025**
- Venue and authors: ASPLOS '25, DOI 10.1145/3669940.3707267, by Cao, Liu, Griggs, Schafhalter, Liu, Sheng, Gonzalez, Zaharia and Stoica (a Berkeley Sky/Stanford group). — [ACM DL](https://dl.acm.org/doi/10.1145/3669940.3707267); [arXiv 2411.11217](https://arxiv.org/abs/2411.11217)
- Hardware: NVIDIA T4 16 GB, both single-GPU and 2-4 T4. Models: Mixtral 8x7B, Mixtral 8x22B and DBRX. — [arXiv 2411.11217](https://arxiv.org/abs/2411.11217)
- Performance model: uses a **Hierarchical Roofline Model (HRM)** to choose policies, i.e. an analytical bandwidth model similar to the assessed work's bytes-over-bandwidth model. — [arXiv 2411.11217](https://arxiv.org/abs/2411.11217)
- Results: up to 10.3x throughput over SOTA offloading systems on a single T4 for Mixtral 8x7B. It is throughput-oriented batch inference, not batch-1 decode. — [arXiv 2411.11217](https://arxiv.org/abs/2411.11217)

**HybriMoE: DAC 2025**
- Venue and group: 62nd DAC (2025), from Meng Li's group at PKU. Code at PKU-SEC-Lab/HybriMoE. — [ACM DL](https://dl.acm.org/doi/10.1109/DAC63849.2025.11133274); [GitHub](https://github.com/PKU-SEC-Lab/HybriMoE)
- Approach and results: implemented on top of kTransformers and evaluated on 3 MoE LLMs. Average speedups are 1.33x prefill and **1.70x decode** over SOTA hybrid MoE frameworks. — [arXiv 2504.05897](https://arxiv.org/abs/2504.05897)

**Fiddler: ICLR 2025 (an ML venue, not a systems venue)**
- Authors: 5 (4 from the University of Washington, 1 from Tsinghua). — [ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/8cd1ce03ea58b3d7dfd809e4d42f08ea-Paper-Conference.pdf)
- Hardware: **2 environments**: a Quadro RTX 6000 24 GB with a Xeon Gold 6126 over PCIe Gen3, and an RTX 6000 Ada with a Xeon Platinum 8480+ over PCIe Gen4. — [ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/8cd1ce03ea58b3d7dfd809e4d42f08ea-Paper-Conference.pdf)
- Models: Mixtral-8x7B in 16-bit, with Phi-3.5-MoE in an appendix. — [ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/8cd1ce03ea58b3d7dfd809e4d42f08ea-Paper-Conference.pdf)
- Baselines: DeepSpeed-MII (ZeRO-Infinity), Mixtral-Offloading and llama.cpp b2956. — [ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/8cd1ce03ea58b3d7dfd809e4d42f08ea-Paper-Conference.pdf)
- Results: **1.26x for single-batch inference**, 1.30x for long prefill and 11.57x for beam search. — [ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/8cd1ce03ea58b3d7dfd809e4d42f08ea-Paper-Conference.pdf)

**FineMoE ("Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading", the fMoE arXiv paper): EuroSys 2026**
- Venue: EuroSys 2026, DOI 10.1145/3767295.3769319. — [ACM DL](https://dl.acm.org/doi/10.1145/3767295.3769319); [GitHub](https://github.com/IntelliSys-Lab/FineMoE-EuroSys26)
- Authors: 5, from Stevens, Rice, Waterloo and Rutgers. — [EuroSys'26 PDF](https://intellisys.haow.us/assets/pdf/Hanfei_FineMoE_EuroSys26.pdf)
- Hardware: **2 testbeds**: 6x RTX 3090 (NVLink, PCIe 4.0) with a Threadripper PRO 3955WX and 480 GB RAM, plus an A100 80 GB. — [EuroSys'26 PDF](https://intellisys.haow.us/assets/pdf/Hanfei_FineMoE_EuroSys26.pdf)
- Models and workloads: Mixtral-8x7B, Qwen1.5-MoE and Phi-3.5-MoE, driven by LMSYS-Chat-1M, ShareGPT and Azure LLM inference traces. — [EuroSys'26 PDF](https://intellisys.haow.us/assets/pdf/Hanfei_FineMoE_EuroSys26.pdf)
- Baselines: DeepSpeed-Inference, Mixtral-Offloading, a reproduced ProMoE and MoE-Infinity. — [EuroSys'26 PDF](https://intellisys.haow.us/assets/pdf/Hanfei_FineMoE_EuroSys26.pdf)
- Metrics and ablations: TTFT, TPOT and expert hit rate. Ablations cover tracking variants, caching policy (LRU/LFU), prefetch distance 1-8, store capacity and **batch size 1-8**. — [EuroSys'26 PDF](https://intellisys.haow.us/assets/pdf/Hanfei_FineMoE_EuroSys26.pdf)
- Results: 47% lower latency and 39% better hit rate. TPOT is 27% lower than MoE-Infinity and 46% lower than DeepSpeed. — [EuroSys'26 PDF](https://intellisys.haow.us/assets/pdf/Hanfei_FineMoE_EuroSys26.pdf)

**MoE-APEX: ASPLOS 2026**
- Venue: "MoE-APEX: An Efficient MoE Inference System with Adaptive Precision Expert Offloading", ASPLOS '26 Vol. 2, DOI 10.1145/3779212.3790187. The ACM DL page returned 403, so its evaluation details were not retrieved. — [ACM DL](https://dl.acm.org/doi/10.1145/3779212.3790187); [ResearchGate](https://www.researchgate.net/publication/403012348_MoE-APEX_An_Efficient_MoE_Inference_System_with_Adaptive_Precision_Expert_Offloading)

**Other MoE papers at MLSys 2026 (mostly datacenter serving, not offloading)**
- Titles: CRAFT (expert replication), LYNX (expert remapping), BLAZE, Cascade (speculative decoding for MoE), "Demystifying the Mixture of Experts Serving Tax" (a characterization paper), "On the Diminishing Returns of Expert Load Balancing in MoE LLM Serving", "From Tokens to Layers ... Layered Prefill", and FarSkip-Collective. — [MLSys 2026 papers](https://mlsys.org/virtual/2026/papers.html?filter=titles)
- Adjacent client and heterogeneous-hardware inference papers: "SD-HC: Heterogeneous Functional Pipelining for Speculative LLM Decoding on AI PCs", "REMIX: Dynamic Partitioning for Fine-Grained Heterogeneous LLM Serving" and "SuperInfer ... on Superchips". — [MLSys 2026 papers](https://mlsys.org/virtual/2026/papers.html?filter=titles)

**EuroSys 2026 and ATC 2025**
- EuroSys 2026: MoE-relevant papers are FineMoE (inference offloading) and MegaScale-MoE (training). On-device LLM papers also appeared (mobile NPU test-time compute, TZ-LLM). — [Awesome Papers EuroSys 2026](https://paper.lingyunyang.com/reading-notes/conference/eurosys-2026)
- ATC 2025: **no MoE-inference offloading paper** in the curated list. The nearest were PopFetcher (MoE *training* prefetch), Toppings (CPU-assisted adapter serving), Weaver (attention offloading), CLONE (edge) and Torpor (serverless GPU swapping). — [Awesome Papers ATC 2025](https://paper.lingyunyang.com/reading-notes/conference/atc-2025)

**Older precedents, as listed by the awesome-moe-inference list**
- Venues listed there: Pre-gated MoE (ISCA'24), APTMoE (SC'24), SiDA (MLSys'24), SwapMoE (ACL'24), Read-ME (NeurIPS'24) and MoE-Lightning (ASPLOS'25). Many offloading papers are still listed as arXiv-only there, including MoE-Infinity, ProMoE, HOBBIT, ExpertFlow, DAOP and MoE-Gen. — [awesome-moe-inference](https://github.com/MoE-Inf/awesome-moe-inference/)
- DAOP: now appears as an IEEE conference publication on IEEE Xplore (document 10992741). — [IEEE Xplore](https://ieeexplore.ieee.org/document/10992741/); [GitHub](https://github.com/ecolab-nus/DAOP)

**Workshops**
- EuroMLSys: publishes archival ACM DL proceedings; the 5th edition (2025) is 10.1145/3721146. EuroMLSys 2026 had an "LLM Inference, Memory, agent" session. — [EuroMLSys 2025 proceedings](https://dl.acm.org/doi/proceedings/10.1145/3721146); [EuroMLSys site](https://euromlsys.eu/)
- An arXiv paper (2512.16473, elsa-lab) proposes a CPU-GPU collaborative MoE framework with an **expert cache** for consumer hardware. No venue is stated on its arXiv page. — [arXiv 2512.16473](https://arxiv.org/abs/2512.16473)

**Engineering prior art inside the frameworks the work patches (titles from search results only; these pages were not fetched)**
- llama.cpp discussion #24528: "RFC: MoE expert cache, VRAM caching of hot CPU-resident experts with hybrid hit/miss execution". — [llama.cpp discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)
- llama.cpp issue #20757: "Feature Request: Two-tier GPU+RAM expert cache for MoE offload (pluggable eviction policy)". — [llama.cpp issue #20757](https://github.com/ggml-org/llama.cpp/issues/20757)
- jklincn/llama.moe: "Optimized inference of MoE models based on llama.cpp, with dynamic expert offloading". — [GitHub llama.moe](https://github.com/jklincn/llama.moe)
- vLLM RFC #38256: "Incremental MoE Expert Offloading — GPU Cache + Async Pipeline". — [vLLM issue #38256](https://github.com/vllm-project/vllm/issues/38256)

### Inferences
- **Hardware breadth:** accepted top-venue papers used 2 GPU configurations (KTransformers, Fiddler, FineMoE), 1-4 of the same GPU (MoE-Lightning) or 3 platforms (Pipelined Sharding). **One A10 VM is below every one of them.** A single platform is the most likely "limited evaluation" reject reason at EuroSys, ASPLOS, OSDI or MLSys.
- **Model breadth:** 4 models (gpt-oss-20b/120b, Qwen3-30B-A3B at Q4 and Q8) is at or above the norm; comparable papers used 1-3 models. Q4 and Q8 of the same model count as one architecture, so the effective count is 3 architectures. That is fine, but reviewers will note the absence of the DeepSeek-V2/V3-class, many-small-experts models that KTransformers evaluated.
- **Baselines:** llama.cpp alone is weaker than every comparable paper. KTransformers used 2 baselines, Fiddler 3, FineMoE 4, and Pipelined Sharding a *tuned* llama.cpp plus vLLM. A 2027 reviewer will expect KTransformers (or SGLang with the KTransformers kernels) and, for llama.cpp-based work, Pipelined Sharding (MLSys'26), because both claim results on Qwen3-30B-class MoE against `-cmoe`.
- **Speedup magnitude:** 1.41-1.81x decode is within the accepted range. Fiddler reached ICLR with 1.26x single-batch, HybriMoE reached DAC with 1.70x average decode, and KTransformers' decode-only gain was 1.25-4.09x. The number alone is not disqualifying.
- **Positioning against Pipelined Sharding:** it reports an *average* 3.7x TPS over tuned llama.cpp offload on a similar model class. The paper must explain why its 1.41-1.81x is not dominated by Pipelined Sharding, for example through different baselines, different VRAM budgets, or the two being orthogonal and composable, or it risks being judged incremental.
- **Differentiator:** the validated speed-of-light bound (Belady plus bandwidth), together with predictions within 2.9% median error and a "% of bound reached" metric, is the genuinely distinctive element. MoE-Lightning (HRM) and Pipelined Sharding (roofline) use models to *choose policies*, but neither found paper frames results as a fraction of a provable bound. This matches Hoefler's rule 11 (see Q4) and is the angle to lead with.
- **Venue fit:** in the findings, ATC and SOSP each carried one or zero MoE-offload papers per year, and hybrid CPU-GPU MoE decode is well covered at ASPLOS, SOSP, EuroSys, DAC and MLSys. Novelty therefore has to come from the analysis and bound, not from "a GPU expert cache for llama.cpp". The llama.cpp and vLLM RFCs show that expert caching is already a community-discussed feature.

### Gaps
- MoE-APEX (ASPLOS'26) evaluation details were not retrieved (ACM DL 403). It may be the venue version of HOBBIT (both are "mixed/adaptive precision expert offloading"), but this is **unverified**.
- A complete per-venue list for ASPLOS 2026, MLSys 2025, ISCA/MICRO 2025, HPCA 2026 and DAC 2026 could not be fetched (tool permission timeouts on the curated-list pages). Other MoE-offload papers may exist at those venues.
- No specific MoE-offloading paper was confirmed at EuroMLSys, MLArchSys, ESFoMo, ENLSP or WMLSys. These workshops were not individually searched in depth. "WMLSys" could not be identified.
- The Pipelined Sharding authors' affiliation was not shown on the fetched pages. Ram Rangan is commonly associated with NVIDIA, but this is unverified here.
- The creation dates of the llama.cpp and vLLM RFC threads were not checked. That is needed to judge whether they predate the assessed patch.
- It was not checked whether the A10 VM's CPU supports AMX, which KTransformers' fastest path depends on. This affects whether a KTransformers baseline on that VM is fair.

## Q2. What do reviewers at these venues expect from the evaluation?

### Takeaway
The CFPs all ask for the same thing in different words: a significant problem, advances beyond recent work that are clearly stated, and a sound evaluation that shows limitations as well as wins. ASPLOS explicitly requires the SIGPLAN empirical-evaluation guidelines. In practice (from Q1) that means **≥2 hardware platforms, ≥3 models, prefill and decode, at least batch 1 plus some larger batches, the strongest current systems as baselines, per-technique ablations, and an artifact**.

### Cited Findings
- **EuroSys 2027:** judges "novelty, significance, interest, clarity, relevance, and correctness" and wants rigorous evaluation showing both benefits and limitations against prior work. It accepts experience papers "that describe interesting observations or lessons learned" if they include quantitative analysis. Artifact evaluation is optional, with Available, Functional and Reproduced badges. The limit is 12 pages plus references. — [EuroSys 2027 CFP](https://2027.eurosys.org/cfp.html)
- **ASPLOS 2027:** wants "sound experimental methods" that show strengths and limitations, and papers "must follow SIGPLAN's empirical evaluation guidelines". A rapid review of the first 2 pages screens fit. Experience papers and studies refuting prior work are welcome. Artifact evaluation is encouraged. The limit is 11 pages (13 if accepted). — [ASPLOS 2027 CFP](https://www.asplos-conference.org/asplos2027/cfp/)
- **SIGPLAN Empirical Evaluation Checklist:** seven categories, including clearly stated claims, principled benchmark choice, adequate data analysis, relevant metrics, experimental design and presentation of results. — [SIGPLAN Empirical Evaluation](https://www.sigplan.org/Resources/EmpiricalEvaluation/)
- **OSDI '27:** papers should "motivate a significant problem" and "clearly articulate the advances beyond recent and historical work". An Operational Systems track does not require novel research ideas. Each author may be on at most 8 submissions. Decisions are conditional acceptances with no revise-and-resubmit. The limit is 12 pages. — [OSDI '27 CFP](https://www.usenix.org/conference/osdi27/call-for-papers)
- **MLSys 2027:** the research track is judged on "novelty, quality, interest, and impact". Review is double-blind, with 10 pages plus unlimited appendix. Artifact evaluation is voluntary and does not affect acceptance. Concurrent archival submission is prohibited, but arXiv and non-archival venues are allowed. — [MLSys 2027 CFP](https://mlsys.org/Conferences/2027/CallForResearchPapers)
- **ACM SIGOPS ATC 2026:** accepts long (12-page) and **short (6-page)** papers reviewed to the same standard. Extended abstracts are screened on scope, problem importance, advance over the state of the art and experimental support. There is an Operational Systems track. — [ATC 2026 CFP](https://sigops.org/s/conferences/atc/2026/cfp.html)
- **Empirical norms in accepted comparable papers (from Q1):**
  - KTransformers: 1 CPU host with 2 GPUs, 3 models at 2 precisions, 2 baselines, prefill over 32-8K tokens, decode, batch 1, ablation breakdown and task accuracy. — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
  - FineMoE: 2 testbeds, 3 models, 3 real traces, 4 baselines, TTFT and TPOT, batch 1-8 and 5 ablations. — [EuroSys'26 PDF](https://intellisys.haow.us/assets/pdf/Hanfei_FineMoE_EuroSys26.pdf)
  - Pipelined Sharding: 3 client platforms, dense, MoE and vision-language models, interactive and batched modes, and a public AE artifact. — [arXiv HTML](https://arxiv.org/html/2604.26334v1)
  - Fiddler: 2 environments, 3 scenarios (single-batch, long prefill, beam search) and 3 baselines. — [ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/8cd1ce03ea58b3d7dfd809e4d42f08ea-Paper-Conference.pdf)
- **Fair baselines:** KTransformers extended llama.cpp "with expert-level offloading for fair comparison". Reviewers expect the baseline to be run in its strongest configuration. — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)

### Inferences
- **What reviewers will flag, bluntly, in likely order:**
  1. A single GPU/CPU/PCIe platform (A10 on one cloud VM), which also leaves the model's cross-hardware predictive claim untested.
  2. No KTransformers or SGLang+KT baseline and no Pipelined Sharding comparison.
  3. A batch-1 decode focus without prefill/TTFT and without batch >1.
  4. A single author with no institution. Review is double-blind, so this matters only indirectly, through resources and hardware breadth.
- **Easy wins:** the equal-VRAM comparison, pre-registration, a published trace tool and an upstreamable llama.cpp patch. Together with an AE-ready artifact, they fit the reproducibility and "sound methods" language of ASPLOS and EuroSys.
- **Cheapest route to "sufficient" breadth:**
  - Add 2 more platforms with different PCIe generations and CPU memory bandwidth, e.g. a consumer RTX 4090/5090 host on PCIe 4/5, plus an L4/L40S or A100/H100 cloud VM. Show the model predicts each *before* running, which turns the pre-registration into a cross-platform prediction test.
  - Add KTransformers (or SGLang+KT) and a tuned llama.cpp `-ot` override-tensor configuration as baselines.
  - Report prefill/TTFT and at least batch sizes 1, 2, 4 and 8.
- **Paper framing:** a characterization, bound and model paper with a system as validation (an "analysis/experience" framing, which EuroSys and ASPLOS explicitly allow) is more defensible for a single author than a "new system beats SOTA" framing.

### Gaps
- No venue publishes explicit numeric minimums (e.g., "≥3 GPUs"). The breadth norms above are inferred from accepted papers, not stated policy.
- Reviewer-specific commentary was not obtained. Public reviews exist only for venues that use OpenReview (MLSys 2027 uses it; older MLSys reviews may be public), and none were retrieved.

## Q3. Upcoming deadlines relevant to late 2026 and 2027

### Takeaway
**No top-venue acceptance can exist by the 27 Oct 2026 ETH deadline.** EuroSys 2027 fall (24 Sep 2026), ASPLOS 2027 September cycle (9 Sep 2026), PPoPP 2027 (3 Aug 2026) and ACM ATC 2026 (10 Jun 2026) have all passed. The live options are:
- **MLSys 2027:** paper due 30 Oct 2026, three days *after* the ETH deadline; notification 28 Feb 2027.
- **OSDI '27:** due 8 Dec 2026.
- **ISPASS 2027:** about mid-December 2026, inferred from 2026.
- **EuroMLSys 2027:** about February 2027, inferred.

For ETH, what can realistically be shown is an arXiv preprint plus code and "under submission to MLSys 2027".

### Cited Findings
- **ETH AI Center Doctoral Fellowship 2027:** applications open 15 Sep 2026 and close **Tue 27 Oct 2026, 16:00 CET**. Reference letters are due **Mon 2 Nov 2026, 23:59 CET**. The doctoral symposium is **1-2 Feb 2027**. — [ETH AI Center FAQ](https://ai.ethz.ch/research/phd-and-postdoc-programs/fellowship-faqs.html); [Fellowship page](https://ai.ethz.ch/research/phd-and-postdoc-programs/phd-fellowships.html)
- **MLSys 2027:**
  - Submissions open 10 Oct 2026. The paper deadline is **30 Oct 2026**, 12:00 PM PDT per the conference page and 20:00 UTC per the CFP page (these differ by an hour; confirm on OpenReview).
  - Reviews come out 18 Jan 2027, rebuttals are due 21 Jan 2027 and notification is **28 Feb 2027**.
  - The conference is in **Bellevue, WA, 21-25 Jun 2027**.
  - — [MLSys 2027](https://mlsys.org/Conferences/2027); [MLSys 2027 CFP](https://mlsys.org/Conferences/2027/CallForResearchPapers); [OpenReview](https://openreview.net/group?id=MLSys.org%2F2027%2FConference)
  - The CFP page as fetched also showed an "Indio, CA" address, which conflicts with Bellevue on the main 2027 page. Treat Bellevue as authoritative. — [MLSys 2027 CFP](https://mlsys.org/Conferences/2027/CallForResearchPapers)
- **EuroSys 2027 (Rabat, Morocco, 19-23 Apr 2027):**
  - Spring cycle: abstract 7 May 2026, paper 14 May 2026, notification 21 Aug 2026.
  - Fall cycle: **abstract 17 Sep 2026 and paper 24 Sep 2026 (both passed)**; notification 29 Jan 2027.
  - — [EuroSys 2027 CFP](https://2027.eurosys.org/cfp.html)
- **ASPLOS 2027 (Heraklion, Crete, 11-15 Apr 2027):** the April cycle closed 15 Apr 2026. The September cycle closed **9 Sep 2026 (passed)**; its author response runs 1-4 Dec 2026 and notification is 21 Dec 2026. — [ASPLOS 2027 CFP](https://www.asplos-conference.org/asplos2027/cfp/)
- **OSDI '27:** abstract registration **1 Dec 2026** and full paper **8 Dec 2026** (5:59 pm EST). Notification is 16 Mar 2027; the conference is 7-9 Jul 2027 in Baltimore. — [OSDI '27 CFP](https://www.usenix.org/conference/osdi27/call-for-papers)
- **USENIX ATC → ACM SIGOPS ATC:**
  - USENIX announced it would discontinue the Annual Technical Conference. — [USENIX ATC Announcement](https://www.usenix.org/blog/usenix-atc-announcement); [LWN: The end of the USENIX ATC](https://lwn.net/Articles/1020306/)
  - "ATC 2026 is now an ACM conference, but with the same community and scope as before". Its deadline was **10 Jun 2026 (passed)**, notification 18 Sep 2026, and the conference runs 16-18 Nov 2026. — [ACM SIGOPS ATC 2026 CFP](https://sigops.org/s/conferences/atc/2026/cfp.html)
  - A listing places ATC 2026 in Hong Kong. — [infosec-conferences listing](https://infosec-conferences.com/event/20261115-acm-sigops-annual-technical-conference-atc-2026/)
  - No ATC 2027 CFP was found.
- **PPoPP 2027:** the paper deadline was **3 Aug 2026 (passed)** and notification is 26 Oct 2026. The conference is 20-24 Mar 2027 in Salt Lake City, co-located with HPCA, CGO and CC. — [PPoPP 2027 CFP](https://ppopp27.sigplan.org/track/PPoPP-2027-papers)
- **ISPASS:** the 2026 edition had an abstract deadline of 8 Dec 2025 and a paper deadline of 15 Dec 2025, notification 23 Feb 2026 and a 9-page limit. It has a "tool and benchmark papers" option. — [ISPASS 2026 submission](https://ispass.org/ispass2026/submission.php)
- **EuroMLSys:** the 2026 edition (co-located with EuroSys'26 in Edinburgh) had a revised deadline of 24 Feb 2026 and a 6-page limit. It is archival in the ACM DL, with opt-out. — [EuroMLSys](https://euromlsys.eu/)
  - EuroSys 2027 has a workshop call, but EuroMLSys 2027 dates were not posted. — [EuroSys 2027 Call for Workshops](https://2027.eurosys.org/workshop.html)
- **MLArchSys @ ISCA 2026:** deadline 1 May 2026, held 28 Jun 2026 in Raleigh. It accepts short papers, position papers and early work. — [SIGARCH MLArchSys 2026](https://www.sigarch.org/call-contributions/mlarchsys-isca-2026/)
- **Other 2027 conference dates:** HPCA 2027 is 20-24 Mar 2027 (Salt Lake City), DATE 2027 22-24 Mar 2027 (Dresden) and DAC 2027 10-16 Jul 2027 (San Jose). Deadlines were not shown. — [casys-kaist deadlines](https://casys-kaist.github.io/)

### Inferences
- **Realistic plan by 27 Oct:**
  - Post an arXiv preprint and public code before 27 Oct so the motivation letter, CV and video can link to it.
  - Submit to MLSys 2027 on 30 Oct. It allows arXiv, and double-blind review is unaffected by a preprint under a different title if anonymization rules are followed.
  - MLSys notification (28 Feb 2027) comes *after* the ETH symposium (1-2 Feb 2027). At the symposium the paper will be "under review", and MLSys reviews (18 Jan 2027) will arrive before it.
- **Fallback venues** if MLSys rejects or the platforms cannot be broadened in time:
  - ISPASS 2027: likely mid-December 2026, inferred from the 2026 pattern. It is a performance-analysis venue, which suits a bound-and-model paper.
  - OSDI '27 (8 Dec 2026): a long shot for a single-platform study.
  - EuroMLSys 2027: about February 2027, 6 pages, archival. This is the most attainable peer-reviewed publication.
  - EuroSys 2028 spring (~May 2027) and ASPLOS 2028 April cycle (~Apr 2027): both are inferred timings.
- **ATC short papers:** the ACM ATC 6-page short-paper format fits this work, but the next cycle (ATC 2027) has no announced dates.

### Gaps
- ISPASS 2027, EuroMLSys 2027, MLArchSys 2027, SOSP 2027, EuroSys 2028 and ASPLOS 2028 CFP dates are not yet posted or were not found. The timings above are inferred from prior editions.
- SC26 and HPCA 2027 deadlines were not retrieved; both are presumably passed.
- ESFoMo (ICML) and ENLSP (NeurIPS) 2026 deadlines were not retrieved. ENLSP 2026 is attached to NeurIPS 2026 (December), so its deadline has very likely passed; unverified.

## Q4. SPCL / Hoefler methodology preferences, SPCL's LLM-inference work, and ETH AI Center fellowship requirements

### Takeaway
Hoefler's "twelve rules" (SC'15) are the methodology bar SPCL applies. Rule 11 ("If possible, show upper performance bounds") and the statistical rules (confidence intervals, no normality assumption, sound comparisons, not summarizing ratios) map directly onto a bound-plus-model study, which is the assessed work's strongest fit with SPCL. SPCL's recent LLM-inference output is mostly quantization and sparsity (QuaRot, QUIK, SpQR, MARLIN) plus characterization (IISWC'25). The fellowship application does **not** require publications. It requires a motivation letter, CV, **video** and **2-3 reference letters from independent PIs or lecturers**, then a research talk at the February symposium.

### Cited Findings
- **The twelve rules** (Hoefler & Belli, SC'15), verbatim:
  - Rule 1: "report if the base case is a single parallel process or best serial execution, as well as the absolute execution performance of the base case".
  - Rule 3: "Use the arithmetic mean only for summarizing costs. Use the harmonic mean for summarizing rates."
  - Rule 4: "Avoid summarizing ratios; summarize the costs or rates that the ratios base on instead."
  - Rule 5: report confidence intervals for nondeterministic data.
  - Rule 6: "Do not assume normality ... without diagnostic checking."
  - Rule 7: compare "in a statistically sound way".
  - Rule 9: "Document all varying factors and their levels as well as the complete experimental setup".
  - Rule 11: "If possible, show upper performance bounds to facilitate interpretability of the measured results."
  - Rule 12: plot enough information to interpret the results.
  - — [Scientific Benchmarking PDF](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf); [ACM DL](https://dl.acm.org/doi/10.1145/2807591.2807644)
- **SC'15 literature survey:**
  - Only 15 of 95 applicable papers mentioned any variance measure, and only 2 reported confidence intervals.
  - 15 of 39 speedup papers (38%) omitted absolute base-case performance.
  - None of the 95 used statistically sound comparisons.
  - — [Scientific Benchmarking PDF](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf)
- **"Data Movement Is All You Need":** "Data Movement Is All You Need: A Case Study on Optimizing Transformers" (Ivanov, Dryden, Ben-Nun, Li, Hoefler), MLSys 2021. It is SPCL's data-movement-centric analysis of transformer performance and the natural lineage for a bytes-over-bandwidth decode model. — [MLSys 2021 proceedings](https://proceedings.mlsys.org/paper_files/paper/2021/hash/bc86e95606a6392f51f95a8de106728d-Abstract.html); [PDF](https://htor.inf.ethz.ch/publications/img/data_movement_is_all_you_need.pdf)
- **SPCL publications 2024-2026 relevant to LLM inference:**
  - Quantization and compression: QUIK (EMNLP'24), QuaRot (NeurIPS'24), SliceGPT (ICLR'24), SpQR (ICLR'24), MARLIN mixed-precision inference kernels (PPoPP'25) and HALO (NeurIPS'25).
  - Characterization: "Confidential LLM Inference: Performance and Cost Across CPU and GPU TEEs" (IISWC 2025).
  - Sparsity: BLaST block-sparse transformers (arXiv 2025) and two ICML'26 sparse-training papers.
  - Communication: "Demystifying NCCL" (HOTI'25).
  - — [SPCL Publications](https://spcl.inf.ethz.ch/Publications/); [IISWC'25 PDF](https://spcl.inf.ethz.ch/Publications/.pdf/IISWC25_confidential_llms.pdf)
- **SPCL direct-hiring page:** apply by email to spcl-hiring@spcl.inf.ethz.ch with "[JOB@SPCL]" in the subject. It requires:
  - a full CV and complete transcripts;
  - a letter of intent (max 500 words);
  - a "brief description of most important achievement";
  - references to previous publications and software projects/code samples;
  - **three** professional references.
  - It seeks experience in "parallel computing, compilers, (performance) optimization, and/or computer systems" and strong C/C++.
  - — [SPCL Jobs](https://spcl.inf.ethz.ch/Jobs/)
- **ETH AI Center fellowship: materials and references:** degree certificates and transcripts, a motivation letter, CV and **video**, plus reference letters. At least 2 letters are required and a 3rd is optional. At least two referees must be "independent investigators, principal scientists, group leaders, or lecturers", with at most one postdoc, and "References from PhD students or class-mates cannot be accepted." — [ETH AI Center FAQ](https://ai.ethz.ch/research/phd-and-postdoc-programs/fellowship-faqs.html)
- **Fellowship process:**
  - Publications are not mentioned as required or evaluated in the FAQ.
  - Applicants do not need to contact faculty beforehand. Mentors are matched at the symposium, where shortlisted candidates "give a research talk and have personal interviews".
  - The application cannot be edited after submission.
  - — [ETH AI Center FAQ](https://ai.ethz.ch/research/phd-and-postdoc-programs/fellowship-faqs.html)
- **Fellowship terms:** 8-12 doctoral positions and two co-supervising PIs "from different fields" with equal weight. A Master's degree by June 2027 is required. Salary is CHF 73,100-83,500. The page asks for "an excellent track record". — [Fellowship page](https://ai.ethz.ch/research/phd-and-postdoc-programs/phd-fellowships.html); [Guidelines 2026 PDF](https://ethz.ch/content/dam/ethz/special-interest/infk/ai-dam/documents/Guidelines_Doctoral_Fellowships_2026.pdf)

### Inferences
- **Presentation changes the rules call for:**
  - Report absolute tokens/s for both baseline and patch (rule 1) instead of only "1.41-1.81x".
  - Do not average speedup ratios across models (rule 4). If a summary is needed, use a harmonic mean of rates or the geometric mean of ratios, and say which.
  - Give confidence intervals from repeated runs with a nonparametric test (rules 5-7), since cloud-VM noise on a single A10 VM is exactly the variance source reviewers worry about.
  - Show the bound on every plot (rule 11).
  - The "46-63% of the speed-of-light bound" framing is the rule-11-style presentation Hoefler asks for. It should be the central figure, not an aside.
- **Fellowship references:** the binding constraint for an independent researcher is the **two letters from independent investigators or lecturers** who can speak to research ability, due 2 Nov 2026. SPCL direct applications need **three** professional references. Publications are not required, but the "most important achievement" (SPCL) and the symposium research talk (AI Center) reward exactly this kind of self-contained, rigorous study.
- **Supervisor pairing:** AI Center fellows need two supervisors from *different fields*. Pairing Hoefler (HPC/systems) with an ML PI is expected, and the "Accessible AI" angle (frontier MoE on consumer hardware) gives a natural ML-side co-supervisor story.
- **SPCL publishing pattern:** SPCL routinely publishes at ML venues (NeurIPS, ICLR, ICML), at PPoPP and SC, and at characterization venues (IISWC, HOTI). An IISWC or ISPASS-style characterization-plus-bound paper is well aligned with how the group itself publishes.

### Gaps
- The 2027-cohort guidelines PDF was not retrieved; only the 2026 guidelines were located. The exact video length, motivation-letter length and research-proposal requirements for 2027 were not confirmed.
- No SPCL paper specifically on MoE *offloading* or CPU-GPU hybrid MoE decode was found. SPCL's list did not surface one.
- No public statement from Hoefler specifically endorsing pre-registration was found (see Q5).
- Acceptance rates and applicant numbers for the AI Center fellowship are not published in the FAQ.

## Q5. Are pre-registered systems evaluations common, and is there precedent?

### Takeaway
Pre-registration is **essentially absent from systems venues**. No registered-report track exists at OSDI, SOSP, EuroSys, ASPLOS, MLSys or ATC; their reproducibility mechanism is optional post-acceptance artifact evaluation. The nearest precedents are the NeurIPS 2020/2021 pre-registration workshops in ML and registered reports in empirical software engineering. It is a genuine differentiator for a methodology-minded reader like Hoefler, but not something systems PCs reward on its own.

### Cited Findings
- NeurIPS 2020 held the workshop "The pre-registration experiment: an alternative publication model for machine learning research", followed by a 2021 edition. The 2021 proceedings appeared as PMLR volume 181. — [NeurIPS 2020 workshop](https://neurips.cc/virtual/2020/workshop/16158); [NeurIPS 2021 workshop](https://neurips.cc/virtual/2021/workshop/21885); [PMLR v181](https://proceedings.mlr.press/v181/); [preregister.science](https://preregister.science/)
- Registered reports have been adopted in empirical software engineering and studied in "Registered reports in software engineering" (Empirical Software Engineering, 2022). — [EMSE paper](https://dl.acm.org/doi/10.1007/s10664-022-10277-5)
- A SIGCSE 2023 session covered "Registered Reports and Preregistration: A new way to conduct research" in computing education. — [SIGCSE TS 2023](https://dl.acm.org/doi/10.1145/3545947.3573352)
- Systems venues rely on artifact evaluation instead:
  - EuroSys 2027: optional, with Available, Functional and Reproduced badges. — [EuroSys 2027 CFP](https://2027.eurosys.org/cfp.html)
  - ASPLOS 2027: continued as a tradition. — [ASPLOS 2027 CFP](https://www.asplos-conference.org/asplos2027/cfp/)
  - MLSys 2027: voluntary and does not affect acceptance. — [MLSys 2027 CFP](https://mlsys.org/Conferences/2027/CallForResearchPapers)
- Hoefler's SC'15 paper frames the problem as poor reporting and reproducibility in HPC performance papers. It prescribes documenting "all varying factors and their levels" and statistically sound comparisons, but it does not prescribe pre-registration. — [Scientific Benchmarking PDF](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf)

### Inferences
- **Value by audience:** a pre-registered, prediction-first evaluation is unusual in systems, and "predict, then measure, then report the error" is the strongest honest way to validate a performance model. For the ETH application and the talk, it signals the scientific-benchmarking mindset SPCL advocates. For a systems PC it is a nice-to-have. It does not compensate for single-platform breadth or missing baselines.
- **Making it count:** pre-register predictions for *new* hardware the model has not seen, e.g. a second and third GPU/PCIe/CPU platform. That converts the pre-registration from a process claim into a falsifiable, cross-platform validation of the bytes-over-bandwidth model, which is exactly what a single-A10 study cannot show.
- **Terminology:** call it "pre-registered predictions" or "prediction-first evaluation" and link a timestamped record (e.g., an OSF or GitHub tagged commit). Reviewers unfamiliar with the practice will want to verify the timestamp.

### Gaps
- No pre-registered or registered-report paper at OSDI, SOSP, EuroSys, ASPLOS, MLSys, ATC, SC, PPoPP, ISPASS or IISWC was found. This rests on absence in searches, not an exhaustive proceedings scan.
- Whether any systems workshop explicitly invites pre-registered or registered-report submissions was not found.
