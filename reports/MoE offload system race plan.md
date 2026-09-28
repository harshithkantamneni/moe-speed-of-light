# Race the cache, sell the yardstick

Harshith should run option C as a pre-registered race rather than a product launch. The setup: rent one whole Vast.ai machine, an **RTX 5090 gaming desktop with a 16-core AMD Ryzen 9 9950X and at least 126 GB of RAM, at $0.74–0.94 an hour**. Hold every system to the same measured GPU memory. Make the headline where each system lands against his speed-of-light bound (the fastest decode the hardware physically allows), with his llama.cpp expert cache as one entrant among several. Before touching a GPU, he should fix a confirmed bug that affects every speed-of-light fraction in the case study. The bound leaves out the model's final output layer (the "LM head"), so it is too optimistic. Corrected, the cache reaches **about 54–76% of the bound (the paper says 47–64%) and llama.cpp 37–53% (paper: 33–45%)**. On this desktop the realistic expectation has two parts:

- **A wide win over the best llama.cpp configuration on gpt-oss-120b.** This is an uncalibrated estimate of roughly 2.3–3.7×.
- **A likely loss of about 10–30% to FreeToken**, also an uncalibrated estimate. FreeToken is an August 2026 engine that already ships the same kind of GPU-signalled CPU hand-off, and its fetch-heavy design suits a machine where the GPU link is about as fast as main memory.

None of the cache's mechanisms is new. The defensible contribution is therefore the protocol and the bound-scored table, whoever wins. The timeline:

- **Friday 2 October:** a gate decides whether a one-week sprint (3–9 October, planning gain +10–20%) chases FreeToken or only the llama.cpp family.
- **Sunday 11 October:** the race runs in one session.
- **13 October:** competitor authors get the logs.
- **Around 22 October:** the arXiv post goes up, ahead of the fellowship deadline of Tuesday 27 October, 16:00 CET (10:00 in Milwaukee).

The plan costs **about 41–42 GPU-hours and $33–42**. It collides with the audit-extension plan, whose measurements fall in the same 5–14 October window and which wants another ~$29–40 and ~74 hours of his time. The five most pressing of his twelve decisions are that choice, a funded Vast.ai account, whether he is the llama.cpp contributor "leloch", whether to race the Qwen3.6 model, and whether MLSys 2027 (deadline 30 October) is in scope.

## The cache beats llama.cpp by about 1.4–1.6×, but the paper overstates its bound

**What the system is.** gpt-oss-120b and Qwen3-30B-A3B are **mixture-of-experts (MoE)** models. Each layer holds 128 small sub-networks called experts, and each generated token uses only a few of them: 4 per layer for gpt-oss and 8 for Qwen3 ([baseline note §3.2](../research_notes/MoE%20offload%20system%20race%20plan/our_system_baseline.md)). **Decode** is generating one token at a time for one user, measured in tokens per second (tok/s). It is limited mainly by how fast weights can be read from memory. When the experts do not fit in the GPU's own memory (**VRAM**), they stay in the computer's main memory (**DRAM**) and run on the CPU; this is "offloading". **llama.cpp**, the most-used open-source engine for running models on home hardware, does this with `--n-cpu-moe K`, which permanently places the experts of the first K layers on the CPU.

Harshith's system is a patch of about 2,600 added lines to llama.cpp, of which about 540 are benchmark tools. It replaces that fixed split with a cache ([baseline note §2.1, §2.3](../research_notes/MoE%20offload%20system%20race%20plan/our_system_baseline.md)), which works like this:

1. **Where the weights live.** Every expert sits in **pinned host memory**: RAM that the GPU can read directly over **PCIe**, the CPU–GPU link.
2. **Hits.** The GPU keeps C copies ("slots") of experts per layer and runs the selected experts it holds.
3. **Misses.** For the other selected experts, a small GPU program (a "kernel") writes the token's input into a **mailbox** in pinned memory. CPU **helper threads** that watch the mailbox compute the missing experts and write the results back. A GPU kernel spins until the results arrive, then merges them. A layer with no miss skips the hand-off entirely.
4. **Admission.** After each token, a CPU policy called decayed-frequency admission (DFA) decides which experts to copy into the cache. The copies take effect two tokens later.

**What it achieved on the A10.** On rented NVIDIA A10 instances (24 GB GPU, 30-vCPU Xeon host), the cache ran **1.31–1.66× faster than llama.cpp `--n-cpu-moe` with the same expert bytes on the GPU**. On gpt-oss-120b that meant **62.6–71.1 tok/s** ([baseline note §1](../research_notes/MoE%20offload%20system%20race%20plan/our_system_baseline.md)). Its answers matched llama.cpp's as closely as llama.cpp matches itself:

- **Top-1 agreement.** At each position of a fixed text, the fraction of positions where both systems pick the same most-likely next token. Cache versus llama.cpp was 0.981–0.993. An all-GPU run versus llama.cpp gave 0.979–0.991.
- **Negative log-likelihood (NLL).** A measure of how surprised the model is by the text. The cache's average differed from llama.cpp's by −0.99% to +1.06% ([baseline note §3.3](../research_notes/MoE%20offload%20system%20race%20plan/our_system_baseline.md)).

**Five corrections before anyone quotes the case study.** Re-reading the committed result files turned up five problems.

| Paper's case-study claim | What the evidence supports | Why |
|---|---|---|
| Cache at 47–64% of speed-of-light (SoL), llama.cpp at 33–45% | **≈54–76% and ≈37–53%** | The SoL scripts split the all-GPU step into "dense" and "expert" work by bytes but omit the LM head (615 MB on gpt-oss). Expert work then looks like 65% of the step, where GPU profiler (Nsight) timings give 45% (gpt-oss-20b) and 42% (Qwen3 Q4_K_M, a 4-bit format). The bound falls from 181 to 156 tok/s (gpt-oss-20b) and from 197 to 164 tok/s (Qwen3 Q4_K_M). The audit's own physical bound already includes the head and is unaffected ([baseline note §4.1, §7](../research_notes/MoE%20offload%20system%20race%20plan/our_system_baseline.md); [synthesis, finding 1](../research_notes/MoE%20offload%20system%20race%20plan/synthesis.md)) |
| 1.31–1.66× over llama.cpp | ≈**1.37–1.60×** from a second instance; every instance-level ratio ≥ 1.31 | The final A10 runs (phase 4.5) have no repeats. llama.cpp moved −7.8% to +1.6% between instances while the cache moved 0 to −1.8%. The 1.66× top comes from one llama.cpp run 7.8% slower than the same configuration elsewhere ([baseline note §3.4](../research_notes/MoE%20offload%20system%20race%20plan/our_system_baseline.md)) |
| The baseline is llama.cpp as users run it | Unverified | The baseline was the patched build's own `--ncmoe` inside the ec-bench tool, never checked against stock `llama-bench` or `llama-server`. `--no-mmap` (loading weights into RAM instead of memory-mapping the file) alone moved it by −13.4% to +1.7% ([baseline note §7](../research_notes/MoE%20offload%20system%20race%20plan/our_system_baseline.md)) |
| gpt-oss-120b fidelity on dataset text | Use the model's own text only | On the dataset text top-1 agreement is 0.890–0.897, because that text is foreign to the model (NLL 5.75 vs 0.58 nats on own text) ([baseline note §3.3](../research_notes/MoE%20offload%20system%20race%20plan/our_system_baseline.md)) |
| The patch header documents its defaults | It does not, in three places | `LLAMA_EC_PACED` is documented as 1 but coded as 0. `LLAMA_EC_KAPPA` defaults to 0, but every evaluated run used 1 or 2. `LLAMA_EC_MXK` is undocumented. Anyone reproducing from the header gets a different system ([baseline note §2.2, §7](../research_notes/MoE%20offload%20system%20race%20plan/our_system_baseline.md)) |

There are two versions of the corrected bound. Adding the head bytes gives 161.6 and 183.2 tok/s; the Nsight split gives 156.4 and 163.7. The note's own summary also quotes 54–76% in one place and 54–74% in another. So the fractions must be **regenerated by script with a regression test**, not typed in by hand ([synthesis, finding 1](../research_notes/MoE%20offload%20system%20race%20plan/synthesis.md)).

**Where the cache loses time.** The analysis took 22 A10 configurations and split the time above the bound (the "excess") into parts:

| Cost | Share of the excess | Size |
|---|---|---|
| **O**, a fixed overhead per MoE layer per token: the hand-off, extra cache kernels, host-side policy work and interference from admission copies | 37% on average | 26–38 µs per layer on gpt-oss-20b and Qwen3, and roughly 50–66 µs on gpt-oss-120b, whose all-GPU time had to be extrapolated. Paid even on layers with no miss |
| The bound's ideal balancing of CPU and GPU work | 27% | Mostly not attainable at current per-request costs |
| **f**, the helpers' fixed cost per request | 16% | About 50 µs on gpt-oss and 18–26 µs on Qwen3 |
| Helper speed below the host's memory bandwidth | 13% | 129 GB/s achieved against a 166 GB/s read |
| DFA's extra misses over the offline-optimal policy | 7% | Material only at the smallest budget |

Source: [baseline note §1, §4.3](../research_notes/MoE%20offload%20system%20race%20plan/our_system_baseline.md).

**O has never been measured directly**: the profiler traces of the cache from job 012 are empty files ([baseline note §4.2](../research_notes/MoE%20offload%20system%20race%20plan/our_system_baseline.md)).

**Why the A10 margin does not transfer.** A Blackwell GPU (NVIDIA's generation that includes the RTX 5090) is projected to shrink the GPU part of each token about 2.5×, while O and f are assumed to barely change. With A10 constants, a structural model projects our margin over llama.cpp by host memory bandwidth ([baseline note §5.4](../research_notes/MoE%20offload%20system%20race%20plan/our_system_baseline.md)):

| Host memory bandwidth | Projected margin over llama.cpp |
|---|---|
| 150 GB/s | 1.35–2.1× |
| 250 GB/s | 1.2–1.66× |
| 400 GB/s | 1.06–1.37× |

The model reproduces the A10 cache within a 2.5% median, but that check is in-sample. All three scenarios are server-class hosts; the recommended desktop is expected (not yet measured) to deliver roughly 50–70 GB/s, outside every scenario ([synthesis C9](../research_notes/MoE%20offload%20system%20race%20plan/synthesis.md)). The headroom note's illustrative estimate, that the system would fall to 39–46% of the bound on a 5090, used an obsolete 84 µs/layer hand-off and constants from a third-party model fit. It should be discarded ([synthesis C4](../research_notes/MoE%20offload%20system%20race%20plan/synthesis.md)).

One more projection matters for scope. On Qwen3.6-35B-A3B the cache is projected at only **0.78–1.02×** llama.cpp on the same 150–400 GB/s server scenarios, assuming independent routing (a pessimistic case, since no trace of that model exists). A GGUF model file (llama.cpp's format) with merged gate and up tensors also silently switches the mailbox off ([baseline note §1, §2.3](../research_notes/MoE%20offload%20system%20race%20plan/our_system_baseline.md)).

## Every mechanism in our cache already ships somewhere else

The race field is small because most published systems either have no code or cannot run on a Blackwell consumer GPU (architecture code **sm_120**).

| System | What it does with experts that do not fit | Code | Headline claim |
|---|---|---|---|
| **FreeToken** (FlashML, Aug 2026) | One cache shared by all layers ("pooled"). A miss is fetched over PCIe (`offload`), computed on the CPU (`cpu`), or, per step, split between the two (`hybrid`), calibrated by a bandwidth benchmark | Apache-2.0; tag v0.1.3 and main 0d652e7 (25 Sep); needs driver r580+ and CUDA 13 ([install](https://github.com/FlashML-org/FreeToken/blob/main/docs/install.md); [models](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md)) | **77–83 tok/s on Qwen3.6-35B-A3B on an RTX 5090, 1.8–2.3× the strongest baseline** ([arXiv 2608.16157](https://arxiv.org/html/2608.16157)). Community: **gpt-oss-120b at 127.1 tok/s with 40.4% of experts in ~29 GB** of a 5090 ([zenn.dev](https://zenn.dev/holy_fox/articles/53b82eed45f956?locale=en)) |
| **KTransformers** (SOSP'25; now kt-kernel v0.7.1 + a fork of the SGLang serving engine) | A fixed set of GPU experts per layer, with fast AVX-512/AMX CPU kernels (wide-vector and matrix instructions); optional Expert Deferral changes outputs | Apache-2.0; sm_120 needs a source build; torch pinned to 2.9.1; no gpt-oss support ([kt-kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)) | 1.25–1.76× over a Feb-2025 llama.cpp the authors patched ([SOSP'25](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)) |
| **Pipelined sharding** (NVIDIA, MLSys'26 oral) | A llama.cpp b6097 fork that plans what stays inside a fixed VRAM budget (`-mva`) | MIT; tag v2.0.3-mlsys26 on Zenodo; developed on Windows ([README](https://github.com/deepshnv/pipeshard-mlsys26-ae/blob/main/README.md)) | 25.7–158.6 tok/s on Qwen3-30B Q4_0 at 2–32 GB. Its own llama.cpp baseline with every expert on the CPU ran at 25.7 tok/s against a predicted 49.5 ([arXiv 2604.26334](https://arxiv.org/html/2604.26334); [entrants note Q5](../research_notes/MoE%20offload%20system%20race%20plan/race_entrants.md)) |
| **llama.cpp community caches** (#27861, leloch v2) | A GPU cache of CPU-resident experts inside llama.cpp | Unmerged. **Shipped #27861 silently does nothing** because a dry-run context latches its initialisation ([everett6](https://github.com/everett6/moe-prefetch/blob/main/docs/UPSTREAM-BUGS.md)) | leloch v2: **+2.1%** on Qwen3-30B-A3B ([branch](https://github.com/leloch/llama.cpp/tree/moe-cache-v2-pr)). The fixed #27861 reached 1.49×, but not at equal VRAM ([results](https://github.com/everett6/moe-prefetch/blob/main/docs/FINAL_RESULTS.md)) |
| **ik_llama.cpp** | A llama.cpp fork with fused-MoE defaults and the same static placement | adce16f; no Blackwell-specific code ([README](https://github.com/ikawrakow/ik_llama.cpp/blob/adce16f50f454da2aa2587801c885ffba7805c72/README.md)) | Its own table shows decode "within the noise" after its placement change ([PR #2262](https://github.com/ikawrakow/ik_llama.cpp/pull/2262)) |
| DALI, SeqMoE, arXiv 2606.10493 | Various | **No public code** ([entrants note Q1](../research_notes/MoE%20offload%20system%20race%20plan/race_entrants.md)) | Can only be cited, never raced |

**What is not new.** Each piece of the cache has earlier public examples:

- **GPU caches whose misses run on the CPU:** HybriMoE ([arXiv 2504.05897](https://arxiv.org/abs/2504.05897)), the ASP-DAC'26 system 2512.16473 ([arXiv](https://arxiv.org/abs/2512.16473)), DALI, KTransformers, FreeToken, and three llama.cpp contributors' branches.
- **Decayed-frequency admission:** llama.cpp PRs #26563 and #26824 ([PR #26824](https://github.com/ggml-org/llama.cpp/pull/26824)).
- **GPU kernels that load experts straight from mapped host memory:** SeqMoE ([arXiv 2609.12978](https://arxiv.org/abs/2609.12978)).
- **One fused fill per step:** FreeToken ([arXiv 2608.16157](https://arxiv.org/abs/2608.16157)).

The most damaging overlap is the hand-off itself. Since its first public commit on **11 August 2026**, FreeToken's CPU executor works like this ([cpu_executor.py](https://github.com/FlashML-org/FreeToken/blob/main/python/freetoken/moe/cpu_executor.py); [initial commit](https://github.com/FlashML-org/FreeToken/commit/3af9d90ee5e7af9bbff1e16f4f1c6201f41fff25)):

1. The GPU raises a "ready" flag in mapped pinned memory (`cuStreamWriteValue64`).
2. A persistent CPU coordinator polls the flag and runs the layer.
3. The GPU waits on a "done" flag with a stream memory operation that occupies no GPU compute unit.

A read of the source at 0d652e7 confirmed this ([synthesis, finding 3](../research_notes/MoE%20offload%20system%20race%20plan/synthesis.md)). It predates the patch's first commit on 26 September ([entrants note Q5](../research_notes/MoE%20offload%20system%20race%20plan/race_entrants.md)). FreeToken's comments also note that its first version used a spin-wait kernel, and dropped it because laptop power managers then clamped the CPU clock. Finally, the scheduler "overlap" patch that an earlier note called the novel systems contribution is not even part of the raced system: it "only matters for split-graph mode, not the mailbox" ([synthesis C1](../research_notes/MoE%20offload%20system%20race%20plan/synthesis.md)).

**What differs, and must be measured rather than claimed.** Three implementation choices separate our hand-off from FreeToken's ([synthesis, finding 3](../research_notes/MoE%20offload%20system%20race%20plan/synthesis.md)):

| | Our cache | FreeToken |
|---|---|---|
| When the CPU is involved | The GPU skips the hand-off on layers with no CPU expert | Submits and waits on every MoE layer |
| How activations and results move | Kernel loads and stores to pinned memory | Copy-engine transfers |
| How the GPU waits | A 4-block spinning kernel | A stream memory operation |

The caches also differ: ours is per-layer and FreeToken's is pooled across layers. What nobody has done is a third-party race at equal VRAM, scored against a bound. The methodology note found no paper that combines five things ([methodology note Q3](../research_notes/MoE%20offload%20system%20race%20plan/fair_race_methodology.md)):

1. One host for all systems.
2. Measured-VRAM equalisation with the components listed.
3. Bit-aligned weights.
4. Harmonic summaries with confidence intervals.
5. A speed-of-light score.

**Why the fair arena is FreeToken's home ground.** Whether copying an expert to the GPU beats running it on the CPU depends on a reuse break-even, r* = p/(b − a):

- **p** is the time to copy the expert over PCIe;
- **b** is the time to run it on the CPU;
- **a** is the time to run it on the GPU.

The break-even moves with the host ([baseline note §5.4](../research_notes/MoE%20offload%20system%20race%20plan/our_system_baseline.md); [synthesis, theme 5](../research_notes/MoE%20offload%20system%20race%20plan/synthesis.md)):

| Host | PCIe | DRAM | Break-even r* |
|---|---|---|---|
| A10 | ~25 GB/s | 166 GB/s | ≈11 |
| Server (projection) | PCIe 5, 63 GB/s assumed | 150 GB/s | ≈3 |
| Desktop 5090 (assumed) | 49–52 GB/s | 54–60 GB/s | **≈1.1–1.3** (synthesis estimate) |

**A copied expert pays for itself after about one reuse**, so on a desktop a fetch-plus-pooled-cache design is structurally favoured. Our CPU-miss design is favoured on servers. FreeToken's own "RTX 5090 desktop" testbed is exactly this class: a Ryzen 9 9950X3D measuring 49.0 GB/s PCIe and 53.8 GB/s CPU expert bandwidth ([methodology note Q3](../research_notes/MoE%20offload%20system%20race%20plan/fair_race_methodology.md)). Pooling the budget across layers adds a further 0.4–2.1 points of hit rate ([baseline note §6](../research_notes/MoE%20offload%20system%20race%20plan/our_system_baseline.md)).

## Rent a whole gaming desktop, not a slice of a server

**Race on a Vast.ai whole-machine RTX 5090 desktop.** Vast.ai is a marketplace that rents out other people's machines. "Whole machine" means no other renter shares the host, which the offer shows as `gpu_frac = 1`. The race host needs a Ryzen 9 9950X or 9950X3D, DDR5 memory, a PCIe 5.0 x16 link, at least 126 GB of RAM and an NVIDIA driver of 580 or newer. A public snapshot of Vast offers on 28 September held these candidates ([Vast API](https://console.vast.ai/api/v0/bundles/); [machine note §2](../research_notes/MoE%20offload%20system%20race%20plan/race_machine.md)):

| Host | RAM | $/h | Status | Notes |
|---|---|---|---|---|
| RTX 5090 + Ryzen 9 9950X, Japan | 255 GB | **0.735** | verified, reliability 0.999 | **3-day maximum rental**; 6.4 Gb/s download |
| RTX 5090 + Ryzen 9 9950X3D, Sweden | 255 GB | 0.936 | verified, 0.960 | 22 days; 0.94 Gb/s download; FreeToken's own desktop CPU |
| RTX 5090 + Ryzen 9 9950X, Sweden | 128 GB | 0.804 | verified, 0.999 | – |
| RTX 5090 + Ryzen 9 9950X, India | 128 GB | 0.843 | verified | A VM host, where GPU clock locking is possible but unverified |

The desktop is the fairer arena for five reasons ([synthesis D1](../research_notes/MoE%20offload%20system%20race%20plan/synthesis.md)):

1. **No co-tenants on the memory bus.** Nobody else shares its memory, and this race is decided by memory bandwidth.
2. **Price.** It costs 2–5× less per hour than the Verda alternatives.
3. **CPU features.** Zen 5's **AVX-512** (wide CPU vector instructions) switches on both KTransformers' BF16 (16-bit) kernel and our MXFP4 (4-bit) helper kernel.
4. **Offload is natural.** 32 GB of VRAM means gpt-oss-120b must offload without any artificial cap.
5. **It matches the users.** It is the machine people run models on at home, which is what the accessible-AI mission is about.

The costs are real:

- **No GPU clock locking** in a Vast Docker container. Clocks must be logged and the runs interleaved.
- **Thin supply.**
- **Derated memory.** Four-DIMM desktop boards run DDR5 below its rated speed, so expect only ~50–70 GB/s of read bandwidth ([machine note §2](../research_notes/MoE%20offload%20system%20race%20plan/race_machine.md)). The result is desktop-specific and must say so.

**Qualify before committing.** Give each candidate a 15-minute check:

- CPU flags (`avx512_bf16`, `avx512_vnni`);
- memory layout (`numactl -H`);
- the GPU's PCIe link (Gen5 x16);
- a read-bandwidth sweep over 1–16 threads;
- pinned host-to-GPU bandwidth;
- Vast's own GPU memory-bandwidth figure of at least 1,400 GB/s. Some listed 5090s read only 633–752 GB/s, which suggests throttled cards ([machine note §2](../research_notes/MoE%20offload%20system%20race%20plan/race_machine.md)).

Keep the host with the highest stable read bandwidth, and prefer the 9950X3D on a tie. If supply fails, the fallbacks are, in order:

1. Another whole 5090 on a Zen 4/5 desktop with at least 126 GB. About ten such hosts cost $0.51–0.88/h.
2. An RTX PRO 6000 workstation card on a Zen 5 desktop, with every entrant capped at 32 GB of measured memory.
3. Only if Vast is impossible, RunPod's RTX PRO 6000 at $2.09/h, in a container with an unknown CPU ([synthesis D1](../research_notes/MoE%20offload%20system%20race%20plan/synthesis.md)).

Avoid Intel Core Ultra (Arrow Lake) hosts. They lack AVX-512, so KTransformers drops to slower kernels and our helpers fall back to generic code.

**Why not Verda, RunPod or Lambda.** Verda's `1RTXPRO6000.30V` (30 vCPUs, 90 GB, **$1.964/h**) is sized as exactly one eighth of an 8-GPU server ([Verda API](https://api.verda.com/v1/instance-types)). Its CPU model, memory layout and PCIe width are published nowhere, and, almost certainly, up to seven other tenants share its memory controllers ([machine note §1](../research_notes/MoE%20offload%20system%20race%20plan/race_machine.md)). It suits an optional ≤3-hour, ~$4–6 cross-check with full profiler counters and probably locked clocks (likely in its VM, but unverified), not the headline. RunPod's 5090 comes with only **35 GB of RAM**, too little for gpt-oss-120b ([RunPod pricing](https://www.runpod.io/pricing)). Lambda's public list has no Blackwell consumer GPU ([Lambda pricing](https://lambda.ai/pricing)).

**Develop on the same class of machine.** Use a whole 5090 + 9950X with 126 GB at **$0.51/h** (unverified), or the verified 128 GB host at $0.80/h. The 62 GB hosts cannot hold gpt-oss-120b's 61 GB of pinned experts ([testing note §6](../research_notes/MoE%20offload%20system%20race%20plan/testing_strategy.md)). Destroy the instance between sessions, because idle storage is the top cost risk ([machine note §4](../research_notes/MoE%20offload%20system%20race%20plan/race_machine.md)).

**Who races.** The entrants follow the synthesis's cut ([synthesis D2](../research_notes/MoE%20offload%20system%20race%20plan/synthesis.md); [entrants note Q6–Q7](../research_notes/MoE%20offload%20system%20race%20plan/race_entrants.md)). Every conditional entrant has a hard effort budget. A system that misses it is reported with status "does not run" and its effort log, never silently dropped.

| Entrant | Status | Version and setup | Races on | Effort budget |
|---|---|---|---|---|
| llama.cpp mainline, tuned | **In** (primary baseline) | Frozen at 4da6337 (27 Sep). Best of `-ncmoe K` (first layers) and `-ot` (last layers); `--fit off`; load mode and threads swept; CPU weight repacking on; plus a labelled "as-published" `-ngl` row | All models | 1–2 h |
| ik_llama.cpp | **In** | adce16f. Its defaults, no `-rtr`, explicit `-ot`. The "best llama.cpp family" result is the better of mainline and ik | GGUF models | – |
| Our cache | **In** | Expert-cache patch alone on 4da6337 | gpt-oss-120b, Qwen3 Q4_K_M; Qwen3 BF16 only after a fix (below) | – |
| FreeToken | **In** (the entrant to beat) | v0.1.3 ("released") and main 0d652e7 ("current"); `offload`, `cpu` and `hybrid` modes; explicit cache and KV-cache sizes, because auto-sizing failed on Blackwell under WSL2 ([zenn.dev](https://zenn.dev/lifona/articles/a8606bb95e17e1?locale=en)) | gpt-oss-120b, Qwen3 BF16 | 3–6 h; decide at 6 h |
| KTransformers | Conditional | kt-kernel v0.7.1 + sglang-kt, deferral 0 (exact) | Qwen3 BF16 only | 1 engineer-day. It segfaulted on exactly a 9950X3D + 5090 desktop in issue #1754 ([issue](https://github.com/kvcache-ai/ktransformers/issues/1754)) |
| One community cache | Conditional | #27861 + everett6's fix, or leloch v2 with `--moe-cache on` (its `auto` mode refuses a single GPU) | GGUF models | ≤3 h, with cache allocation verified in the logs |
| Pipelined sharding | Conditional | v2.0.3-mlsys26, `-mva` set to llama.cpp's measured MB | 120b, Qwen3 GGUF | ≤4 h |
| MoE-Infinity | Only if hours remain | main (its code differs from its paper) | – | 4–8 h |
| HybriMoE, 2512.16473, Fiddler | Out | No sm_120 path, obsolete models, or not enough RAM | – | – |
| DALI, SeqMoE, 2606.10493 | Out | No code; quote their numbers only as "reported" | – | – |

## gpt-oss-120b at three equal-memory budgets carries the headline

**Models and formats.** Formats matter because two engines can only be compared on the same weights. **GGUF** is llama.cpp's file format and **safetensors** is Hugging Face's. **MXFP4** and **BF16** are compact 4-bit and 16-bit ways of storing weights. The plan uses five models ([synthesis D3](../research_notes/MoE%20offload%20system%20race%20plan/synthesis.md)):

| Model | Role | Why |
|---|---|---|
| **gpt-oss-120b** (native MXFP4 experts) | Headline | The GGUF and the official checkpoint carry the same MXFP4 expert blocks, so every engine except KTransformers (no gpt-oss support) reads identical expert weights (whether attention precision also matches between the two is still to be checked). It does not fit 32 GB, so offload is natural. Its large 13.25 MB experts make each llama.cpp CPU layer expensive, which gives our cache its biggest margin ([baseline note §5.4](../research_notes/MoE%20offload%20system%20race%20plan/our_system_baseline.md)). The GGUF is 63.4 GB, and our cache pins 61.1 GB of it |
| **Qwen3-30B-A3B-Instruct-2507 in BF16** | Second headline | The only format FreeToken, KTransformers and llama.cpp share exactly ([entrants note Q3](../research_notes/MoE%20offload%20system%20race%20plan/race_entrants.md)). Our cache joins only after its BF16 bug is fixed |
| **Qwen3-30B Q4_K_M** (a 4-bit GGUF) | llama.cpp-family table only | Continuity with the A10 study, and what home users actually run |
| **gpt-oss-20b** | Calibration only | Fits on the GPU entirely, which anchors the bound |
| **Qwen3.6-35B-A3B BF16** | Stretch | FreeToken's home model: 77.1 tok/s at 23.8 GB on a 5090, a probable loss for us. It adds ~140 GB of downloads |

Excluded formats:

- **FP8**, because llama.cpp converts it and doubles the expert bytes;
- **NVFP4**;
- **IQ1_S, IQ2_S and IQ3_S**, which NVIDIA's CUDA compiler (nvcc) 13.2 miscompiles on sm_120 ([PR #28784](https://github.com/ggml-org/llama.cpp/pull/28784));
- **Q8_0** in cross-engine rows.

Total disk is about 285 GB.

**Budgets.** A budget is the **measured peak GPU memory of the whole process tree**, as reported by NVML (NVIDIA's management library), at a fixed maximum context of, say, 4,096 tokens with a 16-bit KV cache. (The KV cache is the model's memory of earlier tokens, and it also occupies VRAM.) Each budget is anchored to a llama.cpp `-ncmoe` step, and every other system is fitted to it by iterative search, the procedure NVIDIA's pipelined-sharding paper used ([methodology note Q1](../research_notes/MoE%20offload%20system%20race%20plan/fair_race_methodology.md)). Our cache's per-layer slots are set so that the total number of GPU experts equals llama.cpp's exactly ([testing note §4.1](../research_notes/MoE%20offload%20system%20race%20plan/testing_strategy.md)).

For gpt-oss-120b (36 layers × 128 experts):

| Budget | llama.cpp setting | Experts on GPU | Our slots per layer | Approx. VRAM |
|---|---|---|---|---|
| Tight (~12.5%) | `-ncmoe 32` | 512 (11%) | ≈14 on average | ≈10–11 GB |
| Medium (25%) | `-ncmoe 27` | 1,152 | 32 | ≈18.5 GB |
| Large (~40–45%) | `-ncmoe 20` | 2,048 (44%) | ≈57 | ≈30.7 GB. Tight on 32 GB; step down one or two layers if it does not fit (FreeToken's community point was 40.4% in ~29 GB) |

For Qwen3 BF16 (48 layers, ~9.4 MB per expert), the same three budgets are `-ncmoe` 42, 36 and 29. All fit 32 GB ([synthesis D4](../research_notes/MoE%20offload%20system%20race%20plan/synthesis.md)).

Two tool defaults must be overridden and logged, because each silently breaks "equal GPU memory" ([methodology note Q1](../research_notes/MoE%20offload%20system%20race%20plan/fair_race_methodology.md), which read the source at 4d86b2f; that commit is the local `expert-cache` branch on 2145525a, not upstream master, but the same default is in upstream 4da6337's [common.h](https://github.com/ggml-org/llama.cpp/blob/4da6337767f973e2b4d0797e5b323d77d8565e4a/common/common.h)):

- llama.cpp's `--fit` is on by default, with a 1 GiB margin (whether `llama-bench` applies it is unverified).
- FreeToken's and SGLang's memory fractions grab most of the card.

**CPU rules and scoring.** Run every system in two CPU regimes on the same 16 physical cores:

- (a) an equal, fixed thread count;
- (b) each system at its own best count from a sweep;
- plus a FreeToken-style 6-thread row, the discipline FreeToken's own paper used.

Every cell then reports:

- absolute tok/s with a 95% confidence interval;
- the 90th and 99th percentiles of the time between tokens;
- the fraction of the **measured** speed-of-light, computed from same-day bandwidth measurements;
- the fraction of the implementation-relative bound.

Two scoring details matter. **FreeToken is scored against the pooled bound** and the per-layer systems against the per-layer bound, since the bound has both variants ([synthesis D4](../research_notes/MoE%20offload%20system%20race%20plan/synthesis.md)). And on a desktop, PCIe copies and CPU helpers read the *same* DRAM. The synthesis proposes adding a shared-DRAM term to the bound, which, by its own (unchecked) reasoning, keeps the bound valid and tightens it there ([synthesis, insights table](../research_notes/MoE%20offload%20system%20race%20plan/synthesis.md)).

**Expected standings.** The synthesis's expectations are arithmetic on A10 constants, not measurements; all of them must be re-derived after the calibration hour on Friday 2 October ([synthesis, "where we can win"](../research_notes/MoE%20offload%20system%20race%20plan/synthesis.md)).

| Opponent | Model and budget | Expectation (uncalibrated) | Confidence | Basis |
|---|---|---|---|---|
| Tuned llama.cpp / ik | gpt-oss-120b, all budgets | **Wide win, ≈2.3–3.7×**: about 45–50 vs 20 tok/s at the tight budget, 100–110 vs 30–35 at the large one | Medium-high | 1.52–1.61× on the A10 (both text sets); margins grow as host bandwidth falls. Static placement runs every selected expert of a CPU layer, while the cache runs only misses |
| Tuned llama.cpp / ik | Qwen3 Q4_K_M | Win, ≈1.3–1.6× | Medium | A10 1.36–1.49× (the 1.66× top was Q8_0); 5090 projection at 250 GB/s 1.29–1.60× |
| Community caches | Qwen3, 120b | Likely win | Medium-low | leloch v2 gains only 2.1% on Qwen3-30B |
| Pipelined sharding | 120b, Qwen3 GGUF | Win at small budgets; tie possible near 32 GB | Medium | Old b6097 base |
| KTransformers (deferral 0) | Qwen3 BF16 | Toss-up | Low | SeqMoE measured it at 34.2 vs llama.cpp's 30.9 tok/s on a 4090 ([arXiv 2609.12978](https://arxiv.org/abs/2609.12978)) |
| **FreeToken** | 120b at ~40% | **Loss by ~10–30%; tie possible** | Medium-low | 127.1 tok/s community figure (FreeToken v0.1.2; host CPU and memory bandwidth unstated) vs ≈95–110 estimated for ours |
| FreeToken `hybrid` | 120b at 12.5–25% | Loss or tie | Low | It serves *current* misses by fetch and CPU at once; our admissions pay off two tokens later |
| FreeToken | Qwen3.6 BF16 | Likely loss | Medium | Projected 0.78–1.02× even against llama.cpp (server-host scenarios) |
| Speed-of-light | All | Everyone far below it | High | Consistent with the audit's finding that the median published system is far from the bound |

## A Friday gate, then a sprint worth an estimated 10–20%

**Gate G0: does our system build and give the same answers?** G0 is checked at the end of **Thursday 1 October** on the development host ([synthesis D6](../research_notes/MoE%20offload%20system%20race%20plan/synthesis.md)). Passing needs all four of these:

- the correctness tests pass on the rented card (details in the next section);
- the cache runs under `llama-server`, where it has never been measured (only under ec-bench);
- on gpt-oss-20b own-text prompts, a **teacher-forced** comparison (both systems score the same fixed text, position by position) shows **top-1 agreement with stock llama.cpp of at least 0.98**;
- the same comparison shows **|ΔNLL| of at most 1.0%**. The A10 levels were 0.981–0.993 and −0.99% to +1.06%.

If it still fails after 1.5 days of fixing, drop system work and race without our cache.

**Gate G1: where do we stand?** G1 runs on **Friday 2 October**: about 5 hours on a race-class host, about $4–5. The protocol:

- gpt-oss-120b on own-text prompts;
- all three budgets;
- each system at its best thread count;
- at least 3 fresh launches × 5 requests × 256 tokens, interleaved.

The same session runs the calibration hour:

- read-bandwidth and pinned-copy sweeps;
- an all-GPU gpt-oss-20b run under nsys, NVIDIA's timeline profiler;
- a CPU helper thread sweep;
- two cache runs with statistics, which give O;
- an `-ncmoe` sweep;
- recording `asyncEngineCount`, the GPU's number of copy engines.

The gate uses two ratios. **R_ll** is our tok/s divided by the better of mainline and ik; **R_ft** is ours divided by FreeToken's best mode.

| Outcome | Condition | Action |
|---|---|---|
| **GO-FULL** | R_ll ≥ 1.25 at all three budgets **and** R_ft ≥ 0.85 at one budget or more | Full sprint, aimed at FreeToken |
| **GO-LIMITED** | R_ll ≥ 1.25 at two or more budgets, R_ft < 0.85 everywhere | At most 3 sprint days, on O and f only. The saved days go to the protocol and the paper. Frame the result as "fastest llama.cpp-family, X% of SoL; FreeToken leads" |
| **NO-GO** | R_ll < 1.10 at two or more budgets, **or** gpt-oss-120b own-text top-1 < 0.98 | Stop system work. Spend the week on competitor replication, the harness, fidelity and writing. Our cache stays in as a measured entrant |
| FreeToken not running after 6 h | – | Decide on R_ll alone. Give FreeToken at most one more day, log its status and email its authors |

The thresholds have reasons ([synthesis D6](../research_notes/MoE%20offload%20system%20race%20plan/synthesis.md)):

- **0.85:** the sprint's planning gain is +10–20%, so only a gap of 15% or less is closable.
- **1.25:** the A10's lowest instance-level ratio was 1.31, and every projection for a slower host is higher.
- **1.10:** roughly baseline noise plus the effect of upstream kernel fusions that probably do not fire on cached layers.

The table does not cover every outcome. For example, an R_ll between 1.10 and 1.25 at two or more budgets triggers no row, and neither does R_ll ≥ 1.25 at exactly two budgets with R_ft ≥ 0.85 at some budget. I recommend treating the first case as GO-LIMITED, *after* re-auditing the baseline tuning and the host, because the projections for slower hosts and the synthesis's desktop estimate all put the margin wider on this machine; the second needs its own rule in the pre-registration.

**Gate G2: freeze.** A pre-race gate on **Thursday 8 October** freezes the patch ([testing note §6](../research_notes/MoE%20offload%20system%20race%20plan/testing_strategy.md)):

- noise-floor and equivalence checks on both models;
- a free-running output check;
- a robustness battery (timeout, lifecycle, long context);
- six A/B pairs of the final stack against the version frozen at the start of the week.

**What to optimise, in order.** The ordering comes from the A10 decomposition and a review of fixes in the literature ([synthesis D7](../research_notes/MoE%20offload%20system%20race%20plan/synthesis.md); [headroom note Q6](../research_notes/MoE%20offload%20system%20race%20plan/headroom_techniques.md)). The gains are expected to shrink on a desktop, because slow DRAM makes CPU miss work dominate the step.

| Order | Change | Expected gain on the desktop (estimate) | Effort | Condition |
|---|---|---|---|---|
| 0 | **Threads and placement** for everyone. For us: one helper per physical core across both CCDs (the Ryzen's two 8-core chiplets), no SMT siblings (the second hardware thread of a core), explicit pinning. Today's default of hardware threads − 2 gives 30 helpers on 16 cores | 0–15% | Hours | Before G1 |
| 1 | **Profile first.** nsys with graph-node tracing, plus two controls: a static cache with no admissions, and one layer holding all but one expert | Decides items 3–7 | 0.5 day | Always |
| 2 | **Retune the admission policy in simulation** for r* ≈ 1.1–1.3 (the A10 value was ≈11) | 0–5% | Hours, $0 | Always |
| 3 | **Host side of O.** Take the policy loop off the critical path (its copies land two tokens later anyway); cache device pointers; one launch per admitted expert rather than per tensor; skip unchanged map uploads | 2–6% | 1 day | If nsys shows host gaps |
| 4 | **Cut f.** Merge the helpers' four barrier phases: quantise once, fuse gate+up across the request's experts, use per-expert completion counters before down, write weighted sums straight into the mailbox. Target ≤5–10 µs, down from ~50 µs | 3–7% | 1.5–2 days | Always |
| 5 | **GPU side of O.** Fold the lookup and request kernels into the router kernel, merge the wait into the weighted sum, and let upstream's fusions (#25952, #28432) fire on cached layers, where they probably do not today ([ecosystem note Q1](../research_notes/MoE%20offload%20system%20race%20plan/llamacpp_ecosystem.md)) | 3–8% | 2–3 days | If nsys shows cache kernels dominate O |
| 6 | **Admission versus DRAM/copy contention.** Pause admissions while a layer's helpers run (SeqMoE's pause flag); chunk fills to 1–4 MB | 0–5% | 1–2 days | If nsys shows interference |
| 7 | **Run one miss per layer on the GPU** straight from pinned memory while the helpers do the rest, FreeToken-hybrid style | 0–20% at small budgets, ≈0 at large | 2–3 days | Replaces item 5, only if G1's gap to FreeToken sits at small budgets and the helpers fall clearly short of the measured read bandwidth |

Three things are ruled out for this week:

- **Prediction and prefetch:** 0–15% in the literature, over 3–5 days. Most of the remaining cache gap needs future knowledge that learned predictors have not delivered ([arXiv 2608.07911](https://arxiv.org/abs/2608.07911)).
- **Persistent megakernels:** weeks of work ([MPK](https://arxiv.org/abs/2512.22219)).
- **KTransformers-style Expert Deferral:** it changes outputs.

The schedule is:

| Sprint day | Items |
|---|---|
| S1 | Items 1–2 |
| S2 | Item 3 |
| S3–S4 | Item 4 |
| S5 | Item 5, 6 or 7, as nsys and G1 dictate |
| S6 | Freeze and G2 |
| S7 | Buffer |

The combined planning figure is **+10–20%** on the desktop ([synthesis D7](../research_notes/MoE%20offload%20system%20race%20plan/synthesis.md)).

## Tests define "the same answer" and "a fair race" before any timing

**Four levels of "exact", fixed in advance.** Bit-identical output against stock llama.cpp is the wrong bar. Where an expert runs changes the arithmetic by construction: the CPU rounds activations to one 8-bit format (Q8_0), the GPU to another (Q8_1) ([methodology note Q2](../research_notes/MoE%20offload%20system%20race%20plan/fair_race_methodology.md)). Even provably lossless methods reproduce full greedy outputs in only **43–45% of prompts in BF16** ([Orthrus, arXiv 2609.15504](https://arxiv.org/html/2609.15504v1)). The testing plan therefore uses four classes ([testing note §2.1](../research_notes/MoE%20offload%20system%20race%20plan/testing_strategy.md)):

| Class | Meaning | Where it applies | Pass threshold |
|---|---|---|---|
| E0 bit-exact | Identical outputs, step by step | Same binary run twice; any change claimed not to alter arithmetic | 100% identical |
| E1 kernel-exact | A GPU operation matches the CPU reference | Operation tests | Normalised error ≤ 5e-4; rows for "missing" experts exactly 0.0 |
| E2 placement-exact | Within a small multiple of the measured **noise floor**: the spread among harmless llama.cpp reconfigurations (two `-ncmoe` splits, the MoE reduction fusion off, all experts on the CPU) | Ours versus stock, and cross-engine rows | Mean and 99.9th-percentile **KLD** (Kullback–Leibler divergence, how far two next-token probability distributions differ) ≤ 3× the floor. Top-1 disagreement ≤ floor + 0.5 points. \|ΔNLL\| ≤ max(1.5× floor, 0.1%). ≥ 90% of free-running divergences at near-ties |
| E3 lossy | Anything worse, e.g. Q8-level KLD of ~1e-3 | KTransformers with deferral, quantised variants | Reported only as a labelled lossy row |

**What reading the code found, and which tests run first.**

- **BF16 is the one race format the cache has never run, and it is unguarded.** Negative expert ids, the patch's marker for "this expert runs on the CPU", are handled only in the quantised GPU kernel. On NVIDIA, BF16 experts go to other kernels:
  - one leaves those rows unwritten;
  - a fused one reads out of bounds;
  - a fallback path aborts on an assertion.

  All A10 runs used quantised types. The first GPU test (**T01**) therefore runs the operation matrix with BF16, "every expert missing" and fused cases under NVIDIA's memory checker ([testing note §0, §2.2](../research_notes/MoE%20offload%20system%20race%20plan/testing_strategy.md)).
- **A suspected race in the mailbox**, found by reading and not reproduced. A helper that is descheduled between reading the request number and reading the payload can compute stale inputs for the next request. That would give **silent, non-deterministic wrong output**. **T06** is a CPU harness that settles it: a fake GPU thread, 1–64 helpers, delays injected at six points, ThreadSanitizer, and 10⁶ requests checked against a single-threaded answer. The candidate fix is to bail out if the response number already matches after reading the payload.
- **Silent fallbacks can swap in a different system with no error.**
  - A GGUF with merged gate/up tensors loses mailbox mode.
  - A static policy without an initial set leaves the cache **empty**, which was verified on CPU.
  - Unreadable configuration files are ignored.

  An identity gate (**T10**) on every statistics file is therefore mandatory. It checks mailbox on, helpers as intended, paced mode off, per-layer C as intended, and the counter invariants. This is the lesson of the #27861 no-op.
- **The tools cannot yet do what the race needs.**
  - The benchmark dumps only the top-5 logits, so KLD cannot be computed.
  - It times only teacher-forced decode.
  - It inflates VRAM through its batch setting.
  - Nothing samples GPU memory.
  - The timeout is converted assuming a 2 GHz GPU clock, so the default 2,000 ms becomes about 1.4 s on a 5090.
- **Good news from the free CPU tier.** On tiny test models, the cache gave **bit-identical** outputs to cache-off in 14 runs. The AVX-512 MXFP4 kernel differs from llama.cpp's by at most 2.0e-8 of the dot product's magnitude. All 100 existing statistics files satisfy the counter invariants exactly ([testing note, Appendix A](../research_notes/MoE%20offload%20system%20race%20plan/testing_strategy.md)).

Two more gates guard the baseline and the admission path. **T13** checks that the patched build with the cache off equals unpatched llama.cpp at the same commit: bit-identical output and speed within ±1%. **T05**, a "poison canary" that fills evicted slots with NaN, becomes mandatory for any change that moves admissions off the critical path. Such a change reopens the class of bug in which #27861 remapped the cache while a GPU graph still read it ([testing note §3.2](../research_notes/MoE%20offload%20system%20race%20plan/testing_strategy.md)).

**How each sprint change is accepted.** Every optimisation lands behind an off-by-default switch, so the old and new versions (A and B) run interleaved inside one process on one loaded model, in the order W;A;B;B;A;A;B;B;A;A;B;B;A (six A/B pairs). W is a discarded warm-up, because the first configuration in a process ran **1.0–2.2% slow** in the A10 repeats.

- **Blocks and rate.** A block is 8 sequences × 192 tokens. The rate is total tokens over total time (ΣN/ΣT), which is the harmonic mean that Hoefler and Belli's rules require for rates ([Hoefler & Belli](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf)).
- **Confidence intervals.** They come from a paired bootstrap: resampling the measured blocks to see how much the result could move.
- **Resolution.** Six pairs give about ±1% and detect a 2% effect.
- **Stop rule.** After 6 pairs, accept if every primary configuration's 99% interval sits above 1.01, every guard's sits above 0.98, and the exactness gate passes. Reject if every primary's 95% interval lies below 1.02, or any guard's lies below 0.98. Otherwise extend once to 12 pairs.
- **Invalid blocks.** A block is thrown out if the GPU throttles for more than 5% of it, if the read-bandwidth probe drifts more than ±5% from the session median, or if a foreign process uses more than 5% CPU ([testing note §4](../research_notes/MoE%20offload%20system%20race%20plan/testing_strategy.md)).

**What makes the race fair.** Thirteen recorded checks cover:

- equal measured GPU memory, sampled at 20 Hz and required between the budget and the budget minus max(256 MiB, 2%);
- the same cores for everyone, audited every 100 ms;
- identical token IDs in;
- exactly 256 tokens out, greedy, end-of-text ignored;
- matched formats with weight hashes;
- cold and warm results for the dynamic caches;
- randomised, interleaved order with ≥3 launches per point;
- bandwidth probes at the start, before each block and at the end;
- a GPU health check;
- equal or greater tuning effort for the baselines;
- an attempt to reproduce each competitor's own published number (below);
- a harness self-check;
- a complete status block per entrant ([testing note §5](../research_notes/MoE%20offload%20system%20race%20plan/testing_strategy.md)).

Five traps in current tools need explicit handling ([synthesis, finding 8](../research_notes/MoE%20offload%20system%20race%20plan/synthesis.md); [#28334](https://github.com/ggml-org/llama.cpp/commit/14a9d09f75683c94c2c4f229efe54670d4209089)):

- `--no-mmap` was deprecated on 9 September in favour of `--load-mode none`;
- `--fit` is on by default (unverified for `llama-bench`);
- ik's `-ncmoe` now offloads the *last* layers;
- leloch's `auto` mode refuses one GPU;
- shipped #27861 does nothing.

Each competitor must also try to reproduce its own published number, under a tolerance fixed in advance:

| Entrant | Number to reproduce | Tolerance |
|---|---|---|
| FreeToken | 127.1 tok/s, gpt-oss-120b, `offload` mode, ~29 GB on a 5090 | 0.80–1.25× |
| Pipelined sharding | Its speed-up over its own llama.cpp baseline on Qwen3-30B Q4_0 at 8 and 16 GB | Within 90% (its own artifact rule) |
| Community cache | Its reported gain in the author's configuration | Sign and size within ±5 points |
| llama.cpp mainline | gpt-oss-20b all-GPU at 419 tok/s on a 5090 | At least 0.85× (a GPU health check) |

Each competitor then gets one of the methodology note's four statuses: reproduced, runs but not reproduced, does not run, or not runnable, with "reproduced" kept, as in ACM's usage, for runs of the authors' own artifact and "replicated" used otherwise ([methodology note Q4](../research_notes/MoE%20offload%20system%20race%20plan/fair_race_methodology.md); [ACM badging](https://www.acm.org/publications/policies/artifact-review-and-badging-current)). Testing alone costs about **19–22 GPU-hours and $14–19**, roughly half the plan ([testing note §6](../research_notes/MoE%20offload%20system%20race%20plan/testing_strategy.md)).

## The calendar: about 41–42 GPU-hours and $33–42 by 27 October

The schedule merges the synthesis's timeline with the testing plan's tiers. The testing note placed the race on 8–9 October; the synthesis's later date keeps the full sprint week. Dollar figures use the cheapest qualifying hosts: **$0.51/h** for development and **$0.735–0.936/h** for race-class work ([machine note §4](../research_notes/MoE%20offload%20system%20race%20plan/race_machine.md)).

| Date | Work | Where | GPU h | $ |
|---|---|---|---|---|
| **Tue 29 Sep** | Fix the LM-head omission; regenerate the case-study table, macros and abstract by script; add the regression test. Free CPU tests (tiny BF16/MXFP4 models, T06 mailbox harness, analysis tests). Pre-registration and harness skeletons. Open and fund Vast.ai. Ask a cs.DC arXiv endorser ([application plan](../apply/APPLICATION_PLAN.md)). Settle the audit-extension conflict | Local | 0 | 0 |
| **Wed 30 Sep** | Build ours, mainline and ik on 4da6337 for sm_120 (CUDA 12.8/12.9, arch `120a`). T01 with BF16 first; T13 patch-off identity. Check that nsys and NVML work in the container. Start downloads | Dev host | 5 | 2.6 |
| **Thu 1 Oct** | BF16 fix, or drop ours from the BF16 table; raise the timeout; explicit helper pinning; run under `llama-server`. **G0.** FreeToken install (CUDA 13 venv); KTransformers source-build attempt | Dev host | 5 | 2.6 |
| **Fri 2 Oct** | Qualify 2–3 race-class hosts (15 min each). Calibration hour. **G1** on gpt-oss-120b × 3 budgets. Re-run the projection with measured constants. Decide GO-FULL, GO-LIMITED or NO-GO; decide MLSys | Race-class host | 5.5 | 4.0–5.1 |
| **Sat 3 Oct** | S1: nsys profile with controls; policy retune in simulation | Dev host | 2.5 | 1.3 |
| **Sun 4 Oct** | S2: host side of O (T05 canary if admissions move) | Dev host | 2 | 1.0 |
| **Mon 5 – Tue 6 Oct** | S3–S4: merge the helper phases (f); rerun T06; A/B pairs | Dev host | 4 | 2.0 |
| **Wed 7 Oct** | S5: GPU-side O, contention, or GPU-executed misses, as nsys and G1 dictate | Dev host | 2 | 1.0 |
| **Thu 8 Oct** | S6: freeze the patch; **G2** (noise floor, equivalence, robustness, final vs frozen v0) | Dev host | 3 | 1.5 |
| **Fri 9 Oct** | S7 buffer; dry install of every competitor (FreeToken, KTransformers, pipelined sharding, community cache) | Dev host | 2 | 1.0 |
| **Sat 10 Oct** | Re-run the projection; **tag the pre-registration** (protocol, budgets, token IDs, thresholds, predictions, analysis script, seed). Re-query Vast offers; destroy the dev host | Local | 0 | 0 |
| **Sun 11 Oct** | **Race**, one continuous session: qualification, builds overlapped with ~285 GB of downloads, weight hashes, noise floor, ~30 randomised points × 3 launches, reproductions, end-of-session repeat | Race host | 10 | 7.4–9.4 |
| **Mon 12 Oct** | Spill-over and final repeats; destroy the host | Race host | 0–1 | 0–0.9 |
| **Tue 13 Oct** | Scoring (ΣN/ΣT, bootstrap intervals, fractions of both bounds), status blocks. **Email FreeToken, KTransformers and NVIDIA authors with logs** | Local | 0 | 0 |
| **Wed 14 – Fri 16 Oct** | Race section and corrected case study. **Fri 16 Oct checkpoint:** if analysis is late, post without the race and include the registered plan | Local | 0 | 0 |
| **Sat 17 – Tue 20 Oct** | Reserve: one re-run for an author-supplied configuration or failed cells, only if needed | Race host | 0–4 | 0–3.7 |
| **Wed 21 – Thu 22 Oct** | Final paper; **submit to arXiv**, leaving moderation slack before the 27th | Local | 0 | 0 |
| **Fri 23 – Mon 26 Oct** | Fellowship materials (motivation letter, CV, video); fold in author replies for a later v2 | Local | 0 | 0 |
| **Tue 27 Oct** | **ETH AI Center fellowship deadline, 16:00 CET = 10:00 in Milwaukee.** US clocks change only on 1 November, so the application plan's "09:00" is an hour early ([application plan](../apply/APPLICATION_PLAN.md); [ETH FAQ](https://ai.ethz.ch/research/phd-and-postdoc-programs/fellowship-faqs.html)) | – | – | – |

**What it adds up to.**

| Item | GPU hours | Cost |
|---|---|---|
| Development host | ≈25.5 | ≈$13 |
| Race-class host | ≈15.5–16.5 | ≈$11.4–15.4 |
| **Total GPU time** | **≈41–42** | **$24–28** |
| Disks and downloads (every new host re-downloads ~130–285 GB) | – | $6–10 |
| Buffer | – | $3–4 |
| **Plan total** | – | **≈$33–42** |

That overlaps the $25–40 target, but its top end exceeds the target by about $2 ([synthesis, schedule](../research_notes/MoE%20offload%20system%20race%20plan/synthesis.md)). Optional additions:

| Addition | Extra cost |
|---|---|
| The verified $0.80/h development host, if the $0.51 one fails qualification | About $8 |
| The reserve session | $3–4 |
| A Verda profiling check (clock locking likely but unverified) | $4–6 |

The worst case is about $60, still well under the $100 cap on its own. The cheapest saving is kernel-only work on a smaller Blackwell card for the first ~15 development hours: a $0.31–0.39/h RTX 5080 cuts only about $2–3 against the $0.51/h host, and a $0.11–0.18/h RTX 5060 Ti slice about $5–6. The machine note's ~$4–8 assumed a $0.51–0.60/h development host ([machine note §4](../research_notes/MoE%20offload%20system%20race%20plan/race_machine.md)).

**The real constraint is Harshith's time, not money.** The audit-extension plan, written earlier the same day, schedules its own measurements for **5–14 October**. At core scope it costs about $29–40 (GPU time plus storage and transfer) and **~74 hours of his work, about 10 working days** spread from 28 September to 21 October ([audit measurement plan](MoE%20audit%20measurement%20plan.md)). Neither plan budgets for the other. Together they would cost $62–82 at core scope, and up to about $100 with every option C extra (more if the audit plan draws on its own ~$60 reserve), and would ask for two full-time weeks of builds and debugging compressed into one. They overlap usefully in two places, the FreeToken and pipelined-sharding installs and the measurement harness, so running both at full size is not impossible. But the priority has to be set on 29 September, not discovered on 5 October.

## Most risks are toolchain and fairness failures, not physics

The risk register draws on the synthesis and testing notes ([synthesis, risks](../research_notes/MoE%20offload%20system%20race%20plan/synthesis.md); [testing note §0](../research_notes/MoE%20offload%20system%20race%20plan/testing_strategy.md)).

| Risk | Evidence | Mitigation |
|---|---|---|
| The LM-head bug reaches a submission | Confirmed in `sol_a10.py` and `sol_windows.py` | Fix on 29 Sep, add a regression test, regenerate by script |
| Our patch breaks on Blackwell | Unguarded BF16 kernels; timeout assumes 2 GHz; spin kernel inside CUDA graphs (recorded GPU work replayed with less overhead); default pinning pairs helpers on shared cores; a suspected mailbox race | T01 on the card first; BF16 fix or drop; raise the timeout; explicit pinning; T06 harness. Stay on MXFP4/Q4_K where the id handling is patched |
| Race-host supply dries up or the contract ends mid-race | Only 3 whole-machine 5090 offers with ≥180 GB, two of them Zen 5; the best caps rentals at 3 days ([machine note §2](../research_notes/MoE%20offload%20system%20race%20plan/race_machine.md)) | Rent only for G1 and the race; re-target by machine ID; ranked fallbacks; develop on the same class |
| Desktop regime uncalibrated, so projections mislead | Projections cover only 150–400 GB/s hosts | Calibration hour at G1; re-run and pre-register the projection before racing |
| FreeToken fails to install or run | Install hash mismatch ([#554](https://github.com/FlashML-org/FreeToken/issues/554)); an auto-sizing failure on Blackwell under WSL2; an unverified risk that its handshake probe falls back to slower callbacks inside a container | Pinned versions; separate CUDA 13 environment; explicit sizes; log the handshake mode; 6 hours plus at most one more day, then status "does not run" and an email to the authors |
| KTransformers segfaults on a one-socket desktop | Issue #1754 on exactly a 9950X3D + 5090 | `--kt-threadpool-count 1`, then a local `worker_pool.h` patch; torch 2.9.1 environment; 1-day cap |
| Accusations of untuned baselines | Agrawal et al.'s anti-pattern "neglecting parameter tuning" ([arXiv 2507.09019](https://arxiv.org/html/2507.09019)); flags moved llama.cpp by up to 13% on the A10 | Tune llama.cpp and ik harder than ours; log tuning hours; add "as-published" rows |
| Noise without clock locking | llama.cpp moved −7.8% to +1.6% between A10 instances | Interleave; probe and log clocks per block; ≥3 launches × ≥5 requests; intervals of ±2–3% |
| Disputes over "same output" | Upstream says its fused MoE reduction is "not claimed bit-identical" ([#25952](https://github.com/ggml-org/llama.cpp/commit/3466812d1f06728effe7c0f3c0671117f461672d)) | E2 against a pre-registered noise-floor multiple; a baseline run with that fusion off |
| Appearance of favouring his own system, or overclaiming novelty | FreeToken's handshake predates the patch | Public pre-registration tag before renting; state the conflict of interest; the novelty wording below |
| Upstream changes collide mid-plan | Open PRs #26167, #29184 and #29181 touch the same files ([ecosystem note Q1](../research_notes/MoE%20offload%20system%20race%20plan/llamacpp_ecosystem.md)) | Freeze every llama.cpp-family entrant at 4da6337 |
| The container blocks profiling or 61 GB of pinned memory | Untested on Vast | Check on 30 Sep; use Verda for GPU counters if needed |

## Frame a loss to FreeToken as the yardstick working

The framing should be fixed before the race, so a loss reads as a tested prediction rather than an excuse. Pre-register the r* argument, "fetch-plus-pooled designs win where PCIe ≈ DRAM", together with the projected standings. If FreeToken then wins on the desktop, the bound has predicted the winner ([synthesis D9](../research_notes/MoE%20offload%20system%20race%20plan/synthesis.md)).

| Outcome | Definition | Headline | Must also say |
|---|---|---|---|
| **Win** | Ours fastest at equal measured VRAM on ≥2 of 3 budgets for gpt-oss-120b, with non-overlapping intervals | "On a 5090 + Ryzen desktop (measured X GB/s DRAM, Y GB/s PCIe), at equal GPU memory, our cache decodes gpt-oss-120b at A tok/s, B% of the measured speed-of-light, against FreeToken's C and tuned llama.cpp's D" | Credit the win to measured implementation choices through ablations (the per-layer skip, lower O and f); cite FreeToken's handshake; scope it to this host class |
| **Tie** | Intervals overlap, or within ±5% | "Two independent designs land within x% of each other at ~B% of the bound: a Python engine with a pooled cache and fetch, and a C++ llama.cpp per-layer cache with CPU misses" | Evidence *for* the bound: strong systems cluster at the same fraction; the remainder is host DRAM plus per-layer fixed cost. This echoes the audit's finding that published 1.5–2.3× gains mostly reflect weak baselines |
| **Loss to FreeToken** (likely) | FreeToken faster on ≥2 budgets | "FreeToken is the fastest system measured, at B% of SoL; ours is the fastest llama.cpp-family system, at x× tuned llama.cpp" | Use the bound to decompose the gap: hit rate (pooled vs per-layer), O, f, and fetch-versus-CPU at PCIe ≈ DRAM. Say whether FreeToken's own numbers replicated |
| **Loss to llama.cpp** (caught at G1) | R_ll < 1.10 | State it plainly | The cache's fixed costs exceed its hit-rate savings on this host; drop it from the headline |

On novelty, the synthesis proposes this wording for the paper and application, which should replace every "novel" in the abstract, the case study and the upstream draft ([synthesis D10](../research_notes/MoE%20offload%20system%20race%20plan/synthesis.md)):

> "Our system uses no new mechanism. GPU expert caches with CPU-executed misses appear in HybriMoE, 2512.16473, DALI, KTransformers, FreeToken and three llama.cpp contributors' branches; FreeToken has shipped a GPU-signalled, polled CPU hand-off since 11 August 2026; decayed-frequency admission is in llama.cpp PRs #26563/#26824; zero-copy loads from mapped host memory are in SeqMoE. We race it as one entrant under the same protocol as the others. Three implementation choices differ from FreeToken's […]; we report their measured effect where we ablated them and claim nothing where we did not."

What remains claimable:

- the time-domain bound, the validated model and the audit;
- the race protocol;
- the measured standings, with the bound's explanation of them.

Each audience gets a different entry point:

- **SPCL reviewers** (Torsten Hoefler's lab at ETH Zürich): lead with the bound, the pre-registration and the confidence intervals. Hoefler and Belli's rule 11 is "show upper performance bounds", and their survey found only **2 of 95 papers** reporting confidence intervals ([Hoefler & Belli](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf)).
- **People running models at home:** a recipe table of which engine and flags to use at which VRAM.
- **llama.cpp maintainers:** tuned recipes and any bugs found, not the patch. Maintainers have asked for small pieces, and CONTRIBUTING limits new contributors to one open PR, with AI use disclosed ([CONTRIBUTING.md](https://github.com/ggml-org/llama.cpp/blob/4da6337767f973e2b4d0797e5b323d77d8565e4a/CONTRIBUTING.md)).
- **Competitors:** a status block per system, with the "not reproduced under stated conditions" wording. NeurIPS's reproducibility track calls a careful failure to reproduce "a genuine contribution" ([MLRC 2026](https://blog.neurips.cc/2026/05/04/mlrc-2026-reproducibility-as-an-official-track-at-neurips/)).

## Twelve decisions only Harshith can make

| # | Decision | What rides on it | Suggested default | Decide by |
|---|---|---|---|---|
| 1 | **Open and fund a Vast.ai account**, and accept a container with root but no GPU clock locking | Without it the recommended machine is out of reach. RunPod's 5090 has 35 GB of RAM, Lambda has no Blackwell consumer GPU, and Verda's card is a shared slice. The fallback is RunPod's PRO 6000 at $2.09/h with an unknown CPU. Opening needs email verification and a $5 minimum credit ([Vast quickstart](https://docs.vast.ai/guides/get-started/quickstart)) | Open it | Tue 29 Sep |
| 2 | **Is he "leloch"**, author of llama.cpp RFC #24528 (12 Jun 2026)? | If yes, leloch's cache is his own earlier work: self-cite it, and the community-cache entrant becomes a self-comparison. If no, it is third-party prior art older than the patch ([Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)) | State it explicitly in the paper | Before the 13 Oct emails |
| 3 | **Qwen3.6-35B-A3B in or out** | In is the honest replication of FreeToken's audited row, a probable loss for us, +140 GB of downloads and 1–2 race hours. Ours joins only if its GGUF keeps separate gate and up tensors | Stretch: FreeToken plus tuned llama.cpp only, if race hours remain | 10 Oct (pre-registration) |
| 4 | **MLSys 2027 (30 Oct) in scope?** | The novelty check found MLSys reachable only with a KTransformers comparison, a host-callback ablation, confidence intervals and ideally a second platform. The ablation competes with the sprint ([novelty check](MoE%20hybrid%20decode%20novelty%20check.md); [MLSys 2027](https://mlsys.org/Conferences/2027)) | Out by default; revisit at the 16 Oct checkpoint | 2 Oct, after G1 |
| 5 | **Money** | ≈$33–42 core, up to ~$60 with every option; renting the race-class host twice (gate and race) | Approve up to $50 for option C | 29 Sep |
| 6 | **Role of the race in the paper** | The paper currently keeps the system "only as a system under test" | A new race section; the bound stays the headline | 29 Sep |
| 7 | **Contact competitor authors** on ~13 Oct under his own name ("Independent Researcher") and hold the arXiv post until ~22 Oct | No published norm sets a waiting period. The methodology note recommends 1–2 weeks and an invitation to supply a better configuration, mirroring artifact evaluators' practice of giving authors "a chance to fix issues" ([sysartifacts guide](https://sysartifacts.github.io/evaluator-guide.html); [methodology note Q4](../research_notes/MoE%20offload%20system%20race%20plan/fair_race_methodology.md)) | Yes | 13 Oct |
| 8 | **Public pre-registration before the race** | Reveals the plan and the predicted loss; proves the order of events | Public tag on 10 Oct | 10 Oct |
| 9 | **Upstream before 27 Oct?** | AI use must be disclosed; one open PR for new contributors | Tuned recipes and small fixes only; an RFC is optional | 20 Oct |
| 10 | **Base commit**: 4da6337 (current; the patch applies cleanly but has never been built there) or 2145525a (A10 continuity) | Fairness needs one frozen base for every llama.cpp-family entrant | 4da6337 | 30 Sep |
| 11 | **Hands-on hours per day** | FreeToken, KTransformers and Blackwell toolchain debugging are the time sinks | Tell the plan; it sets how many conditional entrants enter | 29 Sep |
| 12 | **Option C versus the audit extension** | Same 5–14 Oct window; together $62–82 at core scope and about two weeks of work in one | Pick one as primary; if both, share the FreeToken/pipelined-sharding installs and the harness | 29 Sep |

## Conclusion

The race is worth running mainly because it can go wrong for the system and still come out right for the paper. The bound says that on a desktop, where the PCIe link is about as fast as main memory, fetching experts should beat computing misses on the CPU. That is a falsifiable prediction about a named competitor on named hardware, made before the measurement. If FreeToken wins by the margin the bound predicts, that is exactly the kind of evidence a benchmarking-minded group trusts. If our cache wins, the ablations say why. The one outcome that hurts is an unregistered race, whose loss reads as defeat and whose win reads as a tuned harness.

Two points were not visible before this synthesis. First, the single most valuable hour this week costs nothing: the LM-head fix raises every system's fraction of the bound and removes an error a reviewer would find in minutes. Second, the binding constraint is not the $100 cap but one person's fortnight. Option C and the audit extension each fit the money, but not both in the same ten days. Choosing between them on 29 September matters more to the 27 October application than any single optimisation in the sprint.
