# Rent the papers' servers, not their GPUs

For the next extension of the audit, plan about **$35 of pre-registered cloud measurements, with the other ~$65 held in reserve, rather than trying to match every paper's GPU**. The uncertainty is concentrated in a few places. Twelve verdicts in 8 of the audit's 22 judged rows change when the predicted llama.cpp speed is scaled across its own error range (×0.83 to ×1.22). **Nine of those twelve belong to KTransformers rows, and they all turn on one question: does llama.cpp actually use both CPU chips of a two-socket server?** Published llama.cpp data say a second socket adds only 3–11 % for DeepSeek-style models. If that holds, KTransformers' weak-baseline flags clear and its gains survive, and a roughly $5 test on any whole two-socket rental can confirm the direction. The other decisive targets are SeqMoE's RTX 5090 and RTX 4090 rows and two knife-edge rows. RTX 5090 and 4090 time on DDR5 server hosts (about $13–18 in the plan below) measures all of them, and pipelined sharding, FreeToken and the CPU-GPU collaborative system can be raced head-to-head against llama.cpp on the same machines. The host decides what "the right equipment" is, not the GPU. **All 22 judged rows ran on multi-channel server or workstation memory (115–614 GB/s on paper), none on a two-channel desktop.** That contradicts a sentence in the paper's limitations, and it rules out RunPod (35–41 GB of RAM on its 4090/5090 pods) and Lambda (no consumer GPUs) as the main venues. Vast.ai shows CPU, RAM and PCIe before you rent and lists DDR5 EPYC and Threadripper PRO machines at $0.43–0.65/h. It has almost none of the Intel Sapphire Rapids Xeons (the 2023 server generation) the papers used, though, so a rule for carrying results across hosts must be written down before measuring. Judging each baseline against the llama.cpp of its paper's date splits the 16 weak baselines. Nine come from papers written after llama.cpp gained per-expert placement and are weak by their own era's standard. Seven (HybriMoE, KTransformers) were written around or before that change and need a two-level label. If you pre-register this week, measure 5–14 October and write up by about 21 October, the results fit before the 27 October deadline.

## Twelve verdicts, each with a known tipping point

The audit asks one question of each published speed-up. What would **llama.cpp** (the most widely used open-source engine for running models on consumer hardware) have reached on the paper's own machine, if given the same GPU memory and set up properly? The models are **mixture-of-experts (MoE)** models. They are built from many small sub-networks called "experts", and each generated token uses only a few of them. The experts that do not fit in GPU memory can therefore live in the computer's ordinary memory (host RAM) and run on the CPU. llama.cpp's option **`--n-cpu-moe K`** does exactly that: it keeps the experts of the first K layers in RAM. Generating one token at a time ("decode", measured in tokens per second, tok/s) is then limited mainly by how fast the CPU can read those experts from RAM. The audit predicts that properly configured baseline with a formula and labels each row twice ([PROTOCOL.md 4.6](../prereg/PROTOCOL.md)):

- **Weak baseline:** the paper's own llama.cpp number falls below the predicted range.
- **Gain survives:** the paper's system still beats the top of that range.

The **22 "adjudicated" rows** are those whose prediction was tight enough to judge. On them, the central run gives **8 surviving gains and 16 of 20 weak baselines** ([audit.json](../prereg/audit/audit.json)). Scaling every prediction by 0.83 gives 10 and 13; scaling by 1.22 gives 4 and 19 ([lo run](../prereg/audit_sens_scale_lo/audit.json), [hi run](../prereg/audit_sens_scale_hi/audit.json)). These two factors are the median measured-to-predicted ratios on the project's earlier first-party check runs (its "anchors") on an A100 host and an A10. They are on the audit's datasheet basis, meaning predictions use the manufacturer's theoretical peak memory bandwidth ([paper.tex](../paper/paper.tex)).

Eight rows change label inside that range, carrying 12 verdicts:

- five KTransformers rows (a dual-socket Xeon 8452Y server with an A100 40GB or RTX 4080);
- three SeqMoE rows (Qwen3.6-35B on an RTX 5090 at 40 % and 15 % expert residency, and DeepSeek-V4-Flash on an RTX PRO 6000).

Three more rows sit on a knife edge at every scaling: KTransformers DeepSeek-V3 BF16, SeqMoE Qwen3-30B at 15 %, and the CPU-GPU collaborative Phi-3.5-MoE row.

Measurement boils every row down to one number: **m, the measured llama.cpp speed divided by the audit's central prediction**. Each row also has a **tipping ratio m\*** at which a label flips, and it can be computed today from the audit's output. The KTransformers Qwen2-57B BF16 gain, for example, survives while m stays below 1.47. SeqMoE's Qwen3-30B 15 % gain needs m below 0.98. The CPU-GPU collaborative row needs m below 1.01 ([audit.json](../prereg/audit/audit.json); [scripts/audit.py](../scripts/audit.py)).

For some model families, the anchor runs already bracket m on the datasheet basis ([basis.json](../prereg/anchors/basis.json)):

| Model family | Existing m range |
|---|---|
| Qwen3-30B | 0.73–1.21 |
| Phi-3.5-MoE | 0.76–0.85 |
| Qwen2-57B | 0.73–0.99 |
| DeepSeek-V2-Lite | 0.44–0.69 |

Where those ranges straddle a tipping ratio, a measurement is worth the most. Where they sit far from it, the verdict is already safe. This is why the budget should chase specific rows, not specific GPUs.

**The KTransformers rows hinge on one modelling choice.** KTransformers is built to use both chips of its server. The audit therefore credited llama.cpp with both sockets' memory bandwidth (563–614 GB/s on paper), and it records that "whether the llama.cpp baseline used both sockets is not stated" ([normalized_v2.jsonl](../data/audit/normalized_v2.jsonl)). Re-running the audit's predictor with only one socket gives these predicted llama.cpp speeds ([scripts/audit.py](../scripts/audit.py) predictor, re-run in the notes):

| KTransformers row | Predicted, one socket | Paper's reported llama.cpp |
|---|---|---|
| DeepSeek-V3 BF16 | 3.3–3.6 tok/s | 4.68 |
| DeepSeek-V2.5 BF16 | 6.8–7.2 tok/s | 7.05 |
| Qwen2-57B BF16 | 9.8–10.5 tok/s | 13.01 |

At that level none of the reported baselines is weak, and every KTransformers gain survives. Published community measurements point the same way. On dual-socket EPYC servers, the second socket gave DeepSeek R1 only **3–11 %** more decode speed, Mixtral 8x22B 46–85 % and dense Llama 70B 81–90 %; the poster blamed synchronisation overhead on the small expert matrices ([llama.cpp Discussion #11733](https://github.com/ggml-org/llama.cpp/discussions/11733)). On a dual Xeon 6980P, using both sockets actually slowed single-request decode ([Discussion #12088](https://github.com/ggml-org/llama.cpp/discussions/12088)). One cheap measurement of how much a second socket helps llama.cpp's all-experts-on-CPU decode would therefore decide the direction of 9 of the 12 flipping verdicts.

**Pipelined sharding's rows are more fragile than the sensitivity range suggests.** The audit treats its three judged RTX 5090 rows as safe: their tipping ratios are 0.46–0.63 for survival and 0.37–0.46 for the weak flag. But the paper also measured a properly configured llama.cpp (all experts on the CPU, `-cmoe`) at 0 GB of expert budget. That gave **25.7 tok/s against a prediction of 49.5, an m of about 0.52** ([audit.json](../prereg/audit/audit.json)), which falls inside the flip zone. If 0.52 carried over to the Qwen3-235B rows, two of the three survival verdicts would flip, and the weak flags would hold only narrowly. The paper was developed on Windows ("we recommend Windows for the smoothest reproduction experience"), which may explain a slow host ([pipeshard README](https://github.com/deepshnv/pipeshard-mlsys26-ae/blob/main/README.md)). A Linux rental separates the two readings. If the rental also gives about 0.5, the audit's formula over-predicts. If it gives 0.7–1.2, the paper's host was slow and the audit stands.

Some rows need no measurement to settle, and some cannot be measured at all:

- **Safe without measurement.** All five HybriMoE rows and the SP-MoE row have tipping ratios of 0.50 or less, below every measured ratio, so their verdicts are safe as they are ([paper.tex](../paper/paper.tex)).
- **Out of reach on a $100 budget.** KTransformers DeepSeek-V3 BF16 needs 1,342 GB of weights. FreeToken's GLM-5.2 row needs a 96 GB card and a 512 GiB host ([mosl/archs.py](../mosl/archs.py); [normalized_v2.jsonl](../data/audit/normalized_v2.jsonl)).

| Priority | Target: GPU + host | Rows it tests | Verdicts at stake | GPU cost |
|---|---|---|---|---|
| 1 | RTX 5090 on a DDR5 server, ≥ 111–128 GB RAM | SeqMoE Qwen3.6 40 % and 15 %; pipelined sharding ×3 plus its 0 GB row; FreeToken 5090-server rows (head-to-head) | 2 of the 12 flipping verdicts; pipelined sharding's hidden fragility; 2 near-miss rows made judgeable | ~$9–13 |
| 2 | RTX 4090 on a DDR5 workstation or server, ≥ 111 GB | SeqMoE Qwen3-30B 15 % and 45 %; CPU-GPU collaborative Phi-3.5 (head-to-head); FreeToken 4090 row | 2 knife-edge rows (tipping ratios 0.98 and 1.01) | ~$4–5 |
| 3 | Any whole dual-socket machine, ≥ 128 GB | KTransformers ×6, via the measured socket-scaling factor | Direction of 9 of the 12 flipping verdicts | ~$3–7 |
| 4 | GPU with ≥ 80 GB plus ≥ 192 GB host | SeqMoE DeepSeek-V4-Flash and gpt-oss-120b | 1 flipping survival verdict (likely holds: tipping ratio 1.50) | ~$2 if it shares target 3's machine |
| Optional | RTX A6000 on a DDR4 Cascade Lake Xeon | HybriMoE ×5 | None; only the date-fair label | ~$2 |

## The audited machines were servers, so the rentals must be too

The paper's limitations say "most audited rows are desktop RTX 4090/5090 machines with dual-channel DRAM", and its trust paragraph argues that "audited desktops do not pay" the cloud-VM bandwidth penalty ([paper.tex](../paper/paper.tex)). That may describe the full set of 147 rows. **It does not describe the 22 judged rows.** Every one sits on a server or workstation platform with 4–16 **memory channels** (independent lanes between CPU and RAM; desktops have 2, workstations 4–8, servers 8–12 per chip). The "paper" bandwidths below are the audit's datasheet figures ([audit.json](../prereg/audit/audit.json); [normalized_v2.jsonl](../data/audit/normalized_v2.jsonl)):

| Paper | Host | Memory | Paper bandwidth |
|---|---|---|---|
| KTransformers | 2× Xeon 8452Y | 16-channel DDR5 (stated) | 563–614 GB/s; the paper measured 220 GB/s within one socket |
| SeqMoE | Xeon 8470Q or Gold 6430, 120 or 256 GB | 8-channel DDR5 (audit's assumption; type not stated) | 256–307 GB/s |
| CPU-GPU collaborative | Threadripper 7960X | 4-channel DDR5 (assumed) | 154–166 GB/s |
| Pipelined sharding | 16-core EPYC | not stated | 153.6 GB/s (stated) |
| HybriMoE | Xeon Gold 5220R | 6-channel DDR4 (assumed) | 115–128 GB/s |
| SP-MoE | Xeon 8358P | not stated | 188–205 GB/s |

Only KTransformers states its memory. Elsewhere the channel count is the CPU platform's maximum, and the audit assumes every channel was populated.

The correction matters twice over:

- **It changes what to rent.** A desktop 4090 would give the right GPU in the wrong memory class.
- **It weakens the paper's argument that the cloud penalty applies only to the audit's own anchors.** FreeToken says its servers were "rented dual-socket servers" capped at six CPU threads ([FreeToken arXiv](https://arxiv.org/html/2608.16157)). SeqMoE's "120 GB host memory" is not a natural total for eight memory sticks (8 × 16 GB = 128 GB), which hints at a virtual-machine share. That is an inference; asking the authors would settle it.

"Matching the host" means matching six things, in this order:

- **RAM.** Enough to hold the CPU-side experts without swapping. This is a hard limit.
- **Memory bandwidth class.** Measure it at the thread count the paper used, rather than trusting the datasheet.
- **Socket count.** Whether the CPU chips have separate memory banks that software must deliberately spread across (called "NUMA"). This matters only for KTransformers.
- **Instruction sets.** Special CPU instructions for fast matrix maths. AVX-512 matters everywhere. Intel's AMX, found only on Xeons from Sapphire Rapids (2023) on, matters only for a KTransformers head-to-head ([kt-kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)).
- **GPU model and PCIe generation** (PCIe is the link between CPU and GPU). The GPU accounts for 6–64 % of predicted decode time depending on the row, and 20–59 % on the KTransformers and SeqMoE rows ([scripts/audit.py](../scripts/audit.py) predictor). PCIe does not enter the audit's `--n-cpu-moe` prediction ([PROTOCOL.md 4.7 outcomes](../prereg/PROTOCOL.md)), but it does matter for systems that copy experts over PCIe.
- **GPU driver.** FreeToken needs driver r580 or newer (CUDA 13) ([FreeToken install docs](https://github.com/FlashML-org/FreeToken/blob/main/docs/install.md)).

**You cannot turn a rented server's bandwidth down to the paper's level, so rescale instead.** The hardware knob that caps memory bandwidth (Intel/AMD "memory bandwidth allocation", set through Linux `resctrl`) needs root on the physical host. Virtual machines generally do not see it, and containers can only join groups the provider created ([Linux resctrl docs](https://docs.kernel.org/filesystems/resctrl.html); [LKML 2019](https://lkml.iu.edu/hypermail/linux/kernel/1901.1/01820.html)). Capping thread counts lowers bandwidth but also removes compute, so it confounds the two. The workable alternative uses a property the anchors already showed: **time per token rises in a straight line with the number of CPU-side layers**, with R² ≥ 0.970 on the A100 host and ≥ 0.997 on the GH200 ([PROTOCOL.md 4.7 outcomes](../prereg/PROTOCOL.md)). The procedure is:

1. Sweep K over at least three values around each row's K.
2. Read off the cost of one extra CPU layer (the slope).
3. Rescale that slope by the ratio of the paper host's bandwidth to the rental's measured bandwidth.

Two refinements keep this honest:

- **Do not divide raw tok/s by a single bandwidth factor.** MoE decode carries fixed per-token overheads, so doubling bandwidth can buy almost nothing ([Discussion #11733](https://github.com/ggml-org/llama.cpp/discussions/11733)).
- **Measure bandwidth with a read-only benchmark** (for example `likwid-bench load`, or "ALL Reads" in Intel's Memory Latency Checker, MLC) at the same thread count and pinning as llama.cpp. The popular STREAM benchmark's "Triad" test under-reports real memory traffic by about a quarter, because it ignores the extra reads that writes cause ([Hager on STREAM](https://blogs.fau.de/hager/archives/8263)). That is how llama.cpp "beat" the STREAM Triad floor in 7 of 29 A100 configurations ([a100.md](../prereg/anchors/a100.md)).

Among the providers you can use, **Vast.ai is the only one that shows the host before you rent**: CPU model, cores, RAM, motherboard, PCIe generation and measured PCIe speed, downlink, and a reliability score ([Vast search docs](https://docs.vast.ai/api-reference/search/search-offers)). No provider shows memory channels or speed, so you must measure after launch. RunPod publishes one typical configuration per GPU (RTX 5090 with 35 GB RAM, RTX 4090 with 41 GB) and shows the CPU only after launch ([RunPod pricing](https://www.runpod.io/pricing)). That is too little RAM for every priority target except pipelined sharding's 16 GB Qwen3-30B file. RunPod also raised Secure Cloud prices on 20 September ([UsagePricing](https://www.usagepricing.com/blueprint/activity/runpod-2026-09-20-secure-cloud-price-hike)). Lambda's public page lists only B200, H100, A100 and V100; the A100 40GB appears only in a multi-GPU configuration at $1.99 per GPU-hour ([Lambda pricing](https://lambda.ai/pricing)). The project's own logs record Lambda having no A100 for hours and substituting an 80 GB card for the 40 GB one ([PROTOCOL.md 4.7 outcomes](../prereg/PROTOCOL.md)). Lambda's real value here is root access in a VM (virtual machine), where GPU clock locking works. On Vast's and RunPod's default containers (isolated environments with no control of the host), locking the GPU clock is root-only and fails ([inference-server PR #30](https://github.com/AdvayMonga/inference-server/pull/30); [Vast FAQ](https://cdn.vast.ai/faq/)).

**A snapshot of 905 single-GPU Vast offers on 27 September** shows both the opportunity and the gap ([Vast API snapshot](https://console.vast.ai/api/v0/bundles/); [saved copy](../research_notes/MoE%20audit%20measurement%20plan/vast_snapshot_2026-09-27.json)):

- Seventy-nine offers ran on DDR5-generation EPYC, Threadripper or Xeon CPUs.
- Whole-machine DDR5 hosts with at least 100 GB of RAM: 5 for the RTX 4090 at $0.43–0.49/h and 8 for the RTX 5090 at $0.59–0.65/h.
- **Only three offers used Sapphire Rapids-class Xeons, all of them partial machines** ("slices"). One was an eighth of a dual Xeon Platinum 8468V machine with an RTX 5090 and 64 GB.
- Three whole dual-socket DDR4 machines did appear. Dual-socket is inferred from their thread counts; confirm with `lscpu` after launch.

Offers change hourly, so treat the table below as a shopping list to re-check the morning you rent.

| Target | What the paper's host was | What the rental must have | Closest offers seen 27 Sep | $/h |
|---|---|---|---|---|
| T1: SeqMoE 5090, FreeToken 5090, pipelined sharding 235B | Xeon 8470Q, one socket, 8-channel DDR5-4800, 120 GB, PCIe 5.0. FreeToken: 2× Xeon Gold 6459C capped at 6 threads, 77.3 GB/s measured | RTX 5090; ≥ 111–128 GB RAM; DDR5 server; PCIe 5.0 x16; driver ≥ r580 | EPYC 9684X whole machine, 111 GB, PCIe 5.0, verified. EPYC 9554 half machine, 129 GB. EPYC 9135 half machine, 387 GB (also fits DeepSeek-V4-Flash) | 0.65; 0.80; 0.94 |
| T1b: pipelined sharding Qwen3-30B rows | 16-core EPYC, 153.6 GB/s, PCIe 5.0, Windows | RTX 5090; 16-core EPYC; ≥ 32 GB | EPYC 9115 (16 cores) whole machine, 64 GB, PCIe 5.0, verified (5 offers) | 0.73 |
| T2: SeqMoE 4090, CPU-GPU collaborative | Xeon Gold 6430, 8-channel DDR5-4400, PCIe 4.0. Threadripper 7960X, 4-channel DDR5, 24 threads | Stock 24 GB RTX 4090 (filter out modified 48 GB cards); ≥ 111 GB (Phi-3.5 16-bit is 84 GB); DDR5 | Threadripper PRO 7985WX or 7995WX whole machine, 111 GB, verified. EPYC 9684X whole machine, 111 GB | 0.43–0.46; 0.46–0.49 |
| T3: KTransformers socket test (+ T4) | 2× Xeon 8452Y, 16-channel DDR5, 1 TB per socket | Whole machine with two sockets, ≥ 128 GB (≥ 258 GB to add DeepSeek-V4-Flash) | 2× Xeon Gold 5118 + A100 80GB, 258 GB (unverified). 2× E5-2696 v3 + 4090, 129 GB. 2× E5-2673 v4 + 4090, 903 GB. All DDR4 | 0.60; 0.38; 1.14 |
| Exact KTransformers host | As above, with AMX | Dual-socket Sapphire Rapids | Only the 1/8 slice with 64 GB | — |

On Vast, allow for a few practical points:

- Opening an account needs only email verification and a $5 minimum credit ([Vast quickstart](https://docs.vast.ai/guides/get-started/quickstart)).
- Storage accrues until an instance is deleted, and bandwidth costs about $5.5 per TB ([Vast FAQ](https://cdn.vast.ai/faq/); [Vast API snapshot](https://console.vast.ai/api/v0/bundles/)).
- Filter for hosts with at least 500 Mbps download, because some advertise under 100 Mbps.
- Default instances are unprivileged containers. VM-enabled offers give full root, but they were only 66 of the 905 ([Vast VM docs](https://docs.vast.ai/linux-virtual-machines)).

## Three systems can race llama.cpp on the same machine

A **head-to-head** runs the paper's own system and llama.cpp on one rented machine at the same GPU memory. It is stronger evidence than any prediction, because host differences cancel out of the ratio. It is only trustworthy if the system reproduces something close to its own published speed on the rental, so the plan below registers a fidelity check first. Code is public for six audited systems, but only three are both runnable on a rental and aimed at uncertain or near-miss verdicts. The table links each repository.

| System | Code status | What it would settle | Needs | Effort and cost | Main risk |
|---|---|---|---|---|---|
| Pipelined sharding (MLSys'26) | MIT artifact with tags and a Zenodo archive. Built on llama.cpp b6097, and **the same binary contains `--n-cpu-moe`** ([repo](https://github.com/deepshnv/pipeshard-mlsys26-ae); [arXiv](https://arxiv.org/html/2604.26334)) | Three judged 5090 rows and the 0 GB ratio of about 0.52 | RTX 5090, CUDA 12.8+, 16 cores, ≥ 128 GB for Qwen3-235B | 2–4 h, $3–6 | Developed on Windows; ~20 min build, 1–3 h of downloads |
| FreeToken (Aug 2026) | Apache-2.0; tag v0.1.2 (19 Aug 2026) is closest to the paper ([releases](https://github.com/FlashML-org/FreeToken/releases)) | FreeToken's 5090 and 4090 server rows. These are "near-miss" rows, excluded only because their predicted range was wider than ±40 % | Driver r580+ and a CUDA 13 toolkit; Qwen3.6 BF16 weights in the original (safetensors) format; 6 threads pinned to the GPU's memory node | 4–8 h, $10–20 | Engine-install hash mismatches ([issue #554](https://github.com/FlashML-org/FreeToken/issues/554)) |
| CPU-GPU collaborative (arXiv 2512.16473) | MIT; last commit October 2025 ([repo](https://github.com/elsa-lab/MoE-CPU-GPU-Collaborative-Inference)) | The Phi-3.5-MoE knife-edge row (tipping ratio 1.01) | RTX 4090, ≥ 128 GB, 24 threads; CPU experts run as PyTorch BF16 maths, so AVX-512 BF16 helps | 4–6 h, $8–15 | Its only published baseline is its own CPU-only mode |
| KTransformers (SOSP'25) | Apache-2.0; the paper-era code now lives in `archive/` ([README](https://github.com/kvcache-ai/ktransformers/blob/main/README.md)) | The six KTransformers rows, faithfully | An AMX Xeon, ideally two sockets, and the legacy Qwen2-57B AMX rules | 1–2 days, $15–40+ | No AMX host rentable; segfault on single-memory-node machines ([issue #1754](https://github.com/kvcache-ai/ktransformers/issues/1754)) |
| HybriMoE (DAC'25) | Apache-2.0, 6 commits ([repo](https://github.com/PKU-SEC-Lab/HybriMoE)) | Five rows, all already safe | RTX A6000 class; 2025 CUDA 12.1 stack | 4–8 h, $3–8 | Marlin-weights bug ([issue #8](https://github.com/PKU-SEC-Lab/HybriMoE/issues/8)); only its DeepSeek rule turns its cache on |
| SeqMoE, SP-MoE, MoE-SpeQ | No code in the papers ([SeqMoE](https://arxiv.org/html/2609.12978v1); [SP-MoE](https://arxiv.org/html/2510.10302); [MoE-SpeQ](https://arxiv.org/html/2511.14102)) | — | — | — | Only the llama.cpp side can be measured |
| ProMoE | Incomplete: its llama.cpp fork sits on a private university server ([install.md](https://github.com/promoe-opensource/promoe/blob/main/install.md)) | — | — | — | — |

**Pipelined sharding is the best head-to-head per dollar.** The authors ship reproduction scripts that "print PASS (or not) along with error margins". Their own binary also provides the properly configured baseline, so system and baseline differ only in the feature under test ([arXiv appendix](https://arxiv.org/html/2604.26334)). Add a current llama.cpp build as a second baseline, because b6097 dates from August 2025.

**FreeToken is the most useful expansion.** Five of its six server and desktop rows have predicted normalised speed-ups whose lower edge already exceeds 1. Its reported llama.cpp baselines sit inside the predicted range. Measuring them head-to-head could lift the headline from 8 of 22 surviving gains toward about 13 of 28 ([audit.json](../prereg/audit/audit.json)). That is a projection, not a result. FreeToken's own method, six threads pinned to one memory node, is easy to copy on any rental. Its paper also reports the host's measured bandwidth at that thread count (77.3 GB/s), which makes the carry-over across hosts unusually clean ([FreeToken arXiv](https://arxiv.org/html/2608.16157)).

**The CPU-GPU collaborative system decides its own knife-edge row directly.** Its Phi-3.5 file is 84 GB in 16-bit form, so the host needs at least 111 GB. A Threadripper 7960X plus RTX 4090, the paper's exact CPU, did appear on Vast, but only as a 32 GB half-machine ([Vast API snapshot](https://console.vast.ai/api/v0/bundles/)).

**KTransformers cannot be tested faithfully this cycle.** Its fast paths need AMX, and the snapshot held no whole AMX machine. On a host without AMX it falls back to generic kernels, which says nothing about its published claim ([kt-kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)).

**SeqMoE's three flipping verdicts can only be settled by measuring llama.cpp**, because SeqMoE has no code. Emailing its authors for the code and their host's memory configuration costs nothing. The protocol already promises every audited team its rows before a second version, so the email has a natural occasion ([PROTOCOL.md 4.6](../prereg/PROTOCOL.md)).

## Judge each baseline by the llama.cpp of its paper's date

**Configure today's baseline the way the audit predicted it.** Build the same llama.cpp commit the anchors used, 2145525a from 26 September 2026 ([002_build.sh](../gpu/jobs/002_build.sh)). Name the GPU architectures explicitly instead of `native`, because an older cmake rejected `native` on the GH200 ([PROTOCOL.md 4.7 outcomes](../prereg/PROTOCOL.md)). That commit includes the benchmark tool's `-d` option, which pre-fills a set number of tokens before timing (added April 2025, [PR #13096](https://github.com/ggml-org/llama.cpp/pull/13096)), and its `--n-cpu-moe` flag (September 2025, [PR #15952](https://github.com/ggml-org/llama.cpp/pull/15952)).

Five rules make the baseline match the row:

- **Offset K for leading dense layers.** `--n-cpu-moe` counts from the first layer, including any leading layers that have no experts. Add 1 for DeepSeek-V2/V2.5 and 3 for DeepSeek-V3; Qwen, Phi, Mixtral and gpt-oss need no offset ([a100.json](../prereg/anchors/a100.json); [mosl/archs.py](../mosl/archs.py)).
- **Check the budget with llama.cpp's own memory printout, not with the benchmark tool.** Use `llama-server` at the row's context length; it prints GPU memory split into model, context and compute since September 2025 ([PR #15860](https://github.com/ggml-org/llama.cpp/pull/15860)). The benchmark tool `llama-bench` sizes its cache for only the tokens it tests ([llama-bench source](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/llama-bench.cpp)). The automatic `--fit` option silently shrinks context to 4,096 tokens ([fit.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/fit.cpp)).
- **Match the paper's context.** Time decode at the row's context depth with `-d`, because a longer context enlarges the **KV cache** (the model's memory of earlier tokens) in GPU memory.
- **Match the paper's threads.** Use the paper's thread count where stated: FreeToken 6 (8 for DeepSeek-V4-Flash), the CPU-GPU collaborative system 24, KTransformers all cores. Otherwise sweep and keep the best.
- **Load the whole model into RAM before timing.** Disable memory-mapped loading (`--mmap 0`), which otherwise lets the operating system pull the model from disk on first use. The benchmark's one-token warm-up touches only a few experts, so later repetitions can stall on disk reads ([llama.cpp notes](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md)).

In the model-file column below, BF16 and fp16 store 16 bits per weight, Q8_0 about 8.5 bits and Q4_0 about 4.5; the suffix is the **quantization**, meaning how compactly the weights are stored.

| Row | Model file (GGUF, llama.cpp's format) | `--n-cpu-moe` K | Pre-filled / timed tokens | CPU threads |
|---|---|---|---|---|
| SeqMoE Qwen3.6 40 % / 15 % | Qwen3.6-35B-A3B BF16, 69.4 GB ([unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)) | 24 / 34 | 416 / 128 | Sweep (paper silent) |
| SeqMoE Qwen3-30B 45 % / 15 % | Qwen3-30B-A3B Q8_0, 32.5 GB, 6 % more expert bytes than the paper's FP8 ([Qwen](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF)) | 27 / 41 | 416 / 128 | Sweep |
| CPU-GPU collaborative Phi-3.5 | Phi-3.5-MoE 16-bit, 84 GB (GGUF availability unverified) | 25 | Not stated; use 512 / 128 | 24 |
| Pipelined sharding 8 GB / 0 GB | Qwen3-30B-A3B-Instruct-2507 Q4_0, 16.4 GB | 32 / 48 (all) | 1024 / 100 | 16 physical cores |
| Pipelined sharding 235B 2 GB / 32 GB | Qwen3-235B-A22B-Instruct-2507 Q2_K, 77 GB | 94 (all) / 63 | 1024 / 100 | 16 |
| FreeToken 5090 server Qwen3.6 | Qwen3.6-35B-A3B BF16 | 26 | 4096 / 128 | 6, pinned |
| KTransformers Qwen2-57B BF16 / int8 | Qwen2-57B-A14B fp16, 114.8 GB / q8_0, 61 GB ([Qwen](https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct-GGUF)) | All (`-cmoe`) | 288 / 128 | All; one socket vs two |

Row configurations come from [normalized_v2.jsonl](../data/audit/normalized_v2.jsonl) and [audit.json](../prereg/audit/audit.json).

**The best llama.cpp configuration changed three times, so each paper deserves the baseline of its own date.** Until 2 April 2025, mainline llama.cpp could offload an MoE model to the GPU only in whole layers (`-ngl N`). Per-expert placement (`-ot`) was merged that day as release b5028, after an open pull request had circulated since late January ([PR #11397](https://github.com/ggml-org/llama.cpp/pull/11397)). `--cpu-moe` followed on 31 July and `--n-cpu-moe` on 5 August 2025 ([PR #14992](https://github.com/ggml-org/llama.cpp/pull/14992); [PR #15077](https://github.com/ggml-org/llama.cpp/pull/15077)). Flash attention, a faster attention method that also frees GPU memory, became the default on 30 August 2025 ([PR #15434](https://github.com/ggml-org/llama.cpp/pull/15434)). Automatic placement (`--fit`) arrived on 15 December 2025 ([PR #16653](https://github.com/ggml-org/llama.cpp/pull/16653)).

Per-expert placement is what matters most: anecdotes in the pull request show +50–66 % decode speed from it. CPU kernel rewrites barely touched decode; one repacking change moved decode speed by −2 % while speeding up prompt processing by 61 % ([PR #12332](https://github.com/ggml-org/llama.cpp/pull/12332)). CUDA graphs are a way to replay recorded GPU work with less launch overhead. They were switched off with `--n-cpu-moe` for builds b7625–b7820 and restored on 24 January 2026, worth 4–10 % ([PR #18593](https://github.com/ggml-org/llama.cpp/pull/18593); [PR #18934](https://github.com/ggml-org/llama.cpp/pull/18934)).

| Paper date (arXiv v1) | Best mainline configuration at equal GPU memory | Audited papers in this era |
|---|---|---|
| Before 2 Apr 2025 | Whole layers on the GPU (`-ngl N`), with flash attention from April 2024 ([PR #5021](https://github.com/ggml-org/llama.cpp/pull/5021)) | Fiddler, ProMoE (neither adjudicated) |
| Around 2 Apr 2025 | Per-expert placement just merged (`-ot`), with CPU experts not yet repacked ([PR #12498](https://github.com/ggml-org/llama.cpp/pull/12498)) | HybriMoE (arXiv April 2025, days after the merge; its experiments very likely predate it). KTransformers (SOSP'25; code tags March–May 2025) |
| From 5 Aug 2025 | `--n-cpu-moe K`; flash attention on by default; later `--fit` and CUDA graphs | SP-MoE (Oct 2025), MoE-SpeQ (Nov 2025), CPU-GPU collaborative (Dec 2025), pipelined sharding (Apr 2026, built on b6097), FreeToken (Aug 2026), SeqMoE (Sep 2026) |

Applied to the 16 weak baselines, the date check splits them cleanly ([audit.json](../prereg/audit/audit.json)):

- **Nine are weak by the standard of their own day:** six SeqMoE and three pipelined sharding. Those papers were written after `--n-cpu-moe` existed and was documented in the official gpt-oss guide ([Discussion #15396](https://github.com/ggml-org/llama.cpp/discussions/15396)).
- **Seven straddle the April 2025 boundary:** five HybriMoE and two KTransformers. They need a two-level verdict. "Weak for its time" means below what `-ngl N` with flash attention gave at equal memory. "Stale by today" means fine then, superseded now, and not held against the authors.

Measuring the "for its time" level does not require old builds. Run today's build with `-ngl N` at the row's budget, then spot-check one old release (b5027, the last before `-ot`) on an RTX 4090. Old CUDA code may not compile for the RTX 5090's Blackwell architecture. Builds before April 2024 cannot read today's stacked-expert GGUF files ([PR #6387](https://github.com/ggml-org/llama.cpp/pull/6387)).

## Write the rules down before timing the first token

**Pre-registration** means writing the predictions and decision rules into the repository, and committing them before measuring, so git history proves the order. The project already works this way. It also needs a new rule now. The current protocol says anchor runs "validate the predictor; they do not replace any audit row's prediction", and it has no tolerance or carry-over rule for measured baselines ([PROTOCOL.md 4.7](../prereg/PROTOCOL.md)). A new section, 4.8, must be committed before any timed run. Its core content:

| Item | Proposed content |
|---|---|
| Scope change | For the listed rows, a measured baseline may replace the predicted one. The predicted-basis audit stays in the paper as registered, and both headlines are reported. |
| Row configuration | Model file and checksum, K, pre-filled and timed tokens, threads, memory-node policy, and the GPU-memory check from the server printout, all as in the table above. |
| Host qualification | RAM at least the CPU-side model plus 25 %. Read-only bandwidth measured at the run's thread count, and hosts outside a stated band released. CPU instruction flags, driver, PCIe link, and memory nodes (`numactl -H`) recorded. |
| Predictions before timing | On each rental, re-run the audit's predictor with that host's measured and datasheet bandwidth, and commit and push the output before the first timed call, as the anchor jobs did. |
| Carry-over rule | Row estimate = the row's predicted baseline × m, with m measured on the rental. Compute it on both the datasheet basis and the measured-bandwidth basis, and settle a row only when both agree. Where a paper reports its measured bandwidth (KTransformers 220 GB/s per socket, FreeToken 77.3 GB/s at 6 threads), use that. |
| Tolerance u | 10 % when GPU model, bytes per weight and socket count match and measured bandwidth is within ±15 % of the paper's. 25 % with any substitution. Never below twice the spread seen between two hosts of the same class. |
| Labels | Survives if the system beats the row estimate × (1+u). Weak if the reported baseline is below the row estimate × (1−u). Otherwise "at strength" or "not established". "Unsettled" if the interval straddles a threshold. |
| Head-to-head | System version pinned. The system must reproduce its paper's own number within ±15 % on the rental, or it is reported as "not reproduced". The verdict is the measured ratio with its 95 % interval. |
| Date-fair label | Era set by arXiv v1 date. "Weak for its time" versus "stale by today", as defined above. |
| Statistics | 3 separate launches × 11 repetitions, first repetition dropped. Median and a 95 % confidence interval from resampling (bootstrap). Harmonic mean for rates (total tokens ÷ total time). Differences below 5 % are ties. Configuration order randomised, with drift repeats at the end. |
| Hypotheses | See the predictions below. |
| Budget and stopping | Per-session dollar caps. A hard stop at $100 including storage and transfer. Every deviation logged. |

The ±15 %, 25 %, twice-the-spread and 5 % figures are proposals for you to fix, not published standards. The statistics follow Hoefler and Belli's benchmarking rules: confidence intervals rather than bare means, the harmonic mean for rates, no assumed normality, and full documentation of the setup ([Hoefler & Belli, SC'15](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf)). The benchmark tool's own default is five repetitions reported as mean ± standard deviation, with raw samples available as JSON ([llama-bench README](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md)).

The decision thresholds can be registered as plain speeds now. B is the measured-basis llama.cpp estimate in tok/s ([audit.json](../prereg/audit/audit.json); thresholds from the notes' formula: survive if B < system/(1+u), weak if B > reported/(1−u)).

| Row | System | Paper's llama.cpp | Predicted low / mid / high | Tipping ratio: survive / weak | u = 10 %: survives if B <, weak if B > | u = 25 % |
|---|---|---|---|---|---|---|
| KTransformers DeepSeek-V3 BF16 | 5.87 | 4.68 | 4.2 / 5.6 / 7.6 | 1.04 / 0.83 | 5.3 / 5.2 | 4.7 / 6.2 |
| KTransformers DeepSeek-V2.5 BF16 | 11.95 | 7.05 | 7.9 / 10.4 / 14.0 | 1.15 / 0.68 | 10.9 / 7.8 | 9.6 / 9.4 |
| KTransformers Qwen2-57B BF16 | 22.88 | 13.01 | 11.8 / 15.5 / 20.9 | 1.47 / 0.84 | 20.8 / 14.5 | 18.3 / 17.3 |
| KTransformers DeepSeek-V3 int4 | 16.15 | 9.06 | 10.1 / 13.3 / 17.8 | 1.22 / 0.68 | 14.7 / 10.1 | 12.9 / 12.1 |
| KTransformers DeepSeek-V2.5 int8 | 17.61 | 9.94 | 9.5 / 12.4 / 16.6 | 1.42 / 0.80 | 16.0 / 11.0 | 14.1 / 13.3 |
| SeqMoE Qwen3.6 40 % | 118.3 | 37.9 | 43.0 / 56.4 / 75.4 | 2.10 / 0.67 | 107.5 / 42.1 | 94.6 / 50.5 |
| SeqMoE Qwen3.6 15 % | 70.2 | 28.2 | 35.3 / 46.5 / 62.3 | 1.51 / 0.61 | 63.8 / 31.3 | 56.2 / 37.6 |
| SeqMoE Qwen3-30B 15 % | 47.9 | 21.1 | 37.0 / 48.9 / 65.7 | 0.98 / 0.43 | 43.5 / 23.4 | 38.3 / 28.1 |
| SeqMoE DeepSeek-V4-Flash | 57.4 | 21.0 | 29.1 / 38.3 / 51.2 | 1.50 / 0.55 | 52.2 / 23.3 | 45.9 / 28.0 |
| CPU-GPU collaborative Phi-3.5 | 10.4 | 6.3 (own CPU mode) | 7.8 / 10.3 / 14.1 | 1.01 / — | 9.5 / — | 8.3 / — |
| Pipelined sharding Qwen3-30B 8 GB | 32.1 | 26.1 | 54.0 / 69.7 / 91.6 | 0.46 / 0.37 | 29.2 / 29.0 | 25.7 / 34.8 |
| Pipelined sharding Qwen3-235B 2 GB | 7.7 | 5.7 | 10.2 / 13.1 / 17.3 | 0.59 / 0.44 | 7.0 / 6.3 | 6.2 / 7.6 |
| Pipelined sharding Qwen3-235B 32 GB | 11.5 | 8.46 | 14.1 / 18.2 / 23.9 | 0.63 / 0.46 | 10.5 / 9.4 | 9.2 / 11.3 |

Two things follow from the table:

- **At u = 25 %, the two thresholds almost meet on several rows.** On KTransformers DeepSeek-V2.5 BF16 they are 9.6 and 9.4, so any measurement settles one label but may leave the other unsettled.
- **Tight matching is what buys decisive weak-flag verdicts.** The 10 % tolerance needs the same GPU and a bandwidth within ±15 % of the paper's host, and it is what makes the weak flags on SeqMoE's rows decisive.

On each rental, the measurement itself runs in four stages:

1. **Qualify the host (about 15 minutes, released if it fails).** Record `lscpu` (checking for AVX-512 and AMX), `numactl -H`, free memory, the container's CPU quota, and `nvidia-smi -q` (driver and PCIe link). Measure read-only bandwidth at 1, 4, 6, 8, 16 and all threads, plus STREAM Triad compiled with `-ffreestanding` for comparability ([Hager](https://blogs.fau.de/hager/archives/8263)). Record disk and network speed, and attempt a GPU clock lock.
2. **Commit and push the rental-specific predictions.**
3. **Time the runs** with `llama-bench -ngl 99 -ncmoe K -fa 1 -p 0 -n 128 -d C --mmap 0 -r 11 -o jsonl`. Use three separate launches in randomised order, pin threads with `--cpu-mask` or `numactl`, and log GPU clocks throughout. Where clocks cannot be locked in a container, record that as a registered deviation, with drift repeats as the control.
4. **Record everything** Hoefler's ninth rule asks for: llama.cpp build and commit, CUDA and driver versions, GPU and PCIe link, CPU model and threads, memory type and nodes, load mode, flash attention, KV-cache types, batch sizes, K, depth, and the GGUF checksum ([Hoefler & Belli](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf)).

Register predictions alongside the rules, as the project did for its anchors. Each follows from existing evidence and would embarrass the audit if it failed:

- **Second socket.** A second socket speeds up llama.cpp's all-experts-on-CPU decode of Qwen2-57B by no more than 25 % ([Discussion #11733](https://github.com/ggml-org/llama.cpp/discussions/11733)).
- **SeqMoE Qwen3.6.** Both gains survive, because m will fall inside the Qwen3 anchor range of 0.73–1.21 ([basis.json](../prereg/anchors/basis.json)).
- **Pipelined sharding's 0 GB row on Linux.** m lands between 0.7 and 1.2. That would mean the paper's host, not the formula, explains the 0.52.
- **CPU-GPU collaborative Phi-3.5.** The gain survives, because the Phi anchor ratios of 0.76–0.85 sit below the tipping ratio of 1.01.

## Conclusion

The audit's uncertainty is far more concentrated than its ±22 % sensitivity range makes it look. One hardware question, whether llama.cpp profits from a second CPU socket, governs three quarters of the verdicts that flip. It can be tested for a few dollars on a whole two-socket machine of any generation. Matching every paper's exact Xeon would take a machine the rental market barely offers. So the extension should be framed as measuring a small set of carry-over factors (socket scaling, per-model measured-to-predicted ratios, and a host effect for pipelined sharding) under rules fixed in advance, not as rebuilding each paper's machine. Framed that way, the measurements are cheap, and they test exactly the assumptions the verdicts rest on.

The evidence in hand also points in one direction. The socket data, the anchor ratios for Qwen3 and Phi, and pipelined sharding's own baseline all suggest measured llama.cpp will land at or below the audit's predictions on these hosts. The headline is therefore likelier to soften than to harden: more surviving gains, fewer weak baselines, plausibly toward or past the ×0.83 run's 10 survivors and 13 weak. Registering that expectation before measuring is what makes a softer headline a credible result rather than a retreat. The fact-check also turned up a claim the paper should correct regardless of the budget: the judged rows came from servers, not dual-channel desktops. That changes both the rental choice and the paper's argument about which hosts pay a cloud bandwidth penalty.

## Execution plan: about $35 and ten working days

Costs use the 27 September prices above and assume per-second billing. Model downloads of about 1–1.5 TB add roughly $6–8 of Vast transfer charges ([Vast API snapshot](https://console.vast.ai/api/v0/bundles/)). The whole A100 anchor sweep of 31 configurations took about 14 minutes, according to job 051's GPU monitor log (`results/051_anchor_sweep@anchor/gpu_monitor.csv` on the repository's `gpu` branch), so setup and downloads, not timing, dominate rental hours. The fellowship form is due Tuesday 27 October at 16:00 CET, and an unaffiliated author will need an arXiv endorser ([APPLICATION_PLAN.md](../apply/APPLICATION_PLAN.md)).

| # | When (2026) | Step | Where | GPU hours | GPU $ | Your hours |
|---|---|---|---|---|---|---|
| 1 | Mon 28 Sep – Wed 30 Sep | Write and commit protocol section 4.8: rules, tolerances, threshold table, hypotheses, per-session caps | Repo | 0 | 0 | 6 |
| 2 | Tue 29 Sep – Thu 1 Oct | Email the SeqMoE, KTransformers, pipelined sharding and FreeToken authors their rows (drafts in `apply/audit_emails/`). Ask for memory channels and speed, bare metal or VM, and llama.cpp version, flags, threads and sockets. Ask SeqMoE for code | Email | 0 | 0 | 2 |
| 3 | Wed 30 Sep – Sat 3 Oct | Extend the job scripts: host qualification, K finder (server memory printout), 3 × 11 timing runner, and scoring (bootstrap intervals, harmonic mean, registered labels) | Repo | 0 | 0 | 10 |
| 4 | Sat 3 – Sun 4 Oct | Open a Vast account ($5 minimum). Dry-run the scripts on a cheap RTX 4090 with Qwen3-30B Q4_0 | Vast | 1–2 | ~1 | 3 |
| 5 | Mon 5 Oct | Qualify 6–8 candidate hosts for 15 minutes each and keep the in-band ones | Vast | ~2 | ~1.5 | 3 |
| 6 | Tue 6 – Wed 7 Oct | **T1 (RTX 5090, DDR5, ≥ 111–128 GB, driver ≥ r580):** predictions committed, then SeqMoE Qwen3.6 K sweep including 24 and 34; pipelined sharding Qwen3-235B runs with its own binary and `--n-cpu-moe`; FreeToken v0.1.2 vs llama.cpp at 6 threads, K = 26, depth 4096 | Vast, EPYC 9684X/9554-class | 10–12 | 7–11 | 10 |
| 7 | Wed 7 Oct | **T1b:** pipelined sharding Qwen3-30B rows at K = 32 and 48 on a 16-core EPYC with an RTX 5090, with its own binary and current llama.cpp | Vast, EPYC 9115 | ~3 | ~2.2 | 3 |
| 8 | Thu 8 – Fri 9 Oct | **T2 (RTX 4090, DDR5, 111 GB):** SeqMoE Qwen3-30B Q8_0 at K = 27 and 41; CPU-GPU collaborative Phi-3.5 head-to-head at 24 threads; FreeToken 4090 row; `-ngl N` date-fair runs and one old-release spot check | Vast, Threadripper PRO 7985WX-class | 8–10 | 4–5 | 8 |
| 9 | Sat 10 – Mon 12 Oct | **T3:** socket-scaling test on a whole dual-socket machine. Qwen2-57B q8_0 and fp16 with `-cmoe`, one socket bound versus both sockets spread, with bandwidth logged per policy. Add SeqMoE DeepSeek-V4-Flash and gpt-oss-120b if the machine has an 80 GB GPU and 258 GB RAM | Vast, dual Xeon Gold 5118 + A100 80GB or similar | 4–6 | 3–7 | 5 |
| 10 | Tue 13 – Wed 14 Oct | Repeat the core llama.cpp configurations of T1 and T2 on a second host of each class, to estimate between-host spread | Vast | ~6 | ~4 | 4 |
| 11 | Throughout | Storage and data transfer | Vast | — | 6–8 | — |
| 12 | Thu 15 – Mon 19 Oct | Score the predictions, apply the registered labels, re-run the audit into a new measured-basis output, add the date-fair labels, and update tables, text (including the desktop sentence), README and application summary | Repo | 0 | 0 | 16 |
| 13 | Tue 20 – Wed 21 Oct | Submit to arXiv (endorser arranged in advance) and finalise the fellowship materials; buffer for moderation and referees | — | 0 | 0 | 4 |
| 14 | Reserve | Failed hosts, repeats, and optional runs: HybriMoE on an A6000 with a Cascade Lake Xeon (~$2), GH200 DeepSeek diagnostic on Lambda (price not captured; check the console) | — | — | ~60 | — |
| | | **Core total** | | **~34–41** | **~$29–40** | **~74 (about 10 days)** |

If time runs short, the minimum useful version is steps 1–8 and 12–13, about $20–25 of GPU spend and eight working days. It settles SeqMoE's RTX 5090 verdicts, the two knife-edge rows and pipelined sharding's host question, and leaves KTransformers explicitly unsettled.

## Decisions only you can make

| # | Decision | Options | Recommendation | What it changes |
|---|---|---|---|---|
| 1 | Where to rent | Vast.ai (new account) versus only RunPod and Lambda | Open a Vast account: only it lets you choose DDR5 server hosts with ≥ 111 GB before renting | Without it, no priority target except pipelined sharding's 30B rows is feasible |
| 2 | Let measurement replace prediction | Amend the protocol (section 4.8) or keep measurements as validation only | Amend, and report both the predicted-basis and measured-basis headlines | Without the amendment, no verdict can formally change |
| 3 | How far to pursue KTransformers | (a) ~$5 socket test on an old dual-socket DDR4 machine, with 25 % tolerance; (b) $30–60 hunting a DDR5 two-socket node; (c) leave the rows unsettled | (a), and state that the KTransformers rows are settled in direction, not exactly | 9 of the 12 flipping verdicts |
| 4 | Bandwidth basis for carrying results across hosts | Datasheet basis, measured basis, or both | Both, settling a row only when they agree; use the paper's own measured bandwidth where given | Whether the cloud bandwidth penalty is counted for or against the papers |
| 5 | Standard for "date-fair" | Strict (released mainline at arXiv v1 date) versus state of practice (includes the January–March 2025 `-ot` branch) | Strict as the primary label, with state of practice as a note | 7 of the 16 weak labels (HybriMoE, KTransformers) |
| 6 | Scope | The 22 judged rows only, or also FreeToken head-to-heads | Include FreeToken only if the T1 host has a CUDA 13 driver | The headline could grow from 22 toward 28 judged rows |
| 7 | GPU clock locking | Accept unlocked clocks in containers with logging, or pay for VM-enabled hosts or Lambda | Accept, logging clocks and repeating drift runs, as a registered deviation | Comparability with the anchors, which locked clocks |
| 8 | Contacting authors now | Email this week, or after posting | This week: the protocol requires it before a second version anyway, and answers on memory configuration are free measurements | Could tighten SeqMoE's tolerance to 10 % |
| 9 | arXiv timing | Post v1 with measurements (~21 Oct), or post v1 now and add measurements as v2 | Set a checkpoint on Fri 16 Oct: if the analysis is not done by then, post without the measurements and include the registered plan | Deadline risk against a stronger first version |
| 10 | The paper's desktop sentence | Keep or correct | Correct it: the judged rows ran on servers | Credibility with reviewers who check the rows |
| 11 | Optional runs | SeqMoE DeepSeek-V4-Flash, HybriMoE date-fair runs, GH200 diagnostic | Only from reserve after steps 6–10 | 1 flipping verdict (likely holds) and labels that are already safe |
