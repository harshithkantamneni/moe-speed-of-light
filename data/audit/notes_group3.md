# Group 3 (2026) extraction notes

Rows: `rows_group3.jsonl` (59 rows: 46 from papers, 13 from the llama.cpp community tier). Values read off figures were taken from
printed bar labels where they exist. Where there were no labels I estimated from the axis on page renders (PDF via pdftoppm, or the
arXiv HTML image), and every such row is marked `figure_only: true`. Tiers apply the SCHEMA rule strictly: an unstated expert dtype or
host CPU gives tier C.

## Papers

**DALI (2602.03495).** RTX 3090, EPYC 7532 (limited to 16 cores / 32 threads), 256 GB DDR4, PCIe 4.0. There is **no batch-1 data**: Fig. 12
starts at batch 16 and reports per-sequence speed. The weight dtype is never stated. The llama.cpp baseline is layer-wise, with "the
number of MoE layers stored and executed on the GPU" set to give "comparable memory usage". No version or flags are given. The headline
3.97x over llama.cpp is dominated by batch 64-128 points, where llama.cpp falls to 0.15-0.16 tok/s. At batch 16 the gap is only 1.8-2.3x.

**FlashMoE (2601.17063).** SSD tier: RTX 5070 Ti, Ryzen 9 9600X, 64 GB DDR5-6000 dual-channel with only ~1 GB left free, and a PCIe 5.0
NVMe. Tok/s values exist only in unlabelled PDF bars (Fig. 7c/f), so they are axis estimates. The llama.cpp configuration (version,
flags, memory) and the batch size are **not stated**. "Up to 2.6x" matches Qwen3-30B at 24/128 cached experts against llama.cpp.

**Speculating Experts (2603.19289).** A6000 / A100 / GH200, bf16, all experts in pinned host memory, batch 1. TPOT values are bar
labels. The baseline is the same YALIS engine with on-demand loading, not llama.cpp. **Not exact routing:** "we report improvements
using the speculated experts and do not re-fetch the true experts". GLM-4.7-Flash and GPT-OSS use random weights. CPU and DRAM type
are missing (tier C). The gain is small (5-16%).

**ST-MoE (2606.15453).** A cycle-accurate simulated accelerator (SCALE-Sim, TSMC 40nm) compared against an A100, reporting normalized
cycles only. It is out of scope and gets a single ratio-only row (60% lower execution time than the GPU).

**EcoSpec (2607.12696).** Speculative decoding on 8x H200 in HF Transformers with no offloading, so it is out of scope. AR baseline
tok/s is converted from Table 3 per-token latency. **Internal inconsistency:** α/T_total from Table 3 does not reproduce the Table 3
speedup column (Qwen 1.53x vs 1.36x reported; GPT-OSS 1.17x vs 1.31x). I left system tok/s null.

**AcceptMoE (2608.02989).** RTX 5090 with 48 expert slots per layer in SGLang 0.5.12.post1, batch 1, 256 output tokens. The baselines
(Vanilla AR, EAGLE-3 SD) run on the same eager runtime with CUDA Graphs off. There is no llama.cpp baseline. Speculative decoding plus
a restricted and residency-pruned verifier expert set, so **not exact routing** (−0.27 pp accuracy). Host CPU, DRAM and PCIe are
missing, and throughput comes only from Fig. 4 axis estimates.

**FreeToken (2608.16157).** The most complete hardware reporting in the group: measured B_P (PCIe) and B_H (CPU expert-kernel
bandwidth) for 6 machines. Budget is stated only for the RTX 5090 server (37% / 11% of the expert pool). **The llama.cpp version and
flags are not stated**. It is described only as a "routing-blind static split" that assigns "whole layers to devices statically at load
time". Weights are bit-identical across engines. Metrics come from agentic workloads (per-request mean); W1 is the closest to batch-1
decode. Rented servers were capped at 6-8 threads.

**Cacheable by Design? (2608.18261).** A llama.cpp measurement with **explicit flags** `-ngl 99 -ncmoe 94`, streaming Qwen3-235B Q4_K_M
(134 GB) from a PCIe-3 NVMe on an RTX 3070 with 32 GB DDR4. It is SSD-bound at 0.441 tok/s warm. The PMS "~0.9 tok/s, 2-7x" is a
projection, not a measurement, so system tok/s is null.

**SeqMoE (2609.12978).** RTX 4090 / 5090 / PRO 6000 on Xeon hosts with 120-256 GB (DRAM type unstated), batch 1, cache fractions
15-45%. Values are bar labels. **The llama.cpp configuration is not stated**: there is no commit (only "accessed 2026-09-09"), no flags,
and no explanation of how an "X% of experts" budget was mapped onto llama.cpp. Precision parity is also unclear (FP8 Qwen3-30B).
FreeToken appears as a baseline at 86.2 tok/s on Qwen3.6 with a 5090; FreeToken's own paper reports 77.1.

**SSD-LLaMA (2609.18110).** SSD tier: RTX 5090, Core Ultra 5 230F, 16 GB DRAM, 9 GiB/s NVMe, batch 1. The llama.cpp baseline flags are
not stated. The ablation implies default GGUF mmap loading. **Stated ranges conflict:** Sec. 5.2 says 1.35-3.44x over llama.cpp, but the
Fig. 10 labels give 2.12-3.44x. Kimi-K3 17.2x is text-only, with no llama.cpp tok/s. The 1.03 tok/s Kimi-K2.7 result uses 32 GB RAM and
has no baseline.

**Edge0 / "The Other Half of the Memory Wall" (2609.18063 v2).** Apple M4 Pro and M2 (unified memory plus SSD, MLX), so it is outside
the dGPU/PCIe scope. **Routing is changed and lossy**: the prerouter prediction is executed as the routing, K=4 instead of the native
top-8, int4 plus LoRA. The headline 20.4 vs 3.9 tok/s compares against a resident mlx-lm run that thrashes at 18.2 of 24 GB.

**NVIDIA pipelined sharding, MLSys'26 (2604.26334).** **The primary baseline uses -ngl only.** "llama-cpp-baseline" searches for the
largest `-ngl` that fits the budget. The artifact (`paper_results/repro_figure2.sh` L378) runs
`llama-cli ... -ub $ub -ngl $ngl -t $threads`, with no `-cmoe`, `--n-cpu-moe`, `-ot` or `--no-mmap`. The headline averages (3.7x TPS,
up to 30x) are measured against this baseline. A secondary Fig. 3 (qwen30b on cli3 only) adds **`-cmoe`**, which puts *all* experts on
the CPU, and/or `-kvo`. That baseline stays at ~26 tok/s at every budget, which produces the 6.1x at 32G. **`--n-cpu-moe N` was never
tested.** Baseline tok/s are not published; I took exact speedups from `paper_results_figure2.csv`. The Qwen3-30B quant is Q4_0 per the
artifact file name (the paper says "q4").

**MoE-APEX (ASPLOS'26, doi 10.1145/3779212.3790187).** Not accessible: ACM DL returned 403 and ResearchGate returned 429. It has the same
authors as HOBBIT (arXiv 2411.01433; Tang, Liu, Hou, Pu, Wang, Heng, Li, Guo), which appears to be its preprint. I extracted no rows
because they would duplicate or conflict with the 2024 preprint. HOBBIT's llama.cpp description is layer-wise: "places a sufficient
number of layers in GPU memory, with the remaining layers stored in CPU memory or on SSD".

## llama.cpp community tier

The session proxy blocked raw GitHub HTML and API access, and a credentialed attach request was denied. All community quotes therefore
come from WebFetch (model-extracted). I fetched each page 2-3 times and kept only numbers that stayed consistent across passes. One
mobilinkd (Arc Pro B70) report gave two different number sets across passes, so I dropped it.

- **Discussion #24528 (MoE expert cache RFC, leloch).** The headline is multi-GPU (4x3090). Single-GPU reports: batot1's GTX 1080 Ti is
  a *regression*: every cache size is slower than `--cpu-moe` with the cache off (flags fully quoted). nibor1896's RTX 5090 run streams
  from SSD with a custom patch and uses its own build as the baseline. xashr's RTX 5090 compares forks, but the upstream flags are unclear.
- **PR #26563 (miltos22, -ehs).** The 8 GB-VRAM table gives 1.72x / 2.07x on Qwen3.6-35B, but the GPU, CPU, RAM and **stock flags are
  not stated**. buha reports a regression (31.51 to 26.46). siganos (RTX 5090) gives only ranges (22-25 to 32-35).
- **PR #26824 (miltos22, successor).** Laptop RTX 3070 8 GB with 32 GB RAM and an SSD. Stock flags and commit are not stated. The author
  notes that thermal throttling affects the later runs.
- **PR #27861 (csantiago78, draft LRU cache).** The PR's own number is multi-GPU on a host with one RAM channel per socket. sdroege's
  baseline is `-ngl 999 -cmoe`, with all experts on the CPU. **sirfyyn's 2.45x is confounded**: stock ran with default settings while
  the PR run was tuned. sissyhistorian (RX 7600, Vulkan, DDR4-3200) is the cleanest A-tier point at +14.6%.
