# Draft notes to audited teams (not sent)

One section per system. Each lists the rows as we read them, what the model predicts, and the label. Send only after you have checked the rows yourself against the sources (every row in `data/audit/normalized_v2.jsonl` carries the quote and location). Contact addresses are not included; use the corresponding-author address on each paper or the PR/discussion thread.

## Template

> Subject: Your MoE-offloading results in an audit against a decode-time bound (draft for your comments)
>
> Dear {authors},
>
> I am preparing a paper that re-expresses published batch-1 MoE-offloading results against a lower bound on decode time and against a predicted equal-memory llama.cpp `--n-cpu-moe` baseline (preprint and code: github.com/harshithkantamneni/moe-speed-of-light). Your paper is one of the audited systems. Below are the rows I extracted from it, with the source location, and what the model predicts. Nothing in the paper calls any result wrong; the labels only compare against the two references. If I misread a number, a configuration or the hardware, I would be grateful for a correction before the next version, and I am happy to include any comment you want recorded.
>
> {rows}
>
> Each prediction can be reproduced with `python -m mosl.calc` (arguments in the repository's audit rows).
>
> Best regards,
> Harshith Kantamneni

## CPU-GPU collaborative inference (unnamed; arXiv 2512.16473)

Source: (see data/audit)

- `cpugpucollab-mixtral8x7b-rtx4090-24t-b1`: Mixtral-8x7B on RTX 4090; system 4.80 tok/s, reported baseline CPU only (own system, no GPU expert cache) 4.20 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 25/32: 4.9 tok/s [3.6, 6.6]; physical speed-of-light 13.1 tok/s (37% reached); label: not adjudicated: claimed speed-up < 1.2x.
- `cpugpucollab-phi3.5moe-rtx4090-24t-b1`: Phi-3.5-MoE on RTX 4090; system 10.40 tok/s, reported baseline CPU only (own system, no GPU expert cache) 6.30 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 25/32: 10.3 tok/s [7.8, 14.1]; physical speed-of-light 55.8 tok/s (19% reached); label: not established.

## Fate

Source: (see data/audit)

- `fate-qwen1.5moe-rtx3090-b1`: Qwen1.5-MoE-A2.7B (14.3B) on RTX 3090; system 14.00 tok/s, reported baseline EAP (expert-activation-path prefetch) 7.00 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 6/24: 29.4 tok/s [16.5, 62.4]; physical speed-of-light 197.9 tok/s (7% reached); label: not adjudicated: band wider than +-40 %.
- `fate-qwen1.5moe-gtx1080ti-b1`: Qwen1.5-MoE-A2.7B (14.3B) on GTX 1080 Ti; system 3.70 tok/s, reported baseline Load on Demand (LoD) 2.60 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 18/24: 18.5 tok/s [12.9, 27.3]; physical speed-of-light 75.5 tok/s (5% reached); label: not adjudicated: band wider than +-40 %.

## Fiddler

Source: (see data/audit)

- `fiddler-mixtral8x7b-quadrortx6000-b1`: Mixtral-8x7B on Quadro RTX 6000; system 2.53 tok/s, reported baseline llama.cpp 1.70 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 25/32: 4.6 tok/s [2.7, 9.0]; physical speed-of-light 17.3 tok/s (15% reached); label: not adjudicated: band wider than +-40 %.
- `fiddler-mixtral8x7b-rtx6000ada-b1`: Mixtral-8x7B on RTX 6000 Ada; system 8.29 tok/s, reported baseline llama.cpp 7.07 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 17/32: 11.2 tok/s [7.1, 18.7]; physical speed-of-light 57.4 tok/s (14% reached); label: not adjudicated: band wider than +-40 %.

## FreeToken

Source: (see data/audit)

- `freetoken-qwen3.6-35b-a3b-rtx5090srv-w1`: Qwen3.6-35B-A3B on RTX 5090; system 77.10 tok/s, reported baseline llama.cpp 42.60 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 26/40: 35.1 tok/s [24.8, 51.3]; physical speed-of-light 227.1 tok/s (34% reached); label: not adjudicated: band wider than +-40 %.
- `freetoken-deepseek-v4-flash-rtx5090srv-w1`: DeepSeek-V4-Flash (284B, 13B active) on RTX 5090; system 24.90 tok/s, reported baseline llama.cpp 13.00 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 39/43: 17.8 tok/s [12.3, 26.5]; physical speed-of-light 80.8 tok/s (31% reached); label: not adjudicated: band wider than +-40 %.
- `freetoken-qwen3.6-35b-a3b-rtx5090desktop-w2`: Qwen3.6-35B-A3B on RTX 5090; system 73.80 tok/s, reported baseline llama.cpp 33.00 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 24/40: 19.1 tok/s [10.9, 38.9]; physical speed-of-light 204.5 tok/s (36% reached); label: not adjudicated: band wider than +-40 %.
- `freetoken-qwen3.6-35b-a3b-rtx4090srv-w2`: Qwen3.6-35B-A3B on RTX 4090; system 42.90 tok/s, reported baseline llama.cpp 25.80 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 29/40: 21.9 tok/s [14.1, 36.1]; physical speed-of-light 123.1 tok/s (35% reached); label: not adjudicated: band wider than +-40 %.
- `freetoken-qwen3.6-35b-a3b-nvfp4-rtx4060laptop-w2`: Qwen3.6-35B-A3B (official NVFP4 release) on RTX 4060 Laptop; system 39.30 tok/s, reported baseline llama.cpp 22.30 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 31/40: 25.6 tok/s [18.6, 36.2]; physical speed-of-light 97.3 tok/s (40% reached); label: not adjudicated: band wider than +-40 %.
- `freetoken-glm-5.2-nvfp4-rtxpro6000-w1`: GLM-5.2 (753B, 40B active) on RTX PRO 6000 Blackwell; system 14.90 tok/s, reported baseline llama.cpp 7.30 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 65/75: 8.0 tok/s [5.8, 11.3]; physical speed-of-light 29.8 tok/s (50% reached); label: not adjudicated: band wider than +-40 %.

## HybriMoE

Source: (see data/audit)

- `hybrimoe-dsv2lite-a6000-cache25`: DeepSeek-V2-Lite-Chat on RTX A6000; system 14.49 tok/s, reported baseline llama.cpp 11.72 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 20/26: 80.9 tok/s [60.8, 109.7]; physical speed-of-light 470.1 tok/s (3% reached); label: weak baseline; not established.
- `hybrimoe-dsv2lite-a6000-cache75`: DeepSeek-V2-Lite-Chat on RTX A6000; system 15.29 tok/s, reported baseline llama.cpp 8.91 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 7/26: 140.0 tok/s [106.5, 187.6]; physical speed-of-light 655.0 tok/s (2% reached); label: weak baseline; not established.
- `hybrimoe-mixtral8x7b-a6000-cache25`: Mixtral-8x7B-Instruct on RTX A6000; system 4.36 tok/s, reported baseline llama.cpp 3.54 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 24/32: 14.6 tok/s [10.8, 20.1]; physical speed-of-light 53.2 tok/s (8% reached); label: weak baseline; not established.
- `hybrimoe-mixtral8x7b-a6000-cache75`: Mixtral-8x7B-Instruct on RTX A6000; system 7.34 tok/s, reported baseline llama.cpp 7.00 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 8/32: 26.4 tok/s [19.9, 35.7]; physical speed-of-light 132.9 tok/s (6% reached); label: not adjudicated: claimed speed-up < 1.2x.
- `hybrimoe-qwen2-57b-a6000-cache25`: Qwen2-57B-A14B-Instruct on RTX A6000; system 9.62 tok/s, reported baseline llama.cpp 7.36 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 21/28: 19.1 tok/s [14.3, 26.0]; physical speed-of-light 86.3 tok/s (11% reached); label: weak baseline; not established.
- `hybrimoe-qwen2-57b-a6000-cache75`: Qwen2-57B-A14B-Instruct on RTX A6000; system 8.98 tok/s, reported baseline llama.cpp 7.18 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 7/28: 29.8 tok/s [22.7, 39.9]; physical speed-of-light 115.9 tok/s (8% reached); label: weak baseline; not established.

## KTransformers

Source: (see data/audit)

- `ktransformers-ds3-bf16-a100-b1`: DeepSeek-V3-0324 (671B) on A100 40GB; system 5.87 tok/s, reported baseline llama.cpp (custom expert-level offload) 4.68 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 58/58: 5.6 tok/s [4.2, 7.6]; physical speed-of-light 12.0 tok/s (49% reached); label: at strength; not established.
- `ktransformers-ds2-bf16-a100-b1`: DeepSeek-V2.5-1210 (236B) on A100 40GB; system 11.95 tok/s, reported baseline llama.cpp (custom expert-level offload) 7.05 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 59/59: 10.4 tok/s [7.9, 14.0]; physical speed-of-light 24.3 tok/s (49% reached); label: weak baseline; not established.
- `ktransformers-qw2-bf16-a100-b1`: Qwen2-57B-A14B on A100 40GB; system 22.88 tok/s, reported baseline llama.cpp (custom expert-level offload) 13.01 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 28/28: 15.5 tok/s [11.8, 20.9]; physical speed-of-light 35.3 tok/s (65% reached); label: at strength; gain survives.
- `ktransformers-ds3-int4-rtx4080-b1`: DeepSeek-V3-0324 (671B) on RTX 4080; system 16.15 tok/s, reported baseline llama.cpp (custom expert-level offload) 9.06 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 58/58: 13.3 tok/s [10.1, 17.8]; physical speed-of-light 33.4 tok/s (48% reached); label: weak baseline; not established.
- `ktransformers-ds2-int8-rtx4080-b1`: DeepSeek-V2.5-1210 (236B) on RTX 4080; system 17.61 tok/s, reported baseline llama.cpp (custom expert-level offload) 9.94 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 59/59: 12.4 tok/s [9.5, 16.6]; physical speed-of-light 31.9 tok/s (55% reached); label: at strength; gain survives.
- `ktransformers-qw2-int8-rtx4080-b1`: Qwen2-57B-A14B on RTX 4080; system 36.99 tok/s, reported baseline llama.cpp (custom expert-level offload) 19.17 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 28/28: 19.2 tok/s [14.6, 25.6]; physical speed-of-light 48.1 tok/s (77% reached); label: at strength; gain survives.

## MoE expert-cache forks (Miltos22 / Leloch / Lindenburg)

Source: (see data/audit)

- `lcpp-d24528-xashr-laguna-s-2.1-q6k-rtx5090`: Laguna S 2.1 on RTX 5090; system 31.30 tok/s, reported baseline Leloch fork (moe-cache) 27.60 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 35/47: 13.2 tok/s [9.1, 19.8]; physical speed-of-light 88.2 tok/s (35% reached); label: not adjudicated: band wider than +-40 %.

## MoE-SpeQ

Source: (see data/audit)

- `moespeq-dsv2lite-a100-lowmem`: DeepSeek-V2-Lite on A100 40GB (PCIe); system 10.12 tok/s, reported baseline Mixtral-Offloading-SM (same memory) 7.47 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 0/26: 44.9 tok/s [21.4, 159.2]; physical speed-of-light 334.8 tok/s (3% reached); label: not adjudicated: band wider than +-40 %.
- `moespeq-dsv2lite-a100-highmem`: DeepSeek-V2-Lite on A100 40GB (PCIe); system 13.85 tok/s, reported baseline Mixtral-Offloading-SM (same memory) 9.80 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 0/26: 44.9 tok/s [21.4, 159.2]; physical speed-of-light 334.8 tok/s (4% reached); label: not adjudicated: band wider than +-40 %.
- `moespeq-qwen1.5moe-a100-lowmem`: Qwen1.5-MoE-A2.7B (labelled 'Qwen2-MoE' in Fig. 13/text) on A100 40GB (PCIe); system 13.16 tok/s, reported baseline Mixtral-Offloading-SM (same memory) 10.30 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 0/24: 59.9 tok/s [30.6, 162.6]; physical speed-of-light 335.0 tok/s (4% reached); label: not adjudicated: band wider than +-40 %.
- `moespeq-qwen1.5moe-a100-highmem`: Qwen1.5-MoE-A2.7B (labelled 'Qwen2-MoE' in Fig. 13/text) on A100 40GB (PCIe); system 28.33 tok/s, reported baseline Mixtral-Offloading-SM (same memory) 13.50 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 0/24: 59.9 tok/s [30.6, 162.6]; physical speed-of-light 335.0 tok/s (8% reached); label: not adjudicated: band wider than +-40 %.
- `moespeq-phi3.5moe-a100-lowmem`: Phi-3.5-MoE on A100 40GB (PCIe); system 6.05 tok/s, reported baseline Mixtral-Offloading-SM (same memory) 2.59 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 18/32: 10.8 tok/s [6.7, 19.2]; physical speed-of-light 129.0 tok/s (5% reached); label: not adjudicated: band wider than +-40 %.
- `moespeq-phi3.5moe-a100-highmem`: Phi-3.5-MoE on A100 40GB (PCIe); system 6.12 tok/s, reported baseline Mixtral-Offloading-SM (same memory) 3.39 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 18/32: 10.8 tok/s [6.7, 19.2]; physical speed-of-light 129.0 tok/s (5% reached); label: not adjudicated: band wider than +-40 %.

## Pipelined sharding (NVIDIA, llama.cpp-based)

Source: (see data/audit)

- `pipeshard-qwen3-30b-a3b-q4_0-cli3-rtx5090-2G`: Qwen3-30B-A3B-Instruct-2507 on RTX 5090; system 25.70 tok/s, reported baseline llama.cpp (-ngl + -cmoe) 25.70 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 48/48: 49.5 tok/s [38.4, 65.1]; physical speed-of-light 199.1 tok/s (13% reached); label: not adjudicated: claimed speed-up < 1.2x.
- `pipeshard-qwen3-30b-a3b-q4_0-cli3-rtx5090-8G`: Qwen3-30B-A3B-Instruct-2507 on RTX 5090; system 32.10 tok/s, reported baseline llama.cpp (-ngl + -cmoe) 26.10 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 32/48: 69.7 tok/s [54.0, 91.6]; physical speed-of-light 1032.2 tok/s (3% reached); label: weak baseline; not established.
- `pipeshard-qwen3-235b-a22b-q2k-cli3-rtx5090-2G`: Qwen3-235B-A22B-Instruct-2507 on RTX 5090; system 7.70 tok/s, reported baseline llama.cpp (-ngl only) 5.70 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 94/94: 13.1 tok/s [10.2, 17.3]; physical speed-of-light 39.9 tok/s (19% reached); label: weak baseline; not established.
- `pipeshard-qwen3-235b-a22b-q2k-cli3-rtx5090-32G`: Qwen3-235B-A22B-Instruct-2507 on RTX 5090; system 11.50 tok/s, reported baseline llama.cpp (-ngl only) 8.46 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 63/94: 18.2 tok/s [14.1, 23.9]; physical speed-of-light 107.9 tok/s (11% reached); label: weak baseline; not established.
- `pipeshard-qwen3-30b-a3b-q4_0-cli2-rtx5070ti-16G`: Qwen3-30B-A3B-Instruct-2507 on RTX 5070 Ti; system 54.90 tok/s, no baseline reported; predicted equal-memory llama.cpp --n-cpu-moe 8/48: 90.5 tok/s [70.1, 118.9]; physical speed-of-light 510.5 tok/s (11% reached); label: not adjudicated: no reported baseline.

## ProMoE

Source: (see data/audit)

- `promoe-llamacpp-deepseekv2lite-fp16-rtx4090-b1`: DeepSeek-V2-Lite (paper: DS-2) on RTX 4090; system 49.86 tok/s, reported baseline LRU cache (authors', in llama.cpp) 45.94 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 8/26: 19.0 tok/s [9.8, 50.5]; physical speed-of-light 215.6 tok/s (23% reached); label: not adjudicated: band wider than +-40 %.
- `promoe-llamacpp-qwen15moe-fp16-rtx4090-b1`: Qwen1.5-MoE-A2.7B (paper: QW-1) on RTX 4090; system 67.19 tok/s, reported baseline LRU cache (authors', in llama.cpp) 59.63 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 6/24: 27.6 tok/s [14.5, 68.8]; physical speed-of-light 216.2 tok/s (31% reached); label: not adjudicated: band wider than +-40 %.
- `promoe-llamacpp-qwen2moe57b-int4-rtx4090-b1`: Qwen2-57B-A14B (paper: QW-2) on RTX 4090; system 26.86 tok/s, reported baseline LRU cache (authors', in llama.cpp) 23.00 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 10/28: 13.7 tok/s [7.3, 32.8]; physical speed-of-light 135.8 tok/s (20% reached); label: not adjudicated: band wider than +-40 %.
- `promoe-llamacpp-mixtral8x7b-int4-rtx4090-b1`: Mixtral-8x7B (paper: Mixt) on RTX 4090; system 27.02 tok/s, reported baseline llama.cpp layer offload (LO) 37.12 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 5/32: 9.9 tok/s [4.6, 39.2]; physical speed-of-light 151.5 tok/s (18% reached); label: not adjudicated: band wider than +-40 %.

## SP-MoE

Source: (see data/audit)

- `spmoe-mixtral8x7b-rtx3090-b1`: Mixtral-8x7B (draft Mistral-7B) on RTX 3090; system 1.20 tok/s, reported baseline AdapMoE+SD 1.05 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 26/32: 5.3 tok/s [3.7, 7.7]; physical speed-of-light 15.5 tok/s (8% reached); label: not adjudicated: band wider than +-40 %.
- `spmoe-mixtral8x7b-rtx4090-b1`: Mixtral-8x7B (draft Mistral-7B) on RTX 4090; system 1.28 tok/s, reported baseline AdapMoE+SD 1.09 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 26/32: 4.4 tok/s [3.0, 6.6]; physical speed-of-light 13.4 tok/s (10% reached); label: not adjudicated: band wider than +-40 %.
- `spmoe-mixtral8x7b-a100-40g-b1`: Mixtral-8x7B (draft Mistral-7B) on A100 40GB; system 1.46 tok/s, reported baseline AdapMoE+SD 1.32 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 20/32: 5.9 tok/s [3.7, 9.7]; physical speed-of-light 32.2 tok/s (5% reached); label: not adjudicated: band wider than +-40 %.
- `spmoe-phi3.5moe-rtx4090-b1`: Phi-3.5-MoE (draft Phi-mini-MoE) on RTX 4090; system 3.31 tok/s, reported baseline AdapMoE+SD 2.80 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 25/32: 9.5 tok/s [6.5, 14.4]; physical speed-of-light 56.9 tok/s (6% reached); label: not adjudicated: band wider than +-40 %.
- `spmoe-deepseekv2lite-a100-gpumem7g-b1`: DeepSeek-V2-Lite (draft Deepseek-Lite-AWQ) on A100 40GB; system 5.56 tok/s, reported baseline Mixtral-Offloading+SD 3.70 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 24/26: 35.4 tok/s [26.5, 48.0]; physical speed-of-light 142.0 tok/s (4% reached); label: not established.

## SeqMoE

Source: (see data/audit)

- `seqmoe-qwen3-30b-a3b-fp8-rtx4090-45pct`: Qwen3-30B-A3B-FP8 on RTX 4090; system 104.10 tok/s, reported baseline llama.cpp 30.90 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 27/48: 61.7 tok/s [46.9, 82.6]; physical speed-of-light 373.7 tok/s (28% reached); label: weak baseline; gain survives.
- `seqmoe-qwen3-30b-a3b-fp8-rtx4090-15pct`: Qwen3-30B-A3B-FP8 on RTX 4090; system 47.90 tok/s, reported baseline llama.cpp 21.10 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 41/48: 48.9 tok/s [37.0, 65.7]; physical speed-of-light 373.7 tok/s (13% reached); label: weak baseline; not established.
- `seqmoe-qwen3.6-35b-a3b-bf16-rtx5090-40pct`: Qwen3.6-35B-A3B-BF16 on RTX 5090; system 118.30 tok/s, reported baseline llama.cpp 37.90 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 24/40: 56.4 tok/s [43.0, 75.4]; physical speed-of-light 305.1 tok/s (39% reached); label: weak baseline; gain survives.
- `seqmoe-qwen3.6-35b-a3b-bf16-rtx5090-15pct`: Qwen3.6-35B-A3B-BF16 on RTX 5090; system 70.20 tok/s, reported baseline llama.cpp 28.20 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 34/40: 46.5 tok/s [35.3, 62.3]; physical speed-of-light 211.5 tok/s (33% reached); label: weak baseline; gain survives.
- `seqmoe-gpt-oss-120b-mxfp4-rtxpro6000-45pct`: GPT-OSS-120B-MXFP4 on RTX PRO 6000 Blackwell; system 139.20 tok/s, reported baseline llama.cpp 33.10 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 20/36: 74.3 tok/s [56.5, 99.6]; physical speed-of-light 377.9 tok/s (37% reached); label: weak baseline; gain survives.
- `seqmoe-deepseek-v4-flash-rtxpro6000-45pct`: DeepSeek-V4-Flash on RTX PRO 6000 Blackwell; system 57.40 tok/s, reported baseline llama.cpp 21.00 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 24/43: 38.3 tok/s [29.1, 51.2]; physical speed-of-light 183.2 tok/s (31% reached); label: weak baseline; gain survives.

## llama.cpp GPU-resident LRU expert cache PR #27861 (csantiago78, draft; --moe-expert-cache)

Source: (see data/audit)

- `lcpp-pr27861-sissyhistorian-qwen3-30b-a3b-q4km-rx7600`: Qwen3-30B-A3B Q4_K_M derivative (architecture assumed identical) on Radeon RX 7600 8GB; system 16.50 tok/s, reported baseline llama.cpp same build, cache OFF 14.40 tok/s; predicted equal-memory llama.cpp --n-cpu-moe 36/48: 23.6 tok/s [18.3, 31.0]; physical speed-of-light 168.6 tok/s (10% reached); label: not adjudicated: claimed speed-up < 1.2x.

## Rows not modelled

- `promoe-llamacpp-deepseekmoe16b-fp16-rtx4090-b1`: architecture needs remote code (not in transformers)
- `fate-deepseekmoe16b-rtx3090-b1`: architecture needs remote code (not in transformers)
- `fate-deepseekmoe16b-gtx1080ti-b1`: architecture needs remote code (not in transformers)
- `lcpp-pr27861-sdroege-qwen3.8-flash-next-q4kxl-r9700`: per-layer n-gram lookup table is not a routed expert or dense weight in the model
- `lcpp-pr27861-sirfyyn-qwen3.8-flash-next-q4kxl-rtxpro4500`: per-layer n-gram lookup table is not a routed expert or dense weight in the model
