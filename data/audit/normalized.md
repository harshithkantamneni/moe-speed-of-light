# Audit rows normalized for the model (PROTOCOL 4.6)

Inputs: `rows_group{1,2,3}.jsonl` (147 rows) and `notes_group{1,2,3}.md`. Outputs: `normalized.jsonl` (58 adjudicable rows), `excluded.jsonl` (89 rows, each with a category and a one-line reason), and `tables.py` (GPU, PCIe, bits-per-weight and CPU-platform tables that extend `mosl/validation_set.py`). This is curation, not new measurement: every number comes from the row, a datasheet or a format definition, and every other value is an assumption listed in the row's `assume`.

## How rows were admitted

A row is adjudicable when all of these hold:

1. A single-request (batch 1) decode speed or TPOT, with an absolute number for the system.
2. Experts partly in host DRAM. Rows with an SSD tier, unified memory, several GPUs, or all experts in VRAM are out.
3. One named GPU.
4. Tier A or B under 4.6: model, quantization and GPU are stated, the host is identified, and at most two of {host DRAM configuration, PCIe generation, GPU expert budget} are missing. Missing items are imputed as a band.

Rows that change routing (deferral, skipping, substitution) or outputs (fewer bits than the baseline) stay in, flagged `routing_exact: false` / `lossy: true`. Speculative-decoding rows also stay in, with `speculative: true`.

Five interpretation choices matter. Each is applied the same way in all three groups:

- **Host identified.** 4.6 lists 'host' among the tier-A items and lets only the *DRAM configuration* be missing in tier B. A row therefore needs either a named CPU (model, or at least vendor family, e.g. 'AMD EPYC, 16 cores') or a stated DRAM configuration (type+speed, channels or bandwidth). Without either there is no platform to impute a DRAM band from, so the row is tier C. Group 1's notes used a looser reading (CPU not needed); under it Mixtral-offloading (6), AdapMoE (6), HOBBIT RTX 4090 (4), FloE (3), MoE-Infinity DeepSeek-V2-Lite (1), Speculating Experts A6000 (2) and AcceptMoE (4) would enter (26 rows), and possibly 4 community rows (batot1, PR #26824 x2, buha), all with host bands several-fold wide. The one row this rule moves *up* is `lcpp-pr27861-sdroege` (C -> B): its DRAM is stated as DDR5-3600 but its CPU is not.
- **Quantization stated.** A stated model or per-expert size that fixes the dtype uniquely counts as stated: SP-MoE and the CPU-GPU collaborative paper (2512.16473). A dtype inferred from budget arithmetic does not (DAOP). Neither does a missing GGUF type (BuddyMoE, MoE-Gen). SeqMoE's DeepSeek-V4-Flash row names only the model; its official checkpoint format (MXFP4 experts, FP8 dense) is assumed and flagged.
- **Batch not stated.** HybriMoE and MoE-SpeQ report per-token TBT/TPOT for a local single-stream system without saying 'batch 1'. Both are kept, with the assumption recorded. (AdapMoE would be too, but it fails the host rule.)
- **Metric.** Fiddler reports end-to-end tok/s. The [32 in, 256 out] configuration is used as the decode number, flagged: it slightly understates pure decode speed. FreeToken reports per-request mean decode tok/s over agentic single-stream sessions. `metric` records the form of each number.
- **b_exp / b_dense** are the bits of the model as the row's baseline serves it, i.e. the reference precision. When a system moves fewer bits (Fate's INT4 transfers), `quant_note` says so and `lossy` is true.

## Imputation rules

- **R1, desktop CPU, DRAM unstated:** 2 channels x [low, high] common speeds for the platform. The caller's rule for Raptor Lake is DDR5-4800..6000. For ProMoE (i9-14900K, 128 GB) the lower edge drops to DDR5-4000, the Intel 2-DIMM-per-channel speed that four DIMMs imply. FreeToken's 9950X3D desktop (192 GiB, two channels stated) uses DDR5-3600..5600 on the same basis.
- **R2, server/HEDT CPU, DRAM unstated:** every channel of one socket populated, speed from one grade below rated (or the 2DPC speed) up to rated. Values are in `tables.CPU`. When the DDR type is stated but channels/speed are not (KTransformers, FreeToken servers), the same rule fixes the band.
- **R3, dual socket:** KTransformers is NUMA-aware, so both sockets count, as for P038 in `validation_set.py`. FreeToken's rented servers were pinned to the GPU's NUMA node, so one socket counts. For Fiddler, whose NUMA use is unknown, the band spans one socket to both. `sockets` records the host.
- **PCIe unstated:** min(GPU link, CPU-platform link) at the GPU's lane count; `bw_pcie_band` widens this when the platform is unknown. HybriMoE's Cascade Lake host caps its A6000 at Gen3 x16.
- **Budget unstated:** `{"kind": "gpu_mem_total", "value": <card memory>, "imputed": true}`, an upper bound. MoE-SpeQ's 'low' and 'high' settings are never given in GB, so both rows get the same bound.
- **ctx:** prompt + output/2. A missing prompt is taken as 128 tokens and a missing output as 256; each row says so.
- **Measured bandwidths** (FreeToken's B_H, KTransformers' MLC 220 GB/s) are kept in `host_bw_measured`. `bw_cpu` stays a datasheet peak, as the model requires.

Fields beyond the requested schema: `model`, `gpu_mem_gb`, `bw_pcie_band`, `sockets`, `host_bw_measured`, `metric`, `figure_only`, `tier_row` (the extractor's tier), `budget.imputed`, `baselines[].note`. Baselines with no number (e.g. 'cannot serve') are dropped. Pipelined-sharding baseline tok/s are *derived* as system TPS / published speedup, and marked so.

## Adjudicable rows (58)

Host = `bw_cpu` band (GB/s peak). Flags: R = routing not exact, L = lossy, S = speculative, F = figure-only, imp. = imputed. Baseline classes: ngl = `-ngl` layer offload, ncmoe = `--n-cpu-moe`/`-ot`, cmoe = `-cmoe` (all experts on CPU), fork = llama.cpp custom fork, lcpp? = llama.cpp with unknown config, other = other engine.

| id | system | model (repo) | GPU | host GB/s | budget | tok/s | baselines tok/s (class) | tier / missing | flags |
|---|---|---|---|---|---|---|---|---|---|
| `fiddler-mixtral8x7b-quadrortx6000-b1` | Fiddler | Mixtral-8x7B-v0.1 | Quadro RTX 6000 | 115.2-255.9 | 22% of experts | 2.53 | llama.cpp 1.7 (ngl); Mixtral-Offloading (Eliseev & Ma.. 0.32 (other); DeepSpeed-MII (ZeRO-Infinity) 0.13 (other) | B / DRAM | F |
| `fiddler-mixtral8x7b-rtx6000ada-b1` | Fiddler | Mixtral-8x7B-v0.1 | RTX 6000 Ada | 281.6-614.4 | 49% of experts | 8.29 | llama.cpp 7.07 (ngl); Mixtral-Offloading (Eliseev & Ma.. 0.94 (other); DeepSpeed-MII (ZeRO-Infinity) 0.22 (other) | B / DRAM | F |
| `promoe-llamacpp-deepseekmoe16b-fp16-rtx4090-b1` | ProMoE | deepseek-moe-16b-base | RTX 4090 | 64-96 | <=24 GB VRAM (imp.) | 41.9 | llama.cpp layer offload (LO) 24.9 (ngl); LRU cache 38.9 (fork); Static cache 24.6 (fork); Unified Memory (UM) 22.6 (fork) | B / DRAM,budget | F |
| `promoe-llamacpp-deepseekv2lite-fp16-rtx4090-b1` | ProMoE | DeepSeek-V2-Lite | RTX 4090 | 64-96 | <=24 GB VRAM (imp.) | 50.2 | llama.cpp layer offload (LO) 27.2 (ngl); LRU cache 46.2 (fork); Static cache 28.2 (fork); Unified Memory (UM) 26.1 (fork) | B / DRAM,budget | F |
| `promoe-llamacpp-qwen15moe-fp16-rtx4090-b1` | ProMoE | Qwen1.5-MoE-A2.7B | RTX 4090 | 64-96 | <=24 GB VRAM (imp.) | 67.5 | llama.cpp layer offload (LO) 32.8 (ngl); LRU cache 60 (fork); Static cache 47.6 (fork); Unified Memory (UM) 40.6 (fork) | B / DRAM,budget | F |
| `promoe-llamacpp-qwen2moe57b-int4-rtx4090-b1` | ProMoE | Qwen2-57B-A14B | RTX 4090 | 64-96 | <=24 GB VRAM (imp.) | 27.2 | llama.cpp layer offload (LO) 15 (ngl); LRU cache 23.3 (fork); Static cache 19.6 (fork); Unified Memory (UM) 16 (fork) | B / DRAM,budget | F |
| `promoe-llamacpp-mixtral8x7b-int4-rtx4090-b1` | ProMoE | Mixtral-8x7B-v0.1 | RTX 4090 | 64-96 | <=24 GB VRAM (imp.) | 27.4 | llama.cpp layer offload (LO) 37.4 (ngl); LRU cache 27.5 (fork); Static cache 24.5 (fork); Unified Memory (UM) 21.6 (fork) | B / DRAM,budget | F |
| `ktransformers-ds3-bf16-a100-b1` | KTransformers | DeepSeek-V3 | A100 40GB | 563.2-614.4 | 0 experts/layer | 5.87 | llama.cpp (custom expert-level o.. 4.68 (fork); Fiddler 2.43 (other) | A / - | F |
| `ktransformers-ds2-bf16-a100-b1` | KTransformers | DeepSeek-V2.5-1210 | A100 40GB | 563.2-614.4 | 0 experts/layer | 11.95 | llama.cpp (custom expert-level o.. 7.05 (fork); Fiddler 3.54 (other) | A / - | F |
| `ktransformers-qw2-bf16-a100-b1` | KTransformers | Qwen2-57B-A14B | A100 40GB | 563.2-614.4 | 0 experts/layer | 22.88 | llama.cpp (custom expert-level o.. 13.01 (fork); Fiddler 5.59 (other) | A / - | F |
| `ktransformers-ds3-int4-rtx4080-b1` | KTransformers | DeepSeek-V3 | RTX 4080 | 563.2-614.4 | 0 experts/layer | 16.15 | llama.cpp (custom expert-level o.. 9.06 (fork) | A / - | F |
| `ktransformers-ds2-int8-rtx4080-b1` | KTransformers | DeepSeek-V2.5-1210 | RTX 4080 | 563.2-614.4 | 0 experts/layer | 17.61 | llama.cpp (custom expert-level o.. 9.94 (fork) | A / - | F |
| `ktransformers-qw2-int8-rtx4080-b1` | KTransformers | Qwen2-57B-A14B | RTX 4080 | 563.2-614.4 | 0 experts/layer | 36.99 | llama.cpp (custom expert-level o.. 19.17 (fork) | A / - | F |
| `hybrimoe-dsv2lite-a6000-cache25` | HybriMoE | DeepSeek-V2-Lite | RTX A6000 | 115.2-128 | 25% of experts | 14.49 | llama.cpp 11.72 (ngl); kTransformers 8.12 (other); AdapMoE 7.59 (other) | B / DRAM,PCIe | F |
| `hybrimoe-dsv2lite-a6000-cache75` | HybriMoE | DeepSeek-V2-Lite | RTX A6000 | 115.2-128 | 75% of experts | 15.29 | llama.cpp 8.91 (ngl); kTransformers 9.4 (other); AdapMoE 8.35 (other) | B / DRAM,PCIe | F |
| `hybrimoe-mixtral8x7b-a6000-cache25` | HybriMoE | Mixtral-8x7B-v0.1 | RTX A6000 | 115.2-128 | 25% of experts | 4.36 | llama.cpp 3.54 (ngl); kTransformers 3.26 (other); AdapMoE 2.13 (other) | B / DRAM,PCIe | F |
| `hybrimoe-mixtral8x7b-a6000-cache75` | HybriMoE | Mixtral-8x7B-v0.1 | RTX A6000 | 115.2-128 | 75% of experts | 7.34 | llama.cpp 7 (ngl); kTransformers 4.88 (other); AdapMoE 4.23 (other) | B / DRAM,PCIe | F |
| `hybrimoe-qwen2-57b-a6000-cache25` | HybriMoE | Qwen2-57B-A14B | RTX A6000 | 115.2-128 | 25% of experts | 9.62 | llama.cpp 7.36 (ngl); kTransformers 4.33 (other); AdapMoE 4.97 (other) | B / DRAM,PCIe | F |
| `hybrimoe-qwen2-57b-a6000-cache75` | HybriMoE | Qwen2-57B-A14B | RTX A6000 | 115.2-128 | 75% of experts | 8.98 | llama.cpp 7.18 (ngl); kTransformers 4.95 (other); AdapMoE 5.17 (other) | B / DRAM,PCIe | F |
| `fate-qwen1.5moe-rtx3090-b1` | Fate | Qwen1.5-MoE-A2.7B | RTX 3090 | 76.8-85.3 | <=24 GB VRAM (imp.) | 14 | Load on Demand (LoD) 3.9 (other); EAP (expert-activation-path pref.. 7 (other) | B / DRAM,budget | LF |
| `fate-deepseekmoe16b-rtx3090-b1` | Fate | deepseek-moe-16b-base | RTX 3090 | 76.8-85.3 | <=24 GB VRAM (imp.) | 8.9 | Load on Demand (LoD) 2.4 (other); EAP (expert-activation-path pref.. 4.1 (other) | B / DRAM,budget | LF |
| `fate-qwen1.5moe-gtx1080ti-b1` | Fate | Qwen1.5-MoE-A2.7B | GTX 1080 Ti | 68.3-76.8 | <=11 GB VRAM (imp.) | 3.7 | Load on Demand (LoD) 2.6 (other); EAP (expert-activation-path pref.. 2.2 (other) | B / DRAM,budget | LF |
| `fate-deepseekmoe16b-gtx1080ti-b1` | Fate | deepseek-moe-16b-base | GTX 1080 Ti | 68.3-76.8 | <=11 GB VRAM (imp.) | 2.1 | Load on Demand (LoD) 1.5 (other); EAP (expert-activation-path pref.. 1.4 (other) | B / DRAM,budget | LF |
| `spmoe-mixtral8x7b-rtx3090-b1` | SP-MoE | Mixtral-8x7B-v0.1 | RTX 3090 | 187.7-204.8 | <=24 GB VRAM (imp.) | 1.2 | Mixtral-Offloading+SD 0.69 (other); AdapMoE+SD 1.05 (other); MoE-Infinity+SD 0.97 (other) | B / DRAM,budget | SF |
| `spmoe-mixtral8x7b-rtx4090-b1` | SP-MoE | Mixtral-8x7B-v0.1 | RTX 4090 | 153.6-170.6 | <=24 GB VRAM (imp.) | 1.28 | Mixtral-Offloading+SD 0.7 (other); AdapMoE+SD 1.09 (other); MoE-Infinity+SD 0.9 (other) | B / DRAM,budget | SF |
| `spmoe-mixtral8x7b-a100-40g-b1` | SP-MoE | Mixtral-8x7B-v0.1 | A100 40GB | 187.7-204.8 | <=40 GB VRAM (imp.) | 1.46 | Mixtral-Offloading+SD 0.82 (other); AdapMoE+SD 1.32 (other); MoE-Infinity+SD 1.03 (other) | B / DRAM,budget | SF |
| `spmoe-phi3.5moe-rtx4090-b1` | SP-MoE | Phi-3.5-MoE-instruct | RTX 4090 | 153.6-170.6 | <=24 GB VRAM (imp.) | 3.31 | Mixtral-Offloading+SD 1.99 (other); AdapMoE+SD 2.8 (other); MoE-Infinity+SD 1.12 (other) | B / DRAM,budget | SF |
| `spmoe-deepseekv2lite-a100-gpumem7g-b1` | SP-MoE | DeepSeek-V2-Lite | A100 40GB | 187.7-204.8 | 7 GB VRAM | 5.56 | Mixtral-Offloading+SD 3.7 (other); MoE-Infinity+SD 2.56 (other) | B / DRAM | S |
| `spmoe-deepseekv2lite-a100-gpumem39g-b1` | SP-MoE | DeepSeek-V2-Lite | A100 40GB | 187.7-204.8 | 39 GB VRAM | 10 | Mixtral-Offloading+SD 10 (other); MoE-Infinity+SD 4.35 (other) | B / DRAM | S |
| `moespeq-dsv2lite-a100-lowmem` | MoE-SpeQ | DeepSeek-V2-Lite | A100 40GB (PCIe) | 153.6-170.6 | <=40 GB VRAM (imp.) | 10.12 | Mixtral-Offloading-SM (same memo.. 7.47 (other); Mixtral-Offloading-SC (same cache) 4.53 (other); HF Transformers (device_map laye.. 2.46 (other) | B / DRAM,budget | SF |
| `moespeq-dsv2lite-a100-highmem` | MoE-SpeQ | DeepSeek-V2-Lite | A100 40GB (PCIe) | 153.6-170.6 | <=40 GB VRAM (imp.) | 13.85 | Mixtral-Offloading-SM (same memo.. 9.8 (other); Mixtral-Offloading-SC (same cache) 6.73 (other); HF Transformers (device_map laye.. 2.9 (other) | B / DRAM,budget | SF |
| `moespeq-qwen1.5moe-a100-lowmem` | MoE-SpeQ | Qwen1.5-MoE-A2.7B | A100 40GB (PCIe) | 153.6-170.6 | <=40 GB VRAM (imp.) | 13.16 | Mixtral-Offloading-SM (same memo.. 10.3 (other); Mixtral-Offloading-SC (same cache) 7.74 (other); HF Transformers (device_map laye.. 2.96 (other) | B / DRAM,budget | SF |
| `moespeq-qwen1.5moe-a100-highmem` | MoE-SpeQ | Qwen1.5-MoE-A2.7B | A100 40GB (PCIe) | 153.6-170.6 | <=40 GB VRAM (imp.) | 28.33 | Mixtral-Offloading-SM (same memo.. 13.5 (other); Mixtral-Offloading-SC (same cache) 9.8 (other); HF Transformers (device_map laye.. 3.87 (other) | B / DRAM,budget | SF |
| `moespeq-phi3.5moe-a100-lowmem` | MoE-SpeQ | Phi-3.5-MoE-instruct | A100 40GB (PCIe) | 153.6-170.6 | <=40 GB VRAM (imp.) | 6.05 | Mixtral-Offloading-SM (same memo.. 2.59 (other); Mixtral-Offloading-SC (same cache) 1.44 (other); HF Transformers (device_map laye.. 0.61 (other) | B / DRAM,budget | SF |
| `moespeq-phi3.5moe-a100-highmem` | MoE-SpeQ | Phi-3.5-MoE-instruct | A100 40GB (PCIe) | 153.6-170.6 | <=40 GB VRAM (imp.) | 6.12 | Mixtral-Offloading-SM (same memo.. 3.39 (other); Mixtral-Offloading-SC (same cache) 1.86 (other); HF Transformers (device_map laye.. 1.04 (other) | B / DRAM,budget | SF |
| `cpugpucollab-mixtral8x7b-rtx4090-24t-b1` | CPU-GPU collab (2512.16473) | Mixtral-8x7B-v0.1 | RTX 4090 | 153.6-166.4 | 22% of experts | 4.8 | Fiddler 2.7 (other); CPU only (own system, no GPU exp.. 4.2 (other); On-demand expert fetching 1.1 (other); Pre-gated MoE (ESTIMATED, not run) 1.1 (other) | B / DRAM | F |
| `cpugpucollab-phi3.5moe-rtx4090-24t-b1` | CPU-GPU collab (2512.16473) | Phi-3.5-MoE-instruct | RTX 4090 | 153.6-166.4 | 24% of experts | 10.4 | Fiddler 0.2 (other); CPU only (own system, no GPU exp.. 6.3 (other); On-demand expert fetching 2.3 (other); Pre-gated MoE (ESTIMATED, not run) 2.4 (other) | B / DRAM | F |
| `freetoken-qwen3.6-35b-a3b-rtx5090srv-w1` | FreeToken | Qwen3.6-35B-A3B | RTX 5090 | 281.6-307.2 | 37% of experts | 77.1 | llama.cpp 42.6 (lcpp?); KTransformers 32.5 (other); Ollama 33.1 (lcpp?); MoE-Infinity 8.8 (other) | A / - | F |
| `freetoken-deepseek-v4-flash-rtx5090srv-w1` | FreeToken | DeepSeek-V4-Flash | RTX 5090 | 281.6-307.2 | 11% of experts | 24.9 | llama.cpp 13 (lcpp?); KTransformers 10.4 (other) | A / - | F |
| `freetoken-qwen3.6-35b-a3b-rtx5090desktop-w2` | FreeToken | Qwen3.6-35B-A3B | RTX 5090 | 57.6-89.6 | <=32 GB VRAM (imp.) | 73.8 | llama.cpp 33 (lcpp?); KTransformers 34.8 (other); Ollama 24.9 (lcpp?) | B / budget | F |
| `freetoken-qwen3.6-35b-a3b-rtx4090srv-w2` | FreeToken | Qwen3.6-35B-A3B | RTX 4090 | 187.7-204.8 | <=24 GB VRAM (imp.) | 42.9 | llama.cpp 25.8 (lcpp?); KTransformers 31.8 (other); Ollama 14.1 (lcpp?) | B / budget | F |
| `freetoken-qwen3.6-35b-a3b-nvfp4-rtx4060laptop-w2` | FreeToken | Qwen3.6-35B-A3B | RTX 4060 Laptop | 76.8-102.4 | <=8 GB VRAM (imp.) | 39.3 | llama.cpp 22.3 (lcpp?); Ollama 18.1 (lcpp?) | B / budget | F |
| `freetoken-glm-5.2-nvfp4-rtxpro6000-w1` | FreeToken | GLM-5.2 | RTX PRO 6000 Blackwell | 307.2-358.4 | <=96 GB VRAM (imp.) | 14.9 | llama.cpp 7.3 (lcpp?) | B / budget | - |
| `seqmoe-qwen3-30b-a3b-fp8-rtx4090-45pct` | SeqMoE | Qwen3-30B-A3B | RTX 4090 | 256-281.6 | 45% of experts | 104.1 | llama.cpp 30.9 (lcpp?); KTransformers 34.2 (other); MoE-Infinity 11.9 (other) | B / DRAM | F |
| `seqmoe-qwen3-30b-a3b-fp8-rtx4090-15pct` | SeqMoE | Qwen3-30B-A3B | RTX 4090 | 256-281.6 | 15% of experts | 47.9 | llama.cpp 21.1 (lcpp?); KTransformers 25.9 (other); MoE-Infinity 6.8 (other) | B / DRAM | F |
| `seqmoe-qwen3.6-35b-a3b-bf16-rtx5090-40pct` | SeqMoE | Qwen3.6-35B-A3B | RTX 5090 | 281.6-307.2 | 40% of experts | 118.3 | llama.cpp 37.9 (lcpp?); KTransformers 35 (other); MoE-Infinity 12.6 (other); FreeToken 86.2 (other) | B / DRAM | F |
| `seqmoe-qwen3.6-35b-a3b-bf16-rtx5090-15pct` | SeqMoE | Qwen3.6-35B-A3B | RTX 5090 | 281.6-307.2 | 15% of experts | 70.2 | llama.cpp 28.2 (lcpp?); KTransformers 26.9 (other); MoE-Infinity 7.3 (other); FreeToken 46.3 (other) | B / DRAM | F |
| `seqmoe-gpt-oss-120b-mxfp4-rtxpro6000-45pct` | SeqMoE | gpt-oss-120b | RTX PRO 6000 Blackwell | 281.6-307.2 | 45% of experts | 139.2 | llama.cpp 33.1 (lcpp?); MoE-Infinity 16.9 (other); FreeToken 101.2 (other) | B / DRAM | F |
| `seqmoe-deepseek-v4-flash-rtxpro6000-45pct` | SeqMoE | DeepSeek-V4-Flash | RTX PRO 6000 Blackwell | 281.6-307.2 | 45% of experts | 57.4 | llama.cpp 21 (lcpp?); KTransformers 14.8 (other); FreeToken 34 (other) | B / DRAM | F |
| `pipeshard-qwen3-30b-a3b-q4_0-cli3-rtx5090-2G` | Pipelined sharding (NVIDIA, llama.cpp-based) | Qwen3-30B-A3B-Instruct-2507 | RTX 5090 | 153.6 | 2 GB VRAM | 25.7 | llama.cpp (-ngl only) 19.32 (ngl); llama.cpp (-ngl + -cmoe) 25.7 (cmoe) | A / - | - |
| `pipeshard-qwen3-30b-a3b-q4_0-cli3-rtx5090-8G` | Pipelined sharding (NVIDIA, llama.cpp-based) | Qwen3-30B-A3B-Instruct-2507 | RTX 5090 | 153.6 | 8 GB VRAM | 32.1 | llama.cpp (-ngl only) 25.08 (ngl); llama.cpp (-ngl + -cmoe) 26.75 (cmoe) | A / - | - |
| `pipeshard-qwen3-235b-a22b-q2k-cli3-rtx5090-2G` | Pipelined sharding (NVIDIA, llama.cpp-based) | Qwen3-235B-A22B | RTX 5090 | 153.6 | 2 GB VRAM | 7.7 | llama.cpp (-ngl only) 5.7 (ngl) | A / - | - |
| `pipeshard-qwen3-235b-a22b-q2k-cli3-rtx5090-32G` | Pipelined sharding (NVIDIA, llama.cpp-based) | Qwen3-235B-A22B | RTX 5090 | 153.6 | 32 GB VRAM | 11.5 | llama.cpp (-ngl only) 8.46 (ngl) | A / - | - |
| `pipeshard-qwen3-30b-a3b-q4_0-cli2-rtx5070ti-16G` | Pipelined sharding (NVIDIA, llama.cpp-based) | Qwen3-30B-A3B-Instruct-2507 | RTX 5070 Ti | 57.6 | 16 GB VRAM | 54.9 | none | A / - | - |
| `lcpp-d24528-xashr-laguna-s-2.1-q6k-rtx5090` | MoE expert-cache forks (Miltos22 / Leloch / Lindenburg) | Laguna-S-2.1 | RTX 5090 | 96 | <=32 GB VRAM (imp.) | 31.3 | llama.cpp (upstream) 18.2 (lcpp?); Leloch fork (moe-cache) 27.6 (fork); Lindenburg fork 25.6 (fork) | B / PCIe,budget | - |
| `lcpp-pr27861-sdroege-qwen3.8-flash-next-q4kxl-r9700` | llama.cpp GPU-resident LRU expert cache PR #27861 (csantiago78, draft; --moe-expert-cache) | Qwen3.8-Flash-Next | AMD Radeon AI PRO R9700 | 57.6-115.2 | 84 experts/layer | 18 | llama.cpp same build, cache disa.. 12.8 (cmoe); llama.cpp same build, '--fit on' 13 (ncmoe) | B / PCIe | - |
| `lcpp-pr27861-sirfyyn-qwen3.8-flash-next-q4kxl-rtxpro4500` | llama.cpp GPU-resident LRU expert cache PR #27861 (csantiago78, draft; --moe-expert-cache) | Qwen3.8-Flash-Next | RTX PRO 4500 Blackwell | 96 | 96 experts/layer | 41.31 | llama.cpp stock 62acc89c, defaul.. 16.83 (lcpp?); PR build, default settings 21.95 (fork) | A / PCIe | - |
| `lcpp-pr27861-sissyhistorian-qwen3-30b-a3b-q4km-rx7600` | llama.cpp GPU-resident LRU expert cache PR #27861 (csantiago78, draft; --moe-expert-cache) | Qwen3-30B-A3B | Radeon RX 7600 8GB | 51.2 | 32 experts/layer | 16.5 | llama.cpp same build, cache OFF 14.4 (lcpp?) | A / PCIe | - |

Rows with at least one llama.cpp-family baseline (the P5 population): 39 of 58. Baselines by class: other 88, fork 24, lcpp? 19, ngl 17, cmoe 3, ncmoe 1. No adjudicable row has a measured `--n-cpu-moe` baseline; the closest is sdroege's `--fit on` run, classed ncmoe on the assumption that auto-fit offloads experts partially.

## Exclusions (89)

| reason | rows |
|---|---|
| tier C: host not identified | 32 |
| not batch-1 decode | 15 |
| SSD or unified-memory tier | 15 |
| tier C: quantization not stated | 12 |
| multi-GPU or no offloading | 5 |
| no usable single-request decode number | 4 |
| not a real single PCIe GPU + host (simulator, NVLink-C2C, synthetic weights) | 4 |
| system runs all-in-VRAM | 1 |
| no named GPU | 1 |

- **tier C: host not identified:** `mixtral-offloading-mixtral8x7b-rtx3080m-2bit-b1`, `mixtral-offloading-mixtral8x7b-rtx3060-2bit-b1`, `mixtral-offloading-mixtral8x7b-t4colab-2bit-b1`, `mixtral-offloading-mixtral8x7b-rtx3080m-3bit-b1`, `mixtral-offloading-mixtral8x7b-rtx3060-3bit-b1`, `mixtral-offloading-mixtral8x7b-t4azure-3bit-b1`, `adapmoe-mixtral8x7b-hqq4plus2-rtx4090-cache64`, `adapmoe-mixtral8x7b-hqq4plus2-rtx4090-cache160`, `adapmoe-mixtral8x7b-hqq4-rtx4090-cache128`, `adapmoe-mixtral8x7b-hqq4-a6000-cache64`, `adapmoe-mixtral8x7b-hqq4-a6000-cache160`, `adapmoe-mixtral8x22b-hqq4-a6000-cache64`, `hobbit-mixtral8x7b-rtx4090-gpucentric-b1`, `hobbit-mixtral8x7b-rtx4090-cpugpu-b1`, `hobbit-phimoe-rtx4090-gpucentric-b1`, `hobbit-phimoe-rtx4090-cpugpu-b1`, `floe-mixtral8x7b-rtx3090-vram12g-b1`, `floe-mixtral8x7b-rtx3090-vram18g-b1`, `floe-mixtral8x7b-rtx3090-vram24g-b1`, `moe-infinity-deepseekv2lite-a5000-b1`, `moe-infinity-mixtral8x7b-a5000-b1`, `moe-infinity-arctic-a5000-b1`, `acceptmoe-qwen3-30b-a3b-instruct-rtx5090-48slots-math500`, `acceptmoe-qwen3-30b-a3b-instruct-rtx5090-48slots-gsm8k`, `acceptmoe-qwen3-coder-30b-a3b-rtx5090-48slots-math500`, `acceptmoe-gpt-oss-120b-rtx5090-48slots-math500`, `specexperts-qwen3-30b-a3b-a6000-b1-ctx1k`, `specexperts-qwen3-30b-a3b-a6000-b1-ctx64k`, `lcpp-d24528-batot1-qwen3.6-35b-q8kxl-gtx1080ti`, `lcpp-pr26824-qwen3.6-35b-q4km-rtx3070laptop`, `lcpp-pr26824-qwen3.5-122b-reap30-iq2m-rtx3070laptop`, `lcpp-pr26563-buha-qwen3.6-35b-q4km-rtx2000laptop`
- **not batch-1 decode:** `expertflow-mixtral8x7b-a40-cs1-bs4`, `expertflow-mixtral8x7b-a40-cs3-bs4`, `moe-lightning-mixtral8x7b-t4-s1-synthreasoning`, `moe-lightning-mixtral8x7b-l4-s2-synthreasoning`, `moe-lightning-mixtral8x7b-t4-s1-summarization`, `moe-lightning-mixtral8x7b-l4-s2-summarization`, `moe-lightning-mixtral8x7b-t4-s1-mtbench-gen128`, `specmoeoff-mixtral8x7b-a30-apps-largebatch`, `specmoeoff-mixtral8x7b-rtx4090d-apps-largebatch`, `klotski-mixtral8x7b-rtx3090-multibatch`, `klotski-mixtral8x22b-rtx3090-multibatch`, `klotski-mixtral8x22b-h800-multibatch`, `dali-deepseekv2lite-rtx3090-b16`, `dali-mixtral8x7b-rtx3090-b16`, `dali-qwen3-30b-a3b-rtx3090-b16`
- **SSD or unified-memory tier:** `hobbit-mixtral8x7b-jetsonorin-b1`, `hobbit-phimoe-jetsonorin-b1`, `flashmoe-qwen3-30b-a3b-rtx5070ti-24of128-ssd`, `flashmoe-qwen3-30b-a3b-rtx5070ti-12of128-ssd`, `flashmoe-olmoe-1b-7b-rtx5070ti-24of64-ssd`, `cacheable-qwen3-235b-a22b-q4km-rtx3070-ssd-b1`, `ssdllama-deepseek-v4-flash-rtx5090-16gb-mmlu`, `ssdllama-deepseek-v4-flash-rtx5090-16gb-alpaca`, `ssdllama-kimi-k2.7-code-rtx5090-16gb-mmlu`, `ssdllama-glm-5.2-rtx5090-16gb-alpaca`, `ssdllama-kimi-k3-rtx5090-16gb-mmlu`, `ssdllama-kimi-k2.7-code-rtx5090-32gb-ram`, `lcpp-d24528-nibor1896-dsv4flash-iq3xxs-rtx5090-ssd`, `edge0-qwen3.6-35b-a3b-int4-k4-macmini-m4pro-24gb`, `edge0-qwen3.6-35b-a3b-int4-k8-macbook-m2-16gb-ssd`
- **tier C: quantization not stated:** `daop-mixtral8x7b-a6000-ecr46.9-b1`, `daop-phi3.5moe-a6000-ecr46.9-b1`, `daop-mixtral8x7b-a6000-ecr25-b1`, `daop-phi3.5moe-a6000-ecr25-b1`, `moegen-dsv2lite-a5000-b1`, `moegen-mixtral8x7b-a5000-b1`, `buddymoe-dsv2lite-a100-cache75`, `buddymoe-dsv2lite-a100-cache50`, `buddymoe-dsv2lite-a100-cache37.5`, `fmoe-mixtral8x7b-a100-b1`, `fmoe-qwen1.5moe-a100-b1`, `fmoe-phi3.5moe-a100-b1`
- **multi-GPU or no offloading:** `ecospec-qwen3-235b-a22b-8xh200-b1`, `ecospec-gpt-oss-120b-8xh200-b1`, `ecospec-deepseek-v3.1-8xh200-b1`, `lcpp-d24528-rfc-glm-5.1-iq2m-4x3090`, `lcpp-pr27861-desc-qwen3.8-flash-next-q4kxl-2x3090`
- **no usable single-request decode number:** `swapmoe-switcht16-jetson-budget2gib-b1`, `pregated-moe-switchlarge128-a100-b1`, `pregated-moe-switchbase128-a100-b1`, `lcpp-pr26563-siganos-dsv4flash-iq3xxs-rtx5090`
- **not a real single PCIe GPU + host (simulator, NVLink-C2C, synthetic weights):** `st-moe-simulated-accelerator-vs-a100`, `specexperts-qwen3-235b-a22b-gh200-b1-ctx1k`, `specexperts-glm-4.7-flash-a6000-b1-ctx1k`, `specexperts-gpt-oss-120b-a100-b1-ctx1k`
- **system runs all-in-VRAM:** `pipeshard-qwen3-30b-a3b-q4_0-cli3-rtx5090-32G-cmoe`
- **no named GPU:** `lcpp-pr26563-qwen3.6-35b-q5kp-8gbvram`

## Model shapes (`mosl.archs.shape`)

Each distinct repo was built with `mosl.archs.shape` (Hub access worked), pointed at a scratch copy of `data/shapes.json` so the repo's cache was not modified. A `static_offload_time` smoke test (default Params, half the MoE layers on CPU, the first row's inputs) then ran on every shape that built.

| repo | rows | result | layers (MoE) | experts, top-k | expert params | dense/token |
|---|---|---|---|---|---|---|
| Qwen/Qwen1.5-MoE-A2.7B | 5 | ok | 24 (24) | 60, 4 | 8.65M | 1.236B |
| Qwen/Qwen2-57B-A14B | 5 | ok | 28 (28) | 64, 8 | 27.53M | 6.995B |
| Qwen/Qwen3-235B-A22B | 2 | ok | 94 (94) | 128, 8 | 18.87M | 6.753B |
| Qwen/Qwen3-30B-A3B | 3 | ok | 48 (48) | 128, 8 | 4.72M | 0.919B |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | 3 | ok | 48 (48) | 128, 8 | 4.72M | 0.919B |
| Qwen/Qwen3.6-35B-A3B | 6 | ok | 40 (40) | 256, 8 | 3.15M | 1.431B |
| Qwen/Qwen3.8-Flash-Next | 2 | ok | 48 (48) | 512, 10 | 4.92M | 54.877B |
| deepseek-ai/DeepSeek-V2-Lite | 7 | ok | 27 (26) | 64, 6 | 8.65M | 0.892B |
| deepseek-ai/DeepSeek-V2.5-1210 | 2 | ok | 60 (59) | 160, 6 | 23.59M | 11.975B |
| deepseek-ai/DeepSeek-V3 | 2 | ok | 61 (58) | 256, 8 | 44.04M | 15.264B |
| deepseek-ai/DeepSeek-V4-Flash | 2 | ok | 43 (43) | 256, 6 | 25.17M | 6.248B |
| deepseek-ai/deepseek-moe-16b-base | 3 | **FAILED**: AssertionError: ['DeepseekForCausalLM'] | | | | |
| microsoft/Phi-3.5-MoE-instruct | 4 | ok | 32 (32) | 16, 2 | 78.64M | 1.345B |
| mistralai/Mixtral-8x7B-v0.1 | 9 | ok | 32 (32) | 8, 2 | 176.16M | 1.343B |
| openai/gpt-oss-120b | 1 | ok | 36 (36) | 128, 4 | 24.89M | 0.969B |
| poolside/Laguna-S-2.1 | 1 | ok | 48 (47) | 256, 10 | 9.44M | 3.397B |
| zai-org/GLM-5.2 | 1 | ok | 78 (75) | 256, 8 | 37.75M | 16.698B |

Caveats:
- `deepseek-ai/deepseek-moe-16b-base` (3 rows: ProMoE DS-1, Fate x2) fails. It uses remote code (`DeepseekForCausalLM`, model_type `deepseek`), and `archs.shape` only falls back for DeepSeek-V3-style configs. The config is public (28 layers, first dense, 64 routed + 2 shared experts of 1408, top-6, MHA), so a small hand-written Shape or a fallback branch would fix it.
- `Qwen/Qwen3.8-Flash-Next`: `archs` counts the 51.2B-parameter per-layer n-gram embedding table (`layers.*.ple.ple_embedding`, a lookup) as dense per-token parameters, so `dense_params_per_token` is 54.9B instead of about 3.7B. Fix the shape before using the 2 rows.
- Hybrid-attention models are not modelled faithfully in `kv_bytes_per_token_ctx`. For Qwen3.6-35B-A3B and Qwen3.8-Flash-Next, only 1 layer in 4 has a KV cache (the rest are linear attention), but `_kv_bytes` charges all layers. DeepSeek-V4-Flash (compressed attention) and GLM-5.2 (sparse DSA index) are also approximate. At the contexts used here KV reads are small next to weight reads.
- Architecture stand-ins with identical configs: DeepSeek-V3-0324 -> `deepseek-ai/DeepSeek-V3`; DeepSeek-V2-Lite-Chat -> `DeepSeek-V2-Lite`; Mixtral Instruct -> `Mixtral-8x7B-v0.1`; Qwen2-57B-A14B-Instruct -> base; Qwen3-235B-A22B-Instruct-2507 -> `Qwen3-235B-A22B`; Qwen3-30B-A3B-FP8 -> `Qwen3-30B-A3B`.

## Least certain assumptions

1. The host rule (above). It decides 32 exclusions. The looser group-1 reading would admit 26 more rows, and up to 30.
2. Quantized formats named only as 'INT4'/'INT8' (ProMoE Qwen2/Mixtral, KTransformers' quantized panel, whose dense dtype is also unstated) are set to 4.5 / 8.5 bpw. The true values span roughly 4.1-4.9 and 8.0-8.5 bpw. HybriMoE's dense layers are assumed Marlin int4 (4.125), but they could be 16-bit.
3. Imputed budgets are upper bounds (the whole card) for ProMoE, Fate, SP-MoE (Mixtral/Phi), MoE-SpeQ, FreeToken (4 rows) and xashr. SP-MoE also keeps a 16-bit draft model in GPU memory, so its real expert budget is well below the card.
4. The custom cloud Xeon SKUs 6459C and 8559C are mapped to Sapphire Rapids and Emerald Rapids memory configurations. This mapping is not verified.
5. Fiddler's end-to-end metric stands in for decode speed, and its hosts are read as dual-socket from '48 core' / '112 core'.
6. `spmoe-deepseekv2lite-a100-gpumem39g-b1` is kept although the model nearly fits in 39 GB. It may be an all-in-VRAM point in practice.
7. All community rows came from WebFetch (model-extracted) and should be re-verified against the pages. sdroege is the one C->B upgrade.
