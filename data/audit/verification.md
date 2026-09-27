# Source re-check of the adjudicable audit rows

Three independent checkers re-read every source (58 rows): 45 ok, 13 with corrections, 0 excluded by the checkers. Changes applied:

- `promoe-llamacpp-deepseekmoe16b-fp16-rtx4090-b1`: Fig. 11(b) values re-read (-0.3 tok/s); PCIe 4.0 x16
- `promoe-llamacpp-deepseekv2lite-fp16-rtx4090-b1`: Fig. 11(b) values re-read (-0.3 tok/s); PCIe 4.0 x16
- `promoe-llamacpp-qwen15moe-fp16-rtx4090-b1`: Fig. 11(b) values re-read (-0.3 tok/s); PCIe 4.0 x16
- `promoe-llamacpp-qwen2moe57b-int4-rtx4090-b1`: Fig. 11(b) values re-read (-0.3 tok/s); PCIe 4.0 x16
- `promoe-llamacpp-mixtral8x7b-int4-rtx4090-b1`: Fig. 11(b) values re-read (-0.3 tok/s); PCIe 4.0 x16
- `spmoe-deepseekv2lite-a100-gpumem39g-b1`: excluded: effectively all-in-VRAM
- `cpugpucollab-mixtral8x7b-rtx4090-24t-b1`: metric: single-request throughput, decode-only vs end-to-end unknown
- `cpugpucollab-phi3.5moe-rtx4090-24t-b1`: metric: single-request throughput, decode-only vs end-to-end unknown
- `freetoken-qwen3.6-35b-a3b-rtx5090srv-w1`: host band from the measured 77.3 GB/s (runs capped at 6-8 threads): [110.4, 154.6] GB/s peak-equivalent; ctx 4096 assumed (long chain-of-thought / agentic workloads)
- `freetoken-deepseek-v4-flash-rtx5090srv-w1`: host band from the measured 77.3 GB/s (runs capped at 6-8 threads): [110.4, 154.6] GB/s peak-equivalent; ctx 4096 assumed (long chain-of-thought / agentic workloads)
- `freetoken-qwen3.6-35b-a3b-rtx5090desktop-w2`: ctx 4096 assumed (long chain-of-thought / agentic workloads)
- `freetoken-qwen3.6-35b-a3b-rtx4090srv-w2`: host band from the measured 63.2 GB/s (runs capped at 6-8 threads): [90.3, 126.4] GB/s peak-equivalent; ctx 4096 assumed (long chain-of-thought / agentic workloads)
- `freetoken-qwen3.6-35b-a3b-nvfp4-rtx4060laptop-w2`: dense weights FP8/NVFP4 in nvidia/Qwen3.6-35B-A3B-NVFP4: b_dense 16 -> 7.8; ctx 4096 assumed (long chain-of-thought / agentic workloads)
- `freetoken-glm-5.2-nvfp4-rtxpro6000-w1`: ctx 4096 assumed (long chain-of-thought / agentic workloads)
- `pipeshard-qwen3-30b-a3b-q4_0-cli3-rtx5090-8G`: -cmoe baseline read from Fig. 3 bar (1.23x): 26.1 tok/s, not 26.75
- `pipeshard-qwen3-235b-a22b-q2k-cli3-rtx5090-2G`: unsloth Q2_K files (artifact README): 2.92 bits/weight, not 2.625
- `pipeshard-qwen3-235b-a22b-q2k-cli3-rtx5090-32G`: unsloth Q2_K files (artifact README): 2.92 bits/weight, not 2.625
- `lcpp-d24528-xashr-laguna-s-2.1-q6k-rtx5090`: upstream baseline used auto-fit (-fit on): partial expert offload; ctx 4096 assumed (long generation)
- `lcpp-pr27861-sdroege-qwen3.8-flash-next-q4kxl-r9700`: baseline --fit on fills VRAM like the cache run: equal VRAM, approximately
- `lcpp-pr27861-sissyhistorian-qwen3-30b-a3b-q4km-rx7600`: model is an unnamed Q4_K_M derivative
