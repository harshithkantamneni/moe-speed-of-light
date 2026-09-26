# Published MoE-offload decode measurements (for retrodiction)

Companion to `published_measurements.json`: 140 data points, one object per measured configuration, collected 2026-09-25.
Every number is copied from the cited URL. The `source_ref` field quotes the table, row or sentence it came from.
Fields that a source does not give are marked `"not stated"`. Where a platform fact was not stated but is implied (for example, AM5 is dual-channel), it is labelled as such in the field or in `notes`.

- **Confidence:** 65 high, 43 medium, 32 low.
- **`spec_complete: true`:** 58 rows. In these, the source itself states the GPU (or CPU-only), CPU, RAM type and speed, model/quant, offload config and tok/s. This is the bar you set for community posts. Filter on this field for the cleanest retrodiction set.
- **Batch size** is 1 everywhere except where `notes` says otherwise. Aggregate multi-stream numbers were not recorded.

## Coverage

| Regime | Rows | Main sources |
|---|---|---|
| GPU-cache / PCIe-fetch offload | 42 | Mixtral-offloading T4/3060/3080M/A100 (32), MoE-Infinity A5000 (6), ProMoE 4090, Fate 3090 PCIe3, tinyserve PoC 8 GB |
| CPU-compute of experts (`--n-cpu-moe`, `-ot exps=CPU`, KTransformers, Fiddler-style) | 69 | KTransformers tutorial (15), OSDI'26 EPYC 9355 + 5090 (8), AesSedai EPYC 9355 (4), ubergarm, ikawrakow, carteakey, #15396 users, RFC #24528 |
| Hybrid (GPU expert cache + CPU computes misses) | 8 | llama.cpp RFC #24528 (4×3090 EPYC; 2×3090 hit-rate sweep), 2512.16473, martinalderson Vulkan cache |
| CPU-only | 20 | fairydreaming dual-EPYC with likwid bandwidth (12), AesSedai EPYC 9355 (3), ikawrakow 7950X (2), Cybernews 14900K/7700X (2), L1T |
| Small VRAM (8–12 GB) | many | RTX 3060 12G, 4070 12G, 5070 12G, RTX PRO 2000 8G, 1080 Ti 11G, T4/3080M 16G |
| Large VRAM (24–48 GB) | many | 3090/3090 Ti/4090/4090D/5090/A5000/A6000 |
| PCIe 3 / 4 / 5 | 17 / ~27 / 8 | PCIe 5 (RTX 5090) is under-covered: only OSDI'26 has it with full specs; xashr and Hardware Corner lack parts of the spec |
| Dual-channel desktop DDR4/DDR5 | ~18 | 5900X DDR4-3200, 5700X DDR4, 12600K DDR5-6000, 9950X DDR5-6400, 9600X DDR5-6000, 14900K DDR5-4800 |
| Many-channel server | ~55 | Xeon 6454S 2×8ch DDR5-4800, EPYC 9355 12ch DDR5-6000, 2×EPYC 9355 24ch DDR5-6400, EPYC 9334QS 12ch, EPYC 9654/9175F/9374F, EPYC 7R13 8ch DDR4 |

**Models covered:** Mixtral-8x7B (35 rows), gpt-oss-120b (17), DeepSeek-V3/R1 (31, in Q4_K_M, Q4_K_S, Q8_0, FP8, UD-Q2_K_XL), Qwen3-235B-A22B (10), Mixtral-8x22B (6), gpt-oss-20b (6), Qwen3-30B-A3B (8), DeepSeek-V2-Lite (5), Kimi-K2, GLM-4.5-Air (low confidence only), and several 2026 models (DeepSeek-V4-Flash, GLM-5.1, Qwen3.5-397B, Laguna S 2.1, Qwen3.6-35B-A3B, Gemma 4 26B-A4B).

**Gaps:**
- **Llama-4-Scout:** no fully specified offload measurement found.
- **GLM-4.5-Air:** only vague KTransformers documentation numbers.

## Sources, in priority order

1. **Mixtral-offloading, Eliseev & Mazur (arXiv 2312.17238v1), Table 2.** Tokens/s for 4 GPUs × 2 expert bit-widths × 4 algorithm variants: full, no pre-loading, no LRU/pre-loading, naive. The "no LRU & pre-loading" row is a clean pure-PCIe-fetch baseline.
   - Model sizes: 17.54 GB (2-bit experts), 21.37 GB (3-bit experts).
   - k=2 for the RTX 3060, k=4 for the other GPUs.
   - CPU and RAM are not stated.
   - LRU hit rate appears only as a plot (Fig. 2), so no numbers could be extracted.
   - The naive 3-bit RTX 3060 value (1.791) looks like a typo and is marked low confidence.
2. **Fiddler (arXiv 2402.07033).** Only ">3 tok/s" (from the v1 abstract and README) survives in text. It is end-to-end, including prefill. Per-configuration values are bar charts only.
3. **KTransformers, SOSP'25 (PDF).** Decode results are figures only. We recorded:
   - the motivating Fiddler-style number: 4.68 tok/s decode / 70.02 prefill on DeepSeek-V3 with A100 + 2× Xeon;
   - Intel MLC bandwidth: 220 GB/s intra-socket, 125 GB/s cross-socket.
4. **KTransformers DeepSeek R1/V3 tutorial (GitHub).** Two tables with full specs, the best-specified CPU-compute source:
   - V0.2: single vs dual socket, llama.cpp baseline.
   - V0.2.1: decode vs prompt length 2–7678 tokens.
   - Hardware: Xeon Gold 6454S, 8×DDR5-4800 per socket, 4090/4090D.
5. **Tsinghua OSDI'26 (arXiv 2606.10493).** 2×EPYC 9355, 24ch DDR5-6400, RTX 5090 on PCIe 5.
   - DeepSeek-R1 FP8: 21.5 tok/s. Kimi-K2: 22.4 tok/s.
   - DeepSeek-R1 Q4_K_M: authors' engine 28, KTransformers 22, ik_llama.cpp 14 (at 1K–8K context); at 128K, 19 vs about 15.
   - It also reports GEMV bandwidths (947 GB/s achieved).
   - The paper does not state whether decode used 1 or 2 GPUs.
6. **arXiv 2512.16473.** TR 7960X + 4090 on PCIe 4.
   - Mixtral up to 4.8 tok/s; Phi-3.5-MoE up to 10.4 tok/s.
   - Per-layer times: expert transfer 28 ms vs CPU compute 7.3 ms at 24 threads.
7. **MoE-Infinity (2401.14361).**
   - DeepSeek-V2-Lite TPOT on A5000, PCIe 4 at 24 GB/s, for 5 systems.
   - Mixtral 836 ms.
8. **ProMoE, Fate, HybriMoE, LayerScope, HOBBIT.**
   - ProMoE: one TPOT point and a 23.9 GB/s H2D bandwidth test.
   - Fate: "up to 14 tok/s" on 3090 over PCIe 3.
   - HybriMoE: LRU hit rates.
   - LayerScope: per-expert PCIe transfer times.
   - HOBBIT: bar charts only.
   - MoE-Lightning (2411.11217) was skipped because it is batched-throughput, not batch-1.
9. **llama.cpp GitHub.**
   - Issue #20757: tinyserve PoC on gpt-oss-120b / RTX PRO 2000 8 GB, 12–14 tok/s at about 98–100% hit rate. This is not llama.cpp, and CPU/RAM are not stated. It also contains an RX 9070 XT Vulkan cache result and an LFU hit-rate simulation.
   - PR #15077 (`--n-cpu-moe`): only one anecdotal comment.
   - PR #15952: `--n-cpu-moe` sweep on an RTX 5070; CPU/RAM not stated.
   - Discussion #15396 (gpt-oss guide): 7900 XT + DDR4-3200, 3060 + 5700X, 5070 + DDR5-6000.
   - Discussion #11733 (fairydreaming): CPU-only on 3 EPYC platforms, with likwid bandwidth.
   - RFC discussion #24528 (2026 MoE expert cache): 4×3090 EPYC 7R13 8ch DDR4; hit-rate vs decode sweep with STREAM bandwidth; 5090 + DDR5-6000.
10. **ik_llama.cpp discussions.**
    - #357 (Qwen3): ikawrakow 7950X/4080 CPU-only and hybrid; ubergarm 9950X DDR5-6400 + 3090 Ti; AesSedai EPYC 9355 12ch DDR5-6000 with likwid ~500 GB/s.
    - #477: EPYC 9334QS + 3070 on R1-0528 Q4_K_M.
11. **Blogs and journalism.**
    - carteakey: 12600K + DDR5-6000 + 4070, gpt-oss-120b, 28 tok/s. Also a confounded DDR5-2000 point.
    - Cybernews: CPU-only gpt-oss-120b with thread scaling.
    - Hardware Corner: EPYC 7343 + 64 GB DDR4 + 3090. The results look anomalously slow, likely from paging.
    - Level1Techs.
    - GeekNews summary of the Reddit 8 GB post.

Reddit was not reachable: it is blocked for both fetch tools. Some well-known posts, such as the KTransformers Qwen3 AMX post, could not be verified and are excluded.

## Auxiliary bandwidth and hit-rate measurements

These are reported alongside results. They are also in the rows' `measured_mem_bw_gbs` / `measured_pcie_bw_gbs` fields.

| Kind | Value | Source |
|---|---|---|
| H2D PCIe | 23.9 GB/s bandwidth test; 23 GB/s achieved expert fetch (RTX 4090; paper says "PCIe 4.0x8") | ProMoE 2410.22134 §2 |
| H2D PCIe | "PCIe4.0 (24GB/s)" (A5000) | MoE-Infinity Table 1 |
| Per-expert transfer | Mixtral expert: 13.98 ms at 76% util (PCIe4 x16); 26.7 ms at 80% (PCIe3 x16) | LayerScope 2509.23638 Table 2 |
| Per-layer transfer | Mixtral MoE layer: 28.02 ms GPU fetch vs 7.34 ms CPU compute at 24 threads (4090, PCIe4) | 2512.16473 Table III |
| PCIe estimate | ~30 GB/s effective PCIe 4.0 x16 for streaming RAM experts during prefill | ikawrakow, ik_llama #477 |
| DRAM (likwid) | EPYC 9374F NPS2: 185.3 per node / 361.5 both; 2×9175F (DDR5-6400): 378.3 / 753.6; 2×9654 (DDR5-4800): 359.9 / 679.2 GB/s | llama.cpp #11733 |
| DRAM (MLC) | 2×Xeon 8452Y: 220 GB/s intra-socket, 125 GB/s cross-socket | KTransformers SOSP'25 |
| DRAM (likwid) | EPYC 9355 12ch DDR5-6000: ~500 GB/s in VM (theoretical ~576) | ik_llama #357 |
| DRAM (STREAM) | 8ch DDR4 EPYC-class: ~100 GB/s (4ch: ~57). With the expert cache on, 4→8 channels gave only +6% decode, and the RAM bus was ~30% busy | llama.cpp #24528 / club-3090 #840 |
| DRAM (GEMV) | 2×EPYC 9355 24ch DDR5-6400: FP8 GEMV 947 GB/s; AOCL FP32 992.7 GB/s (theoretical 1228) | OSDI'26 Table 1 |
| Hit rate (LRU) | 25% cache: Mixtral 30.2%, DS-V2-Lite 47.7%, Qwen2-57B 45.0%; 75% cache: Mixtral 80.6% | HybriMoE 2504.05897 |
| Hit rate vs speed | DS-V4-Flash, 2×3090: cache off 13.61 tok/s; ~53% hit → 15.84; ~76% → 17.58; ~80.6% → 17.66 | llama.cpp #24528 |
| Hit rate (sim) | Qwen3-30B-A3B LFU: 88/128 experts per layer resident → 98.84% hit (Q4_K_M) | llama.cpp #20757 |

## Caveats for retrodiction

- **Different speed metrics.** Some numbers are end-to-end tokens/s including prefill: Fiddler, and partly Mixtral-offloading and HOBBIT-style evaluations. Others are pure decode (llama-bench `tg`, sweep-bench `S_TG`, KTransformers "decode"). Check `source_ref` and `notes`.
- **Modified routing.** KTransformers "6 experts" rows run DeepSeek with top-6 instead of top-8. Use 6 active routed experts when modelling them.
- **Context-dependent decode.** sweep-bench rows are recorded at N_KV=0. The `notes` give a few deeper-context values: decode falls a lot with context on CPU attention paths and with `-ctk q8_0` on mainline FA.
- **VMs and CPU topology.** The AesSedai EPYC runs are inside a VM, which the poster puts at about 10% loss. Hybrid-core Intel parts (12600K, 14900K, 270K) are sensitive to P/E-core thread placement.
- **Multi-GPU rows.** The RFC #24528 4×3090 and noonghunna 2×3090 rows spread dense and expert weights across GPUs. Only the CPU-resident experts follow the simple RAM-bandwidth model.
- **2026 models.** DeepSeek-V4-Flash, GLM-5.1, Qwen3.5-397B, Laguna S 2.1 and Qwen3.6 need their configs looked up (active parameters, expert size) before they can be used.
- **Suspect rows.** The Hardware Corner 3090 system has 64 GB RAM, which is below the model size, and the Cybernews 7700X hit the SSD. Both are likely storage-bound: `confidence: low`.
- **Figure-only results.** Several papers (KTransformers SOSP Fig. 12, HOBBIT, HybriMoE, ProMoE, Fiddler final) give per-configuration results only as figures. They were not digitised, to avoid fabricated precision.
- **tinyserve PoC.** The issue #20757 "12–14 tok/s" is a Python/transformers PoC at a warmed ~98–100% hit rate. It is not llama.cpp, and CPU/RAM are unknown.
