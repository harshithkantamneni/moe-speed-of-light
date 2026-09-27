# Group 1 audit notes (2023–2024 MoE offloading systems)

Rows: `rows_group1.jsonl` (38 rows). All sources were fetched from arXiv HTML (latest version) with curl; figure values were pixel-measured from the published PNGs or 400–600 dpi PDF renders against gridlines/ticks, and each row states its method. Extra fields `routing_exact` and `lossy` sit at top level (true/false/null) and are explained in `notes`.

Tier rule, applied literally: **A** = model+dtype, GPU, host CPU, DRAM (type/channels/BW) and GPU expert budget all stated. **B** = model+dtype and GPU stated, with at most two of {DRAM config, PCIe gen, budget} missing. CPU identity is not part of the B test. **C** = anything else. If you want CPU required for B, the Mixtral-offloading and AdapMoE rows drop to C.

No paper states `equal_vram` for any baseline.

## 1. Mixtral offloading (Eliseev & Mazur, 2312.17238 v1): 6 rows, tier B
- Table 2 gives exact batch-1 tok/s for 4 GPUs × {2-bit, 3-bit HQQ experts}. I kept the 6 consumer/cloud cells and dropped the A100. The budget is stated as k cached experts per layer. CPU and DRAM are never stated, and neither are prompt/output lengths.
- The only baseline is HF accelerate "naive offloading" using the same quantized weights. Its version is not given.
- **Suspicious:** the accelerate value for 3-bit on the RTX 3060 is 1.791 tok/s. That is higher than the paper's own "w/o LRU & pre-loading" ablation (1.346) and than the 2-bit accelerate cell (0.919). It is probably a typo, but the same number appears in both the HTML and the PDF.

## 2. Fiddler (2402.07033 v3, ICLR'25): 2 rows, tier B
- Every number is figure-only. The metric is **end-to-end** tok/s (prefill + decode), so I used the [32,256] config as the most decode-dominated one and put the 15-config means in the notes. The decode-only ITL figure (Fig 12) is too coarse to read reliably; rough readings are in the row notes.
- llama.cpp baseline: **b2956, -ngl 8 / 16**. This is whole-layer offload, so attention for the offloaded layers also runs on the CPU. Threads, mmap and llama.cpp's weight dtype are not stated. DRAM is not stated.
- Mixtral-offloading was forced to 16-bit with offload_per_layer 7 and 5, which is off its design point.
- **Version drift:** v1 had no llama.cpp baseline and claimed 8.2×/10.1× over Eliseev & Mazur. v3 adds llama.cpp and the headline falls to **1.26×**. In three Env2 configs, llama.cpp ≥ Fiddler.

## 3. MoE-Infinity (2401.14361 v3): 3 rows, tier C
- The weight dtype is never stated, and neither are the CPU or the GPU cache budget. DRAM is capacity only (32 GB for DeepSeek, 128 GB for Mixtral). PCIe is given as 24 GB/s in Table 1 and 32 GB/s in Sec 5.
- DeepSeek-V2-Lite TPOTs are exact (Table 1). For Mixtral, the system TPOT (836 ms) comes from the text and the baselines are figure-only (Fig 7, log2 axis).
- **Suspicious baseline:** "Llama.cpp (Ollama)" is given with no version, quantization, num_gpu, threads or mmap setting. It measures 2.59 s/token on DeepSeek-V2-Lite and about 5.2 s/token on Mixtral, which is implausibly slow. The authors describe llama.cpp as storing "all model parameters in CPUs", which suggests a default Ollama run. With a 31 GB model in 32 GB of host RAM, mmap thrash is plausible.
- **Suspicious data:** in Fig 7, the Arctic bars are pixel-identical to the NLLB bars. That row is included but flagged.

## 4. Pre-gated MoE (2308.12066 v3, ISCA'24): 2 rows, tier A (with caveat)
- The hardware is fully stated: EPYC 7V12, 1.8 TB DDR4, A100-80GB, PCIe gen4. The budget is 0 resident experts.
- The dtype is only implied as fp32 from Table I sizes. The model is the encoder-decoder Switch Transformer, and "tokens processed per second" may count input tokens.
- Routing is **not exact**: the pre-gate uses the previous block and the model is fine-tuned.
- The baselines are the authors' FasterTransformer re-implementations (MoE-OnDemand ≈ HF accelerate, MoE-Prefetch ≈ SE-MoE). There is no llama.cpp baseline.
- The Switch-Large number (42 tok/s) is exact from the text. The Switch-Base-128 numbers are figure-only.

## 5. AdapMoE (2408.10284 v1, ICCAD'24): 6 rows, tier B
- The metric is per-token latency, converted to tok/s. Table 2 is exact for 8x7B 4-bit, RTX 4090, 128 cached experts (0.288 vs 0.392 s). The other cells are figure-only (Fig 8). The budget is stated as the total number of cached experts.
- CPU, DRAM, PCIe and batch are not stated. Batch is set to null; single-request use is implied.
- The headline system uses **adaptive gating**, which skips the second expert for about 24% of tokens, so routing is not exact. The exact-routing "w/o gating" value is in the notes.
- The baselines are re-hosted. Mixtral-offloading was modified and is "2x faster than its open-source version". Pre-gated was re-implemented without fine-tuning, and in one config it beats AdapMoE. There is no llama.cpp baseline.

## 6. HOBBIT (2411.01433 v2): 6 rows, tier B (RTX 4090) / C (Jetson)
- All values are figure-only: the "Average" bars of Fig 14 and Fig 15 decode tok/s, with batch 1 stated. The RTX 4090 host has 256 GB and 64 cores (no CPU model), PCIe 4.0. The expert-cache size is never stated.
- llama.cpp baseline: version, -ngl and threads are not stated. It is described as stock layer offload.
- The **Jetson** llama.cpp baseline is pathological: the authors themselves attribute it to **mmap page faults**, and it produces the 13×/18.9× decode claims. Those rows are an SSD tier on unified memory and fall outside a PCIe bound.
- **Asymmetry:** HOBBIT loads int4/int2 copies of cache-miss experts and skips about 3% of experts, while the baselines run fp16/int8. Against llama.cpp on the RTX 4090 with the CPU helping, the gain is only 1.31× (Mixtral) and 1.42× (Phi). Against Fiddler it is 0.99× on Mixtral.

## 7. ProMoE (2410.22134 v3): 5 rows, tier B
- Hardware: i9-14900K, 128 GB (v1 said 64 GB), RTX 4090, PCIe 4.0 (32 GB/s). Sec 3, however, says the measured 23.9 GB/s matches "PCIe 4.0x8". The **default cache rate for the main figures is never stated**.
- Values are figure-only decode TPS from the llama.cpp-codebase figure (Fig 11b). The llama.cpp "LO" baseline is stock layer offload with version, ngl, quant type and threads unstated.
- **Key finding:** on Mixtral INT4, stock llama.cpp LO (about 37 TPS) beats ProMoE (about 27 TPS), and the authors say so. The "LRU" and "Static" baselines are the authors' own caches inside llama.cpp. The speedup over LRU is only 1.09× on average.
- The transformers-codebase numbers (Fig 10) were not extracted.

## 8. ExpertFlow (2410.17954 v2, DAC'26): 2 rows, tier C, batched
- No batch-1 numbers exist in v2 or v1, so I recorded Mixtral at BS=4 (the smallest batch shown) from Fig 6b; these are figure-only. Dtype, DRAM and PCIe are unstated, and the unit of cache size (CS) is undefined.
- The only baseline is "Cache-MoE", described as Mixtral-offloading-style LRU that falls back to the CPU on misses. There is no llama.cpp baseline.
- v1 had a different title and reported only normalized speedups.

## 9. MoE-Lightning (2411.11217 v1): 5 rows, tier C, **batched throughput**
- Exact tables (Tables 4 and 5), with batch N from about 76 to 1500. Throughput includes prefill, and both weights and KV are offloaded.
- The weight dtype is never stated. The L4 instance's host bandwidth (100 GB/s) and CPU→GPU bandwidth (32 GB/s) are given in Fig 3.
- The baselines are FlexGen, FlexGen(c) and DeepSpeed Zero-Inference v0.14.3. There is no llama.cpp or accelerate baseline. None of these rows is comparable to batch-1 decode.

## 10. SwapMoE (2308.15030 v4, ACL'24): 1 row, tier C
- This is not a speed-oriented offloading system. It restricts routing to in-memory "Virtual Experts" (lossy, routing not exact), runs on Jetson devices with a Switch Transformer, and reports latency per request, so tok/s is null.
- The text and Table 1 contradict each other (the text puts 0.82 s at the 2.0 GiB budget; the table gives 0.45 s).

## Other candidates checked
- EdgeMoE (2308.14352) is cited by four of these papers, but it runs on Jetson TX2 / Raspberry Pi with storage offload and has no llama.cpp or accelerate baseline. Excluded.
- PowerInfer is cited by five, but it targets dense ReLU models. Excluded.
- I found no other 2024 single-GPU MoE-offloading paper with a batch-1 decode comparison against llama.cpp or accelerate beyond Fiddler, MoE-Infinity, AdapMoE, HOBBIT and ProMoE.
