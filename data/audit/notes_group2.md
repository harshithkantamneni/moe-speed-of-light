# Group 2 audit notes (2025 MoE-offloading papers)

Rows: `rows_group2.jsonl` (50 rows, 13 papers with rows, 2 with none). Extraction date 2026-09-27.

How figure values were read. Where a PDF figure is vector, bar heights were taken from the PDF drawing
commands (pdfplumber) and calibrated to the axis gridlines or ticks. Each calibration was checked against a
ratio or number the paper states in its text. Where the figure carries numeric bar labels, the labels were
read from the PDF text layer. Two figures were read by other means: MoE-SpeQ Fig. 13 is a raster image, so
its labels were read by eye at 2x zoom, and fMoE Fig. 13 uses pattern-filled bars, so its values come from
pixel analysis (±0.02 s). All such rows have `figure_only: true`.

## KTransformers (SOSP'25; madsys PDF; no arXiv version found) — 6 rows, tier A
- Only paper with a complete hardware description: dual Xeon 8452Y, DDR5, 220 GB/s measured intra-socket bandwidth, PCIe 4.0, and A100-40GB / RTX 4080. Settings: batch 1, prompt 32, max output 512. All routed experts sit on the CPU for the system and both baselines.
- The Fig. 12 decode bars (vector) reproduce every ratio in the text: 1.25–1.76x over llama.cpp (BF16), 1.77–1.93x (quantized), and 1.66–2.56x with Expert Deferral. Deferral changes routing timing, so `system_tok_s` is the no-deferral value and the deferral value is in `notes`.
- The llama.cpp baseline is a custom fork the authors extended to do Fiddler-style expert-level offload, matching the system's placement. But it runs FP16 while KTransformers runs BF16. The quantized panel uses "built-in quantization" with the GGUF type unstated. No version is given (reference retrieved Feb 2025), and no threads, NUMA or flags. GPU memory use is not reported.

## HybriMoE (DAC'25, arXiv 2504.05897v1) — 6 rows, tier B
- Setup: RTX A6000, Xeon 5220R limited to 10 cores, int4 Marlin experts, GPU expert-cache ratio 25/50/75%. DRAM and PCIe are not stated, and neither are batch size or decode lengths.
- The Fig. 8 TBT bars were extracted from the vector PDF. The HybriMoE/kTransformers ratios average 1.70x, which matches the text.
- The llama.cpp baseline does layer-wise static mapping (an -ngl-style split). The paper does not say how the cache ratio maps to GPU layers, which quant was used, or the version.
- Suspicious: for DeepSeek-V2-Lite, llama.cpp gets slower as the cache ratio rises (11.7 → 10.1 → 8.9 tok/s). Also, llama.cpp beats kTransformers, HybriMoE's own base, in most cells. Table III (Qwen2, 25%: KT 0.21 s, HybriMoE 0.11 s) differs from the Fig. 8 bars (0.231 / 0.104 s).

## Fate (arXiv 2502.12224v2) — 4 rows, tier B
- Fig. 12 bar labels give values for RTX 3090 (PCIe 3.0) and GTX 1080 Ti (PCIe 1.0) with Qwen1.5-MoE and DeepSeekMoE-16B. There is no llama.cpp baseline; both baselines (LoD, EAP) are the authors' own. DRAM type is not stated, and neither is the budget for Fig. 12.
- The results are lossy: experts are moved and cached as INT4, with INT2 used for some experts.
- The claims don't match the figures. The abstract says "up to 4.1x" decode over LoD, but Fig. 12 shows at most 3.7x. The low-end text claims "2.3x over LoD", but Fig. 12 shows 1.4x. The Fig. 13 curves (smooth, interpolated) disagree with Fig. 12 at 24 GB.

## fMoE / FineMoE (EuroSys'26, arXiv 2502.05370v2) — 3 rows, tier C
- The main results use a six-GPU RTX 3090 NVLink testbed with expert parallelism, so they are out of scope. The only single-GPU data is Fig. 13 (A100-80GB), which gives no CPU, DRAM, PCIe or cache budget. There is no llama.cpp baseline.
- Qwen1.5-MoE fits in 80 GB, yet decodes at only ~5 tok/s, which implies an unstated expert-cache limit.
- In Fig. 13 FineMoE is slower than MoE-Infinity on Mixtral (~0.25 vs ~0.23 s TPOT), contrary to "consistently outperforms all baselines".

## FloE (ICML'25, arXiv 2505.05950v2) — 3 rows, tier B
- Mixtral-8x7B on an RTX 3090 under 12/18/24 GB total VRAM, batch 1, input/output 64/256. Fig. 8 bars are vector. The CPU is only described as a "64-core 2.3 GHz CPU". DRAM type is not stated.
- The method is lossy: INT2 up_proj plus 80–90% contextual sparsity. Any bound must use FloE's compressed bytes. llama.cpp appears only in related work.
- The baselines are Mixtral-Offloading, Fiddler and FP16 DeepSpeed-MII at equal VRAM. The headline 48.7x is against DeepSpeed-MII (0.13 tok/s). At 24 GB, Mixtral-Offloading reaches 95% of FloE.

## SpecMoEOff (arXiv 2508.21706v2) — 2 rows, tier A/B, not batch-1
- This is a throughput-oriented system. The optimizer picks batch and draft length, and the batch is never reported, so `batch` is null. Fig. 9 decode throughput is hundreds of tok/s aggregate. Table 1 gives measured CPU bandwidth (357/197 GB/s) and CPU–GPU bandwidth (25/23 GB/s).
- The draft model is EAGLE. Acceptance appears only in Fig. 10(b). MoE-Lightning (a replication) and DeepSpeed do not use speculation; the w/o-sd variant isolates a 1.28–1.32x speculation gain. No llama.cpp baseline.

## SP-MoE (arXiv 2510.10302v2) — 6 rows, tier B
- Batch 1, 100 output tokens, 3 environments (3090/4090/A100-40GB, all PCIe 4.0 x16). The Fig. 10 bars are vector and reproduce the text percentages. Sec 5.3 states DeepSeek-V2-Lite TPOT in text at 7 GB and 39 GB.
- All baselines use the same speculative decoding. Drafts are Mistral-7B, Phi-mini-MoE and DeepSeek-Lite-AWQ, with acceptance of 97–98% at one draft token per step. The expert budget for Fig. 10 is not stated; DRAM type is not stated.
- Absolute speeds are very low. Mixtral runs at 1.2–1.5 tok/s even on an A100. DeepSeek-V2-Lite, fully resident at 39 GB, reaches only ~10 tok/s, where SP-MoE and Mixtral-Offloading are equal.
- The text misattributes the Phi 3090 gain: "31.6% vs AdapMoE" is actually the gain vs Mixtral-Offloading; the gap to AdapMoE is ~2%.

## MoE-SpeQ (arXiv 2511.14102v1) — 6 rows, tier B
- A100-40GB with Xeon Silver 4310, 256 GB and PCIe 4.0 x16. Target is FP16; the draft is an INT4 GPTQ copy of the same model sharing KV and non-expert weights, with acceptance ">90%". Fig. 13 is a raster, and its Avg labels were read by eye.
- The "low" and "high" memory settings are never given in GB. Batch size, prompt length and output length are not stated.
- Baselines do not use speculation. They are HF device_map offload and a Mixtral-Offloading re-implementation in same-memory (SM) and same-cache (SC) variants. The paper says llama.cpp comparisons are "infeasible".
- The text contradicts the figure. It says "Qwen2-MoE ... 74.1 ms/token", but that is the SM bar; MoE-SpeQ's own bar is 35.3 ms. For Phi, the text gives SM = 351.8 ms against a figure label of 295.1.

## arXiv 2512.16473 (ASP-DAC'26; unnamed CPU-GPU collaborative framework) — 2 rows, tier B
- RTX 4090 with Threadripper 7960X (24 threads) and PCIe 4.0 x16, at 16-bit (inferred from the stated model and expert sizes). The GPU expert cache is set-associative, and misses are executed on the CPU. Fig. 5 bar labels give 4.8 tok/s (Mixtral) and 10.4 tok/s (Phi-3.5). DRAM is not stated.
- The headline 4.4x is measured against Pre-gated MoE, and the Pre-gated numbers are estimated rather than run ("assuming perfect computation-communication overlap").
- Fiddler scores 0.2 tok/s on Phi-3.5, which is implausible. The system's own CPU-only mode already reaches 4.2 tok/s on Mixtral, so the GPU cache adds only 14%. No llama.cpp baseline.

## Klotski (ASPLOS'25, arXiv 2502.06888v1) — 3 rows, tier B, not decode
- This is a multi-batch throughput system: batch 4–64, 512 input tokens, 32 output tokens. Its metric is generated tokens over the whole prefill+decode time, so `system_tok_s` is null and the Table 3 values are in `raw_metric`.
- There is no batch-1 number and no llama.cpp baseline. Baseline values exist only in Fig. 10, which was not read.

## DAOP (DATE'25, arXiv 2501.10375v3) — 4 rows, tier C
- RTX A6000, i9-10980XE with 130 GB, PCIe 4.0, batch 1. Text states 4.52/8.21 tok/s (Mixtral/Phi, ECR 46.9%) and 3.23/5.03 tok/s (ECR 25%). Baselines were read from Fig. 9. The expert dtype is never stated. No llama.cpp baseline.
- Routing is not exact: an expert on the CPU is swapped for the "next-best expert already on the GPU", and CPU experts are pre-calculated from the previous block's hidden states. This shows as GSM8K accuracy falling from 58.9 to 48.1 at ECR 50%. Rows are outside an exact-routing bound.

## MoE-Gen (arXiv 2503.09716v1) — 2 rows, tier C
- A large-batch system; its Table 9 is the only batch-1 data (A5000 with EPYC 7453, C1 inferred). MoE-Gen(G) reaches 5.0 tok/s on DeepSeek-V2-Lite and 1.0 on Mixtral.
- The llama.cpp baseline is run via Ollama and scores 0.4 and 0.2 tok/s. Version, quant, flags and offload split are all unstated. These values are 1–2 orders of magnitude below normal llama.cpp performance; see BuddyMoE's llama.cpp run of the same model at 25–34 tok/s. Treat as a broken baseline.

## BuddyMoE (arXiv 2511.10054v1) — 3 rows, tier C
- Built inside llama.cpp on an A100 (PCIe) with a Xeon 8457C, using DeepSeek-V2-Lite. The baseline is llama.cpp's own path without substitution at the same cache rate: 34.2/28.6/24.8 tok/s at c = 0.75/0.5/0.375.
- The GGUF type, flags, GPU memory size, DRAM and batch size are all unstated.
- Routing is not exact: missing experts are replaced by "buddy" experts. ARC accuracy falls from 0.735 to 0.695 at c = 0.75, for only a 7–10% speedup.

## eMoE (arXiv 2503.06823v1) — 0 rows
- Multi-GPU serving (4x A100-40GB) under Poisson arrivals, measuring SLO latency against vLLM and DeepSpeed-FastGen. There is no single-GPU decode tok/s and no llama.cpp baseline. It also changes which experts are loaded (task-aware prediction), so it is out of scope.

## DuoServe-MoE (arXiv 2509.07379v2) — 0 rows
- Single-request decode on an A5000/A6000 with AWQ-4bit Mixtral and FP8 Qwen3-30B-A3B. It reports end-to-end latency, TTFT, and "total tokens/s" versus batch size, all in line plots without labels. Output lengths are not stated, so no clean decode tok/s could be extracted.
- The baselines are ODF (HF Accelerate), LFP and MoE-Infinity. No llama.cpp. Its peak GPU memory is only ~3.9 GB for Mixtral, a tiny budget.

## Cross-paper baseline findings
1. Only KTransformers matched llama.cpp's placement to the system: routed experts on the CPU for both. Even there, llama.cpp ran FP16 against BF16, the quant type was unstated, and flags, threads and NUMA were unreported. No paper in this group reports `-ngl`, `--n-cpu-moe`, `-ot` or a llama.cpp commit.
2. Two llama.cpp baselines look broken:
   - MoE-Gen (via Ollama): 0.2–0.4 tok/s at batch 1.
   - HybriMoE: llama.cpp gets slower as the GPU cache grows.
3. Most 2025 systems compare only against HF-Transformers-era offloaders: Mixtral-Offloading, MoE-Infinity, DeepSpeed, Fiddler and AdapMoE, several of them re-implemented by the authors. These baselines decode Mixtral-8x7B at 0.1–2 tok/s. The large claimed speedups (48.7x, 4.4x, 3.5x) are measured against them, or against an estimated baseline.
