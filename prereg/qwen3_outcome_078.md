# Job 078: Qwen3-30B-A3B BF16, four systems — predictions and outcome

The predictions are in the header of `jobs/078_qwen3_4way@vast.sh` (gpu branch commit 23a4a38, pushed before launch).
The recipe is in `research_notes/MoE offload system race plan/qwen3_recipe.md`. Statistics: `scripts/qwen3_stats.py`
(output in `prereg/qwen3_078.json`).

**Setup.**
- **Host:** RTX 5090 + Ryzen 9 9950X (16 cores, AVX-512 BF16, no AMX).
- **Host memory:** slow for this CPU.
  - CPU STREAM read 44.7 GB/s, PCIe H2D 46.7 GB/s (FreeToken's own probe).
  - The CPU path is no faster than the link: 0.95×. On job 076's 9950X it was 62.3 vs 53.0 GB/s.
- **Workload:**
  - Qwen3-30B-A3B in BF16: 48 MoE layers, 128 experts, top-8, 9.44 MB per expert.
  - Same weights everywhere: the HF snapshot, and a BF16 GGUF converted from it.
- **Protocol:**
  - 30 AIME-25 problems, 256 tokens, greedy, thinking on.
  - Held-out warm-up, then a session; one launch per configuration.
  - GPU-side sampling for the llama-server systems.
- **Equal memory:** equal GPU memory for experts, at 12.5 / 25 / 43.75% of experts.
  - llama.cpp `--n-cpu-moe` 42 / 36 / 27.
  - ours 16 / 32 / 56 slots per layer.
  - FreeToken rate 0.125 / 0.25 / 0.4375.
  - KTransformers 16 / 32 / 56 static GPU experts per layer.
- **KTransformers build:** kt-kernel 0.7.0.post4, CPU variant `avx512_bf16`. It runs exact decoding: BF16 CPU experts,
  no deferral.

| Experts on the GPU | llama.cpp | KTransformers | FreeToken (better backend) | ours v2 (better variant) |
|---|---|---|---|---|
| 12.5% | 13.4 | 13.1 | 26.5 (hybrid) | **28.4** (FETCH) |
| 25% | 15.4 | 15.3 | 43.9 (offload) | **46.8** (FETCH) |
| 43.75% | 19.9 | 20.2 | **94.1** (offload) | 89.2 (FETCH) |

Speeds are tok/s. Paired by problem against ours, with 95% bootstrap CIs:

| Experts on the GPU | ours ÷ FreeToken | ours ÷ KTransformers | ours ÷ llama.cpp |
|---|---|---|---|
| 12.5% | **1.071 [1.063, 1.080]** | 2.17 [2.14, 2.20] | 2.12 [2.09, 2.15] |
| 25% | **1.067 [1.043, 1.094]** | 3.06 [2.98, 3.16] | 3.05 [2.97, 3.15] |
| 43.75% | **0.948 [0.935, 0.962]** | 4.42 [4.31, 4.55] | 4.47 [4.35, 4.61] |

1. **Held.** Ours is ahead of FreeToken's better backend at 12.5% and 25%, with both CIs above 1. The margin is +7%,
   against +20–22% on gpt-oss-120b (job 076).
2. **Held.** Ours is ahead of KTransformers at every budget, by 2.2–4.4×.
3. **Held.** Ours is ≥ 1.5× stock llama.cpp at every budget: 2.1 / 3.1 / 4.5×.

**Not predicted: FreeToken leads at 43.75%; ours is 5.2% [3.8, 6.5] slower.** The data behind this:
- **Our cost per token** is 5.44 ms fixed plus 0.194 ms per miss (FETCH, three budgets).
  - On gpt-oss-120b (job 073) it was 5.14 ms + 0.195 ms.
  - At 43.75% we miss 29.8 experts per token (22.4 fetched, 7.4 on the CPU), so the fixed part is about half the
    token time.
  - FreeToken's token time there is 10.63 ms against our 11.21 ms.
- **The same pattern as gpt-oss.** FreeToken's lower fixed cost wins once misses are few. On gpt-oss the crossover sat
  near 19 misses per token.
  - On this host it comes earlier because our other advantage is weak: the CPU path, at 44.7 GB/s, is no faster than
    the link.
  - Job 074 (7950X, CPU no faster than link) showed the same thing on gpt-oss.
- **Both sides of our design are visible:**
  - FETCH adds +22 / +23 / +16% over running every miss on the CPU. That is the largest FETCH gain on any host so far.
  - Without FETCH, ours would lose to FreeToken at every budget (23.3 / 38.0 / 76.6 tok/s).

**Not predicted: KTransformers runs at llama.cpp's speed.**
- Both are static placements that read the same expert bytes from the same DRAM.
- **Post-hoc, not pre-registered:** the CPU expert bytes per token divided by the 44.7 GB/s STREAM read account for
  71 / 61 / 46 ms of the 75 / 65 / 50 ms tokens. The remaining 3.8–5.4 ms is the GPU part.
- KTransformers' published speeds rely on AMX Xeons and INT4/INT8 CPU weights. This host has neither, and BF16 is the
  only exact like-for-like, so this is not its best setting. We report it that way.

**Correctness.**
- Every system's text is coherent.
- Prompt token counts are identical across all 18 configurations.
- The first ~160 characters of our outputs match stock llama.cpp on 29–30 of 30 problems.
- The full 256-token outputs match on 47–67% for ours and 90% for llama.cpp at another `-ncmoe`. The first BF16 CPU
  helper runs end to end on a real model, and its numerics diverge later in the sequence, as on gpt-oss (job 073,
  prediction 4).
- FreeToken and KTransformers render the thinking prefix differently (a leading newline / `<think>`), so their hashes
  never match. Their content does.

**VRAM (GiB).**
- ours 10.5 / 17.2 / 27.4; llama.cpp 10.4 / 17.2 / 27.3; KTransformers 10.9 / 17.6 / 27.7.
- FreeToken 28.6 / 28.6 / 30.1. As on gpt-oss, it reserves memory up front at every rate.
