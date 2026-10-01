# Job 090: output parity against an all-VRAM reference — predictions and outcome

The predictions are in the header of `jobs/090_parity_a100@vast.sh` (gpu branch commit c70e311, pushed before launch).
Numbers: `prereg/parity_090.json` (the job's `parity_kl.json`). The job was written for an A100 80 GB; the first rental
(offer 31632904) turned out to be a 40 GB card and the VRAM gate stopped it in under a minute; it ran on an RTX PRO 6000
(96 GB, offer 50107112, Xeon 6960P, the paper's all-in-VRAM card), so the reference is the same GPU family as Table 1.

**Setup.** Teacher-forced decode, 12 sequences x 128 steps = 1,536 steps per model, each model's own text (job 082's
parity text). Reference: stock llama.cpp (4da6337) with every weight in VRAM, logits dumped over the full vocabulary
(201,088 for gpt-oss, by a bench-tool-only patch adding `--dump-full`). Systems: ours with the cache at 25% (C32, the
law's table computed on the machine), and stock llama.cpp with the `--n-cpu-moe` placement job 082 used (27 / 36). KL(reference ||
system) per step; a control (the patched binary with the cache off, all in VRAM) gives KL 0 at every step.

**Result.**

| Model | System | KL mean (nats) | KL p99.9 | KL max | Top-1 agreement | ΔNLL |
|---|---|---|---|---|---|---|
| gpt-oss-120b | ours, C32 | 0.0019 | 0.121 | 0.21 | 98.5% | +0.16% |
| gpt-oss-120b | stock, --n-cpu-moe 27 | 0.0017 | 0.078 | 0.22 | 98.8% | −0.05% |
| Qwen3-30B-A3B | ours, C32 | 0.0005 | 0.059 | 0.13 | 99.3% | +0.19% |
| Qwen3-30B-A3B | stock, --n-cpu-moe 36 | 0.0008 | 0.074 | 0.22 | 99.3% | +0.10% |

**Predictions.**
1. **Held.** Mean KL 0.0019 / 0.0005 ≤ 0.01; 99.9th percentile 0.121 / 0.059 ≤ 0.5.
2. **Held.** Top-1 agreement 98.5% / 99.3% ≥ 98%; stock's placement agrees with the reference within 1 point of ours
   (98.8 vs 98.5; 99.3 vs 99.3).
3. **Held.** ΔNLL +0.16% / +0.19%, within ±0.5%.

**Reading.** Our cache's divergence from an all-VRAM run is the same size as stock llama.cpp's own CPU placement: it is
the cost of running experts on the CPU (activations quantised to Q8_0 by the helpers where the GPU path uses Q8_1), not
of the cache. 3.3% (gpt-oss) and 0.9% (Qwen3) of steps have KL above 0.01.
