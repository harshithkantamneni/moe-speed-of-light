# Same-machine comparison v2 (job 073): predictions and outcome

The predictions are in the header of `jobs/073_samehost_v2@vast.sh` on the public `gpu` branch (commit a44221b,
pushed before the launch). Attempt 1 died on a CUDA 12.8 image (FreeToken needs the CUDA 13 toolkit); attempt 2
ran the unchanged protocol. Statistics: `scripts/samehost_stats.py` → `prereg/samehost_v2.json`.

**Setup.**
- **Host:** Ryzen 9 9950X3D + RTX 5090. Host memory: CPU / link / both = 69.1 / 53.1 / 80.9 GB/s.
- **Workload:** gpt-oss-120b, 30 AIME-25 problems, 256 decode tokens, greedy decoding.
- **Protocol:** one held-out warm-up prompt, then a session over the 30 problems. Three server launches per main
  configuration, with the group order rotated.
- **Client:** FreeToken's own client code and tok/s formula.

| Experts on the GPU | llama.cpp | ours+FETCH | ours | FreeToken offload | FreeToken hybrid | ours+FETCH ÷ FreeToken's better backend (95% CI) |
|---|---|---|---|---|---|---|
| 11% (C14, `-ncmoe 32`, rate 0.111) | 31.7 | 58.0 | 57.6 | 45.6 | 52.3 | **1.109 [1.098, 1.120]** |
| 25% (C32, `-ncmoe 27`, rate 0.25) | 37.3 | 88.0 | 88.2 | 80.7 | 81.1 (tuned cap 1: 82.1) | **1.084 [1.076, 1.092]** (1.071 vs tuned) |
| 40% (C51, `-ncmoe 22`, rate 0.40) | 44.5 | 117.7 | 122.3 | 125.1 | 113.6 | **0.941 [0.933, 0.951]** (ours without FETCH: 0.978 [0.965, 0.991]) |

tok/s, mean over problems. Intervals: paired bootstrap over the 30 problems, 10,000 resamples.

**Measured GPU memory (GiB):**
- llama.cpp: 9.5 / 17.4 / 25.3
- ours: 9.5 / 17.5 / 25.9
- FreeToken: 28.4 at every rate (it preallocates 90% of the card)

**Launch-to-launch spread:** CV 0.1–0.4% for every system. The variance is across problems, not launches.

1. **Failed.** ours+FETCH is ahead by ≥5% with the CI above 1 at C14 (+10.9%) and C32 (+8.4%). At C51 the
   prediction was "within ±5%"; FreeToken's offload backend is ahead by 5.9% (ours without FETCH trails by 2.2%).
2. **Held.** ours+FETCH ≥ 1.8× llama.cpp at every budget: 1.83 / 2.36 / 2.65×.
3. **Failed.** The prediction was that warming up on the measured problem reads 5–15% faster for the caches. On the
   same 5 problems it reads −0.9% (ours) and −0.3% (FreeToken) against the session, and −3.8% for llama.cpp.
   Warm-up bias is not what separated the systems in jobs 064/067.
4. **Failed.** "≥ 80% of greedy outputs identical to llama.cpp's."
   - The offloading engines' texts diverge within the first ~100 characters: 27–47% share llama.cpp's first 160
     characters.
   - llama.cpp's own placements agree with each other (87–100%).
   - The CPU and GPU kernels round differently, so each system decodes its own continuation of the same prompt.

**Other findings:**
- **FETCH gains nothing in chat on this host:** +0.7% [−0.5, +2.0], −0.3% [−1.6, +1.0], and −3.8% [−4.9, −2.5] at
  C51. On own text through ec-bench on the same host it gains +6.7% at C32. This is still open.
- **FreeToken's CPU path is weak.** Its hybrid backend with the fetch cap at 0 (all misses on the CPU) runs at
  26.1 tok/s at C32. Our CPU-only cache runs at 88.2, 3.4× faster.
- **The law, predicted on the machine before the runs:** own-text configurations within 0.6 / 1.0 / −2.5% and FETCH
  +4.2%.
