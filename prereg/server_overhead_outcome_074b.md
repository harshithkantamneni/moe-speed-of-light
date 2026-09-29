# Job 074b: GPU-side sampling — predictions and outcome

The predictions are in the header of `jobs/074b_server_overhead@vast.sh` (gpu branch commit 65ed351, pushed before
launch).

**Setup.** Ryzen 9 7950X + RTX 5090. Host memory: CPU / link / both = 46.2 / 46.8 / 52.1 GB/s. gpt-oss-120b, 30
AIME-25 problems, greedy, session protocol.

| Configuration | tok/s | Server time between steps |
|---|---|---|
| ours v2 C51 | 98.2 | 2.01 ms |
| ours v2 C51, GPU-side sampling | 110.1 / 109.1 | 0.41–0.44 ms |
| ours v2 + FETCH C51 | 105.0 | 1.67 ms |
| ours v2 + FETCH C51, GPU-side sampling | **123.1 / 122.5** | 0.23 ms |
| FreeToken offload 0.40 | 114.3 / 114.4 | — |
| ours v2 C32, GPU-side sampling | 71.0 | 0.56 ms |
| ours v2 + FETCH C32, GPU-side sampling | **83.0** | 0.32 ms |
| FreeToken hybrid 0.25 | 68.4 | — |
| llama.cpp `-ncmoe 22`, GPU-side sampling | 31.3 | — |

**Paired against FreeToken** (30 problems, 95% CI):
- **40%:** ours v2 + FETCH + GPU sampling, **+7.4% [+6.3, +8.5]**. Without FETCH: −4.1%.
- **25%:** **+21.3% [+19.9, +22.7]**. Without FETCH: +3.7%.

1. **Held.** The between-step time falls by ≥ 1 ms: 2.01 → 0.42 ms, and 1.67 → 0.23 ms with FETCH.
2. **Held for ours; untested for llama.cpp.** Ours gains ≥ 10% at C51: +11.7% [+10.9, +12.5], and +17.0% with FETCH.
   Stock llama.cpp without GPU sampling failed to load: FreeToken had not yet released its GPU memory, so the process
   ran out of memory. Job 075 waits for the memory to be freed.
3. **Held.** With FETCH and GPU sampling, ours beats FreeToken offload at C51 in the same session.

**Reading.** In llama-server the CPU sampler chain over 201,088 logits cost ~1.6 ms per token. That is a large share
of the fixed cost that decided the 40% result. With it removed, and FETCH on (it helps on this host, where the link
matches the CPU path), the cache leads at 25% and 40%.

**Fairness.** GPU-side sampling is a stock llama-server option, and every llama-server system uses it from job 075
on. FreeToken samples inside its own engine.
