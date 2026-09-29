# Job 074: slot maps on the GPU — predictions and outcome

The predictions are in the header of `jobs/074_fix40@vast.sh` (gpu branch commit c4aee57, pushed before launch).
- **Attempt 1:** the host stopped the instance a minute after it started (9950X3D).
- **Attempt 2:** ran on a Ryzen 9 7950X + RTX 5090. Host memory: CPU / link / both = 46.2 / 47.3 / 51.8 GB/s, so the
  CPU path is no faster than the link here (job 073's host: 69 / 53 / 81).

| | tok/s (launches) | vs FreeToken, same session (95% CI) |
|---|---|---|
| ours v2 C14 | 41.5 / 41.8 / 41.0 | +2.3% [+0.8, +3.8] vs hybrid 0.111 (40.5) |
| ours v2 C32 | 64.6 / 66.0 / 65.5 | −4.3% [−5.7, −2.9] vs hybrid 0.25 (68.3) |
| ours v2 C51 | 97.3 / 94.6 / 93.6 | −16.9% [−18.3, −15.5] vs offload 0.40 (114.4, both launches) |
| ours v2 + FETCH, C51 | 104.8 | −8.4% [−9.3, −7.5] |
| maps on the CPU (old placement), C51 | 91.8 | the fix gains +3.6% [+3.1, +4.1] |
| half-life 64, C51 | 98.4 | +1.2% against the same-launch baseline (97.3) |

**Host-side step timer:**
- **Launch time:** 0.19–0.21 ms with the maps on the GPU, 0.61–0.72 ms with the old placement (ec-bench and server).
- **Server time between decode steps ("app"):** 2.0–2.8 ms per token. ec-bench spends 1.2 ms here.

1. **Held.** Launch ≤ 0.25 ms.
2. **Failed.** "C51 ≥ 127 tok/s and ahead of FreeToken": 95.1 against FreeToken's 114.4.
3. **Failed.** "≥ 59 at C14 and ≥ 91 at C32": 41.5 and 65.3. These absolute targets carried job 073's host; this
   host's memory is 33% slower.
4. **Held.** Half-life 64 is within ±2%.

**Reading.** On a host whose CPU path is no faster than its link, our per-miss advantage disappears. FreeToken's
lower fixed cost then wins from 25% of experts up. The winner depends on the host's RAM-to-link balance, which is
what the host-memory law predicts.

**The next fixed cost to cut** is the server's 2–3 ms per token between steps: the CPU sampler over 201k logits,
streaming and bookkeeping. Job 074b tests GPU-side sampling.

**Also:** ours drifted −4% across the three launches here (FreeToken did not).
