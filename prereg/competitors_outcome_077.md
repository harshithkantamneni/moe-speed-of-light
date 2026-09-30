# Job 077: Pipelined Sharding and leloch's llama.cpp cache on gpt-oss-120b — predictions and outcome

The predictions are in the header of `jobs/077_competitors_gptoss@vast.sh` (gpu branch commit 23a4a38, pushed before
launch). The build and run recipes are in `research_notes/MoE offload system race plan/{pipeshard,community_cache}_recipe.md`.

**Setup.**
- **Host:** RTX 5090 + Ryzen 9 9950X3D.
- **Workload and protocol:** gpt-oss-120b, 30 AIME-25 problems, 256 tokens, greedy, session, one launch per
  configuration.
- **Equal memory:** equal GPU memory for experts, checked on measured VRAM.
  - **Pipelined Sharding:** its `-mva` buffer was calibrated by one short run per budget. The final VRAM came within
    0.2 GiB of llama.cpp's.
- **Cache active:** leloch's cache logged `resolved=on` and `granted = cap` at every budget.

| Experts on the GPU | llama.cpp | Pipelined Sharding (MLSys'26) | leloch cache | FreeToken (better backend) | ours v2 (+FETCH) |
|---|---|---|---|---|---|
| 11% | 27.8 | 36.9 | 39.4 | 45.8 (hybrid) | **56.0** |
| 25% | 33.6 | 45.2 | 48.5 | 76.0 (offload) | **92.0** |
| 40% | 40.1 | 56.8 | 51.3 | 119.5 (offload) | **133.0** |

Speeds are tok/s. Paired against the other systems (95% CI):
- **FreeToken:** +22.2% [+20.7, +23.6] / +21.0% [+19.3, +22.7] / +11.3% [+10.2, +12.3].
- **The better of the two new entrants:** 1.42× / 1.90× / 2.34×.
- **The new entrants against llama.cpp:** 1.28–1.45×.

1. **Held.** Both new entrants run end to end on gpt-oss-120b, and both beat stock llama.cpp at 25% and 40% (and at
   11%).
2. **Held.** Ours is ahead of both at every budget.

**Handicap to report.** Neither new entrant can sample on the GPU; they are built on llama.cpp b6097 and an August
2026 base. That costs them 1.0–1.4 ms per token end to end (job 074b, on a 7950X), about 4–8% of their token time, which is far smaller than
the gaps above.

**Other notes.**
- **Hit rates:** leloch's pooled LRU cache hits 48 / 70 / 77% of expert uses; our per-layer cache in the same requests
  hits 59 / 80 / 90%.
- **Pipelined Sharding's planner** chose STATIC_ATTN_PRIO. At the 25% budget it pinned 8 of 36 layers' experts.
