"""First-party llama.cpp static-offload runs on the A10 in the validation-set convention (sensitivity fits)."""

# Sensitivity (not pre-registered): first-party llama.cpp static-offload runs on the A10 (job 025, commit 2145525a,
# response windows), in the validation-set convention (datasheet peaks: A10 600 GB/s; Xeon 8358, 8 x DDR4-3200).
A10_ROWS = [dict(id=f"A10-{r}-{n}", repo=r, kind="static", n_cpu=n, b_exp=b, b_dense=bd, bw_gpu=600.0, bw_cpu=204.8,
                 engine="llama.cpp", tok_s=t, ctx=500)
            for r, b, bd, n, t in [("Qwen/Qwen3-30B-A3B", 4.85, 4.85, 42, 68.2), ("Qwen/Qwen3-30B-A3B", 4.85, 4.85, 36, 72.0),
                                   ("Qwen/Qwen3-30B-A3B", 4.85, 4.85, 24, 85.4), ("Qwen/Qwen3-30B-A3B", 8.5, 8.5, 42, 48.3),
                                   ("Qwen/Qwen3-30B-A3B", 8.5, 8.5, 36, 51.7), ("Qwen/Qwen3-30B-A3B", 8.5, 8.5, 24, 61.6),
                                   ("openai/gpt-oss-20b", 4.25, 8.5, 21, 63.8), ("openai/gpt-oss-20b", 4.25, 8.5, 18, 69.4),
                                   ("openai/gpt-oss-20b", 4.25, 8.5, 12, 80.9), ("openai/gpt-oss-120b", 4.25, 8.5, 32, 41.3),
                                   ("openai/gpt-oss-120b", 4.25, 8.5, 27, 44.8)]]
