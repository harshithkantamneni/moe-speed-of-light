# Audit rows: schema and rules

One JSON object per line in `rows_<group>.jsonl`. One row per measured configuration: system × model × hardware ×
GPU memory budget × batch size. Report batch-1 (single-request) decode first. Record larger batches only when a
paper has no batch-1 number, and set `batch` to the actual value.

## Rules

- **Never invent a number.** A value the source does not state is `null`, with the reason in `notes`.
- **Quote the evidence.** Every number needs a short verbatim quote, or a table row copied as text, in `evidence`,
  with its location (Table 3, Fig. 7, Sec. 5.2) and the URL fetched.
- **Figure-only values.** When a value exists only in a figure, give an approximate reading, set
  `figure_only: true`, and say how it was read (bar label, axis estimate). Do not read values off a figure you
  could not see; set them to `null` instead.
- **Units.** Decode speed in tokens/s. When the paper reports time per output token (TPOT) or latency, store the
  original in `raw_metric` and put the conversion in `system_tok_s` / `tok_s`, noting the conversion.
- **Baselines.** Record every baseline for the same configuration. `config` should hold, as far as the paper says:
  - the engine and version or commit;
  - flags such as `-ngl`, `--n-cpu-moe`, `-ot`, `--no-mmap`, threads;
  - what was offloaded;
  - whether its GPU memory use equals the system's (`equal_vram`: true / false / "unstated").
- **Tier:**
  - **A:** model + quantization, GPU, host CPU, host DRAM (type or channels or bandwidth) and GPU expert budget
    are all stated.
  - **B:** at most two of {DRAM configuration, PCIe generation, budget} are missing.
  - **C:** anything less.

## Fields

```
{
 "id": "freetoken-qwen3-30b-rtx4090-b1",      # slug
 "system": "FreeToken",
 "title": "...", "venue": "arXiv / SOSP'25 / ...", "arxiv": "2608.16157", "first_version_date": "2026-08-17",
 "url": "https://arxiv.org/html/2608.16157v1",
 "code": "https://github.com/... or null",
 "model": "Qwen3-30B-A3B", "expert_dtype": "Q4_K_M / bf16 / int4 / ...", "dense_dtype": "...",
 "gpu": "RTX 4090", "gpu_mem_gb": 24, "pcie": "4.0 x16 or null",
 "cpu": "i9-13900K (24 cores) or null", "dram": "DDR5-5600, 2 channels, 128 GB or null",
 "host_bw_gbs": null,                          # only if the paper states it (measured or peak; say which in notes)
 "budget": "GPU memory given to experts, or the fraction/number of experts on GPU, as stated",
 "batch": 1, "prompt_len": 512, "output_len": 256,
 "raw_metric": "TPOT 23.1 ms",
 "system_tok_s": 43.3,
 "baselines": [{"name": "llama.cpp", "tok_s": 30.9, "config": "...", "equal_vram": "unstated",
                "evidence": "..."}],
 "claimed_speedup": "1.40x over llama.cpp (their headline)",
 "evidence": "Table 4: '...'",
 "figure_only": false,
 "tier": "A",
 "notes": "anything that affects comparability: prefetch, speculative decoding, changed routing (e.g. expert skipping
          or deferral, which puts the row outside an exact-routing bound), accuracy loss, multi-GPU, SSD tier"
}
```

## Also record, for each paper

- `routing_exact`: does the system change which experts run? (skipping, deferral, substitution, lower top-k)
- `lossy`: does it change outputs? (quantization beyond the baseline's, approximate experts)

Put both inside `notes` if unsure.
