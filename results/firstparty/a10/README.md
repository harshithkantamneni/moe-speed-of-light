# First-party measurements: NVIDIA A10 (Lambda gpu_1x_a10, us-west-1, 2026-09-26)

The raw per-kernel Nsight Compute CSVs are on the `gpu` branch,
under `results/007_counters/`.

**Platform** (`../../../prereg/a10/platform_001/`):
- 30 vCPUs of a Xeon Platinum 8358 (KVM)
- STREAM Triad 152 GB/s
- A10 with ECC on; clocks locked at 1695 / 6251 MHz
- PCIe Gen4 x16, with 25.3 GB/s pinned H2D
- Device read 514 GB/s
- llama.cpp 2145525a, built for sm_86

**Runs:**
- `006_sweep/`: the pre-registered static-offload sweep, 31 configurations
  plus 3 drift repeats. `sweep.jsonl` holds the raw llama-bench records.
- `007_counters/ncu_summary.json`: GPU DRAM bytes per decode token, taken
  as the difference between the n=8 and n=4 runs.
- `007_counters/nsys_*_gpu_trace.csv.gz`: kernel and memcpy timelines for
  32 decode tokens.

The scored predictions are in `../../../prereg/a10/scored.md`.
