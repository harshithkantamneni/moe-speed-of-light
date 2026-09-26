# Adding a held-out measurement from your own machine (about 30 minutes)

The model was validated on third-party numbers only. This protocol adds a first-party point that you record end to end. It follows Hoefler and Belli (SC'15): randomized order, repeated runs, medians, full provenance.

1. **Record the platform.** Save these to `platform.txt`: `nvidia-smi -q | head -40`, `lscpu`, `sudo dmidecode -t memory | grep -E "Speed|Size|Type:"` (DIMM speed and count), `nvidia-smi -q | grep -A3 "PCI"` (link generation and width), and `git -C llama.cpp rev-parse HEAD` together with your build flags.
2. **Measure the two bandwidths the model needs.** For host DRAM, build STREAM (`gcc -O3 -fopenmp stream.c`) and record Triad. For PCIe, run `nvbandwidth -t host_to_device_memcpy_ce`, or the CUDA `bandwidthTest --memory=pinned` sample.
3. **Sweep the static split.** Use `gpt-oss-120b` (ggml-org MXFP4 GGUF) or `Qwen3-30B-A3B` (Q4_K_M):
   ```bash
   for n in $(shuf -e 36 32 28 24 20 16 12 8 4 0); do
     ./llama-bench -m MODEL.gguf -ngl 99 --n-cpu-moe $n -fa 1 -p 0 -n 128 -r 5 -o jsonl >> sweep.jsonl
   done
   ```
4. **Predict.** Add each run to `mosl/validation_set.py` as `kind="static"`, with `n_cpu=$n`, your datasheet peaks and a new `src`. Then run `python scripts/validate.py`. Your source is held out automatically, so the reported error on your rows is a true prediction.
