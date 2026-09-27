# Draft: upstream pull request for `runtime/ggml-sched-overlap.patch`

*Prepared, not opened. Opening it is your call; read the "Honest scope" section first.*

## Title

ggml-backend: optionally run an independent CPU split concurrently with the preceding GPU split

## Summary

`ggml_backend_sched_compute_splits` runs splits strictly in order: the inputs of split *i+1* are copied only after
split *i* has been enqueued, and a CPU split then waits for the device. When a CPU split does not consume anything the
preceding GPU split produces, the two can run at the same time. This patch adds an opt-in scheduler flag,
`ggml_backend_sched_set_overlap_cpu(sched, true)`, that detects this case (no input of the CPU split is a node, or the
view source of a node, computed in the GPU split; view ops are skipped because they compute nothing), copies the CPU
split's inputs first, enqueues the GPU split, and runs the CPU split while the GPU works. It also adds a tensor flag,
`GGML_TENSOR_FLAG_SPLIT_BEFORE`, that lets a graph builder force a split boundary, so that independent work can be
placed in its own split. The scheduler logs how many CPU splits ran concurrently when it is freed.

Default behaviour is unchanged (the flag is off).

## Honest scope

- **Stock `--n-cpu-moe` does not benefit.** Its CPU splits consume the router output and the normalized hidden state
  of the preceding GPU split, so they are never independent. The overlap matters for graphs that split a layer's
  experts between devices (GPU-resident experts and CPU-resident experts of the same layer), as our expert cache and
  several community forks do.
- **Measured effect** (A10, gpt-oss-20b and Qwen3-30B-A3B, our expert cache at 12.5–50 % of experts): +8 to +13 %
  decode speed with the overlap on (job 015 on branch `gpu`, `prereg/PROTOCOL.md` phase-2 log). The first version of
  the independence test never fired (views of the layer input placed in the GPU split made every CPU split look
  dependent); skipping view ops fixed it: 24,576 of 24,627 eligible CPU splits then ran concurrently.
- **Testing to do before opening.** Rebase onto current master; run `test-backend-ops`; run `llama-bench` with and
  without the flag on a model where no split is independent (expect identical speed and output) and on a split-expert
  graph (expect the gain above); check CUDA graphs are unaffected.

## Files

`ggml/include/ggml-backend.h`, `ggml/include/ggml.h`, `ggml/src/ggml-backend.cpp` (151 lines of diff, against
llama.cpp 2145525a).
