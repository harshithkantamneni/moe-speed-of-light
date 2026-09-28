# llama.cpp MoE offloading history and a fair baseline / measurement protocol (as of 2026-09-28)

Method note (applies to all llama.cpp dates below): dates are the committer dates of the squash-merge commits on `ggml-org/llama.cpp` `master`, and "first tag" is the earliest `bNNNN` release tag containing that commit (`git describe --contains`). Both come from a full local clone of the [llama.cpp repository](https://github.com/ggml-org/llama.cpp) fetched 2026-09-27 (HEAD `4da6337`, 2026-09-27). Release tags are browsable at `https://github.com/ggml-org/llama.cpp/releases/tag/bNNNN`. PR links go to the PR page. Where the commit's author date differs from its merge date, both are given. Year-boundary tags for orientation: 2024-01-01 ≈ b1742, 2024-07-01 ≈ b3268, 2025-01-01 ≈ b4404, 2025-07-01 ≈ b5787, 2026-01-01 ≈ b7599, 2026-07-01 ≈ b9852, 2026-09-27 ≈ b11206 (last tag before each date on master, same clone).

## Q1. When did llama.cpp (and Ollama) add each relevant feature?

### Takeaway
MoE models could be partially offloaded only by whole layers (`-ngl`) from Mixtral support (2023-12-13, b1629) until tensor-level placement (`-ot`/`--override-tensor`) was merged on 2025-04-02 (b5028). `--cpu-moe` arrived on 2025-07-31 (b6051). `--n-cpu-moe`/`-ncmoe`/`-cmoe` arrived on 2025-08-05 (b6089), the same week as gpt-oss/MXFP4 (b6096). Flash attention became "auto" by default on 2025-08-30 (b6325), and automatic memory fitting (`--fit`, on by default, `--fit-target` 1024 MiB) on 2025-12-15 (b7410). Before v0.30.0 (2026-06-01), Ollama used its own whole-layer GPU estimator and exposed no expert placement. Since then it runs upstream `llama-server` and leaves placement to llama.cpp's `--fit`.

### Cited Findings

**Timeline: llama.cpp features relevant to MoE offload baselines**

| Merge date | PR | First tag | Feature / change |
|---|---|---|---|
| 2023-12-13 | [#4406](https://github.com/ggml-org/llama.cpp/pull/4406) | b1629 | "llama : add Mixtral support". The commit adds CUDA (`ggml-cuda.cu`, +293 lines) and Metal `mul_mat_id` code, so `-ngl` whole-layer GPU offload works for MoE from the start. |
| 2023-12-15 | [#4480](https://github.com/ggml-org/llama.cpp/pull/4480) | b1642 | "ggml : group mul_mat_id rows by matrix (cpu only)" |
| 2024-02-09 | [#5419](https://github.com/ggml-org/llama.cpp/pull/5419) | b2109 | "do not cap thread count when MoE on CPU". The diff changes only the BLAS path for batches of 32 or more tokens (prompt processing), not single-token decode. |
| 2024-02-16 | [#5377](https://github.com/ggml-org/llama.cpp/pull/5377) | b2167 | `--numa` options (distribute/isolate/numactl) |
| 2024-04-03 | [#6387](https://github.com/ggml-org/llama.cpp/pull/6387) | b2589 | "mul_mat_id use the same tensor for all the experts": stacked 3-D expert tensors (`ffn_{gate,up,down}_exps`), a GGUF layout change. The loader keeps backward compatibility by merging old split-expert tensors (`blk.%d.ffn_gate.%d`) at load. |
| 2024-04-16 | [#6414](https://github.com/ggml-org/llama.cpp/pull/6414) | b2685 | llamafile sgemm (CPU GEMM, mainly prompt processing) |
| 2024-04-30 | [#5021](https://github.com/ggml-org/llama.cpp/pull/5021) | b2771 | "ggml : add Flash Attention" (opt-in `-fa`) |
| 2024-06-03 | [#7606](https://github.com/ggml-org/llama.cpp/pull/7606) | b3077 | OpenMP used as the CPU thread pool |
| 2024-08-30 | [#8672](https://github.com/ggml-org/llama.cpp/pull/8672) | b3644 | "Threadpool: take 2" (plus [#9598](https://github.com/ggml-org/llama.cpp/pull/9598), 2024-09-23, barrier scaling for large thread counts) |
| 2024-11-16 / 2024-12-07 | [#10324](https://github.com/ggml-org/llama.cpp/pull/10324), [#10446](https://github.com/ggml-org/llama.cpp/pull/10446) | b4096 / b4282 | Online (load-time) repacking of Q4_0 CPU weights, then a refactor of online repacking |
| 2025-02-13 | [#11666](https://github.com/ggml-org/llama.cpp/pull/11666) | b4702 | "ggml-cpu : add chunking support to mul_mat_id" |
| 2025-03-14 | [#11571](https://github.com/ggml-org/llama.cpp/pull/11571) | b4888 | "Load all MoE experts during warmup" (common warm-up for CLI/server) |
| 2025-03-20 | [#12332](https://github.com/ggml-org/llama.cpp/pull/12332) | b4929 | Q4_K block interleaving (repack) for x86 AVX2 |
| 2025-03-21 | [#12498](https://github.com/ggml-org/llama.cpp/pull/12498) | b4936 | "model : do not repack if a GPU device is present". The diff adds CPU "extra" (repack) buffer types only when no GPU device exists. |
| 2025-03-29 | [#12632](https://github.com/ggml-org/llama.cpp/pull/12632) | b4991 | CPU buffer-type priority order changed to "ACCEL -> GPU host -> CPU extra -> CPU" |
| 2025-04-02 | [#11397](https://github.com/ggml-org/llama.cpp/pull/11397) | b5028 | **`--override-tensor` / `-ot` `<regex>=<buffer type>`** ("add option to override model tensor buffers"). The PR was opened around 2025-01-24/25 (inferred from neighbouring PR numbers #11396 and #11403–#11409, merged 2025-01-24/25). |
| 2025-04-15 | [#12829](https://github.com/ggml-org/llama.cpp/pull/12829) | b5136 | AVX512 Q4_Kx8 GEMM |
| 2025-04-27 | [#12922](https://github.com/ggml-org/llama.cpp/pull/12922) | b5200 | `llama-bench` gets `-ot` |
| 2025-04-28 | [#13096](https://github.com/ggml-org/llama.cpp/pull/13096) | b5212 | `llama-bench -d` (context depth) |
| 2025-06-19 | [#14270](https://github.com/ggml-org/llama.cpp/pull/14270) | b5706 | `llama-bench --no-warmup` |
| 2025-07-18 | [#14753](https://github.com/ggml-org/llama.cpp/pull/14753) | b5933 | "avoid huge warm-up graphs for MoE models" |
| 2025-07-31 | [#14990](https://github.com/ggml-org/llama.cpp/pull/14990) | b6049 | **`--no-repack`**, plus: "when overriding to a CPU buffer, consider the extra buffer types". This is the first point at which experts sent to CPU with `-ot` use repacked CPU kernels while a GPU is present. |
| 2025-07-31 | [#14992](https://github.com/ggml-org/llama.cpp/pull/14992) | b6051 | **`--cpu-moe`**: overrides `ffn_{up,down,gate}_exps` to CPU |
| 2025-08-05 (authored 08-04) | [#15077](https://github.com/ggml-org/llama.cpp/pull/15077) | b6089 | **`--n-cpu-moe N` / `-ncmoe`** ("keep the MoE weights of the first N layers in the CPU"). The same diff adds the `-cmoe` short alias for `--cpu-moe`. It is implemented as per-layer `-ot` regexes `blk\.%d\.ffn_(up\|down\|gate)_exps` → CPU. |
| 2025-08-05 | [#15091](https://github.com/ggml-org/llama.cpp/pull/15091) | b6096 | **gpt-oss + MXFP4** (first MXFP4 commit in the log) |
| 2025-08-13 | [#15191](https://github.com/ggml-org/llama.cpp/pull/15191) | b6148 | Draft-model variants `--override-tensor-draft`, `--cpu-moe-draft`, `--n-cpu-moe-draft` |
| 2025-08-30 | [#15434](https://github.com/ggml-org/llama.cpp/pull/15434) | b6325 | **"use FA + max. GPU layers by default"** (`-fa auto`; `-ngl` defaults to maximum). Before this, `common.h` had `flash_attn = false`. |
| 2025-09-02 | [#15746](https://github.com/ggml-org/llama.cpp/pull/15746) | b6358 | `-fa 1/0/-1` aliases for on/off/auto |
| 2025-09-16 | [#15952](https://github.com/ggml-org/llama.cpp/pull/15952) | b6490 | `llama-bench --n-cpu-moe` |
| 2025-09-24 | [#15860](https://github.com/ggml-org/llama.cpp/pull/15860) | b6569 | "print memory breakdown on exit" (`llama_memory_breakdown_print`). Server support followed in [#16740](https://github.com/ggml-org/llama.cpp/pull/16740) on 2025-10-23 (b6829). |
| 2025-12-15 | [#16653](https://github.com/ggml-org/llama.cpp/pull/16653) | b7410 | **Automatic fitting**: `-fit/--fit [on\|off]` (default on), `-fitt/--fit-target MiB` (default margin 1024 MiB per device), `-fitc/--fit-ctx` (default minimum ctx 4096), and the `llama-fit-params` tool. Title: "automatically set parameters not set by the user in such a way that maximizes GPU utilization". |
| 2025-12-24 | [#17906](https://github.com/ggml-org/llama.cpp/pull/17906) | b7536 | "CUDA: experimental native mxfp4 support for blackwell" |
| 2025-12-24 | [#18267](https://github.com/ggml-org/llama.cpp/pull/18267) | b7525 | `LLAMA_ARG_OVERRIDE_TENSOR` env var for `-ot` |
| 2026-01-05 | [#18593](https://github.com/ggml-org/llama.cpp/pull/18593) | b7625 | "CUDA: disable cuda graph when using n-cpu-moe" |
| 2026-01-08 | [#18679](https://github.com/ggml-org/llama.cpp/pull/18679) | b7672 | `--fit-target` per device |
| 2026-01-08 → 2026-01-28 | [#18166](https://github.com/ggml-org/llama.cpp/pull/18166), [#18841](https://github.com/ggml-org/llama.cpp/pull/18841), [#19109](https://github.com/ggml-org/llama.cpp/pull/19109) | b7668, b7755, b7853 | Direct-IO loading flag. `llama-bench` defaults set to mmap=0/direct_io=0 (2026-01-16). Direct IO disabled by default (2026-01-28). |
| 2026-01-24 | [#18934](https://github.com/ggml-org/llama.cpp/pull/18934) | b7821 | "ggml-cuda: enable cuda-graphs for `n-cpu-moe`" (one CUDA graph per split) |
| 2026-02-27 | [#19738](https://github.com/ggml-org/llama.cpp/pull/19738) | b8175 | "ggml-cpu: add repack for mxfp4" |
| 2026-03-09 | [#20211](https://github.com/ggml-org/llama.cpp/pull/20211) | b8247 | `llama-bench` back to `--mmap 1` by default |
| 2026-03-12 | [#20416](https://github.com/ggml-org/llama.cpp/pull/20416) | b8301 | Fix for `--n-cpu-moe`/`--cpu-moe` on models with fused gate+up expert tensors |
| 2026-03-25 | [#20984](https://github.com/ggml-org/llama.cpp/pull/20984) | b8522 | `llama-bench` prints the `n-cpu-moe` column |
| 2026-04-06 | [#21304](https://github.com/ggml-org/llama.cpp/pull/21304) | b8679 | `llama-bench -fitt/-fitc` |
| 2026-04-21 | [#22171](https://github.com/ggml-org/llama.cpp/pull/22171) | b8868 | fit-params refactor, plus an option to print estimated memory per device (`-fitp`) |
| 2026-05-31 | [#23714](https://github.com/ggml-org/llama.cpp/pull/23714) | b9437 | `llama-bench -fa auto` |
| 2026-07-05 | [#25028](https://github.com/ggml-org/llama.cpp/pull/25028) | (not looked up) | Fix for tensor-parallel + `-ncmoe` crash (confirms `-ncmoe` in use with the 2026 tensor-parallel mode) |
| 2026-07-23 → 2026-08-15 | [#20834](https://github.com/ggml-org/llama.cpp/pull/20834), [#26081](https://github.com/ggml-org/llama.cpp/pull/26081), [#26934](https://github.com/ggml-org/llama.cpp/pull/26934) | b10105, b10369, b10441 | mlock/mmap/directio refactored into `-lm/--load-mode`. Default `auto` ("mmap, unless a device does not support it"). `--mmap/--no-mmap` deprecated in favour of `--load-mode`. |
| 2026-08-27 | [#26622](https://github.com/ggml-org/llama.cpp/pull/26622) | b10645 | `--n-cpu-ffn` (the dense-model analogue of `--n-cpu-moe`) |
| 2026-09-26 | [#27851](https://github.com/ggml-org/llama.cpp/pull/27851) | (after b11206) | "ggml-cpu: tiled mul_mat for k-quants" |

**Current (2026-09) flag semantics, from master [common/arg.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp) and [common/common.h](https://github.com/ggml-org/llama.cpp/blob/master/common/common.h)**
- `-ngl` accepts an exact number, `auto` or `all`. The default is `auto` (−1).
- `-ncmoe/--n-cpu-moe N` keeps "the MoE weights of the first N layers in the CPU". `-cmoe/--cpu-moe` keeps all MoE weights on CPU. `-ot` sets a tensor buffer type by regex.
- `--repack` / `-nr,--no-repack`: repacking is enabled by default (`no_extra_bufts=false`).
- `-lm/--load-mode` takes `auto|none|mmap|mlock|mmap+mlock|dio` (default `auto`).
- `--fit` is on by default. `--fit-target` defaults to 1024 MiB per device. `--fit-ctx` defaults to 4096.
- Flash attention defaults to `LLAMA_FLASH_ATTN_TYPE_AUTO`.

**How `--fit` works** (source: [common/fit.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/fit.cpp) comments and [tools/fit-params/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/fit-params/README.md))
- Step 2 reduces the context size first. Step 3 fills "dense" layers back to front, with "all MoE tensors in system memory". Step 4 converts dense-only layers into full layers front to back until the devices are full.
- It works on parameters the user did not set.
- Example printed in the README (Qwen3-30B-A3B F16 on an RTX 4090): `-c 4096 -ngl 48 -ot blk\.14\.ffn_(up|down|gate)_(ch|)exps=CPU, … blk\.47…=CPU`. The context dropped from 40960 to 4096, and the experts of the *last* 34 layers went to CPU.

**Ollama**
- Up to v0.30.0, Ollama had its own estimator. `llm/memory.go` `EstimateGPULayers` predicted "how many layers and bytes we can load" at whole-layer granularity and stopped "once we hit the users target NumGPU" — [ollama v0.5.0 llm/memory.go](https://github.com/ollama/ollama/blob/v0.5.0/llm/memory.go). A later commit, "llm: Use Ollama engine memory layouts for both old and new engines", is dated 2025-11-11 — [ollama commit f560bd07](https://github.com/ollama/ollama/commit/f560bd07).
- On 2026-05-29, commit 9db4bdba, "runner: Remove CGO engines, use llama-server exclusively for GGML models", replaced the vendored llama.cpp with "llama-server (built from upstream llama.cpp via FetchContent)". The first release tag containing it is v0.30.0, dated 2026-06-01 — [ollama PR #16031](https://github.com/ollama/ollama/pull/16031).
- Current Ollama `llm/llama_server.go` says "llama-server auto-detects GPU layers (-ngl), thread count (-t), and flash attention". It passes `-ngl` only if the user set `num_gpu` ("NumGPU == -1 (default): don't pass -ngl, let llama-server auto-detect"). It also sets `LLAMA_ARG_FIT_TARGET` (padding for projectors) and parses `--fit`'s "N layers (M overflowing)" log line — [ollama llm/llama_server.go](https://github.com/ollama/ollama/blob/main/llm/llama_server.go).
- Ollama has no user-facing expert-placement option (no `-ot`/`--n-cpu-moe` passthrough appears in `llama_server.go`). The feature request "use cpu to offload moe weights to reduce the VRAM usage" has users reporting 2.5–3.5× speedups from using llama.cpp `--n-cpu-moe` directly. One commenter says `LLAMA_ARG_CPU_MOE=1` works as a global env var in 0.30.0. No maintainer response was visible — [ollama issue #11772](https://github.com/ollama/ollama/issues/11772).

### Inferences
- Before 2025-04-02 (b5028), mainline llama.cpp could not place MoE experts separately from attention. A paper dated before then that used `-ngl` partial offload was using the best mainline configuration available. The `-ot` PR branch was public from late January 2025 and widely used for DeepSeek-R1, but it was unmerged.
- Between b5028 (2025-04-02) and b6049 (2025-07-31), experts placed on CPU with `-ot` while a GPU was present lost CPU weight repacking, because of #12498. The CPU half of an `-ot` hybrid config in that window was somewhat handicapped relative to later builds for quant types that repack. For decode specifically, repacking has little effect (see Q2).
- Since Ollama 0.30.0, Ollama's MoE placement is llama.cpp's `--fit` placement (experts of the last layers go to CPU). Ollama-based baselines from before June 2026 used whole-layer offload.

### Gaps
- `-ngl` default before #15434: the commit title says it moved to "max. GPU layers by default", and `common.h` before it had `n_gpu_layers = -1 // use default`. The library default value (0 layers) was not read directly from `llama_model_default_params` at that revision.
- The ik_llama.cpp fork's date for adding `-ot`-style overrides was not verified. The local clone lacked history.
- No primary-source date was found for when llama.cpp's load-time log lines (per-buffer "model buffer size", KV and compute buffer sizes) first appeared. They are widely present in 2024 logs, but this is unverified.

## Q2. Were there large CPU-side decode improvements for quantized MoE between 2024 and 2026?

### Takeaway
Many CPU kernel changes landed: stacked expert tensors (2024-04), OpenMP and custom thread pools (2024-06/08), online repacking (2024-11 onward), `mul_mat_id` chunking (2025-02), Q4_K AVX2/AVX512 repack (2025-03/04), repack for CPU-overridden experts (2025-07), and MXFP4 repack (2026-02). The primary benchmarks found show these mostly help **prompt processing**, not single-token decode, which stays memory-bandwidth-bound. The measurable version-dependent decode gains for hybrid MoE are on the orchestration side: `-ot` versus `-ngl` placement (tens of percent), and CUDA graphs with `n-cpu-moe` (+4–10%). No same-hardware 2024-versus-2026 decode comparison was found.

### Cited Findings
- **Q4_K AVX2 interleaving (#12332, 2025-03-20)**, Llama-2 7B on a Ryzen 7600X with 6 threads: Q4_K_M pp512 went from 45.80 to 73.77 t/s (+61%), while tg128 went from 14.09 to 13.84 t/s (−1.8%). Q4_K_S pp512 went from 46.60 to 81.87 (+76%), while tg128 went from 14.91 to 14.61 (−2.0%). Decode did not improve — [llama.cpp PR #12332](https://github.com/ggml-org/llama.cpp/pull/12332).
- **CUDA graphs for `n-cpu-moe` (#18934, 2026-01-24)**: tg improved 1.04–1.10× across n_cpu_moe 8–64.
  - GLM4-MoE 106B IQ4_XS: 60.84→63.12 t/s (ncmoe 8); 25.08→27.49 (ncmoe 64).
  - gpt-oss-120b MXFP4: 95.96→100.93 (ncmoe 8); 40.87→44.90 (ncmoe 64).
  - A DeepSeek-V3 7-GPU test gave about 5% on tg.
  - Root cause: "cuda graphs get disabled when there are splits as we only keep 1 cuda graph per device" — [llama.cpp PR #18934](https://github.com/ggml-org/llama.cpp/pull/18934).
  - The immediately preceding change (#18593, 2026-01-05, b7625) explicitly disabled CUDA graphs with n-cpu-moe, so builds b7625–b7820 are a slower window.
- **Placement matters more than kernels**, per the `-ot` PR discussion:
  - Mixtral 8x22B Q4_K_M on an i7-10700 with 128 GB RAM, 2×RTX 3090 and a GTX 1660S: tg rose from 2.08 t/s to 3.45 t/s (+66%) moving to tensor overrides, partly by fitting more layers with a q8 KV cache.
  - DeepSeek-R1: `-ngl 0` gave 4.65 t/s; `-ngl 40 -ot exps=CPU` gave 6.95 t/s.
  - ubergarm: "Running with `-ngl 62 -ot exps=CPU` is the fastest way" on a Threadripper Pro 24-core + 1 GPU.
  - Source: [llama.cpp PR #11397](https://github.com/ggml-org/llama.cpp/pull/11397).
- **#14990 (repack for CPU-overridden tensors)**: the PR page shows CPU-only Qwen3-30B Q4_0 benchmarks, with pp speedups of 1.36–1.56× versus master as summarized by the fetch tool. No GPU+CPU-expert tg numbers were shown — [llama.cpp PR #14990](https://github.com/ggml-org/llama.cpp/pull/14990). The summary was ambiguous about which side was repacked; treat it as low confidence.
- **Tiled k-quant mul_mat (#27851, 2026-09-26)**: the commit message reports a "3-6x speed improvement for large matmul, break even at 4096x64 * 64x4096, 80% performance (net loss) for GEMV". This change targets batched and prompt workloads, not decode — [llama.cpp PR #27851](https://github.com/ggml-org/llama.cpp/pull/27851).
- **Thread count cap (#5419, 2024-02-09)**: this touched only the BLAS path when `n_tokens >= 32` (prompt processing), so it did not affect decode — [llama.cpp PR #5419](https://github.com/ggml-org/llama.cpp/pull/5419) (diff inspected in the local clone).
- **Community end-to-end example, 2026-03-28**: Nemotron-3-Super-120B-A12B UD-Q3_K_XL on an RTX 3090 with 128 GB DDR5 and a Core Ultra 265K, at depth 4096:
  - `-ngl 89 -ot <custom regex>`: 16.6 t/s
  - the same with `--mmap 0`: 16.8 t/s
  - `-ncmoe 999` (all experts on CPU, so less VRAM used): 12.72 t/s
  - 8 P-core threads (the thread-count difference is as reported): 12.23 t/s
  - Source: [llama.cpp Discussion #21112](https://github.com/ggml-org/llama.cpp/discussions/21112).

### Inferences
- For bandwidth-bound single-token decode, CPU expert GEMV time is roughly bytes read ÷ achievable DRAM bandwidth. Kernel rewrites that help GEMM (repack, sgemm, tiling) are therefore expected to change CPU-side tg only modestly, as #12332's −2% tg shows. The larger 2024→2026 differences for hybrid MoE decode are expected from:
  - expert-granular placement (`-ot`/`--n-cpu-moe`), which keeps attention and KV on the GPU and puts more of the per-token bytes on the GPU for the same VRAM;
  - CUDA graphs per split (+4–10%);
  - flash attention and quantized KV freeing VRAM;
  - thread pool and barrier overheads (2024), which matter for many small per-token ops. These are unquantified here.
- A 2024 llama.cpp is therefore probably not "much slower" per byte of CPU-resident expert weights. It is slower mainly because it could not choose *which* bytes stay on the GPU. This is a hypothesis to test by running old tags on the rented machine.

### Gaps
- No primary or secondary source was found that runs the same MoE model on the same hardware across llama.cpp builds from 2024 to 2026 for decode. The recommendation is to measure it directly: check out b1742 (Jan 2024), b4404 (Jan 2025), b5028 (first `-ot`), b6096 (Aug 2025) and b11206 (Sep 2026), then run the same GGUF.
  - Pre-b2589 builds cannot load post-b2589 stacked-expert GGUFs. The reverse direction is supported, because the new loader merges split experts. An old-format GGUF would be needed for builds from before 2024-04-03.
  - Old CUDA code may not build for new GPU architectures such as Blackwell sm_120. This is an unverified practical risk.
- No numbers were found for the OpenMP and thread pool changes (#7606, #8672, #9598) on decode latency.

## Q3. For a paper dated at time T, what was the best "equal GPU memory" llama.cpp configuration available?

### Takeaway
Judge a paper against the configuration available at its date: `-ngl` only before 2025-04-02; `-ngl 99 -ot <per-layer expert regex>=CPU` from 2025-04-02; `--n-cpu-moe` from 2025-08-05; `--fit` (automatic) from 2025-12-15. Repacked CPU experts under hybrid offload date from 2025-07-31. FA was available opt-in from 2024-04-30 and on by default from 2025-08-30. CUDA graphs with `n-cpu-moe` exist from 2026-01-24. The expert-regex configuration was publicly known, as an open PR, from about late January 2025.

### Cited Findings
- `-ngl` whole-layer MoE offload since 2023-12-13 (b1629) — [PR #4406](https://github.com/ggml-org/llama.cpp/pull/4406).
- FA opt-in since 2024-04-30 (b2771) — [PR #5021](https://github.com/ggml-org/llama.cpp/pull/5021).
- `-ot` since 2025-04-02 (b5028), with the PR opened around 2025-01-24/25 (inferred) — [PR #11397](https://github.com/ggml-org/llama.cpp/pull/11397).
- `llama-bench -ot` since 2025-04-27 (b5200) — [PR #12922](https://github.com/ggml-org/llama.cpp/pull/12922).
- Repack for CPU-overridden weights with a GPU present since 2025-07-31 (b6049) — [PR #14990](https://github.com/ggml-org/llama.cpp/pull/14990).
- The repack-disabled-with-GPU window was 2025-03-21 → 2025-07-31 — [PR #12498](https://github.com/ggml-org/llama.cpp/pull/12498).
- `--cpu-moe` since b6051 and `--n-cpu-moe` since b6089 — [PR #14992](https://github.com/ggml-org/llama.cpp/pull/14992), [PR #15077](https://github.com/ggml-org/llama.cpp/pull/15077).
- `llama-bench --n-cpu-moe` since 2025-09-16 (b6490) — [PR #15952](https://github.com/ggml-org/llama.cpp/pull/15952).
- FA auto and max `-ngl` by default since 2025-08-30 (b6325) — [PR #15434](https://github.com/ggml-org/llama.cpp/pull/15434).
- `--fit` since 2025-12-15 (b7410) — [PR #16653](https://github.com/ggml-org/llama.cpp/pull/16653).
- CUDA graphs with `n-cpu-moe` since 2026-01-24 (b7821) — [PR #18934](https://github.com/ggml-org/llama.cpp/pull/18934).
- The official gpt-oss guide (2025-08-18) recommends `--n-cpu-moe` for partial offload, for example `llama-server -hf ggml-org/gpt-oss-120b-GGUF --ctx-size 32768 --jinja -ub 4096 -b 4096 --n-cpu-moe 32` for 16 GB VRAM — [llama.cpp Discussion #15396](https://github.com/ggml-org/llama.cpp/discussions/15396).

**Best-available equal-GPU-memory configuration by paper date** (built from the dated findings above; the cell contents are inferences)

| Paper date T | llama.cpp tag range | Best mainline equal-VRAM MoE config | Notes on fairness |
|---|---|---|---|
| 2023-12-13 → 2024-04-29 | b1629–b2770 | `-ngl N` (max N that fits), no FA | Whole layers (attention + all experts + that layer's KV) go to the GPU. KV of CPU layers stays on CPU. Old split-expert GGUFs before b2589. |
| 2024-04-30 → 2025-01-23 | b2771–~b4550 | `-ngl N -fa` (FA opt-in; optional `-ctk/-ctv q8_0`) | FA and quantized KV free VRAM for one or two more layers. Thread pool/OpenMP from mid-2024. |
| ~2025-01-24 → 2025-04-01 | ~b4550–b5027 | Mainline: `-ngl N -fa`. Community: unmerged `-ot` PR branch, ik_llama.cpp, ktransformers. | A strict "released llama.cpp" standard allows `-ngl` only. A "state-of-practice" standard would credit `-ot exps=CPU`, which was publicly used for DeepSeek-R1. |
| 2025-04-02 → 2025-07-30 | b5028–b6048 | `-ngl 99 -fa -ot "blk\.(k..L-1)\.ffn_.*_exps\.=CPU"` (fill the GPU with as many layers' experts as fit) | CPU experts not repacked while a GPU is present (#12498), which is a minor handicap for decode. `llama-bench` supports `-ot` from b5200. |
| 2025-07-31 → 2025-08-29 | b6049–b6324 | `-ngl 99 -fa on --n-cpu-moe K` (≡ the `-ot` regex; from b6089) | Repacked CPU experts. gpt-oss/MXFP4 from b6096. |
| 2025-08-30 → 2025-12-14 | b6325–b7409 | `--n-cpu-moe K` (FA auto by default; `-ngl` max by default) | Memory breakdown print from b6569 (2025-09-24). `llama-bench -ncmoe` from b6490. |
| 2025-12-15 → 2026-01-23 | b7410–b7820 | `--fit on --fit-target M` or explicit `--n-cpu-moe K` | CUDA graphs disabled with n-cpu-moe in b7625–b7820 (~5–10% tg penalty). |
| 2026-01-24 → now | b7821–b11206+ | Explicit `--n-cpu-moe K` (or `-ot`), or `--fit` with a pinned `-c`; FA auto | CUDA graphs per split. Per-device `--fit-target`. MXFP4 CPU repack from b8175. `llama-bench -fitt` from b8679. |

### Inferences
- **Fair-baseline rule for the audit.** Grade a paper's llama.cpp baseline against the row for its submission date, or better its arXiv v1 date. A paper from before April 2025 that used `-ngl` at equal VRAM was not "weak" by the standard of the day. After April 2025, an `-ngl`-only baseline for an MoE model under equal VRAM omits a documented, mainline, roughly 1.5× class improvement (PR #11397 anecdotes: +50–66%). After August 2025, it omits the officially documented method (gpt-oss guide).
- A two-level verdict is defensible:
  - "weak for its time" (did worse than the best config at T);
  - "stale by today" (fine at T but superseded; report the 2026 number separately and do not penalize).
- `--n-cpu-moe K` and `-ot` experts→CPU on the same number of layers are functionally identical: both are implemented as the same buffer overrides. So a 2025-04 paper using `-ot` should be treated as equivalent to `--n-cpu-moe`.
- `--fit` offloads the experts of the last layers and `--n-cpu-moe` those of the first layers. At equal count, and with homogeneous layers, performance should be about equal. This is a hypothesis; verify it once.

### Gaps
- There is no official llama.cpp statement of a "recommended MoE offload config" before the gpt-oss guide (2025-08-18). The pre-August-2025 practice is documented only via PR and discussion comments.
- The exact merge date of the ik_llama.cpp `-ot` equivalent, which is relevant to a "state of practice" standard in Feb–Mar 2025, was not verified.

## Q4. How does llama-bench measure tg, how does that map to paper metrics, and how does context length change decode speed?

### Takeaway
`llama-bench` tg128 times 128 single-token `llama_decode` calls on random token IDs, synchronizing after each call. It excludes tokenization, sampling, model load, warm-up and any `-d` depth prefill. It starts from an empty KV cache unless `-d` is given, and reports the mean ± standard deviation of per-repetition t/s (default `-r 5`). This equals decode throughput ≈ 1/TPOT at near-zero context. It is not end-to-end tok/s, which includes prefill. Papers' numbers must be matched by choosing `-d` (context), `-p/-n` or `-pg`, or by using `llama-server` timings.

### Cited Findings
- `test_gen` loops `n_gen` times over `llama_decode(ctx, llama_batch_get_one(&token, 1))`, calls `llama_synchronize(ctx)` after each, then picks the next token as `std::rand() % n_vocab`. There is no sampler — [tools/llama-bench/llama-bench.cpp (master)](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/llama-bench.cpp).
- In each repetition, `llama_memory_clear` is called. If `n_depth > 0`, the depth prompt is processed, or restored from a cached state after the first repetition, *before* `t_start`. Then the timer covers the `-p` prompt run (if any) plus the `-n` generation run. Per-rep t/s = `1e9 * (n_prompt + n_gen) / t_ns`. `avg_ts` and `stdev_ts` are the arithmetic mean and standard deviation of per-rep t/s. The JSON/JSONL output includes `samples_ns` and `samples_ts` for every repetition — same source.
- Warm-up (default on): a full `-p` prompt run if n_prompt > 0, and a **1-token** generation run if n_gen > 0 — same source.
- `llama-bench` sets `n_ctx = n_prompt + n_gen + n_depth`, so its KV cache is only as large as the test (line ~1308 of master `llama-bench.cpp`) — same source.
- README: default `-p 512 -n 128 -r 5`. "Each test is repeated the number of times given by `-r`, and the results are averaged. The results are given in average tokens per second (t/s) and standard deviation." `-d <n>` runs "at a specified context depth, prefilling the KV cache with `<n>` tokens". "The measurements with `llama-bench` do not include the times for tokenization and for sampling." `-pg <pp,tg>` runs prompt processing followed by generation as one timed test. Current defaults: `-ngl -1`, `-ncmoe 0`, `-fa auto`, `--repack 1`, `-fitt` "(default: off)" — [tools/llama-bench/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md).
- Historic `llama-bench` defaults at b3268 (mid-2024): `n_gpu_layers {99}`, `flash_attn {false}`, `use_mmap {true}`, `reps 5`, `n_threads {cpu_get_num_math()}` — local clone, `examples/llama-bench/llama-bench.cpp` at [b3268](https://github.com/ggml-org/llama.cpp/releases/tag/b3268).
- `llama-server` returns per-request `timings` including `prompt_per_token_ms`, `prompt_per_second`, `predicted_ms`, `predicted_per_token_ms` and `predicted_per_second` — [tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
- Standard metric definitions from NVIDIA (2025-04-02):
  - TTFT covers queuing, prefill and network.
  - e2e_latency = TTFT + generation_time.
  - ITL/TPOT = (e2e_latency − TTFT)/(output_tokens − 1), which isolates "the decoding part".
  - Per-user tok/s = output length / e2e_latency, which "asymptotically approaches 1/ITL".
  - Source: [NVIDIA, LLM Inference Benchmarking: Fundamental Concepts](https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/).
- Community long-context data point: Qwen 35B-A3B Q4_K_M, with MoE experts on CPU and the rest on an RTX 2060 6 GB (32 GB RAM, i7-9750H), gave about 15 t/s tg at 102K context. No depth-0 control was reported — [llama.cpp Discussion #21112](https://github.com/ggml-org/llama.cpp/discussions/21112).
- `--fit` reduces context first ("step 2: try reducing memory use by reducing the context size", down to `--fit-ctx` 4096) before moving experts — [common/fit.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/fit.cpp).

### Inferences
- **Mapping llama-bench output to paper metrics:**
  - Paper "decode tok/s" or "TPOT" at context C ↔ `llama-bench -p 0 -n 128 -d C`, and TPOT ≈ 1000/tg_ts ms. The tg is averaged over positions C..C+127.
  - Paper "end-to-end tok/s" with prompt P and output N (output tokens/(TTFT + decode)) ≈ N / (P/pp_P + N/tg@P), from separate `-p P -n 0` and `-p 0 -n N -d P` runs.
  - `-pg P,N` reports (P+N)/total, which counts prompt tokens as throughput. It is **not** the usual e2e tok/s. Convert it before comparing.
  - Alternatively, run `llama-server` with the paper's prompts and read `predicted_per_token_ms` (TPOT-like) and `prompt_per_second`. This includes sampling, which llama-bench excludes.
- tg128 at depth 0 overstates decode speed relative to a paper that measured at, for example, 512–2K prompt tokens. The overstatement is small for expert-on-CPU hybrids, where attention and KV stay on the GPU and per-token time is dominated by CPU expert reads. It is larger for `-ngl`-partial configs, where the CPU layers keep their KV on CPU and run attention on CPU. This is reasoning from where KV lives, not a measured result.
- More importantly, under **equal GPU memory** a longer context means a larger KV cache in VRAM, which forces more experts to CPU and lowers decode speed. Equal-memory comparisons must therefore be defined at the paper's context length (fix `-c`, `-ctk/-ctv`, `-fa`, `-ub`), not at llama-bench's minimal `n_ctx`. llama-bench's automatically sized context (P+N+d) means its VRAM footprint is smaller than a server's with `-c 32768`. Budget the KV explicitly, for example by running at `-d C`.

### Gaps
- No controlled published measurement was found of tg versus depth for `--n-cpu-moe` hybrids compared with `-ngl`-partial on the same hardware. This should be measured with `-d 0,512,2048,8192,32768`.
- No primary source was found stating which paper metric each audited paper used. That is outside this note's scope (paper-side extraction).

## Q5. How to set "equal GPU memory" precisely, and good practice for threads, NUMA, mmap and warm-up

### Takeaway
Pin every memory-relevant parameter explicitly: `-c`, `-ctk/-ctv`, `-fa`, `-ub/-b` and `--n-cpu-moe K` (or `-ot`). Read the per-device "self = model + context + compute" breakdown that llama.cpp prints (since b6569), and choose the smallest K whose "self" fits the paper's GPU budget. `--fit-target` is a free-memory *margin* per device (default 1024 MiB), not a budget. `--fit` also silently shrinks context down to 4096, so do not rely on it for equal-memory comparisons without a pinned `-c`. Use physical-core threads, drop page caches, disable mmap (or use `--load-mode none`/mlock) when measuring, and keep warm-up on.

### Cited Findings
- **Memory breakdown**:
  - `llama_memory_breakdown_print` prints `| total free self model context compute unaccounted |` per device, plus Host. Example: `CUDA0 (RTX 4090) | 24077 = 945 + (19187 = 17904 + 384 + 898) + 3945`; `Host | 58271 = 58259 + 0 + 12` — [tools/fit-params/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/fit-params/README.md).
  - It was introduced 2025-09-24 in [PR #15860](https://github.com/ggml-org/llama.cpp/pull/15860) (b6569), and in the server in [PR #16740](https://github.com/ggml-org/llama.cpp/pull/16740) (b6829).
  - `-fitp/--fit-print` prints the estimated required memory ([PR #22171](https://github.com/ggml-org/llama.cpp/pull/22171), b8868).
- **`--fit-target`**:
  - Help text: "target margin per device for --fit, comma-separated list of values, single value is broadcast across all devices, default: 1024" (MiB). Log wording: "cannot fulfill margin of 1024 MiB, need to reduce device memory by …" — [common/arg.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp), [tools/fit-params/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/fit-params/README.md).
  - `--fit` adjusts only "parameters not set by the user" — [PR #16653](https://github.com/ggml-org/llama.cpp/pull/16653).
  - `llama-fit-params` prints the equivalent CLI arguments (`-c … -ngl … -ot …`) for reuse — same README.
- **Where Ollama gets its placement**: Ollama injects `LLAMA_ARG_FIT_TARGET` padding and otherwise defers to llama-server's fit — [ollama llm/llama_server.go](https://github.com/ollama/ollama/blob/main/llm/llama_server.go).
- **Threads**:
  - `llama-bench` defaults `n_threads` to `common_cpu_get_num_math()`, a math-core (physical core) count — [llama-bench.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/llama-bench.cpp).
  - `llama-bench` also exposes `-C/--cpu-mask`, `--cpu-strict`, `--poll` (default 50), `--prio` and `--delay` — [llama-bench README](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md).
- **NUMA**: the `--numa distribute|isolate|numactl` help says "if run without this previously, it is recommended to drop the system page cache before using this" — [common/arg.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp).
- **mmap / load mode**:
  - `--load-mode` default `auto` ("mmap, unless a device does not support it"). Options: none, mmap, mlock, mmap+mlock, dio — [common/arg.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp).
  - The `llama-bench` default mmap flipped off on 2026-01-16 ([PR #18841](https://github.com/ggml-org/llama.cpp/pull/18841)) and back on on 2026-03-09 ([PR #20211](https://github.com/ggml-org/llama.cpp/pull/20211)). Measurements from those windows used different defaults.
  - Community measurement: `--mmap 0` gave 16.8 versus 16.6 t/s (Nemotron-120B hybrid) — [Discussion #21112](https://github.com/ggml-org/llama.cpp/discussions/21112).
  - Ollama's code notes that repacked CPU buffers (for example `CPU_REPACK`) are copies, not mmap views — [ollama llm/llama_server.go](https://github.com/ollama/ollama/blob/main/llm/llama_server.go).
- **Warm-up**:
  - The common CLI/server warm-up has "Load[ed] all MoE experts during warmup" since 2025-03-14 ([PR #11571](https://github.com/ggml-org/llama.cpp/pull/11571), b4888). The graph uses `n_expert` instead of `n_expert_used` when `cparams.warmup` is set — [src/llama-graph.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-graph.cpp).
  - `llama-bench` warm-up is a plain 1-token `test_gen` with no warm-up flag set (no `llama_set_warmup`/`warmup` reference in master `llama-bench.cpp`) — [llama-bench.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/llama-bench.cpp).
- **Fused gate+up models**: `--n-cpu-moe` had to be fixed for fused gate+up expert tensors on 2026-03-12 ([PR #20416](https://github.com/ggml-org/llama.cpp/pull/20416)). The current fit regex matches `ffn_(up|down|gate_up|gate)_(ch|)exps` — [common/fit.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/fit.cpp).

### Inferences
- **Recommended equal-memory procedure** for a paper budget B GiB of GPU memory at context C:
  1. Build at a recorded tag.
  2. Run `llama-server` (or `llama-cli`) with `-c C -fa on -ctk f16 -ctv f16 -ub 512 -b 2048 -ngl all --fit off --n-cpu-moe K` for K = L, L−1, …
  3. Read the exit breakdown "self" for CUDA0.
  4. Pick the smallest K with self ≤ B, noting any "unaccounted" such as CUDA context. Report model, context and compute separately.
  5. Measure tg with `llama-bench -ngl 99 -ncmoe K -fa 1 -p 0 -n 128 -d C` (or `-d` 0/512 to match the paper). Note that llama-bench's own KV will be smaller than at `-c C`, so K must come from the server-style run, not from llama-bench.
- To use `--fit` instead: set `-c C` explicitly, so fit cannot shrink context, and set `--fit-target` = (free VRAM − B) in MiB. Then verify with the breakdown. This only emulates a budget on a larger card and is approximate, because fit works at whole-tensor granularity.
- **CUDA context overhead.** For comparisons to papers whose "GPU memory" figure includes the CUDA context and allocator overhead (nvidia-smi-style), add "unaccounted" or measure with `nvidia-smi` in parallel. State which convention is used.
- **Rented machines** (for example vast.ai containers):
  - Set `-t` to the number of physical cores actually allotted to the container (cgroup quota), not host `nproc`.
  - Pin with `-C/--cpu-strict 1`, or with `numactl --cpunodebind/--membind` on multi-socket hosts.
  - Record RAM channels/speed and PCIe generation.
  - Prefer `--mmap 0` (llama-bench) or `-lm none`/`mlock`, so CPU-resident experts are not demand-paged during timed reps. llama-bench's 1-token warm-up touches only the experts routed for that token, so later reps can page-fault on first-use experts when mmap is on.
- Keep warm-up on and add a discard rep: run llama-bench with `-r N+1` in JSON mode and drop sample 0 if it is an outlier. Alternatively, run one untimed `-n 128` pass first.

### Gaps
- No official llama.cpp documentation on "recommended threads for hybrid MoE decode" was found. Best practice (physical cores, avoiding E-cores and SMT) comes from community reports, and the only data point found (P-cores only: 12.23 versus 12.72 t/s) is confounded by a different offload config.
- No primary source was found quantifying mmap-induced variance in llama-bench reps for MoE.

## Q6. Recommended repetition counts and variance reporting

### Takeaway
llama.cpp's own tooling defaults to 5 repetitions and reports mean ± standard deviation of per-rep t/s, with raw per-rep samples in JSON. No stronger llama.cpp-specific statistical guidance was found. The standard academic reference (Hoefler & Belli, SC'15) asks for confidence intervals rather than bare means, no assumed normality, the harmonic mean for rates, statistically sound comparisons, and full documentation of the setup.

### Cited Findings
- `llama-bench`: `-r, --repetitions <n>` default 5; results "averaged" with "standard deviation"; JSON includes individual repetitions — [llama-bench README](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md).
- `avg_ts` is the arithmetic mean of per-rep rates and `stdev_ts` their standard deviation; `samples_ns`/`samples_ts` are output per rep — [llama-bench.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/llama-bench.cpp).
- `scripts/compare-llama-bench.py` tracks `avg_ts`, `stddev_ts` and `n_cpu_moe` among its fields (local clone of [llama.cpp master](https://github.com/ggml-org/llama.cpp/blob/master/scripts/compare-llama-bench.py)).
- The llama.cpp CONTRIBUTING guide asks contributors to "Verify that the perplexity and the performance are not affected negatively by your changes (use `llama-perplexity` and `llama-bench`)", but gives no repetition count — [CONTRIBUTING.md](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md).
- The llm-tracker hardware-review cheat sheet also treats llama-bench's `-p 512 -n 128 -r 5` as the standard and gives no variance guidance — [LLM Inference Benchmarking Cheat-Sheet](https://llm-tracker.info/howto/LLM-Inference-Benchmarking-Cheat%E2%80%91Sheet-for-Hardware-Reviewers).
- Hoefler & Belli, "Scientific Benchmarking of Parallel Computing Systems: Twelve ways to tell the masses when reporting performance results" (SC'15):
  - Rule 3: "Use the arithmetic mean only for summarizing costs. Use the harmonic mean for summarizing rates."
  - Rule 5: "For nondeterministic data, report confidence intervals of the measurement."
  - Rule 6: "Do not assume normality … without diagnostic checking."
  - Rule 7: compare "using non-overlapping confidence intervals or ANOVA."
  - Rule 9: "Document all varying factors … and the complete experimental setup."
  - Rule 11: "show upper performance bounds."
  - Source: [Hoefler & Belli, SC'15 (PDF)](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf).

### Inferences
- **Recommended protocol** for each (model, machine, config) cell:
  - Run at least 3 independent process launches × `-r 10` (plus one discarded warm rep). This captures between-launch variance (memory placement, thread placement, noisy neighbours on rented hosts), which within-process reps miss.
  - Report the median tg and a 95% bootstrap confidence interval over all reps, together with the coefficient of variation.
  - Summarize rates as the harmonic mean (equivalently total tokens ÷ total time) rather than llama-bench's arithmetic mean of rates. The two differ only slightly when CV is small, but the harmonic mean is the correct one per Rule 3.
  - Save the raw `samples_ns` via `-o jsonl`.
  - Space tests with `--delay` to limit thermal and turbo drift.
- Decide "llama.cpp baseline reaches X" claims against a paper's number by confidence-interval overlap. Treat differences below about 5% as ties, given the CUDA-graph-sized (4–10%) version effects above.
- Record in every result: llama.cpp build tag and commit (llama-bench prints `build: <n> (<commit>)`); CUDA/driver; GPU and its PCIe link; CPU model, cores used and `-t`; DRAM type, channels and speed; `numactl`/NUMA mode; load mode (mmap); FA; KV types; `-ub/-b`; `-ncmoe`/`-ot`; context/depth; and the GGUF file hash and quant. Per Hoefler Rule 11, include the bandwidth roofline bound (bytes per token ÷ bandwidth) as the upper bound.

### Gaps
- No community or academic source was found that gives a specific repetition count for LLM decode microbenchmarks beyond llama-bench's default of 5. The ≥3×10 recommendation above is an inference, not a cited standard.
- No published estimate of run-to-run (between-launch) variance for llama.cpp hybrid MoE decode on cloud instances was found. The protocol should measure it in a pilot.
