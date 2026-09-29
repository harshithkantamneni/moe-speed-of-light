# Pipelined Sharding recipe: gpt-oss-120b MXFP4, RTX 5090, CUDA 13.0 (29 Sep 2026)

Repo `deepshnv/pipeshard-mlsys26-ae`, tag `v2.0.3-mlsys26` = `89393160fbf0`. Line references are to that tag.

**Verified here without a GPU:**
- Builds for sm_120 with nvcc 13.0.88 and gcc 13.3, no `-fpermissive`.
- A tiny MXFP4_MOE GGUF quantized by current master loads; first-token top-8 logprobs match master within ±0.01.
- The 120b chat template renders identically to master, and the `bs1_client.py` body works.

**Not run:** anything with `-pipe-shard`, which needs the GPU.

## 1. Build (16 min here on 2 vCPUs; far less on 16 cores, not measured)
```bash
apt-get update && apt-get install -y --no-install-recommends git cmake ninja-build build-essential
git clone https://github.com/deepshnv/pipeshard-mlsys26-ae /work/pipeshard
cd /work/pipeshard && git checkout --detach v2.0.3-mlsys26
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j16 --target llama-server llama-cli concurrent_profiler gpu_profiler
```
- **`=120` is required on CUDA 13.** Without native detection, the fallback architecture list starts with `50-virtual` (`ggml/src/ggml-cuda/CMakeLists.txt:25-38`), and nvcc 13 rejects it: `Unsupported gpu architecture 'compute_50'` (tested). `120` produces `compute_120,sm_120`. b6097 has no FP4-MMA path, so `120a` is not needed.
- **Libraries** land in `build/bin` with an RPATH, so README's `LD_LIBRARY_PATH` note (`README.md:79-82`) is only needed if binaries move.
- **`main` (`7794a3f`) adds only fix `60dc4f7`** (`src/llama-model.cpp:1950`). Without it, the binary segfaults when no GPU is visible (reproduced). This does not matter on the 5090.

## 2. Profiling (once per machine, GPU idle; reuse for all budgets)
```bash
export GGML_CUDA_PIPELINE_SHARDING=1 GGML_CUDA_REGISTER_HOST=1
mkdir -p /work/ps_prof && cd /work/ps_prof          # both write into CWD
time /work/pipeshard/build/bin/concurrent_profiler --cold --fast --threads 16   # -> concurrent_results.txt
time /work/pipeshard/build/bin/gpu_profiler --fast                               # -> gpu_results.txt
```
- **Flags.**
  - These are the AE scripts' invocations (`paper_results/repro_table4.sh:104-114`). For README's "full" mode, drop `--fast` (`README.md:188-200`).
  - `gpu_profiler` parses only `--fast` and `-ub`, so `--cold` is ignored there (`examples/microbench/gpu_profiler.cpp:492-497`).
  - Outputs are written at `concurrent_profiler.cpp:1312` and `gpu_profiler.cpp:852`.
- **`--threads` must equal the server's `-t`.** CPU entries load only when the thread counts match (`src/llama-benchmark.cpp:89`). The server log must show `Loaded N benchmark entries for 16 threads` with N > 0 (`:114`).
- **Duration.** The paper gives about 15 min for the one-time benchmark (arXiv 2604.26334, §3 "Install Phase"). The source does not say how long `--fast` takes, so budget 20 min.
- **MXFP4 is never profiled.**
  - The profilers only cover F32, F16, Q8_0, Q4_0, Q4_1, Q5_0 and Q2_K (`concurrent_profiler.cpp:67`, `gpu_profiler.cpp:581-587,656-662`).
  - `mxfp4` has no bits-per-weight entry (`llama-benchmark.cpp:406-413`).
  - So expert matmul times fall back to bandwidth-only estimates (`:748-749,1039-1048`). The Q8_0 attention is profiled.

## 3. Server launch
```bash
export GGML_CUDA_PIPELINE_SHARDING=1 GGML_CUDA_REGISTER_HOST=1 CUDA_VISIBLE_DEVICES=0
/work/pipeshard/build/bin/llama-server -m /work/models/gpt-oss-120b-MXFP4.gguf \
  -pipe-shard -mva ${MVA_MIB} \
  --cpu-profile /work/ps_prof/concurrent_results.txt --gpu-profile /work/ps_prof/gpu_results.txt \
  -t 16 -np 1 -c 8448 --jinja --no-webui --host 127.0.0.1 --port {port}
```
- **The environment variable must be set when arguments are parsed.** Otherwise `-pipe-shard` is silently ignored (`common/arg.cpp:1726-1736`), and the stock loader tries to put 59 GiB on the GPU. `-pipe-shard` itself sets `n_gpu_layers=1e6` and flash attention.
- **Do not add these flags:**
  - `-ngl`, or `-ncmoe`/`-ot`.
  - `-fa on`: in b6097 `-fa` takes no value (`arg.cpp:1495-1500`).
  - `-lm none` or `--no-mmap`: the pinned-copy loader calls `ml.mappings.at()` (`src/llama-model.cpp:5750`), and the mappings are empty without mmap (`src/llama-model-loader.cpp:846-847`), so loading fails.
- **Matches stock.** `-c 8448 -np 1 -t 16` is the same as job 076's stock `LS` line (`jobs/076_table_5090@vast.sh:95`, gpu branch).

**Budget semantics**
- **`-mva N` is in MiB** (N×1024², `src/llama-pipeline-executor.cpp:67`).
- **It sizes one device buffer allocated up front** (`:86` → `ggml/src/ggml-alloc.c:780-787`). That buffer holds pinned weights, shard slots, KV and compute scratch (`src/llama-plan-builder.cpp:143-144,187`).
- **So it covers neither the experts alone nor the whole process.** The CUDA context, cuBLAS, memory pools and graphs sit outside it. The fork also creates 128 virtual CUDA devices on GPU 0, each with its own streams and pools (`ggml/src/ggml-cuda/ggml-cuda.cu:312-318`; `ggml.h:218-219`).
- **The planner, not the flag, decides where experts go.**
  - STATIC_ATTN_PRIO pins all attention first, then the output layer, then whole layers (attention plus all three expert tensors) from layer 0 up. The remaining experts run on the CPU (`llama-plan-builder.cpp:344-400`; `llama-pipeline.h:51-60,76-83`).
  - So `-ncmoe k` corresponds to about 36−k pinned layers: 4, 9 and 14 here. Each layer's experts are 3×537.9 = 1613.7 MiB (GGUF header).

**Matching `-ncmoe 32/27/22`.** Match measured totals, using the same `nvidia-smi memory.used` reading the client takes: targets are 9728, 17818 and 25907 MiB.
1. Start at `MVA = target − 1024`: **8704 / 16768 / 24832**.
2. Run warm-up plus one measured request and read `used`.
3. Compute `overhead = used − MVA`, then set `MVA = target − overhead` and repeat once. Accept |used − target| ≤ 128 MiB.
4. Record from the log's `PIPELINE PLAN SUMMARY` (`llama-pipeline-executor.cpp:668-790`): `max_vram_alloc`, the per-tier `current_strategy`, `pinned_weights` and `n_pinned_layers`.

**`-psa` (pinned host memory, GiB).**
- The default is auto: total RAM ÷ 2.5, about 50 GB here (`llama-model.cpp:5643-5662`).
- Expert tensors have the lowest priority (`:5628`), so about 9 GB of them stay mmapped and are registered with `cudaHostRegister` (`ggml-cuda.cu:3153-3160`).
- Keep the default. `-psa 60` pins all 59 GiB.

## 4. Readiness and request quirks
- **`/health`** returns 503 "Loading model" through load and planning (`tools/server/server.cpp:3862-3876`), then 200 `{"status":"ok"}` (`:3904-3908`). `/v1/models` works during loading, and `data[0].id` is the model path (verified).
- **`--jinja` is required.**
  - Current stock llama-server uses jinja by default (`/home/claude/llama.cpp/common/common.h:639`).
  - Without it, the fork falls back to a legacy harmony template with no system header (`src/llama-chat.cpp:713-721`), so prompts would differ.
  - The GGUF's embedded template (md5 `4d4366d7…`, identical to `openai/gpt-oss-120b@b5c939` `chat_template.jinja`) renders under the fork with no chatml fallback (`server.cpp:2052-2060`). The result is identical to master's `/apply-template` output.
- **Streaming output** (default `--reasoning-format auto`, format GPT-OSS, `common/chat.cpp:1313-1325,1813-1815`; `server.cpp:387`). I replayed the server's parse and diff over a harmony stream:
  - The first two tokens arrive as `content` ("<|channel|>", "analysis").
  - The analysis tokens arrive as `reasoning_content`.
  - `<|message|>`, `<|end|>` and the five header tokens `<|start|>assistant<|channel|>final<|message|>` produce no text.
  - The final-channel tokens arrive as `content`.
  - `bs1_client` counts either field and uses `usage.completion_tokens`, so its numbers are unaffected. `<|end|>` is not an end-of-generation token in this fork (`src/llama-vocab.cpp:2366-2391`).
- **Request body.**
  - `ignore_eos`, `max_tokens`, `stream_options.include_usage`, `chat_template_kwargs`, `min_p` and `seed` are accepted. The test returned 256 tokens with `usage` and `timings`.
  - `backend_sampling` is silently ignored because b6097 has no GPU-side sampling. Pipeshard therefore pays the CPU sampler over 201,088 logits on every token, about 1.6 ms/token (`prereg/server_overhead_outcome_074b.md:31`), which stock avoids. Report this handicap.

## 5. Risks and fallbacks
1. **Plan choice for MXFP4 rests on bandwidth-only estimates.** If the batch-1 tier is not STATIC_ATTN_PRIO, also run `-fes 2` and label it as a variant.
2. **VRAM outside the `-mva` buffer is unknown** (the 128 virtual devices). Use the calibration loop above. If the 25.3 GiB point runs out of memory, lower `MVA` by 512.
3. **b6097 is an old base.**
   - CUDA flash attention with attention sinks uses only the vec kernel (`ggml/src/ggml-cuda/fattn.cu:283-291`), so prefill and TTFT are slow.
   - It lacks the later MXFP4 CPU repack and GPU sampling.
   - Label the results "as shipped", next to mainline.
4. **Pinned-buffer allocation can fail.** Allocating the ~50 GB buffer with `cudaMallocHost` falls back to mmap if it fails, which is non-fatal (`llama-model.cpp:5738`). Allow a long load; the client's timeout is 1800 s.
5. **Tier plan switches** (`llama-pipeline-executor.cpp:454-505`) may inflate TTFT; the bs1 metric drops the first token. Only physical GPU 0 is used (`ggml-cuda.cu:115-118`).
6. **Unknown until the first GPU run:** whether `-pipe-shard` runs gpt-oss end to end, profiler duration, load time and overhead. Smoke-test at MVA 8704 with a 16-token request.
