# Qwen3-30B-A3B BF16 race recipe (RTX 5090 + Ryzen 9 9950X, 29 Sep 2026)

Sources: KTransformers tag `v0.7.1` (0dce4c9; scratchpad `kt071/`), its SGLang submodule (kvcache-ai/sglang 3424f35; `sglang-kt/`), FreeToken 0d652e7 (`ft/`), our llama.cpp branch 70eb7ba (identical to `jobs/ec2/llama.cpp-expert-cache-4da6337.patch`), PyPI metadata/wheel listings, HF listings. Paths below are relative to those trees.

## 1. Model sources

- **HF safetensors: `Qwen/Qwen3-30B-A3B`** (16 shards, 61.07 GB). It is the checkpoint named in both FreeToken `docs/models.md:15` and the kt-kernel README's Qwen3 example (`kt-kernel/README.md:347,380`). The config gives `Qwen3MoeForCausalLM`, 48 layers, 128 experts, top-8, `norm_topk_prob: true`, hidden size 2048, `moe_intermediate_size` 768 and BF16. `-Instruct-2507` has the same architecture but is non-thinking only. I did not pick it because it is not named in either system's docs.
- **Thinking.** This is the hybrid model, and its template defaults to `enable_thinking=true`. Keep thinking on: AIME prompts then never reach EOS inside 256 tokens. Send the same body to all four servers: `temperature 0, max_tokens 256, ignore_eos true, stream true, stream_options {include_usage:true}`. All four accept `ignore_eos` and `chat_template_kwargs` (FreeToken `server/api_models.py:86`; SGLang `openai/protocol.py:290`; llama-server `server-schema.cpp:475`).
  - FreeToken (`--reasoning-parser auto`) and llama-server (`--reasoning-format auto`) return thinking in `reasoning_content`, and SGLang returns it in `content`. Count both, as `bs1_client.py` already does, and check `usage.completion_tokens == 256`.
  - Assert that `usage.prompt_tokens` is identical across the four servers for every prompt.
- **GGUF: convert it ourselves** from the same HF snapshot:
  ```
  python convert_hf_to_gguf.py /models/Qwen3-30B-A3B --outtype bf16 --outfile /models/Qwen3-30B-A3B-BF16.gguf
  ```
  This gives a single file of about 61.1 GB and needs no second 61 GB download. The weights are byte-identical to the HF checkpoint and the template is copied verbatim. Run it with `--jinja`.
  - Fallback: `unsloth/Qwen3-30B-A3B-GGUF`, `BF16/Qwen3-30B-A3B-BF16-0000{1,2}-of-00002.gguf`, a 2-way split of 49.69 + 11.40 GB (pass part 1 to `-m`). Its embedded template is rewritten (4905 vs 4168 bytes; the diff replaces `messages[::-1]`). In that case add `--chat-template-file` with the HF template.
  - `ggml-org/Qwen3-30B-A3B-GGUF` has only F16, not BF16, so do not use it.
  - llama.cpp's Qwen3MoE loader uses separate `ffn_{gate,up,down}_exps` tensors (`src/models/qwen3moe.cpp:52-54`).

## 2. Equal expert memory

One BF16 expert is 3·2048·768·2 = 9,437,184 B (9.44 MB, 9.0 MiB). One layer holds 1.208 GB, and all 6144 experts hold 57.98 GB. The non-expert weights come to 3.08 GB.

| Budget | llama.cpp `--n-cpu-moe` (GPU layers) | ours `LLAMA_EC_SLOTS` C | FreeToken `--moe-cache-rate` (slots) | KT `--kt-num-gpu-experts` (per layer) | Expert bytes on GPU |
|---|---|---|---|---|---|
| 12.5 % | 42 (6) | 16 | 0.125 (768) | 16 | 7.25 GB / 6.75 GiB |
| 25 % | 36 (12) | 32 | 0.25 (1536) | 32 | 14.50 GB / 13.5 GiB |
| 50 % | 24 (24) | 64 | 0.5 (3072) | 64 | 28.99 GB / 27.0 GiB |
| fallback 43.75 % | 27 (21) | 56 | 0.4375 (2688) | 56 | 25.37 GB / 23.6 GiB |

- FreeToken computes `ceil(num_moe_layers·num_experts·rate)` slots in one pooled LRU (`engine/engine.py:1563-1565`).
- KT's knob **is per layer**: it is multiplied by the MoE layer count (`sglang-kt …/kt_ep_wrapper.py:4522-4528`, help text at `server_args.py:5284`). `--kt-gpu-experts-ratio r` is equivalent and overrides it (`:4507`). The `uniform` placement (the default) pins expert IDs 0…N−1 in every layer statically (`kt_ep_wrapper.py:4327-4366`).
- **The 50 % point probably does not fit.** Weights alone are 29.87 GiB of the 5090's ~31.8 GiB. SGLang would need `--mem-fraction-static ≥0.95` and FreeToken `--memory-ratio ≈0.97`, so an OOM is likely. If any engine OOMs, run the 43.75 % row for **all four** engines; it is the largest multiple of 3 GPU layers below 24, which keeps C an integer.
- Use the same context everywhere: `-c 2048`, `--num-tokens 2048`, `--context-length 2048 --max-total-tokens 2048`. Record the NVML peak.

## 3. KTransformers (kt-kernel + sglang-kt)

**Version identity.** Git tag `v0.7.1` has `version.py = "0.7.0.post4"`. There is no 0.7.1 on PyPI. The matching PyPI release is `kt-kernel`/`sglang-kt`/`ktransformers` **0.7.0.post4** (13–14 Sep).
- The release workflow builds it with torch 2.9.1+cu128, nvcc 12.8, `CPUINFER_CUDA_ARCHS: '80;86;89;90;120'` and all CPU variants (`.github/workflows/release-four-main.yml:147,153,179-185`).
- sgl-kernel ships as a CUDA payload split across the four wheels (`sglang_kt_sgl_kernel_payload`, `sgl_kernel_kt_payload_core` inside kt-kernel). The fork's sgl-kernel adds `sm_120a` for CUDA ≥12.8 (`sgl-kernel/CMakeLists.txt:218-222`).
- **Do not mix a source-built kt-kernel with the PyPI sglang-kt**, because that loses the payload core.
- The README's "SM 80–90" is stale; `doc/en/kt-kernel/GLM-5.3-Flash-Tutorial.md:15,27` documents SM120 via `pip install "ktransformers[sglang]"`.

**Route A (primary, about 10–15 min, roughly 4.5 GB of wheels).** Use a separate venv; FreeToken needs a cu130 torch.
```
apt-get install -y python3.12-venv python3.12-dev build-essential numactl git
python3.12 -m venv /opt/kt && . /opt/kt/bin/activate && pip install -U pip
pip install "ktransformers[sglang]==0.7.0.post4"      # pulls torch==2.9.1 (+cu128 from PyPI), kt-kernel, sglang-kt, transformers-kt
python -c "import kt_kernel as k; print(k.__cpu_variant__, k.__version__)"   # expect avx512_bf16 (_cpu_detect.py:100)
python -c "import torch, sgl_kernel; print(torch.__version__, torch.cuda.get_device_capability())"
```
libnuma and libhwloc are bundled (`kt_kernel.libs/`). python3-dev and gcc are needed for Triton's JIT.

**Route B (fallback, about 30–45 min).**
```
git clone --recursive --branch v0.7.1 https://github.com/kvcache-ai/ktransformers && cd ktransformers
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
apt-get install -y libnuma-dev libhwloc-dev pkg-config && pip install "cmake>=3.31,<4" ninja
SGLANG_KT_VERSION=0.7.0.post4 pip install ./third_party/sglang/python      # pins sgl-kernel==0.3.21 from PyPI
cd kt-kernel && CPUINFER_USE_CUDA=1 CPUINFER_CUDA_ARCHS=120 CPUINFER_PARALLEL=16 ./install.sh build
```
- `install.sh build` auto-detects `-march=native` and enables AVX512 VNNI/BF16/VBMI, with AMX=OFF on Zen 5 (`kt-kernel/install.sh:312-389`).
- The default architecture list is `80;86;89;90` (`kt-kernel/setup.py:762`).
- The upstream Dockerfile uses the same pins (`docker/Dockerfile:207,252`).
- Building with nvcc **13.0** (this container) is untested. Upstream builds with 12.8.

**Launch** (exact, single socket). Pick N from the table.
```
python -m sglang.launch_server --host 127.0.0.1 --port 30000 \
  --model-path /models/Qwen3-30B-A3B --kt-weight-path /models/Qwen3-30B-A3B \
  --kt-method BF16 --kt-cpuinfer 16 --kt-threadpool-count 1 --kt-numa-nodes 0 \
  --kt-num-gpu-experts N --kt-expert-placement-strategy uniform \
  --kt-max-deferred-experts-per-token 0 \
  --attention-backend triton --sampling-backend pytorch \
  --context-length 2048 --max-total-tokens 2048 --chunked-prefill-size 2048 \
  --max-running-requests 1 --cuda-graph-max-bs 1 --mem-fraction-static 0.85 \
  --tensor-parallel-size 1 --served-model-name qwen3-30b-a3b
```
- `--kt-method` defaults to `AMXINT4`, which is lossy and needs converted weights (`server_args.py:5259`). With `BF16`, `--kt-weight-path` is the HF directory. The loader's deepseek-style key format matches Qwen3's `mlp.experts.{i}.gate_proj` (`kt-kernel/python/utils/loader.py:554-601`). On a CPU without AMX the backend is `AMXBF16_MOE`, compiled for AVX-512, with `AVX2BF16_MOE` as the fallback (`utils/amx.py:1009-1015`).
- **Exactness.** Deferral defaults to None, which the wrapper maps to 0, and the last layer is forced to 0 (`kt_ep_wrapper.py:5151-5157`). Pass `0` explicitly anyway. Leave `--kt-gpu-prefill-token-threshold` unset (no layerwise prefill and no extra full-layer VRAM).
- **Threads.** Set `--kt-cpuinfer` to the physical-core count, 16 (README). Also try 12, which is the Docker entrypoint's "cores−4".
- **Attention.** `triton` attention and `pytorch` sampling avoid FlashInfer's sm_120 JIT. Attention costs little at this context length. The optional variant is `flashinfer` with `FLASHINFER_CUDA_ARCH_LIST=12.0a` (`doc/en/DeepSeek-V4-Flash.md:154`).
- **Memory.** Raise `--mem-fraction-static` to about 0.95 for the 50 % row.

**Endpoint and readiness.**
- The API is `POST http://127.0.0.1:30000/v1/chat/completions` with `model: qwen3-30b-a3b`.
- Wait for the log line `The server is fired up and ready to roll!` (`entrypoints/http_server.py:1902`), or poll `GET /v1/models` (`:1487`), then send one warm-up request.
- **Do not poll `/health` during runs.** By default it runs a generation (`environ.py:468`, `http_server.py:483-501`); alternatively set `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false`.

**Blockers and fallbacks.**
1. **Single-NUMA segfault (#1754).** `--kt-threadpool-count` defaults to **2** (`server_args.py:5270`). The numa map becomes `[0,1]` (`kt-kernel/python/experts_base.py:324-339`), so `set_to_numa(1)` runs `numa_bitmask_setbit` on a 1-node mask (`cpu_backend/worker_pool.h:28-32`, `worker_pool.cpp:430`). Use `--kt-threadpool-count 1 --kt-numa-nodes 0`, and first check `lscpu | grep "NUMA node(s)"` shows 1.
2. **Docker without `CAP_SYS_NICE`.** `set_mempolicy`/`mbind` fail, but the code only calls `perror` and continues (`worker_pool.h:48-51`). This is harmless on one node. KT's own Docker docs add `--cap-add SYS_NICE`.
3. **OOM at 50 %.** Fall back to the 43.75 % row.
4. **CPU variant.** If the `avx512_bf16` variant misbehaves, set `KT_KERNEL_CPU_VARIANT=avx2` (AVX2 BF16) and label the run as such.
5. **Route A fails at import or on "no kernel image".** Switch to Route B.
6. **Optional stronger KT variant.** `--kt-expert-placement-strategy frequency --init-expert-location <stats.pt>` needs recorded routing statistics. `--kt-enable-dynamic-expert-update` acts only in layerwise prefill, so it does not affect decode.

**Time (unmeasured estimates).** Route A 10–15 min, Route B 30–45 min; each budget restart loads 61 GB of safetensors, about 3–6 min.

## 4. FreeToken (0d652e7)

- **Support.** Qwen3-MoE is listed with `Qwen/Qwen3-30B-A3B` (`docs/models.md:15`), with an in-tree `models/qwen3_moe/`, and `norm_topk_prob` is honored (`models/qwen3_moe/config.py:48`).
- **CPU executor.** It serves `bf16` banks (`moe/cpu_executor.py:72`, ISA chain AVX-512-BF16 → … → scalar), and SiLU is an allowed CPU activation (`engine/engine.py:1287-1290`). Hybrid therefore works for BF16 Qwen3.
- **Flags.** `--moe-backend` is a deprecated alias of `--moe-strategy` (`server/args.py:587-603`). The default port is 1919 and the default `--memory-ratio` 0.9.
  ```
  ft bench bw --dtype bf16      # once per GPU; hybrid's --moe-hybrid-max-fetch -1 needs this profile, else fetches 1
  ft serve --model /models/Qwen3-30B-A3B --served-model-name qwen3-30b-a3b --port 1919 \
    --moe-strategy offload|hybrid --moe-cache-rate r --num-tokens 2048 --cuda-graph-max-bs 1 [--memory-ratio 0.97 at 50 %]
  ```
- **Prefill overlap** needs at least 2·128 slots, which every budget satisfies (`args.py:728-740`).
- **`cpu` strategy.** It forces 256 slots (4.2 %) regardless of rate (`engine.py:1719-1728`), so it cannot be run at equal memory.
- **Isolation.** Use its own venv (CUDA 13 torch).

## 5. Our patch: Qwen3 routing and BF16 helpers

**Qwen3 routing is supported.**
- The cache path is entered only after the stock gating code: softmax (`src/llama-graph.cpp:2067`), top-k (`:2135`), weight gather (`:2149`) and `norm_w` normalisation (`:2160`).
- `build_moe_ffn_ec` receives only `selected_experts` (`:2196`), and the stock code applies the weights afterwards (`:2354`).
- Its eligibility checks (`:2406-2422`) are: SILU or SWIGLU_OAI, no swiglu clamp, gate tensor present, and `n_expert_used == n_used`. Qwen3MoE passes all of them: `LLM_FFN_SILU`, `norm_w=true`, `SOFTMAX` (`src/models/qwen3moe.cpp:137-151`).
- The PREFETCH/LLC router copy needs F32 or a `to_float` router type, and BF16 qualifies (`src/llama-expert-cache.cpp:372-373`).

**BF16 CPU helpers are supported by construction.**
- The helper takes `vec_dot` and `vec_dot_type` from `ggml_get_type_traits_cpu(type)` for gate, up and down (`llama-expert-cache.cpp:1283-1288`, `1364-1365`, `1415`). It quantizes inputs and activations through that type's `from_float` (`:1200-1211`).
- For BF16 that means `ggml_cpu_fp32_to_bf16` and `ggml_vec_dot_bf16` (`ggml/src/ggml-cpu/ggml-cpu.c:395-399`). With a native build on Zen 5 this uses the AVX512-BF16 path (`vec.cpp:148`), the same numerics as stock ggml CPU experts.
- The AVX-512 kernel is MXFP4-only and is taken only when all three tensors are MXFP4 (`:1289-1290`); otherwise the path is generic. The activation is `silu(g)·up` (`:1378-1380`).
- The mailbox needs separate `w_gate`/`w_up` (`:1091-1094`, `:131-134`), which Qwen3 has, and needs no scaled experts (`:105-107`).
- On the GPU side, BF16 is covered by the guarded `mul_mat_vec_f` for id −1, and the op tests include BF16 EC cases (`port_notes.md` items 2 and 5). FETCH copies are byte copies (`ggml-cuda/ec.cu:169-184`).

**Gaps.**
- The BF16 **helper** path has never run end to end. T03 covered Qwen3-MoE F32 and Q8_0 only. Before the GPU hour, run T03 on a tiny Qwen3-MoE **BF16** model, then a top-1-agreement check against stock on the real model.
- Helpers are off unless `LLAMA_EC_MAILBOX=1` (`:97`).

**Commands** (same flags as job 074):
```
(a) llama-server -m Qwen3-30B-A3B-BF16.gguf -ngl 99 --n-cpu-moe K -fa on -lm none -t 16 -np 1 -c 2048 --no-webui --jinja --port P
(b) env LLAMA_EC_SLOTS=C LLAMA_EC_MAILBOX=1 LLAMA_EC_POLICY=dfa LLAMA_EC_KAPPA=1 LLAMA_EC_TDEC=1 LLAMA_EC_HELPERS=14 \
    llama-server -m … -ngl 99 -fa on -lm none -t 16 -np 1 -c 2048 --no-webui --jinja \
    -ot '\.ffn_(up|down|gate|gate_up)_exps=CUDA_Host' --port P      # optional FETCH variant: LLAMA_EC_FETCH=0,1,1,2,3
```
(b) pins about 58 GB of host memory, and FreeToken pins a similar amount, so run the systems one at a time on the 126 GB host.

## Not determined

- Whether the PyPI 0.7.0.post4 CUDA pieces actually load on this driver and 5090 (the workflow says sm_120 is built; I did not run them).
- Whether any engine fits the 50 % row.
- KT batch-1 decode speed on a single-socket Zen 5.
- Whether a Route B build works with nvcc 13.0.
- Whether FreeToken's stream-memop handshake survives the container.
- How fast the BF16 helper path runs; `ggml_vec_dot_bf16` is called per row.
- How the four engines' non-expert VRAM (KV pool, CUDA graphs, workspaces) differs. It has to be measured with NVML.
