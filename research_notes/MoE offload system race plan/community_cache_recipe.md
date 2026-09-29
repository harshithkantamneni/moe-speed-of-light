# Community expert cache for the race: recipe (29 Sep 2026)

Short labels: **L** = leloch/llama.cpp `moe-cache-v2-pr` @ `e3096b046bb809f7f80bc47801f6579aed1cbc60`. **P** = ggml-org/llama.cpp PR #27861 head `bccbacdb8945` (still the head when re-fetched today). **E6** = everett6/moe-prefetch @ `f9f86d0` (2026-09-19). All code was read from git clones in the scratchpad. Thread content comes from WebFetch summaries.

## 1. Choice: leloch v2 (L)

| Candidate | gpt-oss-120b MXFP4 on 1 GPU via llama-server? |
|---|---|
| **L** | **Yes.** It hooks the CPU `MUL_MAT_ID` node, so gpt-oss's biases and activation run unchanged (L:`ggml/src/ggml-cpu/ggml-cpu.c:1651-1813`). MXFP4 is a supported cache type (L:`ggml/src/ggml-cuda/moe-cache.cu:677-704`; docs `docs/backend/CUDA-MOE-CACHE.md:65-67`). `on` and fixed-N modes need only one device with cc ≥ 7.0 (docs:11-16). The flag is parsed by the common parser, so llama-server accepts it (L:`common/arg.cpp:2690-2719`). |
| P + E6 `0006` | **No: a silent no-op on gpt-oss.** P builds the cache chain only when there are no expert biases and `type_op == LLM_FFN_SILU` (P:`src/llama-graph.cpp:2114-2117`). gpt-oss passes biases and `LLM_FFN_SWIGLU_OAI_MOE` (P:`src/models/openai-moe.cpp:134-144`). `0006` does not touch `llama-graph.cpp`. The cache still allocates VRAM, though. The routing observer also fires on every CPU `ffn_gate_exps` op (P:`ggml-cpu.c:1658`), so uploads run and hit counters rise while the graph never reads a slot. |
| #26563 (`3897637`) | Closed and abandoned. It is untested on gpt-oss (PR page). It is the fallback (§4). |

**How misses are handled.** L computes miss rows on the CPU in the same op. Hits run on the GPU, then there is a D2H copy and a `cudaStreamSynchronize` for each node (L:`moe-cache.cu:2571-2580`). Misses queue asynchronous demand fills: admission after the 2nd miss when the pool is partial, at most 8 fills per node, LRU eviction, and 8 fresh misses before a filled entry can replace another (docs:52-61). L never fetches an expert synchronously. P works the same way (CPU misses, a worker uploads, the new mapping is published at the next step).

## 2. Build (inside `nvidia/cuda:13.0.3-devel-ubuntu24.04`, on the race host)

```bash
apt-get update && apt-get install -y git cmake ninja-build build-essential
git clone --filter=blob:limit=2m -b moe-cache-v2-pr https://github.com/leloch/llama.cpp llama-leloch
cd llama-leloch && git checkout e3096b046bb809f7f80bc47801f6579aed1cbc60   # 29 commits on upstream 15586e2d (2026-08-06)
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=ON   # 120 -> 120a by CMake
cmake --build build -j"$(nproc)" --target llama-server llama-bench test-moe-cache test-backend-ops
CUDA_VISIBLE_DEVICES=0 ./build/bin/test-moe-cache            # model-free hit/fallback test (docs:214-216)
./build/bin/test-backend-ops -o MUL_MAT_ID -b CUDA0
```

- No patches are needed.
- `GGML_NATIVE` is on by default, so build on the Zen 5 host.
- Checked here: with nvcc **12.8.93**, `moe-cache.cu`, `mmvq.cu`, `ggml-cpu.c`, `ggml-backend.cpp`, `llama-context.cpp`, `arg.cpp` and `server-context.cpp` all compile for `120a`. CUDA 13.0 and a full link were not tested.

## 3. Launch

**Budget.** The per-device budget is in MiB and covers the slabs plus about 3.2 MiB of dispatch scratch (docs:38-42).
- There is one pool per (expert bytes, type) (L:`moe-cache.cu:1162-1170`), and every gpt-oss projection has the same shape. The result is one **pooled** LRU of 108·E projection slots across all 36 layers × 3 projections, so "E per layer" is only an average. Score this entrant against the pooled bound.
- One projection is 4,406,400 B. That is `data/gguf_bytes.json` (13,253,760 B/expert) minus 34,560 B of F32 biases, divided by 3.
- The budget in MiB is ceil((108·E·4,406,400 + 3,393,792)/2^20):

| Stock | E/layer | slots | `--moe-cache` N (MiB) |
|---|---|---|---|
| `-ncmoe 32` | 14 | 1512 | **6358** |
| `-ncmoe 27` | 32 | 3456 | **14527** |
| `-ncmoe 22` | 51 | 5508 | **23150** |

The stock `-ncmoe 22` point is 49.8 experts per layer. The table uses 51, as you specified.

```bash
N=6358   # or 14527 / 23150
CUDA_VISIBLE_DEVICES=0 GGML_CUDA_MOE_CACHE_STATS=10800 \
./build/bin/llama-server -m gpt-oss-120b-MXFP4.gguf \
  -ngl 99 --cpu-moe --moe-cache $N --no-repack --fit off \
  -c 4096 -np 1 -fa on -ub 512 -b 2048 -t 16 --load-mode <same as stock arm> \
  -lv 4 --host 127.0.0.1 --port 8080 2>&1 | tee srv_ec_$N.log
```

- `--moe-cache N` turns repacking off by itself (L:`arg.cpp:835-838`).
- `STATS=10800` prints stats about every 100 tokens, because there are 108 nodes per token.

**Proof that the cache is active.** Grep in this order. The pool is created lazily at the first decode, so send warm-up requests before measuring.
1. `MoE cache requested=on resolved=on` (L:`src/llama-context.cpp:688`). If it says `resolved=off`, the cache is a no-op.
2. `[moe-cache] CUDA0 capacity: cap=N MiB granted=G MiB`. G must equal N. It is clipped to free VRAM − 3072 MiB reserve (L:`moe-cache.cu:1195`, `449-475`).
3. `[moe-cache] CUDA0 pool[0]: type=mxfp4 expert=4303 KiB slots=S entries=13824 … total=T MiB`. S must equal the slot count in the table (L:`moe-cache.cu:1295`).
4. `[moe-cache] enabled: first pool allocated on CUDA0` (L:`moe-cache.cu:1303`). The docs say that if this line is absent, "the cache did not become active" (docs:48).
5. `[moe-cache] CUDA0 hits=h/n (…%) … filled=… dispatch-fail=0 collect-fail=0` (L:`moe-cache.cu:1454`). This line needs nonzero hits and zero failures.

**Take the equal-VRAM reading from nvidia-smi after warm-up**, because the slabs are allocated outside ggml's allocator.

**Same-binary control arms** (docs:104-112):
- `--moe-cache off` with `-ncmoe 32/27/22` and repacking on. This is stock placement on the same August base.
- `--moe-cache off --no-repack --cpu-moe`. This is the canonical floor with no cache.

## 4. Risks and fallbacks

**Risks**
- **Old base.** L's base is 2026-08-06, and upstream has since changed L's files by +3048/−1348 lines, so a rebase would not be cheap. Race master `4da6337` too.
- **Slower misses.** Forced no-repack slows CPU misses, because the base has MXFP4 repack kernels that the stock arm uses.
- **No fusion on gpt-oss.** gpt-oss misses L's fused gate/up matcher (L:`ggml-cpu.c:3177-3250`), so each token pays 108 synchronous round trips. The cost is unmeasured.
- **Budget clipping at E=51.** E=51 needs about 26.2 GiB free. If G < N, set `GGML_CUDA_MOE_CACHE_RESERVE_MB=1024` and record it.
- **Output** is not bit-identical (docs:179).
- **No gpt-oss runs** are reported for v2. The MXFP4 evidence is DeepSeek-V4-Flash only.

**Fallback 1: #26563** (`pull/26563/head`, `3897637`).
- `-ehs N` gives N slots **per layer** (`src/llama-expert-hotstore.h:13-28`).
- Its per-matmul masks "lift the no-biases constraint" (`src/llama-expert-tier.h` header). It is decode-only.
- `-ehs -1` silently resolves to 0, so use a manual N. It also has prefill regressions.
- No successor code exists: `miltos22/llama-wackMall`'s head is 2026-08-06.

**Fallback 2: P + 0006, for non-gpt-oss models only.**
- Fetch and patch: `git fetch https://github.com/ggml-org/llama.cpp pull/27861/head:pr27861`, check out `bccbacd`, copy E6 `cpp/moe-predictor.{h,cpp}` into `src/`, then `git apply` E6 `patches/0006-…patch` alone (E6 README "Layout"). `--check` passed.
- 0006 fixes the dry-run latch (P:`src/llama-moecache.cpp:149-155`). It needs `LLAMA_MOE_EARLY_ISSUE=1`; otherwise the default of 2 inserts drains the cache (E6 `docs/UPSTREAM-BUGS.md` §2).
- Flags: `--moe-expert-cache N` (slots per layer) plus `-ncmoe`.
- To check it is active: `MoE expert cache enabled:` (P:317) and `moe-cache: batched uploads ENABLED`.
- A cache too large to allocate only warns (P:243-267), and per E6 that warning is invisible in llama-server.

## Not determined

- L's CUDA 13.0 build, sm_120 runtime and gpt-oss correctness. Nothing was run on a GPU.
- Its decode speed and the free VRAM on the 5090 at the first decode.
- Whether `[moe-cache]` lines appear without `-lv 4`.
- Where the figure of 51 experts per layer comes from.
