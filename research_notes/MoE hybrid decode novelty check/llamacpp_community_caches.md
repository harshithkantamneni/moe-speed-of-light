# Community dynamic GPU expert caching in llama.cpp and forks (2025 to Sept 2026)

Method note: the GitHub REST API and github.com HTML were blocked for curl in this session, so PR and discussion text comes from WebFetch summaries of the GitHub pages. A small model writes those summaries, so treat the quoted numbers as "as summarised". Where it mattered, I checked mechanism claims against source code. I fetched the code anonymously with git from `refs/pull/27861/head` (bccbacd), `refs/pull/26563/head` (3897637), leloch's `moe-cache-v2-pr` branch (e3096b0) and upstream `master` (d7fb90e, 2026-09-27). Upstream dates come from `git log` on llama.cpp master. Where code and the summary text disagree, I say so.

## Q1. Inventory: what each effort does (mechanism, status, numbers, problems, maintainer feedback)

### Takeaway
There are at least 10 distinct community expert-cache efforts around llama.cpp, all dated March to Sept 2026. None is merged upstream. They fall into three mechanism families:
- **(A) Dual-chain hit/miss split:** hits run on GPU cache slots and misses run on CPU. Examples are #27861, #26563, leloch's #24524/#24528 and moe-autopilot.
- **(B) Scheduler-copy cache:** all expert matmuls run on GPU, and misses are copied over PCIe through ggml-backend's "copy only used experts" path. Examples are Lidenburg, #23170, #21609/#21614, #20757, moe-l2 and FATE.
- **(C) Disk streaming into a device cache:** #25294.

Closest to the target patch's admission policy: PR #26563 uses an exponentially decayed heat score (default decay 0.999 per update). It swaps an expert in only when the cold expert's score is at least 1.3 times the incumbent's (default `--expert-hyst 1.3`). That is decayed-frequency admission with a (multiplicative) margin.

### Cited Findings

**PR #27861, "llama: GPU-resident LRU cache for host-offloaded MoE expert weights" (csantiago78)**
- Opened 2026-08-28. Draft. Review from code owners CISC and ggerganov is still pending. — [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861)
- **Mechanism (from code, `src/llama-moecache.h`):**
  - "Mechanism (no custom kernels)". Each cached layer gets companion tensors up_c/gate_c/down_c `[ne0, ne1, n_slots+1]` in the device buffer. The last slot is permanently zero.
  - An I32 table maps expert id to slot. A device copy remaps ids for the "cache-side mul_mat_id chain". A host copy is read by the CPU mul_mat_id via `src[3]` to "SKIP cached ids, zeroing their dst rows". The two down-projection outputs are summed.
  - `llama_moe_cache_step()` runs at the end of `llama_context::decode()` and "performs throttled LRU updates: at most LLAMA_MOE_CACHE_INSERTS expert uploads per layer per step via ggml_backend_tensor_set."
  - Source: [llama-moecache.h @ bccbacd](https://github.com/ggml-org/llama.cpp/blob/bccbacdb8945680f1cfc7e6bffd1e59014705750/src/llama-moecache.h)
- **Code details (`src/llama-moecache.cpp`):**
  - A `std::thread` worker uploads slices with `ggml_backend_tensor_set`. New mappings are "only published at a later step()".
  - Eviction picks "an empty non-in-flight slot if any, else the LRU non-in-flight slot". The default is `max_inserts = 2`.
  - The CPU-side routing observer skips batches with `n_tokens > 4`. The cache graph is built only when `n_tokens == 1` (`src/llama-graph.cpp`).
  - Source: [llama-moecache.cpp @ bccbacd](https://github.com/ggml-org/llama.cpp/blob/bccbacdb8945680f1cfc7e6bffd1e59014705750/src/llama-moecache.cpp)
- **Implication (inferred from the design, not stated in the PR):** hits and misses are two branches of the normal ggml graph. CPU/GPU hand-off therefore goes through ordinary `ggml_backend_sched` splits, and the CPU split runs for every cached layer.
- **Benchmarks, as summarised from the PR thread:** — [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861)
  - 2x RTX 3090 + dual Xeon, "Qwen3.8-Flash-Next" UD-Q4_K_XL: 18.4 → 24.2 tok/s (48 slots), hit ~67–81%, +31%.
  - Single R9700 DDR5, same model: 12.8 → 14.8 tok/s, +15%.
  - RTX PRO 4500 Blackwell, same model: 16.83 → 41.31 tok/s at 96 slots, "+145% (tuned)".
  - Radeon RX 7600 8GB Vulkan, GigaChat-20B-A3B Q4_K_M: 17.1 → 20.4–20.6 tok/s (26 slots), hit 60.4%.
  - **Radeon RX 7600 8GB Vulkan, Qwen3-30B-A3B Q4_K_M: 14.4 → 16.5 tok/s (32 slots), hit 74.7%, ~+14.6%.**
  - sirfyyn (Blackwell) measured hit rates of 69.3–98.5% at 32–512 slots.
  - Source: [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861)
- **Known problems raised in the thread:**
  - The cache "will basically never trigger if there's a draft model" (sdroege).
  - Different prefill and decode graphs cause reallocations: "a realloc every 250-300 decode steps".
  - If uploads fall behind decode, all slots may be in flight.
  - Cache VRAM is not counted by `--fit`.
  - Only the separate gate/up + SILU MoE variant is supported.
  - Source: [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861)

**PR #26563, "Expert caching that greatly increases performance (Self-contained, off by default)" (miltos22), flag `-ehs/--expert-hot-s`**
- **Status:** open. The earliest PR-branch commits by the author are dated 2026-08-03/04 and the head is 2026-08-06 (from git). Green-Sky asked the author to "Mark this pr as draft". The author said he was redesigning "about half of the entire system" and planned a clean new PR. — [PR #26563](https://github.com/ggml-org/llama.cpp/pull/26563)
- **Mechanism (from code, `src/llama-expert-tier.h`):** the hook "remaps real expert ids through hot_lut/cold_lut, runs two mul_mat_id ops (hot on GPU slots, cold on the full CPU tensor), scales and masks each per routed expert, sums them". It uses "Pure stock ggml ops, no custom kernel". — [llama-expert-tier.h @ 3897637](https://github.com/ggml-org/llama.cpp/blob/38976372452722cf5e6846429aa33ee892c9d89a/src/llama-expert-tier.h)
- **Policy (from code):**
  - The heatmap applies `heat[i] *= decay_rate`. The `--expert-heat-decay` default is 0.999 "multiplicative decay per update".
  - `--expert-hyst` has default 1.3: "hysteresis ratio: only swap when cold >= hyst x hot".
  - `--expert-dwell` has default 0: "minimum updates a resident slot must keep before a swap".
  - The victim is the "coldest resident that has dwelled enough AND is beaten by hyst * this cold expert".
  - Resync is cadence-gated (`sync_period` tokens). `multi_slot` freezes the store.
  - Sources: [common.h @ 3897637](https://github.com/ggml-org/llama.cpp/blob/38976372452722cf5e6846429aa33ee892c9d89a/common/common.h); [llama-expert-hotstore.h](https://github.com/ggml-org/llama.cpp/blob/38976372452722cf5e6846429aa33ee892c9d89a/src/llama-expert-hotstore.h)
- **Copies (from code):** slot swaps use synchronous `ggml_backend_tensor_set` in `llama-expert-hotstore.cpp`. They are called from `llama-context.cpp` after the graph runs (`update_from_graph` then `maybe_resync`). No custom copy kernel or separate stream is involved. — [llama-expert-hotstore.cpp @ 3897637](https://github.com/ggml-org/llama.cpp/blob/38976372452722cf5e6846429aa33ee892c9d89a/src/llama-expert-hotstore.cpp)
- **Benchmarks, as summarised; no --n-cpu-moe value is given for "stock":** — [PR #26563](https://github.com/ggml-org/llama.cpp/pull/26563)
  - Qwen3.6-35B-A3B Q2_M: 33.25 → 57.2 tok/s (1.72x).
  - Qwen3.6-35B-A3B Q5_K_P: 17.34 → 35.93 (2.07x).
  - Qwen3.5-122B-A10B IQ2_M: 4.72 → 3.97 (0.84x) with dwell=0, and 5.71 (1.21x) with dwell=16.
  - Laguna-S-2.1 IQ3_XXS: 1.35 → 1.79 (1.33x).
  - RTX 2000 laptop (buha): 31.51 → 26.46 tok/s with `-ehs -1`, a 15% loss.
  - DeepSeek-V4-Flash on 4x3090 (blakemartz): 12.77 vs 13.45 t/s static-pin baseline. "static pin wins", and on 256 balanced experts there is an "11% theoretical hit rate".
  - Prefill with heatmap updates (mobilinkd): 103.78 → 86.60 t/s, recovered to 101.75 with PP updates off.
  - Source: [PR #26563](https://github.com/ggml-org/llama.cpp/pull/26563)
- **Other problems and feedback:**
  - Multi-GPU: `-ehs -1` "silently resolves to 0" with `-ngl 999`/`--n-cpu-moe`, and the store "only ever allocates on card 0".
  - VC++ atomics break the Windows build.
  - voidpush recommended splitting the work into modular PRs. iiLaurens suggested offline calibration instead of a live heatmap.
  - No core-maintainer technical review appears in the summary.
  - Source: [PR #26563](https://github.com/ggml-org/llama.cpp/pull/26563)

**leloch: PR #24524 "CUDA MoE expert cache - adaptive VRAM caching of CPU-resident experts", converted to Discussion #24528 "RFC: MoE expert cache ... with hybrid hit/miss execution"**
- **Status:** the RFC was posted 2026-06-12. The PR was closed and converted to a discussion. The discussion says "closed as too large", while the PR-page summary cites maintainer feedback and AI-generated-code policy. A v2 branch was posted 2026-08-06. It is not merged, and no core-maintainer comments appear in the summary. — [Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528); [PR #24524](https://github.com/ggml-org/llama.cpp/pull/24524)
- **Mechanism (RFC):** the CPU `mul_mat_id` stays on CPU. "thread 0 dispatches one batched matvec over the cached (hit) rows on the GPU while the other threads compute the miss rows". Fill happens during decode only. Hooks sit in `ggml-cpu.c`/`ops.cpp` and `ggml-backend.cpp`, with about 1,700 lines of new CUDA. Hot-set persistence lives in `~/.cache/llama.cpp/`. — [Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)
- **Mechanism (v2 code, `ggml/src/ggml-cuda/moe-cache.cu`):**
  - Dispatch does `cudaMemcpyAsync` (input upload), quantize and matvec on a cache-owned `compute_stream`.
  - Collect does `cudaMemcpyAsync` D2H of the output and then `cudaStreamSynchronize(device.compute_stream)`, so the host blocks per MUL_MAT_ID node.
  - Fills run on a separate `cudaStreamNonBlocking` stream at the least priority. They `memcpy` into a `cudaMallocHost` staging buffer, then call `cudaMemcpyAsync` + `cudaStreamSynchronize`. `serial_fill` defaults to on when there are fewer than 2 devices or compute capability is below Ampere.
  - Admission is "1-complete/2-partial/8-replace" (`admit_after = 2`, `readmit_after = 8`, env `GGML_CUDA_MOE_CACHE_ADMIT_AFTER`), with LRU eviction.
  - In auto mode a session returns nullptr when `devices.size() < 2`.
  - Source: [moe-cache.cu @ leloch e3096b0](https://github.com/leloch/llama.cpp/blob/e3096b046bb809f7f80bc47801f6579aed1cbc60/ggml/src/ggml-cuda/moe-cache.cu)
- **Benchmarks (RFC; 4x RTX 3090, EPYC 7R13, 8-ch DDR4-3200, tg300):** — [Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)
  - GLM-5.1 754B IQ2_M: 13.96 → 17.49 (+25%).
  - Qwen3.5 397B Q3_K_XL: 28.18 → 30.25 (+7%).
  - 13 models / 9 architectures, forced `-ngl 99 -ncmoe 99`: "+10%…+57%, 16/16 ≥ parity".
- **Regression:** GTX 1080 Ti (batot1) went 19.32 → 13.25 tok/s at 4096 MB (−31%). v2 fences it by requiring cc ≥ 8.0 in auto mode. — [Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)
- **v2 ablation (Qwen3.6-35B, sm86):** off 81.55; cache-only 88.12; full 99.26; "minimal core" 100.21 tok/s. — [Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)
- **Composition with speculative decoding:** noonghunna measured 16.7 (base), 24.4 with cache alone (+46%), and 46.8 with CPU drafter + cache. With a GPU drafter the cache "doesn't engage". — [Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)

**Competing forks named in #24528**
- **Lidenburg/llama.cpp, branch `moe-expert-caching`:**
  - Three tiers: VRAM, pinned RAM (`cudaHostRegister`) and disk (`io_uring` + `O_DIRECT`). "when all active experts are cached, `input_cpy` is remapped to point directly at cache slots (zero-copy)".
  - Eviction is "LFU with aging": counts are halved when they exceed 128. It requires `GGML_OP_OFFLOAD_MIN_BATCH=1`, so all expert matmuls go to GPU and misses are copied over PCIe.
  - RTX 5080 + Ryzen 3900X + DDR4: Qwen3.6 35B "~50 → ~80 t/s", Qwen3-Coder-Next "~30 → ~50 t/s".
  - The discussion says "~60% ... with same VRAM as --n-cpu-moe" and "changes concentrated in single ggml-backend.cpp file".
  - Sources: [Lidenburg fork](https://github.com/Lidenburg/llama.cpp); [Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)
- **xashr three-fork comparison (RTX 5090, Intel 270K, DDR5-6000):**
  - Laguna S 2.1: 18.2 upstream, 27.6 leloch (+52%), 31.3 miltos22 (+72%).
  - DeepSeek V4 Q4_K_XL: 16.5 upstream, 21.7 (+32%), 22.7 (+38%).
  - miltos22 had extreme PP regressions (−90% Laguna, −69% DeepSeek). Lidenburg had DeepSeek PP +216%.
  - Source: [Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)

**e1n00r: Issue #20757 (2026-03-19, closed) and follow-on PRs #21609 / #21614 (April 2026, closed)**
- **#20757** proposes a VRAM slot buffer, a pinned-RAM tier and an mmap/SSD tier. Misses are "fetched from pinned RAM via PCIe", changes go into `ggml_backend_sched_compute_splits()`, eviction is SLRU with "frequency-gated admission filter: experts admitted only on second miss", and transfers use `ggml_backend_tensor_set_async()`. — [Issue #20757](https://github.com/ggml-org/llama.cpp/issues/20757)
- **#20757 data (RTX PRO 2000 8GB, GPT-OSS-120B):** steady state 12–14 tok/s at ~98–100% hit vs "CPU-only MoE offload 0.5–1 tok/s". — [Issue #20757](https://github.com/ggml-org/llama.cpp/issues/20757)
- **#21609 "expert-cache: N-slot LFRU cache with FATE prefetch"** (2026-04-07/08, closed):
  - "96.7% hit rate, 1.79x decode speedup on GPT-OSS-120B with RTX PRO 2000 8GB (--n-cpu-moe 36 --expert-cache-slots 16)".
  - Closed by am17an under the policy "This project does not accept PRs ... fully or predominantly AI-generated". Commits listed "Claude Sonnet 4.6" as co-author.
  - Source: [PR #21609](https://github.com/ggml-org/llama.cpp/pull/21609)
- **#21614 "ggml: persistent expert cache for --n-cpu-moe (RFC #20757)"** (2026-04-08, closed the same day):
  - am17an: "Please stop submitting such large PRs. No one will review them unless you demonstrate you can understand and maintain the code". He pointed to #21067 as an alternative. The thread was locked.
  - The summary claims it cached "in system memory". This is likely a summariser error given the title.
  - Source: [PR #21614](https://github.com/ggml-org/llama.cpp/pull/21614)

**PR #23170 "ggml: treat experts as cache residents during MoE offloading" (avifenesh, closed)**
- It tracks `missing_ids = used_ids & ~loaded_ids` in the scheduler's staging buffer. The staging buffer is invalidated every scheduler epoch, so actual hits were zero (ernestuz-b: "pre_invalid_hit=219222" but no real reuse). The author closed it. — [PR #23170](https://github.com/ggml-org/llama.cpp/pull/23170)

**Issue #26448 "run MoE expert weights from host RAM via PCIe DMA (no H2D copy)", open (measurements dated 2026-08-02) / yalun753 moe-l2**
- **Mechanism:** GPU matmuls operate "directly on host pointers — the CUDA driver DMA-reads the expert weights from host RAM", with an optional VRAM LRU cache. Copy-engine contention is not discussed. — [Issue #26448](https://github.com/ggml-org/llama.cpp/issues/26448)
- **moe-l2 (on-demand pin, 2026-08-07):** `cudaHostRegister` on first touch, "A3 LRU, 2048 slots". The README also says "the scheduler copies only the activated experts to GPU each step, hot experts are cached in VRAM". It is not clear whether misses are zero-copy reads or scheduler copies.
- **moe-l2 numbers (RTX 4090):** Qwen3.6-A3B 50.2 t/s @ 2.9 GB VRAM; DeepSeek-V2-Lite 37.9 t/s @ 2.0 GB vs 65 t/s fully on GPU; DeepSeek-V4-Flash 10.1 t/s. No gpt-oss numbers.
- Source: [yalun753/moe-l2](https://github.com/yalun753/moe-l2)

**PR #25294 "SSD streaming of MoE routed experts from disk" (freedomljc), open**
- A device cache of `n_slots` slabs per layer, a CPU id-remap op, `O_DIRECT` async I/O workers, GEMMs on GPU and wave-partitioned prefill.
- GLM-5.2 UD-Q2_K_XL (~254 GB): decode ~1.83–2.20 tok/s at 73–79% hit. That is 2.08 vs 0.87 tok/s against mmap+CPU-MoE (2.4x).
- lee-b urged deferring the merge until competing tiered-cache proposals are evaluated.
- Source: [PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294)

**Standalone repos found by search**
- **JigSawPT/moe-autopilot** (fork, `aipc-hardening` branch):
  - Static load-time hot set from an offline profiler, not a dynamic cache. It runs a "Dual FFN chain at decode" (hot from VRAM, cold from RAM, merged per layer), only when `n_tokens ≤ 8`.
  - RTX 5090 + 9950X3D + DDR5-6000:
    - **gpt-oss-120B `--n-cpu-moe 25`, HOT_N=22: 36.2 → 47.6 tok/s (+31.6%)**
    - Qwen3-Coder-Next 80B: 72.1 → ~91 (+26%)
    - Qwen3.6-35B: 115.8 → 128.3 (+10.3%)
  - It pays "~10% prefill throughput" and states a break-even prompt:output ratio of about 2.3.
  - Source: [moe-autopilot](https://github.com/JigSawPT/moe-autopilot)
- **ongunm/llama-moe-cache ("FATE")** (~500 LoC):
  - LRU cache plus cross-layer/temporal prediction. Misses are fetched over PCIe on a "Dedicated CUDA prefetch stream".
  - **Qwen3-30B-A3B Q4_K_M on RTX 4070 Ti 12GB: 33.74 → 64.45 t/s (1.91x), hit 99.50%.** The baseline is only called "vanilla".
  - Prompt eval dropped to "4 vs 56 t/s".
  - Source: [llama-moe-cache](https://github.com/ongunm/llama-moe-cache)

**ik_llama.cpp**
- The only related item found is PR #698, "Offload only activated experts to the GPU" (ikawrakow). It concerns which experts are copied for offloaded computation, not a persistent VRAM cache. — [ik_llama.cpp PR #698](https://github.com/ikawrakow/ik_llama.cpp/pull/698)

### Inferences
- Every effort whose code I read (#27861, #26563, leloch v2) moves CPU/GPU work either through standard ggml graph splits or through a synchronous CUDA call made from inside the CPU op. None hands work off from GPU to CPU through pinned memory.
- #26563's admission rule (decayed heat, swap if cold ≥ 1.3× incumbent) is the nearest prior art to "evict lowest decayed count; admit only if count exceeds victim's by a margin". It differs in three ways:
  - It resyncs periodically on a cadence rather than per step.
  - Its margin is multiplicative.
  - Its copies are synchronous `ggml_backend_tensor_set` calls.

  Lidenburg's halving-LFU and e1n00r's LFRU/second-miss admission are other frequency-based variants.
- A dual chain with an exact sum is common to #27861, #26563 and moe-autopilot: GPU slots plus a zero sentinel slot, with cached ids masked or skipped on the CPU side. This part of the target patch is not novel.

### Gaps
- #21620 and #23170's exact dates were not retrieved.
- #26563's exact open date is not on the fetched page; it is inferred from branch commits (2026-08-03/04).
- I did not read Lidenburg's and FATE's code, so their sync mechanisms are unverified.
- Whether FATE's "vanilla" baseline used `--n-cpu-moe` at equal VRAM is not stated.
- No r/LocalLLaMA threads surfaced in two searches.

## Q2. Has any effort removed per-layer host-driven synchronisation (GPU-initiated signalling, CUDA-graph-compatible hand-off, persistent CPU threads polling pinned memory)?

### Takeaway
No. I found no community or upstream effort that uses GPU-written mailboxes, spinning CPU helper threads polling pinned memory, or a GPU spin-wait merge that keeps the decode step in one CUDA graph. Every effort I found keeps host-driven hand-offs:
- (i) ggml scheduler splits, with a CUDA sync before each CPU split.
- (ii) A CPU thread inside `mul_mat_id` that launches GPU work and calls `cudaStreamSynchronize`.
- (iii) Scheduler-side D2H readback of expert ids followed by a sync.

### Cited Findings
- **Upstream baseline, current master (2026-09-27):**
  - The CPU backend declares `cpy_tensor_async = NULL`, `event_record = NULL` and `event_wait = NULL`. Its `graph_compute` runs synchronously on the caller's thread. — [ggml-cpu.cpp (master)](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/ggml-cpu.cpp)
  - In `ggml_backend_sched_compute_splits`, a non-async input copy does `ggml_backend_synchronize(input_backend)` followed by a blocking `ggml_backend_tensor_copy`.
  - The "copy only used experts" path reads ids with `ggml_backend_tensor_get_async(ids_backend, ...)` + `ggml_backend_synchronize(ids_backend)` for each MUL_MAT_ID split.
  - Source: [ggml-backend.cpp (master)](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend.cpp)
- **leloch v2:** each hit dispatch ends in `cudaMemcpyAsync(... DeviceToHost, device.compute_stream)` followed by `cudaStreamSynchronize(device.compute_stream)` ("output synchronization"). This is called from the CPU `mul_mat_id`. — [moe-cache.cu @ e3096b0](https://github.com/leloch/llama.cpp/blob/e3096b046bb809f7f80bc47801f6579aed1cbc60/ggml/src/ggml-cuda/moe-cache.cu)
  - The discussion summary says packed dispatch avoids "per-layer sync points". The code still syncs once per dispatched node; packing reduces the number of H2D operations, not the sync. — [Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)
- **#27861 and #26563:** both use "no custom kernels" or "pure stock ggml ops" dual mul_mat_id chains inside the normal graph, so hand-off is via scheduler splits. — [llama-moecache.h](https://github.com/ggml-org/llama.cpp/blob/bccbacdb8945680f1cfc7e6bffd1e59014705750/src/llama-moecache.h); [llama-expert-tier.h](https://github.com/ggml-org/llama.cpp/blob/38976372452722cf5e6846429aa33ee892c9d89a/src/llama-expert-tier.h)
- **Upstream CUDA-graph work for `--n-cpu-moe`:**
  - #18593 (2026-01-05) disabled CUDA graphs with n-cpu-moe.
  - #18934 (am17an, merged 2026-01-24) re-enabled them as "split-wise" graphs: "one CUDA graph per split (a split is keyed via the first node in the split)". A decode step is therefore several graphs separated by host-driven CPU splits.
  - Measured on 2x4090/1x5090:
    - GPT-OSS 120B: 1.05x at n_cpu_moe=8, 1.10x at 64.
    - GLM4-MoE 106B: 1.04x at 8, 1.07x at 32.
  - Sources: [PR #18934](https://github.com/ggml-org/llama.cpp/pull/18934); commits 908a9e5, 81ab64f in [llama.cpp history](https://github.com/ggml-org/llama.cpp/commits/master)
- **#26802 (merged 2026-08-11):** CUDA graphs are now disabled only when `mul_mat_id`'s fallback "needs to read expert IDs back to the host". The MMQ/MMF paths avoid the sync. Examples: Qwen3-30B-A3B Q4_K_M parallel decode 1.13x at npl=16; LFM2-8B-A1B bf16 357.17 → 374.34 t/s. — [PR #26802](https://github.com/ggml-org/llama.cpp/pull/26802)
- **Host-side sync reduction:**
  - #17795 "CUDA: Improve performance via less synchronizations between token" (aendk, merged 2026-03-05) changed the split pattern from `sccccsg` to `saaasg` with async CPU→CUDA input copies. gpt-oss 20B was 6–9% faster on an RTX PRO 6000 (Windows) and Linux 1–2%. The summary notes it "primarily eliminates host-side stalls" and does not overlap splits.
  - #20793 "sched : reintroduce less synchronizations during split compute" was merged 2026-06-26 and reverted 2026-06-30 in #25138 because "perf regressions are not restriced to EOL AMD HW".
  - Sources: [PR #17795](https://github.com/ggml-org/llama.cpp/pull/17795); [PR #25138](https://github.com/ggml-org/llama.cpp/pull/25138); commits 2cd20b7, 3fc4e10, 86b9470 in [llama.cpp history](https://github.com/ggml-org/llama.cpp/commits/master)

### Inferences
- The target patch's GPU-signalled mailbox (GPU writes inputs and ids to pinned memory, 28 spinning CPU helpers, GPU spin-wait merge, whole decode step as one CUDA graph, all-resident layers never touch the CPU) has no match in what I found. It appears novel relative to the llama.cpp community.
- Upstream's direction has been graph-per-split (#18934) and trimming syncs (#17795). The more aggressive sync removal (#20793) was reverted, which shows maintainers are sensitive to scheduler sync changes.
- "Layers whose selected experts are all resident never touch the CPU" is also not implemented by #27861, #26563 or leloch. Their CPU op or split runs every layer, even when every row hits: #27861 zeroes rows and leloch runs inside the CPU op.
  - Lidenburg's "zero-copy remap when all active experts are cached" is the closest analogue. It is a copy skip inside the all-GPU scheduler-offload path, not a CPU-compute skip.

### Gaps
- I did not inspect Lidenburg's `ggml-backend.cpp` changes or FATE's "GPU barrier" code, so I cannot rule out lower-overhead sync there. Neither README mentions GPU-initiated signalling or persistent polling threads.

## Q3. Has any effort identified that admission copies block the step's own transfers on the copy engine, or fixed CPU/GPU split overlap in ggml-backend's scheduler?

### Takeaway
I found no explicit identification of admission copies queuing ahead of the step's own H2D/D2H transfers on the copy engine, and no admission-by-kernel from pinned memory. Efforts instead throttle or serialise fills:
- #27861 caps inserts per layer per step and uses a worker thread.
- leloch uses a least-priority stream plus serial fills.

Upstream scheduler work touched split concurrency, but only as a race-condition fix that adds a sync (#26040) and as multi-GPU concurrent streams (#28198). There is no CPU/GPU split-overlap fix.

### Cited Findings
- **#27861:**
  - Uploads are done by a worker thread through `ggml_backend_tensor_set`, capped at `max_inserts = 2` per layer per step (`--moe-expert-cache-inserts`). — [llama-moecache.cpp](https://github.com/ggml-org/llama.cpp/blob/bccbacdb8945680f1cfc7e6bffd1e59014705750/src/llama-moecache.cpp)
  - The thread's stated risk is that "if upload speed falls behind decode, all slots may be in-flight". — [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861)
- **leloch v2:** fills go through `cudaMemcpyAsync` on a least-priority non-blocking stream, from a `cudaMallocHost` staging buffer, with `serial_fill` defaulting to on for single-GPU or pre-Ampere systems. The code has no comment about the copy engine or PCIe contention (grep for "copy engine|pcie|contention" found nothing). — [moe-cache.cu @ e3096b0](https://github.com/leloch/llama.cpp/blob/e3096b046bb809f7f80bc47801f6579aed1cbc60/ggml/src/ggml-cuda/moe-cache.cu)
- **#26563:** swaps use synchronous `ggml_backend_tensor_set` at resync cadence. — [llama-expert-hotstore.cpp](https://github.com/ggml-org/llama.cpp/blob/38976372452722cf5e6846429aa33ee892c9d89a/src/llama-expert-hotstore.cpp)
- **Transfer-side work on the offload path:**
  - The issue at /issues/25859, "Offloaded-MoE prefill leaves the GPU idle waiting on serial expert H2D copies" (thecodacus), found pageable bounce buffers, a serial copy→compute order and an avoidable router-id readback sync.
  - Pinning plus second-stream overlap took prefill from ~1143 → 1880 tok/s on an RTX 3060 (Qwen3.6-35B-A3B). yalun753 found double-buffered overlap "net negative" on an RTX 4090.
  - The fetched summary also lists "#25859: Tunable ring buffer" as a related item, which conflicts with the page itself; the numbering is uncertain.
  - Source: [llama.cpp #25859](https://github.com/ggml-org/llama.cpp/issues/25859)
- **am17an's #21067 "ggml: allow prefetching tensor overrides"** (draft) prefetches layer N+1 weights during layer N.
  - On Laguna-S-2.1 with 2x3090: prefill +14.7%, TTFT 1085 → 1604 ms, decode 38.40 → 36.55 t/s (−4.8%).
  - Reviewers: 0cc4m said ggml lacks "explicit transfer-queue distinction" in its API. ORippler flagged event-based sync "as bugged".
  - Source: [PR #21067](https://github.com/ggml-org/llama.cpp/pull/21067)
- **Upstream scheduler changes:**
  - #26040 "ggml: fix backend split scheduler race condition" (Ruben Ortlam, merged 2026-08-20): "splits without input were running concurrently with other splits, while potentially reusing memory the other split is accessing". The fix is to "only sync when split has no inputs". Current master synchronises the previous backend before a zero-input split on a different backend.
  - #28198 "CUDA: Allow concurrent streams per split for multi-GPU" (merged 2026-09-03) applies to `GGML_CUDA_GRAPH_OPT=1` only.
  - Sources: commit [8497981](https://github.com/ggml-org/llama.cpp/commit/8497981); [ggml-backend.cpp (master)](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend.cpp); commit [0ba6499](https://github.com/ggml-org/llama.cpp/commit/0ba6499)
- **#26448's host-DMA proposal** does not discuss copy-engine contention. — [Issue #26448](https://github.com/ggml-org/llama.cpp/issues/26448)

### Inferences
- Doing admission with a kernel that reads pinned host memory, specifically so fills do not queue on the copy engine behind or ahead of step transfers, appears unreported in this community.
- leloch's least-priority stream does not address this. Stream priority affects SM scheduling, not DMA-engine ordering; this is general CUDA knowledge, not something stated in the sources.
- The target patch's "scheduler bug fix so CPU and GPU splits overlap" does not duplicate any upstream fix I found. The most recent upstream change in this area (#26040, Aug 2026) moved toward more serialisation. Comparing the two directly would be useful for the novelty write-up.

### Gaps
- I could not open #26040's PR thread to see whether overlap loss was discussed.
- I could not determine whether #17795's sync reduction (merged March 2026) is still active on master after #20793 was reverted. `git log` shows no explicit revert of #17795, yet #20793 is titled "reintroduce".

## Q4. Does any effort report pre-registered or predicted results, or bounds?

### Takeaway
None pre-registers predictions. A few offer after-the-fact analytical bounds or break-even estimates.

### Cited Findings
- blakemartz computed an "11% theoretical hit rate" for 256-expert load-balanced routing at typical VRAM ratios and concluded that "heat caching can't beat static layer pinning" there. — [PR #26563](https://github.com/ggml-org/llama.cpp/pull/26563)
- moe-autopilot states a prefill-cost break-even "prompt:output ≈ 2.3". — [moe-autopilot](https://github.com/JigSawPT/moe-autopilot)
- #27861's header motivates the design with a measured locality statistic ("LRU-64 hit rate ~67% over a mixed workload ... near-uniform" long-run distribution), not a performance prediction. — [llama-moecache.h](https://github.com/ggml-org/llama.cpp/blob/bccbacdb8945680f1cfc7e6bffd1e59014705750/src/llama-moecache.h)
- noonghunna reported a "linear" coverage/hit-rate/decode relationship up to ~31% coverage, and only +6% decode from a 57 → 100 GB/s DRAM upgrade with the cache on. nibor1896 advised "tune on wall-clock, not hit rate" (1.63x decode at flat hit rate). — [Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)
- leloch's RFC uses a skew statistic, "top 10% of experts take ~80% of hits". — [Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)

### Inferences
- A pre-registered prediction with a speed-of-light bound would set the target work apart methodologically. Community reports are empirical before/after tables with heterogeneous baselines.

### Gaps
- I did not read the full thread text beyond the summaries. A buried predictive model in a comment cannot be fully excluded.

## Q5. What speedups over --n-cpu-moe are reported for gpt-oss-120b / gpt-oss-20b / Qwen3-30B-A3B?

### Takeaway
- **gpt-oss-120b:** there are two cache reports: 1.79x (#21609, RTX PRO 2000 8GB, `--n-cpu-moe 36`, 16 slots) and +31.6% (moe-autopilot static hot set, RTX 5090, `--n-cpu-moe 25`).
- **Qwen3-30B-A3B:** there are two: +14.6% (#27861, RX 7600 8GB, Vulkan) and 1.91x (FATE, RTX 4070 Ti, baseline unspecified).
- **gpt-oss-20b:** I found no expert-cache numbers.

None of these is clearly "at equal VRAM"; most caches add VRAM on top of the baseline.

### Cited Findings
- **gpt-oss-120b:**
  - #21609: "96.7% hit rate, 1.79x decode speedup on GPT-OSS-120B with RTX PRO 2000 8GB (--n-cpu-moe 36 --expert-cache-slots 16)" (April 2026, closed). — [PR #21609](https://github.com/ggml-org/llama.cpp/pull/21609)
  - #20757 (same author, same GPU): steady state 12–14 tok/s vs "CPU-only MoE offload 0.5–1 tok/s". — [Issue #20757](https://github.com/ggml-org/llama.cpp/issues/20757)
  - moe-autopilot: `--n-cpu-moe 25`, HOT_N=22, 63.4% coverage, 36.2 → 47.6 tok/s (+31.6%), RTX 5090 + Ryzen 9950X3D + DDR5-6000. — [moe-autopilot](https://github.com/JigSawPT/moe-autopilot)
  - Upstream, not a cache: split-wise CUDA graphs gave GPT-OSS 120B 1.05x at n_cpu_moe=8 and 1.10x at 64. — [PR #18934](https://github.com/ggml-org/llama.cpp/pull/18934)
- **gpt-oss-20b:** the only number found is from #17795's sync reduction (6–9% on an RTX PRO 6000 Blackwell Max-Q, Windows), which is not an expert cache. — [PR #17795](https://github.com/ggml-org/llama.cpp/pull/17795)
- **Qwen3-30B-A3B:**
  - #27861 on Radeon RX 7600 8GB (Vulkan), Q4_K_M: 14.4 → 16.5 tok/s with 32 slots, 74.7% hit. — [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861)
  - FATE, Q4_K_M on RTX 4070 Ti 12GB: 33.74 → 64.45 t/s (1.91x), 99.50% hit. The baseline is labelled "vanilla". — [llama-moe-cache](https://github.com/ongunm/llama-moe-cache)
  - #26802 (CUDA graphs, not offload): parallel decode npl=16 1.13x. — [PR #26802](https://github.com/ggml-org/llama.cpp/pull/26802)
- **Equal-VRAM claims:** only Lidenburg's is phrased that way ("~60% ... with same VRAM as `--n-cpu-moe`"), and it was measured on Qwen3.6-35B / DeepSeek / Laguna, not the three target models. — [Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)

### Inferences
- The target patch's 1.41–1.81x over `--n-cpu-moe` at equal VRAM on an A10 is within the range of community single-number claims: 1.79x for gpt-oss-120b in #21609, 1.91x for Qwen3-30B-A3B in FATE.
- The equal-VRAM control is rarer. Most community baselines hold `--n-cpu-moe` fixed and then add cache VRAM (e.g. moe-autopilot adds HOT_N on top of `--n-cpu-moe 25`).
- A direct gpt-oss-120b comparison should note that #21609 was on an 8 GB RTX PRO 2000, a much lower VRAM fraction than an A10's 24 GB.

### Gaps
- There are no community expert-cache numbers for gpt-oss-20b.
- None of the found reports use an A10.
- The #27861 baseline flag (`-ot` vs `--n-cpu-moe`) for the Qwen3-30B-A3B row, and FATE's baseline configuration, are not stated in the fetched text.
