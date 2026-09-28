# The llama.cpp ecosystem for batch-1 hybrid CPU/GPU MoE decode (state on 28 Sep 2026)

Method note. This note extends three earlier notes: `llamacpp_community_caches.md`, `industry_engines.md` and `llamacpp_baseline_history.md`. Findings already in those notes are only referenced here, not repeated.

- **Upstream code and commits.** I made a blob-less bare clone of ggml-org/llama.cpp (master `4da6337`, committed 2026-09-27 23:28 +0200). I also fetched the head commit of every PR numbered 25500 or higher (2,445 PRs) through `refs/pull/N/head`, then read commit subjects and messages with `git log`.
- **Open or closed state of PRs.** This is **inferred**: a PR counts as open if `refs/pull/N/merge` exists. A missing ref means the PR is closed, or open but unmergeable. "Merged" means a squash commit `(#N)` exists on master.
- **PR and discussion comment threads.** These were read through WebFetch, which returns summaries written by a small model. Quotes marked "as summarised" may be paraphrased. Direct curl to github.com and api.github.com returned HTTP 403 through the proxy.
- **Patch applicability.** Tested with `git apply --check` on a clean worktree of master `4da6337`.
- **ik_llama.cpp.** Tree-less clone at `adce16f` (2026-09-28). Individual files were read lazily with `git show`.

Abbreviations: LC = https://github.com/ggml-org/llama.cpp, IK = https://github.com/ikawrakow/ik_llama.cpp.

## Q1. Which upstream commits and open PRs since about 1 Sep 2026 touch MoE offload, the scheduler, split overlap, CUDA graphs with CPU splits, `--n-cpu-moe`/`--fit`, expert caching or FP4 kernels on Blackwell? Does anything conflict with or supersede a GPU expert cache?

### Takeaway
492 commits landed on master between 1 and 27 Sep 2026. None adds an expert cache or lets CPU and GPU splits overlap. None changes the batch-1 hybrid decode critical path in a way that supersedes the patch.
- **Scheduler:** the `ggml-backend.cpp` changes are three small hygiene fixes (#28387, #28739, #26070).
- **CUDA MoE:** the work is GPU-side kernel fusion (#25952 weighted-reduction fusion, #28432 top-k MoE fusion) and MMVQ tuning.
- **Blackwell FP4:** on Blackwell, batch-1 `MUL_MAT_ID` always runs through MMVQ. The native FP4 (W4A4) tensor-core work therefore affects only prefill and batched MMQ.
- **Rebase:** both of Harshith's patches apply cleanly to current master `4da6337`. Only 22 commits separate the base `2145525a` from master, and none touches `ggml-backend.cpp`.
- **Open PRs that would collide in the next two weeks if merged:**
  - am17an's scheduler sanitizer #26167 (rewrites parts of `ggml-backend.cpp`)
  - #28874 (reorders input copies in `compute_splits`)
  - am17an's "fuse shared experts into MMVQ" #29184 (touches `mmvq.cu` and `ggml-cuda.cu`, both also patched)
  - CISC's "fuse grouped experts" #29181
- **Benchmark methodology:** NVIDIA's open #29539 adds benchmark-only random expert routing.
- **Blackwell correctness:** there are open reports of IQ-quant miscompilation on sm_120 with nvcc 13.2 and a CUDA-13.1 MMQ crash.

### Cited Findings
**Base and rebase status**
- Master HEAD is `4da6337` (2026-09-27), "server : allow RANK pooling batch splitting…" — [commit](https://github.com/ggml-org/llama.cpp/commit/4da6337767f973e2b4d0797e5b323d77d8565e4a).
- The patch base `2145525a` (2026-09-26) is the revert "Change max context length for auto-fitting with unified KV (#28849)" (#29437). #28849 had been merged on 2026-09-16 — [commit 2145525a](https://github.com/ggml-org/llama.cpp/commit/2145525a4081d66ff1a87cf43ef809f95a85ac0c).
- Between `2145525a` and `4da6337`, the only commits touching files the patch also touches are:
  - #29096 (F16 input to FWHT) and #28717 (Nemotron ssm scan) in `ggml-cuda.cu`;
  - #28876 (RANK pooling) and #29502 (hexagon sampler) in `llama-context`/`llama-graph`;
  - tests.
  - Source: `git log 2145525a..4da6337` on the [LC repository](https://github.com/ggml-org/llama.cpp).
- `git apply --check` of `runtime/llama.cpp-expert-cache.patch` and of `runtime/ggml-sched-overlap.patch` on master `4da6337` both succeeded. Applying the EC patch first and then the sched patch fails, because the EC patch already contains the overlap hunks (`overlap_cpu` appears 5 times in `ggml-backend.cpp` after the EC patch). Local test; [LC master](https://github.com/ggml-org/llama.cpp/tree/4da6337767f973e2b4d0797e5b323d77d8565e4a).

**Merged scheduler changes (`ggml/src/ggml-backend.cpp`) since 1 Sep**
- #28387 "ggml: allow backend inputs to not create another split" (am17an, 2026-09-07):
  - Removes the check that forced a new split when a split's input array was full ("FIXME: count the number of inputs instead of only checking when full").
  - Demotes the "increasing split inputs capacity" warning to debug.
  - Source: [commit 992cb503](https://github.com/ggml-org/llama.cpp/commit/992cb503cdacf691ef06c332d05243bc7807257b).
- #28739 "ggml: skip 0-sized ids tensor when offloading selected experts" (am17an, 2026-09-11). Adds `if (ggml_nelements(ids_tensor) == 0) continue;` in the copy-only-used-experts path — [commit 43f3dda6](https://github.com/ggml-org/llama.cpp/commit/43f3dda6237a453a587a8f00230d52decfeaa8e5).
- #26070 "ggml : handle graph buffer reservation failure" (2026-09-18). `ggml_backend_sched_alloc_splits` now returns false if `ggml_gallocr_reserve_n` fails — [commit 911f6cdc](https://github.com/ggml-org/llama.cpp/commit/911f6cdc8ab8a530b2bee09ee61471a6f3178eeb).
- Related but outside `ggml-backend.cpp`:
  - #26625 (ggerganov, 2026-09-21) reports graph inputs and input tensors during sched reserve — [commit 96550613](https://github.com/ggml-org/llama.cpp/commit/96550613656e7f024df65f91cf8b2d80a83cf09e).
  - #29514 (2026-09-27) turns on `GGML_SCHED_DEBUG_REALLOC=1` for ctest CI — [commit 7fb2b082](https://github.com/ggml-org/llama.cpp/commit/7fb2b082ce762cfc8b19f3258a40f07c1fd075ab).
- #28198 "CUDA: Allow concurrent streams per split for multi-GPU" (2026-09-03). The commit says "Default behaviour remains unchanged, only active for GGML_CUDA_GRAPH_OPT=1" — [commit 0ba6499c](https://github.com/ggml-org/llama.cpp/commit/0ba6499c3ba73ed408acac62ea650d3af613794a).
- Pipeline parallelism (event-based, `n_copies > 1`) is enabled only when all of these hold: `model.n_devices() > 1`, all layers are offloaded, split mode is layer, and `!model.has_tensor_overrides()`. Single-GPU `--n-cpu-moe` therefore always runs with `n_copies == 1` — [llama-context.cpp @ master](https://github.com/ggml-org/llama.cpp/blob/4da6337767f973e2b4d0797e5b323d77d8565e4a/src/llama-context.cpp).

**Merged CUDA MoE and decode changes**
- #25952 "cuda: fuse MoE weighted expert reduction" (2026-09-01), quoting the commit:
  - It "matches the full expert-weighting plus ordered-reduction subgraph and replaces it with one weighted-reduction kernel", for k = 2..15.
  - "Results are not claimed bit-identical".
  - "Set GGML_CUDA_MOE_WEIGHTED_REDUCTION=0 to disable the fusion."
  - Source: [commit 3466812d](https://github.com/ggml-org/llama.cpp/commit/3466812d1f06728effe7c0f3c0671117f461672d).
  - The open follow-up #29381 is "cuda: preserve MoE reduction rounding" (2026-09-24) — [PR #29381](https://github.com/ggml-org/llama.cpp/pull/29381).
- #28432 "cuda: top-k MoE should always fire" (am17an, 2026-09-23) adds allocator dependencies "for performance positive fusions. This may increase the overall compute buffer size." — [commit 1a679828](https://github.com/ggml-org/llama.cpp/commit/1a679828f3312ebc53c9285805cc8bd7c7d90366).
- #28475 "cuda: fixes races in mmid and mmf" (2026-09-06) — [commit 73a43d1f](https://github.com/ggml-org/llama.cpp/commit/73a43d1f69345aee8bb186ef4b3172cef892f2e5).
- #26705 "CUDA: branchless Q4_K/Q5_K unpack to speed up mmvq…" (2026-09-07) — [commit 73ab7599](https://github.com/ggml-org/llama.cpp/commit/73ab7599b553c03f6f5d2db24a18ad76f2eb36a3).
- #26079 (NVIDIA co-authored, 2026-08-20) adds per-hardware MMVQ-to-MMQ switch points, including "Blackwell specific switch point".
  - It reports "+23-41% at B=8 on RTX 5090 for Q4_K dense".
  - Master code, "tuned on RTX 5090": Q2_K/Q3_K/Q4_K use MMVQ at ≤5 columns, Q5_K at ≤6, Q6_K at ≤7, and other types up to `MMVQ_MAX_BATCH_SIZE`.
  - Sources: [commit 2b562109](https://github.com/ggml-org/llama.cpp/commit/2b5621094ef383cdcd8428ef6d22efe5df976532); [mmvq.cu @ master](https://github.com/ggml-org/llama.cpp/blob/4da6337767f973e2b4d0797e5b323d77d8565e4a/ggml/src/ggml-cuda/mmvq.cu).
- For `MUL_MAT_ID`, master says "NVIDIA: Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID" (up to `MMVQ_MAX_BATCH_SIZE`) — [mmvq.cu @ master](https://github.com/ggml-org/llama.cpp/blob/4da6337767f973e2b4d0797e5b323d77d8565e4a/ggml/src/ggml-cuda/mmvq.cu).
- On the MMQ side, master says "src1 is quantized to Q8_1 unless the FP4 types can use 4-bit activations, in which case they default to the native W4A4 instructions on Blackwell". It applies to MXFP4 and NVFP4, and can be overridden with `GGML_CUDA_MMQ_PREC=q4|q8|auto` — [mmq.cu @ master](https://github.com/ggml-org/llama.cpp/blob/4da6337767f973e2b4d0797e5b323d77d8565e4a/ggml/src/ggml-cuda/mmq.cu).
- #24364 "llama : add `llama_prec_policy` + model-driven W4A4 path" (NVIDIA author, 2026-09-25) includes "MXFP4 dispatch changes for higher src prec" — [commit e9f824d8](https://github.com/ggml-org/llama.cpp/commit/e9f824d8c0f011662a742c9d15d4aa18a41e32c0).
- Earlier FP4 work, in master's log:
  - "CUDA: Improve NVFP4 W4A4 activation quantization" (#25730, 2026-07-22)
  - "CUDA: Fuse MMVQ post-scale for NVFP4" (#24481, 2026-07-07)
  - "CUDA: experimental native mxfp4 support for blackwell" (#17906, 2025-12-24)
  - Source: [LC commits](https://github.com/ggml-org/llama.cpp/commits/master).
- The CUDA CMake file replaces plain `12X` architectures with `12Xa` because "the Blackwell FP4 tensor core instructions are not forwards compatible and therefore need 12Xa". Default non-native builds add `120a-real` — [ggml-cuda/CMakeLists.txt @ master](https://github.com/ggml-org/llama.cpp/blob/4da6337767f973e2b4d0797e5b323d77d8565e4a/ggml/src/ggml-cuda/CMakeLists.txt).
- #28549 "Enable CUDA graph for MTP draft" (2026-09-16) — [commit 2f3fd025](https://github.com/ggml-org/llama.cpp/commit/2f3fd02526682adbd3ba771d929d271e477a35c5).
- `GGML_CUDA_GRAPH_OPT` is still opt-in (`env != nullptr && atoi(env) == 1`). The `cudaLaunchHostFunc` cross-backend event wait is still `#if 0` — [ggml-cuda.cu @ master](https://github.com/ggml-org/llama.cpp/blob/4da6337767f973e2b4d0797e5b323d77d8565e4a/ggml/src/ggml-cuda/ggml-cuda.cu).

**Merged CPU and common changes**
- #27851 "ggml-cpu: tiled mul_mat for k-quants" (2026-09-26). The commit reports "3-6x speed improvement for large matmul, break even at 4096x64 * 64x4096, 80% performance (net loss) for GEMV".
  - The code gates it with `ggml_tiled_min_batch`: "Profitable at rows >= 8". It skips repacked weights (`src0->extra != NULL`) and covers `MUL_MAT_ID` per expert. `GGML_CPU_TILED_MM=0` disables it.
  - Sources: [commit d834d44e](https://github.com/ggml-org/llama.cpp/commit/d834d44e643681f7b046a22d357335f6f4ff6107); [tiled.cpp @ master](https://github.com/ggml-org/llama.cpp/blob/4da6337767f973e2b4d0797e5b323d77d8565e4a/ggml/src/ggml-cpu/tiled/tiled.cpp).
- #28334 "args: officially deprecate --mmap|mlock|dio" (2026-09-09), in favour of `-lm/--load-mode` — [commit 14a9d09f](https://github.com/ggml-org/llama.cpp/commit/14a9d09f75683c94c2c4f229efe54670d4209089).
- #28323 "src : add n_expert_used_max function" (2026-09-04) — [commit 9a4843cf](https://github.com/ggml-org/llama.cpp/commit/9a4843cf2f1a3fc8e39f8148e92ee6bfe18e2db6).
- #27483 "misc : prevent RAM peaking at model loading stage" (2026-09-03) — [commit 5ec4eab6](https://github.com/ggml-org/llama.cpp/commit/5ec4eab69edbfaa4b01bcf1ee1123bb885b8d892).

**Open PRs (unmerged) in the same code**
- **#26167 "ggml: add a scheduler sanitizer"** (am17an, head 2026-09-20, merge ref present so likely open).
  - Size: adds `ggml-backend-sanitize.cpp` (565 lines) and changes `ggml-backend.cpp` by +60/−26.
  - What it does (as summarised): it detects "happens-before" races between splits and backends. It found a race involved in #26040 and one "introduced in #21067".
  - ggerganov asked for "a minimal internal scheduler callback API".
  - JohannesGaessler said it is "undesirable to have a global state like this" and suggested making the sanitizer "a part of the backend scheduler".
  - Source: [PR #26167](https://github.com/ggml-org/llama.cpp/pull/26167).
  - pwilkin's earlier variant #27714 ("add a non-fatal mode", "only ring buffer graph inputs for an asynchronous backend") has no merge ref — [PR #27714](https://github.com/ggml-org/llama.cpp/pull/27714).
- **#27258** (aendk, head 2026-09-23) adds scheduler stress tests: `tests/test-backend-sched.cpp` (1,029 lines) and a new `GGML_OP_SLEEP` "to simplify scheduler testing" — [PR #27258](https://github.com/ggml-org/llama.cpp/pull/27258).
- **#28874 "ggml-backend : copy user inputs before cross-device inputs in compute_splits"** (douyamv, 2026-09-14, likely open).
  - Commit text: "With n_copies == 1 (no pipeline parallelism, e.g. whenever tensor overrides are used) the user-input copy synchronizes the whole stream of the split backend… the host blocks until the previous GPU has finished its entire graph -- at every split boundary of every decode step."
  - It declares "Co-Authored-By: Claude Fable 5.1".
  - It changes the same input-copy loop that the overlap patch restructures.
  - No maintainer comments were retrievable (the fetch returned only the description).
  - Source: [PR #28874](https://github.com/ggml-org/llama.cpp/pull/28874).
- **#28414 "--prefetch-experts-slots"** (2026-09-04, likely open).
  - Adds 297 lines to `ggml-backend.cpp`. It fires only on `MUL_MAT_ID` splits with batch ≥ 2·n_expert, and "decode is unaffected".
  - Results: TTFT −11% (24B-A3B, `-ncmoe 20`) and −22% (35B-A3B) on an RTX 5070 Ti.
  - Multi-GPU users reported "silent wrong output".
  - pwilkin (as summarised): "I'm not considering any PRs of this sort unless someone can clearly show…this beats purely using `--mmap`".
  - Source: [PR #28414](https://github.com/ggml-org/llama.cpp/pull/28414).
- **#29184 "CUDA: fuse shared experts into MMVQ"** (am17an, head 2026-09-23, likely open).
  - It matches `MUL_MAT_ID, MUL_MAT_ID, GLU` followed by `MUL_MAT, MUL_MAT, GLU` with `ne[1] == 1` (single-token decode). The routed and shared gate/up are computed in one MMVQ launch.
  - It touches `mmvq.cu` (+46/−9) and `ggml-cuda.cu` (+89).
  - Source: [PR #29184](https://github.com/ggml-org/llama.cpp/pull/29184).
- **#29181 "fuse grouped experts"** (CISC, 2026-09-23): `topk-moe.cu` +287, `ggml-cuda.cu` +397 — [PR #29181](https://github.com/ggml-org/llama.cpp/pull/29181).
- **#29539 "Add LLAMA_MOE_RANDOM_ROUTING=1 for benchmark-only random expert selection"** (Gaurav Garg, 2026-09-27, likely open).
  - Code comment: "benchmark only: the expert GEMMs use random ids, the router above is not changed".
  - IDs are seeded from the token position and drawn by a partial Fisher-Yates shuffle, giving distinct experts per token.
  - Source: [PR #29539](https://github.com/ggml-org/llama.cpp/pull/29539).
- **Expert pinning in RAM (not VRAM):**
  - #26414 `--pin-hot-experts N` (Ghimli, head 2026-09-24, likely open) uses `mlock` on the hottest experts inside the mmap and adds `--load-mode mmap+pin` — [PR #26414](https://github.com/ggml-org/llama.cpp/pull/26414).
  - #28545 "experimental mlock pinning of hot MoE expert slices" (2026-09-07; no merge ref) states "on near-uniform routers (measured: Qwen3.8-Flash-Next, 98.6% of slots active in 1.3k tokens) no policy beats LRU" — [PR #28545](https://github.com/ggml-org/llama.cpp/pull/28545).
- **Blackwell correctness**
  - #28784 "CUDA: fix IQ1_S/IQ2_S/IQ3_S garbage output on Blackwell (sm_120)" (2026-09-11; no merge ref).
    - Commit text: "On nvcc 13.2 targeting sm_120 this byte read is miscompiled". The affected sites are `mmq-load-tiles.cuh` and the `vecdotq.cuh` `vec_dot_iq*_s_q8_1` functions. Tests "previously FAIL with NMSE ~0.2-1.1".
    - Source: [PR #28784](https://github.com/ggml-org/llama.cpp/pull/28784).
  - #28823 "cuda: fall back to cuBLAS for IQ quants on Blackwell" (likely open): "IQ MMQ produces incorrect results on sm_120" — [PR #28823](https://github.com/ggml-org/llama.cpp/pull/28823).
  - #27215: on an RTX 5080 with driver 595.84 / CUDA 13.3, a bogus `sharedMemPerBlockOptin` made `ggml_cuda_launch_mm_ids_helper` ("used by mul_mat_id -- i.e. any MoE model") abort — [PR #27215](https://github.com/ggml-org/llama.cpp/pull/27215).
  - A March 2026 write-up (llama.cpp b8240) reports that "CUDA 13.1 causes MMQ kernel segmentation faults on Blackwell" and that a stale `GGML_CUDA_FORCE_CUBLAS=ON` gave a 5x slowdown. It recommends CUDA 12.8 with `-DCMAKE_CUDA_ARCHITECTURES=120` on an RTX 5090, where Qwen3.5-35B-A3B Q4_K_M ran at tg128 = 211 t/s — [zenn.dev](https://zenn.dev/toki_mwc/articles/rtx5090-blackwell-cuda-toolkit-trap-llama-cpp?locale=en).

### Inferences
- **Nothing upstream supersedes the patch.** As of master `4da6337`:
  - the scheduler still executes splits in order;
  - the CPU backend still has no async or event support (no `ggml-cpu.cpp` interface change since 1 Sep; earlier note checked `d7fb90e`);
  - no expert cache or GPU-signalled hand-off exists.
- **Rebasing now is trivial.** The real merge-conflict risk is #26167 (sanitizer rewrite of the scheduler compute path) and #29184/#29181 (fusion code in `mmvq.cu` and `ggml-cuda.cu`). The simplest course is to freeze the race base at `4da6337` or `2145525a` and record it.
- **Stock fusions probably won't fire on the patched graph.** #25952 (weighted reduction), #28432 (top-k MoE) and, if merged, #29184 (shared expert + routed gate/up in one MMVQ) match exact graph shapes. The expert-cache graph rewrites the MoE block into a dual chain plus merge, so these fusions probably do not fire on cached layers.
  - Consequence 1: the baseline gets fusion speed-ups that the patched layers may not.
  - Consequence 2: outputs will not be bit-identical to the baseline even when the cache is exact ("Results are not claimed bit-identical").
  - For output-equivalence checks, run the baseline once with `GGML_CUDA_MOE_WEIGHTED_REDUCTION=0` as well. (Inference; not verified by running.)
- **#29539 as a stress test.** Random routing is a ready-made adversarial control for any expert cache, since uniform routing approximates the worst case. It must not be applied to only one side of a race. It is unmerged, so porting its ~100 lines into the race harness is possible.
- **Blackwell FP4 does not change batch-1 decode.** Batch-1 GPU expert matvecs on Blackwell run through MMVQ with Q8_1 activations, both for baseline experts and presumably for the patch's cache hits. The W4A4 native-FP4 path only changes prefill and batched verification (MTP/speculative), so it does not tilt a batch-1 decode race.
- **Blackwell build hygiene.** Build with CUDA 12.8/12.9, or verify 13.x, with arch `120a`. Run `test-backend-ops -o MUL_MAT_ID` on the rented card. Avoid IQ1_S/IQ2_S/IQ3_S GGUFs (e.g. some Unsloth dynamic quants) unless correctness is checked.

### Gaps
- Open/closed state for #26414, #28545, #28784 and #27714 comes from merge refs only and was not confirmed in the GitHub UI.
- Review threads were not read for #29184, #29181, #29539 or #28874 (the WebFetch of #28874 returned only the description).
- Whether #25952/#28432 fusions still fire on uncached layers of the patched graph, and on cached layers, was not checked. That needs a run with `GGML_SCHED_DEBUG` or a CUDA trace.
- Whether the #28784 miscompile is fixed on master by another route could not be determined. `vecdotq.cuh` has no commit since mid-August other than #24364/#26705.

## Q2. Where does the community expert-cache work stand, what did maintainers say, and what speed-ups and hardware were reported?

### Takeaway
No community expert cache is merged.
- **#27861** (csantiago78, single-GPU, "no custom kernels") is the only one still open. It is a draft; its last code push was 2026-08-28 and its last thread activity is around 2026-09-14. It has no core-maintainer review.
- **#26563 and #26824** (miltos22/Miltos22) are closed without merge. The author promised a "new clean pr", but no successor PR appears among PR heads ≥ #25500. The `Miltos22/llama.cpp` fork is not publicly reachable.
- **leloch's RFC** is still an unanswered discussion. His `moe-cache-v2-pr` branch (2026-08-06) is the most complete: user docs and a regression matrix. It is CUDA-only, and its default "auto" mode needs at least two CUDA devices of compute capability 8.0 or higher.

Maintainers have spoken only about process, never about design:
- am17an asked for an RFC because the PR was "too large for any maintainer to review", and told e1n00r to "stop submitting such large PRs".
- Collaborator pwilkin, on #28414, would not consider such PRs unless they clearly beat plain `--mmap`.

Nobody with merge rights has endorsed a design for expert caching. Meanwhile CONTRIBUTING.md now allows AI-generated code, with disclosure and strict review expectations.

### Cited Findings
**#27861, "GPU-resident LRU cache for host-offloaded MoE expert weights"** (csantiago78)
- Status: draft, label `ggml`, merge ref present. The branch head is `bccbacd` (2026-08-28), and it has no commits after that — [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861).
- September thread, as summarised — [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861):
  - Sept 3–4: Blackwell RTX PRO 4500 with Qwen3.8-Flash-Next (104 GB), "Default settings 21.95 t/s" and "Tuned (96 slots) 41.31 t/s, 81.5% hit rate". The stock figure of 16.83 t/s comes from the earlier note.
  - Sept 6: dual 7900 XTX, cache + MTP "39.3 tok/s hot" vs "38.08 tok/s MTP only". Profiling showed "2.99x stream synchronizations" and "9.80x H2D operations".
  - Sept 8–9: SYCL in-place weight reorder produced garbage.
  - Sept 14: a race, "Cache can change mappings while earlier GPU graph still reads that state". Proposed fix: "finish scheduled graph execution, then publish throttled LRU updates".
  - RX 9070 XT 16 GB (Vulkan), Qwen3.6-35B-A3B Q4_K_XL, 24 layers on CPU: 32.5–33.2 t/s.
  - "None identified from primary maintainers".
- Files touched: `ggml-cpu.c` (+52), `ggml.c`, `ggml.h`, `llama-graph.cpp` (+69), and new `llama-moecache.cpp`/`.h` (468 lines). Total +645 — [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861), via git diff of `refs/pull/27861/head`.

**#26563** (miltos22, head 2026-08-06, no merge ref)
- Closed with plans to reorganise: "I have redesigned about half of the entire system in ways that fix all major issues exposed by this pr". The author said he would push changes or "make a new clean pr with a new commit history". No replacement number appears on the page — [PR #26563](https://github.com/ggml-org/llama.cpp/pull/26563).

**#26824** (Miltos22, the heatmap / mmap-pinning PR, head 2026-08-10)
- No merge ref and no `(#26824)` commit on master, so it is closed, not merged.
- Features, as summarised: heatmap prioritisation, "Memory-mapped (mmap) page pinning", CPU↔GPU swapping with "hash verification", persistent heatmap sidecar files, "adaptive resync cadence".
- RTX 3070 8 GB numbers: Qwen3.6-35B Q4_K_M 32.0 → 47.7 (1.49x); Gemma-4-26B 19.9 → 43.6 (2.19x); Qwen3.5-122B IQ2_M 7.18 → 12.5 (1.74x).
- Author: "i really need a mental health break at the moment. So, likely, soon-ish".
- Collaborator IMbackK (as summarised): "break it down into parts so that they can be merged one by one". The PR violated the "one change per PR rule".
- Source: [PR #26824](https://github.com/ggml-org/llama.cpp/pull/26824).
- `git ls-remote https://github.com/Miltos22/llama.cpp` returned no branches, and the lowercase `miltos22/llama.cpp` asked for credentials (i.e. not public), checked 2026-09-28 — [LC PR list](https://github.com/ggml-org/llama.cpp/pulls).

**leloch: #24524 and Discussion #24528**
- On #24524, am17an wrote: "You can open a RFC in discussions page, as such is PR is just too large for any maintainer to review. Ask the AI to perhaps create it so that it clearly mentions the benefits vs the maintenance burden of such a change." The author disclosed the code as "predominantly AI-generated (Anthropic's Claude Fable 5)" — [PR #24524](https://github.com/ggml-org/llama.cpp/pull/24524).
- Discussion #24528 has "No official maintainer comments". The latest version is the v2-pr branch (2026-08-06) — [Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528).
- leloch's fork currently has four branches: `moe-cache`, `moe-cache-pr`, `moe-cache-v2-pr` (e3096b0) and `v3-expert-cache`.
  - `v3-expert-cache` is an older June 2026 research branch despite its name. It is 29 commits ahead of a 2026-06-10 base, with commit subjects including "matrix rows — gpt-oss-120b adversarial pass" and "gpt-oss-20b, OLMoE; 7 architectures validated".
  - Source: [leloch/llama.cpp](https://github.com/leloch/llama.cpp).

**leloch v2 documentation** (`docs/backend/CUDA-MOE-CACHE.md` on `moe-cache-v2-pr`)
- Mode table:
  - `auto` needs "At least two eligible selected CUDA devices, compute capability 8.0 or newer".
  - `on` or a fixed `N` MiB needs "At least one eligible selected CUDA device, compute capability 7.0 or newer", and turns weight repacking off.
- "A cache hit runs the selected expert matvec on CUDA while the CPU computes the miss rows through the normal `MUL_MAT_ID` kernel."
- Recommended three-arm protocol:
  - "`--moe-cache off --repack on`: optimized CPU-expert baseline"
  - "`--moe-cache off --repack off`: canonical CPU-expert baseline"
  - cache arm with `--repack off`
- "Broader regression matrix", with experts forced to CPU and canonical weights; small models ran on one RTX 3090:

  | Model | Cache off (t/s) | Cache on (t/s) | Change |
  |---|---|---|---|
  | Qwen3-30B-A3B Q4_K_XL | 71.93 | 73.43 | +2.1% |
  | Qwen3.6-35B-A3B Q4_K_XL | 77.60 | 84.04 | +8.3% |
  | OLMoE-1B-7B | — | — | +26.7% |
  | ERNIE-4.5-21B-A3B | 79.66 | 127.40 | +59.9% |
  | DeepSeek-V2-Lite | — | — | +25.5% |
  | Llama-4 Scout | 25.97 | 46.90 | +80.6% |
  | MiniMax M2.7 IQ2_XXS | — | — | +26.6% |

- Listed limitations: "CUDA only"; "Demand fill only"; "No runtime performance bail-out"; the cache is bypassed if `GGML_OP_OFFLOAD_MIN_BATCH` is set low.
- Source: [leloch moe-cache-v2-pr](https://github.com/leloch/llama.cpp/tree/moe-cache-v2-pr).

**Other forks**
- Lidenburg `moe-expert-caching` is 2 commits on a 2026-06-13 base (last commit 2026-06-22, "Track time spent waiting for disk reads"), so it is not active — [Lidenburg/llama.cpp](https://github.com/Lidenburg/llama.cpp).
- JigSawPT/moe-autopilot has one branch, `claude/llama-expert-cache-discussion-hrwo7a` — [moe-autopilot](https://github.com/JigSawPT/moe-autopilot).
- Summer/vnlpscale "tiered: adaptive expert cache for SSD-backed decode" (#26503, #27070; August; no merge refs) — [PR #27070](https://github.com/ggml-org/llama.cpp/pull/27070).

**Contribution policy on master**
- CONTRIBUTING.md reads:
  - "AI-generated code is allowed. You are 100% responsible for every line, however it was produced."
  - "Undisclosed AI usage may result in your account being permanently banned".
  - Contributors must "Explicitly disclose the manner in which AI was employed", "Check for an existing PR addressing the same change; if one exists, comment there", and review for "something like one hour per 200-400 LOC".
  - "It is strictly prohibited to use AI to write your posts for you (… pull request descriptions …)".
  - "New CLI or public API additions carry a **higher bar**".
  - New contributors should "Limit your open PRs to 1".
  - Maintainers should "Be mindful of maintenance … If the PR author is not committed to contribute long-term, someone else needs to take responsibility (you)".
  - Source: [CONTRIBUTING.md @ master](https://github.com/ggml-org/llama.cpp/blob/4da6337767f973e2b4d0797e5b323d77d8565e4a/CONTRIBUTING.md).
- Earlier statements, from the prior note: am17an closed #21609 under the then-policy "This project does not accept PRs … fully or predominantly AI-generated", and on #21614 wrote "Please stop submitting such large PRs. No one will review them unless you demonstrate you can understand and maintain the code" — [PR #21609](https://github.com/ggml-org/llama.cpp/pull/21609); [PR #21614](https://github.com/ggml-org/llama.cpp/pull/21614).
- am17an's own prefetch draft #21067 still has a merge ref (likely open) — [PR #21067](https://github.com/ggml-org/llama.cpp/pull/21067).

### Inferences
- **For the race, the natural "community cache" entrants on one Blackwell card:**
  - #27861: open; builds on master-era code; works on one GPU; default inserts = 2 per layer per step.
  - leloch v2: must use `--moe-cache on` or a fixed `N`, because `auto` refuses a single device. It turns CPU repacking off, so compare it against the repack-on baseline as the end-to-end arm.
  - #26563/#26824 are still fetchable via `refs/pull/N/head` but are abandoned.
- **Maintainers treat expert caching as a maintenance-burden question, not a technical one.** The stated path is an RFC with benefits vs burden, small modular PRs, and proof that the contributor understands the code. Beating `--mmap`/static placement is the bar set by one collaborator. A single ~3,000-line patch with custom CUDA, CPU helper threads and a scheduler change would very likely get the same "too large" response. The application should present the patch as a research artifact, and upstream only small independent pieces.
- **Reported gains on Qwen3-30B-A3B-class models with A3B-size experts are modest** when the baseline is strong (leloch: +2.1% on Qwen3-30B-A3B; +8.3% on Qwen3.6-35B). Gains are large mainly on models with big experts or skewed routing (Scout, ERNIE, Gemma-4) or on small GPUs.

### Gaps
- No maintainer design preference for an expert cache (scheduler-level vs backend-level vs llama-level) was found in any thread.
- #27861 comments after 2026-09-14 were not visible in the summary.
- leloch's gpt-oss-120b/20b "matrix rows" from the June branch were not read, because blobs from his fork were not fetched for `v3-expert-cache`.
- No Miltos22 successor PR was found by commit-subject scan; one titled differently could have been missed.

## Q3. ik_llama.cpp: hybrid MoE features, decode speed against mainline, Blackwell support, and whether it belongs in the race

### Takeaway
ik_llama.cpp turns on several hybrid features by default:
- fused MoE up/gate (`fused_moe_up_gate = true`)
- offload-only-active-experts for prompt processing (`only_active_exps = true`)
- graph reuse (`graph_reuse = true`)

It also has:
- `-ot` tensor overrides and `-ncmoe`. Since 2026-08-07 (#2262), `-ncmoe` puts the **last** N layers' experts on CPU with one GPU; mainline puts the first N.
- optional run-time repacking (`-rtr`) and auto-fit (off by default)
- MXFP4_R8 CPU repacking (July) with faster AVX2/AVX512 GEMM (September) — these help prompt processing, not batch-1 GEMV
- chunked CPU experts
- by default, weights overridden to CPU are dropped from mmap into pinned host memory

It has **no Blackwell-specific CUDA code**: no native FP4 MMA and no sm_120 tuning. It builds for the native architecture by default.

No 2026 source shows ik beating mainline on hybrid batch-1 token generation:
- ik's own #2262 table shows TG unchanged by its placement change.
- A 2026 user report (#1699) had ik slower, attributed to configuration.

Mainline has Blackwell-tuned MMVQ switch points and native FP4 MMQ. ik should appear as a secondary llama.cpp-family entrant, measured with its defaults and a tuned `-ncmoe`/`-ot`, with the better of the two reported as "best llama.cpp-family". Mainline should stay the primary baseline.

### Cited Findings
**Fork status and defaults**
- The README says ik "started as a fork … in June of 2024 and was last synced with upstream in August of 2024".
- README warnings:
  - For hybrid MoE, "**do not use -rtr** … matrix multiplications with these tensors to be **always done on the CPU** … k-quants … do not have CUDA row-interleaved implementation".
  - With split mode graph plus partial offload, "try adding `-cuda graphs=0`".
- Source: [ik README @ adce16f](https://github.com/ikawrakow/ik_llama.cpp/blob/adce16f50f454da2aa2587801c885ffba7805c72/README.md).
- Defaults in `common/common.h`: `fused_moe_up_gate = true`, `graph_reuse = true`, `only_active_exps = true` ("offload only active experts (relevant only for hybrid CPU/GPU)"), `repack_tensors = false`, `fit = false`, `n_threads = cpu_get_num_math()`.
- Flags: `-ger` grouped expert routing, `-muge` merge up/gate experts, `-smgs`, and `-cuda` params including `fusion`, `offload-batch-size`, `mmq-id-size` and `graphs`.
- Source: [ik common.h/common.cpp @ adce16f](https://github.com/ikawrakow/ik_llama.cpp/tree/adce16f50f454da2aa2587801c885ffba7805c72/common).

**Recent hybrid-relevant commits**
- #2262 "Better placement of MoE tensors with -ncmoe and 1 GPU" (2026-08-07). It moves CPU experts to the last N layers.
- On an RTX 3090 with Qwen3.6-35B-A3B Q8_0 at `-ncmoe 22` (ik PR vs ik main), "About 10% better PP, TG within the noise":

  | N_KV | TG t/s, PR | TG t/s, main | PP t/s, PR | PP t/s, main |
  |---|---|---|---|---|
  | 0 | 88.54 | 89.07 | 2233.82 | 2003.82 |
  | 8192 | 86.69 | 86.83 | — | — |
  | 30720 | 81.30 | 81.58 | — | — |

- Source: [ik PR #2262](https://github.com/ikawrakow/ik_llama.cpp/pull/2262); [commit da2293d](https://github.com/ikawrakow/ik_llama.cpp/commit/da2293ded3b89459c0818747ef6f87e41c546549).
- #2316 "CUDA graphs improvements" (2026-08-24) gives each graph a unique ID and adds more thorough node checks. Gain is "minor … only when TG is above 100 t/s or so". It also fixed a hybrid DS4 mask-contiguity bug that caused garbage — [ik PR #2316](https://github.com/ikawrakow/ik_llama.cpp/pull/2316).
- CPU kernel work:
  - #2202 "Chunked experts (CPU)" (2026-07-30)
  - #2196 "MXFP4_R8" (2026-07-28), with AVX2 and AVX512 implementations
  - #2471 "Better MXFP4_R8 GEMM on vanilla AVX2" and #2465 "Slightly faster Q4_0_R8 and MXFP4_R8 on AVX512_VNNI" (2026-09-17)
  - Sources: [commit 74cccfd](https://github.com/ikawrakow/ik_llama.cpp/commit/74cccfd71d074090ae855ea8ad64f1797b793810); [commit f0f6ae4](https://github.com/ikawrakow/ik_llama.cpp/commit/f0f6ae4bb0650c1e1cc49046bc6806926dd1c40d); [commit 5ae7ce9](https://github.com/ikawrakow/ik_llama.cpp/commit/5ae7ce99543c629b1c9a5ea9854693c6e361846c); [commit 0dcc1a5](https://github.com/ikawrakow/ik_llama.cpp/commit/0dcc1a5e24b4690ec5f428ce71657fbb8366f3eb).
- #2444 (2026-09-15): "With -ot overrides to the CPU the loader drops mmap so the overridden weights land in pinned host memory". The new `GGML_CUDA_NO_PINNED_WEIGHTS` keeps them mmapped — [commit 19dfb71](https://github.com/ikawrakow/ik_llama.cpp/commit/19dfb71bb8386d421f237030f0956de3e944d391).
- #2101 `--prefetch-experts` streams mmapped experts into the page cache with `MADV_POPULATE_READ` (2026-07-11) — [commit 6a909f4](https://github.com/ikawrakow/ik_llama.cpp/commit/6a909f4ff656b79d76fb64e7a9a124f35be6ff33).

**Blackwell**
- ik's `ggml/src/ggml-cuda/common.cuh` has no `BLACKWELL` or FP4-MMA definitions. The only Blackwell-named commit in its history is "Add ARM Grace Blackwell (NVIDIA DGX Spark) support (#922)" (2025-11-09). CUDA architectures default to `native` — [ik repository](https://github.com/ikawrakow/ik_llama.cpp).
- Mainline, by contrast, has "Blackwell specific switch point[s]" tuned on an RTX 5090 (#26079) and native FP4 W4A4 MMQ — [LC commit 2b562109](https://github.com/ggml-org/llama.cpp/commit/2b5621094ef383cdcd8428ef6d22efe5df976532).

**ik vs mainline evidence**
- ik issue #1699: GTX 1660 Super 6 GB + Ryzen 7 5800X, Qwen3.6-35B-A3B IQ4_XS with `--cpu-moe`. Mainline PP 239.81 / TG 15.69 vs ik PP 118.79 / TG 10.40.
  - ikawrakow blamed VRAM overflow ("the giant compute buffer is eating all dedicated VRAM") and `-rtr` ("experts that don't have CUDA implementation when run-time-repacked").
  - He recommended dropping `-rtr` and using `-b 8192 -ub 8192` and `-wgt 1`. No resolution is reported.
  - Source: [ik issue #1699](https://github.com/ikawrakow/ik_llama.cpp/issues/1699).
- ik's wiki comparisons with mainline date from July 2024 and January 2025 only — [ik wiki](https://github.com/ikawrakow/ik_llama.cpp/wiki).
- A June 2026 guide (updated 2026-08-14) calls ik "worth evaluating" for "CPU or hybrid CPU/GPU MoE" but gives no numbers — [carteakey.dev](https://carteakey.dev/blog/local-inference/local-llm-optimization/).
- Earlier data from the prior note: 2025 hybrid Qwen3-235B TG was "similar across both"; ik's gpt-oss gains were full-GPU (+15–25%) and CPU-only (~1.5x at 25k context) — [ik Discussion #758](https://github.com/ikawrakow/ik_llama.cpp/discussions/758).

### Inferences
- **Mechanisms are nearly identical.** ik's hybrid batch-1 decode uses the same host-sequenced scheduler model as mainline. Its default-on fused MoE and graph reuse cut per-token overhead. Its CPU-side kernels matter little for bandwidth-bound expert GEMV with attention on the GPU. A small ik advantage (a few percent) on CPU-side overhead or a small disadvantage on Blackwell GPU kernels are both plausible. Only a measurement on the race machine can settle it.
- **Fair-race protocol for ik:**
  - Use its defaults: `-fmoe` and `-ooae` are on; do not pass `-rtr`.
  - Use explicit `-ot` regexes so the same layers' experts are on CPU as in mainline. ik's `-ncmoe` now selects the last N layers.
  - Match `-c`, `-ctk/-ctv`, `-ub`, threads and load mode. Set `-cuda graphs=1`, the default.
  - Report `max(mainline, ik)` as the llama.cpp-family baseline at equal GPU memory.
  - If ik is slower on sm_120, say so, since it lacks Blackwell tuning.
- **ik does not replace mainline as the primary baseline.** Mainline is the upstream target for the patch, has Blackwell-specific tuning, and is what reviewers and papers compare against.

### Gaps
- No credible 2026 same-hardware hybrid TG comparison between ik and mainline was found for gpt-oss-120b, gpt-oss-20b, Qwen3-30B-A3B or Qwen3.6-35B-A3B, and none at all on Blackwell. A WebFetch summary of ik Discussion #1663 produced numbers too garbled to use.
- Whether ik compiles cleanly for sm_120 with CUDA 12.8/13.x (MXFP4 kernels) was not verified.
- r/LocalLLaMA threads were not reachable through search.

## Q4. What is the best-practice mainline configuration today for batch-1 MoE offload, and are there known performance regressions in recent builds?

### Takeaway
For an equal-GPU-memory race, the strongest defensible mainline configuration fixes placement and memory explicitly rather than trusting `--fit`:
- `-ngl 99` with `--n-cpu-moe K`, or an `-ot` expert regex, choosing the smallest K that fits the budget.
- `--fit off` and a pinned `-c`. `--fit` is on by default and shrinks context first.
- `-fa on`, matched `-ctk/-ctv`, and default `-ub 512 -b 2048` held equal, since ub sizes the compute buffer.
- `-t` equal to the physical cores on the GPU's NUMA node, with pinning.
- `--load-mode none` or `mlock`. `--no-mmap` was deprecated on 2026-09-09.
- CPU repacking left on (the default).
- CUDA graphs left at default, one graph per split. `GGML_CUDA_GRAPH_OPT=1` is an optional, test-first extra.
- Default CUDA fusions left on.

On Blackwell, build with CUDA 12.8/12.9 and `-DCMAKE_CUDA_ARCHITECTURES=120` (becomes `120a`), with `GGML_CUDA_FORCE_CUBLAS=OFF`, and avoid IQ1_S/IQ2_S/IQ3_S quants.

Regressions in recent builds are not decode-relevant:
- The fit-context change was reverted on 2026-09-26.
- The tiled k-quant GEMM (2026-09-26) is gated to 8 or more rows.
- An RDNA3 MMQ change was reverted.
- The main known hazards are Blackwell toolkit and compiler bugs, plus fusion rounding differences.

### Cited Findings
**Flag semantics and defaults on master**
- `-lm/--load-mode` defaults to `auto` ("mmap, unless a device does not support it"). Choices: none, mmap, mlock, mmap+mlock and dio — [arg.cpp @ master](https://github.com/ggml-org/llama.cpp/blob/4da6337767f973e2b4d0797e5b323d77d8565e4a/common/arg.cpp).
- `--mmap/--mlock/--dio` were "officially deprecate[d]" on 2026-09-09 (#28334) — [commit 14a9d09f](https://github.com/ggml-org/llama.cpp/commit/14a9d09f75683c94c2c4f229efe54670d4209089).
- `-ncmoe/--n-cpu-moe N` keeps "the MoE weights of the first N layers in the CPU". `-ncffn` is the dense-model analogue. `llama-bench` has `-ncmoe` (default 0) — [server README](https://github.com/ggml-org/llama.cpp/blob/4da6337767f973e2b4d0797e5b323d77d8565e4a/tools/server/README.md); [llama-bench README](https://github.com/ggml-org/llama.cpp/blob/4da6337767f973e2b4d0797e5b323d77d8565e4a/tools/llama-bench/README.md).
- From the prior note: `--fit` is on by default, `--fit-target` is 1024 MiB, `--fit-ctx` is 4096; fit reduces context first and puts the experts of the last layers on CPU — [llamacpp_baseline_history.md sources: common/fit.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/fit.cpp).
- `GGML_OP_OFFLOAD_MIN_BATCH` defaults to 32 for CUDA. Below that, ops with host weights stay on CPU, so decode experts run on CPU — [ggml-cuda.cu @ master](https://github.com/ggml-org/llama.cpp/blob/4da6337767f973e2b4d0797e5b323d77d8565e4a/ggml/src/ggml-cuda/ggml-cuda.cu).
- CUDA graphs with `n-cpu-moe` run one graph per split since #18934 (2026-01-24), giving +4–10% TG (from the prior note). `GGML_CUDA_GRAPH_OPT` is still opt-in — [PR #18934](https://github.com/ggml-org/llama.cpp/pull/18934); [ggml-cuda.cu @ master](https://github.com/ggml-org/llama.cpp/blob/4da6337767f973e2b4d0797e5b323d77d8565e4a/ggml/src/ggml-cuda/ggml-cuda.cu).

**Community guide** (carteakey.dev, updated 2026-08-14)
- Recommended flags:
  - `--fit on --fit-ctx 65536 --fit-target 512` or `-ot ".ffn_(up|down|gate)_(ch|)exps=CPU"`
  - `-ctk q8_0 -ctv q8_0`
  - `--batch-size 1024 --ubatch-size 512`
  - `--flash-attn on`
  - `--no-mmap` ("removes page-fault jitter in hybrid runs")
  - `taskset` P-core pinning with `--threads 10`
  - "Leave `GGML_CUDA_GRAPH_OPT=0` (test before enabling; can regress on varying context)", because it can cause "intermittent OOM at long prompts"
- Environment notes: disabled XMP/EXPO was "the most common culprit for MoE TG underperformance", and power-profiles-daemon left a node "20–30% below its TG baseline".
- Source: [carteakey.dev](https://carteakey.dev/blog/local-inference/local-llm-optimization/).
- The official gpt-oss guide example is `--n-cpu-moe 32 -ub 4096 -b 4096` for 16 GB. The large ub is a prefill choice that costs VRAM — [Discussion #15396](https://github.com/ggml-org/llama.cpp/discussions/15396).

**Blackwell build**
- Recommended: CUDA 12.8, `-DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_CUDA_FORCE_CUBLAS=OFF`, and a clean build directory. The same article reports CUDA 13.1 MMQ segfaults on Blackwell (b8240, March 2026) — [zenn.dev](https://zenn.dev/toki_mwc/articles/rtx5090-blackwell-cuda-toolkit-trap-llama-cpp?locale=en).
- CMake rewrites 120 to 120a for the FP4 instructions — [CMakeLists.txt @ master](https://github.com/ggml-org/llama.cpp/blob/4da6337767f973e2b4d0797e5b323d77d8565e4a/ggml/src/ggml-cuda/CMakeLists.txt).
- IQ1_S/IQ2_S/IQ3_S are miscompiled with nvcc 13.2 on sm_120 (#28784, unmerged) — [PR #28784](https://github.com/ggml-org/llama.cpp/pull/28784).

**Recent reverts and changes affecting performance** (master log since August)
- "Revert 'Change max context length for auto-fitting with unified KV (#28849)'" (#29437, 2026-09-26)
- "Revert 'CUDA: size routed MoE MMQ N-tiles from typical expert width on RDNA3 (#24546)'" (#28551, 2026-09-07); re-landed gated as #28552/d4abd57 (2026-09-09)
- "Revert 'common: share thread pools when `n_threads` differ (#27138)'" (#27337, 2026-08-19)
- "Revert 'tensor-split meta backend fixes (#26502)'" (#27433, 2026-08-20)
- Source: [LC commits](https://github.com/ggml-org/llama.cpp/commits/master).

**Decode-neutral or decode-positive changes**
- The tiled k-quant GEMM is gated to 8 or more rows and is a net loss for GEMV only if forced — [tiled.cpp @ master](https://github.com/ggml-org/llama.cpp/blob/4da6337767f973e2b4d0797e5b323d77d8565e4a/ggml/src/ggml-cpu/tiled/tiled.cpp).
- The mmvq Q4_K/Q5_K unpack speed-up (#26705) and the MoE fusions (#25952, #28432) — [commit 73ab7599](https://github.com/ggml-org/llama.cpp/commit/73ab7599b553c03f6f5d2db24a18ad76f2eb36a3); [commit 3466812d](https://github.com/ggml-org/llama.cpp/commit/3466812d1f06728effe7c0f3c0671117f461672d).
- Earlier scheduler history (prior note): #20793 sync reduction was merged 2026-06-26 and reverted 2026-06-30; #17795 (fewer syncs, March 2026) remains — [PR #25138](https://github.com/ggml-org/llama.cpp/pull/25138); [PR #17795](https://github.com/ggml-org/llama.cpp/pull/17795).

### Inferences
- **Recommended mainline arm** (placeholders in caps; inference built from the findings above):

  ```
  llama-bench -m MODEL -ngl 99 -ncmoe K -fa 1 -ctk f16 -ctv f16 -ub 512 -b 2048 \
      -t PHYS_CORES --cpu-strict 1 -p 0 -n 256 -d DEPTH -r 10 -o jsonl
  ```

  - Use `--load-mode none` in `llama-server`, or `llama-bench --mmap 0`.
  - Choose K from `llama-server --fit off -c C … --n-cpu-moe K` so that the exit memory breakdown "self" is within the budget.
  - Run once with `GGML_CUDA_GRAPH_OPT=1` and keep it only if it wins.
  - Also run the `-ot` variant that sends the last K layers' experts to CPU (matching `--fit` and ik) and keep the faster one.
- **Fairness points specific to this patch:**
  - Keep `--repack` on for the baseline. leloch's cache needed repack off, so the "optimized" baseline is the repack-on arm.
  - Give the baseline the same `-ub`: a smaller ub frees compute-buffer VRAM that the baseline could spend on more GPU experts.
  - Hold KV type and context equal, because KV VRAM competes with experts.
- **gpt-oss on Blackwell:** MXFP4 CPU experts use the MXFP4 CPU repack (mainline #19738, from the prior note). GPU experts use MMVQ. Neither depends on the native-FP4 prefill path, so no special Blackwell flag is needed for batch-1 decode.

### Gaps
- No published September 2026 mainline hybrid TG numbers on an RTX 5090 or RTX PRO 6000 with a stated CPU and DRAM were found for gpt-oss-120b, gpt-oss-20b or Qwen3-30B-A3B. The Hardware Corner 5090 figure of 8–10 t/s from the prior note is low-confidence.
- Issue search pages on GitHub are blocked to WebFetch by robots.txt, so issues reporting `--n-cpu-moe` decode regressions filed in September 2026 could not be enumerated. Only PR commit subjects were scanned.
- Whether `GGML_CUDA_GRAPH_OPT=1` helps or hurts single-GPU `-ncmoe` decode on Blackwell is unmeasured.

## Q5. Would upstream plausibly merge a scheduler change that overlaps independent CPU and GPU splits? Is there an existing issue or PR?

### Takeaway
No existing issue or PR proposes running a CPU split concurrently with an independent preceding GPU split. The nearest items are:
- Discussion #16621 (Oct 2025, unanswered), about using idle-GPU periods for other requests.
- #28874 (open), which reorders input copies so the host does not block at split boundaries.
- #17795 (merged) and #20793 (reverted), which removed some syncs.

Maintainers' current scheduler agenda is correctness: the sanitizer (#26167, with review comments from ggerganov and JohannesGaessler), stress tests (#27258), `GGML_SCHED_DEBUG_REALLOC` in CI, and the #26040 race fix that *added* a sync. The draft upstream PR itself says stock `--n-cpu-moe` never produces independent CPU splits, so the overlap helps only graphs upstream does not have (split-expert graphs). It also adds public API: `ggml_backend_sched_set_overlap_cpu` and a new `ggml.h` tensor flag, which fall under the "higher bar" rule.

Merge plausibility today is therefore low as a standalone PR. It rises if the change:
- is posted as an RFC or discussion first;
- is shown clean under the #26167 sanitizer and the #27258 stress tests;
- avoids new public API (for example, detecting independence automatically);
- is paired with an upstream consumer, such as #27861's dual-chain graph, which has exactly the independent CPU split the overlap targets.

### Cited Findings
- Harshith's draft PR, "Honest scope":
  - "Stock `--n-cpu-moe` does not benefit. Its CPU splits consume the router output and the normalized hidden state of the preceding GPU split, so they are never independent."
  - Measured "+8 to +13 % decode speed with the overlap on" on an A10 with the patch's expert cache.
  - The patch adds `GGML_TENSOR_FLAG_SPLIT_BEFORE = 32` to `ggml.h` and `ggml_backend_sched_set_overlap_cpu` to `ggml-backend.h`.
  - Source: `runtime/UPSTREAM_PR_DRAFT.md` and `runtime/ggml-sched-overlap.patch` (local files).
- Discussion #16621 "Boost performance by using idle GPU periods" (karambaso, 2025-10-16). It proposes filling idle periods with other requests in multi-device layer splits, has "zero comments" and is "Unanswered" — [Discussion #16621](https://github.com/ggml-org/llama.cpp/discussions/16621).
- #28874 (open) targets host/GPU serialisation at split boundaries when `n_copies == 1` "whenever tensor overrides are used". It changes only copy order and does not run splits concurrently — [PR #28874](https://github.com/ggml-org/llama.cpp/pull/28874).
- Scheduler sanitizer #26167 (am17an, open), as summarised: it detects happens-before races between backends. ggerganov wants "a minimal internal scheduler callback API". JohannesGaessler proposed "moving backend scheduler to dedicated file" and objected to global state — [PR #26167](https://github.com/ggml-org/llama.cpp/pull/26167).
- Scheduler stress tests and `GGML_OP_SLEEP` #27258 (aendk, open) — [PR #27258](https://github.com/ggml-org/llama.cpp/pull/27258).
- CI now runs with `GGML_SCHED_DEBUG_REALLOC=1` (#29514, ggerganov, 2026-09-27) — [commit 7fb2b082](https://github.com/ggml-org/llama.cpp/commit/7fb2b082ce762cfc8b19f3258a40f07c1fd075ab).
- #26040 (2026-08-20, from the prior note): "splits without input were running concurrently with other splits, while potentially reusing memory the other split is accessing". The fix was to "only sync when split has no inputs" — [commit 8497981](https://github.com/ggml-org/llama.cpp/commit/8497981).
- #20793 "sched : reintroduce less synchronizations during split compute" was reverted in #25138 because "perf regressions are not restriced to EOL AMD HW" — [PR #25138](https://github.com/ggml-org/llama.cpp/pull/25138).
- `ggml.h`/ggml-backend changes are periodically synced with the standalone ggml repository ("sync : ggml (#28379)", 2026-09-04) — [LC commits](https://github.com/ggml-org/llama.cpp/commits/master).
- CONTRIBUTING: "New CLI or public API additions carry a **higher bar** than internal changes - justify why an existing mechanism doesn't suffice"; "Bug-fix PRs must include a reproducible issue and a regression test" — [CONTRIBUTING.md @ master](https://github.com/ggml-org/llama.cpp/blob/4da6337767f973e2b4d0797e5b323d77d8565e4a/CONTRIBUTING.md).

### Inferences
- **The overlap is exactly the kind of change the sanitizer is meant to police.** It lets a CPU split read memory while the GPU split's stream runs. Maintainers will want proof that the CPU split's inputs are not aliased or reused by the gallocr while the GPU split runs; #26040 was exactly such a reuse race. Running the patch under #26167 (`GGML_SCHED_SANITIZE=1`, if the PR is fetched) and passing #27258's stress tests would be the most persuasive evidence.
- **Without an upstream graph that produces independent CPU splits, maintainers have no in-tree beneficiary.** The two realistic routes are:
  - (a) comment on #27861, whose dual chain places a CPU miss chain that may be independent of the GPU hit chain, and offer the overlap as a follow-up;
  - (b) open an RFC or discussion framed around the sanitizer-validated invariant, not a PR.

  CONTRIBUTING also tells contributors to comment on an existing PR for the same change rather than duplicate it.
- **Positioning for the application.** The overlap plus the GPU-signalled mailbox is the novel systems contribution, so a small, test-backed, sanitizer-clean upstream RFC is a credible fellowship talking point even if it is not merged by 27 Oct.

### Gaps
- #26167's code was not read in detail, so whether it would flag the overlap as a race (true positive or false positive) is untested.
- No maintainer statement specifically on CPU/GPU split concurrency was found.
- The #26040 PR thread (Ruben Ortlam) was not readable, so whether overlap loss was discussed there is unknown.
