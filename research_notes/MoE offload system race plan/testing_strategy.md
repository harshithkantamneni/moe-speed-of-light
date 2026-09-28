# Testing strategy for option C: improve the expert cache, then race it fairly (as of 28 Sep 2026)

Scope: the system under test is `R:runtime/llama.cpp-expert-cache.patch` (sha256 465361ad…, base llama.cpp 2145525a, applies to
master 4da6337) in mailbox mode, plus the scheduler part that `R:runtime/ggml-sched-overlap.patch` duplicates. The plan
covers one week of optimization and one race day on a rented Blackwell machine (most likely the whole-machine Vast RTX 5090,
Ryzen 9 9950X, ~255 GB DDR5, container, no clock locking), racing mainline llama.cpp, ik_llama.cpp, FreeToken, Pipelined
sharding, and possibly KTransformers and a llama.cpp community cache on gpt-oss-120b MXFP4 and Qwen3-30B-A3B BF16.

**How this note was built.** I read the five race-plan notes in this folder (plus `race_machine.md`), the full patch, the
four Python tests, GPU jobs 009, 011, 018, 025/038 and 055 and their results, `gpu/runner.sh`, and the relevant CUDA
dispatch code in the patched llama.cpp tree kept in the scratchpad. I ran three cheap checks on CPU (Appendix A). No GPU
was used, nothing was rented, and no repository file was changed apart from this one.

**Path conventions** (as in `our_system_baseline.md`): `R:` = `/home/claude/moe-speed-of-light` (main), `G:` =
`/home/claude/gpu-branch` (gpu branch), `S:` = this session's scratchpad. `S:llama.cpp` is a patched llama.cpp tree
(HEAD 4d86b2f "ec-bench: --no-mmap") with a CPU-only build in `S:llama.cpp/build-cpu`, and `S:tiny/` holds 4-layer,
32-expert tiny GGUFs. My check files are in `S:tstrat/`. The scratchpad is session-scoped, so copy anything worth keeping.

---

## 0. Summary

**The plan in one paragraph.** Put every optimization behind an `LLAMA_EC_*` switch that is off by default, so that A and B
run interleaved inside one process on one loaded model. Gate every patch change on a free CPU tier in this container and on
a ~20-minute GPU smoke job through the git runner. Accept an optimization only through a pre-registered paired-bootstrap
stop rule and an exactness gate. Exactness is bit-identical output for arithmetic-neutral changes, and "within 3× the
measured placement noise floor" otherwise. Freeze and pre-register before race day. On race day, prove every entrant was
held to the same measured NVML peak, core set, token IDs and output length, and try to reproduce each competitor's own
published number. Testing costs about 19 GPU-hours and $14–19 in total, which leaves room for development inside the
$25–40 budget (§6).

**What reading the code and the CPU checks turned up.** These drive the priorities:

1. **BF16 is the one race format the cache has never run, and the negative-id handling does not cover it.**
   - The patch guards expert id −1 only in the quantized `mul_mat_vec_q` kernel (`mmvq.cu`) and in `add_id`.
   - On NVIDIA, `ggml_cuda_mul_mat_id` sends BF16/F16/F32 expert matrices to `mul_mat_f` with ids, not to mmvq. Its
     `has_ids` kernel matches `id_row[k] == expert_idx` per expert, so a −1 row is **never written** and keeps stale
     memory. If the shapes do not qualify for that kernel, the call goes to the synchronous fallback instead, whose
     `GGML_ASSERT(ids_to_sorted_host.size() == ne_get_rows)` **aborts** on −1.
   - The CUDA fusion pass may also route the exact up, gate and GLU triple that `path()` builds to
     `ggml_cuda_mul_mat_vec_f` with ids. That kernel computes `x += channel_x*stride_channel_x` with `channel_x = −1` and
     has no guard (`mmvf.cu` lines 30–46), so it **reads out of bounds**.
   - In mailbox mode the WAIT kernel discards exactly those rows. The output is therefore probably right, but out-of-bounds
     faults are possible, and split-graph mode would add the garbage rows.
   - All A10 runs used mmvq types (MXFP4, Q4_K_M, Q8_0), and the test-backend-ops EC cases cover only Q8_0, Q4_K, Q6_K
     and MXFP4.
   - This is the first GPU test to run (T01).
2. **A suspected stale-payload window in the mailbox, found by reading the code and not reproduced.**
   - `helper_main` reads REQ = s and then calls `helper_serve`, which reads LAYER, IDS and X from the mailbox without
     checking again that the request is still s.
   - Suppose a helper is descheduled between the two reads while the GPU finishes s and writes s+1's payload, and
     request s+1 has more CPU experts than s. The helper then derives a larger phase-1 chunk count for seq s. It claims
     phantom chunks of s and writes `h_act` rows for s+1's experts from s's quantized input. Those writes can land after
     s+1's own phase-1 writes, which would give **silent, non-deterministic wrong output**.
   - A CPU harness with injected delays (T06) settles it. A candidate fix is to bail out if `RESP == s` after reading the
     payload.
3. **Silent fallbacks can swap in a different system without any error:**
   - a GGUF with merged `ffn_gate_up_exps` loses mailbox mode;
   - `policy=static` without `LLAMA_EC_INIT` leaves the slots **empty**. I verified this: the CPU run below reports
     "0.000 served on the GPU";
   - unreadable `LLAMA_EC_ALLOC`/`INIT` files are ignored;
   - the header says `LLAMA_EC_PACED` defaults to 1, but the code defaults to 0.

   A "system identity" gate on every stats JSON (T10) is mandatory. It is the lesson of everett6's finding that PR #27861
   silently did nothing.
4. **The tools cannot yet produce what the race needs:**
   - `llama-ec-bench` dumps only the top-5 logits, so KLD cannot be computed;
   - it times only teacher-forced decode, with no timed free-running mode;
   - it sets `n_ubatch = n_prefill`, which inflates VRAM and breaks long-context prefills;
   - nothing samples NVML.

   The job setup has problems too:
   - `G:jobs/ec/setup.sh` deletes and fully rebuilds the tree on every patch change;
   - its `sudo nvidia-smi -lgc` will fail silently in a Vast container.
5. **The data sets the A/B design** (Appendix A).
   - In job 018's A/A repeats, the **first configuration in a process ran 1.0–2.2 % slow** on all three models. All
     other configurations repeated within −0.5 % … +0.1 %.
   - The paired per-sequence log-ratio SD was 0.07–2.2 %.
   - Phase 4.5 had no repeats, and the llama.cpp baseline moved by up to 8 % run to run.
6. **Good news: the CPU tier is cheap and strong.**
   - On the tiny models, the cache (C ∈ {1, 3, 4, 8, 16, 32}; static, lru, dfa; `check=1`) gives **bit-identical** top-5
     dumps to the cache-off configuration, and every configuration reports 0 inconsistent ids. That is 14 runs in 26 s on
     2 cores.
   - The AVX-512 MXFP4 helper kernel differs from ggml's `vec_dot` by at most 2.0e-8 of Σ|w·x| over 64,000 rows.
   - All 100 stats JSONs from jobs 02x–03x satisfy the counter invariants exactly.

---

## 1. Risk map: what to test and the test type per area

| # | Area (patch location) | What can go wrong, specifically | Test type | Tier | Tests |
|---|---|---|---|---|---|
| A1 | CUDA negative-id guards (`mmvq.cu`, `add-id.cu`) | Guard misses a path: fused mmvq variants; a new MUL_MAT_ID kernel chosen on sm_120 | Op test vs CPU, plus a same-backend differential and compute-sanitizer | 1 | T01 |
| A2 | Float MUL_MAT_ID paths reached with −1 (`mmf`, `mmvf`, sync fallback) — unpatched | BF16 rows not written; out-of-bounds reads; assert abort (Summary item 1) | Op test on BF16/F16/F32 with −1, all-negative and fused cases, under memcheck | 1 | T01 |
| A3 | CPU negative-id kernels (`ggml-cpu.c`, `ops.cpp`, `repack.cpp`) | Row not zeroed; `add_id` copies bias; `repack.cpp` path never reached in the evaluated configurations | Tiny-model bit-exact runs on CPU | 0 | T03 |
| A4 | Helper math (`helper_serve`, `llama-ec-mx.h`) | Hard-coded SwiGLU-OAI α = 1.702 / limit 7; MXFP4 and Q8_0 block layouts; BF16 `vec_dot`; tail blocks | Kernel differential (CPU); helpers-only model runs vs stock CPU experts (GPU) | 0, 1 | T02, T09 |
| A5 | Mailbox ops (`ec.cu`) | Fence and poll ordering; ctl/REQ sequence; `__ldcv` staleness; timeout merges stale rows | GPU loopback op test with a host "helper"; timeout test | 1, 3 | T07, §3.1 |
| A6 | Helper protocol (chunk claiming, parking) | Stale payload (Summary item 2); phantom chunks; deadlock under preemption; `h_active` toggling | Standalone harness, fake-GPU thread, TSan, injected delays | 0 | T06 |
| A7 | Policy and publication (`post_compute`/`pre_compute`) | Victim or loading-slot bookkeeping; 2-step publication; MAX_ADMIT rotation; divergence from `mosl/ecsim.py` | CPU tiny runs; C++-vs-Python replay from a routing dump; stats invariants | 0 | T03, T10, T16 |
| A8 | Slot copies (`ggml_backend_cuda_zc_copy`) | Unaligned `char` path; torn copy read by a graph (#27861 class) | Byte-exact copy test; NaN-poison canary | 1 | T05, §3.2 |
| A9 | Graph construction (`build_moe_ffn_ec`) | Wrong one of the 4 paths chosen, or silent `nullptr` fallback | Identity gate; path-matrix coverage | 0, 1 | T10 |
| A10 | Scheduler overlap (split mode) and the `compute_splits` lambda refactor (stock path) | Stock path altered; CPU split reads memory the GPU split reuses (#26040 class) | Patch-off identity; overlap on/off bit-identity | 1, 3 | T13 |
| A11 | Context hooks (`sched_reserve`, extra `sched_synchronize`, warm-up) | Environment config leaks into every context (for example a draft model); reload leaks helpers | Lifecycle tests | 3 | T08 |
| A12 | Tools (`ec-bench`) | Teacher-forced timing ≠ served speed; top-5 dumps; ubatch; placement not equal to stock | A/A against stock `llama-bench`/`llama-server`; tool additions (§10) | 1, 3 | T13, T20 |
| A13 | Analysis (`mosl`, `scripts/sol_*`) | SoL without LM-head bytes; arithmetic mean of rates; wrong bound class per entrant | CPU unit tests | 0 | T18 |
| A14 | Race harness | VRAM, threads, tokens, output length, warm-up, order, bandwidth drift | Automated audits and status-block schema | 4 | T15–T19 |

---

## 2. Correctness and exactness

### 2.1 What "exact" means: four classes, pre-registered

| Class | Definition | Where it applies | Pass threshold |
|---|---|---|---|
| **E0 bit-exact** | Identical logits (hash per step) | Same binary and configuration, run twice. Any optimization claimed arithmetic-neutral (batched miss jobs, pinning, huge pages, copy-path changes that keep the residency timeline). Cache vs stock on the CPU-only build. C = E vs stock all-GPU if the fusions match. | 100 % of steps identical |
| **E1 kernel-exact** | Op output vs the CPU reference | test-backend-ops cases; helper kernels | NMSE ≤ 5e-4 (`test_mul_mat_id`'s own tolerance), **and** −1 rows exactly 0.0, **and** non-negative rows bit-identical to stock MUL_MAT_ID on the same backend. Helper dot products: \|Δ\| ≤ 1e-6·Σ\|w·x\| (observed 2.0e-8). |
| **E2 placement-exact** (numerically lossless) | Model-level metrics within a multiple of the measured **placement noise floor** of stock llama.cpp on the race host | Ours vs stock at every budget; any change that moves work between CPU and GPU; cross-engine runs in Tier A formats | Mean KLD ≤ 3× floor. P99.9 KLD ≤ 3× floor. Top-1 disagreement ≤ floor + 0.5 pp. \|ΔNLL\| ≤ max(1.5× floor, 0.1 % relative). 0 non-finite logits. ≥ 90 % of free-running first divergences at a reference top-1/top-2 margin below the P5 margin. |
| **E3 lossy** | Anything worse than E2, e.g. mean KLD ≈ 1e-3 (Q8_0-level) | KTransformers deferral > 0, AMX INT4/INT8, Tier B formats | Reported only as a labelled lossy row, never in the exact headline |

**Noise floor, measured on the race host before any comparison.**
- Reference: stock mainline with experts placed at the medium budget (`-ncmoe n_med`).
- Floor pairs: that reference against `-ncmoe n_tight`, against itself with `GGML_CUDA_MOE_WEIGHTED_REDUCTION=0`, and
  against CPU-only experts (`--cpu-moe`).
- The floor is the maximum over these pairs, per metric and per model.
- The A10 suggests what to expect: cache vs llama.cpp top-1 agreement was 0.981–0.993, and all-GPU vs llama.cpp was
  0.979–0.991. The cache sat inside the floor there, and E2 formalises that.

**Why not bit-exactness against stock?** A cache hit runs on the GPU with Q8_1 activations, while a helper miss runs with
ggml's CPU `vec_dot_type` (Q8_0 for MXFP4, BF16 for BF16). Placement therefore changes the arithmetic by construction
(`fair_race_methodology.md` Q2). E0 is the right bar only where the arithmetic is the same.

### 2.2 Op level

- **T01, the `MUL_MAT_ID_EC` matrix on sm_120** with the CUDA toolkit actually used for the build. Record `nvcc --version`
  in the status block.
  - Add BF16, F16 and F32 to the existing Q8_0/Q4_K/Q6_K/MXFP4 cases.
  - Add id patterns: every-other −1 (existing), **all −1** (a step where every selected expert misses), and a single valid
    id.
  - Add the **fused** variants that `path()` produces: up, gate and GLU with ids, with and without bias (extend
    `test_mul_mat_id_fusion` with `ec_negate_half_ids`).
  - Add `ADD_ID_EC` at the gpt-oss (2880, 128, 4) shape.
  - Run under `compute-sanitizer --tool memcheck` on the EC subset. Run `--tool initcheck` once, as a diagnostic only
    (noisy).
  - The stock `MUL_MAT_ID`/`ADD_ID` suite must stay at 100 % (965/965 on the A10 in job 011).
- **Same-backend differential** (a small ggml program in the smoke job). Run CUDA MUL_MAT_ID with −1 against CUDA
  MUL_MAT_ID on the same inputs with the −1 entries replaced by valid ids. The non-negative rows must be bit-identical and
  the −1 rows exactly 0.0. test-backend-ops compares only against the CPU with an NMSE threshold, which would not catch a
  kernel that returns early for the whole block.
- **Mailbox op loopback (T07 family).** Build a ggml graph with `ec_req`/`ec_wait` and a host thread that plays the helper
  and writes known rows (y_j = j + 0.5).
  - dst rows equal y where cid ≥ 0, g where cid < 0, and 0 when g is absent.
  - An all −1 request leaves ctl and REQ unchanged, and the WAIT is a plain copy.
  - A silent helper makes the WAIT set ERR within `timeout_cycles`. Note that `timeout_cycles` assumes ≤ 2 GHz, so on a
    ~2.9 GHz 5090 boost clock 2000 ms becomes ~1.4 s.
- **`ggml_backend_cuda_zc_copy`**: copy random bytes at sizes {1, 3, 4, 15, 16, 4097, 13.25 MB} and at offsets that hit
  each of the int4/int/char branches. The readback must be byte-exact. The real expert strides happen to be 16-aligned, so
  the `char` path is otherwise untested.
- **T02, helper kernels (CPU, runs in this container)**: `ec_mx_dot_avx512` against ggml `vec_dot` for MXFP4 × Q8_0,
  including tails (n = 96, 2880) and extreme e8m0 scales. Add the BF16 `vec_dot` path the helpers use for Qwen3 BF16, and
  a check of the helpers' scalar SwiGLU/SwiGLU-OAI against ggml's `swiglu`/`swiglu_oai` ops on random inputs, which is E1
  at relative 1e-5.

### 2.3 End to end: our system against stock llama.cpp

1. **T13, patch-off identity.**
   - Compare the patched build with every `LLAMA_EC_*` unset against the unpatched mainline at the same base commit.
   - Required: greedy `llama-server` output bit-identical, and tok/s equal within ±1 % (6 interleaved pairs).
   - This licenses using `llama-ec-bench` with EC off as the "stock decode path" reference, and it guards the
     `compute_splits` lambda refactor.
   - Freeze **one** base commit for ours, mainline and T13. Use 4da6337 or later: the patch applies there, but it has never
     been built there.
2. **Teacher-forced stream (V1).** Use a 4k-token on-policy stream per model: S-arm corpora, plus the chat template.
   - It needs a full log-softmax dump from `ec-bench` (fp16, ~0.4 MB per step for gpt-oss's 201k vocabulary, so ~1.6 GB per
     4k-step configuration).
   - `llama-perplexity --kl-divergence` is **not** a substitute: it evaluates batched prefill, which takes the stock path,
     not the cached decode path.
   - Configurations:
     - ours at three budgets, plus C = 0 (helpers only), plus C = E where the model fits (gpt-oss-20b on the 5090);
     - the noise-floor pairs (§2.1).
   - Metrics: mean and P99.9 KLD, top-1 disagreement, ΔNLL, count of non-finite logits.
   - A **wrong helper constant** shows as KLD far above the floor at every step. A **race** shows as rare, huge per-token
     KLD outliers that are not at near-ties. Report the per-token KLD tail explicitly.
3. **Free-running divergence (V2).** Run 32 prompts × 256 greedy tokens for ours and for stock.
   - Report the exact-sequence match rate and the distribution of first-divergence positions.
   - At each first divergence, record the stock run's top-1/top-2 logit margin at that prefix.
   - Pass: ≥ 90 % of divergences at a margin below the P5 of all margins, and a divergence rate no higher than the
     `-ncmoe n_med` vs `n_tight` floor pair.
4. **T04, determinism.**
   - Batched publication is deterministic: copies take effect exactly two steps later after an event synchronisation, each
     helper row is computed whole by one thread, and there are no atomics in the arithmetic.
   - So two fresh processes with the same configuration must give **E0**.
   - Paced mode (`LLAMA_EC_PACED=1`) publishes whatever has completed and is timing-dependent. Assert that it is off in
     every race configuration.
   - If stock itself is not run-to-run identical on the race host, fall back to "KLD ≤ 1e-6 between runs".

### 2.4 Across engines

Not every engine exposes decode-path logprobs over a supplied stream, so use a test that needs only each engine's output
tokens:

- **Reference-scored greedy outputs.** Each engine generates 256 greedy tokens for each of the 32 race prompts. The
  reference (stock mainline, medium placement, teacher-forced through `ec-bench` with EC off) scores each engine's own
  sequence.
- **Argmax agreement** = the fraction of engine tokens equal to the reference argmax at that prefix. Record the reference
  margin at every disagreement.
- An exact engine disagrees only at near-ties: E2 on the V2 criterion and top-1 agreement within the floor.
- A lossy engine disagrees at large margins. The margin histogram distinguishes the two without any engine-specific
  hooks.
- Where an engine exposes prompt logprobs (the SGLang path for KTransformers, HF for FreeToken), also report
  prefill-path KLD, labelled "prefill path".

### 2.5 Weight identity (V0) and format matching

- **File hashes.** Record the sha256 of every GGUF and safetensors file and the HF revision of every model directory. Pin
  them in the pre-registration. A re-downloaded GGUF with merged `gate_up` would silently disable mailbox mode.
- **Per-tensor identity (CPU, streamed per tensor so it fits in a small RAM).**
  - gpt-oss-120b: dequantize the GGUF MXFP4 experts and HF's MXFP4 blocks and scales to FP32. They must be exactly equal.
  - Qwen3-30B-A3B BF16: GGUF from `convert_hf_to_gguf --outtype bf16` against the safetensors. The raw bytes must be equal
    after name mapping.
  - Report a per-engine dtype table for the **dense** tensors too. If the gpt-oss GGUF stores attention in a different type
    from HF's BF16, the row is Tier B for the dense part and must say so (`race_entrants.md` Q3 gap).
- **Per-engine format rules.**
  - ik_llama.cpp: no `-rtr`, and only types that both forks support.
  - KTransformers: BF16 native backend, deferral 0.
  - Pipelined sharding: the same GGUFs as mainline.
  - FreeToken: HF checkpoints of the same revision.

---

## 3. Concurrency and robustness

### 3.1 Mailbox protocol

- **T06, standalone protocol harness (CPU, Tier 0).**
  - Extract `ec_open`, `ec_claim`, the four-phase `phase()` loop, and the REQ/RESP handshake into a header, so that one
    copy compiles into `llama-expert-cache.cpp` and into a test.
  - A fake-GPU thread plays REQUEST and WAIT: it writes X, IDS and LAYER, publishes REQ, and spins on RESP.
  - H helpers (H ∈ {1, 2, 8, 64}; on this container's 2 cores, 64 means heavy oversubscription) serve requests with random
    m ∈ [0, k], built with ThreadSanitizer.
  - Delay hooks (a `EC_TEST_DELAY(point)` macro that is empty in production) sit at six points: after reading REQ; between
    reading IDS and X; before `ec_open`; after `ec_claim`; mid-chunk; before publishing RESP.
  - Every answered request is checked against a single-threaded golden result.
  - Pass: 0 mismatches in 10^6 requests, 0 TSan reports, and no watchdog timeout (60 s).
  - The delay "after reading REQ" is the targeted probe for Summary item 2. If it fails, fix it (bail out if RESP == s
    after reading the payload, or double-buffer the payload by the parity of seq) and keep the test as the regression
    test.
- **Counter invariants (T10), from every stats JSON:**
  - per cached layer: Σ `miss_hist` = steps; hits + misses = steps·k; misses = Σ m·`miss_hist[m]`;
  - `mb_requests` = Σ_{m≥1} `miss_hist[m]`, which is the "GPU-decided CPU skip": a layer with no miss never reaches the
    helpers;
  - `mb_experts` = Σ m·`miss_hist[m]`;
  - C = 0 layers: `mb_requests` = steps;
  - Σ `mb_count` = Σ `mb_requests`.

  These held exactly in all 100 existing files. A phantom phase-3 publish (Summary item 2) would break the last one.
- **T07, timeout path.** Run gpt-oss-120b with `LLAMA_EC_HELPERS=1` and `LLAMA_EC_TIMEOUT_MS=1`.
  - One helper needs milliseconds for four 13 MB experts, so the WAIT times out.
  - Expected:
    - the process aborts with "a GPU wait for the mailbox helpers timed out" within 5 s of the first decode step;
    - no logits are returned for that step (`check_mailbox` runs in `post_compute`, before `llama_decode` returns);
    - NVML memory returns to the idle baseline within 10 s.
- **Wake-up and parking.** `pre_compute` flips `h_active` between prefill (n_tokens > 1) and decode.
  - Test the first decode step after every prefill: its logits must equal the reference (E2), and its latency is recorded.
    It is 1.77× the median step on the A10.
  - Test a decode → prefill → decode transition every 10 steps for 1,000 steps: no hang, and the helpers' CPU time falls
    to near 0 while prefill runs.

### 3.2 Remapping the cache while a GPU graph still reads it (the #27861 race)

**Why the current design is safe (from the code).**
- `post_compute` runs only after an explicit `ggml_backend_sched_synchronize`, which the patch adds.
- An eviction clears `slot_of[v]` and sets `cpu_of[v] = v` *before* the copy into the victim slot is enqueued.
- The admitted expert keeps `cpu_of[e] = e` until `pre_compute` two steps later has synchronised the copy event.
- So no graph ever maps a slot that is being written.

**What would re-open the hazard.**
- Any "move the policy or admission off the critical path" optimization (§4.5 row 4, and `our_system_baseline.md` §8).
- Pipelined or asynchronous `llama_decode` use.
- Paced mode.

**Tests.**
- **T05, NaN-poison canary.** Add a debug knob, `LLAMA_EC_POISON=1`: `cudaMemsetAsync(0xFF)` on the copy stream fills a
  victim slot with NaN just before its admission copy, and also fills it at eviction if no copy follows.
  - Run LRU with C = k (maximum churn), `MAX_ADMIT=0`, 4 sequences × 192 steps on each race model.
  - Repeat with random routing (a port of llama.cpp #29539's `LLAMA_MOE_RANDOM_ROUTING` into both builds): uniform
    routing is the worst case for churn, and stock with the same seeded routing stays a valid reference.
  - Pass: 0 non-finite logits, and E0 against the same run with the poison off.
  - A slot read mid-copy or after eviction shows up as NaN on the first affected step.
- **Debug assertions** (a `LLAMA_EC_ASSERT=1` build flag):
  - at publication, `cudaEventQuery(events[b]) == cudaSuccess`;
  - at admission, the scheduler has no graph in flight;
  - after publication, `resident`, `loading`, `slot_of` and `cpu_of` are mutually consistent for every layer.
- `LLAMA_EC_CHECK=1` in every smoke run: `check_bad` must be 0. It was used only in job 012 on the A10.
- **Determinism (T04)** is the cheapest broad detector. Any timing-dependent read changes the logits between runs.

### 3.3 Admission during the critical path

The zero-copy copy kernel (16 blocks, on the second stream) runs concurrently with the next step's decode graph and
shares SMs and PCIe with the ec_wait polling and helper writes.

- **Correctness:** T05 and T04 above.
- **Interference, performance:** compare DFA against `policy=static` initialised from the same run's stats
  (`LLAMA_EC_INIT=<stats.json>`), at equal C.
  - Static has no admissions, so the per-token difference, net of the hit-rate difference charged at the measured
    per-expert helper cost, is the admission interference.
  - Also add one layer with C = E−1 to measure the per-layer path cost without misses. These are the controls recommended
    in `our_system_baseline.md` §8.
- **Structural check (nsys, Tier 3):** in steady-state decode, no `cudaMemcpyAsync` H2D on the compute stream other than
  the maps and token inputs, and the host time from `sched_synchronize` returning to the next launch (policy latency)
  is below a stated bound. Add `post_us` to the stats JSON.

### 3.4 Budget edge cases (T09)

| Case | How | Expected | Pass |
|---|---|---|---|
| 0 % | `LLAMA_EC_ALLOC` sets every layer to 0 (mailbox helpers only) | hit_rate 0; every step reaches the helpers | E2 vs stock `--cpu-moe`; invariants |
| 100 % | C = E (gpt-oss-20b on the 5090, since the race models do not fit 32 GB) | `gpu_only` path, no maps | E0 (else E2) vs stock all-GPU; tok/s within 2 % (job 014 found 135.5 vs 133.9) |
| 1 expert per layer | `LLAMA_EC_SLOTS=1` (< k) | Victim exclusion leaves at most 1 resident; no crash | E2; `check_bad = 0` |
| C = k−1, C = k | slots = 3/4 (gpt-oss), 7/8 (Qwen3) | Boundary of "every resident selected, so no victim" | E2; invariants |
| Mixed layers | ALLOC mixing 0, mid and E in one graph | All 3 graph paths in one step | E2; identity gate lists the per-layer C |
| Static without INIT | `policy=static`, no INIT | **Empty cache** (verified on CPU) | Must warn, or the identity gate fails |
| Too large | C with slot bytes above free VRAM | `GGML_ASSERT` "failed to allocate slot memory" | Non-zero exit within 60 s, no hang, NVML back to baseline |
| Budget-to-VRAM map | C in {tight, medium} | Logged "MiB of slots" equals C × expert bytes × L (+ biases) | NVML peak is monotone in C, and Δpeak equals Δslot MiB ± 64 MiB |

### 3.5 Long contexts

- `ec-bench` must accept a context larger than prefill + decode and a chunked prefill (`n_ubatch = 512`), which is §10
  item 3.
- Test: 16k and 32k prompts, then 256 decode steps at that depth, for ours and for stock.
  - Pass: E2 at depth.
  - Check that gpt-oss's 128-token sliding-window layers stay correct beyond the window.
- Equal-VRAM accounting includes KV at the pre-registered maximum context. Qwen3-30B is ~96 KB per token in f16, so
  ~3 GB at 32k, which is 10 % of the card.

### 3.6 Lifecycle: shutdown, reload, concurrency (T08)

- **Create and destroy.** `ec-bench` runs 20 configurations in one process, each creating and freeing a context with
  mailbox on.
  - After each free, `/proc/<pid>/task` returns to baseline (the helpers are joined).
  - RSS and pinned memory grow < 1 % across the 20 configurations.
  - The stats JSON is written at each free.
- **Two models in one process**, gpt-oss-20b then Qwen3: the first model's helpers and mailbox are gone, and the second
  model's outputs are E2.
- **Server concurrency.** Patched `llama-server -np 2`, two clients interleaving requests for 10 minutes.
  - Batched decode (n_tokens > 1) takes the stock path and parks the helpers.
  - Pass: no deadlock, and the greedy outputs of each request are E2 against the same request served alone.
- **The environment applies to every context.** Speculative decoding or a draft model would get its own cache and helpers.
  Assert that the race configuration has none, and document it.

### 3.7 Error paths and silent fallbacks: the identity gate (T10)

A race configuration is valid only if its stats JSON and log show:
- mailbox = 1, with the intended number of helpers;
- mxk = 1 for MXFP4 on the AVX-512 host;
- policy, κ and half-life as pre-registered;
- paced = 0;
- the per-layer C as intended (from ALLOC);
- Σ `mb_count` > 0;
- `check_bad` = 0;
- the §3.1 invariants.

A GGUF with merged `gate_up`, a lost ALLOC file, a missing INIT, or an architecture whose activation falls back
(`swiglu_clamp_exp`) all fail this gate instead of producing numbers for a different system.

---

## 4. Performance A/B harness for each optimization

### 4.1 Design

- **Everything switchable.** Each optimization lands behind an `LLAMA_EC_*` switch that is off by default, recorded in the
  stats JSON, so A and B run in **one process on one loaded model**. That uses `ec-bench --ec "W;A;B;B;A;A;B;B;A;A;B;B;A"`.
  - It removes load-to-load variance, which is costly for a 61 GB pinned load.
  - It keeps the page cache identical for both arms.
  - Changes that cannot be switched (the kernel-node diet) need process-level ABBA with 3 launches per arm, which costs
    ~2× the GPU time.
- **W is a discarded warm-up configuration**, identical to A. Job 018's A/A data show the first configuration in a process
  runs 1.0–2.2 % slow (Appendix A).
- **Block** = one configuration over a fixed set of 8 sequences × 192 decode steps (D and S arms mixed; the same
  sequences in every block). **Pair** = adjacent A and B blocks in ABBA order.
- **Configurations:**
  - primaries P = {gpt-oss-120b, tight; Qwen3-BF16, tight};
  - guards G = {both models, medium; gpt-oss-120b, 0 % (helpers only)}.
  - "Tight" ≈ 25 % of experts and "medium" ≈ 45 %, mapped exactly from the mainline `-ncmoe` values through
    `LLAMA_EC_ALLOC`. For example gpt-oss-120b `-ncmoe 27` → 9 GPU layers × 128 = 1152 experts → C = 32 on each of 36
    layers. Where E·(L−n)/L is not an integer, spread the remainder over layers so that ΣC_i equals the stock
    configuration's expert count exactly.

### 4.2 Statistics

- **Metric.** Decode tokens and decode time per (block, sequence), taken from the raw `step_ms`. The rate per arm is
  ΣN/ΣT over all its blocks and sequences, which is the token-weighted harmonic mean (Hoefler R3). Never average
  per-sequence or per-block rates.
  - Report the first decode step after each prefill separately, since it carries the cold-cache penalty. The A/B primary
    includes it, because it is real system behaviour.
- **Effect.** Speedup = rate_B / rate_A, pooled.
- **Confidence interval.** A hierarchical paired bootstrap, 10,000 resamples: resample pairs, then resample sequences
  within each resampled pair, using the same sequences for A and B. Take the percentile CI.
  - Steps within a sequence are autocorrelated (routing locality, cache state) and are never resampled as independent
    units.
  - Also report each pair's own ratio and a sign test across pairs.
- **Minimum 6 pairs**, since n > 5 is needed for nonparametric CIs (Hoefler).
  - The A10 A/A pairs repeat within about 1–2 % at configuration level (018).
  - So 6 pairs give a 95 % half-width of about ±1 %, and a minimum detectable effect of about 2 %.
  - The expected effects are +5–20 % (`headroom_techniques.md` Q6).
- **Micro-metrics alongside:**
  - Helper service time per request is f + m·c_e, fitted from per-request samples. Add a per-request histogram (m,
    µs) to the stats JSON; today only sums per m are stored.
  - Helper effective bandwidth = CPU-expert bytes / helper busy time, against `llama-ec-cpubench` read bandwidth at the
    same thread count.
  - Hit rate, admissions per token, and O per layer (the decomposition of `our_system_baseline.md` §4.2).

### 4.3 Drift controls without clock locking

1. At session start, 60 s of steady decode (not scored) to bring the GPU and CPU to temperature. Record whether clocks
   are locked or not; on Vast they will not be.
2. ABBA counterbalancing cancels linear drift. The W configuration absorbs first-configuration effects.
3. Samplers per block, at 1 Hz:
   - `nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active`;
   - CPU MHz of the helper cores (`/proc/cpuinfo`);
   - load average and other processes above 5 % CPU (`ps`).
4. Before every block, a 1 GiB read probe at the helper thread count (~0.2 s) plus a pinned H2D probe of 256 MB.
5. **Invalidating a block.** A block is invalid if any of these holds:
   - HW slowdown, HW thermal or SW thermal throttling is active for > 5 % of it;
   - the read probe is outside ±5 % of the session median;
   - a foreign process used > 5 % CPU.

   Re-run the **whole pair**.
6. Plot the A-arm rate against time. A monotone trend > 2 % across a session is reported, and pairs are then analysed only
   within the stable window.

### 4.4 Stop rule (pre-registered; one extension, no open-ended peeking)

- **After 6 pairs:**
  - ACCEPT if every primary has 99 % CI low > 1.01, every guard has 99 % CI low > 0.98, and the exactness gate passes.
  - REJECT if every primary has 95 % CI high < 1.02, or any guard has 95 % CI high < 0.98.
- **Otherwise extend to 12 pairs** and decide at 95 %: accept if every primary has CI low > 1.01 and every guard has CI
  low > 0.98; else record "no demonstrated effect" and leave the switch off.
- **Exactness gate.** E0 (dump hashes identical per step, A vs B, in the same process) for changes declared
  arithmetic-neutral. Otherwise E2 on the 4k-step stream.
  - Any change to the residency timeline (publication delay, admission rule) changes which experts run on the GPU, so it
    changes logits at the E2 level. Declare it as such up front.
- **Also required:** all Tier 1 gates pass, and T06 passes if the helper protocol was touched.
- **Cumulative check.** The final stack is compared against v0, frozen at the start of the week: 6 pairs on the dev host
  (Tier 3) and a 3-pair transfer check on the race host (Tier 4). Only race-host numbers are published, to avoid the
  winner's curse of dev-host decisions.

### 4.5 Per-optimization acceptance

| # | Optimization | Exactness class | "Did it actually apply?" check | Micro-metric that must move | Extra gates |
|---|---|---|---|---|---|
| 1 | Batch each layer's CPU misses into one job (cut the ~21 µs per-expert fixed cost; fewer phase barriers) | E0 (each row's `vec_dot` is unchanged) | New switch recorded in stats | Fitted f (per request) and per-expert fixed cost fall; c_e not worse than +3 % | T06 (protocol changed) |
| 2 | CPU streaming: thread count, pinning, CCD/NUMA placement, huge pages | E0 | `Cpus_allowed_list` of each helper thread equals the plan. Physical cores first: the current `pin i → CPU i+1` reaches SMT siblings beyond 15 helpers on a 16-core part. `AnonHugePages`/`HugetlbPages` in `/proc/<pid>/smaps_rollup` > 0 for the expert pool. THP and hugetlbfs are host settings a container may not be able to change; if so, record "not applicable". | Helper effective bandwidth / read bandwidth ≥ 0.85; thread sweep 1…32 shows the knee (expect well below 30 on dual-channel DDR5) | Default helper count re-derived; the A10's 28 is not carried over |
| 3 | Fewer GPU kernel nodes per layer (fuse `get_rows`/`ec_req`/`ec_wait` into neighbours; upstream fusions) | E2 unless proven E0 | Decode-graph node count and per-step kernel count (nsys) fall by the expected number; the "CUDA graph … reused" log line is still present | O per layer (`our_system_baseline.md` §4.2) falls | T01 re-run if any kernel changed |
| 4 | Move admission fills off the critical copy path (chunking, SeqMoE-style pause flag) | E0 if the residency timeline is unchanged, else E2 | nsys: no copy-engine H2D on the compute stream in steady state | p99 step time and the static-vs-DFA interference (§3.3) fall | **T05 poison canary mandatory** (this re-opens the #27861 class) |
| 5 | Re-derive the admission threshold κ (and half-life) for PCIe 5 and the new per-expert cost | E2 | Stats κ equals the pre-registered κ* | Measured admissions and hits per token within ±2 % of `ecsim` on a routing dump of the same run (T16) | Simulation first on the S traces; A/B only for the chosen κ* |

---

## 5. Race fairness checks

Every check writes a machine-readable record. The report build fails if any entrant × model × budget point is missing one.
That enforces Hoefler R2: explain every subset, and never drop a point silently.

| ID | Check | Procedure | Pass |
|---|---|---|---|
| F1 | **Equal GPU memory** | Device-level `memory.used` via pynvml at 20 Hz (plus `nvidia-smi -lms 50` as a cross-check), from launch through the last measured request, minus the idle baseline taken just before launch. This is device-level because PID-level NVML is unreliable in containers and engines spawn subprocesses. Fit each entrant with its own knob: `-ncmoe`/`-ot`, `LLAMA_EC_ALLOC`, `--moe-cache-size` + `--num-tokens` + `--memory-ratio`, `--kt-num-gpu-experts` + `--mem-fraction-static` + `--max-total-tokens`, `-mva`. Override every auto-sizer, e.g. `--fit off`. | Peak ≤ budget, and ≥ budget − max(256 MiB, 2 %), or the documented granularity limit (one `-ot` tensor). Breakdown recorded: experts / dense / KV at max context / other. Memory back at baseline within 15 s of exit. |
| F2 | **Threads and pinning** | Every entrant runs in the same cpuset (`taskset`). Regime (a): fixed 16 threads on physical cores. Regime (b): each system's best count in the same set, from a sweep. Sample `/proc/<pid>/task/*/stat` (processor field) every 100 ms. | No thread ran outside the set; active-thread count (> 1 % CPU) as configured; ours = helpers + 1 driver thread |
| F3 | **Identical inputs** | Tokenize once with the reference tokenizer and chat template. Feed token IDs where possible; otherwise compare each engine's `prompt_tokens` and echoed tokens. Disable prefix and prompt caching. | Token-ID hash equal per prompt for every engine |
| F4 | **Identical output length** | N = 256 (plus 1024 for one budget), `ignore_eos`, greedy, no repetition penalty or top-p/min-p | Every request produced exactly N tokens (from the engine's usage field and the count of streamed chunks) |
| F5 | **Format and quantization** | V0 hashes and dtype table (§2.5); ik without `-rtr`; KT BF16 native with deferral 0; FreeToken with the same HF revision | Tier A rows proven identical, or the row is labelled Tier B |
| F6 | **Warm-up** | Page cache warmed by reading the files; one discarded request per launch; for dynamic caches (ours, FreeToken, community cache) a fixed 8-prompt warm-up set, then measured. Also 1 cold request per launch, reported separately. | Steady-state and cold numbers both present |
| F7 | **Randomized, interleaved order** | Seeded permutation (seed in the pre-registration) of all (entrant, model, budget) blocks. Each point has 3 fresh launches × ≥ 5 measured requests, spread across the session. | Every point has ≥ 3 launches; median TPOT 95 % CI half-width ≤ 3 %, or extra blocks added (Hoefler's adaptive rule) |
| F8 | **Host bandwidth start and end** | Read and STREAM-triad sweep 1…32 threads, and pinned H2D/D2H, at session start, before every block (the short probe) and at session end | End within ±5 % of start; blocks outside ±5 % of the median re-run |
| F9 | **GPU health** | Vast lists some 5090s at 633–752 GB/s against ~1450 GB/s (`race_machine.md`). Measure a D2D bandwidth probe and gpt-oss-20b all-GPU `llama-bench tg128`. | ≥ 0.85 × 419 t/s (am17an, RTX 5090, fused build) and ≥ 90 % of the Vast-listed `gpu_mem_bw` |
| F10 | **Baseline tuning parity** (AP2) | Mainline: `-ncmoe` vs `-ot` last-K layers, `GGML_CUDA_GRAPH_OPT` 0/1, repack on, `-t` sweep, `--load-mode none`, `--fit off`, and the same `-ub`, `-c` and KV type. Same tuning effort for every entrant, with the hours logged. | Best configuration per entrant recorded, with the search log |
| F11 | **Reproduction of each competitor's own number** | See the table below. Tolerances and rescaling are pre-registered. | Status assigned per `fair_race_methodology.md` Q4 |
| F12 | **Status block complete** | Schema check of: status; commits (paper-era and latest); command lines; equal-VRAM knob value; STREAM and PCIe figures; paper number; pre-registered rescaled number; measured median and 95 % CI; fraction of SoL (pooled bound for FreeToken, per-layer bound for the rest); effort log; author-contact date | All fields present for every entrant, including failed ones (status "does not run" and its log) |
| F13 | **Harness A/A** | Mainline measured by the race driver (client-side timestamps of streamed tokens) against its own server-side `timings`, and ec-bench teacher-forced vs free-running served speed for ours | Driver vs server within 1 %. Teacher-forced vs served difference reported, not assumed zero. |

**Reproduction targets** (F11):

| Entrant | Published number to reproduce | Conditions | Pre-registered tolerance |
|---|---|---|---|
| FreeToken (v0.1.3, plus main 0d652e7 as "current") | 127.1 tok/s, gpt-oss-120b, `--moe-backend offload --moe-cache-auto` (community, RTX 5090 + 128 GB, ~29 GB VRAM, 40.4 % resident) | Offload mode is PCIe- and GPU-bound, so host bandwidth matters little | 0.80–1.25× |
| Pipelined sharding (v2.0.3-mlsys26) | Qwen3-30B-A3B Q4_0 decode, Table 4 cli3 (RTX 5090, 16-core EPYC, 153.6 GB/s): 32.1 at 8 GB, 47.8 at 16 GB | The **speedup over its own llama.cpp baseline**, run with its scripts, since absolute tok/s depends on the host | Its own AE rule: speedup within 90 % of the paper's |
| Community cache (leloch v2, or #27861 + everett6 0006) | leloch: Qwen3-30B-A3B Q4_K_XL +2.1 % (cache on vs off, repack off); everett6: 1.49× (not equal-VRAM) | The relative gain under the author's own configuration | Sign and magnitude within ±5 pp |
| ik_llama.cpp, KTransformers | No published number on comparable hardware (KT's are dual-socket EPYC or Xeon) | – | Status "runs; no comparable published number" |
| Mainline | gpt-oss-20b all-GPU tg128 419 t/s on a 5090 (F9) | Checks the GPU and the build, not the hybrid path | ≥ 0.85× |

---

## 6. Regression tiers, what runs where, and cost

Prices: a dev host with the same CPU as the race host (whole-machine 5090 + Ryzen 9 9950X + 126 GB, $0.51/h unverified or
$0.80/h verified; 62 GB hosts cannot hold gpt-oss-120b's 61 GB of pinned experts). The race host is the 9950X + 255 GB
machine at $0.735/h. Run the runner jobs as `jobs/NNN_name@5090.sh`, since the runner matches "5090" in the GPU name, and
set timeouts inside the job (the runner default `JOB_TIMEOUT` is 100 min).

### Tier 0: CPU only, in this container, on every commit (free, ~10–15 min on 2 cores)

- The existing `tests/`: `test_cachesim`, `test_ecsim_fast`, `test_bound`, and `test_collect_equivalence` (the tiny HF
  models are cached).
- **Patch hygiene:**
  - `git apply --check` of the EC patch against 2145525a and against the frozen race base;
  - the sched patch's hunks equal the scheduler hunks of the EC patch, modulo comments. Only the EC patch is applied,
    because it already contains them and applying both fails;
  - header-documented defaults equal the code defaults (catches the `LLAMA_EC_PACED` mismatch).
- **Incremental CPU-only build** of the patched tree (`S:llama.cpp/build-cpu`), then:
  - **T03, bit-exactness on tiny models.** Cache off against C ∈ {1, k−1, k, E/2, E}, policies static (with INIT), lru and
    dfa, ALLOC mixes, overlap 0/1, `check=1`. Cover Q8_0 (existing), F32 (existing), and new **BF16 and MXFP4** tiny GGUFs.
  - **T02**, the helper-kernel differential, since this CPU has AVX-512 VNNI and BF16.
  - **T06**, the mailbox harness under TSan.
  - gcov over `llama-expert-cache.cpp` (§7).
- **Analysis unit tests (T18):**
  - pooled rate ΣN/ΣT vs the arithmetic mean of rates on synthetic data, including recomputing llama-bench's `avg_ts` from
    `samples_ns`;
  - bootstrap CI coverage on synthetic paired data (~95 % over 1,000 simulations);
  - SoL includes LM-head bytes (the High data-quality issue: `sol_*.py` must match `preregister.py`'s dense + head);
  - the equal-VRAM mapping ΣC_i = (L − n)·E;
  - scorer picks the pooled bound for FreeToken and the per-layer bound otherwise;
  - status-block schema.
- **Result parsers** run on the fetched `gpu` branch: the T10 identity gate and invariants on every new stats JSON, dump
  hash comparison, and the drift flags.
- **T16 once routing dumps exist:** `ecsim.simulate_layer` replay must equal the C++ per-layer hits, misses and admits
  **exactly**, per step. Today the equivalence is shown only in aggregate: hit rates match to the third decimal.

### Tier 1: GPU smoke on every patch change (runner, dev host; ~16–25 min, $0.15–0.33 per run)

| Step | Content | Minutes |
|---|---|---|
| Build | ccache for `nvcc` and C++; keep the build directory between patch versions (today `setup.sh` rebuilds from scratch) | 2–8 |
| Preflight | `nvidia-smi -q`, `nvcc --version`, clock policy **recorded** (not assumed locked), `lscpu` flags, THP state, `asyncEngineCount` | 0.5 |
| T01 | EC op matrix (incl. BF16, fused, all −1) + stock MUL_MAT_ID/ADD_ID, then memcheck on the EC subset | 5–7 |
| Loopback / zc copy | Mailbox op loopback; zc copy byte-exactness | 1 |
| gpt-oss-20b (12 GB) | One process: W; EC off; C = E; C = 8 dfa mailbox ×2 (T04); ALLOC 0 (helpers); LRU C = 4 with poison (T05). 3 sequences × 64 steps each. | 3–4 |
| Qwen3-30B-A3B BF16 (61 GB) | One process: W; C = 32 dfa ×2; LRU C = 8 with poison; ALLOC 0. 2 sequences × 64 steps. | 5–6 |
| Verdict | `verdict.json`: T10 gate and invariants, E0 hashes, finiteness, `check_bad`, and a performance sentinel vs the last accepted build (±5 % **alarm only**, not a decision) | < 1 |

About 12 runs over the week: ~4 GPU-hours, $2–3.2.

### Tier 2: per optimization decision (dev host; 25–45 min each)

The §4 harness, 6–12 pairs × {primaries, guards} × 2 models, run in-process. Exactness comes from the same runs' dumps
(E0), or from a 4k-step E2 stream when needed.

Five optimizations: ~3 GPU-hours, $1.5–2.4.

### Tier 3: pre-race gate, once after the freeze (dev host; ~3 h, $1.5–2.4)

- V1 noise floor and E2 on both models (~30 min).
- V2 free-running (~10 min).
- Robustness battery (~60 min): T07 timeout, T08 lifecycle and the 10-minute server run, OOM, the §3.4 edges, 16k/32k
  long context.
- Calibration hour (~30 min; `our_system_baseline.md` §5.4 and the day-1 measurements in `headroom_techniques.md` Q6):
  - `ec-cpubench` 1…32 threads;
  - static-helper sweep over 3 values of n;
  - all-GPU gpt-oss-20b under nsys;
  - node count per layer.
- Final stack vs v0, 6 pairs (~40 min).
- Upstream-readiness items (T13 and overlap on/off E0 in split mode) if time allows.
- Also worth doing on the dev host: a **competitor dry install**, 2–3 h, $1–2.4. FreeToken JIT, the KT source build for
  sm_120, the pipeshard Linux build, ik and the community cache. It turns race-day blockers into known quantities.

### Tier 4: race day (race host; ~9–10 h in one rental, ≈ $9–11 including storage and downloads)

| Block | Minutes |
|---|---|
| Qualification: F8 sweeps, F9 GPU health, PCIe link, NVML baseline, T01 with the race toolkit(s) | 25 |
| Builds and installs, overlapped with ~270 GB of downloads (GGUF + HF for both models, + Q4_0 for the pipeshard reproduction), and V0 hashes on the CPU | 60–90 |
| Transfer check: final stack vs v0, 3 pairs on one budget per model | 30 |
| Noise floor + short V1 (2k steps) + V2 per entrant | 60 |
| F1 fitting + measured blocks: ≈ 30 points (all entrants × 2 models × 2 budgets, + a 12.5 % point for mainline and ours) × 3 launches × ~3 min, randomized | 270 |
| F11 reproductions | 45 |
| End-of-session repeat of the first block + bandwidth probe | 15 |

### Total

| Tier | GPU-hours | Dollars |
|---|---|---|
| 0 | 0 | 0 |
| 1 | ~4 | 2–3.2 |
| 2 | ~3 | 1.5–2.4 |
| 3 (+ dry install) | ~3 (+2–3) | 1.5–2.4 (+1–2.4) |
| 4 | ~9.5 | 9–11 |
| **Testing total** | **~19–22** | **≈ $14–19** |

That leaves about $6–26 of the $25–40 for interactive development and nsys profiling on the dev host. Keep the dev
host's storage only while it is rented: 250 GB costs about $1.5 a day at $0.20/GB-month.

### Schedule (today is Monday 28 Sep; deadline 27 Oct)

| Dates | Work |
|---|---|
| Mon 28 Sep | Tier 0 build-out on CPU (harness, tiny BF16/MXFP4 GGUFs, analysis tests, pre-registration skeleton) |
| Tue 29 Sep | First dev-host session: Tier 1 on v0, **T01 BF16 first**, then the calibration measurements. Decide the BF16 fix: port the mmvq guard to `mmvf`, and zero unmatched rows in `mul_mat_f`, or skip −1 rows entirely. |
| Wed 30 Sep – Sun 4 Oct | Optimize in the §4.5 order of expected value; Tier 1 per change, Tier 2 per decision (≤ 2 decisions a day) |
| Mon 5 Oct | Tier 3 gate; freeze the patch sha and base commit; tag the pre-registration (thresholds, κ, budgets, predicted numbers, analysis scripts, seed) |
| Tue 6 – Wed 7 Oct | Competitor dry install; re-query Vast offers (the best 9950X/255 GB host allows at most 3 days) |
| Thu 8 – Fri 9 Oct | Race day (Tier 4) |
| Sat 10 – Mon 12 Oct | Analysis and status blocks; **email the competitor authors**, ≥ 2 weeks before posting |
| 17–20 Oct | Reserved for one re-run session (budget buffer), or an author-supplied configuration |

---

## 7. Coverage targets

Line coverage is not the right yardstick for a research runtime with GPU-only branches. The targets are path and
configuration coverage, plus line coverage where CPU tooling can measure it.

- **Graph paths:** 100 % of `build_moe_ffn_ec`'s outcomes are exercised in Tier 1:
  - `gpu_only` (C = E);
  - mailbox C = 0;
  - mailbox cached;
  - split-graph cached (only in Tier 3, because the race does not use it);
  - `nullptr` fallback: warm-up, n_tokens > 1, an unsupported activation.
- **Kernel matrix:** 100 % of {MXFP4, BF16, Q4_K, Q6_K, Q8_0, F16, F32} × {plain, fused} × {some −1, all −1, none −1} on
  sm_120 (T01).
- **Policy space:** 100 % of {static with and without INIT, lru, dfa} × C ∈ {0, 1, k−1, k, mid, E−1, E} on CPU (T03). On
  the GPU, 3 values of C per race model.
- **Protocol:** all 6 delay points × H ∈ {1, cores, 2×cores, 64} in T06. ≥ 90 % line coverage of the extracted protocol
  and `helper_serve` code under gcov.
- **Host code:** ≥ 80 % line coverage of `llama-expert-cache.cpp` in the CPU build (policy, maps, publication, stats,
  ALLOC/INIT parsing), excluding branches that need `copy_backend` or the mailbox. Those are covered by Tier 1 behaviour
  tests instead.
- **Race:** 100 % of entrant × model × budget points carry F1–F4, F6, F8 and F12 records, and V2 results (§2.4).

---

## 8. Example test cases

| ID | Name | Setup | Expected result | Pass threshold |
|---|---|---|---|---|
| T01 | MUL_MAT_ID_EC on sm_120, all types | test-backend-ops `-b CUDA0 -o MUL_MAT_ID_EC,ADD_ID_EC` with the added BF16/F16/F32 types, all −1, single valid id and fused up/gate/GLU(±bias) cases, then under `compute-sanitizer --tool memcheck` | Quantized types pass as on the A10 (18/18). BF16/F16 are likely to fail before the fix: unwritten rows, possible out-of-bounds reads. | All OK at NMSE ≤ 5e-4, −1 rows == 0.0, non-negative rows bit-identical to same-backend stock, 0 memcheck errors |
| T02 | Helper kernel differential | 64,000 random rows; n ∈ {64, 96, 2880, 4096, 5760}; e8m0 extremes; + BF16 `vec_dot`; + SwiGLU(-OAI) vs ggml ops | Accumulation order is the only difference | \|Δ\| ≤ 1e-6·Σ\|w·x\| (observed 2.0e-8); activations relative ≤ 1e-5; 0 non-finite |
| T03 | Tiny-model cache ≡ stock (CPU) | `S:tiny/*` Q8_0, F32, BF16, MXFP4; ec-bench `--host-experts`, 4 sequences × 128 steps; 7+ configurations incl. ALLOC mixes; `check=1` | Same CPU kernels, and x + 0 = x | Dump hashes identical to `slots=0` in all configurations; `check_bad = 0`; ≤ 60 s |
| T04 | Run-to-run determinism | 2 fresh processes, mailbox dfa at tight, 4 sequences × 192 steps, full-logit hashes | Batched publication and per-row helpers are deterministic | 100 % of steps E0 (else ≤ 1e-6 KLD if stock itself is not E0) |
| T05 | Poison canary under maximum churn | `LLAMA_EC_POISON=1`, LRU C = k, `MAX_ADMIT=0`, 4 × 192 steps, ± random routing (ported #29539) | No slot is read mid-copy or after eviction | 0 non-finite logits; E0 against poison off |
| T06 | Mailbox protocol harness | Fake-GPU thread, H ∈ {1, 2, 8, 64} helpers on 2 cores, 10^6 requests, random m, 6 delay points, TSan | Either passes, or exposes the stale-payload window (Summary item 2) | 0 mismatches vs golden; 0 TSan reports; no hang in 60 s |
| T07 | Timeout aborts cleanly | gpt-oss-120b, `HELPERS=1`, `TIMEOUT_MS=1` | `GGML_ABORT` "…timed out" | Non-zero exit ≤ 5 s after the first decode step; no logits returned; NVML at baseline ≤ 10 s |
| T08 | Lifecycle | 20 configurations per process; two models in sequence; `llama-server -np 2` for 10 minutes | Helpers joined, memory freed, stock path for batched decode | Thread count back to baseline after each free; RSS growth < 1 %; no deadlock; outputs E2 vs single-request |
| T09 | Budget edges | ALLOC all-0; slots 1, k−1, k; mixed ALLOC; C = E (gpt-oss-20b); static without INIT; oversize C | Per §3.4 | E2 (E0 for C = E if the fusions match); invariants hold; C = E within 2 % of stock tok/s; oversize exits non-zero ≤ 60 s |
| T10 | System identity gate + invariants | Parser over every stats JSON and log of a run | mailbox = 1, mxk = 1, paced = 0, κ/policy/C as pre-registered; §3.1 invariants | All exact; any failure invalidates the configuration's numbers |
| T11 | E2 against the noise floor (V1) | 4k-token on-policy stream per model, full log-softmax dumps; floor pairs per §2.1; ours at 3 budgets + C = 0 | Ours within the floor | Mean and P99.9 KLD ≤ 3× floor; top-1 disagreement ≤ floor + 0.5 pp; \|ΔNLL\| ≤ max(1.5× floor, 0.1 %) |
| T12 | Free-running near-tie divergence (V2) | 32 prompts × 256 greedy tokens, ours and every entrant scored by the reference | Divergences only at near-ties | ≥ 90 % of first divergences at a margin < P5 margin; divergence rate ≤ the floor pair's |
| T13 | Patch-off identity | Patched build with EC unset vs unpatched mainline, same commit; greedy `llama-server` + 6 interleaved speed pairs | The patch is invisible when off | Outputs E0; tok/s ratio 95 % CI within [0.99, 1.01] |
| T14 | A/B decision: batched miss job | ec-bench `W;A;B;B;A;…`, 6 (→ 12) pairs; primaries and guards per §4.1 | Lower f; +several % at the tight budget | §4.4 rule; E0 dumps A vs B |
| T15 | Equal VRAM per entrant (F1) | NVML sampler through a full request set, 3 launches per point | Each entrant fitted to the budget | Peak ≤ budget and ≥ budget − max(256 MiB, 2 %); breakdown recorded |
| T16 | Simulator ≡ C++ policy | New routing dump (`LLAMA_EC_ROUTE_DUMP`) from a Tier 1 run; replay in `ecsim` with the same C, κ and half-life | Same victims, same publication delay | Per-step, per-layer hits, misses and admits identical |
| T17 | Competitor reproduction (F11) | FreeToken offload gpt-oss-120b; pipeshard Qwen3 Q4_0 at 8 and 16 GB; community-cache gain | Per the reproduction targets table | Within the pre-registered tolerance → "reproduced/replicated"; else "runs, not reproduced" with the log |
| T18 | Scoring code | Synthetic samples; SoL inputs; equal-VRAM arithmetic; status schema | ΣN/ΣT used; head bytes included; the correct bound class | Exact equality on the synthetic cases; bootstrap coverage 93–97 % |
| T19 | Token and length identity (F3/F4) | All engines × all race prompts | Same IDs in, N tokens out | 100 % hash matches; 100 % of requests have exactly N tokens |
| T20 | Harness A/A (F13) | Mainline through the race driver vs server `timings`; ours teacher-forced vs served | The driver adds no bias | \|Δ\| ≤ 1 % driver vs server; teacher-forced vs served reported |

---

## 9. Gaps in the existing coverage

1. **No BF16 anywhere.** The test-backend-ops EC cases are Q8_0, Q4_K, Q6_K and MXFP4 only, and every A10 run used mmvq
   types. The CUDA float MUL_MAT_ID paths are unpatched for −1. There are no fused-variant cases and no all −1 case.
2. **Op tests have never run on sm_120, and never under compute-sanitizer.** Job 011 ran on an A10 (sm_86) with an
   unrecorded toolkit version.
3. **Exactness evidence is thin:**
   - dumps hold the top-5 only, so KLD has never been computed;
   - job 009 validated **split-graph** mode, not mailbox;
   - mailbox fidelity comes only from the performance runs' dumps against `ec-bench --ncmoe` inside the patched build;
   - no noise floor was measured on the same text except all-GPU vs llama.cpp;
   - no comparison with an **unpatched** mainline exists.
4. **No test of the mailbox protocol in isolation.** No TSan, no injected preemption. The stale-payload window is
   untested.
5. **No determinism or poison test.** `LLAMA_EC_CHECK` was used only in job 012.
6. **The C++ policy has not been checked step-by-step against `mosl/ecsim.py`.** The agreement is only in aggregate, to the
   third decimal, and the simulator is the basis for re-deriving κ.
7. **The baseline is `ec-bench --ncmoe` in the patched build,** not stock `llama-server`/`llama-bench`. Whether its CPU
   override receives the CPU_REPACK buffer type as stock `--n-cpu-moe` does is unverified (T13/T20 settle it).
8. **No repeats in phase 4.5 (jobs 025, 038, 039), so no CIs.** The llama.cpp runs shift by up to 8 % run to run.
   Job 018 is the only A/A data, and it shows a first-configuration penalty.
9. **`ec-bench` is not a race harness:**
   - it times teacher-forced decode only;
   - its generation mode is untimed and samples (top-k 40, temperature 0.8);
   - it sets `n_ubatch = n_prefill` and `n_ctx = prefill + decode + 16`;
   - it has no NVML sampling;
   - its "first step" is included without being labelled.
10. **Robustness paths untested:** timeout, helper shutdown under load, reload, `llama-server` concurrency, OOM, long
    contexts beyond 640 prefill tokens, and the silent fallbacks (merged `gate_up`, static without INIT, bad ALLOC/INIT
    files).
11. **Job infrastructure:**
    - `setup.sh` rebuilds from scratch on every patch change;
    - it assumes `sudo nvidia-smi -lgc` works and does not record whether it did;
    - it pins 2145525a, while the race base should be one frozen commit shared with mainline.
12. **The analysis has no regression tests** for the SoL head-bytes issue, harmonic-mean summarization, CI code, equal-VRAM
    mapping or bound class per entrant.
13. **No race-harness tests of any kind:** VRAM fit, threads, tokens, output length, status blocks.
14. **The `repack.cpp` negative-id change is dead code** in the evaluated configurations (host experts are never
    repacked). Either test it or drop it before upstreaming.

---

## 10. Tooling additions the tests require

Each is small and switch-guarded:

1. `ec-bench --dump-logprobs` (fp16 full log-softmax), and `--route-dump` (or `LLAMA_EC_ROUTE_DUMP`) writing the per-step
   `routing_host`.
2. `ec-bench --gen-timed`: greedy free-running, fixed N, EOS ignored, timed per token. Alternatively, confirm that
   `llama-server -ot …=CUDA_Host --load-mode none` places the experts in pinned memory, so ours can be served like
   mainline.
3. `ec-bench --ctx` and `--ubatch` separate from `--n-prefill`, for chunked prefill.
4. `LLAMA_EC_POISON`, `LLAMA_EC_ASSERT`, and the `EC_TEST_DELAY` hooks with the protocol header extracted for T06.
5. Stats JSON additions:
   - a per-request histogram of (m, service µs);
   - `post_us` (policy latency);
   - the value of every optimization switch;
   - header defaults printed next to the effective values.
6. A race driver (Python): token-ID prompts, streaming client timestamps, pynvml sampler, `/proc` thread audit, bandwidth
   probes, and records written as JSON with a schema check.
7. `jobs/ec/setup.sh` changes:
   - ccache, and no `rm -rf` of the build directory;
   - `-DCMAKE_CUDA_ARCHITECTURES=120` with a pinned toolkit, recording `nvcc --version`;
   - record the clock policy instead of assuming it;
   - `@5090` job tags.

---

## Appendix A: checks run for this note (CPU only; files in `S:tstrat/`)

1. **Tiny-model bit-exactness.** `S:llama.cpp/build-cpu/bin/llama-ec-bench` on `S:tiny/qwen3moe-4L32E-q8_0.gguf` and
   `S:tiny/gptoss-4L32E-q8_0.gguf`:
   - `--host-experts -t 2 --n-prefill 32 --n-decode 128`, random-token corpora (4 sequences);
   - 7 configurations: `slots=0`; lru 1; dfa 3 (κ 1); lru 4; dfa 16; slots 32 (= E); static 8.

   Results:
   - all 7 top-5 dumps are byte-identical per model (sha 295b4d50… Qwen, 5945033c… gpt-oss);
   - every `check:` line reports 0 inconsistent ids;
   - `static` without INIT reports "0.000 served on the GPU";
   - 26 s wall time for both models.
2. **AVX-512 MXFP4 helper kernel vs ggml `vec_dot` (MXFP4 × Q8_0).** `S:tstrat/mxk_test.cpp`: 64,000 rows, n ∈ {64, 96,
   2880, 4096, 5760}, weight scales 1e-3–50, inputs ×1 and ×100. Worst |Δ| / Σ|w·x| = 1.97e-8, and 0 non-finite values.
3. **Stats invariants** (§3.1) over all 100 `*_mb_*.json` files in `G:results/02*`–`03*`: 0 violations.
4. **Job 018 A/A repeats** (paired sequences, r2/r1):
   - `pin_mxk`, the **first configuration in each process**: 1.022 (Qwen3 Q4_K_M), 1.010 (gpt-oss-120b), 1.012
     (gpt-oss-20b);
   - `pin_nomxk` and `nopin_mxk`: 0.995–1.001;
   - per-sequence log-ratio SD: 0.07–2.2 % for the mailbox configurations;
   - split-graph mode: 0.894–1.026, SD up to 12.6 %.
5. **Code reading of the CUDA dispatch** in `S:llama.cpp`:
   - `ggml_cuda_mul_mat_id` (`ggml-cuda.cu` ~1906–1948);
   - `mul_mat_f` with ids (`mmf.cu` 64–130, `mmf.cuh` 94–160);
   - `mmvf.cu` 20–46 (no guard for id −1);
   - the synchronous fallback's `GGML_ASSERT(ids_to_sorted_host.size() == ne_get_rows)`;
   - `test_mul_mat_id::max_nmse_err` = 5e-4 (2e-2 for native-FP4 activations on Blackwell).

   Whether the fused `mmvf` path fires for BF16 with ids on sm_120 was not verified by running. T01 decides it.
