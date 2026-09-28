# Our MoE offload system: baseline, where its decode time goes, and what to expect on Blackwell

As of 28 Sep 2026. Scope: the mailbox expert cache (`runtime/llama.cpp-expert-cache.patch`) and its A10 case study,
profiled from the committed result files; no GPU was used and nothing in either repository was changed.

**Path conventions.** `R:` = `/home/claude/moe-speed-of-light` (branch `main`). `G:` = `/home/claude/gpu-branch`
(branch `gpu`; `G:results/<job>/`). `S:` = this session's scratchpad
(`/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/`). The scratch files hold
the derived numbers: `profile_runs.py` → `runs_flat.json`, `configs.json`; `decompose.py` → `decomp.json`;
`hitrates.py` → `hitrates.json`; `project.py` → `projections.json`. They are not in either repository and the scratchpad
is session-scoped. Copy them if the numbers need to be regenerated. About 8 min of CPU in total.

**Instances.** The A10 runs span six VMs (Lambda, Xeon Platinum 8358, 30 vCPU). I number them from the build and
download events in each job's `stdout.log`: I1 = jobs 001–007, I2 = 008–009, I3 = 011–012, I4 = 014–021 (the phase-3
instance), I5 = 025–026, I6 = 038–039b.

---

## 1. Overview

- **What was measured.** On I6 (jobs 038 = arm D, 039a/b = arm S) the mailbox cache runs at:
  - gpt-oss-20b: 84.2–116.4 tok/s;
  - Qwen3-30B-A3B Q4_K_M: 93.3–116.4 tok/s;
  - Qwen3-30B-A3B Q8_0: 65.5–88.2 tok/s;
  - gpt-oss-120b: 62.6–71.1 tok/s.

  That is 1.31–1.66× llama.cpp `--n-cpu-moe` at equal expert VRAM (R:paper/table_a10.tex, R:prereg/a10_windows_038/scored_{D,S}.md).
  On a second instance (I5, job 025, arm D) the same configurations give **1.37–1.60×**. The cache itself agreed across
  instances within 0 to −1.8 %. llama.cpp moved by −7.8 % to +1.6 %, and the 1.66× top of the paper's range (Q8_0, 25 %,
  038) comes from a llama.cpp run that is 7.8 % slower than the same configuration on I5 (§3.4).
- **The paper's speed-of-light (SoL) is too optimistic, so the fractions of it are understated.**
  - `R:scripts/sol_windows.py` / `sol_a10.py` split the all-GPU step into dense and expert work by bytes, but leave out
    the LM head (`dense_bytes` from `R:data/gguf_bytes.json` excludes `head_bytes`: 615 MB for gpt-oss, 255/331 MB for
    Qwen3 Q4_K_M/Q8_0). That puts GPU expert work at 65 % of the all-GPU step.
  - Nsight kernel times put it at **45 %** (gpt-oss-20b) and **42 %** (Qwen3 Q4_K_M) (G:results/007_counters/nsys_*_gpu_trace.csv.gz).
  - With the measured split the SoL drops from 181 to **156 tok/s** (gpt-oss-20b) and from 197 to **164 tok/s** (Q4_K_M).
    Our cache then reaches **54–76 %** of it (paper: 47–64 %) and llama.cpp 37–53 % (paper: 33–45 %).
- **Where the excess over the bound goes (22 configurations on I6).** Measured request counts and helper service
  times attribute the excess as follows:
  - **CPU miss work left exposed** beyond the GPU's hit work: 49 % of the excess on average (69 % at 12.5 %, 21 % at 50 %).
  - **A fixed non-compute overhead O of 26–38 µs per MoE layer per token** (31–38 µs at 12.5–25 %): 37 % on average (47 % at 50 %). That is
    0.8–0.9 ms/token on gpt-oss-20b, 1.3–1.8 ms on Qwen3 and 1.8–1.9 ms on gpt-oss-120b. O covers hand-off,
    cache-path kernels, host policy work and admission interference. It is paid on every cached layer, including
    layers with no miss.
  - **The helpers' per-request fixed cost**: 9 %. It is ≈50 µs per request on gpt-oss and 18–26 µs on Qwen3.
  - **GPU expert work**: −16 % to +32 %.

  In waterfall form the order is: overhead O (mean 37 %), the bound's CPU+GPU bandwidth balancing (27 %; not
  realisable at the current per-request costs), helper per-request cost (16 %), helper throughput (129 GB/s against a
  166 GB/s read, 13 %), and DFA's extra misses over MIN-bypass (7 %, only material at 12.5 %).
- **Helpers are near their microbenchmark.** Service time per request ≈ f + m·c_e.
  - gpt-oss: f ≈ 50 µs, c_e ≈ 103–107 µs per 13.25 MB expert (≈129 GB/s; the AVX-512 kernel alone does 132–135 GB/s at 28–30 threads, G:results/038_ec_ctrl_d@a10/cpubench.txt).
  - Qwen3 Q4_K_M: f ≈ 18–23 µs, c_e ≈ 22–23 µs; Q8_0: f ≈ 21–26 µs, c_e ≈ 37–42 µs.
- **The H17 step-time model cannot be carried to new hardware.**
  - `R:scripts/mb_model.py` is a per-model regression `T = T0 + a0·r + a1·μ + β·α`. Its a0 is negative (−38 to −103 µs),
    and β rests on three cache points per model.
  - I replaced it with a structural model built from the measured counters (§5.4). Back-tested on the same A10 data it
    reproduces our cache within 2.5 % median (max 5.9 %) and llama.cpp within 5.6 % (max 12.8 %). That is in-sample.
- **Projection to the race hardware** (RTX PRO 6000 SE: 1597 GB/s; RTX 5090: 1792 GB/s; PCIe 5.0 at 63 GB/s):
  - The physical SoL is 350–990 tok/s, 1.3–3.3× the projected all-GPU speed. At these GPU bandwidths per-kernel fixed
    costs dominate, so the physical bound is uninformative for 3–5B-active models. Use an implementation-relative bound
    calibrated on the race machine.
  - With A10-derived constants, our cache as-is is projected at **1.35–2.1×** a same-constants llama.cpp with a
    150 GB/s host, **1.2–1.66×** at 250 GB/s and **1.06–1.37×** at 400 GB/s (gpt-oss, Qwen3-30B).
  - On Qwen3.6-35B-A3B (40 layers, 256 small experts, no trace) it is **0.78–1.02×** under independent routing, or
    1.1–1.3× at 250 GB/s (0.95–1.6× across hosts) if its routing locality matches Qwen3-30B's.
  - Removing O is worth +9–45 %.
- **Memory fit.**
  - Every listed model fits one engine in 90 GB of host RAM (gpt-oss-120b: 61.1 GB of pinned experts).
  - The RTX PRO 6000 (96 GB) holds all of them entirely in VRAM, so offload must be imposed through an explicit expert
    budget.
  - The RTX 5090 genuinely needs offload for gpt-oss-120b and Qwen3 Q8_0.
- **Rebase risk is low today.** Both patches pass `git apply --check` against upstream master 4da63377 (27 Sep 2026,
  22 commits after the base 2145525a); only offsets were needed. I did not compile the result. The fragile parts are
  listed in §2.3.

---

## 2. The system

### 2.1 Mechanisms (llama.cpp 2145525a + `R:runtime/llama.cpp-expert-cache.patch`, sha256 465361ad…)

1. **Placement.** `llama-ec-bench --host-experts` overrides `\.ffn_(up|down|gate|gate_up)_exps` to the CUDA pinned-host
   buffer type and loads with `LLAMA_LOAD_MODE_NONE` (no mmap). Every routed expert of every layer then lives in pinned
   host memory, and dense weights live on the GPU.
   - `llama_expert_cache` allocates C GPU "slot" copies per MoE layer (weights and per-expert biases, same strides).
   - `LLAMA_EC_ALLOC` can set C per layer: C = 0 is a CPU-only layer; C = E is a whole layer resident on the GPU with
     expert e in slot e, a stock-equivalent `gpu_only` graph and no maps.
2. **Routing maps.** Host arrays `slot_of[E]` and `cpu_of[E]` per layer are uploaded each decode step as graph input
   `ec_maps` (I32 [E_max, 2L]). In the graph, `get_rows(map, selected_experts)` yields `gid` (slot or −1) and `cid`
   (expert or −1). The selection is copied into a `routing` tensor that the policy reads after the step.
3. **Decode graph (n_tokens = 1 only; prefill and batched decode take the stock path).**
   - *Split-graph mode* (phase 2): the GPU does `mul_mat_id` over the slots with `gid`; the CPU does `mul_mat_id` over the
     host tensors with `cid`; the results are added. Negative ids produce zero rows through the patched CPU
     `mul_mat_id`/`add_id`/repack and CUDA `mmvq`/`add_id`. `GGML_TENSOR_FLAG_SPLIT_BEFORE` plus the scheduler overlap
     let the CPU split run beside the GPU split.
   - *Mailbox mode* (the evaluated system), per cached layer:
     - `ec_req` custom op on the GPU. If any `cid ≥ 0`, one block writes x (n_embd floats), the ids and the layer index
       to a pinned mailbox, then `__threadfence_system`, then publishes a sequence number.
     - The GPU runs the hit experts over the slots.
     - `ec_wait` custom op. Four blocks, each with one thread polling RESP over PCIe, with a timeout of
       `LLAMA_EC_TIMEOUT_MS × 2e6` cycles. The wait then merges the helper rows (read with `__ldcv` from pinned memory)
       with the GPU rows.
     - A layer with no CPU expert still launches both ops, which then do nothing and a plain copy respectively.
     - A C = 0 layer is `ec_req` + `ec_wait` only.
4. **Helpers.** `LLAMA_EC_HELPERS` threads spin on the mailbox during decode and park otherwise.
   - Each request runs four phases, each cut into chunks: quantize the input to the vec_dot type; gate and up rows +
     bias + activation (SwiGLU, or gpt-oss SwiGLU-OAI with α = 1.702, limit 7, hard-coded); quantize the activations;
     down rows + bias written straight into the mailbox rows.
   - Chunks are claimed through atomic counters tagged with the request sequence number, so a descheduled helper delays
     only its own chunk (job 018's fix).
   - MXFP4 experts use an AVX-512 VNNI kernel (`src/llama-ec-mx.h`) on raw GGUF blocks.
   - The last chunk publishes RESP and adds the service time to `mb_busy_us[m]`.
5. **Policy** (host, `post_compute`, after `ggml_backend_sched_synchronize` of every decode step, on the critical path).
   - Reads `routing` (a synchronous D2H copy) and updates per-expert decayed scores (half-life), LRU stamps and counts.
   - For each miss: take a free slot, or evict the lowest decayed score (DFA) or the oldest stamp (LRU) among residents
     not selected this step and not loading. DFA admits only if `score(e) > score(victim) + κ`.
   - `LLAMA_EC_MAX_ADMIT` caps admissions per step, with a rotating first layer. STATIC never admits.
6. **Admission copies.** Issued on a second CUDA stream.
   - The default is a zero-copy kernel (16 blocks reading pinned memory). Each call does a `cudaHostGetDevicePointer`
     plus one launch per expert tensor: 3 per expert on Qwen3, 6 on gpt-oss with biases.
   - Batched mode (the default) records an event; `pre_compute` waits on it and publishes the slots **two steps later**.
   - Paced mode is a copier thread with one piece in flight.
7. **Instrumentation.** `LLAMA_EC_STATS` writes JSON at context free: hits, misses, admits, per-layer `miss_hist`,
   per-expert counts, and `mb_busy_us` / `mb_count` indexed by the number of helper-computed experts m. `LLAMA_EC_CHECK`
   verifies the ids the graph used against the host maps.
8. **Tools.** `llama-ec-bench` is a teacher-forced decode benchmark with per-step timing, next-token NLL, top-5 logit
   dumps and a sampling mode. `llama-ec-cpubench` measures read and row-dot throughput per thread count.
9. **`R:runtime/ggml-sched-overlap.patch`** (151 lines) is exactly the scheduler part of the big patch; only comments
   differ (verified by diff).
   - It adds `ggml_backend_sched_set_overlap_cpu(sched, bool)` (off by default) and `GGML_TENSOR_FLAG_SPLIT_BEFORE = 32`.
   - When a GPU split is followed by a CPU split with no input produced by it (view ops skipped), the CPU inputs are
     copied first, the GPU split is enqueued, and the CPU split runs concurrently.
   - It only matters for split-graph mode, not the mailbox.

### 2.2 Knobs

| knob | default in code | value in the evaluated system (jobs 019–039) | notes / measured effect |
|---|---|---|---|
| `LLAMA_EC_SLOTS` | unset = off | C = E·q/8 (1 plus `ALLOC` for static layouts) | enables the cache |
| `LLAMA_EC_ALLOC` | none | per-layer 0/E files for "helpers, static layers" | `<layer> <slots>` lines |
| `LLAMA_EC_POLICY` | `dfa` | `dfa` | static / lru / dfa |
| `LLAMA_EC_HALF_LIFE` | 16 | 16 | 8 gave < 1 pt more hits at 1.5× the copies (017, protocol log) |
| `LLAMA_EC_KAPPA` | **0** | **1** (gpt-oss), **2** (Qwen3) | κ gave +1.7–16.6 % over κ = 0 in split mode (015: 120b C32 47.7 → 55.6) |
| `LLAMA_EC_INIT` | none | unused | initial set (text file or stats JSON) |
| `LLAMA_EC_OVERLAP` | 1 | 1 (moot in mailbox mode) | split mode: +7–13 % (015, overlap vs `overlap=0`) |
| `LLAMA_EC_MAILBOX` | 0 | **1** | +14–23 % over split-graph at 25 % (016) |
| `LLAMA_EC_HELPERS` | hardware threads − 2 | 28 (30 vCPU) | 28 vs 14: +4–8 % (016) |
| `LLAMA_EC_PIN` | 0 | 0 | pinning: −1.5 to −0.2 % (018) |
| `LLAMA_EC_MXK` | 1 (**undocumented** in the header) | 1 | AVX-512 MXFP4 kernel: +0.1 % (120b) to +1.4 % (20b) (018). The same pair moved Q4_K_M, which never uses the kernel, by −1.8 %: noise level. |
| `LLAMA_EC_TIMEOUT_MS` | 2000 | 2000 | converted at 2 GHz, so shorter on faster-clocked GPUs |
| `LLAMA_EC_ZC` / `_ZC_BLOCKS` | 1 / 16 | 1 / 16 | zero-copy vs copy engine: +17 % (20b) and +24 % (Q4_K_M) at 25 % (014); 8 or 32 blocks: no clear effect (017) |
| `LLAMA_EC_PACED` | **0** (the header comment says 1) | 0 | paced 54.7 vs batched 58.6 tok/s (012) |
| `LLAMA_EC_CHUNK`, `_MAX_PENDING` | 0, 0 | 0, 0 | paced-mode only |
| `LLAMA_EC_MAX_ADMIT` | 0 = unlimited | 0 | 2 or 6 per step: Q4_K_M 67.5 / 79.3 vs 98.6 tok/s (017): throttling hurts |
| `LLAMA_EC_CHECK` | 0 | 0 (1 in 012) | id consistency check |
| `LLAMA_EC_STATS` | none | set | counters used throughout this note |
| ec-bench `tdec` | = `-t` | 1 (decode graph threads); prefill `-t 30` | – |
| ec-bench CLI | `-m --corpus -ngl(999) --ncmoe --host-experts --no-mmap --seqs/--n-seq/--seq0 --n-prefill(64) --n-decode(128) -t --dump --ec "k=v:…;…" --gen-out --gen-temp(0.8) --gen-seed(42)` | `--n-prefill 640 --n-decode 192 -t 30` (phase 4.5) | `--ec` sets any `LLAMA_EC_*` per configuration on one loaded model |
| job env | – | `GGML_NO_BACKTRACE=1`, clocks locked (`nvidia-smi -lgc/-lmc` max) | G:jobs/ec/setup.sh |

### 2.3 Files touched and rebase sensitivity (+ lines / − lines)

| file | +/− | what | sensitivity |
|---|---|---|---|
| `src/llama-graph.cpp` | 213/0 | wraps the stock expert block of `build_moe_ffn` in `if (experts == nullptr) {…}`; adds `build_moe_ffn_ec` and the maps input | **High.** `build_moe_ffn` changes often (fused ops, `*_exps_s`, merged `gate_up`, `weight_before_ffn`). The path re-implements the FFN with hard-coded activations, depends on `hparams.swiglu_clamp_exp` and `LLM_FFN_*`, and silently falls back to stock if anything differs. The mailbox needs **separate** gate/up tensors: a GGUF with merged `ffn_gate_up_exps` silently loses mailbox mode. |
| `ggml/src/ggml-backend.cpp` | 69/2 | `compute_splits` refactored into `copy_inputs`/`compute` lambdas; overlap path; split flag | **High.** Scheduler internals (split inputs, events, pipeline copies). Assumes the CPU backend is the last backend. |
| `ggml/src/ggml-cuda/mmvq.cu` | 9/0 | early return with a zero row for negative ids inside `mul_mat_vec_q` | **High.** Depends on kernel internals (`channel_x`, `rows_per_cuda_block`). Only the `ncols_dst == 1 && ids` mmvq path is patched; any other `MUL_MAT_ID` kernel chosen for single-token decode on sm_120 (for example an FP4-native path) would index out of bounds on −1. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 49/0 | `GGML_OP_CUSTOM` dispatch and `supports_op`; zero-copy copy kernel; proc-address exports | Medium. CUDA-graph capture and compatibility of the spinning `ec_wait` kernel (graphs were in use: "CUDA Graph … reused" in G:results/020_ec_eval_c/stderr_runs.txt). |
| `ggml/src/ggml-cuda/{ec.cu,ec.cuh}`, `ggml/include/ggml-ec.h` | 208/0 | REQUEST/WAIT kernels, mailbox layout (x at +4096, K ≤ 64) | Low (new files). Relies on the `ggml_custom_4d` API and `ggml_custom_op_params`. |
| `ggml/src/ggml-cpu/{ggml-cpu.c,ops.cpp,repack.cpp}`, `ggml-cuda/add-id.cu` | 27/3 | negative-id handling | Medium. Split mode only. Other CPU extra buffer types (AMX, KleidiAI) are **not** patched. |
| `ggml/include/ggml.h`, `ggml-backend.h`, `ggml/src/ggml.c` | 11/0 | `GGML_TENSOR_FLAG_SPLIT_BEFORE = 32`, overlap API, `ggml_ec_op_stub` | Low–medium (a new upstream flag would collide). |
| `src/llama-context.{cpp,h}` | 42/0 | cache created in `sched_reserve`; `pre_compute`/`post_compute` hooks in `process_ubatch` (plus an extra `sched_synchronize`); `graph_params` gains `/*.ec =*/` | Medium. Positional aggregate initialisation of `llm_graph_params`. |
| `src/llama-expert-cache.{cpp,h}`, `src/llama-ec-mx.h` | 1346/0 | cache, policy, helpers, AVX-512 kernel | Medium. Layer tensor names; the `vec_dot`/`from_float` traits; MXFP4 (17 B/32) and Q8_0 (34 B) block layouts hard-coded; pinned-host detection by `strstr(buft_name, "Host")`; experts must not be repacked. |
| `src/llama-graph.h` | 35/0 | input class, params field | Medium. |
| `tests/test-backend-ops.cpp` | 53/0 | `MUL_MAT_ID_EC` / `ADD_ID_EC` cases | Low (18/18 passed on CUDA vs CPU in 011). |
| `tools/ec-bench/*`, CMake | 541/0 | benchmark tools | Low (uses `llama_model_params.load_mode`). |

**Checked.** Both patches apply to upstream master 4da63377 with `git apply --check` (hunk offsets of 1–10 lines).
The upstream changes to the touched files in those 22 commits (FWHT hint in `ggml-cuda.cu`, a
`llama_get_causal_attn` accessor, a reranker condition in `llama-graph.cpp`) do not interact with the patch. I did not
build it.

**Blackwell-specific.**
- The build must target sm_120 (`setup.sh` derives it from `nvidia-smi compute_cap`), which needs CUDA ≥ 12.8.
- The helper and AVX-512 paths assume x86 with AVX-512 VNNI (Zen 4/5 and Ice Lake+ have it); otherwise they fall back
  to ggml's `vec_dot`.
- Helper pinning assumes helper i runs on vCPU i+1. On SMT hosts that pairs helpers on shared cores.

---

## 3. Data profile

### 3.1 Inventory

| dataset | files | grain (one row per) | configurations | reps | status |
|---|---|---|---|---|---|
| G:results/003_sweep | `sweep.jsonl` (34 rows) | llama-bench configuration | 4 GGUFs × n_cpu_moe | – | **failed**: every row rc = 127 (`llama-bench` missing) |
| G:results/004_counters | 22 files, all 0 bytes except `ncu_summary.json` (32 B) | – | – | – | **failed** (no Nsight on the image) |
| G:results/006_sweep | `sweep.jsonl` (34 rows, rc = 0), `threads.jsonl`, `gpu_monitor.csv`, `order.txt` | llama-bench configuration (`-p 0 -n 128 -r 5`, repack on, 30 threads) | 31 configurations (20b n = 0–24 + CPU; Q4_K_M n = 0–48 + CPU; Q8_0 n = 18–48; 120b n = 25–36) + 3 drift repeats | 5 per configuration (`samples_ts`); CV median 1.0 %, max 2.9 % | ok, I1. Scored in R:prereg/a10/scored.md (H0 pass, H1 fail with +12–15 % ncu over-read, H2 fail at 27.4 % median APE, H3 pass R² 0.9965–0.9987, drift ≤ 0.9 %) |
| G:results/007_counters | ncu CSVs (4 configurations × n = 4/8), nsys GPU traces (6), `ncu_summary.json` | kernel launch (ncu, nsys) | 20b n = 0/12/24, Q4_K_M n = 0/24/48 | 1 | ok, I1. Per token: 20b n = 0 reads 3.034 GB, 652 kernels; Q4_K_M 2.178 GB, 1110 kernels |
| G:results/008–009 | build, op tests, `correctness.json`, logit dumps | – | 20b, Q4_K_M | 1 | I2. H6 failed narrowly (top-1 97.4/97.9 %) |
| G:results/010, 013 | `DONE` only | – | – | – | **aborted** (instance lost), no data |
| G:results/011–012 | op tests (18/18 EC, 965/965 regular), 012 `ec.jsonl` + `s_*.json` + static runs | sequence × configuration (4 seqs × 128 steps) | 20b at 25 % (diagnosis) | 1 | I3. The three 012 nsys traces are **0 bytes**. |
| G:results/014–018 | `*_ec.jsonl`, `*_static_n*.jsonl`, stats JSON, `cpubench_*.txt` (018) | sequence × configuration | knob and design sweeps: 136/232/200/112/108 rows | 1, except 018: 2 interleaved (r1/r2) | I4, exploratory |
| G:results/019_ec_calib | 77 files, 432 rows | sequence × configuration (8 profile seqs × 128 steps) | 4 models × {mailbox, helpers static, split, llama static} × budgets + all-GPU | 1 | I4. Fit of H17 (R:prereg/a10_mb/predictions.{json,md}) |
| G:results/020_ec_eval_a/b | 336 + 240 rows, dumps | sequence × configuration (12 test seqs × 192 steps, **phase-3 windows**) | 4 models × 4 systems × budgets | 1 | I4. Scored in R:prereg/a10_mb/scored.{md,json} (H11 hold, H12/H13 fail, H14 hold) |
| G:results/020_ec_eval_c | 192 rows + `*_gen.jsonl` | same, on sampled own text (phase-3 windows) | 4 models × {mailbox, llama} × 2 budgets | 1 | I4. rc = 1 comes only from the summary script (KeyError `nll_sum`); the data is complete. R:prereg/a10_mb/own_text.json |
| G:results/021_llama_nommap | 132 rows | sequence × configuration | llama `--no-mmap`, 11 configurations | 1 | I4. −13.4 % to +1.7 % vs mmap (protocol) |
| G:results/025_ec_windows_d@a10 | 89 files, 420 rows, stats, dumps | sequence × configuration (12 test seqs, whole prompt prefilled, ≤ 192 response tokens: 2226/2246 steps) | 4 models × {mailbox, helpers static, llama static} × budgets + all-GPU; arm D | 1 | I5. R:prereg/a10_windows/scored_D.{md,json}, sol_D.json. No host bandwidth benchmark on this instance. |
| G:results/026_ec_windows_s@a10 | 35 files, all `*_ec/static.jsonl` empty | – | – | – | **void** (S corpora did not exist yet) |
| G:results/038_ec_ctrl_d@a10 | 102 files, 516 rows, `cpubench.txt` | as 025, plus a phase-3-window control at 25 % | as 025 + control | 1 | I6. R:prereg/a10_windows_038/scored_D.*, sol_D.json |
| G:results/039a/039b | 76 + 20 files, 348 + 72 rows | as 025 on the S arm (2304 steps) | 20b/Qwen3 (039a), 120b (039b) | 1 | I6. scored_S.*, sol_S.json |
| R:prereg/a10_ec | `predictions.{json,md}` | predicted configuration | phase-2 split-graph cache | – | **never scored** (010 and 013 aborted, then the design changed) |
| R:paper/table_a10.tex, `numbers4.tex` | 11 table rows; macros `caseMin` = 1.31 … `hSeventeenMed` = 5.0 | model × budget | – | – | matches 038/039 (spot-checked: 20b 12.5 % 58.9 / 86.2, ×1.46 D, ×1.31 S, 48 % / 33 % of SoL) |

**Test design of the phase-4.5 rows (025, 038, 039).**
- Test sequences 7, 10, 12, 15, 16, 18, 19, 20, 25, 26, 27 plus 31 (gpt-oss) or 32 (Qwen3), in that order, with cache state carried over.
- Budgets 12.5 / 25 / 50 % (gpt-oss-120b: 12.5 / 25 %), paired with llama.cpp `--n-cpu-moe n = L − L·q/8`.
- 2226 steps (D) or 2304 (S) per configuration; two D sequences are shorter than 192 steps (114 and 134).
- Metric fields per sequence: `decode_ms`, `tok_s`, `nll_sum`/`nll_n`, and `step_ms` (192 values). Stats JSON per cache or helper configuration.

### 3.2 Configurations covered (model × quant × budget × arm × instance, the phase-4.5 core)

| model / GGUF | budgets | arms × instances | systems | all-GPU reference |
|---|---|---|---|---|
| gpt-oss-20b MXFP4 (L 24, E 32, k 4) | 12.5, 25, 50 % | D × {I5, I6}, S × I6 | mailbox, helpers static, llama static | yes (135.8–138.1 tok/s) |
| Qwen3-30B-A3B-Instruct-2507 Q4_K_M (L 48, E 128, k 8) | 12.5, 25, 50 % | same | same | yes (143.8–146.7) |
| Qwen3-30B-A3B-Instruct-2507 Q8_0 | 12.5, 25, 50 % | same | same | **no** (32.5 GB > 24 GB) |
| gpt-oss-120b MXFP4 (L 36, E 128, k 4) | 12.5, 25 % | same | same | **no** |

The split-graph cache is not in phase 4.5; it is in 019/020 only. Q8_0 and gpt-oss-120b have no measured all-GPU step
anywhere, so every bound for them depends on an extrapolated T0 (§4.1).

### 3.3 Metrics profile (I6, jobs 038/039, 22 configurations per system, 4 for all-GPU; S:configs.json)

| system | per-sequence tok/s CV (median, max) | first step / median step | steps > 3× median, per 1000 | share of time in the first 5 steps of a sequence (uniform = 0.026) |
|---|---|---|---|---|
| mailbox cache | 0.061, 0.136 | 1.77 | 1.7 (max 3.6) | 0.042 |
| helpers static layers | 0.003, 0.057 | 1.30 | 0.0 (max 0.9) | 0.029 |
| llama.cpp static | 0.022, 0.109 | 1.23 | 0.7 (max 2.6) | 0.030 |
| all-GPU | 0.004, 0.006 | 1.46 | 0.2 | 0.032 |

- The cache's sequence-to-sequence spread is real routing-locality variation: its hit rate varies by sequence, while
  the helpers' static layouts (no cache policy) have a per-sequence CV of 0.3 %.
- The first-step penalty after each prefill (new prompt, stale cache) costs about 1.5 % of the total. It is included in tok/s.

**Counters (stats JSON, I6).** Per token at 25 %:

| model | hit rate | helper requests (hand-offs) | CPU experts | admissions |
|---|---|---|---|---|
| 20b | 0.676 | 17.0 | 31.1 | 3.16 |
| Q4_K_M | 0.773 | 34.0 | 87 | 11.15 |
| Q8_0 | 0.773 | 34.0 | 87 | 11.15 |
| 120b | 0.780 | 19.5 | 31.7 | 9.85 |

Measured hit rates equal the phase-4.5 simulation (`DFA hit (sim)` column in scored_*.md) to the third decimal for
Qwen3 (0.604/0.773/0.908).

**Fidelity** (my recomputation from the logit dumps, `*_ec.bin.*` against `*_static_n*.bin`).
- Top-1 agreement of cache vs llama.cpp is 0.981–0.993 on 20b and Qwen3, and 0.983–0.988 on gpt-oss-120b arm S.
  All-GPU vs llama.cpp gives 0.979–0.991.
- gpt-oss-120b arm D: 0.890–0.897. That text is off-distribution for the model: NLL 5.75 nats on D against 0.58 on S.
- Mean NLL differs from llama.cpp by −0.99 % to +1.06 % (Q4_K_M S 50 %: +1.06 %; all others within ±1 %).

### 3.4 Variation

**Instance to instance, same configuration and text** (arm D, 038 on I6 vs 025 on I5, (I6 − I5)/I5; S:configs.json):

| system | median | range | worst configurations |
|---|---|---|---|
| mailbox cache | −1.0 % | −1.8 % … 0.0 % | – |
| helpers static layers | −0.7 % | −2.9 % … +0.3 % | – |
| all-GPU | −1.7 % | −1.8 % … −1.6 % | – |
| llama.cpp static | −2.3 % | **−7.8 % … +1.6 %** | Q8_0 25 % −7.8 %, 20b 12.5 % −7.7 %, 20b 25 % −5.8 % |

The resulting speed-ups (I5 / I6):

| model | 12.5 % | 25 % | 50 % |
|---|---|---|---|
| 20b | 1.37 / 1.46 | 1.45 / 1.52 | 1.44 / 1.47 |
| Q4_K_M | 1.42 / 1.45 | 1.49 / 1.45 | 1.38 / 1.36 |
| Q8_0 | 1.44 / 1.47 | **1.53 / 1.66** | 1.44 / 1.44 |
| 120b | 1.56 / 1.56 | 1.60 / 1.60 | – |

**The llama.cpp anomalies are run-level, not text-level.**
- 038 `Qwen3-…-Q8_0_D_static_n36`: 5 of 12 sequences have median steps of 20.0–23.0 ms against 18.8 in the other
  seven. On 039a (same instance, S text) 4 of 12 sequences have medians of 20.0–22.1 ms, and a fifth falls below
  50 tok/s through outlier steps. On I5 no sequence's median is off; one sequence drops to 43.9 tok/s through outlier steps.
- 038 `gpt-oss-20b…_D_static_n21`: every sequence runs at a 16.46 ms median step against 15.44 on I5 and on 039a
  (same instance, S text): a uniform 6.6 % shift for that run.
- So "own text lowers the speed-up at 12.5 % on 20b (1.46 → 1.31)" is mostly the D-arm llama.cpp run on 038 being
  slow (58.9 tok/s against 63.8 on I5 and 64.1 on S).

**Phase-3 windows vs the other instance** (038 control vs 020 at 25 %; tok/s on I6 relative to I4):

| model | cache | llama.cpp |
|---|---|---|
| 20b | +2.9 % | +13.8 % |
| Q4_K_M | +2.7 % | +7.6 % |
| Q8_0 | −0.4 % | −0.5 % |
| 120b | −1.0 % | +8.7 % |

Host read bandwidth was 151.4 GB/s on I4 (018 cpubench, 30 threads) and 166.1 GB/s on I6 (038 cpubench).

**Run to run, same instance** (018, two interleaved repeats):
- Mailbox cache: −0.5 % to +2.2 % (9 pairs).
- Split-graph: −10.6 % (120b), −1.0 %, +2.6 %.
- llama-bench in 006: rep CV ≤ 2.9 %; drift repeats ≤ 0.9 %.
- The phase-4.5 jobs have **no repeats**.

**Arm S vs D on I6** (039 vs 038): cache −5.6 % to +1.4 % (median −1.3 %); llama.cpp −3.2 % to +8.8 % (median +1.9 %);
helpers static −5.9 % to +2.6 %; all-GPU −0.1 %.

---

## 4. Where the time goes

### 4.1 Models and constants

- **H17 step-time model** (`R:scripts/mb_model.py`, fitted on job 019, R:prereg/a10_mb/predictions.md):
  `T = T0 + a0·r + a1·μ + β·α` for the cache and helper layouts, where r = hand-offs, μ = CPU experts, α = admissions
  per step; llama.cpp is `T0s + s·n`.

  | model | T0 (ms) | a0 (µs) | a1 (µs) | β (µs) | stock s (µs/layer) |
  |---|---|---|---|---|---|
  | 20b | 6.77 | −77 | 122 | 369 | 530 |
  | Q4_K_M | 6.52 | −38 | 26 | 166 | 210 |
  | Q8_0 | 9.19 | −48 | 41 | 159 | 306 |
  | 120b | 9.74 | −103 | 130 | 247 | 544 |

  On I6 it predicts the cache within 2.2 % (D) and 2.7 % (S) median APE, and the helper and llama.cpp layouts within
  5–8 %. Those two are systematically under-predicted: 11/11 and 10–11/11 configurations (my breakdown of scored_{D,S}.json).
  a0 < 0 and β > the copy's PCIe time on Qwen3 (166 µs against 111 µs) show that the coefficients absorb each other.
  They are not physical costs.
- **My decomposition** (S:decompose.py, S:decomp.json). It uses measured counters of each run and a small set of constants:
  - **a, GPU time per expert execution.**
    - Stock-llama.cpp kernel time per token with all experts on the GPU minus with all on the CPU, divided by L·k
      (nsys, job 007): 20b (7219.8 − 3936.7)/96 = 34.2 µs; Q4_K_M (6615.7 − 3864.6)/384 = 7.16 µs.
    - Scaled to I6 by the all-GPU step ratio: 33.8 and 7.24 µs.
    - Q8_0 is set to 12.0 µs, from byte scaling at 418 GB/s; call it ±15 %. gpt-oss-120b uses 20b's value (same expert kernel).
  - **TD, dense GPU time**, = T0 − L·k·a with T0 the measured all-GPU step on the same instance and arm: 20b 7.36 ms,
    Q4_K_M 6.94 ms.
    - Q8_0: T0 = 9.92 ms, the phase-1 affine intercept 9.81 ms (R:prereg/a10/scored.md H4) × the I6/I1 ratio of Q4_K_M.
    - gpt-oss-120b: T0 = 10.5 ms, the midpoint of the I1-based (10.0) and I4-based (11.0, from 019's `stock_T0` 10.85)
      extrapolations, which disagree by 1 ms. Uncertainty ±0.5 ms.
  - **b = s/B_C** with B_C = 166.1 GB/s (038 cpubench, 30-thread read). **p = s/B_P** with B_P = 25.2 GB/s (001 `bw.txt`, H2D pinned).
  - **Helper service time per request** by number of experts m: `mb_busy_us[m] / mb_count[m]` from each run's stats,
    fitted as f + m·c_e.
  - **SoL**: `R:scripts/sol_a10.py:bound` with these a, b, p, TD and MIN-bypass misses M* from the trace pack of the
    same windows (as `sol_windows.py`). Pooled budget via `cachesim.simulate_global`.
- **Why the SoL differs from the paper's.** The paper sets η_g from `(dense_bytes + L·k·s)` with `dense_bytes`
  excluding the LM head. Its per-expert GPU cost is then 49.8 µs (20b) and 11.8 µs (Q4_K_M): 1.46× and 1.65× the
  Nsight value. The bound reaches its speed by moving expert work to the CPU, so it becomes 16–20 % faster.

  | model | paper SoL (tok/s) | head included, byte split | Nsight split (used here) |
  |---|---|---|---|
  | 20b | 181.0 | 161.6 | 156.4 |
  | Q4_K_M | 197.3 | 183.2 | 163.7 |

  On the A10 constants the SoL is **budget-independent** (156.4 tok/s at every budget for 20b). It is the CPU+GPU
  bandwidth-aggregation optimum: in the bound the CPU executes mc* = 1.19 (20b) or 2.40 (Qwen3) of the k experts per
  layer-step, **more than MIN-bypass's misses** at 25 % and 50 %. PCIe loads cover the 12.5 % case. The pooled-budget
  SoL equals the per-layer one on all configurations.

### 4.2 Components: ours vs the speed-of-light (ms per token; S:components.md)

Critical-path attribution per layer-step: time = max(GPU hits (k−m)·a, helper service f + m·c_e). GPU hit work is
always charged to "GPU experts"; any helper time beyond it is "CPU exposed", split proportionally into per-expert work
and the per-request fixed part. O = measured − [TD + Σ layer maxima] is everything else. The SoL executes its CPU work
fully hidden and has no O.

| model | arm | budget | measured | GPU dense (= SoL) | GPU experts, ours / SoL | CPU exposed, ours (per-expert + fixed) / SoL | helper busy (work) / SoL CPU work | O, ours / SoL | SoL |
|---|---|---|---|---|---|---|---|---|---|
| 20b | D | 12.5 % | 11.60 | 4.12 | 1.53 / 2.28 | 5.04 (4.23 + 0.81) / 0 | 6.37 / 2.28 | 0.92 / 0 | 6.40 |
| 20b | D | 25 % | 10.07 | 4.12 | 2.19 / 2.28 | 2.84 (2.34 + 0.51) / 0 | 4.09 / 2.28 | 0.92 / 0 | 6.40 |
| 20b | D | 50 % | 8.75 | 4.12 | 2.85 / 2.28 | 0.95 (0.73 + 0.22) / 0 | 1.68 / 2.28 | 0.82 / 0 | 6.40 |
| Q4_K_M | D | 12.5 % | 10.40 | 4.16 | 1.68 / 1.95 | 2.81 (2.39 + 0.42) / 0 | 4.17 / 1.95 | 1.75 / 0 | 6.11 |
| Q4_K_M | D | 25 % | 9.44 | 4.16 | 2.15 / 1.95 | 1.40 (1.17 + 0.23) / 0 | 2.63 / 1.95 | 1.73 / 0 | 6.11 |
| Q4_K_M | D | 50 % | 8.59 | 4.16 | 2.53 / 1.95 | 0.40 (0.32 + 0.08) / 0 | 1.21 / 1.95 | 1.50 / 0 | 6.11 |
| Q8_0 | D | 12.5 % | 14.40 | 5.31 | 2.78 / 3.30 | 4.58 (3.97 + 0.61) / 0 | 6.79 / 3.30 | 1.73 / 0 | 8.61 |
| Q8_0 | D | 25 % | 12.67 | 5.31 | 3.56 / 3.30 | 2.21 (1.91 + 0.30) / 0 | 4.16 / 3.30 | 1.58 / 0 | 8.61 |
| Q8_0 | D | 50 % | 11.37 | 5.31 | 4.18 / 3.30 | 0.62 (0.52 + 0.10) / 0 | 1.84 / 3.30 | 1.26 / 0 | 8.61 |
| 120b | D | 12.5 % | 15.69 | 5.63 | 3.03 / 3.42 | 5.24 (4.36 + 0.88) / 0 | 7.12 / 3.42 | 1.79 / 0 | 9.05 |
| 120b | D | 25 % | 14.06 | 5.63 | 3.80 / 3.42 | 2.80 (2.25 + 0.54) / 0 | 4.36 / 3.42 | 1.83 / 0 | 9.05 |

S-arm rows (S:components.md) differ from these by up to 0.9 ms (Q8_0 12.5 %: CPU exposed 5.47 vs 4.58 ms), mostly
in exposed CPU work, following the lower S hit rates. **The hand-off and admission split inside O is not identified.**
- Fitting the static-helper runs (no maps, no admissions) as `T − TD − (L−n)·k·a − n·busy_k = c + n·h` gives
  h = 20–33 µs per hand-off, but c = −0.12 to −0.75 ms. That is an unphysical intercept after extrapolating from
  n = 12–42 to 0, so c and h, or a, are biased.
- At its operating points, the static-helper layout pays only ≈0.1–0.2 ms of O per token (c + n·h). The cache pays
  0.8–1.9 ms.
- O per MoE layer is nearly constant: 31–38 µs for 20b and Qwen3 at 12.5–25 % on both arms (26–34 µs at 50 %);
  64–66 µs for 120b, or ≈50 µs with T0 = 11 ms.
- O per hand-off varies 39–120 µs.

So most of O is a per-cached-layer cost (every cached layer runs 2 `get_rows` on the maps, a routing copy, `ec_req`,
the slot-path kernels and `ec_wait`), plus per-step host work: the maps H2D, the routing D2H, the policy loop, and
admission launches with `cudaHostGetDevicePointer` per tensor. At 25 %, admission copies over PCIe take 13–37 % of the step
(120b: 9.85 × 13.25 MB per token at 25.2 GB/s = 5.2 ms per 14.1 ms step), which may explain 120b's larger O. No
profiler trace of the cache exists to confirm any of this: the 012 nsys files are empty.

### 4.3 Ranked gaps (waterfall from the bound to the measurement; S:waterfall.txt)

Each column adds one realism to the previous model. The order matters because the per-layer max is non-linear:
1. **balance + prefetch**: MIN-bypass misses executed per layer-step without CPU/GPU balancing or PCIe prefetch, minus the SoL;
2. **DFA extra misses**: DFA's measured misses (from the run's counters) instead of MIN-bypass's;
3. **helper rate**: helpers at c_e instead of read bandwidth;
4. **helper per-request**: the measured service time including f;
5. **O**: the measured time.

| model | arm | budget | measured (tok/s) | SoL (tok/s) | excess (ms) | balance + prefetch | DFA extra misses | helper rate | helper per-request | O | ours / llama as fraction of SoL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20b | D | 12.5 % | 86.2 | 156 | 5.20 | 1.24 (24 %) | 0.86 (17 %) | 1.03 (20 %) | 1.15 (22 %) | 0.92 (18 %) | 55 % / 38 % |
| 20b | D | 25 % | 99.3 | 156 | 3.68 | 0.88 (24 %) | 0.43 (12 %) | 0.59 (16 %) | 0.85 (23 %) | 0.92 (25 %) | 63 % / 42 % |
| 20b | D | 50 % | 114.3 | 156 | 2.35 | 0.88 (38 %) | 0.01 (1 %) | 0.18 (8 %) | 0.45 (19 %) | 0.82 (35 %) | 73 % / 50 % |
| 20b | S | 25 % | 99.5 | 156 | 3.65 | 0.89 (24 %) | 0.41 (11 %) | 0.65 (18 %) | 0.81 (22 %) | 0.89 (24 %) | 64 % / 45 % |
| Q4_K_M | D | 12.5 % | 96.2 | 164 | 4.29 | 0.71 (17 %) | 0.50 (12 %) | 0.69 (16 %) | 0.64 (15 %) | 1.75 (41 %) | 59 % / 41 % |
| Q4_K_M | D | 25 % | 105.9 | 164 | 3.33 | 0.70 (21 %) | 0.12 (4 %) | 0.40 (12 %) | 0.38 (11 %) | 1.73 (52 %) | 65 % / 45 % |
| Q4_K_M | D | 50 % | 116.4 | 164 | 2.48 | 0.78 (32 %) | −0.07 (−3 %) | 0.11 (4 %) | 0.16 (6 %) | 1.50 (60 %) | 71 % / 52 % |
| Q4_K_M | S | 25 % | 104.6 | 163 | 3.44 | 0.69 (20 %) | 0.12 (3 %) | 0.37 (11 %) | 0.54 (16 %) | 1.73 (50 %) | 64 % / 46 % |
| Q8_0 | D | 12.5 % | 69.4 | 116 | 5.79 | 1.21 (21 %) | 1.00 (17 %) | 0.88 (15 %) | 0.97 (17 %) | 1.73 (30 %) | 60 % / 41 % |
| Q8_0 | D | 25 % | 78.9 | 116 | 4.06 | 1.12 (28 %) | 0.30 (7 %) | 0.50 (12 %) | 0.55 (14 %) | 1.58 (39 %) | 68 % / 41 % |
| Q8_0 | D | 50 % | 87.9 | 116 | 2.76 | 1.23 (45 %) | −0.08 (−3 %) | 0.15 (5 %) | 0.21 (8 %) | 1.26 (46 %) | 76 % / 53 % |
| 120b | D | 12.5 % | 63.7 | 110 | 6.64 | 1.47 (22 %) | 0.83 (13 %) | 1.17 (18 %) | 1.37 (21 %) | 1.79 (27 %) | 58 % / 37 % |
| 120b | D | 25 % | 71.1 | 110 | 5.01 | 1.31 (26 %) | 0.27 (5 %) | 0.61 (12 %) | 0.98 (20 %) | 1.83 (37 %) | 64 % / 40 % |

The remaining S-arm rows are in S:waterfall.txt. Their shares differ from the D arm by up to 10 points (Q8_0 12.5 %:
helper rate 25 % on S vs 15 % on D).

**Ranking over all 22 configurations** (mean share of the excess over the bound, range, ms/token):

| rank | component | share (mean, range) | ms/token | actionable? |
|---|---|---|---|---|
| 1 | O: cache-path per-layer kernels, hand-off, host policy, admissions | 37 % (17–60 %) | 0.78–1.88 | **Yes. The largest fixed cost, and it grows in relative terms on faster GPUs.** |
| 2 | Balance + prefetch (the bound's CPU+GPU bandwidth aggregation) | 27 % (15–45 %) | 0.69–1.50 | Mostly not. Moving mc* ≈ 1.2 extra experts per layer to the helpers costs f + h + m·c_e ≈ 0.2 ms per layer against a 0.04 ms GPU saving on 20b. It becomes attainable only if per-request costs approach 0 and helpers reach read bandwidth. |
| 3 | Helper per-request fixed cost f | 16 % (6–23 %) | 0.16–1.43 | Yes: four barrier phases per request (≈50 µs on gpt-oss; 18–26 µs on Qwen3). |
| 4 | Helper throughput below read bandwidth | 13 % (4–25 %) | 0.10–1.66 | Partly: the kernel runs at 78–81 % of the 30-thread read bandwidth, the ceiling of the current kernel. |
| 5 | DFA's misses above MIN-bypass | 7 % (−3–17 %) | −0.09–1.15 | Yes at 12.5 % (+13.7–18.3 pp hit-rate headroom, §6). Negative at 50 %: extra misses help when the CPU would otherwise idle. |
| – | GPU dense | 0 by construction | – | Against the *physical* bound it is the largest term: TD runs at an effective 316 GB/s (20b) and 198 GB/s (Q4_K_M) of 600 GB/s. It is shared with llama.cpp. |

At 12.5 % the exposed CPU miss work dominates (69 % in the component view); at 50 % O dominates (47 %).

### 4.4 Per-component measurements from the diagnostic sweeps

| job | measurement | result | file |
|---|---|---|---|
| 012 | split-graph cache, 20b 25 %, regression | 84 µs per layer fixed, a copy costs its full PCIe time (0.48 ms), ≈100 µs per CPU expert; all-resident split cache 106.4 vs stock 135.6 tok/s | G:results/012_ec_diag/stdout.log, R:prereg/PROTOCOL.md phase-2 log |
| 014 | zero-copy vs copy engine; all-resident via `gpu_only` | 20b 67.4 vs 57.5; Q4_K_M 69.2 vs 56.0 tok/s; all-resident 135.5 vs stock 133.9 (**no per-layer cost when C = E**) | G:results/014_ec_zc/*_ec.jsonl |
| 015 | overlap on/off, κ, split mode | overlap +7.4–12.6 %; κ +1.7 % (20b C8) to +16.6 % (120b C32) | G:results/015_ec_overlap |
| 016 | mailbox vs split; 28 vs 14 helpers; helpers with llama's layout | mailbox +14 % (20b), +23 % (Q4_K_M), +16 % (Q8_0), +16 % (120b) at 25 %; 28 helpers +4–8 %; helpers static 1.08–1.15× llama | G:results/016_ec_mailbox |
| 017 | helpers 24/28/29 pinned, zero-copy blocks 8/32, max_admit 2/6, half-life 8 | baseline unstable before 018 (per-sequence CV up to 0.20, a step 79× the median); max_admit 2 costs Q4_K_M −32 % | G:results/017_ec_knobs |
| 018 | chunked helpers, AVX-512 kernel, pinning; cpubench | repeats within ±2.2 %; AVX-512 +0.1–1.4 %; pinning −1.5 to −0.2 %; I4 read 151.4 GB/s at 30 threads, MXFP4 kernel 106.8–120.2 GB/s | G:results/018_ec_helpers |
| 019 | calibration (H17 fit), all-CPU anchors | fit max APE 0.1–2.5 %; helpers vs llama with all experts on the CPU: 1.11–1.12× (017: 53.7 vs 48.5; 64.0 vs 57.1) | G:results/019_ec_calib, R:prereg/a10_mb/predictions.md |
| 038 | cpubench on I6 | read 166.1; MXFP4 AVX-512 134.8; Q8_0 151.6 GB/s (30 threads); single-thread MXFP4 8.4 GB/s | G:results/038_ec_ctrl_d@a10/cpubench.txt |

---

## 5. Projections for the race hardware

### 5.1 Inputs

- **GPUs**: RTX PRO 6000 Server Edition, 1597 GB/s; RTX 5090, 1792 GB/s. PCIe 5.0 ×16 at 63 GB/s. Host read bandwidth 150 / 250 / 400 GB/s.
- **Bytes**: GGUF-exact for the four A10 GGUFs (R:data/gguf_bytes.json, dense + head, KV at ctx 1024).
  Qwen3.6-35B-A3B uses R:data/shapes.json (L 40, E 256, k 8, 3.15 M params per expert, 1.43 B dense) with Qwen3
  Q4_K_M's effective bits (4.75 per expert, 5.35 dense). That is an assumption.
- **Routing**: the S-arm trace packs, full response streams:
  - `032_trace_small@40gb/gpt-oss-20b_S.npz`
  - `036_trace_retry@40gb/gpt-oss-120b_S.npz` (35 conversations)
  - `035_trace_rest@40gb/qwen3-30b-a3b_fp8_S.npz`

  Qwen3.6 has no trace, so it uses independent routing (as `mosl.calc` does), which is pessimistic for hit rates.
- **Budgets**: C = ⌊E·q⌋ per layer; llama.cpp n_cpu_moe = L − ⌊L·q⌋ (as `mosl.calc`).
- **Physical SoL**: `perfmodel.speed_of_light_time` with every η = 1 and τ = 0. Pooled M* = full-trace M* × (pooled / per-layer) on an 8000-token prefix.
- **M4**: `mosl.calc.m4_params()` (η_g 0.382, η_c 0.599, η_p 0.863, τ_e 20.7 µs, τ_x 5.8 ms), band 0.775–1.314 (`calc.band()`).

A spot check with `python -m mosl.calc` (gpt-oss-20b, 8 experts per layer, PRO 6000, 250 GB/s, S trace) reproduces
657.1 tok/s for the SoL and 94.7 [73.4–124.4] for M4.

### 5.2 Physical speed-of-light (tok/s, per-layer budget / pooled budget; S:projections.json)

| model | budget (C, n_cpu_moe) | M* | PRO 6000 @150 | @250 | @400 | 5090 @150 | @250 | @400 |
|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b MXFP4 | 12.5 % (16, 32) | 0.85 | 375 / 393 | 463 / 473 | 491 / 491 | 387 / 406 | 484 / 504 | 545 / 545 |
| | 25 % (32, 27) | 0.39 | 460 / 460 | 473 / 473 | 491 / 491 | 514 / 514 | 527 / 527 | 545 / 545 |
| | 45 % (57, 20) | 0.13 | 460 / 460 | 473 / 473 | 491 / 491 | 514 / 514 | 527 / 527 | 545 / 545 |
| gpt-oss-20b MXFP4 | 12.5 % (4, 21) | 1.50 | 354 / 370 | 457 / 475 | 572 / 590 | 363 / 379 | 473 / 492 | 599 / 619 |
| | 25 % (8, 18) | 0.72 | 600 / 639 | 657 / 657 | 680 / 680 | 624 / 667 | 732 / 732 | 756 / 756 |
| | 45 % (14, 14) | 0.26 | 640 / 640 | 657 / 657 | 680 / 680 | 715 / 715 | 732 / 732 | 756 / 756 |
| Qwen3-30B-A3B Q4_K_M | 12.5 % (16, 42) | 1.80 | 637 / 664 | 799 / 827 | 887 / 887 | 656 / 684 | 832 / 862 | 985 / 985 |
| | 25 % (32, 36) | 0.74 | 829 / 829 | 853 / 853 | 887 / 887 | 926 / 926 | 950 / 950 | 985 / 985 |
| | 45 % (57, 27) | 0.21 | 829 / 829 | 853 / 853 | 887 / 887 | 926 / 926 | 950 / 950 | 985 / 985 |
| Qwen3-30B-A3B Q8_0 | 12.5 % (16, 42) | 1.80 | 374 / 390 | 474 / 491 | 535 / 535 | 383 / 401 | 492 / 511 | 594 / 594 |
| | 25 % (32, 36) | 0.74 | 498 / 498 | 514 / 514 | 535 / 535 | 556 / 556 | 572 / 572 | 594 / 594 |
| | 45 % (57, 27) | 0.21 | 498 / 498 | 514 / 514 | 535 / 535 | 556 / 556 | 572 / 572 | 594 / 594 |
| Qwen3.6-35B-A3B ~Q4_K_M (independent routing) | 12.5 % (32, 35) | 4.59 | 451 | 560 | 672 | 464 | 584 | 711 |
| | 25 % (64, 30) | 3.24 | 573 | 682 | 787 | 595 | 720 | 841 |
| | 45 % (115, 22) | 1.87 | 791 | 842 | 859 | 834 | 941 | 959 |

The bound is capacity-limited only at 12.5 % or on slow hosts. Elsewhere it is the bandwidth-aggregation optimum and
budget-independent.

### 5.3 Predicted llama.cpp `--n-cpu-moe` (M4, third-party fit), tok/s [band]

| model | budget | PRO 6000 @150 | @250 | @400 | 5090 @150 | @250 | @400 |
|---|---|---|---|---|---|---|---|
| gpt-oss-120b | 12.5 % | 40 [31–53] | 58 [45–77] | 77 [60–102] | 41 [32–54] | 59 [46–78] | 80 [62–105] |
| | 25 % | 46 [36–60] | 65 [50–85] | 85 [66–111] | 47 [36–61] | 67 [52–88] | 87 [68–115] |
| | 45 % | 57 [44–74] | 77 [60–101] | 97 [75–128] | 58 [45–76] | 80 [62–105] | 102 [79–134] |
| gpt-oss-20b | 12.5 % | 60 [47–79] | 86 [67–113] | 113 [88–149] | 61 [48–81] | 88 [68–116] | 117 [91–154] |
| | 25 % | 68 [52–89] | 95 [73–124] | 122 [95–161] | 69 [53–91] | 97 [75–128] | 127 [98–167] |
| | 45 % | 80 [62–105] | 109 [85–143] | 137 [106–180] | 82 [64–108] | 113 [88–149] | 143 [111–188] |
| Qwen3 Q4_K_M | 12.5 % | 52 [40–68] | 66 [51–87] | 79 [61–103] | 52 [40–68] | 67 [52–88] | 80 [62–105] |
| | 25 % | 59 [45–77] | 75 [58–98] | 88 [68–116] | 59 [46–78] | 76 [59–100] | 90 [70–118] |
| | 45 % | 73 [57–96] | 92 [71–121] | 107 [83–141] | 75 [58–98] | 94 [73–123] | 110 [85–144] |
| Qwen3 Q8_0 | 12.5 % | 35 [27–46] | 48 [37–63] | 60 [46–78] | 35 [28–47] | 48 [38–64] | 61 [47–80] |
| | 25 % | 40 [31–52] | 53 [41–70] | 66 [51–87] | 40 [31–53] | 54 [42–71] | 68 [52–89] |
| | 45 % | 49 [38–65] | 65 [50–85] | 78 [61–103] | 50 [39–66] | 66 [52–87] | 81 [63–107] |
| Qwen3.6-35B-A3B | 12.5 % | 71 [55–94] | 86 [66–113] | 96 [75–127] | 73 [56–96] | 88 [68–115] | 99 [77–130] |
| | 25 % | 80 [62–105] | 96 [74–126] | 107 [83–141] | 82 [64–108] | 98 [76–129] | 110 [85–145] |
| | 45 % | 100 [77–131] | 117 [91–154] | 130 [100–170] | 103 [80–135] | 121 [94–159] | 135 [104–177] |

M4 is conservative for current llama.cpp. On the audit's datasheet basis, measured/predicted is median 1.22 (A10),
0.83 (A100 host) and 0.86 (GH200) (R:prereg/PROTOCOL.md §4.7 outcomes). It over-charges Qwen3 through τ_e = 20.7 µs
per CPU expert: on the A10 the Qwen3 slope was over-predicted by 72–73 % (R:prereg/a10/scored.md H4). The scenario
bandwidths are fed in as B_c; M4's η_c = 0.60 was fitted against peak, not measured read, bandwidth.

### 5.4 Our system: can H17 be re-parameterised?

**No, not H17 as fitted.** Its coefficients are regression mixtures (a0 < 0 is net of GPU work replaced; β exceeds a
full PCIe copy on Qwen3). None of them scales with a named hardware quantity. Instead I built a **structural model**
(S:project.py) from the same counters. Its constants are A10-measured unless marked "assumed":

- **GPU**: a = s/(η_v·B_G) + t_e and TD = D/(η_v·B_G) + L·t_layer, with:
  - η_v = 0.73, the A10 expert GEMV at 440 GB/s over 600 datasheet (nsys);
  - t_layer = 62 µs (gpt-oss) or 48 µs (Qwen3) per layer of non-GEMV kernels (nsys);
  - t_e = 4.1 µs (gpt-oss) or 0.6 µs (Qwen3) per expert;
  - **assumed**: t_layer and t_e scale by 0.7 on Blackwell (clocks), and η_v is unchanged.
- **Helpers**: c_e = s/(η_h·B_C) with η_h = 0.78 (MXFP4), 0.75 (Q4_K), 0.81 (Q8_0); f = 50 / 19 / 25 µs.
  **Assumed**: both are unchanged. At 400 GB/s this needs about 11 GB/s per helper thread against 8.4 GB/s measured
  single-thread on Ice Lake, so the 400 GB/s rows are optimistic unless the host has ≥ 40–48 fast cores.
- **O**: 35 µs per MoE layer per token. **Assumed**: 0.5–1.0× on Blackwell; this is the reported range.
- **Counts**: requests by m per token come from DFA (κ as deployed) replayed on the S traces; hit rates are hardware-independent.
- **llama.cpp**: per CPU layer σ + k·s/(η_ll·B_C), with η_ll = 0.66 / 0.61 / 0.77 (MXFP4 / Q4_K / Q8_0, from the phase-1
  slopes) and σ = 15 µs.
- **Back-test on A10** (in-sample: the constants come from these runs):
  - ours: median 2.5 %, max 5.9 %;
  - llama.cpp: median 5.6 %, max 12.8 % (against the noisy 038 baselines);
  - all-GPU: 20b 127.8 vs 135.9 tok/s (−6 %), Q4_K_M 139.9 vs 144.0 (−3 %).

**RTX PRO 6000 SE** (tok/s). "O = 0" removes the per-layer overhead; "O = f = 0" also removes the helper
per-request cost. Speed-ups are as-is ranges.

| model | budget | host | impl. SoL | all-GPU | llama.cpp | ours as-is (O ×1 … ×0.5) | O = 0 | O = f = 0 | ours / llama.cpp | ours / M4 |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b | 12.5 % | 150 | 208 | 196 | 48 | 81–85 | 90 | 103 | 1.69–1.78 | 1.99–2.10 |
| | | 250 | 215 | 196 | 71 | 101–108 | 116 | 138 | 1.42–1.51 | 1.73–1.85 |
| | | 400 | 223 | 196 | 98 | 118–127 | 138 | 171 | 1.20–1.29 | 1.52–1.64 |
| | 25 % | 150 | 208 | 196 | 54 | 105–112 | 120 | 136 | 1.93–2.06 | 2.28–2.44 |
| | | 250 | 215 | 196 | 79 | 122–132 | 144 | 166 | 1.54–1.66 | 1.87–2.03 |
| | | 400 | 223 | 196 | 107 | 134–146 | 161 | 190 | 1.26–1.37 | 1.58–1.73 |
| | 45 % | 150 | 208 | 196 | 67 | 131–142 | 156 | 169 | 1.95–2.13 | 2.31–2.52 |
| | | 250 | 215 | 196 | 94 | 141–154 | 171 | 186 | 1.50–1.65 | 1.82–2.00 |
| | | 400 | 223 | 196 | 121 | 147–162 | 180 | 197 | 1.22–1.34 | 1.51–1.67 |
| gpt-oss-20b | 12.5 % | 150 | 265 | 282 | 72 | 97–101 | 106 | 121 | 1.35–1.41 | 1.61–1.68 |
| | | 250 | 306 | 282 | 106 | 127–134 | 142 | 170 | 1.20–1.26 | 1.48–1.56 |
| | | 400 | 319 | 282 | 145 | 154–164 | 177 | 222 | 1.06–1.13 | 1.36–1.45 |
| | 25 % | 150 | 298 | 282 | 80 | 128–135 | 143 | 164 | 1.59–1.68 | 1.89–2.00 |
| | | 250 | 307 | 282 | 117 | 156–167 | 180 | 214 | 1.34–1.43 | 1.65–1.77 |
| | | 400 | 319 | 282 | 156 | 179–193 | 210 | 258 | 1.14–1.24 | 1.46–1.58 |
| | 45 % | 150 | 298 | 282 | 96 | 171–185 | 200 | 222 | 1.79–1.93 | 2.14–2.30 |
| | | 250 | 307 | 282 | 134 | 191–208 | 228 | 256 | 1.43–1.55 | 1.75–1.91 |
| | | 400 | 319 | 282 | 173 | 205–224 | 247 | 280 | 1.18–1.29 | 1.49–1.63 |
| Qwen3 Q4_K_M | 12.5 % | 150 | 297 | 285 | 73 | 112–124 | 139 | 157 | 1.54–1.70 | 2.18–2.40 |
| | | 250 | 304 | 285 | 106 | 136–154 | 177 | 206 | 1.29–1.46 | 2.06–2.32 |
| | | 400 | 313 | 285 | 141 | 155–178 | 209 | 249 | 1.10–1.26 | 1.96–2.26 |
| | 25 % | 150 | 297 | 285 | 82 | 142–161 | 186 | 210 | 1.73–1.96 | 2.42–2.75 |
| | | 250 | 304 | 285 | 116 | 160–185 | 218 | 249 | 1.38–1.59 | 2.14–2.48 |
| | | 400 | 313 | 285 | 152 | 172–201 | 242 | 275 | 1.13–1.33 | 1.96–2.29 |
| | 45 % | 150 | 297 | 285 | 100 | 170–199 | 239 | 259 | 1.71–2.00 | 2.33–2.71 |
| | | 250 | 304 | 285 | 136 | 180–212 | 257 | 277 | 1.32–1.55 | 1.96–2.30 |
| | | 400 | 313 | 285 | 172 | 185–219 | 269 | 286 | 1.08–1.28 | 1.73–2.05 |
| Qwen3 Q8_0 | 12.5 % | 150 | 225 | 214 | 55 | 83–89 | 96 | 107 | 1.51–1.63 | 2.35–2.53 |
| | | 250 | 232 | 214 | 80 | 104–115 | 127 | 146 | 1.30–1.43 | 2.19–2.40 |
| | | 400 | 240 | 214 | 109 | 123–137 | 155 | 182 | 1.13–1.26 | 2.06–2.29 |
| | 25 % | 150 | 225 | 214 | 61 | 109–120 | 133 | 149 | 1.78–1.96 | 2.74–3.02 |
| | | 250 | 232 | 214 | 88 | 127–142 | 161 | 182 | 1.44–1.61 | 2.38–2.66 |
| | | 400 | 240 | 214 | 117 | 140–158 | 183 | 205 | 1.19–1.35 | 2.12–2.40 |
| | 45 % | 150 | 225 | 214 | 74 | 136–154 | 177 | 191 | 1.83–2.06 | 2.76–3.12 |
| | | 250 | 232 | 214 | 103 | 146–166 | 193 | 207 | 1.41–1.61 | 2.25–2.57 |
| | | 400 | 240 | 214 | 132 | 152–174 | 204 | 215 | 1.15–1.32 | 1.94–2.22 |
| Qwen3.6-35B-A3B (independent routing; *Qwen3-30B trace proxy*) | 12.5 % | 150 | 295 | 315 | 113 | 107–116 (*147–164*) | 126 | 139 | 0.95–1.02 | 1.50–1.62 |
| | | 250 | 321 | 315 | 152 | 134–147 (*168–190*) | 164 | 188 | 0.88–0.97 | 1.56–1.72 |
| | | 400 | 336 | 315 | 190 | 155–174 (*183–209*) | 199 | 234 | 0.82–0.92 | 1.61–1.81 |
| | 25 % | 250 | 329 | 315 | 165 | 141–157 (*189–217*) | 176 | 203 | 0.86–0.95 | 1.48–1.64 |
| | 45 % | 250 | 329 | 315 | 189 | 155–174 (*206–240*) | 198 | 233 | 0.82–0.92 | 1.32–1.49 |

**RTX 5090, 250 GB/s host.** Values are 1–7 % higher than on the PRO 6000.

| model | budget | impl. SoL | all-GPU | llama.cpp | ours as-is | ours / llama.cpp | VRAM needed at this budget (GB, incl. 1.5 GB) |
|---|---|---|---|---|---|---|---|
| gpt-oss-120b | 12.5 / 25 / 45 % | 228 | 210 | 72 / 80 / 96 | 103–110 / 125–136 / 147–162 | 1.43–1.53 / 1.56–1.69 / 1.53–1.69 | 10.9 / 18.5 / 30.7 (tight) |
| gpt-oss-20b | same | 316–327 | 302 | 108 / 119 / 137 | 129–137 / 160–172 / 199–217 | 1.20–1.27 / 1.35–1.45 / 1.45–1.58 | 4.1 / 5.4 / 7.4 |
| Qwen3 Q4_K_M | same | 319 | 301 | 107 / 118 / 139 | 138–156 / 163–189 / 185–219 | 1.29–1.46 / 1.39–1.60 / 1.33–1.58 | 4.6 / 6.8 / 10.3 |
| Qwen3 Q8_0 | same | 246 | 230 | 81 / 90 / 106 | 106–117 / 130–146 / 151–173 | 1.31–1.43 / 1.45–1.63 / 1.43–1.64 | 6.8 / 10.6 / 16.8 |
| Qwen3.6-35B-A3B | same | 332–348 | 334 | 156 / 168 / 194 | 136–150 / 144–160 / 158–178 | 0.87–0.97 / 0.85–0.95 / 0.81–0.92 | 5.3 / 7.7 / 11.5 |

**Reading.**
- On the race hardware the GPU step shrinks about 2.5× while the helpers' per-request cost and the per-layer O barely
  change. Our advantage then depends mainly on host bandwidth: 1.35–2.1× at 150 GB/s, 1.2–1.66× at 250,
  1.06–1.37× at 400 (gpt-oss, Qwen3-30B).
- The A10 margin of 1.31–1.66× is not guaranteed on a 400 GB/s host.
- Removing O is the largest single lever: +9–45 %, largest on the 48-layer Qwen3. Removing f as well adds another +5–25 %.
- Admission copies over PCIe 5 cost 0.24–2.1 ms/token of link time (DFA κ as deployed), which is feasible. κ should be
  re-tuned: the reuse break-even p/(b−a) is ≈11 on the A10; on the race hardware it is ≈3 at 150 GB/s, ≈5 at 250 GB/s
  and ≈11 at 400 GB/s host bandwidth.
- gpt-oss-120b benefits most: its large experts make each llama.cpp CPU layer expensive.

**What must be calibrated on the race machine before any number above is trusted** (about one GPU-hour):
1. An all-GPU run of every model that fits, with nsys: gives a, TD, t_layer and t_e.
2. `llama-ec-cpubench` at 1…N threads plus one static-helper sweep over 3 values of n: gives η_h, f, h and whether the helpers are compute-capped.
3. A mailbox-cache run at two budgets with stats: gives O.
4. A llama.cpp `--n-cpu-moe` sweep over ≥ 5 values of n with 3 interleaved repeats: gives η_ll, σ and variance.
5. STREAM or read bandwidth and pinned H2D bandwidth.

With those, `project.py` becomes a prediction rather than a scenario, and the implementation-relative SoL can be
computed the same way as in §4.1.

### 5.5 Memory fit

| model | GGUF (GB) | all experts (pinned by our cache, GB) | dense + head + KV at 1k (GB) | fits 90 GB host RAM | fits 180 GB | all-GPU on PRO 6000 (96 GB) | all-GPU on 5090 (32 GB) |
|---|---|---|---|---|---|---|---|
| gpt-oss-120b MXFP4 | 63.4 | 61.1 | 1.73 | yes, one engine (≈ 69 GB with an 8 GB OS reserve); **no** for two copies (for example an engine that repacks, or two engines resident) | yes, with ~110 GB of headroom | yes | no |
| gpt-oss-20b MXFP4 | 12.1 | 10.2 | 1.33 | yes | yes | yes | yes |
| Qwen3-30B-A3B Q4_K_M | 18.6 | 17.6 | 0.92 | yes | yes | yes | yes |
| Qwen3-30B-A3B Q8_0 | 32.5 | 30.8 | 1.45 | yes | yes | yes | no |
| Qwen3.6-35B-A3B ~Q4_K_M | ≈ 20.8 (estimate) | 19.1 | 1.38 | yes | yes | yes | yes |

Sources: `file_tensor_bytes` and `routed_bytes_total` in R:data/gguf_bytes.json; S:projections.json.

- Our cache pins **all** experts, including GPU-resident ones, and loads without mmap.
- llama.cpp with mmap needs page cache for the CPU layers' experts.
- On the 5090 every budget up to 45 % fits; gpt-oss-120b at 45 % needs ≈ 30.7 GB and is tight at long contexts.
- On the RTX PRO 6000 every model fits entirely in VRAM, so a budget must be enforced in the same way for all entrants.

---

## 6. Cache hit rates (full response streams, conversations back to back; S:hitrates.json)

Policies:
- per-layer cache of C experts;
- `cachesim` LRU/LFU/MIN with fetch semantics, and MIN-bypass (optimal when misses may run on the CPU);
- static = the hindsight top-frequency set;
- DFA = the deployed policy (`ecsim_fast`: half-life 16, κ = 1 gpt-oss / 2 Qwen3, 2-step publication);
- pooled = one MIN-bypass budget over all layers, on an 8000-token prefix (with the per-layer value on the same prefix).

At C = k (gpt-oss-20b, 12.5 %) the fetch policies coincide by construction.

| model | arm (tokens, convs) | budget (C) | static | LRU | LFU | DFA | MIN | MIN-bypass | pooled MIN-b (per-layer, same prefix) | admissions per token: LRU / DFA / MIN-b |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-20b | D (86k, 160) | 12.5 % (4) | 0.294 | 0.499 | 0.499 | 0.497 | 0.499 | 0.655 | 0.670 (0.649) | 48.1 / 1.8 / 16.3 |
| | | 25 % (8) | 0.467 | 0.707 | 0.639 | 0.706 | 0.806 | 0.836 | 0.852 (0.835) | 28.1 / 2.7 / 9.8 |
| | | 45 % (14) | 0.665 | 0.870 | 0.787 | 0.867 | 0.932 | 0.938 | 0.956 (0.942) | 12.5 / 2.4 / 4.2 |
| | S (152k, 160) | 12.5 % | 0.254 | 0.464 | 0.464 | 0.443 | 0.464 | 0.626 | 0.640 (0.620) | 51.5 / 2.2 / 18.1 |
| | | 25 % | 0.452 | 0.675 | 0.635 | 0.673 | 0.786 | 0.819 | 0.835 (0.817) | 31.2 / 2.9 / 10.9 |
| | | 45 % | 0.679 | 0.861 | 0.806 | 0.861 | 0.927 | 0.934 | 0.948 (0.934) | 13.4 / 2.4 / 4.5 |
| | G (145k, 160) | 25 % | 0.430 | 0.664 | 0.616 | 0.678 | 0.780 | 0.815 | 0.828 (0.810) | 32.2 / 2.7 / 10.7 |
| gpt-oss-120b | D (86k, 160) | 12.5 % (16) | 0.496 | 0.645 | 0.589 | 0.667 | 0.789 | 0.804 | 0.825 (0.809) | 51.1 / 7.6 / 16.7 |
| | | 25 % (32) | 0.664 | 0.821 | 0.735 | 0.825 | 0.911 | 0.914 | 0.933 (0.922) | 25.8 / 7.4 / 8.6 |
| | | 45 % (57) | 0.819 | 0.935 | 0.863 | 0.927 | 0.970 | 0.971 | 0.980 (0.975) | 9.3 / 4.0 / 3.4 |
| | S (33k, 35) | 12.5 % | 0.355 | 0.624 | 0.511 | 0.618 | 0.772 | 0.788 | 0.797 (0.783) | 54.1 / 8.9 / 19.0 |
| | | 25 % | 0.561 | 0.802 | 0.676 | 0.795 | 0.900 | 0.904 | 0.912 (0.903) | 28.6 / 8.7 / 9.9 |
| | | 45 % | 0.777 | 0.924 | 0.841 | 0.916 | 0.966 | 0.967 | 0.971 (0.966) | 10.9 / 4.6 / 3.8 |
| Qwen3-30B-A3B | D (95k, 160) | 12.5 % (16) | 0.355 | 0.650 | 0.576 | 0.634 | 0.764 | 0.795 | 0.802 (0.790) | 134.5 / 7.8 / 48.7 |
| | | 25 % (32) | 0.515 | 0.826 | 0.689 | 0.809 | 0.909 | 0.915 | 0.927 (0.918) | 66.9 / 8.3 / 23.6 |
| | | 45 % (57) | 0.712 | 0.944 | 0.826 | 0.927 | 0.975 | 0.976 | 0.984 (0.980) | 21.5 / 4.6 / 7.6 |
| | S (136k, 160) | 12.5 % | 0.255 | 0.614 | 0.522 | 0.601 | 0.740 | 0.775 | 0.791 (0.779) | 148.3 / 8.3 / 52.9 |
| | | 25 % | 0.431 | 0.809 | 0.648 | 0.795 | 0.900 | 0.907 | 0.917 (0.907) | 73.5 / 8.7 / 25.1 |
| | | 45 % | 0.653 | 0.939 | 0.799 | 0.923 | 0.973 | 0.974 | 0.979 (0.973) | 23.4 / 4.8 / 8.1 |

The G arm is within 0.6 pp of S for DFA and MIN-bypass, and within 1.3 pp for the static set (S:hitrates.json).

**Own text vs dataset text** (S − D, percentage points; gpt-oss-120b's arms cover different conversations, so its
difference is unpaired):

| model | budget | DFA | MIN-bypass | LRU |
|---|---|---|---|---|
| gpt-oss-20b | 12.5 / 25 / 45 % | −5.4 / −3.3 / −0.6 | −2.9 / −1.7 / −0.4 | −3.6 / −3.2 / −0.9 |
| gpt-oss-120b | same | −4.9 / −3.0 / −1.1 | −1.6 / −1.1 / −0.4 | −2.1 / −2.0 / −1.1 |
| Qwen3-30B-A3B | same | −3.3 / −1.4 / −0.4 | −2.1 / −0.7 / −0.2 | −3.6 / −1.7 / −0.5 |

**Reading.**
- DFA does not beat LRU on hits: it is within −2.1 to +2.2 pp, and slightly below LRU on Qwen3 at every budget and on
  gpt-oss-120b S. It makes **2–27× fewer admissions than LRU**, which is what made it fast on 25 GB/s PCIe.
- MIN-bypass leaves +13.7–18.3 pp of headroom over DFA at 12.5 %, +9–15 pp at 25 % and +4–7 pp at 45 %. It needs up to
  9× DFA's admissions at 12.5 % and 0.9–3.6× at 25–45 %.
- Pooling the budget across layers adds 0.4–2.1 pp.
- With PCIe 5 making copies ≈2.5× cheaper, the policy trade-off moves toward more admissions. Measured hit rates on
  the A10 windows match these (for example 20b S 25 %: 0.662 measured vs 0.673 full-stream).

---

## 7. Data quality issues

| severity | issue | evidence | consequence |
|---|---|---|---|
| **High** | llama.cpp baseline runs shift by up to ~8 % run to run on one instance, and sequences can be bimodal. Phase 4.5 has no repeats. | 038 20b n21: every sequence 16.46 ms median step vs 15.44 ms on I5 and on 039a (same instance). Q8_0 n36: 5/12 sequences below 50 tok/s on 038 and on 039a vs 1/12 on I5. | The paper's range ends (1.66× Q8_0 25 % D; 1.31× 20b 12.5 % S) and the "own text lowers speed-ups" reading are within baseline noise. On I5 the D-arm range is 1.37–1.60×. |
| **High** | The implementation-relative SoL leaves the LM head out of the dense bytes. | `sol_a10.py`/`sol_windows.py` MODELS use `dense_bytes` (687 MB, 20b); `preregister.py` uses `dense_bytes + head_bytes`. Nsight: experts are 45 % / 42 % of the all-GPU step, vs 65 % implied. | SoL 181 → 156 (20b) and 197 → 164 (Q4_K_M) tok/s; fractions 47–64 % → 54–74 %. Headline numbers in the paper need a correction note. |
| **High** (for decomposition) | No profiler trace of the cache. | G:results/012_ec_diag/nsys_ec_*_gpu_trace.csv.gz are 0 bytes; 007 traces are stock llama.cpp only. | The split of O (0.8–1.9 ms/token) between hand-off, per-layer kernels, host policy and admission interference is inferred, not measured. The static-helper fit gives a negative intercept. |
| Medium | No all-GPU step for Q8_0 or gpt-oss-120b on any instance. | They do not fit 24 GB. | Their T0 (and so SoL, TD, O) comes from extrapolation. For 120b the I1- and I4-based intercepts differ by 1 ms (10.0 vs 11.0): ±5 % on the SoL, ±0.5 ms on O. |
| Medium | Instances differ, and I5 has no host bandwidth measurement. | I4 151.4 GB/s, I6 166.1 GB/s at 30 threads; 025 has no cpubench. | Cross-instance comparisons (phase 3 vs 4.5, I5 vs I6) cannot be normalised by bandwidth. |
| Medium | The protocol says 014–015 ran on 012's instance; the logs show a new instance. | 014 `stdout.log`: fresh `git clone`, 4 model downloads, after 013's termination. | Exploratory 012 vs 014+ comparisons are cross-instance. |
| Medium | Patch version drift within the case study. | Patch sha 23ad9d51 (019), 9410e3a4 (020, the pre-registered one), 465361ad (021–039). The diffs touch only `tools/ec-bench/ec-bench.cpp` (verified). | The runtime is identical; the protocol's sha applies to 020 only. |
| Medium | The H17 coefficients are not physical. | a0 = −38 to −103 µs; β (166 µs) > a full copy's PCIe time (111 µs) on Q4_K_M; 3 cache points per model. | It predicts well in place (5.0 % median APE) but cannot be moved to new hardware. |
| Low | Patch documentation vs code. | `LLAMA_EC_PACED` documented default 1, coded 0; `LLAMA_EC_MXK` undocumented; `LLAMA_EC_KAPPA` default 0 while every evaluated run sets 1 or 2. | A user reproducing from the header gets a different system. |
| Low | gpt-oss-120b arm-D text is off-distribution. | NLL 5.75 vs 0.58 nats (S); top-1 cache ~ llama 0.89–0.90 (D) vs 0.98–0.99 (S). | Use S for any 120b fidelity or locality claim (as the protocol does). |
| Low | gpt-oss-120b S/G traces have 35 conversations (D has 160). Qwen3 on-policy text was generated with FP8 weights. | R:data/traces_manifest.json; protocol §4.1 deviations. | Unpaired S−D for 120b in §6. |
| Low | Failed or void jobs: 003 (34 rows rc = 127), 004 (0-byte outputs), 010 and 013 (aborted), 026 (void), 020c rc = 1 (summary only). The phase-2 predictions `R:prereg/a10_ec` were never scored. | – | Documented in the protocol except the unscored phase-2 file. |
| Low | The llama.cpp baseline is ec-bench's `--ncmoe` tensor override inside the patched build, not stock `llama-bench`/`llama-server`. | Only 021 (`--no-mmap`: −13.4 % to +1.7 %) checks a variant. | Baseline equivalence to what users run is assumed. |
| Low | A first-step penalty after each prefill is included in tok/s. | Cache: first step 1.77× the median; first 5 steps 4.2 % of time. | About −1.5 % on the cache's tok/s. It does not change conclusions. |

---

## 8. Recommended follow-ups

1. **Profile the mailbox cache before changing it.**
   - Take an nsys trace (CUDA graph node tracing plus OS runtime) of one cache configuration and one static-helper
     configuration per model, with `LLAMA_EC_STATS`, on the race GPU.
   - Add two controls: `policy=static` with a DFA-derived `LLAMA_EC_INIT` (no admissions), and one C = E−1 layer
     (per-layer path cost without misses).
   - This splits O (37 % of the excess on the A10) into kernels, hand-off, host policy and admission interference. It
     decides whether the next change is fusing `ec_req`/`ec_wait`/`get_rows` into the router and expert kernels, moving
     the policy off the critical path (its admissions only take effect two steps later anyway), or pacing copies.
2. **Run a race-machine calibration hour (§5.4)** and re-run `S:project.py` with measured constants.
   - Report the implementation-relative SoL with the head included (or the Nsight split) for every entrant.
   - Measure helper throughput per thread count to check the compute cap at high host bandwidth; at 400 GB/s the
     projection assumes 11 GB/s per helper thread.
3. **Settle baseline variance.**
   - Run every llama.cpp and cache configuration at least 3× interleaved on one instance, plus one repeat on a second
     instance; report medians and ranges.
   - Cross-check ec-bench `--ncmoe` against stock `llama-bench -ncmoe` and `llama-server` at the race commit, with and
     without mmap and repack.
   - Re-derive the A10 speed-up range from medians (I5 already suggests 1.37–1.60×).
4. **Re-tune the policy for PCIe 5 in simulation**, cheaply on CPU with `ecsim_fast` + `cachesim` + the structural model.
   - κ for the new break-even reuse count (≈3–11 depending on host bandwidth, ≈5 at 250 GB/s, instead of ≈11).
   - Pooled vs per-layer budgets (+0.4–2.1 pp).
   - Prefetch at 63 GB/s.
   - "Work shifting" (sending hits to idle helpers) under measured f.
   - Target the +9–18 pp MIN-bypass headroom at 12.5–25 %.
5. **Trace Qwen3.6-35B-A3B (S arm) and check the patch on it.** Current projections for it are 0.78–1.02× llama.cpp under
   independent routing. Check:
   - whether its GGUF has merged `ffn_gate_up_exps`, which disables mailbox mode;
   - whether its architecture reaches `build_moe_ffn_ec` (SwiGLU, no clamp);
   - its real per-layer hit rates.

   With 40 layers, O alone is ≈1.4 ms/token on A10-like constants.
