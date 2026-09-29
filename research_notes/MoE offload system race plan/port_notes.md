# Engineering log: port, fixes, FETCH (28 Sep 2026)

Branch `ec-4da6337` of the local llama.cpp clone (scratchpad), exported as `runtime/llama.cpp-expert-cache-4da6337.patch`
(and `jobs/ec2/` on the `gpu` branch). The A10 case study's patch against 2145525a stays unchanged in
`runtime/llama.cpp-expert-cache.patch`.

## What changed

1. **Rebase onto llama.cpp 4da6337 (27 Sep 2026).** All 16 commits applied without conflicts; the CPU build and the
   sm_120 CUDA compile of every touched file pass (CUDA 12.8).
2. **Id -1 on every CUDA path.** New tensor flag `GGML_TENSOR_FLAG_NEG_IDS`, set by the cache graph on its
   MUL_MAT_ID/ADD_ID nodes.
   - Float weights (BF16/F16/F32): output zeroed, then the guarded `mul_mat_vec_f` kernel (new guard: a block
     whose id is negative writes a zero row and returns). This avoids `mmf` leaving -1 rows unwritten and the sync
     fallback's assert.
   - The sync fallback maps negative ids to a zeroed spare row.
   - `ggml_cuda_mul_mat_id_needs_sync` agrees with the new routing, so CUDA graphs stay on.
   - Scope: the cache path is decode-only (one token). `mmq`'s gather does not take -1, so multi-token -1 is
     unsupported; job 058 showed a multi-token test case aborting there, and those cases were removed.
3. **Stale-payload check in the helpers.** After reading LAYER/IDS, a helper re-reads RESP and backs off if the
   request is already answered (the GPU may by then have written the next request's payload). The T06 stress test
   (6/3/12 helpers, up to 40 % of calls delayed 50–450 µs, 1,500 back-to-back requests) passes before and after.
4. **FETCH (`LLAMA_EC_FETCH=T0,T1,...`, `LLAMA_EC_FETCH_BLOCKS`).** When m of a layer's selected experts miss, a GPU op
   copies T[m] of them into slots before the GPU path runs, and serves them on the GPU in the same step. It fetches in
   selection order (highest router weight first) into eviction candidates the host ranks by decayed score before
   every step, skipping slots in use this step and slots being loaded. The copy is a kernel reading the pinned host
   experts directly. The helpers get only the rest. The host applies the GPU's fetch log after the step. Stats:
   `fetches`, `fetch_table`. Off by default. Motivation: `admission_simulation.md` (+12–32 % on gpt-oss-120b
   simulated).
5. **Op tests.** EC cases for BF16, F16 and F32 as well as Q8_0/Q4_K/Q6_K/MXFP4; the test graphs carry the new flag.

## Tests so far

| Test | Where | Result |
|---|---|---|
| Tiny-model identity, cache vs stock (T03): gpt-oss 4L32E Q8_0, Qwen3-MoE 4L32E Q8_0 and F32, 6 configurations each (slots 0/1/3/4/16/32, DFA/LRU, check=1) | CPU, here | identical top-5 dumps in every configuration; 0 inconsistent ids; 6.5–54 % of experts served by the "device". Also identical after the maps widened to three rows |
| Mailbox stress T06 (generic kernels) | CPU, here | 0 bad values, before and after the fix |
| Mailbox stress with the AVX-512 MXFP4 kernel | CPU, here | ~2 % of requests exceed the harness's 1e-3 tolerance, deterministic: an activation's Q8_0 rounding flips after a tiny summation-order difference upstream. A test-tolerance issue, not a protocol bug; the harness needs a quantization-step tolerance for that path |
| Op tests `MUL_MAT_ID` (929 cases), patched and stock builds | RTX 5080 (job 058) | 929/929 both |
| `ADD_ID_EC` | RTX 5080 | 2/2 |
| `MUL_MAT_ID_EC` single-token Q8_0 | RTX 5080 | 4/4 OK before the removed multi-token case aborted the run; the rest rerun in job 059 |
| gpt-oss-20b own text, 12 prompts × 128 tokens, 25 % of experts (C = 8 of 32) | RTX 5080 + Ryzen 9 7900X (job 058) | cache 40.3 tok/s vs stock `--ncmoe 18` 18.5 tok/s (**2.18×**); all-GPU 246.6; top-1 agreement with stock 0.9948 (with all-GPU 0.9896); ΔNLL +0.03 %; hit rate 0.64 |

## Open

- FETCH has only been compiled, never run: job 059 runs it on the home-PC host after the baseline runs.
- Patch header defaults (`LLAMA_EC_PACED` documented as 1, code 0; `KAPPA`) still to correct.
- `llama-server` integration not yet exercised (the harness is ec-bench).
