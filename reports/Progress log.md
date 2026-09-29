# Progress log: option C measurement study

*Started Monday 28 September 2026. Newest entries first. Numbers link to result folders on the `gpu` branch
(`results/<job>/`).*

## 28 September, night: PREFETCH on the RTX 5090 host (job `066_prefetch_homepc@vast`)

Same machine as jobs 060 and 064 (RTX 5090 + Ryzen 9 9950X; CPU memory 46.5 GB/s, link 46.5 GB/s).

### Own text: 12 prompts × 128 tokens, every configuration in one process

| C of 128 | Cache | + PREFETCH q=1 | + PREFETCH q=2 | + FETCH | + PREFETCH q=1 + FETCH |
|---|---|---|---|---|---|
| 14 | 40.7 tok/s | 46.2–46.5 (**+14%**) | 45.3 (+11%) | 47.1 (+16%) | **49.7 (+22%)** |
| 32 | 62.7 | 72.7 (**+16%**) | 71.5 (+14%) | 74.7 (+19%) | **76.8 (+22%)** |
| 56 | 93.6 | 102.4–103.2 (**+10%**) | 102.0 (+9%) | **108.8 (+16%)** | 106.1 (+13%) |

- **The hit rates are as simulated:**
  - with q=1: 0.746 / 0.887 / 0.947, against simulated 0.746 / 0.886 / 0.948;
  - without prefetch: 0.541 / 0.758 / 0.875.
- **Prediction against measurement.** The prediction, recorded before the run in
  `prereg/homepc/prefetch_prediction_066.json`, was +17% / +17% / +14%. Measured is +14% / +16% / +10%: the timeline
  model is 1–4 points optimistic.
- **Repeats agree** within 1 point; the base is identical in both runs.
- **Accuracy:** 98.4–99.0% same next token as the cache alone; NLL within 0.6%.
- **What wins:**
  - On this host FETCH alone (+16–19%) beats PREFETCH alone (+10–16%).
  - Both together is best at the two smaller budgets (+22%). At the largest, FETCH alone is best (+16%).
  - Two prefetches per layer is worse than one everywhere, as the simulation said.

### The chat benchmark: job 064's client, AIME-25 problems 0–4 × 256 tokens, `llama-server`

| Budget | Cache (job 064) | + PREFETCH | + FETCH | FreeToken, better mode (job 064) |
|---|---|---|---|---|
| 11% (C = 14) | 42.9 tok/s | 44.6 (+4%) | **46.3 (+8%)** | 39.4 |
| 25% (C = 32) | 68.1 (68.1 again in job 066) | 72.5 (+6%) | **74.9 (+10%)** | 68.6 |
| 40% (C = 51) | 105.2 | 103.2 (−2%) | **112.3 (+7%)** | 108.9 |
| 44% (C = 56) | 114.8 | 111.6 (−3%) | **119.8 (+4%)** | does not fit |

- **With FETCH, our cache is ahead of FreeToken at all three budgets on this host:** +18% / +9% / +3%. The last is
  within the ±10% spread across problems.
- **Against llama.cpp at its best:** 1.9× / 2.7× / 3.3×.
- **The cache-only server run at C = 32 reproduced job 064 to 0.1%.**
- **Both options gain less on the chat benchmark than on the own-text runs.**
  - A likely reason: these prompts are short (109 tokens, against 640+ in the own-text runs), so each layer's
    attention is quicker. That leaves less GPU time for a copy to hide behind.
  - The hit rates with PREFETCH are similar to own text (0.734 / 0.886 / 0.955).

### Consequences

1. **PREFETCH is correct and does what the simulation says to the hit rate.** Its speed gain is real but smaller than
   FETCH's on this host, and it only pays together with FETCH at small budgets.
2. **Both options depend on the host.**
   - On a PCIe 4.0 host (job 065) both lost 16%.
   - A deployable version should pick them from measured bandwidths, as FreeToken's `ft bench bw` calibrates its
     hybrid mode.
3. **The chat-benchmark claim, on this host:** our cache with FETCH is 3–18% faster than FreeToken at equal GPU
   memory, and runs a budget FreeToken cannot fit.

## 28 September, late: same host, same client (job `064_samehost_rerun@vast`)

**Setup:**
- One machine: RTX 5090 + Ryzen 9 9950X, job 060's host.
  - Measured: CPU memory reads 46.6 GB/s with 16 threads (48.6 at best), the link 47.6 GB/s, both at once 52 GB/s in
    total; GPU memory 1,558 GB/s.
- **One client times every system.** It is FreeToken's own benchmark code (`benchmarks/bench_decode_moe.py`, commit
  0d652e7): its AIME-25 prompts, sampling, warm-up request and tok/s formula, over 5 problems × 256 tokens.
  - Every system runs as a server with the OpenAI chat API.
  - Our cache runs in `llama-server`.
- **Equal GPU memory for experts:** llama.cpp `-ncmoe 32 / 27 / 20`, cache C = 14 / 32 / 56 of 128 (and 51),
  FreeToken `--moe-cache-rate` 0.111 / 0.25 / 0.40.
- **Check on the client:** on the same problem (rate 0.25, problem 0), FreeToken's unmodified script gave 67.2 / 67.1
  tok/s for offload / hybrid, and our client 68.1 / 66.6. That is within 1.3%, as in job 062 (within 2%).

| Experts on the GPU | llama.cpp | **Our cache** | FreeToken offload | FreeToken hybrid |
|---|---|---|---|---|
| 11% | 24.3 tok/s | **42.9** | 38.5 | 39.4 |
| 25% | 28.2 | **68.1** | 68.6 | 66.2 |
| 40% (C = 51) | – | **105.2** | 108.9 | 99.2 |
| 44% (C = 56) | 36.4 | **114.8** | does not fit | does not fit |

- **Against FreeToken, we are even:** +9% at 11%, −1% at 25%, −3% at 40% against its better mode.
  - Every system varies by about ±10% across the five problems (sampled text routes differently), so only the 11%
    difference is clearly outside the noise.
  - FreeToken does not fit 44%: its KV cache runs out of room, even at `--memory-ratio 0.95`.
- **Against llama.cpp:** 1.8× / 2.4× / 3.2×.
- **Speed-of-light on this host** (own text, measured bandwidths): 174 / 313 / 313 tok/s.
  - Our cache reaches 24% / 20% / 30% of it, or 27% / 24% / 34% with FETCH; llama.cpp 14% / 9% / 12%.
- **The host reproduces.** The own-text runs repeated job 060's on the same machine within 0.8%:
  - cache 41.0 / 62.7 / 94.0 tok/s, with FETCH 46.9 / 74.3 / 108.0;
  - llama.cpp inside our harness 24.1 / 28.0 / 36.4.
- **Missing:** our cache with FETCH in the server. It aborted at start-up, because the cache also started during
  llama-server's memory-fit pass, when the weights are not loaded. Fixed (commit 33290cf); job 066 runs it.

## 28 September, late: predicting the next layer's experts, and PREFETCH

Details: `research_notes/MoE offload system race plan/prefetch_lookahead.md`.

### Lookahead (job `063_lookahead@vast`)

**The next layer's experts are predictable from the current layer.** On gpt-oss-120b, layer l+1's router applied to
the residual after layer l's attention finds 84% of layer l+1's selected experts in its top 4 and 97% in its top 8
(gpt-oss-20b: 87% and 98%). A host recomputation of every layer's own routing matched 100%, which checks the
extraction.

### Simulated benefit

- **The cache simulation reproduces the measured hit rates of job 060** (0.542 / 0.757 / 0.876 against 0.541 / 0.758 /
  0.875).
- **With a prefetch of one predicted expert per layer, CPU misses fall 45% / 53% / 58%** at C = 14 / 32 / 56.
- **A layer timeline with the measured host-1 constants gives +19% / +18% / +13% tok/s**, against +2–6% measured for
  FETCH on that host. The timeline reproduces the measured base within 2.7%.
- **This needs the copy on a second GPU stream.** On the main stream the model predicts about +0%.

### PREFETCH, built and checked (job `065_prefetch_check@vast`)

- **Implementation** (llama.cpp branch commit 6dd2d1f): `LLAMA_EC_PREFETCH=q`. The next layer's router is precomputed
  with the norm ratio folded in, a plan op updates the next layer's maps, the copy runs on a side stream, and the next
  layer joins it just before its GPU experts.
- **First GPU run:** gpt-oss-20b, C = 8 of 32, RTX 4500 Ada + Ryzen 9 7900X, PCIe 4.0.

| | Cache | + PREFETCH q=1 | + PREFETCH q=2 | + FETCH | + PREFETCH q=1 + FETCH |
|---|---|---|---|---|---|
| Hit rate (simulated) | 0.639 (0.639) | **0.816 (0.816)** | **0.884 (0.885)** | 0.682 | 0.826 |
| Prefetches per layer-step, used (simulated) | – | 0.722, 0.606 (0.721, 0.605) | 1.087, 0.872 (1.086, 0.871) | – | 0.712, 0.595 |
| tok/s | 61.5 | 51.4 (−16%) | 43.2 (−30%) | 51.9 (−16%) | 41.1 (−33%) |
| Same next token as the cache | – | 99.0% | 99.1% | 99.0% | 99.3% |

- **Correct:**
  - With `LLAMA_EC_CHECK=1`, the ids the graph used equal the host's maps on all 1,536 steps.
  - CUDA graphs on and off give identical output.
  - `compute-sanitizer` memcheck reports 0 errors.
  - NLL is within 0.3%.
  - Hit rates and prefetch counts match the simulation to 0.001.
- **Slower on this host, as FETCH is.** Its GPU link is PCIe 4.0 (about 25 GB/s), half its memory bandwidth, so moving
  an expert over the link costs more than computing it on the CPU. Both FETCH and PREFETCH are for hosts whose link is
  about as fast as their memory (PCIe 5.0 with a desktop CPU). **Job 066** tests PREFETCH on the RTX 5090 host.

## 28 September, night: FETCH A/B, FreeToken attempt, same-host comparison started

### FETCH A/B (job `060_fetch_ab_homepc@vast`, another RTX 5090 + Ryzen 9 9950X host)

Own text, 12 prompts × 128 tokens, every configuration in one process on one loaded model:

| Budget (C of 128) | Cache, no FETCH | FETCH, copy overlapped with the CPU | FETCH, copy first (job 059's form) | Same next token as no FETCH |
|---|---|---|---|---|
| 14 | 40.8 tok/s | 46.6 (**+14.2%**) | 46.0 (+12.8%) | 98.8% |
| 32 | 62.5 | 73.7 (**+18.0%**) | 73.2 (+17.1%) | 98.8% |
| 56 | 93.8 | 107.3 (**+14.3%**) | 106.8 (+13.8%) | 98.7% |

- **Repeats agree within 0.4%** (each configuration run again in reverse order).
- **The overlap itself adds only 0.5–1.4 points.** The serial form, which gave +2–6% on job 059's host, gives +13–17% here.
  So the benefit of FETCH depends on the machine far more than on the overlap.
- **This host is 19–22% slower without FETCH** than job 059's (40.8 / 62.5 / 93.8 against 52.6 / 79.0 / 116.5 tok/s), on
  the same hardware class. Its bandwidths were not measured. A likely reading is slower main memory, which makes a
  CPU miss costlier and an over-the-link copy relatively cheaper; job 062 measures this.
- **Tables:** `0,1,1,2,3` and `0,1,1,2,2` are equal within 0.3%; `0,0,1,1,2` (fetch less) is weaker at every budget.
- **Accuracy:** NLL within ±0.4% of no FETCH.

**Consequence for the method:** numbers from different rented hosts are not comparable, even for the same CPU and
GPU model. From now on every job measures its own host (STREAM, `bw.cu`, `concur.cu`), and systems are compared only
within one host.

### FreeToken, first attempt (job `061_freetoken_homepc@vast`, RTX 5090 + Ryzen 9 9950X3D)

- **Failed:** the model folder was empty. `hf download … --exclude "original/*" "metal/*"` read `metal/*` as a file
  name. Fixed in job 062 (Python `snapshot_download` with `ignore_patterns`, and a check for `config.json`).
- **FreeToken's own link measurement ran:** CPU MoE read 36.6 GB/s for MXFP4, PCIe 27.0 GB/s. Its hybrid mode would
  fetch 53% of misses. This host's link is slow (28.9 GB/s, against 57 GB/s on job 059's host), so it was not used again.

### Running now

1. **Job 062 (same host, same client).**
   - Stock llama.cpp, our cache (with and without FETCH) and FreeToken (offload and hybrid) run on one RTX 5090 +
     9950X host (link 45.6 GB/s by Vast's figure), at equal GPU memory for experts.
   - All are timed by one client built on FreeToken's own benchmark code: their AIME-25 prompts, sampling, warm-up and
     tok/s formula, 5 problems × 256 tokens.
   - Their unmodified script runs once as a check on the client.
   - Our cache runs in `llama-server` for this, which needed `-ot` to accept pinned host memory (`CUDA_Host`).
2. **Job 063 (lookahead).** How well can a layer's experts be predicted from the hidden state one layer earlier?
   gpt-oss-20b and 120b, own text. `ec-bench --lookahead` records the actual routing and the top-8 predictions (the
   next layer's router applied to the residual after this layer's attention, or to this layer's output). A host
   recomputation of each layer's own routing checks the arithmetic; it matched 100% on two tiny models.
   `scripts/sim_prefetch.py` turns the recording into misses under the deployed cache plus a prefetch.

## 28 September, evening: first measurements on a home PC

### What was done

1. **Our system was updated to current llama.cpp** (commit 4da6337, 27 Sep) with no conflicts.
2. **Three bugs were fixed and tested:**
   - experts running in BF16 could leave stale numbers behind;
   - a rare timing slip in the GPU-to-CPU mailbox;
   - the paper's case-study speed limit left out the model's final output layer. Fixed in the paper: our system
     now reads 51–72% of the limit on the A10 (was 47–64%), llama.cpp 36–50% (was 33–45%). Erratum recorded.
3. **A new feature, FETCH:** when some of a layer's experts are missing from the GPU, copy a few of them over at
   once and run them on the GPU, while the CPU runs the rest. A trace simulation suggested this for home PCs,
   where the GPU link is about as fast as main memory.
4. **A safe way to rent machines:** jobs run on Vast.ai without ever putting your GitHub key on the machine.
   Results come back through Vast's log feature.
5. **"How our system differs"** was written up for you: `reports/How our system differs.md`.

### Results

**Checkpoint 1: does it build and give the same answers?** (RTX 5080 + Ryzen 9 7900X, gpt-oss-20b, 12 prompts ×
128 tokens, job `058_ckpt1_dev@vast`) — **passed.**

| | tok/s | Same next-token pick as llama.cpp | Error on the text (NLL) vs llama.cpp |
|---|---|---|---|
| llama.cpp, 25% of experts on the GPU (`--n-cpu-moe 18`) | 18.5 | — | — |
| **Our system, same GPU memory** | **40.3 (2.18×)** | 99.5% | +0.03% |
| Everything on the GPU (reference) | 246.6 | 99.2% | −0.05% |

Op tests: 929/929 standard expert matmul cases and, in job 059, 28/28 of our cases (all weight types including BF16),
with 0 memory errors.

**Checkpoint 2: where do we stand on a home PC?** (RTX 5090 + Ryzen 9 9950X, 126 GB DDR5, gpt-oss-120b, the model's
own text, 12 prompts × 128 tokens, job `059_ckpt2_homepc@vast`)

| GPU memory for experts (llama.cpp setting) | llama.cpp, best of `llama-bench` and our harness | **Our system** | Our system + FETCH (copy, then hand over) | Speed limit on this machine |
|---|---|---|---|---|
| Small (`-ncmoe 32`, 14 of 128 experts per layer) | 31.2 tok/s (15% of limit) | **52.6 (1.68×, 25%)** | 53.5 (+1.8%) | 210 |
| Middle (`-ncmoe 27`, 32 per layer) | 36.3 (11%) | **79.0 (2.17×, 23%)** | 83.8 (+6.0%) | 341 |
| Large (`-ncmoe 20`, 56 per layer) | 46.7 (14%) | **116.5 (2.49×, 34%)** | 120.3 (+3.2%) | 341 |

- **Same answers:** 98.6–99.0% of next-token picks match llama.cpp; NLL within −0.75% to +0.14%.
- **The harness agrees with stock llama.cpp:** llama.cpp measured inside our harness is within 1–3% of stock
  `llama-bench` (30.4/36.2/46.4 vs 31.2/36.3/46.7).
- **The paper's decode model held up on this new machine.** On the audit's datasheet basis it predicted llama.cpp at
  25.6/29.3/36.8 tok/s; measured is 1.22–1.27× that, inside its error band, and the same ratio as on the A10.
- **The machine:**
  - The CPU reads memory at 62 GB/s; the GPU link moves 57 GB/s (copy engine) or 53 GB/s (kernel reading host memory).
  - Both at once total 73–77 GB/s, so neither path alone saturates the memory.
  - One gpt-oss-120b expert (13.25 MB) takes ~230–250 µs to copy, about the same as the CPU needs to run it
    (~230–240 µs per expert, measured from the helpers).
- **For comparison:** a community run of FreeToken on a 5090 reported 127 tok/s with 40% of experts on the GPU (host
  not stated). Our 116–120 tok/s at 44% is in the same range; a same-machine comparison is still to do.

### What this means

- **Our system is well ahead of properly set-up llama.cpp on a home PC:** 1.7–2.5× on gpt-oss-120b, with the same
  answers. Checkpoint 2's rule (≥ 1.25× at every budget) is met.
- **Every system is still far from the limit (15–35%).** The biggest gap is the cache's hit rate. The best possible
  policy would miss 0.95 / 0.39 / 0.14 experts per layer and token at the three budgets, against our ~1.8 / 1.0 / 0.5.
- **FETCH gave only +2–6% as first built,** because the copy ran before the CPU started, so the two didn't overlap.
  It is now changed to overlap them (job 060, running).

### Spend

About $1.40 of the $25 so far (ledger estimate in `gpu/vast_ledger.json`).

### Next

1. Job 060: the overlapped FETCH A/B (three tables, plus the serial form, and repeats in reverse order).
2. FreeToken on the same class of machine. It needs the Hugging Face weights (another ~65 GB download) and a CUDA 13
   image.
3. Qwen3-30B-A3B in BF16 as the second model (61 GB).
4. Hit rate: the gap to the best possible policy is the largest remaining one.
