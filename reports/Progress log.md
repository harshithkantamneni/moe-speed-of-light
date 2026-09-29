# Progress log: option C measurement study

*Started Monday 28 September 2026. Newest entries first. Numbers link to result folders on the `gpu` branch
(`results/<job>/`).*

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
