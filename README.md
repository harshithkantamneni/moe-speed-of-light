# Where Do the Experts Go?

A validated speed-of-light model for Mixture-of-Experts (MoE) decode on memory-constrained consumer hardware.

This repo has everything behind the paper in `paper/`: exact routing traces collected without a GPU, a bytes-over-bandwidth decode model validated against published measurements, a policy-independent throughput bound, and a system in llama.cpp (`runtime/llama.cpp-expert-cache.patch`) evaluated under pre-registration on an NVIDIA A10.

## Headline (pre-registered test, 12 fresh sequences × 192 decode steps, A10 24 GB)

Decode speed-up of the expert cache with GPU-signalled CPU helpers ("mailbox") over llama.cpp `--n-cpu-moe` at equal GPU expert memory:

| model | 12.5 % of experts on GPU | 25 % | 50 % |
|---|---|---|---|
| gpt-oss-20b (MXFP4) | 1.52× | 1.63× | 1.60× |
| Qwen3-30B-A3B (Q4_K_M) | 1.46× | 1.52× | 1.41× |
| Qwen3-30B-A3B (Q8_0) | 1.41× | 1.50× | 1.45× |
| gpt-oss-120b (MXFP4) | 1.57× | 1.81× (72.1 vs 39.8 tok/s) | – |

Teacher-forced NLL within 0.6 % of llama.cpp; a step-time model fitted on calibration sequences predicted all 33 test configurations with 2.9 % median error (predictions committed before the run: `prereg/PROTOCOL.md`, phase 3; outcomes: `prereg/a10_mb/scored.md`). Raw runs: `results/firstparty/a10/019_ec_calib`, `020a`–`020c`, `021`. Job scripts: branch `gpu`, `jobs/0*.sh`.

Apply the system: `git checkout 2145525a` of ggml-org/llama.cpp, `git apply runtime/llama.cpp-expert-cache.patch`, build with CUDA, then e.g.
`LLAMA_EC_SLOTS=8 LLAMA_EC_KAPPA=1 LLAMA_EC_MAILBOX=1 LLAMA_EC_HELPERS=28` with experts in pinned host memory (see `tools/ec-bench/ec-bench.cpp --host-experts` and the comment block in `src/llama-expert-cache.h`).

## What is here

| Path | What it does |
|---|---|
| `mosl/collect.py` | Layer-streaming, teacher-forced trace collector. It records the exact experts each decode token is routed to, for models far larger than RAM or disk: one decoder layer at a time, with tensors streamed from the Hub over HTTP range requests (`--remote`). |
| `mosl/cachesim.py` | Trace-driven expert-cache simulator: LRU, LFU, static pinning, Belady MIN, and MIN-with-bypass. Numba-compiled. |
| `mosl/archs.py` | Exact per-category parameter counts for any `transformers` MoE, from a meta-device instantiation. |
| `mosl/perfmodel.py` | Decode model (Eq. 1 in the paper), dynamic-cache timing, and the speed-of-light bound (Belady + Jensen). |
| `mosl/validation_set.py` | Annotated subset of the published measurements, with datasheet bandwidths and bits per weight. |
| `data/published_measurements.json` | 140 published MoE decode measurements. Each row cites a source URL and quotes the table or line. |
| `data/traces/<model>/` | Routing traces: `layerNNN.npy` = int16 `[tokens, top_k]`, plus `meta.json`, `seq_lens.npy` and `domains.json`. |
| `scripts/validate.py` | Fits the model, runs leave-one-source-out CV and the term ablation. Writes `results/validation.{json,csv}`. |
| `scripts/analyze.py` | Locality statistics, cache sweeps, and tok/s per platform and strategy against the bound. |
| `scripts/figures.py`, `scripts/paper_numbers.py` | Regenerate every figure and every number quoted in the paper. |
| `tests/` | Bit-exactness of the collector against reference `transformers` prefill and KV-cache decode on 4 architectures (5 checkpoints), optimality of MIN/MIN-bypass against exhaustive search, and the speed-of-light bound against every simulated policy, including an oracle prefetcher. |

Traced models: OLMoE-1B-7B, Qwen3-30B-A3B, gpt-oss-20b. A gpt-oss-120b trace was also collected, but it is excluded from all results because of an unexplained teacher-forced perplexity anomaly (see the paper's Limitations).

## Reproduce

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers safetensors huggingface_hub numba scipy pandas pyarrow matplotlib

python tests/test_cachesim.py                 # MIN / MIN-bypass optimal vs brute force
python tests/test_collect_equivalence.py      # collector == reference prefill and KV-cache decode routing
python tests/test_bound.py                    # no simulated policy beats the speed-of-light bound

# traces (CPU only; about 30 min for Qwen3-30B-A3B on 2 vCPUs)
python -m mosl.corpus --n 40
python -m mosl.tokenize_corpus --repo Qwen/Qwen3-30B-A3B-Instruct-2507 --out data/tok_qwen3_30b.jsonl
python -m mosl.collect --remote --repo Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --corpus data/tok_qwen3_30b.jsonl --out data/traces/qwen3-30b-a3b

python scripts/validate.py && python scripts/analyze.py && python scripts/figures.py && python scripts/paper_numbers.py
cd paper && latexmk -pdf paper.tex
```

## Why teacher-forced traces are exact

In a causal decoder, a token's hidden state at every layer depends only on the tokens before it. So the experts that decode selects for token *t* are the ones prefill selects at position *t*. Streaming layers makes each weight byte be read once per corpus instead of once per token. The test suite confirms selections match the reference implementation exactly, including MXFP4-packed and sliding-window models.

## Validating on your own GPU (optional)

See `scripts/gpu_validation.md` for a 30-minute `llama-bench` protocol. It adds your machine as a new, held-out data point.

## License

Code: MIT. The traces are derived from public models and datasets and inherit their licenses.
