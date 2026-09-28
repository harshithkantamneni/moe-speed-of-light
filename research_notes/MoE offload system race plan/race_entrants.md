# Race entrants: which published MoE CPU/GPU offloading systems can be built and raced on one Linux cloud machine (as of 28 Sep 2026)

Method note. This extends `MoE audit measurement plan/system_reproducibility.md`, `MoE hybrid decode novelty check/academic_systems.md` and `.../industry_engines.md`. Facts already established there are re-cited to their original sources, not re-derived. New work, on 2026-09-28:
- I cloned FreeToken, pipeshard-mlsys26-ae, ktransformers, HybriMoE, elsa-lab/MoE-CPU-GPU-Collaborative-Inference, fiddler, MoE-Infinity, everett6/moe-prefetch and ongunm/llama-moe-cache with `git clone --filter=blob:limit=2m`, then read docs, build files, source comments and `git log`. Commit and tag dates come from those logs.
- I fetched the arXiv HTML of FreeToken, Pipelined sharding, DALI and 2606.10493.
- I read several GitHub issue/PR pages through WebFetch. A small model summarises those pages, so treat issue and PR details as "as summarised".

The target machine is either:
- a Verda RTX PRO 6000 (96 GB, sm_120) VM with 30 threads and 90 GB RAM, or
- a Vast.ai RTX 5090 (32 GB, sm_120) on a DDR5 server.

"Equal VRAM" means equal total GPU memory of the process, matched to a llama.cpp `--n-cpu-moe` configuration.

## Q1. Is each candidate's code public (URL, license, last commit, activity, released version)?

### Takeaway
Seven candidates have public, buildable code:
- FreeToken: very active, tagged v0.1.3.
- KTransformers kt-kernel + sglang-kt: very active, v0.7.1.
- Pipelined sharding: frozen artifact, tag v2.0.3-mlsys26 plus Zenodo.
- MoE-Infinity: very active, but the code differs from its paper.
- HybriMoE: frozen.
- 2512.16473: frozen.
- Fiddler: frozen since 2024.

Four have no public code: DALI, SeqMoE, the OSDI'26 Tsinghua hybrid system (arXiv 2606.10493) and SP-MoE/MoE-SpeQ.

The only June–Sep 2026 additions with public code for batch-1 hybrid decode are llama.cpp-family community patches: leloch's cache, PR #27861, and a Sept 2026 fix-up repo on top of #27861. None of them is a paper.

### Cited Findings
**FreeToken (arXiv 2608.16157)**
- github.com/FlashML-org/FreeToken, Apache-2.0, ~13.9k stars. The first public commit, "feat: initial open-source release", is dated 2026-08-11 (3af9d90) — [ungh.cc](https://ungh.cc/repos/FlashML-org/FreeToken); [initial commit](https://github.com/FlashML-org/FreeToken/commit/3af9d90ee5e7af9bbff1e16f4f1c6201f41fff25)
- Tags are v0.1.2 (2026-08-19), v0.1.3 (release commit 2026-09-15) and a rolling `nightly`. The head on 2026-09-28 is 0d652e7 (2026-09-25, "feat(rocm): add RDNA3 and RDNA4 runtime foundation"). There were 46 commits in August 2026 and 42 in September — [commits](https://github.com/FlashML-org/FreeToken/commits/main); [releases](https://github.com/FlashML-org/FreeToken/releases)
- These runtime fixes landed after v0.1.3 — [commits](https://github.com/FlashML-org/FreeToken/commits/main):
  - "fix(runtime): correct overlap token limits and cache safety (#546)" (2026-09-23)
  - "fix(install): pin flashinfer packages to one version (#545)" (2026-09-23)
  - "fix(kernel): pin tvm-ffi jit arch to the bound gpu (#521)" (2026-09-18)
  - "fix(engine): preserve greedy sampling in mixed batches (#471)" (2026-09-18)
- The paper says "We release the system at flashml.ai" and carries a "[Code]" link to the GitHub repo. It names no version or commit — [arXiv HTML](https://arxiv.org/html/2608.16157)

**KTransformers (SOSP'25), current kt-kernel + SGLang path**
- kvcache-ai/ktransformers, Apache-2.0, ~19.5k stars — [ungh.cc](https://ungh.cc/repos/kvcache-ai/ktransformers)
- Latest tag v0.7.1 (2026-09-14). Head c40722b (2026-09-23): "fix(kt-kernel): cap compressed-tensors below the torch>=2.10 releases (#2116)" — [commits](https://github.com/kvcache-ai/ktransformers/commits/main); [tags](https://github.com/kvcache-ai/ktransformers/tags)
- The paper-era framework now sits in `archive/`. Inference today is kt-kernel plus the kvcache-ai SGLang fork `sglang-kt` — [README](https://github.com/kvcache-ai/ktransformers/blob/main/README.md)
- kt-kernel's `pyproject.toml` pins `torch==2.9.1` and depends on `sglang-kt`; the comment reads "Pinned, not a range: ktransformers, sglang-kt, accelerate-kt and transformers-kt" — [kt-kernel/pyproject.toml](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/pyproject.toml)
- SOSP'25 AE awarded Available, Functional and Reproduced, with no pinned commit — [sysartifacts SOSP'25](https://sysartifacts.github.io/sosp2025/results)

**Pipelined sharding (NVIDIA, MLSys'26 oral, arXiv 2604.26334)**
- github.com/deepshnv/pipeshard-mlsys26-ae, MIT, archived on Zenodo record 19436383. Built "on llama.cpp tag b6097" — [README](https://github.com/deepshnv/pipeshard-mlsys26-ae/blob/main/README.md); [Zenodo](https://zenodo.org/records/19436383)
- Tags v1.0.0-mlsys26 through v2.0.3-mlsys26 (2026-04-06). The last commit is 2026-06-01 ("Update README.md"). The last code fix, 2026-05-05, is "fix: clamp device_count to devices.size() to avoid OOB access in CPU-only builds" — [commits](https://github.com/deepshnv/pipeshard-mlsys26-ae/commits/main)

**MoE-Infinity**
- EfficientMoE/MoE-Infinity, Apache-2.0. The head on 2026-09-28 is "docs: remove duplicate entries in README and serving flags table (#240)". Recent work (Sept 2026) covers MiniMax-M3, Qwen3.5 VL and a "cap sglang-kernel<0.4.7 to prevent torch 2.13 (cu13) clobber" fix — [commits](https://github.com/EfficientMoE/MoE-Infinity/commits/main)
- The README says the open-source version "differs from the version reported in the paper" — [README](https://github.com/EfficientMoE/MoE-Infinity/blob/main/README.md)

**Frozen academic repos**
- HybriMoE: PKU-SEC-Lab/HybriMoE, Apache-2.0. Last commit 2025-12-16 (508a85b, "Change optimize rule path to DeepSeek-V2-Chat-gpu.yaml"); no tags — [commits](https://github.com/PKU-SEC-Lab/HybriMoE/commits/main)
- 2512.16473 (ASP-DAC'26): elsa-lab/MoE-CPU-GPU-Collaborative-Inference, MIT. Last commit 2025-10-16 ("Upload paper") — [repo](https://github.com/elsa-lab/MoE-CPU-GPU-Collaborative-Inference)
- Fiddler: efeslab/fiddler, Apache-2.0. Last commit 2024-04-28; the README calls it "a proof-of-concept and still under heavy construction" — [repo](https://github.com/efeslab/fiddler)

**No public code**
- DALI (arXiv 2602.03495): the paper "does not provide a code repository link". It states "We implement our proposed DALI framework based on the open-source KTransformers framework" (the summarised fetch notes "over 1,000 lines of C++ and 2,000 lines of Python code"). A web search found only the arXiv page and aggregators. Every GitHub "DALI" hit was NVIDIA's unrelated data-loading library — [arXiv HTML](https://arxiv.org/html/2602.03495); [NVIDIA/DALI](https://github.com/NVIDIA/DALI)
- SeqMoE (arXiv 2609.12978, 11 Sep 2026): no repository link; "implement[ed] … in Hugging Face Transformers … using hooks". A new search on 2026-09-28 found no code, only the arXiv page and aggregators — [arXiv HTML](https://arxiv.org/html/2609.12978); [Pith](https://pith.science/paper/2609.12978)
- "Achieving Cloud-Grade SLOs for Local MoE Inference through CPU–GPU Hybrid Design" (arXiv 2606.10493, 9 Jun 2026, Tsinghua; OSDI'26 per the fetch): "No GitHub URL or public code statement" — [arXiv HTML](https://arxiv.org/html/2606.10493)

**June–Sep 2026 community code (llama.cpp family)**
- leloch `moe-cache-v2-pr` branch, rebased 2026-08-06. PR #27861 (csantiago78, draft, opened 2026-08-28). PR #26563 is closed; PR #26824 is open. None was merged as of 2026-09-27 — [Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528); [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861); [PR #26824](https://github.com/ggml-org/llama.cpp/pull/26824)
- everett6/moe-prefetch (commits 2026-09-18 and 09-19; no LICENSE file in the clone) ships `patches/0006`, "the full current diff against `bccbacd`", which fixes PR #27861's cache. The five upstream bugs it found include "`--moe-expert-cache` silently no-ops because a dry-run context latches its one-shot init guard" — [README](https://github.com/everett6/moe-prefetch); [UPSTREAM-BUGS.md](https://github.com/everett6/moe-prefetch/blob/main/docs/UPSTREAM-BUGS.md)
- llama.cpp PR #28414, `--prefetch-experts-slots` (leshchukandrej): a lookahead H2D prefetch for *prefill* ("Decode unchanged"). The summary also reports that "Multi-GPU deployments produce silent wrong output" — [PR #28414](https://github.com/ggml-org/llama.cpp/pull/28414)
- ongunm/llama-moe-cache ("FATE", 6–7 Apr 2026, AGPL-3.0 plus a commercial license) is a fetch-and-prefetch cache inside llama.cpp. It claims 64.45 vs 33.74 t/s and a 99.50% hit rate for Qwen3-30B-A3B Q4_K_M on an RTX 4070 Ti — [repo](https://github.com/ongunm/llama-moe-cache)

### Inferences
- **FreeToken:** pin a commit rather than a tag. v0.1.3 predates the 2026-09-23 "cache safety" and overlap fixes, and `nightly` moves. Run v0.1.3 as "released", or a dated main commit (0d652e7) as "current", and state which.
- **KTransformers:** race the maintained kt-kernel + sglang-kt v0.7.1 path and call it "KTransformers (kt-kernel v0.7.1)", not "the SOSP'25 artifact".
- **Pipelined sharding:** the frozen tag v2.0.3-mlsys26 (or the Zenodo snapshot) is the right, citable version.
- **Excluded:** DALI, SeqMoE and 2606.10493 cannot enter the race. Only an email to the authors could change that before 27 Oct.
- **Out of scope:** PR #28414 (prefill-only) and FATE (April 2026, fetch-only, extraordinary claims) are not batch-1 hybrid-decode entrants.

### Gaps
- Whether the SeqMoE, DALI or 2606.10493 authors plan a code release. No statement was found.
- Exact SOSP'25 AE commit for KTransformers (not stated).
- GitHub issue search is robots-blocked, so issue coverage is sampled, not exhaustive.

## Q2. Build and runtime requirements, and whether each runs on Blackwell sm_120 (RTX 5090 / RTX PRO 6000) with a 30–60-core, 90–180 GB host (including KTransformers on AMD EPYC without AMX)

### Takeaway
Four systems build and run on sm_120 with documented paths: FreeToken, KTransformers (source build), Pipelined sharding and MoE-Infinity.
- **FreeToken** needs driver r580+ and a CUDA 13 toolkit.
- **KTransformers** needs a source build of kt-kernel with `CPUINFER_CUDA_ARCHS=120`, because the prebuilt wheel covers SM 80/86/89/90 only. It pins torch 2.9.1.
- **Pipelined sharding** is a plain llama.cpp CMake build with CUDA ≥12.8.
- **MoE-Infinity** builds with `MOE_ENABLE_SM120=1`.

HybriMoE and Fiddler pin pre-Blackwell PyTorch/CUDA stacks, so running them on sm_120 means porting. 2512.16473 is pure PyTorch and should run on any Blackwell-capable torch.

KTransformers does not need AMX. Its BF16/FP8 native backends need only AVX-512+BF16, which Zen 4/5 EPYC has, and AVX2-only CPUs fall back automatically. The only AMD datapoints on a 5090 are from dual-socket Zen 5 EPYC 9355 hosts.

### Cited Findings
**FreeToken**
- Requirements: "Linux x86_64, NVIDIA GPU, driver r580+ (CUDA 13)", Python ≥ 3.10. `uv pip install "freetoken[accel]"`. "CUDA kernels are JIT-compiled on first use, need a CUDA 13 toolkit with `nvcc` on PATH". Nightly wheels ship a prebuilt `freetoken-kernel-cache` for CPython 3.12 — [docs/install.md](https://github.com/FlashML-org/FreeToken/blob/main/docs/install.md)
- The README claims "native support for NVIDIA RTX 30, RTX 40, and RTX 50 series GPUs" — [README](https://github.com/FlashML-org/FreeToken/blob/main/README.md)
- Hardware used so far:
  - The paper's RTX PRO 6000 machine: Xeon Platinum 8559C (48 threads), 512 GiB DDR5, PCIe 5.0 x16, B_P = 51.5 GB/s, B_H = 178 GB/s — [arXiv HTML Table 1](https://arxiv.org/html/2608.16157)
  - Community 5090 run: PyTorch 2.11.0+cu130 under WSL2. Auto-sizing failed with "CUDA driver error: unknown error", worked around with `--moe-cache-size 500 --num-tokens 65536` — [zenn.dev/lifona, 25 Aug 2026](https://zenn.dev/lifona/articles/a8606bb95e17e1?locale=en)
- CPU expert executor:
  - Weight formats: "bf16, NVFP4, MXFP4, ds_fp4 and Q4_0 expert banks"
  - "ISA is chosen once at construction (AVX-512-BF16 dpbf16 -> AVX-512F widening -> AVX2+FMA -> scalar)"
  - Source: [cpu_moe_ext.cpp](https://github.com/FlashML-org/FreeToken/blob/main/python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp)
- `--moe-cpu-threads` defaults to physical cores — [docs/cli.md](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md)
- **CPU hand-off mechanism (matters for novelty).** From the initial public release (2026-08-11), the cpu/hybrid executor uses a GPU-signalled flag handshake:
  - "the GPU raises a mapped-pinned 'ready' flag at submit; a persistent CPU coordinator … polls it, runs the layer, and sets a 'done' flag the GPU waits on at sync — no host-func round-trip".
  - The GPU side uses "STREAM MEMORY OPERATIONS (cuStreamWriteValue64 / cuStreamWaitValue64 …) … with no SM-resident kernel".
  - "The first cut used a spin-wait kernel; that pinned reported utilization at 99% and laptop CPU/GPU dynamic power schedulers responded by clamping the CPU frequency".
  - The default host-func path "pays ~30-50us of callback dispatch latency per call".
  - Where memops are unavailable, "Windows WDDM, vGPU, old drivers — functionally probed at startup", it falls back to `cudaLaunchHostFunc`.
  - Sources: [cpu_executor.py](https://github.com/FlashML-org/FreeToken/blob/main/python/freetoken/moe/cpu_executor.py); [cpu_moe_ext.cpp](https://github.com/FlashML-org/FreeToken/blob/main/python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp); [initial commit 3af9d90](https://github.com/FlashML-org/FreeToken/commit/3af9d90ee5e7af9bbff1e16f4f1c6201f41fff25)

**KTransformers (kt-kernel + sglang-kt)**
- The prebuilt `pip install kt-kernel` wheels cover Python 3.10–3.12 with "CUDA support included: GPU acceleration for NVIDIA GPUs (SM 80, 86, 89, 90)". The minimum is a "CPU with AVX2 support (Intel Haswell 2013+, AMD Zen+)" — [kt-kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)
- The source build defaults to `CMAKE_CUDA_ARCHITECTURES "80;86;89;90"`; `setup.py` reads `CPUINFER_CUDA_ARCHS` (default "80;86;89;90") — [kt-kernel/CMakeLists.txt](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/CMakeLists.txt); [kt-kernel/setup.py](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/setup.py)
- Blackwell appears in the docs:
  - The DeepSeek-V4-Flash tutorial targets "1× NVIDIA RTX 5090 (32GB VRAM, SM_120)", uses `export FLASHINFER_CUDA_ARCH_LIST=12.0a`, lists "Consumer Blackwell (RTX 5090) | SM_120 | triton_kernels | Triton fallback | ✓", and reports "Decode throughput: 20+ tok/s on a single RTX 5090" — [DeepSeek-V4-Flash.md](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash.md)
  - The GLM-5.3-Flash tutorial lists "NVIDIA SM89 and SM120 GPUs" — [GLM-5.3-Flash-Tutorial.md](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/GLM-5.3-Flash-Tutorial.md)
- Known SM_120 gap: issue #2081, "fp8.py:788 assert is_sm90 or is_sm100 blocks SM_120" (MiniMax-M3-MXFP8) — [issue #2081](https://github.com/kvcache-ai/ktransformers/issues/2081)
- CPU backends — [kt-kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md):
  - LLAMAFILE (GGUF): AVX2
  - RAWINT4: AVX512F+BW
  - AMXINT4/INT8: "requires AMX hardware", Sapphire Rapids+
  - FP8: AVX512F+BW+BF16+VBMI, "AMD Zen 4+ (e.g., EPYC 9355)"
  - BF16: AVX512F+BW+BF16, "AMD Zen 4+ (e.g., EPYC 9355)"
  - "AMD CPUs with BLIS: Supported (for int8 prefill & decode)"
- The AVX2-only backend supports BF16, FP8, GPTQ_INT4 and RAWINT4: "KT-Kernel will automatically detect the CPU and fall back to the AVX2 backend when AVX512/AMX is unavailable". It needs "Memory: At least the size of the model weights (e.g., Qwen3-30B-A3B BF16 requires 64GB+)" — [AVX2-Tutorial.md](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/AVX2-Tutorial.md)
- AMD without AMX, measured, all dual-socket:
  - Zen 5, Qwen3-Coder-Next FP8: "1 x RTX 5090 (32 GB) | 2 x AMD EPYC 9355 | PCIe 5.0 | FP8 | 362 / 75.9 | 1746 / 75.6 | 2407 / 69.1 | 6233 / 51.7" (prefill/decode tok/s at 64/2048/8192/32768 tokens, single concurrency), with `--kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 100 --kt-method FP8` — [Qwen3-Coder-Next-Tutorial.md](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/Qwen3-Coder-Next-Tutorial.md)
  - Zen 5, MiniMax-M2.1 on 1× 5090 + 2× EPYC 9355: decode 32.1/31.4/27.6 tok/s. The doc adds "We made our best effort to optimize llama.cpp performance" — [MiniMax-M2.1-Tutorial.md](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/MiniMax-M2.1-Tutorial.md)
  - Zen 2 (AVX2): PR #2175 (AVX2 MXFP4 fast path) reports DeepSeek-V4-Flash "512-token decode: 18.4 -> 29.5 tok/s at 65K context" on "2x EPYC 7452 (AVX2, 2x30 threads)" — [PR #2175](https://github.com/kvcache-ai/ktransformers/pull/2175)
- NUMA:
  - #1754 (Ryzen 9950X3D + RTX 5090, a single-NUMA desktop) segfaulted in `numa_bitmask_setbit`. The issue is closed as user-resolved with a local patch to `worker_pool.h`; no merged PR is documented on the page — [issue #1754](https://github.com/kvcache-ai/ktransformers/issues/1754)
  - Later commits add "numa_nodes parameter for explicit NUMA node mapping (#1891)" (2026-03-31) and "[fix](cli): handle edge cases with empty NUMA nodes (#1929)" (2026-04-13) — [commits](https://github.com/kvcache-ai/ktransformers/commits/main)
- The SOSP'25 decode gains include NUMA-aware tensor parallelism (up to 1.63× on dual-socket) and an AVX-512 kernel (2.22× over the Fiddler base) — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)

**Pipelined sharding**
- The README requires CUDA Toolkit 12.8+ and "ideally, an RTX 5090 or 5070 TI". Build with `cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF`. "All development and testing for our paper was done on Windows". On Linux a missing `libllama.so` needs `LD_LIBRARY_PATH`, and `bc` is required by the scripts — [README](https://github.com/deepshnv/pipeshard-mlsys26-ae/blob/main/README.md)
- The fork's ggml-cuda CMake sets `CMAKE_CUDA_ARCHITECTURES "native"` when undefined — [ggml/src/ggml-cuda/CMakeLists.txt](https://github.com/deepshnv/pipeshard-mlsys26-ae/blob/main/ggml/src/ggml-cuda/CMakeLists.txt)
- Linux history: "changes for successful linux compilation" (2026-03-07). `PIPESHARD_THREADS` controls CPU threads; the `run_all_repro` scripts default it to 16 — [commits](https://github.com/deepshnv/pipeshard-mlsys26-ae/commits/main); [README](https://github.com/deepshnv/pipeshard-mlsys26-ae/blob/main/README.md)
- Paper platforms were all Windows 11. "cli3" is an RTX 5090 with a 16-core EPYC, 256 GB and PCIe Gen5 — [arXiv HTML](https://arxiv.org/html/2604.26334)

**MoE-Infinity**
- "The from-source build targets compute capabilities `sm_80`/`sm_90` by default; for Blackwell (`sm_120`, e.g. RTX PRO 6000 / RTX 50-series) build with `MOE_ENABLE_SM120=1`". The command is `MOE_ENABLE_SM120=1 MOE_ENABLE_SM90=0 CUTLASS_DIR=~/cutlass pip install --no-build-isolation -e .`. Requirements: Python 3.10+, torch from the cu128 index, and a CUDA toolkit matching torch's major version — [README](https://github.com/EfficientMoE/MoE-Infinity/blob/main/README.md)

**HybriMoE, 2512.16473, Fiddler**
- HybriMoE: "CUDA 12.1 and above", Python 3.11, Dockerfile base `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel`, submodules pybind11 and llama.cpp. Open issues cover flashinfer (#7) and the Marlin path (#8) — [README](https://github.com/PKU-SEC-Lab/HybriMoE/blob/main/README.md); [Dockerfile](https://github.com/PKU-SEC-Lab/HybriMoE/blob/main/Dockerfile); [issues](https://github.com/PKU-SEC-Lab/HybriMoE/issues)
- 2512.16473: "Python 3.8+, PyTorch 2.0+, CUDA 11.8+", "128GB+ main memory for optimal performance". The code is pure Python (`src/main.py`, `src/model.py`) with PyTorch BF16 CPU matmuls — [README](https://github.com/elsa-lab/MoE-CPU-GPU-Collaborative-Inference/blob/main/README.md); [src/model.py](https://github.com/elsa-lab/MoE-CPU-GPU-Collaborative-Inference/blob/main/src/model.py)
- Fiddler pins `torch==2.1.2` and "is slow if your CPU does not support AVX512" — [requirements.txt](https://github.com/efeslab/fiddler/blob/main/requirements.txt); [README](https://github.com/efeslab/fiddler/blob/main/README.md)
- PyTorch's Blackwell support arrived with PyTorch 2.7. The source is a news headline, "PyTorch 2.7 brings NVIDIA Blackwell support", about the PyTorch 2.7 release; I did not open the release post itself — [PyTorch 2.7 blog](https://pytorch.org/blog/pytorch-2-7/); [AlternativeTo news](https://alternativeto.net/news/2025/4/pytorch-2-7-brings-nvidia-blackwell-support-mega-cache-flexattention-updates-and-more)

**Target rental**
- Verda lists the RTX PRO 6000 in 1x/2x/4x/8x. The 1x instance has "96 GB GPU VRAM", "90 GB … RAM" and "30 CPU threads", at "$1.96/h" on-demand and "$0.9820/h" spot. The page does not state the CPU model, VM vs bare metal, or driver version — [Verda RTX PRO 6000](https://verda.com/rtx-pro-6000)

### Inferences
- **One driver ≥ r580 serves all four Blackwell-ready systems**, because CUDA user-space runtimes are backward-compatible with newer drivers. Give each system its own venv: FreeToken is on torch cu130; KTransformers pins torch 2.9.1; MoE-Infinity is on cu128 and caps sglang-kernel. On Verda (a VM) the driver can be installed. On Vast, filter offers by max CUDA ≥ 13.0.
- **KTransformers without AMX.** On a Zen 4/5 EPYC host (AVX-512+BF16), the BF16 and FP8 native backends apply, and those are what KTransformers' own 5090 + EPYC 9355 tables use. AMX only matters for the AMXINT4/INT8 paths, which quantize experts (lossy relative to BF16).
  - On Zen 2/3 (AVX2 only) it falls back to the AVX2 or LLAMAFILE kernels, i.e. roughly ggml-class. That race would test scheduling, not the SOSP kernels.
  - Every AMD datapoint found is dual-socket (2× EPYC 9355 or 2× EPYC 7452, 96 threads). None exists for a 30-thread single-NUMA VM, so "how fast on AMD without AMX" for Qwen3-30B-A3B at batch 1 is unmeasured. Expect it to be bound by host bandwidth like every other CPU-expert system.
- **Verda's 90 GB of RAM** fits gpt-oss-120b (FreeToken used about 66 GB of host RAM for it; see Q5) and Qwen3-30B-A3B BF16 (about 61 GB, my estimate from 30.5B parameters × 2 bytes). Qwen3.6-35B-A3B BF16 (about 70 GB, estimated) is tight once FreeToken's pinned banks, SGLang and the OS are added. Prefer a ≥128 GB host for it.
- **HybriMoE on sm_120** needs its KTransformers-v0.2 fork, Marlin and flashinfer ported to torch ≥2.7/CUDA ≥12.8. I found no report of that working.
- **Fiddler** would need torch ≥2.7 plus Mixtral BF16 (~93 GB), which does not fit a 90 GB host. Treat both as not raceable on this machine.

### Gaps
- The Verda RTX PRO 6000 host CPU (vendor, generation, AVX-512/AMX) and default driver are not published. Run `lscpu | grep -o 'avx512[a-z_0-9]*\|amx[a-z_]*'`, `numactl -H` and `nvidia-smi` in the first minutes.
- Vast.ai RTX 5090 host CPUs vary by listing, and no primary source summarises them.
- Whether the prebuilt kt-kernel wheel works on sm_120 at all (its CUDA code is host-callback glue) or a source build is always required. The docs only show source builds with 12.0a.
- Whether the #1754 non-NUMA segfault is fixed in v0.7.1. The later "empty NUMA nodes" CLI fix is suggestive but unverified.
- Whether FreeToken's stream-memop handshake works inside a Verda VM, or whether the startup probe drops it to the slower host-func path (it names "vGPU" as unsupported). A full-GPU passthrough VM is probably fine, but this is unverified.

## Q3. Which models and weight formats does each support, and is its decode exact?

### Takeaway
The common ground is two models:
- **gpt-oss-120b with native MXFP4 experts:** llama.cpp and the user's patch, FreeToken, Pipelined sharding and MoE-Infinity. KTransformers is excluded because it has no gpt-oss support.
- **Qwen3-30B-A3B:** everything that builds on sm_120 supports it.

FreeToken reads only HF safetensors, and its CPU executor accepts BF16/NVFP4/MXFP4 experts, not FP8. A like-for-like Qwen3 race is therefore BF16 in every engine, or MXFP4 for gpt-oss.

Qwen3.6-35B-A3B (the audited FreeToken row) is supported by FreeToken, KTransformers (FP8/BF16), MoE-Infinity (Qwen3.5 class) and current llama.cpp. It is not supported by Pipelined sharding, whose b6097 base predates it.

Decode is exact (no algorithmic approximation) in the following modes, though none is bit-identical to llama.cpp:
- FreeToken
- Pipelined sharding
- MoE-Infinity
- KTransformers with deferral 0
- 2512.16473
- DALI (per its paper)

KTransformers' Expert Deferral and AMX INT4/INT8 conversions are lossy.

### Cited Findings
**FreeToken**
- "FreeToken loads HF safetensors checkpoints directly". Known-good checkpoints include Qwen3.6-35B-A3B (BF16, -FP8, NVFP4), Qwen3.5-35B-A3B, Qwen/Qwen3-30B-A3B, openai/gpt-oss-120b and gpt-oss-20b, DeepSeek-V4-Flash-0731 and GLM-5.2-NVFP4 — [docs/models.md](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md)
- `--moe-cpu-layers` "needs an expert format the CPU executor serves (bf16, nvfp4, mxfp4), so fp8 experts cannot use it" — [docs/cli.md](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md)
- The executor's `_WFMT_IDS` are `{"bf16", "nvfp4", "mxfp4_triton", "ds_fp4", "q4_0"}`. The gpt-oss activation is `"gpt_oss_swiglu"` (clamped), which is "fused inside the mxfp4 kernel" — [cpu_executor.py](https://github.com/FlashML-org/FreeToken/blob/main/python/freetoken/moe/cpu_executor.py)
- Q4_0 support is "native GGUF Q4_0 banks (borrowed ggml MoE kernels)". GGUF loaders exist only under `models/gemma4/` (gguf.py) — [fused_q4_0.py](https://github.com/FlashML-org/FreeToken/blob/main/python/freetoken/moe/fused_q4_0.py); [models/gemma4](https://github.com/FlashML-org/FreeToken/tree/main/python/freetoken/models/gemma4)
- Exactness: "the CPU and GPU compute their respective partial sums and merge them, preserving the exact MoE output without algorithmic approximation". The paper served "Qwen3.6-35B-A3B … in BF16, which gives exact precision parity across engines". It did not evaluate gpt-oss — [arXiv HTML](https://arxiv.org/html/2608.16157)
- Strategies: `fused` (all on GPU), `offload` ("an LRU cache of expert slots on GPU; misses stream over PCIe"), `cpu` ("misses are computed on the CPU instead of fetched") and `hybrid` ("per step, fetches some misses over PCIe and computes the rest on CPU, overlapped. Run `ft bench bw` once per machine to calibrate") — [docs/models.md](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md)

**KTransformers (kt-kernel)**
- A grep of the non-archive tree on 2026-09-28 found no "gpt-oss"/"gpt_oss"/"GptOss" string in any .py/.md/.cpp/.h/.toml file — [repo](https://github.com/kvcache-ai/ktransformers)
- Documented examples — [kt-kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md); [AVX2-Tutorial.md](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/AVX2-Tutorial.md):
  - Qwen3-30B-A3B in BF16 native, AMXINT8 (converted) and LLAMAFILE (GGUF Q4_K_M) modes
  - Qwen3.5-35B-A3B-FP8
  - Qwen3-30B-A3B-GPTQ-Int4
- AVX2 BF16 is described as "BF16 native precision | Zero precision loss" — [AVX2-Tutorial.md](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/AVX2-Tutorial.md)
- In LLAMAFILE mode, `--model` points at the HF checkpoint (used for the GPU side) and `--kt-weight-path` at the GGUF directory (the CPU experts) — [kt-kernel README, Option C](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)
- Expert Deferral: `--kt-max-deferred-experts-per-token` takes 0 (synchronous), 1–4 (recommended) or 5–7 ("may introduce noticeable accuracy loss"). The paper reports an average accuracy drop "within 0.5%" — [kt-kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md); [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)

**Pipelined sharding**
- The b6097 fork's architecture table includes `qwen3moe`, `gpt-oss` (LLM_ARCH_OPENAI_MOE), `deepseek2`, `glm4moe`, `qwen2moe` and `gemma3n`, and has `GGML_TYPE_MXFP4`. No Qwen3.5/3.6 (`qwen35moe`) or DeepSeek-V4 entry exists — [src/llama-arch.cpp](https://github.com/deepshnv/pipeshard-mlsys26-ae/blob/main/src/llama-arch.cpp); [ggml.h](https://github.com/deepshnv/pipeshard-mlsys26-ae/blob/main/ggml/include/ggml.h)
- The paper artifacts use Qwen3-30B-A3B-Instruct-2507 Q4_0 and Qwen3-235B-A22B-Instruct-2507 Q2_K GGUFs — [README](https://github.com/deepshnv/pipeshard-mlsys26-ae/blob/main/README.md)
- Exactness: "our optimizations preserve model accuracy and do not change the operations executed" — [arXiv HTML](https://arxiv.org/html/2604.26334)

**MoE-Infinity**
- Supported families include DeepSeek-V2/V3, DeepSeek-V4-Flash (FP4 experts), Mixtral, Qwen3-MoE (`Qwen/Qwen3-30B-A3B`), Qwen3.5-MoE (`Qwen/Qwen3.5-35B-A3B`, needs transformers ≥ 5.12, served text-only) and GPT-OSS (`openai/gpt-oss-*`). It is a fetch-to-GPU design: it offloads "expert weights to host memory and SSD, then fetch[es] them when needed" — [README](https://github.com/EfficientMoE/MoE-Infinity/blob/main/README.md)

**HybriMoE, 2512.16473, DALI**
- HybriMoE: GGUF (DeepSeek-V2-Lite-Chat Q4_K_M in the README). Only `DeepSeek-V2-Chat-gpu.yaml` enables the HybriMoE cache (`KExpertsMarlin`) — [README](https://github.com/PKU-SEC-Lab/HybriMoE/blob/main/README.md); [optimize_rules](https://github.com/PKU-SEC-Lab/HybriMoE/tree/main/ktransformers/optimize/optimize_rules)
- 2512.16473: the code branches only on `PhiMoEForCausalLM` vs `MixtralForCausalLM`. Flags are `--cache-nblocks`, `--cache-nways` and `--cache-replace-policy` (default "FIFO") — [src/main.py](https://github.com/elsa-lab/MoE-CPU-GPU-Collaborative-Inference/blob/main/src/main.py)
- DALI evaluated DeepSeek-V2-Lite-Chat, Qwen3-30B-A3B and Mixtral-8x7B and states that residual prefetching "requires no fine-tuning or retraining" (lossless) — [arXiv HTML](https://arxiv.org/html/2602.03495)

### Inferences
- **Race model A: gpt-oss-120b.** It needs no conversion: the llama.cpp GGUF and the HF checkpoint both carry MXFP4 experts. On a 96 GB RTX PRO 6000 the whole model fits in VRAM, so a "fused"/all-GPU run gives a free full-load ceiling next to the speed-of-light. Entrants:
  - llama.cpp master best config
  - the user's patch
  - FreeToken (offload/cpu/hybrid)
  - Pipelined sharding
  - optionally MoE-Infinity and a llama.cpp community cache
- **Race model B: Qwen3-30B-A3B in BF16.** This is the only format that FreeToken's CPU path, KTransformers' native backend and llama.cpp (BF16 GGUF) share exactly. Add Q4-class GGUF runs only for the llama.cpp-family entrants.
- **Qwen3.6-35B-A3B** (the audited FreeToken rows) is worth a third model only on a ≥128 GB host, and without Pipelined sharding.
- **KTransformers' LLAMAFILE mode** mixes precisions: BF16 GPU experts and non-expert weights with Q4 CPU experts. It is not like-for-like with any other entrant, so use BF16 native.

### Gaps
- Whether ggml-org's gpt-oss-120b GGUF and the HF checkpoint store identical non-expert (attention) precision was not checked. Measure `mosl/gguf_bytes.py` bytes against HF bytes before claiming byte-equal.
- Whether the b6097-era gpt-oss support in Pipelined sharding has the later llama.cpp gpt-oss correctness and speed fixes was not checked.
- HybriMoE's exactness was not verified from its code.

## Q4. How does each system set its GPU memory budget, and how can it be matched to llama.cpp `--n-cpu-moe` at equal VRAM?

### Takeaway
Every raceable system has a direct expert-budget knob:
- FreeToken: a global slot count or rate.
- KTransformers: a static per-layer expert count.
- Pipelined sharding: a total-VRAM MB cap.
- MoE-Infinity: a device-memory fraction.
- HybriMoE: per-layer cache slots.

None of these knobs by itself fixes total VRAM, because KV cache, CUDA graphs and workspace differ by engine. Equal VRAM should be enforced on measured total process memory (NVML/nvidia-smi peak) at a fixed context length, with each engine's expert budget tuned until its total matches the llama.cpp configuration.

### Cited Findings
- **FreeToken**:
  - `--memory-ratio` (default 0.9) is the "Fraction of free VRAM the engine may use (weights + MoE cache + KV)".
  - `--moe-cache-size / --moe-cache-rate / --moe-cache-auto` set the "GPU expert-cache size as slots / fraction of all experts / sized from free VRAM".
  - `--num-pages / --num-tokens` override KV capacity; `--kv-reserve-tokens` (default 8192) is the "KV token floor reserved before `--moe-cache-auto` fills experts".
  - `--cuda-graph-max-bs` sets the captured graph sizes.
  - Sources: [docs/cli.md](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md)
  - The cache is one LRU over all layers ("shared, all-layer"), and hybrid "fetches at most K missing experts per layer" — [arXiv HTML](https://arxiv.org/html/2608.16157); [moe/\_\_init\_\_.py](https://github.com/FlashML-org/FreeToken/blob/main/python/freetoken/moe/__init__.py)
- **KTransformers**:
  - `--kt-num-gpu-experts` is the "Number of experts to keep on GPU". The Qwen3-30B-A3B example says "With 24GB GPU memory, we can fit ~32 experts on GPU for this model" — [kt-kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)
  - SGLang's wrapper treats "GPU experts [as] those with ID < `num_gpu_experts`", i.e. a per-layer count. Initial placement strategies are `uniform` (default), `frequency`, `front-loading` and `random` — [kt_ep_wrapper.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/kt_ep_wrapper.py); [kt-kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)
  - The OOM guidance lists `--kt-num-gpu-experts`, `--chunked-prefill-size`, `--max-total-tokens` and `--mem-fraction-static` (default 0.80) as the VRAM levers — [Qwen3-Coder-Next-Tutorial.md](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/Qwen3-Coder-Next-Tutorial.md)
- **Pipelined sharding**:
  - `-mva N` is the "Max VRAM allocation budget in MB", used with `-pipe-shard` and env `GGML_CUDA_PIPELINE_SHARDING=1 GGML_CUDA_REGISTER_HOST=1` — [README](https://github.com/deepshnv/pipeshard-mlsys26-ae/blob/main/README.md)
  - The budget is partitioned "into pinnable and scratch areas. We pin as many high-priority sub-layers as possible". Priority order is attention, then KV cache, FFN and outputs. Experts are handled by one of three plans ("GPU-only" streaming, "Static" CPU-mapped, "Dynamic"), chosen "using profile-based timing estimation" — [arXiv HTML](https://arxiv.org/html/2604.26334)
  - The same binary also has `--cpu-moe/-cmoe` and `--n-cpu-moe/-ncmoe` — [common/arg.cpp](https://github.com/deepshnv/pipeshard-mlsys26-ae/blob/main/common/arg.cpp)
  - The README warns that over-subscribing VRAM "can cause degraded performance" — [README](https://github.com/deepshnv/pipeshard-mlsys26-ae/blob/main/README.md)
- **MoE-Infinity**: `"device_memory_ratio": 0.75, # 75% of the device memory is used for caching` — [README](https://github.com/EfficientMoE/MoE-Infinity/blob/main/README.md)
- **HybriMoE**: `--cache_size` (experts per layer) and `--prefetch_size` — [README](https://github.com/PKU-SEC-Lab/HybriMoE/blob/main/README.md)
- **2512.16473**: `--cache-nblocks` × `--cache-nways` (N-index, M-way set-associative); the README recommends `--cache-nblocks 14 --cache-nways 4` for a 4090 with Mixtral — [README](https://github.com/elsa-lab/MoE-CPU-GPU-Collaborative-Inference/blob/main/README.md)
- **Community cache precedent**: everett6's headline compares `-ncmoe 22` with no cache against `-ncmoe 48 --moe-expert-cache 72` (~8.8 GB cache). That is not an equal-VRAM comparison — [FINAL_RESULTS.md](https://github.com/everett6/moe-prefetch/blob/main/docs/FINAL_RESULTS.md)

### Inferences
- Let L be MoE layers, E experts per layer and s_e bytes per expert. In llama.cpp, `--n-cpu-moe k` puts (L−k)·E·s_e of expert weights on the GPU. Equal expert bytes then give:
  - the user's per-layer cache: C = E·(L−k)/L slots per layer
  - KTransformers: N = E·(L−k)/L GPU experts per layer, with uniform placement and deferral 0
  - FreeToken: S = (L−k)·E slots, or `--moe-cache-rate (L−k)/L`
  - MoE-Infinity: `device_memory_ratio` ≈ budget / device memory
  - Pipelined sharding: `-mva` = the llama.cpp configuration's *measured* total MB
- Rough estimates from parameter counts; compute exact bytes with `mosl/gguf_bytes.py` or `mosl/archs.py`:
  - gpt-oss-120b: 36 layers × 128 experts, ≈13 MB per MXFP4 expert, ≈1.7 GB per layer
  - Qwen3-30B-A3B BF16: 48 × 128, ≈9.4 MB per expert, ≈1.2 GB per layer
- `--n-cpu-moe` steps in whole layers. `-ot` on individual `ffn_{gate,up,down}_exps` tensors gives about one-third-layer granularity if a finer llama.cpp point is needed. Tensor naming for gpt-oss GGUFs was not verified.
- The equal-VRAM protocol should fix context length (same `-c` / `--num-tokens` / `--max-total-tokens`), disable speculative decoding, fix CPU threads (all engines at the same physical-core count, or also a FreeToken-paper-style 6-thread pin), and record NVML peak memory per process.
- FreeToken's pooled all-layer cache is a structural advantage over per-layer budgets at equal bytes. The speed-of-light already has a pooled-budget variant ("per-layer and pooled budget"), so FreeToken should be scored against the pooled bound, and the per-layer systems against the per-layer bound.

### Gaps
- Whether `--kt-num-gpu-experts` is exactly per layer (inferred from the wrapper's ID < N mask) or a total was not confirmed in kt-kernel source.
- Whether Pipelined sharding's `-mva` includes the CUDA context and cuBLAS workspace, and how much it undershoots or overshoots measured nvidia-smi memory, is unknown.
- FreeToken's CUDA-graph and FlashInfer workspace size at batch 1 is undocumented. `--cuda-graph-max-bs 1` should minimise it, but this was not tested.

## Q5. Reported decode numbers (with hardware), independent reproductions, open issues and known bugs

### Takeaway
- **FreeToken** reports the highest single-GPU numbers: 77–83 tok/s on Qwen3.6-35B-A3B BF16 (RTX 5090), and GLM-5.2 at 14.9 vs llama.cpp's 7.3 on an RTX PRO 6000. The community has independently run gpt-oss-120b on a 5090 at 127.1 tok/s (offload, 40% of experts resident, ~29 GB VRAM). No one has reproduced the paper's comparisons, and the paper gives no llama.cpp flags.
- **KTransformers** has the only formal reproduction (the SOSP AE badge), for the old code. Current numbers on 5090 + dual EPYC 9355 are 76 tok/s (Qwen3-Coder-Next FP8) and 20+ tok/s (DeepSeek-V4-Flash).
- **Pipelined sharding** reports 25.7–158.6 tok/s on Qwen3-30B-A3B Q4_0 (RTX 5090, Windows). It has no independent runs, and its 0 GB baseline already sits far below the user's model's prediction.

### Cited Findings
**FreeToken**
- Paper, RTX 5090: "77–83 tok/s on Qwen3.6 and 22–25 tok/s on DSV4-Flash, 1.8–2.3× and 1.5–1.9× the strongest baseline in each workload" — [arXiv 2608.16157](https://arxiv.org/abs/2608.16157)
- Paper, RTX PRO 6000 (Xeon 8559C, 512 GiB): GLM-5.2 NVFP4 at "14.9 tok/s" vs llama.cpp "7.3". The paper gives "no explicit configuration details (flags, version, commit, or GPU memory matching methodology)" for llama.cpp or KTransformers — [arXiv HTML](https://arxiv.org/html/2608.16157)
- The paper's rented 5090/4090/3090 servers ran "capped at 6 CPU threads and pinned to the GPU's NUMA node" — [arXiv HTML §5.1](https://arxiv.org/html/2608.16157)
- Audited rows (from the project audit):
  - Qwen3.6-35B-A3B BF16, RTX 5090: 77.1 at 23.8 GB vs paper llama.cpp 42.6, predicted 35.1 [24.8, 51.3]
  - Same model, RTX 5090: 73.8 at 26.3 GB
  - Same model, RTX 4090: 42.9 at 18.3 GB
  - All labelled "not adjudicated: band wider than ±40%"
  - Source: [audit.md](/home/claude/moe-speed-of-light/prereg/audit/audit.md)
- Community, RTX 5090 + 128 GB, FreeToken 0.1.2 (2026-08-22): gpt-oss-120b with `--moe-backend offload --moe-cache-auto` held "1863 out of all 4608 experts, about 40.4%" resident. It used "about 29GB" VRAM and "about 66GB" host RAM and ran at "127.1 tok/s single-stream". No llama.cpp run of gpt-oss-120b was made. The same author measured, on a model that fits in VRAM, FreeToken at 196.9 vs llama.cpp Q4_K_M at 296.4 tok/s single-request — [zenn.dev/holy_fox](https://zenn.dev/holy_fox/articles/53b82eed45f956?locale=en)
- Community, RTX 5090 under WSL2 (2026-08-25): gpt-oss-120b at "approximately 50 tokens/s" with `--moe-cache-size 500 --num-tokens 65536`. Auto-sizing failed under WSL2/Blackwell — [zenn.dev/lifona](https://zenn.dev/lifona/articles/a8606bb95e17e1?locale=en)
- A third-party review "did not reproduce the paper's hardware benchmarks" — [Wavect](https://wavect.io/blog/freetoken-ai-inference-engine-review/)
- Open issues found:
  - #554: engine install fails on update ("sglang-kernel hash mismatch, flashinfer version mismatch")
  - #312 and #276: GPU/driver detection
  - #409: VRAM floor on a 12 GB card
  - The repo has ~204 open issues
  - Sources: [#554](https://github.com/FlashML-org/FreeToken/issues/554); [#312](https://github.com/FlashML-org/FreeToken/issues/312); [#276](https://github.com/FlashML-org/FreeToken/issues/276); [#409](https://github.com/FlashML-org/FreeToken/issues/409); [shields.io](https://img.shields.io/github/issues/FlashML-org/FreeToken)

**KTransformers**
- SOSP'25, batch-1 decode: "1.25× to 1.76× over Llama.cpp" at full precision and "1.77× to 1.93×" quantized, both without deferral, against a Feb-2025 llama.cpp the authors extended with custom expert-level offload. Hardware was dual Xeon 8452Y with an A100-40GB or RTX 4080 — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- The SOSP'25 AE awarded "Results Reproduced" — [sysartifacts](https://sysartifacts.github.io/sosp2025/results)
- Current docs, single concurrency:
  - Qwen3-Coder-Next FP8, 1× 5090 + 2× EPYC 9355: 75.9 tok/s decode — [Qwen3-Coder-Next-Tutorial.md](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/Qwen3-Coder-Next-Tutorial.md)
  - MiniMax-M2.1: 32.1 tok/s — [MiniMax-M2.1-Tutorial.md](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/MiniMax-M2.1-Tutorial.md)
  - DeepSeek-V4-Flash: "20+ tok/s on a single RTX 5090" — [DeepSeek-V4-Flash.md](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash.md)
  - Leaderboard: Qwen3.5-35B-A3B FP8 at 86.2 tok/s on 1× RTX 5090, with no CPU or GPU-expert count stated — [ktransformers.net/benchmarks](https://ktransformers.net/en/benchmarks)
- The SeqMoE paper's Fig. 1 (Qwen3-30B-FP8, RTX 4090, 45% loaded) shows KTrans at 34.2, llama.cpp "Static Offload" at 30.9, MoE-Infinity at 11.9 and SeqMoE at 104.1 tok/s — [arXiv 2609.12978](https://arxiv.org/abs/2609.12978)
- Issues:
  - #1754: non-NUMA segfault
  - #2081: SM_120 FP8 assert
  - #1885: CUDA-graph capture assert on DeepSeek-V3.2
  - ~467 open issues
  - Sources: [#1754](https://github.com/kvcache-ai/ktransformers/issues/1754); [#2081](https://github.com/kvcache-ai/ktransformers/issues/2081); [#1885](https://github.com/kvcache-ai/ktransformers/issues/1885); [shields.io](https://img.shields.io/github/issues/kvcache-ai/ktransformers)

**Pipelined sharding**
- Table 4 on cli3 (RTX 5090, 16-core EPYC, 256 GB, Gen5, Windows 11), decode TPS at 1K context:

  | Model | 2G | 4G | 8G | 16G | 32G |
  |---|---|---|---|---|---|
  | Qwen3-30B-A3B Q4_0 | 25.7 | 26.2 | 32.1 | 47.8 | 158.6 |
  | Qwen3-235B Q2_K | 7.7 | 8.7 | 9.1 | 9.7 | 11.5 |

  Source: [arXiv HTML](https://arxiv.org/html/2604.26334)
- The paper's llama.cpp baseline is `-ngl` layer offload found by "trial-and-error" search for the maximal layer count within the VRAM budget — [arXiv HTML](https://arxiv.org/html/2604.26334)
- Audit: at the 0 GB row, the paper's own `-ngl + -cmoe` llama.cpp measured 25.7 against a predicted 49.5 [38.4, 65.1] — [system_reproducibility.md](/home/claude/moe-speed-of-light/research_notes/MoE%20audit%20measurement%20plan/system_reproducibility.md); [audit.md](/home/claude/moe-speed-of-light/prereg/audit/audit.md)
- The artifact ships PASS/FAIL validation scripts ("Our validation scripts will print PASS (or not) along with error margins"). No third-party run was found — [arXiv HTML Appendix A](https://arxiv.org/html/2604.26334)

**MoE-Infinity, HybriMoE, 2512.16473, DALI**
- MoE-Infinity: the paper reports 3.1–16.7× per-token latency gains on an RTX A5000, but the current code differs from the paper — [arXiv 2401.14361](https://arxiv.org/abs/2401.14361); [README](https://github.com/EfficientMoE/MoE-Infinity/blob/main/README.md)
- HybriMoE (RTX A6000, Xeon 5220R with 10 cores): DeepSeek-V2-Lite at 14.5/15.3 tok/s (1.9/5.6 GB), against the model-predicted llama.cpp of 80.9/140.0. Issues #7, #8 and #14 are unanswered — [audit.md](/home/claude/moe-speed-of-light/prereg/audit/audit.md); [issues](https://github.com/PKU-SEC-Lab/HybriMoE/issues)
- 2512.16473 (RTX 4090 + Threadripper 7960X): "up to 4.8 and 10.4 tokens per second" for Mixtral and Phi-3.5-MoE — [arXiv 2512.16473](https://arxiv.org/abs/2512.16473)
- DALI (RTX 3090 + EPYC 7532, 256 GB DDR4): per the fetch summary, "The paper does not report batch size 1 results. Minimum reported batch size is 4 or 8". This conflicts with the earlier note that decode was measured "under various batch sizes", so check it against the PDF — [arXiv HTML](https://arxiv.org/html/2602.03495)

**llama.cpp community caches**
- everett6, Qwen3-30B-A3B Q4_K_M on RTX 5070 12 GB + Ryzen 9 7950X:
  - "PR #27861's cache as it ships": 35.5 tok/s
  - `-ncmoe 22` baseline: 79.1
  - the patched cache: 118.2 (1.49×)
  - The author diagnosed the shipped PR as follows: "the cache drains to 0.6 resident experts per layer — slower than no cache at all"
  - Source: [FINAL_RESULTS.md](https://github.com/everett6/moe-prefetch/blob/main/docs/FINAL_RESULTS.md); [README](https://github.com/everett6/moe-prefetch)
- leloch's RFC reports +10% to +57% decode across forced-offload models, including RTX 5090 + DDR5 runs — [Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)

### Inferences
- **FreeToken is the entrant most likely to beat both llama.cpp and the user's system.** It combines a pooled all-layer cache, a bandwidth-calibrated fetch/CPU split, and a stream-memop GPU→CPU handshake that is architecturally the same idea as the user's GPU-signalled helper threads. That handshake has been public since 11 Aug 2026, before the user's patch was committed (the repo's first commit touching `runtime/llama.cpp-expert-cache.patch` is 2026-09-26). The prior novelty notes, which found no engine with a GPU-initiated, polled hand-off, are superseded on that point. The user's mechanism uses a GPU spin-wait kernel; FreeToken reports abandoning that for power reasons on laptops. The paper should cite FreeToken and differentiate on measured hand-off latency, not claim the mechanism.
- **The 127 tok/s community gpt-oss-120b number is a sanity target for the RTX 5090.** It was a fetch-only (`offload`) run on PCIe 5.0 at ~29 GB VRAM, so the race should include FreeToken `offload` as well as `hybrid`/`cpu`. On a 96 GB PRO 6000, FreeToken will be budget-limited only by the cap set.
- **The PR #27861 cache as shipped is a strawman.** Any llama.cpp-community entrant must use a fixed build (everett6's 0006 patch, or leloch v2).

### Gaps
- No third-party equal-VRAM comparison exists for any candidate against tuned llama.cpp `--n-cpu-moe`. That is the race's contribution.
- KTransformers batch-1 numbers on a single-socket host, or for Qwen3-30B-A3B on a 5090, were not found.
- The exact llama.cpp configurations behind the FreeToken paper baselines are unknown.

## Q6. Estimated effort (hours) to get each running on Linux with an RTX PRO 6000 or RTX 5090 plus 30–60 cores and 90–180 GB RAM, and blockers

### Takeaway
Within about 2 weeks and $40, four entrants besides llama.cpp and the user's patch are realistic:
- FreeToken: ~3–6 h
- Pipelined sharding: ~2–4 h
- KTransformers kt-kernel/SGLang: ~6–12 h, and only on an AVX-512 host
- one llama.cpp community cache: ~1–3 h

MoE-Infinity is an optional fifth at ~4–8 h. HybriMoE, 2512.16473 and Fiddler are poor value on Blackwell with a 90 GB box. DALI, SeqMoE and 2606.10493 cannot be run.

At Verda's $1.96/h on-demand, $40 buys about 20 GPU-hours ($0.98/h spot, about 40). Build and model downloads should therefore be done on CPU-only time or overlapped.

### Cited Findings
- Verda 1× RTX PRO 6000: $1.96/h on-demand, $0.9820/h spot, 30 threads, 90 GB RAM — [Verda](https://verda.com/rtx-pro-6000)
- 1× RTX 5090 offers range from ~$0.35 to $1.49/h, with bundled RAM from 8 to 144 GB (page dated 2026-09-28) — [getdeploying RTX 5090](https://getdeploying.com/gpus/nvidia-rtx-5090)
- Pipelined sharding: "~20 minutes to build from source; 1–3 hours to download all models" (175 GB, including non-MoE models). Repro scripts run in-tree GPU/concurrent profilers first — [arXiv HTML Appendix A](https://arxiv.org/html/2604.26334); [README](https://github.com/deepshnv/pipeshard-mlsys26-ae/blob/main/README.md)
- FreeToken: kernels "JIT-compiled on first use". `ft bench bw` must run once per machine to calibrate hybrid. The `benchmarks/bench_decode_moe.py --backend offload,cpu,hybrid` harness measures "bs=1 decode tok/s of a served MoE model" through the full serving path — [docs/install.md](https://github.com/FlashML-org/FreeToken/blob/main/docs/install.md); [benchmarks/README.md](https://github.com/FlashML-org/FreeToken/blob/main/benchmarks/README.md)
- KTransformers needs a source build for sm_120 (`CPUINFER_CUDA_ARCHS`), torch 2.9.1 and the `sglang-kt` fork. It also needs `--kt-threadpool-count` equal to the number of NUMA nodes, and has non-NUMA segfault history — [kt-kernel/setup.py](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/setup.py); [pyproject.toml](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/pyproject.toml); [issue #1754](https://github.com/kvcache-ai/ktransformers/issues/1754)
- MoE-Infinity's SM120 build needs CUTLASS headers and a CUDA toolkit matching torch's major version — [README](https://github.com/EfficientMoE/MoE-Infinity/blob/main/README.md)

### Inferences
These are hour estimates by judgment from the install docs, for engineer time with the model already downloaded.

- **llama.cpp master, best configuration:** 1–2 h. Build takes ~15 min, then a `llama-bench` sweep over `--n-cpu-moe`, `-t`, `--no-mmap`, `GGML_CUDA_GRAPH_OPT=1` and `-fa`. No blockers.
- **FreeToken:** 3–6 h.
  - Blockers:
    - driver r580+ and a CUDA 13 `nvcc`
    - install hash mismatches (#554)
    - auto-sizing failures on Blackwell (use explicit `--moe-cache-size --num-tokens`)
    - memop handshake possibly disabled in a VM
  - Run `ft bench bw`, then offload, cpu and hybrid at each budget.
  - Record whether flag-sync or host-func was selected; it logs the probe result per the code comment.
- **Pipelined sharding:** 2–4 h.
  - Blockers:
    - Linux is secondary (Windows-first)
    - profiler runs must precede sharding
    - b6097 lacks recent llama.cpp speedups, so also race llama.cpp master
    - no Qwen3.5/3.6 support
- **KTransformers:** 6–12 h, and it enters only if `lscpu` shows avx512_bf16 (Zen 4/5 or Sapphire Rapids+).
  - Blockers:
    - source build for sm_120, with FlashInfer JIT for 12.0a
    - torch pin conflicts, so use a separate venv
    - single-NUMA segfault risk: try `--kt-threadpool-count 1` and patch `worker_pool.h` if needed
    - no gpt-oss
    - 90 GB RAM is tight for BF16 Qwen3-30B-A3B plus SGLang
  - Use `--kt-max-deferred-experts-per-token 0` for exactness, and optionally report deferral 2 as a labelled lossy variant.
- **llama.cpp community cache:** 1–3 h. Use leloch `moe-cache-v2-pr`, or PR #27861 + everett6 `0006` on `bccbacd`. Blocker: the silent no-op bug in shipped #27861, so verify cache allocation in the logs.
- **MoE-Infinity:** 4–8 h. The CUTLASS/SM120 build is the risk. It is fetch-only, so it represents the "fetch camp" (SeqMoE-like) that has no runnable code otherwise.
- **HybriMoE:** 1–3 days (port to CUDA 12.8/torch ≥2.7 for sm_120). It only runs DeepSeek-V2-Lite with its cache. Not worth it for this race; run it on an older-GPU rental only if the audit needs that row.
- **2512.16473:** 3–5 h (pure PyTorch). It needs Mixtral/Phi-3.5-MoE BF16 (~84–93 GB), which will not fit 90 GB alongside anything else. Skip unless the host has ≥128 GB.
- **Fiddler:** skip (torch 2.1.2, Mixtral-only).
- **Not runnable:** DALI, SeqMoE, 2606.10493.

**Budget sketch at Verda on-demand**
- One working day of GPU time (~8 h ≈ $16) covers two models × 4–5 budgets × 5 entrants.
- Keep ~$10 for a second session, and ~$10 as slack for FreeToken/KTransformers build failures.
- Download weights (gpt-oss-120b ≈ 61–65 GB; Qwen3-30B-A3B BF16 ≈ 61 GB) to the instance's volume at the start. Pre-build llama.cpp variants on a CPU-only box when possible.

### Gaps
- Whether Verda allows persistent volumes across stop/start (to avoid re-downloading), and at what storage price, was not checked.
- No published install-time report exists for KTransformers on sm_120 in a VM.

## Q7. Ranked shortlist: which 2–4 systems should enter the race, and why

### Takeaway
1. **FreeToken** (pin v0.1.3, and record the main commit used). It is the newest and strongest published system, the closest mechanism to the user's (cache plus CPU misses plus GPU-signalled handshake), and it natively runs both race models on sm_120.
2. **KTransformers kt-kernel v0.7.1 + sglang-kt**, if the host has AVX-512-BF16. It is the canonical SOSP'25 CPU/GPU hybrid and the most-cited baseline, is exact at deferral 0, and runs Qwen3-30B-A3B BF16 but not gpt-oss.
3. **Pipelined sharding v2.0.3-mlsys26.** It is an MLSys'26 oral with an exact llama.cpp-based artifact, the cheapest to run, supports both gpt-oss-120b and Qwen3-30B-A3B GGUF, and its `-mva` knob is literally "equal VRAM".
4. **One llama.cpp community cache**, either leloch v2 or #27861 with everett6's fix. It is not a paper, but it is the same-codebase competitor reviewers will ask about.

Drop:
- DALI, SeqMoE, 2606.10493: no code.
- HybriMoE, Fiddler: no sm_120 path, obsolete models.
- 2512.16473: Mixtral/Phi only, RAM.
- MoE-Infinity: optional fetch-camp reference if hours remain.

### Cited Findings
- FreeToken: known-good support for gpt-oss-120b and Qwen3-30B-A3B; exact decode; a stream-memop handshake since 2026-08-11; community 5090 runs at 127.1 tok/s on gpt-oss-120b — [docs/models.md](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md); [arXiv HTML](https://arxiv.org/html/2608.16157); [cpu_executor.py](https://github.com/FlashML-org/FreeToken/blob/main/python/freetoken/moe/cpu_executor.py); [zenn.dev/holy_fox](https://zenn.dev/holy_fox/articles/53b82eed45f956?locale=en)
- KTransformers:
  - SOSP'25 with AE "Results Reproduced"
  - BF16/FP8 native backends on AVX-512 (incl. Zen 4+)
  - `--kt-max-deferred-experts-per-token 0` for synchronous (exact) execution
  - no gpt-oss support found
  - Sources: [sysartifacts](https://sysartifacts.github.io/sosp2025/results); [kt-kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)
- Pipelined sharding: tagged artifact on Zenodo; "do not change the operations executed"; `-mva` VRAM budget; gpt-oss and qwen3moe architectures in the fork — [Zenodo](https://zenodo.org/records/19436383); [arXiv HTML](https://arxiv.org/html/2604.26334); [src/llama-arch.cpp](https://github.com/deepshnv/pipeshard-mlsys26-ae/blob/main/src/llama-arch.cpp)
- llama.cpp community caches: unmerged, with documented bugs in the shipped #27861 and fixes in everett6's patches — [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861); [everett6/moe-prefetch](https://github.com/everett6/moe-prefetch); [Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)

### Inferences
**Per-candidate dossier summary** (my synthesis of Q1–Q6)

| System | Code / version | sm_120 | Race models | Budget knob | Exact? | Effort | Verdict |
|---|---|---|---|---|---|---|---|
| llama.cpp master (best config) | ggml-org master | yes | gpt-oss-120b MXFP4, Qwen3-30B BF16/Q4 | `--n-cpu-moe`, `-ot` | yes | 1–2 h | reference |
| User's patch (2145525a) | local | yes | same GGUFs | per-layer slots | yes | done | contender |
| FreeToken | v0.1.3 / main 0d652e7, Apache-2.0 | yes (driver r580+, CUDA 13) | gpt-oss-120b, Qwen3-30B BF16 (Qwen3.6 BF16 if ≥128 GB) | `--moe-cache-size/-rate`, `--num-tokens`, `--memory-ratio` | yes (not bit-identical) | 3–6 h | **enter (#1)** |
| KTransformers kt-kernel + sglang-kt | v0.7.1, Apache-2.0 | source build | Qwen3-30B BF16 only | `--kt-num-gpu-experts` (per layer), `--mem-fraction-static` | yes at deferral 0 | 6–12 h | **enter if AVX-512-BF16 (#2)** |
| Pipelined sharding | v2.0.3-mlsys26, MIT, Zenodo | yes (CUDA ≥12.8, native arch) | gpt-oss-120b, Qwen3-30B GGUF (not Qwen3.6) | `-mva` MB | yes (same ops) | 2–4 h | **enter (#3)** |
| llama.cpp community cache | leloch v2 / #27861 + fix | yes | same GGUFs | cache MB or slots + `-ncmoe` | yes | 1–3 h | **enter (#4)** |
| MoE-Infinity | main, Apache-2.0 | `MOE_ENABLE_SM120=1` | gpt-oss, Qwen3-30B, Qwen3.5-35B | `device_memory_ratio` | yes (fetch-only) | 4–8 h | optional |
| HybriMoE | frozen 2025-12 | port needed | DeepSeek-V2-Lite only | `--cache_size` | likely | 1–3 d | drop |
| 2512.16473 | frozen 2025-10, MIT | likely (pure PyTorch) | Mixtral / Phi-3.5 BF16 | nblocks × nways | yes | 3–5 h + RAM | drop (90 GB) |
| Fiddler | frozen 2024 | no (torch 2.1.2) | Mixtral BF16 | static profile | yes | — | drop |
| DALI, SeqMoE, 2606.10493 | no code | — | — | — | — | — | cannot enter |

**Why this ranking**
- FreeToken and KTransformers are the two systems a reviewer would expect in any 2026 hybrid-MoE comparison. FreeToken is also the direct prior art for the user's hand-off mechanism, so beating or matching it at equal VRAM, scored against the pooled and per-layer speed-of-light, is the headline result.
- Pipelined sharding is nearly free to add, shares GGUFs with llama.cpp, and tests an audited "weak baseline" row on the same GPU class.
- The community cache guards against "you only beat stock llama.cpp".

**Conditional plan if the Verda CPU lacks AVX-512**
- Swap KTransformers for MoE-Infinity, or run KTransformers on AVX2 labelled as such.
- Alternatively, rent a Vast 5090 on a Zen 4/5 or Sapphire Rapids+ host.

**Scoring**
- Score pooled-cache FreeToken against the pooled speed-of-light and per-layer systems against the per-layer bound.
- Also report every entrant as a fraction of the all-in-VRAM ("fused") run that the 96 GB PRO 6000 makes possible.

### Gaps
- Final entry of KTransformers depends on the rented CPU's ISA, which is unknown until provisioning.
- Whether FreeToken's memop handshake survives in a Verda VM, and whether its hybrid split beats its own `offload` on PCIe 5.0 for gpt-oss-120b, is unknown until run.
