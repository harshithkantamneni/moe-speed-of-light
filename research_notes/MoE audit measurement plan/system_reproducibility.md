# Reproducibility of audited MoE-offloading systems on rented cloud hardware (as of 28 Sep 2026)

Method note. I cloned each public repository with `git clone --filter=blob:limit=2m` on 2026-09-28 and read its README, docs, install scripts, optimize rules and `git log`. Commit and tag dates below come from those logs. The GitHub REST API and GitHub's issue-search pages were blocked (403 / robots), so star and fork counts come from ungh.cc and shields.io. Issue content comes from individual issue pages fetched with WebFetch, which a small model summarises. Treat issue details as "as summarised". Code-availability statements were checked by fetching each paper's arXiv HTML and grepping for URLs and phrases like "available at" or "open-source". Audited-row numbers are quoted from the project's own audit table (`/home/claude/moe-speed-of-light/prereg/audit/audit.md`), cited below as [audit.md](/home/claude/moe-speed-of-light/prereg/audit/audit.md).

## Q1. Is code public (URL, license, last commit, activity), and does the release match the paper?

### Takeaway
Seven of the thirteen efforts have public code: KTransformers, FreeToken, HybriMoE, 2512.16473, Pipelined sharding, Fiddler and MoE-Infinity, plus the three llama.cpp community branches. Only Pipelined sharding ships a versioned, paper-matched artifact: tags, Zenodo, Docker and repro scripts. KTransformers passed SOSP artifact evaluation, but the repo has since been restructured and the paper-era code now sits in `archive/`. SeqMoE, SP-MoE and MoE-SpeQ have no code link. ProMoE's public repo is incomplete: its llama.cpp and transformers forks live on an internal SJTU git server.

### Cited Findings
**KTransformers (SOSP'25)**
- The repo is kvcache-ai/ktransformers, Apache-2.0, with ~19.5k stars and 1,581 forks. It was last pushed 2026-09-23 — [ungh.cc](https://ungh.cc/repos/kvcache-ai/ktransformers); [LICENSE](https://github.com/kvcache-ai/ktransformers/blob/main/LICENSE)
- git log: 1,347 commits, 70 of them since 2026-06-01. The latest commit is 2026-09-23 and the latest tag is v0.7.1 (2026-09-14). Tags v0.2.3 (Mar 2025) through v0.3.1 (2025-05-17) cover the period around paper submission — [commit history](https://github.com/kvcache-ai/ktransformers/commits/main); [tags](https://github.com/kvcache-ai/ktransformers/tags)
- SOSP 2025 AE results give KTransformers all three badges: Artifacts Available, Evaluated-Functional and Results Reproduced. The artifact link is the main repo, with no pinned commit — [sysartifacts SOSP'25 results](https://sysartifacts.github.io/sosp2025/results)
- The current README says: "The original integrated KTransformers framework has been archived to the `archive/` directory". Inference is now kt-kernel plus a kvcache-ai fork of SGLang (`sglang-kt`) — [README](https://github.com/kvcache-ai/ktransformers/blob/main/README.md); [kt-kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)
- The archive still holds the paper-era optimize rules, including `Qwen2-57B-A14B-Instruct.yaml` and `Qwen2-serve-amx.yaml` (`backend: "AMXInt8" # or "AMXBF16" or "llamafile" (default)`). The archive was last touched 2026-07-19 by a security fix — [archive optimize_rules](https://github.com/kvcache-ai/ktransformers/tree/main/archive/ktransformers/optimize/optimize_rules)

**FreeToken (arXiv 2608.16157)**
- The paper header lists "[Code] https://github.com/FlashML-org/FreeToken" — [arXiv HTML](https://arxiv.org/html/2608.16157)
- The repo is Apache-2.0 with ~13.9k stars and 1,370 forks. It was created 2026-07-20 and last pushed 2026-09-26 — [ungh.cc](https://ungh.cc/repos/FlashML-org/FreeToken)
- git log: the first commit, "feat: initial open-source release", is dated 2026-08-11. There are 88 commits in total. The last, on 2026-09-25, adds ROCm RDNA3/4. Tags are v0.1.2 (2026-08-19), v0.1.3 (2026-09-15) and a rolling `nightly` — [commits](https://github.com/FlashML-org/FreeToken/commits/main); [releases](https://github.com/FlashML-org/FreeToken/releases)

**HybriMoE (DAC'25, arXiv 2504.05897; the id is confirmed)**
- The paper says "Our code is available at: https://github.com/PKU-SEC-Lab/HybriMoE" — [arXiv HTML](https://arxiv.org/html/2504.05897)
- The repo is Apache-2.0 with 121 stars and 18 forks. There are 6 commits, the first on 2025-04-09 and the last on 2025-12-16 (a README path fix) — [ungh.cc](https://ungh.cc/repos/PKU-SEC-Lab/HybriMoE); [commits](https://github.com/PKU-SEC-Lab/HybriMoE/commits/main)
- It is a full fork of the old KTransformers tree (`ktransformers/local_chat.py`, `ktransformers/operators/experts.py`). `version.py` shows "LastEditTime : 2025-02-15", i.e. KTransformers ~v0.2.x — [repo](https://github.com/PKU-SEC-Lab/HybriMoE)

**CPU-GPU collaborative inference (arXiv 2512.16473; Huang, Lin, Lee; NTU/NTHU; ASP-DAC 2026)**
- The paper says "The implementation of our framework is available at github.com/elsa-lab/MoE-CPU-GPU-Collaborative-Inference" — [arXiv HTML](https://arxiv.org/html/2512.16473)
- The repo is MIT with 10 stars and 1 fork. There are 42 commits from 2024-10-06 to 2025-10-16; the last is "Upload paper" — [ungh.cc](https://ungh.cc/repos/elsa-lab/MoE-CPU-GPU-Collaborative-Inference); [repo](https://github.com/elsa-lab/MoE-CPU-GPU-Collaborative-Inference)

**Pipelined sharding (NVIDIA, MLSys'26 oral, arXiv 2604.26334)**
- The artifact is at github.com/deepshnv/pipeshard-mlsys26-ae (MIT) and archived on Zenodo (records/19436383). It is "implemented … on top of llama.cpp tag b6097" — [arXiv HTML, Appendix A](https://arxiv.org/html/2604.26334); [repo](https://github.com/deepshnv/pipeshard-mlsys26-ae); [Zenodo](https://zenodo.org/records/19436383)
- git log: 26 commits from 2026-02-16 to 2026-06-01. Tags v1.0.0-mlsys26 (2026-02-18) through v2.0.3-mlsys26 (2026-04-06). The repo has 3 stars — [repo](https://github.com/deepshnv/pipeshard-mlsys26-ae); [ungh.cc](https://ungh.cc/repos/deepshnv/pipeshard-mlsys26-ae)

**SeqMoE (arXiv 2609.12978, 11 Sep 2026)**
- The paper's HTML contains no repository link. Its only GitHub URLs point to llama.cpp, the vLLM fused-MoE file, DeepGEMM and a SYCL doc. It says: "We implement SeqMoE in Hugging Face Transformers … using hooks" — [arXiv HTML](https://arxiv.org/html/2609.12978)
- A web search for SeqMoE code found only the arXiv page and aggregators — [search results incl. Pith](https://pith.science/paper/2609.12978)

**SP-MoE (arXiv 2510.10302) and MoE-SpeQ (arXiv 2511.14102)**
- Neither paper's arXiv HTML contains a code URL or availability statement; the only links are dataset/model links and DOIs — [SP-MoE HTML](https://arxiv.org/html/2510.10302); [MoE-SpeQ HTML](https://arxiv.org/html/2511.14102)
- Web searches turned up no official repositories. Unrelated similarly named repos exist, e.g. MoE-SpAc — [MoE-SpAc repo](https://github.com/lshAlgorithm/MoE-SpAc)

**ProMoE (arXiv 2410.22134)**
- The paper says "The source code of ProMoE is publicly available at https://github.com/promoe-opensource/promoe" — [arXiv HTML](https://arxiv.org/html/2410.22134)
- The repo has 20 stars and no LICENSE file. It has 197 commits (2024-03-31 to 2024-10-30) and was created on GitHub 2025-01-27. The README title is "MoE Cache" — [ungh.cc](https://ungh.cc/repos/promoe-opensource/promoe); [repo](https://github.com/promoe-opensource/promoe)
- `install.md` clones seven repos from `git@ipads.se.sjtu.edu.cn:sparsellm/…`: transformers, eval-helper, expert-selection-tracer, hqq, sparse-llm-cache-scripts, **llama.cpp** and sparse-llm-cache. The public repo holds only 70 files: the cache library `src/cpp_worker` and `src/sparse_llm_cache`, plus demos — [install.md](https://github.com/promoe-opensource/promoe/blob/main/install.md)
- The README's llama.cpp run command uses fork-only flags (`--moe_cache 1 --moe_cache_rate 0.375 --pred_model_path …/moe-layer-logits`). It also references predictor models under `/code/moe/moe-predict-models`, which the public repo does not provide — [README](https://github.com/promoe-opensource/promoe/blob/main/README.md)

**Fiddler (ICLR'25)**
- efeslab/fiddler is Apache-2.0 with 267 stars. It has 49 commits from 2024-02-04 to 2024-04-28. The README says "This repository is a proof-of-concept and still under heavy construction" — [repo](https://github.com/efeslab/fiddler); [ungh.cc](https://ungh.cc/repos/efeslab/fiddler)
- The README's own benchmarks use a Quadro RTX 6000 and an L4. The ICLR'25 paper evaluates a Quadro RTX 6000 + Xeon Gold 6126 and an RTX 6000 Ada + Xeon Platinum 8480+ — [README](https://github.com/efeslab/fiddler/blob/main/README.md); [ICLR'25 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/8cd1ce03ea58b3d7dfd809e4d42f08ea-Paper-Conference.pdf)

**MoE-Infinity**
- EfficientMoE/MoE-Infinity is Apache-2.0 with 363 stars. It has 356 commits, 154 of them since 2026-06-01, and the last on 2026-09-28 — [repo](https://github.com/EfficientMoE/MoE-Infinity); [ungh.cc](https://ungh.cc/repos/EfficientMoE/MoE-Infinity)
- The README says: "This open-sourced version is HuggingFace-friendly and differs from the version reported in the paper, which prioritized extreme performance" — [README](https://github.com/EfficientMoE/MoE-Infinity/blob/main/README.md)
- The audit table has no MoE-Infinity row; it appears only as a baseline inside SeqMoE and FreeToken — [audit.md](/home/claude/moe-speed-of-light/prereg/audit/audit.md)

### Inferences
- For KTransformers, "the release matches the paper" holds only for `archive/` (the local_chat/balance_serve path with AMXInt8 rules for Qwen2-57B and DeepSeek). The actively maintained kt-kernel + SGLang path is a different code base, with newer kernels and deferred experts. A head-to-head should state which one it used. The SOSP artifact badge does not name a commit; a v0.2.4–v0.3.1 tag is the likely match, but I did not verify this.
- The FreeToken public release (first commit 2026-08-11) predates arXiv v1 (2026-08-17). Tag v0.1.2 (2026-08-19) is the closest candidate for "paper version". The repo has changed a great deal since then, so pin a tag rather than `main` or `nightly`.
- The HybriMoE repo is a frozen research snapshot: six commits and no maintainer replies visible on its issues.
- The Fiddler repo (last commit April 2024) probably predates the ICLR camera-ready experiments on the RTX 6000 Ada, because the README benchmarks list different hardware.
- The ProMoE llama.cpp integration, which is the variant behind most audited ProMoE rows, is not runnable from public sources. `ipads.se.sjtu.edu.cn` is reached over SSH with an institutional account.

### Gaps
- Exact commit/tag that SOSP AE evaluated for KTransformers is not stated on the AE page.
- Whether SeqMoE, SP-MoE or MoE-SpeQ authors plan a code release (no statement found); emailing authors is the only route.
- Whether the ProMoE internal repos are mirrored anywhere public (none found).

## Q2. How is each installed, and what hardware/software does it need?

### Takeaway
Pipelined sharding and the llama.cpp branches are plain CMake/CUDA builds; Pipelined sharding also has a GHCR image. FreeToken is a pip/uv wheel, but it **requires NVIDIA driver r580+ and a CUDA 13 toolkit**. KTransformers kt-kernel ships prebuilt wheels for SM 80/86/89/90 and needs source builds for Blackwell. Its fast paths **require AMX (Sapphire Rapids+) or AVX-512**, and the paper machine was a dual-socket AMX Xeon. HybriMoE and Fiddler are 2024–25 Python stacks with old pinned dependencies. 2512.16473 and Fiddler compute CPU experts with PyTorch BF16, so their CPU performance depends on AVX-512/BF16 support.

### Cited Findings
**KTransformers**
- `pip install kt-kernel` gives prebuilt wheels for Python 3.10–3.12 with "CUDA support included: GPU acceleration for NVIDIA GPUs (SM 80, 86, 89, 90)" and a static CUDA runtime. Requirements are a "CPU with AVX2 support (Intel Haswell 2013+, AMD Zen+)" and a driver supporting CUDA 11.8+/12.x — [kt-kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)
- Backend CPU requirements:
  - LLAMAFILE (GGUF): AVX2
  - RAWINT4: AVX512F+BW
  - AMXINT4/INT8: "AMX | Intel Sapphire Rapids (2023+) | Best performance, requires AMX hardware"
  - FP8: AVX512F+BW+BF16+VBMI
  - BF16: AVX512F+BW+BF16, on Cooper Lake/Sapphire Rapids or AMD Zen 4+
  - Source: [kt-kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)
- The kt-kernel README also describes "NUMA-Aware Execution: Thread pool and memory layout designed for multi-socket / multi-NUMA machines", and SGLang flags `--kt-method`, `--kt-weight-path`, `--kt-cpuinfer`, `--kt-threadpool-count` (= NUMA nodes), `--kt-num-gpu-experts` and `--kt-max-deferred-experts-per-token`. It requires the kvcache-ai SGLang fork (`pip install kt-kernel sglang-kt`) — [kt-kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)
- The AVX2-only backend was added 2026-03-26. It supports BF16, FP8, GPTQ_INT4 and RAWINT4 and needs "Memory: At least the size of the model weights (e.g., Qwen3-30B-A3B BF16 requires 64GB+)". It is built from source with `./install.sh` after `git submodule update --init --recursive` — [AVX2 tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/AVX2-Tutorial.md)
- Blackwell consumer GPUs appear in the docs through source builds. The DeepSeek-V4-Flash tutorial uses "1× NVIDIA RTX 5090 (32GB VRAM, SM_120)" with `export FLASHINFER_CUDA_ARCH_LIST=12.0a` — [DeepSeek-V4-Flash.md](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash.md)
- The SOSP paper hardware was dual-socket Xeon Platinum 8452Y (AMX, 36 cores/socket), 1 TB DDR5 per socket, an A100 40 GB and an RTX 4080 16 GB on PCIe 4.0. The decode speed-ups it reports come from the AVX-512 kernel (2.22× over the Fiddler base), NUMA-aware tensor parallelism (up to 1.63×) and CUDA Graph (1.23×) — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- The legacy `archive/install.sh` installs `requirements-local_chat.txt` (`torch>=2.3.0`) and builds with `KTRANSFORMERS_FORCE_BUILD=TRUE pip install -v . --no-build-isolation`. balance_serve is gated by `USE_BALANCE_SERVE=1` — [archive/install.sh](https://github.com/kvcache-ai/ktransformers/blob/main/archive/install.sh); [archive/setup.py](https://github.com/kvcache-ai/ktransformers/blob/main/archive/setup.py)

**FreeToken**
- Requirements: "Linux x86_64, NVIDIA GPU, driver r580+ (CUDA 13)", Python ≥ 3.10. Install with `uv pip install "freetoken[accel]"`, which gives PyPI wheels. "CUDA kernels are JIT-compiled on first use, need a CUDA 13 toolkit with `nvcc` on PATH". Nightly wheels include a prebuilt `freetoken-kernel-cache` — [docs/install.md](https://github.com/FlashML-org/FreeToken/blob/main/docs/install.md)
- The README claims "native support for NVIDIA RTX 30, RTX 40, and RTX 50 series GPUs"; an AMD ROCm guide is WIP — [README](https://github.com/FlashML-org/FreeToken/blob/main/README.md)
- The CPU-expert kernel dispatches between ISA tiers "scalar", "avx2", "avx512" and "avx512bf16". The code notes "3 real kernels: scalar/avx2/avx512, no bf16/VNNI variant" — [python/freetoken/moe/benchbw.py](https://github.com/FlashML-org/FreeToken/blob/main/python/freetoken/moe/benchbw.py)
- Key flags:
  - `--moe-strategy {auto,fused,offload,cpu,hybrid}` ("hybrid — per step, fetches some misses over PCIe and computes the rest on CPU… Run `ft bench bw` once per machine to calibrate")
  - `--moe-cache-size / --moe-cache-rate / --moe-cache-auto` (GPU expert-cache slots)
  - `--moe-cpu-threads` (default: physical cores)
  - `--memory-ratio 0.9`
  - Sources: [docs/models.md](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md); [docs/cli.md](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md)
- `benchmarks/bench_decode_moe.py` measures "bs=1 decode tok/s of a served MoE model", e.g. `--backend offload,cpu,hybrid`, with an AIME-25 prompt and the full serving path — [benchmarks/README.md](https://github.com/FlashML-org/FreeToken/blob/main/benchmarks/README.md)
- The paper's 3090, 4090 and 5090 systems "are rented dual-socket servers … every serving run and bandwidth measurement on them is capped at 6 CPU threads and pinned to the GPU's NUMA node". The machines were:
  - 5090 box: 2× Xeon Gold 6459C, DDR5, 180 GiB quota, B_H 77.3 GB/s
  - 4090 box: 2× Xeon Platinum 8358P, DDR4, 240 GiB, B_H 63.2 GB/s
  - desktop: 5090 + Ryzen 9 9950X3D, 192 GB
  - laptop: 4060 Laptop, 32 GB
  - Source: [arXiv HTML, §5.1/Table 1](https://arxiv.org/html/2608.16157)
- A community user on a 5090 with 128 GB RAM installed FreeToken 0.1.2 from source with Python 3.12 and CUDA 13.0. They had to set `CUDA_HOME` explicitly — [zenn.dev, 22 Aug 2026](https://zenn.dev/holy_fox/articles/53b82eed45f956?locale=en)

**HybriMoE**
- Needs "CUDA 12.1 and above" and conda Python 3.11. Setup is `git submodule init && git submodule update` (pybind11 and llama.cpp), then `bash install.sh`. The Dockerfile base is `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel` — [README](https://github.com/PKU-SEC-Lab/HybriMoE/blob/main/README.md); [Dockerfile](https://github.com/PKU-SEC-Lab/HybriMoE/blob/main/Dockerfile)
- Run with `python ktransformers/local_chat.py --model_path deepseek-ai/DeepSeek-V2-Lite-Chat --gguf_path ./DeepSeek-V2-Lite-Chat-GGUF --cache_size 16 --prefetch_size 0 --optimize_rule_path …/DeepSeek-V2-Chat-gpu.yaml` — [README](https://github.com/PKU-SEC-Lab/HybriMoE/blob/main/README.md)

**2512.16473 (elsa-lab)**
- Requirements:
  - GPU: "Consumer-grade GPU (RTX 3080/4090 series)", 16 GB+ VRAM
  - CPU: 8+ cores
  - Software: Python 3.8+, PyTorch 2.0+, CUDA 11.8+
  - Host memory: "CPU with 128GB+ main memory for optimal performance"
  - Steps: `pip install … transformers safetensors mistral-common xformers nvtx-plugins`, then `weights_preprocessor.py`
  - Source: [README](https://github.com/elsa-lab/MoE-CPU-GPU-Collaborative-Inference/blob/main/README.md)
- CPU experts run as PyTorch BF16 matmuls (`nn.functional.silu(x_cpu @ w[0].T) * (x_cpu @ w[2].T)) @ w[1]`, `dtype=torch.bfloat16`) — [src/model.py](https://github.com/elsa-lab/MoE-CPU-GPU-Collaborative-Inference/blob/main/src/model.py)
- The paper's motivation table uses an AMD Ryzen Threadripper 7960X with an RTX 4090 — [arXiv HTML](https://arxiv.org/html/2512.16473)

**Pipelined sharding**
- Requirements: "An x86_64 machine with an NVIDIA RTX (ideally, an RTX 5090 or 5070 TI) or any A100 or newer compute class GPU", CUDA Toolkit 12.8+ — [README](https://github.com/deepshnv/pipeshard-mlsys26-ae/blob/main/README.md)
- Three install routes — [README](https://github.com/deepshnv/pipeshard-mlsys26-ae/blob/main/README.md):
  - (A) source: `cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF`
  - (B) prebuilt binaries: the README shows Windows x64 CUDA 12.9; the paper says Windows and Ubuntu 24.04 x64
  - (C) Docker: `docker pull ghcr.io/deepshnv/pipeshard-mlsys26-ae:v1.0.0`, which runs all 5 repro scripts
- Enable with `GGML_CUDA_PIPELINE_SHARDING=1 GGML_CUDA_REGISTER_HOST=1` and `-mva <vram_mb> -pipe-shard`. Set CPU threads with `PIPESHARD_THREADS`. The README says: "All development and testing for our paper was done on Windows; we recommend Windows for the smoothest reproduction experience" — [README](https://github.com/deepshnv/pipeshard-mlsys26-ae/blob/main/README.md)
- Appendix figures: "~175 GB for all model weights", "~20 minutes to build from source; 1–3 hours to download all models" — [arXiv HTML, Appendix A](https://arxiv.org/html/2604.26334)

**Fiddler**
- Pins `torch==2.1.2`, `transformers==4.36.2` and `accelerate==0.26.1`. It "only supports a 16-bit Mixtral-8x7B model" and "is slow if your CPU does not support AVX512" — [requirements.txt](https://github.com/efeslab/fiddler/blob/main/requirements.txt); [README](https://github.com/efeslab/fiddler/blob/main/README.md)

**MoE-Infinity**
- Install with `pip install moe-infinity`. A from-source build targets sm_80/sm_90 by default; "for Blackwell (sm_120, e.g. RTX PRO 6000 / RTX 50-series) build with `MOE_ENABLE_SM120=1`". The cache size is set by `device_memory_ratio` — [README](https://github.com/EfficientMoE/MoE-Infinity/blob/main/README.md)

**ProMoE**
- Docker base is `nvidia/cuda:12.4.1-devel-ubuntu22.04`. The build needs the private repos listed in Q1 — [docker/Dockerfile.llama.cpp](https://github.com/promoe-opensource/promoe/blob/main/docker/Dockerfile.llama.cpp); [install.md](https://github.com/promoe-opensource/promoe/blob/main/install.md)

**Rental market**
- 1× RTX 5090 cloud offers (page dated 2026-09-28) range from about $0.35/h to $1.49/h. Bundled RAM varies widely, from 8 GB to 144 GB. Examples — [getdeploying RTX 5090](https://getdeploying.com/gpus/nvidia-rtx-5090):
  - Vast.ai: $0.41/h, 63 GB, 24 vCPU
  - GPUhub: $0.46/h, 90 GB
  - Runpod: $0.69/h, 35 GB
  - Runcrate/Sesterce: ~$0.71–0.72/h, 120 GB
  - Cloudzy: $1.49/h, 144 GB
- Vast.ai's search-result titles advertise RTX 5090 "from $0.27/hr" and RTX 4090 "from $0.14/hr". The 5090 page itself showed "No current offers" when fetched — [Vast.ai 5090](https://vast.ai/pricing/gpu/RTX-5090); [Vast.ai 4090](https://vast.ai/pricing/gpu/RTX-4090)

### Inferences
- **The binding constraints for a rental are CPU ISA, host RAM and driver version, not the GPU.** KTransformers' paper-level decode path needs AVX-512, and AMX for its Int8 kernels. Most rented consumer-GPU hosts use AMD EPYC (Zen2/Zen3: no AVX-512) or older Xeons. On those, kt-kernel falls back to the AVX2/llamafile backend, which is essentially ggml-class kernels. A head-to-head there would not test the SOSP claim. To test it, filter for a Sapphire Rapids or newer Xeon host. The CPUs that SeqMoE and FreeToken themselves rented qualify; per public Intel specs, Xeon Gold 6430, Xeon Platinum 8470Q and Xeon Gold 6459C are all 4th-gen Xeon Scalable (Sapphire Rapids) with AMX. Zen 4/5 EPYC offers AVX-512+BF16 but not AMX. The ISA facts are from Intel/AMD specs, not from the fetched sources.
- FreeToken's r580+/CUDA 13 requirement excludes hosts whose driver is older. Container marketplaces such as Vast and Runpod do not let you upgrade the host driver, so filter listings by maximum CUDA version ≥ 13.0.
- Container rentals usually do not allow Docker-in-Docker, so Pipelined sharding's GHCR image is only useful on VM-type rentals. Its source build is a normal llama.cpp CMake build (~20 min) and is the safer route on Linux.
- Single-socket rentals remove KTransformers' NUMA-aware tensor-parallel gain (up to 1.63× on the dual-socket paper machine). Conversely, FreeToken pinned its runs to 6 threads on one NUMA node, which is easy to emulate on any rental with `taskset`/`numactl`.
- Many cheap 5090 offers bundle only 35–90 GB of RAM. BF16 Qwen3.6-35B-A3B, BF16 Mixtral, Phi-3.5-MoE and Qwen3-235B Q2_K all need a ≥128 GB listing, which costs about $0.7–1.5/h. The model sizes below are my estimates from parameter counts, except where cited.

### Gaps
- I could not verify from a primary source which CPU models typical Vast.ai or Runpod 4090/5090 hosts use; the listings are dynamic. Check `lscpu | grep -o 'amx\w*\|avx512\w*'` on the rented box before committing.
- Whether kt-kernel's prebuilt wheels actually run on SM_120 (the README lists SM 80/86/89/90 only), or whether a source build is always needed for 5090.
- Whether HybriMoE's 2025 stack (torch 2.5.1/CUDA 12.1 base, flashinfer) builds cleanly on current rental images. No success or failure report was found.

## Q3. Which models and quantizations work out of the box, and which overlap the audited rows?

### Takeaway
Out-of-the-box overlap with audited rows is strong for three systems:
- **Pipelined sharding:** exact models, scripts and quants for its RTX 5090 rows.
- **FreeToken:** Qwen3.6-35B-A3B BF16 on 4090/5090 is listed as known-good.
- **HybriMoE:** DeepSeek-V2-Lite-Chat Q4_K_M is the README example.

Overlap is partial or needs work elsewhere:
- **KTransformers:** Qwen2-57B-A14B only via the archived legacy path. DeepSeek-V2.5 and V3 exceed a 128 GB box.
- **2512.16473:** Mixtral-8x7B and Phi-3.5-MoE in BF16, which needs a ≥128 GB host.
- **HybriMoE's Mixtral/Qwen2 rows:** its released rules do not enable the HybriMoE cache for those models.

### Cited Findings
**Audited rows (from [audit.md](/home/claude/moe-speed-of-light/prereg/audit/audit.md))**
- KTransformers rows all have a 0.0 GB GPU expert budget, i.e. all routed experts on the CPU:

| Row | KT tok/s | Paper llama.cpp | Model-predicted llama.cpp | S_n | Label |
|---|---|---|---|---|---|
| Qwen2-57B-A14B, RTX 4080 | 37.0 | 19.2 | 19.2 | 1.93 | gain survives |
| Qwen2-57B-A14B, A100 40GB | 22.9 | 13.0 | 15.5 | 1.47 | gain survives |
| DeepSeek-V2.5, RTX 4080 | 17.6 | 9.9 | — | 1.42 | gain survives |
| DeepSeek-V3, A100 | — | — | — | — | at strength; not established |
| DeepSeek-V2.5, A100 | — | — | — | — | weak baseline; not established |
| DeepSeek-V3, RTX 4080 | — | — | — | — | weak baseline; not established |

- The paper used "Int4 (V3) or Int8 (V2.5, Qwen2) on the 4080" and BF16/FP16 on the A100 — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- HybriMoE rows (RTX A6000). All are "weak baseline; not established" except Mixtral at 17.4 GB, which is not adjudicated (<1.2×):

| Model | GPU budget | HybriMoE tok/s | Paper llama.cpp | Model-predicted llama.cpp |
|---|---|---|---|---|
| DeepSeek-V2-Lite-Chat | 1.9 GB | 14.5 | 11.7 | **80.9 [60.8, 109.7]** |
| DeepSeek-V2-Lite-Chat | 5.6 GB | 15.3 | 8.9 | **140.0** |
| Mixtral-8x7B | 5.8 GB | 4.4 | 3.5 | 14.6 |
| Qwen2-57B | 6.4 GB | 9.6 | 7.4 | 19.1 |

- SeqMoE rows:

| Model | GPU | Budget | SeqMoE | Paper llama.cpp | Predicted | S_n | Label |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-FP8 | RTX 4090 | 13.1 GB | 104.1 | 30.9 | 61.7 | 1.69 | gain survives |
| Qwen3-30B-A3B-FP8 | RTX 4090 | 4.4 GB | 47.9 | 21.1 | 48.9 | 0.98 | — |
| Qwen3.6-35B-A3B-BF16 | RTX 5090 | 25.8 GB | 118.3 | 37.9 | 56.4 | 2.10 | — |
| Qwen3.6-35B-A3B-BF16 | RTX 5090 | 9.7 GB | 70.2 | 28.2 | 46.5 | — | — |
| GPT-OSS-120B | RTX PRO 6000 | — | 139.2 | 33.1 | 74.3 | — | — |
| DeepSeek-V4-Flash | RTX PRO 6000 | — | 57.4 | 21.0 | 38.3 | — | — |

- FreeToken rows. All are "not adjudicated: band wider than ±40%":

| Model | GPU | Budget | FreeToken | Paper llama.cpp | Predicted | S_n |
|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | RTX 5090 | 23.8 GB | 77.1 | 42.6 | 35.1 [24.8, 51.3] | 2.19 [1.50, 3.11] |
| Qwen3.6-35B-A3B | RTX 5090 | 26.3 GB | 73.8 | 33.0 | 19.1 | — |
| Qwen3.6-35B-A3B | RTX 4090 | 18.3 GB | 42.9 | 25.8 | 21.9 | — |
| Qwen3.6-35B-A3B (NVFP4) | RTX 4060 Laptop | — | — | — | — | — |
| DeepSeek-V4-Flash | RTX 5090 | — | — | — | — | — |
| GLM-5.2 | RTX PRO 6000 | — | — | — | — | — |

- Pipelined sharding rows (RTX 5090):

| Model | Budget | Pipeshard | Paper llama.cpp | Predicted | S_n | Label |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Instruct-2507 | 0.0 GB | 25.7 | 25.7 (`-ngl + -cmoe`) | **49.5 [38.4, 65.1]** | — | not adjudicated (<1.2×) |
| Qwen3-30B-A3B-Instruct-2507 | 5.7 GB | 32.1 | 26.1 | 69.7 | 0.46 | weak baseline |
| Qwen3-235B-A22B | 0.0 GB | 7.7 | 5.7 (`-ngl only`) | 13.1 | — | weak baseline |
| Qwen3-235B-A22B | 27.6 GB | 11.5 | 8.5 | 18.2 | — | weak baseline |
| Qwen3-30B-A3B-Instruct-2507 (RTX 5070 Ti) | 13.7 GB | 54.9 | none reported | 90.5 | — | — |

- 2512.16473 rows (RTX 4090, 19.7 GB):
  - Mixtral: 4.8 vs its own CPU-only mode 4.2; predicted llama.cpp 4.9
  - Phi-3.5-MoE: 10.4 vs 6.3; predicted llama.cpp 10.3, S_n 1.01 [0.74, 1.34], "not established"
- llama.cpp forks:
  - Laguna S 2.1 on RTX 5090: 31.3 vs Leloch fork 27.6, S_n 2.37 [1.58, 3.44], not adjudicated
  - PR #27861 on RX 7600: 16.5 vs cache-off 14.4, not adjudicated (<1.2×)
- The remaining rows for Fiddler, ProMoE, SP-MoE and MoE-SpeQ are all "not adjudicated: band wider than ±40%" except SP-MoE DeepSeek-V2-Lite, which is "not established" — [audit.md](/home/claude/moe-speed-of-light/prereg/audit/audit.md)

**What each system supports out of the box**
- Pipelined sharding's `download_models.sh` fetches `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF` **Q4_0** and `unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF` **Q2_K**. It also fetches nemo-4b/8b and cosmos-reason1. `paper_results/` contains `repro_table4.sh`, `repro_table8.sh`, `repro_table9.sh`, `repro_figure2.sh`, `repro_figure7.sh` and `compare_*.py` — [download_models.sh](https://github.com/deepshnv/pipeshard-mlsys26-ae/blob/main/download_models.sh); [paper_results](https://github.com/deepshnv/pipeshard-mlsys26-ae/tree/main/paper_results)
- The Pipelined sharding fork's `common/arg.cpp` includes `--cpu-moe/-cmoe`, `--n-cpu-moe/-ncmoe` and `--override-tensor/-ot`. The same binary therefore gives the properly configured llama.cpp baseline — [common/arg.cpp](https://github.com/deepshnv/pipeshard-mlsys26-ae/blob/main/common/arg.cpp)
- FreeToken lists as known-good, among others:
  - Qwen3.6-35B-A3B (BF16, -FP8, nvidia NVFP4)
  - Qwen3-30B-A3B
  - gpt-oss-120b/20b
  - DeepSeek-V4-Flash-0731
  - GLM-5.2-NVFP4
  - "FreeToken loads HF safetensors checkpoints directly" — [docs/models.md](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md)
- The FreeToken paper served "Qwen3.6-35B-A3B … in BF16, which gives exact precision parity across engines" and "every engine consumes DSV4-Flash's native MXFP4 expert blocks bit-exactly" — [arXiv HTML](https://arxiv.org/html/2608.16157)
- KTransformers' documented kt-kernel example is Qwen3-30B-A3B in BF16 native, AMX or LLAMAFILE modes. The example uses `--kt-num-gpu-experts 32` on a 24 GB GPU and 2× Xeon Gold 6454S — [kt-kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)
- The legacy archive rules include `Qwen2-57B-A14B-Instruct.yaml`, `Qwen2-serve-amx.yaml` (AMXInt8/AMXBF16/llamafile), `Mixtral.yaml`, `DeepSeek-V2-Lite-Chat*.yaml` and `DeepSeek-V3-Chat-amx.yaml` — [archive optimize_rules](https://github.com/kvcache-ai/ktransformers/tree/main/archive/ktransformers/optimize/optimize_rules)
- In the current kt-kernel tree, "Qwen2Moe" appears only in the SFT architecture list (`sft/arch.py`, `sft/artifacts.py`), not in the inference docs — [kt-kernel/python/sft/arch.py](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/python/sft/arch.py)
- HybriMoE's README downloads `mzwing/DeepSeek-V2-Lite-Chat-GGUF` **Q4_K_M** — [README](https://github.com/PKU-SEC-Lab/HybriMoE/blob/main/README.md)
- HybriMoE's optimize rules:
  - Only `DeepSeek-V2-Chat-gpu.yaml` sets `generate_op: "KExpertsMarlin"`. That is the class holding `KExpertsCache`, `load_size`/`cache_size`, `prefetch_expert` and `KScoreAwareCache`.
  - `Mixtral.yaml`, `Qwen2-57B-A14B-Instruct.yaml`, `DeepSeek-V2-Lite-Chat.yaml` and `DeepSeek-V2-Chat.yaml` use stock `generate_op: "KExpertsCPU"`.
  - Sources: [optimize_rules](https://github.com/PKU-SEC-Lab/HybriMoE/tree/main/ktransformers/optimize/optimize_rules); [operators/experts.py](https://github.com/PKU-SEC-Lab/HybriMoE/blob/main/ktransformers/operators/experts.py)
- 2512.16473 lists "Mixtral 8x7B … 88 GB" and "Phi-3.5-MoE … 79 GB" as supported models. It recommends `--cache-nblocks 14 --cache-nways 4 --cache-replace-policy LRU` for an RTX 4090 with Mixtral — [README](https://github.com/elsa-lab/MoE-CPU-GPU-Collaborative-Inference/blob/main/README.md)
- MoE-Infinity supports DeepSeek-V2-Lite-Chat, DeepSeek-V3, DeepSeek-V4-Flash, Mixtral-8x7B/8x22B, Qwen3-30B-A3B and Qwen3.5-35B-A3B — [README](https://github.com/EfficientMoE/MoE-Infinity/blob/main/README.md)
- SeqMoE's platforms were:
  - RTX 4090 + Xeon Gold 6430 + 120 GB (PCIe 4.0)
  - RTX 5090 + Xeon Platinum 8470Q + 120 GB (PCIe 5.0)
  - RTX PRO 6000 + 8470Q + 256 GB
  - Source: [arXiv HTML §9.1](https://arxiv.org/html/2609.12978)

### Inferences
- Approximate host-RAM needs, from parameter counts, to fit the model plus OS/page cache:
  - DeepSeek-V2-Lite Q4_K_M: ~10 GB
  - Qwen3-30B-A3B Q4_0: ~17 GB
  - Qwen3-30B-A3B FP8/Q8_0: ~31 GB
  - Qwen2-57B-A14B Int8/Q8_0: ~60 GB
  - Qwen3.6-35B-A3B BF16: ~70 GB
  - Phi-3.5-MoE BF16: ~79–84 GB
  - Mixtral BF16: ~88–93 GB
  - Qwen3-235B Q2_K: ~86 GB
  - Qwen2-57B BF16: ~114 GB
  - DeepSeek-V2.5 Int8: ~236 GB
  - DeepSeek-V3 Int4: ~350 GB+
  - Consequence: on a 128 GB box, only the Qwen2-57B-A14B **Int8 / RTX 4080** KTransformers row is reachable. DeepSeek-V2.5, V3 and the Qwen2-57B BF16 A100 row are out of reach.
- The KTransformers Qwen2-57B/4080 row uses a 0 GB GPU expert budget. The "same GPU memory" constraint is therefore trivial; the test compares CPU kernels plus scheduling against `llama.cpp -ngl 99 --cpu-moe` with Q8_0. A 4090 standing in for the 4080 changes little for this row because the experts all run on the CPU. The host CPU and memory bandwidth dominate.
- The HybriMoE DeepSeek-V2-Lite rows are the cheapest high-contrast test in the whole audit. The model predicts llama.cpp at 81–140 tok/s against HybriMoE's reported 14.5–15.3 tok/s, a 5–9× gap. The model is ~10 GB Q4_K_M, so a 4090 host with only 32–64 GB RAM suffices. HybriMoE's `--cache_size` (experts per layer) must be mapped to the 1.9/5.6 GB budgets.
- Reproducing HybriMoE's Mixtral or Qwen2-57B rows requires writing new optimize rules that route those models' experts to `KExpertsMarlin`. Those rows cannot be tested from the release as-is.
- For SeqMoE, which has no code, the rows can still be half-tested. One can measure llama.cpp `--n-cpu-moe` on a matching rental (RTX 4090 + Sapphire Rapids Xeon, Qwen3-30B-A3B Q8_0 as the nearest GGUF to FP8) and compare it with the model's predicted baseline of 61.7 tok/s and the paper's 30.9. This tests the model, not SeqMoE.
- **The Pipelined sharding 0 GB row already contradicts the model.** The paper's llama.cpp (`-ngl + -cmoe`, which is the properly configured all-experts-on-CPU baseline) measured 25.7 tok/s against a prediction of 49.5 [38.4, 65.1]. That is measured/predicted ≈ 0.52, far outside the 0.78–1.31 validation band. Two readings are possible: the model over-predicts llama.cpp on this platform (a 16-core EPYC desktop), or the paper's host was bandwidth-starved. Re-measuring this row on a rented 5090 costs about 1 GPU-hour, and it is the most decisive single measurement for the "weak baseline" labels. The same direction appears in the SeqMoE 4.4 GB row (21.1 measured vs 48.9 predicted).

### Gaps
- Whether kt-kernel + sglang-kt supports Qwen2MoeForCausalLM for inference (only SFT lists it); needs a test or maintainer confirmation.
- Whether llama.cpp's FP8 handling makes a like-for-like Qwen3-30B-A3B-FP8 comparison possible; Q8_0 is a proxy.
- Exact llama.cpp flags used by the FreeToken and SeqMoE baselines were not found in the passages checked. FreeToken's related-work text says llama.cpp "assign[s] whole layers to devices statically at load time" — [arXiv HTML](https://arxiv.org/html/2608.16157)

## Q4. Known issues (install failures, regressions, hardware incompatibilities)

### Takeaway
Real, documented pitfalls exist for three stacks. KTransformers segfaults on non-NUMA desktops (fix acknowledged), has SM_120 gaps in some FP8 paths, and requires its SGLang fork. FreeToken's issues cluster around driver/GPU detection, packaged-engine dependency hash mismatches and VRAM floors on small GPUs. HybriMoE has unanswered issues on Marlin weights and flashinfer. For the llama.cpp cache branches, reviewers found correctness hazards in #27861, and leloch's cache regressed on Pascal.

### Cited Findings
**KTransformers**
- #1754, "Single Ryzen CPU error": a Ryzen 9950X3D + RTX 5090 + 64 GB + CUDA 13.1 box running kt-kernel/SGLang with Qwen3-30B segfaults in `numa_bitmask_setbit` ("request to allocate mask for invalid number"). kt-kernel "did not properly handle systems without NUMA". The maintainer replied: "I will update this to consider the non-numa machine" — [issue #1754](https://github.com/kvcache-ai/ktransformers/issues/1754)
- #2081: `[MiniMax-M3-MXFP8] fp8.py:788 assert is_sm90 or is_sm100 blocks SM_120 (GB203, RTX PRO 4500)` — [issue #2081](https://github.com/kvcache-ai/ktransformers/issues/2081)
- #1785, "SGLang does not support kt-kernel", and #1885, "Capture cuda graph failed: Assertion error when Running DeepSeek V3.2 with SGLang and KT-Kernel" (titles only) — [issue #1785](https://github.com/kvcache-ai/ktransformers/issues/1785); [issue #1885](https://github.com/kvcache-ai/ktransformers/issues/1885)
- A maintained FAQ issue exists — [issue #1608](https://github.com/kvcache-ai/ktransformers/issues/1608)
- The repo has ~467 open issues — [shields.io badge](https://img.shields.io/github/issues/kvcache-ai/ktransformers)
- A vendor comparison on 2× RTX 5090 + AMD EPYC 9355 claims "up to >4.5x prefill and 30% faster decode" versus llama.cpp, with the caveat "We made our best effort to optimize llama.cpp performance" — [MiniMax-M2.1 tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/MiniMax-M2.1-Tutorial.md)

**FreeToken**
- #554: "Linux engine install fails on beta update (sglang-kernel hash mismatch, flashinfer version mismatch) and leaves no engine installed" — [issue #554](https://github.com/FlashML-org/FreeToken/issues/554)
- #312: "Linux AppImage GPU detection fails when only libnvidia-ml.so.1 is installed" — [issue #312](https://github.com/FlashML-org/FreeToken/issues/312)
- #276: "4070 ti Super - No NVIDIA GPU or driver detected" — [issue #276](https://github.com/FlashML-org/FreeToken/issues/276)
- #409: "Qwen3.8-Flash-Next-NVFP4 cannot start on a 12 GB RTX 3060" — [issue #409](https://github.com/FlashML-org/FreeToken/issues/409)
- #14: "Please help support older hardware (1080/2080 series)" — [issue #14](https://github.com/FlashML-org/FreeToken/issues/14)
- The docs describe an FTW checkpoint-format break ("FTW files converted by builds before the quantization refactor may fail to load") and a repair tool — [docs/models.md](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md)
- The repo has ~204 open issues — [shields.io badge](https://img.shields.io/github/issues/FlashML-org/FreeToken)
- Community result on an RTX 5090 with a model that fits in VRAM (Ornith 1.5 NVFP4, 23.5 GB): single-request FreeToken 196.9 tok/s vs llama.cpp Q4_K_M 296.4 tok/s. With 16 concurrent requests FreeToken led, 77.8 vs 46.2. gpt-oss-120b on FreeToken ran at 127.1 tok/s. The llama.cpp flags were not documented, and the verdict was "not a universal inference engine that replaces vLLM or llama.cpp" — [zenn.dev](https://zenn.dev/holy_fox/articles/53b82eed45f956?locale=en)
- A third-party review explicitly "did not reproduce the paper's hardware benchmarks" — [Wavect review](https://wavect.io/blog/freetoken-ai-inference-engine-review/)

**HybriMoE**
- There are three open issues and no maintainer replies are visible — [issues](https://github.com/PKU-SEC-Lab/HybriMoE/issues):
  - #14 "显存未释放" (GPU memory not released), 2026-04-05
  - #8 "marlin_q_w None" in `gptq_marlin_gemm`, 2025-08-09
  - #7 "flashinfer 无法开启" (flashinfer cannot be enabled), 2025-07-30
- #8 concerns the Marlin expert path, which is the HybriMoE cache path; `b_q_weight` arrives as `None` — [issue #8](https://github.com/PKU-SEC-Lab/HybriMoE/issues/8)

**llama.cpp community branches**
- Reviewers of #27861 raised "Duplicate slot IDs break CUDA batched kernels for n_tokens > 1", a "CPU/GPU table version inconsistency during prefill", and "synchronization hazards between graph execution and cache updates" — [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861)
- The leloch cache regressed on a GTX 1080 Ti (−3% to −31%). v2 gates `auto` mode on compute capability ≥ 8.0 — [Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)

**Pipelined sharding**
- The README warns that a missing `libllama.so` on Linux requires `LD_LIBRARY_PATH`, and that very large image resolutions can hit a 2 GB tensor assertion (VLM path only). Recent commits fix Linux compilation and download scripts ("changes for successful linux compilation", 2026-03-07) — [README](https://github.com/deepshnv/pipeshard-mlsys26-ae/blob/main/README.md); [commits](https://github.com/deepshnv/pipeshard-mlsys26-ae/commits/main)

### Inferences
- On a rented single-socket container with no NUMA exposed, KTransformers may hit the #1754 segfault unless the fix has landed. Check `numactl -H` and try the latest kt-kernel first.
- FreeToken's issues concern the desktop app, AppImage and small-VRAM cases rather than the CLI on a 4090/5090. The main risk on rentals is the driver floor.

### Gaps
- GitHub issue search was robots-blocked, so these issue lists are samples found through web search, not exhaustive.
- No independent report was found of anyone running HybriMoE, 2512.16473 or Pipelined sharding outside the authors.

## Q5. llama.cpp community expert-cache work: merged or not, runnable from a branch?

### Takeaway
None is merged into llama.cpp master as of 2026-09-27. All three can be built from public refs with the standard `cmake -DGGML_CUDA=ON` flow. The cache can then be switched on and off in the same binary as `--n-cpu-moe`, which makes these the cleanest A/B tests available, though none is an audited "gain survives" row.

### Cited Findings
- llama.cpp master head is 2026-09-27 (#28876) (git fetch) — [llama.cpp commits](https://github.com/ggml-org/llama.cpp/commits/master)
- **PR #26563 (miltos22):**
  - Status: "Closed (with plans to reopen)". Quote: "This has been closed with plans to organize and re-open. I have redesigned about half of the entire system in ways that fix all major issues."
  - Flags: `-ehs N/--expert-hot-s N` and `-ecf`
  - Reported speed-ups range from 0.84× to 2.07×. A maintainer suggested creating an issue explaining the architecture before splitting it into multiple PRs.
  - Source: [PR #26563](https://github.com/ggml-org/llama.cpp/pull/26563)
  - The PR head ref (`refs/pull/26563/head`) was last committed 2026-08-06 (git fetch) — [PR #26563](https://github.com/ggml-org/llama.cpp/pull/26563)
- **PR #27861 (csantiago78):**
  - Draft, opened 2026-08-28, last activity 2026-09-16
  - Flags: `--moe-expert-cache N` (slots per layer) and `--moe-expert-cache-inserts`
  - Tested on CUDA, HIP and Vulkan (3090, RTX 5090, 7900 XTX, RX 7600, RX 9070 XT)
  - Reported results: Qwen3.8-Flash-Next on 2× 3090, "18.4 -> 24.2 tok/s (+31%)"; RTX PRO 4500, "16.83 -> 41.31 tok/s (tuned settings)"
  - Source: [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861)
  - GitHub still exposes a `refs/pull/27861/merge` ref, i.e. the PR is open. Its head commit is 2026-08-28 (git ls-remote/fetch) — [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861)
- **leloch (PR #24524 → Discussion #24528):**
  - RFC posted 2026-06-12; the PR was closed and converted to a discussion
  - Primary branch: `leloch/llama.cpp:moe-cache-v2-pr`, "rebased onto current master as of August 6, 2026"
  - Flags: `--moe-cache auto|off|<MB>` plus env vars `GGML_CUDA_MOE_CACHE_ADMIT_AFTER` and `GGML_CUDA_MOE_CACHE_MAX_BATCH`. It is CUDA-only.
  - Source: [Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)
  - Branches in leloch's fork: `moe-cache-v2-pr` (head 2026-08-06, "cuda: pack MoE cache dispatch inputs"), `v3-expert-cache` (2026-06-11) and `moe-cache` (2026-06-12) (git ls-remote) — [leloch/llama.cpp](https://github.com/leloch/llama.cpp)
- The discussion reports an RTX 5090 single-GPU DeepSeek-V4 IQ3_XXS decode gain of +63%, measured against disk-backed streaming, and v2 ablations on sm86. It also warns that "Cross-engine comparisons systematically flatter baseline by ~4%" — [Discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)

### Inferences
- Setup effort is about 1–2 hours per branch on a rented box: clone, build (~10–20 min) and a `llama-bench` sweep over cache on/off × `--n-cpu-moe`.
- The audited "forks" row used Laguna S 2.1 on a 5090. Re-running with a more common model (Qwen3-30B-A3B or Qwen3.6-35B-A3B GGUF) gives a clean same-binary A/B at matched VRAM.
- Because baseline and treatment share one binary, these branches are the best way to test the model's prediction of dynamic-cache speed-ups without confounds from engine differences.

### Gaps
- Whether miltos22's promised re-opened PR exists yet (none identified).
- Exact PR #26563 closing date.

## Q6. Realistic setup effort on a rented RTX 4090/5090 with 64–128 GB RAM, and whether a head-to-head there tests the audited claim; ranking by value and effort

### Takeaway
Within $100 and a few engineering days, rank the systems as follows:

1. **Pipelined sharding.** Highest value per hour: a paper-matched artifact, the same binary as the llama.cpp baseline, the same 5090 GPU as the audited rows, and it directly probes the model's llama.cpp prediction.
2. **HybriMoE on DeepSeek-V2-Lite.** Cheap, and the model predicts a 5–9× discrepancy.
3. **FreeToken on Qwen3.6-35B-A3B BF16 (5090/4090).** Would adjudicate rows the model cannot, but needs a CUDA-13-driver host with ≥128 GB RAM.
4. **KTransformers Qwen2-57B Int8.** The only "gain survives" system, but a faithful test needs a Sapphire Rapids AMX host and the legacy code path.
5. **2512.16473 (Phi-3.5-MoE/Mixtral BF16).**
6. **llama.cpp cache branches.**
7. **Fiddler.**

Not runnable: SeqMoE, SP-MoE, MoE-SpeQ and ProMoE (incomplete release). For SeqMoE, only a baseline-only check is possible.

### Cited Findings
- Pipelined sharding's appendix estimates "~20 minutes to build from source; 1–3 hours to download all models" and ~175 GB of disk for all models. It ships a PASS/FAIL comparison: "Our validation scripts will print PASS (or not) along with error margins" — [arXiv HTML, Appendix A](https://arxiv.org/html/2604.26334)
- The Pipelined sharding paper's 5090 platform was a desktop with a 16-core EPYC on PCIe Gen5 — [arXiv HTML](https://arxiv.org/html/2604.26334)
- FreeToken's own rented-server methodology (6 threads, pinned to the GPU's NUMA node) is described in the paper, so a rental can emulate it — [arXiv HTML §5.1](https://arxiv.org/html/2608.16157)
- The KTransformers AVX2 tutorial states "Memory: At least the size of the model weights", and the kt-kernel README states that AMX INT8 requires Sapphire Rapids — [AVX2 tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/AVX2-Tutorial.md); [kt-kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)
- 1× RTX 5090 rentals run from $0.35 to $0.72/h for offers with 40–120 GB RAM. Offers with 120 GB+ start around $0.71/h — [getdeploying](https://getdeploying.com/gpus/nvidia-rtx-5090)

### Inferences
Effort estimates are my judgment from the install docs above, assuming a Linux container with CUDA preinstalled. Model downloads dominate wall-clock time.

**1. Pipelined sharding: value very high, effort ~2–4 h, cost ~$3–6**
- Target the 5090 rows: Qwen3-30B Q4_0 at 0 GB and 5.7 GB; Qwen3-235B Q2_K at 0 GB and 27.6 GB on a 128 GB host.
- Run `-pipe-shard -mva <MB>`, then the same binary with `-ngl 99 --n-cpu-moe k` at the same VRAM. Add a current llama.cpp master build as a second baseline, because b6097 dates from Aug 2025.
- This directly tests the "weak baseline" labels and the 0.52 measured/predicted anomaly at 0 GB.
- A 4090 host can substitute for the 30B rows only if the paper's budgets fit in 24 GB, which they do (5.7 GB). The 5070 Ti row has no baseline, so it is the model's own prediction test.

**2. HybriMoE: value high, effort ~4–8 h, cost ~$3–8**
- Use DeepSeek-V2-Lite-Chat Q4_K_M on a 4090 or an RTX A6000 (the paper's GPU); 32–64 GB RAM suffices.
- Build risk comes from the 2025 stack (torch 2.5.1/CUDA 12.1 base, flashinfer issue #7, Marlin issue #8). Avoid a 5090, since SM_120 is unlikely to compile with that stack.
- Map `--cache_size` (experts per layer) to the 1.9 GB and 5.6 GB budgets, then run llama.cpp `-ngl 99 --n-cpu-moe k` with the same GGUF.
- A predicted 5–9× gap makes the outcome robust even to large setup noise.
- The Mixtral and Qwen2-57B rows need custom optimize rules (+0.5–1 day), because the release uses stock KExpertsCPU for them.

**3. FreeToken: value high, effort ~4–8 h, cost ~$10–20**
- Use Qwen3.6-35B-A3B BF16 (~70 GB) on a 5090 host with 128 GB and a driver ≥ r580.
- Pin tag v0.1.2 (closest to the paper) and also `main`. Use `bench_decode_moe.py --backend offload,cpu,hybrid` with `--moe-cache-size` matched to the 23.8 and 26.3 GB budgets.
- Baseline: llama.cpp BF16 GGUF with `--n-cpu-moe` tuned to the same VRAM.
- These rows are "not adjudicated" because the model's band is too wide, so only a head-to-head can settle them.
- The 4090 row (18.3 GB) can reuse the same host class. The DSV4-Flash and GLM-5.2 rows exceed 128 GB.

**4. KTransformers: value highest in principle, effort 1–2 days, cost ~$15–40**
- A faithful test needs:
  - a Sapphire Rapids/Emerald Rapids Xeon host with AMX, confirmed via `lscpu`
  - ≥128 GB RAM
  - the archive/legacy `Qwen2-serve-amx.yaml` AMXInt8 path, i.e. the paper-era code
  - Qwen2-57B-A14B converted to Int8, from ~114 GB of BF16 safetensors to download and convert
  - llama.cpp Q8_0 with `-ngl 99 --cpu-moe`
- Even then, a single-socket host removes the dual-socket NUMA gain, and a 4090 replaces the 4080. The model must re-predict for the rental host rather than reuse the audited row's numbers.
- Cheaper but less faithful: kt-kernel + sglang-kt on Qwen3-30B-A3B (the documented example) vs llama.cpp `--cpu-moe` on the same AMX host. This tests the mechanism, not an audited row.
- On an AVX2-only host, kt-kernel falls back to llamafile/AVX2 kernels, and the test says nothing about the SOSP claim. Do not run it there.

**5. 2512.16473: value moderate, effort ~4–6 h, cost ~$8–15**
- Use Phi-3.5-MoE or Mixtral BF16 on a 4090 with 128 GB.
- The Phi-3.5 row's predicted S_n ≈ 1.01 makes it a good falsification test of the model. PyTorch BF16 CPU matmuls need AVX-512-BF16 for paper-like speed; the paper used a Threadripper 7960X (Zen 4).

**6. llama.cpp cache branches: value moderate, effort ~1–2 h each, cost ~$2–4**
- A same-binary A/B at matched VRAM on the 5090 host used for (1) or (3).

**7. Fiddler: value low-moderate, effort ~3–5 h**
- Mixtral BF16 only; torch 2.1.2 means use a 4090, not a 5090. Its rows are not adjudicated and the system is superseded.

**Not runnable**
- **SeqMoE:** a baseline-only check is possible (llama.cpp `--n-cpu-moe` for Qwen3-30B-A3B Q8_0 on a 4090 + Sapphire Rapids host, compared with the predicted 61.7 and the reported 30.9 tok/s). Request code from the authors.
- **SP-MoE, MoE-SpeQ:** no code.
- **ProMoE:** the llama.cpp fork and predictors are private.
- **MoE-Infinity:** runnable, but no audited row, and the release differs from the paper.

**Budget and schedule fit**
- Items 1 + 2 + 3 + 6 fit in ~$40–50 of GPU time and 2–3 engineering days.
- Adding item 4 on an AMX host brings the total to ~$60–90. The main risks are finding an AMX + 128 GB + CUDA-13-capable listing, and multi-hour downloads billed at the GPU rate. Where the marketplace allows it, pre-stage the weights on cheaper CPU-only storage.
- Record `lscpu`, `numactl -H`, the driver version and a STREAM or `ft bench bw` host-bandwidth figure on every box. The model's prediction depends on host bandwidth, and rentals vary.

### Gaps
- No primary source quantifies the availability or price of AMX-Xeon + RTX 4090/5090 listings on Western marketplaces. The papers' own Sapphire Rapids rentals (SeqMoE, FreeToken) suggest such hosts exist, but where they were rented was not stated in the passages read.
- Hour estimates are not validated by any third-party reproduction report for HybriMoE, 2512.16473 or Pipelined sharding.
- Storage and egress pricing on the marketplaces, which matters for the 70–114 GB downloads, was not researched.
