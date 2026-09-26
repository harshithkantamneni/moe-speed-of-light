# Letter of intent: SPCL, ETH Zurich (≤500 words)

*Draft. Change anything that doesn't sound like you. Numbers match the paper as of this draft; re-check them against `paper/numbers.tex` before sending.*

---

Dear Prof. Hoefler,

I am applying for a PhD position at SPCL. I want to work on performance models that treat data movement as the primary cost of large-model inference, and to use those models to design sparse models for the hardware that ordinary people own.

My background is GPU performance engineering. I completed an MS in ECE at UW-Madison in December 2025, with coursework in computer architecture and parallel computing and projects on CUDA/Triton kernels and LLM serving benchmarks. Since then I have worked as an agentic-systems engineer and continued research independently.

My most important result is a paper, *Where Do the Experts Go? A Validated Speed-of-Light Model for MoE Decode on Memory-Constrained Consumer Hardware* (code and data public). More than thirty systems propose expert-caching or offloading policies for MoE models on desktops, yet there was no ceiling to measure them against. I built three pieces:

1. **Exact routing traces without a GPU.** A causal model's routing is fixed by its prefix. Teacher-forced prefill, streamed one layer at a time with tensors read over HTTP, therefore reproduces decode routing exactly. I verified this against KV-cache decode on four architectures and traced Qwen3-30B-A3B (61 GB of weights) on a 2-vCPU, 7 GB VM.
2. **A bytes-over-bandwidth decode model.** It has six fitted parameters. Holding out one source at a time, it predicts 52 published measurements from 9 independent sources with a 16% median error, covering 0.6–128 tok/s, eight models, and CPU-only, static-offload and PCIe-fetch execution. The main residual is cross-socket synchronisation, which the model deliberately does not fit.
3. **A speed-of-light bound for any per-layer expert-placement policy, including prefetchers.** It combines Belady's MIN-with-bypass, load accounting and Jensen's inequality. It comes with a closed-form reuse threshold r\* for when copying an expert to the GPU can pay off. An independent review caught a flaw in my first version of the bound (it ignored prefetching); the corrected bound has a regression test against an oracle prefetcher.

The findings are that short-range expert reuse is strong, that cross-domain popularity is weak and model-dependent, and that on PCIe 4 desktops a cache miss is better executed on the CPU than fetched.

SPCL is the natural place to continue this work because the paper is, at its core, applied *Data Movement Is All You Need* and applied sparsity. For a PhD I would like to push in three directions:

- **(a) Serving and batching.** Extend the model and bound to multi-request serving and to new memory tiers (CXL, high-bandwidth flash), and validate them on real clusters such as Alps.
- **(b) Kernels.** Build offloaded-expert kernels in the spirit of MARLIN that reach the bound.
- **(c) Architecture co-design.** Invert the model to design MoE architectures under hardware constraints: choose expert count, expert size and top-k so that a given memory hierarchy decodes at a target speed, and connect this to scaling laws. My long-term goal is to make frontier-level models run well on consumer hardware.

I work best where claims are measured and falsifiable, and I would value SPCL's standards for scientific benchmarking.

Sincerely,
Harshith Kantamneni
