# Letter of intent: SPCL, ETH Zurich (≤500 words)

*Draft for you to rewrite in your own voice. Every number matches `paper/numbers4.json` as generated on 27 Sep 2026;
re-check after any rerun. Word count from the salutation to the signature: under 500.*

---

Dear Prof. Hoefler,

I am applying for a PhD position at SPCL. I want to build performance models that treat data movement as the first
cost of large-model inference, use them as yardsticks for the systems we publish, and then use them to design sparse
models for the hardware people actually own.

I have an MS in ECE from UW-Madison (December 2025) and work in GPU performance engineering: CUDA/Triton kernels,
computer architecture, LLM serving.

My most important result is a paper and open artifact, *Seconds, Not Blocks: A Validated Speed-of-Light for Offloaded
Mixture-of-Experts Decode, and What It Says About Published Speed-Ups*. Dozens of systems claim faster decoding of large
MoE models on one consumer GPU, but none states how fast the hardware could go, and baselines are rarely configured the
way users run them. I built the missing yardstick and applied it:

1. **A lower bound in seconds** for any exact-routing expert-placement policy under a GPU memory budget (Belady's
   MIN with bypass, load accounting, Jensen), computed from exact routing traces that I collect by streaming one
   decoder layer at a time, verified token-for-token against reference decode.
2. **A bytes-over-bandwidth decode model** with 16% median error, cross-validated, on 52 third-party measurements
   from 9 sources; calibrated with two runs, it predicted 29 configurations on a new platform (GH200) with 6.7%
   median error, as pre-registered.
3. **An audit of 147 published measurements from 41 systems**, extracted and re-checked in separate AI-agent passes,
   every value with a verbatim quote. On the 22 adjudicable rows, 16 of 20 llama.cpp baselines fall below the band of the
   predicted equal-memory baseline, 8 claimed gains survive (4–10 across the model's error), and the median system reaches 24% of its hardware's
   physical speed-of-light.
4. **Trace provenance.** Teacher-forced dataset text, the norm in this literature, overstates cache hit rates on all
   nine models I traced; for gpt-oss-120b it is off-distribution unless the model's own reasoning is inserted.

I pre-registered every first-party hypothesis in the repository before measuring, following your rules for
scientific benchmarking, and the paper reports the ones that failed, along with my own mistakes: an evaluation window that ran
partly through user prompts, and a system I built before checking novelty, which seven groups had already built.

SPCL is where this work fits: it is data-movement modelling and scientific benchmarking applied to sparse inference.
For a PhD I would pursue three directions:

- **(a) Bounds beyond one request:** batched and speculative decoding, unified-memory machines and CXL tiers,
  validated on systems such as Alps' GH200 nodes.
- **(b) Kernels and runtimes that reach the bound:** CPU+GPU bandwidth aggregation, where my bound says the optimum
  often lies, is only partly exploited by current engines.
- **(c) Model co-design:** invert the model to choose expert count, size, top-k and routing locality so that a given
  memory hierarchy decodes at a target speed, and connect this to scaling laws. My long-term goal is frontier-level
  intelligence on consumer hardware.

Code, traces, audit set and pre-registrations: github.com/harshithkantamneni/moe-speed-of-light

Sincerely,
Harshith Kantamneni
