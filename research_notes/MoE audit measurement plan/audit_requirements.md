# MoE audit measurement plan: which measurements would settle the uncertain verdicts

Relative links resolve against the repository root `/home/claude/moe-speed-of-light`. Notation used throughout:

- **B** is a measured llama.cpp `--n-cpu-moe` decode speed (tok/s) at equal GPU memory on the paper's hardware.
- **pred** is the audit's predicted equal-VRAM baseline `[lo, mid, hi]`.
- **m = B / pred_mid** is the measured/predicted ratio.
- **m\*** is the value of m at which a label flips.

"Survives" means system / B > 1. "Weak" means reported baseline < B. Both use the protocol 4.6 label definitions, applied to a measured value instead of the predicted band.

## Which adjudicated rows change label between the lo (x0.83) and hi (x1.22) sensitivity runs?

### Takeaway
Eight of the 22 adjudicated rows flip within the sensitivity range, carrying 12 verdicts between them:

- **Five KTransformers rows.** Dual-socket Xeon 8452Y DDR5 host, A100 40GB or RTX 4080, all routed experts on the CPU.
- **Three SeqMoE rows.** Xeon 8470Q host: Qwen3.6-35B-A3B BF16 on an RTX 5090 at 40 % and 15 % residency, and DeepSeek-V4-Flash on an RTX PRO 6000.

Every flip comes from the same mechanism. Scaling the predicted baseline by 0.83 or 1.22 moves a band edge past a reported number (weak flag) or past the system's own speed (survival). Nothing about the rows changes. Three more rows stay "not established" at every scaling, but their normalized speed-up is centred on 1 (m\* between 0.98 and 1.04). A direct measurement would decide them as well.

### Cited Findings
- Headline counts ([audit.json](prereg/audit/audit.json), [lo run](prereg/audit_sens_scale_lo/audit.json), [hi run](prereg/audit_sens_scale_hi/audit.json), [A10 refit](prereg/audit_sens_a10/audit.json)):

  | Run | Gains surviving (of 22) | Weak llama.cpp baselines (of 20) |
  |---|---|---|
  | Central | 8 | 16 |
  | x0.83 (lo) | 10 | 13 |
  | x1.22 (hi) | 4 | 19 |
  | A10 refit | 8 | 17 |

  The model's leave-one-source-out band is 0.775–1.314.
- The 0.83 and 1.22 factors are the A100-host and A10 medians of measured/predicted llama.cpp with experts offloaded, on the audit's datasheet basis. The GH200 median is 0.86. Single configurations span 0.44–1.35 ([paper.tex, "How much to trust this"](paper/paper.tex); [basis.json](prereg/anchors/basis.json); [PROTOCOL.md 4.7 outcomes](prereg/PROTOCOL.md)).
- Label rules ([PROTOCOL.md 4.6](prereg/PROTOCOL.md); [scripts/audit.py](scripts/audit.py) lines 174–184):
  - **Weak:** the reported llama.cpp baseline is below pred_lo.
  - **Survives:** system / pred_hi > 1.
  - **Not adjudicated:** pred_hi/pred_mid > 1.4, or pred_lo/pred_mid < 0.6, or the claim is < 1.2x.
  - Each row is labelled against its strongest reported llama.cpp baseline ([PROTOCOL.md 4.7](prereg/PROTOCOL.md)).
- **The eight rows that flip.** Values come from [audit.json](prereg/audit/audit.json) (pred, S_n, labels), [normalized_v2.jsonl](data/audit/normalized_v2.jsonl) (hardware, budget) and [rows_group*.jsonl](data/audit/rows_group1.jsonl) (CPU and DRAM text).

  | Row | Model, GPU | Host CPU / DRAM as reported | Expert budget | Central → lo → hi labels | Reason for the flip |
  |---|---|---|---|---|---|
  | ktransformers-ds3-bf16-a100-b1 | DeepSeek-V3-0324 BF16, A100 40GB | 2x Xeon Platinum 8452Y (36C each); DDR5, 1 TB per socket, channels not stated; MLC 220 GB/s intra-socket, 125 GB/s cross-socket | 0 experts per layer on GPU (all 58 MoE layers on CPU) | at strength / not est. → same → **weak** / not est. | Reported 4.68 vs pred_lo 4.24; at hi, pred_lo = 5.17. S_n [0.77, 1.04, 1.38] contains 1 in every run. |
  | ktransformers-ds2-bf16-a100-b1 | DeepSeek-V2.5-1210 BF16, A100 40GB | same | 0 per layer (59 layers on CPU) | **weak / not est.** → at strength / **survives** → weak / not est. | Reported 7.05 vs pred_lo 7.92 (6.57 at lo). S_n lower edge 0.85 at central, 1.03 at lo. |
  | ktransformers-qw2-bf16-a100-b1 | Qwen2-57B-A14B BF16, A100 40GB | same | 0 per layer (28 layers on CPU) | at strength / survives → same → **weak / not est.** | Reported 13.01 vs pred_lo 11.77 (14.36 at hi). S_n lower edge 1.09, falling to 0.90 at hi. |
  | ktransformers-ds3-int4-rtx4080-b1 | DeepSeek-V3-0324, KT int4 (4.5 bpw assumed), RTX 4080 16GB | same | 0 per layer | weak / not est. → **at strength / survives** → weak / not est. | Reported 9.06 vs pred_lo 10.11 (8.39 at lo). S_n lower edge 0.91, rising to 1.09 at lo. |
  | ktransformers-ds2-int8-rtx4080-b1 | DeepSeek-V2.5-1210, KT int8 (8.5 bpw assumed), RTX 4080 | same | 0 per layer | at strength / survives → same → **weak / not est.** | Reported 9.94 vs pred_lo 9.51 (11.6 at hi). S_n lower edge 1.06, falling to 0.87 at hi. |
  | seqmoe-qwen3.6-35b-a3b-bf16-rtx5090-40pct | Qwen3.6-35B-A3B BF16, RTX 5090 32GB | Xeon Platinum 8470Q, 120 GB host memory (type not stated), PCIe 5.0 | 40 % of experts (25.8 GB; 24 of 40 layers on CPU) | weak / survives → **at strength** / survives → weak / survives | Reported 37.9 vs pred_lo 43.0 (35.7 at lo). |
  | seqmoe-qwen3.6-35b-a3b-bf16-rtx5090-15pct | same | same | 15 % (9.7 GB; 34 of 40 on CPU) | weak / survives → same → weak / **not est.** | S_n lower edge 1.13, falling to 0.92 at hi. |
  | seqmoe-deepseek-v4-flash-rtxpro6000-45pct | DeepSeek-V4-Flash (quantization not stated; MXFP4 experts / FP8 dense assumed), RTX PRO 6000 Blackwell 96GB | Xeon Platinum 8470Q, 256 GB host memory (type not stated), PCIe 5.0 | 45 % (66.2 GB; 24 of 43 on CPU) | weak / survives → same → weak / **not est.** | S_n lower edge 1.12, falling to 0.92 at hi. |

- The 12 flipping verdicts split into two groups ([audit.json](prereg/audit/audit.json) and the sensitivity runs):
  - **Six weak/at-strength flags:** ds3-bf16, ds2-bf16, qw2-bf16, ds3-int4, ds2-int8, seqmoe-qwen3.6-40pct.
  - **Six survival flags:** ds2-bf16, qw2-bf16, ds3-int4, ds2-int8, seqmoe-qwen3.6-15pct, seqmoe-dsv4flash.
- The A10-refit run flips one additional flag: ktransformers-ds2-int8 becomes weak and still survives ([audit_sens_a10](prereg/audit_sens_a10/audit.json)).
- **Rows stable in every run whose verdict is still on a knife edge** ([audit.json](prereg/audit/audit.json)):
  - ktransformers-ds3-bf16: S_n [0.77, 1.04, 1.38].
  - seqmoe-qwen3-30b-a3b-fp8-rtx4090-15pct: S_n [0.73, 0.98, 1.29]. Xeon Gold 6430, 120 GB.
  - cpugpucollab-phi3.5moe-rtx4090: S_n [0.74, 1.01, 1.34]. Threadripper 7960X; its only baseline is the authors' own CPU-only mode, so it has no llama.cpp baseline.
- **Decision thresholds** for a measured baseline, computed from [audit.json](prereg/audit/audit.json) fields. u is the combined measurement and transfer uncertainty. Survival needs B < sys/(1+u); weak needs B > rep/(1−u).

  | Row | sys | rep | pred mid [lo, hi] | m\* survive | m\* weak | u=10 %: survive if B < / weak if B > | u=25 % |
  |---|---|---|---|---|---|---|---|
  | KT ds3-bf16 | 5.87 | 4.68 | 5.6 [4.2, 7.6] | 1.04 | 0.83 | 5.3 / 5.2 | 4.7 / 6.2 |
  | KT ds2-bf16 | 11.95 | 7.05 | 10.4 [7.9, 14.0] | 1.15 | 0.68 | 10.9 / 7.8 | 9.6 / 9.4 |
  | KT qw2-bf16 | 22.88 | 13.01 | 15.5 [11.8, 20.9] | 1.47 | 0.84 | 20.8 / 14.5 | 18.3 / 17.3 |
  | KT ds3-int4 | 16.15 | 9.06 | 13.3 [10.1, 17.8] | 1.22 | 0.68 | 14.7 / 10.1 | 12.9 / 12.1 |
  | KT ds2-int8 | 17.61 | 9.94 | 12.4 [9.5, 16.6] | 1.42 | 0.80 | 16.0 / 11.0 | 14.1 / 13.3 |
  | KT qw2-int8 | 36.99 | 19.17 | 19.2 [14.6, 25.6] | 1.93 | 1.00 | 33.6 / 21.3 | 29.6 / 25.6 |
  | SeqMoE qwen3-30b 45 % | 104.1 | 30.9 | 61.7 [46.9, 82.6] | 1.69 | 0.50 | 94.6 / 34.3 | 83.3 / 41.2 |
  | SeqMoE qwen3-30b 15 % | 47.9 | 21.1 | 48.9 [37.0, 65.7] | 0.98 | 0.43 | 43.5 / 23.4 | 38.3 / 28.1 |
  | SeqMoE qwen3.6 40 % | 118.3 | 37.9 | 56.4 [43.0, 75.4] | 2.10 | 0.67 | 107.5 / 42.1 | 94.6 / 50.5 |
  | SeqMoE qwen3.6 15 % | 70.2 | 28.2 | 46.5 [35.3, 62.3] | 1.51 | 0.61 | 63.8 / 31.3 | 56.2 / 37.6 |
  | SeqMoE gpt-oss-120b | 139.2 | 33.1 | 74.3 [56.5, 99.6] | 1.87 | 0.45 | 126.5 / 36.8 | 111.4 / 44.1 |
  | SeqMoE DS-V4-Flash | 57.4 | 21.0 | 38.3 [29.1, 51.2] | 1.50 | 0.55 | 52.2 / 23.3 | 45.9 / 28.0 |
  | cpugpucollab phi3.5 | 10.4 | 6.3 (not llama.cpp) | 10.3 [7.8, 14.1] | 1.01 | n/a | 9.5 / n/a | 8.3 / n/a |
  | HybriMoE (5 rows) | – | – | – | 0.11–0.50 | 0.06–0.38 | – | – |
  | pipeshard (3 rows) | – | – | – | 0.46–0.63 | 0.37–0.46 | – | – |
  | SP-MoE DS-V2-Lite | – | – | – | 0.16 | n/a | – | – |

- **Measured/predicted ratios already known for the audit's models** (audit datasheet basis, configurations with experts offloaded) ([basis.json](prereg/anchors/basis.json)):

  | Model | A100 host | GH200 |
  |---|---|---|
  | DeepSeek-V2-Lite Q8_0 | 0.49–0.66 | 0.44–0.69 |
  | Qwen3-30B-A3B Q4_K_M | 0.73–0.94 | 0.77–1.21 |
  | Phi-3.5-MoE Q4_K_M | 0.76–0.82 | 0.76–0.85 |
  | Qwen2-57B Q4_K_M | 0.73–0.95 | 0.86–0.99 |
  | Mixtral Q8_0 | 0.91–1.02 | 0.83–0.89 |
  | Mixtral Q4_K_M | 0.87–0.91 | 0.89–0.91 |

  A10 median: 1.22.

### Inferences
- **The KT flips hinge on one modelling assumption: that llama.cpp uses both sockets.** The audit counted both sockets because KTransformers is NUMA-aware, and records that "whether the llama.cpp baseline used both sockets is not stated" ([normalized_v2.jsonl](data/audit/normalized_v2.jsonl) `assume`). Re-running the audit's own predictor with changed bandwidth gives:
  - **One socket at datasheet (282–307 GB/s):** predicted llama.cpp is 3.3–3.6 tok/s for ds3-bf16, 6.8–7.2 for ds2-bf16 and 9.8–10.5 for qw2-bf16. This is below every reported KT baseline (4.68 / 7.05 / 13.01).
  - **Paper's MLC figure summed over two sockets (440 GB/s):** 4.63, 8.97 and 13.18.
  - **Audit's 563–614 GB/s band:** 5.5–5.8, 10.2–10.7 and 15.2–15.9.

  These numbers are from my run of `static_offload_time` with `scripts/audit.py`'s `params_without` fit. Only a dual-socket measurement can settle the KT flags.
- **The measurements would probably push survival up.** Where a per-model anchor ratio exists, it mostly sits below the survival thresholds:
  - **Qwen2-57B:** ratio 0.73–0.99 against m\* survive 1.47 for qw2-bf16 → survives.
  - **DeepSeek-V2-Lite:** ratio 0.44–0.69 against m\* survive 1.04–1.22 for the DeepSeek KT rows → survive.
  - **Qwen3:** ratio 0.73–1.21 against m\* 1.51–2.10 for the Qwen3.6 SeqMoE rows → survive.

  This rests on two untested assumptions: that a model's anchor ratio carries over to its larger sibling (DeepSeek-V2-Lite's experts are 8.7M parameters, DeepSeek-V3's are 44M), and that it carries over across hosts. Measuring tests exactly this.
- **The weak flags for KT ds2-bf16, KT ds3-int4 and SeqMoE qwen3.6-40pct would probably turn "at strength"** if the DeepSeek over-prediction seen on DeepSeek-V2-Lite (m 0.44–0.69) holds, since their m\* weak is 0.67–0.68.
- **SeqMoE qwen3-30b-15pct (m\* 0.98) and cpugpucollab-phi (m\* 1.01) are the rows where measurement moves the verdict most per run.** Their Qwen3 and Phi anchor ratios (0.73–1.21 and 0.76–0.85) straddle or sit just below 1. With Phi at 0.76–0.85, the cpugpucollab gain would survive by about 1.2x.

### Gaps
- No anchor ratio exists for DeepSeek-V2.5, DeepSeek-V3, DeepSeek-V4-Flash, Qwen3.6-35B-A3B, gpt-oss-120b (offloaded) or Qwen3-235B Q2_K. Transfer from the smaller siblings is assumed, not measured.
- [basis.json](prereg/anchors/basis.json) records only aggregate A10 ratios, with no per-model breakdown, so the A10 end of the range cannot be assigned to a model.
- The protocol has no rule yet for replacing a predicted baseline with a measured one. [PROTOCOL.md 4.7](prereg/PROTOCOL.md) says "Anchors validate the predictor; they do not replace any audit row's prediction". A tolerance u and a transfer rule would need to be registered before measuring.

## For every adjudicated row and the near-miss rows, what exact configuration must a measurement reproduce?

### Takeaway
All 22 adjudicated rows are single-request decode rows, but their conditions differ widely:

- **Hosts:** a dual-socket 16-channel DDR5 server (KT), single-socket 8-channel DDR5 Xeons (SeqMoE), a 4-channel DDR5 Threadripper (cpugpucollab), 6-channel DDR4 Cascade Lake (HybriMoE), and a 153.6 GB/s EPYC (pipeshard).
- **GPU memory:** from 16 GB (RTX 4080) to 96 GB (RTX PRO 6000).
- **Host RAM:** 17 GB (Qwen3-30B Q4_0) to 1,342 GB (DeepSeek-V3 BF16).
- **Context:** 178 to 4,096 tokens.

Every row has a public GGUF that matches its bytes, except two cases:

- **SeqMoE's FP8 Qwen3-30B:** nearest is Q8_0, with 6 % more expert bytes.
- **KT's and HybriMoE's own int4/int8 formats:** the audit assumed 4.5 or 4.125 and 8.5 bpw, matched by Q4_0 and Q8_0.

### Cited Findings
- **How `--n-cpu-moe` counts layers:** it counts from the first layer, including leading dense layers. On DeepSeek-V2-Lite, `--n-cpu-moe 14` put 13 MoE layers on the CPU ([a100.json](prereg/anchors/a100.json), configs `moe_layers_on_cpu`). So a row's MoE-layer count must be offset by its leading dense layers:
  - DeepSeek-V2-Lite / V2.5: 1
  - DeepSeek-V3 / GLM-5.2: 3
  - Qwen, Mixtral, Phi, gpt-oss, DeepSeek-V4-Flash: 0

  Layer counts are from [mosl/archs.py](mosl/archs.py) `shape()`. The mapping from budget to CPU layers is n_cpu = L − floor(B / (E × expert bytes)) ([scripts/audit.py](scripts/audit.py) lines 150–152).
- **Existing measurement harness** ([jobs/051_anchor_sweep@anchor.sh, `origin/gpu` branch](jobs/051_anchor_sweep@anchor.sh)[^1]):
  - stock llama.cpp 2145525a;
  - `llama-bench -ngl 99 -ncmoe N -fa 1 -p 0 -n 128 -r 5`;
  - SM and memory clocks locked;
  - model files read into the page cache first;
  - threads = best of {8, nproc/2, nproc};
  - one randomized order plus two drift repeats.

  The whole A100 sweep (31 llama-bench configurations) ran from 22:44:29 to 22:58:42 UTC by its `gpu_monitor.csv`, about 14 minutes.
- **Known pitfalls from earlier runs** ([PROTOCOL.md 4.7 outcomes](prereg/PROTOCOL.md)):
  - GH200 build failure: cmake 3.22 rejects `CMAKE_CUDA_ARCHITECTURES=native`.
  - "No A100 had capacity on Lambda during the study (poller from 13:38 UTC)".
  - Lambda supplied an A100-SXM4-80GB when the 40 GB card was requested.
- **Cloud VM hosts reach only part of the host's datasheet bandwidth:** STREAM Triad was 46–59 % of datasheet on the two anchor hosts ([paper.tex](paper/paper.tex)). The A100 host reached 93.7 GB/s of 204.8 GB/s. It scaled with threads: 28.5 (1), 68.4 (4), 80.7 (8), 83.2 (15), 93.7 (30) GB/s ([a100.json](prereg/anchors/a100.json) `stream_triad_by_threads`). The GH200 reached 301 GB/s of 512 ([gh200.json](prereg/anchors/gh200.json)).
- **GGUF files that match the rows' bytes (Hugging Face listings):**

  | Model | File | Size | Source |
  |---|---|---|---|
  | Qwen2-57B-A14B-Instruct | fp16 (3 shards) | 114.8 GB | [Qwen/Qwen2-57B-A14B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct-GGUF) |
  | Qwen2-57B-A14B-Instruct | q8_0 (2 shards) | 61.0 GB | same |
  | Qwen2-57B-A14B-Instruct | q4_0 | 32.5 GB | same |
  | DeepSeek-V2.5-1210 | Q8_0 (7 shards) | 250.6 GB | [bartowski/DeepSeek-V2.5-1210-GGUF](https://huggingface.co/bartowski/DeepSeek-V2.5-1210-GGUF) |
  | DeepSeek-V3-0324 | various | – | [bartowski/deepseek-ai_DeepSeek-V3-0324-GGUF](https://huggingface.co/bartowski/deepseek-ai_DeepSeek-V3-0324-GGUF) |
  | Qwen3-30B-A3B | Q8_0 (no FP8 in that repo) | 32.5 GB | [Qwen/Qwen3-30B-A3B-GGUF](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF) |
  | Qwen3.6-35B-A3B | BF16 (2 shards) | 69.4 GB | [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) |
  | Qwen3.6-35B-A3B | Q8_0 | 36.9 GB | same |
  | Qwen3.6-35B-A3B | MXFP4_MOE | 21.7 GB | same |
  | gpt-oss-120b | MXFP4 | 63.4 GB | [ggml-org/gpt-oss-120b-GGUF](https://huggingface.co/ggml-org/gpt-oss-120b-GGUF) |
  | DeepSeek-V4-Flash | MXFP4 | – | [6ms/DeepSeek-V4-Flash-MXFP4-GGUF](https://huggingface.co/6ms/DeepSeek-V4-Flash-MXFP4-GGUF) |
  | DeepSeek-V4-Flash-0731 | UD-Q4_K_XL (5 shards) | ~155 GB | [unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF) |

- **Whole-model bytes at the rows' bit widths** (computed with [mosl/archs.py](mosl/archs.py)):

  | Model | Size at the row's bit width |
  |---|---|
  | DeepSeek-V3 | BF16 1,342 GB; 4.5 bpw 377 GB; 4.85 bpw 407 GB |
  | DeepSeek-V2.5 | BF16 471 GB; 8.5 bpw 250 GB |
  | Qwen2-57B | BF16 115 GB; 8.5 bpw 61 GB |
  | Qwen3-30B | 8.5 bpw 32 GB; Q4_0 17 GB |
  | Qwen3.6-35B | BF16 69 GB |
  | gpt-oss-120b | MXFP4 62–65 GB |
  | DeepSeek-V4-Flash | 4.25 bpw 151–155 GB |
  | Qwen3-235B | Q2_K 86 GB (77.0 GB on disk per the row) |
  | Phi-3.5-MoE | 16-bit 84 GB |
  | GLM-5.2 | NVFP4 418–445 GB |

- **Share of predicted decode time spent on the GPU at the row's own configuration** (my run of the audit predictor):
  - KT rows: 30–59 %.
  - SeqMoE rows: 20–47 %.
  - FreeToken rows: 19–64 %.
  - pipeshard rows: 6–12 %.
  - cpugpucollab-phi: 14 %.

  Substituting GPUs moves the prediction accordingly:
  - RTX 4080 → RTX 4090: +14 % (qw2-int8), +21 % (ds2-int8).
  - RTX 4080 → A10: −7 to −11 %.
  - RTX PRO 6000 → H100 SXM: +28 % (DS-V4-Flash row), +25 % (gpt-oss).
  - A100 40GB → A100 80GB: +8 to +11 %.
- **Per-row requirements, adjudicated rows.** Hardware, budget, workload and metric are from [normalized_v2.jsonl](data/audit/normalized_v2.jsonl) and [rows_group*.jsonl](data/audit/rows_group1.jsonl). n_cpu is from [audit.json](prereg/audit/audit.json). Host bands are the audit's datasheet bands in GB/s.

  | Row(s) | Model / quant → GGUF | GPU (VRAM, GB/s) | Budget → llama.cpp flag | Host as reported (audit band) | Host RAM needed (model size) | Workload and metric |
  |---|---|---|---|---|---|---|
  | KT ds3-bf16 | DeepSeek-V3-0324 BF16 (llama.cpp ran FP16) → BF16 GGUF | A100 40GB, 1,555, PCIe 4.0 x16 | 0 experts per layer on GPU → `-cmoe` (≡ `--n-cpu-moe 61`) | 2x Xeon 8452Y; DDR5, 1 TB per socket; MLC 220 intra / 125 cross (563–614, both sockets) | ≥ 1.5 TB (1,342 GB) | prompt 32, output 512, ctx 288. Decode tok/s, KT without Expert Deferral, read from Fig. 12 bars (figure only) |
  | KT ds2-bf16 | DeepSeek-V2.5-1210 BF16 | A100 40GB | `-cmoe` (60) | same | ≥ 512–640 GB (471 GB) | same |
  | KT qw2-bf16 | Qwen2-57B-A14B BF16 → fp16 GGUF | A100 40GB | `-cmoe` (28) | same | ≥ 128 GB (115 GB) | same |
  | KT ds3-int4 | DeepSeek-V3-0324, KT int4 (format unstated; 4.5 bpw assumed; llama.cpp "built-in quant", type unstated) → Q4_0 | RTX 4080 16GB, 717 | `-cmoe` (61) | same | ≥ 448–512 GB (377 GB) | same |
  | KT ds2-int8 | DeepSeek-V2.5-1210, KT int8 (8.5 bpw assumed) → Q8_0 | RTX 4080 | `-cmoe` (60) | same | ≥ 300–320 GB (250.6 GB) | same |
  | KT qw2-int8 | Qwen2-57B int8 → Q8_0 | RTX 4080 | `-cmoe` (28) | same | ≥ 80–96 GB (61 GB) | same |
  | HybriMoE DS-V2-Lite 25 % / 75 % | DS-V2-Lite-Chat. GPU int4 Marlin (4.125 bpw); CPU-side and baseline formats unstated → Q4_0 | RTX A6000 48GB, 768; PCIe not stated (Gen3 x16 assumed) | 1.9 / 5.6 GB → `--n-cpu-moe 21` / `8` | Xeon Gold 5220R restricted to 10 cores; DRAM not stated (6-channel DDR4-2400..2666 assumed: 115–128) | ≥ 16 GB | prompt and output not stated (128/256 assumed). TBT from Fig. 8 bars |
  | HybriMoE Mixtral 25 % | Mixtral-8x7B-Instruct, int4 as above | RTX A6000 | 5.8 GB → 24 | same | ≥ 32 GB | same |
  | HybriMoE Qwen2-57B 25 % / 75 % | Qwen2-57B-A14B-Instruct, int4 | RTX A6000 | 6.4 / 19.1 GB → 21 / 7 | same | ≥ 40 GB | same |
  | SP-MoE DS-V2-Lite | DS-V2-Lite 16-bit (+ AWQ draft) | A100 40GB, PCIe 4.0 x16 | 7 GB GPU total (3.3 GB experts) → 25 | Xeon Platinum 8358P, 1 TB (type not stated) (187.7–204.8) | ≥ 40 GB | output 100, prompt not stated. TPOT (text). Speculative decoding in system and baselines |
  | cpugpucollab Phi-3.5 | Phi-3.5-MoE 16-bit (88 GB stated) | RTX 4090 24GB, 1,008 | 125 of 512 expert slots (~19.7 GB) → 25 | Threadripper 7960X, OMP 24 threads; DRAM not stated (4-channel DDR5-4800..5200: 153.6–166.4) | ≥ 96–128 GB (84 GB) | throughput tok/s, single request. Prefill inclusion and lengths not stated. Fig. 5 bar label |
  | SeqMoE Qwen3-30B 45 % / 15 % | Qwen3-30B-A3B-FP8 → Q8_0 (+6 % bytes) | RTX 4090, 1,008, PCIe 4.0 | 13.1 / 4.4 GB → 27 / 41 | Xeon Gold 6430, 120 GB (type not stated) (8-channel DDR5-4000..4400: 256–282) | ≥ 48 GB (32.5 GB) | decode length 128–1024 (midpoint 576), prompt not stated (128 assumed), ctx 416. Datasets MATH, GSM8K, CodeForces, OpenOrca, ShareGPT. Decode tok/s bar labels |
  | SeqMoE Qwen3.6 40 % / 15 % | Qwen3.6-35B-A3B BF16 → BF16 GGUF | RTX 5090 32GB, 1,792, PCIe 5.0 | 25.8 / 9.7 GB → 24 / 34 | Xeon Platinum 8470Q, 120 GB (8-channel DDR5-4400..4800: 281.6–307.2) | ≥ 96 GB (69.4 GB) | same |
  | SeqMoE gpt-oss-120b 45 % | MXFP4 experts (dense unstated) → ggml-org MXFP4 | RTX PRO 6000 96GB, 1,792 | 27.4 GB → 20 | Xeon 8470Q, 256 GB (281.6–307.2) | ≥ 96 GB (63.4 GB) | same |
  | SeqMoE DS-V4-Flash 45 % | quantization not stated → MXFP4 GGUF | RTX PRO 6000 (needs ≥ ~74 GB) | 66.2 GB → 24 | Xeon 8470Q, 256 GB | ≥ 192 GB (~155 GB) | same |
  | pipeshard Qwen3-30B 8G | Qwen3-30B-A3B-Instruct-2507 Q4_0 (16.4 GB) | RTX 5090, PCIe 5.0 x16 | 8 GB total VRAM (5.7 GB experts) → 32 | AMD EPYC, 16 cores (model unstated); 256 GB; "Mem BW 153.6 GBps" | ≥ 32 GB | prompt 1024, output 100, ctx 1074. Decode TPS (Table 4). Baselines derived from the artifact CSV: `-ngl 20` / `1` / `36`, default mmap, threads = physical cores |
  | pipeshard Qwen3-235B 2G / 32G | Qwen3-235B-A22B-Instruct-2507 Q2_K (77.0 GB) | RTX 5090 | 2 GB → all 94 layers; 32 GB (27.6 GB experts) → 63 | same | ≥ 128 GB | same |

- **Per-row requirements, near-miss non-adjudicated rows.** These have a claim ≥ 1.2x and a band just over the ±40 % rule (hi/mid 1.41–1.49), or were widened only by an imputed budget ([audit.json](prereg/audit/audit.json), [normalized_v2.jsonl](data/audit/normalized_v2.jsonl), [rows_group*.jsonl](data/audit/rows_group1.jsonl)).

  | Row | Band hi/mid | S_n | Reported llama.cpp | Model | GPU | Host (measured B_H) | Budget | Host RAM | Workload |
  |---|---|---|---|---|---|---|---|---|---|
  | FreeToken qwen3.6 5090srv-w1 | 1.46 | [1.50, 2.19, 3.11] | 42.6 (unknown config: "routing-blind static split") | Qwen3.6-35B-A3B BF16 | RTX 5090 | 2x Xeon Gold 6459C capped at 6 threads, NUMA-pinned; DDR5 180 GiB container quota (77.3) | 37 % → 26 layers on CPU | ≥ 96 GB | ctx 4096; W1 (AIME single turn); per-request mean decode tok/s |
  | FreeToken DS-V4-Flash 5090srv-w1 | 1.49 | [0.94, 1.40, 2.03] | 13.0 | DS-V4-Flash, MXFP4 experts / FP8 dense | RTX 5090 | same host, 8 threads | 11 % → 39 | ≥ 192 GB | same |
  | FreeToken qwen3.6 NVFP4 4060-laptop-w2 | 1.41 | [1.09, 1.53, 2.11] | 22.3 | Qwen3.6-35B-A3B NVFP4 | RTX 4060 Laptop 8GB, PCIe 4.0 x8 | i9-13900H; LPDDR5 32 GiB (47.5) | imputed | laptop | W2 |
  | FreeToken GLM-5.2 NVFP4 rtxpro6000-w1 | 1.41 | [1.32, 1.87, 2.57] | 7.3 | GLM-5.2, 753B | RTX PRO 6000 | Xeon 8559C, 48 threads; DDR5 512 GiB (178) | imputed | ≥ 512 GB | W1 |
  | FreeToken qwen3.6 4090srv-w2 | 1.64 | [1.19, 1.96, 3.03] | 25.8 | Qwen3.6-35B-A3B BF16 | RTX 4090 | 2x Xeon 8358P, 6 threads; DDR4 240 GiB quota (63.2) | imputed | ≥ 96 GB | W2 |
  | FreeToken qwen3.6 5090desktop-w2 | 2.04 | [1.90, 3.87, 6.77] | 33.0 | Qwen3.6-35B-A3B BF16 | RTX 5090 | Ryzen 9 9950X3D; 2-channel DDR5, 192 GiB (53.8) | imputed | ≥ 96 GB | W2 |

  Other near-miss rows:
  - **fiddler-mixtral8x7b-quadrortx6000-b1:** band 1.94, from the 1-vs-2-socket ambiguity. Claim 1.49; S_n [0.28, 0.55, 0.94]. Mixtral 16-bit (93 GB). Quadro RTX 6000 on PCIe 3. "Xeon Gold 6126 (48 core)". End-to-end tok/s at 32 in / 256 out.
  - **fate-qwen1.5moe-gtx1080ti:** band 1.48. Lossy; S_n ≤ 0.29.
  - **moespeq-phi3.5moe (two rows):** band 1.77, from the imputed budget. Non-llama.cpp baselines; S_n ≤ 0.91.
- **FreeToken's reported llama.cpp baselines all sit inside the predicted band** (at strength). For example, 42.6 lies in [24.8, 51.4] for the 5090srv row (derived from [audit.json](prereg/audit/audit.json) S_n and system values).

### Inferences
- **Suggested llama-bench command per row:**

  ```
  llama-bench -ngl 99 --n-cpu-moe <n_cpu + leading dense layers> -fa 1 -r 5 -d <mean prompt> -n <output>
  ```

  This relies on the depth option `-d` existing in the pinned llama.cpp commit (not verified). Otherwise add a separate context-depth run: the harness's `-p 0 -n 128` measures decode at an empty context, while the rows' contexts reach 1,074 (pipeshard) and 4,096 (FreeToken).
- **Thread settings must follow the paper's own:**
  - FreeToken servers: 6 threads (8 for DS-V4-Flash), pinned to the GPU's NUMA node.
  - cpugpucollab: 24 threads.
  - HybriMoE: 10 cores.
  - KT: all cores of both sockets. Measure both an interleaved/distributed NUMA placement and single-socket placement, and label with the stronger.
- **Imputed-budget rows** (FreeToken w2 rows, ProMoE, SP-MoE non-DS rows, MoE-SpeQ): measuring llama.cpp at the whole-card budget gives an upper bound on the equal-VRAM baseline, because the baseline only gets faster with more VRAM. If the system beats that upper bound, the gain survives whatever the unknown budget was.
- **Rows using FP8 (SeqMoE Qwen3-30B) or KT's own int formats** will run through Q8_0 or Q4_0 GGUFs. Record the byte ratio and scale the per-CPU-layer slope by it. The affine form of decode time in n makes this a one-line correction: R² ≥ 0.970 on A100 and ≥ 0.997 on GH200 ([PROTOCOL.md 4.7 outcomes](prereg/PROTOCOL.md)).
- **Because STREAM on cloud VMs reaches only 46–59 % of datasheet,** every target should record STREAM Triad at the run's thread count. It should also sweep n (at least three values around the row's n) so the per-CPU-layer slope can be rescaled to the paper host's bandwidth when the rented host differs.

### Gaps
- Not verified on Hugging Face: BF16 GGUFs for DeepSeek-V2.5-1210 and DeepSeek-V3-0324, a 16-bit Phi-3.5-MoE GGUF, and a Q4_0 Qwen3-30B-A3B-Instruct-2507 file. The bartowski DeepSeek-V2.5 top-level listing showed quant folders but no BF16.
- DeepSeek-V4-Flash GGUFs exist mainly for the 0731 checkpoint. Whether it has an identical tensor layout to the SeqMoE and FreeToken "DeepSeek-V4-Flash" is not verified.
- Whether llama.cpp commit 2145525a supports DeepSeek-V4-Flash, Qwen3.6 and llama-bench `-d` is not verified.
- Several paper-side values are unstated, and a measurement cannot recover them: DRAM channels and speed for every SeqMoE, HybriMoE, SP-MoE and cpugpucollab host; HybriMoE's PCIe generation; SeqMoE's prompt length.

## Which measurement targets (same model + GPU class + host class) settle the most uncertain verdicts, and at what cost?

### Takeaway
Grouped by GPU class and host class, the rows form four real targets plus two diagnostic ones:

- **Dual-socket KT server:** settles up to 9 of the 12 flipping verdicts. Also the most expensive (full dual-socket DDR5 node, ≥ 512 GB for the DeepSeek rows).
- **RTX 5090 + single-socket DDR5 Xeon:** settles 2 flipping verdicts. Makes two FreeToken near-miss rows adjudicable, re-checks three pipeshard rows, and allows two head-to-heads. Cheap GPU; RAM (96–192 GB) is the cost driver.
- **RTX 4090 + DDR5 host:** settles the two knife-edge rows (m\* 0.98 and 1.01) and one FreeToken near-miss row, with two head-to-heads. Cheapest.
- **≥ 80 GB GPU + 256 GB DDR5 Xeon:** settles 1 flipping verdict (SeqMoE DS-V4-Flash).

Within $100, the best mix is the two consumer-GPU targets first, then a Qwen2-only session on a dual-socket host. That session answers the NUMA question that decides all six KT rows.

### Cited Findings
- **Row membership, GPU, host and RAM figures** are from the tables above ([normalized_v2.jsonl](data/audit/normalized_v2.jsonl), [audit.json](prereg/audit/audit.json); model sizes from [mosl/archs.py](mosl/archs.py) and the Hugging Face listings cited above).
- **SeqMoE hardware as stated in the paper** ([SeqMoE arXiv HTML §9.1](https://arxiv.org/html/2609.12978v1)):

  > "The NVIDIA RTX 4090 platform has 24 GB GPU memory, an Intel Xeon Gold 6430 CPU, and 120 GB host memory, connected via PCIe 4.0. The NVIDIA RTX 5090 platform has 32 GB GPU memory, an Intel Xeon Platinum 8470Q CPU, and 120 GB host memory, connected via PCIe 5.0. The NVIDIA RTX PRO 6000 Blackwell platform has 96 GB GPU memory, an Intel Xeon Platinum 8470Q CPU, and 256 GB host memory"

- **SeqMoE's baseline set** ([SeqMoE arXiv HTML](https://arxiv.org/html/2609.12978v1)): llama.cpp is described only as supporting "concurrent CPU–GPU computation", with no flags. SeqMoE also reports FreeToken as a baseline on the 5090 Qwen3.6 platform: 86.2 / 46.3 tok/s at 40 % / 15 % ([normalized_v2.jsonl](data/audit/normalized_v2.jsonl)).
- **Protocol 4.7 used the GH200 as an out-of-envelope anchor platform:** Grace Arm CPU, LPDDR5X, NVLink-C2C, outside the model's fitted envelope of x86 hosts and PCIe ([PROTOCOL.md 4.7](prereg/PROTOCOL.md)). The GH200 reached 301 GB/s STREAM ([gh200.json](prereg/anchors/gh200.json)).
- **The repository's GPU jobs were run on Lambda** ([gpu/lambda.py](gpu/lambda.py) reads `price_cents_per_hour` from the Lambda API). No price log is committed.

### Inferences
- **Requirements table by target, ranked.** "Flip verdicts" counts only verdicts that change between the lo and hi runs. "Other" counts knife-edge or near-miss rows turned decidable.

  | Rank | Target | GPU | Host class and memory configuration | Host RAM | Model files | Rows | Verdicts settled | Head-to-head | Cost class |
  |---|---|---|---|---|---|---|---|---|---|
  | 1 | **T1: RTX 5090 + single-socket DDR5 Xeon** | RTX 5090 32 GB (1,792 GB/s, PCIe 5.0 x16) | Sapphire/Emerald Rapids-class, 8-channel DDR5-4400..4800 (282–307 GB/s datasheet) for SeqMoE. The same box capped at 6/8 threads for FreeToken. | ≥ 96 GB (Qwen3.6 BF16 69.4 GB); ≥ 128 GB with Qwen3-235B Q2_K (86 GB); ≥ 192 GB with DS-V4-Flash (~155 GB) | unsloth Qwen3.6-35B-A3B BF16; DS-V4-Flash MXFP4 or UD-Q4_K_XL; Qwen3-30B-2507 Q4_0; Qwen3-235B-2507 Q2_K | SeqMoE qwen3.6 40 % and 15 %; FreeToken 5090srv qwen3.6 and DS-V4-Flash; pipeshard x3 | Flip: 2. Other: 2 near-miss rows become adjudicable; 3 robustness checks. | FreeToken, pipeshard | Low–medium: consumer GPU, RAM-driven |
  | 2 | **T2: RTX 4090 + DDR5 host** | RTX 4090 24 GB (1,008 GB/s) | SeqMoE: Gold 6430-class 8-channel DDR5-4000..4400 (256–282). cpugpucollab: 4-channel DDR5 workstation (154–166), 24 threads. FreeToken: DDR4 Ice Lake at 6 threads. | ≥ 48 GB (Qwen3-30B Q8_0); ≥ 96–128 GB with Phi-3.5-MoE 16-bit (84 GB) or Qwen3.6 BF16 | Qwen3-30B-A3B Q8_0; Phi-3.5-MoE 16-bit; Qwen3.6 BF16 | SeqMoE qwen3-30b 15 % and 45 %; cpugpucollab phi; FreeToken 4090srv | Flip: 0. Other: 2 knife-edge rows, 1 near-miss, 1 robustness check. | cpugpucollab, FreeToken | Lowest |
  | 3 | **T3-lite: dual-socket KT host, Qwen2-57B only** | 40 GB-class HBM card (A100 40/80GB) and a ~700 GB/s 16 GB-class card for the int row | 2x Sapphire Rapids, 16-channel DDR5-4400..4800 (563–614), full node (not a VM slice) | ≥ 160–192 GB | Qwen2-57B-A14B-Instruct fp16 and q8_0 | KT qw2-bf16, qw2-int8 | Flip: 2. Also answers whether llama.cpp gets two-socket bandwidth, which decides the direction of all 6 KT rows. | KTransformers | High: full dual-socket node |
  | 4 | **T4: ≥ 80 GB GPU + 8470Q-class host** | RTX PRO 6000 96GB. Nearest bandwidth substitute: A100 80GB (2,039 vs 1,792 GB/s). The row needs ~74 GB of GPU memory. | Single-socket 8-channel DDR5 (282–307) | ≥ 192–256 GB | DS-V4-Flash MXFP4; gpt-oss-120b MXFP4 | SeqMoE DS-V4-Flash, gpt-oss-120b | Flip: 1 (survival). Other: 1 robustness check. | none (no SeqMoE code) | Medium–high |
  | 5 | **T3-full: dual-socket KT host with DeepSeek** | as T3-lite | as T3-lite | ≥ 320 GB (DS-V2.5 Q8_0); ≥ 512 GB (DS-V3 Q4_0, 377 GB); ≥ 640 GB (DS-V2.5 BF16) | DS-V2.5-1210 Q8_0 (and BF16); DS-V3-0324 Q4_0 | KT ds2-int8, ds3-int4, ds2-bf16; ds3-bf16 via substitute | Flip: 7 more (to 9 total with T3-lite). | KTransformers | Highest |
  | diag. | **T5: GH200** (already used for anchors) | H100 96 GB, NVLink-C2C | Grace, 480 GB LPDDR5X (301 GB/s STREAM) | 480 GB | Qwen2-57B BF16/Q8_0, DS-V2.5 Q8_0, DS-V3 Q4 | informs KT DeepSeek rows | Settles none. Gives the per-architecture all-experts-on-CPU ratio. | – | Low–medium |
  | – | **T6: HybriMoE A6000 / DDR4 Cascade Lake** | RTX A6000 | 6-channel DDR4, 10 cores | ≥ 40 GB | as above | HybriMoE x5 | None: every m\* ≤ 0.50, below every anchor ratio | HybriMoE | Low, but no value |

- **Ranked by flip verdicts per target:** T3-full + T3-lite (9) > T1 (2) > T4 (1) > T2 (0 flips, 2 knife-edge). **Ranked by expected cost:** T2 < T1 < T5 < T4 < T3-lite < T3-full.
- **Under $100, the verdicts-per-dollar order is T1, then T2, then T3-lite.** Each llama-bench sweep is short (~14 minutes for 31 configurations on the A100 anchor), so rental time is dominated by setup: build, model download of 32–250 GB per model, and page-cache warm-up.
- **A single T1 session with ≥ 192 GB RAM covers the most ground:** SeqMoE's Qwen3.6 pair, both FreeToken server rows, and pipeshard.
- **If FreeToken's rows became adjudicable, the headline could rise from 8 of 22 toward about 13 of 28.** Five of the six have predicted S_n lower edges > 1.0 (1.09–1.90), and all six reported llama.cpp baselines are at strength. Their m\* survive values at mid are 1.37–2.86.
- **Host-bandwidth sensitivity is steep.** On the SeqMoE 5090 Qwen3.6-15 % row, predicted llama.cpp moves from 32.4 tok/s at 150 GB/s to 46.9 at 300 GB/s (my predictor run). A VM slice reaching only ~50 % of datasheet would under-state the baseline and inflate survival. So either rent a full-socket host or apply the slope rescaling.

### Gaps
- No rental prices, availability or per-instance host RAM, CPU model and memory-channel data were gathered. Web research was out of scope. Whether RTX 5090 or RTX 4090 cloud instances come with ≥ 128–192 GB RAM on 8-channel DDR5 Xeons, and the cost of a full dual-socket node with ≥ 512 GB, must be checked separately.
- Lambda capacity was a documented problem: no A100 for hours, and an 80 GB card substituted for the 40 GB ([PROTOCOL.md 4.7 outcomes](prereg/PROTOCOL.md)).
- How to weigh settling a "not established" knife-edge row against settling a flipping row is a judgment call for the author.

## For which rows is a head-to-head (system's own code next to llama.cpp) possible, and what would it need?

### Takeaway
Code is linked for six systems:

- KTransformers
- HybriMoE
- the CPU-GPU collaborative system (cpugpucollab)
- pipelined sharding (pipeshard)
- FreeToken
- Fiddler

Three head-to-heads fit consumer-GPU rentals and touch the uncertain or near-miss verdicts: FreeToken on an RTX 5090/4090 with Qwen3.6 BF16, cpugpucollab on an RTX 4090 with Phi-3.5-MoE 16-bit, and pipeshard on an RTX 5090. SeqMoE, SP-MoE, ProMoE, Fate and MoE-SpeQ have no code link. For SeqMoE and SP-MoE I confirmed this in the arXiv HTML. So the three flipping SeqMoE rows can be settled only by measuring the llama.cpp baseline, not by head-to-head.

### Cited Findings
- **Code links in the raw rows** ([rows_group1–3.jsonl](data/audit/rows_group1.jsonl)):
  - KTransformers: https://github.com/kvcache-ai/ktransformers
  - HybriMoE: https://github.com/PKU-SEC-Lab/HybriMoE
  - CPU-GPU collaborative inference (arXiv 2512.16473): https://github.com/elsa-lab/MoE-CPU-GPU-Collaborative-Inference
  - Pipelined sharding (MLSys 2026 artifact): https://github.com/deepshnv/pipeshard-mlsys26-ae
  - FreeToken (arXiv 2608.16157): https://github.com/FlashML-org/FreeToken
  - Fiddler: https://github.com/efeslab/fiddler
- **Rows with `code: null`:** SeqMoE, SP-MoE, ProMoE, Fate and MoE-SpeQ. The llama.cpp community rows (discussion #24528 forks; PR #27861) are code in llama.cpp forks and PRs ([rows_group*.jsonl](data/audit/rows_group1.jsonl)).
- **SeqMoE v1 HTML:** its only GitHub links are to ggml-org/llama.cpp, vLLM's fused_moe.py, DeepGEMM and a SYCL graph guide. No SeqMoE repository ([arXiv 2609.12978v1 HTML](https://arxiv.org/html/2609.12978v1); search results also showed none: [arXiv abs](https://arxiv.org/abs/2609.12978), [Pith](https://pith.science/paper/2609.12978)).
- **SP-MoE v2 HTML:** contains no repository link ([arXiv 2510.10302v2 HTML](https://arxiv.org/html/2510.10302v2); [abs](https://arxiv.org/abs/2510.10302)).
- **FreeToken's runs:** exact routing ("FreeToken keeps the routed computation exact and the model unmodified"). Thread-capped servers (6 threads, 8 for DSV4, NUMA-pinned). Its llama.cpp baseline is described only as a "routing-blind static split" assigning "whole layers to devices" ([rows_group*.jsonl](data/audit/rows_group1.jsonl) notes, [normalized_v2.jsonl](data/audit/normalized_v2.jsonl)).
- **Pipeshard's llama.cpp baselines** were derived from the artifact CSV: `-ngl 20` / `1` / `36`, default mmap, threads = physical cores ([normalized_v2.jsonl](data/audit/normalized_v2.jsonl)).
- **cpugpucollab's Fiddler baseline (0.2 tok/s)** was flagged "implausibly low … treat as broken". Its claimed speed-up is over the authors' own CPU-only mode ([normalized_v2.jsonl](data/audit/normalized_v2.jsonl)).
- **KTransformers is itself a baseline** in SeqMoE (34.2 / 25.9 / 35.0 / 26.9 / 14.8 tok/s), FreeToken (32.5 / 10.4 / 34.8 / 31.8) and HybriMoE rows ([normalized_v2.jsonl](data/audit/normalized_v2.jsonl)).

### Inferences
- **FreeToken head-to-head on T1/T2:**
  - Needs: Qwen3.6-35B-A3B BF16 (69.4 GB), RTX 5090 (or 4090 for the 4090srv row), ≥ 96 GB RAM, a DDR5 Xeon with 6 threads pinned to the GPU's NUMA node.
  - DS-V4-Flash on the RTX 5090 at 8 threads if the host has ≥ 192 GB.
  - Running FreeToken, llama.cpp `--n-cpu-moe` and llama.cpp `-ngl` (FreeToken's described baseline) on one box separates "weak baseline" from "real gain".
  - It also cross-checks the FreeToken numbers SeqMoE reports on its 5090 platform.
- **cpugpucollab head-to-head on T2:** Phi-3.5-MoE 16-bit (84 GB), RTX 4090, ≥ 96–128 GB RAM, 24 threads, and ideally a 4-channel DDR5 workstation. This decides the m\* = 1.01 row directly.
- **Pipeshard head-to-head on T1:** Qwen3-30B-2507 Q4_0 and Qwen3-235B Q2_K, RTX 5090, ≥ 128 GB, 16 host cores. The artifact already contains the llama.cpp baseline scripts.
- **KTransformers head-to-head** needs the T3 dual-socket host. KT runs on T1/T2 would only re-check the KT numbers other papers report, not the KT rows.
- **HybriMoE and Fiddler head-to-heads are possible but low value.** HybriMoE's rows are robust (m\* ≤ 0.50). Fiddler's rows are not adjudicated and need 16-bit Mixtral (93 GB) on old Turing/Ada 24–48 GB cards.

### Gaps
- I did not verify that the six linked repositories resolve, build, or support the models and GPUs above. In particular, KTransformers' x86 CPU-kernel ISA requirements, FreeToken on Blackwell/RTX 4090, and cpugpucollab on 16-bit Phi-3.5-MoE are unverified.
- ProMoE, Fate and MoE-SpeQ were not re-checked on the web for later code releases. Their rows are not adjudicable anyway: claims < 1.2x (ProMoE), lossy (Fate), band 2.7–3.6x with no llama.cpp baseline (MoE-SpeQ).

## Which rows are impossible or impractical under the constraints, and what is the cheapest substitute that still tests the same verdict?

### Takeaway
Two rows are impossible under $100:

- **KT DeepSeek-V3 BF16:** 1,342 GB of weights on a dual-socket host.
- **FreeToken GLM-5.2:** ~445 GB on a 96 GB card, 512 GiB host.

KT DeepSeek-V2.5 BF16 (471 GB, needs ≥ 512–640 GB) and KT DeepSeek-V3 int4 (377 GB, needs ≥ 512 GB) are impractical on a dual-socket node. The SeqMoE RTX PRO 6000 rows and KT's RTX 4080 rows face GPU-availability issues that substitute cards (A100 80GB; A10/4090) can cover with a model-based GPU correction.

The cheapest substitute that still tests the KT verdicts keeps the host class and shrinks the model. Qwen2-57B BF16/Q8_0 and DeepSeek-V2.5 Q8_0 on the same dual-socket DDR5 host give the per-CPU-layer slope, and the rows' prediction can then be rescaled by bytes. The GH200 is a cheaper diagnostic that tests only the DeepSeek-architecture ratio, not NUMA.

### Cited Findings
- **Model sizes and host bandwidth for the hard rows:**
  - DeepSeek-V3 BF16 weights: 1,342 GB; int4 at 4.5 bpw: 377 GB.
  - DeepSeek-V2.5 BF16: 471 GB; Q8_0: 250 GB.
  - GLM-5.2 NVFP4: ~418–445 GB.

  (Computed with [mosl/archs.py](mosl/archs.py); DeepSeek-V2.5 Q8_0 is 250.6 GB on [Hugging Face](https://huggingface.co/bartowski/DeepSeek-V2.5-1210-GGUF).) The KT host had "DDR5, 1 TB per socket (2 sockets)", with MLC 220 GB/s intra-socket and 125 GB/s cross-socket ([rows_group*.jsonl](data/audit/rows_group1.jsonl)).
- **FreeToken GLM-5.2:** RTX PRO 6000 Blackwell 96 GB; Xeon Platinum 8559C, 48 threads; DDR5 512 GiB; budget imputed at the whole 96 GB card ([normalized_v2.jsonl](data/audit/normalized_v2.jsonl)).
- **KT rows keep all routed experts on the CPU** (0 experts per layer on the GPU), and the llama.cpp baseline was a custom fork ("authors extended llama.cpp for Fiddler-style expert offload … no version/flags/threads/NUMA reported") ([normalized_v2.jsonl](data/audit/normalized_v2.jsonl)).
- **DeepSeek-V2-Lite is over-predicted on both anchors** (measured/predicted down to 0.44–0.49). The two weak calls on its rows hold at the corrected prediction ([paper.tex](paper/paper.tex); [basis.json](prereg/anchors/basis.json)).
- **The GH200 is outside the fitted envelope.** B_p does not enter `--n-cpu-moe` predictions there, and its offloaded configurations were mostly under-predicted (0.55–1.50, STREAM basis) ([PROTOCOL.md 4.7 outcomes](prereg/PROTOCOL.md)).
- **GPU substitution effects** (my predictor run with the audit's fitted constants):
  - RTX 4080 → RTX 4090: +14 to +21 % for KT int rows.
  - RTX 4080 → A10: −7 to −11 %.
  - RTX PRO 6000 → H100 SXM: +25 to +28 %.
  - RTX PRO 6000 → RTX 5090: 0 % (same 1,792 GB/s; the 5090's 32 GB is too small for the 74 GB SeqMoE DS-V4-Flash configuration).
  - A100 40GB → A100 80GB: +8 to +11 %.

### Inferences
- **KT DeepSeek-V3 BF16 (impossible).** Its uncertain verdict is the weak flag (s\* 1.10); its survival sits on a knife edge (m\* 1.04). Cheapest substitute: measure on the same dual-socket host class, and derive the DeepSeek-V3 prediction from the measured per-CPU-MoE-layer slope scaled by the BF16/Q4 byte ratio. Two runs provide the slope:
  - DeepSeek-V3 Q4_0 (377 GB, ≥ 512 GB host), for the architecture's per-layer overhead;
  - DeepSeek-V2.5 BF16 or Qwen2-57B BF16, for BF16 CPU-kernel efficiency.

  This still tests the same verdict but adds a bytes-scaling assumption, so report it with a wider u (e.g., 25 %).
- **KT DeepSeek-V2.5 BF16 (impractical: ≥ 512–640 GB).** Substitute: DeepSeek-V2.5 Q8_0 (250.6 GB) on the same host, rescaled by bytes (x1.88), plus Qwen2-57B BF16 vs Q8_0 on the same host to measure how BF16 kernel efficiency differs from Q8_0.
- **KT DeepSeek-V3 int4 (impractical: ≥ 512 GB dual-socket).** Substitute: DeepSeek-V2.5 Q4_K_M (~143 GB at 4.85 bpw) or Q8_0 for the per-layer slope, scaled to DeepSeek-V3's active routed bytes (58 x 8 x 44M vs 59 x 6 x 23.6M parameters).
- **All KT rows on a budget.** The GH200 (already provisioned for anchors) can measure all-experts-on-CPU ratios for Qwen2-57B BF16/Q8_0, DeepSeek-V2.5 Q8_0 and probably DeepSeek-V3 Q4_0 within 480 GB. This tells whether the DeepSeek over-prediction seen on DeepSeek-V2-Lite carries over to the larger DeepSeek models, which decides the direction of 4 KT flips. It cannot answer the two-socket question and is outside the x86/PCIe envelope, so it narrows the KT verdicts but does not settle them.
- **SeqMoE RTX PRO 6000 rows.** Substitute GPU: A100 80GB (2,039 GB/s, closest bandwidth among ≥ 80 GB cards), with a model-based GPU correction of about −8 % back to 1,792 GB/s. The DS-V4-Flash survival verdict (m\* 1.50) is robust to that correction unless measured/predicted exceeds ~1.4.
- **KT RTX 4080 rows.** Substitute: any 16 GB-plus card near 717 GB/s. An RTX 4080 (SUPER) is ideal; an A10 (600 GB/s) is the nearest datacenter card, with a +7–11 % correction. A 4090 over-states the baseline by 14–21 %, which is larger than several KT margins (e.g., ds2-int8 m\* weak 0.80), so it needs a correction.
- **FreeToken GLM-5.2.** No cheap substitute tests the same verdict. It needs a 753B model on a 96 GB card with a 512 GiB host. Leave it not adjudicated.
- **Desktop and laptop-host rows** (FreeToken 9950X3D desktop and 4060 laptop; ProMoE i9-14900K; Laguna 270K) need consumer-desktop hosts with two-channel DDR5. They are not adjudicable or are near-miss only, and are low priority.

### Gaps
- Not checked: whether any cloud offers a dual-socket DDR5 host with ≥ 512 GB and a single A100/RTX 4080-class GPU as a whole node (not a VM slice), and at what price.
- The size of the bytes-scaling error from Q8_0/Q4_0 to BF16 on llama.cpp's x86 CPU path is untested. No first-party BF16 CPU-offload run exists in the anchors, which cover Q4_K_M and Q8_0 only ([PROTOCOL.md 4.7](prereg/PROTOCOL.md) configurations).
- Whether DeepSeek-V3 Q4_0 fits in GH200's 480 GB with OS and page-cache headroom is not verified.

[^1]: Job script `jobs/051_anchor_sweep@anchor.sh` and `results/051_anchor_sweep@anchor/gpu_monitor.csv` on the repository's `origin/gpu` branch (read with `git show origin/gpu:<path>`); no public URL.
