# How our system differs from the others

*28 September 2026. Sources: the research notes in `research_notes/MoE offload system race plan/` (race_entrants, llamacpp_ecosystem, synthesis, our_system_baseline) and `reports/MoE hybrid decode novelty check.md`.*

## What our system does, in plain words

A mixture-of-experts (MoE) model has many small sub-networks per layer, called **experts**, and uses only a few of
them for each token. When the model is bigger than the GPU's memory, most experts live in the computer's main memory
(RAM). Our system is a change to llama.cpp that keeps a **cache** of experts on the GPU, separately for every layer.
When a token needs an expert that is on the GPU (a **hit**), the GPU runs it. When it needs one that is not (a
**miss**), the GPU writes a short request into a small shared memory area (the **mailbox**), and CPU **helper threads**
that are always waiting run the missing experts and write the answer back. While they work, the GPU runs the hits.
Experts that are used often are copied to the GPU over time (**admission**), using a rule that favours experts used
recently and often (**decayed-frequency admission, DFA**). On an NVIDIA A10 this was 1.31–1.66× faster than llama.cpp's
standard `--n-cpu-moe` setting at the same GPU memory, and reached 51–72% of the speed limit our paper derives.

## Mechanism by mechanism

"Same" means the same idea, "similar" a close variant, "different" a genuinely different choice.

| Mechanism | Ours | FreeToken (Aug 2026) | KTransformers (SOSP'25) | Pipelined sharding (NVIDIA, MLSys'26) | llama.cpp community caches (leloch RFC #24528, PRs #26563, #27861) | HybriMoE (Apr 2025) / 2512.16473 (Dec 2025) | Mainline llama.cpp `--n-cpu-moe` |
|---|---|---|---|---|---|---|---|
| Where experts live | GPU cache + RAM | GPU cache + RAM — same | Hot experts on GPU, rest in RAM — similar | Layers split between GPU and CPU — different | GPU cache + RAM — same | GPU cache + RAM — same | Whole layers' experts in RAM — different |
| Cache organisation | **Per layer**, fixed slots per layer | **One pool for all layers** — different | Fixed placement chosen from a profile — different | No cache — different | Per layer — same | Per layer (2512.16473: set-associative) — same | No cache |
| What happens on a miss | CPU runs the expert; with FETCH (optional, 28 Sep) a table says how many of a layer's misses are copied to the GPU and run there in the same step, the rest on the CPU | Splits misses between copying to the GPU and running on the CPU, sized from measured bandwidths — different | CPU runs it — same | CPU runs its layers — similar | Mostly copy to the GPU, some CPU — different | CPU runs it while the expert is copied — similar | CPU runs the whole layer's experts |
| How the CPU is told to start | GPU writes a mailbox; helpers spin; a GPU spin kernel waits for the answer, all inside the step's CUDA graph | GPU raises a flag with stream memory operations; a CPU thread polls; the GPU waits on a "done" flag inside the CUDA graph — **same idea, since 11 Aug 2026** | `cudaLaunchHostFunc` callbacks inside a CUDA graph — similar | llama.cpp's scheduler — different | llama.cpp's scheduler — different | Host-driven scheduling — different | llama.cpp's scheduler (CPU and GPU take turns) |
| Prefetch of predicted experts during decode | Optional PREFETCH (28 Sep): the next layer's router on this layer's MoE input picks one expert to copy on a side stream | Not in its decode path (its prefetch double-buffers the next layer for prefill) | Not found | n/a | Not found | HybriMoE prefetches — similar | n/a |
| Skip the CPU when a layer has no misses | GPU decides per layer | Not found in its code | Not found | n/a | Not found | Not found | n/a |
| CPU and GPU at the same time within a layer | Yes | Yes — same | Yes — same | Pipelined across layers — different | Partly | Yes — same | No (they take turns) |
| How admission copies travel | A GPU kernel reads pinned RAM directly, keeping copies off the copy engine | Copy engine | Copy engine | n/a | Copy engine, with throttling or low priority | Copy engine | n/a |
| Admission rule | DFA with a margin κ | LRU over the pool | Static from profiling | n/a | #26563: heat decay with a 1.3× margin — same family | Custom scores | n/a |
| Exact outputs | Yes (same arithmetic up to kernel rounding) | Yes | Yes with deferral off (its "expert deferral" is lossy) | Yes | Yes | Yes | Yes |
| Engine and language | C++ inside llama.cpp, GGUF files | Python engine with C++/CUDA extensions, Hugging Face weights | C++ kernels with SGLang | Fork of llama.cpp b6097 | llama.cpp | Research prototypes | llama.cpp |
| A proven speed limit to compare against | **Yes** (our bound) | No | No | No | No | No | No |

## What is genuinely different about ours

These are implementation choices, not new ideas. Each is worth measuring rather than claiming.

1. **The GPU skips the CPU on layers with no misses.** When every expert a token needs is already on the GPU, the GPU
   does not wait for the helpers at all. We found this in no other system's code.
2. **Admission copies travel through a GPU kernel instead of the copy engine.** The copy engine is the GPU's separate
   unit for moving data. On the A10 we measured that bulk admission copies queue ahead of the step's own small, urgent
   transfers. Reading pinned RAM directly from a kernel keeps them out of the queue. Others throttle or deprioritise
   their copies instead.
3. **The cache is per layer, and the per-layer budget has a proven speed limit.** Our bound applies exactly to this
   class of policies, so we can say how far from optimal each configuration is. FreeToken's shared pool is a different
   class; the bound does not cover it, and the paper must say so.
4. **It lives inside llama.cpp**, the engine most people run at home (Ollama and LM Studio build on it), and reads the
   same GGUF files.

## What is not different, and whose idea it was

- A GPU expert cache whose misses run on the CPU: HybriMoE (April 2025), 2512.16473 (December 2025), DALI
  (February 2026), KTransformers (2025), FreeToken (August 2026), and llama.cpp contributors' branches from June 2026
  (leloch, #26563, #26824, #27861).
- A GPU-signalled CPU hand-off polled by a CPU thread, inside a CUDA graph: FreeToken, public since 11 August 2026.
- Running the whole decode step as one CUDA graph with the CPU work inside it: KTransformers (SOSP'25).
- Decayed-frequency admission with a margin: llama.cpp PR #26563 (August 2026).
- Loading from mapped host memory with GPU kernels: SeqMoE (September 2026).
- Splitting a layer's misses between copying and CPU execution (our FETCH): FreeToken's hybrid mode (August 2026),
  sized from measured bandwidths, and HybriMoE (April 2025).
- Predicting the next layer's experts from the current hidden state and prefetching them (our PREFETCH): Pre-gated MoE
  (ISCA'24), ProMoE (2410.22134), Fate (2502.12224), HybriMoE, DALI (2602.03495), Speculating Experts (2603.19289),
  SeqMoE (2609.12978). Fate uses the next layer's gate on the current gate input, as we do.

## What makes the study itself different

The study, not the system, is where the work stands out:
- **A speed limit in seconds** for the whole class of per-layer policies, so every system is reported as a fraction of
  what the hardware allows.
- **Equal GPU memory for everyone**, measured on the machine rather than taken from each tool's own flag.
- **Rules and predictions written down before measuring** (pre-registration), including the prediction that a
  pooled, copy-heavy design like FreeToken may do best on a home PC where the GPU link is about as fast as main memory.
- **Every system's remaining time broken down** into where it goes, and a practical table of which engine and
  settings to use for a given GPU.

## Suggested related-work paragraph for the paper

> Our case-study system combines ideas that other groups introduced first. GPU expert caches whose misses execute on
> the CPU appear in HybriMoE [2504.05897], the set-associative design of [2512.16473], DALI [2602.03495],
> KTransformers [SOSP'25] and FreeToken [2608.16157], and in several llama.cpp contributors' branches (RFC #24528,
> PRs #26563, #26824 and #27861). FreeToken has shipped a GPU-signalled, polled CPU hand-off inside a CUDA graph since
> August 2026; KTransformers places CPU expert work inside a CUDA graph through host callbacks; decayed-frequency
> admission with a margin is in PR #26563; and SeqMoE loads experts from mapped host memory with GPU kernels. We use
> the system only as one of several systems measured under the same rules. Its differing implementation choices (a
> per-layer skip of the CPU decided on the GPU, admission copies that bypass the copy engine, and a per-layer budget
> that our bound covers exactly) are reported with their measured effect where we ablated them, and claimed nowhere
> else.
