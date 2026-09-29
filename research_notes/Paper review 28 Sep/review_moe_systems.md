# Review: "Seconds, Not Blocks" (draft of 28 Sep 2026), from an MoE-offloading and caching-theory reviewer

Reviewer stance: MLSys 2027 PC member who works on expert offloading (HybriMoE, KTransformers, FreeToken, SeqMoE, Pre-gated MoE, MoE-Infinity) and on offline caching theory (Belady MIN with bypass, Demand-MIN, Cao et al. 1995, Albers–Garg–Leonardi 2000, CHOPT 2020). Line numbers refer to `paper/paper.tex` as of 28 Sep 00:35. My own checks and recomputations are marked **[checked]**. Their scripts are in this session's scratchpad (`.../scratchpad/review/bounds.py`, `audit_aggr.py`, `joint.py`). Each is a few dozen lines on top of `mosl/`, and the method is described inline so the numbers can be regenerated.

## 0. Summary verdict

**Recommendation as submitted: weak reject. After the fixes below: a clear accept, and a paper people will cite.** The idea is right and overdue: a policy-independent floor in seconds, a strong equal-memory baseline, and an audit that places published systems between the two. The pre-registration discipline and the "pitfalls" section are unusually honest. The problems are fixable, but three of them go to the core.

1. **Proposition 1 is not a lower bound "for any exact-routing policy" (l. 20, 36, 68).** The combinatorial half is correct and can be made rigorous; I give the proof below. The timing half, however, assumes a per-layer serial structure: dense work, then max(GPU hits, CPU misses). Real policies escape it. KTransformers' own mechanism, running the shared expert on the GPU while the CPU runs routed experts, is one way out. A lookahead policy that streams predicted CPU experts into the CPU's LLC during the GPU's attention phase is another. On the race host at 11% of experts, an assumption-light version of the bound is **29% faster** than Prop. 1 (225 vs 174 tok/s) **[checked]**. In the audit, KTransformers' "77% of SoL" becomes **41%** **[checked]**. The fix is a resource-form bound (§1.4). It is simpler, needs no Jensen step and is unimpeachable. The current form can stay as a refinement under stated assumptions.
2. **Several denominators are not bounds.** The race's "speed-of-light" and the A10 "implementation-relative" SoL use measured bandwidths, and the paper's own A100 anchor showed a measured floor being beaten by 16% (A1). M\* from independent uniform routing (the "(i)" rows) is not a lower bound on misses. The per-layer bound is applied to policies that are not per-layer: FreeToken's pooled cache, llama.cpp `-ncmoe` (whole layers resident) and Fiddler's popularity placement. Speculative systems (SP-MoE) are adjudicated against a bound that does not apply to them.
3. **The headline audit numbers are fragile and partly mis-attributed.**
   - The "median 24% of SoL" is 19%, 24% or 28% depending on whether one row is included (the no-baseline row, or the speculative SP-MoE row).
   - The "claimed up to 4.2×" (l. 28) is the auditor's ratio against llama.cpp. SeqMoE's own figure also contains FreeToken, against which the same row is 1.38×.
   - 7 of the 16 "weak" calls are baselines that predate `--n-cpu-moe`.
   - The one "uniform rule" on measured host bandwidth was applied to FreeToken rows only, not to KTransformers, whose paper reports MLC 220 GB/s per socket.

The first-party race (not yet in the draft) is the most distinctive new evidence, but as run it cannot carry any claim beyond "+8–12% at 11% of experts on one host" (§4).

---

## 1. The bound (Proposition 1, l. 66–75)

### 1.1 What Proposition 1 actually assumes

The draft states almost none of these. Every one must appear in the proposition or in an assumptions list right before it.

| # | Assumption | Where it is used | Stated? |
|---|---|---|---|
| A1 | Batch 1, one token per forward pass: no speculative, draft-verify or MTP multi-token verification | per-token byte accounting | No. SP-MoE and MoE-SpeQ are in the audit (`speculative: true` in `normalized_v2.jsonl`), and SP-MoE is one of the 22 adjudicated rows |
| A2 | Exact routing on the stored bytes; bytes per expert `s` are the same in every location (no lossless recompression on the link or CPU, no cheaper CPU-side format) | a, b, p all use one `s` | No. Mixed-format systems (KTransformers' CPU format vs GPU Marlin) need bytes per location |
| A3 | Dense work `D` runs on the GPU, from GPU memory | T_D = D/B_g | No. Splitting the LM head or attention to the CPU is legal and aggregates bandwidth |
| A4 | **Per layer, dense work of layer l precedes, and does not overlap, the routed-expert work of layer l, and the CPU reads a routed expert's bytes only after the router selects it** | T_D + L·ℓ(m) | No. Violated by shared-expert overlap (KTransformers) and by LLC prefetch (§1.3) |
| A5 | Only three resources, CPU read, link and GPU DRAM, each capped by its own peak, with no joint host-DRAM cap | ℓ(m) and L x p | Implicit. Valid but loose: on a datasheet basis it lets CPU plus DMA draw B_c + B_p ≈ 1.8× the DRAM peak (§1.3) |
| A6 | Budget counts resident copies; transient staging and zero-copy reads use no budget but do use the link | x | No. Zero-copy (FreeToken's gather kernel, llama.cpp `CUDA_Host`) must be counted in x |
| A7 | Cache contents at the start of the measured window are paid for inside the window | H ≤ G\* + Λ | No. Warm-up and prefill-time loads are free in every measured tok/s (§1.2c) |
| A8 | Equal expert sizes within the budget scope | MIN-bypass optimality is for unit sizes | No. Pooled budgets over Q4_K_M GGUFs (Q6_K `ffn_down` in some layers) have unequal sizes, and unequal-size bypass caching is NP-hard |
| A9 | M\* is computed on the routing of the *measured* text | validity | Partly. The audit uses the paper's corpus for other people's workloads, and uniform routing for the "(i)" rows |
| A10 | Peaks are true upper bounds (datasheet) | validity | Yes for the "physical" version (l. 75). **No** for the measured-bandwidth and implementation-relative versions, which the draft and the progress log also call speed-of-light |

### 1.2 Step-by-step check of the proof sketch (l. 73)

**(a) Per-layer MIN-with-bypass counts. Correct, and provable without the brute-force test.** `cachesim._sim` implements Belady with bypass correctly for set-valued steps:
- victims are the furthest next use, and experts served this step are eligible;
- a miss is admitted only if it is needed sooner than the victim;
- never-reused experts are not admitted.

`tests/test_cachesim.py` checks it against exhaustive search only for E ≤ 6, k ≤ 2, T ≤ 8. Give the real proof. Every GPU hit on expert e at step t, whose previous request was at t′ < t, is either:
- a **bridged gap**: e resident at every step in (t′, t]; or
- charged to a **load** of e in (t′, t], with each load charged at most once.

Bridged gaps are intervals on the step axis, and at most C (or LC) may overlap at any step. The maximum number of bridged gaps is therefore the maximum C-colourable subgraph of an interval graph, which is solved exactly by the greedy that drops the furthest-ending interval. That greedy *is* Belady-with-bypass (Yannakakis & Gavril 1987; Carlisle & Lloyd 1995; please verify the citations). The constraint matrix has the consecutive-ones property and is totally unimodular, so **fractional residency (an expert's rows split between GPU and host) cannot bridge more gaps than the integral optimum.** This closes the "split one expert's rows between CPU and GPU" loophole for the count. Loads must then also be counted fractionally.

**(b) The prefetch reduction. Correct.** "A prefetching schedule bridges no more gaps than some demand schedule with bypass" holds by truncation. Shrink every residency interval to [first request, last request] inside it. Occupancy only falls, so the budget still holds, and the gap count is unchanged. So H ≤ G\* + Λ. The novelty recheck's worry that this "runs into Jain & Lin" (Demand-MIN) is unfounded: Demand-MIN matters when prefetches are free, and here each load is charged through x and the link term. Cite Jain & Lin to show awareness, then write out the charging argument above. The sketch as written is not checkable.

**(c) Loophole: the initial state is free.** With a cold start, H ≤ G\*\_cold + Λ + (resident experts at window start). Every measured tok/s in the paper and in the race excludes prefill, and FreeToken's client warms each problem with the *same* prompt and the *same* 256 sampled tokens before the measured request. That is visible in `064…/srv_ft_*.log`: 109 new prefill tokens, then 1 new plus 108 cached about 8 s later. Fix: compute M\* with the cache contents chosen freely at the start of every measured segment. For Belady the optimal initial set is the C experts with the earliest first use; implement it as a synthetic prefix whose misses are not counted. **[checked]** On gpt-oss-120b (own text, 35 segments) per-layer M\* moves 0.953 → 0.946 at C = 14 and 0.139 → 0.123 at C = 56. That is small here, because the bound sits at the aggregation optimum, but it is necessary for validity. It matters more for short windows (192 or 256 tokens) with warm-up on identical text.

**(d) Load accounting across GPU DRAM, host DRAM and PCIe.**
- *Zero-copy execution* (a GPU kernel reading host memory, with no resident copy) is covered only if x counts **all link bytes of expert weights**, retained or not. Say so. The per-layer-step cost then stays valid as long as p ≥ k·a (link time per expert ≥ GPU DRAM time for k experts). That holds on every platform in the paper (p/a ≈ 28 on the 5090, ≈ 65 on A100 PCIe 4) but is marginal on the GH200 (C2C ≈ 450 GB/s vs 4 TB/s HBM: p/a ≈ 9, with k = 8 for Qwen3). Either state p ≥ k·a or add zero-copy as a third per-layer-step path, ℓ(m, z) = max((k−m−z)a, mb, zp).
- *DMA writes into GPU DRAM* are dropped. That is valid; adding L x s to the GPU term would be a valid tightening.
- *Joint host DRAM* is absent. That is valid, but see §1.3 on why it matters so much.

**(e) Jensen step. Correct.** ℓ(m) = max((k−m)a, mb) is convex, so mean ℓ(m\_i) ≥ ℓ(m̄) ≥ min over m ≥ m\_lo of ℓ(m). In the resource form of §1.4 no Jensen step is needed at all, because every constraint is linear in the totals.

**(f) The timing structure. Not valid for "any policy".** See §1.3, rows 1–3.

**(g) Shared all-layer variant.** It is correct: MIN-bypass on the interleaved stream with LC slots, and a per-layer schedule is feasible under the pooled budget, so M\*\_glob ≤ M\*\_layer. Two issues remain.
- **(i)** The per-layer bound applies *only* to policies that respect a per-layer cap. llama.cpp `-ncmoe` keeps whole layers (128 of 128 experts in 4 layers, 0 in the others), and Fiddler and ProMoE place by popularity, unevenly across layers. These are pooled-budget policies. The case-study column "llama/SoL" (Table `tab:a10`), the race's llama.cpp fractions and `audit.py` (`system_of_sol` always uses the per-layer `t_sol`) all apply the per-layer bound to them. **Use the pooled bound for every system in headlines.** It is valid for all of them, and per-layer is a refinement only for verified per-layer caches.
- **(ii)** Unequal expert sizes (A8) need a byte-weighted LP relaxation.

**(h) Consequence (i) (l. 75) is misattributed.** Budget independence is said to hold "when M\*/N is below ka/(a+b)". On the race host at C = 32, ka/(a+b) = 0.12 while M\* = 0.386, yet the bound equals its M\* = 0 value (313 tok/s) **[checked]**. What makes the bound budget-independent is the **clairvoyant link**: x ≈ 0.26 loads per layer-step, fully overlapped, pull M\*/N − x down to the aggregation optimum. State the condition correctly: there exists x with (M\*/N − x) ≤ ka/(a+b) and L·x·p ≤ T\_agg. State the reading too. On PCIe 5 hosts the "speed-of-light" is essentially **all-in-VRAM speed**: 313 vs 311 tok/s for gpt-oss-120b on host 064 **[checked]**. It is reached only by a perfect predictor streaming about a quarter of an expert per layer-step. A reader who takes "24% of SoL" to mean "76% waste" will be misled unless this is said.

**(i) `tests/test_bound.py` tests the combinatorics, not the physics.** Every candidate policy is scored with `dynamic_time`, the same layer-structured cost model that the bound relaxes. "No policy beats it" (l. 75) therefore means "no simulated policy under the bound's own cost model". Say that. The physical evidence is the measured systems: the anchors reach ≤ 54% of the datasheet floor, and the race systems ≤ 40% of the measured-basis reference.

### 1.3 Counterexamples and loopholes

| Candidate | Beats Prop. 1 as written? | Magnitude | Fix |
|---|---|---|---|
| **1. Shared-expert overlap.** The GPU runs the shared expert of layer l while the CPU runs its routed experts. This is KTransformers' published design and applies to Qwen2-57B (shared = 8 routed), DeepSeek-V2/V3 and Qwen1.5-MoE | **Yes.** Shared-expert bytes are in T\_D, which is added serially | KTransformers rows **[checked]**: 48–77% of Prop. 1 becomes 29–44% of the resource form. The best row (Qwen2-57B, RTX 4080) goes from 77% to 41% | Resource form, or split D into D\_pre (attention, router) and D\_par (shared experts) with D\_par inside the max |
| **2. CPU-side prefetch into LLC.** A lookahead predictor pulls layer l+1's predicted CPU experts into L3 while the GPU runs attention. 64 MB of L3 on the 9950X (96 MB on the X3D) holds several 13 MB experts | **Yes** (A4). The bound already grants clairvoyance to the link, so granting it to the CPU's DRAM path is consistent | Race host, C = 14 **[checked]**: 174 (Prop. 1, per-layer) → 183 (pooled) → **225** (resource form, pooled). About 2% at C ≥ 32. The gap is ≤ T\_D and ≈ T\_D·B\_c/(B\_g+B\_c) when the link does not bind (≈ 20% of T\_D on the A10) | Resource form |
| **3. Dense split.** The LM head or some attention heads run on the CPU from host copies | Yes (A3) | Negligible on the 5090 (B\_c/B\_g ≈ 3%). Up to about 20% of T\_D on the A10 | Minimise over D\_c ∈ [0, D] (dense-split form), or state A3 |
| 4. A resident expert executed on the CPU | No. m ranges over [m\_lo, k] | – | – |
| 5. One expert's rows split between CPU and GPU (or CPU and link) | No for the count (total unimodularity, §1.2a). Note that the fractional optimum is *attainable* only by row-splitting misses, which no system does | – | State it. It is also a design hint: split each miss's rows between CPU and link in proportion B\_c : B\_p |
| **6. DMA and CPU reads sharing DRAM; measured concurrent 73–77 > CPU-alone 62 GB/s** | Does not break Prop. 1, which has no joint term. It **does** break any joint-DRAM constraint calibrated on CPU-alone STREAM or on a measured total: `sim_homepc.sol_time` uses `B_D` = 57 GB/s as an estimate and would be exceeded. It also breaks FreeToken's paper model (B\_R = max(B\_H − B\_P, 0)) | The measurement shows that the CPU-read ceiling on AMD desktop parts is the fabric (per-CCD GMI link), not the DRAM: B\_c^core < B\_h | Three separate caps: B\_c^core (CPU read ceiling), B\_p (link), B\_h (**datasheet** DRAM) in a joint term L(m+x)s ≤ T·B\_h. It is valid and it tightens a lot (next row) |
| 7. Datasheet basis without a joint DRAM term | Valid but badly loose at small budgets | Race host, C = 14, pooled, resource form **[checked]**: 327 tok/s without the joint term (DDR5-4800 ×2) vs **180 / 210 / 240** with it at DDR5-4800 / 5600 / 6400. The DIMM speed of the rentals is unknown | Add the joint term, record `dmidecode -t memory` per host, and report the DIMM band |
| **8. A100 anchor beating the Triad floor by up to 16%** | No modelling error in the bound's structure. It is a **basis error**: Triad (two reads and one write, with write-allocate) is not a read ceiling | The pattern in `prereg/anchors/a100.json` fits this: violations appear only when big-expert CPU work dominates (Mixtral Q4\_K\_M: 1.01, 1.05, 1.12 at n = 16, 24, 32; Q8\_0 up to 1.16; Phi-3.5 reaches 0.99 at n = 32). Small-expert models (Qwen3, DeepSeek-V2-Lite) stay ≤ 0.82 because their per-expert overhead dominates. The implied CPU rate at Mixtral n = 32 is ≈ 105 GB/s at 15 threads, against STREAM Copy of 111 | Physical bound: datasheet only. "Attainable" references: the maximum over a read-only kernel (vector sum or dot product), STREAM Copy and llama.cpp's own CPU `MUL_MAT_ID` microbenchmark, over all thread counts. **Never call a measured-basis figure a floor.** This kills "speed-of-light on this host … 174/313/313" in the progress log and "implementation-relative speed-of-light" as a bound (l. 75, 114). Also note that the registered H0 text (`prereg/PROTOCOL.md` l. 45–47) says "a violation means the byte accounting is wrong". Reading the A100 violation as a bandwidth-basis problem instead is a post-hoc reinterpretation, and it must be logged as a deviation. Discriminate the two explanations directly with CPU-side DRAM read counters (`perf stat` on the EPYC's data-fabric or UMC events) during one Mixtral n = 32 run: bytes read per token should match the GGUF accounting within about 5% |
| 9. Speculative decoding or MTP (SP-MoE, MoE-SpeQ, any Qwen/DeepSeek MTP deployment) | **Yes** (A1). Expert and dense bytes are amortised over accepted tokens | SP-MoE sits at 4%, so nothing is violated today, but the bound does not apply | Exclude with a flag, or derive a per-verify-step bound in expected accepted tokens |
| 10. Lossless recompression (DFloat11/ZipNN-style BF16 on the link or CPU) | Yes (A2) for the BF16 rows (SeqMoE and FreeToken Qwen3.6 BF16): about 30% fewer bytes | – | Define `s` per location as the bytes actually moved |
| 11. M\* from independent uniform routing ("(i)" rows) | Not a valid lower bound on misses | SeqMoE Qwen3.6 at 9.7 GB: 211 → 354 tok/s with M\* = 0 (resource form) **[checked]**, so the row goes from 33% to 20% | Use M\* = 0 (a trivially valid bound) for untraced models, or trace them |
| 12. KV bytes at an assumed ctx (640 in `analyze_homepc.py`; "ctx 4096 assumed" for FreeToken rows) | A too-long ctx makes the "bound" slower than reality | ≈ 0.3% for gpt-oss-120b; larger for long-context rows | Use the smallest ctx in the measured window |
| 13. GPU L2 persistence and CPU LLC reuse across tokens | In principle | Negligible, since the per-token working set is ≫ L2/L3 | A sentence in the scope |
| 14. Warm start / prefill-time loads | Yes (A7) | Small here (§1.2c) | Warm-start M\* per measured segment |

### 1.4 Corrected statement (draft)

> **Setting.** Batch-1 decode producing one token per forward pass (no multi-token verification), computing exactly the router-selected experts from their stored bytes (s bytes per expert, the same bytes wherever they are read), on one GPU and one host. Resources, with datasheet peaks: GPU DRAM B\_g, the CPU read path B\_c, the host-to-GPU link B\_p, host DRAM B\_h. A budget 𝓑 allows, at every instant, at most C resident experts of each MoE layer (per-layer) or LC in total (pooled), with a resident fraction of an expert's rows counted as that fraction. Staging buffers and zero-copy reads use no budget, but their bytes cross the link. Let M\*\_𝓑 be the misses of Belady's MIN with bypass under 𝓑 on the routing of the measured decode window (on the interleaved all-layer stream for the pooled budget), with the cache contents chosen freely at the start of every measured segment. Let N = TL.
>
> **Proposition 1 (resource form).** For every such policy, with x the expert bytes crossing the link per layer-step in units of s,
> T̄ ≥ min over x ≥ 0 and m ∈ [(M\*/N − x)⁺, k] of max{ (D + L(k−m)s)/B\_g , L·m·s/B\_c , L·x·s/B\_p , L(m+x)s/B\_h }.
> If dense weights also reside in host memory, replace D by D − D\_c in the first term, add D\_c to the second, and also minimise over D\_c ∈ [0, D].
>
> **Refinement (layer-structured).** If additionally (S1) the CPU reads no routed-expert bytes of layer l before layer l's router output exists, and (S2) no GPU work of layer l other than its routed experts overlaps the CPU's routed-expert work of layer l, then T̄ ≥ min over x and m of max{ T\_D + L·ℓ(m), L·x·p, L(m+x)s/B\_h }, with ℓ(m) = max((k−m)a, mb). The two forms differ by at most T\_D.
>
> **Corollary (demand-only).** If the loading of an expert starts only after it has been requested at that layer, then GPU hits ≤ G\* and every miss is run on the CPU or fetched within its layer-step: T̄ ≥ T\_D + L·min over μ ≥ M\*/N of ℓ\_d(μ), with ℓ\_d(μ) = min over f ∈ [0, μ] and h ∈ [0, k−μ] of max(a(k−μ+f−h), b(μ−f+h), p·f).
>
> **Pooled budgets with unequal expert sizes:** use byte-weighted misses and the LP relaxation of byte-weighted bypass caching.

**Proof:** the charging argument (§1.2a–b), total unimodularity for the fractional case, then one resource inequality per peak (bytes moved / peak ≤ time). The refinement adds Jensen on the convex ℓ; the corollary drops the "+Λ" term. Put the full proof in an appendix. Cite Belady 1966, Cao et al. 1995, Albers–Garg–Leonardi 2000, Jain & Lin 2018 and CHOPT 2020. Add one paragraph on "why a bound rather than an LP optimum": CPU/GPU concurrency, shared bandwidth, and the fact that an audit needs a closed form valid across execution models.

**Report a ladder, not one number**, all on the datasheet basis:

| Level | Example: race host, C = 14 | C = 32 |
|---|---|---|
| resource bound, pooled, with joint DRAM term | 180–240 | ≈ 373 |
| layer-structured refinement | – | – |
| demand-only | 149 (measured basis) | 255 (measured basis) |
| the system's own miss and load counts at ideal efficiency | – | – |
| measured | 39–48 | 66–77 |

The gaps between rungs are the paper's actual findings: clairvoyance, policy, kernel efficiency and coordination (§5, E3).

---

## 2. The audit (l. 94–107, `scripts/audit.py`, `data/audit/normalized_v2.jsonl`)

### 2.1 Fairness of the adjudication rules

1. **"Claimed" is the auditor's ratio, not the authors' claim.** l. 28 and Table `tab:audit` compute "claimed" against the strongest *llama.cpp* baseline. SeqMoE's own figures also contain FreeToken:
   - gpt-oss-120b: 101.2 tok/s, so the row is 1.38× where the audit says 4.21×;
   - Qwen3.6 at 40%: 86.2, so 1.37× where the audit says 3.12×.

   "Up to 4.2×" will read to SeqMoE's authors as putting a number in their mouth. Add a column with the authors' headline claim, quoted verbatim, next to the auditor-computed ratio, and use "reported speed-up over llama.cpp" wording.
2. **A rule applied non-uniformly.** `apply_verification.py:68–71` replaces the datasheet host band with the measured bandwidth only when `system == "FreeToken"`. KTransformers' paper reports MLC 220 GB/s per socket (125 GB/s cross-socket), yet its rows keep [563, 614] GB/s: both sockets at datasheet speed. The row notes themselves say "whether the llama.cpp baseline used both sockets is not stated". llama.cpp's cross-socket NUMA scaling is poor, so the predicted equal-memory baseline for these rows is probably too fast. Both KTransformers "weak" calls are marginal: 7.05 against a band floor of 7.9, and 9.06 against 10.1. Under a single-socket assumption they would very likely become "at strength". Apply the measured-bandwidth rule to every row that reports one, and use one socket as the band's lower edge unless NUMA use is stated.
3. **Formats llama.cpp cannot run.** SeqMoE's Qwen3-30B-A3B rows are FP8. The equal-memory llama.cpp baseline a user would run is Q8\_0 (8.5 bits), about 6% more bytes. Predict the baseline in the format the baseline engine actually supports.
4. **Metric mismatches.** Rows are converted from TBT figure bars, TPOT, end-to-end tok/s including prefill (Fiddler) and per-request averages. Put a metric column in the main table and exclude from adjudication any row whose metric includes prefill.
5. **Speculative rows.** Exclude SP-MoE (adjudicated) from the SoL statistics, or give it a separate bound (§1.3, row 9). Excluding that one row moves the median fraction from 24% to 28% under the paper's own bound **[checked]**.
6. **Two-way error, one-way band.** The band is the leave-one-source-out 10th–90th percentile (0.78–1.31) times any imputation. On the audit basis, single first-party configurations range 0.44–1.35 (l. 107). The band is therefore not an 80% interval for a given host, and single-row calls near its edge are not robust. See §2.3.

### 2.2 Era-awareness

`-ot`/`--override-tensor` (PR #11397, opened January 2025, merged in spring 2025; confirm the date on the PR page) first made partial expert offload possible. `--n-cpu-moe` (PR #15077) was merged by **5 Aug 2025** ([PR #15077](https://app.semanticdiff.com/gh/ggerganov/llama.cpp/pull/15077/overview); [announcement, 5 Aug 2025](https://x.com/2022_technology/status/1952674975841931433); [PR #11397](https://github.com/ggml-org/llama.cpp/pull/11397)). Of the **16 weak calls**:
- **7** are older baselines:
  - HybriMoE ×5: April 2025, `-ngl` baselines;
  - KTransformers ×2: a February 2025 custom fork that implements Fiddler-style expert offload, i.e. `-cmoe`, the correct equal-memory placement at budget 0.
- **9** are contemporaneous:
  - SeqMoE ×6: config unknown;
  - Pipelined Sharding ×3: `-ngl` or `-cmoe` in 2026.

What to do:
- Add a publication or experiment date and the llama.cpp build to each row (Fiddler's b2956 is already in the notes).
- Compute an **era-appropriate predicted baseline**: `-ngl` at equal *total* VRAM before `-ot`; `-ot`/`-ncmoe` after. The model needs an `-ngl` variant with attention and KV on the CPU for CPU layers; the third-party validation set already contains static-offload rows from four engines.
- Report two labels: "weak by the llama.cpp of its time" and "below today's equal-memory llama.cpp". The honest headline is probably "9 of 13 contemporaneous llama.cpp baselines are weak; all 7 older ones are below what today's llama.cpp would do". It is less punchy and far more defensible.
- For KTransformers, "weak" is a statement about the 2025 fork's CPU kernels and a two-socket host, not about configuration.

### 2.3 Is "survives" well defined?

Operationally yes: the lower edge of S\_n = system / (predicted × q90 × imputation) is above 1. Statistically no: q10–q90 is pooled over sources and says nothing calibrated about a given host. Proposals:
- **Calibrate on first-party data.** There are about 80 offloaded configurations on the audit basis (A10, A100, GH200, plus the 5090 hosts once added). Fit a two-level error model: log(measured/predicted) = platform effect + configuration residual. Report, for each row, the predictive probability that the equal-memory llama.cpp baseline exceeds the reported baseline (weak) or the system (not surviving).
- **Use three tiers**:
  - robust (reported/predicted < 0.5, or P > 0.95);
  - marginal;
  - at strength.

  The paper already says 10 of 16 weak calls are ≤ half the prediction. Those 10 are the robust ones; say so in the finding.
- **Report a range as the result.** "Survivors: 4–10 of 22 across predictor corrections" (l. 107) is already the right form. Make that range the headline, not "8 of 22".
- "Not established" must never read as "refuted". l. 96 says "No row is called wrong"; keep that, and add "not established ≠ no gain" to the table caption.

### 2.4 The SoL fraction statistic

- **The fraction does not depend on any baseline**, yet it is reported only over the 22 rows selected by baseline-adjudication rules (claim ≥ 1.2×, band ≤ ±40%). Report it for all 52 modelled rows, and per system: the row count per system (HybriMoE 5, KTransformers 6, SeqMoE 6) currently weights the median.
- **[checked]** with the valid denominator (resource form, M\* = k at C = 0, trace M\* where traced, M\* = 0 for "(i)" rows with C > 0), over the 22 adjudicated rows:

  | System | Median fraction | Range |
  |---|---|---|
  | KTransformers | 35% | 29–44% |
  | SeqMoE | 27% | 11–33% |
  | CPU-GPU collaborative | 16% | – |
  | HybriMoE | 7% | 2–8% |
  | Pipelined Sharding | 5% | 3–18% |
  | SP-MoE (excluded; speculative) | 3% | – |

  - Median over rows: **19%**, against 24% in the draft.
  - Median of per-system medians: **12%**.
  - Best row: **44%**, against 77% in the draft.

  The qualitative message ("far from the bound") gets stronger, which is good for the paper. The numbers must change.
- The draft's own median is fragile. The middle rows are 19% and 28% (l. 105); removing the no-baseline row moved it from 19% to 24% (l. 96), and removing SP-MoE moves it to 28%. Say "most systems reach 5–35% of the bound" with a per-system table.

### 2.5 Presenting the audit so audited authors accept it

- Send the drafted notes (commit `e3ab0c2`, not sent) **now**, with a two-week reply window. Include per row the command line of the predicted baseline and the host fields the prediction needs, and invite the authors to supply measured STREAM/MLC figures, DIMM configuration and their llama.cpp flags. Publish their replies in the artifact, and put a "response" column in the appendix table.
- Word every call as a comparison with a *prediction*, together with its known error on the paper's own hardware (l. 107 does this; move it into the table caption).
- Measure rather than predict wherever the hardware can be rented (experiment E7). One measured equal-memory llama.cpp run on a matching 4090 or 5090 host, scaled by the measured host-bandwidth ratio, is worth more than any band argument.
- Separate "your baseline was weak" from "your system is slow relative to the bound". The first is about their methods section; the second is not a criticism.

### 2.6 Mis-attribution risks, ranked

1. Auditor-computed "claims" (SeqMoE; §2.1.1).
2. The two-socket host treatment (KTransformers; §2.1.2).
3. Era (§2.2).
4. Metric conversions from figures (TBT, TPOT, end-to-end).
5. Budget semantics: popularity-placed or uneven budgets treated as per-layer (Fiddler, ProMoE) for the SoL (§1.2g).
6. FP8 and BF16 formats unavailable in llama.cpp (§2.1.3).
7. Speculative rows (§2.1.5).
8. M\* from the paper's corpus applied to other workloads. The provenance study itself shows hit rates moving by 1–9 points with the text.

---

## 3. Trace provenance (l. 51–56, 84–91, `table_prov.tex`)

**Sound:**
- three arms over the same prompts;
- a paired bootstrap over prompts;
- falsification thresholds registered before the results;
- extension models labelled as such.

**Confounds to control before the direction result can be called a provenance effect:**

1. **Length and position.** G and S are capped at 1,024 new tokens and D is not.
   - For gpt-oss-20b and 120b and Qwen3, the median S answer is 1,024 (mostly truncated: 129 of 160 for 20b) against 466–540 for D.
   - For Mixtral and DeepSeek-V2-Lite, S is *shorter* than D (492 vs 588; 436 vs 561).

   The direction is the same in both groups, which is real evidence against a pure length artefact; say so. Still: truncate each pair to min(len D, len S), report hit rates by answer position (first 128, 128–512, >512 tokens), and report prompt-weighted as well as token-weighted pooling.
2. **Genre for reasoning models.** For gpt-oss, the S arm is mostly analysis-channel reasoning, truncated at 1,024, while D is dataset text in the *final* channel. That compares own reasoning with human answers, not own text with dataset text. Split S into analysis and final tokens and compare D against S-final. For G, check greedy repetition loops.
3. **Sampling temperature.**
   - OLMoE and gpt-oss S use T = 1.0 and top-p 1.0: pure sampling. For OLMoE this is vLLM's fallback, since its generation config sets no sampling fields.
   - Qwen3 uses T = 0.7, top-p 0.8, top-k 20.

   Locality differences between models are partly entropy differences. Add a T ∈ {0.6, 1.0} sweep on two models.
4. **Generation checkpoint.** S was generated with FP8 for four models and traced with the bf16 weights (l. 133). Quantify with one bf16-generated S arm on one model.
5. **Prompt construction.** The first assistant turn in every prompt is dataset-authored, so S is partly off-policy in context.
6. **Session replay.** Answers are replayed back to back. Short answers create more topic boundaries per token, which lowers locality. This is related to (1); report hit rates with a cache reset per answer as a sensitivity check.

**"The bound does not move" (l. 23, 89) is nearly tautological.** On the A10 platform the bound sits at the aggregation optimum, where M\* barely matters. Test F3 where the bound is budget-dependent: the 5090 host at C = 14 (174 → 183 tok/s between per-layer and pooled M\* alone), or any PCIe 4.0 host. Then qualify "dataset traces are safe for bounds" (l. 89) with "in the bandwidth-aggregation regime".

**Express provenance in seconds.** The case study already shows measured speed changes of −2 to −6% for the cache and −2 to +9% for llama.cpp between D and S. Repeat with the race system on the same prompts, forcing D and S texts through prefill-free replay (`ec-bench` replays teacher-forced text), and add ≥ 3 repeats to separate drift. That is what readers care about.

**gpt-oss-120b (l. 91).** Inserting the model's own reasoning may lower answer NLL because the reasoning *contains the answer content*, not because the format is fixed. Add a **shuffled-reasoning control**: another conversation's reasoning of similar length inserted before the answer, 35 conversations, one vLLM pass. If NLL stays about 6, "format" is confirmed; if it drops to about 3, the effect is format rather than leakage; anything in between means leakage and should be reported as such. Also report the answer NLL with a generic filler analysis ("Let me think.").

---

## 4. The first-party race and the FETCH/PREFETCH system

(`reports/Progress log.md`, jobs 059–068; not yet in the draft)

### 4.1 What is already fair

- Same host and same client. FreeToken's own `bench_decode_moe.py` is matched within 1.3%.
- Equal expert slots by construction: 504 vs 511 vs 512 at 11%; 1,152 at 25%; 1,836 vs 1,843 at 40%.
- A FETCH table chosen on one host and applied out of sample on another.
- A timeline prediction registered before job 066 (+17/+17/+14% predicted; +14/+16/+10% measured).
- Taking FreeToken's *better* mode per budget, which is generous to FreeToken.

### 4.2 What a reviewer will not accept, with evidence from the results tree

1. **Each system decodes different text.** `bs1.jsonl` output hashes differ between systems *and* between budgets of the same system. On problem 0:
   - llama.cpp n = 32, 27 and 20 produce three different texts;
   - the cache at C = 14, 32 and 56 produces three more;
   - FreeToken hybrid differs from FreeToken offload at 11% and 40%.

   "Paired by problem" is therefore not paired work. The ±10% spread across problems mixes problem difficulty with text-dependent routing. FreeToken's bench has `--greedy` for exactly this purpose. Use greedy decoding as the primary arm, report the fraction of identical outputs, and keep sampling with several seeds as a secondary arm.
2. **Warm-up on the identical text.** Each problem is first generated in full (256 tokens, same seed, same text), and then the measured request starts with a cache warmed on that text. This helps every cache over static llama.cpp and changes M\* (§1.2c). Warm up on a disjoint prompt, once per server launch.
3. **Statistics.**
   - There are 5 problems × 1 measured request × 1 server launch per cell.
   - The headline "+18–21% at 11%" compares cache+FETCH from job 066 or 068 with FreeToken from job 064 or 067: different sessions.
   - Summaries are arithmetic means of per-request rates, which differ from pooled rates by only 0.3%, but it is the wrong estimator.

   **[checked]** From job 064, the paired per-problem time ratio FreeToken/cache has SD 4.5% (hybrid, 11%), 9.2% (offload, 11%), 6.1% (25%) and 4.5% (40%). For a 95% CI half-width ≤ 2.5% you need about 20–25 problems; AIME-25 has 30.
4. **Equal memory is by flags, not measured.** Measured total VRAM is 9.5 / 17.4 / 28.1 GiB for the cache and **28.3 GiB for FreeToken at every rate**, because FreeToken sizes its KV pool to `memory_ratio` 0.9 (17.11 GiB of KV at rate 0.111). That is harmless at batch 1, but it must be shown. Read FreeToken's slot count from its log (`cache_report`) and the cache's from its log, and put measured resident expert bytes and total VRAM in the table.
5. **"FreeToken cannot fit 44%" is a configuration claim, not a capability claim.** The failure is `AssertionError: Not enough memory for KV cache, try reducing --num-pages` with `kv_reserve_tokens = 8192` and `max_seq_len_override = 8448`. Retry with `--num-token-override`/`--num-page-override` at about 1,024 tokens and `--memory-ratio 0.98` before claiming it.
6. **Tuning parity.** The cache's DFA parameters and FETCH table were tuned on traces. FreeToken ran its defaults. Give it an equal tuning budget on a disjoint tuning set of problems:
   - `--moe-hybrid-max-fetch` ∈ {0, 1, 2, 3, 4, −1};
   - `--moe-prefill-overlap` on and off (it borrows 2 × 128 slots from the pool);
   - `--moe-cpu-threads` ∈ {6, 8, 12, 16};
   - the attention backends available;
   - KV sizing.

   LRU is FreeToken's only implemented policy (`offload_cache.py: policy_ids = {"lru": 0}`), so policy is a legitimate algorithmic difference, not a handicap. Also sweep llama.cpp's `-t` and try `-ot` placements at tensor granularity and `--fit` with a VRAM cap, to show that `-ncmoe` is the strongest equal-memory static placement. Disclose that the cache is the author's.
7. **Is FreeToken configured well?** The only external check is a community 127 tok/s at 40% (host not stated) against 108.9 measured here. The decisive check is to **reproduce FreeToken's own published numbers on its own model** (Qwen3.6-35B-A3B BF16: 77.1 tok/s on an RTX 5090 at 23.8 GB), normalised by host bandwidth, before racing on it (E2). Also note that the bench fell back to T 1.0 / top-p 0.95 / top-k 64 ("no generation\_config"), while gpt-oss recommends top-p 1.0. That is harmless if applied equally, but it should be stated.
8. **Workload.** Prompts are 109–233 tokens (only problem 0 is 109) and decode is 256 tokens, i.e. the first part of the analysis channel. That is FreeToken's own workload, which is fair to FreeToken, but a claim about decode needs 1,024–4,096 tokens and fixed context depths (E6).

### 4.3 Is each system's SoL fraction computed correctly?

No, on four counts (`scripts/analyze_homepc.py`):
- **Wrong trace.** It uses job 036's own-text corpus trace, not the AIME text each system produced. Save full completions (only `text_head` is kept today), trace each system's text, and compute M\* per system. With greedy decoding, also on one common text.
- **Wrong budget class.** Per-layer M\* is applied to FreeToken (pooled) and to llama.cpp `-ncmoe` (whole layers, i.e. pooled). **[checked]** At C = 14, per-layer gives 174 tok/s and pooled 183 tok/s. At C ≥ 32 both give 313, because the bound is at the aggregation optimum.
- **Not a bound.** Measured bandwidths are used: GPU 1,558 GB/s against the 1,792 datasheet; CPU 48.6 GB/s with the DIMM speed unknown; link 47.6 GB/s against 63.
- **The timing structure** of §1.3 applies here too.

Corrected ladder on host 064, gpt-oss-120b, pooled M\*, ctx 240 **[checked]**:

| C (of 128) | Paper (Prop. 1, per-layer, measured basis) | Resource form, pooled, measured basis (reference) | Resource form + joint DRAM, datasheet 5090/PCIe 5, DDR5-4800 / 5600 / 6400 (bound) | Demand-only, measured basis | Measured: cache / FreeToken best |
|---|---|---|---|---|---|
| 14 | 174 | 225 | 180 / 210 / 240 | 149 | 42.9 / 39.4 |
| 32 | 313 | 320 | ≈ 373 | 255 | 68.1 / 68.6 |
| 56 (51) | 313 | 320 | ≈ 373 | 313 | 114.8 (105.2) / 108.9 at 40% |

At C = 14 the cache is at 18–24% of the valid physical bound (depending on DIMM speed) and **29% of the demand-only floor**. At C ≥ 32 every system is at 18–31% of a bound that is effectively all-in-VRAM speed. Both statements are informative. "24/20/30%" against a measured-basis number is not.

### 4.4 FETCH and PREFETCH

- The sign flip (−16% on the PCIe 4.0 RTX 4500 Ada host, for both FETCH and PREFETCH) was explained *after* the fact. Run the job-066 timeline model on job 065's constants now and report whether it predicts the negative sign. That is the cheapest possible validation of the "choose from measured bandwidths" claim. Register speed predictions, not just hit-rate predictions, for every future host.
- Add the missing arms inside the same cache at equal memory: **fetch-all** (FreeToken's offload rule) and **CPU-only**.
- Frame both mechanisms as test cases of the paper's method, not as contributions: prediction made before the run, measured against it, sign determined by B\_p/B\_c. Credit them to FreeToken, HybriMoE and DALI (FETCH) and to Mixtral-offloading and Speculating Experts (PREFETCH), as the recheck says.

---

## 5. Missing experiments, prioritised by value per dollar

Prices assume rented RTX 5090/4090 at $0.35–1/h. The progress log shows that Vast storage and download charges add roughly 50–70% to the GPU-hour ledger, so estimates include that. $17.38 remains in the current tranche. **E0, E1, E3, E2 and E4, in that order, fit in about $17–19.** E5–E8 go in the next tranche.

| ID | Experiment | Removes the objection | GPU cost | Priority |
|---|---|---|---|---|
| E0 | Recompute every denominator: resource-form and pooled bounds, joint DRAM at datasheet, warm-start M\*, M\* = 0 for "(i)" rows, no speculative rows, audit fractions over all 52 rows, per system | "The bound is beatable"; fragile medians | $0 (CPU) | 1 |
| E1 | Race v2: greedy, disjoint warm-up, same session, randomised order, 30 problems × 3 launches, tuning parity, measured slots and VRAM, 256 and 1,024 tokens | "Differences inside noise"; "handicapped competitor" | ≈ $6–9 | 2 |
| E3 | Time breakdown of the gap to the bound (waterfall), plus an all-GPU ceiling on an RTX PRO 6000 | "24% is a number, not a finding" | ≈ $3 | 3 |
| E2 | Second model on FreeToken's home turf (Qwen3.6-35B-A3B BF16), after reproducing FreeToken's published number | "One model"; "FreeToken misconfigured" | ≈ $4–5 | 4 |
| E4 | Host lottery with pre-registered per-host predictions, 8 hosts, 3 of them PCIe 4.0 (covers the PCIe 4.0 question) | "Two hosts is an anecdote"; "FETCH sign flip rests on one host" | ≈ $5–7 | 5 |
| E5 | PCIe 4.0 dedicated A/B (only if E4 lacks PCIe 4.0 hosts) | Same | ≈ $2 | (6) |
| E6 | Long decode and context depth (4k, 16k) | "Early-decode cherry-pick" | ≈ $1–2 | 6 |
| E7 | Audit baselines measured on rentable matching hardware | "Your weak calls are model artefacts" | ≈ $4–6 | 7 |
| E8 | Provenance confound controls and the shuffled-reasoning control | §3 | ≈ $1–2 | 8 |
| E9 | Tightness: an oracle-scheduled replay of the MIN-bypass solution with prefetch | "The bound is unattainable" | days of engineering | optional |

### E0. Recompute the denominators (free)

- **Hypothesis.** The valid resource-form bound lowers the audit median fraction from 24% to about 19% and the best row from 77% to about 44%; the race fractions fall by 3–10 points.
- **Deliverable.** A `bound_ladder` table: datasheet resource bound (pooled, joint DRAM) ≥ layer-structured ≥ demand-only, per row and per race cell.
- **Decision.** Headlines use the pooled resource bound; per-layer and structured values go in the appendix. Any row in which a measured system exceeds a figure means that figure is removed from the vocabulary of bounds.

### E1. Race v2 (gpt-oss-120b, RTX 5090 + 9950X class, one rental)

- **Hypotheses (register these):**
  - H-R1: cache+FETCH is faster than the better FreeToken mode at 11% of experts by ≥ 5% (paired time ratio, lower 95% CI bound > 1.05).
  - H-R2: at 25% and 40% the two are equivalent within ±5% (two one-sided tests).
  - H-R3: cache is ≥ 1.8× llama.cpp at its best equal-memory placement at 11%.
- **Variables.**
  - System: llama.cpp best of `-ncmoe`, `-ot` and capped `--fit`; cache; cache+FETCH; cache+FETCH+PREFETCH; FreeToken offload; FreeToken hybrid (tuned).
  - Budget: 11%, 25%, 40%.
  - Decode: greedy (primary) and sampled with 3 seeds (secondary).
  - Length: 256 and 1,024 tokens.
- **Controls.**
  - One rental and one session.
  - System order randomised within each block.
  - Warm-up on a disjoint prompt once per launch.
  - Threads recorded; clocks recorded.
  - STREAM, read-only, H2D and concurrent microbenchmarks run before *and* after.
  - Slots and VRAM measured.
  - Full completions saved, with token ids where the API exposes them.
- **Tuning phase.** Five tuning problems from AIME-24, up to 6 configurations per system, the best chosen per budget and frozen before the test phase.
- **Samples.** AIME-25, 30 problems × 3 launches per cell. Size: SD of the paired ratio about 6% ⇒ n ≈ 22 for a ±2.5% half-width.
- **Statistics.**
  - Pooled time per token (Σtime/Σtokens).
  - Hierarchical bootstrap: launches within cell, problems.
  - Wilcoxon signed-rank as a check.
  - P50/P99 inter-token latency and TTFT reported.
- **Cost.** 18 test cells × 3 launches × about 7 min, plus 2 h of tuning ≈ 8.5 h ≈ $4–8.5, plus about $1.5 of storage and download.
- **Decision.** Claim only the hypotheses whose CI clears the margin. If H-R2 holds, say "on par" rather than "ahead".

### E3. Time breakdown of the gap to the bound

- **Hypothesis.** At 11%, over half of the gap to the demand-only floor comes from excess misses (the policy's misses above M\*: 1.8 vs 0.95 per layer-step for the deployed cache) and CPU per-expert efficiency. At 40–44%, fixed per-layer coordination costs dominate (the synthesis note's Theme 2).
- **Method.** For each system × budget, a stacked waterfall of time per token:
  1. Datasheet resource bound.
  2. Plus hardware attainability: the same bound at measured maximum bandwidths.
  3. Plus clairvoyance: the demand-only floor. For prefetching systems, use measured predictor recall instead.
  4. Plus policy: the cost model fed the system's *own* per-layer-step miss, fetch and admit counts at ideal efficiency. The cache has these in its logs; FreeToken needs `--moe-collect-stats`; for llama.cpp they are static.
  5. Plus kernel efficiency: measured per-expert GPU `MUL_MAT_ID` and CPU helper times from microbenchmarks, replacing s/B.
  6. Plus coordination: GPU idle gaps waiting on the CPU and vice versa, measured with nsys and NVTX layer ranges plus CPU helper timestamps (`mb_busy_us` already exists).
  7. Residual up to the measured time.
- **Acceptance.** Residual ≤ 10% of measured time; otherwise the decomposition is declared incomplete.
- **Implementation ceiling.** Run all-GPU gpt-oss-120b (llama.cpp, and FreeToken at cache rate 1.0) on an **RTX PRO 6000 Blackwell**, which has the 5090's 1,792 GB/s. This gives each implementation's measurable η\_g for the same model. It is the only honest "implementation-relative" ceiling for a model that does not fit in 32 GB.
- **Cost.** About 2 h on the race host with short profiled windows, plus about 1 h on a PRO 6000 at about $1–1.3/h: about $3 in total.
- **Decision.** The paper's statement becomes "system X is at Y% of the bound; the largest missing share is Z". That turns finding (4) into an actionable result.

### E2. Second model on FreeToken's home turf

- **Model.** Qwen3.6-35B-A3B BF16: FreeToken's model, about 70 GB. Use a BF16 GGUF converted on the host for llama.cpp and the cache, so the weights are bit-aligned. Fall back to Qwen3-30B-A3B BF16 (61 GB) if needed.
- **Step 1 (gate).** Reproduce FreeToken's published configuration (23.8 GB budget, 77.1 tok/s). Pass: ≥ 90% of the published figure after scaling by the ratio of measured host CPU bandwidth. If it fails, fix FreeToken's configuration before racing.
- **Step 2.** The E1 protocol at 11%, 25% and 40%, with 20 problems × 2 launches, greedy, 512 tokens.
- **Hypothesis.** The ordering at 11% holds on FreeToken's own model; register the predicted tok/s for each system from the step model.
- **Cost.** About 3 h plus about 70 GB of download and conversion: ≈ $3–5.
- **Decision.** If the ordering flips, report it. That is a finding about policy × model interaction, not a failure.

### E4. Host lottery with pre-registered per-host predictions

- **Design.** 8 hosts chosen at random among Vast offers that meet a written filter: 5 × RTX 5090 with PCIe 5 and 3 × RTX 4090 with PCIe 4, all ≥ 96 GB RAM and whole-machine or single-GPU hosts. Record each offer's advertised specifications.
- **Per host, before any decode run** (automated, about 5 min):
  - `dmidecode -t memory` if permitted, `lscpu`, `nvidia-smi -q` (current PCIe generation and width), cgroup quotas, the number of GPUs on the host;
  - STREAM Triad and Copy plus a read-only kernel at 1…N threads;
  - pinned H2D (copy engine and zero-copy);
  - concurrent CPU + DMA.

  Then compute and **commit** predictions from the step-time model *frozen from hosts 059/060/064*: tok/s for llama.cpp `-ncmoe`, the cache and cache+FETCH at 2–3 budgets, and the sign of FETCH's gain.
- **Decode.** Own-text `ec-bench`, 12 × 128, bit-identical workload across hosts, 2 repeats, about 20 min per host. Re-run the bandwidth tests after decode to detect noisy neighbours.
- **Statistics.**
  - Measured vs predicted: slope, R², median APE with a bootstrap CI.
  - Coverage of the ±15% interval.
  - FETCH sign accuracy.
  - Variance decomposition, between host vs within host (within-host is 0.4–0.8% from jobs 060 and 064).
- **Pre-registered pass.** Median APE ≤ 10%, ≥ 6 of 8 hosts within ±15%, FETCH sign correct on ≥ 7 of 8.
- **Cost.** 8 × about 0.7 h × $0.35–0.7, plus 8 × 63 GB downloads (the dominant cost): ≈ $5–7. Use gpt-oss-20b on the 4090 hosts if 120b does not fit the RAM filter; the host physics is the same.
- **Decision.**
  - If it passes: "the host lottery is predictable from three microbenchmarks", which is a strong, new, contract-relevant result (C5).
  - If it fails: report which term fails. My guess would be concurrent DRAM or fabric limits.

### E5. PCIe 4.0 (only if E4 has fewer than 2 PCIe 4.0 hosts)

- RTX 4090 + DDR5 desktop, gpt-oss-120b at 11% and 25% (a 24 GB card), plus gpt-oss-20b as the E4 cross-check.
- **Hypothesis.** FETCH and PREFETCH change the time by the sign the model predicts from B\_p/B\_c (negative when B\_p/B\_c ≲ 0.6), and FreeToken's calibrated hybrid beats its offload mode.
- **Cost.** About 3 h, ≈ $1.5–2.

### E6. Long decode and context depth

- 1,024 and 4,096 decode tokens (in E1), plus fixed prompt depths of 4k and 16k from a long-document set.
- 3 systems × 1 budget (25%) × 10 prompts × 2 launches.
- **Hypothesis.** PREFETCH's gain grows with context, since attention time hides copies (the 109-token vs 640-token contrast in the progress log). The ordering of the cache and FreeToken is stable within ±5%.
- **Cost.** About 1.5 h, ≈ $1–2.

### E7. Audit baselines measured on rentable matching hardware

- **Rows.**
  - SeqMoE Qwen3-30B-A3B (RTX 4090, 4.4 and 13.1 GB budgets): llama.cpp at Q8\_0, the format llama.cpp can run.
  - SeqMoE Qwen3.6-35B-A3B BF16 (RTX 5090, 9.7 and 25.8 GB).
  - Pipelined Sharding Qwen3-30B-A3B Q4\_0 (RTX 5090, 5.7 GB) and Qwen3-235B-A22B Q2\_K (RTX 5090, 27.6 GB; needs ≥ 128 GB RAM).
  - CPU-GPU collaborative Phi-3.5-MoE (RTX 4090).
- **Measure.** Equal-memory llama.cpp at the row's budget, then scale by (row host DRAM band) / (measured host bandwidth), with the scaling uncertainty propagated.
- **Hypothesis.** Rows called robustly weak (reported/predicted < 0.5) remain weak against the measured-and-scaled baseline.
- **Cost.** Two rentals, about 4 h, about 250 GB of downloads: ≈ $4–6.
- **Decision.** Any robust call that flips is reported in the paper, and its row's label changes.

### E8. Provenance controls

- Length-matched and position-stratified ΔDFA on the existing traces (free).
- An S temperature sweep (T 0.6 / 1.0) for Qwen3 and OLMoE.
- A split of gpt-oss-20b S into analysis and final tokens (free, from existing token files).
- The shuffled-reasoning NLL control for gpt-oss-120b (35 conversations).
- One bf16-generated S arm (OLMoE).
- **Cost.** About 1.5 h of vLLM on a 40 GB card, ≈ $1–2.
- **Decision.** If the length-matched effect keeps its sign on ≥ 8 of 9 models with CIs excluding 0, keep the direction claim unqualified. Otherwise report it as length- or genre-mediated.

### E9. Tightness (optional)

- Replay the offline MIN-bypass-with-prefetch schedule (known future) inside the cache: a "bound-hunter" that shows how close real hardware gets on the 5090 host at C = 32.
- If an oracle schedule reaches only about 50% of the bound, say so. It calibrates what "25% of SoL" means.

---

## 6. What the paper should and should not claim

**Should claim**, after the fixes:
1. A policy-independent lower bound in seconds for non-speculative, exact-routing batch-1 decode under per-layer or pooled expert budgets, valid with datasheet peaks, stated in resource form with a layer-structured refinement and a demand-only corollary, and computed from routing traces of the measured text.
2. On PCIe 5 desktops the bound for gpt-oss-120b is, above a small budget, the all-in-VRAM speed. The gap is therefore about prediction and CPU efficiency, not capacity. Quantify this with the E3 waterfall.
3. No published MoE-offloading speed-up is measured against an equal-memory `--n-cpu-moe` baseline, with the era split (9 of 13 contemporaneous llama.cpp baselines weak; older ones below today's llama.cpp).
4. Audited systems reach roughly 5–35% of the valid bound (per-system medians), a range rather than a point.
5. The survivor count is a range (for example 4–10 of 22) under the predictor's documented platform error.
6. Own-text traces are less local than dataset text on 9 of 9 models, by 1–9 points at 12.5% of experts, with length and genre controls; the bound is insensitive only in the bandwidth-aggregation regime.
7. gpt-oss-120b is off-distribution on dataset text in the final channel, reproduced in three implementations, with the shuffled-reasoning control.
8. For the race, only what E1 establishes: probably "about 8–12% faster than FreeToken at 11% of experts, on par at 25–40%, on one model, on hosts of one class". And the host lottery as a predicted, measured phenomenon (E4).

**Must not claim:**
1. That Prop. 1 as written bounds "any exact-routing policy" (l. 20, 36, 68). It does not cover shared-expert overlap, LLC prefetch, dense splitting, speculation or recompression.
2. That any measured-basis figure is a "speed-of-light" or "floor". This covers:
   - the race's 174/313/313;
   - the A10 implementation-relative SoL (l. 75, 114);
   - "no configuration beat the roofline" (l. 80), unless the basis is stated. A1 on the A100 shows why.
3. "The median adjudicated system reaches 24% of the physical speed-of-light" (l. 22, 105, 139) as a point value over a post-hoc subset.
4. "The best reaches 77%" (l. 105). That row is KTransformers, whose shared-expert overlap is exactly what the layer-structured form excludes.
5. "None of the 15 papers configured its partial expert offload" (l. 22, 105), without the era qualifier and the recheck's wording ("no published speed-up is measured against…").
6. "Claimed up to 4.2×" (l. 28) as the authors' claim.
7. "Dataset traces are safe for bounds" (l. 89) in general.
8. "Following Hoefler and Belli, we pre-registered" (l. 32): they prescribe reporting, not pre-registration.
9. "Validated speed-of-light" (title, l. 11). The *model* is cross-validated; the *bound* needs no validation because it is a theorem, provided it is stated as one. "Validated" invites exactly the A1 objection.
10. For the race: "ahead at all three budgets"; "runs 44%, which FreeToken cannot fit" (before the KV-sizing retry); FETCH or PREFETCH as novel mechanisms; any cross-session comparison presented as same-session.

---

## 7. Line notes (`paper/paper.tex`)

- **l. 11.** Drop "Validated" from the title (§6.9).
- **l. 19.** "Evaluation contracts … count cache blocks": add "and seconds-domain models are estimators for one policy or schedule" (WiSP v2, Budgeting Bytes).
- **l. 20.** Add "non-speculative" and "from datasheet peaks" to the scope of (1).
- **l. 22.** Survivors and median as ranges; era split for "none of the 15 papers".
- **l. 23.** "The norm in this literature" is true only for routing-locality studies (see the recheck). "The bound does not move": add "in the bandwidth-aggregation regime".
- **l. 28.** Define "claim" as the auditor-computed ratio, or quote the authors' own claims.
- **l. 30.** "Since mid-2025": give the `-ot` date (spring 2025) and the `--n-cpu-moe` date (by 5 Aug 2025); era matters for the audit.
- **l. 36.** The list "static or dynamic, demand or prefetching, with misses fetched or executed on the CPU" should add "or read by the GPU over the link (zero-copy)" and exclude speculation.
- **l. 52.** "Up to floating-point ties": state it as the measured agreement (≤ 0.07% GPU vs CPU; ≤ 0.35% vs the reference) rather than as an assumption.
- **l. 61–64.** Eq. 1 is the *predictor*; the bound's T\_D, a, b, p are its physical specialisation. Define them inside Prop. 1 (T\_D is never defined).
- **l. 66.** "Each resident copy costs one load over PCIe (p), which may overlap anything" is where A4 and A6 hide; spell them out.
- **l. 68–71.** Replace with §1.4. Name M\* as a total and M\*/N as per layer-step, and give the bounds on x and m.
- **l. 73.** Replace the sketch with the charging, interval-packing and resource argument; move the details to an appendix.
- **l. 75.** Correct consequence (i) (§1.2h). Say that "no policy beats it" is under the bound's own cost model. The implementation-relative version is a reference, not a bound.
- **l. 80.** "No configuration beat the roofline": state the basis. H0 on the A10 used a measured basis, and the A100 shows that such a basis can be beaten.
- **l. 89.** Qualify "safe for bounds"; add the length and genre controls.
- **l. 94.** "Three extraction passes, run by AI agents": fine, and the quote-plus-location protection is good. Add the authors' right of reply.
- **l. 96.** The three post-first-pass choices are disclosed, which is good. Add a fourth: SoL fractions computed over all modelled rows, independent of adjudication.
- **l. 105.** Finding (4) is fragile (§2.4); finding (1) needs era wording.
- **l. 107.** Good paragraph. Move its range into the abstract, and apply the measured-host rule to KTransformers.
- **l. 110–114.** "Implementation-relative speed-of-light" should become "implementation-relative reference". The llama.cpp column uses the per-layer bound for a pooled policy.
- **l. 119.** "Instances of one type are not one machine": good. Add DIMM population (Maricq et al.) and shared-host neighbours, and point to E4.
- **l. 127 (contract).** Add fields for:
  - concurrent CPU+DMA bandwidth;
  - DIMM configuration (`dmidecode`);
  - measured resident expert bytes and total VRAM;
  - decoding (greedy or sampled, seed, generation config);
  - warm-up protocol and cache state (cold or warm, and on which text);
  - decode length and context depth;
  - the tok/s aggregation formula;
  - speculation or MTP;
  - bytes per location (format).
- **l. 131.** "Effects below about 1.3× are inside the band" conflicts with l. 107's single-configuration misses of up to 2×; reconcile them.
- **l. 135.** Scope: add speculation, recompression and shared-expert or LLC overlap (or adopt the resource form, which removes the last one).

---

## Appendix: how the checked numbers were computed

- **Race-host bounds.** Trace: `gpu-branch/results/036_trace_retry@40gb/gpt-oss-120b_S.npz`, response tokens only, 35 segments, 32,601 steps. `mosl.archs.shape("openai/gpt-oss-120b")`, `Workload(s, 4.25, 16, ctx=240 or 640)`: expert 13.22 MB, dense 3.11 GB per token. `cachesim.simulate`/`simulate_global` for per-layer and pooled M\*. The resource form minimises max{(D + L(k−m)s)/B\_g, Lms/B\_c, Lxs/B\_p, [L(m+x)s/B\_h]} over a grid of x and m ≥ (M\* − x)⁺. Warm-start M\* prepends a synthetic prefix requesting the C earliest-first-used experts of each segment and discards its misses. Demand-only uses ℓ\_d as in §1.4. Host 064 constants: B\_g 1,558.4, B\_c 48.6, B\_p 47.6 GB/s measured. Datasheet: 1,792 GB/s (5090), 63 GB/s (PCIe 5 x16), 76.8 / 89.6 / 102.4 GB/s (dual-channel DDR5-4800 / 5600 / 6400).
- **Audit fractions.** `prereg/audit/audit.json` rows plus `data/audit/normalized_v2.jsonl`, recomputed with the resource form at the row's datasheet peaks (top of the DRAM band, as in `audit.py`); M\* = k at C = 0, trace M\* where traced, M\* = 0 for "(i)" rows with C > 0. Traced rows used per-layer M\*; pooled M\* would lower the fractions slightly further.
- **Race statistics.** `064_samehost_rerun@vast/bs1.jsonl`: per-problem `ms_per_token` ratios; output hashes; `vram_used_gib`/`vram_gib`; server logs for FreeToken's arguments, KV allocation and the 44% failure.
