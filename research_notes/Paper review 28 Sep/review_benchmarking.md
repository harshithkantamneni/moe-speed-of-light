# Review: "Seconds, Not Blocks" (draft of 28 September 2026)

*Reviewer role: senior PC member, MLSys 2027 / ISPASS, performance modelling and scientific benchmarking. Inputs:
`paper/paper.tex` at commit `be3b29e` with its macro, table and appendix files; `reports/MoE offload paper novelty
recheck.md`; `reports/Progress log.md` (28 Sep). I also checked the code that produces the numbers
(`mosl/perfmodel.py`, `scripts/audit.py`, `scripts/provenance.py`, `prereg/audit/audit.json`,
`results/validation.csv`). "Lxx" means line xx of `paper/paper.tex`. Each paragraph of that file is one line, so a
line reference points to a paragraph and I quote the phrase.*

**Recommendation as it stands: weak reject (major revision).** After the revision in §7, this could be a clear
accept at MLSys or ISPASS. The instrument is the right one: a physical ceiling in seconds, plus an equal-memory
strong baseline, applied to a whole literature. Nothing like it exists in this subfield. But the evidence does not
yet meet the standard the paper sets for others. Three problems dominate:

1. **The audit rests on a predicted baseline, and one adjudication rule filters rows by system.** The predictor has
   never been checked on the host classes the audit judges. Its ±40% band-width rule removes from the headline the
   very rows (FreeToken's) whose baseline looks correctly configured and whose gain holds.
2. **The "speed-of-light" is not a lower bound everywhere it is used.** It breaks where shared experts overlap CPU
   work, where untraced models get uniform-routing misses, where pooled caches are judged against the per-layer
   bound, where measured bandwidths stand in for peaks, and on speculative-decoding rows.
3. **The newest evidence has almost no statistics.** The same-host comparison and the host effects rest on five
   prompts, one measured request each, and no confidence intervals. The A10 case study has no repeats.

All three are fixable in four weeks, mostly without GPUs.

---

## 1. Summary and strongest contribution

**Summary (five sentences).**

1. The paper argues that batch-1 decode of a Mixture-of-Experts model, with most experts in host DRAM, is a
   data-movement problem across GPU memory, host DRAM and PCIe. Published speed-ups here cannot be interpreted,
   because they come with no ceiling and are measured against baselines whose configuration is unreported and
   usually not at equal memory.
2. It supplies two tools:
   - a ceiling, Proposition 1: a lower bound on mean seconds per token for every exact-routing policy under a
     per-layer or pooled GPU expert budget, built from Belady-with-bypass misses on exact routing traces and a
     concurrent CPU/GPU/link relaxation;
   - a bytes-over-bandwidth decode model (Eq. 1), validated with grouped cross-validation on 52 third-party
     measurements (16% median error) and by pre-registered runs on an A10, an A100 and a GH200.
3. It audits 147 published measurements from 41 systems and predicts an equal-memory `--n-cpu-moe` baseline for the
   52 rows it can model byte by byte. It reports that 16 of 20 adjudicable llama.cpp baselines fall below the
   predicted band, 8 of 22 claimed gains survive, and the median adjudicated system reaches 24% of the physical
   bound.
4. A pre-registered provenance study on nine models shows two things:
   - teacher-forced dataset text is more local than the models' own generations (hit rates 1–9 points higher at
     12.5% of experts, every CI excluding zero);
   - gpt-oss-120b is off-distribution on dataset answers unless its own reasoning is inserted.
5. It closes with the author's own measurement pitfalls, a pre-registration log that records its failures, and a
   time-level reporting contract. The contract fills the "bytes and bandwidth budget" field that the block-level
   contract of arXiv 2608.07911 leaves open.

**The strongest contribution is the audit, used as an instrument.** It re-reads published results against two
references at once: an equal-memory llama.cpp configuration and a physical ceiling. It comes with a 147-row dataset
in which every value has a quote. It is the one result that should change how a reader reads this literature, and
it is reusable: the next paper can place itself on the same chart.

Its textual finding is robust, but needs one wording change (§3.6): no published speed-up is measured against an
equal-memory `--n-cpu-moe` configuration.

The bound is what makes the audit possible, not the headline. The provenance study is the most original secondary
result. The weakest link in the whole chain is that the audit's baselines are predicted, never measured, on host
classes where the model has never been validated.

---

## 2. Contribution and framing

### 2.1 Is the thesis clear and important?

It is important. The thesis is: report offloaded-MoE decode in seconds, against a physical bound and an equal-memory
strong baseline, with the host measured. That is Hoefler and Belli's rule 11 ("show upper performance bounds")
applied to a field where no paper does it, plus a correction for baseline fairness. The problem is real: 15 of 35
papers use llama.cpp as a baseline, and claims reach 4.2×.

The thesis is not clear, because the draft states it three different ways:

- **The title** is about a bound ("A Validated Speed-of-Light").
- **The subtitle** is about an audit ("What It Says About Published Speed-Ups").
- **The opening and conclusion** are about a contract ("seconds, not blocks").

The abstract (L19–24) lists four contributions carrying about 25 numbers. It reads like a results table, and the
one-sentence thesis gets lost.

Two framing choices will cost points with reviewers:

- **"Seconds, Not Blocks" makes the paper a reply to one preprint** (2608.07911). That frame is narrow and depends
  on which reviewer you draw. The title's shape also echoes "Caching for Dollars, Not Hits" (arXiv 2606.20539; see
  the recheck).
- **"Validated Speed-of-Light" mixes up two objects.** A bound is proven, and can at most be checked for not being
  beaten. A model is validated. The physical bound sets every η = 1 and every τ = 0, so it has no fitted parameter
  to validate. Put "validated" on the model, never on the bound.

### 2.2 What the headline should be after the novelty recheck

After the recheck, nothing survives as "first" in general form. The time-domain bound combines elapsed-time caching
theory (Cao 1995, Albers–Garg–Leonardi 2000, CHOPT 2020) with 2025–26 MoE paging work. GenZ already validated an
analytical model on third-party data.

Three things are unoccupied:

- a lower bound in seconds, valid over all policies, for exact-routing offloaded MoE with misses executed on the CPU;
- applying that bound to a literature, next to an equal-memory baseline;
- host measurements showing that the host memory system sets both the ceiling and the sign of a mechanism's effect.

The last is new since 28 Sep and still thin.

**Proposed headline:** *"Measured against what the hardware allows and against an equal-memory llama.cpp
configuration, most published MoE-offloading gains are baseline gains, and systems run at roughly a tenth to a
third of their ceiling. On consumer PCs, the host's memory system, not the GPU, decides both the ceiling and which
mechanism helps."*

The second sentence enters only if experiment E3 (§4) lands, meaning at least six hosts with pre-registered
predictions. Otherwise the host effects stay a pitfall and a contract field, not the headline.

**Tone.** The author has said he does not want to pick fights. The same evidence reads very differently with
neutral words:

| Draft word | Replace with |
|---|---|
| "weak" | "below the equal-memory prediction" |
| "survives" | "established against the equal-memory baseline" |
| "race" | "same-host comparison" |

Also name the positive cases explicitly. FreeToken's llama.cpp baseline is within its band (§3.6). KTransformers'
and SeqMoE's gains hold. A paper that says who got it right reads as measurement; one that only counts failures
reads as a takedown.

### 2.3 Title

- **Preferred:** *How Fast Could It Be? A Time-Domain Bound and Equal-Memory Baselines for Offloaded
  Mixture-of-Experts Decode*
- **If the host results land (E3):** *The Host Decides: Speed-of-Light Analysis of Offloaded Mixture-of-Experts
  Decode*
- **Smallest change:** *Seconds, Not Blocks: A Speed-of-Light and Equal-Memory Baselines for Offloaded MoE Decode.*
  Drop "Validated" and "What It Says About Published Speed-Ups".

### 2.4 Abstract framing (a skeleton, not a rewrite; aim for about 180 words)

> *[Problem, 2 sentences]* Running a large MoE on one GPU keeps most experts in host memory, and dozens of systems
> report faster batch-1 decode in this setting. Their gains cannot be compared: no paper reports how fast the
> hardware could run the model within the same GPU memory, baselines are rarely configured at equal memory, and the
> host's memory system is rarely measured. *[Instrument, 2 sentences]* We give a lower bound on decode time in
> seconds for every exact-routing policy under a per-layer or pooled expert budget, and a bytes-over-bandwidth model
> whose host inputs are measured. The model is validated out of sample on [n] third-party measurements and by
> pre-registered predictions on [m] platforms. *[Findings, 3 sentences, at most 2 numbers each]* Across [P] papers,
> no speed-up is measured against an equal-memory configuration of llama.cpp. Judged by the configuration available
> at each paper's date, [K] papers' baselines ran at or below half the predicted equal-memory speed, and systems
> reach a median [Z]% of their physical bound. On consumer PCs, hosts of one GPU class differ by [~20]% in decode
> speed, and the sign of a mechanism's effect follows the host's link-to-DRAM ratio, both as predicted from measured
> bandwidths. *[Secondary, 1 sentence]* Routing traces built from dataset text overstate cache hit rates at small
> budgets on all nine models we trace. *[Deliverable, 1 sentence]* We release the bound, traces, audit set and a
> time-level reporting contract.

### 2.5 Revised contribution list (drafted)

- **C1. A time-domain lower bound for offloaded MoE decode (§3, full proof in App. A).**
  - It bounds mean seconds per token for every exact-routing policy that keeps whole experts on the GPU under a
    per-layer or pooled budget. That covers static or dynamic policies, demand or prefetching, and misses that are
    copied, streamed or executed on the CPU.
  - It combines Belady-with-bypass misses on the routing trace with three terms: per-layer CPU/GPU concurrency, link
    capacity, and shared host DRAM.
  - It is the elapsed-time analogue, for misses that can run where they are stored, of the prefetching and caching
    results of Cao et al. and Albers et al.
  - It takes the block-level oracles of 2608.07911 and Paging the Experts as its miss input, and contains the
    exposed-latency bound of Budgeting Bytes (no cache, every miss fetched) as a special case.
- **C2. A decode model with measured hosts.**
  - It is a bytes-over-bandwidth model with a per-layer fixed cost. Its host inputs are measured: read bandwidth at
    the engine's thread count, pinned host-to-device bandwidth, and the two together.
  - It is validated out of sample on 52 third-party measurements from 9 sources (nested leave-one-source-out: [x]%
    median error [CI]).
  - It is tested by pre-registered predictions on [3 + N] first-party platforms, including [N] rented consumer hosts.
  - A registered two-run calibration transfers within a model family (≤7%) but not across families (up to 45%). Say
    so.
- **C3. An audit of the literature against both references.**
  - 147 published measurements from 41 systems (2023–Sep 2026), each with a verbatim source quote, human-verified
    where adjudicated; 52 modelled byte by byte.
  - No published speed-up is measured against an equal-memory `--n-cpu-moe` configuration.
  - Judged by the configuration available at each paper's date, [K of P] papers' llama.cpp baselines ran at or
    below half the predicted equal-memory speed. This is spot-checked by measuring llama.cpp on [X] of the audited
    GPU classes.
  - Gains are established against the equal-memory baseline for [S] systems. Systems reach a median [Z]% of their
    physical bound, [Z′]% taking the median per system.
- **C4. Consumer-PC measurements: three systems against the bound, and what the host decides.**
  - A same-host, same-client measurement of llama.cpp, FreeToken and the author's llama.cpp expert cache on RTX 5090
    desktops. The referee is FreeToken's unmodified benchmark client. Expert memory is measured equal. Results are
    fractions of the bound, with a time breakdown.
  - Host measurements on [N] rented machines show three things:
    - CPU reads and GPU DMA from host DRAM contend far less than assumed: 37 of 62 GB/s is retained under DMA, and
      75 GB/s is delivered together.
    - Hosts of one GPU/CPU class differ by about 20% in decode speed, predicted from measured bandwidth.
    - Whether splitting or prefetching misses helps depends on the link-to-DRAM ratio, as the model predicts.
- **C5. Trace provenance.**
  - Paired dataset, greedy and sampled traces of nine models, with pre-registered decision rules.
  - The models' own text is less local than dataset text on every model (1–9 points at 12.5% of experts). On two of
    the five extension models, the drop exceeds the threshold registered for the four main models.
  - The bound moves only in the regime where it depends on the budget.
  - gpt-oss-120b's loss on dataset answers falls from 6.40 to 3.00 nats once its own reasoning is inserted.
- **Deliverable: a time-level reporting contract.** About 14 fields, each justified by an effect measured in this
  paper (§5 lists the six new ones). Present it as the paper's output, not as a separate contribution.

The pitfalls stay as a short section, not a contribution.

### 2.6 What to stop claiming

| Draft claim | Where | Problem | Replace with |
|---|---|---|---|
| "no paper we found states how fast its hardware could decode" | L30 | Fractions of a ceiling exist elsewhere: dense decode (2605.30571) and SeqMoE's fraction of full-load speed | "no MoE-offloading paper reports a lower bound, valid over all policies, at the same GPU memory" |
| "None of the 15 papers … configured its partial expert offload" | L22, L105 | False: Budgeting Bytes discusses `--n-cpu-moe` | "No published speed-up is measured against an equal-memory `--n-cpu-moe` configuration" |
| "the norm in this literature", "as in most trace studies" | L23, L56, L85 | False for cache-systems papers, which mostly generate their text | Scope to routing and locality characterisation and calibration studies |
| "Following Hoefler and Belli, we pre-registered" | L32 | SC'15 does not prescribe pre-registration | "in the spirit of Hoefler and Belli; as 2608.07911 and 2608.18261 did for cache simulations" |
| "Validated Speed-of-Light" | L11 | A bound is proven, not validated | See §2.3 |
| "Dataset traces are therefore safe for bounds" | L89 | True only where the bound does not depend on the budget (§3.7) | Qualify by regime, or drop |
| FETCH and PREFETCH as ideas | 28 Sep work | FETCH is FreeToken, HybriMoE and DALI; PREFETCH is Mixtral-offloading through Speculating Experts | Test cases of the method only |
| "our cache is ahead of FreeToken" | Progress log | n = 5 prompts, one model, 256 tokens, the author's own system, unequal tuning | Fractions of the bound with CIs (E1) |

---

## 3. Soundness

### 3.1 Proposition 1: the statement (L66–71)

**S1. Serialising shared experts makes the bound false for every shared-expert model in the audit (the DeepSeek
family and Qwen1.5/Qwen2-MoE).**

- $T_D$ includes the shared experts (L64: "D the dense per-token bytes (attention, shared experts, routers …)").
- It is added *serially* to $L\,\ell(m)$ (L70; `mosl/perfmodel.py:175`: `t = max(dense + L * lay.min(), L*x*pf)`).
- A policy that runs the shared experts on the GPU *while* the CPU runs routed misses beats the stated bound. That
  is exactly what KTransformers does.
- Size of the error: for DeepSeek-V3 on the RTX 4080 row, the shared expert is about 44 M parameters per layer
  (about 25 MB at 4.5 bits, about 35 µs at 717 GB/s). Over 58 layers that is about 2 ms, or roughly 7% of the
  30 ms bound time.
- The error is small, but it makes the theorem false as stated for DeepSeek-V2-Lite, V2.5, V3 and V4 and for
  Qwen1.5/Qwen2-MoE.
- **Fix:** $\ell(m)=\max(s_g + (k-m)a,\ mb)$, where $s_g$ is the per-layer GPU time that does not depend on routing
  in the MoE phase (shared experts). Restrict $T_D$ to the strictly serial part: attention, norms, router and LM
  head.

**S2. The budget's unit and format must be in the statement.**

- The budget counts whole experts in their stored format. Exact policies outside that class exist:
  - splitting one expert's rows between a resident slice and the CPU;
  - lossless compression of resident or transferred weights, such as entropy-coded BF16. This matters for SeqMoE's
    BF16 Qwen3.6 rows.
- Zero-copy streaming, where the GPU executes from pinned host memory, *is* covered: it is a load for a run of one
  use. But the proof must say so, because it is neither of the two execution modes the sketch names.
- Either exclude sliced and compressed residency by definition, or state the budget in bytes with a fractional
  relaxation of MIN. That relaxation is solvable as a linear program and still gives a valid bound.

**S3. The bound does not cover speculative decoding.**

- SP-MoE and MoE-SpeQ verify several draft tokens per target step and read each expert once for all of them. A
  per-token batch-1 bound does not apply.
- Eleven audit rows carry `speculative: true` in `prereg/audit/audit.json`. One (SP-MoE, DeepSeek-V2-Lite) is
  adjudicated, and its "4% of SoL" is measured against a bound that does not cover it.
- Either exclude these rows from the fractions, or extend the bound: with verification width $w$, use the per-step
  union of routed experts over $w$ tokens from the trace.

**S4. Cold start, prefill and replay semantics are unstated.**

- $M^\star$ comes from answer tokens "replayed back to back as one session" (L56). The statement must say:
  - whether loads during prefill are free (a system can fill the whole budget during prefill);
  - whether the cache starts cold, per session or per request;
  - what happens at conversation boundaries.
- For a per-request protocol such as FreeToken's client, the valid bound lets the policy choose its initial cache
  contents, which gives a smaller $M^\star$.
- Pick the definition that makes the bound valid for the protocol actually used, and state it. Replay semantics is
  precisely what 2608.07911 shows can reorder policies.

**S5. The CPU's last-level cache.**

- $b = s/B_c$ assumes every expert executed on the CPU is read from DRAM.
- On an X3D host (96 MB of L3) with small experts, repeated CPU executions can hit in L3 and beat DRAM bandwidth.
  Qwen3-30B-A3B's experts are about 2.6 MB at Q4, so about 36 fit.
- For gpt-oss-120b (13.25 MB experts, about 450 MB of CPU reads per token) this is negligible.
- State the condition, and check it with an LRU simulation, at the host's L3 size, of the stream of experts the CPU
  executes.

**S6. Notation.**

- $N$ is called "the layer-steps", but the bound is per token with a factor $L$. Define $T$ tokens and $N = LT$.
- The inner minimisation needs $m \le k$.
- Values of $x$ above $M^\star/N$ never help; say so.
- "Mean time per token" should say over which tokens: those of the trace.

**S7. A shared-DRAM term: both valid and necessary on consumer PCs.**

- Every MIN-bypass miss has to be served by one expert read from host DRAM, either by the CPU or by a load, so
  $\bar m + x \ge M^\star/N$ and therefore $\bar T \ge L\,(M^\star/N)\,s/B_{\text{DRAM}}$.
- The current bound lets CPU reads and DMA both run at their independent peaks, more than the DRAM can deliver.
  - On the job-064 host, the bound allows 46.6 + 47.6 = 94 GB/s of DRAM reads.
  - The host delivered 52 GB/s with both running at once.
- With the datasheet peak at the *configured* DIMM speed, the term remains a valid physical bound.
- With the measured combined bandwidth it becomes an attainable ceiling, and it bites. Rough numbers for
  gpt-oss-120b at C = 14:
  - $M^\star/N \approx 0.95$, $s$ = 13.25 MB, $L$ = 36, so about 453 MB of host reads per token;
  - at 52 GB/s that is at least 8.7 ms per token, at most about 115 tok/s;
  - the progress log's "speed-of-light" at this budget is 174 tok/s;
  - so the cache with FETCH (48 tok/s) moves from about 27% of the ceiling to about 40%.
- The fractions the 28 Sep work reports therefore depend heavily on which ceiling is chosen. §3.3 says which one may
  carry claims.

### 3.2 Proposition 1: the proof sketch (L73)

**P1. Write the key reduction out.** "A prefetching schedule bridges no more gaps than some demand schedule with
bypass" is correct, by an interval argument:

- Map each run of an expert, served by one resident copy, to the interval from its first use to its last.
- The demand schedule that admits the expert at its first use (a counted miss) and evicts it after its last use
  occupies a subset of the prefetching schedule's slot-time, so it respects the budget, and it has exactly
  $\sum (r-1)$ hits.
- MIN with bypass maximises hits among demand schedules with bypass.

The appendix needs a proof of the last step: the exchange argument, or a citation that proves it for the bypass
model. It also needs to handle *simultaneous requests*. The $k$ experts of a layer-step have no order, so either show
that MIN-bypass's miss count does not depend on the order inside a step, or take the minimum over orders. This is
the "paging with request sets" subtlety.

**P2. Cite Jain & Lin (ISCA'18) and explain why their result does not apply.** They show MIN does not minimise
demand misses under prefetching. The reduction survives because it counts loads explicitly.

**P3. State the independence assumption behind the per-step term.** The Jensen step is fine: $\ell$ is the maximum
of two affine functions. But the per-step lower bound $\ell(m_i)$ needs one assumption: no CPU work of step $i$
overlaps GPU work of step $i+1$. That holds at batch 1 by data dependence, and fails for speculative verification and
for pipelined multi-request serving. Say it.

**P4. Global variant.** State that the interleaved stream is layer-major within each token, and that no other
interleaving is physical at batch 1.

**P5. "Why bound, not solve."** A caching theorist will point out that Albers et al. solve elapsed-time
prefetch/caching by linear programming and CHOPT solves by min-cost flow. The answer the paper should give in one
paragraph:

- CPU/GPU concurrency adds a max(·) per layer-step;
- bandwidths are shared;
- prefetching must be covered;
- an audit needs one number per (trace, budget, host) that is valid over all execution models, not the exact
  optimum of one.

**P6. The computed bound is a grid minimum.** `speed_of_light_time` minimises over two 801-point grids
(`perfmodel.py:169–175`). The computed value is therefore *above* the true infimum by a small discretisation error,
which is the wrong direction for a lower bound. The optimum has a closed form: the objective is piecewise linear,
and the minimum is where $Lxp$ meets the compute term, or at $m_0 = ka/(a+b)$. Use it. It costs an hour and removes
an easy objection.

**P7. "No policy beats it" (L75) is a unit test, not evidence.** The simulated policies run inside the same cost
model as the bound, and the trace tiles 12 layers to 48 (`tests/test_bound.py:37–44`). Present it as a check of the
implementation. The empirical test is that no measured run exceeds its physical bound, as registered in A1. Report
that for *every* first-party run, including the 28 Sep ones.

### 3.3 Is the "speed-of-light" a bound where the paper uses it?

- **B1. Uniform-routing misses are not a bound.**
  - `scripts/audit.py:103` computes $M^\star$ from "independent uniform routing (approximation)" whenever no trace
    exists. Uniform routing has no locality, so it overstates misses; the resulting time is too high and is not a
    lower bound.
  - Four adjudicated rows use it: SeqMoE Qwen3.6-35B (two rows), SeqMoE DeepSeek-V4-Flash, and Pipelined sharding
    Qwen3-235B at 27.6 GB.
  - Their "fractions of SoL" (39%, 33%, 31%, 11%) are upper limits on the true fractions.
  - Fix: trace these open-weight models (the collector traced a 120B model on a 2-vCPU VM), or report
    $M^\star = 0$, which is trivially valid, next to the uniform value.
- **B2. Per-layer versus pooled budgets.**
  - A per-layer budget $C$ is a special case of a pooled budget $LC$, so the **global** bound is valid for every
    policy. Make it the headline denominator; keep the per-layer bound as the tighter one for per-layer policies.
  - The progress log's 174/313/313 tok/s uses the per-layer bound for FreeToken, whose cache is pooled.
- **B3. Workload mismatch.**
  - The audit's bounds use the author's own-text traces as a proxy workload, not the text each row actually
    evaluated. That is a ceiling for a representative workload, not the row's own bound. Say so.
  - Bound the sensitivity: compute the bound on the D, G and S arms and on each of the four datasets separately, and
    take the fastest proxy for a conservative fraction.
- **B4. Measured-bandwidth denominators.**
  - A1 already failed: 7 of 29 A100 configurations beat a floor built on STREAM Triad by up to 16% (anchor
    paragraph, `numbers4.tex`).
  - The 28 Sep fractions use "measured bandwidths".
  - Only the datasheet version is a bound, at the configured DIMM speed and populated channels (from `dmidecode`),
    at PCIe generation and width, and at GPU memory peak. Measured versions are **attainable ceilings** and must be
    labelled that way.
  - Where a measured host input is used, it should come from a read-only kernel at the engine's thread count and
    pinning, not Triad. The contract's "STREAM Triad (or read)" (Table 4) should say read-only.
- **B5. Quantised routing.**
  - Traces are fp32 routing on the original weights, while the systems route with Q4_K_M, MXFP4 or FP8 numerics.
  - The measured cache hit rates match the simulation to 0.001 (on the engine's own routing, via `ec-bench`), which
    suggests the effect is small.
  - Quantify it on one model: the share of top-k selections that differ, and the change in $M^\star$, between the
    fp32 trace and llama.cpp's own routing of the same text.

### 3.4 The decode model and its validation (L61–64, L78–82)

- **V1. Choosing the variant by the same CV is optimistic.**
  - M4 was selected by the same cross-validation that reports its error (L78). The CV medians of M0–M4 are 21, 17,
    17, 19 and 16% (`numbers.tex`).
  - With 12 groups, the 5-point gain from M0 to M4 is within the noise of a 12-fold CV median.
  - Use nested CV: an outer leave-one-source-out loop for the error, an inner one for selection. Report a
    group-bootstrap CI on the median error.
- **V2. Groups versus sources.** L78 splits one source's demand-fetch rows by GPU. The audit's band uses
  leave-one-source-out on static rows (`audit.py:50`). Use sources everywhere and report both.
- **V3. The relevant error is estimated on the wrong model mix.**
  - The audit only uses static-offload llama.cpp predictions. The matching validation subset is 25 static rows from
    4 engines; 12 are llama.cpp, and 8 of those 12 are gpt-oss-20b.
  - The audit predicts Mixtral, DeepSeek-V2-Lite, Qwen2-57B, Phi-3.5, Qwen3.6 and DeepSeek-V4, most of which appear
    in no static validation row.
  - Measured/predicted on the 25 static rows is **bimodal**: 14 rows at 0.69–0.96, 11 at 1.12–1.43, none in
    between. That points to a missing factor (engine, host class or NUMA), not random error.
  - The band [0.78, 1.31] is the 10th/90th percentile of 25 values, so each end rests on 2–3 rows.
- **V4. The efficiencies are not physical.**
  - $\eta_g$ = 0.38 (third-party fit) and 0.20 (the H5 calibration on the GH200) are not memory efficiencies. They
    absorb per-layer fixed costs (launch, synchronisation, small-kernel latency) that Eq. 1 lacks.
  - The GH200 all-GPU over-prediction (measured/predicted 0.28–0.92) says the same.
  - Add a per-layer (and per-kernel or per-graph-node) α term, as in α–β or LogGP models, and measure it by
    microbenchmark. $\eta_g$ should then come out at 0.6 or above and transfer across GPUs.
  - A model whose parameters are *measured*, not fitted, would make the audit's baseline a prediction rather than a
    regression. The A10 breakdown in the measurement plan already has the numbers: a fixed overhead $O$ of 26–38 µs
    per layer, and $f$ of about 50 µs per request.
- **V5. Two compositions.** Eq. 1 *sums* the GPU, CPU and PCIe terms, which is right for llama.cpp's alternating
  execution. The bound and the case-study model take the *maximum*, for concurrent execution. State which applies to
  which execution model, and that the audit uses only the sum.
- **V6. "Bytes are exact" (L64).** Model bytes are exact; bytes moved are not. H1 failed: ncu measured 12–15% more
  DRAM bytes than the GGUF on the A10, from inline-ECC over-read. Rephrase.
- **V7. H5 is a weaker test than the median suggests.**
  - 13 of the 29 test configurations belong to the calibration model's family.
  - Given the affine structure ($R^2 \ge 0.996$), gpt-oss-20b configurations between the two calibration endpoints
    are close to interpolation.
  - With 13 of 29 in that family, the median lands at the edge of it almost by construction.
  - The informative numbers are the 16 Qwen3 configurations: 3–45% error, under-predicted by up to 28%. Report
    per-family medians with CIs, and call the cross-family transfer failed.
  - Question for the author: are the two calibration runs among the 29? If so, remove them.
- **V8. How host bandwidth enters decides the sign of the error.**
  - With STREAM Triad as the host input, llama.cpp on the A100 runs 1.46× faster than predicted.
  - On the audit's basis (datasheet × $\eta_c$) it runs at 0.83×.
  - This is the largest single uncertainty in the audit. State it as such; experiments E3 and E4 are the remedy.
- **V9. A fourth platform exists but is unreported.** Job 059 (RTX 5090 + 9950X) measured llama.cpp at 1.22–1.27×
  the audit-basis prediction. If that prediction was committed before the measurement, it is a pre-registered
  out-of-sample point for §5. If not, report it as post hoc.

### 3.5 Pre-registration

- **R1. Verifiability.** The registrations live in a git repository the author controls, whose history can be
  rewritten. For the registrations still to come, use a third-party timestamp: OSF, a Zenodo DOI per registration,
  or at minimum signed tags pushed to a public remote plus an archive.org snapshot. Say in the paper which was used.
- **R2. Severity: several registered tests could not have failed, or had little power.**
  - F3 on a platform where the bound does not depend on the budget (§3.7).
  - H5 (V7).
  - P6, "median system ≤ 60% of SoL": a threshold far from the observed 24%. Say whether any estimate existed when
    it was registered.
  - H0, on a roofline whose inputs later proved beatable (A1).
  - Add a column "could this have failed?" to the log in App. C.
  - Keep the recorded failures prominent: H1, H2, H6, H7, H12, H13, P2, P4, A1, A2. They are the paper's
    credibility, and SPCL-type readers will value them.
- **R3. Pilot knowledge.** H5's ≤10% threshold was registered after a post hoc 4.9% on the A10 (L80). State what
  was known when each threshold was set.
- **R4. A knife-edge classification.** The 120b "format" class rests on a ratio of 1.48 against a registered 1.5×,
  with n = 35 conversations. Give a bootstrap CI. If it straddles 1.5, the registered classification is
  indeterminate, and the 21-conversation subset where both models reasoned (2.78 → 1.12) carries the argument.
- **R5. Mixed labels.** The abstract (L23) says "two of which cross our registered threshold" about models traced
  "outside the registration" (L89). Write "would cross the threshold registered for the four registered models".
- **R6. Attribution.** See the L32 row in §2.6.

### 3.6 The audit's adjudication rules (L94–107; `scripts/audit.py`)

- **A1. The band is an 80% interval, so false "weak" calls are expected.**
  - The band runs from the 10th to the 90th percentile (`audit.py:50–54`). If every reported baseline ran at the
    predicted strength, each row would still be called "weak" with probability about 10%: about 2 false calls
    expected among 20 rows.
  - State that rate.
  - Better: derive a prediction interval for a *new host and model* from a hierarchical error model, log(measured /
    predicted) = host effect + model-family effect + residual. Fit it on every static row, first- and third-party:
    25 third-party, 29 each on the A100 and GH200, plus the A10 runs, about 110 rows.
- **A2. The band-width filter selects by system.**
  - The ±40% rule (`audit.py:176–178`) drops a row even when its whole band lies on one side of 1. From
    `audit.json`, among the 30 rows not adjudicated:
    - 11 have a normalised speed-up (point estimate) above 1: all 6 FreeToken rows, all 4 ProMoE rows, and the
      llama.cpp fork row.
    - **6 have the *lower* edge of the band above 1** (5 FreeToken rows and the fork row). These gains would be
      established under any rule that asks the band to exclude 1.
    - 17 have the upper edge below 1.
  - FreeToken's reported llama.cpp baseline sits *inside* its band. So the one paper whose baseline looks like an
    equal-memory configuration is excluded from the headline.
  - Keep the registered rule. Also report, for all 52 rows, the rule "the band excludes the threshold", flagged as
    unregistered. That rule also makes the claim ≥ 1.2× filter unnecessary, since the band already covers it.
- **A3. "Weak" mixes three causes.**
  - The causes are configuration (`-ngl` or `-cmoe` instead of equal-memory placement), llama.cpp version, and host
    or predictor error.
  - On budget-0 rows, all experts are on the CPU in both the system and the equal-memory baseline, so configuration
    cannot be the cause.
  - **3 of the 16 weak calls are budget-0 rows:** KTransformers DeepSeek-V2.5 on the A100, KTransformers
    DeepSeek-V3 on the RTX 4080, and Pipelined sharding Qwen3-235B at 0 GB.
  - They cannot support the configuration message of findings (1) and (2). Report configuration-attributable calls
    separately.
- **A4. Judge each baseline by its own era.**
  - Dates, from the project's own `llamacpp_baseline_history.md`: `-ot` merged 2 Apr 2025 (b5028); `--n-cpu-moe`
    arrived 5 Aug 2025 (b6089).
  - HybriMoE (arXiv 8 Apr 2025) and KTransformers (llama.cpp of Feb 2025) predate or coincide with `-ot`. Pipelined
    sharding (Apr 2026) and SeqMoE (Sep 2026) come after both.
  - Judged by era and counting only configuration causes, the weak rows are **8 rows from 2 papers**: Pipelined
    sharding's two rows with a non-zero budget and SeqMoE's six.
  - That is a sharper and more defensible finding than "16 of 20": *papers from 2026 still compare against
    configurations from before 2025.* Report both readings: by today's llama.cpp and by the paper's era.
- **A5. Count papers, not rows.**
  - The 20 adjudicable llama.cpp rows come from 4 papers: HybriMoE 5, KTransformers 6, Pipelined sharding 3,
    SeqMoE 6.
  - The 8 established gains come from 2 systems: KTransformers 3, SeqMoE 5.
  - The 22 adjudicated rows come from 6 papers.
  - Configuration is chosen per paper, so report per paper first and per row second.
  - The same applies to the fraction of the bound. Per-system medians (from `audit.json`) are KTransformers 52%,
    SeqMoE 32%, CPU-GPU collaborative inference 19%, Pipelined sharding 11%, HybriMoE 8% and SP-MoE 4%. **The median
    of the system medians is 15%, not 24%**; the 24% row median is pulled up by six KTransformers rows.
- **A6. Show every analysis path.**
  - Three choices fixed after a first pass (L96) moved the median fraction from 19% to 24%.
  - Show a specification curve over: those three choices × predictor basis (third-party fit; with the A10 runs; the
    two platform-median scalings) × band rule. Plot the weak count, the established count and the median fraction,
    per row and per system.
  - I expect it to show the robust core: the 10 rows at or below half the predicted speed stay there under every
    specification.
    - Their reported/predicted ratios are 0.06–0.47: HybriMoE 5, Pipelined sharding 3, SeqMoE 2.
    - One of the 10 is a budget-0 row, so 9 are configuration-attributable.
    - Even after the DeepSeek-V2-Lite over-prediction correction, the lowest two stay at 0.14 and 0.33.
  - Make those rows (3 papers) the headline number, not "16 of 20 (13–19)".
- **A7. How reliable is extraction by agents?**
  - Agents extracted and agents checked. The check asked for corrections to 13 of 58 rows, and with two uniform
    rules applied afterwards, 19 of 58 rows changed. That is about a 22% first-pass correction rate, and the residual
    error after one check is not zero.
  - A person must verify every value that enters an adjudicated row (22 rows, 2–3 hours) plus a random 15 of the
    rest, and report field-level agreement.
  - Without this, a reviewer who finds one mis-extracted row will discount the whole audit.
  - The 52-row validation set needs the same check if it was also extracted by agents.
- **A8. Right of reply.** The notes to the 13 teams were drafted (commit `e3ab0c2`) but not sent. Send them at
  least two weeks before submission, and state in the paper that the authors were contacted and how they responded.
- **A9. Check the rows close to the ceiling.** KTransformers reaches 77% of a datasheet bound (L105, "the best
  77%"). The bound uses the top of the DRAM band (`audit.py`, `bw_hi`), which is correct. But STREAM rarely exceeds
  80–85% of peak, so any row above about 70% deserves one sentence confirming its host inputs.

### 3.7 The provenance study (L85–91, Table 1)

- **Pv1. F3 could not fire on the platform used.**
  - L89: "The bound does not move, because on our platform it sits at the bandwidth-aggregation optimum." That is
    the regime of consequence (i), where the bound does not depend on the budget. There F3 cannot fire for any
    trace; it is a tautology, not a test.
  - On the consumer-PC constants the paper now uses, the bound *does* depend on the budget at small budgets (174
    tok/s at C = 14 against 313 at C = 32).
  - A drop of 1.8–2.9 points in MIN-bypass hit rate at 12.5% (k = 4 for gpt-oss) is 0.07–0.12 extra misses per
    layer-step on $M^\star/N \approx 0.95$, about 7–12% more misses. That plausibly moves the bound past F3's 5%.
  - Compute F3 on the consumer constants. It costs nothing on GPUs; report it as post hoc.
  - Then either qualify "safe for bounds" by regime or drop it.
  - Also quantify "often" in consequence (i) (L75). The regime boundary is the dimensionless
    $M^\star/N < ka/(a+b)$, which depends on $B_g/B_c$: about 0.13 for gpt-oss on a 5090 desktop, about 0.8 on an
    A10 server. Plot a regime map (§5).
- **Pv2. Scope.** See §2.6. Cite "Not All Models Suit Expert Offloading" (ICLR'26) as the clearest instance of the
  practice, and "The Myth of Expert Specialization" (arXiv 2604.09780) for the mechanism. The direction of the result
  is then expected, and its size is the contribution.
- **Pv3. What is estimated, and how dependent the samples are.**
  - Hit rates come from one simulation over the concatenated session (`provenance.py:105–118`), credited to
    conversations. The bootstrap resamples conversations (`provenance.py:141–152`), although cache state carries
    across conversation boundaries.
  - State that the estimand is the token-weighted hit rate of a warm session, and that conversations are treated
    as exchangeable. That is roughly right, since cache memory is much shorter than a conversation.
  - Add a replay with a cold cache per request as a sensitivity check.
- **Pv4. Confounds.** Arm D and arm S differ in author (human versus model), length (up to 1024 own tokens) and
  format. Add a cross-model arm: model A teacher-forced on model B's own answers. It separates "not the model's own
  text" from "written by a person". It needs tracing only.
- **Pv5. The 120b routing claim is not shown.** L91 says "its routing there is not what deployment produces", but
  that is inferred from the loss alone: 120b's ΔDFA (−5.3) is indistinguishable from 20b's (−5.4). Either trace the
  dataset answers with the model's own reasoning inserted and show a routing difference, or limit the claim to the
  loss.
- **Pv6. Which policies to show.** DFA is the author's deployed policy. Show LRU (the most common in the literature)
  and MIN-bypass in the main figure, and DFA and LFU in the appendix.
- **Pv7. The own-text arms are near-own.** Some were generated with FP8 checkpoints (L133) and traced in fp32 on
  the original weights. Point to arm S's NLL under the trace model (0.17–0.78 nats, Table 1) as evidence that they
  are close.

### 3.8 The case study (§8, L110–114, Table 3)

- **No repeats, no CIs.**
  - The final A10 runs have no repeats, according to the measurement-plan report. llama.cpp moved −7.8% to +1.6%
    between instances, and the top of the reported range, 1.66×, comes from one llama.cpp run that was 7.8% slow.
  - Table 3 has no CIs.
  - That breaks Hoefler–Belli rules 5–7 in the one section meant to demonstrate them.
- **The "implementation-relative SoL" is a ceiling, not a bound.** It uses the GPU efficiency of a measured all-GPU
  run. Label it as a ceiling.
- **Superseded.** Move the A10 study to an appendix as the record of H16/H17, and replace §8 with the consumer-PC
  study (§6 of this review).
- **The baseline was never checked against stock llama.cpp.** It was the patched build's own `--ncmoe` inside
  `ec-bench`, and `--no-mmap` alone moved it by −13.4% to +1.7% (measurement plan). The 28 Sep work fixed this on
  the 5090 (harness within 1–3% of stock `llama-bench`). Say so for the A10 too, or drop the A10 numbers from the
  main text.

### 3.9 Compliance with Hoefler–Belli (SC'15)

| Rule | Status in the draft | Fix |
|---|---|---|
| 1. Report absolute values next to speed-ups | Mostly met | – |
| 2. Justify the benchmark subset | AIME-25, five problems, 256 tokens: not justified | E1: all 30 AIME-25 problems plus a chat set; state why |
| 3. Harmonic mean for rates | Broken: FreeToken's client averages per-request rates | Σ tokens / Σ time; show both |
| 4. Do not summarise ratios arithmetically | Audit medians of $S_n$ are acceptable; make per-paper summaries geometric | – |
| 5. Report nondeterminism with CIs | Broken: case study and same-host comparison | E1 |
| 6. Nonparametric CIs | Met for provenance (bootstrap) | Extend to everything |
| 7. Statistically sound comparison | Broken: same-host comparison | Paired bootstrap, TOST |
| 8. Justify the central tendency | Not discussed | – |
| 9. Document every varying factor | Host DIMM layout, fabric, client, cache state missing | Contract fields |
| 10. Describe how time is measured and summarised | Unclear (tok/s includes TTFT?) | Define decode time |
| 11. Show upper performance bounds | The core of the paper, but see §3.3 | – |
| 12. Plot enough information | One figure in eight pages | §5 |

---

## 4. Evaluation design

### 4.1 Priorities, by value per cost

Costs assume $0.35–0.9 per hour, **×1.7 for Vast storage and download charges.** The progress log found the ledger
under-counts spend by that factor: $7.62 actual against $4.40 in the ledger. Build a Docker image with prebuilt
binaries once, so that no rental pays to compile.

| # | Experiment | Objection it removes | GPU-h | $ (with overhead) | Priority |
|---|---|---|---|---|---|
| E0 | Analysis bundle (no new runs): nested CV and hierarchical band; bound recomputed in closed form, with shared experts in $\ell$, global budget, datasheet peaks at configured DIMM speed, $M^\star$ from traces for untraced models; per-paper and era-aware audit; specification curve; F3 on consumer constants; CIs for the 120b ratio and for H5 per family | §3.1–3.7 | 0–2 (tracing) | 0–3 | **1** |
| E1 | Same-host comparison with sound statistics, measured VRAM, equal tuning for FreeToken, a bound per system for its own text | "Inside noise"; conflict of interest; "equal memory is asserted" | 12 | 10–18 | **2** |
| E4 | Audit spot-check: equal-memory llama.cpp measured on the audit's GPU classes, with builds from each era | "The predicted baseline was never tested where it is used" | 8 | 7–12 | **3** |
| E3 | Host battery: 8 rentals; microbenchmarks, `dmidecode`, per-host predictions registered in advance; FETCH sign | "Two hosts is an anecdote"; "the FETCH sign flip rests on one host and one model" | 10 | 7–14 | **4** |
| E2 | Time breakdown from bound to measurement (nsys plus model) | "9–34% is a number, not a finding" | 3 | 2–5 | **5** |
| E5 | Context-depth sweep, predictions registered from the timeline model | The post hoc story for the chat-benchmark shortfall | 3 | 2–5 | 6 |
| E8 | Provenance extensions: cross-model arm, cold replay, quantised routing, 120b with reasoning | §3.7 | 3–4 | 3–6 | 7 |
| E7 | Per-layer α and bandwidth β measured on GPUs | $\eta_g$ = 0.2–0.38 | 1–2 | 1–3 | 8 |
| E6 | Second model in the same-host study (Qwen3.6-35B-A3B, FreeToken's own) | "gpt-oss-120b only" | 5 | 5–9 | 9 |

- **Total:** about 45–50 GPU-h, about $37–75 across the two tranches.
- **With the $17.38 left in tranche 1:** E1 on one host (about 6 GPU-h), E4 on the 5090 and 4090 classes (about 3
  GPU-h), and E3 on four hosts (about 5 GPU-h). Put the rest in tranche 2.

**Rules for every experiment below:**

- Report per-cell rates as Σ tokens / Σ decode time (equivalent to a token-weighted harmonic mean).
- Summarise ratios as geometric means of per-prompt ratios, paired by prompt.
- Use BCa bootstrap CIs, hierarchical when there are launches (resample launches, then prompts).
- Use TOST for equivalence claims, with the margin registered in advance.
- Fix n in advance; do not stop when the CI looks good.
- Commit every prediction as JSON before the run. Timestamp it with an OSF entry or a public tag.

### 4.2 Designs

**E1: the same-host comparison, done properly.**

- **Hypotheses.**
  - H-E1a: each system's decode rate is a stable fraction of its *own* global physical bound, with a 95% CI
    half-width of 3 points or less.
  - H-E1b: the timeline/step model predicts each cell within 10% median error.
  - H-E1c: in the direction registered from the current data, the cache with FETCH is faster than FreeToken's
    best-tuned mode at 11%, and equivalent within ±3% at 40%.
- **Independent variables.**
  - System:
    - llama.cpp at its best `-ncmoe`;
    - the cache, alone, with FETCH, and with FETCH and PREFETCH;
    - FreeToken offload and FreeToken hybrid;
    - FreeToken hybrid tuned: `ft bench bw` overlapped profile plus a `--moe-hybrid-max-fetch` sweep over five
      values including 0, commit `0d652e7`.
  - Budget: 11, 25 and 40% (44% for the systems that fit).
  - Host: the 62 GB/s and 46.6 GB/s RTX 5090 desktop classes.
  - Decode length: 256 tokens (primary), 1024 on a subset.
- **Dependent variables.**
  - Decode rate per request, from first to last generated token, excluding TTFT.
  - TTFT; inter-token latency P50, P90 and P99.
  - Total VRAM and expert VRAM, measured (`cudaMemGetInfo` around load; `nvidia-smi --query-compute-apps`),
    including FETCH staging and PREFETCH's 53 MB router matrices.
  - The generated token ids. Traced offline, they give each system's own $M^\star$, its bound and its misses.
  - GPU/CPU clocks and temperatures at 10 Hz.
- **Controls.**
  - FreeToken's unmodified client, with the author's harness as a cross-check (it already agrees within 1.3%).
  - Identical prompts, sampling parameters and seed.
  - A fresh server per launch; cell order randomised per launch (a Latin square over 3 launches).
  - A discarded warm-up request; cache state stated explicitly (warm), plus one cold arm.
  - Host microbenchmarks at the start and end of each session, to detect drift.
  - Nothing else running.
  - A conflict-of-interest statement, and the results sent to the FreeToken authors.
- **Sample size.** 30 AIME-25 problems (all of I and II) × 3 launches per cell.
  - Across prompts, rates spread by about ±10%. With a correlation of about 0.5 between systems on the same prompt,
    the SD of the per-prompt log-ratio is about 7%.
  - Then the SE is about 7%/√30 ≈ 1.3%, giving a 95% half-width of about 2.6%; launches add about 0.8%.
  - That resolves differences of about 4% or more. The 11% and 25% gaps will resolve. The 40% equivalence test may
    be inconclusive; say so in advance.
- **Statistics.**
  - As in the rules above. Also report FreeToken's arithmetic mean of rates, for comparability with its paper, and
    show the gap to Σ tokens / Σ time.
  - **Compare systems on model parameters, not only on tok/s.** Sampled text differs between engines, so each
    engine decodes a different workload. For each system, regress per-request decode time on misses per token (from
    the offline trace of that system's own text). This gives each system a fixed cost per token and a cost per miss,
    with CIs, and removes the workload difference that sampling creates.
- **Registered predictions.** Tok/s for every cell, from the timeline model with the host's measured constants and
  simulated misses (DFA for the cache, a pooled LRU for FreeToken, static placement for llama.cpp), plus the
  directions in H-E1c.
- **Cost.** About 21 cells × 3 launches × (30 × ~4.5 s + ~90 s to load) ≈ 4 h. Add 0.7 h for the 1024-token subset
  and 1 h to set up: about 5.7 h per host, about 11–12 GPU-h for two hosts. $10–18.
- **What changes the conclusion.**
  - The 11% difference's CI contains 0 once FreeToken is tuned: report parity and fractions only, no ordering.
  - FreeToken reaches a larger share of the global bound than the cache: say so plainly.
  - Any run exceeds its *physical* bound: stop. The proposition or its inputs are wrong.
  - Costs per miss agree within 10%: systems differ by policy (misses), not by implementation, which becomes the
    finding.

**E2: where does the time go?**

- **Hypothesis (registered).** For the cache, excess misses over $M^\star$ account for the largest share of the gap
  at 11% (deployed 1.8 misses per layer-step against MIN-bypass 0.95). Fixed costs per layer and per request
  dominate at 40%.
  - On the A10, the existing breakdown (measurement plan) attributes 37% of the excess to per-layer overhead $O$ and
    7% to DFA's extra misses. A *reversal* between the A10 and the 5090 is itself a "the host decides" result.
- **Method.**
  - nsys with CUDA trace plus host-side timestamps in the helper threads (CPU sampling may be blocked on Vast).
  - 3 systems × 3 budgets × 1 launch × 5 prompts × 256 tokens.
  - Credit time to the bound's terms, as a ladder of ceilings from the physical bound to the measured time:
    1. the physical bound;
    2. plus measured bandwidths;
    3. plus concurrent DRAM;
    4. plus per-layer and per-request fixed costs;
    5. plus the policy's misses minus $M^\star$;
    6. the measurement.
- **Control.** Profiler overhead of 3% or less (tok/s with and without nsys).
- **Cost.** About 3 GPU-h, $2–5.
- **What changes the conclusion.** If policy (misses) dominates, the paper says there is headroom in policy. If
  fixed costs and CPU kernel efficiency dominate, it says the headroom is in implementation. Either way, "systems
  run at 20–34% of the bound" gets a cause.

**E3: host battery (host effects, contention, FETCH sign; merges the recheck's items 6 and 9).**

- **Hypotheses (registered).**
  - H-E3a: with measured host read bandwidth, host-to-device bandwidth and their concurrent total, the model
    predicts llama.cpp and cache tok/s across hosts within 10% median error, and explains at least 80% of the
    variance between hosts.
  - H-E3b: the sign of FETCH's gain matches the model's prediction (from $b$, $p$ and the concurrent bandwidth) on at
    least 7 of 8 hosts.
  - H-E3c: on every Zen 4/5 desktop, the CPU keeps more than 50% of its standalone read bandwidth under concurrent
    DMA, against FreeToken's full-contention equation.
    - A likely mechanism, to check with `dmidecode` and the CCD count: a Zen desktop CCD reads through a fabric link
      of about 64 GB/s, while DMA reaches DRAM through the I/O die.
    - So "CPU-alone STREAM" can be a fabric limit, not a DRAM limit.
    - The host-to-host gap (46.6 against 62 GB/s) plausibly reflects DIMM population: 126 GB usually means four
      DIMMs at two per channel, at a lower speed. Maricq et al. (OSDI'18) found the same pattern in servers.
- **Independent variables.**
  - 8 rentals: 4–5 RTX 5090/5080 desktops on Zen 4/5 spanning one or two DIMMs per channel and several speeds,
    including one X3D; and 2 PCIe 4.0 desktops (RTX 4090 or 4500 Ada).
  - FETCH: off; the table; fetch-all (FreeToken's offload rule inside the cache); CPU-only.
  - PREFETCH: off; q = 1.
  - Budget: 11, 25, 40%.
  - Model: gpt-oss-20b on every host (forced budgets); gpt-oss-120b where host RAM is at least 96 GB.
- **Dependent variables.**
  - Microbenchmarks:
    - read-only bandwidth at 1 thread to all threads, with the engine's pinning;
    - STREAM Copy and Triad, for comparison with the anchors;
    - pinned host-to-device bandwidth by copy engine and by zero-copy kernel;
    - the concurrent CPU + DMA total.
  - System facts: `dmidecode -t memory` (configured speed, DIMMs per channel), `lspci -vv` link state, CPU and CCD
    count.
  - Decode rates: 12 prompts × 128 tokens of own text, A/B inside one process in both orders, as in job 060, which
    repeated within 0.4%.
- **Controls.** The same image, weights verified by hash, a fixed seed, and the registered predictions written to
  the log *before* the decode step of the same job, so that log timestamps order them.
- **n.** The host is the unit: 8 points, at least 2 per PCIe class. Report a slope and R² with CIs, the error with
  one host held out, a Spearman correlation between the measured and predicted FETCH gain, and an exact binomial
  test on sign agreement.
- **Cost.** About 1.2 h per host, including a 13 GB download (65 GB where the 120b runs): about 10 GPU-h, $7–14.
- **What changes the conclusion.**
  - H-E3a holds: the host is a *predictable* input. That justifies the contract field and the "host decides"
    headline, and makes unmeasured hosts the audit's main limitation.
  - H-E3a fails: the model is missing a host factor. Report that; no claim of predictability.
  - H-E3b fails: drop the "mechanism sign follows the host" claim.

**E4: audit spot-check (the predictor where the audit uses it).**

- **Hypothesis (registered).** On the audit's GPU classes, measured equal-memory `--n-cpu-moe` falls inside the
  audit's band for at least 80% of configurations.
- **Rows targeted.**
  - RTX 5090 (you already rent this class): Pipelined sharding, Qwen3-30B-A3B at a 5.7 GB budget.
  - RTX 4090: SeqMoE, Qwen3-30B-A3B at 13.1 and 4.4 GB. Use Q8_0 as a proxy for FP8 and say so.
  - RTX A6000: HybriMoE, DeepSeek-V2-Lite at 1.9 and 5.6 GB, Mixtral-8x7B at 5.8 GB, Qwen2-57B-A14B at 6.4 and
    19.1 GB. These are the five rows where the system itself is predicted slower than equal-memory llama.cpp, the
    audit's most consequential claim.
  - Optional, A100-40GB: SP-MoE, DeepSeek-V2-Lite.
- **Era arm, on the A6000.** At equal memory, run a Feb 2025 build (`-ngl` only), b5028 (`-ot`), and the current
  build (`-ncmoe`). This separates configuration from version with measurement.
- **Measurements.** `llama-bench -r 5` at depths 0 and 1024, a `llama-server` check on 5 prompts, and the host
  microbenchmarks.
- **Cost.** 3–4 hosts × about 2 h (downloads of 20–35 GB): about 8 GPU-h, $7–12.
- **A note on the author's 28 Sep decision** not to align the audit to each paper's hardware. This is not that. It
  measures llama.cpp only, on 3–4 rentable GPU classes, to validate the predictor where the audit relies on it.
- **What changes the conclusion.**
  - Measured llama.cpp is systematically below prediction (say 0.7×): weak calls shrink and established gains grow.
    Recompute and report.
  - Inside the band: the baselines become "predicted and spot-checked".
  - Equal-memory llama.cpp on the A6000 is faster than HybriMoE's reported system: the "system below equal-memory
    llama.cpp" category is confirmed by measurement, which it currently is not.

**E5: context depth (a test of the PREFETCH and FETCH mechanism).**

- **Hypothesis.** PREFETCH's gain grows with context depth, because attention time hides the copy. The timeline
  model registers numbers at depths of 128, 1k, 4k and 16k tokens.
- **Design.** Depth × PREFETCH {off, on} × FETCH {off, on} × 3 budgets; 12 prompts per depth, A/B inside one process
  in both orders, on one host. Instrument stall time on the join event.
- **Cost.** About 3 GPU-h, $2–5.
- **What changes the conclusion.** If the gain does not track attention time, the post hoc explanation for the
  chat-benchmark shortfall (+4–6% against +14–16%) is wrong. Remove it.

**E8: provenance extensions.**

- (a) The cross-model arm (tracing only, 2–4 GPU-h on a cheap 48 GB card, or on CPU).
- (b) A cold replay per request (CPU simulation).
- (c) Quantised routing against the fp32 trace on gpt-oss-120b (about 1 GPU-h on the 5090 host, using the
  `ec-bench` route dump).
- (d) A trace of 120b's dataset answers with its own reasoning inserted (1–2 GPU-h).
- **Cost.** $3–6.
- **What changes the conclusion.** If the cross-model arm is as local as the human arm, the effect is "not the
  model's own text", not "human text". That is a different sentence in the paper.

**E7: GPU fixed costs.**

- Fit all-GPU time = L·α + bytes/β on the existing A10 and GH200 all-GPU runs plus a one-hour sweep on the 5090.
  Measure α (per graph node and per kernel) and β (device read) by microbenchmark, and predict without fitting.
- **What changes the conclusion.** If the α term brings the GH200 all-GPU ratio from 0.28–0.92 to within ±15%, and
  $\eta_g$ becomes physical, adopt it for the audit's predictor and recompute.

**E6: a second model (only if time allows).**

- Qwen3.6-35B-A3B, FreeToken's own model, is the fairest choice. Use E1's protocol reduced to 20 prompts × 2
  launches on one host.
- **What changes the conclusion.** If the fractions or the ordering change by model, make no general statement
  about the systems.

---

## 5. Presentation

**Structure.** Build credibility from the bottom up: validate each instrument before applying it.

| § | Content | Pages |
|---|---|---|
| 1 | Introduction with the teaser figure (F1) | 1 |
| 2 | Background, including elapsed-time caching theory and the three September near misses | 0.75 |
| 3 | Model and bound: Eq. 1 with composition rules, Proposition 1 with the shared-DRAM term, the regime map (F2). Proof in App. A; trace exactness in App. B | 1.5 |
| 4 | Validation: third-party cross-validation (F3), H5 by family, an anchor summary table, the host battery (F6) | 1 |
| 5 | Consumer PCs: three systems against the bound (F1, F5), host effects (contention, FETCH sign), the registered PREFETCH prediction | 1.5 |
| 6 | Audit: flow diagram, method, per-paper results (F4), era, spot-check measurements, specification curve | 1.5 |
| 7 | Provenance (F7) | 0.75 |
| 8 | Pitfalls and contract | 0.75 |
| 9 | Limitations and conclusion | 0.4 |

**Figures missing.** There is one figure in eight pages. `figs/validation.pdf` exists and is not used.

- **F1, the teaser.** The speed-of-light curve for gpt-oss-120b on a 5090 host: tok/s against the expert budget
  (0–100%).
  - Lines: the physical bound (global and per layer) and the attainable ceiling with measured and concurrent
    bandwidths.
  - Measured points with CIs: the llama.cpp `-ncmoe` curve, the cache, FreeToken and all-GPU.
  - This one figure tells the paper's story.
- **F2, the regime map.** $B_p/B_c$ on the x-axis, $B_g/B_c$ on the y-axis. Regions: where the bound does not
  depend on the budget, and where splitting or fetching misses helps. Points: A10, A100, GH200, each 5090 host, and
  the PCIe 4 host.
- **F3, validation.** Predicted against measured, log–log, predictions from cross-validation with the source held
  out, coloured by source, first-party anchors as separate markers, band lines.
- **F4, the audit per paper.** A forest plot of $S_n$ with bands, grouped by paper, labelled by era. The current
  Fig. 1 moves beside it, or to the appendix. Add a strip plot of fractions of the bound.
- **F5, a waterfall per system and budget.** From the bound to measured time (E2). On the A10 and on the 5090, if
  the order of the costs flips.
- **F6, the host battery.** Measured against predicted tok/s across hosts, and FETCH gain against $B_p/B_c$, with
  markers for DIMM layout.
- **F7, provenance.** A forest plot of ΔLRU and ΔMIN-bypass per model with CIs. It replaces Table 1, which moves to
  the appendix.
- **Appendix figures.** A flow diagram for the audit (147 → 58 → 57 → 52 → 22) in the style of systematic reviews;
  the specification curve (A6).

**Tables to cut or move.**

- Table 3 (A10) → appendix.
- Table 1 → appendix, replaced by F7.
- Table 2 (22 rows) → a per-paper summary table (6 papers: era, configuration, below prediction, established,
  fraction of bound) in the main text; the rows go to the appendix.
- Keep Table 4 (contract), updated.
- Keep the pre-registration log in the appendix, with registration dates or commits and a "could it have failed?"
  column.

**Length and density.**

- The body runs about eight pages in a generic `article` class. Convert to the MLSys template now; check the 2027
  page limit and whether it is double-blind.
- Cut the abstract from about 320 words to about 180.
- Allow at most two numbers per sentence in the main text. L21, L80, L89, L94 and L107 carry five to eight each.
- `\anchorParagraph` (about 350 words in one macro, L82) → an appendix table plus a three-line summary in the text.

**Move to the appendices.** The proof; the trace exactness details; the anchors; the H-series details; the A10 case
study; the FETCH and PREFETCH mechanism details; the full 120b NLL analysis (keep 2–3 sentences in the main text);
the audit's admission rules; the method for measuring contention.

**Cut.**

- L122's pitfall, "Novelty is cheap to check …": it does not serve the thesis.
- L49's "Our earlier system … found the mechanism taken": rephrase neutrally, as in the paragraph proposed in
  `reports/How our system differs.md`.

**A strength to state.** Every number in the paper is generated by script (`paper_numbers*.py`) from released data.
Say so in one sentence; SPCL-type readers value it.

**Words.**

- Reserve "speed-of-light" for the physical bound; use "attainable ceiling" for anything built on measured
  bandwidth.
- "Below the equal-memory prediction" instead of "weak".
- "Same-host comparison" instead of "race".

**Anonymity.** The GitHub repository is public and ranks first in search. For a double-blind venue, make it private
or submit under the venue's preprint rules. The MLSys version needs self-references in the third person, no
repository link, and no "our earlier system".

---

## 6. Integrating the 28 September results

**The general rule.** None of the 28 Sep numbers should enter any version of the paper in their current form. They
rest on 5 problems × 1 measured request, no CIs, per-layer bounds for a pooled cache, measured-bandwidth
denominators, and flags rather than measured memory.

All five items belong in the paper, as the method applied to consumer PCs, once E1–E3 are done. They replace the A10
case study (§8), which moves to an appendix.

**The same-host comparison (jobs 062–068).**

- **Whether:** yes, after E1 and the bound recomputation in E0.
- **Where:** new §5, "Consumer PCs".
- **What it supports:**
  - The method works end to end, with a competitor's unmodified client as referee, within 1.3% of the author's
    harness.
  - Three systems as fractions of one bound, with a time breakdown.
  - Equal memory can be measured, and it matters: FreeToken cannot fit 44% because of its KV reservation.
  - A tuned equal-memory llama.cpp runs at 1.9–3.3× less than a cache on this host class, which bears directly on
    the audit's premise.
  - The external sanity check against moe-autopilot's community number (about 36 tok/s at `-ncmoe 25`) belongs in
    the text.
- **What it must NOT claim:**
  - that the author's cache is faster than FreeToken in general (one model, one context length, two hosts of one
    class);
  - any ordering at 25–40% before E1;
  - anything about FreeToken's paper numbers on other models.
- **Also required:** disclose that the cache is the author's system; give FreeToken an equal tuning budget; send
  the results to FreeToken's authors.

**The FETCH sign flip.**

- **Whether:** yes, as a *test of the model*, not as a feature.
- **Where:** §3 (the regime map) and §5 (host effects).
- **What it supports:**
  - Whether splitting misses between the link and the CPU helps is predicted by the ratio of $p$ to $b$ and by the
    concurrent DRAM bandwidth.
  - The gain was +2–6% serial on the 62/57 GB/s host, +13–17% serial (+14–18% overlapped) on the 46.5/46.5 host,
    and −16% on a PCIe 4 host.
- **Prerequisites (E3):**
  - At least 2 hosts per PCIe class and 2 models. The current negative result confounds the host (RTX 4500 Ada,
    PCIe 4) with the model (gpt-oss-20b rather than 120b).
  - A fetch-all arm and a CPU-only arm.
  - An ablation separating FETCH's two effects: bandwidth aggregation, and faster admission. Hit rate rose from
    0.639 to 0.682 in job 065, so fetched experts also warm the cache.
- **What it must NOT claim:**
  - that FETCH is new (FreeToken, HybriMoE and DALI came first);
  - that "no paper reports a loss from splitting" means it cannot happen in FreeToken, whose own `ft bench bw`
    switches modes on measured bandwidths.

**The contention measurement.**

- **Whether:** yes. According to the recheck, it is the most novel piece of the 28 Sep work.
- **Where:**
  - §3: justify treating the CPU and link terms as independent (a valid relaxation), and add the shared-DRAM term
    (S7);
  - §4: concurrency in the model;
  - §8: the "concurrent bandwidth" contract field.
- **What it supports:**
  - The first published numbers for CPU/DMA sharing of host DRAM on consumer PCs: the CPU keeps 37 of 62 GB/s under
    DMA, the link keeps 38 of 57, and together they deliver 73–77.
  - A plausible mechanism (the CCD fabric limit versus DRAM), to be confirmed in E3.
  - A quantitative correction to a closed-form contention assumption: FreeToken's paper equation predicts 5 GB/s
    where 37 was measured.
- **Prerequisites:**
  - The measurement method (`concur.cu`: thread counts, pinned memory versus zero-copy, duration, repeats).
  - At least 4 hosts (E3).
- **What it must NOT claim:** that FreeToken's *system* schedules wrongly. Its code already measures overlap, and
  its `measure_overlap_bw` says full contention "over-penalizes". Criticise the equation, credit the code.

**Prefetch: prediction against measurement (job 066).**

- **Whether:** yes, as a short worked example of predicting before measuring, applied to one's own mechanism.
- **Where:** §5, a paragraph or a box.
- **What it supports:**
  - The timeline model, with measured host constants, predicted +17/+17/+14% and measured +14/+16/+10%: 1–4 points
    optimistic.
  - Hit rates matched the simulation to 0.001.
- **Report honestly as well:** the prediction did not carry over to the short-context chat benchmark (+4/+6/−2/−3%).
  The explanation, that attention time hides the copy, is a hypothesis until E5 tests it.
- **What it must NOT claim:**
  - that PREFETCH is new;
  - that its recall (84/97% at top-4/top-8) is better than Speculating Experts' (different estimator, model and
    text);
  - a general +14–16% gain.

**Host-to-host variance.**

- **Whether:** yes, as a pitfall now, and as figure F6 if E3 has at least 6 hosts.
- **Where:** §8 pitfalls (merged with the existing A10 "instances of one type are not one machine" item) and §5.
- **What it supports:**
  - On bit-identical work (hit rates equal to six digits), two rentals of the same GPU/CPU class differ by 19–22% at
    46.6 against 62 GB/s. Re-runs on one host repeat within 0.8%, and a third rental's link ran at only 28.9 GB/s.
  - The per-miss slope scales with DRAM bandwidth (215 µs at 62 GB/s, 286 µs at about 46 GB/s; the ratio 1.33
    matches 1.35). That is the model's prediction, and the strongest single piece of evidence that measured host
    bandwidth is the right input.
- **What it must NOT claim:**
  - a characterisation of the rental market from 2–3 hosts;
  - any attribution to a provider;
  - that results from different hosts can be pooled.
- **Also:** cite Maricq et al. (OSDI'18) and Sinha et al. (SC'22) as precedent for the principle; what is new is
  the setting.

**Small items worth one sentence each:** stock `llama-bench` against the harness (within 1–3%); `setup.sh` retries
after the network reset; the harness agreeing with FreeToken's script (within 1.3%). These belong in the
methodology, as evidence that the measurement chain was validated.

---

## 7. Prioritised revision checklist

Two targets: an arXiv preprint for the ETH application by 27 Oct, and the MLSys submission by 30 Oct. The arXiv
version is not anonymised and links the repository. The MLSys version is anonymised and in the template.

**Week 1 (29 Sep–5 Oct): soundness and text, no GPU needed.**

1. **Fix the three false sentences and five imprecise ones** (§2.6; recheck table). *0.5 day.*
2. **Add the 33 missing citations**, including a bib entry for Pipelined Sharding. Position the bound against Cao,
   Albers–Garg–Leonardi, CHOPT, Jain & Lin, Budgeting Bytes, Paging the Experts and WiSP v2. Write the "why bound,
   not solve" paragraph. *1.5 days.*
3. **Bound:**
   - the full proof in App. A (P1–P4);
   - the shared-expert fix (S1);
   - scope statements (S2–S5);
   - the closed-form minimisation (P6);
   - the shared-DRAM term (S7);
   - the global bound as headline denominator (B2);
   - datasheet peaks at the configured DIMM speed (B4).

   *2.5 days.*
4. **Replace the uniform-routing $M^\star$** by tracing Qwen3.6-35B-A3B, DeepSeek-V4-Flash and Qwen3-235B (or
   report $M^\star = 0$ beside it). Exclude the speculative rows from the fractions, or extend the bound to them.
   *1 day (plus a few GPU or CPU hours of tracing).*
5. **Validation statistics:** nested cross-validation with the source held out, group-bootstrap CIs, a
   hierarchical host/model error model and band, H5 reported per family. *1.5 days.*
6. **Audit:**
   - per-paper reporting first;
   - era-aware and configuration-only labels (A3–A4);
   - the band-excludes-threshold sensitivity for all 52 rows (A2);
   - the specification curve (A6);
   - neutral wording;
   - the 10 rows at or below half the predicted speed (0.06–0.47; 9 configuration-attributable) as the robust
     headline.

   *1.5 days.*
7. **Human check of all 22 adjudicated rows, plus a random 15 others**, reporting agreement. **Send the notes to the
   13 teams now,** to leave two weeks for replies. *0.5 day of the author's time.*
8. **Provenance:** scope the wording; F3 on consumer constants; the CI for the 120b ratio; the cold-replay
   sensitivity; LRU in the main figure. *1 day.*
9. **Register E1–E5 with third-party timestamps** (OSF or Zenodo), writing the predictions from the model *before*
   renting anything. *0.5 day.*

**Week 2 (6–12 Oct): tranche 1 GPU work (about $17).**

10. **E1** on one 5090 host (30 problems × 3 launches, equal tuning for FreeToken, measured VRAM, offline traces of
    each system's own text). *2 days including analysis; about 6 GPU-h.*
11. **E4** on the 5090 and 4090 classes (Pipelined sharding and SeqMoE rows), with the era builds. *1 day; about
    3 GPU-h.*
12. **E3** on the first 4 hosts. *1.5 days; about 5 GPU-h.*

**Week 3 (13–19 Oct): tranche 2 and figures.**

13. E1 on the second host class; E3 up to 8 hosts; E4 on the A6000 (HybriMoE rows). *3 days; about 15 GPU-h.*
14. E2 (time breakdown) and E5 (context depth). *1.5 days; about 6 GPU-h.*
15. Make F1–F7; restructure per §5; convert to the MLSys template. *2 days.*

**Week 4 (20–27 Oct): writing and internal review.**

16. Rewrite the abstract, introduction and contributions per §2.3–2.5, with the numbers from items 5–14. *1.5 days.*
17. Integrate the replies from audited teams. Record in the paper the teams contacted and any corrections. *0.5 day.*
18. **Internal red team.** Ask someone outside the project to try to beat the bound with any measured run, and to
    find one mis-extracted audit row. Fix what they find. *1 day.*
19. **arXiv on 27 Oct:** the repository at a tagged release, a one-command reproduction of every number, and a
    README that maps each figure to a script. *0.5 day.*

**28–30 Oct: MLSys.**

20. **Anonymise:** the repository private or linked anonymously; third-person self-references; remove "our earlier
    system"; check the CFP's rules for preprints and LLM use. Submit. *1 day.*

**If time runs out, drop in this order:** E6, E7, E8, E5, and the second E1 host. **Never drop:** items 1–9, E1 on
one host, E4 on one class, and E3 with at least 4 hosts.

**For the ETH audience specifically.** Everything this paper does well is what SPCL values: a bound, data movement,
predictions made before measuring, and failures reported. Overclaiming is what would hurt it. With items 1–8 done,
the preprint shows research judgement even if the GPU experiments are only partly finished. Without them, a careful
reader at ETH will find the S1, B1 and A2 problems in an afternoon.
