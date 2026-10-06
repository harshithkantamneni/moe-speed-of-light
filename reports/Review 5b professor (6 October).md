# Review: "Where the Seconds Go: A Speed Limit and a Measured Accounting for Mixture-of-Experts Decode on One Consumer GPU"

Reviewer: professor, performance modelling and scientific benchmarking of parallel and ML systems. Blind, independent review, written for a top systems venue (MLSys calibration) and as evidence of research ability. I read the full 41-page PDF and checked claims against the repository (`prereg/`, `scripts/`, `mosl/`, `tests/`) and the results branch (`/home/claude/gpu-branch`: `jobs/`, `results/`). I did not read anything under `reports/`.

---

## 1. Summary of the paper

The paper studies batch-1 decode of MoE models that do not fit on a consumer GPU (gpt-oss-120b in MXFP4 and Qwen3-30B-A3B in BF16 on rented RTX 5090, 4090 and 3090 hosts), with most experts in host DRAM. It makes four contributions.

1. **A speed limit (Section 3, Eq. 2).** A lower bound on time per token for any exact-routing system with C GPU slots per layer. It takes the fewest host expert reads any policy can make (Belady's MIN with bypass on the model's own routing trace), divides them by the machine's highest probed host-read rate, and combines that with the GPU's datasheet bandwidth term. The bound is then tightened four ways (Table 3).
2. **An engine (Section 2).** A 3,500-line llama.cpp expert-cache patch with a calibrated per-machine time model (Eq. 1, "the law") that picks how many misses to copy over PCIe and how many to run on the CPU. It runs 2.0–4.0× stock llama.cpp, leads FreeToken at 11 of 12 cells on two RTX 5090 hosts, and reaches 25–43% of the limit. In an audit, published systems "in its class" reach a median of 13.6% of a datasheet version of the bound.
3. **Oracle policies inside the engine (Section 4, Appendices E–F).** These read future routing from a recorded trace and are run as a randomised factorial on several hosts. MIN-with-bypass admission with each admitted expert read once is worth 16–51%, and 25–81% with a paced prefetch. Belady prefetch and serve-then-copy admission waste the foresight. A measured decomposition of the gap gives bytes 26–60%, overlap 7–23%, and an unexplained rest of 30–62%.
4. **A trace study of nine models (Section 5).** Online policies (LRU, LFU, decayed frequency, ARC, S3-FIFO) read 35–110% more than MIN. The foresight horizon needed to close half that excess fits W50 ≈ 0.59 (C/k)^1.33. Batched speculative verification does not supply the foresight. A learned admission order trained offline removes 17–30% of excess reads, but in the engine it runs only 0.90–1.05× the deployed policy.

Every GPU job's predictions were committed before the job ran, and 565 prediction clauses are scored in Appendix B: 218 held, 193 held on the point estimate only, 142 failed.

---

## 2. Scores

| Criterion | Score | Comment |
|---|---|---|
| Novelty | **3 / 5** | Each ingredient is known: Belady-with-bypass as a cache oracle, bytes-over-bandwidth bounds for offloaded decode, CPU/PCIe miss splitting, Shapley attribution. The genuinely new parts are the oracle factorial run inside a real engine, the per-machine two-path bound with probed ceilings, and the empirical W50 scaling. |
| Technical soundness | **3 / 5** | The bound is valid given its assumptions, and MIN-with-bypass optimality is tested against exhaustive search. But the "limit" mixes probed and datasheet ceilings. The measured accounting depends on the order of attribution, and that dependence is not reported. The "law" is a calibrated regression whose intercept exceeds the whole all-in-VRAM token, and on held-out hosts it does not beat a naive model. |
| Experimental rigor (bounds, statistics, pre-registration, replication) | **4 / 5** | Far above the MLSys norm: predictions are ordered before measurements in git, configurations are randomised and run cold, comparisons are paired, there are confirmation launches, a competitor is tuned, and raw data are public. Held back by intervals that cover only within-launch problem sampling, while between-host effects are 5–10× larger; by a hand-encoded scorecard; and by two analyses that were computed but left unreported (Weaknesses 4 and 5). |
| Clarity | **2 / 5** | The main text is very hard to read. Sentences carry 5–10 numbers each, the vocabulary is idiosyncratic ("cell", "law", "serve-then-copy", "foa", "both3p"), and the argument cannot be followed without the appendices. 17 of the 41 pages are a clause table. |
| Significance | **3 / 5** | Local batch-1 MoE inference is a real and growing setting. The take-aways are useful: admission decisions informed by about 10 tokens of foresight are the lever, and better online policies and learned orders are not. But the paper stops at a diagnosis, with no realisable source of foresight. |
| **Overall** | **5 / 10: weak reject** | Solid and unusually honest empirical work, packaged so densely that a program committee cannot verify it, with framing that runs ahead of the evidence in three places (the "law", the headline comparison with published systems, and "where the seconds go"). A focused rewrite fixing Weaknesses 1–5 would, in my view, be a clear accept. |

---

## 3. Strengths (ranked)

1. **Falsifiability and transparency of a kind I almost never see in systems papers.**
   - Every GPU job carries its predictions in a script header that git orders before its results. I checked jobs 081, 093, 096 and 097: the script commits precede the results commits.
   - The paper reports 142 failed clauses rather than hiding them, including the learned order failing on host O6 (0.90–0.94× against a predicted 1.01–1.18×).
   - Numbers are generated from `prereg/*.json` by scripts. I reproduced Table 2's host-B intervals bit-for-bit from the raw per-problem rows (`results/081_headline_law@vast/bs1.jsonl`, 10,000 paired resamples): 1.275 [1.254, 1.295], 1.294 [1.278, 1.312], 1.154 [1.134, 1.172].
   - The whole study cost $55.94 over 66 rentals (checked against `gpu/vast_ledger.json`).
2. **The in-engine oracle factorial (Section 4, Fig. 1, Tables 10–13).** Rather than simulating oracles, the author builds them into the engine. The design crosses "what to admit" (MIN with bypass vs. Belady prefetch) with "how many reads" (one vs. two) and with overlap, at six cells. Order is shuffled per cell with a recorded seed, each configuration starts with a cold cache, and it runs on two hosts plus a replication launch. The qualitative result is clean and believable:
   - The lever is MIN-like admission informed by the future, with each admitted expert read once.
   - The online policy's own second read costs almost nothing ("foa" runs 0.97–1.03×).
   - Belady prefetch wastes the foresight through over-admission.

   This is a genuine contribution to how cache policies for offloaded MoE should be evaluated.
3. **Bound-first framing with a sensitivity table.** Table 3 shows the limit under no-evict, pooled, median-probe, measured-GPU and per-layer variants. The gap is placed against a ceiling, not a convenient baseline. Optimality of the MIN-with-bypass simulator is verified against exhaustive search in `tests/test_cachesim.py`. In the same spirit, the paper scores the engine's distance to the bound, not just its speed-up.
4. **A strong instrument with a fair competitor treatment.**
   - The engine leads FreeToken at 11 of 12 cells. Re-tuning FreeToken over five settings on a third host (job 098) moves it by at most 1.04×, and FreeToken is given more VRAM (28.4–30.1 GiB against ours at 9.5–27.3).
   - Output parity is checked with teacher-forced KL against an all-VRAM reference (mean 0.0019 and 0.0005 nats, top-1 agreement 98.5% and 99.3%; checked in `prereg/parity_090.json`).
   - Results are extended to long outputs, three card types and FreeToken's own headline model.
5. **A broad trace study with careful negative results.**
   - Nine models and six online policies, including S3-FIFO and ARC. I checked that the best online policy reads 1.35–2.10× MIN across the 27 cells, and that decayed frequency is best in 12 cells and S3-FIFO in 10.
   - Hot-set drift is quantified.
   - The paper shows why batched speculative verification cannot stand in for foresight once rejected drafts are charged.
6. **Self-critical limitations.** Section 7 states most of the caveats I raise below, though often in a weaker form than they deserve.

---

## 4. Weaknesses (ranked)

### W1. The paper cannot be read at the bar of the venue
- **Where.** The abstract and Section 1 (the "Thesis" paragraph), Sections 4–5, Figure 1, Tables 10–11, and Appendix B (pp. 12–28).
- **Why it matters.**
  - The abstract alone carries about 15 numeric ranges.
  - Key terms are coined, not standard: "law" for a fitted regression, "speed limit", "cell", "fetch" vs. "admission", "serve-then-copy", "paced", plus script labels such as foa, nb2, both3p and hitoptp in the tables.
  - Single sentences routinely chain four or five clauses, each with a range and a host.
  - Figure 1's row labels are clipped ("elady, two reads, paced"), and Table 10 is unreadable at print size.
  - A reviewer cannot tell which of the ~200 numbers in the main text carry the argument. At MLSys this alone sinks the paper, independent of the science.
- **Cost / fixability.** High effort, but entirely within the author's control.
  - Choose one thesis.
  - Give the main text four exhibits: the bound, the factorial, the trace/W50 figure, and the engine table.
  - Summarise the scorecard by type and era in one table and leave the 565 rows to the artifact.
  - Use standard vocabulary: "calibrated model", "demand fetch", "prefetch", "admit on miss".

### W2. The reported uncertainty does not cover the variance that matters for the claims
- **Where.** Section 2 "Protocol" (p. 3), Section 4 (p. 5, "every interval is within ±0.037"), Section 7 "Hardware", and Appendix B.
- **Why it matters.**
  - Intervals cover problem resampling within one launch on one machine. By the paper's own account, two rentals of one CPU model differed by 19–22% in speed.
  - Host S's ratios came out 0.06–0.11 below host B's against a pre-registered ±0.06 (job 089 P2: failed at 5 of 6 cells).
  - Fetch's gain differs between O4 and O5 by up to 0.26 against a predicted ≤ 0.15 (096 P10: failed at 5 of 6).
  - So the between-machine component is 5–10× the interval half-widths. Claims phrased as "at every budget on three hosts", with ±0.02 intervals, therefore convey false precision about what another reader's machine will show.
  - The problems within a session are also not exchangeable: the cache carries state from problem to problem in Table 2's runs, and the bootstrap ignores that dependence.
  - In Hoefler–Belli terms (rules 5, 7 and 9), the experimental unit should be the machine. Rule 9 is documented here; the variability that rules 5 and 7 are about is not.
- **Cost / fixability.** Cheap, given the author's own cost model (about $1 per rental-hour).
  - Rent at least 5 machines per CPU class.
  - Use a hierarchical (host, then problem) bootstrap or a mixed model.
  - Report variance components (host, launch, problem).
  - State per-host ratios as a distribution.

### W3. The headline comparison with published systems uses a different bound
- **Where.** Abstract ("reaches 25–43% of this limit, and published systems in its class a median of 13.6% of theirs"), Section 3 "Applying it", and Appendix H / Fig. 7.
- **Why it matters.** The cache is scored against a bound with the probed shared host rate (87.5 GB/s on host B), with the CPU and PCIe paths sharing it. Published rows are scored by `mosl.perfmodel.speed_of_light_time`, which differs in three ways:
  - it uses the top of the datasheet DRAM band;
  - it treats the CPU path and the PCIe path as independent resources (`b` and `pf`), so their rates add;
  - it charges dense work in sequence.

  The repository's own note says the two sets are "not a like-for-like ranking" (`prereg/audit/audit_sol.json`), yet the abstract juxtaposes them. I scored the paper's own host-B cells with the audit's procedure, using the same R*, ctx = 416, PCIe 63 GB/s, and the 89.6 GB/s DRAM top the audit uses for FreeToken's desktop RTX 5090 row:
  - Our cache then sits at **27–30% at the four host-bound cells** (paper: 25–42%) and 36–42% at the GPU-bound ones, **22–42% overall**.
  - The engine still clears the 13.6% median comfortably, so the qualitative claim survives. The specific numbers do not.
  - The audit's limit at gpt-oss 40% (364 tok/s) also differs from Eq. 2's (519 tok/s). The "speed limit" is model-dependent by up to about 1.4×, which should be stated.

  Separately:
  - The in-class median depends on class-membership judgments: Pipelined Sharding's sub-layer placement, and a Radeon RX 7600 llama.cpp PR on a "Q4_K_M derivative (architecture assumed identical)".
  - The published tok/s come from heterogeneous protocols (prompt lengths, output lengths, whether prefill is included).
- **Cost / fixability.** Cheap. Score our engine with the identical audit procedure and put both numbers in Fig. 7. Remove the 13.6% from the abstract, or pair it with the like-for-like number. Report the median's sensitivity to class membership.

### W4. The measured accounting depends on attribution order, and the alternative order is computed but not reported
- **Where.** Section 4 "The accounting, measured" (p. 6), Table 11, and Appendix E.
- **Why it matters.** Appendix E rightly argues that "adding them in a fixed order makes the first one added look largest" and uses Shapley values for the modelled accounting. The measured accounting then uses a fixed order: online → fetch counts as "bytes", and fetch → faster prefetch state counts as "overlap".
  - `scripts/accounting_measured.py` also computes the other feasible order (`measured_overlap_first`), and macros `amOvlFirst*` / `amBytesGivenOvl*` are generated, but neither appears in the paper.
  - In that order, the overlap share at the host-bound cells is negative at four of twelve: −161% at O5 Qwen3 12.5%, −107% at O5 gpt-oss 11%, −84% at O3 Qwen3 12.5%, −68% at O4 Qwen3 12.5%. The bytes share then rises to as much as 199%.
  - A two-order (two-player Shapley) average gives **bytes 37–115% and overlap −77% to +27%**, against the reported bytes 26–60% and overlap 7–23%.
  - To the author's credit, the fixed order was pre-registered (job 096, prediction 9), so this is not p-hacking. But the finding "overlap allows 7–23%" is not robust to order. The strong interaction (overlap without foresight on bytes hurts at low budgets) is itself the interesting result and should be stated.
  - The "rest" (30–62% of the gap) is the largest single bucket at many cells and is not separated. The title "Where the Seconds Go" promises more than the measurement delivers.
- **Cost / fixability.**
  - Report both orders and the two-player Shapley values with intervals; this is cheap.
  - Split the "rest" with Nsight timelines of the oracle states on O-hosts (one session already exists for host B at Qwen3 43.75%); this is moderate.
  - Or retitle.

### W5. The "law" (Eq. 1) is a calibrated regression, and on held-out hosts it does not beat a naive model; that comparison is computed but unreported
- **Where.** Section 2 "The instrument" (pp. 2–3), Appendix C, Table 8, and `prereg/law_frozen_later.json`.
- **Why it matters.**
  - The intercept G (4.82 ms frozen, 5.17 ms refit) exceeds the whole all-in-VRAM token time (3.92 ms). The paper says G "absorbs a shortfall the probe does not see", which means it is not GPU work.
  - On the blind test, the paper reports that a one-path bytes-over-bandwidth model with the same G has "the same median but a 90th percentile of 31%". That supports the two-path max.
  - On the 30 later held-out measurements (4 hosts), the repository computes the same naive baseline: **median 6.1%, 90th percentile 10.5%, max 16.0%**. The law scores **5.6%, 15.5% and 21.4%**. The macros `lawFrozenNaiveMed`/`lawFrozenNaivePninety` are generated but never used in the paper.
  - The law also over-predicts Qwen3's speed by +9%, +11% and +17% at the three budgets (median biases). That is a systematic, budget-dependent error, not noise, and it points to a missing term.
  - Reporting the baseline only where the law wins is selective, even if unintentional.
  - The model's real merit is practical: its per-machine FETCH table gains 3.3–8.0% on host B and 21–34% on slow-link hosts. That merit stands without calling it a law.
- **Cost / fixability.** Cheap.
  - Rename it to "calibrated model".
  - Report the naive baseline on every held-out set.
  - Measure G directly (expert-free GPU time per token) or decompose it.
  - Report errors on the variable part of the time, not on a total of which G is 25–60%.

### W6. The "speed limit" mixes ceilings of different provenance, and that changes the regime classification
- **Where.** Section 3 (p. 4), Table 3, Table 9 "Bound" column.
- **Why it matters.**
  - B_host is "the highest rate the probe reached, not a hardware peak". B_gpu is the datasheet rate, which batch-1 decode reaches only 52–61% of.
  - The result is a limit that is loose by about 1.9× at GPU-bound cells (519 vs. 278 tok/s for gpt-oss 40% with the measured GPU rate). There the engine's "29%" becomes 55% under the "all" tightening.
  - On host B, the 25% cells are labelled host-bound only under the datasheet GPU rate: Table 3's measured-GPU column gives 279 < 430 and 192 < 214. With the realistic rate they bind on the GPU.
  - The Section 4 narrative and Fig. 4's Shapley ranking are scoped to "host-bound cells", so this matters. On the O-hosts (about 50 GB/s) those cells are host-bound either way, but the paper does not say so.
- **Cost / fixability.** Cheap. Make the "all" column (or the measured-GPU variant) the headline, report the datasheet version as an upper envelope, and classify regimes under the realistic ceiling.

### W7. Pre-registration practice is valuable but not yet confirmatory science
- **Where.** Appendix B, Table 7, and `prereg/scorecard_clauses.json`.
- **Why it matters.**
  - The clause split and every status were hand-encoded after all outcomes were known: the file's `meta.written` is 2026-10-06, the day the PDF was built, and `scripts/scorecard.py` reads the status from the file rather than computing it. I re-scored the 180 clauses with simple numeric thresholds. 171 agree with the stated rule. The 9 that differ are deterministic counts scored "held" outright, which is consistent with the deterministic-clause rule.
  - "Held (point)" lumps two things together: estimates whose interval crosses the threshold, and clauses with no interval at all. In jobs 088–098, 126 of the 211 band "holds" are point-only.
  - There is no designation of primary hypotheses and no multiplicity consideration.
  - Most clauses predict the author's own simulator and code rather than scientific hypotheses.
  - The flagship numbers in the abstract (the 13.6% audit median, W50, the measured accounting shares) are exploratory analyses of traces and existing runs, not pre-registered tests.
  - The resulting calibration data are informative: directions held, magnitudes often failed (90 of 301 bands failed in 088–098). The paper should say plainly that the author's models predict signs well and sizes poorly.
- **Cost / fixability.** Cheap to moderate.
  - Compute statuses automatically from the sources each clause names.
  - Separate "no interval" from "interval crosses".
  - Designate a handful of primary confirmatory hypotheses per study.

### W8. Workload and generality are narrow
- **Where.** Section 2 "Workload", Section 7.
- **Why it matters.**
  - Engine measurements use the 30 AIME-25 problems, which were "also used while developing the system", decoding 256 tokens. Held-out long-output tests are also math (AIME 2022, MATH-500).
  - Routing locality, and therefore every cache result, is domain-dependent; the trace study's mixed-domain corpus partly mitigates this.
  - There are two models in the engine and a single engine family (llama.cpp).
  - All hosts are desktop Ryzen and Intel, apart from one EPYC that breaks the model.
- **Cost / fixability.** Moderate: add chat and code workloads with a strict development/test split, one more model family (a DeepSeek-style model with shared experts), and ideally a second platform class.

### W9. The W50 "law" is an empirical fit, and is fragile
- **Where.** Section 5 "How far ahead", Fig. 5.
- **Why it matters.**
  - The fit has 26 points from 9 models, over C/k from 1 to 16.
  - The exponent rises from 1.33 to 1.51 when the six interpolated points below one token are dropped, with a model-bootstrap 95% CI of 1.15–1.74 (`prereg/foresight/w50_exact.json`).
  - The fit under-predicts the paper's own running example at C/k = 16 by about 30% (fit 23.6 vs. measured 33.6 tokens).
  - r = 0.971 in log–log over 1.2 decades is not strong evidence of a power law.
- **Cost / fixability.** Cheap. Call it an empirical fit, show prediction intervals, and test it on held-out models.

### Minor
- The README says "every number in it is a macro generated by the scripts", but the key "19–22%" rental-variance number in Section 2 is hard-coded, as are a few protocol constants.
- Table 2 assembles the host-B block from three jobs (080, 081, 082). I confirmed they ran on the same physical GPU (identical UUID). The two Qwen3 "Ours ÷ llama.cpp" entries are cross-job point ratios without intervals, and this should be stated.
- Contribution 1 ("…the machine's measured ceilings, varied four ways, stated for pooled caches, and placed beside our cache…") packs four ideas into one clause and is hard to parse. Throughout, "law", "speed limit" and "accounting" are marketing-flavoured for what are a calibrated model, a conditional bound, and a partial decomposition.

---

## 5. Questions for the author

1. Which two rentals of which CPU model differed by 19–22%, in which configuration? Can you give a variance decomposition into host, launch and problem for at least one cell?
2. Appendix E argues against fixed attribution orders. Why does the measured accounting use one, and what is your reading of the overlap-first path that your script computes, with its negative overlap shares at the lowest budgets?
3. Why is the naive one-path baseline not reported on the 30 later measurements (`law_frozen_later.json`: p90 10.5% for naive vs. 15.5% for the law)? What explains the law's budget-dependent over-prediction on Qwen3 (+9%, +11%, +17%)?
4. What is physically inside G (4.8 ms) that exceeds a whole all-in-VRAM token (3.9 ms)? Have you measured expert-free GPU time per token directly?
5. Eq. (2) makes CPU and PCIe reads share B_host, while the audit's bound treats them as independent. Which is physically right on AM5 (one IO die, two DDR5 channels)? Did any probe exceed max(B_c, B_p) by more than the measured combined rate?
6. On X3D parts with 96–128 MB of L3, is the background copy in serve-then-copy really a second DRAM read, or can the DMA read of a just-used 13 MB expert be served from cache? Have you checked DRAM traffic with uncore/DF counters? This bears directly on "two reads" and on why the online policy's second read "costs little".
7. On host B, are the 25% cells host-bound under the measured GPU rate (Table 3: 279 < 430 and 192 < 214)? How does Fig. 4's attribution change if the limit uses the measured GPU rate?
8. How sensitive is the 13.6% in-class median to class membership (Pipelined Sharding, the RX 7600 PR) and to the four rows with imputed capacities?
9. How does W50 vary with domain (code, chat), context length and sampling temperature? Are teacher-forced greedy traces representative of sampled decoding?
10. Has anyone from the FreeToken team checked your configuration of their system, in particular the backend carry-over and their 28–30 GiB VRAM reservation?
11. Can the scorecard statuses be computed mechanically from the `source` field of each clause, so that the scoring rule, rather than a hand encoding, produces Table 7?
12. What realisable mechanism do you expect to supply about 10 tokens of routing foresight at C/k = 8: a lookahead predictor, a decoupled router, or draft-model routing? Can you bound what a predictor of a given precision and recall would buy, using your fetch oracle?

---

## 6. What would make this a 9/10 paper

1. **One thesis, four exhibits.** For example: "For offloaded batch-1 MoE decode, the gap to the host-bandwidth bound is dominated by admission decisions that need about W50 tokens of routing foresight; online policies, learned orders and speculative batching cannot supply it." Use the bound with its sensitivity, the factorial, the W50 trace figure and the engine table, and put everything else in a well-indexed artifact.
2. **Make the statistics match the claims.** Treat the machine as the unit, with at least 5 rentals per CPU class. Use a hierarchical bootstrap or mixed model, report variance components, and phrase cross-host claims as distributions. At about $1 per hour this is affordable.
3. **One bound definition everywhere.** Score your engine and the published rows with the identical procedure and ceilings. Make the realistic (measured-GPU or "all") bound the headline and the datasheet bound an upper envelope. Classify regimes under the realistic bound.
4. **Close the accounting.** Report two-player Shapley values with intervals on the measured states. Split the "rest" by direct measurement (Nsight timelines of the oracle states on the O-hosts) so the decomposition sums to the gap with small residuals. Only then is the title earned.
5. **Honest modelling.** Rename the law. Ground G physically. Report the naive baseline on every held-out set. Explain the Qwen3 bias.
6. **Close the loop with a realisable source of foresight.** Even a modest routing predictor or decoupled-router variant that achieves a measured fraction of W50 in the engine would turn the main finding from a diagnosis into a design result, and move significance from 3 to 4–5.
7. **Broaden.** Add non-math workloads with a clean development/test split, a third model family, and either a second engine or a unified-memory platform, to show the bound/W50 relationship is not specific to llama.cpp on AM5.
8. **Make pre-registration confirmatory.** Use machine-scored clauses, a small set of primary hypotheses per study, and an explicit statement of what the calibration record says about the author's predictive skill.

---

## 7. Factual errors, with a log of the claims I verified

### 7a. Claims verified against the repository (correct unless noted)

| # | Claim in paper | Where checked | Result |
|---|---|---|---|
| 1 | Table 2, host B: ours/FreeToken 1.275 [1.254, 1.295] at gpt-oss 25%, 1.294 [1.278, 1.312] at 11%, 1.154 [1.134, 1.172] at 40%; ours/llama.cpp 2.73× | Recomputed from `gpu-branch/results/081_headline_law@vast/bs1.jsonl` with a paired bootstrap (10,000 resamples) | Exact match |
| 2 | Speed limits 172 / 430 / 519 / 96 / 214 / 308 tok/s and ours at 41 / 25 / 29 / 42 / 30 / 35% | `prereg/speed_limit_v2.json`, `prereg/audit/audit_sol.json` reference points | Match |
| 3 | Scorecard: 565 clauses, 218 held, 193 held (point), 142 failed, 8 untested, 4 void; jobs 088 onward: 42 of 82 sign clauses held (75 on the point estimate), 90 of 301 bands failed | Recounted `prereg/scorecard_clauses.json`; `prereg/scorecard.json` by era and type | Match. 171 of the 180 numerically re-scorable statuses agree with the stated rule; the other 9 are deterministic counts scored "held" outright |
| 4 | Audit: 20 in-class rows, median 13.6% | `prereg/audit/audit_sol.json` + `audit.json` with the filter from `wsg_numbers2.py` | Median matches (13.63%, n = 20). The "quartiles" are order statistics; see 7b |
| 5 | Best online policy reads 35–110% more than MIN; decayed frequency best in 12 of 27 cells, S3-FIFO in 10 (within 1% of the best in 20 and 18) | `prereg/policy_study.json` | Match (1.349–2.100) |
| 6 | W50 ≈ 0.59 (C/k)^1.33, r = 0.971 over 26 points; model-bootstrap CI 1.09–1.47; LOMO 1.24–1.37; 1.51 without the 6 interpolated points | Refit from `prereg/foresight/w50_exact.json` | Match |
| 7 | Fetch oracle 1.36–1.51× on O4 and 1.16–1.26× on O5; gains 16–51% on three hosts; paced 25–81% | Table 11 values and `prereg/foresight_09{5,6a,6b}.json` | Match |
| 8 | Measured accounting: bytes 26–60%, overlap 7–23%, rest 30–62%; single read −4 to +4%; foresight given single read 26–57% | `prereg/accounting_measured.json` | Ranges match for the fixed order; the result is order-dependent (W4) |
| 9 | Blind law test: per-host medians and maxima of Table 8; G fitted on another host before the blind jobs | `prereg/perlayer_model.json`, `jobs/ec2/law_predict.py` header, ledger offer IDs | Match. The fitting host (offer 48822557, jobs 064/066) is not among the Table 8 hosts, so "blind" holds |
| 10 | Mean KL 0.0019 / 0.0005 nats, top-1 98.5 / 99.3%, ΔNLL +0.16 / +0.19% | `prereg/parity_090.json` (the job ran on an RTX PRO 6000 despite its "a100" name, as the paper says) | Match |
| 11 | 66 rentals, 39 distinct machines, $55.9 | `gpu/vast_ledger.json` | Match ($55.94) |
| 12 | FreeToken tuned: best ≤ 1.04× its Table 2 setting; ours 0.98–1.17× it; FreeToken reserves 28.4–30.1 GiB | Scorecard clauses 098 P1/P2/P6 | Match |
| 13 | Predictions committed before measurements | `git log` on the gpu branch for jobs 081, 093, 096, 097 (read-only) | Script commits precede results commits |

### 7b. Statements I found to be wrong or misstated

1. **Section 4, p. 5:** *"With a paced single-read prefetch three steps ahead the gain is 25–81% … bringing the engine to 46–75% of the limit there."*
   - The paced state alone reaches 45.8–73.1% at the host-bound cells on O4 and O5. The "75%" is the *unpaced* combined state (both2) at O4 Qwen3 12.5% (0.748 vs. 0.731 for both3p; `prereg/foresight_096a.json`).
   - `scripts/factorial_paper.py` line 80 takes the maximum over both2 and both3p, so the sentence attributes to the paced prefetch a number that belongs to another state.
   - Should read "46–73%", or "with the faster of the two prefetching states".
2. **Appendix A, Rule 3, p. 12:** *"ratios of total time agree with the reported ratios of mean rates within 0.003."*
   - True for the Ours ÷ FreeToken ratios (≤ 0.002). False for the Ours ÷ llama.cpp ratios in Table 2, recomputed from `results/081_headline_law@vast/bs1.jsonl` (arithmetic vs. harmonic):

     | Cell | Arithmetic | Harmonic | Difference |
     |---|---|---|---|
     | gpt-oss 40% | 3.196 | 3.187 | 0.008 |
     | gpt-oss 25% | 2.731 | 2.724 | 0.007 |
     | gpt-oss 11% | 2.001 | 1.997 | 0.004 |

   - The rule-3 row also presents the paper as following "harmonic mean for rates", while the tables report arithmetic means of per-problem rates.
3. **Table 7 caption, p. 13:** *"Prediction scorecard, jobs 073–097."* The table and its totals include job 098's clauses (pp. 25–26), and the Appendix B text says "jobs 073–098". The caption takes the last job in file order, not the highest job number.
4. **Section 3, p. 4:** *"reach a median 13.6% of it (quartiles 8.2–23.1%)."*
   - These are the 6th and 16th of 20 sorted values (`inclass[n//4]`, `inclass[3n//4]` in `scripts/wsg_numbers2.py`), not quartiles under any standard definition.
   - Linear-interpolated quartiles are 8.1% and 20.6%. The upper value is overstated by about 2.5 points. Minor.
5. **Section 2, p. 2:** *"Eq. (1) predicted our cache on gpt-oss within a median 3.7% on 7 RTX 5090 hosts (90th percentile 12%; Table 8)."*
   - The median is per configuration (29) and the 90th percentile is per measurement (33). Per measurement, the median is 3.4% (`lawRowsMedian`; recomputed from `prereg/perlayer_model.json`).
   - Mixing bases in one sentence is an inconsistency rather than a wrong number. Minor.
6. **Section 2, p. 3** (selective, not false): *"A one-path bytes-over-bandwidth model with the same G has the same median but a 90th percentile of 31%."* This is true on the blind set, but on the 30 later held-out measurements the same baseline beats the law on p90 and max (W5). As written, the sentence leaves the reader with the opposite impression of the held-out evidence.
7. **README (artifact):** *"every number in it is a macro generated by the scripts below."* The "19–22%" rental difference (Section 2), "within 2.5%" and "10,000" are literals in `paper.tex`.

I found no fabricated or irreproducible number. Every main-text number I checked traces to a script and a JSON file. The errors above are attribution, labelling and selection errors, not data errors.

---

## 8. Would I take this person as a PhD student?

**Yes, and I would compete for him, with a clear training plan.**

What this artifact shows is rare at any stage, let alone from an independent researcher without a lab:
- the instinct to bound before optimising;
- the discipline to commit predictions before measuring, and to publish 142 failures;
- the engineering to build a competitive llama.cpp expert cache and to instrument oracle policies inside it;
- the economy to run a 66-rental, multi-host campaign for $56;
- a repository in which a stranger can recompute the headline intervals bit-for-bit from raw rows.

Those are the hardest things to teach, and he already has them.

What he lacks is equally clear, and teachable:
- **Distillation.** He accumulates evidence rather than shaping it into a thesis, and his prose is nearly impenetrable.
- **Modelling hygiene.** A fitted intercept is not a "law". A bound built on mixed ceilings is not a "speed limit". An attribution that depends on order must be reported in both orders.
- **Statistical design at the right unit.** The experimental unit is the machine, not the problem.
- **Reporting.** Several of the most informative comparisons sit computed but unreported in his own repository. I read that as the side-effect of producing far more analysis than he could curate, not as concealment, but it is a habit to fix early.

In an interview I would ask him to:
- derive the bound and argue MIN-with-bypass optimality under set requests at the whiteboard;
- explain the factorial's design choices;
- rewrite the abstract in five sentences.

If he can do that, and I expect he can, he is exactly the kind of student a performance-modelling group should want.

---

## Verdict in brief

- **MLSys recommendation:** weak reject (5/10). Scores: novelty 3, soundness 3, rigor 4, clarity 2, significance 3.
- **Why.** The science is careful and unusually transparent: pre-registered, reproducible, honest about failures. The paper is held back by:
  - presentation that a program committee cannot verify;
  - uncertainty that excludes the dominant between-host variance;
  - a headline comparison with published systems made against a different bound (like-for-like, the engine is at 22–42%, still above the 13.6% median);
  - a measured accounting whose "overlap" share flips sign under the other attribution order, which the repository computes but the paper omits;
  - a "law" that loses to a naive baseline on held-out hosts, which is also computed and omitted.
- **Errors found.** Small: a 75% that belongs to the unpaced state, a 0.003 tolerance that is 0.008 for the llama.cpp ratios, a caption job range, nonstandard "quartiles", and mixed bases for the law's median and percentile. No data errors.
- **What a rewrite needs.** A focused rewrite with like-for-like bounds, host-level statistics and an order-robust accounting would be an accept.
- **PhD admission:** yes.
