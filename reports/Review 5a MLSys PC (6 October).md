# Review 5a: MLSys 2027 PC

**Submission:** "Where the Seconds Go: A Speed Limit and a Measured Accounting for Mixture-of-Experts Decode on One Consumer GPU"
**Reviewer:** PC member 5a. This review was written blind and independently, from the paper PDF (41 pages), the repository and the `gpu` results branch.
**Date:** 6 October 2026

---

## 1. Summary of the paper

The paper studies batch-1 decode of MoE models whose experts don't fit on a consumer GPU (gpt-oss-120b and Qwen3-30B-A3B on RTX 5090s; an RTX 4090 and an RTX 3090 in the appendix). It makes four contributions.

- **A per-machine "speed limit" in seconds (Eq. 2).** It is built from three inputs:
  - the fewest expert reads any exact-routing cache with C slots per layer can make, which is Belady's MIN with bypass on the model's own routing trace;
  - the machine's highest probed host-memory rate;
  - the GPU's datasheet bandwidth.

  The authors' 3,500-line llama.cpp expert cache reaches 25–43% of this limit. A set of 52 published measurements, scored with datasheet ceilings, reaches a median of 13.6% for the 20 rows in the bound's class.
- **A measured accounting of the gap (Section 4, App. E/F).** Oracles that read the recorded future routing run inside the engine, on three rented hosts, in a randomised factorial (jobs 093–097). Spending foresight "as MIN does" (admit only what MIN admits, read each admitted expert once) is worth 16–51%, or 25–81% with a paced prefetch. Spending it as Belady prefetch or serve-then-copy reads 1.6–2.5× the optimum's bytes and can lose.
- **A trace study over nine MoE models (Section 5).** It finds three things:
  - online policies sit 35–110% above the optimum;
  - batched speculative verification does not substitute for foresight;
  - half of the online-to-optimal gap closes with W50 ≈ 0.59 (C/k)^1.33 tokens of perfect lookahead.

  A learned admission order fitted on other text removes 17–30% of the excess reads offline, but runs only 0.90–1.05× the deployed cache in the engine.
- **A pre-registration log (App. B).** Every prediction for jobs 073–098 was committed to git before the machine started, and the log scores 565 clauses: 218 held, 193 held on the point estimate only, and 142 failed.

## 2. Scores

| Criterion | Score (1–5) |
|---|---|
| Novelty | 3 |
| Technical soundness | 3 |
| Experimental rigor | 4 |
| Clarity | 2 |
| Significance | 3 |

**Overall: 5 / 10. Recommendation: weak reject.** My confidence is 4/5: I know expert offloading and caching theory well, and I checked the artifact directly.

The measurement work is unusually careful and the artifact is excellent. My recommendation is driven by three things:

- the presentation is very hard to read for an MLSys audience;
- the system contribution over the strongest baseline is modest and depends on the host;
- the headline "accounting" is less separable than the paper implies.

A major rewrite with a sharper story could move this to weak accept.

## 3. Strengths (ranked)

1. **The artifact is verifiable to an unusual degree.**
   - Every job's predictions sit in its script header, and git history orders them before the results. For example, job 096's script was committed at 16:52 on 5 October and its results at 21:39; job 097's script at 21:41 and its results at 00:02 the next day.
   - Raw per-problem rows are kept, and the statistics scripts regenerate the tables.
   - I recomputed several headline numbers from raw rows and JSON (Section 7b) and they reproduce to the third decimal.
   - The paper reports its prediction failures (142 failed clauses, 30% of bands in jobs 088+) rather than hiding them. That is rare and commendable.
2. **Oracles inside a real engine, not only in a simulator.**
   - The randomised, cold-cache factorial in job 096 separates three questions: what to admit (MIN's set vs Belady's set), how many reads (one vs serve-then-copy), and overlap (paced lead vs in-step fetch).
   - It runs on two hosts, and a second launch replicates the result within 0.024.
   - It measures what foresight is worth in seconds rather than in hit rate. This is the paper's most valuable methodological contribution.
   - The clean finding that admission policy matters more than lookahead per se is useful. Belady prefetch admits what MIN bypasses and loses at the low budgets (0.94–1.15× the online policy vs 1.16–1.44× for MIN-fetch).
3. **Useful negative results, stated with numbers.**
   - A learned admission order does not pay in the engine on a strong host: 0.90–0.94× on O6.
   - Batched speculative verification gives the equivalent of a two-to-four-token window, and turns into a loss once rejected drafts' experts are counted.
   - Next-layer prediction never lowers host bytes.
   - These are actionable for the many 2025–26 systems that claim prediction as their lever.
4. **The W50 horizon law is a concrete, testable target for forecasters.** It is fitted over 26 points from 9 models, with a model-bootstrap interval on the exponent of 1.09–1.47 and leave-one-model-out exponents of 1.24–1.37.
5. **A careful competitor comparison at equal GPU expert memory.** It includes per-budget backend choice for FreeToken, a tuned FreeToken (job 098: at most 1.04× its default), CPU-kernel parity checks (llama.cpp 1.13× / 0.99× FreeToken's CPU backend), and output parity by KL divergence.
6. **The audit of 52 published measurements against a common bound.** Despite the caveats in W6, this is the kind of field-level sanity check MLSys should encourage.

## 4. Weaknesses (ranked)

### W1. Clarity: the paper is very hard to read. Severity: high. Cost to fix: moderate (a rewrite, no new experiments).

- **Where:** throughout, worst in the Abstract, the "Thesis" paragraph, Section 4 and Appendices E and F.
- **What is wrong:**
  - Almost every sentence carries two to six numeric ranges and a private term ("cell", "host-bound", "serve-then-copy", "foa", "nb2", "both3p", "hitoptp", "paced").
  - Examples:
    - the abstract sentence beginning "On three hosts, foresight spent as the optimum spends it …";
    - Section 4's paragraph "Spent the usual ways, it pays less…", which compares five oracle states across three hosts and two budget classes in prose.
  - Figure 1, the main-text figure, carries the main result, but a row label is clipped ("elady, two reads, paced").
  - Table 10 is set at a font size that is effectively unreadable.
  - The main text leans on Tables 10–13 in the appendix for the per-state numbers.
  - The scorecard (Table 7) takes about 17 pages and uses internal state names that are defined only in a job-script header.
- **Why it matters:** An MLSys reader cannot extract the three or four take-aways without re-deriving them. The contributions are real but buried, and the density also hides the qualifiers each range needs (see Section 7, items 1–3).
- **Fix:**
  - Lead with three findings, each backed by one figure: limit vs measured; a cleaner, order-free version of the measured accounting; W50.
  - Use plain names for oracle states.
  - Move scorecard rows to the artifact and keep a one-table summary.
  - State every range with its scope (host-bound cells or all cells; which hosts).

### W2. The measured accounting depends on the order of decomposition, and the paper reports only one order. Severity: high. Cost to fix: low (the data are already in the repo).

- **Where:** Section 4, "The accounting, measured"; Intro contribution 2 ("the bytes foresight saves are 26–60% of the gap, the overlap it allows 7–23%, and the rest… 30–62%"); Table 11.
- **What I found:** `prereg/accounting_measured.json` stores both feasible orders.
  - **Bytes first** (online → fetch → paced prefetch) gives the published shares.
  - **Overlap first** (online → Belady prefetch → paced single-read prefetch) gives very different shares at the 12 host-bound cells:
    - overlap: −161% to +32%;
    - bytes: 34% to 199%.
  - Examples: O5 Qwen3 12.5% gives overlap −1.61 and bytes 1.99; O4 Qwen3 12.5% gives −0.68 and 1.35.
  - The "rest" is the same in both orders, but the split between "bytes" and "overlap" is not identified. The two levers interact strongly: overlap without MIN's admission set hurts.
  - The paper applies Shapley values to the *model* (Appendix E), but not to the measured states, and never mentions the alternative order.
- **Why it matters:** "Separating the bytes it saves, the overlap it allows" is the paper's central claim. As measured, only "bytes conditional on choosing MIN's admission set" is identified.
- **Fix:**
  - Report both orders, or a measured Shapley over the 2×2 of {MIN set, Belady set} × {in-step, paced lead}, which job 096 nearly contains.
  - Reword the contribution as a conditional decomposition.

### W3. The system contribution over the strongest baseline is modest and depends on the host. Severity: medium-high. Cost to fix: high (needs a new mechanism or more hosts).

- **Where:** Table 2, Table 9, Fig. 3, App. G "FreeToken tuned".
- **What the numbers show:**
  - Against FreeToken, the lead on Qwen3 is 1.03–1.05 on most hosts and 0.97–0.98 at Qwen3 43.75% on two hosts.
  - On gpt-oss it is 1.09–1.29 on strong-CPU hosts. It grows to 1.5× only on slow-link hosts, where FreeToken also had a thread-configuration problem.
  - Against tuned FreeToken (job 098) the lead is 0.98–1.17×.
  - The 2–4× over stock llama.cpp is largely what any per-expert cache gets (Fig. 6: the static cache alone is 1.31× and 1.89×).
- **The "3–15% on Qwen3" range rests on one row:**
  - The upper end comes from the Qwen3 25% row on host B (1.153). That is the one row without a confirmation launch, and both systems' variants were selected on it.
  - Every other measurement of that cell is 1.04–1.05: host S 1.048, the 4090 1.045, job 098 1.044.
- **Why it matters:** As a systems paper, the deployable artifact gains a few percent over the state of the art on the model class that dominates releases (many small experts). The large numbers (16–81%) are oracle upper bounds that no realisable policy reaches: the learned order gives 0.90–1.05×.
- **Fix:**
  - Drop or confirm the unconfirmed row.
  - Make the paper's identity explicit: it is a measurement and characterisation paper.
  - Ideally, add a realisable mechanism that captures some of the oracle gain (see Section 6).

### W4. The reported uncertainty understates the variance that matters. Severity: medium. Cost to fix: moderate.

- **Where:** Section 2, Protocol ("two rentals of one CPU model differed by 19–22% in speed"); Section 7; every interval in Tables 2 and 9–13.
- **What is wrong:**
  - The 95% intervals cover problem sampling within one launch: 30 problems × 256 tokens.
  - They are narrow, about ±1.5%, while the between-host spread of the same ratio is 5–10× wider. For example, ours/FreeToken at gpt-oss 11% is 1.29 on host B, 1.21 on host S, 1.37 on the 4090 and 1.58 on a slow-link host.
  - The oracle factorial ran one launch per host. Only O4 got a second launch (job 097).
  - Table 2's host B is atypical: its card's memory clock is 1.21× the common one.
- **Why it matters:** Readers will take "[1.278, 1.312]" as the uncertainty of the comparison. Statements such as "leads at 11 of 12 cells" are host-sampling statements that the intervals do not cover.
- **Fix:**
  - Report a host-level summary in the main text: the range or a random-effects interval over the eight hosts of Fig. 3.
  - Label within-launch intervals explicitly as such in every caption.

### W5. The "speed limit" is loose where it matters most, and its ingredients are known. Severity: medium. Cost to fix: low for presentation, moderate for theory.

- **Where:** Section 3, Eq. (2), Table 3.
- **Looseness:**
  - The GPU term uses datasheet bandwidth, yet all-in-VRAM batch-1 runs reach only 52% (gpt-oss) and 61% (Qwen3) of it. With the measured rate, the gpt-oss 25% and 40% limits both collapse to 279 tok/s.
  - So at GPU-bound cells, and at gpt-oss 25%, the "limit" mostly measures llama.cpp's batch-1 kernel efficiency, not anything a cache can change.
  - B_host is the maximum of six noisy probe samples on host B (76.8–87.5 GB/s).
- **Prior art:** The bound is a relaxation (no latency, perfect overlap), so it is valid. But its pieces are established:
  - Belady/MIN with bypass as the cache-level oracle, for which the paper cites Zhang [56], who also decomposes the online-to-optimal gap;
  - bytes over bandwidth ([53, 54]);
  - roofline.

  The novelty is in instantiating these per machine, not in the bound itself.
- **A disclosure gap on optimality:** The simulator's header says MIN-with-bypass optimality under per-step set requests is "verified against exhaustive search in tests". The paper should state this and the instance sizes, since R\* is the bound's foundation.
- **Fix:** Make Table 3's measured-GPU or "all" column the headline limit, or report both side by side in Table 2.

### W6. The audit's headline juxtaposition compares mismatched denominators. Severity: medium. Cost to fix: low.

- **Where:** Abstract ("ours 25–43% … published systems in its class a median of 13.6% of theirs"); Section 3; Fig. 7; Table 15.
- **What is wrong:**
  - The authors' systems are scored against a probed host rate. Published rows are scored against datasheet ceilings, and 21 rows use capacities imputed as upper bounds. Both choices push published percentages down, as the paper acknowledges.
  - The abstract's 13.6% is the median of a 20-row subset: trace rows minus 7 speculative and 2 lossy rows.
  - Fig. 7's "median, trace" line and Table 15's caption show 9.5% (n=29), so the abstract's number appears nowhere in the figure that illustrates it.
- **Fix:**
  - Score the authors' own cache with the same datasheet-ceiling procedure and put that number beside the 13.6%.
  - Show the in-class median in Fig. 7.

### W7. Scope is narrow. Severity: medium. Cost to fix: high.

- **Workload:** batch-1 decode only (prompts and batch > 1 bypass the cache and run at 77–83% of stock), the first 256 tokens, AIME-25 prompts that were also used during development, and two main models.
- **Oracle hosts:** all oracle hosts have weak host memory (43–51 GB/s at best). The headline host B (88 GB/s) never ran an oracle; only O6 (69 GB/s) ran fetch and paced prefetch, in job 097.
- **Mitigation:** The O6 data support the paper (fetch/base 1.20–1.30, paced prefetch/base 1.37–1.61 in `foresight_097b.json`) and should be cited in Section 4.
- **What is still missing:** larger models with many small experts on bigger hosts (for example DeepSeek-V3-class or Qwen3-235B), multi-request serving, and longer contexts where attention grows.

### W8. The prediction record is honestly reported but is weaker than the framing suggests. Severity: low-medium. Cost to fix: low.

- **Where:** Abstract/Intro ("every prediction committed … Appendix B scores all 565 clauses"); App. B.
- **The record:**
  - Strict holds are 218/565 (39%), and 142 (25%) failed.
  - In jobs 088+, 90 of 301 bands failed.
  - The law (Eq. 1), presented as "predicted blind", missed:
    - its own bands on the fetch oracle at 4 of 6 cells (095 P2);
    - the bypass oracle at every cell (094 P4);
    - its host-bound speed bands on the second host by 14–17% (089 P3).
  - The study was iterative: each job's predictions were written after the previous job's results. That makes this a disciplined lab notebook rather than confirmatory pre-registration.
- **Fix:** Frame it that way, and report the hit rate of predictions that were genuinely out of sample (new host or new mechanism) separately from bookkeeping clauses.

### W9. The thesis is only partly supported by the paper's own numbers. Severity: low-medium. Cost to fix: low.

- **The thesis:** "where host memory binds, the time of offloaded decode is its host bytes per token over the host's read rate."
- **What the paper's own numbers show:**
  - At the host-bound cells, 30–62% of the gap is "rest", which no oracle touches and which the paper did not separate.
  - Eq. (1) over-predicts prefetching states by up to 38%, fails on 12-channel EPYC and hybrid-core hosts, and over-predicts Qwen3 by 5–21% with frozen constants.
- **Fix:** Soften the thesis, or attribute the rest. The Nsight breakdown in Section 4 is from an older configuration (job 079, different host) and only covers the GPU-bound cell.

## 5. Questions for the authors

1. **Accounting order.** Why is the measured accounting reported only in the bytes-first order? `accounting_measured.json` also has the overlap-first order, which gives overlap shares of −161% to +32%. Can you give an order-free measured decomposition?
2. **Unconfirmed row.** Table 2's Qwen3 25% host-B row (1.153) is the only unconfirmed row, and it sits far above every other measurement of that cell (1.04–1.05). Will you confirm it or remove it from the "3–15%" range?
3. **Optimality of R\*.** Is MIN with bypass provably optimal under per-step set requests, where all k experts of a layer are requested at once and served experts may be evicted? What instance sizes did the exhaustive-search test cover?
4. **Variance across rentals.** Rentals of one CPU model differ by 19–22%. Can you provide random-effects (host-level) intervals for the headline ratios, or at least the across-host range for each cell in the main text?
5. **Imperfect forecasters.** How does W50 change for an imperfect forecaster, for example one with 90% top-k recall at W tokens or with misses spread over time? Without this, it is hard to say whether SeqMoE/Read-ME-style predictors could approach the oracle's 16–51%.
6. **Speculative verification.** The analysis counts only host reads. Verification also amortises dense and attention reads across K positions. Does the conclusion survive when the GPU term of Eq. (2) is included?
7. **Oracles on a strong host.** Why were no oracles run on host B or another host with ≥ 70 GB/s host memory? Do the O6 results from job 097 (fetch 1.20–1.30×, paced 1.37–1.61×) belong in Section 4's ranges?
8. **Two reads vs one.** The deployed policy reads each admission twice, yet reading it once changes speed by only 0.97–1.03× at host-bound cells. Is that because background copies overlap later steps? If so, does the same overlap explain part of the "rest"?
9. **Audit denominators.** Can you score your own cache with the audit's datasheet-ceiling procedure, so the abstract's 25–43% vs 13.6% comparison uses the same yardstick?
10. **Task accuracy.** Output parity is by KL divergence (mean 0.0019 / 0.0005 nats). Was task accuracy on AIME or MATH ever compared, given the CPU helpers quantise activations to Q8_0?

## 6. What would make this a 9/10 paper

1. **Close part of the gap with a realisable mechanism.** For example:
   - a forecaster that reaches W50 tokens of lookahead (a router decoupled in the Read-ME style, or a lightweight draft model's routing), driving MIN-style admission with single reads in the engine;
   - or showing that the paced single-read path with a realistic imperfect predictor recovers a stated fraction (say ≥ 1/3) of the oracle's 16–51%.

   That would turn an upper-bound study into a systems result.
2. **Rewrite around three figures and three sentences:**
   - (i) limit vs measured, across hosts;
   - (ii) an order-free measured accounting;
   - (iii) W50.

   Use plain state names and move the per-clause scorecard to the artifact.
3. **Make the decomposition identified:** a full measured factorial with interaction terms, or a measured Shapley over the oracle states, plus an attribution of the 30–62% "rest" (prefetch imprecision, partial overlap, kernel efficiency).
4. **Broaden the evidence:**
   - at least one larger many-expert model on a host with more DRAM;
   - longer decodes (2k+ tokens are partly done in App. G);
   - a second GPU generation in the main text;
   - host-level intervals from multiple rentals per CPU class.
5. **Tighten and justify the bound:**
   - report the measured-GPU-rate limit as a first-class column;
   - state or cite the MIN-with-bypass optimality argument;
   - for the audit, re-measure a few published systems on the authors' own hardware with probed ceilings.

## 7. Factual errors and verification

### 7a. Statements that are wrong or wrong as stated

1. **"Not predicted" was in fact predicted, and the range needs its scope.**
   - **Appendix F:** "Not predicted: reading the online policy's admissions once is worth 0.97–1.03×."
   - It was predicted. Job 096's header (`jobs/096_factorial@vast.sh`, prediction 3) says: "foa … runs 0.97-1.12x base at the host-bound cells". The paper's own scorecard holds eight such clauses (096O4/O5-P3b), all scored *held*.
   - The range is also wrong without the host-bound qualifier: on O5's GPU-bound cells, foa/base is **0.947** at gpt-oss 40% (8.73 vs 9.22 ms in Table 11) and **0.966** at Qwen3 43.75% (12.49 vs 12.92 ms), per `prereg/foresight_096b.json`.
2. **The learned-order ranges hold only at host-bound cells.**
   - **Abstract:** "a learned admission order fitted on other text reads 6–13% fewer experts but runs 0.90–1.05× the deployed cache". Similar statements appear in the contributions bullet and the Conclusion: "in the engine such an order is worth 0.90–1.05×".
   - Both ranges hold only at the four host-bound cells. Over all six cells:
     - reads fall by 6–18% (`prereg/learned_outcome_097.md`);
     - speed is 0.896–1.075× the deployed cache. At gpt-oss 40% on O4, learned is 126.36 vs base 117.53 tok/s, which is **1.075×** (`prereg/foresight_097a.json`), and Qwen3 43.75% on O4 is 1.045×.
   - The abstract states the range without its scope.
3. **"Three quarters of the published systems" is true only of a subset.**
   - **Section 3:** "even so, three quarters of the published systems stand below a quarter of a latency-free ceiling".
   - Of the 52 audited rows, **32 (62%)** are below 25% (`prereg/audit/audit_sol.json`).
   - "Three quarters" holds only for the 20-row in-class subset (16/20) or the 29 trace rows (25/29). The sentence should name the subset.
4. **The Table 7 caption has the wrong job range.**
   - **Caption:** "Prediction scorecard, jobs 073–097".
   - The table contains 44 rows from job 098 (`paper/tab_scorecard.tex`, from line 457), and the Appendix B text says "jobs 073–098".
5. **The 75% upper bound comes from a different state than the sentence names (minor).**
   - **Section 4:** "With a paced single-read prefetch three steps ahead the gain is 25–81% … bringing the engine to 46–75% of the limit there".
   - The paced three-step state (both3p) peaks at **73.1%** of the limit (O4, Qwen3 12.5%).
   - The 75% (74.8%) is the *unpaced two-step* state (both2). `scripts/factorial_paper.py` takes the maximum over both2 and both3p, so the sentence attributes the figure to the wrong state.
6. **Unclear antecedent.**
   - **Section 2:** "…ties FreeToken on its own headline model within 3%, and ties it on Mixtral-8x7B".
   - FreeToken was not run on Mixtral. The tie (1.015 [1.008, 1.024], Table 9) is against llama.cpp.

### 7b. Claims checked and found correct

I checked 13 sets of claims. All reproduced except the qualifiers noted in 7a.

| # | Claim | How I checked it | Result |
|---|---|---|---|
| 1 | **Scorecard totals.** 565 clauses: 218 held, 193 held (point), 142 failed, 8 untested, 4 void. Per-era counts 20/23/43/479. In jobs 088+: 42 of 82 sign clauses held (75 on the point estimate), 90 of 301 bands failed, 23 threshold failures, 3 equality failures, 7 sign failures. | Tallied `prereg/scorecard_clauses.json` | All reproduce |
| 2 | **Table 2, host B, gpt-oss 25%.** Ours 109.2, FreeToken 85.6, ratio 1.275 [1.254, 1.295]. | Recomputed from raw `results/081_headline_law@vast/bs1.jsonl`, launch 2, paired by problem, 10,000 resamples, seed 0 | 1.2747 [1.2545, 1.2948] |
| 3 | **The other host B rows of Table 2.** | Compared with `prereg/headline_081.json` and `split_080.json` (same Vast listing, offer 51046112, per `gpu/vast_ledger.json`) | Match |
| 4 | **Ratios of mean rates agree with ratios of total time "within 0.003"** (App. A, rule 3). | Computed from job 081 rows | Agree within 0.002 |
| 5 | **Table 3, all 42 entries.** Also: "8–9% above the all-in-VRAM speed" (279.4/255.3 and 192.1/177.8), pooled +5–10%, median probe −8%. | `prereg/speed_limit_v2.json` | All match |
| 6 | **Audit.** In-class median 13.6% (n=20, quartiles 8.2–23.1); trace median 9.5% (n=29); 21 imputed capacities. | `prereg/audit/audit_sol.json` and `audit.json` | Reproduces |
| 7 | **W50.** Prefactor 0.589, exponent 1.328, r=0.971, n=26. Bootstrap 1.09–1.47; leave-one-model-out 1.24–1.37; no-interpolation 1.51. Savings 26–56%, median 41%. Fit values 9 and 23 tokens at C/k = 8 and 16. | `prereg/foresight/w50_exact.json` | All reproduce |
| 8 | **Table 4.** Medians match. Best online policy 1.349–2.100× the optimum ("35–110%"). Decayed frequency best in 12 cells and S3-FIFO in 10 (within 1% of the best in 20 and 18). Drift 22–58% / 33–75% (median 51%); skew 31–56%; κ cost 1–8% (Qwen3 up to 22%). | `prereg/policy_study.json` | All reproduce |
| 9 | **Foresight in the engine.** MIN-fetch is 1.36–1.51× (O4) and 1.16–1.26× (O5), 16–51% over all cells. Paced is 1.48–1.81× (O4) and 1.25–1.50× (O5), 25–81% overall. Online is at 31–49% of the limit. Fetch differs between hosts by 0.13–0.26. Belady's second read costs 1.01–1.03× (gpt-oss) and 1.41–1.54× (Qwen3 12.5%). | Table 11 and `foresight_095/096a/096b.json` | All reproduce |
| 10 | **Measured shares (bytes-first order).** Bytes 26–60%, overlap 7–23%, rest 30–62%; single read −4 to +4%; foresight given single read 26–57%. | `prereg/accounting_measured.json` | Reproduce; W2 covers the order dependence |
| 11 | **Learned order in the engine.** O4: 1.01–1.03× foa and 1.03–1.05× base. O6: 0.905–0.942× and 0.896–0.936×. Table 5's "closed" column (17/17/23/30/24/17%) and "1.43–1.69× MIN". | `foresight_097a/b.json`; Table 5 recomputed from its own entries | Reproduce at host-bound cells |
| 12 | **Law (Eq. 1).** Median error 3.7% over 29 configurations (3.4% over the 33 measurements); 90th percentile 12% (computed over the 33 measurements, a mixed basis). | `prereg/homepc/law_crosshost.json` | Reproduces |
| 13 | **Remaining checks:** cost; job 098; the Nsight breakdown; pre-registration order. | See the four items below | All hold |

Details for row 13:

- **Cost.** 66 rentals, 39 distinct offers, $55.94 (`gpu/vast_ledger.json`).
- **Job 098.** Ours is 0.98–1.17× FreeToken's best and ahead at 5 of 6 cells. FreeToken uses 28.4–30.1 GiB vs ours 9.5–27.3 GiB (`prereg/freetoken_tuned_098.json`).
- **Nsight breakdown** (4.93/5.20 ms copies, and so on). It matches `prereg/qwen3_profile_outcome_079.md`. Note that it comes from job 079 on a different 9950X3D host and an earlier engine configuration, where ours/FreeToken was 0.991.
- **Pre-registration order.** Job scripts 096, 097 and 098 were committed before their results (read-only `git log` on the gpu branch).

---

## Verdict

This is a meticulous, honest measurement study with a first-rate artifact. Every headline number I checked reproduces, and the few errors I found are scoping slips (Section 7a), not fabrication.

As an MLSys paper it currently falls short on three counts:

- **Clarity:** it is dense to the point of being impenetrable.
- **The accounting:** its central "bytes vs overlap" split depends on decomposition order, and the paper doesn't disclose this.
- **The system:** it gains only a few percent over the strongest baseline. The large gains are oracle upper bounds that no realisable policy achieves.

**Weak reject (5/10).** A rewrite that leads with three crisp findings, reports an order-free accounting, and labels every range's scope would make it a solid weak accept. Adding a realisable forecaster that recovers a meaningful share of the oracle gain would make it a strong paper.
