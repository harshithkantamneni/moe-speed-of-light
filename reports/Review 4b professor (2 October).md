# Review: "Where the Seconds Go: A Speed Limit and a Measured Accounting for Mixture-of-Experts Decode on One Consumer GPU" (H. Kantamneni)

Reviewer: professor, performance modelling, data movement and scientific benchmarking of HPC/ML systems. This is a PhD fellowship assessment and an MLSys 2027-style review. 2 October 2026.

---

## 1. Summary

The paper studies batch-1 decode of Mixture-of-Experts models on a consumer GPU when most experts live in host DRAM. The setups are rented RTX 5090, 4090 and 3090 hosts running gpt-oss-120b (MXFP4) and Qwen3-30B-A3B (BF16). It makes five contributions:

- **The instrument.** A ~3,700-line llama.cpp patch with per-layer expert slots, decayed-frequency admission and GPU-signalled CPU helpers. A per-machine split (the "FETCH" table) decides how many of a layer's misses are copied over PCIe and how many run on the CPU. It is chosen by a calibrated host-memory model, Eq. (1): T = G + max(Xc/Bc, Xp/Bp, (Xc+Xp)/Bcp). At equal expert memory the cache runs 2.0–4.0× stock llama.cpp and 0.97–1.29× FreeToken.
- **A "speed limit".** Eq. (3) takes Belady's MIN with bypass per layer on the model's own routing trace and puts it in a two-resource min–max with the datasheet GPU rate and the best probed host rate. The cache reaches 25–43% of this limit; 52 published measurements reach a median 14.5%.
- **A Shapley attribution.** The gap to the limit is split over five modelled "fixes".
- **In-engine oracles** (jobs 093–095) that replay recorded future routing. From these the author concludes:
  - Foresight "spent as the optimum spends it" (MIN-with-bypass admissions, each read once) is worth 25–38% serialised and 42–67% overlapped.
  - Belady prefetch, or serve-then-copy admission, reads 1.7–2.5× the optimum's bytes and gains nothing at the lowest budgets.
- **A trace study** over nine models. Online policies read 35–110% more than the optimum. W50 ≈ 0.59 (C/k)^1.33 tokens of perfect foresight close half of that gap, and batched speculative verification does not supply it.

Every number is a macro generated from `prereg/*.json`, and 299 pre-registered clauses are scored in an appendix.

What I checked: I read all 30 pages and re-derived the main tables from the raw rows and JSON. I read the job scripts, the oracle patch and the simulators, and wrote and ran an independent re-simulation of the single-read oracle. Section 7 and weakness W8 report what that turned up.

## 2. Scores

| Criterion | Score |
|---|---|
| Novelty | 3 / 5 |
| Technical soundness | 3 / 5 |
| Experimental rigor | 3 / 5 |
| Clarity / writing | 2 / 5 |
| Significance | 3 / 5 |
| Reproducibility | 4 / 5 |
| **Overall** | **4 / 10: weak reject** |

The measurement hygiene and openness are well above the MLSys norm. The paper's headline inferences are not supported at the standard the paper sets for itself:

- that foresight is the largest missing piece;
- that it is recovered "only" when spent as the optimum spends it;
- that most of its value is bytes.

They rest on four things:

- a model that the paper's own oracles show is mis-priced by 1.3–5.6×;
- one oracle launch on one host, in a fixed configuration order;
- an incomplete factorial with a weak same-host comparator;
- prose that most MLSys reviewers will not get through.

A focused rewrite plus about $40 of replicated, randomised oracle runs would make it a 6–7.

**Supervisor verdict.** Yes, I would take this student, and I would support the fellowship.

What convinced me:

- Working alone, for $46.31 of rented hardware (57 rentals on 35 machines; I summed the ledger), they built a llama.cpp expert cache that beats a 2026 system from a strong group at most cells.
- They wrote down a bound, pre-registered 299 predictions and published the 103 that failed.
- Most of all, they built oracles into their own engine to test their own model, then reported that the model mis-prices its terms by factors of 1.3 to 5.6. These are the instincts I spend years trying to teach.

The deficits are as clear:

- They cannot yet write for a reader. The paper is a lab notebook compressed into nine pages.
- The narrative runs ahead of the evidence in the abstract and conclusion, while the appendix is careful.
- They accept a plausible post-hoc explanation without checking it against their own code; the "warm cache" explanation (W8) is contradicted by their own benchmark.
- They run many small sequential jobs instead of a few designed, randomised, replicated experiments.
- They do not engage the classical prefetching-and-caching theory that their oracle results re-derive.

Every one of these is what a PhD trains. My condition: the first year produces one paper with one thesis, written with an editor.

## 3. Strengths

**S1. The question and the bound** (§1, §4, pp. 1 and 4–5; Table 3).
- Asking for a ceiling in seconds for a stated class is the right question, and the paper states the class: exact routing, at most C slots per layer, whole experts, one token per pass.
- It lists the assumptions: CPU-run experts are read from DRAM, no latencies, empty start.
- It reports four tightenings: pooled slots, median B_host, measured GPU rate, per-layer sum.
- R* is the exact per-layer MIN with bypass, not a hit-rate heuristic, and is checked against exhaustive search on small instances (`tests/test_cachesim.py`).
- This is rare in the MoE-offloading literature.

**S2. The model is confronted with measurement** (§5 pp. 6–7; Table 9; Appendix E).
- The author built oracles of future routing into a real engine: single-read fetch, scheduled single-read prefetch, a paced variant, a hit-optimal Belady prefetch, and serve-then-copy MIN with bypass.
- The paper then states in the main text where its own model is wrong: bytes over-priced 1.3–1.5×, overlap 1.6–3.5×, the rest under-priced 3.3–5.6×.
- This is the loop Hoefler and Belli ask for, and almost no paper in this area closes it.

**S3. Openness** (Appendix B pp. 13–21; Appendix H; the repository).
- Numbers are macros generated from JSON; raw per-problem rows, job scripts with committed predictions, probes and the cost ledger are all public, and failures are scored and listed.
- I reproduced from the raw rows or JSON, and all of them match the paper:
  - Table 1's ratios and percent-of-limit;
  - the blind-test medians (3.7%, 3.3%, 90th percentile 7.5%) and the frozen-constant re-scoring (7.0%, 11.9%, 16%);
  - job 095's ratios, reads and hit rates;
  - the measured and model shares of §5;
  - the W50 fit (`w50_exact.json`);
  - the rental cost.

**S4. Large, clean, same-host effects** (§5; Table 8).
- On O3, with each admitted expert read once, MIN-with-bypass foresight gains 1.25–1.38× at every cell, and 1.39–1.67× with prefetch.
- The hit-optimal prefetch on the same host loses 9% and 31% at the two lowest budgets.
- At those budgets the single-read prefetch that obeys MIN's admission rule (lead2) gains 1.26× and 1.22× where the bypass-less prefetch loses. That is a direct, paired demonstration that bypass matters when the host is saturated.

**S5. Engineering** (§2; Table 1; Fig. 6; Appendix F).
- A working cache at 2.0–4.0× stock llama.cpp.
- Output parity by teacher-forced KL (0.0019 and 0.0005 nats), equal to stock llama.cpp's own CPU placement.
- A component ablation, and a probe-driven CPU/PCIe split that pays 21–34% on slow links.
- A stock-clock replication (host S) and a 4090/3090 grid. The author went looking for the result to fail.

**S6. The trace study** (§6; Table 4).
- Nine models, six online policies, static caches and Belady windows.
- "Four tokens of foresight beat every online policy in every cell" and the measured drift of the hot set (22–58% per 100 tokens) are useful, well-quantified facts.

**S7. Good-faith competitor handling** (Appendix F).
- FreeToken as shipped runs its own headline model faster than its paper reports (93.8 vs 77–83 tok/s).
- Backends are picked on a separate launch, and launch-order checks are reported.

**S8. Candour** (§8).
- The limitations section says, among other things, that two of the five Shapley terms are definitions rather than measurements.

## 4. Weaknesses (ranked by cost to the grade)

### W1. The headline interpretation of the oracle experiments is over-generalised and not isolated by the design.

**Where:** Abstract; §1 ¶3; §5 "Foresight, measured" and "The accounting, measured" (pp. 6–7); §9; Appendix E; Table 8.

**What is wrong:**

- **(a) Weak comparator.** Job 095's same-host stand-in for "the usual way" is the *unpaced* hit-optimal prefetch. On O1 the unpaced variant was slower than the paced one at three of the four host-bound cells: 1.18 vs 1.40, 0.72 vs 0.92 and 1.14 vs 1.25 × the online policy (`prereg/foresight_outcome_093.md`), i.e. by 9–21%. The 0.69× quoted for Qwen3 12.5% is the weaker variant. The paced variant was never run on O3.
- **(b) Half of the claim is inferred across hosts.** Serve-then-copy MIN with bypass (job 094) was not re-run on O3.
  - So the "reads an admitted expert twice" half of the claim is an inference from O2 to O3: different CPU, and 50.6 vs 61.5 GB/s of host memory.
  - It is also confounded with the 2–4-step copy lag that put job 094's misses at 1.29–1.47× R*.
  - No single host has both read-count variants with the admission rule held fixed.
- **(c) The two mechanisms are not separated.** At gpt-oss 11% on O3 the hit-optimal oracle has 2.7 misses and 65.0 admissions per token (`foresight_095.json`). Its 1.77× R* is therefore mostly admitting what MIN would bypass, not double reads. The two mechanisms the abstract lists together are not separated anywhere on one host.
- **(d) Order dependence.** §5 argues that a fixed-order decomposition makes the first term look largest, then decomposes the measurement in one order: bytes, then overlap given bytes, then the rest. From that it concludes that "most of what it is worth where host memory binds is the bytes it saves".
  - The other feasible order is overlap first: Belady prefetch at roughly the online policy's bytes (31–32 vs 34 reads per token at gpt-oss 25%).
  - In that order the overlap step alone is worth +40% (O1) and +63% (O2) at gpt-oss 25%, more than the fetch oracle's bytes-only +32% (O3).
  - So at the 25% cells the claim depends on the order. At 11% and 12.5% it holds.
- **(e) Post-hoc selection.** "42–67% overlapped" and the "overlap" step of the measured decomposition both take, per cell, the faster of two prefetch variants after the fact.
- **(f) The conclusion's "only" is contradicted.** The Conclusion says "in the engine that piece is recovered only when foresight is spent as the optimum spends it". Table 10 contradicts this: the hit-optimal oracle recovers 76% and 52% of v(F) at the two host-bound 25% cells on O1, comparable to the fetch oracle's 66% and 70% on O3.
- **(g) The binding resource is the link.** At the two lowest budgets the hit-optimal oracle's PCIe copies run at about 85% (gpt-oss: 65.7 copies/token × 13.25 MB × 50.5 tok/s ≈ 44 GB/s) and about 80% (Qwen3) of the probed 51.8 GB/s copy-engine rate. Eq. (1)'s own link term alone, 65.7 × 13.25 MB / 43.2 GB/s = 20.2 ms, exceeds the measured 19.8 ms token. The resource that binds is the link, not "host memory".

**What is supported:** at the two lowest budgets on one host, single-read MIN-with-bypass foresight gains about 30% where bypass-less Belady prefetch loses, and obeying bypass is what matters.

**Why it matters:**
- This is the paper's scientific headline. As written it reads as a general law of how to spend foresight.
- The evidence is one launch on one host, and the strong form holds at two of six cells.
- The 095 design was conceived after jobs 093 and 094 failed, so it is exploratory with respect to the single-read hypothesis even though its clauses were registered. A confirmatory replication is the step pre-registration exists for.

**Fixable by 30 October:** Yes.
- Run one job per host at all six cells with these configurations, in randomised order per cell:
  - online;
  - MIN-bypass fetch;
  - MIN-bypass serve-then-copy;
  - MIN-bypass single-read prefetch: lead 2 unpaced, and lead 3 paced;
  - Belady-without-bypass prefetch: paced, and unpaced;
  - Belady-without-bypass single-read.
- Two launches on each of three hosts: a stock-clock 9950X3D, an O3-class 9950X, and a slow-link host.
- Cost: job 095 took 2.4 h at $0.50/h for six configurations. Nine configurations is about 3.5 h, about $2 per launch, under $40 in total.
- One engine flag (single-read Belady without bypass) is missing: one to two days.
- Then rewrite the abstract and conclusion to the scope the data support.

### W2. Fig. 1's "accounting" is a what-if under a model the paper itself shows is mis-priced, yet it remains the organising claim.

**Where:** Abstract ("A Shapley accounting names routing foresight the largest missing piece where host memory binds"); §5 p. 6 and Fig. 1; §6's first sentence; §9.

**What is wrong:**

- **The endpoints are fitted.** v(none) equals the measured time by construction, because the GPU factor η (1.25–1.68 on host B) is fitted per cell as the residual. v(all) is the limit. Two of the five players are definitions.
- **The intermediate states do not exist.** The intermediate coalitions are not states any engine can reach: overlap without foresight, or foresight read through the limit's single-rate paths.
- **The players are not independent.** The oracles show that overlap requires cross-token foresight in this engine, so treating O and F as nearly independent players is wrong in kind, not just in degree.
- **The prices are measured to be wrong.** On O3 the model mis-prices bytes by 1.3–1.5×, overlap by 1.6–3.5× and the rest by 3.3–5.6× (§5's own figures).
- **The ranking is a near tie in the running example** (`prereg/shapley_gap.json`, variant `exact_max`):
  - foresight's Shapley value beats overlap's by 0.14 ms of a 6.84 ms gap;
  - foresight is the largest marginal in only 70 of 120 orders, and ranks as low as third;
  - so the ranking rests on 2% of the gap, far inside the model's own measured pricing errors (bytes 1.3–1.5×, overlap 1.6–3.5×, the rest 3.3–5.6×).
- **The measurement does not confirm "largest".** In the running example's measured decomposition the biggest bucket is "the rest" (43%), not the bytes (34%). §6's "the engine confirms it" holds only if bytes and overlap are counted together.

**Why it matters:** the abstract's lead finding and the paper's organising narrative rest on this figure.

**Fixable by 30 October:** Partly.
- Demote Fig. 1 to "what Eqs. (1) and (3) imply".
- Lead with the measured decomposition, taken on the same host class as Fig. 1: one 095-type job on a stock-clock 9950X3D, $2–5.
- Report both feasible orders, and Shapley values over the coalitions that can actually be run.
- A model that prices latency and overlap and is validated out of sample against the 18 measured oracle states needs 2–3 weeks; probably not by the deadline.

### W3. The writing.

**Where:** throughout; worst in the abstract, §2, §5 and Appendix E.

**What is wrong:**
- Sentences routinely carry four to eight numbers and two or three semicolon-joined clauses.
- Core terms are introduced in passing: cell, FETCH, host-bound, foresight, overlap, v(F), v(O,F), fetch/lead/both/paced/hit-optimal, and hosts A, B, S, O1–O3.
- "Speed limit", "limit", "ceiling", "bound" and "speed of light" are used interchangeably.
- The abstract cannot be read without having read the paper: "Belady's optimum with bypass on the model's own routing, at probed and datasheet rates, gives a speed limit in seconds".
- The main oracle results live in Table 8 (about 23 columns in a tiny font) and in Appendix E.
- Seven of the eighteen appendix pages are a clause table.
- An instrument, a bound, a calibrated model, an accounting, oracles, an audit, a trace study and a speculation study make two papers, not one.

**Why it matters:** MLSys reviewers will not reconstruct the argument. The careful hedges are in the appendix and the overclaims are in the abstract.

**Fixable by 30 October:** Yes, in 1.5–2 weeks at $0, and it is the cheapest grade improvement available.
- One thesis.
- A glossary table.
- One running-example figure on one host: limit → model → measured states.
- The scorecard and the grid moved to supplementary material.

### W4. Statistical scope and design.

**Where:** §2 "Equal memory, statistics and protocol"; §5; Appendix E; Tables 1 and 8–11.

**What is wrong:**

- **Single launch, fixed order, no randomisation.** Each oracle job is one launch on one rental, with six to ten configurations run in a fixed order in one process:
  - job 095: base, fetch, lead2, both2, both3p, hitopt;
  - job 093: base, paced, W = 2, 4, 16, 64, all, unpaced, no-overlap, all-CPU.
  There is no randomisation, interleaving or reversed-order repeat.
- **The intervals are narrow because they are narrow in scope.** They cover 30 sequences only (about ±0.01 on ratios), while the paper itself reports 19–22% differences between two rentals of one CPU model.
- **Cross-host comparisons mix many factors:**
  - host memory from 50.5 to 87.5 GB/s;
  - GPU memory clock 14,001 vs 17,001 MHz;
  - SM-clock ceilings from 3,105 to 3,420 MHz;
  - power limits from 400 to 600 W;
  - O1's low device read (1,554 vs about 1,692 GB/s);
  - different drivers.
  The comparisons involved are O1/O2 vs O3, and host B vs host S.
- **Serial dependence.** Within a configuration the cache carries over between sequences, so per-sequence observations are serially dependent, which the paired bootstrap ignores.
- **Table 1 pairs different texts.** In the server protocol, "paired by problem" pairs different token sequences: in job 073 the cache's free-running greedy outputs matched llama.cpp's on 0–3% of problems.
- **In fairness**, within one machine the noise is small:
  - host S's two launches differ by at most 1.2%;
  - in job 095 every configuration shows the same cold-start pattern over its first five sequences;
  so gross order effects are not visible.

**Why it matters:** the oracle results are single-host case studies stated as general findings. By the paper's own account, the regimes are "a property of the host".

**Fixable by 30 October:** Yes. Use W1's design (randomised order, two launches × three hosts) with a hierarchical bootstrap (sequences within launches within hosts). Under $40 combined with W1.

### W5. Eq. (1): what "predicted blind" means.

**Where:** the contribution bullet; §3 "Tested as a predictor" (pp. 3–4); Table 2; Appendix C.

**What is wrong:**

- **The counts are not predicted.**
  - In the blind test, Xc and Xp are fixed per configuration from earlier runs' engine counters (`jobs/ec2/law_predict.py`, `READS`: "identical on every host").
  - In the later re-scoring and in the oracle analyses they are each run's own counters.
  - In the oracle analyses G is also re-calibrated per cell on the online run.
- **G is a fitted intercept, not a GPU term.** G = 4.82 ms (refit 5.17 ms) exceeds the whole all-in-VRAM token (3.92 ms) and is 34–73% of the cache's token on host B. As §3 concedes, it absorbs the helpers' shortfall against the probe.
- **Most blind configurations test one roof.** 19 of the 29 blind configurations copy nothing, so they test T = G + Xc/Bc. Recomputed:
  - 16 desktop configurations that copy nothing: median error 2.4%;
  - 8 that copy: median 4.2%, maximum 11.9%.
- **The later-host errors are signed by model, not noise** (`law_frozen_later.json`):
  - gpt-oss: +0.5% to +16%, growing with budget;
  - Qwen3: −4% to −10%.
- **The law misses every oracle state:** −17% on the serialised fetch (no per-step fetch latency) up to +36% on the prefetching states (no overlap term).
- **The forms cannot be told apart.** Medians of 3.1% (max), 4.5% (sum) and 3.2% (per-layer max) at n = 33 do not distinguish them.
- **No naive baseline is reported.**

**Why it matters:**
- "Predicted blind on 7 machines" reads as a first-principles prediction of speed.
- It is a one-intercept bandwidth model fed known counts. That is useful: it picks the FETCH table, worth 3.3–8.0% on host B and 21–34% on slow links.
- It is not a validated model of the states the accounting reasons about.

**Fixable by 30 October:**
- Rewording, a naive baseline, and the copy / no-copy split: days, $0.
- A model with latency and overlap terms, validated out of sample: 2–3 weeks.

### W6. Eq. (3) is neither a strict bound nor a tight one, and the headline uses the loose version.

**Where:** §4; Table 1 ("Ours, % of limit"); Abstract.

**What is wrong:**

- **The host rate can be exceeded.** B_host is the highest probed rate, so a system can beat the "limit": another rental of the same CPU probed 90.3 GB/s against host B's 87.5.
- **The GPU rate cannot be reached.**
  - B_gpu is the datasheet rate, of which batch-1 kernels reach 52–61%.
  - So at the GPU-bound cells the limit is about twice the all-in-VRAM speed: 519 vs 255 tok/s for gpt-oss at 40%.
  - Because of that unattainable rate, the minimisation over c moves resident experts onto the CPU.
- **Latencies are dropped, and the paper's own data show they matter:**
  - the fetch oracle runs 8–17% slower than the bandwidth law;
  - the EPYC shows a 55–70 µs per-layer floor;
  - audit rows with every expert resident sit at 4% of the bound.
- **Optimality is not proved.** That event-atomic MIN with bypass is "the fewest reads of any policy" for sets of k requests per step is checked by exhaustive search only on tiny instances (E ≤ 6, k ≤ 2, T ≤ 8).
- **The class is narrow.** It excludes pooled caches, speculative and batched decoders, and expert skipping or substitution, which is much of what practitioners deploy.

**Why it matters:**
- "25–43% of the limit" mixes host-bound cells, where the figure is meaningful, with GPU-bound cells, where an unattainable GPU rate dominates it.
- With all tightenings applied it is 33–57%.

**Fixable by 30 October:** Yes, in days at $0.
- Headline the tightened limit: measured GPU rate plus a measured per-layer latency floor.
- Keep the datasheet version as an envelope.
- Prove, or cite a proof, that MIN with bypass is optimal for set requests.

### W7. The audit.

**Where:** §4 "Published systems against their own limit"; Fig. 7; Table 13; Abstract.

**What is wrong:**

- **The headline median mixes incomparable classes.** The 52 rows fall into four classes whose medians differ tenfold:

  | Class | Median % of bound |
  |---|---|
  | no GPU slots | 49% |
  | uniform routing | 35% |
  | trace | 9.5% |
  | all experts fit | 4.0% |

  The abstract's "median 14.5%" summarises nothing.
- **The bound ignores latency.** It is latency-free at datasheet rates; the all-fit class at 4% shows it sits about 25× above what batch-1 systems achieve for small models with nothing offloaded.
- **The caption and the medians disagree.** Table 13's caption says rows below 5% "are not adjudicated", yet all 11 such rows (8 of the 29 trace rows) enter every quoted median. Without them the trace median is 12.8%, not 9.5%.
- **The inputs are the author's, not the audited systems':**
  - per-layer capacity, precision and dense placement are inferred;
  - traces are the author's own sampled text, sometimes of a sibling variant;
  - pooled, layer-pinned and speculative systems are scored against a bound the paper itself calls "not theirs".
- **The sentence goes beyond the measure.** "The trace-scored measurements sit at a median tenth … of what their machines could deliver" is not what the bound measures.

**Why it matters:** it publicly characterises 13 groups' systems as about 10% efficient, using a method the paper's own caveats undercut.

**Fixable by 30 October:**
- Trimming and rewording: yes, days.
- A real audit — reproducing 3–5 open systems on a host-B-class machine against its probed, tightened limit: 2–3 weeks and $50–150.

### W8. A simulator semantics bug, with an incorrect explanation in its place.

**Where:**
- Appendix E: "where its cache is warm from the previous configuration's run and the simulation starts cold";
- Appendix B;
- Table 11: "simulated misses/token", the "fit" column and the "below ideal" entry;
- `scripts/foresight_single_read_sim.py` and `scripts/foresight_latency_sim.py`.

**What is wrong:**

- **The stated explanation cannot be true.**
  - `llama-ec-bench` creates a fresh `llama_context` for each configuration and frees it afterwards.
  - The expert cache is a member of `llama_context` and starts with `resident.assign(L.C, -1)`, so it cannot be warm from the previous configuration.
  - Consistent with that, every configuration's first five sequences show the same cold-start penalty.
- **The actual cause is in the simulators.**
  - They keep each expert's next use incrementally, restricted to the sequence of its last use.
  - Residents carried into a new sequence therefore look "never used again" until they are requested, and are evicted first.
  - The engine's `oracle_plan_fetch` recomputes next uses within the current sequence.
- **My re-simulation with the engine's rule reproduces the engine within 1–3% at all six cells:**

  | Reads per token | gpt-oss 11% | gpt-oss 25% | gpt-oss 40% | Qwen3 12.5% | Qwen3 25% | Qwen3 43.75% |
  |---|---|---|---|---|---|---|
  | Engine-semantics re-simulation | 39.3 | 16.0 | 7.5 | 100.4 | 44.9 | 14.5 |
  | Engine, measured | 39.9 | 16.5 | 7.7 | 100.8 | 45.2 | 14.7 |
  | Repository simulator | 40.0 | 17.6 | 9.3 | 101.3 | 47.3 | 17.7 |

- **Consequences:**
  - Two of job 095's 12 "failures" (P1c, P1f) are failures of the simulator, not of the engine.
  - Table 11's simulated misses and its latency fits, including the "below ideal" entry, come from the same biased simulator.
  - So the copy-latency account of job 094 is quantitatively unreliable at C ≥ 32.

**Why it matters:** numerically it is minor. But it is a plausible post-hoc explanation that entered the paper without a check against the author's own code — the habit that pre-registration is meant to prevent.

**Fixable by 30 October:** Yes: one to two days at $0. Add a regression test comparing simulator and engine counters on a recorded trace.

### W9. The speculation and W50 claims overreach.

**Where:** Abstract ("which speculative batching does not supply"); §6 "Speculation is not foresight" and "How far ahead"; §5's horizon sentence.

**What is wrong:**

- **Speculation is measured in reads, not time.**
  - The batch-K study never measures speculative decoding's main benefit for offloaded MoE: amortising dense, attention and resident-expert work (G ≈ 4.8 ms per token here) over the tokens a pass commits.
  - The routing of rejected drafts is assumed (free / routed like the true token / random), not measured. At K = 8, α = 0.8, E/8 the median bracket runs from +27% to −53% of reads.
  - The main text's figures (8–12%, and a loss of 18–20%) are the middle ("same") variant, unlabelled.
  - Acceptance is treated as i.i.d.
- **W50 is a perfect-foresight number, not a forecaster's.**
  - The fit uses 26 points nested in 9 models, with one text source per model.
  - The exponent is 1.33 [1.09, 1.47], or 1.51 [1.15, 1.74] without the six interpolated sub-token points.
  - At the one C/k = 16 point the fit gives 23 against a measured 34.
  - It is the horizon of *perfect* foresight that closes half of the read gap, not "the horizon a forecaster needs", which depends on forecast accuracy.
- **Selective confirmation.** §5 cites the engine's W-sweep agreeing at gpt-oss 25% (9.2 vs about 10) but not that it disagrees at Qwen3 25% (6.1 vs 3.9, Appendix E).

**Fixable by 30 October:**
- Rewording: yes.
- Real drafter traces (EAGLE or a small draft model) and time measured through the engine: 2–3 weeks and $50–150; after the deadline.

### W10. Baselines, fairness and workload.

**Where:** §2 and Table 1; Appendix D and Table 7; Appendix F; §8.

**What is wrong:**

- **Per-host tuning is asymmetric.**
  - The cache's FETCH table is recomputed per host from a probe.
  - FreeToken's backend is carried over from host B to host S, the 4090 and the 3090. On the 3090 it then ran below llama.cpp, yet 2.12–4.46× ratios are still printed, with a dagger.
  - The cache's split is swept; FreeToken's thread count and fetch cap are left at defaults.
- **Table 1 is on the development set.** The AIME-25 prompts were used during development (§8). Held-out long-output runs mitigate this for gpt-oss, but Table 1 itself is not held out.
- **The closest baselines are barely used.** The most relevant expert-cache baselines (two llama.cpp forks or PRs, and pipelined sharding) appear only in job 077: one model, one host.
- **The lead is explained two ways for the same cell.** §3 says the running example's lead over FreeToken is "largely the CPU path". §2 and Appendix F attribute 77–84% of it to the admission policy.

**Fixable by 30 October:** Yes: about $10–20 and 2–3 days.
- Re-pick FreeToken's backend per host and give it a sweep equal to the cache's.
- Add held-out prompts for Table 1.
- Drop the daggered 3090 ratios.

### W11. Related work misses the theory the oracle results re-derive.

**Where:** §7.

**What is wrong:**
- "Foresight spent as MIN spends it" versus "Belady prefetch" is the conservative-versus-aggressive trade-off of integrated prefetching and caching. The relevant work includes:
  - Cao, Felten, Karlin and Li (SIGMETRICS 1995), which is in `refs.bib` but not cited;
  - Kimbrel and Karlin on prefetching across parallel disks;
  - Albers, Garg and Leonardi on minimising stall time.
- Also missing:
  - optimal replacement with cache bypass in CPU caches;
  - CPU–GPU hot/cold splitting for dense models (PowerInfer);
  - windowed flash offloading (LLM in a flash).

**Why it matters:** this would position the novelty honestly and supply the natural family of oracle schedules for W1's design.

**Fixable by 30 October:** Yes, in days.

## 5. Questions for the author

1. **Comparator.** Why is the *unpaced* hit-optimal oracle the same-host comparator in job 095, when job 093's paced variant was faster at three of the four host-bound cells? What does the paced variant give on O3?
2. **Serve-then-copy on the same host.** Will you run job 094's serve-then-copy MIN-with-bypass oracle on the job 095 host, in the same randomised launch, so that "reads an admitted expert twice" is measured rather than inferred across hosts?
3. **Cold start.** Do you agree that every `llama-ec-bench` configuration starts with a cold cache (fresh `llama_context`, `resident = -1`)? What replaces the "warm from the previous configuration" explanation? My re-simulation, with next uses recomputed within the current sequence, matches your engine to within 1–3%.
4. **Overlap-first order.** What are the measured shares in the overlap-first order (online → Belady prefetch at roughly online bytes → both/paced → limit)? Is "most of what foresight is worth is the bytes" still true at the 25% cells?
5. **Same host for model and measurement.** Why is the measured decomposition on O3 while Fig. 1 is on host B? What does the measured decomposition look like on a host-B-class, stock-clock machine?
6. **G for Qwen3.** How was G = 4.3 ms obtained for the cache on Qwen3, and why is it the llama.cpp value, when gpt-oss uses 4.82 ms (cache) and 5.06 ms (llama.cpp)? Do the opposite-signed residuals by model on the later hosts come from G?
7. **Naive baseline.** On the 33 blind measurements, what error does a naive baseline give: the calibration host's speed scaled by Bc alone, or G + Xc/Bc with no other term?
8. **Link rate.** Which mechanism do background admissions use, the copy engine or zero-copy? Why does Eq. (1) take Bp = 43.2 GB/s (zero-copy, 16 MB, 64 blocks) when the same probe shows 51.8 GB/s by the copy engine? At gpt-oss 11%, Eq. (1)'s link term alone exceeds the hit-optimal oracle's measured token time.
9. **Variance across rentals.** How much do the *ratios* you report (ours/FreeToken, oracle/online) vary between rentals, as opposed to absolute speeds? Can you report hierarchical intervals?
10. **Different texts in Table 1.** In Table 1 each system decodes a different text (only 0–3% of outputs were identical to llama.cpp's in job 073). How much per-problem variance does that divergence add? Would teacher-forced timing of all three systems on one text change any ratio?
11. **Audit inputs.** For how many of the 52 audit rows did you verify per-layer capacity, precision and dense placement against the authors' code or configuration? Why are the rows you call "not adjudicated" inside the medians?
12. **Speculation in time.** With real drafts, what does batch-K verification do to time per committed token in your engine — not to reads?
13. **W50 robustness.** Does W50 move with the text domain (code, chat)? What are the G and D trace arms, whose exponents are 1.38 and 1.43, and why are they not reported?
14. **Online single-read admission.** Have you tried the obvious online consequence of your result: decayed-frequency admission that admits by FETCH (one read) instead of serve-then-copy, with no foresight? Admissions are 4.4 of the online policy's 34.2 reads per token at gpt-oss 25%.
15. **Proof of optimality.** Is there a proof that event-atomic MIN with bypass minimises host reads for sets of k requests per step? Your batch-K note says the step-level version is "not proven optimal for set requests with bypass".
16. **FreeToken per host.** Why were FreeToken's backends not re-picked on host S, the 4090 and the 3090?
17. **Tuning on the benchmark.** Which hyperparameters were tuned on the AIME-25 prompts: half-life, κ, helper count, FETCH tables? What does Table 1 look like on held-out prompts at all six cells?
18. **Source of the lead.** Is the running example's lead over FreeToken the admission policy (Appendix F) or the CPU path (§3)? Which run isolates the CPU path's contribution?

## 6. What would make this a 9

The items are in priority order. Items 1–5 and 9–11 are feasible by 30 October and would make this a 6–7. A 9 also needs items 6–8.

| # | Item | Effort | Cost | By 30 Oct? |
|---|---|---|---|---|
| 1 | One thesis, rewritten for a reader | 1.5–2 weeks | $0 | Yes |
| 2 | Confirmatory, randomised, replicated oracle factorial | ~1 week of engineering + runs | ≤ $40 | Yes |
| 3 | Fix the simulators; regression test | 1–2 days | $0 | Yes |
| 4 | Online single-read admission (no foresight) | 2–3 days | ~$5 | Yes |
| 5 | Lead with the measurement; demote the model accounting | 2–3 days after item 2 | $0 | Yes |
| 6 | A time model validated out of sample | 2–3 weeks | $0–20 | Borderline |
| 7 | Make foresight real with a forecaster | 4–6 weeks | $100–300 | No |
| 8 | Generality: third model, second domain, workstation host | 2–3 weeks | $100–250 | No |
| 9 | Tightened bound as the headline | days | $0 | Yes |
| 10 | Audit: cut, or redo by reproduction | days to cut; 2–3 weeks to redo | $0 / $50–150 | Cut: yes |
| 11 | Fairness hardening and related work | 2–3 days | $10–20 | Yes |

1. **One thesis, rewritten for a reader.** A candidate thesis: "where host memory binds, the time of offloaded MoE decode is set by the bytes per token; MIN with bypass sets the minimum; online policies read 1.4–2.5× it; perfect foresight spent conservatively recovers X, and a realistic forecaster recovers Y."
   - Add a glossary table.
   - Add one figure on one host: limit → model → measured states.
   - Move the scorecard, the grid and the audit table to supplementary material.
   - Shorten the speculation study to one paragraph.
2. **A confirmatory, randomised, replicated oracle factorial** (W1, W4).
   - Admission rule: MIN with bypass / Belady without bypass / online.
   - Reads: single (fetch or scheduled prefetch) / serve-then-copy.
   - Timing: serialised / lead 2 unpaced / lead 3 paced.
   - Six cells, randomised order, two launches × three hosts, including a stock-clock 9950X3D (the class Fig. 1 is computed on), and hierarchical intervals.
   - Report both feasible decomposition orders, and Shapley values over the coalitions that can be run.
   - About one week of engineering and runs, under $40. This alone moves the paper's core claim from case study to finding.
3. **Fix the simulators** (W8). Correct the next-use semantics, re-derive Table 11 and the job 095 predictions, and add a simulator–engine regression test.
4. **Online single-read admission** (Question 14). It tests the double-read claim without foresight and may be a practical win.
5. **Lead with the measurement and demote the model accounting** (W2). Use the item-2 data.
6. **A time model validated out of sample** (W2, W5).
   - A per-layer critical-path model with fetch latency, copy lead and overlap, and helper efficiency instead of a catch-all G.
   - Fit on online runs only; predict the 18 measured oracle states plus the new factorial.
   - Only then attribute — or attribute over measured coalitions and use the model for what-ifs.
7. **Make foresight real.** This is what turns an upper bound into a result.
   - Feed predicted routing through the existing lookahead interface. The forecaster could predict expert sets W tokens ahead from the residual stream, or use a draft model's routing.
   - Report precision and recall against W, and end-to-end speed.
   - Do the same for speculative decoding with real drafts, measuring time per committed token, not reads.
8. **Generality.**
   - A third model family, e.g. DeepSeek-V2-Lite, or Qwen3-235B-A22B in Q4 on a host with 192 GB or more.
   - A non-math workload.
   - A 4–8-channel workstation host, to show where the regime boundary moves.
9. **Headline the tightened bound** (W6), with a measured latency floor, and a proof sketch for set requests with bypass.
10. **The audit** (W7): either cut it or redo it by reproducing 3–5 open systems on one machine against its probed, tightened limit.
11. **Fairness hardening and related work** (W10, W11). FreeToken re-picked and swept per host, held-out prompts for Table 1, the 3090 FreeToken ratios dropped, and the integrated prefetching and caching literature cited.

## 7. Factual and internal-consistency errors

### E1. The "warm cache" explanation is contradicted by the benchmark code

- Appendix E (pp. 23–24): "(the engine reads fewer at the GPU-bound cells, where its cache is warm from the previous configuration's run and the simulation starts cold)".
- Appendix B (p. 14): "(the engine's cache was warm from the previous configuration)".
- Contradicted by `llama-ec-bench`: a fresh `llama_context` per configuration, and a cache constructed with `resident.assign(L.C, -1)`. The cause is the simulators' stale next use for residents carried across sequences. A re-simulation with the engine's rule matches the engine within 1–3% (W8).

### E2. Fig. 5's caption contradicts the figure

- Fig. 5 caption (p. 25): "the oracle, reading every admitted expert twice, stays at 44–45% of the limit at every window."
- The 44–45% is the W = all value only (`foresight_093.json`). The plotted points at the four cells are:
  - W = 2: 37, 25, 44 and 35% of the limit;
  - W = 4: 42, 31, 45 and 38%;
  - W = 16: gpt-oss 25% is still at 42%.
- Only admitted *misses* are read twice. At W = all the oracle has 20.1 admissions and 31 total reads per token at gpt-oss 25% (Table 10), which is impossible if every admitted expert were read twice.

### E3. The Conclusion's "only" is contradicted by Table 10

- §9 (p. 9): "in the engine that piece is recovered only when foresight is spent as the optimum spends it".
- Table 10: the hit-optimal oracle recovers 76% and 52% of v(F) at the two host-bound 25% cells. Appendix E: it gains 41–63% at the four cells of 25% and above on O2.

### E4. "The engine confirms it" does not match §5's own measured shares

- §6 (p. 7): "The accounting names foresight across tokens as the largest cost where the host binds, and the engine confirms it."
- §5 (p. 7) gives the running example's measured shares as bytes 34%, overlap 23%, rest 43%.
- The model's Shapley margin of foresight over overlap there is 0.14 ms of a 6.84 ms gap, and foresight is the largest marginal in only 70 of 120 orders.

### E5. The same lead is attributed to two different causes

- §3 (p. 4): "the lead over FreeToken at that cell, 1.27×, is largely the CPU path that FreeToken's bandwidth model declines to use there".
- §2 (p. 3): "The lead is mostly admission rationing".
- Appendix F (pp. 26–27): "the admission policy is most of the lead over FreeToken (77–84% of it at gpt-oss 25 and 40% …)".

### E6. "Held with their intervals" overstates

- §1 (p. 1): "54% of 287 scored clauses held with their intervals".
- Table 6's caption (p. 14) defines "Held (point)" as a clause whose interval "includes the threshold, or no interval exists".
- 52 of the 156 "held" clauses have no interval: 32 bands, 13 thresholds and 7 equalities (`scorecard_clauses.json`). They include 22 counter-versus-simulation bands in job 095 and 12 KL/agreement/NLL clauses in jobs 082 and 090.
- By the printed rule the strict share is 104/287 = 36%; exempting the 7 deterministic equalities, it is 39%.

### E7. "As the law predicts" claims a prediction the law cannot make

- Appendix D (p. 22): "to 1.51–1.58× behind a link of half the bandwidth, as the law predicts".
- Appendix C (p. 21): "FreeToken's own runs are not predicted, because its counters are not exposed".
- The pre-registered 1.10–1.50× band for exactly this ratio failed at all three budgets (Table 6, job 088, P3d–f).

### E8. Host differences are mis-described and under-reported

- §2 (p. 2): "a Ryzen 9 9950X rental at the common clock with a 22% slower CPU". The 22% is the CPU's DRAM read rate (56 vs 72 GB/s; p. 3 says "whose CPU reads 22% slower").
- §2: host B's card "read device memory 3% above the datasheet rate". Compared with the other gated RTX 5090s:

  | | Host B | Other RTX 5090 hosts |
  |---|---|---|
  | Device read (GB/s) | 1,847 | 1,691–1,694 (9% lower) |
  | SM-clock ceiling (MHz, `gpu.csv`) | 3,420 | 3,105 |
  | Power limit (W) | 575 | 400 on host S |

  None of these is reported.
- §3 nevertheless attributes the host B vs host S difference to the CPU alone: "because with the CPU slower …".

### E9. Two different simulators give two different savings

- §6 (p. 8): "The optimum reads 26–56% fewer experts from host memory than our deployed decayed-frequency policy (median 41%)."
- Table 4 (p. 8): the deployed variant ("with hysteresis κ") reads 1.36–2.55× the optimum, i.e. 26–61% fewer.
- The sentence comes from the foresight study's simulator, the table from the policy study's, and the paper does not say so.

### E10. Rows the caption excludes are counted in the medians

- Table 13 caption (p. 29): "Rows below 5% … are not adjudicated".
- All 11 such rows are inside "trace 9.5% (n=29) … all rows 14.5% (52)", and inside the abstract's 14.5%.
- Without them the trace median is 12.8%.

### E11. The union range does not describe the condition it explains

- §6 (p. 8): "because the union of K positions' expert sets is 20–79% of Kk, several times C (Appendix B lists the variants)".
- The range is pooled over every α, every rejected-draft variant and every budget at K = 8 (`wsg_numbers2.batchk`), not the "same"-routing condition the sentence explains.
- Appendix B names the variants and points to a repository file; it gives no numbers.

### E12. Appendix C's EPYC errors are not the blind errors of Table 2

- Appendix C (p. 21): "(iii) On a 12-channel EPYC 9655 (559 GB/s) host memory stops being the limit (+16 to +29%)".
- Table 2 (p. 4) gives this host's blind cache errors as median 18.7%, maximum 31.9%.
- Appendix C quotes leave-one-host-out refit errors (macros `lohoMaxEpyc*`) without saying so.

### E13. The checklist's answer to Rule 4 is not accurate as written

- Appendix A, Rule 4 (p. 13): "Ratios are computed from mean speeds, never averaged across budgets or models".
- The paper summarises ratios across systems, models and budgets in several places:
  - the audit's medians of percent-of-bound (§4, Fig. 7);
  - Table 4's medians of read ratios over nine models;
  - §6's "median 15–32% of its reads across models".
- These are medians rather than means, but the checklist answer is not accurate as written.

### E14. "The first" and "the second" are ambiguous

- §5 (p. 7): "Our first two oracle jobs … found the first worth 25–63% and the second 13–23%".
- "First" and "second" refer to the two ways of spending foresight, not to the jobs. The 63% is the Belady prefetch in the *second* job (094, O2). As written it reads as the first job.
