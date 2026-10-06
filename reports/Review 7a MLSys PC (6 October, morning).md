# Review 7a: MLSys 2027 PC review (6 October, morning)

**Submission:** *Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth*
**Materials read:** paper/paper.pdf (30 pages: 9 pages of main text, references, Appendices A–K), paper/supplement.pdf (the 28-page prediction scorecard), the repository (scripts/, prereg/*.json, mosl/, tests/) and the GPU results (/home/claude/gpu-branch/jobs, results).
**Independence:** I did not open anything under reports/ or any prereg/*outcome*.md file. One caveat: while checking commit order I saw in the git log a commit message that names an earlier review round's scores. I formed my scores from the paper and the artifact and did not use that message.

**Desk-level issues for the PC chairs (not scored):**
- **Not anonymised.** The title page names the author and "Independent Researcher". MLSys reviewing is double-blind.
- **Wrong template.** The source uses `\documentclass[10pt,twocolumn]{article}` with 0.75-inch margins, not the MLSys style file. The main text (pp. 1–9) may not fit the page limit once it is reset in the official style.

I reviewed the paper on its merits.

---

## 1. Summary

The paper studies batch-1 decode of MoE models whose experts mostly live in host DRAM. The setting is a desktop GPU (RTX 5090, plus a 4090 and a 3090 in the appendix) with C GPU expert slots per layer. Models are gpt-oss-120b (MXFP4) and Qwen3-30B-A3B (BF16), prompted with AIME-25.

The paper makes three claims.

1. **A time bound.** Belady's MIN with bypass, run on the model's own routing trace, gives the fewest host reads R* that any exact-routing policy with C slots per layer can make. Eq. (1) combines R* with the machine's highest probed host-read rate and a GPU term. The authors' llama.cpp expert cache, which beats FreeToken in 11 of 12 configurations, reaches 25–43% of this bound (31–56% if the GPU term uses the measured batch-1 rate). Scored the same way, 20 in-class published measurements reach a median of 13.6%.
2. **Foresight in the engine.** Oracle policies are built into the engine and run on 16 rented hosts, with the host as the unit of replication. Future knowledge pays only when two choices are combined:
   - cache what MIN would cache;
   - read each admitted expert once, via a PCIe copy in the step, rather than CPU-serve-then-copy.

   A 2×2 factorial shows a large interaction between the two (Table 3). Whether the in-step copy pays depends on the host's link-to-CPU read-rate ratio. A two-path bandwidth model, T = G + max(X_c/B_c, X_p/B_p, (X_c+X_p)/B_cp), with G frozen from other hosts, predicts this sign on all 39 host-budgets.
3. **A price on foresight.** Degraded oracles (W-token windows, recall r) are measured in the engine on a 10-host panel and in a trace study of nine models. The horizon that recovers half of MIN's read savings is where the window holds about 0.65·C distinct experts per layer. This one-constant rule predicts a held-out model's horizon about as well as a two-constant power law in C/k. None of the online policies, the learned admission order, speculative batching or the forecasters tested reaches it.

Every GPU job's predictions were committed before the job ran. The supplement scores 1,054 clauses, including 208 failures. The rental cost was about $70.

## 2. Strengths

- **S1. A useful ceiling for an active area.** Many recent offloading papers report speedups over self-chosen baselines. A machine-specific ceiling for exact-routing systems, computed from the model's own routing and the host's probed rates, is the right instrument. Applying it the same way to the authors' system, FreeToken, llama.cpp and 52 published rows is the kind of normalisation the area lacks.
- **S2. Oracles inside a real engine, with careful statistics.** Oracle studies are usually trace simulations. Here the oracles run in llama.cpp with:
  - configurations in a randomised order with a cold cache;
  - relaunch reproducibility (within 0.024–0.030);
  - a crossed bootstrap over hosts and problems;
  - an explicit finding that machines vary more than problems (within-host half-width 0.009 against a between-host SD of 0.154).

  Treating the host as the sample is correct and rare.
- **S3. The factorial and its interaction term are the most informative result.** Choosing what to cache alone closes about 4% of the gap. Choosing how to load alone closes about 4%. Both together close 52%, at O4, gpt-oss 11% (Table 3). This explains why both "better admission" and "better prefetch" papers report small gains, and it tells a designer what to build. The link-to-CPU dependence (Fig. 2; r = 0.93 and 0.87 over 16 hosts) is a practical, machine-level design rule.
- **S4. The horizon rule in distinct experts.** Paging theory measures lookahead in distinct future pages (Albers; Breslauer). Bringing that unit to MoE routing gives a compact rule (D(W50) ≈ 0.65C, CV 0.16 across 26 points and 9 models). The rule was tested leave-one-model-out (Table 5). It is the most portable takeaway for router and predictor designers.
- **S5. Unusual transparency.** Every number I checked reproduces from the raw rows or the trace (Section 6: 15 of 15). The scorecard reports failures prominently (142 of 565, 39 of 273, 27 of 216). The limitations section is candid: the post-hoc half-ratio line, untested achievability of the bound, and the GPU running at half its datasheet rate.
- **S6. Cheap and reproducible.** About $70 of rented consumer GPUs, a single llama.cpp patch, and job scripts with the probes. Others can repeat this.

## 3. Weaknesses (ranked)

**W1. Clarity is the main obstacle. The paper is very hard to read, and a reader cannot easily extract its three findings.**
- The abstract carries about 15 numeric ranges.
- The introduction restates each finding with 4–8 more numbers.
- Section 4 paragraphs routinely stack 3–5 ranges over different host subsets ("on the 10 hosts…", "on the 2 hosts…", "at the two smallest budgets…", "at every host-bound budget…").
- Coined terms are needed to follow the argument but are never defined in one place: *deployed cache / deployed policy / online policy*, *single read*, *MIN, 1 read*, *in step / deployed path*, *host-bound budget*, *the gap*, *the rest*, *admit every miss*, and *the law* (appendices and README) versus *the calibrated model* (main text).
- Some statements are ambiguous enough to change their truth value (W5).
- Appendix B's first page is a single paragraph listing dozens of failed clauses.
- Figure 1 crosses 9 configurations, 6 budgets and 6 host series in one panel.

*Fix:*
- Cut the abstract to at most 6 numbers, one per claim.
- Define every policy name once in Table 1 and use only those names.
- Carry one running example through Sections 3–5 (host B or O4, gpt-oss 11%).
- Replace stacked prose ranges with per-host-class tables.
- Lead Section 4 with one waterfall figure per host class: deployed → MIN's set → one read → prefetched → bound.
- Move the scorecard narrative to the supplement.

**W2. The "bound" is a model-based ceiling of unknown tightness, so "25–43% of the bound" mixes system inefficiency with slack in the bound.**
- B_host is the highest of six one-shot probe samples (87.5 GB/s on host B; median 80.7), which a faster reader could exceed.
- The GPU term at the datasheet rate is roughly 2× loose at batch 1 (all-in-VRAM reaches 52% and 61%).
- All latencies are dropped and overlap is assumed perfect.
- The best oracle reaches only 46–73% of the bound.
- The undecomposed "rest" is 30–62% of the gap on fast-link hosts and 68–99% on slow-link ones.

The two-path tightening (1–7%) rules out one source of slack, but nothing shows the bound is approachable. So the paper's own question, "what the missing time is made of", is answered for less than half of the gap on the hosts where it matters most. Appendix E's Shapley accounting is model-based, and two of its five terms are definitional.

*Fix:*
- Add an achievability measurement: a stripped engine, or a microbenchmark, that executes MIN's per-step byte schedule (real expert sizes, copy engine plus CPU reads, GPU kernels of an all-resident token) with no policy logic. Report its time against Eq. (1) per host.
- Call Eq. (1) a "ceiling" or "speed-of-light model" rather than a bound, unless such evidence is added.

**W3. The headline "price of foresight" is measured against a baseline that loses to the deployed system, so it does not translate into a deployable speedup.**
- The in-engine shares (0.32–0.47 at W=4; 0.20–0.26 at recall 0.5, W=8) are fractions of MIN's gain over *admit every miss*.
- In time, admit-every-miss runs 0.42–1.00× the deployed cache (panel mean 0.77× at gpt-oss 11%).
- The paper concedes that a 4-token window is slower than the deployed cache on 6 of 10 hosts at 11%. On the deployed path a 16-token window gains at most 1.12×.
- The abstract's "half of the oracle's gain … needs exact routing of the next 2 to 10 tokens" is a statement about trace reads, not time. It does not say so.

*Fix:*
- Report window shares relative to the deployed cache as the primary in-engine quantity.
- State the baseline in the abstract.
- If possible, combine the window with single-read loading on the deployed path, which is the deployable design implied by S3.

**W4. Scope is narrow for the generality of the claims.**
- Two models in the engine. The panel is gpt-oss at two budgets plus Qwen3 at one budget on three hosts, with no Qwen3 25% panel run.
- Batch 1, greedy, teacher-forced, the first 256 tokens (the 2,048-token check covers only the system comparison).
- AIME-25 prompts, which "were also used while developing the system".
- RTX 5090 for every oracle result.
- No task accuracy (KL parity only).
- The distinct-expert rule is validated on traces of nine models but measured in the engine on only two.
- Only two hosts have a link slower than half the CPU rate, and the half-ratio line was drawn after the panel ran.

*Fix:*
- Add the missing Qwen3 25% panel cell.
- Add a held-out prompt set for Table 2 and one oracle cell.
- Add 3–4 slow-link hosts with probe-only predictions committed beforehand.
- Run one more model in the engine at a host-bound budget.

**W5. Several stated ranges depend on unstated choices; some are true only under one reading.**
- *"At the two smallest budgets the best of these runs 0.94–1.15× the deployed cache on O4 and O5"* (Section 4) holds only if "two smallest budgets" means the smallest budget of each model (gpt-oss 11%, Qwen3 12.5%). If it means each model's two smallest budgets, O4 gpt-oss 25% Belady-prefetched runs 1.53× (foresight_096a.json).
- *"MIN with its admissions copied ahead beats the best usual way by 1.18–2.03×"* takes its upper end from O3, where only one usual way (unpaced Belady) was run. On O4 and O5, where all usual ways ran, the range is 1.18–1.43×.
- *"Under-predicts MIN with one read by up to 19%"* does not say whether time or speed is meant; the artifact shows predicted time 18.6% *below* measured.

*Fix:* Name the cells behind every range, and state the error convention once and use it.

**W6. The time model's predictive claim is weaker than "predicts the time of every state within a median 2.1%" suggests.**
- Its inputs X_c and X_p are each run's own measured byte counters, so it maps measured bytes to time, not policy to time.
- G (4.1–9.2 ms across fits) is roughly a quarter to a half of the step time, more at the GPU-bound budgets.
- In the blind test the one-path model ties it on time (3.4% vs 3.6% median).
- Its distinctive success, the sign of the in-step copy, rests on 2 losses at 1–2 machines (37 of 39 for one-path vs 39 of 39).
- Background-copy states are mispredicted by a median 16%.
- It fails outright on the 12-channel EPYC (18.7% median, Table 7).

*Fix:*
- Predict ratios from trace-replay counts, which a deployment would have before running, and validate on new slow-link hosts.
- Report the one-path baseline beside every two-path number in the main text.

**W7. The pre-registration is a lab notebook, not a confirmatory design, and the paper leans on it more than it can bear.**
- Predictions are committed per job but are refined sequentially from earlier jobs. For example, job 100's prediction 9 (hosts e and f) was committed at 07:06:35, after hosts a–c had run.
- A third of the 565 clauses "held" only on the point estimate, as did 56% of job 100's.
- Many clauses are low-risk sign or engineering checks.
- Hand and script scoring disagree on 37 clauses.

None of this is misconduct, and the honesty is welcome. But "every prediction was committed before the machine it concerns started" carries less evidential weight than a reader may assume.

*Fix:* In the main text, list only the confirmatory predictions that test the three findings, with their pass rate. Leave the full ledger in the supplement.

**W8. The published-systems audit is not like-for-like and should not be in the abstract.**
- The 52 rows span hardware from a GTX 1080 Ti to an RTX PRO 6000, and some use datasheet ceilings.
- Some traces come from sibling models (an FP8 trace for a Q4 deployment), and some budgets are imputed.
- The caption itself says rows below 5% "are more likely configuration mismatches".
- The 13.6% vs 27% comparison therefore mixes measurement setups with system quality.

*Fix:*
- Keep the audit as an appendix figure.
- In the abstract, compare only systems re-run on the same host: FreeToken, llama.cpp, KTransformers, pipelined sharding and leloch's cache.

## 4. Clarity: **2 / 5**

The prose is precise at the level of individual clauses but very dense at the level of paragraphs, and the narrative is hard to follow. The three hardest passages, with suggested rewrites:

**(a) Abstract, second finding.**
> "Knowing the future helps only if the cache uses it as MIN does: it caches only the experts MIN would cache, and loads each from host memory once, by a copy over PCIe, instead of running it on the CPU and then copying it. On hosts whose PCIe link reads at least half as fast as their CPU, doing both closes 26–60% of the gap to the bound; doing either alone closes at most 26%. Where the link is much slower, the copies must be made ahead of use, and a calibrated model of the two read paths predicts from a host's bandwidth probe which way pays."

*Rewrite:* "Future knowledge pays only when two choices are combined: admit only the experts Belady's MIN would keep, and read each admitted expert from host memory once (one PCIe copy, used in the same step) rather than twice (served on the CPU, then copied). Either choice alone closes at most 26% of the gap to the bound; together they close 26–60% on hosts whose PCIe link reads at least half as fast as their CPU. On hosts with a slower link the in-step copy loses, and copies must be issued ahead of use. A two-path bandwidth model, calibrated on other hosts, predicts from a host's probe which loading strategy wins."

**(b) Section 4, "The rest".**
> "The rest grows as the link slows relative to the CPU. It is not slack from letting the two read paths share the host's rate freely: a bound that splits MIN's reads between the CPU and the link at their probed rates is tighter by only 1–7% of the gap on the fast-link hosts and 0–2% on the slow-link ones. On a slow-link host every state the engine has either puts reads on the slow link or leaves them to the CPU helpers, which take 1.09–1.31× the probe's time per expert (Appendix C)."

*Rewrite:* "**What the best oracle leaves.** After MIN's admissions are prefetched with one read each, 30–62% of the gap remains on fast-link hosts and 68–99% on slow-link hosts. Three measured contributors: (i) the prefetcher reads 1.18–1.58× MIN's bytes; (ii) host reads overlap GPU work only partly (charging them serially over-predicts time by 1–38%); (iii) the GPU runs at about half its datasheet rate at batch 1. The remainder is not an artifact of the bound's single host rate: probing CPU and link separately tightens the bound by only 0–7% of the gap. It is larger on slow-link hosts because every engine state sends reads either over the slow link or to CPU helpers that run 1.09–1.31× slower per expert than the probe."

**(c) Introduction, finding (3).**
> "In the engine, exact routing of the next W tokens recovers 0.32–0.47 of the oracle's gain at W=4 and the smallest gpt-oss budget across a 10-host panel; with each foreseen expert right only half the time, 8 tokens recover 0.20–0.26 and the whole future 0.42–0.55. Given to the deployed policy instead, whose admissions are copied in the background, 16 tokens of foresight are safe but run at most 1.12× the deployed cache."

*Rewrite:* "We price partial foresight as the fraction of the speedup of MIN-with-one-read over admit-every-miss that a W-token window recovers. On a 10-host panel at gpt-oss 11%, an exact 4-token window recovers 32–47% of it. A window whose experts are each correct with probability 0.5 recovers 20–26% at W=8 and 42–55% with the whole future. Admit-every-miss is itself slower than our deployed cache on most hosts, so these windows beat the deployed cache only where the link is fast. Grafted onto the deployed policy instead (CPU serves misses, background copies), a 16-token window never slows decode but gains at most 12%."

Smaller items:
- Figure 1 needs fewer series, or one panel per host class.
- Table 1 should also say which configurations the trace study uses.
- "Host-bound budget" should be defined where Eq. (1) is introduced.
- Error signs should follow one convention ("predicted time over measured time − 1").
- The README and appendices call the time model "the law"; the main text does not.

## 5. Novelty against related work: **3 / 5**

The closest prior or concurrent work, and what this paper adds:

| Work | What it does | What is new here |
|---|---|---|
| Zhang [62], replay semantics | Trace-level evaluation contract for expert caching; 44–46% read gap between causal policies and MIN with bypass. | Same MIN-with-bypass reference, carried into the **time** domain, measured **in the engine**, and decomposed by a factorial. |
| Liang et al. [37], ICLR 2026 | Hit rate of an oracle cache with m-token lookahead as a function of C/k, across 20 models. | Lookahead in **distinct experts** (D(W50) ≈ 0.65C), tested leave-one-model-out; accuracy degradation (recall r); value measured in time. |
| WiSP [59]; Budgeting Bytes [58]; MoE-CAP [28] | Decode bounded by PCIe turnover, or by bytes over bandwidth for a given fetch set; utilisation against one peak. | The fetch set is MIN's optimal reads on the real trace, with two read paths, a GPU term and per-machine probed rates. Still a ceiling, not a proven bound (W2). |
| SAEM [57]; SpecMD [19] | Perfect knowledge for a stage-level planner (≤14% on an A100); benchmarks of speculative expert prefetch across hardware. | Per-step oracles in the engine on 16 hosts; dependence on the host's link-to-CPU ratio. |
| Cao et al. [8]; Jain & Lin [25, 26]; Albers [2]; Breslauer [6] | Integrated prefetching and caching; Belady prefetch over-fetches; lookahead measured in distinct pages. | "Foresight pays only when spent as MIN spends it" is the MoE instance of these classical results, which the paper acknowledges. The new element is the read-path interaction (CPU-serve-then-copy reads twice). |
| FreeToken [55]; DALI [66]; Fiddler, MoE-Infinity, ProMoE, HybriMoE, KTransformers, SeqMoE [50] | Systems that split misses between CPU and PCIe, or predict experts. | An engine 1.02–1.29× faster than FreeToken (1.17× at most after tuning, and 0.98× at one cell). An incremental systems contribution. |

The individual ingredients are mostly known. What is new is the combination: an in-engine factorial with the interaction, the host-ratio dependence with a model that predicts it, and the distinct-expert horizon rule. That is a solid characterization contribution, not a new technique. The paper should state this delta explicitly against [62], [37] and [57] in the introduction.

## 6. Soundness and the artifact checks: **4 / 5**

The measurement methodology is sound and better than usual for the area:
- paired bootstrap within host, and a crossed bootstrap over hosts;
- randomised configuration order with a cold cache;
- relaunch checks;
- FreeToken given its own best backend on a separate selection launch, and tuned in job 098;
- KL parity against an all-in-VRAM reference.

MIN with bypass is verified against exhaustive search: the test suite passes (15/15), with 300 random instances in `test_min_is_optimal_fetch_and_bypass`. The interpretive layer over-reaches in places (W2, W3, W6), and the post-hoc host-class line is acknowledged. That is why the score is 4 rather than 5.

**Artifact checks.** I recomputed each value from raw rows, traces or prereg JSON, using my own short scripts where possible. All 15 held.

| # | Claim (location) | What I checked | Result |
|---|---|---|---|
| 1 | Table 2, host B gpt-oss 11%: llama.cpp 34.9, FreeToken 54.0, ours 69.9, ratio 1.294 [1.278, 1.312] | Paired bootstrap from `results/081_headline_law@vast/bs1.jsonl` (launch 2, FreeToken hybrid) | **Held** (1.2944 [1.278, 1.312]) |
| 2 | Bound on host B: 172 / 430 / 519 tok/s for gpt-oss (Tables 2 and 16, "exact" column) | Recomputed R* (38.32 / 15.34 / 6.79 reads per token) with `mosl.cachesim` MIN with bypass on `route_aime25_gptoss.npz`, then Eq. (1) at B_host = 87.5, B_gpu = 1792 | **Held** (172.3 / 430.4 / 518.8) |
| 3 | 20 in-class published rows: median 13.6%, quartiles 8.1–20.6% (Section 3) | `prereg/audit/audit_sol.json` + `audit.json`, excluding speculative and lossy rows | **Held** (13.63; 8.08–20.62) |
| 4 | MIN with one in-step read 16–51% faster than deployed on O3–O5, all budgets; prefetched 25–81% (Section 4) | Ratios of means in `foresight_095/096a/096b.json` | **Held** (1.159–1.513; 1.254–1.809) |
| 5 | Table 3, O4 gpt-oss 11%: gap 10.6 ms; load 4, cache 4, interaction 43, Shapley 26/25, prefetch 13, rest 36 | Recomputed from per-problem tok/s in `foresight_096a.json` with limit 10.02 ms | **Held** (10.62; 3.9 / 4.5 / 43.2 / 26.1 / 25.5 / 12.7 / 35.7) |
| 6 | Time model with G frozen from other hosts: median 2.1% (90th percentile 10%) on 39 host-budgets of 14 hosts; sign of in-step copy right at 39 of 39; one-path misses both losses; G 4.1–9.2 ms | `prereg/hostdep_model.json` | **Held** (2.11%, p90 10.3; 39/39; one-path 37/39; 4.1–9.2) |
| 7 | Job 100 probe-only model: median 3.8% over 32; up to +20% on the half-width-link host; −19% on the slow-link host | `prereg/job100.json` probe_err | **Held** (3.83%; +20.1%; −18.6%) |
| 8 | Correlation of the in-step gain with log(link/CPU): 0.93 and 0.87 over 16 hosts; 0.80 and 0.70 without the two slow-link hosts | Recomputed with `scripts/fig_hostdep.points()` | **Held** (0.928 / 0.869; 0.802 / 0.698) |
| 9 | D(W50)/C median 0.65, quartiles 0.61–0.72, CV 0.16; Table 5 leave-one-model-out errors | `prereg/foresight/w50_distinct.json`; read the LOMO code (held-out model excluded from the fitted constant) | **Held** |
| 10 | Scorecard 565 = 218 + 193 + 142 + 8 + 4; job 099 273 = 190 + 42 + 39 + 2; job 100 216 = 41 + 122 + 27 + 26 | Recounted `scorecard.json`, `scorecard_099.json`, `scorecard_100.json` | **Held** |
| 11 | Parity: mean KL 0.0019 nats and top-1 98.5% (gpt-oss); 86 rentals on 52 machines for $69.7 | `parity_090.json`; recomputed the cost as hourly price × duration from `gpu/vast_ledger.json` | **Held** ($69.67, 52 distinct offers) |
| 12 | Table 15, gpt-oss 11%: exact W=4 closes 0.52; recall 0.5 at W=8 0.29; whole future 0.43 / 0.48 | Recomputed shares from reads in `prereg/value_map.json` | **Held** |
| 13 | Panel at gpt-oss 11%: W1 0.14–0.17, W4 0.32–0.47, W16 0.78–0.93, recall 0.5 at W=8 0.20–0.26, whole future 0.42–0.55; within-host half-width 0.009, SD 0.154, interval [1.12, 1.30] | Recomputed from per-host ms in `prereg/panel_099.json` | **Held** |
| 14 | FreeToken tuned: best ≤ 1.04× its default; ours over its best 1.17 / 1.17 / 1.09 / 1.02 / 1.04 / 0.98 | `prereg/freetoken_tuned_098.json` | **Held** |
| 15 | Pooled slots lower MIN's reads by 4.5–18.6% | `speed_limit_v2.json`, global vs exact | **Held** |
| (16) | Predictions committed before results (jobs 093–100) | `git log` on the gpu worktree: prediction commits precede result commits for each job | **Held** (job 100's prediction 9 followed hosts a–c's runs; see W7) |

Two stated ranges are true only under one reading (W5): "two smallest budgets", and the 2.03× upper end that comes from the host where only one usual way ran.

## 7. Questions for the authors

1. Can any engine reach the bound? What time does a policy-free executor of MIN's per-step byte schedule achieve on a fast-link and a slow-link host?
2. Why use the maximum of six probe samples for B_host rather than a high percentile of more samples? How do Table 2's percentages change with the mean (81.1 GB/s on host B)?
3. For a deployment decision, the time model would have to run on counts from a trace replay rather than from the run itself. How accurate is it then, especially the sign at slow-link hosts?
4. What are the in-engine window shares relative to the deployed cache rather than to admit-every-miss? Is there any window and recall on any host where a window policy beats MIN-with-two-reads on the deployed path by more than 0.03?
5. AIME-25 was used during development. Do Table 2 and the factorial's main effects hold on a held-out prompt set (MATH-500 or chat)?
6. FreeToken uses one pooled cache. Against the pooled bound, which its design permits, does the ranking in Table 2 or the "% of bound" story change?
7. How does the 0.65C rule behave at very small C/k (Mixtral, top-2 with C = 2–4) and at W < 1, where D(W) is approximated as kW?
8. The CPU helpers quantise activations to Q8_0. Is there any measurable effect on AIME answer accuracy, not only KL?
9. Of the 1,054 scored clauses, how many directly test the three headline findings, and what is their pass rate?
10. Was prediction 9 of job 100 written after seeing hosts a–c's results?

## 8. Scores

| | Score |
|---|---|
| **Overall** | **5 / 10** (borderline, leaning reject in the current form) |
| Confidence | 4 / 5 |
| Novelty | 3 / 5 |
| Soundness | 4 / 5 |
| Significance | 3 / 5 |
| Clarity | 2 / 5 |

**Rationale.** This is a rigorous, reproducible characterization paper with three useful findings for MoE offloading:
- systems sit at roughly a third of a machine ceiling;
- foresight pays only through MIN's admissions combined with single reads, and the link-to-CPU ratio decides which loading path wins;
- the needed horizon is about 0.65·C distinct experts.

At a 20–25% acceptance rate, three things hold it below the line: the density of the writing (W1), the unproven tightness of the ceiling (W2), and a headline "price of foresight" measured against a baseline the deployed system beats (W3). Narrow in-engine scope (W4) and incremental novelty against classical integrated caching and recent MoE-cache evaluation work also count. A clear rewrite with the achievability measurement would be a solid accept.

## 9. What would move the score up one or two points (about three weeks, about $8 of GPU)

The ledger averages about $0.81 per rental, so $8 buys about 10 rentals.

**No GPU cost (worth +1 alone if done well):**
1. Rewrite for clarity (W1):
   - abstract with at most 6 numbers;
   - one running example carried through all sections;
   - one glossary, reusing Table 1's names throughout;
   - Section 4 led by one waterfall figure per host class;
   - stacked prose ranges replaced by small tables naming the cells;
   - the scorecard narrative and the audit list moved to the supplement.
2. Reframe Eq. (1) as a ceiling, and state in the abstract that the best oracle reaches 46–73% of it (W2).
3. Make the deployed cache the primary baseline for every in-engine foresight share, and state the trace-reads basis of the 2-to-10-token horizon (W3).
4. Fix the ambiguous ranges (W5): "two smallest budgets", and 1.18–1.43× on O4/O5 with O3 reported separately. Use one error-sign convention.
5. List only the confirmatory clauses in the main text, with their pass rate (W7). Move the 13.6% audit out of the abstract (W8).
6. Anonymise and reset in the MLSys template.

**About $8 of GPU (+1 more):**
7. **Achievability microbenchmark, about $1.6 (2 rentals: one fast-link, one slow-link host).** Replay MIN's per-step byte schedule for gpt-oss 11% and 25%: CPU reads of the real expert bytes, PCIe copies on the copy engine, and the GPU kernels of an all-resident token, with no policy logic. This measures the attainable time against Eq. (1) and splits "the rest" into bound slack versus engine overhead.
8. **Three or four new slow-link hosts (link/CPU < 0.5), about $3.2.** Commit the probe-only time-model predictions (sign and ratio of the in-step copy versus deployed versus prefetched) before launch. This tests the model where it is distinctive and removes the post-hoc-threshold concern.
9. **Held-out workload, about $1.6 (2 rentals).** Table 2's six cells plus the 2×2 factorial at gpt-oss 11% on MATH-500 or a chat set, on one fast-link host.
10. **Fill the Qwen3 25% panel cell, or run a third model in the engine at one host-bound budget, about $1.6 (2 rentals).** This tests the 0.65·C rule in the engine, not only on traces.

Items 1–5 alone would take me to 6. With items 7 and 8 done, I would expect 6–7.
