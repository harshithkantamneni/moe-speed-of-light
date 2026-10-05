# Review 4a: MLSys 2027 Program Committee

**Submission:** *Where the Seconds Go: A Speed Limit and a Measured Accounting for Mixture-of-Experts Decode on One Consumer GPU*
**Date:** 2 October 2026

**What I examined:** I read the 30-page PDF in full. I also checked `paper/*.tex` (macros and literals), `prereg/*.json` and `*.md`, `scripts/`, `mosl/perfmodel.py`, the `runtime/` patch, and the job scripts and raw results in `/home/claude/gpu-branch`.

**Notation:** numbers marked *(recomputed)* are mine, computed from those files. I was asked not to run git, so I could not check that the predictions were committed before the measurements they concern.

---

## 1. Summary

The paper asks how fast batch-1 decode could run for a Mixture-of-Experts model whose experts mostly live in host DRAM, on one consumer GPU, and what the remaining time consists of. It makes five contributions.

1. **A "speed limit"** for exact-routing systems with C expert slots per layer (Eq. 3). It combines:
   - the reads of Belady's MIN with bypass on teacher-forced routing traces, charged at the highest probed host rate;
   - dense and GPU-run expert bytes, charged at the GPU's datasheet bandwidth;
   - a CPU that is free to run any expert.

   Table 3 tightens the limit four ways and gives a pooled variant.
2. **A "host-memory law"**, T = G + max(X_c/B_c, X_p/B_p, (X_c+X_p)/B_cp), with one fitted constant G per engine and model.
   - It was predicted blind on seven RTX 5090 hosts, with a median error of 3.7%.
   - Its per-layer form (Eq. 2) picks, per machine, how many of a layer's misses to copy over PCIe and how many to run on the CPU.
3. **An accounting of the gap** between the authors' llama.cpp expert cache and the limit.
   - First as a Shapley attribution over five modelled "fixes" (Fig. 1).
   - Then measured with oracle policies built into the engine that read future routing from a trace (Figs. 2 and 5; Tables 8–11).
   - Result on one host (O3): single-read MIN with bypass runs 1.25–1.38× the online policy serialised, and 1.42–1.67× with prefetch overlap.
   - In contrast, a Belady prefetch or serve-then-copy MIN reads 1.7–2.5× the optimum's bytes and loses at the two smallest budgets.
4. **Replay of nine models' own sampled text**, which:
   - compares online cache policies and measures hot-set drift;
   - fits W50 ≈ 0.59·(C/k)^1.33, the lookahead that closes half of the online-to-optimal read gap;
   - simulates speculative batching.
5. **An audit of 52 published batch-1 measurements** against a datasheet version of the bound (median 14.5%).

The cache itself is a roughly 3.5k-line llama.cpp patch. On two RTX 5090 hosts it runs 2.0–4.0× stock llama.cpp and 0.97–1.29× FreeToken. Every GPU job's predictions were committed before launch; 54% of the 287 scored prediction clauses held.

## 2. Scores

| Criterion | Score (1–5) |
|---|---|
| Novelty | 3 |
| Technical soundness | 2 |
| Experimental rigor | 3 |
| Clarity / writing | 2 |
| Significance | 3 |
| Reproducibility | 4 |

**Overall: 4/10 — weak reject.**

The in-engine oracle study is a real contribution, and the measurement process is unusually disciplined. But the paper as submitted is three to four papers compressed into nine very dense pages. Several headline claims are stated more strongly than the evidence supports, and in places they disagree with the authors' own repository:

- the foresight attribution;
- the published-systems audit;
- the blind "law";
- W50;
- "speculation is not foresight".

A focused rewrite that adds the controls listed in §6 could reach 6–7. A 9 needs a foresight mechanism that can actually be built, measured in the engine.

## 3. Strengths

**S1. The right question, answered in seconds (§1, §4, pp. 1–5).**
- The field reports speed-ups over baselines each paper chooses itself. Bounding offloaded batch-1 decode in seconds is the right framing for it.
- The bound puts the policy optimum (exact MIN with bypass, checked against exhaustive search in `tests/`) inside a two-resource roofline.
- Table 3 (p. 5) varies each assumption separately: keeping a served expert for the step versus not, pooled slots, warm start, median versus maximum B_host, measured GPU rate, and a per-layer sum.
- The cache is reported against both the datasheet limit (25–43%) and the limit with every tightening at once (33–57%). I recomputed Table 3 from `prereg/speed_limit_v2.json`; it matches.

**S2. Oracles inside a real engine (§5, Fig. 2, Tables 8–11, App. E). This is the paper's most valuable result.**
- Building future-routing oracles into an engine of this kind, and running five ways of spending foresight at six cells, is rare.
- The same-host contrast on O3 makes a point other builders can use: *how* foresight is spent decides whether it pays.
  - Hit-optimal runs 0.69–1.27× the online policy; single-read fetch runs 1.25–1.38×; fetch plus prefetch runs 1.39–1.49×.
  - A Belady prefetch admits what the optimum bypasses: its admissions alone are 1.57–2.16× R*. It pushes host DRAM to 71–82% of the probe's best rate, and it can lose to the online policy.

**S3. Pre-registration and honest reporting of failures (App. B, Table 6, pp. 13–21).**
- 299 clauses are scored under one rule, and all 103 failures are listed, most with an explanation.
- The single-read hypothesis (job 095) came out of the failures of jobs 093 and 094. It was then tested with predictions committed beforehand, and 44 of 56 clauses held.
- MLSys rarely sees this level of self-scrutiny.

**S4. Artifact quality (App. H).**
- Released: raw per-problem rows, probe outputs, job scripts with their predictions, analysis scripts, and a cost ledger (US$46.3 for 57 rentals).
- From the repository I regenerated the following, and found no discrepancies beyond rounding:
  - Table 1's ratios;
  - Table 3;
  - Table 4 (all 27 cells);
  - the Shapley shares;
  - the W50 fit (0.589·(C/k)^1.328 over 26 points);
  - the audit's class medians.

**S5. Candour about its own model (§5, p. 7; App. C, p. 21).**
- The paper states that its accounting model over-prices bytes 1.3–1.5× and overlap 1.6–3.5×, and under-prices the remainder 3.3–5.6×.
- It documents the law's failures with plausible mechanisms: +119% on a hybrid-core host, and +16% to +32% on a 12-channel EPYC.

**S6. Useful engineering (§3 Eq. 2; App. F, Table 12, Fig. 6).**
- Choosing the CPU/PCIe split per machine from a probe gains 3–8% on a fast link, and 21–34% (gpt-oss) or 9–18% (Qwen3) behind a link of half the bandwidth.
- The ablation (Fig. 6) is clear about where the 2.7–2.8× over stock llama.cpp comes from.
- Output parity is checked by KL divergence over the full vocabulary against an all-VRAM reference (App. F).

**S7. Breadth of the trace study (§6, Table 4).**
- It covers nine models' own sampled text, six online policies, a profiled static cache, and windowed Belady, at three budgets.
- The regime explanation (recency against frequency depends on C/k) usefully reconciles conflicting prior studies.
- I checked all 27 cells: in every one, four tokens of perfect foresight beat every online policy.

## 4. Weaknesses (ranked by cost to the grade)

### W1. Three-to-four papers compressed into nine pages; no single thesis; the abstract is very hard to read (cost: high)

**Where:** abstract and §1 (pp. 1–2); throughout §2–§6; Table 6 (pp. 14–21).

**What is wrong:**
- **Five contributions, each compressed.** The engine and its comparison with two systems; a performance model tested blind; a bound with four tightenings and a 52-row audit; a Shapley model plus an oracle factorial; and a nine-model trace study with a fitted scaling law and a speculation simulation. Each gets a paragraph packed with numeric ranges.
- **The abstract.** It makes about a dozen numeric claims. It uses these terms without defining them: "probed and datasheet rates", "serialised", "overlapped", "Belady prefetch", "copying what the CPU has just served", W50, C/k and "speculative batching".
- **Terms before definitions.** FETCH and "the law of Section 3" are used in Table 1's caption and in §2 before §3 introduces them. "Cell", "host-bound", R*, v(F) and v(O,F) arrive in the middle of arguments.
- **Overloaded sentences.** Many sentences carry three to six numbers with nested qualifications; see for example "Our first two oracle jobs …" on p. 7.
- **Two unnamed error conventions:**
  - Table 2 reports predicted-over-measured *speed*.
  - Tables 8 and 11 and §3's "over-predicts overlapped execution by up to 53%" are in *time*.
- **Undefined internal labels in Table 6:** `both2`, `both3p`, `ours v2`, `C51`, `lite`, `law table`.
- **The running example** (gpt-oss at 25% on host B) helps, but no single figure carries it from the measured time to the limit.

**Why it matters:** A PC member cannot tell:
- what is claimed versus measured;
- which host and which timing harness each number comes from;
- which result is the paper's main contribution.

The strongest result (S2) is buried.

**Fixable before 30 Oct 2026?** Yes, with 1–2 weeks of rewriting and no new experiments:
- pick one thesis;
- put about three numbers in the abstract;
- move the law, the audit and W50/speculation into supporting roles;
- define every term on first use.

### W2. The measured accounting does not isolate foresight; the model it is compared with is calibrated to fit by construction; "largest piece" is fragile at the running example (cost: high)

**Where:** §5 (pp. 6–7); Figs. 1 and 2; Table 9 (p. 24); `prereg/shapley_gap.json`; `prereg/foresight_095.json`.

**(a) The online → fetch step changes three things at once.** The paper labels this step "the bytes foresight saves" (34–49% of the gap). But it changes:
1. the admission decisions — this part *is* foresight;
2. the number of host reads per admitted expert;
3. the critical path — fetch copies sit on the critical path, while the online engine runs misses on CPU helper threads in parallel.

On O3 the online policy reads 3.3–6.6 admitted experts per token twice (once by the CPU, once by the background copy). For example: misses 60.1 versus reads 63.4 at gpt-oss 11%, and misses 29.8 versus reads 34.2 at gpt-oss 25%. Those second reads are 10–25% of the reads the fetch oracle saves at the host-bound cells, and 10–33% across all six cells *(recomputed)*. There is no engine run of an online single-read policy — the authors' own simulator variant `dfa-fetch` — to separate these effects. The paper's thesis is that foresight's value depends on how it is spent, so this control is required, not optional.

**(b) The Shapley model reproduces the measured time by construction.**
- Its GPU term is a per-cell residual, chosen so that the model with no fix applied matches the measured time.
- Its "overlap" fix also moves resident experts onto the CPU to balance the two resources. No engine realises that state, including the oracles.
- At the running example the result is a near-tie: foresight 2.51 ms versus overlap 2.36 ms, a 0.14 ms gap. Foresight is the largest marginal cost in only 70 of 120 orders.
- In the paper's earlier fixed order, overlap was larger (1.99 versus 1.98 ms).
- The macros `\shTieMs` and `\shOrdersGMid` exist in `wsg_numbers2.tex` but are never printed. The abstract states without qualification that the accounting "names routing foresight the largest missing piece where host memory binds".

**(c) About half of the gap is named rather than measured.**
- The measured decomposition (O3, fixed order) leaves 39–43% of the gap at the host-bound cells, and 52–56% at the GPU-bound cells, as a residual that is named but not measured.
- At the running example, the measured bytes term (34%) is *smaller* than this unmeasured remainder (43%).

**(d) Model and measurement are on different hosts, with different methods.**
- The model is Shapley on host B; the measurement is a sequential decomposition on O3.
- §5 does re-derive sequential model shares on O3, which is fair. But Fig. 1's Shapley shares on host B are never checked against a measurement.

**Why it matters:** The title promises to say "where the seconds go". As it stands, about half the gap is attributed by name only, and the measured "foresight" step contains a component that is not foresight.

**Fixable?** Partly.
- For (a): one rental (about 5 h, about US$3) running an online fetch-on-admit configuration beside the existing six, plus a few days of analysis.
- For (b): print the order robustness and the near-tie.
- For (c): measuring the remainder — per-layer timers for CPU misses served in sequence, and GPU kernel time at the same C — is about a week.

### W3. The oracle evidence comes from one host and one launch, in a fixed order with a warm cache, timed by a different harness; the double-read effect is inferred across hosts (cost: high)

**Where:** §5 (pp. 6–7); App. E (pp. 23–26); `gpu-branch/jobs/095_single_read@vast.sh`; the `ec-bench` code in the runtime patch.

**What is wrong:**
- **One host, one launch.** All single-read results come from one rental (O3).
- **Fixed order with a warm cache.** In each cell the six configurations run in a fixed order (`base → fetch → lead2 → both2 → both3p → hitopt`) in one `llama-ec-bench` process, and the expert cache carries over between them. App. E itself says the fetch oracle reads up to 17% fewer than the simulation at the GPU-bound cells "where its cache is warm from the previous configuration's run". There is no reversed order, no cold start per configuration, and no second launch.
- **A different timing harness.** Timing is `ec-bench`'s in-process per-step timer, over teacher-forced text with a 1,024-token prefill. That is not the streaming client used for Table 1, despite App. A rule 10.
- **The paper's own data show how large the uncontrolled effects can be:**
  - two rentals of the same CPU differ by 19–22% in speed;
  - launch order moved a ratio by 11% on the 3090 host (`prereg/grid_outcome_092.md`);
  - the pre-registered ±0.06 replication band failed at 5 of 6 cells on host S.
- **Double reading is inferred across hosts.** The "reading an admitted expert twice" explanation compares:
  - the serve-then-copy MIN oracle, run only on O2 (50.6 GB/s), with
  - the single-read fetch oracle, run only on O3 (61.5 GB/s).

  These are different hosts at different levels of DRAM saturation — exactly the variable the explanation relies on.
- **Over-admission and double reading are never separated on one host.** On O3 the only double-reading oracle is hit-optimal, which also over-admits (its admissions alone are 1.57–2.16× R*).

**Why it matters:** The abstract's two foresight numbers (25–38% and 42–67%) and its causal explanation rest on this experiment.

**Fixable?** Yes. Re-run job 095 plus the bypass oracle on two more hosts (one of them a 9950X3D like host B), with reversed order, a cold start per configuration and two launches. That is about three rentals of 5 h each (US$10–20) and a week of analysis.

### W4. The audit is not like-for-like and applies the bound outside its class; the published-versus-ours comparison is not supported (cost: medium-high)

**Where:** §4, last paragraph (p. 5); Fig. 7 and Table 13 (pp. 29–30); App. G; abstract; conclusion.

**(a) The audit uses a different formula from Eq. (3).** §4 says: "We applied a bound of the same construction as Eq. (3) … `scripts/audit.py`". The implemented bound (`mosl/perfmodel.speed_of_light_time`) differs in three ways:

| | Eq. (3) | Audit bound |
|---|---|---|
| Dense bytes | overlapped with host reads at the token level | added in sequence with expert time |
| Overlap | token level | per layer |
| Host read rates | one shared B_host | separate datasheet CPU and PCIe rates |

Take the best trace-scored row: SeqMoE, gpt-oss-120b, RTX PRO 6000, C = 57. Eq. (3) with the audit's own R* and host rate gives about 579 tok/s, against the audit's 378. That would put the row at 24% instead of 36.8% *(recomputed)*.

**(b) Different ceilings.**
- Published rows use the top of a datasheet DRAM band, for example 563–614 GB/s for the KTransformers hosts, 115–256 GB/s for Fiddler's, and 153–205 GB/s for SP-MoE's.
- The authors' own rows use a probe.
- The paper's Table 2 shows its no-latency model failing on a 12-channel server, where a per-layer latency floor takes over. That is the class of host many of the audited rows ran on.

**(c) Rows outside the bound's class are counted.** By the repository's own flags, 9 of the 29 trace-scored rows are outside the stated class: 7 speculative (SP-MoE ×5, MoE-SpeQ ×2) and 2 lossy (Fate ×2). The in-class trace median is 13.6% (IQR 8.1–20.6%), not 9.5% *(recomputed)*.

**(d) Undisclosed budget imputation.** 21 of the 52 rows, including 12 of the 29 trace rows, set C to "card size minus dense, KV, 1.5 GB (upper bound)". This loosens their bound and lowers their percentage. The paper does not disclose it.

**(e) Incomparable classes pooled.** "52 … a median 14.5%" pools four incomparable classes: trace, i.i.d., no slots, and all fit.

**Why it matters:** The abstract's second number rests on this, and so does the conclusion's sentence: "published systems reach a median tenth of and ours a quarter to two fifths". The repository's own note (`prereg/audit/audit_sol.json`) says "the two sets are not a like-for-like ranking".

**Fixable?** Yes, in 3–5 days without a GPU:
- score your own rows and every published row with the same code;
- report class medians for in-class rows only;
- give a sensitivity band for the host ceilings and for the imputed C;
- drop the cross-comparison from the abstract and conclusion.

### W5. The "law" predicted blind is not the law the paper states, and "law" overclaims (cost: medium)

**Where:** §3 (pp. 3–4); Table 2; App. C (p. 21); `gpu-branch/jobs/ec2/law_predict.py`; `scripts/law_frozen_later.py`.

**(a) The blind predictor left out background copies.**
- §3 says X_c and X_p "come from the engine's counters of misses, fetches and admissions".
- The blind predictor counts link reads as fetches plus prefetches only. In the 19 configurations without FETCH it sets X_p = 0.
- Yet the engine's counters for those same runs record 9.7, 10.9 and 7.5 background admission copies per token at C = 14, 32 and 56 on host A, equal to 15–42% of the CPU reads (`results/073_samehost_v2@vast/ref_C*_r1.json`).
- App. C nonetheless says: "In 19 of the 29 configurations nothing is copied, so they test one roof."
- Under the §3 definition, host A's blind errors at C32 and C56 become −5.6% and −10.4%, instead of +1.0% and −2.5% *(recomputed)*.

**(b) The re-score switched definitions, and neither definition fits both models.** The frozen re-score on three later hosts does count admissions. On host S, with the frozen G:

| X_p definition | gpt-oss error (time) | Qwen3 error (time) |
|---|---|---|
| with admissions (re-score) | +2 / +10 / +15% | −8 / −7 / −7% |
| without admissions (blind test) | −2 / +2 / +6% | −11 / −11 / −15% |

How background copies are treated is therefore an unacknowledged free parameter *(recomputed)*.

**(c) G is a fudge factor, and Qwen3 was never predicted blind.**
- G = 4.82 ms is larger than the whole all-in-VRAM token (3.92 ms).
- Qwen3's G was never fitted on the cache or predicted blind: it is a llama.cpp residual measured on another host.

**(d) Reciprocal error definitions.** Table 2 uses predicted/measured *speed*; the re-score uses predicted/measured *time*. The reported "worst 16%" is −13.9% in Table 2's convention.

**(e) It fails outside desktop Ryzen hosts:** +119% on a hybrid-core host, and +16% to +32% on a 12-channel server.

**Why it matters:**
- The law is a headline contribution ("predicted blind on 7 machines"). It also sets the FETCH table and the accounting's "policy and read paths" term.
- What the evidence supports is a well-performing calibrated roofline for one engine and one model on desktop Ryzen hosts. That is not a law, and the blind validation is weaker than stated.
- The claim that "every mechanism acts only through the bytes" conflicts with the law's own treatment of background copies: free in the blind test, charged in the re-score.

**Fixable?** Yes, in 2–3 days without a GPU:
- choose and justify one definition (bytes on the critical path versus all host bytes);
- re-score both tests under it and report the per-budget bias;
- call it a model.

A blind Qwen3 cache prediction is one rental of about US$3.

### W6. Section 6's headline claims go beyond what the simulation shows (cost: medium)

**Where:** §6 (pp. 7–8); abstract; conclusion; `prereg/foresight/`; `prereg/batchk_outcome.md`; `prereg/policy_study_outcome.md`.

**(a) W50 is not "the horizon a forecaster needs".** W50 is the window of *perfect, exact* future routing (Belady within W, decayed frequency beyond) at which replay closes half of the read gap. A real forecaster makes errors that cost reads, and the paper gives no model of accuracy against horizon.

**(b) The baseline is not the deployed policy.**
- §1 says W50 closes "half of the gap between the deployed policy and the optimum".
- §6 says "the optimum reads 26–56% fewer … than our deployed decayed-frequency policy (median 41%)".
- But `scripts/fig_foresight.py` uses `dfa-fetch`: a single-read variant (each admitted miss fetched once) with the better of κ = 0 and the model's κ, at budgets of 12.5/25/50%.
- The deployed policy is serve-then-copy with κ, which is Table 4's "with hysteresis κ" at budgets E/8, E/4, 3E/8. Against it the optimum reads 26–61% fewer, median 43% *(recomputed)*.

**(c) The fit is loose:**
- the bootstrap exponent interval across models is 1.09–1.47;
- without the six points below one token, which are linear extrapolations below W = 1, the exponent is 1.51 (1.15–1.74);
- at C/k ≈ 7.5–8, W50 ranges from 5.9 to 15.8 across models;
- the largest point (gpt-oss at 50%) is 34, against 23.6 from the fit.

"≈ 0.59·(C/k)^1.33" is an empirical trend with roughly ±2× scatter, not a law.

**(d) "Speculation is not foresight" rests on invented draft routing.**
- It is a simulation with i.i.d. acceptance, and the routing of rejected drafts is invented: "free", "same" or "random". No draft model was run.
- The main text's losses use the "same" variant.
- By the repository's own numbers, at K = 8 and α = 0.9 with free drafts, batching closes 82% (OLMoE) and 133% (Mixtral) of the online-to-optimum gap — more than half, i.e. more than W50. So the abstract's "which speculative batching does not supply" holds only at large C/k.
- Not modelled: using draft-token routing as a prefetch hint ahead of verification, which is how SP-MoE-style systems use drafts. Real speculative systems also verify several tokens per pass, which is outside the bound's class.

**(e) Section 6 was not pre-registered.** The policy study's note says "Not a registered job: no predictions were written before the run". The paper should say so, given how prominently it presents pre-registration.

**Why it matters:** Two of the abstract's last three claims come from this section.

**Fixable?**
- Scoping the claims takes days.
- Recording real draft routing with an actual draft model (for example, a small Qwen3 for Qwen3-30B) and re-running the batching study takes about a week on one GPU.

### W7. The FreeToken comparison is not fully fair, and its explanation is inconsistent (cost: medium)

**Where:** §2 (pp. 2–3); end of §3 (p. 4); Table 1; App. F (pp. 26–28); §8.

**(a) Two different explanations of the same lead.**
- §3: the 1.27× lead at gpt-oss 25% "is largely the CPU path that FreeToken's bandwidth model declines to use there".
- §2 and App. F: "The lead is mostly admission rationing", with "77–84% of it at gpt-oss 25 and 40%" from the LRU ablation. The data give 77% at gpt-oss 25% (`prereg/stockclock_089.json`).

**(b) Engine kernel quality is never separated from cache design.**
- FreeToken's CPU executor alone decodes gpt-oss at 13.7 tok/s on eight performance cores (`prereg/slowlink_outcome_088.md`), far below what llama.cpp's CPU path reaches on the same host.
- The largest leads are on MXFP4 gpt-oss. On BF16 Qwen3 the lead is −3% to +15%. On FreeToken's own BF16 headline model the systems tie (1.00–1.03).

**(c) Tuning was not symmetric.**
- The authors' FETCH table is swept and selected per budget.
- FreeToken runs with its default thread count and fetch cap.
- Its backend was not re-picked on host S.
- On the 3090, FreeToken's own calibration recommended a backend that was not run.

**(d) VRAM is claimed to be reported but is not.** §2 says "we … report measured VRAM", but no table does. The repository shows FreeToken at 28.4–28.6 GB at every budget, against 9.5–25.9 GB for the authors' cache (`prereg/headline_081.json`).

**(e) The hosts differ in more than the CPU.**
- Host B's card reads 1,847 GB/s, against about 1,691 GB/s on the cards at the common clock — about 9% faster. The text emphasises a different comparison: "3% above the datasheet rate".
- Host B's card runs at a 575 W power limit with a 3,420 MHz maximum SM clock; host S's at 400 W and 3,105 MHz (`gpu.csv` of jobs 081 and 089).
- The paper attributes host S's lower numbers to the slower CPU alone.

**(f) KTransformers is missing.** It is the main CPU–GPU hybrid baseline. It was measured (job 078: the authors' cache was 2.2–4.4× faster) but does not appear in the paper.

**Why it matters:** The system comparison is the paper's most-cited number.

**Fixable?** Yes, in 3–5 days and about US$10:
- tune FreeToken's threads and fetch cap per host, as the authors' table was tuned;
- report VRAM;
- measure each engine's CPU-only expert throughput;
- add KTransformers.

### W8. What an MLSys reader would expect is missing (cost: medium)

**Where:** §5–§6; §9.

**(a) No foresight mechanism that can actually be built.** The conclusion is that foresight "must come from knowing the routing a few to tens of tokens ahead". But no predictor is run in the engine — not even a trivial one such as routing persistence, the previous token's experts, or a published forecaster — to show what fraction of the oracle's gain realistic accuracy recovers at the needed horizon.

**(b) The paper's own system keeps the inefficiency the paper diagnoses.** The deployed cache still uses serve-then-copy admission, which the paper identifies as the source of wasted bytes. An online single-read admission path is neither built nor measured.

**(c) Out of scope:** batch > 1 (prompts and batches above one bypass the cache), long contexts, and 16 GB cards.

**Why it matters:**
- Without (a), the paper diagnoses the problem but does not move the field forward.
- Without (b), the paper's own system contradicts its conclusion.

**Fixable?** (b) in 1–2 weeks. (a) in 2–4 weeks, with risk.

### W9. Statistical scope and workload generality (cost: medium)

**Where:** §2 (p. 2); §8 (p. 9); Apps. B and D.

**What is wrong:**
- **Narrow intervals.** Intervals cover 30 problems within one launch. Replication error is several times wider (see W3).
- **Under-reported launch-order effect.** App. D reports "one launch-order check exceeded 3% (7%)". The same job's outcome note also records a −11.0% order effect at gpt-oss 11%, "reported only".
- **One workload.** Every engine number uses AIME math prompts (which were also used during development), the first 256 decoded tokens, and a single request.
- **Two models in the engine.** Mixtral shows no gain.
- **The prediction record says the models get direction right but not size.** 87% of sign clauses held, but after job 088, 68 of 135 band (size) clauses failed. Point attributions such as "37–53% of the gap" should carry that caveat.

**Why it matters:**
- Practitioners care about chat and code workloads, longer contexts and small batches.
- Reviewers need to know whether 3–15% leads survive replication.

**Fixable?** Partly:
- intervals across hosts from existing jobs: days;
- a non-math workload at two cells on one rental: about US$3;
- a third model: a week.

### W10. The headline "% of limit" is dominated by terms no system can attain (cost: low-medium)

**Where:** abstract; §4 (pp. 4–5); Table 3.

**What is wrong:**
- **The GPU-bound cells mostly measure GPU kernel efficiency.** The abstract's 25–43% uses the GPU datasheet rate, of which batch-1 kernels reach 52–61%. Stock llama.cpp with every weight in VRAM reaches only 255/519 = 49% of the gpt-oss 40% limit. The repository's own outcome note says this metric "mostly restates the batch-1 kernel efficiency".
- **The host-bound operating point is infeasible.** At the host-bound cells the bound binds at c* = 0, where all R* reads would have to cross a link probed at 53 GB/s at 87.5 GB/s. The contribution "two host read paths" is not modelled: Eq. (3) constrains only their shared total.
- **B_host is optimistic.** It is the largest of six samples, 8% above their median.
- **A minimisation by normalisation.** §4 calls the per-layer bound "1–5% of a measured token" below the token-level bound. Measured against the limit itself, the difference is 3–18% (365 versus 430 tok/s at the running example).

**Fixable?** Yes, in days:
- lead with the measured-GPU, path-aware limit;
- keep the datasheet limit as the physical ceiling.

### W11. Reproducibility statements and presentation details (cost: low)

**Where:** README; `paper/paper.tex`; Table 6.

**What is wrong:**
- **"Every number is a macro" is not true.** Dozens of literals are hard-coded in `paper.tex`. Examples:
  - G = 4.82 / 5.06 ms; "+3.6%, +4.0%"; G = 4.3 ms;
  - "46% of the law"; 48 / 37 / 20 µs; "27–30 GB/s"; "13 and 30 experts";
  - 1.21×; 1,847 GB/s; 90.3 GB/s; "52–61%"; "8–9%"; "5–10%"; "1.4%"; "8%";
  - "109 tok/s"; the Nsight breakdown; "4 and 16 tokens"; "the fit gives 9 and 23"; "two to four tokens"; "1.8×"; "1.28–1.44×".

  At least one has drifted from its source (§7, item 11).
- **App. A rule 3** (use the harmonic mean for rates) is met only approximately: the paper uses the arithmetic mean of per-problem rates, checked to within 0.003.
- **Table 7's 3090 rows** omit gpt-oss 25% without explanation; the caption explains only the 40% and 43.75% budgets.

**Fixable?** Yes, in days.

## 5. Questions for the authors

1. Which definition of X_p is Eq. (1)? The blind predictor (`jobs/ec2/law_predict.py`) sets X_p = 0 for C14/C32/C56, while the engine recorded 7.5–10.9 background admissions per token in those runs. The frozen re-score counts admissions. What are the blind-test errors under the §3 definition, and what are the re-score errors under the blind definition?
2. G = 4.82 ms is larger than the whole all-in-VRAM token (3.92 ms), and the frozen-law errors on gpt-oss grow with budget on all three later hosts (from about +0.5–4% at 11% to +11–16% at 40%, in time). Is G really one constant per engine and model, and how does it vary with budget?
3. The six oracle configurations run in a fixed order in one process, with the cache carried over. What do you get with the order reversed, or with a cold start per configuration, and with a second launch? How much of the oracle gain at the GPU-bound cells comes from the warm cache?
4. What does an online decayed-frequency policy with fetch-on-admit (single read; your simulator's `dfa-fetch`) achieve in the engine on O3? How much of the online → fetch step is left after that?
5. Why was the serve-then-copy MIN oracle (bypass) not run on O3 beside fetch? Without it, how do you separate "admitting what the optimum bypasses" from "reading an admitted expert twice" on a single host?
6. In what sense is the audit's bound (`mosl/perfmodel.speed_of_light_time`: dense time in sequence, per-layer max, separate CPU and PCIe rates) "of the same construction as Eq. (3)"? What do the 52 rows look like scored with your Eq. (3) code?
7. What is the trace-scored median with the speculative (SP-MoE, MoE-SpeQ) and lossy (Fate) rows excluded? What is it when the 21 rows with an imputed C are bounded at the C each system actually used, or excluded?
8. Do you agree that the conclusion's comparison — "a median tenth" for published systems against "a quarter to two fifths" for yours — should be removed, given that the ceilings and the bound functions differ?
9. Which policy is "the deployed policy" in §1 and §6: `dfa-fetch` at 12.5/25/50%, or serve-then-copy with hysteresis κ at E/8, E/4, 3E/8 (Table 4)? Why do the two analyses in the same section use different budgets and different baselines?
10. How sensitive is W50 to forecaster errors? For example, what precision and recall at horizon W keeps half of the oracle's read saving, and what accuracy does a published forecaster reach at W ≈ 10 on gpt-oss?
11. Can you record actual draft-token routing with a real draft model, and model draft routing as a prefetch hint, which is how SP-MoE-style systems use it, rather than inventing the routing of rejected drafts?
12. FreeToken's CPU-only executor decodes gpt-oss at 13.7 tok/s on eight performance cores. How much of the gpt-oss lead is kernel quality rather than cache policy? Can you report CPU-only expert throughput for both engines?
13. §3 attributes the 1.27× lead at gpt-oss 25% "largely" to the CPU path, while App. F attributes 77% of it to admission rationing. Which is it, and which experiment supports the CPU-path attribution?
14. Host B's card reads about 9% faster than the cards at the common clock, and has a 575 W power limit and a 3,420 MHz SM clock, against 400 W and 3,105 MHz on host S. How do you attribute the host B to host S drop to the CPU alone?
15. Did anything in FreeToken's 28.4 GB footprint (staging buffers, pinned pools) help or hurt its expert path? Why is VRAM not reported, given §2 says it is?
16. Why is the KTransformers comparison (job 078) left out of the paper?
17. Which prompts were used to tune the half-life (16), κ = 1 and the law's constants? What happens to the Table 1 cells on a non-math workload (chat or code)?
18. Fig. 6 suggests GPU-side sampling is worth 2–5%. How much of the 1.047× "cache costs nothing" result on Qwen3 (§4) is sampling rather than cache overhead?
19. Belady's MIN with bypass on *set* requests ("serves a token's hits first and may evict an expert already served this step"): up to what sizes was optimality checked against exhaustive search, and is it optimal over all orders of serving within a step?
20. Is "foresight is the largest piece" still true at the host-bound cells if overlap in the model is restricted to states an engine can realise (no resident experts moved to the CPU) and the per-layer sum is used?

## 6. What would make this a 9 (prioritised)

1. **One thesis and a rewrite (1–2 weeks, no GPU).** For example: *"Foresight is the lever, but only if spent as the optimum spends it."*
   - Main text: setting; the bound, with the measured-GPU and path-aware variant as primary; the oracle factorial; a buildable predictor (item 3).
   - The law becomes a short "per-machine split" subsection.
   - The audit and the trace study move to scoped supporting sections.
   - The abstract carries three numbers, each defined.
2. **A clean, replicated factorial in the engine (1–2 weeks, about US$40–80).**
   - Factors: {online decayed frequency, MIN-with-bypass oracle} × {serve-then-copy, fetch-on-admit} × {no prefetch, single-read prefetch with lead 2–3, paced}, plus hit-optimal.
   - At least three hosts (including a 9950X3D), two launches each, randomised order, cold start per configuration.
   - Every policy timed by the same harness, with intervals across hosts.
   - Measure the remainder (CPU misses in sequence, GPU kernel shortfall at the same C) instead of naming it.
3. **A foresight mechanism that can be built (3–6 weeks; the step that makes this a 9).**
   - Wire at least one real forecaster into the single-read prefetch path. Candidates: routing persistence, a small learned per-layer forecaster trained on the trace corpus to predict the next W tokens' expert sets, or a published forecaster.
   - Report the share of the oracle's gain recovered against horizon and against precision/recall, and close the loop with W50.
4. **Ship the fix the paper diagnoses (1–2 weeks).** Give the deployed cache an online single-read admission path — fetch-on-admit, or deferred admission at the next use — and measure it against serve-then-copy at the host-bound cells.
5. **A like-for-like audit (under 1 week).**
   - One implementation of the bound for your rows and the published ones.
   - Headline medians for in-class rows only.
   - Disclose the imputed C, and give a sensitivity band for the host ceilings.
   - Remove the cross-comparison from the abstract and conclusion.
6. **Repair the law (2–3 days, plus one rental for a blind Qwen3 cache prediction).**
   - One definition of X_p, applied to both tests.
   - Per-budget bias reported.
   - Renamed a "calibrated model", with its domain stated (desktop hosts with DRAM bound below a latency floor).
7. **Generalisation (about 2 weeks).**
   - A chat or code workload at two cells.
   - One more model with many small experts.
   - Batch 2–4 with the cache enabled, or a clear account of why that is out of scope.
8. **Section 6 with scope (about 1 week).**
   - W50 stated as a perfect-foresight read horizon, plus a model of forecaster accuracy.
   - Real draft-model routing for the speculation study.
   - Pre-register the trace predictions.
9. **Fair baselines (3–5 days, about US$10).**
   - FreeToken threads and fetch cap swept per host; VRAM reported.
   - CPU-only kernel throughput for each engine.
   - KTransformers added to the main comparison.

Items 1, 2, 4, 5, 6 and 9 are feasible before 30 October 2026 at a total of about US$100 of rentals. Item 3 is the large one and determines whether the paper becomes significant rather than careful.

## 7. Factual and internal-consistency errors

1. **The 1.7–2.5× range does not belong to the first two oracle jobs.**
   - §5 (p. 7): "Our first two oracle jobs, on hosts O1 and O2 … Both read 1.7–2.5× R*".
   - Conclusion (p. 9): "spent as a Belady prefetch or through the serve-then-copy admission of its first oracles, it read 1.7–2.5× the optimum's bytes".
   - Data: O1 hit-optimal 2.02–2.18×; O2 hit-optimal 1.98–2.21×; O2 bypass 1.73–2.04×. Together that is **1.7–2.2×**.
   - The 2.5 comes from job 095's hit-optimal on O3 (2.47 at Qwen3 12.5%). `scripts/foresight_paper.py` pools all three jobs into `\fsExcessReadsOpt`.

2. **"Deployed policy" names two different baselines.**
   - §6 (p. 8): "The optimum reads 26–56% fewer experts from host memory than our deployed decayed-frequency policy (median 41%)".
   - Table 4 (p. 8) gives the deployed policy ("with hysteresis κ") a range of 1.36–2.55, i.e. **26–61% fewer, median 43%** over the 27 cells.
   - The 26–56% is computed against `dfa-fetch` (single read, best κ, budgets 12.5/25/50%), not the deployed serve-then-copy policy.
   - §1 repeats the mislabel: "the gap between the deployed policy and the optimum".

3. **Two explanations of the same lead.**
   - §3 (p. 4): "the lead over FreeToken at that cell, 1.27×, is largely the CPU path that FreeToken's bandwidth model declines to use there".
   - §2 (p. 3): "The lead is mostly admission rationing".
   - App. F (pp. 26–27): "the admission policy is most of the lead over FreeToken (77–84% of it at gpt-oss 25 and 40% …)".

4. **"Nothing is copied" is false, and the stated definition of X_p was not the one tested.**
   - App. C (p. 21): "In 19 of the 29 configurations nothing is copied, so they test one roof." The engine's counters for those runs show 9.7, 10.9 and 7.5 admission copies per token on host A (`results/073_samehost_v2@vast/ref_C14/C32/C56_r1.json`).
   - §3 (p. 3): "X_c and X_p come from the engine's counters of misses, fetches and admissions". The blind predictor (`law_predict.py`: "link = fetches + prefetches", with link = 0.0 for C14/C32/C56) contradicts this; the frozen re-score (`law_frozen_later.py`: "over the link = fetches + admissions") follows it.

5. **The two error figures use reciprocal definitions.**
   - Table 2 (p. 4): "Error of predicted over measured speed".
   - §3 (p. 4): "worst 16% (gpt-oss-120b 40% on host S)". This is t_pred/t_meas − 1. In Table 2's convention it is **−13.9%** (115.8 predicted against 134.5 measured tok/s).

6. **The audit's bound is not Eq. (3).** §4 (p. 5): "We applied a bound of the same construction as Eq. (3) (… `scripts/audit.py`)". The implemented bound puts dense time in sequence and uses separate CPU and PCIe rates. For SeqMoE gpt-oss-120b at C = 57 it gives 378 tok/s, against about 579 tok/s from Eq. (3) with the same inputs (36.8% versus 24%).

7. **The "median tenth" includes rows outside the bound's class, and the budget imputation is undisclosed.**
   - §4 (p. 5): "Even so, the trace-scored measurements sit at a median tenth, and three quarters of them below a fifth, of what their machines could deliver".
   - Conclusion (p. 9): "published systems reach a median tenth of [the limit]".
   - These include 7 speculative and 2 lossy rows that §4 itself places outside the class; the in-class median is 13.6%.
   - Table 13 does not disclose that 21 of 52 rows use an imputed, upper-bound C.

8. **Figure 5's caption is true only for long windows.**
   - Caption (p. 25): "the oracle, reading every admitted expert twice, stays at 44–45% of the limit at every window".
   - At W = 2 the oracle is at 37% (gpt-oss 11%), 25% (gpt-oss 25%) and 35% (Qwen3 25%) of the O1 limit. At W = 4 it is at 42%, 31% and 38%. At W = 16, gpt-oss 25% is at 42% (`prereg/foresight_093.json`; limits from Table 9).

9. **Not every comparison is timed by the client.**
   - App. A, rule 10 (p. 13): "One client times every system from streamed token arrivals; no server-side timers enter a comparison".
   - All of §5's oracle comparisons (Tables 8–11, Figs. 2 and 5), and the O1/O2 rows of the frozen re-score, use `llama-ec-bench`'s in-process per-step timer on teacher-forced text.

10. **VRAM is not reported.**
    - §2 (p. 2): "we match the number of experts on the GPU (within 2.5%) and report measured VRAM".
    - No table or figure reports VRAM. The repository has FreeToken at 28.4–28.6 GB at every budget, the authors' cache at 9.5–25.9 GB, and llama.cpp at 9.5–25.3 GB.

11. **A hard-coded range has drifted from its source.**
    - App. F (p. 28): "run gpt-oss-120b at 1.28–1.44× stock llama.cpp".
    - `prereg/competitors_outcome_077.md` says 1.28–1.45×, and Table 6 row 077 P1b has 1.446.

12. **App. D omits the largest launch-order effect.**
    - App. D (p. 22): "and one launch-order check exceeded 3% (7%)".
    - The same job's outcome note records **−11.0%** at gpt-oss 11% ("the noisiest host of the study").

13. **"As the law predicts" claims something the law does not model.**
    - App. D (p. 22): "the gpt-oss lead over FreeToken grows with the host's CPU-to-link ratio … as the law predicts".
    - App. C (p. 21): "FreeToken's own runs are not predicted, because its counters are not exposed".
    - The trend also rests on a single host at ratio 2.96 (job 088); every other gpt-oss point lies between 1.05 and 1.28.

14. **Figure 7 uses a different bound for FreeToken and llama.cpp from the one Table 1 prescribes.**
    - Fig. 7 (p. 29, right panel) plots FreeToken and llama.cpp against the per-layer limit (FreeToken at about 20–40%).
    - Table 1's caption (p. 3) and §4 say these designs are to be scored against the pooled bound (18–40% and 8–20%).

15. **Output parity is "similar", not "the same".**
    - §2 (p. 3): "the same divergence as stock llama.cpp's own CPU placement".
    - For gpt-oss: KL at the 99.9th percentile is 0.121 against 0.078 for stock; top-1 agreement is 98.5% against 98.8%; ΔNLL is +0.16% against −0.05% (`prereg/parity_outcome_090.md`).

16. **Eq. (3) does not constrain two read paths.**
    - §1, Contributions (p. 1): "from the exact Belady optimum with bypass, two host read paths and measured ceilings".
    - Eq. (3) has one shared B_host. At the host-bound cells it binds at c* = 0, an operating point that would need 87.5 GB/s over a link probed at 53 GB/s.

17. **The abstract's "largest missing piece" claim is unqualified; the body's data are not.**
    - Abstract: "A Shapley accounting names routing foresight the largest missing piece where host memory binds".
    - At the running example the margin over overlap is 0.14 ms, foresight is largest in 70 of 120 orders, and the measured bytes term (34%) is smaller than the unmeasured remainder (43%).

18. **The abstract's speculation claim does not hold for small caches.**
    - Abstract: "which speculative batching does not supply".
    - The repository's numbers (`prereg/batchk_outcome.md`) show free-draft batching at K = 8, α = 0.9 closing 82% (OLMoE) and 133% (Mixtral) of the online-to-optimum gap, more than the W50 criterion. The claim holds only at large C/k.

19. **A normalisation that understates a difference.**
    - §4 (p. 5): "A per-layer sum … sits 1–5% of a measured token below the token-level maximum".
    - Relative to the limit it lowers, the difference is 3–18% (365 against 430 tok/s at the running example).

20. **A misleading reference point for host B's card.**
    - §2 (p. 2) and §8: host B's card reads "3% above the datasheet rate".
    - True against the datasheet, but against the other RTX 5090s in the study (about 1,691 GB/s at the common clock, job 095's gate) host B's card is about 9% faster. That is the comparison that matters for Table 1.
