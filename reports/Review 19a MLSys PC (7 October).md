# Review 19a: MLSys 2027 PC, main track (7 October 2026)

**Submission:** "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth"

## Materials read

- `paper/paper.pdf`: all 37 pages. The main text I read in full from `paper.tex` because the two-column PDF text interleaves. I rendered pages 1–8 to check the figures and tables as typeset.
- Appendix sources, read in full: `app_wsg.tex`, `app_relation.tex`, `app_more.tex`, `app_traces.tex`, `app_value.tex`.
- Table sources: `tab_dm.tex`, `tab_headline.tex`, `tab_configs.tex`, `tab_names.tex`, `tab_prereg.tex`.
- Macro files, used only to identify which number a sentence prints: `wsg_dm.tex`, `wsg_decomp.tex`, `wsg_dcaime.tex`, plus a few macros grepped from the other `wsg_*.tex` files.
- `paper/supplement.pdf` / `supplement.tex`: structure and scorecard layout, skimmed.
- Authors' scripts, read **for definitions only**:
  - `scripts/decomp_measured.py`: state names and part definitions.
  - `scripts/fig_decomp.py`: machine selection and the Eq. (3) code.
  - `scripts/sumlaw_paper.profiles`: the T_GPU definition.
  - `scripts/speed_limit_v2.py`: docstring for the "exact" R\*.
  - `jobs/ec2/fetch_table.bandwidths` on the gpu branch: B_c, B_p and the ratio.
  - `scripts/speed_limit.host_rates`: B_host.
- Two derived outputs, `prereg/speed_limit_v2.json` and `prereg/decomp_measured.json`. I opened these only after my first pass disagreed with Table 3 by about 1.5 points, to find the cause: their R\* uses a different within-step convention. I then reproduced their R\* independently from the raw trace.
- `gpu/vast_ledger.json`: rental start times.
- Raw results on the gpu branch, `results/` for these jobs:
  - 081 and 089: system comparison, `bs1.jsonl`.
  - 084c: the gpt-oss AIME routing trace, `route_aime25_gptoss.npz`.
  - 093–108: the files `ec_*.jsonl`, `st_*.json`, `concur.txt`, `cores.txt`, `cpu.txt`, `nvidia-smi-q.txt`, `g_prof.json`, `prof_C*.json` and `readsched_C*.txt`.
- Job-script headers (registered predictions): jobs 081, 099, 100, 102, 104, 105, 106 and 107.
- `git log --format='%h %ad'` and file-content diffs of the job scripts edited after their rentals began.

## Independence statement

I did not open anything under `reports/` except to write this file. I did not open:

- `prereg/*outcome*.md`;
- `research_notes/`;
- any review, number-check, plan or progress file;
- `paper/paper_v1_prereview.tex`;
- `apply/`.

I did not read commit messages; I used only `%h %ad` and file contents. Every number in the claims table below comes from my own code in `scratchpad/rev19a/` (`raw.py`, `minsim.py`, `minseq.py`, `minhf.py`, `decomp.py`, `shares25.py`, `sec5.py`, `sec6.py`, `fact.py`, `rs.py`, `cf.py`, `viol.py`, `misc.py`, `sens.py`, `headline.py`), run on the raw rows, counters, probes, profiles and trace. Nothing was modified in either repository.

---

## Summary

The paper studies batch-1 decode of MoE models whose experts mostly live in host DRAM, on rented RTX 5090 machines. Its four contributions:

1. **A host-read bound (Eq. 1).** Belady's MIN-with-bypass gives the fewest host reads R\* any cache with C slots per layer can make. Those reads at the machine's best probed host-read rate give a lower bound on time per token. Two refinements follow: Eq. (2), where reads wait for the router so the GPU's non-expert time T_GPU is in series, and Eq. (3), where CPU reads sit in series with the layer chain and only link copies can be issued ahead.
2. **A measured decomposition of the gap.** The gap is the authors' llama.cpp expert cache's time beyond Eq. (1). In-engine oracles change what is cached (MIN's set) and how admissions are read (once or twice) in a 2×2, then add copies issued ahead. On 15 machines at the two host-bound gpt-oss budgets, neither change alone closes much of the gap at 11%, but together they close 33% (39% on the 13 machines whose link reads at least half the CPU rate, a subset drawn after the data). Reading ahead adds 15%. The rest is attributed to T_GPU and to the prefetching oracle's reads beyond R\*.
3. **How foresight should be spent**: single reads, the fewest-admission optimal schedule, and a read path chosen by the link-to-CPU ratio.
4. **How much foresight is needed**: a window/recall study and an exploratory horizon rule of about 0.65 C distinct experts per layer across 9 models.

Every experiment's predictions were committed to a public branch before its machine started. Appendix D scores 1,483 clauses.

## Strengths

1. **Reproducibility is exceptional, and the numbers are right.** I recomputed 40-odd quantitative claims from per-problem rows, engine counters, probes, Nsight profiles and the routing trace, without the authors' scripts. Every one I could check reproduces to the printed rounding (claims table below). This includes:
   - all 16 cells of Table 3;
   - the 31–54% share of Eq. (1) across 25 consumer machines;
   - Spearman 0.89 over 19 machines;
   - the 1.21× / 1.17× fewest-admission geomeans;
   - the running example of Section 5;
   - Table 4's host-B and host-S cells;
   - the read-schedule microbenchmark.

   My independent MIN simulation reproduces R\* exactly (38.315 and 15.340 reads/token). The engine counters are deterministic across machines; the fewest-admission/greedy counter ratios are identical on six hosts. Results are robust to pooling engine versions: restricted to jobs 099–101, Table 3 moves by at most 0.6 point.
2. **The methodology is right for the question.** Oracles inside a real engine, a factorial (not sequential) attribution of the first two parts, the machine as the statistical unit, and machine-level bootstraps are what a "where does the time go" paper should do. Problems carry ≤3% of the variance of the gain and machines 96%; I reproduced this. That justifies the design.
3. **The registration is real and the failures are reported.** Every rental for jobs 093–108 in the ledger started 4 s to 1.7 h after its job script's first commit. Post-start edits (jobs 100, 102, 105, 106, 107) are host-list amendments with predictions unchanged, except one added prediction in job 100, which Appendix D discloses. The paper reports a failed closed-form model ("three failed, one inconclusive") rather than hiding it.
4. **The central finding is interesting and well supported on the measured machines.** Better caching and single reads interact: interaction 37% [32, 42] of the gap at 11% and 16% [14, 18] at 25%. The paper explains it mechanically (MIN admits ~16 experts/token against the deployed policy's ~3.5, so double reads cost MIN's set much more), and the explanation checks against the counters. The decomposition also shows that nothing realisable beats Eq. (1): the closest configuration is 1.33× it. The dependency-aware Eq. (3) is a sensible refinement.
5. **The results are useful to practitioners.** Section 7 gives concrete rules (report distance to bound; single reads; admit less; choose the read path by the link-to-CPU ratio; predict in distinct experts). The probe that sets the read path takes seconds.

## Weaknesses (most important first)

1. **The decomposition is only partly measured, and parts of it are properties of this engine's copy path rather than of the problem.**
   - (a) **"Left after all three" is an accounting, not a measurement.** T_GPU is not removed in the engine. It is the smallest of 14 Nsight profiles taken on *other* machines (2.94 ms; range 2.94–3.38), and the extra reads are priced at B_host. The residual is −9% [−13, −5] at 25% and becomes −12% [−16, −7] with the median profile (3.16 ms): the two "named parts" over-account. Table 3 nevertheless prints these rows in the same format, with bootstrap intervals, as the oracle differences. Figure 2 is captioned "measured".
   - (b) **"Reading ahead" changes more than timing.** The prefetching oracle (`both3p`: `oracle_bypass=0`, lead 3, paced) is not MIN's set read earlier. It is a different admission rule that reads 1.22–1.31× R\* at 11% and 1.42–1.44× at 25% (I reproduced both from the counters). Part of "ahead" is therefore a change of set.
   - (c) **"MIN's set alone, still read twice" ≈ 0 at 11% depends on the copy path.** The paced background path has a single copier thread, moves 16 MB pieces, and admits nothing while 64 copies are queued. Appendix E itself attributes the bytes-optimal oracle's excess misses to "the queue cap, not only the latency". The headline "pay only together" result is therefore conditional on this implementation, and the paper does not test an uncapped two-read variant on the panel.
   - (d) **The "together" step moves admissions from the CPU to the link**, so "how it is read" is confounded with "which path reads it". The paper acknowledges this (link-ratio dependence), but the Section 7 rule is stated as if the two were separable.
2. **The headline shares rest on a post-hoc subset, and the registered test is reported favourably.**
   - The abstract leads with 39% on 13 machines. The full-sample figure, 33% [22, 42] on 15, appears only in the abstract, not in Section 4 or Table 3. The intro's 39% and Section 7 rule 2 carry no post-hoc flag.
   - Registered prediction 5 of job 099 was "positive interaction at **both** cells on **every** host, **and** reading the deployed admissions once changes the time by at most 5% on every host". As registered, it held on **8 of 10** hosts: it failed on 099f (interaction −0.48 ms at 11%; once-alone +5.6% / +17.3%) and on 099d (once-alone +8.5% at 25%). Table 1 says "held on 9 of 10", which is true only of the interaction's sign at 11%.
   - The other half of "pay *only* together", MIN's set alone ≈ 0, was never registered. Job 099's prediction 6 instead expected fetch/base 1.10–1.55 on every host, and that failed on the same two hosts.
   - The slow-link class is two machines.
3. **Scope and generality are narrow relative to the phrasing.**
   - One GPU SKU, two models, and AIME prompts that were also used in development. Batch-1, teacher-forced replay.
   - The decomposition is at two budgets of one model.
   - The interaction is large partly because the deployed policy admits rarely: about 3.5 admissions/token against about 16 for MIN on host 099e. With a baseline that admits often (LRU; admit-every-miss), single reads alone would matter. "Change what is cached and how it is read together" (Section 7) should be scoped to rationed-admission baselines.
   - The crossover read-path rule ("0.32 or less, copies belong in the background", in the intro) rests on two stably-running machines (Pf and the Threadripper 9960X). Only Section 7 says the crossover location is uncertain.
4. **The bound is relative to one probe run, and R\* uses an unstated relaxation.**
   - B_host is the single highest reading of a short probe, often run during the model download. By the authors' own (unconfirmed) closed-form account, the engine reads more than 5% faster than it on 12 of 84 launch-budgets. My recomputation gives implied rates up to 1.16× B_host at 11% and 1.22× at 25%. So the "31–54%" shares carry an unquantified probe error.
   - R\* is the MIN optimum in which, within a step, hits are served first and an expert already served may be evicted for a later miss in the same step. That gives 38.32 reads/token, against 39.03 if the step's experts are protected, which is what the engine's oracles do. A lower R\* is the conservative choice for a lower bound, but the main text never states it.
   - No measured configuration beats Eq. (1) or Eq. (3); the minimum is 1.33×. There is no observed violation.
   - Notation: Eq. (3)'s B_c and B_p are the best CPU-only and link-only readings, while the link-to-CPU ratio uses B_c at the helper thread count and the zero-copy B_p. The paper uses one symbol for two quantities.
5. **The realisable gains are small, so significance rests on the diagnosis.**
   - Without foresight: the admission margin gives 1.021× / 1.018×; the layer-ahead copy loses on every machine but the fastest-link one (down to 0.72× at 11%, 0.69× at 25%); forecasters close at most 15% of the read gap.
   - Everything larger needs oracle routing. The paper is honest about this ("Nothing realisable reaches it"), but a reader looking for a deployable technique finds little beyond "admit less" and "choose the path by the probe".
6. **Novelty is moderate.**
   - A roofline over MIN's bytes has close precedents, which the paper cites: WiSP, Budgeting Bytes, MoE-CAP, Zhang (2026b)'s 44–46% causal-vs-MIN read gap, and Liang et al.'s oracle-lookahead hit rates.
   - The fewest-admission schedule is the Berger et al. min-cost-flow formulation with a secondary objective.
   - Measuring lookahead in distinct items is from paging theory.
   - The distinct-experts rule is exploratory and only "as good as" a two-constant power law. On the gpt-oss trace it is also steep in W: at 11%, W = 3, 4 and 5 tokens correspond to 0.68 C, 0.84 C and 0.99 C.

   The in-engine oracle decomposition is the genuinely new part.
7. **Inference rests on few machines, and the registration is dilute.**
   - Fewest-admission: 9 stable machines, with narrow percentile intervals (the appendix gives t-intervals).
   - Two losses on registered hosts (106c at 0.846×, 107e at 0.974×) are excluded as "unsteady". The exclusion rule was itself registered, which is fair, but these are also the two slowest links (0.14 and 0.21), where the mechanism predicts a loss anyway.
   - The scorecard has 1,483 clauses: 495 held with an interval, 597 on the point estimate only, 308 failed and 83 untested. Most clauses are cell-level bands and signs. The quantities in the abstract (the shares of the gap, the 0.65 C rule, the ratio crossover) are exploratory or post hoc. "Every prediction committed before the machine started" is true, but it can be read as stronger confirmation than exists.
8. **The audit of published systems compares unlike things.** "Median 13.6% vs. ours 27%" (intro, Section 3) uses datasheet rates, uniform routing where no trace exists, traces of sibling variants, and a per-layer one-token bound applied to pooled, whole-layer and speculative designs. The paper lists these caveats, but the comparison still appears in the intro as a headline.

## Clarity

The paper is markedly better organised than a typical measurement paper. A careful expert can follow it, but it still takes too much work per claim. I score clarity **3/5**, at the upper end of that band.

**What works**

- **The question structure.** Sections 3–6 each answer one question, and their titles say which ("How fast could it be?", "Where the seconds go", "How foresight must be spent", "How much foresight is needed").
- **Table 1 (claims × evidence × section).** It is genuinely helpful, and the inline tags "(registered)" and "(exploratory)" on paragraph heads let a reader weigh each claim as they go.
- **The "Yardsticks" paragraph** (Section 2) says up front that each section uses a different denominator. That was the right fix to make.
- **Figure 1 and Table 2.** Figure 1 makes "read twice" concrete; Table 2 gives every configuration one name and a reads column.
- **The running example** (Section 5: 10.0 / 13.0 / 20.6 / 15.2 / 13.8 ms on one 9950X) anchors the abstractions in milliseconds. I verified it on host O4.
- **Figure 4.** The scatter of gain against link ratio, one marker per machine, is the clearest figure in the paper.
- **Section 7.** It turns the results into five short rules, each with its number.

**What still makes it hard to read**

1. **Four yardsticks and two baselines are still too many.**
   - Section 3: share of a bound's speed. Section 4: share of the gap. Section 5: speed relative to the deployed cache. Section 6: share of MIN's gain over admit-every-miss, a different baseline.
   - Figure 2 plots time/deployed while Table 3 gives shares of the gap, for the same data. A reader cannot move a number from one section to another without recomputing; for example, "closes 39% of the gap" against "1.36× the deployed cache" against "41% of the bound".
   - Suggestion: one primary yardstick (ms per token, with the bound as a reference line) and the others derived in captions.
2. **Populations change from claim to claim.**
   - Counts: 3, 5, 9, 10, 13, 15, 19 and 25 machines, plus 39 launches.
   - Rules: stable only; one launch per GPU; consumer only; "through job 104"; ratio ≥ 0.5; registered vs observed.
   - The abstract alone uses 25, 15, 13 and 9. Table 1 states the population for some rows but not all.
   - Suggestion: a one-line population tag on every numeric claim, or a small "who is in which analysis" table in Section 2.
3. **The abstract is overloaded.**
   - It carries eleven numbers, three machine counts, two qualifiers ("relative to that probe", "a subset we drew after the data") and two horizon ranges.
   - The main result (the interaction, and what is left) is hard to extract on first reading.
   - Exploratory status is flagged for the subset but not for the 0.65 C rule.
4. **The terminology is dense and front-loaded.** The "Terms" paragraph defines about ten terms in eight lines: MIN, admit, greedy, fewest-admission, once, twice, late, ahead, oracle, plus "fetch" and "in the step" from the paragraph before. These are then used without reminders. Appendix tables use codes (`foa`, `bypass`, `both3p`, `fetchplan`) that need Table 7 to decode, and the main text uses names that are not always Table 2's names ("MIN read once in the step", "MIN's set read once", "the best oracle").
5. **Table 3 and Figure 2 hide the measured/modelled distinction.**
   - Table 3's caption is three dense sentences ("*Together*, *ahead* and *left* sum to the gap, and the last three rows sum to *left*…").
   - Measured and modelled rows look identical.
   - In Figure 2, the "one change alone" column overlays two different states (circle and square) at one x position, and the x-axis mixes engine states with bounds.
   - Suggestion: shade or separate the accounted rows, and give the two single changes separate x positions.
6. **Figure 3 is a six-panel dot plot.** It has eight configurations, three named machines and grey panel ticks in small fonts. Its message ("MIN, 1 read gains everywhere; the others do not") would fit in one column showing the two host-bound gpt-oss budgets.
7. **Exploratory side analyses interrupt the main line.**
   - "A closed-form account (exploratory, not confirmed)" sits inside Section 4, between the measured decomposition and Section 5, and its last sentence says the decomposition does not rely on it.
   - The "field leaves most of the machine unused" audit sits in Section 3.

   Both would be better in the appendix, with one sentence left in the main text.
8. **The main text depends too much on the appendix.** It points to Tables 20, 24, 25 and 26 and to Appendices F–K for numbers it states, and the paper runs to 37 pages plus a 48-page supplement. Several main-text claims cannot be evaluated without the appendix: the 86–96% microbenchmark, FreeToken tuned, the probe's sensitivity.
9. **Word use drifts.**
   - "Launch" is defined as one rental (Section 2), but Table 4's "picked on host B on a separate launch" refers to two server runs within *one* rental (job 081).
   - "Registered … held" in Table 1 mixes interval and point holds; the caption says so, but the column does not show which applies to which claim.
   - B_c and B_p mean different probe readings in Eq. (3) and in the ratio.
10. **The prose is compressed.** Many sentences chain three or four claims with semicolons, for example in the "Statistics and registration" paragraph and the Section 9 bullets. It is precise, but a reader must parse every clause; a few more sentence breaks and one example per new term would help.

## Questions for the authors

1. Please make the full 15-machine decomposition the primary row set in Table 3 (my recomputation: together 33% [22, 42], ahead 15% [11, 18], left 52% [45, 62] at 11%), with the ≥ 0.5 subset secondary. Can the abstract and intro lead with it?
2. How much of "MIN's set alone, still read twice ≈ 1% of the gap" at 11% is due to the paced copy path: a single copier, 16 MB pieces and the 64-copy queue cap? Could you rerun `bypass` with the unpaced, uncapped path on two or three panel machines, as Appendix E did for the hit-optimal oracle on O1?
3. Is there a configuration that issues *MIN's own* admissions ahead (same set, earlier copies)? Without one, how much of "ahead" (15% / 20%) is a change of set? The `both3p` oracle reads 1.22–1.44× R\*.
4. Please state in Section 3 that R\* lets an expert served earlier in a step be evicted for a later miss in the same step (38.32 vs 39.03 reads/token at 11%). Can a real engine realise that schedule?
5. Can you profile T_GPU on two or three of the decomposition machines themselves, so the "left" split does not rest on profiles from other machines? With the median profile, the 25% residual is −12% [−16, −7]: what is being double-counted?
6. How sensitive are the 31–54% shares to the probe? For example: the median of repeated probes on an idle machine, against the single highest reading taken during the download.
7. Please report job 099's prediction 5 as registered (8 of 10 hosts), and label "set alone ≈ 0" as exploratory.
8. Does "pay only together" survive with a baseline that admits every miss (LRU, or `aa`)? The panel has `aa` and `w*` states; a 2×2 with `aa` in place of `base` would show whether the interaction is a property of rationed admission.
9. Table 4: were the selection and confirmation runs separate rentals? Job 081 is one rental in the ledger.

## What would raise my score

- **Separate measured from modelled.** Restructure Table 3 and Figure 2 so the oracle differences are visibly distinct from the accounted rows (T_GPU, extra reads, residual). Retitle Figure 2 accordingly.
- **Isolate timing in "ahead"** with an uncapped two-read MIN arm, and a MIN-set-ahead arm if feasible. That would show whether "pay only together" and "ahead" are properties of the problem or of this copy path. Q2 and Q3 would settle it.
- **Lead with the full-sample numbers**, flag every post-hoc number where it first appears (abstract, intro, Section 7), and report registered predictions as registered (Q7).
- **Scope the builder rules.** Rule 2 applies to rationed-admission baselines; rule 4's crossover rests on two machines.
- **Unify the yardsticks** around ms per token and the bound, and add a population table. Moving the closed-form account and the published-systems audit to the appendix would also shorten the main line.
- **Add generality evidence:** either a second GPU class (the supplement has 4090 and 3090 comparisons, but no decomposition), or the decomposition on Qwen3 at its host-bound budgets.

## Scores

| | Score |
|---|---|
| Overall (1–10) | **6** (weak accept) |
| Soundness (1–5) | **4** |
| Significance (1–5) | **3** |
| Novelty (1–5) | **3** |
| Clarity (1–5) | **3** |
| Confidence (1–5) | **4** |

The computations are correct, and the evidence is unusually transparent and reproducible. The interpretation (what "measured" covers, the generality of "only together") and the post-hoc emphasis stop me from going higher. The work is a careful, useful measurement study with a modest conceptual step beyond prior bound/oracle work.

---

## Claims checked against raw data

All values come from my own code on the raw rows (`ec_*.jsonl`: `decode_ms/n_decode`, averaged over problems, mapped to configurations by the `stats=` path), counters (`st_*.json`), probes (`concur.txt` and `cores.txt`), profiles (`prof_C*.json`, `g_prof.json`), the routing trace (`084c/route_aime25_gptoss.npz`) and `bs1.jsonl`. "Fast" means link-to-CPU ratio ≥ 0.5. Intervals are 10,000-sample percentile bootstraps over machines.

| # | Claim | Location | Paper | My value from raw data | Verdict |
|---|---|---|---|---|---|
| 1 | MIN reads/token R\* at gpt-oss 11% / 25% (the bound's input) | §3, Eq. 1 | 38.32 / 15.34 (implied by Table 4 and eq1 values) | 38.315 / 15.340 (hits-first, same-step eviction allowed); 39.030 / 15.466 if the step's experts are protected | Verified; the relaxation is not stated in the main text |
| 2 | R\* on the first 20 problems changes by −0.4% | §9 | −0.4% | −0.38% at both budgets | Verified |
| 3 | T_GPU = 2.9 ms, the smallest Nsight profile | §3, Eq. 2 | 2.9 ms | 2.94 ms (14 C14/C32 profiles: 2.94–3.38, median 3.16) | Verified |
| 4 | 15 machines ran every state; 13 with ratio ≥ 0.5; 2 slow at ≤ 0.40 | §4, Fig. 2 | 15 / 13 / 2 | 18 launches, 15 GPUs (100a, 100c and 101a are relaunches); slow ratios 0.29 and 0.40 | Verified |
| 5 | MIN's set alone, read twice: share of the gap | Table 3 | 1% [−1, 3]; 21% [19, 22] | 1.2% [−0.7, 3.0]; 20.8% [19.2, 22.3] | Verified |
| 6 | One read alone, deployed set | Table 3 | 1% [0, 2]; −1% [−4, 1] | 0.9% [−0.1, 2.0]; −1.1% [−4.0, 1.3] | Verified |
| 7 | Both together | Table 3 | 39% [32, 45]; 35% [30, 40] | 38.8% [31.9, 45.4]; 35.3% [30.1, 40.4] | Verified |
| 8 | Then reading ahead | Table 3 | 15% [11, 18]; 20% [18, 22] | 14.6% [10.5, 18.3]; 20.5% [18.1, 22.3] | Verified |
| 9 | Left after all three | Table 3 | 47% [42, 51]; 44% [40, 48] | 46.5% [42.1, 50.9]; 44.3% [40.1, 48.4] | Verified |
| 10 | Of it, T_GPU; 27–31% over the profiles' range | Table 3 | 27% [24, 29]; 33% [30, 36]; 27–31% | 26.9% [24.0, 29.4]; 33.4% [30.4, 36.0]; 26.9–30.9% | Verified |
| 11 | Of it, the oracle's reads beyond R\*; residual | Table 3 | 22% / 20%; −2% / −9% | 21.8% [20.2, 23.6] / 20.0%; −2.2% [−6.3, 2.4] / −9.1% [−13.3, −5.0] (−11.6% with the median T_GPU) | Verified; the residual signals over-accounting |
| 12 | Together closes 33% on all 15 machines | Abstract | 33% | 32.6% [21.6, 42.2] | Verified; absent from the main body |
| 13 | Best oracle closes 54% (49–58) on fast machines; ≤ 17% on slow at 11% | §4 | 54 [49, 58]; 17 | 53.8 [49.5, 58.0]; 16.8 | Verified |
| 14 | Prefetching oracle reads 1.22–1.31× R\* at 11% (1.42–1.43 at 25%) | §4 | as stated | 1.22–1.31; 1.42–1.44 (all 15) | Verified |
| 15 | All three removed: 1.07–1.38× Eq. (2) at 11% | §4 | 1.07–1.38 | 1.066–1.384 | Verified |
| 16 | Panel interaction registered, "held on 9 of 10" | Table 1 | 9 of 10 | Interaction > 0 at 11% on 9/10, at 25% on 10/10. Registered prediction 5 (both cells, every host, plus once-alone ≤ 5%) held on **8/10**: fails on 099f (inter −0.48 ms; once-alone +5.6% / +17.3%) and 099d (+8.5% at 25%) | Partly: the number is correct only for a sub-clause |
| 17 | 25 consumer machines at 31–54% of Eq. (1) speed at gpt-oss 11% | Abstract, §3, §10 | 31–54% | 25 machines; 31.2–54.0% | Verified |
| 18 | 40–54% of Eq. (3), 55–70% of Eq. (2) | §3 | as stated | 40.5–54.0%; 55.2–70.3% | Verified |
| 19 | Eq. (3)/Eq. (1): median 1.00, up to 1.56 | §3 | 1.00; 1.56 | 1.000; 1.557 (Threadripper 9960X) | Verified |
| 20 | Microbenchmark reaches 86–96% of the bound's read time | §3, Tab. 26 | 86–96% | Per-layer mode 0.861–0.963 (per-token 0.93–0.97); registered ≥ 0.50 / ≥ 0.80 held | Verified |
| 21 | Probe's second-highest reading a median 0.8% below the highest | §3 | 0.8% | 0.8% (max 26% on 107d) | Verified |
| 22 | Host B: ours 69.9, FreeToken 54.0, llama.cpp 34.9 tok/s; ratio 1.294; bound 172 tok/s, 41% | Table 4 | as stated | 69.9 (L2), 54.0 (L2), 34.9 (L1); 1.294; Eq. (1) = 5.80 ms → 172 tok/s; 40.6% | Verified |
| 23 | Host S, gpt-oss 11%: 57.9 / 48.0 / 28.9; Qwen3 12.5%: 33.5 / 32.7 / 16.1 | Table 4 | as stated | 57.9 / 48.0 / 28.9; 33.5 / 32.7 / 16.1 | Verified |
| 24 | Running example on a 9950X: Eq. (1) 10.0, Eq. (2) 13.0, deployed 20.6, MIN once 15.2, ahead 13.8 ms | §5 | as stated | Host O4 (096a): 10.02 / 12.96 / 20.64 / 15.16 / 13.81 | Verified |
| 25 | Same CPU model differs by up to 29%; relaunch within 3.2% | §2 | 29%; 3.2% | 29.1% (285K, 099a vs 099f; at 25%: 15.6%); max relaunch spread 3.2% (105b/106a) | Verified |
| 26 | MIN 1 read beats deployed at every budget on O3–O5 by 16–51%; up to 81% when ahead | §5, Fig. 3 | 16–51%; 81% | 1.160–1.514; both3p max 1.812 | Verified |
| 27 | "Usual ways" (no bypass; two reads) gain at most 15% or lose at each model's smallest budget | §5 | ≤ 15% | Max 1.148 (no bypass 1 read, Qwen3 12.5%, O4); others ≤ 1.047 | Verified |
| 28 | Fewest-admission copies 0.60–0.73 of greedy's; misses +2.5–5.3% in engine | §5 | as stated | 0.598–0.726; +2.5% / +5.1–5.3% (identical on six hosts) | Verified |
| 29 | Fewest-admission, read once at 11%: 9 stable machines, gm 1.21×; with 4 unsteady, 1.17× over 13; lost on the two slowest links | §5, Table 1 | 1.21×; 1.17×; 2 losses | 1.215× (first launch per GPU); 1.166×; losses 0.846 (ratio 0.14), 0.974 (0.21) | Verified |
| 30 | Spearman (MIN-once gain vs link ratio) 0.89 [0.63, 0.97] over 19 machines | §5 | 0.89 | 0.886 [0.62, 0.98], n = 19 | Verified |
| 31 | Machine carries 96% of the gain's variance; problems ≤ 3% | §5 | 96%; ≤ 3% | 96.3% / 1.1% (11%); 96.2% / 2.8% (25%), 19 × 20 | Verified |
| 32 | At ratio 0.28–0.32: greedy read-once 0.85–0.93×; fewest-admission CPU+background 1.11–1.22× | §5 | as stated | 0.853–0.933; 1.112–1.217 (job 103 excluded: it ran the greedy schedule) | Verified (2 machines) |
| 33 | Admission margin: 1.021× at 11%, 1.018× at 25% over 5 stable machines; reads −4–6% | §5 | as stated | 1.0213×; 1.0177×; reads −4.3 to −5.8% | Verified |
| 34 | Layer-ahead copy 1.04× where the link is fast; down to 0.72× at 11% | §6 | 1.04; 0.72 | 1.042 (105f); 0.722 (105e); also 0.691 at 25% | Verified |
| 35 | Next-token routing recovers 0.18 (11%) and 0.07 (25%) of MIN's read gain | §6 | 0.18 / 0.07 (trace replay) | Engine counters on the panel: 0.19 / 0.07 | Consistent |
| 36 | 8-token window at recall 0.5 recovers 0.29 at 11% | §6 | 0.29 | 0.29 (engine counters) | Verified |
| 37 | Exact 16-token window recovers 0.78–0.93 of the gain in time at 11% | §6 | 0.78–0.93 | 0.78–0.93 (mean 0.86, 10 machines) | Verified |
| 38 | Half the gain needs about 4 and 10 tokens; 0.66–0.81 C distinct experts on AIME | §6 | as stated | Engine read share at W = 4 is 0.52 (11%); log-interpolated W₅₀ ≈ 9.6 (25%); distinct experts per layer: 0.84 C at W = 4, 0.68 C at W ≈ 10 | Consistent (steep in W) |
| 39 | Closed-form account: implied rate 0.88–1.16 of B_host over 39 launches; within 8% on 37 | §4, App. H | as stated | 39 launches on 25 machines; 0.88–1.16 (0.94–1.22 at 25%); within 6%: 31, within 8%: 37 | Verified |
| 40 | Nothing beats the bound (implied by "bound") | §3 | — | Minimum over all configurations and launches of T / Eq. (1) = 1.33 (prefetching oracle, 100b); same for Eq. (3) | Consistent |
| 41 | Every prediction committed before its machine started | §2, App. D | as stated | All 71 rentals of jobs 093–108 started 4 s–1.7 h after the job script's first commit; later edits (100, 102, 105, 106, 107) are host amendments, one added prediction (100), all disclosed | Verified |
| 42 | Decomposition robust to pooling engine versions (implicit) | §9 | — | Jobs 099–101 only (n = 11): together 38.9%, ahead 14.6%, left 46.5% at 11% | Consistent |

**Not checked** (inputs not raw results, or too costly to reimplement):

- the 13.6% / 27% audit of published systems;
- the 0.65 C rule across 9 models;
- the "≤ 0.31 at any host-bound budget" next-token maximum (needs Qwen3 window replay);
- the forecaster and speculative-batch numbers;
- the KL-divergence parity (logit dumps not in the checkout);
- FreeToken tuned (job 098).

**Discrepancies found:** none in any number. Two issues of labelling or scoping remain:

1. Table 1's "held on 9 of 10" (8 of 10 as registered).
2. 39% on a post-hoc subset quoted unflagged in the intro and Section 7.

Two issues of definition:

1. The R\* within-step relaxation is unstated in the main text.
2. "Launch" means a rental in Section 2 but a server run in Table 4; B_c and B_p mean different readings in Eq. (3) and in the ratio.
