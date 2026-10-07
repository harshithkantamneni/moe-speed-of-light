# Review 10a: MLSys 2027 program committee, main track

**Submission:** "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth" (anonymous)

## Materials read

- **Paper:** `paper/paper.pdf`, all 32 pages (main text, references, Appendices A–L). I grepped `paper.tex` and the generated `wsg_*.tex` files only to find where specific macros come from.
- **Supplement:** `paper/supplement.pdf`. I read the front matter and Table 1 (jobs 073–098, first pages) in full, and Tables 6–8 (jobs 103, 104 and 105) in full.
- **Job scripts (gpu branch):** the header of `jobs/105_sumlaw@vast.sh`, which holds the registered predictions and the amendment. I also used the commit times of `jobs/104_minadm@vast.sh` and `jobs/105_sumlaw@vast.sh`.
- **Raw results (gpu branch):**
  - Jobs 093–105 (`results/<job>@vast/`): `concur.txt`, `cores.txt`, `nvidia-smi-q.txt`, `ec_g_C*.jsonl`, `ec_q_C*.jsonl`, `st_g_C*_*.json`, `g_prof.json`, `prof_C*.json`, `value_map_replay.jsonl`, `readsched_C*.txt`, `manifest.json`.
  - Job 069c Nsight profiles.
  - Job 081 `bs1.jsonl` (Table 2, host B).
  - The 30-problem gpt-oss routing trace `084c_gptoss_trace@vast/route_aime25_gptoss.npz`.
  - The gpt-oss-120b sampled-text trace `036_trace_retry@40gb/gpt-oss-120b_S.npz`.
- **Pre-registration and analysis JSON (main repo):** `prereg/linkaware.json`, `prereg/foresight/w50_distinct.json`, `prereg/audit/audit_sol.json`, `prereg/audit/audit.json`.
- **Authors' scripts:** read only to learn file formats and definitions:
  - `scripts/job105.py` and `scripts/sumlaw_paper.py`;
  - the loader part of `scripts/reanalysis.py`;
  - `host_info`/`cell_data` in `scripts/panel_099.py`;
  - `bandwidths()` in `jobs/ec2/fetch_table.py`, `host_rates()` in `scripts/speed_limit.py`;
  - the audit block of `scripts/wsg_numbers2.py`.

## Independence statement

This is a blind review. I did not open:

- anything under `reports/`, except to write this file;
- any `prereg/*outcome*.md` file;
- `research_notes/`, or `paper/paper_v1_prereview.tex`;
- the sibling directories `/home/claude/{reports,research_notes,review,scratch}`;
- any file named like a review, plan, number check or progress log.

I did not read commit messages. I used only `git log --format='%h %ad'` to order commits against job start times.

All recomputation used my own code, run on raw data in a scratch copy of the repository with `reports/`, `research_notes/` and the outcome files removed. The scripts are `rv_*.py` in `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/review10a/`:

- my own probe-rate parser and per-problem loader;
- MIN with bypass in three variants;
- a fewest-admission linear program written from scratch with scipy/HiGHS;
- a window-policy replay;
- the sum-relation, correlation and leave-one-model-out fits.

I modified nothing in either repository.

## Desk-level issues (not scored)

- **Template.** The paper uses `mlsys2025.sty`, and the PDF subject says "mlsys 2025". It should use the 2027 template.
- **Length.** The main text, including limitations and conclusion, ends on page 9, with references starting there, so it is within a 10-page limit. But it depends heavily on 12 appendices and a 34-page supplement.
- **Anonymity.** The paper refers to "the public gpu branch" without a link, which is fine. Appendix L describes the rental runner on Vast. I found nothing that identifies the authors.
- **Stale cost statement.** Appendix L gives the cost for "jobs 058–101". Jobs 102–105, which carry the paper's newest claims, are not included.
- **Unverifiable references.** Several cited works are 2026 preprints I cannot check (WiSP, Budgeting Bytes, SAEM, FreeToken, SeqMoE and others). My novelty judgement relies on the paper's descriptions of them.

## 1. Summary

The paper studies batch-1 decode of MoE models whose experts mostly live in host DRAM: gpt-oss-120b in MXFP4 and Qwen3-30B-A3B in BF16, on rented RTX 5090 desktop hosts.

**Two lower bounds on time per token.**

- **Eq. (1).** MIN-with-bypass host reads (Belady) over the machine's best probed host rate, overlapped with a roofline-style GPU term.
- **Eq. (2).** For systems that read only after routing: GPU non-expert time in series with the same reads.

**The authors' llama.cpp expert cache.** It is 2.0–4.0× stock llama.cpp and leads FreeToken at 11 of 12 cells on two hosts.

**An empirical relation, Eq. (3), T = G + R·S/B_host.** G is the profiled GPU kernel time and R is the engine's counted host reads.

- At gpt-oss 11% it holds within 6% on 28 of 30 launches on 21 machines.
- It held on all 4 launches of a pre-registered job (105).
- At 25% it over-predicts, and it failed its registered tolerance there.

**Oracles of future routing inside the engine.** These show that:

- foresight pays when spent as MIN spends it (MIN's set, each expert read once);
- the choice among MIN's hit-optimal schedules matters: the greedy schedule loses on slow-link hosts, while a fewest-admission schedule (computed by an LP) gains.

**The value of partial foresight.** Measured in replay on nine models and in the engine. Half of the gap from admit-every-miss to MIN needs routing for about 0.65·C distinct experts per layer, which no online policy or forecaster they tried reaches.

**Methodology.** Pre-registered predictions per job and a 565+ clause scorecard. Raw data and generating scripts are released.

## 2. Strengths

1. **Unusually reproducible.** Raw per-problem rows, engine counters, bandwidth probes, Nsight summaries and routing traces are all released, and every number is generated by a script. I recomputed about 30 quantitative claims from raw data with my own code (section 4). Almost all match to the printed precision; several match to the third decimal, including:
   - the LP's 12.399 admissions per token;
   - Tables 3, 4 and 19;
   - the O4 row of Table 10.

   I have rarely seen a systems submission this checkable.
2. **Honest, pre-registered reporting.**
   - The registered test of Eq. (3) failed at 25%; the main text says so.
   - The unpredicted layer-ahead loss is flagged as unpredicted.
   - The scorecard lists 142 failed clauses (jobs 073–098) instead of hiding them.
   - Commit times on the gpu branch precede machine start times: job 105's predictions were committed at 22:05:31Z, the hosts started 22:06:11–22:10:22Z, and the amendment (22:09:26Z) preceded the host it concerns.
3. **Statistics that fit noisy rented hardware.**
   - The paper treats the machine as the unit of variation. I confirm that relaunches agree within 1.9% in time and 0.030 in ratios, while two panel hosts with the same CPU model differ by 16–29%.
   - The panel randomises configuration order and uses paired bootstraps.
   - Cross-machine rank correlations come with intervals.
4. **Oracles inside a real engine, crossed with how experts are loaded.** This is the main methodological contribution and a useful correction to trace-only MoE caching studies:
   - it separates hit-optimality from time-optimality;
   - Belady prefetch and two-read loading waste bytes;
   - greedy MIN copied in the step loses on slow links (0.86–0.91×);
   - the fewest-admission set still gains there (1.03–1.04×), and by 0.08–0.17 over greedy on five machines.
5. **Evidence about how tight the bound is.**
   - A microbenchmark reaches 86–96% of the host-read term.
   - A link-aware variant forces admissions over PCIe.
   - Table 18 relaxes the bound's assumptions one at a time.
6. **Measuring lookahead in distinct experts.** This connects the work to paging-with-lookahead theory and gives forecaster designers a concrete target. The value of exact and noisy partial foresight is measured both in replay and in the engine.
7. **Fair baseline effort.** FreeToken was tuned in five settings on a separate host (job 098). The paper also checks a thread-count confound on hybrid-core CPUs and a headline model chosen by FreeToken.

## 3. Weaknesses (most important first)

**W1. The central relation, Eq. (3), is narrower than the framing, and its only confirmatory test failed at one of its two budgets.**

- **Where it holds.** The relation is established at one cell: gpt-oss-120b at 11%.
- **At 25%.** I find 17 of 30 launches within 6%, a median signed error of +5.9%, and a range of −2 to +18%. Job 105's registered test at 25% failed on 3 of 4 hosts (+6.9, +7.0, +9.3%) and on the pooled median (4.24% against ≤4%).
- **Not tested elsewhere.** Qwen3 and the GPU-bound budgets were not tested.
- **The framing goes further.** The Section 4 heading ("Time is GPU compute plus host reads") and the conclusion ("batch-1 MoE decode is the GPU's compute plus host bytes") generalise from that one cell.

**W2. The 30-launch fit is exploratory.**

- `scripts/reanalysis.py`, which contains the sum-law code, was first committed at 21:59:22Z. Job 105's predictions were committed at 22:05:31Z.
- So the model form was chosen with those 30 launches in view, including:
  - B_host as the maximum over about 20 probe readings, including concurrent CPU+PCIe sums;
  - R counting background admissions at the same rate as demand reads;
  - G as the sum of kernel categories other than wait and copy;
  - G as the median of the 7 profiles at that budget.
- "Nothing fitted" is true of numeric constants, not of the model form. The confirmatory evidence is 8 cells on 4 launches, of which 2 are new machines. The abstract should present the 28-of-30 result as exploratory.

**W3. The relation needs an overlap term that the paper does not model.**

- **My test.** I dropped admissions from R. The relation then under-predicts by a median of 5.0% at 11% and by 4.8% at 25%.
- **What that implies.** At 11% the admission reads add fully to time. At 25% roughly half of them overlap. The paper's explanation, that background copies overlap GPU work, is plausible, but it is not measured directly (for example, as copy-engine/kernel overlap in the Nsight traces it already has).
- **What content remains.** For a serial engine at a host-bound budget, "time = GPU busy + waiting for host" is close to an accounting identity. The substantive finding is that the wait equals R·S/B_host. My elasticity of −0.97 [−1.08, −0.81] supports that finding, but it is a property of this engine.

**W4. Novelty and significance are moderate for MLSys.**

The bound:

- is MIN reads over a probed bandwidth plus a roofline GPU term;
- is natural and close to concurrent bounds the paper itself cites (WiSP, Budgeting Bytes, Zhang 2026b's MIN-with-bypass gap, and Liang et al. 2026b's lookahead hit rates).

The fewest-admission result:

- **Not new in caching.** Choosing among hit-optimal schedules by admission (insertion or write) cost is known:
  - Jain & Lin (ISCA 2018) define Demand-MIN and Flex-MIN for the prefetch case (cited);
  - Berger, Beckmann & Harchol-Balter (SIGMETRICS 2018) compute offline-optimal caching as a min-cost flow over reuse intervals (FOO/PFOO; not cited);
  - flash caches ration admissions to save writes (for example, Flashield, NSDI 2019; not cited).
- **The LP.** The paper's LP over reuse intervals is essentially that flow plus an admission term. The paper should position it that way.

The practical yield:

- The main prescriptive results are negative: no realisable policy, learned order or forecaster comes close to the oracles.
- The actionable systems pieces are modest and mostly in appendices: the per-machine fetch table, avoiding layer-ahead copies on slow links, and the CPU path on slow links.
- The system's lead over FreeToken is 1.09–1.29× on gpt-oss but 0.97–1.15× on Qwen3, and a tie on FreeToken's headline model.

Missing related work I know:

- PowerInfer (SOSP 2024; hot/cold CPU–GPU split);
- LLM in a Flash (ACL 2024; windowed reuse of host-resident weights);
- SiDA-MoE (MLSys 2024);
- AdapMoE, SwapMoE, EdgeMoE and ExpertFlow on expert prefetch and caching;
- Michaud (TACO 2016) on the structure of optimal replacement;
- Kimbrel & Karlin on integrated prefetching and caching.

**W5. The external validity of the slow-link conclusions is thin.**

- **Two machines.** The class "link under a third of the CPU rate" is two machines: the Pf 285K (5 launches) and a Threadripper 9960X (1 launch).
- **Small gains.** There the fewest-admission set gains only 1.028–1.039×. That is the same size as the launch-to-launch ratio variation of up to 0.030 that the paper itself reports.
- **Intervals.** "Each 95% interval above 1" refers to within-host bootstrap intervals over problems, which exclude launch variance. Pf's two launches agree (1.035 and 1.028), which helps; the 9960X result is a single launch.
- **Workload.** All in-engine work uses one card class, AIME-25 prompts (also used during development), 20 problems × 256 teacher-forced tokens, and batch 1.

**W6. The headline audit of published systems ("median 13.6% of the first bound", in the abstract) is fragile and hard to trace.**

1. **Wrong bound for the class.** By the paper's own argument, Eq. (1) can be approached only with foresight. The audited systems mostly read on demand, so Eq. (2) is the paper's own bound for them. The paper scores its own cache against both bounds but scores published systems against Eq. (1) only.
2. **Unknown hardware.** The bound uses datasheet ceilings and "the top of the host-DRAM band", often for hosts whose CPU and RAM are not stated.
3. **Mismatched traces.** Routing traces come from different text, and sometimes from sibling model variants.
4. **Unlisted subset.** The n=20 subset is not identified in Appendix K. It is the trace-scored rows minus the lossy Fate rows and the speculative SP-MoE and MoE-SpeQ rows. Table 21 instead prints medians of 9.5% (trace, n=29) and 14.5% (all 52). A reader cannot recover 13.6% from the paper. I recovered it only from `audit_sol.json` and `audit.json`.

**W7. The pre-registration scoring inflates "held".**

- **Already-known outcomes.** Some job-105 clauses had outcomes known or deterministic before the runs:
  - **P5** predicted that fewest-admission copies are at most 0.65× greedy's at 11%. Job 104 had measured 0.597 on the same teacher-forced trace, and I find these counters identical on all 7 launches of jobs 104–105 (19.8 → 11.8 copies per token).
  - **P1** predicted a profiled G of 4.0–5.0 ms after job 069c had measured 4.28–4.69.
- **How much this matters.** Clauses like these make up much of the 20 "held (point)" outcomes among 31 clauses.
- **Timestamps.** The timestamps are git commit dates on the authors' branch, about one minute before launch. They are author-controlled; a third-party timestamp would be stronger.
- **Precision of the performance models.** The scorecard shows frequent misses in size (for example, 90 of 301 bands failed in jobs 088 onward). That tempers how much weight the per-host model predictions elsewhere can carry.

**W8. The bound's R\* is defined differently in different places.**

- **Eq. (1) and Table 2.** These use MIN that sees across problems and may evict an expert after serving it within its step. On the 30-problem trace I get 38.315 reads per token at 11% with my own LP, which reproduces the 172 tok/s bound at host B's 87.5 GB/s.
- **Sections 4–5 and Table 17.** These use within-problem, step-protected MIN. I reproduce 39.33 and 16.04 reads per token exactly with that variant, and 39.17/15.97 on the 20-problem runs.
- **A sequential within-step MIN.** This gives 38.62 reads per token.
- **Why it matters.** The differences are small (up to 3% at 11%; 4.6% at 25%, 15.34 vs 16.04), but "no policy, even one that knows the future, reads fewer" must refer to one stated definition. Evicting within a step should be stated as part of the bound's class.

**W9. The horizon rule's claim to "predict a held-out model" overstates.**

- **Unequal information.** The "one-constant" rule inverts the held-out model's own D(W) curve, which comes from its routing trace. From that same trace W50 could be simulated directly. The power-law comparator uses only C/k.
- **A residual trend.** D(W50)/C falls with budget: Spearman −0.41 against C/k across the 26 points (p = 0.04), with a range of 0.52–0.99.
- **No clear winner.** The rule ties the power law (my leave-one-model-out errors are 1.16/1.42/1.61 against 1.19/1.51/1.67).
- **What does hold.** It is useful as a unit for forecaster targets.

**W10. The paper is hard to read.**

- **Overall.** Sentences routinely carry 4–6 numeric ranges. Key terms are defined across Table 1 and scattered prose: "deployed path", "in the step", "two reads", "host-bound", "fewest-admission set". The abstract tries to state every result.
- **G median.** "G the profiles' median (4.3 ms)" is the median of the 7 profiles at 11%. The median of all 14 profiles is 4.40.
- **"Uncorrelated (Spearman 0.02)".** This is computed per launch; per machine it is −0.13, although the paper argues for the machine as the unit.
- **"28 of 30 launches".** This counts relaunches; by machine it is 19 of 21.
- **The abstract's first long sentence.** It joins the 30-launch relation with the 12-cell FreeToken comparison, which comes from 2 hosts.
- **O4 averaging.** Table 12 gives O4's in-step MIN as 15.13 ms while the text says 15.2, because the two use different averaging.
- **"Order-free (Shapley)" accounting.** For a 2×2 design this is the usual main effects plus half the interaction. It is accurate, but it adds little over reporting the interaction directly.

## 4. Claims checked

"Reproduced" means my own code on raw data matches the paper at its printed precision. "Approx." means it matches within rounding, or within differences that come from small choices such as how relaunches are aggregated.

| # | Claim | Where | My value | Verdict |
|---|---|---|---|---|
| 1 | Eq. (3) within 6% on 28 of 30 launches at gpt-oss 11%; median absolute error 1.1%; range −10 to 14% | Abstract, §4 | 28/30; 1.12%; −10.0 to +13.7% (19 of 21 by machine) | Reproduced |
| 2 | 14 Nsight profiles on 7 hosts give G = 4.1–4.7 ms; G = 4.3 ms | §4 | 4.07–4.69; per-budget medians 4.28 (11%) and 4.50 (25%); median of all 14 is 4.40 | Reproduced; wording imprecise (W10) |
| 3 | G implied per launch: median 4.3 (IQR 4.1–4.6); Spearman with link/CPU ratio 0.02 | §4 | 4.32 (4.14–4.58); 0.02 per launch, −0.13 per machine | Reproduced |
| 4 | Elasticity of T − G with respect to B_host −0.95 [−1.06, −0.84] | §4 | −0.97 [−1.08, −0.81] (bootstrap over machines) | Approx. |
| 5 | At 25% Eq. (3) over-predicts by a median of 6%; implied G 3.8 | §4 | +5.9%; 17/30 within 6%; up to +18%; implied G 3.85 | Reproduced |
| 6 | Job 105 at 11%: −1.5 to +4.7% on all 4 launches | §4, Table 4 | −1.5, +1.4, +2.0, +4.7% | Reproduced |
| 7 | Job 105 at 25%: 6.9–9.3% on 3 hosts, 3.8% on Pf; pooled median 4.2% > 4% | §4, Supp. Table 8 | +6.9, +7.0, +9.3, +3.8%; pooled 4.24% | Reproduced |
| 8 | Predictions committed before machines started (job 105) | App. B | Commit 22:05:31Z; starts 22:06:11–22:10:22Z; amendment 22:09:26Z precedes 105f. Sum-law code first committed 21:59:22Z | Ordering confirmed; the 30-launch fit is exploratory (W2) |
| 9 | Serialisation is 22–51% of the gap at 11% (31–55% at 25%) | Abstract, §4 | 22–52% (32–56%) on the 22 launches with per-host replay | Approx. |
| 10 | Running example on O4: Eq. (1) 10.0, deployed 20.6, MIN in step 15.2, copies ahead 13.8 ms | §4 | 10.02 (my LP's R\* at 50.7 GB/s); 20.64; 15.16; 13.81 | Reproduced |
| 11 | MIN reads 39.3 (11%) and 16.0 (25%) per token on the 30-problem trace | Table 17 | 39.33 / 16.04 (within-problem, step-protected); 38.62 / 15.39 sequential across problems | Reproduced; definitions differ (W8) |
| 12 | Bound on host B at gpt-oss 11%: 172 tok/s | Table 2 | 172.3 (R\* 38.315, B_host 87.5) | Reproduced |
| 13 | Fewest admissions of any hit-optimal schedule: 12.4 per token vs 21.8 greedy at 11% | §3 | My LP: 12.399 admissions, 38.315 misses, integral on all 36 layers; greedy 21.66–22.03 depending on within-step order | Reproduced |
| 14 | Same at 25% (authors' JSON: 15.34 misses, 6.75 admissions) | linkaware.json | 15.340 / 6.746, integral | Reproduced |
| 15 | Table 3, job 104: by CPU and in step, greedy and fewest, at 3 hosts × 2 budgets; copies 19.8 → 11.8 and 9.7 → 7.0 | Table 3 | All 24 ratios match to 2 decimals; copies identical | Reproduced |
| 16 | Table 4, job 105: G_prof, law vs measured, layer-ahead, greedy and fewest | Table 4 | All entries match | Reproduced |
| 17 | Fewest set copies 0.60–0.73× as many experts, for 2.5–5.3% more misses | §4 | 0.597–0.726; +2.5 to +5.3% (identical on every host) | Reproduced |
| 18 | Fewest set adds 0.08–0.17 to greedy's ratio at 11% on 5 machines (6 launches); −0.01 on O4 | §4 | +0.082 to +0.172; −0.015 | Reproduced |
| 19 | Slow links: greedy 0.86–0.91×, fewest 1.03–1.04× (3 launches); every fewest-set interval above 1 | §4 | 0.861–0.912; 1.028–1.039; lowest lower bound 1.015 | Reproduced; see W5 |
| 20 | Layer-ahead copy: 1.04× on O4; losses down to 0.72×; 84% of copies used | §5, Table 4 | 1.042; 0.722–0.909; 84% at 11% (73% at 25%) | Reproduced |
| 21 | Spearman across machines: MIN in step vs ratio 0.89 [0.63, 0.97] (19); copied ahead 0.87 [0.54, 0.99] (17); admit every miss 0.95 [0.75, 0.99] (13); MIN by CPU vs B_p 0.79 [0.49, 0.92] and vs ratio 0.45 | §4 | 0.89 [0.63, 0.98]; 0.87 [0.58, 0.99]; 0.95 [0.75, 0.99]; 0.79 [0.49, 0.92]; 0.45 [−0.01, 0.77] | Reproduced |
| 22 | Kendall's W 0.77 / 0.82; machines carry 96% of log-gain variance, problems 1–3% | §4 | 0.78 / 0.85; 97–98%; 1–3% | Approx. |
| 23 | Relaunches within 1.9% (time) and 0.030 (ratios); same-CPU panel hosts differ 16–29% | §2 | 0.0–1.9%; up to 0.030 (excluding states that changed by design); 16–29% | Reproduced |
| 24 | Read-schedule microbenchmark reaches 86–96% of the host term | §3, Table 19 | 86, 89, 96, 96, 95, 93% per layer; every Table 19 column matches | Reproduced |
| 25 | On O3–O5 at all six budgets, MIN in step is 16–51% faster, and 25–81% with copies ahead | §4 | 1.160–1.514×; 1.253–1.812× | Reproduced |
| 26 | Table 10, O4 rows (gap 10.6; 4/4/43; Shapley 26/25; prefetch 13; rest 36) | App. E | 10.6; 3.9/4.5/43.2; 26.1/25.5; 12.7; 35.7 (25%: 8.5; 4/22/17; 31/13; 22; 34) | Reproduced |
| 27 | Table 17: admit every miss 58.6/28.2; exact-window shares 0.18/0.31/0.52/0.81/0.98 (11%) and 0.07/0.14/0.25/0.42/0.66 (25%) | App. I | My window-policy replay: 58.64/28.22; 0.19/0.31/0.52/0.81/0.98; 0.07/0.14/0.25/0.42/0.66 | Reproduced |
| 28 | D(W50) ≈ 0.65·C (quartiles 0.61–0.72); held-out error table for the three rules | §5, Table 6 | 0.651 (0.610–0.721); distinct 1.16/1.42/1.61, power 1.19/1.51/1.67, C/k 1.20/1.61/2.44 (paper 1.22/1.57/2.42); D/C trends with C/k (ρ = −0.41) | Reproduced from the authors' per-point JSON; see W9 |
| 29 | gpt-oss-120b D(W) curve used by the rule | w50_distinct.json | From the raw S trace (every third layer): 4.0/6.8/11.4/18.2/27.7/39.8/53.7 vs 4.0/6.8/11.2/17.8/26.9/38.6/52.1 | Approx. (within 1–3%) |
| 30 | Table 2, host B: ours/FreeToken 1.294 [1.278, 1.312], 1.275, 1.154; 2.00× llama.cpp; Qwen3 1.032 | Table 2 | 1.294 [1.278, 1.311]; 1.275; 1.154; 2.001; 1.032 | Reproduced |
| 31 | Published systems in the class: median 13.6% (quartiles 8.1–20.6%, n=20) | Abstract, §3 | 13.6 (8.1–20.6), n=20, but only after excluding lossy and speculative rows that App. K does not list | Reproduced from the authors' data; not traceable from the paper (W6) |
| 32 | Link-aware bound raises the bound by at most 1% at 11% and 7% at 25% (24 probes) | §3 | Authors' JSON: maxima 1.009 and 1.072. I checked the frontier endpoints but did not recompute the per-layer frontier | Consistent; not independently recomputed |
| 33 | Table 5 and panel windows: deployed-path W=16 at 1.00–1.03 / 1.06–1.12; panel W=4 slower on 6 hosts, W=16 on 2, W=8 at recall 0.5 on 8 | §5 | All match | Reproduced |

**Not checked:**

- the two-path model's 2.2% median error on 14 hosts;
- the learned admission order;
- the speculative-batch numbers;
- the Qwen3 trace-level claims;
- the 4090/3090 grid.

## 5. Questions for the authors

1. **Eq. (3) beyond one cell.** Will you test it, pre-registered, on Qwen3, at a GPU-bound budget, and on at least one other engine whose counters you can instrument? If it holds only for your serial engine at host-bound budgets, will you say so in the abstract and the conclusion?
2. **The overlap term at 25%.** Can you register an overlap-corrected form, either a measured overlap fraction for background admissions or Nsight-measured copy/kernel overlap, and test it on new machines?
3. **R\* definition.** Which definition is "the" bound: cross-problem with within-step eviction (38.3 reads/token), or within-problem and step-protected (39.3)? Why is evicting a just-served expert within the same step physically allowed for the systems the bound covers?
4. **Launch variance on slow links.** With the launch-to-launch ratio variation you report (up to 0.030), what are the host-level intervals for the fewest-set gains of 1.03–1.04× on slow links? Is the Threadripper 9960X result repeatable?
5. **An online version of the fewest-admission set.** Can it be approximated online, for example with an admission threshold on predicted reuse probability, or a learned policy that imitates the plan? Even a partial online gain on slow-link hosts would make the finding actionable.
6. **The audit.** Please rescore published systems against Eq. (2), list the 20 in-class rows in Appendix K, and give the median using only rows whose host hardware is stated.
7. **Risky vs. known predictions.** How many of the 31 clauses in job 105, and of the 565 for jobs 073–098, predicted a quantity whose value was already known from an earlier job on the same trace? Can you report the risky clauses separately?
8. **The horizon rule.** If the held-out model's routing trace is available, what does the rule give beyond simulating W50 directly? Does the trend of D(W50)/C with budget persist at C/k > 8?
9. **Integrality of the LP.** The admission constraints break the network structure of the interval LP. Is integrality guaranteed, or observed? (I observed integral solutions on all 72 layer-budget instances.)

## 6. What would raise my score

- **Confirm Eq. (3) more broadly.** A confirmatory test with tolerances registered before the runs, covering a second model, both budgets, and at least four new machines, plus an overlap-corrected variant. Present the 30-launch result as exploratory.
- **Something realisable from the fewest-admission finding.** For example, an online admission rule that captures part of the slow-link gain, evaluated across machines with launch-level intervals.
- **Rework the audit.** Rescore published systems against Eq. (2), list the in-class subset, and either drop the 13.6% figure from the abstract or qualify it.
- **One R\* definition** used everywhere, with the within-step eviction assumption stated in the bound's class.
- **Position the LP and the "which MIN" result** against offline cost-aware caching (FOO/PFOO, Demand-MIN), and add the missing MoE-offloading and hybrid-execution work listed in W4.
- **A substantial rewrite for readability.**
  - Focus the main text on three claims: the bounds, where the time goes, and what foresight is worth.
  - Add one overview figure of the time decomposition.
  - Move secondary ranges to the appendices.
  - Fix the small inconsistencies listed in W10.
- **Broader hardware and workload.** A server host with 8–12 DDR5 channels, an x8 vs x16 link on the same machine, a 24 GB card in the oracle factorial, and non-AIME prompts.

## 7. Scores

- **Overall:** 5 / 10 (borderline reject)
- **Soundness:** 4 / 5. The measurements and statistics are careful, and nearly every number I checked reproduces. The deductions are for exploratory results framed as general, an incomplete model at 25%, and inflated pre-registration counts.
- **Significance:** 3 / 5. Pricing foresight in time inside a real engine is useful to the MoE-offloading community. The scope is narrow (one card class, two models, batch 1, AIME), and the main prescriptive results are negative.
- **Novelty:** 3 / 5. The bounds are natural applications of MIN-over-bandwidth. Choosing among hit-optimal schedules by admission cost has precedents in caching. The in-engine oracle factorial and the distinct-expert horizon are the new parts.
- **Clarity:** 2 / 5. Precise but very dense, number-heavy and appendix-dependent. Several headline numbers are hard to trace from the text alone.
- **Confidence:** 4 / 5. I recomputed about 30 claims from raw data and know the MoE-offloading and caching literature well. I could not check the 2026 preprints the paper compares against.

**Overall 5/10.** This is a rigorous and unusually reproducible measurement study: I reproduced about 30 central numbers from raw data, including the fewest-admission LP and jobs 104–105, mostly exactly. It is held back by modest novelty, a central relation established at a single model-budget cell whose only confirmatory test failed at the other budget, a fragile published-systems headline, and writing that makes the contributions hard to extract.
