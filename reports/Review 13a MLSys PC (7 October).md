# Review 13a: MLSys 2027 main track (PC member, blind)

**Submission:** "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth"
**Date of review:** 7 October 2026

## Materials read

- `paper/paper.pdf`: all 37 pages. I read the main text (Sections 1-9) closely and twice. I read Appendices A, B, C, D, E, F, J and K in full, and Appendices G, H, I and L for the claims I checked. I rendered pages 1 and 3-7 to look at the figure and table design.
- LaTeX: `paper/paper.tex`, plus `tab_regtests.tex`, `tab_job106.tex`, `tab_job107.tex` and `tab_job108.tex`. I used these to see table structure and macro provenance. I also read `paper/paper.log` for overfull boxes.
- `paper/supplement.pdf`: the scorecard conventions and the job 108 clause table.
- Artifact: I read `scripts/job106.py`, `job107.py`, `job108.py`, `panel_099.py`, `speed_limit.py`, `reanalysis.py` (the `profiled_G` and `sum_law` functions) and `sumlaw_paper.py` (the `profiles` function), and `jobs/ec2/fetch_table.py` in the gpu checkout. I used them **only for definitions**: how `B_host`, `B_c` and `B_p`, G, the time per token, the error and the validity rules are defined. I also read `gpu/vast_ledger.json`.
- Raw results in `/home/claude/gpu-branch/results/`:
  - `ec_*.jsonl` (`decode_ms`, `n_decode`, `nll_sum`, `nll_n`, config `stats=`);
  - `st_*.json` counters;
  - `concur.txt`, `g_prof.json`, `q_prof.json`, `prof_*.json`;
  - `v0.txt`, `cores.txt`, `cpu.txt`/`lscpu.txt`, `free.txt`, `gpu.csv`, `nvidia-smi-q.txt`, `order_seed.txt`;
  - `readsched_C*.txt` (job 102) and `bs1.jsonl` (job 081).

  These cover jobs 081, 093-108 and 069c.
- Job scripts `jobs/105_sumlaw@vast.sh`, `106_onlineadmit@vast.sh`, `107_newhosts@vast.sh`, `108_smallhosts@vast.sh` and their per-host stubs. I read the headers and the diffs between their committed versions. I also used `git log --format='%h %ad'` for every job script and result folder of jobs 105-108.

## Independence statement

I did not open anything under `reports/` except to write this file. I listed that folder's entry count but no names or contents. I did not open `prereg/*outcome*.md`, `research_notes/`, `apply/`, `paper/paper_v1_prereview.tex`, or any review, number-check, plan or progress-log file. I did not read commit messages; I used only hashes and dates.

All numbers in the claim table were recomputed by my own code from raw rows. The code is in the scratchpad folder `rev13a/`: `lib.py`, `regtests.py`, `classes.py`, `speeds.py` and inline scripts. It reimplements the definitions; it does not import the authors' code. Two inputs are not independently derived:
- MIN's per-token reads R⋆. I took them from the job 102 microbenchmark output and the paper's own running example, because the routing traces are not in the checkout.
- The paper's claim that Pe, Pf and others are re-rentals. I checked it against GPU UUIDs.

I modified nothing in either repository except writing this file, and I committed nothing.

---

## Summary

The paper studies batch-1 decode of MoE models whose experts mostly live in host DRAM. The setting is a llama.cpp patch, with gpt-oss-120b and Qwen3-30B-A3B on rented RTX 5090 hosts. It makes four contributions.

1. **Two time-per-token lower bounds** (Eqs. 1-2). They combine MIN-with-bypass host reads with the host's highest probed read rate. Eq. 2 adds the GPU's non-expert time in series, for on-demand systems. A replay microbenchmark reaches 86-96% of the bound's read time. The authors' cache leads FreeToken at 11 of 12 cells and runs at 2.0-4.0× stock llama.cpp. It stands at 38-54% of Eq. 1.
2. **A "no fitted constant" time relation** (Eq. 3): T = G + (M + (1−G/T)A)·S/B_host. It is used to split time into GPU compute, MIN's reads and excess reads. The relation was preregistered four times (jobs 105-108) and failed every time. The paper then draws, after the fact, a consumer/server processor split. Under that split the relation fits 37 of 39 consumer launches within 8%. Server machines read at 0.27-1.04 of the probe rate.
3. **An in-engine oracle study.** Foresight pays when spent as MIN spends it (one read, in the step: 16-51%). MIN's fewest-admission schedule beats the greedy one where the PCIe link is slow. The online carry-over, "admit less", gains a median 1.02×.
4. **A price on foresight.** Half of MIN's saving over admit-every-miss needs routing for about 0.65·C distinct experts per layer ahead, across nine models. No forecaster tested gets close.

The preregistration and artifact discipline is unusual and commendable. Every number I checked reproduces from raw data to rounding. The difficulties are in framing and readability:
- The abstract leads with the post hoc consumer-processor result. It contains one factual overstatement and one overgeneralisation.
- The registered and exploratory evidence for Eq. 3 is interleaved in a chronological, number-dense narrative.
- The most novel and useful material (Sections 5-6) is compressed.

## Strengths

1. **Exceptional transparency and auditability.**
   - Predictions were committed before the machines started. I confirmed this for jobs 105-108 in three independent ways:
     - script commit times;
     - rental start times in `vast_ledger.json`, which fall 4-5 s after each registering commit;
     - job start times reconstructed from `order_seed` (= epoch mod 10^5), all after the commits.
   - Every amendment (105f/g, 106e, 107f, 107g) was committed seconds before the host it added, and left the predictions unchanged.
   - Rentals followed the registered offer order.
   - Failures, including the failures of the paper's own central relation, are reported, and the supplement scores every clause.
   - I recomputed about 45 quantities from raw rows. With the few wording-level exceptions listed below, all match to the printed precision.
2. **The in-engine oracle study is informative and well controlled.**
   - Same engine, same routing (teacher-forced), and the oracles decide rather than predict.
   - It separates *what* to cache from *how* to load. It shows that the usual ways of spending foresight (Belady prefetch; serve on the CPU, then copy) waste reads and can lose at small budgets.
   - Host dependence on the link-to-CPU ratio is shown clearly, with round-level intervals on the slow-link machines.
3. **The fewest-admission observation is actionable.** Among hit-optimal schedules, prefer the one with the fewest admissions. It gains over the deployed cache on all 10 stable launches (1.02-1.34×), and over greedy by 0.08-0.17 where the link is slower than the CPU. Its online analogue (a larger admission margin) is cheap.
4. **The horizon result** (W50 ≈ 0.65·C distinct experts; forecasters close at most 15%) gives predictor research a concrete, falsifiable target.
5. **A competent systems baseline.** Table 2 compares against FreeToken and llama.cpp at equal GPU expert memory, with paired bootstrap intervals. A grid on 4090/3090 and a FreeToken tuning run are in the appendix.
6. **Honest Limitations section.** It states that the processor split was drawn after the tests, that Eq. 3 describes only the deployed path, and that the probe rate can be exceeded.

## Weaknesses (most important first)

### W1. The central explanatory claim (Eq. 3) has no successful out-of-sample registered test, yet the abstract leads with a post hoc, largely in-sample summary.

- Jobs 105-108 all failed as registered (Table 3; I reproduce every Within and Median entry). The abstract says this ("never passed as registered"). It then leads with: "within 8% on 37 of 39 launches" on "machines with consumer processors". That summary combines three post hoc choices:
  - (a) The consumer/server split was drawn after job 108. It is the *second* redrawing of the class, after the "one NUMA node, ≤32 cores" class registered for job 108 failed.
  - (b) The 8% band was widened from the 6% registered in jobs 105-107, to "the spread of the launches it was found on" (job 108 header).
  - (c) 33 of the 39 launches are the jobs 093-105 launches on which the overlap form was chosen. Only 6 launches came after the form was fixed, and 3 of them are re-rentals.
- **Out-of-sample on consumer machines new to a test**, the record is weaker than the abstract suggests. My recomputation:

  | Machine (job) | gpt-oss 11% | gpt-oss 25% | Qwen3 12.5% | Qwen3 25% |
  |---|---|---|---|---|
  | Ryzen 9 5900XT (107) | −7.1% | −2.3% | **−11.3%** | −6.4% |
  | Ryzen 9 9950X3D (108) | −6.9% | −4.0% | – | – |
  | Ryzen 9 5950X (108) | −6.5% | −0.9% | – | – |

  - All three gpt-oss 11% cells are *under*-predicted beyond the originally registered 6%.
  - The only new consumer machine run on Qwen3 misses by 11.3% and 6.4%.
  - In job 108, the two desktops **alone** fail two of the three registered pooled clauses: 2 of 4 cells beyond 6% (registered: at most one), and a median |error| of 5.3% (registered: at most 4%).

  So the evidence supports "Eq. 3 describes our engine in-sample on consumer desktops, and under-predicts new ones by 1-7% at gpt-oss (11% at Qwen3)". It does not support a validated relation.
- The overlap term itself is not identified. Discounting admissions by half passes 11 of 12 job 106 cells, and misses-only passes 10 of 12. On job 107's desktop the plain form beats the registered overlap form (6.0% vs 6.8% median). The paper reports all of this.

### W2. Specific inaccuracies and overgeneralisations in the abstract, contributions, Table 3 and Conclusion.

- **Abstract.** "...including every cell of the three such machines first rented for a registered test" is false as written. The 5900XT's Qwen3 12.5% cell is −11.3% and its Qwen3 25% cell is −6.4%. The Introduction states the correct scope ("every gpt-oss cell").
- **Abstract.** "On server processors the engine reads at 0.27-1.04 of that rate, and the relation fails." The EPYC 7402P (job 105), a server part by the paper's own definition, fits at +3.0% and +4.0%. The relation fails on 2 of the 3 valid server machines, not on server processors as such.
- **Table 3, row 108** attributes the failure to "(8% band; EPYC)". As noted in W1, the test also fails without the EPYC host. The main text says only that the desktops were "within 8% at all four cells".
- **Contributions and Introduction.** "found on 34 launches ... registered before four later jobs" conflates two equations. The 34 launches include job 105, and job 105 registered the *plain* form, which was found on 30 launches. "Failed whenever a test included machines new to it" is also misleading: job 105 failed on a re-rented machine (O4 again, +6.9% at 25%) because of the form, not the machine.
- **Conclusion.** "batch-1 MoE decode on machines with consumer processors is close to the GPU's compute plus host bytes" generalises from one engine's *deployed* path to MoE decode in general. Section 4 itself shows that states copying in the step depart from Eq. 3 (implied G of 7.4 and 16.4 ms).

### W3. The "lower bounds" are bounds relative to a probe statistic, not to the machine.

- B_host is defined as the maximum over all probe readings (CPU, link, and concurrent; `speed_limit.host_rates`). On the AMD engineering sample, that maximum (204 GB/s) is a single concurrent reading (CPU 181.4 + PCIe 23.0). It is inconsistent with the CPU-only peak of 151 GB/s.
- By Eq. 3's own accounting, the engine read more than 5% faster than B_host on 13 of 68 launch-budgets, up to 1.22× (I reproduce 13/68 and 1.223). Either Eq. 3's accounting is wrong there, or B_host is not an upper limit. In the second case Eq. 1 is not a lower bound "for any system" on those machines.
- The paper acknowledges this in Limitations. The main text and abstract nevertheless call these "lower bounds" without qualification. The GPU term uses datasheet bandwidth, which is loose (batch-1 decode reaches 52% of it).
- The bound is still a useful yardstick. It should be named and reported as probe-relative, with a sensitivity analysis (CPU-only peak; median of concurrent readings).

### W4. Novelty and significance are concentrated in the least developed parts.

- Eq. 1 is a roofline with MIN-with-bypass reads. Trace-level MIN-with-bypass gaps and oracle caches have been reported (the related work cites Zhang 2026b and Liang et al. 2026b).
- Eq. 3 is "compute + bytes/bandwidth" with a post hoc overlap factor.
- The fewest-admission schedule is, as the authors say, the interval/min-cost-flow formulation (Berger et al. 2018) with admissions as a second objective, measured in an engine.
- The genuinely new and useful results take about a page and a half in Sections 5-6, with the detail pushed to Appendices G-J. These are the in-engine value of foresight by how it is spent, the host-dependence crossover and the distinct-expert horizon.
- The online payoff is small (median 1.02×). Its registered thresholds missed: ≥1.01 failed on 2 of the 3 stable job 106 hosts, and ≥0.995 failed at one job 107 cell.
- A reader comes away knowing a great deal about the test history of Eq. 3, and comparatively little about how to build a better cache.

### W5. Clarity remains a serious barrier (score 2). See the dedicated section below.

### W6. Statistical scope and generalisation.

- Table 2 and Table 4 intervals are within-launch, over problems or rounds. The paper's own evidence says machines dominate variance:
  - same-model machines differ by up to about 30% (I find 1.6-31% across same-model pairs);
  - the Table 2 rerun on a second host (job 089, Appendix B) gave ratios 0.06-0.11 lower at 5 of 6 cells.

  Headline ratios such as "1.294 [1.278, 1.312]" therefore carry false precision as a statement about "the system". Report machine-level uncertainty, for example over the grid hosts.
- Counts are launch-weighted: 39 launches but 25 machines, and Pf alone contributes 6 consumer launches. Report the evidence per machine.
- The workload is narrow: AIME-25 only (also used during development), batch 1, two models, teacher-forced, and the cache bypasses prompts.

### W7. Validity exclusions in job 106 were not preregistered.

- Validity gates appear only from job 107. In job 106, the two server hosts were excluded post hoc: one had wrong outputs, with NLL 0.208-0.446 against 0.19; the other's rounds were 14.5, 22.1 and 21.9 ms apart.
- The reasons are compelling, and Table 3 keeps them in the registered pooled median, which is correct. But the main text's success claim for job 106 ("within 6% on all 12 cells") rests entirely on the three re-rented machines.
- Two definitional inconsistencies:
  - The 106d spread is quoted as "39%", which is (max−min)/mean. The V2 rule used for 107c, 107e and 108b is max/min−1, which gives 52% for 106d.
  - The fallback G for job 108 is the median of 8 profiles. These include the EPYC 7402P server profile, labelled "desktop-class" by the earlier class rule. Immaterial (G-sensitivity 26.7-35.1%), but it should be stated.

### W8. Smaller accuracy issues.

- **Table 19 caption.** Says "against Eq. (3)". The "law" column is the *plain* form, as registered (Pf 11%: 13.65 ms plain vs 13.32 ms overlap; the table prints 13.6).
- **Table 22 caption.** Says G was "profiled on the same machine before the timed runs". This is untrue for the EPYC 7543 and EPYC 7K62 rows, which use the fallback; the rows are not flagged.
- **Section 6.** "Loses elsewhere, down to 0.72×" for the layer-ahead copy. Table 19 shows 0.69 (Pf, 25%).
- **"Exceeds Eq. (3) by at most 11% of it"** (Section 4) is ambiguous. The worst consumer launch is 10.6% of the measured time, or 11.9% of Eq. 3's value.
- **Setting:** "two hosts with the same CPU model differ by 16-29%". The set of pairs is unspecified. Over all same-model pairs in jobs 093-108 I find 1.6% (9800X3D) to 31% (285K).
- **Typesetting.** Table 3 overflows its column by 81 pt and its Outcome column is clipped in the PDF. The Appendix L audit table overflows by 70 pt. hyperref reports duplicate destinations (table.1, table.24).
- **Anonymity (note to the chairs).** The PDF is anonymous. The LaTeX source (`\mlsysauthor`) and the artifact repository are not.

---

## Checks requested by the chairs

### Registered tests of jobs 105-108 (Table 3) against script headers and git history

| Job | Registered (header) | Committed → first rental → reconstructed start | My recomputation | Paper | Verdict |
|---|---|---|---|---|---|
| 105 | Plain form within 6% at every cell; median ≤4%; G 4.0-5.0 ms | 17:05:31 → 17:05:36 → 17:06; amendment 17:09:26 before 105f (17:10) and 105g (17:09:57) | 5/8 within; median 4.24%. 11%: −1.5 to +4.7%. 25%: +7.0 (TR), +6.9 (O4 again), +9.3 (EPYC), +3.8 (Pf). G 4.07-4.50 | 5/8, 4.2%, "failed at gpt-oss 25%" | Correct. One failing cell is a re-rented machine (O4), so Table 3's "Machines new to the test" framing does not explain this failure. |
| 106 | Overlap form within 6% at every host/model/budget; median ≤4%; at 25%, median ≤ half the plain form's; rounds ≤2% | 20:43:48 → 20:43:58 → 20:45; amendment 20:44:13 before 106e (rental 20:44:16) | All hosts: 12/20, median 4.67%. Stable 3: 12/12, median 2.34%, max 5.99%. Plain up to 10.08%. Servers −10.2 to −52.4% | 12/20, 4.7%, "failed on new machines" | Correct. Server exclusion from the "stable" claim is post hoc (W7). 106e is panel host Pe (same GPU UUID as 099e), although it was registered as a new 9950X offer. |
| 107 | Overlap within 6% everywhere; median ≤4% and below plain; ≥3 valid hosts, else inconclusive; gates V0/V1/V2 | 00:17:31 → 00:17:44 → 00:18; amendments 01:36:57 / 01:41:16, before 107f (01:37:00) and 107g (01:41:20); predictions unchanged | Valid (5900XT, ES): 1/8 within, median 31.75%; 8/8 under-predicted; plain 2/8 within. Desktop: plain 5.99% vs overlap 6.78% | 1/8, 31.7%, "failed; too few valid" | Correct. By its registered rule the test is *inconclusive*; calling it failed is conservative. Replacing round-check failures was an in-job amendment, disclosed. |
| 108 | Within 8% at every cell; ≤1 cell beyond 6%; median ≤4%; implied rate 0.85-1.25; G 4.0-5.0; class gate V0c (1 NUMA node, ≤32 cores) | 02:39:32 → 02:39:36 → 02:40; offers rented in registered order (ledger) | With fallback G: 2/6 within 6%, 4/6 within 8%, median 6.70%. **Without the EPYC 7543: 2/4 beyond 6%, median 5.25%, so two pooled clauses still fail** | 2/6, 6.7%, "failed (8% band; EPYC)" | Numbers correct. The failure attribution is incomplete (W2). |

### How job 108's EPYC host is handled

- **The failure is real.** 108a's `prof_G14` captured 3 kernels (busy 0.005 ms, G = 0.0015 ms), and `prof_G32.json` was never written. 108b shows the same pattern. Driver 570.x appears only on these two hosts; every host whose profile succeeded ran 580 or newer. The "older driver" explanation is consistent with the data, though not proven.
- **The main text handles it transparently and conservatively.**
  - It names the deviation from the registration.
  - It substitutes the median of 8 earlier profiles. I reproduce 4.292 and 4.469 ms, which round to the printed 4.29 and 4.47.
  - It gives the sensitivity: I reproduce 26.7-35.2%.
  - It keeps the host as valid and failing, which works against the authors.
- **The registered protocol had no rule for a failed profile.** Treating the host as untestable would leave two valid machines (inconclusive by the 3-host rule), and the two pooled clauses would still fail. The verdict therefore does not depend on how the EPYC is handled. The paper should say so explicitly.
- **Two presentation defects remain.** Table 22's caption misstates G's provenance for both EPYC rows. Table 3 attributes the failure to the EPYC alone.

### Is the post hoc consumer/server split presented as post hoc?

- **In the body, yes.**
  - Section 4's paragraph is headed "What does separate them, after the fact".
  - Limitations says the split "was drawn after the tests, not before them".
  - `job108.py` labels the computation "after the fact (not registered)".
- **Elsewhere, no.**
  - The abstract's lead result.
  - The Introduction ("On consumer processors, then, the gap to the bound has two parts").
  - Figure 1's caption and grouping.
  - The Conclusion.
- **Section 2 (Setting) defines "consumer" and "server" processors as standing terms.** A reader meets the split as if it were a design stratification before learning it was drawn after four failed tests.
- **The split is by CPU product name.** It is confounded with core count and helper threads (126 helpers on the ES, where the probe itself reads 118 GB/s at 128 threads against 151 at 16), NUMA layout, container CPU limits, and driver version. The data cannot attribute the shortfall to "server processors" as such.
- **Recomputed numbers** (all match the paper):
  - 39 consumer launches on 25 machines (GPU UUIDs); implied rate 0.880-1.164; 37 within 8% and 31 within 6%.
  - 8 server launches, 5 invalid; valid ones read at 0.269, 0.585 and 1.038 of B_host.

### Are claims about Eq. 3 scoped to the evidence?

- **Section 4 scopes it well:** "For the deployed cache on the machines where we found it", "The relation describes our engine on consumer processors; it is not a property of every machine", and the deployed path versus in-step states.
- **Over-scoped in four places:**
  - the abstract's "every cell" (false for Qwen3);
  - the abstract's "on server processors ... the relation fails" (false for the EPYC 7402P);
  - the Conclusion's "batch-1 MoE decode" (one engine, one path);
  - Contributions bullet 2's "held on machines rented again ... failed whenever a test included machines new to it" (job 105 failed on a re-rented machine).
- **The abstract does not say** that the 8% band was widened after the 6% tests failed.

---

## Clarity: specific problems and fixes

Earlier rounds scored clarity 2/5, and I agree with that score. The paper is careful, but reading it is like auditing a lab notebook. Specific problems and what a clearer version would do:

1. **The abstract is a results log.**
   - *Problem:* It has 11 sentences, about 10 distinct numbers, and at least 6 terms that are undefined at that point ("MIN with bypass", "launch", "cell", "admissions", "serialisation", "registered").
   - *Problem:* About 40% of it narrates the fate of Eq. 3, while the system result (Table 2) is absent. It mixes a post hoc stratum, a widened band and in-sample launches, and contains a false statement (W2).
   - *Fix:* Six sentences: problem → bound → where systems stand → where the time goes (one number) → how foresight must be spent (one number) → how much foresight is needed (one number). Then one honest sentence on Eq. 3, for example: "a compute-plus-bytes model with no fitted constant describes our engine on consumer desktops in-sample, but failed all four of our preregistered out-of-sample tests."

2. **Section 4 is structured chronologically.**
   - *Problem:* The paragraph heads are "Found, then registered", "Registered on new machines, it failed", "Registered on a class drawn after the fact, it failed again" and "What does separate them, after the fact". To follow them, the reader must track two equation forms (plain/overlap), two bands (6%/8%), two class rules, validity gates, re-rented versus new machines, and job numbers 093-108.
   - *Fix:* 4.1 the model, with a derivation sketch and an illustration; 4.2 preregistered tests, as one table (registered criterion, machines, per-cell errors, verdict, verdict without invalid or fallback hosts); 4.3 exploratory analysis, labelled as such (processor class, implied read rate, elasticity); 4.4 decomposition (Fig. 1).

3. **Registered and exploratory results are interleaved inside sentences.**
   - *Problem:* Example from Section 5, "The lesson, carried online": exploratory speed-ups, four registered thresholds (1.01, 1.02, 0.995, 0.03), a post hoc null ("predicting no change does almost as well") and a second rule (lrn) all appear in one paragraph.
   - *Fix:* Never mix the two kinds in one sentence. Put every registered verdict in a table, and mark registered statements typographically (e.g., a "[R]" tag or a shaded box).

4. **Number density.**
   - *Problem:* Sections 4-5 average 4-5 numbers per sentence, and 14 sentences in Section 4 carry six or more. Example from Section 4: "It failed nonetheless: the relation under-predicted the time at 8 of 8 cells and missed the 6% band at 7, by up to 11% on a Ryzen 9 5900XT (1 of 4 cells within) and by 52-59% on a dual-socket AMD engineering sample whose outputs matched every other host's and whose rounds were 0.3% apart; with any errors at all on a third machine's 4 cells, the registered pooled median could not have come below 6.8%." That is 9 numbers, three clauses and a counterfactual.
   - *Problem:* Ranges such as "1.02-1.34× ... by 0.08-0.17 ... at 0.28-0.63 of their CPU rate" cannot be mapped to machines.
   - *Fix:* At most two or three numbers per sentence, each the one that carries the argument. Put the ranges into per-machine tables or forest plots.

5. **Internal identifiers in the main text.**
   - *Problem:* Job numbers (069c, 089, 093-108) appear dozens of times in the main text.
   - *Problem:* Host codes are mostly undefined there. "Pf" and "Pe" first appear in Section 4 without definition, "Pg" appears only in appendices, and O3-O5 are defined only as "three hosts". Host B, host S and host A are defined inside table headers.
   - *Fix:* Add a host table in Setting (code, CPU, cores, link/CPU ratio, B_host, which experiments). In prose, refer to hosts by the property that matters (e.g., "a slow-link desktop") and to experiments by purpose, not job number.

6. **Terminology drift.**
   - *Problem:* Eq. 3 is called "relation", "law" (the column header in Tables 4 and 19-21) and "time relation". Eq. 4 is the "calibrated model". "Bound" alternates with "limit" (Table 7, Fig. 5).
   - *Problem:* Configurations carry several names each: "MIN, 1 read" = "MIN copied in the step" = fetch; "MIN, 2 reads" = "loaded by the CPU" = bypass; "fewest-admission set" = fetchplan/bypassplan; dk and lrn appear only in Table 4.
   - *Problem:* Units of analysis are "job", "launch", "round", "cell" and "launch-budget".
   - *Fix:* One glossary (a `tab_glossary.tex` exists in the source but is not used in the main text). Use one name per object, and use Table 1's names in every table header and figure legend.

7. **The Setting front-loads about a dozen definitions in one "Terms" paragraph,** without a picture.
   - *Problem:* The paragraph defines MIN, admit, greedy, fewest-admission, in the step, by the CPU, oracle, host-bound, B_c, B_p, B_host, the link-to-CPU ratio and consumer/server.
   - *Fix:* Add one timeline figure of a decode step: router → hit / CPU miss / PCIe fetch → background copy, with where G and the reads sit. Equations 2-3, "serialisation" and "two reads" would then be self-evident, and the paragraph could shrink by half.

8. **Table 2 is overloaded.**
   - *Problem:* It has 10 data columns, two variants of the bound, host specifications in row-group headers, and a 9-line caption. The caption contains results (the pooled-bound percentages, the all-in-VRAM speeds) and a footnote about selection.
   - *Fix:* Split it into a speed table (llama.cpp / FreeToken / ours / ratio) and a separate "fraction of bound" table or figure. Move the results out of the caption.

9. **Table 3, the most important table, is clipped and incomplete.**
   - *Problem:* It overflows the column by 81 pt, so the PDF shows "failed on new machine" and "failed (8% band; EPY".
   - *Problem:* Its "Machines new to the test" column implies that new machines caused each failure. That is not true for job 105.
   - *Problem:* The registered criteria differ across rows, but are stated only in the caption.
   - *Fix:* Fit it to the column, and add "registered criterion" and "verdict excluding invalid or fallback hosts" columns.

10. **Table 4 combines four analyses with different interval constructions.**
    - *Problem:* It holds the Eq. 3 test, two online rules and the fewest-admission oracle. Intervals are over rounds for gpt-oss and over problems for Qwen3, across 9 columns, under a column header ("law") the text never uses.
    - *Fix:* Two small tables, or a figure for the speed ratios.

11. **Figure 1.**
    - *Problem:* Its 28 rows are labelled by CPU model with duplicates ("R9 9950X" four times, "U9 285K" three times).
    - *Problem:* Consumer and server machines are interleaved.
    - *Problem:* It uses one pooled-median G per budget, so its Eq. 3 values disagree with the per-host G of Tables 4 and 19-22.
    - *Fix:* Group the rows by class with unique labels, use per-host G where it was profiled, and consider plotting residuals (measured − Eq. 3) next to the stacked bars.

12. **Figures 2-3.**
    - *Problem:* Figure 2 is a 6-panel × 7-configuration dot plot with tiny markers. It omits the fewest-admission schedule that Section 5 is about.
    - *Problem:* Figure 3 overlays 7 series with similar markers and fitted lines.
    - *Fix:* In the main text show only the host-bound budgets and 3-4 configurations, including fewest-admission. Move the rest to an appendix.

13. **Caveats are hidden in parentheses and captions.**
    - *Problem:* For example: "(Its MIN runs over the problems in sequence and may evict an expert in the step that served it; ...)". Table 3's caption "(job 105: its plain form)" is the only place the main text says job 105 tested a different equation.
    - *Fix:* Promote caveats that change the interpretation into sentences in the text.

14. **The main argument depends on appendix tables.**
    - *Problem:* The two decisive tests (jobs 107 and 108) are shown only in Tables 21-22, and the bound's robustness only in Tables 23-24.
    - *Fix:* Bring the per-machine numbers of the registered tests into the main-text table (see item 9).

15. **Appendix B is unreadable.**
    - *Problem:* It is a single paragraph of about 1,500 words enumerating clause outcomes across 30 jobs.
    - *Fix:* A table (job | claim | registered threshold | measured | verdict), plus a five-sentence summary.

16. **Section order and framing.**
    - *Problem:* The system being measured (the cache, the per-machine fetch table and its calibrated model) is described in about 10 lines of Setting and in Appendices C and K. Table 2's system comparison sits inside "Two Bounds".
    - *Fix:* Either add a short "System" section before the bounds, or frame the paper explicitly as an analysis paper and give the comparison its own "Where systems stand" subsection. Align the abstract with the title's two questions (bounding; what foresight is worth).

17. **Caption inaccuracies** (Table 19 "Eq. (3)" for the plain form; Table 22 "G profiled on the same machine") add confusion on exactly the points where readers need precision.

---

## Claims checked (recomputed from raw rows)

Legend: ✓ matches to the printed precision; ~ matches with a caveat; ✗ does not match as stated.

| # | Claim | Location | My value (raw data) | Verdict |
|---|---|---|---|---|
| 1 | Job 105 (plain form): 5/8 cells within 6%, median 4.2%, failed at gpt-oss 25% | Table 3; App. B | 5/8; 4.24%; 11%: −1.5 to +4.7%; 25%: +7.0, +6.9 (O4 again), +9.3, +3.8 (Pf) | ✓ (one failure is on a re-rented machine) |
| 2 | Job 106: 12/20 within, median 4.7% | Table 3 | 12/20; 4.67% | ✓ |
| 3 | Job 107: 1/8 within, median 31.7% | Table 3 | 1/8; 31.75% | ✓ |
| 4 | Job 108: 2/6 within 6%, 4/6 within 8%, median 6.7% | Table 3 caption, §4 | 2/6; 4/6; 6.70% (fallback G) | ✓ |
| 5 | Job 106, 3 stable hosts: 12/12 within 6%, median 2.3%, largest 6.0%; plain form misses by up to 10.1% | §4 | 12/12; 2.34%; 5.99%; 10.08% | ✓ |
| 6 | Half-discount passes 11/12; misses-only 10/12 | §4 | 11/12; 10/12 | ✓ |
| 7 | Job 106 servers under-predicted by 10-52% | §4 | −10.2% to −52.4% | ✓ |
| 8 | Wrong-output host: loss 0.21-0.45 nats vs 0.19 / 0.085 | §4, App. J | 0.208-0.446; others 0.189-0.190 / 0.0854-0.0856 | ✓ |
| 9 | Xeon host "varied by 39% between rounds" | §4, Limitations | Rounds 14.52 / 22.09 / 21.93 ms: (max−min)/mean = 38.8%; max/min−1 (the V2 rule) = 52.2% | ~ (inconsistent definition) |
| 10 | Job 107: under-predicted 8/8, 7 beyond 6%; 5900XT up to 11% (1/4 within); ES 52-59%; rounds 0.3% apart | §4 | 8/8; 7/8; −2.3 to −11.3%; −52.2 to −58.9%; 0.28% | ✓ |
| 11 | Job 107 desktop: plain median 6.0% (2 within) vs overlap 6.8% (1) | §4 | 5.99% / 2 vs 6.78% / 1 | ✓ |
| 12 | Pooled median could not have come below 6.8% with a third machine | §4 | 6.78% | ✓ |
| 13 | ES: B_host 204 GB/s is one CPU+link reading; CPU-only 151 @16 threads; 118 @128; still misses by 51% | §4 | 204.5 (concurrent t=128: CPU 181.4 + PCIe 23.0); 151.4; 117.6; −51.4% | ✓ |
| 14 | Job 108 desktops off by 6.9, 4.0, 6.5 and 0.9% | §4 | −6.94, −4.05, −6.46, −0.95% | ✓ |
| 15 | EPYC 7543: −34.2 / −27.9%; reads at 0.58; any G in range gives 26.7-35.1% | §4 | −34.23 / −27.91%; 0.584; 26.7-35.2% | ✓ |
| 16 | Fallback G = median of 8 earlier profiles (4.29 / 4.47 ms) | App. J | 4.292 / 4.469 (the 8 include the EPYC 7402P profile) | ✓ |
| 17 | EPYC trace "captured no decode kernels under an older driver" | §4, App. J | 3 kernels (busy 0.005 ms); G32 profile missing; driver 570.x only on the two failed hosts | ✓ |
| 18 | Job 108: 6 rented, 2 gated (GPU rented before; 323 GB in use), 1 with rounds 7.0% apart | §4 | Ledger plus v0.txt confirm, in registered order; 7.01%; 108b under-predicted by 26.8% | ✓ |
| 19 | Consumer: 39 valid launches on 25 machines; implied rate 0.88-1.16; within 8% on 37, 6% on 31 | §4, abstract | 39 / 25 (UUIDs); 0.880-1.164; 37; 31 (33 of 39 in-sample) | ✓ |
| 20 | Server: 5 of 8 launches invalid; valid read at 0.27, 0.58, 1.04 | §4, abstract | 0.269, 0.585, 1.038 | ✓ |
| 21 | "Within 8% ... including every cell of the three such machines first rented for a registered test" | Abstract | 5900XT Qwen3 12.5%: −11.3%; Qwen3 25%: −6.4% | ✗ (true for gpt-oss cells only) |
| 22 | "On server processors ... the relation fails" | Abstract | EPYC 7402P: +3.0% (11%), +4.0% (25%) | ✗ (overgeneralised) |
| 23 | Plain form on 30 launches: 28 within 6% at 11%, median +6% at 25%; overlap on 34: 33 vs 18 (25%), 29 vs 32 (11%) | §4 | 28/30; +5.9%; 33 vs 18; 29 vs 32 | ✓ |
| 24 | Elasticity of (T−G) to B_host −0.95 (−1.06 to −0.84), 21 machines | §4 | −0.950 (−1.061, −0.843) | ✓ |
| 25 | Engine read >5% faster than probe max on 13 of 68 launch-budgets, up to 1.22× | Limitations | 13/68 (pooled-median G); 1.223 | ✓ |
| 26 | Table 4: all 12 rows (G, law, measured, dk, lrn, fewest) | Table 4 | All within rounding (e.g., Pf 11%: 13.81 / 13.68; dk 1.024; lrn 1.028; fewest 1.019) | ✓ |
| 27 | dk cuts reads 4-6% (11%) and 6-10% (25%); speed 1.00-1.02× and 1.01-1.06×; Qwen3 1.01-1.04×; ≥1.01 missed on 2 of 3 hosts; stable pooled median 1.01 | §5 | −4.4 to −5.8%; −6.3 to −10.5%; 1.003-1.024; 1.008-1.057; 1.012-1.038; TR 1.003, Pe 1.010 / 1.008; 1.0145 | ✓ |
| 28 | lrn costs 267-439 µs per token; runs 0.98-1.05× | §5 | 267-439 µs; 0.982-1.055 | ✓ |
| 29 | Job 107 dk 0.987-1.074 (median 1.021); abstract: up to 6%, median 1.02× on 5 machines, lost 1% at 1 of 10 cells | §5, abstract | 0.987-1.074, 1.021; over the 10 gpt-oss cells median 1.019, max 1.057, min 0.987 | ✓ |
| 30 | Fewest-admission: 10 launches / 7 machines, 1.02-1.34×; +0.08-0.17 over greedy; slow-link greedy 0.85-0.93, fewest 1.02-1.06; round-CI lower ends 1.01 and 1.05; loaded by CPU 1.11-1.22; 13 GB/s server 0.82-0.88 per round; job 107 1.16-1.42 (in step) and 1.04-1.13 (by CPU) | §5 | 1.019-1.342; 0.082-0.172; 0.853-0.933; 1.019-1.056; 1.013 / 1.048; 1.112-1.217; 0.824-0.878; 1.157-1.419 and 1.038-1.126 | ✓ |
| 31 | Fewest-admission copies 0.60-0.73 of greedy's; misses 2.5-5.3% more | §5, Limitations | 0.598-0.725; +2.5 to +5.2% | ✓ |
| 32 | Layer-ahead copy: 1.04× where link ≈ CPU, "down to 0.72×" | §6 | 1.042; minimum 0.691 (Pf, 25%; Table 19 prints 0.69) | ~ |
| 33 | Table 2, host B: ours/FreeToken 1.294 [1.278, 1.312], 1.275, 1.154, 1.032; tok/s columns | Table 2 | 1.294 [1.278, 1.312]; 1.275; 1.154; 1.032; tok/s match (69.9 / 54.0 / 34.9, ...) | ✓ |
| 34 | Running example O4: deployed 20.6, MIN in step 15.2, MIN prefetched 13.8 ms | §5 | 20.64 / 15.16 / 13.81 ms (096a) | ✓ |
| 35 | MIN, 1 read: 16-51% faster on O3-O5 at six budgets; prefetched 25-81% | §5 | 1.16-1.51; 1.25-1.81 | ✓ |
| 36 | Read-schedule microbenchmark reaches 86-96% of bound read time | §3 | 0.861-0.963 | ✓ |
| 37 | Cache at 38-54% of Eq. 1 and 55-70% of Eq. 2 (21 machines, jobs 093-104) | §3 | 38-54% and 55-71%, using R⋆ ≈ 38.3-38.5 from job 102 / the running example | ~ (R⋆ not recomputed from traces) |
| 38 | Fig. 1 shares at 11%: G 12-47%, MIN reads 31-54%, beyond-MIN 22-35%; G is 23-68% of the gap | §4 | 12-45%, 31-54%, 22-35%, 23-66% (own G where profiled) | ✓ (approximately) |
| 39 | T_GPU 2.9-3.4 ms on 7 hosts | §3 | 2.94-3.38 ms | ✓ |
| 40 | G 4.1-4.7 ms (gpt-oss), 5.4-5.9 ms (Qwen3), job 106 stable hosts | §4 | 4.15-4.65; 5.42-5.93 | ✓ |
| 41 | Re-rented machines within 1.9% (job 101) and 3.2% (job 106) | §2 | 1.7%; 3.1% | ✓ |
| 42 | Same CPU model: 16-29% apart | §2 | 1.6-31% across all same-model pairs | ~ (pairs unspecified) |
| 43 | Predictions committed before each machine started | App. B | Rentals 4-5 s after commits; amendments 3-4 s before the hosts they add; starts reconstructed from order seeds all after commits | ✓ |
| 44 | Table 19 compares against "Eq. (3)" | Table 19 caption | Values are the plain form (Pf 11%: plain 13.65 vs overlap 13.32) | ✗ (caption) |
| 45 | Table 22: "G profiled on the same machine before the timed runs" | Table 22 caption | EPYC 7543 and 7K62 rows use the fallback G | ✗ (caption) |

**Exploratory observation of my own** (not a claim in the paper). The engine's own counters record per-step host-side time (`host_us_per_step`: app + pre + inputs + launch + post). It is 0.85-1.9 ms on consumer launches and 2.1-2.8 ms on the three valid server launches. That includes the EPYC 7402P (2.1 ms), which fits Eq. 3, so host overhead alone cannot be the server story. Over the 39 consumer launches at gpt-oss 11%, the Eq. 3 residual (measured − predicted, in ms) correlates with it at r ≈ 0.43. A per-token host-overhead term outside G may explain part of the systematic under-prediction on new machines (see Q5).

## Questions for the authors

1. Can you report Eq. 3's out-of-sample record separately from the in-sample one, per machine? That means the launches after the form was fixed, on machines new to a test, at the registered 6% band. Why does the abstract use 8% and pool the 33 in-sample launches?
2. Do you agree that the abstract's "every cell of the three such machines" is contradicted by the 5900XT's Qwen3 cells (−11.3%, −6.4%)? And that "on server processors ... the relation fails" is contradicted by the EPYC 7402P (+3.0%, +4.0%)?
3. Do you agree that job 108 fails two registered pooled clauses even without the EPYC 7543 (2 of 4 cells beyond 6%; median 5.3%)? If so, will Table 3 and Section 4 say so?
4. On the ES and the EPYC 7543, did you run the engine with about 16 helper threads, where the probe reads its peak? The probe falls from 151 GB/s at 16 threads to 118 at 128, and the engine used 126 helpers. This is a cheap experiment that could turn "server processors" into a mechanism.
5. Does a per-token host-overhead term (the engine's own `host_us_per_step`) reduce the systematic under-prediction on new consumer machines, and the server shortfall? In my reanalysis the residual correlates with it (r ≈ 0.43).
6. Why is B_host defined as the maximum over all probe readings, rather than as one defined measurement? On how many machines is it a concurrent reading that exceeds the CPU-only peak? How do the bound fractions in Table 2 and Section 3 change with the CPU-only peak or the median concurrent reading?
7. On the 13/68 launch-budgets where the engine "reads" above B_host, which is wrong: Eq. 3's accounting or B_host as a ceiling? If the latter, in what sense is Eq. 1 a lower bound there?
8. For Table 2, can you give machine-level uncertainty for the ours/FreeToken ratio, for example over the four grid hosts plus host S? Job 089 found ratios 0.06-0.11 lower on a second host.
9. Can part of the fewest-admission gain be realised with a bounded lookahead? What horizon does it need, compared with the 0.65·C distinct experts needed by MIN's greedy schedule?
10. The workload is AIME only, which was also used in development. Do the online margin (dk) and the horizon rule hold on a different decode workload, such as code or chat?
11. Applying job 107's gates retroactively to job 106 would exclude 106c (V1) and 106d (V2). Was anything else in jobs 093-106 excluded or rerun on similar grounds? (084 → 084b/c is disclosed.)

## What would raise my score

- **Fix the abstract, Table 3 and the captions** for the inaccuracies in W2 and W8. State the out-of-sample record of Eq. 3 plainly.
- **Restructure for clarity** along the lines above: an abstract with one number per idea; Section 4 split into model / registered tests / exploratory; no job numbers or undefined host codes in the main text; one name per object; fixed tables and figures.
- **Reframe Eq. 3** as an exploratory decomposition tool, with an explicit, honest validation record. Alternatively, run one more preregistered test with the consumer class fixed in advance, ≥3 new machines and both models, at the 6% band, ideally with a mechanism-level server test (helper count).
- **Make the bound's dependence on B_host explicit** ("probe-relative bound"), with a sensitivity table, and fix the definition of B_host.
- **Shift space to Sections 5-6** (how foresight must be spent; host dependence; horizon), which carry most of the novel, reusable insight. Add a realisable policy that captures part of the fewest-admission gain beyond the 2% of dk.
- **Report machine-level uncertainty** for the headline comparisons.

## Scores

| Criterion | Score |
|---|---|
| Overall (1-10; 6 = weak accept, 8 = strong accept) | **5**: borderline, leaning reject in the current form. The data and artifact are excellent, but the headline framing overreaches the registered evidence and the paper is very hard to read. With a clarity rewrite and corrected framing I would expect to move to 6-7. |
| Soundness (1-5) | **3**: measurements, bookkeeping and preregistration are sound and reproducible. Interpretive claims about Eq. 3 and the "lower bounds" go beyond what the tests support. |
| Significance (1-5) | **3** |
| Novelty (1-5) | **3** |
| Clarity (1-5) | **2** |
| Confidence (1-5) | **4**: I recomputed about 45 claims from raw data and checked the git and ledger timeline. I did not recompute MIN from routing traces or the nine-model horizon study. |
