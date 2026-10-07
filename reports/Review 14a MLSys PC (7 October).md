# Review 14a — MLSys 2027 (main track), PC member, blind

**Submission:** "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth"
**Date:** 7 October 2026

## Materials read

- `paper/paper.pdf`, all 37 pages: the main text (pp. 1–9), references, and Appendices A–M. I read the main text in full. I read Appendices B, C, G, J and K in full and skimmed D, E, F, H, I, L and M.
- `paper/paper.tex`, main-text source only, to map macros to numbers. I did not read `paper_v1_prereview.tex` or `paper_v2_prerewrite.tex`.
- `paper/supplement.pdf`: the scorecard structure and the per-job tallies for jobs 073–081 and 108.
- Artifact repository:
  - `gpu/vast_ledger.json`: rental start times.
  - `prereg/reanalysis.json`: the median G used where no profile exists.
  - `prereg/scorecard_106.json`.
  - `prereg/forecaster_linear.json`.
  - Scripts, read only for definitions: `scripts/job106.py` (`law`, `load`), `scripts/job107.py` (`machine_classes`), `scripts/job108.py`, `scripts/panel_099.py` (`host_info`, `cell_data`, `label_of`), `scripts/speed_limit.py` (`host_rates`), part of `scripts/sumlaw_paper.py`, and `gpu-branch/jobs/ec2/fetch_table.py` (`bandwidths`).
- Raw-results checkout (`/home/claude/gpu-branch`):
  - Job-script headers for 095, 100, 102 and 105–108.
  - `results/<job>/` for jobs 081, 089, 090, 093–108: `ec_*.jsonl`, `st_*.json`, `concur.txt`, `cores.txt`, `cpu.txt`, `lscpu.txt`, `numa.txt`, `g_prof.json`, `prof_*_tail.csv.gz`, `prof_*_kern_sum.csv.gz`, `validity.txt`, `v0.txt`, `readsched_C*.txt`, `rates.txt`, `bs1.jsonl` and `summary.txt`.
- Git history of `jobs/*.sh` and `results/*`. I used only `git log --format='%h %ad'` / `'%h %ct'` and `git diff` of file contents.

All my analysis code is in `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/rev14a/`. The files are `rv.py`, `eq3_tests.py`, `table4.py`, `launches.py`, `eff32.py`, `decomp.py`, `frac.py`, `oracles.py`, `minadm.py`, `spearman.py`, `varcomp.py`, `job100.py`, `panel.py`, `readsched.py`, `tab3.py`/`tab3b.py`, `timing.py` and `timing_all.py`. The code parses the raw rows, counters and probes itself. It re-implements Eq. (3) as the exact larger root of its quadratic, and the probe rates (B_host, B_c, B_p) from the definitions in the scripts. It does not import the authors' analysis code.

## Independence statement

- I did not open anything under `reports/` except to write this file.
- I did not open `prereg/*outcome*.md`, `research_notes/`, `apply/`, `paper/paper_v1_prereview.tex`, or any file named like a review, number check, plan or progress log.
- I did not read commit messages.
- All scripts ran from a fresh folder `rev14a/`, never from the parent scratchpad.
- I modified nothing in either repository except writing this file, and I made no commits.
- I read the authors' scripts only to get definitions: how B_host, B_c and B_p are taken from `concur.txt`, which counters are M and A, and the fallback G. Every number in the table below was computed by my own code from raw rows.

**Note for the chairs.** The author block in `paper/paper.tex` contains a real name and a contact address. The style hides it in the PDF, but anyone who opens the artifact's LaTeX sees it. This did not affect my assessment.

---

## Summary

The paper studies batch-1 decode of two MoE models (gpt-oss-120b MXFP4 and Qwen3-30B-A3B BF16) on rented RTX 5090 hosts, with most experts held in host DRAM. It has four parts.

1. **Bounds (§3).**
   - Eq. (1) is a roofline-style bound. Host traffic is Belady's MIN-with-bypass reads per token, R\*, at the machine's highest probed host-read rate; the GPU traffic is the dense weights plus the resident experts.
   - Eq. (2) adds the non-expert GPU time in series, for systems that read on demand.
   - A microbenchmark shows the read term of Eq. (1) is 86–96% reachable when no compute runs.
   - The authors' llama.cpp expert cache beats FreeToken at 11 of 12 cells and runs 2.0–4.0× stock llama.cpp. It stands at 38–54% of Eq. (1) at gpt-oss 11%.
2. **Where the time goes (§4).** An accounting relation, Eq. (3), is proposed: profiled GPU kernel time G plus the counted host reads at B_host, with background admissions discounted by G/T.
   - It was found on earlier runs.
   - It was then registered before four jobs. None passed as registered: one test is "inconclusive", three failed.
   - It fits re-rented consumer machines within 6%, but misses new consumer machines by up to 7% (11% on one Qwen3 cell) and server machines by 28–59%.
   - A processor-class split, explicitly drawn after the tests, separates most fits from misses.
3. **Foresight in the engine (§5).** Trace-driven oracles show that MIN's admission set, copied once in the step, gains 16–51% over the deployed cache on three hosts. Belady prefetch and two-read loading waste reads.
   - Among MIN's hit-optimal schedules, the fewest-admission one gains on every machine that passed the checks.
   - On slow-link machines the greedy schedule loses, and background loading wins.
   - The same "admit less" lesson applied online, via a larger admission margin, gains about 2%.
4. **Horizon (§6).** In replay on nine models' traces, half of MIN's read saving over "admit every miss" needs a window covering about 0.65·C distinct experts. Linear and GRU forecasters trained on other text close at most 15% of the gap.

The artifact is unusually complete. Every job script carries predictions committed before its rental. The appendix and supplement score 1,000+ clauses, failures included.

## Strengths

1. **The numbers reproduce.** I recomputed 43 quantitative claims from raw rows, counters and probe logs; the table at the end lists them. Every number I checked matches to the stated precision or within my aggregation choices. That covers Table 3, all four rows of Table 4, Tables 18, 19, 21, 22 and 24, and the §4.3–§4.4 machine-class and decomposition numbers. This is the most reproducible empirical systems submission I have reviewed.
2. **The registration is real.** All 124 rentals in the ledger started after the first commit of their job script. The header amendments in jobs 100, 102 and 105–107 change only host lists. The one prediction edit, in job 100, was committed before any of its machines started.
3. **Negative results are reported honestly.** Table 4 states that the paper's own descriptive relation failed every registered test with new machines. The abstract says so too. Validity gates were added before the later tests rather than after. Post-hoc exclusions are marked in Table 4's last column. Few papers report their central model's failures this candidly.
4. **The in-engine oracle study is useful.** The three-way contrast is clear on all six budgets of three hosts and is robust to the machine: MIN read once vs Belady prefetch vs MIN read twice. The fewest-admission schedule is a concrete, transferable insight for MoE-offload designers. It makes 0.60–0.73 of the greedy schedule's copies and gains on 9 of 9 machines that passed checks. The link-to-CPU ratio is the variable that decides which loading path pays: Spearman 0.88–0.89 over 19 machines, with machines carrying 96% of the variance.
5. **The statistics treat the machine as the unit.** Same-CPU hosts differ by up to 29% (panel) while relaunches agree within 3.2%. The paper acts on this: per-machine claims, two-stage bootstrap, round-level intervals in later jobs, and Kendall's W.
6. **The system result is solid.** The engine-to-engine comparison is fair on its face. FreeToken's backend is chosen on a separate launch and tuned on a third host. Long-output and own-headline-model checks are reported. The engine leads at 11 of 12 cells.

## Weaknesses (most important first)

### W1. The relation behind the title is exploratory and has not passed an out-of-sample test

The decomposition in §4.4 and Fig. 2 rests entirely on Eq. (3).

- **No out-of-sample pass.** Eq. (3) was found on jobs 093–105. Its only full pass (job 106, "held 12/12") is on three machines already in the data it was found on. On new consumer machines it under-predicts every gpt-oss cell, by 0.95–7.15% (my values). That is close in a practical sense, but it fails the authors' own bands.
- **The pass hinges on one cell.** The 12/12 on re-rented machines depends on 106a Qwen3 25% at +5.99% against a 6% band. The half-discount alternative (A/2) passes 11/12 with a lower median error: 1.9% against 2.3%. The paper concedes the term is "not singled out".
- **A trade-off is omitted.** The main text says the overlap term brings 33 rather than 18 of 34 launches within 6% at 25%. It does not say that at 11% the overlap form is worse: 29 of 34 against the plain form's 32 of 34. I reproduced both counts. The job-106 header states this; §4.1 should too.
- **G is borrowed for most machines.** At gpt-oss 11%, 30 of the 39 "valid consumer launches" behind §4.3 and Fig. 2 have no on-machine profile and use a median G of 4.28 ms. The "within 8% on 37 of 39" result is therefore largely a test of host reads at B_host with a constant, not of profiled compute plus reads.
- **What this does not change.** The data support the qualitative story on consumer desktops: GPU compute in series, plus MIN's reads, plus extra reads. The quantitative decomposition and its generality are weaker than the section heading "Where the seconds go" implies.

### W2. B_host, the single highest probe sample, is fragile, so Eq. (1) is a probe-relative reference rather than a bound

B_host is the maximum over all individual probe lines: CPU-only reads at each thread count, PCIe reads, and concurrent sums.

- **Consumer machines.** The maximum is typically about 2% above the larger of the CPU-only maximum and the median concurrent sum.
- **The two many-core servers** (`107c`, `107d`). It is a single concurrent sample 34–35% above every other sample. On the dual-socket engineering sample, B_host = 204.5 GB/s comes from a line in which the CPU threads read 181.4 GB/s alongside PCIe. The same machine read 117.6 GB/s CPU-only at the same thread count and at most 151.4 GB/s at any count.
  - With B = 151 GB/s, the relation's error on that machine falls from −59% to −46 to −51%.
  - With B = 118 GB/s it falls to −40 to −43%.
  - The conclusion "server machines read well below the probe" survives. Its magnitude (the abstract's "up to 59%") and the "0.27 of the probe" figure are partly artifacts of the statistic.
- **The engine reads faster than the "bound".** By Eq. (3)'s own accounting, the engine reads more than 5% faster than B_host on 13 of 68 launch-budgets, up to 1.22×. I reproduced this with the authors' uniform-G convention; with per-host profiled G it is 11 of 68.
- **The paper's own disclosures.** It acknowledges both issues (§3 "relative to the probe"; §8). Table 23 shows that using the median sample moves the host-bound limits by 8%.
- **Why this matters.** The abstract still says the read time at the probed rate "bound[s] every system that executes the exact routing". At minimum, the abstract should say "probe-relative", and B_host should be a robust statistic (median or trimmed mean of repeated samples) with its dispersion reported.

### W3. Several headline sentences are broader than the evidence

- **Abstract and §3, "38–54% of that bound … on 21 machines".** The 21 machines are the exploratory set (jobs 093–104). Across all 27 consumer machines that passed checks, I get 31.0–54.2%; the Threadripper 9960X is at 31%. Adding the three valid server machines gives 12.5–54.2%. The later machines are exactly where the cache does worst, so the abstract should give the full range or say "exploratory set".
- **Abstract, "knowing the future routing pays when it is spent as MIN spends it, reading each admitted expert once".** This is true on the three oracle hosts at every budget. It is false for MIN's greedy schedule on the two machines whose link reads at 0.28–0.32 of the CPU rate (Pf and the Threadripper 9960X): 0.85–0.93× at 11% on all 8 of their launches (jobs 099f, 101a, 103a, 104a, 105b, 105e, 106a, 106b). §5 explains this ("the machine decides"), but the abstract and Table 1 ("held at most cells") do not say that the cells where it fails are the slow-link machines.
- **§6, "shorter or half-right windows lose".** On the deployed path at gpt-oss 11%, the 4-token window measures as follows:

  | Host | b4/base [95% CI] |
  |---|---|
  | 100a | 1.018 [1.014, 1.022] |
  | 100c | 1.011 [1.008, 1.013] |
  | 100f | 1.009 [0.998, 1.022] |
  | 100b | 0.992 [0.989, 0.995] |

  Shorter windows gain on two hosts, tie on one and lose on one. Only the half-right window (0.94–0.98) loses everywhere.
- **§6 heading "Nothing realisable reaches it".** Two forecasters were tested (ridge and a 512-unit GRU), trained on out-of-domain text. The body correctly limits the claim ("across text, not … in domain"). The heading and the conclusion's "no forecaster we tested" are fine; "nothing realisable" is not.

### W4. Some exclusions in Table 1 were decided after the fact but are not labelled that way

- **Fewest-admission claim.** Table 1 labels "MIN's fewest-admission schedule gains on every valid machine" as "registered; held on valid machines". In job 106, registered prediction 7 (fetchplan/base > 1 on every host) failed on the EPYC 7302: 0.85×, losing in every round. On the Xeon 8347C it lost in round 1 (0.996×). The supplement's job-106 scorecard records both failures correctly. They are excluded in §5 by validity criteria (wrong outputs, unstable rounds) that were registered only from job 107 onward. The exclusions are defensible: the EPYC's teacher-forced loss was 0.21–0.45 nats against 0.19. But Table 1 should say "registered; held on 9 machines; failed on 1 machine excluded after the fact (wrong outputs)".
- **Unlabelled exploratory results.** Two main-text results are exploratory but carry no label: the §4.4 decomposition, and the §5 "machine decides how much" Spearman analysis. The other main-text results are labelled consistently.

### W5. The scope is narrow, and the novelty is modest per contribution

- **Scope.**
  - One GPU model.
  - Two models.
  - One prompt set (AIME-25), which was also used during development.
  - Engine experiments are teacher-forced over 20 (or 30) problems × 256 tokens.
  - Batch size 1.
  - Rented machines whose clocks could not be locked.
- **Novelty, contribution by contribution.**
  - The bound is a roofline with MIN's read count.
  - Choosing a hit-optimal schedule by admission cost has precedents the paper cites (Demand-MIN, FOO).
  - Measuring lookahead in distinct pages is classical (Albers; Breslauer).
  - The system's lead over FreeToken is 3–29% at most cells and −3% at one.
- **The contribution is the combination.** Bound, in-engine oracles, a schedule choice that matters by machine, and a horizon price are measured together with unusual rigor. That is a real contribution to MLSys, but each individual finding is incremental. The paper also spreads itself over many findings, each partly held, which dilutes the message.

### W6. Smaller correctness and consistency issues

- **Number of registered tests.** The introduction says the relation was registered before four later experiments; §4.2 says three. Job 105 tested the plain form, which is not Eq. (3).
- **Problem count.** §2 says engine experiments use the first 20 problems. Jobs 093–097, which supply Fig. 3's three hosts and Tables 9–12, used 30. I checked the row counts.
- **Same-CPU spread.** "Two hosts with the same CPU model differ by up to 29%" is computed within the 10-host panel only. Across all launches, the two Core Ultra 9 285K machines differ by 31–33% (099a at 18.1 ms against Pf at 13.7–14.1 ms).
- **Fallback G.** The fallback G is described as the "median of 8 earlier profiles" from desktop-class hosts, but it includes the EPYC 7402P's profile. This is immaterial (4.29 against 4.28 ms).
- **Two GPU terms.** T_GPU (Eq. 2; 2.9–3.4 ms) and G (Eq. 3; 4.1–4.7 ms) are never reconciled in the main text. From the decode-tail traces, G minus the MXFP4 expert matmuls is about 3.2–3.4 ms on the three hosts I checked, consistent with T_GPU. The paper should say this.

## Clarity

The rewrite is a real improvement over what a 2/5 score would describe.

### What works

- **Question-led structure.** Each section opens with a question, and paragraph headings state the finding with an evidence tag ("(registered)", "(exploratory)", "After the fact"). This is the single most useful change.
- **Table 1.** It maps claims to evidence types. Table 4 puts the entire registered record of the relation on one line per job, including failures and what was set aside.
- **Fig. 1 and the running example.** The step schematic makes "two reads per admission" concrete. The running example (one 9950X host at gpt-oss 11%: 10.0 → 13.0 → 20.6 → 15.2 → 13.8 ms) anchors §5.
- **Fig. 2 and Fig. 4.** Fig. 2's per-machine stacked bars show at a glance where the relation fits and where it does not (hatched). Fig. 4 shows the link-ratio dependence directly.
- **Limitations.** They are specific and quantified.

### What still makes it hard to read

1. **Too many labels, with no key.** The main text and appendices use roughly 35 job numbers (073–108 with letter suffixes). Hosts appear under about 40 labels: A, B, S, O1–O6, Pa–Pj, "Pf again", "285K, second", "Pg again (5950X)", "9950X, x8", "Pd again". Configurations use two vocabularies:
   - **Table 2 names:** "deployed", "single read", "MIN, 1 read", "MIN, fewest adm.", "MIN, prefetched", "Belady", "window W".
   - **Appendix labels:** online, foa, fetch, bypass, both, lead, paced, nb2, hit-opt, fetchplan, bypassplan, dk, lrn, aa, b4/b16/w4/w16, b8r5, allr5.

   No table maps one to the other, so a reader of Appendix G or J must reverse-engineer "fetch = MIN, 1 read" and "both3p = MIN, prefetched". A one-page glossary with both name sets, and a host table (label, CPU, B_c/B_p/B_host, jobs), would fix this.
2. **The core equations are not fully explained.**
   - Eq. (3) is implicit ("solved for T"). It is a quadratic with a one-line closed form, T = ½[b + √(b² − 4kAG)] with b = G + k(M+A) and k = S/B_host. Giving it would let readers check numbers by hand.
   - The two GPU terms (T_GPU in Eq. 2, G in Eq. 3) are introduced two pages apart with different values (2.9–3.4 against 4.1–4.7 ms) and no sentence relating them.
   - B_host is defined as "the highest", but the main text never says it is the maximum of single noisy samples (W2).
3. **The baseline changes silently between sections.** §5 and Figs. 3–4 measure speed against the deployed cache. §6 and Fig. 5 measure "share of MIN's gain over admit every miss". In time, that baseline runs 0.42–0.97× the deployed cache on the panel at 11%, so a window that "recovers 0.78–0.93 of MIN's gain" can still be slower than the deployed cache. Table 17 shows in-step W=16 at 0.89× on slow-link hosts. §6 states the switch in one clause, but Fig. 5 and the 0.65·C rule are read against the weaker baseline without saying how weak it is.
4. **Number density.** Most main-text sentences carry two to four ranges.
   - The abstract has nine numbers.
   - §4.2 reads like a ledger ("by 52–59%, … up to 11%, … 34.2% and 27.9%, … 5.3%, … 0.9–7.1%").
   - Table 3's caption is eight lines and introduces two bound variants plus pooled bounds for other systems.
   - The reader cannot easily tell which numbers carry the argument. Cutting each paragraph to its one or two load-bearing numbers would help, with the rest left to the tables.
5. **The appendices were not rewritten.**
   - They still say "the law", "law's table", "the limit" and "the model" for things the main text now calls "the relation", "the machine's fetch table" and "the bound". Fig. 7's legend says "law's table"; Appendix K says "The law predicted llama.cpp…".
   - Appendix B is a single paragraph of about 1.5 pages listing failures for 30+ jobs, and is very hard to audit.
   - The main text also contradicts itself on counts: "four later experiments" against "three more experiments"; "first 20 problems" against the 30 used by jobs 093–097.

### Smaller points

- Fig. 3 uses "single read" for the deployed policy read once and "MIN, 1 read" for MIN. These are near-synonyms that differ in meaning; "deployed, 1 read" would be clearer.
- Table 1's "registered (3 hosts)" and "registered; held (3 hosts)" read as two different statuses.
- Fig. 2's caption should say that G is a median profile for most of the bars.

**Clarity score: 3/5** (up from 2). The main text can now be followed on a first read. Auditing any one number still requires decoding job and host labels across 28 appendix pages.

## Questions for the authors

1. Why is B_host the single highest sample rather than a robust statistic over repeated samples? On the dual-socket machine (`107d`), is the concurrent t=128 line (CPU 181.4 GB/s alongside PCIe, against 117.6 GB/s CPU-only) a measurement artifact? How do the server conclusions and Eq. (1) change with a median?
2. Please reconcile T_GPU (Eq. 2) and G (Eq. 3). Is G − T_GPU the resident-expert compute? Why use different quantities in the two equations?
3. Why does §4.1 omit that the overlap form fits worse at 11% (29 against 32 of 34 within 6%)? Is there a physical reason the overlap discount should apply at 25% but not at 11%?
4. For the 30 of 39 consumer launches without an on-machine profile, how sensitive is "37 of 39 within 8%" to G across the observed range of 4.05–4.65 ms?
5. Was "computed wrong outputs" pre-specified as an exclusion anywhere before job 107? If not, would you relabel the fewest-admission row of Table 1?
6. Can a third party verify commit-before-launch, for example through pushed-commit timestamps on the hosting service? Local commit timestamps can be set arbitrarily.
7. Does the 0.65·C horizon rule hold in engine time at budgets other than gpt-oss 11%/25% and Qwen3 12.5%, and against the deployed cache rather than "admit every miss"?
8. How do the conclusions change for free-running (not teacher-forced) generation, and for prompts outside AIME, given that AIME was also used during development?

## What would raise my score

1. **Re-scope the abstract and Table 1.**
   - Give the full range of the fraction of the bound, or say "exploratory set".
   - Say that MIN read once loses with the greedy schedule on slow links.
   - Say "probe-relative" for the bound.
   - Relabel the fewest-admission row with its after-the-fact exclusion.
   - Fix "shorter windows lose".
2. **Use a robust B_host** (median or trimmed mean of repeated samples) and report its dispersion. Recompute Eq. (1), the implied read rates and the server findings with it, and show that the conclusions do not depend on the outlier statistic.
3. **Either pass an out-of-sample registered test of Eq. (3)** on new consumer machines with on-machine G, or reframe Eq. (3) as a descriptive accounting identity and report the plain, overlap and half-discount forms at both budgets.
4. **Add a glossary**: configuration names with their appendix labels, and a host table.
5. **State one more set of results.** Even one free-running, non-AIME workload, or a second GPU type, for the headline comparison and the oracle three-way would strengthen the significance case considerably.

## Scores

| Criterion | Score | Reason |
|---|---|---|
| Overall | **6 / 10** (weak accept) | Rigorous and fully reproducible, with useful in-engine findings. The central relation is not validated out of sample, the bound rests on a fragile probe statistic, and several headline sentences overreach. All of these are fixable. |
| Soundness | **3 / 5** | The data and arithmetic are excellent: every number I checked reproduces. Inference and scoping are weaker (W1–W4). |
| Significance | **3 / 5** | Relevant to local MoE inference and offload-system design, but in a narrow setting: batch 1, one GPU model, two models, one prompt set. |
| Novelty | **3 / 5** | A new combination of a bound with in-engine oracles and a schedule choice that depends on the machine. Each idea has close precedents. |
| Clarity | **3 / 5** | A clear improvement over the previous round; still label-dense and appendix-dependent. |
| Confidence | **4 / 5** | I re-derived about 40 claims from raw data. I did not rebuild the engine or re-run the trace studies of §6 and Appendix H. |

## Claims checked against raw data

"Mine" values are from my own code over the raw `ec_*.jsonl`, `st_*.json`, `concur.txt`, `g_prof.json` and Nsight files. Verdicts:

- **Reproduced:** matches to the precision printed.
- **≈:** matches within my aggregation choice.
- **Scope / labelling issue:** the number is right, but the claim built on it is broader than the evidence.
- **Contradicted:** the number does not support the claim.

### Registration and the registered tests of Eq. (3) (Table 4)

| # | Claim | Location | My value | Verdict |
|---|---|---|---|---|
| 1 | Job 105, plain form: 5/8 cells within 6%; new machines 2/4; median 4.2%; failed | Table 4 | 5/8; 2/4; 4.24%. Fails at 25% on 3 hosts (+6.9, +7.0, +9.3%) | Reproduced |
| 2 | Job 106: 12/20 cells; new machines 0/8; median 4.7%. Set aside: 12/12, 2.3% | Table 4, §4.2 | 12/20; 0/8; 4.67%. 12/12, 2.34%, but 106a Qwen3 25% is at +5.99% | Reproduced (fragile) |
| 3 | Job 107: 2 valid of 7; 1/8 cells; median 31.7%; inconclusive | Table 4 | 2/7; 1/8; 31.75% | Reproduced |
| 4 | Job 108: 4/6 cells; median 6.7%. Without the EPYC 7543: 4/4, 5.3%, two pooled clauses fail | Table 4, §4.2 | 4/6; 6.70%. 4/4, 5.25%; two cells beyond 6% | Reproduced |
| 5 | Predictions committed before each machine started | App. B | All 124 ledger rentals start after the first commit of their script. Amendments change hosts only; job 100's prediction 6 edit precedes all its starts | Verified (local timestamps) |
| 6 | Relation misses by up to 11% on new consumer and up to 59% on server machines | Abstract | 11.3% (5900XT, Qwen3 12.5%); 58.9% (dual-socket ES, gpt-oss 11%) | Reproduced; 59% → 43–51% without the one outlier probe sample |
| 7 | EPYC 7543 misses by 34.2% and 27.9% | §4.2 | −34.23% / −27.91% (G = 4.29 / 4.47) | Reproduced |
| 8 | New consumer machines under-predicted at every gpt-oss cell by 0.9–7.1% | §4.2 | 0.95–7.15% | Reproduced |
| 13 | Half-discount form passes 11 of 12 re-rented cells | §4.2 | 11/12; median 1.9% | Reproduced |
| 39 | Excluded hosts: EPYC 7302 loss 0.21–0.45 nats; Xeon rounds 52% apart; EPYC 9754 / 7663 rounds 3.5% / 6.0% apart, misses 31% / 21%; EPYC 7K62 7.0%, 26.8% | §4.2, App. J | 0.208–0.446; 52.2%; 3.5% / 6.0%; −31.1% / −20.9%; 7.0%, −26.8% | Reproduced |
| 40 | Profiled G 4.1–4.7 ms; fallback G 4.29 / 4.47 ms, median of 8 profiles | §4.1, App. J | Decode-tail traces give 4.05, 4.17, 4.43 ms on 3 hosts (reported 4.05, 4.19, 4.45). Fallback 4.29 / 4.47 of 8, one of them an EPYC 7402P | Reproduced |

### The relation on exploratory launches, machine classes and the decomposition (§4.1, §4.3, §4.4)

| # | Claim | Location | My value | Verdict |
|---|---|---|---|---|
| 9 | 39 valid consumer launches on 25 machines; implied read rate 0.88–1.16 of B_host; within 8% on 37 | §4.3 | 39 on 25; 0.88–1.16; 37. 30 of 39 use the median G | Reproduced |
| 10 | Server: 5 of 8 launches invalid; ES 0.27, EPYC 7543 0.58, EPYC 7402P 1.04 | §4.3 | Same | Reproduced |
| 11 | Dual-socket probe reads 118 GB/s at 128 threads against 151 at 16 | §4.3 | 117.6 / 151.4; but B_host = 204.5 from one concurrent sample | Reproduced; adds caveat (W2) |
| 12 | Plain form within 6% on 28 of 30 at 11%; overlap form 33 vs 18 of 34 at 25% | §4.1 | 28/30; 33 vs 18. Omitted: at 11% the overlap form gets 29 vs the plain form's 32 of 34 | Reproduced; selective |
| 14 | At 11%: G 12–45% of time; MIN's reads 31–54%; reads beyond MIN's 22–35%; G is 23–66% of the gap | §4.4 | Same (25 consumer machines) | Reproduced |
| 15 | Reading below the probed rate: ≤11% of time on consumer machines; EPYC 7402P 0%, EPYC 7543 34%, ES 59% | §4.4 | ≤11% (minimum −12%); −3%, 34%, 58% | Reproduced |
| 16 | Elasticity of T−G with respect to B_host: −0.95 [−1.06, −0.84] | §4.4 | −0.97 [−1.08, −0.82] (machine means, 21 machines) | ≈ |
| 17 | Engine reads >5% faster than B_host on 13 of 68 launch-budgets, up to 1.22× | §8 | 13/68 and 1.22 with uniform median G; 11/68 with per-host profiled G | Reproduced |

### Bounds and the system comparison (§3, Table 3, App. K, App. L)

| # | Claim | Location | My value | Verdict |
|---|---|---|---|---|
| 18 | Cache at 38–54% of Eq. (1) on 21 machines at gpt-oss 11% | Abstract, §3 | 38.7–54.2% (R\* ≈ 38.4) on the 21 exploratory machines; 31.0–54.2% over 27 consumer; 12.5–54.2% with the 3 valid servers | Scope issue |
| 19 | Table 3 speeds and ratios, e.g. host B gpt-oss 11% 1.294 [1.278, 1.312]; host S Qwen3 43.75% 0.974 [0.962, 0.987] | Table 3 | Identical | Reproduced |
| 20 | Cache leads FreeToken at 11 of 12 cells; 2.0–4.0× stock llama.cpp | §3 | 11/12; 2.00–4.02× | Reproduced |
| 21 | Host B probe: CPU 72, link 53, together 78, highest 88 GB/s. Host S: 56 / 53 / 66 / 71 | Table 3 | 71.6 / 53.2 / 77.7 / 87.5; 56.2 / 52.8 / 66.1 / 71.3 | Reproduced |
| 22 | Against tuned FreeToken: 1.17, 1.17, 1.09 (gpt-oss); 1.02, 1.04, 0.98 (Qwen3) | App. K | Same | Reproduced |
| 23 | LRU runs 6–21% slower than decayed frequency on host S | App. K | 6.3–20.9% | Reproduced |
| 24 | Read microbenchmark reaches 86–96% of the bound's read term, per layer | §3, Table 24 | 86–96% per layer; 93–97% per token | Reproduced |
| 41 | Parity: mean KL 0.0019 / 0.0005 nats; NLL +0.16% / +0.19% | §8, App. K | Matches the on-machine `summary.txt`; logit dumps not recomputed | Consistent (not independently recomputed) |
| 43 | Published in-class systems reach a median 13.6% of the bound (20 rows) | §3, App. L | 13.6% from Table 26's 20 rows | Internally consistent only |

### Foresight and MIN's schedules (§5, App. G, App. J)

| # | Claim | Location | My value | Verdict |
|---|---|---|---|---|
| 25 | Running example on O4 (9950X), gpt-oss 11%: deployed 20.6, MIN 1 read 15.2, MIN prefetched 13.8 ms; Eq. (1) 10.0 ms | §5 | 20.64 / 15.13 / 13.78 ms; Eq. (1) ≈ 10.0 ms with exact R\* | Reproduced |
| 26 | MIN 1 read beats the deployed cache by 16–51% at every budget on O3–O5; up to 81% prefetched | §5 | 1.160–1.514×; maximum 1.812× | Reproduced |
| 27 | At the smallest budgets, Belady prefetch and MIN 2 reads gain at most 15% or lose | §5 | Maximum 1.148× (Belady 2-step lead, O4 Qwen3 12.5%) | Reproduced |
| 28 | Fewest-admission schedule makes 0.60–0.73 of the greedy one's copies, with 2.5–5.3% more misses | §5, §8 | 0.60 / 0.73; +2.5–5.2% | Reproduced |
| 29 | On links at 0.28–0.32 of CPU: greedy 0.85–0.93× at 11%; fewest-admission gains in every round; loaded by the CPU 1.11–1.22× | §5 | 0.853–0.933×; rounds 1.02–1.06×; 1.112–1.217× | Reproduced |
| 30 | Fewest-admission schedule beats the deployed cache on every machine passing checks (7 + 2) | §5, Table 1 | 9 machines at 1.02–1.42×; 0.85× on EPYC 7302 and 0.97× on EPYC 7663, both excluded (the first after the fact) | Labelling issue |
| 31 | Table 18 (job 104), all 24 entries | App. J | All match | Reproduced |
| 32 | Layer-ahead copy gains 1.04× where link ≈ CPU, down to 0.72× at 11% | §6, Table 19 | 1.042×; 0.722× (0.691× at 25%) | Reproduced |
| 33 | Spearman 0.89 [0.63, 0.97] over 19 machines (MIN copied in the step vs link ratio) | §5 | 0.88 [0.61, 0.97], 19 machines | ≈ |
| 34 | Machines carry 96% of the variance of log gain, problems ≤3%; Kendall's W 0.77 at 11% | §5, App. J | 96.3%, 1.1%; W = 0.77 | Reproduced |
| 35 | Larger admission margin: reads −4–6% at 11%; median 1.02× over 5 machines, up to +6%, −1% at 1 of 10 cells | §5 | 4.3–5.8%; 1.019×; 1.057×; 0.987× | Reproduced |

### Windows, the panel and forecasters (§2, §6)

| # | Claim | Location | My value | Verdict |
|---|---|---|---|---|
| 36 | 16-token window on the deployed path: 1.00–1.03× at 11%; "shorter or half-right windows lose" | §6 | 16-token: 1.00–1.03× (matches). 4-token: 1.018 [1.014, 1.022], 1.011, 1.009, 0.992 | Partly contradicted |
| 37 | Exact 16-token window recovers 0.78–0.93 of MIN's gain in time at 11% (panel) | §6 | 0.78–0.93; but the "admit every miss" baseline runs 0.42–0.97× the deployed cache | Reproduced; baseline caveat |
| 38 | Same-CPU hosts differ by up to 29%; relaunches within 3.2% | §2 | 29% within the panel; 31–33% across all launches (285K); relaunches within 3.2% | Reproduced (panel scope) |
| 42 | Linear forecaster closes 3–15% of the read gap on gpt-oss | §6 | 2.7–14.6% (`prereg/forecaster_linear.json`, a derived file) | Consistent (derived) |

**Not checked:**
- The 0.65·C horizon rule and the nine-model trace study. The traces are outside `results/`.
- The scorecard tallies (565 / 273 / … clauses) clause by clause. I checked only that the job-108 tally (21 clauses: 13 held on the point estimate, 6 failed, 2 untested) matches my recomputation.

## Minor issues

- §3: "Every system we compare reads on demand." FreeToken's prefill overlap lends it cache slots during each prompt. Say why that does not count as reading before routing.
- Table 2: "MIN, fewest adm." lists "2 or 1" reads. Say which load each figure uses.
- Fig. 3: no interval whiskers are visible at this scale. Either say they are smaller than the markers or drop "(95% interval)" from the axis label.
- Appendix M lists costs only through job 101; jobs 102–108 are missing.
- The `INVALID` table in `scripts/job107.py` labels the Xeon of job 106 "rounds 39% apart". The paper says 52%, which is what the raw rows give (max/min − 1 = 52.2%). Harmonise the two.
