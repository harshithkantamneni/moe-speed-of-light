# Review 24b: "Where the Seconds Go" (professor, systems benchmarking; 8 October)

## Materials read

- `paper/paper.pdf`, all 44 pages. I looked at pages 1–10 (the main text) as rendered page images, and read the appendices (A–N, pp. 15–44) as extracted text.
- LaTeX sources, for wording and macro definitions: `paper.tex`, `abstract_body.tex`, `main_body.tex`, `macros.tex`, `app_wsg.tex`, `app_prereg.tex`.
- `paper/supplement.pdf`: its structure, the job 109 clause table (its Table 15) and the job 112 T6 rows.
- `paper/ieee-paper.pdf`: all 10 pages, rendered.
- `paper/ieee-supplement.pdf`: pp. 1–4 rendered, and the full text searched for cross-references.
- The IEEE build files (`ieee-paper.tex`, `ieee_preamble.tex`, `ieee-supplement.tex`, `build_ieee.sh`) and the two IEEE build logs.
- Job script headers on the `gpu` checkout (`/home/claude/gpu-branch/jobs/`): 080, 081, 099, 106, 109, 110 and 112, plus the per-host wrappers 109a and 112a.
- Raw results in `/home/claude/gpu-branch/results/` for jobs 081, 089, 096a, 099a–j, 100b/f, 101a/b, 102a–c, 104a–c, 105a–g, 106a–e, 107a–g, 108a–f, 109a–f/s, 110a–e, 111d/f/g and 112a/b/d/g. The files used were `ec_*.jsonl`, `st_*.json`, `concur.txt`/`concur2.txt`, `g_prof.json`/`prof_*.json`, `validity.txt`, `v0.txt`, `gate.txt`, `cpu.txt`, `nvidia-smi-q.txt` and `bs1.jsonl`.
- `gpu/vast_ledger.json`.
- `git log` on the `gpu` checkout, using only hashes and committer/author times, and `git show --format=` diffs of the six job-script commits made while rentals were running.
- Authors' code, read only for definitions; every number was recomputed with my own scripts:
  - `jobs/ec2/fetch_table.py`: the `bandwidths()` rule for B_c and B_p.
  - `scripts/speed_limit.py`: `host_rates()`, the rule for B_host.
  - `scripts/reanalysis.py`: `features()`.
  - The header and `rows()` of `scripts/decomp_measured.py`: the panel definition.
  - `scripts/sumlaw_paper.py`: `profiles()`, the definition of T_GPU.
- My scripts are in `scratchpad/rev24b/`: `load.py`, `j109.py`, `j099.py`, `panel.py`, `trend.py`, `j112.py`, `few1r.py`, `regtime.py` and `null.py`.

## Independence statement

- I opened nothing under `reports/` other than to write this file.
- I did not open `prereg/*outcome*.md`, `research_notes/`, `apply/`, `paper/paper_v1_prereview.tex`, or any file named like a review, number check, plan or progress log.
- I read no commit messages: git output was limited to hashes, times, file names and diffs.
- All analysis ran from a new folder (`scratchpad/rev24b/`), never from the parent scratchpad.
- I modified nothing in either repository and committed nothing.

---

## Summary

The paper studies batch-1 decode of MoE models whose experts mostly live in host DRAM: gpt-oss-120b and Qwen3-30B-A3B on rented RTX 5090 and RTX 4090 hosts. It makes five contributions.

1. **A bound and its variants.**
   - Eq. (1) is the host reads of Belady's MIN with bypass (R*) at the machine's highest probed host read rate.
   - Eq. (2) is a "demand" bound that adds the GPU's non-expert time.
   - Eq. (3) is an "ordered" bound for CPU-served reads.
   - A microbenchmark replays MIN's reads at 86–96% of the read time.
2. **A llama.cpp expert cache.** It leads FreeToken and llama.cpp at most configurations and reaches 31–54% of Eq. (1) on consumer machines at gpt-oss 11%.
3. **A measured decomposition of the gap** using oracles built into the engine (Table 4):
   - What is cached (MIN's set) and how it is read (once in the step versus CPU-then-copy) pay only together: 33% of the gap on 15 panel machines, 35% on 5 new machines.
   - A read-ahead oracle closes another 15–19%.
   - The rest (about half) is attributed, not measured.
4. **Policies without foresight.** Four variants built from the engine's mechanisms recover at most 6% of the read-ahead oracle's gain.
5. **Two predictive rules.**
   - A trend in the link-to-CPU ratio, frozen on RTX 5090s, predicts the oracles' speed on five RTX 4090s within 7%.
   - A horizon rule, labelled post hoc: half of foresight's value needs about 0.65·C distinct experts of routing per layer.

Every engine experiment carried predictions in its job-script header, committed before the rental started, and an appendix scores all of them (1,672 clauses, 321 failed).

## Strengths

1. **The registration practice is as good as I have seen in a systems paper, and it checks out.**
   - All 92 rentals of jobs 093–112 in the ledger have a commit of their job script before the rental started. The smallest lead is 2.7 s, for 105f, as the paper says.
   - Six commits touched a job script while a rental of that job was running (jobs 102, 105, 106, 107 twice, and 111). Their diffs are all host substitutions or relaunch rules, each marked "predictions unchanged". No main job script was edited after its job's last rental stopped.
   - Failures are reported in full rather than buried: job 110 is voided and its machines are listed, job 107 is "inconclusive by its registered rule", and Table 11 lists the failed clauses job by job.
2. **The numbers reproduce from raw data.** I recomputed 30 quantities from per-problem rows, counters, probes and profiles with my own code (table below). Every one matches to the paper's rounding, including:
   - Table 4's panel and new-machine columns with their t-intervals. I reconstructed the panel machine set independently: one launch per GPU UUID, jobs 096–101, all five states.
   - Every RTX 5090 and RTX 4090 row of Table 6 and every cell of Table 5.
   - The frozen trend coefficients in job 110's header, refitted to four decimals.
   - The 13-machine Few-1R summary.
3. **The statistical unit is chosen deliberately and checked.**
   - Machines differ far more than problems: two 285K panel machines differ by 29% in deployed time, which I confirmed.
   - Summaries over machines use t-intervals over machines. Round-to-round variation is measured: 98 machine-budget-configurations on 12 machines, median 0.17%, maximum 1.74% in my computation.
   - Rental-to-rental differences are used to widen intervals.
   - This follows Hoefler and Belli's rules 5–7 seriously, not cosmetically.
4. **The validity gates are concrete and registered from job 107 on**: GPU UUID, memory in use, NUMA and core class, device-read floor, loss within 2% of reference, rounds within 2%. Every host that started is reported. I confirmed each gate outcome for jobs 107–112 from `v0.txt`/`validity.txt` (for example, 107c failed at 3.54%, 107e at 5.98% and 108b at 7.01%).
5. **The second-card test is a genuine out-of-sample prediction, and it beats more than a straw man.**
   - The paper compares only against "no change". I also tested a constant-gain null: the panel's mean log gain, ignoring the ratio.
   - The frozen ratio trend misses the RTX 4090s by at most 0.064 in log. The constant model misses by up to 0.23. The link-to-CPU ratio therefore carries real information.
6. **The bound framing is useful, and the paper is candid about its looseness.**
   - Table 28 tightens Eq. (1) four ways.
   - The "demand" and "ordered" variants make clear which part of the gap foresight could remove at all.
   - The microbenchmark (Table 29; I reproduced every cell) shows the read time is approachable without the model.
7. **Scoping is mostly explicit.** "In our engine", "at that budget" and "on machines whose link reads at least half as fast" appear in the abstract and Section 9. Post-hoc material carries labels: the 31–54% range, the 13.6% audit median and the horizon rule are all marked "post hoc".

## Weaknesses (most important first)

### 1. The central 2×2 does not manipulate the factor it names, and its control failed, yet Table 1 still reports a "held" mechanism

The headline interaction ("what is cached and how it is read pay only together") treats MIN-2R as "MIN's set, read twice". The counters show the engine does not realize MIN's set on the two-read path.

**Reads at gpt-oss 11% on the five job 109 machines** (from `st_g_C14_r1A_*.json`):

| State | Host reads per token | Misses per token |
|---|---|---|
| Deployed cache | 1.65–1.66 R* | – |
| MIN-1R | 1.04 R* | – |
| MIN-2R | 1.73–1.74 R* | 48.6–50.2, i.e. 1.27–1.31 R* |

So MIN-2R reads more host bytes than the deployed cache it is compared against, and its misses alone exceed R* by more than a quarter.

**What the "set alone" cell actually measures.** It measures what this engine's two-read path (copies published two steps late, the fetch table's in-step fills, a queue cap) does with an aggressive admission set. That is an engine property, not a property of "what is cached".

**The registered control could not separate the causes.** Few-2R-early (job 112) was meant to separate late landing from the second read, and it failed its manipulation check:

- Its misses were 0.968–0.980× those of Few-2R at 11%, against a registered ≤ 0.95 (T1 failed on 5 of 8 machine-budgets).
- It was slower, not faster (T2 failed on 7 of 8).
- Table 1 nevertheless reports "the second read costs more than the late landing (held)".
- That clause, T3, is (Few-1R − early) > (early − Few-2R). Its right-hand side is negative whenever T2 fails; I measured early − Few-2R = −0.031 to −0.053 at 11%. T3 therefore had to hold, and it carries no information once the manipulation failed.
- The body text says this correctly ("the control cannot say how much … is late landing"), but Table 1 and builder rule 2 do not.

**Consequence.** Rule 2 ("Change what is cached and how it is read together") is stated as advice to builders. It is supported only for an engine whose two-read path cannot hold the set.

### 2. Confirmatory strength is thinner than Table 1's labels suggest

- **Table 1's "held" means only that the point estimate meets the threshold.**
  - The appendix's stricter "held (interval)" is not shown in the main table.
  - The key per-machine sign claim (interaction > 0 on every machine) has no interval at all: every `x-P1-int-*` clause in the supplement is "held (point)".
  - The headline 35% is a mean over 5 machines with a t-interval of [22, 48] (I get 34.7 [21.9, 47.5]). Its registered band of 0.25–0.55 was, as the authors concede, nearly impossible to fail.
- **The "loose" label is applied unevenly.**
  - Job 109's "≤ 1.10× for policies without foresight" was as safe as the capture bound, because earlier jobs showed Margin at 1.02× and the layer-ahead copy at most 1.04×. Only the capture bound is flagged.
  - The system-comparison row's thresholds (host B, job 081) were set after jobs 076–080, including job 080 on the same host offer (51046112). The appendix itself calls these predictions "made after a same-machine pilot".
- **The Few-1R row ("gains on stable machines") rests on an exclusion rule that was not registered for job 106.**
  - Job 106 registered no output-correctness gate. Its "≤ 2% between rounds" is prediction 8, not a gate; gates start with job 107.
  - The registered claim (">1 on every machine") failed on job 106's EPYC 7302, which ran at 0.845×.
  - "Held on the 7 registered" uses the post-hoc stability rule. The rule is technically justified (that machine's loss was 0.279 against 0.190), but the row should say the exclusion was post hoc.
  - "The stability rule leaves out exactly those" is also slightly misleading. The rule also removes two machines that gained (Xeon 8347C at 1.30× and EPYC 9754 at 1.18×).

### 3. The second-card result is less blind and less general than the main text implies

- Three of the five RTX 4090s (job 111) were relaunched after their first rounds were in the repository. Those rounds were within the band: I recompute deviations of −0.059 to +0.062 in log for 110b–e.
- The relaunch also changed the V1 reference loss after seeing data.
- One eligible machine was left out by oversight.
- Only the two job 112 machines were fully blind.
- All RTX 4090s sit at ratios 0.37–0.77, and three of them at 0.37–0.40.
- Applied to the new RTX 5090s, which come from the very population the trend describes, the trend misses 5 of 18 cells by up to 18% (I get 17.6%).
- So "within 7%" is the luckier of two out-of-sample samples, not the trend's out-of-sample error. The paper says so in Section 7 ("sets the direction … more reliably than its size"), but the intro and Table 1 lead with 7%.

### 4. The "bound" is relative to a noisy probe, and the uncertainty is not propagated

- **How B_host is chosen.** It is the maximum of about 25 probe readings, including the concurrent CPU-plus-link sum. That makes it an extreme-value statistic of a noisy probe.
- **The engine sometimes beats it.** By Eq. (4)'s accounting, the engine reads faster than the probe on 12 of 84 launch-budgets, by up to 1.22×.
- **The probe moves between rentals of the same machine.** For example:
  - The ratio of the 5950X moved from 0.84 (job 109) to 0.74 (job 112), while the deployed time changed by only 0.7%. I reproduced both numbers.
  - The second-highest reading sits up to 26% below the highest.
- **Consequences:**
  - Table 1's label "derived" for the bound is too strong. The inequality is derived only conditional on B_host being the attainable maximum.
  - Share-of-bound figures (31–54%, 41–48%) carry several percentage points of probe uncertainty that no interval reflects.
  - Classifying machines as fast or slow at 0.5 is itself noisy near the line: job 104's second 285K sits at 0.49.
- **T_GPU and the attribution.** T_GPU (2.94 ms) is the minimum of 14 older profiles, whereas the job 109 machines have their own profiles (2.905–3.284 ms). Table 4's "attributed" rows therefore depend on a choice that per-machine data could replace.

### 5. Generalization is narrow, and the samples are less independent than "never rented before" suggests

- The decomposition, the registered 2×2 and the no-foresight test all come from one engine (the authors'), one model at two budgets, and 20 AIME-25 problems that were also used during development, replayed teacher-forced.
- Hosts are a convenience sample of cheap offers.
- "Never rented before" was enforced by GPU UUID, not by host. The 5950X machines of 108f (offer 54573924) and 109f/112g (offer 54573923) share:
  - the CPU model,
  - 125 GB of RAM,
  - PCIe bus ID 06:00.0,
  - a 28.3 GB/s link,
  - adjacent offer IDs.

  They are at least the same build from the same provider, so they are not independent draws.
- The builder rules (Section 9) are hedged as "hypotheses to test", which is right. Rules 2 and 3 are nonetheless phrased as general advice.

### 6. Within-machine intervals treat problems as independent when the cache carries state between them

- The paired bootstrap over the 20 problems ignores that "the cache carries over between problems" in a fixed problem order.
- Round-to-round noise is about 0.2% while interval half-widths are about 0.6–0.7%. So the intervals mostly measure heterogeneity of the effect across prompts, not measurement uncertainty.
- That is a legitimate target (generalization to similar prompts), but the paper should say which population the interval is about.
- It should also check robustness with a block bootstrap, or by treating each machine's whole 20-problem sequence as the unit.

### 7. The audit of published systems is weak evidence placed prominently

- The median of 13.6% (which I confirmed from Table 31's 20 in-class rows) mixes:
  - datasheet ceilings,
  - traces of sibling variants,
  - per-layer bounds applied to pooled-cache or layer-pinning designs,
  - rows below 5% that the paper declines to adjudicate but still counts in the median.
- It is labelled post hoc but appears in the abstract-level intro alongside measured results. It should be moved to Section 3's end or carry its caveats where it is quoted.

## Clarity

The paper has improved: the claims table, the single names table and the claim-to-clause index (Table 9) all help. It is still hard to read, mostly because it reports everything it measured, everywhere, with full qualification. My score stays at 3/5.

### What still makes it hard to read

1. **Too many names and baselines in the main text.**
   - Table 2 defines about 13 configurations: Dep-1R, Margin, LA, LA-1R, LA-1R-margin, admit every miss, MIN-2R, MIN-1R, the read-ahead oracle, Few-2R, Few-1R, Few-2R-early, Belady-1R/2R.
   - It also defines six machine sets and five yardsticks.
   - The appendix adds about 30 codes (Table 7: base, foa, aa, dk, pf, R1, R2, fetch, bypass, fetchplan, bypassplanS, both3p, hitoptp, nb2, w8r5, …) and host letters (A, B, S, O1–O6, Pa–Pj).
   - A reader of Section 4 must hold MIN-2R, Dep-1R, MIN-1R, Few-2R, Few-1R and Few-2R-early at once, and Section 6 adds four more.
   - The baseline changes by section: speed versus the deployed cache, share of the Eq. (1) gap, share of the Eq. (3) gap, capture versus the read-ahead oracle's gain, and value versus admitting every miss (Fig. 5 has to warn "Baseline here").
2. **Numbers are packed into prose.**
   - Section 4's "What is left" paragraph gives seven ranges in eleven lines, across three machine groups (slow-link panel, RTX 4090s, fast-link median), two bounds (Eq. (1), Eq. (3)) and two T_GPU choices.
   - Section 7's paragraph and the intro's five bold paragraphs repeat Table 1's numbers, each with its scope qualifier.
   - Almost every sentence reports panel and new machines, 11% and 25%, fast and slow links, RTX 5090 and RTX 4090.
3. **Table 1 is the paper's spine, but its cells are hard to parse.**
   - Result cells mix registered outcomes, failures, scope changes and post-hoc numbers. For example, "positive on 9 of 10 panel machines (failed), 5 of 5 new ones (held; fast-link scope set in between); share 35%".
   - The "Mach." column needs a decoding rule ("10 (15)+5").
   - The timing-control row reads as a supported claim with two failures and one "held" in the same cell.
   - "Held" means something weaker than in Appendix D.
4. **Forward references make the first pass circular.**
   - Section 2 cites Table 3 and "both hosts of Table 3" before the hosts exist.
   - Section 3 quotes the "5 machines added in Section 4" and the closed-form account of Appendix H.
   - Section 4 imports Section 7's RTX 4090s and Eq. (3).
   - Section 5's Few-1R result counts RTX 4090s and server machines introduced later.
   - The key timing fact, that the engine publishes a background copy two steps after the admission, is buried in "Terms" although Section 4's argument turns on it.
5. **Scopes are inconsistent across restatements.**
   - The abstract and rule 2 scope the 2×2 to fast links.
   - The intro's "33% of the gap on 15 panel machines" includes the two slow-link machines. On the 13 fast-link panel machines I get 38.8% [31.0, 46.6].
   - The reader cannot tell which number belongs to which scope.
6. **The sentences are dense.** Run-on constructions with stacked qualifiers recur, for example the abstract's "In our engine, at that budget and on machines whose link reads at least half as fast as the CPU, caching the experts MIN's greedy schedule keeps pays only if each admitted expert is read once, not twice." Each one is correct but has to be read twice.
7. **The appendix sprawls.**
   - There are 30 appendix pages and a 55-page supplement.
   - Appendix D alone combines the scoring rule, two tallies, the tallies by era, rounds and rentals, job-by-job narratives and failure tables.
   - Table 9 helps; a navigator cannot easily verify a single claim end to end without it.

### What works

- **Fig. 1** makes "read once" versus "read twice" concrete. It is the single most useful device in the paper.
- **The question-style section titles and Section 9's numbered rules** give the paper a clear argument line.
- **The running example** (Ryzen 9 9950X: 10.0 / 13.0 / 20.6 / 15.2 / 13.8 ms) anchors the yardsticks. I reproduced it from job 096a.
- **Table 4** separates "measured" from "attributed" rows. That is exactly what a reader needs.
- **Table 9** maps every Table 1 claim to jobs and clause identifiers.
- **Captions state baselines and intervals explicitly** (Figs. 3 and 5, Tables 5 and 6).

### Concrete fixes

- Cut the main text to five configurations: deployed, Dep-1R, MIN-2R, MIN-1R and the read-ahead oracle. Move Few-*, LA-*, Belady-* and the window policies into one appendix section.
- Use one yardstick per section and say it in the first sentence.
- In Table 1:
  - split the Result column into "registered outcome" and "post-hoc additions";
  - show interval-held versus point-held;
  - replace "Mach." with plain words.
- Move the 0.5 line and the two-step publication rule into Section 2 as stated facts.
- Give each intro paragraph one number with its scope, and leave the ranges to the tables.

## Questions for the authors

1. Can MIN-2R be run so that it actually realizes MIN's set? For example: no in-step fetch table, immediate publication, or no queue cap, until its misses approach R* (the authors' own simulation reaches 0–4% of R* at d = 1). What is the interaction then? If you cannot, will you restate rule 2 as a property of this engine's two-read path?
2. If Table 1 required the interval criterion for "held", which rows would change? Why does the per-machine interaction sign have no interval? A paired bootstrap of the interaction in ms is straightforward.
3. Was any output-correctness rule registered for job 106 before launch? If not, will Table 1 mark the Few-1R row's exclusion as post hoc?
4. Why compare the second-card prediction only with "no change"? A constant-gain null (the panel's mean gain) is the natural alternative. By my computation the trend beats it by a factor of about 3.6 in maximum log error; reporting that would strengthen the claim. Please also report the trend's out-of-sample error pooled over every new machine (5 RTX 5090s plus 5 RTX 4090s), not each card separately.
5. How variable is B_host over repeated probes on one idle machine (say, 10 runs)? Can share-of-bound carry a probe interval?
6. Are the 108f and 109f/112g machines distinct physical hosts? Did any other "new" machines share a host build with earlier ones?
7. Does the per-machine bootstrap change if problems are resampled as contiguous blocks, given that cache state carries between problems in a fixed order?
8. Capture is measured against the read-ahead oracle, which reads 1.22 R*, more than MIN-1R's 1.04 R*. Would capture against MIN-1R, or against Eq. (2), change the "little is reachable" conclusion?

## What would raise my score

- **Fix the 2×2 or narrow it.** Either an arm in which MIN's set is realized on the two-read path, or an honest restatement of the interaction as an engine-path effect, with the T3 "(held)" removed from Table 1 or marked uninformative.
- **Harden Table 1's labels.**
  - Show interval versus point outcomes.
  - Flag every loose or post-pilot threshold, including the system comparison and the 1.10× bound.
  - Mark post-hoc exclusions.
- **Run a blind second-card replication** on at least 5 new machines spanning ratios 0.3–1.0, scored against both the frozen trend and a constant-gain null.
- **Propagate probe uncertainty** into the bound and the share-of-bound numbers, and use per-machine T_GPU where profiles exist.
- **The clarity restructuring above.** The single most effective change would be to cut the main text's vocabulary to five configurations and one yardstick per section.

## Scores

| Criterion | Score |
|---|---|
| Overall | **6 / 10** (acceptable with revisions) |
| Soundness | **3 / 5** |
| Methodology | **4 / 5** |
| Significance | **3 / 5** |
| Clarity | **3 / 5** |
| Confidence | **4 / 5** |

**Why these scores:**

- **Soundness.** The data, the arithmetic and the registration are sound, and everything I checked reproduces. The interpretation of the core 2×2 and the "derived" status of a probe-relative bound overreach, and that limits soundness.
- **Methodology.** The methodology is close to exemplary: registration before rental, gates, the machine as the unit, a noise audit and failures reported. It is held back by small samples per test, loose thresholds and post-hoc exclusions.
- **Significance.** The bound-plus-oracles framing is a useful template for MoE offloading work. The empirical scope (one engine, batch 1, mostly one model at two budgets) limits its reach.

## Claims checked against raw data

All values are recomputed by my scripts in `scratchpad/rev24b/` from the per-problem rows, counters, probes and profiles. Unless stated otherwise, "11%" means gpt-oss C = 14 and "25%" means C = 32.

| # | Claim (location) | Paper | My recomputation (raw source) | Verdict |
|---|---|---|---|---|
| 1 | Link-to-CPU ratios of job 109 machines (Table 6) | 0.54, 0.59, 0.84, 0.88, 0.89 | 0.54, 0.59, 0.84, 0.88, 0.89 (`concur.txt`, B_p/B_c at the helper count) | matches |
| 2 | Table 6, RTX 5090 rows: MIN-1R, ahead, best none, capture at 11% and 25% (9 machine-budgets) | e.g. 285K 1.22 / 1.46 / 1.02 / 4%; 7945HX at 25%: 1.28 / 1.63 / 0.99 / −1% | 1.215 / 1.460 / 1.021 / 4.5%; 1.283 / 1.626 / 0.995 / −0.8%; all 9 rows match (`ec_g_C*_r*{A,B}.jsonl`, geometric mean over rounds) | matches |
| 3 | 5950X 25% round cut by the deadline | "–" | C32 r1A has 78 rows (pf 18 of 20), r1B skipped | matches |
| 4 | Table 4, New (5) at 11%: MIN-2R / Dep-1R / both / ahead / left | 3[−1,8] / 0[−2,1] / 35[22,48] / 19[14,24] / 46[38,55] | 3.4[−0.9,7.7] / −0.4[−2.2,1.4] / 34.7[21.9,47.5] / 19.0[13.9,24.2] / 46.3[37.7,54.8] | matches |
| 5 | Table 4, New (4) at 25% | 20 / −2 / 27 / 24 / 48 | 19.8 / −2.4 / 27.3 / 24.4 / 48.3, intervals match | matches |
| 6 | Interaction > 0 on all 5 new machines at 11% | 5 of 5 (held, point) | +1.53 to +7.49 ms; both rounds positive on every machine; 4 of 4 at 25% | matches (no interval exists) |
| 7 | Job 099: interaction positive on 9 of 10 at 11% | 9 of 10 (failed) | 9 of 10; 099f (285K, ratio 0.29) at −0.48 ms | matches |
| 8 | Table 4, Panel (15) at 11% and 25% | 11%: 0/0/33/15/52; 25%: 21/−3/31/21/48 | 0.5/0.0/32.6/14.9/52.5; 20.6/−3.2/31.3/20.6/48.1, intervals match (machine set rebuilt by GPU UUID from jobs 096–101) | matches |
| 9 | Panel "both" share restricted to fast links (not in the paper) | – | 38.8% [31.0, 46.6] on 13 machines, against 32.6% on all 15 | scoping inconsistency (Clarity 5) |
| 10 | Frozen RTX 5090 trend in job 110's header | e.g. fetch at 11%: 0.3053 + 0.3184 ln r | refit gives identical coefficients for all 8 lines; largest residuals 0.081 and 0.094, as stated | matches |
| 11 | RTX 4090 rows of Table 6 (jobs 111, 112) | e.g. 14900KF 0.96 / 1.13 / 1.03 / 26% | 0.963 / 1.128 / 1.033 / 26%; all 10 rows match | matches |
| 12 | RTX 4090 deviation from the trend | within 7% at 11%, 5% at 25% | 6.6% and 5.0% | matches |
| 13 | New RTX 5090s against the trend | 5 of 18 cells outside, up to 18% | 5 outside (none among the Dep-1R/MIN-2R cells), worst 17.6% | matches |
| 14 | Constant-gain null for the RTX 4090s (not in the paper) | only "no change" reported | trend max \|log err\| 0.064 against 0.230 for the constant model | supports the claim more strongly than the paper's null |
| 15 | Job 110 first rounds; 110e and MIN-2R deviations | 110e −5.6%; MIN-2R 6.4% (0.062 log) | 110e read-ahead −0.057 log; 110c MIN-2R +0.062 log | matches |
| 16 | Job 110 gate and V1 | ratio gate stopped one host at 0.2499; loss 3% high | 110a ratio 0.2499; 110b–e loss 0.1963–0.1965 against 0.190 | matches |
| 17 | Table 5 (job 112): speeds and misses at 11% on 4 machines | e.g. 5950X 1.41 / 1.03 / 0.98; misses 40.7 / 55.6 / 54.5 | 1.406 / 1.030 / 0.977; 40.7 / 55.6 / 54.5; all 4 rows match | matches |
| 18 | T1 and T2 failures | 5 of 8; 7 of 8 | 5 of 8 (miss ratio 0.968–0.980 at 11%); 7 of 8 | matches |
| 19 | T3 "held" | held | Holds trivially: early − Few-2R = −0.031 to −0.053, so the right-hand side is negative | uninformative (Weakness 1) |
| 20 | Early-copy admissions and reads | 11.7–11.8 against 6.2–7.4; +4–7% reads | 11.72–11.75 against 6.20–7.37; +4.2–7.1% | matches |
| 21 | Second probe | within 4.9% on 4 machines | +0.8, +4.9, −0.4, −0.3% (`concur2.txt`) | matches |
| 22 | Relaunch: deployed time and ratio | within 0.7%; ratio moved up to 12% | +0.42% / −0.71%; ratio 0.888→0.883 and 0.837→0.738 (−11.8%) | matches |
| 23 | Registration timing | 92 rentals; every commit ≥ 2.7 s before its rental | 92 rentals; minimum lead 2.7 s (105f); 6 mid-rental commits, all host substitutions with "predictions unchanged"; no post-run edits | matches |
| 24 | No-foresight reads and Margin's cut (Section 6) | 1.57–1.76 R*; Margin cuts reads 4–6% | 1.57–1.76 R*; 4.3–4.4% on the job 109 machines | matches |
| 25 | Best no-foresight speed and capture | 1.007–1.024×; at most 6% | 1.007–1.024×; at most 5.7% | matches |
| 26 | RTX 4090 no-foresight variants | layer-ahead copy 0.64–0.72×; Margin 1.033–1.041×; capture up to 26% | 0.644–0.724; 1.033–1.041; 26% | matches |
| 27 | MIN-2R's reads at 11% (construct check, not in the paper) | implied "MIN's set" | 1.73–1.74 R* (misses 1.27–1.31 R*), against deployed 1.65–1.66 R* and MIN-1R 1.04 R* | contradicts the "set alone" reading (Weakness 1) |
| 28 | Share of bound | new machines 41–48%; consumer 31–54%; server 12–46% | 41.5–48.0%; 31.2–54.0% over 30 consumer GPUs (25 + 5); server 12.4–46.1% | matches |
| 29 | Table 3, host B rows (job 081) | gpt-oss 11%: 34.9 / 54.0 / 69.9, ratio 1.29 [1.28, 1.31], bound 172 tok/s, 41% | 34.9 / 54.0 / 69.9, 1.294 [1.278, 1.311], B_host 87.5 GB/s → 172.3 tok/s, 40.6%; 25% and 40% rows match | matches |
| 30 | Table 3, host S cells (job 089) | 28.9 / 48.0 / 57.9; Qwen3 43.75%: FreeToken 98.3, ours 95.8 | identical | matches |
| 31 | Table 29 microbenchmark (job 102) | per layer 86–96%; repetitions within 0.9% | 86, 89, 96, 96, 95, 93%; repetitions within 0.88% | matches |
| 32 | T_GPU and G (Nsight) | T_GPU 2.9 ms (smallest of 14); G 4.1–4.7 ms | 2.940 ms smallest of 14 (but 109d's own profile is 2.905); G 4.05–4.67, two EPYC traces empty | matches (note on T_GPU) |
| 33 | Running example (Section 5) | 10.0 / 13.0 / 20.6 / 15.2 / 13.8 ms | 10.02 / 12.96 / 20.64 / 15.16 / 13.81 (job 096a) | matches |
| 34 | Eq. (3) over Eq. (1) | median 1.00; up to 1.56 | median 1.00 over 46 consumer launches; maximum 1.557 (TR 9960X) | matches |
| 35 | Round-to-round noise | 98 configurations on 12 machines; median 0.2%, at most 1.8% | 98 on 12; 0.17%, 1.74%; median interval half-width 0.68% (paper 0.6%) | matches |
| 36 | Few-1R on 13 stable machines | 1.24× (1.16–1.32); 2 RTX 4090s, 2 server CPUs; lost on 2 unsteady machines (ratios 0.14, 0.21) | 1.240× (1.163–1.323); losers 106c 0.845, 107e 0.974; the stability rule also drops 106d (1.30) and 107c (1.18) | matches (Weakness 2 note) |
| 37 | Same-CPU machines differ in deployed time | up to 29% | 29.1% (285K: 099a against 099f) | matches |
| 38 | Exact 16-token window in the engine at 11% (job 099) | 0.78–0.93; lowest machine missed 0.80 | 0.78–0.93 (099f 0.78) | matches |
| 39 | Validity gates of jobs 107 and 108 | 107: 7 rented, 3 gated, 2 failed rounds; 108: 6 rented, 2 gated, 1 failed | 107a/f/g gated (V0), 107c 3.54% and 107e 5.98% failed V2; 108c/e gated, 108b failed (7.01%) | matches |
| 40 | Audit in-class median | 13.6% over 20 rows | 13.6% (median of Table 31's 20 trace rows excluding Fate, SP-MoE and MoE-SpeQ) | matches (from the table, not raw) |

## IEEE format

I reviewed `ieee-paper.pdf` (10 pages, IEEEtran conference) and `ieee-supplement.pdf` (29 pages, one column). Both builds have 0 errors, 0 overfull boxes and no undefined references. Each log has 30 "multiply defined" warnings from xr importing the other document's citation labels; they are harmless but noisy. The issues against IEEE conference conventions:

- **Page count.**
  - The text ends on p. 8 and the references run to p. 10, so the paper is 10 pages including references.
  - That fits venues allowing 10 pages including references. It exceeds 8-page limits unless references are excluded, so check the target venue's rule.
- **Front matter.**
  - The title, anonymous author block, abstract and Index Terms are present, and the Index Terms are alphabetical.
  - The abstract contains math ($C$, "0.65 C"). IEEE discourages symbols in abstracts.
  - The PDF metadata keywords ("Mixture of Experts, Offloading, Caching, Performance Bounds") differ from the Index Terms.
  - The unnumbered first-page footnote names a file ("ieee-supplement.pdf"). Write "the supplementary material" instead.
- **Captions.**
  - Paper table captions are IEEE small caps above the table, which is correct.
  - Table I's caption is a multi-sentence definitional paragraph in small caps, which is very hard to read. IEEE practice is a short caption with the definitions in a note below.
  - The supplement deliberately restyles table captions into body font, so the two documents do not match; IEEE practice would keep small-caps captions.
  - Figure captions ("Fig. n.", below) are correct.
- **References.** These are numbered, sorted and compressed IEEEtran references, as required. Some entries:
  - carry two years (for example "[28] … (MLSys), 2026, arXiv:2506.20675 (2025)");
  - give software as a bare URL ([16]).

  The supplement has its own reference list, which restarts at [1], so "[1]" means different works in the two documents.
- **Float placement.**
  - Table I (full width) is at the top of p. 2, which is fine.
  - Table III is first cited in Section II on p. 2 but appears on p. 5. Fig. 2 and Table IV appear one page after their first citation.
  - Neither is a violation, but the three-page lag for Table III is long. Moving the citation or the float would help.
- **Cross-references between the two documents.**
  - These resolve correctly in both directions:
    - Appendices C–N and Tables S21–S23 from the paper.
    - Sections, Tables I–VI and Eqs. (1)–(2) from the supplement.
    - The closed-form relation renumbered as Eq. (S1).
  - The IEEE supplement, titled "Supplementary Material", still refers repeatedly to "the supplement" and to "paper/supplement.pdf". That is a third document, given as a repository path. Inside the supplementary material this is ambiguous: one example is the appendix map's bullet "The supplement: every clause scored …".
  - Either merge the clause tables into the IEEE supplement or call the third document something distinct (for example "the clause tables") and cite it without a repository path.
- **Anonymity.** Both builds mention "a public branch" and give repository paths. That is fine for single-blind review. For double-blind review, make sure the branch is anonymized.
