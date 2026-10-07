# Review 13b: "Where the Seconds Go" (professor, systems and scientific benchmarking)

Date: 7 October 2026. Reviewer stance: referee for a fellowship and a systems/ML workshop, in the spirit of Hoefler and Belli (SC'15).

## Materials read

- `paper/paper.pdf`: the whole main text (pp. 1–9: abstract, Sections 1–9, Tables 1–4, Figures 1–4). Appendix A (checklist) and Appendix B (prediction log) in full. Appendix C (the calibrated time model), Appendix D (grid), Appendices E and F (accounting) read through. Appendix J (`app_more.tex`: jobs 104–108, every host) in full. Appendices G–I and K–M skimmed for structure only.
- LaTeX: `paper.tex`, `tab_regtests.tex` (Table 3), `tab_job107.tex` (Table 21), `tab_job108.tex` (Table 22), `app_more.tex`, `supplement.tex`. Also the generated macro files `wsg_job108.tex` and `wsg_sumlaw.tex` (part), read to map each macro to its printed value. I checked `paper.log` for overfull boxes.
- `paper/supplement.pdf`: the front matter, the job 073–075 scorecard rows, and Table 11 (job 108 scorecard).
- Artifact. I read only the scripts needed to learn definitions:
  - `scripts/job105.py` (head);
  - `scripts/job107.py` (`machine_classes`, `INVALID`);
  - `scripts/job108.py` (`gate`, `fallback_G`, the family analysis);
  - `scripts/speed_limit.py` (`host_rates`);
  - `gpu-branch/jobs/ec2/fetch_table.py` (`bandwidths`);
  - `prereg/reanalysis.json` (the profile-median G0 only);
  - `gpu/vast_ledger.json`.
- Raw data, second checkout `/home/claude/gpu-branch`:
  - headers and per-host wrappers of `jobs/105–108*.sh`;
  - `git log --format='%h %ad|%ct'` and `--name-status` (no messages), and `git diff` of the four job scripts across their amendment commits;
  - `results/093…108*/`: `ec_*.jsonl` (decode_ms, n_decode, nll_sum, nll_n, config/stats path), `st_*.json`, `concur.txt`, `g_prof.json`, `q_prof.json`, `prof_summary.txt`, `v0.txt`, `validity.txt`, `free.txt`, `cores.txt`, `numa.txt`, `gpu.csv` and `nvidia-smi-q.txt` (GPU UUIDs, drivers);
  - `results/102?_readsched@vast/readsched_*.txt` and `results/081_headline_law@vast/bs1.jsonl`.
- My own analysis code is in `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/rev13b/`:
  - `ana.py`: loaders, Eq. 3 solved as a quadratic, implied read rate, paired bootstrap;
  - `check_law.py`: jobs 105–108, every cell, every form;
  - `families.py`: the post-hoc consumer/server analysis;
  - `ratios.py`: round-then-problem bootstrap;
  - `minadm.py`, `found.py`, `elast.py`, `c108a.py`, `regtime.py`, `density2.py`.

## Independence statement

- I did not open anything under `reports/` (I listed the directory once to confirm it exists, and wrote this file). I did not open `prereg/*outcome*.md`, `research_notes/`, any file named like a review, number check, plan or progress log, `paper/paper_v1_prereview.tex`, `paper/paper_v2_prerewrite.tex` or `apply/`.
- I read no commit messages. I used only hashes, dates, changed-file lists and file diffs.
- Where the artifact's scripts define a quantity, I read the definition: B_host is the maximum probe reading. B_c is the CPU rate interpolated at the helper count. B_p is the 16 MB zero-copy rate. The fallback G is the median of eight profiles from jobs 105–107. The family analysis uses the first round of each launch.
- Every number in the claims table was then recomputed by my own code from the raw files.
- I modified nothing in either repository except this file, and committed nothing.

---

## Summary

The paper studies batch-1 decode of two MoE models (gpt-oss-120b in MXFP4, Qwen3-30B-A3B in BF16) on rented RTX 5090 hosts, with most experts in host DRAM and a per-layer GPU expert cache built into llama.cpp. It has four parts.

1. **Two lower bounds.** It derives two lower bounds on time per token from the host reads of Belady's MIN with bypass and the host's best probed read rate. One bound is fully overlapped (Eq. 1). The other is for systems that read on demand, which pay GPU work in series (Eq. 2). A microbenchmark reaches 86–96% of the bound's read time.
2. **An accounting relation, Eq. 3.** It proposes T = G + (M + (1 − G/T)A)·S/B_host for the deployed cache, with no constant fitted to the times. The relation was found on 30–34 launches, then registered four times (jobs 105–108). It failed all four registered tests. The authors then split hosts after the fact by processor family: on "consumer" processors the relation is within 8% on 37 of 39 launches, and on "server" processors the engine reads at 0.27–1.04 of the probe's rate.
3. **Oracles in the engine.** Oracles show that foresight pays when spent as MIN spends it (one read per admission). Among MIN's hit-optimal schedules, the one with the fewest admissions gains on every stable machine, while the greedy one loses behind slow links. An online rule ("admit less", a larger admission margin) gains a median of 1.02×.
4. **A horizon rule.** From traces, half of MIN's saving needs about 0.65·C distinct experts of look-ahead per layer.

The artifact is unusually complete. Every number is generated by a script, every job's predictions sit in a script header committed before launch, and every rented host is in a ledger.

## Strengths

1. **Registration is real, and it is checkable.**
   - For jobs 105–108, the prediction-bearing header was committed 3 s to 39 min before each machine's ledger start. Each per-host wrapper was committed 3–32 s before.
   - Every commit to a job script after a machine started changed header comments only: host replacements, labelled as amendments. I diffed all of them. No prediction or gate changed after a measurement.
   - This is the practice Hoefler and Belli ask for and rarely see.
2. **Negative results are reported as negative.** Table 3 says "failed" four times, and the abstract says the relation "never passed as registered". The 106 server hosts, the 107 inconclusive count and the 108 EPYC are all reported, with their numbers. Few systems papers do this.
3. **Validity gates are defined before launch and are reported in full.**
   - V0: GPU UUID not seen before, host memory in use, device read rate, disk. V0c: the class. V1: the loss check. V2: the round spread.
   - Every gated or invalid host is listed with its reason, and I reproduced each one: 50/50/96 GB and 323 GB in use, one GPU UUID seen in job 100, round spreads of 3.54%, 5.98% and 7.01%.
4. **The statistical unit is argued, not assumed.**
   - The paper shows that same-model hosts differ by 16–29% while relaunches differ by at most 3.2%, and makes the machine the unit.
   - Round-level intervals (resampling rounds, then problems) were added for the small effects, and they reproduce exactly.
5. **The oracle study is a genuine contribution.**
   - The fewest-admission schedule beats the deployed cache on all 10 stable launches (1.02–1.34× at gpt-oss 11%).
   - It beats greedy MIN by 0.08–0.17 behind slower links, while greedy MIN loses there (0.85–0.93×).
   - These results are robust in the raw data and are the kind of mechanism-level evidence offloading papers lack.
6. **The numbers are correct.** I made 39 checks against raw rows (table at the end). 33 reproduce to the printed precision, two of them with caveats on scope. Six do not hold as stated:
   - a wrong qualifier in the abstract (row 3);
   - an overgeneralisation (row 4);
   - an inconsistent definition (row 11);
   - a ledger fact (row 25);
   - the number of problems (row 38);
   - a clipped table (row 39).

   None is a fabrication or a computational error.

## Weaknesses (most important first)

### W1. The abstract and conclusion lead with a post-hoc result, and overstate it. Under the registered criteria, Eq. 3 also fails on the new consumer machines.

- **The headline.** The abstract's first result is "within 8% on 37 of 39 launches, including every cell of the three such machines first rented for a registered test." Four things are wrong with it:
  1. **Its scope is unstated.** The count is at gpt-oss 11% only, and the abstract does not say so.
  2. **It mixes discovery and test data.** 30 of the 39 launches are the discovery set (jobs 093–104) on which the relation was found. Three more (job 105) are among the 34 launches on which the overlap term was chosen. Only 6 of the 39 came after the form was fixed.
  3. **It counts launches, not machines.** The 39 launches are on 25 machines; Pf alone contributes six. The paper's own statistics paragraph makes the machine the unit.
  4. **The 8% band is post hoc.** It was chosen after job 107. The 108 header says the band is "the spread of the launches it was found on" and cites the 5900XT's −7.1%. The registered band in 105–107 was 6%.
- **"Every cell" is false.** The 5900XT's Qwen3 12.5% cell is 48.3 ms predicted against 54.4 ms measured, −11.3%, as printed in the paper's own Table 21. The introduction correctly says "every gpt-oss cell"; the abstract does not.
- **The new consumer machines fail the registered criteria too.** The three consumer machines new to a registered test are the 5900XT, the 9950X3D and the 5950X.
  - All 8 of their cells under-predict, by −0.9% to −11.3%. In order: −7.1, −2.3, −11.3, −6.4, −6.9, −4.0, −6.5, −0.9.
  - In job 107, the 5900XT alone misses the registered 6% band at 3 of 4 cells, and its median is 6.75% against ≤4%.
  - In job 108 the consumer-only subset still fails two of the three registered relation clauses. Two cells are beyond 6% (6.9 and 6.5) against at most one allowed, and the median is 5.3% against ≤4%.
  - So Table 3's outcome label "failed (8% band; EPYC)" wrongly attributes the 108 failure to the EPYC alone.
- **The server claim is overgeneralised.** "On server processors … the relation fails" is false for one of the three valid server launches. On the EPYC 7402P (job 105) the overlap form is +3.0% and +4.0%, and the implied read rate is 1.04. The Limitations section gets this right ("not on two of the three").
- **The conclusion is too strong.** "Batch-1 MoE decode on machines with consumer processors is close to the GPU's compute plus host bytes" states as a finding what is, at best, a post-hoc hypothesis. Out of sample, that hypothesis shows a consistent negative bias.

### W2. Eq. 3 is not identified by the data, and "nothing fitted" hides form selection.

- **The overlap term was added after the fact.** It came in after the plain form failed at 25% in job 105. It moves the discovery-set fit from 18 to 33 of 34 launches within 6% at 25%, and from 32 to 29 at 11%. I reproduced both.
- **The registered test cannot tell the forms apart.**
  - On job 106's stable hosts, discounting every admission by half passes 11 of 12 cells, and counting misses alone passes 10 of 12.
  - In job 107, the registered advantage of the overlap form over the plain form failed: on the desktop the plain form's median is 6.0% against 6.8%.
- **Compensating errors are possible.** Eq. 3 treats CPU-served misses as fully serial with G. Yet Appendix C says a single CPU-served miss is "hidden behind the GPU's other experts", and per-layer expert GPU work is about 37 µs × 36 layers ≈ 1.3 ms/token. Part of G may therefore overlap with reads. The "implied read rate ≈ B_host" interpretation (0.88–1.16) cannot separate a slower read rate from unmodelled overlap.
- **The interpretation is sensitive to G.** The count of launch-budgets that "read faster than the probe" is 13 of 68 with G = 4.28/4.50 ms. Lowering G at 25% by 0.7 ms, to 3.8 ms, cuts the count to 2.
- **Conclusion.** No constants are fitted, but the functional form, the source of G and the choice of B are all selected. A 6–8% band is only about 2–2.5× the 3.2% relaunch variation of a single machine.

### W3. The processor-family explanation is post hoc, rests on three valid server launches, and is confounded.

The paper says "What does separate them … The processor." Several other factors covary with processor family in the raw data:

- **Co-tenancy.** Host memory in use at the gate was 8–39 GB on the server hosts of jobs 107–108 (108a: 34 GB, 107e: 39 GB) and 131–132 GB on the two servers of job 106. On the three new desktops it was 3–4 GB.
- **Dual-socket NUMA.** The engineering sample has two NUMA nodes and runs 126 helpers. Its probe reads 151 GB/s at 16 threads and 118 GB/s at 128. Even at 117.6 GB/s, Eq. 3 still misses by 39–43%; I computed this, and it is not in the paper.
- **Container core limits.** The cgroup limits differ: 23–30 usable cores on 48- and 32-core EPYCs.
- **Driver.** Both job-108 EPYC hosts ran driver 570.x; the desktops ran 580 and later.

With one counter-example among the three valid servers (the 7402P), "the processor" is a label, not a cause. The paper says "we have not established the cause", but its section heading and abstract claim more than that.

### W4. The bound is relative to one probe run; its reachability is shown only on consumer hosts.

- **B_host is a single maximum.** It is the maximum of a single probe run. On the engineering sample, that maximum (204.5 GB/s) is one concurrent reading in which the CPU threads read 181 GB/s beside PCIe, against 118 GB/s alone at the same thread count. That is a suspect outlier, and the bound inherits it.
- **The engine beats the "lower bound".** By the paper's own accounting the engine read more than 5% faster than B_host on 13 of 68 launch-budgets, up to 1.22×; 12 of those 13 are at 25%. A bound the system exceeds is a probe-referenced reference, not a lower bound. The Limitations section admits this, but Section 3 and the Contributions do not.
- **Reachability was tested only on consumer CPUs.** The 86–96% figure comes from three consumer hosts (13900KF, 9950X, 9800X3D). There is no evidence that the read time is reachable on the servers, where the engine reads at 0.27–0.58 of B_host.

### W5. Clarity remains the main barrier to acceptance (detailed below).

The paper is still written as a chronological lab notebook of jobs 093–108. The reader must track:

- 10 job numbers and 35 job mentions in the main text;
- 7 host nicknames (B, S, O3–O5, Pe, Pf);
- 11 named configurations, three GPU-time quantities and six unit words (launch, round, cell, host, machine, panel).

### W6. Registration and reporting details (each minor; together they matter for a paper whose selling point is the record)

- **Self-attested timestamps.** Registration rests on the author's git clock and a ledger the author wrote. Nothing anchors it externally (pushed-event logs, OSF, signed timestamps). The internal consistency is good; I found no contradiction.
- **The job 108 header is inaccurate about the rental ledger.** It says the offers were "none in the rental ledger", and the paper says "machines never rented before". But offer 54156078, host 108d (the Ryzen 9 9950X3D), is in the ledger as the first 099a rental, on 2026-10-06 from 07:19 to 07:23 UTC. That rental was destroyed after 4 minutes with no data. The impact is nil, but the statement is false, and the V0 gate checks GPU UUIDs only.
- **Job 107's replacement rule changed mid-job.** The registration replaced only V0 failures. Replacing V2 failures, and the rules for choosing hosts f and g, were added during the job, after the first four results. The paper discloses this, and the amendments were committed before hosts f and g started.
- **The problem count is misstated.** The Setting says "the 30 AIME-25 problems". Every registered job from 099 to 108 ran 20 problems: `n=20` in every `ec_*.jsonl`. Only an appendix says so.
- **Table 22's caption is wrong for two rows.** It says "G profiled on the same machine before the timed runs" for every row. Both EPYC rows (7543 and 7K62) use the fallback G, because both Nsight traces were empty under driver 570. The main text mentions only the 7543; Appendix J mentions both. The deviation does not matter to the outcome: the 7543 would need G ≈ 10.6 ms, 2.3× the largest profile, to come within 8%.
- **Round spread has two definitions.** The job 106 Xeon's spread is printed as 39%, which is range over mean. The V2 definition registered in 107–108, max/min − 1, gives 52.2% (rounds of 14.52, 22.09 and 21.93 ms).
- **Problem-level intervals ignore cache carry-over.** The bootstrap treats problems as independent, but the cache carries over between problems. Round-level intervals mitigate this for 106–108 only.

---

## Clarity: specific problems and fixes (prioritised)

Earlier rounds scored clarity 2/5. The paragraph headings ("Registered on new machines, it failed") are a real improvement, and the honesty is exemplary. But the reading load is still too high. In priority order:

1. **The abstract must be rewritten.**
   - *Problem:*
     - It packs four results into about 270 words and introduces about 15 undefined terms: MIN with bypass, consumer processors, best probed rate, launches, cells, serialisation, admissions, "admit less", distinct experts.
     - It leads with the post-hoc 37/39 result and only then says that registration failed. It contains the false "every cell".
     - It has sentences a reader must parse twice: "Registered in advance four times, it never passed as registered"; "Carried online, that lesson, admit less, gains up to 6%…".
   - *Fix:* use five sentences:
     1. Problem and setting.
     2. The bound, and that its read time is 86–96% reachable on consumer hosts.
     3. "We proposed an accounting relation (time = GPU compute + host reads at the probed rate) and registered it four times; it failed all four, by small margins on desktop CPUs and by 28–59% on server CPUs (a split we drew after the tests)."
     4. Foresight spent as MIN spends it, with the fewest-admission finding.
     5. The horizon of about 0.65·C distinct experts.
   - Use six numbers at most, and say "post hoc" in the abstract.
2. **Section 4 tells the story in the order the jobs ran.**
   - *Problem:* the reader follows the relation through plain form → overlap form → 6% → 8% band → all hosts → valid hosts → "desktop class" → "consumer processors" across six paragraphs. Discovery, test, re-test and post-hoc analysis are interleaved.
   - *Fix:* use four labelled subsections:
     - 4.1 The relation: derivation and assumptions, including what it ignores (within-layer overlap).
     - 4.2 Registered tests: one table, one row per registered clause, with criterion, measured value and outcome. This is Table 3 expanded to show both the 6% and the median clauses, and the consumer-only subset.
     - 4.3 Post-hoc analysis (not registered): the family split, with its confounds.
     - 4.4 Decomposition, with an explicit statement that it presumes Eq. 3.
   - Move the rental, gate and amendment narrative to Appendix B.
3. **Mark the status of every claim.**
   - *Problem:* registered, exploratory and post-hoc results are distinguished only by phrases scattered through the sentences ("as registered", "after the fact", "against the registered prediction").
   - *Fix:* tag each numeric claim [R✓], [R✗], [E] or [P], in text, table rows and figure captions. Add a half-page "claims ledger" table to the main text listing the 8–10 headline claims with their status and the job that tests them.
4. **The vocabulary is too large.**
   - *Problem:* in the main text the reader meets:
     - 35 job references (10 distinct numbers plus ranges);
     - 7 host nicknames;
     - 11 configuration names, whose appendix tags differ again (fetchplan, bypassplan, foa, dk, lrn, pf);
     - "relation" in the text but "law" in Table 4's header, with "law tables" belonging to the different Eq. 4 model;
     - six unit words (launch, round, cell, host, machine, panel);
     - three synonyms for valid (valid, stable, ran stably).
   - *Fix:*
     - one term per concept, with a notation box in Section 2;
     - experiments named E1–E5 by purpose ("bound check", "relation test 1–4", "oracle factorial", "online rule"), with a job map in the appendix;
     - hosts named by CPU plus a short ID, from one host table;
     - configuration names identical across Table 1, the figures and the appendix tables.
5. **Three different "GPU times".**
   - *Problem:* T_GPU, the non-expert GPU work in Eq. 2, is 2.9–3.4 ms. G, all GPU kernels in Eq. 3, is 4.1–4.7 ms. The "implied G" values are 2.7, 7.4 and 16.4 ms. Appendix C adds a fitted G of 5.17 ms. Readers will conflate them.
   - *Fix:*
     - Give them distinct symbols (for example T_ne, T_kern and T_resid) and drop "implied G".
     - Define B_c, B_p and B_host operationally in Section 2: B_c is CPU read at the engine's helper count (interpolated); B_p is zero-copy with 16 MB chunks; B_host is the maximum over all probe readings. The 0.35/0.40 "link-to-CPU" crossovers depend on these choices.
6. **Too many numbers per sentence.**
   - *Problem:*
     - Section 4 carries about 140 numeric values in about one page of text, and 35% of its sentences or clauses have five or more numbers.
     - The worst sentence (job 107, "It failed nonetheless: …") has 13 numbers. "Which of MIN's schedules is followed matters" has more than 20 in one paragraph.
   - *Fix:* at most about 3 numbers per sentence. Ranges and per-host values go in tables; the sentence states the comparison and its direction. For example, replace the job 107 paragraph with a 2-row table (5900XT, engineering sample) and two sentences.
7. **The running example arrives too late.**
   - *Problem:* it appears in Section 5. Readers need it in Sections 3–4 to see what "bound", "demand bound", "deployed" and "MIN in the step" mean in milliseconds.
   - *Fix:* introduce the O4 example in Section 2 with one stacked bar: 10.0, 13.0, 20.6, 15.2 and 13.8 ms. Refer back to it in every section.
8. **Figures and tables.**
   - Table 3, the paper's key table, overflows its column. "failed on new machine[s]" and "failed (8% band; EPY[C])" are clipped in the PDF (`paper.log` reports overfull alignments).
   - Table 4's caption runs to 9 lines.
   - Figure 1:
     - it labels 28 rows by CPU model with duplicates (four "R9 9950X"), so a reader cannot tell discovery machines from test machines or find the registered-test hosts;
     - the hatched area, the post-hoc part, is barely visible;
     - it has no uncertainty.
   - Figure 2 omits the fewest-admission configuration, the paper's best oracle result.
   - Figure 3 has 7 series and 3 regression lines on two small panels.
   - *Fix:* resize Table 3. In Figure 1, mark discovery, rented-again and new machines (for example by marker shape) and sort by family. Add "MIN, fewest admissions" to Figure 2. Cut Figure 3 to the 3 series the text discusses.
9. **Too much in one paper.**
   - *Problem:* bound, accounting relation, oracle factorial, online rule, horizon rule, forecasters, system comparison and literature audit together make 9 main pages, 28 pages of appendices and a 41-page supplement.
   - *Fix:* either split it into two papers (bound plus relation; foresight plus horizon), or shrink Section 6 to one paragraph and Table 2 to the essentials. Sixteen pages of appendices would be read; thirty-seven are not.
10. **The contributions list presents a failed hypothesis as a contribution.**
    - *Fix:* reframe it as "a falsified accounting hypothesis, the conditions under which it nearly holds, and the evidence of what breaks it". State the registered failures in the bullet.

---

## Questions for the authors

1. How does Eq. 3 account for within-layer overlap of CPU-served misses with the GPU's work on resident experts? If it does not, how do you rule out compensating errors? Per-layer timers on two hosts would settle this.
2. Please report the Eq. 3 error by machine, split into discovery machines (093–105) and new machines. On new consumer machines all 8 cells and all 3 machines under-predict. Is a bias of about −6% at 11% a better description than "within 8%"?
3. Can you rule out co-tenancy (memory in use at the gate: 8–39 GB on 107–108 servers, 131–132 GB on 106 servers, 3–4 GB on new desktops) and NUMA as the causes of server under-reading? Re-running the probe immediately before and after each timed round would detect contention that the start-of-job probe misses.
4. Why is B_host the maximum of a single probe run? What do the bound and the implied read rates become with the probe at the engine's helper count, or with a median over repeated probes? The engineering sample's 204.5 GB/s reading looks like an outlier.
5. Offer 54156078 (108d) is in the ledger under 099a. Will you correct the 108 header statement, and extend V0 to offer IDs or machine IDs?
6. Can you provide an external anchor for the registration timestamps, such as GitHub push events or an OSF snapshot?
7. The Setting says 30 problems, but jobs 099–108 ran 20. Which tables and figures use which?
8. Did you test whether per-problem times are autocorrelated through the cache that carries across problems? A block bootstrap over consecutive problems would show whether the problem-level intervals are too narrow.
9. The 108 band (8%) was set after observing the 5900XT's −7.1%. In what sense is a test with that band confirmatory?
10. The online rule gains 1.00–1.06× on stable hosts, and relaunches of one machine differ by up to 3.2%. What is the deployment case for κ = 2–3 beyond "it rarely loses"?

## What would raise my score

- **Correct the abstract, the introduction's summary and the conclusion to match Table 3.** Remove "every cell". Say that the consumer/server split is post hoc and has one counter-example. Report that under the registered criteria Eq. 3 also failed on the new consumer machines. This alone would move soundness from 3 to 4.
- **Restructure Section 4 as in clarity items 2–3,** with status tags and an expanded registered-tests table. Fix Table 3's overflow and Table 22's caption. Unify the definitions (round spread, number of problems).
- **Either re-test or demote Eq. 3.** A clean confirmatory test needs a revised hypothesis that includes a read-efficiency term or the within-layer overlap; ≥5 new consumer and ≥3 new server machines; co-tenancy and NUMA recorded and gated; and a band fixed from the discovery set alone. Alternatively, demote Eq. 3 to an accounting identity with an explicit residual, and make the residual the object of study.
- **An identifiability analysis** of the overlap, half-discount and misses-only forms, using per-layer timers.
- **A robust B_host,** with repeated probes and a reported dispersion, and the bound recomputed with it.

## Scores

| Criterion | Score | Basis |
|---|---|---|
| Overall | **6 / 10** (acceptable with revisions) | The measurements are correct and the methodology is unusually honest. The headline is overstated, the main hypothesis failed registration, and clarity is poor. All of this is fixable in writing; none of it needs new experiments, except to rescue Eq. 3. |
| Soundness | **3 / 5** | The data and computations reproduce. Some claims built on them go beyond the evidence: the abstract's "every cell", the server overgeneralisation, a non-identified relation, and a probe-relative "lower bound". |
| Methodology | **4 / 5** | Registration before launch, gates, reporting of every host, the machine as the unit, round-level intervals, and scripts for every number. Points lost for the post-hoc band and class, self-attested timestamps, the ledger error, and inconsistent definitions. |
| Significance | **3 / 5** | The bound and the fewest-admission and oracle findings are useful to the offloading community. The online gains are small, and the accounting relation is not established. |
| Clarity | **2 / 5** | It still reads as a lab notebook. Number density, vocabulary and chronology overwhelm the argument, and the key table is clipped. |
| Confidence | **4 / 5** | I made 39 checks from raw rows. I did not rerun the trace and horizon study (Section 6), the forecasters or the literature audit. |

---

## Claims checked against raw data

"✓" means reproduced to the printed precision. "✗" means not supported as stated. "~" means consistent but defined differently, or approximate. The values in the "Raw-data recomputation" column are my own, computed from `ec_*.jsonl`, `st_*.json`, `concur.txt`, `g_prof.json` and the gate files.

| # | Claim (location) | Paper | Raw-data recomputation | Verdict |
|---|---|---|---|---|
| 1 | Consumer launches, post hoc (Abstract, §4) | 39 valid launches on 25 machines; implied rate 0.88–1.16; 37 within 8%, 31 within 6% (gpt-oss 11%) | 39 launches; 25 GPU UUIDs; 0.880–1.164; 37; 31. First round per launch; G0 = 4.28 ms for 093–104. 30 of the 39 are the 093–104 discovery set, 3 more are job 105, and only 6 came after the form was fixed | ✓ (but post hoc, by launch) |
| 2 | Server launches (Abstract, §4) | 5 of 8 invalid; valid ones read at 0.27, 0.58, 1.04 | 5 of 8 invalid (106c, 106d, 107c, 107e, 108b); engineering sample 0.269, 7543 0.584, 7402P 1.038 | ✓ |
| 3 | "within 8% … every cell of the three such machines first rented" (Abstract) | every cell | 8 cells: −7.1, −2.3, **−11.3** (5900XT Qwen3 12.5%), −6.4, −6.9, −4.0, −6.5, −0.9. True for the 6 gpt-oss cells only | ✗ |
| 4 | "On server processors … the relation fails" (Abstract) | fails | EPYC 7402P (105a), overlap form: +3.0% and +4.0% | ✗ as stated (2 of 3) |
| 5 | Job 105, plain form (Table 3) | 5/8 within 6%; median 4.2%; +6.9 to +9.3% at 25% on 3 hosts (Pf 3.8%); 11% within −1.5 to +4.7% | 5/8; 4.25%; 25%: +9.3, +7.0, +3.8, +6.9; 11%: +4.7, +2.0, −1.5, +1.4 | ✓ |
| 6 | Discovery, overlap form vs plain form (§4) | 34 launches: 33 vs 18 within 6% at 25%; 29 vs 32 at 11%; plain 28/30 at 11%, median about 6% at 25% | 33 vs 18; 29 vs 32; 28/30; 5.9–6.0% | ✓ |
| 7 | Job 106 stable hosts (§4, Table 4) | 12/12 within 6%; median 2.3%; largest 6.0%; plain up to 10.1% | 12/12; 2.35%; +6.0% (TR Qwen3 25%); plain +10.1% | ✓ |
| 8 | Job 106 alternative forms (§4) | half-discount 11/12; misses-only 10/12 | 11/12; 10/12 | ✓ (shows non-identification) |
| 9 | Job 106, all hosts (Table 3) | 12/20; pooled median 4.7%; servers under-predict by 10–52% | 12/20; 4.65%; −10.2 to −52.4% | ✓ |
| 10 | Job 106 wrong-output host (§4) | loss 0.21–0.45 vs 0.19 / 0.085 | 0.208–0.446 (EPYC 7302); others 0.189–0.191 / 0.0855 | ✓ |
| 11 | Job 106 Xeon round variation (§4, Limitations) | 39% | rounds 14.52 / 22.09 / 21.93 ms: 52.2% by the registered V2 definition (max/min − 1); 39% = range/mean | ~ (inconsistent definition) |
| 12 | Relaunch variation (§2) | within 3.2% (job 106) | TR 9960X 9.41 vs 9.12 ms (+3.2%); Pf −1.2%; Pe −0.1% | ✓ |
| 13 | Job 107 gates (§4) | 7 rented; 3 gated at 50–96 GB; 2 invalid at 3.5–6.0%; engineering sample rounds 0.3% apart | 50, 50, 96 GB; 3.54%, 5.98%; 0.28% | ✓ |
| 14 | Job 107, Eq. 3 (Table 3, §4) | under at 8/8; within 6% at 1/8; median 31.7%; 5900XT up to 11% (1/4 within); engineering sample 52–59% | 8/8 negative; 1/8; 31.75%; −11.3%, 1/4; −52.2 to −58.9% | ✓ |
| 15 | Job 107 median floor (§4) | could not be below 6.8% | 6.75% with four zero-error cells added | ✓ |
| 16 | Job 107, plain vs overlap form on the desktop (§4) | plain median 6.0%, 2 within; overlap 6.8%, 1 within | 6.0%, 2; 6.75%, 1 | ✓ |
| 17 | Engineering-sample probe (§4) | best 204 GB/s (one concurrent reading); CPU-only 151 GB/s at 16 threads gives −51%; 118 GB/s at 128 threads | 204.5 (concurrent, t=128, zero-copy); 151.4 → −51.4%; 117.6. At 117.6 the error is still −39 to −43% (my addition) | ✓ |
| 18 | Job 107 admission margin (dk) (§5) | 0.987–1.074, median 1.021; below 0.995 at one cell | 0.987 (engineering sample, 25%) to 1.074; median 1.021 | ✓ |
| 19 | Job 107 fewest-admission set (§5) | in the step 1.16–1.42; by the CPU 1.04–1.13; in-step wins on both; ratios 0.44 and 0.75 | 1.157 / 1.419; 1.126 / 1.038; ratios 0.44 / 0.75 with helper-interpolated B_c and 16 MB zero-copy B_p (not stated in §2) | ✓ |
| 20 | Job 108 gates (§4) | 6 rented; 2 gated (GPU rented before; 323 GB in use); 1 invalid at 7.0% | 108c GPU UUID equals 100b's; 108e 323 GB; 108b 7.01% | ✓ |
| 21 | Job 108, Eq. 3 (Table 3, Table 22) | 6.9, 4.0, 6.5, 0.9; 34.2 and 27.9; 2/6 within 6%; 4/6 within 8%; median 6.7%; implied rates 0.91 / 0.94 / 0.92 / 0.99 / 0.58 / 0.61 | identical | ✓ |
| 22 | Job 108 G deviation (§4, App. J) | 7543's G = median of 8 earlier profiles (4.29 / 4.47 ms); any G in their range gives 26.7–35.1% | 4.292 / 4.469; 26.7–35.1%. Both EPYC traces were empty (driver 570.x). The 7543 would need G ≈ 10.6 ms to come within 8% | ✓ (Table 22 caption inaccurate) |
| 23 | Job 108, consumer machines only (App. J) | median 5.3%, all within 8% | median 5.25%; **2 cells beyond 6%**, so 2 of 3 registered relation clauses still fail without the EPYC (not stated in the main text) | ✓ / omission |
| 24 | Registration before launch (App. B) | committed before each machine started | Main header 3 s–39 min before every 105–108 ledger start; wrappers 3–32 s before. Post-start commits change header comments only (diffs checked) | ✓ (self-attested clocks) |
| 25 | Job 108 offers "none in the rental ledger"; "machines never rented before" | never rented | Offer 54156078 (108d, 9950X3D) is in the ledger as 099a, 2026-10-06 07:19–07:23 UTC, with no data | ✗ (minor) |
| 26 | "Pe again" in job 106 (Table 4) | same machine as panel host Pe | 106e GPU UUID equals 099e's; the amendment named it only "a Ryzen 9 9950X" | ✓ |
| 27 | Elasticity of T − G to B_host (§4) | −0.95 [−1.06, −0.84] | −0.95 over all launches; machine bootstrap −0.97 [−1.08, −0.81]; raw −0.74 | ✓ (interval approximately) |
| 28 | Fewest-admission vs deployed and greedy (§5) | 1.02–1.34× on 10 launches / 7 machines; +0.08–0.17 over greedy; greedy 0.85–0.93 on Pf/TR; fewest 1.02–1.06; round-level CI lower bounds 1.01 / 1.05; by the CPU 1.11–1.22; EPYC 7302 0.82–0.88 per round | 1.019–1.342; 0.082–0.172; 0.853–0.933; 1.019–1.056; 1.013 / 1.049; 1.112–1.217; 0.824 / 0.834 / 0.878 | ✓ |
| 29 | Table 4: 21 speed ratios with round-level 95% CIs | e.g. Pf dk at 11%: 1.02 [1.01, 1.04] | e.g. 1.024 [1.006, 1.038]; all 21 agree to 2 decimals | ✓ |
| 30 | Admission-margin and learned-predictor reads and host time (§5) | dk −4–6% / −6–10%; lrn 7–14% fewer; 267–439 µs; over the cap on Pe | −4.4 to −5.8% / −6.3 to −10.5%; −6.6 to −14.2%; 266.7–439.0 µs; Pe 434.7 / 439.0 µs vs 400 µs | ✓ |
| 31 | Eq. 3 predicts the dk and lrn ratios (§5) | median 0.013; at most 0.045 (Pf, 25%); no-change 0.016 | 0.0135; 0.045 (lrn, Pf, 25%); 0.016 | ✓ |
| 32 | Read-schedule microbenchmark (§3) | 86–96% of the bound's read time | 0.861–0.963 on 13900KF, 9950X, 9800X3D (consumer hosts only) | ✓ (scope) |
| 33 | Cache vs bounds, jobs 093–104 at 11% (§3) | 38–54% of Eq. 1; 55–70% of Eq. 2 | 38.5–54.3% (R* = 38.505 from job 102); 55.6–70.9% with T_GPU ≈ 3.0 ms | ✓ / ~ |
| 34 | Engine reads faster than the probe (Limitations) | 13 of 68 launch-budgets, up to 1.22× | 13 of 68 with G0 = 4.28 / 4.50 ms (12 at 25%); max 1.223. Only 2 with G = 3.8 ms at 25% | ✓ (G-sensitive) |
| 35 | Table 2, host B, gpt-oss 11% / 25% | ours 69.9 / 109.2; FreeToken 54.0 / 85.6; llama.cpp 34.9 / 40.0; ours ÷ FreeToken 1.294 [1.278, 1.312] and 1.275 [1.254, 1.295] | identical (launch 2; llama.cpp from launch 1) | ✓ |
| 36 | Extra misses of the fewest-admission schedule (Limitations) | 2.5–5.3% more than greedy | +2.5% at 11%, +5.1–5.3% at 25% (12.4% on the wrong-output host) | ✓ |
| 37 | Layer-ahead copy (§6) | 1.04× where the link ≈ CPU rate; down to 0.72× | 1.042 (O4); 0.722 (Pf) | ✓ |
| 38 | Setting: "on the 30 AIME-25 problems" | 30 | every `ec_*.jsonl` of jobs 099–108 has 20 problems (093–098: 30) | ✗ (main text) |
| 39 | Table 3 rendering | – | Outcome column clipped in the PDF ("machine[s]", "EPY[C]"); overfull alignment in `paper.log` | ✗ (typesetting) |

### Registration timeline (UTC, from git `%ct` and `vast_ledger.json`)

| Job | Prediction header committed | First machine start | Replacement commit → that machine's start |
|---|---|---|---|
| 105 | 22:05:31 (6 Oct) | 22:05:36 | 22:09:26 → 105f/g 22:09:28–29 |
| 106 | 01:43:48 (7 Oct) | 01:43:58 | 01:44:13 → 106e 01:44:16 |
| 107 | 05:17:31 | 05:17:44 | 05:24:30 → 107e 05:24:36; 06:36:28/06:36:57 → 107f 06:37:00; 06:41:16 → 107g 06:41:20 |
| 108 | 07:39:32 | 07:39:36 | 07:45:57 → 108d 07:46:00; 08:15:49 → 108e 08:15:58; 08:18:27 → 108f 08:18:30 |

Every replacement in job 108 followed the registered offer order:

- c failed V0, so d was taken;
- b failed V2, so e was taken;
- e failed V0, so f was taken.

Job 107's replacements of V2 failures were an amendment, as the paper says.
