# Review 18b: "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth"

Reviewer role: professor in a systems group that works on scientific benchmarking (Hoefler and Belli, SC'15), refereeing for a fellowship and a workshop. Date: 7 October 2026.

## Materials read

- `paper/paper.pdf` (43 pages): the whole main text (Sections 1 to 10) and all appendices A to O. I read the LaTeX source alongside it (`paper/paper.tex`, `paper/app_wsg.tex`) and these generated macro and table files: `wsg_dm.tex`, `wsg_decomp.tex`, `wsg_robust.tex`, `wsg_regtime.tex`, `wsg_dcaime.tex`, `wsg_auditours.tex`, `tab_prereg.tex`.
- `paper/supplement.pdf`: its structure and the first scorecard pages (jobs 073 to 078). I did not read every clause.
- Analysis scripts in the repository. I read these to learn which launches and definitions the authors used, not to reuse their code: `decomp_measured.py`, `fig_decomp.py`, `reanalysis.py` (the `features`, `outcomes` and `dirs` functions), `reg_timing.py`, `sumlaw_paper.py` (the `profiles` function), `speed_limit.py` (`host_rates`, `limit`), the header of `audit_ours.py`, and the read-counting lines of `panel_099.py`. On the gpu branch: `jobs/ec2/fetch_table.py` (`bandwidths`), the boot script under `gpu/`, and the clone lines of `gpu/vast.py`.
- Registration material:
  - `gpu/vast_ledger.json`;
  - `prereg/gpu_pushes.json`;
  - `prereg/reanalysis_hosts.json` (structure and the list of launches only);
  - the job-script headers for jobs 081, 096, 099, 102 and 104 to 108;
  - `git log --format='%H %ct %at'` for every `jobs/*.sh` of jobs 093 to 108;
  - the file diffs (content only) of the three commits that touched a job header after that job's first rental: 3701ceab, a8e2f921 and 89c5c3b9.
- Raw results in `/home/claude/gpu-branch/results/` for jobs 081 to 108:
  - per-problem rows (`ec_g_C*.jsonl`, `ec_g_C*_r*.jsonl`, `bs1.jsonl`, `par_*.jsonl`);
  - counters (`st_*.json`);
  - probes (`concur.txt`, `cores.txt`);
  - CPU and GPU identity (`cpu.txt`, `lscpu.txt`, `nvidia-smi-q.txt`);
  - Nsight summaries (`g_prof.json`, `prof_C14.json`, `prof_C32.json` of jobs 069c and 105);
  - microbenchmark outputs (`readsched_C14.txt`, job 102);
  - `manifest.json`;
  - the AIME routing trace `084c_gptoss_trace@vast/route_aime25_gptoss.npz`.

## Independence statement

- I did not open anything under `reports/` (I only wrote this file), any `prereg/*outcome*.md`, `research_notes/`, any file named like a review, number check, plan or progress log, `paper/paper_v1_prereview.tex`, or `apply/`. I also did not open `paper/paper_v2_prerewrite.tex`.
- I did not read commit messages. I used only hashes, author and committer times, push times from `prereg/gpu_pushes.json`, and file diffs.
- I wrote every analysis script myself, in `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/rev18b/` (`rawlib.py`, `rstar.py`, `decomp.py`, `counters.py`, `share.py`, `headline.py`, `sec5.py`, `fewest.py`, `sec6.py`, `relation.py`, `eq3.py`, `beat.py`, `misc.py`, `rs102.py`, `regcheck.py`, `lag.py`). None imports the authors' code.
  - I reimplemented the probe parsing, the time-per-token means, the bootstrap, the Spearman and variance computations, the closed-form relation and Belady's MIN with bypass on the trace.
  - Where a definition was ambiguous I copied it from the authors' script and say so: the link-to-CPU ratio, the exclusion list of unstable hosts, and the rule of taking the first launch per GPU.
- I modified nothing in either repository and committed nothing.

---

## 1. Summary

The paper studies batch-1 decode of Mixture-of-Experts models whose experts do not fit on one consumer GPU (gpt-oss-120b and Qwen3-30B-A3B on rented RTX 5090 hosts). It makes four contributions.

1. **A bound.** No cache with C slots per layer reads fewer experts from host memory than Belady's MIN with bypass. MIN's reads at the machine's highest probed host rate (Eq. 1) therefore bound every system that executes the exact routing. Two variants follow:
   - Eq. (2) adds the GPU's non-expert time in series, for systems that read only after routing.
   - Eq. (3) respects the order of layers when the CPU serves some reads.

   A microbenchmark replays MIN's reads without the model and reaches 86 to 96% of the bound's read time. The authors' llama.cpp expert cache leads FreeToken at 11 of 12 configurations on two hosts. It runs at 31 to 54% of Eq. (1)'s speed at gpt-oss 11% on 25 machines with consumer processors.
2. **A measured decomposition of the gap**, the deployed cache's time beyond Eq. (1). Oracles built into the engine change what is cached (MIN's set), how it is read (once rather than twice), and when (copies a few steps ahead). At gpt-oss 11%, on the 13 of 15 machines whose link-to-CPU ratio is at least 0.5 (a subset chosen after the data):
   - either change alone closes about 1% of the gap;
   - both together close 39%;
   - reading ahead closes another 15%;
   - 47% remains.
3. **How foresight must be spent.**
   - Read each admitted expert once.
   - Use MIN's fewest-admission schedule: 1.21x over the deployed cache on 9 stable machines.
   - Choose the read path by the link-to-CPU ratio (Spearman 0.89 over 19 machines).
4. **How much foresight is needed.** Half of foresight's value in host reads needs about 0.65 C distinct experts of lookahead per layer, across 9 models' own text (0.66 to 0.81 C on the engine's workload). Next-token routing recovers at most a third.

The paper comes with:

- a registration record of 1,483 clauses in the headers of job scripts committed before each rental, with every failure scored;
- a rental ledger;
- every raw row.

## 2. Strengths

1. **The numbers are what the raw data say.** I recomputed more than 30 quantitative and methodological claims from per-problem rows, counters, probes, profiles and the routing trace, without the authors' code (table in Section 9).
   - Every main-text number I checked reproduces, usually to the printed digit.
   - This includes Table 3 (all ten cells and their intervals), Table 4 (all ten rows on both hosts), the 31 to 54% range, R⋆ itself, and the Section 5 and 6 statistics.
   - For a paper of this size that is unusual, and it is the most important thing a referee can say about soundness.
2. **The bound is argued carefully and holds empirically.**
   - The assumptions of Eqs. (1) to (3) are stated in the text: exact routing, whole experts, per-layer slots, one token per step, no latencies, and a probe-relative rate. Table 28 tightens them one at a time.
   - I checked every measured state of every engine launch from job 093 to job 108 against Eq. (1). Nothing beats it. The closest is the prefetching oracle on a Ryzen 7 5700X3D at 1.33x the bound's time. Counted host reads per token divided by time never exceed 1.02 B_host.
   - The bound is therefore not an artefact of an optimistic probe.
3. **The experimental design is right for the question.**
   - Oracles run inside the production engine rather than in a simulator.
   - A 2x2 factorial (set x number of reads) exposes the interaction, which is the paper's main insight. Pacing is nested under MIN's set.
   - Each configuration runs in a fresh context, in a seeded shuffled order.
   - Machines are treated as the unit, with evidence: by my recomputation, machines carry 96% of the variance of the log gain of MIN read once and problems carry 1 to 3% (19 machines x 20 problems). Kendall's W of 0.77 and 0.82 reproduces.
4. **The registration discipline is real and checkable.** For all 71 rentals of jobs 093 to 108:
   - The job header was committed at least 2.7 s before the rental was created and pushed to GitHub at most 0.8 s after it. Seven pushes fall within one second of the rental and four after it.
   - Every machine started its job at least 26 s after the push. Machines clone the public branch at boot.
   - No job header was edited after its job's results arrived. The only post-launch commits are the disclosed amendments (jobs 100 and 107) and three host substitutions marked "Predictions unchanged" (jobs 102, 105 and 106).
   - Failed predictions (308 clauses) are reported, not hidden. The scorecard is re-scored by a script.
   - This is a standard of evidence that most systems papers do not attempt.
5. **Validity gates and failed hosts are handled honestly.**
   - The gates are: device bandwidth (throttled cards), host memory in use, output correctness by teacher-forced loss, and agreement between rounds within 2%.
   - Gated and unstable hosts are counted in the text: 3 and 2 at the gate, 2 and 1 unsteady, 3 of 8 server launches valid.
   - Every rental is in the ledger, with its cost.
6. **The artifact is unusually complete**: per-problem JSON rows with per-step times, engine counters, probe logs, Nsight summaries and the routing traces. A referee can audit almost every number, and I did.
7. **The rules for builders in Section 7 are concrete and conditional.** They state where they hold (the link-to-CPU ratio, budget and model), and they would help practitioners.

## 3. Weaknesses (most important first)

### W1. The headline decomposition is over a subset chosen after the data, and the abstract does not say so

- The abstract, the Introduction (PDF lines 045 to 052), Table 3 and builders' rule 2 all quote "39% together, another 15% ahead, 47% left". These are means over the 13 machines whose link-to-CPU ratio is at least 0.5, a threshold chosen after the data.
  - Table 3's caption and Table 1 say this.
  - The abstract reads as if the 13 machines were a natural class ("on the 13 machines whose PCIe link reads at least half as fast as their CPU").
- Over all 15 machines that ran every state, my recomputation gives these shares at gpt-oss 11%:
  - "together" 32.6% [21.8, 42.2];
  - "left" 52.5%.
- On the slowest machine (099f, ratio 0.29), MIN's set read once is slower than the deployed cache: "together" is -23% of the gap.
- What was registered (job 099, prediction 5; job 096, predictions 2, 3 and 7) is weaker than the headline.
  - The registered claims are the sign of the interaction "at both cells on every host", plus "base to foa changes time by at most 5% on every host".
  - The interaction was positive on 11 of 12 registered machines at 11% (it failed on 099f) and on 12 of 12 at 25%.
  - The foa clause held on 11 of 12 at 11% and 10 of 12 at 25%.
  - The registered clause as written ("every host") therefore failed. Table 1's "held on 11 of 12 machines" is accurate but generous.
  - The magnitudes (39%, 15%, 47%) were never registered.
- My check of the split's robustness came out favourable: defining the ratio as the maximum link reading over the maximum CPU reading gives the same 13/2 split. But the "slow" class has two machines.
- **Required:**
  - Report the all-machine numbers beside the subset in the abstract and in Table 3.
  - Label the 39/15/47 split as exploratory in the abstract.
  - Ideally, register the 0.5 threshold and the expected shares, and test them on new machines.

### W2. "The other half no oracle removes" is mostly explained, and the framing hides that

- The abstract, Section 4 and the Conclusion present 47% of the gap as what "no oracle we built removes". Section 4 names T_GPU and the prefetching oracle's excess reads, but quantifies only the first.
- With the authors' own counters (`st_g_C*_both3p.json`), I split "left" (both3p minus Eq. 1) into three parts:
  - T_GPU in series: 2.9 ms;
  - the prefetching oracle's reads beyond MIN at B_host;
  - a residual.
- Results over the 13 machines:

  | Budget | T_GPU | Oracle reads beyond MIN | Sum | Measured "left" | Residual |
  |---|---|---|---|---|---|
  | gpt-oss 11% | 26.5% | 21.8% | 48.3% | 46.5% | -1.8% |
  | gpt-oss 25% | 32.9% | 20.0% | 52.9% | 44.3% | -8.7% |

- So the best oracle runs at almost exactly T_GPU plus its own reads at the probe's rate. Its remaining distance to Eq. (1) has two causes:
  1. the GPU's non-expert work in series with the reads;
  2. the oracle reading 1.22 to 1.31x MIN (paced path, queue cap, three-step lead).
- The second cause belongs to this oracle's implementation, not to MIN.
- This is a cleaner and more useful result than "half remains". It also changes the message to builders:
  - roughly a fifth of the gap is attributable to a prefetcher that reads more than MIN;
  - a prefetcher that kept MIN's reads would test whether that fifth is removable.
- **Required:**
  - Report this accounting in Section 4 and Table 3.
  - Reword the abstract and conclusion. "No oracle we built removes" is true but uninformative.
  - The sentence "Removing the rest needs every read issued before routing" (line 265) is not supported for the excess-reads part.

### W3. Eq. (2) is presented as a bound on other systems but uses this engine's kernels; the evidence labels overstate the bound

- Eq. (2) uses T_GPU = 2.9 ms. I reproduced this as the smallest non-expert kernel time over 14 Nsight profiles of the authors' engine, 2.94 ms. The text then states: "Every system we compare reads on demand; only reads issued before routing can go below Eq. (2)."
  - T_GPU is a property of this engine's kernels. The paper says so itself (Section 9).
  - Appendix H shows another engine with lower non-expert GPU time at a cell profiled in both: FreeToken's norms, small kernels and attention projections take 0.40 and 1.34 ms against the authors' 0.97 and 1.65 ms (Qwen3 43.75%).
  - Eq. (2) is a bound for systems with these kernels, not for FreeToken or llama.cpp.
- Table 1 labels Eq. (1) "derived". It is derived conditionally on B_host, one probe run that often ran during the model download.
  - It survives my empirical check (strength 2). But the label should say "derived, relative to one probe run", as the Limitations section does.
  - The 12 of 84 launch-budgets where "the engine's reads imply a rate above the probe" come only from Eq. (6)'s overlap credit. Counted reads divided by time never exceed 1.02 B_host. That sentence frightens the reader more than the data warrant.

### W4. The registration record is strong as a record but weak as confirmation

- **Scale and scoring.** The 1,483 clauses are fine-grained bands and signs, set from pilots on similar or identical machines (Pf was rented six times).
  - 40% of them "held" only on the point estimate.
  - There are no designated primary hypotheses, no stated decision rules for the paper's headline claims, and no accounting for multiplicity.
  - The paper's own phrase, "Held is mostly on the point estimate", is honest. But the record is closer to sequential exploration with dated forecasts than to a confirmatory design.
- **Table 1 misstates two rows.**
  - Row 5 says foresight held "on three machines with ratios near one". O5 (096b) has a ratio of 0.58 by the paper's definition and by mine; the 47 and 27 GB/s readings appear in Appendix H.
  - Row 4 omits that the registered clause was "every host".
- **The provenance chain has one unrecorded link.** `manifest.json` records the patch sha256 but not the commit SHA the machine checked out. "Each machine ran the header as committed" is inferred from timing. Recording `git rev-parse HEAD` and the job script's sha256 would close the chain.
- **Data-label integrity.** In job 103 the solver did not install and the engine ran the greedy schedule. The raw rows and counters are still labelled `fetchplan` and `bypassplan`. A reanalysis from raw data that does not read the logs silently mixes greedy and fewest-admission runs.
  - In my recomputation, including job 103 turns "9 machines, 1.21x [1.13, 1.30], no losses" into "10 machines, 1.18x, one loss".
  - Relabel the rows or add a README in `results/103*`.
- **Rental accounting.** The text accounts for the rentals without results of jobs 099 to 108, but not for:
  - the first rental of 099d (07:19 to 07:32 UTC);
  - the second rental of 095;
  - two extra rentals each of 097 and 098.
  These are in the ledger, but the paper's sentence implies the list is complete.

### W5. Inference over machines rests on small convenience samples

- The machine-level summaries use 9 to 15 machines. They are the cheapest Vast offers that passed the gates, dominated by Ryzen 9 9950X-class desktops, with several re-rentals (Pf six launches, O4 three).
- Percentile bootstrap intervals over 9 to 13 values are anti-conservative. The t-intervals appear only in Appendix L, and only for two statistics.
- The target population of those intervals is never stated.
- The class boundaries were drawn after the tests that failed:
  - "consumer" against "server" processors, after jobs 107 and 108;
  - "desktop-class" in job 108's registration, which admitted EPYC 7543, 7K62 and 9354 hosts.
- The headline 31 to 54% excludes server processors. On those, the cache runs at 12 to 46% of the bound.
- The system comparison (Table 4) uses problem-level intervals on two hosts. The registered rerun on host S missed its ±0.06 band at 5 of 6 cells, which is direct evidence that problem-level intervals understate between-machine variation.
- Some statements rest on one or two machines:
  - "the layer-ahead copy gains 1.04x where the link is as fast as the CPU" is one machine (105f);
  - every slow-link claim in Section 4 rests on two.

### W6. Scope

- One GPU model (RTX 5090) for every oracle experiment, and one model (gpt-oss) for the decomposition.
- AIME-25 prompts only. They were also used during development. The runs are 20 or 30 problems x 256 tokens, teacher-forced.
- The horizon rule's constant differs between the 9 models' own text (0.65 C, quoted in the abstract) and the engine's workload (0.66 to 0.81 C, quoted in the Conclusion).
- On the panel the engine's time shares sit below the read shares that Figure 5 is drawn in: an exact 4-token window recovers 0.40 of the gain in time at 11% against 0.52 in reads. The "half needs about 0.65 C" rule is a statement about reads.

### W7. The comparison with published systems is weak evidence and should not headline the Introduction

- "Published batch-1 systems reach a median 13.6% of their bound, ours 27%" compares rows from 13 sources with different timing methods, precisions and budgets, at datasheet ceilings. Uniform routing stands in where no trace exists, and rows below 5% are "not adjudicated".
- I reproduced 13.6% (20 in-class rows) and 9.5% (29 trace rows) from Table 31's values.
- The comparison is a useful appendix exhibit, but it is not a finding at the level of the rest, and it is missing from Table 1.

### W8. Smaller points

- **Host-bound.**
  - At gpt-oss 25% on the Threadripper 9960X (B_host 178 GB/s), Eq. (1) is GPU-bound: the minimum over c balances at about 25 CPU experts.
  - With the measured GPU rate (52% of datasheet), most decomposition machines with B_host above about 55 GB/s would be GPU-bound at 25% as well.
  - "Host-bound on nearly every machine" deserves the qualifier "at datasheet GPU rates".
- **The link-to-CPU ratio has two definitions.**
  - The class split uses B_p = one zero-copy reading (16 MB, 64 blocks) and B_c interpolated at the helper count.
  - Eq. (3) and Figure 10 use the best link and CPU readings.
  - Section 2 defines only one. The split is robust (see W1), but the text should say which definition is used where.

## 4. Clarity

Earlier rounds scored clarity 2, then 3 four times. The restructuring helped the main text. The appendix and the abstract remain the obstacles. I score it 3 again, closer to 4 than before.

### What works

- **Question-driven sections that state their yardstick.** Sections 3 to 6 each answer one question named in the title and open by naming the measure used, and the "Yardsticks" paragraph in Section 2 lists all four.
- **Table 1.** The claim-to-evidence map with registered, exploratory and derived labels is the best single device in the paper. A reader knows where each claim stands before reading it.
- **Table 2.** Configurations are named in words, with the number of reads per admitted expert, so readers no longer decode `foa`, `both3p` and similar in the main text. Appendix B maps the codes.
- **Figures.**
  - Figure 1, the schematic of the second read, makes the paper's key mechanism visible in one picture.
  - Figure 2, the staircase, shows each machine's path and the two bounds; the decomposition is legible at a glance.
- **The running example** in Section 5 (10.0, 13.0, 20.6, 15.2 and 13.8 ms on one 9950X) anchors the ratios that follow.
- **Limitations** are specific and quantified, not generic.

### What still makes it hard to read

1. **The abstract is overloaded.**
   - The sentence at lines 019 to 022 carries a post-hoc subset condition, two undefined terms ("MIN's set", "read once"), "alone" against "together", and two percentages.
   - "The gap" is defined in passing as "the rest of its time".
   - The last sentence switches to reads and introduces "0.65 C distinct experts" without saying what C is.
   - A reader new to the area will not get through it in one pass.
2. **There are four yardsticks with different baselines.**
   - Section 3: share of the bound's speed.
   - Section 4: share of the gap to Eq. (1).
   - Section 5: speed relative to the deployed cache.
   - Section 6: share of MIN's gain over admitting every miss, a policy that is not deployed and is far slower on slow-link machines (aa/foa is 0.45 on 099f).
   - The reader must re-anchor in every section. Numbers such as "39%", "1.21x", "0.18" and "31 to 54%" look comparable but are not.
3. **Three decompositions of the same gap are never reconciled.**
   - The measured staircase in Section 4.
   - The order-free Shapley accounting in Appendix F: in the running example "both" closes 51.5%, and Shapley gives 26.1% and 25.5%.
   - The modelled Shapley accounting in Appendix G: foresight is 37 to 53% of the gap.
   - Appendix K's three-part split adds a fourth.
   - No paragraph says which is the answer, or why they differ (different machines, baselines and models).
4. **The vocabulary and notation load is heavy.**
   - The appendix tables use about 25 configuration codes and about 20 host nicknames (A, B, S, O1 to O6, Pa to Pj, "Pf again", "285K, second").
   - There are four symbols for GPU time: T_GPU, G, G_fit and g.
   - Eq. (6) appears as "the relation", "the closed-form account", "the time relation" and "the plain form". Figure 7's legend still says "law's table", although the text insists the model "is not a law".
   - "Limit" and "bound" are used interchangeably; Table 7 has to apologise for it.
5. **The same quantity carries different numbers in different places.**
   - Horizon: "about 0.65 C" (abstract) against "0.66 to 0.81 C" (Section 7 and Conclusion).
   - Distance to the bound: "a third to a half" (Introduction) against "31 to 54%".
   - GPU time as a share of the gap: "27 to 31%" (Table 3) against "27 to 38%" (Section 7) against "23 to 66%" (Appendix K, which uses G rather than T_GPU).
   Each is defensible, but together they make a careful reader suspect inconsistency.
6. **The main text still carries an exploratory detour.** Section 4's paragraph on the closed-form account says the decomposition "does not rely on it". One sentence pointing to Appendix K would do.
7. **The appendix reads like a lab notebook.**
   - The paragraph "Failures, jobs 073 to 098" in Appendix C is about 350 words joined by semicolons.
   - Appendix D repeats a leave-one-host-out paragraph almost verbatim, in "The blind test in full" and again in "Is the max the right form?".
   - The appendix order does not follow the main text: the calibrated model (D), the grid (E) and two Shapley accountings (F, G) come before the evidence for Sections 4 to 6 (H to L). The expert cache that Section 2 summarises is in M.
   - Several appendices document earlier lines of inquiry that the main-text claims barely use. I checked the labels in the LaTeX source: Appendices D to H (the calibrated model, the grid, both Shapley accountings, and foresight in the engine) are never cited from the main text. That is about nine pages (pp. 19 to 28), Tables 9 to 15 and Figures 6 to 9. Appendix G is cited nowhere at all.
8. **Table 4's caption is nine lines.** It includes results (FreeToken's pooled-bound percentages and the all-in-VRAM speeds) and a footnote mark, and the table has two bound columns. Moving the results into the text would help.

### Concrete suggestions

- Rewrite the abstract in five sentences, each naming one result and its scope, and mark the subset as post hoc.
- Add one "reconciliation" paragraph that relates Section 4, Appendix F, Appendix G and Appendix K.
- Rename Eq. (6) once and use that name throughout.
- Remove "law" from figures.
- Cut or condense Appendices D, E and G into a "history" appendix.
- Use one canonical value per quantity.
- Split Appendix C's failure paragraph into a table.

## 5. Questions for the authors

1. What are Table 3's shares over all 15 machines, and was the 0.5 threshold, or any analysis choice for the decomposition, registered for jobs 100 and 101, which added machines to it?
2. Do you agree that "left" is accounted for by T_GPU plus the prefetching oracle's reads beyond MIN at B_host (residual about -2% at 11%)? Can you build a prefetch oracle that reads close to R⋆ (no queue cap, enough lead) to test whether that fifth of the gap can be removed?
3. How do you justify Eq. (2) as a bound for FreeToken and llama.cpp, given that their non-expert kernels can be faster than yours (Appendix H)? Would profiling their T_GPU, or restricting the claim, be feasible?
4. What population do the machine-level bootstrap intervals generalise to? How do you weight re-rented machines, for example Pf with six launches? Why are percentile rather than t- or BCa intervals the default at n = 9 to 13?
5. Why does the link-to-CPU ratio use one zero-copy reading and an interpolated CPU rate, while Eq. (3) uses the best readings? Should Section 2 define both?
6. Will you record the cloned commit SHA and the job script's hash in `manifest.json`, and relabel job 103's `fetchplan` and `bypassplan` rows?
7. For builders, should the horizon rule be stated in time rather than reads, given that time shares on the panel are 10 to 20 points below read shares at short windows?
8. Which rows of Table 1 would change if every "held" were required to hold with an interval?

## 6. What would raise my score

- **To 8:**
  - Report the decomposition over all machines beside the post-hoc subset, everywhere it is quoted.
  - Add the accounting of "left" (W2) and reframe the "other half".
  - Restrict Eq. (2)'s claim to systems with comparable kernels.
  - Correct Table 1's rows 4 and 5.
  - Rewrite the abstract.
  - Reconcile the decompositions in one paragraph.
- **Beyond 8:**
  - A small, newly registered confirmatory panel, for example six or more consumer machines not rented before, with the 0.5 threshold, the expected "together" share and its decision rule fixed in advance.
  - A prefetch oracle that reads close to MIN, showing how much of "left" is removable.
  - A second GPU generation, or a second model, for the decomposition.

## 7. Scores

| Criterion | Score |
|---|---|
| Overall (1 to 10; 6 = acceptable with revisions, 8 = strong) | **7** |
| Soundness (1 to 5) | **4** |
| Methodology (1 to 5) | **4** |
| Significance (1 to 5) | **3** |
| Clarity (1 to 5) | **3** |
| Confidence (1 to 5) | **4** |

**Justification.**

- **Soundness: 4.** The numbers reproduce from raw data, and the bound survives an empirical check. Points are lost for the framing of the residual half, for Eq. (2)'s scope, and for a post-hoc subset in the abstract.
- **Methodology: 4.** It is exemplary in registration, validity gates, unit of analysis and the artifact. It falls short in confirmatory design, small-n interval estimation and class definitions drawn after the tests.
- **Significance: 3.** The question matters to a growing local-inference community. The bound-plus-oracle method transfers to other settings. The scope is one GPU, two models and one workload.
- **Clarity: 3.** The main text is now navigable. The abstract, the four yardsticks, the unreconciled decompositions and the appendix keep it below 4.
- **Confidence: 4.** I verified the core claims from raw data. I did not check the published-systems audit beyond Table 31's values, the KL divergences (the full-vocabulary dumps are omitted), or the 9-model trace study.

## 8. Minor and editorial points

- Line 179: "the smallest of our engine's Nsight profiles". Say "non-expert kernel time, smallest of 14 profiles on 7 hosts (2.94 ms)".
- Section 4: "MIN's set admits more experts than the deployed one". Give the factor (4.6x in the running example) in the main text.
- Figure 3: the y-axis order and the grey ticks are hard to read at column width. Consider showing only the host-bound budgets in the main text.
- Appendix D: delete the duplicated leave-one-host-out paragraph.
- Appendix C: "for four GitHub records the push up to 0.8 s after the rental was created". Add that this is before the machine booted (the minimum lag to job start is 26 s), which is what makes it harmless.
- Appendix M, "FreeToken tuned": job 098 had three rentals; state why two were replaced.
- Table 10 caption: "host when the optimum's reads alone take longer than the GPU's whole read". This defines "host-bound" differently from Section 2. Unify.

## 9. Claims checked against raw data

All values below are my recomputations from raw files, using my own scripts in the `rev18b` scratch folder.

| # | Claim (location) | Paper | Recomputed from raw | Source files | Verdict |
|---|---|---|---|---|---|
| 1 | MIN's reads per token R⋆, gpt-oss 11% / 25% (Eq. 1, Tables 13 and 15) | 38.3 / 15.3 | 38.32 / 15.34 (Belady with bypass, per layer, cache carried across all 30 problems, cold start) | `084c/route_aime25_gptoss.npz` | Reproduced |
| 2 | R⋆ on the first 20 problems (Section 9) | -0.4% | -0.38% at both budgets | same | Reproduced |
| 3 | Host B probe and bound (Table 4) | CPU 72, link 53, highest 88 GB/s; bound 172 tok/s at 11% | 71.6, 53.2, 87.5; 172.3 tok/s | `081/concur.txt`, `cores.txt` | Reproduced |
| 4 | Table 4, host B (launch 2) | 69.9 vs 54.0, 1.294 [1.278, 1.312]; 1.275; 1.154; Qwen3 1.032 | 1.294 [1.278, 1.311]; 1.275 [1.254, 1.295]; 1.154 [1.133, 1.172]; 1.032 [1.022, 1.042] | `081/bs1.jsonl` | Reproduced |
| 5 | Table 4, host S (all six rows) | 1.207, 1.196, 1.093, 1.027, 1.048, 0.974 | identical to three decimals; 0.974 [0.962, 0.988] | `089/bs1.jsonl` | Reproduced |
| 6 | Leads FreeToken at 11 of 12; 2.0 to 4.0x llama.cpp | as stated | 11 of 12; 2.00 to 4.02x | 081, 089 | Reproduced |
| 7 | Ratio of mean rates against ratio of total times (Appendix A, rule 3) | within 0.003 | maximum difference 0.003 | 081, 089 | Reproduced |
| 8 | Read time nearly reachable (Table 29) | per layer 86 to 96%, per token 93 to 97%, link-only 35% on Pd | 86.1 to 96.3; 93.3 to 97.5; 35.4 | `102*/readsched_C14.txt`, `concur.txt` | Reproduced |
| 9 | Cache at 31 to 54% of Eq. (1), gpt-oss 11%, 25 consumer machines | 31 to 54% | 31.2 to 54.0% (first launch per GPU, unstable hosts excluded) | ec rows and probes of 093 to 108 | Reproduced |
| 10 | 40 to 54% of Eq. (3), 55 to 70% of Eq. (2); Eq. (3)/Eq. (1) median 1.00, maximum 1.56 | as stated | 40 to 54, 55 to 70, 1.00, 1.56 | same plus profiles | Reproduced |
| 11 | 23 to 36% at gpt-oss 25% | 23 to 36% | 22.8 to 36.1%; Eq. (1) is GPU-bound for TR 9960X (B_host 178) | same | Reproduced; "host-bound" fails on one machine |
| 12 | Closed-form account: 39 launches on 25 machines, implied rate 0.88 to 1.16 of B_host; within 6% on 31, within 8% on 37 | as stated | identical; median abs. error 1.7% | ec rows, `st_*_base.json`, `g_prof.json` | Reproduced |
| 13 | Probe's second-highest reading | median 0.8% below; above 5% on one launch (107d) | 0.8%; 107d only | `concur.txt` (47 launches) | Reproduced |
| 14 | T_GPU 2.9 ms (smallest profile); G 4.1 to 4.7 ms | as stated | 2.94 (105e, C14), maximum 3.38; G 4.07 to 4.69 over 14 profiles on 7 hosts | `prof_C14.json`, `prof_C32.json` | Reproduced |
| 15 | 15 machines ran every state; 13 with ratio at least 0.5; slow ratios at most 0.40 | 15 / 13 / 0.40 | 15 / 13; slow 0.29 and 0.40; same split with max-link/max-CPU ratio | ec rows, probes | Reproduced; split robust |
| 16 | Table 3, gpt-oss 11% | 1 [-1, 3]; 1 [0, 2]; 39 [32, 45]; 15 [11, 18]; 47 [42, 51] | 1.2 [-0.7, 3.0]; 0.9 [-0.2, 2.0]; 38.8 [31.8, 45.4]; 14.6 [10.5, 18.3]; 46.5 [42.2, 51.1] | ec rows of 096a/b, 099a to j, 100b, 100f, 101b | Reproduced |
| 17 | Table 3, gpt-oss 25% | 21; -1; 35; 20; 44 | 20.8 [19.2, 22.3]; -1.1; 35.3 [30.1, 40.3]; 20.5; 44.3 | same | Reproduced |
| 18 | Best oracle closes 54% [49, 58] (fast link), at most 17% (slow link) | as stated | 53.8 [49.4, 58.0]; 16.8 | same | Reproduced |
| 19 | After all three parts, 1.07 to 1.38x Eq. (2) at 11% | as stated | 1.07 to 1.39 | same | Reproduced |
| 20 | Prefetching oracle reads 1.22 to 1.31x MIN at 11% | as stated | 1.22 to 1.31 | `st_g_C14_both3p.json` | Reproduced |
| 21 | Registered interaction "held on 11 of 12 machines" (Table 1) | 11 / 12 | interaction above 0 on 11 of 12 at 11% (fails on 099f), 12 of 12 at 25%. Registered companion "base to foa within 5% on every host": 11 of 12 at 11%, 10 of 12 at 25% | job 099 header, prediction 5; ec rows | Reproduced; the clause as written ("every host") failed |
| 22 | Decomposition over all machines (not reported) | none | all 15: "together" 32.6% [21.8, 42.2], "left" 52.5%; 099f "together" -23% | same | New; see W1 |
| 23 | What "no oracle removes" (Section 4) | "two things we can name" | 11%: T_GPU 26.5% + oracle reads beyond MIN 21.8% vs "left" 46.5% (residual -1.8%). 25%: 32.9 + 20.0 vs 44.3 (residual -8.7) | counters plus ec rows | New; see W2 |
| 24 | Does anything beat Eq. (1)? | (implied: no) | minimum measured time / Eq. (1) = 1.33 (both3p, 100b, 11%); counted reads x S / time at most 1.02 B_host in every state | every ec and st file, 093 to 108 | New; supports the bound |
| 25 | MIN read once gains 16 to 51% on the factorial machines, up to 81% read ahead | as stated | 1.160 to 1.514; 1.812 | 095, 096a, 096b | Reproduced |
| 26 | Fewest-admission schedule: 1.21x [1.13, 1.30] on 9 stable machines; 1.17x over 13; losses on the two slowest links | as stated | 1.215 [1.128, 1.304]; 1.166; losses at ratios 0.14 and 0.21; t-interval 1.11 to 1.33 | 104 to 107 | Reproduced, but only after excluding job 103, whose rows are mislabelled (W4) |
| 27 | Admission margin 1.021x / 1.018x on 5 machines; reads 4 to 6% lower at 11% | as stated | 1.021 / 1.018; 4.3 to 5.8% (6.3 to 10.5% at 25%) | 106a, b, e; 107b, d | Reproduced |
| 28 | Spearman of MIN read once against link-to-CPU ratio, 19 machines | 0.89 [0.63, 0.97] | 0.89 [0.63, 0.98] (0.82 at 25%) | 093 to 104 | Reproduced |
| 29 | Machines carry 96% of variance, problems at most 3%; Kendall's W 0.77 / 0.82 | as stated | 96.3 / 96.2%; 1.1 / 2.8%; W 0.77 / 0.82 | same, 19 x 20 | Reproduced |
| 30 | Same-CPU panel machines differ by up to 29%; a re-rented machine within 3.2% | as stated | 29.1% (285K, 099a vs 099f); 3.2% (TR 9960X, 105b vs 106a) | ec rows | Reproduced |
| 31 | Exact 16-token window recovers 0.78 to 0.93 of the gain in time at 11%; the lowest misses the registered 0.80 | as stated | 0.78 to 0.93 (099f 0.78). Time shares are below read shares: W=4 gives 0.40 vs 0.52 | 099a to j ec and st | Reproduced; see W6 |
| 32 | Layer-ahead copy 1.04x where the link matches the CPU, down to 0.72x | as stated | 1.042 (105f, the only fast-link host); 0.722 (Pf) | 105 ec rows | Reproduced (n = 1 for the gain) |
| 33 | Output parity: loss changes by +0.16% / +0.19% (1,536 steps) | as stated | +0.16% / +0.19% | `090/par_*.jsonl` | Reproduced |
| 34 | Registration timing: commit at least 2.7 s before rental; 7 of 71 pushes within 1 s; 4 after (at most 0.8 s); job start at least 26 s after push | as stated | identical. No header edited after its job's results. Post-launch commits are the disclosed amendments (100, 107) and host substitutions (102, 105, 106) | git log, `gpu_pushes.json`, ledger, manifests | Reproduced; registration verified |
| 35 | Table 8 totals | 495 / 597 / 308 / 83 | column sums match (1,483 clauses; 565 for jobs 073 to 098) | `tab_prereg.tex` | Reproduced |
| 36 | Audit medians: in-class 13.6% (20 rows), trace 9.5% (29 rows) | as stated | 13.6 / 9.5 from Table 31's values | Table 31 | Reproduced (from the table, not the scripts) |
| 37 | Table 1, row 5: "three machines with ratios near one" | near one | O3 0.82, O4 1.05, O5 0.58 | probes of 095, 096a, 096b | Incorrect wording |
