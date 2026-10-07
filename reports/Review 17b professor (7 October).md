# Review 17b: "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth"

Referee: a professor in a systems group that works on scientific benchmarking (in the spirit of Hoefler and Belli, SC'15), reviewing for a fellowship or workshop. Date: 7 October 2026.

## Materials read

- `paper/paper.pdf`, all 42 pages: the main text (pp. 1–12) and Appendices A–O (pp. 13–42). For the main text and for Appendices C, K, L and N, I also read the LaTeX source (`paper.tex`, `app_wsg.tex`, `app_relation.tex`, `app_more.tex`), and I skimmed `app_traces.tex` and `app_value.tex`. I read these generated tables and macro files: `tab_dm.tex`, `tab_limit.tex`, `tab_job107.tex`, `tab_audit.tex`, `wsg_dm.tex` and `wsg_decomp.tex`.
- `paper/supplement.pdf`: its structure, the scoring rule, the totals and the first scorecard table (jobs 073–081).
- From the artifact repository: `README.md`, `gpu/vast_ledger.json`, and `prereg/speed_limit_v2.json` (for the published R* value only). I read these scripts only for definitions: `scripts/decomp_measured.py`, `scripts/sumlaw_paper.py` (`profiles()`), `scripts/factorial_shapley.py` (`limit1_of`), `scripts/speed_limit.py` (`host_rates`, `limit`), `scripts/speed_limit_v2.py` (the definition of R*), `scripts/hostdep_model.py` (the parsing), and `mosl/cachesim.py` (`_sim`).
- From the `gpu` branch at `/home/claude/gpu-branch`:
  - The job headers of 099, 102, 105/105a and 107. I also read every job-script commit's dates and its diff restricted to `#` header lines; I did not read commit messages.
  - Raw results: `ec_*.jsonl`, `st_*.json`, `concur.txt`, `cores.txt`, `cpu.txt`, `nvidia-smi-q.txt`, `g_prof.json`, `prof_C*.json`, `patch_sha.txt`, `stdout.log`, `dl_gguf.txt` and `gate.txt`/`v0.txt`. These cover every launch of jobs 093–108, plus `bs1.jsonl` of jobs 081 and 089 (Table 4), `readsched_C*.txt` of job 102, `summary.txt` of job 090, and the routing trace `084c_gptoss_trace@vast/route_aime25_gptoss.npz`.
- My own analysis code is in `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/rev17b/`:
  - `load.py`: the raw loader and probe parser;
  - `minsim.py` and `minsim2.py`: my own MIN-with-bypass;
  - `decomp.py`: Table 3 from raw;
  - `plan.py` and `plan2.py`: the fewest-admission results;
  - `spear.py`: the link-ratio rank correlation and the variance shares;
  - `relation.py`: the implied read rate of Eq. (5);
  - `relaunch.py`: machine-to-machine and relaunch variation;
  - `tab4.py`: Table 4;
  - `tgpu.py`: T_GPU and G;
  - `regtime.py` and `regtime_early.py`: registration timing;
  - `audit.py`: the audit medians.

## Independence statement

I did not open anything under `reports/` except to write this file. I did not open `prereg/*outcome*.md`, `research_notes/`, any review, number-check, plan or progress file, `paper/paper_v1_prereview.tex`, `paper/paper_v2_prerewrite.tex` or `apply/`. I read no commit messages; I used only `git log` dates and file contents and diffs.

Every number in my table of checked claims comes from my own code reading raw rows, counters, probe files or traces. The authors' scripts I read only for definitions: what "B_host", "T_GPU" and "one launch per GPU" mean. Two checks went further:

- **The authors' cache simulator.** I ran it once, read-only, after my own MIN implementation disagreed with the published R*. The run was to find the semantic difference, which turned out to be step-level versus request-level MIN. I then reimplemented the step-level version myself.
- **The audit median.** I recomputed it from the paper's own Table 31 (`tab_audit.tex`), because the published rows are not GPU raw data.

I modified nothing in either repository. `git status` is clean apart from a `jobs/ec2/__pycache__/` directory on the `gpu` branch, which is dated 6 October and is not mine.

---

## Summary

The paper studies batch-1 decode of MoE models whose experts mostly live in host DRAM. The models are gpt-oss-120b and Qwen3-30B-A3B on rented RTX 5090 hosts. It asks three questions:

1. How fast could such decode be on a given machine?
2. Where does the remaining time go?
3. What is knowledge of future routing worth?

**The bound (Eq. 1).** Belady's MIN with bypass gives the fewest host reads R* for C slots per layer. At the machine's highest probed host read rate, R* gives a lower bound on time per token for any exact-routing system. A tighter version for systems that read after routing (Eq. 2) adds the GPU's non-expert work in series. A microbenchmark replays MIN's per-layer reads at 86–96% of the bound's read time. The authors' llama.cpp expert cache runs at 31–54% of the bound's speed at gpt-oss 11% on 25 consumer machines. It beats FreeToken at 11 of 12 configurations and stock llama.cpp by 2–4×.

**The decomposition (Section 4).** Oracles built into the engine remove three parts of the deployed cache's gap one at a time: what is cached, how it is read (once or twice), and when it is read. The remaining two parts are a profiled GPU constant, T_GPU, and a residual. On 13 fast-link machines at gpt-oss 11%, the five parts are 20%, 19%, 15%, 27% and 20% of the gap.

**Spending foresight (Section 5) and its horizon (Section 6).**

- Foresight pays when each admitted expert is read once, and when the fewest-admission hit-optimal schedule is used.
- The link-to-CPU ratio decides the read path.
- Half of foresight's value in host reads needs about 0.65·C distinct experts of lookahead per layer across 9 models.
- Next-token routing buys little.

**Registration.** Every experiment's predictions were committed in the job script header before the machine started. All clauses are scored, including the failures: 565 + 273 + 216 + … clauses. A closed-form account of the time (Eq. 5) is reported as exploratory, after its registered tests failed.

## Overall assessment

This is the most auditable performance study I have refereed in some time. Every number I recomputed from raw data reproduced, mostly to the printed digit: 35 checks are listed below. The registration practice is real. Every one of the 124 rentals started at least 2.7 s after the last commit to its job script. Post-launch header edits are rare and are disclosed in the headers themselves. The failed predictions are reported prominently.

The weaknesses are not in the arithmetic. They are in interpretation:

- **Shapley split of an interaction.** The abstract presents the headline "measured decomposition" as five separable shares. Two of them are the symmetric (Shapley) split of an interaction that is almost entirely joint: each factor alone closes about 1% of the gap at 11%.
- **The bound is loose.** It ignores the dataflow dependence of CPU-executed experts on the same layer's attention, and it is anchored to a single probe run taken while the model was still downloading.
- **The system comparison.** Its uncertainty is reported within one launch on one machine, and the compared systems decode different text.

Scope is narrow: one GPU model, the authors' own engine, two models, and AIME prompts that were also used during development. Clarity has improved, but the paper remains demanding.

---

## Strengths

1. **Verifiable pre-registration with disclosed amendments.** Predictions are in the job script headers, which machines cloned at boot. I checked commit times against `gpu/vast_ledger.json` start times for all 124 rentals of jobs 058–108. Every rental has a commit to its script(s) before its start, with a minimum gap of 2.7 s (105f), which matches Appendix C. For jobs 093–108, header edits after a launch started are:
   - host replacements in 102, 105, 106 and 107, each stating "Predictions unchanged";
   - one added prediction, job 100's ninth, committed 100 min after hosts a–d started but before the two hosts it concerns;
   - job 107's two amendments. These added a replacement rule for hosts that failed the round check, after four hosts' results were in, and named the next offers by a rule committed before each started.

   All of these are disclosed in the headers and in Appendix C. Job 099's header even registers the 2×2 interaction (prediction 5) and the replayed read shares (w1 = 0.18 and so on) that Section 6 later quotes. This is better practice than almost anything in our field.
2. **The machine is the statistical unit, and the choice is justified with data.** Two panel hosts with the same CPU differ by 29% (Core Ultra 9 285K, 099a against 099f), while a machine rented again reproduces within 3.2% (TR 9960X, 105b against 106a). I reproduced both numbers. A two-way decomposition of the log gain gives 96% of the variance to machines and 1% to problems. This is exactly the Kalibera–Jones argument applied correctly.
3. **Validity gates are registered and failures reported.** For job 107 the logs show three hosts stopped at V0 (50, 50 and 96 GB of host memory in use) and two at the 2% round check (spreads of 3.5% and 6.0%). For job 108 they show one host rejected as previously rented, one with 323 GB in use, and one round failure (EPYC 7K62, 7%). These numbers match the text. Their clauses are reported in the supplement rather than dropped.
4. **The oracles are built into a real engine.** The paper does not only simulate them: the engine's counters show the fewest-admission plan was actually executed (65,514 plan records), and they reproduce the replay (copies 0.597–0.726 of the greedy schedule's). The single-read finding has a clean mechanism and is robust across 13 machines. CPU-serve-then-copy reads each admission twice. "MIN, 1 read" gains 16–51% on all three factorial hosts at all six budgets.
5. **Negative results are reported plainly.** Examples:
   - the closed-form relation failed 3 of 4 registered tests and is labelled exploratory;
   - the layer-ahead copy loses on 3 of 4 hosts;
   - the 16-token window missed its registered 0.80 on one host;
   - the admission-margin gain held only through unstable machines.

   Table 1's "Evidence" column and Appendix C's scorecard are models of candour.
6. **Everything is scripted.** Every number in the paper is a macro, and the artifact's raw rows are complete enough that an outsider can recompute the main tables in a few hundred lines.

## Weaknesses (most important first)

### W1. The "measured decomposition" is presented as five separable shares, but its first two shares are a symmetric split of an interaction, and its last two are a constant and a residual.

**The first two shares are an interaction.** From raw `ec_*.jsonl` on the 13 fast-link machines at gpt-oss 11%:

- reading the deployed admissions once ("load alone") closes a mean 0.9% of the gap, range −2% to 4%;
- caching MIN's set while reading twice ("cache alone") closes 1.2%, range −4% to 6%;
- doing both closes 38.8%.

The interaction is therefore 36.8% of the gap. Table 3 and the abstract report "a fifth … caching the wrong experts, a fifth reading admitted experts twice". That is the Shapley convention dividing a joint effect in half. It is not two separable causes. Section 7's rule 2 ("[reading twice] costs about as much as caching the wrong experts") follows from the symmetry of the split, not from a measurement. The text does say MIN's set "is no faster than the deployed one" with two reads (I find a time ratio of 0.97–1.02), so the information is there. But the abstract, Table 3 and the rules present additive shares. The honest summary at 11% is: "MIN's set read once closes 39% of the gap; neither change alone closes more than a few percent." At 25% the structure is different (cache alone 21%, load alone −1%, interaction 16%). That difference is itself informative and is hidden by the Shapley table.

**The last two shares are not measured removals.** "Serial" is the constant T_GPU = 2.94 ms. This is the *minimum* non-expert time over 14 Nsight profiles from seven hosts, whose range is 2.94–3.38 ms and median 3.16 ms. "Rest" is the residual: MIN prefetched minus Eq. (2). Using the median profile would move about 2 points of the gap at 11% from "rest" to "serial". More importantly, the prefetched oracle issues copies before routing, so it is not bound by Eq. (2). At 25% one fast machine runs at 0.99× Eq. (2), so its "rest" is negative (−1%). The sentence "No oracle we built removes it" is therefore not literally true. The split between "serial" and "rest" is an accounting assumption, not an observation.

**The fast/slow class is post hoc.** The 0.5 link-to-CPU threshold that defines the 13 machines of Table 3 and the abstract was drawn after job 099's bands failed on link-starved hosts. Table 1 calls the split "derived". It is better described as "derived on a post-hoc class".

**The 15 machines pool several protocols.** They come from three jobs and three engine patches (`oracle`, `oracle2`, `oracle3`), with 30 problems for O4 and O5 and 20 for the rest. The deployed path looks invariant across patches (O4: 20.64, 20.85 and 20.48 ms in 096a, 097a and 105f). That should be stated rather than left for the reader to infer.

### W2. The bound is looser than the "headroom" message implies, and it is anchored to a single probe run taken during the model download.

**The CPU path cannot be overlapped.** Eq. (1) lets all host reads, CPU and link alike, overlap with GPU work at B_host. But a CPU-executed expert needs the post-attention activation of its own layer. However good the foresight, CPU-path work cannot start before that layer's attention and router. Only link-path copies can be issued early. So the 27–33% "serial" part is removable only by moving reads onto the link (bounded by B_p, not B_host), or by speculative execution. Rule 5 ("Issue reads before routing") is therefore implementable only for link copies. On the slow-link machines where the paper advises CPU-served misses (rule 4), the serial part is close to irreducible.

The paper should give a dependency-aware bound, for example min over c of max(T_GPU + c·S/B_c, (R*−c)·S/B_p, R*·S/B_cp, GPU term), next to Eq. (1). It should also restate the headroom against it. Eq. (1) remains a valid lower bound; the issue is how much of the gap a builder can actually win.

**The probe is a single reading.** B_host is the highest single reading of one probe run. Every job script I read starts the 65 GB GGUF download in the background before the probe. On most hosts the download was still running when the probe finished ("builds and probe done, waiting for the downloads"):

| Host | Probe finished | Download took |
|---|---|---|
| 099e | 585 s after launch | 1,760 s |
| 096a | 255 s | 1,782 s |
| 105b | 131 s | 917 s |

The script comment calls this phase "the host (idle)". On repeated rentals B_host is repeatable within about 3%: Pf 94.2–96.1, O4 50.5–51.1, Ph 65.6–67.6. This mitigates but does not settle the concern.

**The engine exceeds the probe on some hosts.** By the paper's own Eq. (5) accounting, the engine reads more than 5% faster than the probe on 12 of 84 launch-budgets. I find the extreme case to be 100f (9950X, x8 link): it implies 1.16× B_host at 11% and 1.22× at 25%. It runs 12% faster than Pe, another 9950X, with an identical B_host of 52 GB/s. 100f is also the machine that sets the top of the abstract's "31–54%". The upper end of that range is therefore probably inflated by an under-reading probe.

Please probe each machine idle, repeatedly, and after the timed runs as well as before. Report the bound with the median and second-highest readings in the main text, not only in Table 27.

**"Host-bound" depends on the GPU rate assumed.** Section 3 says the GPU term "binds only at the largest budgets (Table 28)". With the measured batch-1 GPU rate of 52% of datasheet, my recomputation shows that gpt-oss 25% is GPU-bound on any host with B_host above about 55 GB/s. On host B the bound drops from 430 to 279 tok/s, which Table 28 itself shows. With the datasheet rate, the TR 9960X (178 GB/s) is GPU-bound at 25% as well. "Host-bound" at 25% is thus a property of the machine and of the GPU rate assumed, not of the budget.

### W3. The system comparison (Table 4) reports within-launch uncertainty, and the systems decode different text.

The ours/FreeToken ratio at gpt-oss 11% is 1.294 [1.278, 1.312] on host B and 1.207 [1.195, 1.219] on host S. The two hosts differ by 0.09, about six times either interval's half-width. Appendix C admits that job 089's ±0.06 band failed. The intervals in Table 4 are over problems within one launch on one machine. That contradicts the paper's own Section 2 argument that the machine is the unit. Table 4 should either report an interval over machines (the grid of Appendix E has eight hosts for the gpt-oss lead) or label its intervals "this launch only".

Separately, the systems decode different greedy text. In job 081's raw rows at gpt-oss 11%, ours and FreeToken produce identical 256-token outputs on 0 of 30 problems, and ours and llama.cpp on 2 of 30. The supplement records that job 073's "identical outputs" clause failed (0–3%). "Paired by problem" therefore pairs different token sequences with different routing. The effect on mean speed is probably small, but the caption should say so. A teacher-forced system comparison would remove the confound for the two llama.cpp-based systems.

### W4. Scope and external validity are narrow relative to how the abstract and rules are phrased.

- One GPU model; the 4090 and 3090 appear only in the grid.
- One engine, the authors' own patch.
- Two models in the engine, and the decomposition on one model at two budgets.
- AIME-25 prompts, which were also used while developing the system.
- A convenience sample of the cheapest Vast offers.
- The crossover rules (link-to-CPU ratio ≤ 0.32: serve misses on the CPU) rest on two or three slow-link machines (Pf and TR 9960X, with 099d at 0.40).

The decomposition is of *this* deployed cache's gap. FreeToken, KTransformers or a pooled cache would decompose differently. Section 7's rules are written for "system builders" in general. They should carry the qualifier "for a per-layer cache with CPU-served misses on an RTX 5090 host". The horizon rule (0.65·C distinct experts) is exploratory, fitted on sampled text, and not tested with a real predictor in the engine.

### W5. Statistical details that a benchmarking referee will notice

- **Small-sample intervals.** Main-text intervals over machines are percentile bootstraps with n = 9 and n = 13. Appendix L concedes these are narrow and gives t-intervals (1.11–1.33 against 1.13–1.30). The main text should use the t-interval, or BCa, or at least say n.
- **Mixed summary statistics.** Summaries switch between arithmetic means of shares (Table 3), geometric means of ratios (Section 5) and ranges. Each is defensible, but the main text should say which is used where and why.
- **"Ran stably" means different things.** For the 15 decomposition machines (one process each) it means only "outputs matched". For jobs 106–108 it also means "rounds within 2%". The "one launch per GPU, the first" selection rule appears only in a figure caption and in the script.
- **Two time estimators.** Table 4 uses ratios of mean rates; Section 2 says speed ratios are ratios of mean times. I confirm they agree within 0.003, but a single estimator would be cleaner.

### W6. Registration: exemplary, with residual soft spots

- **Hand scoring and point estimates.** The 565 clauses of jobs 073–098 were scored by hand by the authors. The auto-scorer agrees on 432 of 469, and the differences are explained. Many clauses "hold" on the point estimate only: 193 of 565, and 122 of 216 in job 100. Table 1's "registered; held" should more often read "held on the point estimate".
- **Mid-job amendments.** Job 100's ninth prediction compares new hosts with a value already measured. Job 107's replacement rule was added after four hosts had reported. Both are disclosed, but neither is a clean confirmatory test, and Appendix C should say so in one line.
- **Commit time is not push time.** Commit time precedes push time. The paper's claim about push order relies on GitHub records that are not in the artifact. Including the push-event export would close this.

### W7. Significance is moderate

The bound combines a roofline with Belady reads, which is a modest conceptual step. The paper's own related work cites close relatives: WiSP, Budgeting Bytes, Zhang's MIN-with-bypass gap, and Liang's lookahead hit rates. The new contributions are:

- engine-level oracles with a measured 2×2 (whose main lesson is the interaction);
- the "read once" and "fewest admissions" lessons;
- the measurement discipline itself.

These are worth publishing at a workshop. For a fellowship case, the discipline is the strongest evidence of research quality.

### W8. Errors and inconsistencies (minor)

- Tables 23, 24 and 25 (jobs 105–107) head their prediction column "Eq. 3", but the caption and text mean Eq. (5). Eq. (3) is the calibrated model with G_fit.
- Figure 7's legend says "law's table", although Appendix D insists "It is a calibrated model, not a law". Figure 8 says "speed limit", Tables 10 and 12 say "limit", and Appendix O says "speed-of-light". Table 7 tries to reconcile "bound" and "limit"; one word throughout would be better.
- Section 5's "1.11–1.22×" for the fewest-admission set loaded by the CPU on the slowest links mixes 11% (1.11) and 25% (1.22), directly after a sentence about 11%.
- Section 5 says the fewest-admission schedule makes "0.60–0.73 of the greedy one's copies … in replay". These numbers are the engine's counters (Appendix L says so).
- The README describes an older paper: "A host-memory law", "the prediction scorecard (Appendix B)" (it is Appendix C), "W50 ≈ 0.59 (C/k)^1.33" (superseded by the distinct-experts rule), and "re-scored … on three later ones" (Appendix D says four).
- Appendix A, rule 5, says "the panel (job 099) adds intervals over hosts, the unit that varies most". Table 4, the paper's most visible comparison, has none.

---

## Clarity

The earlier rounds scored clarity 2, then 3 three times. The restructured paper is clearly better. I would still give it 3, not 4: the main text is now navigable, but individual paragraphs and the appendix remain hard work.

### What works

- **The skeleton.** Each of four bold claims in the Introduction maps to a section, and Table 1 lists each claim with its evidence type. A reader knows within one page what is registered, what is exploratory and what is derived. This is the single best clarity device in the paper.
- **Table 2 and Figure 1** define the configurations and the double read before they are used. The main text now uses the plain names ("MIN, 1 read", "deployed") rather than the codes. Appendix B's Table 6 maps the codes once.
- **The running example** (Ryzen 9 9950X, gpt-oss 11%: Eq. (1) 10.0 ms, Eq. (2) 13.0, deployed 20.6, MIN read once 15.2, prefetched 13.8) anchors the abstractions. All five numbers reproduce.
- **Figure 2 (the staircase)** is the clearest figure: one line per machine, slow links in a different colour, and two bound levels.
- **Section 7's five rules and the bulleted limitations** are crisp. The Limitations section is unusually specific, with numbers rather than hedges.
- **Labels in paragraph headings** ("(registered)", "(exploratory)", "(exploratory, not confirmed)") separate post-hoc from registered results in the main text.

### What still makes it hard to read

1. **The abstract compresses too much.** One sentence carries five fractions ("a fifth … a fifth … a sixth … a quarter … a fifth") of a "gap" the abstract never defines. The final sentence's "half of it" has an ambiguous antecedent: half of foresight's savings in host reads, not half of the gap of the previous sentence. The abstract also gives the horizon as "≈0.65 C … across 9 models", while the Conclusion gives "0.66–0.81 C" for the paper's own workload. They are two numbers for one rule, measured on two corpora.
2. **The yardstick changes between sections.**
   - Section 3 uses a share of the bound's speed (bound time over measured time).
   - Section 4 uses a share of the deployed cache's gap to Eq. (1).
   - Section 6 uses a share of MIN's gain over *admit every miss*, which on the slowest-link panel host runs at 0.42× the deployed speed.

   "Knowing the future closes about half of the gap" (Section 4) and "half of foresight's value needs 0.65 C" (Section 6) sound like the same half and are not. A one-line "yardstick" note at the head of each section, or a small table of the three denominators in Section 2, would fix this.
3. **Three G-like constants and five equations.** T_GPU (2.9 ms, non-expert, the minimum profile), G (4.1–4.7 ms, all GPU kernels, profiled per host) and G_fit (5.17 ms, fitted) all appear. Appendix D says G_fit "is neither the profiled G of Eq. (5) nor T_GPU of Eq. (2)". A sentence that has to say that shows the notation is fighting the reader. The table headers that call Eq. (5) "Eq. 3" (W8) add to the confusion.
4. **Dense, nested sentences in the result paragraphs.** "Admit as rarely as optimality allows" packs into six lines: a definition, a replay statistic, an engine statistic in parentheses, a geometric mean with a bootstrap interval, a registered-versus-observed split, and an all-machines mean with its losses. Splitting each paragraph into "what we compared / on which machines / what we found / registered or not" would help a lot.
5. **Table 4's caption** is nine lines long and carries three different bound definitions, a pooled-bound aside, an all-in-VRAM aside and a dagger footnote. The table itself reads better with the asides moved to Appendix M.
6. **The appendix reads like a laboratory notebook.** It has thirty pages, 26 tables and appendix sections A–O. Appendix C opens with a six-line parenthetical about sub-second push timings. Appendix L narrates offer IDs, credit limits and download thresholds. This is what makes the paper auditable, and I would not delete it. But a reader needs a one-page "map of the appendix", listing which appendix supports which main-text claim. The supplement could absorb the per-job narratives of Appendices C and L.
7. **Figure 3** packs seven configurations × six budgets × three hosts plus panel ticks into one row. The main-text message, "MIN, 1 read gains everywhere; the usual ways do not", could be shown with three rows instead of seven.
8. **Naming residue.** "law", "limit", "speed limit" and "speed-of-light" survive in figures and appendices next to "bound" (W8). The README uses the old vocabulary throughout.

## Questions for the authors

1. At gpt-oss 11%, cache alone and load alone each close about 1% of the gap, and together 39%. Why present Shapley halves in the abstract and Table 3 rather than "MIN's set read once: 39%; either alone: ≤ 6%"? What would rule 2 say without the symmetric split?
2. Can you state a dependency-aware bound in which CPU-executed experts wait for their layer's attention and only link copies can be issued ahead? How much of the 27–33% "serial" part survives under it on fast-link and on slow-link machines?
3. Was the bandwidth probe ever run with the host truly idle, after the download? How do B_host, B_c and B_p change when it is? On 100f, what explains an engine that implies 1.16–1.22× the probe's highest reading?
4. In Table 4, can you give the ours/FreeToken interval over hosts? Can you run at least the two llama.cpp-based systems teacher-forced, so that both decode identical routing?
5. Why the *minimum* of 14 profiles for T_GPU rather than each machine's own or the median? How do Table 3's "serial" and "rest" change with the median (3.16 ms)?
6. Was the 0.5 link-to-CPU threshold chosen before or after job 099? Do the Table 3 shares change materially at 0.45 or 0.6?
7. Are the deployed path's semantics identical across patches `oracle` through `oracle5`? A short invariance table would justify pooling.
8. Can the GitHub push-event export behind Appendix C's timing statement be added to the artifact?

## What would raise my score

- Rewrite the abstract and Table 3 to report the 2×2 directly: alone, alone, and joint. Keep Shapley as a secondary convention, and label "serial" and "rest" as assumption-based (a profiled constant and a residual).
- Add the dependency-aware bound and restate the headroom and rule 5 against it.
- Re-probe idle; report the bound's sensitivity to the probe reading in the main text; flag the probe-relative upper end of the 31–54%.
- Give Table 4 an interval over hosts, or label its intervals as within-launch, and state that the systems decode different text.
- Fix the Eq. 3/Eq. 5 headers, the naming residue, the mixed-budget range and the README.
- Add a one-paragraph "yardsticks" note and a one-page map of the appendix.

With the first three of these done, I would move to 7/10 and clarity 4.

## Scores

| Criterion | Score |
|---|---|
| Overall (1–10; 6 = acceptable with revisions, 8 = strong) | **6** |
| Soundness (1–5) | **4**: every number I checked reproduces; the deductions are for interpretation (W1, W2, W3) |
| Methodology (1–5) | **4**: the registration, the machine as unit and the gates are exemplary; a single probe taken during downloads, within-launch CIs on the headline table, and a post-hoc class threshold keep it from 5 |
| Significance (1–5) | **3** |
| Clarity (1–5) | **3** |
| Confidence (1–5) | **4** |

---

## Claims checked against raw data

All values are from my own code over the raw files named in "Materials read". "✓" means the paper's number reproduces to its printed precision, or within bootstrap noise for intervals.

| # | Claim (where) | Paper | Recomputed from raw | Verdict |
|---|---|---|---|---|
| 1 | R*, MIN-with-bypass reads per token, gpt-oss 11% / 25% (Eq. 1 input) | 38.315 / – | 38.315 / 15.340, from the 084c routing trace with my step-level MIN. Note: request-level MIN gives 38.62 (+0.8%), so the step semantics should be stated. | ✓ |
| 2 | R* on the first 20 problems changes by −0.4% (§9) | −0.4% | −0.38% at both 11% and 25% | ✓ |
| 3 | Bound share of speed, gpt-oss 11%, 25 consumer machines (abstract, §3) | 31–54% | 25 distinct GPUs; 31.2% (105b, TR 9960X) to 54.0% (100f) | ✓ (top end probe-sensitive, W2) |
| 4 | Same CPU model differs by up to 29% (§2) | 29% | 29.1% (285K: 099a 18.14 ms against 099f 14.06 ms) | ✓ |
| 5 | Relaunch within 3.2% (§2) | 3.2% | 3.2% (TR 9960X, 105b 9.12 against 106a 9.41 ms); others ≤ 2.8% | ✓ |
| 6 | Table 3, gpt-oss 11%: 20 / 19 / 15 / 27 / 20% with 95% CIs | [16,23] [16,23] [11,18] [24,29] [15,25] | 19.5 [16,23] / 19.3 [16,23] / 14.6 [11,18] / 26.9 [24,30] / 19.6 [15,25] | ✓ |
| 7 | Table 3, gpt-oss 25%: 29 / 7 / 20 / 33 / 11% | as printed | 28.6 / 6.7 / 20.5 / 33.4 / 10.9 | ✓ |
| 8 | Best oracle closes 54% [49,58] on fast links, ≤ 17% on slow links (§4) | 54 [49,58]; 17 | 53.8 [49,58]; 17 (099d and 099f) | ✓ |
| 9 | Prefetched oracle at 1.07–1.38 × Eq. (2), 11% (§4) | 1.07–1.38 | 1.07–1.38; at 25% the minimum is 0.99 (below Eq. 2) | ✓ (contradicts "no oracle removes it", W1) |
| 10 | T_GPU = 2.9 ms, the smallest Nsight profile (§3) | 2.9 | 2.94; 14 profiles span 2.94–3.38, median 3.16 | ✓ |
| 11 | Running example: Eq.1 10.0, Eq.2 13.0, deployed 20.6, MIN 1 read 15.2, prefetched 13.8 ms (§5) | as printed | 10.02, 12.96, 20.64, 15.17, 13.82 (O4, 096a) | ✓ |
| 12 | With two reads MIN's set is no faster than the deployed one at 11% (§4) | – | Time ratio 0.97–1.02 on 13 fast machines | ✓ |
| 13 | The 2×2 at 11%, fast links (Appendix F, Table 11) | interaction 38 (panel) | Load alone 0.9%, cache alone 1.2%, joint 38.8%, interaction 36.8% (13 machines) | ✓ (basis of W1) |
| 14 | Fewest-admission, 1 read, at 11%: GM 1.21× [1.13,1.30] on 9 stable machines; 1.17× over 13; lost on two (§5) | as printed | 1.215 [1.129,1.304]; 1.166; losses 106c 0.85×, 107e 0.97× | ✓ |
| 15 | Fewest-admission copies 0.60–0.73 of greedy; misses +2.5–5.3% (§5) | as printed | 0.597–0.599 (11%), 0.726 (25%); +2.47–2.52%, +5.14–5.26% (engine counters, not replay) | ✓ (wording, W8) |
| 16 | Spearman of the MIN-1-read gain against the link-to-CPU ratio: 0.89 [0.63,0.97] over 19 machines (§5) | as printed | 0.886 [0.62,0.98], same 19 GPUs | ✓ |
| 17 | Machines carry 96% of the variance of that gain, problems at most 3% (§5) | 96 / ≤ 3 | 96.2% / 1.0% (15 machines × 20 problems, 11%) | ✓ |
| 18 | At link ratio 0.28–0.32, greedy 1 read loses 0.85–0.93× at 11%; fewest-admission served by the CPU 1.11–1.22× (§5) | as printed | 0.85–0.93 ✓; 1.11 (11%) and 1.22 (25%) | ✓ (range spans budgets, W8) |
| 19 | Admission margin: reads −4–6% at 11%; 1.021× / 1.018× (GM, 5 stable machines) (§5) | as printed | −4.3 to −5.8%; 1.0215 / 1.0177 | ✓ |
| 20 | Implied read rate 0.88–1.16 of B_host over 39 consumer launches at 11% (§4, App. K) | as printed | 39 launches; 0.880 (099i) to 1.164 (100f) | ✓ |
| 21 | Eq. (5) missed by up to 11% on a new consumer machine (§4) | 11% | gpt-oss cells of 107b −7.0% to −7.3% (raw); the 11% is Qwen3 12.5% (48.3 against 54.4 ms, Table 25); Qwen3 G not recomputed | partly |
| 22 | Read-schedule microbenchmark reaches 86–96% of the bound's read time (§3, job 102) | 86–96 | Layer mode 0.861–0.963; token mode 0.929–0.974 | ✓ |
| 23 | Table 4, host B, gpt-oss 11%: 69.9 against 54.0 tok/s, 1.294 [1.278,1.312] | as printed | 69.9 / 54.0; 1.294 [1.279,1.312] (job 081, launch 2) | ✓ |
| 24 | Table 4, host S, all six rows; ours / llama.cpp 2.0–4.0× | as printed | All six ratios and CIs to 3 decimals (job 089); 2.00–4.02× | ✓ |
| 25 | Table 4 percent of bound, gpt-oss 11/25%: B 41/25, S 41/27 | as printed | 40.6/25.4, 41.2/26.8 (B_host 87.5 and 71.3 GB/s) | ✓ |
| 26 | Appendix A rule 3: ratios of total time within 0.003 of ratios of mean rates | 0.003 | Maximum difference 0.003 | ✓ |
| 27 | Factorial: MIN 1 read +16–51%, prefetched up to +81%; usual ways ≤ 15% or lose at the smallest budgets (§5) | as printed | 16.0–51.4%; 81.2%; maximum 14.8% (no bypass, 1 read, Qwen3 12.5%, O4) | ✓ |
| 28 | Exact 16-token window recovers 0.78–0.93 of the gain in time at 11% (§6) | as printed | 0.78–0.93 (10 panel machines) | ✓ |
| 29 | Layer-ahead copy 1.04× where link ≈ CPU, down to 0.72× at 11% (§6) | as printed | 1.042 (O4, ratio 1.04); 0.722 (Pf) | ✓ (n = 1 for the gain) |
| 30 | Gates: job 107 7 / 3 / 2; job 108 6 / 2 / 1; server launches 8, of which 3 valid (§9, App. C) | as printed | V0 logs and round spreads confirm every count | ✓ |
| 31 | Parity: KL 0.0019 / 0.0005 nats (§9) | as printed | 0.00186 / 0.00053 (job 090) | ✓ |
| 32 | Every prediction committed before its machine started; at least 2.7 s (App. C) | 2.7 s | All 124 rentals; minimum 2.7 s (105f); post-start header edits are the disclosed amendments of jobs 100, 102, 105, 106 and 107 | ✓ |
| 33 | Audit: median 13.6% over 20 in-class published rows (§3) | 13.6 | 13.6 from Table 31 (derived, not raw); 9.5% over all 29 trace rows | ✓ |
| 34 | 124 rentals, 74 offers, $87.2 (App. O) | as printed | 124 / 74 / $87.22 | ✓ |
| 35 | No configuration runs below Eq. (1) (implicit in "bound") | – | Closest is 1.33× (100b, MIN prefetched, 11%) | ✓ |

### Two further observations from raw data that the paper does not report

1. The bandwidth probe ran while the model download was still in progress on most hosts (W2).
2. The systems in Table 4 decode different greedy text: ours and FreeToken agree on 0 of 30 problems on host B at gpt-oss 11% (W3).
