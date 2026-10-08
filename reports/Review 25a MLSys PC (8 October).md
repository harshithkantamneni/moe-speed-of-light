# Review 25a: MLSys 2027 main track, PC member

**Submission:** "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth"
**Date:** 8 October 2026

## Materials read

- `paper/paper.pdf`, all 45 pages. I read the main text (pp. 1–11) and Appendices A–N (pp. 15–45) as rendered pages.
- `paper/supplement.pdf`. I read the opening scorecard pages and Table S19 (job 113), and searched the rest for specific clauses.
- `paper/ieee-paper.pdf`, all 10 pages. `paper/ieee-supplement.pdf`: pages 1–10, 14, 24, 29–30 as rendered pages, plus a text extraction of the whole file (searched for unresolved references, table numbering and caption styles).
- LaTeX: the preamble of `paper/paper.tex`, and a search of all `paper/*.tex` for `\resizebox`.
- Job-script headers (registered predictions) on the `gpu` checkout: 099, 102, 109, 110, 112 and 113 in full or in their prediction blocks, plus the prediction lines of 105 and 107. For 100 and 111 I diffed the header between its commits.
- Git history of `/home/claude/gpu-branch`: commit times only (`%h %ad`, `%ct`) for the job scripts and result directories. I read no commit messages.
- Raw results I parsed with my own code:
  - `ec_*.jsonl` and `st_*.json` for jobs 095, 096a/b, 099a–j, 100a/b/c/f, 101b, 103a–e, 104a–c, 105a/b/e/f, 106a–e, 107b–e, 108a/b/d/f, 109a/c/d/e/f, 110b–e, 111d/f/g, 112a/b/d/g and 113b/d;
  - `bs1.jsonl` for 080, 081, 082 and 089;
  - `concur.txt`, `cores.txt` and `cpu.txt` for all of these;
  - `g_prof.json` for every job that has one;
  - the Nsight category medians `prof_C*.json` for 069c and 105;
  - `readsched_C*.txt` for 102;
  - `par_*.jsonl` and `summary.txt` for 090;
  - the routing traces `084c/route_aime25_gptoss.npz` and `084b/route_aime25_qwen3.npz`;
  - `manifest.json` start times;
  - `gpu/vast_ledger.json`.
- Authors' code, read **only for definitions**:
  - `scripts/job109.py`: the ratio, capture and gap definitions;
  - `jobs/ec2/fetch_table.py`: `bandwidths()`, which defines B_c and B_p;
  - `scripts/speed_limit.py`: `host_rates()`, `limit()` and the S and D constants;
  - the docstring of `scripts/speed_limit_v2.py`;
  - `scripts/decomp_measured.py` and `scripts/sumlaw_paper.py`: the T_GPU definition.
- My scripts are in `…/scratchpad/rev25a/`: `rstar2.py`, `load.py`, `t_*.py`.

## Independence statement

I did not open anything under `reports/` except to write this file. I did not open:

- `prereg/*outcome*.md`;
- `research_notes/`;
- any review, number-check, plan or progress-log file;
- `paper/paper_v1_prereview.tex`;
- `apply/`.

I did see file *names* in directory listings of `paper/`, `scripts/` and `prereg/`. The `prereg/` listing was filtered to exclude outcome files. I read no commit messages, and I changed nothing in either repository.

One disclosure. My first implementation of MIN with bypass processed each step's requests in router order and gave R* = 38.625 at gpt-oss 11%, which put Table 3's bound about 0.6% too low. To understand the definitional difference I looked at the `exact` and `B_host` entries of the authors' `prereg/speed_limit_v2.json` and the docstring of `speed_limit_v2.py`. The difference was this: the optimum may serve a step's hits before deciding its evictions. I then reimplemented MIN at step level myself (`rstar2.py`). That implementation gives 38.315 and 15.340, the values registered in the job 109 header, and every bound below uses my own R*.

---

## Summary

The paper studies batch-1 decode of MoE models whose experts mostly live in host DRAM: gpt-oss-120b everywhere, plus Qwen3-30B-A3B in the system comparison and one factorial. It makes five contributions.

1. **A bound.** Eq. (1) is the host reads of Belady's MIN with bypass, at C slots per layer, timed at the machine's highest probed host-read rate. Two variants follow: a demand bound that adds non-expert GPU time in series (Eq. 2), and an ordered bound that splits the reads between the CPU and the link (Eq. 3).
2. **An engine.** The authors' llama.cpp expert cache beats FreeToken and stock llama.cpp on two hosts. On RTX 5090 machines with consumer CPUs it reaches 31–54% of the bound at gpt-oss 11%.
3. **A decomposition.** Oracle configurations are built into the engine (MIN's set; one read versus two; read-ahead copies). They close about half of the deployed cache's gap to the bound. A late registered control on two machines shows the following. In the deployed read path, MIN's set does not pay when it is read twice, because the engine's in-step fetches displace the set. With those fetches off, MIN's set read twice does pay (1.09–1.26×).
4. **No foresight.** Four variants built without foresight recover at most 6% of what the read-ahead oracle gains, and the machine's link-to-CPU ratio decides which read path pays. A trend fitted on RTX 5090s predicts the oracles' speed on five RTX 4090s within 7%.
5. **A trace study.** On traces, half of foresight's value needs a window that covers about 0.65·C distinct experts per layer.

Every engine experiment from job 073 on carries predictions committed before the machine started. The appendices score about 1,700 clauses, including 323 failures.

## Strengths

1. **The numbers reproduce exactly from raw data.** I checked 44 quantitative claims, listed in the table below, against per-problem rows, engine counters, probes and traces with my own code. All reproduce to the printed precision, apart from three cells at an x.xx5 rounding boundary and the KL figure, whose full-vocabulary dumps are not in the repository. This is rare, and it makes the paper a reliable record.
   - Every cell of Tables 4, 5, 6 (job 109 and the RTX 4090 rows), 12, 21, 23 and 30 reproduces.
   - So do the host-B and host-S gpt-oss rows of Table 3, the frozen trend coefficients and the R* constants.
2. **Registration is real and auditable.**
   - Across the 94 rentals of jobs 093–113, every job-script commit precedes its rental, by at least 2.71 s, and every machine start, by at least 29 s.
   - No header was committed after its machine started.
   - The two later header edits I diffed (jobs 100 and 111) are disclosed amendments. The predictions in Table 1's "Registered threshold" column match the job headers I read: 099 P3/P5, 102 P2, 109 P1–P11, 110 Q1, 112 T1–T2 and 113 H1–H6.
3. **Failures and weak tests are reported.**
   - Table 1 marks thresholds that were "loose" and results that were "post hoc".
   - Table 11 lists failures, and the job 109 notes say which bands "could hardly have failed".
   - The trend's post hoc miss on the new RTX 5090s (5 of 18 cells, up to 18%) is in the main text.
   - The 2×2's reinterpretation after job 113 is stated in the abstract.
4. **Eq. (1) is a useful yardstick.** Rule 1, "report the distance to the machine's bound", is well supported. The demand and ordered refinements (Eqs. 2 and 3) are a clean way to say which reads foresight can and cannot free.
5. **The machine is the unit.** Same-CPU panel machines differ by up to 29% (I get 18.14 against 14.06 ms on two Core Ultra 9 285Ks). Treating the machine as the statistical unit, and reporting rounds and rentals separately (Appendix D), is the right design for rented consumer hardware.
6. **Insights builders can act on.** Examples are the in-step fetch table undoing a better admission set, the link-to-CPU ratio deciding the read path, and foresight horizons measured in distinct experts rather than tokens. All are concrete and come with mechanism-level counters.

## Weaknesses (most important first)

**W1. The decomposition's "what is cached" factor is not isolated, yet Table 4, Fig. 2 and Table 1 still present it as if it were.**

Table 4's first row is "MIN's set alone (MIN-2R)", and Fig. 2's legend reads "MIN's set, 2 reads". The engine's own counters show that this arm does not hold MIN's set:

- On job 109 at 11%, MIN-2R reads 1.73 R* host experts per token. The deployed cache reads 1.66 R* and MIN-1R reads 1.04 R*.
- On job 112, Few-2R misses 54.7–55.7 per token, against 40.7 for Few-1R.

The reason is that the fetch table copies misses the schedule did not choose. So "MIN's set alone closes 0%" is a property of this engine's fetch table, not of the set. The paper says this in the text of Section 4, but only after Table 4 and Fig. 2 have framed the result as a causal split.

The replacement evidence, job 113, comes from two machines and one registered budget. Two of its clauses failed on the Core Ultra 9 285K: there, turning the fetches off made the deployed cache faster (1.07), and one read beat two by only 0.024. The Conclusion correctly says "on two machines". Table 1 row 4 is scoped "as deployed", but it still names "MIN's set" for an arm that never held it. The decomposition should be rerun with the fetch-free arms on the panel, or the rows relabelled as the engine states they are. As it stands, the paper's centerpiece (the title's "where the seconds go") measures this engine's read path as much as foresight.

**W2. Every causal claim rests on one engine, one model and one workload.**

- Sections 4–7 are gpt-oss-120b at two budgets, 20 teacher-forced AIME-25 problems, batch 1 and one engine.
- Qwen3 appears only in Table 3 and the three-machine factorial.
- The title and abstract are generic.

Section 9 does say the rules come from one engine and "elsewhere they are hypotheses to test". But several rules go beyond their evidence:

- **Rule 4** ("where the link is slow, serve them on the CPU and copy in the background") rests on two machines below a third. On Pf (ratio 0.28–0.29, jobs 104 and 105) and the Threadripper 9960X (ratio 0.32, job 105), Few-2R ran 1.11–1.22× while MIN-1R ran 0.86–0.98×; I reproduce these from raw rows. The line at one third was drawn after these data. The sentence in Section 7 that states it is not labelled post hoc, and no job registered it.
- **Rule 3** ("Do not expect a policy without foresight to get there") generalizes from four variants built from two mechanisms of this engine, plus one learned admission order.

**W3. The bound is relative to the probe, and the probe's uncertainty is not carried into the headline numbers.**

- B_host is the maximum of about 20 probe readings, each taken over 1.5 s, mostly while the model downloads.
- The link-to-CPU ratio moved by up to 12% between rentals of one machine.
- On 12 of 84 launch-budgets, the engine's reads imply a rate more than 5% above the probe's best reading (up to 1.22×).

The 31–54% range and Table 4's intervals reflect variation over problems and machines only. The sensitivity note in the Table 4 caption (5–11 points for a B_host 10–22% higher) is good, but it is not reflected in the headline range.

The range's two ends are also its least representative machines:

- **31%:** a Threadripper 9960X with 178 GB/s of 8-channel memory. It counts as "consumer" under a classification set after the data. I find 31.2% on its first launch and 30.6% on its relaunch.
- **54%:** a Ryzen 9 9950X behind a slow link (100f, 54.0%). The paper itself says the bound is probably too low on that machine.

**W4. The registered program carries uneven evidential weight.**

- Several thresholds were easy to meet:
  - the per-layer read time at ≥50%, where the header's own analytic estimate was 0.85–0.99;
  - the MIN-1R share band of 0.25–0.55, where the panel showed 33%;
  - capture of at most 35%.
- Tests that could fail often did: Eq. (4) failed all four of its registered tests; the job 112 control failed on 5 of 8 and 7 of 8 machine-budgets. Applied post hoc, the trend also missed 5 of the 18 cells on the new RTX 5090s. Overall, 323 of 1,687 clauses failed.
- n is small for t-intervals over machines: 4–5 in Table 4's New column.
- The "second card" test spans essentially two ratio points. Three of the five RTX 4090s sit at ratios 0.37–0.40, and two of those are the same CPU model.

Table 1 should say which claims were hard tests, and how far the measured effect sits from its threshold.

**W5. The paper's novelty is mostly in measurement method rather than in new ideas.**

The bound combines MIN-with-bypass read counts (used by Zhang 2026b and Liang et al. 2026b for hit-based comparisons) with a roofline at a probed rate. Related bounds exist: WiSP, Budgeting Bytes, MoE-Lens.

What is new is the time-domain decomposition inside a real engine, the read-path finding and the horizon rule in distinct experts. Each of these depends on this patch's specific mechanisms (FETCH table, mailbox, paced copies). The paper would benefit from one paragraph stating what a reader could not have predicted from MIN, a roofline and their own profiles.

**W6. Smaller soundness points.**

- **"What is left" is attributed with one constant.** The attribution uses T_GPU = 2.94 ms for every machine. That value is the minimum of 14 Nsight profiles, which range from 2.94 to 3.38 ms. Per-machine residuals then range from −13% to +47% (my recomputation). "About half" is a mean over heterogeneous machines.
- **The oracle family limits the conclusion.** The read-ahead oracle is one hit-optimal design (MIN-1R plus a paced 3-step lead), not a time-optimal schedule. "Half the gap remains" may therefore partly reflect the oracle's design.
- **KL cannot be recomputed from the artifact.** The full-vocabulary dumps are not in the repository, though the NLL deltas do check.
- **The system comparison is narrow.** It uses two hosts, one of them a single machine rented three times. FreeToken's fetch caps were left at defaults, while ours's FETCH table is tuned per machine. Job 098 mitigates this.

## Clarity

I score clarity **3/5**. The structure works. Reading line by line is still slow, and the main reason is density rather than organization.

### What works

- **Table 1 is a claims ledger.** It gives each claim's registered threshold, result, evidence label, machine count and section, so a reader can audit the paper from one table.
- **Table 2 names configurations systematically.** The admitted-set by read-path grid (MIN-1R, MIN-2R, Few-1R, Dep-1R…) is a good naming scheme, and the yardstick block defines speed, share of the bound, share of the gap and capture in one place.
- **The setup is explicit.** Fig. 1 shows one decode step. The assumptions of Eqs. (1)–(3) are stated next to each equation.
- **Section 5 uses a running example** (Ryzen 9 9950X: 10.0 / 13.0 / 20.6 / 15.2 / 13.8 ms), which anchors the ratios.
- **Section 9's five rules are crisp** and come with an explicit scope sentence.
- **The appendices are navigable.** Appendix A maps them, Table 7 maps codes to names, and Table 9 maps each claim to its clause IDs.

### What still makes it hard to read

1. **The abstract cannot be parsed without the body.** Lines 012–026 use, undefined:
   - the greedy schedule versus the fewest-admission set;
   - read once versus read twice;
   - "the engine as deployed";
   - in-step fetches;
   - the link-to-CPU ratio.

   Two sentences are especially hard:
   - "The experts MIN's greedy schedule keeps pay in the engine as deployed only when each admitted expert is read once, not twice."
   - "Reading it once gains 0.02–0.16 more of the deployed cache's speed." This is a difference of two speed ratios, a unit used nowhere else.
2. **Section 4 asks the reader to revise Table 4 mid-section.** The subsection "As deployed, what is cached and how it is read pay only together" (lines 261–273) and Table 4's row label "MIN's set alone" are qualified in the next subsection (lines 277–296), which says that MIN-2R never held MIN's set. The table and Fig. 2's legend were not relabelled. Lines 284–296 pack into one paragraph:
   - four measured ranges;
   - three registered thresholds;
   - two failures on one machine;
   - a result at a different budget;
   - a derived percentage ("59–93% of what reading it once gained", where "once kept" is ambiguous).
3. **There are too many names, sets and yardsticks.**
   - The main text uses about 15 configuration names. The appendices add about 25 codes (base, foa, aa, dk, lrn, pf, R1, R2, bypassplanS, both3p…) and machine labels (O1–O6, Pa–Pj, hosts A/B/S, "Pd again", "Pg again (5950X)").
   - Two different sets are each called "15 machines": the panel (line 257) and the stable Few-1R machines (lines 135–137, 336).
   - There are five yardsticks and two baselines. Fig. 5's caption has to warn "Baseline here: admit every miss, not the deployed cache".
4. **Results are written as dense parentheticals.** Typical forms are "MIN's greedy set read twice ran 1.09–1.12× (registered: at least 1.10 and 1.03; held)", where two thresholds map onto two quantities in an unstated order, and "(Registered as gain bands per budget on two machines; held at most budgets.)" (lines 323–325), whose referent is unclear. Table 1's machine-count notation ("10 (15) + 5", "a (b) + c") needs its caption to decode.
5. **Fig. 2's x-axis category "MIN-2R (or Dep-1R)"** is a different state for the solid and dashed median lines. Fig. 4 uses five marker and colour encodings with three fitted lines.
6. **The appendices read as a lab notebook.** Appendix D–I is about 20 pages of narrative ordered by job number ("Jobs 099–104", "Jobs 105 and 106", …). The reader needs Table 9 to get back from a job to a claim. Much of this belongs in a table: per job, its claims, n, verdict and pointer.
7. **The typesetting is uneven.**
   - `\resizebox` scales narrow tables (Tables 17, 23, 24, 29, 30, 31) to roughly twice the body font, while Table 13 prints at about 5 pt.
   - Tables 11 and 32 use a different caption style (run-in "Table 11:") and caption width.
   - Page 19 is half empty above Table 10.

These are mostly mechanical fixes, but together with item 3 they make the paper feel larger than its argument.

## Claims checked against raw data

"Mine" is my own computation from raw rows, counters and probes. Ratios are base time over config time within a process, with the geometric mean over rounds. Bounds use my step-level MIN-with-bypass R*.

| # | Claim | Location | Paper | Mine (raw data) | Verdict |
|---|---|---|---|---|---|
| 1 | MIN-with-bypass host reads R*, gpt-oss 11% / 25% | Eq. (1); job 109 header | 38.315 / 15.340 | 38.315 / 15.340 (084c trace, 30 problems) | ✓ |
| 2 | R* on the first 20 problems differs by −0.4% | App. L | −0.4% | −0.38% at gpt-oss 11% and 25%; +0.8% at 40%; +0.5 to +2.7% for Qwen3 | ✓ where used (gpt-oss host-bound); statement unscoped |
| 3 | Host B bound, datasheet GPU: gpt-oss 11%, 25%; Qwen3 12.5% | Table 3 | 172 / 430 / 96 tok/s | 172.3 / 430.4 / 96.1 (B_host 87.5 from 081 `concur.txt`) | ✓ |
| 4 | Bound with measured GPU rate, gpt-oss 25%, host B | Table 3 | 279 | 279.4 | ✓ |
| 5 | Host B tok/s: ours / FreeToken / llama.cpp, gpt-oss 3 budgets + Qwen3 3 budgets | Table 3 | e.g. 69.9 / 54.0 / 34.9 | 69.88 / 53.99 / 34.93, …; all 18 cells match (`bs1.jsonl`, 080–082) | ✓ |
| 6 | Ours ÷ FreeToken and ÷ llama.cpp, host B | Table 3 | 1.29, 1.27, 1.15, 1.03, 1.15, 1.05; 2.00, 2.73, 3.20, 2.05, 2.86, 3.81 | identical | ✓ |
| 7 | Host S gpt-oss rows and B_host | Table 3 | 57.9 / 94.1 / 133.6; 1.21 / 1.20 / 1.09; 71 GB/s | 57.90 / 94.12 / 133.62; 1.206 / 1.196 / 1.093; 71.3 | ✓ |
| 8 | Ours at 31–54% of the bound on consumer machines, 12–46% on server ones (gpt-oss 11%) | §3; Table 1 | 31–54%; 12–46% | consumer RTX 5090 launches 30.6–54.0% (TR 9960X 31.2% first launch; 100f 54.0%); server 12.4–46.1% | ✓ (ends set by the HEDT part and by 100f) |
| 9 | Job 109 machines at 41–48% | §3 | 41–48% | 41.5–48.0% | ✓ |
| 10 | Table 4, panel column (15 machines; 16 cells) | Table 4 | 0, 0, 33, 15, 52; 27, 21, 4; at 25%: 21, −3, 31, 21, 48; 34, 19, −5 | 0.5[−1.8,2.8], 0.0, 32.6[21.0,44.3], 14.9, 52.5[42.6,62.3]; 27.4, 21.4, 3.7; 20.6, −3.2, 31.3, 20.6, 48.1; 33.9, 19.1, −5.0 | ✓ all cells and intervals |
| 11 | Table 4, New column (job 109; 16 cells) | Table 4 | 3, 0, 35, 19, 46; 28, 18, 1; at 25%: 20, −2, 27, 24, 48; 38, 16, −5 | 3.4[−0.9,7.7], −0.4, 34.7[21.9,47.5], 19.0, 46.3; 28.2, 17.5, 0.5; 19.8, −2.4, 27.3, 24.4, 48.3; 37.9, 15.6, −5.1 | ✓ |
| 12 | MIN-1R closes 39% on the 13 fast-link panel machines | §4 | 39% | 38.8% | ✓ |
| 13 | Interaction positive on 9 of 10 panel main-job machines and 5 of 5 new ones | §4; Table 1 | 9/10; 5/5 | 099f negative (−0.48 ms); 109a–f +1.5 to +7.5 ms | ✓ |
| 14 | Unexplained residual at 11%: two slow-link panel machines / three slow-link RTX 4090s | §4 | 35–47% / 31–36% | 35.5, 47.4 / 31.5–35.8 | ✓ |
| 15 | Table 5, all cells (job 113) | Table 5 | 0.98/1.07, 1.12/1.09, 1.26/1.23, 1.23/1.32, 1.40/1.34; misses 55.6/43.0, 54.7/43.0; ratios 0.74/0.56 | 0.981/1.073, 1.122/1.094, 1.259/1.231, 1.235/1.321, 1.397/1.344; 55.64/42.99, 54.71/42.99; 0.744/0.556 | ✓ |
| 16 | Fetch-free Few-2R misses vs with fetches; host reads vs fetch-free deployed | §4; App. D | 0.77–0.79 (11%), 0.72–0.74 (25%); 0.78× | 0.773/0.786; 0.718/0.742; 0.785 | ✓ |
| 17 | One read over fetch-free two-read arm (registered ≥0.05; fails on 285K) | §4; Table 1 | +0.02–0.16 | +0.162 (5950X), +0.024 (285K) | ✓ |
| 18 | Few-2R in-step fetches; Few-2R vs Few-1R misses | §4 | 18.6–24.3; 54.7–55.6 vs 40.7 | 18.65–24.35; 54.7–55.7 vs 40.7–40.9 | ✓ |
| 19 | Table 6, RTX 5090 job 109 (36 cells incl. capture; the 5950X's 25% round is absent, as printed) | Table 6 | e.g. 285K 1.22 / 1.46 / 1.02 / 4% | all reproduce (285K capture 4.5%, 9800X3D 4.1%) | ✓ |
| 20 | Table 6, RTX 4090 rows and bracketed trend predictions | Table 6 | e.g. 14900KF 0.96 [0.99], 1.13 [1.11], 26% | 0.963 [0.989], 1.128 [1.111], 25.7%; all rows reproduce | ✓ |
| 21 | Trend frozen in job 110 header = least squares on the 15 panel machines | §7 | 8 coefficients | my refit reproduces all 8 to 4 decimals | ✓ |
| 22 | RTX 4090s within 7% (11%) and 5% (25%) of the trend; "no change" misses by 46% / 69% | §7 | 7% / 5%; 46% / 69% | max \|dev\| 6.4% / 4.9%; 46.5% / 68.9% | ✓ |
| 23 | Trend misses 5 of the 18 new-5090 cells, by up to 18% (post hoc) | §7 | 5/18, 18% | 5/18, 17.6% | ✓ |
| 24 | No-foresight best variant; capture; variants' reads | §6 | 1.007–1.024×; ≤6%; 1.57–1.76 R* | 1.007–1.024; 5.8%; 1.57–1.76 | ✓ |
| 25 | Layer-ahead variants on slow RTX 4090s; margin share there | §6 | 0.64–0.72×; up to 26% | 0.644–0.724; 25.7% | ✓ |
| 26 | Running example (9950X, gpt-oss 11%): Eq. (1), Eq. (2), deployed, MIN-1R, read-ahead | §5 | 10.0, 13.0, 20.6, 15.2, 13.8 ms | 10.02, 12.96, 20.64, 15.16, 13.81 | ✓ |
| 27 | Few-1R beats deployed on all 15 stable machines, geometric mean 1.26×; loses on two unsteady ones (ratios 0.14, 0.21) | §5; Table 1 | 1.26×; 2 losses | 15 machines all >1 (1.019–1.419), GM 1.257; 0.845 (EPYC 7302), 0.974 (EPYC 7663) | ✓ |
| 28 | Fewest-admission copies relative to greedy; extra misses | §5; App. I | 0.60–0.73; +2.5–5.3% | 0.596 / 0.722; +2.5% / +5.3% | ✓ |
| 29 | Table 12 speeds and misses; early − 2R; early misses; early extra reads | Table 12; App. D | −0.05…−0.03 (11%); 0.97–0.98; +4–7% | −0.053…−0.031; 0.968–0.980; +4.2–7.1% | ✓ |
| 30 | Read-schedule microbenchmark: per-layer share; terms | §3; Table 30 | 86–96%; 7.04 … 3.85 ms | 86, 89, 96, 96, 95, 93%; 7.04, 2.84, 9.70, 3.91, 9.56, 3.85 | ✓ |
| 31 | T_GPU = 2.9 ms, the smallest of 14 Nsight profiles | §3 | 2.9 | 2.940 (range 2.94–3.38, n=14) | ✓ |
| 32 | Profiled G on RTX 5090, gpt-oss | App. H | 4.1–4.7 ms | 4.05–4.67 (two EPYC traces empty, as stated) | ✓ |
| 33 | Eq. (3) / Eq. (1): median consumer machine; maximum | §3 | 1.00; up to 1.56 | 1.000; 1.557 | ✓ |
| 34 | Exact 16-token window recovers 0.78–0.93 of the gain in time at 11% (registered ≥0.80; lowest missed) | §8 | 0.78–0.93 | 0.78 (099f) – 0.93 | ✓ |
| 35 | Window read shares at gpt-oss 11% (W=1, 4, 16; recall 0.5 W=8; all) | Table 20; Fig. 5 | 0.18, 0.52, 0.98, 0.29, 0.43 | engine counters (099): 0.19, 0.52, 0.98, 0.29, 0.43 | ✓ |
| 36 | Table 21 (job 100), 64 cells | Table 21 | — | all reproduce (3 cells at a rounding boundary) | ✓ |
| 37 | Table 23 (job 104), all cells | App. I | — | all reproduce | ✓ |
| 38 | Same-CPU panel machines differ by up to 29% in deployed time | §2 | 29% | 285K: 18.14 vs 14.06 ms (29.0%) | ✓ |
| 39 | Output parity: NLL change; mean KL | §11; App. J | +0.16% / +0.19%; 0.0019 / 0.0005 nats | NLL +0.160% / +0.192%; KL not recomputable (dumps absent) | Partly verified |
| 40 | Registration timing: rentals of jobs 093–113, commit → rental, push → start | App. D | 94; ≥2.7 s; ≥26 s | 94; min 2.71 s; start ≥29 s after the last header commit; no header commit after any start | ✓ |
| 41 | Rental ledger totals | App. N | 147 rentals, 86 offers, $107.5 | 147, 86, $107.5 | ✓ |
| 42 | Published systems' median share of their bound (20 in-class rows) | §3; App. M | 13.6% | 13.6% from Table 32's rows (arithmetic only; not re-derived from the sources) | ✓ (arithmetic) |
| 43 | Single-read oracles, gpt-oss 11% (job 095) | Table 13 | 55.5 tok/s; MIN-1R 1.29; hits 72; 1.04 R*; read-ahead 1.39; Belady-2R 0.91 | 55.4; 1.287; 72; 1.04; 1.394; 0.909 | ✓ |
| 44 | Below a third of the CPU rate, Few-2R pays instead of MIN-1R | §7 | 1.11–1.22× | Few-2R 1.11–1.22× while MIN-1R 0.86–0.98× on Pf (104a, 105e) and the TR 9960X (105b) | ✓, but post hoc and unlabelled (two machines) |

### Registered tests, evidence labels and post hoc labelling

- **Headers against Table 1.** Table 1's registered thresholds match the job headers I read. Job 113's H3–H6 match Table 1's "in-step fetches" row; the manipulation checks H1 and H2 are not shown in Table 1 but are scored in Appendix D. The control row matches job 112's T1–T2, the no-foresight row matches job 109's P7, and the interaction and share rows match job 109's P1 and P3. The trend row matches job 110's Q1, which jobs 111 and 112 carried unchanged. The read-time row matches job 102's P2 (≥0.50).
- **The "loose" and "post hoc" labels are accurate** where I checked them: job 102 P2, the job 109 share band and the consumer scope.
- **Post hoc statements without a label:**
  1. Section 7: "Where the link reads at under a third of the CPU rate, Few-2R pays instead of MIN-1R (1.11–1.22×)". This rests on two machines (Pf in jobs 104 and 105; Threadripper 9960X in job 105), with the one-third line drawn after the data, and Section 9 rule 4 builds on it.
  2. Section 4: "59–93% of what reading it once gained". This is a ratio derived after job 113.
  3. Appendix L's "−0.4%" for R* on 20 problems holds only for the gpt-oss host-bound budgets. I get +0.8% at gpt-oss 40% and +0.5–2.7% for Qwen3.
- **Scoping.** The abstract and Conclusion scope most statements correctly ("in the engine as deployed", "built from the engine's own mechanisms", "on our workload", "on two machines"). The places where the evidence is narrower than the wording are:
  - Table 4's row label and Fig. 2's legend ("MIN's set");
  - Table 1 row 4's claim text, which names "MIN's set" for an arm that did not hold it;
  - Section 9 rule 2, whose imperative heading rests on a two-machine control (its body says so);
  - Section 9 rules 3 and 4.

## Questions for the authors

1. **The decomposition without fetches.** Can the 2×2 decomposition (Table 4) be rerun with fetch-free two-read arms on at least five panel-class machines? If not, how should a reader interpret Table 4's first row, given that MIN-2R reads more host experts than the deployed cache?
2. **The consumer range.** What is the consumer range at gpt-oss 11% if HEDT parts (Threadripper 3970X and 9960X) are excluded? Does the range move if B_host is the second-highest probe reading?
3. **Rule 4's evidence.** Rule 4's one-third line rests on Pf and the Threadripper 9960X. Has any machine with a ratio below 1/3 been tested since, under a registered prediction? And where between 0.32 and 0.49 does the crossover lie? On the 285K at 0.49, both MIN-1R and Few-2R gain.
4. **Qwen3 in Sections 4–6.** Do any of the causal results hold for Qwen3 (top-8, BF16) at panel scale?
5. **The oracle's design.** Is "half the gap remains" robust to a stronger oracle, for example a time-optimal offline schedule that accounts for copy latency and the queue cap, rather than the hit-optimal MIN-1R plus a paced lead?
6. **The fetch table's own cost.** On the 285K, turning off in-step fetches sped up the deployed cache itself (1.07×). Is the deployed fetch table mis-tuned on some machines, and does that affect Table 3's comparisons?

## What would raise my score

- Rewrite the abstract and Section 4 so the fetch-table confound comes first. Then either relabel Table 4 and Fig. 2, or replace their "what is cached" row with fetch-free arms measured on five or more machines.
- Carry the probe's uncertainty into the 31–54% headline range and Table 4. For example, report the ranges at the second-highest reading or the median across rentals, and say which machines set the ends of the range.
- Run at least one more model, for example Qwen3, through the panel decomposition and the no-foresight test. Alternatively, confine the title, abstract and rules explicitly to gpt-oss.
- Turn Appendices D–I into a per-claim evidence table, consolidate the yardsticks (one baseline per section), and cut the main text's vocabulary roughly in half.
- State in Table 1 which claims were tests that could have failed, with each effect's distance from its threshold.

## Scores

| | Score |
|---|---|
| Overall (1–10) | **6** (weak accept) |
| Soundness (1–5) | 3 |
| Significance (1–5) | 3 |
| Novelty (1–5) | 3 |
| Clarity (1–5) | 3 |
| Confidence (1–5) | 4 |

**Rationale.** The measurements are exceptionally verifiable. Every one of the 44 claims I checked reproduces from raw data (the KL figure only in part, because its dumps are not in the repository), and the registration record is clean. Against that, the causal claims are narrow (one engine, one model), and the centerpiece decomposition is confounded by the engine's fetch table; that is admitted in prose but not in the table and figure that present it. The bound is relative to the probe, and the writing remains dense. I would not champion the paper, but I would not object to it either. A fetch-free decomposition on more machines, together with a clarity rewrite of the abstract and Section 4, would move me to 7.

## IEEE format

I reviewed `ieee-paper.pdf` (10 pages, IEEEtran conference) and `ieee-supplement.pdf` (30 pages). Problems against IEEE conference conventions:

1. **The abstract is 294 words.** IEEE conferences generally ask for 150–250 words in one paragraph. Index Terms are present and alphabetical, which is correct.
2. **Heading levels skip.** Section-level text goes straight to run-in level-4 headings ("a) Models and workload:", "b) The engine:", …) with no "A." level-2 subsection headings. IEEEtran expects "A./B." subsections before "1)" and "a)".
3. **Table captions are long paragraphs in small caps.** Tables I–IV and VI are examples. IEEE table captions are short titles, with explanatory notes below the table. The small-caps transform also flattens the configuration names (e.g. "MARGIN, LA, LA-1R, LA-1R-MARGIN" in Table VI), which in the body are set in a distinct sans-serif.
4. **Equation (1) is too wide for the column.** Its number wraps onto a line of its own (p. 3, left column).
5. **Caption styles are inconsistent between the two documents.**
   - The main paper uses centred small-caps "TABLE I".
   - Supplement Tables S1–S4, S6 … use centred "TABLE Sx" with sentence-case captions.
   - S5 and S26 are longtables with run-in "TABLE S5:" captions.
6. **Reference numbering collides between the documents.** The supplement has its own list, [1]–[62], so one work carries different numbers in each document: Kalibera & Jones is [21] in the paper and [2] in the supplement, and Hwang et al. is [11] and [3]. Some entries are incomplete: [49] "Zhang, Gao, and Mitra" has no initials, and [25] "ATSInfer" has no authors.
7. **The cross-references point to a third document.** The main paper cites "Appendix D/J/M/N" and "Tables S22–S24", and these resolve; I found no "??". But the IEEE supplement (Table S3, Appendix D's last paragraph) sends readers to `paper/supplement.pdf` for the clause-level scorecard. That file belongs to the MLSys build and is not part of the IEEE pair, so the scorecard behind Table I is unreachable from the IEEE submission. The first-page footnote also names a file ("ieee-supplement.pdf") rather than "the supplementary material".
8. **Floats and fonts in the supplement.**
   - The supplement is single-column, which is acceptable for supplementary material.
   - Table S4 floats alone mid-page with large blank space (p. 5).
   - `\resizebox` gives inconsistent table font sizes: Tables S11 and S24 print at about 14 pt, while Table S10 prints at about 6 pt.
   - Figure tick labels in Figs. 3 and 5 of the main paper are about 5 pt at print size.
9. **Page count.** The main paper is 10 pages including references (references end on p. 10). That is within common IEEE conference limits of 8–10 pages plus references, but the target venue's limit should be checked. All fonts are embedded (Type 1 or TrueType), so it is PDF eXpress-safe.
