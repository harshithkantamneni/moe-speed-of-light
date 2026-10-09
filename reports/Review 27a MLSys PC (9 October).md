# Review 27a: "Where the Seconds Go" (MLSys 2027, main track, blind)

## Materials read

- `paper/paper.pdf`: the whole main text (pp. 1–10), references, and Appendices A–N (pp. 11–45). I read Appendices D, E, G, H, I, J, L and M closely and skimmed B, C, F and K.
- `paper/supplement.pdf` (the scorecard): skimmed the front matter and the job 073–075 rows to see what it contains. I did not re-score it clause by clause.
- `paper/ieee-paper.pdf` (10 pp.) and `paper/ieee-supplement.pdf` (30 pp.): skimmed for the IEEE-format section. I rendered pages 1, 2 and 6 and a tiled view of all 10 pages, and extracted the text of both.
- LaTeX: I read only `main_body.tex` lines 185–196, `wsg_numbers2.tex` and `wsg_auditours.tex`, to find which macro produces "13.6%" and "27%".
- Scripts, read for definitions only: `scripts/audit_ours.py` (what "ours, scored the same way" means). I used `prereg/speed_limit_v2.json` only to get the authors' R* and B_host so I could compare them with my own R*. I did not use their analysis code.
- Raw data (`/home/claude/gpu-branch`): the `results/<job>/` directories of jobs 080–082, 089, 090, 093–114. I used `ec_*.jsonl` (decode_ms, n_decode, nll), `st_*.json` (counters), `bs1.jsonl` (system comparison), `concur.txt` (probe), `fetch_table_law_gptoss.json` (for B_p/B_c), `g_prof.json` and `prof_summary.txt` (Nsight), `readsched_C*.txt` (microbenchmark), `validity.txt`, `cpu.txt`, `gpu.csv`, `nvidia-smi-q.txt` (GPU UUIDs), and the routing trace `results/084c_gptoss_trace@vast/route_aime25_gptoss.npz`.
- Job script headers: 080, 081, 099, 102, 109, 110, 113, 114 in full, plus the diffs of every header edit made after a job's first rental started.
- Git: `git log --format='%h %at'` per job script in the gpu branch, and `gpu/vast_ledger.json` (rental start times, cost).
- My scripts are in `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/rev27a/`:
  - `load.py`: the raw-row loader;
  - `rstar.py`: my own Belady MIN-with-bypass;
  - `j099.py`, `j109.py`, `j111.py`, `j114.py`, `t4cpu.py`, `t3.py`, `few1r.py`, `share.py`: the per-job analyses;
  - `regtime.py`: commit-versus-rental timing and header diffs.

## Independence statement

I did not open anything under `reports/` (apart from writing this file), any `prereg/*outcome*.md`, `research_notes/`, `apply/`, `paper/paper_v1_prereview.tex`, or any file named as a review, number check, plan or progress log. Directory listings showed some of those file names; I did not open them. I read no commit messages. My git commands printed only hashes and timestamps, plus `git show`/`git diff` of job-script contents. Every value in the verification table below comes from my own code run on the raw rows, counters, probe outputs or trace. The exceptions are labelled: the published-systems median (recomputed from the paper's own Table 33 values) and parity (compared with the job's on-machine summary). I changed nothing in either repository except this file, and I committed nothing.

---

## Summary

The paper studies batch-1 decode of MoE models (gpt-oss-120b, Qwen3-30B-A3B) when most experts live in host DRAM and a consumer GPU (mostly RTX 5090) holds C expert slots per layer. It makes four contributions.

1. **A per-machine bound (Eq. 1).** Belady's MIN-with-bypass host reads per token (R*), multiplied by expert size and divided by the machine's highest probed host read rate, plus variants that account for router serialisation (Eq. 2) and for CPU reads that cannot be issued early (Eq. 3). The authors' llama.cpp expert cache reaches 31–54% of this bound on 25 consumer machines at gpt-oss 11% (post hoc). It beats FreeToken at 11 of 12 configurations and stock llama.cpp at all of them, on two hosts.
2. **A measured decomposition of the gap** using oracles built into the engine: what is cached (MIN's set), how it is read (once in the step versus served by the CPU and copied in the background), and when (read-ahead). The central finding is that the engine's per-machine fetch table copies misses that MIN would bypass into slots MIN would keep, which hides the value of MIN's set:
   - on the deployed path, MIN's set gains nothing unless each admission is also read once;
   - with the in-step fetches off (job 114, three machines), MIN's set alone closes 20% of the gap and set plus single read closes 44%;
   - the read-ahead oracle still leaves about half of the gap.
3. **Negative and transfer results.** Four no-foresight variants built from the engine's own mechanisms recover at most 6% of the read-ahead oracle's gain. A trend in the link-to-CPU read-rate ratio, fitted on RTX 5090s, predicts the oracles on five RTX 4090s within 7%, but misses five new RTX 5090s by up to 18% (post hoc).
4. **Foresight horizon (post hoc, trace replay).** Half of foresight's value needs the routing of about 0.65·C distinct experts ahead.

The methodology is unusual. Every engine experiment since job 073 carried predictions in its job-script header, committed to a public branch before the rental started. Every clause is scored, including about 320 failures, and the paper labels evidence in Table 1 as derived, registered (loose or not), held, held (point), failed or post hoc.

## Strengths

1. **The numbers are real and reproducible.** I recomputed about 40 quantitative claims from raw per-problem rows, counters, probes and the routing trace. With the exceptions noted in the table, they reproduce to the printed precision, usually exactly:
   - all 12 cells of Table 3 (speeds, ratios, bootstrap intervals);
   - every row of Table 4's three column groups;
   - every cell of Tables 5, 6 (RTX 5090 and RTX 4090), 12, 13 and 31;
   - the Few-1R geometric mean and its interval;
   - the second-card deviations;
   - the slow-link residuals;
   - the round-to-round noise.

   This is rare and is the paper's strongest asset.
2. **Pre-registration is real and auditable.** For all 101 rentals of jobs 093–114, the job script's commit time precedes the rental start, by at least 2.7 s (105f). Five base headers were edited after a job's first machines had started (100, 102, 105, 106, 107, and 111's relaunch amendment). Every such edit I diffed is a host substitution or relaunch with "predictions unchanged", or the added prediction 9 of job 100. All of these are disclosed in Appendix D. Failures are reported in full, including inconvenient ones:
   - job 110 void;
   - job 112's control failed;
   - Eq. (4) failed every registered test;
   - job 114's two manipulation checks failed on the zero-table machine.
3. **The bound is a useful reporting norm for this literature.** "Distance to MIN-at-probed-rate on the same machine" is cheap to compute: one trace and one probe. It makes systems comparable in a way that speed-ups over self-chosen baselines are not. The audit of 52 published rows (Appendix M) is a useful, if rough, first application.
4. **Oracles inside a real engine, not only in simulation.** The paper finds an effect that simulation would have missed: the fetch table displaces MIN's set. It then fixes the experiment (jobs 113/114) rather than the story. The account of this in Section 4 and Appendix D is candid.
5. **Machines are treated as the statistical unit.** Many rentals and per-machine reporting (Fig. 4, Table 6) are used, and the paper is honest that machines differ more than problems. I reproduced a 29% spread between two Core Ultra 9 285K panel machines.
6. **The limitations section is unusually concrete.** It covers probe sensitivity, one engine, AIME prompts used in development, three deviations from registration, a convenience sample, and few slow links.

## Weaknesses (most important first)

1. **The external validity of the decomposition, which is the paper's core, is narrow, and the clean version rests on three machines.**
   - All engine results come from one engine. The decomposition uses one model (gpt-oss) at two budgets, teacher-forced on its own greedy AIME text, which was also used during development.
   - The headline mechanistic finding (in-step fetches displace MIN's set) is a property of this engine's fetch-table design.
   - The fetch-free 2×2 that the abstract quotes ("the set alone closes 20% … both together 44%") ran on three valid machines:
     - two are Ryzen 9 5950Xs;
     - one of those (114b) had an all-zero fetch table, so its "deployed" and "CPU-only" paths are the same configuration;
     - the other 5950X (114f) is the same GPU as 109f and 112g (UUID f90ec646), on its third rental.
   - The over-machine summaries are therefore imprecise. My recomputation matches Table 4's CPU-only column: "Both" is 44% [−3, 92], Dep-1R 10% [−2, 22], left 39% [12, 65].
   - The per-machine ratios are tight and were registered, which is fine. But the abstract and Section 9 rule 2 present 20%/44% and "15–22% / 22–56%" without saying that the across-machine uncertainty spans zero for "Both". The 265K (114c) shows set+once at 23% of the gap against about 56% on the two 5950Xs.
2. **Two Table 1 rows are not marked "loose" although, by the paper's own definition, they should be.**
   - Job 114's header states that job 113 had already measured MIN-2R CPU-only at 1.09–1.12× (11%) and 1.28–1.36× (25%) on the same path. H3 then registered thresholds of ≥ 1.05 and ≥ 1.15, which the earlier data made easy to meet. Table 1's "With the in-step fetches off" row shows these thresholds without "(loose)".
   - The "faster than FreeToken / ≥ 1.8× llama.cpp" row (job 081) followed job 080 on the same machine and jobs 073–078 elsewhere. Appendix D itself says that jobs 076–081 contain "sign-only predictions made after a same-machine pilot".
   - Table 1's labels are the paper's main device for telling the reader how much each claim is worth, so they must be applied uniformly. Job 105's Few-1R > 1 clause on Pf, a machine where job 104 had just measured Few-1R > 1, is a milder case of the same thing.
3. **The "bound" is a bound relative to one probe run, and the probe is not an upper bound on the machine's read rate.**
   - By the paper's own Eq. (4) accounting, the engine reads faster than the probe's best reading by more than 5% on 12 of 84 launch-budgets, by up to 1.22×.
   - The probe mostly ran while the model was downloading, and the link-to-CPU ratio moved by up to 12% between two rentals of one machine.
   - Both ends of the headline "31–54%" are set by probe extremes. The low end is the Threadripper 9960X (105b), whose probe reads 178 GB/s (31.2%). The high end is job 100f (54.0%), where the paper itself says the engine's reads imply a rate above the probe.
   - Excluding those two, the consumer range is about 38–52%. A median and IQR over machines would be more informative than the min–max.
   - Smaller points: T_GPU (2.94 ms) is the minimum of 14 profiles. The "attributed" rows of Table 4 therefore depend on a choice the reader cannot see in the main text, and the residual's near-zero value on fast links is partly a consequence of that choice.
   - R*: the paper's "exact" MIN gives 38.315 reads per token at C=14. My sequential per-request MIN gives 38.63 (+0.8%), and their "no-evict" variant matches mine exactly (39.03). Their "exact" presumably lets a step's requests be reordered. That is a valid, slightly more optimistic lower bound, but it should be defined in the text.
4. **Several claims are worded more broadly than the evidence.**
   - Abstract and rule 3: "policies without foresight recover at most 6%". What was tested is four variants built from this engine's admission margin and layer-ahead copy, on five fast-link machines, with a capture threshold the paper itself labels loose. The trace study (Table 18) supports the broader statement only for classic online replacement policies and only in replay.
   - Section 7: "Where the link reads at under a third of the CPU rate, Few-2R pays instead of MIN-1R (1.11–1.22×)". These two numbers are the two budgets of one valid machine (Pf, ratio 0.29, Table 24). The only other sub-⅓ machine with Few-2R (EPYC 7663) failed the round check.
   - Introduction: "Scored like-for-like on datasheet rates (post hoc), ours reaches a median 27% and published systems 13.6%". The procedure is the same, but models, hardware, traces (some sibling variants), budgets and timing methods differ. The 27% is a median over our own 12 Table-3 cells, including GPU-bound budgets; the 13.6% is over a post-hoc in-class subset of 20 of 52 rows. "Like-for-like" overstates this.
   - Section 12: "MIN's set pays alone where the read path keeps it, more when read once" carries no qualifier, although it rests on three machines (see weakness 1).
5. **"The ratio predicts the oracles on a second card" is weaker than Table 1 suggests.**
   - Three of the five RTX 4090s sit at almost the same ratio (0.37, 0.39, 0.40), two of them the same CPU model with near-identical results.
   - At those ratios the trend predicts MIN-1R ≈ 1.0, which "predict no change" also predicts. Most of the gap between "the trend" and "no change" comes from the read-ahead oracle and the two higher-ratio machines.
   - Applied to the five new RTX 5090s, the same frozen trend misses 5 of 18 cells beyond the band (worst 17.6%, on 109e's read-ahead oracle at 11%).
   - The paper reports all of this. Still, the claim should be stated as "the ratio predicts the sign, and the size within about ±18%", which is what Section 7's last paragraph effectively says.
6. **Much of the gap is attributed rather than measured, and the read-ahead oracle is one implementation.** "Left after all three" (about half the gap) is split by attribution into T_GPU in series plus extra reads at B_host. The read-ahead oracle uses paced 16 MB copies on one copier thread with a 64-copy queue cap (Appendix E). "Even the read-ahead oracle leaves about half of the gap" is therefore a statement about this oracle, not a floor on what foresight can achieve. On slow links 35–47% stays unexplained.
7. **The workload is narrow.** Batch-1 only; 256 tokens; 20 problems in later jobs; the cache is carried across problems; three engine versions are pooled in the panel; teacher forcing fixes routing but removes any interaction between caching and generation. Parity is measured only as KL (which matches the job-090 summary). There is no task accuracy, which is acceptable for a systems paper.

## Verification against raw data

Notation: R* = 38.315 per token at C=14 and 15.340 at C=32 (the authors' value, cross-checked against my own MIN below). S = 13,253,760 B. B_host = the maximum of the probe's CPU, link and concurrent readings in `concur.txt`. "Speed vs X" = T_X / T_config, geometric mean over rounds of the same process. Intervals are 10,000-resample paired bootstraps over problems, or t-intervals over machines.

| # | Claim | Location | My value (raw data) | Verdict |
|---|---|---|---|---|
| 1 | Host B speeds: llama.cpp / FreeToken / ours, all six cells (e.g. gpt-oss 11%: 34.9 / 54.0 / 69.9) | Table 3 | identical in all 18 numbers (jobs 080, 081, 082) | ✓ |
| 2 | Host S speeds, all six cells (e.g. Qwen3 43.75%: 23.8 / 98.3 / 95.8) | Table 3 | identical in all 18 numbers (job 089) | ✓ |
| 3 | Ours ÷ FreeToken with CI, 12 cells (e.g. B g11 1.29 [1.28, 1.31]; S q44 0.97 [0.96, 0.99]) | Table 3 | identical, to the second decimal and the interval bounds | ✓ |
| 4 | Ours leads FreeToken at 11 of 12 and llama.cpp at 12 of 12; ours ÷ llama.cpp 2.00–4.02× | §3, Table 3 | 11/12 (S q44 below 1); 12/12; 2.00–4.02 | ✓ |
| 5 | Bound 172 tok/s for gpt-oss 11% on host B (41% of it) | Table 3, Table 30 | 172.3 with their R*; 170.9 with my sequential MIN (R* 38.63, +0.8%); 69.9/172 = 40.6% | ✓ (definition of "exact" MIN should be stated) |
| 6 | "No-evict" MIN → 169 tok/s | Table 30 | my no-evict MIN 39.025 per token (theirs 39.026) → 169.2 | ✓ |
| 7 | R* changes by −0.4% on the first 20 problems | App. L | −0.39% (my MIN) | ✓ |
| 8 | Running example: Eq. (1) 10.0 ms, Eq. (2) 13.0, deployed 20.6, MIN-1R 15.2, read-ahead 13.8 | §5 | host O4 (096a): 10.02, 12.96, 20.64, 15.16, 13.81 | ✓ (machine not named in text) |
| 9 | Microbenchmark reaches 86–96% of the bound's read time per layer; token, link and CPU modes | Table 31, §3 | 86.1% / 96.3% / 95.1% (C=14); token 93.3 / 97.4 / 95.5; link 35.4 / 94.6 / 53.1; CPU 92 / 86 / 91 | ✓ |
| 10 | Ours at 31–54% of the bound on consumer machines, gpt-oss 11% | Abstract, §3, Table 1 | min 31.2% (TR 9960X, 105b, B_host 178 GB/s), max 54.0% (100f) | ✓ (both ends are probe extremes) |
| 11 | The 5 new machines fall at 41–48% | §3 | 41.5–48.0% (109a, c, d, e, f) | ✓ |
| 12 | Table 4 Panel (15), 11%: set 0 [−2, 3], once 0 [−2, 2], both 33 [21, 44], ahead 15 [11, 19], left 52 [43, 62] | Table 4 | identical, every value and bound (096a/b, 099a–j, 100b/f, 101b) | ✓ |
| 13 | Table 4 Panel (15), 25%: 21 [19, 22], −3 [−7, 1], 31 [23, 39], 21 [18, 23], 48 [41, 55] | Table 4 | identical | ✓ |
| 14 | MIN-1R closes 39% on the panel's 13 fast-link machines | §4 | 38.8% | ✓ |
| 15 | Table 4 New (5), 11%, all 8 rows (e.g. both 35 [22, 48]; T_GPU 28 [20, 36]; residual 1 [−3, 4]) | Table 4 | identical except "reads beyond MIN's": mine 17 [15, 20] vs printed 18 [15, 20] | ✓ (one-point difference) |
| 16 | Table 4 New (4), 25%, all rows | Table 4 | identical | ✓ |
| 17 | Table 4 CPU only (3), 11%: set 20 [10, 29], once 10 [−2, 22], both 44 [−3, 92], ahead 14 [5, 22], left 39 [13, 65], in-step fetches 3 [−10, 16] | Table 4 | 19.7 [10, 29], 9.9 [−3, 22], 44.7 [−3, 92], 13.7 [5, 22], 38.5 [12, 65], 3.1 [−10, 16] | ✓ (within 1 point) |
| 18 | Table 4 CPU only (3), 25%: 37, 7, 45, 21, 37, −3 with their intervals | Table 4 | 37.2 [20, 54], 7.4 [−13, 28], 44.7 [5, 84], 21.1 [16, 26], 37.0 [11, 63], −2.7 [−17, 12] | ✓ |
| 19 | Table 5, all 21 cells (e.g. 265K: 1.06, 1.00, 1.10, 1.01, 1.03, 1.16, 0.4 [0.3, 0.5]) | Table 5 | identical. The unlabelled first column is the deployed cache, CPU-only ÷ with table (base0/base). | ✓ |
| 20 | MIN-2R CPU-only 1.10–1.12× (11%) and 1.28–1.36× (25%); Dep-1R 1.03–1.07× | Table 1, §4 | 1.102–1.124; 1.276–1.357; 1.026–1.069 | ✓ |
| 21 | CPU-only interaction +0.4–3.2 ms at 11%; −0.1 to 0.1 at 25%, above zero on one machine | Table 1, App. D | 0.36, 3.10, 3.18 ms; −0.085, +0.043, +0.132 (only 114b's interval excludes 0) | ✓ |
| 22 | MIN-2R misses 48.6–50.8 per token with the fetches, 43.5 without; Dep-1R in-step fetches 23.4–32.0 against 8.2 admissions; MIN's set admits 20.4 | §4 | 48.62 / 50.75; 43.50; 23.42 / 31.96; 8.19; 20.39 | ✓ |
| 23 | Deployed-path interaction falls from 31–52% to 3–21% of the gap without the fetches; 21% either way on the zero-table machine | §4 | 30.7 / 52.4 → 3.2 / 20.8; 22.3 vs 21.2 | ✓ |
| 24 | Interaction positive on 9 of the panel's 10 main-job machines and on 5 of 5 new ones | Table 1, §4 | 9/10 (099f, ratio 0.29: −0.48 ms); 5/5 (1.5–7.5 ms) | ✓ |
| 25 | Table 6 RTX 5090 rows: MIN-1R, ahead, best no-foresight variant, capture, at both budgets | Table 6 | identical to 2 d.p.; capture 4.5 / 1.5 / 1.9 / 4.1 / 5.7% at 11% | ✓ |
| 26 | Best no-foresight variant 1.007–1.024×, capture at most 6% | Abstract, §6 | 1.007–1.024; max 5.7% (7945HX) | ✓ |
| 27 | No-foresight variants read 1.57–1.76 R*; the margin cuts the deployed cache's reads by 4–6% at 11% | §6 | 1.57–1.76; 4.2–4.8% | ✓ |
| 28 | RTX 4090s within 7% (11%) and 5% (25%) of the frozen trend; Table 6 bracketed predictions | §7, Table 6 | max deviation 6.6% / 5.0% using the trend coefficients in job 110's header; brackets identical | ✓ |
| 29 | "No change" would miss by up to 46% (11%) and 69% (25%) | §7 | 46.5% / 68.9% (12400's read-ahead oracle) | ✓ |
| 30 | Frozen trend misses 5 of 18 new-5090 cells beyond the band, by up to 18% | §7 | 5 of 18 beyond 0.10 in log; worst +0.162 in log = 17.6% (109e read-ahead, 11%) | ✓ |
| 31 | Few-1R: geometric mean 1.26× (1.19–1.33) on 15 stable machines; lost on the ratio-0.14 and 0.21 machines, gained on the other two unsteady ones | §5, App. I | 1.257 [1.186, 1.332] over 15 GPU UUIDs (two RTX 4090s, two server CPUs); EPYC 7302 0.845, EPYC 7663 0.974, Xeon 8347C 1.30, EPYC 9754 1.18 | ✓ |
| 32 | Fewest-admission schedule makes 0.60–0.73 of the greedy one's copies | §5 | engine counters: 11.84/19.81 = 0.60 (11%), 7.03/9.69 = 0.73 (25%) | ✓ |
| 33 | Table 12: 1R / 2R / early speeds and misses on 4 machines; early arm's misses 0.97–0.98× | Table 12, App. D | identical (e.g. 5950X 1.41 / 1.03 / 0.98; 40.7 / 55.6 / 54.5); ratio 0.969–0.980; early admits 11.72–11.75 vs 6.2–7.4 | ✓ |
| 34 | Table 13; Few-2R misses fall to 0.77–0.79 (11%) and 0.72–0.74 (25%); one read +0.02–0.16; at 25% Few-2R 1.33–1.36× vs Few-1R 1.29–1.43× | Table 13, App. D | identical; 0.773 / 0.786; 0.718 / 0.743; +0.022 / +0.166; 1.326–1.359 vs 1.293–1.427 | ✓ |
| 35 | Exact 16-token window recovers 0.78–0.93 of the gain in time at 11%, the lowest machine below the registered 0.80 | §8 | 0.78 (099f) to 0.93 | ✓ |
| 36 | Window copied in the step runs 0.89–1.32× at 11% on 4 machines, below 1 on the two lowest ratios | §8, Table 22 | 0.89, 0.89, 1.20, 1.32 (100a, 100f below 1) | ✓ |
| 37 | Slow-link residual 35–47% (panel) and 31–36% (RTX 4090s); 27–31% with the 4090's own 3.5 ms; 83–99% and 73–81% left against Eq. (3) | §4, App. L | 35 / 47; 32–36; 27–31; 83 / 99; 73–81 | ✓ (31 vs my 32 at the low end) |
| 38 | Two panel machines with the same CPU differ by up to 29% in deployed time | §2 | 285K: 18.14 vs 14.06 ms = 1.29 | ✓ |
| 39 | Ratio moves by a median 0.2% (at most 1.8%) between rounds, over 129 machine-budget-configurations | §2, App. D | n = 129, median 0.21%, max 1.75% | ✓ |
| 40 | Profiled G is 4.1–4.7 ms for gpt-oss; two EPYC traces held almost no kernels | App. H, I | RTX 5090 profiles 4.05–4.67 ms; 108a/108b G ≈ 0.00/0.02 | ✓ |
| 41 | Every commit preceded its rental by at least 2.7 s (101 rentals, jobs 093–114); 154 rentals, 90 offers, $112.7 | App. D, N | 101 rentals, minimum lead 2.7 s (105f); 154 / 90 / $112.7 | ✓ (push times not checkable offline) |
| 42 | Published in-class median 13.6% over 20 rows | §3, App. M | 13.6 = median of the 20 trace-scored rows of Table 33 without Fate, SP-MoE and MoE-SpeQ (recomputed from the paper's table, not from raw data) | ✓ (internal consistency) |
| 43 | Parity: KL 0.0019 / 0.0005 nats, top-1 98.5 / 99.3%, loss +0.16 / +0.19%; stock llama.cpp 0.0017 / 0.0008 | App. J | matches `090…/summary.txt` (on-machine computation; I did not recompute from logit dumps) | ✓ (summary-level) |
| 44 | T_GPU = 2.9 ms, the smallest of 14 profiles | §3 | not reproduced: needs a kernel classification I could not match from `prof_*_tail.csv` | not checked |

None of the checked numbers was wrong. The only discrepancies are the definition of the R* "exact" MIN (+0.8% under my sequential implementation) and one-point rounding differences in rows 15, 17 and 37. One artifact nit: `results/090_parity_a100@vast` actually ran on an RTX PRO 6000, as `gpu.csv` and the paper say. The directory name is stale.

### Registered tests, evidence labels and post-hoc labelling

- **Headers against Table 1 thresholds.** Job 114 H3/H4/H5 match the "CPU-only" row. Job 113 H3/H4/H5/H6 match the "fewest-admission set pays read twice" row. Job 109 P1/P3/P7 match the deployed-path 2×2 and no-foresight rows. Job 110 P1, frozen trend coefficients included, matches the second-card row. Job 102 P2 matches the read-time row. Job 081 P2/P4 and job 089 match the system-comparison row. I found no threshold in Table 1 that differs from its header.
- **Timing.** See row 41. Post-start header edits occur in jobs 100 (prediction 9 added about 100 min after the first four starts, disclosed), 102, 105 and 106 (host substitutions, "predictions unchanged"), 107 (amendments) and 111 (relaunch amendment). All are disclosed in Appendix D.
- **Labels.** "Loose" is correctly applied to job 102's ≥ 0.50 floor and to job 109's share band and capture bound. It is missing for job 114's H3 and job 081's FreeToken/llama.cpp thresholds (weakness 2). "Held (point)" is used correctly where no interval exists, as for the counters and job 102.
- **Post-hoc labels.** The consumer scope, the published-system audit, the trend fit, the RTX 5090 trend misses and the horizon rule are all labelled post hoc or exploratory. One unlabelled post-hoc generalisation is the Few-2R "under a third of the CPU rate" sentence in Section 7, which rests on one machine.

---

## Clarity

Earlier rounds scored this 2/5 and then 3/5. This version is readable by a determined expert, and its structure is good. The prose still makes the reader do too much bookkeeping. I score it 3/5 again: better organised than a typical measurement paper, harder to read than it needs to be.

**What works**

- **The introduction's claim-by-claim structure.** Each paragraph opens with a bold claim and a section pointer, and Table 1 gives every claim its registered threshold and outcome. A reader can see in one place what was predicted and what happened. This is the paper's best expository device.
- **Table 2.** A single place for configuration names, machine sets and yardsticks. The "read once / read twice / read ahead" grid of the 2R/1R names is a good idea.
- **Figure 1.** A clear schematic of one decode step and where the second read happens.
- **The running example** in Section 5 (one machine, five times) anchors the abstractions.
- **Table 4** now separates "measured" from "attributed", and its footnotes explain the asterisked row.
- **Section 9's rules** state their own scope ("one engine, batch-1, one prompt set; rules 2–4 from one model at two budgets"), which is how such rules should be written.
- **Appendix D** is precise and unusually honest about deviations.

**What still makes it hard to read**

1. **The abstract and introduction lean on undefined terms.** The abstract uses "fetch table", "in-step fetches", "the set", "read once" and "cache-fulls of distinct experts" before any is defined. Take "with it, the set pays only if each admitted expert is also read once": it can be parsed only after reading Section 4. A sentence explaining that the engine has two read paths (CPU-served with a background copy, or copied in the step) would unlock the rest.
2. **There is too much vocabulary at once.** The main text uses about 15 configuration names (deployed, Dep-1R, MIN-1R/2R, Few-1R/2R, Few-2R-early, Margin, LA, LA-1R, LA-1R-margin, Belady-1R/2R, read-ahead oracle, admit every miss). It uses 7 machine sets (panel, panel main job, new, O3–O5, hosts B and S, RTX 4090s, "CPU only") and 5 yardsticks with different baselines. Fig. 5 switches the baseline to "admit every miss", and the caption has to shout it. Job numbers (096–114) also leak into main-text captions (Tables 4 and 6, Appendix pointers). A reader must keep Table 2 open throughout. Configurations used only once in the main text (Belady-1R/2R, LA-1R-margin, Few-2R-early) could move to the appendix.
3. **Section 4 tells the story in the order the experiments happened, not the order a reader needs.** It first presents the deployed-path decomposition, then explains that its "set alone" row does not measure the set (the asterisk), then presents the fetch-free 2×2. Table 4's first row therefore means different things in different columns, and Fig. 2's second x-axis state means different things for grey and blue lines. Leading with the fetch-free 2×2 as the clean measurement, and presenting the deployed path as "what the fetch table does to it", would remove most of the confusion.
4. **Table 1 is too dense to serve as a scorecard.** Its cells are run-on prose ("held on B; on S, five of six ratios fell outside ±0.06 of B's; faster at 11 of 12 in all"), so a reader cannot tell at a glance whether a claim stands. A one-word verdict column (supported / partly / failed / post hoc) beside the prose would make it usable. The row "MIN's reads at the probed rate bound any exact-routing cache" cites Eq. (4), which lives in the appendix.
5. **Sentences are long, number-dense and pronoun-heavy.** Example: "On the two machines of the second test with a nonzero table, most of it was the table's: it fell from 31–52% of the gap to 3–21% without the in-step fetches". Here "it" is the interaction's share of the gap, which is introduced two sentences earlier. Another: "There, MIN-2R closes 0% of the gap at 11%, Dep-1R 0%, and MIN-1R 33% on the panel (39% on its 13 fast-link machines) and 35% on the new machines". Most paragraphs of Sections 3–8 pack 5–10 numbers with parenthetical registration notes ("(registered: positive)").
6. **Table and figure details.**
   - Table 5's first data column, "CPU only vs table", has no group header. It is the deployed cache, but it sits next to the MIN-2R group header, and I had to recompute it to find out what it was.
   - Fig. 2 encodes five line and marker types across two read paths in two small panels.
   - Fig. 3's tick marks for the panel are hard to separate from the markers.
   - Table 6 mixes two different machine populations (RTX 5090s for Section 6 and RTX 4090s for Section 7) under one caption.
7. **There are three bounds and two closely related quantities.** Eq. (1), Eq. (2) and Eq. (3) are introduced in quick succession, and the text then says the bound is "the larger of the two" (Eqs. 1 and 3). Eq. (4) (G, which differs from T_GPU) appears in Table 1 and Section 3 but lives in Appendix H. A short sentence saying which quantity each later section uses would help: Eq. (1) everywhere, Eq. (2) only in Fig. 2, Eq. (3) only for slow links.
8. **The main text leans on appendix tables for load-bearing claims.** The probe check (Table 29), the bound's tightenings (Table 30), the microbenchmark (Table 31) and the published audit (Appendix M) are all appendix material. That is defensible given the page limit, but the main text should state the one number each one contributes.

---

## Questions for the authors

1. How exactly is the "exact" MIN computed? My sequential per-request MIN with bypass gives R* = 38.63 at C=14 against your 38.315; your no-evict variant matches mine exactly (39.03). Do you let a step's k requests be reordered, or treat them as one set? If so, please say so; the bound remains valid.
2. Over the three CPU-only machines, "Both" is 44% [−3, 92]. Is the 265K's 23% (against about 56% on the 5950Xs) explained by its ratio (0.57) or by its fetch table? Would you accept stating the abstract's 20%/44% as per-machine ranges (15–22% and 22–56%)?
3. Why are job 114's H3 thresholds (≥ 1.05, ≥ 1.15), set below job 113's measured 1.09–1.12 and 1.28–1.36, not labelled "loose" in Table 1? Likewise for job 081's thresholds after the same-machine pilot in job 080.
4. The 31% end of the consumer range comes from a Threadripper 9960X with a 178 GB/s probe. Did its engine reads approach that rate? If not, is the probe's concurrent reading achievable by the engine's access pattern on HEDT memory? What is the median and IQR over the 25 machines?
5. How sensitive are the "attributed" rows of Table 4 and the near-zero fast-link residual to choosing the minimum T_GPU (2.94 ms) rather than each machine's own profile?
6. Is there any evidence, beyond Pf, that Few-2R beats MIN-1R below a ratio of ⅓?
7. Could the read-ahead oracle be made stronger (unpaced, multiple copier threads, no queue cap)? How much of "about half remains" is this oracle's implementation?
8. Do you expect the fetch-table displacement effect to occur in other engines that split misses between the CPU and PCIe (FreeToken, DALI)? Can Eq. (1) plus your oracle methodology be run on FreeToken?

## What would raise my score

- Rescope the abstract, rule 3 and the conclusion. "Four variants built from the engine's own mechanisms" instead of "policies without foresight". Per-machine ranges and n for the CPU-only 2×2, with the 5950X duplication and the zero-table machine noted. Drop "like-for-like".
- Mark job 114 H3 and job 081 as "loose" in Table 1, and add a one-word verdict column.
- Define the "exact" MIN. Report the consumer range as a median and IQR alongside the min–max. Move the T_GPU choice into the main text.
- Rewrite Section 4 to lead with the fetch-free 2×2 and present the deployed path as the engine-specific case, then simplify Fig. 2 accordingly.
- Prune configuration names from the main text, and label Table 5's first column.
- Run more fetch-free 2×2 machines, especially non-5950X ones and at least one slow-link machine. This would move the core finding from "three machines" to a class claim and would most directly raise significance.

## Scores

| | |
|---|---|
| Overall | **6 / 10** (weak accept) |
| Soundness | **4 / 5**: every number I checked reproduces and the registration is genuine; points lost for two unlabelled loose thresholds, a probe-relative "bound", and some over-broad wording |
| Significance | **3 / 5**: a useful reporting norm (distance to a per-machine bound) and engine-level findings, but the decomposition is engine-specific and the clean version is thin |
| Novelty | **3 / 5**: Belady-with-bypass times bandwidth is a roofline-style bound with precedents (WiSP, Budgeting Bytes, Zhang 2026b compare with MIN); the in-engine oracle decomposition and the distinct-expert horizon rule are new |
| Clarity | **3 / 5** (see the Clarity section) |
| Confidence | **4 / 5** |

---

## IEEE format (ieee-paper.pdf, 10 pp.; ieee-supplement.pdf, 30 pp.)

What conforms:
- IEEEtran two-column conference layout, letter paper.
- Bold "Abstract—" and "Index Terms—" in alphabetical order.
- Roman-numbered section heads, italic run-in paragraph heads ending in a colon, an IEEE-style numbered list in Section IX.
- Numbered references [1]–[39] with IEEE abbreviations.
- "TABLE I" small-caps titles above tables and "Fig. n." captions below figures.
- A separate supplement with its own [S1]–[S62] list.

What breaks or strains IEEE conventions:

1. **Table captions are long, multi-sentence explanations set in IEEE's small-caps title style.** Table I's caption defines five evidence labels in small caps and is hard to read; Tables III–VI likewise. IEEE table titles should be short; the definitions belong in a note under the table, as the MLSys version effectively does.
2. **The note "Appendices A–N and the tables, figures and equations numbered S1, S2, … are in the supplementary material available online"** sits at the foot of column 1 on page 1 as an unmarked paragraph. It should be a proper first-page footnote, or a sentence at the end of the introduction.
3. **Float placement.**
   - Table I goes to the top of page 2 although it is first referenced on page 1. That is acceptable for a double-column float.
   - Section IV is referenced on pages 4–5, but Tables IV and V appear together at the top of page 6, after Section V has begun.
   - Fig. 3 and Table VI share page 7, and Figs. 4 and 5 share page 8.

   Floats cluster one page or more after their first reference. Several double-column floats in a row also leave little running text on pages 6–8.
4. **Inconsistent cross-reference forms.**
   - Running text uses "Eq. (S1)" (22 times), but the column headers of Tables S19–S21 use "Eq. S1".
   - The supplement mixes "Figure 3" / "Figure S3" with "Fig. S3" (Appendix B rule 12; Appendix J "Figure S3 builds…"). IEEE uses "Fig." in running text.
   - The main paper uses Roman table numbers (I–VI) while the supplement uses "TABLE S1–S27". This is acceptable, but some IEEE venues expect "TABLE S-I" for consistency.
5. **Document naming collision.** The IEEE supplement (the appendices) refers to "the scorecard … supplement.pdf in the artifact". This collides with "supplementary material" in the main paper and with the MLSys build's `supplement.pdf`. Readers of the IEEE package will not know which "supplement" is meant. Call it "the scorecard (artifact)".
6. **Page budget.** The main paper fills exactly 10 pages including references, and the supplement adds 30. Check the target venue's limit; many IEEE conferences count references within 8–10 pages. Several main-text claims depend on supplement-only tables (S23, S24, S25), which some venues do not allow for load-bearing evidence.
7. **Minor.**
   - Configuration names (MIN-2R, Dep-1R, …) are set in sans-serif inside Times body text.
   - Some references carry non-IEEE annotations, such as "arXiv:2506.20675 (2025)" appended to a 2026 MLSys proceedings entry ([28]).
   - The supplement duplicates main-text references under different numbers (e.g. Belady, Berger), which is normal for self-contained supplements but should be consistent.
   - The anonymous author block is fine for review.
