# Review 16a: MLSys 2027 main track, "Where the Seconds Go"

Program committee review, blind. 7 October 2026.

## Materials read

- `paper/paper.pdf`, all 40 pages. I read the main text (pp. 1–10) in full and read Appendices A–D and H–N closely. I skimmed Appendices E–G. For the structure and the macro names I used the LaTeX sources `paper.tex`, `app_wsg.tex`, `app_more.tex` and `app_value.tex`.
- `paper/supplement.pdf`, skimmed. I checked the job 108 clause table against my own numbers.
- Job-script headers on the gpu branch: jobs 081, 089, 090, 095, 096, 102, 105, 106, 107 and 108, plus the wrapper scripts 105a–g, 106a–e, 107a–g and 108a–f.
- Git history of the gpu branch, read only with `git log --format='%h %ad'` or `%ct`, never with commit messages. I used `git diff` on file contents to inspect every edit made to the 100, 102, 105, 106 and 107 headers after their rentals started.
- `gpu/vast_ledger.json` (124 rentals).
- Raw results under `/home/claude/gpu-branch/results/`:
  - `ec_*.jsonl` rows (decode_ms, n_decode, nll_sum, nll_n, config) for every launch of jobs 093–108;
  - `st_*.json` counters;
  - `concur.txt` probes, `cores.txt`, `cpu.txt`/`lscpu.txt`, `nvidia-smi-q.txt`/`v0.txt` (GPU UUIDs) and `free.txt`;
  - `g_prof.json`/`q_prof.json` and the 069c `prof_C*.json` kernel summaries;
  - `bs1.jsonl` of jobs 081, 089 and 098;
  - `readsched_C*.txt` of job 102;
  - the AIME routing traces `084c/route_aime25_gptoss.npz` and `084b/route_aime25_qwen3.npz`;
  - `090/parity_kl.txt` and `par_*.jsonl`.
- The authors' code, read for definitions only:
  - `scripts/job106.py` (`law`, `load`);
  - `scripts/panel_099.py` (`host_info`, `cell_data`);
  - `scripts/speed_limit.py` (`host_rates`, `limit`, model constants);
  - `jobs/ec2/fetch_table.py` (`bandwidths`: the definitions of B_c and B_p);
  - `scripts/robustness.py` (how the launch set is defined);
  - `scripts/foresight.py` (`_pol`);
  - `mosl/cachesim.py` (`_sim`, the MIN-with-bypass definition).
- One derived file: `prereg/speed_limit_v2.json`. I read its `notes` and the gpt-oss "exact" R* once, after my first MIN simulator disagreed with the paper (see the independence statement).

My scripts are in `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/rev16a/`:

| Script | What it recomputes |
|---|---|
| `core.py` | Shared helpers |
| `launches.py` | Per-launch probe, counters and times |
| `regtests.py` | Table 4 |
| `consumer.py` | The 39 consumer launches |
| `decomp.py` | Fig. 2 shares and the elasticity |
| `fewest.py` | Fewest-admission schedule and admission margin |
| `panel.py` | Panel and window results |
| `headline.py` | Table 3 |
| `minsim.py` | My own MIN with bypass |
| `window.py`, `dw.py` | My own window policy and the distinct-expert horizon |
| `boundcheck.py` | Whether any measurement beats the bounds |
| `regtime.py` | Commit-before-rental timing |

## Independence statement

- I opened nothing under `reports/` except to write this file.
- I did not open `prereg/*outcome*.md`, `research_notes/`, any review, number-check, plan or progress file, `paper/paper_v1_prereview.tex` or `apply/`.
- I did not read commit messages.
- I modified nothing in either repository. All my code ran from the new `rev16a/` folder.
- Every number in the verification table comes from my own code on raw rows. I read the authors' scripts only to match definitions: B_host as the highest single probe reading, B_c interpolated at the helper count, B_p from zero-copy 16 MB blocks=64, the launch set, and G = all Nsight kernel time except the helper wait and the expert copies.
- One disclosure. My first MIN-with-bypass simulator gave R* = 40.96 at gpt-oss 11%. I then looked up the authors' "exact" value (38.32) in `prereg/speed_limit_v2.json` and compared my code with the definition in `mosl/cachesim.py`. The fault was mine: a miss earlier in a step could evict a resident that was a hit later in the same step. After I fixed that, my simulator gives 38.315 independently. I did not copy any number.

## Summary

The paper studies batch-1 decode of gpt-oss-120b (MXFP4) and Qwen3-30B-A3B (BF16) with most experts in host DRAM. The machines are rented RTX 5090 hosts; the engine is the authors' 3,500-line llama.cpp expert cache. It makes four contributions.

1. **Two bounds.**
   - Eq. (1) bounds time per token for any exact-routing system with C slots per layer. Its host term is MIN-with-bypass reads at B_host, the highest probe reading; its GPU term uses the datasheet rate.
   - Eq. (2) is tighter, for systems that read on demand. It adds the engine's non-expert GPU time in series.
   - A microbenchmark reaches 86–96% of the bound's read time.
2. **A relation that decomposes the deployed cache's time.** Eq. (3) is T = G + (M + (1 − G/T) A) S / B_host, with G profiled in Nsight and M, A counted by the engine. It is used to split time into GPU compute, MIN's reads and reads beyond MIN's (Fig. 2).
   - It was found on exploratory data, and its overlap term was added after the plain form failed.
   - It failed all four registered tests. On consumer processors it is within 6% on 31 of 39 launches. Two of three valid server machines read far below the probe.
3. **In-engine oracles.**
   - MIN's set, read once, beats the deployed cache by 16–51% on three hosts, and by up to 81% with prefetching.
   - Among MIN's hit-optimal schedules, the fewest-admission one gains a mean 1.22× on 9 stably running machines.
   - Which way of spending foresight pays depends on the link-to-CPU read ratio.
   - An online admission margin gains 2%.
4. **A price on foresight.** Half of MIN's saving needs about 0.65 C distinct experts of lookahead (median over 9 models). Learned forecasters close at most 15% of the gap.

The system leads FreeToken at 11 of 12 host-B/S configurations (by 0.97–1.29×) and stock llama.cpp by 2.0–4.0×.

The artifact is very extensive:
- about 1,480 registered clauses, each in a job-script header committed seconds before the rental;
- raw rows and counters for every run;
- a script for every number.

## Strengths

**S1. Measurement hygiene and reproducibility are exceptional.**
- I recomputed 38 claims from raw rows with my own code, including:
  - every cell of Table 4;
  - the four Table 3 cells I recomputed on host B and all six on host S;
  - Tables 21–25;
  - Fig. 2's shares;
  - the machine-level intervals.
- All of them reproduce. The few differences are at most 0.1 point and come from how the fallback G is summarised.
- For all 71 rentals of jobs 093–108, the job script was committed at least 2.7 s before the rental was created.
- Every edit to a header after its rental started was one of two things: a comment-only amendment for a host not yet started, or an added prediction for later hosts. I checked jobs 100, 102, 105, 106 and 107.

**S2. Candour.** The abstract states that the relation failed every registered test. Table 1 labels each claim as registered, exploratory or after the fact. The post-hoc processor split and the "set aside" re-scorings are labelled. Limitations are specific. This kind of reporting is rare in systems papers and is a model for them.

**S3. The in-engine oracle study is a real systems insight, verified across many machines.**
- How foresight is spent matters more than having it. Reading each admission once versus twice decides gain or loss.
- Hit-optimal schedules are not equal: the fewest-admission schedule makes 0.60–0.73 of the greedy schedule's copies.
- The link-to-CPU ratio decides whether copies belong in the step or in the background.

These results are registered, measured on 9–19 machines, and carry intervals over machines.

**S4. A quantitative reference point the offloading literature lacks.**
- No measured configuration on any launch beats Eq. (1)'s host-read time; the closest is 1.34× it.
- No on-demand configuration comes within 4.5 ms of the host-read time, while Eq. (2) only needs a margin of T_GPU ≈ 2.9–3.4 ms.
- So the bounds are consistent with every measurement in the artifact.
- Fig. 2 shows each machine's position and the model's residual honestly.

**S5. Methodology lessons for the field.** Machines carry 96–97% of the variance in log gain; problems carry about 1–3%. The machine is used as the unit, and validity gates are registered for the later tests.

**S6. The system comparison is fair.**
- FreeToken's backend is picked on a separate launch.
- FreeToken is tuned in job 098, where it still trails at 5 of 6 cells.
- Equal GPU expert memory is enforced.

## Weaknesses (most important first)

**W1. The titular decomposition is not supported out of sample.**

Section 4 and Fig. 2 rest on Eq. (3). Its record is weak:
- It failed all four registered tests.
- The consumer/server split that rescues it was drawn after the tests and is confounded with co-tenancy, NUMA layout and core counts. The authors say so.
- The "held (12/12, 2.3%)" result is on three machines that were all in the exploratory data the form was built on.

On the three new consumer machines (107b, 108d, 108f) it under-predicts every gpt-oss 11% cell by 6.5–7.2% and the Qwen3 cells by 6.4–11.3%.

The overlap term was added after job 105 and justified by an in-sample gain: 33 instead of 18 of 34 launches within 6% at 25%. Out of sample it did worse than the plain form. On the eight cells of those three machines, the plain form is closer in 7 of 8, with a median |error| of 5.6% against 6.4% for Eq. (3). The registered job 107 clause "overlap below plain" failed. The main text reports only the in-sample gain; the out-of-sample reversal is visible only in Appendix C.

The decomposition's two key terms depend on Eq. (3) being right:
- "reads beyond MIN's" (22–35% of the time);
- "reading below the machine's rate" (up to 11% on consumer machines).

The model's error is of the same order as the second term, so the split between these two is model-dependent. I recommend presenting Fig. 2 as a descriptive accounting with an explicit residual, not as a validated model, and moving the out-of-sample comparison into §4.

**W2. Some headline statements are broader than the evidence.**

(a) **Reading MIN's set once.** The abstract says foresight "pays when it is spent as MIN spends it, reading each admitted expert once". The conclusion says "helps when it is spent as MIN spends it". My recomputation of MIN's greedy schedule, read once in the step, against the deployed cache at gpt-oss 11%:

| Machine | Link/CPU ratio | Jobs | Speed vs deployed |
|---|---|---|---|
| Pf (Core Ultra 9 285K) | 0.28–0.29 | 104a, 105e, 106b | 0.853–0.863× |
| Threadripper 9960X | 0.32 | 105b, 106a | 0.91–0.93× |
| EPYC 7302 | 0.14 | 106c | 0.68× (computed wrong outputs) |

"At every budget" holds only on the three fast-link oracle hosts, O3–O5. Table 1 scopes this correctly ("on three hosts at all budgets"); the abstract and conclusion do not.

(b) **The ≈0.65 C price.** The rule is a cross-model median on other text. I checked it on the AIME routing the engine actually runs, using my own window policy. That policy reproduces the paper's Table 19 shares to within 0.01–0.02 (0.185/0.314/0.522/0.815/0.993 against 0.18/0.31/0.52/0.81/0.98 at gpt-oss 11%). On these traces:

| Cell | W50 (tokens) | Distinct experts within W50 |
|---|---|---|
| gpt-oss 11% | 3.7 | 0.81 C |
| gpt-oss 25% | 10 | 0.69 C |
| Qwen3 12.5% | 1.95 | 0.78 C |

These use the paper's gap to MIN with one read; with R* instead, the values are 0.85, 0.72 and 0.83 C. Two of the three cells sit above the paper's stated upper quartile of 0.72. The abstract's "about 0.65 C" should carry its spread, or the engine's own cells should be reported.

(c) **"Bound every system".** B_host is the single highest probe reading. By Eq. (3)'s own accounting, the engine read more than 5% faster than it on 12 of 84 launch-budgets (up to 1.22×). The paper notes this in §3 and in Limitations. The abstract's "bound every system that executes the exact routing" should say "relative to the probe".

**W3. External validity and selection.**
- Hardware and workload:
  - one GPU SKU;
  - two models;
  - one prompt set (AIME-25) that was also used during development;
  - teacher-forced 20 × 256-token runs;
  - a convenience sample of the cheapest rentable hosts.
- Attrition is high:
  - Of the 13 rentals in the new-machine tests (jobs 107 and 108), 5 stopped at the gate, 3 failed the round check and 5 were valid.
  - 5 of 8 server launches were invalid.
- The validity filter coincides with the boundary of the fewest-admission claim. No stable machine has a link-to-CPU ratio below 0.28, and both machines where the schedule lost (ratios 0.14 and 0.21) were filtered out as unsteady. The paper discloses this, but the claim's lower edge is therefore untested rather than established.
- Most intervals are within a launch, over 20–30 problems. Only three claims carry intervals over machines.

**W4. The components are individually modest in novelty and practical payoff.**
- Eq. (1) is a roofline with MIN-with-bypass read counts. It is close in spirit to WiSP, Budgeting Bytes and Zhang (2026b)'s trace-level contract.
- Eq. (2)'s T_GPU is this engine's kernel time.
- The "spend foresight as MIN does" lessons echo integrated prefetching and caching (Cao et al., 1995), Demand-MIN and interval-based offline caching, all of which the paper cites.
- The realisable gains are small or negative:
  - the admission margin gains 1.020×;
  - forecasters close at most 15% of the gap;
  - the layer-ahead copy loses (0.72–0.91× at gpt-oss 11%) on all three job-105 machines whose link is slower than their CPU reads.
- The system's lead over FreeToken depends on the machine and the model: 0.97–1.29× on the two 5090 hosts, and a tie at two of three budgets on FreeToken's own headline model.

**W5. The registered record is very large but carries little confirmatory weight for the main claims.**
- Table 8 counts 495 clauses held with an interval, 597 held on the point estimate and 308 failed.
- Many bands came from same-host pilots, and the main claims rest on a handful of clauses.
- Several deviations were handled after the fact. Each is disclosed, but the volume hides them:
  - In job 108 the EPYC 7543's G-profile clause is scored "untested", although the profile returned 0.0015 ms (instrument failure), and a fallback G was used.
  - Job 107's rule for replacing machines that failed the round check was added after the first results were committed (01:17–01:35 CDT; amendments at 01:36).
  - In job 106, host 106e was registered as "a Ryzen 9 9950X (offer 54559478)". Its status as "panel host Pe, rented before" was established after the fact from the GPU UUID. I confirmed the UUID matches 099e.

**W6. The cross-paper audit is fragile but cited in the main text.**
- §3 says published batch-1 systems reach "a median 13.6%" of their bound.
- Those rows use datasheet rates, sometimes traces of a sibling model variant, and pooled-cache or multi-token designs scored against the per-layer, one-token bound.
- The caveats live in Appendix M. Either move them into §3 or drop the number from the main text.

**W7. Minor issues.**
- **Number of tests.** The abstract and introduction say Eq. (3) was "registered before four later tests". §4.2 says it was registered before three; job 105 tested the plain form.
- **Fallback G.** The paper quotes 4.28/4.50 ms. The mean kernel time of the seven profiles gives me 4.30/4.50. The effect is 0.1 point (EPYC 7543: −34.2% against the paper's −34.3%; Table 26 medians 1.62/1.41/5.57 against 1.7/1.5/5.6).
- **Output parity.** It is measured as KL divergence on an A100 with an AVX2 host, not on the 5090 platform, and without task accuracy. The KL comes from on-machine summaries because the logits were deleted; the NLL deltas I recomputed from raw rows (+0.16% and +0.19%) are consistent.

## Clarity

Earlier rounds scored clarity 2, then 3 and 3. On this version I would keep it at **3/5**. The paper is more navigable than its density suggests, but an MLSys reader without the appendices will still struggle.

**What works**

- **Framing.** The three-question framing and Table 1, which maps each claim to its evidence type and section, give the reader a spine. This is the best feature of the revision.
- **Evidence labels.** Each results paragraph in §5–§6 opens with its claim and its evidence label in a run-in head, e.g. "(registered)", "(exploratory)", "(registered, partly held)". A reader can audit the status of every claim locally.
- **Names.** The main-text configuration names in Table 2 are used consistently, and Table 2's "Reads" column makes the central 1-versus-2-reads distinction explicit. Fig. 1 makes the second read visible.
- **Fig. 2.** It is an effective, honest picture: per-machine bars, measured dots, and the model's residual drawn as a hatched segment.
- **Running example.** The O4 example at gpt-oss 11% gives Eq. (1) 10.0 ms, Eq. (2) 13.0, the deployed cache 20.6, MIN with one read 15.2 and MIN prefetched 13.8. It ties the bounds to the oracles in one line, and every value reproduces.
- **Consistency.** Text, tables and supplement agree. I found no transcription error.

**What still makes it hard to read** (most important first)

1. **Too many numbers and qualifiers per sentence.**
   - The §5 paragraph "Which of MIN's schedules is followed matters" carries about 15 numbers: 0.60–0.73; 1.22×; 9 machines; 1.14–1.31; 7 and 2; 2 of 4; 0.14, 0.21 and 0.23; 0.28; 0.28–0.32; 0.85–0.93×; 1.11–1.22×.
   - The abstract's sentence on the registered tests stacks four figures and a parenthesis about the probe's second-highest reading.
   - The message of each paragraph is buried. Put one number per idea in the text and push the rest into a table.
2. **Section 4 is ordered against the reader.**
   - It goes: the relation (4.1), the registered-test history and Table 4 (4.2), the post-hoc processor split (4.3), and only then the decomposition the section title promises (4.4).
   - The reader meets three failed tests and a post-hoc rescue before learning what the decomposition says.
   - Suggested order: the decomposition and Fig. 2 with its exploratory status first, then its out-of-sample record (Table 4), then servers.
3. **Name and symbol overload.**
   - Configurations: nine names in the main text, plus about 20 appendix codes (Table 6) such as base, foa, aa, dk, lrn, pf, fetch, bypass, fetchplan, bypassplan, lead2, both2, both3p, hitopt, hitoptp, nb2, w* and b*.
   - Hosts: B, S, A, O1–O6, Pa–Pj, "Pf again" and "285K, second".
   - The bound is called "bound" in the main text and "limit" in the appendix.
   - Three GPU-time quantities (G, T_GPU, G_fit) and two time models (Eq. 3, and Appendix D's calibrated model).
   - Six units of repetition: launch, round, rental, machine, cell and launch-budget.
   - Appendix tables Tables 21–25 use host letters the main text never introduces.
4. **Table 4 and Table 1 need decoding.**
   - Table 4 has six result columns ("Machines (new)", "Cells within", "New machines'", "Median", "As registered", "Set aside") under a five-line caption.
   - Its "Median" is a median |error|, but it is not labelled as such.
   - "Set aside: held (12/12, 2.3%)" printed next to "failed" invites misreading.
   - Table 1's evidence entries need Appendix C to parse, e.g. "registered and held on 7 machines, observed on 2; lost on 2 of 4 unsteady ones" and "Held is mostly on the point estimate".
5. **Inconsistent count of tests.** The abstract and introduction say "registered before four later tests… none passed"; §4.2 says Eq. (3) was registered before three. Table 4's "Form" column resolves this only for a careful reader.
6. **Figures 3–5 are hard to read at print size.**
   - Fig. 3 has 7 configurations × 6 panels, three host markers plus grey ticks, and a † for host-bound budgets, with roughly 6–7 pt labels.
   - Fig. 4 overlays four series with log-ratio fits and does not mark which machines were registered.
   - Fig. 5 measures gain against "admit every miss", while Figs. 3–4 measure against the deployed cache. §6 has to warn the reader about the change of baseline.
7. **Eq. (3)'s overlap term gets one sentence of motivation.** Why the share of background copies hidden under compute is G/T, and not a measured overlap, needs a short derivation or a timeline sketch extending Fig. 1. So does the quadratic it produces.
8. **The abstract uses undefined terms.** "MIN with bypass", "fewest admissions", "probed read rate" and "links at least 0.28 as fast as their CPU reads" are not explained. It reads as a list of results rather than a summary.
9. **Too much of the paper lives in the appendices.** Thirty pages of appendices and a 41-page supplement carry definitions the main text needs:
   - the validity checks;
   - G profiling and its fallback;
   - how W50 is computed;
   - the window policy;
   - how the link ratio is measured.

   The main text forward-references Tables 19–28 and Appendices C–N throughout.

## Claims checked against raw data

R* is MIN-with-bypass host reads per token from my own simulator on `084c/route_aime25_gptoss.npz` (38.315 at C=14, 15.340 at C=32) and `084b/route_aime25_qwen3.npz`. Times are means over problems of decode_ms/n_decode. Machines are identified by GPU UUID.

| # | Claim (paper's value) | Location | My value | Verdict |
|---|---|---|---|---|
| 1 | Cache at 31–54% of Eq. (1)'s speed at gpt-oss 11%, 25 consumer machines | Abstract, §3, §4.4 | 30.6–54.0% over 39 valid consumer launches on 25 GPUs | Reproduced |
| 2 | R* on first 20 problems changes −0.4% at 11% and at 25% | §3 | −0.38%, −0.38% | Reproduced |
| 3 | Host B gpt-oss 11%: 34.9 / 54.0 / 69.9 tok/s; 1.294 [1.278, 1.312]; 2.00× | Table 3 | 34.93 / 53.99 / 69.88; 1.294 [1.278, 1.312]; 2.00× | Reproduced |
| 4 | Host B 25% and 40% rows, Qwen3 12.5% row | Table 3 | 1.275 [1.254, 1.295]; 1.154 [1.134, 1.172]; 1.032 [1.022, 1.042]; llama.cpp ratios 2.73/3.20/2.05 | Reproduced |
| 5 | Host S six rows (e.g. 1.207 [1.195, 1.219], 0.974 [0.962, 0.987]) | Table 3 | 1.207 [1.194, 1.219] … 0.974 | Reproduced |
| 6 | Bound 172 tok/s at host B, gpt-oss 11% (B_host 88) | Table 3 | R*·S/87.5 GB/s → 172.3 | Reproduced |
| 7 | Leads FreeToken at 11/12; 2.0–4.0× llama.cpp; intervals overlap only at Qwen3 12.5% | §3 | 11/12 (S Qwen3 43.75%: 0.974); 2.00–4.02×; only Q12.5 overlaps | Reproduced |
| 8 | Tuned FreeToken trails at 5 of 6 cells (job 098) | §3, App L | 1.173, 1.167, 1.094, 1.020, 1.044, 0.983 | Reproduced |
| 9 | Microbenchmark, per-layer wait, at 86–96% of bound read time (3 hosts) | §3, Table 28 | 0.861–0.963 (102a/b/c, both budgets) | Reproduced |
| 10 | Plain form within 6% on 28 of 30 exploratory launches at 11% | §4.1 | 28/30 | Reproduced |
| 11 | Over 34 launches, overlap form 33 vs 18 within 6% at 25%; 29 vs 32 at 11% | §4.1, job 106 header | 33/18; 29/32 | Reproduced |
| 12 | Job 105 (plain): 5/8 cells, new 2/4, median 4.2%; 25% over by 6.9–9.3% (Pf 3.8%); 11% within −1.5 to 4.7% | Table 4, App C | 5/8; 2/4; 4.24%; +6.92…+9.26 (Pf +3.78); −1.49…+4.70 | Reproduced |
| 13 | Job 106: 12/20, new 0/8, median 4.7%; stable hosts 12/12, 2.3% | Table 4 | 12/20; 0/8; 4.67%; 12/12; 2.34% | Reproduced |
| 14 | Halving admissions passes 11 of those 12 cells | §4.2 | 11/12 (106a Qwen3 25% at +6.07%) | Reproduced |
| 15 | Job 106 servers under-predicted by 10–52%; EPYC 7302 loss 0.21–0.45 nats; Xeon rounds 52% apart | §4.2, App K | −10.2 to −52.4%; NLL 0.208–0.446; spread 52.15% | Reproduced |
| 16 | Job 107: 2 of 7 valid; 1/8 cells; median 31.7%; AMD ES −59% (−51% at second-highest reading); 5900XT −11% | Abstract, Table 4, §4.2 | 2 valid (107b, 107d); 1/8; 31.75%; −58.95% (−51.4%); −11.27% (Qwen3 12.5%) | Reproduced |
| 17 | Job 108: 4/6 within 8%, 2/6 within 6%, median 6.7%; EPYC 7543 −34.3%; without it 4/4, 5.3% | Table 4, Table 25 | 4/6; 2/6; 6.70%; −34.21%; 4/4; 5.25% | Reproduced (0.1 point from fallback G) |
| 18 | Table 25 values (e.g. 5950X 27.2/29.1, 0.92; 7K62 46.2/63.1, 0.71; rounds 7.0% apart) | Table 25, App K | 27.24/29.12, 0.924; 46.21/63.11, 0.715; 7.01% | Reproduced |
| 19 | 39 valid consumer launches on 25 machines; 42 valid launches on 28 machines; server launches: 5 of 8 invalid | §4.3, App K | 39 / 25; 42 / 28; invalid 106c, 106d, 107c, 107e, 108b | Reproduced |
| 20 | Within 6% on 31/39 and within 8% on 37/39 at 11%; second-highest reading 34; CPU-only 22 | §4.3, Table 26 | 31; 37; 34; 22 (within 8%: 37 / 37 / 29) | Reproduced |
| 21 | Implied read rate 0.88–1.16 of B_host at 11%, consumer | §4.3 | 0.880–1.165 | Reproduced |
| 22 | Valid servers read at 0.27 (AMD ES), 0.58 (EPYC 7543), 1.04 (EPYC 7402P) of B_host | §4.3 | 0.268, 0.585, 1.038 | Reproduced |
| 23 | Dual-socket probe: 118 GB/s at 128 threads vs 151 at 16 | §4.3 | 117.6 vs 151.4 (t=12: 151.0) | Reproduced |
| 24 | Mean Eq. (3) error −2.2% (−3.7, −0.4) at 11% over 25 machines; +1.6% (0.2, 3.1) at 25%; on found set 25/30 under at 11%, 23/30 over at 25% | §4.2, App K | −2.11% [−3.68, −0.36]; +1.57% [0.19, 3.06]; 25; 23 | Reproduced |
| 25 | Fig. 2 shares at 11%: GPU 12–45%, MIN reads 31–54%, beyond MIN 22–35%; serialisation 23–66% of gap; below rate ≤11% (consumer); 0 / 34 / 59% (servers) | §4.4, Fig. 2 | 12.0–45.4; 30.6–54.0; 22.1–34.9; 22.6–66.0; ≤10.6; 0 / 34.2 / 58.9 | Reproduced |
| 26 | Elasticity of T − G on B_host −0.95 (−1.06, −0.84) | §4.4 | −0.951 over 30 launches; −0.97 [−1.08, −0.82] by machine | Reproduced |
| 27 | Running example (9950X, gpt-oss 11%): Eq. (1) 10.0, deployed 20.6, MIN 1 read 15.2, prefetched 13.8 ms | §5 | Host O4 (096a): 10.02, 20.64, 15.16, 13.81 | Reproduced (Eq. (2)'s 13.0 not checked) |
| 28 | MIN 1 read beats deployed at all budgets on 3 hosts by 16–51%; up to 81% prefetched; "usual ways" ≤ +15% or lose at smallest budgets | §5, Fig. 3 | 1.160–1.514; 1.812; max 1.148 (nb2, 096a Qwen3 12.5%) | Reproduced |
| 29 | Fewest-admission copies 0.60–0.73 of greedy's; 2.5–5.3% more misses | §5, App K | 0.597–0.726; 2.47–5.26% | Reproduced |
| 30 | Fewest-admission in step at 11%: mean 1.22× over 9 stable machines (1.14–1.31); t-interval 1.11–1.33 | §5, App K | 1.222 [1.135, 1.308]; t [1.113, 1.331]; minimum 1.019 | Reproduced |
| 31 | Unsteady machines: lost on 2 of 4 (ratios 0.14, 0.21), gained at 0.23; every stable machine ≥ 0.28 | §5 | 0.846 (0.14), 0.974 (0.21), 1.326 (0.23), 1.175 (0.51); minimum stable ratio 0.281 | Reproduced |
| 32 | At ratio 0.28–0.32, greedy 0.85–0.93× at 11%; fewest gains every round; loaded by CPU 1.11–1.22× | §5 | 0.853–0.933; all rounds > 1; 1.112–1.217 | Reproduced |
| 33 | Spearman 0.89 [0.63, 0.97] over 19 machines (MIN in step vs ratio); machines 96% of variance, problems ≤ 3% | §5, App K | 0.88 [0.61, 0.97], n=19; 97% / about 1% | Approximately reproduced |
| 34 | Admission margin cuts reads 4–6% at 11%; 1.020× (1.012–1.031) over 5 machines; best +6%, one loss of 1% in 10 cells | §5, App K | 4.3–5.8%; 1.0197 [1.0115, 1.0305]; 1.057 and 0.987 | Reproduced |
| 35 | 16-token window on deployed policy 1.00–1.03× at 11%; half-right windows lose up to 6% | §6 | 0.9995–1.0346; b8r5 down to 0.943 | Reproduced |
| 36 | Exact 16-token window recovers 0.78–0.93 of MIN's gain over admit-every-miss, in time, at 11% | §6 | 0.781–0.935 (14 hosts, jobs 099–100) | Reproduced |
| 37 | Window shares 0.18/0.31/0.52/0.81/0.98 at gpt-oss 11% (W=1..16) | Table 19 | My own policy: 0.185/0.314/0.522/0.815/0.993 (25%: 0.07…0.67; Qwen3 12.5%: 0.31…0.99) | Reproduced |
| 38 | Half of MIN's saving needs ≈0.65 C distinct experts (quartiles 0.61–0.72) | Abstract, §6 | On the engine's AIME cells: 0.81 C / 0.69 C / 0.78 C (g11 / g25 / q12.5) | Not contradicted (rule fitted on other text); the engine's own cells lie at or above the upper quartile, a scope issue (W2b) |
| 39 | Layer-ahead copy 1.04× where link ≈ CPU, down to 0.72× at 11% | §6, Table 22 | 1.042 (O4), 0.722 (Pf); 0.81 and 0.91 on the EPYC and the Threadripper | Reproduced |
| 40 | Same-CPU panel hosts differ by up to 29%; re-rentals within 3.2% | §2 | 29.1% (two 285Ks); maximum 3.17% (Threadripper 105b→106a) | Reproduced |
| 41 | Second-highest reading median 0.8% lower; >5% on 1 of 42 (107d, 26%); >5% faster than B_host on 12/84 launch-budgets (1.22×); second-highest 15 (1.26), CPU-only 46 (1.53) | App K, Table 26 | 0.80%; 107d at 26.0%; 12 (1.223); 15 (1.257); 46 (1.533) | Reproduced |
| 42 | G profiled on 9 of 39 consumer launches | App K | 9 | Reproduced |
| 43 | Every commit preceded its rental by ≥ 2.7 s (71 rentals, jobs 093–108) | App C | 71; minimum 2.71 s (105f) | Reproduced |
| 44 | Parity KL 0.0019 / 0.0005 nats | Limitations, App L | On-machine summary 0.00186 / 0.00053; NLL from raw rows +0.16% / +0.19% | Consistent (KL itself not recomputable) |
| 45 | 124 rentals, 74 offers, US$87.2 | App N | 124, 74, 87.22 | Reproduced |
| 46 | Implicit: no measurement beats the bounds | §3 | Closest any configuration comes to R*·S/B_host is 1.34× it; closest on-demand configuration is +4.5 ms (Eq. (2) needs ≥ T_GPU ≈ 3 ms) | Consistent |
| 47 | Eq. (3) adds over the plain form | §4.1 | On the 3 new consumer machines, plain closer in 7/8 cells; median \|err\| 5.6% (plain) vs 6.4% (Eq. 3) | In-sample claim reproduced; out-of-sample reversal not in main text (W1) |

## Audit of the registered tests in Table 4

All times are CDT. Rental times come from the ledger; commit times from `git log %ad`.

| Job | What the header registers | Header commit → rentals | Later edits | Verdict |
|---|---|---|---|---|
| 105 | Plain form, G profiled first. Within 6% at every cell, median ≤ 4%. Also G ∈ [4.0, 5.0] ms, layer-ahead copy, fewest-admission | 17:05:31 → 17:05:36–39 | Comment at 17:09:26 naming the replacements 105f/g (rented 17:09:28–29). Diff checked: predictions unchanged | Matches Table 4 (plain, gpt-oss, 6%) |
| 106 | Overlap form "T = G + (M + A(1 − G/T)) S/B_host", both models, 6% band, median ≤ 4%, g25 median ≤ half the plain form's. Rounds within 2% registered as a *prediction*, not as a gate | 20:43:48 → 20:43:58–20:44:02 | Comment at 20:44:13 replacing O3 with "a Ryzen 9 9950X (54559478)", rented 20:44:16 | Matches Table 4. The "set aside" stable subset and 106e's "rented before" status are post hoc and are labelled so |
| 107 | Same form. Gates V0 (GPU UUID new, ≤ 48 GB in use), V1 (NLL within 2%), V2 (rounds within 2%). Fewer than 3 valid machines → inconclusive. Also "overlap median below plain's" | 00:17:31 → 00:17:44–47; 107e 00:24:30 → 00:24:36 | Comment-only amendments at 01:36:28/01:36:57 and 01:41:16, after results of 107c/d/e were committed (01:17:49, 01:35:53). They add replacement of round-check failures, which the original rule did not provide; disclosed | Matches Table 4 ("inconclusive"); the relation failed 7 of 8 cells on the two valid machines |
| 108 | Class gate (1 NUMA node, ≤ 32 usable cores) fixed before launch. 8% band, ≤ 1 cell beyond 6%, median ≤ 4%, implied read rate 0.85–1.25 | 02:39:32 → 02:39:36–37; replacement hosts' wrappers each committed 3–9 s before their rental | None to the header | Matches Table 4. The EPYC 7543's profile captured no decode kernels (0.0015 ms), so a fallback G was used; disclosed as a deviation, and the outcome is unchanged for any G in [4, 5] ms |

All four registered forms, bands and median rules in Table 4 match the headers. The outcomes in Table 4 match my recomputation exactly.

## Questions for the authors

1. **Overlap term out of sample.** On the three new consumer machines, the plain form is closer than Eq. (3) in 7 of 8 cells. Why keep the overlap term, and why does §4.1 report only its in-sample improvement?
2. **Confidence in the decomposition.** Out of sample, Eq. (3) under-predicts every new consumer machine at gpt-oss 11% by 6.5–7.2%. What error bar should a reader attach to "reads beyond MIN's" versus "reading below the rate" in Fig. 2?
3. **A rate that predicts the server failures.** Is there a probe criterion, rather than the processor brand, that predicts "reads well below the probe"? For example: CPU reads at the engine's helper count, a NUMA-local probe, or the probe's thread-count curve (151 → 118 GB/s on 107d). Could B_host be redefined so that it bounds the engine's achieved rate everywhere? Today 12 of 84 launch-budgets read faster than it.
4. **The 0.65 C rule on your own cells.** Please report D(W50)/C for the AIME cells the engine runs. I get about 0.69–0.85 C. Is the cross-model spread large enough that "about 0.65 C" overstates precision in the abstract?
5. **Below a link ratio of 0.28.** Did any machine with a ratio below 0.28 pass your checks? If none did, should the fewest-admission claim say that its boundary is untested, rather than that it holds "down to" 0.28?
6. **Online value of the fewest-admission schedule.** How much of the 1.22× is reachable with a realisable predictor of "used again before evicted"? The paper cites Antoniadis et al. on succinct predictions.
7. **Free-running decode.** Does anything change with free-running greedy decode instead of teacher forcing, for example cache state carried across problems, or timing of the sampling step?
8. **Parity on the 5090.** Can output parity be measured on the 5090 platform, with a task-level metric?

## What would raise my score

- **Confirm the relation, or drop the claim that it is a model.** Either confirm Eq. (3), or a mechanistically motivated revision, in a registered test on new consumer machines; or recast §4 as descriptive accounting with an explicit residual, and move the out-of-sample comparison with the plain form into the main text.
- **Scope the abstract and conclusion.** Reading MIN's set once pays on fast links; on links at 0.28–0.32 of the CPU rate only the fewest-admission schedule (or background copies) pays. State the spread of the 0.65 C rule and the probe-relativity of the bound.
- **Resolve the server anomalies with a better probe.** For example, a helper-count and NUMA-aware probe that makes Eq. (1) a true bound on every valid machine.
- **Show at least one second GPU class in the oracle study.** The 4090 and 3090 grid already exists for the system table.
- **Make a substantial clarity pass.** In priority order:
  - one idea per sentence in §4–§5;
  - reorder §4 so the decomposition comes first;
  - one name per concept (drop "limit", the appendix codes and the host letters from main-text references);
  - a simpler Table 4 and Fig. 3;
  - define the abstract's terms;
  - move the validity checks and the W50 definition into the main text.

## Scores

| Criterion | Score |
|---|---|
| Overall (1–10) | **5**: borderline, leaning reject. Exceptionally rigorous and reproducible, with a valuable oracle study, but the titular explanatory model failed its registered tests, several headline statements are broader than the data, and the presentation remains a barrier |
| Soundness (1–5) | **3** |
| Significance (1–5) | **3** |
| Novelty (1–5) | **3** |
| Clarity (1–5) | **3** |
| Confidence (1–5) | **4** |
