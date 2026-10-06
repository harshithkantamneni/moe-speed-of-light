# Number check 7 (6 October, morning)

Independent check of `paper/paper.tex` at HEAD (0823f93) and `prereg/window_outcome_100.md` against the raw results
in `/home/claude/gpu-branch/results/` (jobs 095-097, 099a-j, 100a/b/c/f). Nothing outside this report was edited.

## Method

- Resolved every macro in the order `paper.tex` inputs the `wsg_*.tex` files (`\providecommand` placeholders in
  `wsg_pending.tex` lose to generated `\newcommand`s), and read the abstract, introduction, Sections 2-5, the
  limitations, the conclusion, `tab_shapley`, `tab_window100`, `tab_limit`, `tab_headline`, `tab_lomo`, `tab_configs`,
  `app_value.tex`, and the scoring and calibrated-model passages of `app_wsg.tex` with the values substituted.
- Wrote my own loaders (scratchpad `nc7/load.py`, `hm.py`, `acct.py`), not the generating scripts. Time per token =
  decode_ms / n_decode per problem, mean over the problems common to all states of a cell; ratios are ratios of those
  means. Probe rates use the same rule as `fetch_table.bandwidths` (CPU rate interpolated at H = cores - 2 helpers,
  link = highest zero-copy 16 MB/64-block sample, combined = median of the zero-copy concurrent sums); B_host = highest
  sample of any kind. Counters from `st_g_C<C>_<state>.json` divided by `steps`.
- Recomputed: every job 100 speed ratio (table and outcome note), the probe-only errors (from `predict_probe.json`,
  re-predicted by hand), the sign test with each run's own counters, the relaunch comparisons with 099d and 099h,
  the replay deviations, the leave-one-host-out frozen-G model on 39 host-budgets of 14 hosts (G fitted on each
  host-cell's deployed state, frozen at the median of the other hosts' fits per budget), the one-path sign count, the
  window-share errors, the host-dependence ranges and correlations over 16 hosts, the order-free accounting on O4, O5,
  the panel and the four job 100 hosts, the one-path and two-path bounds, the panel spread statistics with a crossed
  bootstrap, the horizon rule's bootstrap from `w50_distinct.json`, the audit statistics from `audit_sol.json`, and
  the clause tallies of the job 099 and 100 scorecards.
- Checked the job 100 timeline: `start_utc` of each host versus the commits holding its predictions, and the
  `predict_probe.json` timestamps against the first timed run in `stdout.log`.

## Defects

### 1. The fast-link ranges leave out job 100's hosts, and one of them falls well outside (overclaim)

**Where:** abstract (paper.tex:50-51), introduction (2) (paper.tex:83-84), "An order-free accounting"
(paper.tex:277-278, 290-294), conclusion (paper.tex:496-498).

**What the paper says:** "The engine ran all four combinations on O4, O5 and the panel." On hosts whose link reads at
least half as fast as their CPU, doing both closes 26-60% of the gap. Either choice alone is worth -13 to 26%. The
Shapley value for how to load is 1-32%, and prefetching adds -4 to 23%.

**What the data say:** job 100 ran base, foa, bypass, fetch and both3p on all four of its hosts, but
`factorial_shapley.py` reads only 096a/b and 099. Three job 100 hosts are in the fast-link class. Computed the same
way (gap from the one-path bound with that host's highest probe sample):

| Host | Link/CPU | Budget | Both | Load alone | Shapley, load | Prefetch | Rest |
|---|---|---|---|---|---|---|---|
| 9950X, half-width link (100f) | 0.54 | gpt-oss 11% | **13.6%** | -0.3% | 8.2% | **27.6%** | 58.8% |
| 9950X, half-width link (100f) | 0.54 | gpt-oss 25% | **15.2%** | **-14.6%** | **-6.4%** | **26.0%** | 58.8% |

The 5700X3D (0.92) and the Ph relaunch (0.96) fall inside every stated range: both together 51-56% and 31-37%.

**Severity:** overclaim. This is the abstract's headline range, and it fails on the first new host past the half
line, a line drawn after the panel ran.

**Fix:** add 100b and 100f (and say how the relaunches are treated) to `factorial_shapley.py` and regenerate. Or
scope the ranges to "O4, O5 and the panel" and state that the 0.54 host closes 14-15%.

### 2. The model does not over-predict every in-step state on the half-width host (error)

**Where:** paper.tex:257-259.

**What the paper says:** "on a host whose link runs at half width it over-predicts every in-step state by up to 20%,
because the probe read that link more slowly than the engine's copies then ran."

**What the data say:** on 100f the probe-only time errors are:
- admit every miss: +20.1% and +17.9%;
- the 4-token window: +19.1% and +16.7%;
- the 16-token window: +10.9% and +15.9%;
- MIN with one read: **-2.0%** at 11% (predicted 16.60 ms, measured 16.94 ms) and +1.9% at 25%.

The stated cause does not fit the last state. In the model, MIN with one read is link-bound on this host (link term
12.0 ms, CPU term 6.6 ms, combined 10.7 ms), so a link faster than the probe would make the model over-predict it as
well.

**Severity:** error ("every").

**Fix:** "over-predicts the three states that copy every admission in the step by 11-20%". Then state the cause as
likely, not established.

### 3. "The CPU serves every miss" on the deployed path is false (imprecise)

**Where:** paper.tex:362-363 and `tab_window100.tex:5` ("the CPU serving each miss").

**What the data say:** the deployed policy and the deployed-path windows run with each machine's fetch table, which
copies part of each layer's misses into the step over the link:

| Host | Fetch table | Deployed state at gpt-oss 11%: in-step fetches / misses per step |
|---|---|---|
| Pd | 0,0,0,1,1 | 9.1 / 61.1 |
| 100f | 0,0,1,1,1 | 18.7 / 60.1 |
| 100b and 100c | 0,0,1,2,3 | 29.8 / 59.7 |

So 15-50% of misses cross the link in the step. The 16-token deployed-path window (b16) fetches 5.6-24.6 per step,
and MIN with two reads fetches 14.2 per step on 100f.

`tab_configs` simplifies the same way for "deployed" and "MIN, 2 reads". Section 2 states the table, but the new
paragraph says "every".

**Fix:** "the CPU serves the misses the machine's table does not fetch, and admissions are copied in the background".

### 4. "Safe", and "only MIN with its copies made ahead never loses" (overclaim / imprecise)

**Where:** paper.tex:365, 371-372; outcome note line 68.

**What the data say:**
- Only W = 16 is safe on the deployed path. Its worst case is 0.9995, CI [0.995, 1.004].
- W = 4 loses: 0.960 [0.948, 0.971] on 100f at 25%, and 0.992 [0.989, 0.995] on 100b at 11%.
- The half-accurate window loses 0.943 [0.939, 0.947] on Pd at 11%.
- On the same four hosts MIN with one read also never loses (1.048-1.484), and b16 is at parity at worst. The
  "only" contrast mixes host sets: the panel host where MIN with one read lost (Pf) never ran the deployed-path
  windows.

**Fix:** "a 16-token window on this path is safe (...); shorter or half-accurate windows lose up to 4% and 6%".
Scope "only ... never loses" to all hosts tested, or drop it.

### 5. The sign test's 16 of 16 hides the single-read ratios (imprecise, by omission)

**Where:** paper.tex:253-255.

**What the data say:** the 5% filter drops all 8 single-read (foa/base) ratios. The model puts them at 1.006-1.015.
Measured, they are 0.903-1.030, and the sign is wrong at 6 of 8:
- Pd at 25%: predicted 1.015, measured 0.903;
- 100f at 25%: predicted 1.010, measured 0.912.

I recomputed the 16 retained clauses (fetch and aa at 8 host-cells) with each run's counters and G = 4.56/4.53 ms:
all right.

**Fix:** add "it puts the single read within 2% of the deployed cache everywhere; measured, it runs 0.90-1.03×".

### 6. "The deployed state's included" (imprecise)

**Where:** paper.tex:249.

**What the data say:** `hmFrozenStepMed` (2.1%, 90th percentile 10%) is over the 200 rows of the in-step states
foa, aa, fetch, w1, w4, w16, w8r5 and allr5. It excludes the deployed state. The deployed state's own frozen-G
errors are a median 2.4% and a maximum 13% (`hmFrozenBaseMed`/`Max`, which the paper never uses). With them included
(239 rows) the median is 2.15% (2.2), and the 90th percentile stays 10%.

**Fix:** drop the phrase, or add "and the deployed state's within a median \hmFrozenBaseMed\% (at most \hmFrozenBaseMax\%)".

### 7. The two-path bound's 1-7% is not the cost of splitting (imprecise)

**Where:** paper.tex:304-307; limitations, paper.tex:476-477.

**What the data say:** with c free, `limit_two_path`'s host term is R*·S / B_cp at every host-cell: the 28 of the
accounting and the 8 of job 100. I checked lim2 == R*·S / B_cp numerically in each one. The reason is that
B_c + B_p ≥ B_cp on every host, so the per-path caps never bind.

The share (lim2 - lim) / gap is computed as stated: O4 at 11%, (10.558 - 10.016) / 10.622 = 5.1%. But all of that
tightening comes from replacing B_host (the highest sample of any probe line) with B_cp (the median of the
concurrent sums). Splitting the reads costs nothing. The paper's point ("it is not slack from letting the two paths
share") holds more strongly than the text says, but the 1-7% / 0-2% is attributed to the wrong cause.

**Fix:** "splitting MIN's reads between the paths at their probed rates does not tighten the bound at these budgets
(the combined rate binds); taking the median combined rate instead of the highest sample tightens it by 1-7% of the
gap". In the limitations: "does not tighten it".

### 8. The cause of the replay excess is asserted, not measured (imprecise)

**Where:** paper.tex:368-369; `app_wsg.tex:93` ("their copies land late"); outcome note line 57-58.

**What the data say:** the 7-33% excess is right (6.6-33.0% over 24 host-cell-windows). But the deployed state,
whose admissions also land late through background copies, matches its replay (online, κ 1) within -1.8% to +2.4%
on the same hosts. The excess also scales with the host's in-step fetch count:
- Pd: 7-18%;
- 100f: 7-26%;
- 100b and 100c: 7-33%.

The replay has no fetch table.

**Fix:** "the engine misses 7-33% more than a replay that admits at once and has no fetch table; we attribute this to
late-landing copies, untested".

### 9. The outcome note's summary is looser than its own data (imprecise)

**Where:** `prereg/window_outcome_100.md:68-70` (and 56-57).

**What the data say:**
- "losing at most 6% with half-wrong forecasts": exact 4-token forecasts also lose 4% (100f at 25%, 0.960).
- "at most what MIN with two reads gains": b16 exceeds bypass at all four 11% cells, by 0.007-0.026. The prediction
  allowed +0.03.
- "its late-landing copies cost a third of the read saving at 16 tokens": a third holds only on Pd. There the engine
  keeps 67-73% of the replay's miss saving relative to the deployed state. It keeps 53-54% on 100f and 40-48% on
  100b and 100c. In reads (misses plus admits) at 11%, b16 reads more than the deployed cache on 100b, 100c and
  100f (65.2 vs 63.1, 65.2 vs 63.1, 64.8 vs 63.8).
- Line 56-57: "Failed at 18 of 24: the engine misses 7-33% more": the 18 failures are 10.5-33%; 7-33% spans all 24.

**Fix:** restate with these numbers.

### 10. "Predicts from a host's bandwidth probe which way pays" (imprecise)

**Where:** abstract (paper.tex:52-53); paper.tex:262.

**What the data say:** only the times of the four host-independent in-step states were predicted from the probe
alone. Every sign test used each run's measured counters (39/39 and 16/16), including the deployed state's, whose
fetch counts depend on the host's fetch table (9.1-29.8 per step).

**Fix:** "from a host's bandwidth probe and the states' read counts".

### 11. "As for MIN" (imprecise, low)

**Where:** paper.tex:369-371.

**What the data say:** on job 100's two low-ratio hosts, MIN with one read beats MIN with two reads at 3 of 4 cells:
- Pd at 11%: 1.048 vs 1.000;
- 100f: 1.067 vs 0.986 and 1.111 vs 1.096.

Only Pd at 25% reverses (1.111 vs 1.192). The deployed-path window beats the in-step window at all 4. For the
windows the crossover lies above 0.54; for MIN it lies near 0.3-0.4.

**Fix:** drop "as for MIN", or say the crossover is higher for the windows.

### 12. The prefetch's reads range includes the GPU-bound budgets (imprecise, low)

**Where:** paper.tex:299-300.

**What the paper says:** the prefetch "reads 1.18-1.58× MIN's bytes".

**What the data say:** `fsbpReadsOpt*` comes from O3 at all six budgets. 1.54-1.58 are the GPU-bound budgets (40% and
43.75%). At the host-bound budgets that the accounting covers, it is 1.18-1.44 on O3-O5.

**Fix:** use the host-bound range.

### 13. Arithmetic in the worked example (cosmetic)

**Where:** paper.tex:278-281.

**What the data say:**
- 4% + 4% + 43 points = 51, while the text says both close 52% (unrounded 4.48 + 3.88 + 43.19 = 51.55).
- "MIN caches 2-11 times as many experts per token" is the range over all O4/O5 host-bound cells. In the worked
  example (O4 at 11%) the factor is 5.7 (19.9 forced fetches against 3.46 deployed admissions per token).

**Fix:** one decimal, or "about 43 points"; "about 6 times here (2-11 across cells)".

### 14. Section 2: unlike quantities compared, and job 100's clauses missing (cosmetic)

**Where:** paper.tex:147-149, 153-154.

**What the data say:** the same-CPU spread (16-29%) is in deployed time per token. The relaunch figure set against it
(0.030) is a ratio. The like-for-like figure is the relaunch's deployed time within 1.9% (`\jaRelaunchBaseMax`).

"scores the 565 clauses of jobs 073-098 by hand and job 099's 273 by machine" omits job 100's 216 machine-scored
clauses (`app_wsg.tex:91` has them).

### 15. Job 100's frozen G is the lower middle value, not the median (cosmetic, script and job header)

**Where:** `jobs/ec2/predict_100.py`; job header.

**What the data say:** G = 4.56 / 4.53 ms are described as "the median of the 14 earlier hosts' fits". They are the
7th of 14 sorted fits. The medians are 4.635 / 4.553 ms. The pool also contains the relaunched machines' own job 099
fits. The paper's numbers are unaffected: with the true medians, the median error is 3.76% (3.8), the maximum 20.3%,
and the fetch minimum -18.2%.

**Fix:** say "the 7th of 14 sorted fits", or correct the wording.

### 16. Hard-coded counts, undefined host names, and macro choice (cosmetic)

- paper.tex:465-466: "(two relaunched, with two new hosts, in job 100)" and "(Qwen3 at one budget on three)"
  (`\pnHostsQlow`).
- paper.tex:487: "four hosts" (`\jaHosts`).
- paper.tex:383: "33,000-152,000 tokens".
- `tab_window100` rows "Pd again" and "Ph again": Pd and Ph are never defined in the paper or the appendix. Use
  "i9-13900KF (panel, relaunched)" and "7950X (panel, relaunched)".
- paper.tex:370-371 uses `\jaPathLowN` and `\jaPathHighN` where the claim is a count of wins. Use
  `\jaPathLowWins` and `\jaPathHighWins` (equal today, 2 and 2).
- paper.tex:256-258: the error sentences do not name time, against the paper's stated convention.

### 17. `app_value.tex:9-10` (cosmetic)

The κ = 1 parenthetical describes the deployed-path window. The next sentence then says "Admissions are fetched in
the step; every miss is read once", which is false for that variant.

### 18. Smaller overstatements (low)

- Abstract (paper.tex:52): "Where the link is much slower, the copies must be made ahead of use." On the panel's
  other slow-link host (Pd, 0.40) and its relaunch, MIN with one read in the step still gains 1.05-1.12. Only Pf
  (0.29) loses.
- Introduction (3) (paper.tex:92-93): "recovers 0.32-0.47 of the oracle's gain". This is the share of MIN's gain over
  admitting every miss. Against the deployed cache, the 4-token window is slower on 6 of 10 hosts.

## Verified correct

- **Abstract and introduction numbers:**
  - 11 of 12 configurations; 25-43% and 31-56% of the bound;
  - audit: 52 rows, 13 sources, 20 in class, median 13.6%, quartiles 8.1-20.6%, ours 18-42% (median 27%);
  - 2.0-4.0× llama.cpp;
  - 0.29 and 0.87× (Pf, 099f, 285K: 0.8749);
  - the window shares at 4 tokens (0.32-0.47), 8 tokens at half accuracy (0.20-0.26), whole future at half accuracy
    (0.42-0.55), and 1/16 tokens at 11% and 4/16 tokens at 25%;
  - 1.12×; 0.65 C; 2-10 tokens.
- **Section 2:**
  - same-CPU spread 15.5-29.1% (285K pair: 29.1% and 15.5%; 9950X pair: 22.0% and 16.6%);
  - relaunch ratios within 0.0296 over all common states, bypass included (max: both3p on Ph at 11%, +0.030, and on
    Pd at 25%, -0.029);
  - relaunch deployed time within 1.88%;
  - 20 problems; 0.024 for the O4 relaunch (097a vs 096a).
- **Section 3:**
  - GPU at 51.6% and 60.6% of datasheet;
  - the GPU term takes over at gpt-oss 25% on both hosts and at Qwen3 25% only on B (214 → 192; S unchanged at 174);
  - 33-57% with all tightenings on B; pooled slots cut reads by 4.46-18.57%;
  - FreeToken against the pooled bound 18-40%.
- **Section 4, the model:**
  - frozen-G (median of the other hosts' fits per budget, from `hostdep_model.py`, re-implemented): step-state median
    2.11%, 90th percentile 10.3%, N = 200, 39 host-budgets, 14 hosts;
  - fetch sign 39/39; one-path 37/39, missing exactly Pf's two losses;
  - background states: 64 of 64 predicted faster, median 16.1%;
  - G fits 4.08-9.22 ms;
  - window shares: error median 0.03, maximum 0.10 over 60; whole future at half accuracy over-predicted by up to
    0.244, on Pj (0.89, fast link).
- **Section 4, job 100:**
  - probe-only: 32 predictions, median 3.83%, maximum 20.07% (100f, aa, 11%), fetch -18.6% on Pd at 11%;
  - counters within 0.13% of the panel means;
  - predictions written 10:30-12:10Z, after the commits (4fda930 at 10:24:57Z, 752e1c3 at 10:25:59Z, 718bf41 at
    12:06:35Z), at or after each host's start (10:26:44-12:07:04Z), and before every timed run;
  - sign test 16/16.
- **Host dependence:** 16 hosts, ratio 0.29-1.05; fetch 0.87-1.48; prefetched 1.01-1.81 (Pf 1.008); admit every miss
  0.42-1.00; correlations 0.93, 0.87, and 0.80, 0.70 without Pd and Pf.
- **Accounting (O4, O5, panel):**
  - every number of `tab_shapley` and of the text's fast/slow ranges as defined;
  - worked-example components 10.6 ms, 4, 4, 52, 43, 26, 25, 13, 36;
  - slow-link: Shapley value negative at 3 of 4; rest 68-99%;
  - slack shares as computed (Pi at 25%, 0.51, rounds to 1).
- **Panel spread:** half-width 0.0085, SD 0.154, crossed-bootstrap interval [1.12, 1.30].
- **Section 5:**
  - window losses against the deployed cache: 6/10, 8/10, 2/10, 3/10;
  - plan time at most 277 µs;
  - `tab_window100`: all 80 cells; link/CPU 0.41-0.96;
  - deployed-path windows: b16 1.00-1.03 and 1.06-1.12; b16 minus bypass at most 0.026; replay excess 6.6-33.0%;
  - path wins at 16 tokens at both budgets;
  - horizon rule: 9 of 9 models fall from first to last budget; bootstrap ratio 0.95-1.09; LOMO medians 1.22, 1.19,
    1.16;
  - learned order: 0.90-1.05× deployed, 6-13% fewer reads than single read.
- **Outcome note:**
  - tally 41 / 122 / 27 / 26 = 216, matching `scorecard_100.json` clause by clause in counts per prediction;
  - every cell of its table;
  - 7.3%; 27 GB/s implied against 21.8 (zero-copy) and 25.3 (copy engine);
  - relaunch extremes; plan 10-106 µs; 1.07 vs Pe 1.35 and Pj 1.27; spend 5.20.
- **Appendix:** job 099 tally 273 = 190 / 42 / 39 / 2 (the untested pair is Pd's P1; 18 of 39 failures on Pd and Pf);
  the job 100 paragraph's tallies and failure grouping; the `app:law` passages the main text cites (helpers
  1.09-1.31×; uneven sharing under in-step copies; Qwen3 speed +5 to +21% with constants frozen).
