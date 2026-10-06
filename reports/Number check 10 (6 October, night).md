# Number check 10 (6 October, night)

Scope: the changes in commit ca4c6a9 against bb56076 in `paper/paper.tex` and `paper/app_wsg.tex`, plus
`prereg/minadm_outcome_104.md`, and the files that feed them (`wsg_job103.tex`, `wsg_job104.tex`, `tab_job104.tex`,
`tab_scorecard_103/104.tex`, `wsg_linkaware.tex`, `wsg_hostdep.tex`, the new `fsEx*Ms` in `wsg_shapley.tex`,
`fxExAdmitsOverOnline`, `prereg/scorecard_103.json`, `prereg/scorecard_104.json`, `prereg/linkaware.json`). I
recomputed every number with my own Python from these sources:
- the raw job 103/104 files in `results/10[34]?_minadm@vast`: per-problem `decode_ms/n_decode`, paired by problem, ratio
  of means against `base` of the same cell, 10,000-resample bootstrap; counters per step from `st_g_C*_<config>.json`;
  `plan_g*.txt`; probes through `fetch_table.bandwidths` and `speed_limit.host_rates`;
- the earlier Pf, Pg and O4 launches (`099f`, `101a`, `099g`, `096a`) for launch counts, relaunches and the running
  example;
- the 2x2 shares with `factorial_shapley.limit1_of`, the host-dependence points with `fig_hostdep.points()`, and the
  link-aware host term with `linkaware.t_link`, also on the job 103/104 probes;
- GPU UUIDs (`nvidia-smi-q.txt`) and offer ids (`gpu/vast_ledger.json`) to identify machines.

I built the paper and the supplement with latexmk in a scratch copy of `paper/`. Line numbers refer to ca4c6a9.

## Defects (most severe first)

**1. Medium-high: the "new" 5950X of jobs 103/104 is panel host Pg relaunched. The text counts it as a further machine
and the figure plots Pg twice.**
- Evidence: `099g_panel`, `103d_minadm` and `104c_minadm` share GPU UUID `b5fd1878-e3eb-…` (bus 07:00.0) and vast
  offer 52273675.
- The data agree. Link-to-CPU ratio is 0.635 / 0.620 / 0.629. Greedy fetch/base at 11% is 1.245 / 1.246 / 1.256; at 25%
  it is 1.320 / 1.319 / 1.325.
- Where the text gets it wrong:
  - `paper.tex:255-256`: "on \jeNewN{} [3] further machines with ratios 0.40–0.62 it still gains 1.02–1.25×". Only two
    machines are new: the 285K with a 40 GB/s link (0.475, 1.052) and the 3970X (0.397, 1.019).
  - `fig_hostdep.py` drops relaunches (100a, 100c, 101a, 103a) but keeps 103d. So Pg's greedy states appear twice, and
    the caption (`paper.tex:275`) calls 103d one of "the new machines". Without the duplicate, the 11% correlation is
    0.94 over 19 points (the paper says 0.93 over 20).
  - `prereg/minadm_outcome_104.md:13-14`: "three new machines … a Ryzen 9 5950X (0.62: 1.25)".
  - `job103.py` NAMES and the `scorecard_auto.py` captions: "5950X", "the 5950X of 103d".
  - `paper.tex:411`: "(\jcRelaunchN{} [3] relaunched)". Pd, Ph, Pf and Pg have now all been relaunched, which makes 4.
- Fix: "on two further machines (ratios 0.40 and 0.48) it still gains 1.02 and 1.05× at 11%". Add `103d` to the
  relaunch exclusion in `fig_hostdep.py` (104c's star stays). Rename the 5950X "Pg again" in `job103.py`,
  `scorecard_auto.py`, `tab_job104` and the outcome note. Make the relaunch count 4.

**2. Medium: "every machine we ran" overclaims. The fewest-admission set ran on 3 of the paper's 20 machines.**
- `paper.tex:249` says "MIN's set spent once pays on every machine we ran". Spent once, MIN's greedy set loses on Pf on
  four launches (0.86–0.88). The fewest-admission set ran only on Pf, the 285K and Pg.
- `paper.tex:435-436` says "the fewest-admission one gains on every machine we ran, including the slow-link machine
  where the greedy one loses". "Every machine we ran" reads as the whole sample. "The slow-link machine" also clashes
  with the paper's 0.5 line: the 285K it ran on (0.49) is slow-link too, and so are Pd (0.40) and the 3970X (0.40).
- What the data support: on its three machines the fewest set gains over the deployed cache with both loads and at both
  budgets. In the step it runs 1.04–1.35×; by the CPU, 1.06–1.22×. All 12 intervals exclude 1 (lowest: by the CPU on Pg
  at 11%, [1.053, 1.064]).
- Fix: "on the three machines it ran on, the fewest-admission set gains whichever way it is loaded" and "including Pf
  (ratio 0.29), where the greedy one loses".

**3. Medium: the appendix says the speed predictions held. One failed and is missing from the list.**
- `app_wsg.tex:108-111` says "The speed predictions held" and lists the failures as counters, replay, two-read reads and
  interaction.
- Prediction 6 (bypassplan/base ≥ bypass/base − 0.01) failed on Pg at 25%: −0.047 (1.151 against 1.198). Clause
  `104c-P6-g25` in `scorecard_104.json`, and the outcome note's last failure.
- The 24 failures are: 3 copies, 3 misses, 9 replay, 5 two-read reads, 3 interaction and 1 two-read speed.
- Fix: "The in-step speed predictions held (…); the failures are …, and the two-read plan state on Pg at 25%, 0.047
  below the greedy one".

**4. Medium-low: Pf's greedy copy lost on four launches, not three.**
- `paper.tex:95` says "(\jfPfFetch×, on three launches)", and `paper.tex:247` "loses on three launches".
- Launches 099f, 101a, 103a and 104a are the same machine: UUID `b8316d93-…`, offer 40038866, and separate rental ids.
- fetch/base by launch:

  | Budget | 099f | 101a | 103a | 104a |
  |---|---|---|---|---|
  | 11% | 0.875 | 0.860 | 0.859 | 0.863 |
  | 25% | 0.969 | 0.966 | 0.971 | 0.964 |

- Fix: "on four launches (0.86–0.88×)".

**5. Medium-low: the intro says "the bound is nearly reachable". The data show only its read time is.**
- `paper.tex:84` heads finding (1) "the bound is nearly reachable". `paper.tex:111` says the link-aware variant shows
  "its read time is nearly reachable".
- What was measured:
  - Without compute, MIN's reads reach 86–96% of the host term.
  - The best in-engine oracle reaches 46–73% of the bound (`\fxBestFrac`).
  - The limitations admit that overlap with the GPU's work is unmeasured (`paper.tex:419-420`).
  - The link-aware variant shows only that the bound rises by at most 7%; it says nothing about reachability.
- Fix: "…and its read time is nearly reachable" (as in `paper.tex:192` and the conclusion). In `:111`, use "a link-aware
  variant that moves it by at most 7%".

**6. Medium-low: the accounting leaves out job 103's 2×2 on two new slow-link machines. "Three new machines" now
misleads.**
- `paper.tex:284` says "all four combinations on O4, O5, the panel and three new machines". Those three are N100b,
  N100f and N101b. Section 4 has just introduced job 103's "further machines", which also ran base/foa/bypass/fetch.
- `paper.tex:290-293` says "on the \fsSlowHosts{} [2] below that line, loading in the step costs time at 3 of 4
  host-budgets". That leaves out the 285K (0.475) and the 3970X (0.397). Their φ_reads is:

  | Machine | 11% | 25% |
  |---|---|---|
  | 285K | +1.0% | −8.8% |
  | 3970X | −1.3% | −11.6% |

  With them included the count is 6 of 8. The fast-link ranges do not change: Pg's relaunch has either change alone
  ≤ 24% and both 35–39%.
- Fix: add 103b and 103e to `factorial_shapley.hosts()` (and to `\fsAccHosts` in the contributions), or scope both
  sentences to "jobs 096–101".

**7. Low: the appendix points to a cap Section 3 no longer gives.**
- `app_wsg.tex:497-498` says "\cref{sec:limit} gives the cap that the link puts on MIN's one-read schedule
  (`prereg/readsched.json` per host)".
- The rewrite removed the read-once cap. Section 3 now gives the link-aware bound with the fewest admissions
  (`prereg/linkaware.json`).
- Fix: "\cref{sec:limit} bounds the cost of sending every admission over the link (`prereg/linkaware.json`)".

**8. Low: `tab_job104` caption and labels.**
- Line 3 says "(job 103)", but every row is job 104. `job103.py` hard-codes "job 103" in the caption; use `{jobid}`.
- Lines 10-11 call the 285K "faster link" at ratio 0.49, which is below the paper's slow-link line. Use "285K, 40 GB/s
  link".
- Lines 12-13 call Pg "5950X" (defect 1).

**9. Low: "Made ahead of use, the same copies lose nowhere" (`paper.tex:257`) is stronger than the data.**
- On Pf's 101a launch, paced/base is 0.997 [0.987, 1.007]; on 099f it is 1.008 [0.974, 1.057].
- The old text said "break even". `\jbPfPaced` (0.997) is now unused anywhere.
- Paced did not run on job 103's two new slow-link machines.
- Fix: "lose beyond their interval nowhere (at worst \jbPfPaced× on Pf)".

**10. Low: mixed units in one sentence (`paper.tex:303-305`).**
- The sentence puts "tightens it by 1–9% of the gap" next to "raises it by at most 7%". The 7% is of the bound's host
  term (`link_over_bound`), not of the gap.
- As a share of the gap it is at most about 2%: Pf at 25%, 0.069 × 2.16 ms against a gap of about 7.3 ms. At 11% it is
  under 1%.
- Fix: "raises it by at most 7% (about 2% of the gap)".

**11. Low: Section 3's admission counts and `tab_job104`'s look inconsistent.**
- `paper.tex:197-198` says the fewest admissions any hit-optimal schedule needs is 12.4 per token, against 21.8 greedy.
  These come from the 30-problem trace 084c (7,680 tokens).
- The engine in `tab_job104` makes 11.8 against 19.8 in-step copies. That is fewer than "the fewest any schedule needs".
  The causes:
  - the engine ran 20 problems, and the per-host LP gives 12.80, the replay 12.18 against 20.16;
  - the engine misses 2.5–5.3% more;
  - about 2% of the forced copies are not made (forced 12.10 and 20.25 against made 11.84 and 19.81).
- Fix: name the trace in Section 3, and in the table caption say the copies are the engine's, on the 20 problems.

**12. Low: "on \laHosts{} [24] probed hosts" (`paper.tex:199-200`) counts probes, not machines.**
- The 24 probes cover 17 machines: Pd ×3, O4 ×3, Pf ×2, Ph ×2, 9800X3D ×2.
- The 1% / 7% maxima also hold on the job 103/104 probes, at 1.0095 and 1.0733 on Pf.
- Fix: "on 24 probes of 17 machines".

**13. Low: `tab_configs` has no row for the fewest-admission configurations.**
- Section 4 and `tab_job104` compare the fewest-admission configurations (by the CPU, and in the step) with greedy MIN.
- `tab_configs` does not list them, and its "MIN, 1/2 reads" rows do not say greedy.
- Fix: add one row, and "(greedy)" to the MIN rows.

## Checked and held

- **`tab_job104`, all 24 numbers.** The 12 speed ratios and the copies (19.8/11.8 and 9.7/7.0, identical across hosts
  because the counters are deterministic) match my recomputation to 2 dp. Examples:
  - Pf: 0.941, 1.113, 0.8635, 1.0351 (11%); 1.133, 1.217, 0.964, 1.0595 (25%);
  - 285K: bypassplan 1.0845 at 11%, fetchplan 1.1749 at 25%;
  - Pg: fetch 1.3245 at 25%.
- **Job 104 macros.**
  - jfCopyRatio 0.60–0.73 (0.597, 0.726); jfMissDev 2.5–5.3 (2.52, 5.26).
  - jfGainLow 0.08–0.17 (0.082, 0.127, 0.172); jfGainMid 0.03–0.10.
  - jfPfFetch 0.86; jfPfPlan 1.04, interval [1.023, 1.053]; jfPfRatio 0.29; jfRatio 0.29–0.63.
  - jfBypassPlan 1.06–1.11 and 1.15–1.22.
- **2×2 shares (limit1_of).**
  - Both: plan 6–49 (5.6, 26.6, 48.8) against greedy −26–39 at 11%; 25%: 7–39 against −5–36.
  - Interaction: plan 1–33 (Pf 0.6) against greedy −4–30. It grew on all three machines at both budgets.
  - Set alone with the plan: 11–17 and 19–23.
- **Scorecards.**
  - 103: 45 clauses = 4 held (point; P8 on Pf) + 41 untested.
  - 104: 47 clauses = 1 held + 22 held (point) + 24 failed.
  - Every value in the outcome note holds:
    - P1 0.597/0.726 and 2.5/5.3%;
    - replay 3.8/6.8% (plan) and 1.6/2.4% (greedy);
    - P2 0.896–1.007;
    - P8 base ≤ 0.3%, fetch/base ≤ 0.0034;
    - forced 20.25 against 19.81.
- **Job 103 had no plan.**
  - Every 103 stats file has `oracle_plan` 0, and 103's plan states equal greedy.
  - "Cannot uninstall numpy" is in `pip.txt` on all four hosts, and "COULD NOT BE READ" in `err.oracle`.
  - 103c never started (ledger).
- **The greedy copy loses on Pf only.** Of every cell of jobs 093–104 that has a fetch state, fetch/base < 1 only on Pf,
  at both budgets on all four launches. "Loses on one machine only" (`paper.tex:269, 412`) holds.
- **Host dependence.** hdRatioMin 0.29 (0.286); hdAa 0.42–1.00; correlation 0.93 at 11% (0.94 without the Pg
  duplicate); hdFetchPlan 1.04–1.35; jeNew values match the data (see defect 1 for their meaning).
- **Link-aware bound.**
  - The LP at μ = 1e-4 gives misses equal to R* (38.315) and 12.40 admissions. It is integral (fractional share 0).
  - Greedy A* is 21.77. laOverLowMax 1 (1.009); laOverMidMax 7 (1.072).
- **Running example (096a, gpt-oss 11%).** Bound 10.02; base 20.64; fetch 15.16; paced 13.81; gap 10.62.
  fxExAdmitsOverOnline = 15.98 / 3.46 = 4.6.
- **Model on the new hosts.** The two-path model (G calibrated on base) gets the sign of fetch/base right on every
  job 103/104 host-cell. Its optimism on the 285K's in-step copy (−19 to −20%) is about the stated 19%.
- **Relaunch reproducibility.** Measured against each machine's first launch, Pf (×3), Pg (×2) and the 285K (×1) stay
  within 1.8% in base and 0.026 in every ratio, inside the stated 1.9% and 0.030 (only the count of 3 is stale).
- **Fewest set's gains.** It gains on Pf with both loads (in the step 1.04 and 1.06; by the CPU 1.11 and 1.22). Its
  in-step gain over greedy is ≥ 0.03 on every machine at 11% (`app_wsg.tex:109`).
- **Build and page limit.**
  - latexmk builds `paper.pdf` (31 pp) and `supplement.pdf` (33 pp) in a scratch copy.
  - The rebuilt PDF's text is identical to the committed `paper.pdf`.
  - No undefined references or citations, and no "??" or "[pend.]" in the PDF; every \ref/\cref resolves.
  - The Conclusion ends and the References begin on p. 9, so the main text is within 10 pages.
  - Page 1 is anonymous; `\fxMeanKindDiffMax` (0.006) is defined.
