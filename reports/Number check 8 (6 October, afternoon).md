# Number check 8 (6 October, afternoon)

Scope: the changes in commit a4735b3 against 0823f93 in `paper/paper.tex`, `paper/app_wsg.tex`, `paper/app_value.tex`,
`paper/tab_configs.tex` (no change), `prereg/window_outcome_100.md` and `prereg/crossover_outcome_101.md`. I also checked the
macro files and generated tables that changed with them (`wsg_job101`, `wsg_job100`, `wsg_shapley`, `tab_shapley`,
`wsg_hostdep`, `wsg_hostdepmodel`, `wsg_factorial`, `wsg_numbers2` cost). Every recomputation below starts from the raw
`ec_*_C*.jsonl` (decode_ms / n_decode per problem, means over problems, paired ratios) and the `st_*.json` counters in
`/home/claude/gpu-branch/results`. I used my own short Python and did not run the repository's scripts. The exceptions are
the probe parsers (`fetch_table.bandwidths`, `speed_limit.host_rates`, `limit`, `limit_two_path`) and the stored bound
values. Line numbers refer to HEAD. The compiled PDF matches the current macros, with no `??` and no LaTeX warnings.

## Defects (most severe first)

**1. High: an "only" claim in the abstract and the introduction that the data contradict.** `paper/paper.tex:48`
(abstract): "Knowing the future helps only if the cache spends it as MIN does: it caches only the experts MIN would cache,
and loads each from host memory once". `paper/paper.tex:78` (finding 2, which this commit changed): "Foresight pays only
when spent as the optimum spends it". The data say that other ways of spending foresight also beat the deployed cache:
- MIN with two reads (MIN's set, loaded twice): 1.19× at gpt-oss 25% on 100a, 1.10–1.19× at 25% on all four job-100 hosts,
  and 1.22× at gpt-oss 40% on O5.
- Belady with one read: 1.15× at Qwen3 12.5% on O4.
- The deployed policy with a 16-token window: up to 1.12× (100a, 25%).
- Belady prefetched: 1.67× at Qwen3 43.75% on O4.

The body says it correctly: "Spent the usual ways, it pays less and sometimes loses" (l.235). The introduction itself
says two sentences later that the usual ways "gain at most 15%". Fix: "helps most / pays fully only when …".

**2. Medium: the range for "either choice alone" is no longer right with the new machines.** `paper/paper.tex:303-304`:
"either choice alone is worth \fsBalSetTwoMin{} to \fsBalSetTwoMax\% [−13 to 26%] of the gap". The range covers only
*cache alone*. With N100f added, *load alone* on the fast-link hosts reaches −14.6% (9950X x8, gpt-oss 25%). The macro
\fsBalReadsOnlineMin is −15. The old minimum was −4, which fell inside the range, so the new machines are what broke it.
Fix: use the minimum of the two macros, so the text reads "−15 to 26%".

**3. Medium: the abstract overstates the time model.** `paper/paper.tex:52-53`: "A model of the two read paths … predicts
these times to a few percent." "These times" follows the copy "made ahead of use". For every state whose copies run in the
background or ahead (MIN with two reads, MIN prefetched), the model predicts too fast by a median of 16%, at 64 of 64
host-budget-states (l.275). For the in-step states the median error is 2.2% but the 90th percentile is 10%, and the
probe-only errors reach 19–20%. The body adds (l.273) that the model "does not resolve effects of a few percent". Fix:
"predicts the time of the deployed and in-step states to a median of 2–4% (the copies in the background it puts too fast)".

**4. Medium-low: abstract, "the copy pays only where the PCIe link is not much slower … or when it is made ahead of use"
(`paper/paper.tex:51-52`).**
- On Pf at 11% the copy made ahead breaks even rather than pays: 1.008× and 0.997× on the two launches.
- On Pd (link-to-CPU 0.40, which §4 and the conclusion class as a slow link) MIN with one read in the step does pay, on both
  launches: 1.05× and 1.12× (099d), 1.05× and 1.11× (100a).

Fix: name the one machine where it loses, as the introduction does (l.83-84).

**5. Medium-low: "never loses" holds only within a tolerance the text does not state.** `paper/paper.tex:392-393`: "Across
every host we ran, only MIN with its copies made ahead never loses to the deployed cache." On Pf's relaunch (101a) at
gpt-oss 11%, both3p/base is 0.997 (interval 0.987–1.007). That is a loss on the point estimate, which l.84 calls "breaks
even". Under the same tolerance, the deployed-path 16-token window, at no lower than 0.9995× on 4 hosts, also never loses,
so whether "only" holds depends on the unstated tolerance and scope. Fix: "never loses beyond its interval (lowest 0.997×,
Pf relaunched, 11%)".

**6. Medium-low: an unsupported claim about the rental market.** `paper/paper.tex:491-492`: "Searching the rental market on
the day of the follow-ups, we found no other desktop host that slow." No artifact for this search is committed: there is
no 6 October listing snapshot in `prereg/`, `jobs/` or `gpu/`. "That slow" is also undefined:
- By link rate, the claim is contradicted by 101b (26.6 GB/s, the same as Pf), 100f (21.8 GB/s) and Pi (19.0 GB/s).
- By link-to-CPU ratio, a listing does not show the CPU's read rate, so a market search cannot establish it.

Fix: commit the snapshot and state the criterion, or drop the sentence.

**7. Low-medium: the hostdep caption is stale.** `paper/paper.tex:281-282` says "on hosts O3–O6, the panel and the new
machines of job 100". `fig_hostdep.py` now also plots job 101's 9800X3D, and \hdHosts = 17 (l.249). The caption lists 16.
Fix: "… of jobs 100 and 101".

**8. Low: two passages contradict each other.** Terms (`paper/paper.tex:161-162`) says an expert "served by the CPU and
copied in the background" has "two host reads, the second off the critical path". The worked example (l.294-295) says that
loaded the deployed way each expert "is read twice, the second time on the critical path". Fix: reconcile the two, for
example "the second contends with later steps' reads".

**9. Low: the deployed state is an exception to a claim about background copies.** `paper/paper.tex:275-276`: "it predicts
every state whose copies run in the background faster than it runs (median 16% over 64)". The deployed state's admissions
are also copied in the background (l.129, l.161). It is in the 2.2% set, and its frozen-G error is negative at only 19 of
39 host-budgets (range −13% to +3.7%). The 64 cases are bypass and both3p only. Fix: "every other state …".

**10. Low: the abstract says "at most a quarter" where the maximum is 25.8%.** `paper/paper.tex:50`: "against at most a
quarter for either alone". The maximum is 25.8% (\fsBalSetTwoMax = 26). Fix: "at most 26%" or "about a quarter".

**11. Low: the abstract lost a scope qualifier.** `paper/paper.tex:53-55`: "half of its saving in host reads needs exact
routing of the next \vmHalfMin{} to \vmHalfMax{} tokens [2–10], about 0.65 C distinct experts per layer across nine models".
The commit dropped "at the budgets where host memory binds". The 2–10 tokens are the four host-bound cells of gpt-oss and
Qwen3 only (`value_paper.py`). Across nine models W50 runs from 0.6 to 34 tokens (\dcWMin, \dcWMax). As written, "2 to 10"
reads as a nine-model range. Fix: restore the scope.

**12. Low: the rounded numbers in the worked example do not add up.** `paper/paper.tex:291-293`: 4.5 + 3.9 + 43.2 = 51.6,
but "both" is printed as 51.5. The exact values are 4.48 + 3.88 + 43.19 = 51.55. The macros now print one decimal place
specifically so that readers can check the sum. Fix: print the interaction as the difference of the rounded values, or
add "(rounding)".

**13. Low: two descriptions of the deployed path conflict with the body.** `paper/app_value.tex:11` says "on the deployed
path of the follow-up the CPU serves the miss … two reads". The `tab_window100.tex` caption says "the CPU serving each
miss". Both conflict with `paper/paper.tex:383-384`, which says the fetch table sends the rest over the link. On
100b/100c at 11%, 47–50% of the deployed-path misses are fetched in the step (b16: 24.6 of 52.2 per step). "Most" at
l.383 is only just true there: the CPU serves 50.1% of the deployed state's misses. Fix: align the appendix and the
caption with l.383.

**14. Low: an incomplete clause count.** `paper/paper.tex:154-155`: "the \scbN{} and \sccN{} of the panel and its
follow-up by machine" leaves out job 101's 25 machine-scored clauses, which are now in app:prereg (`app_wsg.tex:96`) and in
the supplement. Fix: add "and the 25 of job 101".

**15. Low: a row label contradicts its own caption.** In `paper/tab_shapley.tex:34-35` (label set in
`scripts/factorial_shapley.py:289`), the row "9800X3D, slow (0.61)" contradicts the caption's rule that a fast link reads
at least half as fast as the CPU. At 0.61 the host is fast, and it is counted in \fsBalHosts = 13. Fix: "9800X3D, slower
link".

**16. Low: the job 101 outcome note gets the over-prediction range wrong.** `prereg/crossover_outcome_101.md:30-31` says
the model "over-predicts the in-step gain where the link is slower than the CPU (here by 0.06-0.22)". At Pf 25% it
under-predicts instead: 0.949 predicted against 0.966 measured, a difference of −0.02. The 9800X3D at 11% is over-predicted
by 0.226, which rounds to 0.23. Fix: "at three of four cells, by 0.06–0.23 (under by 0.02 at Pf 25%)".

**17. Low: a duplicate clause id.** In `prereg/scorecard_101.json:8,22`, the id "101a-P1-fetch-g11" is used for two
different clauses: the sign clause and the band of 0.03. This comes from `job101.py`'s `add` for fetch at C=14. Fix: give
the sign clause its own id.

**18. Low: two smaller issues in the methods and section 4.**
- `paper/paper.tex:148-149`: "A relaunch can still move a ratio by up to 0.030, more than the within-launch interval". This
  holds against the median within-launch half-width (0.007–0.008). It does not hold for Pf's first launch at 11%, where
  the half-widths are 0.028 (fetch), 0.033 (bypass) and 0.042 (both3p). Fix: "more than the typical (median 0.008)
  within-launch half-width".
- `paper/paper.tex:290-291`: "The engine ran all four combinations on O4, O5 and the panel" is now incomplete, because the
  accounting also uses the three new machines (13 = O4, O5, 8 panel hosts and 3 new).

**19. Low: three typography and wording points.**
- `paper/paper.tex:307`: \fsBalReadsMin--\fsBalReadsMax now renders as "−6–32%". Use "to", as the other negative ranges do.
- `paper/paper.tex:272`: "a host with a half-width link" does not identify 100f uniquely. Pf (099f and 101a) and Pc also
  run at x8 (`gpu.csv` width.current 8), and on Pf the probe-only aa error is at most 2.2%, so the link width is not
  itself the cause.
- `paper/paper.tex:85`: finding 2 says "every state that reads in the step within a median 2.2%". The 2.2% now includes
  the deployed state; l.261 says so and the introduction does not.

## Verified (recomputed independently; matches)

**Job 101 (`wsg_job101.tex`, outcome note).**
- 101a fetch/base at 11% is 0.860, with a 10,000-resample interval of [0.855, 0.866]; the paper prints 0.86 [0.85, 0.87].
  099f was 0.8749.
- Largest relaunch change over fetch, foa, aa, both3p and bypass at both cells: 0.0256 (foa, 11%), so 0.026.
- Base time differs by 1.65% at 11% and 0.02% at 25%.
- 101b fetch/base is 1.137 [1.132, 1.142] at 11% and 1.223 at 25%. Pb was 1.343.
- Link-to-CPU ratios: 101a 0.289, 101b 0.606. Pb probe: link 45.7 / CPU 44.2.
- The outcome-note table: all 20 ratios check.
- Model with frozen G and the run's counters: 0.922 and 0.949 (Pf), 1.363 and 1.390 (9800X3D). The signs are right at 4 of 4.
- Probe-only errors: aa +1.6%, +2.2%, −0.1%, −0.5%; fetch −10.2%, −4.5%, −13.1%, −11.0%.
- The 101a fetch interval at 25% is [0.957, 0.974], so "lost again" holds at both budgets.
- Tally: 25 clauses, 6 held and 19 held on the point estimate. The predictions in the job header (commit 8b6c67a) match the
  scored clauses.
- Spend: 8.18 − 6.86 = 1.32. The ledger's GPU cost rises by $1.09 (69.7 → 70.8), with 88 rentals on 53 machines.

**Relaunch maxima (jcRelaunch\*).** Over 100a/099d, 100c/099h and 101a/099f, for every common configuration (bypass
included): the largest ratio change is 0.0296 (100c both3p, 11%), so 0.030. The largest base-time change is 1.88%, so 1.9,
and the number of relaunched machines is 3.

**Accounting (fsBal\*, fsSlow\*, fsSlack\*, fsEx\*, `tab_shapley`).**
- Recomputed from raw for 15 hosts and 34 host-budgets, with the stored bounds. For the new machines the bound is
  recomputed by `limit1_of`, which reproduces Pa's and Pf's stored panel bounds exactly. N-host ratios are 0.917, 0.540
  and 0.606.
- Fast-link hosts (13): cache alone −12.6 to 25.8, interaction 10.2 to 58.3, both 13.6 to 60.1 (low end N100f 11%), Shapley
  cache 5.4 to 34.7, Shapley load −6.4 to 32.0, prefetch −3.7 to 27.6, rest 29.9 to 61.6, load alone −14.6 to 4.5.
- Slow-link hosts: load costs time at 3 of 4 host-budgets, Shapley −21.0 to 2.3, rest 68.0 to 98.8.
- Slack: 0.5 to 8.7 on fast-link hosts and 0.4 to 1.9 on slow-link ones. The two-path bound never binds either path on any
  of the 34 host-budgets, and it equals the bound at B_cp exactly.
- The O4 worked example: gap 10.62 ms; cache alone 4.48%, load alone 3.88%, both 51.55%, interaction 43.19%, Shapley 26.08
  and 25.47%, prefetch 12.73%, rest 35.72%.
- fxExAdmitsOverOnline: 156022 / 26560 = 5.87.
- Every row of `tab_shapley`, including the panel-group means and ranges and the new-machine rows, checks.

**Model (hmFrozenStep\*, \hmFetch\*, \hmFrozenBg\*).**
- Over 14 hosts and 39 host-budgets, with 097a excluded: the median of 239 errors is 2.15% and the 90th percentile is 10.2.
- G ranges from 4.08 to 9.22 ms.
- Fetch sign: right at 39 of 39 with frozen G. There are 2 losses, both on Pf. The one-path model is right at 37.
- Background-copy states: 64 of 64 are predicted faster than measured, with a median error of 16.1%.
- Frozen G: the median over 15 fits (097a included) is 4.563 and 4.532 ms; over 14 it is 4.635 and 4.553 ms. This confirms
  the note added to `window_outcome_100.md`.

**Job 100 (ja\*).**
- jaWorstBLossPct: the lowest ratio is 0.9429 (b8r5, 100a, 11%), so 6%.
- Replay deviations run from 6.6% to 33.0%. 18 clauses exceed 10%, the smallest of them 10.5%.
- The deployed state against its replay: at most 2.43%.
- foa runs 0.903–1.030×. The model's base/foa is 1.006–1.015, so it is within 2% (\jaFoaPredDevMax).
- b16 runs 0.9995–1.123. b8r5 at 11% runs 0.943–0.980, all four below 1. b4 is at least 0.960.
- b16 exceeds bypass at all four 11% cells, by at most 0.026.
- At 16 tokens the deployed path wins at link-to-CPU 0.41 and 0.54 and the in-step path wins at 0.92 and 0.96, at both
  budgets.
- MIN with one read beats MIN with two reads at 3 of 4 low-ratio cells.
- Probe-only: median 3.83% over 32 predictions (still 3.83% with job 101's 8 added), 100a fetch −18.6%, 100f aa +20.1%.
  On 100f fetch is not over-predicted (−2.0% and +1.9%), so "admitting every miss and the windows" is right.
- At 11% on the panel, the 8-token half-right window loses on 8 of 10 hosts and the 4-token window on 6 of 10.

**Other.**
- hdHosts 17. Correlations: 0.929 (11%), 0.872 (25%), 0.826 and 0.724 without Pd and Pf.
- In the step, fetch loses only on Pf, on both launches, out of every cell of jobs 095–101.
- both3p is below 1 only at 101a, 11%.
- fxMinOverUsual (O4 and O5): 1.183–1.425. fxUsualLow: 0.938–1.149. fxFetchLow: 1.159–1.438.
- fxFetchAllGain is 16–51% and fxPacedAllGain 25–81%, over O3–O5.
- fxReplicateMax: 0.024 (both3p, gpt-oss 40%).
- pnSameCpuDiff: 15.6–29.1.
- The held-out rule (median 1.16) does predict "as well as" the power law (1.19).
- \audOursMed 27 is in the body (l.205).
- The window algorithm in the text agrees with `app_value.tex`.
- `app_wsg.tex` (job 101 sentence and row counts) and the inclusion of `tab_scorecard_101` in the supplement check.
