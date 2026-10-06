# Number check 9 (6 October, evening)

Scope: the changes in commit f3997ae against 8b4fe22 in `paper/paper.tex`, `paper/app_wsg.tex`, `paper/supplement.tex`,
`prereg/readsched_outcome_102.md`, `scripts/job102.py` and `scripts/readsched.py`, together with the files they generate
(`wsg_job102.tex`, `wsg_readsched.tex`, `tab_readsched.tex`, `tab_scorecard_102.tex`, `prereg/readsched.json`,
`prereg/readsched_gap.json`, `prereg/job102.json`, `prereg/scorecard_102.json`). I recomputed every number with my own
Python from these sources:
- the raw job 102 files (`readsched_C14/C32.txt`, `rates.txt`, `concur.txt` in `results/102?_readsched@vast`);
- MIN with bypass, rerun with `mosl.cachesim.simulate` on `084c_gptoss_trace@vast/route_aime25_gptoss.npz`;
- the probe parsers (`fetch_table.bandwidths`, `speed_limit.host_rates`);
- the stored `limit_ms`, `gap` and shares in `prereg/factorial_shapley.json`.

I also reran `scripts/readsched.py` and `scripts/job102.py` in a scratch copy of the repository. I compiled the paper and
the supplement in a scratch copy of `paper/`. Line numbers refer to f3997ae.

## Defects (most severe first)

**1. High: the read-once "cap" uses B_p, which is not the link's ceiling. Job 102 itself exceeds it.**
`scripts/readsched.py:93` computes `cap = min(B_cp, (R*/A*)·B_p)/B_host`. Here B_p is the probe's zero-copy line
(16 MB, 64 blocks), not the highest rate the probe measured on the link. The text rests on it as a hard limit:
- "caps MIN's one-read schedule at 50–71% … against 74–99% on the others" (`paper.tex:220-222`);
- "no schedule that admits what MIN admits and reads each expert once can close 25–63% of the gap" (`paper.tex:329-331`).

The data contradict B_p as a ceiling:
- In job 102's own *link only* mode, the copy engine moved 49.8 GB/s on the 9950X (B_p 46.4) and 28.4 GB/s on the 9800X3D
  (B_p 26.6).
- The paper itself says the probe read 100f's link "more slowly than the engine's copies ran" (`paper.tex:283-284`). There
  the probe's copy-engine line reads 25.3 GB/s and the engine about 27 GB/s, against B_p = 21.8.

100f sets the low end of the "others" range (74% at 11%, 61% at 25%) and the largest fast-link share of the gap (42%). On
Pf the copy-engine line is 28.7 against B_p = 26.6.

With the highest probed link line in place of B_p (the formula otherwise unchanged):

| | Paper (B_p) | Highest link line |
|---|---|---|
| Cap, slow-link probes, 11% | 50–71% | 53–71% |
| Cap, others, 11% | 74–99% | 86–99% |
| Cap, slow-link probes, 25% | 41–58% | 43–58% |
| Cap, others, 25% | 61–98% | 70–98% |
| Share of the gap, slow-link hosts | 25–63% | 25–54% |
| Share of the gap, fast-link hosts | median 4%, max 42% | median 3%, max 22% |

Fix: use the highest probed link rate, i.e. the maximum over the `pcie_*` lines. Then state the cap "at the probe's
highest link rate", with the same caveat the bound carries for B_host ("a faster reader could exceed it").

**2. Medium-high: on fast-link hosts the cap is the B_cp slack, not the link, and the paragraph counts that slack twice.**
`paper.tex:220-222` says "The link must carry them at its own rate, which caps …". But at 11% the link term
(R*/A*)·B_p/B_host is ≥ 1 on 15 of the 17 probes that are not slow-link (all but O5 and 100f). On those probes the "cap" is
just B_cp/B_host.

In the factorial cells this makes the cap's share of the gap identical to `slack_share` at 18 of 26 fast-link host-budgets.
For example, at O4 11% both are 5.10%. That is the quantity `paper.tex:331-334` goes on to call "little … 1–9% of the
gap". So the fast-link "median of 4%" (`paper.tex:331`) is that slack, not the link. Without the B_cp term the fast-link
median is 0%.

Fix: define the link cap as min(1, (R*/A*)·B_p^max/B_host). Report the B_cp slack only once, in the existing sentence.

**3. Medium: the cap is reported at one budget and by median only, which hides where it binds.**
- `paper.tex:221-222` gives only 11%. At 25% the "others" go down to 61% (100f), 75–76% (O5, 101b) and 82% (Pc). That
  overlaps the slow-link range at 11% (50–71%).
- `paper.tex:331` gives the fast-link hosts "a median of 4%" of the gap. But 5 of the 26 fast-link host-budgets are at
  10–42%: N100f 42% and 34%, N101b 15%, O5 14%, Pc 10%.
- On job 102's own 9800X3D (link/CPU 0.55, "fast" by the paper's 0.5 line), MIN's one-read cap is 88% at 11% and 72% at
  25%. Those caps are below the 95% and 93% that its free split measured.

So "Where the link is much slower than the CPU" (`paper.tex:82-83`) and "on a slow link MIN's one-read schedule is capped"
(`paper.tex:517`) understate where the constraint binds.

Fix:
- Give both budgets: "50–71% at 11% and 41–58% at 25% on the two slow-link machines; 74–99% and 61–98% on the others"
  (use the values from defect 1 once it is fixed).
- Give the range with the median: "median 4%, range 1–42%".
- Word the caveat as "where the link is slower than the CPU", or name where the cap binds. At 25% the link term binds
  on every probe with a link/CPU ratio up to 0.77 (Pi), not only below 0.5.

**4. Medium: abstract and conclusion: "comes within 14% of the bound's read time"** (`paper.tex:48-50, 532-533`).
- The arithmetic is off. The worst per-layer cell (Pd again, C=14) reaches 86.07% of the host term, i.e. 8.178 ms against
  7.039 ms: 16.2% more time. "14%" is 100 − 86, the shortfall in fraction, not in time.
- The claim is also unqualified. The replay puts each layer's reads on either path freely. On 2 of the 3 hosts, MIN's
  one-read schedule cannot use that split:
  - Pd again: cap 69% at 11% and 57% at 25%, against 86–89% measured for the split;
  - 9800X3D: cap 88% and 72%, against 93–95% measured.
- The conclusion's "MIN's reads, replayed alone, come within 14% …, yet our cache reaches 25–43%" reads as if a cache
  could reach the read time.

Fix: "reaches at least 86% of the bound's read time when each layer's reads may go to either path". Add in both places
that MIN's own one-read schedule is capped below that where the link is slow.

**5. Medium: the extrapolation to "the other 21 probed hosts" (`paper.tex:215-217`) is optimistic, and "other" is wrong.**
- The model's worst error (1.14×) is the per-layer mode on the slow-link host, which is exactly the mode being
  extrapolated. On Pd it predicted 98.3% where 86.1% was measured.
- The 21-probe maximum, 99.0%, is 100a, Pd's own relaunch; 099d (Pd) gives 97.2%. The measurement on that same machine is
  86%.
- The 21 are launches of 18 machines: Pd (099d, 100a), Pf (099f, 101a) and Ph (099h, 100c) appear twice each. Pd is also
  job 102's host a, so not every one of the 21 is "other".

Fix: "on 21 earlier probes (18 machines) the model gives 85–99% at 11%; on the slow-link machine it was 12 points
optimistic (98% against 86% measured)".

**6. Medium: wrong description of MIN with one read.** `app_wsg.tex:482-483`: "*link only* sends every read over PCIe, as
MIN with one read does". The same claim is in the comment at `jobs/ec2/readsched.cu:7`. But MIN with one read fetches only
its admissions over PCIe; "every other miss runs on the CPU" (`app_wsg.tex:319-320`, `tab:configs`; `paper.tex:405`:
"50% of MIN's" misses cross the link). Admissions are 57% of MIN's reads at 11% and 69% at 25%. Sending every read over
the link is what *admit every miss* does (`paper.tex:269-270`).

Fix: "as admitting every miss does; MIN with one read sends only its admissions over PCIe".

**7. Medium-low: Section 5's opening overstates the deployed-path result** (`paper.tex:345-348`).
The text: "a window pays only where the link has room: … given to the deployed policy, it gains at most 12%, and shorter
or half-right windows lose". `tab_window100` says otherwise:
- The 12% gain is on Pd again, link/CPU 0.41. On the deployed path a 16-token window gains 1.06–1.12× at 25% on all four
  hosts, whatever the ratio.
- 4-token deployed-path windows run 0.96–1.03× and gain at 6 of 8 host-budgets.
- Half-right 8-token windows run 1.00–1.01× at 25%.

Fix: "Copied in the step, a window pays only where the link has room (0.71–1.31× at 11%); on the deployed path a 16-token
window gains 0–12% on every host, slow link included, and shorter or half-right windows run 0.94–1.03×."

**8. Medium-low: "slow-link" refers to different machines in different places.**
- The Fig. 1 caption (`paper.tex:247-248`, new) says "The one panel machine whose link is much slower than its CPU is the
  exception". But Limitations (`paper.tex:505`) says "Only 2 panel hosts have a link slower than half their CPU rate".
  `paper.tex:318` counts 2 hosts below the line, and the conclusion (`paper.tex:536`) uses "not much slower" for the ≥ 0.5
  class. That makes Pd (0.40) "much slower" too.
- In the model bullet, "costs up to 19% on the slow-link machine" (`paper.tex:283`) refers to Pd (job 100a, −19%). But
  `paper.tex:268` has just introduced Pf as "the one machine whose link reads at 0.29", so readers will take it to mean Pf.

Fix: in the caption, "The panel machine whose link reads at 0.29 of its CPU rate is the exception"; at line 283, "on Pd
(link/CPU 0.41)".

**9. Low-medium: a dropped qualifier** (`paper.tex:318`). "On the 2 hosts below that line, loading in the step costs
time". It does so at only 3 of the 4 host-budgets. At Pd 11% the Shapley value of loading is +2.3% of the gap: loading in
the step gains there. The old text had "at \fsSlowReadsLoseN{} of \fsSlowCells{} host-budgets".

Fix: restore that qualifier.

**10. Low-medium: the conclusion attributes the rest of the gap exhaustively** (`paper.tex:538-539`). "The rest of the gap
on the other hosts, 30–62%, is an imprecise prefetch, partial overlap and a GPU at half its datasheet rate." Section 4 does
not quantify these three parts; it calls them "visible parts". On the same hosts the paper also identifies:
- the B_cp slack, 1–9% of the gap;
- helper compute, 1.09–1.31× the probe's time per expert;
- by the new cap, link-bound shares of up to 42% (100f), or 22% at the highest link line.

Fix: change "is" to "includes".

**11. Low** (`app_wsg.tex:485-486`). "There, reading everything on the CPU beats the split." On Pd, CPU only reaches 92%.
That beats the per-layer split (86% and 89%) but not the per-token split (93% and 93%).

Fix: "beats the per-layer split".

**12. Low: scope dropped in "Spent the usual ways"** (`paper.tex:260-262`).
- `fxUsualLow`, `fxUsualLowGainMax` and `fxFetchLow` cover O4 and O5 only (`factorial_paper.py`, `new`). The paragraph
  before sets "hosts O3–O5". The same applies to `fxBestFrac` at `paper.tex:254-255`.
- The usual ways' range, 0.94–1.15×, includes a loss. The body no longer mentions it; the introduction says "or lose".

Fix: add "on O4 and O5", and "or lose".

**13. Low** (`paper.tex:339`). "A relaunch of a machine reproduces its ratios (\cref{sec:setting})". The cited section
says a relaunch can move a ratio by up to 0.030, more than a within-launch interval (`paper.tex:156-158`). The O4 relaunch
(`\fxReplicateMax` = 0.024) is no longer cited anywhere.

Fix: "reproduces its ratios within 0.03".

**14. Low** (`prereg/readsched_outcome_102.md:26-27`). The note says the model is "pessimistic for the link (copies run
faster than the probe's link rate)". That holds on the two AMD hosts (measured/model 0.93–0.94). On Pd, link only ran 1.11×
the modelled time: 25.7 GB/s against B_p 28.5.

Fix: scope the sentence to the AMD hosts.

## Checked and held

**MIN on the trace**
- My simulation reproduces `minreads_g14.bin` and `minreads_g32.bin` exactly.
- R* is 38.32 at C=14 and 15.34 at C=32 (`\rsRstar`). Over the first 4,000 steps it is 38.505 and 15.520, which match the
  `reads_per_token` the job printed.
- A* is 21.77 and 10.62, so admission shares are 56.8% and 69.3% (`\rsAdmShare` 57, 69).
- Misses per layer are 1.06 and 0.43.

**Job 102 table and macros**
- B_c, B_p and B_cp from `bandwidths` equal `rates.txt` on all three hosts. B_host is 72.5, 52.6 and 53.4 GB/s.
- The host terms are 7.04/2.84, 9.70/3.91 and 9.56/3.85 ms.
- All 36 entries of `tab_readsched` hold, and so do the link/CPU ratios (0.40, 1.02, 0.55) and the Model column.
- The largest disagreement between repetitions is 0.875% ("0.9%").
- Layer 86–96%, token 93–97%, measured/model 0.93–1.14 (24 cells), wait cost 7.2 (Intel) and 1.1 (AMD).

**Job 102 scoring**
- 33 clauses: 11 per host, with P4 applying on all three.
- 28 held on the point estimate. 5 failed:
  - P2below on the 9950X (−0.022, −0.035);
  - P2below on the 9800X3D (0.001, 0.002);
  - P3 on Pd (−0.029).
- 0 untested. The appendix paragraph (`app_wsg.tex:99-103`) and the supplement's Table 5 match.

**Commit order on the gpu branch (UTC)**

| Time | Event |
|---|---|
| 15:14:17 | e089377: predictions committed |
| 15:14:31 | rental 102b created (ledger) |
| 15:14:32 | rental 102c created |
| 15:14:41 | 3701cea: host a changed to Pd |
| 15:14:44 | rental 102a created |
| 15:15:11 | script start (T0), 102b |
| 15:15:17 | script start, 102c |
| 15:19:23 | script start, 102a |

- The predictions precede every rental. The amendment changes only 102a's identity and precedes 102a's rental.
- P4's "CPU ≥ 1.5× link" branch applies to Pd (2.49×) as it would have to Pf.
- `scorecard_102.json` cites e089377, and the outcome note records the amendment.

**The 21-probe model numbers and the cap macros**
- `\rsFrac` 85–99 and 82–99.
- With the formula as written: cap 50–71 / 74–99 at 11% and 41–58 / 61–98 at 25%.
- Share of the gap: slow-link 25–63 (Pd 28.8/25.3, Pf 63.0/42.9); fast-link median 4.2 over 26 cells, range 0.9–41.8.
- `limit_ms` equals R*·S/B_host for the same results directory, so the cap and the bound share one B_host.

**Logic of the cap**
- (limit/cap − limit)/gap is the part of the gap below the cap. "No schedule that admits what MIN admits and reads each
  expert once can close X%" is therefore a fair reading.
- Both measured one-read oracles (in-step and copied ahead) run slower than limit/cap at all 30 cells, so no measurement
  breaks the cap as written.
- Every mention scopes it to MIN's one-read schedule, never to every policy: `paper.tex:221`, 330, 83, 517 and 537.

**Reruns and build**
- In a scratch copy, `scripts/readsched.py` and `scripts/job102.py` regenerate all seven output files byte for byte.
- `latexmk` in a scratch copy builds `paper.pdf` (31 pp.) and `supplement.pdf` (30 pp.) with no undefined references and
  no `??`. Their extracted text is identical to the committed PDFs.

**Section 4 and 5 facts kept from before the rewrite**
- 15 accounting hosts = 13 fast-link + 2 slow-link.
- The model gets the sign right at 39 of 39 host-budgets; the one-path model gets 37. Both losses are on Pf, and the model
  predicts them on both launches.
- The paths at 16 tokens split 2/2 as stated: in step loses (0.89/0.81, 0.89/0.80) and wins (1.32/1.23, 1.20/1.09).
- Deployed path 16 tokens: 1.00–1.03× at 11% and 1.06–1.12× at 25%. The half-right 8-token window loses on all four hosts
  at 11% (0.94–0.98×).
- Copies made ahead: 1.01× minimum, 0.997× on the Pf relaunch, which fits "do not lose … break even".

**Appendix description of the microbenchmark** (`app_wsg.tex:477-482`) matches `readsched.cu`: H = usable cores − 2,
13,253,760-byte copies from pinned memory, a 4 GiB rotation, and the per-layer and per-token waits.
