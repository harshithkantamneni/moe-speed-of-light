# Job 100: foresight on the deployed read path, and the time model from the probe alone — predictions and outcome

Predictions: header of `jobs/100_window@vast.sh` (gpu branch; commit 4fda930, hosts named in 752e1c3, hosts e and f
and prediction 9 added in 718bf41, each pushed before the launches it concerns). Engine: patch oracle3, which adds the
window to the deployed policy (LLAMA_EC_ORACLE_HYBRID without ORACLE_FETCH; CPU test `cputest100.sh`: with no window it
equals the deployed policy counter for counter at C = 4 and 12, and at C = 8 up to one admission at the trace's last
step). Results: `results/100{a,b,c,f}_window@vast`. Scoring by machine: `scripts/job100.py` (clauses
`prereg/scorecard_100.json`, statistics `prereg/job100.json`, macros `paper/wsg_job100.tex`, table
`paper/tab_window100.tex`).

**Tally: 41 held, 122 held (point), 27 failed, 26 untested (216 clauses).**

Launches: 100a = Pd of job 099 relaunched (its first attempt stopped at setup: the host's package index came back
incomplete and cmake was missing; `setup.sh` now retries). 100b = Ryzen 7 5700X3D (new). 100c = Ph of job 099
relaunched. 100d = Core i5-12400F: its 65 GB model download did not finish in 95 minutes; no timed step ran and it was
destroyed without results. 100e = a second i9-13900KF behind a faster link: stuck installing packages for 35 minutes;
destroyed without results. 100f = a second Ryzen 9 9950X behind a half-width link (x8). Spend: $5.20 of Vast credit
(13.38 before, 8.18 after).

## Speed relative to the deployed cache on the same host (gpt-oss-120b, 20 problems)

| Host | CPU | link / CPU / both | budget | foa | aa | fetch | both3p | bypass | w4 | w16 | b4 | b16 | b8r5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100a | i9-13900KF | 28 / 70 / 70 | 11% | 0.98 | 0.55 | 1.05 | 1.11 | 1.00 | 0.66 | 0.89 | 1.02 | 1.03 | 0.94 |
| 100a | i9-13900KF | 28 / 70 / 70 | 25% | 0.90 | 0.63 | 1.11 | 1.29 | 1.19 | 0.69 | 0.81 | 1.03 | 1.12 | 1.01 |
| 100f | Ryzen 9 9950X | 22 / 40 / 49 | 11% | 1.00 | 0.54 | 1.07 | 1.23 | 0.99 | 0.65 | 0.89 | 1.01 | 1.01 | 0.96 |
| 100f | Ryzen 9 9950X | 22 / 40 / 49 | 25% | 0.91 | 0.62 | 1.11 | 1.37 | 1.10 | 0.69 | 0.80 | 0.96 | 1.06 | 1.00 |
| 100b | Ryzen 7 5700X3D | 24 / 27 / 30 | 11% | 1.02 | 0.93 | 1.37 | 1.44 | 0.99 | 1.09 | 1.32 | 0.99 | 1.00 | 0.95 |
| 100b | Ryzen 7 5700X3D | 24 / 27 / 30 | 25% | 1.03 | 0.97 | 1.48 | 1.80 | 1.16 | 1.06 | 1.23 | 1.03 | 1.07 | 1.00 |
| 100c | Ryzen 9 7950X | 49 / 51 / 62 | 11% | 0.99 | 0.85 | 1.27 | 1.45 | 1.03 | 0.99 | 1.20 | 1.01 | 1.03 | 0.98 |
| 100c | Ryzen 9 7950X | 49 / 51 / 62 | 25% | 0.98 | 0.89 | 1.29 | 1.64 | 1.16 | 0.97 | 1.09 | 1.03 | 1.07 | 1.01 |

(w = window on the in-step path, extending admit-every-miss; b = the same window on the deployed path.)

## Predictions

1. **Counters of aa, w4, w16, fetch equal job 099's within 0.5%.** Held (point) at all 32 (deterministic counts).
2. **Probe-only model: pooled median |error| <= 5%; aa and w4 within 12% everywhere; fetch under-predicted where
   link/CPU < 0.7.** Pooled median 3.8% over 32 predictions: held (point). aa and w4 within 7.3% on 100a-c; **failed on
   100f** (+17 to +20%: the probe's zero-copy line read the half-width link at 21.8 GB/s; the engine's admit-every-miss
   implies about 27 GB/s, near the probe's copy-engine line, 25.3). Fetch under-predicted on 100a (-19%, -9%) and on
   100f at 11% (-2%); **failed on 100f at 25%** (+1.9%).
3. **Sign of fetch/base, foa/base, aa/base with frozen G and the run's counters, where |predicted - 1| >= 0.05.** Held
   at all 16.
4. **Deployed-path windows never lose and gain less than MIN with two reads + 0.03.** b16/base 1.00-1.12, never more
   than 0.03 above bypass/base: held at 25 and held (point) at 11; **failed at 4 band or ordering edges**: b8r5 on
   100a at 11% (0.943, band from 0.95), b16 on 100b at 11% (0.9995, band from 1.00), b4 on 100f at 25% (0.960, band
   from 0.98), and b16 not faster than b4 on 100f at 11% (0.999).
5. **The path decides which window pays (11%).** b16 beats w16 on Pd (ratio 0.41), w16 beats b16 on 100b and 100c
   (0.92, 0.96): held (point) at all 3. Not pre-registered for 100f (ratio 0.54, between the classes): b16 beats w16.
6. **Relaunches of Pd and Ph reproduce job 099.** Deployed time within 1.9% and every listed ratio within 0.030 (both3p
   on Pd at 25%, -0.029; on Ph at 11%, +0.030): held (point) at all 28.
7. **Host plan at most 150 us per step.** Held (point) at the 24 in-step windows and fetch (10-106 us). **Untested at
   24**: the deployed-path windows plan after the step, where the engine does not time the plan (it is inside the
   measured step time).
8. **Engine misses of the deployed-path windows within 10% of the instant-admission replay.** **Failed at 18 of 24**:
   the engine misses 7-33% more (most at b16), because a background copy lands steps after it is issued and the
   window's admissions are worth less late. Held (point) at 6.
9. **Same CPU, different link (gpt-oss 11%, fetch/base).** 100f (9950X, link 22) 1.07 below Pe and Pj (1.35, 1.27):
   held (point) at both. 100e never ran: untested.

## What it says

Launches of one machine agree to within 0.03 in every ratio and 2% in time, so the machine, not the launch, is the
unit of variation (job 099's machine-as-unit design stands). The time model, with its constant frozen from earlier
hosts and the counts of a different host, predicts new hosts from their probe alone to a median of 3.8%; its large
errors trace to the probe's link line (one host) and to copies in the step on a slow link (known since job 096). On
the deployed path, foresight changes only what is cached; it is safe (0.94-1.12x, losing at most 6% with half-wrong
forecasts) but small, at most what MIN with two reads gains, and its late-landing copies cost a third of the read
saving at 16 tokens. Which path pays follows the link/CPU ratio as for MIN: the deployed path wins below about 0.55,
the in-step path above about 0.9.
