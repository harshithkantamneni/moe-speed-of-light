# Jobs 103 and 104 outcome: MIN's fewest-admission schedule against the greedy one

Scripts: `jobs/103_minadm@vast.sh` (gpu c19e186) and `jobs/104_minadm@vast.sh` (gpu dbabcc1), predictions in each header,
committed before launch. Engine patch oracle4 (`LLAMA_EC_ORACLE_PLAN`), plan from `jobs/ec2/minadm_plan.py` (LP over
reuse intervals under the engine's rule for a copy in the step; integral on every layer). Scored by machine by
`scripts/job103.py` (`prereg/scorecard_103.json`, `prereg/scorecard_104.json`).

## Job 103: the plan never ran

Every host failed to install scipy beside Debian's numpy ("Cannot uninstall numpy 1.26.4"), so `minadm_plan.py` did not
run and the plan configurations ran greedy MIN (the engine logs "COULD NOT BE READ"; their stats say oracle_plan 0).
Every plan clause is untested. One host (103c, Ryzen 7 7800X3D) never started and was destroyed without results. The
greedy states give two new machines: a second Core Ultra 9 285K (probe: copy engine 49.6, zero-copy 39.9 GB/s; ratio 0.48: fetch/base 1.05
at gpt-oss 11%) and a Threadripper 3970X (0.40: 1.02). The Ryzen 9 5950X of 103d (0.62: 1.25) is panel host Pg again
(same offer and GPU as job 099g; number check 10), so it is a relaunch, not a new machine. Pf, launched a third time, again
loses with the greedy copy in the step (0.859; 0.875 and 0.860 before).

## Job 104: the plan on three machines

| Host (ratio) | cell | bypass | bypassplan | fetch | fetchplan | copies greedy -> plan |
|---|---|---|---|---|---|---|
| Pf (0.29) | 11% | 0.941 | 1.113 | 0.863 | 1.035 [1.023, 1.052] | 19.8 -> 11.8 |
| Pf (0.29) | 25% | 1.133 | 1.217 | 0.964 | 1.059 | 9.7 -> 7.0 |
| 285K, second (0.49) | 11% | 1.032 | 1.085 | 1.053 | 1.180 | |
| 285K, second (0.49) | 25% | 1.197 | 1.206 | 1.119 | 1.175 | |
| Pg again, 5950X (0.63) | 11% | 1.021 | 1.059 | 1.256 | 1.338 | |
| Pg again, 5950X (0.63) | 25% | 1.198 | 1.151 | 1.325 | 1.353 | |

Held: the fewest-admission set copied in the step gains on Pf (prediction 3, with its interval), and beats the greedy
set by 0.08-0.17 at 11% on all three machines (prediction 4); Pf's relaunch reproduces base within 0.3% and fetch/base
within 0.004 (prediction 8); the two-read plan state is at least the greedy one's minus 0.01 at 5 of 6 cells.

Failed: at 25% the plan copies 0.73x greedy's (predicted at most 0.70) and misses 5.3% more (predicted within 3%); the
engine's plan state misses 3.8% (11%) and 6.8% (25%) more than the host's replay of the plan, the greedy state 1.6% and
2.4% (predicted within 2%); the two-read plan reads 0.90-1.01x the greedy reads (predicted at most 0.90); the
interaction of the 2x2 grew with the plan's set on every machine (predicted to shrink); Pg's two-read plan
state at 25% is 0.047 below the greedy one.

The likely cause of the extra misses: about 2% of the forced in-step copies are not made on the GPU in both modes
(forced 20.25 vs fetches 19.81 per token for greedy), and a dropped copy of a planned chain protects a slot that does
not hold the chain's expert, so the chain's later hits are lost too. Not isolated.
