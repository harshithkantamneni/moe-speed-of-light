#!/bin/bash
# Job 124, smoke run (not one of the job's hosts): jobs/124_slowlink@vast.sh with SMOKE=1 (gpt-oss-20b, 2 problems x 32
# steps, budgets 14 and 24, gates recorded but not enforced), on one host of the population. It checks that the engine
# builds, that the card and CPU gates and the inverted VR are recorded (on an RTX 4090: its V1 reference and device-read
# floor), that the plans are computed, that all ten arms complete and write their per-layer stats, and that the summary
# and the manifest are written; its numbers enter no prediction. It counts toward the job's spend rule.
export SMOKE=1
exec bash "$(cd "$(dirname "$0")" && pwd)/124_slowlink@vast.sh"
