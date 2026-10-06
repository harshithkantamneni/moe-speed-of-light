#!/bin/bash
# Job 101, host b: jobs/101_crossover@vast.sh (its header holds the design and the predictions).
exec bash "$(cd "$(dirname "$0")" && pwd)/101_crossover@vast.sh"
