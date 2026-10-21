#!/bin/bash
# Job 122, host j (offer 54227873, RTX 5090, Ryzen 9 7945HX with Radeon G): jobs/122_fastfix@vast.sh (its header holds the gates and the predictions); the listing results/122_offers_20261021T002822Z.txt.
export REPEAT=1   # a repeat host (the header's rule): no new offer qualified
exec bash "$(cd "$(dirname "$0")" && pwd)/122_fastfix@vast.sh"
