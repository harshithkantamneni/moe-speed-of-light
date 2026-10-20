#!/bin/bash
# Job 122, host b (offer 52125965, RTX 4090, Core i9-14900KF): jobs/122_fastfix@vast.sh (its header holds the gates and the predictions); the listing results/122_offers_20261020T211005Z.txt.
export REPEAT=1   # a repeat host (the header's rule): no new offer qualified
exec bash "$(cd "$(dirname "$0")" && pwd)/122_fastfix@vast.sh"
