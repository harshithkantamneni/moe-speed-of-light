#!/bin/bash
# Job 122, host f (offer 54227877, RTX 5090, Ryzen 9 7950X3D): jobs/122_fastfix@vast.sh (its header holds the gates and the predictions); the listing results/122_offers_20261020T223240Z.txt.
export REPEAT=1   # a repeat host (the header's rule): no new offer qualified
exec bash "$(cd "$(dirname "$0")" && pwd)/122_fastfix@vast.sh"
