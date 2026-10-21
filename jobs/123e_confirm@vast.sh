#!/bin/bash
# Job 123, host e (offer 56434459, RTX 5090, Core i9-14900K): jobs/123_confirm@vast.sh (its header holds the gates and the clauses); the listing results/123_offers_20261021T073241Z.txt.

exec bash "$(cd "$(dirname "$0")" && pwd)/123_confirm@vast.sh"
