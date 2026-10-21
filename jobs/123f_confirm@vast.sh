#!/bin/bash
# Job 123, host f (offer 56767427, RTX 5090, Ryzen 9 5950X): jobs/123_confirm@vast.sh (its header holds the gates and the clauses); the listing results/123_offers_20261021T081446Z.txt.

exec bash "$(cd "$(dirname "$0")" && pwd)/123_confirm@vast.sh"
