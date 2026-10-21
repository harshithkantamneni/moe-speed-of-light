#!/bin/bash
# Job 123, host c (offer 56685461, RTX 5090, 12th Gen Core i5-12400F): jobs/123_confirm@vast.sh (its header holds the gates and the clauses); the listing results/123_offers_20261021T062015Z.txt.

exec bash "$(cd "$(dirname "$0")" && pwd)/123_confirm@vast.sh"
