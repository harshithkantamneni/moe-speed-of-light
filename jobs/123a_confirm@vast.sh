#!/bin/bash
# Job 123, host a (offer 55608456, RTX 5090, Core Ultra 9 285K): jobs/123_confirm@vast.sh (its header holds the gates and the clauses); the listing results/123_offers_20261021T060432Z.txt.

exec bash "$(cd "$(dirname "$0")" && pwd)/123_confirm@vast.sh"
