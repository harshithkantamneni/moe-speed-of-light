#!/bin/bash
# Job 123, host b (offer 52870669, RTX 5090, Ryzen 9 9900X): jobs/123_confirm@vast.sh (its header holds the gates and the clauses); the listing results/123_offers_20261021T060457Z.txt.

exec bash "$(cd "$(dirname "$0")" && pwd)/123_confirm@vast.sh"
