#!/bin/bash
# Job 123, host h (offer 56472585, RTX 5090, Ryzen 9 9950X): jobs/123_confirm@vast.sh (its header holds the gates and the clauses); the listing results/123_offers_20261021T084032Z.txt.

exec bash "$(cd "$(dirname "$0")" && pwd)/123_confirm@vast.sh"
