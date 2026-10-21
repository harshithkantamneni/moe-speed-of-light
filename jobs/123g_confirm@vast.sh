#!/bin/bash
# Job 123, host g (offer 55276777, RTX 5090, Ryzen 9 7945HX3D with Radeon): jobs/123_confirm@vast.sh (its header holds the gates and the clauses); the listing results/123_offers_20261021T081954Z.txt.

exec bash "$(cd "$(dirname "$0")" && pwd)/123_confirm@vast.sh"
