#!/bin/bash
# Job 123, host i (offer 53961692, RTX 5090, Ryzen 9 9950X3D): jobs/123_confirm@vast.sh (its header holds the gates and the clauses); the listing results/123_offers_20261021T112938Z.txt.

exec bash "$(cd "$(dirname "$0")" && pwd)/123_confirm@vast.sh"
