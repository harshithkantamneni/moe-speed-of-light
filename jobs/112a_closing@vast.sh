#!/bin/bash
# Job 112, host a (offer 49588631, RTX 4090 next to a Core i5-12400, job 091's machine): jobs/112_closing@vast.sh (its header holds the gates and the predictions).
export CARD=4090
exec bash "$(cd "$(dirname "$0")" && pwd)/112_closing@vast.sh"
