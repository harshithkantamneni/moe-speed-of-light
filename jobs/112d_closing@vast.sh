#!/bin/bash
# Job 112, host d (offer 54227874, RTX 5090, job 109c's Ryzen 9 7945HX machine): jobs/112_closing@vast.sh (its header holds the gates and the predictions).
export CARD=5090
exec bash "$(cd "$(dirname "$0")" && pwd)/112_closing@vast.sh"
