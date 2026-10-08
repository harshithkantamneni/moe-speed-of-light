#!/bin/bash
# Job 112, host g (offer 54573923, RTX 5090, job 109f's Ryzen 9 5950X machine; a replacement): jobs/112_closing@vast.sh (its header holds the gates and the predictions).
export CARD=5090
exec bash "$(cd "$(dirname "$0")" && pwd)/112_closing@vast.sh"
