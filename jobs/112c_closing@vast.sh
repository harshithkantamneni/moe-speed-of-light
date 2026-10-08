#!/bin/bash
# Job 112, host c (offer 51325952, RTX 5090, job 109d's Core i9-13900KF machine): jobs/112_closing@vast.sh (its header holds the gates and the predictions).
export CARD=5090
exec bash "$(cd "$(dirname "$0")" && pwd)/112_closing@vast.sh"
