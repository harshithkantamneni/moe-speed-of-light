#!/bin/bash
# Job 112, host f (offer 54729181, RTX 4090 next to a Core i7-10700KF; a replacement): jobs/112_closing@vast.sh (its header holds the gates and the predictions).
export CARD=4090
exec bash "$(cd "$(dirname "$0")" && pwd)/112_closing@vast.sh"
