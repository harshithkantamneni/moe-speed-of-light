#!/bin/bash
# Job 112, host b (offer 52088382, RTX 4090 next to a Ryzen 7 7800X3D): jobs/112_closing@vast.sh (its header holds the gates and the predictions).
export CARD=4090
exec bash "$(cd "$(dirname "$0")" && pwd)/112_closing@vast.sh"
