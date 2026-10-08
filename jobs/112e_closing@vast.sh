#!/bin/bash
# Job 112, host e (offer 53783561, RTX 4090 next to a Ryzen 9 8945HX; a replacement): jobs/112_closing@vast.sh (its header holds the gates and the predictions).
export CARD=4090
exec bash "$(cd "$(dirname "$0")" && pwd)/112_closing@vast.sh"
