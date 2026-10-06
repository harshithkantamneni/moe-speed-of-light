#!/bin/bash
# Job 100, host e: jobs/100_window@vast.sh (its header holds the design and the predictions).
exec bash "$(cd "$(dirname "$0")" && pwd)/100_window@vast.sh"
