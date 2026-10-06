#!/bin/bash
# Job 099, panel host g: jobs/099_panel@vast.sh with QWEN=0 (its header holds the design and the predictions).
QWEN=0 exec bash "$(cd "$(dirname "$0")" && pwd)/099_panel@vast.sh"
