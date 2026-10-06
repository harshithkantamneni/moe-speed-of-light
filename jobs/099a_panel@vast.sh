#!/bin/bash
# Job 099, panel host a: jobs/099_panel@vast.sh with QWEN=1 (its header holds the design and the predictions).
QWEN=1 exec bash "$(cd "$(dirname "$0")" && pwd)/099_panel@vast.sh"
