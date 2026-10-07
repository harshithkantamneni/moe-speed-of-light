#!/bin/bash
# Job 107, host e (offer 50680404, an EPYC 7663; replaces host a, which failed gate V0): jobs/107_newhosts@vast.sh (its header holds the gates and the predictions).
exec bash "$(cd "$(dirname "$0")" && pwd)/107_newhosts@vast.sh"
