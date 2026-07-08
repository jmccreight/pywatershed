#!/usr/bin/env bash
# Run 1: GSFLOW reference — chained spinup (2000) then analysis (2001).
# The spinup writes a restart IC that the analysis reads. Run from anywhere;
# the script cd's to its own directory so the controls' relative paths resolve.
set -euo pipefail

cd "$(dirname "$0")"

# GSFLOW 2.4.0 binary. Override for another platform with:
#   GSFLOW=/path/to/exe ./run.sh
GSFLOW="${GSFLOW:-../../bin/gsflow_2.4.0_ifort_apple_silicon_dbl_prec}"

echo "=== Spinup 2000 (writes restart IC) ==="
"$GSFLOW" spinup_2000.control

echo "=== Analysis 2001 (reads restart IC, iterates AET) ==="
"$GSFLOW" analysis_2001.control

echo "=== Done. See output_spinup/ and output_analysis/ ==="
