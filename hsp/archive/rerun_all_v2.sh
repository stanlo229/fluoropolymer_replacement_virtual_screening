#!/bin/bash
# Full v2 recalculation after the hsp_calculator second-order fixes:
# CV -> expected error -> monomer predictions -> final table -> downstream.
set -euo pipefail
cd "$(dirname "$0")"
source ~/projects/aip-aspuru/stanlo/.virtualenvs/ocsr/bin/activate
L=results/hsp_reference_benchmark
f() { grep --line-buffered -vi "warn\|reloaded\|=>\|single bond\|BondStereo\|^  [0-9]"; }
python -u analysis/model_cv.py 2>&1 | f > $L/model_cv_log.txt
python -u analysis/expected_error.py 2>&1 | f | tee $L/expected_error_log.txt
AAE=$(grep "use as --method_aae" $L/expected_error_log.txt | sed 's/.*: //')
python -u analysis/predict_monomers.py 2>&1 | f > $L/predict_v2_log.txt
python -u analysis/finalize_hsp.py 2>&1 | f >> $L/predict_v2_log.txt
METHOD_AAE=$AAE ./run_v2_downstream.sh 2>&1 | f > results/v2/downstream_log.txt
echo "ALL DONE (method AAE $AAE)"
