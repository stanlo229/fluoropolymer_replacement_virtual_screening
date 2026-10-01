#!/bin/bash
#SBATCH --job-name=hsp_v2
#SBATCH --account=aip-aspuru
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/hsp_v2_%j.out
# CPU-only job (no --gres): the cluster has only GPU partitions, as with the
# coatings_md_sim CPU runs.
#
# HSPiP-trained HSP pipeline, v2. Stages, in order:
#   cv        analysis/model_cv.py        scaffold-grouped CV of the models
#   err       analysis/expected_error.py  held-out error -> method AAE
#   predict   analysis/predict_monomers.py
#   finalize  analysis/finalize_hsp.py    -> results/monomers_hsp_final.csv, results/v2/
#   down      run_v2_downstream.sh        rankings + plots in results/v2/
#
# Start from a later stage when the earlier outputs are current:
#   sbatch --export=ALL,FROM=finalize jobs/submit_v2_rerun.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
[ -f hsp_calculator.py ] || { echo "submit from hsp/: sbatch jobs/submit_v2_rerun.sh"; exit 1; }
source ~/projects/aip-aspuru/stanlo/.virtualenvs/ocsr/bin/activate
FROM=${FROM:-cv}
L=results/benchmark/reference
mkdir -p logs results/v2
f() { grep --line-buffered -vi "warn\|reloaded\|=>\|single bond\|BondStereo\|^  [0-9]" || true; }
STAGES=(cv err predict finalize down)
run=0
for s in "${STAGES[@]}"; do
    [ "$s" = "$FROM" ] && run=1
    [ $run = 1 ] || { echo "skip $s"; continue; }
    echo "== $s ($(date +%T)) =="
    case $s in
        cv)       python -u analysis/model_cv.py 2>&1 | f > $L/model_cv_log.txt ;;
        err)      python -u analysis/expected_error.py 2>&1 | f > $L/expected_error_log.txt ;;
        predict)  python -u analysis/predict_monomers.py 2>&1 | f > $L/predict_v2_log.txt ;;
        finalize) python -u analysis/finalize_hsp.py 2>&1 | f > $L/finalize_v2_log.txt ;;
        down)     AAE=$(grep "use as --method_aae" $L/expected_error_log.txt | sed 's/.*: //')
                  echo "method AAE $AAE"
                  VER=v2 METHOD_AAE=$AAE jobs/run_downstream.sh 2>&1 | f > results/v2/downstream_log.txt ;;
    esac
done
echo "ALL DONE ($(date +%T))"
