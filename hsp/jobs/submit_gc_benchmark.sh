#!/bin/bash
#SBATCH --job-name=hsp_gc_bench
#SBATCH --account=aip-aspuru
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/hsp_gc_bench_%j.out
# CV benchmark: published S-P vs refitted group-contribution formulas vs XGBoost / ExtraTrees.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
[ -f hsp_calculator.py ] || { echo "submit from hsp/: sbatch jobs/submit_gc_benchmark.sh"; exit 1; }
source ~/projects/aip-aspuru/stanlo/.virtualenvs/ocsr/bin/activate
python -u analysis/gc_benchmark.py 2>&1 | grep -v "single bond\|BondStereo"
