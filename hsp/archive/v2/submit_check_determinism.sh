#!/bin/bash
#SBATCH --job-name=hsp_determinism
#SBATCH --account=aip-aspuru
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --output=logs/hsp_determinism_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
[ -f hsp_calculator.py ] || { echo "submit from hsp/: sbatch jobs/submit_check_determinism.sh"; exit 1; }
source ~/projects/aip-aspuru/stanlo/.virtualenvs/ocsr/bin/activate
python -u analysis/check_determinism.py 2>&1 | grep -v "^  [0-9]"
