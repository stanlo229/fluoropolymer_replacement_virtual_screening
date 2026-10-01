#!/bin/bash
#SBATCH --job-name=hsp_pipeline
#SBATCH --account=aip-aspuru
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/hsp_legacy_%j.out
# CPU-only job (no --gres). Submit from hsp/:  sbatch jobs/submit_hsp.sh
#
# Legacy pipeline: monomer library + published Stefanis-Panayiotou HSP.
#   run_pipeline.py      -> results/legacy_sp/monomers.csv, monomers_hsp.csv,
#                           monomers_hsp_corrected.csv, Hansen-space plots
#   rank_top_monomers.py -> results/legacy_sp/top25_ranked_*
# monomers_hsp.csv is also the input library for the v2 and v3 HSP methods.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
[ -f hsp_calculator.py ] || { echo "submit from hsp/: sbatch jobs/submit_hsp.sh"; exit 1; }
source ~/projects/aip-aspuru/stanlo/.virtualenvs/ocsr/bin/activate

python run_pipeline.py --catalogues ../dataset/catalogues.csv --out_dir results/legacy_sp --top_n 50
python rank_top_monomers.py --input results/legacy_sp/monomers_hsp_corrected.csv \
    --out_dir results/legacy_sp --catalogues ../dataset/catalogues.csv --top_n 25 --ncols 5
