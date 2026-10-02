#!/bin/bash
#SBATCH --job-name=hsp_library_rebuild
#SBATCH --account=aip-aspuru
#SBATCH --partition=gpubase_l40s_b2
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/hsp_library_rebuild_%j.out
# CPU-only job (no --gres). Submit from hsp/:  sbatch jobs/submit_library_rebuild.sh
#
# Rebuild everything downstream of dataset/catalogues.csv:
#   library   run_pipeline.py: monomers (incl. dendron focal-point attachment) +
#             published S-P HSP -> results/legacy_sp/, then the legacy rankings
#   v3        refitted-formula HSP, non-Si + Si rankings, Si analysis (apply, down, si)
#   explorer  viz/build_explorer.py -> results/v3/hsp_explorer.html (+ hosted page build)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
[ -f hsp_calculator.py ] || { echo "submit from hsp/: sbatch jobs/submit_library_rebuild.sh"; exit 1; }
source ~/projects/aip-aspuru/stanlo/.virtualenvs/ocsr/bin/activate
mkdir -p logs
echo "== library ($(date +%T)) =="
python -u run_pipeline.py --catalogues ../dataset/catalogues.csv --out_dir results/legacy_sp --top_n 50
python -u rank_top_monomers.py --input results/legacy_sp/monomers_hsp_corrected.csv \
    --out_dir results/legacy_sp --catalogues ../dataset/catalogues.csv --top_n 25 --ncols 5
echo "== v3 ($(date +%T)) =="
FROM=apply bash jobs/submit_v3_formula.sh
echo "== explorer ($(date +%T)) =="
python -u viz/build_explorer.py
echo "ALL DONE ($(date +%T))"
