#!/bin/bash
#SBATCH --job-name=ssip_pipeline
#SBATCH --account=aip-aspuru
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/pipeline_%j.out
#SBATCH --error=./logs/pipeline_%j.err

# ============================================================
# SSIP virtual screening pipeline — SLURM submission script
#
# Runs all 6 stages:
#   generate → geom → charges → surfaces → calibrate → footprint
#
# Usage:
#   sbatch submit_pipeline.sh               # full run
#   sbatch submit_pipeline.sh --resume      # resume interrupted run
#   WORKDIR=run_02 sbatch submit_pipeline.sh
# ============================================================

# ---- Configurable paths ------------------------------------------------
REPO_DIR="/home/stanlo/scratch/Repos/fluoropolymer_replacement_virtual_screening"
CODE_DIR="${REPO_DIR}/_code"
CATALOGUES="${REPO_DIR}/Dataset/catalogues.csv"
WORKDIR="${WORKDIR:-${REPO_DIR}/runs/run_01}"   # override with: WORKDIR=... sbatch ...
DEVICE="cuda"

# Pass any extra CLI args through (e.g. --resume, --skip generate)
EXTRA_ARGS="$@"

# ---- Environment -------------------------------------------------------
source ~/projects/aip-aspuru/stanlo/.virtualenvs/ocsr/bin/activate

# ---- Setup -------------------------------------------------------------
mkdir -p "${WORKDIR}" "$(dirname "${CODE_DIR}/logs/x")" "${REPO_DIR}/_code/logs"

echo "============================================================"
echo "SSIP pipeline started: $(date)"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Workdir:    ${WORKDIR}"
echo "Catalogues: ${CATALOGUES}"
echo "Extra args: ${EXTRA_ARGS}"
echo "============================================================"

# ---- Run ---------------------------------------------------------------
python "${CODE_DIR}/run_pipeline.py" \
    --catalogues    "${CATALOGUES}"  \
    --workdir       "${WORKDIR}"     \
    --device        "${DEVICE}"      \
    --resume                         \
    ${EXTRA_ARGS}

STATUS=$?

echo "============================================================"
echo "Pipeline finished: $(date)  (exit ${STATUS})"
if [ -f "${WORKDIR}/ssip_results.csv" ]; then
    echo "Results: $(wc -l < "${WORKDIR}/ssip_results.csv") rows in ${WORKDIR}/ssip_results.csv"
fi
echo "============================================================"

exit ${STATUS}
