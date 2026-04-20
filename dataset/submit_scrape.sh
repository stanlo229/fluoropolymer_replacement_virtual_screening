#!/bin/bash
#SBATCH --job-name=scrape_catalogues
#SBATCH --account=aip-aspuru
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=./logs/scrape.out
#SBATCH --error=./logs/scrape.err

# ---- Environment -------------------------------------------------------
source ~/projects/aip-aspuru/stanlo/.virtualenvs/ocsr/bin/activate

# ---- Paths -------------------------------------------------------------
SCRIPT_DIR="/home/stanlo/scratch/Repos/fluoropolymer_replacement_virtual_screening/dataset"
OUTPUT="${SCRIPT_DIR}/catalogues.csv"
CKPT_DIR="${SCRIPT_DIR}/checkpoints"

mkdir -p "${SCRIPT_DIR}/logs" "${CKPT_DIR}"

echo "======================================================"
echo "Catalogue scrape job started: $(date)"
echo "Node: $(hostname)"
echo "Output CSV: ${OUTPUT}"
echo "Checkpoint dir: ${CKPT_DIR}"
echo "======================================================"

# ---- Run ---------------------------------------------------------------
# Add --resume to pick up where a previous run left off if checkpoints exist
python "${SCRIPT_DIR}/scrape_catalogues.py" \
    --output   "${OUTPUT}"   \
    --checkpoint_dir "${CKPT_DIR}" \
    --resume

echo "======================================================"
echo "Job finished: $(date)"
echo "======================================================"
