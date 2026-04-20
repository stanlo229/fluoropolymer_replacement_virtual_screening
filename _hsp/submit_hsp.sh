#!/bin/bash
#SBATCH --job-name=hsp_pipeline
#SBATCH --account=aip-aspuru          # adjust to your Alliance account
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/hsp.out
#SBATCH --error=logs/hsp.err

# -----------------------------------------------------------------------
# HSP prediction pipeline for ~80K+ norbornene monomers
# Estimated wall time: 15–30 min on 4 CPUs.  Adjust --time as needed.
# -----------------------------------------------------------------------
REPO_DIR="/home/stanlo/scratch/Repos/fluoropolymer_replacement_virtual_screening"
CODE_DIR="${REPO_DIR}/_hsp"

set -euo pipefail

# Load environment
module load StdEnv/2023 python/3.11 rdkit/2024.03

# Activate virtualenv — pandas/tqdm/matplotlib not in the rdkit module
# Create once with: python -m venv ~/envs/hsp && pip install pandas tqdm matplotlib
source ~/projects/aip-aspuru/stanlo/.virtualenvs/ocsr/bin/activate

cd $CODE_DIR   # run from _hsp/ directory

python run_pipeline.py \
    --catalogues ../dataset/catalogues.csv \
    --out_dir    ./results/ \
    --top_n      50

echo "======================================================"
echo "Step 5: Solvent incompatibility grids (water/diiodo/hexadecane/PTFE)"
echo "======================================================"
python "${CODE_DIR}/solvent_incompatibility.py" \
    --input      ./results/monomers_hsp_corrected.csv \
    --out_dir    ./results/ \
    --catalogues ../dataset/catalogues.csv \
    --top_n      50 \
    --ncols      5

echo "======================================================"
echo "Step 6: HSP component grids (max dispersion/polar/H-bonding)"
echo "======================================================"
python "${CODE_DIR}/make_hansen_grids.py"
