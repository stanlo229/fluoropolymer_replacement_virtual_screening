#!/bin/bash
#SBATCH --job-name=scrape_catalogues
#SBATCH --account=aip-aspuru
#SBATCH --partition=gpubase_l40s_b2
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/scrape_%j.out
# CPU-only job (no --gres). Submit from dataset/:  sbatch submit_scrape.sh
#
# Catalogue scrape via PubChem for the 7 vendors, no MW limit, with dendron
# detection (dendrons.py). With --resume it reuses the saved SID lists and
# property checkpoints, and only fetches the CIDs that the 2026-04 MW<500 run
# discarded (saved as checkpoints/*_props_all.csv).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
[ -f scrape_catalogues.py ] || { echo "submit from dataset/: sbatch submit_scrape.sh"; exit 1; }
source ~/projects/aip-aspuru/stanlo/.virtualenvs/ocsr/bin/activate
mkdir -p logs checkpoints
echo "start $(date)  node $(hostname)"
python -u test_dendrons.py
python -u scrape_catalogues.py --output catalogues.csv --checkpoint_dir checkpoints --resume
python - <<'PY'
import pandas as pd
c = pd.read_csv("catalogues.csv", low_memory=False)
print(f"catalogue: {len(c):,} compounds, MW max {c.molecular_weight.max():.0f}, "
      f"> 500: {(c.molecular_weight >= 500).sum():,}, dendrons: {int(c.is_dendron.fillna(False).sum())}")
print(c[c.is_dendron.fillna(False).astype(bool)].groupby("dendron_family").size().to_string())
PY
echo "done $(date)"
