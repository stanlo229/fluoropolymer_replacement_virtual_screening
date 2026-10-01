#!/bin/bash
#SBATCH --job-name=hsp6_v2
#SBATCH --account=aip-aspuru
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=logs/hsp6_v2_%j.out
# Final-method HSP (v2) for the six requested monomers, plus the refractivity dD cross-check.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
[ -f hsp_calculator.py ] || { echo "submit from hsp/: sbatch jobs/submit_hsp6.sh"; exit 1; }
source ~/projects/aip-aspuru/stanlo/.virtualenvs/ocsr/bin/activate
python -u analysis/predict_smiles.py \
 "CC(CCC[C@H]([C@@]1([H])CC[C@]2([H])[C@]1(C)CC[C@@]3([H])[C@@]2([H])CC[C@]4([H])[C@]3(C)CC[C@H](OC(C5CC6C=CC5C6)=O)C4)C)C" \
 "O=C(C1C(C(OCC(CC)CCCC)=O)C2C=CC1C2)OCC(CC)CCCC" \
 "O=C(C1C(C(OCCO)=O)C2C=CC1C2)OCCO" \
 "O=C(C1C(C(OCCC#N)=O)C2C=CC1C2)OCCC#N" \
 "O=C(OC(C)(C)C)C1C2C=CC(C2)C1" \
 "O=C(OCCO)C1C2C=CC(C2)C1" \
 --out results/manual/2026-09-30/hsp6_v2.csv
python - <<'PY'
import sys; sys.path[:0]=['analysis','.']
import pandas as pd, numpy as np, finalize_hsp as fh
d=pd.read_csv('results/manual/2026-09-30/hsp6_v2.csv')
d['dD_refractivity']=fh.dd_refractivity(d.monomer_smiles.tolist())
d['Ra_PTFE_rd']=np.sqrt(4*(d.dD_refractivity-12.7)**2+d.P**2+d.H**2)
d.to_csv('results/manual/2026-09-30/hsp6_v2.csv',index=False)
print(d[['D','dD_refractivity','P','H','Ra_PTFE','Ra_PTFE_rd','model','confidence','expected_Ra_err','nn_name']].round(2).to_string())
PY
