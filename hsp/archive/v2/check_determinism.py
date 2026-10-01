"""
check_determinism.py

Train the final models twice from scratch (and once with a different n_jobs)
and compare predictions bitwise on 2,000 library monomers.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import predict_monomers as pm  # noqa: E402

smi = pd.read_csv(HERE.parent / "results" / "legacy_sp" / "monomers_hsp.csv",
                  usecols=["monomer_smiles"]).sample(2000, random_state=0).monomer_smiles.tolist()
runs = []
for n_jobs in (16, 16, 4):
    pm.et = lambda n=n_jobs: pm.ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2,
                                                    max_features=0.3, n_jobs=n, random_state=0)
    ref, fps, m_dir, m_res = pm.fit_models()
    runs.append(pm.predict(smi, ref, fps, m_dir, m_res)[["D", "P", "H", "nn_sim"]].to_numpy())
    print(f"run with n_jobs={n_jobs} done", flush=True)
print("run1 vs run2 (same settings) max |diff|:", np.abs(runs[0] - runs[1]).max())
print("run1 vs run3 (n_jobs=4)      max |diff|:", np.abs(runs[0] - runs[2]).max())
