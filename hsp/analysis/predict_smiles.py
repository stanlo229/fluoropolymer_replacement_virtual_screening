"""
predict_smiles.py

v3 HSP for arbitrary SMILES with the refitted size-intensive group-contribution
formula (fitted tables from gc_benchmark.py; no retraining, runs in seconds):

    python analysis/predict_smiles.py "SMILES1" "SMILES2" ... [--out file.csv]

Columns: D, P, H (MPa^0.5), V_formula (cm3/mol), the refitted 2012-form values
for comparison, Ra to the probe liquids / PTFE, and hsp_note for Si compounds,
whose values are not reliable.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import apply_gc_formula as ag  # noqa: E402
from solvent_incompatibility import REFERENCES  # noqa: E402


def run(smiles: list[str]) -> pd.DataFrame:
    space = pickle.load(open(ag.TAB / "group_space.pkl", "rb"))
    fed = pickle.load(open(ag.TAB / "Fedors_refit.pkl", "rb"))
    spr = pickle.load(open(ag.TAB / "SP_refit.pkl", "rb"))
    out = ag.score(smiles, space, fed, spr)
    for name, ref in REFERENCES.items():
        out[f"Ra_{name}"] = ag.ra(out.D, out.P, out.H, ref)
    out["hsp_note"] = np.where(out.monomer_smiles.str.contains("Si"), ag.SI_NOTE, "")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("smiles", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    df = run(a.smiles)
    pd.set_option("display.width", 250)
    print(df.drop(columns="hsp_note").round(2).to_string(index=False))
    for s, n in zip(df.monomer_smiles, df.hsp_note):
        if n:
            print(f"NOTE {s}: {n}")
    if a.out:
        df.to_csv(a.out, index=False)
