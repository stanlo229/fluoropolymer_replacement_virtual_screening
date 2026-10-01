"""
predict_smiles.py

Final-method HSP for arbitrary SMILES (ENS for organics, si_gc for Si), with
the floored S-P value and the nearest HSPiP structure alongside.

    python analysis/predict_smiles.py "SMILES1" "SMILES2" ... [--out file.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import predict_monomers as pm  # noqa: E402
import si_gc  # noqa: E402
from finalize_hsp import expected_error  # noqa: E402
from solvent_incompatibility import REFERENCES  # noqa: E402



def run(smiles: list[str]) -> pd.DataFrame:
    ref, ref_fps, m_dir, m_res = pm.fit_models()
    out = pm.predict(smiles, ref, ref_fps, m_dir, m_res)
    out["expected_Ra_err"] = out.confidence.astype(str).map(expected_error())
    si = np.array(["Si" in s for s in smiles])
    if si.any():
        out.loc[si, ["D", "P", "H"]] = si_gc.predict(si_gc.fit(), [s for s, f in zip(smiles, si) if f])
        out.loc[si, "hsp_source"] = "si_gc"
        out.loc[si, "expected_Ra_err"] = [7.0 if si_gc.has_silanol(s) else 2.0
                                          for s, f in zip(smiles, si) if f]
    for name, (rD, rP, rH) in REFERENCES.items():
        out[f"Ra_{name}"] = np.sqrt(4 * (out.D - rD) ** 2 + (out.P - rP) ** 2 + (out.H - rH) ** 2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("smiles", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    df = run(a.smiles)
    pd.set_option("display.width", 250)
    print(df.round(2).to_string(index=False))
    if a.out:
        df.to_csv(a.out, index=False)
