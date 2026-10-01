"""
expected_error.py

Held-out error of the size-aware final model (ENS <= 30 heavy atoms,
ExtraTrees direct above), from model_cv.py's out-of-fold predictions, by
nearest-reference similarity bin. Tier 1 (curated) is the honest figure;
tier 2 (Y-MB estimates) is reported for the large-molecule regime only.

Writes expected_error_by_confidence.csv (read by finalize_hsp.py) and prints
the per-component MAE used as --method_aae for the rankings.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "results" / "benchmark" / "reference"
SIZE_SWITCH = 30


def main():
    p = pd.read_csv(OUT / "model_cv_predictions.csv", low_memory=False)
    p = p[p.n_si == 0].copy()
    large = (p.n_heavy > SIZE_SWITCH).to_numpy()
    for c in "DPH":
        ens = (p[f"ET_{c}"] + p[f"ET_resid_{c}"]) / 2
        p[f"F_{c}"] = np.where(large, p[f"ET_{c}"], ens)
    p[["F_P", "F_H"]] = p[["F_P", "F_H"]].clip(lower=0)
    e = {c: p[f"F_{c}"] - p[f"ref_{c}"] for c in "DPH"}
    p["eRa"] = np.sqrt(4 * e["D"] ** 2 + e["P"] ** 2 + e["H"] ** 2)
    p["eSP"] = np.sqrt(4 * (p.SP_D - p.ref_D) ** 2 + (p.SP_P - p.ref_P) ** 2 + (p.SP_H - p.ref_H) ** 2)
    p["confidence"] = pd.cut(p.nn_sim, [-1, .4, .6, 2], labels=["low", "medium", "high"])
    t = p.groupby(["tier", "confidence"], observed=True).agg(
        n=("eRa", "size"), med_Ra_err=("eRa", "median"),
        p90_Ra_err=("eRa", lambda x: x.quantile(.9)), SP_med_Ra_err=("eSP", "median")).round(2)
    t.to_csv(OUT / "expected_error_by_confidence.csv")
    print(t.to_string())
    t1 = p.tier == 1
    mae = [round(float(e[c][t1].abs().mean()), 2) for c in "DPH"]
    print("tier-1 MAE D,P,H (use as --method_aae):", ",".join(map(str, mae)))
    big = (p.tier == 2) & large
    print("tier-2 >30 heavy atoms MAE D,P,H:", [round(float(e[c][big].abs().mean()), 2) for c in "DPH"],
          "n =", int(big.sum()))


if __name__ == "__main__":
    main()
