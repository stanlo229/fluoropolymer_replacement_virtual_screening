"""
benchmark_sp.py

Score every S-P ring-counting variant against the HSPiP reference, stratified
by tier and ring class. Writes per-structure predictions and a summary table.

Domain: C/H/O/N/S/halogen compounds with >= 3 carbons, uncharged (the stated
applicability of Stefanis-Panayiotou 2008/2012). Si compounds are scored
separately with the calculator's Si->C substitution, since the method has no
Si group.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import sp_variants as sv  # noqa: E402

REF = HERE.parent / "reference"
OUT = HERE.parent / "results" / "benchmark" / "reference"

VARIANTS = [("per_ring", 1.0), ("per_system", 1.0), ("isolated_only", 1.0),
            ("discount", 0.25), ("discount", 0.5), ("discount", 0.75)]


def ra(dD, dP, dH):
    return np.sqrt(4 * dD ** 2 + dP ** 2 + dH ** 2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(REF / "hspip_reference.csv")
    ref = ref[(ref.n_carbon >= 3) & ~ref.charged]
    ref["domain"] = np.where(ref.n_si > 0, "si",
                             np.where(ref.sp_elements_only, "sp", "out"))
    ref = ref[ref.domain != "out"].reset_index(drop=True)

    rows = []
    for vname, w in VARIANTS:
        tag = vname if vname != "discount" else f"discount_{w}"
        for i, r in ref.iterrows():
            o = sv.compute(r.canonical_smiles, vname, w)
            rows.append({"ikey14": r.ikey14, "variant": tag, "D": o.D, "P": o.P, "H": o.H,
                         "P_normal": o.P_normal, "H_normal": o.H_normal,
                         "P_low": o.P_low_branch, "H_low": o.H_low_branch,
                         "n_ring_corr": o.n_ring_corr, "n_unavail": o.n_unavail})
    pred = pd.DataFrame(rows).merge(ref, on="ikey14")
    pred.to_csv(OUT / "sp_variant_predictions.csv", index=False)

    summ = []
    for floor in (False, True):
        p = pred.copy()
        if floor:
            p["P"], p["H"] = p.P.clip(lower=0), p.H.clip(lower=0)
        p["eD"], p["eP"], p["eH"] = p.D - p.ref_D, p.P - p.ref_P, p.H - p.ref_H
        p["eRa"] = ra(p.eD, p.eP, p.eH)
        p["neg"] = (p.P < 0) | (p.H < 0)
        p["cls"] = np.where(p.domain == "si", "Si (Si->C)", p.ring_class)
        g = p.groupby(["tier", "cls", "variant"])
        s = g.agg(n=("eD", "size"), MAE_D=("eD", lambda x: x.abs().mean()),
                  MAE_P=("eP", lambda x: x.abs().mean()), MAE_H=("eH", lambda x: x.abs().mean()),
                  bias_P=("eP", "mean"), bias_H=("eH", "mean"),
                  med_Ra_err=("eRa", "median"), frac_neg=("neg", "mean")).reset_index()
        s["floor"] = floor
        summ.append(s)
    summ = pd.concat(summ).round(3)
    summ.to_csv(OUT / "sp_variant_summary.csv", index=False)
    pd.set_option("display.width", 200)
    for floor in (False, True):
        print(f"\n===== floor={floor} =====")
        print(summ[summ.floor == floor].drop(columns="floor").to_string(index=False))


if __name__ == "__main__":
    main()
