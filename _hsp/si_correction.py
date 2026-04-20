"""
si_correction.py

Empirical correction for Si-containing monomers.

Strategy: for each reference Si compound, predict HSP using Si→C substitution,
compare to literature values, compute mean per-Si-atom residual vector,
then apply that vector to all has_si monomers.

References
----------
PDMS:
    Hansen, C.M. Hansen Solubility Parameters: A User's Handbook, 2nd ed.;
    CRC Press: Boca Raton, FL, 2007; Appendix A, Table A.1.
Hexamethyldisiloxane, Trimethylsilanol:
    Barton, A.F.M. Handbook of Solubility Parameters and Other Cohesion
    Parameters; CRC Press: Boca Raton, FL, 1983; Table 5-3.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from hsp_calculator import compute_hsp

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference compounds (SMILES, experimental HSP, n_Si_atoms)
# ---------------------------------------------------------------------------
SI_REFERENCES = [
    {
        "name":   "PDMS (polydimethylsiloxane repeat unit)",
        "smiles": "C[Si](C)(C)O[Si](C)(C)C",
        "delta_D_exp": 14.9,
        "delta_P_exp":  0.5,
        "delta_H_exp":  3.4,
        "source": "Hansen (2007), Appendix A, Table A.1",
    },
    {
        "name":   "Hexamethyldisiloxane",
        "smiles": "C[Si](C)(O[Si](C)(C)C)C",
        "delta_D_exp": 14.5,
        "delta_P_exp":  0.5,
        "delta_H_exp":  1.0,
        "source": "Barton (1983), Table 5-3",
    },
    {
        "name":   "Trimethylsilanol",
        "smiles": "C[Si](C)(C)O",
        "delta_D_exp": 14.0,
        "delta_P_exp":  2.0,
        "delta_H_exp":  4.8,
        "source": "Barton (1983), Table 5-3",
    },
]


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibrate_si_correction(
    save_path: str | Path | None = None,
) -> dict:
    """
    Compute per-Si-atom correction vector from reference compounds.

    Returns calibration dict with keys:
        delta_D_per_si, delta_P_per_si, delta_H_per_si,
        references (list with per-compound residuals),
        mean_abs_residual
    """
    corrections_D = []
    corrections_P = []
    corrections_H = []
    ref_results = []

    for ref in SI_REFERENCES:
        r = compute_hsp(ref["smiles"])
        if r.error or r.delta_D is None:
            log.warning("HSP prediction failed for %s: %s", ref["name"], r.error)
            continue

        n_si = r.n_si_atoms
        if n_si == 0:
            log.warning("No Si atoms detected in %s", ref["name"])
            continue

        res_D = (ref["delta_D_exp"] - r.delta_D) / n_si
        res_P = (ref["delta_P_exp"] - r.delta_P) / n_si
        res_H = (ref["delta_H_exp"] - r.delta_H) / n_si

        corrections_D.append(res_D)
        corrections_P.append(res_P)
        corrections_H.append(res_H)

        ref_results.append({
            "name":         ref["name"],
            "smiles":       ref["smiles"],
            "n_si":         n_si,
            "delta_D_exp":  ref["delta_D_exp"],
            "delta_P_exp":  ref["delta_P_exp"],
            "delta_H_exp":  ref["delta_H_exp"],
            "delta_D_pred": r.delta_D,
            "delta_P_pred": r.delta_P,
            "delta_H_pred": r.delta_H,
            "res_D_per_si": round(res_D, 4),
            "res_P_per_si": round(res_P, 4),
            "res_H_per_si": round(res_H, 4),
            "source":       ref["source"],
        })

    if not corrections_D:
        raise RuntimeError("Si calibration failed: no usable reference compounds")

    mean_D = float(np.mean(corrections_D))
    mean_P = float(np.mean(corrections_P))
    mean_H = float(np.mean(corrections_H))

    # Mean absolute residual after applying correction (should be ~0)
    mar = float(np.mean([
        abs(r["delta_D_exp"] - (r["delta_D_pred"] + mean_D * r["n_si"])) +
        abs(r["delta_P_exp"] - (r["delta_P_pred"] + mean_P * r["n_si"])) +
        abs(r["delta_H_exp"] - (r["delta_H_pred"] + mean_H * r["n_si"]))
        for r in ref_results
    ]) / 3)

    calibration = {
        "delta_D_per_si":    round(mean_D, 4),
        "delta_P_per_si":    round(mean_P, 4),
        "delta_H_per_si":    round(mean_H, 4),
        "mean_abs_residual": round(mar, 4),
        "references":        ref_results,
    }

    log.info(
        "Si correction: ΔδD/Si=%.3f  ΔδP/Si=%.3f  ΔδH/Si=%.3f  MAR=%.3f MPa^0.5",
        mean_D, mean_P, mean_H, mar,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(calibration, f, indent=2)
        log.info("Calibration saved to %s", save_path)

    return calibration


# ---------------------------------------------------------------------------
# Apply correction to DataFrame
# ---------------------------------------------------------------------------
def apply_si_correction(
    df: pd.DataFrame,
    calibration: dict,
) -> pd.DataFrame:
    """
    Add corrected HSP columns to df for rows where has_si=True.
    Non-Si rows get the same values as the uncorrected columns.

    New columns: delta_D_corr, delta_P_corr, delta_H_corr
    """
    df = df.copy()
    dD = calibration["delta_D_per_si"]
    dP = calibration["delta_P_per_si"]
    dH = calibration["delta_H_per_si"]

    n_si = df.get("n_si_atoms", pd.Series(0, index=df.index)).fillna(0).astype(float)
    si_mask = df.get("has_si", pd.Series(False, index=df.index)).fillna(False).astype(bool)

    df["delta_D_corr"] = np.where(si_mask, df["delta_D"] + dD * n_si, df["delta_D"])
    df["delta_P_corr"] = np.where(si_mask, df["delta_P"] + dP * n_si, df["delta_P"])
    df["delta_H_corr"] = np.where(si_mask, df["delta_H"] + dH * n_si, df["delta_H"])

    # Copy uncorrected values for non-Si rows
    df.loc[~si_mask, "delta_D_corr"] = df.loc[~si_mask, "delta_D"]
    df.loc[~si_mask, "delta_P_corr"] = df.loc[~si_mask, "delta_P"]
    df.loc[~si_mask, "delta_H_corr"] = df.loc[~si_mask, "delta_H"]

    n_corrected = si_mask.sum()
    log.info("Applied Si correction to %d monomers", n_corrected)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       default="results/monomers_hsp.csv")
    parser.add_argument("--output",      default="results/monomers_hsp_corrected.csv")
    parser.add_argument("--calibration", default="results/si_correction_calibration.json")
    args = parser.parse_args()

    cal = calibrate_si_correction(save_path=args.calibration)
    print("Calibration:")
    for ref in cal["references"]:
        print(f"  {ref['name']}: res_D={ref['res_D_per_si']:+.3f}  "
              f"res_P={ref['res_P_per_si']:+.3f}  res_H={ref['res_H_per_si']:+.3f}")
    print(f"  Mean: ΔδD={cal['delta_D_per_si']:+.3f}  "
          f"ΔδP={cal['delta_P_per_si']:+.3f}  ΔδH={cal['delta_H_per_si']:+.3f}")
    print(f"  MAR after correction: {cal['mean_abs_residual']:.3f} MPa^0.5")

    df = pd.read_csv(args.input)
    df_out = apply_si_correction(df, cal)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print(f"Written to {args.output}")
