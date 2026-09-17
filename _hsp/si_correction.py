"""
si_correction.py

Empirical correction for Si-containing monomers.

Strategy: for each reference Si compound, predict HSP using Si→C substitution,
compare to literature values, compute mean per-Si-atom residual vector,
then apply that vector to all has_si monomers.

*** THIS CORRECTION IS NOT FIT FOR USE. DO NOT RE-ENABLE WITHOUT NEW DATA. ***

calibrate_si_correction() returns `usable: False`. Diagnosis, 2026-09-17:

  1. Only 2 unique reference compounds remain after removing a duplicate (see
     SI_REFERENCES). Three free parameters fitted to two molecules.

  2. MAR = 2.76 MPa^0.5 — the correction does not reproduce its own references.

  3. The delta_H coefficient is an artifact of the Si→C substitution, not a
     property of silicon. Trimethylsilanol Si→C becomes tert-butanol, whose
     predicted delta_H is 14.0 against an experimental 4.8 for the silanol, so
     that single compound contributes -9.2 per Si while the siloxane
     contributes -0.96. The "correction" is compensating for the substitution.

  4. Applicability domain: both references are siloxanes (Si-O-Si), but of 307
     Si monomers in the library only 4 sidechains are siloxanes; 122 are silyl
     ethers and 160 have no Si-O bond at all.

  5. Applied linearly per Si atom with no floor, it drives 290/307 Si monomers
     to a negative delta. Zeroing the correction still leaves 95/307 negative
     (vs 8.4% for non-Si), because the underlying Si→C substitution is itself
     unsound.

RESOLVED 2026-09-17 against the primary source. Table A.1 of Stefanis &
Panayiotou, Int. J. Pharm. 426 (2012) 29-43, Appendix A, was read in full:
it contains NO silicon group. The method covers C, H, O, N, S, F, Cl, Br, I
and nothing else. This is a property of the METHOD, not of this implementation,
so it cannot be fixed by adding table entries -- there are none to add. UNIFAC,
whose group definitions the method borrows, does define Si groups (main groups
42 and 43), but those are R/Q volume-surface parameters; Stefanis-Panayiotou
never fitted HSP contributions for them.

Consequently there are exactly two honest options for silicon:
  (a) exclude Si-containing compounds -- what the pipeline does today; or
  (b) source their HSP from outside this method entirely (HSPiP Y-MB, or
      measured values) and join them in, bypassing the group-contribution path.
A per-Si-atom offset on top of a Si->C substitution is neither, and is what
this module previously attempted.

References
----------
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

# Mean absolute residual (MPa^0.5) above which the fitted correction is declared
# unusable. A 3-parameter fit should reproduce its own reference compounds to
# well inside experimental scatter (~0.5); 1.0 is already generous.
MAR_THRESHOLD = 1.0

# ---------------------------------------------------------------------------
# Reference compounds (SMILES, experimental HSP, n_Si_atoms)
# ---------------------------------------------------------------------------
SI_REFERENCES = [
    # CORRECTED 2026-09-17. This list previously carried a fourth entry labelled
    # "PDMS (polydimethylsiloxane repeat unit)" with SMILES C[Si](C)(C)O[Si](C)(C)C
    # and values (14.9, 0.5, 3.4). That SMILES is NOT the PDMS repeat unit — the
    # PDMS repeat unit is -[Si(CH3)2-O]-, whereas Me3Si-O-SiMe3 is
    # hexamethyldisiloxane. It canonicalises identically to the entry below, so
    # the same molecule appeared twice under two names with two different
    # "experimental" targets, and the mean residual double-weighted it.
    # Verified by RDKit canonicalisation; both rows produced identical
    # predictions (14.0365, 7.6402, 2.9236), which is what exposed it.
    {
        "name":   "Hexamethyldisiloxane",
        "smiles": "C[Si](C)(C)O[Si](C)(C)C",
        "delta_D_exp": 14.5,
        "delta_P_exp":  0.5,
        "delta_H_exp":  1.0,
        "source": "Barton (1983), Table 5-3",
        # NOTE: hansen-solubility.com (Hansen's own site, solvent-cleaning table,
        # row "OS_10 Hexamethyldisiloxane [14.0, 1.0, 0.0, 100.6]" — 4th column is
        # the 100.6 degC boiling point) gives (14.0, 1.0, 0.0) for this compound.
        # The two sources disagree by ~0.5-1.0 in every component. Resolve against
        # Hansen (2007) Appendix A directly before relying on either.
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
    # ---- guard: no two references may be the same molecule ----------------
    # The original list contained hexamethyldisiloxane twice under two names
    # with two different experimental targets, silently double-weighting it.
    from rdkit import Chem

    seen: dict[str, str] = {}
    for ref in SI_REFERENCES:
        mol = Chem.MolFromSmiles(ref["smiles"])
        if mol is None:
            raise ValueError(f"SI_REFERENCES: unparseable SMILES for {ref['name']!r}")
        canon = Chem.MolToSmiles(mol)
        if canon in seen:
            raise ValueError(
                f"SI_REFERENCES contains the same molecule twice: "
                f"{seen[canon]!r} and {ref['name']!r} both canonicalise to {canon!r}. "
                f"Duplicate references double-weight one compound in the mean residual."
            )
        seen[canon] = ref["name"]

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

    # A 3-parameter correction fitted to this handful of compounds should
    # reproduce its own training set almost exactly. It does not: MAR ~2.6
    # MPa^0.5. Record that verdict in the artifact rather than only logging it,
    # so a downstream consumer can check `usable` instead of assuming.
    usable = mar <= MAR_THRESHOLD

    calibration = {
        "delta_D_per_si":    round(mean_D, 4),
        "delta_P_per_si":    round(mean_P, 4),
        "delta_H_per_si":    round(mean_H, 4),
        "mean_abs_residual": round(mar, 4),
        "mar_threshold":     MAR_THRESHOLD,
        "usable":            usable,
        "n_references":      len(ref_results),
        "applicability_domain": (
            "Both references are siloxanes (Si-O-Si). Of 307 Si-containing "
            "monomers in the screening library only 4 sidechains are siloxanes; "
            "122 are silyl ethers (Si-O-C) and 160 have no Si-O bond at all. "
            "The correction is extrapolating outside its domain for ~99% of them."
        ),
        "references":        ref_results,
    }

    if not usable:
        log.error(
            "Si correction FAILS its own references: MAR=%.3f > %.3f MPa^0.5 "
            "with only %d reference compound(s). Do NOT rely on delta_*_corr "
            "for Si-containing molecules.",
            mar, MAR_THRESHOLD, len(ref_results),
        )

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
