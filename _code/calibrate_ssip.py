#!/usr/bin/env python3
"""
Calibrate the AIMNet2 Coulomb-MEPS → α/β conversion.

Background
----------
Hunter's equations (15) and (16) from Driver & Hunter PCCP 2020 were fitted
against B3LYP/6-31G* MEPS values.  Our pipeline uses AIMNet2 MBIS charges to
approximate the MEPS, which puts E_max and E_min on a different numerical scale.

This script runs a one-time calibration using ~25 small organic molecules with
well-known experimental H-bond parameters (α, β) from the Hunter group's own
training set (Calero et al. 2013 ESI; Abraham & Platts J. Org. Chem. 2001).

Calibration fits:
  α = a0 * E_max²  +  a1 * E_max          (positive SSIP donor)
  β = c * (b0 * E_min²  +  b1 * E_min)    (negative SSIP acceptor, c=1 global fit;
                                             functional-group c factors from Table 2
                                             applied in ssip_footprint.py)

Outputs `calibration.json` with keys: a0, a1, b0, b1 (and residuals for QC).

Usage
-----
    # Build surfaces for calibration compounds first:
    python calibrate_ssip.py --geomdir geoms_calib/ --chargedir charges_calib/ \
                             --surfdir surfaces_calib/ --output calibration.json

    # Or skip recomputation if surfaces already exist:
    python calibrate_ssip.py --surfdir surfaces_calib/ --output calibration.json

    # Just print fit quality against stored calibration:
    python calibrate_ssip.py --check --calibration calibration.json \
                             --surfdir surfaces_calib/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Experimental H-bond parameters (α, β) for calibration compounds
# Source: Abraham & Platts J. Org. Chem. 2001 66, 3484 + Calero et al. 2013 ESI
# All values are the Hunter-scale α and β (not αH₂ / βH₂).
# ---------------------------------------------------------------------------

CALIBRATION_SET = [
    # (name, SMILES, alpha, beta)
    # --- Donors (α > 0) ---
    ("methanol",         "CO",                       2.06, 2.24),
    ("ethanol",          "CCO",                      2.06, 2.24),
    ("1-propanol",       "CCCO",                     2.06, 2.24),
    ("water",            "O",                        2.81, 4.50),
    ("phenol",           "Oc1ccccc1",                2.79, 1.83),  # included for calibration only
    ("chloroform",       "ClC(Cl)Cl",                1.73, 0.00),
    ("acetic_acid",      "CC(=O)O",                  3.35, 3.09),
    # --- Acceptors (β > 0, weak/no donor) ---
    ("acetone",          "CC(=O)C",                  0.00, 3.43),
    ("diethyl_ether",    "CCOCC",                    0.00, 3.40),
    ("ethyl_acetate",    "CCOC(=O)C",                0.00, 3.09),
    ("acetonitrile",     "CC#N",                     0.00, 2.73),
    ("pyridine",         "c1ccncc1",                 0.00, 4.51),
    ("dmf",              "CN(C)C=O",                 0.00, 4.39),
    ("dmso",             "CS(=O)C",                  0.00, 5.71),
    ("thf",              "C1CCOC1",                  0.00, 3.47),
    ("trimethylamine",   "CN(C)C",                   0.00, 4.52),
    ("diethylamine",     "CCNCC",                    1.30, 4.40),
    ("n_methylformamide","CNC=O",                    2.39, 4.26),
    ("nitromethane",     "C[N+](=O)[O-]",            0.00, 2.38),
    ("piperidine",       "C1CCNCC1",                 1.32, 4.30),
    # --- Near-zero polar (non-polar reference) ---
    ("n_pentane",        "CCCCC",                    0.00, 0.00),
    ("cyclohexane",      "C1CCCCC1",                 0.00, 0.00),
    ("benzene",          "c1ccccc1",                 0.00, 1.13),
]

# Functional-group scaling factors c for β (Hunter Table 2, Driver & Hunter 2020)
# atom_assignment → c value
C_FACTORS = {
    "nitrile_N":     0.77,
    "primary_N":     1.00,
    "secondary_N":   1.25,
    "tertiary_N":    1.41,
    "pyridine_N":    1.08,
    "ether_O":       1.16,
    "alcohol_O":     0.95,
    "aldehyde_C=O":  0.89,
    "ester_C=O":     0.96,
    "carbonate_C=O": 0.96,
    "nitro_O":       0.80,
    "sulfoxide_O":   0.99,
    "phosphate_O":   1.09,
    "default":       1.00,
}


# ---------------------------------------------------------------------------
# Fitting functions (Hunter quadratic forms, eqs 15 & 16)
# ---------------------------------------------------------------------------

def _alpha_model(E_max, a0, a1):
    """α = a0·E² + a1·E  (passes through origin; α=0 when E_max=0)"""
    return a0 * E_max ** 2 + a1 * E_max


def _beta_model(E_min, b0, b1):
    """β/c = b0·E² + b1·E  (E_min < 0; β > 0)"""
    return b0 * E_min ** 2 + b1 * E_min


def fit_alpha(E_max_vals: np.ndarray, alpha_vals: np.ndarray) -> tuple:
    """Fit α = a0*E_max² + a1*E_max. Returns (a0, a1, rmsd)."""
    # Only use compounds with α > 0 for the donor fit
    mask = alpha_vals > 0
    if mask.sum() < 3:
        raise ValueError("Need ≥3 donor compounds for α calibration")
    popt, _ = curve_fit(_alpha_model, E_max_vals[mask], alpha_vals[mask],
                        p0=[1e-5, 1e-2], maxfev=10_000)
    resid = alpha_vals[mask] - _alpha_model(E_max_vals[mask], *popt)
    return popt[0], popt[1], float(np.sqrt((resid**2).mean()))


def fit_beta(E_min_vals: np.ndarray, beta_vals: np.ndarray) -> tuple:
    """Fit β = b0*E_min² + b1*E_min. Returns (b0, b1, rmsd)."""
    mask = beta_vals > 0
    if mask.sum() < 3:
        raise ValueError("Need ≥3 acceptor compounds for β calibration")
    popt, _ = curve_fit(_beta_model, E_min_vals[mask], beta_vals[mask],
                        p0=[1e-4, -1e-2], maxfev=10_000)
    resid = beta_vals[mask] - _beta_model(E_min_vals[mask], *popt)
    return popt[0], popt[1], float(np.sqrt((resid**2).mean()))


# ---------------------------------------------------------------------------
# Helpers to run geometry / charge / surface pipeline on calibration set
# ---------------------------------------------------------------------------

def _run_pipeline_for_calib(calib_dir: Path, device: str):
    """
    Run geometry_opt → charge_predict → meps_surface for the calibration set.
    Writes results into <calib_dir>/geoms/, charges/, surfaces/.
    """
    import pandas as pd
    import geometry_opt as go
    import charge_predict as cp
    import meps_surface as ms

    geom_dir    = calib_dir / "geoms"
    charge_dir  = calib_dir / "charges"
    surface_dir = calib_dir / "surfaces"
    for d in (geom_dir, charge_dir, surface_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("Running pipeline on calibration set …")

    for name, smiles, _, _ in CALIBRATION_SET:
        print(f"  {name} … ", end="", flush=True)
        xyz_path = geom_dir / f"{name}.xyz"
        if not xyz_path.exists():
            res = go.optimise_molecule(smiles, name, geom_dir, device)
            if res["error"]:
                print(f"GEOM ERROR: {res['error']}")
                continue

        npz_path = charge_dir / f"{name}.npz"
        if not npz_path.exists():
            parsed = cp.load_xyz(xyz_path)
            if parsed is None:
                print("XYZ PARSE ERROR")
                continue
            numbers, positions = parsed
            model, route = cp.load_aimnet2(device)
            try:
                charges = cp.predict_charges(model, route, numbers, positions, 0)
            except Exception as exc:
                print(f"CHARGE ERROR: {exc}")
                continue
            np.savez(npz_path, numbers=numbers, positions=positions,
                     charges=charges, mol_charge=np.array(0))

        surf_path = surface_dir / f"{name}.npz"
        if not surf_path.exists():
            result = ms.compute_meps(npz_path)
            if "error" in result:
                print(f"SURFACE ERROR: {result['error']}")
                continue
            np.savez_compressed(surf_path, **{k: result[k] for k in result})

        print("done")


# ---------------------------------------------------------------------------
# Main calibration routine
# ---------------------------------------------------------------------------

def load_surface_emax_emin(surface_dir: Path) -> dict[str, tuple[float, float]]:
    """Return {name: (E_max, E_min)} for all calibration compounds with surfaces."""
    out = {}
    for name, *_ in CALIBRATION_SET:
        p = surface_dir / f"{name}.npz"
        if p.exists():
            d = np.load(p)
            out[name] = (float(d["E_max"]), float(d["E_min"]))
    return out


def run_calibration(surface_dir: Path, output_path: Path):
    emax_emin = load_surface_emax_emin(surface_dir)
    found = set(emax_emin)
    expected = {n for n, *_ in CALIBRATION_SET}
    missing = expected - found
    if missing:
        print(f"WARNING: missing surface data for: {sorted(missing)}", file=sys.stderr)

    names    = [n for n, *_ in CALIBRATION_SET if n in found]
    E_max_v  = np.array([emax_emin[n][0] for n in names])
    E_min_v  = np.array([emax_emin[n][1] for n in names])
    alpha_v  = np.array([next(a for nn, _, a, _ in CALIBRATION_SET if nn == n)
                         for n in names])
    beta_v   = np.array([next(b for nn, _, _, b in CALIBRATION_SET if nn == n)
                         for n in names])

    a0, a1, rmsd_alpha = fit_alpha(E_max_v, alpha_v)
    b0, b1, rmsd_beta  = fit_beta(E_min_v, beta_v)

    calib = {
        "a0": a0, "a1": a1, "rmsd_alpha": rmsd_alpha,
        "b0": b0, "b1": b1, "rmsd_beta": rmsd_beta,
        "n_alpha_compounds": int((alpha_v > 0).sum()),
        "n_beta_compounds":  int((beta_v  > 0).sum()),
        "c_factors": C_FACTORS,
    }

    output_path.write_text(json.dumps(calib, indent=2))
    print(f"\nCalibration saved to {output_path}")
    print(f"  α fit:  a0={a0:.4e}  a1={a1:.4e}  RMSD={rmsd_alpha:.3f}")
    print(f"  β fit:  b0={b0:.4e}  b1={b1:.4e}  RMSD={rmsd_beta:.3f}")
    return calib


def apply_calibration(E_max: float, E_min: float, calib: dict,
                      c_factor: float = 1.0) -> tuple[float, float]:
    """
    Convert E_max, E_min (kJ/mol) → α, β using stored calibration coefficients.
    c_factor: functional-group scaling factor for β (from C_FACTORS dict).
    """
    a0, a1 = calib["a0"], calib["a1"]
    b0, b1 = calib["b0"], calib["b1"]
    alpha = max(0.0, a0 * E_max ** 2 + a1 * E_max)
    beta  = max(0.0, c_factor * (b0 * E_min ** 2 + b1 * E_min))
    return alpha, beta


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate AIMNet2 Coulomb-MEPS → α/β for SSIP calculation."
    )
    parser.add_argument("--surfdir",     default="surfaces_calib",
                        help="Directory with calibration compound surface .npz files")
    parser.add_argument("--output",      default="calibration.json")
    parser.add_argument("--run_pipeline", action="store_true",
                        help="Run geom opt + charges + surfaces for calibration set first")
    parser.add_argument("--calib_dir",   default="calib_workdir",
                        help="Working directory when --run_pipeline is set")
    parser.add_argument("--device",      default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--check",       action="store_true",
                        help="Print calibration quality against stored calibration.json")
    parser.add_argument("--calibration", default="calibration.json",
                        help="Stored calibration file for --check mode")
    args = parser.parse_args()

    if args.run_pipeline:
        # Add _code/ to path so geometry_opt etc. are importable
        sys.path.insert(0, str(Path(__file__).parent))
        _run_pipeline_for_calib(Path(args.calib_dir), args.device)

    if args.check:
        calib = json.loads(Path(args.calibration).read_text())
        print(f"Loaded calibration from {args.calibration}")
        print(f"  a0={calib['a0']:.4e}  a1={calib['a1']:.4e}  RMSD_α={calib['rmsd_alpha']:.3f}")
        print(f"  b0={calib['b0']:.4e}  b1={calib['b1']:.4e}  RMSD_β={calib['rmsd_beta']:.3f}")
        return

    run_calibration(Path(args.surfdir), Path(args.output))


if __name__ == "__main__":
    main()
