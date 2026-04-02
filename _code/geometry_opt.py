#!/usr/bin/env python3
"""
3D geometry generation and MACE-OFF23 optimisation.

Workflow per molecule:
  1. SMILES → RDKit ETKDG initial 3D conformer (tries up to MAX_ATTEMPTS)
  2. Attach MACE-OFF23 calculator (medium model)
  3. LBFGS optimisation until fmax < FMAX_EV_ANG (eV/Å)
  4. Save optimised geometry as <mol_id>.xyz in the output directory

Usage:
    python geometry_opt.py --library library.csv --outdir geoms/ [--device cuda]
    python geometry_opt.py --library library.csv --outdir geoms/ --device cpu
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem

# ASE and MACE imports deferred to avoid slow load when not needed
# (imported inside optimise_molecule so partial imports work in testing)

FMAX_EV_ANG  = 0.05    # convergence threshold (eV/Å) — ~1 kcal/mol/Å
MAX_STEPS    = 500
MAX_ATTEMPTS = 5        # RDKit ETKDG retries
RANDOM_SEED  = 42

# vdW radii (Å) used for RDKit ETKDG — fall back to 2.0 for unknown elements
_VDW_RADII = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
              15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98}


def smiles_to_rdmol(smiles: str) -> Chem.Mol | None:
    """Parse SMILES and add Hs for 3D embedding."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    return mol


def embed_rdmol(mol: Chem.Mol) -> Chem.Mol | None:
    """Embed with ETKDG; retry with different seeds on failure."""
    params = AllChem.ETKDGv3()
    params.randomSeed = RANDOM_SEED
    for attempt in range(MAX_ATTEMPTS):
        params.randomSeed = RANDOM_SEED + attempt * 7
        result = AllChem.EmbedMolecule(mol, params)
        if result == 0:
            AllChem.MMFFOptimizeMolecule(mol)   # cheap pre-optimisation
            return mol
    return None


def rdmol_to_ase(mol: Chem.Mol):
    """Convert an RDKit Mol (with 3D coords) to an ASE Atoms object."""
    from ase import Atoms
    from rdkit.Chem import rdchem

    conf   = mol.GetConformer()
    pos    = conf.GetPositions()        # Å
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    return Atoms(symbols=symbols, positions=pos)


def optimise_molecule(smiles: str, mol_id: str, outdir: Path,
                      device: str = "cpu") -> dict:
    """
    Full pipeline: SMILES → optimised xyz.

    Returns a dict with keys: mol_id, converged, n_atoms, xyz_path, error.
    """
    result = {"mol_id": mol_id, "converged": False, "n_atoms": None,
              "xyz_path": None, "error": None}

    # --- 1. RDKit embedding -------------------------------------------------
    rdmol = smiles_to_rdmol(smiles)
    if rdmol is None:
        result["error"] = "RDKit parse failed"
        return result

    rdmol = embed_rdmol(rdmol)
    if rdmol is None:
        result["error"] = "ETKDG embedding failed"
        return result

    # --- 2. Convert to ASE --------------------------------------------------
    atoms = rdmol_to_ase(rdmol)
    result["n_atoms"] = len(atoms)

    # --- 3. Attach MACE-OFF23 calculator ------------------------------------
    try:
        from mace.calculators import mace_off
        calc = mace_off(model="medium", device=device, default_dtype="float64")
        atoms.calc = calc
    except Exception as exc:
        result["error"] = f"MACE load error: {exc}"
        return result

    # --- 4. LBFGS optimisation ----------------------------------------------
    try:
        from ase.optimize import LBFGS
        from ase.io import write as ase_write

        xyz_path = outdir / f"{mol_id}.xyz"
        opt = LBFGS(atoms, logfile=None)
        converged = opt.run(fmax=FMAX_EV_ANG, steps=MAX_STEPS)

        ase_write(str(xyz_path), atoms)
        result["converged"] = converged
        result["xyz_path"]  = str(xyz_path)

    except Exception as exc:
        result["error"] = f"Optimisation error: {exc}"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate and MACE-OFF23-optimise 3D geometries from a library CSV."
    )
    parser.add_argument("--library", default="library.csv")
    parser.add_argument("--outdir",  default="geoms",
                        help="Directory for .xyz output files")
    parser.add_argument("--device",  default="cpu", choices=["cpu", "cuda"],
                        help="Device for MACE-OFF23 (default: cpu)")
    parser.add_argument("--resume",  action="store_true",
                        help="Skip molecules whose .xyz already exists")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lib = pd.read_csv(args.library)
    print(f"Loaded {len(lib):,} molecules from {args.library}")

    results = []
    for _, row in tqdm(lib.iterrows(), total=len(lib),
                       desc="Optimising geometries", unit="mol"):
        mol_id = str(row["mol_id"])
        xyz_path = outdir / f"{mol_id}.xyz"

        if args.resume and xyz_path.exists():
            results.append({"mol_id": mol_id, "converged": True,
                             "xyz_path": str(xyz_path), "error": None,
                             "n_atoms": None})
            continue

        res = optimise_molecule(str(row["smiles"]), mol_id, outdir, args.device)
        results.append(res)

    df = pd.DataFrame(results)
    summary_path = outdir / "opt_summary.csv"
    df.to_csv(summary_path, index=False)

    n_ok  = df["converged"].sum()
    n_err = df["error"].notna().sum()
    print(f"\nConverged: {n_ok}/{len(df)}  |  Errors: {n_err}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
