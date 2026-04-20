#!/usr/bin/env python3
"""
Post-filter catalogues.csv to remove:
  1. Compounds with any chiral centre
  2. Amines connected to an aromatic ring (anilines)
  3. Alcohols / phenols connected to an aromatic ring

Usage:
    python filter_catalogues.py                          # default I/O
    python filter_catalogues.py --input catalogues.csv --output filtered.csv
"""

import argparse

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

# ---------------------------------------------------------------------------
# SMARTS patterns
# ---------------------------------------------------------------------------

# OH directly on any aromatic carbon (phenol / naphthol / hetarene-OH …)
_PHENOL = Chem.MolFromSmarts("[OX2H][c]")

# Amine N directly on any aromatic carbon — same exclusions as scrape_catalogues.py
# (excludes amides, sulfonamides, cationic N, imine N)
_AROMATIC_AMINE = Chem.MolFromSmarts(
    "[NX3;!$(NC=O);!$(NS(=O));!$([N+]);!$(N=*)][c]"
)


def _has_chiral_centre(mol) -> bool:
    """True if the molecule has any assigned or unassigned stereocentre."""
    return bool(Chem.FindMolChiralCenters(mol, includeUnassigned=True))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Remove chiral, phenol, and aniline compounds from a catalogue CSV."
    )
    parser.add_argument("--input",  default="catalogues.csv", help="Input CSV (default: catalogues.csv)")
    parser.add_argument("--output", default="filtered.csv",   help="Output CSV (default: filtered.csv)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} compounds from {args.input}")

    keep = []
    removed = {"chiral": 0, "phenol": 0, "aniline": 0, "invalid_smiles": 0}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Filtering", unit="cpd"):
        smiles = row.get("canonical_smiles") or row.get("smiles", "")
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            removed["invalid_smiles"] += 1
            continue

        if _has_chiral_centre(mol):
            removed["chiral"] += 1
            continue

        if mol.HasSubstructMatch(_PHENOL):
            removed["phenol"] += 1
            continue

        if mol.HasSubstructMatch(_AROMATIC_AMINE):
            removed["aniline"] += 1
            continue

        keep.append(row)

    out = pd.DataFrame(keep)
    out.to_csv(args.output, index=False)

    n_removed = len(df) - len(out)
    pct = n_removed / len(df) * 100 if len(df) else 0
    print(f"\nRemoved {n_removed:,} / {len(df):,} compounds ({pct:.1f}%)")
    print(f"  chiral centres : {removed['chiral']:,}")
    print(f"  phenols        : {removed['phenol']:,}")
    print(f"  anilines       : {removed['aniline']:,}")
    print(f"  invalid SMILES : {removed['invalid_smiles']:,}")
    print(f"\nKept {len(out):,} compounds → {args.output}")


if __name__ == "__main__":
    main()
