#!/usr/bin/env python3
"""
Generate a SMILES library of norbornene ester and amide products.

Scaffolds:
  • Monofunctional: 5-norbornene-2-carboxylic acid (C2, endo and exo).
      monoalcohol  R-OH  → norbornene ester  (R-OC(=O)-scaffold)
      monoamine    R-NH2 → norbornene amide  (R-NHC(=O)-scaffold)
  • Difunctional:  5-norbornene-2,3-dicarboxylic acid (C2+C3, endo and exo).
      monoalcohol  R-OH  → symmetric diester (both COOHs esterified)
      monoamine    R-NH2 → symmetric diamide (both COOHs amidated)

Reference molecules (compound_type="reference", always appended unless --no_reference):
  • Surface-energy reference solvents: water, n-hexadecane, diiodomethane
  • PFAS targets for SSIP similarity comparison:
      PFOA, PFOS, perfluorooctane, HFPO-DA (GenX), PFBS

Output: library.csv  (smiles, mol_id, scaffold_isomer, sidechain_id,
                       sidechain_smiles, compound_type, source, link,
                       molecular_weight_sidechain, price_per_gram)

Usage:
    python generate_library.py --catalogues ../Dataset/catalogues.csv \
                               --output library.csv
    python generate_library.py --catalogues ../Dataset/catalogues.csv \
                               --isomers endo exo \
                               --types ester amide diester diamide
    python generate_library.py --catalogues ../Dataset/catalogues.csv \
                               --no_reference
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Norbornene scaffold definitions
# ---------------------------------------------------------------------------

# 5-norbornene-2-carboxylic acid SMILES.
# The carboxyl OH is the attachment point for esterification / amidation.
# endo (C2-exo carboxyl relative to double bond bridge) and exo isomers
# are represented by different SMILES; RDKit will handle stereochemistry.
#
# We represent the acyl fragment (C(=O) attached to C2) with a dummy [*]
# placeholder for the O (ester) or N (amide) of the sidechain.

SCAFFOLDS = {
    # Acyl fragment: C2-endo acid → COC(=O)[C@@H]1C[C@@H]2C=C[C@H]1C2  (endo)
    # We use reaction SMARTS to attach sidechains cleanly.
    "endo": {
        "acid_smiles": "OC(=O)[C@@H]1C[C@@H]2C=C[C@H]1C2",
        "acyl_smarts": "[C:1](=O)([OH])[c,C]",   # carboxyl C
    },
    "exo": {
        "acid_smiles": "OC(=O)[C@H]1C[C@@H]2C=C[C@@H]1C2",
        "acyl_smarts": "[C:1](=O)([OH])[c,C]",
    },
}

# ---------------------------------------------------------------------------
# Difunctional norbornene scaffold: 5-norbornene-2,3-dicarboxylic acid
# (cis-nadic acid).  Both COOH groups react with the same sidechain to give
# a symmetric diester or diamide.
#
# endo: carboxyl groups on the endo face (same side as C7 methylene bridge);
#       this is the commercially dominant "nadic acid" isomer.
# exo:  carboxyl groups on the exo face.
# ---------------------------------------------------------------------------

SCAFFOLDS_DI = {
    "endo": {
        "acid_smiles": "OC(=O)[C@@H]1[C@H](C(=O)O)[C@H]2C=C[C@@H]1C2",
    },
    "exo": {
        "acid_smiles": "OC(=O)[C@H]1[C@@H](C(=O)O)[C@@H]2C=C[C@H]1C2",
    },
}

# Reaction SMARTS for esterification: acid + alcohol → ester + water
_ESTER_RXN = AllChem.ReactionFromSmarts(
    "[C:1](=O)[OH].[O:2][H]>>[C:1](=O)[O:2]"
)

# Reaction SMARTS for amidation: acid + primary amine → amide + water
_AMIDE_RXN = AllChem.ReactionFromSmarts(
    "[C:1](=O)[OH].[N:2][H]>>[C:1](=O)[N:2]"
)

# ---------------------------------------------------------------------------
# Reference molecules: surface-energy probes + PFAS comparison targets
#
# Surface-energy probes (Owens-Wendt method):
#   water, n-hexadecane, diiodomethane
# PFAS (representative spread of chain length, head group, and regulation status):
#   PFOA  — C8 carboxylic acid; most-regulated long-chain carboxylate
#   PFOS  — C8 sulfonic acid;   most-regulated long-chain sulfonate
#   Perfluorooctane — C8 non-polar fluorocarbon (reference for dispersive component)
#   HFPO-DA (GenX)  — C5 ether-carboxylic acid; key short-chain replacement PFAS
#   PFBS  — C4 sulfonic acid;   short-chain regulated sulfonate
# ---------------------------------------------------------------------------

REFERENCE_MOLECULES = [
    # (mol_id,            smiles,                                         note)
    ("ref_water",         "O",
     "Water — surface-energy probe (polar/H-bond reference)"),
    ("ref_hexadecane",    "CCCCCCCCCCCCCCCC",
     "n-Hexadecane — surface-energy probe (apolar dispersive reference)"),
    ("ref_diiodomethane", "ICI",
     "Diiodomethane — surface-energy probe (high dispersive, low polar)"),
    ("ref_pfoa",          "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
     "PFOA — perfluorooctanoic acid (C8 carboxylate PFAS)"),
    ("ref_pfos",          "OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
     "PFOS — perfluorooctanesulfonic acid (C8 sulfonate PFAS)"),
    ("ref_perfluorooctane", "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
     "Perfluorooctane — non-polar C8 fluorocarbon"),
    ("ref_hfpo_da",       "OC(=O)C(F)(C(F)(F)F)OC(F)(F)C(F)(F)F",
     "HFPO-DA (GenX) — perfluoro-2-(heptafluoropropoxy)propanoic acid (C5 ether-PFAS)"),
    ("ref_pfbs",          "OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
     "PFBS — perfluorobutanesulfonic acid (C4 short-chain sulfonate PFAS)"),
]


def _reference_rows() -> list[dict]:
    """Return library-format rows for all reference molecules."""
    rows = []
    for mol_id, smiles, _note in REFERENCE_MOLECULES:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"WARNING: could not parse reference SMILES for {mol_id}", file=sys.stderr)
            continue
        rows.append({
            "smiles":                    Chem.MolToSmiles(mol),
            "mol_id":                    mol_id,
            "scaffold_isomer":           "reference",
            "sidechain_id":              "",
            "sidechain_smiles":          "",
            "compound_type":             "reference",
            "source":                    "",
            "link":                      "",
            "molecular_weight_sidechain": "",
            "price_per_gram":            "",
        })
    return rows


# ---------------------------------------------------------------------------
# SMARTS for identifying the reactive group in sidechains
# ---------------------------------------------------------------------------

_ALCOHOL_SMARTS = Chem.MolFromSmarts("[OX2H;!$(OC=O)]")   # aliphatic/aromatic OH
_AMINE_SMARTS   = Chem.MolFromSmarts(
    "[NX3H2;!$(NC=O);!$(NS(=O));!$([N+]);!$(N=*)]"        # primary amine only
)


def _react(acid_mol, sidechain_mol, rxn) -> Chem.Mol | None:
    """Run a reaction and return the first valid product, or None."""
    try:
        products = rxn.RunReactants((acid_mol, sidechain_mol))
    except Exception:
        return None
    for prod_tuple in products:
        for prod in prod_tuple:
            try:
                Chem.SanitizeMol(prod)
                return prod
            except Exception:
                continue
    return None


def _react_both(diacid_mol: Chem.Mol, sidechain_mol: Chem.Mol, rxn) -> Chem.Mol | None:
    """
    React a diacid scaffold twice with the same sidechain (symmetric product).
    First run reacts one COOH; second run reacts the remaining COOH.
    Returns the fully substituted product, or None if either step fails.
    """
    intermediate = _react(diacid_mol, sidechain_mol, rxn)
    if intermediate is None:
        return None
    return _react(intermediate, sidechain_mol, rxn)


def build_library(
    catalogues_path: str,
    isomers: list[str],
    compound_types: list[str],
    include_reference: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(catalogues_path)
    rows = []

    for isomer in isomers:
        acid_smiles = SCAFFOLDS[isomer]["acid_smiles"]
        acid_mol    = Chem.MolFromSmiles(acid_smiles)
        if acid_mol is None:
            print(f"ERROR: could not parse scaffold SMILES for {isomer}", file=sys.stderr)
            continue

        for _, sc in tqdm(df.iterrows(), total=len(df),
                          desc=f"  Building {isomer} library", unit="cpd"):
            sc_smiles = sc.get("canonical_smiles") or sc.get("smiles", "")
            sc_mol    = Chem.MolFromSmiles(str(sc_smiles))
            if sc_mol is None:
                continue

            sc_types = str(sc.get("compound_type", "")).split(",")

            if "ester" in compound_types and any(
                t.strip() in ("monoalcohol", "diol") for t in sc_types
            ):
                if sc_mol.HasSubstructMatch(_ALCOHOL_SMARTS):
                    product = _react(acid_mol, sc_mol, _ESTER_RXN)
                    if product is not None:
                        rows.append({
                            "smiles":                    Chem.MolToSmiles(product),
                            "mol_id":                    f"{isomer}_ester_{sc.name}",
                            "scaffold_isomer":           isomer,
                            "sidechain_id":              sc.name,
                            "sidechain_smiles":          sc_smiles,
                            "compound_type":             "ester",
                            "source":                    sc.get("source", ""),
                            "link":                      sc.get("link", ""),
                            "molecular_weight_sidechain": sc.get("molecular_weight", ""),
                            "price_per_gram":            sc.get("price_per_gram", ""),
                        })

            if "amide" in compound_types and "monoamine" in [t.strip() for t in sc_types]:
                if sc_mol.HasSubstructMatch(_AMINE_SMARTS):
                    product = _react(acid_mol, sc_mol, _AMIDE_RXN)
                    if product is not None:
                        rows.append({
                            "smiles":                    Chem.MolToSmiles(product),
                            "mol_id":                    f"{isomer}_amide_{sc.name}",
                            "scaffold_isomer":           isomer,
                            "sidechain_id":              sc.name,
                            "sidechain_smiles":          sc_smiles,
                            "compound_type":             "amide",
                            "source":                    sc.get("source", ""),
                            "link":                      sc.get("link", ""),
                            "molecular_weight_sidechain": sc.get("molecular_weight", ""),
                            "price_per_gram":            sc.get("price_per_gram", ""),
                        })

    # --- Difunctional scaffolds (diester / diamide) ---------------------------
    di_types = [t for t in compound_types if t in ("diester", "diamide")]
    if di_types:
        for isomer in isomers:
            diacid_smiles = SCAFFOLDS_DI[isomer]["acid_smiles"]
            diacid_mol    = Chem.MolFromSmiles(diacid_smiles)
            if diacid_mol is None:
                print(f"ERROR: could not parse difunctional scaffold SMILES for {isomer}",
                      file=sys.stderr)
                continue

            for _, sc in tqdm(df.iterrows(), total=len(df),
                              desc=f"  Building {isomer} difunctional library", unit="cpd"):
                sc_smiles = sc.get("canonical_smiles") or sc.get("smiles", "")
                sc_mol    = Chem.MolFromSmiles(str(sc_smiles))
                if sc_mol is None:
                    continue

                sc_types = str(sc.get("compound_type", "")).split(",")

                if "diester" in di_types and any(
                    t.strip() in ("monoalcohol", "diol") for t in sc_types
                ):
                    if sc_mol.HasSubstructMatch(_ALCOHOL_SMARTS):
                        product = _react_both(diacid_mol, sc_mol, _ESTER_RXN)
                        if product is not None:
                            rows.append({
                                "smiles":                    Chem.MolToSmiles(product),
                                "mol_id":                    f"{isomer}_diester_{sc.name}",
                                "scaffold_isomer":           f"{isomer}_di",
                                "sidechain_id":              sc.name,
                                "sidechain_smiles":          sc_smiles,
                                "compound_type":             "diester",
                                "source":                    sc.get("source", ""),
                                "link":                      sc.get("link", ""),
                                "molecular_weight_sidechain": sc.get("molecular_weight", ""),
                                "price_per_gram":            sc.get("price_per_gram", ""),
                            })

                if "diamide" in di_types and "monoamine" in [t.strip() for t in sc_types]:
                    if sc_mol.HasSubstructMatch(_AMINE_SMARTS):
                        product = _react_both(diacid_mol, sc_mol, _AMIDE_RXN)
                        if product is not None:
                            rows.append({
                                "smiles":                    Chem.MolToSmiles(product),
                                "mol_id":                    f"{isomer}_diamide_{sc.name}",
                                "scaffold_isomer":           f"{isomer}_di",
                                "sidechain_id":              sc.name,
                                "sidechain_smiles":          sc_smiles,
                                "compound_type":             "diamide",
                                "source":                    sc.get("source", ""),
                                "link":                      sc.get("link", ""),
                                "molecular_weight_sidechain": sc.get("molecular_weight", ""),
                                "price_per_gram":            sc.get("price_per_gram", ""),
                            })

    # --- Reference molecules --------------------------------------------------
    if include_reference:
        rows.extend(_reference_rows())

    lib = pd.DataFrame(rows)

    # Deduplicate on canonical SMILES (keeps first occurrence)
    if not lib.empty:
        lib = lib.drop_duplicates(subset="smiles", keep="first").reset_index(drop=True)
        print(f"\nLibrary: {len(lib):,} unique products "
              f"({lib['compound_type'].value_counts().to_dict()})")

    return lib


def main():
    parser = argparse.ArgumentParser(
        description="Generate norbornene ester/amide library from catalogue sidechains."
    )
    parser.add_argument("--catalogues", default="../Dataset/catalogues.csv",
                        help="Path to filtered catalogues CSV")
    parser.add_argument("--output", default="library.csv",
                        help="Output library CSV path")
    parser.add_argument("--isomers", nargs="+", default=["endo", "exo"],
                        choices=["endo", "exo"],
                        help="Scaffold isomers to generate (default: both)")
    parser.add_argument("--types", nargs="+", default=["ester", "amide", "diester", "diamide"],
                        choices=["ester", "amide", "diester", "diamide"],
                        help="Bond types to form (default: all four)")
    parser.add_argument("--no_reference", action="store_true",
                        help="Omit reference molecules (water, hexadecane, PFAS, …)")
    args = parser.parse_args()

    lib = build_library(args.catalogues, args.isomers, args.types,
                        include_reference=not args.no_reference)
    if lib.empty:
        print("No products generated — check catalogue path and compound types.",
              file=sys.stderr)
        sys.exit(1)

    lib.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
