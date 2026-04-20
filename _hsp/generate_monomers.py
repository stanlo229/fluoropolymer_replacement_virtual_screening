"""
generate_monomers.py

Attach sidechains from dataset/catalogues.csv to the exo norbornene-2,3-diacid
scaffold via esterification (alcohols) or amidation (amines), producing a
library of difunctional norbornene monomers (symmetric diesters/diamides)
for HSP prediction.

Output: monomers.csv with columns:
    sidechain_smiles, monomer_smiles, linkage, source, compound_type,
    molecular_weight, has_si
"""

import logging
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# exo-norbornene-2,3-dicarboxylic acid (difunctional scaffold)
SCAFFOLD_EXO = "OC(=O)[C@H]1[C@@H](C(=O)O)[C@@H]2C=C[C@@H]1C2"

# Locates the carboxyl group on the scaffold.
# GetSubstructMatch returns (C_idx, =O_idx, OH_idx).
COOH_SMARTS = Chem.MolFromSmarts("[CX3](=[OX1])[OX2H1]")

# Reactive group detection on the sidechain.
#   Alcohol: any O-H not on a carbonyl (excludes COOH, ester, enol).
#   Amine: primary or secondary, not amide/sulfonamide/charged/imine.
ALCOHOL_SMARTS = Chem.MolFromSmarts("[OX2H;!$(OC=O)]")
AMINE_SMARTS   = Chem.MolFromSmarts(
    "[NX3;H1,H2;!$(NC=O);!$(NS(=O));!$([N+]);!$(N=*)]"
)

SI_SMARTS = Chem.MolFromSmarts("[Si]")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _has_si(mol: Chem.Mol) -> bool:
    return mol.HasSubstructMatch(SI_SMARTS)


def _largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    """Return the largest fragment (by heavy-atom count) of a disconnected mol."""
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())


def _attach_sidechain(scaffold_mol: Chem.Mol, reagent_mol: Chem.Mol,
                      reactive_smarts: Chem.Mol) -> str | None:
    """
    Form Norbornene-C(=O)-X-R by:
      1. Finding scaffold's carboxyl C and its OH oxygen.
      2. Finding each matching reactive atom X (O or N) in reagent.
      3. Removing the OH from scaffold, adding a bond scaffold-C → X.

    RDKit adjusts implicit Hs on X automatically after bond addition.
    Returns canonical SMILES of the first successful product, or None.
    """
    cooh_matches = scaffold_mol.GetSubstructMatches(COOH_SMARTS)
    if not cooh_matches:
        return None

    # Scaffold has exactly one COOH; atom order: (C, =O, OH)
    cooh_c_idx, _, cooh_oh_idx = cooh_matches[0]

    # Strip explicit Hs from reagent so atom indices are stable
    reagent_mol = Chem.RemoveHs(reagent_mol)
    r_matches = reagent_mol.GetSubstructMatches(reactive_smarts)
    if not r_matches:
        return None

    n_scaffold = scaffold_mol.GetNumAtoms()

    for r_match in r_matches:
        r_atom_idx = r_match[0]

        combo = Chem.CombineMols(scaffold_mol, reagent_mol)
        em = Chem.EditableMol(combo)

        # Remove the scaffold's OH oxygen.
        # Atoms with index > cooh_oh_idx shift down by 1.
        em.RemoveAtom(cooh_oh_idx)

        new_c_idx = cooh_c_idx if cooh_c_idx < cooh_oh_idx else cooh_c_idx - 1
        # Reagent atoms were at (n_scaffold + r_atom_idx); after removing one
        # scaffold atom (cooh_oh_idx < n_scaffold), they shift to (n_scaffold - 1 + r_atom_idx).
        new_r_idx = n_scaffold - 1 + r_atom_idx

        em.AddBond(new_c_idx, new_r_idx, Chem.BondType.SINGLE)

        mol = em.GetMol()
        try:
            Chem.SanitizeMol(mol)
            smi = Chem.MolToSmiles(mol)
            if smi:
                return smi
        except Exception:
            continue

    return None


def _attach_two_sidechains(scaffold_mol: Chem.Mol, reagent_mol: Chem.Mol,
                           reactive_smarts: Chem.Mol) -> str | None:
    """
    Attach the same reagent to both COOH groups of the diacid scaffold,
    producing a symmetric diester or diamide.
    Returns canonical SMILES or None if either attachment fails.
    """
    mid_smi = _attach_sidechain(scaffold_mol, reagent_mol, reactive_smarts)
    if mid_smi is None:
        return None
    mid_mol = Chem.MolFromSmiles(mid_smi)
    if mid_mol is None:
        return None
    if not mid_mol.HasSubstructMatch(COOH_SMARTS):
        return None
    return _attach_sidechain(mid_mol, reagent_mol, reactive_smarts)


def _parse_compound_types(raw: str) -> list[str]:
    return [t.strip().lower() for t in raw.split(",") if t.strip()]


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------
def generate_monomers(
    catalogues_path: str | Path,
    out_path: str | Path,
    skipped_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Read catalogues.csv, attach each sidechain to the norbornene scaffold,
    and write monomers.csv.

    Parameters
    ----------
    catalogues_path : path to dataset/catalogues.csv
    out_path        : path to write monomers.csv
    skipped_path    : path to write skipped.csv (default: out_path parent / skipped.csv)

    Returns
    -------
    DataFrame of successfully generated monomers.
    """
    catalogues_path = Path(catalogues_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if skipped_path is None:
        skipped_path = out_path.parent / "skipped.csv"
    skipped_path = Path(skipped_path)

    scaffold_mol = Chem.MolFromSmiles(SCAFFOLD_EXO)
    assert scaffold_mol is not None, "Scaffold SMILES failed to parse"

    df_in = pd.read_csv(catalogues_path)
    log.info("Loaded %d compounds from %s", len(df_in), catalogues_path)

    rows = []
    skipped = []

    for idx, row in df_in.iterrows():
        smiles    = str(row.get("smiles", "") or "").strip()
        source    = str(row.get("source", "") or "").strip()
        comp_type = str(row.get("compound_type", "") or "").strip()
        mw        = row.get("molecular_weight", None)

        if not smiles:
            skipped.append({"smiles": smiles, "compound_type": comp_type, "reason": "empty SMILES"})
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            skipped.append({"smiles": smiles, "compound_type": comp_type, "reason": "RDKit parse failed"})
            continue

        types = _parse_compound_types(comp_type)
        has_alcohol = "monoalcohol" in types or "diol" in types
        has_amine   = "monoamine"   in types

        if not has_alcohol and not has_amine:
            skipped.append({"smiles": smiles, "compound_type": comp_type,
                            "reason": "not monoalcohol/diol or monoamine"})
            continue

        mol_has_oh = mol.HasSubstructMatch(ALCOHOL_SMARTS)
        mol_has_nh = mol.HasSubstructMatch(AMINE_SMARTS)

        si_flag = _has_si(mol)

        linkages_to_try = []
        if has_alcohol and mol_has_oh:
            linkages_to_try.append(("ester", ALCOHOL_SMARTS))
        if has_amine and mol_has_nh:
            linkages_to_try.append(("amide", AMINE_SMARTS))

        if not linkages_to_try:
            reasons = []
            if has_alcohol and not mol_has_oh:
                reasons.append("no free alcohol OH found")
            if has_amine and not mol_has_nh:
                reasons.append("no free amine NH found")
            skipped.append({"smiles": smiles, "compound_type": comp_type,
                            "reason": "; ".join(reasons)})
            continue

        for linkage, reactive_smarts in linkages_to_try:
            monomer_smiles = _attach_two_sidechains(scaffold_mol, mol, reactive_smarts)
            if monomer_smiles is None:
                skipped.append({"smiles": smiles, "compound_type": comp_type,
                                "reason": f"attachment failed ({linkage})"})
                continue

            rows.append({
                "sidechain_smiles": Chem.MolToSmiles(mol),
                "monomer_smiles":   monomer_smiles,
                "linkage":          linkage,
                "source":           source,
                "compound_type":    comp_type,
                "molecular_weight": mw,
                "has_si":           si_flag,
            })

    df_out = pd.DataFrame(rows)
    df_skip = pd.DataFrame(skipped)

    df_out.to_csv(out_path, index=False)
    df_skip.to_csv(skipped_path, index=False)

    n_si = df_out["has_si"].sum() if len(df_out) else 0
    log.info(
        "Generated %d difunctional monomers (%d diester, %d diamide, %d with Si). "
        "Skipped %d. Written to %s",
        len(df_out),
        (df_out["linkage"] == "ester").sum() if len(df_out) else 0,
        (df_out["linkage"] == "amide").sum() if len(df_out) else 0,
        n_si,
        len(df_skip),
        out_path,
    )
    return df_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate norbornene monomers from catalogues.")
    parser.add_argument("--catalogues", default="../dataset/catalogues.csv")
    parser.add_argument("--out",        default="results/monomers.csv")
    parser.add_argument("--skipped",    default=None)
    args = parser.parse_args()

    generate_monomers(args.catalogues, args.out, args.skipped)
