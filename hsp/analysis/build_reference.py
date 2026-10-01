"""
build_reference.py

Merge the two HSPiP sheets into one structure-keyed reference table.

    tier 1 = "HSPiP master" (the ~1.2K curated Hansen set)
    tier 2 = in "HSPiP 10k" only (provenance per row is not stated; many are
             expected to be estimates, so tier 2 is never used to judge accuracy)

Rows are keyed on the InChIKey connectivity block (first 14 chars), so stereo
isomers and duplicate CAS entries collapse to one structure. When a key has
several rows in the same tier, the median HSP is kept and the spread recorded.

Output: _hsp/reference/hspip_reference.csv (gitignored, licensed data)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REF = Path(__file__).resolve().parents[1] / "reference"
SP_ELEMENTS = {1, 6, 7, 8, 9, 16, 17, 35, 53}   # elements Table A.1 covers


def ring_topology(mol: Chem.Mol) -> dict:
    """Classify ring systems: fused (share a bond), bridged (share >2 atoms)."""
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    arom = [all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r) for r in rings]
    # union rings that share atoms into ring systems
    systems: list[set[int]] = []
    for i, r in enumerate(rings):
        hit = [s for s in systems if any(len(r & rings[j]) for j in s)]
        merged = {i}.union(*hit) if hit else {i}
        systems = [s for s in systems if s not in hit] + [merged]
    fused = bridged = False
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            n = len(rings[i] & rings[j])
            if n == 2:
                fused = True
            elif n > 2:
                bridged = True
    max_sys = max((len(s) for s in systems), default=0)
    aliph_in_poly = any(len(s) > 1 and not all(arom[k] for k in s) for s in systems)
    return {
        "n_rings": len(rings),
        "n_ring_systems": len(systems),
        "max_rings_in_system": max_sys,
        "n_aliph_rings": sum(not a for a in arom),
        "fused": fused,
        "bridged": bridged,
        "aliph_polycyclic": aliph_in_poly,   # >=2-ring system with a non-aromatic ring
    }


def ring_class(t: dict) -> str:
    if t["n_rings"] == 0:
        return "acyclic"
    if t["max_rings_in_system"] == 1:
        return "monocyclic" if t["n_ring_systems"] == 1 else "ring_assembly"
    if t["aliph_polycyclic"]:
        return "polycyclic_bridged" if t["bridged"] else (
            "polycyclic_3plus" if t["max_rings_in_system"] >= 3 else "polycyclic_fused2")
    return "fused_aromatic"


def describe(smiles: str) -> dict | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    elems = {a.GetAtomicNum() for a in mol.GetAtoms()}
    t = ring_topology(mol)
    n_c = sum(a.GetAtomicNum() == 6 for a in mol.GetAtoms())
    return {
        "canonical_smiles": Chem.MolToSmiles(Chem.RemoveHs(mol)),
        "ikey14": Chem.MolToInchiKey(mol)[:14],
        "n_heavy": mol.GetNumHeavyAtoms(),
        "n_carbon": n_c,
        "n_si": sum(a.GetAtomicNum() == 14 for a in mol.GetAtoms()),
        "charged": any(a.GetFormalCharge() for a in mol.GetAtoms()),
        "sp_elements_only": elems <= SP_ELEMENTS,
        "ring_class": ring_class(t),
        **t,
    }


def load(sheet_csv: str, tier: int) -> pd.DataFrame:
    df = pd.read_csv(REF / sheet_csv)
    df = df.rename(columns={"δD": "ref_D", "δP": "ref_P", "δH": "ref_H"})
    df = df[["Name", "CAS", "SMILES", "ref_D", "ref_P", "ref_H", "MVol"]].copy()
    df["tier"] = tier
    return df


def main():
    raw = pd.concat([load("hspip_master.csv", 1), load("hspip_10k.csv", 2)],
                    ignore_index=True)
    raw = raw.dropna(subset=["SMILES", "ref_D", "ref_P", "ref_H"])
    info = [describe(s) for s in raw["SMILES"]]
    ok = [i is not None for i in info]
    print(f"parsed {sum(ok)}/{len(raw)} SMILES")
    raw = pd.concat([raw[ok].reset_index(drop=True),
                     pd.DataFrame([i for i in info if i is not None])], axis=1)

    # a structure in master is tier 1 wherever else it appears
    t1_keys = set(raw.loc[raw.tier == 1, "ikey14"])
    raw = raw[(raw.tier == 1) | ~raw.ikey14.isin(t1_keys)]

    agg = {c: "first" for c in raw.columns if c not in ("ref_D", "ref_P", "ref_H", "ikey14")}
    g = raw.groupby("ikey14")
    out = g.agg(agg)
    for c in ("ref_D", "ref_P", "ref_H"):
        out[c] = g[c].median()
        out[c + "_spread"] = g[c].max() - g[c].min()
    out["n_rows"] = g.size()
    out = out.reset_index()
    out.to_csv(REF / "hspip_reference.csv", index=False)

    print(out.groupby("tier").size().rename("structures").to_string())
    print(pd.crosstab(out.ring_class, out.tier, margins=True).to_string())
    print("Si-containing:", out.groupby("tier").n_si.apply(lambda s: (s > 0).sum()).to_dict())
    print("rows with conflicting duplicates (spread > 1):",
          int((out[["ref_D_spread", "ref_P_spread", "ref_H_spread"]].max(axis=1) > 1).sum()))


if __name__ == "__main__":
    main()
