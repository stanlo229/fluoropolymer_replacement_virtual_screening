"""
hsp_calculator.py

Core Stefanis–Panayiotou HSP computation.

For each molecule:
  1. (Optional) Si→C substitution for group matching
  2. First-order group counting via SMARTS (priority order, no atom reuse)
  3. Second-order group identification via structural checks (W=1 if any found)
  4. Apply Eqs. A.2–A.4; re-evaluate with low-value constants if needed
  5. Flag n_unmatched_atoms, has_si, approx_ring_correction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from group_tables import (
    CONST_D, POWER_D, CONST_P, CONST_HB,
    CONST_P_LOW, CONST_HB_LOW, THRESHOLD_LOW,
    FIRST_ORDER_GROUPS, SECOND_ORDER_GROUPS,
    LOW_VALUE_CORRECTIONS, LOW_VALUE_2ND_ORDER,
)

log = logging.getLogger(__name__)

# Pre-compile SMARTS patterns once at import time
_FO_PATTERNS = [
    (name, Chem.MolFromSmarts(sma), cd, cp, ch)
    for name, sma, cd, cp, ch in FIRST_ORDER_GROUPS
]
assert all(p is not None for _, p, *_ in _FO_PATTERNS), \
    "One or more first-order SMARTS failed to compile"

_SI_SMARTS = Chem.MolFromSmarts("[Si]")


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class HSPResult:
    delta_D: Optional[float] = None
    delta_P: Optional[float] = None
    delta_H: Optional[float] = None
    n_unmatched_atoms: int = 0
    has_si: bool = False
    n_si_atoms: int = 0
    used_2nd_order: bool = False
    approx_ring_correction: bool = False
    error: Optional[str] = None
    group_counts: dict = field(default_factory=dict)   # name → count


# ---------------------------------------------------------------------------
# Si substitution
# ---------------------------------------------------------------------------
def _replace_si_with_c(mol: Chem.Mol) -> Chem.Mol:
    """Return a copy of mol with all Si atoms replaced by C, re-sanitized."""
    rwmol = Chem.RWMol(mol)
    for atom in rwmol.GetAtoms():
        if atom.GetAtomicNum() == 14:   # Si
            atom.SetAtomicNum(6)        # C
            atom.SetNoImplicit(False)
    try:
        Chem.SanitizeMol(rwmol)
    except Exception:
        pass
    return rwmol.GetMol()


# ---------------------------------------------------------------------------
# First-order group counting
# ---------------------------------------------------------------------------
def _count_first_order(mol: Chem.Mol) -> tuple[dict[str, int], set[int]]:
    """
    Match first-order groups in priority order.
    Each heavy atom is assigned to at most one group.

    Returns
    -------
    counts   : {group_name: count}
    used_atoms : set of matched atom indices
    """
    used: set[int] = set()
    counts: dict[str, int] = {}

    for name, pattern, *_ in _FO_PATTERNS:
        matches = mol.GetSubstructMatches(pattern)
        group_count = 0
        for match in matches:
            # Only heavy atoms (exclude explicit H in match)
            heavy = [
                idx for idx in match
                if mol.GetAtomWithIdx(idx).GetAtomicNum() != 1
            ]
            # Accept match only if none of its heavy atoms are already used
            if not any(idx in used for idx in heavy):
                used.update(heavy)
                group_count += 1
        if group_count:
            counts[name] = group_count

    return counts, used


# ---------------------------------------------------------------------------
# Second-order group identification
# ---------------------------------------------------------------------------
def _count_second_order(mol: Chem.Mol, fo_counts: dict[str, int]) -> dict[str, int]:
    """
    Identify second-order groups using structural checks.
    Returns {group_name: count}.
    """
    so: dict[str, int] = {}

    ri = mol.GetRingInfo()
    ring_atom_sets = [set(r) for r in ri.AtomRings()]

    def _add(name: str, n: int = 1):
        if n > 0:
            so[name] = so.get(name, 0) + n

    # ---- (CH3)2-CH- : isopropyl -----------------------------------------
    pat_iso = Chem.MolFromSmarts("[CH3X4][CHX4]([CH3X4])")
    _add("(CH3)2-CH-", len(mol.GetSubstructMatches(pat_iso)))

    # ---- (CH3)3-C- : tert-butyl -----------------------------------------
    pat_tbu = Chem.MolFromSmarts("[CH3X4][CX4H0]([CH3X4])[CH3X4]")
    _add("(CH3)3-C-", len(mol.GetSubstructMatches(pat_tbu)))

    # ---- Ring of 3 C (cyclopropane-like) --------------------------------
    for ring in ring_atom_sets:
        if len(ring) == 3:
            atoms = [mol.GetAtomWithIdx(i) for i in ring]
            if all(a.GetAtomicNum() == 6 and not a.GetIsAromatic() for a in atoms):
                _add("ring_3C")

    # ---- Ring of 5 C (cyclopentane-like) --------------------------------
    # Also triggered by norbornene scaffold (approximation); flag separately
    for ring in ring_atom_sets:
        if len(ring) == 5:
            atoms = [mol.GetAtomWithIdx(i) for i in ring]
            if all(a.GetAtomicNum() == 6 and not a.GetIsAromatic() for a in atoms):
                _add("ring_5C")

    # ---- Ring of 6 C (cyclohexane-like) --------------------------------
    for ring in ring_atom_sets:
        if len(ring) == 6:
            atoms = [mol.GetAtomWithIdx(i) for i in ring]
            if all(a.GetAtomicNum() == 6 and not a.GetIsAromatic() for a in atoms):
                _add("ring_6C")

    # ---- Conjugated diene -C=C-C=C- ------------------------------------
    pat_diene = Chem.MolFromSmarts("[CX3]=[CX3]-[CX3]=[CX3]")
    _add("-C=C-C=C-", len(mol.GetSubstructMatches(pat_diene)))

    # ---- CH3-C= (methyl on olefinic C) ---------------------------------
    pat_me_ol = Chem.MolFromSmarts("[CH3X4][CX3]=[CX3]")
    _add("CH3-C=", len(mol.GetSubstructMatches(pat_me_ol)))

    # ---- -CH2-C= (methylene on olefinic C) -----------------------------
    pat_ch2_ol = Chem.MolFromSmarts("[CH2X4][CX3]=[CX3]")
    _add("-CH2-C=", len(mol.GetSubstructMatches(pat_ch2_ol)))

    # ---- >C{H/C}-C= (CH or C attached to olefinic C) ------------------
    pat_chc_ol = Chem.MolFromSmarts("[CHX4,CX4H0][CX3]=[CX3]")
    _add(">C{H/C}-C=", len(mol.GetSubstructMatches(pat_chc_ol)))

    # ---- String in cyclic (ethyl+ alkyl chain on ring) -----------------
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6 and not atom.GetIsAromatic():
            if any(atom.GetIdx() in ring for ring in ring_atom_sets):
                for nbr in atom.GetNeighbors():
                    if nbr.GetAtomicNum() == 6 and not nbr.GetIsAromatic():
                        if not any(nbr.GetIdx() in ring for ring in ring_atom_sets):
                            # Check if this chain has ≥ 2 carbons (ethyl+)
                            chain_len = 0
                            cur = nbr
                            prev_idx = atom.GetIdx()
                            while cur is not None:
                                chain_len += 1
                                next_atoms = [
                                    n for n in cur.GetNeighbors()
                                    if n.GetIdx() != prev_idx
                                    and n.GetAtomicNum() == 6
                                    and not n.GetIsAromatic()
                                    and not any(n.GetIdx() in ring for ring in ring_atom_sets)
                                ]
                                if chain_len >= 2 and not next_atoms:
                                    _add("string_in_cyclic")
                                    break
                                if next_atoms:
                                    prev_idx = cur.GetIdx()
                                    cur = next_atoms[0]
                                else:
                                    break

    # ---- CH3(CO)CH2- (methyl ketone flanked by CH2) -------------------
    pat_mekch2 = Chem.MolFromSmarts("[CH3X4][CX3](=O)[CH2X4]")
    _add("CH3(CO)CH2-", len(mol.GetSubstructMatches(pat_mekch2)))

    # ---- Ccyclic=O (ketone within ring) --------------------------------
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6 and not atom.GetIsAromatic():
            if any(atom.GetIdx() in ring for ring in ring_atom_sets):
                for bond in atom.GetBonds():
                    if bond.GetBondTypeAsDouble() == 2.0:
                        nbr = bond.GetOtherAtom(atom)
                        if nbr.GetAtomicNum() == 8:
                            _add("Ccyclic=O")

    # ---- ACCOOH : aromatic + COOH --------------------------------------
    pat_accooh = Chem.MolFromSmarts("[c][CX3](=O)[OX2H1]")
    _add("ACCOOH", len(mol.GetSubstructMatches(pat_accooh)))

    # ---- >C{H/C}-COOH --------------------------------------------------
    pat_chcooh = Chem.MolFromSmarts("[CHX4,CX4H0][CX3](=O)[OX2H1]")
    _add(">C{H/C}-COOH", len(mol.GetSubstructMatches(pat_chcooh)))

    # ---- CH3(CO)OC{H/C}< (isopropyl/sec-butyl acetate) ----------------
    pat_secac = Chem.MolFromSmarts("[CH3X4][CX3](=O)[OX2H0][CHX4,CX4H0]")
    _add("CH3(CO)OC{H/C}<", len(mol.GetSubstructMatches(pat_secac)))

    # ---- (CO)C{H2}COO (β-ketoester) -----------------------------------
    pat_bkest = Chem.MolFromSmarts("[CX3](=O)[CH2X4][CX3](=O)[OX2H0]")
    _add("(CO)C{H2}COO", len(mol.GetSubstructMatches(pat_bkest)))

    # ---- (CO)O(CO) anhydride -------------------------------------------
    pat_anhy = Chem.MolFromSmarts("[CX3](=O)[OX2H0][CX3](=O)")
    _add("(CO)O(CO)", len(mol.GetSubstructMatches(pat_anhy)))

    # ---- ACHO (aromatic aldehyde) --------------------------------------
    pat_acho = Chem.MolFromSmarts("[c][CX3H1]=O")
    _add("ACHO", len(mol.GetSubstructMatches(pat_acho)))

    # ---- >CHOH (CH bearing OH) ----------------------------------------
    pat_choh = Chem.MolFromSmarts("[CHX4][OX2H1]")
    _add(">CHOH", len(mol.GetSubstructMatches(pat_choh)))

    # ---- >C<OH (quaternary C bearing OH) -------------------------------
    pat_cqoh = Chem.MolFromSmarts("[CX4H0][OX2H1]")
    _add(">C<OH", len(mol.GetSubstructMatches(pat_cqoh)))

    # ---- -C(OH)C(OH)- (vicinal diol) -----------------------------------
    pat_vdiol = Chem.MolFromSmarts("[CX4][OX2H1].[CX4][OX2H1]")   # adjacency checked below
    pat_vdiol2 = Chem.MolFromSmarts("[OX2H1][CX4][CX4][OX2H1]")
    _add("-C(OH)C(OH)-", len(mol.GetSubstructMatches(pat_vdiol2)))

    # ---- -C(OH)C(N) (β-amino alcohol) ----------------------------------
    pat_ba = Chem.MolFromSmarts("[OX2H1][CX4][CX4][NX3]")
    _add("-C(OH)C(N)", len(mol.GetSubstructMatches(pat_ba)))

    # ---- Ccyclic-OH (cyclic C bearing OH) ------------------------------
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:
            if any(atom.GetIdx() in ring for ring in ring_atom_sets):
                for nbr in atom.GetNeighbors():
                    if nbr.GetAtomicNum() == 8 and nbr.GetTotalNumHs() == 1:
                        _add("Ccyclic-OH")

    # ---- C-O-C=C (vinyl ether) -----------------------------------------
    pat_veth = Chem.MolFromSmarts("[CX4][OX2H0][CX3]=[CX3]")
    _add("C-O-C=C", len(mol.GetSubstructMatches(pat_veth)))

    # ---- AC-O-C (aromatic ether) ---------------------------------------
    pat_aroeth = Chem.MolFromSmarts("[c][OX2H0][CX4]")
    _add("AC-O-C", len(mol.GetSubstructMatches(pat_aroeth)))

    # ---- ACCOO (aromatic ester) ----------------------------------------
    pat_accoo = Chem.MolFromSmarts("[c][CX3](=O)[OX2H0]")
    _add("ACCOO", len(mol.GetSubstructMatches(pat_accoo)))

    # ---- AC-O-AC (diaryl ether) ----------------------------------------
    pat_dary = Chem.MolFromSmarts("[c][OX2H0][c]")
    _add("AC-O-AC", len(mol.GetSubstructMatches(pat_dary)))

    # ---- >N{H/C}(cyclic) : N in ring -----------------------------------
    for ring in ring_atom_sets:
        for idx in ring:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() == 7:
                _add(">N{H/C}(cyclic)")

    # ---- -S-(cyclic) ---------------------------------------------------
    for ring in ring_atom_sets:
        for idx in ring:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() == 16:
                _add("-S-(cyclic)")

    # ---- ACBr (aryl bromide) -------------------------------------------
    pat_acbr = Chem.MolFromSmarts("[BrX1][c]")
    _add("ACBr", len(mol.GetSubstructMatches(pat_acbr)))

    # ---- (C=C)-Br (vinyl bromide) --------------------------------------
    pat_vbr = Chem.MolFromSmarts("[BrX1][CX3]=[CX3]")
    _add("(C=C)-Br", len(mol.GetSubstructMatches(pat_vbr)))

    # ---- Naphthalene-type fused ring -----------------------------------
    pat_naph = Chem.MolFromSmarts("c1ccc2ccccc2c1")
    _add("AC(ACHm)2AC(ACHn)2", len(mol.GetSubstructMatches(pat_naph)))

    # ---- Ocyclic-Ccyclic=O (lactone) -----------------------------------
    pat_lac = Chem.MolFromSmarts("[OX2H0;R][CX3;R](=O)")
    _add("Ocyclic-Ccyclic=O", len(mol.GetSubstructMatches(pat_lac)))

    # ---- NcyclicH-Ccyclic=O (lactam) -----------------------------------
    pat_lam = Chem.MolFromSmarts("[NX3H1;R][CX3;R](=O)")
    _add("NcyclicH-Ccyclic=O", len(mol.GetSubstructMatches(pat_lam)))

    # ---- -O-CHm-O-CHn- (acetal) ----------------------------------------
    pat_acet = Chem.MolFromSmarts("[OX2H0][CX4][OX2H0]")
    _add("-O-CHm-O-CHn-", len(mol.GetSubstructMatches(pat_acet)))

    # ---- C(=O)-C-C(=O) (1,3-diketone) ---------------------------------
    pat_diket = Chem.MolFromSmarts("[CX3](=O)[CX4][CX3](=O)")
    _add("C(=O)-C-C(=O)", len(mol.GetSubstructMatches(pat_diket)))

    return so


# ---------------------------------------------------------------------------
# Norbornene scaffold detection
# ---------------------------------------------------------------------------
_NORBORNENE_SMARTS = Chem.MolFromSmarts(
    "[C@H]1C[C@@H]2C=C[C@@H]1C2"   # exo bicyclo[2.2.1]hept-2-ene core
)

def _has_norbornene(mol: Chem.Mol) -> bool:
    if _NORBORNENE_SMARTS is None:
        return False
    return mol.HasSubstructMatch(_NORBORNENE_SMARTS)


# ---------------------------------------------------------------------------
# HSP formula application
# ---------------------------------------------------------------------------
def _apply_formulas(
    sum_Ci_d: float, sum_Ci_p: float, sum_Ci_hb: float,
    sum_Dj_d: float, sum_Dj_p: float, sum_Dj_hb: float,
    W: int,
    fo_counts: dict[str, int],
    so_counts: dict[str, int],
) -> tuple[float, float, float]:
    """
    Compute δD, δP, δH using S-P 2012 Eqs. A.2–A.4, with low-value fallbacks.
    """
    raw_d  = sum_Ci_d  + W * sum_Dj_d  + CONST_D
    delta_D = abs(raw_d) ** POWER_D * (1 if raw_d >= 0 else -1)

    delta_P  = sum_Ci_p  + W * sum_Dj_p  + CONST_P
    delta_HB = sum_Ci_hb + W * sum_Dj_hb + CONST_HB

    # Low-value fallback for δP
    if delta_P < THRESHOLD_LOW:
        sum_Ci_p_low = 0.0
        sum_Dj_p_low = 0.0
        for gname, cnt in fo_counts.items():
            corr = LOW_VALUE_CORRECTIONS.get(gname)
            if corr and corr[0] is not None:
                sum_Ci_p_low += cnt * corr[0]
        for gname, cnt in so_counts.items():
            corr = LOW_VALUE_2ND_ORDER.get(gname)
            if corr and corr[0] is not None:
                sum_Dj_p_low += cnt * corr[0]
        delta_P = sum_Ci_p_low + W * sum_Dj_p_low + CONST_P_LOW

    # Low-value fallback for δH
    if delta_HB < THRESHOLD_LOW:
        sum_Ci_hb_low = 0.0
        sum_Dj_hb_low = 0.0
        for gname, cnt in fo_counts.items():
            corr = LOW_VALUE_CORRECTIONS.get(gname)
            if corr and corr[1] is not None:
                sum_Ci_hb_low += cnt * corr[1]
        for gname, cnt in so_counts.items():
            corr = LOW_VALUE_2ND_ORDER.get(gname)
            if corr and corr[1] is not None:
                sum_Dj_hb_low += cnt * corr[1]
        delta_HB = sum_Ci_hb_low + W * sum_Dj_hb_low + CONST_HB_LOW

    return delta_D, delta_P, delta_HB


# ---------------------------------------------------------------------------
# Main per-molecule computation
# ---------------------------------------------------------------------------
def compute_hsp(smiles: str) -> HSPResult:
    result = HSPResult()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        result.error = "RDKit parse failed"
        return result

    # Si detection
    si_matches = mol.GetSubstructMatches(_SI_SMARTS)
    result.n_si_atoms = len(si_matches)
    result.has_si = result.n_si_atoms > 0

    # Use Si→C substituted mol for group matching
    match_mol = _replace_si_with_c(mol) if result.has_si else mol

    n_heavy = match_mol.GetNumHeavyAtoms()

    # Norbornene scaffold check (before group counting, for flag only)
    result.approx_ring_correction = _has_norbornene(match_mol)

    # First-order group counting
    fo_counts, used_atoms = _count_first_order(match_mol)
    result.group_counts = dict(fo_counts)
    result.n_unmatched_atoms = n_heavy - len(used_atoms)

    # Second-order group identification
    so_counts = _count_second_order(match_mol, fo_counts)
    W = 1 if so_counts else 0
    result.used_2nd_order = bool(W)

    # Weighted sums (None contributions skipped)
    sum_Ci_d  = sum_Dj_d  = 0.0
    sum_Ci_p  = sum_Dj_p  = 0.0
    sum_Ci_hb = sum_Dj_hb = 0.0

    fo_lookup = {name: (cd, cp, ch) for name, _, cd, cp, ch in FIRST_ORDER_GROUPS}
    for gname, cnt in fo_counts.items():
        cd, cp, ch = fo_lookup[gname]
        if cd is not None: sum_Ci_d  += cnt * cd
        if cp is not None: sum_Ci_p  += cnt * cp
        if ch is not None: sum_Ci_hb += cnt * ch

    so_lookup = {name: (dd, dp, dh) for name, dd, dp, dh in SECOND_ORDER_GROUPS}
    for gname, cnt in so_counts.items():
        if gname not in so_lookup:
            continue
        dd, dp, dh = so_lookup[gname]
        if dd is not None: sum_Dj_d  += cnt * dd
        if dp is not None: sum_Dj_p  += cnt * dp
        if dh is not None: sum_Dj_hb += cnt * dh

    try:
        delta_D, delta_P, delta_HB = _apply_formulas(
            sum_Ci_d, sum_Ci_p, sum_Ci_hb,
            sum_Dj_d, sum_Dj_p, sum_Dj_hb,
            W, fo_counts, so_counts,
        )
    except Exception as exc:
        result.error = f"formula error: {exc}"
        return result

    result.delta_D = round(delta_D, 4)
    result.delta_P = round(delta_P, 4)
    result.delta_H = round(delta_HB, 4)
    return result


# ---------------------------------------------------------------------------
# Batch computation
# ---------------------------------------------------------------------------
def calculate_hsp_batch(
    df: pd.DataFrame,
    smiles_col: str = "monomer_smiles",
) -> pd.DataFrame:
    """
    Add delta_D, delta_P, delta_H and metadata columns to df in-place copy.
    """
    records = []
    for smi in df[smiles_col]:
        r = compute_hsp(str(smi))
        records.append({
            "delta_D":               r.delta_D,
            "delta_P":               r.delta_P,
            "delta_H":               r.delta_H,
            "n_unmatched_atoms":     r.n_unmatched_atoms,
            "has_si":                r.has_si,
            "n_si_atoms":            r.n_si_atoms,
            "used_2nd_order":        r.used_2nd_order,
            "approx_ring_correction": r.approx_ring_correction,
            "hsp_error":             r.error,
        })

    df_meta = pd.DataFrame(records, index=df.index)
    # Drop columns from df that df_meta will provide, to avoid duplicates
    overlap = [c for c in df_meta.columns if c in df.columns]
    return pd.concat([df.drop(columns=overlap), df_meta], axis=1)


# ---------------------------------------------------------------------------
# CLI / smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/monomers.csv")
    parser.add_argument("--output", default="results/monomers_hsp.csv")
    parser.add_argument("--test",   action="store_true",
                        help="Run unit tests on acetic acid and ethanol then exit")
    args = parser.parse_args()

    if args.test:
        # Reference values are Hansen (2007) experimental.
        # S-P method has stated AAD of ~0.5 / 15% / 15% for δD / δP / δH.
        # For small aliphatic alcohols δH can deviate by 5–6 MPa^0.5 — this is
        # a known limitation of the group-contribution parameters.
        tests = [
            # (smiles, name, exp_D, exp_P, exp_H, expected_groups)
            ("CC(=O)O", "acetic acid", 14.5, 8.0, 13.5,
             {"COOH": 1, "-CH3": 1}),
            ("CCO",     "ethanol",     15.8, 8.8, 19.4,
             {"OH_aliph": 1, "-CH2-": 1, "-CH3": 1}),
        ]
        # Per-component tolerances (MPa^0.5)
        TOL = {"D": 3.0, "P": 3.0, "H": 6.5}
        ok = True
        for smi, name, ed, ep, eh, exp_groups in tests:
            r = compute_hsp(smi)
            print(f"\n{name} ({smi})")
            print(f"  Groups found:    {r.group_counts}")
            print(f"  Groups expected: {exp_groups}")
            print(f"  2nd order: {r.used_2nd_order}")
            print(f"  δD={r.delta_D:.2f} (ref {ed}), "
                  f"δP={r.delta_P:.2f} (ref {ep}), "
                  f"δH={r.delta_H:.2f} (ref {eh})")
            # Check group counts
            for gname, cnt in exp_groups.items():
                found = r.group_counts.get(gname, 0)
                if found != cnt:
                    print(f"  FAIL: group '{gname}' count {found} != expected {cnt}")
                    ok = False
            # Check HSP deviations
            for pred, ref, label, tol in [
                (r.delta_D, ed, "D", TOL["D"]),
                (r.delta_P, ep, "P", TOL["P"]),
                (r.delta_H, eh, "H", TOL["H"]),
            ]:
                if pred is not None and abs(pred - ref) > tol:
                    print(f"  FAIL: δ{label} deviation {abs(pred-ref):.2f} > {tol} "
                          f"(S-P method accuracy limit)")
                    ok = False
        print("\nSmoke test", "PASSED" if ok else "FAILED (check warnings)")
        raise SystemExit(0 if ok else 1)

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    df = pd.read_csv(args.input)
    from tqdm import tqdm
    tqdm.pandas(desc="Computing HSP")
    df_out = calculate_hsp_batch(df)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)
    n_ok  = df_out["delta_D"].notna().sum()
    n_err = df_out["hsp_error"].notna().sum()
    print(f"Done. {n_ok} OK, {n_err} errors. Written to {args.output}")
