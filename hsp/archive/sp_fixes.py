"""
sp_variants.py

Stefanis-Panayiotou (2012) HSP with switchable fixes, for validating candidate
corrections against the HSPiP master set before touching hsp_calculator.py.

Fixes (each independently switchable):
  olefin  : CH3-C=, -CH2-C=, >C{H/C}-C= only on acyclic C=C.
            Evidence: 2012 Appendix A.4 examples 9 (alpha-terpinene) and 20
            (methylcyclopentadiene) count none of these for ring C=C, while
            examples 1, 2, 16, 18 count them for acyclic C=C.
  arom_oh : Ccyclic-OH only on non-aromatic ring C (example 7, vanillin).
  arom_n  : >N{H/C}(in cyclic) only on non-aromatic ring N (example 19,
            2,4,6-trimethylpyridine).
  string  : string-in-cyclic only for an unbranched (CH2-started) chain
            (example 9: isopropyl on a ring is not counted).
Ring counting for fused/bridged systems (the papers give no rule):
  per_ring      : one ring correction per SSSR ring (current behaviour)
  per_system    : one correction per ring size present in each ring system
  isolated_only : corrections only for rings that are not fused/bridged
Floor: clamp final dP, dH at 0 (HSP components are non-negative by definition).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdkit import Chem

import hsp_calculator as h
from group_tables import FIRST_ORDER_GROUPS, SECOND_ORDER_GROUPS

_FO = {n: (cd, cp, ch) for n, _, cd, cp, ch in FIRST_ORDER_GROUPS}
_SO = {n: (a, b, c) for n, a, b, c in SECOND_ORDER_GROUPS}

_P_CH3_OL = Chem.MolFromSmarts("[CH3X4][CX3]=!@[CX3]")
_P_CH2_OL = Chem.MolFromSmarts("[CH2X4][CX3]=!@[CX3]")
_P_CHC_OL = Chem.MolFromSmarts("[CHX4,CX4H0][CX3]=!@[CX3]")
_P_CYC_OH = Chem.MolFromSmarts("[C;R;!a][OX2H1]")
_P_CYC_N = Chem.MolFromSmarts("[N;R;!a]")
_P_STRING = Chem.MolFromSmarts("[C;R;!a]-!@[CH2X4;!R]-[#6;!R]")


def _ring_systems(mol):
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    systems = []  # list of (atoms, [ring indices])
    for i, r in enumerate(rings):
        hit = [s for s in systems if len(s[0] & r) >= 2]
        atoms, members = set(r), [i]
        for s in hit:
            systems.remove(s)
            atoms |= s[0]
            members += s[1]
        systems.append((atoms, members))
    return rings, systems


def _all_carbon_nonarom(mol, ring):
    return all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6
               and not mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)


def _ring_counts(mol, mode):
    rings, systems = _ring_systems(mol)
    out = {}
    for atoms, members in systems:
        sizes = [len(rings[i]) for i in members if _all_carbon_nonarom(mol, rings[i])]
        sizes = [s for s in sizes if s in (3, 5, 6)]
        if mode == "per_ring":
            use = sizes
        elif mode == "per_system":
            use = sorted(set(sizes))
        elif mode == "isolated_only":
            use = sizes if len(members) == 1 else []
        else:
            raise ValueError(mode)
        for s in use:
            k = f"ring_{s}C"
            out[k] = out.get(k, 0) + 1
    return out


def second_order(mol, fo, fixes=(), ring_mode="per_ring"):
    so = h._count_second_order(mol, fo)
    so = {k: v for k, v in so.items()}

    def _set(k, n):
        if n:
            so[k] = n
        else:
            so.pop(k, None)

    if "olefin" in fixes:
        _set("CH3-C=", len(mol.GetSubstructMatches(_P_CH3_OL)))
        _set("-CH2-C=", len(mol.GetSubstructMatches(_P_CH2_OL)))
        _set(">C{H/C}-C=", len(mol.GetSubstructMatches(_P_CHC_OL)))
    if "arom_oh" in fixes:
        _set("Ccyclic-OH", len(mol.GetSubstructMatches(_P_CYC_OH)))
    if "arom_n" in fixes:
        _set(">N{H/C}(cyclic)", len(mol.GetSubstructMatches(_P_CYC_N)))
    if "string" in fixes and "string_in_cyclic" in so:
        n = len({m[0] for m in mol.GetSubstructMatches(_P_STRING)})
        _set("string_in_cyclic", min(n, so["string_in_cyclic"]))
    for k in ("ring_3C", "ring_5C", "ring_6C"):
        so.pop(k, None)
    for k, v in _ring_counts(mol, ring_mode).items():
        so[k] = v
    return so


def compute(smiles, fixes=(), ring_mode="per_ring", floor=False):
    """Return (dD, dP, dH, info) or (None, None, None, info) on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None, {"error": "parse"}
    if any(a.GetAtomicNum() == 14 for a in mol.GetAtoms()):
        return None, None, None, {"error": "Si: outside S-P method"}
    fo, used = h._count_first_order(mol)
    unmatched = mol.GetNumHeavyAtoms() - len(used)
    so = second_order(mol, fo, fixes, ring_mode)
    W = 1 if so else 0
    s = [0.0] * 6
    unavail = 0
    for g, c in fo.items():
        for j, v in enumerate(_FO[g]):
            if v is None:
                unavail += c
            else:
                s[j] += c * v
    for g, c in so.items():
        if g not in _SO:
            continue
        for j, v in enumerate(_SO[g]):
            if v is None:
                unavail += c
            else:
                s[3 + j] += c * v
    dD, dP, dH = h._apply_formulas(s[0], s[1], s[2], s[3], s[4], s[5], W, fo, so)
    info = {"unmatched": unmatched, "unavail": unavail, "so": so,
            "raw_negative": (dP < 0) or (dH < 0)}
    if floor:
        dP, dH = max(dP, 0.0), max(dH, 0.0)
    return dD, dP, dH, info
