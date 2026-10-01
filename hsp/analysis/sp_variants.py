"""
sp_variants.py

Stefanis-Panayiotou evaluation with switchable ring-correction counting, built
on hsp_calculator's own group counting so every variant shares the same
first-order groups and non-ring second-order groups.

Ring-counting variants (the papers give no rule for fused/bridged systems):
    per_ring      current code: one correction per all-carbon non-aromatic SSSR ring
    per_system    one correction per ring system (size of its first qualifying ring)
    isolated_only only rings sharing no atom with another ring (the alizarin reading)
    discount_w    first qualifying ring of a system counts 1, every further ring w

Returns the pre-fallback ("normal equation") values as well as the final ones,
so the low-value branch and the 0 floor can be studied separately.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from rdkit import Chem, RDLogger

import hsp_calculator as hc
from group_tables import (CONST_D, POWER_D, CONST_P, CONST_HB, CONST_P_LOW,
                          CONST_HB_LOW, THRESHOLD_LOW, FIRST_ORDER_GROUPS,
                          SECOND_ORDER_GROUPS, LOW_VALUE_CORRECTIONS,
                          LOW_VALUE_2ND_ORDER)

RDLogger.DisableLog("rdApp.*")

FO = {n: (d, p, h) for n, _, d, p, h in FIRST_ORDER_GROUPS}
SO = {n: (d, p, h) for n, d, p, h in SECOND_ORDER_GROUPS}
RING_GROUPS = {3: "ring_3C", 5: "ring_5C", 6: "ring_6C"}


@dataclass
class SPOut:
    D: float | None = None
    P: float | None = None
    H: float | None = None
    P_normal: float | None = None     # Eq. A.3 before the low-value branch
    H_normal: float | None = None
    P_low_branch: bool = False
    H_low_branch: bool = False
    n_ring_corr: float = 0.0
    n_unavail: int = 0
    error: str | None = None


def _qualifying_rings(mol):
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    q = []
    for i, r in enumerate(rings):
        atoms = [mol.GetAtomWithIdx(a) for a in r]
        if len(r) in RING_GROUPS and all(a.GetAtomicNum() == 6 and not a.GetIsAromatic() for a in atoms):
            q.append(i)
    return rings, q


def _ring_systems(rings):
    systems: list[set[int]] = []
    for i, r in enumerate(rings):
        hit = [s for s in systems if any(r & rings[j] for j in s)]
        merged = {i}.union(*hit) if hit else {i}
        systems = [s for s in systems if s not in hit] + [merged]
    return systems


def ring_counts(mol, variant: str, w: float = 1.0) -> dict[str, float]:
    rings, q = _qualifying_rings(mol)
    out: dict[str, float] = {}

    def add(i, weight):
        g = RING_GROUPS[len(rings[i])]
        out[g] = out.get(g, 0.0) + weight

    if variant == "per_ring":
        for i in q:
            add(i, 1.0)
    elif variant == "isolated_only":
        for i in q:
            if not any(j != i and rings[i] & rings[j] for j in range(len(rings))):
                add(i, 1.0)
    elif variant in ("per_system", "discount"):
        for s in _ring_systems(rings):
            qs = sorted(i for i in s if i in q)
            for k, i in enumerate(qs):
                if k == 0:
                    add(i, 1.0)
                elif variant == "discount":
                    add(i, w)
    else:
        raise ValueError(variant)
    return out


def compute(smiles: str, variant: str = "per_ring", w: float = 1.0,
            floor: bool = False, trigger: str = "normal") -> SPOut:
    """trigger selects when the low-value (Eq. A.5/A.6) branch is used:
        normal   current code: Eq. A.3/A.4 result < 3
        covered  low-value result < 3 AND every group present has a low-value
                 entry (*** in Table A.5/A.6 means no low-polarity training
                 compound carried that group, so the branch is out of domain)
        either   normal, or covered
    """
    o = SPOut()
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        o.error = "parse"
        return o
    if any(a.GetAtomicNum() == 14 for a in mol.GetAtoms()):
        mol = hc._replace_si_with_c(mol)
    fo, _ = hc._count_first_order(mol)
    so = hc._count_second_order(mol, fo)
    so = {k: v for k, v in so.items() if k not in RING_GROUPS.values()}
    so.update(ring_counts(mol, variant, w))
    o.n_ring_corr = sum(v for k, v in so.items() if k in RING_GROUPS.values())
    W = 1 if any(v > 0 for v in so.values()) else 0

    cover = {1: True, 2: True}

    def tot(idx, low=False):
        s = 0.0
        for g, c in fo.items():
            v = (LOW_VALUE_CORRECTIONS.get(g, (None, None))[idx - 1] if low else FO[g][idx])
            if v is None:
                if low:
                    cover[idx] = False
                else:
                    o.n_unavail += 1
            else:
                s += c * v
        for g, c in so.items():
            if low:
                v = LOW_VALUE_2ND_ORDER.get(g, (None, None))[idx - 1]
            else:
                v = SO.get(g, (None, None, None))[idx]
            if v is not None:
                s += W * c * v
        return s

    raw_d = tot(0) + CONST_D
    o.D = math.copysign(abs(raw_d) ** POWER_D, raw_d)
    o.P_normal = tot(1) + CONST_P
    o.H_normal = tot(2) + CONST_HB
    P_lowv = tot(1, low=True) + CONST_P_LOW
    H_lowv = tot(2, low=True) + CONST_HB_LOW

    def use_low(normal, lowv, covered):
        cov = covered and lowv < THRESHOLD_LOW
        return {"normal": normal < THRESHOLD_LOW, "covered": cov,
                "either": normal < THRESHOLD_LOW or cov}[trigger]

    o.P_low_branch = use_low(o.P_normal, P_lowv, cover[1])
    o.H_low_branch = use_low(o.H_normal, H_lowv, cover[2])
    o.P = P_lowv if o.P_low_branch else o.P_normal
    o.H = H_lowv if o.H_low_branch else o.H_normal
    if floor:
        o.P, o.H = max(o.P, 0.0), max(o.H, 0.0)
    return o
