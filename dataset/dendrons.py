"""
dendrons.py

Structural detection of dendrons among catalogue compounds, and of the focal
point through which a dendron attaches to the norbornene scaffold.

A *branch unit* is a branch point carrying >= 2 identical (symmetry-equivalent)
arms. Families (generation >= 1, i.e. one branch unit is enough):

  frechet   poly(aryl ether): benzylic C on an aryl ring with >= 2 O-CH2-aryl arms
  percec    alkoxy-benzyl:     benzylic C on an aryl ring with >= 2 O-CH2-alkyl arms of >= 8 C
  bis_mpa   2,2-bis(hydroxymethyl)propionic acid ester/amide with 2 identical CH2-O arms
  pamam     poly(amidoamine):  tertiary N with 2 identical CH2CH2C(=O)N arms
  newkome   1->3 C-branched:   quaternary C with 3 identical CH2CH2C(=O)X arms
  glycerol  polyglycerol: CH(-O)(-CH2-O-CH2-R)2, R = aryl (Frechet-type) or glycerol/acetonide (Haag-type)

Generic (generation >= 2 only): a symmetric branch point whose arm contains another
branch point of the same kind (element, degree, ring membership), i.e. nested branching.

Generation estimate: for a perfect 1->k dendron with n branch units,
n = (k^G - 1) / (k - 1), so G = log_k(n (k - 1) + 1), rounded down.

Focal point: among the attachable groups (alcohol OH, primary/secondary amine NH),
the one that is not symmetry-equivalent to any other attachable group. Peripheral
groups come in equivalent sets; the focal group is unique.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from rdkit import Chem

ALCOHOL = Chem.MolFromSmarts("[OX2H;!$(OC=O)]")
AMINE = Chem.MolFromSmarts("[NX3;H1,H2;!$(NC=O);!$(NS(=O));!$([N+]);!$(N=*)]")

FAMILIES = {
    # (SMARTS, index of the branch atom in the match, indices of the arm anchor atoms, branching k)
    "frechet": ("[CH2;!R]-c1cc(-O-[CH2]-c)cc(-O-[CH2]-c)c1", 1, (4, 9), 2),
    "frechet_345": ("[CH2;!R]-c1cc(-O-[CH2]-c)c(-O-[CH2]-c)c(-O-[CH2]-c)c1", 1, (4, 8, 12), 3),
    "percec": ("[CH2;!R]-c1cc(-O-[CH2]-[CH2])cc(-O-[CH2]-[CH2])c1", 1, (4, 9), 2),
    "percec_345": ("[CH2;!R]-c1cc(-O-[CH2]-[CH2])c(-O-[CH2]-[CH2])c(-O-[CH2]-[CH2])c1", 1, (4, 8, 12), 3),
    "percec_34": ("[CH2;!R]-c1ccc(-O-[CH2]-[CH2])c(-O-[CH2]-[CH2])c1", 1, (5, 9), 2),
    "bis_mpa": ("[CH3]-[CX4](-[CH2]-[OX2])(-[CH2]-[OX2])-C(=O)-[O,N]", 1, (2, 4), 2),
    "pamam": ("[NX3](-[CH2]-[CH2]-C(=O)-N)(-[CH2]-[CH2]-C(=O)-N)-[#6]", 0, (1, 6), 2),
    "newkome": ("[CX4](-[CH2]-[CH2]-C(=O)-[O,N])(-[CH2]-[CH2]-C(=O)-[O,N])(-[CH2]-[CH2]-C(=O)-[O,N])-[#6,#7,#8]", 0, (1, 6, 11), 3),
    # polyglycerol dendrons: arms end in benzyl (Frechet-type) or another glycerol /
    # acetonide unit (Haag-type); plain 1,3-dialkoxy/diaryloxy-2-propanols do not count
    "glycerol_bn": ("[CH;!R](-O)(-[CH2]-O-[CH2]-c)-[CH2]-O-[CH2]-c", 0, (2, 6), 2),
    # the outer glycerol's two O must both be capped (acetonide ring or further
    # glycerols): linear oligoglycerols with free CH-OH ends are not dendrons.
    # Recursive SMARTS, because an acetonide caps both O with the same carbon.
    "glycerol_pg": ("[CH;!R](-O)(-[CH2]-O-[CH2]-[CH;$(C(-O-[#6])-[CH2]-O-[#6])])-[CH2]-O-[CH2]-[CH;$(C(-O-[#6])-[CH2]-O-[#6])]", 0, (2, 6), 2),
}
_PATS = {k: (Chem.MolFromSmarts(s), b, a, kk) for k, (s, b, a, kk) in FAMILIES.items()}
MIN_ARM_ATOMS = {"bis_mpa": 2}
LONG_ALKYL = 8   # Percec periphery: alkoxy arms of >= 8 carbons


@dataclass
class DendronInfo:
    is_dendron: bool = False
    family: str = ""
    n_branch_units: int = 0
    generation: int = 0
    focal_alcohol: int | None = None     # atom index (in the given mol) of the focal OH
    focal_amine: int | None = None       # atom index of the focal NH
    notes: list = field(default_factory=list)


def _arm_atoms(mol, branch, anchor):
    """Heavy atoms reachable from anchor without passing through branch."""
    seen, stack = {branch, anchor}, [anchor]
    while stack:
        a = stack.pop()
        for n in mol.GetAtomWithIdx(a).GetNeighbors():
            if n.GetIdx() not in seen:
                seen.add(n.GetIdx())
                stack.append(n.GetIdx())
    seen.discard(branch)
    return seen


def _alkyl_carbons_beyond_o(mol, o_idx, ring_atoms):
    """Carbons in the chain hanging off an ether O (not entering the core ring)."""
    seen, stack, n_c = {o_idx} | ring_atoms, [o_idx], 0
    while stack:
        a = stack.pop()
        for n in mol.GetAtomWithIdx(a).GetNeighbors():
            j = n.GetIdx()
            if j in seen:
                continue
            seen.add(j)
            if n.GetAtomicNum() == 6:
                n_c += 1
            stack.append(j)
    return n_c


def _family_units(mol, ranks):
    units = {}
    for fam, (pat, b, arms, k) in _PATS.items():
        for m in mol.GetSubstructMatches(pat, uniquify=True):
            branch = m[b]
            anchors = [m[a] for a in arms]
            # identical arms: the arm anchors fall in one symmetry class
            if len({ranks[a] for a in anchors}) != 1:
                continue
            if fam.startswith("percec"):
                ring = set(mol.GetRingInfo().AtomRings()[0]) if mol.GetRingInfo().NumRings() else set()
                ring = next((set(r) for r in mol.GetRingInfo().AtomRings() if branch in r), ring)
                ether_o = [m[a] for a in arms]          # the arm anchors are the ether O atoms
                if min(_alkyl_carbons_beyond_o(mol, o, ring) for o in ether_o) < LONG_ALKYL:
                    continue
            need = MIN_ARM_ATOMS.get(fam, 0)
            if need and min(len(_arm_atoms(mol, branch, a)) for a in anchors) < need:
                continue
            units.setdefault(fam.split("_")[0] if fam != "bis_mpa" else fam, {}).setdefault(branch, k)
    return units


def _generic_nested(mol, ranks):
    """Symmetric branch points (>= 2 identical arms of >= 3 heavy atoms) nested in each other."""
    branch_pts = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in (6, 7) or atom.GetDegree() < 3:
            continue
        nb = [n.GetIdx() for n in atom.GetNeighbors() if not mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).IsInRing()]
        by_rank = {}
        for n in nb:
            by_rank.setdefault(ranks[n], []).append(n)
        for group in by_rank.values():
            if len(group) >= 2 and min(len(_arm_atoms(mol, atom.GetIdx(), a)) for a in group) >= 3:
                branch_pts[atom.GetIdx()] = group
                break
    key = lambda i: (mol.GetAtomWithIdx(i).GetAtomicNum(), mol.GetAtomWithIdx(i).GetDegree(),
                     mol.GetAtomWithIdx(i).IsInRing())
    nested = 0
    for b, arms in branch_pts.items():
        inner = _arm_atoms(mol, b, arms[0])
        if any(o in inner and key(o) == key(b) for o in branch_pts if o != b):
            nested += 1
    return branch_pts, nested


def _focal(mol, pattern, ranks):
    idx = [m[0] for m in mol.GetSubstructMatches(pattern)]
    if len(idx) == 1:
        return idx[0]
    classes = {}
    for i in idx:
        classes.setdefault(ranks[i], []).append(i)
    singles = [g[0] for g in classes.values() if len(g) == 1]
    return singles[0] if len(singles) == 1 else None


def analyse(mol: Chem.Mol) -> DendronInfo:
    info = DendronInfo()
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    units = _family_units(mol, ranks)
    if units:
        fam, pts = max(units.items(), key=lambda kv: len(kv[1]))
        k = max(pts.values())
        n = len(pts)
        info.is_dendron, info.family, info.n_branch_units = True, fam, n
        info.generation = max(1, int(math.floor(math.log(n * (k - 1) + 1, k) + 1e-9)))
    else:
        pts, nested = _generic_nested(mol, ranks)
        if nested >= 1:
            n = len(pts)
            info.is_dendron, info.family, info.n_branch_units = True, "generic_nested", n
            info.generation = max(2, int(math.floor(math.log(n + 1, 2) + 1e-9)))
    if info.is_dendron:
        info.focal_alcohol = _focal(mol, ALCOHOL, ranks)
        info.focal_amine = _focal(mol, AMINE, ranks)
    return info
