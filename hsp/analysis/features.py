"""
features.py

Shared featurisation for the data-driven HSP models: folded Morgan count
fingerprint, a small set of size-intensive RDKit descriptors, and the floored
Stefanis-Panayiotou prediction (Si->C substituted for Si compounds).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import sp_variants as sv  # noqa: E402

RDLogger.DisableLog("rdApp.*")
N_BITS = 2048
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_BITS)

# S-P settings carried into the features (best cheap rule set from grid_sp.py)
SP_RING, SP_W, SP_TRIGGER = "discount", 0.5, "either"


def mol_of(smiles):
    return Chem.MolFromSmiles(str(smiles))


def bitvect(mol):
    return _GEN.GetFingerprint(mol)


def count_fp(mol) -> np.ndarray:
    v = _GEN.GetCountFingerprintAsNumPy(mol).astype(np.float32)
    return np.log1p(v)


def descriptors(mol) -> np.ndarray:
    ha = max(mol.GetNumHeavyAtoms(), 1)
    ri = mol.GetRingInfo()
    return np.array([
        ha,
        Descriptors.MolWt(mol) / ha,
        Descriptors.MolLogP(mol) / ha,
        Descriptors.MolMR(mol) / ha,
        rdMolDescriptors.CalcTPSA(mol) / ha,
        rdMolDescriptors.CalcNumHBD(mol) / ha,
        rdMolDescriptors.CalcNumHBA(mol) / ha,
        rdMolDescriptors.CalcFractionCSP3(mol),
        ri.NumRings() / ha,
        rdMolDescriptors.CalcNumAliphaticRings(mol) / ha,
        rdMolDescriptors.CalcNumAromaticRings(mol) / ha,
        sum(a.GetAtomicNum() == 14 for a in mol.GetAtoms()) / ha,
        sum(a.GetAtomicNum() in (8, 7) for a in mol.GetAtoms()) / ha,
        sum(a.GetAtomicNum() in (9, 17, 35, 53) for a in mol.GetAtoms()) / ha,
    ], dtype=np.float32)


def sp_features(smiles) -> np.ndarray:
    o = sv.compute(smiles, SP_RING, SP_W, floor=True, trigger=SP_TRIGGER)
    if o.D is None:
        return np.full(5, np.nan, dtype=np.float32)
    return np.array([o.D, o.P, o.H, o.P_normal, o.H_normal], dtype=np.float32)


def generic_scaffold(mol) -> str:
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(core))
    except Exception:
        return ""


def featurise(smiles_list):
    mols = [mol_of(s) for s in smiles_list]
    X_fp = np.vstack([count_fp(m) for m in mols])
    X_desc = np.vstack([descriptors(m) for m in mols])
    X_sp = np.vstack([sp_features(s) for s in smiles_list])
    fps = [bitvect(m) for m in mols]
    scaf = [generic_scaffold(m) for m in mols]
    return X_fp, X_desc, X_sp, fps, scaf


def max_tanimoto(query_fps, ref_fps, k=1):
    """Top-k Tanimoto similarities and indices of each query against ref."""
    sims = np.zeros((len(query_fps), k), dtype=np.float32)
    idx = np.zeros((len(query_fps), k), dtype=np.int64)
    for i, q in enumerate(query_fps):
        s = np.asarray(DataStructs.BulkTanimotoSimilarity(q, ref_fps))
        top = np.argpartition(-s, min(k, len(s) - 1))[:k]
        top = top[np.argsort(-s[top])]
        sims[i], idx[i] = s[top], top
    return sims, idx
