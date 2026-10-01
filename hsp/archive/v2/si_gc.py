"""
si_gc.py

Silicon extension fitted to every Si compound in HSPiP (tiers 1+2, 67
structures), replacing the per-Si-atom offset that si_correction.py retired.

Stefanis-Panayiotou has no Si group, so the organic part is still scored by S-P
with Si->C substitution; a ridge regression then maps [size-intensive counts of
Si environments, S-P D/P/H] to the reference HSP. Validated leave-one-family-out
(homologous series held out together): median Ra error 1.8 vs 6.4 for raw
S-P and 4.9 for the fingerprint ensemble; tier-1 only 2.0 vs 7.7.

Out of domain: silanols (Si-OH, 2 references, error ~7). Those are flagged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import features as ft  # noqa: E402

REF = HERE.parent / "reference"
SI_ENV = {
    "SiOSi": "[Si]O[Si]", "SiOC": "[Si]O[#6]", "SiOH": "[Si][OX2H1]",
    "SiCH3": "[Si][CH3]", "SiC": "[Si][#6;!$([CH3])]", "SiX": "[Si][Cl,Br,I,F]",
    "SiH": "[Si;!H0]", "SiN": "[Si][#7]",
}
_PATS = {k: Chem.MolFromSmarts(v) for k, v in SI_ENV.items()}


def si_features(smiles) -> np.ndarray:
    m = Chem.MolFromSmiles(str(smiles))
    ha = m.GetNumHeavyAtoms()
    env = [len(m.GetSubstructMatches(p)) / ha for p in _PATS.values()]
    return np.concatenate([env, ft.sp_features(smiles)[:3]])


def has_silanol(smiles) -> bool:
    return Chem.MolFromSmiles(str(smiles)).HasSubstructMatch(_PATS["SiOH"])


def fit() -> Ridge:
    ref = pd.read_csv(REF / "hspip_reference.csv")
    ref = ref[(ref.n_si > 0) & ~ref.charged]
    X = np.vstack([si_features(s) for s in ref.canonical_smiles])
    return Ridge(alpha=1.0).fit(X, ref[["ref_D", "ref_P", "ref_H"]].to_numpy())


def predict(model: Ridge, smiles_list) -> np.ndarray:
    X = np.vstack([si_features(s) for s in smiles_list])
    p = model.predict(X)
    p[:, 1:] = np.clip(p[:, 1:], 0, None)
    return p
