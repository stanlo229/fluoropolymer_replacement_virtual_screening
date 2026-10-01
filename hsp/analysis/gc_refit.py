"""
gc_refit.py

Group-contribution HSP formulas whose group values are re-fitted ("deconvoluted")
on HSPiP, using the Stefanis-Panayiotou (2008/2012) functional groups.

Groups
  FO:<name>   the 76 first-order groups of Table A.1 (hsp_calculator SMARTS,
              priority order, no atom reuse)
  SO:<name>   the 37 second-order groups of Table A.2 (hsp_calculator
              identification, including the worked-example fixes)
  SI:<name>   Si environment second-order groups (no S-P equivalent)
  NEW:<key>   heavy atoms no first-order group covers (Si, sulfoxide S, amide N,
              P, ...), typed by element/aromaticity/degree/H count; kept as a
              group when seen in >= MIN_NEW compounds, else pooled per element

Formula A, "sp" (the exact 2012 equations, Eqs. A.2-A.6):
    dD = (sum N_i C_i + sum M_j D_j + c_D)^0.4126
    dP = sum N_i C_i + sum M_j D_j + c_P          (low-value table if < 3)
    dH = sum N_i C_i + sum M_j D_j + c_H          (low-value table if < 3)
  Fitted by ridge regression shrunk towards the published value of every group
  (*** cells and new groups shrink towards 0), so a group with little HSPiP
  support keeps the paper's number. The main tables are fitted on compounds with
  reference dP / dH >= 3, the low-value tables on those below 3, matching the
  paper's validity statement. dD is fitted on the paper's transformed scale,
  weighted so the residuals approximate errors in dD itself.

Formula B, "fedors" (size-intensive, Fedors / Hoftyzer-van Krevelen style):
    V   = sum N_i v_i                      (molar volume, fitted to HSPiP MVol)
    E_k = sum N_i e_k,i                    (cohesive energy of component k)
    d_k = sqrt(max(E_k, 0) / V)
  e_k is fitted to E_k = d_k(ref)^2 * MVol(ref). A ratio of two additive sums
  cannot drift with molecule size, and no low-value branch is needed.

Both formulas are deterministic closed-form functions of the group counts; the
fitted tables are written out as CSV.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import hsp_calculator as hc  # noqa: E402
import sp_variants as sv  # noqa: E402
from group_tables import (CONST_D, POWER_D, CONST_P, CONST_HB, CONST_P_LOW,  # noqa: E402
                          CONST_HB_LOW, THRESHOLD_LOW, FIRST_ORDER_GROUPS,
                          SECOND_ORDER_GROUPS, LOW_VALUE_CORRECTIONS,
                          LOW_VALUE_2ND_ORDER)

RDLogger.DisableLog("rdApp.*")
MIN_NEW = 10
SI_SO = {
    "Si-O-Si": "[Si]O[Si]", "Si-O-C": "[Si]O[#6]", "Si-OH": "[Si][OX2H1]",
    "Si-CH3": "[Si][CH3]", "Si-X": "[Si][F,Cl,Br,I]", "Si-N": "[Si][#7]",
}
_SI_PATS = {k: Chem.MolFromSmarts(v) for k, v in SI_SO.items()}


# ---------------------------------------------------------------------------
# Group counting
# ---------------------------------------------------------------------------
def _atom_key(a) -> str:
    # e.g. "N.ar_D2_H0" = aromatic N with 2 neighbours and no H
    return (f"{a.GetSymbol()}{'.ar' if a.GetIsAromatic() else ''}"
            f"_D{a.GetDegree()}_H{a.GetTotalNumHs()}")


def raw_counts(smiles: str, ring_mode: str = "per_ring") -> dict[str, float] | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fo, used = hc._count_first_order(mol)          # no Si->C substitution
    so = hc._count_second_order(mol, fo)
    so = {k: v for k, v in so.items() if k not in sv.RING_GROUPS.values()}
    so.update(sv.ring_counts(mol, ring_mode))
    c = {f"FO:{k}": float(v) for k, v in fo.items()}
    c.update({f"SO:{k}": float(v) for k, v in so.items() if v})
    if any(a.GetAtomicNum() == 14 for a in mol.GetAtoms()):
        for k, p in _SI_PATS.items():
            n = len(mol.GetSubstructMatches(p))
            if n:
                c[f"SI:{k}"] = float(n)
    for a in mol.GetAtoms():
        if a.GetAtomicNum() > 1 and a.GetIdx() not in used:
            k = f"NEW:{_atom_key(a)}"
            c[k] = c.get(k, 0.0) + 1.0
    return c


@dataclass
class GroupSpace:
    """Fixed column order; rare NEW atom types pooled per element."""
    columns: list[str] = field(default_factory=list)
    pool: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(cls, count_dicts):
        seen: dict[str, int] = {}
        for c in count_dicts:
            for k in c:
                seen[k] = seen.get(k, 0) + 1
        pool, cols = {}, set()
        for k, n in seen.items():
            if k.startswith("NEW:") and n < MIN_NEW:
                pool[k] = "NEW:other_" + k[4:].split("_")[0].split(".")[0]
                cols.add(pool[k])
            else:
                cols.add(k)
        # every published group gets a column even if HSPiP never shows it
        cols |= {f"FO:{n}" for n, *_ in FIRST_ORDER_GROUPS}
        cols |= {f"SO:{n}" for n, *_ in SECOND_ORDER_GROUPS}
        return cls(sorted(cols), pool)

    def matrix(self, count_dicts) -> np.ndarray:
        idx = {k: i for i, k in enumerate(self.columns)}
        X = np.zeros((len(count_dicts), len(self.columns)))
        for r, c in enumerate(count_dicts):
            for k, v in c.items():
                k = self.pool.get(k, k)
                if k in idx:
                    X[r, idx[k]] += v
        return X


def published_priors(columns: list[str]) -> dict[str, np.ndarray]:
    """Paper values per column (0 for *** cells and new groups)."""
    fo = {n: (d, p, h) for n, _, d, p, h in FIRST_ORDER_GROUPS}
    so = {n: (d, p, h) for n, d, p, h in SECOND_ORDER_GROUPS}
    pri = {k: np.zeros(len(columns)) for k in ("D", "P", "H", "P_low", "H_low")}
    for i, col in enumerate(columns):
        kind, name = col.split(":", 1)
        tab, low = ({"FO": fo, "SO": so}.get(kind),
                    {"FO": LOW_VALUE_CORRECTIONS, "SO": LOW_VALUE_2ND_ORDER}.get(kind))
        if tab and name in tab:
            for j, k in enumerate("DPH"):
                pri[k][i] = tab[name][j] or 0.0
        if low and name in low:
            pri["P_low"][i] = low[name][0] or 0.0
            pri["H_low"][i] = low[name][1] or 0.0
    return pri


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------
def ridge_to_prior(X, y, w, prior, lam, intercept_prior=None):
    """min sum w (y - Xb - c)^2 + lam ||b - prior||^2 (+ lam_c (c - c0)^2 weak)."""
    if intercept_prior is not None:
        X1 = np.hstack([X, np.ones((len(X), 1))])
        p1 = np.append(prior, intercept_prior)
        pen = np.full(X1.shape[1], lam)
        pen[-1] = lam * 1e-3                 # intercept nearly free
    else:
        X1, p1, pen = X, prior, np.full(X.shape[1], lam)
    r = y - X1 @ p1
    sw = np.sqrt(w)
    A = (X1 * sw[:, None])
    G = A.T @ A + np.diag(pen)
    g = np.linalg.solve(G, A.T @ (r * sw))
    b = p1 + g
    return (b[:-1], b[-1]) if intercept_prior is not None else (b, 0.0)


@dataclass
class SPRefit:
    columns: list[str]
    lam: float = 1.0
    coef: dict = field(default_factory=dict)      # key -> (beta, intercept)

    def fit(self, X, Y, w):
        pri = published_priors(self.columns)
        dD = Y[:, 0]
        T = np.clip(dD, 1e-3, None) ** (1 / POWER_D)
        dTd = (1 / POWER_D) * np.clip(dD, 1e-3, None) ** (1 / POWER_D - 1)
        # residuals are weighted to dD units, so rescale the penalty by the same
        # factor: lam then means the same thing for every component
        s = np.median(dTd)
        self.coef["D"] = ridge_to_prior(X, T, w / dTd ** 2, pri["D"], self.lam / s ** 2, CONST_D)
        for j, k, c0, c0_low in ((1, "P", CONST_P, CONST_P_LOW), (2, "H", CONST_HB, CONST_HB_LOW)):
            hi = Y[:, j] >= THRESHOLD_LOW
            self.coef[k] = ridge_to_prior(X[hi], Y[hi, j], w[hi], pri[k], self.lam, c0)
            self.coef[k + "_low"] = ridge_to_prior(X[~hi], Y[~hi, j], w[~hi],
                                                   pri[k + "_low"], self.lam, c0_low)
        return self

    def predict(self, X):
        def lin(k):
            b, c = self.coef[k]
            return X @ b + c
        D = np.clip(lin("D"), 0, None) ** POWER_D
        out = [D]
        for k in ("P", "H"):
            v = lin(k)
            low = v < THRESHOLD_LOW
            v = np.where(low, lin(k + "_low"), v)
            out.append(np.clip(v, 0, None))
        return np.column_stack(out)

    def table(self) -> pd.DataFrame:
        pri = published_priors(self.columns)
        t = pd.DataFrame({"group": self.columns})
        for k in ("D", "P", "H", "P_low", "H_low"):
            t[f"{k}_fit"] = self.coef[k][0]
            t[f"{k}_paper"] = pri[k]
        consts = {k: self.coef[k][1] for k in self.coef}
        return t, consts


@dataclass
class FedorsRefit:
    columns: list[str]
    lam: float = 1.0
    coef: dict = field(default_factory=dict)

    def fit(self, X, Y, w, V):
        z = np.zeros(X.shape[1])
        self.coef["V"] = ridge_to_prior(X, V, w, z, self.lam * 1e-2)[0]
        for j, k in enumerate("DPH"):
            E = Y[:, j] ** 2 * V
            dEd = 2 * np.clip(Y[:, j], 1.0, None) * V      # residuals ~ error in d_k
            s = np.median(dEd)
            self.coef[k] = ridge_to_prior(X, E, w / dEd ** 2, z, self.lam / s ** 2)[0]
        return self

    def volume(self, X):
        return np.clip(X @ self.coef["V"], 1.0, None)

    def predict(self, X):
        V = self.volume(X)
        return np.column_stack([np.sqrt(np.clip(X @ self.coef[k], 0, None) / V) for k in "DPH"])

    def table(self) -> pd.DataFrame:
        t = pd.DataFrame({"group": self.columns, "v_cm3mol": self.coef["V"]})
        for k in "DPH":
            t[f"E_{k}_Jmol"] = self.coef[k]
        return t
