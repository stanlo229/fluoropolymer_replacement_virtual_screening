"""
dispersion.py

Size-intensive dispersion component, after Mathieu (ACS Omega 2018): dD is
driven by the refractivity density R_D / V_m (the Lorentz-Lorenz term), which
cannot drift with molecule size the way the additive S-P sum does.

  V_m  : additive atom-environment volume model, fitted to HSPiP MVol (Fedors /
         McGowan-style additivity; volume is size-extensive, so additivity is
         physically appropriate here).
  R_D  : RDKit Crippen molar refractivity (additive).
  dD   = sqrt(a + b * x + c * x^2), x = R_D / V_m, fitted to tier-1 dD.

Validated with 5-fold scaffold-grouped CV. Writes results/benchmark/dispersion/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

RDLogger.DisableLog("rdApp.*")
HSP = Path(__file__).resolve().parents[1]
OUT = HSP / "results" / "benchmark" / "dispersion"


def atom_types(mol) -> dict:
    """Atom environment keys for the volume model: element, aromaticity,
    hybridisation, attached H count, ring membership."""
    mol = Chem.AddHs(mol)
    c = {}
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 1:
            k = "H"
        else:
            k = (f"{a.GetSymbol()}{'a' if a.GetIsAromatic() else ''}"
                 f"_{str(a.GetHybridization())[-3:]}_H{a.GetTotalNumHs()}")
        c[k] = c.get(k, 0) + 1
    ri = mol.GetRingInfo()
    c["_rings"] = ri.NumRings()
    return c


class VolumeModel:
    def fit(self, mols, vm):
        rows = [atom_types(m) for m in mols]
        self.cols = sorted({k for r in rows for k in r})
        X = self._X(rows)
        self.m = Ridge(alpha=1e-3, fit_intercept=False).fit(X, vm)
        return self

    def _X(self, rows):
        return np.array([[r.get(k, 0) for k in self.cols] for r in rows], float)

    def predict(self, mols):
        rows = [atom_types(m) for m in mols]
        unk = [sum(v for k, v in r.items() if k not in self.cols) for r in rows]
        return self.m.predict(self._X(rows)), np.array(unk)


def _dd(p, x):
    return np.sqrt(np.clip(p[0] + p[1] * x + p[2] * x * x, 1e-6, None))


def fit_dd(x, y):
    return least_squares(lambda p: _dd(p, x) - y, [100, 500, 500]).x


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(HSP / "reference" / "hspip_reference.csv")
    ref = ref[~ref.charged].copy()
    ref["mol"] = [Chem.MolFromSmiles(s) for s in ref.canonical_smiles]
    ref = ref[ref.mol.notna()].copy()
    ref["MVol"] = pd.to_numeric(ref.MVol, errors="coerce")
    ref["RD"] = [Crippen.MolMR(m) for m in ref.mol]
    from rdkit.Chem.Scaffolds import MurckoScaffold
    ref["scaf"] = [Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(
        MurckoScaffold.GetScaffoldForMol(m))) if m.GetNumAtoms() else "" for m in ref.mol]

    # ---- volume model CV (fit on all tiers' MVol: densities are measured or
    # well-estimated; score on tier 1) ----
    vref = ref[ref.MVol.notna() & (ref.MVol > 0)].reset_index(drop=True)
    gkf = GroupKFold(5)
    vm_cv = np.full(len(vref), np.nan)
    for tr, te in gkf.split(vref, groups=vref.scaf):
        vmod = VolumeModel().fit(list(vref.mol.iloc[tr]), vref.MVol.iloc[tr].to_numpy())
        vm_cv[te] = vmod.predict(list(vref.mol.iloc[te]))[0]
    vref["Vm_cv"] = vm_cv
    rel = (vref.Vm_cv / vref.MVol - 1).abs()
    print("volume model CV |rel err| median %.3f p90 %.3f (tier1 median %.3f)" % (
        rel.median(), rel.quantile(.9), rel[vref.tier == 1].median()))

    # ---- dD model: fit on tier 1 using CV-estimated volumes (so the error of
    # the volume estimate is inside the validation) ----
    vref["x"] = vref.RD / vref.Vm_cv
    t1 = vref[vref.tier == 1].reset_index(drop=True)
    pred = np.full(len(t1), np.nan)
    for tr, te in GroupKFold(5).split(t1, groups=t1.scaf):
        p = fit_dd(t1.x.iloc[tr].to_numpy(), t1.ref_D.iloc[tr].to_numpy())
        pred[te] = _dd(p, t1.x.iloc[te].to_numpy())
    t1["dD_rd"] = pred
    sp = pd.read_csv(HSP / "results" / "benchmark" / "reference" / "model_cv_predictions.csv",
                     usecols=["ikey14", "SP_D", "ET_D", "ET_resid_D"])
    t1 = t1.merge(sp, on="ikey14", how="left")
    t1["ENS_D"] = (t1.ET_D + t1.ET_resid_D) / 2
    rows = []
    for name, k in [("all", t1.n_si >= 0), ("non-Si", t1.n_si == 0), ("Si", t1.n_si > 0),
                    (">=20 heavy atoms", t1.n_heavy >= 20),
                    ("aliph polycyclic", t1.aliph_polycyclic)]:
        x = t1[k]
        r = {"subset": name, "n": len(x)}
        for m in ["dD_rd", "SP_D", "ENS_D"]:
            r["MAE_" + m] = (x[m] - x.ref_D).abs().mean()
        rows.append(r)
    tab = pd.DataFrame(rows).round(3)
    print(tab.to_string(index=False))
    tab.to_csv(OUT / "dD_cv_tier1.csv", index=False)

    # final models on all data
    vmod = VolumeModel().fit(list(vref.mol), vref.MVol.to_numpy())
    p = fit_dd(t1.x.to_numpy(), t1.ref_D.to_numpy())
    print("dD = sqrt(%.2f + %.2f x + %.2f x^2), x = R_D/V_m" % tuple(p))
    return vmod, p


if __name__ == "__main__":
    main()
