"""
predict_monomers.py

Final HSP assignment for the monomer library, in order of reliability:

  1. exact structure in HSPiP (tier 1, then tier 2)      -> database value
  2. otherwise a size-aware model trained on all HSPiP structures -> model value
       <= 30 heavy atoms: ENS = mean of ExtraTrees(direct) and ExtraTrees(S-P residual)
        > 30 heavy atoms: ExtraTrees(direct) only. S-P is not size-intensive
          (its dD sum grows with molecule size), so the residual model inherits
          that drift; on held-out >30-heavy-atom HSPiP structures the direct
          model has MAE 0.79/1.47/1.52 vs ENS 1.04/1.59/1.99.
  3. raw S-P (floored) is kept alongside for reference

Every value is >= 0 by construction. Each row carries:
    nn_sim      Tanimoto to the most similar HSPiP structure (applicability)
    nn_name     that structure's name
    ET_*, RES_* the two model outputs, kept for inspection
    confidence  high (nn_sim >= 0.6) / medium (0.4-0.6) / low (< 0.4)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import features as ft  # noqa: E402

REF = HERE.parent / "reference"
RES = HERE.parent / "results"
OUT = RES / "benchmark" / "reference"


def et():
    return ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.3,
                               n_jobs=16, random_state=0)


SIZE_SWITCH = 30   # heavy atoms above which only the direct model is used


def fit_models():
    ref = pd.read_csv(REF / "hspip_reference.csv")
    ref = ref[~ref.charged].reset_index(drop=True)
    X_fp, X_desc, X_sp, fps, _ = ft.featurise(ref.canonical_smiles)
    ok = ~np.isnan(X_sp).any(1)
    ref = ref[ok].reset_index(drop=True)
    X = np.hstack([X_fp, X_desc, X_sp])[ok]
    Y = ref[["ref_D", "ref_P", "ref_H"]].to_numpy()
    SP = X_sp[ok, :3]
    m_dir = et().fit(X, Y)
    m_res = et().fit(X, Y - SP)
    return ref, [f for f, o in zip(fps, ok) if o], m_dir, m_res


def predict(smiles, ref, ref_fps, m_dir, m_res, chunk=5000):
    rows = []
    for s0 in range(0, len(smiles), chunk):
        smi = list(smiles[s0:s0 + chunk])
        X_fp, X_desc, X_sp, fps, _ = ft.featurise(smi)
        X = np.nan_to_num(np.hstack([X_fp, X_desc, X_sp]))
        SP = X_sp[:, :3]
        p_dir = m_dir.predict(X)
        p_res = SP + m_res.predict(X)
        large = (X_desc[:, 0] > SIZE_SWITCH)[:, None]     # column 0 = heavy atoms
        ens = np.where(large, p_dir, (p_dir + p_res) / 2)
        ens[:, 1:] = np.clip(ens[:, 1:], 0, None)
        sims, idx = ft.max_tanimoto(fps, ref_fps, 1)
        for i, s in enumerate(smi):
            rows.append({"monomer_smiles": s,
                         "D": ens[i, 0], "P": ens[i, 1], "H": ens[i, 2],
                         "model": "ET_direct" if large[i, 0] else "ENS",
                         "ET_D": p_dir[i, 0], "ET_P": p_dir[i, 1], "ET_H": p_dir[i, 2],
                         "RES_D": p_res[i, 0], "RES_P": p_res[i, 1], "RES_H": p_res[i, 2],
                         "SP_D": SP[i, 0], "SP_P": SP[i, 1], "SP_H": SP[i, 2],
                         "nn_sim": sims[i, 0], "nn_name": ref.Name.iloc[idx[i, 0]],
                         "nn_ikey14": ref.ikey14.iloc[idx[i, 0]]})
        print(f"  {s0 + len(smi)}/{len(smiles)}", flush=True)
    out = pd.DataFrame(rows)
    out["confidence"] = pd.cut(out.nn_sim, [-1, 0.4, 0.6, 2], labels=["low", "medium", "high"])
    out["hsp_source"] = "model_" + out.model
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref, ref_fps, m_dir, m_res = fit_models()
    print("trained on", len(ref), "structures", flush=True)
    mono = pd.read_csv(RES / "legacy_sp" / "monomers_hsp.csv", low_memory=False)
    pred = predict(mono.monomer_smiles.tolist(), ref, ref_fps, m_dir, m_res)

    # exact database hits override the model
    keys = pd.read_csv(OUT / "_monomer_keys.csv")
    pred = pred.merge(keys, on="monomer_smiles", how="left")
    hit = pred.merge(ref[["ikey14", "ref_D", "ref_P", "ref_H", "tier"]], on="ikey14", how="left")
    m = hit.ref_D.notna().to_numpy()
    pred.loc[m, ["D", "P", "H"]] = hit.loc[m, ["ref_D", "ref_P", "ref_H"]].to_numpy()
    pred.loc[m, "hsp_source"] = "HSPiP_tier" + hit.loc[m, "tier"].astype(int).astype(str)
    print("exact database hits:", int(m.sum()))

    full = mono.merge(pred.drop(columns="ikey14"), on="monomer_smiles", how="left")
    full.to_csv(RES / "v2" / "monomers_hsp_ens.csv", index=False)
    print(full.confidence.value_counts().to_string())
    print(full.groupby("has_si").confidence.value_counts().unstack().to_string())


if __name__ == "__main__":
    main()
