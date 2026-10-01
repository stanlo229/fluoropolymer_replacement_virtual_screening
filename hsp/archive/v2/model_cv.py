"""
model_cv.py

Scaffold-grouped 5-fold CV of data-driven HSP models against the HSPiP
reference. Every model is trained on tiers 1+2 of the training folds and scored
on the held-out fold, split by tier, ring class and Si.

    SP          floored Stefanis-Panayiotou (best rule set from grid_sp.py)
    kNN         similarity-weighted mean of the 5 most similar references
    kNN_resid   SP + similarity-weighted mean residual (ref - SP) of 5 neighbours
    ET          ExtraTrees on [Morgan counts, descriptors, SP] -> (D, P, H)
    ET_resid    ExtraTrees on the same features -> (ref - SP)

A second experiment trains on tier 2 only and scores tier 1, to test whether
the 10k values (provenance unstated) agree with the curated master values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import features as ft  # noqa: E402

REF = HERE.parent / "reference"
OUT = HERE.parent / "results" / "benchmark" / "reference"
POLY = {"polycyclic_fused2", "polycyclic_bridged", "polycyclic_3plus"}
K = 5


def knn_predict(q_fps, r_fps, r_y, k=K):
    sims, idx = ft.max_tanimoto(q_fps, r_fps, k)
    w = np.clip(sims, 1e-3, None) ** 3                       # sharpen towards the closest
    pred = (r_y[idx] * w[..., None]).sum(1) / w.sum(1, keepdims=True)
    return pred, sims[:, 0]


def et():
    return ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.3,
                               n_jobs=16, random_state=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(REF / "hspip_reference.csv")
    ref = ref[(ref.n_carbon >= 1) & ~ref.charged].reset_index(drop=True)
    X_fp, X_desc, X_sp, fps, scaf = ft.featurise(ref.canonical_smiles)
    ok = ~np.isnan(X_sp).any(1)
    ref, X_fp, X_desc, X_sp = ref[ok].reset_index(drop=True), X_fp[ok], X_desc[ok], X_sp[ok]
    fps = [f for f, o in zip(fps, ok) if o]
    scaf = [s for s, o in zip(scaf, ok) if o]
    X = np.hstack([X_fp, X_desc, X_sp])
    Y = ref[["ref_D", "ref_P", "ref_H"]].to_numpy()
    SP = X_sp[:, :3]
    print("structures:", len(ref), " scaffolds:", len(set(scaf)))

    preds = {m: np.zeros_like(Y) for m in ("SP", "kNN", "kNN_resid", "ET", "ET_resid")}
    nn_sim = np.zeros(len(ref))
    for fold, (tr, te) in enumerate(GroupKFold(5).split(X, Y, scaf)):
        r_fps = [fps[i] for i in tr]
        q_fps = [fps[i] for i in te]
        preds["SP"][te] = SP[te]
        preds["kNN"][te], nn_sim[te] = knn_predict(q_fps, r_fps, Y[tr])
        res, _ = knn_predict(q_fps, r_fps, Y[tr] - SP[tr])
        preds["kNN_resid"][te] = SP[te] + res
        preds["ET"][te] = et().fit(X[tr], Y[tr]).predict(X[te])
        preds["ET_resid"][te] = SP[te] + et().fit(X[tr], Y[tr] - SP[tr]).predict(X[te])
        print(f"fold {fold} done")
    for m in preds:
        preds[m][:, 1:] = np.clip(preds[m][:, 1:], 0, None)   # no model may go negative

    ref["nn_sim"] = nn_sim
    ref["cls"] = np.where(ref.n_si > 0, "Si",
                  np.where(ref.ring_class.isin(POLY), "polycyclic_aliph", "other"))
    ref["sim_bin"] = pd.cut(ref.nn_sim, [0, .4, .6, .8, 1.01], labels=["<.4", ".4-.6", ".6-.8", ">.8"])
    rows = []
    for m, p in preds.items():
        e = p - Y
        ref[f"{m}_D"], ref[f"{m}_P"], ref[f"{m}_H"] = p.T
        era = np.sqrt(4 * e[:, 0] ** 2 + e[:, 1] ** 2 + e[:, 2] ** 2)
        for key in ("cls", "sim_bin"):
            for (tier, grp), idx in ref.groupby(["tier", key], observed=True).groups.items():
                i = np.asarray(idx)
                rows.append({"model": m, "tier": tier, "by": key, "group": grp, "n": len(i),
                             "MAE_D": np.abs(e[i, 0]).mean(), "MAE_P": np.abs(e[i, 1]).mean(),
                             "MAE_H": np.abs(e[i, 2]).mean(), "med_Ra_err": np.median(era[i])})
    summ = pd.DataFrame(rows).round(3)
    summ.to_csv(OUT / "model_cv_summary.csv", index=False)
    ref.to_csv(OUT / "model_cv_predictions.csv", index=False)
    pd.set_option("display.width", 200)
    for key in ("cls", "sim_bin"):
        t = summ[summ.by == key].sort_values(["tier", "group", "med_Ra_err"])
        print(f"\n===== by {key} =====")
        print(t.drop(columns="by").to_string(index=False))

    # tier-2-only training, scored on tier 1: do the 10k values transfer?
    t1, t2 = (ref.tier == 1).to_numpy(), (ref.tier == 2).to_numpy()
    p = SP[t1] + et().fit(X[t2], Y[t2] - SP[t2]).predict(X[t1])
    p[:, 1:] = np.clip(p[:, 1:], 0, None)
    e = p - Y[t1]
    print("\ntrain tier2 only -> tier1: MAE D/P/H",
          np.abs(e).mean(0).round(3), " (SP on tier1:", np.abs(SP[t1] - Y[t1]).mean(0).round(3), ")")


if __name__ == "__main__":
    main()
