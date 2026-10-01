"""
gc_benchmark.py

Scaffold-grouped 5-fold CV, identical folds for every model:

  SP_paper     published Stefanis-Panayiotou (hsp_calculator, fixed), floored at 0
  SP_refit     formula A (2012 equations, group values re-fitted, prior = paper)
  Fedors_refit formula B (size-intensive dk = sqrt(sum E / sum V), same groups)
  XGB_groups   XGBoost on the same group counts the formulas use (what the
               group description could achieve with a non-additive model)
  XGB_fp       XGBoost on Morgan counts + descriptors + S-P (features.py)
  ET_fp        ExtraTrees on the same features (the v2 model)

Grid over ridge strength and master-set weight for the two formulas; the best
setting by tier-1 median Ra error is refitted on all data and its group tables
are written to results/benchmark/reference/gc_tables/.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / ".pylib_xgb"), str(HERE.parent), str(HERE)]
import xgboost as xgb  # noqa: E402
import features as ft  # noqa: E402
import gc_refit as gr  # noqa: E402
import hsp_calculator as hc  # noqa: E402

REF = HERE.parent / "reference"
OUT = HERE.parent / "results" / "benchmark" / "reference"
TAB = OUT / "gc_tables"
POLY = {"polycyclic_fused2", "polycyclic_bridged", "polycyclic_3plus"}
LAMS = [0.3, 3.0, 30.0]
W_MASTER = [1.0, 5.0]


def ra_err(P, Y):
    e = P - Y
    return np.sqrt(4 * e[:, 0] ** 2 + e[:, 1] ** 2 + e[:, 2] ** 2)


def xgb_fit_predict(Xtr, Ytr, wtr, Xte):
    out = []
    for j in range(3):
        m = xgb.XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=6,
                             subsample=0.8, colsample_bytree=0.5, min_child_weight=2,
                             tree_method="hist", n_jobs=16, random_state=0)
        m.fit(Xtr, Ytr[:, j], sample_weight=wtr)
        out.append(m.predict(Xte))
    return np.column_stack(out)


def load():
    ref = pd.read_csv(REF / "hspip_reference.csv")
    ref = ref[(ref.n_carbon >= 3) & ~ref.charged].reset_index(drop=True)
    # organics only (S-P scope) plus Si/P/B; drops organometallics and salts
    from rdkit import Chem
    allowed = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}
    ref = ref[[{a.GetAtomicNum() for a in Chem.MolFromSmiles(s).GetAtoms()} <= allowed
               for s in ref.canonical_smiles]].reset_index(drop=True)
    cache = OUT / "_gc_counts.pkl"
    if cache.exists():
        counts = pickle.loads(cache.read_bytes())
    else:
        counts = [gr.raw_counts(s) for s in ref.canonical_smiles]
        cache.write_bytes(pickle.dumps(counts))
    ok = [c is not None for c in counts]
    ref = ref[ok].reset_index(drop=True)
    counts = [c for c in counts if c is not None]
    return ref, counts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    ref, counts = load()
    space = gr.GroupSpace.build(counts)
    Xg = space.matrix(counts)
    Y = ref[["ref_D", "ref_P", "ref_H"]].to_numpy()
    V = pd.to_numeric(ref.MVol, errors="coerce").to_numpy()
    X_fp, X_desc, X_sp, _, scaf = ft.featurise(ref.canonical_smiles)
    Xf = np.nan_to_num(np.hstack([X_fp, X_desc, X_sp]))
    sp = []
    for s in ref.canonical_smiles:
        r = hc.compute_hsp(s)
        sp.append([r.delta_D, max(r.delta_P, 0), max(r.delta_H, 0)] if r.delta_D is not None
                  else [np.nan] * 3)
    SPp = np.array(sp, dtype=float)
    print(f"structures {len(ref)}  groups {len(space.columns)} "
          f"(new {sum(c.startswith('NEW:') for c in space.columns)}, "
          f"Si {sum(c.startswith('SI:') for c in space.columns)})  scaffolds {len(set(scaf))}",
          flush=True)

    preds: dict[str, np.ndarray] = {}

    def slot(name):
        return preds.setdefault(name, np.full_like(Y, np.nan, dtype=float))

    folds = list(GroupKFold(5).split(Xg, Y, scaf))
    for f, (tr, te) in enumerate(folds):
        slot("SP_paper")[te] = SPp[te]
        for wm in W_MASTER:
            w = np.where(ref.tier == 1, wm, 1.0)
            for lam in LAMS:
                slot(f"SP_refit_l{lam}_w{wm}")[te] = gr.SPRefit(space.columns, lam).fit(
                    Xg[tr], Y[tr], w[tr]).predict(Xg[te])
                slot(f"Fedors_refit_l{lam}_w{wm}")[te] = gr.FedorsRefit(space.columns, lam).fit(
                    Xg[tr], Y[tr], w[tr], V[tr]).predict(Xg[te])
        w = np.where(ref.tier == 1, 5.0, 1.0)
        slot("XGB_groups")[te] = xgb_fit_predict(Xg[tr], Y[tr], w[tr], Xg[te])
        slot("XGB_fp")[te] = xgb_fit_predict(Xf[tr], Y[tr], w[tr], Xf[te])
        et = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.3,
                                 n_jobs=16, random_state=0).fit(Xf[tr], Y[tr], sample_weight=w[tr])
        slot("ET_fp")[te] = et.predict(Xf[te])
        print(f"fold {f} done", flush=True)
    for p in preds.values():
        p[:, 1:] = np.clip(p[:, 1:], 0, None)

    cls = np.where(ref.n_si > 0, "Si", np.where(ref.ring_class.isin(POLY), "polycyclic_aliph",
                   np.where(ref.n_heavy > 30, ">30 heavy atoms", "other")))
    rows = []
    for m, p in preds.items():
        era = ra_err(p, Y)
        for tier in (1, 2):
            for grp in ("all", "other", "polycyclic_aliph", "Si", ">30 heavy atoms"):
                k = (ref.tier == tier).to_numpy() & ((cls == grp) if grp != "all" else True)
                k &= ~np.isnan(p).any(1)
                if not k.any():
                    continue
                e = np.abs(p[k] - Y[k])
                rows.append({"model": m, "tier": tier, "subset": grp, "n": int(k.sum()),
                             "MAE_D": e[:, 0].mean(), "MAE_P": e[:, 1].mean(),
                             "MAE_H": e[:, 2].mean(), "med_Ra_err": np.median(era[k])})
    summ = pd.DataFrame(rows).round(3)
    summ.to_csv(OUT / "gc_benchmark_summary.csv", index=False)

    # best formula settings by tier-1 'all' median Ra error
    t1 = summ[(summ.tier == 1) & (summ.subset == "all")]
    best = {}
    for form in ("SP_refit", "Fedors_refit"):
        r = t1[t1.model.str.startswith(form)].sort_values("med_Ra_err").iloc[0]
        lam, wm = r.model.split("_l")[1].split("_w")
        best[form] = (float(lam), float(wm), r.model)
    keep = ["SP_paper", best["SP_refit"][2], best["Fedors_refit"][2], "XGB_groups", "XGB_fp", "ET_fp"]

    # out-of-fold predictions of the compared models (contains HSPiP values:
    # stays in the gitignored benchmark folder)
    oof = ref[["ikey14", "Name", "tier", "canonical_smiles", "n_heavy", "n_si",
               "ring_class", "ref_D", "ref_P", "ref_H", "MVol"]].copy()
    oof["fold"] = -1
    for f, (_, te) in enumerate(folds):
        oof.loc[te, "fold"] = f
    for m in keep:
        short = m.split("_l")[0]
        oof[[f"{short}_D", f"{short}_P", f"{short}_H"]] = preds[m]
    oof.to_csv(OUT / "gc_oof_predictions.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n== grid (tier 1, all) ==")
    print(t1[t1.model.str.contains("refit")].sort_values("med_Ra_err").to_string(index=False))
    print("\n== comparison, best settings ==")
    print(summ[summ.model.isin(keep)].sort_values(["tier", "subset", "med_Ra_err"]).to_string(index=False))

    # final fits on all data -> group tables
    for form, (lam, wm, _) in best.items():
        w = np.where(ref.tier == 1, wm, 1.0)
        n_cmpd = (Xg > 0).sum(0)
        if form == "SP_refit":
            m = gr.SPRefit(space.columns, lam).fit(Xg, Y, w)
            t, consts = m.table()
            t["n_compounds"] = n_cmpd
            t.to_csv(TAB / "sp_refit_groups.csv", index=False)
            json.dump({"lambda": lam, "w_master": wm, "constants": consts},
                      open(TAB / "sp_refit_meta.json", "w"), indent=2)
        else:
            m = gr.FedorsRefit(space.columns, lam).fit(Xg, Y, w, V)
            t = m.table()
            t["n_compounds"] = n_cmpd
            t.to_csv(TAB / "fedors_refit_groups.csv", index=False)
            json.dump({"lambda": lam, "w_master": wm}, open(TAB / "fedors_refit_meta.json", "w"), indent=2)
        pickle.dump(m, open(TAB / f"{form}.pkl", "wb"))
    pickle.dump(space, open(TAB / "group_space.pkl", "wb"))
    print("\nbest:", best)


if __name__ == "__main__":
    main()
