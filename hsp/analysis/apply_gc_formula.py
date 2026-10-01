"""
apply_gc_formula.py

v3: HSP for the monomer library from the refitted size-intensive group-
contribution formula (gc_refit.FedorsRefit, tables from gc_benchmark.py):

    d_k = sqrt( sum_i N_i e_k,i / sum_i N_i v_i ),  k = D, P, H

using the Stefanis-Panayiotou groups plus the added Si / new-atom groups.
The refitted 2012-form formula (SPRefit) and the published S-P are kept
alongside for comparison.

Coverage: atoms that fall into a pooled rare group (seen in < MIN_NEW HSPiP
compounds) are counted in n_unmatched_atoms, which rank_top_monomers.py's C4
rule reads.

Outputs
  results/v3/monomers_hsp_formula.csv     full table
  results/v3/monomers_hsp_corrected.csv   downstream input, non-Si monomers only
  results/v3/si/monomers_hsp_corrected.csv  downstream input, Si monomers only
  results/v3/si/si_monomers.csv           all Si monomers with v3 values

Si monomers are scored, but ranked SEPARATELY (results/v3/si/) because their
values may not be reliable; see results/v3/si/README.md. Every Si row carries
hsp_note and expected_Ra_err = NaN in monomers_hsp_formula.csv.
  results/manual/2026-09-30/hsp6_v3.csv   the six requested monomers
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import gc_refit as gr  # noqa: E402,F401  (needed to unpickle the models)
from solvent_incompatibility import REFERENCES  # noqa: E402

RES = HERE.parent / "results"
TAB = RES / "benchmark" / "reference" / "gc_tables"
V3 = RES / "v3"
SI_NOTE = ("Si monomer: HSP NOT reliable (Si groups fitted on 67 HSPiP Si compounds, "
           "only 7 curated, all much smaller than the monomers)")
SIX = {
    "1 cholesteryl NB ester": "CC(CCC[C@H]([C@@]1([H])CC[C@]2([H])[C@]1(C)CC[C@@]3([H])[C@@]2([H])CC[C@]4([H])[C@]3(C)CC[C@H](OC(C5CC6C=CC5C6)=O)C4)C)C",
    "2 bis(2-ethylhexyl) NB diester": "O=C(C1C(C(OCC(CC)CCCC)=O)C2C=CC1C2)OCC(CC)CCCC",
    "3 bis(2-hydroxyethyl) NB diester": "O=C(C1C(C(OCCO)=O)C2C=CC1C2)OCCO",
    "4 bis(2-cyanoethyl) NB diester": "O=C(C1C(C(OCCC#N)=O)C2C=CC1C2)OCCC#N",
    "5 tert-butyl NB ester": "O=C(OC(C)(C)C)C1C2C=CC(C2)C1",
    "6 2-hydroxyethyl NB ester": "O=C(OCCO)C1C2C=CC(C2)C1",
}


def ra(D, P, H, ref):
    rD, rP, rH = ref
    return np.sqrt(4 * (D - rD) ** 2 + (P - rP) ** 2 + (H - rH) ** 2)


def score(smiles, space, fed, spr):
    counts = [gr.raw_counts(s) or {} for s in smiles]
    X = space.matrix(counts)
    cols = set(space.columns)
    pooled = [sum(v for k, v in c.items()
                  if k.startswith("NEW:") and (space.pool.get(k, k).startswith("NEW:other_")
                                               or space.pool.get(k, k) not in cols))
              for c in counts]
    out = pd.DataFrame({"monomer_smiles": smiles})
    out[["D", "P", "H"]] = fed.predict(X)
    out["V_formula"] = fed.volume(X)
    out[["SPrefit_D", "SPrefit_P", "SPrefit_H"]] = spr.predict(X)
    out["n_pooled_atoms"] = pooled
    out["n_groups_new"] = [sum(v for k, v in c.items() if k.startswith(("NEW:", "SI:"))) for c in counts]
    return out


def main():
    V3.mkdir(parents=True, exist_ok=True)
    space = pickle.load(open(TAB / "group_space.pkl", "rb"))
    fed = pickle.load(open(TAB / "Fedors_refit.pkl", "rb"))
    spr = pickle.load(open(TAB / "SP_refit.pkl", "rb"))
    summ = pd.read_csv(RES / "benchmark" / "reference" / "gc_benchmark_summary.csv")
    best = summ[(summ.tier == 1) & (summ.subset == "all") & summ.model.str.startswith("Fedors")]
    exp_err = float(best.med_Ra_err.min())

    mono = pd.read_csv(RES / "legacy_sp" / "monomers_hsp.csv", low_memory=False)
    f = score(mono.monomer_smiles.tolist(), space, fed, spr)
    df = mono.merge(f, on="monomer_smiles", how="left")
    si = df.has_si.astype(bool)
    df["expected_Ra_err"] = np.where(si, np.nan, exp_err)
    df["hsp_source"] = np.where(si, "gc_formula_si", "gc_formula")
    df["hsp_note"] = np.where(si, SI_NOTE, "")
    for name, ref in REFERENCES.items():
        df[f"Ra_{name}"] = ra(df.D, df.P, df.H, ref)
        df[f"Ra_{name}_legacy"] = ra(df.delta_D, df.delta_P, df.delta_H, ref)
    df.to_csv(V3 / "monomers_hsp_formula.csv", index=False)

    comp = df.rename(columns={"delta_D": "delta_D_sp", "delta_P": "delta_P_sp",
                              "delta_H": "delta_H_sp", "n_unmatched_atoms": "n_unmatched_sp"})
    comp["n_unmatched_atoms"] = comp.n_pooled_atoms
    for k in "DPH":
        comp[f"delta_{k}"] = comp[f"delta_{k}_corr"] = comp[k]
    comp = comp.drop(columns=[c for c in comp.columns if c.startswith("Ra_")])
    # main rankings: non-Si only (Si rows blanked, dropped by the C0 rule);
    # Si monomers get their own rankings and grids in results/v3/si/
    main = comp.copy()
    main.loc[si.to_numpy(), ["delta_D", "delta_P", "delta_H",
                             "delta_D_corr", "delta_P_corr", "delta_H_corr"]] = np.nan
    main.to_csv(V3 / "monomers_hsp_corrected.csv", index=False)
    (V3 / "si").mkdir(exist_ok=True)
    comp[si.to_numpy()].to_csv(V3 / "si" / "monomers_hsp_corrected.csv", index=False)
    df[si].sort_values("Ra_PTFE").to_csv(V3 / "si" / "si_monomers.csv", index=False)

    six = score(list(SIX.values()), space, fed, spr)
    six.insert(0, "id", list(SIX))
    for name, ref in REFERENCES.items():
        six[f"Ra_{name}"] = ra(six.D, six.P, six.H, ref)
    six.to_csv(RES / "manual" / "2026-09-30" / "hsp6_v3.csv", index=False)

    ok = ~si
    print("monomers:", len(df), " Si:", int(si.sum()), " expected Ra err (tier-1 CV):", exp_err)
    print("D/P/H quantiles (non-Si):")
    print(df.loc[ok, ["D", "P", "H"]].quantile([0, .01, .5, .99, 1]).round(2).to_string())
    print("monomers with pooled (rare) atoms:", int((df.n_pooled_atoms > 0).sum()))
    print("Si monomers D range:", df.loc[si, "D"].round(2).describe()[["min", "50%", "max"]].to_dict())
    pd.set_option("display.width", 220)
    print(six[["id", "D", "P", "H", "Ra_PTFE", "SPrefit_D", "SPrefit_P", "SPrefit_H"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
