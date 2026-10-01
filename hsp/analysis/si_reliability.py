"""
si_reliability.py

Evidence for and against trusting the v3 formula's HSP for Si monomers.

  A  cross-validated accuracy on Si references (out-of-fold, gc_benchmark.py),
     by tier and Si class, with bootstrap 95% CIs of the median Ra error
  B  size / Si-share gap between the Si references and the Si monomers
  C  support of the Si groups each Si monomer uses (n HSPiP compounds per group)
  D  Si motifs in the monomers vs in the references
  E  disagreement between methods on the Si monomers (v3 formula, v2 si_gc, S-P Si->C)
  F  value ranges: Si monomers vs HSPiP Si references
  G  ranking sensitivity: how far ahead the Si monomers are in Ra(PTFE)

Writes results/benchmark/reference/si_reliability.md (gitignored: it quotes
reference-derived statistics) and prints the same tables.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import gc_refit as gr  # noqa: E402

RDLogger.DisableLog("rdApp.*")
RES = HERE.parent / "results"
BEN = RES / "benchmark" / "reference"
REF = HERE.parent / "reference"
FED = "Fedors_refit"
rng = np.random.default_rng(0)

MOTIFS = {
    "siloxane Si-O-Si": "[Si]O[Si]",
    "silyl ether Si-O-C": "[Si]O[#6]",
    "TBS (tBuMe2Si-)": "[Si](C)(C)C(C)(C)C",
    "TMS (Me3Si-)": "[Si]([CH3])([CH3])[CH3]",
    "Si-Si bond": "[Si][Si]",
    "silylamine Si-N": "[Si][#7]",
    "silanol Si-OH": "[Si][OX2H1]",
    "Si-H": "[Si;!H0]",
    "alkyl/aryl silane only (no Si-heteroatom)": None,
}
_MP = {k: Chem.MolFromSmarts(v) for k, v in MOTIFS.items() if v}


def si_class(smi):
    m = Chem.MolFromSmiles(smi)
    for k, v in [("siloxane", "[Si]O[Si]"), ("silyl ether", "[Si]O[#6]"), ("silanol", "[Si][OX2H1]"),
                 ("halosilane", "[Si][F,Cl,Br,I]"), ("silylamine", "[Si][#7]")]:
        if m.HasSubstructMatch(Chem.MolFromSmarts(v)):
            return k
    return "C-only silane"


def motif_counts(smiles):
    out = {k: 0 for k in MOTIFS}
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        hit = False
        for k, p in _MP.items():
            if m.HasSubstructMatch(p):
                out[k] += 1
                if k in ("siloxane Si-O-Si", "silyl ether Si-O-C", "Si-Si bond", "silylamine Si-N",
                         "silanol Si-OH"):
                    hit = True
        if not hit:
            out["alkyl/aryl silane only (no Si-heteroatom)"] += 1
    return out


def ra(a, b):
    e = a - b
    return np.sqrt(4 * e[:, 0] ** 2 + e[:, 1] ** 2 + e[:, 2] ** 2)


def boot_median_ci(x, n=5000):
    x = np.asarray(x)
    if len(x) < 2:
        return (np.nan, np.nan)
    b = np.median(rng.choice(x, (n, len(x)), replace=True), axis=1)
    return tuple(np.round(np.percentile(b, [2.5, 97.5]), 2))


def md(df):
    """Minimal GitHub markdown table (no tabulate dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def main():
    out = []
    oof = pd.read_csv(BEN / "gc_oof_predictions.csv")
    Y = oof[["ref_D", "ref_P", "ref_H"]].to_numpy()
    for m in (FED, "SP_paper", "XGB_groups", "XGB_fp"):
        oof[f"e_{m}"] = ra(oof[[f"{m}_D", f"{m}_P", f"{m}_H"]].to_numpy(), Y)
    si = oof.n_si > 0
    oof["si_class"] = [si_class(s) if q else "" for s, q in zip(oof.canonical_smiles, si)]

    # ---- A: CV accuracy ----
    rows = []
    for name, k in [("non-Si, tier 1", ~si & (oof.tier == 1)), ("Si, tier 1 (curated)", si & (oof.tier == 1)),
                    ("non-Si, tier 2 (Y-MB)", ~si & (oof.tier == 2)), ("Si, tier 2 (Y-MB)", si & (oof.tier == 2))]:
        e = oof.loc[k, f"e_{FED}"]
        rows.append({"subset": name, "n": int(k.sum()), "v3 median Ra err": round(e.median(), 2),
                     "95% CI": boot_median_ci(e), "v3 p90": round(e.quantile(.9), 2),
                     "S-P paper median": round(oof.loc[k, "e_SP_paper"].median(), 2),
                     "XGB groups median": round(oof.loc[k, "e_XGB_groups"].median(), 2)})
    A = pd.DataFrame(rows)
    A2 = (oof[si].groupby(["si_class"]).agg(n=("tier", "size"), n_tier1=("tier", lambda t: int((t == 1).sum())),
                                            v3_median=(f"e_{FED}", "median"), SP_median=("e_SP_paper", "median"))
          .round(2).reset_index())
    t1si = oof[si & (oof.tier == 1)][["Name", "si_class", "n_heavy", "ref_D", "ref_P", "ref_H",
                                     f"{FED}_D", f"{FED}_P", f"{FED}_H", f"e_{FED}"]].round(2)
    out += ["## A. Cross-validated accuracy (scaffold-grouped, out-of-fold)", md(A), "",
            "By Si class (tiers 1+2):", md(A2), "", "The 7 curated Si compounds:", md(t1si), ""]

    # ---- monomers ----
    mono = pd.read_csv(RES / "v3" / "monomers_hsp_formula.csv", low_memory=False)
    simo = mono[mono.has_si.astype(bool)].copy()
    nsimo = mono[~mono.has_si.astype(bool)]
    mols = [Chem.MolFromSmiles(s) for s in simo.monomer_smiles]
    simo["n_heavy"] = [m.GetNumHeavyAtoms() for m in mols]
    simo["n_si"] = [sum(a.GetAtomicNum() == 14 for a in m.GetAtoms()) for m in mols]
    ref_si = oof[si]

    # ---- B: size / Si share ----
    B = pd.DataFrame([
        {"set": "Si references (CV set)", "n": len(ref_si), "heavy atoms median": ref_si.n_heavy.median(),
         "heavy atoms max": ref_si.n_heavy.max(),
         "Si atoms / heavy atoms (median)": round((ref_si.n_si / ref_si.n_heavy).median(), 3)},
        {"set": "Si monomers", "n": len(simo), "heavy atoms median": simo.n_heavy.median(),
         "heavy atoms max": simo.n_heavy.max(),
         "Si atoms / heavy atoms (median)": round((simo.n_si / simo.n_heavy).median(), 3)},
    ])
    n_in_range = int((simo.n_heavy <= ref_si.n_heavy.max()).sum())
    out += ["## B. Size gap", md(B), f"Si monomers within the reference size range (<= {ref_si.n_heavy.max()} heavy atoms): {n_in_range}/{len(simo)}", ""]

    # ---- C: group support ----
    groups = pd.read_csv(BEN / "gc_tables" / "fedors_refit_groups.csv").set_index("group")
    import pickle
    space = pickle.load(open(BEN / "gc_tables" / "group_space.pkl", "rb"))
    sup_rows, weak = [], []
    for s in simo.monomer_smiles:
        c = gr.raw_counts(s) or {}
        sig = {space.pool.get(k, k) for k in c if "Si" in k}
        n = [int(groups.n_compounds.get(g, 0)) for g in sig]
        weak.append(min(n) if n else 0)
        sup_rows += [(g, int(groups.n_compounds.get(g, 0))) for g in sig]
    simo["min_si_group_support"] = weak
    sup = (pd.DataFrame(sup_rows, columns=["Si group", "n HSPiP compounds"])
           .value_counts().reset_index(name="n Si monomers using it").sort_values("n Si monomers using it", ascending=False))
    gv = groups.loc[[g for g in groups.index if "Si" in g],
                    ["v_cm3mol", "E_D_Jmol", "E_P_Jmol", "E_H_Jmol", "n_compounds"]].round(0).reset_index()
    out += ["## C. Support of the Si groups the monomers use", md(sup),
            f"Si monomers whose weakest Si group has < 10 HSPiP compounds: {int((simo.min_si_group_support < 10).sum())}/{len(simo)}",
            "", "Fitted Si group values (Fedors form):", md(gv), ""]

    # ---- D: motifs ----
    D = pd.DataFrame({"Si monomers": motif_counts(simo.monomer_smiles),
                      "Si refs tier 1": motif_counts(ref_si[ref_si.tier == 1].canonical_smiles),
                      "Si refs tier 2": motif_counts(ref_si[ref_si.tier == 2].canonical_smiles)}).reset_index(names="motif")
    out += ["## D. Si motifs: monomers vs references", md(D), ""]

    # ---- E: method disagreement ----
    v2 = pd.read_csv(RES / "v2" / "si_monomers_unvalidated.csv", usecols=["monomer_smiles", "si_D", "si_P", "si_H"])
    e = simo.merge(v2, on="monomer_smiles", how="left")
    a = e[["D", "P", "H"]].to_numpy()
    d_v2 = ra(a, e[["si_D", "si_P", "si_H"]].to_numpy())
    sp = np.column_stack([e.delta_D, e.delta_P.clip(lower=0), e.delta_H.clip(lower=0)])
    d_sp = ra(a, sp)
    E = pd.DataFrame([{"comparison": "v3 formula vs v2 si_gc ridge", "median Ra difference": round(np.nanmedian(d_v2), 2),
                       "p90": round(np.nanpercentile(d_v2, 90), 2)},
                      {"comparison": "v3 formula vs published S-P (Si->C, floored)", "median Ra difference": round(np.nanmedian(d_sp), 2),
                       "p90": round(np.nanpercentile(d_sp, 90), 2)}])
    out += ["## E. Method disagreement on the Si monomers", md(E), ""]

    # ---- F: ranges ----
    F = pd.DataFrame([{"set": n, "dD min": round(x.D.min() if "D" in x else x.ref_D.min(), 1),
                       "dD median": round(x.D.median() if "D" in x else x.ref_D.median(), 1),
                       "dD max": round(x.D.max() if "D" in x else x.ref_D.max(), 1),
                       "dP median": round(x.P.median() if "P" in x else x.ref_P.median(), 1),
                       "dH median": round(x.H.median() if "H" in x else x.ref_H.median(), 1)}
                      for n, x in [("HSPiP Si references", ref_si.drop(columns=[c for c in ref_si if c in ("D", "P", "H")])),
                                   ("Si monomers (v3)", simo), ("non-Si monomers (v3)", nsimo)]])
    out += ["## F. Value ranges", md(F), ""]

    # ---- G: ranking sensitivity ----
    best_si, best_nsi = simo.Ra_PTFE.min(), nsimo.Ra_PTFE.min()
    n_si_better = int((simo.Ra_PTFE < best_nsi).sum())
    out += ["## G. Ranking sensitivity (PTFE)",
            f"Best Si monomer Ra(PTFE) {best_si:.2f}; best non-Si {best_nsi:.2f}; gap {best_nsi - best_si:.2f} MPa^0.5.",
            f"Si monomers ahead of the best non-Si monomer: {n_si_better}.", ""]

    text = "\n".join(out)
    (BEN / "si_reliability.md").write_text(text)
    simo[["monomer_smiles", "n_heavy", "n_si", "min_si_group_support"]].to_csv(BEN / "si_monomer_support.csv", index=False)
    print(text)


if __name__ == "__main__":
    main()
