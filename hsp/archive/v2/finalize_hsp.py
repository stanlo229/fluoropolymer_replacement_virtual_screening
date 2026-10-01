"""
finalize_hsp.py

Assemble the final monomer HSP table from predict_monomers.py output and write
the inputs the downstream ranking/plotting scripts expect.

  * non-Si monomers: size-aware model (ENS <= 30 heavy atoms, ExtraTrees
    direct above), every component >= 0
  * Si monomers:     NOT scored in the main table. The si_gc extension is
    validated only on small Si compounds (median 9 heavy atoms vs 41 for the
    Si monomers) and gives impossible values on some monomers (dD 6.9), so its
    output goes to a separate, flagged file for inspection only
  * dD cross-check:  size-intensive refractivity-density dD (analysis/
    dispersion.py); rows where the two dD estimates differ by > 2 are flagged

Outputs
  results/v2/monomers_hsp_final.csv          full table, all columns
  results/v2/monomers_hsp_corrected.csv   downstream-compatible (delta_*_corr)
  results/v2/si_monomers_unvalidated.csv  Si monomers with si_gc values
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import si_gc  # noqa: E402
from solvent_incompatibility import REFERENCES as REFERENCE_HSP  # noqa: E402

RES = HERE.parent / "results"
V2 = RES / "v2"



def expected_error() -> dict:
    """Median held-out |Ra| error on the curated tier-1 set, by confidence bin
    (analysis/expected_error.py -> expected_error_by_confidence.csv)."""
    t = pd.read_csv(RES / "benchmark" / "reference" / "expected_error_by_confidence.csv")
    t = t[t.tier == 1]
    return dict(zip(t.confidence, t.med_Ra_err))


def ra(D, P, H, ref):
    rD, rP, rH = ref
    return np.sqrt(4 * (D - rD) ** 2 + (P - rP) ** 2 + (H - rH) ** 2)


def dd_refractivity(smiles):
    import dispersion as dp
    vmod, p = dp.main()
    out = []
    for i in range(0, len(smiles), 2000):
        mols = [Chem.MolFromSmiles(s) for s in smiles[i:i + 2000]]
        vm, _ = vmod.predict(mols)
        out += list(dp._dd(p, np.array([Crippen.MolMR(m) for m in mols]) / vm))
    return np.array(out)


def main(top_n: int = 50):
    V2.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RES / "v2" / "monomers_hsp_ens.csv", low_memory=False)
    si = df.has_si.astype(bool).to_numpy()

    df["expected_Ra_err"] = df.confidence.map(expected_error())
    df["dD_refractivity"] = dd_refractivity(df.monomer_smiles.tolist())
    df["flag_dD_disagree"] = (df.D - df.dD_refractivity).abs() > 2.0

    # Si: separate, unvalidated
    sim = si_gc.fit()
    df[["si_D", "si_P", "si_H"]] = np.nan
    df.loc[si, ["si_D", "si_P", "si_H"]] = si_gc.predict(sim, df.loc[si, "monomer_smiles"])
    df["flag_silanol"] = False
    df.loc[si, "flag_silanol"] = df.loc[si, "monomer_smiles"].map(si_gc.has_silanol).to_numpy()
    df.loc[si, ["D", "P", "H", "expected_Ra_err"]] = np.nan
    df.loc[si, "hsp_source"] = "si_unvalidated"
    df.loc[si, "confidence"] = "not_scored"

    for name, ref in REFERENCE_HSP.items():
        df[f"Ra_{name}"] = ra(df.D, df.P, df.H, ref)
        df[f"Ra_{name}_legacy"] = ra(df.delta_D, df.delta_P, df.delta_H, ref)
    df.to_csv(RES / "v2" / "monomers_hsp_final.csv", index=False)

    # downstream-compatible table: legacy S-P kept as delta_*_sp
    comp = df.rename(columns={"delta_D": "delta_D_sp", "delta_P": "delta_P_sp",
                              "delta_H": "delta_H_sp"})
    comp["delta_D"] = comp["delta_D_corr"] = comp.D
    comp["delta_P"] = comp["delta_P_corr"] = comp.P
    comp["delta_H"] = comp["delta_H_corr"] = comp.H
    comp = comp.drop(columns=[c for c in comp.columns if c.startswith("Ra_")])
    comp.to_csv(V2 / "monomers_hsp_corrected.csv", index=False)

    s = df[si].copy()
    for name, ref in REFERENCE_HSP.items():
        s[f"Ra_{name}_si_gc"] = ra(s.si_D, s.si_P, s.si_H, ref)
    s["flag_impossible_dD"] = s.si_D < 11.0
    s.sort_values("Ra_PTFE_si_gc").to_csv(V2 / "si_monomers_unvalidated.csv", index=False)

    ok = ~si
    print("negative D/P/H in final table:", int((df.loc[ok, ["D", "P", "H"]] < 0).any(axis=1).sum()))
    print("negative in legacy S-P table: ", int((df[["delta_P", "delta_H"]] < 0).any(axis=1).sum()))
    print("source:", df.hsp_source.value_counts().to_dict())
    print("confidence:", df.confidence.value_counts().to_dict())
    print("D range (scored):", df.loc[ok, "D"].quantile([0, .01, .5, .99, 1]).round(2).to_dict())
    print("dD disagreement > 2 vs refractivity model:", int(df.loc[ok, "flag_dD_disagree"].sum()))
    print("Si: impossible dD (<11):", int(s.flag_impossible_dD.sum()), "/", len(s),
          " silanols:", int(s.flag_silanol.sum()))
    new = set(df[ok].nsmallest(top_n, "Ra_PTFE").monomer_smiles)
    old = set(df[ok].nsmallest(top_n, "Ra_PTFE_legacy").monomer_smiles)
    print(f"non-Si PTFE top-{top_n} overlap new vs legacy: {len(new & old)}/{top_n}")


if __name__ == "__main__":
    main()
