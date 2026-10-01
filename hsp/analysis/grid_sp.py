"""
grid_sp.py

Ring-counting variant x low-value trigger x floor grid, scored on the in-scope
reference structures (S-P elements, >= 3 C, uncharged, every heavy atom matched
to a first-order group). Reports all structures and the aliphatic-polycyclic
subset separately, per tier.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import hsp_calculator as hc  # noqa: E402
import sp_variants as sv  # noqa: E402

REF = HERE.parent / "reference"
OUT = HERE.parent / "results" / "benchmark" / "reference"
POLY = {"polycyclic_fused2", "polycyclic_bridged", "polycyclic_3plus"}
RINGS = [("per_ring", 1.0), ("per_system", 1.0), ("isolated_only", 1.0),
         ("discount", 0.5)]
TRIGGERS = ["normal", "covered", "either"]


def in_scope(ref: pd.DataFrame) -> pd.DataFrame:
    ref = ref[(ref.n_carbon >= 3) & ~ref.charged & ref.sp_elements_only & (ref.n_si == 0)].copy()
    ref["unmatched"] = [hc.compute_hsp(s).n_unmatched_atoms for s in ref.canonical_smiles]
    return ref[ref.unmatched == 0].reset_index(drop=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = in_scope(pd.read_csv(REF / "hspip_reference.csv"))
    ref["poly"] = ref.ring_class.isin(POLY)
    print("in-scope structures:", ref.groupby(["tier", "poly"]).size().to_dict())
    res = []
    for (rv, w), trig in itertools.product(RINGS, TRIGGERS):
        outs = [sv.compute(s, rv, w, trigger=trig) for s in ref.canonical_smiles]
        P = np.array([o.P for o in outs]); H = np.array([o.H for o in outs])
        D = np.array([o.D for o in outs])
        for floor in (False, True):
            Pf, Hf = (np.clip(P, 0, None), np.clip(H, 0, None)) if floor else (P, H)
            eD, eP, eH = D - ref.ref_D, Pf - ref.ref_P, Hf - ref.ref_H
            era = np.sqrt(4 * eD ** 2 + eP ** 2 + eH ** 2)
            for tier in (1, 2):
                for sub, m in (("all", ref.tier == tier), ("poly", (ref.tier == tier) & ref.poly)):
                    res.append({"ring": rv if rv != "discount" else f"discount_{w}",
                                "trigger": trig, "floor": floor, "tier": tier, "subset": sub,
                                "n": int(m.sum()), "MAE_P": eP[m].abs().mean(),
                                "MAE_H": eH[m].abs().mean(), "MAE_D": eD[m].abs().mean(),
                                "med_Ra_err": np.median(era[m]),
                                "neg": float(((Pf[m] < 0) | (Hf[m] < 0)).mean())})
    res = pd.DataFrame(res).round(3)
    res.to_csv(OUT / "grid_ring_trigger_floor.csv", index=False)
    pd.set_option("display.width", 200)
    for tier in (1, 2):
        for sub in ("all", "poly"):
            t = res[(res.tier == tier) & (res.subset == sub) & res.floor].sort_values("med_Ra_err")
            print(f"\n== tier {tier}, {sub}, floored (n={t.n.iloc[0]}) ==")
            print(t.drop(columns=["tier", "subset", "floor", "n"]).to_string(index=False))


if __name__ == "__main__":
    main()
