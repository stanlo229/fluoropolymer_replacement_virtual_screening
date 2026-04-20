"""
make_hansen_grids.py

Generate top-50 grid plots ranked by individual Hansen solubility parameters:
  - top50_max_dispersion_grid.png  (highest δD)
  - top50_max_polar_grid.png       (highest δP)
  - top50_max_hbonding_grid.png    (highest δH)

Reuses make_grid() and compute_ra_columns() from solvent_incompatibility.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd
from rdkit import Chem

# Import helpers from sibling module
sys.path.insert(0, str(Path(__file__).parent))
from solvent_incompatibility import make_grid, compute_ra_columns, REFERENCE_STYLE

OUT_DIR  = Path("results")
INPUT    = OUT_DIR / "monomers_hsp_corrected.csv"
TOP_N    = 50
NCOLS    = 5

D_COL, P_COL, H_COL = "delta_D_corr", "delta_P_corr", "delta_H_corr"


def _canon_no_stereo(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, isomericSmiles=False) if mol else smi


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate (stereo + HSP-triplet) and add Ra columns."""
    df = df.dropna(subset=[D_COL, P_COL, H_COL]).copy()

    smiles_col = "monomer_smiles" if "monomer_smiles" in df.columns else "sidechain_smiles"
    df["_canon"] = df[smiles_col].map(_canon_no_stereo)
    df = df.drop_duplicates(subset=["_canon"]).drop(columns=["_canon"])

    df["_hsp_key"] = (
        df[D_COL].round(4).astype(str) + "|"
        + df[P_COL].round(4).astype(str) + "|"
        + df[H_COL].round(4).astype(str)
    )
    df = df.drop_duplicates(subset=["_hsp_key"]).drop(columns=["_hsp_key"]).copy()
    print(f"Unique monomers after dedup: {len(df)}")

    df, ra_cols, red_cols = compute_ra_columns(df, D_COL, P_COL, H_COL)
    return df, ra_cols, red_cols


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_full = pd.read_csv(INPUT)
    df, ra_cols, red_cols = prepare_df(df_full)

    rankings = [
        (D_COL,      False, "top50_max_dispersion",
         f"Top {TOP_N} Norbornene Monomers — Max Dispersion (δD ↑)\nall Ra in MPa½"),
        (P_COL,      False, "top50_max_polar",
         f"Top {TOP_N} Norbornene Monomers — Max Polar (δP ↑)\nall Ra in MPa½"),
        (H_COL,      False, "top50_max_hbonding",
         f"Top {TOP_N} Norbornene Monomers — Max H-Bonding (δH ↑)\nall Ra in MPa½"),
        ("Ra_PTFE",  True,  "top50_ptfe",
         f"Top {TOP_N} Norbornene Monomers — Most PTFE-like (Ra↓)\nall Ra in MPa½"),
    ]

    for hsp_col, ascending, stem, title in rankings:
        rank_col = hsp_col if hsp_col.startswith("Ra_") else None
        df_top = (
            df.sort_values(hsp_col, ascending=ascending)
              .head(TOP_N)
              .reset_index(drop=True)
        )

        # Save/update CSV with Ra columns included
        csv_path = OUT_DIR / f"{stem}.csv"
        df_top.to_csv(csv_path, index=False)
        print(f"CSV  → {csv_path}  (top {hsp_col}: {df_top[hsp_col].iloc[0]:.2f}–{df_top[hsp_col].iloc[-1]:.2f})")

        grid_path = OUT_DIR / f"{stem}_grid.png"
        make_grid(
            df_top, ra_cols, D_COL, P_COL, H_COL,
            out_path=grid_path,
            title=title,
            rank_col=rank_col,
            ncols=NCOLS,
            red_cols=red_cols,
        )
        print(f"Grid → {grid_path}")


if __name__ == "__main__":
    main()
