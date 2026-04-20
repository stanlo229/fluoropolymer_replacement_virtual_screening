"""
solvent_incompatibility.py

Compute Ra (Hansen distance) from Water, Diiodomethane, n-Hexadecane, and PTFE
for all norbornene monomers. Generate 4 top-50 grid plots:
  - Highest Ra from Water        (least soluble in water)
  - Highest Ra from Diiodomethane
  - Highest Ra from n-Hexadecane
  - Lowest  Ra from PTFE         (most PTFE-like)

Usage:
    python solvent_incompatibility.py [--input PATH] [--out_dir PATH] [--top_n 50]
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image

# ---------------------------------------------------------------------------
# Reference HSP values (Hansen 2007, Appendix A / HSPiP)
# δD, δP, δH in MPa^0.5
# ---------------------------------------------------------------------------
REFERENCES: dict[str, tuple[float, float, float]] = {
    "Water":         (15.5, 16.0, 42.3),
    "Diiodomethane": (17.8,  3.9,  5.5),
    "n-Hexadecane":  (16.3,  0.0,  0.0),
    "PTFE":          (12.7,  0.0,  0.0),
}

# Interaction radius R0 of the Hansen solubility sphere (MPa^0.5).
# RED = Ra / R0; RED < 1 → likely compatible, RED > 1 → likely incompatible.
# Sources: Hansen (2007) Appendix A; approx values noted.
R0_VALUES: dict[str, float | None] = {
    "Water":         17.8,   # Hansen (2007)
    "Diiodomethane":  7.0,   # approx (dispersion-dominated solvent)
    "n-Hexadecane":   5.0,   # approx (aliphatic non-polar)
    "PTFE":           7.0,   # Hansen (2007) Table A.2
}

# Short display name, color, and ranking direction (ascending=True → lowest Ra = most similar)
REFERENCE_STYLE: dict[str, tuple[str, str, bool]] = {
    "Water":         ("Water",   "#5bc0eb", False),   # highest Ra = least soluble
    "Diiodomethane": ("Diiodo.", "#fde74c", False),
    "n-Hexadecane":  ("n-C16",  "#9bc53d", False),
    "PTFE":          ("PTFE",   "#ff6b9d", True),     # lowest Ra = most PTFE-like
}

# Keep old name for backward compat in compute_ra_columns
SOLVENTS = REFERENCES

# Column key per solvent (used in DataFrame)
def _solvent_key(name: str) -> str:
    return name.replace("-", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
# Ra and RED formulas
# ---------------------------------------------------------------------------
def _ra(dD, dP, dH, rD, rP, rH) -> np.ndarray:
    """Hansen distance: Ra = sqrt(4(ΔδD)² + (ΔδP)² + (ΔδH)²)"""
    return np.sqrt(4 * (dD - rD) ** 2 + (dP - rP) ** 2 + (dH - rH) ** 2)


def compute_ra_columns(
    df: pd.DataFrame,
    d_col: str = "delta_D_corr",
    p_col: str = "delta_P_corr",
    h_col: str = "delta_H_corr",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Add Ra_<ref> and RED_<ref> columns for all references; return (df, ra_cols, red_cols)."""
    ra_cols = []
    red_cols = []
    for name, (rD, rP, rH) in REFERENCES.items():
        key = _solvent_key(name)
        ra_col = f"Ra_{key}"
        df[ra_col] = _ra(df[d_col], df[p_col], df[h_col], rD, rP, rH)
        ra_cols.append(ra_col)

        r0 = R0_VALUES.get(name)
        if r0 is not None:
            red_col = f"RED_{key}"
            df[red_col] = df[ra_col] / r0
            red_cols.append(red_col)

    return df, ra_cols, red_cols


# ---------------------------------------------------------------------------
# Molecule rendering
# ---------------------------------------------------------------------------
def _smiles_to_pil(smiles: str, width: int = 460, height: int = 220) -> Image.Image | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    opts = rdMolDraw2D.MolDrawOptions()
    opts.bondLineWidth = 1.8
    opts.addStereoAnnotation = False
    opts.padding = 0.12
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    drawer.SetDrawOptions(opts)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))


# ---------------------------------------------------------------------------
# Grid plot
# ---------------------------------------------------------------------------
BG_DARK   = "#1a1a2e"   # figure background
BG_CELL   = "#16213e"   # molecule cell
BG_TEXT   = "#0f3460"   # text cell
FG_HEADER = "#e0e0e0"   # header text
FG_LABEL  = "#aaaaaa"   # secondary text


def make_grid(
    df_top: pd.DataFrame,
    ra_cols: list[str],
    d_col: str,
    p_col: str,
    h_col: str,
    out_path: str | Path,
    title: str = "",
    rank_col: str | None = None,
    ncols: int = 5,
    red_cols: list[str] | None = None,
) -> Path:
    n = len(df_top)
    nrows = (n + ncols - 1) // ncols

    MOL_H  = 2.8   # inches — molecule image row
    TEXT_H = 1.45  # inches — annotation row (taller to fit 4 Ra lines)

    fig_w = ncols * 4.6
    fig_h = nrows * (MOL_H + TEXT_H) + 0.9

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG_DARK)
    fig.suptitle(
        title or "Top 50 Norbornene Monomers",
        fontsize=11, color=FG_HEADER, fontweight="bold", y=0.998, va="top",
    )

    gs = gridspec.GridSpec(
        nrows * 2, ncols,
        figure=fig,
        height_ratios=[MOL_H, TEXT_H] * nrows,
        hspace=0.08,
        wspace=0.06,
    )

    # Build ordered list of (name, short_label, color) from REFERENCE_STYLE
    ref_items = [(name, short, color) for name, (short, color, _) in REFERENCE_STYLE.items()]

    rows_iter = list(df_top.iterrows())

    for i, (_, row) in enumerate(rows_iter):
        r, c = divmod(i, ncols)

        # ---- molecule image ----
        ax_img = fig.add_subplot(gs[r * 2, c])
        ax_img.set_facecolor(BG_CELL)
        for spine in ax_img.spines.values():
            spine.set_edgecolor("#334466")
            spine.set_linewidth(0.6)

        smi = str(row.get("monomer_smiles", row.get("sidechain_smiles", "")))
        pil_img = _smiles_to_pil(smi)
        if pil_img is not None:
            ax_img.imshow(np.array(pil_img), aspect="auto")
        ax_img.axis("off")

        # Rank badge (top-left)
        ax_img.text(
            0.03, 0.97, f"#{i + 1}",
            transform=ax_img.transAxes,
            fontsize=7, color="#ffffff", va="top", ha="left",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#334466", alpha=0.8, linewidth=0),
        )

        # ---- text annotation ----
        ax_txt = fig.add_subplot(gs[r * 2 + 1, c])
        ax_txt.set_facecolor(BG_TEXT)
        for spine in ax_txt.spines.values():
            spine.set_edgecolor("#334466")
            spine.set_linewidth(0.6)
        ax_txt.axis("off")

        # δD, δP, δH header
        hsp_str = f"δD {row[d_col]:.1f}  δP {row[p_col]:.1f}  δH {row[h_col]:.1f}  MPa½"
        ax_txt.text(
            0.5, 0.97, hsp_str,
            transform=ax_txt.transAxes,
            fontsize=5.8, color=FG_LABEL, va="top", ha="center",
            fontfamily="monospace",
        )

        # Ra and RED per reference (color-coded rows)
        for j, (rname, short, rcol) in enumerate(ref_items):
            rkey = _solvent_key(rname)
            ra_col_name = f"Ra_{rkey}"
            red_col_name = f"RED_{rkey}"
            ra_val = row[ra_col_name]
            red_val = row[red_col_name] if (red_cols and red_col_name in red_cols) else None
            ypos = 0.70 - j * 0.21
            is_ranked = (ra_col_name == rank_col)
            marker = "★" if is_ranked else "■"
            fsize = 6.8 if is_ranked else 6.0
            fw = "bold"
            ax_txt.text(
                0.04, ypos + 0.01, marker,
                transform=ax_txt.transAxes,
                fontsize=9 if is_ranked else 7,
                color=rcol, va="top", ha="left",
            )
            red_str = f"  RED={red_val:.2f}" if (red_val is not None and not np.isnan(red_val)) else ""
            ax_txt.text(
                0.18, ypos + 0.01,
                f"{short:<8} Ra={ra_val:.2f}{red_str}",
                transform=ax_txt.transAxes,
                fontsize=fsize, color=rcol, va="top", ha="left",
                fontfamily="monospace", fontweight=fw,
            )

        # Ranked Ra value (bottom-right, prominent)
        if rank_col and rank_col in row.index:
            ax_txt.text(
                0.97, 0.04,
                f"ranked Ra = {row[rank_col]:.2f}",
                transform=ax_txt.transAxes,
                fontsize=5.5, color="#888888", va="bottom", ha="right",
                fontfamily="monospace",
            )

    # Hide unused cells
    for i in range(len(rows_iter), nrows * ncols):
        r, c = divmod(i, ncols)
        for sub_r in (r * 2, r * 2 + 1):
            ax = fig.add_subplot(gs[sub_r, c])
            ax.set_facecolor(BG_DARK)
            ax.axis("off")

    out_path = Path(out_path)
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="results/monomers_hsp_corrected.csv",
        help="Path to corrected HSP CSV (default: results/monomers_hsp_corrected.csv)",
    )
    parser.add_argument(
        "--out_dir", default="results",
        help="Output directory (default: results/)",
    )
    parser.add_argument(
        "--top_n", type=int, default=50,
        help="Number of monomers to include in the grid (default: 50)",
    )
    parser.add_argument(
        "--ncols", type=int, default=5,
        help="Grid columns (default: 5)",
    )
    parser.add_argument(
        "--catalogues", default="../dataset/catalogues.csv",
        help="Path to catalogues.csv for purchase links (default: ../dataset/catalogues.csv)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load ----
    df = pd.read_csv(args.input)
    d_col = "delta_D_corr" if "delta_D_corr" in df.columns else "delta_D"
    p_col = "delta_P_corr" if "delta_P_corr" in df.columns else "delta_P"
    h_col = "delta_H_corr" if "delta_H_corr" in df.columns else "delta_H"

    # Drop rows missing HSP
    df = df.dropna(subset=[d_col, p_col, h_col]).copy()

    # ---- deduplicate by canonical SMILES (stereo-stripped) ----
    # Group-contribution HSP ignores stereochemistry, so E/Z and R/S isomers
    # share identical HSP values and should be collapsed to one representative.
    smiles_col = "monomer_smiles" if "monomer_smiles" in df.columns else "sidechain_smiles"
    n_before = len(df)

    def _canon_no_stereo(smi: str) -> str:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol, isomericSmiles=False) if mol else smi

    df["_canon"] = df[smiles_col].map(_canon_no_stereo)
    df = df.drop_duplicates(subset=["_canon"]).drop(columns=["_canon"]).copy()
    n_after_stereo = len(df)

    # Also deduplicate by HSP triplet: structural isomers with identical group
    # counts produce the same (δD, δP, δH) and are indistinguishable by HSP.
    # Round to 4 dp to avoid floating-point noise, keep first representative.
    df["_hsp_key"] = (
        df[d_col].round(4).astype(str) + "|"
        + df[p_col].round(4).astype(str) + "|"
        + df[h_col].round(4).astype(str)
    )
    df = df.drop_duplicates(subset=["_hsp_key"]).drop(columns=["_hsp_key"]).copy()
    print(
        f"Deduplicated {n_before} → {n_after_stereo} (stereo) → {len(df)} unique monomers (HSP-triplet)"
    )

    # ---- join catalogues for purchase links ----
    cat_path = Path(args.catalogues)
    if cat_path.exists():
        cat = pd.read_csv(cat_path, usecols=["canonical_smiles", "link"])
        df = df.merge(cat, left_on="sidechain_smiles", right_on="canonical_smiles", how="left")
        df = df.drop(columns=["canonical_smiles"])
        print(f"Purchase links joined: {df['link'].notna().sum()} / {len(df)} monomers")
    else:
        df["link"] = pd.NA
        print(f"Warning: catalogues not found at {cat_path}, link column will be empty")

    # ---- compute Ra and RED from all 4 references ----
    df, ra_cols, red_cols = compute_ra_columns(df, d_col, p_col, h_col)

    # ---- save full ranked CSV (by PTFE Ra ascending) ----
    csv_path = out_dir / "monomers_hsp_solvent_Ra.csv"
    display_cols = (
        ["monomer_smiles", "sidechain_smiles", "linkage", d_col, p_col, h_col]
        + ra_cols + red_cols
    )
    display_cols = [c for c in display_cols if c in df.columns]
    df.sort_values("Ra_PTFE").to_csv(csv_path, index=False)
    print(f"Full CSV → {csv_path}  ({len(df)} unique monomers)")

    # ---- generate one grid per reference ----
    grid_specs = [
        # (ref_name, Ra_col,            ascending, label)
        ("Water",         "Ra_Water",         False, "Least Soluble in Water        (highest Ra↑)"),
        ("Diiodomethane", "Ra_Diiodomethane",  False, "Least Soluble in Diiodomethane (highest Ra↑)"),
        ("n-Hexadecane",  "Ra_n_Hexadecane",   False, "Least Soluble in n-Hexadecane  (highest Ra↑)"),
        ("PTFE",          "Ra_PTFE",           True,  "Most PTFE-like                 (lowest Ra↓)"),
    ]

    for ref_name, rank_col, ascending, label in grid_specs:
        df_sorted = df.sort_values(rank_col, ascending=ascending).reset_index(drop=True)
        df_top = df_sorted.head(args.top_n)

        direction = "↓ lowest" if ascending else "↑ highest"
        print(f"\n--- {ref_name} ({direction} Ra) top {args.top_n} ---")
        print(df_top[[d_col, p_col, h_col, rank_col]].to_string(
            index=True, float_format="{:.3f}".format))

        safe_name = ref_name.replace("-", "").replace(" ", "_").lower()

        # ---- per-ranking CSV ----
        csv_cols = (
            ["sidechain_smiles", "monomer_smiles", "source", "link",
             d_col, p_col, h_col]
            + ra_cols + red_cols
        )
        csv_cols = [c for c in csv_cols if c in df_top.columns]
        csv_out = out_dir / f"top{args.top_n}_{safe_name}.csv"
        df_top[csv_cols].to_csv(csv_out, index=False)
        print(f"CSV  → {csv_out}")

        # ---- grid plot ----
        title = (
            f"Top {args.top_n} Norbornene Monomers — {label}\n"
            f"★ = ranking criterion  |  all Ra in MPa½"
        )
        grid_path = out_dir / f"top{args.top_n}_{safe_name}_grid.png"
        make_grid(df_top, ra_cols, d_col, p_col, h_col, grid_path,
                  title=title, rank_col=rank_col, ncols=args.ncols,
                  red_cols=red_cols)
        print(f"Grid → {grid_path}")


if __name__ == "__main__":
    main()
