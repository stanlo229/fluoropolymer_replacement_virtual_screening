"""
visualize_hsp.py

Hansen space visualization for norbornene monomer HSP predictions.

Outputs:
  hsp_hansen_space.png  — 3D scatter + 2D projections
  hsp_Ra_ranked.csv     — Ra² (distance from PTFE) sorted ascending
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PFAS / solvent landmark references (Hansen 2007, Appendix A)
# ---------------------------------------------------------------------------
LANDMARKS = {
    "PTFE":   (12.7,  0.0,  0.0),
    "PFOA":   (13.7,  3.1,  5.2),
    "PDMS":   (14.9,  0.5,  3.4),
    "Water":  (15.5, 16.0, 42.3),
    "Hexane": (14.9,  0.0,  0.0),
}

LINKAGE_COLORS = {"ester": "#2196F3", "amide": "#FF9800", "other": "#9E9E9E"}


def _ra2_from_ptfe(delta_D, delta_P, delta_H) -> np.ndarray:
    """Hansen distance² from PTFE: Ra² = 4(ΔδD)² + (ΔδP)² + (ΔδH)²"""
    dD, dP, dH = LANDMARKS["PTFE"]
    return 4 * (delta_D - dD)**2 + (delta_P - dP)**2 + (delta_H - dH)**2


def visualize(
    df: pd.DataFrame,
    out_dir: str | Path = "results",
    prefix: str = "hsp",
    d_col: str = "delta_D_corr",
    p_col: str = "delta_P_corr",
    h_col: str = "delta_H_corr",
) -> tuple[Path, Path]:
    """
    Generate Hansen space plots and Ra-ranked CSV.

    Parameters
    ----------
    df      : DataFrame with HSP columns (and optionally 'linkage')
    out_dir : directory to write outputs
    prefix  : filename prefix
    d_col, p_col, h_col : column names for δD, δP, δH

    Returns
    -------
    (png_path, csv_path)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Drop rows without HSP values
    df_plot = df.dropna(subset=[d_col, p_col, h_col]).copy()
    if df_plot.empty:
        log.warning("No valid HSP rows to plot.")
        return None, None

    dD = df_plot[d_col].values
    dP = df_plot[p_col].values
    dH = df_plot[h_col].values

    linkage = df_plot.get("linkage", pd.Series("other", index=df_plot.index))
    colors = [LINKAGE_COLORS.get(str(l).lower(), LINKAGE_COLORS["other"]) for l in linkage]

    # -----------------------------------------------------------------------
    # Figure: 1 row with 4 panels (3D + 3 × 2D projection)
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(20, 5))
    fig.suptitle("Hansen Solubility Parameter Space — Norbornene Monomers", fontsize=13)

    # -- 3D scatter --
    ax3d = fig.add_subplot(141, projection="3d")
    ax3d.scatter(dD, dP, dH, c=colors, s=4, alpha=0.4, linewidths=0)
    for lname, (ld, lp, lh) in LANDMARKS.items():
        ax3d.scatter(ld, lp, lh, marker="*", s=120, zorder=5,
                     color="black", edgecolors="white", linewidths=0.5)
        ax3d.text(ld, lp, lh + 0.5, lname, fontsize=7)
    ax3d.set_xlabel("δD (MPa$^{0.5}$)", fontsize=8)
    ax3d.set_ylabel("δP (MPa$^{0.5}$)", fontsize=8)
    ax3d.set_zlabel("δH (MPa$^{0.5}$)", fontsize=8)
    ax3d.set_title("3D Hansen space", fontsize=9)

    # Legend patches
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=v, label=k) for k, v in LINKAGE_COLORS.items()
                  if k in linkage.values]
    if legend_els:
        ax3d.legend(handles=legend_els, fontsize=7, loc="upper left")

    # -- 2D projections --
    proj_axes = [
        (fig.add_subplot(142), dD, dP, "δD (MPa$^{0.5}$)", "δP (MPa$^{0.5}$)", "δD–δP"),
        (fig.add_subplot(143), dD, dH, "δD (MPa$^{0.5}$)", "δH (MPa$^{0.5}$)", "δD–δH"),
        (fig.add_subplot(144), dP, dH, "δP (MPa$^{0.5}$)", "δH (MPa$^{0.5}$)", "δP–δH"),
    ]
    for ax, x, y, xl, yl, title in proj_axes:
        ax.scatter(x, y, c=colors, s=3, alpha=0.3, linewidths=0)
        for lname, (ld, lp, lh) in LANDMARKS.items():
            lx = ld if xl.startswith("δD") else (lp if xl.startswith("δP") else lh)
            ly = lp if yl.startswith("δP") else lh
            ax.scatter(lx, ly, marker="*", s=80, zorder=5,
                       color="black", edgecolors="white", linewidths=0.5)
            ax.annotate(lname, (lx, ly), fontsize=6, xytext=(3, 3),
                        textcoords="offset points")
        ax.set_xlabel(xl, fontsize=8)
        ax.set_ylabel(yl, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    png_path = out_dir / f"{prefix}_hansen_space.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Plot saved to %s", png_path)

    # -----------------------------------------------------------------------
    # Ra-ranked CSV
    # -----------------------------------------------------------------------
    ra2 = _ra2_from_ptfe(dD, dP, dH)
    df_ra = df_plot.copy()
    df_ra["Ra2_from_PTFE"] = ra2
    df_ra["Ra_from_PTFE"]  = np.sqrt(ra2)
    df_ra = df_ra.sort_values("Ra2_from_PTFE")

    csv_path = out_dir / f"{prefix}_Ra_ranked.csv"
    df_ra.to_csv(csv_path, index=False)
    log.info("Ra-ranked CSV saved to %s  (top 5 Ra=%.2f–%.2f MPa^0.5)",
             csv_path,
             df_ra["Ra_from_PTFE"].iloc[0],
             df_ra["Ra_from_PTFE"].iloc[min(4, len(df_ra)-1)])

    return png_path, csv_path


# ---------------------------------------------------------------------------
# Density static PNG (matplotlib hexbin, no ester/amide split)
# ---------------------------------------------------------------------------

def visualize_density_static(
    df: pd.DataFrame,
    out_dir: str | Path = "results",
    prefix: str = "hsp",
    d_col: str = "delta_D_corr",
    p_col: str = "delta_P_corr",
    h_col: str = "delta_H_corr",
    gridsize: int = 40,
) -> Path | None:
    """
    Static PNG: 3D scatter + 3 × 2D hexbin density projections.
    No ester/amide colour split — light = sparse, dark = dense.
    Truncated Blues colormap + log normalization for even gradient.
    """
    from matplotlib.colors import LogNorm, LinearSegmentedColormap

    # Truncated Blues: skip the near-white bottom 20% so even sparse bins
    # are a visible light blue rather than almost white.
    _blues = plt.cm.Blues(np.linspace(0.2, 1.0, 256))
    CMAP = LinearSegmentedColormap.from_list("Blues_dense", _blues)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_plot = df.dropna(subset=[d_col, p_col, h_col]).copy()
    if df_plot.empty:
        log.warning("No valid HSP rows to plot.")
        return None

    dD = df_plot[d_col].values
    dP = df_plot[p_col].values
    dH = df_plot[h_col].values

    # -- 3D density: bin all points, colour sampled subset by bin count --
    rng = np.random.default_rng(42)
    n_3d = min(5000, len(dD))
    idx_3d = rng.choice(len(dD), n_3d, replace=False)
    dD_3d, dP_3d, dH_3d = dD[idx_3d], dP[idx_3d], dH[idx_3d]

    hist3d, edges3d = np.histogramdd(
        np.column_stack([dD, dP, dH]), bins=30
    )
    def _bin_density(d, p, h):
        di = np.clip(np.digitize(d, edges3d[0]) - 1, 0, hist3d.shape[0] - 1)
        pi = np.clip(np.digitize(p, edges3d[1]) - 1, 0, hist3d.shape[1] - 1)
        hi = np.clip(np.digitize(h, edges3d[2]) - 1, 0, hist3d.shape[2] - 1)
        return hist3d[di, pi, hi]

    density_3d = _bin_density(dD_3d, dP_3d, dH_3d)

    fig = plt.figure(figsize=(24, 5), layout="constrained")
    fig.suptitle("Hansen Solubility Parameter Space — Norbornene Monomers", fontsize=13)

    # -- 3D scatter coloured by density (log norm so sparse points aren't washed out) --
    ax3d = fig.add_subplot(141, projection="3d")
    sc = ax3d.scatter(dD_3d, dP_3d, dH_3d, c=density_3d, cmap=CMAP,
                      norm=LogNorm(vmin=max(1, density_3d.min()), vmax=density_3d.max()),
                      s=4, alpha=0.7, linewidths=0)
    plt.colorbar(sc, ax=ax3d, label="Count (log scale)", pad=0.1, shrink=0.6)
    for lname, (ld, lp, lh) in LANDMARKS.items():
        ax3d.scatter(ld, lp, lh, marker="*", s=160, zorder=10,
                     color="red", edgecolors="darkred", linewidths=0.5,
                     depthshade=False)
        ax3d.text(ld, lp, lh + 0.5, lname, fontsize=7, zorder=10)
    ax3d.set_xlabel("δD (MPa$^{0.5}$)", fontsize=8)
    ax3d.set_ylabel("δP (MPa$^{0.5}$)", fontsize=8)
    ax3d.set_zlabel("δH (MPa$^{0.5}$)", fontsize=8)
    ax3d.set_title("3D Hansen space", fontsize=9)

    # -- 2D hexbin projections (LogNorm + truncated Blues) --
    proj_axes = [
        (fig.add_subplot(142), dD, dP, "δD (MPa$^{0.5}$)", "δP (MPa$^{0.5}$)", "δD–δP",
         lambda d, p, h: d, lambda d, p, h: p),
        (fig.add_subplot(143), dD, dH, "δD (MPa$^{0.5}$)", "δH (MPa$^{0.5}$)", "δD–δH",
         lambda d, p, h: d, lambda d, p, h: h),
        (fig.add_subplot(144), dP, dH, "δP (MPa$^{0.5}$)", "δH (MPa$^{0.5}$)", "δP–δH",
         lambda d, p, h: p, lambda d, p, h: h),
    ]
    for ax, x, y, xl, yl, title, get_lx, get_ly in proj_axes:
        hb = ax.hexbin(x, y, gridsize=gridsize, cmap=CMAP, mincnt=1,
                       norm=LogNorm(vmin=1), linewidths=0.2)
        cb = plt.colorbar(hb, ax=ax, pad=0.02)
        cb.set_label("Count", fontsize=7)
        for lname, coords in LANDMARKS.items():
            lx, ly = get_lx(*coords), get_ly(*coords)
            ax.scatter(lx, ly, marker="*", s=120, zorder=10,
                       color="red", edgecolors="darkred", linewidths=0.5)
            ax.annotate(lname, (lx, ly), fontsize=6, zorder=11,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(xl, fontsize=8)
        ax.set_ylabel(yl, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)

    fig.savefig(out_dir / f"{prefix}_hansen_space_density.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    png_path = out_dir / f"{prefix}_hansen_space_density.png"
    log.info("Density PNG saved to %s", png_path)
    return png_path


# ---------------------------------------------------------------------------
# Interactive HTML visualization (Plotly + hover molecular structure)
# ---------------------------------------------------------------------------

def visualize_interactive(
    df: pd.DataFrame,
    out_dir: str | Path = "results",
    prefix: str = "hsp",
    d_col: str = "delta_D_corr",
    p_col: str = "delta_P_corr",
    h_col: str = "delta_H_corr",
    smiles_col: str = "monomer_smiles",
    nbins: int = 40,
    max_hover_points: int = 2000,
    img_size: tuple[int, int] = (160, 130),
    seed: int = 42,
) -> Path | None:
    """
    Interactive HTML Hansen space visualization.

    - Hexbin-style density heatmaps (all points) for the three 2D projections.
    - Random sample of up to *max_hover_points* shown as invisible scatter
      points; hovering reveals the molecule structure.
    - 3D scatter for spatial overview (sampled points, colour = HSP density
      kernel, no ester/amide distinction).
    - Output: <prefix>_hansen_space_interactive.html

    Requires: plotly  (pip install plotly)
              rdkit   (conda install -c conda-forge rdkit)
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        log.error("plotly not installed — run: pip install plotly")
        return None

    import base64, io, json

    # Truncated Blues: 0→light-but-visible-blue, 1→dark navy.
    # Skipping the near-white bottom ensures sparse bins are still clearly visible.
    PLOTLY_BLUES = [
        [0.00, "#c6dbef"],
        [0.25, "#9ecae1"],
        [0.50, "#4292c6"],
        [0.75, "#2171b5"],
        [1.00, "#08306b"],
    ]

    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        HAS_RDKIT = True
    except ImportError:
        HAS_RDKIT = False
        log.warning("RDKit not found — hover will show SMILES text only.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- full dataset (density heatmap) ---
    df_all = df.dropna(subset=[d_col, p_col, h_col]).copy().reset_index(drop=True)
    if df_all.empty:
        log.warning("No valid HSP rows to plot.")
        return None

    dD_all = df_all[d_col].values
    dP_all = df_all[p_col].values
    dH_all = df_all[h_col].values

    # --- sampled subset (hover layer) ---
    rng = np.random.default_rng(seed)
    n = len(df_all)
    hover_idx = (
        rng.choice(n, size=max_hover_points, replace=False)
        if n > max_hover_points
        else np.arange(n)
    )
    df_hover = df_all.iloc[hover_idx].reset_index(drop=True)
    dD_h = df_hover[d_col].values
    dP_h = df_hover[p_col].values
    dH_h = df_hover[h_col].values
    smiles_h = (
        df_hover[smiles_col].values
        if smiles_col in df_hover.columns
        else np.array([""] * len(df_hover))
    )

    # --- molecule images for hover subset ---
    def smiles_to_b64(smi: str) -> str:
        if not HAS_RDKIT or not smi:
            return ""
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                return ""
            img = Draw.MolToImage(mol, size=img_size)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            return ""

    log.info("Generating %d molecule images for hover layer…", len(df_hover))
    mol_b64 = [smiles_to_b64(s) for s in smiles_h]
    hover_text = [
        f"δD={dD_h[i]:.2f}, δP={dP_h[i]:.2f}, δH={dH_h[i]:.2f}<br>"
        f"<span style='font-size:9px'>{smiles_h[i]}</span>"
        for i in range(len(df_hover))
    ]
    pt_indices = list(range(len(df_hover)))

    # --- build figure ---
    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=["3D Hansen Space", "δD – δP", "δD – δH", "δP – δH"],
        specs=[[{"type": "scatter3d"}, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
        horizontal_spacing=0.05,
    )

    # --- 3D density: colour sampled points by their local bin count ---
    hist3d, edges3d = np.histogramdd(
        np.column_stack([dD_all, dP_all, dH_all]), bins=30
    )
    def _bin_density_3d(d, p, h):
        di = np.clip(np.digitize(d, edges3d[0]) - 1, 0, hist3d.shape[0] - 1)
        pi = np.clip(np.digitize(p, edges3d[1]) - 1, 0, hist3d.shape[1] - 1)
        hi = np.clip(np.digitize(h, edges3d[2]) - 1, 0, hist3d.shape[2] - 1)
        return hist3d[di, pi, hi]

    density_h = _bin_density_3d(dD_h, dP_h, dH_h)

    # 3D scatter (sampled, coloured by density)
    fig.add_trace(
        go.Scatter3d(
            x=dD_h, y=dP_h, z=dH_h,
            mode="markers",
            marker=dict(
                size=2, opacity=0.7,
                color=np.log1p(density_h),
                colorscale=PLOTLY_BLUES,
                showscale=True,
                colorbar=dict(title="Count (log)", thickness=10, len=0.5, x=0.22),
            ),
            customdata=pt_indices,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            name="Monomers",
        ),
        row=1, col=1,
    )
    # 3D landmarks
    for lname, (ld, lp, lh) in LANDMARKS.items():
        fig.add_trace(
            go.Scatter3d(
                x=[ld], y=[lp], z=[lh],
                mode="markers+text",
                marker=dict(symbol="diamond", size=5, color="red"),
                text=[lname], textposition="top center",
                hoverinfo="text",
                showlegend=False,
            ),
            row=1, col=1,
        )

    # 2D projections
    def _landmark_xy(axis_label: str) -> list[float]:
        """Extract landmark coordinate for the given axis label."""
        vals = []
        for ld, lp, lh in LANDMARKS.values():
            if axis_label.startswith("δD"):
                vals.append(ld)
            elif axis_label.startswith("δP"):
                vals.append(lp)
            else:
                vals.append(lh)
        return vals

    proj_configs = [
        (2, dD_all, dP_all, dD_h, dP_h, "δD (MPa½)", "δP (MPa½)"),
        (3, dD_all, dH_all, dD_h, dH_h, "δD (MPa½)", "δH (MPa½)"),
        (4, dP_all, dH_all, dP_h, dH_h, "δP (MPa½)", "δH (MPa½)"),
    ]

    for col, x_all, y_all, x_h, y_h, xl, yl in proj_configs:
        show_cb = col == 4

        # density heatmap: precompute histogram, log1p-transform for even gradient
        counts, xedges, yedges = np.histogram2d(x_all, y_all, bins=nbins)
        counts_log = np.log1p(counts).T  # transpose: shape (ny, nx) for go.Heatmap
        # Actual-count tick labels for colorbar
        log_max = counts_log.max()
        tick_vals = np.linspace(0, log_max, 6)
        tick_text = [f"{int(np.expm1(v)):,}" for v in tick_vals]
        fig.add_trace(
            go.Heatmap(
                z=counts_log,
                x=xedges,
                y=yedges,
                colorscale=PLOTLY_BLUES,
                showscale=show_cb,
                colorbar=dict(
                    title="Count",
                    thickness=12, len=0.6,
                    tickvals=tick_vals,
                    ticktext=tick_text,
                ) if show_cb else None,
                hoverinfo="skip",
                name="Density",
            ),
            row=1, col=col,
        )

        # transparent scatter (hover layer, sampled)
        fig.add_trace(
            go.Scatter(
                x=x_h, y=y_h,
                mode="markers",
                marker=dict(size=8, opacity=0.01, color="white"),
                customdata=pt_indices,
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                hoverlabel=dict(bgcolor="white", font_size=11),
                showlegend=False,
                name="",
            ),
            row=1, col=col,
        )

        # landmarks
        lx = _landmark_xy(xl)
        ly = _landmark_xy(yl)
        fig.add_trace(
            go.Scatter(
                x=lx, y=ly,
                mode="markers+text",
                marker=dict(
                    symbol="star", size=14, color="red",
                    line=dict(width=1, color="darkred"),
                ),
                text=list(LANDMARKS.keys()),
                textposition="top center",
                textfont=dict(size=9),
                hoverinfo="text",
                showlegend=False,
            ),
            row=1, col=col,
        )

        fig.update_xaxes(title_text=xl, row=1, col=col, title_font=dict(size=11))
        fig.update_yaxes(title_text=yl, row=1, col=col, title_font=dict(size=11))

    fig.update_layout(
        title=dict(
            text="Hansen Solubility Parameter Space — Norbornene Monomers"
                 f"<br><sup>{n:,} compounds total · {len(df_hover):,} shown for hover</sup>",
            font=dict(size=14),
        ),
        height=560,
        width=1900,
        hovermode="closest",
        showlegend=False,
        template="plotly_white",
        margin=dict(t=80, b=40, l=40, r=40),
    )

    # --- inject JavaScript for molecule hover panel ---
    mol_images_js = json.dumps(mol_b64)
    js = f"""
(function () {{
    const molImages = {mol_images_js};
    const gd = document.getElementsByClassName('plotly-graph-div')[0];

    // floating molecule panel
    const panel = document.createElement('div');
    panel.id = 'hsp-mol-panel';
    panel.style.cssText = [
        'position:fixed',
        'display:none',
        'background:#fff',
        'border:1px solid #bbb',
        'border-radius:8px',
        'padding:8px 10px',
        'box-shadow:3px 3px 10px rgba(0,0,0,0.25)',
        'z-index:99999',
        'pointer-events:none',
        'max-width:200px',
        'text-align:center',
    ].join(';');
    panel.innerHTML =
        '<img id="hsp-mol-img" src="" style="width:160px;height:130px;display:none;">' +
        '<div id="hsp-mol-smi" style="font-size:8px;color:#444;word-break:break-all;margin-top:4px;"></div>';
    document.body.appendChild(panel);

    gd.on('plotly_hover', function (ev) {{
        const pt = ev.points[0];
        const idx = pt.customdata;
        if (idx === undefined || idx === null) return;

        const img = document.getElementById('hsp-mol-img');
        const smi = document.getElementById('hsp-mol-smi');
        const b64 = molImages[idx];
        if (b64) {{
            img.src = 'data:image/png;base64,' + b64;
            img.style.display = 'block';
        }} else {{
            img.style.display = 'none';
        }}

        // Position near cursor
        const e = ev.event || {{}};
        let cx = (e.clientX !== undefined) ? e.clientX : window.innerWidth / 2;
        let cy = (e.clientY !== undefined) ? e.clientY : window.innerHeight / 2;
        let left = cx + 18;
        let top  = cy - 10;
        if (left + 210 > window.innerWidth)  left = cx - 210;
        if (top  + 180 > window.innerHeight) top  = cy - 180;
        panel.style.left = left + 'px';
        panel.style.top  = top  + 'px';
        panel.style.display = 'block';
    }});

    gd.on('plotly_unhover', function () {{
        document.getElementById('hsp-mol-panel').style.display = 'none';
    }});
}})();
"""

    html_path = out_dir / f"{prefix}_hansen_space_interactive.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn", post_script=js)
    size_mb = html_path.stat().st_size / 1e6
    log.info("Interactive plot saved to %s  (%.1f MB)", html_path, size_mb)
    return html_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   default="results/monomers_hsp_corrected.csv")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--top_n",   type=int, default=50,
                        help="Print top N closest-to-PTFE monomers")
    parser.add_argument("--interactive", action="store_true",
                        help="Generate interactive HTML figure instead of PNG")
    parser.add_argument("--max_hover", type=int, default=2000,
                        help="Max number of molecules shown in hover layer (default 2000)")
    parser.add_argument("--nbins",  type=int, default=40,
                        help="Number of bins per axis in density heatmap (default 40)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = pd.read_csv(args.input)

    # Fall back to uncorrected columns if corrected not present
    d_col = "delta_D_corr" if "delta_D_corr" in df.columns else "delta_D"
    p_col = "delta_P_corr" if "delta_P_corr" in df.columns else "delta_P"
    h_col = "delta_H_corr" if "delta_H_corr" in df.columns else "delta_H"

    if args.interactive:
        visualize_density_static(
            df, out_dir=args.out_dir,
            d_col=d_col, p_col=p_col, h_col=h_col,
            gridsize=args.nbins,
        )
        visualize_interactive(
            df, out_dir=args.out_dir,
            d_col=d_col, p_col=p_col, h_col=h_col,
            max_hover_points=args.max_hover,
            nbins=args.nbins,
        )
    else:
        png, csv = visualize(df, out_dir=args.out_dir, d_col=d_col, p_col=p_col, h_col=h_col)

        if csv:
            df_top = pd.read_csv(csv).head(args.top_n)
            print(f"\nTop {args.top_n} monomers closest to PTFE in Hansen space:")
            print(df_top[["monomer_smiles", "linkage", d_col, p_col, h_col, "Ra_from_PTFE"]].to_string(index=False))
