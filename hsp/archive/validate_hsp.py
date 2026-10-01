"""
validate_hsp.py

Validate the Stefanis-Panayiotou HSP calculator against the HSPiP reference dataset.

Workflow:
  1. Download HSPiPDataSet.xls from hansen-solubility.com (or use cached copy).
  2. For each compound with known δD, δP, δH: fetch canonical SMILES from PubChem by CAS.
  3. Run compute_hsp() on each SMILES and compare predicted vs experimental.
  4. Report MAE/RMSE per component and write scatter plots.

Usage:
    python validate_hsp.py [--dataset PATH] [--cache PATH] [--max N] [--out DIR]

    --dataset  Path to local HSPiPDataSet.xls (downloaded automatically if not given)
    --cache    Path to CAS→SMILES JSON cache (default: results/validation/smiles_cache.json)
    --max      Maximum number of compounds to evaluate (default: all)
    --out      Output directory (default: results/validation/)
"""

import argparse
import json
import logging
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DATASET_URL = "https://hansen-solubility.com/contents/HSPiPDataSet.xls"
PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas}/property/CanonicalSMILES/JSON"
PUBCHEM_DELAY = 0.2  # seconds between requests to respect rate limit


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path | None) -> pd.DataFrame:
    """Download (or load cached) HSPiPDataSet.xls and return a cleaned DataFrame."""
    if path is not None and Path(path).exists():
        log.info("Loading dataset from %s", path)
        df = pd.read_excel(path)
    else:
        log.info("Downloading HSPiP dataset from %s", DATASET_URL)
        resp = requests.get(DATASET_URL, timeout=60)
        resp.raise_for_status()
        import io
        df = pd.read_excel(io.BytesIO(resp.content))
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                f.write(resp.content)
            log.info("Cached dataset to %s", path)

    log.info("Raw dataset: %d rows, columns: %s", len(df), list(df.columns))
    return df


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name from candidates that exists in df (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def extract_reference_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Name, CAS, dD, dP, dH from the raw XLS DataFrame.
    Column names vary between HSPiP versions — try common alternatives.
    """
    col_name = _find_col(df, ["Name", "Chemical Name", "Compound", "name"])
    col_cas  = _find_col(df, ["CAS", "CAS No", "CAS Number", "cas"])
    col_dD   = _find_col(df, ["dD", "delta_D", "D", "deltaD", "Hansen D"])
    col_dP   = _find_col(df, ["dP", "delta_P", "P", "deltaP", "Hansen P"])
    col_dH   = _find_col(df, ["dH", "delta_H", "H", "deltaH", "Hansen H"])

    missing = [k for k, v in {"Name": col_name, "CAS": col_cas,
                               "dD": col_dD, "dP": col_dP, "dH": col_dH}.items() if v is None]
    if missing:
        log.warning("Could not find columns for: %s — printing all column names for debugging:", missing)
        log.warning("%s", list(df.columns))
        raise ValueError(f"Missing expected columns: {missing}. See column list above.")

    out = pd.DataFrame({
        "name": df[col_name],
        "cas":  df[col_cas].astype(str).str.strip(),
        "exp_dD": pd.to_numeric(df[col_dD], errors="coerce"),
        "exp_dP": pd.to_numeric(df[col_dP], errors="coerce"),
        "exp_dH": pd.to_numeric(df[col_dH], errors="coerce"),
    })
    n_before = len(out)
    out = out.dropna(subset=["exp_dD", "exp_dP", "exp_dH"])
    out = out[out["cas"].str.match(r"^\d+-\d+-\d+$")]  # valid CAS format
    log.info("Retained %d / %d rows with complete experimental HSP + valid CAS", len(out), n_before)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CAS → SMILES via PubChem
# ---------------------------------------------------------------------------

def fetch_smiles_pubchem(cas: str) -> str | None:
    """Query PubChem for canonical SMILES given a CAS number. Returns None on failure."""
    url = PUBCHEM_URL.format(cas=cas)
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        return data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
    except Exception:
        return None


def resolve_smiles(df: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    """
    Add 'smiles' column to df by looking up each CAS in cache, then PubChem.
    Cache is a JSON dict {cas: smiles_or_null}.
    """
    # Load existing cache
    cache: dict[str, str | None] = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        log.info("Loaded %d cached CAS→SMILES entries", len(cache))

    cas_to_resolve = [cas for cas in df["cas"].unique() if cas not in cache]
    log.info("Fetching SMILES for %d compounds from PubChem...", len(cas_to_resolve))

    for i, cas in enumerate(cas_to_resolve):
        smi = fetch_smiles_pubchem(cas)
        cache[cas] = smi
        time.sleep(PUBCHEM_DELAY)
        if (i + 1) % 100 == 0:
            log.info("  %d / %d fetched", i + 1, len(cas_to_resolve))
            # Save cache incrementally
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(cache, f)

    # Final cache save
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f)

    df = df.copy()
    df["smiles"] = df["cas"].map(cache)
    n_resolved = df["smiles"].notna().sum()
    log.info("Resolved SMILES for %d / %d compounds", n_resolved, len(df))
    return df


# ---------------------------------------------------------------------------
# HSP prediction
# ---------------------------------------------------------------------------

def predict_hsp(df: pd.DataFrame) -> pd.DataFrame:
    """Run compute_hsp() for each row with a SMILES. Adds pred_dD/P/H columns."""
    from hsp_calculator import compute_hsp

    pred_dD, pred_dP, pred_dH, errors = [], [], [], []

    for smi in df["smiles"]:
        if not smi:
            pred_dD.append(None); pred_dP.append(None); pred_dH.append(None)
            errors.append("no SMILES")
            continue
        r = compute_hsp(str(smi))
        if r.error:
            pred_dD.append(None); pred_dP.append(None); pred_dH.append(None)
            errors.append(r.error)
        else:
            pred_dD.append(r.delta_D)
            pred_dP.append(r.delta_P)
            pred_dH.append(r.delta_H)
            errors.append(None)

    df = df.copy()
    df["pred_dD"] = pred_dD
    df["pred_dP"] = pred_dP
    df["pred_dH"] = pred_dH
    df["pred_error"] = errors
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute MAE and RMSE for each HSP component on rows with valid predictions."""
    ok = df.dropna(subset=["pred_dD", "pred_dP", "pred_dH"])
    metrics = {}
    for comp in ["dD", "dP", "dH"]:
        err = ok[f"pred_{comp}"] - ok[f"exp_{comp}"]
        metrics[comp] = {
            "n":    len(ok),
            "MAE":  float(err.abs().mean()),
            "RMSE": float(math.sqrt((err**2).mean())),
            "bias": float(err.mean()),
        }
    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def scatter_plot(df: pd.DataFrame, comp: str, out_dir: Path) -> None:
    ok = df.dropna(subset=[f"pred_{comp}", f"exp_{comp}"])
    x = ok[f"exp_{comp}"]
    y = ok[f"pred_{comp}"]
    err = y - x

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, s=8, alpha=0.4, color="steelblue")
    lim = [min(x.min(), y.min()) - 1, max(x.max(), y.max()) + 1]
    ax.plot(lim, lim, "k--", lw=0.8, label="y=x")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"Experimental δ{comp[-1]} (MPa½)")
    ax.set_ylabel(f"Predicted δ{comp[-1]} (MPa½)")
    mae  = err.abs().mean()
    rmse = math.sqrt((err**2).mean())
    ax.set_title(f"δ{comp[-1]}  |  MAE={mae:.2f}  RMSE={rmse:.2f}  n={len(ok)}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = out_dir / f"scatter_{comp}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate HSP calculator against HSPiP dataset.")
    parser.add_argument("--dataset", default=None,
                        help="Local path to HSPiPDataSet.xls (downloaded if absent)")
    parser.add_argument("--cache",   default="results/validation/smiles_cache.json")
    parser.add_argument("--max",     type=int, default=None,
                        help="Maximum number of compounds to evaluate")
    parser.add_argument("--out",     default="results/validation/")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache)

    # Step 1: Load reference data
    raw = load_dataset(args.dataset)
    df  = extract_reference_data(raw)

    if args.max:
        df = df.head(args.max)
        log.info("Limiting to %d compounds (--max)", args.max)

    # Step 2: Resolve SMILES
    df = resolve_smiles(df, cache_path)
    df = df[df["smiles"].notna()].reset_index(drop=True)
    log.info("%d compounds with SMILES to evaluate", len(df))

    # Step 3: Predict
    df = predict_hsp(df)

    # Step 4: Save results
    results_path = out_dir / "validation_results.csv"
    df.to_csv(results_path, index=False)
    log.info("Saved per-compound results to %s", results_path)

    # Step 5: Metrics + summary
    metrics = compute_metrics(df)
    summary_lines = [
        "HSP Calculator Validation — Stefanis-Panayiotou (2012)",
        f"Reference: HSPiP dataset ({metrics['dD']['n']} compounds with predictions)\n",
        f"{'Component':<12} {'MAE':>8} {'RMSE':>8} {'Bias':>8}",
        "-" * 42,
    ]
    for comp, label in [("dD", "δD"), ("dP", "δP"), ("dH", "δH")]:
        m = metrics[comp]
        summary_lines.append(f"{label:<12} {m['MAE']:>8.3f} {m['RMSE']:>8.3f} {m['bias']:>+8.3f}")
    summary_lines.append("\n(all values in MPa^0.5)")

    # High-error compounds
    ok = df.dropna(subset=["pred_dD", "pred_dP", "pred_dH"]).copy()
    ok["max_err"] = (ok[["pred_dD","pred_dP","pred_dH"]].values -
                     ok[["exp_dD","exp_dP","exp_dH"]].values).__abs__().max(axis=1)
    high_err = ok[ok["max_err"] > 3].sort_values("max_err", ascending=False).head(20)
    summary_lines.append(f"\nTop high-error compounds (|error| > 3 MPa^0.5): {len(ok[ok['max_err']>3])}")
    for _, r in high_err.iterrows():
        summary_lines.append(
            f"  {r['name'][:40]:<40}  pred=({r['pred_dD']:.1f},{r['pred_dP']:.1f},{r['pred_dH']:.1f})"
            f"  exp=({r['exp_dD']:.1f},{r['exp_dP']:.1f},{r['exp_dH']:.1f})"
        )

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    summary_path = out_dir / "validation_summary.txt"
    summary_path.write_text(summary_text)
    log.info("Saved summary to %s", summary_path)

    # Step 6: Scatter plots
    for comp in ["dD", "dP", "dH"]:
        scatter_plot(df, comp, out_dir)


if __name__ == "__main__":
    main()
