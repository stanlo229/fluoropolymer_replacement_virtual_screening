"""
run_pipeline.py

CLI entry point for the HSP prediction pipeline.

Steps:
  1. generate_monomers   → results/monomers.csv
  2. hsp_calculator      → results/monomers_hsp.csv
  3. si_correction       → results/monomers_hsp_corrected.csv + si_calibration.json
  4. visualize_hsp       → results/hsp_hansen_space.png + hsp_Ra_ranked.csv

Usage:
  python run_pipeline.py [--catalogues PATH] [--out_dir PATH] [--top_n 50]
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_pipeline")


def _timer(label: str, t0: float) -> float:
    elapsed = time.perf_counter() - t0
    log.info("  %s done in %.1f s", label, elapsed)
    return time.perf_counter()


def main():
    parser = argparse.ArgumentParser(description="HSP prediction pipeline for norbornene monomers")
    parser.add_argument("--catalogues", default="../dataset/catalogues.csv",
                        help="Path to input catalogues.csv")
    parser.add_argument("--out_dir",    default="results",
                        help="Directory for all output files")
    parser.add_argument("--top_n",      type=int, default=50,
                        help="Print top N closest-to-PTFE monomers")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1: Generate monomers
    # ------------------------------------------------------------------
    log.info("=== Step 1: Generating monomers ===")
    t0 = time.perf_counter()
    from generate_monomers import generate_monomers

    monomers_path  = out_dir / "monomers.csv"
    skipped_path   = out_dir / "skipped.csv"
    df_monomers = generate_monomers(
        catalogues_path=args.catalogues,
        out_path=monomers_path,
        skipped_path=skipped_path,
    )
    t0 = _timer("generate_monomers", t0)

    if df_monomers.empty:
        log.error("No monomers generated — aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Compute HSP
    # ------------------------------------------------------------------
    log.info("=== Step 2: Computing HSP (Stefanis–Panayiotou) ===")
    t0 = time.perf_counter()
    from hsp_calculator import calculate_hsp_batch
    from tqdm import tqdm

    # Process in chunks for progress reporting
    CHUNK = 5000
    chunks = [df_monomers.iloc[i:i+CHUNK] for i in range(0, len(df_monomers), CHUNK)]
    results = []
    for chunk in tqdm(chunks, desc="HSP chunks", unit="chunk"):
        results.append(calculate_hsp_batch(chunk))
    import pandas as pd
    df_hsp = pd.concat(results, ignore_index=True)

    hsp_path = out_dir / "monomers_hsp.csv"
    df_hsp.to_csv(hsp_path, index=False)
    n_ok  = df_hsp["delta_D"].notna().sum()
    n_err = df_hsp["hsp_error"].notna().sum()
    log.info("  HSP computed: %d OK, %d errors", n_ok, n_err)
    t0 = _timer("hsp_calculator", t0)

    # ------------------------------------------------------------------
    # Step 3: Si correction
    # ------------------------------------------------------------------
    log.info("=== Step 3: Si correction ===")
    t0 = time.perf_counter()
    from si_correction import calibrate_si_correction, apply_si_correction

    cal_path = out_dir / "si_correction_calibration.json"
    calibration = calibrate_si_correction(save_path=cal_path)
    log.info(
        "  Correction per Si atom: ΔδD=%.3f  ΔδP=%.3f  ΔδH=%.3f  MAR=%.3f MPa^0.5",
        calibration["delta_D_per_si"],
        calibration["delta_P_per_si"],
        calibration["delta_H_per_si"],
        calibration["mean_abs_residual"],
    )

    df_corr = apply_si_correction(df_hsp, calibration)
    corr_path = out_dir / "monomers_hsp_corrected.csv"
    df_corr.to_csv(corr_path, index=False)
    t0 = _timer("si_correction", t0)

    # ------------------------------------------------------------------
    # Step 4: Visualize
    # ------------------------------------------------------------------
    log.info("=== Step 4: Visualization ===")
    t0 = time.perf_counter()
    from visualize_hsp import visualize

    d_col = "delta_D_corr" if "delta_D_corr" in df_corr.columns else "delta_D"
    p_col = "delta_P_corr" if "delta_P_corr" in df_corr.columns else "delta_P"
    h_col = "delta_H_corr" if "delta_H_corr" in df_corr.columns else "delta_H"

    png_path, ra_csv = visualize(
        df_corr,
        out_dir=out_dir,
        d_col=d_col, p_col=p_col, h_col=h_col,
    )
    t0 = _timer("visualize_hsp", t0)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = time.perf_counter() - t_total
    log.info("=== Pipeline complete in %.1f s ===", total)
    log.info("Outputs in %s:", out_dir)
    for f in sorted(out_dir.iterdir()):
        log.info("  %s", f.name)

    if ra_csv:
        df_top = pd.read_csv(ra_csv).head(args.top_n)
        print(f"\nTop {args.top_n} monomers closest to PTFE (Ra, MPa^0.5):")
        cols = ["monomer_smiles", "linkage", d_col, p_col, h_col, "Ra_from_PTFE"]
        cols = [c for c in cols if c in df_top.columns]
        print(df_top[cols].to_string(index=False))


if __name__ == "__main__":
    main()
