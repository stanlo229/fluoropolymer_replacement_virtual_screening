#!/usr/bin/env python3
"""
End-to-end SSIP pipeline for norbornene virtual screening.

Stages
------
  1. generate   — build norbornene+sidechain SMILES library
  2. geom        — MACE-OFF23 geometry optimisation
  3. charges     — AIMNet2 MBIS charge prediction
  4. surfaces    — Shrake-Rupley surface + Coulomb MEPS
  5. calibrate   — fit E_max/E_min → α/β (run once; skip if calibration.json exists)
  6. footprint   — Hunter group SSIP footprinting → ei, α, β values

All intermediate results are saved to disk; individual stages can be rerun
independently or skipped with --resume.

Usage examples
--------------
    # Full run (first time):
    python run_pipeline.py --catalogues ../Dataset/catalogues.csv \
                           --workdir run_01 --device cuda

    # Resume after interruption:
    python run_pipeline.py --catalogues ../Dataset/catalogues.csv \
                           --workdir run_01 --device cpu --resume

    # Skip library generation (library.csv already exists):
    python run_pipeline.py --workdir run_01 --skip generate --device cpu --resume

    # Only footprint (surfaces already computed, calibration.json exists):
    python run_pipeline.py --workdir run_01 --only footprint

Output
------
    <workdir>/
      library.csv            SMILES of all norbornene products
      geoms/<mol_id>.xyz     MACE-OFF23 optimised geometries
      charges/<mol_id>.npz   AIMNet2 MBIS charges
      surfaces/<mol_id>.npz  vdW surface + Coulomb MEPS
      calibration.json       α/β calibration coefficients
      ssips/<mol_id>.json    per-molecule SSIP detail
      ssip_results.csv       final results (one row per molecule)
"""

import argparse
import sys
import time
from pathlib import Path

ALL_STAGES = ["generate", "geom", "charges", "surfaces", "calibrate", "footprint"]


def stage_generate(args, workdir: Path):
    import generate_library as gl
    out = workdir / "library.csv"
    if out.exists() and args.resume:
        print(f"[generate] Skipping — {out} exists")
        return
    print("[generate] Building norbornene library …")
    lib = gl.build_library(
        args.catalogues,
        isomers=args.isomers,
        compound_types=args.compound_types,
        include_reference=not args.no_reference,
    )
    if lib.empty:
        sys.exit("ERROR: no products generated — check --catalogues path")
    lib.to_csv(out, index=False)
    print(f"[generate] {len(lib):,} molecules → {out}")


def stage_geom(args, workdir: Path):
    import geometry_opt as go
    import pandas as pd

    geom_dir = workdir / "geoms"
    geom_dir.mkdir(exist_ok=True)

    lib_path = workdir / "library.csv"
    if not lib_path.exists():
        sys.exit(f"ERROR: {lib_path} not found — run 'generate' stage first")

    lib = pd.read_csv(lib_path)
    print(f"[geom] Optimising {len(lib):,} molecules with MACE-OFF23 …")

    results = []
    for _, row in lib.iterrows():
        mol_id   = str(row["mol_id"])
        xyz_path = geom_dir / f"{mol_id}.xyz"
        if args.resume and xyz_path.exists():
            results.append({"mol_id": mol_id, "converged": True,
                             "xyz_path": str(xyz_path), "error": None})
            continue
        res = go.optimise_molecule(str(row["smiles"]), mol_id, geom_dir, args.device)
        results.append(res)

    import pandas as pd
    summary = pd.DataFrame(results)
    summary.to_csv(geom_dir / "opt_summary.csv", index=False)
    n_ok  = summary["converged"].sum()
    n_err = summary["error"].notna().sum()
    print(f"[geom] Converged: {n_ok}/{len(summary)}  Errors: {n_err}")


def stage_charges(args, workdir: Path):
    import charge_predict as cp

    geom_dir   = workdir / "geoms"
    charge_dir = workdir / "charges"
    charge_dir.mkdir(exist_ok=True)

    xyz_files = sorted(geom_dir.glob("*.xyz"))
    if not xyz_files:
        sys.exit(f"ERROR: no .xyz files in {geom_dir}")

    print(f"[charges] Predicting AIMNet2 charges for {len(xyz_files)} molecules …")
    model, route = cp.load_aimnet2(args.device)

    errors = []
    for xyz in xyz_files:
        mol_id  = xyz.stem
        npz_out = charge_dir / f"{mol_id}.npz"
        if args.resume and npz_out.exists():
            continue

        parsed = cp.load_xyz(xyz)
        if parsed is None:
            errors.append(mol_id)
            continue

        numbers, positions = parsed
        try:
            charges = cp.predict_charges(model, route, numbers, positions, 0)
        except Exception as exc:
            errors.append(f"{mol_id}: {exc}")
            continue

        import numpy as np
        np.savez(npz_out, numbers=numbers, positions=positions,
                 charges=charges, mol_charge=np.array(0))

    print(f"[charges] Done. Errors: {len(errors)}")


def stage_surfaces(args, workdir: Path):
    import meps_surface as ms
    import numpy as np
    import pandas as pd

    charge_dir  = workdir / "charges"
    surface_dir = workdir / "surfaces"
    surface_dir.mkdir(exist_ok=True)

    npz_files = sorted(charge_dir.glob("*.npz"))
    if not npz_files:
        sys.exit(f"ERROR: no .npz charge files in {charge_dir}")

    print(f"[surfaces] Computing MEPS for {len(npz_files)} molecules …")
    summary = []
    for npz in npz_files:
        mol_id  = npz.stem
        out_npz = surface_dir / f"{mol_id}.npz"
        if args.resume and out_npz.exists():
            continue

        result = ms.compute_meps(npz)
        if "error" in result:
            summary.append({"mol_id": mol_id, "error": result["error"]})
            continue

        np.savez_compressed(out_npz, **{k: result[k] for k in result})
        summary.append({"mol_id": mol_id,
                         "E_max": result["E_max"], "E_min": result["E_min"],
                         "AvdW": result["AvdW"]})

    pd.DataFrame(summary).to_csv(surface_dir / "meps_summary.csv", index=False)
    print(f"[surfaces] Done. {len(summary)} molecules processed.")


def stage_calibrate(args, workdir: Path):
    import calibrate_ssip as cs

    calib_path = workdir / "calibration.json"
    if calib_path.exists() and args.resume:
        print(f"[calibrate] Skipping — {calib_path} exists")
        return

    # Use surfaces in workdir/surfaces_calib/ if available; otherwise prompt
    calib_surf_dir = workdir / "surfaces_calib"
    if not calib_surf_dir.exists() or not list(calib_surf_dir.glob("*.npz")):
        print("[calibrate] Running pipeline on calibration set …")
        calib_work = workdir / "calib_workdir"
        cs._run_pipeline_for_calib(calib_work, args.device)
        calib_surf_dir = calib_work / "surfaces"

    cs.run_calibration(calib_surf_dir, calib_path)
    print(f"[calibrate] Calibration saved to {calib_path}")


def stage_footprint(args, workdir: Path):
    import ssip_footprint as sf
    import json
    import numpy as np
    import pandas as pd

    surface_dir = workdir / "surfaces"
    calib_path  = workdir / "calibration.json"
    ssip_dir    = workdir / "ssips"
    ssip_dir.mkdir(exist_ok=True)

    if not calib_path.exists():
        sys.exit(f"ERROR: {calib_path} not found — run 'calibrate' stage first")

    calib = json.loads(calib_path.read_text())
    lib_path = workdir / "library.csv"
    smiles_map: dict[str, str] = {}
    if lib_path.exists():
        lib = pd.read_csv(lib_path)
        smiles_map = dict(zip(lib["mol_id"].astype(str), lib["smiles"].astype(str)))

    npz_files = sorted(surface_dir.glob("*.npz"))
    if not npz_files:
        sys.exit(f"ERROR: no surface .npz files in {surface_dir}")

    print(f"[footprint] Footprinting {len(npz_files)} molecules …")
    rows = []
    for npz in npz_files:
        mol_id = npz.stem
        data   = dict(np.load(npz, allow_pickle=False))
        smiles = smiles_map.get(mol_id)

        ssips   = sf.footprint(data, calib, smiles)
        summary = sf.ssip_summary(ssips, float(data.get("AvdW", 0.0)))
        summary["mol_id"] = mol_id

        json_out = ssip_dir / f"{mol_id}.json"
        json_out.write_text(json.dumps({"mol_id": mol_id, "ssips": ssips}, indent=2))
        rows.append(summary)

    # Merge with library metadata
    results = pd.DataFrame(rows)
    if lib_path.exists():
        lib = pd.read_csv(lib_path)[
            ["mol_id", "smiles", "scaffold_isomer", "compound_type",
             "source", "link", "price_per_gram"]
        ]
        results = lib.merge(results, on="mol_id", how="right")

    out_csv = workdir / "ssip_results.csv"
    results.to_csv(out_csv, index=False)
    print(f"[footprint] Saved {len(results):,} rows → {out_csv}")
    print(f"  α_max range: {results['alpha_max'].min():.2f} – {results['alpha_max'].max():.2f}")
    print(f"  β_max range: {results['beta_max'].min():.2f} – {results['beta_max'].max():.2f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end MLIP SSIP pipeline for norbornene virtual screening."
    )
    parser.add_argument("--catalogues",      default="../Dataset/catalogues.csv")
    parser.add_argument("--workdir",         default="run_01",
                        help="Working directory for all outputs")
    parser.add_argument("--device",          default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--resume",          action="store_true",
                        help="Skip stages / molecules whose output files already exist")
    parser.add_argument("--skip",            nargs="+", default=[],
                        choices=ALL_STAGES, metavar="STAGE",
                        help="Stages to skip")
    parser.add_argument("--only",            nargs="+", default=[],
                        choices=ALL_STAGES, metavar="STAGE",
                        help="Run only these stages (overrides --skip)")
    parser.add_argument("--isomers",         nargs="+", default=["endo", "exo"],
                        choices=["endo", "exo"])
    parser.add_argument("--compound_types",  nargs="+",
                        default=["ester", "amide", "diester", "diamide"],
                        choices=["ester", "amide", "diester", "diamide"])
    parser.add_argument("--no_reference",    action="store_true",
                        help="Omit reference molecules (water, hexadecane, PFAS, …)")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # Add _code/ to path so sibling modules are importable
    sys.path.insert(0, str(Path(__file__).parent))

    stages_to_run = args.only if args.only else [s for s in ALL_STAGES if s not in args.skip]
    print(f"Running stages: {stages_to_run}")
    print(f"Working directory: {workdir.resolve()}\n")

    stage_fns = {
        "generate":  stage_generate,
        "geom":      stage_geom,
        "charges":   stage_charges,
        "surfaces":  stage_surfaces,
        "calibrate": stage_calibrate,
        "footprint": stage_footprint,
    }

    for stage in ALL_STAGES:
        if stage not in stages_to_run:
            continue
        t0 = time.time()
        stage_fns[stage](args, workdir)
        print(f"  [{stage}] completed in {time.time() - t0:.1f}s\n")

    print("Pipeline complete.")
    print(f"Final results: {workdir / 'ssip_results.csv'}")


if __name__ == "__main__":
    main()
