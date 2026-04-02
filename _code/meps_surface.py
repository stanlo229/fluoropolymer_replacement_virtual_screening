#!/usr/bin/env python3
"""
Molecular surface generation and electrostatic potential calculation.

Method
------
1. Generate a van der Waals surface using the Shrake-Rupley algorithm at a
   probe radius of 0 Å (i.e. the bare atomic vdW surface) scaled by 1.4×
   to approximate the 0.002 Bohr/Å³ electron-density isosurface used by the
   Hunter group.
2. Compute the Coulomb potential at each surface point from MBIS atomic charges:
       V(r) = Σ_i  q_i / |r − r_i|    (units: e/Å, converted to kJ mol⁻¹ below)
3. Assign each surface point to its nearest heavy atom (for functional-group
   scaling factor c in Hunter eq 16).
4. Return E_max, E_min (kJ mol⁻¹), surface area AvdW (Å²), and per-point data
   for downstream footprinting.

Conversion factor: 1 e/Å = 1389.35 kJ mol⁻¹  (= e²/(4πε₀ × 1 Å × N_A × 10⁻³))

vdW radii (Å) from Bondi 1964 / Alvarez 2013:
  H=1.20, C=1.70, N=1.55, O=1.52, F=1.47, P=1.80, S=1.80, Cl=1.75, Br=1.85, I=1.98

Usage:
    python meps_surface.py --chargedir charges/ --outdir surfaces/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# e/Å → kJ mol⁻¹   (Coulomb constant × e × N_A × 10⁻³ / Å)
_COULOMB_HARTREE_ANG = 1389.35   # kJ mol⁻¹ per (e²/Å)  [= 1/(4πε₀) in SI converted]

# Surface scale factor to approximate the 0.002 Bohr/Å³ isosurface
# (Calero et al. use B3LYP/6-31G* density; 1.4× vdW radius is a standard approximation)
_SURFACE_SCALE = 1.4

# Shrake-Rupley sphere resolution: points per atom sphere
_SR_N_POINTS = 194   # 194-point Lebedev grid — good balance of speed vs density

# vdW radii (Å) — atomic number → radius
_VDW_RADII: dict[int, float] = {
    1: 1.20,   # H
    6: 1.70,   # C
    7: 1.55,   # N
    8: 1.52,   # O
    9: 1.47,   # F
    15: 1.80,  # P
    16: 1.80,  # S
    17: 1.75,  # Cl
    35: 1.85,  # Br
    53: 1.98,  # I
}
_DEFAULT_VDW = 1.80   # fallback for unlisted elements


def _sphere_points(n: int) -> np.ndarray:
    """
    Generate ~n approximately uniformly distributed points on the unit sphere
    using the golden-angle (Fibonacci) spiral method.
    Returns (n, 3) array.
    """
    golden = (1 + 5 ** 0.5) / 2
    i      = np.arange(n)
    theta  = np.arccos(1 - 2 * (i + 0.5) / n)
    phi    = 2 * np.pi * i / golden
    return np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


_UNIT_SPHERE = _sphere_points(_SR_N_POINTS)   # shape (194, 3)


def shrake_rupley_surface(
    positions: np.ndarray,     # (N, 3) Å
    numbers:   np.ndarray,     # (N,)   atomic numbers
    scale:     float = _SURFACE_SCALE,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the solvent-accessible surface points (probe radius = 0).

    Returns:
        surf_pts  : (M, 3) Å   — surface point coordinates
        surf_atoms: (M,)   int  — index of nearest atom for each surface point
        area      : float  Å²  — total surface area (≈ AvdW)
    """
    n_atoms = len(positions)
    radii   = np.array([_VDW_RADII.get(int(z), _DEFAULT_VDW) * scale
                        for z in numbers])

    all_pts   = []
    all_atoms = []

    for i in range(n_atoms):
        r_i  = radii[i]
        pts  = positions[i] + r_i * _UNIT_SPHERE   # (194, 3)

        # Mask out points buried inside neighbouring atom spheres
        exposed = np.ones(len(pts), dtype=bool)
        for j in range(n_atoms):
            if j == i:
                continue
            r_j   = radii[j]
            dists = np.linalg.norm(pts - positions[j], axis=1)
            exposed &= (dists >= r_j)

        if exposed.any():
            all_pts.append(pts[exposed])
            all_atoms.append(np.full(exposed.sum(), i, dtype=np.int32))

    if not all_pts:
        return np.empty((0, 3)), np.empty(0, dtype=np.int32), 0.0

    surf_pts   = np.vstack(all_pts)
    surf_atoms = np.concatenate(all_atoms)

    # Area: fraction of exposed points × 4πr² per atom, summed
    area = 0.0
    for i in range(n_atoms):
        mask  = (surf_atoms == i)
        frac  = mask.sum() / _SR_N_POINTS
        r_i   = radii[i]
        area += frac * 4 * np.pi * r_i ** 2

    return surf_pts, surf_atoms, area


def coulomb_potential(
    surf_pts:  np.ndarray,   # (M, 3) Å
    positions: np.ndarray,   # (N, 3) Å
    charges:   np.ndarray,   # (N,)   e
) -> np.ndarray:
    """
    V(r_k) = Σ_i  q_i / |r_k − r_i|   in kJ mol⁻¹.

    Vectorised computation (may allocate M×N array — fine for <500 molecules
    with N < 200 atoms, M < 40 000 surface points).
    """
    # (M, N, 3)
    diff  = surf_pts[:, None, :] - positions[None, :, :]   # (M, N, 3)
    dist  = np.linalg.norm(diff, axis=2)                    # (M, N)
    dist  = np.where(dist < 1e-6, 1e-6, dist)              # avoid divide-by-zero

    # V in e/Å, then convert
    V = (charges[None, :] / dist).sum(axis=1) * _COULOMB_HARTREE_ANG
    return V   # (M,) kJ mol⁻¹


def compute_meps(npz_path: Path) -> dict:
    """
    Load a .npz charge file, compute the MEPS on the surface, and return a
    dict with all data needed for footprinting.

    Keys in returned dict:
      surf_pts, surf_atoms, V, E_max, E_min, AvdW, numbers, positions, charges
    """
    data      = np.load(npz_path)
    numbers   = data["numbers"]
    positions = data["positions"]
    charges   = data["charges"]

    surf_pts, surf_atoms, AvdW = shrake_rupley_surface(positions, numbers)

    if len(surf_pts) == 0:
        return {"error": "empty surface"}

    V = coulomb_potential(surf_pts, positions, charges)

    E_max = float(V.max())
    E_min = float(V.min())

    return {
        "numbers":   numbers,
        "positions": positions,
        "charges":   charges,
        "surf_pts":  surf_pts,
        "surf_atoms": surf_atoms,
        "V":         V,
        "E_max":     E_max,
        "E_min":     E_min,
        "AvdW":      AvdW,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate vdW surfaces and compute Coulomb MEPS from charge files."
    )
    parser.add_argument("--chargedir", default="charges",
                        help="Directory of .npz charge files")
    parser.add_argument("--outdir",    default="surfaces",
                        help="Output directory for surface .npz files")
    parser.add_argument("--resume",    action="store_true")
    args = parser.parse_args()

    charge_dir = Path(args.chargedir)
    out_dir    = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(charge_dir.glob("*.npz"))
    if not npz_files:
        print(f"No .npz files in {charge_dir}", file=sys.stderr)
        sys.exit(1)

    summary = []
    for npz in tqdm(npz_files, desc="Computing MEPS", unit="mol"):
        mol_id  = npz.stem
        out_npz = out_dir / f"{mol_id}.npz"

        if args.resume and out_npz.exists():
            continue

        result = compute_meps(npz)

        if "error" in result:
            summary.append({"mol_id": mol_id, "error": result["error"]})
            continue

        np.savez_compressed(
            out_npz,
            numbers=result["numbers"],
            positions=result["positions"],
            charges=result["charges"],
            surf_pts=result["surf_pts"],
            surf_atoms=result["surf_atoms"],
            V=result["V"],
            E_max=result["E_max"],
            E_min=result["E_min"],
            AvdW=result["AvdW"],
        )
        summary.append({
            "mol_id": mol_id,
            "E_max": result["E_max"],
            "E_min": result["E_min"],
            "AvdW":  result["AvdW"],
            "n_surf_pts": len(result["surf_pts"]),
        })

    import pandas as pd
    df = pd.DataFrame(summary)
    df.to_csv(out_dir / "meps_summary.csv", index=False)
    print(f"\nSaved MEPS for {len(df)} molecules to {out_dir}/")
    if "E_max" in df.columns:
        print(f"  E_max range: {df['E_max'].min():.1f} – {df['E_max'].max():.1f} kJ/mol")
        print(f"  E_min range: {df['E_min'].min():.1f} – {df['E_min'].max():.1f} kJ/mol")


if __name__ == "__main__":
    main()
