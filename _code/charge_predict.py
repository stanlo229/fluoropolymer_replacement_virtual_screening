#!/usr/bin/env python3
"""
AIMNet2 MBIS atomic charge prediction.

Loads the AIMNet2 wb97m-d3 ensemble model and predicts partial charges
(MBIS-style) for each optimised geometry.  Charges are saved alongside
coordinates in a compressed NumPy archive (<mol_id>.npz).

The .npz contains:
  - 'numbers'  : (N,)   int   — atomic numbers
  - 'positions': (N, 3) float — Cartesian coordinates (Å)
  - 'charges'  : (N,)   float — MBIS partial charges (e)
  - 'mol_charge': scalar int  — total molecular charge (default 0)

AIMNet2 model weights are downloaded automatically on first use via
torch.hub (or from Hugging Face if using aimnet2calc).

Usage:
    python charge_predict.py --geomdir geoms/ --outdir charges/ [--device cuda]
    python charge_predict.py --geomdir geoms/ --outdir charges/ --mol_charge 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Periodic table: symbol → atomic number
_SYMBOL_TO_Z = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9,
    "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53,
}


def load_xyz(xyz_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Parse a minimal XYZ file → (atomic_numbers, positions).
    Returns None on failure.
    """
    lines = xyz_path.read_text().splitlines()
    if len(lines) < 3:
        return None
    try:
        n_atoms = int(lines[0].strip())
    except ValueError:
        return None

    symbols  = []
    positions = []
    for line in lines[2: 2 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            return None
        sym = parts[0].capitalize()
        z   = _SYMBOL_TO_Z.get(sym)
        if z is None:
            # Keep unknown elements as atomic number 0 — AIMNet2 will error
            z = 0
        symbols.append(z)
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return np.array(symbols, dtype=np.int32), np.array(positions, dtype=np.float64)


def load_aimnet2(device: str = "cpu"):
    """
    Load AIMNet2 wb97m-d3 ensemble model.

    Tries two routes:
      1. aimnet2calc package (pip install aimnet2calc) — preferred
      2. torch.hub fallback
    """
    import torch

    # Route 1: aimnet2calc
    try:
        import aimnet2calc
        model = aimnet2calc.AIMNet2Calculator("aimnet2_wb97m_0")
        model.to(device)
        return model, "aimnet2calc"
    except Exception:
        pass

    # Route 2: torch.hub
    try:
        model = torch.hub.load(
            "isayevlab/AIMNet2", "aimnet2_wb97m-d3_ens",
            source="github", trust_repo=True,
        )
        model = model.to(device)
        return model, "torchhub"
    except Exception as exc:
        raise RuntimeError(
            f"Could not load AIMNet2 model. "
            f"Install via: pip install aimnet2calc\n  ({exc})"
        ) from exc


def predict_charges_aimnet2calc(model, numbers: np.ndarray,
                                 positions: np.ndarray,
                                 mol_charge: int = 0) -> np.ndarray:
    """Call aimnet2calc model and extract MBIS charges."""
    import torch

    data = {
        "numbers":  torch.tensor(numbers,   dtype=torch.long).unsqueeze(0),
        "coord":    torch.tensor(positions, dtype=torch.float64).unsqueeze(0),
        "charge":   torch.tensor([mol_charge], dtype=torch.float64),
    }
    with torch.no_grad():
        result = model(data)

    # aimnet2calc returns charges under key 'charges' or 'mulliken_charges'
    for key in ("charges", "mulliken_charges", "q"):
        if key in result:
            return result[key].squeeze(0).cpu().numpy().astype(np.float64)

    raise KeyError(f"No charge key found in AIMNet2 output. Keys: {list(result)}")


def predict_charges_torchhub(model, numbers: np.ndarray,
                              positions: np.ndarray,
                              mol_charge: int = 0) -> np.ndarray:
    """Call torch.hub AIMNet2 model and extract charges."""
    import torch

    numbers_t  = torch.tensor(numbers,   dtype=torch.long).unsqueeze(0)
    positions_t = torch.tensor(positions, dtype=torch.float).unsqueeze(0)
    charge_t    = torch.tensor([[mol_charge]], dtype=torch.float)

    with torch.no_grad():
        out = model({"numbers": numbers_t, "coordinates": positions_t,
                     "charge": charge_t})

    for key in ("charges", "q", "atomic_charges"):
        if key in out:
            return out[key].squeeze(0).cpu().numpy().astype(np.float64)

    raise KeyError(f"No charge key found in AIMNet2 output. Keys: {list(out)}")


def predict_charges(model, route: str, numbers: np.ndarray,
                    positions: np.ndarray, mol_charge: int = 0) -> np.ndarray:
    if route == "aimnet2calc":
        return predict_charges_aimnet2calc(model, numbers, positions, mol_charge)
    return predict_charges_torchhub(model, numbers, positions, mol_charge)


def process_geom_dir(geom_dir: Path, out_dir: Path, device: str,
                     mol_charge: int, resume: bool):
    xyz_files = sorted(geom_dir.glob("*.xyz"))
    if not xyz_files:
        print(f"No .xyz files found in {geom_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(xyz_files)} .xyz files — loading AIMNet2 …")
    model, route = load_aimnet2(device)
    print(f"  Model loaded via {route} on {device}")

    errors = []
    for xyz in tqdm(xyz_files, desc="Predicting charges", unit="mol"):
        mol_id  = xyz.stem
        npz_out = out_dir / f"{mol_id}.npz"

        if resume and npz_out.exists():
            continue

        parsed = load_xyz(xyz)
        if parsed is None:
            errors.append((mol_id, "XYZ parse error"))
            continue

        numbers, positions = parsed

        try:
            charges = predict_charges(model, route, numbers, positions, mol_charge)
        except Exception as exc:
            errors.append((mol_id, str(exc)))
            continue

        np.savez(npz_out,
                 numbers=numbers,
                 positions=positions,
                 charges=charges,
                 mol_charge=np.array(mol_charge))

    print(f"\nDone. Errors: {len(errors)}")
    for mol_id, msg in errors[:20]:
        print(f"  {mol_id}: {msg}")
    if len(errors) > 20:
        print(f"  … and {len(errors) - 20} more")


def main():
    parser = argparse.ArgumentParser(
        description="Predict AIMNet2 MBIS charges for optimised geometries."
    )
    parser.add_argument("--geomdir",    default="geoms",
                        help="Directory of .xyz files (from geometry_opt.py)")
    parser.add_argument("--outdir",     default="charges",
                        help="Output directory for .npz charge files")
    parser.add_argument("--device",     default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--mol_charge", type=int, default=0,
                        help="Total molecular charge (default 0)")
    parser.add_argument("--resume",     action="store_true",
                        help="Skip molecules whose .npz already exists")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    process_geom_dir(
        Path(args.geomdir), out_dir, args.device, args.mol_charge, args.resume
    )


if __name__ == "__main__":
    main()
