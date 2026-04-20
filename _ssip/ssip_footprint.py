#!/usr/bin/env python3
"""
SSIP footprinting algorithm (Hunter group, Calero et al. PCCP 2013).

Algorithm (Section "Conversion of the calculated MEPS to a set of SSIPs"):
  1. N = round(AvdW / ASSIP)  where ASSIP = 9.35 Å²
  2. n_neg = round(N × fraction_negative_surface)  →  n_pos = N − n_neg
  3. Greedy SSIP placement (separately for positive and negative sites):
       a. Rank surface points by |α| or |β| descending.
       b. For each starting point, greedily select the next point ≥ d=3.2 Å
          from all already-placed SSIPs.
       c. Repeat for every starting point; keep the set that maximises Σ|ei|.
  4. Within each placed SSIP footprint: take the most extreme V within r=1.1 Å
     of the centre point.
  5. Convert to ei via calibrated α/β equations.
  6. Assign functional-group type from nearest atom → look up c factor.

Each SSIP is returned as a dict with keys:
  x, y, z, ei, sign (+1 donor / -1 acceptor), alpha, beta,
  atom_idx, functional_group, c_factor

Usage:
    python ssip_footprint.py --surfdir surfaces/ --calibration calibration.json \
                             --outdir ssips/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants (Calero et al. 2013)
# ---------------------------------------------------------------------------

ASSIP   = 9.35    # Å²  — footprint area per SSIP (defined by water × 4 SSIPs)
D_MIN   = 3.2     # Å   — minimum separation between SSIPs
R_MAX   = 1.1     # Å   — radius within which to find extreme V for SSIP value

# ---------------------------------------------------------------------------
# Functional group assignment from RDKit
# (used to look up c-factor in calibration["c_factors"])
# ---------------------------------------------------------------------------

from rdkit import Chem

_FG_PATTERNS: list[tuple[str, Chem.Mol]] = [
    ("nitrile_N",        Chem.MolFromSmarts("[NX1]#[CX2]")),
    ("primary_N",        Chem.MolFromSmarts("[NX3H2;!$(NC=O)]")),
    ("secondary_N",      Chem.MolFromSmarts("[NX3H1;!$(NC=O);!$([NH]c)]")),
    ("tertiary_N",       Chem.MolFromSmarts("[NX3H0;!$(NC=O)]")),
    ("pyridine_N",       Chem.MolFromSmarts("[nX2]")),
    # alcohol_O must come before ether_O — both match ROH, first match wins
    ("alcohol_O",        Chem.MolFromSmarts("[OX2H;!$(OC=O)]")),
    ("ether_O",          Chem.MolFromSmarts("[OX2H0;!$(O=*);!$(OC=O);!$(Oc)]")),
    ("aldehyde_C=O",     Chem.MolFromSmarts("[CX3H1](=O)")),
    # carboxylic_C=O must come before ester_C=O (COOH is more specific)
    ("carboxylic_C=O",   Chem.MolFromSmarts("[CX3H0](=O)[OX2H1]")),
    ("ester_C=O",        Chem.MolFromSmarts("[CX3](=O)[OX2H0]")),
    ("nitro_O",          Chem.MolFromSmarts("[OX1][NX3+]")),
    ("sulfoxide_O",      Chem.MolFromSmarts("[SX3](=O)")),
]


# ---------------------------------------------------------------------------
# Atom-type / molecular filters for SSIP placement
# ---------------------------------------------------------------------------

# Elements that can bear lone-pair H-bond acceptors
_ACCEPTOR_ELEMENTS = frozenset([7, 8, 16])   # N, O, S

# Molecular H-bond donor patterns (used to decide whether positive SSIPs
# should be placed at all — if no HBD group exists, α = 0 by definition)
_HBD_SMARTS = [
    Chem.MolFromSmarts("[OX2H]"),                        # alcohol, phenol, COOH
    Chem.MolFromSmarts("[NX3H]"),                        # amine/amide N-H
    Chem.MolFromSmarts("[NX4H]"),                        # ammonium
    Chem.MolFromSmarts("[SX2H]"),                        # thiol
    Chem.MolFromSmarts("[CX4H;$([CH]([F,Cl,Br,I])[F,Cl,Br,I])]"),  # haloform
]
_HBD_SMARTS = [p for p in _HBD_SMARTS if p is not None]


def _molecule_has_hbd(smiles: str | None) -> bool:
    """Return True if the molecule has at least one genuine H-bond donor."""
    if smiles is None:
        return True   # unknown → conservative (don't suppress)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return True
    mol_h = Chem.AddHs(mol)
    return any(mol_h.HasSubstructMatch(p) for p in _HBD_SMARTS)


def _acceptor_atom_mask(surf_atoms: np.ndarray,
                        numbers: np.ndarray) -> np.ndarray:
    """
    Boolean mask over surface points: True if the nearest atom is a genuine
    lone-pair donor (N, O, S).  Suppresses C and F surface regions so that
    alkanes, perfluoroalkanes, etc. get β = 0.
    """
    return np.isin(numbers[surf_atoms], list(_ACCEPTOR_ELEMENTS))


def _atom_fg_map(rdmol: Chem.Mol) -> dict[int, str]:
    """Return {atom_idx: functional_group_name} for all heavy atoms."""
    fg_map = {}
    for fg_name, smarts in _FG_PATTERNS:
        if smarts is None:
            continue
        for match in rdmol.GetSubstructMatches(smarts):
            for idx in match:
                if idx not in fg_map:   # first match wins
                    fg_map[idx] = fg_name
    return fg_map


# ---------------------------------------------------------------------------
# Core footprinting
# ---------------------------------------------------------------------------

def _greedy_ssip_placement(
    pts:    np.ndarray,   # (M_subset, 3) candidate surface points
    values: np.ndarray,   # (M_subset,)   |α| or |β| for each point
    n:      int,          # target number of SSIPs to place
    d_min:  float = D_MIN,
) -> np.ndarray:
    """
    Greedy SSIP placement.  Returns indices (into pts) of selected SSIPs.

    Implements step 3c: try every point as the seed, keep the set that
    maximises Σ|ei|.  For speed, we only try the top-20 seeds (the full
    exhaustive search is O(M²) and adds little value in practice).
    """
    M = len(pts)
    if M == 0 or n == 0:
        return np.array([], dtype=int)

    # Sort by value descending — used both for seeding and for ranking
    order = np.argsort(values)[::-1]

    best_set   = np.array([], dtype=int)
    best_score = -np.inf

    n_seeds = min(20, M)
    for seed_rank in range(n_seeds):
        seed_idx = order[seed_rank]
        selected = [seed_idx]
        # Greedily add highest-value points ≥ d_min from all placed SSIPs
        for idx in order:
            if len(selected) >= n:
                break
            if idx in selected:
                continue
            dists = np.linalg.norm(pts[selected] - pts[idx], axis=1)
            if dists.min() >= d_min:
                selected.append(idx)

        score = values[selected].sum()
        if score > best_score:
            best_score = score
            best_set   = np.array(selected)

    return best_set


def _ssip_value_at_footprint(
    centre_pt: np.ndarray,   # (3,) Å
    surf_pts:  np.ndarray,   # (M, 3)
    V:         np.ndarray,   # (M,) kJ/mol
    sign:      int,          # +1 for positive SSIP, -1 for negative
    r_max:     float = R_MAX,
) -> tuple[float, int]:
    """
    Find the most extreme V within r_max of centre_pt.
    Returns (V_extreme, index_in_surf_pts).
    """
    dists   = np.linalg.norm(surf_pts - centre_pt, axis=1)
    mask    = dists <= r_max
    if not mask.any():
        # Fall back to the point itself
        return float(V[np.argmin(dists)]), int(np.argmin(dists))

    V_sub = V[mask]
    if sign > 0:
        local_idx = np.argmax(V_sub)
    else:
        local_idx = np.argmin(V_sub)

    global_idx = np.where(mask)[0][local_idx]
    return float(V[global_idx]), int(global_idx)


def footprint(
    surface_data: dict,
    calibration:  dict,
    smiles:       str | None = None,
) -> list[dict]:
    """
    Run the full Hunter group footprinting algorithm on one molecule.

    Parameters
    ----------
    surface_data : output of meps_surface.compute_meps (or loaded from .npz)
    calibration  : dict from calibration.json
    smiles       : optional SMILES for functional-group assignment

    Returns
    -------
    List of SSIP dicts, each with: x, y, z, ei, sign, alpha, beta,
    atom_idx, functional_group, c_factor.
    """
    surf_pts   = surface_data["surf_pts"]    # (M, 3)
    surf_atoms = surface_data["surf_atoms"]  # (M,)   nearest atom index
    V          = surface_data["V"]           # (M,)   kJ/mol
    AvdW       = float(surface_data["AvdW"])
    numbers    = surface_data["numbers"]     # (n_atoms,) atomic numbers

    M = len(surf_pts)
    if M == 0:
        return []

    # --- Molecular HBD check: if no donor groups exist, skip positive SSIPs -
    has_hbd = _molecule_has_hbd(smiles)

    # --- Acceptor atom mask: restrict negative SSIPs to N/O/S surface regions
    # Suppresses false acceptor signal from C-H and C-F bonds (alkanes, PFAS)
    acceptor_mask = _acceptor_atom_mask(surf_atoms, numbers)

    # --- Step 1: total number of SSIPs -------------------------------------
    N = max(1, round(AvdW / ASSIP))

    # --- Step 2: split into positive / negative ----------------------------
    if not has_hbd:
        # No H-bond donor groups → force all SSIPs to acceptor side
        frac_neg = (V < 0).sum() / M if M > 0 else 0.0
        n_neg    = N
        n_pos    = 0
    else:
        frac_neg = (V < 0).sum() / M
        n_neg    = round(N * frac_neg)
        n_pos    = N - n_neg

    # --- Functional-group map (if SMILES given) ----------------------------
    fg_map: dict[int, str] = {}
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Map heavy-atom indices (no Hs in SMILES → re-index after AddHs)
            mol_h = Chem.AddHs(mol)
            fg_map = _atom_fg_map(mol_h)

    c_factors = calibration.get("c_factors", {})

    # --- Helper: convert V at a surface point → (alpha, beta, ei) ----------
    a0, a1 = calibration["a0"], calibration["a1"]
    b0, b1 = calibration["b0"], calibration["b1"]

    def _to_alpha(v):
        return max(0.0, a0 * v ** 2 + a1 * v)

    def _to_beta(v, c=1.0):
        return max(0.0, c * (b0 * v ** 2 + b1 * v))

    ssips = []

    # --- Step 3/4/5: positive SSIPs (H-bond donors) ------------------------
    if n_pos > 0:
        pos_mask   = V >= 0
        pos_pts    = surf_pts[pos_mask]
        pos_V      = V[pos_mask]
        pos_atoms  = surf_atoms[pos_mask]

        # Convert V to α for ranking
        pos_alpha  = np.array([_to_alpha(v) for v in pos_V])
        placed_idx = _greedy_ssip_placement(pos_pts, pos_alpha, n_pos)

        for idx in placed_idx:
            centre    = pos_pts[idx]
            v_extreme, surf_idx = _ssip_value_at_footprint(
                centre, pos_pts, pos_V, sign=+1)
            alpha = _to_alpha(v_extreme)
            ei    = alpha

            atom_idx = int(pos_atoms[idx])
            fg       = fg_map.get(atom_idx, "default")
            c_val    = c_factors.get(fg, c_factors.get("default", 1.0))

            ssips.append({
                "x": float(centre[0]), "y": float(centre[1]), "z": float(centre[2]),
                "ei": ei, "sign": +1,
                "alpha": alpha, "beta": 0.0,
                "atom_idx": atom_idx,
                "functional_group": fg,
                "c_factor": c_val,
            })

    # --- Step 3/4/5: negative SSIPs (H-bond acceptors) ---------------------
    if n_neg > 0:
        # Restrict to N/O/S surface regions — suppresses alkane/perfluoroalkane
        # false acceptor signal from C-H and C-F polarisation
        neg_mask   = acceptor_mask & (V < 0)
        neg_pts    = surf_pts[neg_mask]
        neg_V      = V[neg_mask]
        neg_atoms  = surf_atoms[neg_mask]

        neg_beta   = np.array([_to_beta(-v) for v in -neg_V])  # use |E_min| for ranking
        placed_idx = _greedy_ssip_placement(neg_pts, neg_beta, n_neg)

        for idx in placed_idx:
            centre    = neg_pts[idx]
            v_extreme, _ = _ssip_value_at_footprint(
                centre, neg_pts, neg_V, sign=-1)

            atom_idx = int(neg_atoms[idx])
            fg       = fg_map.get(atom_idx, "default")
            c_val    = c_factors.get(fg, c_factors.get("default", 1.0))

            beta = _to_beta(v_extreme, c=c_val)
            ei   = -beta    # negative by convention for acceptor sites

            ssips.append({
                "x": float(centre[0]), "y": float(centre[1]), "z": float(centre[2]),
                "ei": ei, "sign": -1,
                "alpha": 0.0, "beta": beta,
                "atom_idx": atom_idx,
                "functional_group": fg,
                "c_factor": c_val,
            })

    return ssips


def ssip_summary(ssips: list[dict], AvdW: float) -> dict:
    """Aggregate SSIP list into scalar descriptors for the output CSV."""
    if not ssips:
        return {"N_ssip": 0, "alpha_max": 0.0, "beta_max": 0.0,
                "ei_values": "[]", "AvdW": AvdW}

    alphas = [s["alpha"] for s in ssips]
    betas  = [s["beta"]  for s in ssips]
    eis    = [s["ei"]    for s in ssips]

    return {
        "N_ssip":    len(ssips),
        "alpha_max": round(max(alphas), 4),
        "beta_max":  round(max(betas),  4),
        "ei_values": json.dumps([round(e, 4) for e in eis]),
        "AvdW":      round(AvdW, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Hunter group SSIP footprinting on surface files."
    )
    parser.add_argument("--surfdir",     default="surfaces",
                        help="Directory with surface .npz files")
    parser.add_argument("--calibration", default="calibration.json")
    parser.add_argument("--library",     default="library.csv",
                        help="Library CSV (for SMILES → functional-group assignment)")
    parser.add_argument("--outdir",      default="ssips",
                        help="Output directory for per-molecule SSIP JSON files")
    parser.add_argument("--output_csv",  default="ssip_results.csv")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calib   = json.loads(Path(args.calibration).read_text())
    surf_dir = Path(args.surfdir)

    # Load SMILES map if library CSV available
    smiles_map: dict[str, str] = {}
    lib_path = Path(args.library)
    if lib_path.exists():
        lib = pd.read_csv(lib_path)
        smiles_map = dict(zip(lib["mol_id"].astype(str), lib["smiles"].astype(str)))

    npz_files = sorted(surf_dir.glob("*.npz"))
    if not npz_files:
        print(f"No .npz files in {surf_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for npz in tqdm(npz_files, desc="Footprinting", unit="mol"):
        mol_id = npz.stem
        data   = dict(np.load(npz, allow_pickle=False))
        smiles = smiles_map.get(mol_id)

        ssips  = footprint(data, calib, smiles)
        summary = ssip_summary(ssips, float(data.get("AvdW", 0.0)))
        summary["mol_id"] = mol_id

        # Save per-molecule SSIP detail
        json_path = out_dir / f"{mol_id}.json"
        json_path.write_text(json.dumps({"mol_id": mol_id, "ssips": ssips}, indent=2))

        rows.append(summary)

    df = pd.DataFrame(rows)[["mol_id", "N_ssip", "alpha_max", "beta_max",
                              "ei_values", "AvdW"]]
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved SSIP results for {len(df)} molecules → {args.output_csv}")
    print(f"  α_max range: {df['alpha_max'].min():.2f} – {df['alpha_max'].max():.2f}")
    print(f"  β_max range: {df['beta_max'].min():.2f} – {df['beta_max'].max():.2f}")


if __name__ == "__main__":
    main()
