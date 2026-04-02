# Plan: MLIP-based SSIP pipeline for norbornene virtual screening

## Context
Virtual screening of sidechain attachments to a norbornene ester/amide scaffold to find
non-fluorinated fluoropolymer replacements. For each candidate molecule we need the raw
SSIP interaction parameters (ei, α H-bond donor, β H-bond acceptor) as defined by the
Hunter group (Calero et al. *PCCP* 2013; equations 15/16 in Driver & Hunter *PCCP* 2020).

The traditional route (DFT B3LYP/6-31G* geometry + MEPS) is being replaced with a fully
ML pipeline. Library scale is <500 molecules. SSIP footprinting must be implemented from scratch.
The entire `_code/` directory is currently empty.

---

## Recommended stack

| Step | Tool | Rationale |
|---|---|---|
| 3D init | RDKit ETKDG | Standard, fast, no extra install |
| Geometry opt | **MACE-OFF23** | Trained on SPICE+QMugs; best organic-molecule geometry accuracy; no DFT needed |
| Charges / MEPS | **AIMNet2** | Predicts MBIS atomic charges at near-CCSD(T) accuracy; orders of magnitude faster than DFT |
| MEPS surface | Custom Python | Van der Waals surface grid + Coulomb V(r) from MBIS charges |
| SSIP footprint | Custom Python | Hunter group algorithm (d=3.2 Å, r=1.1 Å, ASSIP=9.35 Å²) |
| Calibration | Linear fit | Recalibrate E_max→α and E_min→β against ~20 experimental H-bond compounds |

**Why MACE-OFF23 over MACE-POLAR / UMA:**
- MACE-POLAR adds long-range polarization for energy/force accuracy but does not output MEPS
- UMA is a general foundation model, less specialized for organic electrostatics
- MACE-OFF23 is the most validated for drug-like / organic molecule geometries

**Critical caveat on charges-based MEPS:**
AIMNet2 MBIS charges give a Coulomb-only MEPS approximation. Hunter's eqs (15)/(16) were
parameterized against B3LYP/6-31G* DFT MEPS — the numerical scale of E_max/E_min will
differ. A calibration step against known experimental α/β compounds is **required** before
applying the equations (see Step 6 below).

**Fallback if AIMNet2 MEPS proves insufficiently correlated with DFT:**
Replace Step 4 with GFN2-xTB single-point (xtb package, ~1 s/molecule). This is
semiempirical QM, not ML, but is accepted as a fast surrogate for MEPS.

---

## Assumptions (confirm before implementation)

- Norbornene scaffold is 5-norbornene-2-carboxylic acid (or ester/amide at C2). Clarify
  exact regiochemistry and whether both **endo and exo** isomers are wanted.
- Sidechains come from the filtered `catalogues.csv` (monoalcohols → esters, monoamines → amides)
- Molecules are neutral, closed-shell singlets

---

## Files to create

```
_code/
├── requirements.txt
├── generate_library.py     # scaffold + sidechain → SMILES library
├── geometry_opt.py         # RDKit 3D init → MACE-OFF23 optimization
├── charge_predict.py       # AIMNet2 → MBIS atomic charges per molecule
├── meps_surface.py         # vdW surface sampling + Coulomb V(r) → E_max, E_min
├── calibrate_ssip.py       # fit linear correction: E_max→α, E_min→β vs. exp. data
├── ssip_footprint.py       # Hunter group footprinting algorithm → ei values
└── run_pipeline.py         # end-to-end orchestration with CSV output
```

---

## Step-by-step implementation plan

### Step 1 — `generate_library.py`
- Define norbornene ester scaffold SMILES with an `[*]` attachment point
- Loop over filtered `catalogues.csv`:
  - monoalcohol R-OH → ester: replace `[*]OH` with `OC(=O){scaffold_ester_fragment}`
  - monoamine R-NH2 → amide: replace `[*]NH2` with `NC(=O){scaffold_amide_fragment}`
- Use RDKit `AllChem.CombineMols` / reaction SMARTS to form product SMILES
- Write output: `library.csv` (smiles, scaffold_type, sidechain_id, compound_type)

### Step 2 — `geometry_opt.py`
```python
from mace.calculators import mace_off
from ase.optimize import LBFGS

calc = mace_off(model="medium", device="cuda")  # or "cpu"
# RDKit ETKDG → ASE Atoms → attach MACE-OFF23 calculator → LBFGS(fmax=0.01)
```
- Handle both endo/exo by generating both RDKit initial structures
- Save optimized geometries as `.xyz` files

### Step 3 — `charge_predict.py`
```python
import aimnet2calc  # pip install aimnet2

model = torch.jit.load("aimnet2_wb97m-d3_ens.jpt")  # best model for organic
# Input: optimized xyz → output: MBIS atomic charges per atom
```
- Save charges alongside coordinates in a `.npz` per molecule
- Package: `pip install aimnet2calc` or load from Hugging Face

### Step 4 — `meps_surface.py`
- Generate van der Waals surface (Shrake-Rupley algorithm, density ~1 point/Å²,
  scaled at 1.4× vdW radii to approximate the 0.002 Bohr/Å³ isosurface)
- Compute Coulomb potential at each surface point:
  `V(r) = sum_i q_i / |r - r_i|` (in Hartree-equivalent units)
- Record E_max (most positive V) and E_min (most negative V)
- Assign each surface point to its nearest atom (for functional-group-dependent c
  scaling factor in eq 16)

### Step 5 — `calibrate_ssip.py`  *(run once; reuse coefficients)*
- Use ~20–30 compounds with known experimental α/β from Hunter's ESI datasets
  (e.g. methanol, acetone, diethyl ether, pyridine, acetonitrile, chloroform…)
- Run Steps 2–4 on these compounds
- Fit linear recalibration:  `α = m_α · E_max + b_α`  and  `β = c · (m_β · E_min² + b_β · E_min)`
  matching Hunter's quadratic forms (eqs 15/16) but re-scaled to AIMNet2 MEPS units
- Save calibration coefficients as `calibration.json`

### Step 6 — `ssip_footprint.py`
Implement Hunter group algorithm from Calero et al. 2013 (Section "Conversion of the
calculated MEPS to a set of SSIPs"):
1. Compute N = AvdW / ASSIP (ASSIP = 9.35 Å², N rounded to nearest int)
2. Determine fraction of surface with negative potential → number of negative SSIPs
3. Greedy algorithm: rank positive surface points by α (descending), place SSIPs with
   minimum separation d=3.2 Å; repeat for negative sites
4. Within each placed SSIP: take maximum/minimum V within r=1.1 Å of the centre point
5. Apply calibrated eqs to convert V → ei (= α for positive, = −β for negative sites)
6. Return list of SSIPs with (x, y, z, ei, atom_assignment, functional_group_type)

### Step 7 — `run_pipeline.py`
- Read `library.csv` → batch through Steps 2–6
- Output `ssip_results.csv`: columns = smiles, mol_id, SSIP_count, alpha_max,
  beta_max, all_ei_values (JSON list), AvdW

---

## Dependencies (`requirements.txt`)

```
mace-torch>=0.3.6       # MACE-OFF23 geometry optimization
aimnet2calc             # AIMNet2 charge prediction
ase>=3.23               # Atoms/optimizer interface
rdkit>=2024.03          # 3D init, SMARTS, vdW radii
numpy
scipy                   # Shrake-Rupley surface, calibration fitting
pandas
tqdm
```

---

## Verification plan

1. **Geometry check**: RMSD of MACE-OFF23 optimized structures vs. DFT for 5 known
   norbornene esters — expect < 0.1 Å heavy-atom RMSD
2. **Charge/MEPS check**: Plot AIMNet2 E_max vs B3LYP/6-31G* E_max for 20 test
   compounds — expect R² > 0.85 after calibration
3. **SSIP sanity check**: Run footprinting on methanol (expect 1 positive SSIP α≈2.1,
   2 negative SSIPs β≈2.2 each) and acetone (expect 0 donor SSIPs, 2 acceptor SSIPs
   β≈3.4 each) to match Table 1 values from Calero et al.
4. **End-to-end**: Run on 10 norbornene variants, inspect ei distributions and
   ensure endo/exo isomers give distinguishable SSIP profiles

---

## Open questions before coding

- Exact norbornene scaffold SMILES and attachment regiochemistry (C2-endo? C2-exo?)
- Whether both endo and exo isomers should be screened or just one
- Which sidechain compound types to include (monoalcohol only → esters, monoamine only → amides, or both?)
