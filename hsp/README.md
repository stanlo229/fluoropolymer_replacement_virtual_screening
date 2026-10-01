# hsp/ — Hansen solubility parameters for the norbornene monomer library

Predicts δD, δP and δH for ~79k norbornene ester/amide monomers built from catalogue
alcohols and amines, and ranks them by Hansen distance (Ra) to PTFE, water,
diiodomethane and n-hexadecane.

## Methods (three versions, all kept for comparison)

| Version | HSP source | Where | Status |
|---|---|---|---|
| `legacy_sp` | Published Stefanis–Panayiotou 2012 group contribution (`hsp_calculator.py`) | `results/legacy_sp/` | Baseline. Goes negative for polycyclics (17% of monomers) and drifts with size. |
| `v2` | ExtraTrees model trained on HSPiP (code in `archive/v2/`) | `results/v2/` | Superseded. Deterministic but not a formula. Si not scored. |
| **`v3` (adopted)** | **Refitted group-contribution formula**, size-intensive form δk = √(Σ Nᵢ eₖ,ᵢ / Σ Nᵢ vᵢ), using the S-P groups plus added Si / new-atom groups, with values fitted to HSPiP (`analysis/gc_refit.py`, `gc_benchmark.py`, `apply_gc_formula.py`) | `results/v3/` | **Current method.** Most accurate in CV (median Ra error 2.83 on the curated set vs 3.34 for published S-P). |

> **Si monomers are not reliable.** v3 scores and ranks Si monomers using fitted Si groups,
> but those rest on 67 HSPiP Si compounds, only 7 of them curated, and all much smaller
> than the monomers. Every Si row carries an `hsp_note` and `expected_Ra_err = NaN`, and the
> ranking grids show a "Si: HSP unreliable" badge. Filter on `has_si` to remove them.
> `results/v3/si_monomers.csv` lists them all.

Accuracy, the literature comparison and every caveat are in
`results/benchmark/reference/REPORT.md` (gitignored, because it quotes HSPiP values) and
`docs/literature_ml_hsp.md`. The legacy method is described in `METHODS.md`.

## Layout

```
hsp/
  hsp_calculator.py, group_tables.py   S-P calculator + published tables (with the 2012 worked-example fixes)
  generate_monomers.py, run_pipeline.py  monomer library + legacy S-P pipeline
  rank_top_monomers.py                   constrained top-25 rankings (C1-C5 filters)
  solvent_incompatibility.py             probe-liquid HSP, Ra / RED helpers (imported)
  visualize_hsp.py                       Hansen-space plots
  analysis/                              reference building, CV benchmarks, v3 formula
  jobs/                                  SLURM scripts (submit from hsp/)
  archive/                               retired code, with reasons (archive/README.md)
  docs/                                  literature review
  reference/                             licensed HSPiP workbook + papers   (gitignored)
  results/legacy_sp/  v2/  v3/           outputs per method version
  results/benchmark/                     CV predictions, fitted group tables, report   (gitignored)
  results/manual/2026-09-30/             the six requested monomers (hsp6*.csv)
  logs/                                  SLURM logs
```

## Running

Everything heavier than a quick check runs as a SLURM job. Submit from `hsp/`:

```bash
cd hsp
sbatch jobs/submit_hsp.sh                                # legacy: library + S-P + rankings
sbatch jobs/submit_v3_formula.sh                         # v3: CV + fit formula, apply, rank
sbatch --export=ALL,FROM=apply jobs/submit_v3_formula.sh # v3 from a later stage (bench|apply|down)
sbatch jobs/submit_gc_benchmark.sh                       # formula vs XGBoost vs ExtraTrees CV only
python analysis/predict_smiles.py "SMILES" ...           # v3 HSP for any SMILES (seconds; fitted tables needed)
```

The cluster has only GPU partitions; these CPU jobs use `gpubase_l40s_b1` with no `--gres`.
`analysis/build_reference.py` must be run once after adding a new HSPiP workbook to `reference/`.

## Licensed data

`reference/` holds the HSPiP workbook and paywalled papers. `results/benchmark/` holds files that
contain HSPiP values. Both are gitignored. The `results/v2` and `v3` tables contain predictions
only. The v2 tables also name each monomer's nearest HSPiP compound, but carry none of its values.

Large per-monomer tables that the jobs regenerate are not tracked in git: all of v2's, and the
v3 duplicates (`monomers_hsp_corrected.csv`, `hsp_Ra_ranked.csv`, `monomers_hsp_solvent_Ra.csv`).
`results/v3/monomers_hsp_formula.csv` is the tracked full v3 table.
